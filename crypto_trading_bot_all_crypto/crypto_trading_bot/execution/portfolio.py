"""Portfolio and position tracker for paper trading.

Syncs with Alpaca paper account when possible, falls back to local tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from execution.risk import OpenPosition, RiskManager
from monitoring.logger import get_logger

log = get_logger("portfolio")


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry: float
    current_price: float = 0.0
    stop_price: float | None = None
    unrealized_pnl: float = 0.0
    side: str = "long"

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.qty * self.avg_entry


@dataclass
class PortfolioSnapshot:
    equity: float
    cash: float
    buying_power: float
    positions: list[Position] = field(default_factory=list)
    realized_pnl_today: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Portfolio:
    def __init__(self, risk_manager: RiskManager) -> None:
        self.risk = risk_manager
        self.positions: dict[str, Position] = {}
        self.cash: float = 0.0
        self.equity: float = 0.0
        self.buying_power: float = 0.0
        self.realized_pnl_today: float = 0.0

    def update_from_alpaca(self, account: Any, positions: list[Any]) -> PortfolioSnapshot:
        """Sync state from Alpaca paper TradingClient responses."""
        self.cash = float(account.cash)
        self.equity = float(account.equity)
        self.buying_power = float(getattr(account, "buying_power", self.cash) or self.cash)

        self.positions.clear()
        for p in positions:
            symbol = p.symbol
            qty = float(p.qty)
            if qty == 0:
                continue
            avg = float(p.avg_entry_price)
            current = float(getattr(p, "current_price", avg) or avg)
            pos = Position(
                symbol=symbol,
                qty=qty,
                avg_entry=avg,
                current_price=current,
                unrealized_pnl=float(getattr(p, "unrealized_pl", 0.0) or 0.0),
            )
            self.positions[symbol] = pos

        self.risk.update_equity(self.equity)
        # Rebuild risk open positions from live data (simplified)
        self.risk.state.open_positions = [
            OpenPosition(
                symbol=pos.symbol,
                qty=pos.qty,
                entry_price=pos.avg_entry,
                stop_price=pos.stop_price or (pos.avg_entry * 0.95),  # fallback
                risk_usd=abs(pos.avg_entry - (pos.stop_price or pos.avg_entry * 0.95)) * pos.qty,
                opened_at=datetime.now(timezone.utc),
            )
            for pos in self.positions.values()
        ]

        return self.snapshot()

    def snapshot(self) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            equity=self.equity,
            cash=self.cash,
            buying_power=self.buying_power,
            positions=list(self.positions.values()),
            realized_pnl_today=self.realized_pnl_today,
        )

    def apply_local_fill(self, symbol: str, qty: float, price: float, *,
                         is_entry: bool, stop: float | None = None,
                         realized_pnl: float = 0.0) -> None:
        """Update local state after a paper fill (used when we don't immediately re-sync)."""
        if is_entry:
            self.positions[symbol] = Position(
                symbol=symbol, qty=qty, avg_entry=price,
                current_price=price, stop_price=stop)
            self.cash -= qty * price
        else:
            pos = self.positions.pop(symbol, None)
            if pos:
                self.cash += qty * price
                self.realized_pnl_today += realized_pnl
        self.equity = self.cash + sum(p.market_value for p in self.positions.values())
        self.risk.register_fill(symbol, qty, price, stop or 0.0, is_entry, realized_pnl)
        log.info("local fill: %s %s qty=%.6f @ %.4f  equity=%.2f",
                 "BUY" if is_entry else "SELL", symbol, qty, price, self.equity)
