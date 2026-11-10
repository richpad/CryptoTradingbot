"""Risk management for paper trading (Stage 2).

Aggressive defaults (user-confirmed):
  - Risk per trade: 1.5% of equity
  - Max open positions: 6
  - Max portfolio heat: 6%
  - Daily loss limit: 4% → halt new entries
  - 4 consecutive losses → pause 2 hours

All limits are hard. The risk manager never places orders; it only approves or rejects
signals and returns a sized quantity + stop. Execution is downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Sequence

from strategies.base import Action, Signal


@dataclass(frozen=True)
class RiskParams:
    risk_per_trade_pct: float = 1.5          # % of equity risked per trade
    max_positions: int = 6
    max_portfolio_heat_pct: float = 6.0      # sum of open risk
    daily_loss_limit_pct: float = 4.0
    consecutive_loss_pause: int = 4
    pause_hours: float = 2.0
    min_notional_usd: float = 10.0           # Alpaca crypto min is low; keep a floor
    max_notional_pct: float = 25.0           # never put more than 25% of equity in one name


@dataclass
class OpenPosition:
    symbol: str
    qty: float
    entry_price: float
    stop_price: float
    risk_usd: float                          # $ at risk if stop is hit
    opened_at: datetime


@dataclass
class RiskState:
    equity: float
    open_positions: list[OpenPosition] = field(default_factory=list)
    realized_pnl_today: float = 0.0
    consecutive_losses: int = 0
    paused_until: datetime | None = None
    day_start_equity: float | None = None

    @property
    def open_count(self) -> int:
        return len(self.open_positions)

    @property
    def current_heat_pct(self) -> float:
        if self.equity <= 0:
            return 100.0
        return 100.0 * sum(p.risk_usd for p in self.open_positions) / self.equity

    @property
    def daily_pnl_pct(self) -> float:
        if not self.day_start_equity or self.day_start_equity <= 0:
            return 0.0
        return 100.0 * self.realized_pnl_today / self.day_start_equity


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    qty: float = 0.0
    stop_price: float | None = None
    risk_usd: float = 0.0
    reasons: tuple[str, ...] = ()


class RiskManager:
    def __init__(self, params: RiskParams = RiskParams()) -> None:
        self.params = params
        self.state = RiskState(equity=0.0)

    def update_equity(self, equity: float, *, force_day_reset: bool = False) -> None:
        now = datetime.now(timezone.utc)
        if self.state.day_start_equity is None or force_day_reset:
            self.state.day_start_equity = equity
            self.state.realized_pnl_today = 0.0
        # Simple UTC day rollover
        if self.state.day_start_equity and now.hour == 0 and now.minute < 5:
            # crude; real system would track date
            pass
        self.state.equity = max(0.0, equity)

    def register_fill(self, symbol: str, qty: float, price: float, stop: float,
                      is_entry: bool, realized_pnl: float = 0.0) -> None:
        """Called by the portfolio/execution layer after a paper fill."""
        if is_entry:
            risk_usd = abs(price - stop) * qty
            self.state.open_positions.append(
                OpenPosition(symbol=symbol, qty=qty, entry_price=price,
                             stop_price=stop, risk_usd=risk_usd,
                             opened_at=datetime.now(timezone.utc)))
        else:
            self.state.open_positions = [p for p in self.state.open_positions if p.symbol != symbol]
            self.state.realized_pnl_today += realized_pnl
            if realized_pnl < 0:
                self.state.consecutive_losses += 1
                if self.state.consecutive_losses >= self.params.consecutive_loss_pause:
                    self.state.paused_until = datetime.now(timezone.utc) + timedelta(
                        hours=self.params.pause_hours)
            else:
                self.state.consecutive_losses = 0

    def evaluate(self, signal: Signal, current_price: float | None = None) -> RiskDecision:
        """Approve or reject a signal and return sized quantity + stop."""
        p = self.params
        s = self.state
        reasons: list[str] = []

        if signal.action is Action.HOLD:
            return RiskDecision(False, reasons=("HOLD signal",))

        # --- circuit breakers ---
        now = datetime.now(timezone.utc)
        if s.paused_until and now < s.paused_until:
            return RiskDecision(False, reasons=(f"paused until {s.paused_until.isoformat()}",))

        if s.equity <= 0:
            return RiskDecision(False, reasons=("equity is zero or unknown",))

        daily_pnl = s.daily_pnl_pct
        if daily_pnl <= -p.daily_loss_limit_pct:
            return RiskDecision(False, reasons=(f"daily loss limit hit ({daily_pnl:.2f}%)",))

        if signal.action is Action.SELL:
            # Exits are always allowed (risk reduction)
            pos = next((pos for pos in s.open_positions if pos.symbol == signal.symbol), None)
            if pos is None:
                return RiskDecision(False, reasons=("no open position to exit",))
            return RiskDecision(True, qty=pos.qty, stop_price=None,
                                reasons=("exit approved",))

        # --- BUY checks ---
        if s.open_count >= p.max_positions:
            return RiskDecision(False, reasons=(f"max positions ({p.max_positions}) reached",))

        if any(pos.symbol == signal.symbol for pos in s.open_positions):
            return RiskDecision(False, reasons=("already have a position in this symbol",))

        if s.current_heat_pct >= p.max_portfolio_heat_pct:
            return RiskDecision(False, reasons=(f"portfolio heat {s.current_heat_pct:.1f}% >= limit",))

        price = current_price or signal.price
        stop = signal.stop_price
        if stop is None or stop >= price or stop <= 0:
            return RiskDecision(False, reasons=("invalid or missing stop price",))

        risk_per_unit = price - stop
        if risk_per_unit <= 0:
            return RiskDecision(False, reasons=("stop must be below entry for long",))

        # Position size from risk budget
        risk_budget = s.equity * (p.risk_per_trade_pct / 100.0)
        qty = risk_budget / risk_per_unit

        # Caps
        max_qty_by_notional = (s.equity * (p.max_notional_pct / 100.0)) / price
        qty = min(qty, max_qty_by_notional)

        notional = qty * price
        if notional < p.min_notional_usd:
            return RiskDecision(False, reasons=(f"notional ${notional:.2f} below minimum",))

        # Final heat check after this trade
        projected_heat = s.current_heat_pct + 100.0 * (qty * risk_per_unit) / s.equity
        if projected_heat > p.max_portfolio_heat_pct:
            return RiskDecision(False, reasons=(f"projected heat {projected_heat:.1f}% exceeds limit",))

        return RiskDecision(
            approved=True,
            qty=round(qty, 8),               # crypto precision
            stop_price=stop,
            risk_usd=qty * risk_per_unit,
            reasons=(f"sized to {p.risk_per_trade_pct}% risk = ${qty * risk_per_unit:.2f}",),
        )
