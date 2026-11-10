"""Paper trading engine (Stage 2).

Loop:
  1. Refresh portfolio from Alpaca paper
  2. For each configured pair/timeframe, compute indicators + Strategy A signal
  3. Pass signal through RiskManager
  4. If approved, submit paper order
  5. Log everything

Safety:
  - Requires PAPER_TRADING_ENABLED=true in .env
  - Still refuses LIVE_TRADING_ENABLED
  - Only uses Alpaca paper client
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from config import Config, load_config
from database.database_manager import DatabaseManager
from execution.orders import PaperOrderExecutor, OrderResult
from execution.portfolio import Portfolio
from execution.risk import RiskManager, RiskParams
from indicators.scoring import score_frame
from indicators.technical_indicators import IndicatorParams, compute_indicators
from monitoring.logger import get_logger
from strategies.base import Action
from strategies.trend_following import TrendFollowing

log = get_logger("engine")


class PaperTradingEngine:
    def __init__(self, config: Config, db: DatabaseManager,
                 executor: PaperOrderExecutor,
                 risk_params: RiskParams | None = None) -> None:
        self.config = config
        self.db = db
        self.executor = executor
        self.risk = RiskManager(risk_params or RiskParams())
        self.portfolio = Portfolio(self.risk)
        self.strategy = TrendFollowing()
        self._stop = asyncio.Event()
        self._cycle = 0

    async def refresh_portfolio(self) -> None:
        account = await self.executor.get_account()
        positions = await self.executor.get_positions()
        snap = self.portfolio.update_from_alpaca(account, positions)
        log.info("portfolio: equity=$%.2f  cash=$%.2f  positions=%d  heat=%.1f%%  daily_pnl=%.2f%%",
                 snap.equity, snap.cash, len(snap.positions),
                 self.risk.state.current_heat_pct, self.risk.state.daily_pnl_pct)

    async def evaluate_symbol(self, symbol: str, timeframe: str = "1h") -> None:
        need = IndicatorParams().warmup_rows
        df = self.db.get_candles(self.config.exchange_id, symbol, timeframe, limit=need + 50)
        if len(df) < need:
            log.debug("%s %s: insufficient history (%d < %d)", symbol, timeframe, len(df), need)
            return

        ind = compute_indicators(df)
        score = score_frame(ind)["score"]
        signal = self.strategy.evaluate(ind, symbol, timeframe, score)

        log.info("%s %s → %s  score=%+.2f  reasons=%s",
                 symbol, timeframe, signal.action.value,
                 signal.score or 0.0, "; ".join(signal.reasons[:2]))

        decision = self.risk.evaluate(signal)
        if not decision.approved:
            log.info("%s rejected by risk: %s", symbol, "; ".join(decision.reasons))
            return

        if signal.action is Action.BUY:
            result = await self.executor.submit_market_buy(symbol, decision.qty)
            if result.success:
                # Optimistic local update; next cycle will re-sync from Alpaca
                self.portfolio.apply_local_fill(
                    symbol, decision.qty, signal.price,
                    is_entry=True, stop=decision.stop_price)
                log.info("PAPER ENTRY %s qty=%.6f stop=%.4f risk=$%.2f",
                         symbol, decision.qty, decision.stop_price or 0, decision.risk_usd)
            else:
                log.error("Order failed: %s", result.message)

        elif signal.action is Action.SELL:
            result = await self.executor.submit_market_sell(symbol, decision.qty)
            if result.success:
                # Rough realized PnL estimate; re-sync will correct
                pos = self.portfolio.positions.get(symbol)
                realized = 0.0
                if pos:
                    realized = (signal.price - pos.avg_entry) * decision.qty
                self.portfolio.apply_local_fill(
                    symbol, decision.qty, signal.price,
                    is_entry=False, realized_pnl=realized)
                log.info("PAPER EXIT %s qty=%.6f", symbol, decision.qty)
            else:
                log.error("Exit order failed: %s", result.message)

    async def run_once(self) -> None:
        self._cycle += 1
        log.info("=== paper cycle %d ===", self._cycle)
        try:
            await self.refresh_portfolio()
        except Exception as exc:
            log.error("portfolio refresh failed: %s", exc)
            return

        # Focus on 1h for decisions (can be extended)
        primary_tf = "1h"
        for symbol in self.config.pairs:
            try:
                await self.evaluate_symbol(symbol, primary_tf)
            except Exception as exc:
                log.error("%s evaluation failed: %s", symbol, exc)
            await asyncio.sleep(0.3)  # light pacing

    async def run_forever(self, interval_seconds: float = 60.0) -> None:
        log.info("Paper trading engine started (Ctrl+C to stop). "
                 "Risk: %.1f%% per trade, max %d positions, %.1f%% daily loss limit",
                 self.risk.params.risk_per_trade_pct,
                 self.risk.params.max_positions,
                 self.risk.params.daily_loss_limit_pct)
        while not self._stop.is_set():
            started = time.monotonic()
            await self.run_once()
            elapsed = time.monotonic() - started
            wait = max(0.0, interval_seconds - elapsed)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=wait)
            except TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()
