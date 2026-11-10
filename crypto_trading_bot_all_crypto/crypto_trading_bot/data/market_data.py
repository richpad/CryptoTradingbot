"""Real-time market data: ticker/order-book snapshots, incremental candle updates, volatility."""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable

import numpy as np
import pandas as pd

from config import Config, timeframe_ms
from data.historical_data import HistoricalDataFetcher, now_ms
from database.database_manager import DatabaseManager
from monitoring.logger import get_logger

log = get_logger("market_data")

BOOTSTRAP_CANDLES = 300  # candles fetched when a series is empty (use `backfill` for real history)


class MarketDataError(Exception):
    """Exchange returned data that failed validation."""


@dataclass(frozen=True)
class MarketSnapshot:
    exchange: str
    symbol: str
    ts_ms: int
    bid: float
    ask: float
    last: float | None
    spread_abs: float
    spread_bps: float
    bid_depth_quote: float
    ask_depth_quote: float
    imbalance: float          # (bid_depth - ask_depth) / (bid_depth + ask_depth), in [-1, 1]
    base_volume_24h: float | None

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def compute_snapshot(exchange_id: str, symbol: str, ticker: dict[str, Any], orderbook: dict[str, Any],
                     depth_levels: int, ts_ms: int) -> MarketSnapshot:
    bids, asks = orderbook.get("bids") or [], orderbook.get("asks") or []
    if not bids or not asks:
        raise MarketDataError(f"{symbol}: empty order book")
    try:
        best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
        bid_depth = sum(float(l[0]) * float(l[1]) for l in bids[:depth_levels])
        ask_depth = sum(float(l[0]) * float(l[1]) for l in asks[:depth_levels])
    except (TypeError, ValueError, IndexError) as exc:
        raise MarketDataError(f"{symbol}: malformed order book ({exc})") from exc
    if not all(math.isfinite(x) for x in (best_bid, best_ask, bid_depth, ask_depth)) \
            or best_bid <= 0 or best_ask <= 0:
        raise MarketDataError(f"{symbol}: non-positive or non-finite book values")
    if best_bid > best_ask:
        raise MarketDataError(f"{symbol}: crossed book (bid {best_bid} > ask {best_ask})")

    spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2
    total_depth = bid_depth + ask_depth
    last = ticker.get("last")
    volume = ticker.get("baseVolume")
    return MarketSnapshot(
        exchange=exchange_id, symbol=symbol, ts_ms=ts_ms, bid=best_bid, ask=best_ask,
        last=float(last) if last is not None else None,
        spread_abs=spread, spread_bps=spread / mid * 10_000,
        bid_depth_quote=bid_depth, ask_depth_quote=ask_depth,
        imbalance=(bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0,
        base_volume_24h=float(volume) if volume is not None else None)


def realized_volatility(closes: pd.Series, timeframe: str, window: int | None = None) -> float | None:
    """Annualized realized volatility from log returns (24/7 market: 365 days/year).

    Uses only the closes supplied, so callers control the information set.
    """
    returns = np.log(closes.astype(float)).diff().dropna()
    if window:
        returns = returns.iloc[-window:]
    if len(returns) < 2:
        return None
    periods_per_year = 365 * 86_400_000 / timeframe_ms(timeframe)
    value = float(returns.std(ddof=1) * math.sqrt(periods_per_year))
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class CycleReport:
    snapshots_ok: int
    candles_inserted: int
    errors: int
    tasks: int


class MarketDataCollector:
    def __init__(self, exchange: Any, db: DatabaseManager, config: Config, *,
                 fetcher: HistoricalDataFetcher | None = None,
                 clock: Callable[[], int] = now_ms) -> None:
        self.exchange, self.db, self.config, self.clock = exchange, db, config, clock
        self.fetcher = fetcher or HistoricalDataFetcher(
            exchange, db, config.exchange_id, clock=clock,
            daily_grace_ms=config.daily_close_grace_minutes * 60_000)
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    async def collect_snapshot(self, symbol: str) -> MarketSnapshot:
        ticker, book = await asyncio.gather(
            self.exchange.fetch_ticker(symbol),
            self.exchange.fetch_order_book(symbol, self.config.orderbook_depth))
        snap = compute_snapshot(self.config.exchange_id, symbol, ticker, book,
                                self.config.orderbook_depth, self.clock())
        await asyncio.to_thread(self.db.insert_snapshot, snap.as_row())
        return snap

    async def update_candles(self, symbol: str, timeframe: str) -> int:
        """Fetch newly CLOSED candles only; makes no API call if none can exist yet."""
        tf_ms, now = timeframe_ms(timeframe), self.clock()
        latest = await asyncio.to_thread(self.db.get_latest_ts, self.config.exchange_id, symbol, timeframe)
        if latest is None:
            start = (now // tf_ms) * tf_ms - BOOTSTRAP_CANDLES * tf_ms
        elif latest + 2 * tf_ms > now:      # next candle (latest+tf) closes at latest+2*tf
            return 0
        else:
            start = latest + 1   # alignment-agnostic: next candle opens after the latest one
        return (await self.fetcher.backfill(symbol, timeframe, start)).inserted

    async def get_volatility(self, symbol: str, timeframe: str, window: int = 30) -> float | None:
        df = await asyncio.to_thread(self.db.get_candles, self.config.exchange_id, symbol,
                                     timeframe, None, None, window + 1)
        return realized_volatility(df["close"], timeframe, window) if len(df) else None

    async def _guarded(self, component: str, func: Callable[..., Awaitable[Any]], *args: Any) -> tuple[bool, Any]:
        """Run one task under the concurrency cap; a failure never stops sibling tasks."""
        async with self._semaphore:
            try:
                return True, await func(*args)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - isolate every per-pair failure
                log.error("%s failed: %s: %s", component, type(exc).__name__, exc)
                try:
                    await asyncio.to_thread(self.db.log_error, component, f"{type(exc).__name__}: {exc}")
                except Exception:  # noqa: BLE001
                    log.exception("could not persist error for %s", component)
                return False, None

    async def run_once(self) -> CycleReport:
        jobs = [self._guarded(f"snapshot:{s}", self.collect_snapshot, s) for s in self.config.pairs]
        jobs += [self._guarded(f"candles:{s}:{tf}", self.update_candles, s, tf)
                 for s in self.config.pairs for tf in self.config.timeframes]
        results = await asyncio.gather(*jobs)
        n_snap = len(self.config.pairs)
        return CycleReport(
            snapshots_ok=sum(1 for ok, _ in results[:n_snap] if ok),
            candles_inserted=sum(v for ok, v in results[n_snap:] if ok),
            errors=sum(1 for ok, _ in results if not ok),
            tasks=len(results))

    async def run_forever(self, stop: asyncio.Event) -> None:
        failed_cycles = 0
        while not stop.is_set():
            started = time.monotonic()
            report = await self.run_once()
            log.info("cycle: snapshots=%d/%d new_candles=%d errors=%d",
                     report.snapshots_ok, len(self.config.pairs), report.candles_inserted, report.errors)
            failed_cycles = failed_cycles + 1 if report.errors == report.tasks else 0
            interval = self.config.poll_interval_seconds * (2 ** min(failed_cycles, 4))
            wait = max(0.0, min(interval, 300.0) - (time.monotonic() - started))
            try:
                await asyncio.wait_for(stop.wait(), timeout=wait)
            except TimeoutError:
                pass
