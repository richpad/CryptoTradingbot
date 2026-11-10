"""Historical OHLCV backfill: paginated, retrying, validated, idempotent.

Rules that prevent look-ahead bias and bad data:
  * Only COMPLETED candles are stored. A candle counts as complete when a later candle exists in
    the response, or when open + width (+ grace for daily bars) is in the past.
  * Paging continues from the last candle's open time + 1 ms, never "+ one candle width", so
    sources whose bars are not exactly 24h/4h wide (DST-affected) can never have a bar skipped.
  * Every row is validated before it touches the database; re-running a backfill is safe.
  * Bar alignment is audited and logged, because sources differ (UTC-aligned vs offset bars).
"""
from __future__ import annotations

import asyncio
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import ccxt

from config import timeframe_ms
from data.errors import TransientDataError
from database.database_manager import DatabaseManager
from monitoring.logger import get_logger

log = get_logger("historical")

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
MAX_PAGES = 10_000
RETRYABLE = (ccxt.NetworkError, TransientDataError)


def now_ms() -> int:
    return int(time.time() * 1000)


def validate_ohlcv_row(row: Any) -> bool:
    """True only for a well-formed, internally consistent [ts, o, h, l, c, v] row."""
    try:
        ts, o, h, l, c, v = (float(x) for x in row[:6])
    except (TypeError, ValueError, IndexError):
        return False
    if not all(math.isfinite(x) for x in (ts, o, h, l, c, v)):
        return False
    if ts <= 0 or not ts.is_integer():
        return False
    if min(o, h, l, c) <= 0 or v < 0:
        return False
    return h >= max(o, c, l) and l <= min(o, c, h)


@dataclass(frozen=True)
class Gap:
    start_ms: int   # open time of first missing candle
    end_ms: int     # open time of last missing candle
    missing: int


def find_gaps(timestamps: Sequence[int], tf_ms: int) -> list[Gap]:
    """Interior gaps of at least one whole missing candle (a 23h/25h daily bar is not a gap)."""
    gaps: list[Gap] = []
    for prev, cur in zip(timestamps, timestamps[1:]):
        missing = int((cur - prev) / tf_ms + 0.5) - 1   # rounded: tolerates +-1h bar widths
        if missing > 0:
            gaps.append(Gap(prev + tf_ms, cur - tf_ms, missing))
    return gaps


@dataclass(frozen=True)
class BackfillResult:
    symbol: str
    timeframe: str
    pages: int
    fetched: int
    inserted: int
    dropped_invalid: int
    first_ts_ms: int | None
    last_ts_ms: int | None
    utc_offset_ms: int | None = None   # most common (open_ts mod candle width); 0 = UTC-aligned
    dst_like_spacing: int = 0          # consecutive candles spaced 1h off the nominal width


class HistoricalDataFetcher:
    def __init__(self, exchange: Any, db: DatabaseManager, exchange_id: str, *,
                 max_retries: int = 5, base_delay: float = 1.0, page_limit: int | None = None,
                 daily_grace_ms: int = 0, clock: Callable[[], int] = now_ms) -> None:
        self.exchange = exchange
        self.db = db
        self.exchange_id = exchange_id
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.page_limit = page_limit
        self.daily_grace_ms = daily_grace_ms
        self.clock = clock

    def _grace(self, tf_ms: int) -> int:
        return self.daily_grace_ms if tf_ms >= DAY_MS else 0

    async def _fetch_page(self, symbol: str, timeframe: str, since: int) -> list[list[float]]:
        for attempt in range(self.max_retries + 1):
            try:
                return await self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=self.page_limit)
            except RETRYABLE as exc:   # network, rate limit, timeout, exchange unavailable
                if attempt == self.max_retries:
                    raise
                delay = min(self.base_delay * 2 ** attempt, 60.0)
                log.warning("%s %s: %s (retry %d/%d in %.1fs)", symbol, timeframe,
                            type(exc).__name__, attempt + 1, self.max_retries, delay)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def backfill(self, symbol: str, timeframe: str, start_ms: int,
                       end_ms: int | None = None) -> BackfillResult:
        tf_ms = timeframe_ms(timeframe)
        grace = self._grace(tf_ms)
        aligned_start = (start_ms // tf_ms) * tf_ms
        end = end_ms if end_ms is not None else self.clock()
        since = aligned_start
        pages = fetched = inserted = dropped = 0
        first_seen = last_stored = prev_ts = None
        offsets: Counter[int] = Counter()
        dst_like = 0

        if since + tf_ms > min(end, self.clock()):       # not even one candle can be complete yet
            return BackfillResult(symbol, timeframe, 0, 0, 0, 0, None, None)

        while pages < MAX_PAGES:
            page = await self._fetch_page(symbol, timeframe, since)
            pages += 1
            if not page:
                break
            valid = [r for r in page if validate_ohlcv_row(r)]
            dropped += len(page) - len(valid)
            if len(valid) != len(page):
                log.warning("%s %s: dropped %d malformed candle(s)", symbol, timeframe, len(page) - len(valid))
            if not valid:
                break

            by_ts = {int(r[0]): r for r in valid}
            newest = max(by_ts)
            now = self.clock()
            rows = [by_ts[t] for t in sorted(by_ts)
                    if aligned_start <= t < end and (t < newest or t + tf_ms + grace <= now)]
            fetched += len(rows)
            for r in rows:
                t = int(r[0])
                offsets[t % tf_ms] += 1
                if prev_ts is not None and t != prev_ts + tf_ms and abs(t - prev_ts - tf_ms) <= HOUR_MS:
                    dst_like += 1
                prev_ts = t
            if rows:
                first_seen = int(rows[0][0]) if first_seen is None else min(first_seen, int(rows[0][0]))
                last_stored = int(rows[-1][0]) if last_stored is None else max(last_stored, int(rows[-1][0]))
                inserted += await asyncio.to_thread(
                    self.db.upsert_candles, self.exchange_id, symbol, timeframe, rows)

            next_since = newest + 1
            if next_since <= since:                      # exchange made no forward progress
                break
            if newest + tf_ms > min(end, self.clock()):  # reached the head of the series (or the range end)
                break
            since = next_since

        utc_offset = offsets.most_common(1)[0][0] if offsets else None
        if utc_offset:
            log.warning("%s %s: candles open at a fixed offset of %.2fh from UTC epoch alignment",
                        symbol, timeframe, utc_offset / HOUR_MS)
        if dst_like:
            log.warning("%s %s: %d candle(s) spaced 1h away from the nominal width (DST-like); "
                        "bar width is not constant for this source", symbol, timeframe, dst_like)
        if first_seen is not None and first_seen > aligned_start + 2 * tf_ms:
            log.warning(
                "%s %s: earliest candle available is %d but %d was requested; the source likely "
                "limits how much history it serves. Use a different source for deeper history.",
                symbol, timeframe, first_seen, aligned_start)
        log.info("%s %s: pages=%d fetched=%d inserted=%d dropped=%d",
                 symbol, timeframe, pages, fetched, inserted, dropped)
        return BackfillResult(symbol, timeframe, pages, fetched, inserted, dropped, first_seen,
                              last_stored, utc_offset, dst_like)

    async def repair_gaps(self, symbol: str, timeframe: str, start_ms: int, end_ms: int | None = None) -> int:
        """Re-fetch interior gaps in stored data. Returns number of candles inserted."""
        tf_ms = timeframe_ms(timeframe)
        stamps = await asyncio.to_thread(
            self.db.get_timestamps, self.exchange_id, symbol, timeframe, start_ms, end_ms)
        total = 0
        for gap in find_gaps(stamps, tf_ms):
            result = await self.backfill(symbol, timeframe, gap.start_ms, gap.end_ms + tf_ms)
            total += result.inserted
        return total
