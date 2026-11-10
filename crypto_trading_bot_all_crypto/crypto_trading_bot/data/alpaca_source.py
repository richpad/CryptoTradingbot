"""Alpaca crypto data adapter.

Exposes the small ccxt-style surface the collector and fetcher already use
(fetch_ohlcv / fetch_ticker / fetch_order_book / load_markets / symbols / has / close), so nothing
downstream changes. Stage 1b is DATA ONLY: this module places no orders. The trading client is
hard-wired to the PAPER endpoint and is used only to list tradable assets.

alpaca-py is synchronous, so calls run in a worker thread behind one lock with light pacing.
Only HTTP status codes are surfaced in errors, never request headers, so keys cannot leak.
"""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

import requests
from alpaca.common.enums import Sort
from alpaca.common.exceptions import APIError
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import (CryptoBarsRequest, CryptoLatestOrderbookRequest,
                                  CryptoLatestTradeRequest)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from data.errors import DataSourceError, TransientDataError
from monitoring.logger import get_logger

log = get_logger("alpaca")

MAX_BARS_PER_REQUEST = 10_000
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
_TIMEFRAMES = {
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "4h": TimeFrame(4, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
}


def _status(exc: APIError) -> int | None:
    try:
        return exc.status_code
    except AttributeError:
        return None


class AlpacaDataAdapter:
    has = {"fetchOHLCV": True, "fetchTicker": True, "fetchOrderBook": True}

    def __init__(self, api_key: str, api_secret: str, *, data_client: Any = None,
                 trading_client: Any = None, min_interval_seconds: float = 0.3) -> None:
        if not api_key or not api_secret:
            raise DataSourceError("Alpaca paper API key and secret are required")
        self._data = data_client or CryptoHistoricalDataClient(api_key, api_secret)
        self._trading = trading_client or TradingClient(api_key, api_secret, paper=True)  # PAPER only
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._min_interval = min_interval_seconds
        self.symbols: list[str] = []
        self.markets: dict[str, dict[str, float | None]] = {}

    # ---- sync core -----------------------------------------------------
    def _call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                return func(*args, **kwargs)
            except APIError as exc:
                status = _status(exc)
                if status in _TRANSIENT_STATUS:
                    raise TransientDataError(f"Alpaca HTTP {status}") from None
                raise DataSourceError(f"Alpaca API error (HTTP {status})") from None
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                raise TransientDataError(f"Alpaca network error: {type(exc).__name__}") from None
            finally:
                self._last_call = time.monotonic()

    async def _run(self, func: Callable[..., Any], *args: Any) -> Any:
        return await asyncio.to_thread(self._call, func, *args)

    # ---- ccxt-style surface -------------------------------------------
    async def load_markets(self) -> None:
        assets = await self._run(
            self._trading.get_all_assets,
            GetAssetsRequest(asset_class=AssetClass.CRYPTO, status=AssetStatus.ACTIVE))
        tradable = [a for a in assets if a.tradable]
        self.symbols = sorted(a.symbol for a in tradable)
        self.markets = {a.symbol: {"min_order_size": a.min_order_size,
                                   "min_trade_increment": a.min_trade_increment,
                                   "price_increment": a.price_increment} for a in tradable}
        log.info("Alpaca lists %d tradable crypto assets", len(self.symbols))

    async def fetch_ohlcv(self, symbol: str, timeframe: str, since: int | None = None,
                          limit: int | None = None) -> list[list[float]]:
        if timeframe not in _TIMEFRAMES:
            raise DataSourceError(f"Unsupported timeframe {timeframe!r} for Alpaca")
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol, timeframe=_TIMEFRAMES[timeframe],
            start=datetime.fromtimestamp(since / 1000, tz=timezone.utc) if since is not None else None,
            limit=min(limit or MAX_BARS_PER_REQUEST, MAX_BARS_PER_REQUEST), sort=Sort.ASC)
        result = await self._run(self._data.get_crypto_bars, request)
        bars = result.data.get(symbol, [])
        return [[int(b.timestamp.timestamp() * 1000), b.open, b.high, b.low, b.close, b.volume]
                for b in bars]

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        trades = await self._run(self._data.get_crypto_latest_trade,
                                 CryptoLatestTradeRequest(symbol_or_symbols=symbol))
        trade = trades.get(symbol)
        if trade is None:
            raise DataSourceError(f"{symbol}: no latest trade returned")
        # Alpaca gives no rolling 24h volume; leave it empty rather than mislabel a calendar-day figure.
        return {"last": float(trade.price), "baseVolume": None,
                "timestamp": int(trade.timestamp.timestamp() * 1000)}

    async def fetch_order_book(self, symbol: str, limit: int | None = None) -> dict[str, Any]:
        books = await self._run(self._data.get_crypto_latest_orderbook,
                                CryptoLatestOrderbookRequest(symbol_or_symbols=symbol))
        book = books.get(symbol)
        if book is None:
            raise DataSourceError(f"{symbol}: no order book returned")
        bids = sorted(([q.price, q.size] for q in book.bids), key=lambda x: -x[0])   # best first
        asks = sorted(([q.price, q.size] for q in book.asks), key=lambda x: x[0])
        if limit:
            bids, asks = bids[:limit], asks[:limit]
        return {"bids": bids, "asks": asks, "timestamp": int(book.timestamp.timestamp() * 1000)}

    async def close(self) -> None:   # alpaca-py holds no persistent connections we need to close
        return None
