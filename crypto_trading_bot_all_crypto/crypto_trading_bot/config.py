"""Central configuration for the trading bot.

Stage 1 only needs *public* market data, so no exchange credentials are read
here. Credentials arrive in the execution stage, via environment variables only.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, TypeVar

from dotenv import load_dotenv

T = TypeVar("T")

SUPPORTED_TIMEFRAMES: dict[str, int] = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}
DEFAULT_PAIRS = ("BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "LINK/USD", "AVAX/USD")

_PAIR_RE = re.compile(r"^[A-Z0-9]{2,10}/[A-Z0-9]{2,10}$")
_TRUE_VALUES = {"1", "true", "yes", "on"}
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigError(ValueError):
    """Raised when configuration is malformed or unsafe."""


def timeframe_ms(timeframe: str) -> int:
    try:
        return SUPPORTED_TIMEFRAMES[timeframe]
    except KeyError:
        raise ConfigError(
            f"Unsupported timeframe {timeframe!r}; supported: {', '.join(SUPPORTED_TIMEFRAMES)}"
        ) from None


@dataclass(frozen=True)
class Config:
    exchange_id: str = "alpaca"   # "alpaca" (paper keys, data only in Stage 1b) or any ccxt exchange id
    pairs: tuple[str, ...] = DEFAULT_PAIRS
    timeframes: tuple[str, ...] = tuple(SUPPORTED_TIMEFRAMES)
    database_url: str = "sqlite:///storage/trading_bot.db"
    history_days: int = 250        # indicators need 201+ candles per series (EMA200); 1d needs ~250 days
    poll_interval_seconds: float = 30.0
    orderbook_depth: int = 10
    max_concurrency: int = 3
    request_timeout_ms: int = 20_000
    log_level: str = "INFO"
    log_dir: Path = Path("logs")
    # Alpaca PAPER credentials only. Never repr'd or logged. Live keys are deliberately not read anywhere.
    alpaca_api_key: str = field(default="", repr=False)
    alpaca_api_secret: str = field(default="", repr=False)
    # Alpaca daily bars appear to be offset from UTC midnight (and may vary by an hour around DST), so the
    # newest daily bar is only trusted this long after its nominal close.
    daily_close_grace_minutes: int = 0
    # Stage 2: paper trading execution (still no live trading)
    paper_trading_enabled: bool = False
    # When True, ignore TRADING_PAIRS and use every tradable crypto pair Alpaca lists
    use_all_crypto: bool = False

    def __post_init__(self) -> None:
        if not self.use_all_crypto and not self.pairs:
            raise ConfigError("At least one trading pair is required (or set USE_ALL_CRYPTO=true)")
        for pair in self.pairs:
            if not _PAIR_RE.match(pair):
                raise ConfigError(f"Invalid trading pair {pair!r}; expected e.g. BTC/USD")
        if not self.timeframes:
            raise ConfigError("At least one timeframe is required")
        for tf in self.timeframes:
            timeframe_ms(tf)
        if self.history_days < 1:
            raise ConfigError("HISTORY_DAYS must be >= 1")
        if self.poll_interval_seconds < 1:
            raise ConfigError("POLL_INTERVAL_SECONDS must be >= 1")
        if not 1 <= self.orderbook_depth <= 100:
            raise ConfigError("ORDERBOOK_DEPTH must be between 1 and 100")
        if self.max_concurrency < 1:
            raise ConfigError("MAX_CONCURRENCY must be >= 1")
        if self.request_timeout_ms < 1000:
            raise ConfigError("REQUEST_TIMEOUT_MS must be >= 1000")
        if not 0 <= self.daily_close_grace_minutes <= 180:
            raise ConfigError("DAILY_CLOSE_GRACE_MINUTES must be between 0 and 180")
        if self.log_level.upper() not in _LOG_LEVELS:
            raise ConfigError(f"LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}")


def _parse(env: Mapping[str, str], name: str, default: T, cast: Callable[[str], T]) -> T:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw.strip())
    except ValueError:
        raise ConfigError(f"{name} must be a valid {cast.__name__}, got {raw!r}") from None


def _csv(env: Mapping[str, str], name: str, default: tuple[str, ...], upper: bool) -> tuple[str, ...]:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    items = [i.strip() for i in raw.split(",") if i.strip()]
    if upper:
        items = [i.upper() for i in items]
    return tuple(dict.fromkeys(items))  # de-duplicate, keep order


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Build a Config from a mapping (defaults to .env + process environment)."""
    if env is None:
        load_dotenv()
        env = os.environ

    if str(env.get("LIVE_TRADING_ENABLED", "false")).strip().lower() in _TRUE_VALUES:
        raise ConfigError(
            "LIVE_TRADING_ENABLED is set, but live trading is not implemented. "
            "It will require paper-trading validation and explicit human approval."
        )

    d = Config()
    exchange_id = _parse(env, "EXCHANGE_ID", d.exchange_id, str).lower()
    paper_enabled = str(env.get("PAPER_TRADING_ENABLED", "false")).strip().lower() in _TRUE_VALUES
    use_all = str(env.get("USE_ALL_CRYPTO", "false")).strip().lower() in _TRUE_VALUES
    # Also treat TRADING_PAIRS=all as a request for every pair
    raw_pairs = (env.get("TRADING_PAIRS") or "").strip().upper()
    if raw_pairs in {"ALL", "*"}:
        use_all = True
    pairs = () if use_all else _csv(env, "TRADING_PAIRS", d.pairs, upper=True)
    return Config(
        exchange_id=exchange_id,
        pairs=pairs,
        timeframes=_csv(env, "TIMEFRAMES", d.timeframes, upper=False),
        database_url=_parse(env, "DATABASE_URL", d.database_url, str),
        history_days=_parse(env, "HISTORY_DAYS", d.history_days, int),
        poll_interval_seconds=_parse(env, "POLL_INTERVAL_SECONDS", d.poll_interval_seconds, float),
        orderbook_depth=_parse(env, "ORDERBOOK_DEPTH", d.orderbook_depth, int),
        max_concurrency=_parse(env, "MAX_CONCURRENCY", d.max_concurrency, int),
        request_timeout_ms=_parse(env, "REQUEST_TIMEOUT_MS", d.request_timeout_ms, int),
        log_level=_parse(env, "LOG_LEVEL", d.log_level, str).upper(),
        log_dir=Path(_parse(env, "LOG_DIR", str(d.log_dir), str)),
        alpaca_api_key=_parse(env, "ALPACA_PAPER_API_KEY", "", str),
        alpaca_api_secret=_parse(env, "ALPACA_PAPER_API_SECRET", "", str),
        daily_close_grace_minutes=_parse(env, "DAILY_CLOSE_GRACE_MINUTES",
                                         65 if exchange_id == "alpaca" else 0, int),
        paper_trading_enabled=paper_enabled,
        use_all_crypto=use_all,
    )
