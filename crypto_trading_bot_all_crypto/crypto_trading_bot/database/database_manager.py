"""Database access layer (SQLAlchemy 2.0; SQLite for dev, PostgreSQL-ready)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database.models import Base, Candle, MarketSnapshot, SystemErrorLog

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DatabaseManager:
    def __init__(self, url: str) -> None:
        parsed = make_url(url)
        kwargs: dict[str, Any] = {}
        if parsed.get_backend_name() == "sqlite":
            kwargs["connect_args"] = {"check_same_thread": False}
            if parsed.database in (None, "", ":memory:"):
                kwargs["poolclass"] = StaticPool
            else:
                Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, **kwargs)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def init_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()

    # ---- candles -------------------------------------------------------
    def _insert_ignore(self, model: type[Base], rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        dialect = self.engine.dialect.name
        if dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert
        elif dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        else:
            raise NotImplementedError(f"Unsupported database dialect: {dialect}")
        with self._sessions.begin() as session:
            session.execute(insert(model).on_conflict_do_nothing(), rows)

    def upsert_candles(self, exchange: str, symbol: str, timeframe: str,
                       rows: Sequence[Sequence[float]]) -> int:
        """Insert validated [ts, o, h, l, c, v] rows. Idempotent; returns NEW row count."""
        if not rows:
            return 0
        lo, hi = int(min(r[0] for r in rows)), int(max(r[0] for r in rows))
        existing = set(self.get_timestamps(exchange, symbol, timeframe, lo, hi))
        new = [
            {"exchange": exchange, "symbol": symbol, "timeframe": timeframe, "ts_ms": int(r[0]),
             "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
             "close": float(r[4]), "volume": float(r[5])}
            for r in rows if int(r[0]) not in existing
        ]
        self._insert_ignore(Candle, new)  # ON CONFLICT DO NOTHING guards races
        return len(new)

    def get_timestamps(self, exchange: str, symbol: str, timeframe: str,
                       start_ms: int | None = None, end_ms: int | None = None) -> list[int]:
        stmt = select(Candle.ts_ms).where(
            Candle.exchange == exchange, Candle.symbol == symbol, Candle.timeframe == timeframe)
        if start_ms is not None:
            stmt = stmt.where(Candle.ts_ms >= start_ms)
        if end_ms is not None:
            stmt = stmt.where(Candle.ts_ms <= end_ms)
        with self._sessions() as session:
            return [int(r[0]) for r in session.execute(stmt.order_by(Candle.ts_ms)).all()]

    def get_latest_ts(self, exchange: str, symbol: str, timeframe: str) -> int | None:
        stmt = select(func.max(Candle.ts_ms)).where(
            Candle.exchange == exchange, Candle.symbol == symbol, Candle.timeframe == timeframe)
        with self._sessions() as session:
            value = session.execute(stmt).scalar()
        return None if value is None else int(value)

    def get_candles(self, exchange: str, symbol: str, timeframe: str,
                    start_ms: int | None = None, end_ms: int | None = None,
                    limit: int | None = None) -> pd.DataFrame:
        """Candles as a UTC-indexed DataFrame, ascending. `limit` keeps the most recent N."""
        cols = [Candle.ts_ms, Candle.open, Candle.high, Candle.low, Candle.close, Candle.volume]
        stmt = select(*cols).where(
            Candle.exchange == exchange, Candle.symbol == symbol, Candle.timeframe == timeframe)
        if start_ms is not None:
            stmt = stmt.where(Candle.ts_ms >= start_ms)
        if end_ms is not None:
            stmt = stmt.where(Candle.ts_ms <= end_ms)
        stmt = stmt.order_by(Candle.ts_ms.desc() if limit else Candle.ts_ms)
        if limit:
            stmt = stmt.limit(limit)
        with self._sessions() as session:
            rows = [tuple(r) for r in session.execute(stmt).all()]
        df = pd.DataFrame(rows, columns=["ts_ms"] + OHLCV_COLUMNS)
        if limit:
            df = df.iloc[::-1]
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts_ms").astype("int64"), unit="ms", utc=True),
                                    name="timestamp")
        return df.astype(float)

    def coverage(self, exchange: str | None = None) -> list[dict[str, Any]]:
        """Stored candle ranges per (exchange, symbol, timeframe); all exchanges when None."""
        stmt = (select(Candle.exchange, Candle.symbol, Candle.timeframe, func.count(),
                       func.min(Candle.ts_ms), func.max(Candle.ts_ms))
                .group_by(Candle.exchange, Candle.symbol, Candle.timeframe)
                .order_by(Candle.exchange, Candle.symbol, Candle.timeframe))
        if exchange is not None:
            stmt = stmt.where(Candle.exchange == exchange)
        with self._sessions() as session:
            return [{"exchange": e, "symbol": s, "timeframe": tf, "count": int(n),
                     "first_ts_ms": int(a), "last_ts_ms": int(b)}
                    for e, s, tf, n, a, b in session.execute(stmt).all()]

    # ---- snapshots & errors -------------------------------------------
    def insert_snapshot(self, row: Mapping[str, Any]) -> None:
        self._insert_ignore(MarketSnapshot, [dict(row)])

    def get_snapshots(self, exchange: str, symbol: str, limit: int = 100) -> list[MarketSnapshot]:
        stmt = (select(MarketSnapshot)
                .where(MarketSnapshot.exchange == exchange, MarketSnapshot.symbol == symbol)
                .order_by(MarketSnapshot.ts_ms.desc()).limit(limit))
        with self._sessions() as session:
            return list(session.execute(stmt).scalars().all())

    def snapshots_df(self, exchange: str, symbol: str, limit: int = 500) -> pd.DataFrame:
        """Most recent market snapshots as a UTC-indexed DataFrame, ascending."""
        cols = ["bid", "ask", "last", "spread_abs", "spread_bps", "bid_depth_quote",
                "ask_depth_quote", "imbalance", "base_volume_24h"]
        rows = self.get_snapshots(exchange, symbol, limit)
        df = pd.DataFrame([[r.ts_ms] + [getattr(r, c) for c in cols] for r in rows], columns=["ts_ms"] + cols)
        df = df.iloc[::-1]
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts_ms").astype("int64"), unit="ms", utc=True),
                                    name="timestamp")
        return df

    def log_error(self, component: str, message: str, details: str | None = None,
                  ts_ms: int | None = None) -> None:
        self._insert_ignore(SystemErrorLog, [{
            "ts_ms": ts_ms if ts_ms is not None else int(time.time() * 1000),
            "component": component[:64], "message": message, "details": details}])

    def get_errors(self, limit: int = 50) -> list[SystemErrorLog]:
        stmt = select(SystemErrorLog).order_by(SystemErrorLog.ts_ms.desc()).limit(limit)
        with self._sessions() as session:
            return list(session.execute(stmt).scalars().all())
