"""SQLAlchemy models. Stage 1 tables: candles, market_snapshots, system_errors.

Timestamps are stored as integer epoch milliseconds (UTC) to avoid timezone bugs.
Prices use Float for market data; money/accounting tables in later stages will
use Numeric to avoid rounding drift.
"""
from __future__ import annotations

from sqlalchemy import BigInteger, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Candle(Base):
    """Closed OHLCV candles only (the forming candle is never stored)."""
    __tablename__ = "candles"
    __table_args__ = (UniqueConstraint("exchange", "symbol", "timeframe", "ts_ms", name="uq_candle"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange: Mapped[str] = mapped_column(String(32))
    symbol: Mapped[str] = mapped_column(String(32))
    timeframe: Mapped[str] = mapped_column(String(8))
    ts_ms: Mapped[int] = mapped_column(BigInteger)  # candle OPEN time
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"
    __table_args__ = (Index("ix_snapshot_lookup", "exchange", "symbol", "ts_ms"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange: Mapped[str] = mapped_column(String(32))
    symbol: Mapped[str] = mapped_column(String(32))
    ts_ms: Mapped[int] = mapped_column(BigInteger)
    bid: Mapped[float] = mapped_column(Float)
    ask: Mapped[float] = mapped_column(Float)
    last: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread_abs: Mapped[float] = mapped_column(Float)
    spread_bps: Mapped[float] = mapped_column(Float)
    bid_depth_quote: Mapped[float] = mapped_column(Float)
    ask_depth_quote: Mapped[float] = mapped_column(Float)
    imbalance: Mapped[float] = mapped_column(Float)
    base_volume_24h: Mapped[float | None] = mapped_column(Float, nullable=True)


class SystemErrorLog(Base):
    __tablename__ = "system_errors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts_ms: Mapped[int] = mapped_column(BigInteger)
    component: Mapped[str] = mapped_column(String(64))
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)
