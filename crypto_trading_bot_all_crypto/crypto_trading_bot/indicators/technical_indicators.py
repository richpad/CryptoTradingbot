"""Technical indicators in pure pandas/NumPy (no TA-Lib / pandas-ta dependency).

Why not TA-Lib or pandas-ta: TA-Lib needs a compiled C library, and pandas-ta has repeatedly broken
against new pandas/NumPy releases; both are install risks on a plain Windows setup. These functions are
validated against TA-Lib in the test suite instead (see tests/test_indicators.py).

NO LOOK-AHEAD: every value at row t is computed from rows <= t only. tests/test_indicators.py proves this
by recomputing on truncated data and requiring identical results.

Conventions (match TA-Lib / most charting platforms):
  * EMA is seeded with the simple mean of its first `n` values; RSI/ATR/ADX use Wilder smoothing.
  * Bollinger bands use POPULATION standard deviation (ddof=0).
  * Warm-up rows are NaN. A NaN never counts as a signal.
  * Input: closed candles only, UTC DatetimeIndex ascending, columns open/high/low/close/volume.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)


@dataclass(frozen=True)
class IndicatorParams:
    rsi_period: int = 14
    ema_periods: tuple[int, ...] = (9, 21, 50, 200)
    sma_periods: tuple[int, ...] = (20, 50, 200)
    macd: tuple[int, int, int] = (12, 26, 9)
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    vol_ma_period: int = 20
    fib_lookback: int = 100

    @property
    def warmup_rows(self) -> int:
        """Rows needed before every indicator (except Fibonacci) is defined, plus one for cross checks."""
        return max(max(self.ema_periods), max(self.sma_periods), self.macd[1] + self.macd[2] - 1,
                   2 * self.adx_period, self.bb_period, self.vol_ma_period) + 1


def validate_ohlcv(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("index must be a DatetimeIndex")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("index must be strictly ascending with no duplicates")


def _seeded_ewm(values: pd.Series, n: int, alpha: float) -> pd.Series:
    """EWM whose first output (at the n-th valid value) equals the plain mean of the first n valid values."""
    v = values.astype(float)
    valid = v.notna().to_numpy()
    out = pd.Series(np.nan, index=v.index)
    if not valid.any():
        return out
    start = int(valid.argmax())
    seed = start + n - 1
    if seed >= len(v):
        return out
    x = v.copy()
    x.iloc[:seed] = np.nan
    x.iloc[seed] = v.iloc[start:seed + 1].mean()
    return x.ewm(alpha=alpha, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.astype(float).rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return _seeded_ewm(s, n, 2.0 / (n + 1))


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI. A flat series (no gains, no losses) returns 50."""
    delta = close.astype(float).diff()
    avg_gain = _seeded_ewm(delta.clip(lower=0), n, 1.0 / n)
    avg_loss = _seeded_ewm((-delta).clip(lower=0), n, 1.0 / n)
    total = avg_gain + avg_loss
    out = 100.0 * avg_gain / total
    return out.where(total != 0, 50.0).where(total.notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    parts = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                       (df["low"] - prev_close).abs()], axis=1)
    return parts.max(axis=1, skipna=False)      # NaN on the first row (no previous close)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _seeded_ewm(true_range(df), n, 1.0 / n)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    c = close.astype(float)
    mid = c.rolling(n).mean()
    sd = c.rolling(n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = upper - lower
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
                         "bb_pctb": (c - lower) / width.where(width != 0),
                         "bb_width": width / mid})


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).where(up.notna())
    minus_dm = down.where((down > up) & (down > 0), 0.0).where(down.notna())
    tr_s = _seeded_ewm(true_range(df), n, 1.0 / n)
    plus_di = 100.0 * _seeded_ewm(plus_dm, n, 1.0 / n) / tr_s
    minus_di = 100.0 * _seeded_ewm(minus_dm, n, 1.0 / n) / tr_s
    di_sum = plus_di + minus_di
    dx = (100.0 * (plus_di - minus_di).abs() / di_sum).where(di_sum != 0, 0.0).where(di_sum.notna())
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": _seeded_ewm(dx, n, 1.0 / n)})


def vwap(df: pd.DataFrame, window: int | None = None) -> pd.Series:
    """Volume-weighted average price on typical price (H+L+C)/3.

    window=None  -> anchored: resets at each UTC midnight (crypto trades 24/7, so the UTC day is the session).
    window=N     -> rolling over the last N candles.
    NaN while cumulative volume is zero (e.g. quote-midpoint bars with no trades).
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    if window is not None:
        vol = df["volume"].rolling(window).sum()
        return (pv.rolling(window).sum() / vol).where(vol > 0)
    day = df.index.floor("D")
    cum_vol = df["volume"].groupby(day).cumsum()
    return (pv.groupby(day).cumsum() / cum_vol).where(cum_vol > 0)


def fib_levels(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """Fibonacci retracement levels of the swing high/low over the trailing `lookback` closed candles.

    fib_trend = +1 when the high came AFTER the low (up-swing; levels measured down from the high),
                -1 when the low came after the high (down-swing; levels measured up from the low).
    """
    hi = df["high"].rolling(lookback).max()
    lo = df["low"].rolling(lookback).min()
    pos_hi = df["high"].rolling(lookback).apply(np.argmax, raw=True)
    pos_lo = df["low"].rolling(lookback).apply(np.argmin, raw=True)
    up = pos_hi > pos_lo
    span = hi - lo
    out = pd.DataFrame({"swing_high": hi, "swing_low": lo,
                        "fib_trend": np.where(hi.isna(), np.nan, np.where(up, 1.0, -1.0))}, index=df.index)
    for r in FIB_RATIOS:
        out[f"fib_{int(round(r * 1000))}"] = np.where(up, hi - r * span, lo + r * span)
    return out.where(hi.notna())


def compute_indicators(df: pd.DataFrame, params: IndicatorParams = IndicatorParams()) -> pd.DataFrame:
    """Return a copy of `df` with all indicator columns added. Never modifies the input."""
    validate_ohlcv(df)
    p = params
    out = df.copy()
    out[f"rsi_{p.rsi_period}"] = rsi(out["close"], p.rsi_period)
    for n in p.ema_periods:
        out[f"ema_{n}"] = ema(out["close"], n)
    for n in p.sma_periods:
        out[f"sma_{n}"] = sma(out["close"], n)
    out = out.join(macd(out["close"], *p.macd))
    out = out.join(bollinger(out["close"], p.bb_period, p.bb_std))
    out[f"atr_{p.atr_period}"] = atr(out, p.atr_period)
    out["vwap"] = vwap(out)
    out = out.join(adx(out, p.adx_period))
    out[f"vol_sma_{p.vol_ma_period}"] = sma(out["volume"], p.vol_ma_period)
    out["vol_ratio"] = out["volume"] / out[f"vol_sma_{p.vol_ma_period}"].where(out[f"vol_sma_{p.vol_ma_period}"] > 0)
    out = out.join(fib_levels(out, p.fib_lookback))
    return out
