"""Multi-indicator scoring (Stage 2). Positive = bullish evidence, negative = bearish evidence.

IMPORTANT: the weights below are UNTUNED PRIORS, not validated parameters. Trend components and
mean-reversion components (RSI zone, Bollinger) deliberately pull in opposite directions in strong
trends, which is why a later stage chooses strategies by market regime instead of trusting one
blended number. Nothing here has been shown to be profitable.

Input must be the output of compute_indicators() (closed candles only). Vectorised and causal:
each row's score uses only that row and earlier rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreWeights:
    trend_stack: float = 1.0      # EMA9 > EMA21 > EMA50 (bull) / reverse (bear)
    price_vs_ema200: float = 1.0  # long-term trend filter
    macd: float = 1.0             # histogram sign AND direction agree
    rsi_zone: float = 1.0         # oversold = entry consideration (+), overbought = exit consideration (-)
    bollinger: float = 0.5        # below lower band (+) / above upper band (-)
    volume: float = 0.5           # unusual volume confirms the candle's direction
    vwap: float = 0.25            # close vs day-anchored VWAP
    adx_strength: float = 0.5     # strong trend (ADX) votes with the dominant directional index


@dataclass(frozen=True)
class ScoreThresholds:
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    adx_trending: float = 25.0
    unusual_volume_ratio: float = 1.5


def _sign(cond_pos: pd.Series, cond_neg: pd.Series) -> pd.Series:
    return cond_pos.astype(float) - cond_neg.astype(float)      # comparisons with NaN are False -> 0


def score_frame(ind: pd.DataFrame, weights: ScoreWeights = ScoreWeights(),
                th: ScoreThresholds = ScoreThresholds(), rsi_col: str = "rsi_14") -> pd.DataFrame:
    c = ind["close"]
    hist = ind["macd_hist"]
    comp = pd.DataFrame(index=ind.index)
    comp["s_trend_stack"] = weights.trend_stack * _sign(
        (ind["ema_9"] > ind["ema_21"]) & (ind["ema_21"] > ind["ema_50"]),
        (ind["ema_9"] < ind["ema_21"]) & (ind["ema_21"] < ind["ema_50"]))
    comp["s_price_vs_ema200"] = weights.price_vs_ema200 * _sign(c > ind["ema_200"], c < ind["ema_200"])
    comp["s_macd"] = weights.macd * _sign((hist > 0) & (hist > hist.shift(1)),
                                          (hist < 0) & (hist < hist.shift(1)))
    comp["s_rsi_zone"] = weights.rsi_zone * _sign(ind[rsi_col] < th.rsi_oversold, ind[rsi_col] > th.rsi_overbought)
    comp["s_bollinger"] = weights.bollinger * _sign(c < ind["bb_lower"], c > ind["bb_upper"])
    unusual = ind["vol_ratio"] > th.unusual_volume_ratio
    comp["s_volume"] = weights.volume * _sign(unusual & (c > ind["open"]), unusual & (c < ind["open"]))
    comp["s_vwap"] = weights.vwap * _sign(c > ind["vwap"], c < ind["vwap"])
    trending = ind["adx"] > th.adx_trending
    comp["s_adx_strength"] = weights.adx_strength * _sign(trending & (ind["plus_di"] > ind["minus_di"]),
                                                          trending & (ind["plus_di"] < ind["minus_di"]))
    comp["score"] = comp.sum(axis=1)
    return comp.replace(-0.0, 0.0)
