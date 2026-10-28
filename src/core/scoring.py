"""Shared stock score used by the live DecisionMaker and the backtest engine.

Score = cross-sectional z-score of 12-1 month momentum (price 21 trading days
ago / price 252 days ago - 1). Measured on 56 stocks 2011-2026: rank-IC 0.042
(t 2.35) vs 0.013 (t 0.70) for the old 60-day momentum formula. Volatility,
drawdown, revenue and news terms were dropped: vol added no IC (0.039 / 0.035
at 0.25 / 0.5 weight), fundamentals have no point-in-time history, news has 30
days.
"""
from __future__ import annotations

import pandas as pd

SKIP = 21      # skip most recent month (short-term reversal)
LOOKBACK = 252  # trading days
CLIP = 3.0     # winsorize z-scores


def mom_12_1(px: pd.DataFrame) -> pd.Series:
    """px: wide closes (index date ascending, columns stock_id), data < decision date only."""
    if len(px) <= LOOKBACK:
        return pd.Series(dtype=float)
    return (px.iloc[-1 - SKIP] / px.iloc[-1 - LOOKBACK] - 1).dropna()


def zscore(mom: pd.Series) -> pd.Series:
    if len(mom) < 2 or mom.std() == 0:
        return mom * 0.0
    return ((mom - mom.mean()) / mom.std()).clip(-CLIP, CLIP)


def score_prices(px: pd.DataFrame) -> pd.Series:
    return zscore(mom_12_1(px))
