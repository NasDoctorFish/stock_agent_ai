"""
US Market Regime Detector
=========================
Classifies the current US market regime using 6 signals sourced from
yfinance and FRED (fredapi).  This replaces the Alpha Vantage dependency
in the original regime_detector.py so the system can run 24/7 without
hitting the 25 req/day limit of the free Alpha Vantage tier.

Signals:
  1. S&P 500 trend     — SPY close vs. 200-week SMA
  2. Yield curve       — 10yr (^TNX) minus 2yr (DGS2 / ^IRX) spread
  3. VIX               — CBOE Volatility Index (^VIX) level
  4. Credit spread     — HYG close vs. 20-week SMA (high-yield ETF proxy)
  5. Liquidity         — Fed Funds Rate direction (FRED: FEDFUNDS)
  6. Momentum          — SPY 12-month minus 1-month return

Regime score (-6 to +6):
  +4 to +6  : Risk-On    (Bull Market)
   0 to +3  : Cautious   (Late Cycle / Mixed)
  -3 to -1  : Defensive  (Early Bear)
  -6 to -4  : Risk-Off   (Bear Market)

Usage:
    from regime_detector.us_regime import USRegimeDetector
    detector = USRegimeDetector()
    result = detector.detect()
    # result is a dict: {regime, score, signals, as_of, error}
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# FRED is optional; fall back to yfinance approximations if unavailable
try:
    from fredapi import Fred
    _FRED_AVAILABLE = True
except ImportError:
    _FRED_AVAILABLE = False
    logger.warning("fredapi not installed — Treasury yields fall back to yfinance ^TNX/^IRX")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _yf_weekly(ticker: str, years: int = 5) -> pd.DataFrame:
    """Download weekly OHLCV from yfinance and return a clean DataFrame."""
    start = (datetime.today() - timedelta(weeks=years * 52)).strftime("%Y-%m-%d")
    raw = yf.download(ticker, start=start, interval="1wk", auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned empty data for {ticker}")
    # Flatten MultiIndex columns if present (yfinance >= 0.2.x)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index)
    return raw.sort_index()


def _fred_series(fred_key: str, series_id: str, years: int = 5) -> Optional[pd.Series]:
    """Fetch a FRED series and return a monthly pd.Series, or None on failure."""
    if not _FRED_AVAILABLE:
        return None
    try:
        fred = Fred(api_key=fred_key)
        start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        s = fred.get_series(series_id, observation_start=start)
        s.index = pd.to_datetime(s.index)
        return s.dropna().sort_index()
    except Exception as exc:
        logger.warning(f"FRED fetch failed for {series_id}: {exc}")
        return None


def _yf_close_weekly(ticker: str, years: int = 5) -> pd.Series:
    """Return weekly close series from yfinance (index = price level, not return)."""
    df = _yf_weekly(ticker, years)
    return df["close"].dropna()


# ── Signal Engineering ────────────────────────────────────────────────────────

class USFeatureEngineer:
    """Computes binary (+1 bullish / -1 bearish) regime signals for the US market."""

    @staticmethod
    def sp_trend(spy_weekly: pd.Series) -> int:
        """S&P trend: +1 if price > 200-week SMA, else -1."""
        if len(spy_weekly) < 50:
            return 0
        sma200 = spy_weekly.rolling(200, min_periods=50).mean()
        return int(np.sign(spy_weekly.iloc[-1] - sma200.iloc[-1])) or 1

    @staticmethod
    def yield_curve(y10: pd.Series, y2: pd.Series) -> int:
        """Yield curve: +1 if 10yr > 2yr (normal), -1 if inverted."""
        try:
            # Align on a common monthly index then take latest
            combined = pd.concat([y10.rename("y10"), y2.rename("y2")], axis=1).ffill().dropna()
            if combined.empty:
                return 0
            spread = combined["y10"].iloc[-1] - combined["y2"].iloc[-1]
            return 1 if spread > 0 else -1
        except Exception:
            return 0

    @staticmethod
    def vix_signal(vix_weekly: pd.Series) -> int:
        """VIX: +1 if < 20 (low fear), -1 if ≥ 20 (elevated fear)."""
        if vix_weekly.empty:
            return 0
        return 1 if vix_weekly.iloc[-1] < 20 else -1

    @staticmethod
    def credit_spread(hyg_weekly: pd.Series) -> int:
        """Credit spread proxy via HYG: +1 if above 20-week SMA, -1 otherwise."""
        if len(hyg_weekly) < 5:
            return 0
        sma20 = hyg_weekly.rolling(20, min_periods=5).mean()
        return 1 if hyg_weekly.iloc[-1] >= sma20.iloc[-1] else -1

    @staticmethod
    def liquidity(fed_funds: pd.Series) -> int:
        """Liquidity: +1 if Fed Funds falling or flat over 3 months, -1 if rising."""
        if len(fed_funds) < 4:
            return 0
        # Use last 13 weeks (~3 months) of change; monthly data → ~3 observations
        change = fed_funds.iloc[-1] - fed_funds.iloc[-4]
        return 1 if change <= 0 else -1

    @staticmethod
    def momentum(spy_weekly: pd.Series) -> int:
        """Momentum: 12-month minus 1-month return. +1 if positive, -1 otherwise."""
        if len(spy_weekly) < 53:
            return 0
        ret_12m = spy_weekly.iloc[-1] / spy_weekly.iloc[-53] - 1
        ret_1m = spy_weekly.iloc[-1] / spy_weekly.iloc[-5] - 1
        mom = ret_12m - ret_1m
        return 1 if mom > 0 else -1


# ── Detector ──────────────────────────────────────────────────────────────────

class USRegimeDetector:
    """
    Fetches data and computes the US market regime.

    Args:
        fred_api_key: FRED API key (optional; improves yield data quality).
        lookback_years: How many years of history to fetch (default 5).
    """

    SCORE_LABELS = {
        (4, 6):   "Risk-On (Bull Market)",
        (0, 3):   "Cautious (Late Cycle / Mixed)",
        (-3, -1): "Defensive (Early Bear)",
        (-6, -4): "Risk-Off (Bear Market)",
    }

    def __init__(self, fred_api_key: Optional[str] = None, lookback_years: int = 5):
        self.fred_api_key = fred_api_key
        self.lookback_years = lookback_years
        self._fe = USFeatureEngineer()

    @staticmethod
    def _classify(score: int) -> str:
        if score >= 4:
            return "Risk-On (Bull Market)"
        elif score >= 0:
            return "Cautious (Late Cycle / Mixed)"
        elif score >= -3:
            return "Defensive (Early Bear)"
        return "Risk-Off (Bear Market)"

    def detect(self) -> Dict:
        """
        Run the 6-signal regime detection pipeline.

        Returns a dict:
            {
                "regime":  str,        # human-readable label
                "score":   int,        # -6 to +6
                "signals": dict,       # each signal's vote (+1 or -1)
                "as_of":   str,        # ISO date of the latest data point
                "market":  "US",
                "error":   str | None, # non-None if something failed
            }
        """
        signals: Dict[str, int] = {}
        errors = []

        # ── 1. Fetch data ─────────────────────────────────────────────────────
        try:
            spy_w = _yf_close_weekly("SPY", self.lookback_years)
        except Exception as exc:
            errors.append(f"SPY fetch failed: {exc}")
            spy_w = pd.Series(dtype=float)

        try:
            vix_w = _yf_close_weekly("^VIX", self.lookback_years)
        except Exception as exc:
            errors.append(f"VIX fetch failed: {exc}")
            vix_w = pd.Series(dtype=float)

        try:
            hyg_w = _yf_close_weekly("HYG", self.lookback_years)
        except Exception as exc:
            errors.append(f"HYG fetch failed: {exc}")
            hyg_w = pd.Series(dtype=float)

        # Treasury yields — prefer FRED, fall back to yfinance
        y10 = _fred_series(self.fred_api_key or "", "DGS10", self.lookback_years) if self.fred_api_key else None
        y2 = _fred_series(self.fred_api_key or "", "DGS2", self.lookback_years) if self.fred_api_key else None
        ffr = _fred_series(self.fred_api_key or "", "FEDFUNDS", self.lookback_years) if self.fred_api_key else None

        if y10 is None:
            try:
                # ^TNX quotes 10yr yield × 10; divide to get actual rate
                raw10 = _yf_close_weekly("^TNX", self.lookback_years)
                y10 = (raw10 / 10.0).rename("DGS10")
            except Exception as exc:
                errors.append(f"10yr yield fallback failed: {exc}")
                y10 = pd.Series(dtype=float)

        if y2 is None:
            try:
                # ^IRX is 13-week T-bill; use as short-rate proxy
                raw2 = _yf_close_weekly("^IRX", self.lookback_years)
                y2 = (raw2 / 10.0).rename("y2_proxy")
            except Exception as exc:
                errors.append(f"2yr yield fallback failed: {exc}")
                y2 = pd.Series(dtype=float)

        if ffr is None:
            # Approximate Fed Funds from ^IRX (13-week T-bill tracks FFR closely)
            ffr = y2.copy() if not y2.empty else pd.Series(dtype=float)

        # ── 2. Compute signals ────────────────────────────────────────────────
        signals["sp_trend"] = self._fe.sp_trend(spy_w)
        signals["yield_curve"] = self._fe.yield_curve(y10, y2)
        signals["vix"] = self._fe.vix_signal(vix_w)
        signals["credit_spread"] = self._fe.credit_spread(hyg_w)
        signals["liquidity"] = self._fe.liquidity(ffr)
        signals["momentum"] = self._fe.momentum(spy_w)

        score = sum(v for v in signals.values() if v != 0)
        regime = self._classify(score)

        # Determine as_of date from most recent spy data
        as_of = spy_w.index[-1].strftime("%Y-%m-%d") if not spy_w.empty else "unknown"

        return {
            "market": "US",
            "regime": regime,
            "score": score,
            "signals": signals,
            "as_of": as_of,
            "error": "; ".join(errors) if errors else None,
        }


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    fred_key = os.getenv("FRED_API_KEY")
    detector = USRegimeDetector(fred_api_key=fred_key)
    result = detector.detect()
    print(f"\nUS Market Regime as of {result['as_of']}")
    print(f"  Regime : {result['regime']}")
    print(f"  Score  : {result['score']:+d} / 6")
    print("  Signals:")
    for k, v in result["signals"].items():
        icon = "[+]" if v == 1 else "[-]" if v == -1 else "[ ]"
        print(f"    {icon}  {k}")
    if result["error"]:
        print(f"\n  Warnings: {result['error']}")
