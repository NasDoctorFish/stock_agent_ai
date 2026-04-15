"""
Korean Market Regime Detector
==============================
Classifies the current Korean market regime using 6 signals sourced from
yfinance and FRED (fredapi).

Signals:
  1. KOSPI trend       — ^KS11 close vs. 200-week SMA
  2. KR yield curve    — KR 10yr yield minus short-term rate
                         (FRED: IRLTLT01KRM156N / IRSTCI01KRM156N)
  3. VKOSPI            — Korean VIX equivalent (^VKOSPI); falls back to
                         absolute KOSPI 20-day volatility rank if unavailable
  4. KRW strength      — KRW/USD direction (USD/KRW falling = KRW strengthening = risk-on)
  5. BOK rate          — Bank of Korea base rate direction (FRED: IRSTCI01KRM156N)
  6. KOSPI momentum    — 12-month minus 1-month KOSPI return

Regime score (-6 to +6):
  +4 to +6  : Risk-On    (KR Bull Market)
   0 to +3  : Cautious   (Late Cycle / Mixed)
  -3 to -1  : Defensive  (Early Bear)
  -6 to -4  : Risk-Off   (KR Bear Market)

Usage:
    from regime_detector.kr_regime import KRRegimeDetector
    detector = KRRegimeDetector(fred_api_key="...")
    result = detector.detect()
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

try:
    from fredapi import Fred
    _FRED_AVAILABLE = True
except ImportError:
    _FRED_AVAILABLE = False
    logger.warning("fredapi not installed — Korean yield curve and BOK rate signals will be skipped")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _yf_weekly_close(ticker: str, years: int = 5) -> pd.Series:
    """Download weekly close prices from yfinance."""
    start = (datetime.today() - timedelta(weeks=years * 52)).strftime("%Y-%m-%d")
    raw = yf.download(ticker, start=start, interval="1wk", auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index)
    return raw["close"].dropna().sort_index()


def _fred_monthly(fred_key: str, series_id: str, years: int = 5) -> Optional[pd.Series]:
    """Fetch a FRED series; returns None on any failure."""
    if not _FRED_AVAILABLE or not fred_key:
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


# ── Signal Engineering ────────────────────────────────────────────────────────

class KRFeatureEngineer:
    """Computes binary (+1 / -1) regime signals specific to the Korean market."""

    @staticmethod
    def kospi_trend(kospi_weekly: pd.Series) -> int:
        """KOSPI trend: +1 if above 200-week SMA, else -1."""
        if len(kospi_weekly) < 50:
            return 0
        sma200 = kospi_weekly.rolling(200, min_periods=50).mean()
        return 1 if kospi_weekly.iloc[-1] >= sma200.iloc[-1] else -1

    @staticmethod
    def yield_curve(kr10yr: pd.Series, kr_short: pd.Series) -> int:
        """
        Korean yield curve: +1 if 10yr > short-term rate (normal),
        -1 if inverted.  Returns 0 if either series is empty.
        """
        if kr10yr.empty or kr_short.empty:
            return 0
        try:
            combined = pd.concat(
                [kr10yr.rename("y10"), kr_short.rename("ys")], axis=1
            ).ffill().dropna()
            if combined.empty:
                return 0
            spread = combined["y10"].iloc[-1] - combined["ys"].iloc[-1]
            return 1 if spread > 0 else -1
        except Exception:
            return 0

    @staticmethod
    def vkospi_signal(vkospi_weekly: Optional[pd.Series], kospi_weekly: pd.Series) -> int:
        """
        Volatility signal:
        - Primary: ^VKOSPI < 20 → +1 (low fear), ≥ 20 → -1
        - Fallback (if VKOSPI unavailable): KOSPI realized 20-day volatility
          annualized; below historical median → +1, above → -1
        """
        if vkospi_weekly is not None and not vkospi_weekly.empty:
            return 1 if vkospi_weekly.iloc[-1] < 20 else -1
        # Fallback: daily volatility rank
        if len(kospi_weekly) < 20:
            return 0
        log_ret = np.log(kospi_weekly).diff().dropna()
        recent_vol = log_ret.tail(4).std() * np.sqrt(52)    # annualized, weekly data
        hist_median = log_ret.std() * np.sqrt(52)
        return 1 if recent_vol < hist_median else -1

    @staticmethod
    def krw_strength(usdkrw_weekly: Optional[pd.Series]) -> int:
        """
        KRW strength signal: measures USD/KRW trend.
        USD/KRW falling (KRW strengthening) → risk-on (+1).
        USD/KRW rising (KRW weakening) → risk-off (-1).
        Uses 13-week (≈3-month) moving average crossover.
        """
        if usdkrw_weekly is None or len(usdkrw_weekly) < 14:
            return 0
        sma13 = usdkrw_weekly.rolling(13, min_periods=4).mean()
        # KRW/USD means: lower USD/KRW = stronger KRW
        latest = usdkrw_weekly.iloc[-1]
        ma = sma13.iloc[-1]
        # If USD/KRW is BELOW its own 13-week SMA → KRW is strengthening → +1
        return 1 if latest < ma else -1

    @staticmethod
    def bok_rate(bok_rate_series: Optional[pd.Series]) -> int:
        """
        BOK rate direction: +1 if rate falling/flat (easing), -1 if rising (tightening).
        Uses change over last ~3 months of monthly data.
        """
        if bok_rate_series is None or len(bok_rate_series) < 4:
            return 0
        change = bok_rate_series.iloc[-1] - bok_rate_series.iloc[-4]
        return 1 if change <= 0 else -1

    @staticmethod
    def momentum(kospi_weekly: pd.Series) -> int:
        """KOSPI momentum: 12m minus 1m return. +1 positive, -1 negative."""
        if len(kospi_weekly) < 53:
            return 0
        ret_12m = kospi_weekly.iloc[-1] / kospi_weekly.iloc[-53] - 1
        ret_1m = kospi_weekly.iloc[-1] / kospi_weekly.iloc[-5] - 1
        return 1 if (ret_12m - ret_1m) > 0 else -1


# ── Detector ──────────────────────────────────────────────────────────────────

class KRRegimeDetector:
    """
    Fetches Korean market data and computes a 6-signal regime score.

    Args:
        fred_api_key: FRED API key for Korean yield curve and BOK rate.
                      Without this, those two signals default to 0.
        lookback_years: History window for signal computation (default 5).
    """

    def __init__(self, fred_api_key: Optional[str] = None, lookback_years: int = 5):
        self.fred_api_key = fred_api_key
        self.lookback_years = lookback_years
        self._fe = KRFeatureEngineer()

    @staticmethod
    def _classify(score: int) -> str:
        if score >= 4:
            return "Risk-On (KR Bull Market)"
        elif score >= 0:
            return "Cautious (Late Cycle / Mixed)"
        elif score >= -3:
            return "Defensive (Early KR Bear)"
        return "Risk-Off (KR Bear Market)"

    def detect(self) -> Dict:
        """
        Run the 6-signal Korean regime detection.

        Returns a dict:
            {
                "market":  "KR",
                "regime":  str,
                "score":   int,       # -6 to +6 (0-weighted signals if data missing)
                "signals": dict,
                "as_of":   str,
                "error":   str | None,
            }
        """
        signals: Dict[str, int] = {}
        errors = []

        # ── 1. KOSPI weekly ───────────────────────────────────────────────────
        try:
            kospi_w = _yf_weekly_close("^KS11", self.lookback_years)
        except Exception as exc:
            errors.append(f"KOSPI fetch failed: {exc}")
            kospi_w = pd.Series(dtype=float)

        # ── 2. VKOSPI ─────────────────────────────────────────────────────────
        vkospi_w = None
        try:
            vkospi_w = _yf_weekly_close("^VKOSPI", self.lookback_years)
        except Exception as exc:
            logger.info(f"VKOSPI unavailable ({exc}); using volatility fallback")

        # ── 3. USD/KRW ────────────────────────────────────────────────────────
        usdkrw_w = None
        try:
            usdkrw_w = _yf_weekly_close("KRW=X", self.lookback_years)
        except Exception as exc:
            errors.append(f"KRW=X fetch failed: {exc}")

        # ── 4. Korean yields (FRED) ───────────────────────────────────────────
        # IRLTLT01KRM156N: KR 10yr government bond yield (monthly)
        # IRSTCI01KRM156N: KR immediate/call rate as BOK policy rate proxy (monthly)
        kr10yr = _fred_monthly(self.fred_api_key or "", "IRLTLT01KRM156N", self.lookback_years)
        kr_short = _fred_monthly(self.fred_api_key or "", "IRSTCI01KRM156N", self.lookback_years)

        if kr10yr is None:
            errors.append("KR 10yr yield (FRED IRLTLT01KRM156N) unavailable; yield curve signal = 0")
        if kr_short is None:
            errors.append("KR short rate (FRED IRSTCI01KRM156N) unavailable; BOK/yield signals = 0")

        # ── 5. Compute signals ────────────────────────────────────────────────
        signals["kospi_trend"] = self._fe.kospi_trend(kospi_w)
        signals["yield_curve"] = (
            self._fe.yield_curve(kr10yr, kr_short)
            if (kr10yr is not None and kr_short is not None) else 0
        )
        signals["vkospi"] = self._fe.vkospi_signal(vkospi_w, kospi_w)
        signals["krw_strength"] = self._fe.krw_strength(usdkrw_w)
        signals["bok_rate"] = (
            self._fe.bok_rate(kr_short) if kr_short is not None else 0
        )
        signals["momentum"] = self._fe.momentum(kospi_w)

        score = sum(v for v in signals.values())
        regime = self._classify(score)
        as_of = kospi_w.index[-1].strftime("%Y-%m-%d") if not kospi_w.empty else "unknown"

        return {
            "market": "KR",
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
    detector = KRRegimeDetector(fred_api_key=fred_key)
    result = detector.detect()
    print(f"\nKorean Market Regime as of {result['as_of']}")
    print(f"  Regime : {result['regime']}")
    print(f"  Score  : {result['score']:+d} / 6")
    print("  Signals:")
    for k, v in result["signals"].items():
        icon = "[+]" if v == 1 else "[-]" if v == -1 else "[ ]"
        print(f"    {icon}  {k}")
    if result["error"]:
        print(f"\n  Warnings: {result['error']}")
