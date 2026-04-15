"""
Market Timing & Valuation Module
==================================
Detects whether the US and Korean markets are cheap, fairly valued, or expensive
using multiple valuation lenses.  The output is a timing signal that can gate or
weight the signals produced by the regime detectors and analysts.

Valuation lenses implemented:
  US:
    1. S&P 500 earnings yield vs. 10yr Treasury (Fed Model)
       - Earnings yield = 1 / trailing P/E of SPY
       - "Cheap" if earnings yield > 10yr yield by > 2pp
    2. Buffett Indicator proxy
       - Wilshire 5000 Total Market cap / US Nominal GDP (FRED)
       - "Expensive" if ratio > 1.5, "Cheap" if < 0.8
    3. SPY trailing P/E (yfinance info)
       - "Cheap" < 15, "Fair" 15–25, "Expensive" > 25

  KR:
    1. KOSPI trailing P/E (from yfinance ^KS11 info)
       - "Cheap" < 10, "Fair" 10–15, "Expensive" > 15  (KOSPI trades at a discount)
    2. KOSPI earnings yield vs. KR 10yr bond (FRED)

Output per market:
    {
        "valuation_signal": -1 | 0 | 1,   # -1=expensive, 0=neutral, +1=cheap
        "pe_ratio": float | None,
        "fed_model_spread_pp": float | None,  # earnings yield minus bond yield (pp)
        "buffett_indicator": float | None,     # market cap / GDP (US only)
        "label": str,                          # "Cheap" | "Fair" | "Expensive" | "N/A"
        "notes": list[str],
    }
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

try:
    from fredapi import Fred
    _FRED_AVAILABLE = True
except ImportError:
    _FRED_AVAILABLE = False
    logger.warning("fredapi not installed — Buffett Indicator and some yield comparisons will be skipped")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fred_latest(fred_key: str, series_id: str) -> Optional[float]:
    """Return the most recent value for a FRED series, or None on failure."""
    if not _FRED_AVAILABLE or not fred_key:
        return None
    try:
        fred = Fred(api_key=fred_key)
        start = (datetime.today() - timedelta(days=180)).strftime("%Y-%m-%d")
        s = fred.get_series(series_id, observation_start=start).dropna()
        return float(s.iloc[-1]) if not s.empty else None
    except Exception as exc:
        logger.warning(f"FRED fetch failed ({series_id}): {exc}")
        return None


def _yf_pe(ticker: str) -> Optional[float]:
    """Attempt to read trailingPE from yfinance ticker info."""
    try:
        info = yf.Ticker(ticker).info
        pe = info.get("trailingPE") or info.get("forwardPE")
        return float(pe) if pe and pe > 0 else None
    except Exception as exc:
        logger.warning(f"yfinance PE fetch failed for {ticker}: {exc}")
        return None


def _yf_close_latest(ticker: str) -> Optional[float]:
    """Return the latest close price from yfinance."""
    try:
        df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].dropna().iloc[-1])
    except Exception as exc:
        logger.warning(f"yfinance close fetch failed for {ticker}: {exc}")
        return None


# ── US Valuation ──────────────────────────────────────────────────────────────

class USValuation:
    """Computes valuation signals for the US equity market."""

    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred_key = fred_api_key or ""

    def compute(self) -> Dict:
        notes: List[str] = []
        signals: List[int] = []

        # --- Lens 1: SPY trailing P/E ---
        pe = _yf_pe("SPY")
        earnings_yield: Optional[float] = None
        if pe is not None:
            earnings_yield = 100.0 / pe          # as percentage
            if pe < 15:
                notes.append(f"SPY P/E {pe:.1f} — historically cheap")
                signals.append(1)
            elif pe > 25:
                notes.append(f"SPY P/E {pe:.1f} — elevated")
                signals.append(-1)
            else:
                notes.append(f"SPY P/E {pe:.1f} — fair value range")
                signals.append(0)
        else:
            notes.append("SPY P/E unavailable")

        # --- Lens 2: Fed Model (earnings yield vs. 10yr Treasury) ---
        y10 = None
        if self.fred_key:
            raw = _fred_latest(self.fred_key, "DGS10")
            y10 = raw  # already in percentage points (e.g., 4.5)
        if y10 is None:
            # Fallback: ^TNX from yfinance (quotes yield × 10)
            raw_tnx = _yf_close_latest("^TNX")
            y10 = raw_tnx / 10.0 if raw_tnx else None

        fed_model_spread: Optional[float] = None
        if earnings_yield is not None and y10 is not None:
            fed_model_spread = earnings_yield - y10
            if fed_model_spread > 2.0:
                notes.append(f"Fed Model: stocks cheap vs bonds (+{fed_model_spread:.1f}pp)")
                signals.append(1)
            elif fed_model_spread < -1.0:
                notes.append(f"Fed Model: bonds more attractive than stocks ({fed_model_spread:.1f}pp)")
                signals.append(-1)
            else:
                notes.append(f"Fed Model: near fair value ({fed_model_spread:.1f}pp)")
                signals.append(0)
        else:
            notes.append("Fed Model: insufficient data")

        # --- Lens 3: Buffett Indicator (Wilshire 5000 / GDP) ---
        buffett: Optional[float] = None
        # FRED: WILL5000IND = Wilshire 5000 Total Market Index (price level)
        # FRED: GDP = US Nominal GDP (quarterly, billions)
        # Approximate ratio: WILL5000 level represents ~$1 per point per index unit
        # The FRED series DDDM01USA156NWDB gives Market Cap / GDP directly but less frequent
        # We use WILL5000INDFC (full cap) and GDP
        wilshire = _fred_latest(self.fred_key, "WILL5000INDFC") if self.fred_key else None
        gdp_bn = _fred_latest(self.fred_key, "GDP") if self.fred_key else None
        if wilshire is not None and gdp_bn is not None:
            # Wilshire 5000 is expressed as an index; 1 point ≈ $1 billion market cap
            # GDP is in billions USD
            buffett = wilshire / gdp_bn
            if buffett > 1.5:
                notes.append(f"Buffett Indicator: {buffett:.2f} — overvalued")
                signals.append(-1)
            elif buffett < 0.8:
                notes.append(f"Buffett Indicator: {buffett:.2f} — undervalued")
                signals.append(1)
            else:
                notes.append(f"Buffett Indicator: {buffett:.2f} — fair")
                signals.append(0)
        else:
            notes.append("Buffett Indicator: FRED data unavailable")

        # Aggregate signal: majority vote
        if signals:
            vote = sum(signals)
            if vote > 0:
                valuation_signal, label = 1, "Cheap"
            elif vote < 0:
                valuation_signal, label = -1, "Expensive"
            else:
                valuation_signal, label = 0, "Fair"
        else:
            valuation_signal, label = 0, "N/A"

        return {
            "market": "US",
            "valuation_signal": valuation_signal,
            "pe_ratio": pe,
            "earnings_yield_pct": round(earnings_yield, 2) if earnings_yield else None,
            "ten_yr_yield_pct": round(y10, 2) if y10 else None,
            "fed_model_spread_pp": round(fed_model_spread, 2) if fed_model_spread is not None else None,
            "buffett_indicator": round(buffett, 3) if buffett else None,
            "label": label,
            "notes": notes,
        }


# ── KR Valuation ──────────────────────────────────────────────────────────────

class KRValuation:
    """Computes valuation signals for the Korean equity market."""

    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred_key = fred_api_key or ""

    def compute(self) -> Dict:
        notes: List[str] = []
        signals: List[int] = []

        # --- Lens 1: KOSPI trailing P/E ---
        pe = _yf_pe("^KS11")
        earnings_yield: Optional[float] = None
        if pe is not None:
            earnings_yield = 100.0 / pe
            # KOSPI historically trades at lower P/E than S&P 500 ("Korea discount")
            if pe < 10:
                notes.append(f"KOSPI P/E {pe:.1f} — historically cheap (Korea discount)")
                signals.append(1)
            elif pe > 16:
                notes.append(f"KOSPI P/E {pe:.1f} — elevated for Korean market")
                signals.append(-1)
            else:
                notes.append(f"KOSPI P/E {pe:.1f} — within normal range")
                signals.append(0)
        else:
            notes.append("KOSPI P/E unavailable via yfinance")

        # --- Lens 2: KR earnings yield vs. KR 10yr bond ---
        kr10yr = _fred_latest(self.fred_key, "IRLTLT01KRM156N") if self.fred_key else None
        fed_model_spread: Optional[float] = None
        if earnings_yield is not None and kr10yr is not None:
            fed_model_spread = earnings_yield - kr10yr
            if fed_model_spread > 2.0:
                notes.append(f"KR Fed Model: equities cheap vs bonds (+{fed_model_spread:.1f}pp)")
                signals.append(1)
            elif fed_model_spread < -1.0:
                notes.append(f"KR Fed Model: bonds more attractive ({fed_model_spread:.1f}pp)")
                signals.append(-1)
            else:
                notes.append(f"KR Fed Model: near fair value ({fed_model_spread:.1f}pp)")
                signals.append(0)
        else:
            notes.append("KR Fed Model: insufficient data (FRED IRLTLT01KRM156N needed)")

        if signals:
            vote = sum(signals)
            valuation_signal = 1 if vote > 0 else (-1 if vote < 0 else 0)
            label = "Cheap" if valuation_signal == 1 else ("Expensive" if valuation_signal == -1 else "Fair")
        else:
            valuation_signal, label = 0, "N/A"

        return {
            "market": "KR",
            "valuation_signal": valuation_signal,
            "pe_ratio": pe,
            "earnings_yield_pct": round(earnings_yield, 2) if earnings_yield else None,
            "kr10yr_yield_pct": round(kr10yr, 2) if kr10yr else None,
            "fed_model_spread_pp": round(fed_model_spread, 2) if fed_model_spread is not None else None,
            "label": label,
            "notes": notes,
        }


# ── Main Entry ────────────────────────────────────────────────────────────────

class MarketTimer:
    """
    Aggregates US and KR valuation signals into a single market timing result.

    Usage:
        timer = MarketTimer(fred_api_key="...")
        result = timer.run()
        # result = {"us": {...}, "kr": {...}, "timestamp": "..."}
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        self._us = USValuation(fred_api_key)
        self._kr = KRValuation(fred_api_key)

    def run(self) -> Dict:
        us_result = self._us.compute()
        kr_result = self._kr.compute()
        return {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "us": us_result,
            "kr": kr_result,
        }


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    fred_key = os.getenv("FRED_API_KEY")
    timer = MarketTimer(fred_api_key=fred_key)
    result = timer.run()
    for mkt in ["us", "kr"]:
        d = result[mkt]
        print(f"\n{'='*40}")
        print(f"  {d['market']} Valuation: {d['label']}  (signal={d['valuation_signal']:+d})")
        print(f"  P/E ratio : {d.get('pe_ratio')}")
        for note in d["notes"]:
            print(f"    • {note}")
