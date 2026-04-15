"""
Market Regime Detector
======================
Classifies the current market regime using 6 signals sourced from Alpha Vantage:

  1. S&P trend       — SPY close vs. 200-week SMA
  2. Yield curve     — 10yr - 2yr Treasury spread
  3. VIX             — CBOE Volatility Index level
  4. Credit spread   — HYG (High Yield ETF) trend proxy
  5. Liquidity       — Direction of Federal Funds Rate
  6. Momentum        — S&P 500 12-month minus 1-month return

Each signal votes +1 (bullish) or -1 (bearish).
The sum (score -6 to +6) determines the regime.

Usage:
    python regime_detector.py --api-key YOUR_KEY
    python regime_detector.py --api-key YOUR_KEY --output history.csv

    # Or set env var:
    export ALPHA_VANTAGE_API_KEY=YOUR_KEY
    python regime_detector.py
"""

import os
import time
import argparse

import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass

BASE_URL = "https://www.alphavantage.co/query"


# ── Data Fetching ─────────────────────────────────────────────────────────────

class DataFetcher:
    """Fetches market and economic data from Alpha Vantage."""

    def __init__(self, api_key: str, request_delay: float = 12.0):
        """
        Args:
            api_key:       Alpha Vantage API key.
            request_delay: Seconds between requests.
                           Free tier allows 25 req/day or 5 req/min → use 12s.
        """
        self.api_key = api_key
        self.request_delay = request_delay

    def _get(self, params: dict) -> dict:
        params["apikey"] = self.api_key
        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "Information" in data:
            raise RuntimeError(f"Alpha Vantage rate limit: {data['Information']}")
        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")
        return data

    def _delay(self):
        time.sleep(self.request_delay)

    def get_weekly_prices(self, symbol: str) -> pd.DataFrame:
        """Weekly OHLCV DataFrame for a ticker symbol."""
        data = self._get({"function": "TIME_SERIES_WEEKLY", "symbol": symbol})
        self._delay()
        key = "Weekly Time Series"
        if key not in data:
            raise RuntimeError(f"No weekly data returned for {symbol}. Response: {list(data.keys())}")
        df = pd.DataFrame.from_dict(data[key], orient="index")
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def get_treasury_yield(self, maturity: str = "10year") -> pd.Series:
        """
        Monthly Treasury yield series.

        maturity options: '3month', '2year', '5year', '7year', '10year', '30year'
        """
        data = self._get({
            "function": "TREASURY_YIELD",
            "interval": "monthly",
            "maturity": maturity,
        })
        self._delay()
        if "data" not in data:
            raise RuntimeError(f"No treasury yield data for {maturity}. Response: {list(data.keys())}")
        df = pd.DataFrame(data["data"])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).set_index("date").sort_index()
        return df["value"].rename(maturity)

    def get_federal_funds_rate(self) -> pd.Series:
        """Monthly Federal Funds Rate series."""
        data = self._get({"function": "FEDERAL_FUNDS_RATE", "interval": "monthly"})
        self._delay()
        if "data" not in data:
            raise RuntimeError(f"No fed funds rate data. Response: {list(data.keys())}")
        df = pd.DataFrame(data["data"])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).set_index("date").sort_index()
        return df["value"].rename("fed_funds_rate")


# ── Signal Engineering ────────────────────────────────────────────────────────

class FeatureEngineer:
    """Computes binary (+1 / -1) regime signals from raw market data."""

    @staticmethod
    def sp_trend_signal(prices: pd.DataFrame) -> pd.Series:
        """
        S&P trend: +1 if close is above 200-week SMA, -1 otherwise.
        Captures long-term bull vs. bear market structure.
        """
        sma200 = prices["close"].rolling(200, min_periods=50).mean()
        signal = np.where(prices["close"] >= sma200, 1, -1)
        return pd.Series(signal, index=prices.index, name="sp_trend")

    @staticmethod
    def yield_curve_signal(yield_10yr: pd.Series, yield_2yr: pd.Series) -> pd.Series:
        """
        Yield curve: +1 if 10yr - 2yr spread > 0 (normal / risk-on),
                     -1 if inverted (recession warning).
        Upsampled from monthly to weekly via forward-fill.
        """
        spread = (yield_10yr - yield_2yr).resample("W-FRI").last().ffill()
        signal = np.where(spread > 0, 1, -1)
        return pd.Series(signal, index=spread.index, name="yield_curve")

    @staticmethod
    def vix_signal(vix_prices: pd.DataFrame) -> pd.Series:
        """
        VIX: +1 if VIX < 20 (low fear / risk-on),
             -1 if VIX >= 20 (elevated fear / risk-off).
        Threshold of 20 is the widely-used 'fear' boundary.
        """
        vix = vix_prices["close"]
        signal = np.where(vix < 20, 1, -1)
        return pd.Series(signal, index=vix.index, name="vix")

    @staticmethod
    def credit_spread_signal(hyg_prices: pd.DataFrame) -> pd.Series:
        """
        Credit spread proxy via HYG (iShares High Yield Corporate Bond ETF).
        +1 if HYG close > 20-week SMA (spreads tightening / risk-on),
        -1 if HYG close < 20-week SMA (spreads widening / risk-off).

        When credit spreads widen, HYG price falls — this captures that dynamic.
        """
        sma20 = hyg_prices["close"].rolling(20, min_periods=5).mean()
        signal = np.where(hyg_prices["close"] >= sma20, 1, -1)
        return pd.Series(signal, index=hyg_prices.index, name="credit_spread")

    @staticmethod
    def liquidity_signal(fed_funds: pd.Series) -> pd.Series:
        """
        Liquidity: direction of the Federal Funds Rate over ~3 months.
        +1 if rate is falling or flat (easing = looser liquidity),
        -1 if rate is rising (tightening = tighter liquidity).
        """
        weekly = fed_funds.resample("W-FRI").last().ffill()
        change_3m = weekly.diff(13)   # 13 weeks ≈ 3 months
        signal = np.where(change_3m <= 0, 1, -1)
        return pd.Series(signal, index=weekly.index, name="liquidity")

    @staticmethod
    def momentum_signal(prices: pd.DataFrame) -> pd.Series:
        """
        Momentum: standard 12-month minus 1-month time-series momentum.
        +1 if positive (trend-following bull signal),
        -1 if negative (trend-following bear signal).

        The 1-month exclusion avoids short-term mean reversion.
        """
        close = prices["close"]
        ret_12m = close.pct_change(52)   # 52 weeks = ~12 months
        ret_1m  = close.pct_change(4)    #  4 weeks = ~1 month
        momentum = ret_12m - ret_1m
        signal = np.where(momentum > 0, 1, -1)
        return pd.Series(signal, index=close.index, name="momentum")


# ── Regime Classification ─────────────────────────────────────────────────────

@dataclass
class RegimeResult:
    date: pd.Timestamp
    regime: str
    score: int
    signals: dict

    def __str__(self) -> str:
        bar = "─" * 54
        icons = {1: "[+]", -1: "[-]"}
        labels = {
            "sp_trend":      "S&P Trend       (close > 200w SMA)",
            "yield_curve":   "Yield Curve     (10yr - 2yr > 0)",
            "vix":           "VIX             (< 20, low fear)",
            "credit_spread": "Credit Spread   (HYG > 20w SMA)",
            "liquidity":     "Liquidity       (FFR falling/flat)",
            "momentum":      "Momentum        (12m - 1m > 0)",
        }
        lines = [
            bar,
            f"  Market Regime as of {self.date.date()}",
            bar,
            f"  Regime : {self.regime}",
            f"  Score  : {self.score:+d} / 6",
            bar,
            "  Signals:",
        ]
        for k, v in self.signals.items():
            label = labels.get(k, k)
            icon = icons.get(v, "[ ]")
            lines.append(f"    {icon}  {label}")
        lines.append(bar)
        return "\n".join(lines)


class RegimeDetector:
    """
    Aggregates binary signals into a regime score and classifies the market.

    Score ranges (sum of 6 signals, each +1 or -1):
        +4 to +6  : Risk-On    (Bull Market)
         0 to +3  : Cautious   (Late Cycle / Mixed)
        -3 to -1  : Defensive  (Early Bear)
        -6 to -4  : Risk-Off   (Bear Market)
    """

    @staticmethod
    def classify(score: int) -> str:
        if score >= 4:
            return "Risk-On    (Bull Market)"
        elif score >= 0:
            return "Cautious   (Late Cycle / Mixed)"
        elif score >= -3:
            return "Defensive  (Early Bear)"
        else:
            return "Risk-Off   (Bear Market)"

    def detect(
        self,
        sp_trend:      pd.Series,
        yield_curve:   pd.Series,
        vix:           pd.Series,
        credit_spread: pd.Series,
        liquidity:     pd.Series,
        momentum:      pd.Series,
    ) -> pd.DataFrame:
        """
        Aligns all signals to a weekly (Friday) index, sums them into a score,
        and assigns a regime label. Returns a full-history DataFrame.
        """
        signal_map = {
            "sp_trend":      sp_trend,
            "yield_curve":   yield_curve,
            "vix":           vix,
            "credit_spread": credit_spread,
            "liquidity":     liquidity,
            "momentum":      momentum,
        }
        # Align on weekly Friday index; forward-fill gaps (e.g. monthly data)
        df = pd.DataFrame(signal_map)
        df = df.resample("W-FRI").last().ffill().dropna()
        df = df.astype(int)
        df["score"] = df[list(signal_map.keys())].sum(axis=1)
        df["regime"] = df["score"].apply(self.classify)
        return df

    def latest(self, df: pd.DataFrame) -> RegimeResult:
        """Returns a RegimeResult snapshot for the most recent date."""
        row = df.iloc[-1]
        signal_cols = ["sp_trend", "yield_curve", "vix", "credit_spread", "liquidity", "momentum"]
        return RegimeResult(
            date=df.index[-1],
            regime=row["regime"],
            score=int(row["score"]),
            signals={c: int(row[c]) for c in signal_cols},
        )


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Market Regime Detector — powered by Alpha Vantage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ALPHA_VANTAGE_API_KEY", "demo"),
        help="Alpha Vantage API key (or set ALPHA_VANTAGE_API_KEY env var). "
             "Default: 'demo' (very limited).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE.csv",
        help="Optional path to save the full weekly regime history as CSV.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=12.0,
        help="Seconds between API requests (default: 12.0 for free-tier safety).",
    )
    args = parser.parse_args()

    if args.api_key == "demo":
        print("WARNING: Using demo API key. Data will be limited/placeholder.")
        print("         Set --api-key or ALPHA_VANTAGE_API_KEY for real results.\n")

    fetcher  = DataFetcher(api_key=args.api_key, request_delay=args.delay)
    fe       = FeatureEngineer()
    detector = RegimeDetector()

    steps = [
        ("S&P 500 (SPY) weekly prices",   lambda: fetcher.get_weekly_prices("SPY")),
        ("10-year Treasury yield",         lambda: fetcher.get_treasury_yield("10year")),
        ("2-year Treasury yield",          lambda: fetcher.get_treasury_yield("2year")),
        ("VIX weekly prices",              lambda: fetcher.get_weekly_prices("VIX")),
        ("HYG (High Yield ETF) weekly",    lambda: fetcher.get_weekly_prices("HYG")),
        ("Federal Funds Rate",             lambda: fetcher.get_federal_funds_rate()),
    ]

    results = {}
    keys    = ["spy", "y10", "y2", "vix_raw", "hyg", "ffr"]

    print("Fetching data from Alpha Vantage...")
    for (label, fn), key in zip(steps, keys):
        print(f"  [{keys.index(key)+1}/{len(keys)}] {label}...")
        results[key] = fn()

    print("\nComputing signals...")
    sig_sp  = fe.sp_trend_signal(results["spy"])
    sig_yc  = fe.yield_curve_signal(results["y10"], results["y2"])
    sig_vix = fe.vix_signal(results["vix_raw"])
    sig_cs  = fe.credit_spread_signal(results["hyg"])
    sig_liq = fe.liquidity_signal(results["ffr"])
    sig_mom = fe.momentum_signal(results["spy"])

    print("Detecting regime...\n")
    regime_df = detector.detect(sig_sp, sig_yc, sig_vix, sig_cs, sig_liq, sig_mom)
    result    = detector.latest(regime_df)

    print(result)

    if args.output:
        regime_df.to_csv(args.output)
        print(f"\nFull history saved to: {args.output}")


if __name__ == "__main__":
    main()
