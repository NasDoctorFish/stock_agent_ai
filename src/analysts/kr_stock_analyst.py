"""
KR Stock Data Analyst
=====================
Analyzes Korean stocks listed in src/config/kr_tickers.yaml using
live yfinance data (^KS11 for the market index, individual .KS tickers
for individual stock metrics).

Generates:
  - Per-stock technical signals: momentum, volatility, drawdown
  - Korean market regime based on KOSPI price dynamics
  - Top-ranked Korean stocks by technical score

Broker routing: Kiwoom API (KRX stocks only).
This analyst does NOT place orders — it only produces signals.
The scheduler/decision layer uses the output to trigger KiwoomBroker.

Inherits BaseAnalyst for consistency with the existing pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from analysts.base_analyst import BaseAnalyst, AnalystResult

logger = logging.getLogger(__name__)

# ── Config loading ─────────────────────────────────────────────────────────────

def _load_kr_tickers() -> List[Dict[str, str]]:
    """Load Korean ticker list from src/config/kr_tickers.yaml."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "kr_tickers.yaml"
    if not config_path.exists():
        logger.warning(f"kr_tickers.yaml not found at {config_path}")
        return []
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("kr_tickers", [])


# ── Data fetch ─────────────────────────────────────────────────────────────────

def _fetch_daily_prices(yf_tickers: List[str], days: int = 365) -> pd.DataFrame:
    """
    Download daily close prices for a list of yfinance tickers.
    Returns a DataFrame with columns = [date, close, stock_id].
    """
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    if not yf_tickers:
        return pd.DataFrame(columns=["date", "close", "stock_id"])

    raw = yf.download(
        yf_tickers,
        start=start,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if raw.empty:
        return pd.DataFrame(columns=["date", "close", "stock_id"])

    rows = []
    for ticker in yf_tickers:
        try:
            if len(yf_tickers) == 1:
                # Single ticker: columns are flat
                if isinstance(raw.columns, pd.MultiIndex):
                    col_df = raw[ticker] if ticker in raw.columns.get_level_values(0) else raw
                else:
                    col_df = raw
                close_s = col_df["Close"] if "Close" in col_df.columns else col_df["close"]
            else:
                close_s = raw[ticker]["Close"]
        except (KeyError, TypeError):
            logger.warning(f"No close data for {ticker}")
            continue

        close_s = close_s.dropna()
        if close_s.empty:
            continue

        for ts, price in close_s.items():
            rows.append({"date": pd.Timestamp(ts).normalize(), "close": float(price), "stock_id": ticker})

    return pd.DataFrame(rows)


# ── Analyst ────────────────────────────────────────────────────────────────────

class KRStockDataAnalyst(BaseAnalyst):
    """
    Technical analysis analyst for Korean stocks.

    Attributes:
        name:            "kr_stock_data" — key in the aggregated context dict
        lookback_days:   How many calendar days of price history to fetch (default 365)
        window:          Technical signal window in trading days (default 60)
        min_obs:         Minimum observations required per stock (default 20)
    """

    name = "kr_stock_data"

    def __init__(
        self,
        db=None,
        logger_=None,
        lookback_days: int = 365,
        window: int = 60,
        min_obs: int = 20,
    ):
        # db is unused for KR (we fetch live from yfinance), but kept for interface compatibility
        super().__init__(db=db, logger=logger_)
        self.lookback_days = lookback_days
        self.window = window
        self.min_obs = min_obs
        self._tickers_cfg = _load_kr_tickers()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _yf_tickers(self) -> List[str]:
        return [t["yf_ticker"] for t in self._tickers_cfg if "yf_ticker" in t]

    def _kiwoom_map(self) -> Dict[str, str]:
        """Maps yf_ticker → kiwoom_code."""
        return {t["yf_ticker"]: t["kiwoom_code"] for t in self._tickers_cfg if "kiwoom_code" in t}

    @staticmethod
    def _compute_stock_metrics(
        df: pd.DataFrame, window: int, min_obs: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        For each stock_id in df, compute momentum, volatility, and drawdown
        over the trailing `window` trading days.
        """
        df = df.sort_values(["stock_id", "date"])
        df["log_return"] = df.groupby("stock_id")["close"].transform(
            lambda s: np.log(s).diff()
        )

        stocks: Dict[str, Dict[str, Any]] = {}
        for stock_id, grp in df.groupby("stock_id"):
            lr = grp["log_return"].dropna()
            if len(lr) < min_obs:
                continue
            recent = lr.tail(window)
            cum = recent.cumsum()
            momentum = float(np.exp(cum.iloc[-1]) - 1) if len(cum) > 0 else None
            volatility = float(recent.std()) if len(recent) > 1 else None
            drawdown = float((cum - cum.cummax()).min()) if len(cum) > 1 else None
            stocks[stock_id] = {
                "momentum": momentum,
                "volatility": volatility,
                "drawdown": drawdown,
                "obs": int(len(recent)),
            }
        return stocks

    @staticmethod
    def _detect_regime(
        df: pd.DataFrame, window: int, min_obs: int
    ) -> tuple[str, Dict[str, float]]:
        """
        Detect Korean market regime from KOSPI average log returns.
        Uses the same 4-state logic as StockDataAnalyst for consistency.
        """
        market_lr = df.groupby("date")["log_return"].mean().dropna()
        if len(market_lr) < min_obs:
            return "unknown", {}

        recent = market_lr.tail(window)
        mean_lr = float(recent.mean())
        vol = float(recent.std()) if len(recent) > 1 else 0.0
        long_vol = float(market_lr.std()) if len(market_lr) > 1 else vol
        vol_cluster = float(recent.abs().autocorr(lag=1)) if len(recent) > 2 else 0.0

        if mean_lr > 0 and vol <= 1.2 * (long_vol or vol):
            regime = "growth"
        elif mean_lr < 0:
            regime = "drawdown"
        elif vol_cluster > 0.3 and vol > 1.2 * (long_vol or vol):
            regime = "volatility_clustering"
        else:
            regime = "neutral"

        metrics = {
            "mean_log_return": mean_lr,
            "volatility": vol,
            "long_run_volatility": long_vol,
            "vol_cluster": vol_cluster,
        }
        return regime, metrics

    # ── Main analyze ───────────────────────────────────────────────────────────

    def analyze(self) -> AnalystResult:
        yf_tickers = self._yf_tickers()
        if not yf_tickers:
            return self._result(False, errors=["No Korean tickers configured in kr_tickers.yaml"])

        # ── Fetch price data ──────────────────────────────────────────────────
        try:
            df = _fetch_daily_prices(yf_tickers, days=self.lookback_days)
        except Exception as exc:
            return self._result(False, errors=[f"yfinance fetch failed: {exc}"])

        if df.empty:
            return self._result(False, summary="No Korean price data returned from yfinance")

        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["date", "close", "stock_id"])

        # Add log returns (needed for regime detection)
        df = df.sort_values(["stock_id", "date"])
        df["log_return"] = df.groupby("stock_id")["close"].transform(
            lambda s: np.log(s).diff()
        )

        # ── Per-stock metrics ─────────────────────────────────────────────────
        stocks = self._compute_stock_metrics(df, self.window, self.min_obs)

        # ── Market regime ─────────────────────────────────────────────────────
        economic_regime, regime_metrics = self._detect_regime(df, self.window, self.min_obs)

        # ── Score and rank stocks ─────────────────────────────────────────────
        kiwoom_map = self._kiwoom_map()
        scored = []
        for stock_id, metrics in stocks.items():
            mom = metrics["momentum"] or 0.0
            vol = metrics["volatility"] or 0.0
            dd = metrics["drawdown"] or 0.0
            score = mom - 0.5 * vol + 0.1 * dd
            scored.append({
                "stock_id": stock_id,
                "kiwoom_code": kiwoom_map.get(stock_id, ""),
                "score": score,
                **metrics,
            })

        top_stocks = sorted(scored, key=lambda x: x["score"], reverse=True)[:10]

        return self._result(
            True,
            data={
                "market": "KR",
                "economic_regime": economic_regime,
                "regime_metrics": regime_metrics,
                "stocks": stocks,
                "top_stocks": top_stocks,
                "kiwoom_map": kiwoom_map,
            },
            summary=f"KR stock analysis complete — {len(stocks)} stocks, regime={economic_regime}",
        )


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyst = KRStockDataAnalyst()
    result = analyst.run()
    if result.success:
        print(f"Regime: {result.data['economic_regime']}")
        print("Top Korean stocks:")
        for s in result.data["top_stocks"][:5]:
            print(f"  {s['stock_id']} ({s['kiwoom_code']}): score={s['score']:.4f}")
    else:
        print("Analysis failed:", result.errors)
