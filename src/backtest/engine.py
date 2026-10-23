"""
BacktestEngine
--------------
Runs a historical simulation of the Stock_ML investment strategy.

Key design principles:
  • No look-ahead bias: all DuckDB queries use `date < rebalance_date`
  • No live API calls: operates solely on data already in DuckDB
  • Replicates DecisionMaker scoring logic inline using SQL aggregations
  • Regime detection uses macro data from `macro_index_full`
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

from backtest.portfolio import PortfolioSimulator

logger = logging.getLogger(__name__)


# ─── Result Container ─────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Container for all outputs produced by BacktestEngine.run()."""

    nav_series: pd.Series                   # daily NAV (DatetimeIndex)
    benchmark_series: pd.Series             # benchmark NAV normalised to initial_capital
    trade_log: List[dict]                   # every executed trade
    regime_history: pd.Series              # {rebalance_date: regime_label}
    rebalance_dates: List[pd.Timestamp]    # dates on which rebalancing occurred
    start_date: str
    end_date: str
    initial_capital: float
    benchmark_ticker: str


# ─── Engine ───────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Runs a walk-forward portfolio simulation over historical DuckDB data.

    Parameters
    ----------
    db_path : str
        Absolute or relative path to stockml.duckdb.
    start_date : str
        Backtest start (inclusive), e.g. '2022-01-01'.
    end_date : str
        Backtest end (inclusive), e.g. '2025-01-01'.
    rebalance_freq : str
        'weekly' or 'monthly'.
    top_n : int
        Number of stocks to hold in equity portion.
    initial_capital : float
        Starting portfolio value.
    benchmark_ticker : str
        Ticker to use as benchmark (must exist in price_daily_full).
    slippage_pct : float
        One-way slippage fraction (default 0.001 = 10 bps).
    commission_pct : float
        Commission fraction of gross trade value (default 0.00015 = 1.5 bps).
    """

    # Regime score thresholds (matches RegimeDetector)
    _REGIME_LABELS = {
        "risk_on":   "Risk-On",
        "cautious":  "Cautious",
        "defensive": "Defensive",
        "risk_off":  "Risk-Off",
    }

    def __init__(
        self,
        db_path: str,
        start_date: str,
        end_date: str,
        rebalance_freq: str = "monthly",
        top_n: int = 10,
        initial_capital: float = 100_000_000.0,
        benchmark_ticker: str = "SPY",
        slippage_pct: float = 0.001,
        commission_pct: float = 0.00015,
    ) -> None:
        self.db_path = db_path
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.rebalance_freq = rebalance_freq
        self.top_n = top_n
        self.initial_capital = initial_capital
        self.benchmark_ticker = benchmark_ticker
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct

        # Open DuckDB in read_only mode to avoid conflicts with live process
        self._conn = duckdb.connect(db_path, read_only=True)
        logger.info(f"BacktestEngine connected to {db_path}")

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass

    # ─── Public ───────────────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        """Execute the backtest and return a BacktestResult."""
        logger.info(
            f"Starting backtest: {self.start_date.date()} → {self.end_date.date()}, "
            f"freq={self.rebalance_freq}, top_n={self.top_n}"
        )

        rebalance_dates = self._build_rebalance_dates()
        logger.info(f"Total rebalance dates: {len(rebalance_dates)}")

        portfolio = PortfolioSimulator(
            initial_capital=self.initial_capital,
            slippage_pct=self.slippage_pct,
            commission_pct=self.commission_pct,
        )

        regime_history: Dict[pd.Timestamp, str] = {}

        # Fetch all daily prices once for NAV recording and benchmark
        all_prices = self._fetch_all_prices()
        benchmark_raw = self._fetch_benchmark_prices()

        # ── Walk-forward loop ──────────────────────────────────────────────
        for i, rebal_date in enumerate(rebalance_dates):
            date_str = rebal_date.strftime("%Y-%m-%d")
            logger.debug(f"Rebalancing [{i+1}/{len(rebalance_dates)}] on {date_str}")

            # ① Compute regime (no look-ahead)
            regime_label, regime_score = self._compute_regime(rebal_date)
            regime_history[rebal_date] = regime_label
            equity_fraction = self._equity_fraction(regime_score)

            # ② Score stocks (no look-ahead)
            scored_stocks = self._score_stocks(rebal_date)

            # ③ Determine target weights
            target_weights = self._build_target_weights(
                scored_stocks, equity_fraction, rebal_date
            )

            # ④ Get current prices for rebalancing
            prices_on_date = self._prices_on_date(all_prices, rebal_date)

            if not prices_on_date and target_weights:
                logger.warning(f"[{date_str}] No price data available; holding current positions.")
            else:
                portfolio.rebalance(target_weights, prices_on_date, rebal_date)

        # ── Daily NAV recording ────────────────────────────────────────────
        nav_series = self._record_daily_nav(portfolio, all_prices)

        # ── Benchmark series ───────────────────────────────────────────────
        bench_series = self._build_benchmark_series(benchmark_raw)

        result = BacktestResult(
            nav_series=nav_series,
            benchmark_series=bench_series,
            trade_log=portfolio.get_trade_log(),
            regime_history=pd.Series(regime_history, name="regime"),
            rebalance_dates=rebalance_dates,
            start_date=self.start_date.strftime("%Y-%m-%d"),
            end_date=self.end_date.strftime("%Y-%m-%d"),
            initial_capital=self.initial_capital,
            benchmark_ticker=self.benchmark_ticker,
        )

        logger.info(
            f"Backtest complete. Trades: {len(result.trade_log)}, "
            f"NAV days: {len(nav_series)}"
        )
        return result

    # ─── Rebalance Date Construction ─────────────────────────────────────────

    def _build_rebalance_dates(self) -> List[pd.Timestamp]:
        """Generate rebalance dates within the backtest window."""
        freq_map = {
            "weekly": "W-FRI",
            "monthly": "MS",   # Month Start
        }
        freq = freq_map.get(self.rebalance_freq, "MS")
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq=freq)
        # Filter to strictly within range
        return [d for d in dates if self.start_date <= d <= self.end_date]

    # ─── Data Fetching (no look-ahead) ────────────────────────────────────────

    def _fetch_all_prices(self) -> pd.DataFrame:
        """
        Fetch all daily closing prices for the full backtest window.
        Returns a DataFrame with columns: [date, stock_id, close]
        """
        try:
            df = self._conn.execute(f"""
                SELECT
                    CAST(date AS DATE) AS date,
                    stock_id,
                    close
                FROM price_daily_full
                WHERE CAST(date AS DATE) BETWEEN '{self.start_date.date()}' AND '{self.end_date.date()}'
                  AND close IS NOT NULL
                  AND close > 0
                ORDER BY date, stock_id
            """).fetchdf()
            df["date"] = pd.to_datetime(df["date"])
            logger.info(f"Fetched {len(df)} price rows.")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch prices: {e}")
            return pd.DataFrame(columns=["date", "stock_id", "close"])

    def _fetch_benchmark_prices(self) -> pd.Series:
        """Fetch benchmark closing prices as a pd.Series indexed by date."""
        try:
            df = self._conn.execute(f"""
                SELECT
                    CAST(date AS DATE) AS date,
                    close
                FROM price_daily_full
                WHERE stock_id = '{self.benchmark_ticker}'
                  AND CAST(date AS DATE) BETWEEN '{self.start_date.date()}' AND '{self.end_date.date()}'
                  AND close IS NOT NULL
                  AND close > 0
                ORDER BY date
            """).fetchdf()
            df["date"] = pd.to_datetime(df["date"])
            s = df.set_index("date")["close"]
            logger.info(f"Benchmark ({self.benchmark_ticker}): {len(s)} rows.")
            return s
        except Exception as e:
            logger.error(f"Failed to fetch benchmark prices: {e}")
            return pd.Series(dtype=float)

    def _prices_on_date(
        self, all_prices: pd.DataFrame, date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Get the most recent prices *before* the given rebalance date.
        This prevents look-ahead bias.
        """
        prior = all_prices[all_prices["date"] < date]
        if prior.empty:
            return {}
        # Take the most recent price per stock
        latest = prior.sort_values("date").groupby("stock_id").last()
        return latest["close"].to_dict()

    # ─── Regime Detection (DuckDB-only) ──────────────────────────────────────

    def _compute_regime(self, rebal_date: pd.Timestamp) -> Tuple[str, int]:
        """
        Compute a simplified regime score from macro_index_full data.

        The 6 signals below approximate the original RegimeDetector signals
        using only DuckDB macro data (no Alpha Vantage calls):

          1. Growth signal     — mean log return of all macro indices > 0
          2. Momentum signal   — 12-month change in value_norm > 0
          3. Volatility signal — rolling std of value_norm (low = bullish)
          4. Trend signal      — current value_norm vs 6-month average
          5. Breadth signal    — fraction of macro indices with positive momentum
          6. Rate signal       — rate-like index (if any) direction

        Each signal votes +1 (bullish) or -1 (bearish).
        Score range -6 to +6 maps to regime labels.
        """
        date_str = rebal_date.strftime("%Y-%m-%d")
        lookback_12m = (rebal_date - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
        lookback_6m  = (rebal_date - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
        lookback_3m  = (rebal_date - pd.DateOffset(months=3)).strftime("%Y-%m-%d")

        signals: Dict[str, int] = {}

        try:
            macro_df = self._conn.execute(f"""
                SELECT
                    index_name,
                    CAST(date AS DATE) AS date,
                    value_norm
                FROM macro_index_full
                WHERE CAST(date AS DATE) < '{date_str}'
                  AND CAST(date AS DATE) >= '{lookback_12m}'
                  AND value_norm IS NOT NULL
                ORDER BY date
            """).fetchdf()
        except Exception as e:
            logger.warning(f"Macro data fetch failed: {e}")
            return "Cautious", 0

        if macro_df.empty:
            logger.warning(f"[{date_str}] No macro data available. Defaulting to Cautious.")
            return "Cautious", 0

        macro_df["date"] = pd.to_datetime(macro_df["date"])

        # ── Signal 1: Mean log return of macro indices (growth proxy) ────────
        try:
            pivot = macro_df.pivot_table(
                index="date", columns="index_name", values="value_norm", aggfunc="last"
            ).sort_index()
            # Filter indices that have enough history
            valid_cols = [c for c in pivot.columns if pivot[c].dropna().shape[0] >= 10]
            pivot = pivot[valid_cols].ffill()

            if pivot.empty:
                signals["growth"] = 0
            else:
                log_rets = np.log(pivot + 1e-9).diff().dropna(how="all")
                mean_log_ret = float(log_rets.mean().mean())
                signals["growth"] = 1 if mean_log_ret > 0 else -1
        except Exception:
            signals["growth"] = 0

        # ── Signal 2: 12-month momentum ───────────────────────────────────────
        try:
            recent = macro_df[macro_df["date"] >= pd.Timestamp(lookback_3m)]
            old    = macro_df[macro_df["date"] < pd.Timestamp(lookback_3m)]
            if recent.empty or old.empty:
                signals["momentum"] = 0
            else:
                recent_mean = recent.groupby("index_name")["value_norm"].mean()
                old_mean    = old.groupby("index_name")["value_norm"].mean()
                common_idx = recent_mean.index.intersection(old_mean.index)
                if len(common_idx) == 0:
                    signals["momentum"] = 0
                else:
                    change = (recent_mean[common_idx] - old_mean[common_idx])
                    signals["momentum"] = 1 if change.mean() > 0 else -1
        except Exception:
            signals["momentum"] = 0

        # ── Signal 3: Low volatility = bullish ────────────────────────────────
        try:
            if "pivot" in dir() and not pivot.empty:
                recent_vol = pivot.iloc[-90:].std().mean() if len(pivot) >= 90 else pivot.std().mean()
                all_vol    = pivot.std().mean()
                signals["volatility"] = 1 if recent_vol < all_vol else -1
            else:
                signals["volatility"] = 0
        except Exception:
            signals["volatility"] = 0

        # ── Signal 4: Trend (current vs 6-month average) ──────────────────────
        try:
            if "pivot" in dir() and not pivot.empty:
                current_level = pivot.iloc[-1].mean()
                ma_6m = pivot[pivot.index >= pd.Timestamp(lookback_6m)].mean().mean()
                signals["trend"] = 1 if current_level > ma_6m else -1
            else:
                signals["trend"] = 0
        except Exception:
            signals["trend"] = 0

        # ── Signal 5: Breadth (fraction of indices with positive 3m momentum) ─
        try:
            if "pivot" in dir() and not pivot.empty and len(pivot) > 10:
                mom_3m = pivot.iloc[-1] - pivot.iloc[max(0, len(pivot)-90)]
                pos_frac = float((mom_3m > 0).mean())
                signals["breadth"] = 1 if pos_frac > 0.5 else -1
            else:
                signals["breadth"] = 0
        except Exception:
            signals["breadth"] = 0

        # ── Signal 6: Rate direction (use index names with 'RATE' or 'YIELD') ─
        try:
            rate_like = macro_df[
                macro_df["index_name"].str.contains("RATE|YIELD|FFR|DGS|TB", case=False, na=False)
            ]
            if len(rate_like) >= 2:
                rate_like = rate_like.sort_values("date")
                old_rate = rate_like.iloc[:len(rate_like)//2]["value_norm"].mean()
                new_rate = rate_like.iloc[len(rate_like)//2:]["value_norm"].mean()
                # Falling rates = bullish (easing)
                signals["rates"] = 1 if new_rate <= old_rate else -1
            else:
                signals["rates"] = 0
        except Exception:
            signals["rates"] = 0

        # Sum all non-zero signals
        score = sum(v for v in signals.values() if v != 0)
        regime_label = self._classify_regime(score)

        logger.debug(f"[{date_str}] Regime signals: {signals} → score={score} ({regime_label})")
        return regime_label, score

    @staticmethod
    def _classify_regime(score: int) -> str:
        """Map regime score (-6..+6) to a label string."""
        if score >= 4:
            return "Risk-On"
        elif score >= 0:
            return "Cautious"
        elif score >= -3:
            return "Defensive"
        else:
            return "Risk-Off"

    @staticmethod
    def _equity_fraction(score: int) -> float:
        """Map regime score to equity allocation fraction."""
        if score >= 4:
            return 1.00
        elif score >= 0:
            return 0.70
        elif score >= -3:
            return 0.40
        else:
            return 0.00

    # ─── Stock Scoring (inline, no live analysts) ─────────────────────────────

    def _score_stocks(self, rebal_date: pd.Timestamp) -> List[Dict]:
        """
        Score all stocks using a simplified version of DecisionMaker._score_stocks().
        Queries DuckDB directly; no look-ahead (all data < rebal_date).

        Returns list of {stock_id, score} dicts sorted descending by score.
        """
        date_str = rebal_date.strftime("%Y-%m-%d")
        lookback_1y  = (rebal_date - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
        lookback_3m  = (rebal_date - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
        lookback_fq  = (rebal_date - pd.DateOffset(months=6)).strftime("%Y-%m-%d")

        scores: Dict[str, float] = {}

        # ── Momentum from price data ───────────────────────────────────────────
        try:
            mom_df = self._conn.execute(f"""
                WITH price_data AS (
                    SELECT
                        stock_id,
                        CAST(date AS DATE) AS date,
                        close,
                        ROW_NUMBER() OVER (PARTITION BY stock_id ORDER BY date DESC) AS rn_latest,
                        ROW_NUMBER() OVER (PARTITION BY stock_id ORDER BY date ASC)  AS rn_oldest
                    FROM price_daily_full
                    WHERE CAST(date AS DATE) < '{date_str}'
                      AND CAST(date AS DATE) >= '{lookback_1y}'
                      AND close IS NOT NULL AND close > 0
                )
                SELECT
                    stock_id,
                    MAX(CASE WHEN rn_latest = 1 THEN close END) AS latest_close,
                    MIN(CASE WHEN rn_oldest = 1 THEN close END) AS oldest_close,
                    STDDEV(LN(close)) AS log_vol,
                    MIN(close) AS min_close,
                    MAX(close) AS max_close
                FROM price_data
                GROUP BY stock_id
                HAVING COUNT(*) >= 20
            """).fetchdf()

            for _, row in mom_df.iterrows():
                sid = row["stock_id"]
                if row["oldest_close"] and row["oldest_close"] > 0:
                    momentum = (row["latest_close"] / row["oldest_close"]) - 1.0
                    volatility = float(row["log_vol"]) if row["log_vol"] else 0.0
                    drawdown = (
                        (row["latest_close"] - row["max_close"]) / row["max_close"]
                        if row["max_close"] > 0 else 0.0
                    )
                    score = 2.0 * momentum - 0.5 * volatility + 0.1 * drawdown
                    scores[sid] = scores.get(sid, 0.0) + score
        except Exception as e:
            logger.warning(f"Price scoring failed: {e}")

        # ── Fundamental scoring ────────────────────────────────────────────────
        try:
            fund_df = self._conn.execute(f"""
                SELECT
                    stock_id,
                    AVG("Total Revenue") AS avg_revenue,
                    STDDEV("Total Revenue") AS std_revenue,
                    AVG("Net Income") AS avg_net_income,
                    AVG("Total Revenue") AS total_rev,
                    AVG("Gross Profit") AS avg_gross_profit
                FROM fundamentals_q_full
                WHERE CAST(fiscal_date AS DATE) < '{date_str}'
                  AND CAST(fiscal_date AS DATE) >= '{lookback_fq}'
                  AND "Total Revenue" IS NOT NULL
                  AND "Total Revenue" > 0
                GROUP BY stock_id
                HAVING COUNT(*) >= 1
            """).fetchdf()

            for _, row in fund_df.iterrows():
                sid = row["stock_id"]
                if row["avg_revenue"] and row["avg_revenue"] > 0:
                    # Revenue growth proxy: std/mean (coefficient of variation)
                    cv = (row["std_revenue"] / row["avg_revenue"]) if row["std_revenue"] else 0.0
                    net_margin = (
                        row["avg_net_income"] / row["avg_revenue"]
                        if row["avg_net_income"] and row["avg_revenue"] > 0 else 0.0
                    )
                    gross_margin = (
                        row["avg_gross_profit"] / row["avg_revenue"]
                        if row["avg_gross_profit"] and row["avg_revenue"] > 0 else 0.0
                    )
                    fund_score = 1.0 * cv + 0.5 * net_margin + 0.3 * gross_margin
                    scores[sid] = scores.get(sid, 0.0) + fund_score
        except Exception as e:
            logger.warning(f"Fundamental scoring failed: {e}")

        # ── News sentiment scoring ─────────────────────────────────────────────
        try:
            news_df = self._conn.execute(f"""
                SELECT
                    stock_id,
                    AVG(sentiment_score) AS avg_sentiment
                FROM news_sentiment
                WHERE CAST(date AS DATE) < '{date_str}'
                  AND CAST(date AS DATE) >= '{lookback_3m}'
                  AND sentiment_score IS NOT NULL
                GROUP BY stock_id
            """).fetchdf()

            for _, row in news_df.iterrows():
                sid = row["stock_id"]
                sentiment = float(row["avg_sentiment"])
                news_score = 0.3 * (sentiment - 0.5)
                scores[sid] = scores.get(sid, 0.0) + news_score
        except Exception as e:
            logger.warning(f"News sentiment scoring failed: {e}")

        # Sort by score descending
        sorted_stocks = sorted(
            [{"stock_id": sid, "score": score} for sid, score in scores.items()],
            key=lambda x: x["score"],
            reverse=True,
        )
        return sorted_stocks

    # ─── Portfolio Weight Construction ────────────────────────────────────────

    def _build_target_weights(
        self,
        scored_stocks: List[Dict],
        equity_fraction: float,
        rebal_date: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        Build equal-weight target_weights for top_n stocks, scaled by equity_fraction.

        Cash is implicit (whatever is left after assigning equity weights).
        """
        if equity_fraction <= 0 or not scored_stocks:
            return {}  # 100% cash

        # Take top_n stocks
        top_stocks = scored_stocks[: self.top_n]
        n = len(top_stocks)

        if n == 0:
            logger.warning(f"[{rebal_date.date()}] No stocks available; holding cash.")
            return {}

        weight_per_stock = equity_fraction / n

        weights = {}
        for stock in top_stocks:
            sid = stock["stock_id"]
            weights[sid] = weight_per_stock

        return weights

    # ─── NAV Recording ────────────────────────────────────────────────────────

    def _record_daily_nav(
        self,
        portfolio: PortfolioSimulator,
        all_prices: pd.DataFrame,
    ) -> pd.Series:
        """
        After all rebalances are done, replay portfolio NAV day by day.

        This method replays the trades chronologically to compute a daily NAV
        series for charting.  Note: since we already executed trades during
        the walk-forward, we use the portfolio's final state and reconstruct
        by walking the price series forward.

        For simplicity and correctness, we record NAV at each calendar date
        using the prices available at that date and the portfolio's
        current holdings.
        """
        if all_prices.empty:
            return pd.Series(dtype=float)

        # Build a dict of date → {stock_id: close} for fast lookup
        dates_sorted = sorted(all_prices["date"].unique())
        nav_records = []

        # Replay: start with initial state, apply trades in order
        # Since portfolio already executed all trades, we need to replay
        # We reconstruct by iterating trades chronologically
        trade_log = pd.DataFrame(portfolio.get_trade_log())
        if not trade_log.empty:
            trade_log["date"] = pd.to_datetime(trade_log["date"])
            trade_log = trade_log.sort_values("date")

        # Reconstruct positions day by day
        current_cash = self.initial_capital
        current_positions: Dict[str, float] = {}
        trade_idx = 0
        trades_list = trade_log.to_dict("records") if not trade_log.empty else []

        for day in dates_sorted:
            # Apply all trades that happened on or before this day
            while trade_idx < len(trades_list) and pd.Timestamp(trades_list[trade_idx]["date"]) <= day:
                t = trades_list[trade_idx]
                current_cash += t["net_cash_flow"]
                sid = t["stock_id"]
                if t["action"] == "BUY":
                    current_positions[sid] = current_positions.get(sid, 0.0) + t["shares"]
                else:  # SELL
                    current_positions[sid] = max(0.0, current_positions.get(sid, 0.0) - t["shares"])
                    if current_positions[sid] < 1e-6:
                        current_positions.pop(sid, None)
                trade_idx += 1

            # Compute NAV
            prices_today = all_prices[all_prices["date"] == day].set_index("stock_id")["close"].to_dict()
            equity_value = sum(
                shares * prices_today.get(sid, 0.0)
                for sid, shares in current_positions.items()
            )
            nav = current_cash + equity_value
            nav_records.append({"date": day, "nav": nav})

        if not nav_records:
            return pd.Series(dtype=float)

        df = pd.DataFrame(nav_records).set_index("date")
        df.index = pd.to_datetime(df.index)
        return df["nav"]

    # ─── Benchmark ────────────────────────────────────────────────────────────

    def _build_benchmark_series(self, raw: pd.Series) -> pd.Series:
        """
        Normalise benchmark price series so that it starts at initial_capital.
        Handles the case where the benchmark has no data gracefully.
        """
        if raw.empty:
            logger.warning(f"No benchmark data for {self.benchmark_ticker}.")
            return pd.Series(dtype=float)

        # Normalise to initial_capital
        first_price = raw.iloc[0]
        if first_price <= 0:
            return pd.Series(dtype=float)
        return raw / first_price * self.initial_capital
