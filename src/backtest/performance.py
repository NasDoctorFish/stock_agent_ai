"""
PerformanceCalculator
---------------------
Computes comprehensive performance statistics from a backtest result.

Inputs:
    - daily NAV series (pd.Series, DatetimeIndex)
    - benchmark series (pd.Series, DatetimeIndex)
    - trade log (list of dicts from PortfolioSimulator)
    - regime history (pd.Series with regime labels)

No external API calls.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ─── Report Dataclass ────────────────────────────────────────────────────────

@dataclass
class PerformanceReport:
    """All computed performance statistics."""

    # ── Returns ──────────────────────────────────────────────────────────────
    total_return: float = float("nan")
    cagr: float = float("nan")
    benchmark_total_return: float = float("nan")
    benchmark_cagr: float = float("nan")
    excess_return: float = float("nan")

    # ── Risk ─────────────────────────────────────────────────────────────────
    annualized_volatility: float = float("nan")
    max_drawdown: float = float("nan")
    max_drawdown_duration_days: int = 0
    var_95: float = float("nan")
    var_99: float = float("nan")
    cvar_95: float = float("nan")
    skewness: float = float("nan")
    kurtosis: float = float("nan")

    # ── Risk-Adjusted ────────────────────────────────────────────────────────
    sharpe_ratio: float = float("nan")
    sortino_ratio: float = float("nan")
    calmar_ratio: float = float("nan")
    information_ratio: float = float("nan")
    beta: float = float("nan")
    alpha: float = float("nan")

    # ── Trade Statistics ─────────────────────────────────────────────────────
    total_trades: int = 0
    win_rate: float = float("nan")
    avg_win_pct: float = float("nan")
    avg_loss_pct: float = float("nan")
    profit_factor: float = float("nan")
    avg_holding_period_days: float = float("nan")
    turnover_per_rebalance: float = float("nan")

    # ── Regime Attribution ────────────────────────────────────────────────────
    # {regime_label: {"period_count": int, "total_days": int, "cumulative_return": float}}
    regime_attribution: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a flat dict for CSV / JSON export."""
        d: Dict[str, Any] = {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "benchmark_total_return": self.benchmark_total_return,
            "benchmark_cagr": self.benchmark_cagr,
            "excess_return": self.excess_return,
            "annualized_volatility": self.annualized_volatility,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "information_ratio": self.information_ratio,
            "beta": self.beta,
            "alpha": self.alpha,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "profit_factor": self.profit_factor,
            "avg_holding_period_days": self.avg_holding_period_days,
            "turnover_per_rebalance": self.turnover_per_rebalance,
        }
        # Flatten regime attribution
        for label, stats in self.regime_attribution.items():
            safe = label.replace(" ", "_").replace("/", "_")
            d[f"regime_{safe}_period_count"] = stats.get("period_count", 0)
            d[f"regime_{safe}_total_days"] = stats.get("total_days", 0)
            d[f"regime_{safe}_cumulative_return"] = stats.get("cumulative_return", float("nan"))
        return d


# ─── Calculator ──────────────────────────────────────────────────────────────

class PerformanceCalculator:
    """
    Computes all performance statistics from a backtest result.

    Usage
    -----
    calc = PerformanceCalculator(nav, benchmark, trade_log, regime_history)
    report = calc.compute()
    """

    def __init__(
        self,
        nav: pd.Series,
        benchmark: pd.Series,
        trade_log: List[dict],
        regime_history: pd.Series,
    ) -> None:
        self.nav = nav.sort_index().dropna()
        self.benchmark = benchmark.sort_index().dropna()
        self.trade_log = trade_log
        self.regime_history = regime_history.sort_index() if regime_history is not None else pd.Series(dtype=str)

    # ─── Public ───────────────────────────────────────────────────────────────

    def compute(self) -> PerformanceReport:
        report = PerformanceReport()

        if len(self.nav) < 2:
            logger.warning("NAV series too short to compute performance.")
            return report

        # Daily returns (strategy + benchmark)
        strat_returns = self.nav.pct_change().dropna()
        bench_returns = self.benchmark.pct_change().dropna()

        # Align to common dates
        common = strat_returns.index.intersection(bench_returns.index)
        if len(common) > 0:
            sr = strat_returns.loc[common]
            br = bench_returns.loc[common]
        else:
            sr = strat_returns
            br = bench_returns

        years = self._years(self.nav)

        # Returns
        report.total_return = self._total_return(self.nav)
        report.cagr = self._cagr(self.nav, years)

        if len(self.benchmark) >= 2:
            bench_years = self._years(self.benchmark)
            report.benchmark_total_return = self._total_return(self.benchmark)
            report.benchmark_cagr = self._cagr(self.benchmark, bench_years)

        report.excess_return = report.total_return - report.benchmark_total_return

        # Risk
        report.annualized_volatility = float(sr.std() * np.sqrt(TRADING_DAYS))
        dd, dd_dur = self._drawdown_stats(self.nav)
        report.max_drawdown = dd
        report.max_drawdown_duration_days = dd_dur
        report.var_95 = float(np.percentile(sr, 5))
        report.var_99 = float(np.percentile(sr, 1))
        report.cvar_95 = float(sr[sr <= report.var_95].mean()) if len(sr[sr <= report.var_95]) > 0 else float("nan")
        report.skewness = float(sr.skew())
        report.kurtosis = float(sr.kurtosis())

        # Risk-adjusted
        ann_ret = report.cagr
        ann_vol = report.annualized_volatility
        report.sharpe_ratio = (ann_ret / ann_vol) if ann_vol > 0 else float("nan")

        downside = sr[sr < 0].std() * np.sqrt(TRADING_DAYS)
        report.sortino_ratio = (ann_ret / downside) if downside > 0 else float("nan")

        report.calmar_ratio = (ann_ret / abs(report.max_drawdown)) if report.max_drawdown < 0 else float("nan")

        if len(common) > 1:
            excess_returns = sr - br
            tracking_error = float(excess_returns.std() * np.sqrt(TRADING_DAYS))
            excess_ann = float(excess_returns.mean() * TRADING_DAYS)
            report.information_ratio = (excess_ann / tracking_error) if tracking_error > 0 else float("nan")
            report.beta, report.alpha = self._ols_beta_alpha(sr, br)

        # Trade statistics
        self._compute_trade_stats(report)

        # Regime attribution
        report.regime_attribution = self._regime_attribution()

        return report

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _years(series: pd.Series) -> float:
        """Calendar years between first and last date."""
        delta = (series.index[-1] - series.index[0]).days
        return max(delta / 365.25, 1e-9)

    @staticmethod
    def _total_return(series: pd.Series) -> float:
        return float(series.iloc[-1] / series.iloc[0] - 1.0)

    @staticmethod
    def _cagr(series: pd.Series, years: float) -> float:
        tr = PerformanceCalculator._total_return(series)
        return float((1.0 + tr) ** (1.0 / years) - 1.0)

    @staticmethod
    def _drawdown_stats(series: pd.Series):
        """Return (max_drawdown_fraction, max_drawdown_duration_days)."""
        cummax = series.cummax()
        drawdown = (series - cummax) / cummax

        max_dd = float(drawdown.min())

        # Duration: find the longest stretch below previous peak
        in_drawdown = drawdown < 0
        max_dur = 0
        current_dur_start: Optional[pd.Timestamp] = None

        for ts, flag in in_drawdown.items():
            if flag:
                if current_dur_start is None:
                    current_dur_start = ts
            else:
                if current_dur_start is not None:
                    dur = (ts - current_dur_start).days
                    max_dur = max(max_dur, dur)
                    current_dur_start = None

        # If still in drawdown at end
        if current_dur_start is not None:
            dur = (series.index[-1] - current_dur_start).days
            max_dur = max(max_dur, dur)

        return max_dd, max_dur

    @staticmethod
    def _ols_beta_alpha(strat_ret: pd.Series, bench_ret: pd.Series):
        """OLS regression: strat = alpha + beta * bench."""
        try:
            x = bench_ret.values
            y = strat_ret.values
            cov = np.cov(x, y)
            beta = cov[0, 1] / cov[0, 0] if cov[0, 0] != 0 else float("nan")
            alpha_daily = float(np.mean(y)) - beta * float(np.mean(x))
            alpha_ann = alpha_daily * TRADING_DAYS
            return float(beta), float(alpha_ann)
        except Exception:
            return float("nan"), float("nan")

    def _compute_trade_stats(self, report: PerformanceReport) -> None:
        """Compute trade-level statistics from the trade log."""
        if not self.trade_log:
            report.total_trades = 0
            return

        trades = pd.DataFrame(self.trade_log)
        report.total_trades = len(trades)

        # Win/loss based on net_cash_flow (sells that made money vs buys that cost)
        # Better: pair buys and sells per stock_id by FIFO
        pnl_list: List[float] = []
        holding_periods: List[float] = []

        # Group by stock_id, reconstruct round trips
        for sid, grp in trades.groupby("stock_id"):
            grp = grp.sort_values("date").reset_index(drop=True)
            buy_queue: List[dict] = []

            for _, row in grp.iterrows():
                if row["action"] == "BUY":
                    buy_queue.append({"date": row["date"], "price": row["exec_price"], "shares": row["shares"]})
                elif row["action"] == "SELL" and buy_queue:
                    sell_price = row["exec_price"]
                    sell_date = row["date"]
                    shares_to_match = row["shares"]

                    while shares_to_match > 1e-6 and buy_queue:
                        buy = buy_queue[0]
                        matched = min(shares_to_match, buy["shares"])
                        pnl_pct = (sell_price - buy["price"]) / buy["price"]
                        pnl_list.append(pnl_pct)
                        hp = (pd.to_datetime(sell_date) - pd.to_datetime(buy["date"])).days
                        holding_periods.append(hp)

                        buy["shares"] -= matched
                        shares_to_match -= matched
                        if buy["shares"] < 1e-6:
                            buy_queue.pop(0)

        if pnl_list:
            pnl_arr = np.array(pnl_list)
            wins = pnl_arr[pnl_arr > 0]
            losses = pnl_arr[pnl_arr <= 0]

            report.win_rate = float(len(wins) / len(pnl_arr))
            report.avg_win_pct = float(wins.mean()) if len(wins) > 0 else float("nan")
            report.avg_loss_pct = float(losses.mean()) if len(losses) > 0 else float("nan")

            gross_profit = wins.sum() if len(wins) > 0 else 0.0
            gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
            report.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("nan")

        if holding_periods:
            report.avg_holding_period_days = float(np.mean(holding_periods))

        # Turnover per rebalance = total gross traded / average NAV
        if "gross_value" in trades.columns:
            total_gross = trades["gross_value"].sum()
            avg_nav = self.nav.mean() if len(self.nav) > 0 else 1.0
            rebalance_count = trades["date"].nunique()
            if rebalance_count > 0 and avg_nav > 0:
                report.turnover_per_rebalance = float(total_gross / rebalance_count / avg_nav)

    def _regime_attribution(self) -> Dict[str, Dict[str, Any]]:
        """Compute return attribution per regime label."""
        if self.regime_history is None or len(self.regime_history) == 0 or len(self.nav) == 0:
            return {}

        # Forward-fill regime labels onto NAV dates
        nav_dates = self.nav.index
        regime_aligned = self.regime_history.reindex(nav_dates, method="ffill")

        result: Dict[str, Dict[str, Any]] = {}
        daily_returns = self.nav.pct_change()

        for label in regime_aligned.dropna().unique():
            mask = regime_aligned == label
            dates_in_regime = nav_dates[mask]

            if len(dates_in_regime) == 0:
                continue

            # Count distinct contiguous periods
            regime_int = mask.astype(int)
            transitions = regime_int.diff().fillna(0)
            period_count = int((transitions == 1).sum())

            total_days = int(mask.sum())

            # Cumulative return during this regime (product of daily returns)
            regime_returns = daily_returns[mask].dropna()
            cum_ret = float((1 + regime_returns).prod() - 1) if len(regime_returns) > 0 else float("nan")

            result[label] = {
                "period_count": period_count,
                "total_days": total_days,
                "cumulative_return": cum_ret,
            }

        return result
