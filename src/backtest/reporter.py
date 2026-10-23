"""
BacktestReporter
----------------
Formats and saves backtest results to:
  • console (tabulate or plain print fallback)
  • CSV files
  • self-contained HTML with Plotly charts

All chart dependencies are loaded from CDN; no local install required.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from backtest.engine import BacktestResult
from backtest.performance import PerformanceReport

logger = logging.getLogger(__name__)


# ─── Colour map for regime bands ─────────────────────────────────────────────

REGIME_COLORS = {
    "Risk-On":   "rgba(0, 200, 0, 0.15)",
    "Cautious":  "rgba(255, 220, 0, 0.20)",
    "Defensive": "rgba(255, 140, 0, 0.20)",
    "Risk-Off":  "rgba(220, 0, 0, 0.20)",
}


class BacktestReporter:
    """Renders backtest results to multiple output formats."""

    # ─── Console ─────────────────────────────────────────────────────────────

    @staticmethod
    def to_console(report: PerformanceReport) -> None:
        """Print a formatted performance summary to stdout."""
        data = report.to_dict()

        # Build rows
        rows = [
            # Returns
            ("─── Returns ───────────────────────────────────────", ""),
            ("Total Return",             f"{data['total_return']:.2%}"),
            ("CAGR",                     f"{data['cagr']:.2%}"),
            ("Benchmark Total Return",   f"{data['benchmark_total_return']:.2%}"),
            ("Benchmark CAGR",           f"{data['benchmark_cagr']:.2%}"),
            ("Excess Return (α)",        f"{data['excess_return']:.2%}"),
            # Risk
            ("─── Risk ──────────────────────────────────────────", ""),
            ("Annualized Volatility",    f"{data['annualized_volatility']:.2%}"),
            ("Max Drawdown",             f"{data['max_drawdown']:.2%}"),
            ("Max Drawdown Duration",    f"{data['max_drawdown_duration_days']} days"),
            ("VaR 95%",                  f"{data['var_95']:.2%}"),
            ("VaR 99%",                  f"{data['var_99']:.2%}"),
            ("CVaR 95%",                 f"{data['cvar_95']:.2%}"),
            ("Skewness",                 f"{data['skewness']:.3f}"),
            ("Kurtosis",                 f"{data['kurtosis']:.3f}"),
            # Risk-Adjusted
            ("─── Risk-Adjusted ────────────────────────────────", ""),
            ("Sharpe Ratio",             f"{data['sharpe_ratio']:.3f}"),
            ("Sortino Ratio",            f"{data['sortino_ratio']:.3f}"),
            ("Calmar Ratio",             f"{data['calmar_ratio']:.3f}"),
            ("Information Ratio",        f"{data['information_ratio']:.3f}"),
            ("Beta",                     f"{data['beta']:.3f}"),
            ("Alpha (annualized)",       f"{data['alpha']:.2%}"),
            # Trades
            ("─── Trade Statistics ─────────────────────────────", ""),
            ("Total Trades",             str(data["total_trades"])),
            ("Win Rate",                 f"{data['win_rate']:.2%}"),
            ("Avg Win",                  f"{data['avg_win_pct']:.2%}"),
            ("Avg Loss",                 f"{data['avg_loss_pct']:.2%}"),
            ("Profit Factor",            f"{data['profit_factor']:.3f}"),
            ("Avg Holding Period",       f"{data['avg_holding_period_days']:.1f} days"),
            ("Turnover per Rebalance",   f"{data['turnover_per_rebalance']:.2%}"),
        ]

        # Regime attribution rows
        regime_keys = [k for k in data if k.startswith("regime_") and k.endswith("_period_count")]
        if regime_keys:
            rows.append(("─── Regime Attribution ────────────────────────────", ""))
            for key in sorted(regime_keys):
                base = key.replace("_period_count", "").replace("regime_", "")
                label = base.replace("_", " ")
                periods = data.get(f"regime_{base}_period_count", "n/a")
                days    = data.get(f"regime_{base}_total_days", "n/a")
                ret     = data.get(f"regime_{base}_cumulative_return", float("nan"))
                ret_str = f"{ret:.2%}" if isinstance(ret, float) and not pd.isna(ret) else "n/a"
                rows.append((f"  {label}", f"{periods} periods, {days} days, {ret_str}"))

        try:
            from tabulate import tabulate
            print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))
        except ImportError:
            # Graceful fallback
            col_w = 42
            print("=" * (col_w + 20))
            print(f"{'Metric':<{col_w}} {'Value'}")
            print("=" * (col_w + 20))
            for metric, value in rows:
                print(f"{metric:<{col_w}} {value}")
            print("=" * (col_w + 20))

    # ─── CSV ─────────────────────────────────────────────────────────────────

    @staticmethod
    def to_csv(
        result: BacktestResult,
        report: PerformanceReport,
        output_dir: str,
    ) -> str:
        """
        Save daily NAV + benchmark to CSV.
        Returns the path of the saved CSV.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nav_path = Path(output_dir) / f"backtest_nav_{timestamp}.csv"
        stats_path = Path(output_dir) / f"backtest_stats_{timestamp}.csv"

        # NAV + Benchmark
        nav_df = pd.DataFrame({
            "strategy_nav": result.nav_series,
            "benchmark_nav": result.benchmark_series,
        })
        nav_df.index.name = "date"
        nav_df.to_csv(nav_path)
        logger.info(f"NAV CSV saved: {nav_path}")

        # Performance stats
        stats_dict = report.to_dict()
        stats_df = pd.DataFrame(
            [(k, v) for k, v in stats_dict.items()],
            columns=["metric", "value"],
        )
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"Stats CSV saved: {stats_path}")

        return str(nav_path)

    # ─── HTML ────────────────────────────────────────────────────────────────

    @staticmethod
    def to_html(
        result: BacktestResult,
        report: PerformanceReport,
        output_dir: str,
    ) -> Optional[str]:
        """
        Generate a self-contained HTML report with Plotly charts.

        Returns path of the HTML file, or None if plotly is unavailable.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            _plotly_available = True
        except ImportError:
            logger.warning("plotly not installed; skipping HTML report. Run: pip install plotly")
            return None

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = Path(output_dir) / f"backtest_{timestamp}.html"

        # ── Normalize NAV series to 100 ────────────────────────────────────
        def normalize(s: pd.Series) -> pd.Series:
            if s.empty or s.iloc[0] == 0:
                return s
            return s / s.iloc[0] * 100.0

        nav_norm   = normalize(result.nav_series)
        bench_norm = normalize(result.benchmark_series)

        # ── Drawdown series ───────────────────────────────────────────────
        nav_s = result.nav_series
        drawdown = pd.Series(dtype=float)
        if not nav_s.empty:
            cummax = nav_s.cummax()
            drawdown = (nav_s - cummax) / cummax * 100.0

        # ── Build Plotly figure ───────────────────────────────────────────
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Equity Curve (Normalized to 100)", "Drawdown (%)"),
            row_heights=[0.65, 0.35],
            vertical_spacing=0.06,
        )

        # Equity curve traces
        if not nav_norm.empty:
            fig.add_trace(
                go.Scatter(
                    x=nav_norm.index.tolist(),
                    y=nav_norm.values.tolist(),
                    name="Strategy",
                    line=dict(color="#1f77b4", width=2),
                ),
                row=1, col=1,
            )
        if not bench_norm.empty:
            fig.add_trace(
                go.Scatter(
                    x=bench_norm.index.tolist(),
                    y=bench_norm.values.tolist(),
                    name=result.benchmark_ticker,
                    line=dict(color="#ff7f0e", width=1.5, dash="dash"),
                ),
                row=1, col=1,
            )

        # Drawdown trace
        if not drawdown.empty:
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index.tolist(),
                    y=drawdown.values.tolist(),
                    name="Drawdown",
                    fill="tozeroy",
                    line=dict(color="#d62728", width=1),
                    fillcolor="rgba(214,39,40,0.15)",
                ),
                row=2, col=1,
            )

        # Regime overlay bands
        if result.regime_history is not None and not result.regime_history.empty:
            regime = result.regime_history.sort_index()
            prev_regime: Optional[str] = None
            band_start: Optional[pd.Timestamp] = None

            def add_band(start: pd.Timestamp, end: pd.Timestamp, label: str) -> None:
                color = REGIME_COLORS.get(label, "rgba(128,128,128,0.1)")
                for row in [1, 2]:
                    fig.add_vrect(
                        x0=str(start.date()),
                        x1=str(end.date()),
                        fillcolor=color,
                        opacity=1.0,
                        layer="below",
                        line_width=0,
                        row=row, col=1,
                    )

            for ts, label in regime.items():
                if label != prev_regime:
                    if prev_regime is not None and band_start is not None:
                        add_band(band_start, ts, prev_regime)
                    band_start = ts
                    prev_regime = label
            if prev_regime is not None and band_start is not None:
                add_band(band_start, regime.index[-1], prev_regime)

        # Add invisible legend items for regime colours
        for label, color in REGIME_COLORS.items():
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color.replace("0.15", "0.7").replace("0.20", "0.7")),
                    legendgroup=label,
                    showlegend=True,
                    name=f"Regime: {label}",
                ),
                row=1, col=1,
            )

        fig.update_layout(
            title=f"Backtest Report  |  {result.start_date} → {result.end_date}",
            template="plotly_white",
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#ececec")
        fig.update_yaxes(showgrid=True, gridcolor="#ececec")

        plotly_json = fig.to_json()

        # ── Stats table HTML ─────────────────────────────────────────────
        stats_html = BacktestReporter._build_stats_table_html(report)

        # ── Trade log table HTML (last 50 trades) ────────────────────────
        trades_html = BacktestReporter._build_trades_table_html(result)

        # ── Assemble HTML ────────────────────────────────────────────────
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Backtest Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px;
         background: #f5f7fa; color: #333; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.2rem; margin-top: 30px; margin-bottom: 8px;
        border-bottom: 2px solid #1f77b4; padding-bottom: 4px; }}
  #chart {{ background: white; border-radius: 8px; padding: 10px;
             box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-bottom: 24px; }}
  table {{ border-collapse: collapse; width: 100%; background: white;
           border-radius: 8px; overflow: hidden;
           box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-bottom: 24px; }}
  th {{ background: #1f77b4; color: white; padding: 8px 12px; text-align: left;
        font-size: .85rem; }}
  td {{ padding: 6px 12px; font-size: .84rem; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #f0f4ff; }}
  .section-label td {{ background: #e8edf5; font-weight: bold; color: #1f77b4; }}
  .meta {{ color: #666; font-size: .85rem; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>📊 Backtest Report</h1>
<div class="meta">
  Period: <strong>{result.start_date}</strong> → <strong>{result.end_date}</strong>
  &nbsp;|&nbsp; Initial Capital: <strong>{result.initial_capital:,.0f}</strong>
  &nbsp;|&nbsp; Benchmark: <strong>{result.benchmark_ticker}</strong>
</div>

<h2>Equity Curve &amp; Drawdown</h2>
<div id="chart"></div>
<script>
  var figure = {plotly_json};
  Plotly.newPlot('chart', figure.data, figure.layout, {{responsive: true}});
</script>

<h2>Performance Statistics</h2>
{stats_html}

<h2>Recent Trades (last 50)</h2>
{trades_html}

<footer style="color:#aaa; font-size:.75rem; margin-top:30px;">
  Generated by Stock_ML BacktestReporter on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</footer>
</body>
</html>"""

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML report saved: {html_path}")
        return str(html_path)

    # ─── Internal Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _fmt(val: object) -> str:
        """Format a value for display."""
        if val is None:
            return "n/a"
        if isinstance(val, float):
            if pd.isna(val):
                return "n/a"
            return f"{val:.4f}"
        return str(val)

    @staticmethod
    def _fmt_pct(val: object) -> str:
        """Format a percentage value for display."""
        if val is None:
            return "n/a"
        if isinstance(val, float):
            if pd.isna(val):
                return "n/a"
            return f"{val:.2%}"
        return str(val)

    @staticmethod
    def _build_stats_table_html(report: PerformanceReport) -> str:
        d = report.to_dict()
        rows_spec = [
            ("section", "Returns"),
            ("Total Return",           BacktestReporter._fmt_pct(d["total_return"])),
            ("CAGR",                   BacktestReporter._fmt_pct(d["cagr"])),
            ("Benchmark Total Return", BacktestReporter._fmt_pct(d["benchmark_total_return"])),
            ("Benchmark CAGR",         BacktestReporter._fmt_pct(d["benchmark_cagr"])),
            ("Excess Return (Alpha)",  BacktestReporter._fmt_pct(d["excess_return"])),
            ("section", "Risk"),
            ("Annualized Volatility",  BacktestReporter._fmt_pct(d["annualized_volatility"])),
            ("Max Drawdown",           BacktestReporter._fmt_pct(d["max_drawdown"])),
            ("Max Drawdown Duration",  f"{d['max_drawdown_duration_days']} days"),
            ("VaR 95%",                BacktestReporter._fmt_pct(d["var_95"])),
            ("VaR 99%",                BacktestReporter._fmt_pct(d["var_99"])),
            ("CVaR 95%",               BacktestReporter._fmt_pct(d["cvar_95"])),
            ("Skewness",               BacktestReporter._fmt(d["skewness"])),
            ("Kurtosis",               BacktestReporter._fmt(d["kurtosis"])),
            ("section", "Risk-Adjusted"),
            ("Sharpe Ratio",           BacktestReporter._fmt(d["sharpe_ratio"])),
            ("Sortino Ratio",          BacktestReporter._fmt(d["sortino_ratio"])),
            ("Calmar Ratio",           BacktestReporter._fmt(d["calmar_ratio"])),
            ("Information Ratio",      BacktestReporter._fmt(d["information_ratio"])),
            ("Beta",                   BacktestReporter._fmt(d["beta"])),
            ("Alpha (annualized)",     BacktestReporter._fmt_pct(d["alpha"])),
            ("section", "Trade Statistics"),
            ("Total Trades",           str(d["total_trades"])),
            ("Win Rate",               BacktestReporter._fmt_pct(d["win_rate"])),
            ("Avg Win %",              BacktestReporter._fmt_pct(d["avg_win_pct"])),
            ("Avg Loss %",             BacktestReporter._fmt_pct(d["avg_loss_pct"])),
            ("Profit Factor",          BacktestReporter._fmt(d["profit_factor"])),
            ("Avg Holding Period",     f"{BacktestReporter._fmt(d['avg_holding_period_days'])} days"),
            ("Turnover / Rebalance",   BacktestReporter._fmt_pct(d["turnover_per_rebalance"])),
        ]

        # Regime attribution
        regime_keys = sorted(k for k in d if k.startswith("regime_") and k.endswith("_period_count"))
        if regime_keys:
            rows_spec.append(("section", "Regime Attribution"))
            for key in regime_keys:
                base = key.replace("_period_count", "").replace("regime_", "")
                label = base.replace("_", " ").title()
                periods = d.get(f"regime_{base}_period_count", "n/a")
                days    = d.get(f"regime_{base}_total_days", "n/a")
                ret_raw = d.get(f"regime_{base}_cumulative_return", float("nan"))
                ret_str = BacktestReporter._fmt_pct(ret_raw) if isinstance(ret_raw, float) else "n/a"
                rows_spec.append((label, f"{periods} periods | {days} days | return: {ret_str}"))

        html = '<table><tr><th>Metric</th><th>Value</th></tr>'
        for row in rows_spec:
            if row[0] == "section":
                html += f'<tr class="section-label"><td colspan="2">{row[1]}</td></tr>'
            else:
                html += f"<tr><td>{row[0]}</td><td>{row[1]}</td></tr>"
        html += "</table>"
        return html

    @staticmethod
    def _build_trades_table_html(result: BacktestResult) -> str:
        trades = result.trade_log[-50:]  # last 50
        if not trades:
            return "<p>No trades recorded.</p>"

        cols = ["date", "stock_id", "action", "shares", "price", "exec_price",
                "gross_value", "commission", "net_cash_flow"]
        html = "<table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
        for t in trades:
            html += "<tr>" + "".join(
                f"<td>{BacktestReporter._fmt_trade_cell(t.get(c, ''))}</td>"
                for c in cols
            ) + "</tr>"
        html += "</table>"
        return html

    @staticmethod
    def _fmt_trade_cell(val: object) -> str:
        if isinstance(val, float):
            return f"{val:,.2f}"
        if isinstance(val, pd.Timestamp):
            return str(val.date())
        return str(val)
