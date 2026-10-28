"""
dashboard.py — Streamlit GUI for the Stock_ML backtester.

Lets you configure a backtest period/params, run it against the real DB,
and inspect the resulting trade records (Trade Log) and performance
evaluation (stats + equity curve + Investment Records / round-trip P&L)
without touching the CLI or opening a generated HTML file.

Launch with ./run_backtest_gui.sh from the repo root.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Ensure src/ is on sys.path so bare imports below resolve regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import pandas as pd
import streamlit as st

from backtest.engine import BacktestEngine, BacktestResult
from backtest.performance import PerformanceCalculator
from backtest.reporter import BacktestReporter, REGIME_SOLID_COLORS

# Validated status palette (see dataviz skill references/palette.md) —
# reserved for good/bad judgments, never reused as a plain series color.
_GOOD = "#0ca30c"
_CRITICAL = "#d03b3b"
_NEUTRAL = "#898781"

_CARD_CSS = """
<style>
.sml-cards { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1.2rem; }
.sml-card {
  flex: 1 1 160px; border-radius: 10px; padding: 14px 16px;
  border-left: 4px solid rgba(128,128,128,0.35); background: rgba(128,128,128,0.08);
}
.sml-card .label {
  font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em;
  opacity: 0.65; margin-bottom: 4px;
}
.sml-card .value {
  font-size: 1.55rem; font-weight: 700; color: var(--accent);
  font-variant-numeric: tabular-nums;
}
</style>
"""


def _status_color(good: Optional[bool]) -> str:
    if good is None:
        return _NEUTRAL
    return _GOOD if good else _CRITICAL


def _metric_card(label: str, value: str, good: Optional[bool]) -> str:
    accent = _status_color(good)
    icon = "" if good is None else ("▲ " if good else "▼ ")
    return (
        f'<div class="sml-card" style="--accent:{accent};">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{icon}{value}</div>'
        f"</div>"
    )


def _metric_cards_row(cards: list) -> None:
    html = '<div class="sml-cards">' + "".join(
        _metric_card(label, value, good) for label, value, good in cards
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _comparison_table(stats: dict, bstats: dict, bench: str) -> None:
    """Strategy vs benchmark; colour/arrow only on the Difference column.
    (label, key, kind, higher_is_better) - kind: pct | num."""
    spec = [
        ("Total Return", "total_return", "pct", True), ("CAGR", "cagr", "pct", True),
        ("Volatility", "annualized_volatility", "pct", False),
        ("Sharpe", "sharpe_ratio", "num", True), ("Sortino", "sortino_ratio", "num", True),
        ("Max DD", "max_drawdown", "pct", True),  # less negative is better
        ("Beta", "beta", "num", None), ("Alpha", "alpha", "pct", True),
        ("Info Ratio", "information_ratio", "num", True), ("Excess", "excess_return", "pct", True),
    ]
    fmt = lambda v, k: "n/a" if v is None or pd.isna(v) else (f"{v:.2%}" if k == "pct" else f"{v:.2f}")
    rows, goods = [], []
    for label, key, kind, hib in spec:
        s_, b_ = stats.get(key), bstats.get(key)
        if key in ("beta", "alpha", "information_ratio", "excess_return"):
            b_ = {"beta": 1.0, "alpha": 0.0, "information_ratio": 0.0, "excess_return": 0.0}[key]  # benchmark vs itself
        d = None if s_ is None or b_ is None or pd.isna(s_) or pd.isna(b_) else s_ - b_
        good = None if d is None or hib is None else (d >= 0) == hib
        arrow = "" if good is None else ("▲ " if good else "▼ ")
        rows.append([label, fmt(s_, kind), fmt(b_, kind),
                     "n/a" if d is None else f"{arrow}{d:+.2%}" if kind == "pct" else f"{arrow}{d:+.2f}"])
        goods.append(good)
    df = pd.DataFrame(rows, columns=["Metric", "Strategy", bench, "Difference"])
    st.subheader(f"Strategy vs {bench}")
    styled = df.style.apply(
        lambda c: [f"color:{_status_color(g)}; font-weight:600;" for g in goods] if c.name == "Difference" else [""] * len(c)
    ).hide(axis="index")
    st.dataframe(styled, width="stretch", height=38 * (len(df) + 1))


def _db_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "src" / "db" / "stockml.duckdb"


def _wal_aware_mtime(db_path: Path) -> float:
    """Latest mtime across the .duckdb file and its .wal — the live daemon
    holds its connection open indefinitely and never checkpoints, so recent
    writes only bump the .wal file's mtime, not the main file's."""
    wal = Path(f"{db_path}.wal")
    mtimes = [db_path.stat().st_mtime]
    if wal.exists():
        mtimes.append(wal.stat().st_mtime)
    return max(mtimes)


@st.cache_data(show_spinner=False)
def _db_meta(db_path_str: str, mtime: float) -> dict:
    """Min/max price date and the real ticker list, read from a snapshot
    (never the live file directly — it may be lock-held by the live process).

    `mtime` should reflect both the main .duckdb file and its .wal (see
    `_wal_aware_mtime`) — the live daemon holds its connection open
    indefinitely and never checkpoints, so recent writes sit in the .wal
    file and the main file's own mtime alone won't bust this cache."""
    tmp_dir = tempfile.mkdtemp(prefix="stockml_dashboard_")
    try:
        tmp_db = str(Path(tmp_dir) / "snapshot.duckdb")
        shutil.copy2(db_path_str, tmp_db)
        wal_path = f"{db_path_str}.wal"
        if Path(wal_path).exists():
            shutil.copy2(wal_path, f"{tmp_db}.wal")
        conn = duckdb.connect(tmp_db, read_only=True)
        min_date, max_date = conn.execute(
            "SELECT MIN(CAST(date AS DATE)), MAX(CAST(date AS DATE)) FROM price_daily_full"
        ).fetchone()
        tickers = [r[0] for r in conn.execute(
            "SELECT DISTINCT stock_id FROM price_daily_full ORDER BY 1"
        ).fetchall()]
        # last date where >=90% of tickers have prices (stale tickers must not define coverage)
        last_common = conn.execute(
            "SELECT MAX(d) FROM (SELECT CAST(date AS DATE) AS d, COUNT(DISTINCT stock_id) AS n "
            "FROM price_daily_full GROUP BY 1) WHERE n >= 0.9 * (SELECT COUNT(DISTINCT stock_id) FROM price_daily_full)"
        ).fetchone()[0] or max_date
        conn.close()
        return {"min_date": min_date, "max_date": max_date, "tickers": tickers, "last_common": last_common}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# price data is USD, so capital is converted to USD; rate = units of currency per 1 USD
# (offline fallbacks) with a default initial capital per currency
_FX_FALLBACK = {"USD": 1.0, "KRW": 1400.0, "EUR": 0.92, "JPY": 150.0, "GBP": 0.78}
_CAPITAL_DEFAULT = {"USD": 100_000.0, "KRW": 100_000_000.0, "EUR": 100_000.0, "JPY": 15_000_000.0, "GBP": 100_000.0}


@st.cache_data(ttl=3600, show_spinner=False)
def _fx_rate(currency: str) -> float:
    if currency == "USD":
        return 1.0
    try:
        import yfinance as yf
        px = yf.Ticker(f"{currency}=X").history(period="5d")["Close"].dropna()
        return float(px.iloc[-1])
    except Exception:
        return _FX_FALLBACK[currency]


@st.cache_data(show_spinner="Running backtest...")
def _run_backtest(
    db_path: str,
    start: str,
    end: str,
    rebalance: str,
    top_n: int,
    capital: float,
    benchmark: str,
    slippage: float,
    commission: float,
    rebalance_days: int = 0,
) -> dict:
    """Thin wrapper around BacktestEngine + PerformanceCalculator returning
    a plain, cache-friendly dict (no dataclasses) of everything the GUI needs."""
    engine = BacktestEngine(
        db_path=db_path,
        start_date=start,
        end_date=end,
        rebalance_freq=f"{rebalance_days}d" if rebalance == "custom" else rebalance,
        top_n=top_n,
        initial_capital=capital,
        benchmark_ticker=benchmark,
        slippage_pct=slippage,
        commission_pct=commission,
    )
    result = engine.run()
    calc = PerformanceCalculator(
        nav=result.nav_series,
        benchmark=result.benchmark_series,
        trade_log=result.trade_log,
        regime_history=result.regime_history,
    )
    report = calc.compute()
    bench = result.benchmark_series
    bench_stats = PerformanceCalculator(
        nav=bench, benchmark=bench, trade_log=[], regime_history=None,
    ).compute().to_dict()
    return {
        "bench_stats": bench_stats,
        "nav": result.nav_series,
        "benchmark": result.benchmark_series,
        "trade_log": result.trade_log,
        "regime_history": result.regime_history,
        "stats": report.to_dict(),
        "round_trips": calc.get_round_trips(),
        "rebalance_dates": result.rebalance_dates,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_capital": result.initial_capital,
        "benchmark_ticker": result.benchmark_ticker,
        "data_end_warning": result.data_end_warning,
    }


def _to_result(d: dict) -> BacktestResult:
    return BacktestResult(
        nav_series=d["nav"],
        benchmark_series=d["benchmark"],
        trade_log=d["trade_log"],
        regime_history=d["regime_history"],
        rebalance_dates=d["rebalance_dates"],
        start_date=d["start_date"],
        end_date=d["end_date"],
        initial_capital=d["initial_capital"],
        benchmark_ticker=d["benchmark_ticker"],
    )


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def main() -> None:
    st.set_page_config(page_title="Stock_ML Backtester", layout="wide")
    st.markdown(_CARD_CSS, unsafe_allow_html=True)
    st.title("📊 Stock_ML Backtest & Performance Evaluation")

    db_path = _db_path()
    if not db_path.exists():
        st.error(f"DuckDB not found at {db_path}. Run the data pipeline first.")
        return

    meta = _db_meta(str(db_path), _wal_aware_mtime(db_path))

    with st.sidebar:
        st.header("Backtest Parameters")
        start = st.date_input("Start", value=meta["min_date"],
                               min_value=meta["min_date"], max_value=meta["max_date"])
        end = st.date_input("End", value=meta["last_common"],
                             min_value=meta["min_date"], max_value=meta["max_date"])
        if end > meta["last_common"]:
            st.warning(f"Most tickers have no prices after {meta['last_common']}; results beyond that are unreliable.")
        rebalance = st.selectbox("Rebalance", ["weekly", "biweekly", "monthly", "quarterly", "custom"], index=2)
        rebalance_days = int(st.number_input("Every N days", min_value=1, max_value=365, value=21)) \
            if rebalance == "custom" else 0
        top_n = st.number_input("Top N", min_value=1, max_value=max(len(meta["tickers"]), 1), value=10)
        currency = st.selectbox("Currency", list(_FX_FALLBACK))
        capital = st.number_input(f"Initial Capital ({currency})", min_value=1.0,
                                  value=_CAPITAL_DEFAULT[currency], step=_CAPITAL_DEFAULT[currency] / 100,
                                  key=f"capital_{currency}")
        fx = st.number_input(f"{currency} per 1 USD", min_value=0.0001, value=_fx_rate(currency), format="%.4f",
                             key=f"fx_{currency}", disabled=currency == "USD")
        st.caption(f"Simulated in USD: ${capital / fx:,.0f}")
        default_bench = next((t for t in ("SPY", "QQQ", "AAPL") if t in meta["tickers"]),
                             meta["tickers"][0] if meta["tickers"] else "")
        benchmark = st.selectbox("Benchmark", options=meta["tickers"],
                                  index=meta["tickers"].index(default_bench) if default_bench in meta["tickers"] else 0)

        with st.expander("Advanced"):
            slippage = st.number_input("Slippage %", min_value=0.0, value=0.1, step=0.01) / 100.0
            commission = st.number_input("Commission %", min_value=0.0, value=0.015, step=0.001) / 100.0

        run_clicked = st.button("Run Backtest", type="primary")

    if run_clicked:
        result_dict = _run_backtest(
            str(db_path), start.isoformat(), end.isoformat(), rebalance,
            int(top_n), float(capital / fx), benchmark, slippage, commission, rebalance_days,
        )
        st.session_state["backtest_result"] = result_dict

    result_dict = st.session_state.get("backtest_result")
    if not result_dict:
        st.info("Set parameters in the sidebar and click **Run Backtest**.")
        return

    stats = result_dict["stats"]
    if result_dict.get("data_end_warning"):
        st.warning(result_dict["data_end_warning"])
    result = _to_result(result_dict)

    # ── Metrics row ──────────────────────────────────────────────────────
    win_rate = stats["win_rate"]
    _metric_cards_row([
        ("Total Return", f"{stats['total_return']:.2%}", stats["total_return"] >= 0),
        ("CAGR", f"{stats['cagr']:.2%}", stats["cagr"] >= 0),
        ("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}" if pd.notna(stats["sharpe_ratio"]) else "n/a",
         None if pd.isna(stats["sharpe_ratio"]) else stats["sharpe_ratio"] >= 1),
        ("Max Drawdown", f"{stats['max_drawdown']:.2%}" if pd.notna(stats["max_drawdown"]) else "n/a",
         None if pd.isna(stats["max_drawdown"]) else stats["max_drawdown"] >= -0.15),
        ("Win Rate", f"{win_rate:.2%}" if pd.notna(win_rate) else "n/a",
         None if pd.isna(win_rate) else win_rate >= 0.5),
    ])
    _comparison_table(stats, result_dict["bench_stats"], result_dict["benchmark_ticker"])

    # ── Equity curve + drawdown ─────────────────────────────────────────
    st.plotly_chart(BacktestReporter.build_figure(result), width="stretch")

    # ── Full stats table ────────────────────────────────────────────────
    st.subheader("Performance Statistics")
    rows = BacktestReporter.stats_rows(stats)
    stats_df = pd.DataFrame(rows, columns=["Metric", "Value", "_good"])
    is_section = stats_df["Value"] == ""

    def _style_stats(row: pd.Series) -> list:
        if is_section.loc[row.name]:
            return ["font-weight:700; background: rgba(128,128,128,0.08)"] * 2
        good = stats_df.loc[row.name, "_good"]
        color = _status_color(good)
        return ["", f"color:{color}; font-variant-numeric: tabular-nums;" if good is not None else ""]

    styled = (
        stats_df[["Metric", "Value"]]
        .style.apply(_style_stats, axis=1)
        .hide(axis="index")
    )
    st.dataframe(styled, width="stretch", height=38 * (len(stats_df) + 1))

    # ── Regime attribution ──────────────────────────────────────────────
    regime_keys = sorted(k for k in stats if k.startswith("regime_") and k.endswith("_cumulative_return"))
    if regime_keys:
        st.subheader("Regime Attribution")
        labels = [k.replace("regime_", "").replace("_cumulative_return", "").replace("_", " ") for k in regime_keys]
        values = [stats[k] for k in regime_keys]
        import plotly.graph_objects as go
        regime_fig = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color=[REGIME_SOLID_COLORS.get(l, _NEUTRAL) for l in labels],
            text=[f"{v:.2%}" for v in values], textposition="outside",
        ))
        regime_fig.update_layout(
            template="plotly_white", height=320, showlegend=False,
            yaxis_title="Cumulative Return", margin=dict(t=20, b=20),
        )
        regime_fig.update_yaxes(tickformat=".0%", gridcolor="#e1e0d9")
        st.plotly_chart(regime_fig, width="stretch")

    # ── Trade Log ────────────────────────────────────────────────────────
    st.subheader("Trade Log")
    trade_df = pd.DataFrame(result_dict["trade_log"])
    if not trade_df.empty:
        stock_filter = st.multiselect("Filter by stock", sorted(trade_df["stock_id"].unique()))
        action_filter = st.multiselect("Filter by action", ["BUY", "SELL"])
        filtered = trade_df
        if stock_filter:
            filtered = filtered[filtered["stock_id"].isin(stock_filter)]
        if action_filter:
            filtered = filtered[filtered["action"].isin(action_filter)]
        st.dataframe(
            filtered, width="stretch",
            column_config={
                "date": st.column_config.DateColumn("Date"),
                "stock_id": st.column_config.TextColumn("Stock"),
                "action": st.column_config.TextColumn("Action"),
                "shares": st.column_config.NumberColumn("Shares", format="%.2f"),
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "exec_price": st.column_config.NumberColumn("Exec Price", format="$%.2f"),
                "gross_value": st.column_config.NumberColumn("Gross Value", format="$%.0f"),
                "commission": st.column_config.NumberColumn("Commission", format="$%.2f"),
                "net_cash_flow": st.column_config.NumberColumn("Net Cash Flow", format="$%.0f"),
            },
        )
        st.download_button("Download Trade Log CSV", data=_csv_bytes(trade_df),
                            file_name="trade_log.csv", mime="text/csv")
    else:
        st.write("No trades recorded.")

    # ── Investment Records (round-trip P&L) ─────────────────────────────
    st.subheader("Investment Records (Round-Trip P&L)")
    rt_df = pd.DataFrame(result_dict["round_trips"])
    if not rt_df.empty:
        rt_display = rt_df.rename(columns={
            "stock_id": "Stock", "buy_date": "Buy Date", "sell_date": "Sell Date",
            "shares": "Shares", "buy_price": "Buy Price", "sell_price": "Sell Price",
            "pnl_pct": "P&L %", "holding_days": "Holding Days",
        })
        styled_rt = (
            rt_display.style
            .map(lambda v: f"color:{_GOOD if v >= 0 else _CRITICAL}; font-weight:600;", subset=["P&L %"])
            .format({
                "Buy Price": "${:.2f}", "Sell Price": "${:.2f}",
                "Shares": "{:.2f}", "P&L %": "{:+.2%}", "Holding Days": "{:.0f}",
            })
        )
        st.dataframe(styled_rt, width="stretch")
        st.download_button("Download Investment Records CSV", data=_csv_bytes(rt_df),
                            file_name="investment_records.csv", mime="text/csv")
    else:
        st.write("No completed round trips.")

    # ── NAV download ─────────────────────────────────────────────────────
    nav_df = pd.DataFrame({
        "date": result_dict["nav"].index,
        "strategy_nav": result_dict["nav"].values,
    })
    st.download_button("Download NAV CSV", data=_csv_bytes(nav_df),
                        file_name="nav.csv", mime="text/csv")


if __name__ == "__main__":
    main()
