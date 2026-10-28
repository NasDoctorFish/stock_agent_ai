"""Backtest service layer: wraps BacktestEngine / PerformanceCalculator for the API."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest.engine import BacktestEngine, _load_benchmark_tickers  # noqa: E402
from backtest.performance import PerformanceCalculator  # noqa: E402
from backtest.portfolio import StrategyConfig  # noqa: E402

from .schemas import BacktestParams, Strategy  # noqa: E402

CURRENCIES = {"USD": 1.0, "KRW": 1400.0, "EUR": 0.92, "JPY": 150.0, "GBP": 0.78}  # offline fallbacks


def db_path() -> str:
    """STOCKML_DB overrides the live DB (use a copy when the live one is locked)."""
    return os.environ.get("STOCKML_DB") or str(SRC / "db" / "stockml.duckdb")


def open_engine(path=None, start="2000-01-01", end="2000-01-01", **kw) -> BacktestEngine:
    return BacktestEngine(db_path=path or db_path(), start_date=start, end_date=end, **kw)


def clean(o):
    """JSON-safe: NaN/inf -> None, numpy/pandas scalars -> python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def to_config(s: Strategy | None) -> StrategyConfig | None:
    if s is None:
        return None
    r = s.sell_rules
    return StrategyConfig(equity_by_regime=s.equity_by_regime.model_dump(),
                          stop_loss_pct=r.stop_loss_pct, take_profit_pct=r.take_profit_pct,
                          max_holding_days=r.max_holding_days, planned_sell_date=r.planned_sell_date)


def run_backtest(p: BacktestParams, path: str | None = None) -> dict:
    freq = f"{p.rebalance_days or 21}d" if p.rebalance == "custom" else p.rebalance
    eng = open_engine(path, p.start, p.end, rebalance_freq=freq, top_n=p.top_n,
                      initial_capital=p.capital / p.fx,  # prices are USD
                      benchmark_ticker=p.benchmark, slippage_pct=p.slippage_pct / 100.0,
                      commission_pct=p.commission_pct / 100.0, strategy=to_config(p.strategy))
    res = eng.run()
    calc = PerformanceCalculator(nav=res.nav_series, benchmark=res.benchmark_series,
                                 trade_log=res.trade_log, regime_history=res.regime_history)
    stats = calc.compute().to_dict()
    b = res.benchmark_series
    bench_stats = PerformanceCalculator(nav=b, benchmark=b, trade_log=[], regime_history=None).compute().to_dict() \
        if len(b) > 1 else {}
    nav = res.nav_series * p.fx  # back to display currency
    bench = (b * p.fx).reindex(nav.index).ffill()
    dd = res.nav_series / res.nav_series.cummax() - 1
    trades = [{**t, "date": str(pd.Timestamp(t["date"]).date())} for t in res.trade_log]
    rts = [{**r, "buy_date": str(pd.Timestamp(r["buy_date"]).date()), "sell_date": str(pd.Timestamp(r["sell_date"]).date())}
           for r in calc.get_round_trips()]
    out = {
        "stats": stats, "bench_stats": bench_stats,
        "nav": [{"date": str(d.date()), "strategy": v, "benchmark": bench.get(d)} for d, v in nav.items()],
        "drawdown": [{"date": str(d.date()), "value": v} for d, v in dd.items()],
        "regimes": [{"date": str(pd.Timestamp(d).date()), "label": l} for d, l in res.regime_history.items()],
        "trades": trades, "round_trips": rts,
        "currency": p.currency,
    }
    if res.data_end_warning:
        out["warning"] = res.data_end_warning
    return clean(out)


def db_meta(path: str | None = None) -> dict:
    eng = open_engine(path)
    c = eng._conn
    last_common, _ = eng._check_data_coverage()
    mx = c.execute("SELECT MAX(CAST(date AS DATE)) FROM price_daily_full").fetchone()[0]
    tickers = [r[0] for r in c.execute("SELECT DISTINCT stock_id FROM price_daily_full ORDER BY 1").fetchall()]
    return {"data_end": last_common or str(mx), "data_max": str(mx), "tickers": tickers,
            "benchmarks": sorted(_load_benchmark_tickers() & set(tickers))}
