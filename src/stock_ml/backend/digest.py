"""Compact stats digest. The AI only ever sees this, never raw data."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .service import SRC, open_engine  # noqa: F401  (SRC puts src/ on sys.path)

MAX_BYTES = 10_000
_STAT_KEYS = ["total_return", "cagr", "annualized_volatility", "sharpe_ratio", "sortino_ratio",
              "max_drawdown", "beta", "alpha", "info_ratio", "excess_return", "win_rate", "n_trades"]


def _r(x, n=3):
    try:
        x = float(x)
        return None if not np.isfinite(x) else round(x, n)
    except Exception:
        return None


def _pick(stats: Optional[dict]) -> Optional[dict]:
    return {k: _r(stats[k]) for k in _STAT_KEYS if stats and k in stats and not isinstance(stats[k], (dict, list, str))} or None


def build_digest(db_path: Optional[str] = None, as_of: Optional[str] = None,
                 backtest: Optional[dict] = None) -> dict:
    """backtest = response dict from service.run_backtest (only stats/bench_stats are used)."""
    eng = open_engine(db_path, as_of or "2000-01-01", as_of or "2000-01-01")
    c = eng._conn
    last_common, _ = eng._check_data_coverage()
    end = pd.Timestamp(as_of or last_common)
    nxt = end + pd.Timedelta(days=1)  # engine uses date < rebal_date

    # --- regime: now + month-end history
    label, score = eng._compute_regime(nxt)
    hist = []
    for k in range(1, 7):
        d = end - pd.DateOffset(months=k)
        l, s = eng._compute_regime(d)
        hist.append({"date": str(d.date()), "regime": l, "score": s})

    # --- macro
    m = c.execute(f"""SELECT index_name, CAST(date AS DATE) d, value_norm FROM macro_index_full
                      WHERE CAST(date AS DATE) <= '{end.date()}' AND value_norm IS NOT NULL
                        AND CAST(date AS DATE) >= '{(end - pd.DateOffset(months=8)).date()}' ORDER BY d""").fetchdf()
    m["d"] = pd.to_datetime(m["d"])
    macro = {}
    for name, g in m.groupby("index_name"):
        s = g.set_index("d")["value_norm"]
        last = s.iloc[-1]

        def chg(days):
            old = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
            return _r((last / old.iloc[-1] - 1) * 100, 1) if len(old) and old.iloc[-1] else None
        macro[name] = {"last": _r(last, 2), "asof": str(s.index[-1].date()), "chg30_pct": chg(30), "chg90_pct": chg(90)}

    # --- stocks
    from backtest.engine import _load_benchmark_tickers
    bench = _load_benchmark_tickers()
    p = c.execute(f"""SELECT CAST(date AS DATE) d, stock_id, close FROM price_daily_full
                      WHERE CAST(date AS DATE) <= '{end.date()}' AND CAST(date AS DATE) >= '{(end - pd.DateOffset(months=13)).date()}'
                        AND close > 0""").fetchdf()
    p["d"] = pd.to_datetime(p["d"])
    wide = p.pivot_table(index="d", columns="stock_id", values="close", aggfunc="last").sort_index()
    stale = [t for t in wide.columns if wide[t].last_valid_index() is None or wide[t].last_valid_index() < wide.index[-5]]
    stocks = wide[[t for t in wide.columns if t not in bench and t not in stale]].ffill()
    ret = stocks.pct_change(fill_method=None)
    mom3 = stocks.iloc[-1] / stocks.iloc[max(0, len(stocks) - 63)] - 1
    mom12 = stocks.iloc[-1] / stocks.iloc[0] - 1
    vol = ret.iloc[-63:].std() * np.sqrt(252)
    above50 = (stocks.iloc[-1] > stocks.iloc[-50:].mean())
    rank = mom12.sort_values(ascending=False).dropna()

    def row(t):
        return {"t": t, "m3": _r(mom3[t], 2), "m12": _r(mom12[t], 2), "vol": _r(vol[t], 2)}
    mkt = {}
    for b in sorted(bench & set(wide.columns)):
        mkt[b] = {"m3": _r(wide[b].iloc[-1] / wide[b].dropna().iloc[max(0, len(wide[b].dropna()) - 63)] - 1, 3)}

    cov = c.execute("SELECT MIN(CAST(date AS DATE)), MAX(CAST(date AS DATE)), COUNT(DISTINCT stock_id) FROM price_daily_full").fetchone()
    digest = {
        "as_of": str(end.date()),
        "regime": {"label": label, "score": score, "scale": "-6..+6", "history_monthly": hist},
        "macro": macro,
        "market": mkt,
        "stocks": {"n": int(len(stocks.columns)), "top_momentum": [row(t) for t in rank.index[:7]],
                   "bottom_momentum": [row(t) for t in rank.index[-5:]],
                   "median_vol_ann": _r(vol.median(), 3),
                   "breadth": {"pct_above_50d": _r(above50.mean(), 2), "pct_positive_3m": _r((mom3 > 0).mean(), 2)}},
        "coverage": {"first": str(cov[0]), "last": str(cov[1]), "last_common": last_common,
                     "tickers": int(cov[2]), "stale_tickers": stale[:10], "n_stale": len(stale)},
    }
    if backtest:
        digest["backtest"] = {"strategy": _pick(backtest.get("stats")), "benchmark": _pick(backtest.get("bench_stats"))}
    # hard cap: drop least important macro series until it fits
    while len(json.dumps(digest, separators=(",", ":"))) > MAX_BYTES and digest["macro"]:
        digest["macro"].pop(sorted(digest["macro"], key=lambda k: abs(digest["macro"][k]["chg90_pct"] or 0))[0])
    return digest


def digest_json(d: dict) -> str:
    return json.dumps(d, separators=(",", ":"))
