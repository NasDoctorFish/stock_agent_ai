"""
Point-in-time data access. One availability rule for every fundamental:
a quarter with fiscal_date F is visible from F + lag calendar days (the DB has
no filing dates, so this is an assumption, never a fact). Universe = current
config tickers: a survivor set, NOT a historical membership list.
"""
import duckdb
import numpy as np
import pandas as pd

from factor_ic import ROOT, load_prices

DB = str(ROOT / "data/research/research.duckdb")
LAGS = (45, 60, 90)
FUND_COLS = {"eps": "Diluted EPS", "revenue": "Total Revenue", "op_income": "Operating Income",
             "net_income": "Net Income"}
SURVIVOR_NOTE = "universe = current config tickers (survivor-biased); no delisted names"


def availability(fiscal_date: pd.Series, lag: int) -> pd.Series:
    return pd.to_datetime(fiscal_date) + pd.Timedelta(days=lag)


def load_universe(db: str = DB) -> pd.DataFrame:
    return load_prices(db)


def load_quarterly(db: str = DB) -> pd.DataFrame:
    sel = ", ".join(f'"{v}" {k}' for k, v in FUND_COLS.items())
    q = duckdb.connect(db, read_only=True).execute(
        f"SELECT stock_id, CAST(fiscal_date AS DATE) fd, {sel} FROM fundamentals_q_full WHERE fiscal_date IS NOT NULL"
    ).fetchdf()
    q["fd"] = pd.to_datetime(q["fd"])
    return q.drop_duplicates(["stock_id", "fd"]).sort_values(["stock_id", "fd"])


def ttm_panel(q: pd.DataFrame, col: str, index: pd.DatetimeIndex, columns, lag: int) -> pd.DataFrame:
    """Trailing-4-quarter sum of `col` as seen on each date. NaN if <4 quarters or a gap > 400d.
    Missing stays NaN (never filled)."""
    out = pd.DataFrame(np.nan, index=index, columns=columns)
    for s, g in q.dropna(subset=[col]).groupby("stock_id"):
        if s not in out.columns:
            continue
        g = g.set_index("fd")[col]
        t = g.rolling(4).sum()
        t[(g.index.to_series().diff(3).dt.days > 400).values] = np.nan
        avail = pd.Series(t.values, index=availability(pd.Series(t.index), lag).values).dropna()
        if not avail.empty:
            out[s] = avail.reindex(index.union(avail.index)).ffill().reindex(index)
    return out
