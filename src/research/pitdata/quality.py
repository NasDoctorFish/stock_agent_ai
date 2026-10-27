"""Data-quality diagnostics: missing data, duplicates, suspicious prices."""
import pandas as pd


def price_report(px: pd.DataFrame, jump: float = 0.4) -> dict:
    r = px.pct_change()
    return {
        "shape": px.shape,
        "range": (str(px.index.min().date()), str(px.index.max().date())),
        "nonpositive": int((px <= 0).sum().sum()),
        "na_frac": float(px.isna().mean().mean()),
        "jumps_gt": {"thr": jump, "n": int((r.abs() > jump).sum().sum()),
                     "where": [(str(d.date()), s) for d, s in r.abs().stack()[lambda x: x > jump].index[:10]]},
        "dup_index": int(px.index.duplicated().sum()),
    }


def fundamentals_report(q: pd.DataFrame) -> dict:
    cols = [c for c in q.columns if c not in ("stock_id", "fd")]
    n_q = q.groupby("stock_id").size()
    return {
        "rows": len(q), "stocks": q.stock_id.nunique(),
        "fiscal_range": (str(q.fd.min().date()), str(q.fd.max().date())),
        "na_frac": {c: round(float(q[c].isna().mean()), 3) for c in cols},
        "quarters_per_stock": {"min": int(n_q.min()), "median": float(n_q.median()), "max": int(n_q.max())},
        "dup_keys": int(q.duplicated(["stock_id", "fd"]).sum()),
        "filing_dates": "absent: availability = fiscal_date + assumed lag",
    }
