"""Dataset version + provenance: content hash of the research inputs, written next to results."""
import hashlib
import json
import subprocess

import pandas as pd

from factor_ic import ROOT


def _h(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


def build(px: pd.DataFrame, q: pd.DataFrame, lag: int) -> dict:
    try:
        git = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git = "unknown"
    m = {"prices_hash": _h(px), "fundamentals_hash": _h(q.reset_index(drop=True)), "eps_lag_days": lag,
         "prices_shape": list(px.shape), "fund_rows": len(q),
         "source": "data/research/research.duckdb (yfinance auto_adjust; fundamentals yfinance quarterly)",
         "git_commit": git}
    m["version"] = hashlib.sha256(json.dumps({k: m[k] for k in ("prices_hash", "fundamentals_hash", "eps_lag_days")},
                                              sort_keys=True).encode()).hexdigest()[:12]
    return m


def write(m: dict) -> str:
    out = ROOT / "reports/research" / f"manifest_{m['version']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    return str(out)
