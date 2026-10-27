"""
Follow-ups on valuation: (1) EPS lag 45/60/90, (2) mom-only vs valuation-only vs mom+valuation,
all on the SAME val-eligible dates/rows, (3) turnover-buffered long-short.
Usage: poetry run python src/research/val_followup.py
"""
import numpy as np
import pandas as pd

import ablation as A
from familiarity import load_eps
from factor_ic import load_prices

LAGS, BUFFER = (45, 60, 90), 0.4  # held names stay while in top/bottom 40%; new entries need 20%
MOM = ["mom_12_1", "rev_1m"]


def buffered_spread(O, col, cost_bps=25):
    L = S = None
    rows = []
    for t, g in O.groupby("date"):
        g = g.dropna(subset=["ret"])
        if len(g) < A.QUINT_MIN:
            continue
        r = g.set_index("stock")[col].rank(pct=True)
        n = len(g) // 5
        def pick(prev, sign):
            x = r if sign > 0 else 1 - r
            keep = [s for s in (prev or []) if x.get(s, 0) >= 1 - BUFFER]
            new = [s for s in x.sort_values(ascending=False).index if s not in keep]
            return (keep + new)[:n]
        nl, ns = pick(L, 1), pick(S, -1)
        ret = g.set_index("stock").ret
        to = 0.0 if L is None else (1 - len(set(nl) & set(L)) / n) + (1 - len(set(ns) & set(S)) / n)
        rows.append((t, ret[nl].mean() - ret[ns].mean(), to))
        L, S = nl, ns
    d = pd.DataFrame(rows, columns=["date", "gross", "to"])
    return d.gross.mean(), d.to.mean(), (d.gross - d.to * 2 * cost_bps / 1e4).mean()


if __name__ == "__main__":
    px, eps = load_prices(A.DB), load_eps(A.DB)
    out = []
    for lag in LAGS:
        P, sets, ff, vf = A.build(px, eps, lag)
        sets = {"mom": MOM, "mom_val": MOM + vf, "val_only": vf, "base": sets["base"], "base_val": sets["valuation"]}
        O, _ = A.walk_forward(P, sets)
        ok = O.groupby("date").has_val.sum() >= A.QUINT_MIN
        Ov = O[O.date.isin(ok[ok].index) & O.has_val]
        print(f"\n=== EPS lag {lag}d: {Ov.date.nunique()} common dates ===")
        for n in sets:
            ics = {a: A.ic_series(Ov, f"{n}|{a}").mean() for a in A.ALPHAS}
            g, to, net = buffered_spread(Ov, f"{n}|{A.HEAD}")
            print(f"{n:9s} IC by alpha {({a: round(v, 4) for a, v in ics.items()})}  buffered: gross {g:+.4f} turn {to:.2f} net25 {net:+.4f}")
        d = A.ic_series(Ov, f"mom_val|{A.HEAD}") - A.ic_series(Ov, f"mom|{A.HEAD}")
        lo, hi = A.boot_ci(d)
        print(f"mom_val - mom dIC {d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}]")
