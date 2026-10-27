"""
Ablation: do price-familiarity / relative-valuation features add Rank IC beyond
the momentum+reversal baseline? Same walk-forward ridge as ridge_wf.py
(sample every H=21d, train on dates <= i-2 => 21d label embargo, expanding).

Sets: base | familiarity | valuation | combined (+ *_only). Missing valuation
values are rank-neutral (0) so the long price history still trains the model.
Valuation rows exist only from ~2025 (EPS history), so the valuation
comparison is evaluated on val-eligible dates only, on val-complete rows.

Usage: poetry run python src/research/ablation.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from factor_ic import H, ROOT, load_prices
from familiarity import load_eps, price_familiarity, valuation
from ridge_wf import features as base_features, rank_norm

DB = str(ROOT / "data/research/research.duckdb")
ALPHAS, HEAD = [10, 100, 1000], 100
START, MIN_TRAIN, QUINT_MIN = 756, 36, 30
COSTS_BPS = (10, 25)
rng = np.random.default_rng(0)


def build(px: pd.DataFrame, eps: pd.DataFrame, lag: int = 45):
    bf, ff, vf = base_features(px), price_familiarity(px), valuation(px, eps, lag)
    allf = {**bf, **ff, **vf}
    fwd = px.shift(-H) / px - 1
    rows = []
    for t in px.index[START::H]:
        d = pd.DataFrame({k: rank_norm(v.loc[t].dropna()) for k, v in allf.items()})
        d["y"] = rank_norm(fwd.loc[t].dropna())
        d["ret"] = fwd.loc[t]
        d["has_val"] = d["ey"].notna() & d["pe_inv_rel"].notna()
        d = d.dropna(subset=list(bf) + list(ff) + ["y"])
        d["date"] = t
        rows.append(d.reset_index(names="stock"))
    P = pd.concat(rows, ignore_index=True)
    sets = {"base": list(bf), "familiarity": list(bf) + list(ff), "valuation": list(bf) + list(vf),
            "combined": list(bf) + list(ff) + list(vf), "familiarity_only": list(ff), "valuation_only": list(vf)}
    P[list(vf)] = P[list(vf)].fillna(0.0)
    return P, sets, list(ff), list(vf)


def walk_forward(P: pd.DataFrame, sets: dict):
    dates = sorted(P.date.unique())
    out, coefs = [], {}
    for i in range(MIN_TRAIN + 2, len(dates)):
        tr, te = P[P.date <= dates[i - 2]], P[P.date == dates[i]].copy()
        for name, X in sets.items():
            for a in ALPHAS:
                m = Ridge(alpha=a).fit(tr[X], tr["y"])
                te[f"{name}|{a}"] = m.predict(te[X])
                if a == HEAD:
                    coefs.setdefault(name, []).append(m.coef_)
        out.append(te)
    return pd.concat(out), {k: pd.Series(np.mean(v, 0), index=sets[k]) for k, v in coefs.items()}


def ic_series(O: pd.DataFrame, col: str) -> pd.Series:
    return O.groupby("date").apply(lambda g: spearmanr(g[col], g["y"])[0])


def spread(O: pd.DataFrame, col: str) -> pd.DataFrame:
    prevL = prevS = None
    rows = []
    for t, g in O.groupby("date"):
        g = g.dropna(subset=["ret"])
        if len(g) < QUINT_MIN:
            continue
        q = pd.qcut(g[col].rank(method="first"), 5, labels=False)
        L, S = set(g.stock[q == 4]), set(g.stock[q == 0])
        gross = g.ret[q == 4].mean() - g.ret[q == 0].mean()
        to = 0.0 if prevL is None else (1 - len(L & prevL) / len(L)) + (1 - len(S & prevS) / len(S))
        rows.append((t, gross, to))
        prevL, prevS = L, S
    return pd.DataFrame(rows, columns=["date", "gross", "turnover"]).set_index("date")


def tstat(s: pd.Series) -> float:
    return s.mean() / (s.std() / np.sqrt(len(s)))


def boot_ci(d: pd.Series, n: int = 2000):
    m = [d.sample(len(d), replace=True, random_state=int(rng.integers(1 << 30))).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def report(O: pd.DataFrame, names: list, title: str, lines: list) -> dict:
    ics = {n: ic_series(O, f"{n}|{HEAD}") for n in names}
    base = ics["base"]
    tab = []
    for n in names:
        s = ics[n]
        yr = s.groupby(pd.to_datetime(s.index).year).mean()
        d = s - base
        lo, hi = boot_ci(d) if n != "base" else (np.nan, np.nan)
        sp = spread(O, f"{n}|{HEAD}")
        net = {c: sp.gross - sp.turnover * 2 * c / 1e4 for c in COSTS_BPS}
        tab.append({
            "set": n, "IC": s.mean(), "ICIR": s.mean() / s.std(), "ICIR_ann": s.mean() / s.std() * np.sqrt(12),
            "t": tstat(s), "hit": (s > 0).mean(), "yrs>0": (yr > 0).mean(),
            "dIC": d.mean(), "dIC_lo": lo, "dIC_hi": hi,
            "gross_mo": sp.gross.mean(), "turn": sp.turnover.mean(),
            f"net{COSTS_BPS[0]}": net[COSTS_BPS[0]].mean(), f"net{COSTS_BPS[1]}": net[COSTS_BPS[1]].mean(),
            f"sharpe_net{COSTS_BPS[1]}": net[COSTS_BPS[1]].mean() / net[COSTS_BPS[1]].std() * np.sqrt(12),
        })
    T = pd.DataFrame(tab).set_index("set")
    byyr = pd.DataFrame({n: ics[n].groupby(pd.to_datetime(ics[n].index).year).mean() for n in names})
    alph = pd.DataFrame({n: {a: ic_series(O, f"{n}|{a}").mean() for a in ALPHAS} for n in names}).T
    lines += [f"## {title}", f"{len(base)} OOS dates {pd.Timestamp(base.index[0]).date()} -> {pd.Timestamp(base.index[-1]).date()}",
              f"head alpha={HEAD}; spread = top-minus-bottom quintile, equal weight, 21d; cost = one-way bps on traded notional",
              "```", T.round(4).to_string(), "", "mean IC by year", byyr.round(3).to_string(),
              "", "mean IC by alpha", alph.round(4).to_string(), "```"]
    return T.to_dict("index")


def partial_ic(P: pd.DataFrame, feats: list, mask=None) -> pd.DataFrame:
    """Per-date IC of each new feature alone and after residualising on mom_12_1 + rev_1m."""
    res = {f: [] for f in feats}
    raw = {f: [] for f in feats}
    for t, g in (P if mask is None else P[mask]).groupby("date"):
        if len(g) < QUINT_MIN:
            continue
        Z = np.c_[np.ones(len(g)), g[["mom_12_1", "rev_1m"]].values]
        proj = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
        ry = proj(g["y"].values)
        for f in feats:
            if g[f].std() == 0:
                continue
            raw[f].append(spearmanr(g[f], g["y"])[0])
            res[f].append(np.corrcoef(proj(g[f].values), ry)[0, 1])
    rows = {f: {"IC_alone": np.mean(raw[f]), "t_alone": tstat(pd.Series(raw[f])),
                "IC_partial": np.mean(res[f]), "t_partial": tstat(pd.Series(res[f])), "n": len(res[f])}
            for f in feats if len(res[f]) > 2}
    return pd.DataFrame(rows).T


def main() -> None:
    px = load_prices(DB)
    eps = load_eps(DB)
    P, sets, ff, vf = build(px, eps)
    L: list = ["# Price familiarity & relative valuation ablation", "",
               f"Universe: {px.shape[1]} current config tickers (survivor bias), prices 2010->2026, first sample at day {START}.",
               "P/B not computed: no book-equity columns in fundamentals_q_full. EPS lag 45d (no filing dates).", ""]
    O, coefs = walk_forward(P, sets)
    price_names = ["base", "familiarity", "familiarity_only"]
    price_T = report(O, price_names, "A. Price-only ablation (full sample)", L)

    ok = O.groupby("date").has_val.sum() >= QUINT_MIN
    Ov = O[O.date.isin(ok[ok].index) & O.has_val]
    val_T = None
    if Ov.date.nunique() >= 6:
        val_T = report(Ov, ["base", "valuation", "combined", "familiarity", "valuation_only"],
                       "B. Valuation ablation (val-eligible dates/rows only; exploratory, short sample)", L)
    else:
        L.append(f"## B. Valuation ablation\nonly {Ov.date.nunique()} val-eligible OOS dates: skipped.")

    L += ["## C. Incremental signal beyond mom_12_1 + rev_1m (per-date cross-sectional, full panel)", "```",
          partial_ic(P, ff).round(4).to_string(), "```"]
    if P.has_val.sum():
        L += ["valuation features (val-eligible rows)", "```", partial_ic(P, vf, P.has_val).round(4).to_string(), "```"]
    L += ["## D. Mean ridge coefficients (alpha=100)", "```",
          pd.DataFrame({k: v for k, v in coefs.items() if k in ("familiarity", "combined")}).round(3).to_string(), "```"]

    # placebo: shuffle new features across stocks within each date -> dIC should be ~0
    Pp = P.copy()
    for t, idx in Pp.groupby("date").groups.items():
        for c in ff + vf:
            Pp.loc[idx, c] = rng.permutation(Pp.loc[idx, c].values)
    Op, _ = walk_forward(Pp, {"base": sets["base"], "combined": sets["combined"]})
    d = ic_series(Op, f"combined|{HEAD}") - ic_series(Op, f"base|{HEAD}")
    L += ["## E. Placebo (new features shuffled across stocks)", f"dIC combined-base = {d.mean():+.4f} (t {tstat(d):.2f}); expect ~0", ""]

    dF = price_T["familiarity"]
    inc = dF["dIC_lo"] > 0
    L += ["## Verdict",
          f"- Familiarity vs base: dIC {dF['dIC']:+.4f}, 95% CI [{dF['dIC_lo']:+.4f}, {dF['dIC_hi']:+.4f}] -> "
          + ("incremental power beyond momentum/reversal." if inc else
             ("CI excludes 0 on the negative side: features HURT OOS." if dF["dIC_hi"] < 0 else
              "no statistically significant incremental power (CI includes 0).")),
          "- Check section C signs: a negative partial IC for price percentile/z-score means high-vs-history prices predicted lower returns; "
          "a positive one means the opposite. Low price was NOT assumed to mean undervalued.",
          "- Caveats: survivor universe of 56 large caps (58 in price table = 56 + SPY,QQQ benchmarks); valuation window ~1.5y; overlapping nothing (H=21 sampling) but n is small."]
    out = ROOT / "reports/price_familiarity_ablation.md"
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
