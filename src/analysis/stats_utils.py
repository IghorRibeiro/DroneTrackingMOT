"""Statistical helpers shared by the analysis scripts.

All aggregate metrics are recomputed from SUMMED counts (never averaged over sequences). Resampling uses seed 2026.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.mot_eval import COUNT_COLS, idf1_from_counts, mota_from_counts, totals


def fisher_ci(r: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95 % confidence interval of a Pearson correlation (Fisher z-transform)."""
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    zr, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
    return float(np.tanh(zr - z * se)), float(np.tanh(zr + z * se))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def corr_row(x, y) -> dict:
    """Pearson (with Fisher CI) and Spearman correlation between two vectors."""
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y, n = x[m], y[m], int(m.sum())
    if n < 4:
        return dict(n=n, pearson_r=np.nan, pearson_p=np.nan, ci95_lo=np.nan, ci95_hi=np.nan, spearman_rho=np.nan, spearman_p=np.nan)
    r, p = stats.pearsonr(x, y)
    rho, ps = stats.spearmanr(x, y)
    lo, hi = fisher_ci(r, n)
    return dict(n=n, pearson_r=r, pearson_p=p, ci95_lo=lo, ci95_hi=hi, spearman_rho=rho, spearman_p=ps)


def paired_bootstrap(base: pd.DataFrame, other: pd.DataFrame, nboot: int, seed: int) -> dict:
    """Paired bootstrap over SEQUENCES (same indices in both systems); MOTA / IDF1 are recomputed from the summed counts."""
    seqs = base.index.intersection(other.index)
    a, b = base.loc[seqs], other.loc[seqs]
    n = len(seqs)
    idx = np.random.default_rng(seed).integers(0, n, size=(nboot, n))
    agg = lambda df: {k: df[k].to_numpy(float)[idx].sum(axis=1) for k in COUNT_COLS}
    A, B = agg(a), agg(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_mota = (mota_from_counts(B) - mota_from_counts(A)) * 100
        d_idf1 = (idf1_from_counts(B) - idf1_from_counts(A)) * 100
        rel_ids = np.where(A["ids"] > 0, (A["ids"] - B["ids"]) / A["ids"] * 100, np.nan)
    ta, tb = totals(a), totals(b)
    point = {"d_mota_pp": (mota_from_counts(tb) - mota_from_counts(ta)) * 100,
             "d_idf1_pp": (idf1_from_counts(tb) - idf1_from_counts(ta)) * 100,
             "ids_reduction_pct": (ta["ids"] - tb["ids"]) / ta["ids"] * 100 if ta["ids"] else np.nan}
    boot = {"d_mota_pp": d_mota, "d_idf1_pp": d_idf1, "ids_reduction_pct": rel_ids}
    out = {}
    for k, v in boot.items():
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        out[k] = dict(point=point[k], boot_mean=float(np.nanmean(v)), ci95_lo=float(lo), ci95_hi=float(hi))
    out["_n"] = n
    return out


def wilcoxon_row(delta: np.ndarray, lower_is_better: bool = False) -> dict:
    """Two-sided Wilcoxon signed-rank test on per-sequence differences, plus win/loss/tie counts."""
    from scipy import stats
    d = np.round(np.asarray(delta, float), 9)
    d = d[np.isfinite(d)]
    improved = int((d < 0).sum() if lower_is_better else (d > 0).sum())
    worsened = int((d > 0).sum() if lower_is_better else (d < 0).sum())
    res = dict(n=len(d), improved=improved, worsened=worsened, tied=int((d == 0).sum()),
               median_delta=float(np.median(d)) if len(d) else np.nan, W=np.nan, p=np.nan)
    if (d != 0).sum() >= 1:
        try:
            w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            res["W"], res["p"] = float(w.statistic), float(w.pvalue)
        except ValueError:
            pass
    return res


def select_config(table: pd.DataFrame, criterion: str = "paper") -> str:
    """Pick a configuration. ``table``: index = configuration; columns MOTA, IDF1 (fractions) and ids.

    'paper'  : MOTA desc, IDF1 desc, IDs asc on values rounded to one decimal (rule of the original figure script);
    'strict' : the same lexicographic rule without rounding.
    """
    t = table.copy()
    if criterion == "paper":
        t["m"], t["f"] = (t["MOTA"] * 100).round(1), (t["IDF1"] * 100).round(1)
    else:
        t["m"], t["f"] = t["MOTA"], t["IDF1"]
    t = t.sort_values(["m", "f", "ids"], ascending=[False, False, True], kind="mergesort")
    return t.index[0]
