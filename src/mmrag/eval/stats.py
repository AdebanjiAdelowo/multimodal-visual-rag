"""Bootstrap confidence intervals and paired significance tests over queries."""

from __future__ import annotations

import numpy as np


def mean_ci(
    values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """Mean with a percentile bootstrap confidence interval. NaNs are dropped."""
    a = np.asarray(values, dtype=np.float64)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    return {
        "mean": float(a.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(a.size),
    }


def paired_bootstrap(
    a: list[float], b: list[float], n_boot: int = 5000, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap of mean(b) - mean(a) over the same queries.

    Returns the observed difference, its 95% interval and a two-sided p-value
    (twice the smaller tail mass of the bootstrap distribution around zero).
    """
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("paired samples must align")
    keep = ~(np.isnan(x) | np.isnan(y))
    d = (y - x)[keep]
    if d.size == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {
        "diff": float(d.mean()),
        "lo": float(np.quantile(boots, 0.025)),
        "hi": float(np.quantile(boots, 0.975)),
        "p": float(min(1.0, p)),
        "n": int(d.size),
    }


def auroc(scores: list[float], labels: list[int]) -> float:
    """Area under the ROC curve via the rank-sum statistic (ties get average rank)."""
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
