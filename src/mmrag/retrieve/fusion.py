"""Rank and score fusion for hybrid retrieval."""

from __future__ import annotations

import numpy as np


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """1-based rank of every item (higher score = better). Ties broken by index."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def rrf(score_lists: list[np.ndarray], weights: list[float], k: int = 60) -> np.ndarray:
    """Weighted reciprocal rank fusion (Cormack et al., 2009).

    Rank-based, so it needs no calibration between BM25 scores and cosine
    similarities, which live on unrelated scales.
    """
    fused = np.zeros(len(score_lists[0]), dtype=np.float64)
    for scores, w in zip(score_lists, weights, strict=True):
        fused += w / (k + ranks_from_scores(scores))
    return fused.astype(np.float32)


def zscore_fusion(score_lists: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted sum of per-retriever standardised scores."""
    fused = np.zeros(len(score_lists[0]), dtype=np.float64)
    for scores, w in zip(score_lists, weights, strict=True):
        finite = np.where(np.isfinite(scores), scores, np.nan)
        mu, sd = np.nanmean(finite), np.nanstd(finite)
        z = (finite - mu) / sd if sd > 0 else np.zeros_like(finite)
        fused += w * np.nan_to_num(z, nan=-10.0)
    return fused.astype(np.float32)
