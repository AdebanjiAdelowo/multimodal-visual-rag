"""Page-retrieval metrics against graded relevance judgements.

nDCG uses linear gain (gain = grade) with a log2 discount, the trec_eval
`ndcg_cut` definition that the ViDoRe leaderboards report through pytrec_eval.
"""

from __future__ import annotations

import math

KS = (1, 3, 5, 10)


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    """Fraction of the relevant pages found in the top k (page/evidence recall)."""
    gold = {p for p, g in rel.items() if g > 0}
    if not gold:
        return float("nan")
    return len(gold & set(ranked[:k])) / len(gold)


def hit_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    """1 if at least one relevant page is in the top k."""
    return float(any(rel.get(p, 0) > 0 for p in ranked[:k]))


def full_recall_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    """1 if every relevant page is in the top k (complete evidence set retrieved)."""
    gold = {p for p, g in rel.items() if g > 0}
    return float(bool(gold) and gold <= set(ranked[:k]))


def mrr_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    for i, p in enumerate(ranked[:k], 1):
        if rel.get(p, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    dcg = sum(rel.get(p, 0) / math.log2(i + 1) for i, p in enumerate(ranked[:k], 1))
    ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else float("nan")


def per_query_metrics(
    ranked: list[str], rel: dict[str, int], ks: tuple[int, ...] = KS
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, rel, k)
        out[f"hit@{k}"] = hit_at_k(ranked, rel, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, rel, k)
    kmax = max(ks)
    out[f"mrr@{kmax}"] = mrr_at_k(ranked, rel, kmax)
    out[f"full_recall@{kmax}"] = full_recall_at_k(ranked, rel, kmax)
    return out
