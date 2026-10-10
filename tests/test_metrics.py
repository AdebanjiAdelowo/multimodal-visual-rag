import math

import pytest

from mmrag.eval import generation_metrics as gm
from mmrag.eval import retrieval_metrics as rm
from mmrag.eval.stats import auroc, mean_ci, paired_bootstrap

REL = {"a": 2, "b": 1, "c": 1}


def test_recall_hit_and_full_recall():
    ranked = ["a", "x", "b", "y", "c"]
    assert rm.recall_at_k(ranked, REL, 1) == pytest.approx(1 / 3)
    assert rm.recall_at_k(ranked, REL, 3) == pytest.approx(2 / 3)
    assert rm.hit_at_k(["x", "y"], REL, 2) == 0.0
    assert rm.full_recall_at_k(ranked, REL, 4) == 0.0
    assert rm.full_recall_at_k(ranked, REL, 5) == 1.0


def test_mrr():
    assert rm.mrr_at_k(["x", "y", "b"], REL, 10) == pytest.approx(1 / 3)
    assert rm.mrr_at_k(["x", "y", "b"], REL, 2) == 0.0


def test_ndcg_linear_gain_matches_hand_computation():
    ranked = ["b", "a", "x"]
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    idcg = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert rm.ndcg_at_k(ranked, REL, 3) == pytest.approx(dcg / idcg)
    assert rm.ndcg_at_k(["a", "b", "c"], REL, 3) == pytest.approx(1.0)


def test_metrics_undefined_without_relevant_pages():
    assert math.isnan(rm.recall_at_k(["a"], {}, 5))
    assert math.isnan(rm.ndcg_at_k(["a"], {}, 5))


def test_token_f1_and_rouge():
    assert gm.token_f1("The cat sat.", "a cat sat") == pytest.approx(1.0)
    assert gm.token_f1("dog", "cat") == 0.0
    assert gm.token_f1("", "cat") == 0.0
    assert gm.rouge_l("w x y z", "w y z") == pytest.approx(2 * (3 / 4) * 1 / (3 / 4 + 1))
    assert gm.token_recall("cat", "cat sat") == pytest.approx(0.5)


def test_boolean_accuracy():
    assert gm.boolean_accuracy("No, it is not.", "No, never.") == 1.0
    assert gm.boolean_accuracy("Yes.", "No, never.") == 0.0
    assert gm.boolean_accuracy("It depends", "No.") == 0.0
    assert gm.boolean_accuracy("Yes", "Paris") is None


def test_citation_scores():
    s = gm.citation_scores(["p1", "p9"], {"p1", "p2", "p3"}, ["p1", "p2", "p9"])
    assert s["citation_precision"] == 0.5
    assert s["citation_recall"] == 0.5  # p3 was never shown, so it is not reachable
    empty = gm.citation_scores([], {"p1"}, ["p7"])
    assert empty["citation_precision"] is None and empty["citation_recall"] is None


def test_mean_ci_and_paired_bootstrap():
    ci = mean_ci([0.0, 1.0, 1.0, float("nan")])
    assert ci["n"] == 3 and ci["lo"] <= ci["mean"] <= ci["hi"]
    same = paired_bootstrap([0.2, 0.4, 0.6], [0.2, 0.4, 0.6])
    assert same["diff"] == 0.0
    better = paired_bootstrap([0.1] * 30, [0.9] * 30)
    assert better["diff"] == pytest.approx(0.8) and better["p"] < 0.05


def test_auroc():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auroc([0.5, 0.5], [0, 1]) == 0.5
    assert math.isnan(auroc([0.1, 0.2], [1, 1]))
