import numpy as np
import pytest

from mmrag.config import BM25Cfg, RetrievalCfg, load_retrieval
from mmrag.eval.retrieval_eval import evaluate_retrieval
from mmrag.index.dense import max_pool_by_page
from mmrag.index.late import maxsim
from mmrag.retrieve.fusion import ranks_from_scores, rrf, zscore_fusion
from mmrag.retrieve.pipeline import IndexRegistry, RetrievalPipeline
from mmrag.text import chunk_words, tokenize


def bm25_pipeline(corpus, **kw):
    cfg = RetrievalCfg(name="bm25", retriever=BM25Cfg(), **kw)
    return RetrievalPipeline(cfg, IndexRegistry(corpus))


def test_tokenize_and_chunk():
    assert tokenize("The for-loop in Python 3.12!") == ["for-loop", "python", "3.12"]
    chunks = chunk_words(" ".join(str(i) for i in range(10)), size=4, overlap=1)
    assert chunks[0] == "0 1 2 3" and chunks[1].startswith("3 ")
    assert chunks[-1].split()[-1] == "9"
    assert chunk_words("") == [""]
    with pytest.raises(ValueError):
        chunk_words("a b", size=2, overlap=2)


def test_fusion():
    a, b = np.array([3.0, 2.0, 1.0]), np.array([0.1, 0.9, 0.5])
    assert ranks_from_scores(a).tolist() == [1, 2, 3]
    fused = rrf([a, b], [1.0, 1.0], k=60)
    assert int(np.argmax(fused)) == 1  # ranks (2, 1) beat (1, 3) and (3, 2)
    assert int(np.argmax(rrf([a, b], [1.0, 0.0]))) == 0  # zero weight removes a component
    z = zscore_fusion([a, np.array([1.0, 1.0, 1.0])], [1.0, 1.0])
    assert int(np.argmax(z)) == 0  # a constant component contributes nothing


def test_max_pool_and_maxsim():
    pooled = max_pool_by_page(np.array([0.1, 0.9, 0.4]), np.array([0, 0, 1]), 2)
    assert pooled.tolist() == pytest.approx([0.9, 0.4])
    q = np.eye(2, dtype=np.float32)
    pages = np.zeros((2, 2, 2), dtype=np.float32)
    pages[0, 0], pages[0, 1] = [1, 0], [0, 1]
    pages[1, 0] = [1, 0]
    mask = np.array([[True, True], [True, False]])
    assert maxsim(q, pages, mask).tolist() == pytest.approx([2.0, 1.0])


def test_bm25_search_and_exclusion(corpus):
    pipe = bm25_pipeline(corpus)
    hits = pipe.search("variable name digit", k=3)
    assert hits[0].page_id == "doc-p0" and len(hits) == 3
    # Withholding the gold page must remove it from the ranking entirely.
    hits = pipe.search("variable name digit", k=5, exclude={"doc-p0"})
    assert "doc-p0" not in [h.page_id for h in hits] and len(hits) == 4


def test_evaluate_retrieval_on_toy_corpus(corpus):
    res = evaluate_retrieval(bm25_pipeline(corpus), corpus.load_queries(), corpus.load_qrels(), 5)
    m = res["summary"]["metrics"]
    assert m["all"]["recall@1"]["mean"] == 1.0
    assert m["visual"]["n_queries"] == 1 and m["text_only"]["n_queries"] == 1
    assert "type:chart" in m


def test_shipped_configs_parse():
    from pathlib import Path

    from mmrag.config import load_generation

    root = Path(__file__).parents[1] / "configs"
    names = [load_retrieval(p).name for p in sorted((root / "retrieval").glob("*.yaml"))]
    assert len(names) == len(set(names)) >= 10
    for p in (root / "retrieval").glob("*.yaml"):
        assert load_retrieval(p).name == p.stem
    for p in (root / "generation").glob("*.yaml"):
        assert load_generation(p).name == p.stem


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError):
        RetrievalCfg.model_validate({"name": "x", "retriever": {"type": "bm25", "typo": 1}})
