import json

from mmrag.data.corpus import Corpus
from mmrag.data.subset import make_pool
from mmrag.eval.report import build_report
from mmrag.eval.retrieval_eval import evaluate_retrieval, save_run
from tests.test_retrieval import bm25_pipeline


def test_report_lists_only_what_was_run(corpus, tmp_path):
    res = evaluate_retrieval(bm25_pipeline(corpus), corpus.load_queries(), corpus.load_qrels(), 5)
    res["summary"]["hardware"] = {"machine": "test"}
    save_run(res, tmp_path / "results/retrieval/toy", tmp_path / "runs/retrieval/toy")
    md = build_report(tmp_path / "results", tmp_path / "runs")
    assert "`bm25`" in md and "Retrieval: `toy`" in md
    assert "Generation" not in md and "dense" not in md
    json.loads((tmp_path / "results/retrieval/toy/bm25.json").read_text())


def test_pool_keeps_every_relevant_page(corpus, tmp_path):
    pool = Corpus(make_pool(corpus, tmp_path / "pool", n_queries=2, n_pages=3, seed=1))
    assert len(pool) == 3
    gold = {p for rel in pool.load_qrels().values() for p in rel}
    assert gold <= set(pool.page_ids)
    assert pool.load_text("ocr_tesseract").keys() == set(pool.page_ids)
    assert pool.image(pool.page_ids[0]).size == (320, 240)
