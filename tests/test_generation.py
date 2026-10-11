from mmrag.config import EvidenceCfg, GenerationCfg, VLMCfg
from mmrag.eval.generation_eval import build_examples, run_system, score_records, summarize
from mmrag.generate.prompts import ABSTAIN, build_prompt, parse_answer
from mmrag.generate.rag import RAGPipeline, select_evidence
from mmrag.generate.vlm import MockVLM
from mmrag.types import Evidence, Query, ScoredPage
from tests.test_retrieval import bm25_pipeline

EV = [Evidence("P1", "doc-p0", 1.0, "alpha"), Evidence("P2", "doc-p2", 0.5, "beta")]


def test_parse_structured_answer():
    a = parse_answer("ANSWER: No, it cannot.\nCITATIONS: P1, P2, P7", EV)
    assert a.text == "No, it cannot." and not a.abstained
    assert a.cited_page_ids == ["doc-p0", "doc-p2"] and a.invalid_citations == ["P7"]


def test_parse_abstention_and_free_text():
    assert parse_answer(ABSTAIN, EV).abstained
    assert parse_answer("ANSWER: insufficient evidence\nCITATIONS: P1", EV).cited_page_ids == []
    assert parse_answer("", EV).abstained
    free = parse_answer("It cannot start with a digit [P2].", EV)
    assert not free.abstained and free.cited_labels == ["P2"]
    assert parse_answer("Just an answer.", EV).cited_labels == []


def test_select_evidence_respects_budget():
    hits = [ScoredPage(f"p{i}", 1.0 - i / 10) for i in range(6)]
    ev = select_evidence(hits, EvidenceCfg(max_pages=2, max_chars_per_page=3), {"p0": "abcdef"})
    assert [e.label for e in ev] == ["P1", "P2"] and ev[0].text == "abc" and ev[1].text == ""


def test_prompt_never_exposes_page_ids():
    content = build_prompt("why?", EV, "text")
    blob = " ".join(p["text"] for p in content)
    assert "[P1]" in blob and "doc-p0" not in blob
    assert build_prompt("why?", [], "closed_book")[0]["text"].count("ANSWER") == 1


def gen_cfg(mode="text"):
    return GenerationCfg(
        name="mock", vlm=VLMCfg(backend="mock", model="mock"), evidence=EvidenceCfg(mode=mode)
    )


def test_rag_pipeline_cites_retrieved_page(corpus):
    rag = RAGPipeline(corpus, MockVLM(), gen_cfg(), bm25_pipeline(corpus))
    res = rag.answer("can a variable name start with a digit")
    assert res.evidence[0].page_id == "doc-p0"
    assert res.answer.cited_page_ids == ["doc-p0"] and "retrieval" in res.timings_s
    closed = RAGPipeline(corpus, MockVLM(), gen_cfg("closed_book")).answer("anything")
    assert closed.answer.abstained and closed.evidence == []


def test_image_mode_builds_image_parts(corpus):
    rag = RAGPipeline(corpus, MockVLM(), gen_cfg("image"), bm25_pipeline(corpus))
    assert rag.answer("revenue chart").answer.abstained  # the mock reads no pixels


def test_generation_eval_conditions_and_resume(corpus, tmp_path):
    pipe = bm25_pipeline(corpus)
    ood = [Query("x0", "parental leave policy", answerable=False)]
    examples = build_examples(pipe, corpus.load_queries(), corpus.load_qrels(), ood, depth=3)
    assert [e.condition for e in examples].count("missing_evidence") == 2
    for e in examples:
        if e.condition == "missing_evidence":
            assert not set(e.gold) & {h.page_id for h in e.hits}

    rag = RAGPipeline(corpus, MockVLM(), gen_cfg())
    path = tmp_path / "run.jsonl"
    first = run_system(rag, "mock", examples, path)
    again = run_system(rag, "mock", examples, path)
    assert first == again and len(path.read_text().splitlines()) == len(examples)

    records = score_records(first, None, corpus.load_text("ocr_tesseract"))
    summary = summarize(records, rag, "mock")
    ans = summary["conditions"]["answerable"]
    assert ans["gold_in_context"]["mean"] == 1.0 and ans["citation_precision"]["mean"] == 1.0
    # The mock always answers from whatever it is given, so it never abstains.
    assert summary["conditions"]["missing_evidence"]["unsupported_answer_rate"]["mean"] == 1.0


def test_report_compares_evidence_modes_of_one_generator(corpus, tmp_path):
    import json

    from mmrag.eval.report import generation_section

    pipe = bm25_pipeline(corpus)
    examples = build_examples(pipe, corpus.load_queries(), corpus.load_qrels(), [], depth=3)
    res, runs = tmp_path / "results", tmp_path / "runs"
    res.mkdir()
    runs.mkdir()
    for mode in ("closed_book", "text"):
        system = f"mock_{mode}" if mode == "closed_book" else "mock_text_rag"
        rag = RAGPipeline(corpus, MockVLM(), gen_cfg(mode))
        exs = [e for e in examples if mode == "text" or e.condition == "answerable"]
        recs = score_records(run_system(rag, system, exs, runs / f"{system}.jsonl"), None, {})
        (res / f"{system}.json").write_text(json.dumps(summarize(recs, rag, system)))
        (runs / f"{system}.scored.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    md = generation_section(res, runs)
    assert "`mock_closed_book` -> `mock_text_rag`" in md and "direct model vs text RAG" in md
    assert "'missing_evidence': 2" in md  # header describes the fullest design
    assert "image RAG" not in md  # comparisons need both systems to exist
