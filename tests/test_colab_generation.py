"""Colab Qwen3-VL-2B runner: frozen evidence, corpus bundle, resume and safety checks.

Everything runs on the toy corpus with the mock generator: no network, GPU or model.
"""

from __future__ import annotations

import ast
import json
import random
import re
import tarfile
from pathlib import Path

import pytest
import yaml

from mmrag import colab_generation as cg
from mmrag.colab_smoke_test import MODEL, REVISION
from mmrag.config import load_generation
from mmrag.data.corpus import write_jsonl
from mmrag.eval import generation_eval as ge
from mmrag.generate.vlm import MockVLM
from mmrag.types import Query

ROOT = Path(__file__).parents[1]
COMMIT = "0" * 40
OOD = [Query(f"ood{i}", f"out of domain question {i}", None, ("Text",)) for i in range(3)]


def baseline_runs(corpus, out: Path, tamper: bool = False) -> Path:
    """SmolVLM-shaped run files built with the CLI's own sampling and ordering."""
    write_jsonl(corpus.root / "ood_queries.jsonl", (q.to_dict() for q in OOD))
    queries = ge.sample_queries(corpus.load_queries(), cg.N_QUERIES, cg.SEED)
    ood = random.Random(cg.SEED).sample(corpus.load_queries("ood_queries"), len(OOD))
    qrels = corpus.load_qrels()
    pages = corpus.page_ids
    rows = []
    for q in queries:
        gold = qrels[q.query_id]
        rows.append(("answerable", q, gold, list(gold) + [p for p in pages if p not in gold][:2]))
        rows.append(("missing_evidence", q, gold, [p for p in pages if p not in gold][:3]))
    rows += [("out_of_domain", q, {}, pages[:3]) for q in ood]
    out.mkdir(parents=True, exist_ok=True)
    for s in cg.SYSTEMS:
        recs = [
            {"key": f"{c}|{q.query_id}", "condition": c, "query_id": q.query_id, "gold": g,
             "provided": [] if s == "closed_book" else p, "system": f"smolvlm_500m_{s}"}
            for c, q, g, p in rows
            if not (s == "closed_book" and c == "missing_evidence")
        ]  # fmt: skip
        if tamper and s == "image_rag":
            recs[0]["provided"] = recs[0]["provided"][::-1]
        write_jsonl(out / f"smolvlm_500m_{s}.jsonl", recs)
    return out


def mock_configs(tmp_path: Path) -> dict[str, Path]:
    """The real Qwen configs with the generator swapped for the mock backend."""
    out = {}
    for s, path in cg.default_configs().items():
        cfg = yaml.safe_load(path.read_text())
        cfg["vlm"] = {"backend": "mock", "model": "mock", "max_new_tokens": 160, "device": "cpu"}
        out[s] = tmp_path / path.name
        out[s].write_text(yaml.safe_dump(cfg))
    return out


@pytest.fixture()
def bundle(corpus, tmp_path):
    """Frozen evidence, packaged corpus and the extracted corpus root."""
    runs = baseline_runs(corpus, tmp_path / "baseline")
    evidence = tmp_path / "frozen" / cg.EVIDENCE_NAME
    cg.freeze_evidence(runs, evidence)
    summary = cg.package_corpus(corpus.root, evidence, tmp_path / "pkg")
    with tarfile.open(tmp_path / "pkg" / summary["archive"]) as tar:
        tar.extractall(tmp_path / "extract", filter="data")
    return {
        "summary": summary,
        "root": tmp_path / "extract" / cg.ARCHIVE_ROOT,
        "evidence": evidence,
    }


def test_qwen_configs_are_pinned_and_match_smolvlm_protocol():
    for s in cg.SYSTEMS:
        qwen = load_generation(ROOT / f"configs/generation/qwen3vl_2b_{s}.yaml")
        smol = load_generation(ROOT / f"configs/generation/smolvlm_500m_{s}.yaml")
        assert (qwen.vlm.model, qwen.vlm.revision) == (MODEL, REVISION)
        assert qwen.vlm.dtype == "float16" and not qwen.vlm.load_in_4bit
        assert qwen.vlm.max_new_tokens == smol.vlm.max_new_tokens == 160
        assert qwen.evidence == smol.evidence
    cg.load_configs(cg.default_configs())


def test_freeze_evidence_rejects_disagreeing_systems(corpus, tmp_path):
    runs = baseline_runs(corpus, tmp_path / "baseline", tamper=True)
    with pytest.raises(ValueError, match="evidence differs"):
        cg.freeze_evidence(runs, tmp_path / "e.jsonl")


def test_freeze_evidence_never_overwrites(corpus, tmp_path):
    runs = baseline_runs(corpus, tmp_path / "baseline")
    cg.freeze_evidence(runs, tmp_path / "e.jsonl")
    with pytest.raises(FileExistsError):
        cg.freeze_evidence(runs, tmp_path / "e.jsonl")


def test_build_examples_reproduces_frozen_evidence(corpus, bundle):
    rows = cg.read_records(bundle["evidence"])
    examples = cg.build_examples(corpus, rows)
    assert [e.key for e in examples] == [r["key"] for r in rows]
    assert [[h.page_id for h in e.hits] for e in examples] == [r["provided"] for r in rows]
    assert len(cg.system_examples("closed_book", examples)) == len(examples) - 2
    with pytest.raises(ValueError, match="question set"):
        cg.build_examples(corpus, rows[::-1])


def test_corpus_archive_is_reproducible_and_verified(corpus, bundle, tmp_path):
    again = cg.package_corpus(corpus.root, bundle["evidence"], tmp_path / "pkg2")
    assert again["sha256"] == bundle["summary"]["sha256"]
    info = cg.verify_corpus(bundle["root"])
    assert info["corpus_fingerprint"] == corpus.fingerprint()
    page = next((bundle["root"] / corpus.name / "images").iterdir())
    page.write_bytes(page.read_bytes() + b"x")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        cg.verify_corpus(bundle["root"])


def test_verify_corpus_rejects_extra_files(bundle):
    (bundle["root"] / "stray.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="file set differs"):
        cg.verify_corpus(bundle["root"])


def run_mock(bundle, out, configs, calls=None, **kw):
    def factory(cfg):
        vlm = MockVLM(cfg)
        if calls is not None:
            original = vlm.generate

            def counted(*a, **k):
                calls.append(1)
                return original(*a, **k)

            vlm.generate = counted
        return vlm

    return cg.run(
        out, bundle["root"], COMMIT, configs=configs, vlm_factory=factory, cuda_required=False, **kw
    )


def test_run_is_resumable_and_never_recomputes(bundle, tmp_path):
    out, configs = tmp_path / "drive" / "qwen", mock_configs(tmp_path)
    calls: list[int] = []
    run_mock(bundle, out, configs, calls)
    n_examples = json.loads((out / "provenance.json").read_text())["experiment"]["n_examples"]
    assert len(calls) == sum(n_examples.values())
    first = {s: (out / "runs" / f"qwen3vl_2b_{s}.jsonl").read_bytes() for s in cg.SYSTEMS}

    calls.clear()
    run_mock(bundle, out, configs, calls)  # everything complete: nothing regenerated
    assert calls == []

    # Simulate an interruption: drop a completion marker and the last record.
    name = "qwen3vl_2b_image_rag"
    (out / "runs" / f"{name}.complete.json").unlink()
    lines = first["image_rag"].splitlines(keepends=True)
    (out / "runs" / f"{name}.jsonl").write_bytes(b"".join(lines[:-1]) + lines[-1][:10])
    run_mock(bundle, out, configs, calls)
    assert len(calls) == 1
    assert list((out / "runs").glob(f"{name}.partial-*"))
    # Records match except wall-clock timings; earlier lines are untouched bytes.
    again = (out / "runs" / f"{name}.jsonl").read_bytes().splitlines(keepends=True)
    assert again[:-1] == lines[:-1]
    strip = [
        {k: v for k, v in json.loads(x).items() if k != "timings_s"} for x in (again[-1], lines[-1])
    ]
    assert strip[0] == strip[1]
    sums = (out / "CHECKSUMS.sha256").read_text()
    assert "provenance.json" in sums and f"runs/{name}.jsonl" in sums
    sessions = cg.read_records(out / "sessions.jsonl")
    assert [s.get("status") for s in sessions if "ended" in s] == ["completed"] * 3


def test_run_refuses_changed_experiment_source_or_results(bundle, tmp_path):
    out, configs = tmp_path / "qwen", mock_configs(tmp_path)
    run_mock(bundle, out, configs, systems=("closed_book",))
    with pytest.raises(ValueError, match="started from"):
        cg.run(out, bundle["root"], "1" * 40, configs=configs,
               vlm_factory=MockVLM, cuda_required=False)  # fmt: skip
    for path in configs.values():  # a setting the protocol check allows to vary
        cfg = yaml.safe_load(path.read_text())
        cfg["vlm"]["dtype"] = "float32"
        path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="different experiment"):
        run_mock(bundle, out, configs)
    with pytest.raises(ValueError, match="outside results"):
        run_mock(bundle, ROOT / "results" / "generation" / "qwen_test", mock_configs(tmp_path))
    assert not (ROOT / "results" / "generation" / "qwen_test").exists()


def test_configs_must_match_smolvlm_protocol(tmp_path):
    configs = mock_configs(tmp_path)
    cfg = yaml.safe_load(configs["text_rag"].read_text())
    cfg["evidence"]["max_pages"] = 2
    configs["text_rag"].write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="differs from smolvlm_500m_text_rag"):
        cg.load_configs(configs)


def test_run_detects_completed_file_tampering(bundle, tmp_path):
    out, configs = tmp_path / "qwen", mock_configs(tmp_path)
    run_mock(bundle, out, configs, systems=("text_rag",))
    with (out / "runs" / "qwen3vl_2b_text_rag.jsonl").open("a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="changed after"):
        run_mock(bundle, out, configs, systems=("text_rag",))


def test_memory_check_writes_separate_checksummed_report(bundle, tmp_path):
    out = tmp_path / "mem"
    status = cg.memory_check(
        out, bundle["root"], COMMIT, configs=mock_configs(tmp_path), n_heaviest=2,
        vlm_factory=MockVLM, cuda_required=False,
    )  # fmt: skip
    assert status == 0
    (report_path,) = out.glob("memory_check/*/report.json")
    report = json.loads(report_path.read_text())
    assert report["status"] == "passed" and not report["errors"]
    assert [e["evidence_mode"] for e in report["examples"]] == ["image_text"] * 2 + [
        "image",
        "text",
    ]
    assert all(e["max_new_tokens"] == 160 for e in report["examples"])
    image_sizes = report["examples"][0]["image_sizes_after_resize"]
    assert len(image_sizes) == 3 and all(max(s) <= 1024 for s in image_sizes)
    digest = (report_path.parent / "report.sha256").read_text().split()[0]
    assert digest == cg.sha256_file(report_path)
    assert not (out / "runs").exists()


def test_score_writes_new_results_and_never_overwrites(corpus, bundle, tmp_path):
    out, configs = tmp_path / "qwen", mock_configs(tmp_path)
    run_mock(bundle, out, configs)
    results, runs = tmp_path / "results", tmp_path / "runs"
    written = cg.score(out, corpus.root, results, runs, nli=False)
    assert len(written) == 2 * len(cg.SYSTEMS)
    summary = json.loads((results / corpus.name / "qwen3vl_2b_image_rag.json").read_text())
    assert summary["provenance"]["source_commit"] == COMMIT
    assert summary["evidence_source"].startswith("frozen from smolvlm_500m")
    with pytest.raises(FileExistsError):
        cg.score(out, corpus.root, results, runs, nli=False)


def test_notebook_is_unexecuted_and_gated():
    nb = json.loads((ROOT / "notebooks/colab_qwen3vl_2b_generation.ipynb").read_text())
    assert nb["nbformat"] == 4
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for cell in code:
        assert cell["execution_count"] is None and cell["outputs"] == []
        ast.parse("".join(cell["source"]))
    source = "".join("".join(c["source"]) for c in code)
    assert "APPROVE_MEMORY_CHECK = False" in source and "APPROVE_FULL_RUN = False" in source
    assert not re.search(r"hf_[A-Za-z0-9]{30,}", source), "token-like string in notebook"
