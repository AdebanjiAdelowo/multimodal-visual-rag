"""Resumable Qwen3-VL-2B generation on Colab, controlled against the SmolVLM-500M study.

The completed SmolVLM-500M study fixed the questions (60 answerable, the same 60
with gold pages withheld, 30 out-of-domain; seed 13) and the retrieved pages for
each of them. Retrieval is not repeated here: the pages SmolVLM saw are frozen
from its run files and given to Qwen3-VL unchanged, so evidence selection is
identical by construction. Generation reuses the production harness
(`generation_eval.run_system`), and scoring runs locally on the CPU, as it did
for SmolVLM.

Stages (all writes are append-only or refuse to overwrite):
  freeze-evidence  local       SmolVLM run JSONL -> evidence JSONL
  package-corpus   local       corpus + evidence -> deterministic tar + MANIFEST.json
  verify-corpus    anywhere    every file against MANIFEST.json
  memory-check     Colab GPU   the heaviest real prompts; peak VRAM and latency
  run              Colab GPU   four systems, append-only JSONL on Drive
  score            local       NLI scoring on CPU, then result JSON files
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import tarfile
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mmrag.colab_smoke_test import MODEL, REVISION, write_report
from mmrag.config import GenerationCfg, VLMCfg, load_generation
from mmrag.data.corpus import Corpus
from mmrag.eval import generation_eval as ge
from mmrag.types import ScoredPage

ROOT = Path(__file__).resolve().parents[2]
STUDY = "qwen3vl_2b_vs_smolvlm_500m"
SYSTEMS = ("closed_book", "text_rag", "image_rag", "image_text_rag")
EVIDENCE_SYSTEMS = ("text_rag", "image_rag", "image_text_rag")
BASELINE = "smolvlm_500m"
CANDIDATE = "qwen3vl_2b"
SEED, N_QUERIES, N_OOD = 13, 60, 30
FAITH_SOURCE = "parser_markdown"
ARCHIVE_ROOT = "mmrag_corpus"
EVIDENCE_NAME = f"{BASELINE}_evidence.jsonl"
MEMORY_HEADROOM = 0.80  # peak reserved VRAM must stay below this fraction of the total
PACKAGES = ("torch", "torchvision", "transformers", "accelerate", "huggingface-hub", "pillow")
JUNK = {".DS_Store", "__pycache__"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def default_configs() -> dict[str, Path]:
    return {s: ROOT / "configs" / "generation" / f"{CANDIDATE}_{s}.yaml" for s in SYSTEMS}


def load_configs(paths: dict[str, Path]) -> dict[str, GenerationCfg]:
    """Load the four systems and check the comparability invariants."""
    cfgs = {s: load_generation(p) for s, p in paths.items()}
    vlms = {c.vlm.model_dump_json() for c in cfgs.values()}
    if len(vlms) != 1:
        raise ValueError("All systems must share one generator configuration")
    vlm = next(iter(cfgs.values())).vlm
    if vlm.backend == "transformers" and (vlm.model, vlm.revision) != (MODEL, REVISION):
        raise ValueError(f"Generator must be {MODEL}@{REVISION}, got {vlm.model}@{vlm.revision}")
    for s, cfg in cfgs.items():
        base = load_generation(ROOT / "configs" / "generation" / f"{BASELINE}_{s}.yaml")
        if cfg.evidence != base.evidence or cfg.vlm.max_new_tokens != base.vlm.max_new_tokens:
            raise ValueError(f"{s}: evidence or token budget differs from {BASELINE}_{s}")
    return cfgs


def packages() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in PACKAGES:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    return out


# ------------------------------------------------------------ frozen evidence
def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def freeze_evidence(baseline_runs: Path, out: Path) -> dict[str, Any]:
    """Extract the questions and pages SmolVLM saw; fail if its systems disagree."""
    if out.exists():
        raise FileExistsError(f"{out} exists; frozen evidence is never overwritten")
    files = {s: baseline_runs / f"{BASELINE}_{s}.jsonl" for s in SYSTEMS}
    runs = {s: read_records(p) for s, p in files.items()}
    ref = runs["text_rag"]
    for s in EVIDENCE_SYSTEMS:
        got = [(r["key"], r["provided"], r["gold"]) for r in runs[s]]
        if got != [(r["key"], r["provided"], r["gold"]) for r in ref]:
            raise ValueError(f"{BASELINE}_{s} evidence differs from {BASELINE}_text_rag")
    closed = [r["key"] for r in runs["closed_book"]]
    if closed != [r["key"] for r in ref if r["condition"] != "missing_evidence"]:
        raise ValueError("closed_book keys are not the non-withheld subset of the evidence keys")
    rows = [
        {
            "key": r["key"],
            "condition": r["condition"],
            "query_id": r["query_id"],
            "provided": r["provided"],
            "gold": r["gold"],
        }
        for r in ref
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    meta = {
        "baseline": BASELINE,
        "n_rows": len(rows),
        "sha256": sha256_file(out),
        "source_run_files": {p.name: sha256_file(p) for p in files.values()},
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def build_examples(corpus: Corpus, evidence_rows: list[dict[str, Any]]) -> list[ge.Example]:
    """Rebuild the study's examples, checking them against the frozen evidence.

    Questions are re-sampled with the CLI's own procedure, so a corpus that does
    not reproduce the SmolVLM question set is rejected rather than silently used.
    """
    queries = ge.sample_queries(corpus.load_queries(), N_QUERIES, SEED)
    ood = corpus.load_queries("ood_queries")
    ood = random.Random(SEED).sample(ood, min(N_OOD, len(ood)))
    qrels = corpus.load_qrels()
    planned = []
    for q in queries:
        planned += [
            (q, "answerable", qrels[q.query_id]),
            (q, "missing_evidence", qrels[q.query_id]),
        ]
    planned += [(q, "out_of_domain", {}) for q in ood]
    if [f"{c}|{q.query_id}" for q, c, _ in planned] != [r["key"] for r in evidence_rows]:
        raise ValueError("Corpus does not reproduce the frozen question set (keys or order)")
    examples = []
    for (q, cond, gold), row in zip(planned, evidence_rows, strict=True):
        if row["gold"] != gold:
            raise ValueError(f"Gold pages differ for {row['key']}")
        if cond == "missing_evidence" and set(row["provided"]) & set(gold):
            raise ValueError(f"Withheld gold page provided for {row['key']}")
        # Scores are not used by evidence selection or the prompt; order is.
        hits = [ScoredPage(pid, 0.0) for pid in row["provided"]]
        examples.append(ge.Example(q, cond, gold, hits))
    return examples


def system_examples(system: str, examples: list[ge.Example]) -> list[ge.Example]:
    # Same rule as `mmrag eval-generation`: closed book skips the withheld condition.
    if system == "closed_book":
        return [e for e in examples if e.condition != "missing_evidence"]
    return examples


# --------------------------------------------------------------- corpus bundle
def _corpus_files(corpus_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in corpus_dir.rglob("*")
        if p.is_file() and not (set(p.parts) & JUNK) and not p.name.startswith("._")
    )


def package_corpus(corpus_dir: Path, evidence: Path, out_dir: Path) -> dict[str, Any]:
    """Write a byte-reproducible tar of the corpus, the frozen evidence and a manifest."""
    corpus = Corpus(corpus_dir)
    members: dict[str, Path] = {
        f"{ARCHIVE_ROOT}/{corpus_dir.name}/{p.relative_to(corpus_dir).as_posix()}": p
        for p in _corpus_files(corpus_dir)
    }
    members[f"{ARCHIVE_ROOT}/evidence/{EVIDENCE_NAME}"] = evidence
    manifest = {
        "study": STUDY,
        "corpus": corpus_dir.name,
        "corpus_fingerprint": corpus.fingerprint(),
        "dataset": {k: corpus.meta().get(k) for k in ("dataset", "revision", "dataset_license")},
        "evidence": f"evidence/{EVIDENCE_NAME}",
        "evidence_sha256": sha256_file(evidence),
        "files": {
            name.removeprefix(ARCHIVE_ROOT + "/"): {
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
            for name, p in sorted(members.items())
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{ARCHIVE_ROOT}-{corpus_dir.name}-{manifest['corpus_fingerprint']}.tar"
    if archive.exists():
        raise FileExistsError(f"{archive} exists; remove it deliberately to rebuild")

    def info(name: str, size: int) -> tarfile.TarInfo:
        ti = tarfile.TarInfo(name)
        ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = (
            size,
            0,
            0o644,
            0,
            0,
            "",
            "",
        )
        return ti

    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as tar:
        tar.addfile(
            info(f"{ARCHIVE_ROOT}/MANIFEST.json", len(manifest_bytes)), io.BytesIO(manifest_bytes)
        )
        for name, p in sorted(members.items()):
            with p.open("rb") as f:
                tar.addfile(info(name, p.stat().st_size), f)
    summary = {
        "archive": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "corpus_fingerprint": manifest["corpus_fingerprint"],
        "evidence_sha256": manifest["evidence_sha256"],
        "n_files": len(members),
    }
    archive.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def verify_corpus(root: Path) -> dict[str, Any]:
    """Check every extracted file against MANIFEST.json; return what the run records."""
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["files"]
    present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    present.discard("MANIFEST.json")
    if present != set(expected):
        raise ValueError(
            f"Corpus file set differs: missing {sorted(set(expected) - present)[:5]}, "
            f"unexpected {sorted(present - set(expected))[:5]}"
        )
    for rel, want in expected.items():
        p = root / rel
        if p.stat().st_size != want["bytes"] or sha256_file(p) != want["sha256"]:
            raise ValueError(f"Checksum mismatch: {rel}")
    corpus = Corpus(root / manifest["corpus"])
    if corpus.fingerprint() != manifest["corpus_fingerprint"]:
        raise ValueError("Corpus fingerprint differs from the manifest")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "corpus": manifest["corpus"],
        "corpus_fingerprint": manifest["corpus_fingerprint"],
        "evidence_sha256": manifest["evidence_sha256"],
        "dataset": manifest["dataset"],
        "n_files": len(expected),
    }


# ----------------------------------------------------------------- GPU helpers
def require_cuda() -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; CPU/MPS execution of this stage is prohibited")
    return torch


def gpu_summary() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False}
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_bytes": total,
        "vram_free_bytes": free,
    }


class TrackedVLM:
    """Delegates to a VLM and logs per-call latency and peak CUDA memory to JSONL."""

    def __init__(self, vlm: Any, log_path: Path):
        self.vlm, self.name, self.log_path, self.keys = vlm, vlm.name, log_path, []

    def info(self) -> dict[str, Any]:
        return self.vlm.info()

    def generate(self, content, max_new_tokens=None, image_max_side=1024):
        torch = None
        try:
            import torch as _torch

            torch = _torch if _torch.cuda.is_available() else None
        except ImportError:
            pass
        if torch:
            torch.cuda.reset_peak_memory_stats()
        out = self.vlm.generate(content, max_new_tokens, image_max_side)
        row = {
            "key": self.keys.pop(0) if self.keys else None,
            "n_images": sum(1 for p in content if p["type"] == "image"),
            "input_tokens": out.input_tokens,
            "output_tokens": out.output_tokens,
            "latency_s": out.latency_s,
        }
        if torch:
            torch.cuda.synchronize()
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        with self.log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return out


# --------------------------------------------------------------- memory check
def heaviest_examples(
    corpus: Corpus, examples: list[ge.Example], cfg: GenerationCfg, n: int
) -> list[ge.Example]:
    """Examples with the most image pixels at the configured resolution, then the most text."""
    side = cfg.evidence.image_max_side
    texts = corpus.load_text(cfg.evidence.text_source)

    def load(e: ge.Example) -> tuple[int, int, str]:
        pixels = 0
        for h in e.hits[: cfg.evidence.max_pages] if cfg.evidence.mode != "text" else []:
            p = corpus.page(h.page_id)
            scale = min(1.0, side / max(p.width, p.height))
            pixels += round(p.width * scale) * round(p.height * scale)
        chars = sum(
            min(len(texts.get(h.page_id, "")), cfg.evidence.max_chars_per_page)
            for h in e.hits[: cfg.evidence.max_pages]
        )
        return -pixels, -chars, e.key

    ranked = sorted((e for e in examples if e.hits), key=load)
    return ranked[:n]


def memory_check(
    output: Path,
    corpus_root: Path,
    source_commit: str,
    configs: dict[str, Path] | None = None,
    n_heaviest: int = 5,
    vlm_factory: Callable[[VLMCfg], Any] | None = None,
    cuda_required: bool = True,
) -> int:
    """Run the heaviest real prompts once; never writes experiment run files."""
    from mmrag.generate.rag import RAGPipeline
    from mmrag.generate.vlm import load_vlm

    out = output / "memory_check" / (time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    path = out / "report.json"
    report: dict[str, Any] = {
        "study": STUDY,
        "stage": "memory_check",
        "source_commit": source_commit,
        "status": "started",
        "note": "Answers here are not experimental results; this checks memory and latency only.",
        "python": platform.python_version(),
        "packages": packages(),
        "headroom_fraction": MEMORY_HEADROOM,
        "examples": [],
        "errors": [],
    }
    write_report(path, report)
    try:
        if cuda_required:
            require_cuda()
        report["gpu"] = gpu_summary()
        report["corpus"] = verify_corpus(corpus_root)
        corpus = Corpus(corpus_root / report["corpus"]["corpus"])
        evidence = read_records(corpus_root / "evidence" / EVIDENCE_NAME)
        examples = build_examples(corpus, evidence)
        cfgs = load_configs(configs or default_configs())
        vlm_cfg = cfgs["image_text_rag"].vlm
        report["generation"] = vlm_cfg.model_dump()
        # The heaviest prompts carry three page images plus their text. One extra
        # image-only and text-only prompt record the cost of each modality.
        plan = [
            ("image_text_rag", e)
            for e in heaviest_examples(corpus, examples, cfgs["image_text_rag"], n_heaviest)
        ]
        plan += [
            (s, heaviest_examples(corpus, examples, cfgs[s], 1)[0])
            for s in ("image_rag", "text_rag")
        ]
        # Recorded first, so an out-of-memory failure still shows which prompt it hit.
        report["plan"] = [{"system": cfgs[s].name, "key": e.key} for s, e in plan]
        write_report(path, report)
        start = time.perf_counter()
        vlm = (vlm_factory or load_vlm)(vlm_cfg)
        report["load_latency_s"] = time.perf_counter() - start
        tracked = TrackedVLM(vlm, out / "calls.jsonl")
        write_report(path, report)
        for system, ex in plan:
            cfg = cfgs[system]
            rag = RAGPipeline(corpus, tracked, cfg)
            tracked.keys = [ex.key]
            res = rag.answer(ex.query.text, hits=ex.hits)
            call = read_records(out / "calls.jsonl")[-1]
            sizes = (
                [
                    list(corpus.image(e.page_id, cfg.evidence.image_max_side).size)
                    for e in res.evidence
                ]
                if cfg.evidence.mode in ("image", "image_text")
                else []
            )
            report["examples"].append(
                {
                    "system": cfg.name,
                    "key": ex.key,
                    "evidence_mode": cfg.evidence.mode,
                    "pages": [e.page_id for e in res.evidence],
                    "image_sizes_after_resize": sizes,
                    "evidence_text_chars": sum(len(e.text or "") for e in res.evidence),
                    "max_new_tokens": cfg.vlm.max_new_tokens,
                    "answer_preview": res.answer.text[:120],
                    **{k: v for k, v in call.items() if k != "key"},
                }
            )
            write_report(path, report)
        peak = max((e.get("peak_reserved_bytes", 0) for e in report["examples"]), default=0)
        total = report["gpu"].get("vram_total_bytes")
        report["peak_allocated_bytes"] = max(
            (e.get("peak_allocated_bytes", 0) for e in report["examples"]), default=0
        )
        report["peak_reserved_bytes"] = peak
        report["fits"] = bool(total) and peak < MEMORY_HEADROOM * total
        report["status"] = "passed" if report["fits"] or not cuda_required else "failed"
        if not report["fits"] and cuda_required:
            report["errors"].append({"type": "Headroom", "message": "Peak exceeds headroom"})
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(
            {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        )
    write_report(path, report)
    (out / "report.sha256").write_text(sha256_file(path) + "  report.json\n")
    print(path, report["status"], flush=True)
    return 0 if report["status"] == "passed" else 1


# ------------------------------------------------------------------ main run
def _refuse_results_dir(output: Path) -> None:
    resolved = output.resolve()
    if (ROOT / "results").resolve() in (resolved, *resolved.parents):
        raise ValueError("Write Colab outputs outside results/; import them with `score`")


def _repair_partial_tail(path: Path) -> None:
    """A disconnect can leave a half-written last line; move it aside, keep the rest."""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    head, _, tail = data.rpartition(b"\n")
    try:
        json.loads(tail)
        path.write_bytes(data + b"\n")
        return
    except json.JSONDecodeError:
        pass
    path.with_suffix(f".partial-{time.strftime('%Y%m%dT%H%M%S')}").write_bytes(tail)
    path.write_bytes(head + b"\n" if head else b"")


def _check_existing(path: Path, system: str, keys: list[str]) -> set[str]:
    done = set()
    for r in read_records(path) if path.exists() else []:
        if r.get("system") != system or r["key"] not in keys:
            raise ValueError(f"{path} contains a record not belonging to {system}: {r.get('key')}")
        done.add(r["key"])
    return done


def run(
    output: Path,
    corpus_root: Path,
    source_commit: str,
    systems: tuple[str, ...] = SYSTEMS,
    configs: dict[str, Path] | None = None,
    vlm_factory: Callable[[VLMCfg], Any] | None = None,
    cuda_required: bool = True,
) -> int:
    """Generate answers for each system; safe to rerun after any interruption."""
    from mmrag.generate.rag import RAGPipeline
    from mmrag.generate.vlm import load_vlm
    from mmrag.runtime import hardware_summary, peak_rss_mb

    _refuse_results_dir(output)
    if cuda_required:
        torch = require_cuda()
        torch.manual_seed(SEED)  # decoding is greedy; recorded for completeness
    corpus_info = verify_corpus(corpus_root)
    corpus = Corpus(corpus_root / corpus_info["corpus"])
    examples = build_examples(corpus, read_records(corpus_root / "evidence" / EVIDENCE_NAME))
    cfg_paths = configs or default_configs()
    cfgs = load_configs(cfg_paths)
    experiment = {
        "study": STUDY,
        "baseline": BASELINE,
        "model": MODEL,
        "model_revision": REVISION,
        "systems": {s: cfgs[s].model_dump() for s in SYSTEMS},
        "seed": SEED,
        "n_queries": N_QUERIES,
        "n_ood": N_OOD,
        "evidence": f"frozen from {BASELINE} run files (hybrid_text retrieval)",
        "corpus": corpus_info,
        "n_examples": {s: len(system_examples(s, examples)) for s in SYSTEMS},
    }
    output.mkdir(parents=True, exist_ok=True)
    prov_path = output / "provenance.json"
    fingerprint = sha256_json(experiment)
    if prov_path.exists():
        prov = json.loads(prov_path.read_text())
        if prov["experiment_sha256"] != fingerprint:
            raise ValueError("Output directory belongs to a different experiment configuration")
        if prov["source_commit"] != source_commit:
            raise ValueError(
                f"Output directory was started from {prov['source_commit']}; use a new directory "
                "or the same source bundle"
            )
    else:
        write_report(
            prov_path,
            {
                "experiment": experiment,
                "experiment_sha256": fingerprint,
                "source_commit": source_commit,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
    session = {
        "session": uuid.uuid4().hex[:12],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_commit": source_commit,
        "systems": list(systems),
        "packages": packages(),
        "hardware": hardware_summary(),
        "gpu": gpu_summary(),
    }
    sessions = output / "sessions.jsonl"
    with sessions.open("a") as f:
        f.write(json.dumps(session) + "\n")

    runs = output / "runs"
    runs.mkdir(exist_ok=True)
    status = "completed"
    vlm = None
    try:
        for s in systems:
            cfg = cfgs[s]
            exs = system_examples(s, examples)
            keys = [e.key for e in exs]
            run_path = runs / f"{cfg.name}.jsonl"
            done_path = runs / f"{cfg.name}.complete.json"
            if done_path.exists():
                done = json.loads(done_path.read_text())
                if sha256_file(run_path) != done["sha256"]:
                    raise ValueError(f"{run_path} changed after it was marked complete")
                print(f"{cfg.name}: complete ({done['n']} records), skipped", flush=True)
                continue
            _repair_partial_tail(run_path)
            already = _check_existing(run_path, cfg.name, keys)
            print(f"{cfg.name}: {len(already)}/{len(keys)} already done", flush=True)
            if vlm is None:
                start = time.perf_counter()
                vlm = (vlm_factory or load_vlm)(cfg.vlm)
                session["load_latency_s"] = time.perf_counter() - start
            tracked = TrackedVLM(vlm, runs / f"{cfg.name}.calls.jsonl")
            tracked.keys = [k for k in keys if k not in already]
            recs = ge.run_system(RAGPipeline(corpus, tracked, cfg), cfg.name, exs, run_path)
            if [r["key"] for r in recs] != keys:
                raise RuntimeError(f"{cfg.name}: records do not match the planned examples")
            # The call log spans every session, so the peak covers resumed runs too.
            calls = read_records(tracked.log_path) if tracked.log_path.exists() else []
            write_report(
                done_path,
                {
                    "system": cfg.name,
                    "n": len(recs),
                    "sha256": sha256_file(run_path),
                    "config": cfg.model_dump(),
                    "vlm_info": vlm.info(),
                    "completed_in_session": session["session"],
                    "n_calls_logged": len(calls),
                    "peak_allocated_bytes": max(
                        (c.get("peak_allocated_bytes", 0) for c in calls), default=0
                    ),
                    "peak_reserved_bytes": max(
                        (c.get("peak_reserved_bytes", 0) for c in calls), default=0
                    ),
                    "peak_rss_mb": round(peak_rss_mb(), 1),
                },
            )
            print(f"{cfg.name}: complete", flush=True)
    except BaseException as exc:
        status = f"interrupted: {type(exc).__name__}: {exc}"
        raise
    finally:
        end = {
            "session": session["session"],
            "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": status,
            "load_latency_s": session.get("load_latency_s"),
        }
        with sessions.open("a") as f:
            f.write(json.dumps(end) + "\n")
        write_checksums(output)
    return 0


def write_checksums(output: Path) -> Path:
    path = output / "CHECKSUMS.sha256"
    lines = [
        f"{sha256_file(p)}  {p.relative_to(output).as_posix()}"
        for p in sorted(output.rglob("*"))
        if p.is_file() and p != path and "memory_check" not in p.relative_to(output).parts
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------- score
def score(
    output: Path,
    corpus_dir: Path,
    results: Path = ROOT / "results" / "generation",
    runs_out: Path = ROOT / "artifacts" / "runs" / "generation",
    systems: tuple[str, ...] = SYSTEMS,
    nli: bool = True,
) -> list[Path]:
    """Score completed Colab runs on the local CPU and write result files (never overwrites)."""
    from mmrag.eval.generation_metrics import NLIScorer

    corpus = Corpus(corpus_dir)
    prov = json.loads((output / "provenance.json").read_text())
    if prov["experiment"]["corpus"]["corpus_fingerprint"] != corpus.fingerprint():
        raise ValueError("Local corpus fingerprint differs from the one the run used")
    sessions = read_records(output / "sessions.jsonl")
    starts = {s["session"]: s for s in sessions if "started" in s}
    pending = []
    for s in systems:
        name = f"{CANDIDATE}_{s}"
        done = json.loads((output / "runs" / f"{name}.complete.json").read_text())
        run_path = output / "runs" / f"{name}.jsonl"
        if sha256_file(run_path) != done["sha256"]:
            raise ValueError(f"{run_path} does not match its completion checksum")
        targets = (
            results / corpus.name / f"{name}.json",
            runs_out / corpus.name / f"{name}.scored.jsonl",
        )
        for t in targets:
            if t.exists():
                raise FileExistsError(f"{t} exists; results are never overwritten")
        pending.append((name, done, run_path, targets))
    # Same scorer, text source and device as the SmolVLM-500M study.
    scorer = NLIScorer(device="cpu") if nli else None
    faith = corpus.load_text(FAITH_SOURCE if corpus.has_text(FAITH_SOURCE) else "ocr_tesseract")
    written = []
    for name, done, run_path, (res_path, scored_path) in pending:
        recs = ge.score_records(read_records(run_path), scorer, faith)
        cfg = GenerationCfg.model_validate(done["config"])
        stub = SimpleNamespace(cfg=cfg, vlm=SimpleNamespace(info=lambda d=done: d["vlm_info"]))
        summary = ge.summarize(recs, stub, name)
        # summarize() reads memory from the current process; use what Colab measured.
        summary["systems"] = {
            k: v for k, v in summary["systems"].items() if not k.startswith(("cuda_", "mps_"))
        }
        summary["systems"].update(
            peak_rss_mb=done["peak_rss_mb"],
            cuda_peak_allocated_mb=round(done["peak_allocated_bytes"] / 1024**2, 1),
            cuda_peak_reserved_mb=round(done["peak_reserved_bytes"] / 1024**2, 1),
        )
        session = starts.get(done["completed_in_session"], {})
        summary.update(
            retrieval="hybrid_text",
            evidence_source=f"frozen from {BASELINE} run files",
            corpus=corpus.name,
            seed=SEED,
            hardware={**session.get("hardware", {}), **session.get("gpu", {})},
            faithfulness_text_source=FAITH_SOURCE if scorer else None,
            scoring_device="cpu" if scorer else None,
            provenance={
                "source_commit": prov["source_commit"],
                "experiment_sha256": prov["experiment_sha256"],
                "run_sha256": done["sha256"],
            },
        )
        res_path.parent.mkdir(parents=True, exist_ok=True)
        scored_path.parent.mkdir(parents=True, exist_ok=True)
        res_path.write_text(json.dumps(summary, indent=2))
        scored_path.write_text("".join(json.dumps(r) + "\n" for r in recs))
        written += [res_path, scored_path]
    return written


# ----------------------------------------------------------------------- CLI
def _commit(value: str) -> str:
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise argparse.ArgumentTypeError("must be a full lowercase Git SHA")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("freeze-evidence")
    p.add_argument("--baseline-runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("package-corpus")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("verify-corpus")
    p.add_argument("--corpus-root", type=Path, required=True)
    for name in ("memory-check", "run"):
        p = sub.add_parser(name)
        p.add_argument("--corpus-root", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--source-commit", type=_commit, required=True)
    sub.choices["memory-check"].add_argument("--n-heaviest", type=int, default=5)
    sub.choices["run"].add_argument("--systems", default=",".join(SYSTEMS))
    p = sub.add_parser("score")
    p.add_argument("--output", type=Path, required=True, help="Downloaded Colab output directory")
    p.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()

    if args.stage == "freeze-evidence":
        print(json.dumps(freeze_evidence(args.baseline_runs, args.out), indent=2))
    elif args.stage == "package-corpus":
        print(json.dumps(package_corpus(args.corpus, args.evidence, args.out_dir), indent=2))
    elif args.stage == "verify-corpus":
        print(json.dumps(verify_corpus(args.corpus_root), indent=2))
    elif args.stage == "memory-check":
        raise SystemExit(
            memory_check(
                args.output, args.corpus_root, args.source_commit, n_heaviest=args.n_heaviest
            )
        )
    elif args.stage == "run":
        systems = tuple(s for s in args.systems.split(",") if s)
        if unknown := set(systems) - set(SYSTEMS):
            parser.error(f"unknown systems: {sorted(unknown)}")
        raise SystemExit(run(args.output, args.corpus_root, args.source_commit, systems))
    elif args.stage == "score":
        for path in score(args.output, args.corpus):
            print(path)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
