"""Run the GPU-tier experiments on a Kaggle T4 (or any CUDA machine).

Setup (see kaggle/README.md):
  1. `make kaggle-bundle` and upload kaggle/mmrag-src.zip as a private Kaggle dataset.
  2. New notebook: accelerator "GPU T4 x2", internet on, attach the dataset.
  3. Paste this file into a cell, set STAGES, run.

The script installs the package from the uploaded source, downloads the public
benchmark from Hugging Face, runs the selected stages and writes
/kaggle/working/mmrag_results.zip after every stage, so a session that times
out still leaves an archive of what finished. The archive holds metric JSON
files and per-query records (rankings, model outputs, and the benchmark's
CC-BY-4.0 questions and reference answers). It never contains page images or
page text.

Results are written under results_<tag>/ and artifacts/runs_<tag>/ (tag "t4" by
default), never under results/, so unpacking the archive over the repository
cannot overwrite results measured on another machine.

Environment overrides (all optional):
  MMRAG_STAGES      comma-separated subset of: data,retrieval,captions,generation,ablations
  MMRAG_GENERATORS  comma-separated subset of: smolvlm_500m,qwen3vl_2b,qwen3vl_4b,qwen3vl_8b_4bit
  MMRAG_N_QUERIES   answerable questions per generator (default 60; 0 = all 215)
  MMRAG_N_OOD       out-of-domain questions (default 30)
  MMRAG_TAG         suffix of the result directories (default t4)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SOURCE = Path(os.environ.get("MMRAG_SOURCE", "/kaggle/input"))  # searched for the source tree
WORK = Path(os.environ.get("MMRAG_WORK", "/kaggle/working"))
ALL_STAGES = ["data", "retrieval", "captions", "generation", "ablations"]
STAGES = [s for s in os.environ.get("MMRAG_STAGES", ",".join(ALL_STAGES)).split(",") if s]
ALL_GENERATORS = ["smolvlm_500m", "qwen3vl_2b", "qwen3vl_4b", "qwen3vl_8b_4bit"]
GENERATORS = [
    g for g in os.environ.get("MMRAG_GENERATORS", ",".join(ALL_GENERATORS)).split(",") if g
]
TAG = os.environ.get("MMRAG_TAG", "t4")
RESULTS, RUNS = f"results_{TAG}", f"artifacts/runs_{TAG}"

CORPUS = "data/processed/vidore_v3_computer_science"
RET = "configs/retrieval"
GEN = "configs/generation"
# The text hybrid is used for generation: adding ColSmol changed nDCG@10 by
# +0.007 [-0.011, +0.024] locally, which does not justify a second index.
BEST_RETRIEVAL = f"{RET}/hybrid_text.yaml"
# Defaults reproduce the local protocol (60 answerable, 30 out-of-domain, seed 13),
# so every generator on every machine answers the same examples.
N_QUERIES = os.environ.get("MMRAG_N_QUERIES", "60")
N_OOD = os.environ.get("MMRAG_N_OOD", "30")
RET_OUT = ["--out", f"{RESULTS}/retrieval", "--runs", f"{RUNS}/retrieval"]
GEN_OUT = ["--out", f"{RESULTS}/generation", "--runs", f"{RUNS}/generation"]

COMMANDS: dict[str, list[list[str]]] = {
    "data": [
        ["mmrag", "hw"],
        ["mmrag", "fetch-vidore", "--domain", "computer_science", "--drop-raw"],
        ["mmrag", "ocr", "-c", CORPUS, "--workers", "4"],
        ["mmrag", "fetch-ood", "-c", CORPUS],
    ],
    # Every first-stage and hybrid strategy on the full corpus, on one machine,
    # so the latency column is comparable across rows.
    "retrieval": [
        ["mmrag", "eval-retrieval", "-c", CORPUS, *RET_OUT]
        + [
            f"{RET}/{n}.yaml"
            for n in (
                "bm25_ocr",
                "bm25_parser",
                "dense_ocr",
                "dense_parser",
                "visual_clip",
                "visual_siglip",
                "late_colsmol",
                "late_colqwen2",
                "hybrid_text",
                "hybrid_bm25_siglip",
                "hybrid_text_siglip",
                "hybrid_text_siglip_w025",
                "hybrid_text_colsmol",
                "hybrid_text_rerank",
                "hybrid_text_siglip_rerank",
                "hybrid_text_colsmol_rerank",
            )
        ],  # fmt: skip
    ],
    # Captions for all 1,360 pages with two captioners, then caption retrieval.
    "captions": [
        ["mmrag", "caption", "-c", CORPUS, "--config", f"{GEN}/caption_smolvlm_500m.yaml"],
        ["mmrag", "caption", "-c", CORPUS, "--config", f"{GEN}/caption_qwen3vl_4b.yaml"],
        ["mmrag", "eval-retrieval", "-c", CORPUS, *RET_OUT]
        + [
            f"{RET}/{n}.yaml"
            for n in ("bm25_caption", "dense_caption", "bm25_caption_qwen", "dense_caption_qwen")
        ],  # fmt: skip
    ],
    # Small vs larger generators, all four evidence modes, identical evidence.
    "generation": [
        [
            "mmrag",
            "eval-generation",
            "-c",
            CORPUS,
            "--retrieval",
            BEST_RETRIEVAL,
            "--n-queries",
            N_QUERIES,
            "--n-ood",
            N_OOD,
            "--oracle",
            *GEN_OUT,
        ]
        + [
            f"{GEN}/{m}_{t}.yaml"
            for t in ("closed_book", "text_rag", "image_rag", "image_text_rag")
        ]
        for m in GENERATORS
    ],  # fmt: skip
    # Number of pages, image resolution and text budget, with the 4B generator.
    "ablations": [
        *[
            [
                "mmrag",
                "eval-generation",
                "-c",
                CORPUS,
                "--retrieval",
                BEST_RETRIEVAL,
                "--n-queries",
                N_QUERIES,
                "--n-ood",
                "0",
                "--max-pages",
                k,
                *GEN_OUT,
                f"{GEN}/qwen3vl_4b_image_rag.yaml",
            ]
            for k in ("1", "2", "5")
        ],
        *[
            [
                "mmrag",
                "eval-generation",
                "-c",
                CORPUS,
                "--retrieval",
                BEST_RETRIEVAL,
                "--n-queries",
                N_QUERIES,
                "--n-ood",
                "0",
                "--image-max-side",
                s,
                *GEN_OUT,
                f"{GEN}/qwen3vl_4b_image_rag.yaml",
            ]
            for s in ("512", "768", "1536")
        ],
        *[
            [
                "mmrag",
                "eval-generation",
                "-c",
                CORPUS,
                "--retrieval",
                BEST_RETRIEVAL,
                "--n-queries",
                N_QUERIES,
                "--n-ood",
                "0",
                "--max-chars",
                c,
                *GEN_OUT,
                f"{GEN}/qwen3vl_4b_text_rag.yaml",
            ]
            for c in ("800", "5000")
        ],
    ],  # fmt: skip
}


def sh(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def find_source() -> Path:
    """Locate the uploaded source tree (unpacked directory or zip) under SOURCE."""
    for p in SOURCE.rglob("pyproject.toml"):
        if (p.parent / "src" / "mmrag").exists():
            return p.parent
    for z in SOURCE.rglob("mmrag-src.zip"):
        shutil.unpack_archive(z, WORK / "src_unpacked")
        return WORK / "src_unpacked" / "multimodal-visual-rag"
    sys.exit(
        f"No mmrag source found under {SOURCE}. Attach the dataset made by `make kaggle-bundle`."
    )


def main() -> None:
    repo = WORK / "multimodal-visual-rag"
    if not repo.exists():
        shutil.copytree(find_source(), repo)  # /kaggle/input is read-only
    sh(["apt-get", "install", "-y", "-qq", "tesseract-ocr"])
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".[models,late,quant]"], cwd=repo)
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout, flush=True)  # fmt: skip

    log = []
    for stage in STAGES:
        entry = {"stage": stage, "status": "completed", "commands": [], "failed_commands": []}
        t0 = time.perf_counter()
        for cmd in COMMANDS[stage]:
            try:
                sh(cmd, cwd=repo)
                entry["commands"].append(" ".join(cmd))
            except subprocess.CalledProcessError as e:
                # A failed command is recorded and the stage moves on, so one generator
                # running out of memory does not cost the others. Its results are absent.
                print(f"COMMAND FAILED in stage {stage}: {e}", flush=True)
                entry["status"] = "failed"
                entry["failed_commands"].append(" ".join(cmd))
                if stage == "data":
                    break
        entry["seconds"] = round(time.perf_counter() - t0, 1)
        log.append(entry)
        archive = write_archive(repo, log)
        if stage == "data" and entry["status"] == "failed":
            sys.exit("The data stage failed; nothing else can run.")

    print(f"\nresults archive: {archive}")
    for e in log:
        print(f"  {e['stage']:<12} {e['status']:<10} {e['seconds']:>9.1f} s")


def model_revisions() -> dict[str, str]:
    """Commit of every Hugging Face model in the cache, for the session record."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    out = {}
    for ref in sorted(hub.glob("models--*/refs/main")):
        out[ref.parts[-3].removeprefix("models--").replace("--", "/")] = ref.read_text().strip()
    return out


def write_archive(repo: Path, log: list[dict]) -> str:
    """Compile the report and zip everything produced so far. Called after every stage."""
    out = WORK / "mmrag_results"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    if (repo / RESULTS).exists():
        sh(["mmrag", "report", "--results", RESULTS, "--runs", RUNS,
            "--out", f"{RESULTS}/SUMMARY.md"], cwd=repo)  # fmt: skip
        shutil.copytree(repo / RESULTS, out / RESULTS)
    if (repo / RUNS).exists():
        shutil.copytree(repo / RUNS, out / RUNS)
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    ).stdout.splitlines()
    session = {
        "tag": TAG,
        "stages": log,
        "n_queries": N_QUERIES,
        "n_ood": N_OOD,
        "generation_retrieval": BEST_RETRIEVAL,
        "model_revisions": model_revisions(),
        "packages": freeze,
    }
    (out / f"session_{TAG}.json").write_text(json.dumps(session, indent=2))
    return shutil.make_archive(str(out), "zip", out)


if __name__ == "__main__":
    main()
