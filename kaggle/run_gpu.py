"""Run the GPU-tier experiments on a Kaggle T4 (or any CUDA machine).

Setup (see kaggle/README.md):
  1. `make kaggle-bundle` and upload kaggle/mmrag-src.zip as a private Kaggle dataset.
  2. New notebook: accelerator "GPU T4 x2", internet on, attach the dataset.
  3. Paste this file into a cell, set STAGES, run.

The script installs the package from the uploaded source, downloads the public
benchmark from Hugging Face, runs the selected stages and writes
/kaggle/working/mmrag_results.zip. The archive holds metric JSON files and
per-query records (rankings, model outputs, and the benchmark's CC-BY-4.0
questions and reference answers). It never contains page images or page text.
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
STAGES = ["data", "retrieval", "captions", "generation", "ablations"]

CORPUS = "data/processed/vidore_v3_computer_science"
RET = "configs/retrieval"
GEN = "configs/generation"
BEST_RETRIEVAL = f"{RET}/hybrid_text_colsmol.yaml"
N_QUERIES, N_OOD = "0", "60"  # 0 = every answerable query

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
        ["mmrag", "eval-retrieval", "-c", CORPUS]
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
        ["mmrag", "eval-retrieval", "-c", CORPUS]
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
        ]
        + [
            f"{GEN}/{m}_{t}.yaml"
            for t in ("closed_book", "text_rag", "image_rag", "image_text_rag")
        ]
        for m in ("smolvlm_500m", "qwen3vl_2b", "qwen3vl_4b", "qwen3vl_8b_4bit")
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
                "100",
                "--n-ood",
                "0",
                "--max-pages",
                k,
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
                "100",
                "--n-ood",
                "0",
                "--image-max-side",
                s,
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
                "100",
                "--n-ood",
                "0",
                "--max-chars",
                c,
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
        entry = {"stage": stage, "status": "completed", "commands": []}
        t0 = time.perf_counter()
        for cmd in COMMANDS[stage]:
            try:
                sh(cmd, cwd=repo)
                entry["commands"].append(" ".join(cmd))
            except subprocess.CalledProcessError as e:
                # A failed stage is recorded and skipped; its results are simply absent.
                print(f"STAGE FAILED: {stage}: {e}", flush=True)
                entry.update(status="failed", failed_command=" ".join(cmd))
                break
        entry["seconds"] = round(time.perf_counter() - t0, 1)
        log.append(entry)

    sh(["mmrag", "report"], cwd=repo)
    out = WORK / "mmrag_results"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(repo / "results", out / "results")
    if (repo / "artifacts" / "runs").exists():
        shutil.copytree(repo / "artifacts" / "runs", out / "artifacts" / "runs")
    (out / "session.json").write_text(json.dumps({"stages": log}, indent=2))
    archive = shutil.make_archive(str(out), "zip", out)
    print(f"\nresults archive: {archive}")
    for e in log:
        print(f"  {e['stage']:<12} {e['status']:<10} {e['seconds']:>9.1f} s")


if __name__ == "__main__":
    main()
