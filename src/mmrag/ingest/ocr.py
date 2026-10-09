"""Tesseract OCR over page images.

OCR text is the input to the lexical and dense text baselines. It is produced
from pixels only, so it is a fair "text pipeline" counterpart to the visual
retrievers that consume the same images.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mmrag.data.corpus import Corpus

TEXT_SOURCE = "ocr_tesseract"


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def tesseract_version() -> str:
    out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
    return (out.stdout or out.stderr).splitlines()[0].strip()


def normalize_ocr(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # undo end-of-line hyphenation
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ocr_image(path: Path, lang: str = "eng", psm: int = 3) -> str:
    import pytesseract

    return normalize_ocr(pytesseract.image_to_string(str(path), lang=lang, config=f"--psm {psm}"))


def run_ocr(
    corpus: Corpus,
    lang: str = "eng",
    psm: int = 3,
    workers: int = 6,
    page_ids: list[str] | None = None,
    progress: bool = True,
) -> dict[str, str]:
    """OCR every page (or the given subset), merging with any existing results."""
    if not tesseract_available():
        raise RuntimeError("tesseract binary not found; install it or use another text source")
    existing: dict[str, str] = {}
    if corpus.text_path(TEXT_SOURCE).exists():
        from mmrag.data.corpus import read_jsonl

        existing = {r["page_id"]: r["text"] for r in read_jsonl(corpus.text_path(TEXT_SOURCE))}
    todo = [pid for pid in (page_ids or corpus.page_ids) if pid not in existing]

    def work(pid: str) -> tuple[str, str]:
        return pid, ocr_image(corpus.image_path(pid), lang, psm)

    # pytesseract shells out to the tesseract binary, so threads give real parallelism.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (pid, text) in enumerate(pool.map(work, todo), 1):
            existing[pid] = text
            if progress and i % 100 == 0:
                print(f"  ocr {i}/{len(todo)}", flush=True)

    corpus.save_text(
        TEXT_SOURCE,
        existing,
        meta={"engine": tesseract_version(), "lang": lang, "psm": psm, "n_pages": len(existing)},
    )
    return existing
