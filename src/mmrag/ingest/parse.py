"""Turn user files (PDF, PNG, JPEG, ...) into page images in a corpus directory."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image

from mmrag.data.corpus import read_jsonl, write_jsonl
from mmrag.types import Page

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()[:48] or "doc"


def render_pdf(path: Path, dpi: int = 150) -> list[Image.Image]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return [pdf[i].render(scale=dpi / 72).to_pil().convert("RGB") for i in range(len(pdf))]
    finally:
        pdf.close()


def load_file_pages(path: Path, dpi: int = 150) -> list[Image.Image]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return render_pdf(path, dpi)
    if suffix in IMAGE_SUFFIXES:
        return [Image.open(path).convert("RGB")]
    raise ValueError(f"Unsupported file type: {path.suffix}")


def ingest_files(
    files: list[Path],
    corpus_dir: Path,
    max_side: int = 1600,
    dpi: int = 150,
    jpeg_quality: int = 88,
) -> list[Page]:
    """Append files to a corpus directory. Returns the pages that were added.

    The document id includes a content hash, so re-ingesting an identical file
    is a no-op and two different files with the same name do not collide.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "images").mkdir(exist_ok=True)
    manifest = corpus_dir / "pages.jsonl"
    existing = [Page(**r) for r in read_jsonl(manifest)] if manifest.exists() else []
    known_docs = {p.doc_id for p in existing}
    added: list[Page] = []

    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
        doc_id = f"{_slug(path.stem)}_{digest}"
        if doc_id in known_docs:
            continue
        for i, img in enumerate(load_file_pages(path, dpi)):
            if max(img.size) > max_side:
                img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            pid = f"{doc_id}-p{i}"
            rel = f"images/{pid}.jpg"
            img.save(corpus_dir / rel, "JPEG", quality=jpeg_quality)
            added.append(Page(pid, doc_id, i, rel, img.width, img.height))
        known_docs.add(doc_id)

    if added:
        write_jsonl(manifest, (p.to_dict() for p in existing + added))
        meta_path = corpus_dir / "meta.json"
        if not meta_path.exists():
            meta_path.write_text(json.dumps({"dataset": "user-upload"}, indent=2))
    return added
