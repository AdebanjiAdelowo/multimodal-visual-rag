"""On-disk corpus layout.

    <corpus_dir>/
        pages.jsonl          page manifest
        images/<page_id>.jpg page renders
        text/<source>.jsonl  page_id -> text, one file per text source
                             (ocr_tesseract, parser_markdown, caption_<model>, ...)
        queries.jsonl        optional evaluation queries
        qrels.jsonl          optional graded page relevance
        meta.json            provenance (dataset, revision, licence)

Text sources are kept apart from the manifest so that every retrieval strategy
names the exact representation it indexes, and so that no query-derived field
can reach an index by accident.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from mmrag.types import Page, Qrels, Query


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Corpus:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not (self.root / "pages.jsonl").exists():
            raise FileNotFoundError(f"No pages.jsonl under {self.root}; run ingestion first.")
        self.pages: list[Page] = [Page(**r) for r in read_jsonl(self.root / "pages.jsonl")]
        self._by_id = {p.page_id: p for p in self.pages}
        if len(self._by_id) != len(self.pages):
            raise ValueError("Duplicate page_id in pages.jsonl")

    # -- pages -----------------------------------------------------------
    @property
    def name(self) -> str:
        return self.root.name

    @property
    def page_ids(self) -> list[str]:
        return [p.page_id for p in self.pages]

    def page(self, page_id: str) -> Page:
        return self._by_id[page_id]

    def __len__(self) -> int:
        return len(self.pages)

    def image_path(self, page_id: str) -> Path:
        return self.root / self._by_id[page_id].image_path

    def image(self, page_id: str, max_side: int | None = None) -> Image.Image:
        img = Image.open(self.image_path(page_id)).convert("RGB")
        if max_side and max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return img

    def fingerprint(self) -> str:
        """Stable hash of the page manifest; index caches are keyed on it."""
        h = hashlib.sha256()
        for p in self.pages:
            h.update(f"{p.page_id}|{p.image_path}|{p.width}x{p.height}\n".encode())
        return h.hexdigest()[:16]

    # -- text sources ------------------------------------------------------
    def text_path(self, source: str) -> Path:
        return self.root / "text" / f"{source}.jsonl"

    def text_sources(self) -> list[str]:
        d = self.root / "text"
        return sorted(p.stem for p in d.glob("*.jsonl")) if d.exists() else []

    def has_text(self, source: str) -> bool:
        return self.text_path(source).exists()

    def load_text(self, source: str) -> dict[str, str]:
        path = self.text_path(source)
        if not path.exists():
            raise FileNotFoundError(
                f"Text source '{source}' missing for corpus '{self.name}'. "
                f"Available: {self.text_sources()}"
            )
        texts = {r["page_id"]: r["text"] for r in read_jsonl(path)}
        missing = [pid for pid in self.page_ids if pid not in texts]
        if missing:
            raise ValueError(
                f"Text source '{source}' lacks {len(missing)} pages, e.g. {missing[:3]}"
            )
        return texts

    def save_text(self, source: str, texts: dict[str, str], meta: dict | None = None) -> None:
        write_jsonl(
            self.text_path(source),
            ({"page_id": pid, "text": texts[pid]} for pid in self.page_ids if pid in texts),
        )
        if meta is not None:
            self.text_path(source).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    # -- evaluation annotations -------------------------------------------
    def load_queries(self, name: str = "queries") -> list[Query]:
        return [Query.from_dict(r) for r in read_jsonl(self.root / f"{name}.jsonl")]

    def load_qrels(self) -> Qrels:
        qrels: Qrels = {}
        for r in read_jsonl(self.root / "qrels.jsonl"):
            qrels.setdefault(r["query_id"], {})[r["page_id"]] = int(r["score"])
        return qrels

    def meta(self) -> dict[str, Any]:
        p = self.root / "meta.json"
        return json.loads(p.read_text()) if p.exists() else {}


def write_corpus(
    root: Path,
    pages: list[Page],
    meta: dict[str, Any],
    queries: list[Query] | None = None,
    qrels: Qrels | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "pages.jsonl", (p.to_dict() for p in pages))
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    if queries is not None:
        write_jsonl(root / "queries.jsonl", (q.to_dict() for q in queries))
    if qrels is not None:
        write_jsonl(
            root / "qrels.jsonl",
            (
                {"query_id": qid, "page_id": pid, "score": s}
                for qid, rel in qrels.items()
                for pid, s in rel.items()
            ),
        )
