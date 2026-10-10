"""Application service: owns corpora, indexes, retrieval pipelines and the generator.

Everything is loaded lazily and cached, so the API starts instantly and a
search-only session never loads a vision-language model.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mmrag.config import GenerationCfg, load_generation, load_retrieval
from mmrag.data.corpus import Corpus
from mmrag.generate.rag import RAGPipeline
from mmrag.retrieve.pipeline import IndexRegistry, RetrievalPipeline
from mmrag.types import ScoredPage

UPLOAD_CORPUS = "uploads"


@dataclass
class Settings:
    corpus_root: Path = field(
        default_factory=lambda: Path(os.environ.get("MMRAG_CORPUS_ROOT", "data/processed"))
    )
    upload_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("MMRAG_UPLOAD_DIR", "uploads/corpus"))
    )
    config_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("MMRAG_CONFIG_DIR", "configs"))
    )
    index_root: Path = field(
        default_factory=lambda: Path(os.environ.get("MMRAG_INDEX_ROOT", "artifacts/index"))
    )
    default_strategy: str = field(
        default_factory=lambda: os.environ.get("MMRAG_STRATEGY", "bm25_ocr")
    )
    generation: str = field(default_factory=lambda: os.environ.get("MMRAG_GENERATION", "mock"))
    device: str = field(default_factory=lambda: os.environ.get("MMRAG_DEVICE", "auto"))
    max_upload_mb: int = 25


class Service:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._lock = threading.RLock()
        self._corpora: dict[str, Corpus] = {}
        self._registries: dict[str, IndexRegistry] = {}
        self._pipelines: dict[tuple[str, str], RetrievalPipeline] = {}
        self._vlm = None
        self._gen_cfg: GenerationCfg | None = None

    # -- corpora -------------------------------------------------------------
    def corpus_dirs(self) -> dict[str, Path]:
        s = self.settings
        found = {}
        if s.corpus_root.exists():
            found = {
                d.name: d for d in sorted(s.corpus_root.iterdir()) if (d / "pages.jsonl").exists()
            }
        if (s.upload_dir / "pages.jsonl").exists():
            found[UPLOAD_CORPUS] = s.upload_dir
        return found

    def corpus(self, name: str) -> Corpus:
        with self._lock:
            if name not in self._corpora:
                dirs = self.corpus_dirs()
                if name not in dirs:
                    raise KeyError(f"Unknown corpus '{name}'. Available: {sorted(dirs)}")
                self._corpora[name] = Corpus(dirs[name])
            return self._corpora[name]

    def invalidate(self, name: str) -> None:
        """Drop cached state for a corpus whose pages changed (after an upload)."""
        with self._lock:
            self._corpora.pop(name, None)
            self._registries.pop(name, None)
            for key in [k for k in self._pipelines if k[0] == name]:
                del self._pipelines[key]

    def describe_corpora(self) -> list[dict[str, Any]]:
        out = []
        for name in self.corpus_dirs():
            c = self.corpus(name)
            out.append(
                {
                    "name": name,
                    "n_pages": len(c),
                    "n_documents": len({p.doc_id for p in c.pages}),
                    "text_sources": c.text_sources(),
                    "license": c.meta().get("dataset_license"),
                    "source": c.meta().get("dataset"),
                }
            )
        return out

    # -- retrieval -----------------------------------------------------------
    def strategies(self) -> list[str]:
        d = self.settings.config_dir / "retrieval"
        return sorted(p.stem for p in d.glob("*.yaml")) if d.exists() else []

    def pipeline(self, corpus: str, strategy: str) -> RetrievalPipeline:
        if strategy not in self.strategies():
            raise KeyError(f"Unknown strategy '{strategy}'. Available: {self.strategies()}")
        with self._lock:
            key = (corpus, strategy)
            if key not in self._pipelines:
                c = self.corpus(corpus)
                if corpus not in self._registries:
                    self._registries[corpus] = IndexRegistry(
                        c, self.settings.index_root, self.settings.device
                    )
                cfg = load_retrieval(self.settings.config_dir / "retrieval" / f"{strategy}.yaml")
                self._pipelines[key] = RetrievalPipeline(cfg, self._registries[corpus])
            return self._pipelines[key]

    def hit_payload(self, corpus: str, hits: list[ScoredPage]) -> list[dict[str, Any]]:
        c = self.corpus(corpus)
        out = []
        for rank, h in enumerate(hits, 1):
            p = c.page(h.page_id)
            out.append(
                {
                    "rank": rank,
                    "page_id": h.page_id,
                    "doc_id": p.doc_id,
                    "page_number": p.page_number,
                    "score": h.score,
                    "components": h.components,
                    "image_url": f"/pages/{corpus}/{h.page_id}/image",
                }
            )
        return out

    # -- generation ----------------------------------------------------------
    def generation_cfg(self) -> GenerationCfg:
        if self._gen_cfg is None:
            path = self.settings.config_dir / "generation" / f"{self.settings.generation}.yaml"
            self._gen_cfg = load_generation(path)
        return self._gen_cfg

    def vlm(self):
        with self._lock:
            if self._vlm is None:
                from mmrag.generate.vlm import load_vlm

                self._vlm = load_vlm(self.generation_cfg().vlm)
            return self._vlm

    def rag(self, corpus: str, strategy: str, mode: str | None, max_pages: int | None):
        base = self.generation_cfg()
        ev = base.evidence.model_copy(
            update={k: v for k, v in {"mode": mode, "max_pages": max_pages}.items() if v}
        )
        c = self.corpus(corpus)
        if ev.mode in ("text", "image_text") and not c.has_text(ev.text_source):
            raise KeyError(f"Corpus '{corpus}' has no text source '{ev.text_source}'")
        cfg = base.model_copy(update={"evidence": ev})
        retrieval = None if ev.mode == "closed_book" else self.pipeline(corpus, strategy)
        return RAGPipeline(c, self.vlm(), cfg, retrieval)
