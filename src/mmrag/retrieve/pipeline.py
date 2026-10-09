"""Configurable retrieval pipeline: first stage (single or hybrid) plus optional rerank."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from mmrag.config import (
    BM25Cfg,
    DenseTextCfg,
    HybridCfg,
    LateInteractionCfg,
    RetrievalCfg,
    VisualCfg,
)
from mmrag.data.corpus import Corpus
from mmrag.index import store
from mmrag.retrieve.fusion import rrf, zscore_fusion
from mmrag.runtime import StageTimer
from mmrag.types import ScoredPage


class Scorer(Protocol):
    name: str

    def score_all(self, query: str) -> np.ndarray:
        """One score per corpus page, in manifest order."""
        ...


class IndexRegistry:
    """Builds each index once and shares it between pipelines.

    Hybrid strategies reuse the indexes of their single-retriever baselines, so
    a benchmark over many strategies loads each model a single time.
    """

    def __init__(self, corpus: Corpus, root: Path = store.DEFAULT_ROOT, device: str = "auto"):
        self.corpus = corpus
        self.root = root
        self.device = device
        self._cache: dict[str, object] = {}

    def _get(self, key: str, factory):
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def leaf(self, cfg) -> Scorer:
        key = cfg.model_dump_json(exclude={"weight"})
        if isinstance(cfg, BM25Cfg):
            from mmrag.index.bm25 import BM25Index

            return self._get(key, lambda: BM25Index(self.corpus, cfg))
        if isinstance(cfg, DenseTextCfg):
            from mmrag.index.dense import DenseTextIndex

            return self._get(key, lambda: DenseTextIndex(self.corpus, cfg, self.root, self.device))
        if isinstance(cfg, VisualCfg):
            from mmrag.index.visual import VisualIndex

            return self._get(key, lambda: VisualIndex(self.corpus, cfg, self.root, self.device))
        if isinstance(cfg, LateInteractionCfg):
            from mmrag.index.late import LateInteractionIndex

            return self._get(
                key, lambda: LateInteractionIndex(self.corpus, cfg, self.root, self.device)
            )
        raise TypeError(f"Unknown retriever config: {type(cfg).__name__}")

    def reranker(self, cfg):
        from mmrag.retrieve.rerank import CrossEncoderReranker

        return self._get(
            "rerank:" + cfg.model_dump_json(exclude={"top_n"}),
            lambda: CrossEncoderReranker(self.corpus, cfg, self.device),
        )


class RetrievalPipeline:
    def __init__(self, cfg: RetrievalCfg, registry: IndexRegistry):
        self.cfg = cfg
        self.corpus = registry.corpus
        self.page_ids = self.corpus.page_ids
        self._pos = {pid: i for i, pid in enumerate(self.page_ids)}
        r = cfg.retriever
        if isinstance(r, HybridCfg):
            self.scorers = [registry.leaf(c) for c in r.components]
            self.weights = [c.weight for c in r.components]
        else:
            self.scorers = [registry.leaf(r)]
            self.weights = [1.0]
        self.reranker = registry.reranker(cfg.rerank) if cfg.rerank else None
        self.timer = StageTimer()

    def search(
        self, query: str, k: int | None = None, exclude: set[str] | None = None
    ) -> list[ScoredPage]:
        """Rank pages for a query.

        `exclude` removes pages from the candidate pool before ranking. The
        evaluation uses it to build the missing-evidence condition, where the
        gold pages are withheld and the system should abstain.
        """
        k = k or self.cfg.top_k
        per: list[np.ndarray] = []
        for s in self.scorers:
            with self.timer.stage(f"retrieve:{s.name}"):
                per.append(s.score_all(query))

        if len(per) == 1:
            fused = per[0].astype(np.float32).copy()
        else:
            with self.timer.stage("fusion"):
                r = self.cfg.retriever
                fused = (
                    rrf(per, self.weights, r.rrf_k)
                    if r.fusion == "rrf"
                    else zscore_fusion(per, self.weights)
                )
        if exclude:
            idx = [self._pos[p] for p in exclude if p in self._pos]
            fused[idx] = -np.inf

        depth = max(k, self.cfg.rerank.top_n) if self.reranker else k
        depth = min(depth, len(fused))
        top = np.argpartition(-fused, depth - 1)[:depth]
        top = top[np.argsort(-fused[top], kind="stable")]
        results = [
            ScoredPage(
                self.page_ids[i],
                float(fused[i]),
                {s.name: float(p[i]) for s, p in zip(self.scorers, per, strict=True)},
            )
            for i in top
            if np.isfinite(fused[i])
        ]
        if self.reranker:
            with self.timer.stage(f"rerank:{self.reranker.name}"):
                # top_n is passed per call: the registry shares one model across configs.
                results = self.reranker.rerank(query, results, self.cfg.rerank.top_n)
        return results[:k]

    def search_by_image(self, image, k: int | None = None) -> list[ScoredPage]:
        """Query-by-image over the first component that supports it (visual indexes)."""
        k = k or self.cfg.top_k
        for s in self.scorers:
            if hasattr(s, "score_image"):
                scores = s.score_image(image)
                top = np.argsort(-scores, kind="stable")[:k]
                return [ScoredPage(self.page_ids[i], float(scores[i]), {}) for i in top]
        raise ValueError(f"Strategy '{self.cfg.name}' has no image-queryable index")
