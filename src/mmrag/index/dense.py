"""Dense text index: sentence-transformer embeddings of page-text chunks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mmrag.config import DenseTextCfg
from mmrag.data.corpus import Corpus
from mmrag.index import store
from mmrag.runtime import pick_device
from mmrag.text import chunk_words


def max_pool_by_page(chunk_scores: np.ndarray, chunk_page: np.ndarray, n_pages: int) -> np.ndarray:
    """Page score = best chunk score. Every page owns at least one chunk."""
    out = np.full(n_pages, -np.inf, dtype=np.float32)
    np.maximum.at(out, chunk_page, chunk_scores.astype(np.float32))
    return out


class DenseTextIndex:
    def __init__(
        self,
        corpus: Corpus,
        cfg: DenseTextCfg,
        root: Path = store.DEFAULT_ROOT,
        device: str = "auto",
    ):
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        self.name = f"dense[{cfg.text_source}|{cfg.model.split('/')[-1]}]"
        self.n_pages = len(corpus)
        self.model = SentenceTransformer(cfg.model, device=pick_device(device))

        texts = corpus.load_text(cfg.text_source)
        chunks: list[str] = []
        chunk_page: list[int] = []
        for i, pid in enumerate(corpus.page_ids):
            for c in chunk_words(texts[pid], cfg.chunk_size, cfg.chunk_overlap):
                chunks.append(c)
                chunk_page.append(i)
        self.chunk_page = np.asarray(chunk_page, dtype=np.int64)

        params = {
            "model": cfg.model,
            "text_source": cfg.text_source,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "corpus": corpus.fingerprint(),
            "text_hash": store.text_hash(chunks),
        }
        self.dir = store.index_dir(root, corpus.name, "dense_text", params)
        if store.is_built(self.dir):
            self.emb = np.load(self.dir / "embeddings.npy")
        else:
            self.emb = self.model.encode(
                chunks,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=len(chunks) > 500,
                convert_to_numpy=True,
            ).astype(np.float32)
            self.dir.mkdir(parents=True, exist_ok=True)
            np.save(self.dir / "embeddings.npy", self.emb)
            store.write_meta(self.dir, params, {"n_chunks": len(chunks), "dim": self.emb.shape[1]})

    def score_all(self, query: str) -> np.ndarray:
        q = self.model.encode(
            [self.cfg.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        return max_pool_by_page(self.emb @ q, self.chunk_page, self.n_pages)
