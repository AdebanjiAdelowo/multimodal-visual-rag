"""Lexical BM25 index over one text source."""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from mmrag.config import BM25Cfg
from mmrag.data.corpus import Corpus
from mmrag.text import tokenize


class BM25Index:
    """Scores every page for a query. Cheap enough to rebuild at load time."""

    def __init__(self, corpus: Corpus, cfg: BM25Cfg):
        self.cfg = cfg
        self.name = f"bm25[{cfg.text_source}]"
        texts = corpus.load_text(cfg.text_source)
        tokenized = [tokenize(texts[pid]) for pid in corpus.page_ids]
        # rank_bm25 divides by the average document length; guard empty pages.
        self._bm25 = BM25Okapi([t or ["<empty>"] for t in tokenized], k1=cfg.k1, b=cfg.b)

    def score_all(self, query: str) -> np.ndarray:
        toks = tokenize(query)
        if not toks:
            return np.zeros(len(self._bm25.doc_len), dtype=np.float32)
        return np.asarray(self._bm25.get_scores(toks), dtype=np.float32)
