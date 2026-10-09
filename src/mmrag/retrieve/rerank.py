"""Cross-encoder reranking of the first-stage candidates over page text."""

from __future__ import annotations

from mmrag.config import RerankCfg
from mmrag.data.corpus import Corpus
from mmrag.runtime import pick_device
from mmrag.text import chunk_words
from mmrag.types import ScoredPage


class CrossEncoderReranker:
    """Rescores the top-N pages; a page's score is that of its best text chunk.

    The reranker reads text only. Pages whose evidence is purely visual can
    therefore be demoted, which is one of the effects the ablations measure.
    """

    def __init__(self, corpus: Corpus, cfg: RerankCfg, device: str = "auto"):
        from sentence_transformers import CrossEncoder

        self.cfg = cfg
        self.name = f"ce[{cfg.model.split('/')[-1]}|{cfg.text_source}]"
        self.model = CrossEncoder(cfg.model, device=pick_device(device))
        self.texts = corpus.load_text(cfg.text_source)

    def rerank(
        self, query: str, candidates: list[ScoredPage], top_n: int | None = None
    ) -> list[ScoredPage]:
        top_n = top_n or self.cfg.top_n
        head, tail = candidates[:top_n], candidates[top_n:]
        if not head:
            return candidates
        pairs: list[tuple[str, str]] = []
        owner: list[int] = []
        for i, c in enumerate(head):
            for chunk in chunk_words(
                self.texts[c.page_id], self.cfg.chunk_size, self.cfg.chunk_overlap
            ):
                pairs.append((query, chunk))
                owner.append(i)
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
        best = [float("-inf")] * len(head)
        for i, s in zip(owner, scores, strict=True):
            best[i] = max(best[i], float(s))
        reranked = [
            ScoredPage(c.page_id, best[i], {**c.components, "first_stage": c.score})
            for i, c in enumerate(head)
        ]
        reranked.sort(key=lambda c: -c.score)
        # Pages beyond top_n keep their first-stage order, below every reranked page.
        floor = min(c.score for c in reranked) - 1.0
        tail = [
            ScoredPage(c.page_id, floor - j, {**c.components, "first_stage": c.score})
            for j, c in enumerate(tail)
        ]
        return reranked + tail
