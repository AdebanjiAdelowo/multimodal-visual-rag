"""End-to-end pipeline: retrieve -> select evidence -> generate -> parse citations.

Retrieval and generation are separate objects. The generator only ever sees
the selected evidence, labelled P1..Pn; page identifiers are mapped back after
generation so a citation always resolves to a concrete page image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mmrag.config import EvidenceCfg, GenerationCfg
from mmrag.data.corpus import Corpus
from mmrag.generate.prompts import RECOGNITION_PROMPT, build_prompt, parse_answer
from mmrag.generate.vlm import VLM, image_part, text_part
from mmrag.retrieve.pipeline import RetrievalPipeline
from mmrag.runtime import StageTimer
from mmrag.types import Answer, Evidence, ScoredPage


def select_evidence(
    hits: list[ScoredPage], cfg: EvidenceCfg, texts: dict[str, str] | None
) -> list[Evidence]:
    """Keep the top pages within the context budget and attach labels (and text)."""
    out = []
    for i, h in enumerate(hits[: cfg.max_pages], 1):
        text = None
        if texts is not None:
            text = texts.get(h.page_id, "")[: cfg.max_chars_per_page]
        out.append(Evidence(f"P{i}", h.page_id, h.score, text))
    return out


@dataclass
class RAGResult:
    question: str
    answer: Answer
    evidence: list[Evidence]
    hits: list[ScoredPage]
    timings_s: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer.to_dict(),
            "evidence": [e.to_dict() for e in self.evidence],
            "hits": [h.to_dict() for h in self.hits],
            "timings_s": self.timings_s,
        }


class RAGPipeline:
    def __init__(
        self,
        corpus: Corpus,
        vlm: VLM,
        cfg: GenerationCfg,
        retrieval: RetrievalPipeline | None = None,
    ):
        self.corpus = corpus
        self.vlm = vlm
        self.cfg = cfg
        self.retrieval = retrieval
        ev = cfg.evidence
        self._texts = (
            corpus.load_text(ev.text_source) if ev.mode in ("text", "image_text") else None
        )
        self.timer = StageTimer()

    def answer(
        self,
        question: str,
        hits: list[ScoredPage] | None = None,
        exclude: set[str] | None = None,
    ) -> RAGResult:
        """Answer a question. Pass `hits` to bypass retrieval (oracle or cached rankings)."""
        ev_cfg = self.cfg.evidence
        timings: dict[str, float] = {}
        if ev_cfg.mode == "closed_book":
            hits = []
        elif hits is None:
            if self.retrieval is None:
                raise ValueError("No retrieval pipeline configured and no hits supplied")
            with self.timer.stage("retrieval"):
                hits = self.retrieval.search(question, exclude=exclude)
            timings["retrieval"] = self.timer.last()["retrieval"]

        with self.timer.stage("evidence_selection"):
            evidence = select_evidence(hits, ev_cfg, self._texts)
            images = (
                {e.label: self.corpus.image(e.page_id, ev_cfg.image_max_side) for e in evidence}
                if ev_cfg.mode in ("image", "image_text")
                else None
            )
            content = build_prompt(question, evidence, ev_cfg.mode, images)
        timings["evidence_selection"] = self.timer.last()["evidence_selection"]

        out = self.vlm.generate(content, image_max_side=ev_cfg.image_max_side)
        self.timer.add("generation", out.latency_s)
        timings["generation"] = out.latency_s

        ans = parse_answer(out.text, evidence)
        ans.mean_logprob = out.mean_logprob
        ans.input_tokens, ans.output_tokens = out.input_tokens, out.output_tokens
        return RAGResult(question, ans, evidence, hits, timings)

    @staticmethod
    def recognize_with(
        vlm: VLM, image, cfg: GenerationCfg, max_new_tokens: int = 120
    ) -> dict[str, Any]:
        """Visual recognition only: describe an image, with no corpus and no grounding.

        Kept apart from `answer` so the application can show what the model
        perceives separately from what the retrieved evidence supports.
        """
        out = vlm.generate(
            [image_part(image), text_part(RECOGNITION_PROMPT)],
            max_new_tokens=max_new_tokens,
            image_max_side=cfg.evidence.image_max_side,
        )
        return {
            "description": out.text,
            "latency_s": out.latency_s,
            "mean_token_logprob": out.mean_logprob,
        }
