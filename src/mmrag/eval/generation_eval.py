"""Generation evaluation under three evidence conditions.

answerable        normal retrieval over the full corpus.
missing_evidence  the same questions with every annotated relevant page withheld
                  from the candidate pool; the correct behaviour is to abstain.
out_of_domain     questions written for a different corpus; correct behaviour is
                  to abstain.

Rankings are computed once per (question, condition) and reused by every
generation system, so text-RAG and image-RAG are compared on identical evidence.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmrag.eval import generation_metrics as gm
from mmrag.eval.stats import auroc, mean_ci
from mmrag.generate.rag import RAGPipeline
from mmrag.retrieve.pipeline import RetrievalPipeline
from mmrag.runtime import accelerator_memory_mb, peak_rss_mb
from mmrag.types import Qrels, Query, ScoredPage

CONDITIONS = ("answerable", "missing_evidence", "out_of_domain")


@dataclass
class Example:
    query: Query
    condition: str
    gold: dict[str, int]
    hits: list[ScoredPage]

    @property
    def key(self) -> str:
        return f"{self.condition}|{self.query.query_id}"


def sample_queries(queries: list[Query], n: int, seed: int = 13) -> list[Query]:
    """Deterministic sample, stratified so visual and text-only questions are both present."""
    if n <= 0 or n >= len(queries):
        return list(queries)
    rng = random.Random(seed)
    visual = [q for q in queries if q.is_visual]
    textual = [q for q in queries if not q.is_visual]
    n_vis = round(n * len(visual) / len(queries))
    picked = rng.sample(visual, min(n_vis, len(visual))) + rng.sample(
        textual, min(n - n_vis, len(textual))
    )
    order = {q.query_id: i for i, q in enumerate(queries)}
    return sorted(picked, key=lambda q: order[q.query_id])


def build_examples(
    retrieval: RetrievalPipeline,
    queries: list[Query],
    qrels: Qrels,
    ood_queries: list[Query],
    depth: int = 10,
) -> list[Example]:
    out: list[Example] = []
    for q in queries:
        gold = qrels[q.query_id]
        out.append(Example(q, "answerable", gold, retrieval.search(q.text, depth)))
        out.append(
            Example(q, "missing_evidence", gold, retrieval.search(q.text, depth, exclude=set(gold)))
        )
    for q in ood_queries:
        out.append(Example(q, "out_of_domain", {}, retrieval.search(q.text, depth)))
    return out


def oracle_examples(queries: list[Query], qrels: Qrels) -> list[Example]:
    """Gold pages as evidence, highest grade first: an upper bound on the generator."""
    out = []
    for q in queries:
        gold = qrels[q.query_id]
        hits = [ScoredPage(p, float(g)) for p, g in sorted(gold.items(), key=lambda kv: -kv[1])]
        out.append(Example(q, "answerable", gold, hits))
    return out


def run_system(
    rag: RAGPipeline,
    system: str,
    examples: list[Example],
    run_path: Path,
    progress_every: int = 10,
) -> list[dict[str, Any]]:
    """Generate answers for every example, appending to a JSONL file (resumable)."""
    run_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if run_path.exists():
        for line in run_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
    with run_path.open("a") as f:
        for i, ex in enumerate(examples, 1):
            if ex.key in done:
                continue
            res = rag.answer(ex.query.text, hits=ex.hits)
            rec = {
                "key": ex.key,
                "system": system,
                "condition": ex.condition,
                "query_id": ex.query.query_id,
                "question": ex.query.text,
                "reference": ex.query.answer,
                "is_visual": ex.query.is_visual,
                "content_types": list(ex.query.content_types),
                "gold": ex.gold,
                "provided": [e.page_id for e in res.evidence],
                "answer": res.answer.to_dict(),
                "timings_s": res.timings_s,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[ex.key] = rec
            if i % progress_every == 0:
                print(f"  [{system}] {i}/{len(examples)}", flush=True)
    return [done[ex.key] for ex in examples]


def score_records(
    records: list[dict[str, Any]],
    nli: gm.NLIScorer | None,
    evidence_texts: dict[str, str],
    support_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Attach per-example metrics. Abstentions score zero on every correctness metric."""
    for r in records:
        a = r["answer"]
        pred, ref = a["text"], r["reference"]
        m: dict[str, Any] = {"abstained": float(a["abstained"])}
        gold = {p for p, g in r["gold"].items() if g > 0}
        m["gold_in_context"] = float(bool(gold & set(r["provided"]))) if gold else None
        m["invalid_citation"] = float(bool(a["invalid_citations"]))
        if r["condition"] == "answerable" and ref:
            m["token_f1"] = gm.token_f1(pred, ref) if not a["abstained"] else 0.0
            m["token_recall"] = gm.token_recall(pred, ref) if not a["abstained"] else 0.0
            m["rouge_l"] = gm.rouge_l(pred, ref) if not a["abstained"] else 0.0
            b = gm.boolean_accuracy(pred, ref)
            m["boolean_accuracy"] = b if b is None or not a["abstained"] else 0.0
            if nli is not None:
                m["answer_entailment"] = (
                    nli.answer_entailment(r["question"], pred, ref) if not a["abstained"] else 0.0
                )
            if not a["abstained"]:
                m.update(gm.citation_scores(a["cited_page_ids"], gold, r["provided"]))
                m["has_citation"] = float(bool(a["cited_page_ids"]))
        if nli is not None and not a["abstained"] and r["provided"]:
            f = nli.faithfulness(pred, [evidence_texts.get(p, "") for p in r["provided"]])
            m["faithfulness"] = f
            m["unsupported"] = None if f is None else float(f < support_threshold)
        r["metrics"] = m
    return records


def _mean(records: list[dict], name: str) -> dict[str, float]:
    vals = [r["metrics"].get(name) for r in records]
    return mean_ci([v for v in vals if v is not None])


def summarize(records: list[dict[str, Any]], rag: RAGPipeline, system: str) -> dict[str, Any]:
    by_cond: dict[str, list[dict]] = {}
    for r in records:
        by_cond.setdefault(r["condition"], []).append(r)

    out: dict[str, Any] = {"system": system, "config": rag.cfg.model_dump(), "conditions": {}}
    quality = [
        "token_f1", "token_recall", "rouge_l", "boolean_accuracy", "answer_entailment",
        "faithfulness", "unsupported", "citation_precision", "citation_recall",
        "has_citation", "invalid_citation", "gold_in_context",
    ]  # fmt: skip
    for cond, recs in by_cond.items():
        block: dict[str, Any] = {"n": len(recs), "abstain_rate": _mean(recs, "abstained")}
        if cond == "answerable":
            for name in quality:
                block[name] = _mean(recs, name)
            for tag, subset in (
                ("visual", [r for r in recs if r["is_visual"]]),
                ("text_only", [r for r in recs if not r["is_visual"]]),
                ("gold_in_context", [r for r in recs if r["metrics"].get("gold_in_context")]),
            ):
                block[f"subset:{tag}"] = {
                    "n": len(subset),
                    **{
                        k: _mean(subset, k)
                        for k in ("token_f1", "rouge_l", "answer_entailment", "abstained")
                    },
                }
        else:
            # No supporting evidence exists, so every non-abstention is an unsupported answer.
            block["unsupported_answer_rate"] = mean_ci(
                [1.0 - r["metrics"]["abstained"] for r in recs]
            )
        out["conditions"][cond] = block

    # Does the confidence signal separate answers given real evidence from answers
    # given none? Label = gold page in context; score = mean token log-probability.
    pool = [
        r
        for r in by_cond.get("answerable", []) + by_cond.get("missing_evidence", [])
        if not r["answer"]["abstained"] and r["answer"]["mean_logprob"] is not None
    ]
    out["confidence"] = {
        "signal": "mean_token_logprob",
        "auroc_gold_in_context": auroc(
            [r["answer"]["mean_logprob"] for r in pool],
            [
                int(bool(r["metrics"].get("gold_in_context")) and r["condition"] == "answerable")
                for r in pool
            ],
        ),
        "n": len(pool),
    }
    gen = [r["timings_s"].get("generation", 0.0) for r in records]
    out["systems"] = {
        "model": rag.vlm.info(),
        "generation_latency_s": mean_ci(gen),
        "input_tokens": mean_ci([r["answer"]["input_tokens"] or 0 for r in records]),
        "output_tokens": mean_ci([r["answer"]["output_tokens"] or 0 for r in records]),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        **{
            k: round(v, 1)
            for k, v in accelerator_memory_mb(rag.vlm.info().get("device", "")).items()
        },
    }
    return out
