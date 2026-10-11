"""Run a retrieval strategy over an annotated query set and aggregate metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from mmrag.eval.retrieval_metrics import KS, per_query_metrics
from mmrag.eval.stats import mean_ci
from mmrag.retrieve.pipeline import RetrievalPipeline
from mmrag.runtime import peak_rss_mb
from mmrag.types import VISUAL_CONTENT_TYPES, Qrels, Query


def strata(q: Query) -> list[str]:
    """Subsets a query belongs to. 'visual' = any gold evidence is non-textual."""
    tags = ["all", "visual" if q.is_visual else "text_only"]
    tags += [f"type:{c.lower()}" for c in q.content_types if c.lower() in VISUAL_CONTENT_TYPES]
    # How the query was written (ViDoRe v3: "human" or "sdg", a synthetic pipeline).
    # Synthetic queries may share wording with the page text they were generated from.
    if q.meta.get("query_generator"):
        tags.append(f"origin:{q.meta['query_generator']}")
    return tags


def aggregate(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[str, list[dict]] = {}
    for r in rows:
        for tag in r["strata"]:
            groups.setdefault(tag, []).append(r)
    for tag, members in sorted(groups.items()):
        out[tag] = {m: mean_ci([r["metrics"][m] for r in members]) for m in metric_names}
        out[tag]["n_queries"] = len(members)
    return out


def evaluate_retrieval(
    pipeline: RetrievalPipeline,
    queries: list[Query],
    qrels: Qrels,
    depth: int = 20,
    ks: tuple[int, ...] = KS,
    warmup: bool = True,
) -> dict[str, Any]:
    """Returns {'summary': aggregates, 'per_query': rows with rankings and metrics}."""
    if warmup and queries:
        # Exclude one-off costs (lazy model init, MPS graph compilation) from latency.
        pipeline.search(queries[0].text, depth)
        pipeline.timer.samples.clear()

    rows = []
    for q in queries:
        t0 = time.perf_counter()
        hits = pipeline.search(q.text, depth)
        dt = time.perf_counter() - t0
        ranked = [h.page_id for h in hits]
        rows.append(
            {
                "query_id": q.query_id,
                "strata": strata(q),
                "ranked": ranked,
                "scores": [round(h.score, 6) for h in hits],
                "metrics": per_query_metrics(ranked, qrels[q.query_id], ks),
                "latency_s": dt,
            }
        )
    names = list(rows[0]["metrics"]) if rows else []
    summary = {
        "strategy": pipeline.cfg.name,
        "config": pipeline.cfg.model_dump(),
        "corpus": pipeline.corpus.name,
        "n_pages": len(pipeline.corpus),
        "n_queries": len(rows),
        "metrics": aggregate(rows, names),
        "latency_ms": {
            "end_to_end": mean_ci([r["latency_s"] * 1e3 for r in rows]),
            "stages": pipeline.timer.summary(),
        },
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }
    return {"summary": summary, "per_query": rows}


def save_run(result: dict[str, Any], summary_dir: Path, run_dir: Path) -> None:
    name = result["summary"]["strategy"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / f"{name}.json").write_text(json.dumps(result["summary"], indent=2))
    with (run_dir / f"{name}.jsonl").open("w") as f:
        for r in result["per_query"]:
            f.write(json.dumps(r) + "\n")
