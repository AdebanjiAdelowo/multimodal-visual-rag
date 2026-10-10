"""Compile result JSON files into Markdown tables.

The report is generated only from files that exist, so it cannot describe an
experiment that was not run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from mmrag.eval.stats import paired_bootstrap

# (baseline, candidate, question the comparison answers)
COMPARISONS = [
    ("bm25_ocr", "dense_ocr", "lexical vs dense text"),
    ("bm25_ocr", "bm25_caption", "OCR vs captions (lexical)"),
    ("dense_ocr", "dense_caption", "OCR vs captions (dense)"),
    ("bm25_ocr", "bm25_parser", "Tesseract OCR vs dataset parser text"),
    ("dense_ocr", "visual_siglip", "dense text vs single-vector visual"),
    ("dense_ocr", "late_colsmol", "dense text vs late-interaction visual"),
    ("visual_siglip", "late_colsmol", "single-vector vs late-interaction visual"),
    ("dense_ocr", "hybrid_text", "dense vs text hybrid"),
    ("hybrid_text", "hybrid_text_siglip", "adding SigLIP to the text hybrid"),
    ("hybrid_text", "hybrid_text_colsmol", "adding ColSmol to the text hybrid"),
    ("hybrid_text", "hybrid_text_rerank", "reranking on/off (text hybrid)"),
    ("hybrid_text_colsmol", "hybrid_text_colsmol_rerank", "reranking on/off (multimodal hybrid)"),
]


def _fmt(ci: dict[str, Any] | None, digits: int = 3, with_ci: bool = False) -> str:
    if not ci or ci.get("n", 0) == 0 or math.isnan(ci["mean"]):
        return "n/a"
    s = f"{ci['mean']:.{digits}f}"
    return f"{s} [{ci['lo']:.{digits}f}, {ci['hi']:.{digits}f}]" if with_ci else s


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _load_run(path: Path) -> dict[str, dict]:
    return {
        (r := json.loads(line))["query_id"]: r
        for line in path.read_text().splitlines()
        if line.strip()
    }


def retrieval_section(corpus_dir: Path, runs_dir: Path) -> str:
    summaries = [json.loads(p.read_text()) for p in sorted(corpus_dir.glob("*.json"))]
    if not summaries:
        return ""
    summaries.sort(key=lambda s: -s["metrics"]["all"]["ndcg@10"]["mean"])
    first = summaries[0]
    vis_n = first["metrics"].get("visual", {}).get("n_queries", 0)
    txt_n = first["metrics"].get("text_only", {}).get("n_queries", 0)
    out = [
        f"### Retrieval: `{corpus_dir.name}`",
        "",
        f"{first['n_pages']} pages, {first['n_queries']} queries "
        f"({vis_n} with non-text gold evidence, {txt_n} text-only). "
        "Brackets are 95% bootstrap intervals over queries.",
        "",
    ]
    rows = []
    for s in summaries:
        m = s["metrics"]
        a = m["all"]
        rows.append(
            [
                f"`{s['strategy']}`",
                _fmt(a["recall@1"]),
                _fmt(a["recall@5"]),
                _fmt(a["recall@10"]),
                _fmt(a["ndcg@10"], with_ci=True),
                _fmt(a["mrr@10"]),
                _fmt(a["full_recall@10"]),
                _fmt(m.get("visual", {}).get("ndcg@10")),
                _fmt(m.get("text_only", {}).get("ndcg@10")),
                _fmt(s["latency_ms"]["end_to_end"], 1),
            ]
        )
    out.append(
        _table(
            [
                "Strategy",
                "R@1",
                "R@5",
                "R@10",
                "nDCG@10",
                "MRR@10",
                "Full R@10",
                "nDCG@10 visual",
                "nDCG@10 text-only",
                "ms/query",
            ],
            rows,
        )  # fmt: skip
    )

    comp_rows = []
    for base, cand, label in COMPARISONS:
        pa, pb = runs_dir / f"{base}.jsonl", runs_dir / f"{cand}.jsonl"
        if not (pa.exists() and pb.exists()):
            continue
        ra, rb = _load_run(pa), _load_run(pb)
        ids = sorted(set(ra) & set(rb))
        row = [label, f"`{base}` -> `{cand}`"]
        for subset in ("all", "visual", "text_only"):
            keep = [i for i in ids if subset in ra[i]["strata"]]
            d = paired_bootstrap(
                [ra[i]["metrics"]["ndcg@10"] for i in keep],
                [rb[i]["metrics"]["ndcg@10"] for i in keep],
            )
            row.append(f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], p={d['p']:.3f}")
        comp_rows.append(row)
    if comp_rows:
        out += [
            "",
            "Paired bootstrap differences in nDCG@10 (candidate minus baseline, same queries):",
            "",
            _table(["Question", "Comparison", "All", "Visual", "Text-only"], comp_rows),
        ]
    return "\n".join(out)


def generation_section(corpus_dir: Path) -> str:
    summaries = [json.loads(p.read_text()) for p in sorted(corpus_dir.glob("*.json"))]
    if not summaries:
        return ""
    out = [f"### Generation: `{corpus_dir.name}`", ""]
    s0 = summaries[0]
    n = {k: v["n"] for k, v in s0["conditions"].items()}
    out += [f"Retrieval strategy: `{s0.get('retrieval')}`. Examples per condition: {n}.", ""]
    quality, behaviour, systems = [], [], []
    for s in summaries:
        c = s["conditions"]
        a = c.get("answerable", {})
        name = f"`{s['system']}`"
        quality.append(
            [name, _fmt(a.get("token_f1"), with_ci=True), _fmt(a.get("rouge_l")),
             _fmt(a.get("answer_entailment")), _fmt(a.get("boolean_accuracy")),
             _fmt(a.get("subset:visual", {}).get("token_f1")),
             _fmt(a.get("subset:text_only", {}).get("token_f1")),
             _fmt(a.get("subset:gold_in_context", {}).get("token_f1"))]
        )  # fmt: skip
        behaviour.append(
            [name, _fmt(a.get("faithfulness")), _fmt(a.get("unsupported")),
             _fmt(a.get("citation_precision")), _fmt(a.get("citation_recall")),
             _fmt(a.get("has_citation")), _fmt(a.get("abstain_rate")),
             _fmt(c.get("missing_evidence", {}).get("abstain_rate")),
             _fmt(c.get("out_of_domain", {}).get("abstain_rate")),
             f"{s['confidence']['auroc_gold_in_context']:.3f}"
             if not math.isnan(s["confidence"]["auroc_gold_in_context"]) else "n/a"]
        )  # fmt: skip
        sy = s["systems"]
        mem = sy.get("mps_driver_allocated_mb") or sy.get("cuda_peak_allocated_mb")
        systems.append(
            [name, sy["model"].get("model", "?"), str(sy["model"].get("params_m", "?")),
             str(sy["model"].get("dtype", "?")), _fmt(sy["generation_latency_s"], 2),
             _fmt(sy["input_tokens"], 0), _fmt(sy["output_tokens"], 0),
             f"{sy['peak_rss_mb']:.0f}", f"{mem:.0f}" if mem else "n/a"]
        )  # fmt: skip
    out += [
        "Answer quality on answerable questions (abstentions score 0):",
        "",
        _table(["System", "Token F1", "ROUGE-L", "NLI entailment", "Yes/no acc.",
                "F1 visual", "F1 text-only", "F1 gold in context"], quality),
        "",
        "Grounding and abstention:",
        "",
        _table(["System", "Faithfulness", "Unsupported rate", "Citation P", "Citation R",
                "Cites a page", "Abstain (answerable)", "Abstain (evidence withheld)",
                "Abstain (out of domain)", "Confidence AUROC"], behaviour),
        "",
        "Cost:",
        "",
        _table(["System", "Model", "Params (M)", "dtype", "Generation s", "Input tokens",
                "Output tokens", "Peak RSS MB", "Accelerator MB"], systems),
    ]  # fmt: skip
    return "\n".join(out)


def build_report(results: Path, runs: Path) -> str:
    parts = [
        "# Results summary",
        "",
        "Generated by `mmrag report` from the JSON files in `results/`.",
        "",
    ]
    hw = None
    for kind in ("retrieval", "generation"):
        root = results / kind
        if not root.exists():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            section = (
                retrieval_section(d, runs / "retrieval" / d.name)
                if kind == "retrieval"
                else generation_section(d)
            )
            if not section:
                continue
            parts += [section, ""]
            if hw is None:
                hw = json.loads(next(d.glob("*.json")).read_text()).get("hardware")
    if hw:
        parts += ["### Hardware", "", "```json", json.dumps(hw, indent=2), "```", ""]
    return "\n".join(parts)
