"""Command-line entry point: `mmrag <command>`."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

CorpusOpt = typer.Option(..., "--corpus", "-c", help="Corpus directory (contains pages.jsonl).")
IndexRoot = typer.Option(Path("artifacts/index"), help="Where index artefacts are cached.")


@app.command()
def hw() -> None:
    """Print the hardware and library summary used in result files."""
    from mmrag.runtime import hardware_summary

    typer.echo(json.dumps(hardware_summary(), indent=2))


@app.command("fetch-vidore")
def fetch_vidore(
    domain: str = typer.Option("computer_science"),
    raw_dir: Path = typer.Option(Path("data/raw")),
    out_dir: Path = typer.Option(Path("data/processed")),
    max_side: int = typer.Option(1600, help="Longest image side kept on disk."),
    drop_raw: bool = typer.Option(False, help="Delete the raw corpus parquet afterwards."),
) -> None:
    """Download a ViDoRe v3 domain and materialise it as a local corpus."""
    from mmrag.data.vidore_v3 import materialize

    root = materialize(domain, raw_dir, out_dir, max_side=max_side, drop_raw=drop_raw)
    typer.echo(f"corpus ready: {root}")


@app.command()
def ingest(
    files: list[Path] = typer.Argument(..., help="PDF or image files."),
    corpus: Path = CorpusOpt,
    ocr: bool = typer.Option(True, help="Run OCR on the new pages."),
) -> None:
    """Add PDFs or images to a corpus directory."""
    from mmrag.data.corpus import Corpus
    from mmrag.ingest.ocr import run_ocr
    from mmrag.ingest.parse import ingest_files

    added = ingest_files(files, corpus)
    typer.echo(f"added {len(added)} pages to {corpus}")
    if ocr and added:
        run_ocr(Corpus(corpus))


@app.command()
def ocr(corpus: Path = CorpusOpt, workers: int = 6, psm: int = 3) -> None:
    """Run Tesseract OCR over every page (resumable)."""
    from mmrag.data.corpus import Corpus
    from mmrag.ingest.ocr import run_ocr

    texts = run_ocr(Corpus(corpus), psm=psm, workers=workers)
    empty = sum(1 for t in texts.values() if not t.strip())
    typer.echo(f"ocr done: {len(texts)} pages, {empty} empty")


@app.command()
def caption(
    corpus: Path = CorpusOpt,
    config: Path = typer.Option(Path("configs/generation/caption_smolvlm_500m.yaml")),
    limit: int = typer.Option(0, help="Caption only the first N pages (0 = all)."),
) -> None:
    """Generate one caption per page with a VLM (resumable)."""
    from mmrag.config import load_generation
    from mmrag.data.corpus import Corpus
    from mmrag.generate.vlm import load_vlm
    from mmrag.ingest.caption import run_captioning

    cfg = load_generation(config)
    c = Corpus(corpus)
    ids = c.page_ids[:limit] if limit else None
    out = run_captioning(c, load_vlm(cfg.vlm), cfg, page_ids=ids)
    typer.echo(f"captions: {len(out)} pages -> text source '{cfg.name}'")


@app.command("eval-retrieval")
def eval_retrieval(
    configs: list[Path] = typer.Argument(..., help="Retrieval strategy YAML files."),
    corpus: Path = CorpusOpt,
    index_root: Path = IndexRoot,
    out: Path = typer.Option(Path("results/retrieval")),
    runs: Path = typer.Option(Path("artifacts/runs/retrieval")),
    depth: int = typer.Option(20, help="Ranking depth stored per query."),
    limit: int = typer.Option(0, help="Evaluate only the first N queries (0 = all)."),
    device: str = "auto",
) -> None:
    """Evaluate retrieval strategies on the corpus query set."""
    from mmrag.config import load_retrieval
    from mmrag.data.corpus import Corpus
    from mmrag.eval.retrieval_eval import evaluate_retrieval, save_run
    from mmrag.retrieve.pipeline import IndexRegistry, RetrievalPipeline
    from mmrag.runtime import hardware_summary

    c = Corpus(corpus)
    queries, qrels = c.load_queries(), c.load_qrels()
    if limit:
        queries = queries[:limit]
    registry = IndexRegistry(c, index_root, device)
    hwinfo = hardware_summary()
    for path in configs:
        cfg = load_retrieval(path)
        try:
            pipe = RetrievalPipeline(cfg, registry)
        except (FileNotFoundError, ImportError) as e:
            typer.echo(f"[skip] {cfg.name}: {e}")
            continue
        res = evaluate_retrieval(pipe, queries, qrels, depth=depth)
        res["summary"]["hardware"] = hwinfo
        save_run(res, out / c.name, runs / c.name)
        m = res["summary"]["metrics"]["all"]
        lat = res["summary"]["latency_ms"]["end_to_end"]["mean"]
        typer.echo(
            f"{cfg.name:28s} R@1={m['recall@1']['mean']:.3f} R@5={m['recall@5']['mean']:.3f} "
            f"R@10={m['recall@10']['mean']:.3f} nDCG@10={m['ndcg@10']['mean']:.3f} "
            f"MRR@10={m['mrr@10']['mean']:.3f} | {lat:.1f} ms/query"
        )


@app.command("make-pool")
def make_pool_cmd(
    corpus: Path = CorpusOpt,
    n_queries: int = 60,
    n_pages: int = 400,
    seed: int = 13,
) -> None:
    """Create a smaller corpus (sampled queries, their gold pages, random distractors)."""
    from mmrag.data.corpus import Corpus
    from mmrag.data.subset import make_pool

    out = corpus.parent / f"{corpus.name}_pool{n_pages}"
    make_pool(Corpus(corpus), out, n_queries, n_pages, seed)
    typer.echo(f"pool corpus ready: {out}")


@app.command("fetch-ood")
def fetch_ood(
    corpus: Path = CorpusOpt,
    domain: str = typer.Option("hr", help="ViDoRe v3 domain to borrow query text from."),
    raw_dir: Path = typer.Option(Path("data/raw")),
) -> None:
    """Download out-of-domain queries (text only) used as unanswerable probes."""
    from mmrag.data.corpus import write_jsonl
    from mmrag.data.vidore_v3 import out_of_domain_queries

    qs = out_of_domain_queries(domain, raw_dir)
    write_jsonl(corpus / "ood_queries.jsonl", (q.to_dict() for q in qs))
    typer.echo(f"{len(qs)} out-of-domain queries -> {corpus / 'ood_queries.jsonl'}")


@app.command("eval-generation")
def eval_generation(
    configs: list[Path] = typer.Argument(..., help="Generation system YAML files."),
    corpus: Path = CorpusOpt,
    retrieval: Path = typer.Option(Path("configs/retrieval/hybrid_text.yaml")),
    index_root: Path = IndexRoot,
    out: Path = typer.Option(Path("results/generation")),
    runs: Path = typer.Option(Path("artifacts/runs/generation")),
    n_queries: int = typer.Option(60, help="Answerable questions sampled (0 = all)."),
    n_ood: int = typer.Option(30, help="Out-of-domain questions (needs fetch-ood)."),
    seed: int = 13,
    oracle: bool = typer.Option(False, help="Also run image systems with the gold pages."),
    max_pages: int = typer.Option(0, help="Override evidence.max_pages (ablation)."),
    image_max_side: int = typer.Option(0, help="Override evidence.image_max_side (ablation)."),
    max_chars: int = typer.Option(0, help="Override evidence.max_chars_per_page (ablation)."),
    nli: bool = typer.Option(True, help="Score faithfulness and entailment with an NLI model."),
    faith_source: str = typer.Option("parser_markdown", help="Page text used for faithfulness."),
    device: str = "auto",
) -> None:
    """Evaluate generation systems on answerable, missing-evidence and out-of-domain questions."""
    import random

    from mmrag.config import load_generation, load_retrieval
    from mmrag.data.corpus import Corpus
    from mmrag.eval import generation_eval as ge
    from mmrag.eval.generation_metrics import NLIScorer
    from mmrag.generate.rag import RAGPipeline
    from mmrag.generate.vlm import load_vlm
    from mmrag.retrieve.pipeline import IndexRegistry, RetrievalPipeline
    from mmrag.runtime import hardware_summary, release_accelerator_memory

    c = Corpus(corpus)
    qrels = c.load_qrels()
    queries = ge.sample_queries(c.load_queries(), n_queries, seed)
    ood = c.load_queries("ood_queries") if (corpus / "ood_queries.jsonl").exists() else []
    ood = random.Random(seed).sample(ood, min(n_ood, len(ood)))
    rcfg = load_retrieval(retrieval)
    pipe = RetrievalPipeline(rcfg, IndexRegistry(c, index_root, device))
    examples = ge.build_examples(pipe, queries, qrels, ood)
    oracle_ex = ge.oracle_examples(queries, qrels)
    typer.echo(f"{len(queries)} answerable, {len(ood)} out-of-domain, retrieval={rcfg.name}")

    scorer = NLIScorer() if nli else None
    faith = c.load_text(faith_source if c.has_text(faith_source) else "ocr_tesseract")
    overrides = {
        k: v
        for k, v in {
            "max_pages": max_pages,
            "image_max_side": image_max_side,
            "max_chars_per_page": max_chars,
        }.items()
        if v
    }
    suffix = "".join(
        f"__{tag}{v}" for tag, v in (("k", max_pages), ("s", image_max_side), ("c", max_chars)) if v
    )
    hwinfo = hardware_summary()
    vlms: dict[str, object] = {}
    for path in configs:
        cfg = load_generation(path)
        cfg = cfg.model_copy(update={"evidence": cfg.evidence.model_copy(update=overrides)})
        key = cfg.vlm.model_dump_json()
        if key not in vlms:
            vlms.clear()  # one generator in memory at a time
            release_accelerator_memory()
            vlms[key] = load_vlm(cfg.vlm)
        rag = RAGPipeline(c, vlms[key], cfg)
        variants = [(f"{cfg.name}{suffix}", examples)]
        if cfg.evidence.mode == "closed_book":
            # Withholding pages cannot change a prompt that contains none.
            variants = [(cfg.name, [e for e in examples if e.condition != "missing_evidence"])]
        elif oracle and cfg.evidence.mode == "image":
            variants.append((f"{cfg.name}{suffix}__oracle", oracle_ex))
        for system, exs in variants:
            recs = ge.run_system(rag, system, exs, runs / c.name / f"{system}.jsonl")
            recs = ge.score_records(recs, scorer, faith)
            summary = ge.summarize(recs, rag, system)
            summary.update(
                retrieval=rcfg.name, corpus=c.name, seed=seed, hardware=hwinfo,
                faithfulness_text_source=faith_source if scorer else None,
            )  # fmt: skip
            (out / c.name).mkdir(parents=True, exist_ok=True)
            (out / c.name / f"{system}.json").write_text(json.dumps(summary, indent=2))
            with (runs / c.name / f"{system}.scored.jsonl").open("w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            a = summary["conditions"]["answerable"]
            miss = summary["conditions"].get("missing_evidence", {})
            typer.echo(
                f"{system:40s} F1={a['token_f1']['mean']:.3f} RL={a['rouge_l']['mean']:.3f} "
                f"abstain={a['abstain_rate']['mean']:.2f} "
                f"abstain|missing={miss.get('abstain_rate', {}).get('mean', float('nan')):.2f} "
                f"gen={summary['systems']['generation_latency_s']['mean']:.1f}s"
            )


@app.command()
def ask(
    question: str = typer.Argument(...),
    corpus: Path = CorpusOpt,
    retrieval: Path = typer.Option(Path("configs/retrieval/hybrid_text.yaml")),
    generation: Path = typer.Option(Path("configs/generation/smolvlm_500m_image_rag.yaml")),
    index_root: Path = IndexRoot,
) -> None:
    """Answer one question with retrieved evidence and print the cited pages."""
    from mmrag.config import load_generation, load_retrieval
    from mmrag.data.corpus import Corpus
    from mmrag.generate.rag import RAGPipeline
    from mmrag.generate.vlm import load_vlm
    from mmrag.retrieve.pipeline import IndexRegistry, RetrievalPipeline

    c = Corpus(corpus)
    gcfg = load_generation(generation)
    pipe = RetrievalPipeline(load_retrieval(retrieval), IndexRegistry(c, index_root))
    res = RAGPipeline(c, load_vlm(gcfg.vlm), gcfg, pipe).answer(question)
    typer.echo("INSUFFICIENT EVIDENCE" if res.answer.abstained else res.answer.text)
    for e in res.evidence:
        p = c.page(e.page_id)
        mark = "*" if e.page_id in res.answer.cited_page_ids else " "
        typer.echo(
            f" {mark} [{e.label}] {p.doc_id} p.{p.page_number + 1}  ({c.image_path(e.page_id)})"
        )
    typer.echo(f"timings: { {k: round(v, 2) for k, v in res.timings_s.items()} }")


@app.command()
def report(
    results: Path = typer.Option(Path("results")),
    runs: Path = typer.Option(Path("artifacts/runs")),
    out: Path = typer.Option(Path("results/SUMMARY.md")),
) -> None:
    """Compile result JSON files into Markdown tables (only what was actually run)."""
    from mmrag.eval.report import build_report

    out.write_text(build_report(results, runs))
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
