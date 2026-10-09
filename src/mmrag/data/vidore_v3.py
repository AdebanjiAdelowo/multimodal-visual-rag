"""Download and materialise a ViDoRe v3 domain as a local corpus.

ViDoRe v3 (Loison et al., 2026, arXiv:2601.08620) provides page images, graded
human-verified page relevance (qrels, scores 1-2) and reference answers. The
public domains are CC-BY-4.0. Nothing downloaded here is committed to Git.

Leakage control: the page manifest carries only page_id / doc_id / page number.
Query text, reference answers and qrels are written to separate files that no
indexing or captioning code reads.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

from mmrag.data.corpus import write_corpus, write_jsonl
from mmrag.types import Page, Qrels, Query

DOMAINS = {
    "computer_science": "vidore/vidore_v3_computer_science",
    "hr": "vidore/vidore_v3_hr",
    "energy": "vidore/vidore_v3_energy",
    "finance_en": "vidore/vidore_v3_finance_en",
    "pharmaceuticals": "vidore/vidore_v3_pharmaceuticals",
    "physics": "vidore/vidore_v3_physics",
    "industrial": "vidore/vidore_v3_industrial",
}


def _read_config(raw: Path, config: str) -> list[dict]:
    files = sorted((raw / config).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files for config '{config}' under {raw}")
    rows: list[dict] = []
    for f in files:
        rows.extend(pq.read_table(f).to_pylist())
    return rows


# The repositories also carry the source PDFs; only the parquet configs are needed.
CONFIGS = ("corpus", "queries", "qrels", "documents_metadata")


def download(domain: str, raw_dir: Path, configs: tuple[str, ...] = CONFIGS) -> tuple[Path, str]:
    """Fetch the parquet files of a domain. Returns (local dir, resolved commit sha)."""
    repo = DOMAINS[domain]
    patterns = [f"{c}/*" for c in configs]
    local = raw_dir / f"vidore_v3_{domain}"
    path = snapshot_download(
        repo, repo_type="dataset", local_dir=local, allow_patterns=patterns
    )
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(repo).sha or "unknown"
    return Path(path), sha


def _to_queries(rows: list[dict], prefix: str, language: str, answerable: bool) -> list[Query]:
    out = []
    for r in rows:
        if language and (r.get("language") or "").lower() != language:
            continue
        out.append(
            Query(
                query_id=f"{prefix}-q{r['query_id']}",
                text=r["query"],
                answer=r.get("answer") if answerable else None,
                content_types=tuple(r.get("content_type") or ()),
                query_types=tuple(r.get("query_types") or ()),
                query_format=r.get("query_format"),
                answerable=answerable,
                meta={
                    "query_generator": r.get("query_generator"),
                    "source_type": r.get("source_type"),
                    "source_dataset": DOMAINS.get(prefix, prefix),
                },
            )
        )
    return out


def materialize(
    domain: str,
    raw_dir: Path,
    out_dir: Path,
    max_side: int = 1600,
    jpeg_quality: int = 88,
    language: str = "english",
    drop_raw: bool = False,
) -> Path:
    """Write page images, manifest, queries, qrels and the dataset's own page text."""
    raw, sha = download(domain, raw_dir)
    root = out_dir / f"vidore_v3_{domain}"
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    pages: list[Page] = []
    markdown: dict[str, str] = {}
    # Stream the corpus one row group at a time; the page images are large.
    for f in sorted((raw / "corpus").glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=16):
            for r in batch.to_pylist():
                pid = f"{domain}-p{r['corpus_id']}"
                img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                if max(img.size) > max_side:
                    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                rel = f"images/{pid}.jpg"
                img.save(root / rel, "JPEG", quality=jpeg_quality)
                pages.append(
                    Page(
                        page_id=pid,
                        doc_id=r["doc_id"],
                        page_number=int(r["page_number_in_doc"]),
                        image_path=rel,
                        width=img.width,
                        height=img.height,
                    )
                )
                markdown[pid] = r.get("markdown") or ""
    pages.sort(key=lambda p: int(p.page_id.rsplit("p", 1)[1]))

    queries = _to_queries(_read_config(raw, "queries"), domain, language, answerable=True)
    keep = {q.query_id for q in queries}
    qrels: Qrels = {}
    for r in _read_config(raw, "qrels"):
        qid = f"{domain}-q{r['query_id']}"
        if qid in keep and int(r["score"]) > 0:
            qrels.setdefault(qid, {})[f"{domain}-p{r['corpus_id']}"] = int(r["score"])
    queries = [q for q in queries if q.query_id in qrels]

    docs = _read_config(raw, "documents_metadata")
    meta = {
        "dataset": DOMAINS[domain],
        "revision": sha,
        "dataset_license": "CC-BY-4.0",
        "citation": "Loison et al., ViDoRe V3, arXiv:2601.08620",
        "language_filter": language,
        "image_max_side": max_side,
        "jpeg_quality": jpeg_quality,
        "n_pages": len(pages),
        "n_queries": len(queries),
        "n_qrels": sum(len(v) for v in qrels.values()),
        "documents": [
            {k: d.get(k) for k in ("doc_id", "url", "doc_type", "doc_year", "license")}
            for d in docs
        ],
    }
    write_corpus(root, pages, meta, queries, qrels)
    # The dataset ships its own parsed page text. It is kept as a separate,
    # explicitly named text source ("parser_markdown") and treated as a strong
    # document-parser reference, distinct from the OCR this project runs itself.
    write_jsonl(
        root / "text" / "parser_markdown.jsonl",
        ({"page_id": p.page_id, "text": markdown[p.page_id]} for p in pages),
    )
    if drop_raw:
        shutil.rmtree(raw / "corpus", ignore_errors=True)
    return root


def out_of_domain_queries(domain: str, raw_dir: Path, language: str = "english") -> list[Query]:
    """Queries from a different ViDoRe v3 domain, used as unanswerable probes.

    Only the query text is used. Their reference answers are discarded because
    they refer to a corpus that is not indexed.
    """
    raw, _ = download(domain, raw_dir, configs=("queries",))
    return _to_queries(_read_config(raw, "queries"), domain, language, answerable=False)
