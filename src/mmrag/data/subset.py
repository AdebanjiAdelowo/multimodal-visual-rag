"""Build a smaller, self-contained corpus for experiments too slow for the full one.

The pool keeps every annotated relevant page of a sampled set of queries and
fills the rest with randomly drawn distractor pages. Because it is a real
corpus directory, every strategy can be evaluated on it unchanged, which keeps
comparisons like-for-like (same pages, same queries) for expensive
representations such as VLM captions.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

from mmrag.data.corpus import Corpus, write_corpus, write_jsonl
from mmrag.eval.generation_eval import sample_queries


def make_pool(
    corpus: Corpus, out_dir: Path, n_queries: int = 60, n_pages: int = 400, seed: int = 13
) -> Path:
    queries = sample_queries(corpus.load_queries(), n_queries, seed)
    qrels = {q.query_id: corpus.load_qrels()[q.query_id] for q in queries}
    gold = {p for rel in qrels.values() for p in rel}
    if len(gold) > n_pages:
        raise ValueError(f"{len(gold)} relevant pages exceed the pool size {n_pages}")
    rng = random.Random(seed)
    others = [p for p in corpus.page_ids if p not in gold]
    keep = gold | set(rng.sample(others, min(n_pages - len(gold), len(others))))
    pages = [p for p in corpus.pages if p.page_id in keep]  # manifest order

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    for p in pages:
        dst = out_dir / p.image_path
        if not dst.exists():
            try:
                os.link(corpus.root / p.image_path, dst)  # hard link: no extra disk
            except OSError:
                shutil.copy2(corpus.root / p.image_path, dst)
    meta = {
        **corpus.meta(),
        "subset_of": corpus.name,
        "subset_seed": seed,
        "n_pages": len(pages),
        "n_queries": len(queries),
        "n_relevant_pages": len(gold),
    }
    write_corpus(out_dir, pages, meta, queries, qrels)
    for source in corpus.text_sources():
        texts = corpus.load_text(source)
        write_jsonl(
            out_dir / "text" / f"{source}.jsonl",
            ({"page_id": p.page_id, "text": texts[p.page_id]} for p in pages),
        )
    return out_dir
