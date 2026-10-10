"""Page captioning with a VLM, producing a text source for caption-based retrieval.

Leakage control: the caption prompt is a fixed string and the only other input
is the page image. This module has no access to queries, answers or qrels.
"""

from __future__ import annotations

import time

from mmrag.config import GenerationCfg
from mmrag.data.corpus import Corpus, read_jsonl
from mmrag.generate.prompts import CAPTION_PROMPT
from mmrag.generate.vlm import VLM, image_part, text_part


def run_captioning(
    corpus: Corpus,
    vlm: VLM,
    cfg: GenerationCfg,
    page_ids: list[str] | None = None,
    save_every: int = 50,
) -> dict[str, str]:
    """Caption pages into the text source named `cfg.name`. Resumable."""
    source = cfg.name
    path = corpus.text_path(source)
    done: dict[str, str] = (
        {r["page_id"]: r["text"] for r in read_jsonl(path)} if path.exists() else {}
    )
    todo = [pid for pid in (page_ids or corpus.page_ids) if pid not in done]
    side = cfg.evidence.image_max_side
    meta = {
        "model": cfg.vlm.model,
        "prompt": CAPTION_PROMPT,
        "image_max_side": side,
        "max_new_tokens": cfg.vlm.max_new_tokens,
    }
    t0 = time.perf_counter()
    for i, pid in enumerate(todo, 1):
        out = vlm.generate(
            [image_part(corpus.image(pid, side)), text_part(CAPTION_PROMPT)],
            image_max_side=side,
        )
        done[pid] = " ".join(out.text.split())
        if i % save_every == 0 or i == len(todo):
            corpus.save_text(source, done, meta={**meta, "n_pages": len(done)})
            rate = (time.perf_counter() - t0) / i
            print(f"  caption {i}/{len(todo)} ({rate:.1f} s/page)", flush=True)
    return done
