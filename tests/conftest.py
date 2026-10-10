"""Shared fixtures: a tiny synthetic corpus built from generated images.

Nothing here touches the network, a GPU or any benchmark asset.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from mmrag.data.corpus import Corpus, write_corpus
from mmrag.types import Page, Query

PAGE_TEXT = {
    "doc-p0": "Python variables store values. A variable name cannot start with a digit.",
    "doc-p1": "A for loop iterates over a sequence. The range function yields integers.",
    "doc-p2": "Bar chart of quarterly revenue. Revenue grew in the fourth quarter.",
    "doc-p3": "Binary search halves the search interval and runs in logarithmic time.",
    "doc-p4": "",
}


@pytest.fixture()
def corpus(tmp_path) -> Corpus:
    root = tmp_path / "toy"
    (root / "images").mkdir(parents=True)
    pages = []
    for i, (pid, text) in enumerate(PAGE_TEXT.items()):
        img = Image.new("RGB", (320, 240), "white")
        ImageDraw.Draw(img).text((10, 10), text[:40], fill="black")
        rel = f"images/{pid}.jpg"
        img.save(root / rel)
        pages.append(Page(pid, "doc", i, rel, 320, 240))
    queries = [
        Query("q0", "can a variable name start with a digit", "No.", ("Text",), ("boolean",)),
        Query("q1", "quarterly revenue chart", "Revenue grew in Q4.", ("Chart", "Text")),
    ]
    qrels = {"q0": {"doc-p0": 2}, "q1": {"doc-p2": 2}}
    write_corpus(root, pages, {"dataset": "toy"}, queries, qrels)
    c = Corpus(root)
    c.save_text("ocr_tesseract", PAGE_TEXT)
    return c
