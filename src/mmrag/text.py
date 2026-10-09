"""Tokenisation and chunking shared by the lexical and dense text indexes."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[._'-][a-z0-9]+)*")

# Small closed-class stopword list. Kept short on purpose: aggressive stopping
# hurts short technical queries ("for loop", "if else").
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was were what which with".split()
)


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def chunk_words(text: str, size: int = 180, overlap: int = 40) -> list[str]:
    """Split text into overlapping word windows. Empty text yields one empty chunk.

    A page is always represented by at least one chunk so that every page has a
    row in every index and page scores can be aggregated uniformly.
    """
    if size <= overlap:
        raise ValueError("size must exceed overlap")
    words = text.split()
    if not words:
        return [""]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks
