"""Deterministic on-disk cache for index artefacts.

An index directory is addressed by a hash of everything that determines its
contents: index parameters, the corpus fingerprint and (for text indexes) a
hash of the indexed text. Changing any of them yields a new directory, so a
stale index can never be served for a changed corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("artifacts/index")


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def text_hash(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def index_dir(root: Path, corpus_name: str, kind: str, params: dict[str, Any]) -> Path:
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    key = hashlib.sha256(blob).hexdigest()[:12]
    label = _slug(str(params.get("model") or params.get("text_source") or kind))[:40]
    return root / corpus_name / f"{kind}__{label}__{key}"


def is_built(d: Path) -> bool:
    return (d / "meta.json").exists()


def write_meta(d: Path, params: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    meta = {"params": params, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **(extra or {})}
    # Written last: its presence marks the index as complete.
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def read_meta(d: Path) -> dict[str, Any]:
    return json.loads((d / "meta.json").read_text())
