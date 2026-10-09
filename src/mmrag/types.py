"""Core data structures shared by ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

VISUAL_CONTENT_TYPES = frozenset({"table", "chart", "infographic", "image"})


@dataclass(frozen=True)
class Page:
    """One retrievable unit: a document page or a standalone image."""

    page_id: str
    doc_id: str
    page_number: int
    image_path: str  # relative to the corpus directory
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    answer: str | None = None
    # Annotated modality of the gold evidence, e.g. ["Text", "Chart"].
    content_types: tuple[str, ...] = ()
    query_types: tuple[str, ...] = ()
    query_format: str | None = None
    # False for queries constructed to have no supporting evidence in the corpus.
    answerable: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_visual(self) -> bool:
        """True when any gold evidence is non-textual (table, chart, infographic, image)."""
        return any(c.lower() in VISUAL_CONTENT_TYPES for c in self.content_types)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["content_types"] = list(self.content_types)
        d["query_types"] = list(self.query_types)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Query:
        d = dict(d)
        d["content_types"] = tuple(d.get("content_types") or ())
        d["query_types"] = tuple(d.get("query_types") or ())
        return cls(**d)


# query_id -> {page_id: graded relevance (>0 means relevant)}
Qrels = dict[str, dict[str, int]]


@dataclass
class ScoredPage:
    page_id: str
    score: float
    # Per-component ranks/scores for hybrid retrieval, kept for inspection in the UI.
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    """A page selected for generation, with the label shown to the generator."""

    label: str  # "P1", "P2", ... the only identifier the generator sees
    page_id: str
    score: float
    text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    raw: str
    text: str
    abstained: bool
    cited_labels: list[str]
    cited_page_ids: list[str]
    # Labels the model emitted that do not correspond to any supplied evidence.
    invalid_citations: list[str] = field(default_factory=list)
    # Mean token log-probability of the generation; an uncalibrated uncertainty signal.
    mean_logprob: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
