"""Typed configuration. Every retrieval and generation strategy is a YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- retrieval
class BM25Cfg(_Cfg):
    type: Literal["bm25"] = "bm25"
    text_source: str = "ocr_tesseract"
    k1: float = 1.5
    b: float = 0.75
    weight: float = 1.0


class DenseTextCfg(_Cfg):
    type: Literal["dense_text"] = "dense_text"
    text_source: str = "ocr_tesseract"
    model: str = "BAAI/bge-small-en-v1.5"
    # bge v1.5 retrieval instruction; set to "" for models that do not use one.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    chunk_size: int = 180
    chunk_overlap: int = 40
    weight: float = 1.0


class VisualCfg(_Cfg):
    """Single-vector image-text dual encoder (CLIP / SigLIP family)."""

    type: Literal["visual"] = "visual"
    model: str = "google/siglip2-base-patch16-224"
    weight: float = 1.0


class LateInteractionCfg(_Cfg):
    """Multi-vector visual document retriever (ColPali family)."""

    type: Literal["late_interaction"] = "late_interaction"
    model: str = "vidore/colSmol-256M"
    weight: float = 1.0


LeafCfg = Annotated[
    BM25Cfg | DenseTextCfg | VisualCfg | LateInteractionCfg, Field(discriminator="type")
]


class HybridCfg(_Cfg):
    type: Literal["hybrid"] = "hybrid"
    components: list[LeafCfg]
    fusion: Literal["rrf", "zscore"] = "rrf"
    rrf_k: int = 60


RetrieverCfg = Annotated[
    BM25Cfg | DenseTextCfg | VisualCfg | LateInteractionCfg | HybridCfg,
    Field(discriminator="type"),
]


class RerankCfg(_Cfg):
    type: Literal["cross_encoder"] = "cross_encoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    text_source: str = "ocr_tesseract"
    top_n: int = 20
    chunk_size: int = 180
    chunk_overlap: int = 40


class RetrievalCfg(_Cfg):
    name: str
    retriever: RetrieverCfg
    rerank: RerankCfg | None = None
    top_k: int = 10


# --------------------------------------------------------------- generation
class VLMCfg(_Cfg):
    backend: Literal["transformers", "mock"] = "transformers"
    model: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    revision: str | None = None
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    load_in_4bit: bool = False  # CUDA + bitsandbytes only
    max_new_tokens: int = 160
    device: str = "auto"


class EvidenceCfg(_Cfg):
    """How retrieved pages are turned into generator context."""

    mode: Literal["closed_book", "text", "image", "image_text"] = "image"
    max_pages: int = 3
    image_max_side: int = 1024
    text_source: str = "ocr_tesseract"
    max_chars_per_page: int = 2500


class GenerationCfg(_Cfg):
    name: str
    vlm: VLMCfg = VLMCfg()
    evidence: EvidenceCfg = EvidenceCfg()


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_retrieval(path: str | Path) -> RetrievalCfg:
    return RetrievalCfg.model_validate(load_yaml(path))


def load_generation(path: str | Path) -> GenerationCfg:
    return GenerationCfg.model_validate(load_yaml(path))
