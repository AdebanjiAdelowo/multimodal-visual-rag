"""FastAPI application.

Endpoints are split by responsibility:

    /search, /search/image   retrieval only
    /recognize               visual recognition of one image (no corpus, no grounding)
    /ask                     retrieval + grounded generation with page citations
    /upload                  add PDFs or images to the upload corpus

Run with:  uvicorn mmrag.api.app:app
"""

from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from mmrag import __version__
from mmrag.api.service import UPLOAD_CORPUS, Service, Settings
from mmrag.ingest.parse import IMAGE_SUFFIXES


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    corpus: str
    strategy: str | None = None
    k: int = Field(10, ge=1, le=50)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    corpus: str
    strategy: str | None = None
    mode: Literal["closed_book", "text", "image", "image_text"] | None = None
    max_pages: int | None = Field(None, ge=1, le=8)


def _open_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, "File is not a readable image") from e


def create_app(settings: Settings | None = None) -> FastAPI:
    svc = Service(settings)
    app = FastAPI(title="Multimodal Visual RAG", version=__version__)
    app.state.service = svc

    def guard(fn, *args, **kwargs):
        """Map lookup and missing-artefact errors to client-facing status codes."""
        try:
            return fn(*args, **kwargs)
        except KeyError as e:
            raise HTTPException(404, str(e.args[0])) from e
        except FileNotFoundError as e:
            raise HTTPException(409, str(e)) from e
        except ImportError as e:
            raise HTTPException(501, str(e)) from e

    async def read_upload(file: UploadFile) -> bytes:
        data = await file.read()
        if len(data) > svc.settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File exceeds {svc.settings.max_upload_mb} MB")
        return data

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "generator": svc.settings.generation}

    @app.get("/corpora")
    def corpora():
        return svc.describe_corpora()

    @app.get("/strategies")
    def strategies():
        return {"default": svc.settings.default_strategy, "available": svc.strategies()}

    @app.post("/search")
    def search(req: SearchRequest):
        strategy = req.strategy or svc.settings.default_strategy
        pipe = guard(svc.pipeline, req.corpus, strategy)
        t0 = time.perf_counter()
        hits = pipe.search(req.query, req.k)
        return {
            "strategy": strategy,
            "latency_ms": (time.perf_counter() - t0) * 1e3,
            "stage_ms": {k: v * 1e3 for k, v in pipe.timer.last().items()},
            "hits": svc.hit_payload(req.corpus, hits),
        }

    @app.post("/search/image")
    async def search_image(
        file: UploadFile = File(...),
        corpus: str = Form(...),
        strategy: str = Form("visual_siglip"),
        k: int = Form(10),
    ):
        image = _open_image(await read_upload(file))
        pipe = guard(svc.pipeline, corpus, strategy)
        try:
            hits = pipe.search_by_image(image, min(max(k, 1), 50))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"strategy": strategy, "hits": svc.hit_payload(corpus, hits)}

    @app.post("/recognize")
    async def recognize(file: UploadFile = File(...)):
        """Describe an uploaded image. Uses no corpus and makes no grounded claims."""
        image = _open_image(await read_upload(file))
        from mmrag.generate.rag import RAGPipeline
        from mmrag.ingest.ocr import normalize_ocr, tesseract_available

        ocr_text = None
        if tesseract_available():
            import pytesseract

            ocr_text = normalize_ocr(pytesseract.image_to_string(image))
        # RAGPipeline.recognize needs only the generator; any corpus object will do.
        out = RAGPipeline.recognize_with(svc.vlm(), image, svc.generation_cfg())
        return {"kind": "recognition", "ocr_text": ocr_text, **out}

    @app.post("/ask")
    def ask(req: AskRequest):
        strategy = req.strategy or svc.settings.default_strategy
        rag = guard(svc.rag, req.corpus, strategy, req.mode, req.max_pages)
        res = rag.answer(req.question)
        hits = svc.hit_payload(req.corpus, res.hits)
        by_id = {h["page_id"]: h for h in hits}
        scores = [h.score for h in res.hits]
        return {
            "kind": "grounded_answer",
            "strategy": strategy,
            "mode": rag.cfg.evidence.mode,
            "answer": res.answer.text,
            "abstained": res.answer.abstained,
            "raw_output": res.answer.raw,
            "citations": [
                {"label": label, **by_id[pid]}
                for label, pid in zip(
                    res.answer.cited_labels, res.answer.cited_page_ids, strict=True
                )
            ],
            "invalid_citations": res.answer.invalid_citations,
            "evidence": [{"label": e.label, **by_id[e.page_id]} for e in res.evidence],
            "hits": hits,
            # Uncalibrated indicators, exposed for inspection rather than as probabilities.
            "uncertainty": {
                "mean_token_logprob": res.answer.mean_logprob,
                "retrieval_top1_margin": scores[0] - scores[1] if len(scores) > 1 else None,
                "has_valid_citation": bool(res.answer.cited_page_ids),
            },
            "tokens": {"input": res.answer.input_tokens, "output": res.answer.output_tokens},
            "timings_s": res.timings_s,
        }

    @app.post("/upload")
    async def upload(files: list[UploadFile] = File(...), ocr: bool = Form(True)):
        from mmrag.data.corpus import Corpus
        from mmrag.ingest.ocr import run_ocr, tesseract_available
        from mmrag.ingest.parse import ingest_files

        allowed = IMAGE_SUFFIXES | {".pdf"}
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, f in enumerate(files):
                # Only the suffix and a sanitised stem of the client filename are used.
                name = Path(f.filename or f"file{i}")
                if name.suffix.lower() not in allowed:
                    raise HTTPException(400, f"Unsupported file type: {name.suffix or '(none)'}")
                safe = "".join(ch if ch.isalnum() else "_" for ch in name.stem)[:60] or "file"
                path = Path(tmp) / f"{i}_{safe}{name.suffix.lower()}"
                path.write_bytes(await read_upload(f))
                paths.append(path)
            try:
                added = ingest_files(paths, svc.settings.upload_dir)
            except Exception as e:
                raise HTTPException(400, f"Could not parse upload: {e}") from e
        ocr_done = False
        if added and ocr and tesseract_available():
            run_ocr(Corpus(svc.settings.upload_dir), progress=False)
            ocr_done = True
        svc.invalidate(UPLOAD_CORPUS)
        return {
            "corpus": UPLOAD_CORPUS,
            "pages_added": [p.page_id for p in added],
            "ocr": ocr_done,
        }

    @app.get("/pages/{corpus}/{page_id}/image")
    def page_image(corpus: str, page_id: str):
        c = guard(svc.corpus, corpus)
        # The path comes from the manifest entry, never from the request string.
        try:
            path = c.image_path(page_id)
        except KeyError as e:
            raise HTTPException(404, f"Unknown page '{page_id}'") from e
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/pages/{corpus}/{page_id}/text")
    def page_text(corpus: str, page_id: str, source: str = "ocr_tesseract"):
        c = guard(svc.corpus, corpus)
        if source not in c.text_sources():
            raise HTTPException(404, f"Unknown text source '{source}'")
        texts = guard(c.load_text, source)
        if page_id not in texts:
            raise HTTPException(404, f"Unknown page '{page_id}'")
        return {"page_id": page_id, "source": source, "text": texts[page_id]}

    return app


app = create_app()
