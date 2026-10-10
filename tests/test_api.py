import io
import shutil
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from mmrag.api.app import create_app  # noqa: E402
from mmrag.api.service import Settings  # noqa: E402

CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.fixture()
def client(corpus, tmp_path):
    root = tmp_path / "processed"
    root.mkdir()
    shutil.move(str(corpus.root), root / "toy")
    settings = Settings(
        corpus_root=root,
        upload_dir=tmp_path / "uploads",
        config_dir=CONFIGS,
        index_root=tmp_path / "index",
        default_strategy="bm25_ocr",
        generation="mock",
    )
    return TestClient(create_app(settings))


def png_bytes(color="white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, "PNG")
    return buf.getvalue()


def test_health_and_listing(client):
    assert client.get("/health").json()["status"] == "ok"
    corpora = client.get("/corpora").json()
    assert corpora[0]["name"] == "toy" and corpora[0]["n_pages"] == 5
    assert "bm25_ocr" in client.get("/strategies").json()["available"]


def test_search_returns_ranked_pages_with_images(client):
    r = client.post("/search", json={"query": "variable name digit", "corpus": "toy", "k": 3})
    hits = r.json()["hits"]
    assert r.status_code == 200 and hits[0]["page_id"] == "doc-p0" and hits[0]["rank"] == 1
    img = client.get(hits[0]["image_url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"


def test_ask_returns_answer_with_resolvable_citations(client):
    r = client.post("/ask", json={"question": "variable name digit", "corpus": "toy"}).json()
    assert r["kind"] == "grounded_answer" and not r["abstained"]
    assert r["citations"][0]["page_id"] == "doc-p0"
    assert {e["label"] for e in r["evidence"]} >= {c["label"] for c in r["citations"]}
    closed = client.post(
        "/ask", json={"question": "anything", "corpus": "toy", "mode": "closed_book"}
    ).json()
    assert closed["abstained"] and closed["evidence"] == []


def test_errors_are_client_errors(client):
    assert client.post("/search", json={"query": "x", "corpus": "nope"}).status_code == 404
    bad = {"query": "x", "corpus": "toy", "strategy": "../../etc/passwd"}
    assert client.post("/search", json=bad).status_code == 404
    assert client.get("/pages/toy/..%2F..%2Fsecret/image").status_code == 404
    assert client.post("/search", json={"query": "", "corpus": "toy"}).status_code == 422
    r = client.post("/recognize", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_recognition_is_separate_from_grounding(client):
    r = client.post("/recognize", files={"file": ("x.png", png_bytes(), "image/png")}).json()
    assert r["kind"] == "recognition" and "citations" not in r and "description" in r


def test_upload_creates_searchable_corpus(client):
    r = client.post(
        "/upload", files=[("files", ("a.png", png_bytes(), "image/png"))], data={"ocr": "false"}
    )
    assert r.status_code == 200 and len(r.json()["pages_added"]) == 1
    names = [c["name"] for c in client.get("/corpora").json()]
    assert "uploads" in names
    again = client.post(
        "/upload", files=[("files", ("a.png", png_bytes(), "image/png"))], data={"ocr": "false"}
    )
    assert again.json()["pages_added"] == []
    bad = client.post("/upload", files=[("files", ("a.exe", b"MZ", "application/octet-stream"))])
    assert bad.status_code == 400
