import pytest
from PIL import Image

from mmrag.data.corpus import Corpus
from mmrag.index import store
from mmrag.ingest.ocr import normalize_ocr
from mmrag.ingest.parse import ingest_files


def test_ingest_is_idempotent_and_content_addressed(tmp_path):
    a, b = tmp_path / "scan.png", tmp_path / "other" / "scan.png"
    b.parent.mkdir()
    Image.new("RGB", (3000, 1000), "white").save(a)
    Image.new("RGB", (100, 100), "black").save(b)
    root = tmp_path / "corpus"
    assert len(ingest_files([a], root, max_side=800)) == 1
    assert ingest_files([a], root) == []  # same bytes: nothing added
    assert len(ingest_files([b], root)) == 1  # same name, different bytes: new document
    c = Corpus(root)
    assert len(c) == 2 and max(c.image(c.page_ids[0]).size) == 800


def test_pdf_is_split_into_pages(tmp_path):
    pdf = tmp_path / "two.pdf"
    pages = [Image.new("RGB", (200, 260), "white") for _ in range(2)]
    pages[0].save(pdf, save_all=True, append_images=pages[1:])
    added = ingest_files([pdf], tmp_path / "corpus")
    assert [p.page_number for p in added] == [0, 1]


def test_unsupported_file_type(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("x")
    with pytest.raises(ValueError):
        ingest_files([f], tmp_path / "corpus")


def test_text_sources_must_cover_every_page(corpus):
    corpus.save_text("partial", {"doc-p0": "only one"})
    with pytest.raises(ValueError):
        corpus.load_text("partial")
    with pytest.raises(FileNotFoundError):
        corpus.load_text("missing")


def test_normalize_ocr():
    assert normalize_ocr("re-\ntrieval   works\n\n\n\nnow\x0c") == "retrieval works\n\nnow"


def test_index_dir_changes_with_parameters(tmp_path, corpus):
    base = {"model": "m", "corpus": corpus.fingerprint()}
    d1 = store.index_dir(tmp_path, "c", "visual", base)
    assert d1 == store.index_dir(tmp_path, "c", "visual", dict(base))
    assert d1 != store.index_dir(tmp_path, "c", "visual", {**base, "model": "other"})
    assert not store.is_built(d1)
    store.write_meta(d1, base)
    assert store.is_built(d1) and store.read_meta(d1)["params"] == base
