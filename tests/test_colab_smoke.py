import ast
import hashlib
import json
from pathlib import Path

from mmrag.colab_smoke_test import MODEL, REVISION, checksum, write_report
from mmrag.config import VLMCfg


def test_report_roundtrip(tmp_path):
    path = tmp_path / "report.json"
    report = {"status": "failed", "errors": ["CUDA unavailable"]}
    write_report(path, report)
    assert json.loads(path.read_text()) == report
    assert checksum(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_model_revision_config():
    cfg = VLMCfg(model=MODEL, revision=REVISION, device="cuda", dtype="float16")
    assert cfg.revision == REVISION and not cfg.load_in_4bit


def test_notebook_structure_and_syntax():
    root = Path(__file__).parents[1]
    notebook = json.loads((root / "notebooks/colab_gpu_smoke_test.ipynb").read_text())
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))
