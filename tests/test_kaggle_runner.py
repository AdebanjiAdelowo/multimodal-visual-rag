"""The GPU runner cannot be executed in CI, so its command table is checked statically."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load_runner(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location("run_gpu", ROOT / "kaggle" / "run_gpu.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_commands_reference_existing_configs_and_cli_options(monkeypatch):
    runner = load_runner(monkeypatch)
    cli = (ROOT / "src" / "mmrag" / "cli.py").read_text()
    commands = set(re.findall(r'@app\.command\("([\w-]+)"\)', cli)) | {"hw", "ocr", "caption"}
    assert set(runner.COMMANDS) == set(runner.ALL_STAGES)
    for cmds in runner.COMMANDS.values():
        for cmd in cmds:
            assert cmd[0] == "mmrag" and cmd[1] in commands
            for arg in cmd[2:]:
                if arg.endswith(".yaml"):
                    assert (ROOT / arg).exists(), arg
                elif arg.startswith("--"):
                    assert f'"{arg}"' in cli or arg.lstrip("-").replace("-", "_") in cli, arg


def test_results_never_land_in_the_local_results_tree(monkeypatch):
    runner = load_runner(monkeypatch, MMRAG_TAG="t4", MMRAG_GENERATORS="qwen3vl_4b")
    assert runner.RESULTS == "results_t4" and runner.RUNS == "artifacts/runs_t4"
    for stage in ("retrieval", "captions", "generation", "ablations"):
        for cmd in runner.COMMANDS[stage]:
            if cmd[1] in ("eval-retrieval", "eval-generation"):
                assert cmd[cmd.index("--out") + 1].startswith("results_t4/")
                assert cmd[cmd.index("--runs") + 1].startswith("artifacts/runs_t4/")
    # The generator filter keeps one command per selected model.
    assert len(runner.COMMANDS["generation"]) == 1
    systems = [a for a in runner.COMMANDS["generation"][0] if a.startswith(runner.GEN)]
    assert len(systems) == 4 and all("qwen3vl_4b" in a for a in systems)


def test_dry_run_writes_verifiable_archive_and_respects_budget(monkeypatch, tmp_path):
    """Run the whole control flow with every external command replaced by a stub."""
    import hashlib
    import json
    import types
    import zipfile

    source = tmp_path / "input" / "bundle" / "multimodal-visual-rag"
    (source / "src" / "mmrag").mkdir(parents=True)
    (source / "pyproject.toml").write_text("")
    (source.parent / "COMMIT").write_text("abc1234\n")
    runner = load_runner(
        monkeypatch,
        MMRAG_SOURCE=str(tmp_path / "input"),
        MMRAG_WORK=str(tmp_path / "working"),
        MMRAG_SCRATCH=str(tmp_path / "temp"),
        MMRAG_STAGES="data,retrieval,generation",
        MMRAG_GENERATORS="smolvlm_500m,qwen3vl_2b",
    )
    (tmp_path / "working").mkdir()
    repo = tmp_path / "temp" / "multimodal-visual-rag"
    calls = []

    def fake_sh(cmd, cwd=None):
        calls.append(cmd)
        if cmd[:2] == ["mmrag", "eval-retrieval"]:
            (repo / "results_t4" / "retrieval").mkdir(parents=True, exist_ok=True)
            (repo / "results_t4" / "retrieval" / "x.json").write_text("{}")
        if cmd[:2] == ["mmrag", "eval-generation"]:
            if any("qwen3vl_2b" in a for a in cmd):
                raise runner.subprocess.CalledProcessError(1, cmd)
            runner.T_START -= 99 * 3600  # the first generator exhausts the time budget

    monkeypatch.setattr(runner, "sh", fake_sh)
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="pkg==1.0\n")
    )
    runner.main()

    assert not any(c[:2] == ["mmrag", "eval-generation"] and "qwen3vl_2b" in str(c) for c in calls)
    with zipfile.ZipFile(tmp_path / "working" / "mmrag_results.zip") as z:
        names = set(z.namelist())
        session = json.loads(z.read("session_t4.json"))
        sums = dict(reversed(line.split("  ")) for line in z.read("SHA256SUMS").decode().split("\n")
                    if line)  # fmt: skip
        for name, digest in sums.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    assert {"results_t4/retrieval/x.json", "session_t4.json", "SHA256SUMS"} <= names
    assert session["source_commit"] == "abc1234" and session["packages"] == ["pkg==1.0"]
    status = {s["stage"]: s for s in session["stages"]}
    assert status["retrieval"]["status"] == "completed"
    assert status["generation"]["status"] == "partial"
    assert len(status["generation"]["skipped_commands"]) == 1
    assert not (tmp_path / "working" / "multimodal-visual-rag").exists()  # no pages in the output
