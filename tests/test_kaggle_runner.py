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
