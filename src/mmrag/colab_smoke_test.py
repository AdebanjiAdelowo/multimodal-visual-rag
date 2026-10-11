"""CUDA-only smoke test using the production VLM backend; never runs on MPS."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import traceback
import uuid
from pathlib import Path

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)
    if json.loads(path.read_text()) != report:
        raise RuntimeError("Report read-back mismatch")


def run(output: Path, source_commit: str, runtime_type: str) -> int:
    output = output / (time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "source_commit": source_commit,
        "runtime_type": runtime_type,
        "python": platform.python_version(),
        "model": MODEL,
        "model_revision": REVISION,
        "dtype": "float16",
        "quantization": None,
        "status": "started",
        "model_loading": "not_started",
        "inference": "not_started",
        "warnings": ["Synthetic examples test execution, not benchmark correctness."],
        "outputs": [],
        "errors": [],
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "accelerate", "huggingface-hub", "pillow")
        },
    }
    path = output / "report.json"
    write_report(path, report)
    try:
        import torch
        from PIL import Image, ImageDraw

        from mmrag.config import VLMCfg
        from mmrag.generate.vlm import image_part, load_vlm, text_part

        report.update(cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; CPU/MPS fallback is prohibited")
        free, total = torch.cuda.mem_get_info()
        report.update(
            gpu_name=torch.cuda.get_device_name(0), vram_total_bytes=total, vram_free_bytes=free
        )
        write_report(path, report)
        if free < 8 * 1024**3:
            raise RuntimeError("At least 8 GiB free VRAM required for this conservative 2B test")
        torch.cuda.reset_peak_memory_stats()
        report["model_loading"] = "running"
        write_report(path, report)
        start = time.perf_counter()
        vlm = load_vlm(
            VLMCfg(
                model=MODEL, revision=REVISION, device="cuda", dtype="float16", max_new_tokens=48
            )
        )
        torch.cuda.synchronize()
        report.update(model_loading="passed", load_latency_s=time.perf_counter() - start)
        write_report(path, report)
        report["inference"] = "running"
        for i, color in enumerate(("red", "blue")):
            image = Image.new("RGB", (512, 512), "white")
            ImageDraw.Draw(image).rectangle((120, 120, 390, 390), fill=color)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = vlm.generate(
                [image_part(image), text_part("What color is the square? Answer briefly.")],
                image_max_side=512,
            )
            torch.cuda.synchronize()
            latency = time.perf_counter() - start
            if not result.text.strip():
                raise RuntimeError("Empty inference output")
            artifact = output / f"example_{i}.json"
            payload = {
                "expected_color": color,
                "answer": result.text,
                "latency_s": latency,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            artifact.write_text(json.dumps(payload, indent=2))
            assert json.loads(artifact.read_text()) == payload
            report["outputs"].append(
                {"path": str(artifact), "sha256": checksum(artifact), **payload}
            )
            write_report(path, report)
        report.update(
            inference="passed",
            status="passed",
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    except Exception as exc:
        report["status"] = "failed"
        key = "inference" if report["model_loading"] == "passed" else "model_loading"
        report[key] = "failed"
        report["errors"].append(
            {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        )
    write_report(path, report)
    (output / "report.sha256").write_text(checksum(path) + "  report.json\n")
    print(path, report["status"], flush=True)
    return 0 if report["status"] == "passed" else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-type", default="Google Colab GPU (user selected)")
    args = parser.parse_args()
    if len(args.source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in args.source_commit
    ):
        parser.error("source-commit must be a full lowercase Git SHA")
    raise SystemExit(run(args.output, args.source_commit, args.runtime_type))


if __name__ == "__main__":
    main()
