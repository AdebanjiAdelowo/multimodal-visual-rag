"""Device selection, per-stage timing and memory accounting."""

from __future__ import annotations

import os
import platform
import resource
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any


def pick_device(prefer: str = "auto") -> str:
    """Resolve 'auto' to cuda > mps > cpu. Explicit names are returned unchanged."""
    if prefer != "auto":
        return prefer
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def peak_rss_mb() -> float:
    """Peak resident set size of this process (ru_maxrss is bytes on macOS, KiB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2) if sys.platform == "darwin" else rss / 1024


def accelerator_memory_mb(device: str) -> dict[str, float]:
    """Memory held by the accelerator allocator, where the backend exposes it."""
    try:
        import torch
    except ImportError:
        return {}
    if device.startswith("cuda") and torch.cuda.is_available():
        return {
            "cuda_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }
    if device == "mps" and torch.backends.mps.is_available():
        return {
            "mps_allocated_mb": torch.mps.current_allocated_memory() / 1024**2,
            "mps_driver_allocated_mb": torch.mps.driver_allocated_memory() / 1024**2,
        }
    return {}


def release_accelerator_memory() -> None:
    """Return cached accelerator memory to the system after dropping a model."""
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def hardware_summary() -> dict[str, Any]:
    import psutil

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["device"] = pick_device()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
    except ImportError:
        info["torch"] = None
    return info


class StageTimer:
    """Accumulates wall-clock time per named stage."""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.samples[name].append(time.perf_counter() - t0)

    def add(self, name: str, seconds: float) -> None:
        self.samples[name].append(seconds)

    def last(self) -> dict[str, float]:
        return {k: v[-1] for k, v in self.samples.items() if v}

    def summary(self) -> dict[str, dict[str, float]]:
        import numpy as np

        out = {}
        for k, v in self.samples.items():
            a = np.asarray(v)
            out[k] = {
                "n": int(a.size),
                "mean_ms": float(a.mean() * 1e3),
                "p50_ms": float(np.percentile(a, 50) * 1e3),
                "p95_ms": float(np.percentile(a, 95) * 1e3),
                "total_s": float(a.sum()),
            }
        return out


def model_size(model: Any) -> dict[str, float]:
    """Parameter count and in-memory weight footprint of a torch module."""
    n = sum(p.numel() for p in model.parameters())
    b = sum(p.numel() * p.element_size() for p in model.parameters())
    return {"params_m": round(n / 1e6, 1), "weights_mb": round(b / 1024**2, 1)}
