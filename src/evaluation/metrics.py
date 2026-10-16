"""Reusable inference-efficiency measurement and model metadata helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class InferenceMeasurement:
    """Context manager measuring wall time and peak CUDA memory.

    CUDA is synchronized before and after inference so the reported wall time
    includes completed GPU work rather than only asynchronous kernel dispatch.
    Peak counters are reset per generation, making records directly comparable.
    """

    torch_module: Any
    clock: Callable[[], float] = time.perf_counter
    inference_latency_seconds: float | None = None
    peak_gpu_memory_allocated_bytes: int = 0
    peak_gpu_memory_reserved_bytes: int = 0
    _started_at: float | None = None
    _cuda_available: bool = False

    def __enter__(self) -> "InferenceMeasurement":
        cuda = self.torch_module.cuda
        self._cuda_available = bool(cuda.is_available())
        if self._cuda_available:
            cuda.synchronize()
            cuda.reset_peak_memory_stats()
        self._started_at = self.clock()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._started_at is None:
            raise RuntimeError("Inference measurement was not started.")
        cuda = self.torch_module.cuda
        if self._cuda_available:
            cuda.synchronize()
        self.inference_latency_seconds = self.clock() - self._started_at
        if self._cuda_available:
            self.peak_gpu_memory_allocated_bytes = int(cuda.max_memory_allocated())
            self.peak_gpu_memory_reserved_bytes = int(cuda.max_memory_reserved())
        return False

    def to_dict(self) -> dict[str, int | float]:
        if self.inference_latency_seconds is None:
            raise RuntimeError("Inference measurement has not completed.")
        return {
            "inference_latency_seconds": self.inference_latency_seconds,
            "peak_gpu_memory_allocated_bytes": self.peak_gpu_memory_allocated_bytes,
            "peak_gpu_memory_reserved_bytes": self.peak_gpu_memory_reserved_bytes,
        }


def collect_model_metadata(
    model: Any,
    torch_module: Any,
    *,
    model_id: str,
    precision: str | None,
) -> dict[str, Any]:
    """Collect JSON-safe model/device metadata for result reproduction."""
    cuda_available = bool(torch_module.cuda.is_available())
    device = str(getattr(model, "device", "unknown"))
    gpu_name: str | None = None
    if cuda_available:
        try:
            gpu_name = str(torch_module.cuda.get_device_name(model.device))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            gpu_name = str(torch_module.cuda.get_device_name())

    parameter_count: int | None = None
    try:
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
    except (AttributeError, TypeError):
        pass

    return {
        "model_id": model_id,
        "model_class": type(model).__name__,
        "parameter_count": parameter_count,
        "precision": precision,
        "device": device,
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch_module.version, "cuda", None),
        "hip_version": getattr(torch_module.version, "hip", None),
        "model_revision": (
            getattr(getattr(model, "config", None), "_commit_hash", None)
            if isinstance(getattr(getattr(model, "config", None), "_commit_hash", None), str)
            else None
        ),
        "gpu_name": gpu_name,
    }

