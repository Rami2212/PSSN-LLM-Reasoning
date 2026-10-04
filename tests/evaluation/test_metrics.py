"""Tests for inference timing, CUDA memory, and model metadata."""

from types import SimpleNamespace
from unittest.mock import Mock

from src.evaluation.metrics import InferenceMeasurement, collect_model_metadata


class Clock:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def fake_torch(cuda_available):
    cuda = Mock()
    cuda.is_available.return_value = cuda_available
    cuda.max_memory_allocated.return_value = 1024
    cuda.max_memory_reserved.return_value = 2048
    cuda.get_device_name.return_value = "Test GPU"
    return SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(cuda="12.test" if cuda_available else None),
    )


def test_cuda_measurement_synchronizes_and_collects_peak_memory():
    torch = fake_torch(True)
    with InferenceMeasurement(torch, clock=Clock(10.0, 10.25)) as measurement:
        pass
    assert measurement.to_dict() == {
        "inference_latency_seconds": 0.25,
        "peak_gpu_memory_allocated_bytes": 1024,
        "peak_gpu_memory_reserved_bytes": 2048,
    }
    assert torch.cuda.synchronize.call_count == 2
    torch.cuda.reset_peak_memory_stats.assert_called_once_with()


def test_cpu_measurement_reports_zero_gpu_memory():
    torch = fake_torch(False)
    with InferenceMeasurement(torch, clock=Clock(3.0, 3.5)) as measurement:
        pass
    assert measurement.to_dict()["inference_latency_seconds"] == 0.5
    assert measurement.to_dict()["peak_gpu_memory_allocated_bytes"] == 0
    torch.cuda.synchronize.assert_not_called()


def test_collect_model_metadata_is_json_safe():
    torch = fake_torch(True)
    model = SimpleNamespace(
        device="cuda:0",
        parameters=lambda: [SimpleNamespace(numel=lambda: 3), SimpleNamespace(numel=lambda: 4)],
    )
    metadata = collect_model_metadata(
        model,
        torch,
        model_id="Qwen/Qwen3-4B",
        precision="bfloat16",
    )
    assert metadata["model_id"] == "Qwen/Qwen3-4B"
    assert metadata["parameter_count"] == 7
    assert metadata["gpu_name"] == "Test GPU"
    assert metadata["cuda_version"] == "12.test"

