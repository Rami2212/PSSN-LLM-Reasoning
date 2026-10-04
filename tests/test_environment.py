"""Unit tests for environment decisions that do not download model weights."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.utils.environment import (
    collect_environment_info,
    load_qwen_model,
    select_inference_dtype,
)


def fake_torch(*, cuda_available: bool, bf16_supported: bool = False):
    cuda = Mock()
    cuda.is_available.return_value = cuda_available
    cuda.is_bf16_supported.return_value = bf16_supported
    cuda.device_count.return_value = 0
    return SimpleNamespace(
        __version__="test",
        float32="fp32",
        float16="fp16",
        bfloat16="bf16",
        cuda=cuda,
        version=SimpleNamespace(cuda=None),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: None)),
    )


@pytest.mark.parametrize(
    ("cuda_available", "bf16_supported", "expected"),
    [(True, True, ("bf16", "bfloat16")), (True, False, ("fp16", "float16")),
     (False, False, ("fp32", "float32"))],
)
def test_select_inference_dtype(cuda_available, bf16_supported, expected):
    assert select_inference_dtype(
        fake_torch(cuda_available=cuda_available, bf16_supported=bf16_supported)
    ) == expected


def test_collect_environment_info_without_cuda():
    info = collect_environment_info(fake_torch(cuda_available=False))
    assert info["cuda_available"] is False
    assert info["selected_dtype"] == "float32"
    assert info["gpu_count"] == 0
    assert info["gpus"] == []


def test_load_model_requires_cuda_before_loading_transformers_model():
    transformers = Mock()
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        load_qwen_model(
            torch_module=fake_torch(cuda_available=False),
            transformers_module=transformers,
        )
    transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()


def test_load_model_uses_bf16_without_quantization():
    torch = fake_torch(cuda_available=True, bf16_supported=True)
    tokenizer = object()
    model = Mock()
    transformers = Mock()
    transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    transformers.AutoModelForCausalLM.from_pretrained.return_value = model

    result = load_qwen_model(
        torch_module=torch,
        transformers_module=transformers,
    )

    assert result == (tokenizer, model, "bfloat16")
    transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
        "Qwen/Qwen3-4B",
        torch_dtype="bf16",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval.assert_called_once_with()

