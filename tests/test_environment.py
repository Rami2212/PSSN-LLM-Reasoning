"""Unit tests for environment decisions that do not download model weights."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.utils.environment import (
    collect_environment_info,
    generate_validation_response,
    load_qwen_model,
    select_inference_dtype,
)


def fake_torch(*, cuda_available: bool, bf16_supported: bool = False):
    cuda = Mock()
    cuda.is_available.return_value = cuda_available
    cuda.is_bf16_supported.return_value = bf16_supported
    cuda.get_device_capability.return_value = (8, 0) if bf16_supported else (7, 5)
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


def test_t4_uses_fp16_even_when_bf16_emulation_is_available():
    torch = fake_torch(cuda_available=True, bf16_supported=True)
    torch.cuda.get_device_capability.return_value = (7, 5)
    assert select_inference_dtype(torch) == ("fp16", "float16")


def test_rocm_preserves_fp16_without_using_nvidia_capability():
    torch = fake_torch(cuda_available=True, bf16_supported=True)
    torch.version.hip = "10.1.0"
    torch.cuda.get_device_capability.side_effect = RuntimeError("not applicable")
    assert select_inference_dtype(torch) == ("fp16", "float16")
    info = collect_environment_info(torch)
    assert info["backend"] == "rocm"
    assert info["hip_version"] == "10.1.0"


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
        dtype="bf16",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval.assert_called_once_with()


def test_validation_generation_supports_batch_encoding(monkeypatch):
    class FakeInputIds:
        shape = (1, 3)

        def size(self, dimension):
            return self.shape[dimension]

    class FakeBatchEncoding(dict):
        def to(self, device):
            self.device = device
            return self

    class FakeOutput:
        def __getitem__(self, key):
            assert key == (0, slice(3, None))
            return [21, 22]

    tokenizer = Mock(eos_token_id=99)
    tokenizer.apply_chat_template.return_value = "rendered prompt"
    tokenizer.return_value = FakeBatchEncoding(input_ids=FakeInputIds())
    tokenizer.decode.return_value = "  The answer is 42.  "
    model = Mock(device="cuda:0")
    model.generate.return_value = FakeOutput()
    torch = SimpleNamespace(
        inference_mode=lambda: nullcontext(),
        as_tensor=lambda value, **kwargs: value,
        long="long",
    )
    monkeypatch.setattr(
        "src.utils.environment._import_dependency",
        lambda name: torch,
    )

    response = generate_validation_response(tokenizer, model)

    assert response == "The answer is 42."
    tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "What is 17 + 25? Give a short explanation and the answer."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    tokenizer.assert_called_once_with(
        ["rendered prompt"],
        return_tensors="pt",
        add_special_tokens=False,
    )
    model.generate.assert_called_once()
    assert model.generate.call_args.kwargs["input_ids"].shape == (1, 3)

