"""Offline tests for the Qwen generation wrapper."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.models.qwen import GenerationSettings, QwenGenerator


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (1, len(self.values))
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self):
        self.messages = None
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return "rendered prompt"

    def __call__(self, prompts, **kwargs):
        assert prompts == ["rendered prompt"]
        return {"input_ids": FakeTensor([10, 11, 12])}

    def decode(self, ids, **kwargs):
        assert ids == [21, 22]
        return " reasoning and final answer "


class FakeTorch:
    def __init__(self):
        self.manual_seed = Mock()
        self.cuda = SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=Mock(),
            synchronize=Mock(),
            reset_peak_memory_stats=Mock(),
            max_memory_allocated=Mock(return_value=4096),
            max_memory_reserved=Mock(return_value=8192),
            get_device_name=Mock(return_value="Test GPU"),
        )
        self.version = SimpleNamespace(cuda="12.test")

    @staticmethod
    def inference_mode():
        return nullcontext()


def test_qwen_generator_preserves_full_output_and_token_counts():
    tokenizer = FakeTokenizer()
    model = Mock(device="cuda:0")
    model.generate.return_value = [[10, 11, 12, 21, 22]]
    torch = FakeTorch()
    settings = GenerationSettings(max_new_tokens=50, seed=7)
    generator = QwenGenerator(
        tokenizer,
        model,
        settings=settings,
        precision="bfloat16",
        torch_module=torch,
    )

    result = generator.generate("Solve this")

    assert result.text == "reasoning and final answer"
    assert result.input_tokens == 3
    assert result.output_tokens == 2
    assert result.metrics["inference_latency_seconds"] >= 0
    assert result.metrics["peak_gpu_memory_allocated_bytes"] == 4096
    assert result.metrics["peak_gpu_memory_reserved_bytes"] == 8192
    assert result.model_metadata["gpu_name"] == "Test GPU"
    assert result.generation_config["max_new_tokens"] == 50
    assert result.generation_config["seed"] == 7
    assert result.generation_config["thinking_enabled"] is True
    assert tokenizer.messages == [{"role": "user", "content": "Solve this"}]
    assert tokenizer.template_kwargs["enable_thinking"] is True
    torch.manual_seed.assert_called_once_with(7)
    torch.cuda.manual_seed_all.assert_called_once_with(7)
    model.generate.assert_called_once()
    call_kwargs = model.generate.call_args.kwargs
    assert call_kwargs["do_sample"] is False
    assert "temperature" not in call_kwargs
    assert call_kwargs["pad_token_id"] == 99


def test_sampling_parameters_are_only_sent_when_sampling():
    deterministic = GenerationSettings(do_sample=False).model_kwargs()
    sampled = GenerationSettings(
        do_sample=True, temperature=0.7, top_p=0.8
    ).model_kwargs()
    assert "temperature" not in deterministic
    assert sampled["temperature"] == 0.7
    assert sampled["top_p"] == 0.8


def test_continuation_extends_assistant_turn_and_does_not_double_think_marker():
    class ContinuationTokenizer(FakeTokenizer):
        def __call__(self, prompts, **kwargs):
            assert prompts == ["user prompt <think>\nReasoning prefix\n"]
            return {"input_ids": FakeTensor([10, 11, 12, 13, 14, 15])}
    tokenizer = ContinuationTokenizer()
    tokenizer.apply_chat_template = Mock(return_value="user prompt <think>\n")
    model = Mock(device="cuda:0")
    model.generate.return_value = [[10, 11, 12, 13, 14, 15, 21, 22]]
    generator = QwenGenerator(tokenizer, model, torch_module=FakeTorch())
    result = generator.generate_continuation("Solve this", "<think>\nReasoning prefix")
    assert result.input_tokens == 6
    assert result.output_tokens == 2


def test_continuation_restores_thinking_tag_omitted_by_state_segmentation():
    class ContinuationTokenizer(FakeTokenizer):
        def __call__(self, prompts, **kwargs):
            assert prompts == ["assistant turn\n<think>\nRetained state\n"]
            return {"input_ids": FakeTensor([10, 11, 12])}
    tokenizer = ContinuationTokenizer()
    tokenizer.apply_chat_template = Mock(return_value="assistant turn\n")
    model = Mock(device="cuda:0")
    model.generate.return_value = [[10, 11, 12, 21, 22]]
    generator = QwenGenerator(tokenizer, model, torch_module=FakeTorch())
    assert generator.generate_continuation("Solve this", "Retained state").output_tokens == 2


@pytest.mark.parametrize("last_token,expected", [(22, "length"), (99, "eos")])
def test_generation_records_cap_stop_and_eos_at_cap(last_token, expected):
    tokenizer = FakeTokenizer()
    tokenizer.decode = Mock(return_value="Final answer: 42")
    model = Mock(device="cuda:0")
    model.generate.return_value = [[10, 11, 12, 21, last_token]]
    generator = QwenGenerator(
        tokenizer, model, settings=GenerationSettings(max_new_tokens=2),
        torch_module=FakeTorch(),
    )
    assert generator.generate("Solve this").finish_reason == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0},
        {"temperature": 0},
        {"top_p": 0},
        {"top_p": 1.1},
        {"repetition_penalty": 0},
    ],
)
def test_generation_settings_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        GenerationSettings(**kwargs)

