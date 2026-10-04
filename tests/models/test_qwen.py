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
        )

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

