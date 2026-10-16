from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from src.models.hidden_states import extract_span_hidden_vectors, token_positions_for_span
from src.semantic_states import SemanticState, extract_semantic_state_representations


class FakeTorch:
    long = np.int64

    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def tensor(values, dtype=None, device=None):
        return np.asarray(values, dtype=dtype)

    @staticmethod
    def stack(values, dim=0):
        return np.stack(values, axis=dim)


class CharOffsetTokenizer:
    name_or_path = "fake/qwen-tokenizer"
    init_kwargs = {"_commit_hash": "tokenizer-sha"}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is False
        assert add_generation_prompt is False
        assert enable_thinking is True
        return (
            "<user>" + messages[0]["content"] + "</user>"
            "<assistant>" + messages[1]["content"] + "</assistant>"
        )

    def __call__(
        self,
        text,
        *,
        return_tensors,
        return_offsets_mapping,
        add_special_tokens,
    ):
        assert return_tensors == "pt"
        assert return_offsets_mapping is True
        assert add_special_tokens is False
        offsets = np.asarray([[(i, i + 1) for i in range(len(text))]], dtype=np.int64)
        return {
            "input_ids": np.arange(len(text), dtype=np.int64)[None, :],
            "attention_mask": np.ones((1, len(text)), dtype=np.int64),
            "offset_mapping": offsets,
        }


class FakeModel:
    device = "cpu"
    config = SimpleNamespace(
        _commit_hash="model-sha",
        model_type="qwen3",
        num_hidden_layers=1,
    )

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, *, output_hidden_states, use_cache, return_dict):
        assert output_hidden_states is True
        assert use_cache is False
        assert return_dict is True
        sequence_length = input_ids.shape[1]
        positions = np.arange(sequence_length, dtype=np.float32)
        final_layer = np.stack([positions, positions * 2], axis=-1)[None, :, :]
        return SimpleNamespace(
            hidden_states=(np.zeros_like(final_layer), final_layer),
        )


def make_state(trace, text, index, *, is_final=False):
    start = trace.index(text)
    return SemanticState(
        problem_id="gsm8k-train-00053",
        trace_id="run-123:gsm8k-train-00053",
        state_id=f"S{index + 1}",
        state_index=index,
        text=text,
        start_char=start,
        end_char=start + len(text),
        token_count=len(text.split()),
        is_final_state=is_final,
    )


def test_token_positions_select_only_overlapping_offsets():
    offsets = [(0, 0), (0, 2), (2, 4), (4, 6), (6, 0)]

    assert token_positions_for_span(offsets, (2, 5)) == [2, 3]


def test_hidden_span_extraction_mean_pools_deterministically():
    text = "prefix STATE suffix"
    tokenizer = CharOffsetTokenizer()
    model = FakeModel()
    span = (text.index("STATE"), text.index("STATE") + len("STATE"))

    first, token_positions, meta = extract_span_hidden_vectors(
        model,
        tokenizer,
        text,
        [span],
        layer=-1,
        torch_module=FakeTorch,
    )
    second, _, _ = extract_span_hidden_vectors(
        model,
        tokenizer,
        text,
        [span],
        layer=-1,
        torch_module=FakeTorch,
    )

    assert first.shape == (1, 2)
    assert token_positions == [list(range(span[0], span[1]))]
    assert np.array_equal(first, second)
    assert meta == {"resolved_layer_index": 1, "hidden_size": 2, "pooling": "mean"}


def test_trace_extraction_returns_fixed_vectors_and_link_metadata():
    trace = "Compute 2 + 2. Final answer: 4"
    states = [
        make_state(trace, "Compute 2 + 2.", 0),
        make_state(trace, "Final answer: 4", 1, is_final=True),
    ]

    result = extract_semantic_state_representations(
        FakeModel(),
        CharOffsetTokenizer(),
        prompt="What is 2 + 2?",
        reasoning_trace=trace,
        states=states,
        torch_module=FakeTorch,
    )

    assert result.vectors.shape == (2, 2)
    assert result.vectors.dtype == np.float32
    assert [row["state_id"] for row in result.index_records] == ["S1", "S2"]
    assert [row["vector_index"] for row in result.index_records] == [0, 1]
    assert all(row["resolved_layer_index"] == 1 for row in result.index_records)
    assert result.configuration["pooling"] == "mean"
    assert result.configuration["model_revision"] == "model-sha"
    assert result.configuration["tokenizer_revision"] == "tokenizer-sha"


def test_trace_extraction_rejects_wrong_state_order():
    trace = "Compute 2 + 2. Final answer: 4"
    states = [
        make_state(trace, "Final answer: 4", 1, is_final=True),
        make_state(trace, "Compute 2 + 2.", 0),
    ]

    with pytest.raises(ValueError, match="ordered"):
        extract_semantic_state_representations(
            FakeModel(),
            CharOffsetTokenizer(),
            prompt="What is 2 + 2?",
            reasoning_trace=trace,
            states=states,
            torch_module=FakeTorch,
        )


def test_hidden_extraction_rejects_missing_offsets_and_invalid_layer():
    class NoOffsetsTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": np.ones((1, 3), dtype=np.int64)}

    with pytest.raises(ValueError, match="offset_mapping"):
        extract_span_hidden_vectors(
            FakeModel(), NoOffsetsTokenizer(), "abc", [(0, 1)], torch_module=FakeTorch
        )

    with pytest.raises(IndexError, match="outside"):
        extract_span_hidden_vectors(
            FakeModel(), CharOffsetTokenizer(), "abc", [(0, 1)], layer=4, torch_module=FakeTorch
        )
