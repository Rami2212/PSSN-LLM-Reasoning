from contextlib import nullcontext
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.black_hole.sentence_activations import (
    build_activation_context,
    extract_row_activation,
    load_question_disjoint_splits,
)


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


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is True
        return f"<question>{messages[0]['content']}</question><think>"

    def __call__(self, text, *, return_tensors, return_offsets_mapping, add_special_tokens):
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

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, *, output_hidden_states, use_cache, return_dict):
        assert output_hidden_states is True
        assert use_cache is False
        assert return_dict is True
        positions = np.arange(input_ids.shape[1], dtype=np.float32)
        hidden = np.stack([positions, positions * 2], axis=-1)[None, :, :]
        return SimpleNamespace(hidden_states=(np.zeros_like(hidden), hidden))


def row(row_id="r1", problem_id="q1", split="train"):
    return {
        "dataset_row_id": row_id,
        "problem_id": problem_id,
        "split": split,
        "question": "What is 2 + 2?",
        "reasoning_prefix": "First add 2 and 2. ",
        "sentence_text": "The result is 4.",
        "reference_answer": "4",
        "label": "low_value_in_context",
        "label_id": 1,
    }


def test_activation_context_is_question_prefix_and_target_only():
    context, span = build_activation_context(FakeTokenizer(), row())
    assert context.endswith("First add 2 and 2. The result is 4.")
    assert context[span[0]:span[1]] == "The result is 4."
    assert "reference_answer" not in context


def test_extracts_mean_pooled_target_span():
    vector = extract_row_activation(
        FakeModel(), FakeTokenizer(), row(), layer=-1, torch_module=FakeTorch
    )
    assert vector.shape == (2,)
    assert np.isfinite(vector).all()


def test_split_loader_rejects_question_leakage(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for split in ("train", "validation", "test"):
        problem_id = "same-question" if split != "test" else "other-question"
        (split_dir / f"{split}.jsonl").write_text(
            json.dumps(row(f"row-{split}", problem_id, split)) + "\n",
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="Question leakage"):
        load_question_disjoint_splits(tmp_path)


def test_split_loader_rejects_mismatched_label_id(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for split in ("train", "validation", "test"):
        item = row(f"row-{split}", f"q-{split}", split)
        if split == "train":
            item["label_id"] = 0
        (split_dir / f"{split}.jsonl").write_text(
            json.dumps(item) + "\n", encoding="utf-8"
        )
    with pytest.raises(ValueError, match="label_id"):
        load_question_disjoint_splits(tmp_path)
