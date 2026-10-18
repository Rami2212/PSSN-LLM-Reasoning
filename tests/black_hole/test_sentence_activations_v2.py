import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

import pytest

from src.black_hole.sentence_activations_v2 import (
    extract_row_representations,
    load_and_split,
    run_extraction,
    split_rows_by_question,
)


class FakeTorch:
    long = np.int64

    @staticmethod
    def inference_mode():
        return nullcontext()


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        return f"<question>{messages[0]['content']}</question><think>"

    def __call__(self, text, *, return_tensors, return_offsets_mapping, add_special_tokens):
        offsets = np.asarray([[(i, i + 1) for i in range(len(text))]], dtype=np.int64)
        return {
            "input_ids": np.arange(len(text), dtype=np.int64)[None, :],
            "attention_mask": np.ones((1, len(text)), dtype=np.int64),
            "offset_mapping": offsets,
        }


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, *, output_hidden_states, use_cache, return_dict):
        self.calls += 1
        positions = np.arange(input_ids.shape[1], dtype=np.float32)
        first = np.stack([positions, positions * 2], axis=-1)[None, :, :]
        second = np.stack([positions * 3, positions * 4], axis=-1)[None, :, :]
        embedding = np.zeros_like(first)
        return SimpleNamespace(hidden_states=(embedding, first, second))


def _row(index, problem, question, label_id):
    return {
        "dataset_row_id": f"row-{index}",
        "problem_id": problem,
        "trace_id": f"trace-{problem}",
        "question": question,
        "reasoning_prefix": "Earlier step. ",
        "sentence_text": "Target sentence.",
        "label": "low_value_in_context" if label_id else "answer_sensitive_in_context",
        "label_id": label_id,
        "sentence_index": index,
    }


def test_v2_extracts_final_and_all_layer_features_in_one_forward_pass():
    model = FakeModel()
    final, all_layers, layer_count = extract_row_representations(
        model, FakeTokenizer(), _row("r1", "q1", "Question?", 1),
        torch_module=FakeTorch,
    )
    assert model.calls == 1
    assert layer_count == 2
    assert final.shape == (2,)
    assert all_layers.shape == (4,)
    np.testing.assert_array_equal(all_layers[-2:], final)


def test_v2_saves_two_trainer_compatible_feature_sets(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    rows = []
    for question_index in range(12):
        for label in (0, 1):
            rows.append(_row(
                f"{question_index}-{label}", f"q-{question_index}",
                f"Question {question_index}?", label,
            ))
    (dataset_dir / "sentence_training_data.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output_dir = tmp_path / "activations"
    result = run_extraction(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        model_id="fake-model",
        model_revision="fake-revision",
        precision="float32",
        seed=1,
        torch_module=FakeTorch,
    )
    with np.load(output_dir / "train.npz", allow_pickle=False) as final:
        with np.load(output_dir / "all_layers" / "train.npz", allow_pickle=False) as all_layers:
            assert final["features"].shape == (16, 2)
            assert all_layers["features"].shape == (16, 4)
            np.testing.assert_array_equal(final["labels"], all_layers["labels"])
            np.testing.assert_array_equal(final["row_ids"], all_layers["row_ids"])
    assert result["feature_representations"]["all_layers"]


def test_v2_splits_all_rows_from_question_together_and_is_deterministic():
    rows = []
    for question_index in range(30):
        for label in (0, 1):
            rows.append(_row(
                f"{question_index}-{label}", f"q-{question_index}",
                f"Question {question_index}?", label,
            ))
    first = split_rows_by_question(rows, seed=7)
    second = split_rows_by_question(rows, seed=7)
    assert first == second
    memberships = {}
    for split, split_rows in first.items():
        assert {row["split"] for row in split_rows} == {split}
        for row in split_rows:
            previous = memberships.setdefault(row["problem_id"], split)
            assert previous == split
    assert set(memberships.values()) == {"train", "validation", "test"}


def test_v2_rejects_same_normalized_question_with_conflicting_problem_ids():
    rows = [
        _row("1", "q1", "Same question?", 0),
        _row("2", "q2", "  Same   question?  ", 1),
        _row("3", "q3", "Other question?", 0),
    ]
    with pytest.raises(ValueError, match="conflicting problem_ids"):
        split_rows_by_question(rows)


def test_v2_loads_consolidated_training_file(tmp_path):
    dataset = tmp_path / "sentence_training_data.jsonl"
    rows = []
    for question_index in range(12):
        for label in (0, 1):
            rows.append(_row(
                f"{question_index}-{label}", f"q-{question_index}",
                f"Question {question_index}?", label,
            ))
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    splits = load_and_split(tmp_path, seed=1)
    assert sum(map(len, splits.values())) == len(rows)
