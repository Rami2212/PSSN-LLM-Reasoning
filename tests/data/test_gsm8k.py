"""Offline tests for GSM8K normalization and deterministic sampling."""

import pytest

from src.data.gsm8k import (
    create_development_subset,
    extract_reference_answer,
    load_gsm8k,
    normalize_example,
)


def raw_example(number: int) -> dict[str, str]:
    return {
        "question": f"What is {number} + 1?",
        "answer": f"Add one to {number}.\n#### {number + 1}",
    }


@pytest.mark.parametrize(
    ("solution", "expected"),
    [
        ("Work shown here.\n#### 42", "42"),
        ("Calculation\n#### -12.5", "-12.5"),
        ("Calculation\n#### 1,234", "1,234"),
    ],
)
def test_extract_reference_answer(solution, expected):
    assert extract_reference_answer(solution) == expected


@pytest.mark.parametrize("solution", ["", "No final marker", "Work\n####   "])
def test_extract_reference_answer_rejects_malformed_solution(solution):
    with pytest.raises(ValueError):
        extract_reference_answer(solution)


def test_normalize_example_uses_consistent_schema_and_id():
    normalized = normalize_example(raw_example(4), split="train", index=7)
    assert normalized == {
        "id": "gsm8k-train-00007",
        "question": "What is 4 + 1?",
        "reference_solution": "Add one to 4.\n#### 5",
        "reference_answer": "5",
    }


def test_development_sampling_is_deterministic_and_does_not_mutate_train():
    train = [{"id": index} for index in range(20)]
    original = list(train)

    first = create_development_subset(train, size=5, seed=123)
    second = create_development_subset(train, size=5, seed=123)
    different = create_development_subset(train, size=5, seed=124)

    assert first == second
    assert first != different
    assert train == original
    assert [row["id"] for row in first] == sorted(row["id"] for row in first)


def test_load_gsm8k_preserves_splits_and_samples_train_only():
    source = {
        "train": [raw_example(index) for index in range(10)],
        "test": [raw_example(100), raw_example(200)],
    }
    calls = []

    def loader(dataset_id, config):
        calls.append((dataset_id, config))
        return source

    prepared = load_gsm8k(
        development_size=3,
        seed=9,
        dataset_loader=loader,
    )

    assert calls == [("openai/gsm8k", "main")]
    assert len(prepared["train"]) == 10
    assert len(prepared["test"]) == 2
    assert len(prepared["development"]) == 3
    assert all(row["id"].startswith("gsm8k-train-") for row in prepared["development"])
    assert all(row["id"].startswith("gsm8k-test-") for row in prepared["test"])
    assert set(prepared["train"][0]) == {
        "id", "question", "reference_solution", "reference_answer"
    }


def test_load_gsm8k_requires_official_splits():
    with pytest.raises(ValueError, match="test"):
        load_gsm8k(dataset_loader=lambda *_: {"train": [raw_example(1)]})


@pytest.mark.parametrize("size", [0, -1, 4])
def test_development_subset_rejects_invalid_size(size):
    with pytest.raises(ValueError):
        create_development_subset([1, 2, 3], size=size)

