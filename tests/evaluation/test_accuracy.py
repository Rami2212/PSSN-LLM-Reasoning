"""Tests for per-record scoring and aggregate accuracy."""

import pytest

from src.evaluation.accuracy import (
    calculate_accuracy,
    evaluate_prediction,
    evaluate_records,
    summarize_accuracy,
)


def test_evaluate_prediction_records_extracted_and_normalized_answers():
    result = evaluate_prediction(
        problem_id="gsm8k-test-00001",
        reference_answer="1,200",
        generated_response="Reasoning...\nFinal answer: $1200.00",
    )
    assert result == {
        "problem_id": "gsm8k-test-00001",
        "reference_answer": "1,200",
        "predicted_answer": "$1200.00",
        "normalized_reference_answer": "1200",
        "normalized_predicted_answer": "1200",
        "is_correct": True,
    }


def test_evaluate_records_uses_embedded_references_and_preserves_order():
    records = [
        {
            "problem_id": "first",
            "reasoning_trace": "Final answer: 8",
            "reference_answer": "8.0",
        },
        {
            "problem_id": "second",
            "reasoning_trace": "Final answer: 12",
            "reference_answer": "10",
        },
    ]
    evaluations = evaluate_records(records)
    assert [item["problem_id"] for item in evaluations] == ["first", "second"]
    assert [item["is_correct"] for item in evaluations] == [True, False]


def test_evaluate_records_accepts_separate_reference_mapping():
    records = [{"problem_id": "p1", "reasoning_trace": "#### 3"}]
    assert evaluate_records(records, references={"p1": "3"})[0]["is_correct"]


def test_accuracy_and_summary():
    evaluations = [
        {"is_correct": True},
        {"is_correct": False},
        {"is_correct": True},
        {"is_correct": True},
    ]
    assert calculate_accuracy(evaluations) == 0.75
    assert summarize_accuracy(evaluations) == {
        "number_of_problems": 4,
        "correct_answers": 3,
        "incorrect_answers": 1,
        "accuracy": 0.75,
    }


def test_malformed_generation_is_scored_incorrectly():
    result = evaluate_prediction(
        problem_id="p1",
        reference_answer="7",
        generated_response="I could not solve this problem.",
    )
    assert result["predicted_answer"] is None
    assert result["is_correct"] is False


def test_empty_accuracy_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        calculate_accuracy([])


def test_missing_reference_is_rejected():
    with pytest.raises(ValueError, match="No reference"):
        evaluate_records([{"problem_id": "p1", "reasoning_trace": "Answer: 2"}])

