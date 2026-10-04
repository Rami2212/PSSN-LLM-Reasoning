"""Per-example and aggregate GSM8K answer-accuracy evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .answer_extraction import (
    answers_equal,
    extract_final_answer,
    normalize_numeric_answer,
)


def evaluate_prediction(
    *,
    problem_id: str,
    reference_answer: Any,
    generated_response: str,
) -> dict[str, Any]:
    """Evaluate one generated response against one reference answer."""
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError("problem_id must be a non-empty string.")

    predicted_answer = extract_final_answer(generated_response)
    return {
        "problem_id": problem_id,
        "reference_answer": str(reference_answer),
        "predicted_answer": predicted_answer,
        "normalized_reference_answer": normalize_numeric_answer(reference_answer),
        "normalized_predicted_answer": normalize_numeric_answer(predicted_answer),
        "is_correct": answers_equal(predicted_answer, reference_answer),
    }


def evaluate_record(
    record: Mapping[str, Any],
    *,
    reference_answer: Any | None = None,
) -> dict[str, Any]:
    """Evaluate a stored CoT record without invoking or depending on a model."""
    problem_id = record.get("problem_id")
    generated_response = record.get("reasoning_trace")
    if reference_answer is None:
        reference_answer = record.get("reference_answer")
    if reference_answer is None:
        raise ValueError(f"No reference answer supplied for problem {problem_id!r}.")
    if not isinstance(generated_response, str):
        raise ValueError(
            f"Record {problem_id!r} must contain a string 'reasoning_trace'."
        )
    return evaluate_prediction(
        problem_id=problem_id,
        reference_answer=reference_answer,
        generated_response=generated_response,
    )


def evaluate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    references: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate records in order using embedded or separately supplied targets."""
    evaluations: list[dict[str, Any]] = []
    for record in records:
        problem_id = record.get("problem_id")
        reference = None
        if references is not None:
            if problem_id not in references:
                raise ValueError(f"No reference answer found for problem {problem_id!r}.")
            reference = references[problem_id]
        evaluations.append(evaluate_record(record, reference_answer=reference))
    return evaluations


def calculate_accuracy(evaluations: Iterable[Mapping[str, Any]]) -> float:
    """Calculate exact-match accuracy in the interval [0, 1]."""
    items = list(evaluations)
    if not items:
        raise ValueError("Cannot calculate accuracy for an empty evaluation set.")
    if any("is_correct" not in item for item in items):
        raise ValueError("Every evaluation must contain 'is_correct'.")
    return sum(bool(item["is_correct"]) for item in items) / len(items)


def summarize_accuracy(
    evaluations: Iterable[Mapping[str, Any]],
) -> dict[str, int | float]:
    """Return counts and aggregate exact-match accuracy."""
    items = list(evaluations)
    accuracy = calculate_accuracy(items)
    correct = sum(bool(item["is_correct"]) for item in items)
    return {
        "number_of_problems": len(items),
        "correct_answers": correct,
        "incorrect_answers": len(items) - correct,
        "accuracy": accuracy,
    }

