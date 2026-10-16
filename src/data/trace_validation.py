"""Validation and selection of baseline traces for semantic segmentation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


_REQUIRED_FIELDS = (
    "problem_id",
    "question",
    "reasoning_trace",
    "reference_answer",
    "predicted_answer",
    "is_correct",
    "input_tokens",
    "output_tokens",
    "finish_reason",
    "generation_config",
    "inference_latency_seconds",
    "peak_gpu_memory_allocated_bytes",
    "peak_gpu_memory_reserved_bytes",
)
_METRIC_FIELDS = (
    "inference_latency_seconds",
    "peak_gpu_memory_allocated_bytes",
    "peak_gpu_memory_reserved_bytes",
)


def validate_trace(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return explicit structural/completion validity and segmentation eligibility.

    A complete, parseable but incorrect response remains a valid completed trace
    for audit. Only complete, correct traces are marked eligible for the
    successful-trace semantic-state dataset.
    """
    errors: list[str] = []
    if not isinstance(record, Mapping):
        return {
            "problem_id": None,
            "trace_valid": False,
            "segmentation_eligible": False,
            "validation_status": "invalid",
            "validation_errors": ["record_not_mapping"],
        }

    problem_id = record.get("problem_id")
    for field in _REQUIRED_FIELDS:
        if field not in record or record[field] is None:
            errors.append(f"missing_{field}")

    for field in ("problem_id", "question", "reasoning_trace"):
        value = record.get(field)
        if field in record and (not isinstance(value, str) or not value.strip()):
            errors.append(f"empty_or_invalid_{field}")

    for field in ("input_tokens", "output_tokens"):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"invalid_{field}")
        elif value == 0:
            errors.append(f"empty_{field}")

    generation_config = record.get("generation_config")
    cap = None
    if not isinstance(generation_config, Mapping):
        errors.append("invalid_generation_config")
    else:
        cap = generation_config.get("max_new_tokens")
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            errors.append("invalid_max_new_tokens")

    for field in _METRIC_FIELDS:
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"invalid_{field}")
        elif not math.isfinite(value) or value < 0:
            errors.append(f"invalid_{field}")

    if not isinstance(record.get("is_correct"), bool):
        errors.append("missing_correctness_label")
    if not isinstance(record.get("predicted_answer"), str) or not str(
        record.get("predicted_answer", "")
    ).strip():
        errors.append("missing_extracted_answer")

    finish_reason = record.get("finish_reason")
    if finish_reason == "length":
        errors.append("incomplete_token_limit_stop")
    elif finish_reason != "eos":
        errors.append("unverified_completion_stop")

    output_tokens = record.get("output_tokens")
    if cap is not None and isinstance(output_tokens, int) and output_tokens > cap:
        errors.append("output_exceeds_token_cap")
    if (
        cap is not None
        and isinstance(output_tokens, int)
        and output_tokens >= cap
        and finish_reason != "eos"
        and "incomplete_token_limit_stop" not in errors
    ):
        errors.append("reached_cap_without_eos")

    # Keep the error list deterministic if multiple checks identify the same issue.
    errors = list(dict.fromkeys(errors))
    valid = not errors
    correct = record.get("is_correct") is True
    eligible = valid and correct
    status = (
        "segmentation_eligible"
        if eligible
        else "complete_incorrect"
        if valid
        else "invalid"
    )
    return {
        "problem_id": problem_id,
        "trace_valid": valid,
        "segmentation_eligible": eligible,
        "validation_status": status,
        "validation_errors": errors,
    }


def validate_traces(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate records in their original order."""
    return [validate_trace(record) for record in records]


def summarize_trace_validation(
    records: Iterable[Mapping[str, Any]],
    validations: Iterable[Mapping[str, Any]],
    *,
    target_valid_traces: int = 50,
) -> dict[str, Any]:
    """Summarize attempted, valid, excluded, and segmentation-ready traces."""
    raw = list(records)
    checked = list(validations)
    if len(raw) != len(checked):
        raise ValueError("Each raw trace must have exactly one validation result.")
    if isinstance(target_valid_traces, bool) or not isinstance(
        target_valid_traces, int
    ) or target_valid_traces <= 0:
        raise ValueError("target_valid_traces must be a positive integer.")

    valid_count = sum(item.get("trace_valid") is True for item in checked)
    eligible_count = sum(
        item.get("segmentation_eligible") is True for item in checked
    )
    invalid = [item for item in checked if item.get("trace_valid") is not True]
    return {
        "number_of_attempts": len(raw),
        "valid_completed_traces": valid_count,
        "invalid_or_incomplete_traces": len(invalid),
        "segmentation_eligible_correct_traces": eligible_count,
        "completed_incorrect_traces": sum(
            item.get("validation_status") == "complete_incorrect" for item in checked
        ),
        "target_valid_traces": target_valid_traces,
        "target_valid_traces_met": valid_count >= target_valid_traces,
        "target_segmentation_traces_met": eligible_count >= target_valid_traces,
        "completion_rate": valid_count / len(raw) if raw else 0.0,
        "invalid_problem_ids": [item.get("problem_id") for item in invalid],
        "segmentation_problem_ids": [
            item.get("problem_id")
            for item in checked
            if item.get("segmentation_eligible") is True
        ],
    }
