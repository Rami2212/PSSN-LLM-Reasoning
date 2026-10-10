"""Reproducible selection of Week 2 traces for Black Hole experiments."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.semantic_states.validation import validate_state_sequence


_REQUIRED_METADATA_KEYS = ("generation_config", "model_metadata")


def _non_empty_string(record: Mapping[str, Any], field: str) -> bool:
    value = record.get(field)
    return isinstance(value, str) and bool(value.strip())


def _eligibility_reasons(
    record: Mapping[str, Any],
    *,
    min_state_count: int,
    min_state_tokens: int,
) -> list[str]:
    """Return stable reason codes for every reason a record is ineligible."""
    reasons: list[str] = []
    trace_id = record.get("trace_id")
    if not _non_empty_string(record, "trace_id"):
        reasons.append("missing_trace_id")
    if not _non_empty_string(record, "problem_id"):
        reasons.append("missing_problem_id")
    if record.get("is_correct") is not True:
        reasons.append("baseline_not_correct")
    for field in ("question", "reference_answer", "predicted_answer", "reasoning_trace"):
        if not _non_empty_string(record, field):
            reasons.append(f"missing_{field}")

    validation = record.get("semantic_state_validation")
    if not isinstance(validation, Mapping):
        reasons.append("missing_validation_report")
    elif validation.get("is_valid") is not True or validation.get("is_usable_for_week3") is not True:
        reasons.append("semantic_state_validation_failed")
    if record.get("segmentation_valid") is not True:
        reasons.append("segmentation_invalid")

    states = record.get("states")
    if not isinstance(states, list) or not states:
        reasons.append("missing_states")
        return reasons
    if len(states) < min_state_count:
        reasons.append("insufficient_state_count")
    declared_count = record.get("state_count")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int) or declared_count != len(states):
        reasons.append("state_count_mismatch")

    if _non_empty_string(record, "trace_id") and isinstance(validation, Mapping):
        if validation.get("trace_id") != trace_id:
            reasons.append("validation_trace_id_mismatch")
    if _non_empty_string(record, "problem_id") and isinstance(validation, Mapping):
        if validation.get("problem_id") != record.get("problem_id"):
            reasons.append("validation_problem_id_mismatch")

    state_ids: set[str] = set()
    final_states: list[Mapping[str, Any]] = []
    usable_states = 0
    for position, state in enumerate(states):
        if not isinstance(state, Mapping):
            reasons.append("invalid_state_record")
            continue
        state_id = state.get("state_id")
        if not isinstance(state_id, str) or not state_id.strip() or state_id in state_ids:
            reasons.append("missing_or_duplicate_state_id")
        else:
            state_ids.add(state_id)
        if state.get("trace_id") != trace_id or state.get("problem_id") != record.get("problem_id"):
            reasons.append("state_metadata_mismatch")
        if state.get("state_index") != position or state_id != f"S{position + 1}":
            reasons.append("invalid_state_order")
        if state.get("is_final_state") is True:
            final_states.append(state)
        token_count = state.get("token_count")
        if (
            isinstance(token_count, int)
            and not isinstance(token_count, bool)
            and token_count >= min_state_tokens
        ):
            usable_states += 1
        representation = state.get("representation")
        if not isinstance(representation, Mapping):
            reasons.append("missing_state_representation")
        else:
            vector_index = representation.get("vector_index")
            if isinstance(vector_index, bool) or not isinstance(vector_index, int) or vector_index < 0:
                reasons.append("invalid_state_representation")
            if not _non_empty_string(representation, "vector_file"):
                reasons.append("missing_state_representation")

    if isinstance(record.get("reasoning_trace"), str) and all(isinstance(state, Mapping) for state in states):
        sequence_report = validate_state_sequence(
            states,
            source_trace=record["reasoning_trace"],
            min_state_tokens=min_state_tokens,
            min_usable_states=min_state_count,
            max_state_tokens=None,
        )
        if not sequence_report["is_usable_for_week3"]:
            reasons.append("semantic_state_sequence_invalid")

    if len(final_states) != 1 or (states and final_states and final_states[0] is not states[-1]):
        reasons.append("invalid_final_state_assignment")
    if isinstance(validation, Mapping) and validation.get("state_count") != len(states):
        reasons.append("validation_state_count_mismatch")
    if usable_states < min_state_count:
        reasons.append("insufficient_usable_states")

    metadata = record.get("experiment_metadata")
    if not isinstance(metadata, Mapping):
        reasons.append("missing_experiment_metadata")
    else:
        for field in _REQUIRED_METADATA_KEYS:
            if not isinstance(metadata.get(field), Mapping) or not metadata[field]:
                reasons.append(f"missing_{field}")

    return list(dict.fromkeys(reasons))


def select_eligible_traces(
    records: Iterable[Mapping[str, Any]],
    *,
    random_seed: int = 42,
    sample_size: int | None = None,
    min_state_count: int = 2,
    min_state_tokens: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Filter valid, correct traces and optionally take a seeded sample.

    Returns selected trace records, an eligibility audit row for every input,
    and a summary. Each selected record contains an unchanged copy of its full
    source record in ``baseline_result`` and the non-final removal candidates.
    The source iterable and its nested records are never mutated.
    """
    for name, value in (
        ("random_seed", random_seed),
        ("min_state_count", min_state_count),
        ("min_state_tokens", min_state_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "random_seed" else 1):
            raise ValueError(f"{name} must be an integer >= {0 if name == 'random_seed' else 1}")
    if sample_size is not None and (
        isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1
    ):
        raise ValueError("sample_size must be a positive integer or None")

    source_records = [dict(record) if isinstance(record, Mapping) else {} for record in records]
    trace_ids = [
        record.get("trace_id")
        for record in source_records
        if _non_empty_string(record, "trace_id")
    ]
    if len(trace_ids) != len(set(trace_ids)):
        raise ValueError("input dataset contains duplicate trace_id values")

    eligible_indices: list[int] = []
    audit: list[dict[str, Any]] = []
    for index, record in enumerate(source_records):
        reasons = _eligibility_reasons(
            record,
            min_state_count=min_state_count,
            min_state_tokens=min_state_tokens,
        )
        eligible = not reasons
        if eligible:
            eligible_indices.append(index)
        audit.append(
            {
                "problem_id": record.get("problem_id"),
                "trace_id": record.get("trace_id"),
                "eligible": eligible,
                "reason": None if eligible else reasons[0],
                "reasons": reasons,
                "state_count": (
                    len(record.get("states", []))
                    if isinstance(record.get("states"), list)
                    else 0
                ),
                "baseline_correct": record.get("is_correct") is True,
            }
        )

    if sample_size is not None and len(eligible_indices) > sample_size:
        eligible_indices = sorted(random.Random(random_seed).sample(eligible_indices, sample_size))

    selected: list[dict[str, Any]] = []
    selected_set = set(eligible_indices)
    for index in eligible_indices:
        record = source_records[index]
        states = record["states"]
        selected_record = {
            "problem_id": record["problem_id"],
            "trace_id": record["trace_id"],
            "eligible": True,
            "state_count": len(states),
            "baseline_correct": True,
            "removable_state_ids": [state["state_id"] for state in states if state.get("is_final_state") is not True],
            "baseline_result": record,
        }
        selected.append(selected_record)
    for index, audit_record in enumerate(audit):
        audit_record["selected_for_experiment"] = index in selected_set

    summary = {
        "schema_version": "1.0",
        "random_seed": random_seed,
        "sample_size": sample_size,
        "min_state_count": min_state_count,
        "min_state_tokens": min_state_tokens,
        "input_trace_count": len(source_records),
        "eligible_trace_count": (
            len(eligible_indices)
            if sample_size is None
            else sum(row["eligible"] for row in audit)
        ),
        "selected_trace_count": len(selected),
        "excluded_trace_count": sum(not row["eligible"] for row in audit),
        "sampled_out_trace_count": sum(row["eligible"] and not row["selected_for_experiment"] for row in audit),
        "selected_trace_ids": [row["trace_id"] for row in selected],
        "final_answer_states_removable": False,
    }
    return selected, audit, summary


def save_trace_selection(
    output_dir: str | Path,
    selected: Iterable[Mapping[str, Any]],
    audit: Iterable[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Path]:
    """Save the selection, full eligibility audit, and summary to a new folder."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    paths = {
        "selected": destination / "eligible_traces.jsonl",
        "audit": destination / "eligibility_audit.jsonl",
        "summary": destination / "selection_summary.json",
    }
    for key, records in (("selected", selected), ("audit", audit)):
        with paths[key].open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        dict(record),
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
    paths["summary"].write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return paths
