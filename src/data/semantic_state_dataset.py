"""Build reproducible Week 2 datasets from validated traces and state vectors."""

from __future__ import annotations

import json
from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.semantic_states.validation import validate_state_sequence


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL records and report malformed lines with their source location."""
    source = Path(path)
    records = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {source} at line {line_number}.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {source} at line {line_number}.")
            records.append(record)
    return records


def _trace_id(record: Mapping[str, Any]) -> str:
    trace_id = record.get("trace_id")
    if isinstance(trace_id, str) and trace_id.strip():
        return trace_id
    run_id, problem_id = record.get("run_id"), record.get("problem_id")
    if isinstance(run_id, str) and run_id.strip() and isinstance(problem_id, str) and problem_id.strip():
        return f"{run_id}:{problem_id}"
    raise ValueError("each baseline trace must have trace_id or both run_id and problem_id")


def _require_text(record: Mapping[str, Any], field: str, trace_id: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"baseline trace {trace_id} has no non-empty {field}")
    return value


def build_semantic_state_dataset(
    baseline_records: Iterable[Mapping[str, Any]],
    state_records: Iterable[Mapping[str, Any]],
    validation_reports: Iterable[Mapping[str, Any]],
    representation_index: Iterable[Mapping[str, Any]],
    *,
    representation_config: Mapping[str, Any],
    vector_file: str | Path,
    random_seed: int = 42,
    validation_thresholds: Mapping[str, Any] | None = None,
    source_files: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Join valid baseline traces, whole validated state sequences, and vectors.

    Representation vectors stay in their separate NumPy file. Each state stores
    the vector row and extraction metadata needed to retrieve its representation.
    The input iterables are never mutated, and output order follows the baseline
    trace order and each trace's original state order.
    """
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer")
    if not isinstance(representation_config, Mapping):
        raise TypeError("representation_config must be a mapping")
    vector_shape = representation_config.get("vector_shape")
    if (
        not isinstance(vector_shape, list)
        or len(vector_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in vector_shape)
    ):
        raise ValueError("representation_config.vector_shape must be a positive [rows, dimensions] list")

    baselines = [dict(record) for record in baseline_records]
    states = [dict(record) for record in state_records]
    reports = [dict(record) for record in validation_reports]
    vector_rows = [dict(record) for record in representation_index]

    baseline_by_trace: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record in baselines:
        trace_id = _trace_id(record)
        if trace_id in baseline_by_trace:
            raise ValueError(f"duplicate baseline trace_id: {trace_id}")
        if record.get("is_correct") not in (True, False):
            raise ValueError(f"baseline trace {trace_id} has no boolean correctness result")
        baseline_by_trace[trace_id] = record

    report_by_trace: dict[str, dict[str, Any]] = {}
    for report in reports:
        trace_id = report.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id.strip():
            raise ValueError("every semantic-state validation report must have a trace_id")
        if trace_id in report_by_trace:
            raise ValueError(f"duplicate semantic-state validation report: {trace_id}")
        report_by_trace[trace_id] = report

    states_by_trace: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for state in states:
        trace_id = state.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id.strip():
            raise ValueError("every semantic state must have a trace_id")
        states_by_trace.setdefault(trace_id, []).append(state)

    representation_by_state: dict[tuple[str, str, str], dict[str, Any]] = {}
    vector_indices: list[int] = []
    for row in vector_rows:
        trace_id, problem_id, state_id = row.get("trace_id"), row.get("problem_id"), row.get("state_id")
        if not all(isinstance(value, str) and value.strip() for value in (trace_id, problem_id, state_id)):
            raise ValueError("each representation index row needs trace_id, problem_id, and state_id")
        key = (trace_id, problem_id, state_id)
        if key in representation_by_state:
            raise ValueError(f"duplicate vector mapping for state {key}")
        vector_index = row.get("vector_index")
        if isinstance(vector_index, bool) or not isinstance(vector_index, int) or vector_index < 0:
            raise ValueError(f"state {key} has an invalid vector_index")
        vector_indices.append(vector_index)
        representation_by_state[key] = row

    if vector_shape[0] != len(vector_rows):
        raise ValueError("representation vector row count does not match representation index length")
    if sorted(vector_indices) != list(range(vector_shape[0])):
        raise ValueError("representation vector_index values must cover the vector matrix exactly once")

    vector_path = str(vector_file)
    records: list[dict[str, Any]] = []
    included_trace_ids: set[str] = set()
    token_counts: list[int] = []
    correctness_counts: Counter[str] = Counter()

    orphan_state_traces = set(states_by_trace) - set(baseline_by_trace)
    if orphan_state_traces:
        raise ValueError(f"semantic states have no matching baseline traces: {sorted(orphan_state_traces)[:3]}")

    for trace_id in baseline_by_trace:
        trace_states = states_by_trace.get(trace_id)
        if not trace_states:
            continue
        report = report_by_trace.get(trace_id)
        if report is None:
            raise ValueError(f"states for {trace_id} have no quality-validation report")
        if report.get("is_valid") is not True or report.get("is_usable_for_week3") is not True:
            continue
        baseline = baseline_by_trace.get(trace_id)
        if baseline is None:
            raise ValueError(f"no baseline record matches validated trace {trace_id}")
        if report.get("problem_id") != baseline.get("problem_id"):
            raise ValueError(f"problem_id mismatch between validation report and baseline for {trace_id}")
        reasoning_trace = _require_text(baseline, "reasoning_trace", trace_id)
        thresholds = dict(validation_thresholds or {})
        sequence_validation = validate_state_sequence(
            trace_states,
            source_trace=reasoning_trace,
            min_state_tokens=thresholds.get("min_state_tokens", 3),
            min_usable_states=thresholds.get("min_usable_states", 2),
            max_state_tokens=thresholds.get("max_state_tokens_warning", 512),
        )
        if not sequence_validation["is_usable_for_week3"]:
            raise ValueError(f"trace {trace_id} failed re-validation: {sequence_validation['issues']}")
        if sequence_validation["state_count"] != report.get("state_count"):
            raise ValueError(f"state count differs from validation report for {trace_id}")

        ordered_states = sorted(trace_states, key=lambda state: state.get("state_index", -1))
        joined_states = []
        for state in ordered_states:
            state_id = state["state_id"]
            if state.get("problem_id") != baseline.get("problem_id"):
                raise ValueError(f"problem_id mismatch for {trace_id}/{state_id}")
            key = (trace_id, baseline["problem_id"], state_id)
            representation = representation_by_state.get(key)
            if representation is None:
                raise ValueError(f"no vector mapping for validated state {key}")
            if representation.get("state_index") != state.get("state_index"):
                raise ValueError(f"state_index mismatch for {key}")
            if representation.get("token_count") != state.get("token_count"):
                raise ValueError(f"token_count mismatch for {key}")
            state_representation = dict(representation)
            state_representation["vector_file"] = vector_path
            joined_states.append(
                {
                    **state,
                    "representation": state_representation,
                }
            )
            token_counts.append(state["token_count"])

        included_trace_ids.add(trace_id)
        correctness_counts["correct" if baseline["is_correct"] else "incorrect"] += 1
        excluded_content_fields = {
            "problem_id", "question", "prompt", "reasoning_trace", "reference_answer",
            "predicted_answer", "normalized_reference_answer", "normalized_predicted_answer",
            "is_correct", "trace_id",
        }
        experiment_metadata = {
            key: value for key, value in baseline.items() if key not in excluded_content_fields
        }
        records.append(
            {
                "schema_version": "1.0",
                "problem_id": baseline["problem_id"],
                "trace_id": trace_id,
                "question": _require_text(baseline, "question", trace_id),
                "reference_answer": _require_text(baseline, "reference_answer", trace_id),
                "predicted_answer": baseline["predicted_answer"],
                "is_correct": baseline["is_correct"],
                "reasoning_trace": reasoning_trace,
                "states": joined_states,
                "state_count": len(joined_states),
                "segmentation_valid": True,
                "semantic_state_validation": report,
                "experiment_metadata": experiment_metadata,
            }
        )

    # Prevent a partially represented dataset: every vector must point to one
    # included, validated state, with no extra or missing representations.
    expected_keys = {
        (state["trace_id"], state["problem_id"], state["state_id"])
        for trace_id, trace_states in states_by_trace.items()
        if trace_id in included_trace_ids
        for state in trace_states
    }
    if set(representation_by_state) != expected_keys:
        missing = expected_keys - set(representation_by_state)
        extra = set(representation_by_state) - expected_keys
        raise ValueError(
            f"representation index does not exactly cover validated states (missing={len(missing)}, extra={len(extra)})"
        )

    if not records:
        raise ValueError("no validated semantic-state traces were available to build the dataset")

    validation_report_count = len(reports)
    metadata = {
        "dataset_name": "PSSN-sementic-state-math-problems",
        "schema_version": "1.0",
        "random_seed": random_seed,
        "randomness_used": False,
        "ordering": "baseline trace order; original state_index order within each trace",
        "included_trace_count": len(records),
        "included_state_count": sum(record["state_count"] for record in records),
        "validation_report_count": validation_report_count,
        "representation_config": dict(representation_config),
        "vector_file": vector_path,
        "validation_thresholds": dict(validation_thresholds or {}),
        "source_files": dict(source_files or {}),
        "labels_note": "No Black Hole or low/high-value state labels are included.",
    }
    metadata["included_state_count"] = sum(record["state_count"] for record in records)
    summary = {
        "baseline_trace_count": len(baselines),
        "validation_report_count": validation_report_count,
        "validated_week3_trace_count": sum(
            report.get("is_valid") is True and report.get("is_usable_for_week3") is True
            for report in reports
        ),
        "included_trace_count": len(records),
        "included_state_count": metadata["included_state_count"],
        "excluded_unvalidated_trace_count": sum(
            report.get("is_valid") is not True or report.get("is_usable_for_week3") is not True
            for report in reports
        ),
        "correct_trace_count": correctness_counts["correct"],
        "incorrect_trace_count": correctness_counts["incorrect"],
        "state_token_count": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": sum(token_counts) / len(token_counts),
            "total": sum(token_counts),
        },
        "representation_dimension": vector_shape[1],
        "representation_model_id": representation_config.get("model_id"),
        "representation_model_revision": representation_config.get("model_revision"),
    }
    return records, metadata, summary


def save_semantic_state_dataset(
    output_dir: str | Path,
    records: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Path]:
    """Write dataset JSONL, metadata, and summary into a new directory only."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    paths = {
        "dataset": destination / "semantic_states.jsonl",
        "metadata": destination / "semantic_state_metadata.json",
        "summary": destination / "semantic_state_summary.json",
    }
    with paths["dataset"].open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
    paths["metadata"].write_text(
        json.dumps(dict(metadata), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["summary"].write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths
