"""End-to-end execution and summarization of the GSM8K CoT baseline."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from src.evaluation.accuracy import evaluate_record
from src.utils.experiment_logger import ExperimentLogger

from .cot import ReasoningGenerator, generate_cot_record


def run_baseline_experiment(
    examples: Iterable[Mapping[str, Any]],
    generator: ReasoningGenerator,
    logger: ExperimentLogger,
    *,
    progress_callback: Callable[[int, Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Generate, evaluate, and immediately persist each baseline record.

    Logging each completed example protects earlier results if a long Colab run
    is interrupted. Generation errors intentionally propagate: a run should not
    claim its configured sample size unless every example completed successfully.
    Malformed textual outputs are valid records with ``predicted_answer=None``
    and can therefore be inspected and counted in the final summary.
    """
    completed: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        generation_record = generate_cot_record(example, generator)
        evaluation = evaluate_record(generation_record)
        combined = dict(generation_record)
        combined.update(
            {
                "predicted_answer": evaluation["predicted_answer"],
                "normalized_reference_answer": evaluation[
                    "normalized_reference_answer"
                ],
                "normalized_predicted_answer": evaluation[
                    "normalized_predicted_answer"
                ],
                "is_correct": evaluation["is_correct"],
            }
        )
        persisted = logger.log(combined)
        completed.append(persisted)
        if progress_callback is not None:
            progress_callback(index, persisted)
    return completed


def _require_numeric(record: Mapping[str, Any], field: str) -> int | float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Every result must contain numeric field {field!r}.")
    return value


def _consistent_value(records: list[Mapping[str, Any]], field: str) -> Any:
    value = records[0].get(field)
    if any(record.get(field) != value for record in records[1:]):
        raise ValueError(f"Result field {field!r} is inconsistent within the run.")
    return value


def build_baseline_summary(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate accuracy, token, latency, memory, and run metadata."""
    items = list(records)
    if not items:
        raise ValueError("Cannot summarize an empty baseline experiment.")
    if any("is_correct" not in record for record in items):
        raise ValueError("Every result must contain 'is_correct'.")

    output_tokens = [
        _require_numeric(record, "output_tokens") for record in items
    ]
    input_tokens = [_require_numeric(record, "input_tokens") for record in items]
    latencies = [
        _require_numeric(record, "inference_latency_seconds") for record in items
    ]
    allocated = [
        _require_numeric(record, "peak_gpu_memory_allocated_bytes")
        for record in items
    ]
    reserved = [
        _require_numeric(record, "peak_gpu_memory_reserved_bytes")
        for record in items
    ]
    correct = sum(bool(record["is_correct"]) for record in items)
    malformed = sum(record.get("predicted_answer") is None for record in items)
    generation_config = _consistent_value(items, "generation_config")
    model_metadata = _consistent_value(items, "model_metadata")
    run_metadata = _consistent_value(items, "run_metadata") or {}
    if not isinstance(generation_config, Mapping):
        raise ValueError("Every result must contain a generation configuration.")
    if not isinstance(model_metadata, Mapping):
        raise ValueError("Every result must contain model metadata.")
    if not isinstance(run_metadata, Mapping):
        raise ValueError("Every result must contain mapping-valued run metadata.")

    return {
        "run_id": _consistent_value(items, "run_id"),
        "experiment_name": _consistent_value(items, "experiment_name"),
        "experiment_timestamp_utc": _consistent_value(
            items, "experiment_timestamp_utc"
        ),
        "number_of_problems": len(items),
        "correct_answers": correct,
        "incorrect_answers": len(items) - correct,
        "malformed_outputs": malformed,
        "accuracy": correct / len(items),
        "average_input_tokens": statistics.fmean(input_tokens),
        "average_generated_tokens": statistics.fmean(output_tokens),
        "median_generated_tokens": statistics.median(output_tokens),
        "average_inference_latency_seconds": statistics.fmean(latencies),
        "peak_gpu_memory_allocated_bytes": max(allocated),
        "peak_gpu_memory_reserved_bytes": max(reserved),
        "model": model_metadata.get("model_id"),
        "precision": model_metadata.get("precision"),
        "model_metadata": dict(model_metadata),
        "generation_configuration": dict(generation_config),
        "random_seed": run_metadata.get("seed", generation_config.get("seed")),
        "run_metadata": dict(run_metadata),
    }


def save_baseline_summary(
    summary: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Save a human-readable JSON summary using an atomic file replacement."""
    path = Path(output_path)
    if path.suffix.lower() != ".json":
        raise ValueError("Baseline summary path must use the .json extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    serialized = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
    temporary_path.write_text(serialized + "\n", encoding="utf-8")
    temporary_path.replace(path)
    return path

