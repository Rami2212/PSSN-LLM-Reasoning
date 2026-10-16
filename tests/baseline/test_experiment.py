"""Tests for streaming baseline execution and summary generation."""

import json
from datetime import datetime, timezone

import pytest

from src.baseline.experiment import (
    build_baseline_summary,
    run_baseline_experiment,
    save_baseline_summary,
)
from src.models.qwen import GenerationResult
from src.utils.experiment_logger import ExperimentLogger


class StubGenerator:
    def generate(self, prompt):
        answer = "2" if "one" in prompt else "9"
        return GenerationResult(
            text=f"Reasoning. Final answer: {answer}",
            input_tokens=10,
            output_tokens=6,
            generation_config={"seed": 42, "do_sample": False},
            metrics={
                "inference_latency_seconds": 0.5,
                "peak_gpu_memory_allocated_bytes": 100,
                "peak_gpu_memory_reserved_bytes": 200,
            },
            model_metadata={
                "model_id": "Qwen/Qwen3-4B",
                "precision": "bfloat16",
            },
        )


def make_logger(path):
    return ExperimentLogger(
        path,
        experiment_name="test-baseline",
        run_id="run-1",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        run_metadata={"seed": 42, "subset_size": 2},
    )


def test_experiment_generates_evaluates_and_streams_records(tmp_path):
    examples = [
        {"id": "one", "question": "one plus one?", "reference_answer": "2"},
        {"id": "two", "question": "three squared?", "reference_answer": "8"},
    ]
    progress = []
    output = tmp_path / "results.jsonl"
    records = run_baseline_experiment(
        examples,
        StubGenerator(),
        make_logger(output),
        progress_callback=lambda index, record: progress.append(
            (index, record["problem_id"])
        ),
    )

    persisted = [json.loads(line) for line in output.read_text().splitlines()]
    assert records == persisted
    assert [record["is_correct"] for record in records] == [True, False]
    assert progress == [(1, "one"), (2, "two")]
    assert all(record["reasoning_trace"] for record in records)


def test_build_and_save_baseline_summary(tmp_path):
    examples = [
        {"id": "one", "question": "one plus one?", "reference_answer": "2"},
        {"id": "two", "question": "three squared?", "reference_answer": "8"},
    ]
    records = run_baseline_experiment(
        examples,
        StubGenerator(),
        make_logger(tmp_path / "results.jsonl"),
    )
    records[1]["output_tokens"] = 10
    records[1]["inference_latency_seconds"] = 1.5
    records[1]["peak_gpu_memory_allocated_bytes"] = 150
    summary = build_baseline_summary(records)

    assert summary["number_of_problems"] == 2
    assert summary["correct_answers"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["average_generated_tokens"] == 8
    assert summary["median_generated_tokens"] == 8
    assert summary["average_inference_latency_seconds"] == 1.0
    assert summary["peak_gpu_memory_allocated_bytes"] == 150
    assert summary["model"] == "Qwen/Qwen3-4B"
    assert summary["random_seed"] == 42

    path = save_baseline_summary(summary, tmp_path / "summary.json")
    assert json.loads(path.read_text()) == summary


def test_summary_rejects_empty_or_inconsistent_runs():
    with pytest.raises(ValueError, match="empty"):
        build_baseline_summary([])
    records = [
        {
            "is_correct": True,
            "output_tokens": 1,
            "input_tokens": 1,
            "inference_latency_seconds": 1.0,
            "peak_gpu_memory_allocated_bytes": 1,
            "peak_gpu_memory_reserved_bytes": 1,
            "generation_config": {"seed": 1},
            "model_metadata": {},
            "run_metadata": {},
            "run_id": run_id,
            "experiment_name": "test",
            "experiment_timestamp_utc": "time",
        }
        for run_id in ("first", "second")
    ]
    with pytest.raises(ValueError, match="run_id"):
        build_baseline_summary(records)

