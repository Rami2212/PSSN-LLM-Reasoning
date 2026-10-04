"""Tests for append-only JSONL experiment logging."""

import json
from datetime import datetime, timezone

import pytest

from src.utils.experiment_logger import ExperimentLogger


def test_logger_writes_jsonl_with_reproducible_run_metadata(tmp_path):
    output = tmp_path / "baseline" / "results.jsonl"
    logger = ExperimentLogger(
        output,
        experiment_name="qwen3-gsm8k-cot",
        run_id="run-001",
        timestamp=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc),
        run_metadata={"seed": 42},
    )
    payloads = logger.log_many(
        [
            {"problem_id": "p1", "output_tokens": 10},
            {"problem_id": "p2", "output_tokens": 20},
        ]
    )

    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert lines == payloads
    assert [line["problem_id"] for line in lines] == ["p1", "p2"]
    assert all(line["run_id"] == "run-001" for line in lines)
    assert all(line["run_metadata"] == {"seed": 42} for line in lines)
    assert lines[0]["experiment_timestamp_utc"] == "2026-01-02T03:04:00+00:00"


def test_logger_appends_without_overwriting_existing_results(tmp_path):
    output = tmp_path / "results.jsonl"
    logger = ExperimentLogger(output, experiment_name="baseline", run_id="run")
    logger.log({"problem_id": "first"})
    logger.log({"problem_id": "second"})
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_record_cannot_override_logger_owned_run_identity(tmp_path):
    logger = ExperimentLogger(
        tmp_path / "results.jsonl",
        experiment_name="baseline",
        run_id="trusted-run",
    )
    payload = logger.log({"problem_id": "p1", "run_id": "spoofed"})
    assert payload["run_id"] == "trusted-run"


def test_logger_rejects_non_jsonl_path_and_naive_timestamp(tmp_path):
    with pytest.raises(ValueError, match="jsonl"):
        ExperimentLogger(tmp_path / "results.csv", experiment_name="baseline")
    with pytest.raises(ValueError, match="timezone"):
        ExperimentLogger(
            tmp_path / "results.jsonl",
            experiment_name="baseline",
            timestamp=datetime(2026, 1, 1),
        )

