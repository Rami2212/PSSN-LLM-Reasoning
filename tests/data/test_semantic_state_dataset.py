import json

import pytest

from src.data.semantic_state_dataset import (
    build_semantic_state_dataset,
    save_semantic_state_dataset,
)


def state(trace_id, problem_id, index, text, start, *, final=False):
    return {
        "problem_id": problem_id,
        "trace_id": trace_id,
        "state_id": f"S{index + 1}",
        "state_index": index,
        "text": text,
        "start_char": start,
        "end_char": start + len(text),
        "token_count": len(text.split()),
        "is_final_state": final,
    }


def inputs():
    trace_id = "run-a:gsm8k-train-00001"
    reasoning = "First compute 2 plus 2. Then the answer is 4."
    states = [
        state(trace_id, "gsm8k-train-00001", 0, "First compute 2 plus 2.", 0),
        state(trace_id, "gsm8k-train-00001", 1, "Then the answer is 4.", 24, final=True),
    ]
    baseline = {
        "problem_id": "gsm8k-train-00001",
        "run_id": "run-a",
        "question": "What is 2 + 2?",
        "prompt": "Solve carefully.",
        "reasoning_trace": reasoning,
        "reference_answer": "4",
        "predicted_answer": "4",
        "is_correct": True,
        "generation_config": {"max_new_tokens": 1024},
        "model_metadata": {"model_id": "Qwen/Qwen3-4B"},
        "run_metadata": {"seed": 42},
    }
    report = {
        "trace_id": trace_id,
        "problem_id": "gsm8k-train-00001",
        "is_valid": True,
        "is_usable_for_week3": True,
        "state_count": 2,
        "usable_state_count": 2,
        "issues": [],
    }
    representations = [
        {
            "vector_index": index,
            "problem_id": "gsm8k-train-00001",
            "trace_id": trace_id,
            "state_id": f"S{index + 1}",
            "state_index": index,
            "layer": -1,
            "resolved_layer_index": 36,
            "pooling": "mean",
            "hidden_size": 2,
            "token_count": item["token_count"],
            "pooled_token_count": item["token_count"],
            "is_final_state": item["is_final_state"],
        }
        for index, item in enumerate(states)
    ]
    invalid_trace = "run-a:gsm8k-train-00002"
    invalid_states = [
        state(invalid_trace, "gsm8k-train-00002", 0, "Bad tiny.", 0, final=True)
    ]
    invalid_report = {
        "trace_id": invalid_trace,
        "problem_id": "gsm8k-train-00002",
        "is_valid": False,
        "is_usable_for_week3": False,
        "state_count": 1,
        "usable_state_count": 0,
        "issues": [{"code": "extremely_short_state", "severity": "error"}],
    }
    invalid_baseline = {
        **baseline,
        "problem_id": "gsm8k-train-00002",
        "run_id": "run-a",
        "reasoning_trace": "Bad tiny.",
    }
    return [baseline, invalid_baseline], states + invalid_states, [report, invalid_report], representations


def test_build_joins_only_valid_complete_sequences_and_keeps_vector_file_separate():
    baseline, states, reports, representations = inputs()
    records, metadata, summary = build_semantic_state_dataset(
        baseline,
        states,
        reports,
        representations,
        representation_config={"model_id": "Qwen/Qwen3-4B", "vector_shape": [2, 2]},
        vector_file="state_vectors.npy",
        source_files={"baseline": "baseline.jsonl"},
    )

    assert len(records) == 1
    record = records[0]
    assert record["question"] == "What is 2 + 2?"
    assert record["reasoning_trace"] == "First compute 2 plus 2. Then the answer is 4."
    assert record["is_correct"] is True
    assert [item["state_id"] for item in record["states"]] == ["S1", "S2"]
    assert [item["token_count"] for item in record["states"]] == [5, 5]
    assert [item["representation"]["vector_index"] for item in record["states"]] == [0, 1]
    assert all(item["representation"]["vector_file"] == "state_vectors.npy" for item in record["states"])
    assert record["semantic_state_validation"]["is_usable_for_week3"] is True
    assert "black_hole_label" not in record
    assert metadata["random_seed"] == 42
    assert metadata["randomness_used"] is False
    assert summary["included_trace_count"] == 1
    assert summary["included_state_count"] == 2
    assert summary["excluded_unvalidated_trace_count"] == 1


def test_build_rejects_missing_or_misaligned_vector_mappings():
    baseline, states, reports, representations = inputs()
    with pytest.raises(ValueError, match="no vector mapping"):
        build_semantic_state_dataset(
            baseline[:1], states[:2], reports[:1], representations[:1],
            representation_config={"vector_shape": [1, 2]}, vector_file="vectors.npy",
        )

    representations[0]["token_count"] += 1
    with pytest.raises(ValueError, match="token_count mismatch"):
        build_semantic_state_dataset(
            baseline[:1], states[:2], reports[:1], representations,
            representation_config={"vector_shape": [2, 2]}, vector_file="vectors.npy",
        )


def test_save_writes_machine_readable_files_and_refuses_to_overwrite(tmp_path):
    baseline, states, reports, representations = inputs()
    records, metadata, summary = build_semantic_state_dataset(
        baseline[:1], states[:2], reports[:1], representations,
        representation_config={"model_id": "Qwen/Qwen3-4B", "vector_shape": [2, 2]},
        vector_file="vectors.npy",
    )
    output = tmp_path / "dataset"
    paths = save_semantic_state_dataset(output, records, metadata, summary)
    loaded = json.loads(paths["dataset"].read_text(encoding="utf-8").splitlines()[0])
    assert loaded["states"][1]["representation"]["vector_index"] == 1
    assert json.loads(paths["metadata"].read_text(encoding="utf-8"))["randomness_used"] is False
    assert json.loads(paths["summary"].read_text(encoding="utf-8"))["included_state_count"] == 2
    with pytest.raises(FileExistsError):
        save_semantic_state_dataset(output, records, metadata, summary)
