import json

import pytest

from src.black_hole.trace_selection import save_trace_selection, select_eligible_traces


def make_record(index=0, *, correct=True, valid=True):
    trace_id = f"run-a:gsm8k-train-{index:05d}"
    problem_id = f"gsm8k-train-{index:05d}"
    reasoning = "First calculate 2 plus 2. Then the final answer is 4."
    first = "First calculate 2 plus 2."
    second = "Then the final answer is 4."
    return {
        "schema_version": "1.0",
        "problem_id": problem_id,
        "trace_id": trace_id,
        "question": "What is 2 + 2?",
        "reference_answer": "4",
        "predicted_answer": "4" if correct else "5",
        "is_correct": correct,
        "reasoning_trace": reasoning,
        "segmentation_valid": valid,
        "state_count": 2,
        "semantic_state_validation": {
            "trace_id": trace_id,
            "problem_id": problem_id,
            "is_valid": valid,
            "is_usable_for_week3": valid,
            "state_count": 2,
        },
        "experiment_metadata": {
            "generation_config": {"max_new_tokens": 1024},
            "model_metadata": {"model_id": "Qwen/Qwen3-4B"},
            "output_tokens": 20,
        },
        "states": [
            {
                "problem_id": problem_id,
                "trace_id": trace_id,
                "state_id": "S1",
                "state_index": 0,
                "text": first,
                "start_char": 0,
                "end_char": len(first),
                "token_count": 6,
                "is_final_state": False,
                "representation": {"vector_index": index * 2, "vector_file": "state_vectors.npy"},
            },
            {
                "problem_id": problem_id,
                "trace_id": trace_id,
                "state_id": "S2",
                "state_index": 1,
                "text": second,
                "start_char": len(first) + 1,
                "end_char": len(first) + 1 + len(second),
                "token_count": 7,
                "is_final_state": True,
                "representation": {"vector_index": index * 2 + 1, "vector_file": "state_vectors.npy"},
            },
        ],
    }


def test_selects_only_eligible_records_and_never_offers_final_state_for_removal():
    records = [make_record(0), make_record(1, correct=False), make_record(2, valid=False)]
    selected, audit, summary = select_eligible_traces(records)

    assert len(selected) == 1
    assert selected[0]["trace_id"] == records[0]["trace_id"]
    assert selected[0]["removable_state_ids"] == ["S1"]
    assert selected[0]["baseline_result"] == records[0]
    assert [row["eligible"] for row in audit] == [True, False, False]
    assert audit[1]["reason"] == "baseline_not_correct"
    assert "semantic_state_validation_failed" in audit[2]["reasons"]
    assert summary["eligible_trace_count"] == 1
    assert summary["final_answer_states_removable"] is False


def test_sampling_is_reproducible_and_reports_sampled_out_traces():
    records = [make_record(index) for index in range(10)]
    first = select_eligible_traces(records, random_seed=17, sample_size=4)
    second = select_eligible_traces(records, random_seed=17, sample_size=4)

    assert [row["trace_id"] for row in first[0]] == [row["trace_id"] for row in second[0]]
    assert first[2]["eligible_trace_count"] == 10
    assert first[2]["selected_trace_count"] == 4
    assert first[2]["sampled_out_trace_count"] == 6


def test_source_records_are_not_mutated_and_incomplete_metadata_is_excluded():
    record = make_record()
    record["experiment_metadata"].pop("model_metadata")
    before = json.loads(json.dumps(record))
    selected, audit, _ = select_eligible_traces([record])

    assert selected == []
    assert "missing_model_metadata" in audit[0]["reasons"]
    assert record == before


def test_rejects_duplicate_trace_ids_and_invalid_arguments():
    with pytest.raises(ValueError, match="duplicate trace_id"):
        select_eligible_traces([make_record(), make_record()])
    with pytest.raises(ValueError, match="sample_size"):
        select_eligible_traces([make_record()], sample_size=0)


def test_save_selection_writes_files_and_refuses_to_overwrite(tmp_path):
    selected, audit, summary = select_eligible_traces([make_record()])
    output = tmp_path / "selection"
    paths = save_trace_selection(output, selected, audit, summary)

    assert json.loads(paths["selected"].read_text(encoding="utf-8"))[
        "removable_state_ids"
    ] == ["S1"]
    assert json.loads(paths["audit"].read_text(encoding="utf-8"))["eligible"] is True
    assert json.loads(paths["summary"].read_text(encoding="utf-8"))["selected_trace_count"] == 1
    with pytest.raises(FileExistsError):
        save_trace_selection(output, selected, audit, summary)
