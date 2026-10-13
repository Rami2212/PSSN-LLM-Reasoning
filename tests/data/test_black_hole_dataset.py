import json

from src.data.black_hole_dataset import write_dataset


def label(label_id="label-1", state_id="S1"):
    return {"schema_version": "black-hole-sentence-label-v1", "label_id": label_id,
            "label": "low_value_in_context", "label_reason": "validated_removal",
            "problem_id": "q1", "trace_id": "t1", "question": "Q", "reference_answer": "15",
            "sentence_id": "T1", "sentence_text": "Work.",
            "sentence_start_char_in_cot": 0, "sentence_end_char_in_cot": 5,
            "sentence_start_char_in_source": 7, "sentence_end_char_in_source": 12,
            "source_state_ids": [state_id], "context_before_mask": [True, True],
            "removal_mask": [False, True], "correct_count": 3, "sample_count": 3,
            "attainability": 0.9, "sufficiency": 1.0, "evaluation_key": "eval-1",
            "raw_trials_directory": "q1/trials", "core_summary_path": "q1/summary.json",
            "run_identity": "run", "model_id": "model", "model_revision": "revision",
            "protocol": {"name": "test"}}


def state_dataset():
    return {"problem_id": "q1", "trace_id": "t1", "states": [{"state_id": "S1", "state_index": 0,
            "text": "Work.", "token_count": 2, "is_final_state": False,
            "representation": {"vector_index": 0, "vector_file": "vectors.npy"}}]}


def test_builds_sentence_context_dataset_and_keeps_representation_reference(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(label()) + "\n", encoding="utf-8")
    states = tmp_path / "states.jsonl"
    states.write_text(json.dumps(state_dataset()) + "\n", encoding="utf-8")
    output = tmp_path / "output"
    summary = write_dataset(labels, states, output)
    assert summary["training_row_count"] == 1
    row = json.loads((output / "black_hole_training_dataset.jsonl").read_text())
    assert row["source_state_references"][0]["representation"]["vector_index"] == 0
    assert row["unit_type"] == "reasoning_sentence_context"


def test_excludes_missing_state_reference(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(label(state_id="S2")) + "\n", encoding="utf-8")
    states = tmp_path / "states.jsonl"
    states.write_text(json.dumps(state_dataset()) + "\n", encoding="utf-8")
    output = tmp_path / "output"
    try:
        write_dataset(labels, states, output)
    except ValueError as exc:
        assert "No accepted training records" in str(exc)
    else:
        raise AssertionError("expected missing references to prevent an empty dataset")
