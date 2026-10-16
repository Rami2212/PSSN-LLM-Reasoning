import json

from src.black_hole.labeling import classify_evaluation, label_evidence, write_outputs


PROTOCOL = {"n_resamples": 3, "minimum_correct": 2, "attainability_threshold": 0.8}


def evaluation(correct=3, attainability=0.9, calibration=True):
    return {"correct_count": correct, "complete_count": 3, "incomplete_count": 0,
            "attainability": attainability, "calibration_valid": calibration,
            "evaluation_key": f"key-{correct}-{attainability}-{calibration}", "mask": [False, True]}


def evidence(trace_id="trace-1", kind="gradient_greedy", **kwargs):
    return {"trace_id": trace_id, "problem_id": "q1", "question": "question", "reference_answer": "15",
            "run_identity": "run", "evidence_type": kind, "unit": {"sentence_id": "T1", "text": "work",
            "start_char_in_cot": 0, "end_char_in_cot": 4, "start_char_in_source": 7,
            "end_char_in_source": 11, "source_state_ids": ["S1"], "source_state_overlaps": []},
            "context_before_mask": [True, True], "evaluation": evaluation(**kwargs), "protocol": PROTOCOL,
            "raw_trials_directory": "q1/trials", "core_summary_path": "q1/summary.json",
            "model_id": "model", "model_revision": "revision"}


def test_classifies_rules():
    assert classify_evaluation(evaluation(3), PROTOCOL)[0] == "low_value_in_context"
    assert classify_evaluation(evaluation(1), PROTOCOL)[0] == "answer_sensitive_in_context"
    assert classify_evaluation(evaluation(2, 0.7), PROTOCOL)[0] == "plausibility_rejected"
    assert classify_evaluation(evaluation(3, calibration=False), PROTOCOL)[0] == "unscored"


def test_labels_only_validated_gradient_evidence(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    summary = {"total_traces": 2, "completed_traces": 2, "pending_traces": 0, "screened_eligible": 2,
               "validated_cores": 1, "removal_evidence_rows": 3, "protocol": PROTOCOL,
               "traces": [{"trace_id": "trace-1", "problem_id": "q1", "eligible": True,
                           "core_validation_passed": True},
                          {"trace_id": "trace-2", "problem_id": "q1", "eligible": True,
                           "core_validation_passed": False}]}
    (source / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = [evidence(), evidence(kind="random_greedy_42"), evidence("trace-2")]
    (source / "sentence_removal_evidence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _, labels, rejected, unscored, audits = label_evidence(source)
    assert len(labels) == 1
    assert not rejected
    assert not unscored
    assert {row["audit_reason"] for row in audits} == {"final_core_not_validated", "diagnostic_or_random_evidence"}
    output = tmp_path / "output"
    report = write_outputs(source, output)
    assert report["training_label_counts"] == {"low_value_in_context": 1}
    assert (output / "black_hole_sentence_labels.jsonl").exists()


def test_plausibility_rejected_is_not_a_training_label(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    summary = {"total_traces": 1, "completed_traces": 1, "pending_traces": 0, "screened_eligible": 1,
               "validated_cores": 1, "removal_evidence_rows": 1, "protocol": PROTOCOL,
               "traces": [{"trace_id": "trace-1", "problem_id": "q1", "eligible": True,
                           "core_validation_passed": True}]}
    (source / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (source / "sentence_removal_evidence.jsonl").write_text(
        json.dumps(evidence(attainability=0.7)) + "\n", encoding="utf-8")
    _, labels, rejected, _, _ = label_evidence(source)
    assert not labels
    assert rejected[0]["label"] == "plausibility_rejected"
