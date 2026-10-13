from src.data.black_hole_splits import build_splits, make_question_splits, validate_records


def row(problem, index, label):
    return {"schema_version": "black-hole-training-dataset-v1", "dataset_row_id": f"id-{index}",
            "problem_id": problem, "trace_id": f"trace-{problem}", "sentence_id": f"T{index}",
            "label": label, "evaluation_key": f"evaluation-{index}",
            "context_before_mask": [True, True], "removal_mask": [False, True],
            "source_state_references": [{"state_id": "S1", "is_final_state": False}]}


def test_question_split_is_reproducible_and_has_no_leakage(tmp_path):
    rows = [row(f"q{question}", question * 2 + offset,
                "low_value_in_context" if offset else "answer_sensitive_in_context")
            for question in range(20) for offset in range(2)]
    groups = validate_records(rows)
    assert make_question_splits(groups, 42) == make_question_splits(groups, 42)
    source = tmp_path / "dataset.jsonl"
    source.write_text("".join(__import__("json").dumps(item) + "\n" for item in rows), encoding="utf-8")
    quality = build_splits(source, tmp_path / "splits", 42)
    seen = set()
    for split in quality["splits"].values():
        for problem in split["problem_ids"]:
            assert problem not in seen
            seen.add(problem)
    assert len(seen) == 20
    assert quality["leakage_problem_count"] == 0


def test_validation_rejects_duplicate_evaluation_key():
    records = [row("q1", 1, "low_value_in_context"), row("q2", 2, "answer_sensitive_in_context")]
    records[1]["evaluation_key"] = records[0]["evaluation_key"]
    try:
        validate_records(records)
    except ValueError as error:
        assert "Duplicate evaluation_key" in str(error)
    else:
        raise AssertionError("duplicate evidence must be rejected")
