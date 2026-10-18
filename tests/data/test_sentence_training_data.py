import hashlib
import json
import pytest

from src.data.sentence_training_data import (
    assign_splits,
    build_dataset,
    collect_records,
    make_training_record,
    select_balanced,
    record_paths,
)


def write_release(path, records, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([records, metadata]), encoding="utf-8")


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "source"
    for model in ("model-a", "model-b"):
        generation_name = "harp-standard_generations_difficulty=1_seed=42.json"
        generations, steps, attrs, tags = {}, {}, {}, {}
        for index in range(8):
            question = f"Question {index}?"
            key = f"results/{model}/{generation_name}:{index:05d}"
            generations[question] = {
                "input_str": question, "gt_answer": "15", "is_cot_complete": True,
                "cot_output": "Same.\nImportant.\nSame.\n", "full_input": "source tokens",
                "output": "source output",
            }
            steps[key] = {"input_str": question, "post_removal": {
                "sentences": ["Same.\n", "Important.\n", "Same.\n", "Answer: 15"],
                "answer_sentence_idx": 3,
            }}
            attrs[key] = {"prefix_length": 3}
            tags[key] = {
                str(i): {"sentence": sentence, "ground_truth_is_removable": i != 1,
                         "label_external": "nonremovable"}
                for i, sentence in enumerate(("Same.", "Important.", "Same."))
            }
        metadata = {"model_id": f"test/{model}", "n_resamples": 5}
        folder = root / model
        write_release(folder / generation_name, generations, metadata)
        write_release(folder / generation_name.replace("_generations", "_processed_sentences"),
                      steps, metadata)
        write_release(folder / generation_name.replace("_generations", "_attribution_pruning"),
                      attrs, metadata)
        write_release(folder / "harp-standard_sentence_removability_tags_seed=42.json",
                      tags, metadata)
    return root


def test_build_balances_and_groups_shared_questions_across_models(release, tmp_path):
    output = tmp_path / "dataset"
    summary = build_dataset(release, output, limit=24)
    rows = [json.loads(line) for line in (output / "sentence_training_data.jsonl").read_text().splitlines()]
    assert summary["row_count"] == 24
    assert summary["label_counts"] == {"answer_sensitive_in_context": 12, "low_value_in_context": 12}
    forbidden = {"label_source", "label_reason", "source_model_id", "source_dataset",
                 "source_entry_key", "source_protocol", "source_judge_predictions"}
    assert len({row["dataset_row_id"] for row in rows}) == 24
    question_splits = {}
    for row in rows:
        assert not forbidden.intersection(row)
        assert question_splits.setdefault(row["problem_id"], row["split"]) == row["split"]
        assert row["activation_status"] == "not_extracted"
    source_notes = output.with_name(output.name + "-source-notes")
    assert not (output / "data_source.jsonl").exists()
    assert not (output / "data_source.json").exists()
    assert "technion-cs-nlp/reasoning-importance" in (source_notes / "DATA_SOURCE.md").read_text()
    assert (source_notes / "data_source.json").is_file()
    source_summary = json.loads((source_notes / "data_source.json").read_text())
    assert source_summary["record_level_crosswalk"] is False
    assert "dataset_row_id" not in (source_notes / "data_source.json").read_text()
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert not manifest["activation_extraction_performed"]
    for name, expected in manifest["file_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    with pytest.raises(ValueError, match="Output already exists"):
        build_dataset(release, output, limit=24)


def test_repeated_sentences_use_position_not_first_text_match(release):
    rows, _, rejected, _ = collect_records(release)
    assert rejected == []
    target = next(row for row in rows if row["sentence_index"] == 2)
    assert target["reasoning_prefix"] == "Same.\nImportant.\n"
    assert target["sentence_text"] == "Same.\n"
    assert target["sentence_offsets"]["start_char_in_processed_cot"] == len(target["reasoning_prefix"])
    # Judge predicts nonremovable, but the experiment label is removable.
    assert target["label"] == "low_value_in_context"
    assert "source_judge_predictions" not in target


def test_wrong_sentence_is_rejected_and_ground_truth_false_is_kept(release):
    file = release / "model-a/harp-standard_sentence_removability_tags_seed=42.json"
    tags, metadata = json.loads(file.read_text())
    first_key = next(iter(tags))
    tags[first_key]["0"]["sentence"] = "This is the wrong trace."
    write_release(file, tags, metadata)
    rows, _, rejected, _ = collect_records(release)
    assert len(rejected) == 1
    assert rejected[0]["reason"] == "sentence_text_does_not_match_processed_index"
    assert any(row["label"] == "answer_sensitive_in_context" for row in rows)


def test_selection_deterministic_and_shortage_does_not_create_output(release, tmp_path):
    rows, source_details, _, _ = collect_records(release)
    first = select_balanced(rows, source_details, 24, 42)
    second = select_balanced(list(reversed(rows)), source_details, 24, 42)
    assert [r["dataset_row_id"] for r in first] == [r["dataset_row_id"] for r in second]
    assert assign_splits(first, 42) == assign_splits(second, 42)
    output = tmp_path / "too_many"
    with pytest.raises(ValueError, match="found only"):
        build_dataset(release, output, limit=5000)
    assert not output.exists()


def test_conflicting_experiment_labels_fail(release):
    file = release / "model-a/harp-standard_sentence_removability_tags_seed=42.json"
    tags, metadata = json.loads(file.read_text())
    key = next(iter(tags))
    tag = {**tags[key]["0"], "ground_truth_is_removable": False}
    write_release(file.with_name(file.name.replace("42", "43")), {key: {"0": tag}}, metadata)
    with pytest.raises(ValueError, match="Conflicting source labels"):
        collect_records(release)


def test_source_path_escape_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        record_paths(tmp_path, "results/../outside_generations.json:00000")


def test_answer_and_out_of_prefix_are_not_imported():
    trace = {"processed_sentences": ["Work.", "15"], "sufficient_prefix_length": 2,
             "processed_answer_sentence_index": 1, "trace_id": "t", "problem_id": "p",
             "question": "Question?", "reference_answer": "15"}
    with pytest.raises(ValueError, match="answer_suffix"):
        make_training_record({"sentence": "15", "ground_truth_is_removable": True}, 1, trace)
    with pytest.raises(ValueError, match="outside_sufficient_prefix"):
        make_training_record({"sentence": "15", "ground_truth_is_removable": True}, 2, trace)
