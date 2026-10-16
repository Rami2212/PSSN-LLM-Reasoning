import json

import numpy as np
import pytest

from src.black_hole.baselines import load_activation_splits, run_baselines


def _write_split(path, labels, groups, offset=0, state=False):
    count = len(labels)
    payload = {
        "features": np.arange(count * 4, dtype=np.float32).reshape(count, 4) + offset,
        "labels": np.asarray(labels, dtype=np.int64),
        "row_ids": np.asarray([f"row-{offset}-{i}" for i in range(count)]),
        "group_ids": np.asarray(groups),
    }
    if state:
        payload["state_features"] = np.arange(count * 2, dtype=np.float32).reshape(count, 2) + offset
    np.savez_compressed(path, **payload)


def _fixture_data(tmp_path, *, overlap=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _write_split(tmp_path / "train.npz", [0, 1, 0, 1, 1, 0], ["q1", "q1", "q2", "q2", "q3", "q3"], state=True)
    validation_groups = ["q3", "q4", "q4", "q5"] if overlap else ["q4", "q4", "q5", "q5"]
    _write_split(tmp_path / "validation.npz", [0, 1, 1, 0], validation_groups, offset=20, state=True)
    _write_split(tmp_path / "test.npz", [0, 1, 0, 1], ["q6", "q6", "q7", "q7"], offset=40, state=True)
    return tmp_path


def test_runs_metrics_on_held_out_validation_and_test_and_saves(tmp_path):
    data_dir = _fixture_data(tmp_path / "data")
    raw_path = tmp_path / "raw.jsonl"
    rows = []
    for split in ("train", "validation", "test"):
        with np.load(data_dir / f"{split}.npz", allow_pickle=False) as archive:
            rows.extend({"dataset_row_id": row_id, "sentence_text": "short sentence" if i % 2 else "a longer sentence has several words"}
                        for i, row_id in enumerate(archive["row_ids"].astype(str)))
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = run_baselines(data_dir, tmp_path / "out", raw_dataset=raw_path)

    assert "logistic_regression_sentence_activations" in result["baselines"]
    assert "random_forest_sentence_activations" in result["baselines"]
    assert "logistic_regression_state_vectors" in result["baselines"]
    assert "sentence_length_heuristic" in result["baselines"]
    for baseline in result["baselines"].values():
        assert "validation" in baseline and "test" in baseline
        assert "answer_sensitive_false_skip_rate" in baseline["test"]
    saved = json.loads((tmp_path / "out" / "baseline_results.json").read_text(encoding="utf-8"))
    assert saved["group_leakage_check"].startswith("passed")


def test_rejects_question_leakage_across_splits(tmp_path):
    data_dir = _fixture_data(tmp_path / "data", overlap=True)
    with pytest.raises(ValueError, match="group leakage"):
        load_activation_splits(data_dir)


def test_heuristic_requires_raw_dataset_rows(tmp_path):
    data_dir = _fixture_data(tmp_path / "data")
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(json.dumps({"dataset_row_id": "not-a-row", "sentence_text": "hello"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks"):
        run_baselines(data_dir, tmp_path / "out", raw_dataset=raw_path)
