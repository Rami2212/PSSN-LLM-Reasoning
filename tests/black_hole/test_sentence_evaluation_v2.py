import hashlib
import json

import numpy as np
import pytest
import torch

from src.black_hole.config import BlackHoleDetectorConfig
from src.black_hole.detector import BlackHoleDetector
from src.black_hole.sentence_evaluation_v2 import evaluate_test_split


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_split(path, split, labels, groups):
    count = len(labels)
    np.savez_compressed(
        path,
        features=np.arange(count * 4, dtype=np.float32).reshape(count, 4),
        labels=np.asarray(labels, dtype=np.int64),
        row_ids=np.asarray([f"{split}-row-{i}" for i in range(count)]),
        group_ids=np.asarray(groups),
    )


def _prepare_evaluation(tmp_path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    _write_split(features_dir / "train.npz", "train", [0, 1], ["train-q0", "train-q1"])
    _write_split(features_dir / "validation.npz", "validation", [0, 1], ["val-q0", "val-q1"])
    _write_split(features_dir / "test.npz", "test", [0, 1], ["test-q0", "test-q1"])

    split_jsonl = tmp_path / "test.jsonl"
    rows = []
    for index, label in enumerate([0, 1]):
        rows.append({
            "dataset_row_id": f"test-row-{index}",
            "problem_id": f"test-q{index}",
            "trace_id": f"trace-{index}",
            "sentence_id": f"T{index + 1}",
            "label": "low_value_in_context" if label else "answer_sensitive_in_context",
        })
    split_jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    config = BlackHoleDetectorConfig(input_size=4, hidden_size=3, dropout=0.0)
    model = BlackHoleDetector(config)
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "detector_config": config.to_dict(),
        "feature_mean": torch.zeros(4),
        "feature_scale": torch.ones(4),
        "best_epoch": 4,
        "input_hashes": {
            "train": _sha256(features_dir / "train.npz"),
            "validation": _sha256(features_dir / "validation.npz"),
        },
    }, checkpoint_path)
    return features_dir, checkpoint_path, split_jsonl


def test_v2_evaluator_scores_only_test_and_saves_metrics_and_predictions(tmp_path):
    features_dir, checkpoint_path, split_jsonl = _prepare_evaluation(tmp_path)
    output_dir = tmp_path / "evaluation"

    result = evaluate_test_split(
        features_dir, checkpoint_path, split_jsonl, output_dir, device="cpu"
    )

    assert result["status"] == "completed"
    assert result["evaluation_split"] == "test"
    assert result["threshold_tuned_on_test"] is False
    assert result["metrics"]["row_count"] == 2
    assert result["metrics"]["probability_threshold"] == 0.5
    assert (output_dir / "test_evaluation_summary.json").is_file()
    prediction_rows = [
        json.loads(line)
        for line in (output_dir / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(prediction_rows) == 2
    assert {row["dataset_row_id"] for row in prediction_rows} == {"test-row-0", "test-row-1"}


def test_v2_evaluator_rejects_split_row_mismatch(tmp_path):
    features_dir, checkpoint_path, split_jsonl = _prepare_evaluation(tmp_path)
    split_jsonl.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_row_id"):
        evaluate_test_split(
            features_dir, checkpoint_path, split_jsonl, tmp_path / "evaluation", device="cpu"
        )


def test_v2_evaluator_rejects_checkpoints_from_other_training_features(tmp_path):
    features_dir, checkpoint_path, split_jsonl = _prepare_evaluation(tmp_path)
    _write_split(features_dir / "train.npz", "train", [0, 1], ["different-q0", "different-q1"])
    with pytest.raises(ValueError, match="train activation hash does not match"):
        evaluate_test_split(
            features_dir, checkpoint_path, split_jsonl, tmp_path / "evaluation", device="cpu"
        )
