import json

import numpy as np
import torch

from src.black_hole.config import BlackHoleDetectorConfig
from src.black_hole.detector import BlackHoleDetector
from src.black_hole.evaluation import evaluate_detector


def _write_split(path, labels, groups, offset):
    count = len(labels)
    np.savez_compressed(
        path,
        features=np.arange(count * 4, dtype=np.float32).reshape(count, 4) + offset,
        labels=np.asarray(labels, dtype=np.int64),
        row_ids=np.asarray([f"row-{offset}-{i}" for i in range(count)]),
        group_ids=np.asarray(groups),
    )


def _write_jsonl(path, labels, offset):
    rows = []
    for index, label in enumerate(labels):
        rows.append({
            "dataset_row_id": f"row-{offset}-{index}",
            "problem_id": f"q-{offset}-{index}",
            "label": "low_value_in_context" if label else "answer_sensitive_in_context",
            "source_state_references": [{"state_id": "S1"}],
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_evaluate_detector_uses_checkpoint_and_writes_validation_test_and_state_metrics(tmp_path):
    activation_dir = tmp_path / "activations"
    activation_dir.mkdir()
    _write_split(activation_dir / "train.npz", [0, 1, 0, 1], ["tr1", "tr1", "tr2", "tr2"], 0)
    _write_split(activation_dir / "validation.npz", [0, 1, 0, 1], ["va1", "va1", "va2", "va2"], 20)
    _write_split(activation_dir / "test.npz", [0, 1, 0, 1], ["te1", "te1", "te2", "te2"], 40)

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    _write_jsonl(split_dir / "validation.jsonl", [0, 1, 0, 1], 20)
    _write_jsonl(split_dir / "test.jsonl", [0, 1, 0, 1], 40)

    config = BlackHoleDetectorConfig(input_size=4, hidden_size=3, dropout=0.0)
    model = BlackHoleDetector(config)
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "detector_config": config.to_dict(),
        "feature_mean": torch.zeros(4),
        "feature_scale": torch.ones(4),
        "best_epoch": 3,
    }, checkpoint_path)

    result = evaluate_detector(
        activation_dir,
        checkpoint_path,
        split_dir,
        tmp_path / "out",
        device="cpu",
    )

    assert result["status"] == "completed"
    assert result["fit_or_threshold_selection_performed"] is False
    assert result["training_examples_used_for_fit"] is False
    assert result["validation"]["row_count"] == 4
    assert result["test"]["state_level"]["state_count"] == 4
    assert (tmp_path / "out" / "detector_test_metrics.json").is_file()
    assert (tmp_path / "out" / "baseline_comparison.json").is_file()

