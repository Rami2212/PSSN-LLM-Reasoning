import json

import numpy as np
import torch

from src.black_hole.config import BlackHoleDetectorConfig
from src.black_hole.detector import BlackHoleDetector
from src.black_hole.thresholding import _select_threshold, select_and_export


def _write_split(path, labels, groups, prefix):
    count = len(labels)
    np.savez_compressed(
        path,
        features=np.zeros((count, 4), dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        row_ids=np.asarray([f"{prefix}-{i}" for i in range(count)]),
        group_ids=np.asarray(groups),
    )


def _write_rows(path, labels, prefix):
    rows = [
        {
            "dataset_row_id": f"{prefix}-{i}",
            "problem_id": f"{prefix}-q{i // 2}",
            "label": "low_value_in_context" if label else "answer_sensitive_in_context",
            "source_state_references": [{"state_id": "S1"}],
        }
        for i, label in enumerate(labels)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_threshold_selection_obeys_risk_budget_and_falls_back_safely():
    metrics = {
        "0.30": {"state_level": {"classification": {
            "answer_sensitive_false_skip_rate": 0.25, "recall_low_value": 0.9, "f1_low_value": 0.8,
        }}},
        "0.50": {"state_level": {"classification": {
            "answer_sensitive_false_skip_rate": 0.05, "recall_low_value": 0.7, "f1_low_value": 0.75,
        }}},
        "0.80": {"state_level": {"classification": {
            "answer_sensitive_false_skip_rate": 0.02, "recall_low_value": 0.1, "f1_low_value": 0.2,
        }}},
    }
    assert _select_threshold(metrics, 0.10) == (
        0.5, True, "max_state_low_value_recall_under_false_skip_risk_budget"
    )
    assert _select_threshold(metrics, 0.01) == (
        0.8, False,
        "minimum_validation_state_false_skip_rate; risk budget not met, automated skipping disabled",
    )


def test_select_and_export_scores_test_only_after_validation_selection(tmp_path):
    activation_dir = tmp_path / "activations"
    activation_dir.mkdir()
    labels = [0, 1, 0, 1]
    _write_split(activation_dir / "train.npz", labels, ["tr-q0", "tr-q0", "tr-q1", "tr-q1"], "tr")
    _write_split(activation_dir / "validation.npz", labels, ["va-q0", "va-q0", "va-q1", "va-q1"], "va")
    _write_split(activation_dir / "test.npz", labels, ["te-q0", "te-q0", "te-q1", "te-q1"], "te")

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    _write_rows(split_dir / "validation.jsonl", labels, "va")
    _write_rows(split_dir / "test.jsonl", labels, "te")

    config = BlackHoleDetectorConfig(input_size=4, hidden_size=3, dropout=0.0)
    model = BlackHoleDetector(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint = tmp_path / "best_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "detector_config": config.to_dict(),
        "feature_mean": torch.zeros(4),
        "feature_scale": torch.ones(4),
        "best_epoch": 2,
    }, checkpoint)

    result = select_and_export(
        activation_dir,
        checkpoint,
        split_dir,
        tmp_path / "out",
        thresholds=(0.3, 0.8),
        max_false_skip_rate=0.1,
        device="cpu",
    )

    assert result["status"] == "completed"
    assert result["selected_threshold"] == 0.8
    assert result["risk_budget_met"] is True
    assert result["automatic_skipping_allowed"] is False
    assert result["test_used_for_selection"] is False
    for filename in result["exported_files"]:
        assert (tmp_path / "out" / filename).is_file()
    threshold_record = json.loads((tmp_path / "out" / "threshold.json").read_text(encoding="utf-8"))
    assert threshold_record["selection_split"] == "validation"
    assert threshold_record["automatic_skipping_allowed"] is False

