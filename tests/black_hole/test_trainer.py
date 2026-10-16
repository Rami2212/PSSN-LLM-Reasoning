import json

import numpy as np
import torch

from src.black_hole.trainer import train_detector


def _save_split(path, labels, groups, offset):
    count = len(labels)
    rng = np.random.default_rng(offset)
    features = rng.normal(size=(count, 12)).astype(np.float32)
    features[:, 0] += np.asarray(labels, dtype=np.float32) * 1.5
    np.savez_compressed(
        path,
        features=features,
        labels=np.asarray(labels, dtype=np.int64),
        row_ids=np.asarray([f"row-{offset}-{i}" for i in range(count)]),
        group_ids=np.asarray(groups),
    )


def _make_data(data_dir):
    data_dir.mkdir(parents=True)
    _save_split(data_dir / "train.npz", [0, 1] * 12, [f"train-q{i // 2}" for i in range(24)], 1)
    _save_split(data_dir / "validation.npz", [0, 1] * 5, [f"val-q{i // 2}" for i in range(10)], 2)
    # The training command must not need or inspect test.npz.
    (data_dir / "test.npz").write_text("deliberately not loaded", encoding="utf-8")


def test_trains_saves_history_and_leaves_test_split_untouched(tmp_path):
    data_dir = tmp_path / "activations"
    output_dir = tmp_path / "training"
    _make_data(data_dir)

    summary = train_detector(
        data_dir,
        output_dir,
        epochs=3,
        batch_size=8,
        patience=2,
        hidden_size=8,
        dropout=0,
        use_class_weights=True,
        device="cpu",
    )

    assert summary["status"] == "completed"
    assert summary["test_split_used"] is False
    assert summary["training_rows"] == 24
    assert summary["validation_rows"] == 10
    for filename in (
        "best_model.pt", "final_model.pt", "last_checkpoint.pt", "training_history.json",
        "training_config.json", "detector_config.json", "training_summary.json",
    ):
        assert (output_dir / filename).is_file()
    config = json.loads((output_dir / "training_config.json").read_text(encoding="utf-8"))
    assert config["device"] == "cpu"
    assert config["positive_class_weight"] == 1.0


def test_resume_continues_from_checkpoint(tmp_path):
    data_dir = tmp_path / "activations"
    output_dir = tmp_path / "training"
    _make_data(data_dir)
    common = dict(batch_size=8, patience=10, hidden_size=8, dropout=0, use_class_weights=False, device="cpu")
    first = train_detector(data_dir, output_dir, epochs=2, **common)
    assert first["final_epoch"] == 2

    resumed = train_detector(data_dir, output_dir, epochs=4, resume=True, **common)
    assert resumed["final_epoch"] == 4
    history = json.loads((output_dir / "training_history.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in history] == [1, 2, 3, 4]


def test_rejects_train_validation_question_leakage(tmp_path):
    data_dir = tmp_path / "activations"
    data_dir.mkdir()
    _save_split(data_dir / "train.npz", [0, 1], ["same-question", "train-q"], 3)
    _save_split(data_dir / "validation.npz", [0, 1], ["same-question", "val-q"], 4)
    try:
        train_detector(data_dir, tmp_path / "training", device="cpu", epochs=1)
    except ValueError as exc:
        assert "group leakage" in str(exc)
    else:
        raise AssertionError("Expected a question-level leakage error")
