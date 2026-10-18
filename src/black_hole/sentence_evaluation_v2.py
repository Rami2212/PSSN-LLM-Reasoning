"""Evaluate a trained V2 sentence predictor on the untouched question-held-out test split.

The evaluator uses the checkpoint's training-only normalization and a fixed
0.5 threshold. It does not fit, tune, or update the predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from .config import BlackHoleDetectorConfig
from .detector import BlackHoleDetector

LABEL_TO_ID = {"answer_sensitive_in_context": 0, "low_value_in_context": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
THRESHOLD = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _load_test_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing test activation file: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "labels", "row_ids", "group_ids"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        result = {key: archive[key] for key in required}
    result["features"] = np.asarray(result["features"], dtype=np.float32)
    result["labels"] = np.asarray(result["labels"], dtype=np.int64)
    result["row_ids"] = np.asarray(result["row_ids"]).astype(str)
    result["group_ids"] = np.asarray(result["group_ids"]).astype(str)
    count = len(result["labels"])
    if result["features"].ndim != 2 or not count:
        raise ValueError("Test features must be a nonempty 2-D matrix")
    if not np.isfinite(result["features"]).all():
        raise ValueError("Test features contain NaN or infinity")
    if not np.isin(result["labels"], [0, 1]).all():
        raise ValueError("Test labels must be binary 0/1")
    if any(len(result[key]) != count for key in ("row_ids", "group_ids")):
        raise ValueError("Test activation arrays have inconsistent row counts")
    if len(set(result["row_ids"].tolist())) != count:
        raise ValueError("Test activation file contains duplicate row IDs")
    return result


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel().tolist()
    low_precision = tp / (tp + fp) if tp + fp else 0.0
    low_recall = tp / (tp + fn) if tp + fn else 0.0
    sensitive_precision = tn / (tn + fn) if tn + fn else 0.0
    sensitive_recall = tn / (tn + fp) if tn + fp else 0.0
    result: dict[str, Any] = {
        "row_count": int(len(labels)),
        "class_counts": {
            ID_TO_LABEL[label]: int(np.sum(labels == label)) for label in (0, 1)
        },
        "predicted_low_value_count": int(np.sum(predictions == 1)),
        "accuracy": float(np.mean(predictions == labels)),
        "precision_low_value": float(low_precision),
        "recall_low_value": float(low_recall),
        "f1_low_value": float(
            2 * low_precision * low_recall / (low_precision + low_recall)
        ) if low_precision + low_recall else 0.0,
        "precision_answer_sensitive": float(sensitive_precision),
        "recall_answer_sensitive": float(sensitive_recall),
        "answer_sensitive_false_skip_rate": float(fp / (tn + fp)) if tn + fp else None,
        "low_value_miss_rate": float(fn / (fn + tp)) if fn + tp else None,
        "confusion_matrix": {
            "labels": [ID_TO_LABEL[0], ID_TO_LABEL[1]],
            "rows_true_cols_pred": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
        "probability_threshold": THRESHOLD,
    }
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
        result["pr_auc_average_precision"] = float(
            average_precision_score(labels, probabilities)
        )
    else:
        result["roc_auc"] = None
        result["pr_auc_average_precision"] = None
    return result


def evaluate_test_split(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    split_jsonl: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    """Score only test examples and save metrics plus row-level predictions."""
    data_dir = Path(data_dir)
    checkpoint_path = Path(checkpoint_path)
    split_jsonl = Path(split_jsonl)
    output_dir = Path(output_dir)
    activation_path = data_dir / "test.npz"
    data = _load_test_arrays(activation_path)
    rows = _read_jsonl(split_jsonl)
    row_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("dataset_row_id")
        if not isinstance(row_id, str) or not row_id or row_id in row_map:
            raise ValueError(f"Missing or duplicate dataset_row_id in {split_jsonl}")
        row_map[row_id] = row
    if set(row_map) != set(data["row_ids"].tolist()):
        raise ValueError("Test JSONL row IDs do not exactly match test activation row IDs")

    for index, row_id in enumerate(data["row_ids"].tolist()):
        row = row_map[row_id]
        expected_label = ID_TO_LABEL[int(data["labels"][index])]
        if row.get("label") != expected_label:
            raise ValueError(f"Test label mismatch for row {row_id}")
        if row.get("problem_id") != data["group_ids"][index]:
            raise ValueError(f"Test question/group mismatch for row {row_id}")

    split_membership: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        split_path = data_dir / f"{split}.npz"
        split_data = _load_test_arrays(split_path)
        for group_id in set(split_data["group_ids"].tolist()):
            previous = split_membership.setdefault(group_id, split)
            if previous != split:
                raise ValueError(f"Question/group leakage across splits: {group_id}")

    if device == "auto":
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = torch.device(device)
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=False)
    required_checkpoint = {
        "model_state_dict", "detector_config", "feature_mean", "feature_scale", "input_hashes"
    }
    missing = required_checkpoint - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")

    for split in ("train", "validation"):
        path = data_dir / f"{split}.npz"
        expected_hash = checkpoint["input_hashes"].get(split)
        actual_hash = _sha256(path)
        if expected_hash != actual_hash:
            raise ValueError(f"{split} activation hash does not match the trained checkpoint")

    config = BlackHoleDetectorConfig.from_dict(checkpoint["detector_config"])
    if data["features"].shape[1] != config.input_size:
        raise ValueError(
            f"Test feature width {data['features'].shape[1]} does not match model input {config.input_size}"
        )
    model = BlackHoleDetector(config).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32, device=selected_device)
    scale = torch.as_tensor(checkpoint["feature_scale"], dtype=torch.float32, device=selected_device)
    if mean.shape != (config.input_size,) or scale.shape != (config.input_size,) or torch.any(scale <= 0):
        raise ValueError("Checkpoint normalization does not match the model input")

    features = torch.as_tensor(data["features"], dtype=torch.float32, device=selected_device)
    normalized = torch.nan_to_num((features - mean) / scale)
    with torch.inference_mode():
        probabilities = model.predict_proba(normalized).detach().cpu().numpy().astype(np.float64)
    metrics = _classification_metrics(data["labels"], probabilities)
    predictions = (probabilities >= THRESHOLD).astype(np.int64)
    prediction_rows = []
    for index, row_id in enumerate(data["row_ids"].tolist()):
        row = row_map[row_id]
        gold = int(data["labels"][index])
        predicted = int(predictions[index])
        prediction_rows.append({
            "dataset_row_id": row_id,
            "problem_id": str(row["problem_id"]),
            "trace_id": row.get("trace_id"),
            "sentence_id": row.get("sentence_id"),
            "gold_label": ID_TO_LABEL[gold],
            "predicted_label": ID_TO_LABEL[predicted],
            "probability_low_value": float(probabilities[index]),
            "is_correct": bool(gold == predicted),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "test_predictions.jsonl"
    temporary_path = output_dir / "test_predictions.jsonl.tmp"
    temporary_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    os.replace(temporary_path, predictions_path)
    summary = {
        "schema_version": "pssn-sentence-activation-test-evaluation-v1",
        "status": "completed",
        "evaluation_split": "test",
        "threshold": THRESHOLD,
        "threshold_tuned_on_test": False,
        "model_selection_used_test": False,
        "feature_directory": str(data_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "feature_width": int(data["features"].shape[1]),
        "device": str(selected_device),
        "test_activation_sha256": _sha256(activation_path),
        "test_split_sha256": _sha256(split_jsonl),
        "test_predictions": str(predictions_path.resolve()),
        "metrics": metrics,
        "interpretation_note": (
            "This is question-held-out evaluation within the supplied dataset, not an external dataset test. "
            "Labels are transferred dataset labels and are not independently validated here."
        ),
    }
    summary_path = output_dir / "test_evaluation_summary.json"
    temporary_summary = output_dir / "test_evaluation_summary.json.tmp"
    temporary_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_summary, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Feature directory containing test.npz")
    parser.add_argument("--checkpoint", required=True, help="Trained best_model.pt checkpoint")
    parser.add_argument("--split-jsonl", required=True, help="Matching question-held-out test.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for test metrics and predictions")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()
    result = evaluate_test_split(
        args.data_dir, args.checkpoint, args.split_jsonl, args.output_dir, device=args.device
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
