"""Evaluate a trained Week 4 Black Hole detector on held-out splits.

The detector checkpoint is loaded without fitting or threshold tuning. The
validation and test splits are reported separately; the checkpoint's stored
training-split normalization and fixed 0.5 probability threshold are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from .baselines import LABEL_NAMES, load_activation_splits
from .checkpointing import atomic_json_save
from .config import BlackHoleDetectorConfig
from .detector import BlackHoleDetector


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel().tolist()
    answer_sensitive_count = tn + fp
    low_value_count = fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / low_value_count if low_value_count else 0.0
    result: dict[str, Any] = {
        "row_count": int(len(labels)),
        "class_counts": {LABEL_NAMES[k]: int(np.sum(labels == k)) for k in (0, 1)},
        "predicted_low_value_count": int(np.sum(predictions == 1)),
        "predicted_low_value_fraction": float(np.mean(predictions == 1)),
        "accuracy": float(np.mean(predictions == labels)),
        "precision_low_value": float(precision),
        "recall_low_value": float(recall),
        "f1_low_value": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "confusion_matrix": {
            "labels": [LABEL_NAMES[0], LABEL_NAMES[1]],
            "rows_true_cols_pred": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
        "answer_sensitive_false_skip_rate": float(fp / answer_sensitive_count) if answer_sensitive_count else None,
        "low_value_miss_rate": float(fn / low_value_count) if low_value_count else None,
        "probability_threshold": float(threshold),
    }
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
        result["pr_auc_average_precision"] = float(average_precision_score(labels, probabilities))
    else:
        result["roc_auc"] = None
        result["pr_auc_average_precision"] = None
    return result


def _state_level_metrics(
    rows: list[dict[str, Any]], row_probabilities: dict[str, float], threshold: float
) -> dict[str, Any]:
    """Aggregate sentence predictions by source semantic state.

    A state's score is the mean probability of its linked sentence rows. Its
    gold label is conservative: it is answer-sensitive if any linked sentence
    is answer-sensitive. This avoids declaring a state safe to skip when any
    mapped part was labeled answer-sensitive.
    """
    collected: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"scores": [], "labels": []})
    for row in rows:
        row_id = row.get("dataset_row_id")
        if row_id not in row_probabilities:
            raise ValueError(f"No activation prediction for dataset row {row_id!r}")
        label = 1 if row.get("label") == LABEL_NAMES[1] else 0 if row.get("label") == LABEL_NAMES[0] else None
        if label is None:
            raise ValueError(f"Unknown label for dataset row {row_id!r}")
        refs = row.get("source_state_references")
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"Missing source_state_references for dataset row {row_id!r}")
        for ref in refs:
            state_id = ref.get("state_id") if isinstance(ref, dict) else None
            if not isinstance(state_id, str) or not state_id:
                raise ValueError(f"Invalid source state reference in dataset row {row_id!r}")
            key = (str(row.get("problem_id", "")), state_id)
            collected[key]["scores"].append(row_probabilities[row_id])
            collected[key]["labels"].append(label)

    gold, scores = [], []
    for item in collected.values():
        scores.append(float(np.mean(item["scores"])))
        gold.append(0 if 0 in item["labels"] else 1)
    return {
        "aggregation": "mean sentence low-value probability per (problem_id, state_id); state is answer-sensitive if any linked sentence is",
        "state_count": len(gold),
        "classification": _classification_metrics(np.asarray(gold, dtype=np.int64), np.asarray(scores), threshold),
    }


def _validate_baseline_counts(
    baseline_results: dict[str, Any], splits: dict[str, dict[str, np.ndarray]]
) -> None:
    baseline_splits = baseline_results.get("split_summary", {})
    for split in ("validation", "test"):
        expected = baseline_splits.get(split)
        if not isinstance(expected, dict):
            raise ValueError(f"Baseline results do not contain {split} split summary")
        data = splits[split]
        actual_counts = {LABEL_NAMES[k]: int(np.sum(data["labels"] == k)) for k in (0, 1)}
        if expected.get("row_count") != len(data["labels"]) or expected.get("class_counts") != actual_counts:
            raise ValueError(f"Baseline {split} split does not match activation split row/class counts")


def evaluate_detector(
    activation_dir: str | Path,
    checkpoint_path: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
    *,
    baseline_results_path: str | Path | None = None,
    device: str = "auto",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate validation/test without training or selecting a threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    activation_dir, checkpoint_path = Path(activation_dir), Path(checkpoint_path)
    split_dir, output_dir = Path(split_dir), Path(output_dir)
    if device == "auto":
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = torch.device(device)
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

    splits = load_activation_splits(activation_dir)
    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=False)
    required_checkpoint = {"model_state_dict", "detector_config", "feature_mean", "feature_scale"}
    missing = required_checkpoint - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    config = BlackHoleDetectorConfig.from_dict(checkpoint["detector_config"])
    model = BlackHoleDetector(config).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32, device=selected_device)
    scale = torch.as_tensor(checkpoint["feature_scale"], dtype=torch.float32, device=selected_device)
    if mean.shape != (config.input_size,) or scale.shape != (config.input_size,) or torch.any(scale <= 0):
        raise ValueError("Checkpoint normalization does not match detector input size")

    split_rows = {split: _read_jsonl(split_dir / f"{split}.jsonl") for split in ("validation", "test")}
    metrics: dict[str, Any] = {}
    row_probabilities: dict[str, dict[str, float]] = {}
    for split in ("validation", "test"):
        data = splits[split]
        rows = split_rows[split]
        ids = data["row_ids"].tolist()
        row_map = {row.get("dataset_row_id"): row for row in rows}
        if len(row_map) != len(rows) or set(row_map) != set(ids):
            raise ValueError(f"{split}: JSONL rows do not exactly match activation row IDs")
        for row_id, label in zip(ids, data["labels"], strict=True):
            expected = LABEL_NAMES[int(label)]
            if row_map[row_id].get("label") != expected:
                raise ValueError(f"{split}: label mismatch for {row_id}")
        features = torch.as_tensor(data["features"], dtype=torch.float32, device=selected_device)
        normalized = torch.nan_to_num((features - mean) / scale)
        with torch.inference_mode():
            probabilities = model.predict_proba(normalized).detach().cpu().numpy().astype(np.float64)
        metrics[split] = _classification_metrics(data["labels"], probabilities, threshold)
        row_probabilities[split] = dict(zip(ids, probabilities.tolist(), strict=True))
        metrics[split]["state_level"] = _state_level_metrics(rows, row_probabilities[split], threshold)

    baseline_results = None
    comparison: dict[str, Any] = {
        "detector": {split: metrics[split] for split in ("validation", "test")},
        "baselines": {},
        "comparison_note": "All reported classifiers use the same split counts. Baselines were fit on train; threshold tuning, if any, used validation only.",
    }
    if baseline_results_path is not None:
        baseline_results_path = Path(baseline_results_path)
        baseline_results = json.loads(baseline_results_path.read_text(encoding="utf-8"))
        if baseline_results.get("schema_version") != "black-hole-baseline-results-v1":
            raise ValueError("Unrecognized baseline results schema")
        _validate_baseline_counts(baseline_results, splits)
        for name, result in baseline_results.get("baselines", {}).items():
            comparison["baselines"][name] = {
                split: result.get(split) for split in ("validation", "test")
            }

    summary = {
        "schema_version": "black-hole-detector-evaluation-v1",
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "activation_files": {
            split: {"path": str(activation_dir / f"{split}.npz"), "sha256": _sha256(activation_dir / f"{split}.npz")}
            for split in ("validation", "test")
        },
        "split_files": {
            split: {"path": str(split_dir / f"{split}.jsonl"), "sha256": _sha256(split_dir / f"{split}.jsonl")}
            for split in ("validation", "test")
        },
        "device": str(selected_device),
        "threshold": float(threshold),
        "fit_or_threshold_selection_performed": False,
        "training_examples_used_for_fit": False,
        "validation": metrics["validation"],
        "test": metrics["test"],
        "baseline_results_path": str(baseline_results_path) if baseline_results_path else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_save(metrics["validation"], output_dir / "detector_validation_metrics.json")
    atomic_json_save(metrics["test"], output_dir / "detector_test_metrics.json")
    atomic_json_save(comparison, output_dir / "baseline_comparison.json")
    atomic_json_save(
        {split: metrics[split]["confusion_matrix"] for split in ("validation", "test")},
        output_dir / "confusion_matrix.json",
    )
    atomic_json_save(summary, output_dir / "evaluation_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", required=True, help="W4-001 directory with validation.npz and test.npz")
    parser.add_argument("--checkpoint", required=True, help="W4-004 best_model.pt")
    parser.add_argument("--split-dir", required=True, help="W3-006 directory with validation.jsonl and test.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for structured metrics")
    parser.add_argument("--baseline-results", help="Optional W4-002 baseline_results.json")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed probability threshold; do not tune on test")
    args = parser.parse_args()
    result = evaluate_detector(
        args.activation_dir,
        args.checkpoint,
        args.split_dir,
        args.output_dir,
        baseline_results_path=args.baseline_results,
        device=args.device,
        threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
