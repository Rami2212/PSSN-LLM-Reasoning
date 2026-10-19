"""Select a conservative V2 predictor threshold from validation data only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import BlackHoleDetectorConfig
from .detector import BlackHoleDetector
from .sentence_evaluation_v2 import (
    ID_TO_LABEL,
    _classification_metrics,
    _load_test_arrays,
    _read_jsonl,
    _sha256,
)


def select_validation_threshold(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    split_jsonl: str | Path,
    output_path: str | Path,
    *,
    max_false_skip_rate: float = 0.10,
    device: str = "auto",
) -> dict[str, Any]:
    """Choose the lowest threshold satisfying a validation false-skip cap.

    The false-skip rate is the fraction of answer-sensitive validation rows
    classified as low-value. Choosing the lowest qualifying threshold keeps
    low-value recall as high as possible under the configured safety cap.
    """
    if not 0 <= max_false_skip_rate <= 1:
        raise ValueError("max_false_skip_rate must be between 0 and 1")
    data_dir = Path(data_dir)
    checkpoint_path = Path(checkpoint_path)
    split_jsonl = Path(split_jsonl)
    output_path = Path(output_path)
    activation_path = data_dir / "validation.npz"
    data = _load_test_arrays(activation_path)
    rows = _read_jsonl(split_jsonl)
    row_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("dataset_row_id")
        if not isinstance(row_id, str) or not row_id or row_id in row_map:
            raise ValueError(f"Missing or duplicate dataset_row_id in {split_jsonl}")
        row_map[row_id] = row
    if set(row_map) != set(data["row_ids"].tolist()):
        raise ValueError("Validation JSONL row IDs do not exactly match activation row IDs")
    for index, row_id in enumerate(data["row_ids"].tolist()):
        row = row_map[row_id]
        if row.get("label") != ID_TO_LABEL[int(data["labels"][index])]:
            raise ValueError(f"Validation label mismatch for row {row_id}")
        if row.get("problem_id") != data["group_ids"][index]:
            raise ValueError(f"Validation question/group mismatch for row {row_id}")
    if len(np.unique(data["labels"])) != 2:
        raise ValueError("Validation split must contain both classes")

    train_path = data_dir / "train.npz"
    train = _load_test_arrays(train_path)
    overlap = set(train["group_ids"].tolist()) & set(data["group_ids"].tolist())
    if overlap:
        raise ValueError(f"Question/group leakage between train and validation: {sorted(overlap)[:5]}")

    if device == "auto":
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = torch.device(device)
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=False)
    required = {"model_state_dict", "detector_config", "feature_mean", "feature_scale", "input_hashes"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    for split in ("train", "validation"):
        actual_hash = _sha256(data_dir / f"{split}.npz")
        if checkpoint["input_hashes"].get(split) != actual_hash:
            raise ValueError(f"{split} activation hash does not match the trained checkpoint")

    config = BlackHoleDetectorConfig.from_dict(checkpoint["detector_config"])
    if data["features"].shape[1] != config.input_size:
        raise ValueError("Validation feature width does not match model input")
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

    # Each unique predicted probability changes the predicted set. The value
    # just above the maximum represents the valid all-negative option, even if
    # floating-point sigmoid output saturated at exactly 1.0.
    candidates = sorted(set(float(value) for value in probabilities))
    candidates.append(float(np.nextafter(np.max(probabilities), np.inf)))
    selected: tuple[float, dict[str, Any]] | None = None
    for threshold in candidates:
        metrics = _classification_metrics(data["labels"], probabilities, threshold)
        rate = metrics["answer_sensitive_false_skip_rate"]
        if rate is not None and rate <= max_false_skip_rate:
            selected = threshold, metrics
            break
    if selected is None:  # Defensive fallback; the all-negative candidate should qualify.
        raise RuntimeError("Could not find a threshold satisfying the validation false-skip cap")
    threshold, selected_metrics = selected
    summary = {
        "schema_version": "pssn-sentence-validation-threshold-v1",
        "status": "completed",
        "selection_split": "validation",
        "selection_rule": "lowest threshold whose validation answer-sensitive false-skip rate is at or below the configured maximum",
        "max_validation_false_skip_rate": float(max_false_skip_rate),
        "selected_threshold": float(threshold),
        "validation_metrics_at_selected_threshold": selected_metrics,
        "validation_metrics_at_default_0_5": _classification_metrics(data["labels"], probabilities, 0.5),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "best_epoch": checkpoint.get("best_epoch"),
        "validation_activation_sha256": _sha256(activation_path),
        "train_activation_sha256": _sha256(train_path),
        "validation_split_sha256": _sha256(split_jsonl),
        "device": str(selected_device),
        "test_data_used": False,
        "note": "This threshold is selected on validation only. Test data must be evaluated once after this file is frozen.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Feature directory containing validation.npz")
    parser.add_argument("--checkpoint", required=True, help="Trained best_model.pt checkpoint")
    parser.add_argument("--split-jsonl", required=True, help="Matching question-held-out validation.jsonl")
    parser.add_argument("--output", required=True, help="Path for validation threshold_selection.json")
    parser.add_argument(
        "--max-false-skip-rate", type=float, default=0.10,
        help="Maximum allowed fraction of answer-sensitive validation rows flagged low-value (default: 0.10)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()
    result = select_validation_threshold(
        args.data_dir,
        args.checkpoint,
        args.split_jsonl,
        args.output,
        max_false_skip_rate=args.max_false_skip_rate,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
