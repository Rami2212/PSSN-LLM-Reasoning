"""Select a conservative Week 4 detector threshold and export its artifacts.

Threshold selection is performed on validation data only. Test data is scored
once, after selection, and is never used to alter the threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import LABEL_NAMES, load_activation_splits
from .checkpointing import atomic_json_save, atomic_torch_save
from .config import BlackHoleDetectorConfig
from .detector import BlackHoleDetector
from .evaluation import _classification_metrics, _read_jsonl, _state_level_metrics


DEFAULT_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


def _state_scores(rows: list[dict[str, Any]], row_probabilities: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Map sentence probabilities to conservative (problem, state) examples."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("dataset_row_id")
        if row_id not in row_probabilities:
            raise ValueError(f"No activation prediction for dataset row {row_id!r}")
        if row.get("label") == LABEL_NAMES[1]:
            label = 1
        elif row.get("label") == LABEL_NAMES[0]:
            label = 0
        else:
            raise ValueError(f"Unknown label for dataset row {row_id!r}")
        refs = row.get("source_state_references")
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"Missing source_state_references for dataset row {row_id!r}")
        for ref in refs:
            state_id = ref.get("state_id") if isinstance(ref, dict) else None
            if not isinstance(state_id, str) or not state_id:
                raise ValueError(f"Invalid source state reference in dataset row {row_id!r}")
            key = (str(row.get("problem_id", "")), state_id)
            item = grouped.setdefault(key, {"probabilities": [], "labels": []})
            item["probabilities"].append(row_probabilities[row_id])
            item["labels"].append(label)
    labels = np.asarray([0 if 0 in value["labels"] else 1 for value in grouped.values()], dtype=np.int64)
    probabilities = np.asarray([np.mean(value["probabilities"]) for value in grouped.values()], dtype=np.float64)
    return labels, probabilities


def _predict(
    model: BlackHoleDetector,
    data: dict[str, np.ndarray],
    mean: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    features = torch.as_tensor(data["features"], dtype=torch.float32, device=device)
    normalized = torch.nan_to_num((features - mean) / scale)
    model.eval()
    with torch.inference_mode():
        return model.predict_proba(normalized).detach().cpu().numpy().astype(np.float64)


def _score_split(
    labels: np.ndarray,
    probabilities: np.ndarray,
    rows: list[dict[str, Any]],
    row_ids: list[str],
    threshold: float,
) -> dict[str, Any]:
    by_id = dict(zip(row_ids, probabilities.tolist(), strict=True))
    state_labels, state_probabilities = _state_scores(rows, by_id)
    return {
        "sentence_level": _classification_metrics(labels, probabilities, threshold),
        "state_level": {
            "aggregation": "mean sentence low-value probability per (problem_id, state_id); state is answer-sensitive if any linked sentence is",
            "state_count": int(len(state_labels)),
            "classification": _classification_metrics(state_labels, state_probabilities, threshold),
        },
    }


def _select_threshold(
    threshold_metrics: dict[str, dict[str, Any]], max_false_skip_rate: float
) -> tuple[float, bool, str]:
    """Maximize state low-value recall subject to a false-skip risk budget.

    If no candidate meets the budget, return the minimum-risk candidate but
    mark automated skipping disabled. Ties prefer lower risk, higher F1, then
    the more conservative (higher) threshold.
    """
    candidates = []
    for key, result in threshold_metrics.items():
        state = result["state_level"]["classification"]
        risk = state["answer_sensitive_false_skip_rate"]
        # No answer-sensitive examples means risk cannot be estimated; treat
        # it as unsafe rather than granting a zero-risk pass.
        risk = float(risk) if risk is not None else 1.0
        candidates.append((float(key), risk, float(state["recall_low_value"]), float(state["f1_low_value"])))
    feasible = [item for item in candidates if item[1] <= max_false_skip_rate]
    if feasible:
        selected = max(feasible, key=lambda item: (item[2], -item[1], item[3], item[0]))
        return selected[0], True, "max_state_low_value_recall_under_false_skip_risk_budget"
    selected = max(candidates, key=lambda item: (-item[1], item[2], item[3], item[0]))
    return selected[0], False, "minimum_validation_state_false_skip_rate; risk budget not met, automated skipping disabled"


def select_and_export(
    activation_dir: str | Path,
    checkpoint_path: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
    *,
    baseline_comparison_path: str | Path | None = None,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    max_false_skip_rate: float = 0.10,
    device: str = "auto",
) -> dict[str, Any]:
    """Select threshold on validation, score test once, and export artifacts."""
    if not 0.0 <= max_false_skip_rate <= 1.0:
        raise ValueError("max_false_skip_rate must be between 0 and 1")
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("thresholds must be a nonempty sequence between 0 and 1")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("thresholds must not contain duplicates")

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
    required = {"model_state_dict", "detector_config", "feature_mean", "feature_scale"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    config = BlackHoleDetectorConfig.from_dict(checkpoint["detector_config"])
    model = BlackHoleDetector(config).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32, device=selected_device)
    scale = torch.as_tensor(checkpoint["feature_scale"], dtype=torch.float32, device=selected_device)
    if mean.shape != (config.input_size,) or scale.shape != (config.input_size,) or torch.any(scale <= 0):
        raise ValueError("Checkpoint normalization does not match detector input size")

    def load_and_predict(split: str) -> tuple[list[dict[str, Any]], list[str], np.ndarray]:
        data = splits[split]
        rows = _read_jsonl(split_dir / f"{split}.jsonl")
        ids = data["row_ids"].tolist()
        row_map = {row.get("dataset_row_id"): row for row in rows}
        if len(row_map) != len(rows) or set(row_map) != set(ids):
            raise ValueError(f"{split}: JSONL rows do not exactly match activation row IDs")
        for row_id, group_id, label in zip(ids, data["group_ids"].astype(str), data["labels"], strict=True):
            if str(row_map[row_id].get("problem_id")) != group_id:
                raise ValueError(f"{split}: problem/group mismatch for {row_id}")
            if row_map[row_id].get("label") != LABEL_NAMES[int(label)]:
                raise ValueError(f"{split}: label mismatch for {row_id}")
        return rows, ids, _predict(model, data, mean, scale, selected_device)

    # Do not score or inspect test predictions while selecting the threshold.
    validation_rows, validation_ids, validation_probabilities = load_and_predict("validation")

    validation_thresholds = {}
    for threshold in sorted(thresholds):
        validation_thresholds[f"{threshold:.8g}"] = _score_split(
            splits["validation"]["labels"], validation_probabilities, validation_rows,
            validation_ids, threshold,
        )
    chosen, risk_budget_met, selection_rule = _select_threshold(
        validation_thresholds, max_false_skip_rate
    )
    selected_validation = validation_thresholds[f"{chosen:.8g}"]
    has_validation_skip_predictions = (
        selected_validation["state_level"]["classification"]["predicted_low_value_count"] > 0
    )
    automatic_skipping_allowed = risk_budget_met and has_validation_skip_predictions

    # Only now score the held-out test data, once, at the fixed validation threshold.
    test_rows, test_ids, test_probabilities = load_and_predict("test")
    test_metrics = _score_split(
        splits["test"]["labels"], test_probabilities, test_rows, test_ids, chosen,
    )

    baseline_comparison: dict[str, Any] = {
        "schema_version": "black-hole-selected-threshold-comparison-v1",
        "detector": {"validation": selected_validation, "test": test_metrics},
        "baselines": {},
        "threshold": chosen,
        "baseline_comparison_source": str(baseline_comparison_path) if baseline_comparison_path else None,
        "note": "Threshold selected using validation only; test metrics were computed after selection.",
    }
    if baseline_comparison_path is not None:
        with Path(baseline_comparison_path).open(encoding="utf-8") as stream:
            previous = json.load(stream)
        baseline_comparison["baselines"] = previous.get("baselines", {})

    threshold_record = {
        "schema_version": "black-hole-threshold-v1",
        "selected_threshold": chosen,
        "selection_split": "validation",
        "selection_unit": "semantic_state",
        "selection_rule": selection_rule,
        "max_validation_answer_sensitive_false_skip_rate": max_false_skip_rate,
        "risk_budget_met": risk_budget_met,
        "automatic_skipping_allowed": automatic_skipping_allowed,
        "validation_has_state_skip_predictions": has_validation_skip_predictions,
        "validation_state_answer_sensitive_false_skip_rate": selected_validation["state_level"]["classification"]["answer_sensitive_false_skip_rate"],
        "validation_state_low_value_recall": selected_validation["state_level"]["classification"]["recall_low_value"],
        "candidate_thresholds": list(sorted(thresholds)),
        "test_used_for_selection": False,
    }

    report = _build_report(
        chosen, risk_budget_met, max_false_skip_rate, selected_validation, test_metrics,
        validation_thresholds, baseline_comparison.get("baselines", {}), automatic_skipping_allowed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({
        "model_state_dict": {key: value.detach().cpu() for key, value in checkpoint["model_state_dict"].items()},
        "detector_config": config.to_dict(),
        "feature_mean": torch.as_tensor(checkpoint["feature_mean"]).cpu(),
        "feature_scale": torch.as_tensor(checkpoint["feature_scale"]).cpu(),
        "selected_threshold": chosen,
        "automatic_skipping_allowed": automatic_skipping_allowed,
        "source_checkpoint_sha256": _file_hash(checkpoint_path),
    }, output_dir / "detector.pt")
    atomic_json_save(config.to_dict(), output_dir / "config.json")
    atomic_json_save(threshold_record, output_dir / "threshold.json")
    atomic_json_save({
        "schema_version": "black-hole-threshold-analysis-v1",
        "selection_split": "validation",
        "selection_unit": "semantic_state",
        "risk_budget": max_false_skip_rate,
        "candidates": validation_thresholds,
        "selected_threshold": chosen,
    }, output_dir / "threshold_analysis.json")
    atomic_json_save(baseline_comparison, output_dir / "baseline_comparison.json")
    atomic_json_save(selected_validation, output_dir / "detector_validation_metrics.json")
    atomic_json_save(test_metrics, output_dir / "detector_test_metrics.json")
    (output_dir / "week4_black_hole_detector_report.md").write_text(report, encoding="utf-8")
    summary = {
        "schema_version": "black-hole-threshold-export-v1",
        "status": "completed",
        "output_dir": str(output_dir),
        "selected_threshold": chosen,
        "risk_budget_met": risk_budget_met,
        "automatic_skipping_allowed": automatic_skipping_allowed,
        "validation_state_false_skip_rate": threshold_record["validation_state_answer_sensitive_false_skip_rate"],
        "validation_state_low_value_recall": threshold_record["validation_state_low_value_recall"],
        "test_state_false_skip_rate": test_metrics["state_level"]["classification"]["answer_sensitive_false_skip_rate"],
        "test_state_low_value_recall": test_metrics["state_level"]["classification"]["recall_low_value"],
        "test_used_for_selection": False,
        "exported_files": [
            "detector.pt", "config.json", "threshold.json", "threshold_analysis.json",
            "baseline_comparison.json", "detector_validation_metrics.json",
            "detector_test_metrics.json", "week4_black_hole_detector_report.md",
        ],
    }
    atomic_json_save(summary, output_dir / "export_summary.json")
    return summary


def _file_hash(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_report(
    threshold: float,
    risk_budget_met: bool,
    risk_budget: float,
    validation: dict[str, Any],
    test: dict[str, Any],
    candidates: dict[str, Any],
    baselines: dict[str, Any],
    automatic_skipping_allowed: bool,
) -> str:
    val_state = validation["state_level"]["classification"]
    test_state = test["state_level"]["classification"]
    val_sentence = validation["sentence_level"]
    test_sentence = test["sentence_level"]
    lines = [
        "# Week 4 Black Hole Detector Report",
        "",
        "## Decision",
        "",
        f"Selected threshold: **{threshold:.2f}**, chosen on validation data only using state-level aggregation.",
        f"Validation false-skip risk budget ({risk_budget:.1%}) met: **{risk_budget_met}**.",
        f"Automatic state skipping enabled by the selected policy: **{automatic_skipping_allowed}**. This is a validation policy, not a safety guarantee.",
        "If the selected validation threshold predicts no state to skip, skipping remains disabled even when the risk budget is met.",
        "",
        "## Held-out results",
        "",
        "| Split/unit | Rows/states | Accuracy | Low-value F1 | Answer-sensitive false-skip rate | Low-value recall |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Validation sentences | {val_sentence['row_count']} | {val_sentence['accuracy']:.3f} | {val_sentence['f1_low_value']:.3f} | {_fmt(val_sentence['answer_sensitive_false_skip_rate'])} | {val_sentence['recall_low_value']:.3f} |",
        f"| Validation states | {validation['state_level']['state_count']} | {val_state['accuracy']:.3f} | {val_state['f1_low_value']:.3f} | {_fmt(val_state['answer_sensitive_false_skip_rate'])} | {val_state['recall_low_value']:.3f} |",
        f"| Test sentences | {test_sentence['row_count']} | {test_sentence['accuracy']:.3f} | {test_sentence['f1_low_value']:.3f} | {_fmt(test_sentence['answer_sensitive_false_skip_rate'])} | {test_sentence['recall_low_value']:.3f} |",
        f"| Test states | {test['state_level']['state_count']} | {test_state['accuracy']:.3f} | {test_state['f1_low_value']:.3f} | {_fmt(test_state['answer_sensitive_false_skip_rate'])} | {test_state['recall_low_value']:.3f} |",
        "",
        "A false-skip rate is the fraction of answer-sensitive items predicted low-value. These are label predictions, not measured changes in final-answer accuracy or end-to-end speed.",
        "",
        "## Threshold selection",
        "",
        "Candidates are ranked by state-level low-value recall subject to the configured validation false-skip budget. If no threshold satisfies the budget, the minimum-risk threshold is recorded and automatic skipping remains disabled.",
        "",
        "| Threshold | Validation state false-skip | Validation state low-value recall | Validation state F1 |",
        "|---:|---:|---:|---:|",
    ]
    for key, result in candidates.items():
        state = result["state_level"]["classification"]
        lines.append(f"| {float(key):.2f} | {_fmt(state['answer_sensitive_false_skip_rate'])} | {state['recall_low_value']:.3f} | {state['f1_low_value']:.3f} |")
    lines.extend(["", "## Baseline comparison", ""])
    if baselines:
        lines.extend([
            "Test-set baseline values are from W4-002 and are included for context; compare against the detector at the selected threshold.",
            "",
            "| Model | Test accuracy | Low-value F1 | Answer-sensitive false-skip rate |",
            "|---|---:|---:|---:|",
            f"| Black Hole detector | {test_sentence['accuracy']:.3f} | {test_sentence['f1_low_value']:.3f} | {_fmt(test_sentence['answer_sensitive_false_skip_rate'])} |",
        ])
        for name, scores in baselines.items():
            test_scores = scores.get("test") or {}
            if "accuracy" in test_scores:
                lines.append(f"| {name} | {test_scores['accuracy']:.3f} | {test_scores['f1_low_value']:.3f} | {_fmt(test_scores.get('answer_sensitive_false_skip_rate'))} |")
    else:
        lines.append("No W4-002 baseline comparison file was supplied.")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- The data contains a small number of independent questions; sentence/state rows are clustered by question.",
        "- State aggregation averages sentence probabilities and labels a state answer-sensitive if any linked sentence is answer-sensitive.",
        "- A low-value label means removal was supported in the recorded trace context; it does not prove hidden computation was erased.",
        "- Threshold selection cannot establish deployment safety; Week 5 must measure final-answer accuracy under actual skipping.",
        "",
    ])
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", required=True, help="W4-001 directory with train/validation/test NPZs")
    parser.add_argument("--checkpoint", required=True, help="W4-004 best_model.pt")
    parser.add_argument("--split-dir", required=True, help="W3-006 directory with validation/test JSONL files")
    parser.add_argument("--output-dir", required=True, help="Week 4 export/evaluation directory")
    parser.add_argument("--baseline-comparison", help="Optional W4-005 baseline_comparison.json")
    parser.add_argument("--thresholds", default=",".join(f"{value:.2f}" for value in DEFAULT_THRESHOLDS))
    parser.add_argument("--max-false-skip-rate", type=float, default=0.10)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()
    thresholds = tuple(float(value.strip()) for value in args.thresholds.split(",") if value.strip())
    result = select_and_export(
        args.activation_dir,
        args.checkpoint,
        args.split_dir,
        args.output_dir,
        baseline_comparison_path=args.baseline_comparison,
        thresholds=thresholds,
        max_false_skip_rate=args.max_false_skip_rate,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
