"""Leakage-aware Week 4 baseline classifiers for Black Hole labels.

The positive class is ``low_value_in_context`` (1). Baselines fit only on the
training split; validation is used to select the sentence-length heuristic
threshold, and test is held for final reporting.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SPLITS = ("train", "validation", "test")
LABEL_NAMES = {0: "answer_sensitive_in_context", 1: "low_value_in_context"}
SEED = 42


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_activation_splits(data_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Read W4-001 NPZ files and reject malformed or leaking question splits."""
    data_dir = Path(data_dir)
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        path = data_dir / f"{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} activation file: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {"features", "labels", "row_ids", "group_ids"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            data = {key: archive[key] for key in archive.files}
        features = np.asarray(data["features"])
        labels = np.asarray(data["labels"], dtype=np.int64)
        row_ids = np.asarray(data["row_ids"]).astype(str)
        group_ids = np.asarray(data["group_ids"]).astype(str)
        if features.ndim != 2 or not len(features):
            raise ValueError(f"{path}: features must be a nonempty 2-D array")
        if not np.isfinite(features).all():
            raise ValueError(f"{path}: features contain NaN or infinity")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{path}: labels must be 0 or 1")
        if not (len(features) == len(labels) == len(row_ids) == len(group_ids)):
            raise ValueError(f"{path}: features, labels, row_ids, and group_ids lengths differ")
        if len(set(row_ids.tolist())) != len(row_ids):
            raise ValueError(f"{path}: duplicate row_ids")
        data.update(features=features.astype(np.float32, copy=False), labels=labels,
                    row_ids=row_ids, group_ids=group_ids)
        loaded[split] = data

    feature_widths = {loaded[split]["features"].shape[1] for split in SPLITS}
    if len(feature_widths) != 1:
        raise ValueError(f"Sentence feature widths differ across splits: {feature_widths}")

    seen_rows: set[str] = set()
    seen_groups: set[str] = set()
    for split in SPLITS:
        rows = set(loaded[split]["row_ids"].tolist())
        groups = set(loaded[split]["group_ids"].tolist())
        if seen_rows & rows:
            raise ValueError(f"Duplicate dataset rows across splits: {len(seen_rows & rows)}")
        if seen_groups & groups:
            raise ValueError(f"Question/group leakage across splits: {sorted(seen_groups & groups)[:5]}")
        seen_rows.update(rows)
        seen_groups.update(groups)

    has_state_features = ["state_features" in loaded[split] for split in SPLITS]
    if any(has_state_features) and not all(has_state_features):
        raise ValueError("state_features must be present in all three splits or none")
    if all(has_state_features):
        state_widths = set()
        for split in SPLITS:
            state_features = np.asarray(loaded[split]["state_features"])
            if state_features.ndim != 2 or len(state_features) != len(loaded[split]["labels"]):
                raise ValueError(f"{split}: state_features shape does not match labels")
            if not np.isfinite(state_features).all():
                raise ValueError(f"{split}: state_features contain NaN or infinity")
            loaded[split]["state_features"] = state_features.astype(np.float32, copy=False)
            state_widths.add(state_features.shape[1])
        if len(state_widths) != 1:
            raise ValueError(f"State feature widths differ across splits: {state_widths}")
    return loaded


def load_sentence_lengths(dataset_path: str | Path) -> dict[str, int]:
    """Read sentence text keyed by row ID; whitespace words are a tokenizer-free proxy."""
    lengths: dict[str, int] = {}
    with Path(dataset_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {dataset_path}:{line_number}") from exc
            row_id = row.get("dataset_row_id")
            sentence = row.get("sentence_text")
            if not row_id or not isinstance(sentence, str):
                raise ValueError(f"Missing dataset_row_id or sentence_text at line {line_number}")
            if row_id in lengths:
                raise ValueError(f"Duplicate dataset_row_id in raw dataset: {row_id}")
            lengths[row_id] = len(re.findall(r"\S+", sentence))
    return lengths


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    result: dict[str, Any] = {
        "row_count": int(len(y_true)),
        "class_counts": {LABEL_NAMES[label]: int(np.sum(y_true == label)) for label in (0, 1)},
        "predicted_low_value_count": int(np.sum(y_pred == 1)),
        "predicted_low_value_fraction": float(np.mean(y_pred == 1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_low_value": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_low_value": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_low_value": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": {"labels": [LABEL_NAMES[0], LABEL_NAMES[1]], "rows_true_cols_pred": [[tn, fp], [fn, tp]]},
        "answer_sensitive_false_skip_rate": float(fp / (tn + fp)) if tn + fp else None,
        "low_value_miss_rate": float(fn / (fn + tp)) if fn + tp else None,
    }
    if probabilities is not None:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None
        result["pr_auc_average_precision"] = (
            float(average_precision_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None
        )
    return result


def _fit_eval_model(
    model: Any,
    splits: dict[str, dict[str, np.ndarray]],
    *,
    feature_key: str = "features",
) -> dict[str, Any]:
    model.fit(splits["train"][feature_key], splits["train"]["labels"])
    scores = {}
    for split in ("validation", "test"):
        data = splits[split]
        predictions = model.predict(data[feature_key]).astype(np.int64)
        probabilities = model.predict_proba(data[feature_key])[:, 1] if hasattr(model, "predict_proba") else None
        scores[split] = _metrics(data["labels"], predictions, probabilities)
    return scores


def _select_length_threshold(
    splits: dict[str, dict[str, np.ndarray]], lengths_by_row: dict[str, int]
) -> tuple[int, dict[str, Any]]:
    results = {}
    validation = splits["validation"]
    validation_lengths = np.asarray([lengths_by_row[row_id] for row_id in validation["row_ids"]])
    y = validation["labels"]
    candidates = sorted(set(validation_lengths.tolist()))
    for threshold in candidates:
        pred = (validation_lengths <= threshold).astype(np.int64)
        results[threshold] = _metrics(y, pred)
    # Maximize validation F1; break ties toward greater precision, then fewer skips.
    best = max(candidates, key=lambda t: (
        results[t]["f1_low_value"],
        results[t]["precision_low_value"],
        -results[t]["predicted_low_value_count"],
    ))
    return int(best), {str(k): v for k, v in results.items()}


def run_baselines(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    raw_dataset: str | Path | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    """Fit baselines on train, report validation/test metrics, and save JSON."""
    splits = load_activation_splits(data_dir)
    train_y = splits["train"]["labels"]
    baselines: dict[str, Any] = {
        "majority_class": _fit_eval_model(DummyClassifier(strategy="most_frequent"), splits),
        "logistic_regression_sentence_activations": _fit_eval_model(
            make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            ),
            splits,
        ),
        "random_forest_sentence_activations": _fit_eval_model(
            RandomForestClassifier(
                n_estimators=150,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
            splits,
        ),
    }

    state_available = "state_features" in splits["train"]
    if state_available:
        baselines["logistic_regression_state_vectors"] = _fit_eval_model(
            make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            ),
            splits,
            feature_key="state_features",
        )

    heuristic: dict[str, Any] | None = None
    if raw_dataset is not None:
        lengths_by_row = load_sentence_lengths(raw_dataset)
        missing_ids = sorted({row_id for split in SPLITS for row_id in splits[split]["row_ids"]
                              if row_id not in lengths_by_row})
        if missing_ids:
            raise ValueError(f"Raw dataset lacks {len(missing_ids)} activation row IDs; first: {missing_ids[:3]}")
        threshold, threshold_metrics = _select_length_threshold(splits, lengths_by_row)
        heuristic = {
            "description": "predict low-value when sentence whitespace-token count <= threshold; proxy, not model-token count",
            "selected_threshold": threshold,
            "selection_split": "validation",
            "validation_threshold_search": threshold_metrics,
            "validation": None,
            "test": None,
        }
        for split in ("validation", "test"):
            data = splits[split]
            lengths = np.asarray([lengths_by_row[row_id] for row_id in data["row_ids"]])
            heuristic[split] = _metrics(data["labels"], (lengths <= threshold).astype(np.int64))
        baselines["sentence_length_heuristic"] = heuristic

    result = {
        "schema_version": "black-hole-baseline-results-v1",
        "positive_class": LABEL_NAMES[1],
        "label_to_id": {LABEL_NAMES[label]: label for label in (0, 1)},
        "seed": seed,
        "fit_split": "train",
        "selection_split": "validation",
        "final_report_split": "test",
        "group_leakage_check": "passed; question/group IDs are disjoint across splits",
        "split_summary": {
            split: {
                "row_count": int(len(splits[split]["labels"])),
                "question_count": int(len(np.unique(splits[split]["group_ids"]))),
                "class_counts": {LABEL_NAMES[label]: int(np.sum(splits[split]["labels"] == label))
                                 for label in (0, 1)},
            }
            for split in SPLITS
        },
        "feature_summary": {
            "sentence_activation_width": int(splits["train"]["features"].shape[1]),
            "state_vector_width": int(splits["train"]["state_features"].shape[1]) if state_available else None,
            "state_vector_baseline_included": state_available,
        },
        "baselines": baselines,
        "limitations": [
            "Sentence rows are grouped by question; row counts are not independent-question counts.",
            "Sentence-length baseline uses whitespace-separated words as a tokenizer-free proxy.",
            "False-skip rate is a label-level proxy and is not itself a measured change in final-answer accuracy.",
        ],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_results.json"
    output_path.write_text(json.dumps(_json_safe(result), indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="W4-001 directory containing train/validation/test.npz")
    parser.add_argument("--output-dir", required=True, help="Directory for baseline_results.json")
    parser.add_argument("--raw-dataset", help="Week 3 JSONL; enables the sentence-length heuristic baseline")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = run_baselines(args.data_dir, args.output_dir, raw_dataset=args.raw_dataset, seed=args.seed)
    print(json.dumps({
        "output": str(Path(args.output_dir) / "baseline_results.json"),
        "split_summary": result["split_summary"],
        "baselines": list(result["baselines"]),
    }, indent=2))


if __name__ == "__main__":
    main()
