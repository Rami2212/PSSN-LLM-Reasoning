"""Train word+character TF-IDF logistic regression on QUESTION TEXT only.

Split distinct questions before fitting vocabulary. Validation alone selects
regularization and confidence fallback. The test partition is never scored here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .common import (DEPTHS, PROTOCOL, Progress, atomic_json, decision_metrics,
                     load_labeled, manifest, route)


def make_pipeline(c=1.0, seed=42):
    return Pipeline([
        ("text", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                     token_pattern=r"(?u)\b\w+\b", max_features=30000)),
            ("characters", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                          sublinear_tf=True, max_features=30000)),
        ])),
        ("classifier", LogisticRegression(C=c, class_weight="balanced", max_iter=2000,
                                          random_state=seed, solver="lbfgs")),
    ])


def train(labels_dir, output_dir, seed=42, max_accuracy_drop=0.02):
    if not 0 <= max_accuracy_drop <= 1:
        raise ValueError("max-validation-accuracy-drop must be between 0 and 1")
    records, source, evidence = load_labeled(labels_dir)
    if len(records) < 20:
        raise ValueError("Need at least 20 distinct questions for three disjoint partitions")
    keys = [r["question_key"] for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate question text would leak across partitions")
    order = np.random.default_rng(seed).permutation(len(records))
    a, b = int(0.7 * len(records)), int(0.85 * len(records))
    partitions = {"train": [records[i] for i in order[:a]],
                  "validation": [records[i] for i in order[a:b]],
                  "test": [records[i] for i in order[b:]]}
    fitting = [r for r in partitions["train"] if r["target_depth"] is not None]
    if len(fitting) < 10 or len({r["target_depth"] for r in fitting}) < 2:
        raise ValueError("Training needs >=10 usable questions and >=2 target depths. "
                         "Inspect labels/summary.json: all-36 targets mean no useful shallow exits yet.")
    grid_c = (0.1, 1.0, 10.0)
    thresholds = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.01)
    config = {"source_identity": source["identity"], "evidence_sha256": evidence,
              "seed": seed, "max_validation_accuracy_drop": max_accuracy_drop,
              "regularization_grid": list(grid_c), "threshold_grid": list(thresholds),
              "split_question_ids": {k: [r["question_id"] for r in rows] for k, rows in partitions.items()},
              "features": "word 1-2 and character 3-5 TF-IDF; question text only"}
    output_dir = Path(output_dir)
    manifest(output_dir, config, "train")
    atomic_json(output_dir / "splits.json", {k: [{"question_id": r["question_id"], "question_key": r["question_key"]}
                                               for r in rows] for k, rows in partitions.items()})
    history, best = [], None
    with Progress(output_dir) as progress:
        for c in grid_c:
            progress.update(phase="fitting_text_classifier", regularization_c=c)
            pipeline = make_pipeline(c, seed)
            pipeline.fit([r["question"] for r in fitting], [r["target_depth"] for r in fitting])
            for threshold in thresholds:
                selected, _, _ = route(pipeline, [r["question"] for r in partitions["validation"]], threshold)
                metrics = decision_metrics(partitions["validation"], selected)
                allowed = metrics["accuracy_difference"] >= -max_accuracy_drop - 1e-12
                row = {"C": c, "threshold": threshold, "passes_validation_budget": allowed, **metrics}
                history.append(row)
                # Minimize compute, subject to the empirical validation accuracy budget.
                ranking = (metrics["mean_depth"], -metrics["accuracy_all_questions"], c, -threshold)
                if allowed and (best is None or ranking < best[0]):
                    best = (ranking, pipeline, row)
        if best is None:
            raise RuntimeError("No valid validation policy, including the full-depth fallback")
        _, pipeline, selected = best
        payload = {"protocol": PROTOCOL, "pipeline": pipeline, "threshold": selected["threshold"],
                   "generation": source["config"]["generation"], "source_identity": source["identity"],
                   "dataset_revision": source["config"]["dataset_revision"],
                   "evidence_sha256": evidence, "training_seed": seed,
                   "split_question_ids": config["split_question_ids"],
                   "excluded_question_keys": keys,
                   "validation_selection": selected,
                   "label_generation_code_sha256": source["config"].get("code_sha256", {}).get("common.py"),
                   "feature_definition": config["features"]}
        temp = output_dir / "controller.tmp.joblib"
        joblib.dump(payload, temp, compress=3)
        os.replace(temp, output_dir / "controller.joblib")
        atomic_json(output_dir / "validation_search.json", history)
        report = {"status": "completed", "controller": "controller.joblib",
                  "predictor": "TF-IDF word+character features with balanced logistic regression",
                  "question_counts": {k: len(v) for k, v in partitions.items()},
                  "training_usable_questions": len(fitting),
                  "training_target_counts": {str(d): sum(r["target_depth"] == d for r in fitting) for d in DEPTHS},
                  "selected_validation_policy": selected,
                  "always_full_depth_fallback": selected["threshold"] > 1,
                  "test_used_for_selection": False,
                  "limitation": "The accuracy budget is empirical on validation, not a guarantee on new questions. "
                                "A 100-question dataset is only a feasibility pilot."}
        atomic_json(output_dir / "training_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-validation-accuracy-drop", type=float, default=0.02)
    args = parser.parse_args()
    print(json.dumps(train(args.labels_dir, args.output_dir, args.seed, args.max_validation_accuracy_drop), indent=2))


if __name__ == "__main__":
    main()
