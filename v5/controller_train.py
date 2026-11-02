"""Train a binary question+depth answer-preservation controller on 100 x 3 trials."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .controller_common import (DEPTHS, PROTOCOL, Progress, atomic_json, digest,
    load_labels, manifest, model_text, read_json, route_depth, stratified_partitions)


def make_pipeline(c, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    return Pipeline([
        ("text", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                       token_pattern=r"(?u)\b\w+\b", max_features=20000)),
            ("characters", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                            sublinear_tf=True, max_features=20000)),
        ])),
        ("classifier", LogisticRegression(C=c, class_weight="balanced", max_iter=2000,
                                            random_state=seed, solver="lbfgs")),
    ])


def expand(rows):
    features, labels = [], []
    for row in rows:
        for depth in DEPTHS:
            result = row["outcomes"][str(depth)]
            features.append(model_text(row["question"], depth))
            labels.append(int(result["correct"] and result["complete"]))
    return features, labels


def route_report(rows, pipeline):
    selected, safe, baseline = [], [], []
    for row in rows:
        depth, probabilities = route_depth(pipeline, row["question"])
        selected.append(depth)
        chosen = row["outcomes"][str(depth)]
        full = row["outcomes"][str(DEPTHS[-1])]
        safe.append(bool(chosen["correct"] and chosen["complete"]))
        baseline.append(bool(full["correct"] and full["complete"]))
    return {"questions": len(rows), "controller_answer_accuracy": sum(safe) / len(rows),
            "full_depth_answer_accuracy": sum(baseline) / len(rows),
            "correct_to_wrong": sum(a and not b for a, b in zip(baseline, safe)),
            "depth_counts": {str(d): selected.count(d) for d in DEPTHS},
            "mean_selected_depth": sum(selected) / len(selected),
            "routing_rule": "choose shallowest depth with predicted answer-preservation probability >= 0.5; otherwise use depth 32"}


def train(labels_dir, output_dir, seed=42):
    import joblib
    records, source, evidence = load_labels(labels_dir)
    partitions = stratified_partitions(records, seed)
    regularization_grid = (0.1, 1.0, 10.0)
    split_ids = {name: [r["question_id"] for r in rows] for name, rows in partitions.items()}
    config = {"label_manifest_identity": source["identity"], "selected_dataset_sha256": evidence,
              "seed": seed, "regularization_grid": list(regularization_grid),
              "selection": "validation routed answer accuracy; ties use lower mean depth then smaller C",
              "features": "word 1-2 and character 3-5 TF-IDF of question text plus categorical depth token",
              "routing": "predict answer preservation for each depth; choose shallowest probability >= 0.5, otherwise depth 32",
              "split_question_ids": split_ids}
    output_dir = Path(output_dir)
    source_train = manifest(output_dir, config, "train")
    atomic_json(output_dir / "splits.json", partitions)
    train_x, train_y = expand(partitions["train"])
    if len(set(train_y)) != 2:
        raise ValueError("Training split needs both answer-preserved and unsafe question-depth trials; inspect labels first")
    history, best = [], None
    with Progress(output_dir) as progress:
        for c in regularization_grid:
            progress.update(phase="fitting", regularization_c=c)
            pipeline = make_pipeline(c, seed)
            pipeline.fit(train_x, train_y)
            routed = route_report(partitions["validation"], pipeline)
            val_x, val_y = expand(partitions["validation"])
            val_predictions = pipeline.predict(val_x)
            binary_accuracy = sum(int(a) == int(b) for a, b in zip(val_y, val_predictions)) / len(val_y)
            row = {"C": c, "binary_depth_outcome_accuracy": binary_accuracy, **routed}
            history.append(row)
            ranking = (-routed["controller_answer_accuracy"], routed["mean_selected_depth"], c)
            if best is None or ranking < best[0]:
                best = (ranking, pipeline, history[-1])
        _, pipeline, selected = best
        payload = {"protocol": PROTOCOL, "pipeline": pipeline,
                   "training_manifest_identity": source_train["identity"],
                   "label_manifest_identity": source["identity"], "selected_dataset_sha256": evidence,
                   "model_revision": source["config"]["model_revision"],
                   "dataset_revision": source["config"]["dataset_revision"],
                   "max_new_tokens": source["config"]["max_new_tokens"],
                   "generation_seed": source["config"]["seed"],
                   "demonstrations": read_json(Path(labels_dir) / "demonstrations.json"),
                   "excluded_question_keys": [r["question_key"] for r in records],
                   "split_question_ids": split_ids, "feature_definition": config["features"],
                   "routing": config["routing"],
                   "common_code_sha256": digest(Path(__file__).with_name("controller_common.py")),
                   "smoke_code_sha256": digest(Path(__file__).with_name("smoke.py"))}
        temporary = output_dir / "controller.tmp.joblib"
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, output_dir / "controller.joblib")
        atomic_json(output_dir / "validation_search.json", history)
        class_counts = {str(label): train_y.count(label) for label in (0, 1)}
        report = {"status": "completed", "labeling_questions": len(records),
                  "labeled_question_depth_trials": len(records) * len(DEPTHS),
                  "question_counts": {name: len(part) for name, part in partitions.items()},
                  "training_trial_count": len(train_y), "training_outcome_counts": class_counts,
                  "selected_validation_model": selected, "test_used_for_model_selection": False,
                  "controller_sha256": digest(output_dir / "controller.joblib"),
                  "routing": config["routing"], "features": config["features"],
                  "limitation": "100 questions are a small pilot. Splits are question-disjoint. One greedy outcome per depth is noisy; the controller predicts observed answer preservation, not guaranteed safety."}
        atomic_json(output_dir / "training_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.labels_dir, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
