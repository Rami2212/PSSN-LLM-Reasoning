"""Evaluate predicted depth outcomes and routing on question-disjoint held-out data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .controller_common import (DEPTHS, Progress, atomic_json, digest, load_controller,
    load_labels, manifest, model_text, route_depth)


def evaluate(labels_dir, checkpoint, output_dir):
    payload = load_controller(checkpoint)
    records, source, evidence = load_labels(labels_dir)
    if payload["selected_dataset_sha256"] != evidence or payload["label_manifest_identity"] != source["identity"]:
        raise ValueError("Controller and labeled evidence differ")
    # Use the exact held-out question IDs recorded by training.
    ids = set(payload["split_question_ids"]["test"])
    rows = [r for r in records if r["question_id"] in ids]
    if len(rows) != len(ids) or not rows:
        raise ValueError("Missing test questions")
    output_dir = Path(output_dir)
    manifest(output_dir, {"controller_sha256": digest(checkpoint),
                          "selected_dataset_sha256": evidence, "partition": "test"}, "evaluate")
    pipeline = payload["pipeline"]
    predictions, flat_y, flat_hat = [], [], []
    with Progress(output_dir):
        for row in rows:
            depth_scores = {}
            for depth in DEPTHS:
                trial = row["outcomes"][str(depth)]
                text = model_text(row["question"], depth)
                probabilities = pipeline.predict_proba([text])[0]
                positive_idx = list(pipeline.classes_).index(1)
                probability = float(probabilities[positive_idx])
                predicted_safe = probability >= 0.5
                depth_scores[str(depth)] = {"predicted_answer_preservation_probability": probability,
                                            "predicted_safe": predicted_safe,
                                            "observed_safe": bool(trial["correct"] and trial["complete"])}
                flat_y.append(int(trial["correct"] and trial["complete"]))
                flat_hat.append(int(predicted_safe))
            selected, route_probs = route_depth(pipeline, row["question"])
            predictions.append({"question_id": row["question_id"], "depth_predictions": depth_scores,
                                "selected_depth": selected, "routing_probabilities":
                                {str(d): route_probs[d] for d in DEPTHS},
                                "selected_answer_correct": depth_scores[str(selected)]["observed_safe"]})
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        binary = {"question_depth_trials": len(flat_y), "accuracy": accuracy_score(flat_y, flat_hat),
                  "f1_answer_preserved": f1_score(flat_y, flat_hat, zero_division=0),
                  "confusion_matrix_true_rows_predicted_columns": confusion_matrix(flat_y, flat_hat, labels=[0, 1]).tolist()}
        routed = {"questions": len(rows),
                  "controller_answer_accuracy": sum(p["selected_answer_correct"] for p in predictions) / len(rows),
                  "full_depth_answer_accuracy": sum(r["outcomes"]["32"]["correct"] and r["outcomes"]["32"]["complete"] for r in rows) / len(rows),
                  "correct_to_wrong": sum(r["outcomes"]["32"]["correct"] and r["outcomes"]["32"]["complete"]
                                          and not p["selected_answer_correct"] for r, p in zip(rows, predictions)),
                  "selected_depth_counts": {str(d): sum(p["selected_depth"] == d for p in predictions) for d in DEPTHS},
                  "mean_selected_depth": sum(p["selected_depth"] for p in predictions) / len(rows),
                  "threshold": 0.5}
        fixed = {str(d): {"questions": len(rows),
                          "answer_accuracy": sum(r["outcomes"][str(d)]["correct"] and r["outcomes"][str(d)]["complete"] for r in rows) / len(rows)}
                 for d in DEPTHS}
        report = {"status": "completed", "partition": "question_disjoint_held_out_test",
                  "depth_outcome_classifier": binary, "controller_routing": routed,
                  "fixed_depths": fixed, "test_used_for_model_selection": False,
                  "timing_note": "Uses cached generation evidence; live speed is measured in controller_test.",
                  "population_note": "Fixed random GSM8K training questions. This 100-question pilot is not a population-level guarantee."}
        atomic_json(output_dir / "predictions.json", predictions)
        atomic_json(output_dir / "metrics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.labels_dir, args.controller, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
