"""CPU evaluation on the 15 held-out label articles, without fitting or tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .controller_common import (DEPTHS, Progress, atomic_json, digest, load_controller,
    load_labels, manifest, model_text, route_depth)
from .controller_train import route_report


def evaluate(labels_dir, checkpoint, output_dir):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    payload = load_controller(checkpoint)
    rows, labels, _ = load_labels(labels_dir)
    if (payload["labels_identity"] != labels["identity"]
            or payload["labels_sha256"] != digest(Path(labels_dir) / "selected_dataset.json")):
        raise ValueError("Controller and labeling evidence differ")
    ids = set(payload["split_article_ids"]["test"])
    held = [r for r in rows if r["article_id"] in ids]
    if not held or len(held) != len(ids):
        raise ValueError("Missing held-out articles")
    output_dir = Path(output_dir)
    manifest(output_dir, {"checkpoint_sha256": digest(checkpoint), "labels_identity": labels["identity"],
                          "partition": "held_out_label_articles"})
    with Progress(output_dir):
        y, predicted, details = [], [], []
        for row in held:
            selected, probabilities = route_depth(payload["pipeline"], row["visible_article"], payload["threshold"])
            details.append({"article_id": row["article_id"], "selected_depth": selected,
                            "probabilities": probabilities, "observed_quality_labels": row["quality_labels"]})
            for d in DEPTHS:
                y.append(int(row["quality_labels"][str(d)]))
                predicted.append(int(probabilities[d] >= payload["threshold"]))
        report = {"status": "completed", "test_used_for_selection": False,
                  "depth_label_classifier": {"trials": len(y), "threshold": payload["threshold"],
                      "accuracy": float(accuracy_score(y, predicted)),
                      "precision": float(precision_score(y, predicted, zero_division=0)),
                      "recall": float(recall_score(y, predicted, zero_division=0)),
                      "f1": float(f1_score(y, predicted, zero_division=0)),
                      "confusion_matrix_true_rows_predicted_columns_0_1": confusion_matrix(y, predicted, labels=[0, 1]).tolist()},
                  "routing": route_report(held, payload["pipeline"], payload["threshold"]),
                  "fixed_depth_mean_rougeL": {str(d): sum(r["outcomes"][str(d)]["rougeL"] for r in held) / len(held)
                                              for d in DEPTHS},
                  "note": "Cached quality evidence on held-out articles. This is not a live latency benchmark. Binary label accuracy is not summary accuracy."}
        atomic_json(output_dir / "predictions.json", details)
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
