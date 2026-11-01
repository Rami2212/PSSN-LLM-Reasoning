"""Evaluate a frozen text controller using saved outcomes on held-out questions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import (DEPTHS, Progress, atomic_json, decision_metrics, digest,
                     load_controller, load_labeled, manifest, route)


def evaluate(labels_dir, checkpoint, output_dir, partition="test"):
    controller = load_controller(checkpoint)
    records, source, evidence = load_labeled(labels_dir)
    if controller["source_identity"] != source["identity"] or controller["evidence_sha256"] != evidence:
        raise ValueError("Controller and labeling evidence differ")
    ids = set(controller["split_question_ids"][partition])
    selected_rows = [r for r in records if r["question_id"] in ids]
    if len(selected_rows) != len(ids):
        raise ValueError("Held-out questions are missing")
    manifest(output_dir, {"controller_sha256": digest(checkpoint), "evidence_sha256": evidence,
                          "partition": partition}, "evaluate")
    with Progress(output_dir):
        depths, confidence, raw = route(controller["pipeline"], [r["question"] for r in selected_rows], controller["threshold"])
        predictions = [{"question_id": r["question_id"], "question": r["question"],
                        "target_depth": r["target_depth"], "raw_predicted_depth": int(p),
                        "selected_depth": int(d), "confidence": float(c),
                        "fallback_to_36": bool(p != 36 and d == 36),
                        "selected_answer_correct": r["outcomes"][str(d)]["correct"],
                        "full_depth_correct": r["outcomes"]["36"]["correct"]}
                       for r, d, c, p in zip(selected_rows, depths, confidence, raw)]
        oracle = [min((d for d in DEPTHS if r["outcomes"][str(d)]["correct"]), default=36) for r in selected_rows]
        report = {"partition": partition, "controller": decision_metrics(selected_rows, depths),
                  "fixed_depths": {str(d): decision_metrics(selected_rows, [d] * len(selected_rows)) for d in DEPTHS},
                  "hindsight_oracle_diagnostic": decision_metrics(selected_rows, oracle),
                  "threshold": controller["threshold"], "test_used_for_selection": False,
                  "evaluation_type": "reuse of previously collected held-out generation outcomes",
                  "note": "Oracle uses answers and is not deployable. Cached evaluation provides no new runtime-speedup measurement."}
        atomic_json(Path(output_dir) / "predictions.json", predictions)
        atomic_json(Path(output_dir) / "metrics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.labels_dir, args.controller, args.output_dir, args.partition), indent=2))


if __name__ == "__main__":
    main()
