"""Evaluate a frozen four-layer-block controller on fresh GSM8K questions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import Controller, atomic_json, binary_metrics, generation_args, read_json
from .block_common import BLOCK4_PROTOCOL, group_matrices
from .block_label import collect


def main():
    parser = generation_args()
    parser.set_defaults(rows=10, split="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["protocol"] != BLOCK4_PROTOCOL:
        raise ValueError("Wrong controller protocol")
    if args.revision != checkpoint["model_revision"] or args.max_new_tokens != checkpoint["max_new_tokens"]:
        raise ValueError("Use the checkpoint's model revision and generation budget")
    args.dataset_revision = args.dataset_revision or checkpoint["dataset_revision"]
    collect(args, checkpoint["excluded_question_keys"])
    records = read_json(args.output_dir / "questions.json")
    x, y, metadata = group_matrices(args.output_dir, records, range(len(records)))
    model = Controller(checkpoint["width"], checkpoint["hidden"]).eval()
    model.load_state_dict(checkpoint["state_dict"])
    with torch.inference_mode():
        probs = torch.sigmoid(model((torch.tensor(x) - checkpoint["mean"]) / checkpoint["std"])).numpy()
    predictions = [{**row, "probability_safe": float(p),
                    "predicted_safe": bool(p >= checkpoint["threshold"])}
                   for row, p in zip(metadata, probs)]
    report = {
        "protocol": BLOCK4_PROTOCOL,
        "metrics": binary_metrics(y, probs >= checkpoint["threshold"]),
        "always_skip_baseline": binary_metrics(y, np.ones(len(y))),
        "never_skip_baseline": binary_metrics(y, np.zeros(len(y))),
        "threshold": checkpoint["threshold"], "threshold_tuned_here": False,
        "questions": len(records), "predictions": predictions,
        "unscored_rows_mapped_to_unsafe": sum(row["was_unscored"] for row in metadata),
        "interpretation": "Fresh GSM8K questions; each label is for one isolated four-layer group. Unscored outcomes count as unsafe.",
    }
    atomic_json(args.output_dir / "predictor_evaluation.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
