"""Generate fresh GSM8K test labels and evaluate the frozen predictor.

No training or threshold tuning occurs here. The default ten test questions
exclude every question used in training/validation/internal test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .common import PROTOCOL, Controller, atomic_json, binary_metrics, generation_args, read_json
from .label import collect
from .train import matrices


def main():
    parser = generation_args()
    parser.set_defaults(rows=10, split="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["protocol"] != PROTOCOL:
        raise ValueError("Wrong controller protocol")
    if args.revision != checkpoint["model_revision"] or args.max_new_tokens != checkpoint["max_new_tokens"]:
        raise ValueError("Use the checkpoint's model revision and generation budget")
    # Forbid an accidental dataset-version change between stages.
    args.dataset_revision = args.dataset_revision or checkpoint["dataset_revision"]
    collect(args, checkpoint["excluded_question_keys"])
    records = read_json(args.output_dir / "questions.json")
    x, y, metadata = matrices(args.output_dir, records, range(len(records)))
    model = Controller(checkpoint["width"], checkpoint["hidden"]).eval()
    model.load_state_dict(checkpoint["state_dict"])
    with torch.inference_mode():
        probabilities = torch.sigmoid(model((torch.tensor(x) - checkpoint["mean"]) / checkpoint["std"])).numpy()
    predictions = [{**m, "probability_safe": float(p), "predicted_safe": bool(p >= checkpoint["threshold"])} for m, p in zip(metadata, probabilities)]
    report = {
        "metrics": binary_metrics(y, probabilities >= checkpoint["threshold"]),
        "always_skip_baseline": binary_metrics(y, np.ones(len(y))),
        "never_skip_baseline": binary_metrics(y, np.zeros(len(y))),
        "threshold": checkpoint["threshold"], "threshold_tuned_here": False,
        "questions": len(records), "predictions": predictions,
        "interpretation": "Fresh question evaluation within GSM8K, not cross-dataset generalization. Unscored trials excluded from classification metrics; see labeling summary.",
    }
    atomic_json(args.output_dir / "predictor_evaluation.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
