"""Train a small controller, splitting by question before fitting preprocessing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import (PROTOCOL, Controller, atomic_json, binary_metrics, digest,
                     prepare_manifest, read_json, seed_all, status)


def matrices(directory, records, indices):
    xs, ys, metadata = [], [], []
    for i in indices:
        question = records[i]
        folder = directory / "questions" / question["question_id"]
        labels = read_json(folder / "labels.json")
        if digest(folder / "features.npz") != labels["feature_sha256"]:
            raise ValueError("Feature checksum mismatch")
        with np.load(folder / "features.npz", allow_pickle=False) as archive:
            features = archive["features"]
        layers = len(features)
        for layer, label in enumerate(labels["labels"]):
            if label < 0:
                continue
            xs.append(np.concatenate([features[layer], np.eye(layers, dtype=np.float32)[layer]]))
            ys.append(label)
            metadata.append({"question_id": question["question_id"], "layer": layer, "label": label})
    if not xs:
        raise ValueError("Split has no scorable labels; inspect labeling summary")
    return np.stack(xs), np.array(ys, dtype=np.float32), metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("Invalid training settings")
    seed_all(args.seed)
    records = read_json(args.labels_dir / "questions.json")
    source = read_json(args.labels_dir / "manifest.json")
    summary = read_json(args.labels_dir / "summary.json")
    if source["config"]["protocol"] != PROTOCOL:
        raise ValueError("Wrong labeling protocol")
    if len(records) < 10:
        raise ValueError("Need at least 10 questions for question-disjoint splits")
    order = np.random.default_rng(args.seed).permutation(len(records)).tolist()
    a, b = int(len(order) * 0.6), int(len(order) * 0.8)
    splits = {"train": order[:a], "validation": order[a:b], "test": order[b:]}
    config = {**vars(args), "labels_dir": str(args.labels_dir.resolve()),
              "output_dir": str(args.output_dir.resolve()), "source_identity": source["identity"],
              "labels_sha256": {q["question_id"]: digest(args.labels_dir / "questions" / q["question_id"] / "labels.json") for q in records},
              "split_question_ids": {k: [records[i]["question_id"] for i in v] for k, v in splits.items()}}
    prepare_manifest(args.output_dir, config)
    train_x, train_y, train_metadata = matrices(args.labels_dir, records, splits["train"])
    val_x, val_y, _ = matrices(args.labels_dir, records, splits["validation"])
    if len(np.unique(train_y)) != 2:
        raise ValueError("Training needs BOTH safe and unsafe examples. Collect more questions if all skips preserve answers.")
    mean = train_x.mean(axis=0)
    std = np.maximum(train_x.std(axis=0), 1e-5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor((train_x - mean) / std, device=device)
    y = torch.tensor(train_y, device=device)
    vx = torch.tensor((val_x - mean) / std, device=device)
    net = Controller(train_x.shape[1], 64).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((train_y == 0).sum() / (train_y == 1).sum()), device=device))
    history, best, stale = [], None, 0
    checkpoint = None
    for epoch in range(1, args.epochs + 1):
        net.train()
        permutation = torch.randperm(len(x), device=device)
        total_loss = 0.0
        for batch in permutation.split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        net.eval()
        with torch.inference_mode():
            probabilities = torch.sigmoid(net(vx)).cpu().numpy()
            val_loss = nn.functional.binary_cross_entropy_with_logits(net(vx), torch.tensor(val_y, device=device)).item()
        metrics = binary_metrics(val_y, probabilities >= args.threshold)
        score = (metrics["f1_safe"], -val_loss)
        row = {"epoch": epoch, "train_loss": total_loss / len(x), "validation_loss": val_loss, **metrics}
        history.append(row)
        if best is None or score > best:
            best, stale = score, 0
            checkpoint = {
                "protocol": PROTOCOL, "state_dict": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                "width": train_x.shape[1], "hidden": 64, "mean": torch.tensor(mean), "std": torch.tensor(std),
                "num_layers": summary["layers"], "hidden_size": summary["hidden_size"],
                "threshold": args.threshold, "best_epoch": epoch,
                "model_revision": source["config"]["model_revision"],
                "dataset_revision": source["config"]["dataset_revision"],
                "max_new_tokens": source["config"]["max_new_tokens"],
                "excluded_question_keys": [r["question_key"] for r in records],
                "source_identity": source["identity"],
                "eligible_layers": sorted({r["layer"] for r in train_metadata}),
            }
            temp = args.output_dir / "controller.tmp.pt"
            torch.save(checkpoint, temp)
            os.replace(temp, args.output_dir / "controller.pt")
        else:
            stale += 1
        atomic_json(args.output_dir / "history.json", history)
        status(args.output_dir, status="running", phase="training", epoch=epoch, best_epoch=checkpoint["best_epoch"])
        print(f"epoch={epoch} val_f1={metrics['f1_safe']:.4f} best={checkpoint['best_epoch']}", flush=True)
        if stale >= args.patience:
            break
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    # Held-out test evaluated only after validation has selected the checkpoint.
    test_x, test_y, test_metadata = matrices(args.labels_dir, records, splits["test"])
    with torch.inference_mode():
        test_prob = torch.sigmoid(net(torch.tensor((test_x - mean) / std, device=device))).cpu().numpy()
    predictions = [{**m, "probability_safe": float(p), "predicted_safe": bool(p >= args.threshold)} for m, p in zip(test_metadata, test_prob)]
    atomic_json(args.output_dir / "test_predictions.json", predictions)
    # Per-layer training frequency baseline avoids flattering a majority classifier.
    rates = {}
    for layer in range(summary["layers"]):
        labels = [r["label"] for r in train_metadata if r["layer"] == layer]
        rates[layer] = float(np.mean(labels)) if labels else 0.0
    report = {"best_epoch": checkpoint["best_epoch"], "test": binary_metrics(test_y, test_prob >= args.threshold),
              "best_validation_f1_safe": best[0],
              "always_skip_baseline": binary_metrics(test_y, np.ones(len(test_y))),
              "never_skip_baseline": binary_metrics(test_y, np.zeros(len(test_y))),
              "training_layer_frequency_baseline": binary_metrics(test_y, [rates[r["layer"]] >= args.threshold for r in test_metadata]),
              "rows": {"train": len(train_y), "validation": len(val_y), "test": len(test_y)},
              "question_counts": {k: len(v) for k, v in splits.items()},
              "scorable_question_counts": {"train": len({r["question_id"] for r in train_metadata}),
                                           "test": len({r["question_id"] for r in test_metadata})},
              "threshold": args.threshold, "test_used_for_selection": False,
              "limitation": "50 questions is a feasibility pilot; layer rows from one question are correlated."}
    atomic_json(args.output_dir / "evaluation.json", report)
    status(args.output_dir, status="completed", phase="finished")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
