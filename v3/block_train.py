"""Train/evaluate a controller whose actions are groups of four adjacent MLPs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import (Controller, atomic_json, binary_metrics, digest, read_json,
                     seed_all)
from .block_common import BLOCK4_PROTOCOL, group_matrices, prepare_manifest, status


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
    if source["config"]["protocol"] != BLOCK4_PROTOCOL:
        raise ValueError("Labels are not from the four-layer block protocol")
    if len(records) < 10:
        raise ValueError("Need at least 10 questions for question-disjoint splits")
    order = np.random.default_rng(args.seed).permutation(len(records)).tolist()
    a, b = int(len(order) * 0.6), int(len(order) * 0.8)
    splits = {"train": order[:a], "validation": order[a:b], "test": order[b:]}
    label_hashes = {
        q["question_id"]: digest(args.labels_dir / "questions" / q["question_id"] / "labels.json")
        for q in records
    }
    config = {
        "protocol": BLOCK4_PROTOCOL, "labels_dir": str(args.labels_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()), "seed": args.seed,
        "epochs": args.epochs, "patience": args.patience, "threshold": args.threshold,
        "source_identity": source["identity"], "label_sha256": label_hashes,
        "split_question_ids": {k: [records[i]["question_id"] for i in v] for k, v in splits.items()},
    }
    prepare_manifest(args.output_dir, config)
    train_x, train_y, train_meta = group_matrices(args.labels_dir, records, splits["train"])
    val_x, val_y, val_meta = group_matrices(args.labels_dir, records, splits["validation"])
    if len(np.unique(train_y)) != 2:
        raise ValueError("Training needs both safe and unsafe block examples; collect more questions.")
    mean = train_x.mean(axis=0)
    std = np.maximum(train_x.std(axis=0), 1e-5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor((train_x - mean) / std, device=device)
    y = torch.tensor(train_y, device=device)
    vx = torch.tensor((val_x - mean) / std, device=device)
    vy = torch.tensor(val_y, device=device)
    net = Controller(train_x.shape[1], 64).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    # Labels use 1=safe. Weight the majority safe class down when unsafe is rare.
    positive_weight = float((train_y == 0).sum() / (train_y == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    history, best, stale, best_checkpoint = [], None, 0, None
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
            logits = net(vx)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            val_loss = nn.functional.binary_cross_entropy_with_logits(logits, vy).item()
        metrics = binary_metrics(val_y, probabilities >= args.threshold)
        score = (metrics["f1_safe"], -val_loss)
        history.append({"epoch": epoch, "train_loss": total_loss / len(x),
                        "validation_loss": val_loss, **metrics})
        if best is None or score > best:
            best, stale = score, 0
            best_checkpoint = {
                "protocol": BLOCK4_PROTOCOL,
                "state_dict": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                "width": train_x.shape[1], "hidden": 64,
                "mean": torch.tensor(mean), "std": torch.tensor(std),
                "num_layers": summary["layers"], "hidden_size": summary["hidden_size"],
                "groups": summary["groups"], "threshold": args.threshold,
                "best_epoch": epoch, "model_revision": source["config"]["model_revision"],
                "dataset_revision": source["config"]["dataset_revision"],
                "max_new_tokens": source["config"]["max_new_tokens"],
                "excluded_question_keys": [r["question_key"] for r in records],
                "source_identity": source["identity"],
                "eligible_group_indices": sorted({r["group_index"] for r in train_meta}),
            }
            temp = args.output_dir / "controller.tmp.pt"
            torch.save(best_checkpoint, temp)
            os.replace(temp, args.output_dir / "controller.pt")
        else:
            stale += 1
        atomic_json(args.output_dir / "history.json", history)
        status(args.output_dir, status="running", phase="training", epoch=epoch,
               best_epoch=best_checkpoint["best_epoch"])
        print(f"epoch={epoch} val_f1_safe={metrics['f1_safe']:.4f} "
              f"best={best_checkpoint['best_epoch']}", flush=True)
        if stale >= args.patience:
            break
    net.load_state_dict(best_checkpoint["state_dict"])
    net.eval()
    test_x, test_y, test_meta = group_matrices(args.labels_dir, records, splits["test"])
    with torch.inference_mode():
        test_probs = torch.sigmoid(net(torch.tensor((test_x - mean) / std, device=device))).cpu().numpy()
    predictions = [{**m, "probability_safe": float(p),
                    "predicted_safe": bool(p >= args.threshold)} for m, p in zip(test_meta, test_probs)]
    atomic_json(args.output_dir / "test_predictions.json", predictions)
    rates = {}
    for group_index in range(len(summary["groups"])):
        labels = [r["label"] for r in train_meta if r["group_index"] == group_index]
        rates[group_index] = float(np.mean(labels)) if labels else 0.0
    report = {
        "protocol": BLOCK4_PROTOCOL,
        "best_epoch": best_checkpoint["best_epoch"],
        "test": binary_metrics(test_y, test_probs >= args.threshold),
        "best_validation_f1_safe": best[0],
        "always_skip_baseline": binary_metrics(test_y, np.ones(len(test_y))),
        "never_skip_baseline": binary_metrics(test_y, np.zeros(len(test_y))),
        "training_group_frequency_baseline": binary_metrics(
            test_y, [rates[r["group_index"]] >= args.threshold for r in test_meta]),
        "rows": {"train": len(train_y), "validation": len(val_y), "test": len(test_y)},
        "question_counts": {k: len(v) for k, v in splits.items()},
        "unscored_rows_mapped_to_unsafe": {
            "train": sum(r["was_unscored"] for r in train_meta),
            "validation": sum(r["was_unscored"] for r in val_meta),
            "test": sum(r["was_unscored"] for r in test_meta),
        },
        "threshold": args.threshold, "test_used_for_selection": False,
        "limitation": "Rows from each question are correlated; only one isolated 4-layer group is tested at a time.",
    }
    atomic_json(args.output_dir / "evaluation.json", report)
    status(args.output_dir, status="completed", phase="finished")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
