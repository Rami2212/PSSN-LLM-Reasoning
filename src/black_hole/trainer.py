"""Train the Week 4 Black Hole detector from saved activation arrays.

Only train.npz and validation.npz are read. The test split is deliberately
left untouched for W4-005 final evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .checkpointing import atomic_json_save, atomic_torch_save
from .config import BlackHoleDetectorConfig
from .detector import BlackHoleDetector
from .loss import BlackHoleLoss, positive_class_weight


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_split(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing activation split: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "labels", "row_ids", "group_ids"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        result = {key: archive[key] for key in required}
    result["features"] = np.asarray(result["features"], dtype=np.float32)
    result["labels"] = np.asarray(result["labels"], dtype=np.int64)
    result["row_ids"] = np.asarray(result["row_ids"]).astype(str)
    result["group_ids"] = np.asarray(result["group_ids"]).astype(str)
    count = len(result["labels"])
    if result["features"].ndim != 2 or not count:
        raise ValueError(f"{path}: features must be a non-empty 2-D matrix")
    if not np.isfinite(result["features"]).all():
        raise ValueError(f"{path}: features contain NaN or infinity")
    if not np.isin(result["labels"], [0, 1]).all():
        raise ValueError(f"{path}: labels must be 0 or 1")
    if any(len(result[key]) != count for key in ("features", "row_ids", "group_ids")):
        raise ValueError(f"{path}: arrays have inconsistent row counts")
    if len(set(result["row_ids"].tolist())) != count:
        raise ValueError(f"{path}: duplicate row IDs")
    return result


def _load_train_validation(data_dir: str | Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    data_dir = Path(data_dir)
    splits = {
        split: _load_split(data_dir / f"{split}.npz")
        for split in ("train", "validation")
    }
    widths = {data["features"].shape[1] for data in splits.values()}
    if len(widths) != 1:
        raise ValueError(f"Train/validation feature widths differ: {widths}")
    train_groups = set(splits["train"]["group_ids"].tolist())
    validation_groups = set(splits["validation"]["group_ids"].tolist())
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError(f"Question/group leakage across train and validation: {sorted(overlap)[:5]}")
    hashes = {split: _sha256(data_dir / f"{split}.npz") for split in splits}
    return splits, hashes


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "row_count": int(len(y_true)),
        "accuracy": float((tp + tn) / len(y_true)),
        "precision_low_value": float(precision),
        "recall_low_value": float(recall),
        "f1_low_value": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "answer_sensitive_false_skip_rate": float(fp / (tn + fp)) if tn + fp else 0.0,
        "probability_threshold": float(threshold),
    }


def _evaluate(
    model: BlackHoleDetector,
    features: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: BlackHoleLoss,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    with torch.inference_mode():
        logits = model.forward_logits(features)
        loss = float(loss_fn(logits, labels).item())
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    result = _metrics(labels.detach().cpu().numpy().astype(np.int64), probabilities)
    result["loss"] = loss
    return result, probabilities


def train_detector(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 12,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_size: int = 256,
    dropout: float = 0.2,
    seed: int = 42,
    use_class_weights: bool = True,
    device: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Train on train split, early-stop on validation F1, and persist checkpoints."""
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")

    splits, input_hashes = _load_train_validation(data_dir)
    train_x = splits["train"]["features"]
    train_y = splits["train"]["labels"]
    val_x = splits["validation"]["features"]
    val_y = splits["validation"]["labels"]
    if len(np.unique(train_y)) != 2:
        raise ValueError("Training split must contain both classes")

    if device == "auto":
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = torch.device(device)
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this runtime")

    _seed_everything(seed)
    train_mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    train_scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    train_scale[train_scale < 1e-6] = 1.0
    train_x = np.nan_to_num((train_x - train_mean) / train_scale, copy=False)
    val_x = np.nan_to_num((val_x - train_mean) / train_scale, copy=False)

    config = BlackHoleDetectorConfig(
        input_size=int(train_x.shape[1]), hidden_size=hidden_size, dropout=dropout
    )
    model = BlackHoleDetector(config).to(selected_device)
    positive_weight = positive_class_weight(torch.as_tensor(train_y)) if use_class_weights else None
    loss_fn = BlackHoleLoss(positive_weight).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_features = torch.as_tensor(train_x, dtype=torch.float32, device=selected_device)
    train_labels = torch.as_tensor(train_y, dtype=torch.float32, device=selected_device)
    validation_features = torch.as_tensor(val_x, dtype=torch.float32, device=selected_device)
    validation_labels = torch.as_tensor(val_y, dtype=torch.float32, device=selected_device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_config = {
        "schema_version": "black-hole-detector-training-v1",
        "seed": seed,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "use_class_weights": use_class_weights,
        "positive_class_weight": positive_weight,
        "early_stopping_metric": "validation_f1_low_value",
        "validation_probability_threshold": 0.5,
        "feature_normalization": "train-split mean/std; std below 1e-6 replaced with 1",
        "device": str(selected_device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if selected_device.type == "cuda" else None,
        "input_hashes": input_hashes,
        "split_rows": {split: int(len(data["labels"])) for split, data in splits.items()},
        "split_question_counts": {split: int(len(np.unique(data["group_ids"]))) for split, data in splits.items()},
        "detector_config": config.to_dict(),
    }
    atomic_json_save(training_config, output_dir / "training_config.json")
    config.save(output_dir / "detector_config.json")

    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    last_checkpoint = output_dir / "last_checkpoint.pt"
    if resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {last_checkpoint}")
        saved = torch.load(last_checkpoint, map_location=selected_device, weights_only=False)
        old_config = saved["training_config"]
        for key in (
            "seed", "batch_size", "patience", "learning_rate", "weight_decay",
            "use_class_weights", "positive_class_weight", "input_hashes", "detector_config",
        ):
            if old_config.get(key) != training_config.get(key):
                raise ValueError(f"Resume settings differ from checkpoint for {key}")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        best_f1 = float(saved["best_f1"])
        best_loss = float(saved["best_validation_loss"])
        best_epoch = int(saved["best_epoch"])
        stale_epochs = int(saved["stale_epochs"])
        history = list(saved["history"])
        if start_epoch > epochs:
            raise ValueError(f"Checkpoint already reached epoch {start_epoch - 1}; increase --epochs to resume")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        # A per-epoch generator makes shuffling reproducible even after resume.
        generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
        order = torch.randperm(len(train_features), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size].to(selected_device)
            batch_features = train_features[batch_indices]
            batch_labels = train_labels[batch_indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_logits(batch_features)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(batch_indices)

        validation_metrics, _ = _evaluate(model, validation_features, validation_labels, loss_fn)
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_features),
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        current_f1 = float(validation_metrics["f1_low_value"])
        current_loss = float(validation_metrics["loss"])
        improved = current_f1 > best_f1 or (current_f1 == best_f1 and current_loss < best_loss)
        if improved:
            best_f1, best_loss, best_epoch = current_f1, current_loss, epoch
            stale_epochs = 0
            atomic_torch_save({
                "model_state_dict": model.state_dict(),
                "detector_config": config.to_dict(),
                "feature_mean": torch.as_tensor(train_mean),
                "feature_scale": torch.as_tensor(train_scale),
                "best_epoch": best_epoch,
                "validation_metrics": validation_metrics,
                "input_hashes": input_hashes,
            }, output_dir / "best_model.pt")
        else:
            stale_epochs += 1

        atomic_json_save(history, output_dir / "training_history.json")
        atomic_torch_save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": training_config,
            "feature_mean": torch.as_tensor(train_mean),
            "feature_scale": torch.as_tensor(train_scale),
            "best_f1": best_f1,
            "best_validation_loss": best_loss,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
        }, last_checkpoint)
        print(
            f"epoch {epoch:03d}/{epochs} | train_loss={epoch_record['train_loss']:.4f} "
            f"| val_f1={current_f1:.4f} | best_epoch={best_epoch} "
            f"| patience={stale_epochs}/{patience}",
            flush=True,
        )
        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.", flush=True)
            break

    final_epoch = history[-1]["epoch"]
    atomic_torch_save({
        "model_state_dict": model.state_dict(),
        "detector_config": config.to_dict(),
        "feature_mean": torch.as_tensor(train_mean),
        "feature_scale": torch.as_tensor(train_scale),
        "final_epoch": final_epoch,
        "input_hashes": input_hashes,
    }, output_dir / "final_model.pt")
    summary = {
        "status": "completed",
        "output_dir": str(output_dir),
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "best_validation_f1_low_value": best_f1,
        "best_validation_loss": best_loss,
        "early_stopped": final_epoch < epochs,
        "test_split_used": False,
        "checkpoint_files": ["best_model.pt", "final_model.pt", "last_checkpoint.pt"],
        "training_rows": int(len(train_y)),
        "validation_rows": int(len(val_y)),
        "training_questions": int(len(np.unique(splits["train"]["group_ids"]))),
        "validation_questions": int(len(np.unique(splits["validation"]["group_ids"]))),
    }
    atomic_json_save(summary, output_dir / "training_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="W4-001 folder with train.npz and validation.npz")
    parser.add_argument("--output-dir", required=True, help="Persistent directory for model/checkpoints/logs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from output-dir/last_checkpoint.pt")
    args = parser.parse_args()
    summary = train_detector(
        args.data_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        seed=args.seed,
        use_class_weights=not args.no_class_weights,
        device=args.device,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
