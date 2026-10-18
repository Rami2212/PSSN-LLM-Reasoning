"""Extract causal target-sentence activations for sentence training data.

Input rows come from ``sentence_training_data.py``. Each activation is computed
from a Qwen chat-formatted question followed by the supplied reasoning prefix
and target sentence. No answer is generated and no future sentence is included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.models.hidden_states import extract_span_hidden_vectors


SPLITS = ("train", "validation", "test")
LABEL_TO_ID = {"answer_sensitive_in_context": 0, "low_value_in_context": 1}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}")
                rows.append(row)
    except OSError as exc:
        raise FileNotFoundError(f"Could not read split file: {path}") from exc
    return rows


def load_question_disjoint_splits(dataset_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load split JSONL files and reject question, row, or label leakage."""
    dataset_dir = Path(dataset_dir)
    splits: dict[str, list[dict[str, Any]]] = {}
    seen_rows: set[str] = set()
    seen_questions: dict[str, str] = {}
    for split in SPLITS:
        rows = _read_rows(dataset_dir / "splits" / f"{split}.jsonl")
        if not rows:
            raise ValueError(f"The {split} split is empty")
        for row in rows:
            row_id = row.get("dataset_row_id")
            problem_id = row.get("problem_id")
            if not isinstance(row_id, str) or not row_id or row_id in seen_rows:
                raise ValueError(f"Missing or duplicated dataset_row_id in {split}: {row_id!r}")
            if not isinstance(problem_id, str) or not problem_id:
                raise ValueError(f"Missing problem_id for row {row_id}")
            if row.get("split") != split:
                raise ValueError(f"Row {row_id} says split={row.get('split')!r}, expected {split!r}")
            previous_split = seen_questions.get(problem_id)
            if previous_split is not None and previous_split != split:
                raise ValueError(
                    f"Question leakage: {problem_id} appears in {previous_split} and {split}"
                )
            if row.get("label") not in LABEL_TO_ID:
                raise ValueError(f"Unknown label for row {row_id}: {row.get('label')!r}")
            if row.get("label_id") != LABEL_TO_ID[row["label"]]:
                raise ValueError(f"label_id does not match label for row {row_id}")
            for field in ("question", "reasoning_prefix", "sentence_text"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Missing or empty {field} for row {row_id}")
            seen_rows.add(row_id)
            seen_questions[problem_id] = split
        splits[split] = rows
    return splits


def build_activation_context(tokenizer: Any, row: dict[str, Any]) -> tuple[str, tuple[int, int]]:
    """Render question + prior reasoning + target; return target character span."""
    messages = [{"role": "user", "content": row["question"]}]
    prefix = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("Tokenizer did not return a rendered Qwen chat prefix")
    reasoning_prefix = row["reasoning_prefix"]
    target = row["sentence_text"]
    start = len(prefix) + len(reasoning_prefix)
    context = prefix + reasoning_prefix + target
    return context, (start, start + len(target))


def extract_row_activation(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    layer: int = -1,
    torch_module: Any | None = None,
) -> np.ndarray:
    context, target_span = build_activation_context(tokenizer, row)
    vectors, _, _ = extract_span_hidden_vectors(
        model, tokenizer, context, [target_span], layer=layer,
        pooling="mean", torch_module=torch_module,
    )
    return vectors[0]


def prepare_activation_splits(
    splits: dict[str, list[dict[str, Any]]],
    model: Any,
    tokenizer: Any,
    *,
    layer: int = -1,
    torch_module: Any | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Extract and deduplicate equivalent causal contexts within each split."""
    output: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        rows = splits[split]
        cache: dict[str, np.ndarray] = {}
        features, labels, row_ids, group_ids = [], [], [], []
        for index, row in enumerate(rows, 1):
            context, _ = build_activation_context(tokenizer, row)
            key = hashlib.sha256(context.encode("utf-8")).hexdigest()
            if key not in cache:
                cache[key] = extract_row_activation(
                    model, tokenizer, row, layer=layer, torch_module=torch_module
                )
            features.append(cache[key])
            labels.append(LABEL_TO_ID[row["label"]])
            row_ids.append(row["dataset_row_id"])
            group_ids.append(row["problem_id"])
            if index % 50 == 0 or index == len(rows):
                print(f"{split}: {index}/{len(rows)} rows ({len(cache)} unique contexts)", flush=True)
        output[split] = {
            "features": np.stack(features).astype(np.float32, copy=False),
            "labels": np.asarray(labels, dtype=np.int64),
            "row_ids": np.asarray(row_ids, dtype=str),
            "group_ids": np.asarray(group_ids, dtype=str),
        }
    return output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_activation_splits(
    prepared: dict[str, dict[str, np.ndarray]],
    *,
    dataset_dir: Path,
    output_dir: Path,
    model_id: str,
    model_revision: str | None,
    layer: int,
    precision: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"Output already exists; choose a new --output-dir: {output_dir}")
    output_dir.mkdir(parents=True)
    split_summaries = {}
    for split, arrays in prepared.items():
        path = output_dir / f"{split}.npz"
        np.savez_compressed(path, **arrays)
        split_summaries[split] = {
            "row_count": len(arrays["labels"]),
            "question_count": len(set(arrays["group_ids"].tolist())),
            "feature_shape": list(arrays["features"].shape),
            "label_counts": {
                str(label): int((arrays["labels"] == label).sum()) for label in (0, 1)
            },
            "sha256": _sha256(path),
        }
    manifest = {
        "schema_version": "pssn-sentence-activations-v1",
        "model_id": model_id,
        "model_revision": model_revision,
        "precision": precision,
        "layer": layer,
        "pooling": "mean over target sentence tokens",
        "context": "Qwen chat-formatted question + reasoning_prefix + target sentence; no future sentences or reference answer",
        "split_strategy": "prebuilt 70/15/15 question-disjoint splits; verified before extraction",
        "dataset_dir": str(dataset_dir.resolve()),
        "source_split_sha256": {
            split: _sha256(dataset_dir / "splits" / f"{split}.jsonl") for split in SPLITS
        },
        "splits": split_summaries,
        "limitations": [
            "Transferred sentence labels have not been independently validated for this activation model.",
            "Only the selected layer is extracted; this does not reproduce the paper's all-layer probe input.",
        ],
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="sentence-training-5000 directory with splits/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--revision", help="Optional immutable Hugging Face revision")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Hidden-state layer index; -1 means final layer")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)

    splits = load_question_disjoint_splits(args.dataset_dir)
    from src.utils.environment import load_qwen_model

    tokenizer, model, precision = load_qwen_model(
        args.model_id, revision=args.revision, require_cuda=args.require_cuda
    )
    revision = getattr(getattr(model, "config", None), "_commit_hash", None) or args.revision
    prepared = prepare_activation_splits(splits, model, tokenizer, layer=args.layer)
    summary = save_activation_splits(
        prepared,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        model_revision=revision,
        layer=args.layer,
        precision=precision,
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "splits": summary["splits"]}, indent=2))


if __name__ == "__main__":
    main()
