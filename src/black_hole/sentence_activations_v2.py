"""V2 activation extraction for the 5k sentence dataset.

This is separate from the original Week 4 extractor. It reads the consolidated
sentence_training_data.jsonl, assigns question-disjoint train/validation/test
splits, then writes trainer-compatible activation arrays and split JSONL files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.black_hole.sentence_activations import (
    LABEL_TO_ID,
    build_activation_context,
)
from src.models.hidden_states import token_positions_for_span

SPLITS = ("train", "validation", "test")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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
    return rows


def _normalized_question(row: dict[str, Any]) -> str:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Missing question for row {row.get('dataset_row_id')!r}")
    return " ".join(unicodedata.normalize("NFKC", question).split())


def split_rows_by_question(
    rows: list[dict[str, Any]], *, seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Assign all sentences from each normalized question to one split."""
    if not rows:
        raise ValueError("The input dataset is empty")
    seen_ids: set[str] = set()
    problem_to_question: dict[str, str] = {}
    question_to_problem: dict[str, str] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_id = row.get("dataset_row_id")
        problem_id = row.get("problem_id")
        if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
            raise ValueError(f"Missing or duplicate dataset_row_id: {row_id!r}")
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError(f"Missing problem_id for row {row_id}")
        if row.get("label") not in LABEL_TO_ID:
            raise ValueError(f"Unknown label for row {row_id}: {row.get('label')!r}")
        if row.get("label_id") != LABEL_TO_ID[row["label"]]:
            raise ValueError(f"label_id does not match label for row {row_id}")
        if not isinstance(row.get("reasoning_prefix"), str):
            raise ValueError(f"Missing or invalid reasoning_prefix for row {row_id}")
        if not isinstance(row.get("sentence_text"), str) or not row["sentence_text"].strip():
            raise ValueError(f"Missing or empty sentence_text for row {row_id}")
        question = _normalized_question(row)
        if problem_id in problem_to_question and problem_to_question[problem_id] != question:
            raise ValueError(f"problem_id {problem_id} maps to more than one question")
        if question in question_to_problem and question_to_problem[question] != problem_id:
            raise ValueError(f"Identical normalized question has conflicting problem_ids: {problem_id}")
        problem_to_question[problem_id] = question
        question_to_problem[question] = problem_id
        seen_ids.add(row_id)
        groups[question].append(row)

    questions = sorted(groups)
    if len(questions) < 3:
        raise ValueError("At least three distinct questions are required for 3-way splits")
    random.Random(seed).shuffle(questions)
    n_train = min(len(questions) - 2, max(1, round(len(questions) * 0.70)))
    n_validation = min(len(questions) - n_train - 1, max(1, round(len(questions) * 0.15)))
    assignments = {
        **{question: "train" for question in questions[:n_train]},
        **{question: "validation" for question in questions[n_train:n_train + n_validation]},
        **{question: "test" for question in questions[n_train + n_validation:]},
    }
    result = {split: [] for split in SPLITS}
    for question, group in groups.items():
        split = assignments[question]
        result[split].extend({**row, "split": split} for row in group)
    for split, split_rows in result.items():
        if {row["label_id"] for row in split_rows} != {0, 1}:
            raise ValueError(f"{split} split does not contain both labels; choose another --seed")
        split_rows.sort(key=lambda row: (row["problem_id"], row["trace_id"], row["sentence_index"], row["dataset_row_id"]))
    question_membership: dict[str, str] = {}
    for split, split_rows in result.items():
        for row in split_rows:
            problem_id = row["problem_id"]
            previous = question_membership.setdefault(problem_id, split)
            if previous != split:
                raise AssertionError(f"Question leakage: {problem_id} in {previous} and {split}")
    return result


def load_and_split(dataset_dir: str | Path, *, seed: int = 42):
    dataset_dir = Path(dataset_dir)
    path = dataset_dir / "sentence_training_data.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing consolidated 5k dataset: {path}")
    return split_rows_by_question(_read_jsonl(path), seed=seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_row_representations(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    torch_module: Any | None = None,
) -> tuple[Any, Any, int]:
    """Pool target tokens at every transformer layer in one model forward pass.

    Returns (final-layer vector, concatenated all-layer vector, layer count).
    The embedding-output entry in Hugging Face ``hidden_states`` is excluded;
    the remaining entries correspond to transformer layers.
    """
    import numpy as np

    if torch_module is None:
        import torch as torch_module

    context, target_span = build_activation_context(tokenizer, row)
    encoded = tokenizer(
        context,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    encoded = dict(encoded)
    offsets = encoded.pop("offset_mapping", None)
    if offsets is None:
        raise ValueError("Tokenizer must provide offset_mapping for target sentence pooling")
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        offsets = offsets[0]
    offsets = [(int(start), int(end)) for start, end in offsets]
    positions = token_positions_for_span(offsets, target_span)
    if not positions:
        raise ValueError(f"Target sentence has no tokens for row {row.get('dataset_row_id')}")

    device = getattr(model, "device", None)
    if device is None and callable(getattr(model, "get_input_embeddings", None)):
        device = model.get_input_embeddings().weight.device
    model_inputs = {
        name: value.to(device) if device is not None and callable(getattr(value, "to", None)) else value
        for name, value in encoded.items()
    }
    if callable(getattr(model, "eval", None)):
        model.eval()
    inference_mode = getattr(torch_module, "inference_mode", None)
    context_manager = inference_mode() if callable(inference_mode) else None
    if context_manager is None:
        from contextlib import nullcontext
        context_manager = nullcontext()
    with context_manager:
        outputs = model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and isinstance(outputs, dict):
        hidden_states = outputs.get("hidden_states")
    if not hidden_states or len(hidden_states) < 2:
        raise ValueError("Model did not return transformer hidden states")

    pooled_layers = []
    for hidden in hidden_states[1:]:
        if len(hidden.shape) != 3 or hidden.shape[0] != 1:
            raise ValueError("Expected hidden states with shape [1, sequence_length, hidden_size]")
        if hidden.shape[1] != len(offsets):
            raise ValueError("Token offsets do not match hidden-state sequence length")
        target_hidden = hidden[0, positions, :]
        if callable(getattr(target_hidden, "float", None)):
            target_hidden = target_hidden.float()
        try:
            pooled = target_hidden.mean(dim=0)
        except TypeError:
            pooled = target_hidden.mean(axis=0)
        if callable(getattr(pooled, "detach", None)):
            pooled = pooled.detach()
        if callable(getattr(pooled, "cpu", None)):
            pooled = pooled.cpu()
        if callable(getattr(pooled, "numpy", None)):
            pooled = pooled.numpy()
        pooled_layers.append(np.asarray(pooled, dtype=np.float32))

    stacked = np.stack(pooled_layers).astype(np.float32, copy=False)
    return stacked[-1], stacked.reshape(-1), len(pooled_layers)


def run_extraction(
    *, dataset_dir: str | Path, output_dir: str | Path, model: Any, tokenizer: Any,
    model_id: str, model_revision: str | None, precision: str, seed: int = 42,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Extract both final-layer and all-layer features in one pass per context."""
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"Output already exists; choose a new --output-dir: {output_dir}")
    input_path = dataset_dir / "sentence_training_data.jsonl"
    splits = load_and_split(dataset_dir, seed=seed)
    import numpy as np

    output_dir.mkdir(parents=True)
    all_layers_dir = output_dir / "all_layers"
    all_layers_dir.mkdir()
    split_summary = {}
    for split in SPLITS:
        rows = splits[split]
        cache: dict[str, Any] = {}
        final_features, all_layer_features = [], []
        labels, row_ids, group_ids = [], [], []
        split_path = output_dir / f"{split}.jsonl"
        with split_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for index, row in enumerate(rows, 1):
            context, _ = build_activation_context(tokenizer, row)
            context_id = hashlib.sha256(context.encode("utf-8")).hexdigest()
            if context_id not in cache:
                cache[context_id] = extract_row_representations(
                    model, tokenizer, row, torch_module=torch_module
                )
            final_vector, all_layer_vector, _ = cache[context_id]
            final_features.append(final_vector)
            all_layer_features.append(all_layer_vector)
            labels.append(LABEL_TO_ID[row["label"]])
            row_ids.append(row["dataset_row_id"])
            group_ids.append(row["problem_id"])
            if index % 50 == 0 or index == len(rows):
                print(f"{split}: {index}/{len(rows)} rows ({len(cache)} unique contexts)", flush=True)
        shared_arrays = {
            "labels": np.asarray(labels, dtype=np.int64),
            "row_ids": np.asarray(row_ids, dtype=str),
            "group_ids": np.asarray(group_ids, dtype=str),
        }
        final_arrays = {
            **shared_arrays,
            "features": np.stack(final_features).astype(np.float32, copy=False),
        }
        all_arrays = {
            **shared_arrays,
            "features": np.stack(all_layer_features).astype(np.float32, copy=False),
        }
        final_path = output_dir / f"{split}.npz"
        all_path = all_layers_dir / f"{split}.npz"
        np.savez_compressed(final_path, **final_arrays)
        np.savez_compressed(all_path, **all_arrays)
        split_summary[split] = {
            "row_count": len(rows),
            "question_count": len(set(group_ids)),
            "label_counts": {str(label): int((shared_arrays["labels"] == label).sum()) for label in (0, 1)},
            "final_layer": {
                "path": final_path.name,
                "feature_shape": list(final_arrays["features"].shape),
                "sha256": _sha256(final_path),
            },
            "all_layers": {
                "path": f"all_layers/{all_path.name}",
                "feature_shape": list(all_arrays["features"].shape),
                "sha256": _sha256(all_path),
            },
        }
    layer_counts = {entry["all_layers"]["feature_shape"][1] for entry in split_summary.values()}
    final_widths = {entry["final_layer"]["feature_shape"][1] for entry in split_summary.values()}
    if len(layer_counts) != 1 or len(final_widths) != 1:
        raise ValueError("Feature dimensions changed between splits")
    hidden_size = next(iter(final_widths))
    layer_count = next(iter(layer_counts)) // hidden_size
    manifest = {
        "schema_version": "pssn-sentence-activations-v2.1",
        "model_id": model_id,
        "model_revision": model_revision,
        "precision": precision,
        "feature_representations": {
            "final_layer": "mean-pooled target tokens from the final transformer layer; saved at output root",
            "all_layers": "mean-pooled target tokens from every transformer layer, concatenated layer-wise; saved under all_layers/",
        },
        "transformer_layer_count": layer_count,
        "embedding_output_included": False,
        "pooling": "mean over target sentence tokens",
        "context": "Qwen chat-formatted question + reasoning_prefix + target sentence; no reference answer or future sentences",
        "split_strategy": "deterministic 70/15/15 split by normalized question text; no question crosses splits",
        "seed": seed,
        "dataset_dir": str(dataset_dir.resolve()),
        "input_file": str(input_path.resolve()),
        "input_sha256": _sha256(input_path),
        "splits": split_summary,
        "limitations": [
            "Transferred sentence labels have not been independently validated for this activation model.",
            "All-layer feature vectors are much wider and may overfit this dataset; compare held-out question results.",
        ],
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="sentence-training-5000 folder containing sentence_training_data.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--revision", help="Optional immutable Hugging Face revision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)

    from src.utils.environment import load_qwen_model

    tokenizer, model, precision = load_qwen_model(
        args.model_id, revision=args.revision, require_cuda=args.require_cuda
    )
    revision = getattr(getattr(model, "config", None), "_commit_hash", None) or args.revision
    result = run_extraction(
        dataset_dir=args.dataset_dir, output_dir=args.output_dir,
        model=model, tokenizer=tokenizer, model_id=args.model_id,
        model_revision=revision, precision=precision, seed=args.seed,
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "splits": result["splits"]}, indent=2))


if __name__ == "__main__":
    main()
