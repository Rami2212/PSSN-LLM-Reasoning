"""Load Week 3 Black Hole labels and extract leakage-safe sentence activations.

The primary feature for a row is a pooled hidden-state vector over the target
sentence, evaluated after only the retained earlier sentences from its recorded
Week 3 context. Optional Week 2 semantic-state vectors are exported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.models.hidden_states import extract_span_hidden_vectors


SCHEMA_VERSION = "black-hole-activation-dataset-v1"
LABEL_TO_ID = {"answer_sensitive_in_context": 0, "low_value_in_context": 1}
SPLITS = ("train", "validation", "test")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                rows.append(row)
    except OSError as exc:
        raise ValueError(f"Could not read {path}") from exc
    return rows


def load_splits(split_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load and validate all split files and their question-level manifests."""
    split_dir = Path(split_dir)
    splits: dict[str, list[dict[str, Any]]] = {}
    memberships: dict[str, str] = {}
    seen_row_ids: set[str] = set()
    source_hashes = set()
    for split in SPLITS:
        path = split_dir / f"{split}.jsonl"
        manifest_path = split_dir / f"{split}_manifest.json"
        manifest = _read_json(manifest_path)
        if manifest.get("schema_version") != "black-hole-splits-v1" or manifest.get("split") != split:
            raise ValueError(f"Unexpected or mismatched split manifest: {manifest_path}")
        if not isinstance(manifest.get("dataset_sha256"), str):
            raise ValueError(f"{split}: manifest has no source dataset hash")
        source_hashes.add(manifest["dataset_sha256"])
        rows = _read_jsonl(path)
        if len(rows) != manifest.get("row_count"):
            raise ValueError(f"{split}: row count does not match its manifest")
        problem_ids = set(manifest.get("problem_ids", []))
        if len(problem_ids) != manifest.get("question_count"):
            raise ValueError(f"{split}: problem IDs do not match its manifest count")
        actual_ids = {row.get("problem_id") for row in rows}
        if actual_ids != problem_ids:
            raise ValueError(f"{split}: rows do not match manifest problem IDs")
        for problem_id in problem_ids:
            if problem_id in memberships:
                raise ValueError(f"Question leakage across {memberships[problem_id]} and {split}: {problem_id}")
            memberships[problem_id] = split
        for row in rows:
            row_id = row.get("dataset_row_id")
            if not isinstance(row_id, str) or not row_id or row_id in seen_row_ids:
                raise ValueError(f"{split}: missing or duplicate dataset_row_id {row_id!r}")
            seen_row_ids.add(row_id)
            if row.get("schema_version") != "black-hole-training-dataset-v1":
                raise ValueError(f"{split}: unexpected training row schema")
            if row.get("label") not in LABEL_TO_ID:
                raise ValueError(f"{split}: invalid label for row {row_id}")
        splits[split] = rows
    if len(source_hashes) != 1:
        raise ValueError("Train/validation/test manifests refer to different source datasets")
    return splits


def _source_model_configuration(splits: dict[str, list[dict[str, Any]]]) -> tuple[str, str]:
    model_ids, revisions = set(), set()
    for rows in splits.values():
        for row in rows:
            metadata = row.get("raw_evidence") or {}
            if metadata.get("model_id"):
                model_ids.add(metadata["model_id"])
            if metadata.get("model_revision"):
                revisions.add(metadata["model_revision"])
    if len(model_ids) != 1 or len(revisions) != 1:
        raise ValueError("Dataset rows must identify one consistent model ID and immutable revision")
    return next(iter(model_ids)), next(iter(revisions))


def _index_evidence(evidence_dir: Path, problem_id: str) -> dict[str, dict[str, Any]]:
    path = evidence_dir / problem_id / "label_evidence.jsonl"
    rows = _read_jsonl(path)
    evidence = {}
    for row in rows:
        evaluation = row.get("evaluation") or {}
        key = evaluation.get("evaluation_key")
        if key:
            if key in evidence:
                raise ValueError(f"Duplicate evaluation key in {path}: {key}")
            evidence[key] = row
    return evidence


def _trace_assets(evidence_dir: Path, problem_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = evidence_dir / problem_id
    embedded = _read_json(directory / "embedded_question_and_trace.json")
    manifest = _read_json(directory / "manifest.json")
    sentences = embedded.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        raise ValueError(f"No sentence units found for {problem_id}")
    sentence_map = {row.get("sentence_id"): row for row in sentences}
    if len(sentence_map) != len(sentences) or None in sentence_map:
        raise ValueError(f"Invalid or duplicate sentence IDs for {problem_id}")
    if not isinstance(manifest.get("thinking_prefix"), str) or not manifest["thinking_prefix"]:
        raise ValueError(f"Missing stored Qwen thinking prefix for {problem_id}")
    return {"embedded": embedded, "sentences": sentence_map}, manifest


def _build_causal_context(row: dict[str, Any], evidence: dict[str, Any], assets: dict[str, Any]) -> tuple[str, tuple[int, int]]:
    """Return prefix+retained-prior-context+target and its target char span."""
    evaluation = evidence.get("evaluation") or {}
    source_ids = evaluation.get("source_sentence_ids")
    before_mask = row.get("context_before_mask")
    if not isinstance(source_ids, list) or not isinstance(before_mask, list):
        raise ValueError(f"Missing source sentence IDs or context mask for {row.get('dataset_row_id')}")
    if len(before_mask) not in {len(source_ids), len(source_ids) + 1}:
        raise ValueError(f"Context-mask length does not match source sentences for {row.get('dataset_row_id')}")
    if any(type(flag) is not bool for flag in before_mask):
        raise ValueError(f"Non-boolean context mask for {row.get('dataset_row_id')}")

    target_id = row.get("sentence_id")
    sentence_map = assets["sentences"]
    target = sentence_map.get(target_id)
    if target is None or target.get("text") != row.get("sentence_text"):
        raise ValueError(f"Target sentence does not match Week 3 trace for {row.get('dataset_row_id')}")
    target_index = target.get("start_char_in_cot")
    if not isinstance(target_index, int):
        raise ValueError(f"Missing target trace offset for {row.get('dataset_row_id')}")
    if target_id not in source_ids:
        raise ValueError(f"Target sentence is absent from source_sentence_ids for {row.get('dataset_row_id')}")
    target_mask_index = source_ids.index(target_id)
    if not before_mask[target_mask_index]:
        raise ValueError(f"Target sentence is not present in its recorded pre-removal context: {row.get('dataset_row_id')}")

    selected_prior = []
    for index, source_id in enumerate(source_ids):
        if source_id not in sentence_map:
            raise ValueError(f"Missing source sentence {source_id} for {row.get('problem_id')}")
        source = sentence_map[source_id]
        start = source.get("start_char_in_cot")
        if not isinstance(start, int):
            raise ValueError(f"Missing source sentence offset for {source_id}")
        # Future trace units are never put into the target's activation context.
        if before_mask[index] and start < target_index:
            selected_prior.append(source)
    selected_prior.sort(key=lambda item: item["start_char_in_cot"])
    prefix = assets["thinking_prefix"]
    prior_text = "".join(item["text"] for item in selected_prior)
    target_text = target["text"]
    context = prefix + prior_text + target_text
    span = (len(prefix) + len(prior_text), len(prefix) + len(prior_text) + len(target_text))
    return context, span


def _load_optional_state_features(rows: list[dict[str, Any]], vectors_path: Path) -> np.ndarray:
    try:
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load Week 2 state vectors from {vectors_path}") from exc
    if vectors.ndim != 2:
        raise ValueError("Week 2 state vectors must be a 2D array")
    pooled = []
    for row in rows:
        refs = row.get("source_state_references") or []
        indices = [ref.get("representation", {}).get("vector_index") for ref in refs]
        if not indices or any(not isinstance(index, int) or index < 0 or index >= len(vectors) for index in indices):
            raise ValueError(f"Missing or invalid Week 2 state vector reference: {row.get('dataset_row_id')}")
        pooled.append(np.asarray(vectors[indices], dtype=np.float32).mean(axis=0))
    return np.stack(pooled).astype(np.float32, copy=False)


def _activation_for_context(model: Any, tokenizer: Any, context: str, span: tuple[int, int], layer: int) -> np.ndarray:
    vectors, _, _ = extract_span_hidden_vectors(model, tokenizer, context, [span], layer=layer, pooling="mean")
    return vectors[0]


def prepare_split_features(
    splits: dict[str, list[dict[str, Any]]],
    *,
    evidence_dir: str | Path,
    model: Any,
    tokenizer: Any,
    layer: int = -1,
    state_vectors: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract target-sentence activations; deduplicate identical causal inputs."""
    evidence_dir = Path(evidence_dir)
    cache: dict[str, np.ndarray] = {}
    evidence_cache: dict[str, dict[str, dict[str, Any]]] = {}
    asset_cache: dict[str, dict[str, Any]] = {}
    output = {}
    for split, rows in splits.items():
        vectors, labels, row_ids, groups, state_rows = [], [], [], [], []
        for row in rows:
            problem_id = row["problem_id"]
            if problem_id not in evidence_cache:
                evidence_cache[problem_id] = _index_evidence(evidence_dir, problem_id)
                assets, manifest = _trace_assets(evidence_dir, problem_id)
                if assets["embedded"].get("trace_id") != row.get("trace_id"):
                    raise ValueError(f"Trace identity mismatch for {problem_id}")
                asset_cache[problem_id] = assets | {"thinking_prefix": manifest["thinking_prefix"]}
            evidence = evidence_cache[problem_id].get(row.get("evaluation_key"))
            if evidence is None:
                raise ValueError(f"Missing raw Week 3 evidence for row {row.get('dataset_row_id')}")
            if evidence.get("evidence_type") != "gradient_greedy":
                raise ValueError(f"Unexpected evidence source for row {row.get('dataset_row_id')}")
            if evidence.get("unit", {}).get("sentence_id") != row.get("sentence_id"):
                raise ValueError(f"Evidence/label sentence mismatch for row {row.get('dataset_row_id')}")
            context, span = _build_causal_context(row, evidence, asset_cache[problem_id])
            cache_key = hashlib.sha256(context.encode("utf-8")).hexdigest()
            if cache_key not in cache:
                cache[cache_key] = _activation_for_context(model, tokenizer, context, span, layer)
            vectors.append(cache[cache_key])
            labels.append(LABEL_TO_ID[row["label"]])
            row_ids.append(row["dataset_row_id"])
            groups.append(problem_id)
        features = np.stack(vectors).astype(np.float32, copy=False)
        result: dict[str, Any] = {
            "features": features,
            "labels": np.asarray(labels, dtype=np.int64),
            "row_ids": np.asarray(row_ids, dtype=str),
            "group_ids": np.asarray(groups, dtype=str),
        }
        if state_vectors is not None:
            result["state_features"] = _load_optional_state_features(rows, Path(state_vectors))
        output[split] = result
    return output


def save_prepared_splits(
    prepared: dict[str, dict[str, Any]],
    output_dir: str | Path,
    *,
    split_dir: str | Path,
    evidence_dir: str | Path,
    model_id: str,
    model_revision: str | None,
    layer: int,
    state_vectors: str | Path | None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for split, data in prepared.items():
        path = output_dir / f"{split}.npz"
        payload = {key: value for key, value in data.items()}
        np.savez_compressed(path, **payload)
        manifests[split] = {
            "row_count": len(data["labels"]),
            "feature_shape": list(data["features"].shape),
            "label_counts": {str(label): int((data["labels"] == label).sum()) for label in (0, 1)},
            "has_state_baseline_features": "state_features" in data,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "label_to_id": LABEL_TO_ID,
        "positive_class": "low_value_in_context",
        "features": "mean-pooled target sentence hidden states at requested model layer",
        "causal_context": "Qwen thinking prefix + retained earlier Week 3 sentences + target sentence; no later sentence tokens",
        "model_id": model_id,
        "model_revision": model_revision,
        "layer": layer,
        "pooling": "mean",
        "split_strategy": "Week 3 question-level manifests; no question appears in multiple splits",
        "split_dir": str(Path(split_dir).resolve()),
        "evidence_dir": str(Path(evidence_dir).resolve()),
        "state_vectors_path": str(Path(state_vectors).resolve()) if state_vectors else None,
        "splits": manifests,
        "limitations": [
            "Labels estimate sentence-context removal sufficiency, not erasure of hidden internal computation.",
            "Week 2 state vectors are a separate coarser representation and are not concatenated into sentence features.",
        ],
    }
    summary_path = output_dir / "dataset_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True, help="W3-006 directory with split JSONL and manifests")
    parser.add_argument("--evidence-dir", type=Path, required=True, help="W3 run root containing per-question evidence folders")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-vectors", type=Path, help="Optional Week 2 state_vectors.npy for a separate baseline")
    parser.add_argument("--model-id", help="Defaults to the model recorded in every Week 3 row")
    parser.add_argument("--revision", help="Pin the model and tokenizer to an immutable revision")
    parser.add_argument("--layer", type=int, default=-1, help="Hidden-state layer index; default is final layer")
    parser.add_argument("--require-cuda", action="store_true", help="Fail instead of running on CPU")
    args = parser.parse_args(argv)

    from src.utils.environment import load_qwen_model

    splits = load_splits(args.split_dir)
    source_model_id, source_revision = _source_model_configuration(splits)
    model_id = args.model_id or source_model_id
    if model_id != source_model_id:
        raise ValueError(f"Requested model {model_id} does not match labelled model {source_model_id}")
    if args.revision and args.revision != source_revision:
        raise ValueError("Requested revision does not match the immutable revision used for Week 3 labels")
    tokenizer, model, _ = load_qwen_model(
        model_id, revision=source_revision, require_cuda=args.require_cuda
    )
    revision = getattr(getattr(model, "config", None), "_commit_hash", None) or source_revision
    if revision != source_revision:
        raise ValueError("Loaded model revision does not match the Week 3 label-generating revision")
    prepared = prepare_split_features(
        splits,
        evidence_dir=args.evidence_dir,
        model=model,
        tokenizer=tokenizer,
        layer=args.layer,
        state_vectors=args.state_vectors,
    )
    summary = save_prepared_splits(
        prepared,
        args.output_dir,
        split_dir=args.split_dir,
        evidence_dir=args.evidence_dir,
        model_id=model_id,
        model_revision=revision,
        layer=args.layer,
        state_vectors=args.state_vectors,
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "splits": summary["splits"]}, indent=2))


if __name__ == "__main__":
    main()
