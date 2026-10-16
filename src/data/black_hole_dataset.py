"""Build the CPU-only Week 3.005 Black Hole sentence-context training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "black-hole-training-dataset-v1"
SOURCE_LABEL_SCHEMA = "black-hole-sentence-label-v1"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_state_index(path):
    """Index representation metadata; vector contents remain in their existing file."""
    index = {}
    for trace in read_jsonl(path):
        trace_id = trace.get("trace_id")
        problem_id = trace.get("problem_id")
        for state in trace.get("states", []):
            key = (trace_id, problem_id, state.get("state_id"))
            if not all(isinstance(value, str) and value for value in key):
                raise ValueError("State dataset contains an incomplete state identity")
            if key in index:
                raise ValueError(f"Duplicate state identity in Week 2 dataset: {key}")
            index[key] = {
                "state_id": state["state_id"],
                "state_index": state["state_index"],
                "state_text": state["text"],
                "state_token_count": state["token_count"],
                "is_final_state": state["is_final_state"],
                "representation": state.get("representation"),
            }
    if not index:
        raise ValueError("No states found in Week 2 dataset")
    return index


def build_training_dataset(labels_path, state_dataset_path):
    states = build_state_index(state_dataset_path)
    records, rejected, seen = [], [], set()
    for label in read_jsonl(labels_path):
        label_id = label.get("label_id")
        if label.get("schema_version") != SOURCE_LABEL_SCHEMA:
            raise ValueError(f"Unexpected W3-004 label schema: {label.get('schema_version')}")
        if label.get("label") not in {"low_value_in_context", "answer_sensitive_in_context"}:
            rejected.append({"label_id": label_id, "reason": "not_an_accepted_training_label"})
            continue
        if not isinstance(label_id, str) or not label_id:
            raise ValueError("W3-004 label has no label_id")
        if label_id in seen:
            raise ValueError(f"Duplicate W3-004 label ID: {label_id}")
        seen.add(label_id)
        refs = []
        missing = []
        for state_id in label["source_state_ids"]:
            state = states.get((label["trace_id"], label["problem_id"], state_id))
            if state is None:
                missing.append(state_id)
            else:
                refs.append(state)
        if missing:
            rejected.append({"label_id": label_id, "problem_id": label["problem_id"],
                             "sentence_id": label["sentence_id"],
                             "reason": "missing_week2_state_reference", "missing_state_ids": missing})
            continue
        if any(ref["is_final_state"] for ref in refs):
            rejected.append({"label_id": label_id, "problem_id": label["problem_id"],
                             "sentence_id": label["sentence_id"],
                             "reason": "final_answer_state_mapping"})
            continue
        context_mask = label["context_before_mask"]
        removal_mask = label["removal_mask"]
        if len(context_mask) != len(removal_mask) or not any(context_mask) or any(
                after and not before for before, after in zip(context_mask, removal_mask)):
            raise ValueError(f"Invalid removal-mask transition for {label_id}")
        record = {
            "schema_version": SCHEMA_VERSION,
            "dataset_row_id": label_id,
            "label": label["label"],
            "label_reason": label["label_reason"],
            "problem_id": label["problem_id"],
            "trace_id": label["trace_id"],
            "question": label["question"],
            "reference_answer": label["reference_answer"],
            "unit_type": "reasoning_sentence_context",
            "sentence_id": label["sentence_id"],
            "sentence_text": label["sentence_text"],
            "sentence_character_count": len(label["sentence_text"]),
            "sentence_offsets": {
                "start_char_in_cot": label["sentence_start_char_in_cot"],
                "end_char_in_cot": label["sentence_end_char_in_cot"],
                "start_char_in_source": label["sentence_start_char_in_source"],
                "end_char_in_source": label["sentence_end_char_in_source"],
            },
            "source_state_references": refs,
            "context_before_mask": context_mask,
            "removal_mask": removal_mask,
            "context_sentence_count": len(context_mask) - 1,
            "retained_before_removal_count": sum(context_mask),
            "retained_after_removal_count": sum(removal_mask),
            "correct_count": label["correct_count"],
            "sample_count": label["sample_count"],
            "attainability": label["attainability"],
            "sufficiency": label["sufficiency"],
            "evaluation_key": label["evaluation_key"],
            "raw_evidence": {
                "raw_trials_directory": label["raw_trials_directory"],
                "core_summary_path": label["core_summary_path"],
                "run_identity": label["run_identity"],
                "model_id": label["model_id"],
                "model_revision": label["model_revision"],
                "protocol_name": label["protocol"]["name"],
            },
            "source_label_id": label_id,
        }
        records.append(record)
    records.sort(key=lambda row: (row["problem_id"], row["sentence_id"], row["evaluation_key"]))
    return records, rejected


def write_dataset(labels_path, state_dataset_path, output_dir):
    labels_path, state_dataset_path, output_dir = map(Path, (labels_path, state_dataset_path, output_dir))
    records, rejected = build_training_dataset(labels_path, state_dataset_path)
    if not records:
        raise ValueError("No accepted training records were produced")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output_dir / "black_hole_training_dataset.jsonl", records)
    atomic_jsonl(output_dir / "excluded_dataset_rows.jsonl", rejected)
    labels = Counter(row["label"] for row in records)
    question_ids = {row["problem_id"] for row in records}
    unit_contexts = defaultdict(set)
    for row in records:
        unit_contexts[(row["trace_id"], row["sentence_id"])].add(tuple(row["context_before_mask"]))
    state_reference_count = sum(len(row["source_state_references"]) for row in records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_row_count": len(records),
        "question_count": len(question_ids),
        "label_counts": dict(sorted(labels.items())),
        "average_rows_per_question": len(records) / len(question_ids),
        "unique_sentence_unit_count": len(unit_contexts),
        "multi_context_sentence_unit_count": sum(len(contexts) > 1 for contexts in unit_contexts.values()),
        "state_reference_count": state_reference_count,
        "average_state_references_per_row": state_reference_count / len(records),
        "excluded_dataset_row_count": len(rejected),
        "conversion_policy": (
            "One W3-004 sentence-context label becomes one training row. Week 2 state representations "
            "are attached as references only; no sentence label is copied to an entire semantic state."),
        "source_files": {
            "w3_004_labels": str(labels_path.resolve()),
            "week2_state_dataset": str(state_dataset_path.resolve()),
            "w3_004_labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            "week2_state_dataset_sha256": hashlib.sha256(state_dataset_path.read_bytes()).hexdigest(),
        },
    }
    atomic_json(output_dir / "black_hole_training_dataset_summary.json", summary)
    atomic_json(output_dir / "black_hole_training_dataset_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "source_files": summary["source_files"],
        "output_hashes": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                          for file in sorted(output_dir.glob("*.json*"))
                          if file.name != "black_hole_training_dataset_manifest.json"},
    })
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--state-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = write_dataset(args.labels, args.state_dataset, args.output_dir)
    print(json.dumps({key: summary[key] for key in ("training_row_count", "question_count", "label_counts",
                                                     "excluded_dataset_row_count")}, indent=2))


if __name__ == "__main__":
    main()
