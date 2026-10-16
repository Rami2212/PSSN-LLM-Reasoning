"""Validate W3-005 Black Hole records and create leakage-safe question splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "black-hole-splits-v1"
VALID_LABELS = {"low_value_in_context", "answer_sensitive_in_context"}
SPLITS = ("train", "validation", "test")
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{number}") from exc


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


def validate_records(records):
    """Return question groups or raise before any split files are written."""
    groups, seen_ids, seen_evaluations = defaultdict(list), set(), set()
    for number, row in enumerate(records, 1):
        required = ("dataset_row_id", "problem_id", "trace_id", "sentence_id", "label",
                    "evaluation_key", "context_before_mask", "removal_mask", "source_state_references")
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f"Dataset row {number} is missing: {', '.join(missing)}")
        if row.get("schema_version") != "black-hole-training-dataset-v1":
            raise ValueError(f"Dataset row {number} has an unexpected schema version")
        if row["label"] not in VALID_LABELS:
            raise ValueError(f"Dataset row {number} has an invalid label")
        if row["dataset_row_id"] in seen_ids:
            raise ValueError(f"Duplicate dataset_row_id: {row['dataset_row_id']}")
        if row["evaluation_key"] in seen_evaluations:
            raise ValueError(f"Duplicate evaluation_key: {row['evaluation_key']}")
        seen_ids.add(row["dataset_row_id"])
        seen_evaluations.add(row["evaluation_key"])
        before, after = row["context_before_mask"], row["removal_mask"]
        if len(before) != len(after) or not any(before) or any(a and not b for b, a in zip(before, after)):
            raise ValueError(f"Dataset row {number} has an invalid context-mask transition")
        references = row["source_state_references"]
        if not references or any(reference.get("is_final_state") for reference in references):
            raise ValueError(f"Dataset row {number} has an invalid state reference")
        groups[row["problem_id"]].append(row)
    if len(groups) < 3:
        raise ValueError("At least three distinct problems are required for train/validation/test splits")
    return groups


def integer_targets(total, ratios):
    raw = {name: total * ratios[name] for name in SPLITS}
    values = {name: int(raw[name]) for name in SPLITS}
    for name in sorted(SPLITS, key=lambda item: (raw[item] - values[item], item), reverse=True)[:total - sum(values.values())]:
        values[name] += 1
    return values


def make_question_splits(groups, seed=42):
    """Deterministically assign whole questions while approximately balancing both labels."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    problems = list(groups)
    rng = random.Random(seed)
    rng.shuffle(problems)
    problems.sort(key=lambda problem: -len(groups[problem]))
    question_targets = integer_targets(len(problems), RATIOS)
    global_labels = Counter(row["label"] for rows in groups.values() for row in rows)
    label_targets = {name: integer_targets(global_labels[name], RATIOS) for name in VALID_LABELS}
    assigned = {name: [] for name in SPLITS}
    counts = {name: Counter() for name in SPLITS}
    question_counts = Counter()
    for problem in problems:
        group_labels = Counter(row["label"] for row in groups[problem])
        candidates = [name for name in SPLITS if question_counts[name] < question_targets[name]]
        def cost(name):
            new_counts = {split: Counter(counts[split]) for split in SPLITS}
            new_counts[name].update(group_labels)
            label_error = sum(abs(new_counts[split][label] - label_targets[label][split])
                              for split in SPLITS for label in VALID_LABELS)
            group_error = sum(abs((question_counts[split] + (split == name)) - question_targets[split])
                              for split in SPLITS)
            overshoot = sum(max(0, new_counts[split][label] - label_targets[label][split])
                            for split in SPLITS for label in VALID_LABELS)
            return label_error + group_error * 4 + overshoot * 2
        selected = min(candidates, key=lambda name: (cost(name), name))
        assigned[selected].append(problem)
        counts[selected].update(group_labels)
        question_counts[selected] += 1
    return assigned


def build_splits(dataset_path, output_dir, seed=42):
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    records = [row for _, row in read_jsonl(dataset_path)]
    groups = validate_records(records)
    assignments = make_question_splits(groups, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_records = {}
    for split in SPLITS:
        rows = [row for problem in sorted(assignments[split]) for row in groups[problem]]
        split_records[split] = rows
        atomic_jsonl(output_dir / f"{split}.jsonl", rows)
        atomic_json(output_dir / f"{split}_manifest.json", {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "split": split,
            "problem_ids": sorted(assignments[split]),
            "question_count": len(assignments[split]),
            "row_count": len(rows),
            "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        })
    memberships = defaultdict(set)
    for split, problems in assignments.items():
        for problem in problems:
            memberships[problem].add(split)
    leakage = {problem: sorted(parts) for problem, parts in memberships.items() if len(parts) != 1}
    if leakage:
        raise AssertionError(f"Question leakage found: {leakage}")
    all_labels = Counter(row["label"] for row in records)
    quality = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "input_row_count": len(records),
        "input_question_count": len(groups),
        "input_label_counts": dict(sorted(all_labels.items())),
        "split_strategy": "deterministic question-level greedy label balancing",
        "splits": {split: {
            "question_count": len(assignments[split]),
            "row_count": len(split_records[split]),
            "label_counts": dict(sorted(Counter(row["label"] for row in split_records[split]).items())),
            "problem_ids": sorted(assignments[split]),
        } for split in SPLITS},
        "leakage_problem_count": len(leakage),
        "limitations": [
            "Rows from the same question are intentionally dependent and remain in one split.",
            "Labels come from three-sample context-sufficiency experiments, not direct hidden-computation measurements.",
            "The 99-question dataset is suitable for a pilot classifier experiment, not strong broad generalization claims.",
        ],
    }
    atomic_json(output_dir / "week3_quality_summary.json", quality)
    return quality


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    summary = build_splits(args.dataset, args.output_dir, args.seed)
    print(json.dumps({"input_row_count": summary["input_row_count"],
                      "input_question_count": summary["input_question_count"],
                      "leakage_problem_count": summary["leakage_problem_count"],
                      "splits": summary["splits"]}, indent=2))


if __name__ == "__main__":
    main()
