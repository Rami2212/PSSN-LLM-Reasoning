"""Prepare sentence-level training examples from the configured data release.

Run: python -m src.data.sentence_training_data --source-dir <release-directory> --limit 5000
Source and license details are written separately from model-facing rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from src.data.black_hole_dataset import atomic_json, atomic_jsonl, canonical_hash


SCHEMA = "pssn-sentence-training-v1"
DEFAULT_OUTPUT = Path("PSSN/artifacts/black_hole/sentence-training-5000-v2")
LABELS = {False: "answer_sensitive_in_context", True: "low_value_in_context"}


def load_bundle(path: Path) -> tuple[dict, dict]:
    """Released JSON is [records, metadata], not a flat list of row objects."""
    with path.open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not (isinstance(value, list) and len(value) == 2
            and all(isinstance(part, dict) for part in value)):
        raise ValueError(f"Expected [records, metadata] in {path}")
    return value[0], value[1]


def problem_identity(question: str) -> str:
    """Group identical normalized questions across source models/datasets."""
    normalized = " ".join(unicodedata.normalize("NFKC", question).split())
    return "question-" + canonical_hash(normalized)


def record_paths(root: Path, record_reference: str) -> tuple[Path, Path, Path, int]:
    """Resolve a release record reference without trusting its embedded path."""
    location, raw_index = record_reference.rsplit(":", 1)
    parts = PurePosixPath(location.replace("\\", "/")).parts
    if (len(parts) != 3 or parts[0] != "results"
            or any(p in {".", ".."} for p in parts)
            or not raw_index.isdigit() or "_generations" not in parts[-1]):
        raise ValueError(f"Unexpected record reference: {record_reference}")
    generation = root / parts[1] / parts[2]
    if not generation.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Record reference escapes the configured input: {record_reference}")
    return (generation,
            generation.with_name(parts[2].replace("_generations", "_processed_sentences", 1)),
            generation.with_name(parts[2].replace("_generations", "_attribution_pruning", 1)),
            int(raw_index))


def resolve_trace_context(entry_key, generation, step, attribution, model_id, dataset):
    """Preserve post-shortcut sentence indexing; never re-split generated text."""
    post = step.get("post_removal", {})
    sentences = post.get("sentences")
    prefix_length = attribution.get("prefix_length")
    if (not isinstance(sentences, list) or not sentences
            or not all(isinstance(text, str) for text in sentences)
            or type(prefix_length) is not int or not 0 < prefix_length <= len(sentences)):
        raise ValueError("missing_or_invalid_processed_prefix")
    if step.get("skipped") or generation.get("is_cot_complete") is not True:
        raise ValueError("source_trace_skipped_or_incomplete")
    question = generation.get("input_str")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing_question")
    if step.get("input_str") != question:
        raise ValueError("generation_question_does_not_match_processed_trace")
    if not isinstance(generation.get("gt_answer"), str) or not generation["gt_answer"]:
        raise ValueError("missing_reference_answer")
    trace_id = "trace-" + canonical_hash([model_id, entry_key])
    return {
        "trace_id": trace_id,
        "problem_id": problem_identity(question),
        "record_key": entry_key,
        "model_name": model_id,
        "dataset_name": dataset,
        "question": question,
        "reference_answer": generation["gt_answer"],
        "processed_sentences": sentences,
        "processed_answer_sentence_index": post.get("answer_sentence_idx"),
        "sufficient_prefix_length": prefix_length,
    }


def make_training_record(tag, sentence_index, trace):
    """Map the released binary target to the PSSN sentence-training schema."""
    removable = tag.get("ground_truth_is_removable")
    if type(removable) is not bool:
        raise ValueError("missing_boolean_experiment_label")
    sentences = trace["processed_sentences"]
    if not 0 <= sentence_index < trace["sufficient_prefix_length"]:
        raise ValueError("sentence_outside_sufficient_prefix")
    sentence = sentences[sentence_index]
    if not sentence.strip() or sentence.strip() != str(tag.get("sentence", "")).strip():
        raise ValueError("sentence_text_does_not_match_processed_index")
    if sentence_index == trace["processed_answer_sentence_index"]:
        raise ValueError("answer_suffix_is_not_a_reasoning_sentence")
    prefix = "".join(sentences[:sentence_index])
    row_id = canonical_hash([trace["trace_id"], sentence_index])
    return {
        "schema_version": SCHEMA,
        "dataset_row_id": row_id,
        "problem_id": trace["problem_id"],
        "trace_id": trace["trace_id"],
        "question": trace["question"],
        "reference_answer": trace["reference_answer"],
        "unit_type": "reasoning_sentence_context",
        "sentence_id": f"T{sentence_index + 1}",
        "sentence_index": sentence_index,
        "sentence_text": sentence,
        "reasoning_prefix": prefix,
        "sentence_offsets": {
            "start_char_in_processed_cot": len(prefix),
            "end_char_in_processed_cot": len(prefix) + len(sentence),
        },
        "label": LABELS[removable],
        "label_id": int(removable),
        "activation_status": "not_extracted",
    }


def collect_records(root: Path, models=None, datasets=None):
    root = root.resolve()
    tag_files = sorted(root.glob("*/*_sentence_removability_tags_seed=*.json"))
    tag_files = [p for p in tag_files
                 if (not models or p.parent.name in models)
                 and (not datasets or p.name.split("_sentence_removability_tags")[0] in datasets)]
    if not tag_files:
        raise ValueError(f"No released sentence label files found under {root}")
    records, provenance, rejected, inventory = {}, {}, [], {}

    def remember(path, metadata):
        relative = path.relative_to(root).as_posix()
        if relative not in inventory:
            inventory[relative] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "metadata": metadata,
            }

    for tag_file in tag_files:
        tags, tag_metadata = load_bundle(tag_file)
        remember(tag_file, tag_metadata)
        dataset = tag_file.name.split("_sentence_removability_tags")[0]
        by_generation = defaultdict(list)
        for entry_key, entries in tags.items():
            generation_path, processed_path, attr_path, index = record_paths(root, entry_key)
            if generation_path.parent != tag_file.parent:
                raise ValueError(f"Tag model does not match its source trace: {entry_key}")
            by_generation[(generation_path, processed_path, attr_path)].append((entry_key, index, entries))
        for paths, entries in sorted(by_generation.items()):
            missing = [p.relative_to(root).as_posix() for p in paths if not p.is_file()]
            if missing:
                for key, _, items in entries:
                    for index in items:
                        rejected.append({"record_reference": key, "sentence_index": index,
                                         "reason": "missing_source_files", "missing": missing})
                continue
            (generations, generation_meta), (steps, step_meta), (attrs, attr_meta) = [
                load_bundle(path) for path in paths
            ]
            for path, metadata in zip(paths, (generation_meta, step_meta, attr_meta)):
                remember(path, metadata)
            model_id = generation_meta.get("model_id")
            if not model_id or (tag_metadata.get("model_id") not in (None, model_id)):
                raise ValueError(f"Inconsistent source model metadata in {tag_file}")
            ordered_generations = list(generations.values())
            for key, index, items in entries:
                try:
                    if index >= len(ordered_generations) or key not in steps or key not in attrs:
                        raise ValueError("missing_generation_or_processed_evidence")
                    trace = resolve_trace_context(key, ordered_generations[index], steps[key], attrs[key],
                                                  model_id, dataset)
                except ValueError as exc:
                    for raw_index in items:
                        rejected.append({"record_reference": key, "sentence_index": raw_index,
                                         "reason": str(exc)})
                    continue
                for raw_index, tag in items.items():
                    try:
                        if not re.fullmatch(r"\d+", raw_index):
                            raise ValueError("invalid_sentence_index")
                        row = make_training_record(tag, int(raw_index), trace)
                    except ValueError as exc:
                        rejected.append({"record_reference": key, "sentence_index": raw_index,
                                         "reason": str(exc)})
                        continue
                    prior = records.get(row["dataset_row_id"])
                    if prior and prior["label_id"] != row["label_id"]:
                        raise ValueError(f"Conflicting source labels for {key}:{raw_index}")
                    records.setdefault(row["dataset_row_id"], row)
                    provenance[row["dataset_row_id"]] = {
                        "dataset_row_id": row["dataset_row_id"],
                        "model_name": model_id,
                        "dataset_name": dataset,
                        "record_key": key,
                        "label_file": tag_file.relative_to(root).as_posix(),
                        "label_protocol": tag_metadata,
                    }
            print(f"Matched {paths[0].name}: {len(records)} valid labels so far", flush=True)
    return list(records.values()), provenance, rejected, inventory


def select_balanced(candidates, provenance, limit, seed):
    """Balance labels, spread sources and questions, without replacement."""
    if limit < 2 or limit % 2:
        raise ValueError("--limit must be a positive even number (balanced labels)")
    selected = []
    rng = random.Random(seed)
    for label_id in (0, 1):
        pool = sorted((r for r in candidates if r["label_id"] == label_id),
                      key=lambda r: r["dataset_row_id"])
        if len(pool) < limit // 2:
            raise ValueError(f"Need {limit // 2} rows for label {label_id}; found only {len(pool)}")
        buckets = defaultdict(lambda: defaultdict(list))
        for row in pool:
            source_info = provenance[row["dataset_row_id"]]
            stratum = (source_info["model_name"], source_info["dataset_name"])
            buckets[stratum][row["problem_id"]].append(row)
        queues = []
        for stratum in sorted(buckets):
            question_buckets = buckets[stratum]
            question_ids = sorted(question_buckets)
            rng.shuffle(question_ids)
            for rows in question_buckets.values():
                rng.shuffle(rows)
            queue = []
            while question_ids:
                next_round = []
                for problem_id in question_ids:
                    rows = question_buckets[problem_id]
                    queue.append(rows.pop())
                    if rows:
                        next_round.append(problem_id)
                question_ids = next_round
            queues.append(iter(queue))
        remaining = limit // 2
        while remaining:
            active = []
            for queue in queues:
                row = next(queue, None)
                if row is not None:
                    selected.append(row)
                    remaining -= 1
                    active.append(queue)
                    if not remaining:
                        break
            queues = active
    return selected


def assign_splits(rows, seed):
    """Assign 70/15/15 splits by question so related rows never leak across splits."""
    questions = sorted({row["problem_id"] for row in rows})
    if len(questions) < 3:
        raise ValueError("Need at least three distinct questions for question-disjoint splits")
    random.Random(seed).shuffle(questions)
    n_train = min(len(questions) - 2, max(1, int(len(questions) * 0.7)))
    n_validation = min(len(questions) - n_train - 1, max(1, int(len(questions) * 0.15)))
    names = (["train"] * n_train + ["validation"] * n_validation
             + ["test"] * (len(questions) - n_train - n_validation))
    mapping = dict(zip(questions, names))
    result = sorted(({**r, "split": mapping[r["problem_id"]]} for r in rows),
                    key=lambda r: (r["split"], r["problem_id"], r["trace_id"], r["sentence_index"]))
    for split in ("train", "validation", "test"):
        if {r["label_id"] for r in result if r["split"] == split} != {0, 1}:
            raise ValueError(f"{split} lacks one label class; use more questions or another seed")
    return result


def describe(rows):
    return {
        "row_count": len(rows),
        "question_count": len({row["problem_id"] for row in rows}),
        "trace_count": len({row["trace_id"] for row in rows}),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
    }


def build_dataset(source_dir, output_dir=DEFAULT_OUTPUT, *, source_output_dir=None,
                  limit=5000, seed=42, models=None, datasets=None):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    source_output_dir = (Path(source_output_dir) if source_output_dir else
                         output_dir.with_name(output_dir.name + "-source-notes"))
    if output_dir.exists():
        raise ValueError(f"Output already exists; choose a new --output-dir: {output_dir}")
    if source_output_dir.exists():
        raise ValueError(
            f"Source-notes output already exists; choose a new --source-output-dir: {source_output_dir}"
        )
    candidates, provenance, rejected, inventory = collect_records(source_dir, models, datasets)
    selected = assign_splits(select_balanced(candidates, provenance, limit, seed), seed)
    summary = {
        "status": "completed_before_activation_extraction",
        **describe(selected),
        "candidate_row_count": len(candidates),
        "rejected_row_count": len(rejected),
        "rejection_counts": dict(Counter(r["reason"] for r in rejected)),
        "splits": {split: describe([r for r in selected if r["split"] == split])
                   for split in ("train", "validation", "test")},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    source_output_dir.mkdir(parents=True, exist_ok=False)
    atomic_jsonl(output_dir / "sentence_training_data.jsonl", selected)
    atomic_jsonl(output_dir / "excluded_rows.jsonl", [
        {"record_id": canonical_hash(row), "reason": row["reason"]} for row in rejected
    ])
    source_counts = Counter(
        (item["model_name"], item["dataset_name"])
        for item in (provenance[row["dataset_row_id"]] for row in selected)
    )
    atomic_json(source_output_dir / "data_source.json", {
        "repository": "https://github.com/technion-cs-nlp/reasoning-importance",
        "source_files": inventory,
        "selected_rows_by_model_and_dataset": {
            f"{model} | {dataset}": count
            for (model, dataset), count in sorted(source_counts.items())
        },
        "record_level_crosswalk": False,
        "license_note": "DATA_SOURCE.md",
    })
    for split in ("train", "validation", "test"):
        atomic_jsonl(output_dir / "splits" / f"{split}.jsonl",
                     [row for row in selected if row["split"] == split])
    atomic_json(output_dir / "summary.json", summary)
    files = {
        path.relative_to(output_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.rglob("*.json*"))
    }
    atomic_json(output_dir / "dataset_manifest.json", {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_disclosure": "Imported-label provenance is recorded separately at dataset level; no row-level source crosswalk is distributed.",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": seed, "requested_rows": limit,
        "label_mapping": {"false": LABELS[False], "true": LABELS[True]},
        "selection": "balanced labels; round-robin data-source strata and questions",
        "split_strategy": "70/15/15 of normalized unique question texts",
        "activation_extraction_performed": False,
        "file_sha256": files,
        "limitations": [
            "Transferred labels have not been validated for Qwen3-4B.",
            "Mapped label names do not assert that PSSN's 2-of-3 removal criterion was used.",
            "Missing per-row sufficiency, attainability and model revisions are not fabricated.",
            "Question grouping normalizes Unicode and whitespace; paraphrase overlap is not detected.",
        ],
    })
    readme = f"""# PSSN sentence training data

{len(selected):,} sentence-context examples in the PSSN training schema.

The training rows contain model-facing fields. Dataset-level source and reuse
details are stored separately in the sibling `{source_output_dir.name}` folder.
There is no row-ID-to-source-record crosswalk.

`sentence_training_data.jsonl` contains all selected rows.
`splits/` contains question-disjoint train, validation and test rows.
`excluded_rows.jsonl` contains hashed record identifiers and rejection reasons.

True maps to low_value_in_context (1); false maps to answer_sensitive_in_context (0).
These are transferred labels, not labels produced by PSSN's 2-of-3 rule.

Sentence indices are zero-based; exported T1/T2 IDs are one-based. Offsets refer
to the processed sentence list concatenated.

No activations or new answers have been generated. Each row provides the question,
earlier reasoning text and target sentence needed for a later causal forward pass.
When extracting Qwen3 activations, render a fresh Qwen3 prompt from `question`,
then append `reasoning_prefix` and `sentence_text`. Never append future sentences
or `reference_answer` in the feature context. These transferred labels should be
evaluated on held-out questions before drawing conclusions.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    source_note = """# Data source and reuse notes

This dataset imports sentence-level labels and associated reasoning context from
the public release accompanying *Reasoning Models Know What's Important, and Encode
It in Their Activations*: https://github.com/technion-cs-nlp/reasoning-importance

The release uses HARP and MATH-500 and includes model-generated reasoning outputs.
The research repository has no explicit repository license file. HARP and the
original MATH repository list MIT licenses, but that does not by itself license
the research team's derived labels, generated traces, or every upstream question.
Check each model's terms and obtain permission before publicly redistributing the
imported records.

Source files, aggregate source counts, and reuse notes are stored separately.
No row-level mapping between training examples and source records is distributed.
The binary target is mapped to PSSN label names for model-input convenience only;
the dataset is not being presented as newly PSSN-generated labels.
The imported label rule is not equivalent to PSSN's 2-of-3 labeling rule. Cite
the paper and identify this set as transferred data in any analysis or publication.
This note records provenance and is not legal advice.
"""
    (source_output_dir / "DATA_SOURCE.md").write_text(source_note, encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-output-dir", type=Path,
                        help="Separate sibling directory for aggregate source and reuse notes")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", help="Optional source model directory names")
    parser.add_argument("--datasets", nargs="+", choices=("harp-standard", "math-500"))
    args = parser.parse_args(argv)
    if args.limit < 2 or args.limit % 2:
        parser.error("--limit must be a positive even number")
    try:
        summary = build_dataset(args.source_dir, args.output_dir,
                                source_output_dir=args.source_output_dir, limit=args.limit,
                                seed=args.seed, models=args.models, datasets=args.datasets)
    except ValueError as exc:
        parser.exit(1, f"Dataset preparation failed: {exc}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
