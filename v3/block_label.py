"""Label fixed, non-overlapping four-MLP block bypasses on GSM8K."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .common import (MODEL_ID, atomic_json, digest, generation_args,
                     load_model, questions, read_json, trial_label)
from .block_common import (BLOCK4_PROTOCOL, generate_group, make_groups,
                           prepare_manifest, status)


def collect(args, excluded_keys=()):
    if (args.output_dir / "manifest.json").exists() and (args.output_dir / "questions.json").exists():
        saved = read_json(args.output_dir / "manifest.json")
        records = read_json(args.output_dir / "questions.json")
        dataset_revision = saved["config"]["dataset_revision"]
        if args.dataset_revision and args.dataset_revision != dataset_revision:
            raise ValueError("Dataset revision changed; use a new output directory")
        if set(excluded_keys) & {r["question_key"] for r in records}:
            raise ValueError("Resume questions overlap previously used questions")
    else:
        records, dataset_revision = questions(
            args.split, args.rows, args.offset, args.dataset_revision, args.seed, excluded_keys,
        )
    groups = [list(group) for group in make_groups(36)]
    config = {
        "protocol": BLOCK4_PROTOCOL, "model_id": MODEL_ID, "model_revision": args.revision,
        "source_dataset": "openai/gsm8k", "source_config": "main",
        "dataset_revision": dataset_revision, "split": args.split,
        "rows": args.rows, "offset": args.offset, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "do_sample": False,
        "enable_thinking": False, "precision": "float16", "attention": "sdpa",
        "question_keys": [r["question_key"] for r in records],
        "questions_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        "label_definition": "1 only when complete correct answer preserved; all unscored outcomes mapped to unsafe 0",
        "candidate_policy": "9 non-overlapping consecutive four-MLP groups: (0-3), (4-7), ..., (32-35)",
        "candidate_groups": groups, "trials_per_question_group": 1,
        "routing_implementation": "torch.cond+torch.compile; device-resident predicate; no CPU selection",
        "unscored_policy": "map -1 to 0 (unsafe/not approved), preserve original label and reason for audit",
    }
    prepare_manifest(args.output_dir, config)
    atomic_json(args.output_dir / "questions.json", records)
    tokenizer, model = load_model(args.revision)
    layer_count, width = len(model.model.layers), model.config.hidden_size
    candidate_groups = make_groups(layer_count)
    if [list(group) for group in candidate_groups] != groups:
        raise ValueError("This pilot expects Qwen3-4B with exactly 36 decoder MLPs")
    generate_group(tokenizer, model, records[0], 2, args.seed)
    total_trials = len(records) * (len(candidate_groups) + 1)
    completed_questions = 0
    try:
        for q in records:
            folder = args.output_dir / "questions" / q["question_id"]
            folder.mkdir(parents=True, exist_ok=True)
            features_path, baseline_path = folder / "features.npz", folder / "baseline.json"
            if baseline_path.exists() and features_path.exists():
                baseline = read_json(baseline_path)
            else:
                status(args.output_dir, status="running", phase="baseline", question=q["question_id"],
                       completed_questions=completed_questions, total_questions=len(records))
                baseline, features = generate_group(
                    tokenizer, model, q, args.max_new_tokens, args.seed, capture=True,
                )
                temp = folder / "features.tmp.npz"
                np.savez_compressed(temp, features=features.astype(np.float32))
                temp.replace(features_path)
                atomic_json(baseline_path, baseline)
            raw_labels, reasons = [], []
            if not baseline["complete"] or not baseline["correct"]:
                reason = "baseline_incomplete_or_incorrect"
                raw_labels = [-1] * len(candidate_groups)
                reasons = [reason] * len(candidate_groups)
                atomic_json(folder / "labels.json", {
                    "question_id": q["question_id"], "question_key": q["question_key"],
                    "action_type": "consecutive_four_mlp_group",
                    "groups": [list(group) for group in candidate_groups],
                    "labels": [0] * len(candidate_groups), "raw_labels": raw_labels,
                    "label_reasons": reasons,
                    "feature_sha256": digest(features_path),
                    "unscored_policy": "-1 mapped to 0 (unsafe/not approved)",
                })
                completed_questions += 1
                print(f"{q['question_id']}: baseline unusable; all groups marked unsafe (0).", flush=True)
                continue
            for group_index, group in enumerate(candidate_groups):
                group_id = f"G{group[0]:02d}_{group[-1]:02d}"
                trial_path = folder / f"skip_{group_id}.json"
                status(args.output_dir, status="running", phase="group_bypass",
                       question=q["question_id"], group=list(group), group_index=group_index,
                       completed_questions=completed_questions, total_questions=len(records))
                if trial_path.exists():
                    result = read_json(trial_path)
                else:
                    result, _ = generate_group(
                        tokenizer, model, q, args.max_new_tokens, args.seed, forced_group=group,
                    )
                    atomic_json(trial_path, result)
                label, reason = trial_label(baseline, result)
                raw_labels.append(label)
                reasons.append(reason)
                used_label = 0 if label < 0 else label
                print(f"{q['question_id']} / MLP group {group[0]:02d}-{group[-1]:02d}: "
                      f"label={used_label} (raw={label}; {reason})", flush=True)
            atomic_json(folder / "labels.json", {
                "question_id": q["question_id"], "question_key": q["question_key"],
                "action_type": "consecutive_four_mlp_group",
                "groups": [list(group) for group in candidate_groups],
                "labels": [0 if label < 0 else label for label in raw_labels],
                "raw_labels": raw_labels, "label_reasons": reasons,
                "feature_sha256": digest(features_path),
                "unscored_policy": "-1 mapped to 0 (unsafe/not approved); raw_labels retain original outcomes",
                "note": "One observed isolated four-MLP group outcome per question; no combined-group guarantee.",
            })
            completed_questions += 1
        summary = summarize(args.output_dir, records, candidate_groups, width)
        status(args.output_dir, status="completed", phase="finished",
               completed_questions=completed_questions, total_questions=len(records),
               total_trials=total_trials)
        return summary
    except BaseException as exc:
        status(args.output_dir, status="interrupted", phase="saved_trials_retained",
               completed_questions=completed_questions, total_questions=len(records),
               total_trials=total_trials, error=type(exc).__name__)
        raise


def summarize(directory, records, groups, width):
    counts = {"safe": 0, "unsafe": 0, "originally_unscored_mapped_to_unsafe": 0}
    baseline_correct = 0
    for q in records:
        folder = Path(directory) / "questions" / q["question_id"]
        baseline_correct += int(read_json(folder / "baseline.json")["correct"])
        labels_path = folder / "labels.json"
        if labels_path.exists():
            labels = read_json(labels_path)
            for label in labels["labels"]:
                counts["safe" if label == 1 else "unsafe"] += 1
            counts["originally_unscored_mapped_to_unsafe"] += sum(
                label < 0 for label in labels.get("raw_labels", labels["labels"])
            )
    result = {
        "protocol": BLOCK4_PROTOCOL, "questions": len(records), "layers": len(groups) * 4,
        "groups_per_question": len(groups), "hidden_size": width,
        "generation_runs": len(records) * (len(groups) + 1),
        "baseline_correct": baseline_correct, "label_counts": counts,
        "groups": [list(group) for group in groups],
        "feature": "concatenated last-prompt-token MLP inputs for all four layers in a group",
        "unscored_policy": "All -1 outcomes count as unsafe (0); original labels/reasons are retained.",
        "limitation": "Group labels cannot identify which member caused a failure; combined groups were not tested.",
    }
    atomic_json(Path(directory) / "summary.json", result)
    return result


def main():
    parser = generation_args()
    parser.set_defaults(rows=100)
    args = parser.parse_args()
    print(json.dumps(collect(args), indent=2))


if __name__ == "__main__":
    main()
