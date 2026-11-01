"""Run the SAME training questions at 12/24/36 layers; save every trial."""
from __future__ import annotations

import argparse
import json

from pathlib import Path
from .common import (DEPTHS, MODEL_REVISION, Progress, atomic_json, digest, identity,
                     load_model, manifest, read_json, select_questions, settings,
                     training_target, warmup, generate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    config = settings(args.revision, args.max_new_tokens, args.seed)
    if (args.output_dir / "manifest.json").exists():
        old = read_json(args.output_dir / "manifest.json")["config"]
        records = read_json(args.output_dir / "questions.json")
        dataset_revision = old["dataset_revision"]
        if args.dataset_revision and args.dataset_revision != dataset_revision:
            raise ValueError("Dataset revision changed; use a new directory")
    else:
        records, dataset_revision = select_questions("train", args.rows, args.seed, args.offset, args.dataset_revision)
    tokenizer, model = load_model(args.revision)
    manifest(args.output_dir, {"generation": config, "rows": args.rows, "offset": args.offset,
                              "source_dataset": "openai/gsm8k", "dataset_config": "main",
                              "split": "train", "dataset_revision": dataset_revision,
                              "questions_sha256": identity(records)}, "label")
    if len(records) != args.rows:
        raise ValueError("Question count differs from requested rows")
    atomic_json(args.output_dir / "questions.json", records)
    completed, counts, excluded = 0, {str(d): 0 for d in DEPTHS}, 0
    with Progress(args.output_dir) as progress:
        progress.update(phase="warmup", completed_questions=0, total_questions=len(records))
        warmup(tokenizer, model, config)
        for question in records:
            folder = args.output_dir / "questions" / question["question_id"]
            outcomes, hashes = {}, {}
            # Collect every depth even if the baseline fails, retaining the full audit.
            for depth in (36, 12, 24):
                path = folder / f"depth_{depth}.json"
                progress.update(phase="generating", current_question=question["question_id"],
                                current_depth=depth, completed_questions=completed)
                if not path.exists():
                    atomic_json(path, generate(tokenizer, model, question, depth, config))
                trial = read_json(path)
                if trial["depth"] != depth or trial["question_id"] != question["question_id"]:
                    raise ValueError("Saved trial has wrong question or depth")
                outcomes[str(depth)] = {key: trial[key] for key in ("correct", "complete", "finish_reason", "extracted_answer")}
                hashes[str(depth)] = digest(path)
                print(f"{question['question_id']} / depth {depth}: "
                      f"correct={trial['correct']} complete={trial['complete']} tokens={trial['output_tokens']}", flush=True)
            target, reason = training_target(outcomes)
            row = {"question_id": question["question_id"], "question_key": question["question_key"],
                   "target_depth": target, "label_reason": reason, "outcomes": outcomes,
                   "trial_sha256": hashes}
            atomic_json(folder / "label.json", row)
            if target is None:
                excluded += 1
            else:
                counts[str(target)] += 1
            completed += 1
            summary = {"completed_questions": completed, "total_questions": len(records),
                       "generation_trials": completed * 3, "target_depth_counts": counts,
                       "excluded_full_depth_not_complete_correct": excluded,
                       "controller_trainable_classes": sum(v > 0 for v in counts.values()),
                       "shallow_targets": counts["12"] + counts["24"],
                       "trials_per_question_depth": 1,
                       "label_reliability": "One greedy outcome per depth; observed correctness is not a safety guarantee.",
                       "early_exits_adapted": False}
            atomic_json(args.output_dir / "summary.json", summary)
            progress.update(completed_questions=completed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
