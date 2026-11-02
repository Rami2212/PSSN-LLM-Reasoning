"""Label 100 GSM8K questions at each of 3 fixed depths (300 generations total)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .controller_common import (DATASET_REVISION, DEPTHS, MODEL_ID, MODEL_REVISION,
    Progress, atomic_json, checked_trial, digest, fingerprint, generate, load_model,
    make_prompt, manifest, prepare_candidates, read_json, warmup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=100,
                        help="Number of distinct questions; each gets one generation at each depth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    args = parser.parse_args()
    if min(args.rows, args.max_new_tokens) < 1:
        parser.error("row count and token limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with Progress(args.output_dir) as progress:
        import torch  # Include GPU metadata in the manifest before loading model weights.
        progress.update(phase="preparing_questions")
        candidates_path = args.output_dir / "questions.json"
        if candidates_path.exists():
            questions = read_json(candidates_path)
            examples = read_json(args.output_dir / "demonstrations.json")
            if len(questions) != args.rows:
                raise ValueError("Saved question count differs; choose a new output directory")
        else:
            candidates, examples = prepare_candidates(args.seed, args.dataset_revision)
            if len(candidates) < args.rows:
                raise ValueError("Dataset has fewer unique eligible questions than requested")
            questions = candidates[:args.rows]
        config = {"model_id": MODEL_ID, "model_revision": args.revision,
                  "dataset_revision": args.dataset_revision, "dataset_split": "train",
                  "seed": args.seed, "question_count": args.rows,
                  "max_new_tokens": args.max_new_tokens, "depths": list(DEPTHS),
                  "question_sha256": fingerprint(questions), "examples_sha256": fingerprint(examples),
                  "label": "complete-and-correct answer at each independently tested depth",
                  "sampling": "first seeded unique GSM8K training questions after fixed exclusions",
                  "generation": "one greedy DIRECT fixed-prefix generation per question/depth; no speculation"}
        source = manifest(args.output_dir, config, "label", require_same_environment=True)
        atomic_json(candidates_path, questions)
        atomic_json(args.output_dir / "demonstrations.json", examples)
        tokenizer = model = None
        labeled = []
        total_trials = len(questions) * len(DEPTHS)
        for index, question in enumerate(questions):
            folder = args.output_dir / "questions" / question["question_id"]
            atomic_json(folder / "prompt.json", {"prompt": make_prompt(question, examples)})
            outcomes, hashes = {}, {}
            depth_order = list(DEPTHS)
            shift = index % len(depth_order)
            depth_order = depth_order[shift:] + depth_order[:shift]
            for depth_index, depth in enumerate(depth_order):
                path = folder / f"depth_{depth}.json"
                progress.update(phase="generating", current_question=question["question_id"],
                                current_depth=depth, completed_questions=index,
                                completed_trials=index * len(DEPTHS) + depth_index,
                                total_trials=total_trials)
                if not path.exists():
                    if model is None:
                        progress.update(phase="loading_model")
                        tokenizer, model = load_model(args.revision, args.seed)
                        progress.update(phase="warmup")
                        warmup(tokenizer, model, examples)
                    trial = generate(tokenizer, model, question, examples, depth, args.max_new_tokens)
                    trial["run_identity"] = source["identity"]
                    atomic_json(path, trial)
                trial = checked_trial(path, question, source["identity"], depth)
                outcomes[str(depth)] = {key: trial[key] for key in
                    ("correct", "complete", "finish_reason", "extracted_answer", "input_tokens",
                     "output_tokens", "seconds", "peak_allocated_gib", "peak_reserved_gib")}
                hashes[str(depth)] = digest(path)
                progress.update(completed_trials=index * len(DEPTHS) + depth_index + 1)
            label = {**question, "outcomes": outcomes, "trial_sha256": hashes,
                     "label_note": "Binary observed answer-preservation labels; incomplete generations count as unsafe."}
            labeled.append(label)
            atomic_json(folder / "label.json", label)
            atomic_json(args.output_dir / "selected_dataset.json", labeled)
            summary = {"status": "running", "questions_labeled": len(labeled),
                       "requested_questions": args.rows, "generation_trials": len(labeled) * len(DEPTHS),
                       "depth_counts": {str(d): len(labeled) for d in DEPTHS},
                       "answer_preserved_counts": {str(d): sum(r["outcomes"][str(d)]["correct"] and
                                                                 r["outcomes"][str(d)]["complete"]
                                                                 for r in labeled) for d in DEPTHS},
                       "incomplete_counts": {str(d): sum(not r["outcomes"][str(d)]["complete"]
                                                         for r in labeled) for d in DEPTHS},
                       "labeling_unit": "question-depth pair", "trials_per_question_depth": 1,
                       "limitation": "One greedy generation per depth is a noisy binary outcome, not a safety guarantee."}
            atomic_json(args.output_dir / "summary.json", summary)
            progress.update(questions_labeled=len(labeled), completed_trials=len(labeled) * len(DEPTHS))
            print(f"{question['question_id']} labeled at depths {list(DEPTHS)} "
                  f"({len(labeled)}/{args.rows} questions)", flush=True)
        summary["status"] = "completed"
        atomic_json(args.output_dir / "summary.json", summary)
    progress.update(status="completed", phase="finished")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
