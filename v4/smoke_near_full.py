"""Check frozen Qwen3 at 28/32/36 layers on the existing five smoke questions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import (MODEL_REVISION, NEAR_FULL_DEPTHS, Progress, atomic_json,
                     identity, load_model, manifest, read_json, settings,
                     warmup, generate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-file", type=Path, required=True,
                        help="questions.json saved by the original five-question V4 smoke run")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Use a NEW folder, e.g. smoke-near-full")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    questions = read_json(args.questions_file)
    if len(questions) != 5:
        raise ValueError(f"Expected the same five smoke questions, found {len(questions)}")
    if len({q["question_id"] for q in questions}) != 5:
        raise ValueError("Question IDs must be unique")

    config = settings(args.revision, args.max_new_tokens, seed=42)
    config["depths"] = list(NEAR_FULL_DEPTHS)
    config["protocol"] = "v4-frozen-qwen-near-full-depth-smoke-28-32-36-v1"
    config["question_source_sha256"] = identity(questions)
    config["question_source"] = str(args.questions_file.resolve())
    config["rows"] = len(questions)
    config["trial_order"] = [36, 28, 32]
    manifest(args.output_dir, {"generation": config}, "smoke_near_full")
    atomic_json(args.output_dir / "questions.json", questions)

    tokenizer, model = load_model(args.revision)
    totals = {str(d): {"correct": 0, "complete": 0, "output_tokens": 0,
                       "seconds": 0.0} for d in NEAR_FULL_DEPTHS}
    trials = 0
    with Progress(args.output_dir) as progress:
        progress.update(phase="warmup", completed_trials=0,
                       total_trials=len(questions) * len(NEAR_FULL_DEPTHS))
        warmup(tokenizer, model, config, NEAR_FULL_DEPTHS)
        for index, question in enumerate(questions, start=1):
            folder = args.output_dir / "questions" / question["question_id"]
            for depth in (36, 28, 32):
                path = folder / f"depth_{depth}.json"
                progress.update(phase="generating", current_question=question["question_id"],
                                current_depth=depth, completed_trials=trials)
                if not path.exists():
                    atomic_json(path, generate(tokenizer, model, question, depth, config))
                trial = read_json(path)
                if trial["question_id"] != question["question_id"] or trial["depth"] != depth:
                    raise ValueError(f"Saved result does not match question/depth: {path}")
                item = totals[str(depth)]
                item["correct"] += int(trial["correct"])
                item["complete"] += int(trial["complete"])
                item["output_tokens"] += trial["output_tokens"]
                item["seconds"] += trial["seconds"]
                trials += 1
                print(f"{question['question_id']} / depth {depth}: "
                      f"correct={trial['correct']} complete={trial['complete']} "
                      f"tokens={trial['output_tokens']} seconds={trial['seconds']:.2f}", flush=True)

            summary = {
                "status": "running", "completed_questions": index,
                "total_questions": len(questions), "completed_trials": trials,
                "total_trials": len(questions) * len(NEAR_FULL_DEPTHS),
                "depth_results": _finalize(totals, len(questions)),
                "note": "Five-question diagnostic only; frozen Qwen was not adapted for early exits.",
            }
            atomic_json(args.output_dir / "summary.json", summary)
            progress.update(completed_trials=trials, completed_questions=index)

    summary["status"] = "completed"
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def _finalize(totals, question_count):
    result = {}
    for depth, values in totals.items():
        result[depth] = {
            **values,
            "accuracy": values["correct"] / question_count,
            "completion_rate": values["complete"] / question_count,
            "mean_seconds": values["seconds"] / question_count,
        }
    return result


if __name__ == "__main__":
    main()
