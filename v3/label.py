"""One deterministic baseline + one bypass run per candidate layer/question.

Run: python -m v3.label --output-dir /path/labels --rows 50
The full baseline and all 36 bypass responses are saved. Each bypass applies
only after full prompt prefill and always affects one MLP, not combinations.
"""
from __future__ import annotations

import os
import hashlib
import json

import numpy as np

from .common import (MODEL_ID, PROTOCOL, atomic_json, digest, generate,
                     generation_args, load_model, prepare_manifest, questions,
                     read_json, status, trial_label)


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
    config = {
        "protocol": PROTOCOL, "model_id": MODEL_ID, "model_revision": args.revision,
        "source_dataset": "openai/gsm8k", "source_config": "main",
        "dataset_revision": dataset_revision, "split": args.split,
        "rows": args.rows, "offset": args.offset, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "do_sample": False,
        "enable_thinking": False, "precision": "float16", "attention": "sdpa",
        "question_keys": [r["question_key"] for r in records],
        "questions_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        "checkpoint_sha256": digest(args.checkpoint) if getattr(args, "checkpoint", None) else None,
        "label_definition": "complete correct bypass given complete correct full baseline",
        "trials_per_question_layer": 1,
    }
    prepare_manifest(args.output_dir, config)
    atomic_json(args.output_dir / "questions.json", records)
    tokenizer, model = load_model(args.revision)
    layer_count, width = len(model.model.layers), model.config.hidden_size
    # Compile kernels and populate allocator before recording any trial timings.
    generate(tokenizer, model, records[0], 2, args.seed)
    completed = 0
    trials_done = 0
    total_trials = len(records) * (layer_count + 1)
    try:
        for q in records:
            directory = args.output_dir / "questions" / q["question_id"]
            directory.mkdir(parents=True, exist_ok=True)
            features_path = directory / "features.npz"
            baseline_path = directory / "baseline.json"
            if baseline_path.exists() and features_path.exists():
                baseline = read_json(baseline_path)
            else:
                status(args.output_dir, status="running", phase="baseline", question=q["question_id"],
                       saved_trials=trials_done, total_trials=total_trials)
                baseline, features = generate(tokenizer, model, q, args.max_new_tokens, args.seed, capture=True)
                temp = directory / "features.tmp.npz"
                np.savez_compressed(temp, features=features.astype(np.float32))
                os.replace(temp, features_path)
                atomic_json(baseline_path, baseline)
            trials_done += 1
            labels, reasons = [], []
            for layer in range(layer_count):
                path = directory / f"skip_{layer:02d}.json"
                status(args.output_dir, status="running", phase="bypass", question=q["question_id"],
                       layer=layer, saved_trials=trials_done, total_trials=total_trials)
                if path.exists():
                    result = read_json(path)
                else:
                    result, _ = generate(tokenizer, model, q, args.max_new_tokens, args.seed, forced=layer)
                    atomic_json(path, result)
                label, reason = trial_label(baseline, result)
                labels.append(label)
                reasons.append(reason)
                trials_done += 1
                print(f"{q['question_id']} / MLP {layer:02d}: label={label} ({reason})", flush=True)
            atomic_json(directory / "labels.json", {
                "question_id": q["question_id"], "question_key": q["question_key"],
                "labels": labels, "label_reasons": reasons,
                "feature_sha256": digest(features_path),
                "note": "Single-run observed safety for this question; not a guarantee on other questions or combined skips.",
            })
            completed += 1
        summary = summarize(args.output_dir, records, layer_count, width)
        status(args.output_dir, status="completed", phase="finished", completed_questions=completed,
               saved_trials=trials_done, total_trials=total_trials)
        return summary
    except BaseException as exc:
        status(args.output_dir, status="interrupted", phase="saved_trials_retained",
               completed_questions=completed, saved_trials=trials_done, total_trials=total_trials,
               error=type(exc).__name__)
        raise


def summarize(directory, records, layers, width):
    counts = {"safe": 0, "unsafe": 0, "unscored": 0}
    baseline_correct = 0
    for q in records:
        folder = directory / "questions" / q["question_id"]
        baseline_correct += int(read_json(folder / "baseline.json")["correct"])
        for label in read_json(folder / "labels.json")["labels"]:
            counts[{1: "safe", 0: "unsafe", -1: "unscored"}[label]] += 1
    result = {"questions": len(records), "layers": layers, "hidden_size": width,
              "generation_runs": len(records) * (layers + 1), "baseline_correct": baseline_correct,
              "label_counts": counts, "feature": "last prompt token at each MLP input",
              "warmup_generations_excluded": 1,
              "limitation": "One observed outcome per question/layer; no causal per-token label or combined-skip guarantee."}
    atomic_json(directory / "summary.json", result)
    return result


def main():
    args = generation_args().parse_args()
    print(json.dumps(collect(args), indent=2))


if __name__ == "__main__":
    main()
