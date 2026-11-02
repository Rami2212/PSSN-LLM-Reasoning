"""Live test on unseen GSM8K questions: full depth vs text-controller generation."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .controller_common import (DEPTHS, Progress, atomic_json, checked_trial, digest,
    fingerprint, generate, load_controller, load_model, manifest, read_json,
    record, route_depth, text_key, warmup)


def choose_questions(payload, rows, seed):
    from datasets import load_dataset
    kwargs = {"revision": payload["dataset_revision"]}
    # Exclude every training-split question, including label candidates and demos.
    training = load_dataset("openai/gsm8k", "main", split="train", **kwargs)
    excluded = {text_key(r["question"]) for r in training}
    excluded.update(payload["excluded_question_keys"])
    data = load_dataset("openai/gsm8k", "main", split="test", **kwargs)
    indices = list(range(len(data)))
    random.Random(seed).shuffle(indices)
    chosen = []
    for index in indices:
        row = record(index, data[index], "test")
        if row["question_key"] not in excluded:
            chosen.append(row)
            excluded.add(row["question_key"])
        if len(chosen) == rows:
            return chosen
    raise ValueError("Not enough disjoint unseen questions")


def summarize(trials, total_questions):
    metrics = {}
    for mode in ("full_depth", "controller"):
        rows = [r for r in trials if r["mode"] == mode]
        count = len(rows)
        generation_seconds = sum(r["seconds"] for r in rows)
        routing_seconds = sum(r["routing_seconds"] for r in rows)
        elapsed = generation_seconds + routing_seconds
        tokens = sum(r["output_tokens"] for r in rows)
        metrics[mode] = {"questions": count, "correct": sum(r["correct"] for r in rows),
                         "accuracy": sum(r["correct"] for r in rows) / count if count else None,
                         "complete": sum(r["complete"] for r in rows),
                         "output_tokens": tokens, "input_tokens": sum(r["input_tokens"] for r in rows),
                         "generation_seconds": generation_seconds, "routing_seconds": routing_seconds,
                         "total_seconds": elapsed, "mean_seconds": elapsed / count if count else None,
                         "output_tokens_per_second": tokens / elapsed if elapsed else None,
                         "depth_counts": {str(d): sum(r["depth"] == d for r in rows) for d in DEPTHS},
                         "decoder_layer_token_evaluations": sum(r["decoder_layer_token_evaluations"] for r in rows),
                         "max_peak_allocated_gib": max((r["peak_allocated_gib"] for r in rows), default=None),
                         "max_peak_reserved_gib": max((r["peak_reserved_gib"] for r in rows), default=None)}
    by_key = {(r["question_id"], r["mode"]): r for r in trials}
    ids = sorted({r["question_id"] for r in trials})
    pairs = [(by_key[(q, "full_depth")], by_key[(q, "controller")]) for q in ids
             if (q, "full_depth") in by_key and (q, "controller") in by_key]
    baseline_time = sum(a["seconds"] for a, _ in pairs)
    controller_time = sum(b["seconds"] + b["routing_seconds"] for _, b in pairs)
    baseline_work = sum(a["decoder_layer_token_evaluations"] for a, _ in pairs)
    chosen_work = sum(b["decoder_layer_token_evaluations"] for _, b in pairs)
    return {"status": "completed" if len(pairs) == total_questions else "running",
            "completed_pairs": len(pairs), "total_questions": total_questions, "metrics": metrics,
            "paired": {"correct_to_wrong": sum(a["correct"] and not b["correct"] for a, b in pairs),
                       "wrong_to_correct": sum(not a["correct"] and b["correct"] for a, b in pairs),
                       "baseline_seconds_divided_by_controller_seconds": baseline_time / controller_time if controller_time else None,
                       "output_token_difference": sum(b["output_tokens"] - a["output_tokens"] for a, b in pairs),
                       "estimated_layer_token_work_reduction": 1 - chosen_work / baseline_work if baseline_work else None},
            "notes": ["Random unseen test questions: no filtering by correctness or class balance.",
                      "Controller sees question text only; predictions freeze before any answer generation.",
                      "Routing uses CPU TF-IDF/logistic regression; its latency is included in controller total time.",
                      "Depth is fixed per question for prompt and all generated tokens; no speculative decoding.",
                      "Layer-token counts estimate work, not measured FLOPs. All model weights stay on GPU.",
                      "Single greedy outcome per method; cap and missing answers count as incorrect. No test-based tuning."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    payload = load_controller(args.controller)
    if (payload["common_code_sha256"] != digest(Path(__file__).with_name("controller_common.py"))
            or payload["smoke_code_sha256"] != digest(Path(__file__).with_name("smoke.py"))):
        raise ValueError("Labeling/generation code changed since training; use the archived source version")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with Progress(args.output_dir) as progress:
        import torch
        progress.update(phase="preparing_unseen_questions")
        questions_path = args.output_dir / "questions.json"
        questions = read_json(questions_path) if questions_path.exists() else choose_questions(payload, args.rows, args.seed)
        if len(questions) != args.rows:
            raise ValueError("Saved question count differs; choose a new output directory")
        if any(q["question_key"] in payload["excluded_question_keys"] for q in questions):
            raise ValueError("Unseen questions overlap controller preparation data")
        config = {"checkpoint_sha256": digest(args.controller), "question_sha256": fingerprint(questions),
                  "rows": args.rows, "seed": args.seed, "model_revision": payload["model_revision"],
                  "dataset_revision": payload["dataset_revision"], "max_new_tokens": payload["max_new_tokens"],
                  "examples_sha256": fingerprint(payload["demonstrations"]),
                  "order": "alternate baseline/controller first by question index"}
        source = manifest(args.output_dir, config, "test", require_same_environment=True)
        atomic_json(questions_path, questions)
        examples = payload["demonstrations"]
        progress.update(phase="loading_model")
        tokenizer, model = load_model(payload["model_revision"], payload["generation_seed"])
        progress.update(phase="warmup")
        warmup(tokenizer, model, examples)
        route_depth(payload["pipeline"], "Warmup arithmetic question")
        trials = []
        for index, question in enumerate(questions):
            folder = args.output_dir / "questions" / question["question_id"]
            route_path = folder / "decision.json"
            if route_path.exists():
                decision = read_json(route_path)
                if decision["run_identity"] != source["identity"]:
                    raise ValueError("Mismatched saved controller decision")
            else:
                started = time.perf_counter()
                selected, probabilities = route_depth(payload["pipeline"], question["question"])
                routing_seconds = time.perf_counter() - started
                decision = {"run_identity": source["identity"], "question_id": question["question_id"],
                            "selected_depth": selected, "routing_seconds": routing_seconds,
                            "predicted_answer_preservation_probability_by_depth":
                            {str(d): probabilities[d] for d in DEPTHS},
                            "routing_rule": "shallowest predicted safe depth at probability >= 0.5; fallback depth 32"}
                atomic_json(route_path, decision)
            order = ["full_depth", "controller"] if index % 2 == 0 else ["controller", "full_depth"]
            for mode in order:
                depth = 32 if mode == "full_depth" else decision["selected_depth"]
                path = folder / f"{mode}.json"
                progress.update(phase="generating", current_question=question["question_id"],
                                current_mode=mode, completed_trials=len(trials))
                if not path.exists():
                    trial = generate(tokenizer, model, question, examples, depth, payload["max_new_tokens"])
                    trial.update(run_identity=source["identity"], mode=mode,
                                 routing_seconds=0.0 if mode == "full_depth" else decision["routing_seconds"])
                    atomic_json(path, trial)
                trial = checked_trial(path, question, source["identity"], depth)
                if trial.get("mode") != mode:
                    raise ValueError("Saved trial has the wrong method")
                trials.append(trial)
                atomic_json(args.output_dir / "summary.json", summarize(trials, len(questions)))
                progress.update(completed_trials=len(trials))
                print(f"{question['question_id']} {mode} depth={depth} correct={trial['correct']} "
                      f"finish={trial['finish_reason']} tokens={trial['output_tokens']} seconds={trial['seconds']:.2f}", flush=True)
        print(json.dumps(summarize(trials, len(questions)), indent=2), flush=True)


if __name__ == "__main__":
    main()
