"""Paired fresh GSM8K test generation: full Qwen vs question-text routing."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .common import (DEPTHS, Progress, atomic_json, digest, generate, identity,
                     load_controller, load_model, manifest, read_json, route,
                     select_questions, warmup)


def aggregate(rows, routed):
    times = [r["seconds"] + (r.get("controller_seconds", 0) if routed else 0) for r in rows]
    layer_work = sum(r["decoder_layer_token_evaluations"] for r in rows)
    full_equivalent = sum(36 * (r["input_tokens"] + max(r["output_tokens"] - 1, 0)) for r in rows)
    return {"questions": len(rows), "correct": sum(r["correct"] for r in rows),
            "accuracy_all_questions": sum(r["correct"] for r in rows) / len(rows),
            "completed": sum(r["complete"] for r in rows),
            "total_seconds": sum(times), "mean_seconds": statistics.mean(times),
            "median_seconds": statistics.median(times),
            "controller_seconds": sum(r.get("controller_seconds", 0) for r in rows) if routed else 0,
            "input_tokens": sum(r["input_tokens"] for r in rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "aggregate_output_tokens_per_second": sum(r["output_tokens"] for r in rows) / sum(times),
            "max_peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows),
            "max_peak_reserved_gib": max(r["peak_reserved_gib"] for r in rows),
            "depth_counts": {str(d): sum(r["depth"] == d for r in rows) for d in DEPTHS},
            "mean_depth": statistics.mean(r["depth"] for r in rows),
            "decoder_layer_token_evaluations": layer_work,
            "layer_work_avoided_vs_36_at_same_token_count": 1 - layer_work / full_equivalent}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    controller = load_controller(args.controller)
    config = controller["generation"]
    if controller["label_generation_code_sha256"] != digest(Path(__file__).parent / "common.py"):
        raise ValueError("Qwen execution code differs from the labeling version; use matching V4 source.")
    if (args.output_dir / "manifest.json").exists():
        records = read_json(args.output_dir / "questions.json")
        dataset_revision = read_json(args.output_dir / "manifest.json")["config"]["dataset_revision"]
    else:
        records, dataset_revision = select_questions(
            "test", args.rows, args.seed, args.offset, controller["dataset_revision"],
            controller["excluded_question_keys"],
        )
    if len(records) != args.rows or set(controller["excluded_question_keys"]) & {r["question_key"] for r in records}:
        raise ValueError("Invalid benchmark question set or overlap with controller development data")
    tokenizer, model = load_model(config["model_revision"])
    manifest(args.output_dir, {"controller_sha256": digest(args.controller), "generation": config,
                              "rows": args.rows, "seed": args.seed, "offset": args.offset,
                              "dataset_revision": dataset_revision, "split": "test",
                              "questions_sha256": identity(records),
                              "order": "alternate baseline-first and controller-first by question"}, "benchmark")
    atomic_json(args.output_dir / "questions.json", records)
    baseline, guided = [], []
    with Progress(args.output_dir) as progress:
        progress.update(phase="warmup", completed_questions=0, total_questions=len(records))
        warmup(tokenizer, model, config)
        route(controller["pipeline"], ["A warmup math question"], controller["threshold"])
        for index, question in enumerate(records):
            folder = args.output_dir / "questions" / question["question_id"]
            decision_path = folder / "decision.json"
            if decision_path.exists():
                decision = read_json(decision_path)
            else:
                started = time.perf_counter()
                depths, confidence, raw = route(controller["pipeline"], [question["question"]], controller["threshold"])
                seconds = time.perf_counter() - started
                decision = {"depth": int(depths[0]), "raw_predicted_depth": int(raw[0]),
                            "confidence": float(confidence[0]), "threshold": controller["threshold"],
                            "controller_seconds": seconds}
                atomic_json(decision_path, decision)
            order = ("baseline", "controller") if index % 2 == 0 else ("controller", "baseline")
            for mode in order:
                depth = 36 if mode == "baseline" else decision["depth"]
                path = folder / f"{mode}.json"
                progress.update(phase="generating", current_question=question["question_id"],
                                mode=mode, depth=depth, completed_questions=index)
                if not path.exists():
                    result = generate(tokenizer, model, question, depth, config)
                    if mode == "controller":
                        result.update(decision)
                    atomic_json(path, result)
            base, controlled = read_json(folder / "baseline.json"), read_json(folder / "controller.json")
            baseline.append(base)
            guided.append(controlled)
            report = {"completed_questions": len(baseline), "total_questions": len(records),
                      "baseline": aggregate(baseline, False), "controller": aggregate(guided, True),
                      "correct_to_wrong": sum(a["correct"] and not b["correct"] for a, b in zip(baseline, guided)),
                      "wrong_to_correct": sum(not a["correct"] and b["correct"] for a, b in zip(baseline, guided)),
                      "threshold": controller["threshold"],
                      "notes": ["Routed latency includes CPU text routing, prompt formatting, tokenization, prefix setup and synchronized GPU generation.",
                                "Loading weights, warmups and disk/Drive I/O are excluded. Interrupted runs may span separate warmed sessions.",
                                "All 36-layer weights remain resident: peak memory need not decrease.",
                                "Layer-token counts are analytic work proxies, not FLOPs or GPU utilization.",
                                "A controller may select depth 36 for every question; report that outcome.",
                                "Original frozen Qwen exits are unadapted. Token counts and answer quality may worsen at shallow depth."]}
            report["baseline_seconds_divided_by_controller_seconds"] = report["baseline"]["total_seconds"] / report["controller"]["total_seconds"]
            report["actual_layer_work_ratio_controller_over_baseline"] = report["controller"]["decoder_layer_token_evaluations"] / report["baseline"]["decoder_layer_token_evaluations"]
            atomic_json(args.output_dir / "metrics.json", report)
            progress.update(completed_questions=len(baseline))
            print(f"{question['question_id']}: baseline={base['correct']} "
                  f"controller={controlled['correct']} depth={decision['depth']}", flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
