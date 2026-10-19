"""Paired normal vs controller-guided real answer generation on fresh questions.

The policy chooses at most one layer per question. Controller overhead and
the unchanged prefill are included in generation time. No draft is generated
and pruned. No label/reference answer is provided to the model or controller.
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import (MODEL_ID, PROTOCOL, Policy, atomic_json, digest, generate, generation_args,
                     load_model, prepare_manifest, questions, read_json, status)


def main():
    parser = generation_args()
    parser.set_defaults(rows=10, split="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--unseen-dir", type=Path, required=True,
                        help="Exclude the fresh questions already used in test_unseen")
    args = parser.parse_args()
    policy = Policy(args.checkpoint)
    if args.revision != policy.data["model_revision"] or args.max_new_tokens != policy.data["max_new_tokens"]:
        raise ValueError("Use checkpoint model revision and generation budget")
    prior = read_json(args.unseen_dir / "questions.json")
    records, revision = questions(
        args.split, args.rows, args.offset, args.dataset_revision or policy.data["dataset_revision"], args.seed,
        policy.data["excluded_question_keys"] + [r["question_key"] for r in prior],
    )
    config = {"protocol": PROTOCOL, "checkpoint_sha256": digest(args.checkpoint),
              "model_id": MODEL_ID, "precision": "float16", "attention": "sdpa",
              "source_dataset": "openai/gsm8k", "split": args.split,
              "model_revision": args.revision, "dataset_revision": revision,
              "question_keys": [r["question_key"] for r in records], "seed": args.seed,
              "max_new_tokens": args.max_new_tokens, "enable_thinking": False, "do_sample": False,
              "threshold": policy.threshold, "candidate_policy": "at_most_one_mlp_after_prefill",
              "timing": "GPU-synchronized generate wall time, tokenization excluded, controller included",
              "order": "alternating baseline-first/controller-first by question"}
    prepare_manifest(args.output_dir, config)
    atomic_json(args.output_dir / "questions.json", records)
    tokenizer, model = load_model(args.revision)
    # Warm both execution paths. Warm-up answers aren't counted as evaluation.
    generate(tokenizer, model, records[0], 2, args.seed)
    generate(tokenizer, model, records[0], 2, args.seed, policy=policy)
    pairs = []
    try:
        for index, q in enumerate(records):
            folder = args.output_dir / "questions" / q["question_id"]
            folder.mkdir(parents=True, exist_ok=True)
            modes = ["baseline", "controller"] if index % 2 == 0 else ["controller", "baseline"]
            pair = {"question_id": q["question_id"]}
            for mode in modes:
                path = folder / f"{mode}.json"
                status(args.output_dir, status="running", phase=mode,
                       question=q["question_id"], completed_questions=len(pairs))
                if path.exists():
                    result = read_json(path)
                else:
                    result, _ = generate(tokenizer, model, q, args.max_new_tokens, args.seed,
                                         policy=policy if mode == "controller" else None)
                    # Analytic GEMM count proxy, not total GPU FLOPs or GPU utilization.
                    count = result["token_mlp_evaluations_skipped"]
                    result["estimated_mlp_matmul_flops_avoided"] = (
                        count * 6 * model.config.hidden_size * model.config.intermediate_size
                    )
                    atomic_json(path, result)
                pair[mode] = result
            pairs.append(pair)
            print(f"{q['question_id']}: normal={pair['baseline']['correct']} "
                  f"controller={pair['controller']['correct']} selected={pair['controller']['selected_layer']}", flush=True)
        summary = report(pairs)
        atomic_json(args.output_dir / "metrics.json", summary)
        status(args.output_dir, status="completed", phase="finished", completed_questions=len(pairs))
        print(json.dumps(summary, indent=2))
    except BaseException as exc:
        status(args.output_dir, status="interrupted", phase="saved_pairs_retained",
               completed_questions=len(pairs), error=type(exc).__name__)
        raise


def report(pairs):
    summary = {"questions": len(pairs)}
    for mode in ("baseline", "controller"):
        rows = [p[mode] for p in pairs]
        completed = [r for r in rows if r["complete"]]
        seconds = sum(r["seconds"] for r in rows)
        decode_slots = sum(r["token_mlp_evaluations_decode"] for r in rows)
        total_slots = sum(r["token_mlp_evaluations_total"] for r in rows)
        skipped = sum(r["token_mlp_evaluations_skipped"] for r in rows)
        summary[mode] = {
            "correct": sum(r["correct"] for r in rows),
            "accuracy_all_questions": sum(r["correct"] for r in rows) / len(rows),
            "completed": len(completed), "incomplete": len(rows) - len(completed),
            "accuracy_completed_only": sum(r["correct"] for r in completed) / len(completed) if completed else None,
            "total_seconds": seconds, "mean_seconds": seconds / len(rows),
            "input_tokens": sum(r["input_tokens"] for r in rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "aggregate_output_tokens_per_second": sum(r["output_tokens"] for r in rows) / seconds,
            "max_peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows),
            "max_peak_reserved_gib": max(r["peak_reserved_gib"] for r in rows),
            "questions_with_skip": sum(r["selected_layer"] is not None and r["token_mlp_evaluations_skipped"] > 0 for r in rows),
            "decode_mlp_evaluations_skipped_fraction": skipped / decode_slots if decode_slots else 0.0,
            "all_mlp_evaluations_skipped_fraction": skipped / total_slots if total_slots else 0.0,
            "estimated_mlp_matmul_flops_avoided": sum(r["estimated_mlp_matmul_flops_avoided"] for r in rows),
        }
    summary["baseline_seconds_divided_by_controller_seconds"] = summary["baseline"]["total_seconds"] / summary["controller"]["total_seconds"]
    summary["correct_to_wrong"] = sum(p["baseline"]["correct"] and not p["controller"]["correct"] for p in pairs)
    summary["wrong_to_correct"] = sum(not p["baseline"]["correct"] and p["controller"]["correct"] for p in pairs)
    summary["notes"] = [
        "No guarantee of fewer output tokens. This intervention targets MLP computation per decode token.",
        "Memory is PyTorch allocator usage, not whole-device utilization or energy. No sampled GPU-utilization claim.",
        "FLOPs avoided counts bypassed MLP matmuls only; excludes policy overhead/attention/LM head and is not a measured end-to-end speedup.",
        "The last prompt position produces the first output token with no skip. Attention and KV cache always run.",
        "At most one of 36 MLPs can be bypassed per question, so possible compute savings are modest.",
        "Incomplete outputs fail all-question accuracy and are excluded only from completed-only accuracy.",
        "Single paired measurements on ten questions are exploratory; output lengths may differ and affect wall time.",
    ]
    return summary


if __name__ == "__main__":
    main()
