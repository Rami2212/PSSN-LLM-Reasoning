"""Compare normal generation with up to four controller-selected MLP group bypasses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import MODEL_ID, atomic_json, digest, generation_args, load_model, questions, read_json
from .block_common import (BLOCK4_PROTOCOL, BlockPolicy, generate_group,
                           prepare_manifest, status)


def main():
    parser = generation_args()
    parser.set_defaults(rows=100, split="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--unseen-dir", type=Path, required=True)
    parser.add_argument("--max-removal-groups", type=int, choices=range(1, 5), default=4,
                        help="Skip up to this many highest-scoring groups above the checkpoint threshold (default: 4)")
    args = parser.parse_args()
    policy = BlockPolicy(args.checkpoint, max_removal_groups=args.max_removal_groups)
    if args.revision != policy.data["model_revision"] or args.max_new_tokens != policy.data["max_new_tokens"]:
        raise ValueError("Use checkpoint model revision and generation budget")
    prior = read_json(args.unseen_dir / "questions.json")
    records, revision = questions(
        args.split, args.rows, args.offset, args.dataset_revision or policy.data["dataset_revision"],
        args.seed, policy.data["excluded_question_keys"] + [r["question_key"] for r in prior],
    )
    config = {
        "protocol": BLOCK4_PROTOCOL, "checkpoint_sha256": digest(args.checkpoint),
        "checkpoint_protocol": policy.data["protocol"],
        "model_id": MODEL_ID, "precision": "float16", "attention": "sdpa",
        "source_dataset": "openai/gsm8k", "split": args.split,
        "model_revision": args.revision, "dataset_revision": revision,
        "question_keys": [r["question_key"] for r in records], "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "enable_thinking": False, "do_sample": False,
        "prompt_protocol": "Solve carefully; optional short calculation; exact FINAL line (same as block4-label-v1)",
        "threshold": policy.threshold,
        "candidate_policy": "highest_scoring_above_threshold_nonoverlapping_four_mlp_groups_after_prefill",
        "max_removal_groups": args.max_removal_groups,
        "combination_labels_used_for_training": False,
        "routing_implementation": "torch.cond+torch.compile; device-resident predicate; no CPU selection",
        "timing": "GPU-synchronized generate wall time, tokenization excluded, controller included",
        "order": "alternating baseline-first/controller-first by question",
    }
    prepare_manifest(args.output_dir, config)
    atomic_json(args.output_dir / "questions.json", records)
    tokenizer, model = load_model(args.revision)
    generate_group(tokenizer, model, records[0], 2, args.seed)
    generate_group(tokenizer, model, records[0], 2, args.seed, policy=policy)
    comparisons = []
    try:
        for index, question in enumerate(records):
            folder = args.output_dir / "questions" / question["question_id"]
            folder.mkdir(parents=True, exist_ok=True)
            modes = ["baseline", "controller"] if index % 2 == 0 else ["controller", "baseline"]
            row = {"question_id": question["question_id"]}
            for mode in modes:
                path = folder / f"{mode}.json"
                status(args.output_dir, status="running", phase=mode,
                       question=question["question_id"], completed_questions=len(comparisons))
                if path.exists():
                    result = read_json(path)
                else:
                    result, _ = generate_group(
                        tokenizer, model, question, args.max_new_tokens, args.seed,
                        policy=policy if mode == "controller" else None,
                    )
                    count = result["token_mlp_evaluations_skipped"]
                    result["estimated_mlp_matmul_flops_avoided"] = (
                        count * 6 * model.config.hidden_size * model.config.intermediate_size
                    )
                    atomic_json(path, result)
                row[mode] = result
            comparisons.append(row)
            print(f"{question['question_id']}: normal={row['baseline']['correct']} "
                  f"controller={row['controller']['correct']} "
                  f"selected_groups={row['controller']['selected_groups']}", flush=True)
        report = summarize(comparisons, args.max_removal_groups)
        atomic_json(args.output_dir / "metrics.json", report)
        status(args.output_dir, status="completed", phase="finished", completed_questions=len(comparisons))
        print(json.dumps(report, indent=2))
    except BaseException as exc:
        status(args.output_dir, status="interrupted", phase="saved_comparisons_retained",
               completed_questions=len(comparisons), error=type(exc).__name__)
        raise


def summarize(comparisons, max_removal_groups=4):
    report = {"questions": len(comparisons), "max_removal_groups": max_removal_groups,
              "combination_labels_used_for_training": False}
    for mode in ("baseline", "controller"):
        rows = [row[mode] for row in comparisons]
        completed = [row for row in rows if row["complete"]]
        seconds = sum(row["seconds"] for row in rows)
        skipped = sum(row["token_mlp_evaluations_skipped"] for row in rows)
        decode = sum(row["token_mlp_evaluations_decode"] for row in rows)
        total = sum(row["token_mlp_evaluations_total"] for row in rows)
        report[mode] = {
            "correct": sum(row["correct"] for row in rows),
            "accuracy_all_questions": sum(row["correct"] for row in rows) / len(rows),
            "completed": len(completed), "incomplete": len(rows) - len(completed),
            "accuracy_completed_only": sum(row["correct"] for row in completed) / len(completed) if completed else None,
            "total_seconds": seconds, "mean_seconds": seconds / len(rows),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "output_tokens": sum(row["output_tokens"] for row in rows),
            "aggregate_output_tokens_per_second": sum(row["output_tokens"] for row in rows) / seconds,
            "max_peak_allocated_gib": max(row["peak_allocated_gib"] for row in rows),
            "max_peak_reserved_gib": max(row["peak_reserved_gib"] for row in rows),
            "questions_with_group_skip": sum(row["selected_group"] is not None and
                row["token_mlp_evaluations_skipped"] > 0 for row in rows),
            "mean_selected_groups": sum(row["selected_group_count"] for row in rows) / len(rows),
            "max_selected_groups": max(row["selected_group_count"] for row in rows),
            "selected_group_count_histogram": {
                str(count): sum(row["selected_group_count"] == count for row in rows)
                for count in range(max_removal_groups + 1)
            },
            "decode_mlp_evaluations_skipped_fraction": skipped / decode if decode else 0.0,
            "all_mlp_evaluations_skipped_fraction": skipped / total if total else 0.0,
            "estimated_mlp_matmul_flops_avoided": sum(
                row["estimated_mlp_matmul_flops_avoided"] for row in rows),
        }
    report["baseline_seconds_divided_by_controller_seconds"] = (
        report["baseline"]["total_seconds"] / report["controller"]["total_seconds"]
    )
    report["correct_to_wrong"] = sum(row["baseline"]["correct"] and not row["controller"]["correct"] for row in comparisons)
    report["wrong_to_correct"] = sum(not row["baseline"]["correct"] and row["controller"]["correct"] for row in comparisons)
    report["notes"] = [
        "The controller selects up to the configured maximum of four-MLP groups per question, ranked by score and filtered by the checkpoint threshold.",
        "Training labels measure isolated group removals. This benchmark measures combined removals; their safety was not established by the training labels.",
        "Group labels cannot tell which member caused a failure. The action is the group itself.",
        "FLOPs are an analytic estimate for bypassed MLP matmuls only, not measured utilization or end-to-end speedup.",
        "Prompt prefill, attention, KV cache, and other model operations still run normally.",
        "Ten benchmark questions are exploratory; they do not establish publication-level gains.",
    ]
    return report


if __name__ == "__main__":
    main()
