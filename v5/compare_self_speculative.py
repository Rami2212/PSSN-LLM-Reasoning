"""Compare full LayerSkip decoding with self-speculative early-exit decoding.

Uses the exact questions and few-shot demonstrations from a completed
``v5.smoke`` run. It does not truncate the model: early-exit candidates are
verified by the remaining layers, as implemented by Transformers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from v5.smoke import (
    MODEL_ID,
    STOP_STRING,
    Progress,
    atomic_json,
    environment,
    fingerprint,
    make_prompt,
    numeric_equal,
    parsed_answer,
    read_json,
)

PROTOCOL = "v5-layerskip-self-speculative-comparison-v1"
EXIT_LAYERS = (24, 28)


def generate(tokenizer, model, question, examples, exit_layer, token_cap, num_assistant_tokens):
    import torch
    from transformers import GenerationConfig, StopStringCriteria, StoppingCriteriaList

    prompt = make_prompt(question, examples)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    kwargs = {}
    if exit_layer is not None:
        kwargs.update(
            assistant_early_exit=exit_layer,
            num_assistant_tokens=num_assistant_tokens,
            num_assistant_tokens_schedule="constant",
        )
    config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=model.generation_config.eos_token_id or tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=token_cap,
        **kwargs,
    )
    # Do not use generate(stop_strings=..., tokenizer=...) here. In Transformers
    # self-speculative mode the internal early-exit draft calls generate again,
    # but that call does not receive the tokenizer argument. A pre-built
    # StopStringCriteria is tokenizer-independent when called, and the target
    # generation loop applies it to the verified token stream.
    stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer, [STOP_STRING])])

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            generation_config=config,
            stopping_criteria=stopping_criteria,
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    input_tokens = inputs["input_ids"].shape[1]
    generated = output[0, input_tokens:]
    raw_response = tokenizer.decode(generated, skip_special_tokens=True)
    eos_ids = config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]
    stopped_on_eos = bool(len(generated) and generated[-1].item() in eos_ids)
    stopped_on_boundary = STOP_STRING in raw_response
    complete = stopped_on_eos or stopped_on_boundary
    response = raw_response.split(STOP_STRING, 1)[0].strip()
    extracted = parsed_answer(response)
    return {
        "question_id": question["question_id"],
        "mode": "full_depth" if exit_layer is None else "self_speculative",
        "assistant_early_exit": exit_layer,
        "num_assistant_tokens": None if exit_layer is None else num_assistant_tokens,
        "reference_answer": question["reference_answer"],
        "response": response,
        "raw_response": raw_response,
        "generated_token_ids": generated.tolist(),
        "extracted_answer": extracted,
        "complete": complete,
        "finish_reason": "eos" if stopped_on_eos else "next_question" if stopped_on_boundary else "length",
        "correct": complete and numeric_equal(extracted, question["reference_answer"]),
        "input_tokens": input_tokens,
        "output_tokens": len(generated),
        "seconds": seconds,
        "output_tokens_per_second": len(generated) / seconds if seconds else None,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }


def summarize(trials, questions):
    grouped = {"full_depth": [], **{f"self_speculative_{layer}": [] for layer in EXIT_LAYERS}}
    for row in trials:
        key = "full_depth" if row["assistant_early_exit"] is None else f"self_speculative_{row['assistant_early_exit']}"
        grouped[key].append(row)

    metrics = {}
    for key, rows in grouped.items():
        count = len(rows)
        seconds = sum(row["seconds"] for row in rows)
        output_tokens = sum(row["output_tokens"] for row in rows)
        metrics[key] = {
            "trials": count,
            "correct": sum(bool(row["correct"]) for row in rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / count if count else None,
            "complete": sum(bool(row["complete"]) for row in rows),
            "output_tokens": output_tokens,
            "total_seconds": seconds,
            "mean_seconds": seconds / count if count else None,
            "aggregate_output_tokens_per_second": output_tokens / seconds if seconds else None,
            "max_peak_allocated_gib": max((row["peak_allocated_gib"] for row in rows), default=None),
            "max_peak_reserved_gib": max((row["peak_reserved_gib"] for row in rows), default=None),
        }

    by_question_mode = {(row["question_id"], row["assistant_early_exit"]): row for row in trials}
    comparisons = {}
    for layer in EXIT_LAYERS:
        pairs = [
            (by_question_mode[(question["question_id"], None)],
             by_question_mode[(question["question_id"], layer)])
            for question in questions
            if (question["question_id"], None) in by_question_mode
            and (question["question_id"], layer) in by_question_mode
        ]
        baseline_time = sum(base["seconds"] for base, _ in pairs)
        speculative_time = sum(spec["seconds"] for _, spec in pairs)
        comparisons[str(layer)] = {
            "paired_questions": len(pairs),
            "self_speculative_exact_token_match_to_baseline": sum(
                base["generated_token_ids"] == spec["generated_token_ids"] for base, spec in pairs
            ),
            "correct_to_wrong": sum(base["correct"] and not spec["correct"] for base, spec in pairs),
            "wrong_to_correct": sum(not base["correct"] and spec["correct"] for base, spec in pairs),
            "baseline_seconds_divided_by_self_speculative_seconds":
                baseline_time / speculative_time if speculative_time else None,
            "output_token_difference_vs_baseline": sum(
                spec["output_tokens"] - base["output_tokens"] for base, spec in pairs
            ),
        }
    return {
        "status": "completed" if len(trials) == len(questions) * 3 else "running",
        "completed_trials": len(trials),
        "total_trials": len(questions) * 3,
        "metrics": metrics,
        "paired_vs_full_depth": comparisons,
        "notes": [
            "Self-speculative mode drafts at the selected early layer and verifies/corrects candidates with the remaining layers; it does not permanently truncate the model.",
            "Greedy decoding is used. Hugging Face documents self-speculative decoding as preserving the target model output distribution; exact token equality is reported empirically.",
            "A speed ratio above 1 means self-speculative generation was faster in these paired measurements.",
            "Five questions are a smoke comparison only; they do not establish reliable accuracy or speed claims.",
            "Timing includes tokenization and synchronized generation, but excludes model loading, warmup and result writes.",
            "Memory is PyTorch allocated/reserved peak, not device utilization. Model weights remain resident in GPU memory.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True,
                        help="Completed v5.smoke output directory (contains manifest/questions/demonstrations)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-assistant-tokens", type=int, default=5,
                        help="Draft tokens per self-speculative round (default: 5)")
    parser.add_argument("--revision", help="Optional model commit; defaults to source smoke manifest revision")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.num_assistant_tokens < 1:
        parser.error("token limits must be positive")
    source_manifest_path = args.source_run / "manifest.json"
    source_manifest = read_json(source_manifest_path)
    source_config = source_manifest.get("config", {})
    if source_config.get("model_id") != MODEL_ID or not source_config.get("model_revision"):
        raise ValueError("Source run manifest is not a compatible, revision-pinned V5 smoke run")
    questions = read_json(args.source_run / "questions.json")
    examples = read_json(args.source_run / "demonstrations.json")
    if not questions or len(examples) != 3:
        raise ValueError("Source run must have saved questions and exactly three demonstrations")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with Progress(args.output_dir) as progress:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if transformers.__version__ != "4.57.1":
            raise RuntimeError("Install v5/requirements.txt (transformers==4.57.1)")
        if not torch.cuda.is_available():
            raise RuntimeError("A ROCm or CUDA GPU is required")
        model_revision = args.revision or source_config["model_revision"]
        if model_revision != source_config["model_revision"]:
            raise ValueError("Use the exact model revision from the source run for a fair comparison")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        config = {
            "protocol": PROTOCOL,
            "source_run": str(args.source_run.resolve()),
            "source_manifest_identity": source_manifest.get("identity"),
            "model_id": MODEL_ID,
            "model_revision": model_revision,
            "tokenizer_revision": model_revision,
            "questions_sha256": fingerprint(questions),
            "demonstrations_sha256": fingerprint(examples),
            "baseline": "full_depth_32_greedy",
            "self_speculative_exit_layers": list(EXIT_LAYERS),
            "num_assistant_tokens": args.num_assistant_tokens,
            "num_assistant_tokens_schedule": "constant",
            "max_new_tokens": args.max_new_tokens,
            "dtype": str(dtype),
            "attention": "sdpa",
            "do_sample": False,
            "num_beams": 1,
            "stop_string": STOP_STRING,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        env = environment(torch)
        identity = fingerprint({"config": config, "environment": env})
        manifest_path = args.output_dir / "manifest.json"
        old = read_json(manifest_path) if manifest_path.exists() else None
        if old and old.get("identity") != identity:
            raise ValueError("Settings/source/environment changed; use a NEW output directory")
        atomic_json(manifest_path, {"identity": identity, "config": config, "environment": env})
        atomic_json(args.output_dir / "questions.json", questions)
        atomic_json(args.output_dir / "demonstrations.json", examples)

        progress.update(phase="loading_model", total_trials=len(questions) * 3)
        print(f"Loading {MODEL_ID} revision={model_revision} dtype={dtype}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_revision)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=model_revision, dtype=dtype, attn_implementation="sdpa",
        ).to("cuda").eval().requires_grad_(False)
        if model.config.model_type != "llama" or len(model.model.layers) != 32:
            raise ValueError("Expected the 32-layer LayerSkip Llama3-8B checkpoint")

        warm_question = {"question_id": "warmup", "question": "What is 17 plus 25?", "reference_answer": "42"}
        print("Warming up full-depth and self-speculative decoding...", flush=True)
        progress.update(phase="warmup")
        for layer in (None, *EXIT_LAYERS):
            generate(tokenizer, model, warm_question, examples, layer, 8, args.num_assistant_tokens)

        trials = []
        mode_order = [None, *EXIT_LAYERS]
        for question_index, question in enumerate(questions):
            shift = question_index % len(mode_order)
            order = mode_order[shift:] + mode_order[:shift]
            for layer in order:
                label = "full_depth" if layer is None else f"self_speculative_{layer}"
                trial_path = args.output_dir / "questions" / question["question_id"] / f"{label}.json"
                progress.update(phase="generating", current_question=question["question_id"],
                                current_mode=label, completed_trials=len(trials))
                if trial_path.exists():
                    trial = read_json(trial_path)
                    if trial.get("run_identity") != identity or trial.get("assistant_early_exit") != layer:
                        raise ValueError(f"Saved trial does not match this run: {trial_path}")
                    reused = True
                else:
                    trial = generate(tokenizer, model, question, examples, layer,
                                     args.max_new_tokens, args.num_assistant_tokens)
                    trial["run_identity"] = identity
                    atomic_json(trial_path, trial)
                    reused = False
                trials.append(trial)
                atomic_json(args.output_dir / "summary.json", summarize(trials, questions))
                progress.update(completed_trials=len(trials))
                print(f"{question['question_id']} / {label}: correct={trial['correct']} "
                      f"complete={trial['complete']} tokens={trial['output_tokens']} "
                      f"seconds={trial['seconds']:.2f}"
                      + (" [saved]" if reused else ""), flush=True)
        print(json.dumps(summarize(trials, questions), indent=2), flush=True)


if __name__ == "__main__":
    main()
