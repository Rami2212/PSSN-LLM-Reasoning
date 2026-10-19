"""Run a paired 10-question GSM8K pilot with and without the final-layer detector.

The ordinary Qwen answer is also used as the draft trace for the detector arm.
This keeps both arms tied to the same reasoning text: one returns Qwen's
ordinary answer, while the other scores draft sentences at threshold 0.5,
removes those predicted low-value, and asks Qwen to answer from the retained
notes. This is a text-context intervention, not an intervention on Qwen's
private/internal computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.black_hole.config import BlackHoleDetectorConfig
from src.black_hole.detector import BlackHoleDetector
from src.black_hole.sentence_experiment import split_sentences
from src.data.gsm8k import GSM8K_CONFIG, GSM8K_DATASET_ID, normalize_example
from src.models.hidden_states import extract_span_hidden_vectors

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"
DEFAULT_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
PREDICTOR_THRESHOLD = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _normalize_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).split()).casefold()


def _known_training_questions(activation_jsonl_dir: Path) -> set[str]:
    known: set[str] = set()
    for split in ("train", "validation", "test"):
        path = activation_jsonl_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing activation source rows needed to exclude seen questions: {path}")
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
                question = row.get("question") if isinstance(row, dict) else None
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f"Missing question in {path}:{line_number}")
                known.add(_normalize_question(question))
    return known


def _load_questions(rows: int, seed: int, known_questions: set[str]) -> list[dict[str, str]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on server environment
        if exc.name == "datasets":
            raise RuntimeError(
                "Missing 'datasets'. Install the project dependencies in the active virtual environment."
            ) from exc
        raise RuntimeError(
            "The Hugging Face dataset loader has a missing dependency. On the ROCm 7.2 environment, "
            "install the compatible pair: datasets==3.6.0 and pyarrow==19.0.1."
        ) from exc
    except ImportError as exc:  # pragma: no cover - dependency ABI/version mismatch
        raise RuntimeError(
            "The Hugging Face dataset loader could not import its dependencies. "
            "On the ROCm 7.2 environment, install the compatible pair: "
            "datasets==3.6.0 and pyarrow==19.0.1 (keep NumPy at 1.26.4)."
        ) from exc
    dataset = load_dataset(GSM8K_DATASET_ID, GSM8K_CONFIG, split="test")
    eligible = []
    for index, source in enumerate(dataset):
        record = normalize_example(source, split="test", index=index)
        if _normalize_question(record["question"]) not in known_questions:
            eligible.append(record)
    if len(eligible) < rows:
        raise ValueError(f"Only {len(eligible)} unseen GSM8K test questions remain; requested {rows}")
    chosen = random.Random(seed).sample(eligible, rows)
    return sorted(chosen, key=lambda item: item["id"])


def _device_of(model: Any) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return model.device


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]], *, thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def _generate(
    tokenizer: Any,
    model: Any,
    torch: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    thinking: bool,
) -> dict[str, Any]:
    rendered = _render_prompt(tokenizer, [{"role": "user", "content": prompt}], thinking=thinking)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    device = _device_of(model)
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in dict(encoded).items()
    }
    input_tokens = int(model_inputs["input_ids"].shape[-1])
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated_ids = output[0, input_tokens:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": int(generated_ids.numel()),
        "seconds": elapsed,
    }


def _extract_trace(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    open_match = re.search(r"<think>(.*)$", text, flags=re.DOTALL)
    if open_match:
        text = open_match.group(1)
    # Accommodate a response without visible Qwen thinking tags.
    text = re.split(r"(?:^|\n)\s*####\s*", text, maxsplit=1)[0]
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return text.strip()


def _extract_answer(text: str) -> str | None:
    patterns = (
        r"####\s*([-+]?\s*\$?\s*\d[\d,]*(?:\.\d+)?)",
        r"\\boxed\{\s*([-+]?\s*\$?\s*\d[\d,]*(?:\.\d+)?)\s*\}",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    candidates = re.findall(r"(?<![\w.])[-+]?\s*\$?\s*\d[\d,]*(?:\.\d+)?", text)
    return candidates[-1].strip() if candidates else None


def _answers_equal(predicted: str | None, reference: str) -> bool | None:
    if predicted is None:
        return None
    from decimal import Decimal, InvalidOperation

    def number(value: str) -> Decimal:
        cleaned = re.sub(r"[\s,$]", "", value)
        return Decimal(cleaned)

    try:
        return number(predicted) == number(reference)
    except (InvalidOperation, ValueError):
        return None


def _reset_gpu_peak(torch: Any) -> int | None:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return int(torch.cuda.memory_allocated())


def _gpu_memory(torch: Any, start_allocated: int | None) -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"peak_allocated_gib": None, "peak_reserved_gib": None,
                "peak_allocated_delta_gib": None}
    torch.cuda.synchronize()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    return {
        "peak_allocated_gib": round(peak_allocated / 2**30, 3),
        "peak_reserved_gib": round(peak_reserved / 2**30, 3),
        "peak_allocated_delta_gib": round(max(0, peak_allocated - (start_allocated or 0)) / 2**30, 3),
    }


def _process_memory_gib() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024**2, 3)
    except (OSError, ValueError, IndexError):
        pass
    try:  # pragma: no cover - optional non-Linux fallback
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(value / (1024**2), 3)  # Linux reports KiB.
    except (ImportError, AttributeError):
        return None


def _predict_trace(
    *, tokenizer: Any, model: Any, detector: Any, feature_mean: Any, feature_scale: Any,
    torch: Any, question: str, trace: str, threshold: float,
) -> dict[str, Any]:
    sentence_items = split_sentences(trace)
    if not sentence_items:
        return {"sentences": [], "kept_trace": "", "activation_context_tokens": 0,
                "activation_seconds": 0.0, "skipped_count": 0}

    # Build the same Qwen user-question prefix used during predictor training.
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    spans = [(len(prefix) + start, len(prefix) + end) for _, start, end in sentence_items]
    full_context = prefix + trace
    encoded_for_count = tokenizer(full_context, add_special_tokens=False)
    activation_context_tokens = len(encoded_for_count["input_ids"])

    start_allocated = _reset_gpu_peak(torch)
    started = time.perf_counter()
    vectors, _, _ = extract_span_hidden_vectors(
        model, tokenizer, full_context, spans, layer=-1, pooling="mean", torch_module=torch
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    activation_seconds = time.perf_counter() - started

    features = torch.as_tensor(vectors, dtype=torch.float32, device=feature_mean.device)
    normalized = torch.nan_to_num((features - feature_mean) / feature_scale)
    detector.eval()
    with torch.inference_mode():
        probabilities = detector.predict_proba(normalized).detach().cpu().numpy().astype(float)
    scores = []
    kept = []
    for index, (sentence, start, end) in enumerate(sentence_items):
        probability = float(probabilities[index])
        skip = probability >= threshold
        scores.append({
            "sentence_index": index,
            "sentence_text": sentence,
            "start_char": start,
            "end_char": end,
            "low_value_probability": probability,
            "predicted_skip": bool(skip),
        })
        if not skip:
            kept.append(sentence)
    return {
        "sentences": scores,
        "kept_trace": "".join(kept),
        "activation_context_tokens": int(activation_context_tokens),
        "activation_seconds": activation_seconds,
        "skipped_count": sum(item["predicted_skip"] for item in scores),
        "gpu_memory": _gpu_memory(torch, start_allocated),
    }


def _read_completed(output_path: Path) -> dict[str, dict[str, Any]]:
    completed = {}
    if not output_path.is_file():
        return completed
    with output_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "completed" and isinstance(row.get("problem_id"), str):
                completed[row["problem_id"]] = row
    return completed


def _mean(records: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in records if isinstance(row.get(field), (int, float))]
    return round(statistics.mean(values), 4) if values else None


def run_pilot(
    *, checkpoint_path: Path, activation_jsonl_dir: Path, output_dir: Path,
    rows: int = 10, seed: int = 42, threshold: float = PREDICTOR_THRESHOLD,
    max_new_tokens: int = 1024, model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION, require_cuda: bool = True,
) -> dict[str, Any]:
    import torch

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    for path in (checkpoint_path, activation_jsonl_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    known_questions = _known_training_questions(activation_jsonl_dir)
    selected = _load_questions(rows, seed, known_questions)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_questions.jsonl"
    selected_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    if selected_path.exists() and selected_path.read_text(encoding="utf-8") != selected_text:
        raise ValueError(f"{selected_path} already exists with a different sample; choose a new output directory")
    if not selected_path.exists():
        selected_path.write_text(selected_text, encoding="utf-8")

    from src.utils.environment import collect_environment_info, load_qwen_model

    environment = collect_environment_info(torch)
    tokenizer, model, precision = load_qwen_model(
        model_id, revision=revision, require_cuda=require_cuda, torch_module=torch
    )
    resolved_revision = getattr(getattr(model, "config", None), "_commit_hash", None) or revision

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "detector_config", "feature_mean", "feature_scale"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Predictor checkpoint missing fields: {sorted(missing)}")
    config = BlackHoleDetectorConfig.from_dict(checkpoint["detector_config"])
    if config.input_size != int(model.config.hidden_size):
        raise ValueError(
            f"Final-layer predictor expects {config.input_size} features, Qwen has hidden size "
            f"{model.config.hidden_size}; use the matching final-layer checkpoint."
        )
    detector = BlackHoleDetector(config)
    detector.load_state_dict(checkpoint["model_state_dict"])
    detector.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector.to(device)
    feature_mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32, device=device)
    feature_scale = torch.as_tensor(checkpoint["feature_scale"], dtype=torch.float32, device=device)
    if feature_mean.shape != (config.input_size,) or feature_scale.shape != (config.input_size,):
        raise ValueError("Saved predictor normalization has the wrong feature size")
    if torch.any(feature_scale <= 0):
        raise ValueError("Saved predictor feature_scale must be positive")

    result_path = output_dir / "question_results.jsonl"
    completed = _read_completed(result_path)
    run_started = time.perf_counter()
    for index, example in enumerate(selected, 1):
        if example["id"] in completed:
            print(f"[{index}/{rows}] {example['id']}: already saved; skipping", flush=True)
            continue
        question_started = time.perf_counter()
        print(f"[{index}/{rows}] Running {example['id']}", flush=True)

        baseline_start_allocated = _reset_gpu_peak(torch)
        baseline = _generate(
            tokenizer, model, torch,
            "Solve this math problem. Show your working, then put the final numerical answer on a line starting with ####.\n\n"
            + example["question"],
            max_new_tokens=max_new_tokens,
            thinking=True,
        )
        baseline_memory = _gpu_memory(torch, baseline_start_allocated)
        trace = _extract_trace(baseline["text"])
        prediction = _predict_trace(
            tokenizer=tokenizer, model=model, detector=detector,
            feature_mean=feature_mean, feature_scale=feature_scale, torch=torch,
            question=example["question"], trace=trace, threshold=threshold,
        )

        predictor_prompt = (
            "Solve the math problem. Use the retained reasoning notes as helpful context, but verify the calculation. "
            "Return only the final numerical answer on a line starting with ####.\n\n"
            f"Problem:\n{example['question']}\n\nRetained reasoning notes:\n"
            f"{prediction['kept_trace'] or '[No reasoning notes retained.]'}"
        )
        answer_start_allocated = _reset_gpu_peak(torch)
        predicted_answer = _generate(
            tokenizer, model, torch, predictor_prompt,
            max_new_tokens=128, thinking=False,
        )
        answer_memory = _gpu_memory(torch, answer_start_allocated)

        baseline_answer = _extract_answer(baseline["text"])
        wormhole_answer = _extract_answer(predicted_answer["text"])
        predictor_peak_values = [
            baseline_memory.get("peak_allocated_gib"),
            prediction.get("gpu_memory", {}).get("peak_allocated_gib"),
            answer_memory.get("peak_allocated_gib"),
        ]
        predictor_reserved_values = [
            baseline_memory.get("peak_reserved_gib"),
            prediction.get("gpu_memory", {}).get("peak_reserved_gib"),
            answer_memory.get("peak_reserved_gib"),
        ]
        predictor_delta_values = [
            baseline_memory.get("peak_allocated_delta_gib"),
            prediction.get("gpu_memory", {}).get("peak_allocated_delta_gib"),
            answer_memory.get("peak_allocated_delta_gib"),
        ]
        def maximum(values):
            finite = [value for value in values if isinstance(value, (int, float))]
            return round(max(finite), 3) if finite else None

        baseline_total_tokens = baseline["input_tokens"] + baseline["output_tokens"]
        predictor_input_tokens = (
            baseline["input_tokens"] + prediction["activation_context_tokens"]
            + predicted_answer["input_tokens"]
        )
        predictor_generated_tokens = baseline["output_tokens"] + predicted_answer["output_tokens"]
        elapsed = time.perf_counter() - question_started
        row = {
            "status": "completed",
            "problem_id": example["id"],
            "question": example["question"],
            "reference_answer": example["reference_answer"],
            "baseline": {
                "answer": baseline_answer,
                "correct": _answers_equal(baseline_answer, example["reference_answer"]),
                "response": baseline["text"],
                "input_tokens": baseline["input_tokens"],
                "output_tokens": baseline["output_tokens"],
                "total_prompt_plus_generated_tokens": baseline_total_tokens,
                "generation_seconds": round(baseline["seconds"], 4),
                "gpu_memory": baseline_memory,
            },
            "final_layer_predictor_threshold_0_5": {
                "answer": wormhole_answer,
                "correct": _answers_equal(wormhole_answer, example["reference_answer"]),
                "response": predicted_answer["text"],
                "threshold": threshold,
                "sentences_total": len(prediction["sentences"]),
                "sentences_predicted_skipped": prediction["skipped_count"],
                "sentences_retained": len(prediction["sentences"]) - prediction["skipped_count"],
                "sentence_predictions": prediction["sentences"],
                "draft_trace": trace,
                "retained_trace": prediction["kept_trace"],
                "draft_generation_tokens": baseline["output_tokens"],
                "activation_context_tokens_processed": prediction["activation_context_tokens"],
                "answer_prompt_tokens": predicted_answer["input_tokens"],
                "final_answer_tokens": predicted_answer["output_tokens"],
                "total_input_or_activation_tokens": predictor_input_tokens,
                "total_generated_tokens": predictor_generated_tokens,
                "total_tokens_processed_estimate": predictor_input_tokens + predictor_generated_tokens,
                "draft_generation_seconds_shared_with_baseline": round(baseline["seconds"], 4),
                "activation_scoring_seconds": round(prediction["activation_seconds"], 4),
                "final_answer_generation_seconds": round(predicted_answer["seconds"], 4),
                "end_to_end_seconds_including_shared_draft": round(elapsed, 4),
                "gpu_memory_peak_across_pipeline": {
                    "peak_allocated_gib": maximum(predictor_peak_values),
                    "peak_reserved_gib": maximum(predictor_reserved_values),
                    "peak_allocated_delta_gib": maximum(predictor_delta_values),
                },
            },
            "question_elapsed_seconds": round(elapsed, 4),
            "process_peak_rss_gib": _process_memory_gib(),
        }
        with result_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
        completed[example["id"]] = row
        print(
            f"  baseline={row['baseline']['correct']} predictor={row['final_layer_predictor_threshold_0_5']['correct']} "
            f"skipped={prediction['skipped_count']}/{len(prediction['sentences'])} "
            f"tokens={baseline_total_tokens}->{predictor_input_tokens + predictor_generated_tokens} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

    ordered = [completed[item["id"]] for item in selected if item["id"] in completed]
    if len(ordered) != rows:
        raise RuntimeError(f"Only {len(ordered)}/{rows} selected questions have completed; completed rows are saved for resume")
    baseline_records = [row["baseline"] for row in ordered]
    predictor_records = [row["final_layer_predictor_threshold_0_5"] for row in ordered]
    baseline_correct = sum(row["correct"] is True for row in baseline_records)
    predictor_correct = sum(row["correct"] is True for row in predictor_records)
    baseline_tok = [row["total_prompt_plus_generated_tokens"] for row in baseline_records]
    predictor_tok = [row["total_tokens_processed_estimate"] for row in predictor_records]
    summary = {
        "schema_version": "pssn-final-layer-wormhole-pilot-v1",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "10 unseen GSM8K test questions; baseline Qwen answer compared with Qwen answer conditioned on the same draft trace after final-layer predictor filtering at fixed threshold 0.5.",
        "interpretation_limit": "This tests text-context filtering, not erasure/control of Qwen's internal reasoning. Labels are transferred labels; correctness is scored against GSM8K reference answers.",
        "protocol": {
            "dataset": f"{GSM8K_DATASET_ID}/{GSM8K_CONFIG}:test",
            "sample_seed": seed,
            "requested_rows": rows,
            "excluded_questions_seen_in_predictor_dataset": len(known_questions),
            "predictor_threshold": threshold,
            "threshold_source": "fixed 0.5; no 10% false-skip rule and no test-based tuning",
            "qwen_model_id": model_id,
            "qwen_revision": resolved_revision,
            "qwen_precision": precision,
            "draft_max_new_tokens": max_new_tokens,
            "final_answer_max_new_tokens": 128,
            "generation": "greedy do_sample=false",
            "sentence_feature": "mean-pooled final transformer-layer activations for each generated trace sentence, using Qwen question prefix and causal preceding trace",
        },
        "predictor_checkpoint": str(checkpoint_path.resolve()),
        "predictor_checkpoint_sha256": _sha256(checkpoint_path),
        "predictor_best_epoch": checkpoint.get("best_epoch"),
        "environment": environment,
        "results": {
            "completed_questions": len(ordered),
            "baseline_correct_count": baseline_correct,
            "baseline_accuracy": baseline_correct / len(ordered),
            "predictor_correct_count": predictor_correct,
            "predictor_accuracy": predictor_correct / len(ordered),
            "mean_baseline_prompt_plus_generated_tokens": round(statistics.mean(baseline_tok), 2),
            "mean_predictor_total_tokens_processed_estimate": round(statistics.mean(predictor_tok), 2),
            "mean_baseline_generation_seconds": _mean(baseline_records, "generation_seconds"),
            "mean_predictor_end_to_end_seconds": _mean(predictor_records, "end_to_end_seconds_including_shared_draft"),
            "mean_predictor_skipped_sentence_count": round(statistics.mean(row["sentences_predicted_skipped"] for row in predictor_records), 2),
            "mean_baseline_gpu_peak_allocated_gib": _mean([row["gpu_memory"] for row in baseline_records], "peak_allocated_gib"),
            "mean_predictor_gpu_peak_allocated_gib": _mean([row["gpu_memory_peak_across_pipeline"] for row in predictor_records], "peak_allocated_gib"),
            "max_process_peak_rss_gib": max((row["process_peak_rss_gib"] for row in ordered if row.get("process_peak_rss_gib") is not None), default=None),
        },
        "outputs": {
            "selected_questions": str(selected_path.resolve()),
            "question_results": str(result_path.resolve()),
        },
        "elapsed_seconds_this_invocation": round(time.perf_counter() - run_started, 3),
    }
    _atomic_json(output_dir / "run_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Final-layer training best_model.pt")
    parser.add_argument("--activation-jsonl-dir", type=Path, required=True,
                        help="V2 activation folder containing train.jsonl, validation.jsonl, test.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=PREDICTOR_THRESHOLD,
                        help="Fixed low-value probability cutoff; default 0.5, no safety cap")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow running without a GPU")
    args = parser.parse_args(argv)
    if args.rows < 1:
        parser.error("--rows must be positive")
    result = run_pilot(
        checkpoint_path=args.checkpoint,
        activation_jsonl_dir=args.activation_jsonl_dir,
        output_dir=args.output_dir,
        rows=args.rows,
        seed=args.seed,
        threshold=args.threshold,
        max_new_tokens=args.max_new_tokens,
        model_id=args.model_id,
        revision=args.revision,
        require_cuda=not args.allow_cpu,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
