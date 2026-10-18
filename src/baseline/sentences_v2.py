"""Generate GSM8K traces with the authors' prompt and sentence boundaries.

python -m src.baseline.sentences_v2 --compare-v1 --limit 1

This adapts the published procedure to Qwen3/GSM8K. It does not promise the
same wording as the authors' models, or produce removability labels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

from src.baseline.cot import build_cot_prompt
from src.black_hole.sentence_checkpoint import atomic_json
from src.black_hole.sentence_experiment import split_sentences
from src.data.gsm8k import normalize_example
from src.evaluation.accuracy import evaluate_prediction
from src.evaluation.metrics import InferenceMeasurement
from src.models.qwen import GenerationSettings
from src.utils.environment import collect_environment_info, load_qwen_model

UPSTREAM_COMMIT = "ceeb7201b55858476a21e9e63e145ba2916ab159"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def build_v2_prompt(question: str) -> str:
    """Exact upstream wrap_math_boxed_prompt string, including final newline."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return (
        "Solve the following problem. You MUST put your final answer in "
        "\\boxed{}. Problem: " + question + "\n"
    )


def split_trace(raw_text: str) -> dict:
    """Extract Qwen thinking text, preserving offsets into the saved raw output."""
    opening = raw_text.find("<think>")
    start = opening + len("<think>") if opening >= 0 else 0
    closing = raw_text.find("</think>", start)
    end = closing if closing >= 0 else len(raw_text)
    cot = raw_text[start:end]
    return {
        "thinking_closed": closing >= 0,
        "reasoning_text": cot,
        "answer_text": raw_text[closing + len("</think>"):] if closing >= 0 else "",
        "sentences": [
            {"sentence_id": f"T{i + 1}", "text": text,
             "start_char_in_cot": left, "end_char_in_cot": right,
             "start_char_in_raw_output": start + left,
             "end_char_in_raw_output": start + right}
            for i, (text, left, right) in enumerate(split_sentences(cot))
        ],
    }


def generate_trace(tokenizer, model, prompt: str, settings: GenerationSettings) -> dict:
    """Same greedy decoding policy as V1, retaining structural tokens for parsing."""
    import torch

    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    inputs = tokenizer([rendered], return_tensors="pt", add_special_tokens=False)
    inputs = inputs.to(model.device)
    input_count = int(inputs["input_ids"].shape[-1])
    with InferenceMeasurement(torch) as measurement, torch.inference_mode():
        output = model.generate(
            **inputs, **settings.model_kwargs(), pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, input_count:]
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    finished = bool(len(generated)) and int(generated[-1]) in eos
    return {
        "prompt": prompt, "rendered_prompt": rendered,
        "raw_output": tokenizer.decode(generated, skip_special_tokens=False),
        "generated_token_ids": generated.tolist(),
        "reasoning_trace": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "input_tokens": input_count, "output_tokens": len(generated),
        "finish_reason": "eos" if finished else "length" if len(generated) >= settings.max_new_tokens else "unknown",
        "generation_config": settings.to_dict(), **measurement.to_dict(),
    }


def source_rows(args) -> tuple[list[dict], dict]:
    if args.input_jsonl:
        rows = []
        with args.input_jsonl.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append({"id": row.get("id") or row["problem_id"],
                             "question": row["question"],
                             "reference_answer": row["reference_answer"]})
                if len(rows) >= args.limit:
                    break
        metadata = {"input_jsonl": str(args.input_jsonl.resolve()),
                    "sha256": hashlib.sha256(args.input_jsonl.read_bytes()).hexdigest()}
    else:
        from datasets import load_dataset

        data = load_dataset("openai/gsm8k", "main", revision=args.dataset_revision,
                            split=f"train[{args.start_index}:{args.start_index + args.limit}]")
        rows = [normalize_example(row, split="train", index=args.start_index + i)
                for i, row in enumerate(data)]
        metadata = {"dataset": "openai/gsm8k", "config": "main", "split": "train",
                    "start_index": args.start_index, "requested_revision": args.dataset_revision,
                    "fingerprint": data._fingerprint}
    if not rows:
        raise ValueError("No source questions found")
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError("Source contains duplicate problem IDs")
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1, help="Number of distinct questions; defaults to one")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--input-jsonl", type=Path, help="Optional existing normalized/raw-attempt JSONL; uses its first N rows")
    parser.add_argument("--compare-v1", action="store_true", help="One old-prompt and one V2-prompt generation per question")
    parser.add_argument("--prepare-only", action="store_true", help="Save real inputs/prompts without invoking a model")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Shared cap; project default 1024. Authors' script default was 16380.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sentence_generation_v2"))
    args = parser.parse_args()
    if args.limit < 1 or args.start_index < 0 or args.max_new_tokens < 1:
        parser.error("limit/token cap must be positive; start index must be nonnegative")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output_dir / stamp
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"protocol": "pssn-gsm8k-authors-sentences-v2", "upstream_commit": UPSTREAM_COMMIT,
                "status": "preparing", "model_id": args.model_id, "revision": args.revision,
                "max_new_tokens": args.max_new_tokens, "seed": args.seed,
                "do_sample": False, "thinking_enabled": True,
                "adaptations": ["Qwen3-4B and GSM8K replace authors' models and datasets",
                                "Shared project token cap; configurable independently of sentence rules"],
                "labels_generated": False, "sentences_file": "each version record contains sentences"}
    manifest["code_sha256"] = {
        str(path.relative_to(Path(__file__).resolve().parents[2])):
            hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__).resolve(),
                     Path(__file__).resolve().parents[1] / "black_hole/sentence_experiment.py",
                     Path(__file__).resolve().with_name("cot.py")]
    }
    manifest["packages"] = {}
    for package in ("torch", "transformers", "accelerate", "datasets"):
        try:
            manifest["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest["packages"][package] = None
    manifest_path = destination / "manifest.json"
    atomic_json(manifest_path, manifest)
    try:
        rows, source = source_rows(args)
        manifest.update(source=source, question_count=len(rows))
        versions = ["v1", "v2"] if args.compare_v1 else ["v2"]
        plans = [{**row, "prompts": {version: (build_cot_prompt(row['question']) if version == 'v1'
                                              else build_v2_prompt(row['question']))
                                    for version in versions}} for row in rows]
        atomic_json(destination / "inputs.json", plans)
        if args.prepare_only:
            manifest["status"] = "prepared_only_no_generation"
            print(json.dumps(plans, ensure_ascii=False, indent=2))
            return
        import torch

        manifest["environment"] = collect_environment_info(torch)
        atomic_json(manifest_path, manifest)
        if not torch.cuda.is_available():
            raise RuntimeError("Generation requires CUDA/ROCm-enabled PyTorch and enough model memory. "
                               "This run generated no model output.")
        tokenizer, model, precision = load_qwen_model(model_id=args.model_id, revision=args.revision)
        manifest.update(status="generating", precision=precision,
                        resolved_model_revision=getattr(model.config, "_commit_hash", None) or args.revision,
                        resolved_tokenizer_revision=tokenizer.init_kwargs.get("_commit_hash") or args.revision)
        atomic_json(manifest_path, manifest)
        settings = GenerationSettings(max_new_tokens=args.max_new_tokens, seed=args.seed, do_sample=False)
        for plan in plans:
            print(f"\n{plan['id']}: {plan['question']}\nReference answer: {plan['reference_answer']}", flush=True)
            for version, prompt in plan["prompts"].items():
                record = generate_trace(tokenizer, model, prompt, settings)
                parsed = split_trace(record["raw_output"])
                complete = record["finish_reason"] == "eos" and parsed["thinking_closed"]
                evaluation = evaluate_prediction(
                    problem_id=plan["id"], reference_answer=plan["reference_answer"],
                    generated_response=parsed["answer_text"], incomplete=not complete,
                )
                record.update(version=version, problem_id=plan["id"], question=plan["question"],
                              reference_answer=plan["reference_answer"], **parsed,
                              evaluation=evaluation, sentence_count=len(parsed["sentences"]),
                              eligible_for_removal_screening=complete and evaluation["is_correct"] is True)
                atomic_json(destination / f"{plan['id']}_{version}.json", record)
                print(f"\n{version}: {record['output_tokens']} tokens | stop={record['finish_reason']} "
                      f"| {evaluation['evaluation_status']} | correct={evaluation['is_correct']}", flush=True)
                for sentence in parsed["sentences"]:
                    print(f"{sentence['sentence_id']}: {sentence['text'].strip()}", flush=True)
                print(f"FINAL RESPONSE: {parsed['answer_text'].strip()}", flush=True)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(manifest_path, manifest)
        print(f"Results: {destination.resolve()}", flush=True)


if __name__ == "__main__":
    main()
