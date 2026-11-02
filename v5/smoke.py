"""Run five GSM8K questions through a released LayerSkip model at 24/28/32.

No model fine-tuning or controller training. Run: python -m v5.smoke --help
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

MODEL_ID = "facebook/layerskip-llama3-8B"
DEPTHS = (24, 28, 32)
PROTOCOL = "v5-layerskip-llama3-8b-fixed-prefix-24-28-32-v1"
INSTRUCTION = (
    "Solve each math problem. Show the calculation and end the answer with "
    "a line of the form FINAL: <number>.\n\n"
)
STOP_STRING = "\nQuestion:"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def text_key(text):
    return fingerprint(" ".join(text.split()).casefold())


class Progress:
    def __init__(self, directory):
        self.path = directory / "run_status.json"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.state = {"pid": os.getpid(), "status": "running", "phase": "starting",
                      "started_at_utc": datetime.now(timezone.utc).isoformat(),
                      "heartbeat_stale_after_seconds": 120}

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            self.state["heartbeat_at_utc"] = datetime.now(timezone.utc).isoformat()
            self.state["is_running"] = self.state["status"] == "running"
            atomic_json(self.path, self.state)

    def heartbeat(self):
        while not self.stop.wait(15):
            self.update()

    def __enter__(self):
        self.update()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        self.thread.join()
        status = "completed" if kind is None else "interrupted" if kind is KeyboardInterrupt else "failed"
        self.update(status=status, phase="finished" if kind is None else "saved_results_retained",
                    error=None if error is None else f"{kind.__name__}: {error}")


def environment(torch):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    props = torch.cuda.get_device_properties(0)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "datasets", "numpy", "pyarrow", "huggingface-hub")},
            "gpu_name": props.name, "gpu_total_memory_gib": props.total_memory / 1024**3,
            "hip_version": torch.version.hip, "cuda_version": torch.version.cuda,
            "repository_commit": commit}


def load_data(rows, seed, dataset_revision):
    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train", revision=dataset_revision)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    def record(index):
        row = dataset[index]
        return {"question_id": f"gsm8k-train-{index:05d}", "source_index": index,
                "question": row["question"], "reference_solution": row["answer"],
                "reference_answer": row["answer"].rsplit("####", 1)[1].strip()}

    questions = [record(index) for index in indices[:rows]]
    excluded = {text_key(q["question"]) for q in questions}
    examples = []
    for index in range(len(dataset)):
        candidate = record(index)
        if text_key(candidate["question"]) not in excluded:
            examples.append(candidate)
        if len(examples) == 3:
            break
    if len(questions) != rows or len(examples) != 3:
        raise ValueError("Not enough disjoint questions and demonstration examples")
    return questions, examples


def make_prompt(question, examples):
    parts = [INSTRUCTION]
    for example in examples:
        work = example["reference_solution"].rsplit("####", 1)[0].strip()
        work = re.sub(r"<<.*?>>", "", work)
        parts.append(f"Question: {example['question']}\nAnswer: {work}\n"
                     f"FINAL: {example['reference_answer']}\n\n")
    # The current question's reference solution/answer never enters the prompt.
    parts.append(f"Question: {question['question']}\nAnswer:")
    return "".join(parts)


def parsed_answer(text):
    matches = re.findall(
        r"(?im)^\s*(?:\*\*|__)?\s*(?:FINAL\s*:|####)\s*(?:\*\*|__)?\s*"
        r"(?:[$£€]\s*)?([-+]?\d[\d,]*(?:\.\d+)?)\s*(?:\*\*|__)?\s*[.!]?\s*$", text)
    return matches[-1].replace(",", "") if matches else None


def numeric_equal(answer, reference):
    if answer is None:
        return False
    try:
        return Decimal(answer.replace(",", "")) == Decimal(reference.replace(",", ""))
    except InvalidOperation:
        return False


@contextlib.contextmanager
def prefix_depth(model, depth):
    """Run only the first N whole transformer blocks; preserve norm and LM head."""
    import torch
    layers = model.model.layers
    if len(layers) != 32 or depth not in DEPTHS:
        raise ValueError("Expected the full 32-layer LayerSkip Llama3-8B model")
    configs = {id(config): config for config in (model.config, model.model.config)}
    saved = [(config, config.num_hidden_layers) for config in configs.values()]
    try:
        model.model.layers = torch.nn.ModuleList(list(layers)[:depth])
        for config, _ in saved:
            config.num_hidden_layers = depth
        yield
    finally:
        model.model.layers = layers
        for config, count in saved:
            config.num_hidden_layers = count


def generate(tokenizer, model, question, examples, depth, token_cap):
    import torch
    from transformers import GenerationConfig
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    prompt = make_prompt(question, examples)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    config = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
                              eos_token_id=model.generation_config.eos_token_id or tokenizer.eos_token_id,
                              pad_token_id=tokenizer.eos_token_id, max_new_tokens=token_cap)
    with prefix_depth(model, depth), torch.inference_mode():
        output = model.generate(**inputs, generation_config=config,
                                stop_strings=[STOP_STRING], tokenizer=tokenizer, logits_to_keep=1)
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
    return {"question_id": question["question_id"], "depth": depth,
            "reference_answer": question["reference_answer"], "response": response,
            "raw_response": raw_response, "generated_token_ids": generated.tolist(),
            "extracted_answer": extracted, "complete": complete,
            "finish_reason": "eos" if stopped_on_eos else "next_question" if stopped_on_boundary else "length",
            "numeric_answer_matches": numeric_equal(extracted, question["reference_answer"]),
            "correct": complete and numeric_equal(extracted, question["reference_answer"]),
            "input_tokens": input_tokens, "output_tokens": len(generated),
            "seconds": seconds, "output_tokens_per_second": len(generated) / seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "decoder_layer_token_evaluations": depth * (input_tokens + max(len(generated) - 1, 0)),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def summarize(trials, questions):
    results = {}
    for depth in DEPTHS:
        rows = [row for row in trials if row["depth"] == depth]
        n = len(rows)
        seconds = sum(row["seconds"] for row in rows)
        tokens = sum(row["output_tokens"] for row in rows)
        results[str(depth)] = {"trials": n, "correct": sum(row["correct"] for row in rows),
            "accuracy": sum(row["correct"] for row in rows) / n if n else None,
            "completed": sum(row["complete"] for row in rows),
            "missing_answer": sum(row["extracted_answer"] is None for row in rows),
            "incomplete": sum(not row["complete"] for row in rows),
            "input_tokens": sum(row["input_tokens"] for row in rows), "output_tokens": tokens,
            "total_seconds": seconds, "mean_seconds": seconds / n if n else None,
            "aggregate_output_tokens_per_second": tokens / seconds if seconds else None,
            "max_peak_allocated_gib": max((row["peak_allocated_gib"] for row in rows), default=None),
            "max_peak_reserved_gib": max((row["peak_reserved_gib"] for row in rows), default=None)}
    by_key = {(row["question_id"], row["depth"]): row for row in trials}
    paired = {}
    for depth in (24, 28):
        pairs = [(by_key[(q["question_id"], 32)], by_key[(q["question_id"], depth)])
                 for q in questions if (q["question_id"], 32) in by_key
                 and (q["question_id"], depth) in by_key]
        shallow_seconds = sum(b["seconds"] for a, b in pairs)
        paired[str(depth)] = {"paired_questions": len(pairs),
            "correct_to_wrong": sum(a["correct"] and not b["correct"] for a, b in pairs),
            "wrong_to_correct": sum(not a["correct"] and b["correct"] for a, b in pairs),
            "baseline_seconds_divided_by_exit_seconds":
                sum(a["seconds"] for a, b in pairs) / shallow_seconds if shallow_seconds else None,
            "output_token_difference": sum(b["output_tokens"] - a["output_tokens"] for a, b in pairs)}
    return {"status": "completed" if len(trials) == len(questions) * 3 else "running",
            "completed_trials": len(trials), "total_trials": len(questions) * 3,
            "depth_results": results, "paired_vs_depth_32": paired,
            "notes": ["Five-question default is a feasibility check, not a reliable accuracy estimate.",
                      "Direct fixed-depth generation; no speculative verification and no training.",
                      "Incomplete and missing-final answers count as incorrect in accuracy.",
                      "Timing includes prompt tokenization, prefix setup and synchronized generation; excludes model load and result I/O.",
                      "All weights stay on GPU. Reserved memory can include allocator history across depths.",
                      "GSM8K train questions are development examples; use an untouched test split for final claims."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--revision", help="Optional model commit; otherwise resolve main once and save it")
    parser.add_argument("--dataset-revision", help="Optional GSM8K commit; otherwise resolve and save it")
    args = parser.parse_args()
    if not 1 <= args.rows <= 100 or args.max_new_tokens < 1:
        parser.error("--rows must be 1..100 and --max-new-tokens must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with Progress(args.output_dir) as progress:
        import torch
        import transformers
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if transformers.__version__ != "4.57.1":
            raise RuntimeError("Install v5/requirements.txt (transformers==4.57.1)")
        if not torch.cuda.is_available():
            raise RuntimeError("A ROCm or CUDA GPU is required")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        progress.update(phase="resolving_model_and_data")
        path = args.output_dir / "manifest.json"
        previous = read_json(path) if path.exists() else None
        api = HfApi()
        model_revision = args.revision or (previous["config"]["model_revision"] if previous else None)
        dataset_revision = args.dataset_revision or (previous["config"]["dataset_revision"] if previous else None)
        model_revision = api.model_info(MODEL_ID, revision=model_revision or "main").sha
        dataset_revision = api.dataset_info("openai/gsm8k", revision=dataset_revision or "main").sha
        questions, examples = load_data(args.rows, args.seed, dataset_revision)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        config = {"protocol": PROTOCOL, "model_id": MODEL_ID, "model_revision": model_revision,
                  "tokenizer_revision": model_revision, "dataset_id": "openai/gsm8k",
                  "dataset_config": "main", "dataset_split": "train", "dataset_revision": dataset_revision,
                  "rows": args.rows, "seed": args.seed, "depths": list(DEPTHS),
                  "max_new_tokens": args.max_new_tokens, "dtype": str(dtype), "attention": "sdpa",
                  "do_sample": False, "num_beams": 1, "few_shot_count": 3,
                  "questions_sha256": fingerprint(questions), "examples_sha256": fingerprint(examples),
                  "prompt_instruction": INSTRUCTION, "stop_string": STOP_STRING,
                  "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "trial_order": "rotate [32,24,28] by question index; one generation per depth"}
        env = environment(torch)
        signature = fingerprint({"config": config, "environment": env})
        if previous and previous["identity"] != signature:
            raise ValueError("Settings, source, or environment changed; choose a NEW output directory")
        atomic_json(path, {"identity": signature, "config": config, "environment": env})
        atomic_json(args.output_dir / "questions.json", questions)
        atomic_json(args.output_dir / "demonstrations.json", examples)
        for question in questions:
            atomic_json(args.output_dir / "questions" / question["question_id"] / "prompt.json",
                        {"prompt": make_prompt(question, examples)})
        progress.update(phase="loading_model", total_trials=args.rows * 3)
        print(f"Loading {MODEL_ID} revision={model_revision} dtype={dtype}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_revision)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=model_revision, dtype=dtype, attn_implementation="sdpa",
        ).to("cuda").eval().requires_grad_(False)
        if model.config.model_type != "llama" or len(model.model.layers) != 32:
            raise ValueError("Expected a 32-layer Llama checkpoint")
        print("Loaded. Warming up 24, 28 and 32 layers...", flush=True)
        progress.update(phase="warmup")
        warm_question = {"question_id": "warmup", "question": "What is 17 plus 25?", "reference_answer": "42"}
        for depth in DEPTHS:
            generate(tokenizer, model, warm_question, examples, depth, 8)
        trials = []
        for index, question in enumerate(questions):
            order = [32, 24, 28]
            shift = index % 3
            order = order[shift:] + order[:shift]
            for depth in order:
                progress.update(phase="generating", current_question=question["question_id"],
                                current_depth=depth, completed_trials=len(trials))
                trial_path = args.output_dir / "questions" / question["question_id"] / f"depth_{depth}.json"
                if trial_path.exists():
                    trial = read_json(trial_path)
                    if trial["run_identity"] != signature or trial["depth"] != depth or trial["question_id"] != question["question_id"]:
                        raise ValueError(f"Saved trial does not match this run: {trial_path}")
                    reused = True
                else:
                    trial = generate(tokenizer, model, question, examples, depth, args.max_new_tokens)
                    trial["run_identity"] = signature
                    atomic_json(trial_path, trial)
                    reused = False
                trials.append(trial)
                atomic_json(args.output_dir / "summary.json", summarize(trials, questions))
                progress.update(completed_trials=len(trials))
                print(f"{question['question_id']} / depth {depth}: correct={trial['correct']} "
                      f"complete={trial['complete']} answer={trial['extracted_answer']} "
                      f"expected={question['reference_answer']} tokens={trial['output_tokens']} "
                      f"seconds={trial['seconds']:.2f}" + (" [saved]" if reused else ""), flush=True)
        print(json.dumps(summarize(trials, questions), indent=2), flush=True)


if __name__ == "__main__":
    main()
