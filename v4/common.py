"""Reproducible text routing and real transformer-prefix early exits.

No V3 dependency. A selected depth applies to prefill AND all decode tokens.
The original final RMSNorm and LM head decode the chosen prefix's output.
Qwen's weights are frozen; this module does not adapt intermediate exits.
"""
from __future__ import annotations

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

DEPTHS = (12, 24, 36)
MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
PROTOCOL = "v4-question-text-prefix-depth-12-24-36-v1"
PROMPT_SUFFIX = (
    "\nSolve carefully. You may use a short calculation. "
    "End with a line exactly: FINAL: <number>."
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def text_key(text):
    return identity(" ".join(text.split()).casefold())


def environment():
    packages = {}
    for name in ("torch", "transformers", "datasets", "numpy", "pyarrow", "scikit-learn", "joblib", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    # Lazy Torch import keeps CPU text training independent of GPU libraries.
    import sys
    torch = sys.modules.get("torch")
    gpu = None
    if torch is not None and torch.cuda.is_available():
        gpu = {"name": torch.cuda.get_device_name(0), "hip": torch.version.hip,
               "cuda": torch.version.cuda,
               "total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "gpu": gpu, "repository_commit": commit}


def manifest(directory, config, stage):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = {**config, "protocol": PROTOCOL,
              "code_sha256": {name: digest(Path(__file__).parent / name)
                              for name in {"common.py", f"{stage}.py"}}}
    run_identity = identity(config)
    path = directory / "manifest.json"
    if path.exists():
        previous = read_json(path)
        if previous["identity"] != run_identity:
            raise ValueError("Inputs, settings or code changed. Use a NEW output directory.")
        if stage in {"label", "benchmark"}:
            current = environment()
            if previous["environment"]["packages"] != current["packages"] or previous["environment"]["gpu"] != current["gpu"]:
                raise ValueError("Generation environment changed. Use a new output directory.")
        return previous
    result = {"identity": run_identity, "config": config, "environment": environment()}
    atomic_json(path, result)
    return result


class Progress:
    """Local heartbeat, independent of generation progress or Drive uploads."""
    def __init__(self, directory):
        self.path = Path(directory) / "run_status.json"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.state = {"status": "running", "phase": "starting", "pid": os.getpid(),
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
        self.update(status="completed" if kind is None else "interrupted" if kind is KeyboardInterrupt else "failed",
                    phase="finished" if kind is None else "saved_results_retained",
                    error=None if error is None else f"{kind.__name__}: {error}")


def settings(revision=MODEL_REVISION, max_new_tokens=1024, seed=42):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use an immutable 40-character model commit hash for --revision")
    return {"model_id": MODEL_ID, "model_revision": revision,
            "depths": list(DEPTHS), "max_new_tokens": max_new_tokens, "seed": seed,
            "do_sample": False, "enable_thinking": False, "precision": "float16",
            "attention": "sdpa", "prompt_suffix": PROMPT_SUFFIX,
            "intervention": "fixed transformer prefix for both prompt prefill and decode",
            "qwen_weights": "frozen_original_checkpoint_no_early_exit_finetuning"}


def select_questions(split, rows, seed, offset=0, revision=None, excluded=()):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    if rows < 1 or offset < 0:
        raise ValueError("rows must be positive and offset non-negative")
    resolved = HfApi().dataset_info("openai/gsm8k", revision=revision or "main").sha
    dataset = load_dataset("openai/gsm8k", "main", split=split, revision=resolved)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    seen, records = set(excluded), []
    for index in indices:
        row = dataset[index]
        key = text_key(row["question"])
        if key in seen:
            continue
        seen.add(key)
        records.append({"question_id": f"gsm8k-{split}-{index:05d}", "question_key": key,
                        "source_index": index, "question": row["question"],
                        "reference_solution": row["answer"],
                        "reference_answer": row["answer"].rsplit("####", 1)[1].strip()})
    chosen = records[offset:offset + rows]
    if len(chosen) != rows:
        raise ValueError("Not enough unique questions after exclusions and offset")
    return chosen, resolved


def load_model(revision):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("V4 generation requires transformers==4.57.1; install v4/requirements.txt")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen labeling/benchmarking requires a CUDA or ROCm GPU")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, dtype=torch.float16, attn_implementation="sdpa",
    ).to("cuda").eval().requires_grad_(False)
    if model.config.model_type != "qwen3" or len(model.model.layers) != 36:
        raise ValueError("Expected Qwen3-4B with 36 transformer decoder layers")
    return tokenizer, model


@contextlib.contextmanager
def prefix_depth(model, depth):
    """Only the first N transformer blocks execute, with a fresh per-run cache.

    Replace the layer list for the whole generation rather than installing
    checks/hooks in each layer or token. Restore structure/config on failure.
    The original module list remains alive, so all weights stay resident.
    """
    from torch import nn
    decoder = model.model
    original = decoder.layers
    if depth not in DEPTHS or len(original) != 36:
        raise ValueError("Expected a full 36-layer model and depth 12, 24 or 36")
    configs = {id(c): c for c in (model.config, decoder.config)}
    old_counts = [(c, c.num_hidden_layers, c.layer_types) for c in configs.values()]
    try:
        decoder.layers = nn.ModuleList(list(original)[:depth])
        for config, _, layer_types in old_counts:
            config.num_hidden_layers = depth
            config.layer_types = layer_types[:depth]
        yield
    finally:
        decoder.layers = original
        for config, count, layer_types in old_counts:
            config.num_hidden_layers = count
            config.layer_types = layer_types


def parsed_answer(text):
    matches = re.findall(
        r"(?im)^\s*(?:\*\*|__)?\s*FINAL\s*:\s*(?:\*\*|__)?\s*"
        r"(?:[$£€]\s*)?([-+]?\d[\d,]*(?:\.\d+)?)\s*(?:\*\*|__)?\s*[.!]?\s*$", text,
    )
    return matches[-1].replace(",", "") if matches else None


def correct_answer(answer, reference):
    if answer is None:
        return False
    try:
        return Decimal(answer.replace(",", "")) == Decimal(reference.replace(",", ""))
    except InvalidOperation:
        return False


def generate(tokenizer, model, question, depth, config):
    """Time tokenization, prefix setup and generation, excluding model load/I/O."""
    import copy
    import torch
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    gpu = model.device.type == "cuda"
    if gpu:
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question["question"] + config["prompt_suffix"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    generation = copy.deepcopy(model.generation_config)
    generation.do_sample = False
    generation.temperature, generation.top_p, generation.top_k = 1.0, 1.0, 50
    generation.num_beams = 1
    generation.pad_token_id = tokenizer.eos_token_id
    generation.use_cache = True
    with prefix_depth(model, depth), torch.inference_mode():
        outputs = model.generate(**inputs, generation_config=generation,
                                 max_new_tokens=config["max_new_tokens"], logits_to_keep=1)
    if gpu:
        torch.cuda.synchronize()
    input_count = inputs["input_ids"].shape[1]
    ids = outputs[0, input_count:]
    eos = generation.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    complete = bool(len(ids) and ids[-1].item() in eos_ids)
    response = tokenizer.decode(ids, skip_special_tokens=True)
    answer = parsed_answer(response)
    elapsed = time.perf_counter() - started
    return {"question_id": question["question_id"], "depth": depth, "response": response,
            "extracted_answer": answer, "complete": complete,
            "finish_reason": "eos" if complete else "length",
            "correct": bool(complete and correct_answer(answer, question["reference_answer"])),
            "input_tokens": input_count, "output_tokens": len(ids),
            "seconds": elapsed, "tokens_per_second": len(ids) / elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if gpu else None,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3 if gpu else None,
            "decoder_layer_token_evaluations": depth * (input_count + max(len(ids) - 1, 0))}


def warmup(tokenizer, model, config):
    warm_config = {**config, "max_new_tokens": 8}
    for depth in DEPTHS:
        generate(tokenizer, model, {"question_id": "warmup", "question": "What is 17 + 25?",
                                   "reference_answer": "42"}, depth, warm_config)


def training_target(outcomes):
    if not outcomes["36"]["correct"]:
        return None, "full_depth_not_complete_correct"
    return min(d for d in DEPTHS if outcomes[str(d)]["correct"]), "shallowest_observed_complete_correct"


def load_labeled(directory):
    directory = Path(directory)
    source = read_json(directory / "manifest.json")
    if source["config"]["protocol"] != PROTOCOL:
        raise ValueError("Wrong labeling protocol")
    if source["identity"] != identity(source["config"]):
        raise ValueError("Label manifest checksum mismatch")
    records = read_json(directory / "questions.json")
    if source["config"]["questions_sha256"] != identity(records):
        raise ValueError("Question list checksum mismatch")
    joined = []
    evidence = {}
    for question in records:
        if question["question_key"] != text_key(question["question"]):
            raise ValueError("Question key differs from question text")
        path = directory / "questions" / question["question_id"] / "label.json"
        row = read_json(path)
        if row["question_key"] != question["question_key"]:
            raise ValueError("Question/label mismatch")
        target, reason = training_target(row["outcomes"])
        if target != row["target_depth"] or reason != row["label_reason"]:
            raise ValueError("Label inconsistent with saved outcomes")
        for depth in DEPTHS:
            trial_path = path.parent / f"depth_{depth}.json"
            if digest(trial_path) != row["trial_sha256"][str(depth)]:
                raise ValueError("Trial checksum mismatch")
            trial = read_json(trial_path)
            if trial["correct"] != row["outcomes"][str(depth)]["correct"]:
                raise ValueError("Trial/label correctness mismatch")
        evidence[question["question_id"]] = digest(path)
        joined.append({**question, **row})
    return joined, source, identity(evidence)


def route(pipeline, texts, threshold):
    """Only question strings enter the predictor; no answers or activations."""
    import numpy as np
    probabilities = pipeline.predict_proba(texts)
    indices = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(texts)), indices]
    raw = pipeline.classes_[indices].astype(int)
    selected = np.where(confidence >= threshold, raw, 36)
    return selected, confidence, raw


def decision_metrics(records, depths):
    if len(records) != len(depths) or not records:
        raise ValueError("Expected non-empty paired records and decisions")
    n = len(records)
    chosen = [r["outcomes"][str(int(d))]["correct"] for r, d in zip(records, depths)]
    full = [r["outcomes"]["36"]["correct"] for r in records]
    eligible = [(r["target_depth"], int(d)) for r, d in zip(records, depths) if r["target_depth"] is not None]
    confusion = [[sum(a == true and b == predicted for a, b in eligible) for predicted in DEPTHS] for true in DEPTHS]
    f1 = []
    for index, d in enumerate(DEPTHS):
        tp = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        if support or predicted:
            f1.append(2 * tp / (support + predicted))
    return {"questions": n, "accuracy_all_questions": sum(chosen) / n,
            "baseline_accuracy_all_questions": sum(full) / n,
            "accuracy_difference": (sum(chosen) - sum(full)) / n,
            "correct_to_wrong": sum(a and not b for a, b in zip(full, chosen)),
            "wrong_to_correct": sum(not a and b for a, b in zip(full, chosen)),
            "mean_depth": sum(map(int, depths)) / n,
            "depth_counts": {str(d): sum(int(x) == d for x in depths) for d in DEPTHS},
            "completed": sum(r["outcomes"][str(int(d))]["complete"] for r, d in zip(records, depths)),
            "target_depth_accuracy_eligible_only": sum(a == b for a, b in eligible) / len(eligible) if eligible else None,
            "target_depth_macro_f1": sum(f1) / len(f1) if f1 else None,
            "target_depth_confusion": {"labels": list(DEPTHS), "rows_true_columns_selected": confusion},
            "eligible_target_questions": len(eligible)}


def load_controller(path):
    import joblib
    controller = joblib.load(path)  # Only load your own trusted artifact.
    if controller["protocol"] != PROTOCOL:
        raise ValueError("Wrong controller protocol")
    return controller
