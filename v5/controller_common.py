"""Shared evidence, scoring and direct-depth generation for the V5 controller."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import re
import sys
import time
from pathlib import Path

from .smoke import (DEPTHS, MODEL_ID, STOP_STRING, Progress, atomic_json,
                    fingerprint, make_prompt, numeric_equal, prefix_depth,
                    read_json, text_key)

PROTOCOL = "v5-question-depth-outcome-controller-24-28-32-v2"
MODEL_REVISION = "3b9e03eaf3cc90812a7bd2b69106b626d63bac1f"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
PILOT_INDICES = {1458, 3620, 6312, 4833, 5032}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    packages = {}
    for name in ("torch", "transformers", "datasets", "numpy", "scikit-learn", "joblib", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    torch = sys.modules.get("torch")
    gpu = None
    if torch is not None and torch.cuda.is_available():
        gpu = {"name": torch.cuda.get_device_name(0), "hip": torch.version.hip,
               "cuda": torch.version.cuda,
               "selected_dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
               "memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "gpu": gpu}


def manifest(directory, config, stage, require_same_environment=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = {**config, "protocol": PROTOCOL,
              "code_sha256": {name: digest(Path(__file__).parent / name) for name in
                              ("controller_common.py", "smoke.py", f"controller_{stage}.py")}}
    signature = fingerprint(config)
    path = directory / "manifest.json"
    env = environment()
    if path.exists():
        previous = read_json(path)
        if previous["identity"] != signature:
            raise ValueError("Inputs/settings/code changed. Use a NEW output directory.")
        if require_same_environment and previous["environment"] != env:
            raise ValueError("Generation environment changed. Use a NEW output directory.")
        return previous
    value = {"identity": signature, "config": config, "environment": env}
    atomic_json(path, value)
    return value


def record(index, row, split):
    return {"question_id": f"gsm8k-{split}-{index:05d}", "source_index": index,
            "question_key": text_key(row["question"]), "question": row["question"],
            "reference_solution": row["answer"],
            "reference_answer": row["answer"].rsplit("####", 1)[1].strip()}


def prepare_candidates(seed, dataset_revision=DATASET_REVISION):
    from datasets import load_dataset
    data = load_dataset("openai/gsm8k", "main", split="train", revision=dataset_revision)
    examples = [record(i, data[i], "train") for i in range(3)]
    seen = {r["question_key"] for r in examples}
    # The five development smoke questions are excluded from the new dataset.
    seen.update(text_key(data[i]["question"]) for i in PILOT_INDICES)
    indices = list(range(len(data)))
    random.Random(seed).shuffle(indices)
    candidates = []
    for index in indices:
        row = record(index, data[index], "train")
        if row["question_key"] not in seen:
            candidates.append(row)
            seen.add(row["question_key"])
    return candidates, examples


def parse_answer(response):
    """Accept explicit FINAL markers inline or on a new line; never guess from other numbers."""
    first_answer = response.split(STOP_STRING, 1)[0]
    matches = re.findall(
        r"(?i)(?:\bFINAL\s*:|####)\s*(?:\*\*|__)?\s*(?:[$£€]\s*)?"
        r"([-+]?\d[\d,]*(?:\.\d+)?)(?![\d,/]|\.\d)", first_answer)
    return matches[-1].replace(",", "") if matches else None


def score_response(raw, last_is_eos, reference):
    before, marker, after = raw.partition(STOP_STRING)
    overrun = bool(marker and after.strip())
    boundary_end = bool(marker and not after.strip())
    complete = not overrun and (last_is_eos or boundary_end)
    response = before.strip() if marker else raw.strip()
    answer = parse_answer(response)
    return {"response": response, "extracted_answer": answer, "complete": complete,
            "finish_reason": "boundary_overrun" if overrun else "next_question" if boundary_end
                             else "eos" if last_is_eos else "length",
            "correct": bool(complete and numeric_equal(answer, reference))}


def target_depth(outcomes):
    if not (outcomes["32"]["correct"] and outcomes["32"]["complete"]):
        return None
    return min(d for d in DEPTHS if outcomes[str(d)]["correct"] and outcomes[str(d)]["complete"])


def load_model(revision=MODEL_REVISION, seed=42):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, StopStringCriteria, StoppingCriteriaList
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("Install v5/requirements.txt (transformers==4.57.1)")
    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires a ROCm or CUDA GPU")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a pinned 40-character model revision")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, dtype=dtype, attn_implementation="sdpa",
    ).to("cuda").eval().requires_grad_(False)
    if model.config.model_type != "llama" or len(model.model.layers) != 32:
        raise ValueError("Expected the 32-layer LayerSkip Llama3-8B checkpoint")
    tokenizer._pssn_controller_stop = StoppingCriteriaList([StopStringCriteria(tokenizer, [STOP_STRING])])
    return tokenizer, model


def generate(tokenizer, model, question, examples, depth, token_cap=1024):
    """Plain greedy generation at a fixed prefix; no draft/verification calls."""
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
                                stopping_criteria=tokenizer._pssn_controller_stop,
                                logits_to_keep=1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    input_tokens = inputs["input_ids"].shape[1]
    ids = output[0, input_tokens:]
    eos = config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]
    raw = tokenizer.decode(ids, skip_special_tokens=True)
    return {"question_id": question["question_id"], "depth": depth,
            "dtype": str(model.dtype), "attention": "sdpa", "do_sample": False,
            "reference_answer": question["reference_answer"], "raw_response": raw,
            "generated_token_ids": ids.tolist(),
            **score_response(raw, bool(len(ids) and ids[-1].item() in eos), question["reference_answer"]),
            "input_tokens": input_tokens, "output_tokens": len(ids), "seconds": elapsed,
            "output_tokens_per_second": len(ids) / elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "decoder_layer_token_evaluations": depth * (input_tokens + max(len(ids) - 1, 0)),
            "prompt_sha256": fingerprint(prompt)}


def warmup(tokenizer, model, examples):
    question = {"question_id": "warmup", "question": "What is 17 plus 25?", "reference_answer": "42"}
    for depth in DEPTHS:
        generate(tokenizer, model, question, examples, depth, 8)


def checked_trial(path, question, identity, depth):
    row = read_json(path)
    if (row.get("run_identity") != identity or row.get("question_id") != question["question_id"]
            or row.get("depth") != depth):
        raise ValueError(f"Saved trial belongs to different settings: {path}")
    return row


def load_labels(directory):
    directory = Path(directory)
    source = read_json(directory / "manifest.json")
    if source["identity"] != fingerprint(source["config"]) or source["config"]["protocol"] != PROTOCOL:
        raise ValueError("Invalid labeling manifest")
    rows = read_json(directory / "selected_dataset.json")
    expected_rows = int(source["config"]["question_count"])
    if len(rows) != expected_rows or len({r["question_key"] for r in rows}) != len(rows):
        raise ValueError("Expected the configured number of unique labeled questions")
    for row in rows:
        if row["question_key"] != text_key(row["question"]):
            raise ValueError("Question text/key inconsistency")
        if set(row["outcomes"]) != {str(d) for d in DEPTHS}:
            raise ValueError("Every question must have measured outcomes at all three depths")
        for depth in DEPTHS:
            path = directory / "questions" / row["question_id"] / f"depth_{depth}.json"
            if digest(path) != row["trial_sha256"][str(depth)]:
                raise ValueError(f"Trial checksum mismatch: {path}")
            trial = checked_trial(path, row, source["identity"], depth)
            expected = {key: trial[key] for key in row["outcomes"][str(depth)]}
            if expected != row["outcomes"][str(depth)]:
                raise ValueError("Dataset outcomes differ from saved generation evidence")
    return rows, source, digest(directory / "selected_dataset.json")


def stratified_partitions(rows, seed):
    """Split question IDs 70/15/15 so the three trials for one question stay together."""
    rng = random.Random(seed)
    ids = [r["question_id"] for r in rows]
    rng.shuffle(ids)
    a, b = int(0.7 * len(ids)), int(0.85 * len(ids))
    id_splits = {"train": set(ids[:a]), "validation": set(ids[a:b]), "test": set(ids[b:])}
    return {name: [r for r in rows if r["question_id"] in subset]
            for name, subset in id_splits.items()}


def model_text(question, depth):
    return f"{question} [DEPTH_{depth}]"


def route_depth(classifier, question, threshold=0.5):
    """Choose the shallowest depth whose predicted answer-preservation chance clears threshold."""
    texts = [model_text(question, depth) for depth in DEPTHS]
    probabilities = classifier.predict_proba(texts)
    classes = list(classifier.classes_)
    positive_index = classes.index(1)
    scores = {depth: float(probabilities[i][positive_index]) for i, depth in enumerate(DEPTHS)}
    selected = next((depth for depth in DEPTHS if scores[depth] >= threshold), DEPTHS[-1])
    return selected, scores


def route_metrics(rows, classifier, threshold=0.5):
    if not rows:
        raise ValueError("Expected nonempty questions")
    selected = [route_depth(classifier, r["question"], threshold)[0] for r in rows]
    chosen = [r["outcomes"][str(d)] for r, d in zip(rows, selected)]
    baseline = [r["outcomes"][str(DEPTHS[-1])] for r in rows]
    return {"questions": len(rows), "answer_accuracy": sum(r["correct"] and r["complete"] for r in chosen) / len(rows),
            "baseline_answer_accuracy": sum(r["correct"] and r["complete"] for r in baseline) / len(rows),
            "correct_to_wrong": sum(a["correct"] and a["complete"] and not (b["correct"] and b["complete"])
                                    for a, b in zip(baseline, chosen)),
            "depth_counts": {str(d): selected.count(d) for d in DEPTHS},
            "mean_depth": sum(selected) / len(selected),
            "selected_output_tokens": sum(r["output_tokens"] for r in chosen),
            "baseline_output_tokens": sum(r["output_tokens"] for r in baseline),
            "cached_selected_seconds": sum(r["seconds"] for r in chosen),
            "cached_baseline_seconds": sum(r["seconds"] for r in baseline),
            "threshold": threshold,
            "population_note": "Cached outcome evidence on the fixed 100-question collection; this is not a live speed benchmark."}


def load_controller(path):
    import joblib
    payload = joblib.load(path)  # Load only the checkpoint created by this project.
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Wrong controller protocol")
    return payload
