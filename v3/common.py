"""Shared V3 utilities. No dependency on the earlier sentence pipeline.

Intervention: prefill runs normally. A controller chooses at most one MLP
using its input at the last prompt token. That MLP returns zero on subsequent
cached decoding calls, preserving the decoder residual and all attention/KV
updates. This is a question-level decision, not a per-token learned policy.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import random
import re
import subprocess
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
PROTOCOL = "v3-question-level-single-mlp-decode-only-v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    import transformers
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "transformers": transformers.__version__,
        "hip": torch.version.hip, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit,
        "code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
    }


def execution_fingerprint():
    """Hash the parts that run Qwen, excluding parser and labeling code."""
    material = "\n".join((
        MODEL_ID, MODEL_REVISION, PROTOCOL,
        inspect.getsource(load_model), inspect.getsource(Bypass), inspect.getsource(generate),
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def prepare_manifest(directory, config):
    """Reject resume when inputs/settings/code change; never mix runs silently."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    current_environment = environment()
    if path.exists():
        old = read_json(path)
        if old["identity"] != identity:
            raise ValueError("Run configuration changed. Use a new output directory.")
        current_execution = execution_fingerprint()
        previous_execution = old.get("execution_fingerprint")
        old_hashes = old["environment"].get("code_sha256", {})
        current_hashes = current_environment["code_sha256"]
        changed_files = {
            name for name in set(old_hashes) | set(current_hashes)
            if old_hashes.get(name) != current_hashes.get(name)
        }
        # Permit upgrading the answer parser/label runner for existing saved
        # trials, but only when the Qwen loading, bypass and generation code is
        # unchanged. Record both code versions in the manifest for the audit.
        if previous_execution is not None and previous_execution != current_execution:
            raise ValueError("Qwen execution code changed. Use a new output directory.")
        if previous_execution is None and changed_files - {"common.py", "label.py"}:
            raise ValueError("V3 execution files changed. Use a new output directory.")
        if changed_files or previous_execution is None:
            history = old.setdefault("code_history", [])
            history.append({
                "recorded_unix": time.time(),
                "previous_code_sha256": old_hashes,
                "migration": "Updated answer parsing/label runner; Qwen execution fingerprint unchanged.",
            })
            old["environment"]["code_sha256"] = current_hashes
            old["execution_fingerprint"] = current_execution
            atomic_json(path, old)
        return old
    result = {
        "identity": identity, "config": config, "environment": current_environment,
        "execution_fingerprint": execution_fingerprint(), "code_history": [],
    }
    atomic_json(path, result)
    return result


def status(directory, **kwargs):
    atomic_json(Path(directory) / "run_status.json", {
        "updated_unix": time.time(), "pid": os.getpid(), **kwargs,
        "note": "Updated at trial boundaries; check PID for live process status.",
    })


def text_key(question):
    return hashlib.sha256(" ".join(question.split()).casefold().encode()).hexdigest()


def questions(split, count, offset=0, revision=None, seed=42, exclude_keys=()):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    resolved = HfApi().dataset_info("openai/gsm8k", revision=revision or "main").sha
    data = load_dataset("openai/gsm8k", "main", split=split, revision=resolved)
    order = list(range(len(data)))
    random.Random(seed).shuffle(order)
    seen = set(exclude_keys)
    eligible = []
    for i in order:
        key = text_key(data[i]["question"])
        if key not in seen:
            eligible.append(i)
            seen.add(key)
    chosen = eligible[offset:offset + count]
    if count < 1 or offset < 0 or len(chosen) != count:
        raise ValueError("Invalid question count/offset or not enough non-overlapping questions")
    records = []
    for i in chosen:
        row = data[i]
        records.append({
            "question_id": f"gsm8k-{split}-{i:05d}", "source_index": i,
            "question": row["question"], "question_key": text_key(row["question"]),
            "reference_solution": row["answer"],
            "reference_answer": row["answer"].rsplit("####", 1)[1].strip(),
        })
    return records, resolved


def load_model(revision=MODEL_REVISION):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("This generation stage requires the GPU runtime.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, dtype=torch.float16, attn_implementation="sdpa",
    ).to("cuda").eval()
    if model.config.model_type != "qwen3" or not hasattr(model.model, "layers"):
        raise ValueError("V3 expects the Qwen3 decoder architecture")
    model.requires_grad_(False)
    return tokenizer, model


def parsed_answer(text):
    """Read the requested final-answer line, with or without Markdown bold."""
    matches = re.findall(
        r"(?im)^\s*(?:\*\*|__)?\s*FINAL\s*:\s*"
        r"(?:[$£€]\s*)?([-+]?\d[\d,]*(?:\.\d+)?)\s*(?:\*\*|__)?\s*[.!]?\s*$",
        text,
    )
    return matches[-1].replace(",", "") if matches else None


def correct_answer(answer, reference):
    if answer is None:
        return False
    try:
        return Decimal(answer.replace(",", "")) == Decimal(reference.replace(",", ""))
    except InvalidOperation:
        return False


def trial_label(baseline, bypassed):
    if not baseline["complete"] or not baseline["correct"]:
        return -1, baseline_unscored_reason(baseline)
    if not bypassed["complete"] or bypassed["extracted_answer"] is None:
        return -1, "bypass_incomplete_or_missing_final_answer"
    if bypassed["token_mlp_evaluations_skipped"] == 0:
        return -1, "no_decode_intervention_executed"
    return (1, "answer_preserved") if bypassed["correct"] else (0, "answer_changed_to_wrong")


def baseline_unscored_reason(baseline):
    if not baseline["complete"]:
        return "baseline_incomplete"
    if baseline.get("extracted_answer") is None:
        return "baseline_final_answer_not_parsed"
    return "baseline_answer_incorrect"


def refresh_answer_result(result, reference_answer):
    """Recompute parsed answer/correctness from saved text after parser fixes."""
    result = dict(result)
    answer = parsed_answer(result.get("response", ""))
    result["extracted_answer"] = answer
    result["correct"] = bool(result.get("complete") and correct_answer(answer, reference_answer))
    return result


class Controller(nn.Module):
    def __init__(self, width, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Policy:
    def __init__(self, path):
        self.path = Path(path)
        self.data = torch.load(path, map_location="cpu", weights_only=True)
        if self.data["protocol"] != PROTOCOL:
            raise ValueError("Wrong controller protocol")
        self.model = Controller(self.data["width"], self.data["hidden"]).eval().to("cuda")
        self.model.load_state_dict(self.data["state_dict"])
        self.mean = self.data["mean"].to("cuda")
        self.std = self.data["std"].to("cuda")
        self.threshold = self.data["threshold"]
        self.eligible_layers = self.data["eligible_layers"]

    def score(self, features):
        if len(features) != self.data["num_layers"] or features.shape[1] != self.data["hidden_size"]:
            raise ValueError("Controller and Qwen feature shapes differ")
        one_hot = torch.eye(len(features), device=features.device)
        x = torch.cat([features.float(), one_hot], dim=1)
        return torch.sigmoid(self.model((x - self.mean) / self.std))


class Bypass:
    """Replace MLP forward itself: skipped GEMMs are never executed.

    Every layer's first call is full prompt prefill. Capture the last prompt
    token's post-attention-normalized MLP input. On later calls, exactly one
    MLP may return zero, so Qwen's surrounding residual adds zero. No attention
    layer, KV cache, or position update is replaced. Restore methods on exit.
    """
    def __init__(self, model, forced=None, policy=None, capture=False):
        self.layers = model.model.layers
        self.forced, self.policy, self.capture = forced, policy, capture
        self.selected = forced
        self.features, self.originals = {}, []
        self.calls = [0] * len(self.layers)
        self.skipped_token_mlp_evaluations = 0
        self.total_token_mlp_evaluations = 0
        self.decode_token_mlp_evaluations = 0
        self.scores = None

    def __enter__(self):
        for index, layer in enumerate(self.layers):
            module = layer.mlp
            original = module.forward
            self.originals.append((module, original, "forward" in module.__dict__))

            def forward(x, *args, _i=index, _original=original, **kwargs):
                prefill = self.calls[_i] == 0
                self.calls[_i] += 1
                n = x.shape[0] * x.shape[1]
                self.total_token_mlp_evaluations += n
                if prefill and (self.capture or self.policy is not None):
                    self.features[_i] = x[0, -1].detach().float().clone()
                if not prefill:
                    if x.shape[0] != 1 or x.shape[1] != 1:
                        raise RuntimeError("V3 requires batch-one cached, single-token decoding")
                    self.decode_token_mlp_evaluations += n
                    if _i == self.selected:
                        self.skipped_token_mlp_evaluations += n
                        return torch.zeros_like(x)
                result = _original(x, *args, **kwargs)
                if prefill and _i == len(self.layers) - 1 and self.policy is not None:
                    matrix = torch.stack([self.features[j] for j in range(len(self.layers))])
                    self.scores = self.policy.score(matrix)
                    eligible = list(getattr(self.policy, "eligible_layers", range(len(self.layers))))
                    value, chosen = self.scores[eligible].max(dim=0)
                    self.selected = eligible[int(chosen.item())] if value.item() >= self.policy.threshold else None
                return result

            module.forward = forward
        return self

    def __exit__(self, *exc):
        for module, original, had_override in self.originals:
            if had_override:
                module.forward = original
            else:
                del module.forward


def generate(tokenizer, model, question, max_new_tokens, seed, forced=None, policy=None, capture=False):
    seed_all(seed)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": (
            question["question"] + "\nSolve carefully. You may use a short calculation. "
            "End with a line exactly: FINAL: <number>."
        )}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode(), Bypass(model, forced, policy, capture) as bypass:
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    ids = outputs[0, inputs["input_ids"].shape[1]:]
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    complete = bool(len(ids) and ids[-1].item() in eos_ids)
    text = tokenizer.decode(ids, skip_special_tokens=True)
    answer = parsed_answer(text)
    result = {
        "question_id": question["question_id"], "response": text,
        "extracted_answer": answer, "complete": complete,
        "finish_reason": "eos" if complete else "length",
        "correct": complete and correct_answer(answer, question["reference_answer"]),
        "input_tokens": inputs["input_ids"].shape[1], "output_tokens": len(ids),
        "seconds": elapsed, "tokens_per_second": len(ids) / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "selected_layer": bypass.selected,
        "layer_probabilities": bypass.scores.cpu().tolist() if bypass.scores is not None else None,
        "token_mlp_evaluations_total": bypass.total_token_mlp_evaluations,
        "token_mlp_evaluations_decode": bypass.decode_token_mlp_evaluations,
        "token_mlp_evaluations_skipped": bypass.skipped_token_mlp_evaluations,
    }
    features = np.stack([bypass.features[j].cpu().numpy() for j in range(len(bypass.layers))]) if capture else None
    return result, features


def binary_metrics(y, predicted):
    y, predicted = np.asarray(y, dtype=int), np.asarray(predicted, dtype=int)
    tp = int(((y == 1) & (predicted == 1)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"rows": len(y), "accuracy": float((y == predicted).mean()) if len(y) else None,
            "precision_safe": precision, "recall_safe": recall,
            "f1_safe": 2 * precision * recall / max(precision + recall, 1e-12),
            "unsafe_false_skip_rate": fp / (fp + tn) if fp + tn else None,
            "predicted_safe": tp + fp, "confusion_true_rows_pred_columns": [[tn, fp], [fn, tp]]}


def generation_args(parser=None):
    parser = parser or argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser
