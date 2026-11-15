"""Shared, reproducible generation and quality rules for V6. No V5 dependency."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

PROTOCOL = "v6-cnndm-article-depth-24-28-32-v1"
DEPTHS = (24, 28, 32)
MODEL_ID = "facebook/layerskip-llama3-8B"
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DATASET_ID = "abisee/cnn_dailymail"
DATASET_CONFIG = "3.0.0"
STOP_STRING = "Article:"
MODES = ("base_full", "layerskip_full", "layerskip_controller")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_key(text):
    return fingerprint(" ".join(text.split()).casefold())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def source_hashes():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def environment():
    packages = {}
    for name in ("torch", "transformers", "datasets", "numpy", "scikit-learn", "amdsmi",
                 "joblib", "rouge-score", "huggingface-hub", "pyarrow", "scipy", "nltk"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import sys
    torch = sys.modules.get("torch")
    gpu = None
    if torch is not None and torch.cuda.is_available():
        gpu = {"name": torch.cuda.get_device_name(0), "hip": torch.version.hip,
               "cuda": torch.version.cuda,
               "memory_bytes": torch.cuda.get_device_properties(0).total_memory}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "gpu": gpu}


def manifest(directory, config, same_environment=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = {**config, "protocol": PROTOCOL, "source_sha256": source_hashes()}
    value = {"identity": fingerprint(config), "config": config, "environment": environment()}
    path = directory / "manifest.json"
    if path.exists():
        old = read_json(path)
        if old["identity"] != value["identity"] or old["identity"] != fingerprint(old["config"]):
            raise ValueError("Inputs/code changed. Use a new output directory.")
        if same_environment and old["environment"] != value["environment"]:
            raise ValueError("Generation environment changed. Use a new output directory.")
        return old
    atomic_json(path, value)
    return value


class Progress:
    def __init__(self, directory):
        self.path = Path(directory) / "run_status.json"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.state = {"pid": os.getpid(), "status": "running", "phase": "starting",
                      "started_at_utc": datetime.now(timezone.utc).isoformat()}

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
        self.update(status="completed" if kind is None else "failed",
                    phase="finished" if kind is None else "saved_results_retained",
                    error=None if error is None else str(error))


def load_prepared(directory):
    directory = Path(directory)
    meta = read_json(directory / "manifest.json")
    if (meta["identity"] != fingerprint(meta["config"])
            or meta["config"]["protocol"] != PROTOCOL
            or meta["config"]["source_sha256"] != source_hashes()):
        raise ValueError("Invalid or changed preparation source. Use its archived V6 code.")
    bundle = read_json(directory / "dataset.json")
    if fingerprint(bundle) != meta["config"]["dataset_sha256"]:
        raise ValueError("Prepared dataset checksum mismatch")
    return bundle, meta


@lru_cache(maxsize=1)
def rouge_scorer():
    from rouge_score.rouge_scorer import RougeScorer
    return RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def score_summary(summary, reference):
    return {name: float(value.fmeasure)
            for name, value in rouge_scorer().score(reference, summary).items()}


def quality_preserved(trial, full, config):
    """Observed reference quality, not a factuality/safety guarantee."""
    return bool(trial["complete"] and trial["response"].strip()
                and trial["rougeL"] >= config["min_rouge_l"]
                and trial["rougeL"] >= full["rougeL"] - config["max_rouge_l_drop"])


@contextlib.contextmanager
def prefix_depth(model, depth):
    import torch
    layers = model.model.layers
    if len(layers) != 32 or depth not in DEPTHS:
        raise ValueError("Expected a full 32-layer Llama model and depth 24/28/32")
    configs = {id(c): c for c in (model.config, model.model.config)}
    saved = [(c, c.num_hidden_layers) for c in configs.values()]
    try:
        model.model.layers = torch.nn.ModuleList(list(layers)[:depth])
        for config, _ in saved:
            config.num_hidden_layers = depth
        yield
    finally:
        model.model.layers = layers
        for config, count in saved:
            config.num_hidden_layers = count


def load_model(model_id, revision, tokenizer_revision, seed):
    import re
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("Install v6/requirements.txt: transformers==4.57.1")
    if not torch.cuda.is_available():
        raise RuntimeError("Generation needs a CUDA or ROCm GPU")
    if not all(re.fullmatch(r"[0-9a-f]{40}", rev) for rev in (revision, tokenizer_revision)):
        raise ValueError("Expected resolved 40-character model/tokenizer commit IDs")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=tokenizer_revision)
    own = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if own.get_vocab() != tokenizer.get_vocab() or own.eos_token_id != tokenizer.eos_token_id:
        raise ValueError("Model tokenizers differ; this paired token comparison is invalid")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=dtype, attn_implementation="sdpa"
    ).to("cuda").eval().requires_grad_(False)
    if model.config.model_type != "llama" or len(model.model.layers) != 32:
        raise ValueError("Expected a 32-layer Llama checkpoint")
    return tokenizer, model


def generate(tokenizer, model, row, depth, cap, energy_meter=None):
    """Synchronized input-transfer + generation latency; quality scoring excluded."""
    import torch
    from transformers import GenerationConfig, StopStringCriteria, StoppingCriteriaList
    prompt_ids = row["input_ids"]
    if len(prompt_ids) + cap > model.config.max_position_embeddings:
        raise ValueError("Prompt + output cap exceeds model context window")
    config = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
                              max_new_tokens=cap, eos_token_id=tokenizer.eos_token_id,
                              pad_token_id=tokenizer.eos_token_id)
    stop = StoppingCriteriaList([StopStringCriteria(tokenizer, [STOP_STRING])])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    energy_before = energy_meter.snapshot_joules() if energy_meter else None
    started = time.perf_counter()
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    with prefix_depth(model, depth), torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                generation_config=config, stopping_criteria=stop, logits_to_keep=1)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    energy_after = energy_meter.snapshot_joules() if energy_meter else None
    energy_joules = (energy_meter.delta_joules(energy_before, energy_after)
                     if energy_meter else None)
    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    ids = output[0, len(prompt_ids):].tolist()
    raw = tokenizer.decode(ids, skip_special_tokens=True)
    before, boundary, after = raw.partition(STOP_STRING)
    eos = bool(ids and ids[-1] == tokenizer.eos_token_id)
    complete = bool((eos or boundary) and not (boundary and after.strip()) and before.strip())
    response = before.strip()
    return {"article_id": row["article_id"], "depth": depth,
            "model_id": model.config._name_or_path, "dtype": str(model.dtype),
            "prompt_sha256": fingerprint(prompt_ids), "raw_response": raw, "response": response,
            "generated_token_ids": ids, "complete": complete,
            "finish_reason": "boundary_overrun" if boundary and after.strip() else
                             "next_article" if boundary else "eos" if eos else "length",
            **score_summary(response, row["reference_summary"]),
            "input_tokens": len(prompt_ids), "output_tokens": len(ids), "seconds": seconds,
            "gpu_energy_joules": energy_joules,
            "gpu_average_power_watts": energy_joules / seconds if energy_joules is not None and seconds else None,
            "peak_allocated_gib": allocated, "peak_reserved_gib": reserved,
            "decoder_layer_token_evaluations": depth * (len(prompt_ids) + max(len(ids) - 1, 0))}


def warmup(tokenizer, model, row, depths):
    warm = {**row, "input_ids": row["input_ids"][:128]}
    for depth in depths:
        generate(tokenizer, model, warm, depth, 8)


def checked_trial(path, row, identity, depth, mode=None):
    trial = read_json(path)
    if (trial["run_identity"] != identity or trial["article_id"] != row["article_id"]
            or trial["depth"] != depth or trial["prompt_sha256"] != fingerprint(row["input_ids"])
            or (mode is not None and trial.get("mode") != mode)):
        raise ValueError(f"Saved trial identity mismatch: {path}")
    return trial


def partitions(rows, seed):
    ids = list(range(len(rows)))
    random.Random(seed).shuffle(ids)
    a, b = int(len(ids) * .70), int(len(ids) * .85)
    if min(a, b - a, len(ids) - b) < 1:
        raise ValueError("Need at least 10 articles for nonempty train/validation/test splits")
    return {name: [rows[i] for i in subset] for name, subset in
            (("train", ids[:a]), ("validation", ids[a:b]), ("test", ids[b:]))}


def model_text(article, depth):
    return f"{article} [DEPTH_{depth}]"


def route_depth(pipeline, article, threshold):
    probabilities = pipeline.predict_proba([model_text(article, d) for d in DEPTHS])
    index = list(pipeline.classes_).index(1)
    scores = {d: float(probabilities[i][index]) for i, d in enumerate(DEPTHS)}
    depth = next((d for d in DEPTHS[:-1] if scores[d] >= threshold), 32)
    return depth, scores


def load_labels(directory):
    directory = Path(directory)
    meta = read_json(directory / "manifest.json")
    if (meta["identity"] != fingerprint(meta["config"]) or meta["config"]["protocol"] != PROTOCOL
            or meta["config"]["source_sha256"] != source_hashes()):
        raise ValueError("Invalid label manifest or changed source")
    bundle, prepared = load_prepared(directory / meta["config"]["prepared_dir"])
    if prepared["identity"] != meta["config"]["prepared_identity"]:
        raise ValueError("Prepared evidence differs from labeling")
    rows = read_json(directory / "selected_dataset.json")
    originals = {r["article_id"]: r for r in bundle["label_articles"]}
    if len(rows) != len(originals) or {r["article_id"] for r in rows} != set(originals):
        raise ValueError("Incomplete or duplicate labeled article collection")
    for row in rows:
        original = originals[row["article_id"]]
        if any(row[k] != v for k, v in original.items()):
            raise ValueError("Labeled input differs from prepared article")
        for depth in DEPTHS:
            path = directory / "articles" / row["article_id"] / f"depth_{depth}.json"
            trial = checked_trial(path, row, meta["identity"], depth)
            if digest(path) != row["trial_sha256"][str(depth)] or trial != row["outcomes"][str(depth)]:
                raise ValueError("Label trial evidence mismatch")
        for depth in DEPTHS:
            if row["quality_labels"][str(depth)] != quality_preserved(
                    row["outcomes"][str(depth)], row["outcomes"]["32"], bundle["config"]):
                raise ValueError("Quality labels differ from observed outcomes")
    return rows, meta, bundle


def load_controller(path):
    import joblib
    payload = joblib.load(path)  # Only load the locally produced, trusted checkpoint.
    if payload["protocol"] != PROTOCOL or payload["source_sha256"] != source_hashes():
        raise ValueError("Controller protocol/source changed. Use archived V6 source.")
    return payload
