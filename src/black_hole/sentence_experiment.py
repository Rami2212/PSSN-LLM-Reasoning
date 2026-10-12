"""Shared three-sample sentence removal, attainability and core-search engine.

Adapted from the pinned authors procedure and the validated Q62 experiment.
No package installation or model loading occurs when this module is imported.
"""
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
TRACE_MODE = "embedded"  # "generate" makes a new trace with the authors' prompt.
TRACE_TOKEN_CAP = 1024  # Applies only to fresh full-trace generation.
N_RESAMPLES = 3
SEED = 42
TEMPERATURE = 0.6
TOP_P = 0.95
SUFFICIENCY_THRESHOLD = 2 / 3
ATTAINABILITY_THRESHOLD = 0.8
NECESSITY_THRESHOLD = 2 / 3
RUN_ATTRIBUTION_CORE = True
RUN_RANDOM_CORE = True
RANDOM_SEEDS = [42, 43, 44]
PRINT_EXACT_PROMPTS = False
AUTO_DOWNLOAD_RESULTS = False
THRESHOLDS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
              0.6, 0.7, 0.8, 0.9, 0.99]
UPSTREAM_COMMIT = "ceeb7201b55858476a21e9e63e145ba2916ab159"
UPSTREAM_URL = "https://github.com/technion-cs-nlp/reasoning-importance"
SOURCE_SELECTION = "PSSN/artifacts/expanded_baseline/w2-001-700-attempts-seed44/black_hole_trace_selection/20261010T184131921450Z/eligible_traces.jsonl"

from .sentence_checkpoint import SentenceCheckpoint, atomic_json, collect_trials, identity, sentence_units

QUESTION = REFERENCE_ANSWER = ORIGINAL_TRACE_ID = ORIGINAL_TRACE = ""

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def split_sentences(text):
    """Authors' punctuation/paragraph split and <10-character fragment rule.

    Whitespace and original order remain intact. This is their sentence unit,
    not the four-group PSSN protocol or the existing semantic-state IDs.
    """
    spans, start = [], 0
    for i, char in enumerate(text):
        paragraph_end = text.startswith("\n\n", i) or text.startswith("\r\n\r\n", i)
        sentence_end = (i + 1 < len(text) and char in ".?!"
                        and text[i + 1] in " \n")
        if paragraph_end or sentence_end:
            spans.append((text[start:i + 1], start, i + 1))
            start = i + 1
    if text[start:].strip():
        spans.append((text[start:], start, len(text)))
    merged = []
    for sentence, start, end in spans:
        if merged and len(sentence.strip()) < 10:
            previous, previous_start, _ = merged[-1]
            merged[-1] = (previous + sentence, previous_start, end)
        else:
            merged.append((sentence, start, end))
    return merged


def boxed_answers(text):
    """Only closed boxes count: truncated output is recorded as unparsed."""
    found = []
    for match in re.finditer(r"\\boxed\{", text):
        start, depth = match.end(), 1
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                found.append(text[start:i])
                break
    return found


def numeric_equal(value, reference=None):
    if reference is None:
        reference = REFERENCE_ANSWER
    try:
        normalized = re.sub(r"[\s,$]", "", value)
        return Decimal(normalized) == Decimal(reference)
    except (InvalidOperation, ValueError):
        return False


def states_answer(sentence):
    answer = r"(?<![\d.,+-])" + re.escape(REFERENCE_ANSWER) + r"(?!\d|[.,]\d)"
    patterns = [r"(Thus|Therefore|So|Hence|Consequently|This means).*"
                r"(answer|result)[^\n]+" + answer,
                r"(The|Our|My)\s+(final\s+)?(answer|result)[^\n]+" + answer,
                r"(Thus|Therefore|So|Hence|Consequently),? (answer|result)[^\n]+"]
    return (any(re.search(p, sentence, re.I) for p in patterns)
            or any(numeric_equal(a) for a in boxed_answers(sentence)))


def normalize_ppl(ppl, full_cot_ppl, empty_cot_ppl):
    if full_cot_ppl > empty_cot_ppl:
        print("Full CoT PPL is greater than Empty CoT PPL!")

    if empty_cot_ppl == full_cot_ppl:
        return 0.0
    return (ppl - full_cot_ppl) / (empty_cot_ppl - full_cot_ppl)


def masked_text(sentences, mask):
    if len(sentences) != len(mask):
        raise ValueError("Sentence/mask lengths differ")
    return "".join(sentence for sentence, keep in zip(sentences, mask) if keep)


def recursive_influence_mask(matrix, threshold, answer_idx):
    """Keep cumulative influence dependencies, recursively, as upstream prune_cot.

    Zero-total rows retain all tied zero scores, matching the upstream comparison
    behavior without introducing NaNs into saved JSON.
    """
    n = len(matrix)
    keep = [False] * n
    keep[answer_idx] = True
    for target in range(answer_idx, 0, -1):
        if not keep[target]:
            continue
        scores = matrix[target]
        descending = sorted(scores, reverse=True)
        total = sum(descending)
        cutoff = descending[-1]
        accumulated = 0.0
        if total > 0:
            for score in descending:
                accumulated += score
                if accumulated / total >= threshold:
                    cutoff = score
                    break
        keep = [old or score >= cutoff for old, score in zip(keep, scores)]
    return keep


def queue_prune(sentences, answer_idx, starting_mask, removal_order, evaluate):
    """Accept passing deletions; retry failures after subsequent accepted ones."""
    mask = list(starting_mask)
    protected = {answer_idx, answer_idx - 1}
    queue = deque(i for i in removal_order if mask[i] and i not in protected)
    failures, attempts = 0, []
    # A failed candidate is put back; stop only after a complete failed pass.
    while queue and failures < len(queue):
        idx = queue.popleft()
        candidate = list(mask)
        candidate[idx] = False
        result = evaluate(candidate, idx)
        attempts.append({"sentence_index": idx, "accepted": result["passes_both"],
                         "context_before_mask": list(mask),
                         "evaluation_key": result["evaluation_key"],
                         "evaluation_id": result["evaluation_id"]})
        if result["passes_both"]:
            mask, failures = candidate, 0
        else:
            queue.append(idx)
            failures += 1
    return mask, attempts


def run_question(candidate, shared, batch_dir):
    global QUESTION, REFERENCE_ANSWER, ORIGINAL_TRACE_ID, ORIGINAL_TRACE
    QUESTION = candidate["question"]
    REFERENCE_ANSWER = candidate["reference_answer"]
    ORIGINAL_TRACE_ID = candidate["trace_id"]
    ORIGINAL_TRACE = candidate["reasoning_trace"]
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU is accessible to PyTorch.")
    if TRACE_MODE not in {"embedded", "generate"}:
        raise ValueError("TRACE_MODE must be embedded or generate")
    dtype = torch.float16  # Explicit FP16; avoid emulated BF16 on T4.
    device = torch.device("cuda:0")
    out_dir = batch_dir / candidate["problem_id"]
    out_dir.mkdir(exist_ok=True)
    checkpoint = SentenceCheckpoint(out_dir, shared.get("persist"))
    evaluation_rows = []
    cache = {}
    manifest = {"created_at": utc_now(), "upstream_repo": UPSTREAM_URL,
                "upstream_commit": UPSTREAM_COMMIT, "question": QUESTION,
                "reference_answer": REFERENCE_ANSWER, "source_trace_id": ORIGINAL_TRACE_ID,
                "problem_id": candidate["problem_id"], "source_selection": SOURCE_SELECTION,
                "model_id": MODEL_ID, "requested_revision": MODEL_REVISION,
                "trace_mode": TRACE_MODE, "n_resamples": N_RESAMPLES,
                "seed": SEED, "temperature": TEMPERATURE, "top_p": TOP_P,
                "sufficiency_threshold": SUFFICIENCY_THRESHOLD,
                "attainability_threshold": ATTAINABILITY_THRESHOLD,
                "necessity_threshold": NECESSITY_THRESHOLD,
                "thresholds": THRESHOLDS, "random_seeds": RANDOM_SEEDS,
                "adaptations": ["Qwen3-4B and saved Week 2 GSM8K traces",
                                "FP16 matching the validated Q62 procedure",
                                "Historical trace used under authors' problem prompt in embedded mode",
                                "1024-token fresh-generation cap instead of authors' 16380",
                                "Independent sentence-deletion diagnostics added",
                                "Numeric answer comparison for the saved question",
                                "Three sampled answers; at least two correct for sufficiency",
                                "Authors' different empty/nonempty DeepSeek-style answer suffixes adapted to Qwen3",
                                "Authors' full-sequence PPL and unmodified normalize_ppl formula",
                                "Authors' earliest sufficient prefix, recursive masks and queue-based greedy search",
                                "Authors' incomplete answer scoring retained; incomplete samples logged separately",
                                "Embedding gradient norms use full-token absolute borders and float32 softmax",
                                "Screen each selected question before diagnostics and core extraction"],
                "environment": {"python": sys.version, "platform": platform.platform(),
                                "torch": torch.__version__, "transformers": transformers.__version__,
                                "cuda": torch.version.cuda, "hip": getattr(torch.version, "hip", None), "dtype": str(dtype),
                                "gpu": torch.cuda.get_device_name(0),
                                "gpu_vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30}}

    def save_json(filename, data):
        path = out_dir / filename
        atomic_json(path, data)
        if shared.get("persist"):
            shared["persist"](path)


    def status(state, phase):
        save_json("run_status.json", {"status": state, "phase": phase, "updated_at": utc_now(),
                                     "completed_evaluations": len(evaluation_rows)})
        if shared.get("on_phase"):
            shared["on_phase"](candidate["problem_id"], state, phase, len(evaluation_rows))

    def rng_snapshot():
        numpy_state = np.random.get_state()
        return {"python": random.getstate(),
                "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                "torch": torch.get_rng_state().tolist(),
                "cuda": [state.tolist() for state in torch.cuda.get_rng_state_all()]}

    def restore_rng(state):
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value
        random.setstate(tuples(state["python"]))
        n = state["numpy"]
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
        torch.cuda.set_rng_state_all([torch.tensor(v, dtype=torch.uint8) for v in state["cuda"]])

    def seed_all():
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def tokenize(prompt):
        return tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    def generate(prompt, cap, sample=True):
        inputs = tokenize(prompt)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        kwargs = dict(max_new_tokens=cap, do_sample=sample, pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=model.generation_config.eos_token_id, repetition_penalty=1.0)
        if sample:
            kwargs.update(temperature=TEMPERATURE, top_p=TOP_P)
        with torch.inference_mode():
            generated = model.generate(**inputs, **kwargs)
        torch.cuda.synchronize()
        ids = generated[0, inputs["input_ids"].shape[1]:]
        return {"completion": tokenizer.decode(ids, skip_special_tokens=False),
                "input_tokens": inputs["input_ids"].shape[1], "output_tokens": ids.numel(),
                "latency_seconds": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "hit_token_cap": ids.numel() >= cap}

    def perplexity(prompt):
        # Same full-sequence shifted-token NLL calculation as general_utils.py.
        # The code scores the question/template too, despite paper PPL(T | x) notation.
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[:, :-1, :].contiguous()
            labels = inputs["input_ids"][:, 1:].contiguous()
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1).mean()
        return torch.exp(nll).item()

    def score(sentences, mask, answer_idx, phase, calibration=None, ppl_mode="masked"):
        # The designated answer sentence is never supplied to answer generation.
        generation_mask = list(mask)
        if answer_idx is not None:
            generation_mask[answer_idx] = False
        context = thinking_prefix + masked_text(sentences, generation_mask)
        ending = answer_prefill if any(generation_mask) else empty_prefill
        prompt = context + ending
        key = hashlib.sha256(prompt.encode()).hexdigest()
        status("running", phase)
        ppl_prompt = context if ppl_mode == "generation_context" else thinking_prefix + masked_text(sentences, mask)
        request = {"phase": phase, "prompt_sha256": key, "mask": list(map(bool, mask)),
                   "answer_sentence_index": answer_idx, "ppl_mode": ppl_mode,
                   "ppl_prompt_sha256": identity(ppl_prompt),
                   "calibration": list(calibration) if calibration is not None else None}
        restored = checkpoint.load("evaluations", request)
        if restored is not None:
            restored["evaluation_id"] = len(evaluation_rows)
            evaluation_rows.append(restored)
            return restored
        if key not in cache:
            def sample(sample_index):
                status("running", f"{phase}_sample_{sample_index + 1}_of_{N_RESAMPLES}")
                trial = generate(prompt, answer_token_cap)
                answers = boxed_answers(ending + trial["completion"])
                trial.update(sample_index=sample_index, parsed_answer=answers[-1] if answers else None,
                             complete_answer=bool(answers),
                             correct=bool(answers and numeric_equal(answers[-1])), prompt_sha256=key,
                             phase=phase, max_new_tokens=answer_token_cap, prompt=prompt)
                return trial
            cache[key] = collect_trials(checkpoint, key, N_RESAMPLES, seed_all,
                                        sample, rng_snapshot, restore_rng)
        trials = cache[key]
        completed = sum(t["complete_answer"] for t in trials)
        suff = sum(t["correct"] for t in trials) / N_RESAMPLES
        ppl, attainability, calibration_valid = None, None, False
        # Prefix PPL uses the answer-withheld context returned by eval_sufficiency.
        # Threshold/greedy PPL uses the supplied mask, keeping its synthetic anchor.
        if ppl_mode not in {"masked", "generation_context"}:
            raise ValueError("Unknown PPL scoring stage")
        ppl_prompt = context if ppl_mode == "generation_context" else thinking_prefix + masked_text(sentences, mask)
        if calibration is not None:
            ppl = perplexity(ppl_prompt)
            full_ppl, empty_ppl = calibration
            calibration_valid = all(math.isfinite(v) for v in (ppl, full_ppl, empty_ppl))
            if calibration_valid:
                # Identical upstream normalize_ppl, including equal/negative bounds.
                # Do not clamp scores or reject reversed bounds behind the scenes.
                attainability = 1.0 - normalize_ppl(ppl, full_ppl, empty_ppl)
        row = {"evaluation_id": len(evaluation_rows), "evaluation_key": identity(request), "phase": phase,
               "mask": list(map(bool, mask)), "answer_sentence_index": answer_idx,
               "kept_sentence_count": sum(mask), "sufficiency": suff,
               "correct_count": sum(t["correct"] for t in trials), "complete_count": completed,
               "incomplete_count": N_RESAMPLES - completed,
               "completed_only_accuracy": sum(t["correct"] for t in trials) / completed if completed else None,
               "perplexity": ppl, "attainability": attainability,
               "calibration_valid": calibration_valid, "perplexity_mode": ppl_mode,
               "calibration_bounds": list(calibration) if calibration is not None else None,
               "calibration_warning": ("equal_bounds" if calibration and calibration[0] == calibration[1]
                                       else "reversed_bounds" if calibration and calibration[0] > calibration[1]
                                       else None),
               "passes_both": bool(calibration_valid and suff >= SUFFICIENCY_THRESHOLD
                                   and attainability >= ATTAINABILITY_THRESHOLD),
               "input_tokens": trials[0]["input_tokens"],
               "mean_output_tokens": sum(t["output_tokens"] for t in trials) / N_RESAMPLES,
               "mean_latency_seconds": sum(t["latency_seconds"] for t in trials) / N_RESAMPLES,
               "peak_allocated_gib": max(t["peak_allocated_gib"] for t in trials),
               "answers": [t["parsed_answer"] for t in trials],
               "prompt": prompt, "perplexity_prompt": ppl_prompt, "prompt_sha256": key}
        row["trial_checkpoint_ids"] = [identity({"prompt_sha256": key, "sample_index": i})
                                       for i in range(N_RESAMPLES)]
        checkpoint.save("evaluations", request, row)
        evaluation_rows.append(row)
        att = f"{attainability:.3f}" if attainability is not None else "undefined"
        print(f"{phase}: correct={row['correct_count']}/{N_RESAMPLES} | complete={completed}/{N_RESAMPLES} | "
              f"attainability={att} | tokens={row['input_tokens']}+{row['mean_output_tokens']:.1f}", flush=True)
        return row

    def attribution_matrix(sentences):
        # Same objective as the authors: sum target-token probabilities, then
        # norm its gradients with respect to every source sentence's embeddings.
        prompt = thinking_prefix + "".join(sentences)
        ids = tokenize(prompt)["input_ids"]
        encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]
        if len(offsets) != ids.shape[1]:
            raise ValueError("Token/offset alignment failed")
        borders, char_start = [], len(thinking_prefix)
        for sentence in sentences:
            char_end = char_start + len(sentence)
            positions = [i for i, (a, b) in enumerate(offsets)
                         if b > char_start and a < char_end]
            if not positions:
                raise ValueError("Empty sentence token span")
            borders.append((min(positions), max(positions) + 1))
            char_start = char_end
        n = len(sentences)
        matrix = [[0.0] * n for _ in range(n)]
        # Frozen model parameters avoid allocating full 4B-weight gradients.
        with torch.enable_grad():
            was_training = model.training
            if getattr(model.config, "attention_dropout", 0.0) != 0.0:
                raise RuntimeError("Attribution checkpointing requires zero dropout for this model")
            model.train()  # HF applies gradient checkpointing in training mode.
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            try:
                for target, (start, end) in enumerate(borders):
                    status("running", f"gradient_attribution_{target + 1}_of_{n}")
                    row_request = {"prompt_sha256": identity(prompt), "target_index": target}
                    saved_row = checkpoint.load("gradients", row_request)
                    if saved_row is not None:
                        matrix[target] = saved_row
                        continue
                    embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
                    logits = model(inputs_embeds=embeds, use_cache=False).logits[0]
                    positions = torch.arange(start, end, device=device)
                    probabilities = logits[positions - 1].float().softmax(-1)
                    objective = probabilities.gather(-1, ids[0, positions, None]).sum()
                    grads = torch.autograd.grad(objective, embeds)[0]
                    for source, (left, right) in enumerate(borders):
                        if source != target:
                            matrix[target][source] = float(grads[0, left:right].float().norm())
                    if not all(math.isfinite(v) for v in matrix[target]):
                        raise RuntimeError("Nonfinite attribution; no substitute ranking will be used")
                    checkpoint.save("gradients", row_request, matrix[target])
                    print(f"Attribution row {target + 1}/{n}", flush=True)
                    del embeds, logits, probabilities, objective, grads
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                model.gradient_checkpointing_disable()
                model.train(was_training)
        means = [sum(row[j] for row in matrix) / max(1, sum(row[j] != 0 for row in matrix))
                 for j in range(n)]
        return matrix, sorted(range(n), key=lambda j: means[j])

    save_json("manifest.json", manifest)
    status("running", "preparing_question")
    print(f"\nCHECKING {candidate['problem_id']}: {QUESTION}", flush=True)
    try:
        tokenizer, model = shared["tokenizer"], shared["model"]
        manifest["resolved_model_revision"] = getattr(model.config, "_commit_hash", None) or MODEL_REVISION
        manifest["resolved_tokenizer_revision"] = tokenizer.init_kwargs.get("_commit_hash") or MODEL_REVISION
        problem_prompt = f"Solve the following problem. You MUST put your final answer in \\boxed{{}}. Problem: {QUESTION}\n"
        messages = [{"role": "user", "content": problem_prompt}]
        generation_prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                         add_generation_prompt=True, enable_thinking=True)
        thinking_prefix = generation_prompt if generation_prompt.rstrip().endswith("<think>") else generation_prompt + "<think>\n"
        answer_prefill = " Therefore, the final answer is \\boxed{"
        empty_prefill = "The answer is \\boxed{"  # Authors use this for an empty CoT.
        # Exact upstream answer budget: two times the GT token length, including
        # tokenizer default special tokens. This is NOT the 1024 trace cap.
        answer_token_cap = 2 * len(tokenizer(REFERENCE_ANSWER)["input_ids"])
        manifest.update(problem_prompt=problem_prompt, thinking_prefix=thinking_prefix,
                        answer_prefill=answer_prefill, empty_prefill=empty_prefill,
                        answer_token_cap=answer_token_cap)
        trace = ORIGINAL_TRACE
        if TRACE_MODE == "generate":
            seed_all()
            fresh = generate(generation_prompt, TRACE_TOKEN_CAP, sample=False)
            save_json("fresh_generation.json", fresh)
            trace = fresh["completion"]
            if "</think>" not in trace or fresh["hit_token_cap"]:
                raise RuntimeError("Fresh trace is incomplete; excluded before sentence tests.")
            final_boxes = boxed_answers(trace.split("</think>", 1)[1])
            if not final_boxes or not numeric_equal(final_boxes[-1]):
                raise RuntimeError("Fresh baseline answer is incorrect/unparsed; excluded.")
        cot = trace.split("</think>", 1)[0].split("<think>", 1)[-1]
        spans = split_sentences(cot)
        sentences = [s for s, _, _ in spans]
        anchor = None
        for i in reversed(range(len(sentences))):
            if any(numeric_equal(a) for a in boxed_answers(sentences[i])):
                anchor = i
                break
            if re.search(r"(?<![\d.,+-])" + re.escape(REFERENCE_ANSWER) + r"(?!\d|[.,]\d)", sentences[i]):
                anchor = i
                break
        if anchor is None:
            report = {"problem_id": candidate["problem_id"], "question": QUESTION,
                      "eligible_under_adapted_rules": False, "exclusion_reasons": ["no_answer_anchor"],
                      "core_experiment_status": "excluded_by_screening"}
            save_json("summary.json", report)
            status("completed", "screened_out")
            return report
        manifest["trace_sha256"] = hashlib.sha256(trace.encode()).hexdigest()
        save_json("manifest.json", manifest)
        save_json("embedded_question_and_trace.json", {"question": QUESTION, "trace": trace,
                  "reference_answer": REFERENCE_ANSWER, "answer_sentence_index": anchor,
                  "trace_id": candidate["trace_id"],
                  "original_generation_metadata": candidate.get("original_generation_metadata", {}),
                  "source_states": candidate.get("source_states", []),
                  "sentences": sentence_units(candidate, spans, cot)})
        print("\nSentence units (no four-group conversion):")
        for i, sentence in enumerate(sentences):
            print(f"T{i+1}: {sentence.strip()}")
        if PRINT_EXACT_PROMPTS:
            print("\nAUTHORS' PROBLEM PROMPT:\n" + problem_prompt)
            print("\nFULL-TRACE ANSWER TEST PROMPT:\n" + thinking_prefix
                  + masked_text(sentences, [i != anchor for i in range(len(sentences))]) + answer_prefill)
        direct = score([], [], None, "no_trace_control")
        necessity = 1 - direct["sufficiency"]
        if necessity < NECESSITY_THRESHOLD:
            screen_reason = "cot_necessity_low"
            report = {"problem_id": candidate["problem_id"], "question": QUESTION,
                      "reference_answer": REFERENCE_ANSWER, "necessity": necessity,
                      "no_trace_control": direct, "eligible_under_adapted_rules": False,
                      "exclusion_reasons": [screen_reason],
                      "core_experiment_status": "excluded_by_screening",
                      "interpretation": "Question-only answers were sufficient; continue to next candidate."}
            save_json("summary.json", report)
            status("completed", "screened_out")
            print(f"SKIPPED: no-trace correct={direct['correct_count']}/{N_RESAMPLES}; "
                  f"necessity={necessity:.2f}; reason={screen_reason}.", flush=True)
            return report
        original_full_ppl = perplexity(thinking_prefix + "".join(sentences))
        empty_ppl = perplexity(thinking_prefix + answer_prefill + REFERENCE_ANSWER + "}")
        original_calibration = (original_full_ppl, empty_ppl)
        full = score(sentences, [True] * len(sentences), anchor, "original_full_trace", original_calibration)
        if full["sufficiency"] < SUFFICIENCY_THRESHOLD:
            report = {"problem_id": candidate["problem_id"], "question": QUESTION,
                      "reference_answer": REFERENCE_ANSWER, "necessity": necessity,
                      "no_trace_control": direct, "full_trace": full,
                      "eligible_under_adapted_rules": False,
                      "exclusion_reasons": ["original_trace_insufficient"],
                      "core_experiment_status": "excluded_by_screening"}
            save_json("summary.json", report)
            status("completed", "screened_out")
            print("SKIPPED: full trace was not sufficient.", flush=True)
            return report
        omissions = []
        # Authors' shortcut filtering: direct answer statement OR sufficient alone.
        shortcuts, shortcut_audit = [], []
        for i, sentence in enumerate(sentences):
            if i == anchor:
                continue
            heuristic = states_answer(sentence)
            trial = None if heuristic else score(sentences, [j == i for j in range(len(sentences))],
                                                anchor, f"shortcut_probe_T{i+1}")
            is_shortcut = heuristic or trial["sufficiency"] >= SUFFICIENCY_THRESHOLD
            if is_shortcut:
                shortcuts.append(i)
            shortcut_audit.append({"sentence_id": f"T{i+1}", "heuristic": heuristic,
                                   "singleton_sufficiency": trial["sufficiency"] if trial else None,
                                   "shortcut": is_shortcut})
        clean = [s for i, s in enumerate(sentences) if i not in shortcuts]
        clean_ids = [f"T{i+1}" for i in range(len(sentences)) if i not in shortcuts]
        clean_anchor = clean_ids.index(f"T{anchor+1}")
        clean_ppl = perplexity(thinking_prefix + "".join(clean))
        clean_calibration = (clean_ppl, empty_ppl)
        cleaned = score(clean, [True] * len(clean), clean_anchor, "shortcut_filtered_trace", clean_calibration)
        reasons = []
        if necessity < NECESSITY_THRESHOLD:
            reasons.append("cot_necessity_low")
        if cleaned["sufficiency"] < SUFFICIENCY_THRESHOLD:
            reasons.append("shortcut_filtered_trace_insufficient")
        if not cleaned["calibration_valid"]:
            reasons.append("attainability_calibration_invalid")
        eligible = not reasons
        report = {"problem_id": candidate["problem_id"], "question": QUESTION, "reference_answer": REFERENCE_ANSWER,
                  "necessity": necessity, "eligible_under_adapted_rules": eligible,
                  "exclusion_reasons": reasons, "shortcut_checks": shortcut_audit,
                  "original_sentence_count": len(sentences), "clean_sentence_ids": clean_ids,
                  "single_sentence_deletions": omissions, "full_trace": full,
                  "no_trace_control": direct, "cleaned_trace": cleaned,
                  "core_experiment_status": "not_started"}
        save_json("summary.json", report)
        print(f"\nSCREENING ELIGIBILITY: {eligible}; necessity={necessity:.2f}; reasons={reasons}")
        if not eligible:
            report["core_experiment_status"] = "excluded_by_screening"
            save_json("summary.json", report)
            status("completed", "screened_out")
            print("SKIPPED: shortcut filtering or attainability calibration failed.", flush=True)
            return report
        print("\nELIGIBLE QUESTION FOUND. Running the complete sentence-skipping experiment.", flush=True)
        for i in range(len(sentences)):
            if i == anchor:
                continue
            mask = [j != i for j in range(len(sentences))]
            row = score(sentences, mask, anchor, f"independent_remove_T{i+1}", original_calibration)
            row.update(removed_sentence_id=f"T{i+1}", removed_sentence_text=sentences[i])
            omissions.append(row)
        report["single_sentence_deletions"] = omissions
        save_json("summary.json", report)
        if RUN_ATTRIBUTION_CORE:
            prefix_len, prefix_audit = None, []
            for count in range(1, len(clean) + 1):
                result = score(clean, [i < count for i in range(len(clean))], clean_anchor,
                               f"prefix_{count}", clean_calibration, ppl_mode="generation_context")
                prefix_audit.append(result)
                if result["sufficiency"] >= SUFFICIENCY_THRESHOLD:
                    prefix_len = count
                    break
            if prefix_len is None:
                prefix_len = len(clean)  # Authors fall back to the entire cleaned trace.
            # As upstream, append a separate answer anchor for gradient scoring;
            # this anchor is ALWAYS excluded from the generation context.
            # Exclude the historical answer anchor too, if prefix includes it.
            # Only the synthetic anchor enters likelihood/gradient scoring.
            prefix = [s for i, s in enumerate(clean[:prefix_len]) if i != clean_anchor]
            prefix_ids = [sid for i, sid in enumerate(clean_ids[:prefix_len]) if i != clean_anchor]
            prefix = prefix + [answer_prefill + REFERENCE_ANSWER + "}"]
            answer_idx = len(prefix) - 1
            matrix, order = attribution_matrix(prefix)
            save_json("attribution_matrix.json", {"matrix": matrix, "removal_order": order,
                      "sentences": prefix, "answer_anchor_for_likelihood_only": answer_idx})
            starting_mask = [True] * len(prefix)
            threshold_audit = []
            for threshold in THRESHOLDS:
                mask = recursive_influence_mask(matrix, threshold, answer_idx)
                result = score(prefix, mask, answer_idx, f"attribution_threshold_{threshold}", clean_calibration)
                threshold_audit.append({"threshold": threshold, "evaluation": result})
                if result["passes_both"] or all(mask):
                    starting_mask = mask if result["passes_both"] else [True] * len(prefix)
                    break
            def evaluate_gradient(mask, idx):
                return score(prefix, mask, answer_idx, f"gradient_greedy_remove_{idx}", clean_calibration)
            final_mask, attempts = queue_prune(prefix, answer_idx, starting_mask, order, evaluate_gradient)
            final_check = score(prefix, final_mask, answer_idx, "final_attribution_core", clean_calibration)
            report["attribution_core"] = {"prefix_length": prefix_len, "prefix_checks": prefix_audit,
                "source_sentence_ids": prefix_ids, "final_check": final_check,
                "threshold_checks": threshold_audit, "starting_mask": starting_mask,
                "final_mask": final_mask, "attempts": attempts,
                "retained_text_excluding_answer": masked_text(prefix, [v and i != answer_idx for i, v in enumerate(final_mask)])}
            if RUN_RANDOM_CORE:
                report["random_cores"] = []
                for random_seed in RANDOM_SEEDS:
                    order = list(range(len(prefix)))
                    random.Random(random_seed).shuffle(order)
                    def evaluate_random(mask, idx):
                        return score(prefix, mask, answer_idx, f"random_{random_seed}_remove_{idx}", clean_calibration)
                    random_mask, random_attempts = queue_prune(prefix, answer_idx, [True] * len(prefix), order, evaluate_random)
                    random_final = score(prefix, random_mask, answer_idx,
                                         f"random_{random_seed}_final", clean_calibration)
                    report["random_cores"].append({"seed": random_seed, "final_mask": random_mask,
                                                   "attempts": random_attempts, "final_check": random_final})
                    save_json("summary.json", report)
            report["core_validation_passed"] = final_check["passes_both"]
            report["core_experiment_status"] = "completed"
        else:
            report["core_experiment_status"] = "excluded_by_screening" if not eligible else "disabled_in_config"
        report["interpretation"] = (
            "Independent deletions are additional diagnostics, not the authors' final core labels. "
            "Excluded questions must not supply paper-style training labels.  "
            "Tests measure trace-context sufficiency and attainability, not internal computation erasure. "
            "Answer generation uses the authors' tiny token budget, not a full new trace. "
            "Input tokens and attribution/pruning costs are reported separately; no end-to-end speedup is claimed.")
        save_json("summary.json", report)
        with (out_dir / "single_sentence_results.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["removed_sentence_id", "sufficiency", "complete_count", "incomplete_count",
                      "attainability", "passes_both", "input_tokens", "mean_output_tokens", "mean_latency_seconds"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(omissions)
        print("\nRESULTS: each independent test starts from the original trace.")
        print(f"{'Removed':<9} {'Correct':<9} {'Complete':<10} {'Attainability':<15} {'Both pass':<10}")
        for row in omissions:
            attain = f"{row['attainability']:.3f}" if row["attainability"] is not None else "undefined"
            print(f"{row['removed_sentence_id']:<9} {row['correct_count']}/{N_RESAMPLES}       "
                  f"{row['complete_count']}/{N_RESAMPLES}        {attain:<15} {str(row['passes_both']):<10}")
        if "attribution_core" in report:
            core = report["attribution_core"]
            print("\nFINAL GRADIENT CORE:")
            print(core["retained_text_excluding_answer"])
            print(f"Final core correct={core['final_check']['correct_count']}/{N_RESAMPLES}; "
                  f"passes both={core['final_check']['passes_both']}")
            print("Removal mask:", core["final_mask"])
        print("\n" + report["interpretation"])
        status("completed", "finished")
        return report
    except BaseException as error:
        status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", str(error))
        print(f"Saved completed trials and evaluations in {out_dir}", flush=True)
        raise
    finally:
        print(f"\nLOCAL RESULTS: {out_dir}")
        # Export complete unique trials even on a graceful interruption.
        for filename, kind in (("trials.jsonl", "trials"), ("evaluations.jsonl", "evaluations")):
            records = []
            for path in sorted((out_dir / "checkpoints" / kind).glob("*.json")):
                value = json.loads(path.read_text(encoding="utf-8"))["result"]
                records.append(value["trial"] if kind == "trials" else value)
            target = out_dir / filename
            temporary = target.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                import os
                os.fsync(stream.fileno())
            temporary.replace(target)
            if shared.get("persist"):
                shared["persist"](target)
