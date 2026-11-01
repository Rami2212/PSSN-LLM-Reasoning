"""Critical intervention, supervision, and held-out routing checks (no downloads)."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from v4.common import (DEPTHS, Progress, atomic_json, correct_answer, digest,
                       identity, parsed_answer, prefix_depth, route, settings,
                       text_key, training_target)


def test_prefix_executes_only_selected_transformer_blocks_and_restores():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(1)
    config = Qwen3Config(vocab_size=40, hidden_size=16, intermediate_size=24,
                        num_hidden_layers=36, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=8,
                        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    original = model.model.layers
    original_head = model.lm_head
    original_norm = model.model.norm
    calls = Counter()
    handles = [layer.register_forward_hook(lambda m, a, o, i=i: calls.update([i]))
               for i, layer in enumerate(original)]
    inputs = torch.tensor([[1, 4, 7]])
    try:
        for depth in DEPTHS:
            calls.clear()
            with torch.inference_mode(), prefix_depth(model, depth):
                prefill = model(inputs, use_cache=True)
                # Make an actual cached decode call; no omitted tail should run.
                decoded = model(torch.tensor([[9]]), past_key_values=prefill.past_key_values, use_cache=True)
                assert decoded.logits.shape == (1, 1, 40)
                assert set(calls) == set(range(depth))
                assert all(count == 2 for count in calls.values())
                assert model.lm_head is original_head and model.model.norm is original_norm
            assert model.model.layers is original
            assert model.config.num_hidden_layers == 36
        with pytest.raises(RuntimeError, match="intentional"):
            with prefix_depth(model, 12):
                raise RuntimeError("intentional")
        assert model.model.layers is original and model.config.num_hidden_layers == 36
        with torch.inference_mode():
            before = model(inputs).logits
            with prefix_depth(model, 36):
                full = model(inputs).logits
        torch.testing.assert_close(before, full, rtol=0, atol=0)
        # Exercise the production .generate path as well as direct cached forward.
        from transformers import BatchEncoding
        from v4.common import generate

        class TinyTokenizer:
            eos_token_id = 2

            def apply_chat_template(self, *args, **kwargs):
                assert kwargs["enable_thinking"] is False
                return "test prompt"

            def __call__(self, *args, **kwargs):
                return BatchEncoding({"input_ids": inputs.clone(), "attention_mask": torch.ones_like(inputs)})

            def decode(self, *args, **kwargs):
                return "FINAL: 42"

        model.generation_config.eos_token_id = None  # Deterministically reach the cap.
        for depth in DEPTHS:
            calls.clear()
            result = generate(TinyTokenizer(), model,
                              {"question_id": "tiny", "question": "17+25", "reference_answer": "42"},
                              depth, settings(max_new_tokens=3))
            assert result["output_tokens"] == 3
            assert result["complete"] is False and result["correct"] is False
            assert result["decoder_layer_token_evaluations"] == depth * 5
            assert set(calls) == set(range(depth))
            assert all(count == 3 for count in calls.values())
            assert model.model.layers is original and model.config.num_hidden_layers == 36
    finally:
        for handle in handles:
            handle.remove()


def test_labels_parser_and_confidence_fallback():
    assert parsed_answer("**FINAL: 108**") == "108"
    assert parsed_answer("FINAL: **1,200**") == "1200"
    assert correct_answer("1200", "1,200")
    assert parsed_answer("The total is 108.") is None
    assert training_target({str(d): {"correct": d >= 24} for d in DEPTHS})[0] == 24
    assert training_target({str(d): {"correct": False} for d in DEPTHS})[0] is None

    class TextOnly:
        classes_ = np.array([12, 24, 36])

        def predict_proba(self, texts):
            assert texts == ["question one", "question two"]
            return np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]])

    depth, confidence, raw = route(TextOnly(), ["question one", "question two"], 0.8)
    assert depth.tolist() == [12, 36]
    assert raw.tolist() == [12, 12]
    assert confidence.tolist() == [0.9, 0.4]
    assert route(TextOnly(), ["question one", "question two"], 1.01)[0].tolist() == [36, 36]


def make_evidence(directory, n=90, only_full=False):
    records = []
    for i in range(n):
        target = 36 if only_full else DEPTHS[i % 3]
        keyword = {12: "add", 24: "multiply", 36: "fractions"}[target]
        question = {"question_id": f"q{i}", "question": f"{keyword} example number {i}",
                    "question_key": text_key(f"{keyword} example number {i}"),
                    "reference_answer": "42"}
        folder = directory / "questions" / question["question_id"]
        outcomes, hashes = {}, {}
        for d in DEPTHS:
            trial = {"correct": d >= target, "complete": True, "finish_reason": "eos", "extracted_answer": "42" if d >= target else "43"}
            atomic_json(folder / f"depth_{d}.json", trial)
            outcomes[str(d)] = trial
            hashes[str(d)] = digest(folder / f"depth_{d}.json")
        target, reason = training_target(outcomes)
        atomic_json(folder / "label.json", {"question_id": question["question_id"],
                    "question_key": question["question_key"], "target_depth": target,
                    "label_reason": reason, "outcomes": outcomes, "trial_sha256": hashes})
        records.append(question)
    import v4.common as common
    from v4.common import PROTOCOL
    atomic_json(directory / "questions.json", records)
    config = {"protocol": PROTOCOL, "generation": settings(), "dataset_revision": "test-fixture",
              "questions_sha256": identity(records),
              "code_sha256": {"common.py": digest(Path(common.__file__))}}
    atomic_json(directory / "manifest.json", {"identity": identity(config), "config": config})
    return records


def test_text_training_and_heldout_evaluation(tmp_path):
    from v4.train import train
    from v4.evaluate import evaluate
    from v4.common import load_controller, read_json

    labels, training, evaluation = tmp_path / "labels", tmp_path / "training", tmp_path / "evaluation"
    make_evidence(labels)
    report = train(labels, training)
    assert report["test_used_for_selection"] is False
    controller = load_controller(training / "controller.joblib")
    splits = controller["split_question_ids"]
    assert not set(splits["train"]) & set(splits["validation"])
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["validation"]) & set(splits["test"])
    # Vocabulary must not contain IDs/numbers occurring only outside train.
    vocab = controller["pipeline"].named_steps["text"].transformer_list[0][1].vocabulary_
    for qid in splits["test"] + splits["validation"]:
        assert qid[1:] not in vocab
    result = evaluate(labels, training / "controller.joblib", evaluation)
    assert result["controller"]["questions"] == len(splits["test"])
    assert result["test_used_for_selection"] is False
    assert read_json(evaluation / "run_status.json")["status"] == "completed"
    # Corrupting trial evidence must prevent training/evaluation with mixed data.
    path = labels / "questions" / "q0" / "depth_12.json"
    path.write_text(json.dumps({"correct": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        evaluate(labels, training / "controller.joblib", tmp_path / "bad")


def test_no_shallow_supervision_stops_training(tmp_path):
    from v4.train import train
    make_evidence(tmp_path / "labels", only_full=True)
    with pytest.raises(ValueError, match="target depths"):
        train(tmp_path / "labels", tmp_path / "training")


def test_interruption_status_retains_checkpoint(tmp_path):
    checkpoint = tmp_path / "saved.json"
    atomic_json(checkpoint, {"correct": True})
    with pytest.raises(KeyboardInterrupt):
        with Progress(tmp_path) as progress:
            progress.update(phase="generating", completed_questions=1)
            raise KeyboardInterrupt()
    assert json.loads((tmp_path / "run_status.json").read_text())["status"] == "interrupted"
    assert checkpoint.exists()


def test_fresh_benchmark_saves_pairs_and_resumes(tmp_path, monkeypatch):
    import sys
    from v4.train import train
    from v4 import benchmark
    from v4.common import read_json

    labels, training = tmp_path / "labels", tmp_path / "training"
    make_evidence(labels)
    train(labels, training)
    records = [{"question_id": f"new{i}", "question": f"new add question {i}",
                "question_key": text_key(f"new add question {i}"), "reference_answer": "42"} for i in range(2)]
    monkeypatch.setattr(benchmark, "select_questions", lambda *a, **kw: (records, "test-fixture"))
    monkeypatch.setattr(benchmark, "load_model", lambda *a: (None, None))
    monkeypatch.setattr(benchmark, "warmup", lambda *a: None)
    calls = []

    def fake_generate(tokenizer, model, question, depth, config):
        calls.append((question["question_id"], depth))
        return {"question_id": question["question_id"], "depth": depth, "correct": depth == 36,
                "complete": True, "input_tokens": 5, "output_tokens": 2, "seconds": 0.5,
                "peak_allocated_gib": 1.0, "peak_reserved_gib": 2.0,
                "decoder_layer_token_evaluations": depth * 6}

    monkeypatch.setattr(benchmark, "generate", fake_generate)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--controller", str(training / "controller.joblib"),
                                     "--output-dir", str(tmp_path / "benchmark"), "--rows", "2"])
    benchmark.main()
    assert len(calls) == 4
    report = read_json(tmp_path / "benchmark" / "metrics.json")
    assert report["completed_questions"] == 2
    assert report["baseline"]["correct"] == 2
    assert report["controller"]["controller_seconds"] > 0
    benchmark.main()
    assert len(calls) == 4  # No completed generation is repeated on resume.
    assert read_json(tmp_path / "benchmark" / "run_status.json")["status"] == "completed"
