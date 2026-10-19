"""Offline checks: bypass actually avoids MLP execution, labels and split integrity."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from v3.common import (Bypass, PROTOCOL, atomic_json, correct_answer, digest,
                       parsed_answer, questions, read_json, text_key, trial_label)
from v3 import train


class CountingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x * 2


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module() for _ in range(3)])
        for layer in self.layers:
            layer.mlp = CountingMLP()

    def forward(self, x):
        for layer in self.layers:
            x = x + layer.mlp(x)
        return x


class PipelineTest(unittest.TestCase):
    def test_question_selection_deduplicates_text_and_excludes_prior_questions(self):
        data = [{"question": q, "answer": "solution\n#### 42"}
                for q in ["same question", " same   question ", "prior", "new one", "new two"]]
        datasets = SimpleNamespace(load_dataset=lambda *a, **k: data)
        hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(dataset_info=lambda *a, **k: SimpleNamespace(sha="fixed")))
        with patch.dict("sys.modules", {"datasets": datasets, "huggingface_hub": hub}):
            selected, revision = questions("train", 3, seed=42, exclude_keys=[text_key("prior")])
        self.assertEqual(revision, "fixed")
        self.assertEqual(len({r["question_key"] for r in selected}), 3)
        self.assertNotIn(text_key("prior"), {r["question_key"] for r in selected})

    def test_real_tiny_qwen_cached_generation_bypasses_only_decode_mlp(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(5)
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=48,
                            num_hidden_layers=3, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=16, max_position_embeddings=64,
                            bos_token_id=1, eos_token_id=2, pad_token_id=0)
        model = Qwen3ForCausalLM(config).eval()
        ids = torch.tensor([[1, 7, 8, 9]])
        kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=4,
                      do_sample=False, use_cache=True, eos_token_id=None, pad_token_id=0)
        calls = [0, 0, 0]
        handles = []
        for index, layer in enumerate(model.model.layers):
            def record(module, inputs, output, i=index):
                calls[i] += 1
            handles.append(layer.mlp.gate_proj.register_forward_hook(record))
        try:
            with torch.inference_mode(), Bypass(model, forced=1, capture=True) as gate:
                output = model.generate(**kwargs)
            self.assertEqual(output.shape[1], 8)
            self.assertEqual(calls, [4, 1, 4])
            self.assertEqual(gate.skipped_token_mlp_evaluations, 3)
            self.assertEqual(gate.decode_token_mlp_evaluations, 9)
            self.assertEqual(len(gate.features), 3)
            # Full prefill is numerically identical, including all KV updates.
            with torch.inference_mode():
                plain = model(ids, use_cache=True)
                with Bypass(model, forced=1):
                    gated = model(ids, use_cache=True)
            torch.testing.assert_close(plain.logits, gated.logits, rtol=0, atol=0)
            self.assertEqual(plain.past_key_values.get_seq_length(), gated.past_key_values.get_seq_length())
        finally:
            for handle in handles:
                handle.remove()

    def test_noop_and_forced_skip_restore(self):
        decoder = ToyDecoder()
        model = SimpleNamespace(model=decoder)
        with Bypass(model, forced=1, capture=True) as gate:
            prefill = decoder(torch.ones(1, 4, 2))
            decode = decoder(torch.ones(1, 1, 2))
        self.assertTrue(torch.all(prefill == 27))
        self.assertTrue(torch.all(decode == 9))
        self.assertEqual([l.mlp.calls for l in decoder.layers], [2, 1, 2])
        self.assertEqual(gate.skipped_token_mlp_evaluations, 1)
        self.assertEqual(gate.total_token_mlp_evaluations, 15)
        self.assertEqual(gate.decode_token_mlp_evaluations, 3)
        self.assertEqual(gate.features[1].tolist(), [3, 3])
        self.assertTrue(torch.all(decoder(torch.ones(1, 1, 2)) == 27))
        self.assertNotIn("forward", decoder.layers[1].mlp.__dict__)

    def test_learned_decision_only_after_prefill(self):
        decoder = ToyDecoder()
        class FixedPolicy:
            threshold = 0.5

            def score(self, features):
                self.features = features
                return torch.tensor([0.1, 0.2, 0.9])

        policy = FixedPolicy()
        with Bypass(SimpleNamespace(model=decoder), policy=policy) as gate:
            self.assertTrue(torch.all(decoder(torch.ones(1, 2, 2)) == 27))
            self.assertEqual(gate.selected, 2)
            self.assertTrue(torch.all(decoder(torch.ones(1, 1, 2)) == 9))
        self.assertEqual(policy.features[:, 0].tolist(), [1, 3, 9])
        policy.threshold = 1.0
        with Bypass(SimpleNamespace(model=decoder), policy=policy) as gate:
            decoder(torch.ones(1, 2, 2))
            self.assertIsNone(gate.selected)
            self.assertTrue(torch.all(decoder(torch.ones(1, 1, 2)) == 27))

    def test_restores_after_exception(self):
        decoder = ToyDecoder()
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with Bypass(SimpleNamespace(model=decoder), forced=0):
                raise RuntimeError("interrupted")
        self.assertTrue(torch.all(decoder(torch.ones(1, 1, 2)) == 27))

    def test_labels_exclude_incomplete_or_no_intervention(self):
        full = {"complete": True, "correct": True}
        skip = {"complete": True, "correct": True, "extracted_answer": "42",
                "token_mlp_evaluations_skipped": 1}
        self.assertEqual(trial_label(full, skip)[0], 1)
        self.assertEqual(trial_label(full, dict(skip, correct=False))[0], 0)
        self.assertEqual(trial_label(full, dict(skip, complete=False))[0], -1)
        self.assertEqual(trial_label(dict(full, correct=False), skip)[0], -1)
        self.assertEqual(trial_label(full, dict(skip, token_mlp_evaluations_skipped=0))[0], -1)
        self.assertEqual(trial_label(full, dict(skip, extracted_answer=None))[0], -1)
        self.assertIsNone(parsed_answer("We first calculate 42."))
        self.assertTrue(correct_answer(parsed_answer("Work\nFINAL: 1,234.5"), "1234.5"))

    def test_train_uses_question_disjoint_splits_and_train_only_statistics(self):
        with tempfile.TemporaryDirectory(prefix=".pytest-tmp-v3-", dir=Path(__file__).resolve().parents[2]) as temp:
            root = Path(temp)
            labels, output = root / "labels", root / "training"
            records = [{"question_id": f"q{i}", "question_key": f"key{i}"} for i in range(10)]
            atomic_json(labels / "questions.json", records)
            atomic_json(labels / "summary.json", {"layers": 3, "hidden_size": 2})
            atomic_json(labels / "manifest.json", {"identity": "fixture", "config": {
                "protocol": PROTOCOL,
                "model_revision": "fixture_revision", "dataset_revision": "fixture_dataset",
                "max_new_tokens": 10}})
            for i, q in enumerate(records):
                folder = labels / "questions" / q["question_id"]
                folder.mkdir(parents=True)
                np.savez_compressed(folder / "features.npz", features=np.full((3, 2), i, dtype=np.float32))
                atomic_json(folder / "labels.json", {"labels": [0, 1, -1], "feature_sha256": digest(folder / "features.npz")})
            argv = ["train", "--labels-dir", str(labels), "--output-dir", str(output), "--epochs", "2"]
            with patch("sys.argv", argv), patch("v3.common.environment", return_value={"code_sha256": {}}):
                train.main()
            checkpoint = torch.load(output / "controller.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["protocol"], PROTOCOL)
            self.assertEqual(checkpoint["width"], 5)
            splits = read_json(output / "manifest.json")["config"]["split_question_ids"]
            self.assertFalse(set(splits["train"]) & set(splits["test"]))
            self.assertFalse(set(splits["validation"]) & set(splits["test"]))
            train_indices = [int(q[1:]) for q in splits["train"]]
            self.assertAlmostEqual(checkpoint["mean"][0].item(), np.mean(train_indices), places=5)
            self.assertEqual(read_json(output / "evaluation.json")["rows"], {"train": 12, "validation": 4, "test": 4})


if __name__ == "__main__":
    unittest.main()
