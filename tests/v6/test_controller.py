"""CPU checks with synthetic evidence; no remote datasets, model weights or GPU."""
import contextlib
import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from v6.controller_common import (DEPTHS, atomic_json, fingerprint, load_labels,
    manifest, partitions, quality_preserved, read_json, score_summary, text_key)
from v6.controller_label import label
from v6.controller_train import train
from v6.controller_evaluate import evaluate
from v6.controller_test import summarize, test as live_test
from v6.prepare import build_inputs, sample_records


def article(index, split="train"):
    text = f"category{index % 3} reported a significant event number {index} in {split} today"
    return {"article_id": f"synthetic-{split}-{index}", "article_key": text_key(text),
            "article": text, "visible_article": text, "reference_summary": "Residents approved the new park.",
            "input_ids": [1, index + 2, 7], "article_truncated": False}


def fake_generate(tokenizer, model, row, depth, cap):
    index = int(row["article_id"].rsplit("-", 1)[1])
    success = depth == 32 or (depth == 28 and index % 3 < 2) or (depth == 24 and index % 3 == 0)
    response = row["reference_summary"] if success else "Bananas orbit distant planets."
    return {"article_id": row["article_id"], "depth": depth, "prompt_sha256": fingerprint(row["input_ids"]),
            "response": response, "raw_response": response, "generated_token_ids": [depth, 9],
            "complete": True, "finish_reason": "eos", **score_summary(response, row["reference_summary"]),
            "input_tokens": len(row["input_ids"]), "output_tokens": 2, "seconds": depth / 32,
            "decoder_layer_token_evaluations": depth * (len(row["input_ids"]) + 1),
            "peak_allocated_gib": 1., "peak_reserved_gib": 2.}


def fake_torch():
    return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))


class QualityAndDatasetTests(unittest.TestCase):
    def test_quality_rule_requires_floor_completion_and_relative_quality(self):
        config = {"min_rouge_l": .15, "max_rouge_l_drop": .02}
        full = {"rougeL": .40}
        trial = {"complete": True, "response": "summary", "rougeL": .39}
        self.assertTrue(quality_preserved(trial, full, config))
        self.assertFalse(quality_preserved({**trial, "rougeL": .30}, full, config))
        self.assertFalse(quality_preserved({**trial, "complete": False}, full, config))
        self.assertFalse(quality_preserved({**trial, "response": " "}, full, config))
        self.assertFalse(quality_preserved({**trial, "rougeL": .1}, {"rougeL": .1}, config))

    def test_reference_rouge_is_real_and_empty_output_is_zero(self):
        same = score_summary("Residents approved the new park.", "Residents approved the new park.")
        self.assertEqual(same, {"rouge1": 1., "rouge2": 1., "rougeL": 1.})
        self.assertEqual(score_summary("", "Residents approved the new park.")["rougeL"], 0.)

    def test_sample_excludes_duplicate_text_and_keeps_target_reference_out_of_prompt(self):
        data = [{"article": "A report", "highlights": "private target reference", "id": "a"},
                {"article": " a  REPORT ", "highlights": "duplicate", "id": "b"},
                {"article": "B report", "highlights": "another reference", "id": "c"}]
        chosen = sample_records(data, 2, 42, "train", set())
        self.assertEqual(len({r["article_key"] for r in chosen}), 2)
        with self.assertRaises(ValueError):
            sample_records(data, 3, 42, "train", set())
        tokenizer = types.SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()),
                                         decode=lambda ids, **kwargs: bytes(ids).decode())
        config = {"article_tokens": 100, "demo_article_tokens": 100, "demo_summary_tokens": 100,
                  "max_new_tokens": 192}
        row = {"article": "Current article", "reference_summary": "secret target summary"}
        demo = {"article": "Example article", "reference_summary": "Example summary"}
        prepared = build_inputs(row, [demo], tokenizer, config)
        self.assertIn("Example summary", prepared["prompt"])
        self.assertNotIn("secret target summary", prepared["prompt"])

    def test_article_split_keeps_three_trials_together(self):
        parts = partitions([article(i) for i in range(100)], 42)
        self.assertEqual({k: len(v) for k, v in parts.items()}, {"train": 70, "validation": 15, "test": 15})
        all_ids = [r["article_id"] for rows in parts.values() for r in rows]
        self.assertEqual(len(set(all_ids)), 100)


class WorkflowTests(unittest.TestCase):
    def test_label_train_evaluate_and_three_system_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [article(i) for i in range(100)]
            held_ids = {r["article_id"] for r in partitions(rows, 42)["test"]}
            for row in rows:
                if row["article_id"] in held_ids:
                    row["visible_article"] += " heldoutonlykeyword"
            config = {"min_rouge_l": .15, "max_rouge_l_drop": .02,
                      "model_revision": "a" * 40, "base_model_revision": "b" * 40,
                      "max_new_tokens": 192, "seed": 42}
            bundle = {"config": config, "label_articles": rows,
                      "test_articles": [article(i, "test") for i in range(6)], "demonstrations": []}
            manifest(root / "prepared", {"dataset_sha256": fingerprint(bundle)})
            atomic_json(root / "prepared/dataset.json", bundle)
            with patch.dict("sys.modules", {"torch": fake_torch()}), contextlib.redirect_stdout(io.StringIO()):
                with patch("v6.controller_label.load_model", return_value=(object(), object())), \
                     patch("v6.controller_label.warmup"), patch("v6.controller_label.generate", side_effect=fake_generate) as gen:
                    labels = label(root / "prepared", root / "labels")
                    self.assertEqual(gen.call_count, 300)
                    self.assertEqual(labels["generation_trials"], 300)
                    label(root / "prepared", root / "labels")
                    self.assertEqual(gen.call_count, 300, "Resuming must reuse all 300 saved trials")
                result = train(root / "labels", root / "training")
                self.assertEqual(result["actual_training_trials"], 210)
                self.assertFalse(result["test_used_for_selection"])
                import joblib
                checkpoint = root / "training/controller.joblib"
                payload = joblib.load(checkpoint)
                vocab = payload["pipeline"].named_steps["text"].transformer_list[0][1].vocabulary_
                self.assertNotIn("heldoutonlykeyword", vocab)
                evaluated = evaluate(root / "labels", checkpoint, root / "evaluation")
                self.assertEqual(evaluated["depth_label_classifier"]["trials"], 45)
                with patch("v6.controller_test.load_model", return_value=(object(), object())), \
                     patch("v6.controller_test.warmup"), patch("v6.controller_test.generate", side_effect=fake_generate) as gen:
                    report = live_test(root / "prepared", checkpoint, root / "comparison", repeats=2)
                    self.assertEqual(gen.call_count, 36)
                    self.assertEqual(report["completed_trials"], 36)
                    self.assertEqual(report["status"], "completed")
                    self.assertEqual(report["metrics"]["base_full"]["unique_articles"], 6)
                    self.assertEqual(report["metrics"]["base_full"]["timing_trials"], 12)
                    self.assertEqual(report["repeat_token_mismatches"], 0)
                    self.assertTrue((root / "comparison/comparison.csv").exists())
                    live_test(root / "prepared", checkpoint, root / "comparison", repeats=2)
                    self.assertEqual(gen.call_count, 36)
            trial_path = next((root / "labels/articles").glob("*/depth_24.json"))
            bad = read_json(trial_path)
            bad["rougeL"] = .12345
            atomic_json(trial_path, bad)
            with self.assertRaisesRegex(ValueError, "evidence mismatch"):
                load_labels(root / "labels")

    def test_efficiency_uses_paired_times_including_routing_and_not_quality_repeat_counts(self):
        row = article(0)
        trials = []
        for repeat in range(2):
            for mode, depth, seconds, routing in (("base_full", 32, 2., 0.),
                                                 ("layerskip_full", 32, 2., 0.),
                                                 ("layerskip_controller", 24, .8, .2)):
                trial = fake_generate(None, None, row, depth, 192)
                trials.append({**trial, "mode": mode, "repeat": repeat,
                               "seconds": seconds, "routing_seconds": routing})
        result = summarize(trials, 1, 2, {"min_rouge_l": .15, "max_rouge_l_drop": .02})
        paired = result["paired_comparisons"]["layerskip_controller_vs_layerskip_full"]
        self.assertAlmostEqual(paired["time_reduction_percent"], 50.)
        self.assertAlmostEqual(paired["speedup_baseline_divided_by_candidate"], 2.)
        self.assertAlmostEqual(paired["output_token_reduction_percent"], 0.)
        self.assertAlmostEqual(paired["estimated_layer_work_reduction_percent"], 25.)
        self.assertEqual(paired["paired_quality_articles"], 1)
        self.assertEqual(paired["paired_timing_trials"], 2)


if __name__ == "__main__":
    unittest.main()
