"""CPU checks for scoring, independent questions, quotas, and the full controller fit/evaluation path."""
import tempfile
import unittest
import contextlib
import io
import types
from pathlib import Path
from unittest.mock import patch

from v5.controller_common import (DATASET_REVISION, DEPTHS, MODEL_REVISION,
    atomic_json, digest, load_controller, load_labels, manifest, parse_answer,
    read_json, score_response, stratified_partitions, target_depth, text_key)


class ControllerEvidenceTests(unittest.TestCase):
    def test_inline_final_is_scored_and_following_questions_are_excluded(self):
        self.assertEqual(parse_answer("The cost is 108. FINAL: 108"), "108")
        self.assertEqual(parse_answer("**FINAL: $1,080**"), "1080")
        self.assertIsNone(parse_answer("The cost might be 108"))
        self.assertIsNone(parse_answer("FINAL: 1/2"))
        self.assertEqual(parse_answer("FINAL: 108.5"), "108.5")
        self.assertEqual(parse_answer("FINAL: 108\nQuestion: Other? FINAL: 5"), "108")
        scored = score_response("The cost is 108. FINAL: 108\n\nQuestion:", False, "108")
        self.assertTrue(scored["correct"])
        self.assertEqual(scored["finish_reason"], "next_question")

    def test_boundary_overrun_never_counts_as_complete(self):
        for eos in (False, True):
            row = score_response("FINAL: 108\nQuestion: Other question FINAL: 5", eos, "108")
            self.assertEqual(row["finish_reason"], "boundary_overrun")
            self.assertFalse(row["correct"])
            self.assertFalse(row["complete"])
        self.assertFalse(score_response("FINAL: 108", False, "108")["complete"])
        self.assertTrue(score_response("FINAL: 108", True, "108")["correct"])

    def test_target_needs_complete_correct_baseline(self):
        outcomes = {str(d): {"correct": True, "complete": True} for d in DEPTHS}
        self.assertEqual(target_depth(outcomes), 24)
        outcomes["24"]["correct"] = False
        self.assertEqual(target_depth(outcomes), 28)
        outcomes["32"]["complete"] = False
        self.assertIsNone(target_depth(outcomes))

    def test_300_questions_have_disjoint_stratified_splits(self):
        rows = [{"question_id": f"{d}-{i}", "target_depth": d} for d in DEPTHS for i in range(100)]
        parts = stratified_partitions(rows, 42)
        self.assertEqual({name: len(part) for name, part in parts.items()},
                         {"train": 210, "validation": 45, "test": 45})
        seen = []
        for name, part in parts.items():
            self.assertEqual({d: sum(r["target_depth"] == d for r in part) for d in DEPTHS},
                             {d: 70 if name == "train" else 15 for d in DEPTHS})
            seen.extend(r["question_id"] for r in part)
        self.assertEqual(len(seen), len(set(seen)))


def create_evidence(directory, per_depth=10):
    source = manifest(directory, {"per_depth": per_depth, "model_revision": MODEL_REVISION,
                                 "dataset_revision": DATASET_REVISION,
                                 "max_new_tokens": 1024, "seed": 42}, "label")
    rows = []
    for depth in DEPTHS:
        for i in range(per_depth):
            text = f"Question category{depth} amount {i} calculate value"
            row = {"question_id": f"sample-{depth}-{i}", "question": text,
                   "question_key": text_key(text), "reference_answer": "1",
                   "target_depth": depth, "outcomes": {}, "trial_sha256": {}}
            for action in DEPTHS:
                trial = {"run_identity": source["identity"], "question_id": row["question_id"],
                         "depth": action, "correct": action >= depth, "complete": True,
                         "finish_reason": "eos", "input_tokens": 10, "output_tokens": 2,
                         "seconds": 0.1, "peak_allocated_gib": 1., "peak_reserved_gib": 1.}
                path = directory / "questions" / row["question_id"] / f"depth_{action}.json"
                atomic_json(path, trial)
                row["outcomes"][str(action)] = {key: value for key, value in trial.items()
                                               if key not in ("run_identity", "question_id", "depth")}
                row["trial_sha256"][str(action)] = digest(path)
            rows.append(row)
    test_ids = {r["question_id"] for r in stratified_partitions(rows, 42)["test"]}
    for row in rows:
        if row["question_id"] in test_ids:
            row["question"] += " heldoutonlykeyword"
            row["question_key"] = text_key(row["question"])
    atomic_json(directory / "selected_dataset.json", rows)
    atomic_json(directory / "demonstrations.json", [])
    return rows


class ControllerTrainingTests(unittest.TestCase):
    def test_quota_shortage_resumes_without_rerunning_saved_generations(self):
        from v5.controller_label import main
        candidates = [{"question_id": f"candidate-{i}", "question": f"Candidate question {i}",
                       "question_key": text_key(f"Candidate question {i}"), "reference_answer": "1",
                       "fake_target": depth} for i, depth in enumerate((24, 24, 28, 28, 32, 32))]
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

        def generate_fake(tokenizer, model, q, examples, depth, cap):
            return {"question_id": q["question_id"], "depth": depth,
                    "correct": depth >= q["fake_target"], "complete": True,
                    "finish_reason": "eos", "extracted_answer": "1", "input_tokens": 1,
                    "output_tokens": 2, "seconds": 0.01,
                    "peak_allocated_gib": 1., "peak_reserved_gib": 1.}

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2],
                                         prefix=".pytest-tmp-v5-controller-") as tmp:
            output = Path(tmp) / "labels"
            with patch.dict("sys.modules", {"torch": fake_torch}), \
                 patch("v5.controller_label.prepare_candidates", return_value=(candidates, [])), \
                 patch("v5.controller_label.load_model", return_value=(object(), object())), \
                 patch("v5.controller_label.warmup"), \
                 patch("v5.controller_label.generate", side_effect=generate_fake) as generation, \
                 contextlib.redirect_stdout(io.StringIO()):
                argv = ["label", "--output-dir", str(output), "--per-depth", "2", "--max-candidates", "3"]
                with patch("sys.argv", argv):
                    self.assertEqual(main(), 2)
                self.assertEqual(generation.call_count, 9)
                with self.assertRaisesRegex(ValueError, "quotas"):
                    load_labels(output)
                argv[-1] = "6"
                with patch("sys.argv", argv):
                    self.assertEqual(main(), 0)
                self.assertEqual(generation.call_count, 18)
                rows, _, _ = load_labels(output)
                self.assertEqual(len(rows), 6)
                self.assertEqual(read_json(output / "summary.json")["selected_target_counts"],
                                 {"24": 2, "28": 2, "32": 2})

    def test_incomplete_quotas_cannot_be_trained_and_modified_trials_fail_checksums(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2],
                                         prefix=".pytest-tmp-v5-controller-") as tmp:
            folder = Path(tmp)
            rows = create_evidence(folder)
            atomic_json(folder / "selected_dataset.json", rows[:-1])
            with self.assertRaisesRegex(ValueError, "quotas"):
                load_labels(folder)
            atomic_json(folder / "selected_dataset.json", rows)
            path = folder / "questions" / rows[0]["question_id"] / "depth_24.json"
            trial = read_json(path)
            trial["correct"] = not trial["correct"]
            atomic_json(path, trial)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_labels(folder)

    def test_fit_uses_train_vocabulary_and_frozen_test_evaluation(self):
        from v5.controller_train import train
        from v5.controller_evaluate import evaluate
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2],
                                         prefix=".pytest-tmp-v5-controller-") as tmp:
            root = Path(tmp)
            create_evidence(root / "labels")
            report = train(root / "labels", root / "training")
            self.assertFalse(report["test_used_for_model_selection"])
            checkpoint = root / "training" / "controller.joblib"
            controller = load_controller(checkpoint)
            vocabulary = controller["pipeline"].named_steps["text"].transformer_list[0][1].vocabulary_
            self.assertNotIn("heldoutonlykeyword", vocabulary)
            before = digest(checkpoint)
            metrics = evaluate(root / "labels", checkpoint, root / "evaluation")
            self.assertEqual(metrics["controller"]["questions"], 6)
            self.assertEqual(digest(checkpoint), before)
            self.assertFalse(metrics["test_used_for_model_selection"])


if __name__ == "__main__":
    unittest.main()
