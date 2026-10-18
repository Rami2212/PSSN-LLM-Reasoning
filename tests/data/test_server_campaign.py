"""Check campaign data integrity and cross-batch vector relocation without a GPU."""

from argparse import Namespace
import copy
import json

import numpy as np
import pytest

from src.data.black_hole_dataset import atomic_jsonl
from src.data.server_campaign import (
    MODEL_ID, REVISION, baseline, campaign_sources, finalize, merge_states, resolve_vector_file, sentences,
)


def source(tmp_path, batch, problem, value):
    root = tmp_path / "campaign/batches" / batch
    evidence = root / "experiments" / problem
    evidence.mkdir(parents=True)
    for filename in ("manifest.json", "embedded_question_and_trace.json", "label_evidence.jsonl"):
        (evidence / filename).write_text('{"preserved": true}\n', encoding="utf-8")
    vectors = root / "representations/state_vectors.npy"
    vectors.parent.mkdir()
    np.save(vectors, np.array([[value, value + 1]], dtype=np.float32))
    row = {
        "schema_version": "black-hole-training-dataset-v1", "dataset_row_id": batch,
        "problem_id": problem, "trace_id": "trace:" + problem,
        "label": "low_value_in_context",
        "raw_evidence": {"model_id": MODEL_ID, "model_revision": REVISION,
                         "protocol_name": "pssn-sentence-removal-3samples-v1"},
        "source_state_references": [{"state_id": "S1", "representation": {
            "vector_index": 0, "vector_file": str(vectors)}}],
    }
    dataset = root / "dataset/black_hole_training_dataset.jsonl"
    atomic_jsonl(dataset, [row])
    atomic_jsonl(root / "labels/black_hole_sentence_labels.jsonl", [
        {"label_id": row["dataset_row_id"], "problem_id": problem,
         "trace_id": row["trace_id"], "label": row["label"]}])
    return dataset, row


def arguments(tmp_path, target=2):
    return Namespace(campaign=tmp_path / "campaign", include_existing=[], revision=REVISION,
                     target=target, state_root=tmp_path)


def test_combines_local_vector_indices_without_aliasing_or_changing_sources(tmp_path):
    first, original = source(tmp_path, "batch-001", "q1", 10)
    second, _ = source(tmp_path, "batch-002", "q2", 20)
    # The same vector can occur in more than one context for a question.
    extra = copy.deepcopy(original)
    extra["dataset_row_id"] = "another-context"
    atomic_jsonl(first, [original, extra])
    labels_path = first.parent.parent / "labels/black_hole_sentence_labels.jsonl"
    original_label = json.loads(labels_path.read_text())
    atomic_jsonl(labels_path, [original_label, {**original_label, "label_id": extra["dataset_row_id"]}])
    before = first.read_bytes(), second.read_bytes()
    finalize(arguments(tmp_path))
    destination = tmp_path / "campaign/final"
    rows = [json.loads(line) for line in (destination / "black_hole_training_dataset.jsonl").read_text().splitlines()]
    matrix = np.load(destination / "state_vectors.npy")
    assert matrix.shape == (2, 2)
    for row in rows:
        ref = row["source_state_references"][0]["representation"]
        np.testing.assert_array_equal(matrix[ref["vector_index"]],
                                      [10, 11] if row["problem_id"] == "q1" else [20, 21])
        assert ref["original_vector_index"] == 0
    assert (destination / "evidence/q2/manifest.json").is_file()
    assert json.loads((destination / "dataset_summary.json").read_text())["question_count"] == 2
    combined_labels = [json.loads(line) for line in (destination / "black_hole_sentence_labels.jsonl").read_text().splitlines()]
    assert len(combined_labels) == 3
    assert original_label in combined_labels
    assert (first.read_bytes(), second.read_bytes()) == before
    with pytest.raises(ValueError, match="already exists"):
        finalize(arguments(tmp_path))


def test_target_counts_questions_instead_of_label_rows(tmp_path):
    dataset, row = source(tmp_path, "batch-001", "q1", 10)
    other = copy.deepcopy(row)
    other["dataset_row_id"] = "context-2"
    atomic_jsonl(dataset, [row, other])
    with pytest.raises(ValueError, match="Only 1/2 usable questions"):
        finalize(arguments(tmp_path))
    assert not (tmp_path / "campaign/final").exists()


def test_rejects_question_overlap_and_incompatible_revision(tmp_path):
    source(tmp_path, "batch-001", "q1", 10)
    dataset, row = source(tmp_path, "batch-002", "q1", 20)
    with pytest.raises(ValueError, match="Question appears in multiple"):
        campaign_sources(arguments(tmp_path))
    row["raw_evidence"]["model_revision"] = "0" * 40
    atomic_jsonl(dataset, [row])
    with pytest.raises(ValueError, match="different model revisions"):
        campaign_sources(arguments(tmp_path))


def test_relocates_colab_and_windows_paths_into_local_pssn2(tmp_path):
    local = tmp_path / "artifacts/expanded_baseline/a/representations/stamp/state_vectors.npy"
    local.parent.mkdir(parents=True)
    np.save(local, np.zeros((1, 2)))
    suffix = "artifacts/expanded_baseline/a/representations/stamp/state_vectors.npy"
    assert resolve_vector_file("/content/drive/MyDrive/PSSN2/" + suffix, tmp_path) == local.resolve()
    assert resolve_vector_file("C:\\old\\PSSN\\" + suffix.replace("/", "\\"), tmp_path) == local.resolve()
    with pytest.raises(FileNotFoundError, match="Missing Week 2 vector"):
        resolve_vector_file("/content/drive/MyDrive/PSSN2/missing.npy", tmp_path)


def test_invalid_vector_index_does_not_write_final_directory(tmp_path):
    dataset, row = source(tmp_path, "batch-001", "q1", 10)
    row["source_state_references"][0]["representation"]["vector_index"] = -1
    atomic_jsonl(dataset, [row])
    with pytest.raises(ValueError, match="Invalid source vector index"):
        finalize(arguments(tmp_path, target=1))
    assert not (tmp_path / "campaign/final").exists()


def test_baseline_interrupt_saves_completed_record_and_resume_skips_it(tmp_path, monkeypatch):
    from src.data import server_campaign as campaign
    from src.models import qwen
    from src.utils import environment

    batch = tmp_path / "campaign/batches/batch-001"
    examples = [{"id": f"q{i}", "question": "What is 20 + 22?", "reference_answer": "42"} for i in (1, 2)]
    sample = {"run_id": "fixed-run", "sample_sha256": "sample"}
    config = {"seed": 44, "max_new_tokens": 1024, "revision": REVISION, "created_at_utc": "2026-10-16T00:00:00+00:00"}
    monkeypatch.setattr(campaign, "freeze_batch", lambda *_: (examples, sample))
    monkeypatch.setattr(campaign, "provenance", lambda: {"code_sha256": {}, "packages": {}})
    monkeypatch.setattr(environment, "load_qwen_model", lambda *a, **k: (None, None, "float16"))
    monkeypatch.setattr(qwen, "QwenGenerator", lambda *a, **k: None)
    calls = []

    def generate(example, _):
        calls.append(example["id"])
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return {"problem_id": example["id"], "question": example["question"],
                "prompt": "Calculate 20 + 22.", "reference_answer": "42",
                "reasoning_trace": "20 + 22 = 42. Final answer: 42", "input_tokens": 12,
                "output_tokens": 14, "finish_reason": "eos", "model_metadata": {},
                "generation_config": {"max_new_tokens": 1024}, "inference_latency_seconds": 0.1,
                "peak_gpu_memory_allocated_bytes": 100, "peak_gpu_memory_reserved_bytes": 100}

    from src.baseline import cot
    monkeypatch.setattr(cot, "generate_cot_record", generate)
    with pytest.raises(KeyboardInterrupt):
        baseline(None, config, batch)
    assert len(campaign.load_rows(batch / "baseline/raw_attempts.jsonl")) == 1
    assert campaign.read_json(batch / "baseline/run_status.json")["status"] == "stopped"
    baseline(None, config, batch)
    assert calls == ["q1", "q2", "q2"]
    assert len(campaign.load_rows(batch / "baseline/segmentation_ready_traces.jsonl")) == 2
    assert campaign.read_json(batch / "baseline/run_status.json")["status"] == "completed"


def test_merge_state_pool_includes_old_and_new_without_changing_sources(tmp_path):
    args = arguments(tmp_path)
    old = tmp_path / "pilot/semantic_states.jsonl"
    batch = args.campaign / "batches/batch-001"
    new = batch / "state-dataset/semantic_states.jsonl"
    vectors = tmp_path / "states.npy"
    np.save(vectors, np.array([[1, 2]], dtype=np.float32))
    def trace(problem):
        return {"problem_id": problem, "trace_id": "trace:" + problem, "states": [
            {"state_id": "S1", "representation": {"vector_file": str(vectors), "vector_index": 0}}]}
    atomic_jsonl(old, [trace("old-q")])
    atomic_jsonl(new, [trace("new-q")])
    args.existing_states = [old]
    before = old.read_bytes(), new.read_bytes()
    merge_states(args, {}, batch)
    merged = [json.loads(line) for line in (batch / "combined-states/semantic_states.jsonl").read_text().splitlines()]
    assert {row["problem_id"] for row in merged} == {"old-q", "new-q"}
    assert (old.read_bytes(), new.read_bytes()) == before
    # Selection's new-only input is still the original new batch file.
    assert [json.loads(line)["problem_id"] for line in new.read_text().splitlines()] == ["new-q"]
    atomic_jsonl(new, [{**trace("old-q"), "trace_id": "conflicting-trace"}])
    with pytest.raises(ValueError, match="Conflicting state traces"):
        merge_states(args, {}, batch)


def test_sentence_inventory_uses_only_new_selection(tmp_path):
    batch = tmp_path / "batch"
    selection = {"eligible": True, "baseline_correct": True, "problem_id": "new-q", "trace_id": "new-trace",
                 "baseline_result": {"question": "What is 20 + 22?", "reference_answer": "42", "states": [],
                                     "reasoning_trace": "<think>First add the tens. Then add the units.</think>42"}}
    atomic_jsonl(batch / "selection/eligible_traces.jsonl", [selection])
    atomic_jsonl(batch / "combined-states/semantic_states.jsonl", [{"problem_id": "old-q"}])
    sentences(None, {}, batch)
    rows = [json.loads(line) for line in (batch / "sentences/new_sentences.jsonl").read_text().splitlines()]
    assert rows and {row["problem_id"] for row in rows} == {"new-q"}


def test_finalizer_preserves_all_existing_questions_and_original_labels(tmp_path):
    old_dataset, _ = source(tmp_path, "batch-000", "old-q", 1)
    source(tmp_path, "batch-001", "new-q", 2)
    source(tmp_path, "batch-002", "surplus-q", 3)
    args = arguments(tmp_path)
    args.include_existing = [old_dataset]
    finalize(args)
    summary = json.loads((args.campaign / "final/dataset_summary.json").read_text())
    assert "old-q" in summary["selected_problem_ids"]
    assert summary["existing_questions_preserved"] == 1
