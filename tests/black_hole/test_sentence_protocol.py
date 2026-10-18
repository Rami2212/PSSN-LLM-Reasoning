"""Checkpoint replay, source mapping and Drive failure behavior without a GPU."""

import json
import random
from unittest.mock import patch

import pytest

from src.black_hole import sentence_experiment as engine
from src.black_hole.run_sentence_experiments import ContinuousMirror, LocalOnlyMirror, prepare_candidates, protocol
from src.black_hole.sentence_checkpoint import SentenceCheckpoint, collect_trials, sentence_units


def test_partial_trials_resume_same_random_stream_without_regenerating(tmp_path):
    checkpoint = SentenceCheckpoint(tmp_path)
    rng = random.Random()
    calls = []
    def seed():
        rng.seed(42)
    def sample(index):
        calls.append(index)
        if index == 1:
            raise KeyboardInterrupt()
        return {"sample_index": index, "value": rng.random()}
    def restore(state):
        # JSON turns the two tuple levels of Python's RNG state into lists.
        rng.setstate((state[0], tuple(state[1]), state[2]))
    with pytest.raises(KeyboardInterrupt):
        collect_trials(checkpoint, "prompt", 3, seed, sample, rng.getstate, restore)
    resumed_calls = []
    def resumed_sample(index):
        resumed_calls.append(index)
        return {"sample_index": index, "value": rng.random()}
    actual = collect_trials(checkpoint, "prompt", 3, seed, resumed_sample, rng.getstate, restore)
    expected = random.Random(42)
    assert [trial["value"] for trial in actual] == [expected.random() for _ in range(3)]
    assert resumed_calls == [1, 2]
    assert len(list((tmp_path / "checkpoints" / "trials").glob("*.json"))) == 3


def test_tampered_checkpoint_is_rejected(tmp_path):
    checkpoint = SentenceCheckpoint(tmp_path)
    request = {"prompt": "one"}
    checkpoint.save("trials", request, {"correct": True})
    path = next((tmp_path / "checkpoints" / "trials").glob("*.json"))
    record = json.loads(path.read_text())
    record["result"]["correct"] = False
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="integrity"):
        checkpoint.load("trials", request)
    assert checkpoint.load("trials", {"prompt": "different"}) is None


def test_sentence_offsets_map_to_overlapping_states():
    cot = "First reasoning sentence. Second reasoning sentence."
    trace = "<think>" + cot + "</think>"
    middle = 18
    candidate = {"reasoning_trace": trace, "source_states": [
        {"state_id": "S1", "start_char": 7, "end_char": middle,
         "text": trace[7:middle], "is_final_state": False},
        {"state_id": "S2", "start_char": middle, "end_char": 7 + len(cot),
         "text": trace[middle:7 + len(cot)], "is_final_state": False}]}
    units = sentence_units(candidate, engine.split_sentences(cot), cot)
    assert units[0]["source_state_ids"] == ["S1", "S2"]
    assert units[1]["source_state_ids"] == ["S2"]
    for unit in units:
        assert trace[unit["start_char_in_source"]:unit["end_char_in_source"]] == unit["text"]
    candidate["source_states"][0]["text"] = "wrong"
    with pytest.raises(ValueError, match="span mismatch"):
        sentence_units(candidate, engine.split_sentences(cot), cot)


def test_three_sample_gates_and_truncated_answer():
    assert engine.N_RESAMPLES == 3
    assert protocol()["minimum_correct"] == 2
    assert 2 / 3 >= engine.SUFFICIENCY_THRESHOLD
    assert 1 / 3 < engine.SUFFICIENCY_THRESHOLD
    assert 1 - 1 / 3 >= engine.NECESSITY_THRESHOLD
    assert 1 - 2 / 3 < engine.NECESSITY_THRESHOLD
    assert engine.ATTAINABILITY_THRESHOLD == 0.8
    assert engine.boxed_answers(r"\boxed{15") == []


def test_drive_batch_upload_failure_keeps_accepting_local_checkpoints(tmp_path):
    mirror = ContinuousMirror(tmp_path, "gdrive:PSSN2/new-run")
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text("{}")
    second.write_text("{}")
    mirror(first)
    mirror(second)
    lists = []
    def command(*args):
        assert args[0] == "copy"
        listing = args[args.index("--files-from-raw") + 1]
        from pathlib import Path
        lists.append(Path(listing).read_text().splitlines())
    with patch.object(mirror.mirror, "command", side_effect=command):
        mirror.flush()
    assert lists == [["a.json", "b.json"]]
    mirror(first)
    with patch.object(mirror.mirror, "command", side_effect=RuntimeError("offline")):
        with pytest.raises(RuntimeError, match="offline"):
            mirror.flush()
    assert first.exists()
    assert "a.json" in mirror.pending
    mirror(second)
    assert mirror.pending == {"a.json", "b.json"}


def test_uploader_runner_change_can_resume_existing_run_identity():
    from src.black_hole.run_sentence_experiments import _runner_only_change

    previous = {"protocol": {"samples": 3}, "code_sha256": {
        "run_sentence_experiments.py": "old-runner", "sentence_experiment.py": "same-engine"}}
    current = {"protocol": {"samples": 3}, "code_sha256": {
        "run_sentence_experiments.py": "new-runner", "sentence_experiment.py": "same-engine"}}
    assert _runner_only_change(previous, current)
    current["code_sha256"]["sentence_experiment.py"] = "changed-engine"
    assert not _runner_only_change(previous, current)


def test_manual_backup_mode_never_calls_rclone(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text('{"complete": true}')
    with patch("subprocess.run", side_effect=AssertionError("unexpected external command")):
        mirror = LocalOnlyMirror()
        mirror.start()
        mirror(path)
        mirror.close()
    assert mirror.pending_count == 0
    assert json.loads(path.read_text())["complete"] is True


def test_selection_validates_all_rows_and_keeps_original_metadata(tmp_path):
    row = {"eligible": True, "baseline_correct": True, "problem_id": "gsm8k-train-00000",
           "trace_id": "run:q0", "baseline_result": {"question": "A question", "reference_answer": "15",
           "reasoning_trace": "<think>Calculate an answer of 15.</think>", "states": [],
           "experiment_metadata": {"seed": 44}}}
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    candidates = prepare_candidates(path)
    assert candidates[0]["original_generation_metadata"] == {"seed": 44}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_candidates(path)


def test_server_limit_and_resume_process_all_traces(tmp_path, monkeypatch):
    """Exercise the actual session loop with model/cloud boundaries stubbed."""
    import sys
    from types import SimpleNamespace
    from src.black_hole import run_sentence_experiments as runner
    source = tmp_path / "source.jsonl"
    rows = [{"eligible": True, "baseline_correct": True, "problem_id": f"q{i}", "trace_id": f"run:q{i}",
             "baseline_result": {"question": f"Question {i}", "reference_answer": "15",
                                 "reasoning_trace": "<think>Calculate an answer of 15.</think>", "states": []}}
            for i in range(2)]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    class Parameter:
        dtype = "float16"
        def requires_grad_(self, value):
            pass
    class Model:
        config = SimpleNamespace(_commit_hash=engine.MODEL_REVISION)
        def parameters(self):
            return iter([Parameter()])
        def eval(self):
            return self
    tokenizer = SimpleNamespace(pad_token_id=0, init_kwargs={"_commit_hash": engine.MODEL_REVISION})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="test", float16="float16",
                        cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="5.0.0",
                        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
                        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: Model())))
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    monkeypatch.setattr(runner, "collect_environment_info", lambda: {"backend": "test"})
    monkeypatch.setattr(runner.RcloneMirror, "restore", lambda self: None)
    monkeypatch.setattr(runner.RcloneMirror, "command", lambda *a: None)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="test", returncode=0))
    calls = []
    def experiment(candidate, shared, directory):
        calls.append(candidate["problem_id"])
        folder = directory / candidate["problem_id"]
        folder.mkdir(exist_ok=True)
        return {"eligible_under_adapted_rules": False, "exclusion_reasons": ["cot_necessity_low"],
                "core_experiment_status": "excluded_by_screening"}
    monkeypatch.setattr(engine, "run_question", experiment)
    out = tmp_path / "output"
    arguments = ["--input", str(source), "--output-dir", str(out), "--drive-output", "gdrive:PSSN2/test"]
    runner.main(arguments + ["--limit", "1"])
    assert calls == ["q0"]
    assert json.loads((out / "run_status.json").read_text())["status"] == "paused"
    runner.main(arguments)
    assert calls == ["q0", "q1"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["completed_traces"] == 2
    assert summary["pending_traces"] == 0
    assert json.loads((out / "run_status.json").read_text())["status"] == "completed"
    runner.main(arguments)
    assert calls == ["q0", "q1"]


def test_queue_records_context_for_later_labeling():
    states = ["A", "B", "C", "Answer"]
    def evaluate(mask, idx):
        return {"passes_both": True, "evaluation_id": idx, "evaluation_key": f"condition-{idx}"}
    final, attempts = engine.queue_prune(states, 3, [True] * 4, [0, 1, 2, 3], evaluate)
    assert final == [False, False, True, True]
    assert attempts[0]["context_before_mask"] == [True] * 4
    assert attempts[1]["context_before_mask"] == [False, True, True, True]
    assert attempts[1]["evaluation_key"] == "condition-1"
