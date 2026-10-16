import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.black_hole.removal_experiment import run_removal_experiments
from src.black_hole.run_experiments import RcloneMirror
from src.models.qwen import GenerationResult


def selected_trace():
    texts = ["Compute the initial amount carefully.", "Combine the calculated amounts together.", "Final answer: 42 is the result."]
    source = "\n".join(texts)
    states = []
    start = 0
    for index, text in enumerate(texts):
        states.append({"problem_id": "p1", "trace_id": "t1", "state_id": f"S{index + 1}",
                       "state_index": index, "text": text, "start_char": start,
                       "end_char": start + len(text), "token_count": 7, "is_final_state": index == 2})
        start += len(text) + 1
    baseline = {"problem_id": "p1", "trace_id": "t1", "question": "What is the amount?",
                "reference_answer": "42", "is_correct": True, "reasoning_trace": source,
                "states": states, "experiment_metadata": {}}
    return [{"problem_id": "p1", "trace_id": "t1", "eligible": True,
             "removable_state_ids": ["S1", "S2"], "baseline_result": baseline}]


def response(*, reason="eos", text="Final answer: 42"):
    return GenerationResult(text=text, input_tokens=30, output_tokens=12,
                            generation_config={"max_new_tokens": 1024}, finish_reason=reason,
                            metrics={"inference_latency_seconds": 0.5}, model_metadata={"model_id": "Qwen/Qwen3-4B"})


def test_continuations_are_independent_hide_saved_answer_and_preserve_baseline(tmp_path):
    source = selected_trace()
    original = json.dumps(source, sort_keys=True)
    generator = Mock()
    generator.generate_continuation.return_value = response()
    persisted = []
    summary = run_removal_experiments(source, generator, tmp_path, run_config={"seed": 42}, persist_callback=persisted.append)
    assert summary["correct"] == 2
    assert json.dumps(source, sort_keys=True) == original
    for call in generator.generate_continuation.call_args_list:
        assert "Final answer" not in call.args[1]
    assert generator.generate_continuation.call_args_list[0].args[1] == source[0]["baseline_result"]["states"][1]["text"]
    assert generator.generate_continuation.call_args_list[1].args[1] == source[0]["baseline_result"]["states"][0]["text"]
    records = [json.loads(path.read_text()) for path in (tmp_path / "records").glob("*.json")]
    assert all(record["baseline_result"] == source[0]["baseline_result"] for record in records)
    assert sum(path.parent.name == "records" for path in persisted) == 2


def test_resume_limit_and_config_guard(tmp_path):
    generator = Mock()
    generator.generate_continuation.return_value = response()
    first = run_removal_experiments(selected_trace(), generator, tmp_path, run_config={"seed": 42}, limit=1)
    assert first["pending_experiments"] == 1
    second = run_removal_experiments(selected_trace(), generator, tmp_path, run_config={"seed": 42})
    assert second["new_attempts"] == 1
    assert generator.generate_continuation.call_count == 2
    third = run_removal_experiments(selected_trace(), generator, tmp_path, run_config={"seed": 42})
    assert third["new_attempts"] == 0
    with pytest.raises(ValueError, match="manifest differs"):
        run_removal_experiments(selected_trace(), generator, tmp_path, run_config={"seed": 99})


def test_incomplete_and_generation_failure_remain_unscored(tmp_path):
    generator = Mock()
    generator.generate_continuation.side_effect = [response(reason="length"), RuntimeError("generation failed")]
    summary = run_removal_experiments(selected_trace(), generator, tmp_path, run_config={})
    assert summary["incomplete"] == summary["failed"] == 1
    assert summary["incorrect"] == 0
    assert all(json.loads(path.read_text())["is_correct"] is None for path in (tmp_path / "records").glob("*.json"))


def test_cloud_failure_stops_after_durable_local_record_and_resume_reuploads(tmp_path):
    generator = Mock()
    generator.generate_continuation.return_value = response()
    def broken_upload(path):
        if path.parent.name == "records":
            raise RuntimeError("Drive unavailable")
    with pytest.raises(RuntimeError, match="Drive unavailable"):
        run_removal_experiments(selected_trace(), generator, tmp_path, run_config={}, persist_callback=broken_upload)
    assert len(list((tmp_path / "records").glob("*.json"))) == 1
    assert generator.generate_continuation.call_count == 1
    uploads = []
    run_removal_experiments(selected_trace(), generator, tmp_path, run_config={}, persist_callback=uploads.append)
    assert generator.generate_continuation.call_count == 2
    assert sum(path.parent.name == "records" for path in uploads) == 2


def test_drive_mirror_uses_completed_file_uploads_and_rejects_manifest_conflict(tmp_path):
    mirror = RcloneMirror(tmp_path, "gdrive:PSSN2/artifacts/run1")
    mirror.command = Mock(return_value=SimpleNamespace(stdout=""))
    record = tmp_path / "records" / "record.json"
    mirror(record)
    assert mirror.command.call_args.args[:3] == ("copyto", str(record), "gdrive:PSSN2/artifacts/run1/records/record.json")
    (tmp_path / "manifest.json").write_text('{"seed":42}')
    mirror.command.side_effect = [SimpleNamespace(stdout=""), SimpleNamespace(stdout="manifest.json\n"), SimpleNamespace(stdout='{"seed":99}')]
    with pytest.raises(ValueError, match="manifests differ"):
        mirror.restore()
