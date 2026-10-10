"""Resumable state-removal continuations with durable per-experiment records."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.baseline.cot import build_cot_prompt
from src.evaluation.accuracy import evaluate_record
from .state_removal import remove_semantic_state

PROTOCOL = "retain-surviving-nonfinal-states-v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def save_json(path: Path, record: Any) -> None:
    """Flush a complete JSON file before replacing its previous version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run_removal_experiments(
    selected_traces: Iterable[Mapping[str, Any]],
    generator: Any,
    output_dir: str | Path,
    *,
    run_config: Mapping[str, Any],
    persist_callback: Callable[[Path], None] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run independent removals, checkpoint each outcome, and resume by identity.

    ``persist_callback`` must return only after remote persistence succeeds.
    Persistence failures stop the run, while generation failures are saved as
    unscored records. ``limit`` counts new attempts this invocation.
    """
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    selected = [dict(row) for row in selected_traces]
    if not selected:
        raise ValueError("no selected traces were provided")
    tasks = []
    seen = set()
    for selection in selected:
        baseline = selection.get("baseline_result")
        if selection.get("eligible") is not True or not isinstance(baseline, Mapping) or baseline.get("is_correct") is not True:
            raise ValueError("every selection must contain an eligible, baseline-correct result")
        if selection.get("trace_id") != baseline.get("trace_id") or selection.get("problem_id") != baseline.get("problem_id"):
            raise ValueError("selection and baseline identifiers do not match")
        expected_candidates = [state["state_id"] for state in baseline["states"] if not state["is_final_state"]]
        if selection.get("removable_state_ids") != expected_candidates:
            raise ValueError("removal candidates do not match the validated baseline")
        for state_id in expected_candidates:
            removal = remove_semantic_state(baseline["states"], state_id, source_trace=baseline["reasoning_trace"])
            key = _digest([baseline["problem_id"], baseline["trace_id"], state_id])
            if key in seen:
                raise ValueError("duplicate removal experiment identity")
            seen.add(key)
            tasks.append((key, baseline, removal))

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    manifest = {"protocol": PROTOCOL, "source_sha256": _digest(selected), "run_config": dict(run_config)}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("checkpoint manifest differs: use a new output directory")
    else:
        if list((destination / "records").glob("*.json")):
            raise ValueError("checkpoint records exist without a manifest")
        save_json(manifest_path, manifest)
    if persist_callback:
        persist_callback(manifest_path)

    completed: list[dict[str, Any]] = []
    attempted = 0
    for key, baseline, removal in tasks:
        record_path = destination / "records" / f"{key}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("experiment_id") != key or record.get("trace_id") != baseline["trace_id"] or record.get("removed_state_id") != removal["removed_state_id"]:
                raise ValueError("corrupt checkpoint identity")
            if persist_callback:
                persist_callback(record_path)
            completed.append(record)
            continue
        if limit is not None and attempted >= limit:
            continue
        attempted += 1
        prompt = baseline.get("experiment_metadata", {}).get("prompt") or build_cot_prompt(baseline["question"])
        record = {
            **removal,
            "experiment_id": key,
            "protocol": PROTOCOL,
            "experiment_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "modified_context": removal["continuation_context"],
            "baseline_result": dict(baseline),
            "baseline_correct": True,
            "reference_answer": baseline["reference_answer"],
        }
        try:
            result = generator.generate_continuation(prompt, removal["continuation_context"])
            evaluation = evaluate_record({
                "problem_id": baseline["problem_id"], "reference_answer": baseline["reference_answer"],
                "reasoning_trace": result.text, "finish_reason": result.finish_reason,
                "output_tokens": result.output_tokens, "generation_config": result.generation_config,
            })
            record.update({
                "continuation": result.text, **evaluation,
                "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                "finish_reason": result.finish_reason, "generation_config": result.generation_config,
                "model_metadata": result.model_metadata, **result.metrics,
                "experiment_status": "incomplete" if evaluation["is_correct"] is None else "completed",
            })
        except Exception as exc:
            record.update({"experiment_status": "failed", "is_correct": None,
                           "predicted_answer": None, "error_type": type(exc).__name__, "error": str(exc)})
        save_json(record_path, record)
        if persist_callback:
            persist_callback(record_path)
        completed.append(record)
        if progress_callback:
            progress_callback(record)

    summary = {
        "total_experiments": len(tasks), "saved_experiments": len(completed),
        "pending_experiments": len(tasks) - len(completed), "new_attempts": attempted,
        "correct": sum(row.get("is_correct") is True for row in completed),
        "incorrect": sum(row.get("is_correct") is False for row in completed),
        "incomplete": sum(row.get("experiment_status") == "incomplete" for row in completed),
        "failed": sum(row.get("experiment_status") == "failed" for row in completed),
    }
    save_json(destination / "removal_summary.json", summary)
    # Shards are authoritative; the combined export is rebuilt on every resume.
    export = destination / "state_removal_results.jsonl"
    temporary = export.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in completed:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(export)
    if persist_callback:
        persist_callback(destination / "removal_summary.json")
        persist_callback(export)
    return summary
