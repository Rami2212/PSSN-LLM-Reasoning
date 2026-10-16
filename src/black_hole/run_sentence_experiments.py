"""Week 3.1: resumable three-sample sentence experiments with Drive backup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.utils.environment import collect_environment_info
from src.utils.run_status import RunStatus
from . import sentence_experiment as engine
from .run_experiments import RcloneMirror, _resolve_cached_model_revision
from .sentence_checkpoint import atomic_json, identity, sentence_units


class ContinuousMirror:
    """Batch closed files for background upload; keep local execution independent."""

    def __init__(self, directory, remote, seconds=45):
        self.mirror = RcloneMirror(directory, remote)
        self.directory = Path(directory).resolve()
        self.seconds = seconds
        self.pending = set()
        self.lock = threading.Lock()
        self.upload_lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.thread = None

    def __call__(self, path):
        relative = Path(path).resolve().relative_to(self.directory).as_posix()
        with self.lock:
            self.pending.add(relative)

    @property
    def pending_count(self):
        with self.lock:
            return len(self.pending)

    def flush(self):
        with self.upload_lock:
            with self.lock:
                batch, self.pending = self.pending, set()
            if not batch:
                return
            listing = self.directory / ".upload-files.txt"
            listing.write_text("\n".join(sorted(batch)) + "\n", encoding="utf-8")
            try:
                self.mirror.command("copy", str(self.directory), self.mirror.remote,
                                    "--files-from-raw", str(listing), "--checksum", "--retries", "5")
            except BaseException as exc:
                with self.lock:
                    self.pending.update(batch)
                self.error = exc
                raise
            else:
                self.error = None
            finally:
                listing.unlink(missing_ok=True)

    def start(self):
        def worker():
            while not self.stop.wait(self.seconds):
                try:
                    self.flush()
                except BaseException as exc:
                    # Keep inference/checkpointing moving. The failed batch is
                    # requeued by flush() and retried on the next interval.
                    print(f"Background Drive sync failed; will retry. Local files are safe: {exc}",
                          file=sys.stderr, flush=True)
        self.thread = threading.Thread(target=worker, daemon=True, name="sentence-drive-backup")
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join()
        self.flush()


def prepare_candidates(path):
    candidates = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            baseline = record.get("baseline_result", record)
            problem_id = record.get("problem_id", baseline.get("problem_id"))
            if not isinstance(problem_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", problem_id):
                raise ValueError(f"Invalid problem_id at input line {line_number}")
            if problem_id in seen:
                raise ValueError(f"Duplicate problem_id: {problem_id}; select one trace per question")
            seen.add(problem_id)
            if record.get("baseline_correct") is not True or record.get("eligible") is not True:
                raise ValueError(f"Expected W3001 baseline-correct eligible row: {problem_id}")
            candidate = {"problem_id": problem_id, "trace_id": record["trace_id"],
                         "question": baseline["question"], "reference_answer": str(baseline["reference_answer"]),
                         "reasoning_trace": baseline["reasoning_trace"],
                         "original_generation_metadata": baseline.get("experiment_metadata", {}),
                         "source_states": baseline.get("states", []), "source_baseline_correct": True,
                         "source_line_number": line_number}
            for field in ("trace_id", "question", "reference_answer", "reasoning_trace"):
                if not isinstance(candidate[field], str) or not candidate[field].strip():
                    raise ValueError(f"Missing {field}: {problem_id}")
            cot = candidate["reasoning_trace"].split("</think>", 1)[0].split("<think>", 1)[-1]
            sentence_units(candidate, engine.split_sentences(cot), cot)
            candidates.append(candidate)
    if not candidates:
        raise ValueError("Input contains no eligible traces")
    return candidates


def protocol():
    return {"name": "pssn-sentence-removal-3samples-v1", "n_resamples": 3,
            "minimum_correct": 2, "sufficiency_threshold": 2 / 3,
            "necessity_threshold": 2 / 3, "maximum_no_trace_correct": 1,
            "attainability_threshold": 0.8, "seed": engine.SEED,
            "temperature": engine.TEMPERATURE, "top_p": engine.TOP_P,
            "trace_mode": "embedded", "precision": "float16", "attention": "eager",
            "answer_budget": "2 * tokenizer(reference_answer) token length",
            "thresholds": engine.THRESHOLDS, "random_seeds": engine.RANDOM_SEEDS,
            "run_attribution_core": True, "run_random_core": True,
            "upstream_commit": engine.UPSTREAM_COMMIT,
            "unit": "reasoning_sentence", "source_state_mapping": "character_overlap",
            "incomplete_samples": "count_as_non_correct", "labels_implemented": False}


def _runner_only_change(previous, current):
    """Allow uploader-only runner edits to resume an otherwise identical study."""
    old = json.loads(json.dumps(previous))
    new = json.loads(json.dumps(current))
    old.get("code_sha256", {}).pop("run_sentence_experiments.py", None)
    new.get("code_sha256", {}).pop("run_sentence_experiments.py", None)
    return old == new


def save_evidence(directory, candidate, report, run_identity, persist):
    """Export raw independent-removal evidence for the next labeling ticket."""
    data = json.loads((directory / "embedded_question_and_trace.json").read_text(encoding="utf-8"))
    units = {row["sentence_id"]: row for row in data["sentences"]}
    evidence = []
    def add(unit, result, role, context_before=None):
        evidence.append({"schema_version": "sentence-evidence-v1", "run_identity": run_identity,
                         "problem_id": candidate["problem_id"], "trace_id": candidate["trace_id"],
                         "question": candidate["question"], "reference_answer": candidate["reference_answer"],
                         "model_id": engine.MODEL_ID, "model_revision": engine.MODEL_REVISION,
                         "unit_type": "reasoning_sentence", "evidence_type": role,
                         "unit": unit, "screening_eligible": report["eligible_under_adapted_rules"],
                         "source_line_number": candidate["source_line_number"],
                         "context_before_mask": context_before,
                         "shortcut_filter_applied": role != "original_trace_diagnostic",
                         "protocol": protocol(), "evaluation": result,
                         "raw_trials_directory": f"{candidate['problem_id']}/checkpoints/trials",
                         "core_summary_path": f"{candidate['problem_id']}/summary.json"})
    for result in report.get("single_sentence_deletions", []):
        add(units[result["removed_sentence_id"]], result, "original_trace_diagnostic")
    core = report.get("attribution_core")
    if core:
        evaluations = {row["evaluation_key"]: row for row in
                       (json.loads(line) for line in (directory / "evaluations.jsonl").read_text(encoding="utf-8").splitlines())}
        searches = [("gradient_greedy", core)] + [
            (f"random_greedy_{item['seed']}", item) for item in report.get("random_cores", [])]
        for role, search in searches:
            for attempt in search["attempts"]:
                source_id = core["source_sentence_ids"][attempt["sentence_index"]]
                result = dict(evaluations[attempt["evaluation_key"]])
                result["source_sentence_ids"] = core["source_sentence_ids"]
                add(units[source_id], result, role, attempt["context_before_mask"])
    path = directory / "label_evidence.jsonl"
    atomic_jsonl(path, evidence)
    persist(path)
    return len(evidence)


def atomic_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--drive-output", required=True)
    parser.add_argument("--revision", default=engine.MODEL_REVISION)
    parser.add_argument("--limit", type=int, help="Maximum NEW completed traces this session, not individual trials")
    parser.add_argument("--sync-seconds", type=int, default=45, help="Background Drive backup interval")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.sync_seconds < 1:
        parser.error("--sync-seconds must be positive")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("--revision must be a pinned 40-character commit")
    candidates = prepare_candidates(args.input)
    source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # POSIX flock releases automatically after crashes; do not delete another run's lock.
    import fcntl
    lockfile = (args.output_dir / ".run.lock").open("a")
    try:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another process is already using this local run folder") from exc
    mirror = ContinuousMirror(args.output_dir, args.drive_output, args.sync_seconds)
    status = None
    original_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    try:
        mirror.mirror.restore()
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access the accelerator")
        files = [Path(__file__), Path(engine.__file__), Path(__file__).with_name("sentence_checkpoint.py"),
                 Path(__file__).with_name("run_experiments.py"), Path(sys.modules[RunStatus.__module__].__file__)]
        code_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        identity_record = {"schema_version": "w3.1-sentence-v1", "input_sha256": source_hash,
                           "problem_ids": [row["problem_id"] for row in candidates],
                           "model_id": engine.MODEL_ID, "model_revision": args.revision,
                           "tokenizer_revision": args.revision, "protocol": protocol(),
                           "code_sha256": code_hashes,
                           "torch_version": torch.__version__, "transformers_version": transformers.__version__}
        manifest_path = args.output_dir / "manifest.json"
        if manifest_path.exists():
            saved_identity = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved_identity != identity_record and not _runner_only_change(saved_identity, identity_record):
                raise ValueError("Run input, protocol, code or software changed; choose a new output folder")
            # Keep the original run identity so existing checkpoints and trace
            # completion markers remain valid. This session's manifest records
            # the new runner hash separately in its session metadata.
            run_identity = identity(saved_identity)
        else:
            atomic_json(manifest_path, identity_record)
            run_identity = identity(identity_record)
        for path in args.output_dir.rglob("*"):
            if path.is_file() and not path.name.startswith(".") and not path.name.endswith(".tmp"):
                mirror(path)
        source_copy = args.output_dir / "source" / "eligible_traces.jsonl"
        source_copy.parent.mkdir(parents=True, exist_ok=True)
        if not source_copy.exists():
            temporary = source_copy.with_suffix(".tmp")
            shutil.copyfile(args.input, temporary)
            temporary.replace(source_copy)
        if hashlib.sha256(source_copy.read_bytes()).hexdigest() != source_hash:
            raise ValueError("Preserved source file hash differs from input")
        mirror(source_copy)
        # Start syncing recovered/queued files in parallel with model loading
        # and inference; do not make startup wait for a potentially large copy.
        mirror.start()
        status = RunStatus(args.output_dir / "run_status.json", run_id=args.output_dir.name,
                           total_experiments=len(candidates), persist_callback=mirror)
        status.start(phase="loading_model", metadata={"protocol": protocol(), "input_sha256": source_hash,
                     "input_path": str(args.input), "output_dir": str(args.output_dir),
                     "drive_output": args.drive_output, "environment": collect_environment_info(),
                     "sync_interval_seconds": args.sync_seconds, "progress_unit": "traces",
                     "run_identity": run_identity})
        already_completed = sum((args.output_dir / row["problem_id"] / "trace_completed.json").exists()
                                for row in candidates)
        status.update(progress={"completed_experiments": already_completed,
                               "pending_experiments": len(candidates) - already_completed})
        engine.MODEL_REVISION = args.revision
        engine.SOURCE_SELECTION = str(args.input)
        tokenizer = AutoTokenizer.from_pretrained(engine.MODEL_ID, revision=args.revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype_key = "dtype" if tuple(map(int, transformers.__version__.split('.')[:2])) >= (4, 56) else "torch_dtype"
        model = AutoModelForCausalLM.from_pretrained(
            engine.MODEL_ID, revision=args.revision, device_map={"": 0},
            attn_implementation="eager", **{dtype_key: torch.float16}).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        resolved = _resolve_cached_model_revision(engine.MODEL_ID, args.revision, model, tokenizer)
        if resolved != args.revision or next(model.parameters()).dtype != torch.float16:
            raise ValueError("Actual revision or precision differs from the pinned protocol")
        session_path = args.output_dir / "sessions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        atomic_json(session_path, {"created_at_utc": engine.utc_now(), "environment": collect_environment_info(),
                    "model_revision": resolved, "tokenizer_revision": resolved,
                    "code_sha256": code_hashes, "repository_commit": git.stdout.strip(),
                    "pip_freeze": subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                                 check=True, capture_output=True, text=True).stdout})
        mirror(session_path)
        def phase(problem_id, state, name, count):
            status.update(status="running", phase=name,
                          current_experiment={"problem_id": problem_id, "question_status": state,
                                              "completed_conditions": count})
        shared = {"tokenizer": tokenizer, "model": model, "persist": mirror, "on_phase": phase}
        new = 0
        for candidate in candidates:
            directory = args.output_dir / candidate["problem_id"]
            marker = directory / "trace_completed.json"
            if marker.exists():
                saved = json.loads(marker.read_text(encoding="utf-8"))
                if saved["run_identity"] != run_identity:
                    raise ValueError("Completed trace belongs to another protocol")
                continue
            if args.limit is not None and new >= args.limit:
                break
            result = engine.run_question(candidate, shared, args.output_dir)
            count = save_evidence(directory, candidate, result, run_identity, mirror) if (directory / "embedded_question_and_trace.json").exists() else 0
            atomic_json(marker, {"problem_id": candidate["problem_id"], "trace_id": candidate["trace_id"],
                        "run_identity": run_identity, "finished_at_utc": engine.utc_now(),
                        "eligible": result["eligible_under_adapted_rules"],
                        "core_validation_passed": result.get("core_validation_passed"),
                        "exclusion_reasons": result.get("exclusion_reasons", []), "evidence_rows": count})
            mirror(marker)
            new += 1
            completed = sum((args.output_dir / row["problem_id"] / "trace_completed.json").exists() for row in candidates)
            status.update(status="running", phase="trace_saved", progress={"completed_experiments": completed,
                          "pending_experiments": len(candidates) - completed, "new_attempts_this_session": new})
            print(f"TRACE CHECKPOINTED LOCALLY: {candidate['problem_id']} | {completed}/{len(candidates)} traces | "
                  f"{mirror.pending_count} file(s) queued for background Drive sync", flush=True)
        summaries, evidence = [], []
        for candidate in candidates:
            directory = args.output_dir / candidate["problem_id"]
            marker = directory / "trace_completed.json"
            if marker.exists():
                summaries.append(json.loads(marker.read_text(encoding="utf-8")))
                path = directory / "label_evidence.jsonl"
                if path.exists():
                    evidence.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        export = args.output_dir / "sentence_removal_evidence.jsonl"
        atomic_jsonl(export, evidence)
        mirror(export)
        summary = {"total_traces": len(candidates), "completed_traces": len(summaries),
                   "pending_traces": len(candidates) - len(summaries), "new_traces_this_session": new,
                   "screened_eligible": sum(row["eligible"] for row in summaries),
                   "validated_cores": sum(row["core_validation_passed"] is True for row in summaries),
                   "removal_evidence_rows": len(evidence),
                   "original_trace_diagnostic_rows": sum(row["evidence_type"] == "original_trace_diagnostic" for row in evidence),
                   "protocol": protocol(), "traces": summaries}
        atomic_json(args.output_dir / "summary.json", summary)
        mirror(args.output_dir / "summary.json")
        status.update(progress={"completed_experiments": len(summaries),
                      "pending_experiments": summary["pending_traces"], "new_attempts_this_session": new})
        status.close("completed" if summary["pending_traces"] == 0 else "paused")
        mirror.close()
        print(json.dumps({key: value for key, value in summary.items() if key != "traces"}, indent=2))
    except BaseException as exc:
        if status:
            try:
                status.close("stopped" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
                mirror.close()
            except BaseException as sync_error:
                print(f"Final Drive backup failed; local files retained: {sync_error}", file=sys.stderr)
        if isinstance(exc, KeyboardInterrupt):
            print("Stopped. Rerun the same command to restore completed trials and resume.", flush=True)
        else:
            raise
    finally:
        mirror.stop.set()
        signal.signal(signal.SIGTERM, original_sigterm)
        lockfile.close()


if __name__ == "__main__":
    main()
