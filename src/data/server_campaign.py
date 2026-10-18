"""Run explicit local data-preparation stages for a larger Black Hole dataset.

See docs/mi300x_500_question_runbook.md. No stage uploads files to Drive.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import uuid

from src.data.black_hole_dataset import atomic_json, atomic_jsonl, read_jsonl

MODEL_ID = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_snapshot():
    return {str(path.relative_to(Path("src"))): digest(path)
            for path in sorted(Path("src").rglob("*.py"))}


def provenance():
    from src.utils.environment import collect_environment_info
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": collect_environment_info(),
        "repository_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                             capture_output=True, text=True).stdout.strip(),
        "code_sha256": source_snapshot(),
        "packages": {name: version(name) for name in
                     ("torch", "transformers", "datasets", "accelerate", "numpy")},
        "pip_freeze": subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                     check=True, capture_output=True, text=True).stdout,
    }


def load_rows(path):
    return list(read_jsonl(path))


def campaign_configuration(args):
    path = args.campaign / "campaign_configuration.json"
    if path.exists():
        saved = read_json(path)
        if saved["revision"] != args.revision:
            raise ValueError("Campaign revision changed; use the saved revision")
        return saved
    config = {"model_id": MODEL_ID, "revision": args.revision, "seed": 44,
              "max_new_tokens": 1024, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(path, config)
    return config


def freeze_batch(args, config, batch):
    """Persist the sample before model work; exclude old attempts and reserved batches."""
    directory = batch / "baseline"
    sample = directory / "sample.jsonl"
    if sample.exists():
        saved = read_json(directory / "sample_manifest.json")
        if saved["sample_sha256"] != digest(sample) or saved["attempts"] != args.attempts:
            raise ValueError("Saved batch sample differs; keep --attempts unchanged on resume")
        return load_rows(sample), saved
    from datasets import DatasetDict, load_from_disk
    from src.data.gsm8k import load_gsm8k
    source = args.campaign / "source_dataset"
    if not source.exists():
        DatasetDict(load_gsm8k(seed=config["seed"])).save_to_disk(str(source))
    train = load_from_disk(str(source))["train"]
    excluded, sources = set(), {}
    for path in sorted(args.exclude_root.rglob("raw_attempts.jsonl")):
        sources[str(path)] = digest(path)
        excluded.update(row["problem_id"] for row in read_jsonl(path))
    # A copied pilot ZIP can omit its old raw-attempt export. Its accepted
    # dataset still provides authoritative IDs that must not be regenerated.
    for path in sorted(args.exclude_root.rglob("black_hole_training_dataset.jsonl")):
        sources[str(path)] = digest(path)
        excluded.update(row["problem_id"] for row in read_jsonl(path))
    for path in sorted(args.exclude_root.rglob("semantic_states.jsonl")):
        sources[str(path)] = digest(path)
        excluded.update(row["problem_id"] for row in read_jsonl(path))
    for path in sorted(args.campaign.glob("batches/*/baseline/sample.jsonl")):
        sources[str(path)] = digest(path)
        excluded.update(row["id"] for row in read_jsonl(path))
    available = [i for i, row in enumerate(train) if row["id"] not in excluded]
    if len(available) < args.attempts:
        raise ValueError(f"Only {len(available)} unused GSM8K training questions remain")
    indices = sorted(random.Random(config["seed"]).sample(available, args.attempts))
    examples = list(train.select(indices))
    atomic_jsonl(sample, examples)
    manifest = {"attempts": args.attempts, "run_id": str(uuid.uuid4()),
                "seed": config["seed"], "sample_sha256": digest(sample),
                "source_train_fingerprint": train._fingerprint,
                "selected_example_ids": [row["id"] for row in examples],
                "excluded_ids": sorted(excluded), "excluded_sources_sha256": sources}
    atomic_json(directory / "sample_manifest.json", manifest)
    return examples, manifest


def baseline(args, config, batch):
    from src.baseline.cot import BASELINE_PROMPT_TEMPLATE, generate_cot_record
    from src.data.trace_validation import summarize_trace_validation, validate_traces
    from src.evaluation.accuracy import evaluate_record
    from src.models.qwen import GenerationSettings, QwenGenerator
    from src.utils.environment import load_qwen_model
    from src.utils.run_status import RunStatus

    examples, sample = freeze_batch(args, config, batch)
    directory = batch / "baseline"
    settings = GenerationSettings(max_new_tokens=config["max_new_tokens"], seed=config["seed"])
    metadata = provenance()
    execution = {"campaign": config, "sample_sha256": sample["sample_sha256"],
                 "code_sha256": metadata["code_sha256"], "packages": metadata["packages"],
                 "generation": settings.to_dict(), "prompt_template": BASELINE_PROMPT_TEMPLATE}
    identity_file = directory / "execution_identity.json"
    if identity_file.exists() and read_json(identity_file) != execution:
        raise ValueError("Batch source code, configuration or software changed; use a new batch")
    atomic_json(identity_file, execution)
    atomic_json(directory / "sessions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"), metadata)
    status = RunStatus(directory / "run_status.json", run_id=sample["run_id"], total_experiments=len(examples))
    status.start(phase="loading_model", metadata={**metadata, **execution})
    final_status = "failed"
    try:
        pending = [row for row in examples if not (directory / "records" / (row["id"] + ".json")).exists()]
        if pending:
            tokenizer, model, precision = load_qwen_model(MODEL_ID, revision=config["revision"], require_cuda=True)
            generator = QwenGenerator(tokenizer, model, settings=settings, model_id=MODEL_ID, precision=precision)
            for example in pending:
                status.update(status="running", phase="generating", current_experiment={"problem_id": example["id"]})
                record = generate_cot_record(example, generator)
                record.update(evaluate_record(record))
                record.update(run_id=sample["run_id"], experiment_name="black-hole-500-expanded-baseline",
                              experiment_timestamp_utc=config["created_at_utc"],
                              run_metadata={"seed": config["seed"], "model_revision": config["revision"]})
                record["model_metadata"].update(model_revision=config["revision"], tokenizer_revision=config["revision"])
                atomic_json(directory / "records" / (example["id"] + ".json"), record)
                completed = len(list((directory / "records").glob("*.json")))
                status.update(phase="saving_checkpoint", progress={"completed_experiments": completed,
                              "pending_experiments": len(examples) - completed})
                print(f"BASELINE {completed}/{len(examples)}: {example['id']} | correct={record['is_correct']}", flush=True)
        final_status = "completed"
    except KeyboardInterrupt:
        final_status = "stopped"
        raise
    finally:
        rows = [read_json(directory / "records" / (row["id"] + ".json")) for row in examples
                if (directory / "records" / (row["id"] + ".json")).exists()]
        atomic_jsonl(directory / "raw_attempts.jsonl", rows)
        checks = validate_traces(rows)
        atomic_jsonl(directory / "trace_validation.jsonl", checks)
        atomic_jsonl(directory / "segmentation_ready_traces.jsonl",
                     [row for row, check in zip(rows, checks) if check["segmentation_eligible"]])
        atomic_json(directory / "trace_validation_summary.json", summarize_trace_validation(rows, checks, target_valid_traces=500))
        status.update(progress={"completed_experiments": len(rows), "pending_experiments": len(examples) - len(rows)})
        status.close(final_status)
    print(json.dumps(read_json(directory / "trace_validation_summary.json"), indent=2))


def segment(args, config, batch):
    from src.semantic_states import UnsegmentableTraceError, segment_reasoning_trace
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=config["revision"])
    states, review = [], []
    for row in read_jsonl(batch / "baseline/segmentation_ready_traces.jsonl"):
        try:
            states.extend(state.to_dict() for state in segment_reasoning_trace(row, tokenizer))
        except UnsegmentableTraceError as exc:
            review.append({"problem_id": row["problem_id"], "error": str(exc)})
    atomic_jsonl(batch / "segmentation/semantic_states.jsonl", states)
    atomic_jsonl(batch / "segmentation/review.jsonl", review)
    if not states:
        raise ValueError("No segmentable traces in this batch; preserve it and add another batch")


def validate(args, config, batch):
    from src.semantic_states.validation import validate_semantic_state_records
    rows = load_rows(batch / "baseline/segmentation_ready_traces.jsonl")
    sources = {f"{row['run_id']}:{row['problem_id']}": row["reasoning_trace"] for row in rows}
    result = validate_semantic_state_records(load_rows(batch / "segmentation/semantic_states.jsonl"), source_traces=sources)
    atomic_jsonl(batch / "validation/week3_ready_states.jsonl", result["week3_ready_states"])
    atomic_jsonl(batch / "validation/trace_validation_reports.jsonl", result["trace_reports"])
    atomic_json(batch / "validation/validation_summary.json", {k: v for k, v in result.items()
                                                              if k not in ("week3_ready_states", "trace_reports")})
    if not result["week3_ready_states"]:
        raise ValueError("No validated states in this batch; add another batch")


def represent(args, config, batch):
    import numpy as np
    from src.semantic_states.representations import extract_semantic_state_representations
    from src.utils.environment import load_qwen_model
    tokenizer, model, precision = load_qwen_model(MODEL_ID, revision=config["revision"], require_cuda=True)
    groups = defaultdict(list)
    for row in read_jsonl(batch / "validation/week3_ready_states.jsonl"):
        groups[row["trace_id"]].append(row)
    sources = {f"{row['run_id']}:{row['problem_id']}": row
               for row in read_jsonl(batch / "baseline/segmentation_ready_traces.jsonl")}
    chunks, index, configuration = [], [], None
    for number, (trace_id, states) in enumerate(groups.items(), 1):
        source = sources[trace_id]
        result = extract_semantic_state_representations(model, tokenizer, prompt=source["prompt"],
                    reasoning_trace=source["reasoning_trace"], states=states, model_id=MODEL_ID)
        offset = len(index)
        for item in result.index_records:
            index.append({**item, "vector_index": offset + item["vector_index"]})
        chunks.append(result.vectors)
        configuration = result.configuration
        print(f"STATE VECTORS {number}/{len(groups)}", flush=True)
    vectors = np.concatenate(chunks).astype(np.float32)
    directory = batch / "representations"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "state_vectors.npy.tmp").open("wb") as stream:
        np.save(stream, vectors, allow_pickle=False)
    os.replace(directory / "state_vectors.npy.tmp", directory / "state_vectors.npy")
    atomic_jsonl(directory / "representation_index.jsonl", index)
    atomic_json(directory / "representation_config.json", {**configuration, "vector_shape": list(vectors.shape),
                "model_revision": config["revision"], "tokenizer_revision": config["revision"],
                "precision": precision, "runtime": provenance()})


def state_dataset(args, config, batch):
    from src.data.semantic_state_dataset import build_semantic_state_dataset, save_semantic_state_dataset
    files = {"baseline_traces": batch / "baseline/segmentation_ready_traces.jsonl",
             "week3_ready_states": batch / "validation/week3_ready_states.jsonl",
             "state_validation_reports": batch / "validation/trace_validation_reports.jsonl",
             "representation_index": batch / "representations/representation_index.jsonl",
             "state_vectors": batch / "representations/state_vectors.npy"}
    rows, metadata, summary = build_semantic_state_dataset(
        load_rows(files["baseline_traces"]), load_rows(files["week3_ready_states"]),
        load_rows(files["state_validation_reports"]), load_rows(files["representation_index"]),
        representation_config=read_json(batch / "representations/representation_config.json"),
        vector_file=files["state_vectors"], random_seed=config["seed"],
        validation_thresholds=read_json(batch / "validation/validation_summary.json").get("thresholds", {}),
        source_files={key: {"path": str(path), "sha256": digest(path)} for key, path in files.items()})
    save_semantic_state_dataset(batch / "state-dataset", rows, metadata, summary)


def state_pool_sources(args):
    paths = [path.resolve() for path in args.existing_states]
    paths += sorted(args.campaign.glob("batches/*/state-dataset/semantic_states.jsonl"))
    return list(dict.fromkeys(paths))


def merge_states(args, config, batch):
    """Snapshot the combined state pool; keep new-only data separate for labeling."""
    if not args.existing_states:
        raise ValueError("Supply --existing-states with the original Week 2 dataset")
    traces, sources = {}, []
    for path in state_pool_sources(args):
        sources.append({"path": str(path), "sha256": digest(path)})
        for row in read_jsonl(path):
            problem_id = row["problem_id"]
            if not isinstance(row.get("states"), list) or not row["states"]:
                raise ValueError(f"Expected a trace-level Week 2 dataset: {path}")
            if problem_id in traces and traces[problem_id] != row:
                raise ValueError(f"Conflicting state traces for question {problem_id}")
            traces[problem_id] = row
    rows = json.loads(json.dumps([traces[key] for key in sorted(traces)]))
    for row in rows:
        for state in row["states"]:
            ref = state["representation"]
            original_path = ref["vector_file"]
            ref["vector_file"] = str(resolve_vector_file(original_path, args.state_root))
            if ref["vector_file"] != original_path:
                ref["original_vector_file"] = original_path
    directory = batch / "combined-states"
    atomic_jsonl(directory / "semantic_states.jsonl", rows)
    atomic_json(directory / "summary.json", {
        "question_count": len(rows), "state_count": sum(len(row["states"]) for row in rows),
        "new_batch_question_count": len(load_rows(batch / "state-dataset/semantic_states.jsonl")),
        "sources": sources, "new_only_dataset": str(batch / "state-dataset/semantic_states.jsonl"),
        "vector_policy": "Each state retains its own vector_file and vector_index; no cross-file index mixing",
        "labeling_policy": "Use only the new batch dataset for sentence experiments; previous traces are not relabeled"})


def select(args, config, batch):
    from src.black_hole.trace_selection import save_trace_selection, select_eligible_traces
    selected, audit, summary = select_eligible_traces(load_rows(batch / "state-dataset/semantic_states.jsonl"),
                                                    random_seed=42, sample_size=None)
    save_trace_selection(batch / "selection", selected, audit, summary)
    print(json.dumps(summary, indent=2))


def sentences(args, config, batch):
    from src.black_hole.run_sentence_experiments import prepare_candidates
    from src.black_hole.sentence_experiment import split_sentences
    from src.black_hole.sentence_checkpoint import sentence_units
    rows = []
    for candidate in prepare_candidates(batch / "selection/eligible_traces.jsonl"):
        cot = candidate["reasoning_trace"].split("</think>", 1)[0].split("<think>", 1)[-1]
        for unit in sentence_units(candidate, split_sentences(cot), cot):
            rows.append({"problem_id": candidate["problem_id"], "trace_id": candidate["trace_id"], **unit})
    atomic_jsonl(batch / "sentences/new_sentences.jsonl", rows)
    atomic_json(batch / "sentences/summary.json", {"new_question_count": len({row["problem_id"] for row in rows}),
                "sentence_count": len(rows), "labels_assigned": False,
                "split_method": "Same code and character offsets as the sentence experiment engine"})


STAGES = {"baseline": baseline, "segment": segment, "validate": validate,
          "represent": represent, "state-dataset": state_dataset, "merge-states": merge_states,
          "select": select, "sentences": sentences}
OUTPUTS = {"baseline": "baseline/raw_attempts.jsonl", "segment": "segmentation/semantic_states.jsonl",
           "validate": "validation/week3_ready_states.jsonl", "represent": "representations/state_vectors.npy",
           "state-dataset": "state-dataset/semantic_states.jsonl",
           "merge-states": "combined-states/semantic_states.jsonl",
           "select": "selection/eligible_traces.jsonl", "sentences": "sentences/new_sentences.jsonl"}


def campaign_sources(args):
    paths = sorted(args.campaign.glob("batches/*/dataset/black_hole_training_dataset.jsonl"))
    paths += [path.resolve() for path in args.include_existing]
    sources, groups, row_ids = [], {}, set()
    for path in dict.fromkeys(paths):
        evidence = path.parent.parent / "experiments"
        if not evidence.is_dir():
            evidence = path.parent.parent
        labels = path.parent.parent / "labels/black_hole_sentence_labels.jsonl"
        if not labels.is_file():
            labels = path.parent.parent / "w3-004-labels/black_hole_sentence_labels.jsonl"
        source = {"dataset": str(path), "sha256": digest(path), "evidence_dir": str(evidence),
                  "labels": str(labels), "labels_sha256": digest(labels) if labels.is_file() else None}
        sources.append(source)
        local = defaultdict(list)
        for row in read_jsonl(path):
            if row["dataset_row_id"] in row_ids:
                raise ValueError(f"Duplicate dataset row: {row['dataset_row_id']}")
            row_ids.add(row["dataset_row_id"])
            if row["schema_version"] != "black-hole-training-dataset-v1":
                raise ValueError("Unexpected source dataset schema")
            if row["raw_evidence"]["model_id"] != MODEL_ID:
                raise ValueError("Cannot merge different source models")
            if row["raw_evidence"]["model_revision"] != args.revision:
                raise ValueError("Cannot merge different model revisions")
            if row["raw_evidence"]["protocol_name"] != "pssn-sentence-removal-3samples-v1":
                raise ValueError("Cannot merge different experiment protocols")
            local[row["problem_id"]].append(row)
        for problem_id, rows in local.items():
            if problem_id in groups:
                raise ValueError(f"Question appears in multiple source runs: {problem_id}")
            groups[problem_id] = (rows, source)
    return groups, sources


def resolve_vector_file(value, state_root):
    """Relocate original Colab/Windows PSSN paths into the copied server PSSN2."""
    path = Path(value)
    if path.is_file():
        return path.resolve()
    normalized = str(value).replace("\\", "/")
    for marker in ("PSSN2/", "PSSN/"):
        if marker in normalized:
            candidate = state_root / normalized.split(marker, 1)[1]
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"Missing Week 2 vector file: {value}; copy it into {state_root}")


def finalize(args):
    import numpy as np
    groups, sources = campaign_sources(args)
    if len(groups) < args.target:
        raise ValueError(f"Only {len(groups)}/{args.target} usable questions; finish another batch first")
    destination = args.campaign / "final"
    previous_datasets = {str(path.resolve()) for path in args.include_existing}
    previous_ids = sorted(key for key, (_, source) in groups.items()
                          if str(Path(source["dataset"]).resolve()) in previous_datasets)
    if len(previous_ids) > args.target:
        raise ValueError("Existing labeled questions exceed the target; choose a larger target")
    new_ids = sorted(set(groups) - set(previous_ids))
    selected_ids = sorted(previous_ids + random.Random(44).sample(new_ids, args.target - len(previous_ids)))
    if destination.exists():
        raise ValueError("Final folder already exists; preserve the frozen dataset and use a new campaign for changes")
    # Resolve all evidence and vector inputs before creating the final directory.
    required_names = ("manifest.json", "embedded_question_and_trace.json", "label_evidence.jsonl")
    arrays, vector_map, vector_rows, rows, evidence_sources = {}, {}, [], [], {}
    label_indices, combined_labels = {}, []
    for problem_id in selected_ids:
        originals, source = groups[problem_id]
        evidence = Path(source["evidence_dir"]) / problem_id
        for name in required_names:
            if not (evidence / name).is_file():
                raise FileNotFoundError(evidence / name)
        evidence_sources[problem_id] = str(evidence)
        label_path = Path(source["labels"])
        if label_path not in label_indices:
            label_index = {}
            for label in read_jsonl(label_path):
                if label["label_id"] in label_index:
                    raise ValueError(f"Duplicate label ID in {label_path}")
                label_index[label["label_id"]] = label
            label_indices[label_path] = label_index
        for original in originals:
            label = label_indices[label_path].get(original.get("source_label_id", original["dataset_row_id"]))
            if label is None or any(label[key] != original[key] for key in ("problem_id", "trace_id", "label")):
                raise ValueError(f"Original accepted label missing or inconsistent: {original['dataset_row_id']}")
            combined_labels.append(label)
            row = json.loads(json.dumps(original))
            for ref in row["source_state_references"]:
                representation = ref["representation"]
                path = resolve_vector_file(representation["vector_file"], args.state_root)
                if path not in arrays:
                    arrays[path] = np.load(path, mmap_mode="r", allow_pickle=False)
                    if arrays[path].ndim != 2:
                        raise ValueError(f"Expected a 2D vector matrix: {path}")
                old_index = representation["vector_index"]
                if not isinstance(old_index, int) or old_index < 0 or old_index >= len(arrays[path]):
                    raise ValueError(f"Invalid source vector index: {path}:{old_index}")
                key = (path, old_index)
                if key not in vector_map:
                    vector_map[key] = len(vector_rows)
                    vector_rows.append(np.asarray(arrays[path][old_index], dtype=np.float32))
                representation["original_vector_file"] = representation["vector_file"]
                representation["original_vector_index"] = old_index
                representation["vector_file"] = str(destination / "state_vectors.npy")
                representation["vector_index"] = vector_map[key]
            rows.append(row)
    vectors = np.stack(vector_rows)
    destination.mkdir(parents=True)
    for problem_id, evidence in evidence_sources.items():
        for name in required_names:
            target = destination / "evidence" / problem_id / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(evidence) / name, target)
    np.save(destination / "state_vectors.npy", vectors, allow_pickle=False)
    atomic_jsonl(destination / "black_hole_training_dataset.jsonl", rows)
    atomic_jsonl(destination / "black_hole_sentence_labels.jsonl", combined_labels)
    summary = {"question_count": len(selected_ids), "training_row_count": len(rows),
               "available_question_count": len(groups), "selected_problem_ids": selected_ids,
               "selection_seed": 44, "label_counts": dict(Counter(row["label"] for row in rows)),
               "existing_questions_preserved": len(previous_ids),
               "sources": sources, "evidence_sources": evidence_sources,
               "source_vector_sha256": {str(path): digest(path) for path in arrays},
               "dataset_sha256": digest(destination / "black_hole_training_dataset.jsonl"),
               "labels_sha256": digest(destination / "black_hole_sentence_labels.jsonl"),
               "state_vectors_sha256": digest(destination / "state_vectors.npy"),
               "state_vector_shape": list(vectors.shape),
               "full_evidence_policy": "Full trials remain in source run folders; final/evidence contains activation inputs only"}
    atomic_json(destination / "dataset_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("question_count", "training_row_count", "label_counts")}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "count", "finalize"])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--batch", default="batch-001")
    parser.add_argument("--attempts", type=int, default=500, help="Raw questions per batch, not accepted questions")
    parser.add_argument("--exclude-root", type=Path, default=Path("/root/pssn-work/PSSN2"))
    parser.add_argument("--include-existing", type=Path, action="append", default=[], help="Existing W3-005 dataset JSONL")
    parser.add_argument("--existing-states", type=Path, action="append", default=[], help="Original trace-level Week 2 semantic_states.jsonl for merge-states")
    parser.add_argument("--state-root", type=Path, default=Path("/root/pssn-work/PSSN2"))
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args(argv)
    if args.attempts < 1 or args.target < 1:
        parser.error("--attempts and --target must be positive")
    if not args.batch.startswith("batch-") or not args.batch[6:].isdigit():
        parser.error("Use a batch name such as batch-001")
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        parser.error("--revision must be a pinned lowercase commit hash")
    args.campaign = args.campaign.resolve()
    args.campaign.mkdir(parents=True, exist_ok=True)
    config = campaign_configuration(args)
    # All commands in the runbook execute on Linux; avoid overlapping prep commands.
    import fcntl
    with (args.campaign / ".prep.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "count":
            groups, _ = campaign_sources(args)
            print(json.dumps({"usable_questions": len(groups), "target_questions": args.target,
                              "remaining_questions": max(0, args.target - len(groups)),
                              "label_rows": sum(len(rows) for rows, _ in groups.values())}, indent=2))
        elif args.stage == "finalize":
            finalize(args)
        else:
            batch = args.campaign / "batches" / args.batch
            stage_index = list(STAGES).index(args.stage)
            if stage_index:
                previous = list(STAGES)[stage_index - 1]
                marker = batch / (previous + "_completed.json")
                if not marker.is_file():
                    raise ValueError(f"Finish {previous} first")
            dependencies = {str(path.relative_to(batch)): digest(path) for key, relative in OUTPUTS.items()
                            if list(STAGES).index(key) < stage_index
                            for path in [batch / relative]}
            identity = {"configuration": config, "inputs_sha256": dependencies, "code_sha256": source_snapshot()}
            if args.stage == "baseline":
                identity["attempts"] = args.attempts
            if args.stage == "merge-states":
                identity["state_sources_sha256"] = {str(path): digest(path) for path in state_pool_sources(args)}
            marker = batch / (args.stage + "_completed.json")
            if marker.exists():
                saved = read_json(marker)
                if saved["identity"] != identity or saved["output_sha256"] != digest(batch / OUTPUTS[args.stage]):
                    raise ValueError("Completed stage changed; preserve this batch and investigate")
                print(f"Already completed: {args.stage} ({batch})")
                return
            if args.stage in ("state-dataset", "select"):
                # Writers intentionally require a fresh directory. Preserve any
                # interrupted output before replaying this inexpensive stage.
                directory = batch / ("state-dataset" if args.stage == "state-dataset" else "selection")
                if directory.exists():
                    directory.rename(directory.with_name(directory.name + ".interrupted-" +
                                     datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
            STAGES[args.stage](args, config, batch)
            atomic_json(marker, {"identity": identity, "output_sha256": digest(batch / OUTPUTS[args.stage]),
                                 "finished_at_utc": datetime.now(timezone.utc).isoformat()})
            print(f"COMPLETED: {args.stage} ({batch})", flush=True)


if __name__ == "__main__":
    main()
