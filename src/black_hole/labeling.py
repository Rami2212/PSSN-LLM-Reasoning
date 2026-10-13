"""Week 3.004: label sentence-context removal evidence without a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "black-hole-sentence-label-v1"
LABEL_RULE_VERSION = "w3-004-sentence-context-v1"
TRAINING_EVIDENCE_TYPE = "gradient_greedy"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def classify_evaluation(evaluation, protocol):
    """Return label and reason for one fully scored removal evaluation."""
    required = ("correct_count", "complete_count", "incomplete_count", "attainability",
                "calibration_valid", "evaluation_key", "mask")
    missing = [field for field in required if field not in evaluation]
    if missing:
        return "unscored", "missing_evaluation_fields:" + ",".join(missing)
    if evaluation["calibration_valid"] is not True:
        return "unscored", "invalid_calibration"
    sample_count = protocol["n_resamples"]
    correct = evaluation["correct_count"]
    if not isinstance(correct, int) or correct < 0 or correct > sample_count:
        return "unscored", "invalid_correct_count"
    if evaluation["complete_count"] + evaluation["incomplete_count"] != sample_count:
        return "unscored", "invalid_completion_counts"
    if correct < protocol["minimum_correct"]:
        return "answer_sensitive_in_context", "fewer_than_two_correct_answers"
    if evaluation["attainability"] < protocol["attainability_threshold"]:
        return "plausibility_rejected", "attainability_below_threshold"
    return "low_value_in_context", "validated_removal"


def make_label_row(evidence, label, reason):
    evaluation = evidence["evaluation"]
    unit = evidence["unit"]
    row = {
        "schema_version": SCHEMA_VERSION,
        "label_rule_version": LABEL_RULE_VERSION,
        "label": label,
        "label_reason": reason,
        "problem_id": evidence["problem_id"],
        "trace_id": evidence["trace_id"],
        "question": evidence["question"],
        "reference_answer": evidence["reference_answer"],
        "sentence_id": unit["sentence_id"],
        "sentence_text": unit["text"],
        "sentence_start_char_in_cot": unit["start_char_in_cot"],
        "sentence_end_char_in_cot": unit["end_char_in_cot"],
        "sentence_start_char_in_source": unit["start_char_in_source"],
        "sentence_end_char_in_source": unit["end_char_in_source"],
        "source_state_ids": unit["source_state_ids"],
        "source_state_overlaps": unit["source_state_overlaps"],
        "context_before_mask": evidence["context_before_mask"],
        "removal_mask": evaluation["mask"],
        "correct_count": evaluation["correct_count"],
        "sample_count": evidence["protocol"]["n_resamples"],
        "complete_count": evaluation["complete_count"],
        "incomplete_count": evaluation["incomplete_count"],
        "attainability": evaluation["attainability"],
        "sufficiency": evaluation.get("sufficiency"),
        "passes_both": evaluation.get("passes_both"),
        "input_tokens": evaluation.get("input_tokens"),
        "mean_output_tokens": evaluation.get("mean_output_tokens"),
        "mean_latency_seconds": evaluation.get("mean_latency_seconds"),
        "peak_allocated_gib": evaluation.get("peak_allocated_gib"),
        "answers": evaluation.get("answers"),
        "evidence_type": evidence["evidence_type"],
        "evaluation_key": evaluation["evaluation_key"],
        "raw_trials_directory": evidence["raw_trials_directory"],
        "core_summary_path": evidence["core_summary_path"],
        "model_id": evidence.get("model_id"),
        "model_revision": evidence.get("model_revision"),
        "run_identity": evidence["run_identity"],
        "protocol": evidence["protocol"],
    }
    row["label_id"] = canonical_hash({
        "rule": LABEL_RULE_VERSION,
        "run": row["run_identity"],
        "evaluation": row["evaluation_key"],
    })
    return row


def label_evidence(input_dir):
    input_dir = Path(input_dir)
    summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("completed_traces") != summary.get("total_traces") or summary.get("pending_traces") != 0:
        raise ValueError("W3-003 input is incomplete; label only a completed run")
    protocol = summary["protocol"]
    for field, expected in (("n_resamples", 3), ("minimum_correct", 2),
                            ("attainability_threshold", 0.8)):
        if protocol.get(field) != expected:
            raise ValueError(f"Unexpected W3-003 protocol: {field}")

    trace_status = {row["trace_id"]: row for row in summary["traces"]}
    labels, rejected, unscored, audits, seen = [], [], [], [], set()
    for line_number, evidence in read_jsonl(input_dir / "sentence_removal_evidence.jsonl"):
        trace = trace_status.get(evidence.get("trace_id"))
        if trace is None:
            raise ValueError(f"Evidence line {line_number} references unknown trace")
        if evidence.get("problem_id") != trace.get("problem_id"):
            raise ValueError(f"Evidence line {line_number} has inconsistent problem ID")
        base_audit = {"schema_version": SCHEMA_VERSION, "source_line_number": line_number,
                      "problem_id": evidence["problem_id"], "trace_id": evidence["trace_id"],
                      "sentence_id": evidence["unit"]["sentence_id"],
                      "evidence_type": evidence["evidence_type"],
                      "evaluation_key": evidence["evaluation"].get("evaluation_key")}
        if evidence["evidence_type"] != TRAINING_EVIDENCE_TYPE:
            audits.append(base_audit | {"audit_reason": "diagnostic_or_random_evidence"})
            continue
        if trace.get("eligible") is not True:
            audits.append(base_audit | {"audit_reason": "screening_ineligible"})
            continue
        if trace.get("core_validation_passed") is not True:
            audits.append(base_audit | {"audit_reason": "final_core_not_validated"})
            continue
        label, reason = classify_evaluation(evidence["evaluation"], protocol)
        row = make_label_row(evidence, label, reason)
        if row["label_id"] in seen:
            raise ValueError(f"Duplicate label identity: {row['label_id']}")
        seen.add(row["label_id"])
        if label == "unscored":
            unscored.append(row)
        elif label == "plausibility_rejected":
            rejected.append(row)
        else:
            labels.append(row)
    labels.sort(key=lambda row: (row["problem_id"], row["sentence_id"], row["evaluation_key"]))
    rejected.sort(key=lambda row: (row["problem_id"], row["sentence_id"], row["evaluation_key"]))
    unscored.sort(key=lambda row: (row["problem_id"], row["sentence_id"], row["evaluation_key"]))
    audits.sort(key=lambda row: (row["problem_id"], row["sentence_id"], row["evaluation_key"] or ""))
    return summary, labels, rejected, unscored, audits


def write_outputs(input_dir, output_dir):
    summary, labels, rejected, unscored, audits = label_evidence(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output_dir / "black_hole_sentence_labels.jsonl", labels)
    atomic_jsonl(output_dir / "plausibility_rejected_label_evidence.jsonl", rejected)
    atomic_jsonl(output_dir / "unscored_label_evidence.jsonl", unscored)
    atomic_jsonl(output_dir / "labeling_audit.jsonl", audits)
    label_counts = Counter(row["label"] for row in labels)
    report = {
        "schema_version": SCHEMA_VERSION,
        "label_rule_version": LABEL_RULE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(Path(input_dir).resolve()),
        "input_summary_sha256": hashlib.sha256((Path(input_dir) / "summary.json").read_bytes()).hexdigest(),
        "input_evidence_sha256": hashlib.sha256(
            (Path(input_dir) / "sentence_removal_evidence.jsonl").read_bytes()).hexdigest(),
        "source_run": {key: summary[key] for key in ("total_traces", "completed_traces", "screened_eligible",
                                                        "validated_cores", "removal_evidence_rows", "protocol")},
        "training_evidence_type": TRAINING_EVIDENCE_TYPE,
        "training_label_rows": len(labels),
        "training_label_counts": dict(sorted(label_counts.items())),
        "plausibility_rejected_rows": len(rejected),
        "unscored_rows": len(unscored),
        "audit_rows": len(audits),
        "interpretation": (
            "Labels describe sentence removal in the recorded context. They do not prove removal "
            "of hidden model computation or end-to-end inference speedup."),
        "outputs": {"labels": "black_hole_sentence_labels.jsonl",
                    "rejected": "plausibility_rejected_label_evidence.jsonl",
                    "unscored": "unscored_label_evidence.jsonl", "audit": "labeling_audit.jsonl"},
    }
    atomic_json(output_dir / "labeling_summary.json", report)
    atomic_json(output_dir / "labeling_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "label_rule_version": LABEL_RULE_VERSION,
        "input_summary_sha256": report["input_summary_sha256"],
        "input_evidence_sha256": report["input_evidence_sha256"],
        "output_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(output_dir.glob("*.json*"))
                          if path.name != "labeling_manifest.json"},
    })
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = write_outputs(args.input_dir, args.output_dir)
    print(json.dumps({key: report[key] for key in ("training_label_rows", "training_label_counts",
                                                    "plausibility_rejected_rows", "unscored_rows", "audit_rows")}, indent=2))


if __name__ == "__main__":
    main()
