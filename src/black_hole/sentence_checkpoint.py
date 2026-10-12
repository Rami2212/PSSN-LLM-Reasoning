"""Atomic, content-addressed evidence for resumable sentence experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class SentenceCheckpoint:
    def __init__(self, directory, persist=None):
        self.directory = Path(directory)
        self.persist = persist

    def load(self, kind, request):
        path = self.directory / "checkpoints" / kind / (identity(request) + ".json")
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["request"] != request or identity(record["result"]) != record["result_sha256"]:
            raise ValueError(f"Checkpoint identity/integrity mismatch: {path}")
        return record["result"]

    def save(self, kind, request, result):
        path = self.directory / "checkpoints" / kind / (identity(request) + ".json")
        atomic_json(path, {"request": request, "result": result,
                           "result_sha256": identity(result)})
        if self.persist:
            self.persist(path)


def collect_trials(checkpoint, prompt_key, count, seed, sample, snapshot, restore):
    """Resume the original seeded stream without regenerating completed samples."""
    seed()
    trials = []
    for index in range(count):
        request = {"prompt_sha256": prompt_key, "sample_index": index}
        saved = checkpoint.load("trials", request)
        if saved is not None:
            restore(saved["rng_after"])
            trials.append(saved["trial"])
            continue
        trial = sample(index)
        checkpoint.save("trials", request, {"trial": trial, "rng_after": snapshot()})
        trials.append(trial)
    return trials


def sentence_units(candidate, spans, cot):
    """Map half-open sentence spans back to the exact Week 2 trace offsets."""
    trace = candidate["reasoning_trace"]
    offset = trace.index("<think>") + len("<think>") if "<think>" in trace else 0
    if trace[offset:offset + len(cot)] != cot:
        raise ValueError("Thinking text does not match source offsets")
    states = candidate.get("source_states", [])
    for state in states:
        if trace[state["start_char"]:state["end_char"]] != state["text"]:
            raise ValueError(f"Source-state span mismatch: {state['state_id']}")
    units = []
    for i, (text, start, end) in enumerate(spans):
        absolute_start, absolute_end = start + offset, end + offset
        overlaps = []
        for state in states:
            left, right = max(absolute_start, state["start_char"]), min(absolute_end, state["end_char"])
            if left < right:
                overlaps.append({"state_id": state["state_id"], "overlap_start_char": left,
                                 "overlap_end_char": right, "is_final_state": state["is_final_state"]})
        units.append({"sentence_id": f"T{i + 1}", "text": text,
                      "start_char_in_cot": start, "end_char_in_cot": end,
                      "start_char_in_source": absolute_start, "end_char_in_source": absolute_end,
                      "source_state_ids": [item["state_id"] for item in overlaps],
                      "source_state_overlaps": overlaps})
    return units
