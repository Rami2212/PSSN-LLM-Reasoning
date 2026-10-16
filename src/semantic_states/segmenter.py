"""Deterministic segmentation of validated reasoning traces into semantic states."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .schema import SemanticState
from .tokenizer_utils import count_tokens


class UnsegmentableTraceError(ValueError):
    """Raised when a trace violates the segmentation contract."""


_THINK_TAG_RE = re.compile(r"</?think\s*>", re.IGNORECASE)
_FINAL_MARKER_RE = re.compile(
    r"Final\s+answer\s*:|####\s*|<answer\s*>", re.IGNORECASE
)
_STEP_START_RE = re.compile(r"(?mi)^\s*(?:(?:step\s+)?\d+[.):]|[-*•])\s+")
_SENTENCE_END_RE = re.compile(r"[.!?]+[\"'”’)]*(?=\s|$)")
_CONTINUATION_START_RE = re.compile(
    r"^(?:and\b|because\b|for example\b|i\.e\.\b|in other words\b|"
    r"that means\b|this means\b|that gives\b|this gives\b|which\b|so\s+that\b)",
    re.IGNORECASE,
)
_ABBREVIATIONS = {
    "e.g.", "i.e.", "mr.", "mrs.", "ms.", "dr.", "vs.", "etc."
}


def _in_ranges(position: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in ranges)


def _find_think_ranges(
    trace: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Return marker, visible-text, and think-text ranges; reject malformed tags."""
    marker_ranges: list[tuple[int, int]] = []
    visible_ranges: list[tuple[int, int]] = []
    think_ranges: list[tuple[int, int]] = []
    open_start: int | None = None
    content_start = 0

    for match in _THINK_TAG_RE.finditer(trace):
        marker_ranges.append(match.span())
        if match.group(0).lower().startswith("</"):
            if open_start is None:
                raise UnsegmentableTraceError("</think> appears without a matching <think>")
            think_ranges.append((content_start, match.start()))
            visible_ranges.append((content_start, match.start()))
            open_start = None
            content_start = match.end()
        else:
            if open_start is not None:
                raise UnsegmentableTraceError("nested <think> blocks are not supported")
            if trace[content_start : match.start()].strip():
                visible_ranges.append((content_start, match.start()))
            open_start = match.start()
            content_start = match.end()

    if open_start is not None:
        raise UnsegmentableTraceError("<think> appears without a matching </think>")
    if trace[content_start:].strip():
        visible_ranges.append((content_start, len(trace)))
    return marker_ranges, visible_ranges, think_ranges


def _find_terminal_answer_start(
    trace: str,
    marker_ranges: list[tuple[int, int]],
    think_ranges: list[tuple[int, int]],
) -> int:
    candidates = [
        match.start()
        for match in _FINAL_MARKER_RE.finditer(trace)
        if not _in_ranges(match.start(), marker_ranges)
    ]
    outside_think = [
        position
        for position in candidates
        if not _in_ranges(position, think_ranges)
    ]
    if outside_think:
        # The earliest explicit marker outside the reasoning block begins the
        # terminal answer presentation (including any following answer tags).
        return min(outside_think)
    if candidates:
        # A marked answer wholly inside <think> cannot form one contiguous final
        # state if the closing structural tag must be excluded.
        raise UnsegmentableTraceError(
            "recognized answer marker occurs only inside <think>; review the trace"
        )
    raise UnsegmentableTraceError("no recognized terminal answer marker was found")


def _sentence_spans(text: str, absolute_start: int) -> list[tuple[int, int]]:
    """Return trimmed absolute sentence spans, avoiding common abbreviations."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        prefix = text[start : match.end()]
        last_word_match = re.search(r"(?:^|\s)([^\s]+)$", prefix)
        last_word = last_word_match.group(1).lower() if last_word_match else ""
        if last_word in _ABBREVIATIONS:
            continue
        left, right = start, match.end()
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((absolute_start + left, absolute_start + right))
        start = match.end()

    left, right = start, len(text)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if left < right:
        spans.append((absolute_start + left, absolute_start + right))
    return spans


def _is_explicit_step_block(text: str) -> bool:
    return bool(_STEP_START_RE.match(text))


def _meaningful_spans(
    trace: str,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    """Split a content region using explicit steps and deterministic sentence cues."""
    region = trace[start:end]
    if not region.strip():
        return []

    # Explicit numbered/bulleted steps are atomic candidate units, including
    # their wrapped continuation lines and equation lines.
    step_matches = list(_STEP_START_RE.finditer(region))
    if step_matches:
        blocks: list[tuple[int, int]] = []
        if region[: step_matches[0].start()].strip():
            blocks.append((0, step_matches[0].start()))
        for index, match in enumerate(step_matches):
            block_start = match.start()
            block_end = (
                step_matches[index + 1].start()
                if index + 1 < len(step_matches)
                else len(region)
            )
            blocks.append((block_start, block_end))
        output: list[tuple[int, int]] = []
        for block_start, block_end in blocks:
            block = region[block_start:block_end]
            if _is_explicit_step_block(block):
                left, right = block_start, block_end
                while left < right and region[left].isspace():
                    left += 1
                while right > left and region[right - 1].isspace():
                    right -= 1
                if left < right:
                    output.append((start + left, start + right))
            else:
                output.extend(_sentence_spans(block, start + block_start))
        return output

    sentence_spans = _sentence_spans(region, start)
    # A continuation sentence belongs to the prior operation. Likewise, an
    # unfinished fragment is attached to the previous state instead of being
    # emitted as an empty/stranded state.
    merged: list[tuple[int, int]] = []
    for span in sentence_spans:
        content = trace[span[0] : span[1]].strip()
        starts_as_continuation = bool(_CONTINUATION_START_RE.match(content))
        has_terminal_punctuation = bool(re.search(r"[.!?][\"'”’)]*$", content))
        if merged and (starts_as_continuation or not has_terminal_punctuation):
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    return merged


def _merge_short_leading_fragment(trace: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Attach a leading clause fragment to its following action when possible."""
    if len(spans) < 2:
        return spans
    first = trace[spans[0][0] : spans[0][1]].strip()
    if not re.search(r"[.!?][\"'”’)]*$", first):
        return [(spans[0][0], spans[1][1]), *spans[2:]]
    return spans


def segment_reasoning_trace(
    record: Mapping[str, Any],
    tokenizer: Any,
) -> list[SemanticState]:
    """Convert one validated, correct trace into ordered semantic states.

    The returned character offsets index the original ``reasoning_trace``.
    The only source spans deliberately omitted are structural ``<think>`` tags
    and whitespace at state edges. An explicit terminal answer is a single,
    separate final state.
    """
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    trace = record.get("reasoning_trace")
    if not isinstance(trace, str) or not trace.strip():
        raise UnsegmentableTraceError("reasoning_trace must be a non-empty string")
    if record.get("is_correct") is not True:
        raise UnsegmentableTraceError("only validated correct traces may be segmented")
    if record.get("evaluation_status", "scored") != "scored":
        raise UnsegmentableTraceError("trace evaluation_status must be 'scored'")
    if record.get("finish_reason", "eos") != "eos":
        raise UnsegmentableTraceError("trace must have completed naturally (finish_reason='eos')")

    problem_id = record.get("problem_id")
    run_id = record.get("run_id")
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise UnsegmentableTraceError("problem_id must be present")
    if not isinstance(run_id, str) or not run_id.strip():
        raise UnsegmentableTraceError("run_id must be present")
    trace_id = f"{run_id}:{problem_id}"

    think_markers, visible_ranges, think_ranges = _find_think_ranges(trace)
    answer_start = _find_terminal_answer_start(trace, think_markers, think_ranges)

    reasoning_spans: list[tuple[int, int]] = []
    for region_start, region_end in visible_ranges:
        if region_start >= answer_start:
            continue
        clipped_end = min(region_end, answer_start)
        reasoning_spans.extend(_meaningful_spans(trace, region_start, clipped_end))

    # If a closing think marker lies after the answer marker, there is no
    # contiguous source slice for a final answer state while excluding tags.
    final_region_end = len(trace)
    if any(start >= answer_start for start, _ in think_markers):
        raise UnsegmentableTraceError(
            "terminal answer intersects a structural <think> marker; review the trace"
        )
    final_start = answer_start
    while final_start < final_region_end and trace[final_start].isspace():
        final_start += 1
    while final_region_end > final_start and trace[final_region_end - 1].isspace():
        final_region_end -= 1
    if final_start >= final_region_end:
        raise UnsegmentableTraceError("terminal answer marker has no answer text")
    final_span = (final_start, final_region_end)

    reasoning_spans = _merge_short_leading_fragment(trace, reasoning_spans)
    spans = reasoning_spans + [final_span]
    states: list[SemanticState] = []
    for state_index, (start_char, end_char) in enumerate(spans):
        text = trace[start_char:end_char]
        if not text.strip():
            raise UnsegmentableTraceError("segmentation produced an empty semantic state")
        state = SemanticState(
            problem_id=problem_id,
            trace_id=trace_id,
            state_id=f"S{state_index + 1}",
            state_index=state_index,
            text=text,
            start_char=start_char,
            end_char=end_char,
            token_count=count_tokens(text, tokenizer),
            is_final_state=state_index == len(spans) - 1,
        )
        state.validate_trace_span(trace)
        states.append(state)

    if not states or not states[-1].is_final_state or any(
        state.is_final_state for state in states[:-1]
    ):
        raise UnsegmentableTraceError("segmentation must produce exactly one final state")
    return states
