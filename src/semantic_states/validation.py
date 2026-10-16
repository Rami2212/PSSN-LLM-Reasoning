"""Non-mutating quality checks for segmented semantic-state sequences."""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any

from .schema import SemanticState


def _state_record(state: Mapping[str, Any] | SemanticState) -> dict[str, Any]:
    if isinstance(state, SemanticState):
        return state.to_dict()
    if isinstance(state, Mapping):
        return dict(state)
    raise TypeError("each semantic state must be a mapping or SemanticState")


def _add_issue(
    issues: list[dict[str, Any]],
    code: str,
    message: str,
    *,
    severity: str = "error",
    state_id: Any = None,
    state_position: int | None = None,
) -> None:
    issue: dict[str, Any] = {"code": code, "severity": severity, "message": message}
    if state_id is not None:
        issue["state_id"] = state_id
    if state_position is not None:
        issue["state_position"] = state_position
    issues.append(issue)


def validate_state_sequence(
    states: Iterable[Mapping[str, Any] | SemanticState],
    *,
    source_trace: str | None = None,
    min_state_tokens: int = 3,
    min_usable_states: int = 2,
    max_state_tokens: int | None = 512,
) -> dict[str, Any]:
    """Validate one ordered trace's states without changing any input records.

    States shorter than ``min_state_tokens`` are errors and are excluded from
    the usable-state count. States longer than ``max_state_tokens`` generate a
    non-blocking manual-review warning. A trace is Week-3-ready only when it
    has no errors and at least ``min_usable_states`` usable states.
    """
    if isinstance(min_state_tokens, bool) or not isinstance(min_state_tokens, int) or min_state_tokens < 1:
        raise ValueError("min_state_tokens must be a positive integer")
    if isinstance(min_usable_states, bool) or not isinstance(min_usable_states, int) or min_usable_states < 1:
        raise ValueError("min_usable_states must be a positive integer")
    if max_state_tokens is not None and (
        isinstance(max_state_tokens, bool)
        or not isinstance(max_state_tokens, int)
        or max_state_tokens < min_state_tokens
    ):
        raise ValueError("max_state_tokens must be None or an integer >= min_state_tokens")
    if source_trace is not None and not isinstance(source_trace, str):
        raise TypeError("source_trace must be a string or None")

    records = [_state_record(state) for state in states]
    trace_ids = [record.get("trace_id") for record in records if record.get("trace_id")]
    trace_id = trace_ids[0] if trace_ids else None
    problem_ids = [record.get("problem_id") for record in records if record.get("problem_id")]
    problem_id = problem_ids[0] if problem_ids else None
    issues: list[dict[str, Any]] = []
    usable_count = 0

    if not records:
        _add_issue(issues, "empty_sequence", "Trace has no semantic states.")
    if trace_id is None:
        _add_issue(issues, "missing_trace_id", "Trace sequence has no trace_id.")

    previous_index: int | None = None
    previous_end: int | None = None
    previous_text: str | None = None
    final_flags: list[bool] = []

    for position, record in enumerate(records):
        state_id = record.get("state_id")
        state_usable = True

        final_flag = record.get("is_final_state")
        if not isinstance(final_flag, bool):
            _add_issue(
                issues,
                "missing_or_invalid_final_state_flag",
                "is_final_state must be a boolean.",
                state_id=state_id,
                state_position=position,
            )
            final_flags.append(False)
            state_usable = False
        else:
            final_flags.append(final_flag)

        if not isinstance(state_id, str) or not state_id.strip():
            _add_issue(issues, "missing_state_id", "State identifier is missing or empty.", state_position=position)
            state_usable = False
        elif state_id != f"S{position + 1}":
            _add_issue(
                issues,
                "state_id_mismatch",
                f"Expected state_id S{position + 1}, received {state_id!r}.",
                state_id=state_id,
                state_position=position,
            )
            state_usable = False

        state_index = record.get("state_index")
        if isinstance(state_index, bool) or not isinstance(state_index, int):
            _add_issue(
                issues,
                "missing_state_index",
                "state_index must be an integer.",
                state_id=state_id,
                state_position=position,
            )
            state_usable = False
        else:
            if state_index != position or (previous_index is not None and state_index <= previous_index):
                _add_issue(
                    issues,
                    "invalid_state_order",
                    f"Expected state_index {position}, received {state_index}.",
                    state_id=state_id,
                    state_position=position,
                )
                state_usable = False
            previous_index = state_index

        current_trace_id = record.get("trace_id")
        if not isinstance(current_trace_id, str) or not current_trace_id.strip():
            _add_issue(
                issues,
                "missing_trace_id",
                "State is missing trace_id.",
                state_id=state_id,
                state_position=position,
            )
            state_usable = False
        elif trace_id is not None and current_trace_id != trace_id:
            _add_issue(
                issues,
                "inconsistent_trace_id",
                "States in one sequence must share the same trace_id.",
                state_id=state_id,
                state_position=position,
            )
            state_usable = False

        current_problem_id = record.get("problem_id")
        if not isinstance(current_problem_id, str) or not current_problem_id.strip():
            _add_issue(
                issues,
                "missing_problem_id",
                "State is missing problem_id.",
                state_id=state_id,
                state_position=position,
            )
            state_usable = False
        elif problem_id is not None and current_problem_id != problem_id:
            _add_issue(
                issues,
                "inconsistent_problem_id",
                "States in one sequence must share the same problem_id.",
                state_id=state_id,
                state_position=position,
            )
            state_usable = False

        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            _add_issue(
                issues,
                "empty_state",
                "State text must not be empty or whitespace-only.",
                state_id=state_id,
                state_position=position,
            )
            state_usable = False
            normalized_text = None
        else:
            normalized_text = re.sub(r"\s+", " ", text).strip().casefold()
            if previous_text is not None and normalized_text == previous_text:
                _add_issue(
                    issues,
                    "duplicate_adjacent_state",
                    "State duplicates the immediately preceding state after whitespace/case normalization.",
                    state_id=state_id,
                    state_position=position,
                )
                state_usable = False
        previous_text = normalized_text

        token_count = record.get("token_count")
        if "token_count" not in record or token_count is None:
            _add_issue(
                issues,
                "missing_token_count",
                "State is missing token_count.",
                state_id=state_id,
                state_position=position,
            )
            state_usable = False
        elif isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            _add_issue(
                issues,
                "invalid_token_count",
                "token_count must be a non-negative integer.",
                state_id=state_id,
                state_position=position,
            )
            state_usable = False
        elif token_count < min_state_tokens:
            _add_issue(
                issues,
                "extremely_short_state",
                f"State has {token_count} tokens; minimum is {min_state_tokens}.",
                state_id=state_id,
                state_position=position,
            )
            state_usable = False
        elif max_state_tokens is not None and token_count > max_state_tokens:
            _add_issue(
                issues,
                "long_state_review",
                f"State has {token_count} tokens, above the {max_state_tokens}-token review threshold.",
                severity="warning",
                state_id=state_id,
                state_position=position,
            )

        start_char = record.get("start_char")
        end_char = record.get("end_char")
        valid_range = (
            isinstance(start_char, int)
            and not isinstance(start_char, bool)
            and isinstance(end_char, int)
            and not isinstance(end_char, bool)
            and start_char >= 0
            and end_char > start_char
        )
        if not valid_range:
            _add_issue(
                issues,
                "invalid_character_range",
                "Character range must have non-negative start_char and end_char > start_char.",
                state_id=state_id,
                state_position=position,
            )
            state_usable = False
        else:
            if previous_end is not None and start_char < previous_end:
                _add_issue(
                    issues,
                    "overlapping_character_ranges",
                    "State character range overlaps the preceding state.",
                    state_id=state_id,
                    state_position=position,
                )
                state_usable = False
            if previous_end is not None and start_char < previous_end:
                # Keep the furthest prior endpoint so a later state cannot hide
                # an overlap with an earlier, longer interval.
                previous_end = max(previous_end, end_char)
            else:
                previous_end = end_char
            if source_trace is not None:
                if end_char > len(source_trace):
                    _add_issue(
                        issues,
                        "character_range_out_of_bounds",
                        "State range extends past the source reasoning trace.",
                        state_id=state_id,
                        state_position=position,
                    )
                    state_usable = False
                elif isinstance(text, str) and source_trace[start_char:end_char] != text:
                    _add_issue(
                        issues,
                        "source_span_mismatch",
                        "State text does not match its declared source trace span.",
                        state_id=state_id,
                        state_position=position,
                    )
                    state_usable = False

        if state_usable:
            usable_count += 1

    if records and (sum(final_flags) != 1 or not final_flags[-1]):
        _add_issue(
            issues,
            "invalid_final_state_assignment",
            "Exactly the final state in the sequence must have is_final_state=true.",
        )

    if usable_count < min_usable_states:
        _add_issue(
            issues,
            "too_few_usable_states",
            f"Trace has {usable_count} usable states; minimum is {min_usable_states}.",
        )

    has_errors = any(issue["severity"] == "error" for issue in issues)
    return {
        "trace_id": trace_id,
        "problem_id": problem_id,
        "is_valid": not has_errors,
        "is_usable_for_week3": not has_errors and usable_count >= min_usable_states,
        "state_count": len(records),
        "usable_state_count": usable_count,
        "issues": issues,
    }


def validate_semantic_state_records(
    state_records: Iterable[Mapping[str, Any] | SemanticState],
    *,
    source_traces: Mapping[str, str] | None = None,
    min_state_tokens: int = 3,
    min_usable_states: int = 2,
    max_state_tokens: int | None = 512,
) -> dict[str, Any]:
    """Validate flat state records and select unchanged Week-3-ready sequences."""
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for position, state in enumerate(state_records):
        try:
            record = _state_record(state)
        except TypeError:
            record = {"_invalid_record": repr(state)}
        trace_id = record.get("trace_id")
        group_key = trace_id if isinstance(trace_id, str) and trace_id.strip() else f"__missing_trace_id_{position}"
        grouped.setdefault(group_key, []).append(record)

    reports: list[dict[str, Any]] = []
    usable_records: list[dict[str, Any]] = []
    for group_key, records in grouped.items():
        trace_id = records[0].get("trace_id")
        problem_id = records[0].get("problem_id")
        source_trace = None
        if source_traces is not None:
            source_trace = source_traces.get(group_key)
            if source_trace is None and isinstance(problem_id, str):
                source_trace = source_traces.get(problem_id)
        report = validate_state_sequence(
            records,
            source_trace=source_trace,
            min_state_tokens=min_state_tokens,
            min_usable_states=min_usable_states,
            max_state_tokens=max_state_tokens,
        )
        if not trace_id:
            report["trace_id"] = None
        reports.append(report)
        if report["is_usable_for_week3"]:
            # Preserve fields and values; only select complete sequences.
            usable_records.extend(dict(record) for record in records)

    return {
        "trace_reports": reports,
        "week3_ready_states": usable_records,
        "trace_count": len(reports),
        "valid_trace_count": sum(report["is_valid"] for report in reports),
        "week3_ready_trace_count": sum(report["is_usable_for_week3"] for report in reports),
        "state_count": sum(report["state_count"] for report in reports),
        "week3_ready_state_count": len(usable_records),
        "invalid_trace_count": sum(not report["is_valid"] for report in reports),
        "warning_count": sum(
            issue["severity"] == "warning"
            for report in reports
            for issue in report["issues"]
        ),
        "thresholds": {
            "min_state_tokens": min_state_tokens,
            "min_usable_states": min_usable_states,
            "max_state_tokens_warning": max_state_tokens,
        },
    }
