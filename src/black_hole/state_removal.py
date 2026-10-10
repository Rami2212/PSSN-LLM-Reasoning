"""Controlled leave-one-state-out transformations for Week 3 experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from src.semantic_states.schema import SemanticState
from src.semantic_states.validation import validate_state_sequence


def _state_dict(state: Mapping[str, Any] | SemanticState) -> dict[str, Any]:
    if isinstance(state, SemanticState):
        return state.to_dict()
    if isinstance(state, Mapping):
        return deepcopy(dict(state))
    raise TypeError("each semantic state must be a mapping or SemanticState")


def remove_semantic_state(
    states: Iterable[Mapping[str, Any] | SemanticState],
    state_id: str,
    *,
    source_trace: str,
    reconstruction_separator: str = "\n",
) -> dict[str, Any]:
    """Remove one validated non-final state and return a reproducible record.

    The modified reasoning context is the text of the remaining states in
    original order, joined by ``reconstruction_separator``. State dictionaries
    are copied, and the caller's sequence is never modified.
    """
    if not isinstance(state_id, str) or not state_id.strip():
        raise ValueError("state_id must be a non-empty string")
    if not isinstance(source_trace, str) or not source_trace.strip():
        raise ValueError("source_trace must be a non-empty string")
    if not isinstance(reconstruction_separator, str):
        raise TypeError("reconstruction_separator must be a string")

    source_states = [_state_dict(state) for state in states]
    if not source_states:
        raise ValueError("states must not be empty")

    validation = validate_state_sequence(
        source_states,
        source_trace=source_trace,
        min_state_tokens=3,
        min_usable_states=2,
    )
    if not validation["is_usable_for_week3"]:
        issue_codes = sorted({issue["code"] for issue in validation["issues"]})
        raise ValueError(f"semantic-state sequence is not valid for Week 3: {issue_codes}")

    matches = [index for index, state in enumerate(source_states) if state.get("state_id") == state_id]
    if not matches:
        raise ValueError(f"state_id {state_id!r} was not found in the sequence")
    if len(matches) != 1:
        raise ValueError(f"state_id {state_id!r} is not unique in the sequence")

    removed_index = matches[0]
    removed_state = source_states[removed_index]
    if removed_state.get("is_final_state") is True:
        raise ValueError("the final-answer state cannot be removed")

    remaining_states = [
        deepcopy(state)
        for index, state in enumerate(source_states)
        if index != removed_index
    ]
    problem_id = source_states[0]["problem_id"]
    trace_id = source_states[0]["trace_id"]
    return {
        "schema_version": "1.0",
        "problem_id": problem_id,
        "trace_id": trace_id,
        "removed_state_id": removed_state["state_id"],
        "removed_state_index": removed_state["state_index"],
        "removed_state_text": removed_state["text"],
        "removed_state_token_count": removed_state["token_count"],
        "original_state_count": len(source_states),
        "modified_state_count": len(remaining_states),
        "remaining_state_ids": [state["state_id"] for state in remaining_states],
        "remaining_states": remaining_states,
        "modified_reasoning_context": reconstruction_separator.join(
            state["text"] for state in remaining_states
        ),
        "reconstruction_separator": reconstruction_separator,
    }
