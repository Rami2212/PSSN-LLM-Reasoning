from copy import deepcopy

import pytest

from src.black_hole.state_removal import remove_semantic_state


def make_states(count=5):
    texts = [f"State {index + 1} contains enough words to validate." for index in range(count - 1)]
    texts.append("The final answer is 42 with these words.")
    parts = []
    states = []
    offset = 0
    for index, text in enumerate(texts):
        if parts:
            parts.append(" ")
            offset += 1
        start = offset
        parts.append(text)
        offset += len(text)
        states.append(
            {
                "problem_id": "gsm8k-train-00001",
                "trace_id": "run-a:gsm8k-train-00001",
                "state_id": f"S{index + 1}",
                "state_index": index,
                "text": text,
                "start_char": start,
                "end_char": offset,
                "token_count": len(text.split()),
                "is_final_state": index == count - 1,
            }
        )
    return states, "".join(parts)


@pytest.mark.parametrize("index", [0, 2, 3])
def test_removes_beginning_middle_or_near_end_without_reordering(index):
    states, source = make_states()
    removed_id = f"S{index + 1}"
    result = remove_semantic_state(states, removed_id, source_trace=source)

    expected_ids = [state["state_id"] for i, state in enumerate(states) if i != index]
    assert result["remaining_state_ids"] == expected_ids
    assert result["removed_state_index"] == index
    assert result["modified_state_count"] == len(states) - 1
    assert result["modified_reasoning_context"] == "\n".join(
        state["text"] for i, state in enumerate(states) if i != index
    )


def test_removing_only_non_final_state_from_minimum_valid_trace_is_supported():
    states, source = make_states(count=2)
    result = remove_semantic_state(states, "S1", source_trace=source)

    assert result["original_state_count"] == 2
    assert result["modified_state_count"] == 1
    assert result["remaining_state_ids"] == ["S2"]
    assert result["modified_reasoning_context"] == states[-1]["text"]
    assert result["continuation_context"] == ""
    assert result["continuation_state_ids"] == []


def test_final_answer_cannot_be_removed_and_input_is_unchanged():
    states, source = make_states()
    original = deepcopy(states)

    with pytest.raises(ValueError, match="final-answer state"):
        remove_semantic_state(states, "S5", source_trace=source)

    result = remove_semantic_state(states, "S2", source_trace=source)
    assert states == original
    assert result["remaining_states"][0] == original[0]
    assert result["remaining_states"][0] is not states[0]


def test_rejects_missing_state_invalid_trace_and_bad_source_span():
    states, source = make_states()
    with pytest.raises(ValueError, match="not found"):
        remove_semantic_state(states, "S99", source_trace=source)
    with pytest.raises(ValueError, match="not valid"):
        remove_semantic_state(states, "S1", source_trace="different source trace")

    states[1]["text"] = "mismatched text"
    with pytest.raises(ValueError, match="not valid"):
        remove_semantic_state(states, "S1", source_trace=source)


def test_custom_separator_is_applied_deterministically():
    states, source = make_states(count=3)
    first = remove_semantic_state(states, "S2", source_trace=source, reconstruction_separator=" ")
    second = remove_semantic_state(states, "S2", source_trace=source, reconstruction_separator=" ")

    assert first == second
    assert first["modified_reasoning_context"] == f"{states[0]['text']} {states[2]['text']}"
