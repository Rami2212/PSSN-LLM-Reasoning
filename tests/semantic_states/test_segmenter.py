import pytest

from src.semantic_states import UnsegmentableTraceError, segment_reasoning_trace


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return text.split()


def record(trace, **overrides):
    values = {
        "problem_id": "gsm8k-train-00053",
        "run_id": "run-123",
        "reasoning_trace": trace,
        "is_correct": True,
        "evaluation_status": "scored",
        "finish_reason": "eos",
    }
    values.update(overrides)
    return values


def test_segments_sentences_preserves_source_offsets_and_final_answer():
    trace = (
        "<think>First calculate 10 times 4. Next subtract 4."
        "</think>\nFinal answer: 40"
    )
    states = segment_reasoning_trace(record(trace), FakeTokenizer())

    assert [state.state_id for state in states] == ["S1", "S2", "S3"]
    assert [state.text for state in states] == [
        "First calculate 10 times 4.",
        "Next subtract 4.",
        "Final answer: 40",
    ]
    assert [state.is_final_state for state in states] == [False, False, True]
    assert [state.token_count for state in states] == [5, 3, 3]
    for state in states:
        state.validate_trace_span(trace)


def test_numbered_steps_keep_wrapped_lines_as_one_state():
    trace = (
        "<think>1. Multiply 10 by 4.\n   10 × 4 = 40.\n"
        "2. Divide by 2 to get 20.</think>\n<answer>20</answer>"
    )
    states = segment_reasoning_trace(record(trace), FakeTokenizer())

    assert len(states) == 3
    assert states[0].text == "1. Multiply 10 by 4.\n   10 × 4 = 40."
    assert states[1].text == "2. Divide by 2 to get 20."
    assert states[2].text == "<answer>20</answer>"
    assert states[-1].is_final_state


def test_continuation_sentence_merges_with_previous_state():
    trace = "<think>Calculate 8 times 2. Which equals 16.</think>####16"
    states = segment_reasoning_trace(record(trace), FakeTokenizer())

    assert len(states) == 2
    assert states[0].text == "Calculate 8 times 2. Which equals 16."
    assert states[1].text == "####16"


def test_explanatory_continuation_stays_with_its_operation():
    trace = "<think>Calculate 8 times 2. This gives 16.</think>Final answer: 16"
    states = segment_reasoning_trace(record(trace), FakeTokenizer())

    assert len(states) == 2
    assert states[0].text == "Calculate 8 times 2. This gives 16."


@pytest.mark.parametrize(
    "trace",
    [
        "<think>Reason without a terminal answer marker.</think>",
        "<think>Answer: 42.</think>",
        "<think>Reason. </think>Final answer: 42</think>",
    ],
)
def test_missing_or_malformed_final_marker_is_flagged(trace):
    with pytest.raises(UnsegmentableTraceError):
        segment_reasoning_trace(record(trace), FakeTokenizer())


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_correct": False},
        {"evaluation_status": "incomplete"},
        {"finish_reason": "length"},
    ],
)
def test_rejects_unvalidated_or_incomplete_records(overrides):
    with pytest.raises(UnsegmentableTraceError):
        segment_reasoning_trace(
            record("<think>Compute 2 + 2.</think>Final answer: 4", **overrides),
            FakeTokenizer(),
        )


def test_identifiers_are_stable_across_runs():
    source = record("<think>Compute 3 + 3.</think>Final answer: 6")

    first = segment_reasoning_trace(source, FakeTokenizer())
    second = segment_reasoning_trace(source, FakeTokenizer())

    assert [state.to_dict() for state in first] == [state.to_dict() for state in second]
    assert [state.trace_id for state in first] == ["run-123:gsm8k-train-00053"] * 2
