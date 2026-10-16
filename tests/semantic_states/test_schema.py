import pytest

from src.semantic_states import SemanticState


def make_state(**overrides):
    values = {
        "problem_id": "gsm8k-train-00053",
        "trace_id": "run-123:gsm8k-train-00053",
        "state_id": "S1",
        "state_index": 0,
        "text": "Compute 10 × 4 = 40.",
        "start_char": 0,
        "end_char": 20,
        "token_count": 8,
        "is_final_state": False,
    }
    values.update(overrides)
    return SemanticState(**values)


def test_state_validates_source_span_and_round_trips():
    state = make_state()

    state.validate_trace_span("Compute 10 × 4 = 40.")
    assert SemanticState.from_dict(state.to_dict()) == state


def test_state_rejects_mismatched_identifier():
    with pytest.raises(ValueError, match="state_id must be 'S2'"):
        make_state(state_index=1)


def test_state_rejects_source_span_mismatch():
    with pytest.raises(ValueError, match="does not equal"):
        make_state().validate_trace_span("Compute 10 × 4 = 41.")


def test_state_rejects_empty_text_and_invalid_positions():
    with pytest.raises(ValueError, match="text must be"):
        make_state(text=" ")
    with pytest.raises(ValueError, match="end_char must be greater"):
        make_state(end_char=0)


def test_from_dict_requires_exact_schema_fields():
    with pytest.raises(ValueError, match="missing semantic-state fields"):
        SemanticState.from_dict({"problem_id": "only-one-field"})
