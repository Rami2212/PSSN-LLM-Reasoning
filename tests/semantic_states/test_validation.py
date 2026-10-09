from copy import deepcopy

import pytest

from src.semantic_states import validate_semantic_state_records, validate_state_sequence


def valid_states():
    trace = "First step. Next step. Final answer: 2"
    texts = ["First step.", "Next step.", "Final answer: 2"]
    records = []
    cursor = 0
    for index, text in enumerate(texts):
        start = trace.index(text, cursor)
        end = start + len(text)
        records.append(
            {
                "problem_id": "gsm8k-train-00053",
                "trace_id": "run-123:gsm8k-train-00053",
                "state_id": f"S{index + 1}",
                "state_index": index,
                "text": text,
                "start_char": start,
                "end_char": end,
                "token_count": 4,
                "is_final_state": index == len(texts) - 1,
            }
        )
        cursor = end
    return trace, records


def issue_codes(report):
    return {issue["code"] for issue in report["issues"]}


def test_valid_sequence_passes_source_and_order_checks():
    trace, states = valid_states()

    report = validate_state_sequence(states, source_trace=trace)

    assert report["is_valid"] is True
    assert report["is_usable_for_week3"] is True
    assert report["state_count"] == report["usable_state_count"] == 3
    assert report["issues"] == []


def test_short_fragment_is_flagged_and_not_counted_usable():
    _, states = valid_states()
    states[1]["token_count"] = 2

    report = validate_state_sequence(states, min_state_tokens=3)

    assert "extremely_short_state" in issue_codes(report)
    assert report["usable_state_count"] == 2
    assert report["is_valid"] is False


def test_duplicate_adjacent_states_are_flagged():
    _, states = valid_states()
    states[1]["text"] = "  FIRST   STEP. "

    report = validate_state_sequence(states)

    assert "duplicate_adjacent_state" in issue_codes(report)
    assert report["is_valid"] is False


def test_bad_ids_and_state_order_are_reported():
    _, states = valid_states()
    states[1]["state_id"] = "S9"
    states[1]["state_index"] = 4

    report = validate_state_sequence(states)

    assert {"state_id_mismatch", "invalid_state_order"} <= issue_codes(report)


def test_missing_identifiers_token_count_and_ranges_are_reported():
    _, states = valid_states()
    states[1].pop("state_id")
    states[1].pop("token_count")
    states[1]["start_char"] = None

    report = validate_state_sequence(states)

    assert {
        "missing_state_id",
        "missing_token_count",
        "invalid_character_range",
    } <= issue_codes(report)


def test_overlapping_ranges_and_source_mismatch_are_reported():
    trace, states = valid_states()
    states[1]["start_char"] = states[0]["end_char"] - 1
    states[1]["end_char"] = states[1]["start_char"] + len(states[1]["text"])

    report = validate_state_sequence(states, source_trace=trace)

    assert "overlapping_character_ranges" in issue_codes(report)
    assert "source_span_mismatch" in issue_codes(report)


def test_ranges_outside_source_are_reported():
    trace, states = valid_states()
    states[1]["start_char"] = len(trace) + 1
    states[1]["end_char"] = len(trace) + 5

    report = validate_state_sequence(states, source_trace=trace)

    assert "character_range_out_of_bounds" in issue_codes(report)


def test_long_state_is_non_blocking_manual_review_warning():
    _, states = valid_states()
    states[1]["token_count"] = 513

    report = validate_state_sequence(states, max_state_tokens=512)

    assert report["is_valid"] is True
    assert report["is_usable_for_week3"] is True
    warning = next(issue for issue in report["issues"] if issue["code"] == "long_state_review")
    assert warning["severity"] == "warning"


def test_empty_sequence_and_too_few_states_are_flagged():
    report = validate_state_sequence([])

    assert {"empty_sequence", "too_few_usable_states", "missing_trace_id"} <= issue_codes(report)
    assert report["is_usable_for_week3"] is False


def test_inconsistent_trace_identity_is_flagged():
    _, states = valid_states()
    states[1]["trace_id"] = "different-run"

    report = validate_state_sequence(states)

    assert "inconsistent_trace_id" in issue_codes(report)


def test_validator_does_not_mutate_input_records():
    _, states = valid_states()
    original = deepcopy(states)

    validate_state_sequence(states)

    assert states == original


def test_flat_validation_selects_only_valid_sequences_without_rewriting():
    _, good = valid_states()
    bad = deepcopy(good)
    bad_trace_id = "run-456:gsm8k-train-00054"
    for state in bad:
        state["trace_id"] = bad_trace_id
        state["problem_id"] = "gsm8k-train-00054"
    bad[1]["token_count"] = 1
    original_good = deepcopy(good)
    original_bad = deepcopy(bad)

    result = validate_semantic_state_records([*good, *bad])

    assert result["trace_count"] == 2
    assert result["valid_trace_count"] == 1
    assert result["week3_ready_trace_count"] == 1
    assert result["week3_ready_states"] == original_good
    assert good == original_good
    assert bad == original_bad


def test_missing_trace_identifier_gets_its_own_invalid_report():
    _, states = valid_states()
    states[0].pop("trace_id")

    result = validate_semantic_state_records(states)

    assert result["trace_count"] == 2
    assert result["invalid_trace_count"] == 2
    assert sum(report["trace_id"] is None for report in result["trace_reports"]) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_state_tokens": 0},
        {"min_usable_states": 0},
        {"min_state_tokens": 4, "max_state_tokens": 3},
    ],
)
def test_threshold_configuration_is_validated(kwargs):
    with pytest.raises(ValueError):
        validate_state_sequence([], **kwargs)
