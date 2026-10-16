"""Tests for final-answer extraction and exact numeric normalization."""

import pytest

from src.evaluation.answer_extraction import (
    answers_equal,
    extract_final_answer,
    normalize_numeric_answer,
)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Reasoning gives 40 + 2.\nFinal answer: 42", "42"),
        ("I obtain 11. The final answer is **$1,234.00**.", "$1,234.00"),
        ("Work\n#### -17", "-17"),
        (r"Therefore, \boxed{\frac{1}{2}}.", "1/2"),
        ("First 3, then 7, and ultimately 9.", "9"),
        ("Answer = 50%", "50%"),
        ("Final answer: 42 dollars (from 40 + 2)", "42"),
    ],
)
def test_extract_final_answer_common_formats(response, expected):
    assert extract_final_answer(response) == expected


@pytest.mark.parametrize("response", ["", "No numeric result", None])
def test_extract_final_answer_returns_none_for_malformed_output(response):
    assert extract_final_answer(response) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1,234.00", "1234"),
        ("$0012.50", "25/2"),
        ("0.5", "1/2"),
        ("1/2", "1/2"),
        (r"\frac{2}{4}", "1/2"),
        ("5e2", "500"),
        ("50%", "50"),
        ("-0", "0"),
    ],
)
def test_normalize_numeric_answer(value, expected):
    assert normalize_numeric_answer(value) == expected


@pytest.mark.parametrize("value", [None, True, "", "not a number", "1/0"])
def test_normalize_numeric_answer_rejects_invalid_values(value):
    assert normalize_numeric_answer(value) is None


@pytest.mark.parametrize(
    ("predicted", "reference"),
    [("1,000.00", "1000"), ("2/4", "0.5"), ("42%", "42")],
)
def test_answers_equal_for_equivalent_formats(predicted, reference):
    assert answers_equal(predicted, reference)


def test_answers_equal_rejects_missing_or_different_answers():
    assert not answers_equal(None, "0")
    assert not answers_equal("41", "42")

