"""Extract and normalize numeric final answers independently of inference."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

_EXPLICIT_ANSWER_PATTERN = re.compile(
    r"(?im)(?:^|\b)(?:the\s+)?(?:final\s+answer|answer)\s*"
    r"(?:is\s*|:\s*|=\s*)([^\r\n]+)"
)
_GSM8K_MARKER_PATTERN = re.compile(r"(?im)####\s*([^\r\n]+)")
_LATEX_FRACTION_PATTERN = re.compile(
    r"\\(?:d?frac)\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}"
    r"\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}"
)
_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[-+]?\s*[$£€]?\s*"
    r"(?:"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+"
    r")"
    r"(?:[eE][-+]?\d+)?"
    r"(?:\s*/\s*[-+]?\s*(?:\d+(?:\.\d+)?|\.\d+))?"
    r"\s*%?"
)


def _boxed_contents(text: str) -> list[str]:
    """Return balanced contents of every LaTeX ``\\boxed{...}`` expression."""
    results: list[str] = []
    marker = r"\boxed{"
    start = 0
    while True:
        marker_index = text.find(marker, start)
        if marker_index < 0:
            break
        content_start = marker_index + len(marker)
        depth = 1
        index = content_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            results.append(text[content_start : index - 1])
        start = max(index, content_start)
    return results


def _replace_latex_fractions(text: str) -> str:
    return _LATEX_FRACTION_PATTERN.sub(r"\1/\2", text)


def _last_numeric_token(text: str) -> str | None:
    matches = _NUMBER_PATTERN.findall(_replace_latex_fractions(text))
    return matches[-1].strip() if matches else None


def _first_numeric_token(text: str) -> str | None:
    match = _NUMBER_PATTERN.search(_replace_latex_fractions(text))
    return match.group(0).strip() if match else None


def extract_final_answer(response: str) -> str | None:
    """Extract the most strongly signaled final numeric answer from a response.

    Explicit ``Final answer``/``Answer`` labels take precedence, followed by the
    GSM8K ``####`` marker, LaTeX ``\\boxed{}``, and finally the last numeric token.
    ``None`` is returned for empty, malformed, or entirely non-numeric output.
    """
    if not isinstance(response, str) or not response.strip():
        return None

    explicit = _EXPLICIT_ANSWER_PATTERN.findall(response)
    if explicit:
        return _first_numeric_token(explicit[-1])

    marked = _GSM8K_MARKER_PATTERN.findall(response)
    if marked:
        return _first_numeric_token(marked[-1])

    boxed = _boxed_contents(response)
    if boxed:
        return _first_numeric_token(boxed[-1])

    return _last_numeric_token(response)


def _fraction_from_text(value: str) -> Fraction | None:
    text = _replace_latex_fractions(value).strip()
    text = text.replace("−", "-").replace("–", "-")
    text = re.sub(r"[*_`]", "", text)
    text = text.replace(",", "").replace(" ", "")
    text = text.replace("$", "").replace("£", "").replace("€", "")
    text = text.removesuffix("%")
    text = text.rstrip(".。;,!?")
    if not text:
        return None

    try:
        if "/" in text:
            numerator, denominator = text.split("/", maxsplit=1)
            denominator_value = Fraction(Decimal(denominator))
            if denominator_value == 0:
                return None
            return Fraction(Decimal(numerator)) / denominator_value
        return Fraction(Decimal(text))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def normalize_numeric_answer(value: Any) -> str | None:
    """Return an exact canonical numeric string, or ``None`` when invalid.

    Commas, currency symbols, percent notation, decimal formatting, scientific
    notation, simple fractions, and LaTeX fractions are normalized. Percent signs
    are treated as presentation (``50%`` equals GSM8K's numeric answer ``50``),
    matching the dataset's convention of storing the requested numeric value.
    """
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    fraction = _fraction_from_text(text)
    if fraction is None:
        token = _last_numeric_token(text)
        fraction = _fraction_from_text(token) if token is not None else None
    if fraction is None:
        return None
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def answers_equal(predicted: Any, reference: Any) -> bool:
    """Compare two answers after exact numeric normalization."""
    predicted_normalized = normalize_numeric_answer(predicted)
    reference_normalized = normalize_numeric_answer(reference)
    return (
        predicted_normalized is not None
        and reference_normalized is not None
        and predicted_normalized == reference_normalized
    )

