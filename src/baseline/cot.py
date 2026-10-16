"""Fixed-prompt Chain-of-Thought baseline for normalized GSM8K examples."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from src.models.qwen import GenerationResult

BASELINE_PROMPT_TEMPLATE = """Solve the following math problem carefully.

Problem:
{question}

Show your reasoning step by step. End your response with exactly:
Final answer: <answer>"""


class ReasoningGenerator(Protocol):
    def generate(self, prompt: str) -> GenerationResult: ...


def build_cot_prompt(question: str) -> str:
    """Apply the single canonical prompt used for every GSM8K problem."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("GSM8K question must be a non-empty string.")
    return BASELINE_PROMPT_TEMPLATE.format(question=question.strip())


def generate_cot_record(
    example: Mapping[str, Any],
    generator: ReasoningGenerator,
) -> dict[str, Any]:
    """Generate and serialize one complete baseline reasoning record."""
    problem_id = example.get("id")
    question = example.get("question")
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError("Normalized example must contain a non-empty string 'id'.")

    prompt = build_cot_prompt(question)
    generation = generator.generate(prompt)
    record = {
        "problem_id": problem_id,
        "question": question.strip(),
        "prompt": prompt,
        "reasoning_trace": generation.text,
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "finish_reason": generation.finish_reason,
        "generation_config": dict(generation.generation_config),
        "model_metadata": dict(generation.model_metadata),
    }
    record.update(generation.metrics)
    if example.get("reference_answer") is not None:
        record["reference_answer"] = str(example["reference_answer"])
    return record


def run_cot_baseline(
    examples: Iterable[Mapping[str, Any]],
    generator: ReasoningGenerator,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run the baseline over examples while preserving input order."""
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer or None.")

    records: list[dict[str, Any]] = []
    for example in examples:
        if limit is not None and len(records) >= limit:
            break
        records.append(generate_cot_record(example, generator))
    return records

