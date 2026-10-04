"""Tests for fixed-prompt Chain-of-Thought record creation."""

from src.baseline.cot import build_cot_prompt, generate_cot_record, run_cot_baseline
from src.models.qwen import GenerationResult


class StubGenerator:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return GenerationResult(
            text="First calculate. Final answer: 5",
            input_tokens=20,
            output_tokens=8,
            generation_config={"do_sample": False, "seed": 42},
        )


def test_prompt_format_is_fixed_while_question_changes():
    first = build_cot_prompt("What is 2 + 3?")
    second = build_cot_prompt("What is 4 + 7?")
    assert first.replace("What is 2 + 3?", "QUESTION") == second.replace(
        "What is 4 + 7?", "QUESTION"
    )
    assert first.endswith("Final answer: <answer>")


def test_generate_cot_record_keeps_trace_counts_and_config():
    generator = StubGenerator()
    record = generate_cot_record(
        {
            "id": "gsm8k-train-00001",
            "question": "What is 2 + 3?",
            "reference_answer": "5",
        },
        generator,
    )
    assert record["problem_id"] == "gsm8k-train-00001"
    assert record["reasoning_trace"] == "First calculate. Final answer: 5"
    assert record["input_tokens"] == 20
    assert record["output_tokens"] == 8
    assert record["generation_config"] == {"do_sample": False, "seed": 42}
    assert record["reference_answer"] == "5"
    assert generator.prompts == [record["prompt"]]


def test_run_cot_baseline_preserves_order_and_honors_limit():
    examples = [
        {"id": f"id-{index}", "question": f"Question {index}?"}
        for index in range(3)
    ]
    records = run_cot_baseline(examples, StubGenerator(), limit=2)
    assert [record["problem_id"] for record in records] == ["id-0", "id-1"]

