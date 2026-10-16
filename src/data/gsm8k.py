"""Reproducible loading and preparation of the official GSM8K dataset."""

from __future__ import annotations

import importlib
import random
import re
from collections.abc import Callable, Mapping
from typing import Any

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_CONFIG = "main"
DEFAULT_DEVELOPMENT_SEED = 42
REQUIRED_SPLITS = frozenset({"train", "test"})

_FINAL_ANSWER_PATTERN = re.compile(r"####\s*([^\r\n]+?)\s*$")


def extract_reference_answer(solution: str) -> str:
    """Extract the answer following GSM8K's final ``####`` marker.

    Parsing is intentionally strict: malformed source records fail during data
    preparation instead of silently producing an incorrect evaluation target.
    Numeric normalization belongs to the independent evaluation pipeline.
    """
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError("GSM8K reference solution must be a non-empty string.")

    match = _FINAL_ANSWER_PATTERN.search(solution)
    if match is None:
        raise ValueError("GSM8K reference solution has no final '####' answer marker.")

    answer = match.group(1).strip()
    if not answer:
        raise ValueError("GSM8K reference solution contains an empty final answer.")
    return answer


def normalize_example(
    example: Mapping[str, Any],
    *,
    split: str,
    index: int,
) -> dict[str, str]:
    """Convert a raw GSM8K row to the project's stable internal schema."""
    question = example.get("question")
    solution = example.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"GSM8K {split} example {index} has no valid question.")
    if not isinstance(solution, str):
        raise ValueError(f"GSM8K {split} example {index} has no valid solution.")

    return {
        "id": f"gsm8k-{split}-{index:05d}",
        "question": question.strip(),
        "reference_solution": solution.strip(),
        "reference_answer": extract_reference_answer(solution),
    }


def _prepare_split(dataset_split: Any, split_name: str) -> Any:
    """Normalize either a Hugging Face Dataset or a sequence used in tests."""
    if hasattr(dataset_split, "map") and hasattr(dataset_split, "column_names"):
        return dataset_split.map(
            lambda example, index: normalize_example(
                example, split=split_name, index=index
            ),
            with_indices=True,
            remove_columns=dataset_split.column_names,
            desc=f"Normalizing GSM8K {split_name}",
        )

    return [
        normalize_example(example, split=split_name, index=index)
        for index, example in enumerate(dataset_split)
    ]


def create_development_subset(
    train_split: Any,
    *,
    size: int,
    seed: int = DEFAULT_DEVELOPMENT_SEED,
) -> Any:
    """Sample a deterministic subset from training data only.

    Selected rows are returned in their original dataset order. The official
    test split is never accepted or sampled by this function implicitly; callers
    must explicitly provide the prepared training split.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("Development subset size must be a positive integer.")
    if size > len(train_split):
        raise ValueError(
            f"Development subset size {size} exceeds training size "
            f"{len(train_split)}."
        )

    indices = sorted(random.Random(seed).sample(range(len(train_split)), size))
    if hasattr(train_split, "select"):
        return train_split.select(indices)
    return [train_split[index] for index in indices]


def load_gsm8k(
    *,
    development_size: int | None = None,
    seed: int = DEFAULT_DEVELOPMENT_SEED,
    dataset_loader: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load, validate, and normalize the official GSM8K train/test splits.

    When ``development_size`` is provided, the returned mapping also contains a
    deterministic ``development`` subset sampled exclusively from ``train``.
    The complete official train and test datasets remain separate and unchanged.
    ``dataset_loader`` is injectable to support offline tests.
    """
    if dataset_loader is None:
        try:
            datasets = importlib.import_module("datasets")
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'datasets'. Install the project requirements "
                "first: python -m pip install -r requirements.txt"
            ) from exc
        dataset_loader = datasets.load_dataset

    raw_dataset = dataset_loader(GSM8K_DATASET_ID, GSM8K_CONFIG)
    missing_splits = REQUIRED_SPLITS.difference(raw_dataset.keys())
    if missing_splits:
        missing = ", ".join(sorted(missing_splits))
        raise ValueError(f"GSM8K dataset is missing required split(s): {missing}.")

    prepared = {
        "train": _prepare_split(raw_dataset["train"], "train"),
        "test": _prepare_split(raw_dataset["test"], "test"),
    }
    if development_size is not None:
        prepared["development"] = create_development_subset(
            prepared["train"], size=development_size, seed=seed
        )
    return prepared

