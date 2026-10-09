"""Tokenizer loading and token counting for semantic-state records."""

from __future__ import annotations

from typing import Any


DEFAULT_TOKENIZER_ID = "Qwen/Qwen3-4B"


def load_qwen_tokenizer(
    model_id: str = DEFAULT_TOKENIZER_ID,
    revision: str | None = None,
    **kwargs: Any,
) -> Any:
    """Load the model's Hugging Face tokenizer without loading model weights."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "Install transformers to load a tokenizer, or pass a tokenizer "
            "object directly to segment_reasoning_trace()."
        ) from exc

    return AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        **kwargs,
    )


def count_tokens(text: str, tokenizer: Any) -> int:
    """Count tokenizer tokens without adding BOS/EOS or other special tokens."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if not callable(getattr(tokenizer, "encode", None)):
        raise TypeError("tokenizer must provide an encode() method")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)
