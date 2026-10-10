"""Reusable hidden-state extraction and span-pooling helpers for causal LMs."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import numpy as np


def _import_torch(torch_module: Any | None = None) -> Any:
    if torch_module is not None:
        return torch_module
    try:
        return importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - optional in local test env
        raise RuntimeError("PyTorch is required for model hidden-state extraction.") from exc


def _to_plain_offsets(offset_mapping: Any) -> list[tuple[int, int]]:
    if hasattr(offset_mapping, "tolist"):
        offset_mapping = offset_mapping.tolist()
    if offset_mapping and isinstance(offset_mapping[0], list):
        offset_mapping = offset_mapping[0]
    return [(int(start), int(end)) for start, end in offset_mapping]


def token_positions_for_span(
    offsets: Sequence[tuple[int, int]],
    span: tuple[int, int],
) -> list[int]:
    """Return tokens whose character offsets overlap a half-open target span."""
    start, end = span
    if isinstance(start, bool) or isinstance(end, bool) or start < 0 or end <= start:
        raise ValueError("span must be a non-empty half-open character range")
    return [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_end > start and token_start < end
    ]


def extract_span_hidden_vectors(
    model: Any,
    tokenizer: Any,
    rendered_context: str,
    target_spans: Sequence[tuple[int, int]],
    *,
    layer: int = -1,
    pooling: str = "mean",
    torch_module: Any | None = None,
) -> tuple[np.ndarray, list[list[int]], dict[str, int | str]]:
    """Run a context once, then mean-pool selected hidden-state token spans.

    With a causal language model, each token's hidden state sees its preceding
    context but not future tokens. This allows multiple states from the same
    trace to be represented in their original reasoning context in one pass.
    """
    if not isinstance(rendered_context, str) or not rendered_context:
        raise ValueError("rendered_context must be a non-empty string")
    if not target_spans:
        raise ValueError("at least one target span is required")
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise TypeError("layer must be an integer hidden-state index")
    if pooling != "mean":
        raise ValueError("only deterministic mean pooling is currently supported")
    if not callable(getattr(tokenizer, "__call__", None)):
        raise TypeError("tokenizer must be callable and support offset mappings")

    encoded = tokenizer(
        rendered_context,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    if not isinstance(encoded, dict) and not hasattr(encoded, "items"):
        raise TypeError("tokenizer output must be a mapping")
    encoded = dict(encoded)
    offsets_value = encoded.pop("offset_mapping", None)
    if offsets_value is None:
        raise ValueError("tokenizer must provide fast-tokenizer offset_mapping")
    offsets = _to_plain_offsets(offsets_value)
    if not offsets:
        raise ValueError("tokenizer returned no token offsets")

    span_positions = [token_positions_for_span(offsets, span) for span in target_spans]
    if any(not positions for positions in span_positions):
        empty_index = next(i for i, positions in enumerate(span_positions) if not positions)
        raise ValueError(f"target span {empty_index} contains no tokenizer tokens")

    device = getattr(model, "device", None)
    if device is None and callable(getattr(model, "get_input_embeddings", None)):
        device = model.get_input_embeddings().weight.device
    model_inputs = {
        name: value.to(device) if device is not None and callable(getattr(value, "to", None)) else value
        for name, value in encoded.items()
    }

    torch = _import_torch(torch_module)
    inference_context = getattr(torch, "inference_mode", None)
    context_manager = inference_context() if callable(inference_context) else nullcontext()
    if callable(getattr(model, "eval", None)):
        model.eval()
    with context_manager:
        outputs = model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and isinstance(outputs, dict):
        hidden_states = outputs.get("hidden_states")
    if not hidden_states:
        raise ValueError("model did not return hidden_states")

    resolved_layer = layer if layer >= 0 else len(hidden_states) + layer
    if resolved_layer < 0 or resolved_layer >= len(hidden_states):
        raise IndexError(f"layer {layer} is outside {len(hidden_states)} returned hidden-state layers")
    selected = hidden_states[resolved_layer]
    if len(selected.shape) != 3 or selected.shape[0] != 1:
        raise ValueError("expected hidden-state tensor shape [1, sequence_length, hidden_size]")
    if selected.shape[1] != len(offsets):
        raise ValueError("token offsets do not match the model hidden-state sequence length")

    vectors = []
    for positions in span_positions:
        token_index = _to_index_tensor(positions, selected, torch)
        if callable(getattr(selected[0], "index_select", None)):
            token_hidden = selected[0].index_select(0, token_index)
        else:  # Support array-backed test adapters as well as PyTorch tensors.
            token_hidden = selected[0, positions, :]
        if callable(getattr(token_hidden, "float", None)):
            token_hidden = token_hidden.float()
        try:
            vectors.append(token_hidden.mean(dim=0))
        except TypeError:  # NumPy mean uses axis rather than dim.
            vectors.append(token_hidden.mean(axis=0))

    stacked = torch.stack(vectors, dim=0)
    if callable(getattr(stacked, "detach", None)):
        stacked = stacked.detach()
    if callable(getattr(stacked, "float", None)):
        stacked = stacked.float()
    if callable(getattr(stacked, "cpu", None)):
        stacked = stacked.cpu()
    if callable(getattr(stacked, "numpy", None)):
        vector_array = stacked.numpy()
    else:  # pragma: no cover - fallback for non-PyTorch adapters
        vector_array = np.asarray(stacked)
    vector_array = np.asarray(vector_array, dtype=np.float32)

    return vector_array, span_positions, {
        "resolved_layer_index": resolved_layer,
        "hidden_size": int(vector_array.shape[1]),
        "pooling": pooling,
    }


def _to_index_tensor(positions: list[int], tensor: Any, torch: Any) -> Any:
    if callable(getattr(torch, "tensor", None)):
        device = getattr(tensor, "device", None)
        return torch.tensor(positions, dtype=getattr(torch, "long", None), device=device)
    return np.asarray(positions, dtype=np.int64)


