"""Create reusable vector representations linked to semantic-state records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.models.hidden_states import extract_span_hidden_vectors
from .schema import SemanticState


@dataclass(frozen=True, slots=True)
class StateRepresentationBatch:
    """Vectors, their ordered state links, and the exact extraction settings."""

    vectors: np.ndarray
    index_records: list[dict[str, Any]]
    configuration: dict[str, Any]


def _state_from_record(record: Mapping[str, Any] | SemanticState) -> SemanticState:
    if isinstance(record, SemanticState):
        return record
    if not isinstance(record, Mapping):
        raise TypeError("each state must be a SemanticState or mapping")
    return SemanticState.from_dict(record)


def _model_revision(model: Any) -> str | None:
    config = getattr(model, "config", None)
    revision = getattr(config, "_commit_hash", None)
    return revision if isinstance(revision, str) and revision else None


def _tokenizer_revision(tokenizer: Any) -> str | None:
    init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
    revision = init_kwargs.get("_commit_hash")
    return revision if isinstance(revision, str) and revision else None


def extract_semantic_state_representations(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    reasoning_trace: str,
    states: Sequence[Mapping[str, Any] | SemanticState],
    model_id: str = "Qwen/Qwen3-4B",
    layer: int = -1,
    pooling: str = "mean",
    torch_module: Any | None = None,
) -> StateRepresentationBatch:
    """Extract one mean-pooled vector for each ordered state in one trace.

    The model sees the original user prompt and complete assistant trace in the
    model's Qwen chat format. Because Qwen is causal, hidden states at a state
    span incorporate its prompt and preceding reasoning context, not future
    state tokens. The prompt/trace input is forwarded once per trace.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(reasoning_trace, str) or not reasoning_trace.strip():
        raise ValueError("reasoning_trace must be a non-empty string")
    if not states:
        raise ValueError("at least one semantic state is required")
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise TypeError("tokenizer must support apply_chat_template()")

    state_objects = [_state_from_record(record) for record in states]
    problem_id = state_objects[0].problem_id
    trace_id = state_objects[0].trace_id
    for index, state in enumerate(state_objects):
        if state.state_index != index or state.state_id != f"S{index + 1}":
            raise ValueError("states must be ordered with contiguous S1..Sn identifiers")
        if state.problem_id != problem_id or state.trace_id != trace_id:
            raise ValueError("all states in the batch must belong to one trace")
        state.validate_trace_span(reasoning_trace)

    rendered_context = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning_trace},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    trace_start = rendered_context.rfind(reasoning_trace)
    if trace_start < 0:
        raise ValueError("chat template output does not preserve the source reasoning trace")
    target_spans = [
        (trace_start + state.start_char, trace_start + state.end_char)
        for state in state_objects
    ]

    vectors, token_positions, extraction = extract_span_hidden_vectors(
        model,
        tokenizer,
        rendered_context,
        target_spans,
        layer=layer,
        pooling=pooling,
        torch_module=torch_module,
    )
    if vectors.shape[0] != len(state_objects):
        raise RuntimeError("model produced a different number of vectors than source states")

    index_records = []
    for state, positions in zip(state_objects, token_positions, strict=True):
        index_records.append(
            {
                "vector_index": len(index_records),
                "problem_id": state.problem_id,
                "trace_id": state.trace_id,
                "state_id": state.state_id,
                "state_index": state.state_index,
                "layer": layer,
                "resolved_layer_index": extraction["resolved_layer_index"],
                "pooling": pooling,
                "hidden_size": extraction["hidden_size"],
                "token_count": state.token_count,
                "pooled_token_count": len(positions),
                "is_final_state": state.is_final_state,
            }
        )

    model_config = getattr(model, "config", None)
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
    configuration = {
        "model_id": model_id,
        "model_revision": _model_revision(model),
        "tokenizer_id": getattr(tokenizer, "name_or_path", model_id),
        "tokenizer_revision": _tokenizer_revision(tokenizer),
        "selected_layer": layer,
        "resolved_layer_index": extraction["resolved_layer_index"],
        "hidden_dimension": extraction["hidden_size"],
        "pooling": pooling,
        "input_context": "Qwen chat-formatted original prompt and full assistant trace",
        "contextualization": "causal prefix; mean pool token offsets overlapping each state span",
        "model_type": getattr(model_config, "model_type", None),
        "tokenizer_commit_hash": tokenizer_kwargs.get("_commit_hash"),
        "vector_dtype": str(vectors.dtype),
        "vector_shape": list(vectors.shape),
    }
    return StateRepresentationBatch(
        vectors=vectors,
        index_records=index_records,
        configuration=configuration,
    )
