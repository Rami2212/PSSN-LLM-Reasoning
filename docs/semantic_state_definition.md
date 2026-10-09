# Semantic-State Segmentation Contract

This document defines the reproducible contract for turning a validated
reasoning trace into semantic states. It defines what a state means and how
states are represented; the segmentation algorithm is implemented separately.

## Definition and scope

A semantic state is the smallest contiguous span of a trace that expresses one
coherent reasoning action or sub-goal (for example, identifying relevant
quantities, performing one calculation, or interpreting a result). A state is
not a fixed-size token window. States preserve the source text verbatim and in
the original order. This contract carries no useful/low-value label and makes
no Black Hole or Wormhole prediction.

Input is a complete, parseable, correct record from
`segmentation_ready_traces.jsonl`. The source is its `reasoning_trace` string.
Segmentation must not alter that string. Character offsets refer to Python
Unicode string indices in that exact string; `start_char` is inclusive and
`end_char` is exclusive. The state text must equal
`reasoning_trace[start_char:end_char]`.

## Boundary rules

Apply these rules in order and deterministically:

1. Preserve source order and contiguous coverage of all semantic content. Do
   not reorder, paraphrase, normalize, or drop reasoning/answer content.
   Structural delimiters explicitly excluded by rule 6 are the only uncovered
   source ranges. Whitespace between content spans belongs to the adjacent
   state; leading/trailing whitespace may be excluded from a state's span.
2. Use explicit numbered/bulleted reasoning steps as candidate boundaries.
   Keep a heading or short continuation line (including equation continuations)
   with the following or preceding step when it does not express a separate
   reasoning action.
3. Otherwise, use sentence boundaries as candidate boundaries. Split adjacent
   sentences when they perform distinct operations or move to a new sub-goal;
   keep sentences together when one is a direct explanation, check, or
   continuation of the same operation.
4. A mathematical operation and its immediate interpretation form one state
   unless the trace clearly starts a different operation or sub-goal. A
   transition such as “next,” “therefore,” or “now” is a candidate boundary,
   not an automatic split when the following text merely completes the current
   action.
5. Do not emit empty states. Attach a short fragment (including a standalone
   equation, connective, or one-line continuation) to the adjacent state that
   it completes. If it cannot be attached without changing source order, keep
   it as its own non-empty state; there is no arbitrary minimum character or
   token threshold.
6. Treat `<think>` and `</think>` as structural delimiters, not reasoning
   content: exclude the marker text from state spans, while preserving all
   reasoning text inside the block. Other text outside the block remains part
   of the trace and must be represented.
7. Keep the terminal answer presentation separate from preceding reasoning
   when it is explicitly marked by `Final answer:`, a GSM8K `####` answer
   marker, or a terminal `<answer>...</answer>` span. The answer presentation
   is one final state and is marked `is_final_state=true`. Do not discard an
   earlier answer-like sentence inside the reasoning block. If no recognized
   marker exists, segmentation must not guess: mark the trace as requiring
   review rather than silently labeling a guessed state as final.

Sentence punctuation (`.`, `?`, `!`) is a candidate boundary only when it ends
a sentence; decimal points, abbreviations, and punctuation inside formulas do
not create boundaries by themselves. Newlines are candidate boundaries, not
mandatory ones. These rules are applied in their listed order; an explicit
reasoning-step boundary takes precedence over sentence/newline grouping.

## Identifiers and token counts

For each source trace, assign states in source order, starting at zero for
`state_index`. The corresponding stable display identifier is `S{n}`, where
`n = state_index + 1` (`S1`, `S2`, ...). `trace_id` identifies the source generation and must be
stable across segmentation reruns; use the source `run_id` plus `problem_id`
(for example, `<run_id>:<problem_id>`). `problem_id` remains the dataset
identity.

`token_count` is the number of tokens in `text` under the exact model tokenizer
used by the experiment, encoded without adding special tokens. Record model and
tokenizer revisions in the containing artifact manifest so counts can be
reproduced. Do not substitute character count or an unrelated tokenizer.

## Semantic-state schema

`SemanticState` in `src/semantic_states/schema.py` is the canonical typed form:

```python
{
    "problem_id": "gsm8k-train-00053",
    "trace_id": "<run_id>:gsm8k-train-00053",
    "state_id": "S1",
    "state_index": 0,
    "text": "Calculate the total cost of the ice cream: 10 × 4 = 40.",
    "start_char": 0,
    "end_char": 58,
    "token_count": 14,
    "is_final_state": False
}
```

The schema validates field types, non-empty text, non-negative positions/counts,
and the `S{index+1}` identifier convention. Call `validate_trace_span()` to
check that offsets point to the exact source substring. A sequence must have
unique, contiguous indices and IDs, consistent problem/trace IDs, non-overlap,
source order, and exactly one final state when the trace has a recognized final
answer marker. Final-state/marker detection and coverage checks are part of the
segmentation pipeline, not state-value labeling.

## Exclusions and later work

Do not add labels such as useful, redundant, low-value, skippable, or
Black-Hole-positive to these records. Such labels require separate experiments
and belong to later tasks. A trace with missing/ambiguous final-answer
structure remains preserved in the source data but is flagged for review; its
semantic states must not be silently treated as segmentation-ready.
