# Semantic-State Quality Validation

`src.semantic_states.validation` checks segmented records before their use in
state-removal experiments. It reports issues without changing, joining,
deleting, or otherwise rewriting the input state records.

## Checks and defaults

- Required, sequential state identifiers (`S1`, `S2`, ...; `state_index` starts
  at zero).
- Non-empty text, valid non-overlapping character ranges, and an exact match to
  the source trace when that trace is supplied.
- Required non-negative integer tokenizer counts.
- Duplicate adjacent text, compared after whitespace collapse and
  case-folding.
- Extremely short fragments: fewer than 3 Qwen tokenizer tokens are an error.
- At least 2 usable states per trace for Week 3 state-removal work.
- States longer than 512 tokens receive a non-blocking manual-review warning.

Thresholds are explicit function arguments and are persisted in the validation
summary. A trace is valid when it has no error-level issues. Long-state warnings
do not invalidate a trace, but remain visible for human review. Invalid traces
are excluded as whole sequences; the validator never drops an individual state
from a sequence.

`validate_state_sequence` returns a per-trace report. For a flat JSONL file,
`validate_semantic_state_records` groups records by `trace_id`, returns all
reports, and selects unchanged records from valid Week-3-ready sequences in
`week3_ready_states`.
