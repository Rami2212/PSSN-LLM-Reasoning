# W3003 — State-removal continuation experiments

The runner creates one independent experiment for every W3001 removable state.
It applies W3002 to the original validated states, reuses the baseline user
prompt, and extends an unfinished Qwen assistant turn with the surviving
non-final states as context. The saved final-answer state is excluded from
model input. W3002 still retains all surviving states in its removal record;
the additional `continuation_context` field is the context used for generation.

This protocol retains later surviving reasoning states; it measures deletion
from a stored reasoning context. It does not measure an online policy that
regenerates every later state after skipping. Later states may retain information
computed by the removed state, so final-answer preservation is evidence of
redundancy in this context, not proof that the state was unnecessary at the
moment it was originally generated. The final state may itself contain reasoning;
its entire text is excluded consistently from all model inputs.

Only newly generated text is evaluated with the Week 1 answer extractor. The
reference answer and saved baseline answer are never supplied as prompt fields.
Length-capped continuations remain unscored. Input token counts include the user
prompt and supplied reasoning context; output counts include only new tokens.
Input/output counts must be reported separately rather than claiming total
compute savings from fewer output tokens alone.

Each result links the original problem, baseline trace and removed state. It
preserves removal metadata, baseline results, model/generation settings,
continuation, predicted answer, correctness, tokens, latency and memory metrics.
Failures remain explicit, unscored records for later label processing.

Checkpoint records are written and flushed locally, then synchronously copied
to Drive. The manifest blocks resuming a run with different data or settings.
Generation interruptions and upload failures are recoverable from the saved
per-experiment records; the in-progress computation may need to be repeated.

Run notebook 11 or the server entry point as described in
[the server setup document](server_setup_rocm_10_1.md). Labels are assigned in
W3004, after reviewing completion and failure counts.
