# Semantic-State Vector Representations

W2-005 represents only the validated Week-3-ready states. It uses the
unquantized `Qwen/Qwen3-4B` model and its matching tokenizer, loaded with the
shared environment policy (FP16 on a T4). No representation settings are
silently changed by the extraction utility.

## Context and pooling

For each trace, construct the Qwen chat-formatted original user prompt and the
complete assistant trace, then run one causal forward pass with hidden-state
output enabled. A causal model's hidden state for a token attends to preceding
context but not future tokens, so each state vector reflects the prompt and
reasoning prefix available at that point. Use the final transformer layer
(`layer=-1`) and deterministic mean pooling over token offsets overlapping each
state's character span. The pooling mask is found with the fast tokenizer's
offset mapping. Pooling is accumulated in FP32 and vectors are stored as FP32.

Model/tokenizer revisions are recorded when the loaded libraries expose commit
hashes. The selected layer, resolved layer index, hidden dimension, pooling,
context method, and output shape are stored in the representation config.

## Output format

The Colab notebook writes to a new timestamped directory next to the W2-004
validation run:

- `state_vectors.npy`: one row per state, in index-file order.
- `representation_index.jsonl`: maps each vector row to `problem_id`,
  `trace_id`, `state_id`, state index, layer, pooling, and token counts.
- `representation_config.json`: model, tokenizer, layer, pooling, runtime, and
  source artifact metadata.

Vectors remain separate from human-readable state records. The extraction
utility performs one model forward pass per trace and does not alter the source
records or validation artifacts.
