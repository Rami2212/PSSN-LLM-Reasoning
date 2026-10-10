# Week 2 Semantic-State Dataset

W2-006 builds a machine-readable dataset from the completed baseline trace run,
the W2-004 Week-3-ready state sequences, their validation reports, and the
W2-005 representation index. Run
[`notebooks/09_build_semantic_state_dataset.ipynb`](../notebooks/09_build_semantic_state_dataset.ipynb)
after notebooks 07 and 08.

## Inclusion and integrity

Only complete state sequences marked valid and Week-3-ready are included. The
builder revalidates each sequence against its exact source reasoning trace,
checks state order, source spans and token counts, then requires a one-to-one
mapping to vector rows. Invalid/incomplete sequences are excluded whole; states
are never silently dropped to make a sequence pass. Input artifacts are read
only and output is written to a new timestamped directory.

## Output

- `semantic_states.jsonl`: one record per included problem/trace, retaining the
  original question, reference and predicted answers, correctness, complete
  reasoning trace, ordered states, validation report and baseline metadata.
- `semantic_state_metadata.json`: schema, source paths and hashes, seed,
  representation configuration, vector-file link and builder provenance.
- `semantic_state_summary.json`: included/excluded trace counts, state counts,
  token-count summary and representation dimensions.

Vectors are not copied into JSONL. Every state has a `representation` object
with its `vector_index`, extraction metadata and path to the separate
`state_vectors.npy`. No Black Hole or low/high-value labels are introduced.
Generation is deterministic and preserves baseline trace order and state order;
the recorded seed is provenance from the baseline experiment, not a sampling
operation in this builder.
