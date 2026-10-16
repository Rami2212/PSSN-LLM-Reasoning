# Week 3 — Eligible trace selection

W3-001 prepares validated Week 2 semantic-state traces for controlled
single-state removal. It only selects traces whose baseline answer is correct,
whose segmentation and stored Week 2 quality report are valid, whose state
sequence revalidates against the original reasoning trace, and whose states
all link to vector representations. Required problem/trace identifiers,
question, answers, generation configuration, and model metadata must be
present. Ineligible records remain visible in the audit with stable reason
codes.

The final-answer state is retained in the baseline record but never appears in
`removable_state_ids`. Every selected record embeds an unchanged copy of the
source record under `baseline_result`; source data are not modified.

## Run

Run [`notebooks/10_black_hole_trace_selection.ipynb`](../notebooks/10_black_hole_trace_selection.ipynb)
after Week 2 notebook 09. The notebook reads the latest semantic-state dataset
under `MyDrive/PSSN2/artifacts/expanded_baseline/` and writes a new timestamped
`black_hole_trace_selection/` directory beside it. To keep all eligible traces,
leave `SAMPLE_SIZE = None`. Set a positive integer to take a seeded sample;
the default seed is 42. Re-running with the same input, seed, and sample size
produces the same selected trace IDs. Existing outputs are never overwritten.

## Output files

- `eligible_traces.jsonl`: selected traces, original baseline result, and
  removable non-final state IDs.
- `eligibility_audit.jsonl`: one eligibility decision and explicit reason code
  list for every input trace, including records sampled out.
- `selection_summary.json`: input/eligible/selected/excluded counts, seed,
  selection settings, selected IDs, source hash, and repository revision.

This step does not remove states or generate new model responses; those belong
to the next Week 3 ticket.
