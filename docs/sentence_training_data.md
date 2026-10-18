# PSSN sentence training data

Run from the repository root, providing the local release directory:

```powershell
python -m src.data.sentence_training_data `
  --source-dir "research/results-from-other-research" `
  --limit 5000 `
  --output-dir "PSSN/artifacts/black_hole/sentence-training-5000-v2" `
  --source-output-dir "PSSN/artifacts/black_hole/sentence-training-5000-v2-source-notes"
```

This creates a balanced 5,000-row set (2,500 per class), with question-disjoint
70/15/15 train, validation, and test splits. The labels are transferred from the
release's binary removability target: `true` maps to `low_value_in_context` (1),
and `false` maps to `answer_sensitive_in_context` (0). This is not the PSSN 2-of-3
labeling rule and is not verified for Qwen3.

Training data and source documentation are written to separate sibling folders.
The model-facing dataset contains no row-level source crosswalk. The separate
source-notes folder records aggregate provenance and licensing context. Do not
publish the derived rows until redistribution rights are clear.

Output files:

- `sentence_training_data.jsonl`: all training-ready rows.
- `splits/train.jsonl`, `validation.jsonl`, `test.jsonl`: question-disjoint splits.
- `summary.json`, `dataset_manifest.json`: counts, configuration, and checksums.
- `excluded_rows.jsonl`: hashed IDs and reasons for rejected source records.

The sibling `sentence-training-5000-v2-source-notes/` folder contains
`data_source.json` (source-file inventory and aggregate counts) and
`DATA_SOURCE.md` (human-readable provenance and reuse notes). It has no row-ID
mapping back to individual training examples.

Activations are not extracted by this command. For later Qwen3 feature extraction,
use a Qwen3-formatted question followed by `reasoning_prefix` and `sentence_text`;
do not include future reasoning text or the reference answer in the feature context.

## Extract Qwen activations

The `splits/` files are already split by normalized question (70/15/15). The
activation command checks that the same question or row does not cross splits,
then writes two sets of `train.npz`, `validation.npz`, and `test.npz` in a format
accepted by the Week 4 classifier trainer: final-layer features at the output
root and all-layer features under `all_layers/`. Both are collected in one model
forward pass per context; no answer is generated.

Run from the repository root:

```powershell
python -m src.black_hole.sentence_activations_v2 `
  --dataset-dir "PSSN/artifacts/black_hole/sentence-training-5000-v2" `
  --output-dir "PSSN/artifacts/black_hole/sentence-training-5000-v2/activations-v2"
```

Add `--require-cuda` to require a GPU, or `--revision <commit-hash>` to pin the
Qwen checkpoint. The generated feature manifest records the model, revision,
feature shapes, input context, split counts, and hashes. The existing W3 activation
command remains unchanged; this command is specifically for the 5,000-row
sentence-training schema. V2 assigns and verifies question-disjoint splits before
activation extraction.
