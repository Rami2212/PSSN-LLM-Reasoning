# Sentence activations V2

V2 keeps the Week 4 activation pipeline unchanged. It reads the consolidated
`sentence_training_data.jsonl` from the 5,000-row dataset, reassigns rows into
deterministic 70/15/15 splits grouped by normalized question text, and verifies
that questions and row IDs do not cross splits. For every target sentence, V2
captures final-layer and all-transformer-layer activations in the same forward
pass. It mean-pools target tokens; the all-layer representation concatenates
the pooled vectors from every transformer layer. The Qwen chat prompt contains
the question, earlier reasoning, and target sentence only; it does not generate
an answer or include the answer key/future sentences.

Run from the repository root:

```powershell
python -m src.black_hole.sentence_activations_v2 `
  --dataset-dir "PSSN/artifacts/black_hole/sentence-training-5000-v2" `
  --output-dir "PSSN/artifacts/black_hole/sentence-training-5000-v2/activations-v2" `
  --seed 42
```

Add `--require-cuda` to require a CUDA GPU. Output contains question-disjoint
split JSONLs, plus two trainer-compatible feature sets:

- At the output root: final-layer `train.npz`, `validation.npz`, `test.npz`.
- Under `all_layers/`: all-layer versions of the same three files.

Each NPZ has `features`, `labels`, `row_ids`, and `group_ids`, so the current
Week 4 trainer can use either set without code changes. Train the final-layer
version with:

```powershell
python -m src.black_hole.trainer `
  --data-dir "PSSN/artifacts/black_hole/sentence-training-5000-v2/activations-v2" `
  --output-dir "PSSN/artifacts/black_hole/sentence-training-5000-v2/training-final-layer"
```

For all layers, use `.../activations-v2/all_layers` as `--data-dir` and a
different training output directory. The manifest records dimensions, split
counts, hashes, model revision, and settings. All-layer files are much larger;
choose a new output directory if rerunning because V2 does not overwrite data.

The two feature sets use identical rows and labels. They train separate
predictors; they are not concatenated into one. Transferred labels have not been
independently validated for this activation model, so compare both predictors
on the same held-out questions.

## Test the trained V2 predictor

After training, choose a threshold from the validation split only, then use the
V2 row-level evaluator on the held-out test split. The example uses a
predeclared maximum 10% validation false-skip rate. This is a safety/coverage
trade-off: verify validation recall remains useful before proceeding. Run the
selection separately for each model:

```bash
python -m src.black_hole.sentence_threshold_v2 \
  --data-dir "PSSN/artifacts/black_hole/sentence-training-5000-v2/activations-v2" \
  --checkpoint "PSSN/artifacts/black_hole/sentence-training-5000-v2/training-final-layer/best_model.pt" \
  --split-jsonl "PSSN/artifacts/black_hole/sentence-training-5000-v2/activations-v2/validation.jsonl" \
  --output "PSSN/artifacts/black_hole/sentence-training-5000-v2/evaluation-final-layer/threshold_selection.json" \
  --max-false-skip-rate 0.10

python -m src.black_hole.sentence_threshold_v2 \
  --data-dir "PSSN/artifacts/black_hole/sentence-training-5000-v2/activations-v2/all_layers" \
  --checkpoint "PSSN/artifacts/black_hole/sentence-training-5000-v2/training-all-layers/best_model.pt" \
  --split-jsonl "PSSN/artifacts/black_hole/sentence-training-5000-v2/activations-v2/validation.jsonl" \
  --output "PSSN/artifacts/black_hole/sentence-training-5000-v2/evaluation-all-layers/threshold_selection.json" \
  --max-false-skip-rate 0.10
```

Inspect `validation_metrics_at_selected_threshold`; if the low-value recall is
near zero, the chosen safety cap leaves too little useful coverage. Freeze the
threshold before running test evaluation:

```bash
python -m src.black_hole.sentence_evaluation_v2 \
  --data-dir "PSSN/artifacts/black_hole/sentence-training-5000-v2/activations-v2" \
  --checkpoint "PSSN/artifacts/black_hole/sentence-training-5000-v2/training-final-layer/best_model.pt" \
  --split-jsonl "PSSN/artifacts/black_hole/sentence-training-5000-v2/activations-v2/test.jsonl" \
  --output-dir "PSSN/artifacts/black_hole/sentence-training-5000-v2/evaluation-final-layer" \
  --threshold-selection "PSSN/artifacts/black_hole/sentence-training-5000-v2/evaluation-final-layer/threshold_selection.json"

python -m src.black_hole.sentence_evaluation_v2 \
  --data-dir "PSSN/artifacts/black_hole/sentence-training-5000-v2/activations-v2/all_layers" \
  --checkpoint "PSSN/artifacts/black_hole/sentence-training-5000-v2/training-all-layers/best_model.pt" \
  --split-jsonl "PSSN/artifacts/black_hole/sentence-training-5000-v2/activations-v2/test.jsonl" \
  --output-dir "PSSN/artifacts/black_hole/sentence-training-5000-v2/evaluation-all-layers" \
  --threshold-selection "PSSN/artifacts/black_hole/sentence-training-5000-v2/evaluation-all-layers/threshold_selection.json"
```

Outputs are `test_evaluation_summary.json` and `test_predictions.jsonl`.
Metrics include accuracy, precision/recall/F1 for low-value sentences, ROC-AUC,
PR-AUC, confusion matrix, and answer-sensitive false-skip rate. The test rows
are question-disjoint from training and validation. The threshold was selected
on validation only; test results do not tune it. This is held-out evaluation
within the supplied dataset, not an external-domain test; transferred labels
have not been independently validated for Qwen3.

For the complete MI300X ROCm 7.2 and Google Drive workflow, see
[`mi300x_sentence_activations_v2_runbook.md`](mi300x_sentence_activations_v2_runbook.md).
