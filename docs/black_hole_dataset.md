# Week 3.005 Black Hole Training Dataset

W3-005 converts accepted W3-004 labels into one row per **sentence-context removal experiment**. The same sentence may occur more than once if it was tested in different retained contexts; this is intentional and the context masks remain part of the row.

Each row links to the corresponding Week 2 semantic-state metadata and vector reference. The vector is not copied into the W3-005 JSONL. A sentence label is not automatically assigned to an entire overlapping semantic state.

Run locally without a GPU:

```powershell
python -m src.data.black_hole_dataset `
  --labels PSSN/artifacts/black_hole/w3-1-sentence-3samples-run1/w3-004-labels/black_hole_sentence_labels.jsonl `
  --state-dataset PSSN/artifacts/expanded_baseline/w2-001-700-attempts-seed44/semantic_state_dataset/20261010T145731106290Z/semantic_states.jsonl `
  --output-dir PSSN/artifacts/black_hole/w3-1-sentence-3samples-run1/w3-005-dataset
```

The output folder contains the training JSONL, excluded-row audit, dataset summary and manifest. W3-006 will validate these records and split them by question.
