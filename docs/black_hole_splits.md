# Week 3.006 Black Hole Validation and Splits

W3-006 validates the W3-005 dataset before Week 4 and splits by `problem_id`. Every sentence-context row from one GSM8K question stays in exactly one partition, preventing question-level leakage.

Run locally without a GPU:

```powershell
python -m src.data.black_hole_splits `
  --dataset PSSN/artifacts/black_hole/w3-1-sentence-3samples-run1/w3-005-dataset/black_hole_training_dataset.jsonl `
  --output-dir PSSN/artifacts/black_hole/w3-1-sentence-3samples-run1/w3-006-splits `
  --seed 42
```

The output includes `train.jsonl`, `validation.jsonl`, `test.jsonl`, one manifest per split, and `week3_quality_summary.json`.
