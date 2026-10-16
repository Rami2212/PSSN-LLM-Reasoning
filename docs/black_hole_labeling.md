# Week 3.004 Black Hole Sentence-Context Labels

W3-004 labels removal **experiments**, not hidden model computation. A row identifies a reasoning sentence and the exact retained context mask used when that sentence was tested.

The main training source is `gradient_greedy` evidence from a trace that passed screening and whose final shortened core passed validation. Independent deletions and random-pruning results remain diagnostic/audit evidence. Traces whose final core failed validation are retained in the audit export.

| Label | Meaning |
|---|---|
| `low_value_in_context` | At least 2 of 3 answers remained correct, attainability was at least 0.8, and calibration was valid. |
| `answer_sensitive_in_context` | Fewer than 2 of 3 answers remained correct. |
| `plausibility_rejected` | The answer-count rule passed but attainability was below 0.8. |
| `unscored` | Calibration or required evaluation data was invalid. |

The W3-004 command is CPU-only:

```powershell
python -m src.black_hole.labeling `
  --input-dir PSSN/artifacts/black_hole/w3-1-sentence-3samples-run1 `
  --output-dir PSSN/artifacts/black_hole/w3-1-sentence-3samples-run1/w3-004-labels
```

It writes accepted training labels, plausibility-rejected records, unscored records, an audit export, a summary and a manifest. Rejected records do not enter the W3-005 training dataset.
