# Week 3.1 server run: three samples, two correct to pass

Use the existing Ubuntu 24.04 / ROCm 7.2 environment and configured `gdrive` remote. The active tickets are in [3.1.md](../tickets/3.1.md). This runs the Q62 sentence procedure across every selected trace. It uses FP16, the pinned Qwen3 revision, three sampled answers per condition, at least two correct for sufficiency, and attainability >=0.8. It retains gradient and random comparisons.

## Update the server files

Transfer the updated checkout, or pull the changes after publishing them. Required Python files:

- `src/black_hole/sentence_checkpoint.py`
- `src/black_hole/sentence_experiment.py`
- `src/black_hole/run_sentence_experiments.py`
- `src/black_hole/run_experiments.py` (updated cache/revision resolver)
- `notebooks/12_sentence_removal_experiments.py`

The existing `src/utils/environment.py` and `src/utils/run_status.py` are also required. The old continuation runner remains a separate protocol.

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-rocm72/bin/activate
python -m pip check
python -m src.black_hole.run_sentence_experiments --help
rclone lsd gdrive:PSSN2
```

If the environment or remote is missing, follow [the original server setup](server_setup_rocm_7_2.md), including its read-only Drive mount. AMD's [versioned ROCm 7.2 Ubuntu instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/install-methods/package-manager/package-manager-ubuntu.html) remain applicable; this change does not need different Torch wheels.

## First run: two traces

`--limit 2` means two new completed or screened-out **traces**, each with its own three-sample conditions. Use the new output folder below.

```bash
tmux new -s pssn-week3-1
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-rocm72/bin/activate
python -m src.black_hole.run_sentence_experiments \
  --input /root/gdrive/PSSN2/artifacts/expanded_baseline/w2-001-700-attempts-seed44/black_hole_trace_selection/20261010T184131921450Z/eligible_traces.jsonl \
  --output-dir /root/pssn-work/checkpoints/w3-1-sentence-3samples-run1 \
  --drive-output gdrive:PSSN2/artifacts/black_hole/w3-1-sentence-3samples-run1 \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --limit 2
```

Equivalent entry file: replace `python -m src.black_hole.run_sentence_experiments` with `python notebooks/12_sentence_removal_experiments.py`, keeping all arguments.

Inspect `summary.json`, the per-question `summary.json` and `checkpoints/` in the new Drive folder. A trace excluded by screening is a valid audited outcome, not a failed server run. Measure elapsed time on several eligible traces before estimating all 251; exclusions finish faster.

## Run all remaining traces

```bash
python -m src.black_hole.run_sentence_experiments \
  --input /root/gdrive/PSSN2/artifacts/expanded_baseline/w2-001-700-attempts-seed44/black_hole_trace_selection/20261010T184131921450Z/eligible_traces.jsonl \
  --output-dir /root/pssn-work/checkpoints/w3-1-sentence-3samples-run1 \
  --drive-output gdrive:PSSN2/artifacts/black_hole/w3-1-sentence-3samples-run1 \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c
```

Detach with Ctrl+B, then D. Reconnect with `tmux attach -t pssn-week3-1`. Run only one process for a given output folder, including across servers. Ctrl+C or SIGTERM writes stopped status and flushes completed evidence. Rerun the same command to resume.

## Checkpointing and status

Each trial is saved atomically with its post-sampling RNG state. Resume restores that state, so completed samples are reused and the original seeded sequence continues. Each completed score and attribution matrix row is also checkpointed. Completed traces have a final marker; an interrupted trace replays its deterministic search with saved scores and rows, generating only missing work.

Drive backups are batched every 45 seconds (configurable with `--sync-seconds`) and uploaded by a background thread while model work continues. A trace is checkpointed locally first; the runner does not wait for Drive after each trace. Temporary upload failures retain and requeue local files, print a warning, and retry on the next interval. At session end, the runner waits for a final sync; if that fails, local results remain available and the run reports the sync failure. A hard kill or server loss can leave the newest unsynced work on local disk; it resumes locally if that disk survives. Another server restores only the last successful cloud backup.

The runner uses [rclone copy](https://rclone.org/commands/rclone_copy/) with a [file list](https://rclone.org/filtering/#files-from-read-list-of-source-file-names), uploading closed files without deleting older results. Changed protocol, source data, experiment-engine code, Torch or Transformers versions are rejected within an existing run. An uploader-only edit to `run_sentence_experiments.py` preserves the original run identity so saved checkpoints can resume; each session records the active code fingerprint. Use a new folder for changed experiments.

```bash
watch -n 10 'cat /root/pssn-work/checkpoints/w3-1-sentence-3samples-run1/run_status.json'
```

Drive status: `PSSN2/artifacts/black_hole/w3-1-sentence-3samples-run1/run_status.json`. The heartbeat is refreshed every 45 seconds; a heartbeat older than 120 seconds means the `is_running` flag may be stale. Upload duration can delay the cloud view. Counts in root progress are traces, while the current question reports completed conditions.

## Files for W3004 labeling

Under `PSSN2/artifacts/black_hole/w3-1-sentence-3samples-run1/`:

- `manifest.json`: fixed protocol, input hash, selected IDs, model/tokenizer revisions, software and code fingerprints.
- `source/eligible_traces.jsonl`: exact original input, including all original baseline data and representations.
- `sessions/*.json`: actual environment, repository commit, package versions and runtime identities per invocation.
- `<problem_id>/embedded_question_and_trace.json`: original thinking text, sentence IDs/offsets and overlaps with Week 2 semantic states.
- `<problem_id>/checkpoints/trials/*.json`: each answer, correctness/completion, prompt, timing, tokens, memory and resumable sampling state.
- `<problem_id>/checkpoints/evaluations/*.json`: condition masks, all answers, sufficiency, perplexity, attainability, calibration warnings and pass/fail.
- `<problem_id>/checkpoints/gradients/*.json`: completed attribution rows.
- `<problem_id>/summary.json`: screening, shortcuts, independent removal results, prefix search, gradient core and random comparisons.
- `<problem_id>/label_evidence.jsonl`: structured removal evidence with unit mappings and raw-trial references.
- `sentence_removal_evidence.jsonl`: combined evidence from completed traces, rebuilt at session end.
- `summary.json`: completed/pending traces, screening outcomes and evidence counts.

W3004 is the next task; these files contain evidence, not automatically assigned training labels. Independent original-trace deletions are diagnostics. Greedy evidence records the cleaned-prefix context in which a removal was tested. Label sentences explicitly; overlap with an original semantic state does not by itself give that whole state the same label. Three samples and the 2/3 rule are a documented adaptation of the authors' five-sample protocol.

The protocol discovers removable supplied reasoning text. It does not establish erasure of latent reasoning or faster unseen-question inference.
