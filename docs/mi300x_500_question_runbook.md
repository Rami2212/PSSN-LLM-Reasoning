# MI300X: build a 500-question Black Hole dataset, with manual Drive backups

Run the numbered sections in order. All computation runs on the Ubuntu 24.04 / ROCm 7.2 MI300X server. Clone the updated repository on the server and copy the existing PSSN2 data from Drive. Your PC is used for browser authorization and downloading results.

The workflow is: **clone the updated repo -> copy PSSN2 from Drive -> generate new questions/traces -> create and validate new states -> combine old and new state datasets -> split only new traces into sentences -> experiment on and label only those new sentences -> combine their labels with the saved old labels -> extract fresh activations for all 500 questions -> train and validate the predictor**. Run the manual backup commands after each important stage.

**Target: 500 distinct usable questions after labeling**, with all accepted sentence-context rows for those questions. This is more than 500 rows or 500 generation attempts. Your pilot has 99 usable questions / 1,008 rows. This guide counts those 99 if the copied pilot dataset is available, then adds new questions until the combined pool reaches at least 500. All new generation and preparation, including the final activation extraction, run on this server.

At the pilot yield of 99 / 700, approximately 2,835 **additional raw attempts** might provide the remaining 401 questions. This is a planning estimate; batch yields can differ. Process batches of 500 raw questions and check the actual count after each batch. Do not relax label checks to reach the target. The available unused GSM8K training pool is finite; the script reports if it runs out.

The fixed setup remains Qwen/Qwen3-4B at revision `1cfa9a7208912126459214e8b04321603b3df60c`, FP16 on ROCm, a 1,024-token baseline cap, three answer samples, 2/3 sufficiency, and attainability >= 0.8. Generation attempts that hit the cap remain in the raw data and are excluded from successful-trace preparation.

## 1. Connect and prepare the server

Use a server image with **ROCm 7.2 on Ubuntu 24.04** already installed. These commands assume the root account you used before.

```bash
rocm-smi
apt update
apt install -y git python3.12-venv python3.12-dev rclone tmux zstd
mkdir -p /root/pssn-work
cd /root/pssn-work
```

If the checkout is absent, clone it:

```bash
git clone https://github.com/Rami2212/PSSN-LLM-Reasoning.git
```

Use the updated GitHub repository as the code source. A fresh clone is the normal path. If a clean checkout already exists, update it before starting the campaign:

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
git pull --ff-only
```

If Git reports local changes, preserve that checkout and clone into a separate directory instead of resetting it. The version cloned must include the current `server_campaign.py` stages `merge-states` and `sentences`, plus local-only sentence experiments. If you are reading this guide from an unpushed working copy, commit/push its updated files before cloning them on the server.

Back on the **server**:

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
python3.12 -m venv .venv-rocm72
source .venv-rocm72/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-rocm-7.2.txt
python -m pip check
python -c 'import torch; assert str(torch.version.hip).startswith("7.2."), torch.version.hip; assert torch.cuda.is_available(); print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))'
python -m src.data.server_campaign --help
python -m src.black_hole.run_sentence_experiments --help
```

If the environment already works, activate it and run the checks; reinstalling is unnecessary. Keep the same installed package versions and source files throughout a batch, including its removal experiments. Baseline and removal checkpoints reject changed experiment identities on resume.

AMD's [Ubuntu ROCm 7.2 installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/install-methods/package-manager/package-manager-ubuntu.html) covers a missing system installation. The project requirements retain the previously used explicit ROCm Torch wheels. AMD's [Instinct PyTorch guidance](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/3rd-party/pytorch-install.html) describes supported framework installation options.

## 2. Configure Google Drive

On the **server**:

```bash
rclone config
```

Choose these answers:

| Prompt | Answer |
|---|---|
| New remote | `n` |
| Name | `gdrive` |
| Storage | `drive` (Google Drive) |
| Client ID | Enter |
| Client secret | Enter |
| Scope | `1` (full access) |
| Service account file | Enter |
| Edit advanced config? | `n` |
| Use browser / auto config? | `n` |

Leave the server waiting at `config_token>`. On your **Windows PC**, install rclone if needed:

```powershell
winget install --id Rclone.Rclone -e
```

Open a new PowerShell window after installation. Run the exact `rclone authorize ...` command printed by the server. With the full-access scope it commonly looks like:

```powershell
rclone authorize "drive" "eyJzY29wZSI6ImRyaXZlIn0"
```

Sign in through the browser. Copy the complete returned token JSON into the server's `config_token>` prompt. Answer `n` to Shared Drive unless you use one, confirm with `y`, then quit with `q`. Keep authorization tokens out of research logs and archives.

Check the connection on the **server**:

```bash
rclone lsd gdrive:
rclone lsd gdrive:PSSN2
```

If `gdrive` is already configured and both commands succeed, keep that configuration. See [rclone headless authorization](https://rclone.org/remote_setup/).

## 3. Copy PSSN2 from Drive onto the server

Work from the downloaded local files. No Drive mount is needed.

```bash
mkdir -p /root/pssn-work/PSSN2
rclone copy gdrive:PSSN2 /root/pssn-work/PSSN2 --progress --stats 10s --transfers 4 --checkers 4 --timeout 2m --contimeout 30s --retries 3
rclone check gdrive:PSSN2 /root/pssn-work/PSSN2 --one-way
df -h /root/pssn-work
```

Wait for the copy and check to finish. If Drive contains only the old experiment ZIP, extract it locally:

```bash
cd /root/pssn-work/PSSN2/artifacts/black_hole
ls -lh w3-1-sentence-3samples-run1.zip
python3 -m zipfile -l w3-1-sentence-3samples-run1.zip | head -n 15
```

If the expected `w3-1-sentence-3samples-run1` directory is absent and the listing shows that directory inside the ZIP:

```bash
python3 -m zipfile -e w3-1-sentence-3samples-run1.zip /root/pssn-work/PSSN2/artifacts/black_hole
```

Do not extract over a populated run directory. The original ZIP might predate your locally created labels, dataset, and splits. Check:

```bash
find /root/pssn-work/PSSN2 -name black_hole_training_dataset.jsonl -print
find /root/pssn-work/PSSN2 -path '*/w3-004-labels/black_hole_sentence_labels.jsonl' -print
find /root/pssn-work/PSSN2 -path '*/semantic_state_dataset/*/semantic_states.jsonl' -print
find /root/pssn-work/PSSN2 -name state_vectors.npy -print
```

To count the existing 99 questions, the following file and its parent run's per-question evidence folders must be present:

```text
/root/pssn-work/PSSN2/artifacts/black_hole/w3-1-sentence-3samples-run1/w3-005-dataset/black_hole_training_dataset.jsonl
```

Before continuing, verify that the Drive copy includes the original `w3-004-labels`, `w3-005-dataset`, the per-question experiment evidence, and the Week 2 `expanded_baseline` state dataset/vector files. A ZIP made before labeling may lack those processed folders. Restore the missing folders from your existing backups before generating new data; this workflow reuses the old labels and does not recreate them.

## 4. Start a persistent session and set the paths

On the **server**, start tmux once:

```bash
tmux new -s pssn-500
```

If it says the session exists, use `tmux attach -t pssn-500`. If already inside tmux, keep using that session instead of nesting another one.

Run this setup block inside tmux, and again after reconnecting to a new shell:

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-rocm72/bin/activate
set -o pipefail
export PSSN_DATA=/root/pssn-work/PSSN2
export CAMPAIGN="$PSSN_DATA/artifacts/black_hole_500_v1"
export BATCH=batch-001
export BATCH_DIR="$CAMPAIGN/batches/$BATCH"
export PILOT_DATASET="$PSSN_DATA/artifacts/black_hole/w3-1-sentence-3samples-run1/w3-005-dataset/black_hole_training_dataset.jsonl"
export PILOT_LABELS="$PSSN_DATA/artifacts/black_hole/w3-1-sentence-3samples-run1/w3-004-labels/black_hole_sentence_labels.jsonl"
export EXISTING_STATES="$PSSN_DATA/artifacts/expanded_baseline/w2-001-700-attempts-seed44/semantic_state_dataset/20261010T145731106290Z/semantic_states.jsonl"
export BACKUPS=/root/pssn-work/backups/black_hole_500_v1
mkdir -p "$BATCH_DIR/logs" "$BACKUPS"
PILOT_ARGS=(--include-existing "$PILOT_DATASET")
python -c 'import os; from pathlib import Path; paths=[os.environ[k] for k in ("PILOT_DATASET","PILOT_LABELS","EXISTING_STATES")]; assert all(Path(p).is_file() for p in paths), paths; print("Existing datasets and labels found")'
python -m src.data.server_campaign count --campaign "$CAMPAIGN" "${PILOT_ARGS[@]}"
```

Both checks must succeed. Expected initial count: `usable_questions: 99`. If the timestamps differ in your copy, set `EXISTING_STATES` to the trace-level Week 2 file listed by the `find` command above. Do not point it at flat segmentation rows or the labeled sentence dataset.

The new campaign does not overwrite the pilot. Earlier raw attempts and samples already reserved in other campaign batches are excluded from new generation, so each added question is distinct. Only the official GSM8K **training** split supplies new questions; its official test split is preserved separately in the source snapshot.

Optional authenticated HF downloads, without putting the token in command history:

```bash
read -rsp 'HF token: ' HF_TOKEN
export HF_TOKEN
printf '\n'
```

### Manual backup command used after important stages

Define this shell function in the same tmux shell. It makes one compressed archive of the current batch, uploads it, then checks it. Run it only after a stage has stopped writing files.

```bash
backup_batch() {
  local snapshot_label="$1"
  local archive_name="${BATCH}-${snapshot_label}.tar.zst"
  tar -I 'zstd -T0 -3' -cf "$BACKUPS/$archive_name" -C "$CAMPAIGN" campaign_configuration.json "batches/$BATCH" || return
  (cd "$BACKUPS" && sha256sum "$archive_name" > "$archive_name.sha256") || return
  rclone copyto "$BACKUPS/$archive_name" "gdrive:PSSN2/artifacts/black_hole_500_v1/backups/$archive_name" --progress --stats 10s --timeout 2m --contimeout 30s --retries 3 || return
  rclone copyto "$BACKUPS/$archive_name.sha256" "gdrive:PSSN2/artifacts/black_hole_500_v1/backups/$archive_name.sha256" --progress || return
  rclone check "$BACKUPS" gdrive:PSSN2/artifacts/black_hole_500_v1/backups --one-way --include "$archive_name" --include "$archive_name.sha256" || return
  printf 'BACKUP VERIFIED: %s\n' "$archive_name"
}
```

Wait for `BACKUP VERIFIED` before continuing. These are cumulative snapshots, so later archives contain earlier stages too; they use additional disk and Drive space. `rclone copyto` preserves the server files. It does not move or delete them. If an upload fails, run the same backup command again while the source is idle. Archives avoid uploading tens of thousands of individual trial files.

## 5. Generate the first batch: 500 raw questions (W2-001/002)

```bash
python -u -m src.data.server_campaign baseline --campaign "$CAMPAIGN" --batch "$BATCH" --attempts 500 --exclude-root "$PSSN_DATA" 2>&1 | tee "$BATCH_DIR/logs/01-baseline.log"
```

This saves a fixed question sample, the normalized source train/test datasets, per-question raw records, generation settings, code hashes, the pinned revision, package versions, prompts, runtime/GPU configuration, correctness, token counts and timing. Baseline generation uses the same fixed prompt as your earlier Week 2 pipeline. A completed raw record is saved before moving to the next question.

The batch ends with `COMPLETED: baseline`. Inspect:

```bash
cat "$BATCH_DIR/baseline/trace_validation_summary.json"
cat "$BATCH_DIR/baseline/run_status.json"
backup_batch 01-baseline
```

`number_of_attempts` should be 500. `segmentation_eligible_correct_traces` is the number that can proceed, not the final usable-question count.

The first batch also creates the complete source dataset snapshot. Back it up once:

```bash
tar -I 'zstd -T0 -3' -cf "$BACKUPS/source_dataset.tar.zst" -C "$CAMPAIGN" source_dataset campaign_configuration.json
rclone copyto "$BACKUPS/source_dataset.tar.zst" gdrive:PSSN2/artifacts/black_hole_500_v1/backups/source_dataset.tar.zst --progress
rclone check "$BACKUPS" gdrive:PSSN2/artifacts/black_hole_500_v1/backups --one-way --include source_dataset.tar.zst
```

## 6. Segment the successful traces into states (W2-003)

```bash
python -u -m src.data.server_campaign segment --campaign "$CAMPAIGN" --batch "$BATCH" 2>&1 | tee "$BATCH_DIR/logs/02-segment.log"
backup_batch 02-segment
```

Outputs: `segmentation/semantic_states.jsonl` and `segmentation/review.jsonl`. Segmentation uses code and tokenizer offsets.

## 7. Validate the state sequences (W2-004)

```bash
python -u -m src.data.server_campaign validate --campaign "$CAMPAIGN" --batch "$BATCH" 2>&1 | tee "$BATCH_DIR/logs/03-validate.log"
cat "$BATCH_DIR/validation/validation_summary.json"
backup_batch 03-validate
```

Only complete valid sequences go into `validation/week3_ready_states.jsonl`. Review warnings and exclusion reports are kept.

## 8. Extract the state vectors (W2-005)

```bash
python -u -m src.data.server_campaign represent --campaign "$CAMPAIGN" --batch "$BATCH" 2>&1 | tee "$BATCH_DIR/logs/04-represent.log"
backup_batch 04-represent
```

Outputs: `representations/state_vectors.npy`, `representation_index.jsonl`, and `representation_config.json`. These are the Week 2 coarser-state vectors. W4 sentence-context activations are extracted later from the actual removal contexts.

## 9. Build new states, combine them with existing states, then select only new traces

```bash
python -u -m src.data.server_campaign state-dataset --campaign "$CAMPAIGN" --batch "$BATCH" 2>&1 | tee "$BATCH_DIR/logs/05-state-dataset.log"
python -u -m src.data.server_campaign merge-states --campaign "$CAMPAIGN" --batch "$BATCH" --existing-states "$EXISTING_STATES" --state-root "$PSSN_DATA" 2>&1 | tee "$BATCH_DIR/logs/05b-merge-states.log"
cat "$BATCH_DIR/combined-states/summary.json"
python -u -m src.data.server_campaign select --campaign "$CAMPAIGN" --batch "$BATCH" 2>&1 | tee "$BATCH_DIR/logs/06-select.log"
cat "$BATCH_DIR/selection/selection_summary.json"
backup_batch 05-selected
```

The files have different purposes:

| File | Contains | Used for new sentence experiments? |
|---|---|---|
| `state-dataset/semantic_states.jsonl` | Only this batch's new traces/states | Yes |
| `combined-states/semantic_states.jsonl` | Original Week 2 state dataset plus every prepared new batch | No; retained as the combined state archive |
| `selection/eligible_traces.jsonl` | Eligible traces from this new batch only | Yes; this is the experiment input |

Combining states preserves each state's original trace identity, text, vector file and vector row reference. It does not assign labels or reprocess the old traces. A new combined snapshot is stored per batch. The original state dataset may contain 251 traces, while only 99 of its questions supplied accepted labels; the combined state archive and the labeled training pool therefore have different counts.

The selection file is fixed for this batch. Do not append old traces or additional questions after the removal run starts.

## 10. Extract only the new sentences, then run their three-sample experiments

Save the new sentence inventory first:

```bash
python -u -m src.data.server_campaign sentences --campaign "$CAMPAIGN" --batch "$BATCH" 2>&1 | tee "$BATCH_DIR/logs/06b-sentences.log"
cat "$BATCH_DIR/sentences/summary.json"
backup_batch 05b-new-sentences
```

`sentences/new_sentences.jsonl` contains only the new selected traces' sentence units, original character offsets, and overlap mappings to their semantic states. Sentences are split from the original generated reasoning text using the same code as the experiment engine. The states and sentences are two segmentations of that text. No extra model generation is used to split it.

First, run two traces as a check:

```bash
python -u -m src.black_hole.run_sentence_experiments --input "$BATCH_DIR/selection/eligible_traces.jsonl" --output-dir "$BATCH_DIR/experiments" --revision 1cfa9a7208912126459214e8b04321603b3df60c --limit 2 2>&1 | tee "$BATCH_DIR/logs/07-experiment-check.log"
cat "$BATCH_DIR/experiments/summary.json"
```

Then continue the same folder without a limit:

```bash
python -u -m src.black_hole.run_sentence_experiments --input "$BATCH_DIR/selection/eligible_traces.jsonl" --output-dir "$BATCH_DIR/experiments" --revision 1cfa9a7208912126459214e8b04321603b3df60c 2>&1 | tee "$BATCH_DIR/logs/08-experiments.log"
```

There is deliberately **no `--drive-output` argument**. The updated runner saves local checkpoints and starts no uploader. It cannot become stuck waiting for a final Drive upload in this mode.

The protocol still performs necessity and shortcut checks, prefix search, gradient attribution, greedy core search, independent-deletion diagnostics, and random comparisons. It keeps the three-sample / 2-of-3 rule. A screened-out question is an audited outcome. It is not forced into training to meet the target.

Check completion:

```bash
cat "$BATCH_DIR/experiments/run_status.json"
python -c 'import json,os; p=os.environ["BATCH_DIR"]+"/experiments/summary.json"; s=json.load(open(p)); print({k:s[k] for k in ("total_traces","completed_traces","pending_traces","screened_eligible","validated_cores")}); assert s["pending_traces"] == 0'
backup_batch 06-experiments
```

Proceed only when `pending_traces` is zero. `validated_cores` is useful progress information; the definitive usable count comes from the accepted labeling/dataset rows below.

## 11. Assign labels (W3-004)

```bash
python -u -m src.black_hole.labeling --input-dir "$BATCH_DIR/experiments" --output-dir "$BATCH_DIR/labels" 2>&1 | tee "$BATCH_DIR/logs/09-labels.log"
cat "$BATCH_DIR/labels/labeling_summary.json"
backup_batch 07-labels
```

Only accepted gradient-greedy sentence-context evidence supplies labels. Plausibility-rejected, unscored, excluded-question and diagnostic evidence stays available for audit. This command reads **only this new batch's** experiment directory. The existing `PILOT_LABELS` file is reused later without running its questions through the labeling procedure again.

If `training_label_rows` is zero, this batch contributes zero usable questions. Preserve its audit and continue with a new batch; do not run the non-empty dataset builder below for that batch.

## 12. Build this batch's training rows and count the pool (W3-005)

```bash
python -u -m src.data.black_hole_dataset --labels "$BATCH_DIR/labels/black_hole_sentence_labels.jsonl" --state-dataset "$BATCH_DIR/state-dataset/semantic_states.jsonl" --output-dir "$BATCH_DIR/dataset" 2>&1 | tee "$BATCH_DIR/logs/10-dataset.log"
cat "$BATCH_DIR/dataset/black_hole_training_dataset_summary.json"
python -m src.data.server_campaign count --campaign "$CAMPAIGN" "${PILOT_ARGS[@]}"
backup_batch 08-dataset
```

The pool counter reads accepted datasets, counts each question once, and rejects overlapping source runs or incompatible model revisions/protocols. One question can supply many labeled rows. **Stop adding batches when `usable_questions` reaches at least 500.**

## 13. Add batches until 500 usable questions are available

If fewer than 500 are available, change only the batch name:

```bash
export BATCH=batch-002
export BATCH_DIR="$CAMPAIGN/batches/$BATCH"
mkdir -p "$BATCH_DIR/logs"
```

Repeat **sections 5 through 12**. For subsequent batches use `batch-003`, `batch-004`, etc. Keep `CAMPAIGN`, `EXISTING_STATES` and `PILOT_ARGS` unchanged. Each state-merge snapshot includes old states plus prepared new batches, while each sentence experiment still reads only its own batch's selection. The batch backup function automatically uses the current `BATCH`.

Use 500 raw attempts per batch by default. Near the target, a smaller **new** batch can use `--attempts 100`, for example. If resuming that smaller batch, keep `--attempts 100`; a saved sample cannot be enlarged in place. Completed baseline/removal records resume in their original batch. New input questions always get a new batch and experiment identity.

Manual uploads happen only when you invoke the backup commands. Until a backup finishes, new results exist on the server disk. A browser disconnection leaves tmux running; destruction of the server/disk loses unbacked-up files.

## 14. Combine old and new labels and freeze the 500-question training dataset

After the pool counter reports at least 500:

```bash
python -u -m src.data.server_campaign finalize --campaign "$CAMPAIGN" --target 500 --state-root "$PSSN_DATA" "${PILOT_ARGS[@]}"
cat "$CAMPAIGN/final/dataset_summary.json"
```

The finalizer preserves **all existing labeled questions** supplied through `PILOT_ARGS` (currently 99) and uses a fixed seed to choose the remaining questions from the new accepted pool. It includes every accepted row from each selected question, for exactly 500 questions total. Selection does not inspect predictor performance or prefer a label class. Surplus new usable questions remain in their source batches.

It copies the original accepted label records from old and new label files into one combined label file. It checks that each label matches its training row's question, trace and class. Old labels are neither regenerated nor re-scored.

The finalizer combines and remaps the Week 2 vectors so different batch-local vector indices cannot accidentally refer to another batch's vectors. Old Colab or Windows paths containing `PSSN/` or `PSSN2/` are resolved against the copied local PSSN2. A missing vector file raises an error rather than silently dropping the coarser baseline.

Outputs under `$CAMPAIGN/final/`:

| File/folder | Purpose |
|---|---|
| `black_hole_sentence_labels.jsonl` | Original old labels plus newly produced labels for the selected 500 questions |
| `black_hole_training_dataset.jsonl` | All accepted sentence-context rows from exactly 500 questions |
| `dataset_summary.json` | Counts, selected IDs, label distribution, source hashes, vector hashes and source evidence paths |
| `state_vectors.npy` | Combined Week 2 vectors with remapped row indices |
| `evidence/<problem_id>/` | Copies of the three per-question files needed for W4 activation extraction |

The `final/evidence/` directory contains activation inputs. Full sampled answers, trial checkpoints, attribution matrices and audit logs remain in the original pilot and batch experiment directories. Preserve their backups too.

If interrupted during finalization, the inputs remain safe. Preserve the incomplete output by renaming it, then rerun finalization:

```bash
mv "$CAMPAIGN/final" "$CAMPAIGN/final-interrupted-$(date -u +%Y%m%dT%H%M%SZ)"
```

Only use that rename for an unfinished finalization. Once finalized and split, treat this dataset as a frozen version.

## 15. Create question-level train/validation/test splits (W3-006)

```bash
python -u -m src.data.black_hole_splits --dataset "$CAMPAIGN/final/black_hole_training_dataset.jsonl" --output-dir "$CAMPAIGN/final/splits" --seed 42
cat "$CAMPAIGN/final/splits/week3_quality_summary.json"
```

Expect **350 / 75 / 75 questions** for train / validation / test, with zero question leakage. Row counts depend on the number of accepted contexts per question. Keep the split files and their manifests together.

These are new versioned splits. Do not combine a previous 99-question test score with the new test score or claim previously inspected pilot examples are a completely untouched benchmark. If you require a strictly untouched final evaluation, reserve new question IDs before collecting/inspecting results and design that holdout separately.

Back up the frozen dataset and splits manually:

```bash
tar -I 'zstd -T0 -3' -cf "$BACKUPS/final-dataset-and-splits.tar.zst" -C "$CAMPAIGN" final campaign_configuration.json
rclone copyto "$BACKUPS/final-dataset-and-splits.tar.zst" gdrive:PSSN2/artifacts/black_hole_500_v1/backups/final-dataset-and-splits.tar.zst --progress --stats 10s --timeout 2m --retries 3
rclone check "$BACKUPS" gdrive:PSSN2/artifacts/black_hole_500_v1/backups --one-way --include final-dataset-and-splits.tar.zst
```

## 16. Extract training sentence activations (W4-001)

```bash
mkdir -p "$CAMPAIGN/final/w4-001-activations"
python -u -m src.black_hole.dataset --split-dir "$CAMPAIGN/final/splits" --evidence-dir "$CAMPAIGN/final/evidence" --output-dir "$CAMPAIGN/final/w4-001-activations" --state-vectors "$CAMPAIGN/final/state_vectors.npy" --revision 1cfa9a7208912126459214e8b04321603b3df60c --require-cuda 2>&1 | tee "$CAMPAIGN/final/w4-001-activations/run.log"
cat "$CAMPAIGN/final/w4-001-activations/dataset_manifest.json"
```

This produces the primary sentence-context activation features and the separate optional coarser-state baseline features. The sentence features use retained preceding context plus the target sentence, matching the labeled context and excluding future sentences.

The current W4 extractor saves its split files at the end; it does not persist its in-memory context cache. If interrupted, rerun this extraction command. Baseline generation and W3 trials have finer-grained resume checkpoints.

Back up after successful extraction:

```bash
tar -I 'zstd -T0 -3' -cf "$BACKUPS/final-with-activations.tar.zst" -C "$CAMPAIGN" final campaign_configuration.json
rclone copyto "$BACKUPS/final-with-activations.tar.zst" gdrive:PSSN2/artifacts/black_hole_500_v1/backups/final-with-activations.tar.zst --progress --stats 10s --timeout 2m --retries 3
rclone check "$BACKUPS" gdrive:PSSN2/artifacts/black_hole_500_v1/backups --one-way --include final-with-activations.tar.zst
```

These activations are extracted afresh for **both the old and new labeled rows** in the frozen 500-question dataset. Old activation files and trained checkpoints are not inputs to this command. Data preparation is now complete; continue with the predictor stages below on the same server.

## 17. Train the comparison baselines (W4-002)

```bash
python -u -m src.black_hole.baselines --data-dir "$CAMPAIGN/final/w4-001-activations" --raw-dataset "$CAMPAIGN/final/black_hole_training_dataset.jsonl" --output-dir "$CAMPAIGN/final/w4-002-baselines" --seed 42
cat "$CAMPAIGN/final/w4-002-baselines/baseline_results.json"
```

The baselines and the predictor use the same frozen splits. The coarser-state baseline is available because the finalizer remapped the state vectors.

Define a manual final-results backup command, and use it after the remaining important stages:

```bash
backup_final() {
  local archive_name="final-$1.tar.zst"
  tar -I 'zstd -T0 -3' -cf "$BACKUPS/$archive_name" -C "$CAMPAIGN" final campaign_configuration.json || return
  (cd "$BACKUPS" && sha256sum "$archive_name" > "$archive_name.sha256") || return
  rclone copyto "$BACKUPS/$archive_name" "gdrive:PSSN2/artifacts/black_hole_500_v1/backups/$archive_name" --progress --stats 10s --timeout 2m --retries 3 || return
  rclone copyto "$BACKUPS/$archive_name.sha256" "gdrive:PSSN2/artifacts/black_hole_500_v1/backups/$archive_name.sha256" --progress || return
  rclone check "$BACKUPS" gdrive:PSSN2/artifacts/black_hole_500_v1/backups --one-way --include "$archive_name" --include "$archive_name.sha256" || return
  printf 'BACKUP VERIFIED: %s\n' "$archive_name"
}
backup_final baselines
```

## 18. Train the predictor using train and validation (W4-003/004)

```bash
mkdir -p "$CAMPAIGN/final/w4-004-training"
python -u -m src.black_hole.trainer --data-dir "$CAMPAIGN/final/w4-001-activations" --output-dir "$CAMPAIGN/final/w4-004-training" --epochs 100 --batch-size 32 --patience 12 --learning-rate 0.001 --weight-decay 0.0001 --hidden-size 256 --dropout 0.2 --seed 42 --device cuda 2>&1 | tee "$CAMPAIGN/final/w4-004-training/run.log"
cat "$CAMPAIGN/final/w4-004-training/training_summary.json"
backup_final trained-predictor
```

ROCm PyTorch uses `cuda` as its device API name. This trains a fresh small MLP detector on the combined sentence activation features; it does not fine-tune Qwen. Training uses the train split; validation F1 chooses the best checkpoint and early stopping. The trainer writes the detector configuration, fitted train-only normalization, history, `best_model.pt`, `final_model.pt` and `last_checkpoint.pt`. It does not use the test split for training.

If interrupted, run the **same training command and settings** with `--resume` added. The last completed epoch checkpoint is reused. Do not point this command at your previous 99-question model folder.

## 19. Evaluate the predictor and select its threshold (W4-005/006)

Evaluate the fixed initial threshold and compare baselines:

```bash
python -u -m src.black_hole.evaluation --activation-dir "$CAMPAIGN/final/w4-001-activations" --checkpoint "$CAMPAIGN/final/w4-004-training/best_model.pt" --split-dir "$CAMPAIGN/final/splits" --baseline-results "$CAMPAIGN/final/w4-002-baselines/baseline_results.json" --output-dir "$CAMPAIGN/final/w4-005-evaluation" --threshold 0.5 --device cuda
cat "$CAMPAIGN/final/w4-005-evaluation/evaluation_summary.json"
backup_final evaluated-predictor
```

Choose the export threshold using validation data and the existing false-skip risk limit:

```bash
python -u -m src.black_hole.thresholding --activation-dir "$CAMPAIGN/final/w4-001-activations" --checkpoint "$CAMPAIGN/final/w4-004-training/best_model.pt" --split-dir "$CAMPAIGN/final/splits" --baseline-comparison "$CAMPAIGN/final/w4-005-evaluation/baseline_comparison.json" --output-dir "$CAMPAIGN/final/w4-006-export" --max-false-skip-rate 0.10 --device cuda
cat "$CAMPAIGN/final/w4-006-export/threshold.json"
cat "$CAMPAIGN/final/w4-006-export/export_summary.json"
backup_final validated-predictor
```

The output includes `detector.pt`, `config.json`, `threshold.json`, validation/test metrics, and `week4_black_hole_detector_report.md`. Threshold selection uses validation; test results report performance after that choice. Keep the test split out of subsequent parameter choices. If the validation risk limit is not met, the existing export logic leaves automatic skipping disabled.

Read accuracy, low-value precision/recall/F1 and the answer-sensitive false-skip rate together. More data may improve them, but does not guarantee improvement or establish end-to-end reasoning speedup.

## 20. Preserve the complete campaign and download to your PC

After all computation has finished, create one archive containing the campaign source dataset, all new raw and prepared data, combined state snapshots, full new experiment evidence, old/new combined labels, logs, final splits, activations, trained predictor and evaluation/export outputs:

```bash
tar -I 'zstd -T0 -3' -cf "$BACKUPS/black_hole_500_v1-complete.tar.zst" -C "$PSSN_DATA/artifacts" black_hole_500_v1
cd "$BACKUPS"
sha256sum black_hole_500_v1-complete.tar.zst > black_hole_500_v1-complete.tar.zst.sha256
rclone copy "$BACKUPS" gdrive:PSSN2/artifacts/black_hole_500_v1/backups --include 'black_hole_500_v1-complete.tar.zst*' --progress --stats 10s --timeout 2m --retries 3
rclone check "$BACKUPS" gdrive:PSSN2/artifacts/black_hole_500_v1/backups --include 'black_hole_500_v1-complete.tar.zst*' --one-way
```

If you included the old pilot, its full trials and original state-vector artifacts are outside the campaign. Keep the original PSSN2 copy/ZIP plus the transferred processed pilot folders, or archive the complete pilot run separately before removing the server:

```bash
tar -I 'zstd -T0 -3' -cf "$BACKUPS/pilot-evidence-and-labels.tar.zst" -C "$PSSN_DATA/artifacts/black_hole" w3-1-sentence-3samples-run1
rclone copyto "$BACKUPS/pilot-evidence-and-labels.tar.zst" gdrive:PSSN2/artifacts/black_hole_500_v1/backups/pilot-evidence-and-labels.tar.zst --progress
rclone check "$BACKUPS" gdrive:PSSN2/artifacts/black_hole_500_v1/backups --include pilot-evidence-and-labels.tar.zst --one-way
```

Download directly from the server using **Windows PowerShell**:

```powershell
scp root@SERVER_IP:/root/pssn-work/backups/black_hole_500_v1/black_hole_500_v1-complete.tar.zst "$env:USERPROFILE\Downloads\"
scp root@SERVER_IP:/root/pssn-work/backups/black_hole_500_v1/black_hole_500_v1-complete.tar.zst.sha256 "$env:USERPROFILE\Downloads\"
Get-FileHash "$env:USERPROFILE\Downloads\black_hole_500_v1-complete.tar.zst" -Algorithm SHA256
Get-Content "$env:USERPROFILE\Downloads\black_hole_500_v1-complete.tar.zst.sha256"
```

The hash values should match. Download the pilot archive as well if its full evidence is not already safe locally. Use a current 7-Zip version to extract `.tar.zst`, then extract the `.tar`. Creating, copying and downloading these archives leaves the original server directories intact.

## Progress, interruption and recovery

Detach tmux: **Ctrl+B, then D**. Reconnect: `tmux attach -t pssn-500`.

From another server terminal:

```bash
pgrep -af '[s]erver_campaign|[r]un_sentence_experiments'
cat /root/pssn-work/PSSN2/artifacts/black_hole_500_v1/batches/batch-001/experiments/run_status.json
rocm-smi --showpids
```

Use the current batch name in that path. PID presence shows a process exists; an advancing heartbeat/progress/log shows recent activity. A stale `is_running: true` field alone is not proof of a live process. In this workflow no rclone process is expected during computation.

For a controlled stop, press Ctrl+C in the running tmux pane, wait for the prompt, then back up the batch. Rerun the same baseline or sentence-experiment command with the same input, folder, arguments, source and environment to resume. Completed baseline records and W3 trial/gradient checkpoints are retained. Work interrupted before its checkpoint may repeat. Do not start two compute processes for the same run folder.

Baseline and W3 have detailed resume support. Segmentation, validation and state-vector extraction can be rerun if interrupted; the latter recomputes the batch. Completed preparation stages have hash markers and are reused only when their tracked inputs/code match. If a server fails before you run a manual backup, only the last completed backup is recoverable elsewhere.

To restore one backup on another configured server, first download and extract the source dataset snapshot, then the **latest snapshot per batch**, preserving the same `/root/pssn-work/PSSN2` paths. A batch archive is rooted at `campaign_configuration.json` and `batches/`; extract it into `$CAMPAIGN`:

```bash
rclone copyto gdrive:PSSN2/artifacts/black_hole_500_v1/backups/batch-001-08-dataset.tar.zst /root/pssn-work/backups/batch-001-08-dataset.tar.zst --progress
tar --use-compress-program=unzstd -xf /root/pssn-work/backups/batch-001-08-dataset.tar.zst -C "$CAMPAIGN"
```

Restore the previous software versions from the saved session metadata before resuming a partial removal experiment. Extract into an empty campaign location; preserve any existing local run rather than overwriting it.

## Run order

Clone updated repo and set up environment -> Drive authorization -> copy PSSN2 locally -> verify existing states and labels -> generate new baseline traces -> segment -> validate -> new state vectors -> new state dataset -> combine old/new states -> select only new traces -> extract new sentences -> experiment on new sentences -> label only new data -> new batch dataset -> count.

Repeat batches until at least 500 usable questions -> preserve existing labeled questions and select enough new ones -> combine original old/new labels and training rows -> freeze question splits -> extract fresh activations for all 500 -> train baselines -> train/validate predictor -> evaluate -> select validation threshold/export -> complete archive -> manual Drive upload / PC download.

Sources checked for this guide: [AMD ROCm 7.2 Ubuntu](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/install-methods/package-manager/package-manager-ubuntu.html), [rclone remote setup](https://rclone.org/remote_setup/), [rclone copy](https://rclone.org/commands/rclone_copy/), and the current repository's Week 2–4 implementation. These instructions were prepared locally; remote MI300X execution and Google Drive authorization are performed by you.
