# V3: GPU-conditional four-consecutive-MLP bypass pilot (protocol V3)

This optimized run is a new experiment; it does not reuse the earlier block-pilot
labels or checkpoint, nor the single-MLP artifacts. Its protocol identifier is
`v3-question-level-nonoverlapping-mlp-block4-device-cond-v3`.

V3 keeps features, group selection, thresholding, and the per-layer skip mask
on the GPU. After prompt prefill, each MLP uses a `torch.compile`-compiled
`torch.cond` branch. The device predicate bypasses the selected MLP without
transferring the choice to Python. The selected group is copied to CPU only
after generation for result logging, outside the timed interval. This removes
the controller-selection synchronization, but does not guarantee an end-to-end
speedup because conditional routing and compilation also have costs.

Both baseline and bypass prompts request only a final-answer line, with a
64-token output cap. This is a fresh protocol and output directory: rerun
labels, training, unseen evaluation, and benchmark. Earlier V1/V2 artifacts
remain unchanged and are not results for this implementation.

PyTorch documents `torch.cond` as a prototype control-flow operator. This code
requires `torch.cond` and `torch.compile(fullgraph=True)` to work in the
server's PyTorch/ROCm build; it deliberately has no CPU-routing fallback. If
the first routed decode reports a compile/backend error, stop and retain the
log: the server build needs a compatibility fix before the full labeling job.

The 36 Qwen3-4B MLPs are divided into nine **non-overlapping groups of four**:
`0–3`, `4–7`, `8–11`, `12–15`, `16–19`, `20–23`, `24–27`, `28–31`, and
`32–35`. The prompt prefill runs normally. During cached answer decoding, the
selected MLPs return zero and their surrounding residual connections
remain active. The final benchmark chooses up to four groups per question
(up to 16 MLPs), taking the highest scores above the checkpoint's threshold.
If fewer groups pass, it skips fewer; if none pass, it skips none. Selection
and mask construction remain on the GPU.

For **100 training questions**, labeling runs one baseline plus nine group
interventions per question: **1,000 recorded generations**, plus one short
warm-up. Unscored `-1` results are conservatively mapped to unsafe label `0`
for training. The original `raw_labels` and reason text are kept in each
question's `labels.json`, so mapped rows can be audited. A baseline that is
incomplete or wrong makes every group for that question `0`; that is
conservative but is not evidence that each particular group caused the error.

The target is safety of one isolated four-MLP group. It cannot tell which MLP
inside an unsafe group caused a failure, and it does not establish that
skipping multiple groups together is safe. The 900 group rows are correlated
within questions; the 100 questions are the independent units.

The maximum-four benchmark tests those combined removals directly against
normal answer generation. Adding this evaluation limit does not require
relabeling or retraining an existing matching V3-protocol controller. The
unseen predictor evaluation still measures isolated-group predictions. Use a
fresh benchmark directory because the maximum is part of the run identity.

## Upload the updated code to the ROCm 7.2 server

From Windows PowerShell, replace `SERVER_IP` with the server's public IP:

```powershell
scp "C:\Users\LOQ\Downloads\Uni\PSSN\Repository\PSSN-LLM-Reasoning\v3\block_common.py" root@SERVER_IP:/root/pssn-work/PSSN-LLM-Reasoning/v3/
scp "C:\Users\LOQ\Downloads\Uni\PSSN\Repository\PSSN-LLM-Reasoning\v3\block_label.py" root@SERVER_IP:/root/pssn-work/PSSN-LLM-Reasoning/v3/
scp "C:\Users\LOQ\Downloads\Uni\PSSN\Repository\PSSN-LLM-Reasoning\v3\block_benchmark.py" root@SERVER_IP:/root/pssn-work/PSSN-LLM-Reasoning/v3/
```

The existing V3 environment and `v3/common.py` answer-parser fix are required.
Do not run the old single-skip job and this job simultaneously on the same GPU.
Use new results directories so all earlier artifacts stay unchanged.

## Run on the server

Start or reconnect to tmux:

```bash
tmux new-session -A -s pssn-v3-block4
```

If already inside tmux, use `tmux new-window -n v3-block4`. In that pane:

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-v3-rocm72/bin/activate
export V3_BLOCK4_RUN=/root/pssn-work/PSSN2/v3/block4-pilot-100-v3
mkdir -p "$V3_BLOCK4_RUN/logs"
set -o pipefail
```

Detach from tmux with Ctrl+B, then D. Reconnect with `tmux attach -t pssn-v3-block4`.

1. First run this one-question compatibility smoke job (10 generations). It is
   separate from the training dataset. Continue only if it finishes and writes
   `labels/summary.json`:

```bash
export V3_SMOKE_RUN=/root/pssn-work/PSSN2/v3/block4-device-cond-smoke
python -u -m v3.block_label \
  --output-dir "$V3_SMOKE_RUN/labels" \
  --rows 1 --split train --seed 42 --max-new-tokens 64 \
  2>&1 | tee -a "$V3_BLOCK4_RUN/logs/device-cond-smoke.log"
```

2. Label 100 new GSM8K training questions. Resume an interrupted run using
   the exact same command and output directory. After it completes, back up:

```bash
python -u -m v3.block_label \
  --output-dir "$V3_BLOCK4_RUN/labels" \
  --rows 100 --split train --seed 42 --max-new-tokens 64 \
  2>&1 | tee -a "$V3_BLOCK4_RUN/logs/label.log"
  rclone copy "$V3_BLOCK4_RUN/labels" gdrive:PSSN2/v3/block4-pilot-100-v3/labels --progress --checksum
```

3. Train and evaluate the group controller. Questions are split 60/20/20;
   validation selects the checkpoint, and the held-out split is evaluated
   afterward. Training fails clearly if the training split has only one class.

```bash
python -u -m v3.block_train \
  --labels-dir "$V3_BLOCK4_RUN/labels" \
  --output-dir "$V3_BLOCK4_RUN/training" \
  --seed 42 --epochs 100 --patience 12 --threshold 0.5 \
  2>&1 | tee -a "$V3_BLOCK4_RUN/logs/train.log"
rclone copy "$V3_BLOCK4_RUN/training" gdrive:PSSN2/v3/block4-pilot-100-v3/training --progress --checksum
```

4. Evaluate the frozen controller on 10 unseen GSM8K test questions. This
   measures one baseline and nine group interventions per question: 100
   recorded generations, plus a short warm-up.

```bash
python -u -m v3.block_test_unseen \
  --checkpoint "$V3_BLOCK4_RUN/training/controller.pt" \
  --output-dir "$V3_BLOCK4_RUN/unseen" \
  --rows 10 --split test --seed 42 --max-new-tokens 64 \
  2>&1 | tee -a "$V3_BLOCK4_RUN/logs/unseen.log"
rclone copy "$V3_BLOCK4_RUN/unseen" gdrive:PSSN2/v3/block4-pilot-100-v3/unseen --progress --checksum
```

5. Compare normal generation with controller-guided generation on 100 further
   unseen questions, allowing up to four groups together (200 recorded
   generations, plus two warm-ups). The checkpoint threshold is frozen.

```bash
python -u -m v3.block_benchmark \
  --checkpoint "$V3_BLOCK4_RUN/training/controller.pt" \
  --unseen-dir "$V3_BLOCK4_RUN/unseen" \
  --output-dir "$V3_BLOCK4_RUN/benchmark-max4-100q" \
  --max-removal-groups 4 \
  --rows 100 --split test --seed 42 --max-new-tokens 64 \
  2>&1 | tee -a "$V3_BLOCK4_RUN/logs/benchmark-max4-100q.log"
rclone copy "$V3_BLOCK4_RUN/benchmark-max4-100q" gdrive:PSSN2/v3/block4-pilot-100-v3/benchmark-max4-100q --progress --checksum
rclone copy "$V3_BLOCK4_RUN/logs" gdrive:PSSN2/v3/block4-pilot-100-v3/logs --progress --checksum
```

Inspect `labels/summary.json`, `training/evaluation.json`,
`unseen/predictor_evaluation.json`, and `benchmark-max4-100q/metrics.json`. Each
controller result records `selected_groups`, `selected_group_indices`,
`selected_group_count`, and `selected_layers`. Metrics include the group-count
histogram and actual MLP skip counts. The legacy `selected_group` field is a
flattened layer list; use `selected_groups` to see group boundaries.

Use `--max-removal-groups 1` and a separate `benchmark-max1` output folder to
compare the single-group policy with this combined policy. The unseen
and benchmark stages are small exploratory checks; they do not prove a
publication-level accuracy or speed improvement.
