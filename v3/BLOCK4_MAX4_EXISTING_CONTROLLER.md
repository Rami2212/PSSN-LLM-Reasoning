# Run the final max-four comparison with the saved controller

This runs only the final benchmark. It reuses the saved four-MLP-group
controller at `PSSN2/v3/block4-pilot-100/training/controller.pt`; it does not
repeat labeling or controller training. The saved V1 controller is accepted
because its feature layout and group definitions match the new runner.

For each of 100 fresh GSM8K test questions, the controller scores all nine
groups. The benchmark skips up to four groups that score at least the saved
threshold, choosing the highest scores first. It compares that answer with
normal Qwen generation. The score predicts each group individually; the
benchmark measures whether combining up to four remains safe.

The run uses the controller's original prompt, model revision, and 1,024-token
limit so its prompt-derived features remain comparable to its training. The
selected groups, answer correctness, tokens, elapsed time, peak memory, and
estimated MLP work saved are written per question and summarized in
`metrics.json`.

## Upload the two updated files

From Windows PowerShell, replace `SERVER_IP` with the server address:

```powershell
scp "C:\Users\LOQ\Downloads\Uni\PSSN\Repository\PSSN-LLM-Reasoning\v3\block_common.py" root@SERVER_IP:/root/pssn-work/PSSN-LLM-Reasoning/v3/
scp "C:\Users\LOQ\Downloads\Uni\PSSN\Repository\PSSN-LLM-Reasoning\v3\block_benchmark.py" root@SERVER_IP:/root/pssn-work/PSSN-LLM-Reasoning/v3/
```

## Run on the ROCm server

Reconnect to tmux or start a session:

```bash
tmux new-session -A -s pssn-v3-max4
```

In that session:

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-v3-rocm72/bin/activate
export V3_ROOT=/root/pssn-work/PSSN2/v3
export CHECKPOINT="$V3_ROOT/block4-pilot-100/training/controller.pt"
export PRIOR_UNSEEN="$V3_ROOT/block4-pilot-100/unseen"
export OUT_DIR="$V3_ROOT/block4-max4-100q-final-v1-controller"
mkdir -p "$OUT_DIR/logs"
set -o pipefail
```

Run the comparison:

```bash
python -u -m v3.block_benchmark \
  --checkpoint "$CHECKPOINT" \
  --unseen-dir "$PRIOR_UNSEEN" \
  --output-dir "$OUT_DIR/benchmark-max4-100q" \
  --rows 100 --split test --seed 42 \
  --max-new-tokens 1024 --max-removal-groups 4 \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  2>&1 | tee -a "$OUT_DIR/logs/benchmark-max4-100q.log"
```

When complete, copy results to Drive:

```bash
rclone copy "$OUT_DIR" gdrive:PSSN2/v3/block4-max4-100q-final-v1-controller --progress --checksum
```

Detach from tmux with Ctrl+B, then D. Reconnect with
`tmux attach -t pssn-v3-max4`. The ROCm server must support `torch.cond` inside
`torch.compile(fullgraph=True)`; a compile error means the runtime needs a small
compatibility adjustment before this benchmark can run.
