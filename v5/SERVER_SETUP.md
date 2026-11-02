# V5 server setup and smoke test

For the **100-question × 3-depth labeling, controller training, evaluation,
and unseen test**, follow [CONTROLLER_300_SERVER.md](CONTROLLER_300_SERVER.md).
It includes Drive backups after each stage and uses direct depths 24/28/32.

Goal: run Meta's already adapted LayerSkip Llama3-8B on five GSM8K questions at
24/28/32 layers, then compare full-depth generation with self-speculative early
exits at layers 24 and 28. There is **no fine-tuning**. Back up results manually.
Use Ubuntu 24.04 with ROCm 7.2 installed on the MI300X server.

Server IP below is your last supplied IP, `134.199.192.72`. Replace it if the
server has changed. Windows commands run in PowerShell; server commands run
in Bash.

## 1. Request access before starting the GPU run

Sign into Hugging Face and request/accept access to
[facebook/layerskip-llama3-8B](https://huggingface.co/facebook/layerskip-llama3-8B).
Wait until your account can download its weights. Have a read token for that
same account, including access to this gated repository.

The weights are Llama-based, not Qwen. The checkpoint has 32 layers, so V5 uses
24/28/32. Its early exits still need the smoke test to establish usefulness.

## 2. Connect and prepare the repository

On your Windows PC:

```powershell
ssh root@134.199.192.72
```

On the server:

```bash
sudo apt update
sudo apt install -y git python3.12-venv python3.12-dev tmux rclone
rocm-smi
mkdir -p /root/pssn-work
cd /root/pssn-work
```

If the repository is not present, clone it:

```bash
git clone https://github.com/Rami2212/PSSN-LLM-Reasoning.git
```

If it already exists, use its existing directory. Upload the V5 source in the
next step even if your GitHub checkout does not yet contain it.

## 3. Upload only the complete v5 folder

Open a second **Windows PowerShell** terminal:

```powershell
$repo = 'C:\Users\LOQ\Downloads\Uni\PSSN\Repository\PSSN-LLM-Reasoning'
scp -r "$repo\v5" root@134.199.192.72:/root/pssn-work/PSSN-LLM-Reasoning/
```

The folder includes `__init__.py`, `smoke.py`, `compare_self_speculative.py`,
`requirements.txt`, `README.md` and this document. There is no V3/V4 dependency and no existing data/model to
upload. GSM8K and the gated model download automatically. Do not upload `.env`.

## 4. Persistent session and Python environment

On the server, outside an existing tmux session:

```bash
tmux new-session -A -s pssn-v5
```

If already inside tmux, continue there. Do not nest another session.

Create a separate V5 environment:

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
python3.12 -m venv .venv-v5-rocm72
source .venv-v5-rocm72/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl'
python -m pip install -r v5/requirements.txt
python -m pip check
python -c 'import torch, transformers; assert torch.cuda.is_available(); assert str(torch.version.hip).startswith("7.2"); assert transformers.__version__ == "4.57.1"; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))'
```

These use AMD's [ROCm 7.2 PyTorch installation guidance](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/3rd-party/pytorch-install.html)
and [published wheels](https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/).
The image must already include ROCm 7.2 and GPU drivers.

Login interactively, using your approved Hugging Face account's token:

```bash
hf auth login
```

Alternatively, an already exported `HF_TOKEN` is read automatically. Never
include the token in source, logs or an uploaded archive. A 401/403/gated-repo
error means access/authentication must be resolved before generation.

## 5. Configure Drive (manual backups)

Skip configuration if `rclone lsd gdrive:PSSN2` already works.

```bash
rclone config
```

Choose new remote `n`, name `gdrive`, storage `drive`, blank client ID/secret,
scope `1`, blank service-account file, advanced config `n`, and auto/browser
config `n` on this headless server.

Run the exact `rclone authorize "drive" ...` command it prints on your Windows
PC with rclone installed. If necessary, open PowerShell beside `rclone.exe`
and use `./rclone.exe authorize ...`. Sign in, paste the returned token into
the server's `config_token>` field, then finish the remote configuration.
See [rclone headless setup](https://rclone.org/remote_setup/).

```bash
rclone lsd gdrive:PSSN2
rclone mkdir gdrive:PSSN2/v5/layerskip-smoke-5q
```

No previous PSSN2 data needs to be pulled for this test.

## 6. Run five questions (15 generations)

Inside your tmux session:

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-v5-rocm72/bin/activate
export V5_RUN=/root/pssn-work/PSSN2/v5/layerskip-smoke-5q
export V5_REMOTE=gdrive:PSSN2/v5/layerskip-smoke-5q
mkdir -p "$V5_RUN/logs"
set -o pipefail
python -m pip freeze > "$V5_RUN/logs/pip-freeze.txt"
rocm-smi > "$V5_RUN/logs/rocm-smi.txt"
tar --exclude='__pycache__' -czf "$V5_RUN/v5-source.tar.gz" v5
flock -n "$V5_RUN/.smoke.lock" python -u -m v5.smoke \
  --output-dir "$V5_RUN/smoke" \
  --rows 5 --seed 42 --max-new-tokens 1024 \
  2>&1 | tee -a "$V5_RUN/logs/smoke.log"
```

The first run downloads approximately 16 GB of model weights. BF16 is used
when supported (including MI300X); otherwise FP16 is selected. The run records
model/tokenizer/dataset commit hashes before loading the weights. You do not
pass the old Qwen revision. The prompt contains three worked examples, but no
reference answer/solution for the question being evaluated.

Detach using **Ctrl+B**, release, then **D**. The run continues on the server.
Reconnect after losing SSH or closing the console:

```bash
ssh root@134.199.192.72
tmux attach-session -t pssn-v5
```

In a second server terminal, monitor:

```bash
pgrep -af '[v]5.smoke'
cat /root/pssn-work/PSSN2/v5/layerskip-smoke-5q/smoke/run_status.json
```

`status: completed`, `is_running: false`, and 15 completed trials mean it
finished. Heartbeats update every 15 seconds during downloads/loading/generation.
A stale heartbeat requires checking the PID; the JSON alone cannot prove liveness.

Each answer is saved immediately. After interruption, re-enter the environment,
re-export V5_RUN/V5_REMOTE, and repeat the same `flock ... python ...` command
with the same options to resume. Existing trials are reused. `flock` prevents
two copies from writing this run at the same time. Changing source/settings/
GPU/packages requires a new output directory to preserve comparable evidence.

Inspect completion and the results:

```bash
cat "$V5_RUN/smoke/run_status.json"
cat "$V5_RUN/smoke/summary.json"
```

`accuracy` counts all five questions, including incomplete/missing answers as
incorrect. `paired_vs_depth_32` compares the same questions' full-depth answers
with each early exit. A timing ratio greater than 1 means the early exit was
faster on the paired runs. Inspect individual responses before scaling up.

## 7. Compare full-depth vs self-speculative decoding

This comparison reuses the five exact questions and demonstrations saved by
the smoke run. It runs full 32-layer greedy generation and self-speculative
generation with early-exit drafts at layer 24 and layer 28. The remaining
layers verify/correct draft tokens, so this is different from the earlier test
that permanently cut off the model at a selected depth. Transformers' official
`assistant_early_exit` API is used; its docs describe self-speculative decoding
as verifying early-exit candidates to preserve the target model's output
distribution. [Transformers 4.57.1 docs](https://huggingface.co/docs/transformers/v4.57.1/generation_strategies#self-speculative-decoding)

Run after the smoke test has finished, in the same tmux session:

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-v5-rocm72/bin/activate
export V5_RUN=/root/pssn-work/PSSN2/v5/layerskip-smoke-5q
export V5_REMOTE=gdrive:PSSN2/v5/layerskip-smoke-5q
mkdir -p "$V5_RUN/logs"
flock -n "$V5_RUN/.selfspec.lock" python -u -m v5.compare_self_speculative \
  --source-run "$V5_RUN/smoke" \
  --output-dir "$V5_RUN/selfspec-comparison-stopfix1" \
  --max-new-tokens 1024 --num-assistant-tokens 5 \
  2>&1 | tee -a "$V5_RUN/logs/selfspec-comparison.log"
```

This is 15 measured generations (5 questions x full, exit-24, exit-28), plus
three short warmups. Model weights should already be cached. It records
correctness, completion, output tokens, synchronized latency, peak allocated
and reserved GPU memory, and exact token-sequence matches against baseline.
`baseline_seconds_divided_by_self_speculative_seconds` above 1 means the paired
self-speculative timing was faster. Results are in
`$V5_RUN/selfspec-comparison-stopfix1/summary.json`; monitor with:

```bash
cat "$V5_RUN/selfspec-comparison-stopfix1/run_status.json"
```

Do not describe this as layer skipping with guaranteed speedup: verification
still runs the remaining layers, and five questions are only a smoke test.
Compare actual latency and answer correctness. The script requires the exact
smoke outputs (`manifest.json`, `questions.json`, `demonstrations.json`) in
`$V5_RUN/smoke`.

## 8. Back up finished results to Drive

After generation completes:

```bash
tar -C "$V5_RUN" -czf "$V5_RUN/smoke.tar.gz" smoke
rclone copyto "$V5_RUN/smoke.tar.gz" "$V5_REMOTE/smoke.tar.gz" --progress --checksum --contimeout 20s --timeout 2m --retries 3
tar -C "$V5_RUN" -czf "$V5_RUN/selfspec-comparison-stopfix1.tar.gz" selfspec-comparison-stopfix1
rclone copyto "$V5_RUN/selfspec-comparison-stopfix1.tar.gz" "$V5_REMOTE/selfspec-comparison-stopfix1.tar.gz" --progress --checksum --contimeout 20s --timeout 2m --retries 3
rclone copy "$V5_RUN/logs" "$V5_REMOTE/logs" --progress --checksum
rclone copyto "$V5_RUN/v5-source.tar.gz" "$V5_REMOTE/v5-source.tar.gz" --progress --checksum
rclone check "$V5_RUN" "$V5_REMOTE" --one-way --include '*.tar.gz'
```

Drive location: **My Drive / PSSN2 / v5 / layerskip-smoke-5q**. Uploads are
independent of the experiment. A failed copy can be retried while local files
remain available. The archive avoids thousands of separate Drive requests.

## 9. Download everything directly to your PC

On the server after the run finishes:

```bash
tar -C /root/pssn-work/PSSN2/v5 -czf /root/pssn-work/v5-layerskip-smoke-5q.tar.gz layerskip-smoke-5q
```

On Windows PowerShell:

```powershell
scp root@134.199.192.72:/root/pssn-work/v5-layerskip-smoke-5q.tar.gz "$env:USERPROFILE\Downloads\"
```

Creating/downloading an archive preserves the original server folder.

## 10. Restore a backup to a replacement server

Upload the same V5 source and recreate its environment, then restore to an
empty local run folder:

```bash
export V5_RUN=/root/pssn-work/PSSN2/v5/layerskip-smoke-5q
export V5_REMOTE=gdrive:PSSN2/v5/layerskip-smoke-5q
mkdir -p "$V5_RUN"
rclone copy "$V5_REMOTE" "$V5_RUN" --progress --checksum
tar -xzf "$V5_RUN/smoke.tar.gz" -C "$V5_RUN"
```

Restoring lets you inspect results without rerunning. Resume generation only
with a matching recorded environment. Use a fresh run folder for new timings.
