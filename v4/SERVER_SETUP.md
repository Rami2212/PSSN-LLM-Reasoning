# V4 from a fresh MI300X / Ubuntu 24.04 / ROCm 7.2 server

Run server commands in Bash. Windows commands are explicitly marked.
Default run: 100 training questions at 12/24/36 layers (300 generations),
a small CPU text controller, held-out evaluation, then 100 fresh official
test questions for paired baseline/controller generation (200 generations).

Local results: `/root/pssn-work/PSSN2/v4/depth-text-pilot-100`.
Drive backup: `gdrive:PSSN2/v4/depth-text-pilot-100`.

Nothing from V2/V3 needs to be uploaded to Drive. Qwen and GSM8K download from
Hugging Face automatically. You need the new `v4/` source folder on the server.
This pipeline trains the text classifier; it does not fine-tune Qwen for early
exits. See [README.md](README.md) for the label definition and limitations.

## 1. Connect, install tools, and clone

Choose the cloud image with **ROCm 7.2 already installed**. AMD documents
[ROCm 7.2 Ubuntu installation](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/install-methods/package-manager/package-manager-ubuntu.html)
for bare images and [PyTorch setup](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/3rd-party/pytorch-install.html)
for the framework. The commands below use AMD's published Python 3.12 wheels
from its [7.2 release directory](https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/).

On your Windows PC, open PowerShell, replacing `SERVER_IP`:

```powershell
ssh root@SERVER_IP
```

On the server:

```bash
sudo apt update
sudo apt install -y git python3.12-venv python3.12-dev tmux rclone
rocm-smi
mkdir -p /root/pssn-work
cd /root/pssn-work
git clone https://github.com/Rami2212/PSSN-LLM-Reasoning.git
cd /root/pssn-work/PSSN-LLM-Reasoning
```

If the repository already exists, `cd` to it. Use its current checkout or
update it after your local V4 changes have been committed and pushed.

If V4 has not been pushed to GitHub, open a **second Windows PowerShell**
terminal and upload the complete folder directly:

```powershell
$repo = 'C:\Users\LOQ\Downloads\Uni\PSSN\Repository\PSSN-LLM-Reasoning'
scp -r "$repo\v4" root@SERVER_IP:/root/pssn-work/PSSN-LLM-Reasoning/
```

Add `-P PORT` to `scp` or `-p PORT` to `ssh` if your server uses another SSH port.

## 2. Create the V4 Python environment

On the server, start a persistent terminal before installation:

```bash
tmux new-session -A -s pssn-v4
cd /root/pssn-work/PSSN-LLM-Reasoning
python3.12 -m venv .venv-v4-rocm72
source .venv-v4-rocm72/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4
python -m pip install \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl'
python -m pip install -r v4/requirements.txt
python -m pip check
python -c 'import torch, transformers; assert torch.cuda.is_available(); assert str(torch.version.hip).startswith("7.2"); assert transformers.__version__ == "4.57.1"; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))'
```

To leave the persistent terminal while a command runs: **Ctrl+B**, release,
then **D**. Do not use Ctrl+C to detach; Ctrl+C interrupts the command.
To return after a disconnection:

```bash
tmux attach-session -t pssn-v4
```

If already inside tmux, use that session; do not nest another one.

## 3. Configure Google Drive

On the server:

```bash
rclone config
```

Choose:

1. `n`: new remote; name `gdrive`.
2. Storage: `drive` (Google Drive).
3. Client ID and secret: Enter to leave blank.
4. Scope: `1` for read/write; service account file: blank.
5. Advanced configuration: `n`.
6. Auto/browser configuration on this headless server: `n`.

The server prints an `rclone authorize "drive" ...` command. Install rclone
on the Windows PC from [rclone downloads](https://rclone.org/downloads/) if
needed, open PowerShell in the directory containing `rclone.exe`, and run the
**exact command printed by the server**, using `./rclone.exe` if it is not on
PATH. Sign in using the browser, then paste the returned token at the server's
`config_token>` prompt. Complete the remote setup; select no Shared Drive
unless you actually use one. Keep the token private.

This follows [rclone's headless authentication guide](https://rclone.org/remote_setup/).
Verify and create the V4 backup destination:

```bash
rclone lsd gdrive:PSSN2
rclone mkdir gdrive:PSSN2/v4/depth-text-pilot-100
```

## 4. Set paths and save the environment

Inside the tmux session:

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-v4-rocm72/bin/activate
export V4_RUN=/root/pssn-work/PSSN2/v4/depth-text-pilot-100
export V4_REMOTE=gdrive:PSSN2/v4/depth-text-pilot-100
mkdir -p "$V4_RUN/logs"
set -o pipefail
python -m pip freeze > "$V4_RUN/logs/pip-freeze.txt"
rocm-smi > "$V4_RUN/logs/rocm-smi.txt"
git rev-parse HEAD > "$V4_RUN/logs/repository-commit.txt"
tar --exclude='__pycache__' -czf "$V4_RUN/v4-source.tar.gz" v4
```

Re-export these two paths after entering a new shell. Do not put an HF access
token in a command log or source file; an existing `HF_TOKEN` environment
variable is read automatically by Hugging Face. A token is optional for the
public model and dataset.

## 5. Optional: pull a previous V4 backup on a replacement server

Skip this on the first run. It is unnecessary to copy the full previous PSSN2
dataset. On a **new server with an empty V4_RUN**, restore saved V4 artifacts:

```bash
rclone copy "$V4_REMOTE" "$V4_RUN" --progress --checksum
if [ -f "$V4_RUN/labels.tar.gz" ]; then
  tar -xzf "$V4_RUN/labels.tar.gz" -C "$V4_RUN"
fi
if [ -f "$V4_RUN/benchmark.tar.gz" ]; then
  tar -xzf "$V4_RUN/benchmark.tar.gz" -C "$V4_RUN"
fi
```

Use the same source version, settings and dependencies to resume. Do not
extract an old archive over more recent local results. A different GPU/software
environment requires a new generation output directory to avoid mixing timings.

## 6. Generate depth labels

Optional first check: run only five questions to inspect shallow outputs
before spending time on all 100. This uses its own diagnostic folder:

```bash
python -u -m v4.label --output-dir "$V4_RUN/smoke-labels" --rows 5 --seed 42 --max-new-tokens 1024
cat "$V4_RUN/smoke-labels/summary.json"
```

Five questions cannot establish performance, but obviously broken shallow
answers are a reason to investigate early-exit adaptation before scaling.
The full pilot below is a separate run with the same first five questions
included in its 100-question selection:

```bash
python -u -m v4.label \
  --output-dir "$V4_RUN/labels" \
  --rows 100 --seed 42 --max-new-tokens 1024 \
  2>&1 | tee -a "$V4_RUN/logs/label.log"
cat "$V4_RUN/labels/summary.json"
cat "$V4_RUN/labels/run_status.json"
```

This tests the same 100 questions at all three depths. Each depth trial is
saved immediately. To resume after Ctrl+C or a process failure, run the exact
same command. Warmups run again; existing trial files are reused.

Check `target_depth_counts` and `shallow_targets` before training. If all usable
targets are 36, stop here: the frozen checkpoint has not demonstrated useful
shallow exits. Training a text controller cannot fix that; early-exit model
adaptation would be a separate experiment. A pilot can also fail because too
few full-depth answers are complete/correct. Inspect raw responses rather than
silently dropping those results from accuracy reports.

After the command has finished, back up labels as one archive to avoid slow
uploads of hundreds of small files:

```bash
tar -C "$V4_RUN" -czf "$V4_RUN/labels.tar.gz" labels
rclone copyto "$V4_RUN/labels.tar.gz" "$V4_REMOTE/labels.tar.gz" --progress --checksum --contimeout 20s --timeout 2m --retries 3
rclone copy "$V4_RUN/logs" "$V4_REMOTE/logs" --progress --checksum
rclone copyto "$V4_RUN/v4-source.tar.gz" "$V4_REMOTE/v4-source.tar.gz" --progress --checksum
```

Uploads are manual commands. Upload failure leaves local trials intact and
does not fail or pause the Python runner. Retry the upload independently.

## 7. Train the question-text controller

This stage runs on CPU, even on the GPU server:

```bash
python -u -m v4.train \
  --labels-dir "$V4_RUN/labels" \
  --output-dir "$V4_RUN/training" \
  --seed 42 --max-validation-accuracy-drop 0.02 \
  2>&1 | tee -a "$V4_RUN/logs/train.log"
cat "$V4_RUN/training/training_summary.json"
rclone copy "$V4_RUN/training" "$V4_REMOTE/training" --progress --checksum
rclone copy "$V4_RUN/logs" "$V4_REMOTE/logs" --progress --checksum
```

Output: `training/controller.joblib`, split IDs, validation search, and summary.
The default uses 70/15/15 questions for train/validation/test. Only training
text fits the vocabulary. Validation selects the uncertainty fallback and
regularization. An `always_full_depth_fallback: true` result means no validated
compute reduction was selected; keep and report that result.

## 8. Evaluate held-out labeled questions

```bash
python -u -m v4.evaluate \
  --labels-dir "$V4_RUN/labels" \
  --controller "$V4_RUN/training/controller.joblib" \
  --output-dir "$V4_RUN/evaluation" \
  2>&1 | tee -a "$V4_RUN/logs/evaluate.log"
cat "$V4_RUN/evaluation/metrics.json"
rclone copy "$V4_RUN/evaluation" "$V4_REMOTE/evaluation" --progress --checksum
```

This uses saved answers from the untouched test partition, with no new Qwen
generations. It reports correctness/depth choices against fixed-depth baselines.
It does not measure a fresh runtime gain. With 100 total questions, this test
partition has only 15 questions: treat it as a pilot.

## 9. Generate answers on 100 new official test questions

Inside tmux, with the environment and paths from step 4 active:

```bash
python -u -m v4.benchmark \
  --controller "$V4_RUN/training/controller.joblib" \
  --output-dir "$V4_RUN/benchmark" \
  --rows 100 --seed 43 \
  2>&1 | tee -a "$V4_RUN/logs/benchmark.log"
cat "$V4_RUN/benchmark/metrics.json"
cat "$V4_RUN/benchmark/run_status.json"
```

This runs two generations for each new GSM8K test question: full 36-layer
baseline and text-controller-selected depth. No development question is reused.
The same checkpoint revision, prompt and token cap are carried from training.
Both answers, correctness, completion, token counts, synchronized latency,
routing overhead, memory and depth decisions are saved. No controller thresholds
are tuned on these test questions. Rerunning this exact command resumes saved
pairs; it does not create an independent repeated timing trial. Use a new
benchmark directory for a repeat.

Detach using **Ctrl+B, D**, then reconnect using `tmux attach-session -t pssn-v4`.
In another SSH terminal, inspect progress with:

```bash
pgrep -af '[v]4.label|[v]4.benchmark'
cat /root/pssn-work/PSSN2/v4/depth-text-pilot-100/benchmark/run_status.json
```

After completion, back up the benchmark and verify stored archives:

```bash
tar -C "$V4_RUN" -czf "$V4_RUN/benchmark.tar.gz" benchmark
rclone copyto "$V4_RUN/benchmark.tar.gz" "$V4_REMOTE/benchmark.tar.gz" --progress --checksum --contimeout 20s --timeout 2m --retries 3
rclone copy "$V4_RUN/logs" "$V4_REMOTE/logs" --progress --checksum
rclone check "$V4_RUN" "$V4_REMOTE" --one-way --include '*.tar.gz'
```

## 10. Download everything directly to the Windows PC

Once the runs finish, create an archive on the server. This preserves the
original folders:

```bash
tar -C /root/pssn-work/PSSN2/v4 -czf /root/pssn-work/v4-depth-text-pilot-100.tar.gz depth-text-pilot-100
```

On Windows PowerShell:

```powershell
scp root@SERVER_IP:/root/pssn-work/v4-depth-text-pilot-100.tar.gz "$env:USERPROFILE\Downloads\"
```

For a larger training experiment, choose a new folder such as
`depth-text-1000` and set `--rows 1000` in step 6; then repeat steps 7-9.
The 100-question pilot is for feasibility, not a claim of reliable generalized
question difficulty prediction.
