# V3: MI300X / ROCm 7.2 pilot, from setup to final generation test

Run order: **label → train → test_unseen → benchmark**. Run commands from the
repository root. V3 starts fresh from GSM8K; the old sentence dataset and old
predictor are not inputs. All results go in a new `PSSN2/v3/pilot-50/`
directory. Keep this directory when resuming.

## What this pilot actually tests

- Qwen3-4B has 36 decoder layers. Only a layer's MLP is bypassed; attention,
  residual connections, and KV cache updates remain active.
- Full prompt processing (prefill) runs normally. Save the input to each MLP
  at the last prompt token. These features are available before generation.
- During labeling, run one normal answer and one answer for each candidate
  MLP bypass, for each of 50 questions: **50 × (1 + 36) = 1,850 generations**.
  There are no three-repeat tests. One warm-up generation is additional.
- A bypass affects that same single MLP on subsequent cached decode steps.
  The first answer token comes from unchanged prefill. This avoids changing
  attention cache semantics and makes labeling and deployed use consistent.
- If the full answer is complete/correct and the bypassed answer is
  complete/correct, label 1 (observed safe). A complete wrong bypassed answer
  is label 0 (unsafe). Incomplete/missing-answer trials, failed baselines, and
  cases with no executed bypass are retained but unscored, label -1.
- Features are Qwen MLP inputs plus layer identity. The controller is a small
  MLP trained separately; Qwen weights stay frozen. It chooses **at most one
  MLP per question** after prefill. It can also choose none at threshold 0.5.
  No additional false-skip-rate constraint is applied.
- This is a question-level controller, not token-by-token routing. Independent
  layer experiments do not establish that several layers can be skipped
  together. Each safe label reflects one observed answer, not guaranteed safety.
- Fifty questions is a feasibility pilot. The 1,800 layer rows are correlated
  within questions; they are not 1,800 independent question examples.

## 1. Create the server and install Python packages

Choose an MI300X server image with **Ubuntu 24.04 and ROCm 7.2 already installed**.
For a bare Ubuntu image, first install the supported driver/runtime using
[AMD's ROCm 7.2 installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/install-methods/package-manager/package-manager-ubuntu.html).
The cloud image's host driver should not be replaced just to install Python.
AMD documents the validated ROCm 7.2 / PyTorch 2.9.1 combination in its
[PyTorch guide](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/3rd-party/pytorch-install.html).
The native wheels below are the same release family as the existing project
ROCm 7.2 environment; see [AMD's versioned wheel instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/native_linux/install-pytorch.html).

Run in the server terminal:

```bash
sudo apt update
sudo apt install -y git python3.12-venv python3.12-dev tmux rclone unzip
rocm-smi
mkdir -p /root/pssn-work
cd /root/pssn-work
git clone https://github.com/Rami2212/PSSN-LLM-Reasoning.git
cd /root/pssn-work/PSSN-LLM-Reasoning
python3.12 -m venv .venv-v3-rocm72
source .venv-v3-rocm72/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4
python -m pip install \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl'
```

If the repository already exists, use its updated checkout instead of running
`git clone` again. If V3 has not yet been pushed, upload **the entire `v3/`
folder**, including `__init__.py` and `common.py`, from your PC. In Windows
PowerShell (replace `SERVER_IP` with your public IP; use `scp -P PORT` if needed):

```powershell
scp -r "C:\Users\LOQ\Downloads\Uni\PSSN\Repository\PSSN-LLM-Reasoning\v3" root@SERVER_IP:/root/pssn-work/PSSN-LLM-Reasoning/
```

Back on the server:

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-v3-rocm72/bin/activate
python -m pip install -r v3/requirements.txt
python -m pip check
python -c 'import torch; assert str(torch.version.hip).startswith("7.2"), torch.version.hip; assert torch.cuda.is_available(); print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))'
```

V3 pins Transformers 4.57.1 because it intercepts the Qwen3 MLP boundary. The
[pinned Qwen implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py)
adds the MLP output through a residual connection, so returning zero bypasses
the MLP computation while keeping that residual. Optional CPU checks (also
upload `tests/v3/` if using SCP rather than an updated repository checkout):

```bash
python -m unittest discover -s tests/v3 -v
```

## 2. Configure Google Drive

```bash
rclone config
```

Select `n` (new remote), name it `gdrive`, choose `drive` as the storage type.
Leave client ID, client secret, and service-account file blank. Choose scope
`1` (read/write), advanced config `n`, browser auto config `n` on the server.
Run the **exact `rclone authorize ...` command printed by the server** on your
PC, which must have rclone installed and a browser. Complete browser login and
paste the resulting token into the server terminal. Do not put the token in
source code or chat. Finish/confirm configuration.

```bash
rclone lsd gdrive:
rclone mkdir gdrive:PSSN2/v3/pilot-50
```

[rclone's remote setup instructions](https://rclone.org/remote_setup/) cover
the browser authorization step. No Drive mount is required: use local disk
for all running experiments and copy only after each stage finishes.

Optional Hugging Face token (not printed or written into artifacts):

```bash
read -rsp 'HF token (optional): ' HF_TOKEN
export HF_TOKEN
```

## 3. Start tmux and set paths

Run this **outside** an existing tmux session:

```bash
tmux new-session -A -s pssn-v3
```

If you are already inside tmux, use `tmux new-window -n v3` instead.
Inside the tmux pane:

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-v3-rocm72/bin/activate
export V3_RUN=/root/pssn-work/PSSN2/v3/pilot-50
mkdir -p "$V3_RUN/logs"
python -m pip freeze > "$V3_RUN/packages.txt"
set -o pipefail
```

For every long command below: **detach with Ctrl+B, then D**. The process keeps
running. Reconnect with:

```bash
tmux attach -t pssn-v3
```

The exported paths remain in that pane. If using a new pane/session, repeat the
`cd`, `source`, `export`, and `set -o pipefail` commands above.

## 4. Label 50 new GSM8K training questions

```bash
python -u -m v3.label \
  --output-dir "$V3_RUN/labels" \
  --rows 50 --split train --seed 42 --max-new-tokens 1024 \
  2>&1 | tee -a "$V3_RUN/logs/label.log"
```

One trial per question/layer, not three. Download/model load and warm-up are
additional to the 1,850 recorded generations. The question selection and raw
GSM8K reference solutions are saved in `labels/questions.json`; answers and
features are under `labels/questions/<question-id>/`. Dataset and model
revisions, generation settings, environment, and code hashes are recorded.

Detach/reconnect using step 3 during this command. Resume with the **same
command and directory** after an interruption: completed trials are reused.
Do not start two writers to the same directory. Altered code/settings require
a new directory to avoid mixing results. Failed trials are not labeled safe.

Inspect:

```bash
cat "$V3_RUN/labels/summary.json"
cat "$V3_RUN/labels/run_status.json"
pgrep -af 'python.*v3.label'
```

`run_status.json` is updated at trial boundaries; it is not a continuously
updated heartbeat. A stale `running` value after a server crash does not prove
the process is still alive. Check `pgrep` and the tmux pane.

**Manual Drive backup after labeling finishes:**

```bash
rclone copy "$V3_RUN/labels" gdrive:PSSN2/v3/pilot-50/labels --progress --checksum
rclone copyto "$V3_RUN/packages.txt" gdrive:PSSN2/v3/pilot-50/packages.txt --progress
```

On a **new server** before the next stage, restore the completed stage from Drive:

```bash
rclone copy gdrive:PSSN2/v3/pilot-50/labels "$V3_RUN/labels" --progress --checksum
```

## 5. Train and evaluate the predictor

Run in the same tmux pane with the environment activated:

```bash
python -u -m v3.train \
  --labels-dir "$V3_RUN/labels" \
  --output-dir "$V3_RUN/training" \
  --seed 42 --epochs 100 --patience 12 --threshold 0.5 \
  2>&1 | tee -a "$V3_RUN/logs/train.log"
```

Questions are split **30 training / 10 validation / 10 test** before collecting
scorable layer rows. Questions with failed baselines contribute no training
rows, so effective question counts can be lower. Normalization uses training
rows only. Only layers with scorable training examples are eligible for the
controller's selection. Validation selects the best checkpoint; the internal test split is
evaluated after selection. Threshold 0.5 is fixed. If there are no unsafe
training examples, training stops with an explanation: all-safe data cannot
teach a useful safe/unsafe distinction.

Outputs: `controller.pt`, `history.json`, `test_predictions.json`,
`evaluation.json`, and a manifest with exact question splits. Evaluation
includes always-skip, never-skip, and training-layer-frequency baselines.
The training stage restarts from the same seed if interrupted; generation
labeling is not repeated. Best checkpoints are written after improvements.

Detach with Ctrl+B, D; reconnect with `tmux attach -t pssn-v3`.
After completion, back up:

```bash
rclone copy "$V3_RUN/training" gdrive:PSSN2/v3/pilot-50/training --progress --checksum
```

To restore before the next stage on another server:

```bash
rclone copy gdrive:PSSN2/v3/pilot-50/training "$V3_RUN/training" --progress --checksum
```

## 6. Test the frozen predictor on 10 unseen questions

This deliberately costs another **10 × 37 = 370 generations** (plus warm-up):
the true bypass outcomes must be measured to test classification. Questions
come from GSM8K's test split and exclude all 50 earlier question texts.

```bash
python -u -m v3.test_unseen \
  --checkpoint "$V3_RUN/training/controller.pt" \
  --output-dir "$V3_RUN/unseen" \
  --rows 10 --split test --seed 42 --max-new-tokens 1024 \
  2>&1 | tee -a "$V3_RUN/logs/unseen.log"
```

Read `unseen/predictor_evaluation.json` and `unseen/summary.json`. No training
or threshold tuning happens on these questions. Classification scores exclude
unscored labels, whose counts remain in the summary. This is held-out GSM8K,
not proof of transfer to a different dataset. Detach/reconnect as in step 3.

After completion:

```bash
rclone copy "$V3_RUN/unseen" gdrive:PSSN2/v3/pilot-50/unseen --progress --checksum
```

To restore before the benchmark on another server:

```bash
rclone copy gdrive:PSSN2/v3/pilot-50/unseen "$V3_RUN/unseen" --progress --checksum
```

## 7. Test real generation: baseline vs controller

This selects another **10 new test questions**, excluding both the 50 original
questions and the 10 unseen-evaluation questions. For each, generate an answer
normally and with the frozen controller: **20 recorded generations**, plus
two short warm-ups. Both modes use the same prompts, decoding settings,
FP16 precision, and output cap. Their run order alternates between questions.

```bash
python -u -m v3.benchmark \
  --checkpoint "$V3_RUN/training/controller.pt" \
  --unseen-dir "$V3_RUN/unseen" \
  --output-dir "$V3_RUN/benchmark" \
  --rows 10 --split test --seed 42 --max-new-tokens 1024 \
  2>&1 | tee -a "$V3_RUN/logs/benchmark.log"
```

Detach/reconnect as in step 3. Read:

```bash
cat "$V3_RUN/benchmark/metrics.json"
```

Metrics include final-answer correctness, incomplete counts, input/output
tokens, synchronized generation time including controller overhead, token
throughput, PyTorch peak allocated/reserved memory, selected layers, actual
MLP bypass counts, and an analytic estimate of bypassed MLP matrix operations.
Generation timing excludes loading/tokenization; prompt prefill is included.
All-question accuracy treats incomplete answers as failures; a separate
completed-only accuracy excludes them. No incomplete trial is used as a
safe training label.

The FLOP estimate is **not** measured device utilization/energy or total-model
FLOPs. For decode-only skipping, at most one of 36 MLPs is bypassed, while
attention/other layers still run. Controller cost can exceed savings. There is
no promised speedup or reduction in generated tokens. Ten questions gives an
exploratory paired comparison; it cannot establish a publication-level gain.

Back up the final results and logs:

```bash
rclone copy "$V3_RUN/benchmark" gdrive:PSSN2/v3/pilot-50/benchmark --progress --checksum
rclone copy "$V3_RUN/logs" gdrive:PSSN2/v3/pilot-50/logs --progress --checksum
```

## 8. Download all results directly to your PC

From **Windows PowerShell**, replace `SERVER_IP` with the server's public IP:

```powershell
scp -r root@SERVER_IP:/root/pssn-work/PSSN2/v3/pilot-50 "C:\Users\LOQ\Downloads\pssn-v3-pilot-50"
```

Or restore all results from Drive on a server/PC with rclone:

```bash
rclone copy gdrive:PSSN2/v3/pilot-50 ./pssn-v3-pilot-50 --progress --checksum
```

`rclone copy` and `scp` copy files; the originals remain. Check the local copy
and Drive backup before deleting the GPU server. No automatic background
upload is used, so Drive cannot block an active generation trial.
