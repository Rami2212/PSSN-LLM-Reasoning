# V6 server setup and complete 100 × 3 experiment

This guide targets the same **Ubuntu 24.04 / Python 3.12 / ROCm 7.2 / MI300X**
environment used by V5. It supplies executable commands; it does not claim that
V6 has already been trained or benchmarked. Run preparation and labeling on the
server before expecting a controller checkpoint or comparison results.

## 1. Access and resources

Accept/request Hugging Face access using the same account for both:

- [facebook/layerskip-llama3-8B](https://huggingface.co/facebook/layerskip-llama3-8B)
- [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

Use a read token for that account. Both checkpoints are gated. The original
base checkpoint is deliberately used, not `Meta-Llama-3-8B-Instruct`.

One GPU is used. The checkpoints load sequentially, each with approximately
16 GB of BF16/FP16 weights plus KV cache/working memory. MI300X has sufficient
capacity; smaller GPUs still need headroom for the full prompt. Plan for about
32 GB of cached weights for the two checkpoints, the CNN/DailyMail download,
the Python environment and experiment archives. A 60+ GB free disk allowance
is a practical starting budget, not a measured total. Use an otherwise idle
GPU for comparable timings. V6 does not quantize either model.

## 2. Connect and upload from Windows

Replace `YOUR_SERVER_IP` with your current server address. In PowerShell:

```powershell
$serverHost = 'YOUR_SERVER_IP'
ssh "root@$serverHost"
```

On the server in Bash:

```bash
sudo apt update
sudo apt install -y git python3.12-venv python3.12-dev tmux rclone util-linux
rocm-smi
amd-smi metric -p -E --json
mkdir -p /root/pssn-work
cd /root/pssn-work
```

Clone if the repository is absent:

```bash
git clone https://github.com/Rami2212/PSSN-LLM-Reasoning.git
```

In a second Windows PowerShell terminal, upload the new folder:

```powershell
$serverHost = 'YOUR_SERVER_IP'
$repo = 'C:\Users\LOQ\Downloads\Uni\PSSN\Repository\PSSN-LLM-Reasoning'
scp -r "$repo\v6" "root@${serverHost}:/root/pssn-work/PSSN-LLM-Reasoning/"
```

V6 is self-contained. No V5 data, models, `.env` or controller needs uploading.
If you want to run the CPU checks on the server too, upload the V6 tests:

```powershell
scp -r "$repo\tests\v6" "root@${serverHost}:/root/pssn-work/PSSN-LLM-Reasoning/tests/"
```

## 3. Persistent session and environment

On the server, outside any existing tmux session:

```bash
tmux new-session -A -s pssn-v6
```

Inside that session:

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
python3.12 -m venv .venv-v6-rocm72
source .venv-v6-rocm72/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl'
python -m pip install -r v6/requirements.txt
# Optional but required for GPU energy/power metrics when ROCm ships these bindings:
if [ -d /opt/rocm/share/amd_smi ]; then python -m pip install /opt/rocm/share/amd_smi; fi
python -m pip check
python -c 'import torch, transformers; assert torch.cuda.is_available(); assert str(torch.version.hip).startswith("7.2"); assert transformers.__version__ == "4.57.1"; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))'
python -c 'import amdsmi; amdsmi.amdsmi_init(); g=amdsmi.amdsmi_get_processor_handles()[0]; print(amdsmi.amdsmi_get_energy_count(g)); amdsmi.amdsmi_shut_down()'
hf auth login
```

The AMD SMI check verifies the cumulative GPU energy counter used for V6 power
measurement. The optional install puts ROCm's Python bindings into the venv;
AMD documents this install path for the ROCm 7.2 Python interface. If the
counter is unsupported by the server/VM or unavailable to the account, V6
continues and records the reason with energy/power as `n/a`.

These wheel commands assume ROCm 7.2 drivers/runtime are already installed.
They follow AMD's [ROCm 7.2 PyTorch installation guidance](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/3rd-party/pytorch-install.html)
and [published Python 3.12 wheels](https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/).
They are not a driver installation procedure. For a CUDA server, install its
compatible CUDA PyTorch build instead and create a distinct experiment run.

The token is entered interactively. A 401/403/gated-repo error must be resolved
with your approved account before either checkpoint can download.

Optional CPU checks, with no model or dataset download:

```bash
python -m unittest discover -s tests/v6 -v
```

## 4. Set experiment and Drive destinations

Configure `gdrive` using [DRIVE_SAVE.md](DRIVE_SAVE.md) before running the
automatic pipeline. Keep this new directory separate from V5:

```bash
export V6_RUN=/root/pssn-work/PSSN2/v6/controller-text-100x3
export V6_REMOTE=gdrive:PSSN2/v6/controller-text-100x3
mkdir -p "$V6_RUN/logs"
rclone mkdir "$V6_REMOTE"
rclone lsd gdrive:PSSN2/v6
```

## 5. Automatic run with a Drive snapshot after each stage

```bash
bash v6/run_pipeline.sh
```

The script takes an exclusive run lock, records dependencies/source, and runs:

1. Freeze revisions, three demonstrations, 100 labeling articles and 100 unseen
   test articles. Main revisions for the original model/dataset resolve to
   exact commit IDs once; the LayerSkip revision starts from V5's pinned commit.
2. Label 100 articles at depths 24/28/32: **300 measured generations**.
3. Fit controller on 70 articles / 210 labels; choose settings on 15 articles.
4. Evaluate the remaining 15 articles / 45 labels without tuning.
5. Compare the three systems on 100 unseen test articles with three timing
   repeats: **900 measured generations**.

Total: 1,200 measured generations plus short warmups. Runtime depends on model
downloads, input/output lengths and GPU performance; no completion time is
promised. All labels remain one greedy output per article/depth.

After each successful stage, `controller_backup.sh` creates and verifies a
Drive snapshot containing that stage and its prerequisites. A command or backup
failure stops the script. Local evidence remains saved. Fix the cause and rerun
the same script to resume/rebuild the matching artifacts. Do not edit code or
change dependency versions midway through a run.

## 6. Manual stage commands

Use these instead of the automatic runner if you want to inspect each stage
before continuing. Do not run them concurrently with the pipeline. Start by
recording the source and environment:

```bash
mkdir -p "$V6_RUN/source-v6" "$V6_RUN/logs"
cp -a v6/. "$V6_RUN/source-v6/"
python -m pip freeze > "$V6_RUN/logs/pip-freeze.txt"
rocm-smi > "$V6_RUN/logs/gpu.txt"
set -o pipefail
```

Prepare and label:

```bash
flock -n "$V6_RUN/.pipeline.lock" python -u -m v6.prepare \
  --output-dir "$V6_RUN/prepared" --rows 100 --test-rows 100 \
  --max-rouge-l-drop 0.02 --min-rouge-l 0.15 \
  2>&1 | tee -a "$V6_RUN/logs/prepared.log"
bash v6/controller_backup.sh "$V6_RUN" prepared "$V6_REMOTE"

flock -n "$V6_RUN/.pipeline.lock" python -u -m v6.controller_label \
  --prepared-dir "$V6_RUN/prepared" --output-dir "$V6_RUN/labels" \
  2>&1 | tee -a "$V6_RUN/logs/labels.log"
cat "$V6_RUN/labels/summary.json"
bash v6/controller_backup.sh "$V6_RUN" labels "$V6_REMOTE"
```

Inspect the saved summaries, completion rates and positive label counts. The
0.15 quality floor and 0.02 allowed drop are declared pilot choices. If the
training labels contain only one class, fitting fails explicitly. Investigate
prompting/termination/full-depth quality, collect more examples, or predeclare
a revised quality rule in a **new** experiment before using its test outputs.

Train and evaluate:

```bash
flock -n "$V6_RUN/.pipeline.lock" python -u -m v6.controller_train \
  --labels-dir "$V6_RUN/labels" --output-dir "$V6_RUN/training" \
  2>&1 | tee -a "$V6_RUN/logs/training.log"
bash v6/controller_backup.sh "$V6_RUN" training "$V6_REMOTE"

flock -n "$V6_RUN/.pipeline.lock" python -u -m v6.controller_evaluate \
  --labels-dir "$V6_RUN/labels" --controller "$V6_RUN/training/controller.joblib" \
  --output-dir "$V6_RUN/evaluation" \
  2>&1 | tee -a "$V6_RUN/logs/evaluation.log"
bash v6/controller_backup.sh "$V6_RUN" evaluation "$V6_REMOTE"
```

Live three-system comparison:

```bash
flock -n "$V6_RUN/.pipeline.lock" python -u -m v6.controller_test \
  --prepared-dir "$V6_RUN/prepared" --controller "$V6_RUN/training/controller.joblib" \
  --output-dir "$V6_RUN/comparison" --repeats 3 \
  2>&1 | tee -a "$V6_RUN/logs/comparison.log"
bash v6/controller_backup.sh "$V6_RUN" comparison "$V6_REMOTE"
cat "$V6_RUN/comparison/comparison.md"
```

For a shorter test use `--repeats 1` and a fresh comparison directory such as
`comparison-once`. Its 300 generations provide less reliable timing evidence.
The supplied backup script expects the standard `comparison` stage name, so
use `comparison` in a separate run tree if you want that snapshot automatically.

## 7. Monitor, detach and resume

Detach with **Ctrl+B**, release, then **D**. Reconnect:

```bash
tmux attach-session -t pssn-v6
```

From a second server terminal:

```bash
pgrep -af '[v]6\.'
cat /root/pssn-work/PSSN2/v6/controller-text-100x3/labels/run_status.json
tail -n 10 /root/pssn-work/PSSN2/v6/controller-text-100x3/logs/labels.log
```

For the test stage:

```bash
cat /root/pssn-work/PSSN2/v6/controller-text-100x3/comparison/run_status.json
cat /root/pssn-work/PSSN2/v6/controller-text-100x3/comparison/progress.json
```

A heartbeat updates every 15 seconds. Check the PID if it becomes stale; status
alone cannot prove a process is alive. The final `status: completed` and
`completed_trials: 900` in `comparison/summary.json` indicate the default test
has finished. Re-enter the venv and export `V6_RUN`/`V6_REMOTE` again after SSH
reconnection if the session was lost. Matching saved generations resume without
rerunning. Generation environment mismatches require a fresh output directory.

## 8. Read results

```bash
cat "$V6_RUN/training/training_summary.json"
cat "$V6_RUN/evaluation/metrics.json"
cat "$V6_RUN/comparison/comparison.md"
cat "$V6_RUN/comparison/summary.json"
```

Use `layerskip_controller_vs_layerskip_full` to assess controller impact.
Use `layerskip_full_vs_base_full` to assess the checkpoint difference and
`layerskip_controller_vs_base_full` for the combined result. Positive time/work
reduction means savings; a negative value means an increase. Check ROUGE,
completion and quality preservation alongside every efficiency claim.

ROUGE is reference overlap, not factual accuracy. No token reduction or speedup
is guaranteed. See [README.md](README.md) for formulas and limits, and
[DRIVE_SAVE.md](DRIVE_SAVE.md) to restore or download the complete evidence.
