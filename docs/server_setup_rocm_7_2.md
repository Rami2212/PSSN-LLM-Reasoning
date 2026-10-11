# W3003 server setup — ROCm 7.2 on Ubuntu 24.04 and Google Drive

These instructions run W3003 on a Linux server using the existing Week 2 data
and W3001 selection. Complete notebook 10 first. No rerun of notebooks 01–09
is needed if their saved artifacts already exist.

## 1. Install the software environment

Start from your Ubuntu 24.04 x86_64 server provisioned with ROCm 7.2.0.
For a fresh system installation, follow [AMD's versioned ROCm 7.2 Ubuntu
guide](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/install-methods/package-manager/package-manager-ubuntu.html).
The project uses Python 3.12 and AMD's PyTorch 2.9.1 / ROCm 7.2.0 wheels from
the [official native PyTorch instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/native_linux/install-pytorch.html).
Only Torch and Triton are needed for these text experiments; the vision and
audio packages in AMD's example are unnecessary here. The server requirements
pin the AMD wheel URLs and NumPy, while retaining the project's other dependencies.

```bash
sudo apt update
sudo apt install -y git python3.12-venv python3.12-dev rclone fuse3 tmux
mkdir -p ~/pssn-work
cd ~/pssn-work
git clone https://github.com/Rami2212/PSSN-LLM-Reasoning.git
cd PSSN-LLM-Reasoning
python3.12 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
python3.12 -m venv .venv-rocm72
source .venv-rocm72/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-rocm-7.2.txt
python -m pip check
python -c 'import torch; assert str(torch.version.hip).startswith("7.2."), torch.version.hip; assert torch.cuda.is_available(), "AMD device is not accessible"; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))'
```

If the repository is already on the server, use that checkout with the updated
W3003 files instead of cloning another copy.

Use this new virtual environment instead of the former ROCm 10.1 environment.
PyTorch exposes AMD devices through `torch.cuda`; the experiment code detects
ROCm using `torch.version.hip`, so the Python runner works with ROCm 7.2 without
a version-specific inference change. Environment snapshots and live run status
record the actual installed HIP/PyTorch versions on each invocation.

## 2. Connect and mount Google Drive

Configure a remote named `gdrive`, choose the Google Drive backend and permit
access to read inputs and write results:

```bash
rclone config
```

On a server without a browser, answer `n` to automatic browser authorization.
Run `rclone authorize "drive"` on your own computer, complete its browser login,
and paste the returned authorization result into the server's prompt. Follow
[rclone's remote authorization guide](https://rclone.org/remote_setup/) and
[Google Drive backend documentation](https://rclone.org/drive/).

```bash
mkdir -p ~/gdrive
rclone lsd gdrive:
rclone mount gdrive: ~/gdrive --read-only --vfs-cache-mode full --daemon
ls ~/gdrive/PSSN2/artifacts/expanded_baseline
```

The mount exposes your Drive's contents directly; there is no `MyDrive` layer
inside `~/gdrive`. The [mount documentation](https://rclone.org/commands/rclone_mount/)
explains its caching behavior. The runner uploads closed checkpoint files using
`rclone copyto`, rather than depending on delayed mount writes.

## 3. Run a small check, then continue the same run

The Python entry script finds the latest W3001 selection file on the Drive
mount. Keep the same run name to resume. Run the small check first:

```bash
tmux new -s pssn-week3
cd ~/pssn-work/PSSN-LLM-Reasoning
source .venv-rocm72/bin/activate
python -m src.black_hole.run_experiments \
  --input /home/YOUR_USER/gdrive/PSSN2/artifacts/expanded_baseline/YOUR_RUN/black_hole_trace_selection/YOUR_TIMESTAMP/eligible_traces.jsonl \
  --output-dir /home/YOUR_USER/pssn-work/checkpoints/w3-003-run1 \
  --drive-output gdrive:PSSN2/artifacts/black_hole/w3-003-run1 \
  --limit 2
```

Equivalent Python-file command:

```bash
python notebooks/11_black_hole_removal_experiments.py --limit 2
```

Check the two saved result files in Drive, then run the same command without
`--limit 2`. Completed attempts are skipped. Detach with Ctrl+B, then D; reconnect
with `tmux attach -t pssn-week3`. Use one active process per run folder.

The runner creates and continuously updates
`PSSN2/artifacts/black_hole/w3-003-run1/run_status.json`. To view it from the
server while the run continues:

```bash
watch -n 10 'rclone cat gdrive:PSSN2/artifacts/black_hole/w3-003-run1/run_status.json'
```

`status` / `is_running` show the lifecycle and `phase` shows setup, model load,
generation, checkpoint saving, or completion. `progress` includes completed,
pending, correct, incorrect, incomplete, and failed counts plus the current
experiment. `metadata` captures the input hash, environment, generation setup,
model revision, and code hashes. A heartbeat is uploaded every 45 seconds. If
the process is forcibly killed or the machine loses power, it cannot write a
stopped marker; treat `is_running: true` with a heartbeat older than
`heartbeat_stale_after_seconds` (120 seconds) as not currently running. Ctrl+C
and SIGTERM write `status: stopped`; reaching a `--limit` with work remaining
writes `status: paused`; finishing all selected work writes `status: completed`.

To run all pending removals, omit `--limit 2` from either command.

The runner reads the baseline generation settings, including the 1,024-token
cap, and records the resolved model and tokenizer revisions. Use `--revision
COMMIT_HASH` if you have an explicit baseline model revision to preserve.

## 4. Optional notebook access

```bash
python -m ipykernel install --user --name pssn-week3
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

From your computer, create an SSH tunnel with `ssh -L 8888:127.0.0.1:8888
YOUR_USER@SERVER_ADDRESS`, then open the token-bearing local URL printed by
Jupyter. Choose the `pssn-week3` kernel and open
`notebooks/11_black_hole_removal_experiments.ipynb`.

## Saved outputs and interruptions

Drive folder: `PSSN2/artifacts/black_hole/w3-003-run1/`.

- `records/<experiment_id>.json`: complete removal metadata, original baseline,
  modified context, newly generated continuation, answer score, token counts,
  timing and runtime metrics. Each attempt is uploaded before the next begins.
- `manifest.json`: source-data hash, protocol, settings and revision identities.
- `sessions/`: environment, package versions, repository and code hashes for
  each invocation.
- `run_status.json`: live lifecycle/progress and run setup metadata, refreshed
  every 45 seconds and at each experiment/checkpoint transition.
- `state_removal_results.jsonl` and `removal_summary.json`: combined exports
  rebuilt when an invocation finishes, including a limited run.

If generation is interrupted, previously completed records remain resumable.
If upload fails, the runner stops with the record saved locally; rerunning
uploads it before continuing. A computation interrupted before checkpointing is
rerun. A new server restores checkpoint records from the same Drive output folder.
Incomplete and failed attempts retain `is_correct=null`. Failed attempts are
preserved and skipped on resume; use a separate run folder for a retry study.

The setup commands are based on the versioned AMD ROCm 7.2 documentation checked
on 2026-10-11. Remote installation and authorization must be performed on your server.
