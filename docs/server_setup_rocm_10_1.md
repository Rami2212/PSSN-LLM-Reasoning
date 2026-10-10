# W3003 server setup — ROCm 10.1 and Google Drive

These instructions run W3003 on a Linux server using the existing Week 2 data
and W3001 selection. Complete notebook 10 first. No rerun of notebooks 01–09
is needed if their saved artifacts already exist.

## 1. Install the software environment

Start from a server provisioned with ROCm 10.1.0. For a fresh installation,
follow the distribution-specific selector in the [official AMD ROCm 10.1
installation guide](https://rocm.docs.amd.com/en/latest/install/rocm.html).
Use Python 3.11 or newer from the versions listed in [AMD's PyTorch installation
guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).
The following pins PyTorch to the published ROCm 10.1 build:

```bash
sudo apt update
sudo apt install -y git python3-venv rclone fuse3 tmux
mkdir -p ~/pssn-work
cd ~/pssn-work
git clone https://github.com/Rami2212/PSSN-LLM-Reasoning.git
cd PSSN-LLM-Reasoning
python3 -c 'import sys; assert sys.version_info >= (3, 11), "Use Python 3.11 or newer"'
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ 'torch[device-all]==2.13.0+rocm10.1.0'
python -m pip install -r requirements.txt jupyterlab ipykernel
python -c 'import torch; assert "rocm10.1" in torch.__version__, torch.__version__; assert torch.version.hip and torch.cuda.is_available(); print(torch.__version__, torch.version.hip)'
```

If the repository is already on the server, use that checkout with the updated
W3003 files instead of cloning another copy.

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
source .venv/bin/activate
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
- `state_removal_results.jsonl` and `removal_summary.json`: combined exports
  rebuilt when an invocation finishes, including a limited run.

If generation is interrupted, previously completed records remain resumable.
If upload fails, the runner stops with the record saved locally; rerunning
uploads it before continuing. A computation interrupted before checkpointing is
rerun. A new server restores checkpoint records from the same Drive output folder.
Incomplete and failed attempts retain `is_correct=null`. Failed attempts are
preserved and skipped on resume; use a separate run folder for a retry study.

The setup commands are based on the official documentation checked on
2026-10-11. Remote installation and authorization must be performed on your server.
