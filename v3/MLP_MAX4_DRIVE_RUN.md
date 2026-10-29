# MI300X / ROCm 7.2: run the 100-question MLP max-four comparison

This reuses the saved four-MLP-group controller. It bypasses selected MLP
modules only; it does not bypass whole transformer layers. The controller
checkpoint is trained for MLP-group actions.

The run compares normal Qwen generation with controller-guided generation on
100 fresh GSM8K test questions: 200 answer generations plus warm-ups. It
measures answer correctness, output tokens, wall time, peak memory, and
estimated MLP matrix work avoided. Training tested groups individually, so
combining up to four remains an exploratory test.

## 1. Create and set up the server

Choose a server image with an MI300X, Ubuntu 24.04, and ROCm 7.2 installed.
For a bare Ubuntu image, follow [AMD's ROCm 7.2 Ubuntu installation
guide](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/install-methods/package-manager/package-manager-ubuntu.html).
Do not replace a cloud image's working GPU driver just to install Python.
AMD's [ROCm 7.2 PyTorch guide](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/3rd-party/pytorch-install.html)
documents the supported framework setup.

In the server terminal, install tools and clone the repository:

~~~~bash
sudo apt update
sudo apt install -y git python3.12-venv python3.12-dev tmux rclone
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
python -m pip install -r v3/requirements.txt
python -m pip check
python -c 'import torch; assert str(torch.version.hip).startswith("7.2"), torch.version.hip; assert torch.cuda.is_available(); print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))'
~~~~

If the repository already exists on the server, do not clone it again. Update
it or use its existing checkout, then activate the existing ROCm environment.

## 2. Upload the updated MLP benchmark files

From Windows PowerShell, replace SERVER_IP with the server IP. Add
-P PORT if SSH uses a non-default port.

~~~~powershell
$repo = "C:\Users\LOQ\Downloads\Uni\PSSN\Repository\PSSN-LLM-Reasoning"
scp "$repo\v3\block_common.py" root@SERVER_IP:/root/pssn-work/PSSN-LLM-Reasoning/v3/
scp "$repo\v3\block_benchmark.py" root@SERVER_IP:/root/pssn-work/PSSN-LLM-Reasoning/v3/
~~~~

## 3. Configure Google Drive on the server

Do this once per server. Start the interactive setup:

~~~~bash
rclone config
~~~~

At the prompts, choose a new remote named gdrive, Google Drive storage, scope
1 (read/write), leave client ID, client secret, and service-account file
blank, choose no for advanced config, and choose no for auto config on the
headless server. The server prints an exact rclone authorize command.

On your Windows PC, install rclone if needed and run that exact command in
PowerShell. Finish the browser sign-in, copy the returned authorization token,
and paste it at the server's config_token prompt. Keep the token private; do
not put it in the repository or chat. Finish configuration, then verify:

~~~~bash
rclone lsd gdrive:PSSN2
~~~~

See [rclone's remote setup guide](https://rclone.org/remote_setup/) for the
browser authorization step.

## 4. Pull the saved controller and prior unseen-question list

The benchmark needs the MLP controller and the prior unseen question IDs.
Only these artifacts are copied from Drive, not the entire PSSN2 folder.

~~~~bash
tmux new-session -A -s pssn-v3-max4
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-v3-rocm72/bin/activate
export V3_ROOT=/root/pssn-work/PSSN2/v3
export CHECKPOINT="$V3_ROOT/block4-pilot-100/training/controller.pt"
export PRIOR_UNSEEN="$V3_ROOT/block4-pilot-100/unseen"
export OUT_DIR="$V3_ROOT/block4-max4-100q-final-v1-controller"
mkdir -p "$V3_ROOT/block4-pilot-100/training" "$PRIOR_UNSEEN"
rclone copy gdrive:PSSN2/v3/block4-pilot-100/training \
  "$V3_ROOT/block4-pilot-100/training" --progress --checksum
rclone copy gdrive:PSSN2/v3/block4-pilot-100/unseen \
  "$PRIOR_UNSEEN" --progress --checksum
test -f "$CHECKPOINT" || { echo "Controller checkpoint not found"; exit 1; }
test -f "$PRIOR_UNSEEN/questions.json" || { echo "Prior unseen question file not found"; exit 1; }
mkdir -p "$OUT_DIR/logs"
set -o pipefail
~~~~

The prior question list prevents repeating questions used in the earlier
unseen evaluation. Qwen3-4B is downloaded from Hugging Face automatically on
the first run if it is not already cached. The old MLP controller itself is
pulled from Drive by the rclone commands above.

## 5. Run the comparison

~~~~bash
python -u -m v3.block_benchmark \
  --checkpoint "$CHECKPOINT" \
  --unseen-dir "$PRIOR_UNSEEN" \
  --output-dir "$OUT_DIR/benchmark-max4-100q" \
  --rows 100 --split test --seed 42 \
  --max-new-tokens 1024 --max-removal-groups 4 \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  2>&1 | tee -a "$OUT_DIR/logs/benchmark-max4-100q.log"
~~~~

The benchmark excludes questions used for controller training and the prior
unseen run. It selects 100 new GSM8K test questions, then generates one normal
and one controller-guided answer for each.

## 6. Copy results back to Drive and check them

~~~~bash
rclone copy "$OUT_DIR" gdrive:PSSN2/v3/block4-max4-100q-final-v1-controller \
  --progress --checksum
cat "$OUT_DIR/benchmark-max4-100q/run_status.json"
cat "$OUT_DIR/benchmark-max4-100q/metrics.json"
~~~~

The output folder is new so it will not overwrite the earlier 10-question
benchmark. Expect roughly 15 minutes of generation based on the previous
10-question run; model loading, first-run compilation, and longer answers may
add time.

Detach from tmux without stopping the job with Ctrl+B, then D. Reconnect with:

~~~~bash
tmux attach -t pssn-v3-max4
~~~~
