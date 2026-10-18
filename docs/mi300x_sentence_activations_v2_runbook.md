# MI300X ROCm 7.2: 5,000-row V2 activations and predictor training

This guide starts with the existing `sentence-training-5000-v2` dataset and
ends with two trained predictors: one using Qwen's final-layer features and
one using features concatenated from all transformer layers. Both feature sets
are made during the same Qwen forward pass. Results are copied to Google Drive
manually after extraction and after each model finishes training.

## What you need to upload

Before connecting to the server, make sure the latest repository version,
including `src/black_hole/sentence_activations_v2.py` and
`src/black_hole/sentence_evaluation_v2.py`, is available there. These are local
changes; a fresh `git clone` will not include them until you commit and push
them, or transfer the updated repository to the server.

The activation command needs this dataset file:

```text
PSSN2/artifacts/black_hole/sentence-training-5000-v2/sentence_training_data.jsonl
```

Copy the **whole** `sentence-training-5000-v2` directory if possible. This also
preserves its manifest, summary, README, exclusions, and source-provenance
notes. The notes are useful for research records but are not sent into Qwen.
The code makes fresh 70/15/15 question-disjoint splits from the JSONL.

If the complete folder is already in Google Drive at
`PSSN2/artifacts/black_hole/sentence-training-5000-v2`, no PC-to-server upload
is needed; the `rclone copy` command below downloads it. Otherwise, upload the
folder to that Drive location first, or transfer it directly to the server.

## 1. Set up the server environment

These commands assume Ubuntu 24.04 with the MI300X driver and ROCm 7.2 already
installed by the server provider. For a fresh ROCm installation, follow the
[AMD ROCm 7.2 Ubuntu instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/install-methods/package-manager/package-manager-ubuntu.html).
The project's Python wheels follow AMD's [ROCm 7.2 PyTorch instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/native_linux/install-pytorch.html).

```bash
sudo apt update
sudo apt install -y git python3.12-venv python3.12-dev rclone tmux
mkdir -p /root/pssn-work
cd /root/pssn-work
git clone https://github.com/Rami2212/PSSN-LLM-Reasoning.git
cd /root/pssn-work/PSSN-LLM-Reasoning
```

If the repository already exists on the server, use that checkout and update it
to the commit that contains V2 instead of cloning a second copy. Confirm that
the V2 command is present before continuing:

```bash
test -f src/black_hole/sentence_activations_v2.py && echo "V2 code found"
```

Create the isolated ROCm environment and verify that PyTorch sees the GPU:

```bash
python3.12 -m venv .venv-rocm72
source .venv-rocm72/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-rocm-7.2.txt
python -m pip check
python -c 'import torch; print("torch", torch.__version__, "HIP", torch.version.hip); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

PyTorch uses the `torch.cuda` API name for both CUDA and ROCm devices. Seeing
`torch.version.hip` start with `7.2` and the MI300X name is the expected result.

## 2. Configure Google Drive once

Create a remote called `gdrive`:

```bash
rclone config
```

Choose `n` (new remote), name it `gdrive`, choose Google Drive, and select a
read/write Drive scope. On the headless server choose `n` for automatic browser
authorization. Run the `rclone authorize "drive" ...` command that rclone prints
on your Windows PC, finish Google sign-in there, then paste the returned token
into the server's `config_token>` prompt. Keep that token private. See
[rclone's headless setup guide](https://rclone.org/remote_setup/).

Check that the remote can see the dataset:

```bash
rclone lsd gdrive:PSSN2/artifacts/black_hole
rclone lsf gdrive:PSSN2/artifacts/black_hole/sentence-training-5000-v2
```

## 3. Download and verify the 5,000-row dataset on the server

Keep working files on the server disk, not inside the Drive mount:

```bash
export REPO=/root/pssn-work/PSSN-LLM-Reasoning
export PSSN_DATA=/root/pssn-work/PSSN2
export DATASET_DIR="$PSSN_DATA/artifacts/black_hole/sentence-training-5000-v2"
export ACT_DIR="$PSSN_DATA/artifacts/black_hole/sentence-training-5000-v2/activations-v2"
export FINAL_DIR="$PSSN_DATA/artifacts/black_hole/sentence-training-5000-v2/training-final-layer"
export ALL_DIR="$PSSN_DATA/artifacts/black_hole/sentence-training-5000-v2/training-all-layers"
export LOG_DIR=/root/pssn-work/logs/sentence-activations-v2
mkdir -p "$DATASET_DIR" "$LOG_DIR"

rclone copy \
  gdrive:PSSN2/artifacts/black_hole/sentence-training-5000-v2 \
  "$DATASET_DIR" --progress --checksum

test -f "$DATASET_DIR/sentence_training_data.jsonl"
wc -l "$DATASET_DIR/sentence_training_data.jsonl"
```

The line count should be `5000`. If it is not, stop and check that the Drive
folder is complete before extracting activations.

## 4. Extract both activation feature sets

Run this from the repository root. Use a new output directory for a fresh run;
V2 will not overwrite an existing one. The revision below pins the Qwen3-4B
checkpoint used in the earlier project experiments.

Start a persistent tmux session before the long extraction. Run extraction,
training, and evaluation inside this same session:

```bash
tmux new -s pssn-v2
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-rocm72/bin/activate
```

To leave the server while a command keeps running, press `Ctrl+B`, release,
then press `D`. This detaches without stopping the process. Reconnect with
`tmux attach -t pssn-v2`; check for the session with `tmux ls`. Do not create
another session with the same name if it is already running.

```bash
cd "$REPO"
source .venv-rocm72/bin/activate
mkdir -p "$LOG_DIR"
python -u -m src.black_hole.sentence_activations_v2 \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$ACT_DIR" \
  --model-id Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --seed 42 \
  --require-cuda 2>&1 | tee "$LOG_DIR/activation-extraction.log"
```

This loads Qwen3-4B and records activations for the question, previous
reasoning prefix, and target sentence. It does not generate a new answer. The
original model that produced a dataset row is not used here: these are Qwen3
activations for the supplied text, so describe them as Qwen3 representations
of the dataset text (not the source model's internal activations).

The output contains:

```text
activations-v2/train.npz, validation.npz, test.npz   # final transformer layer
activations-v2/all_layers/train.npz, validation.npz, test.npz
activations-v2/train.jsonl, validation.jsonl, test.jsonl
activations-v2/dataset_manifest.json
```

The manifest reports split counts, feature widths, hashes, model revision, and
the number of transformer layers. All-layer features are much wider and need
more server disk space. Check the manifest before continuing:

```bash
cat "$ACT_DIR/dataset_manifest.json"
```

### Back up the created activation dataset to Drive

Wait for extraction to finish successfully, then copy and verify the full
activation directory. `rclone copy` leaves the server files in place.

```bash
rclone copy "$ACT_DIR" \
  gdrive:PSSN2/artifacts/black_hole/sentence-training-5000-v2/activations-v2 \
  --progress --checksum
rclone check "$ACT_DIR" \
  gdrive:PSSN2/artifacts/black_hole/sentence-training-5000-v2/activations-v2 \
  --one-way
```

Proceed only if `rclone check` reports no differences.

## 5. Train the final-layer predictor

The trainer fits on `train.npz`, chooses the checkpoint using validation F1,
and leaves `test.npz` unused. Run both feature variants into separate folders
inside the same tmux session:

```bash
cd "$REPO"
source .venv-rocm72/bin/activate
python -u -m src.black_hole.trainer \
  --data-dir "$ACT_DIR" \
  --output-dir "$FINAL_DIR" \
  --device cuda 2>&1 | tee "$LOG_DIR/train-final-layer.log"
```

Check that `training_summary.json` says `"status": "completed"` and that
`best_model.pt` exists:

```bash
cat "$FINAL_DIR/training_summary.json"
ls -lh "$FINAL_DIR"
```

### Back up the final-layer model to Drive

```bash
rclone copy "$FINAL_DIR" \
  gdrive:PSSN2/artifacts/black_hole/sentence-training-5000-v2/training-final-layer \
  --progress --checksum
rclone check "$FINAL_DIR" \
  gdrive:PSSN2/artifacts/black_hole/sentence-training-5000-v2/training-final-layer \
  --one-way
```

## 6. Train the all-layer predictor

This is a separate model trained on the all-layer features from the exact same
rows and splits:

```bash
python -u -m src.black_hole.trainer \
  --data-dir "$ACT_DIR/all_layers" \
  --output-dir "$ALL_DIR" \
  --device cuda 2>&1 | tee "$LOG_DIR/train-all-layers.log"
```

Check and back up the second model:

```bash
cat "$ALL_DIR/training_summary.json"
ls -lh "$ALL_DIR"
rclone copy "$ALL_DIR" \
  gdrive:PSSN2/artifacts/black_hole/sentence-training-5000-v2/training-all-layers \
  --progress --checksum
rclone check "$ALL_DIR" \
  gdrive:PSSN2/artifacts/black_hole/sentence-training-5000-v2/training-all-layers \
  --one-way
```

## 7. Test both final predictors on held-out questions

Run this only after both training runs finish. The test questions were kept out
of training and validation. The evaluator loads each model's saved
training-only normalization and uses the fixed 0.5 threshold; it does not tune
the model or threshold using test data.

```bash
export FINAL_EVAL="$PSSN_DATA/artifacts/black_hole/sentence-training-5000-v2/evaluation-final-layer"
export ALL_EVAL="$PSSN_DATA/artifacts/black_hole/sentence-training-5000-v2/evaluation-all-layers"

python -u -m src.black_hole.sentence_evaluation_v2 \
  --data-dir "$ACT_DIR" \
  --checkpoint "$FINAL_DIR/best_model.pt" \
  --split-jsonl "$ACT_DIR/test.jsonl" \
  --output-dir "$FINAL_EVAL" \
  --device cuda 2>&1 | tee "$LOG_DIR/evaluate-final-layer.log"

python -u -m src.black_hole.sentence_evaluation_v2 \
  --data-dir "$ACT_DIR/all_layers" \
  --checkpoint "$ALL_DIR/best_model.pt" \
  --split-jsonl "$ACT_DIR/test.jsonl" \
  --output-dir "$ALL_EVAL" \
  --device cuda 2>&1 | tee "$LOG_DIR/evaluate-all-layers.log"
```

Each evaluation writes `test_evaluation_summary.json` and
`test_predictions.jsonl`. The summary includes accuracy, low-value precision,
recall and F1, ROC-AUC, PR-AUC, confusion matrix, and the rate at which
answer-sensitive examples would be wrongly predicted as low-value. It also
checks that checkpoint training hashes, test rows, labels, and question groups
match the saved files.

Copy and verify the test outputs in Drive:

```bash
rclone copy "$FINAL_EVAL" \
  gdrive:PSSN2/artifacts/black_hole/sentence-training-5000-v2/evaluation-final-layer \
  --progress --checksum
rclone check "$FINAL_EVAL" \
  gdrive:PSSN2/artifacts/black_hole/sentence-training-5000-v2/evaluation-final-layer \
  --one-way
rclone copy "$ALL_EVAL" \
  gdrive:PSSN2/artifacts/black_hole/sentence-training-5000-v2/evaluation-all-layers \
  --progress --checksum
rclone check "$ALL_EVAL" \
  gdrive:PSSN2/artifacts/black_hole/sentence-training-5000-v2/evaluation-all-layers \
  --one-way
```

## Finished outputs and interpretation

Drive will contain the input 5,000-row dataset, extracted final/all-layer
features, both trained model folders, and both held-out test reports. Each
model folder includes `best_model.pt`, `final_model.pt`, `last_checkpoint.pt`,
config, training history, and summary. Compare the models on the same held-out
questions. The test result estimates performance on new questions from this
dataset; it is not an external-dataset or new-domain evaluation. The labels
were transferred into this dataset and are not independently validated for
Qwen3, so describe that limitation when reporting results.

The extraction, training, and evaluation `tee` commands save logs under
`/root/pssn-work/logs/sentence-activations-v2/`. If your server connection
drops, reconnect with `tmux attach -t pssn-v2`; the command continues inside
tmux while you are disconnected.
