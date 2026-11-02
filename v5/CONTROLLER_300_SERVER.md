# V5 controller run: 100 questions × 3 depths

This setup labels **100 distinct GSM8K questions**. Each question is answered
once at 24, 28, and 32 layers: **300 measured labeling generations total**.
The label for each run is `1` when the answer is complete and correct, and `0`
when it is wrong or incomplete. The controller learns from question text plus
the proposed depth, then chooses the shallowest depth it predicts is safe.

This replaces the earlier quota plan of finding 100 questions in each target
depth class. Use a fresh run folder named `controller-100x3`; the old outputs
and prior smoke/speculative results are separate and unchanged.

Model: `facebook/layerskip-llama3-8B`, revision
`3b9e03eaf3cc90812a7bd2b69106b626d63bac1f`. Actions are direct exits after
24, 28, or 32 transformer layers. No model fine-tuning or speculative decoding
is done.

| Stage | What it does | Output | GPU |
|---|---|---|---|
| Label | Generate 100 questions at all 3 depths | 100 questions, 300 saved trials and observed safe/unsafe labels | Yes |
| Train | Train on question-depth pairs; split by question | Frozen text controller | No |
| Evaluate | Score its predictions and simulated routing on held-out labeled questions | Classification and answer-accuracy metrics | No |
| Unseen test | Compare full depth with controller on 100 new test questions | 200 live generations plus timing, token, and memory metrics | Yes |

The 100 labeled questions split 70/15/15 by question: 70 questions (210
question-depth trials) for training, 15 (45 trials) for validation, and 15
(45 trials) for held-out evaluation. All three observations for one question
stay in one split. The label set is the actual answer outcome at each depth;
there is no requirement to find balanced target-depth classes. Training needs
both safe and unsafe examples in its training split. If the collector happens
to produce only one outcome class, training stops and the label counts must be
reviewed; it does not invent examples.

The validation set selects regularization using routed answer accuracy. At run
time, the controller predicts answer-preservation probability for each depth,
chooses the shallowest whose score is at least 0.5, and falls back to depth 32
if none reaches 0.5. This is a pilot heuristic, not a safety guarantee. A
100-question test is exploratory, not enough for a strong general claim.

## 1. Upload the code from Windows PowerShell

Server IP: `134.199.192.72`. This assumes the repository already exists there.

```powershell
$repo = 'C:\Users\LOQ\Downloads\Uni\PSSN\Repository\PSSN-LLM-Reasoning'
scp "$repo\v5\*.py" "$repo\v5\*.md" "$repo\v5\requirements.txt" "$repo\v5\controller_backup.sh" root@134.199.192.72:/root/pssn-work/PSSN-LLM-Reasoning/v5/
ssh root@134.199.192.72
```

For a new server, install git/venv/tmux/rclone and clone first, then upload:

```bash
apt update
apt install -y git python3.12-venv python3.12-dev tmux rclone
mkdir -p /root/pssn-work
cd /root/pssn-work
git clone https://github.com/Rami2212/PSSN-LLM-Reasoning.git
mkdir -p /root/pssn-work/PSSN-LLM-Reasoning/v5
```

## 2. Start a persistent server session

From a server shell outside tmux:

```bash
tmux new-session -A -s pssn-v5-controller
```

If already inside tmux, continue in the current session; do not nest another.

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-v5-rocm72/bin/activate
python -m pip install -r v5/requirements.txt
python -m pip check
```

For a new Ubuntu 24.04 / ROCm 7.2 venv, with ROCm drivers already installed:

```bash
python3.12 -m venv .venv-v5-rocm72
source .venv-v5-rocm72/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl'
python -m pip install -r v5/requirements.txt
python -m pip check
```

Check the runtime and authenticate to Hugging Face using the account approved
for the gated LayerSkip model:

```bash
python -c 'import torch, transformers; assert torch.cuda.is_available(); assert str(torch.version.hip).startswith("7.2"); assert transformers.__version__ == "4.57.1"; print(torch.cuda.get_device_name(0))'
hf auth login
```

An existing `HF_TOKEN` also works. Do not save the token in a script or log.

## 3. Configure Drive and the output folder

If `rclone lsd gdrive:PSSN2` succeeds, the Drive remote is ready. Otherwise
run `rclone config`, create remote `gdrive`, select Google Drive and scope `1`,
then finish the headless authorization flow using `rclone authorize` on the PC.

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-v5-rocm72/bin/activate
export CTRL_RUN=/root/pssn-work/PSSN2/v5/controller-100x3
export CTRL_REMOTE=gdrive:PSSN2/v5/controller-100x3
mkdir -p "$CTRL_RUN/logs" "$CTRL_RUN/source-v5"
rclone mkdir "$CTRL_REMOTE"
cp v5/*.py v5/*.md v5/requirements.txt v5/controller_backup.sh "$CTRL_RUN/source-v5/"
python -m pip freeze > "$CTRL_RUN/logs/pip-freeze.txt"
rocm-smi > "$CTRL_RUN/logs/rocm-smi.txt"
set -o pipefail
```

## 4. Label: 100 questions × 3 depths

```bash
flock -n "$CTRL_RUN/.labels.lock" python -u -m v5.controller_label \
  --output-dir "$CTRL_RUN/labels" \
  --rows 100 --seed 42 --max-new-tokens 1024 \
  2>&1 | tee -a "$CTRL_RUN/logs/labels.log"
cat "$CTRL_RUN/labels/summary.json"
```

Expect `questions_labeled: 100` and `generation_trials: 300`. The runner also
does three short, unscored warm-up generations (one per depth). Each labeled
generation and each question label is saved locally as it finishes. Progress
is in `labels/run_status.json`; Ctrl+C retains completed trials. Repeat the
same command to resume in the same folder, provided code, settings, model
revision, environment, and GPU are unchanged.

Back up the completed stage to Drive:

```bash
bash v5/controller_backup.sh "$CTRL_RUN" labels "$CTRL_REMOTE"
```

Check the per-depth safe counts in `labels/summary.json`. These count actual
complete correct answers; unsafe or incomplete outputs remain labeled `0`.

## 5. Train the controller

```bash
flock -n "$CTRL_RUN/.training.lock" python -u -m v5.controller_train \
  --labels-dir "$CTRL_RUN/labels" \
  --output-dir "$CTRL_RUN/training" --seed 42 \
  2>&1 | tee -a "$CTRL_RUN/logs/training.log"
cat "$CTRL_RUN/training/training_summary.json"
bash v5/controller_backup.sh "$CTRL_RUN" training "$CTRL_REMOTE"
```

Training fits word and character TF-IDF features plus the candidate depth. The
split is by question: training uses 70 questions/210 outcomes, validation uses
15/45 and selects regularization, and test uses 15/45. If the training outcomes
contain only safe or only unsafe labels, it stops and explains the issue.

## 6. Evaluate on the held-out labeled questions

```bash
flock -n "$CTRL_RUN/.evaluation.lock" python -u -m v5.controller_evaluate \
  --labels-dir "$CTRL_RUN/labels" \
  --controller "$CTRL_RUN/training/controller.joblib" \
  --output-dir "$CTRL_RUN/evaluation" \
  2>&1 | tee -a "$CTRL_RUN/logs/evaluation.log"
cat "$CTRL_RUN/evaluation/metrics.json"
bash v5/controller_backup.sh "$CTRL_RUN" evaluation "$CTRL_REMOTE"
```

This reports safe/unsafe classification and the answer accuracy from routing
on the 15 held-out questions. It also shows each fixed-depth result. Generation
time is cached diagnostic evidence, not a fresh speed benchmark.

## 7. Test on 100 new GSM8K test questions

```bash
flock -n "$CTRL_RUN/.unseen.lock" python -u -m v5.controller_test \
  --controller "$CTRL_RUN/training/controller.joblib" \
  --output-dir "$CTRL_RUN/unseen-test" --rows 100 --seed 2026 \
  2>&1 | tee -a "$CTRL_RUN/logs/unseen-test.log"
cat "$CTRL_RUN/unseen-test/summary.json"
bash v5/controller_backup.sh "$CTRL_RUN" unseen-test "$CTRL_REMOTE"
```

The test selects 100 questions from GSM8K test, with no correctness filtering.
It runs 100 full-depth and 100 controller generations, alternating which runs
first. The controller chooses one depth from question text before generation.
Results report correctness, completion, tokens, time including routing, peak
memory, selected depths, and estimated layer-token work. This live comparison
is the test of whether the controller helps; smaller depth alone does not
prove a speedup.

## 8. Detach, reconnect, monitor, and resume

Detach with **Ctrl+B**, then **D**. Reconnect from PowerShell:

```powershell
ssh root@134.199.192.72
```

On the server:

```bash
tmux attach-session -t pssn-v5-controller
```

Status can also be checked in another terminal:

```bash
pgrep -af '[v]5.controller_'
cat /root/pssn-work/PSSN2/v5/controller-100x3/labels/run_status.json
cat /root/pssn-work/PSSN2/v5/controller-100x3/unseen-test/run_status.json
```

For a new shell, repeat `cd`, venv activation, and the two `export` commands.
Resume labels or unseen-test by re-running the same command. Stage backups run
after each stage and verify the archive on Drive; local files are retained.

## 9. Download results to the PC

After all stages complete, archive the run on the server:

```bash
tar -C "$CTRL_RUN" -czf /root/pssn-work/v5-controller-100x3.tar.gz \
  labels training evaluation unseen-test logs source-v5
```

Then in Windows PowerShell:

```powershell
scp root@134.199.192.72:/root/pssn-work/v5-controller-100x3.tar.gz "$env:USERPROFILE\Downloads\"
New-Item -ItemType Directory -Force "$env:USERPROFILE\Downloads\v5-controller-100x3"
tar -xzf "$env:USERPROFILE\Downloads\v5-controller-100x3.tar.gz" -C "$env:USERPROFILE\Downloads\v5-controller-100x3"
```

The source folders remain on the server. Drive stage archives are under
**My Drive / PSSN2 / v5 / controller-100x3**.
