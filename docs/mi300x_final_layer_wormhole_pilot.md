# MI300X pilot: final-layer predictor vs ordinary Qwen answers

Run a small paired pilot on 10 GSM8K test questions. It compares ordinary
Qwen3-4B answers with answers produced after the final-layer sentence
predictor scores and removes generated reasoning sentences at the fixed
probability threshold **0.5**. This run does not use the previous 10% safety
rule or a validation-selected threshold.

For each question, the ordinary Qwen response supplies the draft trace for the
predictor arm too. The predictor scores the same trace, then Qwen gets the
question and the retained sentences and generates a final answer. This pairing
helps isolate the text-filtering effect. It is not a direct intervention on
Qwen's hidden/private reasoning. The predictor's labels were transferred from
another dataset, so treat the result as a pilot rather than proof of safe
reasoning removal.

The script selects 10 deterministic, unseen questions from the official
GSM8K test split, excluding normalized question text found in the predictor's
saved train/validation/test JSONL files. It never sends reference answers to
Qwen. Each question result is appended and flushed immediately, so completed
rows remain on the server if the process stops. Rerunning with the same output
directory resumes saved rows.

## Files already required on the server

The existing server data and model are reused:

```text
/root/pssn-work/PSSN2/artifacts/black_hole/sentence-training-5000-v2/
  activations-v2/{train,validation,test}.jsonl
  training-final-layer/best_model.pt
```

Upload only the new file `src/black_hole/wormhole_pilot.py` from this repo to
the matching path in the server checkout:

```text
/root/pssn-work/PSSN-LLM-Reasoning/src/black_hole/wormhole_pilot.py
```

No activation NPZ files need to be uploaded for this pilot. The script computes
sentence activations online using the cached/downloadable pinned Qwen3-4B
revision and the existing final-layer checkpoint. The server environment must
already have the repository ROCm 7.2 virtual environment and requirements.

This ROCm requirements file uses NumPy 1.26.4. If `datasets` fails while
importing PyArrow with a “requires NumPy 2” message, fix only those two packages
inside the active virtual environment (do not upgrade NumPy or PyTorch):

```bash
python -m pip install --no-deps "datasets==3.6.0" "pyarrow==19.0.1"
python -c "import numpy, datasets, pyarrow; print('numpy', numpy.__version__, '| datasets', datasets.__version__, '| pyarrow', pyarrow.__version__)"
```

Expected versions are `1.26.4`, `3.6.0`, and `19.0.1` respectively.

## Run on the server

First verify the existing files and upload:

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-rocm72/bin/activate
python -m pip install --no-deps "datasets==3.6.0" "pyarrow==19.0.1"
python -c "import numpy, datasets, pyarrow; print('numpy', numpy.__version__, '| datasets', datasets.__version__, '| pyarrow', pyarrow.__version__)"
export PSSN_DATA=/root/pssn-work/PSSN2
export PILOT_DIR="$PSSN_DATA/artifacts/black_hole/sentence-training-5000-v2"
test -f src/black_hole/wormhole_pilot.py && echo "pilot code found"
test -f "$PILOT_DIR/training-final-layer/best_model.pt" && echo "predictor found"
for file in train validation test; do test -s "$PILOT_DIR/activations-v2/$file.jsonl" || exit 1; done
```

Run in tmux so it continues if the terminal disconnects. If currently inside
another tmux session, detach first with `Ctrl-b`, then `d`, and create this
session from the regular shell:

```bash
tmux new -s w4-wormhole-pilot
```

Inside tmux, run:

```bash
cd /root/pssn-work/PSSN-LLM-Reasoning
source .venv-rocm72/bin/activate
export PSSN_DATA=/root/pssn-work/PSSN2
export PILOT_DIR="$PSSN_DATA/artifacts/black_hole/sentence-training-5000-v2"
export OUT_DIR="$PILOT_DIR/wormhole-pilot-final-layer-threshold-0.5-10q"
mkdir -p /root/pssn-work/logs
python -u -m src.black_hole.wormhole_pilot \
  --checkpoint "$PILOT_DIR/training-final-layer/best_model.pt" \
  --activation-jsonl-dir "$PILOT_DIR/activations-v2" \
  --output-dir "$OUT_DIR" \
  --rows 10 \
  --seed 42 \
  --threshold 0.5 \
  --max-new-tokens 1024 \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  2>&1 | tee /root/pssn-work/logs/wormhole-pilot-final-layer.log
```

Detach with `Ctrl-b`, then `d`. Reconnect later with:

```bash
tmux attach -t w4-wormhole-pilot
```

Check progress/output from another shell:

```bash
tail -n 20 /root/pssn-work/logs/wormhole-pilot-final-layer.log
wc -l "$PILOT_DIR/wormhole-pilot-final-layer-threshold-0.5-10q/question_results.jsonl"
cat "$PILOT_DIR/wormhole-pilot-final-layer-threshold-0.5-10q/run_summary.json"
```

## Outputs and measurements

The output directory contains:

- `selected_questions.jsonl`: the fixed 10-question sample and references.
- `question_results.jsonl`: one flushed result per completed question, including both answers, correctness, each sentence's skip score, and token/time/memory measurements.
- `run_summary.json`: aggregate accuracy and average cost after all 10 finish.

Token counts distinguish generated tokens from input tokens. Predictor-arm
counts additionally include tokens processed in the activation forward pass;
these are reported separately and in a combined estimate, not mislabeled as
generated tokens. GPU memory records PyTorch's peak allocated/reserved memory
(ROCm exposes this through PyTorch's `torch.cuda` API); CPU peak RSS is recorded
when Linux exposes it. GPU utilization is not measured. Memory peaks include
the loaded Qwen model, so compare peaks within the same server/runtime.

The predictor arm includes the ordinary draft generation, activation scoring,
and a second answer generation. Therefore it may use more time/tokens than
ordinary answering even if it filters sentences. The pilot measures whether the
answer remains correct and what the real pipeline costs; it does not presume a
speedup.

## Copy results to Drive

After `run_summary.json` says `"status": "completed"`, copy and verify the
whole result folder. `rclone copy` leaves the server copy in place:

```bash
export OUT_DIR=/root/pssn-work/PSSN2/artifacts/black_hole/sentence-training-5000-v2/wormhole-pilot-final-layer-threshold-0.5-10q
rclone copy "$OUT_DIR" \
  gdrive:PSSN2/artifacts/black_hole/sentence-training-5000-v2/wormhole-pilot-final-layer-threshold-0.5-10q \
  --progress --stats 10s --checksum
rclone check "$OUT_DIR" \
  gdrive:PSSN2/artifacts/black_hole/sentence-training-5000-v2/wormhole-pilot-final-layer-threshold-0.5-10q \
  --one-way
```
