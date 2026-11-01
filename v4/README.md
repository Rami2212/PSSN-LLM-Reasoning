# V4: question-text controller for 12 / 24 / 36 transformer layers

Run the complete pipeline using [SERVER_SETUP.md](SERVER_SETUP.md).

| Command | Input | Output | Hardware |
| --- | --- | --- | --- |
| `python -m v4.label` | GSM8K **train**, downloaded automatically | Three answers per question, depth targets, audit, summary | GPU |
| `python -m v4.train` | V4 labeling directory | `controller.joblib`, disjoint splits, validation search | CPU |
| `python -m v4.evaluate` | Labels and controller | Held-out prediction rows and fixed-depth comparisons | CPU |
| `python -m v4.benchmark` | Controller; fresh GSM8K **test** questions | Real baseline/controller answers, tokens, latency, memory | GPU |

## Experimental definition

The controller is a small, reproducible **text baseline**: word/character
TF-IDF features and balanced logistic regression. Its only input is the
question string. It does not receive answers, gold solutions, Qwen activations,
question IDs or target labels during prediction. Word patterns are not a
guarantee of understanding mathematical difficulty; the experiment measures
whether these features predict useful depth choices on new questions.

Qwen3-4B has 36 decoder transformer layers. A choice of 12 executes layers
1-12; 24 executes 1-24; 36 executes all layers. Each retained layer includes
attention and MLP. The same choice applies to prompt prefill and every decode
token. The final RMSNorm and LM head remain in use at every depth. This is
whole-transformer prefix early exit, rather than the V3 MLP-group bypass.

The default Qwen checkpoint is frozen and **has not been trained for early
exits**. V4 trains only the text controller. It does not implement LayerSkip
model adaptation. The labeling pilot must first establish whether usable
12/24-layer outputs exist. A trained controller cannot repair bad shallow
outputs.

## Data and labels

The pilot uses the **same 100 distinct questions** at all three depths: 300
greedy generations, plus excluded warmups. Thinking mode is disabled; the
prompt permits a short calculation and requests `FINAL: <number>`. All depths
use the same prompt and 1,024-new-token cap. Numeric equality is checked
against GSM8K's answer after `####`; no LLM judge is used. Markdown-bold final
lines are supported. Missing final-answer markers and token-limit outputs
cannot be correct, even if a correct number appears elsewhere.

For a question whose full-depth answer is complete and correct, the target
is the shallowest depth whose answer is also complete and correct. Other
questions have a null target and remain in the audit and downstream accuracy
evaluation. One greedy observation per depth is an experimental label, not a
proof of reliability across sampling settings or different prompts.

Every trial is atomically saved immediately. An interrupted label run resumes
using the same command and skips saved trials. A local heartbeat updates
`run_status.json` every 15 seconds. A killed machine may leave an old `running`
status; check heartbeat freshness and the process PID. Changing source code,
generation dependencies, GPU environment, questions or settings requires a
new generation output directory. No Drive uploads are performed inside a run.

## Training and selection

Distinct question texts are split 70% train, 15% validation and 15% test
**before** fitting the vocabulary. Only non-null targets in the training
partition fit the classifier. Fewer than ten usable training questions, or
fewer than two target classes, stops training with an explanatory error.

Regularization `C` and an uncertainty threshold are selected using validation
data only. If maximum class probability is below that threshold, the action
is depth 36. Selection minimizes mean depth subject to a default empirical
accuracy drop of at most 0.02 (two percentage points) compared to full depth.
Threshold 1.01 explicitly represents the always-36 fallback. If it wins, report
that outcome; do not call it a speedup. This validation rule is not a statistical
guarantee, and a 15-question validation set is very small. Increase data and
inspect learning curves after the pilot.

`v4.evaluate` uses saved outcomes for the untouched 15% test partition. It
reports actual selected-answer correctness, correct-to-wrong changes, depth
distribution, target-depth classification accuracy, fixed 12/24/36 baselines,
and a non-deployable hindsight oracle. It does not make a new latency claim.

`v4.benchmark` generates two answers per fresh official test question. It
excludes every question used for V4 development, alternates baseline/controller
execution order, warms all depths first, and includes CPU routing time in
controller latency. All outcomes, including incomplete answers, enter accuracy.
Checkpoint confidence thresholds remain fixed throughout test evaluation.

Memory metrics are Torch peak allocated/reserved GPU GiB. All model weights
remain resident, so memory need not shrink. Layer-token evaluation counts
are analytic compute proxies, not FLOPs or GPU utilization. Shorter depth can
still produce longer, incorrect or truncated answers; inspect actual token
counts and end-to-end latency. Repeated runs are needed to characterize timing
noise. Model loading, warmup and disk/Drive transfer time are excluded.

## Reproducibility artifacts

`manifest.json` records settings, model/dataset commit revisions, package and
hardware metadata, code hashes, and run identity. The exact chosen questions,
gold answers, generated text, labels, evidence hashes and excluded examples
are saved. Training keeps its split IDs, train class counts and validation
search. `controller.joblib` contains the fitted vocabulary, classifier,
threshold and generation settings. Load only a controller you trust, because
joblib uses Python pickle serialization. No credentials are stored.

For a larger dataset use a new run folder and `--rows 1000` (or another count)
when labeling, then rerun training/evaluation. Keep generation settings fixed
for a data-size comparison; changing them creates a different experiment.

## Sources

- [Qwen3-4B architecture configuration](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json): 36 layers and original model dimensions.
- [Pinned Qwen3 implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py): decoder prefix, final RMSNorm, LM head and cached generation boundaries.
- [LayerSkip](https://arxiv.org/abs/2404.16710): related early-exit adaptation; its model-training recipe is not implemented here.
- [Dynamic layer selection in decoder-only transformers](https://arxiv.org/abs/2410.20022): related sequence-level compute selection and oracle limitations.
- [CALM](https://arxiv.org/abs/2207.07061): related per-token early exit; V4 instead fixes depth for each question.
