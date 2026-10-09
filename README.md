# PSSN-LLM-Reasoning

Predictive Semantic State Navigation (PSSN) for improving LLM reasoning
efficiency using Qwen, Hugging Face, GSM8K, and Google Colab.

## Colab environment setup

The first baseline uses the unquantized `Qwen/Qwen3-4B` model. Open
[`notebooks/01_environment_setup.ipynb`](notebooks/01_environment_setup.ipynb)
in Google Colab, select a GPU runtime, and run every cell in order. The notebook
clones this repository when necessary, installs `requirements.txt`, reports the
CUDA/GPU configuration, selects BF16 when supported (otherwise FP16), loads the
tokenizer and model, and runs a deterministic validation prompt.

For a local setup:

```bash
python -m pip install -r requirements.txt
python -m pytest
```

A CUDA GPU with enough memory for an unquantized 4B-parameter model is required
for the validation inference. CPU execution is available only for environment
diagnostics and is not the supported baseline configuration.

## Project structure

```text
notebooks/              Colab experiment notebooks
src/data/gsm8k.py         GSM8K loading, normalization, and sampling
src/models/qwen.py        Reusable Qwen3 generation wrapper
src/baseline/cot.py       Fixed-prompt Chain-of-Thought baseline
src/evaluation/           Answer extraction and accuracy scoring
src/utils/                Environment helpers and experiment logging
tests/                  Fast tests that do not download model weights
```

## GSM8K data preparation

`src.data.gsm8k.load_gsm8k` downloads the official `openai/gsm8k` `main`
configuration, preserves its train and test splits, extracts reference answers,
and assigns stable split-based IDs. An optional development subset is sampled
only from training data with a fixed seed:

```python
from src.data.gsm8k import load_gsm8k

gsm8k = load_gsm8k(development_size=100, seed=42)
print(gsm8k["development"][0])
```

Run [`notebooks/02_gsm8k_dataset.ipynb`](notebooks/02_gsm8k_dataset.ipynb) in
Colab to download the dataset and validate its splits and normalized schema.

Notebooks 01 and 02 also create timestamped research snapshots. By default they
persist to `MyDrive/PSSN/artifacts` so the files survive Colab shutdown. The
environment snapshot contains GPU/CUDA details, package versions, `pip freeze`,
repository commit, model/tokenizer revisions, precision, and validation output.
The dataset snapshot contains every normalized split in Arrow format plus a JSON
manifest with fingerprints, split sizes, schema, development IDs, and sampling
configuration. Set `PERSIST_TO_GOOGLE_DRIVE=False` to use local runtime storage.

## Chain-of-Thought baseline

The baseline uses one fixed GSM8K prompt, Qwen3 thinking mode, and deterministic
greedy decoding by default. Each result preserves the full reasoning response,
input/output token counts, and its complete generation configuration:

```python
from src.baseline.cot import run_cot_baseline
from src.models.qwen import GenerationSettings, QwenGenerator

generator = QwenGenerator.from_pretrained(
    settings=GenerationSettings(max_new_tokens=1024, seed=42)
)
records = run_cot_baseline(gsm8k["development"], generator, limit=5)
```

Run [`notebooks/03_cot_baseline.ipynb`](notebooks/03_cot_baseline.ipynb) in a
GPU-enabled Colab runtime for an end-to-end sample generation.

## Week 2 trace expansion and validation

Run [`notebooks/05_expand_baseline_traces.ipynb`](notebooks/05_expand_baseline_traces.ipynb)
after syncing the current repository to Colab. It runs a reproducible 100-question
training-derived sample, preserves each raw attempt, and writes a validation log,
an invalid/incomplete trace file, and a separate file of completed correct traces
for semantic-state segmentation. The summary reports whether the target of 50
valid completed traces was reached. Artifacts are saved under
`MyDrive/PSSN2/artifacts/expanded_baseline/`.

Before segmenting those traces, review the deterministic semantic-state
definition in [`docs/semantic_state_definition.md`](docs/semantic_state_definition.md).
The reusable, validated record schema is available as
`src.semantic_states.SemanticState`; the actual segmentation algorithm is a
separate Week 2 task.

## Answer extraction and accuracy

Evaluation is independent from model inference. It recognizes the baseline's
`Final answer:` label, GSM8K `####` markers, LaTeX boxed answers, and a final
numeric fallback. Equivalent commas, currency formatting, decimals, fractions,
scientific notation, and percentage presentation are normalized exactly before
comparison:

```python
from src.evaluation import evaluate_records, summarize_accuracy

evaluations = evaluate_records(records)
summary = summarize_accuracy(evaluations)
print(summary)
```

The T4 uses FP16; BF16 requires native NVIDIA compute capability 8.0 or newer.
The default generation limit is 1,024 tokens. Responses stopped by that limit
are stored with `finish_reason="length"` and evaluated with `is_correct=None`.
Accuracy excludes these incomplete outputs; summaries also report scored and
unscored counts and completion rate. Older records
that reach their recorded token cap are handled conservatively as incomplete.

## Efficiency metrics and experiment logging

Every generated baseline record includes input/output token counts, synchronized
end-to-end latency, peak allocated and reserved CUDA memory, generation settings,
and model/device metadata. Persist records as append-only JSONL with run identity
and a UTC timestamp:

```python
from src.utils import ExperimentLogger

logger = ExperimentLogger(
    "results/baseline/baseline_results.jsonl",
    experiment_name="qwen3-gsm8k-cot",
    run_metadata={"dataset": "openai/gsm8k", "seed": 42},
)
logger.log_many(records)
print(logger.run_id)
```

## Initial baseline experiment

Run [`notebooks/04_baseline_experiment.ipynb`](notebooks/04_baseline_experiment.ipynb)
in a GPU-enabled Colab runtime to execute the complete pipeline on a deterministic
75-problem GSM8K development subset. The runner streams evaluated records to
`results/baseline/baseline_results.jsonl` and writes aggregate metrics to
`results/baseline/baseline_summary.json`. The exact controlled configuration and
artifact schema are documented in
[`docs/baseline_experiment.md`](docs/baseline_experiment.md).
