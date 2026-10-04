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
src/utils/environment.py  Runtime inspection and Qwen loading helpers
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
