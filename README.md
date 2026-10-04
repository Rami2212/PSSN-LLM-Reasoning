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
