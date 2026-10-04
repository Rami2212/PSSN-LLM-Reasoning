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
src/utils/environment.py  Runtime inspection and Qwen loading helpers
tests/                  Fast tests that do not download model weights
```
