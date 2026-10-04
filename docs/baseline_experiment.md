# Initial Qwen3-4B GSM8K baseline experiment

This experiment is the Week 1 control run for later PSSN comparisons. Execute
`notebooks/04_baseline_experiment.ipynb` in a fresh Google Colab GPU runtime.

## Fixed configuration

- Model: `Qwen/Qwen3-4B`, without quantization
- Dataset: `openai/gsm8k`, `main` configuration
- Evaluation source: deterministic 75-example subset of the official training split
- Random seed: `42`
- Prompt: the fixed template in `src/baseline/cot.py`
- Thinking mode: enabled
- Decoding: greedy (`do_sample=False`)
- Maximum new tokens: `1024`
- Precision: BF16 when supported, otherwise FP16

The official GSM8K test split remains untouched. The training-derived development
subset is used because Week 1 validates the experiment pipeline rather than
reporting a final test-set result.

## Outputs

The notebook writes:

- `results/baseline/baseline_results.jsonl`: complete reasoning traces,
  extracted answers, correctness, token counts, latency, GPU peaks, model/device
  metadata, generation settings, timestamp, and run ID for every problem.
- `results/baseline/baseline_summary.json`: aggregate accuracy, token and latency
  statistics, maximum GPU memory, malformed-output count, and full run settings.

The notebook also displays malformed outputs and a sample of incorrect responses
for manual inspection. A run is considered valid only when all 75 configured
examples complete and both artifact files are created. Actual metric values are
produced by the GPU run and must not be filled in manually.

