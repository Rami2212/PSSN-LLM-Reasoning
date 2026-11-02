# V5: released LayerSkip model and 100-question controller pilot

For the controller workflow, use [CONTROLLER_300_SERVER.md](CONTROLLER_300_SERVER.md).
It labels 100 questions at depths 24, 28, and 32 (300 generations total),
trains a question-and-depth answer-preservation predictor, evaluates it, and
tests it live on 100 unseen questions. It includes verified Drive backups after
each stage.

V5 downloads `facebook/layerskip-llama3-8B` and generates answers at depths
**24, 28 and 32**. There is no fine-tuning step: Meta already trained this
checkpoint for early exits. It is a 32-layer Llama model; the Qwen depths
28/32/36 do not apply to it.

Start with [SERVER_SETUP.md](SERVER_SETUP.md) for PC upload commands, ROCm 7.2
installation, Hugging Face access, tmux, execution, Drive backup and PC download.
Only the `v5/` source folder is required; no previous artifacts are needed.

After the smoke run, the optional `v5.compare_self_speculative` experiment
compares full-depth greedy output against LayerSkip's official self-speculative
decoding using early exits at layers 24 and 28. Unlike fixed-prefix generation,
the early layers draft tokens and the remaining layers verify/correct them.
It uses the exact same five questions and demonstrations and saves separately
under `selfspec-comparison/`. See the corresponding run and backup steps in
`SERVER_SETUP.md`.

## What runs

1. Resolve model and GSM8K revisions to immutable commits and record them.
2. Select five GSM8K **train** questions using seed 42, matching the V4 default
   selection: 01458, 03620, 06312, 04833 and 05032.
3. Build a three-example prompt using the first three GSM8K training examples
   whose question texts are different from every evaluated question. This is a
   base/continually pretrained checkpoint, so V5 uses a plain few-shot prompt
   rather than an instruction-chat template. The current question's answer and
   worked solution are never supplied to the model.
4. Warm up each depth, then run one greedy generation per question/depth:
   **15 measured generations**, each with a 1,024-new-token cap. Trial order
   rotates to reduce a simple fixed-order timing bias.
5. Stop on EOS or the start of another `Question:` block. Parse the last numeric
   `FINAL:` or `####` line and compare against the dataset answer using Decimal.
   Missing answers and capped/incomplete responses count as incorrect.
6. Save each trial immediately. The same command resumes saved trials when
   source, settings, model revision, package versions and GPU match.

The actual intervention temporarily runs the first N **whole transformer
blocks**, including their attention and MLP, and applies the checkpoint's
original final normalization and output head. The selected depth is fixed for
the entire prompt and every generated token. A new KV cache is created for
each generation. No speculative verification or question controller is used.

## Outputs

All default server results go to
`/root/pssn-work/PSSN2/v5/layerskip-smoke-5q`; backup commands target
`gdrive:PSSN2/v5/layerskip-smoke-5q`.

| File | Contents |
|---|---|
| `manifest.json` | Checkpoint/tokenizer/dataset commits, prompt and stopping settings, code hash, packages, GPU/backend and repository commit |
| `questions.json` | Five actual questions and reference solutions/answers |
| `demonstrations.json` | Exact three prompt examples and their original GSM8K answers |
| `questions/<id>/prompt.json` | Exact input text, identical across the three depths |
| `questions/<id>/depth_24.json` (and 28/32) | Raw and cleaned output, token IDs, answer, completion, correctness, tokens, synchronized latency and peak memory |
| `summary.json` | Accuracy/completion, tokens/time/memory by depth; paired failures and timing ratios against depth 32 |
| `run_status.json` | PID, phase, progress and a 15-second heartbeat; final completed/interrupted/failed status |

Timing includes tokenization and prefix setup plus synchronized generation;
downloads, model loading, warmup and result saving are excluded. Input tokens
include demonstrations. The 1,024-token cap covers newly generated tokens only.
Reported tokens/second are aggregate output tokens divided by measured seconds.
Memory is PyTorch allocated/reserved GPU memory, not sampled GPU utilization.
All model weights remain resident at every depth, and reserved memory can
reflect the allocator's earlier trials. Smaller depth does not guarantee
shorter answers, lower peak memory, better accuracy or lower wall-clock time.

Five development questions cannot establish an accuracy or speedup claim.
Full-depth LayerSkip Llama performance is the baseline for this test. Comparing
these numbers directly with V4 Qwen would confound model, training and prompt.
There is no guarantee the released model's early exits solve GSM8K reliably.

## Research used

| Source | What its authors did | What V5 uses |
|---|---|---|
| [LayerSkip, ACL 2024](https://aclanthology.org/2024.acl-long.681/) and [official code](https://github.com/facebookresearch/LayerSkip) | Trained with deeper-layer dropout and shared-head early-exit losses; evaluated early exits and self-speculative decoding with full-model verification. | Their already adapted checkpoint and the shared final-head interpretation of prefix outputs. V5 measures direct exits and does not reproduce their training or verification benchmark. |
| [Official LayerSkip Llama3-8B model card](https://huggingface.co/facebook/layerskip-llama3-8B) | Publishes continually pretrained Llama3 weights and usage examples. | Actual weights/tokenizer, resolved revision and required gated access. Follow the model's access/license conditions. |
| [LITE, Findings NAACL 2024](https://aclanthology.org/2024.findings-naacl.232/) | Instruction-tuned intermediate outputs with extra losses and studied fixed and dynamic exits. | Background for the earlier adaptation proposal. No LITE code, weights or tuning is claimed in this test. |
| [GSM8K](https://github.com/openai/grade-school-math) | Publishes math questions with worked solutions and reference final answers. | Questions, disjoint prompt demonstrations and numeric reference scoring. |
| [AMD ROCm 7.2 PyTorch installation](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/3rd-party/pytorch-install.html) | Provides ROCm-supported framework installation options. | Python 3.12 and AMD's ROCm 7.2 PyTorch wheels in the server instructions. |
| [Transformers 4.57.1 self-speculative decoding](https://huggingface.co/docs/transformers/v4.57.1/generation_strategies#self-speculative-decoding) | Uses early exits to draft candidate tokens, then verifies/corrects them with the remaining layers; it is intended to preserve the full model's output distribution while sharing weights/cache. | V5 comparison mode using `assistant_early_exit`; V5 does not claim training the checkpoint or guaranteed latency gains. |

The prompt and smoke-test protocol are project choices, not a claim to exactly
reproduce the authors' GSM8K evaluation. The released model is not evidence
that question-based routing is novel; that would require its own experiment
and related-work comparison.
