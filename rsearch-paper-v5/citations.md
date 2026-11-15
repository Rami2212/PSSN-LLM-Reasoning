# V5 citations, source use, and research gaps

**Project:** Predictive Semantic-State Navigation (PSSN), V5  
**Purpose:** Source ledger for the V5 paper and experiments. It distinguishes work directly used in V5 from related work, software documentation, and sources inspected for context.  
**Scope note:** The V5 implementation in `v5/` is a pilot of question-level routing over fixed LayerSkip exits. It does not train LayerSkip, and its fixed-depth controller is separate from the optional self-speculative decoding comparison.

## 1. Primary method and model sources

### LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding

- **Citation:** Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, Ahmed Aly, Beidi Chen, and Carole-Jean Wu. 2024. “LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding.” *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 12622–12642. [ACL Anthology](https://aclanthology.org/2024.acl-long.681/) · [arXiv:2404.16710](https://arxiv.org/abs/2404.16710)
- **Official code:** [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)
- **How V5 uses it:** The released `facebook/layerskip-llama3-8B` checkpoint was trained/adapted by the LayerSkip authors. V5 runs direct fixed-prefix exits at layers 24, 28, and 32 using the checkpoint's shared final normalization and output head. The optional smoke comparison uses the authors' self-speculative approach, where early layers draft tokens and later layers verify or correct them.
- **What V5 does not reproduce:** V5 does not reproduce LayerSkip's training losses, deeper-layer dropout, full model benchmark suite, or authors' experimental claims. The question controller is PSSN's experiment.

### LayerSkip Llama 3 8B model card

- **Source:** [facebook/layerskip-llama3-8B on Hugging Face](https://huggingface.co/facebook/layerskip-llama3-8B)
- **How V5 uses it:** Supplies the actual gated model and tokenizer. V5 records the resolved model revision (`3b9e03eaf3cc90812a7bd2b69106b626d63bac1f` in the pilot report), and uses the released 32-layer model without additional fine-tuning. The model card and its access/license terms should be cited alongside the LayerSkip paper when describing the checkpoint.

### Llama 3: The Llama 3 Herd of Models

- **Citation:** Grattafiori et al. 2024. “The Llama 3 Herd of Models.” [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- **How V5 uses it:** Background/model-family provenance for the Llama 3 base architecture underlying the released LayerSkip checkpoint. It is not the source of V5's early-exit method or controller.

## 2. Related early-exit work

### LITE: Investigating Acceleration of LLaMA Inference by Enabling Intermediate Layer Decoding via Instruction Tuning

- **Citation:** Neeraj Varshney, Agneet Chatterjee, Mihir Parmar, and Chitta Baral. 2024. “Investigating Acceleration of LLaMA Inference by Enabling Intermediate Layer Decoding via Instruction Tuning with ‘LITE’.” *Findings of the Association for Computational Linguistics: NAACL 2024*, 3656–3677. [ACL Anthology](https://aclanthology.org/2024.findings-naacl.232/)
- **How V5 uses it:** Background for earlier PSSN adaptation planning and a related approach to intermediate-layer outputs and fixed/dynamic exits. V5 does not use LITE weights, code, or training procedure. It belongs in related work only unless the final manuscript makes a specific comparison.

### CALM: Confident Adaptive Language Modeling

- **Citation:** Schuster et al. 2022. “Confident Adaptive Language Modeling.” [arXiv:2207.07061](https://arxiv.org/abs/2207.07061)
- **How V5 uses it:** Related-work context for token-level adaptive early exiting. V5 instead chooses one depth from question text before generation and holds that depth for the prompt and all generated tokens.

### Dynamic Layer Selection in Decoder-Only Transformers

- **Citation:** Theodore Glavas, Joud Chataoui, Florence Regol, Wassim Jabbour, Antonios Valkanas, Boris N. Oreshkin, and Mark Coates. 2024. “Dynamic Layer Selection in Decoder-Only Transformers.” [arXiv:2410.20022](https://arxiv.org/abs/2410.20022)
- **How V5 uses it:** Related-work context for sequence-level compute selection and oracle/selection limitations, referenced in the V4-to-V5 project materials. It is not implemented in V5. Verify the final bibliographic metadata and the precise claims to cite before submission.

## 3. Dataset, evaluation, and model/data interfaces

### GSM8K: Training Verifiers to Solve Math Word Problems

- **Citation:** Cobbe et al. 2021. “Training Verifiers to Solve Math Word Problems.” [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) · [official dataset repository](https://github.com/openai/grade-school-math) · [Hugging Face dataset card](https://huggingface.co/datasets/openai/gsm8k)
- **How V5 uses it:** Supplies the math questions, worked examples used as few-shot demonstrations, and reference final answers. The pilot labels use train questions; the live comparison uses held-out test questions. Numeric final answers are parsed and compared with the dataset references. Cite the paper for the benchmark and the dataset card/repository for the exact loaded source and split.
- **Scope limitation:** GSM8K is grade-school math. It cannot by itself establish performance on harder mathematics, other reasoning tasks, or non-math domains.

### Transformers library and V5 versioned documentation

- **Citation:** Wolf et al. 2020. “Transformers: State-of-the-Art Natural Language Processing.” *EMNLP 2020 System Demonstrations*, 38–45. [ACL Anthology](https://aclanthology.org/2020.emnlp-demos.6/)
- **Documentation used:** [Transformers 4.57.1 self-speculative decoding guide](https://huggingface.co/docs/transformers/v4.57.1/generation_strategies#self-speculative-decoding) and [versioned model/generation documentation](https://huggingface.co/docs/transformers/v4.57.1/)
- **How V5 uses it:** Loads the model and tokenizer, performs generation, and exposes the `assistant_early_exit` path used by the optional self-speculative comparison. The run pins Transformers 4.57.1. The direct fixed-depth experiment is V5 code and should not be represented as a result guaranteed by the library documentation.

### Hugging Face Datasets

- **Citation:** Lhoest et al. 2021. “Datasets: A Community Library for Natural Language Processing.” *EMNLP 2021 System Demonstrations*, 175–184. [ACL Anthology](https://aclanthology.org/2021.emnlp-demo.21/)
- **How V5 uses it:** Retrieves GSM8K and provides dataset records and split access. The run pins `datasets==3.6.0` and records dataset revision/fingerprint information where available.

## 4. Software and infrastructure sources

### Scikit-learn

- **Citation:** Pedregosa et al. 2011. “Scikit-learn: Machine Learning in Python.” *Journal of Machine Learning Research* 12, 2825–2830. [JMLR article](https://www.jmlr.org/papers/v12/pedregosa11a.html)
- **How V5 uses it:** Implements the CPU controller with word/character TF-IDF features and logistic regression; validation selects regularization and the routing threshold policy uses predicted answer-preservation probabilities. Cite the paper when reporting the classifier implementation, not as evidence that this particular router is reliable.

### PyTorch

- **Citation:** Paszke et al. 2019. “PyTorch: An Imperative Style, High-Performance Deep Learning Library.” *NeurIPS 2019*. [NeurIPS paper](https://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library)
- **How V5 uses it:** Tensor operations and inference for the LayerSkip checkpoint, plus device synchronization and allocated/reserved GPU memory measurements.

### AMD ROCm 7.2 and PyTorch installation documentation

- **Technical documentation:** [ROCm 7.2 PyTorch installation](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/3rd-party/pytorch-install.html) · [ROCm documentation](https://rocm.docs.amd.com/)
- **How V5 uses it:** Server environment setup for the AMD Instinct MI300X run, using ROCm 7.2 PyTorch wheels. This is infrastructure documentation rather than a research paper. Report the actual recorded Python, PyTorch, ROCm, GPU, precision, and attention backend from each run manifest.

## 5. Source-to-claim map

| V5 manuscript claim or section | Sources | Attribution boundary |
|---|---|---|
| Early-exit and self-speculative decoding background | LayerSkip; Transformers guide | Describe the published method separately from V5's direct-depth comparison. |
| Checkpoint, architecture, access, and revision | LayerSkip model card; Llama 3 report | Identify the checkpoint as author-released and cite its exact revision in experiment details. |
| GSM8K task and scoring | Cobbe et al.; official repository/HF card | Scores are exact numeric final-answer outcomes, not verification of reasoning quality. |
| Fixed-depth, question-based routing | V5 methods and code; cite LayerSkip as model/method context | The router, labels, threshold rule, and evaluation are PSSN contributions/choices; do not imply they were proposed by LayerSkip. |
| TF-IDF/logistic regression controller | Pedregosa et al.; V5 code | Library attribution does not validate the chosen features or threshold. |
| Framework and runtime | Wolf et al.; Lhoest et al.; Paszke et al.; ROCm docs | Pair software citations with exact versions and saved run manifests. |
| Related adaptive early-exit methods | LITE; CALM; dynamic layer selection paper | Cite only for claims checked against the actual papers; V5 does not implement these methods. |

## 6. Research gaps and open questions

These are gaps in the evidence represented by the current V5 artifacts, not claims that no prior work has studied the topic. The related-work section should be updated with a systematic search before making novelty claims.

1. **Accuracy versus speed remains unresolved.** The checked-in `v5/results/unseen-test/unseen-test/summary.json` reports full-depth accuracy of 39/100 and controller accuracy of 33/100. The controller is faster in that run (187.49 s versus 212.31 s, including routing), but converts nine full-depth-correct answers to wrong and three wrong answers to correct. This is a measured tradeoff, not evidence that accuracy is maintained. The narrative `research/progress-report-v5.md` gives a conflicting 59/100 versus 55/100 table; reconcile it against the underlying per-question artifacts before quoting results. Report these pilot numbers only with their sample size, prompt, model revision, and one-run limitations.
2. **The evaluation set is too small for a reliable performance claim.** A 100-question test and one greedy generation per method provide noisy accuracy and timing estimates. Run larger, pre-registered evaluations with repeated seeds or decoding trials and confidence intervals; keep all items and any filtering rules explicit.
3. **Generalization is untested.** Current V5 evidence is one released Llama 3 8B LayerSkip checkpoint and GSM8K. Test other checkpoints/model families, task domains, difficulty bands, and prompt formats before claiming broad applicability.
4. **The predictor's safety calibration is unestablished.** The controller learns from a single observed outcome per question-depth pair, and the 0.5 probability threshold is a heuristic. Measure calibration, selective risk, false-safe rate, and risk/coverage curves; choose thresholds against a stated accuracy or risk constraint using validation data only.
5. **Training-data scale and split stability need study.** The earlier pilot used 100 labeled train questions (70/15/15 grouped by question). The 1,000-question × 3-depth protocol in `v5/CONTROLLER_1000X3_SERVER.md` is documented, but no corresponding completed run summary was found in the checked-in results tree. Paper claims should use completed run artifacts rather than infer outcomes from the protocol. Compare learning curves and repeated question-grouped splits.
6. **Comparison to alternative policies is incomplete.** Compare controller routing with fixed depths, a random policy matched for depth frequency, oracle routing (clearly labeled as an upper bound), confidence-based or token-adaptive exits, and the LayerSkip self-speculative path. Use the same model revision, prompts, questions, and measurement setup wherever methodologically valid.
7. **Compute and memory savings need stronger measurement.** Layer-token counts are estimates, not measured FLOPs or energy. All weights stay resident, and the observed peak allocated/reserved GPU memory is unchanged. Add hardware counters or energy measurement if making compute/energy claims; distinguish latency savings from memory savings.
8. **Latency protocol needs replication.** A single run is sensitive to warmup, ordering, GPU state, and generation length. Use repeated paired trials, report distribution/uncertainty, and separate prompt prefill, decoding, routing, and end-to-end wall time.
9. **The depth policy is coarse and static.** V5 selects 24, 28, or 32 layers once per question and uses that depth for every generated token. Evaluate finer-grained exits or per-token dynamic policies, while controlling quality and implementation overhead.
10. **Causal explanation of routing is absent.** The text classifier predicts outcomes from question wording but does not show which question features or model states signal that an early exit will work. Analyze calibration and feature stability and compare text-only routing with model-state-based predictors.

## 7. Reproducibility and citation checklist

- Cite the LayerSkip paper and official implementation when describing the checkpoint's early-exit design; cite the model card and record its immutable revision for V5 results.
- Cite the GSM8K paper and dataset source; report split, selection procedure, prompt demonstrations, answer parser, and incomplete/capped-answer handling.
- State clearly whether a result uses direct fixed-depth exit, question-controller routing, or self-speculative decoding. They are distinct conditions.
- Cite the software papers and record versions: Transformers 4.57.1, Datasets 3.6.0, scikit-learn 1.5.2, PyTorch/ROCm versions, and GPU.
- Keep the V5 pilot report's results distinct from results produced by the 1,000-question controller workflow. Use completed manifests and summaries for final paper figures.
- Do not claim maintained accuracy, guaranteed safety, reduced peak GPU memory, measured FLOP reduction, or generality from the current small pilot.
- Keep related-work sources (LITE, CALM, and dynamic layer selection) labeled as contextual unless a specific V5 method or result was actually compared with them.

