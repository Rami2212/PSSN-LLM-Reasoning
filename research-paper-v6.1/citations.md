# Citations and source-use record

**Project:** Text summarization controller  
**Purpose:** Record the research papers, model and dataset sources, evaluation references, software, and hardware documentation used to design and run this study. The notes distinguish published methods from this study's own choices.

## 1. Research papers used directly

### LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding

Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, Ahmed Aly, Beidi Chen, and Carole-Jean Wu. 2024. “LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding.” *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 12622–12642. [ACL Anthology](https://aclanthology.org/2024.acl-long.681/) · [arXiv:2404.16710](https://arxiv.org/abs/2404.16710) · [authors' code](https://github.com/facebookresearch/LayerSkip)

**Use in this study:** Motivates early-exit inference and supplies the released `facebook/layerskip-llama3-8B` checkpoint. The 2500-article test compares the original Llama 3 base model, this checkpoint at full depth, and controller-routed generation using this checkpoint at depths 24, 28, or 32.

**Attribution boundary:** This study does not train or fine-tune the released checkpoint's weights and does not reproduce its layer-dropout training or verified self-speculative decoding. The controller and routing policy are this study's code. Cite the paper for the checkpoint's research method, but do not attribute this study's controller or results to Elhoushi et al.

### Teaching Machines to Read and Comprehend

Karl Moritz Hermann, Tomáš Kočiský, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. 2015. “Teaching Machines to Read and Comprehend.” *Advances in Neural Information Processing Systems 28*. [NeurIPS paper](https://proceedings.neurips.cc/paper/2015/hash/afdec7005cc9f14302cd0474fd0f3c96-Abstract.html) · [arXiv:1506.03340](https://arxiv.org/abs/1506.03340)

**Use in this study:** Original source for the CNN/DailyMail article and highlight corpus. This study uses the article text and reference highlights for controller labels and summarization evaluation.

**Attribution boundary:** The original paper introduced a reading-comprehension task using the corpus; cite See et al. (2017) as a direct reference for the CNN/DailyMail summarization task.

### Get To The Point: Summarization with Pointer-Generator Networks

Abigail See, Peter J. Liu, and Christopher D. Manning. 2017. “Get To The Point: Summarization with Pointer-Generator Networks.” *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 1073–1083. [ACL Anthology](https://aclanthology.org/P17-1099/) · [arXiv:1704.04368](https://arxiv.org/abs/1704.04368)

**Use in this study:** Summarization-task background for CNN/DailyMail and an established example of reporting ROUGE against reference summaries. This study does not implement pointer-generator networks.

### ROUGE: A Package for Automatic Evaluation of Summaries

Chin-Yew Lin. 2004. “ROUGE: A Package for Automatic Evaluation of Summaries.” *Text Summarization Branches Out*, 74–81. [ACL Anthology](https://aclanthology.org/W04-1013/)

**Use in this study:** Defines the ROUGE family used to compare generated summaries with CNN/DailyMail reference highlights. This study reports stemmed ROUGE-1, ROUGE-2, and ROUGE-L F1.

**Limit:** ROUGE measures reference-text overlap; it does not establish factual correctness or entailment.

### The Llama 3 Herd of Models

Aaron Grattafiori et al. 2024. “The Llama 3 Herd of Models.” [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) · [Meta research page](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)

**Use in this study:** Model-family background for the original `meta-llama/Meta-Llama-3-8B` full-depth reference checkpoint. The exact model and immutable revision used in the 2500-article run are recorded in its prepared manifest.

## 2. Related early-exit work considered for design context

The following papers concern adaptive computation or intermediate-layer generation. They are relevant to this study's motivation and related-work discussion; this study does **not** reproduce their algorithms or training.

### Confident Adaptive Language Modeling (CALM)

Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Q. Tran, Yi Tay, and Donald Metzler. 2022. “Confident Adaptive Language Modeling.” [arXiv:2207.07061](https://arxiv.org/abs/2207.07061)

**Relevance:** Background on confidence-based allocation of computation and token-level early exit. This study routes by article text before generation and keeps the selected depth fixed for that generation.

### Investigating Acceleration of LLaMA Inference by Enabling Intermediate Layer Decoding via Instruction Tuning with ‘LITE’

Neeraj Varshney, Agneet Chatterjee, Mihir Parmar, and Chitta Baral. 2024. *Findings of the Association for Computational Linguistics: NAACL 2024*, 3656–3677. [ACL Anthology](https://aclanthology.org/2024.findings-naacl.232/)

**Relevance:** Related work on training intermediate layers for generation and confidence-based early exit. This study does not instruction-tune the LLM or use LITE's token-level exit policy.

### Dynamic Layer Selection in Decoder-Only Transformers

Theodore Glavas, Joud Chataoui, Florence Regol, Wassim Jabbour, Antonios Valkanas, Boris N. Oreshkin, and Mark Coates. 2024. “Dynamic Layer Selection in Decoder-Only Transformers.” [arXiv:2410.20022](https://arxiv.org/abs/2410.20022)

**Relevance:** Related work on sample-dependent layer allocation and controller-based selection. This study's logistic-regression controller is its own implementation; it does not reproduce the paper's oracle or layer-selection methods.

## Research gap

Prior work establishes several approaches to adaptive computation, including token-level confidence-based early exit (Schuster et al., 2022), intermediate-layer tuning and exit policies (Varshney et al., 2024), and dynamic layer selection (Glavas et al., 2024). The sources reviewed here do not directly evaluate the specific setup studied in this project: a separate controller uses article text to choose a fixed inference depth before summarization, with its labels derived from completion and ROUGE-L criteria. This study evaluates the original full-depth Llama 3 8B base model, the early-exit-capable checkpoint at full depth, and controller-routed generation from that checkpoint on the same 2500-article test under shared inputs and generation settings. The gap is limited to this applied setup and does not imply that no prior work has explored controller-based depth selection.

## 3. Dataset, model, and code sources

### CNN/DailyMail dataset, version 3.0.0

- [Hugging Face dataset card: `abisee/cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail)
- [Dataset source repository](https://github.com/abisee/cnn-dailymail)
- **Use:** Supplies article text and reference highlights. The 2500-article run used 2500 labeling articles from the training split and 2500 separate articles from the test split. The recorded dataset revision is `96df5e686bee6baa90b8bee7c28b81fa3fa6223d`.
- **Boundary:** The model weights remain frozen; CNN/DailyMail is used for controller labels and summarization evaluation, not to fine-tune either LLM.

### Released LayerSkip checkpoint

- [Model card: `facebook/layerskip-llama3-8B`](https://huggingface.co/facebook/layerskip-llama3-8B)
- **Use:** Early-exit-capable model and tokenizer for both full-depth checkpoint inference and controller-routed generation in the 2500-article run. The run records revision `3b9e03eaf3cc90812a7bd2b69106b626d63bac1f`.
- **Boundary:** This study does not fine-tune this checkpoint.

### Original Llama 3 8B checkpoint

- [Model card: `meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- **Use:** Original-model full-depth baseline. The 2500-article run records revision `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920`.

### Controller method (internal project source)

- **Use:** This study continues the project's separate-controller approach: word/character TF-IDF features plus balanced logistic regression predict labels for candidate depths. It applies the approach to summarization using CNN/DailyMail outputs.
- **Boundary:** This is an internal prior implementation, not an external research paper. The exact prompt, label construction, and routing policy are project-specific.

## 4. Software and evaluation implementation sources

### Transformers

Thomas Wolf et al. 2020. “Transformers: State-of-the-Art Natural Language Processing.” *Proceedings of EMNLP 2020: System Demonstrations*, 38–45. [ACL Anthology](https://aclanthology.org/2020.emnlp-demos.6/) · [Transformers 4.57.1 generation documentation](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation)

**Use:** Model/tokenizer loading and autoregressive generation. This study pins Transformers 4.57.1. Cite the paper for software attribution and the versioned documentation for API behavior.

### Datasets

Quentin Lhoest et al. 2021. “Datasets: A Community Library for Natural Language Processing.” *Proceedings of EMNLP 2021: System Demonstrations*, 175–184. [ACL Anthology](https://aclanthology.org/2021.emnlp-demo.21/)

**Use:** Hugging Face Datasets library for loading CNN/DailyMail and its dataset splits. This study pins `datasets==3.6.0` and records the resolved dataset commit.

### Scikit-learn

Fabian Pedregosa et al. 2011. “Scikit-learn: Machine Learning in Python.” *Journal of Machine Learning Research* 12, 2825–2830. [JMLR article](https://www.jmlr.org/papers/v12/pedregosa11a.html)

**Use:** TF-IDF feature extraction, logistic-regression classifier, and pipeline components for the CPU controller. This citation attributes the library; it does not validate the controller's accuracy.

### PyTorch

Adam Paszke et al. 2019. “PyTorch: An Imperative Style, High-Performance Deep Learning Library.” *Advances in Neural Information Processing Systems 32*. [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html)

**Use:** Tensor operations, GPU inference, and synchronized timing.

### ROUGE implementation

- [Google Research `rouge-score` source](https://github.com/google-research/google-research/tree/master/rouge)
- **Use:** This study pins `rouge-score==0.1.2` to calculate stemmed ROUGE-1/2/L F1. Cite Lin (2004) for the metric and the implementation repository/package version for the code used.

## 5. Hardware and reproducibility documentation

- [AMD ROCm 7.2 PyTorch installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/3rd-party/pytorch-install.html): server installation guidance; run metadata records the actual ROCm/HIP and PyTorch versions.
- [AMD SMI 7.2.3 CLI documentation](https://rocm.docs.amd.com/projects/amdsmi/en/docs-7.2.3/how-to/amdsmi-cli-tool.html) and [AMD SMI Python API reference](https://rocm.docs.amd.com/projects/amdsmi/en/docs-7.2.4/reference/amdsmi-py-api.html): GPU power telemetry used in the experiment.
- The experiment setup, implementation, protocol, and 2500-article run manifest are project artifacts; preserve their exact paths, revisions, source hashes, package versions, and GPU details in the reproducibility package.

These are technical sources, not research papers. Include exact environment versions and hardware in methods/reproducibility sections rather than treating documentation as research evidence.

## 6. Source-to-claim map

| Study claim or method | Cite | Keep clear |
|---|---|---|
| Early-exit model and checkpoint | Elhoushi et al. (2024); model card and code | Released model training is external; this study neither retrains it nor implements verified self-speculation. |
| Llama 3 base-model comparator | Grattafiori et al. (2024); Llama 3 model card | Record the exact checkpoint revision used in this run. |
| CNN/DailyMail data and summarization task | Hermann et al. (2015); See et al. (2017); dataset card/repository | Report the 3.0.0 config, split, sampling, and resolved commit. |
| Summary-quality score | Lin (2004); `rouge-score` implementation | ROUGE overlap is not factuality. The quality thresholds are project choices. |
| TF-IDF/logistic-regression controller | Pedregosa et al. (2011); internal controller code | The controller is this study's separate classifier, not the released model's training procedure. |
| Model/data software | Wolf et al. (2020); Lhoest et al. (2021); Paszke et al. (2019) | Pair these citations with pinned software versions. |
| GPU power | AMD ROCm and AMD SMI documentation; run manifest | Power is a whole-device reading during inference windows. |

