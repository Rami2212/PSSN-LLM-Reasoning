# Research Focus

## Research title

**PSSN: Predicting Adaptive Inference Depth for Large Language Model Text Generation: A Controller-Based Study**

**PSSN:** Predictive Stage Selection Navigator

## Research problem

Running every request through all model layers can spend computation on examples that may be adequately generated at a shallower depth. This investigates whether a lightweight controller can predict an acceptable depth from the input article and whether that routing produces a useful efficiency change while preserving answer quality.

## Research questions

**Main research question:** Can PSSN select an inference depth for an early-exit-capable model while preserving generated-text quality and improving inference efficiency compared with the original base model and the checkpoint running at full depth?

1. **RQ1:** Can a controller trained on article text predict a suitable inference depth (24, 28, or 32 layers) for an early-exit-capable Llama 3 8B model?
2. **RQ2:** Under the same prompts, inputs, and generation settings, how does full-depth inference with the early-exit-capable checkpoint compare with the original Llama 3 8B base model in ROUGE-1/2/L, completion rate, and efficiency?
3. **RQ3:** On the same 2500-article live test, how does controller-routed early-exit inference compare with both full-depth systems in generated-text quality and efficiency?

## Research objectives

### Main objective

To implement and evaluate a controller that selects an inference depth for text generation with an early-exit-capable Llama 3 8B model, comparing the original base model, the checkpoint at full depth, and controller-routed generation from the checkpoint on the same 2500-article live test.

### Specific objectives

1. Prepare article-level training, validation, and test partitions; generate candidate text at 24, 28, and 32 layers; and create labeled article-depth examples for controller training and evaluation.
2. Evaluate the controller's ability to predict acceptable inference depths using held-out articles and controller-label metrics.
3. Compare the original Llama 3 8B base model, the early-exit-capable checkpoint at full depth, and PSSN-routed generation on the same 2500-article test under identical settings, reporting generated-text quality and latency.

## Controller–model architecture

```mermaid
flowchart LR
    subgraph FIT[Offline controller fitting]
        A[Training articles and reference highlights] --> B[Early-exit-capable Llama 3 8B checkpoint<br/>frozen model weights]
        B --> C[Generate summaries at depths 24 28 and 32]
        C --> D[Create binary labels<br/>completion and ROUGE-L rule]
        D --> E[Fit word and character TF-IDF vocabularies]
        E --> F[Fit balanced logistic-regression controller<br/>trainable component]
    end

    subgraph RUN[Inference for a new article]
        G[Article text] --> H[Build candidate features<br/>article text plus depth token]
        H --> I[TF-IDF transforms and fitted controller]
        I --> J[Select fixed depth d<br/>24 28 or 32]
        G --> N[Build model prompt and input tokens]
        subgraph MODEL[Early-exit-capable Llama 3 8B · frozen weights]
            X[Internal depth selector<br/>receives controller choice]
            N --> O[Input embeddings]
            O --> P[Transformer blocks 1–24]
            P --> Q{Exit at 24?}
            Q -->|No| R[Transformer blocks 25–28]
            R --> S{Exit at 28?}
            S -->|No| T[Transformer blocks 29–32]
            T --> U[Exit at 32]
            Q -->|Yes| V[Shared LM head]
            S -->|Yes| V
            U --> V
            V --> M[Generate summary]
            X -. d = 24 .-> Q
            X -. d = 28 .-> S
            X -. d = 32 .-> U
        end
        J -. selected depth .-> X
    end

    F -. saved fitted pipeline .-> I
```

The controller sends a fixed depth choice to an internal cutoff in the model. The forward pass stops at that selected layer boundary, and the selected layer output continues through the shared language-model head. The controller is fitted for this study; the Llama weights remain frozen.


## Contributions

This study makes four contributions:

- applies the project's separate-controller approach to text summarization;
- documents a reproducible pipeline for creating article-depth quality labels, training and evaluating the controller, and testing on a separate article set;
- compares the original base model, the early-exit-capable checkpoint at full depth, and controller-routed generation from that checkpoint on the same 2500-article test; and
- reports quality and efficiency together, reporting latency and discussing the limits of article-level TF-IDF for predicting per-depth summary quality.
