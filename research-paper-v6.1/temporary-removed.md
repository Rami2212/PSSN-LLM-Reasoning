# Temporarily Removed: Controller Pass Rule

This file preserves the controller-label and quality-pass criteria temporarily removed from the active v6.1 documents.

## Controller-training label rule

For article `i` and candidate depth `d`, let `R(i,d)` be reference ROUGE-L F1. Assign a positive label when the summary is complete and nonempty, `R(i,d) >= 0.15`, and `R(i,d) >= R(i,32) - 0.02`. Otherwise assign a negative label.

The `0.15` floor and `0.02` allowance were project-defined criteria, not values prescribed by the ROUGE reference.

## Live quality-pass rule (previously used)

A live output passed if it was complete and nonempty, had ROUGE-L at least `0.15`, and was no more than `0.02` below the original Llama 3 full-depth system's ROUGE-L on the same article. The training-label rule instead used the early-exit checkpoint's depth-32 output as its reference.

## Validation selection criterion

The controller-setting search also used a validation mean ROUGE-L drop budget of `0.02`, then minimized mean selected depth; ties preferred higher mean ROUGE-L, smaller `C`, and higher routing threshold.
