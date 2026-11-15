"""Freeze model/data commits, three demonstrations, 100 label articles and 100 test articles."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .controller_common import (BASE_MODEL_ID, DATASET_CONFIG, DATASET_ID, MODEL_ID,
    Progress, atomic_json, fingerprint, load_prepared, manifest, text_key)


def sample_records(data, count, seed, split, excluded):
    indices = list(range(len(data)))
    random.Random(seed).shuffle(indices)
    rows = []
    for index in indices:
        item = data[index]
        key = text_key(item["article"])
        if key in excluded or not item["article"].strip() or not item["highlights"].strip():
            continue
        rows.append({"article_id": f"cnndm-{split}-{index:06d}", "source_index": index,
                     "source_id": item["id"], "article_key": key,
                     "article": item["article"], "reference_summary": item["highlights"]})
        excluded.add(key)
        if len(rows) == count:
            return rows
    raise ValueError(f"Not enough unique {split} articles")


def crop(tokenizer, text, cap):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[:cap], skip_special_tokens=True), len(ids), len(ids) > cap


def build_inputs(row, demos, tokenizer, config):
    visible, original_tokens, truncated = crop(tokenizer, row["article"], config["article_tokens"])
    sections = []
    for demo in demos:
        article, _, _ = crop(tokenizer, demo["article"], config["demo_article_tokens"])
        summary, _, _ = crop(tokenizer, demo["reference_summary"], config["demo_summary_tokens"])
        sections.append(f"Article: {article}\nSummary: {summary}\n\n")
    # Only demonstration references enter the prompt; the current reference never does.
    prompt = "".join(sections) + f"Article: {visible}\nSummary:"
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    if len(ids) + config["max_new_tokens"] > 8192:
        raise ValueError("Prompt + generation cap exceeds Llama 3's 8192-token context; reduce article/demo budgets")
    return {**row, "visible_article": visible, "original_article_tokens": original_tokens,
            "article_truncated": truncated, "prompt": prompt, "input_ids": ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--test-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--article-tokens", type=int, default=2048)
    parser.add_argument("--demo-article-tokens", type=int, default=1024)
    parser.add_argument("--demo-summary-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-rouge-l-drop", type=float, default=.02,
                        help="Allowed absolute ROUGE-L F1 drop (0.02 = two points on a 0-100 scale)")
    parser.add_argument("--min-rouge-l", type=float, default=.15)
    parser.add_argument("--revision", default="3b9e03eaf3cc90812a7bd2b69106b626d63bac1f")
    parser.add_argument("--base-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    args = parser.parse_args()
    if args.rows < 10 or min(args.test_rows, args.article_tokens, args.demo_article_tokens,
                             args.demo_summary_tokens, args.max_new_tokens) < 1:
        parser.error("Need >=10 labeling articles and positive counts/token budgets")
    if not (0 <= args.max_rouge_l_drop <= 1 and 0 <= args.min_rouge_l <= 1):
        parser.error("ROUGE settings must be in [0,1]")
    request = {key: value for key, value in vars(args).items() if key != "output_dir"}
    if (args.output_dir / "dataset.json").exists():
        bundle, old = load_prepared(args.output_dir)
        if old["config"]["request"] != request:
            raise ValueError("Preparation options changed; use a new output directory")
        print(json.dumps({"status": "reused", "identity": old["identity"], "config": bundle["config"]}, indent=2))
        return
    with Progress(args.output_dir) as progress:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer
        progress.update(phase="resolving_revisions")
        hub = HfApi()
        config = {**request, "model_revision": hub.model_info(MODEL_ID, revision=args.revision).sha,
                  "base_model_revision": hub.model_info(BASE_MODEL_ID, revision=args.base_revision).sha,
                  "dataset_commit": hub.dataset_info(DATASET_ID, revision=args.dataset_revision).sha,
                  "model_id": MODEL_ID, "base_model_id": BASE_MODEL_ID,
                  "dataset_id": DATASET_ID, "dataset_config": DATASET_CONFIG,
                  "prompt_format": "three-shot-article-summary-v1",
                  "rouge": "rouge-score 0.1.2; stemmed ROUGE-1/2/L F1; scores in [0,1]"}
        progress.update(phase="downloading_dataset")
        train = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", revision=config["dataset_commit"])
        test = load_dataset(DATASET_ID, DATASET_CONFIG, split="test", revision=config["dataset_commit"])
        chosen = sample_records(train, args.rows + 3, args.seed, "train", set())
        demos, labeled = chosen[:3], chosen[3:]
        # Exclude exact normalized duplicates of ANY original training article from live test.
        progress.update(phase="checking_train_test_duplicates")
        train_keys = {text_key(text) for text in train["article"]}
        unseen = sample_records(test, args.test_rows, args.seed + 1, "test", train_keys)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=config["model_revision"])
        progress.update(phase="freezing_prompts")
        bundle = {"config": config, "demonstrations": demos,
                  "label_articles": [build_inputs(r, demos, tokenizer, config) for r in labeled],
                  "test_articles": [build_inputs(r, demos, tokenizer, config) for r in unseen]}
        meta = manifest(args.output_dir, {"request": request, "resolved": config,
                        "dataset_sha256": fingerprint(bundle)})
        atomic_json(args.output_dir / "dataset.json", bundle)
        atomic_json(args.output_dir / "summary.json", {
            "identity": meta["identity"], "label_articles": len(labeled),
            "label_trials": len(labeled) * 3, "unseen_test_articles": len(unseen),
            "demonstrations": len(demos), "test_overlap_with_original_train": 0,
            "truncated_label_articles": sum(r["article_truncated"] for r in bundle["label_articles"]),
            "truncated_test_articles": sum(r["article_truncated"] for r in bundle["test_articles"]),
            "note": "All three systems receive the same frozen token IDs; current reference summaries stay outside inputs."})
        print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
