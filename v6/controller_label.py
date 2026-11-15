"""Generate independent LayerSkip outcomes for 100 articles x depths 24/28/32."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .controller_common import (DEPTHS, MODEL_ID, Progress, atomic_json, checked_trial,
    digest, generate, load_model, load_prepared, manifest, quality_preserved, warmup)


def label(prepared_dir, output_dir):
    import torch  # Register GPU metadata before creating the manifest.
    bundle, prepared = load_prepared(prepared_dir)
    config, rows = bundle["config"], bundle["label_articles"]
    output_dir = Path(output_dir)
    source = manifest(output_dir, {"prepared_identity": prepared["identity"],
                      "prepared_dir": os.path.relpath(Path(prepared_dir).resolve(), output_dir.resolve()),
                      "article_count": len(rows), "depths": list(DEPTHS)}, same_environment=True)
    labeled, tokenizer, model = [], None, None
    with Progress(output_dir) as progress:
        for index, row in enumerate(rows):
            folder = output_dir / "articles" / row["article_id"]
            outcomes, hashes = {}, {}
            order = list(DEPTHS[index % 3:]) + list(DEPTHS[:index % 3])
            for depth in order:
                progress.update(phase="generating", article=index + 1, total_articles=len(rows), depth=depth)
                path = folder / f"depth_{depth}.json"
                if not path.exists():
                    if model is None:
                        progress.update(phase="loading_model")
                        tokenizer, model = load_model(MODEL_ID, config["model_revision"],
                                                       config["model_revision"], config["seed"])
                        warmup(tokenizer, model, rows[0], DEPTHS)
                    trial = generate(tokenizer, model, row, depth, config["max_new_tokens"])
                    atomic_json(path, {**trial, "run_identity": source["identity"]})
                outcomes[str(depth)] = checked_trial(path, row, source["identity"], depth)
                hashes[str(depth)] = digest(path)
            labeled.append({**row, "outcomes": outcomes, "trial_sha256": hashes,
                            "quality_labels": {str(d): quality_preserved(outcomes[str(d)], outcomes["32"], config)
                                               for d in DEPTHS}})
            atomic_json(output_dir / "selected_dataset.json", labeled)
            summary = {"status": "completed" if len(labeled) == len(rows) else "running",
                       "articles": len(labeled), "generation_trials": len(labeled) * 3,
                       "quality_preserved_counts": {str(d): sum(r["quality_labels"][str(d)] for r in labeled)
                                                    for d in DEPTHS},
                       "mean_rougeL": {str(d): sum(r["outcomes"][str(d)]["rougeL"] for r in labeled) / len(labeled)
                                       for d in DEPTHS},
                       "quality_rule": {k: config[k] for k in ("min_rouge_l", "max_rouge_l_drop")}}
            atomic_json(output_dir / "summary.json", summary)
            print(f"Labeled {len(labeled)}/{len(rows)} articles at {DEPTHS}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(label(args.prepared_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
