"""Train V5-style TF-IDF/logistic regression; select settings on validation articles only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .controller_common import (DEPTHS, PROTOCOL, Progress, atomic_json, digest, load_labels,
    manifest, model_text, partitions, route_depth, source_hashes)


def make_pipeline(c, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    return Pipeline([
        ("text", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                       token_pattern=r"(?u)\b\w+\b", max_features=20000)),
            ("characters", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                            sublinear_tf=True, max_features=20000)),
        ])),
        ("classifier", LogisticRegression(C=c, class_weight="balanced", max_iter=2000,
                                            random_state=seed, solver="lbfgs")),
    ])


def expand(rows):
    return ([model_text(r["visible_article"], d) for r in rows for d in DEPTHS],
            [int(r["quality_labels"][str(d)]) for r in rows for d in DEPTHS])


def route_report(rows, pipeline, threshold):
    selected = [route_depth(pipeline, r["visible_article"], threshold)[0] for r in rows]
    chosen = [r["outcomes"][str(d)] for r, d in zip(rows, selected)]
    baseline = [r["outcomes"]["32"] for r in rows]
    mean = lambda values: sum(values) / len(rows)
    return {"articles": len(rows), "mean_depth": mean(selected),
            "depth_counts": {str(d): selected.count(d) for d in DEPTHS},
            "controller_mean_rougeL": mean([r["rougeL"] for r in chosen]),
            "full_depth_mean_rougeL": mean([r["rougeL"] for r in baseline]),
            "mean_rougeL_drop": mean([a["rougeL"] - b["rougeL"] for a, b in zip(baseline, chosen)]),
            "quality_preserved_rate": mean([int(r["quality_labels"][str(d)]) for r, d in zip(rows, selected)]),
            "full_depth_quality_floor_pass_rate": mean([int(r["quality_labels"]["32"]) for r in rows]),
            "complete_rate": mean([int(r["complete"]) for r in chosen]),
            "quality_label_failures_after_early_exit": sum(d < 32 and not r["quality_labels"][str(d)]
                                                          for r, d in zip(rows, selected)),
            "threshold": threshold}


def train(labels_dir, output_dir, seed=42):
    import joblib
    rows, labels, bundle = load_labels(labels_dir)
    parts = partitions(rows, seed)
    x, y = expand(parts["train"])
    if len(set(y)) < 2:
        raise ValueError("Training labels have only one class. Inspect quality/finish reasons; collect more articles or predeclare a new quality rule in a NEW run. No usable controller was trained.")
    config = bundle["config"]
    output_dir = Path(output_dir)
    split_ids = {name: [r["article_id"] for r in items] for name, items in parts.items()}
    meta = manifest(output_dir, {"labels_identity": labels["identity"],
                    "labels_sha256": digest(Path(labels_dir) / "selected_dataset.json"),
                    "seed": seed, "split_article_ids": split_ids,
                    "c_grid": [.1, 1., 10.], "threshold_grid": [.5, .7, .9, 1.1],
                    "selection": "meet validation mean ROUGE-L drop budget; minimize mean depth; ties prefer higher ROUGE-L, smaller C, higher threshold"})
    history, best = [], None
    with Progress(output_dir) as progress:
        for c in (.1, 1., 10.):
            progress.update(phase="fitting", regularization_c=c)
            pipeline = make_pipeline(c, seed)
            pipeline.fit(x, y)
            for threshold in (.5, .7, .9, 1.1):
                report = {"C": c, **route_report(parts["validation"], pipeline, threshold)}
                report["within_mean_drop_budget"] = report["mean_rougeL_drop"] <= config["max_rouge_l_drop"]
                history.append(report)
                # 1.1 explicitly permits choosing the full-depth fallback if early exits degrade quality.
                ranking = (not report["within_mean_drop_budget"], report["mean_depth"],
                           -report["controller_mean_rougeL"], c, -threshold)
                if best is None or ranking < best[0]:
                    best = ranking, pipeline, report
        _, pipeline, selected = best
        payload = {"protocol": PROTOCOL, "pipeline": pipeline, "source_sha256": source_hashes(),
                   "threshold": selected["threshold"], "config": config,
                   "labels_identity": labels["identity"],
                   "labels_sha256": digest(Path(labels_dir) / "selected_dataset.json"),
                   "prepared_identity": labels["config"]["prepared_identity"],
                   "training_identity": meta["identity"], "split_article_ids": split_ids,
                   "excluded_article_keys": [r["article_key"] for r in rows] +
                                            [r["article_key"] for r in bundle["demonstrations"]]}
        tmp = output_dir / "controller.tmp.joblib"
        joblib.dump(payload, tmp, compress=3)
        os.replace(tmp, output_dir / "controller.joblib")
        atomic_json(output_dir / "splits.json", split_ids)
        atomic_json(output_dir / "validation_search.json", history)
        result = {"status": "completed", "label_articles": len(rows), "label_trials": len(rows) * 3,
                  "split_article_counts": {k: len(v) for k, v in parts.items()},
                  "actual_training_trials": len(y), "training_label_counts": {str(k): y.count(k) for k in (0, 1)},
                  "selected_validation_settings": selected,
                  "test_used_for_selection": False, "controller_sha256": digest(output_dir / "controller.joblib"),
                  "features": "visible article text plus depth token; word 1-2 and character 3-5 TF-IDF; balanced logistic regression",
                  "limitation": "100 articles / 300 labels is a pilot; only 210 trials fit the model. Predicted probabilities are not calibrated safety guarantees. No LLM weights are trained."}
        atomic_json(output_dir / "training_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.labels_dir, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
