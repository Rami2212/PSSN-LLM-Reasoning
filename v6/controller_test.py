"""Live paired comparison: original base, LayerSkip full depth, LayerSkip + controller."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import statistics
import time
from pathlib import Path

from .controller_common import (BASE_MODEL_ID, DEPTHS, MODEL_ID, MODES, Progress,
    atomic_json, checked_trial, digest, fingerprint, generate, load_controller,
    load_model, load_prepared, manifest, quality_preserved, route_depth, warmup)
from .gpu_energy import GpuEnergyMeter


def percent_reduction(baseline, candidate):
    return 100 * (1 - candidate / baseline) if baseline else None


def bootstrap_mean_ci(values, seed=42, samples=1000):
    import numpy as np
    rng = random.Random(seed)
    if not values:
        return None
    estimates = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples)]
    return [float(v) for v in np.percentile(estimates, [2.5, 97.5])]


def summarize(trials, total_articles, repeats, config):
    import numpy as np
    metrics = {}
    for mode in MODES:
        rows = [r for r in trials if r["mode"] == mode]
        # Repeated timings are not counted as additional independent quality examples.
        first = [r for r in rows if r["repeat"] == 0]
        timings = [r["seconds"] + r["routing_seconds"] for r in rows]
        elapsed = sum(timings)
        metrics[mode] = {
            "unique_articles": len(first), "timing_trials": len(rows),
            **{f"mean_{key}": statistics.mean(r[key] for r in first) if first else None
               for key in ("rouge1", "rouge2", "rougeL")},
            "complete_rate": statistics.mean(int(r["complete"]) for r in first) if first else None,
            "generation_seconds": sum(r["seconds"] for r in rows),
            "routing_seconds": sum(r["routing_seconds"] for r in rows),
            "total_seconds": elapsed,
            "mean_seconds": statistics.mean(timings) if timings else None,
            "p50_seconds": float(np.percentile(timings, 50)) if timings else None,
            "p95_seconds": float(np.percentile(timings, 95)) if timings else None,
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "input_tokens": sum(r["input_tokens"] for r in rows),
            "mean_output_tokens_first_repeat": statistics.mean(r["output_tokens"] for r in first) if first else None,
            "output_tokens_per_second": sum(r["output_tokens"] for r in rows) / elapsed if elapsed else None,
            "decoder_layer_token_evaluations": sum(r["decoder_layer_token_evaluations"] for r in rows),
            "mean_depth": statistics.mean(r["depth"] for r in first) if first else None,
            "depth_counts_first_repeat": {str(d): sum(r["depth"] == d for r in first) for d in DEPTHS},
            "max_peak_allocated_gib": max((r["peak_allocated_gib"] for r in rows), default=None),
            "max_peak_reserved_gib": max((r["peak_reserved_gib"] for r in rows), default=None),
            "gpu_energy_joules": sum(r["gpu_energy_joules"] for r in rows
                                      if r.get("gpu_energy_joules") is not None)
                                  if rows and all(r.get("gpu_energy_joules") is not None for r in rows) else None,
            "mean_gpu_energy_joules_per_generation": statistics.mean(
                r["gpu_energy_joules"] for r in rows if r.get("gpu_energy_joules") is not None)
                if rows and all(r.get("gpu_energy_joules") is not None for r in rows) else None,
            "mean_gpu_average_power_watts": statistics.mean(
                r["gpu_average_power_watts"] for r in rows if r.get("gpu_average_power_watts") is not None)
                if rows and all(r.get("gpu_average_power_watts") is not None for r in rows) else None}
    index = {(r["article_id"], r["repeat"], r["mode"]): r for r in trials}
    keys = sorted({(r["article_id"], r["repeat"]) for r in trials})
    comparisons = {}
    for baseline, candidate in (("base_full", "layerskip_full"),
                                ("layerskip_full", "layerskip_controller"),
                                ("base_full", "layerskip_controller")):
        pairs = [(index[(q, rep, baseline)], index[(q, rep, candidate)]) for q, rep in keys
                 if (q, rep, baseline) in index and (q, rep, candidate) in index]
        first = [(a, b) for a, b in pairs if a["repeat"] == 0]
        base_seconds = sum(a["seconds"] + a["routing_seconds"] for a, _ in pairs)
        new_seconds = sum(b["seconds"] + b["routing_seconds"] for _, b in pairs)
        by_article = {}
        for a, b in pairs:
            by_article.setdefault(a["article_id"], []).append(
                b["seconds"] + b["routing_seconds"] - a["seconds"] - a["routing_seconds"])
        comparisons[f"{candidate}_vs_{baseline}"] = {
            "paired_timing_trials": len(pairs), "paired_quality_articles": len(first),
            "time_reduction_percent": percent_reduction(base_seconds, new_seconds),
            "speedup_baseline_divided_by_candidate": base_seconds / new_seconds if new_seconds else None,
            "output_token_reduction_percent": percent_reduction(
                sum(a["output_tokens"] for a, _ in pairs), sum(b["output_tokens"] for _, b in pairs)),
            "estimated_layer_work_reduction_percent": percent_reduction(
                sum(a["decoder_layer_token_evaluations"] for a, _ in pairs),
                sum(b["decoder_layer_token_evaluations"] for _, b in pairs)),
            "gpu_energy_reduction_percent": percent_reduction(
                sum(a["gpu_energy_joules"] for a, _ in pairs),
                sum(b["gpu_energy_joules"] for _, b in pairs))
                if pairs and all(a.get("gpu_energy_joules") is not None
                                 and b.get("gpu_energy_joules") is not None for a, b in pairs) else None,
            **{f"{key}_delta_points": 100 * statistics.mean(b[key] - a[key] for a, b in first) if first else None
               for key in ("rouge1", "rouge2", "rougeL")},
            "rougeL_delta_points_bootstrap_95ci": bootstrap_mean_ci([100 * (b["rougeL"] - a["rougeL"]) for a, b in first]),
            "mean_seconds_delta_bootstrap_95ci": bootstrap_mean_ci([statistics.mean(v) for v in by_article.values()]),
            "quality_rule_pass_rate_against_this_baseline": statistics.mean(
                int(quality_preserved(b, a, config)) for a, b in first) if first else None,
            "exact_output_token_match_rate": statistics.mean(
                int(a["generated_token_ids"] == b["generated_token_ids"]) for a, b in first) if first else None}
    complete = len(trials) == total_articles * repeats * len(MODES)
    repeat_mismatches = 0
    for q, rep in keys:
        if rep == 0:
            continue
        for mode in MODES:
            if (q, 0, mode) in index and (q, rep, mode) in index:
                repeat_mismatches += index[(q, 0, mode)]["generated_token_ids"] != index[(q, rep, mode)]["generated_token_ids"]
    return {"status": "completed" if complete else "running", "requested_articles": total_articles,
            "timing_repeats": repeats, "completed_trials": len(trials),
            "expected_trials": total_articles * repeats * len(MODES), "metrics": metrics,
            "paired_comparisons": comparisons, "repeat_token_mismatches": repeat_mismatches,
            "gpu_energy_meter": "AMD SMI cumulative energy counter around the synchronized GPU inference window",
            "notes": ["ROUGE F1 scores are 0-1; differences are points on a 0-100 scale.",
                      "Every method receives identical prompt token IDs and output cap; reference is used only after generation.",
                      "Latency includes input transfer, generation, and controller routing; excludes tokenization, ROUGE, disk I/O, loading and warmups.",
                      "All output tokens including EOS/boundary tokens are counted; fewer tokens alone do not imply better efficiency.",
                      "Layer-token work is a proxy, not measured FLOPs; all 32 layers' weights stay resident.",
                      "GPU energy and average power are whole-device readings during each inference window; no idle-energy subtraction is applied, and another GPU workload can contaminate readings.",
                      "Confidence intervals resample articles, not timing repetitions. Small pilots are uncertain.",
                      "Systems run one loaded model at a time; model order reverses each repeat; LayerSkip mode order alternates by article.",
                      "ROUGE overlap and quality labels do not measure factual correctness. Review saved summaries manually."]}


def write_reports(directory, summary):
    atomic_json(directory / "summary.json", summary)
    if summary["status"] != "completed":
        return
    rows = [{"mode": mode, **values} for mode, values in summary["metrics"].items()]
    with (directory / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# V6 measured comparison", "", "ROUGE is F1 on a 0–1 scale. Latency includes routing. GPU energy is measured during each inference window.", "",
             "| System | ROUGE-L | Mean seconds | Mean output tokens | Mean depth | GPU energy (J) | Mean GPU power (W) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, m in summary["metrics"].items():
        energy = "n/a" if m["gpu_energy_joules"] is None else f"{m['gpu_energy_joules']:.2f}"
        power = "n/a" if m["mean_gpu_average_power_watts"] is None else f"{m['mean_gpu_average_power_watts']:.2f}"
        lines.append(f"| {mode} | {m['mean_rougeL']:.4f} | {m['mean_seconds']:.3f} | {m['mean_output_tokens_first_repeat']:.1f} | {m['mean_depth']:.1f} | {energy} | {power} |")
    lines.extend(["", "See `summary.json` for paired savings, confidence intervals, completion, memory and repeat consistency.", "",
                  "Output shortening must be assessed alongside ROUGE, completion and saved summary text."])
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test(prepared_dir, checkpoint, output_dir, repeats=3):
    import torch
    payload = load_controller(checkpoint)
    bundle, prepared = load_prepared(prepared_dir)
    if prepared["identity"] != payload["prepared_identity"] or bundle["config"] != payload["config"]:
        raise ValueError("Prepared input differs from the trained controller's protocol")
    rows, config = bundle["test_articles"], bundle["config"]
    if any(r["article_key"] in payload["excluded_article_keys"] for r in rows):
        raise ValueError("Live test overlaps controller preparation examples")
    output_dir = Path(output_dir)
    source = manifest(output_dir, {"prepared_identity": prepared["identity"],
                      "checkpoint_sha256": digest(checkpoint), "repeats": repeats,
                      "modes": list(MODES), "article_ids": [r["article_id"] for r in rows]}, same_environment=True)
    trials = []
    energy_meter = GpuEnergyMeter(device_index=0)
    with Progress(output_dir) as progress:
        for repeat in range(repeats):
            # Release the previous model before loading the other checkpoint.
            groups = ["base", "layerskip"] if repeat % 2 == 0 else ["layerskip", "base"]
            for group in groups:
                modes = ["base_full"] if group == "base" else ["layerskip_full", "layerskip_controller"]
                tokenizer = model = None
                for index, row in enumerate(rows):
                    order = modes if (index + repeat) % 2 == 0 else list(reversed(modes))
                    for mode in order:
                        folder = output_dir / f"repeat_{repeat}" / row["article_id"]
                        decision_path = folder / "decision.json"
                        if mode == "layerskip_controller":
                            if decision_path.exists():
                                from .controller_common import read_json
                                decision = read_json(decision_path)
                                if (decision["run_identity"] != source["identity"]
                                        or decision["article_id"] != row["article_id"]
                                        or decision["repeat"] != repeat):
                                    raise ValueError("Saved decision identity mismatch")
                            else:
                                # Warm CPU prediction so first-call imports do not bias routing time.
                                route_depth(payload["pipeline"], "Warmup article", payload["threshold"])
                                start = time.perf_counter()
                                depth, scores = route_depth(payload["pipeline"], row["visible_article"], payload["threshold"])
                                seconds = time.perf_counter() - start
                                decision = {"run_identity": source["identity"], "article_id": row["article_id"],
                                            "repeat": repeat, "depth": depth, "scores": scores,
                                            "routing_seconds": seconds, "threshold": payload["threshold"]}
                                atomic_json(decision_path, decision)
                            depth, routing_seconds = decision["depth"], decision["routing_seconds"]
                        else:
                            depth, routing_seconds = 32, 0.
                        path = folder / f"{mode}.json"
                        progress.update(phase="generating", repeat=repeat, article=index + 1, mode=mode,
                                        completed_trials=len(trials), total_trials=len(rows) * repeats * 3)
                        if not path.exists():
                            if model is None:
                                progress.update(phase="loading_model", model_group=group)
                                model_id = BASE_MODEL_ID if group == "base" else MODEL_ID
                                revision = config["base_model_revision"] if group == "base" else config["model_revision"]
                                tokenizer, model = load_model(model_id, revision, config["model_revision"], config["seed"])
                                warmup(tokenizer, model, rows[0], (32,) if group == "base" else DEPTHS)
                            trial = generate(tokenizer, model, row, depth, config["max_new_tokens"],
                                             energy_meter=energy_meter)
                            atomic_json(path, {**trial, "run_identity": source["identity"], "repeat": repeat,
                                               "mode": mode, "routing_seconds": routing_seconds})
                        trial = checked_trial(path, row, source["identity"], depth, mode)
                        if trial["repeat"] != repeat:
                            raise ValueError("Saved timing repeat differs")
                        trials.append(trial)
                        progress.update(completed_trials=len(trials))
                        # Lightweight progress each generation; full bootstrap reports once all finish.
                        atomic_json(output_dir / "progress.json", {"completed_trials": len(trials),
                                    "expected_trials": len(rows) * repeats * 3})
                        print(f"repeat={repeat} {row['article_id']} {mode} depth={depth} ROUGE-L={trial['rougeL']:.3f} seconds={trial['seconds']:.2f}", flush=True)
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()
        progress.update(phase="reporting")
        report = summarize(trials, len(rows), repeats, config)
        report["gpu_energy_measurement"] = {
            "available": energy_meter.available,
            "device_index": 0,
            "unavailable_reason": energy_meter.reason,
            "scope": "per-generation synchronized inference window",
            "counter_units": "joules",
            "power_units": "watts",
        }
        write_reports(output_dir, report)
    energy_meter.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print(json.dumps(test(args.prepared_dir, args.controller, args.output_dir, args.repeats), indent=2))


if __name__ == "__main__":
    main()
