"""Server entry point for Week 3 removal experiments and continuous Drive backup."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

from src.data.semantic_state_dataset import read_jsonl
from src.models.qwen import GenerationSettings, QwenGenerator
from src.utils.environment import collect_environment_info, load_qwen_model
from .removal_experiment import run_removal_experiments, save_json


class RcloneMirror:
    """Synchronously upload closed checkpoint files; restore them on another server."""

    def __init__(self, local_dir: Path, remote: str):
        if ":" not in remote or not remote.split(":", 1)[1].strip("/"):
            raise ValueError("drive-output must name a remote subfolder, such as gdrive:PSSN2/artifacts/black_hole/run1")
        self.local_dir = local_dir.resolve()
        self.remote = remote.rstrip("/")

    @staticmethod
    def command(*args: str):
        return subprocess.run(["rclone", *args], check=True, capture_output=True, text=True)

    def restore(self):
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.command("mkdir", self.remote)
        files = self.command("lsf", self.remote, "--files-only").stdout.splitlines()
        if "manifest.json" in files:
            cloud_manifest = json.loads(self.command("cat", self.remote + "/manifest.json").stdout)
            local_manifest = self.local_dir / "manifest.json"
            if local_manifest.exists() and json.loads(local_manifest.read_text(encoding="utf-8")) != cloud_manifest:
                raise ValueError("local and Drive manifests differ; choose matching run folders")
        self.command("copy", self.remote, str(self.local_dir), "--ignore-existing")

    def __call__(self, path: Path):
        relative = path.resolve().relative_to(self.local_dir).as_posix()
        self.command("copyto", str(path), f"{self.remote}/{relative}", "--retries", "5")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="W3001 eligible_traces.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path, help="local checkpoint directory, reused for resume")
    parser.add_argument("--drive-output", required=True, help="rclone remote folder for continuous persistence")
    parser.add_argument("--revision", help="optional model/tokenizer commit to use")
    parser.add_argument("--limit", type=int, help="maximum NEW experiments this invocation")
    args = parser.parse_args(argv)

    selected = read_jsonl(args.input)
    if not selected:
        raise ValueError("the selected trace file is empty")
    baselines = [row["baseline_result"] for row in selected]
    setting_names = {field.name for field in fields(GenerationSettings)}
    configs = [
        {key: value for key, value in baseline["experiment_metadata"]["generation_config"].items() if key in setting_names}
        for baseline in baselines
    ]
    if any(config != configs[0] for config in configs):
        raise ValueError("baseline generation configurations differ; select one consistent run")
    settings = GenerationSettings(**configs[0])
    models = {baseline["experiment_metadata"]["model_metadata"].get("model_id") for baseline in baselines}
    if len(models) != 1 or None in models:
        raise ValueError("baseline model identity is missing or inconsistent")
    model_id = models.pop()
    mirror = RcloneMirror(args.output_dir, args.drive_output)
    mirror.restore()
    old_manifest = args.output_dir / "manifest.json"
    previous = json.loads(old_manifest.read_text(encoding="utf-8")) if old_manifest.exists() else None
    baseline_revisions = {baseline["experiment_metadata"]["model_metadata"].get("model_revision") for baseline in baselines}
    known_revisions = baseline_revisions - {None}
    if len(known_revisions) > 1:
        raise ValueError("baseline model revisions are inconsistent")
    revision = args.revision or (previous["run_config"]["model_revision"] if previous else next(iter(known_revisions), None))
    tokenizer, model, precision = load_qwen_model(model_id, revision=revision)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if not isinstance(resolved_revision, str) or not resolved_revision:
        raise ValueError("model revision could not be resolved; use --revision with a commit hash")
    tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or resolved_revision
    generator = QwenGenerator(tokenizer, model, settings=settings, model_id=model_id, precision=precision)
    run_config = {
        "generation_settings": settings.to_dict(), "model_id": model_id,
        "model_revision": resolved_revision, "tokenizer_revision": tokenizer_revision,
        "precision": precision,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot = args.output_dir / "sessions" / f"{stamp}.json"
    code_hashes = {}
    for file in (Path(__file__), Path(__file__).with_name("removal_experiment.py"), Path(__file__).with_name("state_removal.py")):
        code_hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    save_json(snapshot, {
        "created_at_utc": stamp, "environment": collect_environment_info(),
        "run_config": run_config, "input_path": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "pip_freeze": subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout,
        "repository_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "code_sha256": code_hashes,
    })
    mirror(snapshot)
    summary = run_removal_experiments(
        selected, generator, args.output_dir, run_config=run_config,
        persist_callback=mirror, limit=args.limit,
        progress_callback=lambda row: print(
            f"{row['trace_id']} / {row['removed_state_id']}: {row['experiment_status']} | correct={row['is_correct']} | saved to Drive",
            flush=True,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
