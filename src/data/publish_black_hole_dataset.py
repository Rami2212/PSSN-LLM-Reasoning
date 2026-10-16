"""Publish the validated PSSN Week 3 Black Hole dataset to Hugging Face."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


DEFAULT_REPO_ID = "ramitha2002/pssn-reasoning-removability-gsm8k"
REQUIRED_DATASET_FILES = (
    "black_hole_training_dataset.jsonl",
    "black_hole_training_dataset_summary.json",
    "black_hole_training_dataset_manifest.json",
    "excluded_dataset_rows.jsonl",
)
REQUIRED_SPLIT_FILES = (
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "train_manifest.json",
    "validation_manifest.json",
    "test_manifest.json",
    "week3_quality_summary.json",
)


def _token_from_environment() -> str:
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        token = os.environ.get(name, "").strip()
        if token:
            return token
    raise RuntimeError(
        "No Hugging Face token was found. Set HF_TOKEN or HUGGINGFACE_TOKEN to a write token."
    )


def _require_files(directory: Path, expected: tuple[str, ...]) -> None:
    missing = [name for name in expected if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {directory}: {', '.join(missing)}")


def _build_publish_tree(dataset_dir: Path, splits_dir: Path, run_dir: Path, card_path: Path, staging: Path) -> None:
    _require_files(dataset_dir, REQUIRED_DATASET_FILES)
    _require_files(splits_dir, REQUIRED_SPLIT_FILES)
    if not (run_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing W3 run manifest: {run_dir / 'manifest.json'}")
    if not card_path.is_file():
        raise FileNotFoundError(f"Missing dataset card: {card_path}")

    shutil.copy2(card_path, staging / "README.md")
    shutil.copy2(
        dataset_dir / "black_hole_training_dataset.jsonl",
        staging / "all_accepted_sentence_context_labels.jsonl",
    )
    for name in REQUIRED_DATASET_FILES[1:]:
        shutil.copy2(dataset_dir / name, staging / name)
    for name in REQUIRED_SPLIT_FILES:
        shutil.copy2(splits_dir / name, staging / name)
    shutil.copy2(run_dir / "manifest.json", staging / "w3_run_manifest.json")


def publish(
    dataset_dir: Path,
    splits_dir: Path,
    run_dir: Path,
    card_path: Path,
    repo_id: str,
    private: bool,
    allow_existing: bool,
) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - dependency environment specific
        raise RuntimeError("Install huggingface_hub before publishing.") from exc

    token = _token_from_environment()
    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=allow_existing)
    except Exception as exc:
        raise RuntimeError(
            f"Could not create the new dataset repository {repo_id}. "
            "If it already exists, rerun with --allow-existing."
        ) from exc

    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    if bool(info.private) != private:
        requested = "private" if private else "public"
        raise PermissionError(f"Repository visibility is not the requested {requested} setting.")

    # Some Windows sandbox environments deny writes to the system temp folder.
    # Stage inside the current workspace so publishing works consistently there
    # and in notebook/server environments, then let TemporaryDirectory remove it.
    with TemporaryDirectory(prefix=".pssn_hf_black_hole_", dir=Path.cwd()) as temp_dir:
        staging = Path(temp_dir)
        _build_publish_tree(dataset_dir, splits_dir, run_dir, card_path, staging)
        commit = api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload validated PSSN reasoning-removability GSM8K dataset",
        )
    return commit.oid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--card", type=Path, default=Path("huggingface_dataset/black_hole_README.md"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true", help="Create a private dataset instead of the public default.")
    parser.add_argument("--allow-existing", action="store_true", help="Allow updating an existing dataset repository.")
    args = parser.parse_args()

    commit = publish(
        dataset_dir=args.dataset_dir,
        splits_dir=args.splits_dir,
        run_dir=args.run_dir,
        card_path=args.card,
        repo_id=args.repo_id,
        private=args.private,
        allow_existing=args.allow_existing,
    )
    print(f"Published: https://huggingface.co/datasets/{args.repo_id}")
    print(f"Commit: {commit}")


if __name__ == "__main__":
    main()
