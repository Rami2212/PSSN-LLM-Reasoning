"""Run W3003 from a Python process on the configured ROCm server.

Examples:
    python notebooks/11_black_hole_removal_experiments.py --limit 2
    python notebooks/11_black_hole_removal_experiments.py
    python notebooks/11_black_hole_removal_experiments.py --input /path/to/eligible_traces.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive-mount",
        type=Path,
        default=Path.home() / "gdrive",
        help="local rclone mount of Google Drive (default: ~/gdrive)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="W3001 eligible_traces.jsonl; defaults to the latest file under the Drive mount",
    )
    parser.add_argument("--run-name", default="w3-003-run1", help="stable run folder name for resume")
    parser.add_argument(
        "--limit",
        type=int,
        help="maximum NEW experiments this invocation; omit to run all pending experiments",
    )
    parser.add_argument("--revision", help="optional model commit hash to reproduce the baseline")
    args = parser.parse_args()

    if args.input is not None:
        input_path = args.input.expanduser().resolve()
    else:
        selection_root = args.drive_mount.expanduser() / "PSSN2/artifacts/expanded_baseline"
        candidates = sorted(
            selection_root.rglob("black_hole_trace_selection/*/eligible_traces.jsonl")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No W3001 selection found under {selection_root}. "
                "Mount Google Drive and run notebook 10 first, or pass --input."
            )
        input_path = candidates[-1]

    if not input_path.is_file():
        raise FileNotFoundError(f"W3001 selection file does not exist: {input_path}")

    local_output = Path.home() / "pssn-work/checkpoints" / args.run_name
    remote_output = f"gdrive:PSSN2/artifacts/black_hole/{args.run_name}"
    command = [
        "--input", str(input_path),
        "--output-dir", str(local_output),
        "--drive-output", remote_output,
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.revision:
        command.extend(["--revision", args.revision])

    print(f"W3001 input: {input_path}", flush=True)
    print(f"Local checkpoints: {local_output}", flush=True)
    print(f"Drive results: {remote_output}", flush=True)

    from src.black_hole.run_experiments import main as run_experiments

    run_experiments(command)


if __name__ == "__main__":
    main()
