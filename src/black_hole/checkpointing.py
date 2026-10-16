"""Atomic persistence helpers for the Week 4 detector trainer."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    """Write a torch checkpoint via a same-directory temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_save(payload: dict[str, Any] | list[Any], path: str | Path) -> None:
    """Write JSON atomically so Drive interruptions do not leave partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
