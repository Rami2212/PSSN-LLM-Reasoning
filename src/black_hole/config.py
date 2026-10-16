"""Configuration for the Week 4 Black Hole sentence detector."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BlackHoleDetectorConfig:
    """Serializable model settings; inputs are pooled sentence activations."""

    input_size: int = 2560
    hidden_size: int = 256
    dropout: float = 0.2
    positive_label: str = "low_value_in_context"
    negative_label: str = "answer_sensitive_in_context"
    schema_version: str = "black-hole-detector-config-v1"

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.positive_label == self.negative_label:
            raise ValueError("positive_label and negative_label must differ")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "BlackHoleDetectorConfig":
        allowed = {field for field in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BlackHoleDetectorConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Detector config must be a JSON object")
        return cls.from_dict(values)
