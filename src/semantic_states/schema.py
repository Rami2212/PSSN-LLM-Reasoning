"""Typed, serializable semantic-state record (no segmentation algorithm)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SemanticState:
    """One contiguous, ordered reasoning unit from a source trace.

    Character offsets are half-open Python Unicode indices into the exact
    source ``reasoning_trace``. Token counts use the experiment tokenizer with
    special tokens disabled.
    """

    problem_id: str
    trace_id: str
    state_id: str
    state_index: int
    text: str
    start_char: int
    end_char: int
    token_count: int
    is_final_state: bool

    def __post_init__(self) -> None:
        for name in ("problem_id", "trace_id", "state_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if isinstance(self.state_index, bool) or not isinstance(self.state_index, int):
            raise TypeError("state_index must be an integer")
        if self.state_index < 0:
            raise ValueError("state_index must be non-negative")
        expected_id = f"S{self.state_index + 1}"
        if self.state_id != expected_id:
            raise ValueError(f"state_id must be {expected_id!r} for this state_index")

        for name in ("start_char", "end_char", "token_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if self.token_count == 0:
            raise ValueError("token_count must be positive for a non-empty state")
        if not isinstance(self.is_final_state, bool):
            raise TypeError("is_final_state must be a boolean")

    def validate_trace_span(self, reasoning_trace: str) -> None:
        """Raise if this record's character span does not match its source."""
        if not isinstance(reasoning_trace, str):
            raise TypeError("reasoning_trace must be a string")
        if self.end_char > len(reasoning_trace):
            raise ValueError("state span extends past the source reasoning trace")
        if reasoning_trace[self.start_char : self.end_char] != self.text:
            raise ValueError("text does not equal the declared source trace span")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-serializable field mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "SemanticState":
        """Build and validate a state from a mapping with the canonical keys."""
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        fields = cls.__dataclass_fields__
        missing = fields.keys() - record.keys()
        extra = record.keys() - fields.keys()
        if missing:
            raise ValueError(f"missing semantic-state fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"unexpected semantic-state fields: {sorted(extra)}")
        return cls(**{key: record[key] for key in fields})
