"""Semantic-state schema and utilities."""

from .schema import SemanticState
from .segmenter import UnsegmentableTraceError, segment_reasoning_trace

__all__ = [
    "SemanticState",
    "UnsegmentableTraceError",
    "segment_reasoning_trace",
]
