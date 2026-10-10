"""Week 3 Black Hole trace-selection utilities."""

from .trace_selection import save_trace_selection, select_eligible_traces
from .state_removal import remove_semantic_state
from .removal_experiment import run_removal_experiments

__all__ = ["remove_semantic_state", "save_trace_selection", "select_eligible_traces", "run_removal_experiments"]
