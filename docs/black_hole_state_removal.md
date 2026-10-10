# Week 3 — Controlled semantic-state removal

W3-002 implements a deterministic leave-one-state-out transformation. The
function first validates the complete state sequence against its source trace,
then removes exactly one requested state. The final-answer state is protected.
Remaining states and identifiers stay in original order, and the source input
is never modified.

The modified reasoning context is reconstructed by joining the remaining state
texts with a newline by default. Pass `reconstruction_separator` to choose a
different separator. The returned record preserves the removed state ID,
zero-based index, text and token count, plus original and modified counts and
the complete remaining state records.

```python
from src.black_hole.state_removal import remove_semantic_state

removal = remove_semantic_state(
    states=validated_states,
    state_id="S3",
    source_trace=baseline_record["reasoning_trace"],
)
print(removal["modified_reasoning_context"])
```

Each candidate is removed independently from the original validated sequence;
do not apply one removal to the already modified output when creating a
leave-one-out experiment set.
