"""Shared utilities for the separate non-overlapping four-MLP block pilot."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .common import (MODEL_ID, atomic_json, correct_answer,
                     digest, load_model, parsed_answer, read_json, seed_all)

BLOCK4_PROTOCOL = "v3-question-level-nonoverlapping-mlp-block4-device-cond-v3"
LEGACY_BLOCK4_PROTOCOLS = {"v3-question-level-nonoverlapping-mlp-block4-decode-only-v1"}


def make_groups(layer_count, group_size=4):
    if group_size != 4 or layer_count < group_size or layer_count % group_size:
        raise ValueError(f"Expected a layer count divisible by 4, got {layer_count}")
    return [tuple(range(i, i + group_size)) for i in range(0, layer_count, group_size)]


def prepare_manifest(directory, config):
    """Strict run identity/code check; block artifacts never mix with other protocols."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    code_hashes = {
        p.name: digest(p) for p in Path(__file__).parent.glob("block_*.py")
    }
    if path.exists():
        old = read_json(path)
        if old.get("identity") != identity or old.get("code_sha256") != code_hashes:
            raise ValueError("Block run settings/code changed; use a new output directory.")
        return old
    result = {"identity": identity, "config": config, "code_sha256": code_hashes}
    atomic_json(path, result)
    return result


def status(directory, **kwargs):
    atomic_json(Path(directory) / "run_status.json", {
        "updated_unix": time.time(), "pid": os.getpid(), **kwargs,
        "note": "Updated at trial boundaries; check PID for live process status.",
    })


def group_action_features(features, groups=None):
    """Concatenate all four group MLP inputs and a group-id one-hot vector."""
    features = np.asarray(features, dtype=np.float32)
    groups = groups or make_groups(len(features))
    identity = np.eye(len(groups), dtype=np.float32)
    return np.stack([
        np.concatenate((*[features[layer] for layer in group], identity[index]))
        for index, group in enumerate(groups)
    ])


def group_action_features_torch(features, groups):
    """Build controller inputs on the activation device without a CPU round-trip."""
    layer_features = torch.stack([features[index] for index in range(len(features))]).float()
    identity = torch.eye(len(groups), dtype=layer_features.dtype, device=layer_features.device)
    return torch.stack([
        torch.cat((layer_features[list(group)].reshape(-1), identity[index]))
        for index, group in enumerate(groups)
    ])


def select_group_mask(scores, eligible_indices, threshold, max_groups):
    """Select up to max_groups above-threshold groups without reading GPU values."""
    if not 1 <= max_groups <= 4:
        raise ValueError("max_groups must be between 1 and 4")
    scores = scores.reshape(-1)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if not eligible_indices:
        return mask
    candidate_ids = torch.as_tensor(eligible_indices, dtype=torch.long, device=scores.device)
    values = scores.index_select(0, candidate_ids)
    finite = torch.isfinite(values)
    ranked_values, offsets = values.masked_fill(~finite, -float("inf")).topk(
        min(max_groups, len(eligible_indices)),
    )
    selected_ids = candidate_ids.index_select(0, offsets)
    return mask.scatter(0, selected_ids, torch.isfinite(ranked_values) & (ranked_values >= threshold))


class BlockBypass:
    """Capture prefill features, then route MLP bypasses with a device predicate."""
    def __init__(self, model, forced_group=None, policy=None, capture=False):
        self.model = model
        self.layers = model.model.layers
        self.forced_group = tuple(forced_group) if forced_group is not None else None
        self.policy, self.capture = policy, capture
        self.selected_group = self.forced_group
        self.features, self.originals = {}, []
        self._feature_hooks = []
        self._model_forward_original = None
        self._model_forward_had_override = False
        self._model_forward_wrapped = False
        self._prefill_complete = False
        self.scores = None
        self.score_values = None
        self.skip_mask = None
        self.selected_group_mask = None

    def __enter__(self):
        needs_features = self.capture or self.policy is not None
        if not needs_features and self.selected_group is None:
            return self

        if needs_features:
            for index, layer in enumerate(self.layers):
                def capture_input(module, args, _index=index):
                    if _index not in self.features:
                        self.features[_index] = args[0][0, -1].detach().clone()

                self._feature_hooks.append(
                    layer.mlp.register_forward_pre_hook(capture_input)
                )

        # Wrap model.forward only for prompt prefill. Once features are captured,
        # each MLP gets a compiled torch.cond route whose predicate remains on GPU.
        self._model_forward_had_override = "forward" in self.model.__dict__
        self._model_forward_original = self.model.forward
        original = self._model_forward_original

        def finish_prefill(*args, **kwargs):
            output = original(*args, **kwargs)
            if not self._prefill_complete:
                self._after_prefill()
                self._restore_model_forward()
            return output

        self.model.forward = finish_prefill
        self._model_forward_wrapped = True
        return self

    def _restore_model_forward(self):
        if self._model_forward_had_override:
            self.model.forward = self._model_forward_original
        elif "forward" in self.model.__dict__:
            del self.model.forward
        self._model_forward_wrapped = False

    def _after_prefill(self):
        for hook in self._feature_hooks:
            hook.remove()
        self._feature_hooks.clear()

        if self.capture or self.policy is not None:
            if len(self.features) != len(self.layers):
                raise RuntimeError("Could not capture prompt features from every MLP")

        device = next(self.model.parameters()).device
        if self.policy is not None:
            self.scores = self.policy.score(self.features)
            self.selected_group_mask = select_group_mask(
                self.scores, self.policy.eligible_group_indices,
                self.policy.threshold, self.policy.max_removal_groups,
            )

        # Construct a layer-to-group table on the activation device. For a
        # controller-selected groups, both ranking/threshold and this mask stay on
        # the device; no .item() or CPU decision is needed before decoding.
        group_ids = torch.full((len(self.layers),), -1, dtype=torch.long, device=device)
        groups = self.policy.groups if self.policy is not None else [self.forced_group]
        for group_index, group in enumerate(groups):
            if group is not None:
                group_ids[list(group)] = group_index

        if self.policy is not None:
            self.skip_mask = self.selected_group_mask.index_select(0, group_ids)
        elif self.forced_group is not None:
            self.skip_mask = group_ids == 0
        else:
            self.skip_mask = torch.zeros(len(self.layers), dtype=torch.bool, device=device)

        if self.policy is not None or self.forced_group is not None:
            self._install_conditional_routes()
        self._prefill_complete = True

    def _install_conditional_routes(self):
        """Install compiled conditional MLPs; never fall back to host-side routing."""
        if not hasattr(torch, "cond") or not hasattr(torch, "compile"):
            raise RuntimeError(
                "GPU-side bypass requires torch.cond and torch.compile; this PyTorch build does not provide both."
            )

        for index, layer in enumerate(self.layers):
            module = layer.mlp
            original = module.forward
            self.originals.append((module, original, "forward" in module.__dict__))
            compiled = getattr(module, "_pssn_device_cond_forward", None)
            if compiled is None:
                def conditional_forward(skip, hidden_states, _original=original):
                    return torch.cond(
                        skip,
                        lambda hidden: torch.zeros_like(hidden),
                        lambda hidden: _original(hidden),
                        (hidden_states,),
                    )

                # fullgraph=True prevents Dynamo from graph-breaking around the
                # predicate, which could otherwise reintroduce a host sync.
                compiled = torch.compile(conditional_forward, fullgraph=True)
                module._pssn_device_cond_forward = compiled

            def routed_forward(hidden_states, *args, _compiled=compiled,
                               _index=index, _owner=self, **kwargs):
                if args or kwargs:
                    raise TypeError("Qwen MLP conditional route expects only hidden_states")
                return _compiled(_owner.skip_mask[_index], hidden_states)

            module.forward = routed_forward

    def resolved_selected_groups(self):
        """Resolve device-selected groups for logging after timed generation."""
        if self.forced_group is not None:
            return [self.forced_group]
        if self.selected_group_mask is None or self.policy is None:
            return []
        selected = self.selected_group_mask.detach().cpu().tolist()
        return [tuple(group) for group, keep in zip(self.policy.groups, selected) if keep]

    def __exit__(self, *exc):
        for hook in self._feature_hooks:
            hook.remove()
        self._feature_hooks.clear()
        if self._model_forward_wrapped:
            self._restore_model_forward()
        for module, original, had_override in self.originals:
            if had_override:
                module.forward = original
            else:
                del module.forward


class BlockPolicy:
    def __init__(self, checkpoint, max_removal_groups=1, device=None):
        from .common import Controller
        if not 1 <= max_removal_groups <= 4:
            raise ValueError("max_removal_groups must be between 1 and 4")
        self.max_removal_groups = max_removal_groups
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if self.data["protocol"] != BLOCK4_PROTOCOL and self.data["protocol"] not in LEGACY_BLOCK4_PROTOCOLS:
            raise ValueError(f"Unsupported four-layer controller protocol: {self.data['protocol']}")
        self.groups = [tuple(group) for group in self.data["groups"]]
        if self.groups != make_groups(self.data["num_layers"]):
            raise ValueError("Controller groups do not match the non-overlapping four-MLP layout")
        expected_width = 4 * self.data["hidden_size"] + len(self.groups)
        if self.data["width"] != expected_width:
            raise ValueError("Controller feature width does not match its saved group/hidden dimensions")
        self.eligible_group_indices = list(self.data["eligible_group_indices"])
        self.threshold = float(self.data["threshold"])
        self.model = Controller(self.data["width"], self.data["hidden"]).eval().to(self.device)
        self.model.load_state_dict(self.data["state_dict"])
        self.mean, self.std = self.data["mean"].to(self.device), self.data["std"].to(self.device)

    def score(self, layer_features):
        x = group_action_features_torch(layer_features, self.groups)
        return torch.sigmoid(self.model((x - self.mean) / self.std))


def generate_group(tokenizer, model, question, max_new_tokens, seed,
                   forced_group=None, policy=None, capture=False):
    seed_all(seed)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": (
            question["question"] + "\nSolve carefully. You may use a short calculation. "
            "End with a line exactly: FINAL: <number>."
        )}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode(), BlockBypass(model, forced_group, policy, capture) as bypass:
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    ids = outputs[0, inputs["input_ids"].shape[1]:]
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    complete = bool(len(ids) and ids[-1].item() in eos_ids)
    text = tokenizer.decode(ids, skip_special_tokens=True)
    answer = parsed_answer(text)
    # The mask transfer is metadata-only and occurs after elapsed time is
    # captured; the decision that controls MLP execution never leaves the GPU.
    selected_groups = [list(group) for group in bypass.resolved_selected_groups()]
    selected = sorted(layer for group in selected_groups for layer in group) or None
    all_groups = policy.groups if policy is not None else make_groups(len(bypass.layers))
    selected_indices = [index for index, group in enumerate(all_groups) if list(group) in selected_groups]
    # Greedy cached generation obtains its first new token from the prompt pass;
    # each later output token requires one single-token decoder pass.
    decode_steps = max(len(ids) - 1, 0)
    total_mlp_evaluations = len(bypass.layers) * (int(inputs["input_ids"].shape[1]) + decode_steps)
    decode_mlp_evaluations = len(bypass.layers) * decode_steps
    skipped_mlp_evaluations = len(selected or []) * decode_steps
    result = {
        "question_id": question["question_id"], "response": text,
        "extracted_answer": answer, "complete": complete,
        "finish_reason": "eos" if complete else "length",
        "correct": complete and correct_answer(answer, question["reference_answer"]),
        "input_tokens": int(inputs["input_ids"].shape[1]), "output_tokens": len(ids),
        "seconds": elapsed, "tokens_per_second": len(ids) / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "routing_implementation": "torch.cond+torch.compile",
        "selection_sync_in_timed_region": False,
        "selected_group": selected, "selected_layers": selected,
        "selected_group_legacy_field": "flattened selected layer indices; use selected_groups for group boundaries",
        "selected_groups": selected_groups, "selected_group_indices": selected_indices,
        "selected_group_count": len(selected_groups),
        "max_removal_groups": policy.max_removal_groups if policy is not None else None,
        "group_probabilities": (bypass.scores.reshape(-1).detach().cpu().tolist()
                                 if bypass.scores is not None else None),
        "token_mlp_evaluations_total": total_mlp_evaluations,
        "token_mlp_evaluations_decode": decode_mlp_evaluations,
        "token_mlp_evaluations_skipped": skipped_mlp_evaluations,
        "token_mlp_evaluation_count_method": "derived_from_prompt_length_and_greedy_cached_decode_output_length",
    }
    features = (np.stack([bypass.features[j].cpu().numpy() for j in range(len(bypass.layers))])
                if capture else None)
    return result, features


def group_matrices(directory, records, indices):
    from .common import read_json
    xs, ys, metadata = [], [], []
    for i in indices:
        question = records[i]
        folder = Path(directory) / "questions" / question["question_id"]
        labels = read_json(folder / "labels.json")
        if digest(folder / "features.npz") != labels["feature_sha256"]:
            raise ValueError("Feature checksum mismatch")
        with np.load(folder / "features.npz", allow_pickle=False) as archive:
            features = archive["features"]
        groups = make_groups(len(features))
        action_x = group_action_features(features, groups)
        raw_labels = labels.get("raw_labels", labels["labels"])
        if len(raw_labels) != len(groups) or len(labels["labels"]) != len(groups):
            raise ValueError("The saved labels do not match the four-layer group layout")
        for index, (group, label, raw_label) in enumerate(zip(groups, labels["labels"], raw_labels)):
            # Conservative policy: any unscored outcome is treated as unsafe to skip.
            was_unscored = raw_label < 0
            label = 0 if was_unscored else label
            xs.append(action_x[index])
            ys.append(label)
            metadata.append({"question_id": question["question_id"], "group_index": index,
                             "group_layers": list(group), "label": label,
                             "was_unscored": was_unscored})
    if not xs:
        raise ValueError("Split has no block labels")
    return np.stack(xs), np.asarray(ys, dtype=np.float32), metadata
