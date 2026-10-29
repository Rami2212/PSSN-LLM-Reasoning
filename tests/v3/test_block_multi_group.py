"""CPU checks for capped group selection and combined conditional bypasses."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from v3.block_common import BlockBypass, BlockPolicy, make_groups, select_group_mask
from v3.common import Controller


class GroupSelectionTest(unittest.TestCase):
    def test_highest_four_eligible_scores(self):
        scores = torch.tensor([.90, .40, .80, .99, .75, .50, .10, .85, .70])
        mask = select_group_mask(scores, list(range(9)), .5, 4)
        self.assertEqual(mask.nonzero().flatten().tolist(), [0, 2, 3, 7])

    def test_threshold_does_not_force_four_removals(self):
        scores = torch.tensor([.8, .4, .5, .1])
        mask = select_group_mask(scores, [0, 1, 2, 3], .5, 4)
        self.assertEqual(mask.tolist(), [True, False, True, False])
        self.assertFalse(select_group_mask(scores, [0, 1, 2, 3], .99, 4).any())

    def test_ineligible_and_nonfinite_scores_are_excluded(self):
        scores = torch.tensor([.99, .8, float("nan"), .7, float("inf")])
        mask = select_group_mask(scores, [1, 2, 3, 4], .5, 4)
        self.assertEqual(mask.tolist(), [False, True, False, True, False])
        self.assertFalse(select_group_mask(scores, [], .5, 4).any())

    def test_single_group_mode_and_limit_validation(self):
        scores = torch.tensor([.8, .9, .7])
        self.assertEqual(select_group_mask(scores, [0, 1, 2], .5, 1).tolist(),
                         [False, True, False])
        for invalid in (0, 5):
            with self.assertRaises(ValueError):
                select_group_mask(scores, [0, 1, 2], .5, invalid)

    def test_existing_v1_controller_loads_for_new_multi_group_benchmark(self):
        import tempfile
        from pathlib import Path

        groups = make_groups(8)
        width, hidden = 4 * 2 + len(groups), 4
        controller = Controller(width, hidden)
        checkpoint = {
            "protocol": "v3-question-level-nonoverlapping-mlp-block4-decode-only-v1",
            "state_dict": controller.state_dict(), "width": width, "hidden": hidden,
            "mean": torch.zeros(width), "std": torch.ones(width),
            "num_layers": 8, "hidden_size": 2, "groups": [list(group) for group in groups],
            "eligible_group_indices": [0, 1], "threshold": .5,
        }
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "controller.pt"
            torch.save(checkpoint, path)
            policy = BlockPolicy(path, max_removal_groups=4, device="cpu")
        self.assertEqual(policy.max_removal_groups, 4)
        self.assertEqual(policy.groups, groups)
        self.assertEqual(policy.score({i: torch.ones(2) for i in range(8)}).shape, (2,))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module() for _ in range(20)])
        for layer in self.model.layers:
            layer.mlp = nn.Linear(1, 1, bias=False)
            nn.init.constant_(layer.mlp.weight, .01)

    def forward(self, hidden):
        for layer in self.model.layers:
            hidden = hidden + layer.mlp(hidden)
        return hidden


class CombinedBypassTest(unittest.TestCase):
    def test_prefill_all_layers_decode_four_groups_then_restore(self):
        model = TinyModel().eval()
        groups = make_groups(20)
        policy = SimpleNamespace(
            groups=groups, eligible_group_indices=list(range(5)),
            threshold=.5, max_removal_groups=4,
            score=lambda features: torch.tensor([.9, .8, .7, .6, .1]),
        )
        actual_compile = torch.compile

        def cpu_compile(fn, **kwargs):
            # Exercise real Dynamo/torch.cond graph capture without requiring
            # the server's ROCm/Inductor toolchain on the local Windows PC.
            kwargs["backend"] = "eager"
            return actual_compile(fn, **kwargs)

        with patch("v3.block_common.torch.compile", side_effect=cpu_compile):
            with torch.inference_mode(), BlockBypass(model, policy=policy) as bypass:
                prefill = model(torch.ones(1, 3, 1))
                decode = model(torch.ones(1, 1, 1))
                self.assertEqual(bypass.resolved_selected_groups(), groups[:4])
                self.assertEqual(bypass.skip_mask.sum().item(), 16)
            with torch.inference_mode():
                restored = model(torch.ones(1, 1, 1))

        torch.testing.assert_close(prefill, torch.full_like(prefill, 1.01 ** 20))
        torch.testing.assert_close(decode, torch.full_like(decode, 1.01 ** 4))
        torch.testing.assert_close(restored, torch.full_like(restored, 1.01 ** 20))
        self.assertTrue(all("forward" not in layer.mlp.__dict__ for layer in model.model.layers))


if __name__ == "__main__":
    unittest.main()
