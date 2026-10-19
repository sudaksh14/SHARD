"""
Tests for the final SHARD model class (`networks.shard_deit_v3`) on a tiny configuration (CPU, seconds).

  * every level is a self-contained regular model: shard(x) at level i == make_base_copy()(x)
  * nested weights: the parameters of level i are a leading slice of the parameters of every wider level
  * level deltas (export -> apply up/down) reproduce the neighbouring level's regular model
  * the released checkpoint loads strictly (only when SHARD_CHECKPOINT points to SHARD_deit_v3_5level.pt)
"""
import os
import unittest

import torch

from utils.switching_harness import common

torch.manual_seed(0)


def randomize(model, scale=0.01):
    for p in model.parameters():
        p.data = torch.rand(p.shape).to(p.data) * scale
    return model


class TestShardDeitV3Tiny(unittest.TestCase):
    def setUp(self):
        self.cfg = common.tiny_config("shard_deit_v3", levels=3)
        self.model = randomize(self.cfg.make_model()).eval()
        self.x = torch.rand(2, 3, 32, 32)

    def test_levels(self):
        self.assertEqual(self.model.max_level(), 2)
        self.assertEqual(self.model.current_level(), 2)

    def test_output_shape_per_level(self):
        for level in range(self.model.max_level() + 1):
            self.model.set_level_use(level)
            self.assertEqual(self.model(self.x).shape, (2, 10))

    def test_level_equals_regular_model(self):
        with torch.no_grad():
            for level in range(self.model.max_level() + 1):
                self.model.set_level_use(level)
                reg = self.model.make_base_copy().eval()
                torch.testing.assert_close(self.model(self.x), reg(self.x), rtol=1e-4, atol=1e-5)

    def test_regular_model_size_grows_with_level(self):
        sizes = []
        for level in range(self.model.max_level() + 1):
            self.model.set_level_use(level)
            sizes.append(sum(p.numel() for p in self.model.make_base_copy().parameters()))
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(len(set(sizes)), len(sizes))

    def test_nested_weights(self):
        """Patch-embedding and token parameters of level i are the leading slice of level i+1."""
        with torch.no_grad():
            copies = []
            for level in range(self.model.max_level() + 1):
                self.model.set_level_use(level)
                copies.append(self.model.make_base_copy().state_dict())
            for lo, hi in zip(copies[:-1], copies[1:]):
                d_lo = lo["class_token.token"].shape[-1]
                torch.testing.assert_close(hi["class_token.token"][..., :d_lo], lo["class_token.token"])
                torch.testing.assert_close(hi["conv_proj.weight"][:d_lo], lo["conv_proj.weight"])

    def test_up_down_delta(self):
        with torch.no_grad():
            self.model.set_level_use(1)
            reg1 = self.model.make_base_copy().eval()
            self.model.set_level_use(2)
            down, up = self.model.export_level_delta()
            up.apply(reg1)                       # level 1 regular model -> level 2
            self.model.set_level_use(2)
            torch.testing.assert_close(self.model(self.x), reg1.eval()(self.x), rtol=1e-4, atol=1e-5)
            self.model.set_level_use(1)
            down1, _ = self.model.export_level_delta()
            down1.apply(reg1)                    # level 2 regular model -> level 1 (down delta exported at level 1)
            torch.testing.assert_close(self.model(self.x), reg1.eval()(self.x), rtol=1e-4, atol=1e-5)

    def test_invalid_level_rejected(self):
        with self.assertRaises(AssertionError):
            self.model.set_level_use(self.model.max_level() + 1)


@unittest.skipUnless(os.environ.get("SHARD_CHECKPOINT"), "set SHARD_CHECKPOINT=/path/to/SHARD_deit_v3_5level.pt to run")
class TestReleasedCheckpoint(unittest.TestCase):
    """Needs the released 5-level checkpoint (~355 MB), CPU only."""

    def test_strict_load_and_forward(self):
        cfg = common.paper_config("shard_deit_v3")
        model = common.build_shard_model(cfg, os.environ["SHARD_CHECKPOINT"])
        self.assertEqual(model.max_level(), 4)
        x = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            for level in range(5):
                model.set_level_use(level)
                out = model(x)
                self.assertEqual(out.shape, (1, 1000))
                self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
