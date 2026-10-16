"""
Tests for `utils.switching_harness` (real DRM classes, tiny model, CPU) and for the published switching numbers.
"""
import csv
import io
import os
import tempfile
import unittest

import torch

from utils.switching_harness import common, delta_switching, full_model_switching, verify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestSwitchingCorrectness(unittest.TestCase):
    def test_delta_switching_matches_shard_model(self):
        for arch in common.ARCHS:
            with self.subTest(arch=arch):
                self.assertEqual(verify.check(arch), [])

    def test_sequences_cover_up_down_and_jumps(self):
        seqs = verify.sequences(4)
        self.assertEqual(seqs["down"], [3, 2, 1, 0])
        self.assertEqual(seqs["up"], [1, 2, 3, 4])
        self.assertIn(4, seqs["jumps"])
        self.assertTrue(all(0 <= v <= 4 for v in seqs["random_walk"]))


class TestSwitchingMeasurement(unittest.TestCase):
    def test_delta_measurement_runs_and_serialises(self):
        cfg = common.tiny_config("shard_vit")
        model = common.build_shard_model(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            total_ns, size = delta_switching.measure(cfg, model, "cpu", iters=1, no_cache=False,
                                                     shared_manager=False, workdir=tmp)
        self.assertGreater(size, 0)
        self.assertEqual(len(total_ns), 3)
        self.assertTrue(all(total_ns[i][i] == 0 for i in range(3)))   # no self-switch is timed
        self.assertTrue(all(total_ns[i][j] >= 0 for i in range(3) for j in range(3)))

    def test_summarize_orientation(self):
        m = [[0, 10, 20], [1, 0, 30], [2, 3, 0]]
        s = delta_switching.summarize(m)
        self.assertAlmostEqual(s["mean_up_ms"], 20.0)      # (10 + 20 + 30) / 3
        self.assertAlmostEqual(s["mean_down_ms"], 2.0)     # (1 + 2 + 3) / 3
        self.assertAlmostEqual(s["mean_all_ms"], 11.0)

    def test_full_model_export_and_load(self):
        cfg = common.tiny_config("shard_deit_v3")
        model = common.build_shard_model(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            paths = full_model_switching.export_level_models(model, tmp)
            self.assertEqual(len(paths), 3)
            total_ns = full_model_switching.measure(paths, "cpu", trials=1)
            self.assertEqual(len(total_ns), 3)
            m = full_model_switching._load(paths[1], "cpu")
            self.assertIsInstance(m, torch.nn.Module)

    def test_compact_level_files_are_not_larger(self):
        cfg = common.tiny_config("shard_deit_v3")
        model = common.build_shard_model(cfg)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            plain = full_model_switching.export_level_models(model, a, compact=False)
            compact = full_model_switching.export_level_models(model, b, compact=True)
            sizes = [(os.path.getsize(p), os.path.getsize(c)) for p, c in zip(plain, compact)]
            self.assertTrue(all(c <= p for p, c in sizes))
            self.assertLess(sizes[0][1], sizes[0][0])        # smallest level: views drag the full storage along
            # same function after compaction
            x = torch.rand(1, 3, 32, 32)
            m_plain = full_model_switching._load(plain[0], "cpu")
            m_compact = full_model_switching._load(compact[0], "cpu")
            torch.testing.assert_close(m_plain(x), m_compact(x))

    def test_delta_file_base_chunk_is_compact(self):
        """The delta file must not carry the full-size storage behind the level-0 views."""
        cfg = common.tiny_config("shard_deit_v3")
        model = common.build_shard_model(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.delta")
            size = delta_switching.write_delta_file(model, path, starting_level=0)
        model.set_level_use(model.max_level())
        full = io.BytesIO()
        torch.save(model.make_base_copy().state_dict(), full)
        # base (level 0) + all up/down deltas ~ one full model; allow metadata overhead but not the old ~1.5x bloat
        self.assertLess(size, 1.2 * len(full.getvalue()))

    @unittest.skipUnless(hasattr(os, "posix_fadvise"), "Linux only")
    def test_nocache_manager_switches_correctly(self):
        from utils.switching_harness.nocache import file_delta_manager_nocache
        cfg = common.tiny_config("shard_vit")
        model = common.build_shard_model(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.delta")
            delta_switching.write_delta_file(model, path, starting_level=0)
            with file_delta_manager_nocache(path, cfg.create_base_config(0).no_prebuilt()) as mgr:
                x = torch.rand(1, 3, 32, 32)
                for level in (2, 0, 1):
                    mgr.move_to(level)
                    model.set_level_use(level)
                    torch.testing.assert_close(mgr.managed_model().eval()(x), model(x), rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
