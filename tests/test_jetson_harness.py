"""
CPU-only tests for `utils.jetson_harness`: argument handling, serialisation, preflight on a non-Jetson machine, and the
Nothing here needs a Jetson; the benchmark itself is only exercised with a tiny model without power sensor.
"""
import json
import os
import tempfile
import unittest

from utils.jetson_harness import benchmark_levels, energy_stats, preflight

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestBenchmarkHarnessCPU(unittest.TestCase):
    def test_smoke_run_without_power_sensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "r.json")
            cfg = os.path.join(ROOT, "utils", "jetson_harness", "configs", "smoke_cpu.json")
            benchmark_levels.main(["--config", cfg, "--out", out])
            data = json.load(open(out))
            meta = json.load(open(os.path.join(tmp, "r.meta.json")))
        self.assertEqual(list(data), ["Level 0", "Level 1", "Level 2"])
        self.assertEqual(len(data["Level 0"]["latency_ms"]), 3)
        self.assertEqual(data["Level 0"]["power_mW"], [None] * 3)
        self.assertIsNone(data["Level 0"]["energy_J"])
        self.assertFalse(meta["power_available"])

    def test_configs_are_valid_json_with_known_options(self):
        for name in os.listdir(os.path.join(ROOT, "utils", "jetson_harness", "configs")):
            with self.subTest(config=name):
                cfg = benchmark_levels.load_defaults(os.path.join(ROOT, "utils", "jetson_harness", "configs", name))
                args = benchmark_levels.parse_args(["--config", os.path.join(ROOT, "utils", "jetson_harness", "configs", name),
                                                    "--out", "unused.json"])
                for key, value in cfg.items():
                    self.assertEqual(getattr(args, key), value, key)


class TestPreflight(unittest.TestCase):
    def test_report_structure_off_jetson(self):
        report = preflight.collect()
        self.assertIn("checks", report)
        self.assertIn("software", report)
        names = {c["name"] for c in report["checks"]}
        self.assertIn("jtop power reading ('Power TOT')", names)
        json.dumps(report)   # serialisable


if __name__ == "__main__":
    unittest.main()
