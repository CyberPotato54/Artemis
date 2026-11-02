"""Tests for tools/py/gpu_link.py's link judgement (no ADL, no GPU): judge_link and the check mode's exit codes.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "gpu_link.py"
spec = importlib.util.spec_from_file_location("gpu_link", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules["gpu_link"] = m
spec.loader.exec_module(m)


class JudgeLink(unittest.TestCase):
    def test_gen4_is_ok(self):
        ok, gen, why = m.judge_link({40: 4})
        self.assertTrue(ok)
        self.assertEqual(gen, 4)
        self.assertIn("4", why)

    def test_gen5_is_ok(self):
        self.assertTrue(m.judge_link({40: 5})[0])

    def test_gen3_is_not_ok_and_says_how_to_retrain(self):
        ok, gen, why = m.judge_link({40: 3})
        self.assertFalse(ok)
        self.assertEqual(gen, 3)
        self.assertIn("gen 3", why)
        self.assertIn("pnputil /restart-device", why)

    def test_min_gen_is_a_parameter(self):
        self.assertTrue(m.judge_link({40: 3}, min_gen=3)[0])
        self.assertFalse(m.judge_link({40: 4}, min_gen=5)[0])

    def test_a_failed_query_is_unreadable(self):
        ok, gen, why = m.judge_link(None)
        self.assertFalse(ok)
        self.assertIsNone(gen)
        self.assertTrue(why)

    def test_a_sample_without_sensor_40_is_unreadable(self):
        ok, gen, why = m.judge_link({1: 3250})
        self.assertFalse(ok)
        self.assertIsNone(gen)
        self.assertIn("40", why)


class CheckMode(unittest.TestCase):
    def run_check(self, sample, argv=("check",)):
        """Run the check mode against a stub reader; returns (exit code, the JSON line it printed)."""
        import contextlib
        import io
        import json

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = m.check(list(argv), reader=lambda name: ("AMD Radeon RX 9070 XT", sample))
        return code, json.loads(out.getvalue().strip().splitlines()[-1])

    def test_ok_exits_0(self):
        code, j = self.run_check({40: 4})
        self.assertEqual(code, 0)
        self.assertEqual(j["pcie_gen"], 4)
        self.assertTrue(j["ok"])
        self.assertEqual(j["adapter"], "AMD Radeon RX 9070 XT")
        self.assertEqual(set(j), {"adapter", "pcie_gen", "ok", "why"})

    def test_below_min_gen_exits_1(self):
        code, j = self.run_check({40: 3})
        self.assertEqual(code, 1)
        self.assertFalse(j["ok"])
        self.assertEqual(j["pcie_gen"], 3)

    def test_min_gen_flag(self):
        code, _ = self.run_check({40: 3}, ("check", "--min-gen", "3"))
        self.assertEqual(code, 0)

    def test_unreadable_exits_2(self):
        code, j = self.run_check(None)
        self.assertEqual(code, 2)
        self.assertIsNone(j["pcie_gen"])
        self.assertFalse(j["ok"])

    def test_a_reader_that_cannot_find_the_adapter_exits_2(self):
        import contextlib
        import io
        import json

        def reader(name):
            raise m.LinkUnreadable(f"no adapter named like {name!r}")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = m.check(["check", "--name", "nope"], reader=reader)
        j = json.loads(out.getvalue().strip())
        self.assertEqual(code, 2)
        self.assertFalse(j["ok"])
        self.assertIn("nope", j["why"])


if __name__ == "__main__":
    unittest.main()
