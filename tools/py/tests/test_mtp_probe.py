"""Tests for tools/py/mtp_probe.py with a stub launcher and a stub server: no llama-server, no network.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "mtp_probe.py"
spec = importlib.util.spec_from_file_location("mtp_probe", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules["mtp_probe"] = m
spec.loader.exec_module(m)

SCALE = 10**12


def counts_for(rates):
    """The pooled counts llama-server would report for these depth rates: every step drafts N tokens, and the mean
    accepted per step is the sum of the prefix products."""
    out, prefix, total = {}, 1.0, 0.0
    for n, a in enumerate(rates, start=1):
        prefix *= a
        total += prefix
        out[n] = (n * SCALE, round(total * SCALE))
    return out


class FakeProc:
    def __init__(self, exits):
        self.exits = exits
        self.stopped = False

    def poll(self):
        return 1 if self.exits else None

    def terminate(self):
        self.stopped = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.stopped = True


def reply(draft_n=10, accepted=7, predicted_n=100, tok_s=90.0):
    return {"timings": {"draft_n": draft_n, "draft_n_accepted": accepted, "predicted_n": predicted_n,
                        "predicted_per_second": tok_s}}


JOBS = [("write-1234", {"x": 1}, 1234), ("edit-1234", {"x": 2}, 1234)]


class DepthRates(unittest.TestCase):
    def test_depth_rates_inverts_the_mean_accepted_length(self):
        rates = [0.7, 0.5, 0.4, 0.3]
        counts = counts_for(rates)
        got = [a / d for d, a in counts.values()]
        for want, have in zip([0.7, 0.525, 1.19 / 3, 1.232 / 4], got):
            self.assertAlmostEqual(want, have, places=9)
        for want, have in zip(rates, m.depth_rates(counts)):
            self.assertAlmostEqual(want, have, delta=1e-9)

    def test_depth_rates_clamp_to_0_and_1(self):
        # r1 = 0.5 and r2 = 0.9 would need a2 = (1.8 - 0.5) / 0.5 = 2.6; r2 = 0.1 needs a negative a2.
        high = m.depth_rates({1: (100, 50), 2: (200, 180)})
        self.assertEqual(high, [0.5, 1.0])
        low = m.depth_rates({1: (100, 50), 2: (200, 20)})
        self.assertEqual(low, [0.5, 0.0])
        # A zero prefix leaves the deeper rates at 0 rather than dividing by it.
        self.assertEqual(m.depth_rates({1: (100, 0), 2: (200, 10), 3: (300, 10)}), [0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            m.depth_rates({1: (0, 0)})
        with self.assertRaises(ValueError):
            m.depth_rates({2: (10, 5)})


class Probe(unittest.TestCase):
    def run_probe(self, post, launcher, n_maxes=(1,)):
        return m.probe(JOBS, n_maxes, launcher, lambda proc: not proc.exits, post, lambda: False)

    def test_a_reply_without_draft_counts_is_an_error(self):
        replies = iter([reply(), {"timings": {"predicted_n": 100, "predicted_per_second": 90.0}}])
        out = self.run_probe(lambda body: next(replies), lambda n, moe: FakeProc(False))
        self.assertIn("error", out)
        self.assertIn("draft_n", out["error"])
        self.assertIn("edit-1234", out["error"])
        # The run before it is kept, counts and all.
        self.assertEqual(len(out["runs"]), 1)
        self.assertEqual(out["runs"][0]["draft_n_accepted"], 7)
        self.assertNotIn("depth_rates", out)
        with self.assertRaises(m.ProbeError):
            m.counts_of({})

    def test_a_request_that_raises_keeps_the_runs_so_far(self):
        import json
        import urllib.error

        for exc in (urllib.error.HTTPError("u", 500, "boom", {}, None), TimeoutError("t"),
                    ConnectionRefusedError("c"), json.JSONDecodeError("bad", "x", 0)):
            calls = iter([reply()])

            def post(body, exc=exc):
                try:
                    return next(calls)
                except StopIteration:
                    raise exc

            out = self.run_probe(post, lambda n, moe: FakeProc(False))
            self.assertEqual(len(out["runs"]), 1, exc)
            self.assertIn("request failed", out["error"])
            self.assertIn("edit-1234", out["error"])

    def test_depth_1_is_pooled_per_prompt_over_both_seeds(self):
        jobs = [("write-1234", {}, 1234), ("write-1235", {}, 1235), ("edit-1234", {}, 1234)]
        replies = iter([reply(10, 5), reply(10, 7), reply(10, 9)])
        out = m.probe(jobs, (1,), lambda n, moe: FakeProc(False), lambda p: True, lambda b: next(replies),
                      lambda: False)
        self.assertEqual(out["depth1_per_prompt"], {"write": 0.6, "edit": 0.9})

    def test_a_failed_load_retries_with_more_cpu_moe(self):
        tried, procs = [], []

        def launcher(n_max, moe):
            tried.append(moe)
            procs.append(FakeProc(exits=moe < 40))
            return procs[-1]

        out = self.run_probe(lambda body: reply(), launcher)
        self.assertEqual(tried, [24, 32, 40])
        self.assertNotIn("error", out)
        self.assertTrue(all(p.stopped for p in procs))
        self.assertEqual({r["n_cpu_moe"] for r in out["runs"]}, {40})
        # Nothing loads: a refusal that names every value.
        out = self.run_probe(lambda body: reply(), lambda n, moe: FakeProc(True))
        self.assertIn("24, 32, 40", out["error"])
        self.assertEqual(out["runs"], [])

    def test_the_pooled_counts_give_the_depth_rates(self):
        rates = [0.7, 0.5, 0.4, 0.3]
        counts = counts_for(rates)
        seen = []

        def launcher(n_max, moe):
            seen.append(n_max)
            return FakeProc(False)

        want = iter([c for n in (1, 2, 3, 4) for c in [counts[n]] * len(JOBS)])
        out = self.run_probe(lambda body: reply(*[x // len(JOBS) for x in next(want)]), launcher, (1, 2, 3, 4))
        self.assertEqual(seen, [1, 2, 3, 4])
        self.assertEqual(len(out["runs"]), 8)
        for want_rate, have in zip(rates, out["depth_rates"]):
            self.assertAlmostEqual(want_rate, have, delta=1e-9)
        self.assertEqual(set(out["depth1_per_prompt"]), {"write", "edit"})
        self.assertEqual(out["pooled"]["1"]["draft_n"], counts[1][0])

    def test_a_busy_port_is_refused_before_a_server_starts(self):
        started = []
        out = m.probe(JOBS, (1,), lambda n, moe: started.append(moe), lambda p: True, lambda b: reply(), lambda: True)
        self.assertIn("8095", out["error"])
        self.assertEqual(started, [])

    def test_the_server_args_carry_mtp_and_the_moe_value(self):
        args = m.server_args("Vulkan1", 3, 32)
        self.assertEqual(args[:2], ["--device", "Vulkan1"])
        self.assertEqual(args.count("--device"), 1)
        self.assertNotIn("Vulkan0", args)
        self.assertEqual(args[args.index("--n-cpu-moe") + 1], "32")
        self.assertEqual(args[-6:], ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-p-min", "0"])
        self.assertNotIn("8080", args)
        self.assertEqual(m.PORT, 8095)


class Module(unittest.TestCase):
    def test_the_module_imports_without_running_main(self):
        # Loading the module above started nothing: it only defines main.
        self.assertTrue(callable(m.main))
        self.assertEqual(m.__name__, "mtp_probe")


if __name__ == "__main__":
    unittest.main()
