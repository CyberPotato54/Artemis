"""Tests for tools/py/llama_search.py with a fake measure and a fake clock. No llama-server, no GPU, no network.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v
"""

import contextlib
import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("llama_search", HERE.parent / "llama_search.py")
m = importlib.util.module_from_spec(spec)
sys.modules["llama_search"] = m
spec.loader.exec_module(m)

SYSTEM = 0xFFFFFFFF
LOW, HIGH = 0x0000FFFF, 0xFFFF0000


def start_variant():
    return {"args": list(m.sl.BASE_ARGS), "affinity": SYSTEM, "label": "start"}


def value(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def table_measure(table, clock=None, cost=0.0, seen=None):
    """A measure that looks the variant's label up in `table` (default 50) and records what it was asked."""

    def measure(v):
        if seen is not None:
            seen.append(v)
        if clock is not None:
            clock.t += cost
        got = table.get(v["label"], 50.0)
        if isinstance(got, Exception):
            raise got
        return got

    return measure


class Axes(unittest.TestCase):
    def test_axes_order_and_the_cpu_masks_split_the_system_mask(self):
        axes = m.axes_for(SYSTEM)
        self.assertEqual([a for a, _ in axes], ["cpu", "batch", "kv", "flash_attn", "experts", "moe_cache", "ngl",
                                                "load_mode"])
        cpu = dict((p["label"], p) for p in axes[0][1])
        self.assertEqual(len(cpu), 8)
        self.assertEqual(cpu["cpu all default threads"]["affinity"], SYSTEM)
        self.assertEqual(cpu["cpu all -t 32"]["set"], {"-t": "32"})
        self.assertEqual(cpu["cpu low half -t 8"]["affinity"], LOW)
        self.assertEqual(cpu["cpu high half -t 16"]["affinity"], HIGH)

    def test_halves_come_from_the_system_mask_not_a_count(self):
        self.assertEqual(m.split_halves(0xFFFFFFFF), (0x0000FFFF, 0xFFFF0000))
        self.assertEqual(m.split_halves(0xFF), (0x0F, 0xF0))
        with self.assertRaises(SystemExit):
            m.split_halves(1)

    def test_experts_axis_has_the_six_counts_and_two_ot_forms(self):
        experts = dict(m.axes_for(SYSTEM))["experts"]
        self.assertEqual([p["set"].get("--n-cpu-moe") for p in experts[:6]], ["16", "20", "24", "28", "32", "40"])
        for p in experts[6:]:
            self.assertEqual(p["drop"], ["--n-cpu-moe"])
            self.assertTrue(p["set"]["-ot"].endswith("_exps\\.=CPU"))
        self.assertEqual(experts[6]["set"]["-ot"], r"blk\.(\d|1\d|2[0-3])\.ffn_(gate|up)_exps\.=CPU")
        self.assertEqual(experts[7]["set"]["-ot"], r"blk\.(\d|1\d|2\d|3[01])\.ffn_(gate|up)_exps\.=CPU")

    def test_ot_replaces_n_cpu_moe_and_back(self):
        ot = apply = m.apply_patch(start_variant(), dict(m.axes_for(SYSTEM))["experts"][6])
        self.assertNotIn("--n-cpu-moe", ot["args"])
        self.assertEqual(value(ot["args"], "-ot"), m.OT_24)
        back = m.apply_patch(apply, dict(m.axes_for(SYSTEM))["experts"][0])
        self.assertNotIn("-ot", back["args"])
        self.assertEqual(value(back["args"], "--n-cpu-moe"), "16")
        self.assertEqual(value(back["args"], "-c"), "32768")


class Search(unittest.TestCase):
    def test_axes_are_searched_in_order_keeping_the_best(self):
        seen = []
        table = {
            "start": 100.0,
            "cpu high half -t 16": 110.0,  # cpu winner (the affinity and -t carry on)
            "cpu low half -t 16": 105.0,
            "batch 512/512": 112.0,
            "kv f16": 111.0,  # no better than 112: the q8_0 cache stays
            "fa off": 90.0,
            "n-cpu-moe 20": 115.0,
            "ot gate+up of 24 layers on cpu": 118.0,  # beats n-cpu-moe 20
            "moe-cache 3000": 119.0,
            "ngl 36": 80.0,
            "load-mode absent": 119.5,
        }
        result = m.search(table_measure(table, seen=seen), start_variant(), m.axes_for(SYSTEM), Clock(), 1e9)
        best = result["best"]
        self.assertFalse(result["budget_hit"])
        self.assertEqual(best["tok_s"], 119.5)
        self.assertEqual(best["affinity"], HIGH)
        self.assertEqual(value(best["args"], "-t"), "16")
        self.assertEqual(value(best["args"], "-b"), "512")
        self.assertEqual(value(best["args"], "-ub"), "512")
        self.assertEqual(value(best["args"], "-ctk"), "q8_0")
        self.assertEqual(value(best["args"], "-fa"), "on")
        self.assertEqual(value(best["args"], "-ot"), m.OT_24)
        self.assertNotIn("--n-cpu-moe", best["args"])
        self.assertEqual(value(best["args"], "--moe-cache-mib"), "3000")
        self.assertNotIn("--load-mode", best["args"])
        # Each axis's values were tried on top of the earlier axes' winners.
        batch = [v for v in seen if v["label"].startswith("batch")]
        self.assertTrue(all(v["affinity"] == HIGH and value(v["args"], "-t") == "16" for v in batch))
        experts = [v for v in seen if v["label"].startswith(("n-cpu-moe", "ot "))]
        self.assertTrue(all(value(v["args"], "-b") == "512" for v in experts))
        # The order of the records is the order of the axes; the start was measured first, once.
        order = [v["axis"] for v in result["variants"]]
        self.assertEqual(order[0], "start")
        self.assertEqual(order, sorted(order, key=["start", "cpu", "batch", "kv", "flash_attn", "experts", "moe_cache",
                                                  "ngl", "load_mode"].index))
        self.assertEqual(len(result["variants"]), len({m.canon(v) for v in result["variants"]}))

    def test_a_variant_equal_to_the_best_is_not_run_again(self):
        seen = []
        m.search(table_measure({}, seen=seen), start_variant(), m.axes_for(SYSTEM), Clock(), 1e9)
        labels = [v["label"] for v in seen]
        for skipped in ("cpu all default threads", "batch 2048/2048", "kv q8_0", "fa on", "n-cpu-moe 24",
                        "moe-cache 1500", "ngl 99", "load-mode none"):
            self.assertNotIn(skipped, labels)

    def test_fa_off_runs_with_an_f16_cache(self):
        seen = []
        table = {"kv q8_0": 60.0, "kv f16": 40.0}
        m.search(table_measure(table, seen=seen), start_variant(), m.axes_for(SYSTEM), Clock(), 1e9)
        off = [v for v in seen if v["label"] == "fa off"]
        self.assertEqual(len(off), 1)
        self.assertEqual(value(off[0]["args"], "-fa"), "off")
        self.assertEqual(value(off[0]["args"], "-ctk"), "f16")
        self.assertEqual(value(off[0]["args"], "-ctv"), "f16")
        # Whatever the KV axis chose: f16 won it here, and the q8_0 start also gets f16 under -fa off.
        seen.clear()
        table = {"kv f16": 70.0}
        m.search(table_measure(table, seen=seen), start_variant(), m.axes_for(SYSTEM), Clock(), 1e9)
        off = [v for v in seen if v["label"] == "fa off"][0]
        self.assertEqual((value(off["args"], "-ctk"), value(off["args"], "-ctv")), ("f16", "f16"))
        on = [v for v in seen if v["label"] == "fa on"]
        self.assertEqual(on, [])  # -fa on is the best so far already

    def test_refused_and_no_fit_are_recorded_apart_and_the_search_goes_on(self):
        tail = "\n".join(f"line {i}" for i in range(20))
        table = {"batch 512/512": m.Refused(tail), "kv f16": m.NoFit(), "ngl 36": m.RunError("no timings"),
                 "moe-cache 3000": 77.0}
        result = m.search(table_measure(table), start_variant(), m.axes_for(SYSTEM), Clock(), 1e9)
        by = {v["label"]: v for v in result["variants"]}
        self.assertEqual(by["batch 512/512"]["outcome"], "refused")
        self.assertEqual(by["batch 512/512"]["stderr_tail"], tail)
        self.assertIsNone(by["batch 512/512"]["tok_s"])
        self.assertEqual(by["kv f16"]["outcome"], "no_fit")
        self.assertEqual(by["kv f16"]["stderr_tail"], "")
        self.assertEqual(by["ngl 36"]["outcome"], "error")
        self.assertEqual(by["ngl 36"]["stderr_tail"], "")
        self.assertEqual(by["ngl 36"]["error"], "no timings")
        self.assertEqual(by["batch 512/512"]["error"], "")
        # The variants after the failures still ran.
        self.assertIn("moe-cache 3000", by)
        self.assertEqual(by["moe-cache 3000"]["outcome"], "ok")
        self.assertEqual(result["best"]["tok_s"], 77.0)

    def test_an_unexpected_exception_is_an_error_outcome_and_on_record_sees_every_variant(self):
        snapshots = []
        table = {"kv f16": ValueError("boom"), "ngl 36": OSError("gone"), "moe-cache 3000": 77.0}
        result = m.search(table_measure(table), start_variant(), m.axes_for(SYSTEM), Clock(), 1e9,
                          on_record=lambda r: snapshots.append(len(r["variants"])))
        by = {v["label"]: v for v in result["variants"]}
        self.assertEqual(by["kv f16"]["outcome"], "error")
        self.assertEqual(by["kv f16"]["error"], "ValueError: boom")
        self.assertEqual(by["ngl 36"]["error"], "OSError: gone")
        self.assertEqual(by["moe-cache 3000"]["outcome"], "ok")
        self.assertEqual(snapshots, list(range(1, len(result["variants"]) + 1)))

    def test_too_slow_is_recorded(self):
        result = m.search(table_measure({"fa off": m.TooSlow()}), start_variant(), m.axes_for(SYSTEM), Clock(), 1e9)
        slow = [v for v in result["variants"] if v["label"] == "fa off"][0]
        self.assertEqual(slow["outcome"], "too_slow")
        self.assertIsNone(slow["tok_s"])
        self.assertIn("ngl 36", [v["label"] for v in result["variants"]])

    def test_a_failed_start_lets_the_first_success_lead(self):
        result = m.search(table_measure({"start": m.NoFit(), "cpu all -t 8": 60.0}), start_variant(),
                          m.axes_for(SYSTEM), Clock(), 1e9)
        self.assertEqual(result["variants"][0]["outcome"], "no_fit")
        self.assertIsNotNone(result["best"]["tok_s"])

    def test_budget_stops_the_search_and_keeps_the_best(self):
        clock = Clock()
        seen = []
        table = {"start": 100.0, "cpu all -t 8": 120.0}
        result = m.search(table_measure(table, clock=clock, cost=60.0, seen=seen), start_variant(), m.axes_for(SYSTEM),
                          clock, 150.0)
        self.assertTrue(result["budget_hit"])
        # Variants start at 0, 60 and 120 s; the one that would start at 180 s does not.
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(result["variants"]), 3)
        self.assertEqual(result["best"]["label"], "cpu all -t 8")
        self.assertEqual(result["best"]["tok_s"], 120.0)

    def test_budget_not_hit_when_the_axes_finish_in_time(self):
        result = m.search(table_measure({}), start_variant(), m.axes_for(SYSTEM), Clock(), 10.0)
        self.assertFalse(result["budget_hit"])


EXPECTED_PATCHES = [
    ("cpu", "cpu all default threads", {}, ["-t"], SYSTEM),
    ("cpu", "cpu all -t 8", {"-t": "8"}, [], SYSTEM),
    ("cpu", "cpu all -t 16", {"-t": "16"}, [], SYSTEM),
    ("cpu", "cpu all -t 32", {"-t": "32"}, [], SYSTEM),
    ("cpu", "cpu low half -t 8", {"-t": "8"}, [], LOW),
    ("cpu", "cpu low half -t 16", {"-t": "16"}, [], LOW),
    ("cpu", "cpu high half -t 8", {"-t": "8"}, [], HIGH),
    ("cpu", "cpu high half -t 16", {"-t": "16"}, [], HIGH),
    ("batch", "batch 2048/2048", {"-b": "2048", "-ub": "2048"}, [], None),
    ("batch", "batch 2048/512", {"-b": "2048", "-ub": "512"}, [], None),
    ("batch", "batch 512/512", {"-b": "512", "-ub": "512"}, [], None),
    ("kv", "kv q8_0", {"-ctk": "q8_0", "-ctv": "q8_0"}, [], None),
    ("kv", "kv f16", {"-ctk": "f16", "-ctv": "f16"}, [], None),
    ("flash_attn", "fa on", {"-fa": "on"}, [], None),
    ("flash_attn", "fa off", {"-fa": "off", "-ctk": "f16", "-ctv": "f16"}, [], None),
    *[("experts", f"n-cpu-moe {n}", {"--n-cpu-moe": str(n)}, ["-ot"], None) for n in (16, 20, 24, 28, 32, 40)],
    ("experts", "ot gate+up of 24 layers on cpu", {"-ot": m.OT_24}, ["--n-cpu-moe"], None),
    ("experts", "ot gate+up of 32 layers on cpu", {"-ot": m.OT_32}, ["--n-cpu-moe"], None),
    ("moe_cache", "moe-cache none", {}, ["--moe-cache-mib"], None),
    ("moe_cache", "moe-cache 1500", {"--moe-cache-mib": "1500"}, [], None),
    ("moe_cache", "moe-cache 3000", {"--moe-cache-mib": "3000"}, [], None),
    ("ngl", "ngl 99", {"-ngl": "99"}, [], None),
    ("ngl", "ngl 36", {"-ngl": "36"}, [], None),
    ("load_mode", "load-mode none", {"--load-mode": "none"}, [], None),
    ("load_mode", "load-mode absent", {}, ["--load-mode"], None),
]


class PatchTable(unittest.TestCase):
    def test_every_patch_matches_the_brief(self):
        got = [(axis, p["label"], p["set"], p["drop"], p["affinity"])
               for axis, patches in m.axes_for(SYSTEM) for p in patches]
        self.assertEqual(got, EXPECTED_PATCHES)
        self.assertEqual(m.OT_24, r"blk\.(\d|1\d|2[0-3])\.ffn_(gate|up)_exps\.=CPU")
        self.assertEqual(m.OT_32, r"blk\.(\d|1\d|2\d|3[01])\.ffn_(gate|up)_exps\.=CPU")


class Main(unittest.TestCase):
    def test_a_flag_missing_from_help_stops_main_before_any_server_is_made(self):
        import shutil
        import tempfile
        from unittest import mock

        made = []
        out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        done = type("Done", (), {"stdout": "-t, --threads", "stderr": ""})()
        patches = [
            mock.patch.object(m.sl, "port_in_use", return_value=False),
            mock.patch.object(m.sl, "list_devices", return_value=[("Vulkan1", "GPU X")]),
            mock.patch.object(m.sl, "system_affinity", return_value=SYSTEM),
            mock.patch.object(m.subprocess, "run", return_value=done),
            mock.patch.object(m, "make_measure", side_effect=lambda *a, **k: made.append(a)),
            mock.patch.object(m, "search", side_effect=lambda *a, **k: made.append(a)),
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with self.assertRaises(SystemExit) as e:
                m.main(["repo", "model", "bin", "--gpu-name", "GPU X", "--out", out])
        self.assertIn("--moe-cache-mib", str(e.exception.code))
        self.assertEqual(made, [])
        import json
        written = json.loads((Path(out) / "search.json").read_text(encoding="utf-8"))
        self.assertIn("--moe-cache-mib", written["error"])


class Flags(unittest.TestCase):
    HELP = "\n".join([
        "-t,    --threads N", "-b,    --batch-size N", "-ub,   --ubatch-size N", "-ctk,  --cache-type-k TYPE",
        "-ctv,  --cache-type-v TYPE", "-fa,   --flash-attn [on|off|auto]", "--n-cpu-moe N", "-ot,   --override-tensor",
        "-ngl,  --n-gpu-layers N", "--load-mode MODE", "--moe-cache-mib N"])

    def test_every_flag_listed_passes(self):
        m.check_flags(self.HELP, m.axes_for(SYSTEM))

    def test_unknown_flag_is_refused_before_any_run(self):
        help_text = self.HELP.replace("--moe-cache-mib N", "")
        with self.assertRaises(SystemExit) as e:
            m.check_flags(help_text, m.axes_for(SYSTEM))
        self.assertIn("--moe-cache-mib", str(e.exception))
        self.assertNotIn("--n-cpu-moe", str(e.exception))
        # Several missing flags are all named.
        with self.assertRaises(SystemExit) as e:
            m.check_flags("-t", m.axes_for(SYSTEM))
        self.assertIn("--load-mode", str(e.exception))
        self.assertIn("--n-cpu-moe", str(e.exception))

    def test_a_flag_that_only_prefixes_another_does_not_count(self):
        with self.assertRaises(SystemExit) as e:
            m.check_flags("-ngl2 --n-cpu-moe-extra", [("a", [m.patch("x", {"-ngl": "1", "--n-cpu-moe": "2"})])])
        self.assertIn("-ngl", str(e.exception))
        self.assertIn("--n-cpu-moe", str(e.exception))


class Finalists(unittest.TestCase):
    def test_finalists_are_the_top_two_distinct_plus_the_start(self):
        table = {"start": 100.0, "cpu all -t 8": 110.0, "cpu all -t 16": 108.0, "batch 512/512": 109.0}
        result = m.search(table_measure(table), start_variant(), m.axes_for(SYSTEM), Clock(), 1e9)
        picks = m.finalists(result, start_variant())
        self.assertEqual([p["label"] for p in picks][:2], ["cpu all -t 8", "batch 512/512"])
        self.assertEqual(picks[-1]["label"], "start")
        self.assertEqual(len(picks), 3)
        self.assertEqual(set(picks[0]), {"args", "affinity", "label"})

    def test_the_start_is_not_added_twice_when_it_is_among_the_top(self):
        table = {"start": 200.0}
        result = m.search(table_measure(table), start_variant(), m.axes_for(SYSTEM), Clock(), 1e9)
        picks = m.finalists(result, start_variant())
        self.assertEqual(len(picks), 2)
        self.assertEqual(picks[0]["label"], "start")
        self.assertEqual(sum(1 for p in picks if m.canon(p) == m.canon(start_variant())), 1)

    def test_failed_variants_are_not_finalists(self):
        result = {"variants": [
            {"axis": "a", "label": "x", "args": ["-t", "8"], "affinity": 1, "tok_s": None, "outcome": "refused",
             "stderr_tail": ""},
            {"axis": "a", "label": "y", "args": ["-t", "16"], "affinity": 1, "tok_s": 5.0, "outcome": "ok",
             "stderr_tail": ""}]}
        picks = m.finalists(result, {"args": ["-t", "4"], "affinity": 1, "label": "start"})
        self.assertEqual([p["label"] for p in picks], ["y", "start"])

    def test_pick_best_ignores_a_finalist_that_failed(self):
        done = [{"label": "a", "median": 90.0, "error": None}, {"label": "b", "median": 120.0, "error": "x"},
                {"label": "c", "median": None, "error": None}]
        self.assertEqual(m.pick_best(done)["label"], "a")
        self.assertIsNone(m.pick_best([done[1], done[2]]))


if __name__ == "__main__":
    unittest.main()
