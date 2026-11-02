"""Tests for tools/py/m6b_summary.py over synthetic records. Standard library only.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v

The module is loaded by path (python -I keeps the script's directory off sys.path). The records are shaped like the
runner's (tools/run-gpu-tests.ps1); the numbers are made up and hold no keys, hosts, prompts or replies.
"""

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "m6b_summary.py"
spec = importlib.util.spec_from_file_location("m6b_summary", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules["m6b_summary"] = m
spec.loader.exec_module(m)

GATE = "abc1234"
OTHER = "def5678"
DATE = "2026-11-10T12:00:00.0000000-07:00"
DEVICE = "AMD Radeon RX 9070 XT"
# The runner records a llama.cpp device as {id, name}.
RUNNER_DEVICE = {"id": "Vulkan0", "name": DEVICE}
HYGIENE = {"pcie_gen": 4, "adapter": DEVICE, "affinity": "0xffff0000", "desktop_gpu_mib": 640.0,
           "scanner_cpu_s": {"MsMpEng": 2400.0, "SearchIndexer": 2690.0}, "busy_pct": 3.2}
QUIET = {"MsMpEng": 1.5, "SearchIndexer": 0.25}  # the scanners' CPU seconds used during a block


def rec(test, commit=GATE, **fields):
    r = {"test": test, "commit": commit, "date": DATE}
    r.update(fields)
    return r


def speed_spec(on=137.3, off=109.6, prefill=2126.0, attempt=1, commit=GATE):
    return rec(
        "serve-speed-spec", commit, plan="5b", status="pass", attempt=attempt, on_median=on, bar=110.0,
        prefill=[
            {"which": "long", "on": prefill - 2.0, "off": prefill, "ratio": 0.999, "pass": True},
            {"which": "edit", "on": 1726.0, "off": 1728.0, "ratio": 0.999, "pass": True},
        ],
        off_bounds={"decode_median": off, "decode_bound": 46.5, "prefill_slowest": prefill, "prefill_bound": 1000.0,
                    "pass": True},
        problems=[],
    )


def first_run_runs(attempt=1, rate=135.0, commit=GATE):
    out = []
    for i, (p, seed) in enumerate([("write", 1234), ("edit", 1234), ("explain", 1234),
                                   ("write", 1235), ("edit", 1235), ("explain", 1235)]):
        out.append(rec(
            "first-run run", commit, input=f"{p}-{seed}-tuned", run=f"{p}:{seed}", mode="tuned", attempt=attempt,
            status="pass", affinity="0xffff0000", affinity_at_stop="0xffff0000", tuned_applied=True,
            figures={"decode_steps_tok_s": rate + i * 0.0, "completion_fnv": f"{i:016x}"},
        ))
    return out


def first_run(tuned=135.0, on=137.5, llama=120.0, attempt=1, status="pass", commit=GATE, **extra):
    r = rec(
        "first-run", commit, attempt=attempt, status=status, tuned_median=tuned, spec_on_median=on,
        tuned_over_hand=tuned / on, llama_median=llama, tuned_over_llama=tuned / llama, llama_device=DEVICE,
        llama_affinity="0xffff0000", llama_config_source="search", identity=[{"run": "write-1234", "equal": True}],
        hygiene=HYGIENE, link_dropped=False, problems=[],
    )
    r.update(extra)
    return r


def first_launch(attempt=1, seconds=410.0, commit=GATE):
    return rec(
        "first-run launch", commit, attempt=attempt, status="pass", first_launch_s=455.0,
        setup={"setup_done": True, "seconds": seconds, "limit_hit": False}, imports=["kernel32.dll", "ws2_32.dll"],
        network_imports=[],
    )


def llama_record(test="speed-llama (report only)", commit=GATE, device=RUNNER_DEVICE, **extra):
    r = rec(
        test, commit, status="ran", config_source="search", affinity="0xffff0000", affinity_at_stop="0xffff0000",
        affinity_asked="0xffff0000",
        runs=[{"prompt": "write", "seed": 1234, "decoded": 700, "tok_s": 120.0 + i} for i in range(6)],
        median=122.5, hygiene=HYGIENE, link_dropped=False,
    )
    if device is not None:
        r["device"] = device
    r.update(extra)
    return r


def passing_set(commit=GATE):
    recs = [
        rec("llama-search hygiene", commit, input="llama-search", status="pass", hygiene=HYGIENE),
        rec("llama-search (report only)", commit, status="ran", budget_minutes=45, budget_hit=False, variants=40,
            device=RUNNER_DEVICE,
            best={"args": ["-fa", "on", "-ub", "512"], "affinity": "0xffff0000", "median": 125.0,
                  "runs": [{"tok_s": 124.0 + i * 0.4} for i in range(6)]}),
        rec("llama-search link after", commit, input="llama-search", status="pass", pcie_gen=4, adapter=DEVICE,
            link_dropped=False, scanner_cpu_delta_s=QUIET),
        llama_record(commit=commit),
        llama_record("speed-llama-65k (report only)", commit, ctx=65536),
        speed_spec(commit=commit),
        rec("attn-depth run", commit, input="65000", status="ran", stage="after", base_tok_s=100.0, new_tok_s=130.0,
            ratio=1.3, fingerprints_equal=True),
        rec("attn-depth", commit, status="pass", stage="after", gain_65k=1.3, problems=[], link_dropped=False,
            depths=[{"depth": 65000, "ratio": 1.3, "fingerprints_equal": True}]),
        first_launch(commit=commit),
        *first_run_runs(commit=commit),
        first_run(commit=commit),
    ]
    return recs


class HoldTests(unittest.TestCase):
    def test_hold_passes_at_the_stated_figures(self):
        ok, problems = m.m5c_hold([speed_spec(on=137.3, off=109.6, prefill=2126.0)], GATE)
        self.assertTrue(ok, problems)
        self.assertEqual(problems, [])

    def test_hold_fails_on_a_low_on_median_and_names_it(self):
        ok, problems = m.m5c_hold([speed_spec(on=133.0)], GATE)
        self.assertFalse(ok)
        self.assertEqual(len(problems), 1)
        self.assertIn('"on" median', problems[0])
        self.assertIn("133", problems[0])
        self.assertIn("133.6", problems[0])

    def test_hold_fails_on_each_of_the_other_two_figures(self):
        ok, problems = m.m5c_hold([speed_spec(off=106.0)], GATE)
        self.assertFalse(ok)
        self.assertIn('"off" median', problems[0])
        ok, problems = m.m5c_hold([speed_spec(prefill=2050.0)], GATE)
        self.assertFalse(ok)
        self.assertIn("prefill", problems[0])

    def test_the_bars_are_inclusive(self):
        ok, problems = m.m5c_hold([speed_spec(on=133.6, off=106.7, prefill=2061.0)], GATE)
        self.assertTrue(ok, problems)

    def test_records_of_other_commits_are_left_out(self):
        # A passing record of another commit does not rescue a failing one of this commit, and the reverse.
        bad_here = speed_spec(on=130.0)
        good_there = speed_spec(on=140.0, commit=OTHER)
        ok, problems = m.m5c_hold([good_there, bad_here], GATE)
        self.assertFalse(ok)
        ok, problems = m.m5c_hold([speed_spec(on=140.0), speed_spec(on=100.0, commit=OTHER)], GATE)
        self.assertTrue(ok, problems)

    def test_no_record_for_the_commit_fails(self):
        ok, problems = m.m5c_hold([speed_spec(commit=OTHER)], GATE)
        self.assertFalse(ok)
        self.assertIn("no serve-speed-spec record", problems[0])

    def test_the_last_attempt_counts(self):
        first_misses = speed_spec(on=120.0, attempt=1)
        second_passes = speed_spec(on=137.0, attempt=2)
        self.assertTrue(m.m5c_hold([first_misses, second_passes], GATE)[0])
        self.assertTrue(m.m5c_hold([second_passes, first_misses], GATE)[0])  # file order does not decide
        self.assertFalse(m.m5c_hold([speed_spec(on=137.0, attempt=1), speed_spec(on=120.0, attempt=2)], GATE)[0])

    def test_on_a_tie_the_later_record_counts(self):
        self.assertTrue(m.m5c_hold([speed_spec(on=120.0), speed_spec(on=137.0)], GATE)[0])
        self.assertFalse(m.m5c_hold([speed_spec(on=137.0), speed_spec(on=120.0)], GATE)[0])

    def test_a_record_with_a_missing_figure_fails_naming_it(self):
        r = speed_spec()
        del r["on_median"]
        ok, problems = m.m5c_hold([r], GATE)
        self.assertFalse(ok)
        self.assertIn('"on" median', problems[0])
        self.assertIn("missing", problems[0])
        r = speed_spec()
        r["off_bounds"] = None
        r["prefill"] = []
        ok, problems = m.m5c_hold([r], GATE)
        self.assertFalse(ok)
        self.assertEqual(len(problems), 2)

    def test_a_commit_prefix_selects_the_records(self):
        ok, _ = m.m5c_hold([speed_spec(on=140.0)], "abc1234567")
        self.assertTrue(ok)

    def test_an_attempt_other_than_1_or_2_fails_the_hold_and_is_named(self):
        ok, problems = m.m5c_hold([speed_spec(on=137.0, attempt=1), speed_spec(on=140.0, attempt=3)], GATE)
        self.assertFalse(ok)
        self.assertTrue(any("attempt 3" in p and "1 or 2" in p for p in problems), problems)
        ok, problems = m.m5c_hold([speed_spec(on=140.0, attempt=3)], GATE)
        self.assertFalse(ok)
        self.assertTrue(any("no attempt 1 or 2" in p for p in problems), problems)

    def test_a_record_with_no_attempt_fails_the_hold(self):
        r = speed_spec(on=140.0)
        del r["attempt"]
        ok, problems = m.m5c_hold([r], GATE)
        self.assertFalse(ok)
        self.assertTrue(any("no attempt number" in p for p in problems), problems)

    def test_attempt_2_without_attempt_1_fails_the_hold(self):
        ok, problems = m.m5c_hold([speed_spec(on=140.0, attempt=2)], GATE)
        self.assertFalse(ok)
        self.assertTrue(any("attempt 2 without attempt 1" in p for p in problems), problems)

    def test_attempts_1_and_2_are_the_only_ones_that_count_even_beside_a_higher_one(self):
        recs = [speed_spec(on=120.0, attempt=1), speed_spec(on=140.0, attempt=2), speed_spec(on=100.0, attempt=3)]
        ok, problems = m.m5c_hold(recs, GATE)
        self.assertFalse(ok)  # the stray attempt 3 is named; the figures judged are attempt 2's
        self.assertFalse(any("below the bar" in p for p in problems), problems)


class SummaryTests(unittest.TestCase):
    def text(self, recs, gate=GATE):
        return m.summarize(recs, gate)

    def test_a_passing_set_prints_every_table_and_no_failure(self):
        t = self.text(passing_set())
        for heading in ("Speed hygiene and the link", "llama-search", "speed-llama (report only)",
                        "speed-llama-65k (report only)", "attn-depth", "first-run", "m5c hold"):
            self.assertIn(heading, t)
        self.assertNotIn("FAIL", t)
        self.assertNotIn("FLAG", t)
        self.assertIn("m5c holds.", t)
        self.assertIn("0.9818", t)  # 135.0 / 137.5, the tuned-over-hand ratio, recomputed from the medians
        self.assertIn(DEVICE, t)
        # the device is printed as `id: name`, not as a dict
        self.assertIn(f"Vulkan0: {DEVICE}", t)
        self.assertNotIn("{'id'", t)

    def test_a_device_recorded_as_a_plain_string_is_printed_as_it_is(self):
        recs = [r for r in passing_set() if r["test"] != "speed-llama (report only)"] + [llama_record(device="Old Card")]
        self.assertIn("Old Card", self.text(recs))

    def test_tuned_over_hand_of_0_96_is_printed_as_fail(self):
        recs = passing_set()
        recs = [r for r in recs if r["test"] != "first-run"] + [first_run(tuned=132.0, on=137.5, status="fail")]
        t = self.text(recs)
        row = next(line for line in t.splitlines() if line.startswith("| tuned / hand-tuned"))
        self.assertIn("0.9600", row)
        self.assertIn("FAIL", row)
        # the llama bar is met in the same record
        self.assertIn("pass", next(line for line in t.splitlines() if line.startswith("| tuned / llama.cpp")))

    def test_tuned_not_above_llama_is_printed_as_fail(self):
        recs = [r for r in passing_set() if r["test"] != "first-run"] + [first_run(tuned=135.0, llama=136.0)]
        row = next(line for line in self.text(recs).splitlines() if line.startswith("| tuned / llama.cpp"))
        self.assertIn("FAIL", row)

    def test_a_speed_llama_record_without_a_device_is_flagged(self):
        recs = [r for r in passing_set() if r["test"] != "speed-llama (report only)"] + [llama_record(device=None)]
        t = self.text(recs)
        self.assertIn("FLAG speed-llama: the record names no device", t)
        self.assertNotIn("FLAG speed-llama-65k", t)

    def test_a_llama_search_record_without_a_device_is_flagged(self):
        recs = [copy.deepcopy(r) for r in passing_set()]
        for r in recs:
            if r["test"] == "llama-search (report only)":
                del r["device"]
        self.assertIn("FLAG llama-search: the record names no device", self.text(recs))

    def test_records_of_other_commits_are_left_out_of_the_tables(self):
        other = [first_run(tuned=90.0, on=137.5, commit=OTHER, status="fail"), speed_spec(on=100.0, commit=OTHER),
                 llama_record(device={"id": "Vulkan1", "name": "Other Card"}, commit=OTHER)]
        t = self.text(passing_set() + other)
        self.assertNotIn("Other Card", t)
        self.assertNotIn("0.6545", t)
        self.assertIn(f"{OTHER} (3)", t)
        self.assertIn("m5c holds.", t)

    def test_every_first_run_attempt_is_printed_and_the_last_counts(self):
        recs = passing_set() + [first_run(tuned=132.0, attempt=2, status="fail")]
        t = self.text(recs)
        self.assertIn("2 attempt(s); the last counts.", t)
        self.assertIn("### attempt 1: status pass", t)
        self.assertIn("### attempt 2: status fail", t)

    def test_a_dropped_link_and_a_low_gen_are_flagged_in_the_hygiene_table(self):
        recs = passing_set()
        for r in recs:
            if r["test"] == "llama-search link after":
                r.update(status="fail", link_dropped=True, pcie_gen=3, problem="link dropped", why="gen 3")
        t = self.text(recs)
        self.assertIn("FLAG llama-search: link after: link dropped (gen 3)", t)

    def test_the_scanners_cpu_seconds_and_the_busy_percentage_are_printed(self):
        t = self.text(passing_set())
        row = next(line for line in t.splitlines() if line.startswith("| llama-search |"))
        self.assertIn("3.2", row)
        self.assertIn("1.5 / 0.2", row)  # Defender / indexer
        self.assertNotIn("FLAG", t)

    def test_scanners_over_30_cpu_seconds_are_flagged(self):
        recs = passing_set()
        for r in recs:
            if r["test"] == "llama-search link after":
                r["scanner_cpu_delta_s"] = {"MsMpEng": 25.0, "SearchIndexer": 6.0}
        t = self.text(recs)
        self.assertIn("FLAG llama-search: the background scanners used 31.0 CPU seconds during the block, above 30", t)

    def test_scanners_at_exactly_30_cpu_seconds_are_not_flagged(self):
        recs = passing_set()
        for r in recs:
            if r["test"] == "llama-search link after":
                r["scanner_cpu_delta_s"] = {"MsMpEng": 20.0, "SearchIndexer": 10.0}
        self.assertNotIn("FLAG", self.text(recs))

    def test_a_missing_scanner_reading_is_not_counted(self):
        recs = passing_set()
        for r in recs:
            if r["test"] == "llama-search link after":
                r["scanner_cpu_delta_s"] = {"MsMpEng": None, "SearchIndexer": 2.0}
        t = self.text(recs)
        self.assertNotIn("FLAG", t)
        self.assertIn("- / 2.0", t)

    def test_a_start_busier_than_10_percent_is_flagged(self):
        recs = [copy.deepcopy(r) for r in passing_set()]
        for r in recs:
            if r["test"] == "llama-search hygiene":
                r["hygiene"]["busy_pct"] = 10.1
        self.assertIn("FLAG llama-search: the PC was 10.1% busy before the block, above 10%", self.text(recs))

    def test_a_start_at_exactly_10_percent_is_not_flagged(self):
        recs = [copy.deepcopy(r) for r in passing_set()]
        for r in recs:
            if r["test"] == "llama-search hygiene":
                r["hygiene"]["busy_pct"] = 10.0
        self.assertNotIn("FLAG", self.text(recs))

    def test_records_from_before_the_quietness_fields_print_dashes_and_no_flag(self):
        recs = [copy.deepcopy(r) for r in passing_set()]
        for r in recs:
            r.get("hygiene", {}).pop("busy_pct", None)
            r.get("hygiene", {}).pop("scanner_cpu_s", None)
            r.pop("scanner_cpu_delta_s", None)
        t = self.text(recs)
        self.assertNotIn("FLAG", t)
        self.assertIn("- / -", t)

    def test_a_setup_over_the_limit_is_printed_as_fail(self):
        recs = [r for r in passing_set() if r["test"] != "first-run launch"] + [first_launch(seconds=640.0)]
        row = next(line for line in self.text(recs).splitlines() if line.startswith("| setup seconds"))
        self.assertIn("640.0", row)
        self.assertIn("FAIL", row)

    def test_a_network_import_is_printed_as_fail(self):
        recs = [r for r in passing_set() if r["test"] != "first-run launch"]
        recs.append(first_launch())
        recs[-1]["network_imports"] = ["winhttp.dll"]
        row = next(line for line in self.text(recs).splitlines() if line.startswith("| WinHTTP"))
        self.assertIn("winhttp.dll", row)
        self.assertIn("FAIL", row)

    def test_a_recomputed_median_that_differs_from_the_record_is_shown(self):
        recs = [r for r in passing_set() if r["test"] != "first-run"] + [first_run(tuned=130.0, on=137.5)]
        self.assertIn("DIFFERENT", self.text(recs))  # the six run records give 135.0, the aggregate says 130.0

    def test_a_tuned_run_whose_mask_changed_is_flagged(self):
        recs = passing_set()
        for r in recs:
            if r["test"] == "first-run run" and r["input"].startswith("edit-1235"):
                r["affinity_at_stop"] = "0x0000ffff"
        self.assertIn("FLAG runs whose mask changed", self.text(recs))


    def test_a_launch_of_another_attempt_is_not_taken_for_this_one(self):
        recs = [r for r in passing_set() if r["test"] != "first-run launch"]
        recs.append(first_launch(attempt=2, seconds=500.0))
        row = next(line for line in self.text(recs).splitlines() if line.startswith("| setup seconds"))
        self.assertNotIn("500.0", row)
        self.assertIn("FLAG first-run attempt 1: no first-run launch record", self.text(recs))

    def test_a_launch_without_a_setup_done_line_is_a_flag_and_a_fail(self):
        recs = [r for r in passing_set() if r["test"] != "first-run launch"]
        launch = first_launch()
        del launch["setup"]
        recs.append(launch)
        t = self.text(recs)
        row = next(line for line in t.splitlines() if line.startswith("| setup seconds"))
        self.assertIn("FAIL", row)
        self.assertIn("FLAG first-run attempt 1: the launch record has no setup_done line", t)

    def test_a_recorded_ratio_that_differs_from_the_medians_is_flagged(self):
        recs = [r for r in passing_set() if r["test"] != "first-run"]
        recs.append(first_run(tuned_over_hand=1.2))
        t = self.text(recs)
        self.assertIn("tuned / hand-tuned 1.2000 (DIFFERENT)", t)
        self.assertIn("FLAG first-run attempt 1: the recorded tuned / hand-tuned", t)
        self.assertIn("tuned / llama.cpp 1.1250 (equal)", self.text(recs))

    def test_two_attempts_of_one_block_each_keep_their_hygiene_row(self):
        recs = passing_set()
        recs.append(rec("first-run hygiene", input="first-run", status="pass", hygiene=HYGIENE))
        recs.append(rec("first-run link after", input="first-run", status="fail", pcie_gen=3, adapter=DEVICE,
                        link_dropped=True, problem="link dropped", why="gen 3"))
        recs.append(rec("first-run hygiene", input="first-run", status="pass", hygiene=HYGIENE))
        recs.append(rec("first-run link after", input="first-run", status="pass", pcie_gen=4, adapter=DEVICE,
                        link_dropped=False))
        t = self.text(recs)
        rows = [line for line in t.splitlines() if line.startswith("| first-run")]
        self.assertEqual(len(rows), 2, rows)
        self.assertIn("FLAG first-run (1 of 2): link after: link dropped (gen 3)", t)

    def test_stray_first_run_attempts_are_flagged(self):
        recs = passing_set() + [first_run(attempt=3, status="fail")]
        self.assertIn("first-run: attempt 3 is not 1 or 2", self.text(recs))


    def test_the_hand_bar_is_a_product_with_an_allowance_not_a_rounded_ratio(self):
        on = 140.0
        tuned = 0.97 * on
        self.assertLess(tuned / on, 0.97)  # the ratio form misses by one rounding step; the product does not
        recs = [r for r in passing_set() if r["test"] not in ("first-run", "first-run run")]
        recs += first_run_runs(rate=tuned)
        recs.append(first_run(tuned=tuned, on=on))
        row = next(line for line in self.text(recs).splitlines() if line.startswith("| tuned / hand-tuned"))
        self.assertIn("pass", row)
        recs[-1] = first_run(tuned=tuned * 0.999, on=on)
        row = next(line for line in self.text(recs).splitlines() if line.startswith("| tuned / hand-tuned"))
        self.assertIn("FAIL", row)

    def test_a_link_after_pairs_with_the_latest_passing_hygiene_not_a_failed_one(self):
        recs = [r for r in passing_set() if not r["test"].startswith("llama-search")]
        recs.append(rec("attn-depth hygiene", input="attn-depth", status="fail", problem="PCIe gen 3", hygiene=HYGIENE))
        recs.append(rec("attn-depth hygiene", input="attn-depth", status="pass", hygiene=HYGIENE))
        recs.append(rec("attn-depth link after", input="attn-depth", status="pass", pcie_gen=4, adapter=DEVICE,
                        link_dropped=False))
        t = self.text(recs)
        rows = [line for line in t.splitlines() if line.startswith("| attn-depth")]
        self.assertEqual(len(rows), 2, rows)
        self.assertIn("| fail |", rows[0])
        self.assertNotIn("no link-after", t)  # the failed hygiene has none by design, the passing one has its own
        self.assertIn("FLAG attn-depth (1 of 2): hygiene fail: PCIe gen 3", t)
        self.assertIn("| pass | 4 | False |", rows[1])

    def test_a_passing_hygiene_with_no_link_after_is_still_flagged(self):
        recs = [r for r in passing_set() if r["test"] != "llama-search link after"]
        self.assertIn("FLAG llama-search: no link-after record", self.text(recs))

    def test_a_first_run_precheck_is_flagged_even_with_no_aggregate(self):
        recs = [r for r in passing_set() if r["test"] not in ("first-run", "first-run run", "first-run launch")]
        recs.append(rec("first-run precheck", status="fail",
                        error="run serve-speed-spec and speed-llama in the same invocation"))
        t = self.text(recs)
        self.assertIn("FLAG first-run precheck failed (not an attempt): run serve-speed-spec and speed-llama", t)
        self.assertIn("no first-run record for this commit.", t)

    def test_the_first_run_table_shows_the_spec_status_and_both_contexts(self):
        recs = [r for r in passing_set() if r["test"] != "first-run"] + [first_run(spec_status="fail")]
        t = self.text(recs)
        row = next(line for line in t.splitlines() if line.startswith("| serve-speed-spec"))
        self.assertIn("137.50 (serve-speed-spec status fail)", row)
        lines = t.splitlines()
        self.assertTrue(any(x.startswith("| llama.cpp searched best median, 32,768 positions") for x in lines))
        self.assertTrue(any("Artemis 65,536 vs llama.cpp 32,768 positions" in line for line in t.splitlines()))
        self.assertIn("llama.cpp (32,768 positions; Artemis runs 65,536)", t)


class CliTests(unittest.TestCase):
    def run_cli(self, recs, *extra):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "results.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = m.main(["--evidence", d, *extra])
        return code, out.getvalue(), err.getvalue()

    def test_exit_0_when_the_hold_passes(self):
        code, out, _ = self.run_cli(passing_set(), "--gate", GATE)
        self.assertEqual(code, 0)
        self.assertIn("m5c holds.", out)

    def test_exit_1_when_the_hold_fails(self):
        recs = [r for r in passing_set() if r["test"] != "serve-speed-spec"] + [speed_spec(on=133.0)]
        code, out, _ = self.run_cli(recs, "--gate", GATE)
        self.assertEqual(code, 1)
        self.assertIn("FAIL m5c does not hold", out)

    def test_the_default_gate_is_the_commit_of_the_latest_record(self):
        older = rec("serve-speed-spec", OTHER, attempt=1, on_median=100.0, date="2026-11-01T12:00:00.0000000-07:00")
        code, out, _ = self.run_cli([older] + passing_set())
        self.assertEqual(code, 0)
        self.assertIn(f"commit {GATE}", out)

    def test_an_unknown_gate_exits_1(self):
        code, _, err = self.run_cli(passing_set(), "--gate", "0000000")
        self.assertEqual(code, 1)
        self.assertIn("no record of commit 0000000", err)

    def test_lines_outside_tables_stay_within_120_columns(self):
        recs = passing_set()
        recs[-1]["problems"] = ["x " * 100]
        _, out, _ = self.run_cli(recs, "--gate", GATE)
        for line in out.splitlines():
            if not line.startswith("|"):
                self.assertLessEqual(len(line), 120, line)


if __name__ == "__main__":
    unittest.main()
