"""Tests for tools/py/m4a_summary.py over small synthetic records written to a temp dir. Standard library only.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v

The module is loaded by path (python -I keeps the script's directory off sys.path). The fixtures copy the shapes of
the milestone 4 results.jsonl; they hold no keys, hosts or request text.
"""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "m4a_summary.py"
spec = importlib.util.spec_from_file_location("m4a_summary", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules["m4a_summary"] = m
spec.loader.exec_module(m)

# OLD sorts after NEW as a string, so a gate chosen by commit order or by file order alone would pick the wrong one.
OLD = "9999999"
NEW = "2222222"
# OLD's instant is earlier although its string sorts later (the offsets differ): +14:00 is 11:00 UTC on 27 Oct,
# -06:00 is 02:00 UTC on 28 Oct.
OLD_DATE = "2026-10-28T01:00:00.0000000+14:00"
NEW_DATE = "2026-10-27T20:00:00.0000000-06:00"
SHA_A = "a" * 64
SHA_B = "b" * 64
TEMPLATE = "c" * 64
OVR = {"log_dir": "D:\\runs\\x\\logs", "model": "M.gguf", "api_key_file": "D:\\keys\\k.txt", "extra": [], "port": 8090}


def rec(test, commit, inp=None, sha=SHA_A, **kw):
    r = {"test": test, "commit": commit, "date": NEW_DATE, "status": "pass"}
    if inp is not None:
        r["input"] = inp
    if sha:
        r["artemis_sha256"] = sha
    r.update(kw)
    return r


def stall(commit):
    return rec("serve-stall", commit, sha=None, template_sha256=TEMPLATE, config_sha256="d" * 64, exit_code=0,
               stall_lines=1, problems=[], port_free=True,
               figures={"load_s": 13.0, "stall_s": 120, "quiet_s": 120.2, "t_end_minus_t_last_s": 122.9,
                        "content_chunks": 15, "exit_code": 4, "problem": None})


def speed_run(commit, name, steps, misses, tokens=100, slots=133):
    return rec("serve-speed run", commit, name, status="ran", slots_per_layer=slots, desktop_gpu_mib=559.3,
               checks=[{"name": "answers", "pass": True, "detail": "200"}],
               figures={"prompt": name, "prompt_tokens": 44, "completion_tokens": tokens + 1, "decode_tokens": tokens,
                        "decode_steps_tok_s": steps, "decode_tok_s": steps + 0.1, "prefill_tok_s": 150.0,
                        "decode_misses": misses, "miss_mib_per_token": misses / tokens * 1.8})


def speed_set(commit, tool_decode=80.0):
    runs = [speed_run(commit, "write-1", 100.0, 2500), speed_run(commit, "edit-1", 90.0, 3000),
            speed_run(commit, "explain-1", 80.0, 2000)]
    agg_runs = [{"run": "write-1", "decode_steps_tok_s": 100.0, "long": False, "prefill_tok_s": 150.0},
                {"run": "edit-1", "decode_steps_tok_s": 90.0, "long": False, "prefill_tok_s": 1700.0},
                {"run": "explain-1", "decode_steps_tok_s": 80.0, "long": False, "prefill_tok_s": 120.0},
                {"run": "long-1", "decode_steps_tok_s": None, "long": True, "prefill_tok_s": 2100.0},
                {"run": "long-2", "decode_steps_tok_s": None, "long": True, "prefill_tok_s": 2000.0}]
    agg = rec("serve-speed", commit, decode_bound=46.5, prefill_bound=1000, decode_median=90.0, prefill_slowest=2000.0,
              long_prompt_tokens=27416, long_prompt_reference=27416, problems=[], runs=agg_runs,
              desktop_gpu_mib=[{"run": "write-1", "mib": 559.3}])
    tool = rec("speed", commit, sha=None, decode_bound=46.5, prefill_bound=1000.0, decode_median=tool_decode,
               prefill_slowest=2200.0, gated=False, problems=[], warnings=[],
               base={"decoded": 699, "seconds": 6.0, "tok_s": 100.0, "ms_per_token": 10.0, "slots": 133},
               decode=[{"prompt": "write", "seed": 1, "tok_s": tool_decode, "from_vram_pct": 91.0, "slots": 133,
                        "held_back_mib": 887.5}],
               prefill=[{"tokens": 27416, "tok_s": 2200.0, "from_vram_pct": 100.0, "slots": 133, "held_back_mib": 887.5}],
               desktop_gpu_mib=[{"run": "write-1", "mib": 559.3}])
    return runs + [agg, tool]


def mixed(commit, a, b):
    return rec("serve-speed mixed (report only)", commit, "mixed", status="ran", slots_per_layer=133, exit_code=0,
               graceful=True, stall_lines=0, desktop_gpu_mib=559.3, checks=[{"name": "runs_answer", "pass": True, "detail": "ok"}],
               figures={"skipped": [], "runs": [{"prompt": "write", "seed": 1, "status": 200, "prompt_tokens": 44,
                                                 "completion_tokens": 70, "prefill_tok_s": 150.0, "prefilled_tokens": 44,
                                                 "reused_tokens": 0, "restored": "none"}],
                        "decode_tok_s": a, "decode_steps_tok_s": b, "decode_median_tok_s": 93.0,
                        "artemis_3a_decode_tok_s": 95.69, "milestone2_decode_tok_s": 95.57,
                        "decode_median_vs_3a_pct": -2.8})


def case(commit, name, figures=None):
    return rec("7", commit, name, checks=[{"name": "c1", "pass": True, "detail": "fine"}], overrides=dict(OVR),
               template_sha256=TEMPLATE, config_sha256="e" * 64, stall_lines=0, figures=figures or {"n": 3})


def full_set(commit, sha=SHA_A, date=None):
    recs = [stall(commit)] + speed_set(commit) + [mixed(commit, [90.0, 95.0, 100.0], [89.0, 94.0])]
    recs += [case(commit, "chat-tools", {"finish_reasons": ["tool_calls", "stop"], "requests": 2})]
    recs += [rec("serve stop", commit, "prod-check", graceful=True, exit_code=0, exit_line=True, port_free=True,
                 stall_lines=0, seconds=1.0, overrides=dict(OVR), template_sha256=TEMPLATE, config_sha256="e" * 64)]
    for r in recs:
        if "artemis_sha256" in r:
            r["artemis_sha256"] = sha
        r["date"] = date or (OLD_DATE if commit == OLD else NEW_DATE)
    return recs


def flat(text):
    """The output with each wrapped line joined, for assertions on prose."""
    return " ".join(text.split())


class Run:
    """Writes the records and runs main over them; holds the output."""

    def __init__(self, tc, recs, *args, files=None):
        self.tmp = tempfile.TemporaryDirectory()
        tc.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        with open(self.dir / "results.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        for name, v in (files or {}).items():
            (self.dir / name).write_text(json.dumps(v), encoding="utf-8")
        o, e = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
            self.code = m.main(["--dir", str(self.dir), *args])
        self.out, self.err = o.getvalue(), e.getvalue()


class GateChoice(unittest.TestCase):
    def two_gates(self):
        return full_set(OLD, SHA_A) + full_set(NEW, SHA_B)

    def test_default_is_the_commit_of_the_latest_dated_record(self):
        # The old commit's strings sort later and one of its records comes last in the file, but its date is earlier.
        r = Run(self, full_set(NEW, SHA_B) + full_set(OLD, SHA_A))
        self.assertEqual(r.code, 0, r.err)
        self.assertIn(f"Gate commit: {NEW} ({len(full_set(NEW))} records used", r.out)
        self.assertIn(f"Superseded: commit {OLD}, {len(full_set(OLD))} records", r.out)
        # Only the newer commit's binary is reported: one value, and it is the newer one.
        self.assertIn(f"artemis_sha256: one value over", r.out)
        self.assertIn(SHA_B, r.out)
        self.assertNotIn(SHA_A, r.out)

    def test_without_dates_the_last_record_in_the_file_decides(self):
        recs = full_set(NEW, SHA_B) + full_set(OLD, SHA_A)
        for x in recs:
            del x["date"]
        r = Run(self, recs)
        self.assertIn(f"Gate commit: {OLD} ", r.out)

    def test_gate_option_overrides_and_accepts_a_prefix(self):
        r = Run(self, self.two_gates(), "--gate", OLD)
        self.assertEqual(r.code, 0, r.err)
        self.assertIn(f"Gate commit: {OLD} ", r.out)
        self.assertIn(f"Superseded: commit {NEW}", r.out)
        self.assertIn(SHA_A, r.out)
        self.assertNotIn(SHA_B, r.out)

    def test_gate_prefix_matches_the_full_commit(self):
        full = NEW + "abcdef"
        r = Run(self, full_set(OLD) + full_set(full), "--gate", NEW)
        self.assertIn(f"Gate commit: {full} ", r.out)
        self.assertIn(f"Superseded: commit {OLD}", r.out)

    def test_unknown_gate_is_an_error(self):
        r = Run(self, full_set(OLD), "--gate", "8888888")
        self.assertEqual(r.code, 1)
        self.assertIn("no record of commit 8888888", r.err)
        self.assertEqual(r.out, "")

    def test_a_record_without_a_commit_does_not_pick_the_gate(self):
        recs = full_set(OLD) + [{"test": "m4-soak", "status": "pass", "date": "2026-11-03T00:00:00Z"}]
        r = Run(self, recs)
        self.assertIn(f"Gate commit: {OLD} ", r.out)
        self.assertIn("1 records carry no commit field", r.out)


class Hashes(unittest.TestCase):
    def test_two_different_binaries_are_flagged_loudly(self):
        recs = full_set(NEW)
        recs[2]["artemis_sha256"] = SHA_B  # one serve-speed run measured another binary
        r = Run(self, recs)
        self.assertIn("!!! 2 DIFFERENT artemis_sha256 VALUES", r.out)
        self.assertIn(SHA_A, r.out)
        self.assertIn(SHA_B, r.out)

    def test_one_binary_and_the_stall_note(self):
        r = Run(self, full_set(NEW))
        self.assertNotIn("!!!", r.out)
        self.assertIn("The serve-stall record has no artemis_sha256 by design", r.out)
        self.assertIn(f"Template SHA-256: one value over", r.out)
        self.assertIn(TEMPLATE, r.out)

    def test_two_templates_are_listed(self):
        recs = full_set(NEW)
        recs[-1]["template_sha256"] = "f" * 64
        r = Run(self, recs)
        self.assertIn("!!! 2 DIFFERENT TEMPLATE SHA-256 VALUES", r.out)
        self.assertIn("f" * 64, r.out)

    def test_a_server_run_without_a_binary_hash_is_flagged(self):
        recs = full_set(NEW)
        del recs[-1]["artemis_sha256"]
        r = Run(self, recs)
        self.assertIn("server-run records other than serve-stall have no artemis_sha256", r.out)


class Figures(unittest.TestCase):
    def test_stall_figures_and_limit(self):
        r = Run(self, full_set(NEW))
        self.assertIn("| stall_s (the configured stall) | 120 |", r.out)
        self.assertIn("| t_end - t_last (limit stall_s + 10 = 130.00) | 122.900 |", r.out)
        self.assertIn("| t_end - t_last within the limit | True |", r.out)
        self.assertIn("| server exit code (figures.exit_code) | 4 |", r.out)
        recs = full_set(NEW)
        recs[0]["figures"]["t_end_minus_t_last_s"] = 131.0
        self.assertIn("| t_end - t_last within the limit | False |", Run(self, recs).out)

    def test_speed_against_bounds_and_the_tool(self):
        r = Run(self, full_set(NEW))
        self.assertIn("| decode median (tok/s) | 90.00 | 80.00 | 1.125 | 46.5 | True |", r.out)
        self.assertIn("| slower long-prompt prefill (tok/s) | 2000.00 | 2200.00 | 0.909 | 1000.0 | True |", r.out)
        self.assertIn("| server decode median / tool base (tok/s) | 90.00 | 100.00 | 0.900 | - | - |", r.out)
        self.assertIn("(the record says 90.0000: equal)", r.out)
        # Misses per token: 2500/100, 3000/100, 2000/100.
        self.assertIn("| write-1 | 2500 | 25.00 | 45.00 | 133 | 559.3 |", r.out)
        self.assertIn("Server misses per token over the decode runs: n=3 min 20.00 median 25.00 max 30.00", r.out)
        self.assertIn("the tool's record has no misses-per-token figure", r.out)

    def test_a_server_below_a_bound_is_reported_so(self):
        recs = full_set(NEW)
        for x in recs:
            if x.get("test") == "serve-speed":
                x["runs"][3]["prefill_tok_s"] = 900.0
        r = Run(self, recs)
        self.assertIn("| slower long-prompt prefill (tok/s) | 900.00 | 2200.00 | 0.409 | 1000.0 | False |", r.out)

    def test_mixed_lists_are_summarised_separately_never_paired(self):
        # The two lists differ in length (3 and 2): a zip would drop or crash; each gets its own summary.
        r = Run(self, full_set(NEW))
        self.assertEqual(r.code, 0, r.err)
        self.assertIn("decode_tok_s list (completion tokens over decode time): n=3 min 90.00 median 95.00 max 100.00", r.out)
        self.assertIn("decode_steps_tok_s list (decode tokens over decode time): n=2 min 89.00 median 91.50 max 94.00", r.out)
        self.assertIn("never paired by index", r.out)

    def test_case_figures_and_checks_are_printed(self):
        r = Run(self, full_set(NEW))
        self.assertIn("**chat-tools** (pass)", r.out)
        self.assertIn("| c1 | True | fine |", r.out)
        self.assertIn('finish_reasons=["tool_calls", "stop"]', r.out)

    def test_overrides_common_keys_once(self):
        r = Run(self, full_set(NEW))
        self.assertIn('- model (every block): "M.gguf"', r.out)
        self.assertIn("- port (every block): 8090", r.out)

    def test_no_line_is_wider_than_120_columns_except_tables(self):
        r = Run(self, full_set(OLD) + full_set(NEW))
        for line in r.out.splitlines():
            if not line.startswith("|"):
                self.assertLessEqual(len(line), 120, line)


class Robustness(unittest.TestCase):
    def test_null_slots_and_null_memory_figures_do_not_crash(self):
        recs = full_set(NEW)
        for x in recs:
            if x.get("test") == "serve-speed run":
                x["slots_per_layer"] = None
                x["desktop_gpu_mib"] = None
            if x.get("test") == "speed":
                x["decode"][0]["slots"] = None
                x["decode"][0]["held_back_mib"] = None
                x["prefill"][0]["slots"] = None
                x["prefill"][0]["held_back_mib"] = None
                x["desktop_gpu_mib"] = [{"run": "write-1", "mib": None}]
            if x.get("test") == "serve-speed":
                x["desktop_gpu_mib"] = [{"run": "write-1", "mib": None}]
        recs[3]["slots_per_layer"] = 133  # one run recorded it, the others did not
        r = Run(self, recs)
        self.assertEqual(r.code, 0, r.err)
        self.assertIn("Engine tool slots per layer none recorded; held back from the budget at load none recorded MiB; "
                      "desktop GPU memory in use at each run none recorded MiB.", flat(r.out))
        self.assertIn("Server slots per layer [133]; desktop GPU memory in use at each run none recorded MiB.", r.out)

    def test_the_last_of_a_duplicated_block_is_used_and_the_duplicate_is_flagged(self):
        recs = full_set(NEW)
        again = speed_run(NEW, "write-1", 50.0, 1000)  # the block was rerun at the same commit
        again["artemis_sha256"] = SHA_A
        again["date"] = NEW_DATE
        r = Run(self, recs + [again])
        self.assertEqual(r.code, 0, r.err)
        self.assertIn(f"!!! 2 records of serve-speed run/write-1 at {NEW}: the last is used !!!", r.out)
        self.assertIn("| write-1 | 44 | 101 | 100 | 50.00 |", r.out)
        self.assertNotIn("| write-1 | 44 | 101 | 100 | 100.00 |", r.out)
        self.assertIn(f"({len(recs)} records used", r.out)
        self.assertNotIn("!!!", Run(self, recs).out)

    def test_a_malformed_results_line_is_skipped_and_named(self):
        r = Run(self, full_set(NEW))
        text = (r.dir / "results.jsonl").read_text(encoding="utf-8")
        (r.dir / "results.jsonl").write_text('{broken\n' + text + '[1, 2]\n', encoding="utf-8")
        o = io.StringIO()
        with contextlib.redirect_stdout(o):
            self.assertEqual(m.main(["--dir", str(r.dir)]), 0)
        n = len(full_set(NEW)) + 2
        self.assertIn("Line 1 of results.jsonl skipped: not valid JSON.", o.getvalue())
        self.assertIn(f"Line {n} of results.jsonl skipped: not a JSON object.", o.getvalue())
        self.assertIn(f"({len(full_set(NEW))} records used", o.getvalue())

    def test_a_smoke_file_that_is_not_an_object_is_unreadable(self):
        r = Run(self, full_set(NEW), files={"smoke-opencode.json": [1, 2], "smoke-claude-code.json": "x"})
        self.assertIn("smoke-opencode.json: present but unreadable", r.out)
        self.assertIn("smoke-claude-code.json: present but unreadable", r.out)

    def test_the_tool_pairing_rests_on_order_not_a_hash(self):
        r = Run(self, full_set(NEW))
        self.assertIn("speed record carries no artemis_sha256", r.out)

    def test_a_decode_median_below_its_bound_is_reported_so(self):
        recs = full_set(NEW)
        for x in recs:
            if x.get("test") == "serve-speed":
                for run in x["runs"]:
                    if run["decode_steps_tok_s"] is not None:
                        run["decode_steps_tok_s"] = 40.0
        r = Run(self, recs)
        self.assertIn("| decode median (tok/s) | 40.00 | 80.00 | 0.500 | 46.5 | False |", r.out)

    def test_the_swapped_ratio_would_fail_this_fixture(self):
        # server 90 over tool 80 is 1.125; tool over server would be 0.889.
        r = Run(self, full_set(NEW))
        self.assertIn("1.125", r.out)
        self.assertNotIn("0.889", r.out)


class Missing(unittest.TestCase):
    def test_missing_blocks_print_missing_lines_and_do_not_crash(self):
        r = Run(self, [stall(NEW)])
        self.assertEqual(r.code, 0, r.err)
        for what in ("serve-speed: missing", "speed (the engine tool): missing", "serve-speed run records: missing",
                     "serve-speed mixed (report only): missing", "test 7 cases: missing", "serve stop: missing"):
            self.assertIn(f"**{what}** (no such record at commit {NEW}).", r.out)
        self.assertIn("### The stall test", r.out)

    def test_a_missing_stall_record_prints_a_missing_line(self):
        r = Run(self, speed_set(NEW))
        self.assertEqual(r.code, 0, r.err)
        self.assertIn(f"**serve-stall: missing** (no such record at commit {NEW}).", r.out)
        self.assertIn("| decode median (tok/s) |", r.out)

    def test_missing_smokes_say_not_yet_run(self):
        r = Run(self, full_set(NEW))
        self.assertIn("smoke-opencode.json: not yet run.", r.out)
        self.assertIn("smoke-claude-code.json: not yet run.", r.out)

    def test_present_smoke_is_printed_and_the_other_is_not_yet_run(self):
        smoke = {"client": "opencode", "version": "2.0.30", "status": "pass", "reasons": [], "launch_utc": "2026-11-01T00:00:00Z",
                 "attempts": [{"seconds": 61.5, "tests_pass": True, "client_exit": 0}],
                 "wire": {"requests": 12, "tool_calls": 4, "malformed_calls": 0}, "server": {"graceful": True, "exit_code": 0}}
        r = Run(self, full_set(NEW), files={"smoke-opencode.json": smoke})
        self.assertIn("smoke-opencode.json: client opencode 2.0.30, status pass", r.out)
        self.assertIn("attempt 1: seconds 61.5, tests_pass True, client_exit 0", r.out)
        self.assertIn("smoke-claude-code.json: not yet run.", r.out)

    def test_a_broken_smoke_file_does_not_crash(self):
        r = Run(self, full_set(NEW))
        (r.dir / "smoke-opencode.json").write_text("{not json", encoding="utf-8")
        o = io.StringIO()
        with contextlib.redirect_stdout(o):
            self.assertEqual(m.main(["--dir", str(r.dir)]), 0)
        self.assertIn("smoke-opencode.json: present but unreadable", o.getvalue())


class Reuse(unittest.TestCase):
    REPORT = {"overall": {"requests": 4, "prompt_tokens": 100, "reused_tokens": 60, "rate": 0.6},
              "by_source": {"live": {"requests": 4, "prompt_tokens": 100, "reused_tokens": 60, "rate": 0.6}},
              "by_cause": {"none": {"requests": 4, "prompt_tokens": 100, "reused_tokens": 60, "rate": 0.6}},
              "budgets": {"1024": {"requests": 4, "closed": 1}}}

    def test_default_path_in_the_evidence_dir(self):
        r = Run(self, full_set(NEW), files={"reuse-report.json": self.REPORT})
        self.assertIn("4 requests, 60 of 100 prompt tokens reused = 0.600", r.out)
        self.assertIn("| live | 4 | 100 | 60 | 0.600 |", r.out)

    def test_without_a_file_the_results_record_is_the_source(self):
        recs = full_set(NEW)
        recs.append(rec("reuse-report (report only)", NEW, "prod-check", status="ran", **self.REPORT))
        r = Run(self, recs)
        self.assertIn("using the `reuse-report (report only)` record in results.jsonl", flat(r.out))
        self.assertIn("results.jsonl record: 4 requests, 60 of 100 prompt tokens reused = 0.600", r.out)
        self.assertIn("| live | 4 | 100 | 60 | 0.600 |", r.out)

    def test_explicit_path_and_absent_report(self):
        r = Run(self, full_set(NEW))
        self.assertIn("No reuse report in the evidence directory", r.out)
        other = r.dir / "elsewhere.json"
        other.write_text(json.dumps(self.REPORT), encoding="utf-8")
        o = io.StringIO()
        with contextlib.redirect_stdout(o):
            m.main(["--dir", str(r.dir), "--reuse", str(other)])
        self.assertIn("| live | 4 | 100 | 60 | 0.600 |", o.getvalue())
        o = io.StringIO()
        with contextlib.redirect_stdout(o):
            m.main(["--dir", str(r.dir), "--reuse", str(r.dir / "nope.json")])
        self.assertIn("No reuse report at", o.getvalue())


if __name__ == "__main__":
    unittest.main()
