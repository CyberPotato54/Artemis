"""Tests for tools/py/m5_summary.py over synthetic records written to a temp dir. Standard library only.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v

The module is loaded by path (python -I keeps the script's directory off sys.path). The fixtures are built from copies of
the pinned record examples (EXAMPLES below, one line each, as `ConvertTo-Json -Compress` writes
them); the numbers are made up and the records hold no keys, hosts, prompts or replies, only counts and hashes.
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

MODULE_PATH = Path(__file__).resolve().parents[1] / "m5_summary.py"
spec = importlib.util.spec_from_file_location("m5_summary", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules["m5_summary"] = m
spec.loader.exec_module(m)

# The pinned record examples, verbatim (the `date` and `commit` keys are added by the runner's Record).
EXAMPLES = """
{"test":"serve-speed-spec run","input":"write-1234-on","run":"write:1234","mode":"on","attempt":1,"status":"pass","slots_per_layer":135,"desktop_gpu_mib":640.0,"stall_lines":0,"graceful":true,"exit_code":0,"test7_exit":0,"checks":{"fresh_engine":true},"figures":{"prompt":"write","seed":1234,"prompt_tokens":44,"completion_tokens":700,"decode_tokens":699,"decode_ms":6100.0,"decode_steps_tok_s":114.6,"decode_tok_s":114.8,"prefill_tok_s":880.0,"prefilled_tokens":44,"decode_misses":15000,"miss_mib_per_token":40.1,"spec_steps":180,"spec_plain_steps":200,"spec_drafted":420,"spec_accepted":300,"spec_accept_rate":0.714,"spec_replays":120,"completion_fnv":"0123456789abcdef"},"config_template":"…","template_sha256":"…","config_sha256":"…","overrides":["--park-snapshots","false"],"artemis_sha256":"…"}
{"test":"serve-speed-spec","status":"pass","attempt":1,"on_median":111.2,"bar":110.0,"per_prompt":[{"prompt":"write","on_median":104.0,"off_median":98.4,"ratio":1.057,"pass":true}],"prefill":[{"which":"long","on":2120.0,"off":2124.0,"ratio":0.998,"pass":true},{"which":"edit","on":1726.0,"off":1728.0,"ratio":0.999,"pass":true}],"off_bounds":{"decode_median":98.4,"decode_bound":46.5,"prefill_slowest":2124.0,"prefill_bound":1000.0,"pass":true},"identity":[{"run":"write-1234","equal":false}],"checks":{"dw1":true,"dw2":true,"dw3":true,"dw6":true},"runs":[{"run":"write-1234","mode":"on","decode_steps_tok_s":114.6,"prefill_tok_s":880.0,"slots_per_layer":135}],"desktop_gpu_mib":[640.0],"problems":[],"artemis_sha256":"…"}
{"test":"serve-shrunk run","input":"write-1234-s85-on","run":"write:1234","slots":85,"mode":"on","status":"ran","slots_per_layer":85,"figures":{"decode_steps_tok_s":70.1,"miss_mib_per_token":120.0,"spec_accept_rate":0.6,"spec_steps":100,"spec_plain_steps":300}}
{"test":"serve-shrunk","status":"report","pairs":[{"prompt":"write","slots":85,"off_tok_s":69.0,"on_tok_s":70.1,"ratio":1.016,"off_miss_mib_per_token":118.0,"on_miss_mib_per_token":120.0,"accept_rate":0.6,"plain_share":0.75}],"needs_owner":[]}
{"test":"spec-golden","status":"pass","outside_near_tie":0,"identical":14,"prompts":[{"id":"01-code","at":null,"near_tie":null,"margin":null,"identical_tokens":true,"identical_logits":false}],"spec":{"steps":900,"plain_steps":3000,"drafted":2100,"accepted":1500,"replays":600}}
{"test":"spec-long (report only)","status":"ran","prompt_tokens":20000,"runs":[{"mode":"off","decode_tok_s":80.0,"decoded":511,"decode_misses":12000,"spec":null},{"mode":"on","decode_tok_s":92.0,"decoded":511,"decode_misses":12500,"spec":{"steps":150,"plain_steps":100,"drafted":400,"accepted":260,"replays":120}}],"identical_tokens":false}
{"test":"test 1 identity","status":"report","kernels":[{"kernel":"matvec_tokens_q8_0","equal":24,"differ":0},{"kernel":"gdn_seq","equal":0,"differ":2}]}
{"test":"5","input":"05-code-write-lru","status":"pass","reports":[{"label":"plain","identical_tokens":false,"identical_logits":false}]}
""".strip().splitlines()
RUN, AGG, SHRUNK_RUN, SHRUNK, GOLDEN, LONG, T1, T5 = (json.loads(x) for x in EXAMPLES)

GATE = "abc1234"
DATE = "2026-11-10T12:00:00.0000000-07:00"
LEAVES = [("write", 1234), ("edit", 1234), ("explain", 1234), ("write", 1235), ("edit", 1235), ("explain", 1235)]
SHA = "a" * 64


def stamp(r, attempt=None):
    r = copy.deepcopy(r)
    r["commit"], r["date"] = GATE, DATE
    if attempt is not None:
        r["attempt"] = attempt
    return r


def run_record(mode, prompt, name, decode, prefill, attempt=1, fnv="0123456789abcdef", **fig):
    """A `serve-speed-spec run` record: the pinned example with this run's own figures. Off runs carry no spec counts."""
    r = copy.deepcopy(RUN)
    r["input"] = f"{name}-{mode}"
    r["mode"] = mode
    r["attempt"] = attempt
    r["artemis_sha256"] = SHA
    f = r["figures"]
    f.update(prompt=prompt, decode_steps_tok_s=decode, prefill_tok_s=prefill, completion_fnv=fnv)
    if mode == "off":
        for k in [k for k in f if k.startswith("spec_")]:
            f[k] = None
    f.update(fig)
    return stamp(r)


def attempt_runs(
    attempt=1,
    on=None,
    off=None,
    prefill_on=None,
    prefill_off=None,
    fnv_off="0123456789abcdef",
    fnv_on="0123456789abcdef",
):
    """Sixteen run records. `on` and `off` map a prompt (or a (prompt, seed)) to its decode tok/s; the prefill maps give
    the two long runs and the two edit runs. Defaults: on 120/115/130, off 100 each, no prefill change."""
    on = {"write": 120.0, "edit": 115.0, "explain": 130.0} | (on or {})
    off = {"write": 100.0, "edit": 100.0, "explain": 100.0} | (off or {})
    prefill_on = {"long": (2100.0, 2200.0), "edit": (1700.0, 1800.0)} | (prefill_on or {})
    prefill_off = {"long": (2100.0, 2200.0), "edit": (1700.0, 1800.0)} | (prefill_off or {})
    out = []
    for mode, decode, prefill, fnv in (("off", off, prefill_off, fnv_off), ("on", on, prefill_on, fnv_on)):
        for i, (prompt, seed) in enumerate(LEAVES):
            d = decode.get((prompt, seed), decode[prompt])
            p = prefill["edit"][seed - 1234] if prompt == "edit" else 150.0
            out.append(run_record(mode, prompt, f"{prompt}-{seed}", d, p, attempt, fnv))
        for i in (0, 1):
            out.append(run_record(mode, "long", f"long-{i + 1}", None, prefill["long"][i], attempt, fnv))
    return out


def aggregate(attempt=1, status="pass", **kw):
    """The pinned aggregate, with a passing default; `kw` replaces whole keys. The numbers are the defaults of
    attempt_runs, worked out by hand: the six "on" runs are 120, 120, 115, 115, 130, 130 (median 120)."""
    a = copy.deepcopy(AGG)
    a.update(
        status=status,
        attempt=attempt,
        artemis_sha256=SHA,
        on_median=120.0,
        per_prompt=[
            {"prompt": "write", "on_median": 120.0, "off_median": 100.0, "ratio": 1.2, "pass": True},
            {"prompt": "edit", "on_median": 115.0, "off_median": 100.0, "ratio": 1.15, "pass": True},
            {"prompt": "explain", "on_median": 130.0, "off_median": 100.0, "ratio": 1.3, "pass": True},
        ],
        prefill=[
            {"which": "long", "on": 2100.0, "off": 2100.0, "ratio": 1.0, "pass": True},
            {"which": "edit", "on": 1700.0, "off": 1700.0, "ratio": 1.0, "pass": True},
        ],
        off_bounds={"decode_median": 100.0, "decode_bound": 46.5, "prefill_slowest": 2100.0, "prefill_bound": 1000.0, "pass": True},
        checks={"dw1": True, "dw2": True, "dw3": True, "dw6": True},
        identity=[{"run": "write-1234", "equal": True}],
    )
    a.update(kw)
    return stamp(a)


def others():
    """The other blocks' records, from the pinned examples."""
    return [stamp(x) for x in (SHRUNK_RUN, SHRUNK, GOLDEN, LONG, T1, T5)]


def m4_file(path, decode=None):
    """A milestone 4 results.jsonl with 68 filler lines, then bf832e6's serve-speed record on line 69."""
    runs = [
        {"run": "write-1234", "long": False, "decode_steps_tok_s": decode or 98.113, "prefill_tok_s": 146.8},
        {"run": "edit-1234", "long": False, "decode_steps_tok_s": 93.8, "prefill_tok_s": 1728.9},
        {"run": "long-1", "long": True, "decode_steps_tok_s": None, "prefill_tok_s": 2124.6},
    ]
    line = {"test": "serve-speed", "commit": "bf832e6", "decode_median": 98.453, "prefill_slowest": 2124.4, "runs": runs}
    rows = [json.dumps({"test": "filler", "n": i}) for i in range(68)] + [json.dumps(line)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")


class Run:
    """Writes records to <tmp>/m5/results.jsonl (and, unless `m4` is False, a milestone 4 file beside it), runs main and
    keeps its output."""

    def __init__(self, test, recs, m4=True, argv=()):
        tmp = tempfile.TemporaryDirectory()
        test.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "m5"
        self.dir.mkdir()
        (self.dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
        if m4:
            m4_file(Path(tmp.name) / "m4" / "results.jsonl")
        o, e = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
            self.code = m.main(["--evidence", str(self.dir), *argv])
        self.out, self.err = o.getvalue(), e.getvalue()


def flat(text):
    """The output with its wrapped prose joined back into single lines."""
    return " ".join(text.split())


def full(attempt=1, **kw):
    return attempt_runs(attempt, **kw) + [aggregate(attempt)] + others()


class Imports(unittest.TestCase):
    def test_the_module_imports_without_running_main(self):
        self.assertTrue(callable(m.main))
        self.assertEqual(m.__name__, "m5_summary")
        # Loading it again prints nothing and does not exit.
        o, e = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
            again = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(again)
        self.assertEqual((o.getvalue(), e.getvalue()), ("", ""))

    def test_the_pinned_examples_parse_and_the_fixtures_keep_their_keys(self):
        self.assertEqual(len(EXAMPLES), 8)
        self.assertEqual(
            set(RUN["figures"]),
            {"prompt", "seed", "prompt_tokens", "completion_tokens", "decode_tokens", "decode_ms", "decode_steps_tok_s",
             "decode_tok_s", "prefill_tok_s", "prefilled_tokens", "decode_misses", "miss_mib_per_token", "spec_steps",
             "spec_plain_steps", "spec_drafted", "spec_accepted", "spec_accept_rate", "spec_replays", "completion_fnv"},
        )  # fmt: skip
        for r in attempt_runs():
            self.assertEqual(set(r) - {"commit", "date"}, set(RUN))
            self.assertEqual(set(r["figures"]), set(RUN["figures"]))
        self.assertEqual(set(aggregate()) - {"commit", "date"}, set(AGG))


class Median(unittest.TestCase):
    def test_the_median_of_six_is_the_mean_of_the_middle_two(self):
        self.assertEqual(m.median([10, 20, 30, 40, 50, 60]), 35)
        # Written out of order, with a pair that differs by seed: 90, 100, 108, 110, 112, 130 -> (108 + 110) / 2.
        on = {("write", 1234): 100.0, ("write", 1235): 108.0, ("edit", 1234): 90.0, ("edit", 1235): 130.0,
              ("explain", 1234): 110.0, ("explain", 1235): 112.0}  # fmt: skip
        j = m.judge(attempt_runs(on=on))
        self.assertEqual(j["on_median"], 109.0)
        self.assertEqual(j["counts"], {"on": 6, "off": 6})
        # 109.0 misses the 110.0 bar by one tok/s: done-when 1 fails, and nothing else follows from it.
        self.assertIs(j["checks"]["dw1"], False)
        self.assertEqual(j["status"], "pass")  # done-whens 3 and 6 hold in this fixture
        # Not the mean of all six (108.33), not the 3rd or the 4th value alone.
        self.assertNotIn(j["on_median"], (108.0, 110.0))
        self.assertAlmostEqual(j["per_prompt"][1]["on_median"], 110.0)  # edit: the two seeds, 90 and 130
        # A median of 110.0 exactly meets the bar.
        self.assertIs(m.judge(attempt_runs(on={("write", 1234): 110.0, ("write", 1235): 110.0, "edit": 110.0, "explain": 110.0}))["checks"]["dw1"], True)

    def test_the_median_of_six_is_what_the_printout_judges(self):
        on = {("write", 1234): 100.0, ("write", 1235): 108.0, ("edit", 1234): 90.0, ("edit", 1235): 130.0,
              ("explain", 1234): 110.0, ("explain", 1235): 112.0}  # fmt: skip
        r = Run(self, attempt_runs(on=on) + [aggregate(on_median=109.0, checks={"dw1": False, "dw2": True, "dw3": True, "dw6": True})] + others())
        self.assertIn("on median of six 109.00 (recomputed 109.00)", r.out)
        self.assertIn("| 1 | on median of six 109.00 (recomputed 109.00) | >= 110.0 | FAIL | FAIL | equal |", r.out)


class DoneWhen2(unittest.TestCase):
    def test_a_prompt_below_98_percent_of_off_fails_done_when_2(self):
        # Both seeds of "edit": off 100, on 98 (exactly 0.98) or 97.99 (0.9799).
        at = m.judge(attempt_runs(on={"edit": 98.0}))
        edit = at["per_prompt"][1]
        self.assertEqual((edit["prompt"], edit["on_median"], edit["off_median"]), ("edit", 98.0, 100.0))
        self.assertAlmostEqual(edit["ratio"], 0.98)
        self.assertIs(edit["pass"], True)
        self.assertIs(at["checks"]["dw2"], True)
        below = m.judge(attempt_runs(on={"edit": 97.99}))
        self.assertAlmostEqual(below["per_prompt"][1]["ratio"], 0.9799)
        self.assertIs(below["per_prompt"][1]["pass"], False)
        self.assertIs(below["checks"]["dw2"], False)
        # The other two prompts still pass; one miss fails the done-when.
        self.assertEqual([p["pass"] for p in below["per_prompt"]], [True, False, True])
        # Done-when 2 is reported, not gating: the 5a gate (done-whens 3 and 6) still passes.
        self.assertEqual(below["status"], "pass")

    def test_the_ratio_is_over_the_two_seeds_medians_not_a_single_run(self):
        # "edit" on: 90 and 106 -> median 98 against off 100 (0.98); one seed alone would pass or fail.
        j = m.judge(attempt_runs(on={("edit", 1234): 90.0, ("edit", 1235): 106.0}))
        self.assertEqual(j["per_prompt"][1]["on_median"], 98.0)
        self.assertIs(j["per_prompt"][1]["pass"], True)

    def test_the_printout_shows_both_sides_of_98_percent(self):
        for on, word in ((98.0, "pass"), (97.99, "FAIL")):
            checks = {"dw1": True, "dw2": on >= 98.0, "dw3": True, "dw6": True}
            agg = aggregate(
                checks=checks,
                per_prompt=[
                    {"prompt": "write", "on_median": 120.0, "off_median": 100.0, "ratio": 1.2, "pass": True},
                    {"prompt": "edit", "on_median": on, "off_median": 100.0, "ratio": on / 100.0, "pass": on >= 98.0},
                    {"prompt": "explain", "on_median": 130.0, "off_median": 100.0, "ratio": 1.3, "pass": True},
                ],
            )
            r = Run(self, attempt_runs(on={"edit": on}) + [agg] + others())
            self.assertIn(f"| edit | {on:.2f} | 100.00 | {on / 100.0:.4f} | {word} | {word} | equal |", r.out)
            self.assertIn(f"| 2 | each prompt's ratio, below | >= 0.98 of off | {word} | {word} | equal |", r.out)


class Prefill(unittest.TestCase):
    def test_both_prefill_figures_are_judged(self):
        ok = m.judge(attempt_runs())
        self.assertEqual([(p["which"], p["on"], p["off"], p["pass"]) for p in ok["prefill"]],
                         [("long", 2100.0, 2100.0, True), ("edit", 1700.0, 1700.0, True)])  # fmt: skip
        self.assertIs(ok["checks"]["dw3"], True)
        # The slower of the two runs counts: long 2100/2200, edit 1700/1800.
        slow_long = m.judge(attempt_runs(prefill_on={"long": (2200.0, 2057.0)}))
        self.assertEqual(slow_long["prefill"][0]["on"], 2057.0)
        self.assertIs(slow_long["prefill"][0]["pass"], False)  # 2057 / 2100 = 0.9795
        self.assertIs(slow_long["prefill"][1]["pass"], True)
        self.assertIs(slow_long["checks"]["dw3"], False)
        self.assertEqual(slow_long["status"], "fail")
        # The edit figure alone can fail it, too.
        slow_edit = m.judge(attempt_runs(prefill_on={"edit": (1600.0, 1800.0)}))
        self.assertEqual([p["pass"] for p in slow_edit["prefill"]], [True, False])
        self.assertIs(slow_edit["checks"]["dw3"], False)
        self.assertEqual(slow_edit["status"], "fail")
        # 0.98 exactly passes: 1666 / 1700.
        edge = m.judge(attempt_runs(prefill_on={"edit": (1666.0, 1800.0)}))
        self.assertIs(edge["prefill"][1]["pass"], True)

    def test_the_off_bounds_are_the_off_runs_alone(self):
        j = m.judge(attempt_runs(off={"write": 46.0, "edit": 46.0, "explain": 46.0}, on={"write": 46.0, "edit": 46.0, "explain": 46.0}))
        self.assertEqual(j["off_bounds"]["decode_median"], 46.0)
        self.assertIs(j["off_bounds"]["pass"], False)
        self.assertEqual(j["status"], "fail")
        j = m.judge(attempt_runs(prefill_off={"long": (999.0, 2200.0), "edit": (1700.0, 1800.0)}, prefill_on={"long": (999.0, 2200.0)}))
        self.assertEqual(j["off_bounds"]["prefill_slowest"], 999.0)
        self.assertIs(j["checks"]["dw6"], False)
        self.assertEqual(j["status"], "fail")
        edge = m.judge(attempt_runs(off={"write": 46.5, "edit": 46.5, "explain": 46.5}, on={"write": 46.5, "edit": 46.5, "explain": 46.5},
                                    prefill_off={"long": (1000.0, 1000.0)}, prefill_on={"long": (1000.0, 1000.0)}))
        self.assertIs(edge["checks"]["dw6"], True)

    def test_missing_runs_leave_the_verdicts_blank_not_passing(self):
        j = m.judge([])
        self.assertIsNone(j["on_median"])
        self.assertEqual(j["checks"], {"dw1": None, "dw2": None, "dw3": None, "dw6": None})
        self.assertIsNone(j["status"])

    def test_the_printout_gives_both_prefill_figures_on_and_off_and_the_ratio(self):
        r = Run(self, full(prefill_on={"long": (2057.0, 2200.0)}))
        self.assertIn("| long | 2057.00 | 2100.00 | 0.9795 | pass | FAIL | DIFFERENT |", r.out)
        self.assertIn("| edit | 1700.00 | 1700.00 | 1.0000 | pass | pass | equal |", r.out)
        # The recomputation disagrees with the record on done-when 3 and says so; it is not hidden.
        self.assertIn("| 3 | both prefill figures, below | >= 0.98 of off | pass | FAIL | DIFFERENT |", r.out)
        self.assertIn("| 6 | off median of six 100.00 (recomputed 100.00), off long prefill 2100.00 (recomputed 2100.00) |", r.out)


def drop(runs, *inputs):
    return [r for r in runs if r["input"] not in inputs]


class Completeness(unittest.TestCase):
    """The recomputation exists to catch a runner bug, so it never passes a run set that is not whole or not healthy."""

    def test_a_missing_run_never_recomputes_to_a_pass(self):
        for gone in ("explain-1235-on", "write-1234-off", "long-2-on", "long-1-off", "edit-1235-off", "edit-1234-on"):
            j = m.judge(drop(attempt_runs(), gone))
            self.assertNotEqual(j["status"], "pass", gone)
            self.assertIsNone(j["status"], gone)
            self.assertIn(None, j["checks"].values(), gone)
        # Which figure goes blank, run by run.
        j = m.judge(drop(attempt_runs(), "explain-1235-on"))
        self.assertIsNone(j["on_median"])
        self.assertIsNone(j["per_prompt"][2]["on_median"])
        self.assertEqual(
            (j["checks"]["dw1"], j["checks"]["dw2"], j["checks"]["dw3"], j["checks"]["dw6"]), (None, None, True, True)
        )
        j = m.judge(drop(attempt_runs(), "long-2-on"))
        self.assertIsNone(j["prefill"][0]["on"])
        self.assertIsNone(j["checks"]["dw3"])
        j = m.judge(drop(attempt_runs(), "edit-1235-off"))
        self.assertIsNone(j["prefill"][1]["off"])
        self.assertIsNone(j["off_bounds"]["decode_median"])
        self.assertIsNone(j["checks"]["dw6"])
        # A run whose figure is null is as missing as a run that was never recorded.
        runs = attempt_runs()
        runs[0]["figures"]["decode_steps_tok_s"] = None
        self.assertIsNone(m.judge(runs)["status"])
        # A third run of one prompt is not a whole set either.
        extra = attempt_runs() + [run_record("on", "write", "write-9999", 500.0, 150.0)]
        self.assertIsNone(m.judge(extra)["on_median"])
        self.assertEqual(m.judge(attempt_runs())["status"], "pass")

    def test_the_printout_does_not_repeat_a_pass_for_an_incomplete_attempt(self):
        runs = drop(attempt_runs(), "explain-1235-on", "long-2-on", "edit-1235-off")
        r = Run(self, runs + [aggregate()] + others())
        self.assertIn("recomputed status - (DIFFERENT)", r.out)
        self.assertNotIn("recomputed status pass", r.out)
        self.assertIn('the "on" median uses 0 runs and the "off" median 0 (six each', flat(r.out))

    def test_a_run_that_is_not_fresh_or_did_not_pass_fails_the_attempt(self):
        runs = attempt_runs()
        runs[3]["checks"]["fresh_engine"] = False
        j = m.judge(runs)
        self.assertEqual(j["status"], "fail")
        self.assertEqual(j["problems"], [f"{runs[3]['input']}: fresh_engine is not true"])
        self.assertIs(j["checks"]["dw3"], True)  # the figures themselves are fine; the run is not
        r = Run(self, runs + [aggregate()] + others())
        self.assertIn(f"!!! run problem: {runs[3]['input']}: fresh_engine is not true !!!", r.out)
        self.assertIn("recomputed status fail (DIFFERENT)", r.out)
        runs = attempt_runs()
        runs[9]["status"] = "fail"
        j = m.judge(runs)
        self.assertEqual(j["status"], "fail")
        self.assertEqual(j["problems"], [f"{runs[9]['input']}: status fail"])
        del runs[9]["checks"]["fresh_engine"]
        self.assertEqual(len(m.judge(runs)["problems"]), 2)  # an absent check is not a pass

    def test_a_plain_vs_plain_set_is_a_fail_not_a_pass(self):
        # The "on" runs ran plain: no spec_steps, so the figures alone (dw3 and dw6) would pass.
        runs = attempt_runs()
        for r in runs:
            if r["mode"] == "on":
                for k in [k for k in r["figures"] if k.startswith("spec_")]:
                    r["figures"][k] = None
        j = m.judge(runs)
        self.assertEqual(j["status"], "fail")
        self.assertIs(j["checks"]["dw3"], True)
        self.assertIs(j["checks"]["dw6"], True)
        self.assertEqual(len([p for p in j["problems"] if "did not speculate" in p]), 6)  # long rows are skipped
        self.assertIn("made no verify step", j["problems"][-1])
        r = Run(self, runs + [aggregate()] + others())
        self.assertIn("recomputed status fail (DIFFERENT)", r.out)

    def test_an_off_run_that_speculated_is_a_fail(self):
        runs = attempt_runs()
        off = next(r for r in runs if r["input"] == "edit-1235-off")
        off["figures"]["spec_steps"] = 12
        j = m.judge(runs)
        self.assertEqual(j["status"], "fail")
        self.assertEqual(j["problems"], ["edit-1235-off: spec_steps is 12, so the \"off\" process speculated"])

    def test_zero_verify_steps_over_the_on_runs_is_a_fail_but_one_run_may_have_none(self):
        runs = attempt_runs()
        for r in runs:
            if r["mode"] == "on" and r["figures"]["prompt"] != "long":
                r["figures"]["spec_steps"] = 0
        j = m.judge(runs)
        self.assertEqual(j["status"], "fail")
        self.assertEqual(j["problems"], ['the "on" decode runs made no verify step between them: speculation never ran'])
        runs = attempt_runs()
        next(r for r in runs if r["input"] == "write-1234-on")["figures"]["spec_steps"] = 0
        self.assertEqual(m.judge(runs)["status"], "pass")

    def test_long_runs_that_counted_different_prompts_are_a_fail(self):
        runs = attempt_runs()
        self.assertEqual(m.judge(runs)["status"], "pass")
        next(r for r in runs if r["input"] == "long-2-on")["figures"]["prompt_tokens"] = 19999
        j = m.judge(runs)
        self.assertEqual(j["status"], "fail")
        self.assertEqual(len(j["problems"]), 1)
        self.assertTrue(j["problems"][0].startswith("the long runs counted different prompts: "))
        self.assertTrue(j["problems"][0].endswith(" tokens"))

    def test_a_known_dw3_miss_is_a_fail_even_when_dw6_is_unjudged(self):
        runs = drop(attempt_runs(prefill_on={"edit": (1600.0, 1800.0)}), "long-1-off", "long-2-off")
        j = m.judge(runs)
        self.assertIs(j["checks"]["dw3"], False)
        self.assertIsNone(j["checks"]["dw6"])
        self.assertEqual(j["status"], "fail")

    def test_the_off_bounds_pass_is_read_from_the_record_not_recomputed(self):
        bounds = {"decode_median": 100.0, "decode_bound": 46.5, "prefill_slowest": 2100.0, "prefill_bound": 1000.0}
        rec = aggregate(off_bounds=bounds | {"pass": False})
        r = Run(self, attempt_runs() + [rec] + others())
        self.assertIn("off_bounds.pass: record FAIL, recomputed pass (DIFFERENT).", r.out)
        r = Run(self, full())
        self.assertIn("off_bounds.pass: record pass, recomputed pass (equal).", r.out)
        self.assertIn("| off decode median of six | 100.00 | 100.00 | 46.5 | pass |", r.out)


class Attempts(unittest.TestCase):
    def test_both_attempts_are_printed(self):
        first = attempt_runs(1, on={"edit": 90.0}) + [aggregate(1, status="fail", on_median=120.0,
            checks={"dw1": True, "dw2": False, "dw3": True, "dw6": True})]  # fmt: skip
        second = attempt_runs(2, on={"edit": 118.0}) + [aggregate(2)]
        r = Run(self, first + second + others())
        self.assertEqual(r.code, 0, r.err)
        self.assertIn("#### Attempt 1", r.out)
        self.assertIn("#### Attempt 2", r.out)
        self.assertIn("Attempts at this commit: 1, 2; every one is printed below. Only attempts 1 and 2 are the "
                      "gate's, and the last of those counts.", flat(r.out))
        # The same inputs in two attempts are not a duplicate record.
        self.assertNotIn("records of", r.out)
        self.assertNotIn("!!!", r.out)
        # Each attempt's own edit figure is in its own block.
        a1, a2 = r.out.split("#### Attempt 2")
        self.assertIn("| edit | 90.00 | 100.00 | 0.9000 |", a1)
        self.assertIn("| edit | 118.00 | 100.00 | 1.1800 |", a2)
        self.assertNotIn("| edit | 90.00 |", a2)
        # Sixteen run rows in each attempt's table.
        self.assertEqual(a1.count("| write-1234-on |"), 2)  # the performance table and the health table
        self.assertEqual(a2.count("| write-1234-on |"), 2)

    def test_a_second_miss_goes_to_the_owner(self):
        miss = lambda n: attempt_runs(n, prefill_on={"long": (2057.0, 2200.0)}) + [aggregate(n, status="fail", checks={"dw1": True, "dw2": True, "dw3": False, "dw6": True})]  # noqa: E731
        r = Run(self, miss(1) + miss(2) + others())
        self.assertIn("a second miss goes to the owner", flat(r.out))
        self.assertIn("THIS BLOCK GOES TO THE OWNER: attempts 1 and 2 both missed", flat(r.out))
        r = Run(self, miss(1) + others())
        self.assertNotIn("a second miss goes to the owner", flat(r.out))
        self.assertNotIn("GOES TO THE OWNER", r.out)
        # A pass after a miss is a rerun that worked, not a case for the maintainer.
        r = Run(self, miss(1) + full(2) + others())
        self.assertNotIn("GOES TO THE OWNER", r.out)
        self.assertIn("The counting attempt is 2.", r.out)

    def test_a_third_attempt_goes_to_the_owner_and_its_pass_is_not_the_gates(self):
        miss = lambda n: attempt_runs(n, prefill_on={"long": (2057.0, 2200.0)}) + [aggregate(n, status="fail", checks={"dw1": True, "dw2": True, "dw3": False, "dw6": True})]  # noqa: E731
        r = Run(self, miss(1) + miss(2) + full(3) + others())
        self.assertIn("THIS BLOCK GOES TO THE OWNER", r.out)
        self.assertIn("The counting attempt is 2.", r.out)
        self.assertNotIn("The counting attempt is 3.", r.out)
        self.assertIn("#### Attempt 3 (NOT THE GATE'S)", r.out)
        self.assertIn("no verdict below is the gate's", flat(r.out))
        # Attempt 3 alone, a lost predecessor, an attempt numbered 0, and 2 without 1 each stop it too.
        for recs, counting in (
            (full(3), None),
            (full(2), "2"),
            (full(0), None),
            (full(5) + full(1), "1"),
        ):
            r = Run(self, recs + others())
            self.assertIn("THIS BLOCK GOES TO THE OWNER", r.out, recs[-1].get("attempt"))
            if counting is None:
                self.assertIn("No attempt 1 or 2: nothing here counts as the gate.", r.out)
            else:
                self.assertIn(f"The counting attempt is {counting}.", r.out)
        # A record with no attempt key at all reads as attempt 0.
        bare = [{k: v for k, v in x.items() if k != "attempt"} for x in full(1)]
        r = Run(self, bare + others())
        self.assertIn("THIS BLOCK GOES TO THE OWNER", r.out)
        self.assertIn("#### Attempt 0 (NOT THE GATE'S)", r.out)

    def test_two_done_when_1_misses_go_to_the_owner_even_when_the_status_is_pass(self):
        low = {"write": 105.0, "edit": 105.0, "explain": 105.0}
        dw1_miss = lambda n: attempt_runs(n, on=low) + [aggregate(n, on_median=105.0, checks={"dw1": False, "dw2": True, "dw3": True, "dw6": True})]  # noqa: E731
        r = Run(self, dw1_miss(1) + dw1_miss(2) + others())
        self.assertIn("THIS BLOCK GOES TO THE OWNER: attempts 1 and 2 both missed", flat(r.out))
        self.assertIn("done-when 1", flat(r.out))
        self.assertIn("goes to 5b through the owner", flat(r.out))
        r = Run(self, dw1_miss(1) + others())
        self.assertNotIn("GOES TO THE OWNER", r.out)

    def test_a_recomputed_fail_counts_as_a_miss_even_when_the_record_says_pass(self):
        def stale(n):
            runs = attempt_runs(n)
            runs[2]["checks"]["fresh_engine"] = False
            return runs + [aggregate(n)]  # the record says pass; the run it summarises was not fresh

        r = Run(self, stale(1) + stale(2) + others())
        self.assertIn("recomputed status fail", flat(r.out))
        self.assertIn("THIS BLOCK GOES TO THE OWNER: attempts 1 and 2 both missed", flat(r.out))

    def test_an_attempt_that_ended_before_its_aggregate_is_named(self):
        r = Run(self, attempt_runs(1)[:5] + others())
        self.assertIn("No aggregate record for attempt 1", r.out)
        self.assertIn("Runs recorded: 5", r.out)

    def test_a_record_of_another_commit_is_superseded_not_used(self):
        old = [dict(x, commit="9999999") for x in full(1)]
        r = Run(self, old + full(1))
        self.assertIn(f"Gate commit: {GATE}", r.out)
        self.assertIn("Superseded: commit 9999999, ", r.out)
        r = Run(self, old + full(1), argv=["--gate", "9999999"])
        self.assertIn("Gate commit: 9999999", r.out)


class Verdicts(unittest.TestCase):
    def test_a_record_that_disagrees_with_its_runs_is_flagged_different(self):
        liar = aggregate(on_median=111.2)  # the runs say 120.0
        r = Run(self, attempt_runs() + [liar] + others())
        self.assertIn("On median of six: record 111.2000, recomputed 120.0000 (DIFFERENT), against 110.0.", r.out)
        r = Run(self, full())
        self.assertIn("On median of six: record 120.0000, recomputed 120.0000 (equal), against 110.0.", r.out)

    def test_the_gate_status_is_done_whens_3_and_6_only(self):
        r = Run(self, full())
        self.assertIn("Record: status pass, attempt 1, problems []; recomputed status pass (equal).", r.out)
        # Done-when 1 missed (the median is below 110) but the block still passes.
        r = Run(self, attempt_runs(on={"write": 105.0, "edit": 105.0, "explain": 105.0}) + [aggregate(on_median=105.0, checks={"dw1": False, "dw2": True, "dw3": True, "dw6": True})] + others())
        self.assertIn("recomputed status pass (equal)", r.out)
        self.assertIn("| 1 | on median of six 105.00 (recomputed 105.00) | >= 110.0 | FAIL | FAIL | equal |", r.out)


class ForScale(unittest.TestCase):
    def test_each_run_sits_beside_bf832e6s_same_run(self):
        r = Run(self, full())
        self.assertIn("bf832e6's serve-speed (line 69 of milestone 4's results.jsonl): decode median of six 98.45 tok/s, "
                      "slower long prefill 2124.40 tok/s.", flat(r.out))
        # write-1234-on: 120 against 98.113 -> 1.223; prefill 150 against 146.8 -> 1.022.
        self.assertIn("| write-1234-on | on | 120.00 | 98.11 | 1.223 | 150.00 | 146.80 | 1.022 |", r.out)
        # The long runs have no decode figure on either side.
        self.assertIn("| long-1-off | off | - | - | - | 2100.00 | 2124.60 | 0.988 |", r.out)
        # A run bf832e6 lacks stays blank rather than failing.
        self.assertIn("| explain-1234-on | on | 130.00 | - | - | 150.00 | - | - |", r.out)

    def test_no_m4_file_blanks_the_columns(self):
        r = Run(self, full(), m4=False)
        self.assertEqual(r.code, 0, r.err)
        self.assertIn("No milestone 4 reference", r.out)
        self.assertIn("| write-1234-on | on | 120.00 | - | - | 150.00 | - | - |", r.out)

    def test_line_69_must_be_bf832e6s_serve_speed_record(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "results.jsonl"
            m4_file(path)
            rec, note = m.load_m4(path)
            self.assertEqual((rec["test"], note), ("serve-speed", None))
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(["{}"] + lines) + "\n", encoding="utf-8")  # every line moves down one
            rec, note = m.load_m4(path)
            self.assertIsNone(rec)
            self.assertIn("is not bf832e6's serve-speed record", note)
            path.write_text("\n".join(lines[:60]) + "\n", encoding="utf-8")
            self.assertIn("has 60 lines, not 69", m.load_m4(path)[1])

    def test_the_m4_flag_names_another_file(self):
        with tempfile.TemporaryDirectory() as d:
            other = Path(d) / "elsewhere" / "results.jsonl"
            m4_file(other, decode=97.0)
            r = Run(self, full(), m4=False, argv=["--m4", str(other)])
            self.assertIn("| write-1234-on | on | 120.00 | 97.00 |", r.out)


class Identity(unittest.TestCase):
    def test_completion_hashes_are_listed_per_pair_and_never_judged(self):
        r = Run(self, full(fnv_off="1111111111111111", fnv_on="2222222222222222"))
        self.assertIn("| write-1234 | 1111111111111111 | 2222222222222222 | False | True | DIFFERENT |", r.out)
        self.assertIn("Equal in 0 of 8 pairs with both hashes.", r.out)
        r = Run(self, full())
        self.assertIn("| write-1234 | 0123456789abcdef | 0123456789abcdef | True | True | equal |", r.out)
        self.assertIn("Equal in 8 of 8 pairs with both hashes.", r.out)

    def test_spec_golden_test_5_and_test_1_are_reported(self):
        r = Run(self, full())
        self.assertIn("spec-golden: identical 14 (of the 1 prompts listed, 1 have identical tokens and 0 identical logits).", r.out)
        self.assertIn("test 5 (05-code-write-lru): status pass, 1 report run(s).", r.out)
        self.assertIn("| plain | False | False |", r.out)
        self.assertIn("test 1 identity: 2 kernels, status report.", r.out)
        self.assertIn("| matvec_tokens_q8_0 | 24 | 0 |", r.out)
        self.assertIn("| gdn_seq | 0 | 2 |", r.out)

    def test_test_5_without_reports_says_so(self):
        old = stamp(T5)
        del old["reports"]
        r = Run(self, attempt_runs() + [aggregate(), old])
        self.assertIn("No `reports` on this record", r.out)


class OtherBlocks(unittest.TestCase):
    def test_the_spec_golden_verdict_is_recomputed(self):
        r = Run(self, full())
        self.assertIn("Verdict pass: 0 divergence(s) outside a near tie (recomputed 0: equal); status recomputed pass (equal).", r.out)
        self.assertIn("Speculation counts over the run: steps 900, plain 3000, drafted 2100, accepted 1500, replays 600; "
                      "accept rate 0.714, plain share 0.769.", flat(r.out))
        self.assertIn("| 01-code | - | - | - | True | False |", r.out)
        bad = stamp(GOLDEN)
        bad["prompts"] = [{"id": "02-x", "at": 17, "near_tie": False, "margin": 0.2, "identical_tokens": False, "identical_logits": None}]
        bad["status"], bad["outside_near_tie"] = "fail", 1
        r = Run(self, attempt_runs() + [aggregate(), bad])
        self.assertIn("Verdict fail: 1 divergence(s) outside a near tie (recomputed 1: equal); status recomputed fail (equal).", r.out)
        self.assertIn("| 02-x | 17 | False | 0.2000 | False | - |", r.out)

    def test_the_spec_golden_status_folds_in_the_problems_the_exit_code_and_subset(self):
        def status_of(**kw):
            g = stamp(GOLDEN)
            g.update(kw)
            return Run(self, attempt_runs() + [aggregate(), g]).out

        # Nothing outside a near tie, but the verdict says why it is no pass: the recomputation agrees with the record.
        out = status_of(status="fail", problems=["speculation made no verify step in any on run, so nothing was checked"])
        self.assertIn("status recomputed fail (equal)", out)
        out = status_of(status="pass", problems=["no prompt was compared"])
        self.assertIn("status recomputed fail (DIFFERENT)", out)
        # A subset run: exit 1, the status "subset", and the recomputation says subset too (never pass).
        sub = ["only 3 of 20 golden prompts compared: a subset run is not the gate's pass"]
        out = status_of(status="subset", exit_code=1, problems=sub)
        self.assertIn("status recomputed subset (equal)", out)
        out = status_of(status="pass", problems=sub)
        self.assertIn("status recomputed subset (DIFFERENT)", out)
        # A subset with a second problem is a fail.
        out = status_of(status="fail", exit_code=1, problems=sub + ["no prompt was compared"])
        self.assertIn("status recomputed fail (equal)", out)
        # A pass verdict whose command exited non-zero is a fail.
        out = status_of(status="pass", exit_code=3)
        self.assertIn("status recomputed fail (DIFFERENT)", out)
        out = status_of(status="pass", exit_code=0)
        self.assertIn("status recomputed pass (equal)", out)
        # A 5b record whose on runs never drafted with MTP is a fail (the gate tested lookup alone); a 5a record, or one
        # that drafted with MTP, is not touched.
        spec = {"steps": 900, "plain_steps": 3000, "drafted": 2100, "accepted": 1500, "replays": 600}
        out = status_of(plan="5b", status="pass", spec=dict(spec, drafted_mtp=0))
        self.assertIn("status recomputed fail (DIFFERENT)", out)
        out = status_of(plan="5b", status="fail", spec=dict(spec, drafted_mtp=0))
        self.assertIn("status recomputed fail (equal)", out)
        out = status_of(plan="5b", status="pass", spec=dict(spec, drafted_mtp=40))
        self.assertIn("status recomputed pass (equal)", out)
        out = status_of(plan="5b", status="fail", sources=["lookup", "mtp"], mtp_loaded=True, spec=dict(spec, drafted_mtp=0))
        self.assertIn("status recomputed fail (equal)", out)
        out = status_of(plan="5b", status="pass", sources=["lookup"], mtp_loaded=False, spec=dict(spec, drafted_mtp=0))
        self.assertIn("status recomputed pass (equal)", out)
        out = status_of(plan="5b", status="pass", sources=["lookup"], spec=dict(spec, drafted_mtp=0))
        self.assertIn("status recomputed pass (equal)", out)
        out = status_of(plan="5a", status="pass", spec=dict(spec, drafted_mtp=0))
        self.assertIn("status recomputed pass (equal)", out)

    def test_the_shrunk_and_long_tables(self):
        r = Run(self, full())
        self.assertIn("| write | 85 | 69.00 | 70.10 | 1.0160 | 118.0 | 120.0 | 0.600 | 0.750 |", r.out)
        self.assertIn("Status report; 1 pairs; 0 below 0.98 listed for the owner (recomputed from the pairs: 0).", r.out)
        self.assertIn("| off | 80.00 | 511 | 12000 | none (speculation off) |", r.out)
        self.assertIn("| on | 92.00 | 511 | 12500 | steps 150, plain 100, drafted 400, accepted 260, replays 120; accept rate 0.650, plain share 0.400 |", r.out)
        self.assertIn("identical: False", r.out)
        slow = stamp(SHRUNK)
        slow["status"] = "ask-owner"
        slow["pairs"][0]["ratio"] = 0.9
        slow["needs_owner"] = [{"prompt": "write", "slots": 85, "ratio": 0.9}]
        r = Run(self, attempt_runs() + [aggregate(), slow])
        self.assertIn("Status ask-owner; 1 pairs; 1 below 0.98 listed for the owner (recomputed from the pairs: 1).", r.out)
        self.assertIn("Needs the owner: write at 85 slots, ratio 0.9000.", r.out)

    def test_a_shrunk_record_with_the_old_key_and_status_still_lists_its_pair(self):
        old = stamp(SHRUNK)
        old["status"] = "ask-" + "dev" + "on"  # the old spelling, put together so no private name sits in this file
        old["pairs"][0]["ratio"] = 0.9
        old.pop("needs_owner")
        old["needs_" + "dev" + "on"] = [{"prompt": "write", "slots": 85, "ratio": 0.9}]
        r = Run(self, attempt_runs() + [aggregate(), old])
        self.assertIn("Status ask-owner; 1 pairs; 1 below 0.98 listed for the owner (recomputed from the pairs: 1).", r.out)
        self.assertIn("Needs the owner: write at 85 slots, ratio 0.9000.", r.out)

    def test_missing_blocks_are_named_and_do_not_crash(self):
        r = Run(self, [stamp(T1)])
        self.assertEqual(r.code, 0, r.err)
        for name in ("serve-speed-spec", "spec-golden", "test 5", "serve-shrunk", "spec-long"):
            self.assertIn(f"**{name}: missing**", r.out)

    def test_the_binary_hash_is_one_value_or_a_loud_warning(self):
        r = Run(self, full())
        self.assertIn("artemis_sha256: one value over", r.out)
        runs = attempt_runs()
        runs[0]["artemis_sha256"] = "b" * 64
        r = Run(self, runs + [aggregate()] + others())
        self.assertIn("2 DIFFERENT artemis_sha256 VALUES", r.out)


class Hygiene(unittest.TestCase):
    def test_no_prose_line_is_wider_than_120_columns(self):
        r = Run(self, full() + attempt_runs(2) + [aggregate(2)])
        for line in r.out.splitlines():
            if not line.startswith("|"):
                self.assertLessEqual(len(line), 120, line)

    def test_a_missing_file_or_commit_is_an_error_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()) as e:
            self.assertEqual(m.main(["--evidence", d]), 1)
            self.assertIn("not found", e.getvalue())
        r = Run(self, full(), argv=["--gate", "ffffffff"])
        self.assertEqual(r.code, 1)
        self.assertIn("no record of commit ffffffff", r.err)

    def test_the_output_holds_no_text_beyond_the_records_counts_and_hashes(self):
        # A planted marker in a figure that is not a printed key never reaches the output.
        runs = attempt_runs()
        runs[0]["figures"]["reply_text"] = "MARKER-REPLY"
        runs[0]["extra"] = "MARKER-EXTRA"
        r = Run(self, runs + [aggregate()] + others())
        self.assertNotIn("MARKER", r.out)


# ---- plan 5b -----------------------------------------------------------------------------------------------------------
# The pinned record examples for plan "5b", one line each; the numbers are illustrative.
EXAMPLES_5B = """
{"test":"serve-speed-spec","plan":"5b","status":"pass","attempt":1,"on_median":112.4,"bar":110.0,"per_prompt":[{"prompt":"write","on_median":110.9,"off_median":98.4,"ratio":1.127,"pass":true}],"prefill":[{"which":"long","on":2118.0,"off":2124.0,"ratio":0.997,"pass":true},{"which":"edit","on":1720.0,"off":1728.0,"ratio":0.995,"pass":true}],"off_bounds":{"decode_median":98.4,"decode_bound":46.5,"prefill_slowest":2124.0,"prefill_bound":1000.0,"pass":true},"identity":[{"run":"write-1234","equal":false}],"slots":[{"run":"write-1234","off":135,"on":134,"equal":false}],"checks":{"dw1":true,"dw2":true,"dw3":true,"dw6":true},"lookup_median":102.9,"mtp_gain":1.092,"mtp_pays":true,"runs":[{"run":"write-1234","mode":"on","decode_steps_tok_s":111.0,"prefill_tok_s":880.0,"slots_per_layer":134},{"run":"write-1234","mode":"lookup","decode_steps_tok_s":97.1,"prefill_tok_s":881.0,"slots_per_layer":135}],"desktop_gpu_mib":[640.0],"problems":[],"artemis_sha256":"…"}
{"test":"mtp-32gb","plan":"5b","status":"pass","free_before_gib":26.0,"runs":[{"mode":"off","host_bytes":20615843840,"host_heap_usage":20685000000,"mtp_vram_bytes":0,"pool_gib":2},{"mode":"on","host_bytes":21102383104,"host_heap_usage":21171600000,"mtp_vram_bytes":61341696,"pool_gib":2}]}
{"test":"2-mtp","plan":"5b","status":"pass","form":"taps","worst":{"tap":"mtp_ffn_out","err":3.1e-6,"bound":1e-5},"taps":18}
{"test":"2-mtp-gpu","plan":"5b","status":"pass","positions":15,"top1_equal":15,"near_tie_differences":0,"worst_rel_l2":2.0e-3}
{"test":"spec-long (report only)","plan":"5b","status":"ran","prompt_tokens":20000,"runs":[{"mode":"off","decode_tok_s":80.0,"decoded":511,"decode_misses":12000,"spec":null},{"mode":"on","decode_tok_s":95.0,"decoded":511,"decode_misses":12400,"spec":{"steps":150,"plain_steps":100,"drafted":400,"accepted":260,"replays":120,"drafted_lookup":250,"accepted_lookup":190,"drafted_mtp":150,"accepted_mtp":70,"probes":2}},{"mode":"on-full-window","decode_tok_s":95.4,"decoded":511,"decode_misses":12400,"spec":{"steps":152,"plain_steps":98,"drafted":405,"accepted":266,"replays":118,"drafted_lookup":250,"accepted_lookup":190,"drafted_mtp":155,"accepted_mtp":76,"probes":2}}],"identical_tokens":false}
""".strip().splitlines()
AGG5, MTP32, T2MTP, T2MTPGPU, LONG5 = (json.loads(x) for x in EXAMPLES_5B)
SOURCES = dict(spec_drafted_lookup=250, spec_accepted_lookup=190, spec_drafted_mtp=150, spec_accepted_mtp=70)
THREE = ("write", "edit", "explain")


def attempt_runs_5b(attempt=1, lookup=None, **kw):
    """Plan 5b's run records: `attempt_runs` with `plan`, the per-source figures on every speculating run, and the six
    "lookup" decode runs (`lookup` maps a prompt to its decode tok/s; default 100 each, drafting nothing with MTP)."""
    runs = attempt_runs(attempt, **kw)
    for r in runs:
        r["plan"] = "5b"
        if r["mode"] == "on" and r["figures"]["spec_steps"] is not None:
            r["figures"].update(SOURCES)
        elif r["mode"] == "off":
            r["figures"].update({k: None for k in SOURCES})
    lookup = dict.fromkeys(THREE, 100.0) | (lookup or {})
    for prompt, seed in LEAVES:
        r = run_record("lookup", prompt, f"{prompt}-{seed}", lookup[prompt], 150.0, attempt)
        r["plan"] = "5b"
        r["figures"].update(spec_drafted_lookup=420, spec_accepted_lookup=300, spec_drafted_mtp=0, spec_accepted_mtp=0)
        runs.append(r)
    return runs


def aggregate_5b(attempt=1, **kw):
    a = aggregate(attempt)
    a["plan"] = "5b"
    a.update(lookup_median=100.0, mtp_gain=1.2, mtp_pays=True)
    a.update(kw)
    return a


def full_5b(attempt=1, **kw):
    return attempt_runs_5b(attempt, **kw) + [aggregate_5b(attempt)] + others()


class Plan5b(unittest.TestCase):
    def test_the_5b_examples_parse(self):
        self.assertEqual(len(EXAMPLES_5B), 5)
        for r in (AGG5, MTP32, T2MTP, T2MTPGPU, LONG5):
            self.assertEqual(r["plan"], "5b")

    def test_5b_status_needs_dw1_and_dw2(self):
        low = dict.fromkeys(THREE, 105.0)  # dw1 misses (median 105 < 110); dw2 and dw3 hold
        checks = {"dw1": False, "dw2": True, "dw3": True, "dw6": True}
        r = Run(self, attempt_runs_5b(on=low) + [aggregate_5b(on_median=105.0, checks=checks)] + others())
        self.assertIn("Plan 5b. Record: status pass,", r.out)
        self.assertIn("recomputed status fail (DIFFERENT)", r.out)
        self.assertIn("Done-when 1 MISSED", r.out)
        self.assertIn("| 1 | on median of six 105.00 (recomputed 105.00) | >= 110.0 | FAIL | FAIL | equal |", r.out)
        # The same figures under 5a (no plan key) pass on done-whens 3 and 6.
        r = Run(self, attempt_runs(on=low) + [aggregate(on_median=105.0, checks=checks)] + others())
        self.assertIn("recomputed status pass (equal)", r.out)
        # Done-when 2 alone: dw1 holds (every on run 120) but edit's ratio is 120 / 125, under 0.98.
        hold = dict.fromkeys(THREE, 120.0)
        r = Run(self, attempt_runs_5b(on=hold, off={"edit": 125.0}) + [aggregate_5b(status="fail")] + others())
        self.assertIn("recomputed status fail (equal)", r.out)
        r = Run(self, attempt_runs(on=hold, off={"edit": 125.0}) + [aggregate()] + others())
        self.assertIn("recomputed status pass (equal)", r.out)
        # A whole passing 5b set passes.
        r = Run(self, full_5b())
        self.assertIn("Plan 5b. Record: status pass, attempt 1, problems []; recomputed status pass (equal).", flat(r.out))
        # A missing "lookup" run never recomputes to a pass; the MTP verdict goes blank too.
        short = [x for x in full_5b() if x.get("input") != "write-1234-lookup"]
        out = flat(Run(self, short).out)
        self.assertIn("recomputed status - (DIFFERENT)", out)
        self.assertIn("MTP pays (on at least 1.02 times lookup): record True, recomputed - (DIFFERENT)", out)

    def test_5a_records_keep_5as_rule(self):
        # Records with no `plan` are 5a's: judged on done-whens 3 and 6, with none of 5b's sections.
        r = Run(self, full())
        self.assertIn("Record: status pass, attempt 1, problems []; recomputed status pass (equal).", r.out)
        for text in ("Plan 5a", "Plan 5b", "ookup", "mtp-32gb", "2-mtp", "Window comparison"):
            self.assertNotIn(text, r.out)
        # An explicit "5a" is the same rule.
        explicit = full()
        for x in explicit:
            x["plan"] = "5a"
        r2 = Run(self, explicit)
        self.assertIn("recomputed status pass (equal)", r2.out)
        self.assertNotIn("Plan 5b", r2.out)
        # The committed evidence (5b's gate records, which supersede 5a's) summarises, and every recomputation agrees
        # with its record.
        evidence = Path(__file__).resolve().parents[3] / "docs" / "superpowers" / "evidence" / "m5"
        if (evidence / "results.jsonl").exists():
            o, e = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
                code = m.main(["--evidence", str(evidence)])
            self.assertEqual(code, 0, e.getvalue())
            self.assertNotIn("THE ATTEMPTS NAME", o.getvalue())
            self.assertNotIn("DIFFERENT", o.getvalue())

    def test_an_unknown_plan_is_a_problem_and_a_dw1_miss_is_named_from_the_record(self):
        runs = attempt_runs_5b()
        for r in runs:
            r["plan"] = "5c"
        out = Run(self, runs + [aggregate_5b(plan="5c")] + others()).out
        self.assertIn("unknown plan '5c'", out)
        self.assertIn("recomputed status fail (DIFFERENT)", out)
        # The record says dw1 missed but a run is missing, so the recomputation is blank: still named plainly.
        short = [x for x in attempt_runs_5b() if x.get("input") != "write-1234-on"]
        agg = aggregate_5b(checks={"dw1": False, "dw2": True, "dw3": True, "dw6": True})
        out = Run(self, short + [agg] + others()).out
        self.assertIn("Done-when 1 MISSED (record only; the recomputation is blank)", out)

    def test_the_32gb_pool_is_read_and_judged(self):
        def run32(on_gib, heap):
            rec = copy.deepcopy(MTP32)
            rec["runs"][1].update(pool_gib=on_gib, host_heap_usage=heap)
            rec["status"] = "pass" if on_gib >= 2 else "ask-owner"
            return Run(self, full_5b() + [stamp(rec)]).out

        out = run32(2, 21171600000)
        self.assertIn("Status pass; recomputed pass (equal)", out)
        self.assertIn("| on | 21102383104 | 21171600000 | 61341696 | 2 | 2 | equal |", out)
        # 1 GiB with MTP on is ask-owner.
        out = run32(1, 21800000000)
        self.assertIn("Status ask-owner; recomputed ask-owner (equal)", out)
        self.assertIn("| on | 21102383104 | 21800000000 | 61341696 | 1 | 1 | equal |", out)
        # A record that says pass over a 1 GiB pool is DIFFERENT.
        liar = copy.deepcopy(MTP32)
        liar["runs"][1]["pool_gib"] = 1
        out = Run(self, full_5b() + [stamp(liar)]).out
        self.assertIn("Status pass; recomputed ask-owner (DIFFERENT)", out)
        # A null pool, or a null heap with a pool of 2, is not measured: never a pass.
        for key in ("pool_gib", "host_heap_usage"):
            nul = copy.deepcopy(MTP32)
            nul["runs"][1][key] = None
            out = flat(Run(self, full_5b() + [stamp(nul)]).out)
            self.assertIn("Status pass; recomputed not measured (DIFFERENT)", out)
        # The worked-out pool is at 26 GiB free whatever free_before_gib says, and capped at 8.
        far = copy.deepcopy(MTP32)
        far["free_before_gib"] = 60.0
        self.assertIn("| on | 21102383104 | 21171600000 | 61341696 | 2 | 2 | equal |", Run(self, full_5b() + [stamp(far)]).out)
        self.assertEqual(m.pool_gib(0), 8)
        # No such block: named, not a crash.
        self.assertIn("**mtp-32gb: missing**", Run(self, full_5b()).out)

    def test_mtp_pays_needs_two_percent_over_lookup(self):
        def gate(on):
            return attempt_runs_5b(on=dict.fromkeys(THREE, on))

        self.assertTrue(m.judge(gate(102.0), "5b")["mtp_pays"])  # ratio 1.02 exactly
        self.assertFalse(m.judge(gate(101.99), "5b")["mtp_pays"])  # ratio 1.0199
        self.assertAlmostEqual(m.judge(gate(102.0), "5b")["mtp_gain"], 1.02)
        self.assertFalse(m.judge(gate(100.0), "5b")["mtp_pays"])
        self.assertIsNone(m.judge(attempt_runs_5b()[:-6], "5b")["mtp_pays"])  # no lookup runs: nothing to judge
        out = flat(Run(self, gate(102.0) + [aggregate_5b()] + others()).out)
        self.assertIn("MTP pays (on at least 1.02 times lookup): record True, recomputed True (equal)", out)
        self.assertIn("mtp_gain): record 1.2000, recomputed 1.0200 (DIFFERENT)", out)
        out = flat(Run(self, gate(101.99) + [aggregate_5b(mtp_pays=False)] + others()).out)
        self.assertIn("record False, recomputed False (equal)", out)
        self.assertIn("recomputed 1.0199", out)
        self.assertIn("| write | 100.00 | 100.00 | 101.99 | 1.0199 | 1.0199 |", out)
        self.assertIn("Lookup median of six: record 100.0000, recomputed 100.0000 (equal)", out)

    def test_an_attempt_whose_on_runs_never_drafted_with_mtp_fails_unless_mtp_was_not_a_source(self):
        def attempt(sources=None):
            runs = attempt_runs_5b()
            for r in runs:
                if r["mode"] == "on":
                    r["figures"]["spec_drafted_mtp"] = 0
                    r["figures"]["spec_accepted_mtp"] = 0
                    if sources is not None:
                        r["sources"] = sources
            return runs

        never = "never drafted with MTP"
        # MTP is a source (the key's absence is the default's) and nothing drafted with it: a problem, and a fail.
        for sources in (None, ["lookup", "mtp"]):
            j = m.judge(attempt(sources), "5b")
            self.assertTrue(any(never in p for p in j["problems"]), sources)
            self.assertEqual(j["status"], "fail")
        self.assertIn("recomputed status fail (DIFFERENT)", Run(self, attempt() + [aggregate_5b()] + others()).out)
        # The fallback: the template's sources = ["lookup"]. No problem; judged on done-whens 1, 2, 3 and 6.
        j = m.judge(attempt(["lookup"]), "5b")
        self.assertFalse(any(never in p for p in j["problems"]))
        self.assertEqual(j["status"], "pass")
        # One "on" run that did draft with MTP is enough where MTP is a source.
        runs = attempt()
        next(r for r in runs if r["mode"] == "on" and r["figures"]["prompt"] == "write")["figures"]["spec_drafted_mtp"] = 5
        self.assertFalse(any(never in p for p in m.judge(runs, "5b")["problems"]))
        # 5a never asks.
        self.assertFalse(any(never in p for p in m.judge(attempt(), "5a")["problems"]))

    def test_mtp_32gb_fails_when_the_mtp_was_not_loaded_or_the_host_bytes_did_not_grow_by_its_experts(self):
        def judged(**on):
            rec = copy.deepcopy(MTP32)
            rec["runs"][1].update(on)
            return m.judge_32gb(rec)

        self.assertEqual(judged(), "pass")
        self.assertEqual(judged(mtp_vram_bytes=0), "fail")
        self.assertEqual(judged(mtp_vram_bytes=None), "fail")
        self.assertEqual(judged(host_bytes=MTP32["runs"][0]["host_bytes"]), "fail")  # 5a's bytes: no growth
        self.assertEqual(judged(host_bytes=MTP32["runs"][1]["host_bytes"] + 1), "fail")
        self.assertEqual(judged(host_bytes=None), "fail")
        self.assertIsNone(judged(pool_gib=None, mtp_vram_bytes=0))  # not measured comes first
        self.assertEqual(judged(pool_gib=1), "ask-owner")
        rec = copy.deepcopy(MTP32)
        rec["runs"][1]["mtp_vram_bytes"] = 0
        self.assertIn("Status pass; recomputed fail (DIFFERENT)", flat(Run(self, full_5b() + [stamp(rec)]).out))
    def test_a_lookup_run_that_drafted_with_mtp_is_a_mode_problem(self):
        runs = attempt_runs_5b()
        next(r for r in runs if r["mode"] == "lookup")["figures"]["spec_drafted_mtp"] = 12
        r = Run(self, runs + [aggregate_5b()] + others())
        self.assertIn('so the "lookup" process drafted with MTP', r.out)
        self.assertIn("recomputed status fail (DIFFERENT)", r.out)

    def test_spec_figures_split_by_source(self):
        out = Run(self, full_5b() + [stamp(LONG5)]).out
        self.assertIn("| write-1234-on | on | 250 | 190 | 0.760 | 150 | 70 | 0.467 |", out)
        self.assertIn("| write-1234-lookup | lookup | 420 | 300 | 0.714 | 0 | 0 | - |", out)
        self.assertNotIn("| write-1234-off |", out.split("Draft counts by source")[1].split("###")[0])
        self.assertIn("lookup 190 of 250 (0.760), mtp 70 of 150 (0.467), probes 2", out)
        out = flat(out)
        self.assertIn('Window comparison (report only): "on" (the default window) 95.00 tok/s, "on-full-window" 95.40 tok/s, '
                      "full / default 1.0042.", out)
        self.assertIn("MTP accept rate, default window: 70 of 150 drafted (0.467).", out)
        self.assertIn("MTP accept rate, full window: 76 of 155 drafted (0.490).", out)

    def test_2_mtp_records_are_reported_and_checked(self):
        out = flat(Run(self, full_5b() + [stamp(T2MTP), stamp(T2MTPGPU)]).out)
        self.assertIn("2-mtp: status pass, form taps, 18 taps; worst tap mtp_ffn_out error 3.1e-06 against bound 1e-05 "
                      "(recomputed within bound: pass).", out)
        self.assertIn("2-mtp-gpu: status pass, 15 positions, top-1 equal at 15, 0 differences at a near tie (every position "
                      "equal or a near tie: pass)", out)
        bad = copy.deepcopy(T2MTPGPU)
        bad["top1_equal"] = 13
        self.assertIn("(every position equal or a near tie: FAIL)", flat(Run(self, full_5b() + [stamp(bad)]).out))
        # The runner names the inputs it expected; one with no verdict is reported, and a record that lists all four passing
        # inputs is clean.
        whole = stamp(dict(T2MTP, expected_inputs=["a", "b"], inputs=[{"input": "a", "status": "pass"}, {"input": "b", "status": "pass"}]))
        self.assertIn("2 of 2 expected inputs passed; a missing input fails the record (recomputed: pass).", flat(Run(self, full_5b() + [whole]).out))
        part = stamp(dict(T2MTP, expected_inputs=["a", "b"], inputs=[{"input": "a", "status": "pass"}, {"input": "b", "status": "missing"}]))
        self.assertIn("1 of 2 expected inputs passed (not passed: b); a missing input fails the record (recomputed: fail).",
                      flat(Run(self, full_5b() + [part]).out))
        out = Run(self, full_5b()).out
        self.assertIn("**2-mtp: missing**", out)
        self.assertIn("**2-mtp-gpu: missing**", out)


if __name__ == "__main__":
    unittest.main()
