"""Tests for tools/py/request_quality.py over synthetic logs. Standard library only.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v

The module is loaded by path (python -I keeps the script's directory off sys.path). The fixtures use the field names in
docs/server.md (the Logs section); they hold no keys, hosts or request text.
"""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "request_quality.py"
spec = importlib.util.spec_from_file_location("request_quality", MODULE_PATH)
rq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rq)


def req(day="2026-11-01", **kw):
    line = {
        "time": f"{day}T10:00:00Z",
        "id": "r1",
        "endpoint": "/v1/chat/completions",
        "model": "m",
        "status": 200,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "reasoning_tokens": 5,
        "tool_calls": 0,
        "malformed_calls": 0,
        "budget_closed": False,
        "reused_tokens": 0,
        "finish": "stop",
        "error_kind": None,
        "total_ms": 1000,
    }
    line.update(kw)
    return line


def write(path, lines, tail=""):
    text = "".join(json.dumps(x) + "\n" for x in lines) + tail
    Path(path).write_text(text, encoding="utf-8")


class Normalize(unittest.TestCase):
    def test_finish_names_of_both_apis_fold_together(self):
        n = rq.normalize_finish
        self.assertEqual(n("stop"), "stop")
        self.assertEqual(n("end_turn"), "stop")
        self.assertEqual(n("stop_sequence"), "stop")
        self.assertEqual(n("length"), "length")
        self.assertEqual(n("max_tokens"), "length")
        self.assertEqual(n("tool_calls"), "tool_calls")
        self.assertEqual(n("tool_use"), "tool_calls")
        self.assertEqual(n("cancelled"), "cancelled")
        self.assertEqual(n(None), "none")
        self.assertEqual(n(""), "none")
        self.assertEqual(n("something_new"), "something_new")


class Reading(unittest.TestCase):
    def test_partial_last_line_garbage_and_lifecycle_lines_are_tolerated(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "artemis-2026-11-01.jsonl"
            write(
                p,
                [req(id="a"), {"time": "2026-11-01T09:00:00Z", "event": "start", "version": "x"}, req(id="b")],
                tail='{"time": "2026-11-01T11:00:00Z", "id": "c", "endp',
            )
            with open(p, "a", encoding="utf-8") as f:
                f.write("\n")
                f.write("not json\n[1, 2]\n")
            records, skipped = rq.read_requests([p])
            self.assertEqual([r["id"] for r in records], ["a", "b"])
            self.assertEqual(skipped, 3)

    def test_a_directory_is_searched_for_log_files_below_it(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "run" / "logs").mkdir(parents=True)
            write(root / "artemis-2026-11-01.jsonl", [req(id="a")])
            write(root / "run" / "logs" / "artemis-2026-11-02.jsonl", [req(id="b", day="2026-11-02")])
            write(root / "other.txt", [req(id="z")])
            records, _ = rq.read_requests([root])
            self.assertEqual(sorted(r["id"] for r in records), ["a", "b"])

    def test_a_file_given_twice_or_with_its_folder_counts_once(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "artemis-2026-11-01.jsonl"
            write(p, [req(id="a"), req(id="b")])
            records, _ = rq.read_requests([p, p, root, root / "." / p.name])
            self.assertEqual([r["id"] for r in records], ["a", "b"])

    def test_an_unreadable_file_is_reported_and_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write(root / "artemis-2026-11-01.jsonl", [req(id="a")])
            (root / "sub").mkdir()
            (root / "sub" / "artemis-2026-11-02.jsonl").mkdir()  # a folder with a log's name cannot be read
            unreadable = []
            records, _ = rq.read_requests([root], unreadable)
            self.assertEqual([r["id"] for r in records], ["a"])
            self.assertEqual(len(unreadable), 1)
            self.assertIn("artemis-2026-11-02.jsonl", unreadable[0][0])

    def test_since_keeps_the_given_day_and_later(self):
        records = [req(id="a", day="2026-10-30"), req(id="b", day="2026-11-01"), req(id="c", day="2026-11-02")]
        kept = rq.since(records, "2026-11-01")
        self.assertEqual([r["id"] for r in kept], ["b", "c"])

    def test_since_drops_lines_with_an_unknown_day(self):
        no_time = req(id="u")
        del no_time["time"]
        odd = req(id="v", time="yesterday")
        kept = rq.since([no_time, odd, req(id="a", day="2026-11-02")], "2026-11-01")
        self.assertEqual([r["id"] for r in kept], ["a"])

    def test_a_line_with_no_time_belongs_to_the_unknown_day(self):
        r = req(id="a")
        del r["time"]
        self.assertEqual(rq.day_of(r), "unknown")


class Summary(unittest.TestCase):
    def test_counts_shares_and_totals(self):
        records = [
            req(id="1", finish="stop", prompt_tokens=100, completion_tokens=10, reused_tokens=60),
            req(id="2", finish="length", prompt_tokens=200, completion_tokens=50, reused_tokens=100, total_ms=9000),
            req(id="3", endpoint="/v1/messages", model="n", finish="tool_use", tool_calls=2, malformed_calls=0),
            req(id="4", finish="tool_calls", tool_calls=1, malformed_calls=1, budget_closed=True),
            req(id="5", finish="stop", tool_calls=0, malformed_calls=2),
            req(id="6", finish="cancelled", status=499),
            req(id="7", status=503, finish=None, error_kind="overloaded", completion_tokens=None),
        ]
        s = rq.summarize(records)
        self.assertEqual(s["requests"], 7)
        self.assertEqual(s["finish"], {"stop": 2, "length": 1, "tool_calls": 2, "cancelled": 1, "none": 1})
        self.assertAlmostEqual(s["length_share"], 1 / 7)
        self.assertEqual(s["tool_call_requests"], 3)
        self.assertEqual(s["malformed_calls"], 3)
        self.assertEqual(s["malformed_requests"], 2)
        self.assertAlmostEqual(s["malformed_share"], 2 / 3)
        self.assertEqual(s["budget_closed"], 1)
        self.assertEqual(s["errors"], 1)  # the 499 is a client that left, counted under cancelled
        self.assertEqual(s["by_endpoint"], {"/v1/chat/completions": 6, "/v1/messages": 1})
        self.assertEqual(s["by_model"], {"m": 6, "n": 1})
        self.assertEqual(s["prompt_tokens"], 100 + 200 + 100 * 5)
        self.assertEqual(s["completion_tokens"], 10 + 50 + 20 * 4)
        self.assertAlmostEqual(s["reuse_rate"], 160 / 800)
        self.assertEqual(s["slowest"][0]["id"], "2")

    def test_a_4xx_status_without_an_error_kind_is_an_error_but_499_is_not(self):
        s = rq.summarize([req(status=400, error_kind=None), req(status=404), req(status=499), req(status=200)])
        self.assertEqual(s["errors"], 2)

    def test_old_lines_without_the_newer_fields_still_count(self):
        old = {
            "time": "2026-11-01T10:00:00Z",
            "id": "o",
            "endpoint": "/v1/messages",
            "status": 200,
            "finish": "end_turn",
        }
        s = rq.summarize([old, dict(old, id="p", finish="max_tokens", future_field={"x": 1})])
        self.assertEqual(s["requests"], 2)
        self.assertEqual(s["finish"], {"stop": 1, "length": 1})
        self.assertIsNone(s["reuse_rate"])
        self.assertIsNone(s["malformed_share"])
        self.assertEqual(s["prompt_tokens"], 0)
        self.assertEqual(s["slowest"], [])

    def test_the_slowest_list_is_capped_and_sorted(self):
        records = [req(id=str(i), total_ms=i * 100) for i in range(1, 12)]
        s = rq.summarize(records)
        self.assertEqual([x["id"] for x in s["slowest"]], ["11", "10", "9", "8", "7"])

    def test_wrongly_typed_values_do_not_stop_the_report(self):
        bad = req(prompt_tokens="many", total_ms="slow", tool_calls=None, malformed_calls=True, status="ok")
        s = rq.summarize([bad])
        self.assertEqual(s["requests"], 1)
        self.assertEqual(s["prompt_tokens"], 0)

    def test_long_reasoning_is_counted(self):
        s = rq.summarize([req(reasoning_tokens=rq.LONG_REASONING_TOKENS), req(reasoning_tokens=10)])
        self.assertEqual(s["long_reasoning"], 1)


class Flags(unittest.TestCase):
    def test_a_share_above_its_threshold_is_flagged(self):
        records = [req(finish="length") for _ in range(10)] + [req() for _ in range(90)]
        flags = rq.flags_for(rq.summarize(records))
        self.assertEqual([f["name"] for f in flags], ["length_share"])
        self.assertAlmostEqual(flags[0]["value"], 0.10)
        self.assertEqual(flags[0]["threshold"], rq.THRESHOLDS["length_share"])

    def test_shares_at_or_below_the_threshold_are_not_flagged(self):
        records = [req(finish="length")] + [req() for _ in range(19)]
        self.assertEqual(rq.flags_for(rq.summarize(records)), [])

    def test_a_small_sample_is_not_flagged(self):
        records = [req(finish="length") for _ in range(rq.MIN_FLAG_REQUESTS - 1)]
        self.assertEqual(rq.flags_for(rq.summarize(records)), [])

    def test_malformed_budget_and_error_shares_are_flagged(self):
        records = (
            [req(tool_calls=1, malformed_calls=1, finish="tool_calls") for _ in range(5)]
            + [req(tool_calls=1, finish="tool_calls") for _ in range(5)]
            + [req(budget_closed=True) for _ in range(5)]
            + [req(status=500, error_kind="engine") for _ in range(5)]
        )
        names = {f["name"] for f in rq.flags_for(rq.summarize(records))}
        self.assertEqual(names, {"malformed_share", "budget_closed_share", "error_share"})


class Command(unittest.TestCase):
    def run_main(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = rq.main(argv)
        return code, out.getvalue()

    def test_report_per_day_and_overall_and_json_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write(root / "artemis-2026-11-01.jsonl", [req(id="a"), req(id="b", finish="length")])
            write(root / "artemis-2026-11-02.jsonl", [req(id="c", day="2026-11-02")])
            out_json = root / "out.json"
            code, text = self.run_main([str(root), "--json", str(out_json)])
            self.assertEqual(code, 0)
            self.assertIn("2026-11-01", text)
            self.assertIn("2026-11-02", text)
            self.assertIn("overall", text)
            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(data["overall"]["requests"], 3)
            self.assertEqual(sorted(data["days"]), ["2026-11-01", "2026-11-02"])
            self.assertEqual(data["days"]["2026-11-01"]["requests"], 2)
            self.assertIn("thresholds", data)

    def test_since_limits_the_report(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write(root / "artemis-2026-11-01.jsonl", [req(id="a")])
            write(root / "artemis-2026-11-02.jsonl", [req(id="c", day="2026-11-02")])
            out_json = root / "out.json"
            self.run_main([str(root), "--since", "2026-11-02", "--json", str(out_json)])
            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(data["overall"]["requests"], 1)

    def test_a_log_with_no_requests_reports_that_and_succeeds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write(root / "artemis-2026-11-01.jsonl", [{"time": "2026-11-01T09:00:00Z", "event": "start"}])
            code, text = self.run_main([str(root)])
            self.assertEqual(code, 0)
            self.assertIn("no request lines", text)

    def test_an_unreadable_file_is_named_and_the_rest_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write(root / "artemis-2026-11-01.jsonl", [req(id="a")])
            (root / "artemis-2026-11-02.jsonl").mkdir()
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code, text = self.run_main([str(root)])
            self.assertEqual(code, 0)
            self.assertIn("overall: 1 requests", text)
            self.assertIn("artemis-2026-11-02.jsonl: cannot read", err.getvalue())
            self.assertIn("skipped unreadable files: 1", text)

    def test_a_missing_path_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = rq.main([str(Path(d) / "nope")])
            self.assertEqual(code, 2)
            self.assertIn("nope", err.getvalue())

    def test_a_bad_since_date_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                code = rq.main([d, "--since", "yesterday"])
            self.assertEqual(code, 2)

    def test_flags_are_printed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write(root / "artemis-2026-11-01.jsonl", [req(finish="length") for _ in range(12)])
            code, text = self.run_main([str(root)])
            self.assertEqual(code, 0)
            self.assertIn("FLAG", text)
            self.assertIn("length_share", text)


if __name__ == "__main__":
    unittest.main()
