"""Tests for tools/py/m4_soak.py over synthetic logs. Standard library only.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v

The module is loaded by path (python -I keeps the script's directory off sys.path). The fixtures copy the shapes in
crates/server/src/logs.rs (RequestLine and LifecycleLine) and the guard's `yyyy-MM-dd HH:mm:ss <message>` lines; they
hold no keys, hosts or request text.
"""

import contextlib
import importlib.util
import io
import json
import os
import statistics
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "m4_soak.py"
spec = importlib.util.spec_from_file_location("m4_soak", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules["m4_soak"] = m
spec.loader.exec_module(m)

UTC = timezone.utc
T0 = datetime(2026, 11, 3, 12, 0, 0, tzinfo=UTC)
GIB = 1 << 30


def at(seconds):
    return T0 + timedelta(seconds=seconds)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def as_utc(naive):
    """A stand-in for the local-time conversion: the guard's clock read as UTC, never ambiguous."""
    return naive.replace(tzinfo=UTC), False


def req(seconds, **kw):
    line = {
        "time": iso(at(seconds)),
        "id": f"req_{seconds}",
        "endpoint": "/v1/chat/completions",
        "model": "qwen",
        "status": 200,
        "prompt_tokens": 100,
        "stream": True,
        "reused_tokens": 0,
        "cause": "none",
        "completion_tokens": 10,
        "decode_tok_s": None,
        "decode_ms": None,
        "decode_tokens": None,
        "decode_misses": None,
        "finish": "stop",
        "error_kind": None,
        "error_message": None,
    }
    line.update(kw)
    return line


def life(offset, event, **kw):
    line = {"time": iso(at(offset)), "event": event}
    line.update(kw)
    return line


def stats_body(**kw):
    body = {
        "snapshots_held": 0,
        "pool_bytes_used": 0,
        "pool_bytes_limit": 4 * GIB,
        "restores": 0,
        "restore_bytes": 0,
        "reused_tokens": 0,
        "prefilled_tokens": 0,
        "evicted_protected": 0,
        "evicted_unprotected": 0,
        "park_errors": 0,
        "chunks_refused": 0,
        "loads": 1,
        "device_losses": 0,
        "after_done_errors": 0,
        "parked_snapshots": 0,
        "cache": None,
    }
    body.update(kw)
    return body


def stats(seconds, body=None, write_failures=0, uptime=0):
    return life(
        seconds,
        "stats",
        uptime_s=uptime,
        age_s=1.5 if body is not None else None,
        log_write_failures=write_failures,
        stats=body,
    )


def glines(*rows):
    """Guard lines from (seconds after T0, message) pairs, in the guard's local-time format."""
    return [f"{at(s).strftime('%Y-%m-%d %H:%M:%S')} {msg}" for s, msg in rows]


def guard(*rows):
    return m.parse_guard_lines(glines(*rows), to_utc=as_utc)


@contextlib.contextmanager
def time_zone(rule):
    old = os.environ.get("TZ")
    os.environ["TZ"] = rule
    time.tzset()
    try:
        yield
    finally:
        if old is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = old
        time.tzset()


CHICAGO = "CST6CDT,M3.2.0,M11.1.0"  # a POSIX rule: no tzdata needed


class Explaining(unittest.TestCase):
    def test_the_module_imports_without_running_main(self):
        self.assertTrue(callable(m.main))
        self.assertEqual(m.__name__, "m4_soak")

    def test_a_503_with_its_own_message_is_explained(self):
        f = req(
            100,
            status=503,
            error_kind="overloaded_error",
            error_message="the request queue is full",
        )
        why = m.explain(f, [], [])
        self.assertIsNotNone(why)
        self.assertIn("the request queue is full", why)

    def test_the_engine_stopped_needs_a_line_or_a_window(self):
        f = req(
            100,
            status=500,
            error_kind="server_error",
            error_message="the engine stopped",
        )
        self.assertIsNone(m.explain(f, [], []))
        self.assertIsNone(m.explain(f, [life(0, "start")], []))
        self.assertIn("exit", m.explain(f, [life(90, "exit", reason="stall")], []))
        pause = guard((50, "state paused: game"), (300, "state available"))
        self.assertIsNotNone(m.explain(f, [], pause))
        restart = guard((60, "artemis exited (code 3)"), (130, "started artemis (pid 7)"))
        self.assertIsNotNone(m.explain(f, [], restart))
        later = guard((200, "state paused: game"), (300, "state available"))
        self.assertIsNone(m.explain(f, [], later))

    def test_the_prompt_could_not_be_built_points_elsewhere_too(self):
        f = req(
            100,
            status=500,
            error_kind="server_error",
            error_message="the prompt could not be built",
        )
        self.assertIsNone(m.explain(f, [], []))
        self.assertIsNotNone(m.explain(f, [life(95, "panic", thread="artemis-http")], []))

    def test_a_500_after_a_stall_line_within_60_s_is_explained(self):
        # The message points elsewhere, so only the lifecycle line can explain it.
        g = req(100, status=500, error_kind="server_error", error_message="the engine stopped")
        for seconds in (40, 99, 100):
            with self.subTest(seconds=seconds):
                self.assertIn("stall", m.explain(g, [life(seconds, "stall", doing="request")], []))

    def test_a_500_after_reload_failed_is_explained(self):
        g = req(100, status=500, error_kind="server_error", error_message="the engine stopped")
        why = m.explain(g, [life(70, "reload_failed", message="no device")], [])
        self.assertIn("reload_failed", why)
        why = m.explain(g, [life(70, "load_failed", message="no memory")], [])
        self.assertIn("load_failed", why)

    def test_a_failure_61_s_after_the_line_is_unexplained(self):
        def failure(seconds):
            return req(seconds, status=500, error_kind="server_error", error_message="the engine stopped")

        for event in ("device_lost", "unload", "load_failed", "reload_failed", "stall", "panic", "exit"):
            with self.subTest(event=event):
                self.assertIsNotNone(m.explain(failure(160), [life(100, event)], []))
                self.assertIsNone(m.explain(failure(161), [life(100, event)], []))

    def test_a_cause_logged_during_the_request_explains_it(self):
        # `time` is the arrival; the request ran 5 s (`total_ms`), so a panic at +3 s (mid-decode) is its cause.
        def failure(**kw):
            return req(100, status=500, error_kind="server_error", error_message="the engine stopped", **kw)

        ran = failure(total_ms=5000.0)
        for seconds in (103, 105, 106):
            with self.subTest(seconds=seconds):
                self.assertIn("panic", m.explain(ran, [life(seconds, "panic", thread="artemis-engine")], []))
        # After the request's end plus the second its whole-second stamps can lose, a line does not explain it.
        self.assertIsNone(m.explain(ran, [life(107, "panic", thread="artemis-engine")], []))
        # The window before the arrival is as it was.
        self.assertIsNotNone(m.explain(ran, [life(40, "stall", doing="request")], []))
        self.assertIsNone(m.explain(ran, [life(39, "stall", doing="request")], []))
        # A line with no duration (an older log) keeps today's window: nothing after the arrival explains it.
        self.assertIsNone(m.explain(failure(), [life(103, "panic", thread="artemis-engine")], []))
        self.assertIsNone(m.explain(failure(total_ms=None), [life(103, "panic", thread="artemis-engine")], []))

    def test_other_lifecycle_lines_do_not_explain(self):
        g = req(100, status=500, error_kind="server_error", error_message="the engine stopped")
        for line in (life(95, "start"), life(95, "load", seconds=9.0), stats(95, stats_body())):
            self.assertIsNone(m.explain(g, [line], []))


class Requests(unittest.TestCase):
    def test_a_broken_stream_is_a_failure(self):
        broken = req(10, status=200, error_kind="server_error", error_message="the engine stopped")
        fine = req(11)
        self.assertEqual(m.failures([broken, fine]), [broken])
        self.assertEqual(m.failures([req(12, status=503), req(13, status=500)]), [req(12, status=503), req(13, status=500)])

    def test_client_errors_and_cancellations_are_listed_not_judged(self):
        bad = req(10, status=400, error_kind="invalid_request_error", error_message="bad field")
        gone = req(11, status=404, error_kind="not_found_error", error_message="no such model")
        cancelled = req(12, status=499, finish="cancelled")
        self.assertEqual(m.failures([bad, gone, cancelled]), [])
        self.assertEqual(m.client_errors([bad, gone, cancelled]), [bad, gone])
        self.assertEqual(m.cancellations([bad, gone, cancelled]), [cancelled])

    def test_status_counts(self):
        rows = [
            req(1),
            req(2),
            req(3, status=400, error_kind="invalid_request_error"),
            req(4, status=503, error_kind="overloaded_error"),
            req(5, status=200, error_kind="server_error"),
            req(6, status=499, finish="cancelled"),
        ]
        c = m.status_counts(rows)
        self.assertEqual(c["by_status"], {"200": 3, "400": 1, "499": 1, "503": 1})
        self.assertEqual(
            c["by_error_kind"],
            {"invalid_request_error": 1, "overloaded_error": 1, "server_error": 1},
        )
        self.assertEqual(c["broken_streams"], 1)
        self.assertEqual(c["cancellations"], 1)

    def test_reuse_by_cause(self):
        rows = [
            req(1, cause="new_conversation", prompt_tokens=100, reused_tokens=0),
            req(2, cause="new_input", prompt_tokens=300, reused_tokens=200),
            req(3, cause="new_input", prompt_tokens=100, reused_tokens=100),
            req(4, cause="reasoning_dropped", prompt_tokens=50, reused_tokens=25),
            req(5, cause=None, prompt_tokens=None, reused_tokens=None, status=503),
        ]
        r = m.reuse_by_cause(rows)
        self.assertEqual(set(r), {"new_conversation", "new_input", "reasoning_dropped"})
        self.assertEqual(r["new_input"]["requests"], 2)
        self.assertEqual(r["new_input"]["reused_tokens"], 300)
        self.assertEqual(r["new_input"]["prompt_tokens"], 400)
        self.assertAlmostEqual(r["new_input"]["rate"], 0.75)
        self.assertEqual(r["new_conversation"]["rate"], 0.0)

    def test_decode_figures_take_requests_of_256_tokens_or_more(self):
        rows = [
            req(
                i,
                completion_tokens=300,
                decode_tok_s=90.0 + i,
                decode_tokens=300,
                decode_ms=300 / (100.0 + i) * 1000,
            )
            for i in range(5)
        ]
        rows.append(req(9, completion_tokens=10, decode_tok_s=1000.0, decode_tokens=9, decode_ms=9.0))
        d = m.decode_figures(rows)
        self.assertEqual(d["requests"], 5)
        self.assertAlmostEqual(d["tok_s"]["median"], 92.0)
        self.assertAlmostEqual(d["tok_s"]["p10"], 90.4)
        self.assertAlmostEqual(d["tokens_over_ms"]["median"], 102.0)
        self.assertAlmostEqual(d["tokens_over_ms"]["p10"], 100.4)
        # The same cut, by the statistics module's own inclusive method.
        self.assertAlmostEqual(
            d["tok_s"]["p10"],
            statistics.quantiles([90.0 + i for i in range(5)], n=10, method="inclusive")[0],
        )
        self.assertEqual(m.decode_figures(rows, min_completion=5)["requests"], 6)

    def test_decode_figures_leave_out_broken_streams_and_keep_zero(self):
        rows = [req(i, completion_tokens=300, decode_tok_s=float(v)) for i, v in enumerate((0.0, 50.0, 100.0))]
        broken = req(9, completion_tokens=300, decode_tok_s=1000.0, error_kind="server_error", error_message="x")
        d = m.decode_figures(rows + [broken])
        self.assertEqual(d["requests"], 3)
        self.assertEqual(d["tok_s"]["n"], 3)
        self.assertEqual(d["tok_s"]["median"], 50.0)
        self.assertEqual(d["tok_s"]["p10"], 10.0)

    def test_decode_figures_are_none_below_two_points(self):
        one = [req(1, completion_tokens=300, decode_tok_s=90.0, decode_tokens=299, decode_ms=3000.0)]
        d = m.decode_figures(one)
        self.assertEqual(d["requests"], 1)
        self.assertIsNone(d["tok_s"])
        self.assertIsNone(d["tokens_over_ms"])
        d = m.decode_figures([])
        self.assertIsNone(d["tok_s"])
        # A line with no decode_ms adds nothing to the second figure.
        rows = [req(i, completion_tokens=300, decode_tok_s=90.0 + i) for i in range(3)]
        d = m.decode_figures(rows)
        self.assertIsNotNone(d["tok_s"])
        self.assertIsNone(d["tokens_over_ms"])

    def test_spec_figures_read_spec_lines_and_skip_old_ones(self):
        # The six keys: counts and a rate, no text. Lines from before speculation have none of them.
        def spec(i, steps, plain, drafted, accepted, **kw):
            rate = accepted / drafted if drafted else None
            return req(
                i,
                spec_steps=steps,
                spec_plain_steps=plain,
                spec_drafted=drafted,
                spec_accepted=accepted,
                spec_accept_rate=rate,
                spec_replays=steps,
                **kw,
            )

        rows = [
            spec(1, 100, 100, 200, 100, decode_tok_s=100.0),
            spec(2, 100, 300, 200, 150, decode_tok_s=110.0),
            spec(3, 50, 50, 100, 90, decode_tok_s=120.0),
            spec(4, 10, 0, 0, 0, decode_tok_s=130.0),  # steps ran but nothing was drafted: no rate
            req(5, spec_steps=0, spec_plain_steps=400, spec_accept_rate=None, decode_tok_s=500.0),  # all plain: in the share only
            req(6, spec_steps=None, spec_accept_rate=None, decode_tok_s=500.0),  # speculation off or never decoded
            req(7, decode_tok_s=500.0),  # an old line with no spec_* keys at all
            spec(8, 20, 5, 40, 30, decode_tok_s=999.0, error_kind="server_error", error_message="x"),
        ]
        f = m.spec_figures(rows)
        # The broken stream (request 8) is left out of all three figures, tok_s among them.
        self.assertEqual((f["requests"], f["verified_requests"]), (5, 4))
        self.assertEqual(f["accept_rate"]["n"], 3)
        self.assertAlmostEqual(f["accept_rate"]["median"], 0.75)
        self.assertAlmostEqual(
            f["accept_rate"]["p10"], statistics.quantiles([0.5, 0.75, 0.9], n=10, method="inclusive")[0]
        )
        # 100 + 100 + 50 + 10 steps; 100 + 300 + 50 + 0 plain steps, and request 5's 400 plain steps (it never verified,
        # but it ran plain steps, so it counts toward the share and toward nothing else).
        self.assertAlmostEqual(f["plain_share"], 850 / (260 + 850))
        self.assertEqual(f["tok_s"]["n"], 4)
        self.assertAlmostEqual(f["tok_s"]["median"], 115.0)
        # Old logs: no figures at all, and no KeyError.
        self.assertIsNone(m.spec_figures([req(1), req(2, spec_steps=None), req(3, spec_steps=0)]))
        self.assertIsNone(m.spec_figures([]))

    def test_spec_figures_split_by_source(self):
        # 5b's four per-source keys: the rates are pooled counts per source; old lines (no such keys) add nothing.
        def spec(i, dl, al, dm, am, **kw):
            return req(
                i, spec_steps=10, spec_plain_steps=5, spec_drafted=dl + dm, spec_accepted=al + am,
                spec_accept_rate=(al + am) / (dl + dm) if dl + dm else None, spec_replays=1,
                spec_drafted_lookup=dl, spec_accepted_lookup=al, spec_drafted_mtp=dm, spec_accepted_mtp=am, **kw,
            )  # fmt: skip

        rows = [
            spec(1, 100, 90, 50, 20, decode_tok_s=100.0),
            spec(2, 100, 80, 50, 10, decode_tok_s=100.0),
            spec(3, 0, 0, 0, 0, decode_tok_s=100.0),  # nothing drafted by either
            spec(4, 40, 30, 10, 5, error_kind="server_error", error_message="x"),  # a broken stream: left out
            req(5, spec_steps=10, spec_plain_steps=5, spec_drafted=8, spec_accepted=4, spec_accept_rate=0.5,
                decode_tok_s=100.0),  # a 5a line: no per-source keys
        ]  # fmt: skip
        src = m.spec_figures(rows)["sources"]
        self.assertEqual(src["lookup"]["drafted"], 200)
        self.assertEqual(src["lookup"]["accepted"], 170)
        self.assertAlmostEqual(src["lookup"]["accept_rate"], 0.85)
        self.assertEqual((src["mtp"]["drafted"], src["mtp"]["accepted"]), (100, 30))
        self.assertAlmostEqual(src["mtp"]["accept_rate"], 0.3)
        # A source that drafted nothing has no rate (not 0), and an all-5a log has none to split.
        only_lookup = [spec(1, 10, 5, 0, 0, decode_tok_s=100.0)]
        s = m.spec_figures(only_lookup)["sources"]
        self.assertAlmostEqual(s["lookup"]["accept_rate"], 0.5)
        self.assertIsNone(s["mtp"]["accept_rate"])
        old = m.spec_figures([rows[4]])["sources"]
        self.assertEqual((old["lookup"]["drafted"], old["mtp"]["drafted"]), (0, 0))
        # They reach the digest and the final record.
        none = {"lines": 0, "panics": [], "present": False}
        rep = m.report(rows, [], [], none, at(0), at(600), 5)
        text = m.render(rep)
        self.assertIn("lookup accept rate (170 of 200 drafted): 0.850", text)
        self.assertIn("mtp accept rate (30 of 100 drafted): 0.300", text)
        record = m.results_record(rep, at(600))
        self.assertAlmostEqual(record["spec_accept_rate_lookup"], 0.85)
        self.assertAlmostEqual(record["spec_accept_rate_mtp"], 0.3)
        rep = m.report([req(10)], [], [], none, at(0), at(600), 5)
        record = m.results_record(rep, at(600))
        self.assertIsNone(record["spec_accept_rate_lookup"])
        self.assertIsNone(record["spec_accept_rate_mtp"])

    def test_spec_figures_reach_the_report_the_digest_and_the_final_record(self):
        none = {"lines": 0, "panics": [], "present": False}
        rows = [
            req(10 + i, spec_steps=100, spec_plain_steps=300, spec_drafted=200, spec_accepted=100 + 10 * i,
                spec_accept_rate=(100 + 10 * i) / 200, spec_replays=90, decode_tok_s=100.0 + i)
            for i in range(3)
        ]  # fmt: skip
        rep = m.report(rows, [], [], none, at(0), at(600), 5)
        text = m.render(rep)
        self.assertIn("Speculation (report only), over the 3 speculating requests", text)
        self.assertIn("plain-step share: 0.750", text)
        self.assertIn("that verified): median 0.55", text)
        record = m.results_record(rep, at(600))
        self.assertAlmostEqual(record["spec_accept_rate_median"], 0.55)
        self.assertAlmostEqual(record["spec_accept_rate_p10"], 0.51)
        self.assertAlmostEqual(record["spec_plain_share"], 0.75)
        # Without speculation lines the digest says so and the record keys are null.
        rep = m.report([req(10)], [], [], none, at(0), at(600), 5)
        self.assertIn("no request line has any speculation steps", m.render(rep))
        record = m.results_record(rep, at(600))
        for key in ("spec_accept_rate_median", "spec_accept_rate_p10", "spec_plain_share"):
            self.assertIn(key, record)
            self.assertIsNone(record[key])

    def test_the_traffic_floor(self):
        ok = [req(i) for i in range(200)]
        noise = [
            req(1, endpoint="/health"),
            req(2, endpoint="/v1/models"),
            req(3, status=503),
            req(4, status=200, error_kind="server_error"),
        ]
        t = m.traffic(ok + noise, qwen_runs=5, available_hours=72.0)
        self.assertEqual(t["generation_requests"], 200)
        self.assertTrue(t["met"])
        t = m.traffic(ok[:-1] + noise, qwen_runs=5, available_hours=72.0)
        self.assertEqual(t["generation_requests"], 199)
        self.assertFalse(t["requests_ok"])
        self.assertFalse(t["met"])
        self.assertFalse(m.traffic(ok, qwen_runs=4, available_hours=72.0)["runs_ok"])
        self.assertFalse(m.traffic(ok, qwen_runs=5, available_hours=71.9)["hours_ok"])
        t = m.traffic(ok + [req(1, endpoint="/v1/messages")], qwen_runs=5, available_hours=80)
        self.assertEqual(t["generation_requests"], 201)


class Lifecycle(unittest.TestCase):
    def test_a_start_without_an_exit_before_it_is_an_unclean_end(self):
        lifecycle = [
            life(0, "start"),
            life(5, "load", seconds=9.0),
            stats(600, stats_body()),
            life(1000, "start"),
            life(1005, "load", seconds=9.0),
        ]
        g = guard((990, "stopped artemis (game pause)"))
        ends = m.unclean_ends(lifecycle, g)
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]["time"], at(1000))
        self.assertEqual(ends[0]["previous"], "stats")
        self.assertIn("game pause", ends[0]["guard"])
        # With no guard line in the window there is no annotation.
        self.assertIsNone(m.unclean_ends(lifecycle, [])[0]["guard"])
        # A guard line from before the previous lifecycle line is not this end's.
        old = guard((300, "stopped artemis (game pause)"))
        self.assertIsNone(m.unclean_ends(lifecycle, old)[0]["guard"])

    def test_an_exit_code_in_the_window_annotates_too(self):
        lifecycle = [life(0, "start"), stats(100, stats_body()), life(500, "start")]
        g = guard((450, "artemis exited (code 3)"))
        self.assertIn("code 3", m.unclean_ends(lifecycle, g)[0]["guard"])

    def test_a_stall_exit_then_start_is_clean(self):
        lifecycle = [
            life(0, "start"),
            life(600, "stall", doing="request", request="req_1", quiet_s=62.0, bound_s=60, ticks=9),
            stats(601, stats_body()),
            life(602, "exit", reason="stall"),
            life(700, "start"),
        ]
        self.assertEqual(m.unclean_ends(lifecycle, []), [])

    def test_stats_gaps_over_the_limit_while_available_are_flagged(self):
        lifecycle = [stats(s, stats_body()) for s in (300, 900, 1800, 2400, 3000)]
        available = [(at(0), at(3600)), (at(10000), at(11000))]
        gaps = m.stats_gaps(lifecycle, available)
        self.assertEqual([(g["from"], g["to"]) for g in gaps], [(at(900), at(1800)), (at(10000), at(11000))])
        self.assertEqual([g["gap_s"] for g in gaps], [900, 1000])
        # Under a longer limit only the empty interval is left.
        self.assertEqual(len(m.stats_gaps(lifecycle, available, limit_s=950)), 1)
        # Nothing available, nothing to flag.
        self.assertEqual(m.stats_gaps([], []), [])

    def test_lifecycle_figures_read_the_last_stats_and_the_parked_ram(self):
        lifecycle = [
            life(0, "start"),
            stats(1, None, write_failures=0),
            life(5, "load", seconds=9.0),
            stats(600, stats_body(park_errors=0, chunks_refused=1), write_failures=0),
            life(700, "unload", cause="idle"),
            stats(700, stats_body(pool_bytes_used=3 * GIB, parked_snapshots=4, park_errors=0, chunks_refused=1)),
            stats(1300, stats_body(park_errors=1, chunks_refused=2), write_failures=1),
            stats(1399, stats_body(park_errors=1, chunks_refused=2), write_failures=1),
            life(1400, "exit", reason="shutdown"),
            life(1500, "start"),
            stats(1600, stats_body(park_errors=0, chunks_refused=5), write_failures=2),
        ]
        f = m.lifecycle_figures(lifecycle)
        self.assertEqual(len(f["per_exit"]), 1)
        self.assertEqual(f["per_exit"][0]["park_errors"], 1)
        self.assertEqual(f["per_exit"][0]["chunks_refused"], 2)
        self.assertEqual(f["per_exit"][0]["log_write_failures"], 1)
        self.assertEqual(f["per_exit"][0]["reason"], "shutdown")
        self.assertEqual(f["at_end"]["chunks_refused"], 5)
        self.assertEqual(f["at_end"]["log_write_failures"], 2)
        self.assertEqual(f["totals"], {"park_errors": 1, "chunks_refused": 7, "log_write_failures": 3})
        self.assertEqual(len(f["parked"]), 1)
        self.assertEqual(f["parked"][0]["pool_bytes_used"], 3 * GIB)
        self.assertEqual(f["parked"][0]["parked_snapshots"], 4)
        self.assertEqual(f["parked"][0]["cause"], "idle")
        self.assertEqual(f["parked_max_bytes"], 3 * GIB)
        events = [t["event"] for t in f["timeline"]]
        self.assertEqual(events, ["start", "load", "unload", "exit", "start"])

    def test_lifecycle_figures_tolerate_stats_null_and_no_stats(self):
        f = m.lifecycle_figures([life(0, "start"), stats(10, None), life(20, "exit", reason="shutdown")])
        self.assertIsNone(f["per_exit"][0]["park_errors"])
        self.assertEqual(f["per_exit"][0]["log_write_failures"], 0)
        self.assertIsNone(f["at_end"])
        self.assertEqual(f["parked"], [])
        f = m.lifecycle_figures([])
        self.assertEqual(f["per_exit"], [])
        self.assertIsNone(f["parked_max_bytes"])

    def test_read_logs_splits_requests_from_lifecycle_lines(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "artemis-2026-11-03.jsonl").write_text(
                "\n".join(json.dumps(x) for x in (life(0, "start"), req(1), life(2, "exit", reason="stall")))
                + "\n{not json\n\n",
                encoding="utf-8",
            )
            (d / "artemis-2026-11-02.jsonl").write_text(json.dumps(req(-90000)) + "\n", encoding="utf-8")
            (d / "notes.txt").write_text("skipped", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()) as err:
                requests, lifecycle = m.read_logs(d)
            self.assertEqual([r["id"] for r in requests], ["req_-90000", "req_1"])
            self.assertEqual([x["event"] for x in lifecycle], ["start", "exit"])
            self.assertIn("1", err.getvalue())


class Guard(unittest.TestCase):
    def test_the_guard_exit_line_is_spec_5_5s(self):
        g = guard(
            (0, "artemis exited (code 4)"),
            (1, "engine exited (code -1)"),
            (2, "tunnel up some-name"),
            (3, "something else entirely"),
        )
        exits = [(e.kind, e.engine, e.code) for e in g if e.kind != "other"]
        self.assertEqual(exits, [("engine_exit", "artemis", 4), ("engine_exit", "engine", -1)])
        # Lines this script reads nothing from are kept only as signs of life.
        self.assertEqual([e.kind for e in g if e.kind == "other"], ["other", "other"])

    def test_availability_sums_the_available_intervals(self):
        g = guard(
            (0, "guard started (pid 10)"),
            (60, "state available"),
            (3660, "state paused: game"),
            (7260, "state available"),
            (10860, "guard stopped"),
            (14400, "guard started (pid 11)"),
            (14460, "state available"),
        )
        end = at(14460 + 1800)
        # The engine's heartbeat is a sign of life within the last 660 s, so the open interval runs to `end`.
        intervals, hours = m.availability(g, at(0), end, [stats(14460 + 1500, stats_body())])
        self.assertEqual(intervals, [(at(60), at(3660)), (at(7260), at(10860)), (at(14460), end)])
        self.assertAlmostEqual(hours, 1.0 + 1.0 + 0.5)
        # A window that starts mid-interval counts from its start.
        intervals, hours = m.availability(g, at(3000), at(7260))
        self.assertEqual(intervals, [(at(3000), at(3660))])
        # An interval cut by `state` of another name, or by the end, never runs backwards.
        self.assertEqual(m.availability([], at(0), at(100)), ([], 0.0))

    def test_the_guard_summary_lists_starts_restarts_pauses_and_exit_codes(self):
        g = guard(
            (0, "guard started (pid 10)"),
            (5, "started artemis (pid 100)"),
            (60, "state available"),
            (1000, "state paused: game"),
            (1000, "stopped artemis (game pause)"),
            (2800, "state starting"),
            (2810, "started artemis (pid 101)"),
            (2900, "state available"),
            (5000, "artemis exited (code 4)"),
            (5010, "started artemis (pid 102)"),
            (5100, "state available"),
            (6000, "artemis exited (code 2)"),
            (6005, "state error: 3 starts in 15 min"),
            (9000, "guard stopped"),
            (9500, "guard started (pid 12)"),
        )
        s = m.guard_summary(g, end=at(20000))
        self.assertEqual([x["time"] for x in s["guard_starts"]], [at(0), at(9500)])
        self.assertEqual(
            [(r["time"], r["pid"], r["after"]) for r in s["restarts"]],
            [
                (at(2810), 101, "stopped artemis (game pause)"),
                (at(5010), 102, "artemis exited (code 4)"),
            ],
        )
        self.assertEqual(
            [(p["state"], p["reason"], p["start"], p["end"]) for p in s["pauses"]],
            [
                ("paused", "game", at(1000), at(2800)),
                ("error", "3 starts in 15 min", at(6005), at(9000)),
            ],
        )
        self.assertEqual([(c["code"], c["time"]) for c in s["exit_codes"]], [(4, at(5000)), (2, at(6000))])

    def test_read_guard_reads_files_and_skips_tunnel_lines(self):
        text = "\n".join(
            glines(
                (0, "guard started (pid 1)"),
                (1, "tunnel up some-name"),
                (2, "state available"),
                (3, "started artemis (pid 5) with the production profile"),
            )
        )
        with tempfile.TemporaryDirectory() as d:
            older = Path(d) / "guard.log.1"
            newer = Path(d) / "guard.log"
            older.write_text(text, encoding="utf-8")
            newer.write_text("\n".join(glines((100, "state paused: game"), (200, "state available"))) + "\n", encoding="utf-8")
            g = m.read_guard([newer, older], to_utc=as_utc)
        self.assertEqual([e.kind for e in g], ["guard_started", "other", "state", "engine_started", "state", "state"])
        self.assertEqual(g[3].pid, 5)
        self.assertEqual(g[4].reason, "game")

    def test_read_guard_skips_a_byte_order_mark_and_counts_unreadable_lines(self):
        body = "\n".join(glines((0, "guard started (pid 1)"), (5, "state available")))
        tail = "\n2026-13-45 99:99:99 a date that is not one\n2026-11-03 broken\nnot a log line\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "guard.log"
            path.write_bytes(("\ufeff" + body + tail).encode("utf-8"))
            with contextlib.redirect_stderr(io.StringIO()) as err:
                g = m.read_guard([path], to_utc=as_utc)
        # The first line is read (the BOM did not hide it); two digit-led lines did not parse; the last is not counted.
        self.assertEqual([e.kind for e in g], ["guard_started", "state"])
        self.assertIn("2 unreadable guard line(s)", err.getvalue())

    def test_a_resume_countdown_is_one_pause(self):
        rows = [(0, "state available"), (100, "state paused: game running")]
        rows += [(100 + 3 * i, f"state paused: no game; resuming in {180 - 3 * i} s") for i in range(1, 60)]
        rows += [(400, "state available"), (500, "state error: 3 starts in 15 min"), (503, "state error: again")]
        rows += [(900, "state available")]
        pauses = m.guard_summary(guard(*rows), end=at(2000))["pauses"]
        self.assertEqual(
            [(p["state"], p["reason"], p["start"], p["end"]) for p in pauses],
            [("paused", "game running", at(100), at(400)), ("error", "3 starts in 15 min", at(500), at(900))],
        )

    def test_an_interval_closed_only_by_guard_started_ends_at_the_last_sign_of_life(self):
        hour = 3600
        g = guard(
            (0, "guard started (pid 1)"),
            (60, "state available"),
            (10 * hour, "guard started (pid 2)"),  # the guard was killed: it never logged `guard stopped`
            (10 * hour + 60, "state available"),
        )
        end = at(10 * hour + 600)  # the second interval is open at the end, with a sign of life 480 s before it
        # With the engine's heartbeat the interval runs to the last stats line; the guard's own last line is the floor.
        lifecycle = [stats(2 * hour, stats_body()), stats(2 * hour + 600, stats_body()), stats(10 * hour + 120, stats_body())]
        intervals, hours = m.availability(g, at(0), end, lifecycle)
        self.assertEqual(intervals, [(at(60), at(2 * hour + 600)), (at(10 * hour + 60), end)])
        self.assertAlmostEqual(hours, (2 * hour + 540 + 540) / 3600.0)
        # With no lifecycle line the guard's own last line is all the evidence there is (the line that opened it).
        intervals, hours = m.availability(g, at(0), end)
        self.assertEqual(intervals, [(at(10 * hour + 60), end)])
        # A `tunnel` line is a guard line too.
        g = guard((0, "state available"), (3 * hour, "tunnel up some-name"), (10 * hour, "guard started (pid 2)"))
        self.assertEqual(m.availability(g, at(0), at(11 * hour))[0], [(at(0), at(3 * hour))])
        # `guard stopped` still closes at its own time.
        g = guard((0, "state available"), (4 * hour, "guard stopped"), (10 * hour, "guard started (pid 2)"))
        self.assertEqual(m.availability(g, at(0), at(11 * hour))[0], [(at(0), at(4 * hour))])

    def test_an_interval_still_open_at_the_end_needs_a_sign_of_life_in_the_last_660_s(self):
        hour = 3600
        end = at(10 * hour)
        g = guard((0, "state available"), (3 * hour, "tunnel up some-name"))
        # No guard or Artemis line in the last 660 s: it ends at the last evidence of life, as when `guard started` closes it.
        self.assertEqual(m.availability(g, at(0), end)[0], [(at(0), at(3 * hour))])
        lifecycle = [stats(5 * hour, stats_body()), stats(5 * hour + 600, stats_body())]
        intervals, hours = m.availability(g, at(0), end, lifecycle)
        self.assertEqual(intervals, [(at(0), at(5 * hour + 600))])
        self.assertAlmostEqual(hours, 5 + 600 / 3600.0)
        # A lifecycle line before the interval opened is no evidence for it.
        self.assertEqual(m.availability(g, at(0), end, [stats(-hour, stats_body())])[0], [(at(0), at(3 * hour))])
        # A sign of life inside the last 660 s, from the engine or the guard, keeps it open to the end.
        self.assertEqual(m.availability(g, at(0), end, [stats(10 * hour - 600, stats_body())])[0], [(at(0), end)])
        g = guard((0, "state available"), (10 * hour - 600, "tunnel up some-name"))
        self.assertEqual(m.availability(g, at(0), end)[0], [(at(0), end)])
        # One 700 s old is too old.
        g = guard((0, "state available"), (10 * hour - 700, "tunnel up some-name"))
        self.assertEqual(m.availability(g, at(0), end)[0], [(at(0), at(10 * hour - 700))])
        # The end clips the evidence: a window that ends before the last line closes at its end.
        g = guard((0, "state available"), (10 * hour, "tunnel up some-name"))
        self.assertEqual(m.availability(g, at(0), at(2 * hour))[0], [(at(0), at(2 * hour))])

    def test_a_utc_request_inside_a_pause_written_in_local_time_is_explained(self):
        if not hasattr(time, "tzset"):
            self.skipTest("time.tzset is missing (Windows); CI runs this on Linux")
        with time_zone(CHICAGO):
            g = m.parse_guard_lines(["2026-07-04 12:00:00 state paused: game", "2026-07-04 13:00:00 state available"])

        def failure(when):
            return {**req(0), "time": when, "status": 500, "error_kind": "server_error", "error_message": "the engine stopped"}

        # 12:00-13:00 CDT is 17:00-18:00 UTC. Read as UTC, 12:30 would have fallen inside.
        self.assertIn("paused", m.explain(failure("2026-07-04T17:30:00Z"), [], g))
        self.assertIsNone(m.explain(failure("2026-07-04T12:30:00Z"), [], g))
        self.assertIsNone(m.explain(failure("2026-07-04T18:30:00Z"), [], g))

    def test_guard_times_are_read_in_local_time(self):
        if not hasattr(time, "tzset"):
            self.skipTest("time.tzset is missing (Windows); CI runs this on Linux")
        with time_zone(CHICAGO):
            summer, amb = m.local_to_utc(datetime(2026, 7, 4, 12, 0, 0))
            self.assertEqual(summer, datetime(2026, 7, 4, 17, 0, 0, tzinfo=UTC))
            self.assertFalse(amb)
            winter, amb = m.local_to_utc(datetime(2026, 1, 15, 12, 0, 0))
            self.assertEqual(winter, datetime(2026, 1, 15, 18, 0, 0, tzinfo=UTC))
            self.assertFalse(amb)
            g = m.parse_guard_lines(["2026-07-04 12:00:00 state available"])
            self.assertEqual(g[0].time, datetime(2026, 7, 4, 17, 0, 0, tzinfo=UTC))
            self.assertFalse(g[0].ambiguous)

    def test_the_dst_change_is_crossed(self):
        if not hasattr(time, "tzset"):
            self.skipTest("time.tzset is missing (Windows); CI runs this on Linux")
        with time_zone(CHICAGO):
            g = m.parse_guard_lines(
                [
                    "2026-11-01 00:30:00 state available",
                    "2026-11-01 01:30:00 state paused: game",
                    "2026-11-01 03:00:00 state available",
                ]
            )
            self.assertEqual(g[0].time, datetime(2026, 11, 1, 5, 30, tzinfo=UTC))  # CDT
            self.assertEqual(g[1].time, datetime(2026, 11, 1, 6, 30, tzinfo=UTC))  # the earlier 01:30, in CDT
            self.assertEqual(g[2].time, datetime(2026, 11, 1, 9, 0, tzinfo=UTC))  # CST
            self.assertEqual([e.ambiguous for e in g], [False, True, False])
            # The hours across the change are the real ones: 00:30 CDT to 03:00 CST is 3.5 h.
            intervals, hours = m.availability(g[:1] + g[2:], g[0].time, g[2].time)
            self.assertAlmostEqual(hours, 3.5)
            # A wall time the change skips (spring forward) still converts, unflagged.
            gap, amb = m.local_to_utc(datetime(2026, 3, 8, 2, 30, 0))
            self.assertFalse(amb)
            self.assertEqual(gap.tzinfo, UTC)


class Reports(unittest.TestCase):
    def test_the_err_log_lists_panics_since_the_start(self):
        text = "\n".join(
            [
                "2026-11-03T11:00:00.000000Z  WARN server::logs: panic in artemis-http at old.rs:1:1: before the start",
                "2026-11-03T12:30:00.123456Z  WARN server::logs: panic in artemis-http at crates/x.rs:10:5: (message withheld)",
                "   0: a backtrace line",
                "2026-11-03T12:40:00.000000Z  INFO server::logs: model loaded in 9.0 s",
                "thread 'artemis-engine' panicked at crates/y.rs:3:4:",
            ]
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "artemis.err.log"
            p.write_text(text + "\n", encoding="utf-8")
            r = m.err_digest(p, at(0))
            self.assertEqual(r["lines"], 5)
            self.assertEqual([x["line"] for x in r["panics"]], [2, 5])
            self.assertIn("crates/x.rs:10:5", r["panics"][0]["text"])
            self.assertEqual(m.err_digest(Path(d) / "missing.log", at(0)), {"lines": 0, "panics": [], "present": False})

    def test_the_digest_marks_unexplained_and_ambiguous_times(self):
        requests = [
            req(100, status=500, error_kind="server_error", error_message="the engine stopped"),
            req(200, status=503, error_kind="overloaded_error", error_message="the request queue is full"),
        ]
        lifecycle = [life(0, "start"), stats(50, stats_body())]
        flagged = m.parse_guard_lines(
            ["2026-11-01 01:30:00 guard started (pid 1)", "2026-11-01 01:31:00 state available"],
            to_utc=lambda naive: (naive.replace(tzinfo=UTC), naive.minute == 30),
        )
        text = m.digest(
            requests,
            lifecycle,
            flagged,
            {"lines": 0, "panics": [], "present": False},
            start=datetime(2026, 10, 31, tzinfo=UTC),
            end=datetime(2026, 11, 4, tzinfo=UTC),
            qwen_runs=5,
        )
        self.assertIn("UNEXPLAINED", text)
        self.assertIn("the request queue is full", text)
        self.assertIn("ambiguous", text)
        self.assertIn("2026-11-01 01:30:00", text)
        # With every failure explained, the word is absent.
        clean = m.digest(
            requests[1:], lifecycle, [], {"lines": 0, "panics": [], "present": False},
            start=at(-3600), end=at(3600), qwen_runs=5,
        )
        self.assertNotIn("UNEXPLAINED", clean)

    def test_guard_events_outside_the_window_stay_out_of_the_summary_and_the_verdict(self):
        day = 86400
        requests = [req(100 + i) for i in range(200)]
        # A pre-window error state, then available from before the window to past its end.
        g = guard(
            (-9 * day, "guard started (pid 1)"),
            (-9 * day + 5, "state error: 3 starts in 15 min"),
            (-9 * day + 9, "artemis exited (code 4)"),
            (-8 * day, "state available"),
            (4 * day, "guard started (pid 2)"),
            (4 * day + 5, "state available"),
            (30 * day, "state paused: game"),  # after the window
        )
        end = at(5 * day)
        none = {"lines": 0, "panics": [], "present": False}
        beats = [stats(100, stats_body()), stats(4 * day - 60, stats_body())]
        rep = m.report(requests, beats, g, none, at(0), end, 5)
        self.assertEqual(rep["guard"]["pauses"], [])
        self.assertEqual(rep["guard"]["exit_codes"], [])
        self.assertEqual([s["time"] for s in rep["guard"]["guard_starts"]], [at(4 * day)])
        self.assertTrue(rep["pass"], rep["traffic"])
        record = m.results_record(rep, at(5 * day))
        self.assertEqual((record["status"], record["error_states"], record["exit_codes"]), ("pass", 0, []))
        self.assertIn("owner", record["owner_rules"])
        # Availability still counts the state open at the start.
        self.assertGreater(rep["available_hours"], 72)
        # The same error state inside the window does fail it.
        g = guard((100, "state error: 3 starts in 15 min"), (200, "state available"))
        rep = m.report(requests, [], g, none, at(0), at(300), 5)
        self.assertFalse(rep["pass"])

    def test_final_writes_the_summary_and_appends_one_record(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            logs = d / "logs"
            logs.mkdir()
            rows = [life(0, "start"), life(5, "load", seconds=9.0), stats(600, stats_body())]
            rows += [req(10 + i) for i in range(3)]
            rows.append(req(900, status=500, error_kind="server_error", error_message="the engine stopped"))
            (logs / "artemis-2026-11-03.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
            err = d / "artemis.err.log"
            err.write_text("2026-11-03T12:00:01.000000Z  INFO server::logs: artemis listening\n", encoding="utf-8")
            guard_log = d / "guard.log"
            guard_log.write_text(
                "\n".join(
                    glines(
                        (0, "guard started (pid 1)"),
                        (1, "state available"),
                        (1 + 72 * 3600, "state paused: game"),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            evidence = d / "evidence" / "m4"
            results = evidence / "results.jsonl"
            evidence.mkdir(parents=True)
            results.write_text('{"test":"earlier","status":"pass"}\n', encoding="utf-8")
            out = d / "digest.md"
            argv = [
                "--logs", str(logs), "--err", str(err), "--guard", str(guard_log),
                "--start", "2026-11-02T00:00:00Z", "--end", "2026-11-09T00:00:00Z",
                "--qwen-runs", "5", "--out", str(out), "--final", "--evidence", str(evidence),
            ]  # fmt: skip
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(m.main(argv), 0)
            summary = (evidence / "soak-summary.md").read_text(encoding="utf-8")
            self.assertIn("UNEXPLAINED", summary)
            self.assertTrue(out.read_text(encoding="utf-8").strip())
            records = [json.loads(x) for x in results.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["test"], "earlier")
            r = records[1]
            self.assertEqual(r["test"], "m4-soak")
            self.assertEqual(r["failures"], 1)
            self.assertEqual(r["unexplained"], 1)
            self.assertEqual(r["generation_requests"], 3)
            self.assertEqual(r["qwen_runs"], 5)
            self.assertEqual(r["status"], "fail")
            self.assertAlmostEqual(r["available_hours"], 72.0)
            # A second run appends a second record.
            with contextlib.redirect_stdout(io.StringIO()):
                m.main(argv)
            self.assertEqual(len(results.read_text(encoding="utf-8").splitlines()), 3)

    def test_final_needs_an_evidence_directory(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            m.main(["--logs", ".", "--err", "x", "--guard", "y", "--start", "2026-11-02T00:00:00Z", "--qwen-runs", "1", "--final"])


if __name__ == "__main__":
    unittest.main()
