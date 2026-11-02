"""The soak report: reads the production request/lifecycle logs, artemis.err.log and guard.log, and prints
a Markdown digest. Read-only; sends nothing anywhere. Standard library only (Python 3.11 or later).

    python -I tools/py/m4_soak.py --logs DIR --err FILE --guard FILE [--guard FILE] --start ISO [--end ISO]
        --qwen-runs N [--out FILE] [--final --evidence DIR]

`--start` and `--end` take an ISO time. One with an offset or `Z` is read as given; one without is local time, like
guard.log's. `--end` defaults to now. `--guard` may repeat (guard.log and guard.log.1). `--out` also writes the digest to
a file. `--final` also writes `<evidence>/soak-summary.md` and appends one record to `<evidence>/results.jsonl`.

Log times are UTC with whole-second resolution; guard.log times are local with no offset, converted by the OS's own
rules (`time.mktime`). The repeated hour of a DST change is ambiguous: it takes the earlier instant and the digest
flags it.
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc
FAR_FUTURE = datetime.max.replace(tzinfo=UTC)

# The traffic floors.
FLOOR_REQUESTS = 200
FLOOR_RUNS = 5
FLOOR_HOURS = 72.0

EXPLAIN_WINDOW_S = 60
# A request line's `time` is its arrival and `total_ms` how long it ran; a lifecycle line stamped to the whole second
# up to this long after its end still explains it.
EXPLAIN_AFTER_END_S = 1
STATS_GAP_S = 660
GENERATION_ENDPOINTS = ("/v1/chat/completions", "/v1/messages")
# Messages that point at another event instead of naming a cause: the failure needs a line or a window.
POINTS_ELSEWHERE = ("the engine stopped", "the prompt could not be built")
# Lifecycle lines that explain a failure within EXPLAIN_WINDOW_S after them (including reload_failed).
CAUSES = ("device_lost", "unload", "load_failed", "reload_failed", "stall", "panic", "exit")
TIMELINE = ("start", "load", "unload", "load_failed", "reload_failed", "device_lost", "stall", "panic", "exit")


def parse_time(s):
    """A log time (`YYYY-MM-DDTHH:MM:SSZ`, or with an offset) as an aware UTC datetime."""
    t = datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", s))
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def stamp(t):
    return t.strftime("%Y-%m-%d %H:%M:%SZ")


def read_logs(directory):
    """The request lines and the lifecycle lines (those with an `event`) of every `*.jsonl` in `directory`, in time
    order. A line that is not JSON is skipped and counted on stderr."""
    requests, lifecycle, bad = [], [], 0
    for path in sorted(Path(directory).glob("*.jsonl")):
        with open(path, encoding="utf-8", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    line = json.loads(raw)
                except ValueError:
                    bad += 1
                    continue
                if not isinstance(line, dict) or "time" not in line:
                    bad += 1
                    continue
                (lifecycle if "event" in line else requests).append(line)
    if bad:
        print(f"m4_soak: {bad} unreadable log line(s) skipped", file=sys.stderr)
    # Stable: lines of one second keep the order they were written in.
    requests.sort(key=lambda line: line["time"])
    lifecycle.sort(key=lambda line: line["time"])
    return requests, lifecycle


# ---- the guard ----------------------------------------------------------------------------------------------------


def local_to_utc(naive):
    """A guard.log wall time as (the UTC instant, whether the wall time is ambiguous), by the OS's own rules.

    Ambiguous is the repeated hour of a DST change: `tm_isdst` 0 and 1 give two instants that both read back as the
    wall time. It takes the earlier. A wall time the change skips has no instant; the OS's normalised one is used."""
    wall = tuple(naive.timetuple()[:6])
    found = []
    for dst in (0, 1):
        try:
            t = time.mktime(wall + (0, 0, dst))
            if tuple(time.localtime(t)[:6]) == wall and t not in found:
                found.append(t)
        except (OverflowError, ValueError, OSError):
            pass
    if not found:
        found.append(time.mktime(wall + (0, 0, -1)))
    return datetime.fromtimestamp(min(found), UTC), len(found) > 1


@dataclass
class GuardEvent:
    time: datetime
    kind: str  # guard_started, guard_stopped, state, engine_started, engine_stopped, engine_exit or other
    text: str  # the line's message, as the guard wrote it
    local: str  # the line's own wall time
    ambiguous: bool = False
    state: str | None = None
    reason: str | None = None
    engine: str | None = None
    pid: int | None = None
    code: int | None = None


GUARD_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*?)\s*$")
GUARD_STATE = re.compile(r"^state ([^\s:]+)(?::?\s*(.*))?$")
GUARD_PID = re.compile(r"\(pid (\d+)\)")
ENGINE_STARTED = re.compile(r"^started (\S+) \(pid (\d+)\)")
ENGINE_STOPPED = re.compile(r"^stopped (\S+) \(([^)]*)\)")
ENGINE_EXIT = re.compile(r"^(\S+) exited \(code (-?\d+)\)")  # a named engine matches too
STATE_KINDS = ("state", "guard_stopped", "guard_started")


def _parse_guard(lines, to_utc):
    """(the events, the count of lines that start with a digit but did not parse)."""
    events, bad = [], 0
    for raw in lines:
        match = GUARD_LINE.match(raw)
        if not match:
            if raw[:1].isdigit():
                bad += 1
            continue
        local, text = match.groups()
        try:
            utc, ambiguous = to_utc(datetime.strptime(local, "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            bad += 1
            continue
        base = dict(time=utc, text=text, local=local, ambiguous=ambiguous)
        if text.startswith("guard started"):
            pid = GUARD_PID.search(text)
            events.append(GuardEvent(kind="guard_started", pid=int(pid.group(1)) if pid else None, **base))
        elif text.startswith("guard stopped"):
            events.append(GuardEvent(kind="guard_stopped", **base))
        elif m := GUARD_STATE.match(text):
            events.append(GuardEvent(kind="state", state=m.group(1), reason=m.group(2) or None, **base))
        elif m := ENGINE_STARTED.match(text):
            events.append(GuardEvent(kind="engine_started", engine=m.group(1), pid=int(m.group(2)), **base))
        elif m := ENGINE_STOPPED.match(text):
            events.append(GuardEvent(kind="engine_stopped", engine=m.group(1), reason=m.group(2), **base))
        elif m := ENGINE_EXIT.match(text):
            events.append(GuardEvent(kind="engine_exit", engine=m.group(1), code=int(m.group(2)), **base))
        else:
            # `tunnel ...` and the rest: nothing is read from them, but they show the guard was alive.
            events.append(GuardEvent(kind="other", **base))
    return events, bad


def parse_guard_lines(lines, to_utc=local_to_utc):
    """GuardEvents from `yyyy-MM-dd HH:mm:ss <message>` lines. A line this script reads nothing from (`tunnel ...`
    and the rest) becomes an `other` event: it only counts as a sign of life for `availability`."""
    return _parse_guard(lines, to_utc)[0]


def read_guard(paths, to_utc=local_to_utc):
    """The events of guard.log and its rotations, oldest file first (by its first event), each in file order. The
    real file starts with a UTF-8 BOM. Lines that start with a digit and do not parse are counted on stderr."""
    files, bad = [], 0
    for path in paths:
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            events, n = _parse_guard(f.read().splitlines(), to_utc)
        bad += n
        if events:
            files.append(events)
    if bad:
        print(f"m4_soak: {bad} unreadable guard line(s) skipped", file=sys.stderr)
    files.sort(key=lambda events: events[0].time)
    return [e for events in files for e in events]


def availability(guard, start, end, lifecycle=()):
    """(the intervals, their hours): each `state available` up to the next `state` line, `guard stopped`,
    `guard started` or `end`, clipped to [start, end].

    The real guard never logs `guard stopped` (it is killed at logoff or shutdown), so an interval closed by a
    `guard started` alone ends at the last sign of life before it: the later of the last guard line and the last
    lifecycle line (the engine's `stats` heartbeat among them) inside the interval. One still open at `end` ends the
    same way unless a guard or lifecycle line falls in its last `STATS_GAP_S` (a heartbeat is 600 s apart)."""
    intervals, opened, last_guard = [], None, None
    times = [parse_time(line["time"]) for line in lifecycle]

    def close(at):
        nonlocal opened
        if opened is not None:
            a, b = max(opened, start), min(at, end)
            if b > a:
                intervals.append((a, b))
            opened = None

    for e in guard:
        if e.kind == "guard_started" and opened is not None:
            close(max([last_guard] + [t for t in times if opened <= t <= e.time]))
        elif e.kind == "state":
            if e.state == "available":
                if opened is None:
                    opened = e.time
            else:
                close(e.time)
        elif e.kind in STATE_KINDS:
            close(e.time)
        last_guard = e.time
    if opened is not None:
        # Still open at `end`: with no guard or Artemis line in its last STATS_GAP_S the guard was likely killed
        # without a word (as for `guard started` above), so it ends at the last evidence of life, clipped to `end`.
        alive = min(max([last_guard] + [t for t in times if opened <= t <= end]), end)
        if (end - alive).total_seconds() > STATS_GAP_S:
            close(alive)
    close(end)
    return intervals, sum((b - a).total_seconds() for a, b in intervals) / 3600.0


def _is_pause(e):
    return e.kind == "state" and (e.state.startswith("pause") or e.state == "error")


def _pauses(guard, end):
    """Each pause and `error` state as one interval: consecutive `state` lines of the same name (the guard writes
    `state paused: no game; resuming in N s` every 3 s while it counts down) merge into the first, which keeps its
    reason. It ends at the next `state` line of another name, `guard stopped`, `guard started` or `end`."""
    out, current = [], None
    for e in guard:
        if e.kind == "state":
            if current is not None and e.state == current["state"]:
                continue
            if current is not None:
                current["end"] = e.time
                current = None
            if _is_pause(e):
                current = {"state": e.state, "reason": e.reason, "start": e.time, "end": end, "ambiguous": e.ambiguous}
                out.append(current)
        elif e.kind in STATE_KINDS and current is not None:
            current["end"] = e.time
            current = None
    return out


def _restart_windows(guard, end):
    """From each engine exit or stop to the next engine start (or `end`): (start, stop, the line's text)."""
    out, pending = [], []
    for e in guard:
        if e.kind in ("engine_exit", "engine_stopped"):
            pending.append((e.time, e.text))
        elif e.kind == "engine_started":
            out += [(t, e.time, text) for t, text in pending]
            pending = []
    out += [(t, end, text) for t, text in pending]
    return sorted(out)


def guard_summary(guard, end=None):
    """Every guard start, the restarts (an engine start after an exit or a stop), each pause and `error` state with
    its reason and interval, and each exit code."""
    end = end or FAR_FUTURE
    starts = [
        {"time": e.time, "local": e.local, "ambiguous": e.ambiguous, "pid": e.pid}
        for e in guard
        if e.kind == "guard_started"
    ]
    restarts, previous = [], None
    for e in guard:
        if e.kind == "engine_started":
            if previous is not None:
                restarts.append(
                    {"time": e.time, "engine": e.engine, "pid": e.pid, "after": previous.text, "ambiguous": e.ambiguous}
                )
            previous = None
        elif e.kind in ("engine_exit", "engine_stopped"):
            previous = e
    codes = [
        {"time": e.time, "engine": e.engine, "code": e.code, "ambiguous": e.ambiguous}
        for e in guard
        if e.kind == "engine_exit"
    ]
    return {"guard_starts": starts, "restarts": restarts, "pauses": _pauses(guard, end), "exit_codes": codes}


# ---- requests ------------------------------------------------------------------------------------------------------


def failures(requests):
    """Status 500 or more, status 503, or `error_kind` set on a 200 (a broken stream)."""
    return [
        r
        for r in requests
        if r.get("status", 0) >= 500 or (r.get("status") == 200 and r.get("error_kind"))
    ]


def client_errors(requests):
    return [r for r in requests if 400 <= r.get("status", 0) <= 498]


def cancellations(requests):
    return [r for r in requests if r.get("status") == 499 or r.get("finish") == "cancelled"]


def status_counts(requests):
    by_status = Counter(str(r.get("status")) for r in requests)
    by_kind = Counter(r["error_kind"] for r in requests if r.get("error_kind"))
    return {
        "by_status": dict(sorted(by_status.items())),
        "by_error_kind": dict(sorted(by_kind.items())),
        "broken_streams": sum(1 for r in requests if r.get("status") == 200 and r.get("error_kind")),
        "cancellations": len(cancellations(requests)),
    }


def guard_windows(guard, end=None):
    """(the pause and `error` intervals, the restart windows): what `explain` checks a failure's time against.
    Computed once per report and passed in."""
    end = end or FAR_FUTURE
    return _pauses(guard, end), _restart_windows(guard, end)


def explain(failure, lifecycle, guard, end=None, windows=None):
    """Why `failure` happened, or None. Its own `error_message`, unless that points elsewhere; else the nearest
    lifecycle line in CAUSES from 60 s before its arrival (`time`) to the end of the request (`time` plus `total_ms`,
    and a second); else a guard pause or restart window that holds the arrival. A line with no `total_ms` (an older
    log) ends at its arrival. `windows` is `guard_windows(guard, end)`, to avoid rebuilding it for each failure."""
    t = parse_time(failure["time"])
    ran = failure.get("total_ms")
    ended = t
    if isinstance(ran, (int, float)) and not isinstance(ran, bool) and ran >= 0:
        ended = t + timedelta(seconds=ran / 1000.0 + EXPLAIN_AFTER_END_S)
    message = failure.get("error_message")
    if message and not any(p in message.lower() for p in POINTS_ELSEWHERE):
        return f"its own message: {message}"
    nearest = None
    for line in lifecycle:
        if line.get("event") in CAUSES:
            lt = parse_time(line["time"])
            if t - timedelta(seconds=EXPLAIN_WINDOW_S) <= lt <= ended and (nearest is None or lt >= nearest[0]):
                nearest = (lt, line)
    if nearest:
        lt, line = nearest
        return f"{line['event']} at {stamp(lt)}"
    pauses, restarts = windows or guard_windows(guard, end)
    for p in pauses:
        if p["start"] <= t <= p["end"]:
            return f"guard {p['state']} since {stamp(p['start'])}"
    for start, stop, text in restarts:
        if start <= t <= stop:
            return f"guard restart window after '{text}' at {stamp(start)}"
    return None


def reuse_by_cause(requests):
    """Per `cause`: the requests, and the reused tokens over the prompt tokens."""
    out = {}
    for r in requests:
        cause, prompt, reused = r.get("cause"), r.get("prompt_tokens"), r.get("reused_tokens")
        if cause is None or prompt is None or reused is None:
            continue
        c = out.setdefault(cause, {"requests": 0, "reused_tokens": 0, "prompt_tokens": 0})
        c["requests"] += 1
        c["reused_tokens"] += reused
        c["prompt_tokens"] += prompt
    for c in out.values():
        c["rate"] = c["reused_tokens"] / c["prompt_tokens"] if c["prompt_tokens"] else None
    return out


def _figure(values):
    """Median and p10, or None below 2 points (`statistics.quantiles` raises there)."""
    if len(values) < 2:
        return None
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p10": statistics.quantiles(values, n=10, method="inclusive")[0],
    }


def decode_figures(requests, min_completion=256):
    """The figures over the answered requests of `min_completion` tokens or more: `decode_tok_s` (completion
    over decode time, as at milestone 3) and `decode_tokens` over `decode_ms` where present. Broken streams are left
    out; a logged 0.0 is a figure like any other."""
    rows = [
        r
        for r in requests
        if r.get("status") == 200 and not r.get("error_kind") and (r.get("completion_tokens") or 0) >= min_completion
    ]
    logged = [r["decode_tok_s"] for r in rows if r.get("decode_tok_s") is not None]
    measured = [
        r["decode_tokens"] / (r["decode_ms"] / 1000.0)
        for r in rows
        if r.get("decode_tokens") is not None and (r.get("decode_ms") or 0) > 0
    ]
    return {
        "min_completion": min_completion,
        "requests": len(rows),
        "tok_s": _figure(logged),
        "tokens_over_ms": _figure(measured),
    }


def spec_figures(requests):
    """Speculation's figures over the answered requests (status 200, no `error_kind`: broken streams are left out of all
    of them, as `decode_figures` does) that speculated, or None when none did: an older log has no
    `spec_*` keys, and a request with speculation off has `spec_steps` null, so everything is read with `.get`. A
    speculating request has a `spec_steps` count, which is 0 when every step of it ran plain.
    `plain_share` is the plain steps over all steps, summed over every speculating request (those that never verified
    count, or the share would read low); `requests` counts them. `accept_rate` is the median and p10 of the requests' own
    `spec_accept_rate` (null where nothing was drafted, so left out) and `tok_s` the logged `decode_tok_s`, both over the
    `verified_requests` among them (`spec_steps > 0`). `sources` pools each source's drafted and accepted
    counts (`spec_drafted_lookup`, ... read with `.get`: a 5a line has none and adds 0) over the same requests, with the
    pooled `accept_rate` per source (null where it drafted nothing). Report only. No minimum completion length applies
    (unlike `decode_figures`' 256): these are report-only."""
    rows = [
        r
        for r in requests
        if r.get("spec_steps") is not None and r.get("status") == 200 and not r.get("error_kind")
    ]
    steps = sum(r["spec_steps"] for r in rows)
    plain = sum(r.get("spec_plain_steps") or 0 for r in rows)
    if steps + plain <= 0:
        return None
    verified = [r for r in rows if r["spec_steps"] > 0]
    rates = [r["spec_accept_rate"] for r in verified if r.get("spec_accept_rate") is not None]
    tok_s = [r["decode_tok_s"] for r in verified if r.get("decode_tok_s") is not None]
    sources = {}
    for name in ("lookup", "mtp"):
        drafted = sum(r.get(f"spec_drafted_{name}") or 0 for r in rows)
        accepted = sum(r.get(f"spec_accepted_{name}") or 0 for r in rows)
        sources[name] = {
            "drafted": drafted,
            "accepted": accepted,
            "accept_rate": accepted / drafted if drafted > 0 else None,
        }
    return {
        "requests": len(rows),
        "verified_requests": len(verified),
        "accept_rate": _figure(rates),
        "sources": sources,
        "plain_share": plain / (steps + plain),
        "tok_s": _figure(tok_s),
    }


def traffic(requests, qwen_runs, available_hours):
    """The floors: 200 generation requests, 5 qwen-queue runs, 72 available hours. A generation request is an
    answered chat or messages request (status 200, no `error_kind`)."""
    n = sum(
        1
        for r in requests
        if r.get("endpoint") in GENERATION_ENDPOINTS and r.get("status") == 200 and not r.get("error_kind")
    )
    out = {
        "generation_requests": n,
        "requests_ok": n >= FLOOR_REQUESTS,
        "qwen_runs": qwen_runs,
        "runs_ok": qwen_runs >= FLOOR_RUNS,
        "available_hours": available_hours,
        "hours_ok": available_hours >= FLOOR_HOURS,
    }
    out["met"] = out["requests_ok"] and out["runs_ok"] and out["hours_ok"]
    return out


# ---- lifecycle -----------------------------------------------------------------------------------------------------


def unclean_ends(lifecycle, guard):
    """A `start` whose previous lifecycle line is not `exit`, annotated with the guard's `stopped ...` or
    `... exited (code N)` lines between that previous line and the start."""
    ends, previous = [], None
    for line in lifecycle:
        if line["event"] == "start" and previous is not None and previous["event"] != "exit":
            t, pt = parse_time(line["time"]), parse_time(previous["time"])
            notes = [
                e.text for e in guard if e.kind in ("engine_stopped", "engine_exit") and pt <= e.time <= t
            ]
            ends.append(
                {"time": t, "previous": previous["event"], "previous_time": pt, "guard": "; ".join(notes) or None}
            )
        previous = line
    return ends


def stats_gaps(lifecycle, available, limit_s=STATS_GAP_S):
    """Gaps over `limit_s` between `stats` lines (or an available interval's ends) while the guard says available."""
    times = sorted(parse_time(line["time"]) for line in lifecycle if line["event"] == "stats")
    gaps = []
    for a, b in available:
        points = [a] + [t for t in times if a <= t <= b] + [b]
        for x, y in zip(points, points[1:]):
            gap = (y - x).total_seconds()
            if gap > limit_s:
                gaps.append({"from": x, "to": y, "gap_s": gap})
    return gaps


def _detail(line):
    for key in ("cause", "reason", "message", "doing"):
        if line.get(key):
            return f"{key}: {line[key]}"
    if line.get("seconds") is not None:
        return f"{line['seconds']} s"
    return ""


def _counters(line):
    s = line.get("stats") or {}
    return {
        "park_errors": s.get("park_errors"),
        "chunks_refused": s.get("chunks_refused"),
        "log_write_failures": line.get("log_write_failures"),
    }


def lifecycle_figures(lifecycle):
    """The counters of the last `stats` line before each `exit` and at the end, the parked RAM (the `stats` line
    written at each `unload`) and the timeline of the lifecycle lines. The counters are per process, so
    the totals add the processes up."""
    per_exit, parked, timeline, last = [], [], [], None
    for i, line in enumerate(lifecycle):
        event = line["event"]
        if event in TIMELINE:
            timeline.append({"time": parse_time(line["time"]), "event": event, "detail": _detail(line)})
        if event == "stats":
            last = line
        elif event == "exit":
            counters = _counters(last) if last else {"park_errors": None, "chunks_refused": None, "log_write_failures": None}
            per_exit.append({"time": parse_time(line["time"]), "reason": line.get("reason"), **counters})
            last = None
        elif event == "unload":
            after = next((n for n in lifecycle[i + 1 :]), None)
            if after is not None and after["event"] == "stats":
                s = after.get("stats") or {}
                parked.append(
                    {
                        "time": parse_time(line["time"]),
                        "cause": line.get("cause"),
                        "pool_bytes_used": s.get("pool_bytes_used"),
                        "parked_snapshots": s.get("parked_snapshots"),
                    }
                )
    at_end = None
    if last is not None:
        at_end = {"time": parse_time(last["time"]), **_counters(last)}
    totals = {}
    for key in ("park_errors", "chunks_refused", "log_write_failures"):
        totals[key] = sum((r[key] or 0) for r in per_exit + ([at_end] if at_end else []))
    used = [p["pool_bytes_used"] for p in parked if p["pool_bytes_used"] is not None]
    return {
        "per_exit": per_exit,
        "at_end": at_end,
        "totals": totals,
        "parked": parked,
        "parked_max_bytes": max(used) if used else None,
        "timeline": timeline,
    }


ANSI = re.compile(r"\x1b\[[0-9;]*m")
ERR_STAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z)")


def err_digest(path, start):
    """The line count of artemis.err.log and its `panicked at` and `panic in` lines since `start`, listed, not
    judged. A line with no time of its own (a backtrace) takes the time of the line before it."""
    path = Path(path)
    if not path.exists():
        return {"lines": 0, "panics": [], "present": False}
    count, panics, since = 0, [], True
    with open(path, encoding="utf-8", errors="replace") as f:
        for number, raw in enumerate(f, 1):
            count = number
            text = ANSI.sub("", raw).rstrip()
            stamped = ERR_STAMP.match(text)
            if stamped:
                since = parse_time(stamped.group(1)) >= start
            if since and ("panicked at" in text or "panic in" in text):
                panics.append({"line": number, "text": text[:300]})
    return {"lines": count, "panics": panics, "present": True}


# ---- the report ----------------------------------------------------------------------------------------------------


def report(requests, lifecycle, guard, err, start, end, qwen_runs):
    # `availability` and `explain` need the whole guard list (the state open at `start`); the summary, the ambiguous
    # times and the error check see only the window's own events.
    intervals, hours = availability(guard, start, end, lifecycle)
    windows = guard_windows(guard, end)
    explained = [{"request": r, "why": explain(r, lifecycle, guard, end, windows)} for r in failures(requests)]
    inside = [e for e in guard if start <= e.time <= end]
    summary = guard_summary(inside, end)
    t = traffic(requests, qwen_runs, hours)
    unexplained = sum(1 for f in explained if f["why"] is None)
    return {
        "start": start,
        "end": end,
        "available_hours": hours,
        "traffic": t,
        "counts": status_counts(requests),
        "failures": explained,
        "unexplained": unexplained,
        "client_errors": client_errors(requests),
        "cancellations": cancellations(requests),
        "unclean_ends": unclean_ends(lifecycle, guard),
        "stats_gaps": stats_gaps(lifecycle, intervals),
        "figures": lifecycle_figures(lifecycle),
        "guard": summary,
        "decode": decode_figures(requests),
        "spec": spec_figures(requests),
        "reuse": reuse_by_cause(requests),
        "err": err,
        "ambiguous": [e for e in inside if e.ambiguous and e.kind != "other"],
        "pass": t["met"] and unexplained == 0 and not any(p["state"] == "error" for p in summary["pauses"]),
    }


def _num(x, places=1):
    return "n/a" if x is None else f"{x:.{places}f}"


def _fig(f):
    return "n/a (fewer than 2 requests)" if f is None else f"median {f['median']:.2f}, p10 {f['p10']:.2f} ({f['n']} requests)"


def render(rep):
    out = []
    w = out.append
    t = rep["traffic"]
    w("# Artemis soak digest")
    w("")
    w(f"Window: {stamp(rep['start'])} to {stamp(rep['end'])}. Computed rules only: the owner judges manual action and "
      "whether any boot was the GPU's.")
    w("")
    w("## Traffic floors (spec 9.2)")
    w("")
    w("| floor | have | needs | met |")
    w("|---|---|---|---|")
    w(f"| generation requests | {t['generation_requests']} | {FLOOR_REQUESTS} | {t['requests_ok']} |")
    w(f"| qwen-queue runs | {t['qwen_runs']} | {FLOOR_RUNS} | {t['runs_ok']} |")
    w(f"| available hours | {t['available_hours']:.2f} | {FLOOR_HOURS:.0f} | {t['hours_ok']} |")
    w("")
    c = rep["counts"]
    w("## Requests")
    w("")
    w("By status: " + (", ".join(f"{k}: {v}" for k, v in c["by_status"].items()) or "none"))
    w("")
    w("By error_kind: " + (", ".join(f"{k}: {v}" for k, v in c["by_error_kind"].items()) or "none"))
    w("")
    w(f"Broken streams (200 with error_kind): {c['broken_streams']}. Cancellations: {c['cancellations']}. "
      f"Client errors (400-498): {len(rep['client_errors'])}. The last two are listed, not judged.")
    w("")
    w(f"### Failures: {len(rep['failures'])}, unexplained: {rep['unexplained']}")
    w("")
    if rep["failures"]:
        w("| time | id | status | error_kind | explanation |")
        w("|---|---|---|---|---|")
        for f in rep["failures"]:
            r = f["request"]
            w(f"| {r['time']} | {r.get('id')} | {r.get('status')} | {r.get('error_kind')} | {f['why'] or 'UNEXPLAINED'} |")
        w("")
    w("## Lifecycle")
    w("")
    fig = rep["figures"]
    if fig["timeline"]:
        for e in fig["timeline"]:
            w(f"- {stamp(e['time'])} {e['event']} {e['detail']}".rstrip())
    else:
        w("No lifecycle lines.")
    w("")
    w(f"Unclean ends (a start with no exit before it): {len(rep['unclean_ends'])}")
    w("")
    for u in rep["unclean_ends"]:
        w(f"- start at {stamp(u['time'])}, previous line {u['previous']} at {stamp(u['previous_time'])}; "
          f"guard: {u['guard'] or 'no stop or exit line in the window'}")
    w("")
    w(f"Stats gaps over {STATS_GAP_S} s while available: {len(rep['stats_gaps'])}")
    w("")
    for g in rep["stats_gaps"]:
        w(f"- {stamp(g['from'])} to {stamp(g['to'])} ({g['gap_s']:.0f} s)")
    w("")
    w("### Counters (per process, last stats line before each exit and at the end)")
    w("")
    tot = fig["totals"]
    w(f"Totals: park_errors {tot['park_errors']}, chunks_refused {tot['chunks_refused']}, "
      f"log_write_failures {tot['log_write_failures']}.")
    w("")
    for p in fig["per_exit"]:
        w(f"- exit at {stamp(p['time'])} ({p['reason']}): park_errors {p['park_errors']}, "
          f"chunks_refused {p['chunks_refused']}, log_write_failures {p['log_write_failures']}")
    if fig["at_end"]:
        a = fig["at_end"]
        w(f"- at the end ({stamp(a['time'])}): park_errors {a['park_errors']}, chunks_refused {a['chunks_refused']}, "
          f"log_write_failures {a['log_write_failures']}")
    w("")
    mx = fig["parked_max_bytes"]
    w(f"Parked RAM (pool_bytes_used at each unload): {len(fig['parked'])} unloads, "
      f"max {_num(mx / (1 << 30) if mx is not None else None, 2)} GiB.")
    w("")
    g = rep["guard"]
    w("## Guard")
    w("")
    w(f"Guard starts and boots seen: {len(g['guard_starts'])}")
    w("")
    for s in g["guard_starts"]:
        w(f"- guard started {stamp(s['time'])} (local {s['local']}){' AMBIGUOUS' if s['ambiguous'] else ''}")
    w("")
    w(f"Restarts: {len(g['restarts'])}")
    w("")
    for r in g["restarts"]:
        w(f"- {stamp(r['time'])} {r['engine']} pid {r['pid']}, after '{r['after']}'")
    w("")
    w(f"Pauses and error states: {len(g['pauses'])}")
    w("")
    for p in g["pauses"]:
        end = "open" if p["end"] == FAR_FUTURE else stamp(p["end"])
        w(f"- {p['state']} ({p['reason'] or 'no reason'}) {stamp(p['start'])} to {end}")
    w("")
    w("Engine exit codes: " + (", ".join(f"{e['code']} at {stamp(e['time'])}" for e in g["exit_codes"]) or "none"))
    w("")
    if rep["ambiguous"]:
        w("### Ambiguous guard times")
        w("")
        w("These wall times fall in the repeated hour of a DST change; the earlier instant was taken.")
        w("")
        for e in rep["ambiguous"]:
            w(f"- {e.local} {e.text} (ambiguous)")
        w("")
    d = rep["decode"]
    w("## Decode and reuse")
    w("")
    w(f"Requests of {d['min_completion']} tokens or more: {d['requests']}.")
    w("")
    w(f"- decode_tok_s: {_fig(d['tok_s'])}")
    w(f"- decode_tokens / decode_ms: {_fig(d['tokens_over_ms'])}")
    w("")
    sp = rep["spec"]
    if sp is None:
        w("Speculation (report only): no request line has any speculation steps.")
    else:
        w(f"Speculation (report only), over the {sp['requests']} speculating requests:")
        w("")
        w(f"- plain-step share: {_num(sp['plain_share'], 3)}")
        w(f"- accept rate (over the {sp['verified_requests']} that verified): {_fig(sp['accept_rate'])}")
        for name, c in sp["sources"].items():
            w(f"- {name} accept rate ({c['accepted']} of {c['drafted']} drafted): {_num(c['accept_rate'], 3)}")
        w(f"- decode_tok_s (over the {sp['verified_requests']} that verified): {_fig(sp['tok_s'])}")
    w("")
    if rep["reuse"]:
        w("| cause | requests | reused tokens | prompt tokens | rate |")
        w("|---|---|---|---|---|")
        for cause, r in sorted(rep["reuse"].items()):
            w(f"| {cause} | {r['requests']} | {r['reused_tokens']} | {r['prompt_tokens']} | {_num(r['rate'], 3)} |")
        w("")
    e = rep["err"]
    w("## artemis.err.log")
    w("")
    if not e["present"]:
        w("Not found.")
    else:
        w(f"{e['lines']} lines; {len(e['panics'])} panic line(s) since the start, listed, not judged.")
        w("")
        for p in e["panics"]:
            w(f"- line {p['line']}: {p['text']}")
    w("")
    return "\n".join(out)


def digest(requests, lifecycle, guard, err, start, end, qwen_runs):
    return render(report(requests, lifecycle, guard, err, start, end, qwen_runs))


def results_record(rep, now):
    """The `results.jsonl` record for --final. `pass` covers the rules the logs can compute."""
    t, fig, g, d = rep["traffic"], rep["figures"], rep["guard"], rep["decode"]
    sp = rep["spec"]
    rate = sp["accept_rate"] if sp else None
    return {
        "test": "m4-soak",
        "status": "pass" if rep["pass"] else "fail",
        "owner_rules": "status is the computed rules only; manual action and GPU reboots are the owner's call (spec 9.2)",
        "date": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start": rep["start"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": rep["end"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "available_hours": round(rep["available_hours"], 3),
        "generation_requests": t["generation_requests"],
        "qwen_runs": t["qwen_runs"],
        "floors_met": t["met"],
        "failures": len(rep["failures"]),
        "unexplained": rep["unexplained"],
        "client_errors": len(rep["client_errors"]),
        "cancellations": len(rep["cancellations"]),
        "unclean_ends": len(rep["unclean_ends"]),
        "stats_gaps": len(rep["stats_gaps"]),
        "guard_starts": len(g["guard_starts"]),
        "restarts": len(g["restarts"]),
        "pauses_and_errors": len(g["pauses"]),
        "error_states": sum(1 for p in g["pauses"] if p["state"] == "error"),
        "exit_codes": [e["code"] for e in g["exit_codes"]],
        "park_errors": fig["totals"]["park_errors"],
        "chunks_refused": fig["totals"]["chunks_refused"],
        "log_write_failures": fig["totals"]["log_write_failures"],
        "parked_max_bytes": fig["parked_max_bytes"],
        "decode_tok_s_median": d["tok_s"]["median"] if d["tok_s"] else None,
        "decode_tok_s_p10": d["tok_s"]["p10"] if d["tok_s"] else None,
        "spec_accept_rate_median": rate["median"] if rate else None,
        "spec_accept_rate_p10": rate["p10"] if rate else None,
        "spec_plain_share": sp["plain_share"] if sp else None,
        "spec_accept_rate_lookup": sp["sources"]["lookup"]["accept_rate"] if sp else None,
        "spec_accept_rate_mtp": sp["sources"]["mtp"]["accept_rate"] if sp else None,
        "panic_lines": len(rep["err"]["panics"]),
    }


def parse_when(s):
    """An ISO time as UTC; without an offset it is local time, like guard.log's."""
    t = datetime.fromisoformat(s)
    return local_to_utc(t)[0] if t.tzinfo is None else t.astimezone(UTC)


def in_window(lines, start, end):
    return [line for line in lines if start <= parse_time(line["time"]) <= end]


def main(argv=None):
    p = argparse.ArgumentParser(description="The soak report (spec 9.3). Read-only.")
    p.add_argument("--logs", required=True, type=Path, help="the production JSONL directory")
    p.add_argument("--err", required=True, type=Path, help="artemis.err.log")
    p.add_argument("--guard", required=True, action="append", type=Path, help="guard.log; repeat for guard.log.1")
    p.add_argument("--start", required=True, help="the soak's start (ISO; no offset means local time)")
    p.add_argument("--end", help="the report's end (ISO); default now")
    p.add_argument("--qwen-runs", required=True, type=int, help="completed qwen-queue runs, counted from the forge")
    p.add_argument("--out", type=Path, help="also write the digest to this file")
    p.add_argument("--final", action="store_true", help="write soak-summary.md and append a results.jsonl record")
    p.add_argument("--evidence", type=Path, help="the evidence directory for --final")
    args = p.parse_args(argv)
    if args.final and args.evidence is None:
        p.error("--final needs --evidence DIR")

    now = datetime.now(UTC)
    start = parse_when(args.start)
    end = parse_when(args.end) if args.end else now
    requests, lifecycle = read_logs(args.logs)
    requests, lifecycle = in_window(requests, start, end), in_window(lifecycle, start, end)
    guard = read_guard(args.guard)
    err = err_digest(args.err, start)
    rep = report(requests, lifecycle, guard, err, start, end, args.qwen_runs)
    text = render(rep)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
    if args.final:
        args.evidence.mkdir(parents=True, exist_ok=True)
        (args.evidence / "soak-summary.md").write_text(text, encoding="utf-8", newline="\n")
        with open(args.evidence / "results.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(results_record(rep, now), separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
