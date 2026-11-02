"""The request-quality report: reads the server's request logs and prints the quality signals they already record.
Read-only; sends nothing anywhere. Standard library only (Python 3.11 or later).

    python -I tools/py/request_quality.py PATH [PATH ...] [--since YYYY-MM-DD] [--json OUT]

A PATH is a log file or a folder; a folder is searched, at any depth, for `artemis-*.jsonl` (the server's log folder, or
a runner run folder that holds a `logs/` folder). The report is per UTC day and overall: requests, by endpoint and
model, finish reasons, the share that stopped at the token limit (a request that filled the context window counts here
too, as the server ends it with `length`), malformed tool calls, thinking-budget closes, errors, token totals, the
prefix reuse rate, and the slowest requests. A share above its threshold (THRESHOLDS below) is flagged, once a group
holds at least MIN_FLAG_REQUESTS requests. The logs carry no client name, so there is no per-client split; `endpoint`
and `model` are what they record.

The reader skips what it cannot use: a partial last line, a line that is not JSON, lifecycle lines (they carry an
`event`), and fields it does not know. A file it cannot open or read is named on stderr and skipped. A file given
twice, or a file and the folder that holds it, is read once. A field an old log lacks is read as absent. No prompt or
output text is in a log, and none is printed here.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

# A share strictly above its threshold is flagged. They are plain starting points, not measured limits.
THRESHOLDS = {
    "length_share": 0.05,  # requests that ended with `length` (max_tokens or a full context), of all requests
    "malformed_share": 0.02,  # requests with a malformed tool call, of the requests that made or tried a call
    "budget_closed_share": 0.10,  # requests whose reasoning the engine closed at its budget
    "error_share": 0.05,  # failed requests (status 400 or more, except 499, or an error kind)
    "long_reasoning_share": 0.10,  # requests that reasoned for LONG_REASONING_TOKENS or more
}
MIN_FLAG_REQUESTS = 10
LONG_REASONING_TOKENS = 8192
SLOWEST = 5

# The two APIs name the same endings differently; `cancelled` is the server's own.
FINISH_NAMES = {
    "stop": "stop",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "length": "length",
    "max_tokens": "length",
    "tool_calls": "tool_calls",
    "tool_use": "tool_calls",
    "cancelled": "cancelled",
}


def normalize_finish(value):
    if not isinstance(value, str) or not value:
        return "none"
    return FINISH_NAMES.get(value, value)


def num(record, key):
    """An integer or float field, else 0. A bool is not a count."""
    v = record.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return 0
    return v


def has_num(record, key):
    v = record.get(key)
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def day_of(record):
    t = record.get("time")
    if isinstance(t, str) and len(t) >= 10 and t[4] == "-" and t[7] == "-":
        return t[:10]
    return "unknown"


def log_files(paths):
    """The log files of the given files and folders, each once (by resolved path), in the order given."""
    files = []
    seen = set()
    for p in paths:
        p = Path(p)
        found = sorted(p.rglob("artemis-*.jsonl")) if p.is_dir() else [p]
        for f in found:
            try:
                key = f.resolve()
            except OSError:
                key = f
            if key not in seen:
                seen.add(key)
                files.append(f)
    return files


def read_requests(paths, unreadable=None):
    """The request lines of the given files and folders, and how many lines were skipped as unusable.

    A file that cannot be opened or read is skipped, and `(path, reason)` is appended to `unreadable` when given. Lines
    already read from such a file are kept."""
    records = []
    skipped = 0
    for f in log_files(paths):
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        skipped += 1
                        continue
                    if not isinstance(obj, dict):
                        skipped += 1
                        continue
                    if "event" in obj or "endpoint" not in obj:
                        continue
                    records.append(obj)
        except OSError as e:
            if unreadable is not None:
                unreadable.append((str(f), e.strerror or type(e).__name__))
    return records, skipped


def since(records, day):
    return [r for r in records if day_of(r) != "unknown" and day_of(r) >= day]


def is_error(r):
    status = r.get("status")
    failed = has_num(r, "status") and status >= 400 and status != 499
    return failed or bool(r.get("error_kind"))


def share(n, d):
    return n / d if d else None


def summarize(records):
    n = len(records)
    finish = Counter()
    by_endpoint = Counter()
    by_model = Counter()
    tool_call_requests = malformed_calls = malformed_requests = 0
    budget_closed = errors = long_reasoning = 0
    prompt = completion = reasoning = 0
    reuse_prompt = reuse_tokens = 0
    timed = []
    for r in records:
        finish[normalize_finish(r.get("finish"))] += 1
        by_endpoint[str(r.get("endpoint"))] += 1
        if r.get("model") is not None:
            by_model[str(r["model"])] += 1
        calls, bad = num(r, "tool_calls"), num(r, "malformed_calls")
        if calls > 0 or bad > 0:
            tool_call_requests += 1
        if bad > 0:
            malformed_requests += 1
            malformed_calls += bad
        if r.get("budget_closed") is True:
            budget_closed += 1
        if is_error(r):
            errors += 1
        if num(r, "reasoning_tokens") >= LONG_REASONING_TOKENS:
            long_reasoning += 1
        prompt += num(r, "prompt_tokens")
        completion += num(r, "completion_tokens")
        reasoning += num(r, "reasoning_tokens")
        if has_num(r, "reused_tokens") and has_num(r, "prompt_tokens"):
            reuse_prompt += r["prompt_tokens"]
            reuse_tokens += r["reused_tokens"]
        if has_num(r, "total_ms"):
            timed.append(r)
    timed.sort(key=lambda r: r["total_ms"], reverse=True)
    slowest = [
        {
            "id": r.get("id"),
            "time": r.get("time"),
            "endpoint": r.get("endpoint"),
            "total_ms": r["total_ms"],
            "finish": normalize_finish(r.get("finish")),
            "completion_tokens": num(r, "completion_tokens"),
        }
        for r in timed[:SLOWEST]
    ]
    return {
        "requests": n,
        "by_endpoint": dict(by_endpoint),
        "by_model": dict(by_model),
        "finish": dict(finish),
        "length_share": share(finish["length"], n),
        "tool_call_requests": tool_call_requests,
        "malformed_calls": malformed_calls,
        "malformed_requests": malformed_requests,
        "malformed_share": share(malformed_requests, tool_call_requests),
        "budget_closed": budget_closed,
        "budget_closed_share": share(budget_closed, n),
        "errors": errors,
        "error_share": share(errors, n),
        "long_reasoning": long_reasoning,
        "long_reasoning_share": share(long_reasoning, n),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "reuse_rate": share(reuse_tokens, reuse_prompt),
        "slowest": slowest,
    }


def flags_for(summary):
    if summary["requests"] < MIN_FLAG_REQUESTS:
        return []
    out = []
    for name, limit in THRESHOLDS.items():
        value = summary.get(name)
        if value is not None and value > limit:
            out.append({"name": name, "value": value, "threshold": limit})
    return out


def pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


def format_group(title, s, flags):
    lines = [f"== {title}: {s['requests']} requests"]
    if not s["requests"]:
        return "\n".join(lines)
    lines.append("  endpoints: " + ", ".join(f"{k} {v}" for k, v in sorted(s["by_endpoint"].items())))
    if s["by_model"]:
        lines.append("  models: " + ", ".join(f"{k} {v}" for k, v in sorted(s["by_model"].items())))
    lines.append("  finish: " + ", ".join(f"{k} {v}" for k, v in sorted(s["finish"].items())))
    lines.append(f"  stopped at the token limit: {s['finish'].get('length', 0)} ({pct(s['length_share'])})")
    lines.append(
        f"  malformed tool calls: {s['malformed_calls']} in {s['malformed_requests']} of "
        f"{s['tool_call_requests']} tool-call requests ({pct(s['malformed_share'])})"
    )
    lines.append(f"  thinking budget closes: {s['budget_closed']} ({pct(s['budget_closed_share'])})")
    lines.append(
        f"  reasoning of {LONG_REASONING_TOKENS} tokens or more: "
        f"{s['long_reasoning']} ({pct(s['long_reasoning_share'])})"
    )
    lines.append(f"  errors: {s['errors']} ({pct(s['error_share'])})")
    lines.append(
        f"  tokens: prompt {s['prompt_tokens']}, completion {s['completion_tokens']} "
        f"(reasoning {s['reasoning_tokens']})"
    )
    lines.append(f"  prefix reuse rate: {pct(s['reuse_rate'])}")
    if s["slowest"]:
        lines.append("  slowest:")
        for x in s["slowest"]:
            lines.append(
                f"    {x['total_ms']} ms  {x['time']}  {x['endpoint']}  {x['finish']}  "
                f"{x['completion_tokens']} completion tokens  id {x['id']}"
            )
    for f in flags:
        lines.append(f"  FLAG {f['name']}: {pct(f['value'])} is above {pct(f['threshold'])}")
    return "\n".join(lines)


def build_report(records, skipped, unreadable=()):
    days = {}
    for r in records:
        days.setdefault(day_of(r), []).append(r)
    day_summaries = {d: summarize(rs) for d, rs in sorted(days.items())}
    overall = summarize(records)
    return {
        "days": day_summaries,
        "overall": overall,
        "flags": {"overall": flags_for(overall), **{d: flags_for(s) for d, s in day_summaries.items()}},
        "skipped_lines": skipped,
        "unreadable_files": [{"path": p, "reason": r} for p, r in unreadable],
        "thresholds": THRESHOLDS,
        "min_flag_requests": MIN_FLAG_REQUESTS,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Report the quality signals in the server's request logs.")
    ap.add_argument("paths", nargs="+", help="log files or folders")
    ap.add_argument("--since", help="only requests from this UTC day on (YYYY-MM-DD)")
    ap.add_argument("--json", metavar="OUT", help="also write the report as JSON to this file")
    args = ap.parse_args(argv)

    if args.since:
        try:
            date.fromisoformat(args.since)
        except ValueError:
            print(f"--since {args.since!r} is not a date (YYYY-MM-DD)", file=sys.stderr)
            return 2
    for p in args.paths:
        if not Path(p).exists():
            print(f"{p}: no such file or folder", file=sys.stderr)
            return 2

    unreadable = []
    records, skipped = read_requests(args.paths, unreadable)
    for path, reason in unreadable:
        print(f"{path}: cannot read ({reason}), skipped", file=sys.stderr)
    if args.since:
        records = since(records, args.since)
    if not records:
        print("no request lines found" + (f" (skipped {skipped} unusable lines)" if skipped else ""))
        report = build_report(records, skipped, unreadable)
    else:
        report = build_report(records, skipped, unreadable)
        for d, s in report["days"].items():
            print(format_group(d, s, report["flags"][d]))
            print()
        print(format_group("overall", report["overall"], report["flags"]["overall"]))
        print(f"\nskipped unusable lines: {skipped}")
        if unreadable:
            print(f"skipped unreadable files: {len(unreadable)}")
        print(
            "flag thresholds: "
            + ", ".join(f"{k} > {pct(v)}" for k, v in THRESHOLDS.items())
            + f"; groups under {MIN_FLAG_REQUESTS} requests are not flagged"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
