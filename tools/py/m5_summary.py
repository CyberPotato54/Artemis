"""Prints the gate figures of plans "5a" and "5b" from the committed evidence.

Reads results.jsonl and changes nothing. Every figure in the milestone summaries is pasted from this output. A record's
`plan` key says which plan's rule judges it: "5b" adds the third "lookup" mode, the per-source counts, the `2-mtp`,
`2-mtp-gpu` and `mtp-32gb` blocks and the stricter status (done-whens 1, 2, 3 and 6); a record with no `plan` (an older
record) is "5a" and is judged as it always was. Standard library
only (Python 3.11 or later); run from the repository root:

    python -I tools/py/m5_summary.py --evidence DIR [--gate COMMIT] [--m4 PATH]

results.jsonl holds records per gate commit (a later GPU window reruns blocks for a newer commit). `--gate` names the
commit to report (a prefix of at least 7 characters is enough); the default is the commit of the record with the latest
date. Records of any other commit are listed as superseded, with their counts, and not used.

The `serve-speed-spec` block may run twice on one commit (one rerun of a miss is allowed, then the maintainer decides).
Every attempt is printed, each with its own figures, and the last is the one that counts. The verdicts printed are the
records' own (`checks`, `status`, `pass`); the script recomputes each from the run records and prints equal or DIFFERENT
beside it.

`--m4` is the milestone 4 results.jsonl, for scale: its line 69 is bf832e6's `serve-speed` record. It defaults to
<DIR>/../m4/results.jsonl; without it the "for scale" columns are blank. Nothing is printed beyond what the records
hold: counts, rates and hashes, no prompts or replies.
"""

import argparse
import json
import re
import statistics
import sys
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

WIDTH = 120
# The pass rules, which no run of this script weakens.
ON_BAR = 110.0  # done-when 1: the "on" median of six
RATIO_BAR = 0.98  # done-when 2 and 3: "on" against "off"
OFF_DECODE_BOUND = 46.5  # done-when 6: the "off" run still meets milestone 4's bounds
OFF_PREFILL_BOUND = 1000.0
PROMPTS = ("write", "edit", "explain")
# bf832e6's `serve-speed` record: line 69 (1-based) of the milestone 4 results.jsonl.
M4_LINE = 69
M4_COMMIT = "bf832e6"
# The block tests whose records repeat per attempt: they are grouped by `attempt`, never reduced to the last.
ATTEMPTED = ("serve-speed-spec", "serve-speed-spec run")
# Under plan "5b", MTP pays iff the "on" median of six is at least 2% over the "lookup" median of six.
# The comparison allows 1e-9 for the rounding of a ratio of two medians (5a's ratio verdicts have none).
MTP_PAYS_RATIO = 1.02
RATIO_EPS = 1e-9
# The spellings that records written before the rename use (put together so this file holds no private name).
OLD_NEEDS_KEY = "needs_" + "dev" + "on"
OLD_ASK_STATUS = "ask-" + "dev" + "on"
# The 32 GB machine keeps a 2 GiB host-cache pool with MTP loaded, else the block is `ask-owner`.
POOL_FLOOR_GIB = 2
# `pool_at_26gib`'s own definition: the pool of a machine with 26 GiB free, capped at the requested 8 GiB.
POOL_FREE_GIB = 26.0
# blk.40's 256 experts in host memory (1,900,544 B each): the growth of the host bytes with MTP on.
MTP_EXPERT_BYTES = 486539264
POOL_CAP_GIB = 8
SPEC_BARS = {"5a": ("dw3", "dw6"), "5b": ("dw1", "dw2", "dw3", "dw6")}  # the done-whens that make the block pass


def out(s=""):
    """A line, prose wrapped to 120 columns (a continuation keeps the line's indent plus two); table rows are never
    wrapped."""
    if len(s) <= WIDTH or s.startswith("|"):
        print(s)
        return
    indent = " " * (len(s) - len(s.lstrip()))
    print(textwrap.fill(s, width=WIDTH, subsequent_indent=indent + "  ", break_long_words=False, break_on_hyphens=False))


def table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def read_jsonl(path):
    """The records of a jsonl file and the lines it could not use: [(line number, why)]."""
    recs, bad = [], []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f.read().splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad.append((n, "not valid JSON"))
                continue
            if not isinstance(r, dict):
                bad.append((n, "not a JSON object"))
                continue
            recs.append(r)
    return recs, bad


def when(s):
    """An ISO time (the runner writes 7 fractional digits, trimmed to 6) as an aware datetime; None if unparseable."""
    if not isinstance(s, str):
        return None
    try:
        t = datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", s))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def newest_commit(recs):
    """The commit of the record with the latest date (the later one in the file on a tie); file order alone only when
    no record that has a commit also has a readable date."""
    dated = [(when(r.get("date")), i, r["commit"]) for i, r in enumerate(recs) if r.get("commit")]
    good = [x for x in dated if x[0] is not None]
    if good:
        return max(good)[2]
    return dated[-1][2] if dated else None


def same_commit(a, b):
    if not a or not b:
        return False
    return a == b or (min(len(a), len(b)) >= 7 and (a.startswith(b) or b.startswith(a)))


def keep_last(recs):
    """Within one gate commit, one record per (test, input): the last, except the attempted blocks, which are kept whole
    (they are grouped by attempt). Returns them in file order and the duplicated keys with their counts."""
    n = Counter((r.get("test"), r.get("input")) for r in recs if r.get("test") not in ATTEMPTED)
    seen = Counter()
    kept = []
    for r in recs:
        if r.get("test") in ATTEMPTED:
            kept.append(r)
            continue
        k = (r.get("test"), r.get("input"))
        seen[k] += 1
        if seen[k] == n[k]:
            kept.append(r)
    return kept, {k: c for k, c in n.items() if c > 1}


def find(recs, test, inp=None):
    return [r for r in recs if r.get("test") == test and (inp is None or r.get("input") == inp)]


def last(recs, test):
    hits = find(recs, test)
    return hits[-1] if hits else None


def nums(xs):
    return [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]


def median(xs):
    """The median: the middle value, or the mean of the middle two (so a median of six is the mean of the 3rd and 4th
    sorted). None for no values."""
    xs = nums(xs)
    return statistics.median(xs) if xs else None


def fmt(v, nd=2):
    """A figure for a table cell: '-' for a missing one, nd decimals for a float."""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def div(a, b):
    return None if a is None or not b else a / b


def ge(a, bar):
    return None if a is None else a >= bar


def every(flags):
    """True if all are True, False if any is False, None when there is nothing to judge (or something unjudged)."""
    flags = list(flags)
    if not flags or any(f is None for f in flags):
        return False if any(f is False for f in flags) else None
    return all(flags)


def verdict(flag):
    return "-" if flag is None else ("pass" if flag else "FAIL")


def agree(a, b, tol=1e-6):
    """Whether a recorded figure and its recomputation are the same: numbers within a relative 1e-6, anything else
    exactly."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b


def same(a, b):
    return "equal" if agree(a, b) else "DIFFERENT"


def figures(r):
    return r.get("figures") or {}


def leaf(inp):
    """A run's name without its mode: 'write-1234-on' is 'write-1234'."""
    return re.sub(r"-(on|off|lookup)$", "", str(inp))


def plan_of(r):
    """The plan a record was made under: its `plan` key, "5a" where it has none (older records predate the key)."""
    p = (r or {}).get("plan")
    return p if isinstance(p, str) and p else "5a"


def plan_of_set(*groups):
    """The plan of an attempt's records (aggregate and runs): the one they all name, or the first with the rest listed
    as a disagreement. Returns (plan, other plans seen)."""
    plans = [plan_of(r) for g in groups for r in g]
    if not plans:
        return "5a", []
    return plans[0], sorted(set(plans) - {plans[0]})


def source_rate(f, name):
    """(drafted, accepted, rate) of one source from a record's figures, or None when the record has no such figures."""
    d, a = f.get(f"spec_drafted_{name}"), f.get(f"spec_accepted_{name}")
    if not isinstance(d, (int, float)) or not isinstance(a, (int, float)) or isinstance(d, bool) or isinstance(a, bool):
        return None
    return d, a, (a / d if d > 0 else None)


def plain_share(f):
    steps, plain = f.get("spec_steps"), f.get("spec_plain_steps")
    if not isinstance(steps, (int, float)) or not isinstance(plain, (int, float)) or steps + plain <= 0:
        return None
    return plain / (steps + plain)


# --- the done-whens ---------------------------------------------------------------------------------------------------


def spec_mode_problems(runs):
    """The runner's `SpecModeProblems` over one attempt's records: which mode each process really ran, from its
    `spec_steps` figure. An "off" decode run with `spec_steps`, an "on" decode run without, and no verify step summed
    over the "on" decode runs are problems. Under plan "5b" a "lookup" decode run must speculate too, and one that
    drafted with MTP (`spec_drafted_mtp` above 0) is a mode problem: its process was not the lookup-only one. The
    1-token `long` runs are skipped. Without these a plain-vs-plain set would recompute to a pass."""
    found = []
    on_rows = on_steps = 0
    for r in runs:
        f = figures(r)
        if f.get("prompt") == "long":
            continue
        steps = f.get("spec_steps")
        if r.get("mode") == "off" and steps is not None:
            found.append(f"{r.get('input')}: spec_steps is {steps}, so the \"off\" process speculated")
        if r.get("mode") == "lookup":
            if steps is None:
                found.append(f"{r.get('input')}: no spec_steps, so the \"lookup\" process did not speculate")
            mtp = f.get("spec_drafted_mtp")
            if isinstance(mtp, (int, float)) and not isinstance(mtp, bool) and mtp > 0:
                found.append(f"{r.get('input')}: spec_drafted_mtp is {mtp}, so the \"lookup\" process drafted with MTP")
        if r.get("mode") == "on":
            on_rows += 1
            if steps is None:
                found.append(f"{r.get('input')}: no spec_steps, so the \"on\" process did not speculate")
            elif isinstance(steps, (int, float)) and not isinstance(steps, bool):
                on_steps += steps
    if on_rows > 0 and on_steps == 0:
        found.append('the "on" decode runs made no verify step between them: speculation never ran')
    return found


def run_sources(r):
    """The draft sources a run was configured with (its `sources` key); the default's, lookup and mtp, where it has none."""
    s = (r or {}).get("sources")
    return [str(x) for x in s] if isinstance(s, list) else ["lookup", "mtp"]


def mtp_never_drafted(runs):
    """The runner's rule: the "on" decode runs whose sources include mtp must have drafted with it
    between them (`spec_drafted_mtp` summing above 0), or the gate would pass with MTP silently off. With the template's
    `sources = ["lookup"]` (the fallback) no "on" run has mtp, and there is nothing to ask."""
    on = [r for r in runs if r.get("mode") == "on" and figures(r).get("prompt") != "long" and "mtp" in run_sources(r)]
    if not on:
        return []
    drafted = sum(
        v for v in (figures(r).get("spec_drafted_mtp") for r in on) if isinstance(v, (int, float)) and not isinstance(v, bool)
    )
    if drafted > 0:
        return []
    return ['the "on" decode runs never drafted with MTP (spec_drafted_mtp is 0 or missing across them), so MTP was off']


def judge(runs, plan="5a"):
    """Done-whens 1, 2, 3 and 6 from one attempt's `serve-speed-spec run` records.

    `plan` picks the status rule: "5a" passes on done-whens 3 and 6 (1 and 2 are reported); "5b" passes only on all
    four. Under "5b" the "lookup" mode's runs give `lookup_median` (the median of six, as "on"), `mtp_gain` (the "on"
    median over it) and `mtp_pays` (the gain at least `MTP_PAYS_RATIO`, report only: it is not in the status), and each
    prompt carries its "lookup" median.

    - `on_median`: the median of the six write/edit/explain "on" runs' `decode_steps_tok_s`, against `ON_BAR`.
    - `per_prompt`: each prompt's "on" and "off" medians over its two seeds, their ratio, and whether it is at least
      `RATIO_BAR`.
    - `prefill`: `long` and `edit`, each the slower of its two runs in a mode, on against off.
    - `off_bounds`: the "off" median of six and its slowest long prefill against milestone 4's bounds.
    A figure is None unless its run set is whole: two runs of the prompt in the mode, each with a number (the prefill
    figures: two runs of `long` or `edit` in the mode). A verdict on a None figure is None, so an attempt with a run
    missing never recomputes to a pass. `problems` lists each run that did not pass or whose `fresh_engine` check is
    not True, plus the speculation-mode problems (`spec_mode_problems`); any of them makes the status "fail"."""
    by = {}
    for r in runs:
        f = figures(r)
        by.setdefault((r.get("mode"), f.get("prompt")), []).append(f)

    def pair(mode, prompt, key):
        """The two numbers of one prompt's runs in a mode, or None when there are not exactly two."""
        xs = nums([f.get(key) for f in by.get((mode, prompt), [])])
        return xs if len(xs) == 2 else None

    def six(mode):
        sets = [pair(mode, p, "decode_steps_tok_s") for p in PROMPTS]
        return None if any(x is None for x in sets) else [v for x in sets for v in x]

    def slower(mode, prompt):
        xs = pair(mode, prompt, "prefill_tok_s")
        return min(xs) if xs else None

    on_six, off_six = six("on"), six("off")
    on_median, off_median = median(on_six or []), median(off_six or [])
    per_prompt = []
    for p in PROMPTS:
        on, off = median(pair("on", p, "decode_steps_tok_s") or []), median(pair("off", p, "decode_steps_tok_s") or [])
        ratio = div(on, off)
        per_prompt.append({"prompt": p, "on_median": on, "off_median": off, "ratio": ratio, "pass": ge(ratio, RATIO_BAR)})
        if plan == "5b":
            lk = median(pair("lookup", p, "decode_steps_tok_s") or [])
            per_prompt[-1]["lookup_median"] = lk
            per_prompt[-1]["on_over_lookup"] = div(on, lk)
    prefill = []
    for which in ("long", "edit"):
        on, off = slower("on", which), slower("off", which)
        ratio = div(on, off)
        prefill.append({"which": which, "on": on, "off": off, "ratio": ratio, "pass": ge(ratio, RATIO_BAR)})
    prefill_slowest = slower("off", "long")
    off_pass = every([ge(off_median, OFF_DECODE_BOUND), ge(prefill_slowest, OFF_PREFILL_BOUND)])
    off_bounds = {
        "decode_median": off_median,
        "decode_bound": OFF_DECODE_BOUND,
        "prefill_slowest": prefill_slowest,
        "prefill_bound": OFF_PREFILL_BOUND,
        "pass": off_pass,
    }
    checks = {
        "dw1": ge(on_median, ON_BAR),
        "dw2": every(p["pass"] for p in per_prompt),
        "dw3": every(p["pass"] for p in prefill),
        "dw6": off_pass,
    }
    problems = []
    for r in runs:
        if r.get("status") != "pass":
            problems.append(f"{r.get('input')}: status {r.get('status')}")
        if (r.get("checks") or {}).get("fresh_engine") is not True:
            problems.append(f"{r.get('input')}: fresh_engine is not true")
    problems.extend(spec_mode_problems(runs))
    if plan == "5b":
        problems.extend(mtp_never_drafted(runs))
    long_tokens = {f.get("prompt_tokens") for f in (figures(r) for r in runs) if f.get("prompt") == "long"} - {None}
    if len(long_tokens) > 1:
        problems.append(
            "the long runs counted different prompts: "
            + ", ".join(f"{t}" for t in (figures(r).get("prompt_tokens") for r in runs if figures(r).get("prompt") == "long") if t is not None)
            + " tokens"
        )
    if plan not in SPEC_BARS:
        problems.append(f"unknown plan {plan!r}: judged by the strictest rule")
    needed = SPEC_BARS.get(plan, SPEC_BARS["5b"])
    lookup_six = six("lookup") if plan == "5b" else None
    if problems or any(checks[k] is False for k in needed):
        status = "fail"
    elif all(checks[k] for k in needed) and checks["dw1"] is not None and checks["dw2"] is not None and (
        plan != "5b" or lookup_six is not None
    ):
        status = "pass"  # 5a: done-whens 3 and 6; 5b: all four and a whole "lookup" set; only a whole run set passes
    else:
        status = None
    result = {
        "on_median": on_median,
        "bar": ON_BAR,
        "counts": {"on": len(on_six or []), "off": len(off_six or [])},
        "per_prompt": per_prompt,
        "prefill": prefill,
        "off_bounds": off_bounds,
        "checks": checks,
        "problems": problems,
        "status": status,
    }
    if plan == "5b":
        lookup_median = median(lookup_six or [])
        gain = div(on_median, lookup_median)
        result.update(
            counts={"on": len(on_six or []), "off": len(off_six or []), "lookup": len(lookup_six or [])},
            lookup_median=lookup_median,
            mtp_gain=gain,
            mtp_pays=None if gain is None else gain >= MTP_PAYS_RATIO - RATIO_EPS,
        )
    return result


def by_attempt(recs, test):
    """{attempt: [records]} of one attempted block's records, in file order."""
    d = {}
    for r in find(recs, test):
        a = r.get("attempt")
        d.setdefault(a if isinstance(a, int) and not isinstance(a, bool) else 0, []).append(r)
    return d


def last_per_input(runs):
    """One record per input within an attempt (the last), and the inputs that repeated."""
    n = Counter(r.get("input") for r in runs)
    seen, kept = Counter(), []
    for r in runs:
        seen[r.get("input")] += 1
        if seen[r.get("input")] == n[r.get("input")]:
            kept.append(r)
    return kept, sorted(k for k, c in n.items() if c > 1)


# --- the milestone 4 reference -----------------------------------------------------------------------------------------


def load_m4(path):
    """bf832e6's `serve-speed` record (line 69 of milestone 4's results.jsonl) and a note: (record, None), or
    (None, why) when the file or the line is not what it should be."""
    path = Path(path)
    if not path.exists():
        return None, f"{path} not found"
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as e:
        return None, f"{path} unreadable ({type(e).__name__})"
    if len(lines) < M4_LINE:
        return None, f"{path} has {len(lines)} lines, not {M4_LINE}"
    try:
        r = json.loads(lines[M4_LINE - 1])
    except ValueError:
        return None, f"{path} line {M4_LINE} is not valid JSON"
    if not isinstance(r, dict) or r.get("test") != "serve-speed" or not same_commit(r.get("commit"), M4_COMMIT):
        return None, f"{path} line {M4_LINE} is not {M4_COMMIT}'s serve-speed record"
    return r, None


def m4_runs(m4):
    return {x.get("run"): x for x in (m4 or {}).get("runs") or [] if isinstance(x, dict)}


# --- sections ---------------------------------------------------------------------------------------------------------


def section_commits(allrecs, recs, gate, name, dups, bad):
    out("### Gate records")
    out()
    out(f"{name}: {len(allrecs)} records.")
    for n, why in bad:
        out(f"Line {n} of {name} skipped: {why}.")
    for (t, i), c in dups.items():
        out(f"!!! {c} records of {t}/{i} at {gate}: the last is used !!!")
    out()
    dates = sorted(str(r.get("date")) for r in recs)
    out(f"Gate commit: {gate} ({len(recs)} records used, first {dates[0]}, last {dates[-1]}).")
    others = Counter(r.get("commit") for r in allrecs if r.get("commit") and not same_commit(r.get("commit"), gate))
    for c, n in others.items():
        ds = sorted(str(r.get("date")) for r in allrecs if r.get("commit") == c)
        out(f"Superseded: commit {c}, {n} records ({ds[0]} to {ds[-1]}); not used.")
    if not others:
        out("Superseded: none (no other commit in the file).")
    nocommit = sum(1 for r in allrecs if not r.get("commit"))
    if nocommit:
        out(f"{nocommit} records carry no commit field (not part of any gate); not used.")
    out()
    statuses = dict(sorted(Counter(r.get("status") for r in recs).items(), key=lambda kv: str(kv[0])))
    out(f"Statuses at the gate commit: {statuses}.")
    out()


def section_binary(recs):
    out("### The binary")
    out()
    by = {}
    for r in recs:
        if r.get("artemis_sha256"):
            by.setdefault(r["artemis_sha256"], []).append(r)
    if len(by) == 1:
        sha, rs = next(iter(by.items()))
        out(f"artemis_sha256: one value over {len(rs)} records: {sha}")
    elif not by:
        out("!!! NO RECORD CARRIES AN artemis_sha256 !!!")
    else:
        out(f"!!! {len(by)} DIFFERENT artemis_sha256 VALUES: THE GATE MEASURED MORE THAN ONE BINARY !!!")
        for sha, rs in by.items():
            out(f"  {sha}: {len(rs)} records, e.g. {rs[0].get('test')} {rs[0].get('input')}")
    out()


def section_speed_spec(recs, gate, m4, m4_note):
    out("### serve-speed-spec (done-whens 1, 2, 3 and 6)")
    out()
    aggs, runs = by_attempt(recs, "serve-speed-spec"), by_attempt(recs, "serve-speed-spec run")
    attempts = sorted(set(aggs) | set(runs))
    if not attempts:
        out(f"**serve-speed-spec: missing** (no such record at commit {gate}).")
        out()
        return
    plans = {plan_of_set(aggs.get(a, []), runs.get(a, []))[0] for a in attempts}
    counting = max([a for a in attempts if a in (1, 2)] or attempts[:1])
    plan = plan_of_set(aggs.get(counting, []), runs.get(counting, []))[0]
    if len(plans) > 1:
        out(f"!!! THE ATTEMPTS NAME DIFFERENT PLANS {sorted(plans)}: the figures below are judged per attempt !!!")
        out()
    out(f"Bars: the \"on\" median of six at least {ON_BAR} tok/s; each prompt's \"on\" median and each prefill figure at "
        f"least {RATIO_BAR} of \"off\"; the \"off\" run at least {OFF_DECODE_BOUND} tok/s and {OFF_PREFILL_BOUND:,.0f} prefill "
        "tok/s. " + (
            "Plan 5b: the block passes iff done-whens 1, 2, 3 and 6 all do. MTP pays iff the \"on\" median of six is at "
            f"least {MTP_PAYS_RATIO} times the \"lookup\" median of six (reported; it is not in the status)."
            if plan == "5b" else
            "The block passes iff done-whens 3 and 6 do (the 5a gate); done-whens 1 and 2 are reported."))
    out()
    if m4 is not None:
        out(f"For scale, {M4_COMMIT}'s serve-speed (line {M4_LINE} of milestone 4's results.jsonl): decode median of six "
            f"{fmt(m4.get('decode_median'))} tok/s, slower long prefill {fmt(m4.get('prefill_slowest'))} tok/s.")
    else:
        out(f"No milestone 4 reference ({m4_note}); the \"for scale\" columns are blank.")
    out()
    valid = [a for a in attempts if a in (1, 2)]
    odd = [a for a in attempts if a not in (1, 2)]
    out(f"Attempts at this commit: {', '.join(str(a) for a in attempts)}; every one is printed below. Only attempts 1 and 2 "
        "are the gate's, and the last of those counts.")
    out()
    stop = []
    if odd:
        stop.append(f"attempt(s) {odd} are not 1 or 2 (the runner allows one rerun, then the owner; an attempt 0 has no number "
                    "or is a lost predecessor)")
    if 2 in attempts and 1 not in attempts:
        stop.append("attempt 2 is here without attempt 1")
    misses = {a: attempt_missed(aggs.get(a, []), runs.get(a, [])) for a in valid}
    if 1 in misses and 2 in misses and misses[1] and misses[2]:
        stop.append(f"attempts 1 and 2 both missed (attempt 1: {', '.join(misses[1])}; attempt 2: {', '.join(misses[2])}): "
                    "a second miss goes to the owner (a done-when 1 or 2 miss goes to 5b through the owner)")
    for why in stop:
        out(f"!!! THIS BLOCK GOES TO THE OWNER: {why} !!!")
    if stop:
        out()
    if valid:
        out(f"The counting attempt is {max(valid)}.")
    else:
        out("No attempt 1 or 2: nothing here counts as the gate.")
    out()
    for a in attempts:
        section_attempt(a, aggs.get(a, []), runs.get(a, []), m4, counts=a in valid)


def attempt_missed(aggs, runs):
    """Why an attempt missed (a miss of done-when 1, 2 or 3 earns the one rerun), as a list of short reasons; empty
    when it did not. Either the record or the recomputation flagging a miss counts. An attempt with no aggregate missed
    too."""
    if not aggs:
        return ["no aggregate record"]
    a = aggs[-1]
    rc = a.get("checks") or {}
    j = judge(last_per_input(runs)[0], plan_of_set(aggs, runs)[0])
    why = []
    if a.get("status") != "pass":
        why.append(f"record status {a.get('status')}")
    if j["status"] == "fail":
        why.append("recomputed status fail")
    for dw in ("dw1", "dw2", "dw3"):
        if rc.get(dw) is False or j["checks"][dw] is False:
            why.append(f"done-when {dw[2]}")
    return why


def section_attempt(attempt, aggs, runs, m4, counts=True):
    out(f"#### Attempt {attempt}" + ("" if counts else " (NOT THE GATE'S)"))
    out()
    if not counts:
        out(f"Attempt {attempt} is not attempt 1 or 2: its figures are listed for reference and no verdict below is the "
            "gate's.")
        out()
    if len(aggs) > 1:
        out(f"!!! {len(aggs)} aggregate records for attempt {attempt}: the last is used !!!")
    agg = aggs[-1] if aggs else None
    runs, repeated = last_per_input(runs)
    if repeated:
        out(f"!!! inputs recorded more than once in attempt {attempt}: {repeated}; the last of each is used !!!")
    if agg is None:
        out(f"**No aggregate record for attempt {attempt}**: the block ended before it wrote one. Its {len(runs)} run "
            "records are below.")
    plan, other = plan_of_set([agg] if agg else [], runs)
    if other:
        out(f"!!! the records of attempt {attempt} name more than one plan ({plan} and {other}); {plan} is used !!!")
    j = judge(runs, plan)
    a = agg or {}
    rc = a.get("checks") or {}
    out((f"Plan {plan}. " if plan == "5b" else "") + f"Record: status {a.get('status')}, attempt {a.get('attempt')}, problems {a.get('problems')}; recomputed status "
        f"{fmt(j['status'])} ({same(a.get('status'), j['status'])}). Runs recorded: {len(runs)}; the \"on\" median uses "
        f"{j['counts']['on']} runs and the \"off\" median {j['counts']['off']}"
        + (f" and the \"lookup\" median {j['counts']['lookup']}" if plan == "5b" else "")
        + " (six each when the block is whole; a figure with a run missing is blank).")
    for why in j["problems"]:
        out(f"!!! run problem: {why} !!!")
    out()
    ob = j["off_bounds"]
    rb = a.get("off_bounds") or {}
    table(
        ["Done-when", "Figure", "Bar", "Record says", "Recomputed", "Agree"],
        [
            [1, f"on median of six {fmt(a.get('on_median'))} (recomputed {fmt(j['on_median'])})", f">= {ON_BAR}",
             verdict(rc.get("dw1")), verdict(j["checks"]["dw1"]),
             same(rc.get("dw1"), j["checks"]["dw1"]) if "dw1" in rc else "no record",
             ],
            [2, "each prompt's ratio, below", f">= {RATIO_BAR} of off", verdict(rc.get("dw2")),
             verdict(j["checks"]["dw2"]), same(rc.get("dw2"), j["checks"]["dw2"]) if "dw2" in rc else "no record"],
            [3, "both prefill figures, below", f">= {RATIO_BAR} of off", verdict(rc.get("dw3")),
             verdict(j["checks"]["dw3"]), same(rc.get("dw3"), j["checks"]["dw3"]) if "dw3" in rc else "no record"],
            [6, f"off median of six {fmt(rb.get('decode_median'))} (recomputed {fmt(ob['decode_median'])}), off long "
                f"prefill {fmt(rb.get('prefill_slowest'))} (recomputed {fmt(ob['prefill_slowest'])})",
             f">= {OFF_DECODE_BOUND} and >= {OFF_PREFILL_BOUND:,.0f}", verdict(rc.get("dw6")),
             verdict(j["checks"]["dw6"]), same(rc.get("dw6"), j["checks"]["dw6"]) if "dw6" in rc else "no record"],
        ],
    )  # fmt: skip
    out(f"On median of six: record {fmt(a.get('on_median'), 4)}, recomputed {fmt(j['on_median'], 4)} "
        f"({same(a.get('on_median'), j['on_median'])}), against {a.get('bar', ON_BAR)}.")
    if plan == "5b" and (j["checks"]["dw1"] is False or rc.get("dw1") is False):
        who = "recomputed and record" if j["checks"]["dw1"] is False and rc.get("dw1") is False else (
            "recomputed" if j["checks"]["dw1"] is False else "record only; the recomputation is blank")
        out(f"Done-when 1 MISSED ({who}): the \"on\" median of six is {fmt(j['on_median'])} tok/s against the {ON_BAR} bar. "
            "Under the Q1 ruling 5b ships at the speed it reaches when MTP pays; the block's status is the "
            "recomputed one above.")
    out()
    out("Done-when 2, per prompt (medians over the two seeds):")
    out()
    rows = []
    recorded = {p.get("prompt"): p for p in a.get("per_prompt") or [] if isinstance(p, dict)}
    for p in j["per_prompt"]:
        r = recorded.get(p["prompt"]) or {}
        rows.append([p["prompt"], fmt(p["on_median"]), fmt(p["off_median"]), fmt(p["ratio"], 4), verdict(r.get("pass")),
                     verdict(p["pass"]), same(r.get("ratio"), p["ratio"]) if r else "no record"])
    table(["Prompt", "On median", "Off median", "On / off", "Record says", "Recomputed", "Ratio agrees"], rows)
    out("Done-when 3, prefill (the slower of the two runs of each):")
    out()
    rows = []
    recorded = {p.get("which"): p for p in a.get("prefill") or [] if isinstance(p, dict)}
    for p in j["prefill"]:
        r = recorded.get(p["which"]) or {}
        rows.append([p["which"], fmt(p["on"]), fmt(p["off"]), fmt(p["ratio"], 4), verdict(r.get("pass")),
                     verdict(p["pass"]), same(r.get("ratio"), p["ratio"]) if r else "no record"])
    table(["Prefill", "On tok/s", "Off tok/s", "On / off", "Record says", "Recomputed", "Ratio agrees"], rows)
    out("Done-when 6, the \"off\" run's bounds:")
    out()
    table(
        ["Figure", "Record's figure", "Recomputed figure", "Bound", "Recomputed meets bound"],
        [
            ["off decode median of six", fmt(rb.get("decode_median")), fmt(ob["decode_median"]), OFF_DECODE_BOUND,
             verdict(ge(ob["decode_median"], OFF_DECODE_BOUND))],
            ["off slowest long prefill", fmt(rb.get("prefill_slowest")), fmt(ob["prefill_slowest"]),
             f"{OFF_PREFILL_BOUND:,.0f}", verdict(ge(ob["prefill_slowest"], OFF_PREFILL_BOUND))],
        ],
    )  # fmt: skip
    out(f"off_bounds.pass: record {verdict(rb.get('pass'))}, recomputed {verdict(ob['pass'])} "
        f"({same(rb.get('pass'), ob['pass']) if 'pass' in rb else 'no record'}).")
    out()
    if plan == "5b":
        section_modes(j, a)
    section_runs(runs, m4)
    if plan == "5b":
        section_sources(runs)
    section_hashes(runs, a.get("identity"))
    gpu = sorted({x for x in nums(a.get("desktop_gpu_mib") or [])})
    out(f"Desktop GPU memory in use at each run: {gpu if gpu else 'none recorded'} MiB.")
    out()


def section_modes(j, a):
    """5b's three-mode speed table (off, lookup, on, per prompt) and the "MTP pays" verdict, the record's beside the
    recomputation. Report only."""
    out("The three modes, per prompt (medians over the two seeds; report only):")
    out()
    table(["Prompt", "Off", "Lookup", "On", "On / off", "On / lookup"],
          [[p["prompt"], fmt(p["off_median"]), fmt(p.get("lookup_median")), fmt(p["on_median"]), fmt(p["ratio"], 4),
            fmt(p.get("on_over_lookup"), 4)] for p in j["per_prompt"]])  # fmt: skip
    out(f"Lookup median of six: record {fmt(a.get('lookup_median'), 4)}, recomputed {fmt(j['lookup_median'], 4)} "
        f"({same(a.get('lookup_median'), j['lookup_median']) if 'lookup_median' in a else 'no record'}); on / lookup "
        f"(mtp_gain): record {fmt(a.get('mtp_gain'), 4)}, recomputed {fmt(j['mtp_gain'], 4)} "
        f"({same(a.get('mtp_gain'), j['mtp_gain']) if 'mtp_gain' in a else 'no record'}).")
    out(f"MTP pays (on at least {MTP_PAYS_RATIO} times lookup): record {fmt(a.get('mtp_pays'))}, recomputed "
        f"{fmt(j['mtp_pays'])} ({same(a.get('mtp_pays'), j['mtp_pays']) if 'mtp_pays' in a else 'no record'}).")
    out()


def section_sources(runs):
    """Each run's drafted and accepted counts by source (the "5b" request-line figures), counts and rates only."""
    def cells(x):
        d, acc, rate = x or (None, None, None)
        return [fmt(d), fmt(acc), fmt(rate, 3)]

    rows = []
    for r in runs:
        lk, mt = source_rate(figures(r), "lookup"), source_rate(figures(r), "mtp")
        if lk is not None or mt is not None:
            rows.append([r.get("input"), r.get("mode")] + cells(lk) + cells(mt))
    if not rows:
        return
    out("Draft counts by source, per run (a rate is blank where the source drafted nothing):")
    out()
    table(["Run", "Mode", "Lookup drafted", "Lookup accepted", "Lookup rate", "MTP drafted", "MTP accepted", "MTP rate"], rows)


def section_runs(runs, m4):
    ref = m4_runs(m4)
    rows, health = [], []
    for r in runs:
        f = figures(r)
        base = ref.get(leaf(r.get("input"))) or {}
        d, p = f.get("decode_steps_tok_s"), f.get("prefill_tok_s")
        rows.append([
            r.get("input"), r.get("mode"), fmt(d), fmt(base.get("decode_steps_tok_s")),
            fmt(div(d, base.get("decode_steps_tok_s")), 3), fmt(p), fmt(base.get("prefill_tok_s")),
            fmt(div(p, base.get("prefill_tok_s")), 3), fmt(f.get("spec_accept_rate"), 3), fmt(plain_share(f), 3),
            fmt(f.get("spec_replays")),
        ])  # fmt: skip
        c = r.get("checks") or {}
        health.append([
            r.get("input"), r.get("status"), r.get("slots_per_layer"), c.get("fresh_engine"), r.get("stall_lines"),
            r.get("graceful"), r.get("exit_code"), r.get("test7_exit"), f.get("decode_misses"), fmt(f.get("miss_mib_per_token"), 1),
        ])  # fmt: skip
    out(f"Each run beside {M4_COMMIT}'s same run (milestone 4, speculation did not exist; a blank is a figure it lacks):")
    out()
    table(
        ["Run", "Mode", "Decode tok/s", f"{M4_COMMIT} decode", "Ratio", "Prefill tok/s", f"{M4_COMMIT} prefill", "Ratio",
         "Accept rate", "Plain share", "Replays"],
        rows,
    )
    out("Each run's health (fresh_engine is a hard check):")
    out()
    table(
        ["Run", "Status", "Slots/layer", "fresh_engine", "Stall lines", "Graceful", "Exit", "Test 7 exit", "Decode misses",
         "Miss MiB/token"],
        health,
    )


def section_hashes(runs, identity):
    """Report only: on and off differ at near-ties by design, so unequal hashes are listed, not judged."""
    out("Completion hashes, off against on (report only; the DeltaNet rows are within tolerance of decode's, so equal is "
        "not expected):")
    out()
    pairs = {}
    for r in runs:
        pairs.setdefault(leaf(r.get("input")), {})[r.get("mode")] = figures(r).get("completion_fnv")
    recorded = {x.get("run"): x.get("equal") for x in identity or [] if isinstance(x, dict)}
    rows = []
    for name, m in pairs.items():
        off, on = m.get("off"), m.get("on")
        eq = None if off is None or on is None else off == on
        rows.append([name, off or "-", on or "-", fmt(eq), fmt(recorded.get(name)), same(recorded.get(name), eq) if name in recorded else "no record"])
    table(["Run", "Off hash", "On hash", "Equal", "Record says", "Agree"], rows)
    judged = [r[3] for r in rows if r[3] in ("True", "False")]
    out(f"Equal in {sum(1 for x in judged if x == 'True')} of {len(judged)} pairs with both hashes.")
    out()


def section_identity(recs, gate):
    out("### Identity reports (report only)")
    out()
    g = last(recs, "spec-golden")
    if g:
        prompts = [p for p in g.get("prompts") or [] if isinstance(p, dict)]
        tokens = sum(1 for p in prompts if p.get("identical_tokens") is True)
        logits = sum(1 for p in prompts if p.get("identical_logits") is True)
        out(f"spec-golden: identical {g.get('identical')} (of the {len(prompts)} prompts listed, {tokens} have identical "
            f"tokens and {logits} identical logits).")
    else:
        out(f"**spec-golden: missing** (no such record at commit {gate}).")
    out()
    fives = find(recs, "5")
    if not fives:
        out(f"**test 5: missing** (no such record at commit {gate}).")
        out()
    for r in fives:
        reports = [x for x in r.get("reports") or [] if isinstance(x, dict)]
        out(f"test 5 ({r.get('input')}): status {r.get('status')}, {len(reports)} report run(s).")
        out()
        if reports:
            table(["Report run", "Identical tokens", "Identical logits"],
                  [[x.get("label"), fmt(x.get("identical_tokens")), fmt(x.get("identical_logits"))] for x in reports])
        else:
            out("No `reports` on this record (an older record, or no report run).")
            out()
    t1 = last(recs, "test 1 identity")
    if t1:
        ks = [k for k in t1.get("kernels") or [] if isinstance(k, dict)]
        out(f"test 1 identity: {len(ks)} kernels, status {t1.get('status')}.")
        out()
        table(["Kernel", "Equal", "Differ"], [[k.get("kernel"), fmt(k.get("equal")), fmt(k.get("differ"))] for k in ks])
    else:
        out(f"**test 1 identity: missing** (no such record at commit {gate}).")
        out()


def spec_line(spec):
    """'steps 150, plain 100, ...' for a record's spec counts, with the accept rate and the plain share worked out."""
    if not isinstance(spec, dict):
        return "none (speculation off)"
    drafted, accepted = spec.get("drafted"), spec.get("accepted")
    rate = div(accepted, drafted) if isinstance(drafted, (int, float)) and drafted > 0 else None
    share = plain_share({"spec_steps": spec.get("steps"), "spec_plain_steps": spec.get("plain_steps")})
    line = (f"steps {spec.get('steps')}, plain {spec.get('plain_steps')}, drafted {drafted}, accepted {accepted}, "
            f"replays {spec.get('replays')}; accept rate {fmt(rate, 3)}, plain share {fmt(share, 3)}")
    parts = []
    for name in ("lookup", "mtp"):
        c = source_rate({f"spec_drafted_{name}": spec.get(f"drafted_{name}"), f"spec_accepted_{name}": spec.get(f"accepted_{name}")}, name)
        if c is not None:
            parts.append(f"{name} {c[1]} of {c[0]} ({fmt(c[2], 3)})")
    if parts:  # plan 5b's per-source counts and the probes
        line += "; " + ", ".join(parts) + f", probes {spec.get('probes')}"
    return line


def golden_status(g, outside):
    """spec-golden's status worked out again: "fail" when a prompt parts outside a near tie or the verdict listed a problem
    other than the subset one (no prompt compared, no verify step); "subset" when the only problem is that fewer than all
    the golden prompts were compared; else "pass", unless the command's exit code was not 0, which is a "fail" (any
    verdict that is not a pass already exits 1, so the code only changes a pass). A 5b record whose on runs never drafted
    with MTP (spec.drafted_mtp is 0) is a fail too, but only when the run's sources include "mtp" and the load had it (the
    record's `sources` and `mtp_loaded`; a record without them is taken to have both): the gate then tested lookup
    alone. A lookup-only run (the fallback) is not failed by this rule."""
    problems = [str(x) for x in g.get("problems") or []]
    if outside > 0 or any("subset run" not in x for x in problems):
        return "fail"
    spec = g.get("spec")
    sources = g.get("sources")
    expects_mtp = "mtp" in sources if isinstance(sources, list) else True
    if g.get("mtp_loaded") is False:
        expects_mtp = False
    if g.get("plan") == "5b" and expects_mtp and isinstance(spec, dict) and spec.get("drafted_mtp") == 0:
        return "fail"
    if problems:
        return "subset"
    code = g.get("exit_code")
    return "fail" if isinstance(code, int) and not isinstance(code, bool) and code != 0 else "pass"


def section_golden(recs, gate):
    out("### spec-golden")
    out()
    g = last(recs, "spec-golden")
    if not g:
        out(f"**spec-golden: missing** (no such record at commit {gate}).")
        out()
        return
    prompts = [p for p in g.get("prompts") or [] if isinstance(p, dict)]
    outside = sum(1 for p in prompts if p.get("at") is not None and p.get("near_tie") is False)
    status = golden_status(g, outside)
    out(f"Verdict {g.get('status')}: {g.get('outside_near_tie')} divergence(s) outside a near tie (recomputed {outside}: "
        f"{same(g.get('outside_near_tie'), outside)}); status recomputed {status} ({same(g.get('status'), status)}).")
    out(f"Speculation counts over the run: {spec_line(g.get('spec'))}.")
    out()
    table(["Prompt", "Diverges at", "Near tie", "Margin", "Identical tokens", "Identical logits"],
          [[p.get("id"), fmt(p.get("at")), fmt(p.get("near_tie")), fmt(p.get("margin"), 4), fmt(p.get("identical_tokens")),
            fmt(p.get("identical_logits"))] for p in prompts])


def section_shrunk(recs, gate):
    out("### serve-shrunk (report only)")
    out()
    s = last(recs, "serve-shrunk")
    if not s:
        out(f"**serve-shrunk: missing** (no such record at commit {gate}).")
        out()
        return
    pairs = [p for p in s.get("pairs") or [] if isinstance(p, dict)]
    low = [p for p in pairs if isinstance(p.get("ratio"), (int, float)) and p["ratio"] < RATIO_BAR]
    # Records written before the rename name the owner by his first name in the list key and the status; they read as
    # the current names. The old spellings are put together so that this file holds no private name.
    needs = s.get("needs_owner") or s.get(OLD_NEEDS_KEY) or []
    status = "ask-owner" if s.get("status") == OLD_ASK_STATUS else s.get("status")
    out(f"Status {status}; {len(pairs)} pairs; {len(needs)} below {RATIO_BAR} listed for the owner "
        f"(recomputed from the pairs: {len(low)}).")
    out()
    table(["Prompt", "Slots", "Off tok/s", "On tok/s", "On / off", "Off MiB/token", "On MiB/token", "Accept rate", "Plain share"],
          [[p.get("prompt"), p.get("slots"), fmt(p.get("off_tok_s")), fmt(p.get("on_tok_s")), fmt(p.get("ratio"), 4),
            fmt(p.get("off_miss_mib_per_token"), 1), fmt(p.get("on_miss_mib_per_token"), 1), fmt(p.get("accept_rate"), 3),
            fmt(p.get("plain_share"), 3)] for p in pairs])  # fmt: skip
    for n in needs:
        if isinstance(n, dict):
            out(f"Needs the owner: {n.get('prompt')} at {n.get('slots')} slots, ratio {fmt(n.get('ratio'), 4)}.")
    out()


def section_long(recs, gate):
    out("### spec-long (report only)")
    out()
    r = last(recs, "spec-long (report only)")
    if not r:
        out(f"**spec-long: missing** (no such record at commit {gate}).")
        out()
        return
    out(f"Status {r.get('status')}; {r.get('prompt_tokens')} prompt tokens; the completions' tokens are identical: "
        f"{fmt(r.get('identical_tokens'))}.")
    out()
    table(["Mode", "Decode tok/s", "Decoded", "Decode misses", "Speculation counts"],
          [[x.get("mode"), fmt(x.get("decode_tok_s")), x.get("decoded"), x.get("decode_misses"), spec_line(x.get("spec"))]
           for x in r.get("runs") or [] if isinstance(x, dict)])
    modes = {x.get("mode"): x for x in r.get("runs") or [] if isinstance(x, dict)}
    on, full = modes.get("on"), modes.get("on-full-window")
    if on and full:  # plan "5b": the 2,048-position window against the whole prompt (report only)
        out(f"Window comparison (report only): \"on\" (the default window) {fmt(on.get('decode_tok_s'))} tok/s, "
            f"\"on-full-window\" {fmt(full.get('decode_tok_s'))} tok/s, full / default "
            f"{fmt(div(full.get('decode_tok_s'), on.get('decode_tok_s')), 4)}.")
        for label, x in (("default window", on), ("full window", full)):
            sp = x.get("spec") if isinstance(x.get("spec"), dict) else {}
            c = source_rate({"spec_drafted_mtp": sp.get("drafted_mtp"), "spec_accepted_mtp": sp.get("accepted_mtp")}, "mtp")
            out(f"  MTP accept rate, {label}: " + (f"{c[1]} of {c[0]} drafted ({fmt(c[2], 3)})" if c else "not recorded") + ".")
        out()


def section_2mtp(recs, gate):
    """Test 2's MTP block (`2-mtp`, the taps against llama.cpp's) and the GPU check (`2-mtp-gpu`): the record's own
    figures, with the two verdicts that follow from them worked out again."""
    out("### 2-mtp and 2-mtp-gpu")
    out()
    t = last(recs, "2-mtp")
    if not t:
        out(f"**2-mtp: missing** (no such record at commit {gate}).")
    else:
        w = t.get("worst") if isinstance(t.get("worst"), dict) else {}
        err, bound = w.get("err"), w.get("bound")
        ok = err <= bound if nums([err, bound]) == [err, bound] else None
        out(f"2-mtp: status {t.get('status')}, form {t.get('form')}, {t.get('taps')} taps; worst tap {w.get('tap')} error "
            f"{err} against bound {bound} (recomputed within bound: {verdict(ok)}).")
        # The runner fails the record when an expected input has no verdict (the loop stopped early); worked out again.
        expected = t.get("expected_inputs")
        if isinstance(expected, list):
            got = {x.get("input"): x.get("status") for x in t.get("inputs") or [] if isinstance(x, dict)}
            bad = [n for n in expected if got.get(n) != "pass"]
            out(f"2-mtp inputs: {len(expected) - len(bad)} of {len(expected)} expected inputs passed"
                f"{' (not passed: ' + ', '.join(map(str, bad)) + ')' if bad else ''}; a missing input fails the record "
                f"(recomputed: {'pass' if not bad else 'fail'}).")
    out()
    g = last(recs, "2-mtp-gpu")
    if not g:
        out(f"**2-mtp-gpu: missing** (no such record at commit {gate}).")
    else:
        pos, top1, ties = g.get("positions"), g.get("top1_equal"), g.get("near_tie_differences")
        whole = None if not nums([pos, top1, ties]) == [pos, top1, ties] else top1 + ties == pos
        out(f"2-mtp-gpu: status {g.get('status')}, {pos} positions, top-1 equal at {top1}, {ties} differences at a near tie "
            f"(every position equal or a near tie: {verdict(whole)}); worst relative L2 {g.get('worst_rel_l2')} (reported).")
    out()


def pool_gib(heap_bytes):
    """`pool_at_26gib`'s definition worked out again: the pool of a machine with `POOL_FREE_GIB` free, whole GiB of free
    minus the heap's measured usage minus the 4 GiB left to the system, capped at `POOL_CAP_GIB`; None where the heap
    figure is missing. A cross-check only: it does not feed the status."""
    if not nums([heap_bytes]):
        return None
    return min(POOL_CAP_GIB, max(0, int(POOL_FREE_GIB - heap_bytes / 2**30 - 4.0)))


def judge_32gb(rec):
    """The `mtp-32gb` status from the recorded runs, as the runner judges it: None (not measured, never a pass) unless the
    "on" run's `host_heap_usage` and `pool_gib` are both numbers; "fail" when the "off" or "on" run has no `host_bytes`,
    the "on" run's `mtp_vram_bytes` is not above 0 (the MTP was not loaded), or the host bytes did not grow by exactly
    `MTP_EXPERT_BYTES` (blk.40's experts); else "pass" iff the "on" run kept `POOL_FLOOR_GIB` or more, else "ask-owner"."""
    runs = [x for x in rec.get("runs") or [] if isinstance(x, dict)]
    on = [x for x in runs if x.get("mode") == "on"]
    off = [x for x in runs if x.get("mode") == "off"]
    run, base = (on[-1] if on else {}), (off[-1] if off else {})
    pool, heap = run.get("pool_gib"), run.get("host_heap_usage")
    if nums([pool, heap]) != [pool, heap]:
        return None
    on_bytes, off_bytes, vram = run.get("host_bytes"), base.get("host_bytes"), run.get("mtp_vram_bytes")
    if nums([on_bytes, off_bytes]) != [on_bytes, off_bytes]:
        return "fail"
    if nums([vram]) != [vram] or vram <= 0:
        return "fail"
    if on_bytes - off_bytes != MTP_EXPERT_BYTES:
        return "fail"
    return "pass" if pool >= POOL_FLOOR_GIB else "ask-owner"


def section_32gb(recs, gate):
    out("### mtp-32gb")
    out()
    r = last(recs, "mtp-32gb")
    if not r:
        out(f"**mtp-32gb: missing** (no such record at commit {gate}).")
        out()
        return
    status = judge_32gb(r)
    out(f"Status {r.get('status')}; recomputed {'not measured' if status is None else status} "
        f"({same(r.get('status'), status)}): \"pass\" iff the pool with MTP on is at least {POOL_FLOOR_GIB} GiB, else "
        f"\"ask-owner\" (not measured when the \"on\" run lacks its heap usage or pool; \"fail\" when the MTP was not loaded or the host bytes did not grow by blk.40's {MTP_EXPERT_BYTES}). Free before the load: "
        f"{r.get('free_before_gib')} GiB (recorded; the worked-out pool is at {POOL_FREE_GIB:g} GiB free, capped at "
        f"{POOL_CAP_GIB}).")
    out()
    rows = []
    for x in r.get("runs") or []:
        if isinstance(x, dict):
            calc = pool_gib(x.get("host_heap_usage"))
            rows.append([x.get("mode"), x.get("host_bytes"), x.get("host_heap_usage"), x.get("mtp_vram_bytes"),
                         fmt(x.get("pool_gib")), fmt(calc), same(x.get("pool_gib"), calc)])
    table(["Mode", "Host bytes", "Host heap usage", "MTP VRAM bytes", "Pool GiB (record)", "Pool GiB (worked out)", "Agree"], rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plan 5a's and 5b's gate figures from the evidence. Read-only.")
    ap.add_argument("--evidence", required=True, help="the evidence directory holding results.jsonl")
    ap.add_argument("--gate", help="the gate commit to report (default: the commit of the record with the latest date)")
    ap.add_argument("--m4", help=f"milestone 4's results.jsonl, for bf832e6 at line {M4_LINE} (default: <evidence>/../m4/results.jsonl)")
    args = ap.parse_args(argv)
    d = Path(args.evidence)
    path = d / "results.jsonl"
    if not path.exists():
        print(f"{path}: not found", file=sys.stderr)
        return 1
    allrecs, bad = read_jsonl(path)
    commits = [r["commit"] for r in allrecs if r.get("commit")]
    if not commits:
        print(f"{path}: no record carries a commit", file=sys.stderr)
        return 1
    gate = args.gate or newest_commit(allrecs)
    recs = [r for r in allrecs if same_commit(r.get("commit"), gate)]
    if not recs:
        print(f"no record of commit {gate}; commits present: {sorted(set(commits))}", file=sys.stderr)
        return 1
    gate = recs[0]["commit"]  # the full commit as the records spell it, when --gate was a prefix
    recs, dups = keep_last(recs)
    m4, note = load_m4(args.m4 or (d.resolve().parent / "m4" / "results.jsonl"))
    section_commits(allrecs, recs, gate, path.name, dups, bad)
    section_binary(recs)
    section_speed_spec(recs, gate, m4, note)
    section_identity(recs, gate)
    section_golden(recs, gate)
    section_shrunk(recs, gate)
    section_long(recs, gate)
    if any(plan_of(r) == "5b" for r in recs):
        section_2mtp(recs, gate)
        section_32gb(recs, gate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
