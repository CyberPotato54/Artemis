"""Prints plan 6b's gate tables from the committed evidence, and checks that the m5c figures still hold.

Reads results.jsonl and changes nothing. Every figure in the milestone summary is pasted from this output. Standard
library only (Python 3.11 or later); run from the repository root:

    python -I tools/py/m6b_summary.py --evidence docs/superpowers/evidence/m6 --gate COMMIT

`--gate` names the commit to report (a prefix of at least 7 characters is enough); the default is the commit of the
record with the latest date. Records of any other commit are left out and counted at the top.

The tables, in order: the speed hygiene (the link, the scanners' CPU seconds and the busy percentage; a block whose
scanners used more than 30 CPU seconds or whose start was more than 10% busy is flagged) and the link after each speed
block, `llama-search`, `speed-llama`,
`speed-llama-65k`, `attn-depth`, `first-run` (every attempt; the last counts) and the m5c hold. A llama.cpp record that
names no device is flagged: llama.cpp's device is chosen by name, and a figure with no device cannot back the claim.

The m5c hold (`m5c_hold`) judges the counting attempt of `serve-speed-spec` against 0.97 of milestone 5c's figures: the
"on" median of six at least 133.6 tok/s (0.97 x 137.75), the "off" median of six at least 106.7 (0.97 x 110.01) and the
slowest "off" long prefill at least 2,061 tok/s (0.97 x 2,125). The counting attempt follows m5_summary.py: only
attempts 1 and 2 are the gate's, the higher of them counts (the later record on a tie), and any other attempt number, a
record with none, or an attempt 2 without an attempt 1 fails the hold and is named, because the rerun budget is one per
commit and the rest goes to the owner. The script exits 1 when one of the three misses or is missing, naming the figure.
"""

import argparse
import json
import re
import statistics
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

WIDTH = 120
# Gate bars (plan 6b, Global Constraints).
HAND_BAR = 0.97  # the tuned median of six against serve-speed-spec's "on" median in the same session
LLAMA_BAR = 1.0  # the tuned median of six must be above llama.cpp's searched best (ratio above 1)
SETUP_LIMIT_S = 600  # setup's own seconds, loads included; mirrors SETUP_LIMIT_S in crates/tools/src/setup/choose.rs
MIN_PCIE_GEN = 4
# CPU quietness (a speed block's hygiene and link-after records): the background scanners (Windows Defender, the search
# indexer) may use at most this many CPU seconds during a block, and the PC may be at most this busy at its start. A
# block over either is flagged: such a PC slows llama.cpp's CPU expert threads (about 2.3 times on 2026-11-01).
SCANNER_CPU_LIMIT_S = 30.0
BUSY_PCT_LIMIT = 10.0
# Milestone 5c's figures and 0.97 of each: the m5c hold.
M5C_ON, M5C_OFF, M5C_PREFILL = 137.75, 110.01, 2125.0
HOLD_ON, HOLD_OFF, HOLD_PREFILL = 133.6, 106.7, 2061.0
# Test names are written "<name> (report only)" by the runner for the llama.cpp blocks.
REPORT_ONLY = " (report only)"


def base_name(test):
    """A record's test name without the runner's ' (report only)' suffix."""
    t = test if isinstance(test, str) else ""
    return t[: -len(REPORT_ONLY)] if t.endswith(REPORT_ONLY) else t


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


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def nums(xs):
    return [x for x in xs if num(x)]


def median(xs):
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


def fmt_device(d):
    """A llama.cpp device for a table cell: the runner records {id, name}, printed as `id: name`; a plain string (an
    older record) is printed as it is."""
    if isinstance(d, dict):
        ident, name = d.get("id"), d.get("name")
        if ident is not None and name is not None:
            return f"{ident}: {name}"
        return fmt(name if name is not None else ident)
    return fmt(d)


def scanner_cell(d):
    """The scanners' CPU seconds as `Defender / indexer` (a missing one is '-')."""
    d = d if isinstance(d, dict) else {}
    return f"{fmt(d.get('MsMpEng'), 1)} / {fmt(d.get('SearchIndexer'), 1)}"


def div(a, b):
    return a / b if num(a) and num(b) and b else None


def verdict(flag):
    return "-" if flag is None else ("pass" if flag else "FAIL")


def agree(a, b, tol=1e-6):
    """Whether a recorded figure and its recomputation are the same: numbers within a relative 1e-6, else exactly."""
    if num(a) and num(b):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b


def same(a, b):
    return "equal" if agree(a, b) else "DIFFERENT"


def dget(d, *keys):
    """d[k1][k2]... where every step is a dict; None when one is missing."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def last_of(recs, test):
    hits = [r for r in recs if base_name(r.get("test")) == test]
    return hits[-1] if hits else None


def last_per(recs, test, key):
    """The last record of `test` for each value of `key`, in order of first appearance."""
    seen, order = {}, []
    for r in recs:
        if base_name(r.get("test")) != test:
            continue
        k = r.get(key)
        if k not in seen:
            order.append(k)
        seen[k] = r
    return [seen[k] for k in order]


GATE_ATTEMPTS = (1, 2)


def counting_attempt(recs, test):
    """The record of `test` that counts: of attempts 1 and 2 the higher, the later one in the file on a tie; None when
    neither attempt is there (m5_summary.py's rule: no other number is the gate's)."""
    best, best_n = None, None
    for r in recs:
        if base_name(r.get("test")) != test:
            continue
        n = r.get("attempt")
        if not num(n) or n not in GATE_ATTEMPTS:
            continue
        if best is None or n >= best_n:
            best, best_n = r, n
    return best


def attempt_problems(recs, test):
    """What is wrong with the attempt numbers of `test`'s aggregates, as sentences: an attempt that is not 1 or 2, a
    record with none, an attempt 2 without an attempt 1. Empty when all is as the runner writes it."""
    nums_seen = []
    out = []
    for r in recs:
        if base_name(r.get("test")) != test:
            continue
        n = r.get("attempt")
        if not num(n):
            out.append(f"{test}: a record has no attempt number")
        elif n not in GATE_ATTEMPTS:
            out.append(f"{test}: attempt {n:g} is not 1 or 2 (the budget is one rerun per commit, then the owner)")
        nums_seen.append(n)
    if 2 in nums_seen and 1 not in nums_seen:
        out.append(f"{test}: attempt 2 without attempt 1")
    return list(dict.fromkeys(out))


def at_gate(records, gate):
    return [r for r in records if same_commit(r.get("commit"), gate)]


# ---------------------------------------------------------------------------------------------------------------------
# The m5c hold.


def hold_figures(rec):
    """The three figures the hold judges, from a serve-speed-spec aggregate: (name, value, m5c figure, bar, unit)."""
    ob = rec.get("off_bounds")
    long_prefill = next((p for p in rec.get("prefill") or [] if isinstance(p, dict) and p.get("which") == "long"), None)
    off_prefill = dget(ob, "prefill_slowest")
    if off_prefill is None:
        off_prefill = dget(long_prefill, "off")
    return [
        ('the "on" median of six', rec.get("on_median"), M5C_ON, HOLD_ON, "tok/s"),
        ('the "off" median of six', dget(ob, "decode_median"), M5C_OFF, HOLD_OFF, "tok/s"),
        ('the slowest "off" long prefill', off_prefill, M5C_PREFILL, HOLD_PREFILL, "tok/s"),
    ]


def m5c_hold(records, gate):
    """Whether milestone 5c's speed holds at the gate commit: (ok, problems). The counting attempt of serve-speed-spec
    must give an "on" median of six >= 133.6, an "off" median of six >= 106.7 and a slowest "off" long prefill >= 2,061.
    A missing record or figure is a problem naming it."""
    recs = at_gate(records, gate)
    if not any(base_name(r.get("test")) == "serve-speed-spec" for r in recs):
        return False, [f"no serve-speed-spec record for commit {gate}"]
    problems = attempt_problems(recs, "serve-speed-spec")
    rec = counting_attempt(recs, "serve-speed-spec")
    if rec is None:
        return False, problems + [f"serve-speed-spec: no attempt 1 or 2 at commit {gate}, so nothing counts"]
    where = f"serve-speed-spec attempt {rec.get('attempt'):g}"
    figures = [(name, value, bar, unit) for name, value, _, bar, unit in hold_figures(rec)]
    for name, value, bar, unit in figures:
        if not num(value):
            problems.append(f"{where}: {name} is missing, so the m5c figure (bar {bar:g} {unit}) cannot be checked")
        elif value < bar:
            problems.append(f"{where}: {name} is {value:g} {unit}, below the bar {bar:g}")
    return not problems, problems


# ---------------------------------------------------------------------------------------------------------------------
# The tables. Each returns lines; `summarize` joins them.


def wrap(s):
    """A line, prose wrapped to 120 columns (a continuation keeps the line's indent plus two); table rows never wrap."""
    if len(s) <= WIDTH or s.startswith("|"):
        return [s]
    indent = " " * (len(s) - len(s.lstrip()))
    return textwrap.fill(
        s, width=WIDTH, subsequent_indent=indent + "  ", break_long_words=False, break_on_hyphens=False
    ).split("\n")


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    lines.append("")
    return lines


def section_hygiene(recs):
    """One row per hygiene record of a block (a block run twice has two rows). A link-after record pairs with the
    latest passing hygiene record before it in file order: a hygiene that failed stopped its block, so it has no
    link-after, and the rerun's link-after is not given to it. FLAG lines for a failed hygiene, a low gen and a missing
    or dropped link."""
    lines = ["## Speed hygiene and the link", ""]
    slots = {}
    for r in recs:
        t = r.get("test")
        if isinstance(t, str) and t.endswith(" hygiene"):
            slots.setdefault(t[: -len(" hygiene")], []).append([r, None])
        elif isinstance(t, str) and t.endswith(" link after"):
            mine = slots.setdefault(t[: -len(" link after")], [])
            open_slot = next(
                (x for x in reversed(mine) if x[0] is not None and x[0].get("status") == "pass" and x[1] is None), None
            )
            if open_slot is not None:
                open_slot[1] = r
            else:
                mine.append([None, r])
    if not slots:
        return lines + ["no hygiene record for this commit.", ""]
    rows = []
    flags = []
    for b, pairs in slots.items():
        n_rows = len(pairs)
        for i, (h, a) in enumerate(pairs):
            label = b if n_rows == 1 else f"{b} ({i + 1} of {n_rows})"
            hh = (h or {}).get("hygiene") or {}
            gen = hh.get("pcie_gen")
            gen_after = (a or {}).get("pcie_gen")
            dropped = (a or {}).get("link_dropped")
            rows.append([
                label, fmt((h or {}).get("status")), fmt(gen), fmt(hh.get("adapter")), fmt(hh.get("affinity")),
                fmt(hh.get("desktop_gpu_mib"), 1), fmt(hh.get("busy_pct"), 1),
                scanner_cell((a or {}).get("scanner_cpu_delta_s")), fmt((a or {}).get("status")), fmt(gen_after),
                fmt(dropped),
            ])
            if h is None:
                flags.append(f"{label}: no hygiene record before the block")
            elif h.get("status") != "pass":
                flags.append(f"{label}: hygiene {h.get('status')}: {h.get('problem') or 'no reason recorded'}")
            elif num(gen) and gen < MIN_PCIE_GEN:
                flags.append(f"{label}: PCIe gen {gen} before the block, below {MIN_PCIE_GEN}")
            busy = hh.get("busy_pct")
            if num(busy) and busy > BUSY_PCT_LIMIT:
                flags.append(f"{label}: the PC was {busy:g}% busy before the block, above {BUSY_PCT_LIMIT:g}%")
            used = [v for v in ((a or {}).get("scanner_cpu_delta_s") or {}).values() if num(v)]
            if sum(used) > SCANNER_CPU_LIMIT_S:
                flags.append(
                    f"{label}: the background scanners used {sum(used):.1f} CPU seconds during the block, above "
                    f"{SCANNER_CPU_LIMIT_S:g}"
                )
            if a is None:
                if h is not None and h.get("status") == "pass":
                    flags.append(f"{label}: no link-after record")
            elif dropped or a.get("status") != "pass":
                why = a.get("why") or "no reason recorded"
                flags.append(f"{label}: link after: {a.get('problem') or a.get('status')} ({why})")
    lines += table(
        ["block", "hygiene", "gen", "adapter", "affinity", "desktop MiB", "busy % at start",
         "scanner CPU s (Defender / indexer)", "link after", "gen after", "dropped"],
        rows,
    )
    lines += [f"FLAG {f}" for f in flags]
    if flags:
        lines.append("")
    return lines


def section_llama_search(recs):
    lines = ["## llama-search (report only)", ""]
    r = last_of(recs, "llama-search")
    if r is None:
        return lines + ["no record for this commit.", ""]
    best = r.get("best") if isinstance(r.get("best"), dict) else {}
    device = r.get("device")
    runs = llama_runs(best)
    rows = [[
        fmt(r.get("status")), fmt_device(device), fmt(r.get("variants")), fmt(r.get("budget_minutes")),
        fmt(r.get("budget_hit")),
        fmt(best.get("affinity")), fmt(median(runs), 1) if runs else "-", fmt(best.get("median"), 1),
        " ".join(str(x) for x in best.get("args") or []) or "-",
    ]]
    lines += table(
        ["status", "device", "variants", "budget min", "budget hit", "best affinity", "runs median", "best median",
         "best args"],
        rows,
    )
    if not device:
        lines += ["FLAG llama-search: the record names no device", ""]
    if r.get("status") != "ran":
        why = r.get("problem") or r.get("error") or "no reason recorded"
        lines += [f"FLAG llama-search: status {r.get('status')}: {why}", ""]
    return lines


def llama_runs(r):
    """The decode figures of a speed-llama record's runs: a list of numbers or of {tok_s} dicts."""
    out = []
    for x in r.get("runs") or []:
        v = x.get("tok_s") if isinstance(x, dict) else x
        if num(v):
            out.append(v)
    return out


def section_llama(recs, test, title):
    lines = [f"## {title}", ""]
    r = last_of(recs, test)
    if r is None:
        return lines + ["no record for this commit.", ""]
    runs = llama_runs(r)
    rows = [[
        fmt(r.get("status")), fmt(r.get("config_source")), fmt_device(r.get("device")), fmt(r.get("affinity")),
        fmt(r.get("affinity_at_stop")), fmt(r.get("affinity_asked")), fmt(len(runs)), fmt(r.get("median"), 1),
        fmt(median(runs), 1) if runs else "-",
    ]]
    lines += table(
        ["status", "config", "device", "affinity", "at stop", "asked", "runs", "median", "recomputed"], rows
    )
    flags = []
    if not r.get("device"):
        flags.append(f"{test}: the record names no device")
    if r.get("status") == "fail":
        flags.append(f"{test}: failed: {r.get('problem') or r.get('error') or 'no reason recorded'}")
    elif r.get("median") is None and not r.get("error"):
        flags.append(f"{test}: no median and no error recorded")
    elif r.get("error"):
        flags.append(f"{test}: error recorded (a figure at 65k only): {r.get('error')}")
    if num(r.get("median")) and runs and not agree(r.get("median"), median(runs)):
        flags.append(f"{test}: the recorded median {r.get('median')} differs from the runs' {median(runs)}")
    lines += [f"FLAG {f}" for f in flags]
    if flags:
        lines.append("")
    return lines


def section_attn(recs):
    lines = ["## attn-depth", ""]
    aggs = [r for r in recs if base_name(r.get("test")) == "attn-depth"]
    if not aggs:
        return lines + ["no record for this commit.", ""]
    for a in aggs:
        stage = a.get("stage")
        runs = [r for r in last_per([r for r in recs if r.get("stage") == stage], "attn-depth run", "input")]
        rows = []
        for r in runs:
            rows.append([
                fmt(r.get("input")), fmt(r.get("base_tok_s"), 1), fmt(r.get("new_tok_s"), 1), fmt(r.get("ratio"), 4),
                fmt(r.get("fingerprints_equal")),
            ])
        lines.append(
            f"stage {fmt(stage)}: status {fmt(a.get('status'))}, gain at 65,000 positions {fmt(a.get('gain_65k'), 4)}, "
            f"link dropped {fmt(a.get('link_dropped'))}"
        )
        lines.append("")
        lines += table(["depth", "base tok/s", "new tok/s", "new / base", "fingerprints equal"], rows)
        for p in a.get("problems") or []:
            lines += wrap(f"FLAG attn-depth stage {fmt(stage)}: {p}")
        if a.get("problems"):
            lines.append("")
    return lines


def tuned_runs(recs, attempt):
    """The first-run decode runs of one attempt: the last record per input."""
    mine = [r for r in recs if r.get("attempt") == attempt]
    return last_per(mine, "first-run run", "input")


def decode_of(run):
    f = run.get("figures") if isinstance(run.get("figures"), dict) else {}
    v = f.get("decode_steps_tok_s")
    return v if num(v) else run.get("decode_steps_tok_s")


def hand_ok_of(a, hand):
    """Whether the tuned median meets the hand-tuned bar: tuned >= 0.97 x the "on" median (a product, with the 1e-9
    allowance mtp_pays has), else the ratio against the bar when a median is missing. None when nothing can be
    judged."""
    tuned, on = a.get("tuned_median"), a.get("spec_on_median")
    if num(tuned) and num(on):
        return tuned >= HAND_BAR * on - 1e-9
    return None if hand is None else hand >= HAND_BAR - 1e-9


def first_run_ratios(a):
    """(tuned_over_hand, tuned_over_llama): from the medians when both are there, else the recorded ratio."""
    hand = div(a.get("tuned_median"), a.get("spec_on_median"))
    llama = div(a.get("tuned_median"), a.get("llama_median"))
    hand = hand if hand is not None else a.get("tuned_over_hand")
    llama = llama if llama is not None else a.get("tuned_over_llama")
    return hand, llama


def section_first_run(recs):
    lines = ["## first-run", ""]
    aggs = [r for r in recs if r.get("test") == "first-run"]
    prechecks = [r for r in recs if r.get("test") == "first-run precheck"]
    for r in prechecks:
        lines += wrap(f"FLAG first-run precheck failed (not an attempt): {r.get('error') or 'no reason recorded'}")
    if prechecks:
        lines.append("")
    if not aggs:
        return lines + ["no first-run record for this commit.", ""]
    launches = [r for r in recs if r.get("test") == "first-run launch"]
    lines.append(f"{len(aggs)} attempt(s); the last counts.")
    lines.append("")
    for p in attempt_problems(recs, "first-run"):
        lines += wrap(f"FLAG {p}")
    if attempt_problems(recs, "first-run"):
        lines.append("")
    for a in aggs:
        n = a.get("attempt")
        hand, llama = first_run_ratios(a)
        hand_ok = hand_ok_of(a, hand)
        llama_ok = None if llama is None else llama > LLAMA_BAR
        launch = next((r for r in reversed(launches) if r.get("attempt") == n), None)
        setup_raw = (launch or {}).get("setup")
        setup = setup_raw if isinstance(setup_raw, dict) else {}
        seconds, limit_hit = setup.get("seconds"), setup.get("limit_hit")
        setup_ok = None if seconds is None else (seconds <= SETUP_LIMIT_S and not limit_hit)
        if launch is None:
            launch_flag = "no first-run launch record for this attempt"
        elif not isinstance(setup_raw, dict):
            launch_flag = "the launch record has no setup_done line (setup did not finish)"
        else:
            launch_flag = None
        if launch_flag:
            setup_ok = False
        net = (launch or {}).get("network_imports")
        net_text = "-" if net is None else (", ".join(net) if isinstance(net, list) and net else (net or "none"))
        runs = tuned_runs(recs, n)
        recomputed = median([decode_of(r) for r in runs]) if runs else None
        lines.append(f"### attempt {fmt(n)}: status {fmt(a.get('status'))}")
        lines.append("")
        lines += table(["figure", "value", "bar", "result"], [
            ["tuned median of six, Artemis at 65,536 positions (tok/s)", fmt(a.get("tuned_median")), "-", "-"],
            ["serve-speed-spec \"on\" median (tok/s)",
             f"{fmt(a.get('spec_on_median'))} (serve-speed-spec status {fmt(a.get('spec_status'))})", "-", "-"],
            ["tuned / hand-tuned", fmt(hand, 4), f">= {HAND_BAR}", verdict(hand_ok)],
            ["llama.cpp searched best median, 32,768 positions (tok/s)", fmt(a.get("llama_median")), "-", "-"],
            ["tuned / llama.cpp (Artemis 65,536 vs llama.cpp 32,768 positions)", fmt(llama, 4), f"> {LLAMA_BAR:g}",
             verdict(llama_ok)],
            ["setup seconds", fmt(seconds, 1), f"<= {SETUP_LIMIT_S}", verdict(setup_ok)],
            ["setup limit hit", fmt(limit_hit), "false", verdict(None if limit_hit is None else not limit_hit)],
            ["first launch to /health (s)", fmt((launch or {}).get("first_launch_s"), 1), "-", "-"],
            ["WinHTTP or WinINet imports", net_text, "none", verdict(None if net is None else not net)],
        ])
        lines += wrap(
            f"llama.cpp (32,768 positions; Artemis runs 65,536): device {fmt(a.get('llama_device'))}, "
            f"affinity {fmt(a.get('llama_affinity'))}, configuration {fmt(a.get('llama_config_source'))}. "
            f"Link dropped {fmt(a.get('link_dropped'))}."
        )
        ratio_notes = []
        for label, field, value in (
            ("tuned / hand-tuned", "tuned_over_hand", hand),
            ("tuned / llama.cpp", "tuned_over_llama", llama),
        ):
            recorded = a.get(field)
            if num(recorded) and num(value):
                verdict_word = same(recorded, value)
                ratio_notes.append(f"{label} {fmt(recorded, 4)} ({verdict_word})")
                if verdict_word != "equal":
                    lines += wrap(
                        f"FLAG first-run attempt {fmt(n)}: the recorded {label} {fmt(recorded, 4)} differs from the "
                        f"medians' {fmt(value, 4)}"
                    )
        if ratio_notes:
            lines += wrap("Recorded ratios against the medians: " + ", ".join(ratio_notes) + ".")
        if launch_flag:
            lines += wrap(f"FLAG first-run attempt {fmt(n)}: {launch_flag}")
        if runs:
            lines += wrap(
                f"The {len(runs)} tuned run records give a median of {fmt(recomputed)} "
                f"(record {fmt(a.get('tuned_median'))}, {same(a.get('tuned_median'), recomputed)})."
            )
            bad_mask = [r.get("input") for r in runs if r.get("affinity") != r.get("affinity_at_stop")]
            not_applied = [r.get("input") for r in runs if r.get("tuned_applied") is not True]
            if bad_mask:
                lines += wrap(f"FLAG runs whose mask changed between start and stop: {', '.join(map(str, bad_mask))}")
            if not_applied:
                lines += wrap(f"FLAG runs whose tuned line did not apply the file: {', '.join(map(str, not_applied))}")
        else:
            lines.append("No first-run run record for this attempt.")
        ident = a.get("identity")
        if isinstance(ident, list) and ident:
            eq = sum(1 for x in ident if isinstance(x, dict) and x.get("equal") is True)
            lines.append(f"Identity (report only): {eq} of {len(ident)} tuned completions equal the \"on\" run's.")
        elif ident is not None:
            lines.append(f"Identity (report only): {ident}")
        for p in a.get("problems") or []:
            lines += wrap(f"FLAG first-run attempt {fmt(n)}: {p}")
        lines.append("")
    return lines


def section_hold(records, gate):
    ok, problems = m5c_hold(records, gate)
    recs = at_gate(records, gate)
    rec = counting_attempt(recs, "serve-speed-spec")
    lines = ["## m5c hold (serve-speed-spec, the counting attempt)", ""]
    if rec is None:
        lines += table(["figure", "m5c", "bar", "value", "result"], [])
        for p in problems:
            lines += wrap("FAIL " + p)
        return lines + [""]
    rows = []
    for name, value, base, bar, _ in hold_figures(rec):
        ok_here = None if not num(value) else value >= bar
        rows.append([name.replace("the ", "", 1), f"{base:g}", f">= {bar:g}", fmt(value), verdict(ok_here)])
    lines.append(f"Attempt {fmt(rec.get('attempt'))}, status {fmt(rec.get('status'))}.")
    lines.append("")
    lines += table(["figure", "m5c", "bar (0.97 x m5c)", "value", "result"], rows)
    lines += wrap(("m5c holds." if ok else "FAIL m5c does not hold:"))
    for p in problems:
        lines += wrap(f"  {p}")
    lines.append("")
    return lines


def summarize(records, gate):
    """The 6b tables for the records of commit `gate` (a prefix of 7 or more characters is enough), as text. Records
    of other commits are left out and counted in the header."""
    recs = at_gate(records, gate)
    others = {}
    for r in records:
        c = r.get("commit")
        if c and not same_commit(c, gate):
            others[c] = others.get(c, 0) + 1
    full = next((r["commit"] for r in recs if r.get("commit")), gate)
    lines = [f"# Plan 6b gate figures, commit {full}", ""]
    lines += wrap(f"{len(recs)} records of this commit used.")
    if others:
        lines += wrap(
            "Left out (other commits): " + ", ".join(f"{c} ({n})" for c, n in sorted(others.items())) + "."
        )
    lines.append("")
    lines += section_hygiene(recs)
    lines += section_llama_search(recs)
    lines += section_llama(recs, "speed-llama", "speed-llama (report only)")
    lines += section_llama(recs, "speed-llama-65k", "speed-llama-65k (report only)")
    lines += section_attn(recs)
    lines += section_first_run(recs)
    lines += section_hold(records, gate)
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plan 6b's gate tables from the evidence, and the m5c hold. Read-only.")
    ap.add_argument("--evidence", required=True, help="the evidence directory holding results.jsonl")
    ap.add_argument("--gate", help="the gate commit to report (default: the commit of the record with the latest date)")
    args = ap.parse_args(argv)
    path = Path(args.evidence) / "results.jsonl"
    if not path.exists():
        print(f"{path}: not found", file=sys.stderr)
        return 1
    records, bad = read_jsonl(path)
    for n, why in bad:
        print(f"{path}:{n}: {why}", file=sys.stderr)
    gate = args.gate or newest_commit(records)
    if not gate or not at_gate(records, gate):
        present = sorted({r["commit"] for r in records if r.get("commit")})
        print(f"no record of commit {gate}; commits present: {present}", file=sys.stderr)
        return 1
    sys.stdout.write(summarize(records, gate))
    ok, _ = m5c_hold(records, gate)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
