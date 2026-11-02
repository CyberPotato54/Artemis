"""Prints the milestone 4 gate figures from the committed evidence.

Reads results.jsonl (and, when present, smoke-opencode.json, smoke-claude-code.json and a reuse report) and changes
nothing. Every figure in the milestone summary is pasted from this output. Standard library only; run from the
repository root:

    python -I tools/py/m4a_summary.py [--dir DIR] [--gate COMMIT] [--reuse PATH]

results.jsonl holds one set of records per gate commit (a later GPU window reruns every block for a newer commit).
`--gate` names the commit to report (a prefix of at least 7 characters is enough); the default is the commit of the
last record that has one. Records of any other commit are listed as superseded, with their counts, and not used. The
smokes are not per commit; their files are read as they stand. `--reuse` defaults to <dir>/reuse-report.json when that
file exists. Nothing is printed beyond what the records hold: no prompts, no replies, no key contents (the records
hold key-file paths only, and this script opens none of them).
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
# The bounds for the server's speed (the records carry them too; they are checked against these).
DECODE_BOUND = 46.5
PREFILL_BOUND = 1000.0
# The stall test's slack over its stall time: the run must end within stall_s + this many seconds of the last chunk.
STALL_SLACK_S = 10.0
SMOKES = ["smoke-opencode.json", "smoke-claude-code.json"]


def out(s=""):
    """A line, prose wrapped to 120 columns (a continuation keeps the line's indent plus two); table rows and the
    one-word-per-line hash values are never wrapped."""
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
    """The records of a jsonl file and the lines it could not use: [(line number, why)]. A line that is not valid JSON, or
    not a JSON object, is skipped."""
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
    """An ISO time (the runner writes 7 fractional digits, trimmed to 6), as an aware datetime; None if unparseable. A time
    without an offset is read as UTC."""
    if not isinstance(s, str):
        return None
    try:
        t = datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", s))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def newest_commit(recs):
    """The commit of the record with the latest date (the later one in the file on a tie); file order alone only when no
    record that has a commit also has a readable date."""
    dated = [(when(r.get("date")), i, r["commit"]) for i, r in enumerate(recs) if r.get("commit")]
    good = [x for x in dated if x[0] is not None]
    if good:
        return max(good)[2]
    return dated[-1][2] if dated else None


def distinct(xs):
    """The distinct values of xs, sorted, with None left out."""
    return sorted({x for x in xs if x is not None})


def shown(xs):
    v = distinct(xs)
    return v if v else "none recorded"


def keep_last(recs):
    """Within one gate commit, one record per (test, input): the last. Returns them in file order and the duplicated keys
    with their counts."""
    n = Counter((r.get("test"), r.get("input")) for r in recs)
    seen = Counter()
    kept = []
    for r in recs:
        k = (r.get("test"), r.get("input"))
        seen[k] += 1
        if seen[k] == n[k]:
            kept.append(r)
    return kept, {k: c for k, c in n.items() if c > 1}


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(v, nd=2):
    """A figure for a table cell: '-' for a missing one, nd decimals for a float."""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def ratio(a, b):
    return "-" if a is None or not b else f"{a / b:.3f}"


def nums(xs):
    return [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]


def summary(xs):
    """One list's own summary (count, min, median, max), or '-' for no numbers."""
    xs = nums(xs or [])
    if not xs:
        return "-"
    return f"n={len(xs)} min {min(xs):.2f} median {statistics.median(xs):.2f} max {max(xs):.2f}"


def label(r):
    return f"{r.get('test')} | {r.get('input')}" if r.get("input") is not None else str(r.get("test"))


def find(recs, test, inp=None):
    return [r for r in recs if r.get("test") == test and (inp is None or r.get("input") == inp)]


def last(recs, test, inp=None):
    hits = find(recs, test, inp)
    return hits[-1] if hits else None


def missing(what, gate):
    out(f"**{what}: missing** (no such record at commit {gate}).")
    out()


def same_commit(a, b):
    if not a or not b:
        return False
    return a == b or (min(len(a), len(b)) >= 7 and (a.startswith(b) or b.startswith(a)))


def fmt_figures(fig):
    """Every figure of a record as 'key=value' text. Long number lists become their own summary; lists of records and
    nested records are counted, not printed; long strings are cut."""
    parts = []
    for k, v in (fig or {}).items():
        if isinstance(v, dict):
            parts.append(f"{k}=<{len(v)} entries not printed>")
        elif isinstance(v, list):
            if v and all(isinstance(x, dict) for x in v):
                parts.append(f"{k}=<{len(v)} entries not printed>")
            elif len(nums(v)) == len(v) and len(v) > 6:
                parts.append(f"{k}=[{summary(v)}]")
            else:
                parts.append(f"{k}={json.dumps(v)}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.4g}" if abs(v) < 1000 else f"{k}={v:,.1f}")
        elif isinstance(v, str) and len(v) > 80:
            parts.append(f"{k}={v[:77]!r}...")
        else:
            parts.append(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}")
    return parts


def checks_table(rec):
    checks = rec.get("checks") or []
    table(["Check", "Pass", "Detail"], [[c.get("name"), c.get("pass"), c.get("detail")] for c in checks])


# --- sections ------------------------------------------------------------------------------------------------------

def section_commits(allrecs, recs, gate, path, dups=None, bad=None):
    out("### Gate records")
    out()
    out(f"{path}: {len(allrecs)} records.")
    for n, why in bad or []:
        out(f"Line {n} of {path} skipped: {why}.")
    out()
    for (t, i), c in (dups or {}).items():
        out(f"!!! {c} records of {t}/{i} at {gate}: the last is used !!!")
    if dups:
        out()
    mine = Counter(r.get("commit") for r in allrecs if not same_commit(r.get("commit"), gate))
    out(f"Gate commit: {gate} ({len(recs)} records used, "
         f"first {min(str(r.get('date')) for r in recs)}, last {max(str(r.get('date')) for r in recs)}).")
    if any(c for c in mine):
        for c, n in mine.items():
            if c:
                dates = sorted(str(r.get("date")) for r in allrecs if r.get("commit") == c)
                out(f"Superseded: commit {c}, {n} records ({dates[0]} to {dates[-1]}); not used.")
    else:
        out("Superseded: none (no other commit in the file).")
    nocommit = mine.get(None, 0)
    if nocommit:
        out(f"{nocommit} records carry no commit field (not part of any gate); not used.")
    out()
    statuses = dict(sorted(Counter(r.get("status") for r in recs).items(), key=lambda kv: str(kv[0])))
    out(f"Statuses at the gate commit: {statuses}.")
    out()


def section_hashes(recs):
    out("### The binary and the template")
    out()
    by = {}
    for r in recs:
        if r.get("artemis_sha256"):
            by.setdefault(r["artemis_sha256"], []).append(r)
    if len(by) == 1:
        sha, rs = next(iter(by.items()))
        out(f"artemis_sha256: one value over {len(rs)} measured records: {sha}")
    elif not by:
        out("!!! NO RECORD CARRIES AN artemis_sha256 !!!")
    else:
        out(f"!!! {len(by)} DIFFERENT artemis_sha256 VALUES OVER THE MEASURED RECORDS: THE GATE MEASURED MORE THAN ONE BINARY !!!")
        for sha, rs in by.items():
            out(f"  {sha}: {len(rs)} records, e.g. {label(rs[0])}")
    out()
    # Only the stall test is a test build; any other record that is a server run without the binary's hash is a gap.
    gaps = [r for r in recs if (r.get("config_sha256") or r.get("overrides")) and not r.get("artemis_sha256")
            and r.get("test") != "serve-stall"]
    stall = find(recs, "serve-stall")
    if stall:
        out("The serve-stall record has no artemis_sha256 by design: it is a test build (`cargo test`), not the release "
            "binary the other blocks measure.")
    if gaps:
        out(f"!!! {len(gaps)} server-run records other than serve-stall have no artemis_sha256: "
            f"{[label(r) for r in gaps]} !!!")
    out()
    tpl = {}
    for r in recs:
        if r.get("template_sha256"):
            tpl.setdefault(r["template_sha256"], []).append(r)
    if len(tpl) == 1:
        sha, rs = next(iter(tpl.items()))
        out(f"Template SHA-256: one value over {len(rs)} records: {sha}")
    elif not tpl:
        out("Template SHA-256: no record carries one.")
    else:
        out(f"!!! {len(tpl)} DIFFERENT TEMPLATE SHA-256 VALUES !!!")
        for sha, rs in tpl.items():
            out(f"  {sha}: {len(rs)} records, e.g. {label(rs[0])}")
    templates = sorted({r["config_template"] for r in recs if r.get("config_template")})
    out(f"Template file(s) as recorded: {templates}")
    cfg = Counter(r["config_sha256"] for r in recs if r.get("config_sha256"))
    out(f"Distinct rendered-config SHA-256 values (one per block's run): {len(cfg)}.")
    out()


def section_overrides(recs):
    out("### Each block's overrides")
    out()
    rows = [r for r in recs if isinstance(r.get("overrides"), dict)]
    if not rows:
        out("No record carries overrides.")
        out()
        return
    keys = []
    for r in rows:
        for k in r["overrides"]:
            if k not in keys:
                keys.append(k)
    common = {k: json.dumps(rows[0]["overrides"].get(k)) for k in keys
              if len({json.dumps(r["overrides"].get(k)) for r in rows}) == 1 and k != "log_dir"}
    for k, v in common.items():
        out(f"- {k} (every block): {v}")
    rest = [k for k in keys if k not in common and k != "log_dir"]
    groups = {}
    for r in rows:
        sig = json.dumps({k: r["overrides"].get(k) for k in rest})
        names = groups.setdefault(sig, [])
        name = str(r.get("input") or r.get("test")).replace("serve-speed ", "")
        if name not in names:
            names.append(name)
    if rest:
        out("- Keys that differ between blocks:")
        for sig, names in groups.items():
            out(f"  - {sig}")
            out(f"    blocks: {', '.join(names)}")
    logs = {r["overrides"].get("log_dir") for r in rows}
    out(f"- log_dir: {len(logs)} distinct (each block its own).")
    out()


def section_stall(recs, gate):
    out("### The stall test (serve-stall)")
    out()
    r = last(recs, "serve-stall")
    if not r:
        missing("serve-stall", gate)
        return
    f = r.get("figures") or {}
    t_end = f.get("t_end_minus_t_last_s")
    stall = f.get("stall_s")
    limit = None if stall is None else stall + STALL_SLACK_S
    ok = None if t_end is None or limit is None else t_end <= limit
    out(f"Status {r.get('status')}; cargo test exit {r.get('exit_code')}; stall lines {r.get('stall_lines')}; "
        f"port free after {r.get('port_free')}; problems {r.get('problems')}.")
    out()
    table(["Figure", "Value"], [
        ["load_s", fmt(f.get("load_s"), 3)],
        ["stall_s (the configured stall)", fmt(stall)],
        ["quiet_s (the quiet period measured)", fmt(f.get("quiet_s"), 3)],
        [f"t_end - t_last (limit stall_s + {STALL_SLACK_S:g} = {fmt(limit)})", fmt(t_end, 3)],
        ["t_end - t_last within the limit", ok],
        ["content_chunks", fmt(f.get("content_chunks"))],
        ["server exit code (figures.exit_code)", fmt(f.get("exit_code"))],
        ["figures.problem", fmt(f.get("problem"))],
    ])
    out("This record has no artemis_sha256 by design: the stall test is a test build. It carries "
        f"template_sha256 {str(r.get('template_sha256'))[:16]}... and config_sha256 {str(r.get('config_sha256'))[:16]}....")
    ks = last(recs, "serve-stall key scan")
    out(f"Key scan: {ks.get('status')} over {ks.get('files_scanned')} files." if ks else "Key scan: missing.")
    out()


def speed_rows(recs):
    return find(recs, "serve-speed run")


def section_speed(recs, gate):
    out("### serve-speed against the engine tool's speed, same run")
    out()
    agg = last(recs, "serve-speed")
    tool = last(recs, "speed")
    if not agg:
        missing("serve-speed", gate)
    if not tool:
        missing("speed (the engine tool)", gate)
    runs = speed_rows(recs)
    if not runs:
        missing("serve-speed run records", gate)
    if agg:
        out(f"serve-speed: status {agg.get('status')}, problems {agg.get('problems')}; the record's bounds: decode "
            f"{agg.get('decode_bound')}, prefill {agg.get('prefill_bound')}; long prompt {agg.get('long_prompt_tokens')} tokens "
            f"(reference {agg.get('long_prompt_reference')}).")
        out()
    arows = (agg or {}).get("runs") or []
    steps = [x.get("decode_steps_tok_s") for x in arows if not x.get("long")]
    longs = [x.get("prefill_tok_s") for x in arows if x.get("long")]
    s_med = statistics.median(nums(steps)) if nums(steps) else None
    s_pre = min(nums(longs)) if nums(longs) else None
    t_med = (tool or {}).get("decode_median")
    t_pre = (tool or {}).get("prefill_slowest")
    base = ((tool or {}).get("base") or {}).get("tok_s")
    d_ok = None if s_med is None else s_med >= DECODE_BOUND
    p_ok = None if s_pre is None else s_pre >= PREFILL_BOUND
    table(["Figure", "Server (serve-speed)", "Engine tool (speed)", "Server / tool", "Bound", "Server meets bound"], [
        ["decode median (tok/s)", fmt(s_med), fmt(t_med), ratio(s_med, t_med), DECODE_BOUND, d_ok],
        ["slower long-prompt prefill (tok/s)", fmt(s_pre), fmt(t_pre), ratio(s_pre, t_pre), PREFILL_BOUND, p_ok],
        ["server decode median / tool base (tok/s)", fmt(s_med), fmt(base), ratio(s_med, base), "-", "-"],
    ])
    if agg:
        same_d = agg.get("decode_median") is not None and s_med is not None and abs(agg["decode_median"] - s_med) < 1e-9
        same_p = agg.get("prefill_slowest") is not None and s_pre is not None and abs(agg["prefill_slowest"] - s_pre) < 1e-9
        out(f"Recomputed from the record's runs[]: decode median {fmt(s_med, 4)} over {len(nums(steps))} decode runs "
            f"(the record says {fmt(agg.get('decode_median'), 4)}: {'equal' if same_d else 'DIFFERENT'}); slower prefill "
            f"{fmt(s_pre, 4)} over {len(nums(longs))} long runs (the record says {fmt(agg.get('prefill_slowest'), 4)}: "
            f"{'equal' if same_p else 'DIFFERENT'}).")
        out(f"Server decode run list (decode_steps_tok_s of the runs): {summary(steps)}.")
        out()
    if tool:
        out("The engine tool's speed record carries no artemis_sha256: that it measured the same boot as serve-speed "
            "rests on the runner's order (the speed block follows serve-speed), not on a hash.")
        drows = tool.get("decode") or []
        prows = tool.get("prefill") or []
        out(f"Engine tool: status {tool.get('status')}, gated {tool.get('gated')}, problems {tool.get('problems')}, warnings "
            f"{tool.get('warnings')}. Decode median {fmt(t_med, 4)} (bound {tool.get('decode_bound')}), prefill slowest "
            f"{fmt(t_pre, 4)} (bound {tool.get('prefill_bound')}), base {fmt(base, 4)} tok/s "
            f"({fmt((tool.get('base') or {}).get('decoded'))} tokens, {fmt((tool.get('base') or {}).get('ms_per_token'), 3)} ms/token).")
        out(f"Engine tool decode list (tok_s of its decode rows): {summary([x.get('tok_s') for x in drows])}.")
        out(f"Engine tool prefill list (tok_s of its prefill rows): {summary([x.get('tok_s') for x in prows])}.")
        out()
    if runs:
        rows = []
        mrows = []
        for r in runs:
            f = r.get("figures") or {}
            long = f.get("prompt") == "long"
            toks = f.get("decode_tokens")
            per = None if not toks or f.get("decode_misses") is None else f["decode_misses"] / toks
            ck = f"{sum(1 for c in r.get('checks') or [] if c.get('pass'))}/{len(r.get('checks') or [])}"
            rows.append([r.get("input"), f.get("prompt_tokens"), f.get("completion_tokens"), toks,
                         "-" if long else fmt(f.get("decode_steps_tok_s")), fmt(f.get("prefill_tok_s")), r.get("status"), ck])
            mrows.append([r.get("input"), fmt(f.get("decode_misses")), fmt(per), fmt(f.get("miss_mib_per_token")),
                          fmt(r.get("slots_per_layer")), fmt(r.get("desktop_gpu_mib"), 1)])
        out("Server side, one record per run (a fresh server each):")
        out()
        table(["Run", "Prompt tok", "Completion", "Decode tok", "Decode tok/s", "Prefill tok/s", "Status", "Checks"], rows)
        out("Server side, misses and memory (misses per token = decode_misses / decode_tokens):")
        out()
        table(["Run", "Decode misses", "Misses / token", "Miss MiB / token", "Slots / layer", "Desktop GPU MiB"], mrows)
        per = [r["figures"]["decode_misses"] / r["figures"]["decode_tokens"] for r in runs
               if (r.get("figures") or {}).get("decode_tokens") and r["figures"].get("decode_misses") is not None]
        out(f"Server misses per token over the decode runs: {summary(per)}.")
    if tool:
        vr = [x.get("from_vram_pct") for x in (tool.get("decode") or [])]
        pv = [x.get("from_vram_pct") for x in (tool.get("prefill") or [])]
        out(f"Engine tool, misses: the tool's record has no misses-per-token figure, only the share of routed-expert reads "
            f"served from VRAM, prefill's routes included. Decode rows {summary(vr)} percent; prefill rows {summary(pv)} "
            f"percent. The server/tool ratio for misses per token is therefore not computed.")
        slots = shown(x.get("slots") for x in (tool.get("decode") or []) + (tool.get("prefill") or []))
        held = shown(x.get("held_back_mib") for x in (tool.get("decode") or []) + (tool.get("prefill") or []))
        gpu = shown(x.get("mib") for x in (tool.get("desktop_gpu_mib") or []) if isinstance(x, dict))
        out(f"Engine tool slots per layer {slots}; held back from the budget at load {held} MiB; desktop GPU "
            f"memory in use at each run {gpu} MiB.")
    sl = shown(r.get("slots_per_layer") for r in runs)
    gpu = shown(x.get("mib") for x in ((agg or {}).get("desktop_gpu_mib") or []) if isinstance(x, dict))
    out(f"Server slots per layer {sl}; desktop GPU memory in use at each run {gpu} MiB.")
    out()


def section_mixed(recs, gate):
    out("### Mixed traffic (report only)")
    out()
    r = last(recs, "serve-speed mixed (report only)")
    if not r:
        missing("serve-speed mixed (report only)", gate)
        return
    f = r.get("figures") or {}
    a = f.get("decode_tok_s") or []
    b = f.get("decode_steps_tok_s") or []
    out(f"Status {r.get('status')}; graceful stop {r.get('graceful')}, exit {r.get('exit_code')}, stall lines "
        f"{r.get('stall_lines')}, slots per layer {r.get('slots_per_layer')}, desktop GPU {r.get('desktop_gpu_mib')} MiB; "
        f"skipped {f.get('skipped')}.")
    out()
    out(f"Decode median in the record: {fmt(f.get('decode_median_tok_s'), 4)} tok/s; recomputed over its own decode_tok_s "
        f"list: {fmt(statistics.median(nums(a)) if nums(a) else None, 4)}. Beside plan 3a's {f.get('artemis_3a_decode_tok_s')} "
        f"({fmt(f.get('decode_median_vs_3a_pct'), 1)} percent) and milestone 2's own gate {f.get('milestone2_decode_tok_s')}.")
    out(f"decode_tok_s list (completion tokens over decode time): {summary(a)}.")
    out(f"decode_steps_tok_s list (decode tokens over decode time): {summary(b)}. "
        f"The two lists are summarised on their own and never paired by index.")
    out()
    rows = [[x.get("prompt"), x.get("seed"), x.get("status"), x.get("prompt_tokens"), x.get("completion_tokens"),
             fmt(x.get("prefill_tok_s"), 1), x.get("prefilled_tokens"), x.get("reused_tokens"), x.get("restored")]
            for x in f.get("runs") or []]
    rows = [[fmt(c) for c in row] for row in rows]
    table(["Prompt", "Seed", "HTTP", "Prompt tok", "Completion", "Prefill tok/s", "Prefilled", "Reused", "Restored"], rows)
    checks_table(r)


def section_cases(recs, gate):
    out("### The prod-check cases (test 7 on the production template)")
    out()
    cases = find(recs, "7")
    if not cases:
        missing("test 7 cases", gate)
        return
    rows = [[c.get("input"), c.get("status"), f"{sum(1 for k in c.get('checks') or [] if k.get('pass'))}"
             f"/{len(c.get('checks') or [])}", c.get("stall_lines")] for c in cases]
    table(["Case", "Status", "Checks passing", "Stall lines"], rows)
    for c in cases:
        out(f"**{c.get('input')}** ({c.get('status')})")
        out()
        checks_table(c)
        out("Figures: " + "; ".join(fmt_figures(c.get("figures"))))
        out()
    tm = last(recs, "test-messages")
    if tm:
        out(f"test-messages ({tm.get('input')}): {tm.get('status')}, exit {tm.get('exit_code')}, "
            f"cache_read_input_tokens {tm.get('cache_read_input_tokens')}.")
    else:
        out("test-messages: missing.")
    out()


def section_lifecycle(recs, gate):
    out("### Server stops, key scans and preflights")
    out()
    stops = find(recs, "serve stop")
    if stops:
        table(["Server block", "Graceful", "Exit code", "Exit line", "Port free", "Stall lines", "Seconds"],
              [[r.get("input"), r.get("graceful"), r.get("exit_code"), r.get("exit_line"), r.get("port_free"),
                r.get("stall_lines"), r.get("seconds")] for r in stops])
    else:
        missing("serve stop", gate)
    scans = [r for r in recs if r.get("test") in ("serve key scan", "serve-stall key scan", "serve-speed key scan")]
    if scans:
        table(["Scan", "Block", "Status", "Files scanned"],
              [[r.get("test"), r.get("input"), r.get("status"), r.get("files_scanned")] for r in scans])
    else:
        missing("key scans", gate)
    pre = find(recs, "serve preflight")
    if pre:
        table(["Block", "Status", "Adapter", "Driver", "Free RAM GiB", "Minimum GiB", "Model GiB"],
              [[r.get("input"), r.get("status"), r.get("adapter"), r.get("driver"), r.get("free_ram_gib"),
                r.get("min_free_gib"), r.get("model_gib")] for r in pre])
    else:
        missing("serve preflight", gate)


def section_smokes(d):
    out("### The client smokes (G-d)")
    out()
    for name in SMOKES:
        p = d / name
        if not p.exists():
            out(f"{name}: not yet run.")
            continue
        try:
            v = read_json(p)
            if not isinstance(v, dict):
                raise TypeError("not a JSON object")
        except (OSError, ValueError, TypeError) as e:
            out(f"{name}: present but unreadable ({type(e).__name__}).")
            continue
        out(f"{name}: client {v.get('client')} {v.get('version')}, status {v.get('status')}, launch {v.get('launch_utc')}, "
            f"reasons {v.get('reasons')}.")
        for i, a in enumerate(x for x in (v.get("attempts") or []) if isinstance(x, dict)):
            out(f"  attempt {i + 1}: " + ", ".join(f"{k} {a[k]}" for k in a if not isinstance(a[k], (dict, list))))
        w = v.get("wire")
        if isinstance(w, dict):
            out("  wire: " + ", ".join(f"{k} {json.dumps(x)}" for k, x in w.items()))
        if isinstance(v.get("server"), dict):
            out(f"  server stop: {json.dumps(v['server'])}")
        if isinstance(v.get("preflight"), dict):
            out(f"  preflight: {json.dumps(v['preflight'])}")
    out()


def section_reuse(path, recs, explicit):
    out("### Hit rate by source (reuse report)")
    out()
    rec = last(recs, "reuse-report (report only)")
    if path is None or not Path(path).exists():
        if not rec:
            out("No reuse report" + (f" at {path}" if explicit else " in the evidence directory") + " and no "
                "`reuse-report (report only)` record at the gate commit (pass --reuse PATH).")
            out()
            return
        out("No reuse report" + (f" at {path}" if explicit else " in the evidence directory") + " (pass --reuse PATH); "
            "using the `reuse-report (report only)` record in results.jsonl.")
        rep, source = rec, "results.jsonl record"
    else:
        try:
            rep = read_json(path)
            if not isinstance(rep, dict):
                raise TypeError("not a JSON object")
        except (OSError, ValueError, TypeError) as e:
            out(f"{path}: unreadable ({type(e).__name__}).")
            out()
            return
        source = str(path)
    o = rep.get("overall") or {}
    out(f"{source}: {o.get('requests')} requests, {o.get('reused_tokens')} of {o.get('prompt_tokens')} prompt tokens reused "
        f"= {fmt(o.get('rate'), 3)}.")
    if rec and rep is not rec:
        same = all(rec.get(k) == rep.get(k) for k in ("overall", "by_source", "by_cause", "budgets"))
        out(f"The file equals the results.jsonl record: {same}.")
    out()
    for field, title in (("by_source", "Source"), ("by_cause", "Cause")):
        rows = [[k, v.get("requests"), v.get("prompt_tokens"), v.get("reused_tokens"), fmt(v.get("rate"), 3)]
                for k, v in sorted((rep.get(field) or {}).items())]
        table([title, "Requests", "Prompt tokens", "Reused tokens", "Rate"], rows)
    rows = [[k, v.get("requests"), v.get("closed")] for k, v in sorted((rep.get("budgets") or {}).items())]
    table(["Think budget", "Requests", "Closed by the budget"], rows)


def main(argv=None):
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(here / "docs" / "superpowers" / "evidence" / "m4"))
    ap.add_argument("--gate", help="the gate commit to report (default: the commit of the record with the latest date)")
    ap.add_argument("--reuse", help="a reuse report (default: <dir>/reuse-report.json if it exists)")
    args = ap.parse_args(argv)
    d = Path(args.dir)
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
    # Use the full commit as the records spell it, when --gate was a prefix.
    gate = recs[0]["commit"]
    recs, dups = keep_last(recs)
    reuse = args.reuse or (str(d / "reuse-report.json") if (d / "reuse-report.json").exists() else None)
    section_commits(allrecs, recs, gate, path.name, dups, bad)
    section_hashes(recs)
    section_stall(recs, gate)
    section_speed(recs, gate)
    section_mixed(recs, gate)
    section_cases(recs, gate)
    section_lifecycle(recs, gate)
    section_overrides(recs)
    section_smokes(d)
    section_reuse(reuse, recs, bool(args.reuse))
    return 0


if __name__ == "__main__":
    sys.exit(main())
