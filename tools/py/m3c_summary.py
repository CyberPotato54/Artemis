"""Prints the milestone 3 gate figures from the committed evidence.

Reads results.jsonl, reuse-report.json, smoke-opencode.json, smoke-claude-code.json and the five pool2/<case>.json
verdicts (the supplementary 2 GiB run) and changes nothing. Every figure in the milestone summary is pasted from this
output, except those marked as taken from the ledger (G1's figures are constants below: G1's records were not
committed). Standard library only; run from the repository root:

    python -I tools/py/m3c_summary.py [--log <oracle>\\runs\\m3c-gate.log]
        [--serve-log <oracle>\\runs\\serve\\logs\\artemis-<date>.jsonl]
        [--pool2-log <oracle>\\runs\\m3c-pool2\\logs\\artemis-<date>.jsonl]

`--log` takes the gate run's start from the log's creation time (the runner's own "exit 0; took" line is not in the
log). `--serve-log` and `--pool2-log` are server A's request logs of the gate and of the 2 GiB run (neither is committed)
and add the snapshot-pool tables. Without the options those lines and tables are skipped.
"""

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

GATE_3C = "4c6dde8"
GATE_3A = "7886a96"
# Blocks whose records the 3a gate also holds, in the runner's order. Tests 1 to 5 and 8, with the generate smoke
# and the prefill block that run beside them.
COMPARED = ["8", "2", "2-free (report only)", "1", "2-gpu", "4", "generate (smoke)", "prefill", "3", "5"]
TIMING = {"seconds", "tok_s", "longest_submit_ms"}
IGNORED = {"commit", "date"}

# G1 (2026-10-26 17:21-17:31, the runner's serve, serve-idle and serve-queue blocks at 6e4fe12). Its records were not
# committed; these are the ledger entry for it.
G1 = {
    "load seconds": 12.24,
    "slots per layer (load)": 133,
    "vram_used at A's load (MiB)": 13876.9,
    "vram_used at B's loads (MiB)": 13531.9,
    "disconnect t1-t0 (ms)": 76.3,
    "budget reasoning tokens": 1025,
    "queue 503 (s)": 5.009,
    "idle unload after (s)": 31.0,
    "decode median (tok/s, server log)": 91.53,
    "reuse-report overall rate": 0.691,
    "reuse-report requests": 33,
    "test-messages cache_read_input_tokens": 386,
}


def out(s=""):
    print(s)


def table(headers, rows):
    out("| " + " | ".join(headers) + " |")
    out("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        out("| " + " | ".join(str(c) for c in r) + " |")
    out()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f.read().splitlines() if line.strip()]


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def when(s):
    """An ISO time with the runner's 7 fractional digits, trimmed to 6 for datetime."""
    return datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", s))


def flatten(v, prefix=""):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from flatten(x, f"{prefix}.{k}" if prefix else k)
    elif isinstance(v, list):
        for i, x in enumerate(v):
            yield from flatten(x, f"{prefix}[{i}]")
    else:
        yield prefix, v


def rec_key(r, seen):
    k = (r.get("test"), r.get("input", ""), r.get("kernel", ""))
    n = seen[k]
    seen[k] += 1
    return k + (n,)


def short(v, n=60):
    s = repr(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def top(path):
    return re.split(r"[.\[]", path, maxsplit=1)[0]


def by_test7(recs, case):
    hits = [r for r in recs if r.get("test") == "7" and r.get("input") == case]
    if len(hits) != 1:
        sys.exit(f"expected one test-7 record for {case}, found {len(hits)}")
    return hits[0]


def section_gate(recs, log):
    out("### Gate records")
    out()
    commits = sorted({r["commit"] for r in recs})
    statuses = dict(sorted(Counter(r.get("status") for r in recs).items(), key=lambda kv: str(kv[0])))
    dates = [when(r["date"]) for r in recs]
    out(f"{len(recs)} records, commit(s) {commits}; statuses {statuses}.")
    out(f"First record {min(dates).isoformat()}, last record {max(dates).isoformat()}.")
    if log:
        start = datetime.fromtimestamp(os.path.getctime(log)).astimezone()
        mins = (max(dates) - start).total_seconds() / 60
        out(f"Log created {start.isoformat()}; log creation to the last record {mins:.1f} minutes.")
    out()
    c = Counter(r["test"] for r in recs)
    table(["Block (test field)", "Records", "Statuses"],
          [[t, n, dict(Counter(r.get("status") for r in recs if r["test"] == t))] for t, n in c.items()])


def section_compare(old, new):
    out("### Tests 1 to 5 and 8 against plan 3a's gate (" + GATE_3A + ")")
    out()
    seen = Counter()
    old_by = {rec_key(r, seen): r for r in old}
    seen = Counter()
    new_by = {rec_key(r, seen): r for r in new}
    rows = []
    details = []
    for t in COMPARED:
        ks = [k for k in old_by if k[0] == t]
        nk = [k for k in new_by if k[0] == t]
        equal = 0
        timing_only = 0
        other = Counter()
        missing = len(set(ks) ^ set(nk))
        for k in ks:
            if k not in new_by:
                continue
            a = dict(flatten({x: y for x, y in old_by[k].items() if x not in IGNORED}))
            b = dict(flatten({x: y for x, y in new_by[k].items() if x not in IGNORED}))
            diff = sorted(p for p in set(a) | set(b) if a.get(p, "<absent>") != b.get(p, "<absent>"))
            if not diff:
                equal += 1
                continue
            non_timing = [p for p in diff if top(p) not in TIMING]
            if not non_timing:
                timing_only += 1
            for p in non_timing:
                other[top(p)] += 1
                details.append((k, p, a.get(p, "<absent>"), b.get(p, "<absent>")))
        stat_old = dict(Counter(old_by[k].get("status") for k in ks))
        stat_new = dict(Counter(new_by[k].get("status") for k in nk))
        rows.append([t, f"{len(ks)} / {len(nk)}", f"{stat_old} / {stat_new}", f"{equal}/{len(ks)}",
                     f"{timing_only}/{len(ks)}", dict(other) if other else "-", missing or "-"])
    table(["Block", "Records (3a / 3c)", "Statuses (3a / 3c)", "Equal in every field but commit and date",
           "Differ in timing fields only", "Other fields that differ (records)", "Unmatched records"], rows)
    out('("equal" means every field but commit and date; "timing" fields are seconds, tok_s and longest_submit_ms.)')
    out()
    for label, group in (("3a", old), ("3c", new)):
        pooled = [r for r in group if r["test"] == "3" and r.get("input") == "(pooled)"][0]
        out(f"Test 3 pooled, {label}: {pooled['flips']} flips in {pooled['counted']} counted steps over {pooled['prompts']} prompts "
            f"(floor {pooled['floor_flips']}, bound {pooled['bound']}), {pooled['status']}.")
    out()
    desk = []
    for r in old:
        v = r.get("desktop_gpu_mib")
        if v is None:
            continue
        vals = sorted({x["mib"] for x in v}) if isinstance(v, list) else [v]
        desk.append([r["test"] + (" " + str(r["input"]) if r.get("input") else ""), len(v) if isinstance(v, list) else 1, vals])
    out("The desktop's dedicated GPU memory in use, as 3a's gate recorded it (`desktop_gpu_mib`; this gate's blocks record none):")
    out()
    table(["3a record", "Samples", "Distinct readings (MiB)"], desk)
    if details:
        out("Every field that differs and is not a timing field (3a value | 3c value):")
        out()
        rows = [[f"{k[0]} {k[1] or k[2]}".strip(), p, short(a), short(b)] for k, p, a, b in details]
        table(["Record", "Field", "3a", "3c"], rows)
    else:
        out("No field differs beyond commit, date and timings.")
        out()


def section_test7(recs):
    out("### Test 7 on the real engine")
    out()
    load = by_test7(recs, "load")
    f = load["figures"]
    out(f"**load** ({load['status']}): load {f['load_seconds']:.2f} s ({f['detail']}), {f['slots_per_layer']} slots per layer, "
        f"vram_used {f['vram_used_mib']:,.1f} MiB, pool {f['pool_limit_gib']} GiB (cut: {f['pool_cut']}), unparked {f['unparked']}; "
        f"{f['polls']} /health polls, {f['loading_polls']} read 503 loading (first at {f['first_loading_ms']:.0f} ms, "
        f"last at {f['last_loading_ms']:.0f} ms); the chat answered at {f['chat_ms']:.0f} ms.")
    out()
    out("**vram_used at each load** (the engine's own figure):")
    out()
    idle = by_test7(recs, "idle")["figures"]
    stream = by_test7(recs, "unload-stream")["figures"]
    a = f["vram_used_mib"]
    rows = [
        ["serve: A (first load)", f["slots_per_layer"], f"{a:,.3f}", "-"],
        ["serve-idle: L1", idle["l1_slots_per_layer"], f"{idle['l1_vram_used_mib']:,.3f}", "-"],
        ["serve-idle: L2 (after the idle unload)", idle["l2_slots_per_layer"], f"{idle['l2_vram_used_mib']:,.3f}",
         f"{idle['l2_vram_used_mib'] - idle['l1_vram_used_mib']:+.3f} against L1 (limit 64)"],
        ["serve-idle: L3 (after /v1/unload in a stream)", "not in the record (B pins 128)", f"{stream['l3_vram_used_mib']:,.3f}",
         f"{stream['l3_vram_used_mib'] - stream['l1_vram_used_mib']:+.3f} against the stream's L1 {stream['l1_vram_used_mib']:,.3f} (limit 64)"],
    ]
    table(["Load", "Slots per layer", "vram_used MiB", "Difference"], rows)
    d = a - idle["l1_vram_used_mib"]
    n = f["slots_per_layer"] - idle["l1_slots_per_layer"]
    out(f"A against L1: {d:+.3f} MiB for {n} more slots = {d / n:.3f} MiB a slot (serve-idle pins `--cache-slots 128`, serve takes the default).")
    out()

    req = by_test7(recs, "requests")
    out(f"**requests** ({req['status']}): {len(req['figures']['steps'])} steps, "
        f"{sum(1 for c in req['checks'] if c['pass'])}/{len(req['checks'])} checks pass.")
    out()
    rows = []
    for name, s in req["figures"]["steps"].items():
        rows.append([name, s["status"], s["finish"], s["stream"], s["tool_calls"], s["malformed_calls"], s["think_budget"],
                     s["reasoning_tokens"], s["prompt_tokens"], s["completion_tokens"]])
    table(["Step", "Status", "Finish", "Stream", "Tool calls", "Malformed", "Think budget", "Reasoning tokens",
           "Prompt", "Completion"], rows)

    bud = by_test7(recs, "budget")
    b = bud["figures"]
    out(f"**budget** ({bud['status']}): think_budget {b['think_budget']}, reasoning_tokens {b['reasoning_tokens']} "
        f"(the log says {b['logged_reasoning_tokens']}; {b['think_budget']} plus the close's {b['close_len']} tokens = "
        f"{b['think_budget'] + b['close_len']}, the window is 1024 to 1027), budget_closed {b['budget_closed']}, "
        f"finish {b['finish']}, {b['content_chars']} chars of content, {b['completion_tokens']} completion tokens.")
    out()

    ru = by_test7(recs, "reuse")
    r = ru["figures"]
    out(f"**reuse** ({ru['status']}): conversation A's prompt {r['a_prompt_tokens']} tokens, B's {r['b_prompt_tokens']}; "
        f"scripted calls {r['scripted_calls']}; reply-end hits {r['reply_end_hits']}; pool "
        f"{r['pool_limit_gib']} GiB in every load ({[p['pool_cut'] for p in r['pool_loads']]} cut); inconclusive {r['inconclusive']}.")
    out()
    turns = r["turns"]
    rows = []
    for t in turns:
        rows.append([t["name"], t["scripted"], t["prompt_tokens"], t["cut_at"], t["user_query"], t["reused"], t["restored"], t["cause"],
                     f"{t['restore_bytes']:,}", f"{t['restore_ms']:.2f}", t["prefilled_tokens"]])
    table(["Turn", "Scripted call", "Prompt", "cut_at", "user_query", "reused", "Source", "Cause", "Restore bytes", "Restore ms", "Prefilled"], rows)
    out("Each reuse check as the driver recorded it (floor against `cut_at`):")
    out()
    table(["Check", "Pass", "Detail"], [[c["name"], c["pass"], c["detail"]] for c in ru["checks"]])
    # The floors again, from the turns (the driver's checks read the same numbers).
    tt = {t["name"]: t for t in turns}
    floors = [("t2 reused >= t1.cut_at", "t2", "t1"), ("t3 reused >= t2.cut_at", "t3", "t2"),
              ("t4 reused >= t1.cut_at (the loop's user-query snapshot)", "t4", "t1"),
              ("a5 reused >= t4.cut_at", "a5", "t4"), ("a6 reused >= a5.cut_at", "a6", "a5")]
    rows = [[lbl, tt[x]["reused"], tt[y]["cut_at"], tt[x]["reused"] >= tt[y]["cut_at"],
             tt[x]["reused"] - tt[y]["cut_at"]] for lbl, x, y in floors]
    table(["Floor", "reused", "cut_at", "Holds", "Margin"], rows)
    c5 = tt["a5"]
    out(f"(c): a5 restored `{c5['restored']}`, {c5['restore_bytes']:,} bytes in {c5['restore_ms']:.2f} ms "
        f"({c5['restore_bytes'] / (c5['restore_ms'] / 1000) / 1e9:.1f} GB/s).")
    out()

    dis = by_test7(recs, "disconnect")
    d = dis["figures"]
    out(f"**disconnect** ({dis['status']}): the first request was cancelled after {d['deltas_read']} deltas read, its cut_at "
        f"{d['first_cut_at']}; the second reused {d['second_reused']} (`{d['second_restored']}`), {d['reused_minus_cut']} above "
        f"the floor; t1 - t0 = {d['t1_minus_t0_ms']:.1f} ms (window 0 to 250 ms).")
    out()

    idl = by_test7(recs, "idle")
    i = idl["figures"]
    out(f"**idle** ({idl['status']}): the unload line came {i['unload_after_s']:.2f} s after C's reply (window 30 to 45 s); "
        f"C's next turn reused {i['c_next_reused']} (`{i['c_next_restored']}`).")
    uns = by_test7(recs, "unload-stream")
    out(f"**unload-stream** ({uns['status']}): {uns['figures']['deltas_before_unload']} deltas before the unload, "
        f"{uns['figures']['events_after_unload']} events after it, the last an error event.")
    q = by_test7(recs, "queue")
    qf = q["figures"]
    out(f"**queue** ({q['status']}): the waiting request got {qf['q_status']} after {qf['q_seconds']:.3f} s (window 5.0 to 8.0 s); "
        f"the long request L answered {qf['l_status']}.")
    out()
    out("Checks per test-7 case:")
    out()
    rows = []
    for case in ("load", "requests", "budget", "reuse", "disconnect", "speed", "idle", "unload-stream", "queue"):
        rec = by_test7(recs, case)
        rows.append([case, rec["status"], f"{sum(1 for c in rec['checks'] if c['pass'])}/{len(rec['checks'])}"])
    table(["Case", "Status", "Checks passing"], rows)


def section_stops(recs):
    out("### Stops, key scans and test-messages")
    out()
    rows = []
    for r in recs:
        if r["test"] == "serve stop":
            rows.append([r["input"], r["graceful"], r["exit_code"], r["exit_line"], r["port_free"], r["seconds"]])
    table(["Server block", "Graceful", "Exit code", "Exit line logged", "Port free after", "Seconds"], rows)
    rows = [[r["input"], r["status"], r["files_scanned"]] for r in recs if r["test"] == "serve key scan"]
    table(["Server block", "Key scan", "Files scanned"], rows)
    rows = [[r["input"], r["status"], r["adapter"], r["free_ram_gib"], r["min_free_gib"], r["model_gib"]]
            for r in recs if r["test"] == "serve preflight"]
    table(["Server block", "Preflight", "Adapter", "Free RAM GiB", "Minimum GiB", "Model GiB"], rows)
    tm = [r for r in recs if r["test"] == "test-messages"]
    for r in tm:
        out(f"test-messages: {r['status']}, exit {r['exit_code']}, cache_read_input_tokens {r['cache_read_input_tokens']} "
            f"(the output file stays under the oracle folder's runs\\serve\\).")
    out()


def section_speed(recs, old):
    out("### Speed through the server (reported, not gated)")
    out()
    sp = by_test7(recs, "speed")
    f = sp["figures"]
    out(f"Status {sp['status']}; decode source: {f['decode_source']}.")
    out()
    old_speed = [r for r in old if r["test"] == "speed"][0]
    prev = {(d["prompt"], d["seed"]): d for d in old_speed["decode"]}
    rows = []
    for run in f["runs"]:
        p = prev[(run["prompt"], run["seed"])]
        rows.append([run["prompt"], run["seed"], run["completion_tokens"], f"{run['decode_tok_s']:.1f}", f"{p['tok_s']:.1f}",
                     f"{run['decode_tok_s'] - p['tok_s']:+.1f}"])
    table(["Prompt", "Seed", "Tokens (server)", "Through the server tok/s", "3a speed block tok/s", "Difference"], rows)
    # Milestone 2's own gate median. The record committed at 3c names it `milestone2_llama_decode_tok_s`, wrongly (it is
    # Artemis's figure, not llama.cpp's); later records say `milestone2_decode_tok_s`.
    m2 = f.get("milestone2_decode_tok_s", f.get("milestone2_llama_decode_tok_s"))
    med = statistics.median(f["decode_tok_s"])
    out(f"Decode median through the server {med:.2f} tok/s (the record says {f['decode_median_tok_s']:.2f}; "
        f"{'equal' if abs(med - f['decode_median_tok_s']) < 1e-9 else 'DIFFERENT'}). Beside plan 3a's {f['artemis_3a_decode_tok_s']} "
        f"(its own tool; the script recomputes it from the 3a record: {old_speed['decode_median']:.2f}) and milestone 2's "
        f"own gate {m2}: {100 * (med / f['artemis_3a_decode_tok_s'] - 1):+.1f}% against 3a "
        f"(the record says {f['decode_median_vs_3a_pct']:+.1f}%), {100 * (med / m2 - 1):+.1f}% "
        f"against milestone 2's gate; the 3a gate's bound is {old_speed['decode_bound']} tok/s and {med:.2f} is "
        f"{med / old_speed['decode_bound']:.2f} times it.")
    out()
    pre = [r for r in recs if r["test"] == "7" and r["input"] == "speed"][0]["figures"]["runs"]
    edit = [r for r in pre if r["prompt"] == "edit" and "prefill_tok_s" in r]
    for r in edit:
        out(f"Prefill through the server, edit prompt: {r['prompt_tokens']} tokens at {r['prefill_tok_s']:.1f} tok/s "
            f"(3a's speed block, {old_speed['prefill'][0]['tokens']:,} tokens: {old_speed['prefill_slowest']:.1f}; a different length and stopwatch).")
    out()
    out("The engine itself, in this gate's engine-direct blocks, beside 3a's:")
    out()
    new_pf = [r for r in recs if r["test"] == "prefill" and r["input"] == "long-prompt"][0]
    old_pf = [r for r in old if r["test"] == "prefill" and r["input"] == "long-prompt"][0]
    table(["Figure", "This gate", "3a gate"],
          [["long prompt tok/s", f"{new_pf['tok_s']:.1f}", f"{old_pf['tok_s']:.1f}"],
           ["long prompt longest submit ms", new_pf["longest_submit_ms"], old_pf["longest_submit_ms"]],
           ["long prompt slots", new_pf["cache_slots"], old_pf["cache_slots"]]])


def section_generate(recs, old):
    new_g = [r for r in recs if r["test"] == "generate (smoke)"][0]
    old_g = [r for r in old if r["test"] == "generate (smoke)"][0]
    out(f"Generate smoke (reported): this gate: {new_g['stats']}")
    out()
    out(f"3a: {old_g['stats']}")
    out()
    out(f"Same text as 3a: {new_g['text'] == old_g['text']}.")
    out()


def section_smoke(recs, files):
    out("### The client smoke")
    out()
    rows = []
    for name, v in files.items():
        rec = [r for r in recs if r["test"] == "smoke" and r["input"] == name][0]
        agree = all(rec[k] == v[k] for k in ("status", "version", "attempts", "wire", "reasons", "server"))
        w = v["wire"]
        att = v["attempts"]
        rows.append([name, v["status"], v["version"], len(att),
                     "; ".join(f"{a['seconds']} s, tests {'pass' if a['tests_pass'] else 'fail'}, hosts {a['hosts_status']}, "
                               f"client exit {a['client_exit']}, timed out {a['timed_out']}, guard {a['guard']}" for a in att),
                     w["requests"], w["streamed"], w["tool_calls"], w["malformed_calls"], w["status_5xx"],
                     w["status_other"], w["foreign"], w["unparsed"], w["repairs"] or "none", w["ignored"] or "none",
                     f"{v['server']['graceful']}, exit {v['server']['exit_code']}", agree])
    table(["Client", "Verdict", "Version", "Attempts", "Attempt detail", "Requests", "Streamed", "Tool calls", "Malformed",
           "5xx", "Other non-200", "Foreign", "Unparsed", "Repairs", "Ignored fields", "Server stop",
           "File equals the results.jsonl record"], rows)
    for name, v in files.items():
        out(f"{name}: reasons {v['reasons']}; launch {v['launch_utc']}; preflight free RAM {v['preflight']['free_ram_gib']} GiB "
            f"of a {v['preflight']['min_free_gib']} GiB minimum.")
    out()


def section_reuse(rep, recs):
    out("### Hit rate by source and by cause (spec 5.9)")
    out()
    for src, label in (("serve", "Server A (test 7's `serve` block)"), ("smoke", "The two smoke runs")):
        r = rep[src]
        rec = [x for x in recs if x["test"] == "reuse-report (report only)" and x["input"] == src][0]
        same = all(rec[k] == r[k] for k in ("overall", "by_source", "by_cause", "budgets"))
        o = r["overall"]
        out(f"**{label}**: {o['requests']} requests, {o['reused_tokens']:,} of {o['prompt_tokens']:,} prompt tokens reused "
            f"= {o['rate']:.3f}. The file equals the results.jsonl record: {same}.")
        out()
        for field, title in (("by_source", "Source"), ("by_cause", "Cause")):
            rows = [[k, v["requests"], f"{v['prompt_tokens']:,}", f"{v['reused_tokens']:,}", f"{v['rate']:.3f}"]
                    for k, v in sorted(r[field].items())]
            table([title, "Requests", "Prompt tokens", "Reused tokens", "Rate"], rows)
        rows = [[k, v["requests"], v["closed"]] for k, v in sorted(r["budgets"].items())]
        table(["Think budget", "Requests", "Closed by the budget"], rows)


def section_g1(recs):
    out("### G1 against G2")
    out()
    load = by_test7(recs, "load")["figures"]
    idle = by_test7(recs, "idle")["figures"]
    dis = by_test7(recs, "disconnect")["figures"]
    bud = by_test7(recs, "budget")["figures"]
    q = by_test7(recs, "queue")["figures"]
    sp = by_test7(recs, "speed")["figures"]
    rr = [x for x in recs if x["test"] == "reuse-report (report only)" and x["input"] == "serve"][0]["overall"]
    tm = [x for x in recs if x["test"] == "test-messages"][0]
    g2 = {
        "load seconds": round(load["load_seconds"], 2),
        "slots per layer (load)": load["slots_per_layer"],
        "vram_used at A's load (MiB)": round(load["vram_used_mib"], 1),
        "vram_used at B's loads (MiB)": round(idle["l1_vram_used_mib"], 1),
        "disconnect t1-t0 (ms)": round(dis["t1_minus_t0_ms"], 1),
        "budget reasoning tokens": bud["reasoning_tokens"],
        "queue 503 (s)": round(q["q_seconds"], 3),
        "idle unload after (s)": round(idle["unload_after_s"], 1),
        "decode median (tok/s, server log)": round(sp["decode_median_tok_s"], 2),
        "reuse-report overall rate": round(rr["rate"], 3),
        "reuse-report requests": rr["requests"],
        "test-messages cache_read_input_tokens": tm["cache_read_input_tokens"],
    }
    table(["Figure", "G1 (6e4fe12; from the ledger)", "G2 (" + GATE_3C + ")"], [[k, G1[k], g2[k]] for k in G1])


def pool_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f.read().splitlines():
            r = json.loads(line)
            if r.get("pool_bytes_used") is None or r.get("prompt_tokens") is None:
                continue
            rows.append([r["endpoint"], r["prompt_tokens"], r["reused_tokens"], r["restored"], r["pool_bytes_used"] / 2**20])
    return rows


def section_pool(path, recs):
    """The snapshot pool in use after each of server A's generations. Server A's request log is not committed, so this
    section runs only with --serve-log; it reads token counts and byte counts and no text."""
    out("### The snapshot pool on server A (from its request log, not committed)")
    out()
    rows = pool_rows(path)
    peak = max(rows, key=lambda r: r[4])
    table(["Endpoint", "Prompt tokens", "Reused", "Restored from", "Pool in use MiB"],
          [r[:4] + [f"{r[4]:.1f}"] for r in rows])
    a6 = by_test7(recs, "reuse")["figures"]["turns"][-1]
    at_a6 = [r for r in rows if (r[1], r[2], r[3]) == (a6["prompt_tokens"], a6["reused"], a6["restored"])]
    if len(at_a6) != 1:
        sys.exit(f"the reuse case's last turn matched {len(at_a6)} request lines in the pool log")
    out(f"Pool in use after the reuse case's last turn ({a6['name']}: {a6['prompt_tokens']} prompt tokens, {a6['reused']} reused, "
        f"restored from {a6['restored']}): {at_a6[0][4]:.1f} MiB.")
    out(f"Peak over the run: {peak[4]:.1f} MiB ({peak[4] / 1024:.2f} GiB), at {peak[0]} with {peak[1]} prompt tokens. "
        "A 2 GiB pool is 2,048 MiB.")
    out()


def section_pool_2gib(path):
    """The same reading on the 2 GiB run's log (--pool2-log)."""
    out("### The snapshot pool in the 2 GiB run (from its request log, not committed)")
    out()
    rows = pool_rows(path)
    peak = max(rows, key=lambda r: r[4])
    last = rows[-1]
    out(f"{len(rows)} generation requests logged; the last ({last[0]}, {last[1]} prompt tokens, {last[2]} reused, restored from "
        f"{last[3]}) left {last[4]:.1f} MiB in use. Peak {peak[4]:.1f} MiB against the 2 GiB limit of 2,048 MiB.")
    out()


def section_pool2(d, recs):
    """The supplementary run of server A's pool-sensitive cases at --snapshot-pool-gib 2, outside the gate."""
    out("### Server A at a 2 GiB pool (supplementary, outside the gate)")
    out()
    out("Run after the gate, on the gate binary (" + GATE_3C + "), `artemis test7` against a server started with "
        "`--snapshot-pool-gib 2`; not a runner block and not in results.jsonl. Verdicts: `pool2/<case>.json`.")
    out()
    cases = ["load", "requests", "budget", "reuse", "disconnect"]
    v = {c: read_json(d / "pool2" / (c + ".json")) for c in cases}
    rows = []
    for c in cases:
        g2 = by_test7(recs, c)
        rows.append([c, v[c]["status"], f"{sum(1 for x in v[c]['checks'] if x['pass'])}/{len(v[c]['checks'])}",
                     g2["status"], f"{sum(1 for x in g2['checks'] if x['pass'])}/{len(g2['checks'])}"])
    table(["Case", "2 GiB run", "Checks passing", "G2 (8 GiB)", "Checks passing"], rows)
    lf = v["load"]["figures"]
    g2l = by_test7(recs, "load")["figures"]
    out(f"Load: pool {lf['pool_limit_gib']} GiB, cut {lf['pool_cut']}, {lf['slots_per_layer']} slots a layer "
        f"(G2: {g2l['slots_per_layer']}), vram_used {lf['vram_used_mib']:,.3f} MiB (G2: {g2l['vram_used_mib']:,.3f}), "
        f"{lf['load_seconds']:.2f} s (G2: {g2l['load_seconds']:.2f}), unparked {lf['unparked']}.")
    out()
    for c in ("reuse", "disconnect"):
        f = v[c]["figures"]
        out(f"{c}: pool in each load {[p['pool_limit_gib'] for p in f['pool_loads']]} GiB, cut {[p['pool_cut'] for p in f['pool_loads']]}, "
            f"inconclusive {f['inconclusive']}.")
    out()
    t2 = {t["name"]: t for t in v["reuse"]["figures"]["turns"]}
    t8 = {t["name"]: t for t in by_test7(recs, "reuse")["figures"]["turns"]}
    rows = []
    for n in t2:
        a, b = t2[n], t8[n]
        rows.append([n, a["cut_at"], b["cut_at"], a["reused"], b["reused"], a["restored"], b["restored"],
                     f"{a['restore_bytes']:,}", f"{b['restore_bytes']:,}", f"{a['restore_ms']:.2f}", f"{b['restore_ms']:.2f}"])
    table(["Turn", "cut_at (2 GiB)", "cut_at (G2)", "reused (2 GiB)", "reused (G2)", "Source (2 GiB)", "Source (G2)",
           "Restore bytes (2 GiB)", "Restore bytes (G2)", "Restore ms (2 GiB)", "Restore ms (G2)"], rows)
    same = all(t2[n]["reused"] == t8[n]["reused"] and t2[n]["cut_at"] == t8[n]["cut_at"] and t2[n]["restored"] == t8[n]["restored"]
               and t2[n]["restore_bytes"] == t8[n]["restore_bytes"] for n in t2)
    out(f"Reuse floors, sources and restore bytes identical to G2's: {same}.")
    dd, dg = v["disconnect"]["figures"], by_test7(recs, "disconnect")["figures"]
    out(f"Disconnect: first cut_at {dd['first_cut_at']} (G2 {dg['first_cut_at']}), second reused {dd['second_reused']} (G2 {dg['second_reused']}), "
        f"t1 - t0 {dd['t1_minus_t0_ms']:.1f} ms (G2 {dg['t1_minus_t0_ms']:.1f}).")
    bd, bg = v["budget"]["figures"], by_test7(recs, "budget")["figures"]
    out(f"Budget: {bd['reasoning_tokens']} reasoning tokens (G2 {bg['reasoning_tokens']}).")
    out()


def main():
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(here / "docs" / "superpowers" / "evidence" / "m3"))
    ap.add_argument("--log")
    ap.add_argument("--serve-log", help="server A's request log (jsonl), for the pool table")
    ap.add_argument("--pool2-log", help="the 2 GiB run's request log (jsonl), for its pool reading")
    args = ap.parse_args()
    d = Path(args.dir)
    allrecs = read_jsonl(d / "results.jsonl")
    new = [r for r in allrecs if r["commit"] == GATE_3C]
    old = [r for r in allrecs if r["commit"] == GATE_3A]
    other = [r for r in allrecs if r["commit"] not in (GATE_3A, GATE_3C)]
    if other:
        sys.exit(f"{len(other)} records of another commit")
    rep = read_json(d / "reuse-report.json")
    files = {"opencode": read_json(d / "smoke-opencode.json"), "claude-code": read_json(d / "smoke-claude-code.json")}
    out(f"results.jsonl: {len(allrecs)} records, {len(old)} of {GATE_3A} (plan 3a's gate) and {len(new)} of {GATE_3C} (this gate).")
    out()
    section_gate(new, args.log)
    section_compare(old, new)
    section_generate(new, old)
    section_test7(new)
    section_stops(new)
    section_speed(new, old)
    section_smoke(new, files)
    section_reuse(rep, new)
    section_g1(new)
    section_pool2(d, new)
    if args.serve_log:
        section_pool(args.serve_log, new)
    if args.pool2_log:
        section_pool_2gib(args.pool2_log)


main()
