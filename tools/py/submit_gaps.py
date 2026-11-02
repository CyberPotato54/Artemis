"""Where a speculative decode's wall time goes between and around its GPU work, from `artemis generate`'s submit trace.

Run a generate with ARTEMIS_SUBMIT_TRACE=FILE (the trace covers decode only), then, from the repository root:

    python -I tools/py/submit_gaps.py TRACE.json [TRACE.json ...] [--json OUT]

Each traced compute submit carries host times (the recorder made, vkQueueSubmit called and returned, the wait's
return) and its first and last GPU timestamps mapped onto the host clock. Per submit:

- host: from the previous submit's wake to this submit's vkQueueSubmit (sampling, drafting, the policy's prepare,
  the input writes and the recording; `record` is the part after the recorder was made)
- launch: from the vkQueueSubmit call to the GPU's first timestamp
- gpu: the GPU's first timestamp to its last
- wake: the GPU's last timestamp to the host's wake
- idle: the GPU's time between the previous submit's last timestamp and this one's first (host + launch + the
  previous wake)

host + launch + gpu + wake is the submit's wall time, from the previous wake to its own. A submit is classed by its
label and its neighbours: `mtp` is the MTP draft pass; a `verify` right after an `mtp` verified MTP drafts, any other
verified lookup drafts; a `replay` (the commit's DeltaNet replay) takes its verify's class; `decode` is a plain step. A
step is the submits of one speculative step: plain [decode], lookup [verify, replay], MTP [mtp, verify, replay] (an MTP
pass that drafted nothing is followed by a plain decode, and the pair counts as an MTP step with no verify). Per class
and per step type it prints the median and p90 of each gap in microseconds and each gap's share of the wall time. The
first submit of each trace has no previous wake and is left out. Standard library only.
"""

import argparse
import json
import math
import statistics
import sys

GAPS = ["host", "record", "launch", "gpu", "wake", "idle", "wall"]
# The four that add up to the wall time; `record` is part of `host`, and `idle` overlaps them.
SHARES = ["host", "launch", "gpu", "wake"]


def p90(v):
    s = sorted(v)
    return s[max(0, math.ceil(0.9 * len(s)) - 1)]


def per_submit(subs):
    """Each submit after the first, with its class and its gaps in microseconds."""
    out = []
    prev_kind = None
    verify_class = None
    for i, s in enumerate(subs):
        kind = s["kind"]
        if kind == "verify":
            verify_class = "verify_mtp" if prev_kind == "mtp" else "verify_lookup"
            cls = verify_class
        elif kind == "replay":
            cls = "replay_mtp" if verify_class == "verify_mtp" else "replay_lookup"
        elif kind == "mtp":
            cls = "mtp_draft"
        elif kind == "decode":
            cls = "decode"
        else:
            cls = kind
        prev_kind = kind
        if i == 0:
            continue
        p = subs[i - 1]
        us = lambda ns: ns / 1e3  # noqa: E731
        host = us(s["submit_ns"] - p["woke_ns"])
        g = {
            "host": host,
            "record": us(s["submit_ns"] - max(s["record_ns"], p["woke_ns"])),
            "launch": us(s["gpu_start_ns"] - s["submit_ns"]),
            "gpu": us(s["gpu_end_ns"] - s["gpu_start_ns"]),
            "wake": us(s["woke_ns"] - s["gpu_end_ns"]),
            "idle": us(s["gpu_start_ns"] - p["gpu_end_ns"]),
            "wall": us(s["woke_ns"] - p["woke_ns"]),
        }
        out.append({"class": cls, "kind": kind, "gaps": g, "cal_err_us": s["cal_err_ns"] / 1e3})
    return out


def steps(rows):
    """Groups the submits into speculative steps: (type, [rows])."""
    out = []
    cur = None
    for r in rows:
        c = r["class"]
        if c == "mtp_draft":
            cur = ["mtp", [r]]
            out.append(cur)
        elif c in ("verify_mtp", "replay_mtp") and cur is not None and cur[0] == "mtp":
            cur[1].append(r)
        elif c == "decode" and cur is not None and cur[0] == "mtp" and len(cur[1]) == 1:
            # An MTP pass that drafted nothing: the plain decode after it closes the step.
            cur[1].append(r)
            cur = None
        elif c == "verify_lookup":
            cur = ["lookup", [r]]
            out.append(cur)
        elif c == "replay_lookup" and cur is not None and cur[0] == "lookup":
            cur[1].append(r)
        elif c == "decode":
            cur = None
            out.append(["plain", [r]])
        else:
            cur = None
            out.append([c, [r]])
    return [(t, rs) for t, rs in out]


def summarize(groups):
    """Per group: count, median and p90 of each gap, and the share of the wall time of the four that add up to it."""
    out = {}
    for name, items in groups.items():
        if not items:
            continue
        wall = sum(g["wall"] for g in items)
        out[name] = {
            "n": len(items),
            "median_us": {k: round(statistics.median(g[k] for g in items), 1) for k in GAPS},
            "p90_us": {k: round(p90([g[k] for g in items]), 1) for k in GAPS},
            "share_of_wall": {k: round(sum(g[k] for g in items) / wall, 4) for k in SHARES + ["record"]},
        }
    return out


def analyze(subs):
    rows = per_submit([s for s in subs if s["kind"] != "prefill"])
    by_class = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r["gaps"])
    by_step = {}
    for t, rs in steps(rows):
        by_step.setdefault(t, []).append({k: sum(r["gaps"][k] for r in rs) for k in GAPS})
    wall = sum(r["gaps"]["wall"] for r in rows)
    gpu = sum(r["gaps"]["gpu"] for r in rows)
    return {
        "submits": len(rows),
        "wall_ms": round(wall / 1e3, 2),
        "gpu_busy_ms": round(gpu / 1e3, 2),
        "gpu_busy_share": round(gpu / wall, 4) if wall else None,
        "max_cal_err_us": round(max((r["cal_err_us"] for r in rows), default=0.0), 2),
        "by_class": summarize(by_class),
        "by_step": summarize(by_step),
    }


def table(title, summary):
    lines = [title]
    head = f"  {'':16}{'n':>5}" + "".join(f"{k:>16}" for k in ["host", "record", "launch", "gpu", "wake", "idle"])
    lines.append(head + "   shares host/launch/gpu/wake")
    for name, s in summary.items():
        med, hi, sh = s["median_us"], s["p90_us"], s["share_of_wall"]
        cells = "".join(f"{med[k]:>8.1f}/{hi[k]:<7.1f}" for k in ["host", "record", "launch", "gpu", "wake", "idle"])
        shares = "/".join(f"{100 * sh[k]:.1f}" for k in SHARES)
        lines.append(f"  {name:16}{s['n']:>5}{cells}   {shares} %")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--json", help="write the summaries here")
    a = ap.parse_args(argv)
    out = {}
    for path in a.traces:
        with open(path, encoding="utf-8") as f:
            subs = json.load(f)["submits"]
        r = analyze(subs)
        out[path] = r
        print(
            f"{path}: {r['submits']} submits, wall {r['wall_ms']} ms, GPU busy {r['gpu_busy_ms']} ms "
            f"({100 * r['gpu_busy_share']:.1f}%), calibration within {r['max_cal_err_us']} us"
        )
        print("  median/p90 us per submit class:")
        print(table("", r["by_class"]))
        print("  median/p90 us per step (sums of its submits):")
        print(table("", r["by_step"]))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
