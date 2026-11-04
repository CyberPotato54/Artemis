"""mtp_route_hits: how much of block 40's expert traffic would a few VRAM slots serve.

Input: one or more route logs written with ARTEMIS_MTP_ROUTES=FILE (one JSON line per draft row:
{"pass": N, "pos": P, "j": J, "ids": [eight expert ids]}). Each file is one run (one prompt). For each run the lines
are split in half by line count: the top N experts by read count over the first half are the resident set, and the
output is the fraction of the second half's expert reads they would serve, for N = 16, 32 and 64. The last line says
whether 32 slots serve at least 60% on every run.

Run from the repository root: python -I tools/py/mtp_route_hits.py write.jsonl edit.jsonl explain.jsonl
"""

import json
import sys
from collections import Counter

SLOTS = (16, 32, 64)
CHECK_SLOTS = 32
THRESHOLD = 0.60


def read_rows(path):
    """The expert-id lists of a route log, in file order."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ids = json.loads(line)["ids"]
            except (ValueError, KeyError, TypeError) as err:
                raise SystemExit(f"{path}:{n}: not a route line ({err})")
            rows.append([int(i) for i in ids])
    return rows


def hit_rates(rows, slots=SLOTS):
    """{N: fraction of the second half's reads served by the first half's top N}, or None for a run too short to split."""
    half = len(rows) // 2
    if half == 0:
        return None
    counts = Counter(i for row in rows[:half] for i in row)
    scored = [i for row in rows[half:] for i in row]
    out = {}
    for n in slots:
        # Ties in count break to the lower expert id, so the answer does not depend on file order.
        top = {i for i, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]}
        out[n] = sum(1 for i in scored if i in top) / len(scored)
    return out


def verdict(results):
    """(passes, line): whether 32 slots serve at least 60% on every run."""
    ok = bool(results) and all(r is not None and r[CHECK_SLOTS] >= THRESHOLD for r in results.values())
    word = "yes" if ok else "no"
    return ok, f"{CHECK_SLOTS} slots serve at least {THRESHOLD:.0%} on every run: {word}"


def main(argv):
    if not argv:
        raise SystemExit("usage: mtp_route_hits.py TRACE [TRACE ...]")
    results = {}
    for path in argv:
        rows = read_rows(path)
        rates = hit_rates(rows)
        results[path] = rates
        if rates is None:
            print(f"{path}: {len(rows)} draft rows, too few to split")
            continue
        cells = "  ".join(f"top{n} {rates[n]:.1%}" for n in SLOTS)
        print(f"{path}: {len(rows)} draft rows ({len(rows) // 2} scored)  {cells}")
    print(verdict(results)[1])


if __name__ == "__main__":
    main(sys.argv[1:])
