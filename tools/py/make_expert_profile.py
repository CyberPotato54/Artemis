"""Build crates/engine/data/qwen36-expert-profile.bin from recorded routing counts.

Usage: uv run --project tools/py --extra fixtures python -I tools/py/make_expert_profile.py <expert-counts.npz> <out.bin>

The npz holds one int32 [40][256] array of routed-expert counts per recorded session (2026-10-08). This sums them, in
sorted key order, as u64 and writes the totals row-major, [layer][expert], little-endian: 81,920 bytes. It prints both
files' sha256 for the data README.
"""
import hashlib
import pathlib
import sys

import numpy as np

LAYERS, EXPERTS = 40, 256


def main() -> None:
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    z = np.load(src)
    total = np.zeros((LAYERS, EXPERTS), dtype=np.uint64)
    for key in sorted(z.files):
        a = z[key]
        if a.shape != (LAYERS, EXPERTS) or a.dtype.kind not in "iu" or (a < 0).any():
            sys.exit(f"{key}: expected non-negative integer counts of shape ({LAYERS}, {EXPERTS}), got {a.dtype} {a.shape}")
        total += a.astype(np.uint64)
    data = total.astype("<u8").tobytes()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    sums = sorted({int(s) for s in total.sum(axis=1)})
    print(f"{len(z.files)} sessions; every layer sums to {sums}")
    print(f"source {src.name}: sha256 {hashlib.sha256(src.read_bytes()).hexdigest()}")
    print(f"wrote {out} ({len(data):,} bytes): sha256 {hashlib.sha256(data).hexdigest()}")


main()
