"""Write one real Q4_K row and one real Q8_0 row from the model, with gguf-py's dequantization, as test fixtures.

Usage: python -I make_fixtures.py <model.gguf> <out_dir>
"""
import pathlib
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

model, out = sys.argv[1], pathlib.Path(sys.argv[2])
want = {"blk.0.ffn_gate_exps.weight": "q4k_row", "blk.0.attn_qkv.weight": "q8_0_row"}
for t in GGUFReader(model).tensors:
    if t.name not in want:
        continue
    data = np.asarray(t.data)
    raw = data.reshape(-1, data.shape[-1])[0]  # the first row, as bytes
    vals = dequantize(raw.reshape(1, -1), t.tensor_type).reshape(-1).astype("<f4")
    (out / f"{want[t.name]}.bin").write_bytes(raw.tobytes())
    (out / f"{want[t.name]}.f32").write_bytes(vals.tobytes())
    print(t.name, t.tensor_type.name, raw.nbytes, "bytes ->", vals.size, "values")
