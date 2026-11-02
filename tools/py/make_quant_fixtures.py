"""Write one real row of each weight type in the model, with gguf-py's dequantization, as test fixtures.

Usage: uv run --project tools/py --extra fixtures python -I tools/py/make_quant_fixtures.py <model.gguf> <out_dir>
"""
import pathlib
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

ROWS = {
    "blk.0.ssm_alpha.weight": "f32_row",
    "blk.40.ffn_gate_inp.weight": "bf16_row",
    "blk.0.attn_qkv.weight": "q8_0_row",
    "blk.0.ffn_gate_exps.weight": "q4_k_row",
    "blk.0.ffn_down_exps.weight": "q5_k_row",
    "blk.34.ffn_down_exps.weight": "q6_k_row",
}

model, out = sys.argv[1], pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
found = set()
for t in GGUFReader(model).tensors:
    if t.name not in ROWS:
        continue
    data = np.asarray(t.data)
    row = np.ascontiguousarray(data.reshape(-1, data.shape[-1])[0])
    raw = row.view(np.uint8)
    if t.tensor_type.name == "F32":
        vals = row.astype("<f4")
    else:
        vals = dequantize(raw.reshape(1, -1), t.tensor_type).reshape(-1).astype("<f4")
    (out / f"{ROWS[t.name]}.bin").write_bytes(raw.tobytes())
    (out / f"{ROWS[t.name]}.f32").write_bytes(vals.tobytes())
    print(t.name, t.tensor_type.name, raw.nbytes, "bytes ->", vals.size, "values")
    found.add(t.name)
missing = set(ROWS) - found
if missing:
    sys.exit(f"not in the model: {sorted(missing)}")
