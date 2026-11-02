# The llama.cpp oracle

`artemis-oracle` links llama.cpp at tag `b11509` (commit `1167d3f4…9110`) and writes what Artemis's tests compare
against. llama.cpp is MIT-licensed and used as a library; none of its code is copied into Artemis.

Build (PowerShell): `oracle\build.ps1 -Root <folder>`. It clones llama.cpp into `<folder>\llama.cpp`, checks the tag's
commit, and builds `artemis-oracle`, `llama-tokenize`, `llama-perplexity` and `llama-server` into
`<folder>\build\bin\Release`. Give the same folder to `tools\run-gpu-tests.ps1` as `-Oracle` (or set `ARTEMIS_ORACLE`).

| Mode | Writes |
|---|---|
| `taps --model M --backend cpu\|vulkan [--experts-on-cpu] --tokens F --out DIR [--filter REGEX]` | Every named intermediate tensor matching the filter, as raw f32 (or i32), plus `index.json`. Get the filter from `artemis taps-filter` |
| `mtp-taps --model M --backend cpu --tokens F --out DIR [--filter REGEX]` | The MTP block `blk.40` over the main model's own hidden rows: `DIR/hidden.bin` (raw f32 `[n][n_embd]`, each position's final hidden row after `output_norm`), and the matching `mtp_*`, `h_nextn` and `result_output` tensors of an MTP context (plus `mtp_attn_gated-40`, the oracle-named `pregate * sigmoid(gate)` node llama.cpp leaves unnamed) run over rows `q = 1 .. n-1` (token `q` with the hidden row at `q-1`, at position `q`; llama.cpp's own driver also writes a position-0 row with an all-zero hidden row, which this mode and `dump --mtp` both leave out), plus `index.json` whose `n_tokens` counts those rows. Get the filter from `artemis taps-filter --mtp` |
| `greedy --model M --backend vulkan --experts-on-cpu --tokens F --n 256 --out FILE` | Greedy tokens with each step's top two logits, as JSON |
| `forced --model M --backend cpu\|vulkan [--experts-on-cpu] --tokens F --force F [--n 256] --out FILE` | `greedy` with the tokens given: the prompt, then `--force`'s tokens fed one at a time. Each step records the token fed, the top two logits and the fed token's logit, as JSON. Test 3's floor runs it on the CPU along the oracle's tokens |
| `tokenize --model M --file F --out FILE [--no-parse-special]` | Token ids, little-endian u32. The file is read as bytes, so CRLF survives |
| `vocab --model M --out FILE` | The end-of-generation set and the CONTROL and USER_DEFINED token ids, as JSON |

`--experts-on-cpu` keeps the 18.7 GiB of experts in host memory, as `-ot exps=CPU` does; the real model doesn't
fit in 16 GB of VRAM otherwise.
