# Engine data

## `qwen36-expert-profile.bin`

The routing profile that fills the expert cache at load (see docs/architecture.md): how many times each of
Qwen3.6-35B-A3B's experts was routed, per layer, in the 2026-10-08 expert-cache spike. u64 counts, row-major
[layer][expert] for 40 layers of 256 experts, little-endian: 81,920 bytes.

| | |
|---|---|
| Source | `expert-counts.npz` from the 2026-10-08 Qwen3.6 expert-cache engine spike |
| Source sha256 | `893dbdc4bf394857860ee68eaff0d4739e6dc7cd1e24955d80e3b88dba3ccbe1` |
| This file's sha256 | `36cc6fea47d00f6ad7a156a69341514079049ce6c3c9af40dcf3c432e39c99c5` |
| Data | 16 sessions of code, prose and multilingual chat, 364,544 tokens; every layer sums to 2,916,352 (8 per token) |
| License | MIT OR Apache-2.0: the owner's own measurement |

Made by summing the 16 per-session arrays:

    uv run --project tools/py --extra fixtures python -I tools/py/make_expert_profile.py <expert-counts.npz> crates/engine/data/qwen36-expert-profile.bin

`engine::Profile` ranks each layer's experts by these counts, ties to the lower id. Its tests pin the file's
FNV-1a 64 (`0xeb42_e627_90eb_94da`), so a changed file fails CI.
