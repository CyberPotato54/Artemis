# Test data

The test inputs and expected outputs. `manifest.toml` lists every file with how it was made, its license and its sha256;
regenerate it with `tools/py/write_manifest.py` after any change.

## Rules

- Public-domain, MIT, Apache and CC-BY text may be committed, CC-BY with attribution (see `tokenizer/sources.json`). Each
  upstream's copyright notice and licence text is in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
- PSF-2.0 text may be committed with the PSF License Agreement beside the source (`speed/sources/LICENSE-PSF`).
- Share-alike text (CC-BY-SA and the like) is never committed; scripts fetch it at a pinned revision.
- Text files here are byte-exact (`.gitattributes` marks them `-text`): their line endings are part of the test.

## Contents

| Path | What |
|---|---|
| `tokenizer/corpus.txt` | About 1 MB of prose, code, multilingual text and edge cases, for test 8 |
| `tokenizer/corpus.special.ids`, `corpus.plain.ids` | llama.cpp b11509's token ids for it, with and without special-token parsing (little-endian u32) |
| `tokenizer/qwen36-vocab.bin` | The model's tokens, types and merges, so CI can build the tokenizer without the 21 GB file |
| `tokenizer/eog.json` | llama.cpp's end-of-generation set and special-token ids |
| `test2/inputs/` | Test 2's four short inputs and their token ids |
| `test2/tiny/` | llama.cpp's taps on the tiny model, so CI runs test 2's first rung |
| `golden/prompts.json` | Test 3's 20 prompts: each one's category, its chat messages and tools or the file it comes from, and the sources it quotes |
| `golden/sources/` | Rust 1.90.0's `core/src/option.rs`, quoted whole by the long prompt (MIT OR Apache-2.0) |
| `golden/prompts/` | Each prompt as the model sees it (`<id>.txt`, byte-exact) and llama.cpp's ids for it (`<id>.tokens`) |
| `golden/expected/` | llama.cpp b11509's greedy run on each prompt: up to 256 steps, each step's token and top two logits |
| `golden/forced-cpu/` | Test 3's floor: llama.cpp b11509's CPU backend fed each prompt and then `expected/`'s tokens, one per step. Each step: the token fed, the run's own top two logits and the fed token's logit |
| `golden/made-from.json` | For each prompt, the FNV-1a 64 of the tokens (and of `expected/`'s file) its outputs were made from, so a stale output fails the golden-set test |
| `speed/prompts.json` | Milestone 2's speed prompts: the spike's write, edit and explain prompts, their sampling and the pinned source the edit prompt quotes |
| `speed/prompts/` | The write, edit and explain prompts as the model sees them (`<id>.txt`) and llama.cpp's ids (`<id>.tokens`) |
| `speed/sources/` | CPython v3.13.3's `Lib/json/encoder.py` (PSF-2.0), byte for byte, which the edit prompt quotes, and `LICENSE-PSF`, the PSF License Agreement from CPython's `LICENSE` at that tag |
| `speed/made-from.json` | Each speed prompt's token count and the FNV-1a 64 of its ids, the edit prompt's included, so `artemis speed` can check every run's prompt |
| `kl/corpus.txt` | Test 4's corpus: 107,136 tokens, code from four permissive sources, then public-domain prose |
| `kl/sources.json` | Where each part of it comes from, with sha256 and the characters used |
| `template/qwen36-chat-template.jinja` | The GGUF's own `tokenizer.chat_template`, byte for byte (8,057 bytes), so CI renders prompts without the 21 GB file; `crates/qwen36/tests/chat_template.rs` checks it against the GGUF |
| `template/requests/` | The 18 OpenAI chat bodies written for Artemis (`t01` to `t18`) that pin the template's rules (spec 6.5): reasoning dropped (`t01`, `preserve_thinking: false`), kept (`t02`, `preserve_thinking: true`) and kept by default (`t18`), tool loops, parallel calls, non-string arguments, `developer` and extra system messages, non-ASCII, empty `tools`, thinking off, content parts, extra tool keys, text before tool messages |
| `template/prompts/` | Each body's prompt as llama-server b11509's `/apply-template` renders it (`<id>.txt`, byte-exact) and llama.cpp's ids for it (`<id>.tokens`) |
| `template/anthropic/` | The 4 Anthropic bodies (`a01` to `a04`) written for Artemis and, in `<id>.count`, the `input_tokens` llama-server b11509's `/v1/messages/count_tokens` gives for each (one integer and a newline) |
| `clients/` | What OpenCode 2.0.30 and Claude Code 2.1.302 really send: one request per file (`NNN-<route>.json`: client, version, seq, method, path, query, headers, body), captured against a recording stub and scrubbed by `tools/py/scrub_captures.py`. System text and tool descriptions are placeholder `x`s; Claude Code's strings are all replaced by `x`s of the same length. `clients/opencode/LICENSE-OpenCode` is OpenCode's MIT copyright notice, copied verbatim from the upstream repository at the installed version's tag; OpenCode's captures are committed under that licence |

Expected ids come from `artemis-oracle tokenize`, which reads files as bytes. Stock `llama-tokenize` was checked
against it on the corpus (2026-10-08). With `--no-escape` the ids are identical, CRLF included. Without it they differ
from token 83,112 on, because `llama-tokenize` turns escape sequences such as a literal
`\n` in the code samples into real characters by default.

## The golden set

`tools/py/build_golden_set.py` makes it from `golden/prompts.json`, in four stages:

1. `render`: test 2's four inputs are copied byte for byte. The 16 chat prompts are rendered by llama-server b11509's
   `/apply-template`, which gives exactly the prompt llama-server would feed the model.
2. `tokenize`: `artemis-oracle tokenize`, special tokens parsed.
3. `expected`: `artemis-oracle greedy --backend vulkan --experts-on-cpu --n 256`, about 25 minutes for all 20.
4. `floor`: `artemis-oracle forced --backend cpu --n 256`, fed each prompt and then `expected/`'s tokens, about 20
   minutes for all 20. Rerun it whenever an expected output changes.

`--stage` runs one stage and `--only ID` one prompt. `crates/tools/tests/golden_set.rs` checks the set is whole, that
our tokenizer gives llama.cpp's ids for every prompt, and that the floor was fed the expected outputs' tokens.

## The speed prompts

`tools/py/build_speed_prompts.py` makes them from `speed/prompts.json` (stages `render`, `made-from` and `check`). The
edit prompt quotes CPython's `Lib/json/encoder.py` (PSF-2.0). The rule above allows it with its licence file beside it,
so the source is committed in `speed/sources/` (pinned by sha256 in `prompts.json`) and nothing is fetched: a missing
source is an error that names the file. The runner's `speed` block runs the `check` stage first: it remakes any prompt
whose token file is missing and fails unless every prompt's ids match `speed/made-from.json`.

## The template fixtures

`tools/py/build_template_fixtures.py` renders `template/requests/*.json` through `/apply-template` and
`template/anthropic/*.json` through `/v1/messages/count_tokens` on llama-server b11509 (CPU, port 8097, started and
stopped by the script), then tokenizes each prompt with `artemis-oracle`. What the oracle does that the template alone
doesn't say, found by these fixtures: llama-server enables `--reasoning-preserve` by default, so it renders every
echoed turn's reasoning (`t18` equals `t02`) unless the request says `preserve_thinking: false` (`t01`) or
`preserve_reasoning: false`; and it writes a float as C++ does, `%g` with six digits (`1.0` is `1`, `1e-07` stays,
`1234567.0` is `1.23457e+06`, in `string` and in `tojson` alike), where Python writes `1.0`. `t08` pins the floats on both
paths. The script starts the server with `--reasoning-preserve` spelled out.

## The KL corpus

`tools/py/build_kl_corpus.py` builds `kl/corpus.txt` from pinned sources: code first, then prose, LF line ends, no
trailing newline (llama-perplexity reads the file in text mode and drops one). Test 4 scores its first 25 chunks of
4,096 tokens. llama.cpp's KL base and Vulkan floor for it live under the oracle folder's `runs\kl`, made once by
`tools/run-gpu-tests.ps1 -Tests 4`; the base is 25,416,371,020 bytes, so it is never committed.

Next to them, `base-cpu.json` and `vulkan-floor.json` record the corpus sha256, the model (path, size, modification time)
and, for the floor, the GPU driver. The runner remakes the base, and then the floor, whenever one differs. For a base
and floor already on disk, `tools/run-gpu-tests.ps1 -Tests 4 -WriteKlSidecars` writes both once.
