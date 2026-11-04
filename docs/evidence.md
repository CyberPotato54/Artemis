# Evidence

This page restates the figures the [README](../README.md) cites, with the date each was measured on. They come from the
gate records of the milestone summaries (milestones 1 to 6). The raw records and run logs are kept outside this
repository. Nothing here is rounded further than the source gave it.

## Hardware and method

All runs are on one machine: AMD Radeon RX 9070 XT (16 GB), AMD Ryzen 9 7950X3D, 64 GB DDR5-6000, PCIe 5.0 x16,
Windows 11 Pro. The model file is Unsloth's `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.

The speed figures use three prompts from `testdata/speed/prompts.json`: write (44 tokens), edit (2,206 tokens) and
explain (46 tokens). Seeds are 1234 and 1235, sampled at temperature 0.6, top-p 0.95 and top-k 20, for 700 tokens with
thinking off. Since milestone 4 each speed run is a fresh `artemis serve` process started from the production config
with the default cache. A prompt's figure is the median of its two seeds. "Median of six" is the median of all six runs.

## Decode and prefill

Measured on 2026-11-04, in the gate of the final 0.1.1 binary (with the MTP expert slots and the faster decode
kernels). A browser on the desktop held about 1.2 GiB of VRAM, so the desktop held 1,941 MiB during the runs, and the
cache had 125 or 126 of 256 expert slots a layer with speculation on and 128 or 129 with it off: the MTP's 64 expert
slots (116 MiB) come out of the cache's budget.

| Prompt | Speculation off | Prompt lookup only | Lookup and MTP (default) |
|---|---|---|---|
| write | 116.07 tok/s | 118.66 | 167.83 |
| edit | 109.62 | 179.96 | 177.32 |
| explain | 118.73 | 118.56 | 145.66 |
| **median of six** | **116.07** | **119.33** | **167.83** |

With speculation on, over off: write 1.4459, edit 1.6176, explain 1.2268. The gate's bar for each prompt is 0.98 of off.
The default runs MTP and lookup together; MTP pays when the median of six is at least 1.02 times the lookup-only
median, and it was 1.4065. Each pair of runs gave the same completion with speculation on and off. These are the
block's second attempt. The first, while the desktop was in use, measured 167.57 tok/s on and 115.88 off but missed the
prefill bar: with speculation on, the slower long prefill was 0.9719 of off and the slower edit prefill 0.9647, each
from one slow run.

The release run of 2026-11-03, on the build before the decode kernel change, with 544 MiB of desktop VRAM and 132 and
134 slots a layer, measured 161.19 tok/s on, 111.82 off and 115.84 with lookup only. The 0.1.1 gate, earlier that day
on the build before the MTP expert slots, measured 156.44 tok/s on, 113.47 off and 115.16 with lookup only.

Prefill, the slower of two cold runs each:

| Prompt | Speculation off | Speculation on | On over off |
|---|---|---|---|
| 27,416 tokens (golden prompt 20) | 2,129.67 tok/s | 2,132.37 | 1.0013 |
| 2,206 tokens (edit) | 1,710.78 | 1,699.62 | 0.9935 |

0.1.0's figures, from the speed round of 2026-10-31 (173 MiB of desktop VRAM, 146 slots a layer): median of six
110.01 tok/s with speculation off, 111.15 with lookup only and 137.75 with lookup and MTP; prefill 2,125.03 and 1,779.71
tok/s off. The gate also kept the bar of 0.97 times those figures: 167.83 against 133.6 on, 116.07 against 106.7 off
and 2,129.67 against 2,061 for the slower long prefill.

0.1.1's decode gain comes from six changes, each measured against the build before it in the same session and each
giving the same output bits:

- The routed-expert kernels of plain decode read at higher bandwidth: in a microbenchmark (best of 1,000) the
  gate/up kernel went from 273.1 to 333.7 GB/s and the down kernel from 229.6 to 327.7 GB/s on Q5_K rows and from
  277.5 to 334.0 on Q6_K rows. Plain decode rose about 5%.
- The same layout for the routed-expert kernels of the verify pass, specialized for 2 to 5 tokens: a Q5_K layer at 2
  tokens went from 143.4 to 81.4 µs and at 5 tokens from 327.9 to 186.4 µs. Speculative decode rose about 13%.
- The verify pass's attention reads its KV cache in larger tiles: at 4 tokens, 545.8 to 388.2 µs at 65,536 positions
  and 288.1 to 182.7 µs at 27,000. It is neutral at the speed prompts' short contexts.
- The CPU work between the verify pass and the replay: the sampler's top-k pass skips runs of logits below the k-th
  best (186 to 19 µs a pick), and the check for non-finite logits scans in blocks (89 to 16 µs), which halves the host
  time before the replay (1,538 to 787 µs on the write prompt).
- The MTP layer's 64 most-routed experts stay in VRAM slots (116 MiB, taken from the cache's budget), so 75 to 81% of
  its expert reads on the speed prompts no longer cross PCIe. In five sets of `spec-check speed` against the build
  before it, speculative decode rose 5.0% to 8.1% (pooled 163.3 against 152.8 tok/s); plain decode loads no MTP and
  does not change.
- The other decode kernels: the dense matrix-vector products sum each row's 32 lanes with subgroup shuffles in place
  of shared memory, the DeltaNet state update keeps its state in registers, the verify pass's attention takes the
  grouped-query form at every context length, and the verify pass's dense products have pipelines specialized for 2 to
  5 tokens. In nine sets of `spec-check speed` interleaved with the release run's binary (2026-11-04), the median of
  the set medians rose 6.4% with speculation on (163.34 to 173.80 tok/s) and 6.2% off (108.83 to 115.58); every
  completion was the same.

The first four together were measured against 0.1.0's binary in one session on the night of 2026-11-02, in three sets
of the six speed runs on each binary: median of 18 runs 153.5 against 133.0 tok/s with speculation on (+15.4%) and
112.4 against 104.95 with it off (+7.1%), every completion equal.
Taking out the verify pass's expert kernels cost 9.4% and 10.2% of the "on" figure, taking out the host change 4.1%.

The bars are 110.0 tok/s for the median of six with speculation on and 0.98 of off for each prompt and each prefill
figure. The "off" run must keep at least 46.5 tok/s of decode and 1,000 tok/s of prefill.

## Against llama.cpp

The context column is the context window (`-c` in llama.cpp, `ctx` in Artemis), the memory reserved for the
conversation. The speed prompts are 44 to 2,206 tokens long, so every figure here decodes at a shallow depth. Decode at
depth was measured separately, with 128 forced tokens and no speculation, so it is not comparable with the table: 80.0
tok/s at 4,096 positions, 60.71 (an earlier run) and 77.4 (the first-run gate's run) at 27,000, and 58.06 (the earlier
run) and 58.3 (the gate's run) at 65,000, all on 2026-11-01 (see [decode attention at long
context](#decode-attention-at-long-context)). llama.cpp was not measured at depth. llama.cpp's own MTP drafting
(`--spec-type draft-mtp`, 1 to 3 draft tokens) was measured on the same two configurations and was slower than its plain
decode at both contexts, so the ratios below use llama.cpp without speculation. With `--spec-draft-p-min 0` (every step
drafts its full length), the 32,768-position configuration measured 47.14, 59.76 and 61.41 tok/s at 1, 2 and 3 draft
tokens, and the 65,536-position configuration 51.02, 50.26 and 56.36 (2026-11-02; two runs whose PCIe link fell to
generation 3 are recorded as failures and were rerun after the link was retrained).

| Engine | Context window | Decode, median of six | Date |
|---|---|---|---|
| Artemis, speculation on | 65,536 positions | 167.83 tok/s | 2026-11-04 |
| Artemis, speculation off | 65,536 positions | 116.07 tok/s | 2026-11-04 |
| Artemis 0.1.0, tuned file from `artemis setup` | 65,536 positions | 135.67 tok/s | 2026-11-01 |
| Artemis 0.1.0, speculation on | 65,536 positions | 137.75 tok/s | 2026-10-31 |
| Artemis 0.1.0, speculation off | 65,536 positions | 110.01 tok/s | 2026-10-31 |
| llama.cpp b11509, Vulkan, no speculation, searched for 65,536 positions | 65,536 positions | 57.90 tok/s | 2026-11-02 |
| llama.cpp b11509, Vulkan, no speculation, searched for 32,768 positions | 32,768 positions | 65.15 tok/s | 2026-11-02 |
| llama.cpp b11509, Vulkan, no speculation | 32,768 positions | 35.7 tok/s | 2026-10-10 |
| llama.cpp b11509, Vulkan, its own MTP drafting, best of 1 to 3 draft tokens | 65,536 positions | 56.36 tok/s | 2026-11-02 |
| llama.cpp b11509, Vulkan, its own MTP drafting, best of 1 to 3 draft tokens | 32,768 positions | 61.41 tok/s | 2026-11-02 |

The searched rows: `llama-search` tries llama.cpp's flags on the RX 9070 XT (found by adapter name, not by index) one
axis at a time, keeping each axis's winner, and `speed-llama` and `speed-llama-65k` run a configuration on the same
prompts and seeds as Artemis's gate. Two searches on a quiet PC picked different configurations, and each is fastest at
one context:

- 32,768 positions: `--device Vulkan0 -ngl 99 --fit off -c 32768 --parallel 1 --jinja -fa on -ctk q8_0 -ctv q8_0
  --load-mode none -b 2048 -ub 512 --n-cpu-moe 20 --moe-cache-mib 3000 -t 8`. Six runs 63.50, 64.07, 70.36, 66.24,
  64.06 and 69.19 tok/s, median 65.15. With `-c 65536` it measures 44.88 tok/s.
- 65,536 positions: `--device Vulkan0 -ngl 99 --fit off -c 65536 --parallel 1 --jinja -fa on -ctk f16 -ctv f16
  --load-mode none -b 2048 -ub 2048 --n-cpu-moe 20 --moe-cache-mib 1500 -t 16`. Six runs 57.46, 54.65, 61.10, 58.34,
  54.13 and 60.49 tok/s, median 57.90. With `-c 32768` it measures 58.03 tok/s.

All four runs were on 2026-11-02, with the CPU mask 0xffff, 561.1 MiB of desktop VRAM and a PCIe link
check that read generation 4 before and after each block. The 32,768-position configuration was first found the night
before, when it measured 65.6 tok/s at 32,768 positions and 44.6 at 65,536. Searches run while Windows Defender and the
Windows Search indexer were busy on the CPU ranked configurations wrongly and measured one configuration at less than
half its quiet speed, so llama.cpp runs need those quiet too (see AGENTS.md).

With both engines decoding without speculation, Artemis 0.1.1 (116.07 tok/s) runs at 2.00 times llama.cpp's
65,536-position figure (116.07 / 57.90) and 1.78 times its 32,768-position figure (116.07 / 65.15); with speculation on,
2.90 (167.83 / 57.90) and 2.58 (167.83 / 65.15). The llama.cpp rows are from 2026-11-02 and Artemis 0.1.1's from
2026-11-04, on the same PC with the same prompts and seeds. The release run of 2026-11-03, before the decode kernel
change, gave 1.93 and 1.72 without speculation and 2.78 and 2.47 with it. For 0.1.0 the ratios were 1.90 and 1.69
without speculation, 2.34 and 2.08 with the tuned file (135.67) and 2.38 and 2.11 with speculation on (137.75).

The 2026-10-10 row is milestone 2's baseline, run through `llama-server` built from source with `--device Vulkan0 -ngl
99 --fit off -c 32768 --parallel 1 --jinja -fa on -ctk q8_0 -ctv q8_0 --load-mode none -b 2048 -ub 2048 --n-cpu-moe 24
--moe-cache-mib 1500`. On that date `Vulkan0` was the RX 9070 XT. The six runs were 34.7, 32.2, 36.7, 36.6, 32.0 and
38.9 tok/s (write, edit and explain at seed 1234, then at seed 1235); the median is 35.68. On that date Artemis's decode
median was 49.5 tok/s. The same flags measured 51.01 tok/s on a quiet PC on 2026-11-01. The summary that recorded the
baseline notes that an earlier measurement of the same configuration saw 51.3, 47.0 and 53.5 tok/s on the three prompts.
Milestone 5's bar, 110 tok/s, was 2.05 times an earlier llama.cpp measurement of 53.5 tok/s (109.7, rounded up to
110.0).

## Accuracy

Measured on 2026-10-31, from the gate's records unless a row says otherwise. Bounds are the ones in
[architecture.md](architecture.md#correctness-and-testing).

| Test | Result |
|---|---|
| 1. Kernels | Pass in the gate's final run of the block, each kernel within its stated tolerance. The block's first run failed one assertion in a real-model cache test (it asserted at least 6 lent cache rows and got 5, because less desktop VRAM was in use and the cache fitted more slots of its own); the assertion was changed and the block passed on the rerun |
| 2. Reference | Four real inputs, all pass. Worst tap outside attention 2.3e-6 (03-chat; bound 1e-5); worst `attn.pregate` 6.9e-5 (01-code; bound 1e-3). No routing mismatch and no excused tie |
| 3. Greedy golden set | 29 flips in 4,477 counted steps over 20 prompts; llama.cpp's own CPU backend flips 32 times; bound 48 |
| 4. Distribution | Mean KL divergence from llama.cpp's CPU backend 0.011360 ± 0.000484 over 51,175 positions (25 chunks of 4,096 tokens), same top token 98.744%; llama.cpp's Vulkan floor 0.01146; bound 0.01719 (1.5 times the floor); 164 cache slots |
| 5. Cache invariance | Pass on three prompts (05-code-write-lru, 12-prose-story, 16-tool-weather): tokens and logits bit-identical at 0, 26, 102 and 163 (maximum) slots and at 102 slots with a forced swap every step |
| 6. Prompt reuse | 29 flips pooled on each of the two warm paths, the same as the cold run's 29; bound 48; a restore reproduces the cold run's logits bit for bit |
| 7. Server | Every gated case passes on the production config: load, requests, budget, reuse, disconnect, chat-tools, idle, unload-stream and queue. Both client smokes pass (see below) |
| 8. Tokenizer | Pass: token ids equal `llama-tokenize`'s on the 1.2 MB corpus (308,431 and 308,714 tokens, with and without special parsing; 2026-10-08) |
| Speculation | `spec-golden`: 20 of 20 golden prompts identical in tokens and logits with speculation on and off, 256 tokens each, 0 divergences outside a near tie |

Tests 1, 3, 4, 5 and 6, and test 2's GPU half (the engine against the reference), also passed in the first-run gate
on 2026-11-01, and again in the 0.1.1 gate on 2026-11-03 with the same figures: test 3 at 29 flips in 4,477 steps with
every prompt's flips at the same steps, test 4 at 0.011360, test 5 bit-identical, test 6 at 29 flips on both warm
paths, and `spec-golden` 20 of 20 identical. Test 1's identity lines (whether each verify kernel's bits equal the
single-token kernel's) showed 0 differing cases in every kernel. The gate of the final 0.1.1 binary (2026-11-04, with
the faster decode kernels) gave the same figures again: tests 1 (54 kernels, 1,187 cases, 0 differing identity cases),
2's GPU half, 3, 4, 5, 6, prefill, `spec-golden` and the 32 GB memory check.

MTP logits (2026-10-31): 83 of 84 top-1 equal, with one flip at a margin of 0.056 against the 0.05 tie bound, so
the check records a fail. It was accepted because a wrong draft costs only speed and `spec-golden` is the output gate.

Prefill against decode (2026-10-31): decode forced onto prefill's routing differs by a mean of 7.13e-5 in logits
(bound 1e-3) with a worst position at 8.78e-3 (bound 5e-2) and a final state difference of 6.76e-4 (bound 1e-3). Chunks
of 2,048 and 500 tokens give the same bits (0 positions differ). The 27,416-token prompt prefilled in 12.63 s
(2,171 tok/s in that test) with a longest submit of 38.41 ms against the bound of 300 ms.

Client smokes (2026-10-31): OpenCode 2.0.30 passed in 31.8 s with 7 requests and 6 tool calls; Claude Code 2.1.316
passed in 29.8 s with 7 requests and 9 tool calls. Both had 0 malformed calls and 0 5xx responses. The reuse report over
both servers: 14 requests, 90,530 prompt tokens, 76,466 reused (0.845).

Client smokes (2026-11-03, the 0.1.1 gate): OpenCode 2.0.30 passed in 29.5 s with 6 requests and 6 tool calls; Claude
Code 2.1.317 passed in 29.3 s with 7 requests and 9 tool calls. Both had 0 malformed calls and 0 5xx responses. The
reuse report over both servers: 13 requests, 86,403 prompt tokens, 72,487 reused (0.839). The production checks of
test 7 passed on the same binary.

Client smokes (2026-11-03, the release run of the 0.1.1 binary with the MTP expert slots): OpenCode 2.0.30 passed in
31.5 s with 7 requests and 6 tool calls; Claude Code 2.1.317 passed in 31.1 s with 6 requests and 9 tool calls, each on
its first attempt. Both had 0 malformed calls and 0 5xx responses. The reuse report over both servers: 13 requests,
79,539 prompt tokens, 65,415 reused (0.822). The production checks of test 7 passed on the same binary. Tests 1 and 5,
`spec-golden` (20 of 20 identical) and the 32 GB memory check passed on it too.

Client smokes (2026-11-04, the gate of the final 0.1.1 binary): OpenCode 2.0.30 passed in 32.2 s with 6 requests and 6
tool calls; Claude Code 2.1.317 passed in 31.9 s with 7 requests and 9 tool calls, each on its first attempt. Both had
0 malformed calls and 0 5xx responses. The reuse report over both servers: 13 requests, 85,952 prompt tokens, 72,193
reused (0.840). The production checks of test 7 passed on the same binary on the block's second attempt. The first
failed one check of the idle unload: the reload got 124 expert slots a layer against 126 at the first load (138 MiB
less VRAM), because another application took VRAM on the desktop between the two loads.

## Smaller caches

Measured on 2026-11-04 in the gate of the final 0.1.1 binary, report only (the 16 GB card with its cache capped to the
estimate for a 12 GB and an 8 GB card; the MTP's 64 expert slots fit beside both caps):

| Prompt | Slots a layer | Speculation off | Speculation on | On over off |
|---|---|---|---|---|
| write | 85 | 91.44 tok/s | 124.34 | 1.3599 |
| edit | 85 | 86.70 | 129.06 | 1.4885 |
| explain | 85 | 94.54 | 99.53 | 1.0528 |
| write | 30 | 63.35 | 79.35 | 1.2527 |
| edit | 30 | 56.74 | 75.20 | 1.3254 |
| explain | 30 | 63.84 | 65.64 | 1.0281 |

The explain prompt has a low MTP accept rate (0.479 at 85 slots, 0.571 at 30), and the gate treats this block as a
report. With the MTP expert slots it runs above plain decode at both sizes: 1.0253 and 1.0084 in the release run of
2026-11-03. Before them it ran below the 0.98 bar: 0.9477 at 85 slots and 0.9757 at 30 in the 0.1.1 gate of
2026-11-03, and 0.9452 and 0.9821 on 2026-10-31 (0.1.0).

## Setup and the first run

The first-run gate (plan 6b), 2026-11-01. The runner pinned its CPU mask to 0xffff0000 and every block checked the
PCIe link (generation 4 before and after each block, none dropped). The desktop held 568.6 MiB of VRAM. The block
`first-run` starts `artemis serve` with no tuned file, so setup runs before the port opens, then makes the six decode
runs of `serve-speed-spec` in fresh processes that read the file setup wrote (write, edit and explain at seeds 1234 and
1235, thinking off, 700 tokens, a context of 65,536 positions).

| Figure | Value | Bar |
|---|---|---|
| Setup seconds | 80.4 | at most 600 |
| Setup limit hit | no | no |
| First launch to `/health` | 83.2 s | none |
| Median of six with the file setup wrote | 135.67 tok/s | above llama.cpp's same-session median (reported separately) |
| `serve-speed-spec` "on" median in the same invocation (hand-tuned production config) | 136.48 tok/s | none |
| First-run over hand-tuned | 0.9941 | at least 0.97 |
| WinHTTP or WinINet imports in `artemis.exe` | none | none |

Setup chose speculation on with prompt lookup and MTP, which matches the defaults. All 6 tuned completions equal the
"on" run's of the same prompt and seed (reported, not gated). Milestone 5's speed round holds in the same invocation:
the "on" median is 136.48 against a bar of 133.6 (0.97 of 137.75), the "off" median 109.20 against 106.7, and the
slowest "off" long prefill 2,134.27 tok/s against 2,061.

Three further runs of `artemis setup --force --json` on an earlier build of the same day (2026-11-01, the model file in
the OS cache, a PCIe generation 4 link measured at 28.5 GB/s) took 85.8, 80.4 and 80.7 s, each choosing prompt lookup
and MTP. Speculation beat plain decode by 1.19 to 1.20 times and MTP beat lookup alone by 1.10 to 1.11 times in the
calibration. The measured-priors candidate scored 117.1 to 118.1 tok/s, below 1.03 times the base candidate (118.7 to
119.7), so the built-in priors stayed. The "about 119 tok/s" setup prints is the pooled figure over its three
calibration prompts at 256 tokens; it is not comparable with the speed gate's 130 to 140 on the write prompt. A cold
first start also reads the model file from disk for each of its two loads; no cold first start was measured. The limit
of 600 s is a ceiling and was kept.

## Decode attention at long context

Plan 6b made decode attention read its KV cache with more parallelism and kept every output bit. One layer's call on
the RX 9070 XT, GPU clock, best of 1,000 runs (`attn_decode_speed`; before and after the change, both 2026-11-01):

| Positions | `attn_decode_gqa` plus combine, before | After | After over before |
|---|---|---|---|
| 4,096 | 60.80 µs | 38.40 | 0.63 |
| 27,000 | 231.44 | 139.08 | 0.60 |
| 65,536 | 728.00 | 307.64 | 0.42 |

The `attn-depth` block decodes 128 forced tokens at three depths on a pinned binary built before the change and on the
current build, and the logits fingerprints are equal at every depth. Tokens per second, each the mean of two timed runs:

| Positions | Run | Pinned binary | Current build | Current over pinned |
|---|---|---|---|---|
| 4,096 | earlier | 78.53 | 79.70 | 1.0149 |
| 27,000 | earlier | 57.40 | 60.71 | 1.0577 |
| 65,000 | earlier | 50.04 | 58.06 | 1.1602 |
| 4,096 | gate | 78.6 | 80.0 | 1.0175 |
| 27,000 | gate | 72.7 | 77.4 | 1.0638 |
| 65,000 | gate | 41.3 | 58.3 | 1.4132 |

Both runs are 2026-11-01. At 65,000 positions the current build measured 58.1 and 58.3 tok/s in the two runs and the
pinned binary 50.0 and 41.3, so the ratio moved from 1.16 to 1.41 with the pinned binary's rate; the lower ratio, 1.16,
is the one to quote. The bars are 0.98 of the pinned rate at 4,096 and 27,000 positions and at least equal at 65,000.

The 0.1.1 gate (2026-11-03) ran the block against 0.1.0's binary: 47.7 against 49.8 tok/s at 4,096 positions (1.0447),
76.0 against 78.3 at 27,000 (1.0294) and 57.2 against 58.0 at 65,000 (1.0141), with equal fingerprints at every depth.
Every timed run at 4,096 positions, on both binaries, held a decode stall of about 1 s (127 tokens in 2.3 to 2.8 s
against about 1.6 s on 2026-11-01), so that depth's rates are low on both sides. The block decodes without
speculation; 0.1.1's attention change is in the verify pass (see [decode and prefill](#decode-and-prefill)).

The gate of the final 0.1.1 binary (2026-11-04) ran the block against the release run's binary of 2026-11-03: 81.7
against 79.4 tok/s at 4,096 positions (1.0283), 78.5 against 76.5 at 27,000 (1.0261) and 57.3 against 46.7 at 65,000
(1.2264), with equal fingerprints at every depth. Earlier builds measured 57.2 to 58.3 tok/s at 65,000 in the runs
above, so the pinned binary's 46.7 here is a slow run and 1.2264 overstates the change at that depth.

## The adaptive cache

Plan 2d's gate, which closed milestone 2 (2026-10-13). The hit rate is the cache's share of every routed expert read,
prefill included. The adaptive policy against `--cache-policy static`, on the golden prompts of each group:

| Group | Adaptive | Static | Difference |
|---|---|---|---|
| code | 82.24% | 76.25% | +5.99 |
| prose | 66.88% | 49.34% | +17.55 |
| tool use | 64.31% | 61.56% | +2.74 |

The rule was that code prompts average at most 1 point below static and the other two groups above it. Plan 2c's gate
(2026-10-12) measured the same figures to the hundredth.

The CPU miss path (plan 2e, 2026-10-18): at its one non-zero setting, a share of 1, the decode median of
the write, edit and explain prompts was 79.714 tok/s against 91.801 with the path off, 13.2% lower. The path needed a
gain of 2% to be turned on, so it is off by default and stays in the tree.

## The PCIe link

On 2026-10-31, during the speed round, every binary ran about 22% slow (medians of 96 to 98 tok/s) after the GPU's PCIe
link had dropped to half rate. `artemis snapshot-times` measured 14.26 GB/s to VRAM and 13.95 GB/s to the host, against
28.39 and 27.58 on 2026-10-19. The same state was seen on 2026-10-13 (14.22 GB/s). Restarting the GPU device from an
elevated shell retrained the link (28.51 GB/s), and the pinned gate binary went from 100.0 to 131.4 tok/s. The cause is
not known. `tools/py/gpu_link.py` reads the live link generation (3 while slow, 4 after the retrain). The figures on
this page were measured at full rate.

## Gates by milestone

Each milestone was built in plans named by letter: 2a to 2e, 3a to 3c, 5a and 5b, 6a, 6q and 6b. The table gives each
plan's gate date.

| Milestone | Gate date | Result |
|---|---|---|
| 1. Correct engine | 2026-10-09 (plan 1c-ii) | Tests 1, 2, 3, 4, 5 and 8 pass. Test 3: 29 flips in 4,477 counted steps (floor 32, bound 48). Test 4: KL 0.011157 ± 0.000471 over 51,175 positions (floor 0.011460, bound 0.017190). Test 5: logits bit-identical at every cache size on three prompts. The 27,416-token prompt prefilled in 18 chunks in 57.2 s (480 tok/s) with a longest submit of 139 ms (bound 300 ms) |
| 2. Fast engine | 2026-10-13 (plan 2d); closing run 2026-10-18 (cooperative-matrix prefill attention) | On 2026-10-13: decode median 78.68 tok/s (bound 46.5), slower prefill run 1,522.1 tok/s (bound 1,000). On 2026-10-18: decode median 95.57 tok/s, prefill 2,149.2 tok/s, test 3 at 29 flips, test 4 at 0.011360, prefill against decode with a mean of 7.13e-5 and a worst of 8.78e-3 |
| 3. Server | 2026-10-19 (plan 3a); 2026-10-26 (plan 3c) | 3a: decode median 95.69 tok/s, slower prefill run 2,147.9 tok/s, test 6 (restore from a snapshot) at 29 flips. 3c: test 7 on the real `artemis serve` passes first try; decode through the server 89.06 tok/s (median of six); the edit prompt (2,206 tokens) prefills at 1,774.4 tok/s through the server; OpenCode 2.0.30 and Claude Code 2.1.310 smokes pass |
| 4. Production readiness | 2026-10-28 (plan 4a) | Stall test: the server exits with code 4, 124.98 s after the last token (limit 130 s). Decode median 98.45 tok/s, slower prefill 2,124 tok/s. The production checks pass, including `chat-tools` 10 of 10. OpenCode (35.9 s) and Claude Code (33.8 s) smokes pass on the first attempt with 0 malformed calls |
| 5. Speculative decoding | 2026-10-29 (plan 5a); 2026-10-30 and 31 (plan 5b); 2026-10-31 (speed round) | 5a (prompt lookup): median of six 98.14 tok/s against the bar of 110.0, a miss that called for plan 5b; off 96.96 tok/s. 5b (adds the MTP drafter): 124.27 tok/s, lookup alone 95.89, off 95.15. Speed round: 137.75 tok/s, lookup alone 111.15, off 110.01. `spec-golden` is 20 of 20 identical at all three |
| 6a. Self-contained (milestone 6, in progress) | 2026-10-31 | 64 shaders byte-identical to the Vulkan SDK 1.4.363.0 compiler's output; a source archive builds offline with no shader compiler; median of six with speculation on 137.31 tok/s on attempt 2, with write, edit and explain at 1.252, 1.385 and 1.000 of off and prefill at 0.998 and 0.998 of off; `spec-golden` 20 of 20 identical; both client smokes pass |
| 6q. QA and security fixes (milestone 6) | 2026-11-01 | Median of six with speculation on 133.49 tok/s on attempt 1 and 132.55 on attempt 2 (attempt 2 counts), against the bar of 110.0 and the plan's own target of 135.0 (2% under the milestone 5 figure), which both attempts missed; off 105.83 and 106.31, lookup alone 107.12 and 107.18; prefill at 0.9997 and 0.9998 of off on attempt 2; `spec-golden` 20 of 20 identical; both client smokes pass. A same-session A/B of this binary against the 6a gate's binary (four alternating rounds, 300 tokens, greedy) gave off 94.59 against 93.86 tok/s and on 124.11 against 123.78, so the code had not slowed. The gate was accepted on that evidence |
| 6b. First-run setup (milestone 6), speed blocks | 2026-11-01 | `artemis setup` took 80.4 s on a first start; the six decode runs with the file it wrote have a median of 135.67 tok/s at 65,536 positions, 0.9941 of the hand-tuned "on" median of 136.48 in the same invocation (bar 0.97); no WinHTTP or WinINet import; milestone 5's speed round holds ("on" 136.48 against 133.6, "off" 109.20 against 106.7, slowest "off" long prefill 2,134.27 against 2,061). Decode attention at 65,000 positions: 1.16 times the pinned binary's rate in an earlier run that day (the figure to quote; 1.4132 in the gate's run, where the pinned binary was slower), with equal logits fingerprints at 4,096, 27,000 and 65,000 positions |
| Release check (milestone 6) | 2026-11-02 | The production checks of test 7 on the release commit (load, requests, budget, reuse, disconnect, `chat-tools`, idle unload, unload while streaming, the queue) pass, with the speed case reported; the test messages pass; the OpenCode 2.0.30 and Claude Code 2.1.316 smokes pass. The binary is linked reproducibly (`/Brepro`), so a build of the release commit gives the same file; its SHA-256 begins ab83bb05. An earlier build of the same engine code gave `spec-golden` 20 of 20 identical and a median of six with speculation on of 136.32 tok/s |
| 0.1.1 speed track | 2026-11-03 | Every correctness block passes with the 0.1.0 gate's figures (tests 1, 2's GPU half, 3, 4, 5, 6, prefill, `spec-golden` 20 of 20, the 32 GB memory check). Median of six with speculation on 156.44 tok/s, off 113.47, lookup alone 115.16, slowest long prefill 2,114.15 tok/s, all above 0.97 of the 0.1.0 speed round. Decode at depth against 0.1.0's binary with equal fingerprints. The production checks and the OpenCode 2.0.30 and Claude Code 2.1.317 smokes pass |
| 0.1.1 release check | 2026-11-03 | On the 0.1.1 binary with the MTP expert slots (its SHA-256 begins 321a7254): tests 1 and 5, `spec-golden` 20 of 20 identical and the 32 GB memory check pass; every completion of `spec-check speed` equals the build's before the slots; median of six with speculation on 161.19 tok/s, off 111.82, lookup alone 115.84, slowest long prefill 2,117.69 tok/s; the production checks and the OpenCode 2.0.30 and Claude Code 2.1.317 smokes pass |
| 0.1.1 final gate | 2026-11-04 | On the final 0.1.1 binary, with the faster decode kernels (its SHA-256 begins cbd970cf): every correctness block passes with the earlier gates' figures (tests 1, 2's GPU half, 3, 4, 5, 6, prefill, `spec-golden` 20 of 20, the 32 GB memory check); a source archive builds offline. Median of six with speculation on 167.83 tok/s, off 116.07, lookup alone 119.33, slowest long prefill 2,129.67 tok/s, on the block's second attempt (the first missed the 0.98 prefill bar with speculation on); decode at depth against the release run's binary with equal fingerprints. The production checks pass on the block's second attempt (the first failed the reload's VRAM check when a desktop application took VRAM between the loads); the OpenCode 2.0.30 and Claude Code 2.1.317 smokes pass |
