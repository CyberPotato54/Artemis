# Artemis agent context

This is the maintainers' working guide to the repository: the rules and test procedures they follow, including their
release-gate vocabulary (plans, blocks, done-whens). Contributors need only the build and test steps in the
[README](README.md).

Read [docs/architecture.md](docs/architecture.md) (the model contract, correctness and testing, and the decisions and
alternatives) before changing anything.

- No code is copied or ported from other inference engines. llama.cpp is read for the model's semantics only; the
  kernels are our own.
- GPU code runs only on a Windows machine with a Vulkan GPU (development used an RX 9070 XT). GPU tests are
  `#[ignore]`d; run them there with `cargo test --release -- --ignored --test-threads=1`. Every GPU run is `--release`
  (the runner's too), and the workspace keeps the release profile's default `debug-assertions = false`, so a
  `debug_assert!` on a path only the GPU reaches never executes: guard such a path with `assert!`.
- CI has no GPU or Vulkan SDK: it builds with `ARTEMIS_SKIP_SHADERS=1` and runs fmt, clippy and the CPU tests only.
- The model's math is in `docs/reference/qwen35moe-semantics-b11509.md`; the reference forward pass
  (`crates/qwen36/src/reference.rs`) implements it line for line.
- Release and gate binaries are built with `tools/ps/ReleaseFlags.ps1`'s flags (the cargo home remapped, so the binary
  holds no path from the build machine): `tools/build-release.ps1` and `tools/run-gpu-tests.ps1` both set them, for
  every cargo command they run. A plain `cargo build` differs only in those path strings, and the first gate build after
  one rebuilds the workspace. Leave `RUSTFLAGS` unset; the scripts refuse to run with it.
- Tests that need the model file are `#[ignore]`d and read `ARTEMIS_MODEL`. `tools/run-gpu-tests.ps1` runs them and
  the oracle comparisons, and appends its records to the milestone's evidence folder (milestone 2's by default,
  `-Milestone m1` for milestone 1's and `-Milestone m3` for milestone 3's; [docs/evidence.md](docs/evidence.md)
  summarises them). The server milestone adds four blocks that start a real `artemis serve`
  (`tools/ps/ArtemisServer.psm1` starts and stops it): `serve`, `serve-idle` and `serve-queue` run test 7 against it,
  and `smoke` runs OpenCode and Claude Code against it (`tools/smoke/`). All four bind
  `127.0.0.1:8090` with a temporary key, refuse 8080, and refuse to start if 8090 is busy or an `artemis` process runs.
- The llama.cpp oracle builds with `oracle/build.ps1 -Root <folder>`; `tools/run-gpu-tests.ps1 -Oracle` (or
  `ARTEMIS_ORACLE`) names the same folder.
- Test data follows `testdata/README.md`: permissive licenses only in the repo, every file in `manifest.toml`.
- Test 2 is teacher-forced and per operation ([docs/architecture.md](docs/architecture.md#correctness-and-testing)): the
  reference rounds as llama.cpp's CPU does and continues from llama.cpp's value at every tap.
- The GPU side is three crates: `vk` (device, memory, recording), `kernels` (GLSL shaders, and a CPU twin per kernel
  in `kernels::twin`) and `engine`. Every kernel states its tolerance (`Kernel::tol`); `tests/twins.rs` checks the
  twins against `qwen36::ops` in CI, and test 1 checks the kernels against the twins on the GPU.
- The shaders' SPIR-V is committed in `crates/kernels/spirv/` (one `.spv` per shader and a `MANIFEST` of source and
  SPIR-V hashes), so a build with no Vulkan SDK makes real kernels. `build.rs` uses a committed file when the shader's
  sources match the manifest (the hash covers its `.comp` and every other file under `shaders/` that is not a `.comp`,
  so a stray file there marks every shader stale), and runs glslc from `VULKAN_SDK` only for a stale shader, with a
  warning. To change a shader: edit it, run `cargo run -p kernels --bin update-spirv` on a machine with the SDK (it also
  deletes a `.spv` that no shader produces), and commit the `.comp`, the `.spv` files and `MANIFEST` together. Before a
  gate, run `cargo run -p kernels --bin update-spirv -- --check` (exit 0 means the committed bits are glslc's output and
  `spirv/` holds no other .spv). `cargo test -p kernels --test spirv_fresh` fails for a stale, missing or extra file, in
  CI too.
- The GPU engine is checked tap by tap, teacher-forced against the reference with an f16 KV cache (`artemis dump --cpu
  --f16-kv`, then `artemis gpu-taps`): each tap's bound comes from its kernels' tolerances (`test2::gpu_bound`).
  `tools/run-gpu-tests.ps1 -Tests 1, 2-gpu` runs both.
- The expert cache: VRAM slots filled at load from `crates/engine/data/qwen36-expert-profile.bin`, sized to the budget
  or `--cache-slots` (the most it may hold, lent and grown rows included). It is allocated last and fitted against the
  budget the driver gives once it and the session are allocated (`engine::fit_cache`): this driver lowers the budget by
  1,741 MiB past 11.5 GB of usage. Prefill's buffers are allocated before it. The cache leaves the 0.5 GiB margin
  (`VRAM_MARGIN`) and `DRIVER_WORK` (64 MiB) free. `DRIVER_WORK` is the VRAM the driver allocates on first use, about 25
  MiB on the first prefill chunk and 1 MiB on the first decode, so the margin stays free once prefill has run.
  `check_fits` counts both when it refuses a model before the load. Its memory is chunks of whole slot rows (a row is
  one slot in every layer, 76,439,552 bytes; 14 rows a chunk), plus the rows the staging buffers lend between staged
  prefill chunks. The adaptive policy (`policy.rs`, `--cache-policy static` turns it off) scores each expert by its
  routes, decayed with a half-life of 128 tokens, and swaps within an upload cap of 8 MiB a token (`allowance`: a
  prefill chunk's share goes to the step after it, a staged chunk swaps nothing, no step spends more than 2 GiB); every
  256 tokens and between sessions `BudgetWatch` (`watch.rs`) retires the last chunk when the margin is gone and grows
  one back after two checks with room for it, the margin and `GROW_SLACK` (32 MiB: the driver has spent part of
  `DRIVER_WORK` by then). Table changes are staged and applied by `apply_slot_updates` at the start of the next step;
  swaps, fills and retired rows go through `engine::swap::SwapPlanner`, which the fuzz tests, and a step's copies go out
  in one transfer submit. Nothing may make a kernel's arithmetic depend on where an expert lives: test 5 needs
  bit-identical logits.
- The adaptive cache's rules (plan 2c). `--upload-cap` takes 1 to 2,048 MiB a token (up to 2 GiB), and `Engine::open`
  refuses a cap of 0 or past 2 GiB before the model loads (`Error::BadUploadCap`). `--cache-policy static` still fills
  rows that come back empty (lent or grown) in the profile's order and never swaps, so a comparison run isn't a smaller
  cache; `--force-swap-every-token` always runs static, so its swap is the step's only one. The policy's scores and its
  bank (the copies prefill chunks saved for the step after them) belong to the engine and persist across sessions and
  `Session::reset`: a new session starts from the last one's scores. Growing stops at `--cache-slots`, or at every
  expert without it, so the cache may grow past its size at load; a grow is checked again with its chunk allocated and
  given back if the margin and `GROW_SLACK` aren't kept, and growing then waits 2, 4, … 64 checks with room. A chunk is
  freed only once its rows are dead (every eviction drained, every cancelled copy landed), and
  `Engine::cache_slot_bytes` and `cache_slot_addr` refuse a dead row (`Error::DeadSlot`). Each poll sends the copies
  started for the last step prepared, so one submit carries one step's copies (at most 2 GiB of picks, the forced swap
  and `request_swap`'s). That rests on the engine waiting for each step (`submit_step`), and `prepare_step` `assert!`s
  each poll's bytes against the copies started for the last step prepared (and since): a pipelined engine must bound the
  batch itself. Test hooks: `Engine::set_budget_cut` and `set_grow_cut` (bytes taken off the budget readings, so a test
  can shrink the cache and refuse a grow), and `Options::budget_every`, `own_slots` and `chunk_rows`.
- Prefill: `Session::prefill` and `prefill_each` (`crates/engine/src/prefill.rs`) run a prompt in chunks of up to 2,048
  tokens, one submit per layer, logits only where asked. Every prefill kernel (`matmul_*`, `matmul_cm_*`, `conv_seq`,
  `gdn_seq`, `attn_prefill`, `attn_prefill_cm`, `moe_group`, `experts_matmul_*`, `experts_matmul_cm_*`, and decode's
  kernels' `*_tokens` forms) sums each output in an order fixed by the positions, so results don't depend on the
  chunking; keep it so. Prefill's buffers come out of the cache's VRAM. `tools/run-gpu-tests.ps1 -Tests prefill` checks
  it against decode forced onto its routing, checks that chunks of 2,048 and of 500 give the same bits, and times the
  27K-token prompt's submits (each under 300 ms). A chunk's attention is capped in query-key pairs a layer, so a long
  prompt's deep chunks come out shorter than 2,048 and no submit nears the watchdog: `ATTN_PAIRS` (2,048 × 12,288) for
  the f32 `attn_prefill`, `ATTN_PAIRS_CM` (2,048 × 41,984) for `attn_prefill_cm`, which `Engine::attn_pairs` picks by
  the kernel the engine runs. Each is measured (`ATTN_PAIRS_CM` from `examples/attn_speed.rs`: the same time a submit as
  the f32 kernel has under its cap); re-measure it when its kernel changes. Chunk lengths never change a result.
- How chunk invariance is guaranteed differs by kernel. In the f32 matmul family (`matmul_*` and `experts_matmul_*`, the
  `_cm_` kernels aside: see the split-f16 bullet below) it is guaranteed by `precise` accumulators with unfused
  products: each product is rounded, then added, in order, as the twin adds. Explicit `fma` was tried and moved prefill
  away from decode (final state 4.8e-4 against 2.25e-4), so don't reintroduce it. In `attn_prefill` (even and odd
  queries in different registers) and `gdn_seq` (a token's slot in a 16-token block) it relies on the compiler treating
  every lane alike, which the batch and chunk tests pin on every gate run. `precise` there would likely move them away
  from decode as `fma` moved matmul. `attn_prefill_cm` is guaranteed by its structure and by the hardware: key blocks
  of 32 align to position 0, so a query meets the same blocks in any batch; every sum runs in an order fixed by shapes
  (score slices in dimension order, a block's four quarter-sums in order, one PV slice a block); a block that lies
  wholly past a row's own position is an exact no-op for that row (its max stays, α is exactly 1.0 by a select, p is 0.0
  by the mask, V's rows past the last position are staged as zeros); and rows of a tile are treated alike by the
  hardware, as in the `_cm_` matmuls. `attn_prefill_cms_results_dont_depend_on_the_batch`,
  `attn_prefill_cm_results_dont_depend_on_the_subgroup_count` and the 2,048-vs-500 check pin it on every gate run.
- Profiling: `artemis generate --profile FILE` turns on `Engine::set_timing`. Every step's recorder is then profiled
  (`vk::Device::record_profiled`): a timestamp after every command, tagged with its layer (`TAG_EMBED`, `0..n_layer`,
  `TAG_OUTPUT`), plus each step's CPU time by phase. `crates/tools/src/profile.rs` sums them per kernel and per layer. A
  profiled run is never a timed one. Every step's recorder comes from `Engine::record`; keep it so, or its commands go
  untimed. The submit trace is the light alternative: with `ARTEMIS_SUBMIT_TRACE=FILE` set, `artemis generate` writes
  each decode submit's host times (recorded, submitted, woken) and its first and last GPU timestamps on the host clock
  (`vk::SubmitTrace`, calibrated with `VK_KHR_calibrated_timestamps`), labelled by step; `tools/py/submit_gaps.py`
  turns it into the gaps per step. Unset, the device opens as before and no command or clock read is added. The log
  holds at most `vk::MAX_TRACE_ENTRIES` submits; the rest, and any submit whose timestamps can't be read or calibrated,
  are counted (`Engine::take_submit_trace_dropped`) and reported by `generate`, never failing a step.
- The speed gate: `tools/run-gpu-tests.ps1 -Tests speed` runs the spike's write, edit and explain prompts
  (`testdata/speed/`) at seeds 1234 and 1235 and two cold prefills of the 27K-token prompt, then `artemis speed`. It
  also runs the write prompt once with `--base-time` and reports the base time (`artemis speed --base`), never gated.
  `-Tests depth` reports decode at 8,192, 16,384 and 27,000 positions (the long prompt cut there, its next 128 tokens
  forced, 127 decoded). `-Tests attn-depth -BaselineArtemis <pinned artemis.exe>` decodes at 4,096, 27,000 and 65,000
  positions on the pinned binary and on the current build, and passes when the current one decodes at least 0.98 times
  the pinned rate at the first two, at least equal at 65,000 and, under `-AttnStage A`, bit-identical in its logits
  fingerprints. The speed block reports, and gates only with `-SpeedGate`; records that aren't the gate's runs
  always fail it. From plan 2c that includes a record run with `--cache-policy static`, `--upload-cap` (whatever its
  value) or `--no-lend`, and a record from before plan 2c, which has no cache policy and reads as static. The edit
  prompt quotes PSF text: CPython v3.13.3's `Lib/json/encoder.py` is committed in `testdata/speed/sources/` with its
  licence (`LICENSE-PSF`), and so is the rendered prompt, so `tools/py/build_speed_prompts.py --stage check` fetches
  nothing. `-Tests profile` and `-Tests speed-llama` (llama.cpp, port 8095) report only. Evidence goes to milestone 2's
  folder (`-Milestone m1` for milestone 1's). Every speed block checks the PCIe link with `tools/py/gpu_link.py check`
  before and after it: gen 3, or a link that cannot be read, stops the run at that block's start (retrain the link with
  `pnputil /restart-device`, see that script), and a link that fell during the block fails its aggregate with
  `link dropped`. The runner sets its own CPU affinity when it starts (`-Affinity`, hex, default every CPU in the
  system's mask) and children inherit it; it restores the console's own mask on exit. Each server process's mask is
  read right after it starts and again, freshly, before it stops, and a record fails with problem `affinity` if either
  differs from the runner's. `-GpuName` names the adapter whose link is checked.
- Queues: a submit waits only for what the host has seen the other queue finish (`Queue::synced`), so a swap's copy can
  overlap the next step. A copy is published only after the planner has polled it complete, and copied only after
  polling its drain: that is what makes the free wait sufficient. A step's copies (the policy's swaps and fills, the
  forced swap) go out as one `stage_async` submit, whose leading barrier lands them after every earlier transfer, a
  failed chunk's staging copies included; sent one submit a copy, a 1,072-copy reshape grew the driver's usage by 82 MiB
  (plan 2c). `crates/engine/tests/planner_fuzz.rs` checks the planner's invariants. `Options::poison_swaps` fills a slot
  with NaNs before each copy, so a step that reads mid-copy fails (`gpu_ordering.rs`, `real_model.rs`). Work that reads
  something still in flight on the other queue needs an explicit wait (`Recorder::wait`).
- Prefill's staging: two VRAM buffers, each with room for a layer's non-resident experts, come out of the cache
  (`fit_cache` sizes them together; `--no-staging` turns them off; a budget too small for them loads without them).
  Staging takes about 7 slots per layer (131 → 124 at `--ctx 65536`, measured in plan 2b). A chunk of at least
  `STAGE_FROM` tokens (128, measured: `examples/stage_threshold.rs`) is staged: the prefill table is built from the
  engine's mirror of the slot table (`stage::plan`), layer l's copy (`vk::Device::stage_async`) goes into buffer l % 2
  once layer l − 2 has finished, and layer l's submit waits for it (`Recorder::wait`). HOST entries past a buffer's
  capacity stay HOST. Between staged chunks the buffers lend their VRAM to the cache as whole rows (`stage::lend_plan`;
  `--no-lend` turns it off): a staged chunk retires them, and the next step that isn't staged adds them back empty for
  the policy to fill. `Options::poison_staging` makes every copy late and poisoned, so a missing wait fails
  (`gpu_staging.rs`). Reads from staging count as `Stats::staged`, not hits.
- Prefill's split-f16 kernels (the decision of 2026-10-10): `matmul_cm_*` and `experts_matmul_cm_*` run on 16 × 16 × 16
  f16 cooperative matrices with every weight and input split into two f16s, hi = f16(v) and lo = f16(v − hi), so that hi
  + lo holds each value to about 2⁻²² (three products per 16-value block: lo·hi, hi·lo, hi·hi). Plain f16 failed the
  prefill-against-decode bounds. Each 32-value slice of k is summed into a zeroed partial on the cooperative matrices
  and then added to the output in f32: the hardware's own accumulate measured biased, its error growing linearly with k
  (plan 2b Task 2's ruling). Their outputs measured within 1e-6 relative L2 of their twins in plan 2b's gate (up to
  7.2e-7 for the dense kernels, 9.5e-7 for the grouped ones at k = 4,096), inside the 1e-5 tolerance. The router,
  `ssm_alpha`, `ssm_beta`, the shared expert's gate input and the output head stay on the f32 kernels. An input above
  f16's 65,504 makes its hi infinite, the outputs it reaches NaN, and the step fails (`Error::NonFinite`);
  `--f32-prefill` runs the f32 kernels, for that and for comparisons. Their chunk invariance rests on the hardware
  treating every row of a tile alike; the batch tests and the 2,048-vs-500 check pin it on every gate. Attention joins
  them (`attn_prefill_cm`, the cooperative-matrix prefill attention plan): q and the softmax weights p are split the
  same way (K and V are exact f16 in the cache, so each needs no lo); QKᵀ and PV each take two products a 16-value block
  (lo·K then hi·K; lo·V then hi·V), summed per 32-value slice into a zeroed partial and then added in f32 (`precise`;
  the output is `O ⊙ α + partial`). It measured at most 7.44e-7 relative L2 from its twin (14 test-1 cases; tolerance
  1e-5). Plain f16 for q or p was not built; the twin tests show it would sit about 4.4e-4 (q) and 6.0e-5 (p) from exact
  attention, against 4.9e-7 split. The f32 `attn_prefill` stays the fallback: no cooperative-matrix shape,
  `--f32-prefill`, or a head dimension that is not a multiple of 64 or is over 256 (`attn_prefill_cm_fits`).
- Profiling (from plan 2b) also records each prefill step's compute submits and staging copies as spans on one device
  clock (`StepTiming::spans`); `profile.rs` reports the copies' time and how much of it ran beside compute (hidden).
- Decode (plan 2d). Past 3,072 cached positions (`ATTN_GQA_FROM`) decode's attention runs `attn_decode_gqa`: one
  workgroup per 256-position chunk and KV head, serving the 8 query heads that read it, so each cached row is read
  once. Each head's arithmetic is `attn_decode`'s, operation for operation, and the two write the same bits on this
  driver (`the_gqa_attention_writes_attn_decodes_bits`): keep it so, or the step that crosses 3,072 positions changes a
  result. Neither kernel's sums are `precise`, so that equality rests on the compiler treating the two alike (fusing or
  not fusing the same products), as `attn_prefill`'s chunk invariance does; the bit test pins it on every gate.
  Plan 6b made decode attention faster with every output bit kept (the example's hashes equal the base's): each lane
  of the old `attn_decode_gqa` read its own position's K row, 16 bytes a load, which held the K pass near 100 GB/s.
  Now each chunk's K goes through shared memory in tiles of 32 positions × 128 values, 16 lanes reading 256 contiguous
  bytes, and thread 32g + r runs the dot product of tile position r for head g, the same chain of products as before.
  `attn_decode_gqa_tokens` runs one workgroup per query, chunk and KV head. `attn_combine` and `attn_combine_tokens`
  spread each head over workgroups of 64 dimensions (`ATTN_COMBINE_DIMS`), take M by a tree and load 16 splits ahead of
  their sums, each sum still one thread's in split order. One layer's call on the RX 9070 XT, GPU clock, best of
  1,000 (`attn_decode_speed`; before 6bbc6fb, after 99fa805, 2026-11-01): 4,096 positions
  60.8 to 38.4 µs, 27,000 231.4 to 139.1, 65,536 728.0 to 307.6 (0.42); the verify form at t = 4, 210.8 to 50.3,
  462.4 to 275.9 and 1,602.2 to 545.4.
  `quant.glsl`'s `unit_dot` (decode's matvecs and expert kernels only) reads x four floats at a time, F32, BF16, Q4_K and
  Q5_K weights 16 bytes at a time and Q8_0 and Q6_K weights 2 bytes at a time, and its sums are `precise`: every decode
  dot product equals its twin bit for bit (`matvec_equals_its_twin_bit_for_bit`). The wide loads need 16-byte-aligned
  rows, slots and activations: `matvec` and the expert dispatches assert it, and the engine refuses a misaligned VRAM or
  HOST slot entry (`kernels::dispatch::check_slot_addr`). The sampler takes its top-k in one pass over the logits
  (`sample.rs`'s `top_k`), the same tokens as selecting from every candidate, for k up to 256 (`TOP_K_ONE_PASS_MAX`;
  the pass is O(n·k), so a larger k keeps the old selection). `generate --base-time` (`Options::base_time`) has decode
  read a resident expert for every HOST one, for the base time; the speed and hit-rate gates refuse such runs. Test 1's
  block runs the bit tests and two engine tests that pin the alignment checks (`staging_a_misaligned_slot_is_refused`,
  `writing_a_misaligned_slot_is_refused`).
- The CPU miss path (plan 2e). Off by default: `--cpu-share S` (`Options::cpu_share`, 0.0 to 1.0, default 0;
  `set_cpu_share` changes it between tokens) sends that share of each layer's HOST misses to AVX-512 workers
  (`cpu_miss::share_q8` of 256 experts, the same ones in every layer; below 1/512 it is taken as 0, `path_on`). The plan
  2e A/B found share 1 13.2% slower than share 0 (79.714 against 91.801 tok/s), so the default stays 0 and the path is
  built but off; the figures are in [docs/evidence.md](docs/evidence.md#the-adaptive-cache). How it works:
  - A decode-only overlay, the way `base_time` is one: each step's expert kernels read a copy of the slot table in which
    the chosen HOST entries (matched by their host address) say `WHERE_CPU`. Prefill, staging, the planner and the MTP
    block read the real table, so a `WHERE_CPU` entry there still fails the step (`ExpertOnCpu`), and at share 0 no path
    starts and the expert dispatches refuse `WHERE_CPU` as before. `--base-time` and a non-zero share are refused
    together, and the speed and hit-rate gates refuse a run with a non-zero share.
  - The GPU publishes a layer's `x` and routed ids (`miss_publish`) into a type-5 mailbox (`MemKind::AmdUncached`, the
    only memory whose atomics both sides see promptly) during the submit; the routing log is not readable until the submit
    returns. A coordinator thread polls it and fans the experts out to a worker pool (`crates/cpu_expert`); `miss_wait`
    spin-waits for the answer and the combine adds it at the routed index, in the same order as a GPU expert. The
    wait is bounded at 20 ms a token: on a timeout the rest of the token's waits are skipped, and a following dispatch
    computes the missed experts from the host address, so the token still matches share 0 (`Stats::cpu_timeouts`;
    `Stats::cpu_misses` counts the CPU's experts). The fallback keys on the wait's outcome, never on the done flag
    alone: a late answer is wasted, not trusted.
  - Bit-identical to the HOST path (the decision of 2026-10-13): the four expert gate/up shaders (`experts_gate_up`,
    `experts_matmul_gate_up`, `experts_matmul_cm_gate_up` and plan 5a's `experts_gate_up_tokens`) use `silu_det` (f32 add,
    sub and mul only), the worker uses the twin's dot order plus the same `silu_det`, and every CPU compute runs with FTZ
    and DAZ on (`cpu_expert::with_ftz`, including the pool's jobs), because this driver's plain shaders flush f32
    denormals. Keep the four together.
    Conv, `gated_rmsnorm` and the shared expert keep `silu`.
  - Expert copies are host-cached memory (type 3, `MemKind::HostCached`) whatever the share: the CPU reads it at DRAM
    speed, the GPU as fast as type 1. Embedding rows stay type 1. The workers (eight, one per physical core) pin inside
    the process affinity mask (`cpu_expert::pin_set`) at `TIME_CRITICAL`; no code calls `SetProcessAffinityMask`, and a
    mask the pins would fill is refused. Eight spinners run while the path is on (a share of at least 1/512,
    `path_on`). `tools/run-gpu-tests.ps1 -Tests 5 -CpuShare 1` runs test 5 with the path on. The hit-rate formula is
    unchanged.
- Tests 3, 4 and 5 compare end to end, each prompt through prefill: `tools/run-gpu-tests.ps1 -Tests 3, 4, 5`. Test 3
  feeds `artemis generate --force-tokens` llama.cpp's greedy tokens from `testdata/golden/expected` and counts where its
  top token differs, against llama.cpp CPU's run along the same tokens in `testdata/golden/forced-cpu` (`artemis test3`;
  `artemis-oracle forced` made that floor). Test 4 runs `artemis kl` against a `llama-perplexity` KL base in the oracle
  folder's `runs\kl` (`artemis test4`). Test 5 compares five `generate --all-logits` runs at different cache sizes
  (`artemis test5`). From plan 2c its runs at 26, 102 and the most slots must run the adaptive policy and start at least
  one swap, so they swap live, and the forced run (102 slots, a swap forced every step) must run static, so its one swap
  a step stays exact.
- Test 6: `tools/run-gpu-tests.ps1 -Tests 6` runs the 20 golden prompts cold, by path 1 and by path 2 (`artemis warm`,
  one warm path in a fresh process; `artemis test6` judges). Each path's pooled flips against the oracle are at most 48,
  and prompt 20's prefilled positions 12,288 to 27,415 match the unbroken run bit for bit. `warm` poisons the live KV
  between the dirtying prompt and the restore (`Session::poison_kv`), so a restore that skipped its copy cannot pass by
  luck. `-Tests snapshot-times` (`artemis snapshot-times`) only reports: the copy times and the link's rates for
  `HostCached` and `HostPinned`.
- The worker and snapshots (plan 3a). `request.rs` (the request and its events), `logprobs.rs`, `reuse.rs` (the snapshot
  tree, the choice, protection and eviction, the unpark order: pure), `pool.rs` (the host pool's chunks, its free list
  and its limit at load), `snapshot.rs` (the copies, restore, parking), `turn.rs` (the generation loop on a `Live`),
  `worker.rs` (the thread, its commands and the lifecycle over a `Loader`), `loaded.rs` (the real engine, session and
  snapshots as a `Live`, and its `Loader`) and `fake.rs` (a fake `Live` and `Loader` for CI). The pool is
  `MemKind::HostCached`: the GPU copies it as fast as `HostPinned` (27.79 against 27.79 GB/s device to host, measured
  2026-10-19), but the CPU reads it at 30.25 GB/s against 0.31, and parking reads all of it with the CPU. `KvRecord`
  (the tokens behind the live KV), the usable flag and the pending-copy wait live in `Session`, and every write goes
  through them: no caller keeps its own record. A restore is one blocking compute submit. The snapshot copies come from
  `dev.record()`, are not steps and are not profiled. `Snapshots`' `Drop` waits for the device through a `!Send`
  `vk::IdleHandle`. Parking fences with the engine (`fence_for_park`), the engine and session drop, then `into_parked`
  copies the pool into host `Vec`s, so the engine's 18.68 GiB of expert copies are freed before the copy needs the room.
  The policy's CI tests are in `reuse.rs`, the loop's in `turn.rs`, the worker's in `worker.rs` on `fake.rs`; the
  tiny-model GPU tests are `crates/engine/tests/gpu_snapshot.rs` and `gpu_worker.rs` (in the runner's test-1 block).
- The server (plan 3b): the `server` crate is the OpenAI- and Anthropic-compatible HTTP API on an engine behind its own
  `Backend` seam. It depends on `qwen36` and never on `engine` (C2); `fake.rs` (a scripted `FakeBackend` on its own
  thread, the same FIFO and lifecycle as 3a's worker) is the seam's only non-test implementation in that crate. Plan 3c
  adds `artemis serve` (`crates/tools/src/serve/`: `backend.rs` is the adapter from `engine::Worker` to `Backend`,
  `run.rs` the command). Its settings, endpoints, logs, lifecycle, exit codes and known windows are documented in
  `docs/server.md`: change that page with any setting or behaviour you change. The hidden `--stop-on-stdin-close` is how
  scripts stop it on Windows, and `artemis test7` drives it. `template.rs` renders the GGUF's own chat template
  (committed as `testdata/template/qwen36-chat-template.jinja`) through minijinja with our filters (`pyfmt.rs`: floats
  as C++'s `%g`, as the oracle writes them); `convo.rs`'s `is_user_query` is R1 (the cut's `user_query` flag);
  `pipeline/` splits reasoning, tool calls and stop strings. Conformance is two files: `tests/conformance.rs` (the
  oracle's prompts byte for byte, `testdata/template/{requests,prompts,anthropic}`) and `tests/clients.rs` (T1 and T2
  over HTTP). The tests bind port 0, never 8080. `tools/py/` holds the capture and fixture scripts: `record_clients.py`
  (the recording stub), `capture_clients.py` (drives OpenCode and Claude Code against it), `scrub_captures.py` (`--check
  testdata/clients`; no Claude Code prompt text is committed, C25) and `build_template_fixtures.py` (llama-server
  b11509's `/apply-template`).
- Soak readiness (milestone 4; plan 4a). The server gained a stall watchdog (`crates/server/src/watchdog.rs`: it reads
  the progress counter the adapter in `crates/tools/src/serve/backend.rs` keeps, and ends the process with exit code 4
  after writing `stall`, `stats` and `exit`), `GET /v1/stats` with a `stats` heartbeat line, a panic hook
  (`crates/tools/src/serve/panic.rs`: the `panic` line withholds any message that is not a fixed string), the
  `load_failed` and `reload_failed` lines, and the request line's `error_message`, `decode_ms`, `decode_tokens` and
  `decode_misses`. `docs/server.md` documents all of it; the engine crate is unchanged. The hidden `artemis serve`
  flag `--test-stall-at-token N` exists for the stall test alone: it writes a `config_warning` and `serve` refuses it
  on port 8080. `tools/config/artemis-production.toml.in` is the production config as a template (`{{MODEL}}`,
  `{{PORT}}`, `{{API_KEY_FILE}}`, `{{LOG_DIR}}` inside TOML literal strings), filled by `fill_template` (Rust) and
  `New-ArtemisConfig` (`tools/ps/ArtemisServer.psm1`; `Start-ArtemisServer -Config`) under the same rules. Test 7 gained
  the cases `speed-fresh` (`--speed-run <prompt>:<seed>|long`, by itself, one server process per run) and `chat-tools`.
  Four runner blocks use the template: `serve-stall` (runs the one Rust GPU test,
  `cargo test --release -p artemis-tools --test serve_stall_gpu -- --ignored --test-threads=1`, which starts the server
  itself and rebuilds `artemis.exe` with the dev-dependencies' features, so the block builds the shipping one again
  after it), `serve-speed`, `prod-check` (the `serve`, `serve-idle` and `serve-queue` cases from the template, plus
  `chat-tools`) and `prod-check-smoke` (the two client smokes from the template). All four bind `127.0.0.1:8090` with a
  temporary key, refuse 8080 (a filled config that names it is refused too), refuse to start if 8090 is busy or an
  `artemis` process runs. `tools/py/m4_soak.py` prints the
  soak report from the production logs (read-only; its tests run in CI).
- First-run setup (milestone 6; plan 6b). `artemis setup` (`crates/tools/src/setup/`: `mod.rs` is the command and the
  lock, `rig_bench.rs` the calibration's bench over a real engine, `choose.rs` and `calibrate.rs` the rules, order and
  time limit) measures the PC and writes a per-user tuned file (`crates/server/src/tuned.rs`), which `serve` layers
  under `artemis.toml`. The first `artemis serve` that finds no tuned file runs it before the port opens
  (`first_run_setup`); `docs/server.md` documents the command, the lock, the log lines and the GPU and driver warning.
  Set ARTEMIS_TUNED=off when you run a smoke script or a GPU test by hand: a serve started without it runs setup and
  writes your real tuned file. The Rust tests that start `serve` set it on their `Command`, and so does the runner: it
  sets `off` when it starts and puts the caller's value back when it exits. The one exception is the `first-run` block,
  which points it at a `tuned.toml` in its own run folder for the length of that block.
- Plan 6b's runner blocks. `llama-search` searches llama.cpp b11509's best non-speculative configuration on the GPU
  named by `-GpuName`, and `speed-llama` and `speed-llama-65k` measure it (report only; `-LlamaConfig` names a searched
  `best.json` to reuse, and when it names one whose driver and llama.cpp build match today's, `llama-search` warns that
  the search can be skipped). `first-run` is the optimizer's gate: run it in the same invocation as `serve-speed-spec`
  and `speed-llama`, which give it the hand-tuned "on" median and llama.cpp's median. It starts `artemis serve` with no
  tuned file, so setup runs before the port opens, waits up to 720 s for `/health`, stops the server by closing its
  stdin, then makes the six decode runs of `serve-speed-spec` in fresh processes that read the file setup wrote. It
  passes when setup took at most 600 s without hitting its limit, the binary imports no WinHTTP or WinINet DLL, every
  run applied the file on the runner's CPU mask, the tuned median is at least 0.97 times the "on" median and above
  llama.cpp's, and llama.cpp ran a searched configuration on the `-GpuName` device. The comparison with llama.cpp is
  Artemis at 65,536 positions against llama.cpp at 32,768. A missing reference, or a llama.cpp run that is not a
  searched best on that device, is recorded as `first-run precheck` and is not an attempt; like `serve-speed-spec`,
  the block allows two attempts per commit. `tools/py/m6b_summary.py --evidence <dir> --gate <commit>` prints these
  blocks' tables from `results.jsonl` and checks that milestone 5c's speed still holds (exit 1 if not).
- Plan 6b's figures and their rules. The gate ran on 04af2d0 (2026-11-01): the speed blocks' tables, the three
  `artemis setup` runs (80.4 to 85.8 s, on 81345d6) and the decode attention figures (before 6bbc6fb, after 99fa805).
  The raw records are kept with the run records outside this repository; quote each figure with its commit.
  llama.cpp's Vulkan device order is not stable from one boot to the next (the RX 9070 XT has been Vulkan0 and
  Vulkan1, and the integrated GPU the other), so `llama-search` and `speed-llama` pick the device by name
  (`-GpuName`) and every record names the device it used. Read a llama.cpp figure with its record's `device`,
  `affinity` and `hygiene` beside it: those runs depend on the machine's state, and a figure from another day is not
  comparable with it. A llama.cpp speed run needs a CPU-quiet PC: no builds, downloads or scans before or during
  it. On 2026-11-01 background scanners (Windows Defender's MsMpEng and the search indexer's SearchIndexer, working
  through new build outputs) slowed llama.cpp's CPU expert threads about 2.3 times while Artemis was unaffected,
  so check their CPU time around a run. The runner records it: `hygiene.scanner_cpu_s` and `hygiene.busy_pct` at a
  speed block's start, `scanner_cpu_delta_s` at its end, and `tools/py/m6b_summary.py` flags a block whose
  scanners used more than 30 CPU seconds or whose start was more than 10% busy. A flagged llama.cpp figure is
  rerun on a quiet PC before anyone publishes it. The `attn-depth` block compares Artemis with Artemis.
- Speculative decoding (milestone 5; plan 5a). On by default; `[speculation] enabled = false`, `artemis serve
  --speculation-enabled false` and `artemis generate --speculation off` turn it off, and then the engine makes today's
  decode calls and allocates nothing for it. Each step drafts tokens from the request's own history (prompt lookup) and
  checks them in one decode-shaped pass.
  - The verify step: `Session::verify(e, tokens)` runs `[x, d1 … dk]` (at most 16 tokens, `MAX_VERIFY`) in one command
    buffer with the token kernels (`matvec_tokens_*`, `attn_decode_tokens`, `attn_decode_gqa_tokens`,
    `attn_combine_tokens`, `moe_unique`, `experts_gate_up_tokens`, `experts_down_tokens`), each doing its single-token
    kernel's arithmetic per token. It reads the DeltaNet and conv state and never stores it (`conv_seq` and `gdn_seq`
    gain a `store` flag), saving each layer's inputs. `Session::commit(e, kept)` then rewinds to the kept position and
    **replays the kept tokens through the saved inputs in place** (plan 5a D1: there is no second state set, so no
    new VRAM at the production config: the step's buffers are a second carve of prefill's idle work area). Every
    sampled token still comes from the request's one `Sampler`, in order (`engine::spec::SpecState::step` is the one
    step loop; the turn, `generate` and test 6 all use it).
  - `gdn_seq` is within tolerance of `gdn_step`, not bit-equal (D1), so speculation on and off are **not** expected to
    be identical: they can differ at near-ties. The bar is test 3's near-tie rule (`spec-golden`), not identity.
    Identity is reported only. A kernel test calls `kernels::testing::identity`, which appends a
    `{"kind":"identity",…}` line to `ARTEMIS_EVIDENCE` that carries no `pass`; the runner's test 1 keeps those lines out of
    its pass grouping and records them as `test 1 identity` (equal and differing cases per kernel). Test 5 reports its
    sixth, speculation-off run beside the five gating ones (`reports`), and `serve-speed-spec` lists each pair's
    completion hashes. Never turn an identity report into a check without the maintainer's word.
  - Four runner blocks: `serve-speed-spec` (serve-speed's eight runs, each as an "off" process,
    `--speculation-enabled false`, then an "on" one; done-when 1 is the "on" median of six against 110.0, 2 and 3 are
    98% of "off" per prompt and for both prefill figures, 6 the "off" run's old bounds. **In 5a only done-whens 3 and 6
    gate its status**: 1 and 2 are measured and decide 5b through the maintainer, so a dw1 miss is not a runner bug and a pass is
    not dw1 met. One whole rerun after a miss, then the maintainer: the budget is per commit, and the third attempt on one commit
    is refused. It also fails when an "off" run shows speculation counts, an "on" run shows none, or the "on" decode runs
    made no verify step between them), `serve-shrunk` (report only: 85 and 30 cache slots; a pair more than 2%
    slower with speculation goes to the maintainer), `spec-golden` (`artemis spec-check golden`: the 20 golden prompts off then
    on in one process; gated on the command's exit code and `verdict.json`) and `spec-long` (report only: 20,000
    prompt tokens). `serve-speed-spec` and `serve-shrunk` start servers like the other milestone-4 blocks: they bind
    `127.0.0.1:8090` with a temporary key, refuse 8080, refuse to start if 8090 is busy or an `artemis` process runs.
    They leave `tools/ps/ArtemisServer.psm1` and `tools/smoke/*` alone, so the
    recorded hashes of the smoke scripts stay valid. Runs that can't speculate pass `--speculation off`: `hits`, `speed`'s base-time run and
    test 5's `-CpuShare` mode (the engine refuses a CPU share, and `--base-time`, with speculation); test 5's five gating
    runs use `--draft-len 4`. `tools/py/m5_summary.py` prints the gate's tables from `results.jsonl`.
  - Plan 5b adds the model's own MTP block (`blk.40`) as a second draft source beside prompt lookup (`sources = ["lookup",
    "mtp"]`; `--sources` / `--speculation-sources`; `artemis serve --speculation-sources lookup` loads without it).
    - A row at position `q` takes token `q` and the main model's post-`output_norm` hidden row at `q - 1`. Only a
      **draft** row runs the whole block; a row that just feeds later attention (the prompt window, the catch-up) is
      **KV-only** (M2): `e`, `g`, `eh_proj`, `attn_norm`, `attn_k`, `attn_v`, `k_norm`, RoPE and the KV write, with no
      query, attention, MoE or head.
    - The MTP's KV is a **capped buffer of rows** in position order (M3): `mtp_ctx` rows (8,192 by default, 16 MiB), the
      newest half slid to the front when a pass would overflow it, rolled back by position. The prompt's last `mtp_window`
      positions (2,048) are written during prefill.
    - **Catch-up** (M5): every step copies its final hidden rows into a ring of `mtp_ring` rows (256); a draft pass first
      writes KV-only rows for every position with no row whose `h` is in the ring. Positions further back stay holes.
    - **`blk.40`'s experts are HOST** (464 MiB of host-cached memory, 486,539,264 B; its non-expert weights are 40.4 MiB
      of VRAM, about 58.5 MiB with the session's KV and ring). They are never in the expert cache, and their reads are
      not counted in `decode_misses`. The request line and `generate --json` carry each source's drafted and accepted
      counts (`spec_drafted_lookup`, `spec_drafted_mtp` and so on); `drafted` and `accepted` stay the sums.
    - Checks: `artemis-oracle mtp-taps` runs llama.cpp's MTP graph teacher-forced and the CPU reference continues from its
      values at every tap (`artemis dump --cpu --mtp`, `artemis test2-mtp`: test 2's rule over the MTP's taps);
      `artemis mtp-check --tokens <the file given to mtp-taps>` checks the GPU's MTP logits against the oracle's on the real
      model (top-1 equal outside near-ties; relative L2 reported). The oracle binary is the one test 2 uses, so it must be
      built from this checkout's `oracle\`. A wrong MTP draft can only lower speed: `spec-golden` stays the correctness
      gate.
    - New runner blocks: `2-mtp` (the oracle's `mtp-taps` over test 2's four inputs, cached under
      `runs\mtp-taps\<model sha256>`; records `2-mtp` and `2-mtp-gpu`, both gating) and `mtp-32gb` (`artemis generate` at 8
      tokens, speculation off then on with MTP: each run's `memory` block and its `pool_at_26gib` in whole GiB as
      `pool_gib`; `pass` if the pool with MTP on is at least 2 GiB, `ask-owner` (needs review) if less, and `fail` when the "on" run's heap
      usage or pool is not a number, `mtp_vram_bytes` is not above 0 or the host bytes did not grow by exactly 486,539,264).
      `spec-long` gains a third run, "on" with `--mtp-window 20000 --mtp-ctx 65536` (report only).
    - **`-SpecPlan 5a|5b`** (default `5b`; every record carries `plan`): under 5a `serve-speed-spec` passes on done-whens 3
      and 6, as above; under 5b on done-whens 1, 2, 3 and 6, and each of the six decode runs adds a third fresh process,
      mode `lookup` (`--speculation-sources lookup`). The aggregate records `lookup_median`, `mtp_gain` (the "on" median over
      the lookup median) and `mtp_pays` (`mtp_gain >= 1.02 - 1e-9`), report only: the maintainer's word decides whether MTP ships. A
      "lookup" run with no `spec_steps` or with `spec_drafted_mtp` above 0, "on" runs that never drafted with MTP, and an
      attempt missing a lookup run all fail the block. A rerun of a missed `serve-speed-spec` is its own invocation (22
      processes, over an hour). Test 5's gating runs keep `--draft-len 4`: MTP drafts where lookup has none, so its rows run
      at every cache size, and test 5 fails a set in which no run drafted with MTP when MTP is a source.
- The hit-rate gate: `tools/run-gpu-tests.ps1 -Tests hits` runs golden prompts 01 and 05 to 18, 128 greedy tokens at
  `--ctx 65536`, each with the adaptive policy and with `--cache-policy static`, then `artemis hits`
  (`crates/tools/src/hits.rs`) judges them. A prompt's rate is the run's, the cache's share of every routed read,
  prefill included (`hits / (hits + staged + misses)`, as test 5 counts it; the decision of 2026-10-12), with
  decode's alone reported beside it. The code prompts (01, 05 to 11) may average at most 1 point below static; prose
  (12 to 15) and tool use (16 to 18) must average above it. It refuses a pair that isn't greedy with `--max-tokens 128` on
  the committed prompt, that was given `--upload-cap`, `--no-lend`, `--profile`, `--force-swap-every-token` or a
  prefill option (`--no-staging`, `--stage-from`, `--f32-prefill`), or whose two runs differ in their tokens, `--ctx`,
  cache size or lent rows. The block always runs its 15 prompts: `-Prompts` doesn't apply.
- Device rules: the f32 kernels need 24,576 bytes of compute shared memory per workgroup (`experts_matmul_gate_up`;
  `attn_prefill` declares 19,136 and `moe_group` 18,432), past the 16,384 Vulkan guarantees. `Kernels::new` refuses a
  device below that (`vk::Error::SharedMemory`); the RX 9070 XT reports 32,768. Every other kernel stays within 16 KiB:
  keep it so, or add the kernel's size to `BIG_SHARED_KERNELS` (`crates/kernels/src/dispatch.rs`), with a comment at its
  shader's arrays. The cooperative-matrix kernels declare up to 30,720 bytes (`COOP_SHARED_KERNELS`: 20,480 for
  `matmul_cm` and `experts_matmul_cm_down`, 30,720 for `experts_matmul_cm_gate_up` and 29,952 for `attn_prefill_cm`) and
  are built only where the device lists the f16 shape and allows that much (`Kernels::coop`); otherwise prefill runs the
  f32 kernels. Workgroup counts stay at most 65,535 per dimension (`Recorder::dispatch` asserts it). Whenever a kernel
  or its dispatch changes, run the kernel and engine GPU tests once under `VK_LAYER_KHRONOS_validation`
  (`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; `VK_LOADER_DEBUG=layer` shows it loaded): they must report no VUID.
  Plan 1c-ii's first `attn_prefill` (51,904 bytes) passed every test while exceeding the device's limit; only validation
  caught it.
- The CPU miss path's probe is `m0 roundtrip --mode spin --flag-mem amd-uncached --decode-shaped`: 40 waits a command
  buffer, each after a layer's worth of VRAM reads, timed on the GPU. It spins the GPU on a host flag: run it with its
  timeouts, in the foreground. Plan 2e built the path on this handoff (the paragraph above).
- Secrets: the forge's `secret-scan` job runs gitleaks 8.30.1 over the full history on every push, and the same pinned
  gitleaks runs locally. `tools/scan-secrets.ps1` scans the working tree (`-Mode tree`, the default), the commits no
  remote has (`-Mode unpushed`), a range (`-Mode range -Range a..b`) or a folder (`-Mode dir -Path <folder>`); a
  finding is printed with the secret redacted and the exit code is 1, and exit code 3 means the scan could not run (a
  bad range, a failed download), which is never reported as clean. The rules are gitleaks' defaults from
  `tools/gitleaks.toml`, so a `.gitleaks.toml` in the scanned content has no effect. `tools/git-hooks/pre-push` runs the
  scan over each pushed range and stops a push that holds a finding or cannot be scanned. Git cannot install a hook on
  clone, so run `pwsh tools/install-hooks.ps1` once in each clone: it sets `artemis.gitleaksUrl`, the forge release
  folder the first scan downloads gitleaks from, derived from `origin` (or pass `-GitleaksUrl`), and then
  `core.hooksPath` to `tools/git-hooks` (shared by every worktree). A clone of the public GitHub repository has no forge
  release, so the script leaves the hook off there unless the pinned zip is already cached; that repository's secret
  check is the `secrets` job in `ci.yml`. A checkout that has no `tools/git-hooks`, such as an older
  branch, runs no hook. The download uses git's own credential, is checked against a pinned SHA-256 and is cached under
  `%LOCALAPPDATA%\artemis-tools`. `tools/check-public-snapshot.ps1` also scans its export. A reviewed false positive gets
  an inline `gitleaks:allow` comment, which every scan honours, or a `.gitleaksignore` line. The hook and CI print
  commit-qualified fingerprints (`commit:file:rule:line`), which suppress only the git-history scans; a line without the
  commit (`file:rule:line`) also covers the tree and dir scans. The publish set leaves `.gitleaksignore` out, so an
  export needs the inline comment. `git push --no-verify` skips the hook and is for a reviewed false positive only.
