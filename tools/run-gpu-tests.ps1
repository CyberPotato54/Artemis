#Requires -Version 7.2
# The GPU tests. Appends one JSON line per result to results.jsonl in the evidence folder of -Milestone (m2 by default),
# and exits non-zero if any gating result fails.
#   8      test 8 (the tokenizer) and the GGUF-only checks
#   2      test 2: the reference against llama.cpp b11509, teacher-forced per operation
#   1      test 1: every kernel against its twin, on random inputs and real tensors; plus the vk and engine GPU tests.
#          The token kernels' and the verify pass's identity lines (kind = identity: whether a kernel's bits equal the
#          single-token kernel's) are not checks: they are summarised per kernel as `test 1 identity` (equal
#          and differing cases, report only) and never count as a failure
#   2-gpu  the GPU engine teacher-forced against the reference with an f16 KV cache, tap by tap, on the
#          tiny model and the real inputs in testdata/test2/inputs
#   gen    a greedy generate smoke run: its text is reported, not checked, and a crash fails the run
#   prefill  the long golden prompt's first 4,600 tokens through prefill and through decode, with
#          decode forced onto prefill's routing (the free-running comparison is reported); the same bits with chunks
#          of 2,048 and 500; and the 27K-token prompt with every submit well under the watchdog
#   3      test 3: fed llama.cpp b11509's greedy tokens on the 20 golden prompts after reading each prompt through
#          prefill, Artemis's top token may leave them at most 1.5 times as often as llama.cpp CPU's does
#   4      test 4: Artemis's mean KL from llama.cpp's CPU against llama.cpp's own Vulkan-vs-CPU KL, over testdata/kl,
#          each chunk through prefill
#   5      test 5: three golden prompts at 0, 26 (10%), 102 (40%) and the most slots that fit, and at 102 with a swap
#          forced every step; all five must give bit-identical logits at every position, the prompt's through prefill.
#          The runs with a cache swap live under the adaptive policy; the forced run's policy is static.
#          The five runs speculate with a fixed draft length (--draft-len 4, so every cache size runs the same steps;
#          MTP drafts where lookup has none, so MTP's rows run at every cache size, and test 5's verdict needs one run
#          to have drafted with it) and each must have made verify steps; a sixth run,
#          --speculation off --cache-slots 256, is
#          passed as --report-run plain=... and only reported (`reports` in the record). With -CpuShare above 0 the
#          runs pass --speculation off and no --draft-len (the engine refuses a CPU share with speculation) and there
#          is no sixth run
#   hits   the cache policy's hit-rate gate: golden prompts 01 and 05-18 with 128 greedy tokens, the
#          adaptive cache policy against --cache-policy static; code may average at most 1 point below static, prose
#          and tool use must average above it. Always runs its 15 prompts: -Prompts doesn't apply, the judge needs them
#          all. Both runs of each pair pass --speculation off: the gate judges the cache policy, and verify steps would
#          change what its hit rates mean
#   speed  the speed gate: the write, edit and explain prompts at seeds 1234 and 1235
#          (median decode tok/s, bound 46.5) and two cold prefills of the 27K-token prompt (slower tok/s, bound 1,000);
#          reported, unless -SpeedGate makes a speed below its bound fail the run. Records that aren't the gate's runs
#          always fail it. Also reports decode's base time: the write prompt at seed 1234 with
#          --base-time, every routed expert read from VRAM (never gated); that run passes --speculation off, since the
#          engine refuses --base-time with speculation
#   profile  generate --profile on each speed prompt (seed 1234) and on the 27K-token prompt's prefill (report only)
#   depth  decode at long contexts (report only): the 27K-token prompt cut at 8,192, 16,384 and 27,000 tokens,
#          then its next 128 tokens forced one a step (greedy): the first comes from prefill's logits, so each run decodes
#          127 tokens at that depth
#          `speed`, `depth` and `profile` speculate, as the shipped default does (only the base-time run
#          above, test 5's -CpuShare mode and `hits` pass --speculation off); their records carry settings.speculation
#   attn-depth  decode against a pinned binary at 4,096, 27,000 and 65,000 positions (the long prompt cut at the first
#          two; at 65,000 its tokens repeated three times, cut there), the next 128 tokens forced (127 decoded), greedy,
#          --speculation off, --ctx 65536. -BaselineArtemis names the pinned artemis.exe (required); the current build
#          is the new one. Per depth, in order: base with fingerprints, new with fingerprints, then base, new, base, new
#          timed (--no-fingerprints); each side's tok/s is the mean of its two timed runs. Records: an `attn-depth run`
#          per depth (base_tok_s, new_tok_s, ratio, fingerprints_equal, hygiene) and an `attn-depth` line with `status`
#          and `problems`. Pass iff new >= 0.98 x base at 4,096 and 27,000, new >= base at 65,000 and, under -AttnStage
#          A (the default), the fingerprints are equal at every depth; under B equality is reported. The 65,000 ratio is
#          reported as gain_65k. Each generate process's CPU mask is read right after it starts and polled until it
#          exits; a mask other than the runner's at the start or at the last read fails the row and the aggregate with
#          problem 'affinity' (the run record's `masks` lists them per run).
#   llama-search  a coordinate-descent search for llama.cpp b11509's best non-speculative configuration on the GPU
#          named by -GpuName (report only): tools\py\llama_search.py screens each variant with one request for up to
#          -LlamaSearchMinutes (default 75), then runs the top two and the start through six speed runs each (outside
#          that budget, about 15 minutes more). It writes runs\llama-search\search.json and best.json; the record
#          carries `best`, `device`, `budget_hit`, `variants` and `hygiene`. A best.json it wrote is the configuration
#          speed-llama and speed-llama-65k then run (`config_source` search). Runs just before speed-llama
#   speed-llama  llama.cpp b11509's llama-server on the speed prompts without speculation (report only), on the GPU
#          named by -GpuName (chosen by name from `llama-server --list-devices`, never by index). Its record carries the
#          device, `affinity` and `affinity_at_stop` (the server's mask, which must equal `affinity_asked`: the searched
#          config's own, else the runner's) and `config_source`: `search` (the best.json a llama-search block wrote in
#          this invocation), `file` (the searched best.json that -LlamaConfig names) or `plan 2a` (neither: the spike's
#          configuration)
#   first-run  plan 6b's first-run gate. Run it in the same invocation as serve-speed-spec and speed-llama (after
#          them), with -LlamaConfig or llama-search so that llama.cpp ran a searched best on the -GpuName device. If the
#          "on" median of one or the llama.cpp median of the other is missing, or llama.cpp ran another configuration
#          or device, it records `first-run precheck` (status fail, with the reason) and runs nothing; a precheck is
#          not an attempt. The block sets ARTEMIS_TUNED to a tuned.toml in its own run folder (restored to off
#          afterwards) and starts `artemis serve` itself from the production template, with stdin redirected: that is a
#          first launch, so it runs setup before the port opens. It waits up to the setup limit plus 120 s (720 s) for
#          /health, checks that the 8090 listener is its child, records `first-run launch` (first_launch_s, the
#          setup_done line as `setup`, whether setup's start text was printed, the exe's imports and any WinHTTP or
#          WinINet one as network_imports, both CPU masks) and stops the server by closing its stdin. Then six fresh
#          processes (write, edit and explain at seeds 1234 and 1235, mode "tuned", `first-run run`, each with
#          affinity, affinity_at_stop and tuned_applied: its `tuned` line must name this block's file and apply keys
#          from it) and one `first-run` aggregate: attempt (1 plus this commit's first-run aggregates; a third is
#          refused, a second miss goes to the owner), tuned_median, spec_on_median, tuned_over_hand, llama_median,
#          tuned_over_llama, llama_device, llama_affinity, llama_config_source, identity (report only, only when setup
#          chose the defaults), link_dropped, status and problems. Pass iff the launch reached /health with a
#          setup_done line of at most 600 s that did not hit its limit, no network import, every run whole on the
#          runner's mask and applying the file, the link held, the tuned median at least 0.97 x the "on" median (a
#          product, with a 1e-9 allowance), above llama.cpp's median (llama.cpp runs at 32,768 positions, Artemis at
#          65,536), and llama.cpp ran a searched best (`search` or `file`) on the -GpuName device. Binds 127.0.0.1:8090
#          with a temporary key.
#   speed-llama-65k  the same configuration at a 65,536-token context (report only): llama.cpp's like-for-like
#          figure beside Artemis at 65,536; a load failure at that context is recorded as the figure
#   6      test 6: each golden prompt run cold (the oracle's tokens forced, every
#          position fingerprinted), then on warm path 1 (prefill half the prompt, snapshot, dirty the engine, restore,
#          prefill the rest) and warm path 2 (decode half the oracle's tokens, snapshot, dirty, restore), each in a
#          fresh process. Each path's flips, pooled, may be at most 48, and the 27K-token prompt's prefilled positions
#          after the path-1 restore must equal the cold run's bit for bit
#   snapshot-times  the snapshot copy measurement (report only): the PCIe link's rates
#          for the pool's two memory kinds, the state and the KV copied to the host and back, a restore after a
#          dirtying, and the long prompt's prefill with and without the server's snapshots; writes
#          runs\snapshot-times\times.json
#   serve  test 7 on the real engine, server A (the default config) on
#          127.0.0.1:8090 with a temporary key: artemis test7 --case load,requests,budget,reuse,disconnect,speed, then
#          tools/test-messages.ps1 (its exit code and cache_read_input_tokens are recorded), then the reuse
#          report over its logs. Preflight first (26 GiB of RAM free). A case whose pool check says
#          "inconclusive: pool below 2 GiB" is a stop-and-report: the run stops after the block, nothing reruns
#   serve-idle  server B (--idle-unload-s 30 --cache-slots 128): --case idle,unload-stream
#   serve-queue  server C (--queue-timeout-s 5): --case queue
#          The three server blocks refuse to start if 8090 is busy or an artemis process runs, keep their key under
#          runs\keys\<block>\, and scan their run directory for it when they stop (tools\ps\ArtemisServer.psm1).
#          The script refuses to run at all while $env:CARGO_TARGET_DIR is set: it runs the binary in target\release.
#   smoke  the client smoke: tools\smoke\opencode.ps1, then tools\smoke\claude-code.ps1.
#          Each starts its own server on 127.0.0.1:8090 (preflight first), runs the real client against a scratch repo
#          with a planted bug, judges the wire log and writes runs\m3c-smoke\<client>-<time>\verdict.json; the runner
#          records both verdicts and the reuse report over both servers' logs (report only). A script refuses to start
#          without its canary marker (runs\m3c-smoke\canary-ok-smoke-<client>, written by capture_clients.py smoke-canary).
#          The smoke root is fixed in capture_clients.py, so -Oracle must be the folder that script names
#   Production-template blocks. These four start the server from the production config template
#   (tools\config\artemis-production.toml.in, filled for 127.0.0.1:8090, a temporary key and the run's own log
#   directory) and record the template, its hash, the filled config's hash, the overrides and the artemis.exe's hash:
#   serve-stall  the #[ignore]d GPU test a_stuck_engine_logs_stall_and_exits_4 (cargo test --release -p artemis-tools
#          --test serve_stall_gpu): a hidden hook stalls the engine at token 16 and the process must log `stall` and
#          exit 4 within stall_s plus 10 s. Built first (not timed), then run with a 10-minute limit. Its cargo test
#          rebuilds artemis.exe with the dev-dependencies' features, so a later server block builds it again
#   serve-speed  the speed gate through the server: one fresh server process per run with --park-snapshots false, the
#          six decode runs (test 7's speed-fresh, seed outermost) and prompt 20's prefill twice, then one process for
#          the mixed-traffic `speed` case (report only). Decode median (decode_tokens / decode_ms) must reach 46.5
#          tok/s, the slower prefill 1,000 tok/s, every verdict `ran`, every stop graceful and no `stall` line in any
#          log. Run as -Tests serve-speed,speed, the engine tool's speed gate follows on the same boot
#   prod-check  test 7's real-engine cases and chat-tools on servers started from the template: prod-check (load,
#          requests, budget, reuse, disconnect, speed, chat-tools, then tools/test-messages.ps1 and the reuse report),
#          prod-check-idle (--idle-unload-s 30 --cache-slots 128: idle, unload-stream) and prod-check-queue
#          (--queue-timeout-s 5: queue). A `stall` line in any block's logs fails it
#   prod-check-smoke  the two client smokes of the smoke block, with the server started from the template, in a block of
#          its own so it can follow the canaries in a separate window
#   Speculative decoding (prompt lookup, and the MTP drafter). Speculation is on in the shipped config, so every block
#   above that decodes speculates unless its entry says otherwise. These blocks measure it:
#   serve-speed-spec  the gate's speed block (done-whens 1, 2, 3 and 6): the sixteen processes of
#          serve-speed's eight runs, each as an "off" process (--speculation-enabled false) and then an "on" one (the
#          template as it is), back to back, seed outermost. The leaves are <run>-off and <run>-on. Records a
#          `serve-speed-spec run` per process and one `serve-speed-spec` aggregate: the "on" median of six against 110.0
#          (done-when 1), each prompt's on/off ratio (2) and both prefill figures' (3) at 0.98, and the "off" runs
#          against serve-speed's bounds (6); the status is pass iff done-whens 3 and 6 pass and every run is whole under
#          -SpecPlan 5a, and iff done-whens 1, 2, 3 and 6 pass under 5b. Under 5b each of the six decode runs
#          (not the two long ones) adds a third fresh process after its "off" and "on": mode "lookup", the template with
#          --speculation-sources lookup (so it loads without MTP's memory); 22 processes. The aggregate then records
#          lookup_median, mtp_gain (the "on" median over the lookup median) and mtp_pays (mtp_gain >= 1.02 - 1e-9),
#          report only: they are not in the status. A "lookup" run with no spec_steps or with spec_drafted_mtp above 0
#          is a mode problem, "on" runs whose sources include mtp and that never drafted with it are a problem (each run
#          records its `sources`), and an attempt missing a lookup run never passes. A rerun of a missed attempt is its
#          own invocation of this block alone. Every record carries plan. The
#          record's `attempt` is 1 plus the serve-speed-spec aggregates this commit already holds in results.jsonl; a
#          third attempt on one commit is refused ("a second miss goes to the owner"): the rerun budget is per
#          commit, so a new commit is a new candidate with its own two attempts, never a way round a miss. Every row is also
#          checked for its mode: an "off" decode run with spec_steps, an "on" one without, or "on" runs that made no verify
#          step between them are problems. Binds 127.0.0.1:8090
#   serve-shrunk  report only: write, edit and explain at seed 1234 with --cache-slots 85, then 30, each off then on in
#          fresh processes. Lists per pair decode tok/s, miss MiB per token, the accept rate, the plain-step share and the
#          on/off ratio; a pair where "on" is more than 2% slower than "off" is listed in needs_owner and makes the
#          status ask-owner. Binds 127.0.0.1:8090
#   spec-golden  artemis spec-check golden: the 20 golden prompts greedy for 256 tokens, free-running, speculation off
#          and then on in one process; a first divergence is allowed only at a near-tie (test 3's 0.05). It is gated on
#          the command's exit code and verdict.json's status
#   spec-long  report only: the first 20,000 tokens of the long golden prompt and 512 sampled tokens (seed 1234), off,
#          then on, then on with the MTP's window over the whole prompt (--mtp-window 20000 --mtp-ctx 65536);
#          records decode tok/s, misses, the speculation counts (with each source's) and whether the completions' tokens are
#          identical
#   The MTP drafter adds:
#   2-mtp  test 2's MTP half: artemis-oracle mtp-taps over each of test 2's four inputs (cached under
#          runs\mtp-taps\<model sha256>, like test 2's main taps), artemis dump --cpu --mtp forced from them,
#          artemis test2-mtp; recorded as one 2-mtp record (status pass iff all four pass; worst tap over them). Then
#          artemis mtp-check (GPU: the GPU's MTP logits against the oracle's, top-1 equal outside near-ties), recorded as
#          2-mtp-gpu. The oracle is the binary test 2 uses, <-Oracle>\build\bin\Release\artemis-oracle.exe, so it must be
#          a build of this checkout's oracle\ (it needs mtp-taps)
#   mtp-32gb  the 32 GB check: artemis generate at 8 tokens, speculation off and then on with MTP, --json; records
#          each run's memory block and its pool_at_26gib in whole GiB as pool_gib. Judged only when the "on" run's
#          host_heap_usage and pool_gib are numbers (else status fail, "not measured"), the MTP was really loaded
#          (mtp_vram_bytes above 0 and host_bytes up by exactly blk.40's 486,539,264 B of experts) and the pool with MTP on
#          is at least 2 GiB: pass, else ask-owner
# -SpecPlan 5a|5b (default 5b) picks the serve-speed-spec status rule and its run set (above); every 5a/5b record carries it.
# -Milestone picks the evidence directory: m2 (the default), m1, m3, m4, m5 or m6.
# -Model (the Qwen3.6 GGUF) and -Oracle (the folder oracle\build.ps1 -Root built into) default to the environment
# variables ARTEMIS_MODEL and ARTEMIS_ORACLE; the script stops at once when either is missing.
# -Prompts limits the golden-set tests to the prompt ids given, for checks and reruns.
# -WriteKlSidecars (with -Tests 4) writes test 4's base-cpu.json and vulkan-floor.json for a base and floor already on
# disk, once, then goes on as usual.
# -Affinity (hex, such as 0xffff0000; default every CPU in the system's mask) is the CPU affinity the runner sets on
# itself at start-up (and puts the console's own mask back on exit); every child inherits it. -GpuName (default 'AMD
# Radeon RX 9070 XT') names the adapter whose PCIe link is checked. Speed hygiene: the speed blocks (speed, profile,
# depth, attn-depth, spec-long, llama-search, speed-llama, speed-llama-65k, serve-speed, serve-speed-spec, serve-shrunk,
# first-run) start with SpeedHygiene (the link must read gen 4 or better, else a failed "<block> hygiene" record and a
# stop) and end with LinkAfter (a "<block> link after" record, taken after the last measured process stops; WithLink
# folds the result into the block's aggregate, and a link that fell fails it with problem 'link dropped'). Every speed
# record carries `hygiene` (pcie_gen, adapter, affinity, desktop_gpu_mib and the CPU quietness at the block's start:
# scanner_cpu_s, the CPU seconds of MsMpEng and SearchIndexer, and busy_pct, the system's busy percentage over 2 s); the
# "link after" record and the aggregate carry scanner_cpu_delta_s, the scanners' CPU seconds used during the block
# (m6b_summary.py flags more than 30, or a start more than 10% busy). Every server-backed speed run record carries
# `affinity` (read right after the server started) and `affinity_at_stop` (read just before it stopped); a mask other
# than the runner's at either read fails the record with problem 'affinity' (a child re-pinned by Process Lasso shows
# here). The `generate` runs of speed, profile, depth, spec-long and attn-depth are read the same way (`masks` on their
# records): a mask other than the runner's fails the block's record with problem 'affinity'.
# -LlamaSearchMinutes (default 75) is the screening budget of the llama-search block.
# -BaselineArtemis (a path) is the pinned artemis.exe the attn-depth block compares the current build against;
# -AttnStage (A or B, default A) says whether that block requires equal fingerprints (A) or only reports them (B).
# -LlamaConfig (a path) names a searched best.json (best, finalists, driver and llama_build present) for speed-llama and
# speed-llama-65k to run; a llama-search block in the same invocation takes precedence, and with neither the plan 2a
# configuration runs. A llama-search block run beside a -LlamaConfig whose driver and llama_build match today's warns
# that the search can be skipped.
# ARTEMIS_TUNED: the runner sets it to off when it starts (every block that starts `artemis serve` then skips the tuned
# file and the first-run setup) and puts the caller's value back on exit. Only the first-run block points it at a file,
# its own, for the length of that block.
# -CpuShare (with -Tests 5) runs test 5's five configurations with generate --cpu-share (test 5 with the CPU
# miss path on, with --speculation off), into runs\test5-share<value> so the share-0 set is kept; 0, the default, is
# test 5 with speculation as above.
param(
  [string]$Model = $env:ARTEMIS_MODEL,
  [string]$Oracle = $env:ARTEMIS_ORACLE,
  [string[]]$Tests = @('8', '2', '1', '2-gpu', 'gen', 'prefill', '3', '4', '5', 'speed'),
  [string[]]$Prompts = @(),
  [switch]$WriteKlSidecars,
  [ValidateSet('m1', 'm2', 'm3', 'm4', 'm5', 'm6')][string]$Milestone = 'm2',
  [ValidateSet('5a', '5b')][string]$SpecPlan = '5b',
  [switch]$SpeedGate,
  [ValidateRange(0.0, 1.0)][double]$CpuShare = 0,
  [string]$Affinity = '',
  [string]$GpuName = 'AMD Radeon RX 9070 XT',
  [string]$LlamaConfig = '',
  [int]$LlamaSearchMinutes = 75,
  [string]$BaselineArtemis = '',
  [ValidateSet('A', 'B')][string]$AttnStage = 'A'
)
$ErrorActionPreference = 'Stop'
if (-not $Model) { throw 'pass -Model or set ARTEMIS_MODEL' }
if (-not $Oracle) { throw 'pass -Oracle or set ARTEMIS_ORACLE' }
# First of all, before any build, model check or block: cargo honours CARGO_TARGET_DIR but this script always runs
# $repo\target\release\artemis.exe, so a set variable builds one binary and tests a stale other.
if ($env:CARGO_TARGET_DIR) {
  throw "CARGO_TARGET_DIR is set ('$env:CARGO_TARGET_DIR'): cargo would build there, but this script runs target\release\artemis.exe in its own checkout. Clear it (Remove-Item Env:CARGO_TARGET_DIR) and run again"
}
# Every cargo command below builds with the release flags (tools\ps\ReleaseFlags.ps1: the cargo home remapped, so the
# binary holds no local path), the same as tools\build-release.ps1, so the gate measures the binary a release ships.
. (Join-Path $PSScriptRoot 'ps\ReleaseFlags.ps1')
Set-ArtemisReleaseFlags
# The prefill block's three real-model checks: each test's name (the test-1 block skips them by full name, since --skip
# matches by substring) and the evidence file it writes (the prefill block records them).
$prefillTests = @(
  @{ test = 'prefill_matches_decode_on_a_long_prompt'; evidence = 'vs-decode' },
  @{ test = 'prefill_is_bit_identical_whatever_the_chunk_size'; evidence = 'chunks' },
  @{ test = 'the_long_prompt_prefills_with_every_submit_well_under_the_watchdog'; evidence = 'long-prompt' }
)
$skipPrefill = @($prefillTests | ForEach-Object { '--skip'; $_.test })
# A name the runner doesn't know runs nothing, so it is refused before anything runs: Git Bash once split
# "-Tests 1, gen, 3" into "1," and two stray arguments, and the run passed having tested nothing.
$known = @('8', '2', '1', '2-gpu', 'gen', 'prefill', '3', '4', '5', 'hits', 'speed', 'profile', 'llama-search', 'speed-llama', 'speed-llama-65k', 'depth', 'attn-depth', '6', 'snapshot-times', 'serve', 'serve-idle', 'serve-queue', 'smoke', 'serve-stall', 'serve-speed', 'prod-check', 'prod-check-smoke', 'serve-speed-spec', 'serve-shrunk', 'first-run', 'spec-golden', 'spec-long', '2-mtp', 'mtp-32gb')
$unknown = @($Tests | Where-Object { $known -notcontains $_ })
if ($unknown.Count) { throw "unknown -Tests $($unknown -join ', '); the tests are $($known -join ', '). Run this script from PowerShell, not as pwsh -File from Git Bash" }
if ($Tests -contains 'attn-depth') {
  if (-not $BaselineArtemis) { throw 'attn-depth needs -BaselineArtemis, the path of the pinned artemis.exe to compare against' }
  if (-not (Test-Path -LiteralPath $BaselineArtemis -PathType Leaf)) { throw "-BaselineArtemis '$BaselineArtemis' is not a file" }
  # The block runs from the repo, so a relative path is made absolute against the caller's directory here.
  $BaselineArtemis = (Resolve-Path -LiteralPath $BaselineArtemis).Path
}
# The caller's ARTEMIS_TUNED, put back by the final `finally` (the runner runs in the caller's process). The runner sets
# it to off at the top of the main `try`, before any block, so every way out restores it.
$startTuned = $env:ARTEMIS_TUNED
$repo = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'ps\PeImports.ps1')
# The Python tools (tools\py) use only the standard library, so a plain interpreter runs them; nothing is installed or
# fetched. Found once here, so a missing or broken interpreter stops the run before any build or test.
. (Join-Path $PSScriptRoot 'ps\Python.ps1')
$python = Get-ArtemisPython
# Speed hygiene: the runner pins itself explicitly (the system's whole mask unless -Affinity says otherwise), so the
# children it starts inherit a known mask and the speed blocks can check it.
Import-Module (Join-Path $PSScriptRoot 'ps\Hygiene.psm1') -Force
$systemMask = Get-ArtemisAllCpuMask
$runnerMask = $systemMask
if ($Affinity) {
  $hex = $Affinity.Trim() -replace '^0[xX]', ''
  if ($hex -notmatch '^[0-9a-fA-F]{1,16}$') { throw "-Affinity '$Affinity' is not a hex mask such as 0xffff0000" }
  $runnerMask = [Convert]::ToInt64($hex, 16)
  if ($runnerMask -eq 0 -or ($runnerMask -band (-bnot $systemMask))) {
    throw "-Affinity $(Format-ArtemisMask $runnerMask) names no CPU or a CPU outside the system's mask $(Format-ArtemisMask $systemMask)"
  }
}
# The runner runs in the caller's process, so the console's own mask is saved here and put back in the final `finally`.
$startMask = Get-ArtemisAffinity
if (@($Tests | Where-Object { $_ -like 'serve*' -or $_ -like 'prod-check*' -or $_ -eq 'first-run' }).Count) {
  Import-Module (Join-Path $PSScriptRoot 'ps\ArtemisServer.psm1') -Force
}
# The production config's template: the production-template blocks start the server from it.
$template = Join-Path $repo 'tools\config\artemis-production.toml.in'
$bin = Join-Path $Oracle 'build\bin\Release\artemis-oracle.exe'
$runs = Join-Path $Oracle 'runs'
$evidence = Join-Path $repo "docs\superpowers\evidence\$Milestone\results.jsonl"
New-Item -ItemType Directory -Force (Split-Path $evidence) | Out-Null
$env:ARTEMIS_MODEL = $Model
if (-not $env:VULKAN_SDK) { $env:VULKAN_SDK = 'C:\VulkanSDK\1.4.363.0' }
# The committed SPIR-V must be glslc's output for the committed shaders before any block measures a binary built from it.
Assert-ArtemisSpirvFresh $repo
$commit = (git -C $repo rev-parse --short HEAD).Trim()
$script:failed = @()
# Set by serve-speed-spec for the first-run block of the same invocation: its "on" median of six, its status and the
# "on" runs' completion hashes by run (leaf). Set by speed-llama: median, device, affinity and config_source ($null
# when it did not run or failed).
$script:specOnMedian = $null
$script:specOnStatus = $null
$script:specOnFnv = $null
$script:llamaBest = $null

# Set by SpeedHygiene for the block it opens, cleared by LinkAfter: every record written in between carries it.
$script:hygieneNow = $null
$script:linkGenBefore = $null
$script:hygieneBlock = $null

function Record([hashtable]$r) {
  $r.date = (Get-Date -Format o)
  $r.commit = $commit
  if ($script:hygieneNow -and -not $r.ContainsKey('hygiene')) { $r.hygiene = $script:hygieneNow }
  Add-Content -LiteralPath $evidence -Value ($r | ConvertTo-Json -Compress -Depth 6) -Encoding utf8NoBOM
}

function Check([string]$what) { if ($LASTEXITCODE) { throw "$what failed (exit $LASTEXITCODE)" } }

# Windows' total dedicated GPU memory in use, in MiB, summed over the adapters; $null if the counter can't be read.
# Sampled while no artemis process runs, so it is the desktop's own use. Never fails the run.
function DesktopGpuMib {
  try {
    $c = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples
    return [math]::Round((($c | Measure-Object CookedValue -Sum).Sum) / 1MB, 1)
  } catch { return $null }
}

# The CPU seconds of the background scanners (MsMpEng, SearchIndexer) and the system's busy percentage over a 2 s
# sample, read at a speed block's start (SpeedHygiene) and, for the scanners, at its end (LinkAfter): a PC that is not
# CPU-quiet slows llama.cpp's CPU expert threads. Never fail a run; a reading that can't be taken is $null.
function ScannerCpu { return Get-ArtemisScannerCpu }
function BusyPct { return Get-ArtemisBusyPct -Seconds 2 }

# Speed hygiene, first in every speed block: the PCIe link must read gen 4 or better (gpu_link.py check) and the runner
# must still hold the mask it set. Records "<Block> hygiene" with the figures; a bad link (or an unreadable one) or a
# changed mask records status fail, notes the block as failing and throws, so no figure is taken. Returns the figures
# (pcie_gen, adapter, affinity, desktop_gpu_mib, scanner_cpu_s, busy_pct), which Record then attaches as `hygiene` to
# every record up to LinkAfter.
function SpeedHygiene([string]$Block) {
  $link = Test-ArtemisLink -Python $python -Repo $repo -Name $GpuName
  $mask = Get-ArtemisAffinity
  $h = @{ pcie_gen = $link.pcie_gen; adapter = $link.adapter; affinity = (Format-ArtemisMask $mask); desktop_gpu_mib = (DesktopGpuMib)
    scanner_cpu_s = (ScannerCpu); busy_pct = (BusyPct) }
  $problem = $null
  if (-not $link.ok) { $problem = $link.why }
  elseif ($mask -ne $runnerMask) { $problem = "the runner's affinity is $(Format-ArtemisMask $mask), not the $(Format-ArtemisMask $runnerMask) it set" }
  if ($problem) {
    Record @{ test = "$Block hygiene"; input = $Block; status = 'fail'; problem = $problem; hygiene = $h }
    $script:failed += "$Block hygiene"
    throw "${Block}: speed hygiene failed: $problem"
  }
  Record @{ test = "$Block hygiene"; input = $Block; status = 'pass'; hygiene = $h }
  $script:hygieneNow = $h
  $script:hygieneBlock = $Block
  $script:linkGenBefore = $link.pcie_gen
  return $h
}

# The link again, taken once the block's last measured process has stopped (and before its aggregate record, so the
# aggregate can carry the result): link_dropped is true when it is now below gen 4 or below what the block started with
# (or unreadable). Writes a "<Block> link after" record, adds "<Block> link dropped" to the failing list when it fell,
# closes the block (Record stops attaching `hygiene`) and returns @{ link_dropped; pcie_gen; hygiene;
# scanner_cpu_delta_s } for WithLink to fold into the aggregate. The record and the return carry the scanners' CPU
# seconds used since SpeedHygiene. A call when no block is open (already closed, or SpeedHygiene threw) does nothing and
# returns $null, so the final `finally` can call it for any block that threw.
function LinkAfter([string]$Block) {
  if (-not $script:hygieneNow) { return $null }
  $link = Test-ArtemisLink -Python $python -Repo $repo -Name $GpuName
  $dropped = (-not $link.ok) -or ($null -ne $script:linkGenBefore -and $null -ne $link.pcie_gen -and $link.pcie_gen -lt $script:linkGenBefore)
  $scanners = Get-ArtemisScannerDelta $script:hygieneNow.scanner_cpu_s (ScannerCpu)
  $r = @{ test = "$Block link after"; input = $Block; status = $(if ($dropped) { 'fail' } else { 'pass' }); pcie_gen = $link.pcie_gen; adapter = $link.adapter; link_dropped = $dropped
    scanner_cpu_delta_s = $scanners }
  if ($dropped) {
    $r.problem = 'link dropped'
    $r.why = $link.why
    $script:failed += "$Block link dropped"
    Write-Warning "${Block}: link dropped: $($link.why)"
  }
  Record $r
  $h = $script:hygieneNow
  $script:hygieneNow = $null
  $script:hygieneBlock = $null
  $script:linkGenBefore = $null
  return @{ link_dropped = $dropped; pcie_gen = $link.pcie_gen; why = $link.why; hygiene = $h; scanner_cpu_delta_s = $scanners }
}

# Adds a name to a record's `problem` without losing the one already there: 'affinity' then 'link dropped' reads
# 'affinity, link dropped'. A name already in the text is not added twice.
function AddProblem([hashtable]$Rec, [string]$Name) {
  if (-not $Rec.problem) { $Rec.problem = $Name; return }
  if (@("$($Rec.problem)" -split ', ') -notcontains $Name) { $Rec.problem = "$($Rec.problem), $Name" }
}

# A block's aggregate record with the after-link result folded in: `hygiene`, link_dropped and pcie_gen_after, and, when
# the link fell, status fail, problem 'link dropped' and the reason added to `problems` (when it has that list).
function WithLink([hashtable]$Rec, $Link) {
  if (-not $Link) { return $Rec }
  $Rec.hygiene = $Link.hygiene
  $Rec.link_dropped = $Link.link_dropped
  $Rec.pcie_gen_after = $Link.pcie_gen
  if ($Link.scanner_cpu_delta_s) { $Rec.scanner_cpu_delta_s = $Link.scanner_cpu_delta_s }
  if ($Link.link_dropped) {
    $Rec.status = 'fail'
    AddProblem $Rec 'link dropped'
    if ($Rec.ContainsKey('problems')) { $Rec.problems = @($Rec.problems) + "link dropped: $($Link.why)" }
  }
  return $Rec
}

# A mask as 0x-hex, or 'unreadable' for $null.
function MaskText($Mask) { if ($null -ne $Mask) { return (Format-ArtemisMask $Mask) } else { return 'unreadable' } }

# Whether two runs' logits fingerprints (the `positions` lists of their --json records) are the same: both lists must
# hold something, in the same count and order.
function SamePositions($A, $B) {
  $a = @($A | Where-Object { $null -ne $_ }); $b = @($B | Where-Object { $null -ne $_ })
  return ($a.Count -gt 0 -and $a.Count -eq $b.Count -and (($a -join ',') -ceq ($b -join ',')))
}

# The attn-depth verdict from one row per depth (@{ depth; base_tok_s; new_tok_s; fingerprints_equal }). Pass iff the
# new decode is at least 0.98 of the base figure at every depth under 65,000 and at least the base figure at 65,000 and,
# under stage A only, the fingerprints are equal at every depth (under B equality is reported, never judged). The ratios
# are not rounded. Returns @{ status; problems; gain_65k } (gain_65k is the 65,000 ratio, or $null when no such row
# ran).
function AttnDepthJudge($Rows, [string]$Stage = 'A') {
  $problems = @()
  $gain = $null
  $has65k = $false
  foreach ($row in @($Rows)) {
    $d = [int]$row.depth
    if ($d -ge 65000) { $has65k = $true }
    $base = [double]$row.base_tok_s
    $new = [double]$row.new_tok_s
    if ($row.affinity_problem) { $problems += "depth ${d}: affinity: $($row.affinity_problem)" }
    if ($base -le 0 -or $new -le 0) {
      $problems += "depth ${d}: a decode rate is missing or zero (base $base, new $new)"
    } else {
      $ratio = $new / $base
      if ($d -ge 65000) {
        $gain = $ratio
        if ($new -lt $base) { $problems += "depth ${d}: new $new tok/s is below base $base (ratio $ratio, bound 1)" }
      } elseif ($new -lt 0.98 * $base) {
        $problems += "depth ${d}: new $new tok/s is below 0.98 x base $base (ratio $ratio)"
      }
    }
    if ($Stage -eq 'A' -and -not $row.fingerprints_equal) { $problems += "depth ${d}: the fingerprints differ (stage A requires them equal)" }
  }
  if (@($Rows).Count -eq 0) { $problems += 'no depth ran' }
  elseif (-not $has65k) { $problems += 'depth 65000 did not run' }
  return @{ status = $(if ($problems.Count) { 'fail' } else { 'pass' }); problems = $problems; gain_65k = $gain }
}

# The fields every server-backed run record carries: the mask read after the start and before the stop, and, when either
# differs from the runner's, status fail with problem 'affinity' (the record's other fields stay).
function AffinityFields($r) {
  $f = @{ affinity = $(if ($null -ne $r.affinity) { Format-ArtemisMask $r.affinity } else { $null })
    affinity_at_stop = $(if ($null -ne $r.affinity_at_stop) { Format-ArtemisMask $r.affinity_at_stop } else { $null }) }
  if ($r.affinity_problem) { $f.status = 'fail'; $f.problem = 'affinity'; $f.affinity_why = $r.affinity_problem }
  return $f
}

# The affinity mask of a server process, or $null when it can't be read (the process is gone).
function ProcAffinity($Proc) { return Get-ArtemisProcessAffinity -Process $Proc }

# Runs one measured child to its end with its output and error redirected, and reads its CPU mask right after it starts
# and again on every poll until it exits (the last read taken is the one just before the stop; a child that is gone
# before the first poll keeps the start read). Arguments with white space are quoted. Returns @{ exit; affinity;
# affinity_at_stop; affinity_problem }, the problem set when either read is not the runner's mask ($runnerMask) or
# could not be read. The caller fails on `exit` as Check does.
function RunMeasured([string]$Exe, [string[]]$ArgList, [string]$Out, [string]$Err) {
  $quoted = @($ArgList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } })
  $p = Start-Process -FilePath $Exe -ArgumentList $quoted -NoNewWindow -PassThru -RedirectStandardOutput $Out -RedirectStandardError $Err
  $null = $p.Handle   # keeps the handle open, so ExitCode is readable after the exit
  $first = Get-ArtemisProcessAffinity -Process $p
  $last = $first
  while (-not $p.WaitForExit(250)) {
    $m = Get-ArtemisProcessAffinity -Process $p
    if ($null -ne $m) { $last = $m }
  }
  $p.WaitForExit()
  $r = @{ exit = $p.ExitCode; affinity = $first; affinity_at_stop = $last; affinity_problem = $null }
  $bad = @(@($first, $last) | Where-Object { $null -eq $_ -or $_ -ne $runnerMask })
  if ($bad.Count) { $r.affinity_problem = "the process's affinity was $(MaskText $first) after it started and $(MaskText $last) before it stopped, not the runner's $(MaskText $runnerMask)" }
  return $r
}

# Files one RunMeasured result under its tag: its mask fields into $Masks (AffinityFields) and, when a mask read was not
# the runner's, "<tag>: <why>" into $Problems.
function MaskNote($Gen, [string]$Tag, [hashtable]$Masks, [System.Collections.Generic.List[string]]$Problems) {
  $Masks[$Tag] = AffinityFields $Gen
  if ($Gen.affinity_problem) { $Problems.Add("${Tag}: $($Gen.affinity_problem)") }
}

# What serve-speed-spec hands to the first-run block of this invocation: this attempt's status always, and, only when
# the record passed (a miss or a dropped link makes its figures a poor reference), the "on" median of six and the "on"
# runs' completion hashes by run. A first-run block that finds no median stops at its precheck, which is not an attempt.
function HandOnSpec($Agg, $Rows) {
  $script:specOnMedian = $null
  $script:specOnFnv = $null
  $script:specOnStatus = $Agg.status
  if ($Agg.status -ne 'pass') { return }
  $script:specOnMedian = $(if (IsNum $Agg.on_median) { [double]$Agg.on_median } else { $null })
  $script:specOnFnv = @{}
  foreach ($row in @($Rows)) { if ($row.mode -eq 'on' -and $row.prompt -ne 'long' -and $row.fnv) { $script:specOnFnv[$row.leaf] = $row.fnv } }
}

# What llama-search hands to speed-llama and the first-run block: the path of the best.json it wrote, only when the
# record passed (a dropped link fails it). $null otherwise.
function HandOnLlamaSearch($Rec) {
  $script:llamaSearchBest = $null
  if ($Rec.status -ne 'fail' -and $Rec.path) { $script:llamaSearchBest = $Rec.path }
}

# Set by a llama-search block (the path of the best.json it wrote in this invocation); $null when none ran.
$script:llamaSearchBest = $null

# Which configuration llama.cpp runs, in order of preference: the best.json a llama-search block wrote in this
# invocation, the file -LlamaConfig names when it holds a searched result (best, finalists, driver and llama_build),
# else plan 2a's (no --config-json). Returns @{ source = 'search' | 'file' | 'plan 2a'; file }.
function LlamaChoice {
  if ($script:llamaSearchBest -and (Test-Path -LiteralPath $script:llamaSearchBest)) {
    return @{ source = 'search'; file = $script:llamaSearchBest }
  }
  if ($LlamaConfig) {
    if (-not (Test-Path -LiteralPath $LlamaConfig -PathType Leaf)) { throw "-LlamaConfig '$LlamaConfig' is not a file" }
    $names = @((Get-Content -LiteralPath $LlamaConfig -Raw | ConvertFrom-Json).PSObject.Properties.Name)
    if (@('best', 'finalists', 'driver', 'llama_build' | Where-Object { $names -notcontains $_ }).Count -eq 0) {
      return @{ source = 'file'; file = (Resolve-Path -LiteralPath $LlamaConfig).Path }
    }
    Write-Warning "-LlamaConfig '$LlamaConfig' holds no searched result (best, finalists, driver and llama_build); llama.cpp runs plan 2a's configuration"
  }
  return @{ source = 'plan 2a'; file = $null }
}

# One llama.cpp measurement (speed_llama.py) into $Dir\results.json on the GPU named $GpuName. The script pins itself to
# the runner's mask (the server inherits it) and reads the server's mask after it starts and before it stops. Returns
# the record's fields. status is 'ran', or 'fail' with a `problem` when the script wrote nothing, a mask read is not the
# runner's ('affinity'), or the script reported an error (unless $LoadFailureIsFigure: then the error is the figure).
function MeasureLlama([string]$Dir, $Choice, [int]$Ctx, [bool]$LoadFailureIsFigure) {
  New-Item -ItemType Directory -Force $Dir | Out-Null
  $out = Join-Path $Dir 'results.json'
  if (Test-Path $out) { Remove-Item $out }
  $a = @((Join-Path $repo 'tools\py\speed_llama.py'), $repo, $Model, (Join-Path $Oracle 'build\bin\Release'), '--out', $out,
    '--gpu-name', $GpuName, '--affinity', (Format-ArtemisMask $runnerMask))
  if ($Choice.file) { $a += @('--config-json', $Choice.file) }
  if ($Ctx) { $a += @('--ctx', "$Ctx") }
  # Out-Host: the script's console lines would otherwise join this function's return value.
  & $python -I @a | Out-Host
  $exit = $LASTEXITCODE
  $rec = @{ config_source = $Choice.source; status = 'ran' }
  if (-not (Test-Path $out)) {
    $rec.status = 'fail'
    $rec.problem = "speed_llama.py wrote no result (exit $exit); it crashed or was not started, see the console output above"
    return $rec
  }
  $r = Get-Content $out -Raw | ConvertFrom-Json
  $rec.config = $r.config; $rec.device = $r.device; $rec.args = $r.args
  $rec.affinity = $r.affinity; $rec.affinity_at_stop = $r.affinity_at_stop
  $rec.runs = $r.runs; $rec.median = $r.median
  # The mask this run asked for: the searched config's own (search or file), else the runner's, which is what
  # --affinity carried.
  $want = Format-ArtemisMask $runnerMask
  if ($Choice.file) {
    # The script reports the mask it asked for; failing that, the file's own, which is hex text or a JSON integer.
    $asked = $r.affinity_asked
    if (-not $asked) { $asked = (Get-Content -LiteralPath $Choice.file -Raw | ConvertFrom-Json).best.affinity }
    if ($asked -is [string]) { $want = Format-ArtemisMask ([Convert]::ToInt64(($asked -replace '^0[xX]', ''), 16)) }
    else { $want = Format-ArtemisMask ([long]$asked) }
  }
  $rec.affinity_asked = $want
  # A server that never became ready has no mask to compare: that is a load failure, not an affinity one. The one error
  # with no mask that is an affinity problem is the script's own "could not read the server's affinity"; any other error
  # text that happens to name affinity (a refusal about the mask it was given, say) is reported as the error it is.
  $neverStarted = $r.error -and $null -eq $r.affinity -and $r.error -notlike "could not read the server's affinity*"
  if (-not $neverStarted -and @(@($r.affinity, $r.affinity_at_stop) | Where-Object { $_ -ne $want }).Count) {
    $rec.status = 'fail'; $rec.problem = 'affinity'
    $rec.affinity_why = "the server's affinity was $(if ($r.affinity) { $r.affinity } else { 'unreadable' }) after it started and $(if ($r.affinity_at_stop) { $r.affinity_at_stop } else { 'unreadable' }) before it stopped, not the $want it asked for"
  } elseif ($r.error) {
    $rec.error = $r.error
    # Only a server that never became ready is a figure at 65k; any other error fails as usual.
    $loadFailure = $r.error -match 'exited with|wasn.t ready'
    if (-not ($LoadFailureIsFigure -and $loadFailure)) { $rec.status = 'fail'; $rec.problem = $r.error }
  } elseif ($exit) {
    $rec.status = 'fail'; $rec.problem = "speed_llama.py exited $exit"
  }
  return $rec
}

# The GPU driver version of the adapter named $GpuName (Win32_VideoController), or $null when it is not listed.
function GpuDriver {
  $cards = @(Get-CimInstance Win32_VideoController)
  $card = $cards | Where-Object { $_.Name -eq $GpuName } | Select-Object -First 1
  if (-not $card) { $card = $cards | Where-Object { $_.Name -like "*$GpuName*" } | Select-Object -First 1 }
  if ($card) { return $card.DriverVersion }
  return $null
}

# `llama-server --version`'s build line (the one starting `version:`, else the first line), '' when it can't run.
function LlamaBuild {
  try {
    $lines = @(& (Join-Path $Oracle 'build\bin\Release\llama-server.exe') --version 2>&1 | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
  } catch { return '' }
  $line = $lines | Where-Object { $_ -match '^version:' } | Select-Object -First 1
  if (-not $line -and $lines.Count) { $line = $lines[0] }
  return "$line"
}

# One llama_search.py run into $Dir (search.json, best.json) on the GPU named $GpuName. The script pins itself to each
# variant's mask. Returns the record's fields: status 'ran' with `best` (best.json's best: args, affinity, median,
# runs), `device`, `budget_hit` and `variants` (the count), or 'fail' with a `problem` when it wrote no best.json (or
# exited non-zero); `path` is the best.json when one was written.
function MeasureLlamaSearch([string]$Dir, [string]$Driver, [string]$Build) {
  New-Item -ItemType Directory -Force $Dir | Out-Null
  foreach ($n in 'search.json', 'best.json') { if (Test-Path (Join-Path $Dir $n)) { Remove-Item (Join-Path $Dir $n) } }
  & $python -I (Join-Path $repo 'tools\py\llama_search.py') $repo $Model (Join-Path $Oracle 'build\bin\Release') --out $Dir --gpu-name $GpuName --budget-min $LlamaSearchMinutes --driver "$Driver" --llama-build "$Build" | Out-Host
  $exit = $LASTEXITCODE
  $rec = @{ status = 'ran'; budget_minutes = $LlamaSearchMinutes }
  $searchFile = Join-Path $Dir 'search.json'
  $bestFile = Join-Path $Dir 'best.json'
  if (Test-Path $searchFile) {
    $s = Get-Content $searchFile -Raw | ConvertFrom-Json
    $rec.budget_hit = [bool]$s.budget_hit
    $rec.variants = @($s.variants).Count
    if ($s.error) { $rec.error = $s.error }
  }
  if ($exit -or -not (Test-Path $bestFile)) {
    $rec.status = 'fail'
    $rec.problem = $(if ($rec.error) { $rec.error } elseif (Test-Path $searchFile) { "llama_search.py exited $exit and wrote no best.json" } else { "llama_search.py wrote no result (exit $exit); it crashed or was not started, see the console output above" })
    return $rec
  }
  $b = Get-Content $bestFile -Raw | ConvertFrom-Json
  $rec.best = $b.best
  $rec.device = $b.device
  $rec.path = $bestFile
  return $rec
}

# The parsed lines with `event` = $Name in every *.jsonl under a run's log directory, oldest file first. Each file is
# read as the server may still be writing it (shared), line by line: a line that is not complete JSON (the one being
# written) is left out. A directory that doesn't exist holds none.
function LogEvents([string]$LogDir, [string]$Name) {
  if (-not (Test-Path -LiteralPath $LogDir -PathType Container)) { return @() }
  $found = @()
  foreach ($f in Get-ChildItem -LiteralPath $LogDir -Filter *.jsonl -Recurse -File | Sort-Object FullName) {
    $fs = [System.IO.FileStream]::new($f.FullName, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete)
    try {
      $sr = [System.IO.StreamReader]::new($fs, [System.Text.UTF8Encoding]::new($false))
      while ($null -ne ($line = $sr.ReadLine())) {
        if (-not $line.Contains($Name, [System.StringComparison]::Ordinal)) { continue }
        try { $j = $line | ConvertFrom-Json -ErrorAction Stop } catch { continue }
        if ($j.event -eq $Name) { $found += $j }
      }
    } finally { $fs.Dispose() }
  }
  return $found
}

# How many `stall` lines (the watchdog's) the logs under $LogDir hold: the record's stall_lines.
function StallLines([string]$LogDir) { return @(LogEvents $LogDir 'stall').Count }

# Lowercase SHA-256 of a file, the spelling the config hashes use.
function FileSha([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }

# The fields every production-template server record carries: the template, the config as filled and the overrides
# (from the Process Start-ArtemisServer returned; $null without one), and the hash of the artemis.exe the blocks serve.
function ConfigFields($Proc) {
  $cfg = if ($Proc) { $Proc.PSObject.Properties['ArtemisConfig']?.Value } else { $null }
  return @{
    config_template = $cfg.template; template_sha256 = $cfg.template_sha256; config_sha256 = $cfg.sha256; overrides = $cfg.overrides
    artemis_sha256 = $(if (Test-Path -LiteralPath (Join-Path $repo 'target\release\artemis.exe')) { FileSha (Join-Path $repo 'target\release\artemis.exe') } else { $null })
  }
}

# The key scan of a block that made its own key (serve-stall, serve-speed), ServerBlock's pattern: a scan that fails
# Records "<Block> key scan" as a failure, keeps the key file and throws; one that passes deletes the key file and
# Records it. Run it in a finally, after the block's last Record.
function ScanKey([string]$Block, [string]$KeyFile, [string[]]$Paths) {
  $scanOk = $false
  try {
    $scanned = Assert-NoKey -KeyFile $KeyFile -Paths $Paths
    $scanOk = $true
  } catch {
    $why = "$($_.Exception.Message) (the key file is kept at $KeyFile)"
    Record @{ test = "$Block key scan"; input = $Block; status = 'fail'; error = $why }
    $script:failed += "$Block key scan"
    throw $why
  } finally {
    if ($scanOk) {
      try { Remove-Item -LiteralPath $KeyFile -Force -ErrorAction Stop } catch { Write-Warning "could not delete the key file $KeyFile`: $($_.Exception.Message)" }
    } else {
      Write-Warning "the key scan did not pass: the key file stays at $KeyFile (its value is in no log); find the leak and clean it"
    }
  }
  # Outside the try above, so a Record that throws can't be written up as a failed scan with the key kept.
  Record @{ test = "$Block key scan"; input = $Block; status = 'pass'; files_scanned = $scanned }
}

# A leftover key file means the block's last scan failed (a passed scan deletes it): refuse until that run is checked.
function RefuseOldKey([string]$Block, [string[]]$Also = @()) {
  $oldKey = Join-Path $runs "keys\$Block\key.txt"
  if (Test-Path -LiteralPath $oldKey) {
    $files = $(if ($Also.Count) { " and $($Also -join ', ')" } else { '' })
    throw "${Block}: $oldKey exists, so the last run of this block failed its key scan (or died before the scan). Check that run for a leak of the key ($runs\$Block$files, and the evidence file), clean it, then delete the key file and run again"
  }
}

# Moves a non-empty run directory aside (never deletes it), then creates it.
function FreshRunDir([string]$Block) {
  $dir = Join-Path $runs $Block
  if ((Test-Path $dir) -and @(Get-ChildItem -Force $dir).Count) {
    $aside = "$dir.old-$(Get-Date -Format yyyyMMdd-HHmmss)"
    Move-Item $dir $aside
    Write-Host "${Block}: moved the old run directory to $aside"
  }
  New-Item -ItemType Directory -Force $dir | Out-Null
  return $dir
}

# Assert-Preflight, Recorded as 'serve preflight' (a failure Records and rethrows), for a block that isn't ServerBlock.
function BlockPreflight([string]$Block) {
  try {
    $pre = Assert-Preflight -Model $Model
  } catch {
    Record @{ test = 'serve preflight'; input = $Block; status = 'fail'; error = $_.Exception.Message }
    $script:failed += "serve preflight $Block"
    throw
  }
  Record @{ test = 'serve preflight'; input = $Block; status = 'pass'; free_ram_gib = $pre.free_ram_gib; min_free_gib = $pre.min_free_gib; model_gib = $pre.model_gib; adapter = $pre.adapter; driver = $pre.driver }
}

# err / bound, with a zero bound (an exact tap) ranking 0 when met and first when missed.
function Ratio($row) {
  if ($row.bound -gt 0) { return $row.err / $row.bound }
  if ($row.err -eq 0) { return 0 }
  return [double]::PositiveInfinity
}

# Records a test-2-style verdict file and notes a failing one.
function RecordVerdict([string]$test, [string]$name, [string]$path) {
  if (-not (Test-Path $path)) {
    Record @{ test = $test; input = $name; status = 'fail'; error = 'no verdict was written' }
    $script:failed += "$test $name"
    return
  }
  $v = Get-Content $path -Raw | ConvertFrom-Json
  $worst = $v.rows | Sort-Object { Ratio $_ } -Descending | Select-Object -First 1
  $other = $v.rows | Where-Object { $_.tap -notlike '*attn.pregate' } | Sort-Object { Ratio $_ } -Descending | Select-Object -First 1
  $pre = $v.rows | Where-Object { $_.tap -like '*attn.pregate' } | Sort-Object { $_.err } -Descending | Select-Object -First 1
  Record @{
    test = $test; input = $name; status = $v.status
    worst_tap = $worst.tap; worst_err = $worst.err; worst_bound = $worst.bound
    worst_other_tap = $other.tap; worst_other_err = $other.err; worst_other_bound = $other.bound
    pregate_tap = $pre.tap; pregate_err = $pre.err; pregate_bound = $pre.bound
    topk_mismatches = @($v.topk_mismatches).Count; excused_ties = @($v.near_ties).Count; missing = @($v.missing).Count
  }
  if ($v.status -ne 'pass') { $script:failed += "$test $name" }
}

# Writes the oracle's tokens from a golden expected output as little-endian u32, for generate --force-tokens.
function OracleTokens([string]$expected, [string]$path) {
  $steps = @((Get-Content $expected -Raw | ConvertFrom-Json).steps)
  $bytes = [byte[]]::new(4 * $steps.Count)
  for ($i = 0; $i -lt $steps.Count; $i++) { [BitConverter]::GetBytes([uint32]$steps[$i].token).CopyTo($bytes, 4 * $i) }
  [IO.File]::WriteAllBytes($path, $bytes)
}

# Records test 3's verdict: a line per prompt with its flips and its run's numbers, then the pooled line, which gates.
function RecordTest3([string]$verdict, [string]$dir, [string[]]$notRun = @()) {
  if (-not (Test-Path $verdict)) {
    $why = 'no verdict was written'
    if ($notRun.Count) { $why += "; generate failed for $($notRun -join ', ')" }
    Record @{ test = '3'; input = '(pooled)'; status = 'fail'; error = $why }
    $script:failed += 'test 3'
    return
  }
  $v = Get-Content $verdict -Raw | ConvertFrom-Json
  foreach ($p in $v.prompts) {
    $g = Get-Content (Join-Path $dir "$($p.id).json") -Raw | ConvertFrom-Json
    $free = $p.free_divergence
    $floorFree = $p.floor_free_divergence
    Record @{
      test = '3'; input = $p.id; status = $p.status; problem = $p.problem
      steps = $p.steps; counted = $p.counted; flips = @($p.flips).Count; floor_flips = @($p.floor_flips).Count
      flip_steps = @($p.flips | ForEach-Object { $_.step })
      free_diverged_at = $(if ($free) { $free.step } else { $null })
      free_oracle_gap = $(if ($free) { $free.oracle_gap } else { $null })
      free_excused = $(if ($free) { $free.excused } else { $null })
      floor_free_diverged_at = $(if ($floorFree) { $floorFree.step } else { $null })
      prompt_tokens = @($g.prompt_tokens).Count; stop = $g.stop
      tok_s = $(if ($g.timings.generate_s -gt 0) { $g.timings.decoded / $g.timings.generate_s } else { $null })
    }
    if ($p.status -ne 'pass') { $script:failed += "test 3 $($p.id)" }
  }
  # A prompt whose generate failed is missing from the pooled flips, so the pooled result can't pass.
  $problems = @($v.problems)
  $status = $v.status
  if ($notRun.Count) {
    $problems += "generate failed for: $($notRun -join ', '); they are not in the pooled flips"
    $status = 'fail'
  }
  Record @{
    test = '3'; input = '(pooled)'; status = $status; prompts = @($v.prompts).Count; counted = $v.counted
    flips = $v.flips; floor_flips = $v.floor_flips; bound = $v.bound; problems = $problems
  }
  if ($status -ne 'pass') { $script:failed += 'test 3' }
}

# Records one test-5 verdict, with each run's cache numbers, and notes a failing one.
function RecordTest5([string]$id, [string]$verdict) {
  if (-not (Test-Path $verdict)) {
    Record @{ test = '5'; input = $id; status = 'fail'; error = 'no verdict was written' }
    $script:failed += "test 5 $id"
    return
  }
  $v = Get-Content $verdict -Raw | ConvertFrom-Json
  $each = @($v.runs | ForEach-Object {
      $routed = $_.hits + $_.staged + $_.misses
      @{ label = $_.label; slots = $_.cache_slots; policy = $_.cache_policy; positions = $_.positions; steps = $_.steps; swaps = $_.swaps_started
         hit_rate = $(if ($routed) { [math]::Round($_.hits / $routed, 4) } else { 0 })
         from_vram = $(if ($routed) { [math]::Round(($_.hits + $_.staged) / $routed, 4) } else { 0 }) }
    })
  # The report runs (the sixth, speculation-off run): compared with the gating runs and reported, never part of the status.
  $reports = @($v.reports | Where-Object { $null -ne $_ } | ForEach-Object {
      @{ label = $_.label; identical_tokens = $_.identical_tokens; identical_logits = $_.identical_logits }
    })
  Record @{ test = '5'; input = $id; status = $v.status; runs = $each; first_difference = $v.first_difference; problems = @($v.problems); reports = $reports }
  if ($v.status -ne 'pass') { $script:failed += "test 5 $id" }
}

# Records test 6's verdict: a line per prompt with its flips and bit comparisons (reported), then the pooled line, which
# gates (each path's flips, the bit check).
function RecordTest6([string]$verdict, [string[]]$notRun = @()) {
  if (-not (Test-Path $verdict)) {
    $why = 'no verdict was written'
    if ($notRun.Count) { $why += "; a run failed for $($notRun -join ', ')" }
    Record @{ test = '6'; input = '(pooled)'; status = 'fail'; error = $why }
    $script:failed += 'test 6'
    return
  }
  $v = Get-Content $verdict -Raw | ConvertFrom-Json
  foreach ($p in $v.per_prompt) {
    Record @{
      test = '6'; input = $p.id; status = 'ran'
      cold_flips = $p.cold_flips; path1_flips = $p.path1_flips; path2_flips = $p.path2_flips
      path1_prefill_bits = $p.path1_prefill_bits; path1_decode_bits = $p.path1_decode_bits; path2_decode_bits = $p.path2_decode_bits
    }
  }
  # A prompt whose run failed is missing from a path's pooled flips (the verdict names it when it ran cold), so the
  # pooled result can't pass.
  $problems = @($v.problems)
  $status = $v.status
  if ($notRun.Count) {
    $problems += "a run failed for: $($notRun -join ', '); they are not in the pooled flips"
    $status = 'fail'
  }
  Record @{
    test = '6'; input = '(pooled)'; status = $status; bound = $v.bound
    cold_flips = $v.cold.flips; path1_flips = $v.path1.flips; path2_flips = $v.path2.flips; m2_cold_flips = $v.m2_cold_flips
    bits_equal = $v.bits.equal; first_difference = $v.bits.first_difference; problems = $problems
  }
  if ($status -ne 'pass') { $script:failed += 'test 6' }
}

# The golden prompts to run: all of them, or those -Prompts names.
function GoldenIds([string[]]$all) {
  if ($Prompts.Count) { return @($all | Where-Object { $Prompts -contains $_ }) }
  return $all
}

# One server block (tools\ps\ArtemisServer.psm1 does the starting and stopping). Its run directory is
# $runs\<Block>\ (a non-empty old one is renamed aside, never deleted): the server's artemis.toml, logs\ and stderr.txt,
# the verdicts\ of `artemis test7 --case <Cases>`, and the reports. Its key is under $runs\keys\<Block>\, outside every
# directory the key scan reads. The server starts in a try and stops in the finally, which then scans the run directory
# for the key. One Record line per case verdict (test = '7'), one 'serve stop' line, and with -ReuseReport the
# reuse-report over the block's logs (report only). -AfterCases ({ param($RunDir, $KeyFile) ... }) runs against the
# server after the cases, before the stop. A case whose pool check says "inconclusive: pool below 2 GiB" is a
# stop-and-report: it is recorded with inconclusive = $true, warned about, counted as failing, and the block throws
# after its cleanup so no later block runs and nothing reruns.
# -Config names a config template (the production one): the server then starts from it filled in (the flags still
# apply). Every `7` and `serve stop` line carries config_template, template_sha256, config_sha256 and overrides (what
# Start-ArtemisServer reports; the template fields are $null without -Config), artemis_sha256 (the binary served) and
# stall_lines (the `stall` lines in the block's logs when the line was written). A `stall` line in the logs fails the block.
$script:serveBuilt = $false
function ServerBlock {
  param(
    [Parameter(Mandatory)][string]$Block,
    [Parameter(Mandatory)][string[]]$Cases,
    [string[]]$Extra = @(),
    [switch]$ReuseReport,
    [scriptblock]$AfterCases,
    [string]$Config
  )
  # A key file left by an earlier run means its scan failed (a passed scan deletes it): that run's leak must be found and
  # cleaned first. Starting again would overwrite the key and rename that run's directory aside, out of the next scan.
  $oldKey = Join-Path $runs "keys\$Block\key.txt"
  if (Test-Path -LiteralPath $oldKey) {
    throw "${Block}: $oldKey exists, so the last run of this block failed its key scan (or died before the scan). Check that run for a leak of the key ($runs\$Block and the evidence file), clean it, then delete the key file and run again"
  }
  $dir = Join-Path $runs $Block
  if ((Test-Path $dir) -and @(Get-ChildItem -Force $dir).Count) {
    $aside = "$dir.old-$(Get-Date -Format yyyyMMdd-HHmmss)"
    Move-Item $dir $aside
    Write-Host "${Block}: moved the old run directory to $aside"
  }
  New-Item -ItemType Directory -Force $dir | Out-Null
  $logs = Join-Path $dir 'logs'
  $verdicts = Join-Path $dir 'verdicts'

  try {
    $pre = Assert-Preflight -Model $Model
  } catch {
    Record @{ test = 'serve preflight'; input = $Block; status = 'fail'; error = $_.Exception.Message }
    $script:failed += "serve preflight $Block"
    throw
  }
  Record @{ test = 'serve preflight'; input = $Block; status = 'pass'; free_ram_gib = $pre.free_ram_gib; min_free_gib = $pre.min_free_gib; model_gib = $pre.model_gib; adapter = $pre.adapter; driver = $pre.driver }

  # The 2-gpu block's `cargo test --release -p artemis-tools` rebuilds artemis.exe with the dev-dependencies' features
  # (engine/fake among them): the first server block builds it again, so both gates serve the same binary.
  if (-not $script:serveBuilt) {
    cargo build --release -p artemis-tools; Check 'cargo build (server blocks)'
    $script:serveBuilt = $true
  }

  $keyFile = New-ArtemisKey -Dir (Join-Path $runs "keys\$Block")
  $inconclusive = @()
  $proc = $null
  $meta = ConfigFields $null
  try {
    $start = @{ Exe = $artemis; Model = $Model; Name = $Block; RunDir = $dir; KeyFile = $keyFile; Extra = $Extra }
    if ($Config) { $start.Config = $Config }
    $proc = Start-ArtemisServer @start
    $meta = ConfigFields $proc
    New-Item -ItemType Directory -Force $verdicts | Out-Null
    $t7out = Join-Path $dir 'test7.txt'
    $t7err = Join-Path $dir 'test7.err'
    & $artemis test7 --base-url 'http://127.0.0.1:8090' --key-file $keyFile --log-dir $logs --case ($Cases -join ',') --out $verdicts --prompts-root $repo 2> $t7err > $t7out
    if ($LASTEXITCODE) {
      # A harness error (2): the cases after the failing one did not run, and their verdicts are missing below.
      Record (@{ test = '7'; input = $Block; status = 'fail'; error = "artemis test7 failed (exit $LASTEXITCODE); see $t7err"; stall_lines = (StallLines $logs) } + $meta)
      $script:failed += "7 $Block"
    }
    foreach ($case in $Cases) {
      $path = Join-Path $verdicts "$case.json"
      if (-not (Test-Path $path)) {
        Record (@{ test = '7'; input = $case; status = 'fail'; error = 'no verdict was written'; stall_lines = (StallLines $logs) } + $meta)
        $script:failed += "7 $case"
        continue
      }
      $v = Get-Content $path -Raw | ConvertFrom-Json
      $line = @{ test = '7'; input = $case; status = $v.status; checks = @($v.checks); figures = $v.figures; stall_lines = (StallLines $logs) } + $meta
      $pool = @($v.checks | Where-Object { $_.name -eq 'pool_at_least_2_gib' -and -not $_.pass -and $_.detail -eq 'inconclusive: pool below 2 GiB' })
      if ($pool.Count) {
        $line.inconclusive = $true
        $inconclusive += $case
        Write-Warning "INCONCLUSIVE: test 7 case '$case' ran with a snapshot pool below 2 GiB. This is not a failure of the engine. It needs review; do not rerun."
      }
      Record $line
      if ($v.status -notin @('pass', 'ran')) { $script:failed += "7 $case" }
    }
    if ($AfterCases) { & $AfterCases $dir $keyFile }
  } catch {
    Record (@{ test = '7'; input = $Block; status = 'fail'; error = $_.Exception.Message } + $meta)
    $script:failed += "7 $Block"
    throw
  } finally {
    # The key scan runs whatever the stop or its Record does. It covers the run dir and the evidence file, which has
    # had this block's Record lines by now (the key is fresh, so older lines can't hold it). The key file is deleted
    # only after a scan that passed; after a failed one it stays, so a leak can be found and cleaned.
    # If a pipeline stop landed as Start-ArtemisServer returned, $proc was never assigned: ask the module for the child.
    if (-not $proc) { $proc = Get-ArtemisServerProcess }
    try {
      if ($proc) {
        try {
          $stop = Stop-ArtemisServer -Proc $proc -RunDir $dir
        } catch {
          # Recorded here: the key scan's own failure, below, can replace this error.
          Record (@{ test = 'serve stop'; input = $Block; graceful = $false; error = $_.Exception.Message } + $meta)
          $script:failed += "serve stop $Block"
          throw
        }
        # The server has stopped: the logs are complete, so this count is the block's.
        $stalls = StallLines $logs
        Record (@{ test = 'serve stop'; input = $Block; graceful = $stop.graceful; exit_code = $stop.exit_code; seconds = $stop.seconds; exit_line = $stop.exit_line; port_free = $stop.port_free; stall_lines = $stalls } + $meta)
        if (-not $stop.graceful -or -not $stop.port_free) { $script:failed += "serve stop $Block" }
        if ($stalls) { $script:failed += "stall $Block" }
      }
    } finally {
      $scanOk = $false
      try {
        $scanned = Assert-NoKey -KeyFile $keyFile -Paths $dir, $evidence
        $scanOk = $true
      } catch {
        $why = "$($_.Exception.Message) (the key file is kept at $keyFile)"
        Record @{ test = 'serve key scan'; input = $Block; status = 'fail'; error = $why }
        $script:failed += "serve key scan $Block"
        throw $why
      } finally {
        if ($scanOk) {
          try { Remove-Item -LiteralPath $keyFile -Force -ErrorAction Stop } catch { Write-Warning "could not delete the key file $keyFile`: $($_.Exception.Message)" }
        } else {
          Write-Warning "the key scan did not pass: the key file stays at $keyFile (its value is in no log); find the leak and clean it"
        }
      }
      # Outside the try above, so a Record that throws can't be written up as a failed scan with the key kept.
      Record @{ test = 'serve key scan'; input = $Block; status = 'pass'; files_scanned = $scanned }
    }
  }

  if ($ReuseReport) {
    $report = Join-Path $dir 'reuse-report.json'
    $rrOut = Join-Path $dir 'reuse-report.txt'
    $rrErr = Join-Path $dir 'reuse-report.err'
    & $artemis reuse-report --log-dir $logs --out $report 2> $rrErr > $rrOut
    if ($LASTEXITCODE) {
      Write-Warning "artemis reuse-report failed (exit $LASTEXITCODE); report only, so the gate is unchanged"
      Record (@{ test = 'reuse-report (report only)'; input = $Block; status = 'error'; error = "exit $LASTEXITCODE" } + $meta)
    } else {
      $r = Get-Content $report -Raw | ConvertFrom-Json
      Record (@{ test = 'reuse-report (report only)'; input = $Block; status = 'ran'; overall = $r.overall; by_source = $r.by_source; by_cause = $r.by_cause; budgets = $r.budgets } + $meta)
    }
  }
  if ($inconclusive.Count) {
    throw "NEEDS REVIEW: test 7 case(s) $($inconclusive -join ', ') ran with a snapshot pool below 2 GiB, so their reuse verdicts are inconclusive. Have a maintainer review it; do not rerun. Failing so far: $($script:failed -join ', ')"
  }
}

# Records one client's smoke result (the smoke block, below). A key file still at $KeyFile means the script's key scan
# failed or never ran, or the key could not be deleted after a passed one (a passed scan deletes it), so the verdict's text may hold what the scan was to find: only a minimal
# failing line goes into the evidence file (no reasons, attempts or server), the client is counted as failing, and it throws
# so nothing runs after it, as the server blocks do. Otherwise one `smoke` line from verdict.json (or a failure line when
# the script wrote none). The line also carries the config the server started from (verdict.json's `config`: template,
# template_sha256, sha256 as config_sha256, overrides; $null for a smoke run without -Config), artemis_sha256 and
# stall_lines, the `stall` lines in the server's logs; any of them fails the client.
function RecordSmoke([string]$Client, [string]$Root, [string]$KeyFile, [int]$Exit, [string]$ScriptOut, [string]$Block = 'smoke') {
  if (Test-Path -LiteralPath $KeyFile) {
    Record @{ test = 'smoke'; input = $Client; block = $Block; status = 'fail'; exit_code = $Exit; error = "the key file is still present (the scan failed, did not run, or the key could not be deleted): $KeyFile; see $Root" }
    $script:failed += "$Block $Client"
    throw "$Block ${Client}: the key file is still present at $KeyFile (the scan failed, did not run, or the key could not be deleted). Check $Root for a leak of the key, clean it, delete the key file and run again"
  }
  $verdictPath = Join-Path $Root 'verdict.json'
  if (Test-Path -LiteralPath $verdictPath) {
    $v = Get-Content -LiteralPath $verdictPath -Raw | ConvertFrom-Json
    $stalls = StallLines (Join-Path $Root 'server\logs')
    $art = (ConfigFields $null).artemis_sha256
    Record @{ test = 'smoke'; input = $Client; block = $Block; status = $v.status; exit_code = $Exit; version = $v.version; attempts = @($v.attempts)
      wire = $v.wire; reasons = @($v.reasons); server = $v.server; root = $Root
      config_template = $v.config.template; template_sha256 = $v.config.template_sha256; config_sha256 = $v.config.sha256; overrides = $v.config.overrides
      artemis_sha256 = $art; stall_lines = $stalls }
    if ($Exit -ne 0 -or $v.status -notin @('pass', 'model-result') -or $stalls) { $script:failed += "$Block $Client" }
  } else {
    # No verdict: the script refused or died before it wrote one (a missing canary marker, a busy port, preflight).
    Record @{ test = 'smoke'; input = $Client; block = $Block; status = 'fail'; exit_code = $Exit; error = "no verdict was written; see $ScriptOut" }
    $script:failed += "$Block $Client"
  }
}

# --- The speed blocks' shared helpers ---------------------------------------------------------------------------------
# FreshServer, SpeedOk, SpeedWhy and Median were serve-speed's own; they are shared here, unchanged in what
# they do, so serve-speed-spec and serve-shrunk start their servers the same way. FreshServer reads the calling
# block's $block, $speedRunDir, $keyFile and $speedExtra, and $artemis, $template, $Model and $repo.

# FNV-1a 64 of $Bytes as 16 lowercase hex digits, the spelling of qwen36::testing::fnv1a64 and of the completion_fnv
# figures. Done in 32-bit halves (offset basis 0xcbf29ce484222325, prime 2^40 + 0x1b3 = 2^40 + 435): PowerShell's integer
# multiply throws on overflow.
function Fnv1a64Hex([byte[]]$Bytes) {
  $mask = [long]4294967295
  $hi = [long]3421674724
  $lo = [long]2216829733
  foreach ($b in $Bytes) {
    $lo = $lo -bxor [long]$b
    $a = $lo * 435
    $newHi = ($hi * 435 + ($a -shr 32) + (($lo -band 16777215) -shl 8)) -band $mask
    $lo = $a -band $mask
    $hi = $newHi
  }
  return ('{0:x8}{1:x8}' -f $hi, $lo)
}

# One process: the desktop's GPU memory with none running, the server from the template (with $Extra on top of it, or
# $speedExtra when none is given), `artemis test7 --case $Case $CaseArgs`, the stop. Returns what it found; the Records
# of the stop are written here.
function FreshServer([string]$Leaf, [string]$Case, [string[]]$CaseArgs, [string[]]$Extra = $null) {
  $serverExtra = $(if ($null -ne $Extra) { $Extra } else { $speedExtra })
  $rdir = Join-Path $speedRunDir $Leaf
  New-Item -ItemType Directory -Force $rdir | Out-Null
  $logs = Join-Path $rdir 'logs'
  $verdicts = Join-Path $rdir 'verdicts'
  if (@(Get-Process -Name artemis -ErrorAction SilentlyContinue).Count) {
    throw "${block}: an artemis process is running before $Leaf starts; the desktop's GPU memory is sampled with none"
  }
  $r = @{ leaf = $Leaf; mib = (DesktopGpuMib); test7_exit = $null; verdict = $null; stop = $null; stall_lines = 0; slots_per_layer = $null; meta = (ConfigFields $null); affinity = $null; affinity_at_stop = $null; affinity_problem = $null }
  $proc = $null
  try {
    $proc = Start-ArtemisServer -Exe $artemis -Model $Model -Name "$block-$Leaf" -RunDir $rdir -KeyFile $keyFile -Config $template -Extra $serverExtra
    $r.meta = ConfigFields $proc
    $r.affinity = ProcAffinity $proc
    New-Item -ItemType Directory -Force $verdicts | Out-Null
    & $artemis test7 --base-url 'http://127.0.0.1:8090' --key-file $keyFile --log-dir $logs --case $Case @CaseArgs --out $verdicts --prompts-root $repo 2> (Join-Path $rdir 'test7.err') > (Join-Path $rdir 'test7.txt')
    $r.test7_exit = $LASTEXITCODE
    $verdictPath = Join-Path $verdicts "$Case.json"
    if (Test-Path -LiteralPath $verdictPath) { $r.verdict = Get-Content -LiteralPath $verdictPath -Raw | ConvertFrom-Json }
  } finally {
    if (-not $proc) { $proc = Get-ArtemisServerProcess }
    if ($proc) {
      # The mask again, just before the stop: a child that Process Lasso re-pinned mid-run shows here.
      $r.affinity_at_stop = ProcAffinity $proc
      $bad = @(@($r.affinity, $r.affinity_at_stop) | Where-Object { $_ -ne $runnerMask })
      if ($bad.Count) { $r.affinity_problem = "the server's affinity was $(MaskText $r.affinity) after it started and $(MaskText $r.affinity_at_stop) before it stopped, not the runner's $(MaskText $runnerMask)" }
      try {
        $r.stop = Stop-ArtemisServer -Proc $proc -RunDir $rdir
      } catch {
        Record (@{ test = 'serve stop'; input = "$block $Leaf"; graceful = $false; error = $_.Exception.Message } + $r.meta)
        $script:failed += "serve stop $block $Leaf"
        throw
      }
      $r.stall_lines = StallLines $logs
      Record (@{ test = 'serve stop'; input = "$block $Leaf"; graceful = $r.stop.graceful; exit_code = $r.stop.exit_code; seconds = $r.stop.seconds; exit_line = $r.stop.exit_line; port_free = $r.stop.port_free; stall_lines = $r.stall_lines } + $r.meta)
      if (-not $r.stop.graceful -or -not $r.stop.port_free) { $script:failed += "serve stop $block $Leaf" }
    }
  }
  $loads = @(LogEvents $logs 'load')
  $r.slots_per_layer = $(if ($loads.Count) { $loads[-1].slots_per_layer } else { $null })
  return $r
}
# A process is usable when its test 7 exited 0 and wrote its verdict, and the server stopped gracefully.
function SpeedOk($r) { return -not (SpeedWhy $r) }
# What made a process unusable, in words, or '' when none did: the first term of SpeedOk's rule that failed.
function SpeedWhy($r) {
  if ($r.test7_exit -ne 0) { return "artemis test7 exited $($r.test7_exit)" }
  if ($null -eq $r.verdict) { return 'test 7 wrote no verdict' }
  if (-not $r.stop.graceful) { return "the server did not stop gracefully (exit code $($r.stop.exit_code), exit line $($r.stop.exit_line))" }
  if (-not $r.stop.port_free) { return 'port 8090 was still busy after the server stopped' }
  if ($r.affinity_problem) { return $r.affinity_problem }
  return ''
}
function Median([double[]]$x) {
  $s = @($x | Sort-Object)
  if (-not $s.Count) { return $null }
  if ($s.Count % 2) { return $s[($s.Count - 1) / 2] }
  return ($s[$s.Count / 2 - 1] + $s[$s.Count / 2]) / 2
}

# --- The first-run block's helpers ------------------------------------------------------------------------------------
# Pure where they can be (rows and lines in, a record out), so tools\ps\Test-Hygiene.ps1 checks them without a GPU.

# The setup limit the first-run block holds setup to. It mirrors setup's SETUP_LIMIT_S
# (crates/tools/src/setup/choose.rs), as m6b_summary.SETUP_LIMIT_S does; change them together. The wait for /health is
# the limit and two minutes for the model's checks and the port.
$script:SetupLimitS = 600
$script:LaunchWaitS = $script:SetupLimitS + 120

# The `tuned` lines of one process's log, judged against the tuned file the block pointed ARTEMIS_TUNED at: the last
# line that names that file (paths compare in full form, case ignored) and the keys it applied and ignored. The file is
# applied when that line lists at least one applied key. A process whose log holds no such line applied nothing.
function TunedApplied($Events, [string]$Path) {
  $same = {
    param($a, $b)
    try { return [System.IO.Path]::GetFullPath([string]$a).Equals([System.IO.Path]::GetFullPath([string]$b), [System.StringComparison]::OrdinalIgnoreCase) }
    catch { return ([string]$a).Equals([string]$b, [System.StringComparison]::OrdinalIgnoreCase) }
  }
  $all = @($Events | Where-Object { $_ })
  $mine = @($all | Where-Object { $_.path -and (& $same $_.path $Path) })
  $applied = @()
  $ignored = @()
  if ($mine.Count) {
    $applied = @($mine[-1].applied | Where-Object { $_ })
    $ignored = @($mine[-1].ignored | Where-Object { $_ })
  }
  return @{ tuned_applied = ($applied.Count -gt 0); tuned_applied_keys = $applied; tuned_ignored_keys = $ignored; tuned_lines = $all.Count }
}

# The WinHTTP and WinINet entries of a list of imported DLL names.
function NetworkImports($Imports) {
  return @(@($Imports) | Where-Object { $_ -match 'winhttp|wininet' })
}

# Whether a setup_done line shows setup choosing the shipped defaults: speculation on, the sources lookup then mtp and
# no measured priors. Only then are the tuned runs expected to repeat the "on" runs' completions (and then only as a
# report). $null (no line) is not the defaults.
function ChoiceIsDefault($Setup) {
  if ($null -eq $Setup) { return $false }
  return ($Setup.enabled -eq $true) -and (@($Setup.sources) -join ',') -ceq 'lookup,mtp' -and ($null -eq $Setup.priors)
}

# The name of a llama.cpp device record ({id, name}) or a plain name; $null when there is none.
function LlamaDeviceName($Dev) {
  if ($null -eq $Dev) { return $null }
  if ($Dev -is [string]) { return $Dev }
  $n = $Dev.name
  if ($n) { return [string]$n }
  return $null
}

# Whether $Path is a searched llama.cpp result (best, finalists, driver and llama_build present) taken with the given
# GPU driver and llama.cpp build. An empty driver or build never matches (today's could not be read).
function SearchedConfigCurrent([string]$Path, [string]$Driver, [string]$Build) {
  if (-not $Driver -or -not $Build -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
  try { $j = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -ErrorAction Stop } catch { return $false }
  $names = @($j.PSObject.Properties.Name)
  if (@('best', 'finalists', 'driver', 'llama_build' | Where-Object { $names -notcontains $_ }).Count) { return $false }
  return ([string]$j.driver -ceq $Driver) -and ([string]$j.llama_build -ceq $Build)
}

# Why the first-run block cannot be judged before it starts, or $null: a missing reference (serve-speed-spec's "on"
# median, speed-llama's figure), or a llama.cpp figure that is not a searched best (`search` or `file`) on the device
# $Gpu names. Such a failure is recorded as `first-run precheck`, which is not an attempt.
function FirstRunPrecheck($SpecOnMedian, $LlamaBest, [string]$Gpu) {
  if ($null -eq $SpecOnMedian -or $null -eq $LlamaBest) { return 'run serve-speed-spec and speed-llama in the same invocation' }
  if ($LlamaBest.config_source -cne 'search' -and $LlamaBest.config_source -cne 'file') {
    return "llama.cpp ran its `"$($LlamaBest.config_source)`" configuration, not a searched best: run llama-search in this invocation or pass -LlamaConfig with a searched best.json"
  }
  $name = LlamaDeviceName $LlamaBest.device
  if (-not ($name -and $Gpu -and $name.IndexOf($Gpu, [System.StringComparison]::OrdinalIgnoreCase) -ge 0)) {
    return "llama.cpp's device is `"$name`", not the one -GpuName names (`"$Gpu`")"
  }
  return $null
}

# The first-run aggregate from what the block found. $Launch: health (the launch reached /health), reason, setup (the
# setup_done line or $null), network_imports and affinity_problem. $Rows: one per decode run, with leaf (write-1234
# ...), prompt, decode (decode_steps_tok_s), fnv (completion_fnv), ok (the process was whole and on the runner's mask)
# and tuned_applied. $SpecOnMedian and $SpecOnFnv come from serve-speed-spec, $LlamaBest from speed-llama (median,
# device, affinity, config_source). $RunProblems are the block's own (a stop, a stall, the launch's stop); a row that is
# not whole or did not apply the file is named here besides, unless a run problem already names its leaf. The link is
# not judged here: WithLink fails an aggregate whose link fell.
function FirstRunJudge($Launch, $Rows, [int]$Attempt, $SpecOnMedian, $SpecOnFnv, $LlamaBest, [string]$Gpu, [string[]]$RunProblems = @(), [string]$Plan = '5b') {
  $problems = @($RunProblems | Where-Object { $_ })
  $setup = $Launch.setup
  $healthy = [bool]$Launch.health
  if (-not $healthy) {
    $problems += "the launch did not reach /health$(if ($Launch.reason) { ": $($Launch.reason)" })"
  } elseif ($null -eq $setup) {
    $problems += 'the launch reached /health but its log holds no setup_done line, so setup did not measure this PC'
  }
  $setupOk = $false
  if ($null -ne $setup) {
    if (-not (IsNum $setup.seconds)) { $problems += 'the setup_done line has no seconds' }
    elseif ($setup.seconds -gt $script:SetupLimitS) { $problems += "setup took $($setup.seconds) s, over its limit of $($script:SetupLimitS) s" }
    elseif ($setup.limit_hit) { $problems += 'setup hit its time limit and stopped measuring early' }
    else { $setupOk = $true }
  }
  $net = @($Launch.network_imports)
  if ($net.Count) { $problems += "artemis.exe imports $($net -join ', '), a network client" }
  if ($Launch.affinity_problem) { $problems += "the launch: $($Launch.affinity_problem)" }

  $wanted = @('write-1234', 'edit-1234', 'explain-1234', 'write-1235', 'edit-1235', 'explain-1235')
  $rowOf = @{}
  foreach ($row in @($Rows)) { $rowOf[$row.leaf] = $row }
  $whole = $true
  foreach ($leaf in $wanted) {
    $row = $rowOf[$leaf]
    $named = @($RunProblems | Where-Object { $_ -like "$leaf*" }).Count -gt 0
    if ($null -eq $row) { $whole = $false; if (-not $named) { $problems += "${leaf}: no tuned run was made" }; continue }
    if (-not $row.ok) { $whole = $false; if (-not $named) { $problems += "${leaf}: the run was not whole or not on the runner's mask (its record says why)" } }
    if (-not $row.tuned_applied) { $whole = $false; $problems += "${leaf}: its tuned line did not apply the tuned file" }
    if (-not (IsNum $row.decode)) { $whole = $false; $problems += "${leaf}: no decode figure" }
  }
  $six = @($wanted | Where-Object { $rowOf.ContainsKey($_) -and (IsNum $rowOf[$_].decode) } | ForEach-Object { [double]$rowOf[$_].decode })
  $tuned = $(if ($six.Count -eq 6) { Median $six } else { $null })

  $llamaMedian = $(if ($LlamaBest -and (IsNum $LlamaBest.median)) { [double]$LlamaBest.median } else { $null })
  $onMedian = $(if (IsNum $SpecOnMedian) { [double]$SpecOnMedian } else { $null })
  if ($null -eq $onMedian) { $problems += 'serve-speed-spec gave no "on" median of six to compare with' }
  if ($null -eq $llamaMedian) { $problems += 'speed-llama gave no median to compare with' }
  $hand = SpecRatio $tuned $onMedian
  $vsLlama = SpecRatio $tuned $llamaMedian
  # The bar is a product, tuned >= 0.97 x on, with the 1e-9 allowance mtp_pays has (a ratio form can miss by rounding).
  $handOk = ($null -ne $tuned -and $null -ne $onMedian -and $tuned -ge (0.97 * $onMedian - 1e-9))
  $llamaOk = ($null -ne $vsLlama -and $vsLlama -gt 1.0)
  if ($null -ne $tuned -and $null -ne $onMedian -and -not $handOk) { $problems += "the tuned median $tuned is $([math]::Round($hand, 4)) of the hand-tuned `"on`" median $onMedian, below 0.97" }
  if ($null -ne $tuned -and $null -ne $llamaMedian -and -not $llamaOk) { $problems += "the tuned median $tuned is $([math]::Round($vsLlama, 4)) of llama.cpp's median $llamaMedian, not above 1" }
  if ($null -eq $tuned) { $problems += 'the tuned runs do not give six decode figures' }

  $source = $(if ($LlamaBest) { $LlamaBest.config_source } else { $null })
  $deviceName = $(if ($LlamaBest) { LlamaDeviceName $LlamaBest.device } else { $null })
  $searched = ($source -ceq 'search' -or $source -ceq 'file')
  if ($LlamaBest -and -not $searched) { $problems += "llama.cpp ran its `"$source`" configuration, not a searched best (llama-search or -LlamaConfig)" }
  $onDevice = [bool]($deviceName -and $Gpu -and $deviceName.IndexOf($Gpu, [System.StringComparison]::OrdinalIgnoreCase) -ge 0)
  if ($LlamaBest -and -not $onDevice) { $problems += "llama.cpp's device is `"$deviceName`", not the one named by -GpuName (`"$Gpu`")" }

  $default = ChoiceIsDefault $setup
  $identity = $null
  if ($default -and $SpecOnFnv) {
    $identity = @(foreach ($leaf in $wanted) {
        $mine = $(if ($rowOf.ContainsKey($leaf)) { $rowOf[$leaf].fnv } else { $null })
        $on = $SpecOnFnv[$leaf]
        @{ run = $leaf; equal = $(if ($mine -and $on) { $mine -ceq $on } else { $null }) }
      })
  }
  $checks = @{ launch = ($healthy -and $null -ne $setup); setup = $setupOk; no_network_import = ($net.Count -eq 0); runs = $whole
    hand = $handOk; llama = $llamaOk; searched = ($searched -and $onDevice) }
  $pass = $true
  foreach ($k in $checks.Keys) { if (-not $checks[$k]) { $pass = $false } }
  if ($problems.Count) { $pass = $false }
  return @{
    test = 'first-run'; plan = $Plan; status = $(if ($pass) { 'pass' } else { 'fail' }); attempt = $Attempt
    tuned_median = $tuned; spec_on_median = $onMedian; tuned_over_hand = $hand; llama_median = $llamaMedian; tuned_over_llama = $vsLlama
    llama_device = $deviceName; llama_affinity = $(if ($LlamaBest) { $LlamaBest.affinity } else { $null }); llama_config_source = $source
    choice_is_default = $default; identity = $identity; checks = $checks
    runs = @($wanted | Where-Object { $rowOf.ContainsKey($_) } | ForEach-Object { @{ run = $_; mode = 'tuned'; decode_steps_tok_s = $rowOf[$_].decode } })
    problems = $problems
  }
}

# /health on 127.0.0.1:$Port as the status code, or $null when nothing answers.
function HealthStatus([int]$Port) {
  if (-not $script:HealthHttp) {
    $script:HealthHttp = [System.Net.Http.HttpClient]::new()
    $script:HealthHttp.Timeout = [TimeSpan]::FromSeconds(5)
  }
  try {
    $r = $script:HealthHttp.GetAsync("http://127.0.0.1:$Port/health").GetAwaiter().GetResult()
    try { return [int]$r.StatusCode } finally { $r.Dispose() }
  } catch { return $null }
}

# The first-run block's first launch: starts $Exe with $ArgList (the argument list Start-ArtemisServer -WhatIf gave) as
# a child with stdin, stdout and stderr redirected (stdout and stderr into $Dir\launch-stdout.txt and
# launch-stderr.txt), polls /health on $Port until it answers 200, the child exits or $LimitS seconds pass, reading the
# child's CPU mask right after the start and on every poll, then once more just before the stop, and stops it with
# Stop-ArtemisServer (closing its stdin, as every server block does). Returns @{ health; first_launch_s; reason;
# affinity; affinity_at_stop; affinity_problem; stop; stdout; stderr; exit_code }: first_launch_s is the time from the
# start to the first 200; reason says why /health was not reached; stop is Stop-ArtemisServer's result. A server that is
# still running when this function leaves (an error, a stopped pipeline) is stopped in the finally.
function FirstRunLaunch([string]$Exe, [string[]]$ArgList, [string]$Dir, [int]$Port, [double]$LimitS) {
  New-Item -ItemType Directory -Force $Dir | Out-Null
  $r = @{ health = $false; first_launch_s = $null; reason = $null; affinity = $null; affinity_at_stop = $null; affinity_problem = $null
    stop = $null; exit_code = $null; stdout = (Join-Path $Dir 'launch-stdout.txt'); stderr = (Join-Path $Dir 'launch-stderr.txt') }
  $psi = [System.Diagnostics.ProcessStartInfo]::new($Exe)
  foreach ($a in $ArgList) { $psi.ArgumentList.Add($a) }
  $psi.UseShellExecute = $false
  $psi.RedirectStandardInput = $true
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $so = [System.IO.FileStream]::new($r.stdout, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite, 1)
  $se = [System.IO.FileStream]::new($r.stderr, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite, 1)
  $proc = [System.Diagnostics.Process]::new()
  $proc.StartInfo = $psi
  $started = $false
  $copies = @()
  $exitedEarly = $false
  $clock = [System.Diagnostics.Stopwatch]::StartNew()
  try {
    $null = $proc.Start()
    $started = $true
    $procId = $proc.Id
    $copies = @($proc.StandardOutput.BaseStream.CopyToAsync($so), $proc.StandardError.BaseStream.CopyToAsync($se))
    $r.affinity = Get-ArtemisProcessAffinity -Process $proc
    $last = $r.affinity
    while ($true) {
      if ($proc.HasExited) { $exitedEarly = $true; $r.exit_code = $proc.ExitCode; break }
      if ((HealthStatus $Port) -eq 200) {
        $r.first_launch_s = [math]::Round($clock.Elapsed.TotalSeconds, 1)
        # The listener must be the child just started: across a long setup another program could have taken the port.
        $owner = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique)
        if ($owner.Count -eq 1 -and $owner[0] -eq $procId) { $r.health = $true }
        else { $r.first_launch_s = $null; $r.reason = "port $Port is held by pid $($owner -join ', '), not by the server just started (pid $procId)" }
        break
      }
      if ($clock.Elapsed.TotalSeconds -gt $LimitS) { $r.reason = "/health did not answer 200 within $LimitS s"; break }
      $m = Get-ArtemisProcessAffinity -Process $proc
      if ($null -ne $m) { $last = $m }
      Start-Sleep -Milliseconds 500
    }
    # The mask once more, just before the stop (a process that has gone reads $null and keeps the last good read).
    $m = Get-ArtemisProcessAffinity -Process $proc
    if ($null -ne $m) { $last = $m }
    $r.affinity_at_stop = $last
    $bad = @(@($r.affinity, $r.affinity_at_stop) | Where-Object { $null -eq $_ -or $_ -ne $runnerMask })
    if ($bad.Count) { $r.affinity_problem = "the server's affinity was $(MaskText $r.affinity) after it started and $(MaskText $r.affinity_at_stop) before it stopped, not the runner's $(MaskText $runnerMask)" }
  } finally {
    if ($started) { $r.stop = Stop-ArtemisServer -Proc $proc -RunDir $Dir }
    foreach ($c in $copies) { try { $null = $c.Wait(5000) } catch { } }
    $so.Dispose()
    $se.Dispose()
  }
  if ($exitedEarly) {
    $tail = $(if (Test-Path -LiteralPath $r.stderr) { (Get-Content -LiteralPath $r.stderr -Tail 20) -join "`n" } else { '(no stderr)' })
    $r.reason = "artemis serve exited with code $($r.exit_code) before /health answered. stderr tail:`n$tail"
  }
  return $r
}

# --- serve-speed-spec and serve-shrunk --------------------------------------------------------------
# The judging is pure (rows in, a record out) so it can be checked without a GPU. A row is a hashtable: leaf (the run,
# "write-1234" or "long-1"), mode ("off" or "on"), prompt ("write", "edit", "explain" or "long") and the figures decode
# (decode_steps_tok_s), prefill (prefill_tok_s), fnv (completion_fnv), slots (the load line's slots_per_layer) and
# prompt_tokens. Nothing here weakens the pass rules.

# How many aggregates of $Test (the record whose `test` is exactly that: `first-run` is not `first-run run`) the
# evidence file already holds for $Commit, plus one: this run's attempt. A block allows attempts 1 and 2 only.
function AttemptOf([string]$Path, [string]$Commit, [string]$Test) {
  $n = 0
  $needle = '"' + $Test + '"'
  if (Test-Path -LiteralPath $Path) {
    foreach ($line in Get-Content -LiteralPath $Path) {
      if (-not $line.Contains($needle, [System.StringComparison]::Ordinal)) { continue }
      try { $j = $line | ConvertFrom-Json -ErrorAction Stop } catch { continue }
      if ($j.test -eq $Test -and $j.commit -eq $Commit) { $n++ }
    }
  }
  return $n + 1
}
# How many `serve-speed-spec` aggregates the evidence file already holds for $Commit, plus one: this run's attempt.
function SpecAttempt([string]$Path, [string]$Commit) { return AttemptOf $Path $Commit 'serve-speed-spec' }
# The same for the first-run block's `first-run` aggregates.
function FirstRunAttempt([string]$Path, [string]$Commit) { return AttemptOf $Path $Commit 'first-run' }

# $Key of a prompt's two runs in a mode as an array, or $null unless there are exactly two numbers: a figure over a
# partial run set is never computed, so an attempt with a run missing cannot judge.
function SpecPair($Rows, [string]$Mode, [string]$Prompt, [string]$Key) {
  $v = @($Rows | Where-Object { $_.mode -eq $Mode -and $_.prompt -eq $Prompt -and $null -ne $_.$Key } | ForEach-Object { [double]$_.$Key })
  if ($v.Count -ne 2) { return $null }
  return , $v
}
# The six decode figures of a mode (write, edit and explain, two seeds each), or $null unless all six are there.
function SpecSix($Rows, [string]$Mode) {
  $all = @()
  foreach ($p in @('write', 'edit', 'explain')) {
    $pair = SpecPair $Rows $Mode $p 'decode'
    if ($null -eq $pair) { return $null }
    $all += $pair
  }
  return , $all
}
function SpecRatio($On, $Off) {
  if ($null -ne $On -and $null -ne $Off -and $Off -gt 0) { return [double]$On / [double]$Off }
  return $null
}
function AtLeast($X, [double]$Bound) { return ($null -ne $X -and $X -ge $Bound) }

# Which mode a row's process really ran, from its spec_steps figure (null with speculation off, a count with it on once a
# decode step ran). Rows carry leaf, mode, prompt and steps. A problem when an "off" decode row speculated, when an "on"
# decode row has no spec_steps, or when the "on" decode rows made no verify step between them (a short prompt may draft
# nothing, the set must not). The 1-token `long` rows are null in both modes and skipped. Without these a flag that was
# dropped or ignored would compare plain decode with plain decode and pass. A "lookup" decode row (the
# lookup-only process) must speculate too (spec_steps), and one that drafted with MTP (row.mtp, spec_drafted_mtp, above 0)
# is not the lookup-only process: both are problems.
function SpecModeProblems($Rows) {
  $found = @()
  $onRows = 0
  $onSteps = 0
  foreach ($row in $Rows) {
    if ($row.prompt -eq 'long') { continue }
    if ($row.mode -eq 'off' -and $null -ne $row.steps) {
      $found += "$($row.leaf)-off: spec_steps is $($row.steps), so the `"off`" process speculated"
    }
    if ($row.mode -eq 'lookup') {
      if ($null -eq $row.steps) { $found += "$($row.leaf)-lookup: no spec_steps, so the `"lookup`" process did not speculate" }
      if ($null -ne $row.mtp -and $row.mtp -gt 0) { $found += "$($row.leaf)-lookup: spec_drafted_mtp is $($row.mtp), so the `"lookup`" process drafted with MTP" }
    }
    if ($row.mode -eq 'on') {
      $onRows++
      if ($null -eq $row.steps) { $found += "$($row.leaf)-on: no spec_steps, so the `"on`" process did not speculate" } else { $onSteps += $row.steps }
    }
  }
  if ($onRows -gt 0 -and $onSteps -eq 0) { $found += "the `"on`" decode runs made no verify step between them: speculation never ran" }
  return $found
}

# The serve-speed-spec aggregate's fields from a whole attempt's rows. $RunProblems are the run-level ones the block found.
#   dw1  the "on" median of the six decode runs >= 110.0 (absolute)
#   dw2  each prompt's "on" median >= 98% of its "off" median (medians over the two seeds)
#   dw3  the slower long prefill and the slower edit prefill, "on" >= 98% of "off"
#   dw6  the "off" median of six >= 46.5 and the "off" slowest long prefill >= 1,000
# The status is pass iff dw3 and dw6 pass under -SpecPlan 5a (dw1 and dw2 are reported, a miss earns the one rerun and
# then the owner) and iff dw1, dw2, dw3 and dw6 pass under 5b, no run had a problem, and the run set was
# whole; under 5b the "lookup" runs are part of the run set, so an attempt missing one never passes. Ratios are not
# rounded. Under 5b the aggregate also records lookup_median (the median of the six "lookup" figures), mtp_gain (the "on"
# median over it) and mtp_pays (mtp_gain >= 1.02 - 1e-9), report only: none of the three is in the status. A plan
# that is neither is a problem (and the status fails).
function SpeedSpecJudge($Rows, [int]$Attempt, [string[]]$RunProblems = @(), [string]$Plan = '5a') {
  $problems = @($RunProblems | Where-Object { $_ })
  if ($Plan -ne '5a' -and $Plan -ne '5b') { $problems += "unknown plan '$Plan' (the plans are 5a and 5b)" }
  $problems += @(SpecModeProblems $Rows)
  $onSix = SpecSix $Rows 'on'
  $offSix = SpecSix $Rows 'off'
  $onMedian = $(if ($onSix) { Median $onSix } else { $null })
  $offMedian = $(if ($offSix) { Median $offSix } else { $null })
  if (-not $onSix) { $problems += 'the "on" runs do not give six decode figures' }
  if (-not $offSix) { $problems += 'the "off" runs do not give six decode figures' }
  $lookupSix = $null
  $lookupMedian = $null
  if ($Plan -eq '5b') {
    # The gate must not pass with MTP silently off: the six "on" decode runs together must have drafted with it.
    # Only when those runs' configured sources include mtp: with the template's sources = ["lookup"] the
    # "on" runs are lookup-only and the gate is judged on dw1-dw6 without this problem. A row with no sources is the default's.
    $mtpDrafted = 0
    $mtpOn = @($Rows | Where-Object { $_.mode -eq 'on' -and $_.prompt -ne 'long' -and (($null -eq $_.sources) -or (@($_.sources) -contains 'mtp')) })
    foreach ($row in $mtpOn) { if (IsNum $row.mtp) { $mtpDrafted += $row.mtp } }
    if ($mtpOn.Count -gt 0 -and $mtpDrafted -le 0) { $problems += 'the "on" decode runs never drafted with MTP (spec_drafted_mtp is 0 or missing across them), so MTP was off' }
    $lookupSix = SpecSix $Rows 'lookup'
    $lookupMedian = $(if ($lookupSix) { Median $lookupSix } else { $null })
    if (-not $lookupSix) { $problems += 'the "lookup" runs do not give six decode figures' }
  }
  $mtpGain = SpecRatio $onMedian $lookupMedian
  $perPrompt = @()
  foreach ($p in @('write', 'edit', 'explain')) {
    $on = SpecPair $Rows 'on' $p 'decode'
    $off = SpecPair $Rows 'off' $p 'decode'
    $onM = $(if ($on) { Median $on } else { $null })
    $offM = $(if ($off) { Median $off } else { $null })
    $ratio = SpecRatio $onM $offM
    $perPrompt += @{ prompt = $p; on_median = $onM; off_median = $offM; ratio = $ratio; pass = (AtLeast $ratio 0.98) }
    if ($Plan -eq '5b') {
      $lk = SpecPair $Rows 'lookup' $p 'decode'
      $perPrompt[-1].lookup_median = $(if ($lk) { Median $lk } else { $null })
    }
  }
  $prefill = @()
  $slowest = @{}
  foreach ($which in @('long', 'edit')) {
    foreach ($mode in @('on', 'off')) {
      $pair = SpecPair $Rows $mode $which 'prefill'
      $slowest["$which-$mode"] = $(if ($pair) { ($pair | Measure-Object -Minimum).Minimum } else { $null })
    }
    $on = $slowest["$which-on"]
    $off = $slowest["$which-off"]
    $ratio = SpecRatio $on $off
    $prefill += @{ which = $which; on = $on; off = $off; ratio = $ratio; pass = (AtLeast $ratio 0.98) }
    if ($null -eq $on -or $null -eq $off) { $problems += "the $which prefill figure is missing a run" }
  }
  $offLong = $slowest['long-off']
  $offBounds = @{ decode_median = $offMedian; decode_bound = 46.5; prefill_slowest = $offLong; prefill_bound = 1000.0
    pass = ((AtLeast $offMedian 46.5) -and (AtLeast $offLong 1000.0)) }
  # Report only: whether each pair's completion hashes are equal (not expected to be); $null without both.
  $identity = @()
  foreach ($leaf in @($Rows | ForEach-Object { $_.leaf } | Select-Object -Unique)) {
    $offHash = @($Rows | Where-Object { $_.leaf -eq $leaf -and $_.mode -eq 'off' } | ForEach-Object { $_.fnv })[0]
    $onHash = @($Rows | Where-Object { $_.leaf -eq $leaf -and $_.mode -eq 'on' } | ForEach-Object { $_.fnv })[0]
    $identity += @{ run = $leaf; equal = $(if ($offHash -and $onHash) { $offHash -ceq $onHash } else { $null }) }
  }
  # Report only (that speculation allocates 0 new bytes is pinned in CI): each pair's "off" and "on" slots_per_layer
  # side by side, and whether they are equal. Never a problem: the desktop's VRAM can move a slot between two processes
  # on its own.
  $slots = @()
  foreach ($leaf in @($Rows | ForEach-Object { $_.leaf } | Select-Object -Unique)) {
    $offSlots = @($Rows | Where-Object { $_.leaf -eq $leaf -and $_.mode -eq 'off' } | ForEach-Object { $_.slots })[0]
    $onSlots = @($Rows | Where-Object { $_.leaf -eq $leaf -and $_.mode -eq 'on' } | ForEach-Object { $_.slots })[0]
    $slots += @{ run = $leaf; off = $offSlots; on = $onSlots; equal = $(if ($null -ne $offSlots -and $null -ne $onSlots) { $offSlots -eq $onSlots } else { $null }) }
  }
  $longTokens = @($Rows | Where-Object { $_.prompt -eq 'long' -and $null -ne $_.prompt_tokens } | ForEach-Object { $_.prompt_tokens })
  if (@($longTokens | Select-Object -Unique).Count -gt 1) { $problems += "the long runs counted different prompts: $($longTokens -join ', ') tokens" }
  $checks = @{
    dw1 = (AtLeast $onMedian 110.0)
    dw2 = (@($perPrompt | Where-Object { -not $_.pass }).Count -eq 0)
    dw3 = (@($prefill | Where-Object { -not $_.pass }).Count -eq 0)
    dw6 = $offBounds.pass
  }
  $whole = ($null -ne $onMedian) -and ($null -ne $offMedian) -and ($null -ne $offLong) -and
    (@($perPrompt | Where-Object { $null -eq $_.ratio }).Count -eq 0) -and (@($prefill | Where-Object { $null -eq $_.ratio }).Count -eq 0)
  $gate = $(if ($Plan -eq '5b') { $checks.dw1 -and $checks.dw2 -and $checks.dw3 -and $checks.dw6 -and ($null -ne $lookupMedian) } else { $checks.dw3 -and $checks.dw6 })
  $status = $(if ($whole -and $gate -and -not $problems.Count) { 'pass' } else { 'fail' })
  $rec = @{
    test = 'serve-speed-spec'; plan = $Plan; status = $status; attempt = $Attempt
    on_median = $onMedian; bar = 110.0; per_prompt = $perPrompt; prefill = $prefill; off_bounds = $offBounds; identity = $identity
    checks = $checks
    runs = @($Rows | ForEach-Object { @{ run = $_.leaf; mode = $_.mode; decode_steps_tok_s = $_.decode; prefill_tok_s = $_.prefill; slots_per_layer = $_.slots } })
    slots = $slots
    problems = $problems
  }
  if ($Plan -eq '5b') {
    $rec.lookup_median = $lookupMedian
    $rec.mtp_gain = $mtpGain
    $rec.mtp_pays = $(if ($null -ne $mtpGain) { $mtpGain -ge (1.02 - 1e-9) } else { $null })
  }
  return $rec
}

# The serve-shrunk aggregate's pairs from its rows (each also has leaf, slots and, for "on", accept, steps and plain: the
# figures spec_accept_rate, spec_steps and spec_plain_steps; tok is decode_steps_tok_s and miss miss_mib_per_token). A pair
# whose "on" is more than 2% slower than "off" (ratio below 0.98) is listed in needs_owner.
function ShrunkJudge($Rows) {
  $pairs = @()
  $needs = @()
  $problems = @(SpecModeProblems $Rows)
  foreach ($key in @($Rows | ForEach-Object { "$($_.prompt)|$($_.slots)" } | Select-Object -Unique)) {
    $prompt, $slots = $key -split '\|'
    $off = @($Rows | Where-Object { "$($_.prompt)|$($_.slots)" -eq $key -and $_.mode -eq 'off' })[0]
    $on = @($Rows | Where-Object { "$($_.prompt)|$($_.slots)" -eq $key -and $_.mode -eq 'on' })[0]
    $ratio = SpecRatio $on.tok $off.tok
    $share = $null
    if ($null -ne $on.steps -and $null -ne $on.plain -and ($on.steps + $on.plain) -gt 0) { $share = [double]$on.plain / ([double]$on.steps + [double]$on.plain) }
    $pairs += @{ prompt = $prompt; slots = [int]$slots; off_tok_s = $off.tok; on_tok_s = $on.tok; ratio = $ratio
      off_miss_mib_per_token = $off.miss; on_miss_mib_per_token = $on.miss; accept_rate = $on.accept; plain_share = $share }
    if ($null -eq $ratio) { $problems += "$prompt at $slots slots: a figure is missing, so the pair has no ratio" }
    elseif ($ratio -lt 0.98) { $needs += @{ prompt = $prompt; slots = [int]$slots; ratio = $ratio } }
  }
  return @{ pairs = $pairs; needs_owner = $needs; problems = $problems }
}

# --- the MTP judges (pure, like the two above) ---------------------------------------------------------------------

function IsNum($X) { return ($null -ne $X -and ($X -is [int] -or $X -is [long] -or $X -is [double] -or $X -is [decimal] -or $X -is [single] -or $X -is [uint32] -or $X -is [uint64])) }

# blk.40's 256 experts in host memory: 1,900,544 B each (recomputed from the type table). With the MTP loaded
# the host copies grow by exactly this much.
$script:MtpExpertBytes = 486539264
# The snapshot pool the 32 GB machine must keep with MTP on: 2 GiB.
$script:MtpPoolFloorGib = 2

# The `mtp-32gb` verdict from the two runs' memory figures (hashtables with host_bytes, host_heap_usage, mtp_vram_bytes and
# pool_gib, each possibly $null). The pool is `pool_at_26gib`'s: a machine with 26 GiB free and its cap, not this machine's
# free memory. Returns status ('pass', 'ask-owner' or 'fail') and why (a string for a fail, else $null).
#   fail       the "on" run's host_heap_usage or pool_gib is not a number ("not measured"), the "off" run's host_bytes is
#              not, the MTP was not loaded (mtp_vram_bytes not above 0), or the host bytes did not grow by exactly the
#              MTP's experts (the "on" run loaded the bytes of a build without the MTP, or something else)
#   pass       the "on" run's pool is at least 2 GiB
#   ask-owner  it is less
function Mtp32Judge($Off, $On) {
  if (-not $On -or -not (IsNum $On.host_heap_usage) -or -not (IsNum $On.pool_gib)) {
    return @{ status = 'fail'; why = 'not measured: the "on" run has no host_heap_usage or pool_gib' }
  }
  if (-not $Off -or -not (IsNum $Off.host_bytes) -or -not (IsNum $On.host_bytes)) {
    return @{ status = 'fail'; why = 'not measured: a run has no host_bytes' }
  }
  if (-not (IsNum $On.mtp_vram_bytes) -or $On.mtp_vram_bytes -le 0) {
    return @{ status = 'fail'; why = "the `"on`" run's mtp_vram_bytes is $($On.mtp_vram_bytes), so the MTP was not loaded" }
  }
  $grew = [long]$On.host_bytes - [long]$Off.host_bytes
  if ($grew -ne $script:MtpExpertBytes) {
    return @{ status = 'fail'; why = "host_bytes grew by $grew with the MTP on, not by blk.40's $script:MtpExpertBytes B of experts" }
  }
  if ($On.pool_gib -ge $script:MtpPoolFloorGib) { return @{ status = 'pass'; why = $null } }
  return @{ status = 'ask-owner'; why = "the pool with the MTP on is $($On.pool_gib) GiB, under $script:MtpPoolFloorGib" }
}

# `2-mtp`'s one record from the four inputs' verdicts ($Verdicts: name -> the parsed test2-mtp verdict, or $null where none
# was written). Pass iff every input has a verdict, every verdict passes and none is empty; `worst` is the tap with the
# highest err/bound over all inputs; `taps` the number of distinct taps the verdicts hold. $Expected names the inputs the
# block should have run: one with no entry in $Verdicts (the loop stopped on a device failure) fails the record and is
# listed as `missing`, so a pass is never a pass on part of the inputs.
function Mtp2Record($Verdicts, $Expected = @()) {
  $status = 'pass'
  $worst = $null
  $worstRatio = -1.0
  $taps = @{}
  $inputs = @()
  foreach ($name in $Verdicts.Keys) {
    $v = $Verdicts[$name]
    if (-not $v) { $status = 'fail'; $inputs += @{ input = $name; status = 'fail' }; continue }
    if ($v.status -ne 'pass') { $status = 'fail' }
    $inputs += @{ input = $name; status = $v.status }
    if (-not @($v.rows).Count) { $status = 'fail' }
    foreach ($row in @($v.rows)) {
      $taps[[string]$row.tap] = $true
      $r = Ratio $row
      if ($r -gt $worstRatio) { $worstRatio = $r; $worst = @{ tap = $row.tap; err = $row.err; bound = $row.bound } }
    }
  }
  foreach ($name in @($Expected)) {
    if (-not $Verdicts.Contains($name)) { $status = 'fail'; $inputs += @{ input = $name; status = 'missing' } }
  }
  if ($Verdicts.Count -eq 0) { $status = 'fail' }
  return @{ test = '2-mtp'; plan = '5b'; status = $status; form = 'taps'; worst = $worst; taps = $taps.Count; inputs = $inputs; expected_inputs = @($Expected) }
}

# `2-mtp-gpu`'s one record from the inputs' mtp-check verdicts: positions, top-1 equal and near-tie differences summed,
# the worst relative L2 the largest. Pass iff every input has a verdict that passes, none has a position outside a near
# tie, and every position is equal or a near tie.
function Mtp2GpuRecord($Verdicts) {
  $status = 'pass'
  $pos = 0; $eq = 0; $ties = 0; $outside = 0
  $worstL2 = $null
  foreach ($name in $Verdicts.Keys) {
    $v = $Verdicts[$name]
    if (-not $v) { $status = 'fail'; continue }
    if ($v.status -ne 'pass') { $status = 'fail' }
    $pos += [int]$v.positions; $eq += [int]$v.top1_equal; $ties += [int]$v.near_tie_differences; $outside += [int]$v.outside_near_tie
    if ($null -ne $v.worst_rel_l2 -and ($null -eq $worstL2 -or $v.worst_rel_l2 -gt $worstL2)) { $worstL2 = $v.worst_rel_l2 }
  }
  if ($Verdicts.Count -eq 0 -or $pos -eq 0 -or $outside -ne 0 -or $eq + $ties -ne $pos) { $status = 'fail' }
  return @{ test = '2-mtp-gpu'; plan = '5b'; status = $status; positions = $pos; top1_equal = $eq; near_tie_differences = $ties; worst_rel_l2 = $worstL2 }
}

# One process of serve-speed-spec or serve-shrunk (the two blocks' per-run loop body): the fresh server with $Extra, its
# `<Test> run` record (the pinned shape; $Fields adds the block's own keys, attempt or slots; the status is $GoodStatus
# or `fail`), and the problems it found. A run is good when its verdict is `ran`, its process stopped well and its
# fresh_engine check passed (a hard check: otherwise the figures are not a fresh engine's). Returns r (FreshServer's
# result), fig (the verdict's figures), leaf, problems and stop (the process was unusable: the caller ends its loop).
# The draft sources a process is configured with: --speculation-sources in its overrides, else the template's `sources`
# line, else the default (lookup, mtp). Recorded on each run so the judges know whether MTP was a source (the template
# may ship sources = ["lookup"]).
function SpecSourcesOf([string]$TemplatePath, [string[]]$Extra) {
  for ($i = 0; $i -lt $Extra.Count - 1; $i++) {
    if ($Extra[$i] -eq '--speculation-sources') { return @($Extra[$i + 1] -split ',' | ForEach-Object { $_.Trim() }) }
  }
  if ($TemplatePath -and (Test-Path -LiteralPath $TemplatePath)) {
    foreach ($line in Get-Content -LiteralPath $TemplatePath) {
      if ($line -match '^\s*sources\s*=\s*\[(.*)\]') { return @([regex]::Matches($Matches[1], '"([^"]*)"') | ForEach-Object { $_.Groups[1].Value }) }
    }
  }
  return @('lookup', 'mtp')
}

function SpecRun([string]$Test, $Spec, [string]$Mode, [string[]]$Extra, [string]$GoodStatus, $Fields) {
  $leaf = "$($Spec.leaf)-$Mode"
  $r = FreshServer $leaf 'speed-fresh' @('--speed-run', $Spec.arg) $Extra
  $v = $r.verdict
  $fig = $v.figures
  $checks = @{}
  foreach ($c in @($v.checks)) { if ($c) { $checks[$c.name] = [bool]$c.pass } }
  $fresh = ($checks['fresh_engine'] -eq $true)
  $good = [bool]($v -and $v.status -eq 'ran' -and (SpeedOk $r) -and $fresh)
  $line = @{ test = "$Test run"; plan = $SpecPlan; input = $leaf; run = $Spec.arg; mode = $Mode; sources = @(SpecSourcesOf $template $Extra); status = $(if ($good) { $GoodStatus } else { 'fail' })
    slots_per_layer = $r.slots_per_layer; desktop_gpu_mib = $r.mib; stall_lines = $r.stall_lines; graceful = $r.stop.graceful; exit_code = $r.stop.exit_code
    test7_exit = $r.test7_exit; checks = $checks; overrides = $Extra }
  foreach ($k in $Fields.Keys) { $line[$k] = $Fields[$k] }
  $af = AffinityFields $r
  foreach ($k in $af.Keys) { $line[$k] = $af[$k] }
  # The first-run block's runs: the process's own `tuned` line must apply the block's tuned file.
  $tunedApplied = $null
  if ($Fields -and $Fields.ContainsKey('tuned_file')) {
    $ta = TunedApplied (LogEvents (Join-Path $speedRunDir "$leaf\logs") 'tuned') $Fields.tuned_file
    foreach ($k in $ta.Keys) { $line[$k] = $ta[$k] }
    $tunedApplied = $ta.tuned_applied
    if (-not $ta.tuned_applied) { $line.status = 'fail'; $line.tuned_problem = "its tuned line did not apply $($Fields.tuned_file)" }
  }
  if ($v) {
    $line.figures = $fig
  } else {
    $line.error = "no verdict was written (artemis test7 exit $($r.test7_exit)); see $(Join-Path $speedRunDir "$leaf\test7.err")"
  }
  # The template's fields (the overrides are this run's own flags above, as the pinned record has them).
  $rec = $r.meta.Clone()
  foreach ($k in $line.Keys) { $rec[$k] = $line[$k] }
  Record $rec
  $problems = @()
  if ($v -and $v.status -ne 'ran') { $problems += "${leaf}: the verdict is $($v.status), not ran" }
  if (-not $fresh) { $problems += "${leaf}: the fresh_engine check did not pass (it is a hard check: the figures are not a fresh engine's)" }
  if ($r.stall_lines) { $problems += "${leaf}: $($r.stall_lines) stall line(s) in its log" }
  $stop = -not (SpeedOk $r)
  if ($stop) { $problems += "${leaf}: $(SpeedWhy $r); the loop stopped here" }
  return @{ r = $r; fig = $fig; leaf = $leaf; problems = $problems; stop = $stop; sources = $line.sources; good = $good; tuned_applied = $tunedApplied }
}

# A -Prompts id that names no golden prompt is a typo: warn once, for tests 3, 5 and 6 alike.
if ($Prompts.Count -and ($Tests -contains '3' -or $Tests -contains '5' -or $Tests -contains '6')) {
  $goldenIds = @(Get-ChildItem (Join-Path $repo 'testdata\golden\prompts') -Filter *.txt | ForEach-Object { $_.BaseName })
  foreach ($p in $Prompts) {
    if ($goldenIds -notcontains $p) { Write-Warning "-Prompts $p matches no golden prompt" }
  }
}

Push-Location $repo
try {
  # Every process that starts `artemis serve` for a test must find no tuned file and run no setup: the first-run hook
  # would write the tester's real tuned file. Off for the whole run; the first-run block alone points it at its own
  # file for the length of that block.
  $env:ARTEMIS_TUNED = 'off'
  Set-ArtemisAffinity -Mask $runnerMask | Out-Null
  Write-Host "runner affinity $(Format-ArtemisMask $runnerMask)"
  cargo build --release -p artemis-tools; Check 'cargo build'
  $artemis = Join-Path $repo 'target\release\artemis.exe'

  if ($Tests -contains '8') {
    cargo test --release -p qwen36 --test tokenizer_corpus
    $corpus = $LASTEXITCODE -eq 0
    cargo test --release -p qwen36 --test real_model -- --ignored --test-threads=1
    $gguf = $LASTEXITCODE -eq 0
    Record @{ test = '8'; status = $(if ($corpus -and $gguf) { 'pass' } else { 'fail' }); corpus = $corpus; gguf_checks = $gguf }
    if (-not ($corpus -and $gguf)) { $script:failed += 'test 8' }
  }

  if ($Tests -contains '2') {
    $filter = (& $artemis taps-filter).Trim(); Check 'artemis taps-filter'
    foreach ($tok in Get-ChildItem (Join-Path $repo 'testdata\test2\inputs') -Filter *.tokens) {
      $dir = Join-Path $runs "test2\$($tok.BaseName)"
      & $bin taps --model $Model --backend cpu --tokens $tok.FullName --out "$dir\llama-cpu" --filter $filter; Check "oracle cpu on $($tok.Name)"
      & $bin taps --model $Model --backend vulkan --experts-on-cpu --tokens $tok.FullName --out "$dir\llama-vulkan" --filter $filter; Check "oracle vulkan on $($tok.Name)"
      # Gating: teacher-forced, per operation. test2 --forced exits 1 on a failing verdict; it is recorded either way.
      & $artemis dump --cpu --llama-rounding --force-from "$dir\llama-cpu" --model $Model --tokens $tok.FullName --out "$dir\forced"; Check "artemis dump (forced) on $($tok.Name)"
      & $artemis test2 --forced --llama-cpu "$dir\llama-cpu" --candidate "$dir\forced" --out "$dir\verdict.json"
      RecordVerdict '2' $tok.BaseName "$dir\verdict.json"
      # Reported, not gating: the free-running comparison against the Vulkan-vs-CPU floor.
      & $artemis dump --cpu --llama-rounding --model $Model --tokens $tok.FullName --out "$dir\free"; Check "artemis dump (free) on $($tok.Name)"
      & $artemis test2 --llama-cpu "$dir\llama-cpu" --llama-vulkan "$dir\llama-vulkan" --candidate "$dir\free" --out "$dir\free-verdict.json"; Check "artemis test2 (free) on $($tok.Name)"
      $f = Get-Content "$dir\free-verdict.json" -Raw | ConvertFrom-Json
      $fw = $f.rows | Sort-Object { Ratio $_ } -Descending | Select-Object -First 1
      Record @{ test = '2-free (report only)'; input = $tok.BaseName; status = $f.status; worst_tap = $fw.tap; worst_err = $fw.err; worst_bound = $fw.bound; topk_mismatches = @($f.topk_mismatches).Count; near_ties = @($f.near_ties).Count }
    }
  }

  # Test 2's MTP half. The oracle runs llama.cpp's MTP graph teacher-forced over each of test 2's inputs
  # (`mtp-taps`, CPU; hidden.bin holds the main model's hidden rows); the result is cached under
  # runs\mtp-taps\<model sha256>\<input>\oracle with a stamp of the token file, the filter and the oracle binary, so a rerun
  # with all three unchanged skips the oracle. Then the CPU reference forced from it (`dump --cpu --mtp`) is judged by
  # `test2-mtp` (gating, recorded as one `2-mtp`), and `mtp-check` runs the GPU's MTP against the oracle's logits
  # (gating, recorded as `2-mtp-gpu`). The oracle binary is $bin, the one test 2 uses: it must be a build of this
  # checkout's oracle\ (mtp-taps). GPU for the second half.
  if ($Tests -contains '2-mtp') {
    $mtpFilter = (& $artemis taps-filter --mtp).Trim(); Check 'artemis taps-filter --mtp'
    $mtpBase = Join-Path $runs "mtp-taps\$(FileSha $Model)"
    # The oracle's key is the exe and every DLL beside it (llama.cpp's libraries are separate files).
    $binSha = (@(Get-ChildItem -LiteralPath (Split-Path $bin -Parent) -File | Where-Object { $_.Extension -in '.exe', '.dll' } | Sort-Object Name |
        ForEach-Object { "$($_.Name):$(FileSha $_.FullName)" }) -join ',')
    $binSha = [BitConverter]::ToString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($binSha))).Replace('-', '').ToLowerInvariant()
    $mtpV = [ordered]@{}
    $gpuV = [ordered]@{}
    foreach ($tok in Get-ChildItem (Join-Path $repo 'testdata\test2\inputs') -Filter *.tokens) {
      $dir = Join-Path $mtpBase $tok.BaseName
      $od = Join-Path $dir 'oracle'
      $stamp = Join-Path $dir 'stamp.txt'
      $want = "$(FileSha $tok.FullName) $mtpFilter $binSha"
      if (-not ((Test-Path -LiteralPath $stamp) -and (Test-Path -LiteralPath (Join-Path $od 'hidden.bin')) -and ((Get-Content -LiteralPath $stamp -Raw).Trim() -ceq $want))) {
        if (Test-Path -LiteralPath $dir) { Remove-Item -Recurse -Force -LiteralPath $dir }
        New-Item -ItemType Directory -Force $dir | Out-Null
        & $bin mtp-taps --model $Model --backend cpu --tokens $tok.FullName --filter $mtpFilter --out $od; Check "oracle mtp-taps on $($tok.Name)"
        Set-Content -LiteralPath $stamp -Value $want -Encoding utf8NoBOM
      }
      foreach ($f in @('reference', 'verdict.json', 'gpu.json')) { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $dir $f) }
      & $artemis dump --cpu --mtp --model $Model --tokens $tok.FullName --hidden (Join-Path $od 'hidden.bin') --force-from $od --llama-rounding --out (Join-Path $dir 'reference'); Check "artemis dump --mtp on $($tok.Name)"
      # test2-mtp and mtp-check exit 1 on a failing verdict; recorded either way.
      # A non-zero exit with no verdict written is a process or device failure, not a failing verdict: record it and stop,
      # as the other blocks do (the next input would meet the same device).
      & $artemis test2-mtp --oracle $od --reference (Join-Path $dir 'reference') --out (Join-Path $dir 'verdict.json')
      $t2Exit = $LASTEXITCODE
      $mtpV[$tok.BaseName] = $(if (Test-Path (Join-Path $dir 'verdict.json')) { Get-Content (Join-Path $dir 'verdict.json') -Raw | ConvertFrom-Json } else { $null })
      if ($t2Exit -and -not $mtpV[$tok.BaseName]) { $gpuV[$tok.BaseName] = $null; Write-Warning "2-mtp: test2-mtp exited $t2Exit with no verdict on $($tok.Name); stopping"; break }
      & $artemis mtp-check --model $Model --oracle $od --tokens $tok.FullName --json (Join-Path $dir 'gpu.json')
      $gpuExit = $LASTEXITCODE
      $gpuV[$tok.BaseName] = $(if (Test-Path (Join-Path $dir 'gpu.json')) { Get-Content (Join-Path $dir 'gpu.json') -Raw | ConvertFrom-Json } else { $null })
      if ($gpuExit -and -not $gpuV[$tok.BaseName]) { Write-Warning "2-mtp: mtp-check exited $gpuExit with no verdict on $($tok.Name); stopping"; break }
    }
    $r2 = Mtp2Record $mtpV @(Get-ChildItem (Join-Path $repo 'testdata\test2\inputs') -Filter *.tokens | ForEach-Object { $_.BaseName })
    Record $r2
    if ($r2.status -ne 'pass') { $script:failed += 'test 2-mtp' }
    $r2g = Mtp2GpuRecord $gpuV
    Record $r2g
    if ($r2g.status -ne 'pass') { $script:failed += 'test 2-mtp-gpu' }
  }

  if ($Tests -contains '1') {
    $ev = Join-Path $runs 'test1.jsonl'
    New-Item -ItemType Directory -Force $runs | Out-Null
    if (Test-Path $ev) { Remove-Item $ev }
    $env:ARTEMIS_EVIDENCE = $ev
    cargo test --release -p kernels -- --include-ignored --test-threads=1
    $kernels = $LASTEXITCODE -eq 0
    Remove-Item Env:ARTEMIS_EVIDENCE
    $lines = if (Test-Path $ev) { @(Get-Content $ev | ForEach-Object { $_ | ConvertFrom-Json }) } else { @() }
    # The identity lines (`kind` = identity: whether a kernel's bits equal the single-token kernel's) carry no
    # `pass`: they are not checks, so they leave the per-kernel pass grouping and are summarised below.
    $idLines = @($lines | Where-Object { $_.kind -eq 'identity' })
    $cases = @($lines | Where-Object { $_.kind -ne 'identity' })
    foreach ($k in ($cases | Group-Object kernel)) {
      $w = $k.Group | Sort-Object { if ($null -eq $_.err) { [double]::PositiveInfinity } else { $_.err / $_.tol } } -Descending | Select-Object -First 1
      $bad = @($k.Group | Where-Object { -not $_.pass }).Count
      Record @{ test = '1'; kernel = $k.Name; cases = $k.Count; status = $(if ($bad) { 'fail' } else { 'pass' }); worst_case = $w.case; worst_err = $w.err; tol = $w.tol }
    }
    # The prefill block runs the three real-model prefill checks, which take about 8 minutes; not twice. The engine's
    # verify tests write their identity lines (verify_rows, verify_state) to a file of their own; only those lines are
    # read.
    $evEngine = Join-Path $runs 'test1-engine.jsonl'
    if (Test-Path $evEngine) { Remove-Item $evEngine }
    $env:ARTEMIS_EVIDENCE = $evEngine
    cargo test --release -p vk -p engine -- --include-ignored --test-threads=1 @skipPrefill
    $engine = $LASTEXITCODE -eq 0
    Remove-Item Env:ARTEMIS_EVIDENCE
    if (Test-Path $evEngine) {
      $idLines += @(Get-Content $evEngine | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.kind -eq 'identity' })
    }
    Record @{ test = '1'; kernel = '(all)'; status = $(if ($kernels -and $engine) { 'pass' } else { 'fail' }); cases = $cases.Count; kernel_tests = $kernels; vk_and_engine_tests = $engine }
    if (-not ($kernels -and $engine)) { $script:failed += 'test 1' }
    # Report only: per kernel, how many identity cases found the bits equal and how many differ (gdn_seq against gdn_step
    # differs by design).
    $idKernels = @($idLines | Group-Object kernel | ForEach-Object {
        $equal = @($_.Group | Where-Object { $_.equal -eq $true }).Count
        @{ kernel = $_.Name; equal = $equal; differ = $_.Count - $equal }
      })
    Record @{ test = 'test 1 identity'; status = 'report'; kernels = $idKernels }
  }

  if ($Tests -contains '2-gpu') {
    $base = Join-Path $runs 'gpu-taps'
    New-Item -ItemType Directory -Force $base | Out-Null
    # The tiny model, in-process: the engine forced from the exact reference.
    $tv = Join-Path $base 'tiny-verdict.json'
    if (Test-Path $tv) { Remove-Item $tv }
    $env:ARTEMIS_VERDICT = $tv
    cargo test --release -p artemis-tools --test gpu_tiny -- --ignored
    Remove-Item Env:ARTEMIS_VERDICT
    RecordVerdict '2-gpu' 'tiny' $tv
    foreach ($tok in Get-ChildItem (Join-Path $repo 'testdata\test2\inputs') -Filter *.tokens) {
      $dir = Join-Path $base $tok.BaseName
      & $artemis dump --cpu --f16-kv --model $Model --tokens $tok.FullName --out "$dir\reference"; Check "artemis dump (reference, f16 KV) on $($tok.Name)"
      & $artemis dump --gpu --force-from "$dir\reference" --model $Model --tokens $tok.FullName --out "$dir\gpu"; Check "artemis dump --gpu on $($tok.Name)"
      # gpu-taps exits 1 on a failing verdict; it is recorded either way.
      & $artemis gpu-taps --reference "$dir\reference" --candidate "$dir\gpu" --out "$dir\verdict.json"
      RecordVerdict '2-gpu' $tok.BaseName "$dir\verdict.json"
    }
  }

  if ($Tests -contains '4') {
    $kl = Join-Path $runs 'kl'
    New-Item -ItemType Directory -Force $kl | Out-Null
    $perplexity = Join-Path $Oracle 'build\bin\Release\llama-perplexity.exe'
    $corpus = Join-Path $repo 'testdata\kl\corpus.txt'
    $base = Join-Path $kl 'base-cpu.bin'
    # 25 chunks of 4,096 tokens: the header, the tokens, and per chunk 2,047 records of 248,324 u16s.
    $baseBytes = 25416371020
    # What the base and the floor were made from. The base's size depends only on -c, --chunks and the vocabulary, so
    # a sidecar next to each says which corpus and model it came from; a mismatch remakes it.
    $baseMeta = Join-Path $kl 'base-cpu.json'
    $modelFile = Get-Item -LiteralPath $Model
    $identity = [ordered]@{
      corpus_sha256 = (Get-FileHash $corpus -Algorithm SHA256).Hash.ToLower()
      model = $Model; model_bytes = $modelFile.Length
      model_mtime = $modelFile.LastWriteTimeUtc.ToString('o')
      c = 4096; chunks = 25; llama = 'b11509'
    }
    $identityJson = $identity | ConvertTo-Json -Compress
    # Why the sidecar at $path doesn't match $want (a JSON string), or $null if it does.
    function KlStale([string]$path, [string]$want) {
      if (-not (Test-Path $path)) { return "$(Split-Path $path -Leaf) is missing" }
      $have = Get-Content $path -Raw | ConvertFrom-Json
      foreach ($k in ($want | ConvertFrom-Json).PSObject.Properties) {
        if ("$($have.($k.Name))" -ne "$($k.Value)") { return "its $($k.Name) differs" }
      }
      return $null
    }
    $floor = Join-Path $kl 'vulkan-floor.txt'
    $floorMeta = Join-Path $kl 'vulkan-floor.json'
    $card = Get-CimInstance Win32_VideoController | Where-Object Name -match '9070' | Select-Object -First 1
    $floorIdentity = [ordered]@{}
    foreach ($k in $identity.Keys) { $floorIdentity[$k] = $identity[$k] }
    if ($card) { $floorIdentity.driver = $card.DriverVersion }
    $floorJson = $floorIdentity | ConvertTo-Json -Compress
    if ($WriteKlSidecars) {
      # One-off: vouch for a base and floor already on disk, without running anything.
      if ((Get-Item $base).Length -ne $baseBytes) { throw "the KL base is not $baseBytes bytes" }
      $identityJson | Set-Content $baseMeta -Encoding utf8NoBOM
      $floorJson | Set-Content $floorMeta -Encoding utf8NoBOM
      Write-Host "wrote $baseMeta and $floorMeta"
    }
    # llama.cpp's distribution on its CPU backend, made once: about 25 GB and 15 minutes.
    $why = $null
    if (-not (Test-Path $base)) { $why = 'there is no KL base' }
    elseif ((Get-Item $base).Length -ne $baseBytes) { $why = 'the KL base is the wrong size' }
    elseif ($s = KlStale $baseMeta $identityJson) { $why = "the KL base was made from another corpus or model ($s)" }
    if ($why) {
      Write-Host "$why; remaking the base and the floor"
      Remove-Item -Force -ErrorAction SilentlyContinue $base, $baseMeta, $floor, $floorMeta
      & $perplexity -m $Model -f $corpus -c 4096 -b 2048 --chunks 25 -t 16 --no-escape --device none -ngl 0 --kl-divergence-base $base > "$kl\base-cpu.log" 2>&1
      if ($LASTEXITCODE) {
        # A half-written base must not pass for a whole one next time.
        Remove-Item -Force -ErrorAction SilentlyContinue $base, $baseMeta
        throw "llama-perplexity (the CPU base) failed (exit $LASTEXITCODE); the partial base was deleted; see $kl\base-cpu.log"
      }
      $got = (Get-Item $base).Length
      if ($got -ne $baseBytes) { throw "the KL base is $got bytes, not $baseBytes; see $kl\base-cpu.log" }
      if (Select-String -Path "$kl\base-cpu.log" -Pattern 'using device' -Quiet) { throw "the KL base ran on a GPU; see $kl\base-cpu.log" }
      $identityJson | Set-Content $baseMeta -Encoding utf8NoBOM
    } else { Write-Host 'the KL base is up to date' }
    # The noise floor: llama.cpp's Vulkan backend scored against its own CPU base, the experts in host memory.
    $why = $null
    if (-not (Test-Path $floor) -or -not (Select-String -Path $floor -Pattern 'Mean +KLD:' -Quiet)) { $why = 'there is no Vulkan floor' }
    elseif ($s = KlStale $floorMeta $floorJson) { $why = "the Vulkan floor was made from another base, model or driver ($s)" }
    if ($why) {
      Write-Host "$why; remaking the floor"
      Remove-Item -Force -ErrorAction SilentlyContinue $floor, $floorMeta
      & $perplexity -m $Model -c 4096 -b 2048 --chunks 25 -ngl 999 -ot '\.ffn_(gate|up|down)_exps\.=CPU' --kl-divergence-base $base --kl-divergence > $floor 2>&1
      Check 'llama-perplexity (the Vulkan floor)'
      if (-not (Select-String -Path $floor -Pattern 'Mean +KLD:' -Quiet)) { throw "no mean KLD in $floor" }
      $floorJson | Set-Content $floorMeta -Encoding utf8NoBOM
    } else { Write-Host 'the Vulkan floor is up to date' }
    # Artemis on the same chunks, each through prefill.
    # A crashed run must not leave the last run's report or verdict to be recorded as new.
    Remove-Item -Force -ErrorAction SilentlyContinue "$kl\artemis.json"
    & $artemis kl --model $Model --base $base --corpus $corpus -c 4096 --out "$kl\artemis.json" 2> "$kl\artemis.err"
    if ($LASTEXITCODE) {
      # The other blocks still run.
      Record @{ test = '4'; status = 'fail'; path = 'prefill'; error = 'artemis kl failed: ' + (Get-Content "$kl\artemis.err" | Select-Object -Last 1) }
      $script:failed += 'test 4'
    } else {
    Remove-Item -Force -ErrorAction SilentlyContinue "$kl\verdict.json"
    # test4 exits 1 on a failing verdict; it is recorded either way.
    & $artemis test4 --floor-log $floor --kl "$kl\artemis.json" --out "$kl\verdict.json"
    if (-not (Test-Path "$kl\verdict.json")) { throw 'artemis test4 wrote no verdict' }
    $v = Get-Content "$kl\verdict.json" -Raw | ConvertFrom-Json
    $a = Get-Content "$kl\artemis.json" -Raw | ConvertFrom-Json
    Record @{ test = '4'; status = $v.status; path = 'prefill'; artemis_kld = $v.artemis; artemis_kld_unc = $a.kld_unc; floor_kld = $v.floor; bound = $v.bound; ppl = $a.ppl; same_top = $a.same_top; scored = $a.count; chunks = $a.chunks; base_chunks = $a.base_chunks; problem = $v.problem; cache_slots = $a.cache_slots; seconds = $a.seconds }
    if ($v.status -ne 'pass') { $script:failed += 'test 4' }
    }
  }
  if ($Tests -contains 'gen') {
    $tok = Join-Path $repo 'testdata\test2\inputs\02-prose.tokens'
    $out = Join-Path $runs 'generate-smoke.txt'
    $outJson = Join-Path $runs 'generate-smoke.json'
    if (Test-Path $outJson) { Remove-Item $outJson }
    & $artemis generate --model $Model --tokens-file $tok --max-tokens 32 --greedy --ctx 4096 --json $outJson 2> "$out.err" > $out
    $ok = $LASTEXITCODE -eq 0
    # The model's text goes into the evidence as a hash and a count, never as text: text_fnv is the
    # FNV-1a 64 of the output file's bytes, tokens the number of tokens the run generated (the record's steps).
    $smoke = @{ test = 'generate (smoke)'; status = $(if ($ok) { 'ran' } else { 'fail' }); stats = (Get-Content "$out.err" | Where-Object { $_ -notlike 'speculation:*' } | Select-Object -Last 1); speculation_line = (Get-Content "$out.err" | Where-Object { $_ -like 'speculation:*' } | Select-Object -Last 1) }
    if ($ok -and (Test-Path $outJson)) {
      $smoke.text_fnv = Fnv1a64Hex ([IO.File]::ReadAllBytes($out))
      $smoke.text_bytes = (Get-Item $out).Length
      $smoke.tokens = @((Get-Content $outJson -Raw | ConvertFrom-Json).steps).Count
    } elseif ($ok) {
      # generate exited 0 but wrote no record: the hash and the count would be silently missing, so this is a failure.
      $ok = $false
      $smoke.status = 'fail'
      $smoke.error = "generate exited 0 but wrote no --json record at $outJson"
    }
    Record $smoke
    if (-not $ok) { $script:failed += 'generate smoke' }
  }

  if ($Tests -contains 'prefill') {
    # The real model, in-process: prefill against decode, the same bits whatever the chunk size, and the
    # long prompt's longest submit. Each test writes its measurements whether or not they pass.
    $dir = Join-Path $runs 'prefill'
    New-Item -ItemType Directory -Force $dir | Out-Null
    Remove-Item -Force -ErrorAction SilentlyContinue "$dir\*.json"
    $env:ARTEMIS_EVIDENCE_DIR = $dir
    cargo test --release -p engine --test prefill_real -- --ignored --test-threads=1 --nocapture 2> "$dir\test.err"
    $ok = $LASTEXITCODE -eq 0
    Remove-Item Env:ARTEMIS_EVIDENCE_DIR
    foreach ($name in @($prefillTests | ForEach-Object { $_.evidence })) {
      $f = Join-Path $dir "$name.json"
      if (-not (Test-Path $f)) {
        Record @{ test = 'prefill'; input = $name; status = 'fail'; error = "no measurements were written; see $dir\test.err" }
        $script:failed += "prefill $name"
        continue
      }
      $v = Get-Content $f -Raw | ConvertFrom-Json
      $r = @{ test = 'prefill' }
      foreach ($p in $v.PSObject.Properties) { $r[$p.Name] = $p.Value }
      $r.input = $name
      Record $r
      if ($v.status -ne 'pass') { $script:failed += "prefill $name" }
    }
    if (-not $ok -and -not ($script:failed | Where-Object { $_ -like 'prefill*' })) { $script:failed += 'prefill (cargo test)' }
  }

  if ($Tests -contains '3') {
    $golden = Join-Path $repo 'testdata\golden'
    $dir = Join-Path $runs 'test3'
    New-Item -ItemType Directory -Force $dir | Out-Null
    $allIds = @(Get-ChildItem (Join-Path $golden 'prompts') -Filter *.txt | Sort-Object Name | ForEach-Object { $_.BaseName })
    $ids = GoldenIds $allIds
    $runArgs = @()
    $notRun = @()
    foreach ($id in $ids) {
      $run = Join-Path $dir "$id.json"
      $force = Join-Path $dir "$id.oracle.tokens"
      if (Test-Path $run) { Remove-Item $run }
      OracleTokens (Join-Path $golden "expected\$id.json") $force
      # The prompt runs through prefill, then the oracle's tokens are fed one per step, as llama.cpp's CPU floor was fed
      # them. --max-tokens 256 is the oracle's --n 256.
      & $artemis generate --model $Model --prompt-file (Join-Path $golden "prompts\$id.txt") --greedy --max-tokens 256 --force-tokens $force --json $run 2> "$dir\$id.err" > "$dir\$id.txt"
      if ($LASTEXITCODE) {
        Record @{ test = '3'; input = $id; status = 'fail'; error = (Get-Content "$dir\$id.err" | Select-Object -Last 1) }
        $script:failed += "test 3 $id"
        $notRun += $id
        continue
      }
      $runArgs += @('--run', "$id=$run")
    }
    $verdict = Join-Path $dir 'verdict.json'
    if (Test-Path $verdict) { Remove-Item $verdict }
    # One verdict over every prompt run: the flips are pooled. test3 exits 1 on a failing verdict; it is recorded
    # either way.
    if ($runArgs.Count) { & $artemis test3 --golden $golden @runArgs --out $verdict }
    RecordTest3 $verdict $dir $notRun
  }

  if ($Tests -contains '6') {
    $golden = Join-Path $repo 'testdata\golden'
    $dir = Join-Path $runs 'test6'
    New-Item -ItemType Directory -Force $dir | Out-Null
    $allIds = @(Get-ChildItem (Join-Path $golden 'prompts') -Filter *.txt | Sort-Object Name | ForEach-Object { $_.BaseName })
    $ids = GoldenIds $allIds
    $coldArgs = @()
    $path1Args = @()
    $path2Args = @()
    $notRun = @()
    foreach ($id in $ids) {
      $force = Join-Path $dir "$id.oracle.tokens"
      OracleTokens (Join-Path $golden "expected\$id.json") $force
      # Three fresh processes a prompt: the cold run (test 3's, from the prompt's tokens and with every position
      # fingerprinted), then each warm path. Each warm run reads its own tokens and the oracle's from the golden set.
      $cold = Join-Path $dir "$id.cold.json"
      if (Test-Path $cold) { Remove-Item $cold }
      & $artemis generate --model $Model --tokens-file (Join-Path $golden "prompts\$id.tokens") --greedy --max-tokens 256 --force-tokens $force --all-logits --json $cold 2> "$dir\$id.cold.err" > "$dir\$id.cold.txt"
      if ($LASTEXITCODE) {
        Record @{ test = '6'; input = $id; status = 'fail'; run = 'cold'; error = (Get-Content "$dir\$id.cold.err" | Select-Object -Last 1) }
        $script:failed += "test 6 $id cold"
        $notRun += "$id cold"
        continue
      }
      $coldArgs += @('--cold', "$id=$cold")
      foreach ($path in 1, 2) {
        $run = Join-Path $dir "$id.path$path.json"
        if (Test-Path $run) { Remove-Item $run }
        & $artemis warm --model $Model --golden $golden --id $id --path $path --all-logits --json $run 2> "$dir\$id.path$path.err"
        if ($LASTEXITCODE) {
          Record @{ test = '6'; input = $id; status = 'fail'; run = "path $path"; error = (Get-Content "$dir\$id.path$path.err" | Select-Object -Last 1) }
          $script:failed += "test 6 $id path $path"
          $notRun += "$id path $path"
          continue
        }
        if ($path -eq 1) { $path1Args += @('--path1', "$id=$run") } else { $path2Args += @('--path2', "$id=$run") }
      }
    }
    $verdict = Join-Path $dir 'verdict.json'
    if (Test-Path $verdict) { Remove-Item $verdict }
    # One verdict over every prompt run: each path's flips are pooled. test6 exits 1 on a failing verdict; it is
    # recorded either way.
    if ($coldArgs.Count) { & $artemis test6 --golden $golden @coldArgs @path1Args @path2Args --out $verdict }
    RecordTest6 $verdict $notRun
  }
  if ($Tests -contains '5') {
    $golden = Join-Path $repo 'testdata\golden\prompts'
    $dir = Join-Path $runs $(if ($CpuShare -gt 0) { "test5-share$CpuShare" } else { 'test5' })
    # With a CPU share the engine refuses speculation, and generate refuses --draft-len with --speculation off:
    # that mode runs plain decode, with no --draft-len and no sixth run. Otherwise the five runs speculate with a
    # fixed draft length, so every cache size runs the same plain and verify steps whatever the timing.
    $shareArgs = @(if ($CpuShare -gt 0) { '--cpu-share'; "$CpuShare"; '--speculation'; 'off' })
    $draftArgs = @(if ($CpuShare -le 0) { '--draft-len'; '4' })
    New-Item -ItemType Directory -Force $dir | Out-Null
    # 10% and 40% of 256 experts; 256 asks for every slot and is clamped to what fits beside a 4,096-position session.
    $configs = [ordered]@{
      'cache-0' = @('--cache-slots', '0')
      'cache-26' = @('--cache-slots', '26')
      'cache-102' = @('--cache-slots', '102')
      'cache-max' = @('--cache-slots', '256')
      'cache-102-swaps' = @('--cache-slots', '102', '--force-swap-every-token')
    }
    $t5ids = @(GoldenIds @('05-code-write-lru', '12-prose-story', '16-tool-weather'))
    if (-not $t5ids.Count) {
      Write-Host "test 5 has nothing to run: -Prompts selects none of 05-code-write-lru, 12-prose-story, 16-tool-weather"
      Record @{ test = '5'; input = '(none)'; status = 'fail'; error = '-Prompts selects none of test 5''s three prompts' }
      $script:failed += 'test 5'
    }
    foreach ($id in $t5ids) {
      $runArgs = @()
      $ran = $true
      foreach ($label in $configs.Keys) {
        $json = Join-Path $dir "$id.$label.json"
        if (Test-Path $json) { Remove-Item $json }
        # --all-logits: every prompt position's logits come from prefill and are compared too.
        & $artemis generate --model $Model --prompt-file (Join-Path $golden "$id.txt") --greedy --max-tokens 64 --ctx 4096 --json $json --all-logits $configs[$label] @draftArgs @shareArgs 2> "$dir\$id.$label.err" > "$dir\$id.$label.txt"
        if ($LASTEXITCODE) {
          Record @{ test = '5'; input = $id; status = 'fail'; error = "${label}: " + (Get-Content "$dir\$id.$label.err" | Select-Object -Last 1) }
          $script:failed += "test 5 $id"
          $ran = $false
          break
        }
        $runArgs += @('--run', "$label=$json")
      }
      # The sixth run: speculation off at the largest cache, compared with the five and reported, never gating.
      # It is expected to differ in the DeltaNet layers. Not with a CPU share: that mode is plain decode throughout.
      $reportArgs = @()
      if ($ran -and $CpuShare -le 0) {
        $plainJson = Join-Path $dir "$id.plain.json"
        if (Test-Path $plainJson) { Remove-Item $plainJson }
        & $artemis generate --model $Model --prompt-file (Join-Path $golden "$id.txt") --greedy --max-tokens 64 --ctx 4096 --json $plainJson --all-logits --cache-slots 256 --speculation off 2> "$dir\$id.plain.err" > "$dir\$id.plain.txt"
        if ($LASTEXITCODE) {
          Record @{ test = '5'; input = $id; status = 'fail'; error = 'plain: ' + (Get-Content "$dir\$id.plain.err" | Select-Object -Last 1) }
          $script:failed += "test 5 $id"
          $ran = $false
        } else {
          $reportArgs = @('--report-run', "plain=$plainJson")
        }
      }
      if (-not $ran) { continue }
      $verdict = Join-Path $dir "$id.verdict.json"
      if (Test-Path $verdict) { Remove-Item $verdict }
      # test5 exits 1 on a failing verdict; it is recorded either way.
      & $artemis test5 @runArgs @reportArgs --out $verdict
      RecordTest5 $id $verdict
    }
  }

  if ($Tests -contains 'hits') {
    $golden = Join-Path $repo 'testdata\golden\prompts'
    $dir = Join-Path $runs 'hits'
    New-Item -ItemType Directory -Force $dir | Out-Null
    $ids = @('01-code', '05-code-write-lru', '06-code-write-sql', '07-code-write-ts', '08-code-edit-bugfix',
      '09-code-edit-refactor', '10-code-explain-regex', '11-code-explain-rust', '12-prose-story', '13-prose-email',
      '14-prose-summary', '15-prose-explain', '16-tool-weather', '17-tool-result', '18-tool-multi')
    $runArgs = @()
    $desktop = @()
    $ran = $true
    foreach ($id in $ids) {
      $pair = @()
      foreach ($policy in 'adaptive', 'static') {
        $json = Join-Path $dir "$id.$policy.json"
        if (Test-Path $json) { Remove-Item $json }
        $desktop += @{ run = "$id-$policy"; mib = (DesktopGpuMib) }
        & $artemis generate --model $Model --tokens-file (Join-Path $golden "$id.tokens") --greedy --max-tokens 128 --ctx 65536 --cache-policy $policy --speculation off --json $json --no-fingerprints 2> "$dir\$id.$policy.err" > "$dir\$id.$policy.txt"
        if ($LASTEXITCODE) {
          Record @{ test = 'hits'; input = $id; status = 'fail'; error = "${policy}: " + (Get-Content "$dir\$id.$policy.err" | Select-Object -Last 1); desktop_gpu_mib = $desktop }
          $script:failed += "hits $id"
          $ran = $false
          break
        }
        $pair += $json
      }
      if (-not $ran) { break }
      $runArgs += @('--run', "$id=$($pair[0]),$($pair[1])")
    }
    if ($ran) {
      $verdict = Join-Path $dir 'verdict.json'
      if (Test-Path $verdict) { Remove-Item $verdict }
      # hits exits 1 on a failing verdict; it is recorded either way.
      & $artemis hits --golden (Join-Path $repo 'testdata\golden') @runArgs --out $verdict
      if (-not (Test-Path $verdict)) {
        Record @{ test = 'hits'; input = 'golden'; status = 'fail'; error = 'no verdict was written'; desktop_gpu_mib = $desktop }
        $script:failed += 'hits'
      } else {
        $v = Get-Content $verdict -Raw | ConvertFrom-Json
        Record @{ test = 'hits'; input = 'golden'; status = $v.status; groups = @($v.groups); rows = @($v.rows); problems = @($v.problems); desktop_gpu_mib = $desktop }
        if ($v.status -ne 'pass') { $script:failed += 'hits' }
      }
    }
  }

  # The on/off golden comparison (done-when 4): `artemis spec-check golden`
  # loads once and runs the 20 golden prompts greedy for 256 tokens, free-running, off then on for each prompt. A first
  # divergence is allowed only at a near-tie (test 3's 0.05). The block gates on the command's exit code AND the status in
  # verdict.json: either one failing fails it, and no verdict at all is a failure. The per-prompt records and the verdict
  # are in $runs\spec-golden; the evidence line carries the verdict's counts, the prompts' divergences (ids, positions and
  # margins) and the speculation counts, never text.
  if ($Tests -contains 'spec-golden') {
    $dir = Join-Path $runs 'spec-golden'
    New-Item -ItemType Directory -Force $dir | Out-Null
    $verdict = Join-Path $dir 'verdict.json'
    if (Test-Path $verdict) { Remove-Item $verdict }
    & $artemis spec-check golden --model $Model --out $dir --golden (Join-Path $repo 'testdata\golden') 2> "$dir\run.err" > "$dir\run.txt"
    $goldenExit = $LASTEXITCODE
    if (-not (Test-Path $verdict)) {
      Record @{ test = 'spec-golden'; plan = $SpecPlan; status = 'fail'; exit_code = $goldenExit; error = "no verdict was written (artemis spec-check exit $goldenExit): $(Get-Content "$dir\run.err" -ErrorAction SilentlyContinue | Select-Object -Last 1)" }
      $script:failed += 'spec-golden'
    } else {
      $v = Get-Content $verdict -Raw | ConvertFrom-Json
      $goldenOk = ($goldenExit -eq 0) -and ($v.status -eq 'pass')
      $status = $(if ($goldenOk) { 'pass' } elseif ($v.status -and $v.status -ne 'pass') { $v.status } else { 'fail' })
      $rec = @{ test = 'spec-golden'; plan = $SpecPlan; status = $status; exit_code = $goldenExit; outside_near_tie = $v.outside_near_tie; identical = $v.identical
        prompts = @($v.prompts | Where-Object { $null -ne $_ }); spec = $v.spec }
      # The on runs' sources and whether the load had the MTP, from the first on record: the summary applies the
      # never-drafted-MTP rule only when the sources include "mtp" (and the load had it).
      $onRec = Get-ChildItem -LiteralPath $dir -Filter '*.on.json' -ErrorAction SilentlyContinue | Select-Object -First 1
      if ($onRec) {
        $onJson = Get-Content -LiteralPath $onRec.FullName -Raw | ConvertFrom-Json
        if ($onJson.settings.speculation) { $rec.sources = @($onJson.settings.speculation.sources) }
        if ($null -ne $onJson.memory.mtp_vram_bytes) { $rec.mtp_loaded = ([int64]$onJson.memory.mtp_vram_bytes -gt 0) }
      }
      $why = @($v.problems | Where-Object { $_ })
      if ($why.Count) { $rec.problems = $why }
      Record $rec
      if (-not $goldenOk) { $script:failed += 'spec-golden' }
    }
  }

  # Test-Messages, run against a block's server after test7's cases, before the stop: a child pwsh, so the
  # script's `exit` ends only itself. Its output holds no key; the printed cache_read_input_tokens goes into the record,
  # whose input is the block's name (the run directory's leaf). Declared here, before every block that uses it (`serve`
  # and `prod-check`); $script:testMessageInputs lists the blocks it has recorded for, so a block can check that it ran.
  $script:testMessageInputs = @()
  $testMessages = {
    param($RunDir, $KeyFile)
    $tmOut = Join-Path $RunDir 'test-messages.txt'
    $tmInput = Split-Path $RunDir -Leaf
    $hostPwsh = (Get-Process -Id $PID).Path
    & $hostPwsh -NoProfile -NonInteractive -File (Join-Path $repo 'tools\test-messages.ps1') -BaseUrl 'http://127.0.0.1:8090' -KeyFile $KeyFile *> $tmOut
    $tmExit = $LASTEXITCODE
    $tmCached = $null
    $m = Select-String -LiteralPath $tmOut -Pattern '^cache_read_input_tokens: (\d+)$' | Select-Object -Last 1
    if ($m) { $tmCached = [int64]$m.Matches[0].Groups[1].Value }
    Record @{ test = 'test-messages'; input = $tmInput; status = $(if ($tmExit -eq 0) { 'pass' } else { 'fail' }); exit_code = $tmExit; cache_read_input_tokens = $tmCached; output = $tmOut; artemis_sha256 = (ConfigFields $null).artemis_sha256 }
    $script:testMessageInputs += $tmInput
    if ($tmExit -ne 0) { $script:failed += 'test-messages' }
  }

  # The watchdog test on the real engine: the #[ignore]d test a_stuck_engine_logs_stall_and_exits_4
  # starts `artemis serve` itself from the filled template, with the hidden --test-stall-at-token hook, on 8090. It uses
  # $runs\serve-stall as its scratch directory (artemis.toml, logs\, stderr.txt, stall.json) and the key file given in
  # ARTEMIS_KEY_FILE, which it only reads. Built first, outside the time limit; the run has 10 minutes. The record's
  # config_sha256 is this runner's hash of the artemis.toml the test wrote (stall.json carries none). The cargo test
  # rebuilds artemis.exe with the dev-dependencies' features, as 2-gpu's does, so the block builds the shipping one again
  # when the run is over. Beyond the test's own exit code the block fails when: the run passes its 10 minutes (its process
  # tree is killed); the test wrote no artemis.toml or no stall.json; or the filled artemis.toml names port 8080. The
  # record also says whether 8090 was free afterwards (port_free) and how many `stall` lines the logs hold.
  if ($Tests -contains 'serve-stall') {
    $block = 'serve-stall'
    BlockPreflight $block
    if (@(Get-Process -Name artemis -ErrorAction SilentlyContinue).Count) { throw "${block}: an artemis process is running; refusing to start" }
    if (@(Get-NetTCPConnection -LocalPort 8090 -State Listen -ErrorAction SilentlyContinue).Count) { throw "${block}: port 8090 is in use; refusing to start" }
    # Beside the run directory, not in it: the test reads its scratch directory while cargo's output is still written.
    $cargoOut = Join-Path $runs "$block.cargo-test.txt"
    $cargoErr = Join-Path $runs "$block.cargo-test.err"
    RefuseOldKey $block @($cargoOut, $cargoErr)
    cargo test --release -p artemis-tools --test serve_stall_gpu --no-run; Check 'cargo test --no-run (serve-stall)'
    $script:serveBuilt = $false
    $stallDir = FreshRunDir $block
    $keyFile = New-ArtemisKey -Dir (Join-Path $runs "keys\$block")
    try {
      $env:ARTEMIS_RUN_DIR = $stallDir
      $env:ARTEMIS_KEY_FILE = $keyFile
      $cargoExe = (Get-Command cargo -CommandType Application | Select-Object -First 1).Source
      $p = Start-Process -FilePath $cargoExe -ArgumentList @('test', '--release', '-p', 'artemis-tools', '--test', 'serve_stall_gpu', '--', '--ignored', '--test-threads=1') -WorkingDirectory $repo -NoNewWindow -PassThru -RedirectStandardOutput $cargoOut -RedirectStandardError $cargoErr
      $null = $p.Handle   # keeps the exit code readable after the process has gone
      $timedOut = -not $p.WaitForExit(600000)
      if ($timedOut) {
        try { $p.Kill($true) } catch { }
        $null = $p.WaitForExit(30000)
      }
      $exit = $(if ($timedOut) { $null } else { $p.ExitCode })
      # Whatever the test left running is ours: nothing ran before it.
      foreach ($stray in @(Get-Process -Name artemis -ErrorAction SilentlyContinue)) {
        Write-Warning "${block}: killing the artemis process (pid $($stray.Id)) the test left running"
        try { $stray.Kill($true); $null = $stray.WaitForExit(10000) } catch { }
      }
      $portFree = $false
      for ($i = 0; $i -lt 20 -and -not $portFree; $i++) {
        $portFree = -not @(Get-NetTCPConnection -LocalPort 8090 -State Listen -ErrorAction SilentlyContinue).Count
        if (-not $portFree) { Start-Sleep -Milliseconds 500 }
      }
      if (-not $portFree) { Write-Warning "${block}: port 8090 is still busy after the test" }
      $problems = @()
      if ($timedOut) { $problems += 'the test did not finish within 10 minutes; its process tree was killed' }
      $stallJson = Join-Path $stallDir 'stall.json'
      $figures = $null
      if (Test-Path -LiteralPath $stallJson) { $figures = Get-Content -LiteralPath $stallJson -Raw | ConvertFrom-Json } else { $problems += 'the test wrote no stall.json' }
      $toml = Join-Path $stallDir 'artemis.toml'
      $configSha = $null
      if (Test-Path -LiteralPath $toml) {
        $configSha = FileSha $toml
        if ((Get-Content -LiteralPath $toml -Raw) -match '(?<![0-9])8080(?![0-9])') { $problems += 'the filled config names port 8080' }
      } else {
        $problems += 'the test wrote no artemis.toml'
      }
      $status = $(if ($exit -eq 0 -and -not $problems.Count) { 'pass' } else { 'fail' })
      Record @{ test = 'serve-stall'; status = $status; exit_code = $exit; figures = $figures; config_template = $template; template_sha256 = (FileSha $template); config_sha256 = $configSha
        stall_lines = (StallLines (Join-Path $stallDir 'logs')); port_free = $portFree; problems = $problems; output = $cargoOut }
      if ($status -ne 'pass') { $script:failed += 'serve-stall' }
    } catch {
      Record @{ test = 'serve-stall'; status = 'fail'; error = $_.Exception.Message }
      $script:failed += 'serve-stall'
      throw
    } finally {
      Remove-Item Env:ARTEMIS_RUN_DIR, Env:ARTEMIS_KEY_FILE -ErrorAction SilentlyContinue
      ScanKey $block $keyFile (@($stallDir, $evidence) + @($cargoOut, $cargoErr | Where-Object { Test-Path -LiteralPath $_ }))
    }
    # The test's cargo run left the dev-dependencies' feature-unified artemis.exe in target\release: build the shipping one
    # now (untimed), so no later block, the engine tool's `speed` included, runs the other.
    cargo build --release -p artemis-tools; Check 'cargo build (after serve-stall)'
    $script:serveBuilt = $true
  }

  if ($Tests -contains 'speed' -or $Tests -contains 'profile' -or $Tests -contains 'llama-search' -or $Tests -contains 'speed-llama' -or $Tests -contains 'speed-llama-65k' -or $Tests -contains 'depth' -or $Tests -contains 'attn-depth' -or $Tests -contains 'serve' -or $Tests -contains 'serve-speed' -or $Tests -contains 'prod-check' -or $Tests -contains 'serve-speed-spec' -or $Tests -contains 'serve-shrunk' -or $Tests -contains 'first-run') {
    # The edit prompt quotes PSF-licensed text, committed with its licence in testdata\speed\sources.
    # The check stage fails unless every speed prompt's ids match made-from.json.
    & $python -I (Join-Path $repo 'tools\py\build_speed_prompts.py') $repo $Model (Join-Path $Oracle 'build\bin\Release') --stage check; Check 'build_speed_prompts check'
    $speedDir = Join-Path $repo 'testdata\speed'
    $speedTokens = @{
      write = Join-Path $speedDir 'prompts\write.tokens'
      edit = Join-Path $speedDir 'prompts\edit.tokens'
      explain = Join-Path $speedDir 'prompts\explain.tokens'
    }
    $long = Join-Path $repo 'testdata\golden\prompts\20-long-option-rs.tokens'
  }

  # The speed gate through the server: one fresh `artemis serve` process, started
  # from the template with --park-snapshots false, for each of eight requests, so every figure is a fresh engine's: the
  # six decode runs of test 7's speed-fresh (seed outermost, as `speed`'s) and prompt 20's prefill twice (`long`). The
  # decode rate is the server log's decode_tokens / decode_ms, which test 7 reports as decode_steps_tok_s. Then one more
  # process runs the existing `speed` case, six requests back to back, reported only. Run directories:
  # $runs\serve-speed\<prompt>-<seed>, long-1, long-2, mixed (each with its artemis.toml, logs\, verdicts\). One key
  # for the block, under $runs\keys\serve-speed\. Records: a `serve-speed run` line per process (the verdict, the load
  # line's slots_per_layer, the desktop's GPU memory sampled with no artemis running, stall_lines), a `serve stop` line
  # per process, `serve-speed mixed (report only)`, and the block's `serve-speed` line. Pass: decode median >= 46.5,
  # the slower prefill >= 1,000, every verdict `ran`, every stop graceful and no `stall` line in any log. A process
  # that fails to answer or to stop gracefully ends the loop: nothing after it runs. A stop must leave 8090 free. The
  # block fails on three more grounds: a decode or prefill run that gave no rate (fewer than 6 decode_steps_tok_s or 2
  # prefill_tok_s); the two long runs counting different prompt lengths; and the mixed-traffic process (report only for
  # its figure) writing no verdict, stopping ungracefully or logging a `stall` line. A nonzero test7 exit in the mixed
  # run fails the block.
  if ($Tests -contains 'serve-speed') {
    $block = 'serve-speed'
    SpeedHygiene $block | Out-Null
    $speedExtra = @('--park-snapshots', 'false')
    $decodeBound = 46.5
    $prefillBound = 1000
    $longReference = 27416
    RefuseOldKey $block
    $speedRunDir = FreshRunDir $block
    BlockPreflight $block
    if (-not $script:serveBuilt) {
      cargo build --release -p artemis-tools; Check 'cargo build (serve-speed)'
      $script:serveBuilt = $true
    }
    $keyFile = New-ArtemisKey -Dir (Join-Path $runs "keys\$block")

    $specs = @()
    foreach ($seed in @(1234, 1235)) {
      foreach ($id in @('write', 'edit', 'explain')) { $specs += @{ leaf = "$id-$seed"; arg = "${id}:$seed"; long = $false } }
    }
    foreach ($i in @(1, 2)) { $specs += @{ leaf = "long-$i"; arg = 'long'; long = $true } }
    $runRows = @()
    $desktop = @()
    $problems = @()
    $meta = $null
    $mixedOk = $true
    try {
      foreach ($spec in $specs) {
        $r = FreshServer $spec.leaf 'speed-fresh' @('--speed-run', $spec.arg)
        if (-not $meta) { $meta = $r.meta }
        $desktop += @{ run = $spec.leaf; mib = $r.mib }
        $v = $r.verdict
        $fig = $v.figures
        $row = @{ run = $spec.leaf; long = $spec.long; status = $(if ($v) { $v.status } else { 'fail' }); decode_steps_tok_s = $fig.decode_steps_tok_s; prefill_tok_s = $fig.prefill_tok_s; prompt_tokens = $fig.prompt_tokens
          slots_per_layer = $r.slots_per_layer; stall_lines = $r.stall_lines; graceful = $r.stop.graceful }
        $line = @{ test = 'serve-speed run'; input = $spec.leaf; run = $spec.arg; status = $row.status; slots_per_layer = $r.slots_per_layer; desktop_gpu_mib = $r.mib; stall_lines = $r.stall_lines
          graceful = $r.stop.graceful; exit_code = $r.stop.exit_code; test7_exit = $r.test7_exit }
        if ($v) {
          $line.checks = @($v.checks)
          $line.figures = $fig
        } else {
          $line.error = "no verdict was written (artemis test7 exit $($r.test7_exit)); see $(Join-Path $speedRunDir "$($spec.leaf)\test7.err")"
        }
        $af = AffinityFields $r
        foreach ($k in $af.Keys) { $line[$k] = $af[$k] }
        if ($af.status) { $row.status = 'fail' }
        Record ($line + $r.meta)
        $runRows += $row
        if ($v -and $v.status -ne 'ran') { $problems += "$($spec.leaf): the verdict is $($v.status), not ran" }
        if ($r.stall_lines) { $problems += "$($spec.leaf): $($r.stall_lines) stall line(s) in its log" }
        if (-not (SpeedOk $r)) {
          $problems += "$($spec.leaf): $(SpeedWhy $r); the loop stopped here"
          $mixedOk = $false
          break
        }
      }

      # The mixed-traffic figure (report only): the existing `speed` case, six requests back to back on one process.
      if ($mixedOk) {
        $r = FreshServer 'mixed' 'speed' @()
        $v = $r.verdict
        $desktop += @{ run = 'mixed'; mib = $r.mib }
        $mixedStatus = $(if ($v) { $v.status } else { 'error' })
        $line = @{ test = 'serve-speed mixed (report only)'; input = 'mixed'; status = $mixedStatus; slots_per_layer = $r.slots_per_layer; desktop_gpu_mib = $r.mib; stall_lines = $r.stall_lines
          graceful = $r.stop.graceful; exit_code = $r.stop.exit_code; test7_exit = $r.test7_exit }
        if ($v) {
          $line.checks = @($v.checks)
          $line.figures = $v.figures
        } else {
          $line.error = "no verdict was written (artemis test7 exit $($r.test7_exit)); see $(Join-Path $speedRunDir 'mixed\test7.err')"
        }
        $af = AffinityFields $r
        foreach ($k in $af.Keys) { $line[$k] = $af[$k] }
        Record ($line + $r.meta)
        # Report only for its figure; a process that crashed, hung or stalled still fails the block.
        if (-not (SpeedOk $r)) { $problems += "mixed: $(SpeedWhy $r)" }
        if ($r.stall_lines) { $problems += "mixed: $($r.stall_lines) stall line(s) in its log" }
      }

      # The last measured process has stopped: the link now, before the aggregate that carries the result.
      $link = LinkAfter $block
      $decodeRows = @($runRows | Where-Object { -not $_.long })
      $prefillRows = @($runRows | Where-Object { $_.long })
      $rates = @($decodeRows | Where-Object { $null -ne $_.decode_steps_tok_s } | ForEach-Object { [double]$_.decode_steps_tok_s })
      $fills = @($prefillRows | Where-Object { $null -ne $_.prefill_tok_s } | ForEach-Object { [double]$_.prefill_tok_s })
      $decodeMedian = Median $rates
      $prefillSlowest = $(if ($fills.Count) { ($fills | Measure-Object -Minimum).Minimum } else { $null })
      if ($rates.Count -ne 6) { $problems += "$($rates.Count) of 6 decode runs gave a decode_steps_tok_s" }
      elseif ($decodeMedian -lt $decodeBound) { $problems += "the decode median $decodeMedian tok/s is below $decodeBound" }
      if ($fills.Count -ne 2) { $problems += "$($fills.Count) of 2 prefill runs gave a prefill_tok_s" }
      elseif ($prefillSlowest -lt $prefillBound) { $problems += "the slower prefill $prefillSlowest tok/s is below $prefillBound" }
      $longTokens = @($prefillRows | ForEach-Object { $_.prompt_tokens } | Where-Object { $null -ne $_ })
      if (@($longTokens | Select-Object -Unique).Count -gt 1) { $problems += "the two long runs counted different prompts: $($longTokens -join ', ') tokens" }
      $status = $(if ($problems.Count) { 'fail' } else { 'pass' })
      Record (WithLink @{ test = 'serve-speed'; status = $status; decode_median = $decodeMedian; decode_bound = $decodeBound; prefill_slowest = $prefillSlowest; prefill_bound = $prefillBound
        long_prompt_tokens = $(if ($longTokens.Count) { $longTokens[0] } else { $null }); long_prompt_reference = $longReference; runs = $runRows; desktop_gpu_mib = $desktop
        config_template = $template; template_sha256 = (FileSha $template); overrides = $meta.overrides; artemis_sha256 = $meta.artemis_sha256; problems = $problems } $link)
      if ($status -ne 'pass') { $script:failed += 'serve-speed' }
    } catch {
      $failMeta = $(if ($meta) { $meta } else { ConfigFields $null })
      Record (@{ test = 'serve-speed'; status = 'fail'; error = $_.Exception.Message; runs = $runRows; desktop_gpu_mib = $desktop; config_template = $template; template_sha256 = (FileSha $template); overrides = $failMeta.overrides; artemis_sha256 = $failMeta.artemis_sha256 })
      $script:failed += 'serve-speed'
      throw
    } finally {
      ScanKey $block $keyFile @($speedRunDir, $evidence)
    }
  }

  # The speed gate with speculation on and off: serve-speed's eight
  # runs (seed outermost: write, edit and explain at 1234, then at 1235, then `long` twice), each as an "off" process and
  # then an "on" process, back to back, each its own fresh `artemis serve` from the template with --park-snapshots false.
  # "Off" adds --speculation-enabled false (the server's flag over the template; the module and the template are not
  # edited). The leaves are $runs\serve-speed-spec\<run>-off and <run>-on. Records: a `serve-speed-spec run` per process
  # (the pinned shape: input <run>-<mode>, mode, attempt, status pass or fail, checks as {name: pass}, the verdict's
  # figures, with the six spec_* counts and completion_fnv), a `serve stop` per process, and the block's `serve-speed-spec`
  # aggregate (SpeedSpecJudge). `attempt` is 1 plus the aggregates this commit already holds; one whole rerun is allowed
  # after a miss of done-when 1, 2 or 3, and a third attempt on one commit is refused. A process that fails to answer or
  # to stop gracefully ends the loop: nothing after it runs. Every run's fresh_engine check is hard: a run without it is
  # a problem and fails the block. Binds 127.0.0.1:8090.
  if ($Tests -contains 'serve-speed-spec') {
    $block = 'serve-speed-spec'
    $script:specOnMedian = $null
    $script:specOnStatus = $null
    $script:specOnFnv = $null
    $attempt = SpecAttempt $evidence $commit
    if ($attempt -gt 2) {
      throw "${block}: $evidence already holds $($attempt - 1) attempts at commit $commit, and a third attempt on one commit is refused: a second miss goes to the owner"
    }
    SpeedHygiene $block | Out-Null
    $speedExtra = @('--park-snapshots', 'false')
    RefuseOldKey $block
    $speedRunDir = FreshRunDir $block
    BlockPreflight $block
    if (-not $script:serveBuilt) {
      cargo build --release -p artemis-tools; Check 'cargo build (serve-speed-spec)'
      $script:serveBuilt = $true
    }
    $keyFile = New-ArtemisKey -Dir (Join-Path $runs "keys\$block")

    $specs = @()
    foreach ($seed in @(1234, 1235)) {
      foreach ($id in @('write', 'edit', 'explain')) { $specs += @{ leaf = "$id-$seed"; arg = "${id}:$seed"; prompt = $id } }
    }
    foreach ($i in @(1, 2)) { $specs += @{ leaf = "long-$i"; arg = 'long'; prompt = 'long' } }
    $specRows = @()
    $specProblems = @()
    $specMibs = @()
    $meta = $null
    $stopLoop = $false
    try {
      foreach ($spec in $specs) {
        # Under 5b each of the six decode runs adds a third fresh process, the lookup-only one; `long` does not.
        $modes = @('off', 'on')
        if ($SpecPlan -eq '5b' -and $spec.prompt -ne 'long') { $modes += 'lookup' }
        foreach ($mode in $modes) {
          $extra = $speedExtra
          if ($mode -eq 'off') { $extra = $speedExtra + @('--speculation-enabled', 'false') }
          if ($mode -eq 'lookup') { $extra = $speedExtra + @('--speculation-sources', 'lookup') }
          $run = SpecRun 'serve-speed-spec' $spec $mode $extra 'pass' @{ attempt = $attempt }
          if (-not $meta) { $meta = $run.r.meta }
          $specMibs += $run.r.mib
          $fig = $run.fig
          $specRows += @{ leaf = $spec.leaf; mode = $mode; prompt = $spec.prompt; decode = $fig.decode_steps_tok_s; prefill = $fig.prefill_tok_s
            fnv = $fig.completion_fnv; slots = $run.r.slots_per_layer; prompt_tokens = $fig.prompt_tokens; steps = $fig.spec_steps; mtp = $fig.spec_drafted_mtp; sources = $run.sources }
          $specProblems += $run.problems
          if ($run.stop) {
            $stopLoop = $true
            break
          }
        }
        if ($stopLoop) { break }
      }

      $link = LinkAfter $block
      $agg = WithLink (SpeedSpecJudge $specRows $attempt $specProblems $SpecPlan) $link
      $agg.desktop_gpu_mib = @($specMibs | Where-Object { $null -ne $_ })
      $agg.artemis_sha256 = $meta.artemis_sha256
      Record $agg
      HandOnSpec $agg $specRows
      foreach ($dw in @('dw1', 'dw2', 'dw3')) {
        if (-not $agg.checks[$dw]) {
          $next = $(if ($attempt -eq 1) { 'the block may be rerun once, whole' } else { 'this was the rerun: a second miss goes to the owner' })
          Write-Warning "${block} attempt ${attempt}: done-when $($dw.Substring(2)) missed; $next"
        }
      }
      foreach ($why in $agg.problems) { Write-Warning "${block} attempt ${attempt}: $why" }
      if ($agg.status -ne 'pass') { $script:failed += 'serve-speed-spec' }
    } catch {
      $failMeta = $(if ($meta) { $meta } else { ConfigFields $null })
      Record @{ test = 'serve-speed-spec'; plan = $SpecPlan; status = 'fail'; attempt = $attempt; error = $_.Exception.Message; runs = @($specRows | ForEach-Object { @{ run = $_.leaf; mode = $_.mode; decode_steps_tok_s = $_.decode; prefill_tok_s = $_.prefill; slots_per_layer = $_.slots } })
        desktop_gpu_mib = @($specMibs | Where-Object { $null -ne $_ }); problems = $specProblems; artemis_sha256 = $failMeta.artemis_sha256 }
      $script:failed += 'serve-speed-spec'
      throw
    } finally {
      ScanKey $block $keyFile @($speedRunDir, $evidence)
    }
  }

  # The shrunk-cache report (report only): write, edit and explain at seed 1234
  # with --cache-slots 85, then 30, each as an "off" and then an "on" process (twelve fresh servers, the same flags as
  # serve-speed-spec). The aggregate lists per pair the decode rate, the miss MiB per token, the accept rate, the
  # plain-step share and the on/off ratio; a pair whose "on" is more than 2% slower than "off" is in needs_owner and
  # makes the status ask-owner. A process that fails to answer or stop gracefully, a verdict that is not `ran`, a failed
  # fresh_engine check, or a pair whose "off" speculated or whose "on" did not (SpecModeProblems) fails the block like
  # serve-speed's. Binds 127.0.0.1:8090.
  if ($Tests -contains 'serve-shrunk') {
    $block = 'serve-shrunk'
    SpeedHygiene $block | Out-Null
    $speedExtra = @('--park-snapshots', 'false')
    RefuseOldKey $block
    $speedRunDir = FreshRunDir $block
    BlockPreflight $block
    if (-not $script:serveBuilt) {
      cargo build --release -p artemis-tools; Check 'cargo build (serve-shrunk)'
      $script:serveBuilt = $true
    }
    $keyFile = New-ArtemisKey -Dir (Join-Path $runs "keys\$block")

    $specs = @()
    foreach ($slots in @(85, 30)) {
      foreach ($id in @('write', 'edit', 'explain')) { $specs += @{ leaf = "$id-1234-s$slots"; arg = "${id}:1234"; prompt = $id; slots = $slots } }
    }
    $shrunkRows = @()
    $shrunkProblems = @()
    $shrunkMibs = @()
    $meta = $null
    $stopLoop = $false
    try {
      foreach ($spec in $specs) {
        foreach ($mode in @('off', 'on')) {
          $extra = $speedExtra + @('--cache-slots', "$($spec.slots)")
          if ($mode -eq 'off') { $extra += @('--speculation-enabled', 'false') }
          $run = SpecRun 'serve-shrunk' $spec $mode $extra 'ran' @{ slots = $spec.slots }
          if (-not $meta) { $meta = $run.r.meta }
          $shrunkMibs += $run.r.mib
          $fig = $run.fig
          $shrunkRows += @{ leaf = $spec.leaf; prompt = $spec.prompt; slots = $spec.slots; mode = $mode; tok = $fig.decode_steps_tok_s; miss = $fig.miss_mib_per_token
            accept = $fig.spec_accept_rate; steps = $fig.spec_steps; plain = $fig.spec_plain_steps }
          $shrunkProblems += $run.problems
          if ($run.stop) {
            $stopLoop = $true
            break
          }
        }
        if ($stopLoop) { break }
      }

      $link = LinkAfter $block
      $j = ShrunkJudge $shrunkRows
      $shrunkProblems += $j.problems
      $shrunkStatus = $(if ($shrunkProblems.Count) { 'fail' } elseif ($j.needs_owner.Count) { 'ask-owner' } else { 'report' })
      $shrunkRec = WithLink @{ test = 'serve-shrunk'; plan = $SpecPlan; status = $shrunkStatus; pairs = $j.pairs; needs_owner = $j.needs_owner; problems = $shrunkProblems
        desktop_gpu_mib = @($shrunkMibs | Where-Object { $null -ne $_ }); artemis_sha256 = $meta.artemis_sha256 } $link
      Record $shrunkRec
      $shrunkStatus = $shrunkRec.status
      foreach ($n in $j.needs_owner) {
        Write-Warning "${block}: $($n.prompt) at $($n.slots) slots runs at $($n.ratio) of off (below 0.98): ask the owner"
      }
      foreach ($why in $shrunkProblems) { Write-Warning "${block}: $why" }
      if ($shrunkStatus -eq 'fail') { $script:failed += 'serve-shrunk' }
    } catch {
      $failMeta = $(if ($meta) { $meta } else { ConfigFields $null })
      Record @{ test = 'serve-shrunk'; plan = $SpecPlan; status = 'fail'; error = $_.Exception.Message; problems = $shrunkProblems; desktop_gpu_mib = @($shrunkMibs | Where-Object { $null -ne $_ }); artemis_sha256 = $failMeta.artemis_sha256 }
      $script:failed += 'serve-shrunk'
      throw
    } finally {
      ScanKey $block $keyFile @($speedRunDir, $evidence)
    }
  }

  if ($Tests -contains 'speed') {
    SpeedHygiene 'speed' | Out-Null
    $dir = Join-Path $runs 'speed'
    New-Item -ItemType Directory -Force $dir | Out-Null
    Remove-Item -Force -ErrorAction SilentlyContinue "$dir\*.json"
    $speedArgs = @()
    $desktop = @()
    # Each generate runs through RunMeasured, which reads the process's CPU mask after the start and before the stop.
    $speedMasks = @{}
    $speedAff = [System.Collections.Generic.List[string]]::new()
    # A failed generate may be a device loss: record the failing speed line, then stop as Check does.
    function SpeedFail([string]$what, [int]$exit) {
      Record @{ test = 'speed'; status = 'fail'; gated = [bool]$SpeedGate; error = "$what failed (exit $exit)"; desktop_gpu_mib = $desktop; masks = $speedMasks }
      $script:failed += 'speed'
      throw "$what failed (exit $exit)"
    }
    foreach ($id in @('write', 'edit', 'explain')) {
      foreach ($seed in @(1234, 1235)) {
        $json = Join-Path $dir "$id-$seed.json"
        $desktop += @{ run = "$id-$seed"; mib = (DesktopGpuMib) }
        $gen = RunMeasured $artemis @('generate', '--model', $Model, '--tokens-file', $speedTokens[$id], '--max-tokens', '700', '--temp', '0.6', '--top-p', '0.95', '--top-k', '20', '--min-p', '0', '--seed', "$seed", '--ctx', '65536', '--json', $json, '--no-fingerprints') "$dir\$id-$seed.txt" "$dir\$id-$seed.err"
        MaskNote $gen "$id-$seed" $speedMasks $speedAff
        if ($gen.exit) { SpeedFail "generate $id seed $seed" $gen.exit }
        $speedArgs += @('--decode', "$id=$json")
      }
    }
    # Decode's base time: every HOST expert read from a resident one. Reported, never gated.
    $baseJson = Join-Path $dir 'base.json'
    $desktop += @{ run = 'base'; mib = (DesktopGpuMib) }
    $gen = RunMeasured $artemis @('generate', '--model', $Model, '--tokens-file', $speedTokens['write'], '--max-tokens', '700', '--temp', '0.6', '--top-p', '0.95', '--top-k', '20', '--min-p', '0', '--seed', '1234', '--ctx', '65536', '--json', $baseJson, '--no-fingerprints', '--base-time', '--speculation', 'off') "$dir\base.txt" "$dir\base.err"
    MaskNote $gen 'base' $speedMasks $speedAff
    $baseArg = @('--base', $baseJson)
    $baseWarn = @()
    if ($gen.exit) {
      $baseArg = @()
      $baseErr = if (Test-Path "$dir\base.err") { Get-Content "$dir\base.err" -Tail 1 } else { $null }
      $baseWarn = @("generate --base-time failed (exit $($gen.exit)): no base time" + $(if ($baseErr) { "; its last error line: $baseErr" } else { '' }))
    }
    foreach ($i in @(1, 2)) {
      $json = Join-Path $dir "prefill-$i.json"
      $desktop += @{ run = "prefill-$i"; mib = (DesktopGpuMib) }
      $gen = RunMeasured $artemis @('generate', '--model', $Model, '--tokens-file', $long, '--max-tokens', '1', '--greedy', '--ctx', '65536', '--json', $json, '--no-fingerprints') "$dir\prefill-$i.txt" "$dir\prefill-$i.err"
      MaskNote $gen "prefill-$i" $speedMasks $speedAff
      if ($gen.exit) { SpeedFail "generate prefill run $i" $gen.exit }
      $speedArgs += @('--prefill', $json)
    }
    $verdict = Join-Path $dir 'verdict.json'
    $gateArg = @(if ($SpeedGate) { '--gate' })
    & $artemis speed @speedArgs --made-from (Join-Path $speedDir 'made-from.json') --long-tokens $long @gateArg @baseArg --out $verdict
    $speedExit = $LASTEXITCODE
    $link = LinkAfter 'speed'
    # The desktop's own GPU memory per run (the driver's held-back figure is a fixed reserve and can't show it).
    $deskWarn = @($baseWarn)
    $read = @($desktop | Where-Object { $null -ne $_.mib })
    if ($read.Count -lt $desktop.Count) {
      $deskWarn += "the desktop's GPU memory could not be read before $($desktop.Count - $read.Count) of $($desktop.Count) runs"
    }
    if ($read.Count) {
      $lo = $read | Sort-Object { $_.mib } | Select-Object -First 1
      $hi = $read | Sort-Object { $_.mib } -Descending | Select-Object -First 1
      if ($hi.mib - $lo.mib -gt 500) {
        $deskWarn += "the desktop's GPU memory ranged from $($lo.mib) MiB ($($lo.run)) to $($hi.mib) MiB ($($hi.run)) during the gate"
      }
    }
    if (-not (Test-Path $verdict)) {
      $speedRec = WithLink @{ test = 'speed'; status = 'fail'; gated = [bool]$SpeedGate; problems = @(@('no verdict was written') + @($speedAff)); desktop_gpu_mib = $desktop; warnings = $deskWarn; masks = $speedMasks } $link
      if ($speedAff.Count) { AddProblem $speedRec 'affinity' }
      Record $speedRec
      $script:failed += 'speed'
    } else {
      $v = Get-Content $verdict -Raw | ConvertFrom-Json
      $speedRec = WithLink @{ test = 'speed'; status = $v.status; gated = [bool]$SpeedGate; decode_median = $v.decode_median; decode_bound = $v.decode_bound; prefill_slowest = $v.prefill_slowest; prefill_bound = $v.prefill_bound; decode = $v.decode; prefill = $v.prefill; base = $v.base; problems = @(@($v.problems) + @($speedAff) | Where-Object { $_ }); warnings = @(@($v.warnings) + $deskWarn | Where-Object { $_ }); desktop_gpu_mib = $desktop; masks = $speedMasks } $link
      if ($speedAff.Count) { $speedRec.status = 'fail'; AddProblem $speedRec 'affinity' }
      Record $speedRec
      if ($speedExit -or $speedAff.Count) { $script:failed += 'speed' }
    }
  }

  if ($Tests -contains 'profile') {
    SpeedHygiene 'profile' | Out-Null
    $dir = Join-Path $runs 'profile'
    New-Item -ItemType Directory -Force $dir | Out-Null
    Remove-Item -Force -ErrorAction SilentlyContinue "$dir\*.json"
    $profMasks = @{}
    $profAff = [System.Collections.Generic.List[string]]::new()
    foreach ($id in @('write', 'edit', 'explain')) {
      $p = Join-Path $dir "$id.json"
      $gen = RunMeasured $artemis @('generate', '--model', $Model, '--tokens-file', $speedTokens[$id], '--max-tokens', '700', '--seed', '1234', '--ctx', '65536', '--profile', $p) "$dir\$id.txt" "$dir\$id.err"
      MaskNote $gen $id $profMasks $profAff
      if ($gen.exit) { throw "generate --profile $id failed (exit $($gen.exit))" }
      $r = Get-Content $p -Raw | ConvertFrom-Json
      $steps = [math]::Max(1, $r.decode.steps)
      Record @{ test = 'profile (report only)'; input = $id; decode_steps = $r.decode.steps; verify_steps = $r.verify.steps; verify_tokens = $r.verify.tokens; verify_gpu_ms = $r.verify.gpu_us / 1000; speculation = ($r.verify.steps -gt 0); decode_gpu_ms_per_token = $r.decode.gpu_us / 1000 / $steps; decode_cpu_us_per_token = @{ prepare = $r.decode.cpu.prepare_us / $steps; record = $r.decode.cpu.record_us / $steps; submit = $r.decode.cpu.submit_us / $steps; read = $r.decode.cpu.read_us / $steps; sample = $r.decode.cpu.sample_us / $steps }; top_kernels = @($r.decode.kernels | Select-Object -First 10); prefill_gpu_ms = $r.prefill.gpu_us / 1000; masks = $profMasks[$id] }
    }
    $p = Join-Path $dir 'long-prefill.json'
    $gen = RunMeasured $artemis @('generate', '--model', $Model, '--tokens-file', $long, '--max-tokens', '1', '--greedy', '--ctx', '65536', '--profile', $p) "$dir\long-prefill.txt" "$dir\long-prefill.err"
    MaskNote $gen 'long-prefill' $profMasks $profAff
    if ($gen.exit) { throw "generate --profile on the long prompt failed (exit $($gen.exit))" }
    $r = Get-Content $p -Raw | ConvertFrom-Json
    Record @{ test = 'profile (report only)'; input = 'long-prefill'; masks = $profMasks['long-prefill']; prefill_steps = $r.prefill.steps; prefill_tokens = $r.prefill.tokens; prefill_gpu_ms = $r.prefill.gpu_us / 1000; prefill_cpu_us = $r.prefill.cpu; prefill_copies = $r.prefill.copies; top_kernels = @($r.prefill.kernels | Select-Object -First 12) }
    LinkAfter 'profile' | Out-Null
    foreach ($why in $profAff) { Write-Warning "profile: affinity: $why" }
    if ($profAff.Count) { $script:failed += 'profile affinity' }
  }

  if ($Tests -contains 'depth') {
    SpeedHygiene 'depth' | Out-Null
    # Decode at long contexts (reported, not gated): the long golden prompt cut at each depth, then its next 128
    # tokens forced one a step (the first from prefill's logits), so every run decodes 127 tokens there. Greedy, so no
    # sampling time.
    $dir = Join-Path $runs 'depth'
    New-Item -ItemType Directory -Force $dir | Out-Null
    Remove-Item -Force -ErrorAction SilentlyContinue "$dir\*.json"
    $all = [IO.File]::ReadAllBytes($long)
    $depthAff = [System.Collections.Generic.List[string]]::new()
    foreach ($n in @(8192, 16384, 27000)) {
      $tf = Join-Path $dir "prompt-$n.tokens"
      $ff = Join-Path $dir "force-$n.tokens"
      [IO.File]::WriteAllBytes($tf, $all[0..(4 * $n - 1)])
      [IO.File]::WriteAllBytes($ff, $all[(4 * $n)..(4 * ($n + 128) - 1)])
      $json = Join-Path $dir "depth-$n.json"
      $mib = DesktopGpuMib
      $gen = RunMeasured $artemis @('generate', '--model', $Model, '--tokens-file', $tf, '--greedy', '--force-tokens', $ff, '--max-tokens', '128', '--ctx', '65536', '--json', $json, '--no-fingerprints') "$dir\depth-$n.txt" "$dir\depth-$n.err"
      $depthMasks = @{}
      MaskNote $gen "depth-$n" $depthMasks $depthAff
      if ($gen.exit) { throw "generate at depth $n failed (exit $($gen.exit))" }
      $r = Get-Content $json -Raw | ConvertFrom-Json
      $dr = $r.cache.decode_hits + $r.cache.decode_misses
      Record @{ test = 'depth (report only)'; input = "$n"; status = $(if ($gen.affinity_problem) { 'fail' } else { 'ran' }); masks = $depthMasks; decoded = $r.timings.decoded; seconds = $r.timings.generate_s; tok_s = $r.timings.decoded / $r.timings.generate_s; decode_hit_pct = 100 * $r.cache.decode_hits / [math]::Max(1, $dr); longest_submit_ms = $r.cache.longest_submit_ms; slots = $r.settings.cache_slots; desktop_gpu_mib = $mib }
    }
    LinkAfter 'depth' | Out-Null
    foreach ($why in $depthAff) { Write-Warning "depth: affinity: $why" }
    if ($depthAff.Count) { $script:failed += 'depth affinity' }
  }

  if ($Tests -contains 'attn-depth') {
    SpeedHygiene 'attn-depth' | Out-Null
    # The current build against a pinned one at 4,096, 27,000 and 65,000 positions: the long prompt cut at each depth
    # (at 65,000, its tokens repeated three times), the next 128 forced, 127 decoded, greedy, no speculation. Per depth:
    # base with fingerprints, new with fingerprints, then base, new, base, new timed without them.
    $dir = Join-Path $runs 'attn-depth'
    New-Item -ItemType Directory -Force $dir | Out-Null
    Remove-Item -Force -ErrorAction SilentlyContinue "$dir\*.json"
    $all = [IO.File]::ReadAllBytes($long)
    $stream = $all + $all + $all
    $baseSha = (Get-FileHash -LiteralPath $BaselineArtemis -Algorithm SHA256).Hash.ToLowerInvariant()
    $newSha = (Get-FileHash -LiteralPath $artemis -Algorithm SHA256).Hash.ToLowerInvariant()
    $attnRows = @()
    foreach ($n in @(4096, 27000, 65000)) {
      $tf = Join-Path $dir "prompt-$n.tokens"
      $ff = Join-Path $dir "force-$n.tokens"
      [IO.File]::WriteAllBytes($tf, $stream[0..(4 * $n - 1)])
      [IO.File]::WriteAllBytes($ff, $stream[(4 * $n)..(4 * ($n + 128) - 1)])
      $results = @{}
      $masks = @{}
      $affProblems = @()
      foreach ($step in @(@('base', 'fp'), @('new', 'fp'), @('base', 't1'), @('new', 't1'), @('base', 't2'), @('new', 't2'))) {
        $side = $step[0]; $kind = $step[1]
        $exe = $(if ($side -eq 'base') { $BaselineArtemis } else { $artemis })
        $tag = "$side-$kind-$n"
        $json = Join-Path $dir "$tag.json"
        $extra = $(if ($kind -eq 'fp') { @() } else { @('--no-fingerprints') })
        $gen = RunMeasured $exe (@('generate', '--model', $Model, '--tokens-file', $tf, '--greedy', '--force-tokens', $ff, '--max-tokens', '128', '--ctx', '65536', '--speculation', 'off', '--json', $json) + $extra) "$dir\$tag.txt" "$dir\$tag.err"
        if ($gen.exit) { throw "generate $tag failed (exit $($gen.exit))" }
        $masks[$tag] = AffinityFields $gen
        if ($gen.affinity_problem) { $affProblems += "${tag}: $($gen.affinity_problem)" }
        $results["$side-$kind"] = Get-Content $json -Raw | ConvertFrom-Json
      }
      $rate = { param($r) $r.timings.decoded / $r.timings.generate_s }
      $baseTok = ((& $rate $results['base-t1']) + (& $rate $results['base-t2'])) / 2
      $newTok = ((& $rate $results['new-t1']) + (& $rate $results['new-t2'])) / 2
      $equal = SamePositions $results['base-fp'].positions $results['new-fp'].positions
      $row = @{ depth = $n; base_tok_s = $baseTok; new_tok_s = $newTok; fingerprints_equal = $equal; affinity_problem = $(if ($affProblems.Count) { $affProblems -join '; ' } else { $null }) }
      $attnRows += $row
      Record @{ test = 'attn-depth run'; input = "$n"; status = $(if ($affProblems.Count) { 'fail' } else { 'ran' }); masks = $masks; stage = $AttnStage; base_tok_s = $baseTok; new_tok_s = $newTok; ratio = $newTok / $baseTok
        fingerprints_equal = $equal; decoded = $results['new-t1'].timings.decoded; baseline = $BaselineArtemis; baseline_sha256 = $baseSha; new_sha256 = $newSha }
    }
    $link = LinkAfter 'attn-depth'
    $v = AttnDepthJudge $attnRows $AttnStage
    $attnRec = WithLink @{ test = 'attn-depth'; status = $v.status; stage = $AttnStage; gain_65k = $v.gain_65k; problems = $v.problems; baseline = $BaselineArtemis; baseline_sha256 = $baseSha; new_sha256 = $newSha
      depths = @($attnRows | ForEach-Object { @{ depth = $_.depth; ratio = $_.new_tok_s / $_.base_tok_s; fingerprints_equal = $_.fingerprints_equal } }) } $link
    if (@($attnRows | Where-Object { $_.affinity_problem }).Count) { AddProblem $attnRec 'affinity' }
    Record $attnRec
    if ($attnRec.status -ne 'pass') { $script:failed += 'attn-depth' }
  }

  if ($Tests -contains 'spec-long') {
    SpeedHygiene 'spec-long' | Out-Null
    # Speculation at a long context (report only): the first 20,000 tokens of the long
    # golden prompt (cut in bytes, as `depth` does) with no appended question, then 512 sampled tokens at seed 1234
    # (0.6/0.95/20/0, --ctx 65536), once with speculation off and once with it on. Records decode tok/s, the decode
    # misses, the speculation counts (the accept rate and the plain-step share are worked out from them) and whether the
    # two completions' tokens are identical. No fingerprints: they are timed. A failed run is recorded and stops the block.
    $dir = Join-Path $runs 'spec-long'
    New-Item -ItemType Directory -Force $dir | Out-Null
    Remove-Item -Force -ErrorAction SilentlyContinue "$dir\*.json"
    $longAll = [IO.File]::ReadAllBytes((Join-Path $repo 'testdata\golden\prompts\20-long-option-rs.tokens'))
    $longPrompt = Join-Path $dir 'prompt-20000.tokens'
    [IO.File]::WriteAllBytes($longPrompt, $longAll[0..(4 * 20000 - 1)])
    $longRuns = @()
    $longTokens = @{}
    $longPromptTokens = $null
    $longMasks = @{}
    $longAff = [System.Collections.Generic.List[string]]::new()
    # Under 5b a third run, report only: "on" with the MTP's window over the whole prompt (the default window is 2,048
    # of the 20,000 positions) and an MTP KV that holds them (--mtp-window 20000, --mtp-ctx 65536; the window is at most
    # half the context).
    $longModes = @(@{ name = 'off'; speculation = 'off'; extra = @() }, @{ name = 'on'; speculation = 'on'; extra = @() })
    if ($SpecPlan -eq '5b') { $longModes += @{ name = 'on-full-window'; speculation = 'on'; extra = @('--mtp-window', '20000', '--mtp-ctx', '65536') } }
    foreach ($lm in $longModes) {
      $mode = $lm.name
      $longExtra = @($lm.extra)
      $json = Join-Path $dir "$mode.json"
      $gen = RunMeasured $artemis (@('generate', '--model', $Model, '--tokens-file', $longPrompt, '--max-tokens', '512', '--temp', '0.6', '--top-p', '0.95', '--top-k', '20', '--min-p', '0', '--seed', '1234', '--ctx', '65536', '--speculation', $lm.speculation) + $longExtra + @('--json', $json, '--no-fingerprints')) "$dir\$mode.txt" "$dir\$mode.err"
      MaskNote $gen $mode $longMasks $longAff
      if ($gen.exit) {
        Record @{ test = 'spec-long (report only)'; plan = $SpecPlan; status = 'fail'; error = "generate ($mode) failed (exit $($gen.exit)): $(Get-Content "$dir\$mode.err" -ErrorAction SilentlyContinue | Select-Object -Last 1)"; masks = $longMasks }
        $script:failed += 'spec-long'
        throw "generate (spec-long, $mode) failed (exit $($gen.exit))"
      }
      $r = Get-Content $json -Raw | ConvertFrom-Json
      $longPromptTokens = @($r.prompt_tokens).Count
      $longTokens[$mode] = (@($r.steps | ForEach-Object { $_.token }) -join ',')
      $spec = $null
      if ($r.spec) {
        $spec = @{ steps = $r.spec.steps; plain_steps = $r.spec.plain_steps; drafted = $r.spec.drafted; accepted = $r.spec.accepted; replays = $r.spec.replays }
        if ($SpecPlan -eq '5b') {
          foreach ($k in @('drafted_lookup', 'accepted_lookup', 'drafted_mtp', 'accepted_mtp', 'probes')) { $spec[$k] = $r.spec.$k }
        }
      }
      $longRuns += @{ mode = $mode; decode_tok_s = $(if ($r.timings.generate_s -gt 0) { $r.timings.decoded / $r.timings.generate_s } else { $null })
        decoded = $r.timings.decoded; decode_misses = $r.cache.decode_misses; spec = $spec }
    }
    # Which mode each run really was (as SpecModeProblems asks of the server blocks): the "off" run has no spec counts, the
    # "on" run has some and made a verify step. Otherwise identical_tokens could come from two plain runs.
    $link = LinkAfter 'spec-long'
    $longProblems = @()
    foreach ($why in $longAff) { $longProblems += "affinity: $why" }
    if ($longRuns[0].spec) { $longProblems += 'the "off" run has speculation counts, so it speculated' }
    if (-not $longRuns[1].spec) { $longProblems += 'the "on" run has no speculation counts, so it did not speculate' }
    if ($longRuns.Count -gt 2 -and -not $longRuns[2].spec) { $longProblems += 'the "on-full-window" run has no speculation counts, so it did not speculate' }
    # No verify step is not a mode error here: at 20,000 positions the controller may rightly choose no draft throughout.
    # It is recorded as a warning and the block still ran.
    $longWarnings = @()
    if ($longRuns[1].spec -and -not $longRuns[1].spec.steps) { $longWarnings += 'the "on" run made no verify step (the controller drafted nothing)' }
    $longRec = WithLink @{ test = 'spec-long (report only)'; plan = $SpecPlan; status = $(if ($longProblems.Count) { 'fail' } else { 'ran' }); prompt_tokens = $longPromptTokens; runs = $longRuns
      identical_tokens = ($longTokens['off'] -ceq $longTokens['on']); problems = $longProblems; warnings = $longWarnings; masks = $longMasks } $link
    if ($longAff.Count) { AddProblem $longRec 'affinity' }
    Record $longRec
    foreach ($why in $longProblems) { Write-Warning "spec-long: $why" }
    foreach ($why in $longWarnings) { Write-Warning "spec-long: $why" }
    if ($longProblems.Count) { $script:failed += 'spec-long' }
  }

  # The 32 GB check: `artemis generate` at 8 tokens, speculation off and then on with MTP, each in
  # its own process; the record holds each run's memory block and its pool_at_26gib in whole GiB as pool_gib. Judged by
  # Mtp32Judge: pass, ask-owner, or fail (not measured, MTP not loaded, wrong host-byte growth). The pool is a machine
  # with 26 GiB free's, whatever this machine has free (free_before_gib is the figure measured before the first load).
  if ($Tests -contains 'mtp-32gb') {
    $dir = Join-Path $runs 'mtp-32gb'
    New-Item -ItemType Directory -Force $dir | Out-Null
    $freeGib = $null
    try { $freeGib = [math]::Round((Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).FreePhysicalMemory / 1MB, 1) } catch { }
    $m32 = @()
    foreach ($mode in @('off', 'on')) {
      $json = Join-Path $dir "$mode.json"
      if (Test-Path $json) { Remove-Item $json }
      $modeArgs = @(if ($mode -eq 'on') { '--speculation'; 'on'; '--sources'; 'lookup,mtp' } else { '--speculation'; 'off' })
      & $artemis generate --model $Model --tokens-file (Join-Path $repo 'testdata\golden\prompts\01-code.tokens') --greedy --max-tokens 8 --ctx 4096 @modeArgs --json $json --no-fingerprints 2> "$dir\$mode.err" > "$dir\$mode.txt"
      if ($LASTEXITCODE) {
        Record @{ test = 'mtp-32gb'; plan = $SpecPlan; status = 'fail'; error = "generate ($mode) failed (exit $LASTEXITCODE): $(Get-Content "$dir\$mode.err" -ErrorAction SilentlyContinue | Select-Object -Last 1)" }
        $script:failed += 'mtp-32gb'
        throw "generate (mtp-32gb, $mode) failed (exit $LASTEXITCODE)"
      }
      $mem = (Get-Content $json -Raw | ConvertFrom-Json).memory
      $pool = $(if (IsNum $mem.pool_at_26gib) { [long][math]::Floor($mem.pool_at_26gib / 1GB) } else { $null })
      $m32 += @{ mode = $mode; host_bytes = $mem.host_bytes; host_heap_usage = $mem.host_heap_usage; mtp_vram_bytes = $mem.mtp_vram_bytes; pool_gib = $pool }
    }
    $j32 = Mtp32Judge $m32[0] $m32[1]
    Record @{ test = 'mtp-32gb'; plan = $SpecPlan; status = $j32.status; free_before_gib = $freeGib; runs = $m32; why = $j32.why }
    if ($j32.status -eq 'fail') { $script:failed += 'mtp-32gb' }
    if ($j32.why) { Write-Warning "mtp-32gb: $($j32.why)" }
  }

  if ($Tests -contains 'snapshot-times') {
    # The snapshot copy measurement (reported, not gated): the PCIe link's rates, the state and the
    # KV to the host and back, a restore after a dirtying, and the long prompt's prefill with and without snapshots.
    # The command's own exit code is the only gate: a run that can't measure is a fail.
    $golden = Join-Path $repo 'testdata\golden'
    $dir = Join-Path $runs 'snapshot-times'
    New-Item -ItemType Directory -Force $dir | Out-Null
    $times = Join-Path $dir 'times.json'
    if (Test-Path $times) { Remove-Item $times }
    $mib = DesktopGpuMib
    & $artemis snapshot-times --model $Model --golden $golden --json $times 2> "$dir\run.err" > "$dir\run.txt"
    if ($LASTEXITCODE) {
      Record @{ test = 'snapshot-times (report only)'; status = 'fail'; error = (Get-Content "$dir\run.err" | Select-Object -Last 1); desktop_gpu_mib = $mib }
      $script:failed += 'snapshot-times'
    } else {
      $r = Get-Content $times -Raw | ConvertFrom-Json
      Record @{
        test = 'snapshot-times (report only)'; status = 'ran'
        link = $r.link
        first_tok_s = $r.first_tok_s; plain_tok_s = $r.plain_tok_s; overlapped_tok_s = $r.segmented.overlapped_tok_s
        cost_vs_plain_pct = $r.segmented.cost_vs_plain_pct
        whole_restore_after_dirty_ms = $r.whole.restore_after_dirty_ms; chain_restore_after_dirty_ms = $r.segmented.restore_after_dirty_ms
        chain_len = $r.segmented.chain_len
        take_state_ms = $r.whole.take_state_ms; take_kv_ms = $r.whole.take_kv_ms; restore_state_only_ms = $r.whole.restore_state_only_ms
        state_ms = @($r.segmented.waited | ForEach-Object { $_.state_ms })
        desktop_gpu_mib = $mib
      }
    }
  }

  if ($Tests -contains 'llama-search') {
    # Report only. A best.json it writes is the configuration speed-llama and speed-llama-65k run below.
    $script:llamaSearchBest = $null
    $driverNow = GpuDriver
    $buildNow = LlamaBuild
    if ($LlamaConfig -and (SearchedConfigCurrent $LlamaConfig $driverNow $buildNow)) {
      Write-Warning "-LlamaConfig '$LlamaConfig' is a searched result taken with today's GPU driver ($driverNow) and llama.cpp build, so this search can be skipped: drop llama-search from -Tests to run that configuration. This search writes its own best.json and does not change that file (about $LlamaSearchMinutes minutes of screening, then the finalists)"
    }
    SpeedHygiene 'llama-search' | Out-Null
    $rec = MeasureLlamaSearch (Join-Path $runs 'llama-search') $driverNow $buildNow
    $link = LinkAfter 'llama-search'
    $rec.test = 'llama-search (report only)'
    if ($rec.status -eq 'fail') { $script:failed += 'llama-search' }
    $rec = WithLink $rec $link
    HandOnLlamaSearch $rec
    Record $rec
  }

  if ($Tests -contains 'speed-llama') {
    # The configuration is chosen once, so speed-llama-65k below runs the same one.
    $script:llamaChoice = LlamaChoice
    SpeedHygiene 'speed-llama' | Out-Null
    $rec = MeasureLlama (Join-Path $runs 'speed-llama') $script:llamaChoice 0 $false
    $script:llamaBest = $null
    $link = LinkAfter 'speed-llama'
    $rec.test = 'speed-llama (report only)'
    if ($rec.status -eq 'fail') { $script:failed += 'speed-llama' }
    $rec = WithLink $rec $link
    # For the first-run block: the figure, the device and mask it ran with, and which configuration it was; only a
    # record that passed (a dropped link fails it) is one.
    if ($rec.status -ne 'fail') {
      $script:llamaBest = @{ median = $rec.median; device = $rec.device; affinity = $rec.affinity; config_source = $rec.config_source }
    }
    Record $rec
  }

  if ($Tests -contains 'speed-llama-65k') {
    # Report only: the same configuration at 65,536 positions, the like-for-like figure beside Artemis's. A server that
    # can't load at that context is a figure too (`error`, no median), not a failure.
    if (-not $script:llamaChoice) { $script:llamaChoice = LlamaChoice }
    SpeedHygiene 'speed-llama-65k' | Out-Null
    $rec = MeasureLlama (Join-Path $runs 'speed-llama-65k') $script:llamaChoice 65536 $true
    $link = LinkAfter 'speed-llama-65k'
    $rec.test = 'speed-llama-65k (report only)'
    $rec.ctx = 65536
    if ($rec.status -eq 'fail') { $script:failed += 'speed-llama-65k' }
    Record (WithLink $rec $link)
  }

  # Plan 6b's first-run gate, after serve-speed-spec and speed-llama of the same invocation (their medians are this
  # block's references: $script:specOnMedian, $script:llamaBest). ARTEMIS_TUNED is `off` for the whole runner; here it
  # is a tuned.toml in this block's own folder, so the first launch below is a first launch (no tuned file: `artemis
  # serve` runs setup before it opens the port) and writes only there. The launch is started by the block itself (setup
  # takes minutes, and Start-ArtemisServer waits 60 s for /health) with the argument list Start-ArtemisServer -WhatIf
  # builds, and stopped by closing its stdin. Then the six decode runs of serve-speed-spec, each a fresh process in mode
  # "tuned" (the file is there now, so `serve` only reads it). Records: `first-run launch`, a `serve stop` for it, six
  # `first-run run` (SpecRun's shape plus tuned_applied) and the `first-run` aggregate (FirstRunJudge). `attempt` is 1
  # plus the `first-run` aggregates this commit already holds, and a third is refused. Binds 127.0.0.1:8090.
  if ($Tests -contains 'first-run') {
    $block = 'first-run'
    $attempt = FirstRunAttempt $evidence $commit
    if ($attempt -gt 2) {
      throw "${block}: $evidence already holds $($attempt - 1) attempts at commit $commit, and a third attempt on one commit is refused: a second miss goes to the owner"
    }
    # A failure the inputs already decide is not an attempt (a `first-run` aggregate would count as one): a missing
    # reference, or a llama.cpp figure that is not a searched best on the -GpuName device.
    $why = FirstRunPrecheck $script:specOnMedian $script:llamaBest $GpuName
    if ($why) {
      Record @{ test = 'first-run precheck'; plan = $SpecPlan; status = 'fail'; error = $why; has_spec_on_median = ($null -ne $script:specOnMedian); has_llama_best = ($null -ne $script:llamaBest) }
      $script:failed += 'first-run'
      Write-Warning "${block}: $why"
    } else {
      SpeedHygiene $block | Out-Null
      $speedExtra = @('--park-snapshots', 'false')
      RefuseOldKey $block
      $speedRunDir = FreshRunDir $block
      BlockPreflight $block
      if (-not $script:serveBuilt) {
        cargo build --release -p artemis-tools; Check 'cargo build (first-run)'
        $script:serveBuilt = $true
      }
      $keyFile = New-ArtemisKey -Dir (Join-Path $runs "keys\$block")
      $tunedFile = Join-Path $speedRunDir 'tuned.toml'
      $launchDir = Join-Path $speedRunDir 'launch'
      $launchLogs = Join-Path $launchDir 'logs'

      $specs = @()
      foreach ($seed in @(1234, 1235)) {
        foreach ($id in @('write', 'edit', 'explain')) { $specs += @{ leaf = "$id-$seed"; arg = "${id}:$seed"; prompt = $id } }
      }
      $frRows = @()
      $frProblems = @()
      $frMibs = @()
      $frMeta = $null
      $launchInfo = @{ health = $false; reason = 'the launch did not start'; setup = $null; network_imports = @(); affinity_problem = $null }
      $envBefore = $env:ARTEMIS_TUNED
      try {
        $env:ARTEMIS_TUNED = $tunedFile
        # --- the first launch: setup runs before the port opens
        New-Item -ItemType Directory -Force $launchLogs | Out-Null
        $filled = New-ArtemisConfig -Template $template -Out (Join-Path $launchDir 'artemis.toml') -Model $Model -KeyFile $keyFile -LogDir $launchLogs
        $argv = @(Start-ArtemisServer -Exe $artemis -Model $Model -Name "$block-launch" -RunDir $launchDir -KeyFile $keyFile -Config $template -Extra $speedExtra -WhatIf)
        if ($argv.Count -lt 2 -or $argv[1] -ne 'serve') { throw "${block}: Start-ArtemisServer -WhatIf gave no argument list ($($argv -join ' '))" }
        $frMeta = @{ config_template = $template; template_sha256 = $filled.template_sha256; config_sha256 = $filled.sha256; overrides = @{ extra = $speedExtra }
          artemis_sha256 = $(if (Test-Path -LiteralPath $artemis) { FileSha $artemis } else { $null }) }
        $imports = @(Get-ArtemisPeImports $artemis)
        $netImports = @(NetworkImports $imports)
        $frMibs += DesktopGpuMib
        $launch = FirstRunLaunch $argv[0] @($argv[1..($argv.Count - 1)]) $launchDir 8090 $script:LaunchWaitS
        $setupLines = @(LogEvents $launchLogs 'setup_done')
        $setupFailed = @(LogEvents $launchLogs 'setup_failed')
        $setupDone = $(if ($setupLines.Count) { $setupLines[-1] } else { $null })
        $startText = $false
        if (Test-Path -LiteralPath $launch.stdout) {
          $outText = Get-Content -LiteralPath $launch.stdout -Raw
          $startText = [bool]($outText -and $outText.Contains('First run: Artemis is measuring this PC', [System.StringComparison]::Ordinal))
        }
        $launchStall = StallLines $launchLogs
        $stop = $launch.stop
        $launchStopped = [bool]($stop -and $stop.graceful -and $stop.port_free)
        $launchGood = [bool]($launch.health -and $setupDone -and -not $launch.affinity_problem -and $launchStopped -and -not $launchStall)
        $launchRec = @{ test = 'first-run launch'; plan = $SpecPlan; input = 'first-run launch'; attempt = $attempt; status = $(if ($launchGood) { 'pass' } else { 'fail' })
          health = $launch.health; first_launch_s = $launch.first_launch_s; setup = $setupDone; start_text = $startText
          setup_failed = $(if ($setupFailed.Count) { $setupFailed[-1].message } else { $null })
          tuned_file = $tunedFile; tuned_file_written = (Test-Path -LiteralPath $tunedFile)
          imports = $imports; network_imports = $netImports; stall_lines = $launchStall
          graceful = $stop.graceful; exit_code = $stop.exit_code; port_free = $stop.port_free; desktop_gpu_mib = $frMibs[0] }
        if ($launch.reason) { $launchRec.error = $launch.reason }
        $af = AffinityFields $launch
        foreach ($k in $af.Keys) { $launchRec[$k] = $af[$k] }
        foreach ($k in $frMeta.Keys) { $launchRec[$k] = $frMeta[$k] }
        Record $launchRec
        if ($stop) {
          Record (@{ test = 'serve stop'; input = "$block launch"; graceful = $stop.graceful; exit_code = $stop.exit_code; seconds = $stop.seconds; exit_line = $stop.exit_line; port_free = $stop.port_free; stall_lines = $launchStall } + $frMeta)
          if (-not $stop.graceful -or -not $stop.port_free) {
            $script:failed += "serve stop $block launch"
            $frProblems += "the launch server did not stop well (graceful $($stop.graceful), exit code $($stop.exit_code), port free $($stop.port_free))"
          }
        }
        if ($launchStall) { $frProblems += "the launch: $launchStall stall line(s) in its log" }
        $launchInfo = @{ health = $launch.health; reason = $launch.reason; setup = $setupDone; network_imports = $netImports; affinity_problem = $launch.affinity_problem }

        # --- the six decode runs, each a fresh server reading the tuned file the launch wrote
        if ($launch.health -and $setupDone -and (Test-Path -LiteralPath $tunedFile)) {
          $tunedSources = @($setupDone.sources)
          foreach ($spec in $specs) {
            $run = SpecRun 'first-run' $spec 'tuned' $speedExtra 'pass' @{ attempt = $attempt; tuned_file = $tunedFile; sources = $tunedSources }
            $frMibs += $run.r.mib
            $fig = $run.fig
            $frRows += @{ leaf = $spec.leaf; prompt = $spec.prompt; decode = $fig.decode_steps_tok_s; fnv = $fig.completion_fnv; ok = $run.good; tuned_applied = [bool]$run.tuned_applied }
            $frProblems += $run.problems
            if ($run.stop) { break }
          }
        } elseif ($launch.health) {
          $frProblems += "the launch reached /health but left no tuned file at $tunedFile (or no setup_done line), so no decode run was made"
        }

        $link = LinkAfter $block
        $agg = WithLink (FirstRunJudge $launchInfo $frRows $attempt $script:specOnMedian $script:specOnFnv $script:llamaBest $GpuName $frProblems $SpecPlan) $link
        $agg.spec_status = $script:specOnStatus
        $agg.desktop_gpu_mib = @($frMibs | Where-Object { $null -ne $_ })
        $agg.artemis_sha256 = $frMeta.artemis_sha256
        Record $agg
        if ($agg.status -ne 'pass') {
          $next = $(if ($attempt -eq 1) { 'the block may be rerun once, whole' } else { 'this was the rerun: a second miss goes to the owner' })
          Write-Warning "${block} attempt ${attempt}: not a pass; $next"
          foreach ($why in $agg.problems) { Write-Warning "${block} attempt ${attempt}: $why" }
          $script:failed += 'first-run'
        }
      } catch {
        $failMeta = $(if ($frMeta) { $frMeta } else { ConfigFields $null })
        Record @{ test = 'first-run'; plan = $SpecPlan; status = 'fail'; attempt = $attempt; error = $_.Exception.Message
          runs = @($frRows | ForEach-Object { @{ run = $_.leaf; mode = 'tuned'; decode_steps_tok_s = $_.decode } })
          desktop_gpu_mib = @($frMibs | Where-Object { $null -ne $_ }); problems = $frProblems; artemis_sha256 = $failMeta.artemis_sha256 }
        $script:failed += 'first-run'
        throw
      } finally {
        if ($null -ne $envBefore) { $env:ARTEMIS_TUNED = $envBefore } else { Remove-Item Env:ARTEMIS_TUNED -ErrorAction SilentlyContinue }
        ScanKey $block $keyFile @($speedRunDir, $evidence)
      }
    }
  }

  # Test 7 on the real engine. Server A is the default config (pool, idle_unload_s 600,
  # queue_timeout_s 300, automatic cache slots); B pins the cache slots so both its loads allocate the same cache, and
  # unloads after 30 s idle; C times a queued request out after 5 s. test-messages runs against A after
  # test7's cases, before the stop ($testMessages, declared above).
  if ($Tests -contains 'serve') {
    ServerBlock -Block 'serve' -Cases @('load', 'requests', 'budget', 'reuse', 'disconnect', 'speed') -ReuseReport -AfterCases $testMessages
  }
  if ($Tests -contains 'serve-idle') {
    ServerBlock -Block 'serve-idle' -Cases @('idle', 'unload-stream') -Extra @('--idle-unload-s', '30', '--cache-slots', '128')
  }
  if ($Tests -contains 'serve-queue') {
    ServerBlock -Block 'serve-queue' -Cases @('queue') -Extra @('--queue-timeout-s', '5')
  }
  # The production check: the three server blocks again, each started from the production
  # config template (so the idle, queue and cache-slot overrides sit on top of the production settings), with test 7's
  # new chat-tools case added to the first. test-messages runs against the first, as in `serve`; the block fails unless
  # it recorded one (a $null scriptblock would be skipped without a word). A `stall` line in any block's logs fails it
  # (ServerBlock). The smokes are the prod-check-smoke block.
  if ($Tests -contains 'prod-check') {
    ServerBlock -Block 'prod-check' -Cases @('load', 'requests', 'budget', 'reuse', 'disconnect', 'speed', 'chat-tools') -ReuseReport -AfterCases $testMessages -Config $template
    if ($script:testMessageInputs -notcontains 'prod-check') {
      Record @{ test = 'test-messages'; input = 'prod-check'; status = 'fail'; error = 'no test-messages record was written for prod-check'; artemis_sha256 = (ConfigFields $null).artemis_sha256 }
      $script:failed += 'test-messages prod-check'
    }
    ServerBlock -Block 'prod-check-idle' -Cases @('idle', 'unload-stream') -Extra @('--idle-unload-s', '30', '--cache-slots', '128') -Config $template
    ServerBlock -Block 'prod-check-queue' -Cases @('queue') -Extra @('--queue-timeout-s', '5') -Config $template
  }
  # The client smoke (prod-check-smoke is the same block, started from the template). Each script starts and stops its
  # own server, so this block holds no server and no key. The scripts run as child pwsh processes (their `exit` ends
  # only themselves); a script's output goes beside the run root, never in it (the script refuses a root that is not
  # empty). The scripts refuse to start without their canary marker, and the marker is written separately
  # (capture_clients.py smoke-canary), so a missing one is recorded as a failure here, not worked around. $Config is the
  # template the scripts pass to their server ('' for flags alone). A client's `smoke` line fails when its server's logs
  # hold a `stall` line (RecordSmoke).
  function SmokeBlock([string]$Block, [string]$Config) {
    $smokeBase = Join-Path $runs 'm3c-smoke'
    if ($smokeBase -ne 'T:\artemis-oracle\runs\m3c-smoke') {
      throw "${Block}: the smoke root is fixed at T:\artemis-oracle\runs\m3c-smoke (capture_clients.py); -Oracle gives $smokeBase"
    }
    if (-not $script:serveBuilt) {
      cargo build --release -p artemis-tools; Check "cargo build ($Block)"
      $script:serveBuilt = $true
    }
    New-Item -ItemType Directory -Force $smokeBase | Out-Null
    $hostPwsh = (Get-Process -Id $PID).Path
    $stamp = Get-Date -Format yyyyMMdd-HHmmss
    $smokeLogs = @()
    $configArgs = @(if ($Config) { '-Config'; $Config })
    foreach ($client in @('opencode', 'claude-code')) {
      $root = Join-Path $smokeBase "$client-$stamp"
      $scriptOut = "$root.script.txt"
      & $hostPwsh -NoProfile -NonInteractive -File (Join-Path $repo "tools\smoke\$client.ps1") -Exe $artemis -Model $Model -Root $root @configArgs *> $scriptOut
      $smokeExit = $LASTEXITCODE
      # The scripts keep the key at <base>\keys\<root leaf>\key.txt and delete it after a passing scan.
      RecordSmoke -Client $client -Root $root -KeyFile (Join-Path $smokeBase "keys\$client-$stamp\key.txt") -Exit $smokeExit -ScriptOut $scriptOut -Block $Block
      if (Test-Path -LiteralPath (Join-Path $root 'server\logs')) { $smokeLogs += (Join-Path $root 'server\logs') }
    }
    if ($smokeLogs.Count) {
      $rrReport = Join-Path $smokeBase "reuse-report-$stamp.json"
      $rrOut = Join-Path $smokeBase "reuse-report-$stamp.txt"
      $rrErr = Join-Path $smokeBase "reuse-report-$stamp.err"
      & $artemis reuse-report --log-dir @smokeLogs --out $rrReport 2> $rrErr > $rrOut
      if ($LASTEXITCODE) {
        Write-Warning "artemis reuse-report failed (exit $LASTEXITCODE); report only, so the gate is unchanged"
        Record @{ test = 'reuse-report (report only)'; input = $Block; status = 'error'; error = "exit $LASTEXITCODE"; artemis_sha256 = (ConfigFields $null).artemis_sha256 }
      } else {
        $r = Get-Content $rrReport -Raw | ConvertFrom-Json
        Record @{ test = 'reuse-report (report only)'; input = $Block; status = 'ran'; overall = $r.overall; by_source = $r.by_source; by_cause = $r.by_cause; budgets = $r.budgets; artemis_sha256 = (ConfigFields $null).artemis_sha256 }
      }
    }
  }
  if ($Tests -contains 'smoke') { SmokeBlock 'smoke' '' }
  if ($Tests -contains 'prod-check-smoke') { SmokeBlock 'prod-check-smoke' $template }
} finally {
  # A block that threw still gets its after-link reading (best effort, recorded), then the console's mask comes back.
  try { LinkAfter $script:hygieneBlock | Out-Null } catch { Write-Warning "the after-link reading failed: $($_.Exception.Message)" }
  try { Set-ArtemisAffinity -Mask $startMask | Out-Null } catch { Write-Warning "could not restore the console's affinity $(MaskText $startMask): $($_.Exception.Message)" }
  if ($null -ne $startTuned) { $env:ARTEMIS_TUNED = $startTuned } else { Remove-Item Env:ARTEMIS_TUNED -ErrorAction SilentlyContinue }
  Pop-Location
}
if ($script:failed.Count) {
  Write-Host "failing: $($script:failed -join ', ')"
  exit 1
}
