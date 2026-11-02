# Architecture notes

These are the parts of Artemis's architecture notes that the README and the other pages rely on: the model contract,
how correctness is tested, and the decisions with the alternatives that were turned down. The model's math is in
[reference/qwen35moe-semantics-b11509.md](reference/qwen35moe-semantics-b11509.md). The figures from running the tests
are in [evidence.md](evidence.md).

## Model contract

Verified from `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (sha256 `0b21525e…c58b`) on 2026-10-08:

| Item | Value |
|---|---|
| Architecture | `qwen35moe`, 41 blocks: 40 layers plus 1 MTP layer (`blk.40`, `nextn_predict_layers = 1`) |
| Hidden size | 2048; RMSNorm epsilon 1e-6 |
| Gated DeltaNet layers | 30 (every layer except 3, 7, …, 39). Fused QKV 2048→8192 (Q8_0), gate 2048→4096, conv1d of width 4 over 8192 channels, 32 value heads with a 128×128 state, 16 key groups, gated RMSNorm, output 4096→2048. Value heads are tiled: head `h` uses key group `h % 16`. `ssm_alpha` and `ssm_beta` are F32 |
| Full-attention layers | 10 (`full_attention_interval = 4`). Q 2048→8192 (query and gate, interleaved per head), K and V 2048→512, 16 query heads and 2 KV heads of size 256, q/k RMSNorm, RoPE on 64 of 256 dims with sections [11, 11, 10, 0] and base 1e7, gated output 4096→2048 |
| MoE | 256 experts, 8 routed per token, plus 1 shared expert with a sigmoid gate. Expert and shared-expert width 512. Router F32 (BF16 in `blk.40`) |
| Expert quants | gate and up Q4_K; down Q5_K, or Q6_K in layers 34, 38 and 39. One expert is 1,900,544 bytes (1.81 MiB), or 2,039,808 bytes (1.95 MiB) with Q6_K down |
| Other quants | Projections Q8_0, output layer Q6_K (2048 × 248,320), embedding Q8_0 |
| Norm weights | Stored with HF's +1 already added, except `ssm_norm`; all are applied as a plain multiply |
| Sizes | Experts 18.68 GiB, everything else 2.42 GiB |
| Tokenizer | GPT-2-style BPE, pre-tokenizer `qwen35`, 248,320 tokens, 247,587 merges. BOS 248044 (not added), EOS 248046 |
| Context | 262,144 trained; Artemis defaults to 65,536 |

**Semantic reference.** llama.cpp's `qwen35moe` graph at b11509 defines the math: the routing normalization, the
DeltaNet update, the RoPE mode and the gating. Artemis reads it for semantics only; all of Artemis's kernels are written
from scratch. Where this page and that graph disagree, the graph wins and this page gets corrected.

**Load-time validation.** `qwen36` checks the architecture, every hyperparameter above, and the name, shape and quant
type of every tensor. Any mismatch refuses the file with a message naming the first difference.

## Correctness and testing

Matching llama.cpp bit for bit isn't possible, because different kernels round floats differently. "Matches the stock
model" means all of the tests below pass.

| # | Test | Pass condition |
|---|---|---|
| 1 | Kernels | Each GPU kernel matches its CPU twin within its stated per-operation tolerance, on random inputs and on real weights |
| 2 | Reference forward pass | Our CPU reference agrees with llama.cpp's intermediate tensors layer by layer on short inputs, within tolerance |
| 3 | Greedy golden set | 20 fixed prompts (code, prose, tool use, one of 27,416 tokens), up to 256 steps each, teacher-forced along llama.cpp b11509's greedy tokens (Vulkan, f16 KV). A flip is a step where Artemis's top token differs and llama.cpp's own top two logits are more than 0.05 apart. Pooled flips are at most 1.5 times llama.cpp CPU's own count along the same tokens (48 against its 32) |
| 4 | Distribution | Mean KL divergence from llama.cpp's CPU backend over 50K teacher-forced tokens of code and prose is at most 1.5× the KL between llama.cpp's own Vulkan and CPU backends |
| 5 | Cache invariance | Bit-identical logits at cache sizes of 0%, 10%, 40% and the maximum, and with a forced swap every token |
| 6 | Prompt reuse | Test 3's teacher-forced count (0.05 margin) on the golden set on two warm paths, each after a whole different golden prompt was prefilled from position 0: (1) a snapshot built by prefill, restored, then the rest of the prompt prefilled; (2) a snapshot built by decode partway through the forced tokens, restored, then the remaining tokens forced. Each path's pooled flips are at most 48, test 3's bound, reported against the cold run's. In addition, a restore at a 2,048-aligned cut in the 27,416-token prompt gives the cold run's logits bit for bit |
| 7 | Server | OpenAI and Anthropic conformance, streaming and not; single, multiple and malformed tool calls; reasoning split and thinking control; auth; 503 while loading; idle and on-request unload; queue timeout; template conformance against the golden prompts. The CI cases run on a fake engine and the rest on the real one. Plus a port of `Test-Messages.ps1` and two smoke tasks, one OpenCode and one Claude Code |
| 8 | Tokenizer | Token IDs equal `llama-tokenize`'s on 1.2 MB of mixed code, prose and multilingual text |
| 9 | Speed | The write, edit and explain prompts of `testdata/speed/prompts.json`, against Artemis and llama.cpp b11509's best configuration. Results recorded at each milestone |

- CI runs the tests that need no GPU: `gguf`, the reference, the tokenizer, the server against a fake engine, and CPU
  twins.
- GPU tests run on a machine with the GPU through `tools/run-gpu-tests.ps1`, which records its results.

## Setup and the tuned file

The settings that depend on the machine are measured once and kept apart from the ones a person writes. `artemis
setup` loads the model, times short decodes of three calibration prompts (write, edit and explain, 256 tokens each) with
speculation on and with it off, and writes what it chose to a per-user tuned file. The first `artemis serve` that finds
no such file runs the same measuring before it opens the port. Setup chooses only settings the user left unset:
whether speculation is on, which draft sources it uses (prompt lookup alone, or lookup and MTP), and the controller's
cost figures when the measured ones differ from the built-in ones by more than a quarter and are at least 1.03 times as
fast. It keeps MTP when it is at least 1.02 times as fast as lookup alone. The context size, the cache size and the
prefill mode are never tuned: setup measures with the values the user set for them. The whole run has a limit of 600 s.

`serve` layers the file under `artemis.toml`: the defaults, then the tuned file, then the config file, then the flags,
so a key the user sets is never replaced. A tuned value meets the same checks as the same key in `artemis.toml`. The
file records the GPU and driver it was measured on, and a load on a different GPU or driver writes a warning that names
`artemis setup --force`. Setting `ARTEMIS_TUNED=off` turns the layer off for a test or a script that must run with the
defaults. The file, its keys and the command are described in [server.md](server.md#the-tuned-file). The first-run gate
(2026-11-01) measured a first run of 80.4 s and a decode median 0.9941 of the hand-tuned config's
([evidence](evidence.md#setup-and-the-first-run)).

## Decisions and alternatives

| Decision | Chosen | Alternatives and why not |
|---|---|---|
| How much to own | Everything on the per-token path | ggml as the tensor library would reach parity in weeks, but the hot path wouldn't be ours |
| GPU API | Vulkan compute | HIP: easier kernels but AMD-only, and ROCm on Windows relies on dev libraries. On this card llama.cpp decoded 10 to 15% faster on Vulkan. Both: doubles the kernel work |
| Language | Rust | C++ is closer to llama.cpp, but Windows builds are harder and the safety net is thinner. Zig: no tokenizer or HTTP stack |
| Model | Qwen3.6-35B-A3B only | Qwen3.8-Flash-Next was the other candidate, but its architecture is much larger and more unusual |
| Serving misses | **A**: the GPU reads uncached experts from pinned RAM over PCIe, with the CPU path as a measured add-on | **B** (the CPU computes misses with a handoff every layer) has the higher ceiling, and a calibration of an earlier offload design on the development machine was fastest at a PCIe fraction of 0.20, which suggests most misses went to the CPU. But Vulkan has no command that waits on a memory value, so B needs a spin-wait of unknown reliability or per-layer submits, which cost llama.cpp 0.125 ms per layer. **C** (upload on miss, as in llama.cpp #29887) pays B's round trip without its benefit. The slot table's `CPU` value keeps B open |
| Speculative decoding | After milestone 4's production-readiness work | Was to wait until the server was in daily use: that bar does not need it (the estimate without MTP, 55 to 65 tok/s, already clears 46.5 tok/s), but the clients are agents and the throughput was wanted first |

