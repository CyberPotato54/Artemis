<p align="center"><img src="docs/images/artemis-cat.svg" alt="A black cat sitting in front of the moon" width="200"></p>

<p align="center">
  <a href="docs/install.md"><b>Install</b></a> ·
  <a href="#requirements">Requirements</a> ·
  <a href="#the-data">Speed</a> ·
  <a href="docs/server.md">API and settings</a> ·
  <a href="docs/evidence.md">Evidence</a> ·
  <a href="https://github.com/CyberPotato54/Artemis/issues">Help</a>
</p>

<p align="center">
  <a href="https://github.com/CyberPotato54/Artemis/releases/latest"><img src="https://img.shields.io/github/v/release/CyberPotato54/Artemis?label=download&color=success" alt="Download"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-blue" alt="License: MIT or Apache-2.0"></a>
</p>

# Artemis

**TL;DR**

- Runs Qwen3.6-35B-A3B on one 16 GB GPU, as a local server with the OpenAI and Anthropic APIs on one port. Works with
  OpenCode and Claude Code. Inspired by Strata.
- Decodes at 167.83 tok/s on an AMD Radeon RX 9070 XT with speculative decoding, 116.07 without. llama.cpp's fastest
  configuration on the same PC: 57.90.
- **Model:** Unsloth's `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (21.1 GiB). 35B parameters with about 3B active per token: 40
  layers (30 Gated DeltaNet, 10 full attention), 256 experts per layer with 8 routed per token, and an MTP layer for
  drafting.
- **Context and KV cache:** 65,536 tokens by default, up to the model's 262,144. The f16 KV cache serves the 10
  attention layers: 20 KiB per token, 1.25 GiB at 65,536. The DeltaNet layers keep a fixed 60 MiB f32 state.
- **Memory:** the dense weights and the KV cache live in VRAM. All experts sit in pinned system RAM (about 19 GiB),
  the hot ones are cached in the VRAM left over, and the kernels read a miss over PCIe.
- **Needs:** Windows, a Vulkan 1.3 GPU with 16 GB of VRAM and 32 GB of RAM or more.
- **Install:** [docs/install.md](docs/install.md), or give your coding agent the prompt in
  [Set it up with your coding agent](#set-it-up-with-your-coding-agent).

Artemis is an inference server for Qwen3.6-35B-A3B, written in Rust on Vulkan compute. It serves the OpenAI and
Anthropic HTTP APIs on one port, and on an AMD Radeon RX 9070 XT it decodes at a median of 167.83 tok/s with
speculative decoding on.

## Why it is almost twice as fast

- **A custom engine for one model.** Artemis is written from scratch in Rust, with its own Vulkan kernels shaped for
  this model's tensors. There is no general-purpose tensor library on the per-token path.
- **The GPU does all the math.** llama.cpp's fastest setup on this card computes the experts of 20 layers on the CPU,
  handing off between GPU and CPU at each of those layers. When an expert is not in VRAM, Artemis's GPU kernel reads
  its weights straight from pinned system RAM over PCIe and keeps going.
- **A cache that follows the conversation.** Every layer keeps its hottest experts in VRAM and swaps in the ones the
  conversation keeps routing to: 82.2% of routed experts are found in VRAM on code, against 76.3% with a fixed profile.
- **Speculative decoding on top.** Prompt lookup and the model's own MTP layer take it from 2.00 to 2.90 times
  llama.cpp's decode rate.

## How it works

Qwen3.6-35B-A3B is a mixture of experts of about 21 GiB. Each of its 40 layers has 256 experts and routes each token to
8 of them. The experts take 18.68 GiB and the rest of the model 2.42 GiB
([model contract](docs/architecture.md#model-contract)), so a 16 GB card holds the dense weights, the KV cache and part
of the experts. Engines that offload experts usually keep some layers' experts on the GPU and compute the rest on the
CPU, with the split fixed at load.

Artemis gives each layer the same number of expert slots in VRAM (146 of 256 on the RX 9070 XT) and fills them from a
routing profile at load. Between tokens it counts which experts the conversation routes to and swaps the hot ones in,
up to an upload cap per token. When a routed expert is not in VRAM, the same kernel reads it from pinned host memory
over PCIe, with no handoff to the CPU. Both paths run the same shader code, so the logits are bit for bit the same at
any cache size. On the golden prompts the adaptive cache finds 82.2% of the routed experts in VRAM on code and 66.9% on
prose, against 76.3% and 49.3% for a static profile ([evidence](docs/evidence.md#the-adaptive-cache)).

Speculative decoding suits coding agents such as OpenCode and Claude Code, whose requests are long and repeat what they
have read. Each decode step drafts a few tokens, from prompt lookup over the conversation's own text or from the
model's MTP layer, and checks them in one pass. The request's own sampler, on the same seed stream, decides which
drafted tokens to keep. On the 20 golden prompts, greedy output with speculation on matches it off in tokens and logits.

The kernels are Artemis's own, written for this model's tensor shapes, with llama.cpp b11509's `qwen35moe` graph as the
reference for the model's math. The loader checks every hyperparameter and tensor of the GGUF, its chat template and
its vocabulary, and refuses any other model with a message naming the first difference. A different chat template
loads with `allow_custom_template`. Vulkan leaves other vendors' GPUs and Linux open; the comparison with HIP is in
[decisions and alternatives](docs/architecture.md#decisions-and-alternatives).

## The data

Measured on an AMD Radeon RX 9070 XT (16 GB), AMD Ryzen 9 7950X3D, 64 GB DDR5-6000, PCIe 5.0 x16, Windows 11 Pro, from
2026-10-31 to 2026-11-04, with Unsloth's `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`. Method and records:
[docs/evidence.md](docs/evidence.md).

### Decode

Three prompts from `testdata/speed/prompts.json`: write (a 44-token request for new code), edit (a 2,206-token Python
file to change) and explain (prose), 700 tokens each at temperature 0.6. Each cell is the median of two seeds, and the
last row is the median of all six runs.

| Prompt | Speculation off | Prompt lookup only | Lookup and MTP (default) |
|---|---|---|---|
| write | 116.07 tok/s | 118.66 | 167.83 |
| edit | 109.62 | 179.96 | 177.32 |
| explain | 118.73 | 118.56 | 145.66 |
| **median of six** | **116.07** | **119.33** | **167.83** |

Speculation makes write 1.45 times, edit 1.62 times and explain, which is prose, 1.23 times as fast as plain decode.
Details: [docs/evidence.md](docs/evidence.md#decode-and-prefill).

### Prefill

Prefill runs at 2,130 tok/s on the 27,416-token golden prompt and 1,711 tok/s on the 2,206-token edit prompt with
speculation off, and at 2,132 and 1,700 tok/s with it on.

### Against llama.cpp

Same PC, prompts and seeds. Each llama.cpp row is the fastest configuration a search of its flags found at that context
window.

| Engine | Context window | Decode, median of six |
|---|---|---|
| Artemis, speculation on | 65,536 positions | 167.83 tok/s |
| Artemis, speculation off | 65,536 positions | 116.07 tok/s |
| Artemis 0.1.0, tuned file from `artemis setup` | 65,536 positions | 135.67 tok/s |
| llama.cpp b11509, Vulkan, no speculation, best searched configuration | 65,536 positions | 57.90 tok/s |
| llama.cpp b11509, Vulkan, no speculation, best searched configuration | 32,768 positions | 65.15 tok/s |
| llama.cpp b11509, Vulkan, its own MTP drafting, best of 1 to 3 draft tokens | 65,536 positions | 56.36 tok/s |
| llama.cpp b11509, Vulkan, its own MTP drafting, best of 1 to 3 draft tokens | 32,768 positions | 61.41 tok/s |

With neither engine speculating, at 65,536 positions, Artemis decodes at 2.00 times llama.cpp's rate (116.07 against
57.90 tok/s), and at 1.78 times llama.cpp's figure at 32,768 positions. With speculation on the ratios are 2.90 and
2.58. The tuned-file row was measured on 0.1.0, before this release's speed work. The flags, the searches and decode
at long context: [docs/evidence.md](docs/evidence.md#against-llamacpp).

### First run

The first `artemis serve` on a PC with no tuned file runs `artemis setup` before it opens the port. Setup lists the
GPUs, loads the model, times short decodes with speculation on and off, and writes the settings it chose, each with its
reason, to a per-user file, `tuned.toml`. Later starts read that file and begin at once. A setting in `artemis.toml` or
a flag overrides it.

On the RX 9070 XT, with the model file in the OS cache, setup takes 80.4 to 85.8 s and `/health` answers 83.2 s after
launch. Setup picks speculation on, with prompt lookup and MTP, and on 0.1.0 the file it wrote decoded at 135.67 tok/s,
0.9941 of the hand-tuned production config's 136.48 in the same run.

Run `artemis setup --force` after a driver update, a new GPU or a different model file
([docs/server.md](docs/server.md#the-tuned-file)). Details:
[docs/evidence.md](docs/evidence.md#setup-and-the-first-run).

### Accuracy

Test definitions and bounds: [architecture notes](docs/architecture.md#correctness-and-testing). Records:
[docs/evidence.md](docs/evidence.md#accuracy).

| Test | What it checks | Result |
|---|---|---|
| 1. Kernels | Each GPU kernel against its CPU twin, on random inputs and real weights | Pass |
| 2. Reference | The CPU reference forward pass against llama.cpp's CPU backend, per operation, teacher-forced | Worst tap outside attention 2.3e-6 (bound 1e-5); attention 6.9e-5 (bound 1e-3) |
| 3. Greedy golden set | Teacher-forced along llama.cpp's greedy tokens on 20 prompts (code, prose, tool use, one of 27,416 tokens): how often Artemis's top token differs from llama.cpp's | 29 in 4,477 steps (bound 48) |
| 4. Distribution | Mean KL divergence from llama.cpp's CPU backend over 51,175 tokens of code and prose | 0.01136 (bound 0.01719) |
| 5. Cache invariance | Logits at cache sizes from 0 slots to the maximum, and with a forced swap every token | Tokens and logits bit-identical |
| 6. Prompt reuse | Test 3 after restoring a snapshot, on two warm paths | Same as cold: 29 differences, logits bit for bit |
| 7. Server | OpenAI and Anthropic conformance, tool calls, reasoning, auth, lifecycle; an OpenCode and a Claude Code agent task | Pass |
| 8. Tokenizer | Token ids against `llama-tokenize` on a 1.2 MB corpus of code, prose and multilingual text | Identical |
| Speculation | Greedy output with speculation on against off, 20 golden prompts, 256 tokens each | 20 of 20 identical in tokens and logits |

### Smaller caches

With the 16 GB card's cache capped at 85 and 30 slots a layer, the estimated cache of a 12 GB and an 8 GB card, the
write prompt decodes at 91.4 tok/s with speculation off and 124.3 with it on at 85 slots, and at 63.3 and 79.4 at 30
slots. Details: [docs/evidence.md](docs/evidence.md#smaller-caches).

## Requirements

- **GPU:** a discrete GPU with 16 GB of VRAM and Vulkan 1.3 with buffer device address, timeline semaphores, 8- and
  16-bit storage, `shaderFloat16`, `shaderInt8`, subgroup arithmetic and shuffle, a required subgroup size of 32 in
  compute shaders (with full subgroups and 8 subgroups a workgroup), and `VK_EXT_memory_budget`.
  `VK_KHR_cooperative_matrix` is optional and gives fast prefill. A missing feature refuses the load with a list of
  what is missing.
- **System memory:** a loaded model pins about 18.7 GiB of host memory for the experts and the embedding, and 19.2 GiB
  with the MTP layer's experts. The snapshot pool for prompt reuse is 8 GiB by default, sized at load to the free
  memory less 4 GiB ([docs/server.md](docs/server.md), "Memory").
- **Visual C++ runtime:** the Microsoft Visual C++ Redistributable for Visual Studio 2015-2022, x64
  (https://aka.ms/vs/17/release/vc_redist.x64.exe). The binary imports `VCRUNTIME140.dll`.
- **Vulkan:** the GPU driver provides the Vulkan loader.
- **Disk:** the GGUF file, 22,663,387,424 bytes (about 21.1 GiB).
- **To build:** Rust 1.98.1 (pinned in `rust-toolchain.toml`) and Visual Studio 2022 Build Tools with the C++ workload.
  A plain `cargo build` needs no Vulkan SDK. Changing a shader or running `tools\build-release.ps1` needs it, with
  `VULKAN_SDK` pointing at the SDK.

## Limits

- One model: Qwen3.6-35B-A3B GGUFs that include the MTP layer, with the pinned chat template and vocabulary and tensors
  of type F32, BF16, Q8_0, Q4_K, Q5_K or Q6_K. Other quants are refused
  ([docs/install.md](docs/install.md#what-loads)).
- Windows, one discrete GPU, one sequence at a time, text only. There is no CPU-only mode, no image input, no OpenAI
  Responses API, no grammars and no logit bias.
- Decode reads expert misses over PCIe, so its speed follows the link: at half rate, about 14 GB/s against 28, decode
  runs about 22% slower ([evidence](docs/evidence.md#the-pcie-link)).
- By default the text of a control token in a message, a file an agent read or a tool result (for example
  `<|im_start|>`) stays plain text, so untrusted text cannot forge a turn. A prompt with such text then tokenizes
  differently from llama.cpp's. Set `special_tokens_in_content = true` for llama.cpp's parsing, where that text becomes
  a real token and can forge a turn ([docs/server.md](docs/server.md), "Not enforced").

## Set it up with your coding agent

To have an AI coding agent do the install in [docs/install.md](docs/install.md) for you, give it this prompt. The agent
needs to run PowerShell on your Windows machine (Claude Code, OpenCode, Codex and the like can). Replace the path on the
`Release archive` line first. The agent checks the machine, installs the one prerequisite if you agree, downloads and
verifies the model, starts the server and sends a test request.

```
Install Artemis on this Windows machine and check that it works. Artemis is a local inference server for the model
Qwen3.6-35B-A3B. Ask me before you install anything, and change no other system settings. If a step fails, stop
and show me the exact error instead of working around it.

Release archive: [the full path of the artemis-<version>-<commit>-windows-x64.zip I downloaded; if this is still in
brackets, ask me]

1. Check the machine: Windows x64, a GPU with 16 GB of VRAM and a current driver (Artemis checks the Vulkan features
   itself when it loads), at least 30 GiB of RAM and 23 GB of free disk. Read the VRAM from the registry value
   HardwareInformation.qwMemorySize under
   HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0* (Win32_VideoController stops
   at 4 GB). A 16 GB card reports about 15.9 GiB; under 15.5 GiB is too small. If anything is short, stop and tell me.
2. Check for the Visual C++ Redistributable 2015-2022 x64 (HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64,
   Installed = 1). If it is missing, ask me, then install https://aka.ms/vs/17/release/vc_redist.x64.exe.
3. Compare the archive's SHA-256 with its line in SHA256SUMS.txt from https://github.com/CyberPotato54/Artemis/releases
   (ask me for that file if you can't get it). If they differ, stop. Unzip the archive into $env:LOCALAPPDATA\Programs
   and rename the folder it makes to Artemis, so artemis.exe is at $env:LOCALAPPDATA\Programs\Artemis\artemis.exe. If
   that Artemis folder already exists, ask me first.
4. Download the model in the background (it is 22,663,387,424 bytes and takes longer than a command timeout). Run this
   again if it stops early; it resumes:
   curl.exe -L -C - --create-dirs -o C:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
   When the file has exactly that size, its SHA-256 must be
   0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b. If it differs, delete the file and download it
   once more.
5. Start the server in the background:
   $dir = "$env:LOCALAPPDATA\Programs\Artemis"
   Start-Process "$dir\artemis.exe" -WorkingDirectory $dir -ArgumentList 'serve','--model','C:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf' -RedirectStandardOutput "$dir\artemis-out.log" -RedirectStandardError "$dir\artemis.log" -WindowStyle Hidden
6. The first start measures the PC for a few minutes before the port opens (artemis-out.log shows the progress). Poll
   http://127.0.0.1:8080/health every 10 seconds for up to 10 minutes; any answer means it is ready ("idle" and
   "loaded": false are normal, because the model loads on the first request). Then send one request, allowing 300
   seconds for that load:
   Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType 'application/json' -TimeoutSec 300 -Body '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":50,"reasoning_effort":"none"}'
   The reply's choices[0].message.content should be a short greeting.
7. Tell me what you did, the hash results and the reply. Leave the server running; Stop-Process -Name artemis stops
   it.
```

## Quick start

[docs/install.md](docs/install.md) walks through the install step by step. The release archive,
`artemis-<version>-<commit>-windows-x64.zip`, is on the
[release page](https://github.com/CyberPotato54/Artemis/releases) and needs no build. It unzips to one folder that
holds `artemis.exe`, an example config and the install guide, and install.md says how to check it against
`SHA256SUMS.txt`. The steps below build from a checkout.

Build the `artemis` binary. This needs Rust and the Visual Studio Build Tools:

```
cargo build --release -p artemis-tools
```

`tools/build-release.ps1` is the release build. It also checks the compiled shaders against their sources with the
Vulkan SDK's `glslc` and prints the binary's sha256:

```
.\tools\build-release.ps1
```

Get the model, `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, from https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF (sha256
`0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b`). After this download Artemis needs no network.

Run the server:

```
.\target\release\artemis.exe serve --model <path to Qwen3.6-35B-A3B-UD-Q4_K_M.gguf>
```

It listens on `127.0.0.1:8080`. With no `api_key_file`, auth is off and the server binds a loopback address. The first
start measures your PC for a few minutes before the port opens (see [First run](#first-run)); later starts skip that.
The model loads on the first request, in about 9 s right after that first start or while the file stays in the OS
cache, and about 95 s on a later start with the file not cached. Then, in a second window:

```
Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType 'application/json' -Body '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

The Anthropic API is at `POST /v1/messages` on the same port. Settings come from a TOML file (`--config`) or from
flags; [docs/server.md](docs/server.md) lists every setting, endpoint, log line and exit code. Speculation is on by
default, and `--speculation-enabled false` turns it off. The setup for Claude Code, OpenCode and other clients,
including the origin check for browser-based ones, is in
[docs/install.md](docs/install.md#use-it-with-your-coding-agent).

### Tests

`cargo test --workspace` runs the CPU tests. CI builds with `ARTEMIS_SKIP_SHADERS=1`, which needs no Vulkan SDK. The
GPU tests are `#[ignore]`d, read the model path from `ARTEMIS_MODEL`, and run with
`cargo test --release -- --ignored --test-threads=1`. The ignored pwsh tests, such as `scan_secrets_pwsh`, need an
`ARTEMIS_TOOLS_DIR` seeded as in `.github/workflows/ci.yml`. The comparisons against llama.cpp run through
`tools/run-gpu-tests.ps1` ([AGENTS.md](AGENTS.md)).

## Status

What works today:

- The GGUF loader and its checks, a CPU reference forward pass, and Vulkan kernels for decode and prefill, with host
  reads for the experts that are not in VRAM.
- The expert cache with its adaptive policy and swaps between tokens.
- Fast prefill, with staging and cooperative-matrix attention.
- The OpenAI and Anthropic APIs on one port, with the chat template, tool calls, reasoning, snapshots and prompt reuse,
  and the server's lifecycle.
- A stall watchdog and `/v1/stats`.
- Speculative decoding: prompt lookup, the MTP drafter, state rollback and a draft-length controller.
- A build that needs no shader compiler, and a source archive that builds offline.
- The first-run setup that measures the PC and picks the settings (see [First run](#first-run)).

## Next

- A second mixture-of-experts model, so that Artemis is ready for Qwen 4. The model-specific parts of the engine are
  kept in named places for this.
- Modes for Vulkan GPUs with less than 16 GB of VRAM, with more of the experts in system RAM.

The design notes are in [docs/architecture.md](docs/architecture.md) (the model contract, correctness and testing, and
the decisions) and [docs/new-moe-architecture.md](docs/new-moe-architecture.md) (what a second model needs). The
figures are in [docs/evidence.md](docs/evidence.md). The model's math, as read from llama.cpp b11509, is in
[docs/reference/qwen35moe-semantics-b11509.md](docs/reference/qwen35moe-semantics-b11509.md).

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the [MIT license](LICENSE-MIT), at your
option.
