# GitHub "About" for Artemis

Maintainer note: the text to paste into the repository's About box. Figures and their sources are in [evidence.md](../evidence.md).

## Description (314 characters; the limit is 350)

Rust inference engine on Vulkan for Qwen3.6-35B-A3B. Runs the 21 GiB mixture-of-experts model on a 16 GB GPU by keeping hot experts in an adaptive VRAM cache and reading the rest from system RAM over PCIe. OpenAI and Anthropic APIs, speculative decoding checked against plain decode, 137.75 tok/s on an RX 9070 XT.

## Topics

- `rust`
- `vulkan`
- `llm-inference`
- `mixture-of-experts`
- `qwen`
- `speculative-decoding`
- `amd-gpu`
- `gpu-compute`
- `openai-api`
- `local-llm`

## Alternative descriptions

1. (277 characters) Qwen3.6-35B-A3B on a 16 GB GPU: a Rust and Vulkan engine with an adaptive VRAM expert cache, experts read over PCIe from pinned system RAM, OpenAI and Anthropic APIs on one port, speculative decoding checked against plain decode, and the model checked against llama.cpp b11509.
2. (212 characters) Single-model LLM server for Qwen3.6-35B-A3B on Vulkan GPUs with 16 GB of VRAM. Adaptive expert cache, OpenAI and Anthropic APIs, speculative decoding. 137.75 tok/s decode and 2,125 tok/s prefill on an RX 9070 XT.
