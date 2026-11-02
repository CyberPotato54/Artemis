# Bringing up a new MoE architecture

This is the checklist for running a mixture-of-experts model other than Qwen3.6-35B-A3B in Artemis, for example a
Qwen 4. Dense models are out of scope. It is written for the builder who opens it when the new model's GGUF exists.

Read it with these in hand: the new GGUF, llama.cpp's source for the model's architecture, and
`docs/reference/qwen35moe-semantics-b11509.md`, which shows the form the new model's semantics notes should take.
llama.cpp is read for the model's semantics only. No llama.cpp code is copied and every kernel is Artemis's own.

## Where things stand

The second-model seam is designed and not built.
A later refactor builds it, once the model and llama.cpp's support for it exist. Until then `artemis` loads one model:
`Dims::QWEN36` (`crates/qwen36/src/dims.rs:34`) is the only dims value the loader and the server accept, and the
contract refuses any other GGUF.

The planned cut is one `ModelSpec` per architecture, chosen from the GGUF's `general.architecture`, holding the dims
read from metadata, a per-layer mixer schedule, the tensor contract, the tokenizer parameters, the chat format, the
reasoning tags and tool-call parser, sampling defaults, an optional expert profile and an optional MTP entry. Kernel
limits become load-time checks. `qwen36` splits into a generic core and a `qwen35moe` module. This checklist is
written against today's code, so each row says where Qwen3.6 is assumed now. When that refactor lands, the rows still say
what to check and which test proves it. Only the file names move.

Line numbers below are at the commit that wrote this file. Search for the symbol if a line has drifted.

## Step 0: the facts to collect first

1. **The architecture name** in `general.architecture`, and llama.cpp's source file for it (Qwen3.6's is
   `src/models/qwen35moe.cpp`). Read its graph builder end to end and write the semantics down as
   `docs/reference/<arch>-semantics-<tag>.md`, in the form of the Qwen3.6 one: hyperparameters, each layer kind, the
   MoE block, the MTP block if any, types and casts, the tokenizer, and the names llama.cpp gives its intermediate
   tensors (the `cb()` names, which the taps use).
2. **Every metadata key** the GGUF has under `<arch>.*` and `tokenizer.*`, with its value. Compare the list with
   `expected_meta` (`crates/qwen36/src/contract.rs:79`). A key the new model has and Qwen3.6 lacks marks a feature to
   understand. A key Qwen3.6 has and the new model lacks marks an assumption to remove.
3. **The tensor table.** Compare names, shapes and types with `expected_tensors`
   (`crates/qwen36/src/contract.rs:131`). Extra or renamed tensors mean extra or different math.
4. **The llama.cpp tag.** The oracle pins b11509 (`oracle/build.ps1`, `oracle/README.md`). A new model may need a
   newer tag. A new tag means a new commit hash in the build script, a rebuilt oracle, and every expected output in
   `testdata/` remade, including Qwen3.6's, or the two models carry two oracles.
5. **Quant types.** Artemis reads F32, BF16, Q8_0, Q4_K, Q5_K and Q6_K (`crates/qwen36/src/quant.rs:7`). IQ types and
   anything else are outside the seam. A GGUF that uses one needs a new quant type through the whole stack (twin,
   shader, CPU expert kernel) or a different quantisation of the model.

## The tests that prove each row

Each row below names the test that proves it. `tools/run-gpu-tests.ps1 -Tests <name>` runs one, and its header comment
describes every block. `AGENTS.md` says how each runs. All of them run against the new model's own llama.cpp oracle:
the expected outputs under `testdata/` are Qwen3.6's today and must be remade for the new GGUF (`testdata/README.md`
lists what each file is and which script or oracle mode makes it).

| Test | What it proves | Model-specific input to remake |
|---|---|---|
| 1 | Each GPU kernel equals its CPU twin within its stated tolerance, on random inputs and on the model's real tensors | Real tensors come from the GGUF (`ARTEMIS_MODEL`) |
| 2 | The CPU reference equals llama.cpp's intermediate tensors, operation by operation, teacher-forced | Taps from `artemis-oracle taps` over the inputs in `testdata/test2/inputs`; the tap filter from `artemis taps-filter` |
| 2-gpu | The GPU engine equals the reference tap by tap, with an f16 KV cache | The same inputs and taps |
| 3 | Fed llama.cpp's greedy tokens on the 20 golden prompts, Artemis's top token leaves them at most 1.5 times as often as llama.cpp CPU's does | `testdata/golden/` (`expected/`, `forced-cpu/`), made by `tools/py/build_golden_set.py` |
| 4 | Mean KL from llama.cpp's CPU is at most 1.5 times llama.cpp's own Vulkan-versus-CPU KL | The `llama-perplexity` KL base over `testdata/kl` |
| 5 | Bit-identical logits at every cache size and with a forced swap each step | None beyond the golden prompts |
| 6 | Prompt reuse: a restored snapshot gives the cold run's tokens within the flip bound, on both warm paths | The golden prompts and the oracle's tokens |
| 7 | The server: APIs, template, tool calls, reasoning, lifecycle, queue, reuse | `testdata/template/` (prompts rendered by llama-server's `/apply-template`), `tools/py/build_template_fixtures.py` |
| 8 | Token ids equal llama.cpp's on the 1 MB corpus; the GGUF-only checks (contract, vocabulary extract, end-of-generation set) | `testdata/tokenizer/` (`corpus.*.ids`, the vocab extract, `eog.json`) |
| 2-mtp | The MTP block's taps and the GPU's MTP logits equal the oracle's (only if the model has an MTP block) | `artemis-oracle mtp-taps` over test 2's inputs |
| spec-golden | Speculation on gives speculation off's tokens on the 20 golden prompts, differing only at near-ties | The golden set |

The speculation correctness gate is spec-golden. A wrong draft can only lower speed, so 2-mtp and `mtp-check` guard the
MTP graph and spec-golden guards the output.

## The rows

### 1. Model shape

**What a new architecture may change:** every count and size: layers, hidden size, vocabulary, heads, head size,
experts, experts used, expert and shared-expert widths, DeltaNet sizes, rope base, the context length, the epsilon.

**Check in the GGUF and llama.cpp:** each `<arch>.*` key against `Dims`' fields. Read how llama.cpp derives any value
the GGUF does not state. Check that the vocabulary size equals the token array's length. Check that the shared expert
exists and has the same gate form.

**Where Qwen3.6 assumes it today:**
- `Dims` and its `QWEN36` constant: `crates/qwen36/src/dims.rs:4`, `:34`. The tiny test model is `dims.rs:61`.
- The contract compares the GGUF with `Dims` through `qwen35moe.*` keys, `general.architecture = qwen35moe`,
  `tokenizer.ggml.model = gpt2` and `tokenizer.ggml.pre = qwen35`: `crates/qwen36/src/contract.rs:79`, with the
  tokenizer keys at `:118`; the tensor list at `:131`; the check itself at `:218`.
- Entry points pass `Dims::QWEN36`: `crates/engine/src/lib.rs:369` (`Engine::load`), `crates/tools/src/serve/run.rs:73`
  (the server's contract check), `crates/tools/src/main.rs:528` (the CLI's dims choice), and several tool commands
  (`grep -rn 'Dims::QWEN36' crates --include=*.rs` lists them, tests included).
- Sizes derive from `Dims`: the session's state (`crates/engine/src/session.rs:107`, `:117`), prompt-reuse snapshots
  (`crates/engine/src/reuse.rs:62`), the verify step's layout (`crates/engine/src/verify.rs:121`).

**What to change:** read `Dims` from metadata (`expected_meta` is already the inverse mapping, so the reader is its
mirror), and replace the hard-coded keys and the architecture string with the new model's. Rewrite the tensor list for
any added or renamed tensor.

**Verified by:** test 8's GGUF-only checks (`crates/qwen36/tests/real_model.rs:14`, the contract against the real
file), then test 1, which loads real tensors.

### 2. Layer schedule

**What may change:** which layers use which mixer. Qwen3.6 has Gated DeltaNet and full attention, one rule for all 40
layers plus the MTP block. A new model may change the interval, or give an explicit per-layer list (read
`full_attention_interval` or its replacement in the GGUF and in llama.cpp's loader).

**Check:** how llama.cpp decides each layer's kind (its hyperparameter loader), whether the MTP block's mixer follows
the same rule, and whether any layer has no mixer or a second kind.

**Where Qwen3.6 assumes it today:**
- One formula: `(l + 1) % full_attn_interval == 0`, and the MTP block is always attention:
  `crates/qwen36/src/dims.rs:89`.
- Everything that branches on it: the contract (`contract.rs:142`), the reference model's layer builder
  (`crates/qwen36/src/model.rs:129`), the tap list and names (`crates/qwen36/src/taps.rs:372`, `:427`), the GPU weight
  names and the `GpuMixer` enum (`crates/engine/src/weights.rs:171`, `:358`), session state
  (`crates/engine/src/session.rs:121`, `:253`), reuse sizes (`crates/engine/src/reuse.rs:65`), the verify layout
  (`crates/engine/src/verify.rs:167`).
- The matches over `GpuMixer`: decode (`crates/engine/src/decode.rs:523`, `:603`), prefill
  (`crates/engine/src/prefill.rs:646`, `:722`), the verify pass and its replay (`crates/engine/src/verify.rs:606`,
  `:701`, `:770`), the MTP block (`crates/engine/src/mtp.rs:745`, `:912`).

**What to change:** a per-layer schedule over the same two mixers is small: replace the formula with the model's
schedule (`fn mixer(layer)` in the planned `ModelSpec`) and keep the matches. A new mixer is large: its kernels and
CPU twins, its reference code and taps, its session state, its snapshot and reuse sizes, its decode, prefill and verify
paths, its weights, and its rollback for speculation (row 9).

**Verified by:** test 2 and 2-gpu for the math, test 1 for the kernels, test 5 for cache invariance, test 6 for the new
state kind's snapshots, spec-golden for the rollback.

### 3. Kernel limits

**What may change:** the shapes the new model needs. Today these are baked into shaders and asserted at dispatch, so a
model past a limit fails deep inside a dispatch. The plan makes each a load-time check that names the limit.

**Check:** the new model's value for each limit below, in the GGUF.

| Limit | Qwen3.6 | Where |
|---|---|---|
| DeltaNet key and value head size at most 128 | exactly 128 | `crates/kernels/src/dispatch.rs:522` (`gdn_step`), `:594` (`gdn_seq`); shared arrays in `crates/kernels/shaders/gdn_step.comp:24` |
| Conv window 2 to 8 | 4 | `dispatch.rs:558` |
| Attention head size at most 256 | exactly 256 | `dispatch.rs:909` (`qk_norm_rope`), `:1011` (`attn_decode`), `:1084` (`attn_prefill`) |
| RoPE: `n_rot / 2` at most 64, so `n_rot` at most 128 | `n_rot` 64 | `dispatch.rs:909` (`qk_norm_rope`, the same assertion as the head size) |
| Cooperative-matrix prefill attention: head size a multiple of 64, 64 to 256 | 256 | `dispatch.rs:860` (`attn_prefill_cm_fits`), `:1147` |
| At most 8 query heads to a KV head, and head size a multiple of 32, for the GQA decode kernel (decode uses the plain kernel otherwise) | 8 query heads to a KV head | `dispatch.rs:669` (`ATTN_GQA_GROUP`), `:677` (`gqa_takes`), `:1018` |
| Query heads divide over KV heads; head size a multiple of 8 for prefill | 16 and 2 | `dispatch.rs:1088`, `:1100` |
| At most 256 experts and top-k at most 16 in the router | exactly 256 and 8 | `dispatch.rs:1529` (`router_topk`), `:1885` (`moe_group`); `crates/kernels/shaders/router_topk.comp:22` |
| Expert matrix widths multiples of 32 | 2,048 and 512 | `dispatch.rs:2071`, `:2127` |
| CPU-miss mailbox: `n_embd` at most 2,048, at most 8 experts used, at most 48 layers counting the MTP block | exactly 2,048, 8 and 41 | `dispatch.rs:1243` to `:1251`, `:1693`, `:1734`; `crates/engine/src/cpu_miss.rs:163` |
| Verify step: 16 tokens, and tokens times experts used at most 256 | 16 times 8 is 128 | `crates/kernels/src/tokens/mod.rs:10`, `:14`; `crates/engine/src/verify.rs:249`, `:259` |

**What to change:** where the new model fits, nothing but a load-time check that names the limit. Where it does not,
parameterise the shader (the shared arrays and workgroup sizes), then re-run test 1 for the kernel. Qwen3.6 sits
exactly at several limits, so a model with larger heads or more experts needs a shader change, because a check would
only refuse it.

**Verified by:** test 1 (each kernel against its twin, random inputs and real tensors), then tests 2-gpu and 5. A
parameterised kernel also needs its CPU twin updated (`crates/kernels/src/twin`), which CI checks against
`qwen36::ops`.

### 4. Math variants

**What may change:** the position encoding, the router, the activation, the attention form, the DeltaNet recurrence,
the norms and the shared expert.

**Check in llama.cpp, one by one:**
- RoPE: the type (NeoX, interleaved, multi-section), `rope.dimension_count`, the sections, the base, any scaling.
- Router: softmax or sigmoid scoring, top-k selection, renormalising the chosen weights and the floor on their sum,
  any routing bias or group-limited selection, any scale factor.
- Activation in the experts and the shared expert (SwiGLU today), and the shared expert's sigmoid gate.
- Attention: the output gate (Q and gate interleaved per head today), QK norm, bias terms, sliding windows, the scale.
- DeltaNet: the gate and beta computation, value-head tiling, the state's layout, the conv.
- Norm weights: whether the converter folded in a constant (Qwen3.6's already contain +1).

**Where Qwen3.6 assumes it today:**
- NeoX RoPE on the first `n_rot` dims; the section values are checked in the contract and unused (text only):
  `crates/qwen36/src/ops.rs:99`, the contract at `crates/qwen36/src/contract.rs:98`, the shader
  `crates/kernels/shaders/qk_norm_rope.comp`.
- Router: softmax, top-k, renormalise, with the floor `6.103515625e-5`: `crates/qwen36/src/ops.rs:73`, `:271`, `:279`;
  `crates/qwen36/src/reference.rs:398`; the shader `crates/kernels/shaders/router_topk.comp`.
- SwiGLU: `crates/qwen36/src/ops.rs:287`. The shared expert's sigmoid gate: `crates/qwen36/src/reference.rs:448`.
- The attention output gate: `crates/qwen36/src/ops.rs:111` (`split_q_gate`) and `crates/qwen36/src/reference.rs:376`.
- The DeltaNet block: `crates/qwen36/src/reference.rs:212`; the value-head tiling (`h % n_group`) in
  `crates/kernels/shaders/gdn_step.comp:29`.

**What to change:** each differing variant needs the reference code changed (line for line from llama.cpp's graph), a
kernel and twin changed or added, and a tap or two added for the new intermediates. Medium to large, per variant.

**Verified by:** test 2 first, per operation. It teacher-forces, so a wrong operation shows at its own tap and does not
hide behind the ones after it. Then test 1, 2-gpu, 3 and 4 end to end.

### 5. Tokenizer

**What may change:** the pre-tokenizer, the special tokens and the end-of-generation set. Qwen 4 may keep Qwen3.6's
vocabulary, extend it or replace it.

**Check:** `tokenizer.ggml.model` and `tokenizer.ggml.pre`, then llama.cpp's handling of that pre type (the regex and
the splitter), the BOS and EOS settings, the token types, the end-of-generation tokens llama.cpp derives for this
vocabulary, and whether `add_bos_token` is set.

**Where Qwen3.6 assumes it today:**
- The pre-tokenizer regex (`qwen35`): `crates/qwen36/src/tokenizer.rs:18`, used at `:306`.
- The end-of-generation texts: `crates/qwen36/src/tokenizer.rs:27`, applied by `Vocab::eog` at `:144`.
- The pre key in the contract: `crates/qwen36/src/contract.rs:118`.
- The chat template is read from the GGUF (`tokenizer.rs:152`), so it needs no change unless it uses constructs the
  minijinja renderer lacks.

**What to change:** small: the regex for a new pre type, the end-of-generation list, the special-token names.

**Verified by:** test 8 (the corpus ids and the GGUF checks, including the end-of-generation set against the oracle's
`vocab` mode), and test 7's template cases for the chat template.

### 6. Expert profile

**What may change:** the count of how often each expert is routed, which fills the VRAM expert cache before the first
request.

**Where Qwen3.6 assumes it today:** an 81,920-byte profile is compiled in and used only when
`Dims == Dims::QWEN36`; any other dims start from zeros, so the cache fills the lowest expert ids:
`crates/engine/src/profile.rs:8`, `:39`. It is read at `crates/engine/src/lib.rs:570`.

**What to change:** nothing is required. A model with zeros loads and runs, with a slow start while the adaptive policy
learns. A profile per model is the tuning step for later. The Qwen3.6 one was built from routing
counts by `tools/py/make_expert_profile.py`; a new model needs counts from its own routing.

**Verified by:** test 5 does not depend on the profile (logits are identical whatever the cache holds). The hit-rate
gate (`-Tests hits`) and the speed blocks show what a profile buys.

### 7. Verification (the reference and the taps)

**What may change:** everything under test 2. The reference forward pass is the model's math, line for line, so it
changes with the model.

**Where Qwen3.6 assumes it today:**
- The reference forward pass: `crates/qwen36/src/reference.rs`. The model loader: `crates/qwen36/src/model.rs`.
- The tap keys: 29 per-layer keys (`Gdn*`, `Attn*`, `Router*`, `Moe*`, `Shexp*` and the rest):
  `crates/qwen36/src/taps.rs:9`, `:42`, with the llama.cpp names at `:366` and the expected tap list at `:424`.
- The oracle dumps llama.cpp's named tensors: `oracle/dump.cpp`. A new model whose graph names its nodes differently,
  or leaves a node unnamed, needs the oracle changed (Qwen3.6's MTP gate node is one such case, `oracle/dump.cpp:161`).

**What to change:** large. Write the reference from the semantics notes of step 0, add or rename taps, and add the
llama.cpp name for each. If a node has no name in llama.cpp, the oracle names it itself.

**Verified by:** test 2 (CPU reference against the oracle), then 2-gpu (GPU against the reference). Test 3 and test 4
check the end result against llama.cpp's own run of the same GGUF. Do not start the GPU work for a changed op until
test 2 passes for it.

### 8. Server

**What may change:** the chat format (role markers), the reasoning tags, the tool-call grammar, the sampling defaults.

**Check:** the GGUF's chat template for the generation prompt, its thinking switch, how it renders a tool call and a
tool result, and the special tokens it uses. Then llama.cpp's server for how it parses the model's output.

**Where Qwen3.6 assumes it today:**
- `Specials` finds `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, `<tool_call>` and `</tool_call>` by text, and
  fails the load if one is missing: `crates/server/src/tokens.rs:28`.
- The prompt cut needs the generation prompt to open with `<|im_start|>`: `crates/server/src/prompt.rs:74`.
- The reasoning state machine (reasoning, content, call) is keyed on those token ids:
  `crates/server/src/pipeline/mod.rs:72` (the modes), `:206` (the transitions).
- The tool-call parser reads Qwen's XML form (`<function=NAME>` and `<parameter=KEY>`):
  `crates/server/src/pipeline/toolcall.rs:13`, `:15`.
- The reuse cause logic reads turn boundaries from `<|im_start|>`, `<|im_end|>` and the call tags:
  `crates/server/src/logs.rs:66`.
- The server's own dependency on the model crate is `qwen36` (tokenizer, vocabulary, text stream):
  `crates/server/Cargo.toml:10`. The load path is `crates/tools/src/serve/run.rs:73`.
- Sampling defaults (temperature 0.6, top-p 0.95, top-k 20): `crates/server/src/config.rs:32`.

**What to change:** another chat format with the same style of tags is a medium change (names and transitions). Another
tool-call grammar is a large one: a new incremental parser with the same repair and boundary rules, tested the way
`toolcall.rs` is.

**Verified by:** test 7. Its CI cases run on the fake engine and need the new model's tokens in the fake's fixtures.
Its template cases need `testdata/template/` remade from the oracle. Its real-engine cases, the
`tools/test-messages.ps1` port and the two client smokes run on the GPU.

### 9. Speculation

Speculation has a generic part that works unchanged and a model-specific part.

**Generic (no change for a new model):** the draft, verify, accept and commit loop; exact-match acceptance through the
request's sampler; prompt lookup (token ids only); the controller, its config and its reporting; the engine's verify
and commit calls; the position rollback `Session::rewind` (`crates/engine/src/verify.rs:412`), which cuts the pending
verify's tokens, `pos` and KV record back to a position whatever the layers are; and `PolicyStep::Verify`
(`crates/engine/src/policy.rs:62`, used at `crates/engine/src/verify.rs:510`), which prepares a verify step's expert
copies like a one-token decode step. `Verifier` and `Drafter` are the seams: `crates/engine/src/spec/step.rs:12` and
`crates/engine/src/spec/mod.rs:271`.

**Model-specific:**
- **How each state kind rolls back.** An attention layer rolls back by position, which `rewind` (above) does. A
  DeltaNet layer rolls back by replaying the kept tokens through its saved inputs: `crates/engine/src/verify.rs:585`
  (`replay`, whose `match` is the per-layer rollback). A new mixer needs its own rollback in that `match` and its own
  saved inputs in the layout.
- **The MTP drafter.** Whether the GGUF has an MTP block and how many (`nextn_predict_layers`): the contract
  (`crates/qwen36/src/contract.rs:88`), `Dims::n_nextn` (`dims.rs:7`), the block's tensors
  (`contract.rs:180`), its graph (`crates/engine/src/mtp.rs`) and the reference's MTP pass
  (`MtpReference`, `crates/qwen36/src/reference.rs:482`). It drafts through `Verifier::mtp_draft` and `has_mtp`
  (`crates/engine/src/spec/step.rs:36`, `:41`). A model with no MTP block runs on prompt lookup alone. A model with one
  needs its graph written from llama.cpp's semantics and verified.
- **The multi-token kernels' shape limits**, checked like every kernel limit (row 3, last line of the table).

**Verified by:** 2-mtp for the MTP graph, spec-golden for output equality, test 5 (its runs speculate, and one must have
drafted with MTP if MTP is a source), then the `serve-speed-spec` block for the speed figures.

## Order of work

### A model of the same family (the same two mixers, other counts or sizes)

1. Step 0: the metadata list, tensor table, tag decision, semantics notes.
2. Rows 1 and 2: read `Dims` from metadata, set the layer schedule, rewrite the contract keys. Run test 8's GGUF-only
   checks until the contract passes.
3. Row 3: compare each limit with the new values. Add the load-time checks. Parameterise a shader only if a limit fails.
4. Row 5: the tokenizer parameters. Run test 8.
5. Remake the oracle data (test 2 taps, golden set, KL base, template fixtures). Run tests 1, 2, 2-gpu, 3, 4, 5 and 6.
6. Row 8: the server's tags and sampling defaults. Run test 7.
7. Row 9: if there is an MTP block of the same shape, run 2-mtp and spec-golden. Without one, run spec-golden with
   lookup alone.
8. Row 6: measure a profile when the model is worth tuning.

### A model with a new mixer, router, attention variant or tool-call grammar

Do steps 1 to 4 above, then:

1. Row 4: write the reference for each new or changed operation from llama.cpp's graph, with its taps. Remake test 2's
   taps from the oracle (step 5 above), then run test 2 for it before any GPU code. This is the first gate, and it needs
   the oracle to name the tensors (row 7).
2. Write the CPU twin and the kernel for the operation. Run test 1 until it passes, then 2-gpu.
3. For a new mixer: add its session state, snapshot and reuse sizes, and its decode, prefill and verify paths (row 2).
   Run tests 5 and 6, which check cache invariance and snapshot restore for the new state kind.
4. Add its rollback (row 9). Run spec-golden.
5. For a new tool-call grammar: write the parser (row 8) and run test 7.
6. Then the rest of the same-family list: tests 3 and 4 end to end, the speed blocks, the profile.

Keep Qwen3.6's results unchanged throughout: tests 1 to 5 and 8 on Qwen3.6 must equal the values recorded before the
refactor. The refactor that builds the seam is its own piece of work, and its gate is that
equality.
