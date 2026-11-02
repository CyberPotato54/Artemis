# qwen35moe semantics at llama.cpp b11509

Date: 2026-10-08
Purpose: the exact math Artemis implements, read from llama.cpp's `qwen35moe` graph at tag **b11509** for semantics only. No llama.cpp code is copied; every kernel is our own. Citations are `path:line` at that tag.
Checked against: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (sha256 `0b21525e…c58b`) metadata and tensor table, 2026-10-08.

Notation: GGML's `ne {a, b}` means `a` is the contiguous (input) dimension. `mul_mat(W{in,out}, x{in,T})` gives `y{out,T}`. T is the number of tokens in a batch.

## Five facts that are easy to get wrong

1. **Norm weights already contain +1.** The converter adds 1 to every HF `*norm.weight` except `linear_attn.norm.weight` (`conversion/qwen.py:402-403`). So `attn_norm`, `post_attention_norm`, `attn_q_norm`, `attn_k_norm`, `output_norm` and the MTP norms are applied as a plain multiply. `ssm_norm` is the raw HF weight, also applied as a plain multiply.
2. **Value heads are tiled, not grouped.** The converter reorders the 32 DeltaNet value heads (`conversion/qwen.py:454-492, 585-634`). Value head `h` uses q/k group `h % 16`, not `h / 2` (`ggml-cpu/ops.cpp:11199-11200`; `llama-model.cpp:640-642`).
3. **`ssm_a` holds `-exp(A_log)`,** not `A_log` (`conversion/qwen.py:396-397`).
4. **The DeltaNet state is stored transposed:** per head `M[v][k]`, so `M[j][i] = S[i][j]`, with the key index contiguous (`ops.cpp:11227`).
5. **Q and the attention gate are interleaved per head,** `[q_h (256) | gate_h (256)]` for each of the 16 heads, not two halves (`src/models/qwen35moe.cpp:299-301, 313-316`).

## 1. Hyperparameters and layer types

Values verified in the GGUF:

| Key (`qwen35moe.*`) | Value |
|---|---|
| `block_count` | 41 (40 layers + 1 MTP block) |
| `nextn_predict_layers` | 1 |
| `embedding_length` | 2048 |
| `attention.head_count` / `head_count_kv` | 16 / 2 |
| `attention.key_length` / `value_length` | 256 / 256 |
| `attention.layer_norm_rms_epsilon` | 1e-6 (f32 `9.99999997e-07`), used by every norm |
| `rope.dimension_count` | 64 |
| `rope.dimension_sections` | [11, 11, 10, 0] |
| `rope.freq_base` | 1e7 |
| `expert_count` / `expert_used_count` | 256 / 8 |
| `expert_feed_forward_length` / `expert_shared_feed_forward_length` | 512 / 512 |
| `ssm.conv_kernel` / `state_size` / `group_count` / `time_step_rank` / `inner_size` | 4 / 128 / 16 / 32 / 4096 |
| `full_attention_interval` | 4 |
| `context_length` | 262,144 |

- Keys read by the shared loader: `src/llama-model.cpp:1317-1504`. Keys read only for this arch: `src/models/qwen35moe.cpp:4-26`.
- `expert_weights_scale` and `expert_weights_norm` are not read for this arch; normalization is hard-coded on (§5). `attention.scale` is not set, so the scale is 1/√head_dim (`llama-hparams.h:227`).
- **Layer type** (`qwen35moe.cpp:20-26`): if `attention.recurrent_layers` exists it decides; this GGUF has none, so `recurrent[i] = (i+1) % 4 != 0`. Layers 3, 7, …, 39 are full attention; the other 30 are Gated DeltaNet.
- `n_layer = block_count − nextn_predict_layers = 40` (`llama-hparams.cpp:348-349`). The MTP block is never run in the main pass (§6).

## 2. Embedding, norms, output

- **Embedding:** `get_rows(token_embd.weight {2048, 248320}, ids)` dequantized to f32, not scaled (`qwen35moe.cpp:43,168`; `llama-graph.cpp:2461-2586`).
- **RMSNorm** (`llama-graph.cpp:1640-1673`; `ops.cpp:4140-4147`): `y = x / sqrt(mean(x²) + eps) ⊙ w`. The CPU sums squares in double.
- **Final:** `logits = output.weight · (RMSNorm(x_40) ⊙ output_norm.weight)` (`qwen35moe.cpp:230, 243`). `output.weight {2048, 248320}` is present (Q6_K), so the head is not tied.

## 3. Full-attention layer (`qwen35moe.cpp:278-357`)

Tensors: `attn_q {2048, 8192}`, `attn_k {2048, 512}`, `attn_v {2048, 512}`, `attn_q_norm {256}`, `attn_k_norm {256}`, `attn_output {4096, 2048}`. H = 16, Hkv = 2, D = 256.

```
Qfull = attn_q · x                                   # 8192 = 16 heads × [q (256) | gate (256)]
q_h, gate_h = Qfull[h*512 : h*512+256], Qfull[h*512+256 : h*512+512]
k = attn_k · x ;  v = attn_v · x                     # 2 heads × 256 each
q_h = RMSNorm_256(q_h) ⊙ attn_q_norm                 # per head, same eps   (305)
k_h = RMSNorm_256(k_h) ⊙ attn_k_norm                 #                      (309-310)
q, k = RoPE(q), RoPE(k)                              # see below            (323-333)
KV cache ← k, v (post-norm, post-RoPE)               # f16 by default       (llama-graph.cpp:3038-3043)
a_h = softmax(q_h·Kᵀ / 16 + causal_mask) · V         # query head h uses KV head h / 8   (ggml-cpu.c:1186,1221)
o = concat_h(a_h) ⊙ sigmoid(concat_h(gate_h))        # gate applied before wo (342-354)
out = attn_output · o
```

**RoPE.** `ggml_rope_multi` with mode IMROPE (40) (`llama-model.cpp:3240-3244`). For text, each token's four position ids are `[p, p, p, 0]` (`llama-batch.cpp:128-145`), and with sections [11, 11, 10, 0] every rotated pair picks t, h or w, all equal to p (`ops.cpp:6255-6264`). So for text it is exactly NeoX RoPE on the first 64 of 256 dims:

```
for i in 0..32:  θ = p · 1e7^(−2i/64)
    (x[i], x[i+32]) ← (x[i]·cos θ − x[i+32]·sin θ,  x[i]·sin θ + x[i+32]·cos θ)
dims 64..255 unchanged
```

No frequency scaling: `freq_scale = 1`, `ext_factor = 0`, `attn_factor = 1` (`llama-context.cpp:117-118, 137-138, 168-217`).

## 4. Gated DeltaNet layer (`qwen35moe.cpp:359-492`, `src/models/delta-net-base.cpp`)

Tensors (verified types in parentheses):

| GGUF | Shape | Type | HF source |
|---|---|---|---|
| `attn_qkv.weight` | {2048, 8192} | Q8_0 | `in_proj_qkv` |
| `attn_gate.weight` | {2048, 4096} | Q8_0 | `in_proj_z` |
| `ssm_beta.weight` | {2048, 32} | F32 | `in_proj_b` |
| `ssm_alpha.weight` | {2048, 32} | F32 | `in_proj_a` |
| `ssm_conv1d.weight` | {4, 8192} | F32 | `conv1d` |
| `ssm_dt.bias` | {32} | F32 | `dt_bias` |
| `ssm_a` | {32} | F32 | `-exp(A_log)` |
| `ssm_norm.weight` | {128} | F32 | `linear_attn.norm`, shared by all heads, no +1 |
| `ssm_out.weight` | {4096, 2048} | Q8_0 | `out_proj` |

Per token, with `x` the `attn_norm` output:

```
mixed = attn_qkv · x        # 8192: q[0:2048] | k[2048:4096] | v[4096:8192]
                            # q, k: 16 groups × 128 (group-major); v: 32 heads × 128
z     = attn_gate · x       # 32 heads × 128
beta  = sigmoid(ssm_beta · x)                              # 32   (382-387)
g     = ssm_a ⊙ softplus(ssm_alpha · x + ssm_dt.bias)      # 32, ≤ 0   (389-398)
        softplus(u) = u > 20 ? u : log(1 + exp(u))         # (ggml-cpu/unary-ops.cpp:80-82)
```

**Causal conv1d** (`delta-net-base.cpp:449-525`; `ops.cpp:9959-9979`), per channel c of the 8192:

```
window = [st_c[0], st_c[1], st_c[2], mixed_c]       # oldest → newest
y_c    = Σ_{τ=0..3} window[τ] · W[c][τ]             # W = ssm_conv1d; W[c][3] multiplies the current token; no bias
st_c   ← window[1..3]                               # the state keeps raw inputs, not post-SiLU
y      = silu(y)                                    # (qwen35moe.cpp:415-419)
```

The conv state is f32 `[8192][3]`, time contiguous within a channel. A new sequence starts from zeros (`llama-graph.cpp:3652-3653`).

**Normalize and recur**, for value head h = 0..31 with key group g = h % 16:

```
q_g = y[g*128 : g*128+128];  k_g = y[2048 + g*128 : …];  v_h = y[4096 + h*128 : …]
q̂ = q / sqrt(Σq² + eps);  k̂ = k / sqrt(Σk² + eps)       # eps = rms eps (models.h:14-18; qwen35moe.cpp:451-454)

S_h        ← exp(g_h) · S_h                             # decay first   (ops.cpp:11239-11241)
u_j        = Σ_i S_h[i][j] · k̂_g[i]
δ_j        = (v_h[j] − u_j) · β_h                       # (11243-11248)
S_h[i][j] += k̂_g[i] · δ_j                               # rank-1 update (11250-11253)
o_h[j]     = (Σ_i S_h[i][j] · q̂_g[i]) / sqrt(128)       # uses the updated state (11255-11260)
```

i is the key index and j the value index. llama.cpp stores `S_h` as `[v=128][k=128]` f32 per head, `[32][128][128]` per layer.

**Output** (`qwen35moe.cpp:267-276, 475-489`):

```
n_h = RMSNorm_128(o_h) ⊙ ssm_norm ⊙ silu(z_h)      # per head
out = ssm_out · concat_h(n_h)                      # 4096 → 2048
```

**Prefill.** By default both decode and prefill use the fused `ggml_gated_delta_net` op (`llama-context.cpp:236-238`; `delta-net-base.cpp:425-447`), which on CPU loops over tokens in order. The chunked form (`delta-net-base.cpp:16-287`, chunks of 64) is used only when the fused op is off. It is an algebraic reformulation: the same result in exact arithmetic, different rounding.

## 5. MoE block (`qwen35moe.cpp:494-548`; `llama-graph.cpp:2050-2419`)

Tensors: `ffn_gate_inp {2048, 256}` (F32), `ffn_gate_exps` and `ffn_up_exps {2048, 512, 256}` (Q4_K), `ffn_down_exps {512, 2048, 256}` (Q5_K, or Q6_K in layers 34, 38 and 39), `ffn_gate_inp_shexp {2048}` (F32), `ffn_gate_shexp` and `ffn_up_shexp {2048, 512}` (Q8_0), `ffn_down_shexp {512, 2048}` (Q8_0).

```
h    = RMSNorm(r) ⊙ post_attention_norm                 # r = x + mixer output
p    = softmax(ffn_gate_inp · h)                        # over 256, no bias   (2082, 2098-2101)
E    = indices of the 8 largest p, descending           # (2166)
w_e  = p_e / max(Σ_{e∈E} p_e, 6.103515625e-5)           # normalization on, no scale (qwen35moe.cpp:506; llama-graph.cpp:2191-2206)
y_e  = W_down_e · (silu(W_gate_e · h) ⊙ (W_up_e · h))   # (2305-2306)
moe  = Σ_{e∈E} w_e · y_e                                # weights applied after down (2380-2409)
sh   = W_down_sh · (silu(W_gate_sh · h) ⊙ (W_up_sh · h))
ffn  = moe + sigmoid(ffn_gate_inp_shexp · h) · sh       # scalar gate per token (529-541)
```

**Residual structure** (`qwen35moe.cpp:181-225`):

```
r      = x + Mixer(RMSNorm(x) ⊙ attn_norm)
x_next = r + FFN(RMSNorm(r) ⊙ post_attention_norm)
```

`post_attention_norm` is the pre-FFN norm; there is no `ffn_norm` tensor.

## 6. MTP block (`blk.40`)

A full-attention block with MoE, plus `nextn.eh_proj {4096, 2048}` (Q8_0), `nextn.enorm`, `nextn.hnorm` and `nextn.shared_head_norm` (F32) (`qwen35moe.cpp:106-139`). Its router `ffn_gate_inp` and shared-expert gate `ffn_gate_inp_shexp` are BF16. The main loop runs `il < n_layer = 40` (`qwen35moe.cpp:177-178`), the MTP tensors are skipped unless `load_mtp` is set (`llama-model.cpp:2972, 3496-3497`), and the MTP graph runs only as `LLM_GRAPH_TYPE_DECODER_MTP`, from a context made with `ctx_type = LLAMA_CONTEXT_TYPE_MTP` (`include/llama.h:226, 377`). Its graph is `llama_model_qwen35moe::graph_mtp` (`qwen35moe.cpp:550-742`); `n_layer_nextn` must be 1 (`:553-554`) and its block index is `il = n_layer = 40` (`:559`).

The GGUF has no `nextn.embed_tokens` and no `nextn.shared_head_head`, so the graph falls back to the main model's `token_embd` (`:584`) and `output` (`:734-737`). The tiny test model has neither either.

**Inputs of the row at position `q`:** the token `t_q` and `h_{q-1}`, the main model's final hidden row at `q − 1` **after** `output_norm`: the main graph's `h_nextn` is `build_norm(cur, model.output_norm)` taken before the head (`:229-233`, "post-norm hidden state feeds both the LM head and the MTP seed"; `llama-graph.h`'s comment says "before", the code wins). The driver feeds it with `llama_set_embeddings_nextn` / `llama_get_embeddings_nextn_ith` (`src/llama-ext.h:93-122`; `common/speculative.cpp:1581, 1637`). A chained draft at `q + 1` takes the draft token and the MTP graph's own `h_nextn`, `m_q` below, in place of `h_q` (`speculative.cpp:1729, 1785-1786`).

```
e = RMSNorm(embd(t_q)) ⊙ enorm                      # embd = main token_embd row, dequantized, unscaled (:584-590, 608)
g = RMSNorm(h_{q-1})  ⊙ hnorm                       # (:605)
u = eh_proj · [e ; g]                               # ggml_concat(e_norm, h_norm, dim 0): e FIRST, g second (:611-614)
a = RMSNorm(u) ⊙ attn_norm                          # (:619)
r = u + Wo · ( Attn(a) ⊙ sigmoid(gate) )            # attention as §3, over the MTP's own KV rows (:617-672)
f = RMSNorm(r) ⊙ post_attention_norm                # (:675)
y = r + FFN(f)                                      # MoE + gated shared expert as §5 (:678-720)
m = RMSNorm(y) ⊙ shared_head_norm                   # h_nextn (:722-728)
logits = output · m                                 # the main model's output (:734-738)
```

- **Attention (`:622-672`):** `[Q ; gate]` is `attn_q · a`, interleaved per head as §3; `Q = RMSNorm_head(Q) ⊙ attn_q_norm`, `K = RMSNorm_head(attn_k · a) ⊙ attn_k_norm`, `V = attn_v · a`, with no norm on `V`. IMRoPE (`ggml_rope_multi`, `n_rot`, the model's sections) is applied to `Q` and `K` at the row's position, **after** the taps `mtp_Qcur_normed` and `mtp_Kcur_normed` (`:634-656`); `build_attn` renames nothing, so those two taps are before RoPE, unlike the main layers' `Qcur`/`Kcur` (§9). `kq_scale = 1/sqrt(head_dim)` (`:659-660`). The row writes its K and V into the MTP's KV and then attends all rows in it, itself included (`:662-664`). It is exactly §3's attention, over the rows the MTP has written (its own KV cache, separate from the main model's), so a position the MTP never ran on is simply absent, not a zero row. The output is `Wo · (pregate ⊙ sigmoid(gate))` (`:666-669`) and `r = that + u`, the residual being the `eh_proj` output, not the normed one (`:617, 671-672`).
- **FFN (`:678-720`):** `build_moe_ffn` with softmax gating, SiLU, normalized top-8 weights and `expert_weights_scale`, as §5; the shared expert is `build_ffn` (SiLU, parallel) gated by `sigmoid(ffn_gate_inp_shexp · f)` (`:696-715`). `ffn_out = moe + shexp_gated`, `post_ffn = ffn_out + r` (`:717-720`).
- **Only the key and value of a row feed later rows.** K and V depend on `u` alone, so a row that only has to be attended later needs `e`, `g`, `eh_proj`, `attn_norm`, `attn_k`, `attn_v`, the K norm, RoPE and the KV write, and nothing else (plan 5b M2). That is a fact of the math above, not of llama.cpp, which computes the whole block for every row and keeps only what it needs.
- **Drafting.** `common/speculative.cpp:1539-1646` runs the MTP over every prompt and verify token and drafts the top candidate of a greedy top-k pass (`p_min` 0 by default, `common.h:333`). Artemis drafts the argmax.

## 7. Types and casts

- f32 activations throughout; no explicit casts in the graph.
- 1-D tensors, `*_norm.weight`, `ffn_gate_inp`, `ffn_gate_inp_shexp`, `ssm_conv1d`, `ssm_a`, `ssm_dt.bias` are never quantized (`conversion/base.py:1084-1121`; `src/llama-quant.cpp:292-325`). This file also keeps `ssm_alpha` and `ssm_beta` in F32.
- **Hidden cast on the CPU backend:** a matmul first converts activations to the weight type's `vec_dot_type`: Q8_K for K-quants, Q8_0 for Q8_0, BF16 for BF16 (`ggml-cpu.c:222-227, 272-275, 311-314, 395-399`). llama.cpp's CPU results therefore carry activation-quantization error that an f32 reference doesn't.
- KV cache f16 by default (`llama-context.cpp:3890-3891`); recurrent conv and DeltaNet states always f32 (`llama-model.cpp:2810-2811`).
- **Activation rounding, measured 2026-10-08.**
  - Both the CPU and the Vulkan backend round matmul activations the same way, so they agree with each other to about 1e-7. They differ from exact f32 by about 4.5e-3 on a Q8_0 projection.
  - The CPU's Q8_0 rounding (x86 AVX2 `quantize_row_q8_0`, `ggml-cpu/arch/x86/quants.c:333-346`): d = amax/127 stored as f16, q = round-half-even(x · 127/amax).
  - The CPU's Q8_K rounding (`quantize_row_q8_K_ref`, `ggml-quants.c`): iscale = −127/max over the signed element of largest magnitude, q = min(127, nearest_int(iscale · x)), d = 1/iscale kept in f32.
- **Flash attention.** Both backends run it (`flash_attn = auto` resolves to enabled). The CPU kernel (`ops.cpp`, `ggml_compute_forward_flash_attn_ext_f16_one_chunk`) works like this:
  - the query is rounded to f16;
  - an online softmax runs in f32;
  - V accumulates in an f16 buffer, rounded after every rescale and fused multiply-add (`ggml_vec_scale_f16`, `ggml_vec_mad_f16`);
  - the result is divided by the f32 sum.

  That is the one-chunk kernel, used when a batch has fewer than 64 query tokens (`GGML_FA_TILE_Q`, `ggml-cpu/common.h:9`). From 64 tokens the CPU uses `flash_attn_ext_tiled` (`ops.cpp:9546-9566`), which keeps the query and the accumulators in f32. Long prompts, and `llama-perplexity` with `-c 4096`, take that path.

## 8. Tokenizer

- `tokenizer.ggml.model = gpt2` (byte-level BPE), `tokenizer.ggml.pre = qwen35` → `LLAMA_VOCAB_PRE_TYPE_QWEN35`, `clean_spaces = false` (`src/llama-vocab.cpp:2382-2385`).
- Pre-tokenizer regex (`llama-vocab.cpp:394-396`), in its tokenizer.json form:
  ```
  (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  ```
  llama.cpp runs it with a hand-written splitter (`src/unicode.cpp:755-883`), not a regex engine.
- BOS 248044 is not added (`tokenizer.ggml.add_bos_token = false`). EOS 248046 (`<|im_end|>`). Padding 248055.
- Token types in this vocab: 27 CONTROL (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, …), 6 USER_DEFINED (`<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<think>`, `</think>`), 243 UNUSED.
- **Special-token matching** (`llama-vocab.cpp:3160-3172, 3385-3396`): CONTROL, USER_DEFINED and UNKNOWN tokens are matched longest-first before BPE. With `parse_special = false`, CONTROL and UNKNOWN are not matched but USER_DEFINED still are.

## 9. Named intermediates (`cb()` names)

Names are `name-il`, or bare `name` outside layers (`llama-context.cpp:2739-2745`). Several calls rename the same tensor, and the last name wins.

- **Global:** `inp_embd`, `model.input_embed`, `result_norm`, `result_output`.
- **Every layer:** `attn_norm`, `attn_residual`, `attn_post_norm`, `ffn_moe_logits`, `ffn_moe_probs`, `ffn_moe_topk`, `ffn_moe_weights_norm`, `ffn_moe_gate`, `ffn_moe_up`, `ffn_moe_swiglu`, `ffn_moe_down`, `ffn_moe_weighted`, `ffn_moe_out`, `ffn_shexp`, `shared_expert_gate_sigmoid`, `ffn_shexp_gated`, `ffn_out`, `l_out`.
- **Full attention:** `Qcur_full`, `Qcur_normed`, `Kcur_normed`, `gate_reshaped`, `Qcur` / `Kcur` / `Vcur` (`Kcur` and `Vcur` appear twice, first as the projection; the last occurrence of each is after RoPE), `attn_pregate`, `gate_sigmoid`, `attn_gated`, `attn_output` (after `wo`).
- **Gated DeltaNet:** `linear_attn_qkv_mixed`, `z`, `beta_sigmoid`, `gate`, `conv_output_silu`, `q_conv_predelta`, `k_conv_predelta`, `v_conv_predelta`, `attn_output` (the raw delta-rule output, before the gated norm), `final_output`, `linear_attn_out`.
- **MTP graph** (`qwen35moe.cpp:550-742`; run only as `LLM_GRAPH_TYPE_DECODER_MTP`, so these names never appear in the main graph's dump). With the block index suffix `-40`: `mtp_tok_embd`, `mtp_hnorm`, `mtp_enorm`, `mtp_concat` (`[e ; g]`, the embedding half first), `mtp_eh_proj`, `mtp_attn_norm`, `mtp_Qcur_full`, `mtp_Qcur_normed` and `mtp_Kcur_normed` (after the per-head norm, **before** RoPE: the RoPE results get no name and `build_attn` renames nothing), `mtp_gate`, `mtp_Vcur`, `mtp_attn_pregate`, `mtp_attn_out` (after `wo`, before the residual), `mtp_attn_residual`, `mtp_attn_post_norm`, then from `build_moe_ffn` and `build_ffn`, **unprefixed** with the same `-40` suffix: `ffn_moe_logits`, `ffn_moe_probs`, `ffn_moe_topk`, `ffn_moe_weights_norm`, `ffn_moe_swiglu` and `ffn_swiglu` (the shared expert's activation; `llama-graph.cpp:2086, 2117, 2169, 2202, 2307, 1909`), then `mtp_ffn_moe_out`, `mtp_ffn_shexp`, `mtp_shared_expert_gate_sigmoid`, `mtp_ffn_shexp_gated`, `mtp_ffn_out`, `mtp_post_ffn`. Bare (no suffix, `cb(..., -1)`): `h_nextn` (`m`, after `shared_head_norm`; the **main** graph also names its post-`output_norm` row `h_nextn` at `:232`, but only as a name that `result_norm` (`:239`) replaces unless `embeddings_nextn` is on and unmasked (`crop_after_nextn`, `llama-graph.h:1066-1068`); with it off, `crop_before_nextn` also crops the last layer to the output rows (`llama-graph.h:1061-1063`). Either way `result_output` is shared, so the two dumps need separate directories), `mtp_shared_head_norm` (the same row after the `inp_out_ids` crop) and `result_output` (the MTP's logits; the main graph has the same name). Artemis's `MtpKey` taps use `mtp_enorm`, `mtp_hnorm`, `mtp_concat`, `mtp_eh_proj`, `mtp_attn_norm`, `mtp_Qcur_normed`, `mtp_Kcur_normed`, `mtp_Vcur`, `mtp_attn_pregate`, `mtp_attn_gated`, `mtp_attn_out`, `mtp_attn_residual`, `mtp_attn_post_norm`, `ffn_moe_logits`, `ffn_moe_probs`, `ffn_moe_topk`, `ffn_moe_weights_norm`, `ffn_moe_swiglu`, `mtp_ffn_moe_out`, `ffn_swiglu`, `mtp_ffn_shexp_gated`, `mtp_ffn_out`, `mtp_post_ffn`, `h_nextn` and `result_output` (25 taps). `mtp_attn_gated` is **oracle-assigned**: llama.cpp leaves `pregate ⊙ sigmoid(gate)` (`:667`) unnamed, and `artemis-oracle mtp-taps` names that `GGML_OP_MUL` node (src[0] = `mtp_attn_pregate-40`) so `wo`'s Q8_0 activation can be forced as the main layers' `attn_gated` is.

Pitfalls when dumping:

- `attn_output-N` means different things in the two layer types, and `norm-N` appears several times per layer.
- In the last layer, rows are cropped to the output tokens before the residual (`qwen35moe.cpp:197-200`). Requesting logits for every position avoids it.
- The CPU backend fuses RMS_NORM with MUL unless `GGML_CPU_DISABLE_FUSION=1` (`ggml-cpu.c:3073-3082, 3928-3929`).
