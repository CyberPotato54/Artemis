//! The token kernels: decode's kernels with a token dimension, for a speculative verify step of up to
//! `MAX_TOKENS` tokens. This module holds what their dispatches and the engine's `VerifyLayout` share: the token cap
//! and the sizes of the scratch the kernels write.

pub mod attention;
pub mod matvec;

/// The most tokens one verify step carries: the last accepted token and up to 15 drafts. The token kernels' general
/// pipelines (their shaders' `TOKENS` default, pinned equal by a test) and every token kernel's dispatch assert are
/// sized by it.
pub const MAX_TOKENS: usize = 16;

/// The most (token, choice) pairs one layer's `moe_unique` and `experts_*_tokens` take: one invocation each in one
/// workgroup of 256. A verify of `MAX_TOKENS` tokens needs `MAX_TOKENS` times the model's experts per token to fit.
pub const MAX_PAIRS: usize = 256;

/// The token counts the token kernels (`experts_gate_up_tokens`, `experts_down_tokens`, `matvec_tokens_*`) have
/// pipelines specialized to (their shaders' `TOKENS`, specialization constant 0): the verify step's usual sizes. Other
/// counts run the general pipeline, `TOKENS` = `MAX_TOKENS`.
pub const SPECIALIZED_TOKENS: std::ops::RangeInclusive<usize> = 2..=5;

// The dispatches take at most 16 choices a token, so `MAX_TOKENS` tokens always fit `MAX_PAIRS`.
const _: () = assert!(MAX_TOKENS * 16 <= MAX_PAIRS);

/// Bytes of the attention partials of `tokens` queries: `[tokens][n_head][max_split][hd + 2]` f32 (per split: `hd`
/// weighted values, the max and the sum). `attn_decode_tokens` writes them and `attn_combine_tokens` reads them.
pub fn attn_tokens_part_bytes(n_head: usize, hd: usize, max_split: usize, tokens: usize) -> u64 {
    4 * (tokens as u64) * (n_head as u64) * (max_split as u64) * (hd as u64 + 2)
}

/// Bytes of the list of unique experts a layer's `tokens` tokens chose `k` each: `[count: u32]`, then per unique
/// expert `u` (at most one for every token and choice) `[expert: u32, mask: u32]`, the mask naming the tokens that
/// chose it. `moe_unique` writes it and `experts_*_tokens` read it.
pub fn moe_unique_bytes(tokens: usize, k: usize) -> u64 {
    4 + 8 * (tokens as u64) * (k as u64)
}

pub(crate) mod experts;
