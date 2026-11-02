//! Decode attention over up to `MAX_TOKENS` queries: query `i` of a step whose first query attends `n_ctx0`
//! cached positions attends `n_ctx0 + i`, and runs the kernel and split count decode would use at that length, so its
//! partials and output equal one decode step's (`attention_tokens_writes_decodes_bits_per_query` reports it).
use super::MAX_TOKENS;
use crate::{
    dispatch::{attn_decode_kernel, attn_splits, groups, ATTN_COMBINE_DIMS},
    Kernel, Kernels,
};
use vk::Recorder;

/// Push constants of `attn_decode_tokens` and `attn_decode_gqa_tokens`. The dispatch serves queries `first..tokens`;
/// query `i` attends `n_ctx0 + i` positions, reads `q` at `[i][n_head·hd]` and writes `part` at
/// `[i][n_head][max_split][hd + 2]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AttnTokensPush {
    pub q: u64,
    pub kc: u64,
    pub vc: u64,
    pub part: u64,
    pub n_ctx0: u32,
    pub first: u32,
    pub tokens: u32,
    pub hd: u32,
    pub group: u32,
    pub kv_width: u32,
    pub max_split: u32,
    pub scale: f32,
}
push_constants!(AttnTokensPush: u64, u64, u64, u64, u32, u32, u32, u32, u32, u32, u32, f32);

/// Push constants of `attn_combine_tokens`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AttnCombineTokensPush {
    pub part: u64,
    pub y: u64,
    pub n_ctx0: u32,
    pub tokens: u32,
    pub hd: u32,
    pub max_split: u32,
}
push_constants!(AttnCombineTokensPush: u64, u64, u32, u32, u32, u32);

impl Kernels {
    /// `part` (`[tokens][n_head][max_split][hd + 2]`) ← each chunk's partial attention for every query head and each of
    /// the `tokens` queries in `q` (`[tokens][n_head·hd]`), query `i` at position `n_ctx0 + i - 1`, attending the first
    /// `n_ctx0 + i` cached positions (causal within the step; the step's keys and values must already be in the caches).
    /// Each query takes decode's kernel for its own length (`attn_decode_kernel`): `attn_decode_tokens` up to
    /// `ATTN_GQA_FROM` positions, `attn_decode_gqa_tokens` past it, so a step records one dispatch, or two where it
    /// straddles the switch. A chunk past a query's last is left unwritten. `attn_combine_tokens` merges the partials.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_tokens(
        &self,
        r: &mut Recorder,
        q: u64,
        kc: u64,
        vc: u64,
        part: u64,
        n_ctx0: usize,
        tokens: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        max_split: usize,
    ) {
        assert!(
            (1..=MAX_TOKENS).contains(&tokens),
            "attn_decode_tokens: 1 to {MAX_TOKENS} queries, not {tokens}"
        );
        assert!(
            hd <= 256 && n_ctx0 >= 1 && n_kv >= 1 && n_head.is_multiple_of(n_kv),
            "attn_decode_tokens: heads of at most 256, at least one position, whole query-head groups"
        );
        let last = n_ctx0 + tokens - 1;
        assert!(
            max_split >= attn_splits(last),
            "attn_decode_tokens: {max_split} chunks of partials for a query over {last} positions ({} needed)",
            attn_splits(last)
        );
        // Decode's choice per query is monotone in the length: the first `plain` queries run attn_decode's arithmetic,
        // the rest attn_decode_gqa's (`attn_decode_kernel` also decides whether the shape allows the latter).
        let plain = (0..tokens)
            .take_while(|&i| attn_decode_kernel(n_ctx0 + i, n_head, n_kv, hd) == Kernel::AttnDecode)
            .count();
        let push = |first: usize, end: usize| AttnTokensPush {
            q,
            kc,
            vc,
            part,
            n_ctx0: n_ctx0 as u32,
            first: first as u32,
            tokens: end as u32,
            hd: hd as u32,
            group: (n_head / n_kv) as u32,
            kv_width: (n_kv * hd) as u32,
            max_split: max_split as u32,
            scale: 1.0 / (hd as f32).sqrt(),
        };
        let n_split = groups(attn_splits(last), 1);
        if plain > 0 {
            r.dispatch(
                self.pipe(Kernel::AttnDecodeTokens),
                &push(0, plain),
                [n_split, groups(n_head, 1), 1],
            );
        }
        if plain < tokens {
            assert!(
                kc.is_multiple_of(16),
                "attn_decode_gqa_tokens: a 16-byte-aligned key cache"
            );
            // One workgroup per query, chunk and KV head, the queries side by side so a chunk's are neighbours.
            r.dispatch(
                self.pipe(Kernel::AttnDecodeGqaTokens),
                &push(plain, tokens),
                [groups(tokens - plain, 1), n_split, groups(n_kv, 1)],
            );
        }
    }

    /// `y` (`[tokens][n_head·hd]`) ← every query head's attention output for each of the `tokens` queries, from
    /// `attn_decode_tokens`'s partials, each query merged over its own `attn_splits(n_ctx0 + i)` chunks.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_combine_tokens(
        &self,
        r: &mut Recorder,
        part: u64,
        y: u64,
        n_ctx0: usize,
        tokens: usize,
        n_head: usize,
        hd: usize,
        max_split: usize,
    ) {
        assert!(
            (1..=MAX_TOKENS).contains(&tokens),
            "attn_combine_tokens: 1 to {MAX_TOKENS} queries, not {tokens}"
        );
        assert!(
            hd <= 256 && n_ctx0 >= 1 && max_split >= attn_splits(n_ctx0 + tokens - 1),
            "attn_combine_tokens: heads of at most 256, at least one position, {max_split} chunks of partials"
        );
        let push = AttnCombineTokensPush {
            part,
            y,
            n_ctx0: n_ctx0 as u32,
            tokens: tokens as u32,
            hd: hd as u32,
            max_split: max_split as u32,
        };
        r.dispatch(
            self.pipe(Kernel::AttnCombineTokens),
            &push,
            [
                groups(n_head, 1),
                tokens as u32,
                groups(hd, ATTN_COMBINE_DIMS),
            ],
        );
    }
}
