//! One push-constant struct and one recording helper per kernel. Each struct's fields follow its shader's `Push`
//! block: 64-bit addresses first, then 32-bit values, so `#[repr(C)]` lays them out as std430 does. Helpers record
//! the dispatch only; the caller places barriers.
use crate::{wtype_code, Kernel, Kernels};
/// The weight types the slot layouts and dispatches take (re-exported from `qwen36::quant`, so `cpu_expert` and the
/// tests name it through `kernels`).
pub use qwen36::quant::WType;
use vk::{Recorder, MAX_WORKGROUPS};

/// `n` workgroups, checked against Vulkan's per-dimension limit rather than truncated to a u32.
fn count(n: usize) -> u32 {
    groups(n, 1)
}

/// Workgroups covering `n` items at `per` per workgroup. Vulkan guarantees `MAX_WORKGROUPS` per dimension; a prefill
/// chunk's widest elementwise kernel needs 32,768.
pub fn groups(n: usize, per: usize) -> u32 {
    let g = n.div_ceil(per);
    assert!(
        g <= MAX_WORKGROUPS as usize,
        "{n} items at {per} a workgroup need {g} workgroups"
    );
    g as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmbedRowPush {
    pub row: u64,
    pub y: u64,
    pub n: u32,
    pub wtype: u32,
}
push_constants!(EmbedRowPush: u64, u64, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmbedRowsPush {
    pub rows: u64,
    pub y: u64,
    pub n: u32,
    pub wtype: u32,
}
push_constants!(EmbedRowsPush: u64, u64, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ArgmaxRowsPush {
    pub logits: u64,
    pub ids: u64,
    pub rows: u64,
    pub embd: u64,
    pub n: u32,
    pub row_bytes: u32,
    pub slot: u32,
    pub pad: u32,
}
push_constants!(ArgmaxRowsPush: u64, u64, u64, u64, u32, u32, u32, u32);

/// `argmax_rows`' id for a row with no finite value.
pub const NO_DRAFT: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AddPush {
    pub a: u64,
    pub b: u64,
    pub y: u64,
    pub n: u32,
    pub pad: u32,
}
push_constants!(AddPush: u64, u64, u64, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RmsnormPush {
    pub x: u64,
    pub w: u64,
    pub y: u64,
    pub n: u32,
    pub eps: f32,
}
push_constants!(RmsnormPush: u64, u64, u64, u32, f32);

impl Kernels {
    /// `y` ← the `n` values of the embedding row at address `row`, dequantized from type `ty`.
    pub fn embed_row(&self, r: &mut Recorder, row: u64, ty: WType, n: usize, y: u64) {
        let push = EmbedRowPush {
            row,
            y,
            n: n as u32,
            wtype: wtype_code(ty),
        };
        r.dispatch(self.pipe(Kernel::EmbedRow), &push, [groups(n, 256), 1, 1]);
    }

    /// `y` (`[tokens][n]`) ← each token's embedding row dequantized from type `ty`; `rows` holds the rows' addresses,
    /// one u64 per token.
    pub fn embed_rows(
        &self,
        r: &mut Recorder,
        rows: u64,
        ty: WType,
        n: usize,
        y: u64,
        tokens: usize,
    ) {
        let push = EmbedRowsPush {
            rows,
            y,
            n: n as u32,
            wtype: wtype_code(ty),
        };
        r.dispatch(
            self.pipe(Kernel::EmbedRows),
            &push,
            [groups(n, 256), count(tokens), 1],
        );
    }

    /// `ids[slot]` ← the index of `logits`' largest finite value (ties to the lower index), or `NO_DRAFT` when the row has
    /// no finite value; `rows[slot]` ← `embd + id · row_bytes` (row 0's address for `NO_DRAFT`): the next draft row's
    /// embedding address, for `embed_rows`.
    #[allow(clippy::too_many_arguments)]
    pub fn argmax_rows(
        &self,
        r: &mut Recorder,
        logits: u64,
        n: usize,
        ids: u64,
        rows: u64,
        embd: u64,
        row_bytes: usize,
        slot: usize,
    ) {
        let (n, row_bytes, slot) = (
            u32::try_from(n).expect("n fits u32"),
            u32::try_from(row_bytes).expect("row_bytes fits u32"),
            u32::try_from(slot).expect("slot fits u32"),
        );
        let push = ArgmaxRowsPush {
            logits,
            ids,
            rows,
            embd,
            n,
            row_bytes,
            slot,
            pad: 0,
        };
        r.dispatch(self.pipe(Kernel::ArgmaxRows), &push, [1, 1, 1]);
    }

    /// `y` ← a + b over `n` values.
    pub fn add(&self, r: &mut Recorder, a: u64, b: u64, y: u64, n: usize) {
        let push = AddPush {
            a,
            b,
            y,
            n: n as u32,
            pad: 0,
        };
        r.dispatch(self.pipe(Kernel::Add), &push, [groups(n, 256), 1, 1]);
    }

    /// `y` ← x / sqrt(mean(x²) + eps) ⊙ w over `n` values, in one workgroup.
    pub fn rmsnorm(&self, r: &mut Recorder, x: u64, w: u64, y: u64, n: usize, eps: f32) {
        self.rmsnorm_tokens(r, x, w, y, n, eps, 1);
    }

    /// `rmsnorm` for each of `tokens` vectors of `n` values (`[tokens][n]`), one workgroup each.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_tokens(
        &self,
        r: &mut Recorder,
        x: u64,
        w: u64,
        y: u64,
        n: usize,
        eps: f32,
        tokens: usize,
    ) {
        let push = RmsnormPush {
            x,
            w,
            y,
            n: n as u32,
            eps,
        };
        r.dispatch(self.pipe(Kernel::Rmsnorm), &push, [count(tokens), 1, 1]);
    }
}
/// A weight matrix on the GPU: `rows` rows of `cols` values of type `ty`, row after row, at device address `addr`.
#[derive(Clone, Copy, Debug)]
pub struct MatW {
    pub addr: u64,
    pub ty: WType,
    pub rows: usize,
    pub cols: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MatvecPush {
    pub w: u64,
    pub x: u64,
    pub y: u64,
    pub rows: u32,
    pub k: u32,
}
push_constants!(MatvecPush: u64, u64, u64, u32, u32);

impl Kernels {
    /// `y` ← W·x: `w.rows` values from `w.cols` inputs.
    pub fn matvec(&self, r: &mut Recorder, w: MatW, x: u64, y: u64) {
        let per = w.ty.block().0.max(32);
        assert!(
            w.cols.is_multiple_of(per),
            "a matvec over {} rows of {} values: rows hold whole {per}-value units",
            w.ty.name(),
            w.cols
        );
        // `unit_dot` reads x, and F32, BF16, Q4_K and Q5_K rows, 16 bytes at a time; Q8_0 and Q6_K rows two bytes at a
        // time.
        let wide = matches!(w.ty, WType::F32 | WType::Bf16 | WType::Q4K | WType::Q5K);
        let need = if wide { 16 } else { 2 };
        assert!(
            x.is_multiple_of(16) && w.addr.is_multiple_of(need),
            "a matvec over {} at {:#x} with x at {x:#x}: x must be 16-byte aligned and the weights {need}-byte aligned",
            w.ty.name(),
            w.addr
        );
        let push = MatvecPush {
            w: w.addr,
            x,
            y,
            rows: w.rows as u32,
            k: w.cols as u32,
        };
        r.dispatch(
            self.pipe(Kernel::matvec(w.ty)),
            &push,
            [groups(w.rows, 8), 1, 1],
        );
    }
}

/// Rows, and input vectors, per `matmul_{type}` workgroup.
pub const MATMUL_TILE: usize = 64;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MatmulPush {
    pub w: u64,
    pub x: u64,
    pub y: u64,
    pub rows: u32,
    pub k: u32,
    pub tokens: u32,
    pub pad: u32,
}
push_constants!(MatmulPush: u64, u64, u64, u32, u32, u32, u32);

impl Kernels {
    /// `y` (`[tokens][w.rows]`) ← W·x for each of the `tokens` vectors of `w.cols` values at `x`. Each output sums its
    /// row's products in order whatever the batch, so a token's outputs don't depend on the other tokens in it.
    pub fn matmul(&self, r: &mut Recorder, w: MatW, x: u64, y: u64, tokens: usize) {
        self.record_matmul(r, Kernel::matmul(w.ty), w, x, y, tokens);
    }

    /// The dispatch `matmul` and `matmul_cm` share: kernel `k` (one of the two families, for `w`'s type) over `w.rows`
    /// rows and `tokens` inputs, `MATMUL_TILE` of each per workgroup. The assert names the kernel.
    fn record_matmul(&self, r: &mut Recorder, k: Kernel, w: MatW, x: u64, y: u64, tokens: usize) {
        assert!(
            w.cols.is_multiple_of(w.ty.block().0.max(32)),
            "{}: a matmul over {} rows of {} values: cols must be a multiple of the weight type's block and of 32",
            k.name(),
            w.ty.name(),
            w.cols
        );
        let push = MatmulPush {
            w: w.addr,
            x,
            y,
            rows: w.rows as u32,
            k: w.cols as u32,
            tokens: tokens as u32,
            pad: 0,
        };
        r.dispatch(
            self.pipe(k),
            &push,
            [groups(w.rows, MATMUL_TILE), groups(tokens, MATMUL_TILE), 1],
        );
    }
}

impl Kernels {
    /// `matmul` on cooperative matrices, every input split into two f16s: `y`
    /// (`[tokens][w.rows]`) ← W·x for each of the `tokens` vectors of `w.cols` values at `x`. The split bounds each
    /// input, not the outputs: hi + lo is within about 2⁻²² of the input's f32 value. The outputs measured within about
    /// 7e-7 relative L2 of the twin's (worst 7.2e-7 in the GPU run), inside the kernel's 1e-5 tolerance.
    /// Each 32-value slice of k is summed from zero on the cooperative matrices and then added to the output with an
    /// ordinary f32 add: the hardware's own accumulate measured biased, its error growing linearly with k, where
    /// per-slice sums grow as √k. A token's outputs don't depend on the batch. Panics on a device without the
    /// cooperative-matrix kernels (`Kernels::coop`).
    pub fn matmul_cm(&self, r: &mut Recorder, w: MatW, x: u64, y: u64, tokens: usize) {
        self.record_matmul(r, Kernel::matmul_cm(w.ty), w, x, y, tokens);
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ConvStepPush {
    pub x: u64,
    pub w: u64,
    pub state: u64,
    pub y: u64,
    pub n: u32,
    pub kw: u32,
}
push_constants!(ConvStepPush: u64, u64, u64, u64, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GdnGatesPush {
    pub b: u64,
    pub alpha: u64,
    pub dt_bias: u64,
    pub a: u64,
    pub beta: u64,
    pub g: u64,
    pub n: u32,
    pub pad: u32,
}
push_constants!(GdnGatesPush: u64, u64, u64, u64, u64, u64, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct L2normPush {
    pub x: u64,
    pub n: u32,
    pub eps: f32,
    pub stride: u32,
    pub pad: u32,
}
push_constants!(L2normPush: u64, u32, f32, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GdnStepPush {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub g: u64,
    pub beta: u64,
    pub state: u64,
    pub o: u64,
    pub dk: u32,
    pub dv: u32,
    pub n_group: u32,
    pub scale: f32,
}
push_constants!(GdnStepPush: u64, u64, u64, u64, u64, u64, u64, u32, u32, u32, f32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ConvSeqPush {
    pub x: u64,
    pub w: u64,
    pub state: u64,
    pub y: u64,
    pub n: u32,
    pub kw: u32,
    pub tokens: u32,
    /// 1 writes the final state back, 0 leaves it as it was (the verify pass).
    pub store: u32,
}
push_constants!(ConvSeqPush: u64, u64, u64, u64, u32, u32, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GdnSeqPush {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub g: u64,
    pub beta: u64,
    pub state: u64,
    pub o: u64,
    pub dk: u32,
    pub dv: u32,
    pub n_group: u32,
    pub scale: f32,
    pub tokens: u32,
    pub stride: u32,
    /// 1 writes the final state back, 0 leaves it as it was (the verify pass).
    pub store: u32,
    pub pad: u32,
}
push_constants!(GdnSeqPush: u64, u64, u64, u64, u64, u64, u64, u32, u32, u32, f32, u32, u32, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GatedRmsnormPush {
    pub o: u64,
    pub z: u64,
    pub w: u64,
    pub y: u64,
    pub dv: u32,
    pub eps: f32,
}
push_constants!(GatedRmsnormPush: u64, u64, u64, u64, u32, f32);

impl Kernels {
    /// `y` ← the post-SiLU causal conv of `n` channels of `x` with kernel `w` (`[n][kw]`); `state` (`[n][kw − 1]`)
    /// shifts in `x`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_step(
        &self,
        r: &mut Recorder,
        x: u64,
        w: u64,
        state: u64,
        y: u64,
        n: usize,
        kw: usize,
    ) {
        let push = ConvStepPush {
            x,
            w,
            state,
            y,
            n: n as u32,
            kw: kw as u32,
        };
        r.dispatch(self.pipe(Kernel::ConvStep), &push, [groups(n, 256), 1, 1]);
    }

    /// `beta` ← sigmoid(b) and `g` ← a ⊙ softplus(alpha + dt_bias), over `n` value heads.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gates(
        &self,
        r: &mut Recorder,
        b: u64,
        alpha: u64,
        dt_bias: u64,
        a: u64,
        beta: u64,
        g: u64,
        n: usize,
    ) {
        self.gdn_gates_tokens(r, b, alpha, dt_bias, a, beta, g, n, 1);
    }

    /// `gdn_gates` for each of `tokens` tokens: `b`, `alpha`, `beta` and `g` hold `[tokens][n]`; `dt_bias` and `a`
    /// are shared.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gates_tokens(
        &self,
        r: &mut Recorder,
        b: u64,
        alpha: u64,
        dt_bias: u64,
        a: u64,
        beta: u64,
        g: u64,
        n: usize,
        tokens: usize,
    ) {
        let push = GdnGatesPush {
            b,
            alpha,
            dt_bias,
            a,
            beta,
            g,
            n: n as u32,
            pad: 0,
        };
        r.dispatch(
            self.pipe(Kernel::GdnGates),
            &push,
            [groups(n, 64), count(tokens), 1],
        );
    }

    /// Each of `groups` consecutive vectors of `n` values at `x`, L2-normalized in place.
    pub fn l2norm(&self, r: &mut Recorder, x: u64, groups: usize, n: usize, eps: f32) {
        self.l2norm_tokens(r, x, groups, n, eps, 1, 0);
    }

    /// `l2norm` for each of `tokens` tokens, whose `groups` vectors start `stride` values apart.
    #[allow(clippy::too_many_arguments)]
    pub fn l2norm_tokens(
        &self,
        r: &mut Recorder,
        x: u64,
        groups: usize,
        n: usize,
        eps: f32,
        tokens: usize,
        stride: usize,
    ) {
        let push = L2normPush {
            x,
            n: n as u32,
            eps,
            stride: stride as u32,
            pad: 0,
        };
        r.dispatch(
            self.pipe(Kernel::L2norm),
            &push,
            [count(groups), count(tokens), 1],
        );
    }

    /// `o` ← one delta-rule step for `heads` value heads of `dv` values over keys of `dk`, updating `state`.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_step(
        &self,
        r: &mut Recorder,
        q: u64,
        k: u64,
        v: u64,
        g: u64,
        beta: u64,
        state: u64,
        o: u64,
        heads: usize,
        dk: usize,
        dv: usize,
        n_group: usize,
    ) {
        assert!(
            dk <= 128 && dv <= 128,
            "gdn_step holds at most 128 keys and values per head"
        );
        // Heads of 128 keys read and write the state 16 bytes at a time.
        assert!(
            dk != 128 || state.is_multiple_of(16),
            "gdn_step over heads of 128 keys needs a 16-byte-aligned state, not {state:#x}"
        );
        let push = GdnStepPush {
            q,
            k,
            v,
            g,
            beta,
            state,
            o,
            dk: dk as u32,
            dv: dv as u32,
            n_group: n_group as u32,
            scale: 1.0 / (dk as f32).sqrt(),
        };
        r.dispatch(self.pipe(Kernel::GdnStep), &push, [count(heads), 1, 1]);
    }

    /// `y` (`[tokens][n]`) ← `conv_step` for each of the `tokens` inputs at `x` (`[tokens][n]`) in order, the state
    /// carried from one to the next. With `store` false the state is read and never written (the verify pass keeps
    /// state set A for its replay); the outputs are the same.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_seq(
        &self,
        r: &mut Recorder,
        x: u64,
        w: u64,
        state: u64,
        y: u64,
        n: usize,
        kw: usize,
        tokens: usize,
        store: bool,
    ) {
        assert!((2..=8).contains(&kw), "conv_seq holds a window of 2 to 8");
        let push = ConvSeqPush {
            x,
            w,
            state,
            y,
            n: n as u32,
            kw: kw as u32,
            tokens: tokens as u32,
            store: store as u32,
        };
        r.dispatch(self.pipe(Kernel::ConvSeq), &push, [groups(n, 256), 1, 1]);
    }

    /// `o` (`[tokens][heads][dv]`) ← `gdn_step` for each of `tokens` tokens in order, the state carried. Token t's q,
    /// k and v start at `q`, `k` and `v` plus t·`stride` values; `g` and `beta` hold `[tokens][heads]`. With `store`
    /// false the state is read and never written; the outputs are the same.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_seq(
        &self,
        r: &mut Recorder,
        q: u64,
        k: u64,
        v: u64,
        g: u64,
        beta: u64,
        state: u64,
        o: u64,
        heads: usize,
        dk: usize,
        dv: usize,
        n_group: usize,
        tokens: usize,
        stride: usize,
        store: bool,
    ) {
        assert!(
            dk <= 128 && dv <= 128,
            "gdn_seq holds at most 128 keys and values per head"
        );
        let push = GdnSeqPush {
            q,
            k,
            v,
            g,
            beta,
            state,
            o,
            dk: dk as u32,
            dv: dv as u32,
            n_group: n_group as u32,
            scale: 1.0 / (dk as f32).sqrt(),
            tokens: tokens as u32,
            stride: stride as u32,
            store: store as u32,
            pad: 0,
        };
        r.dispatch(self.pipe(Kernel::GdnSeq), &push, [count(heads), 1, 1]);
    }

    /// `y` ← per head of `dv` values, RMSNorm(o) ⊙ w ⊙ silu(z). Vulkan guarantees only `MAX_WORKGROUPS` workgroups per
    /// dimension, so a run of more heads (prefill's chunk of tokens × heads) is recorded as several dispatches, each
    /// offsetting `o`, `z` and `y` to its first head; `w` is shared by every head. Each head is still one workgroup
    /// with the same arithmetic, so the result is bit-identical however the heads are cut.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_rmsnorm(
        &self,
        r: &mut Recorder,
        o: u64,
        z: u64,
        w: u64,
        y: u64,
        heads: usize,
        dv: usize,
        eps: f32,
    ) {
        let mut h0 = 0;
        while h0 < heads {
            let run = (heads - h0).min(MAX_WORKGROUPS as usize);
            let off = (h0 * dv * 4) as u64;
            let push = GatedRmsnormPush {
                o: o + off,
                z: z + off,
                w,
                y: y + off,
                dv: dv as u32,
                eps,
            };
            r.dispatch(self.pipe(Kernel::GatedRmsnorm), &push, [count(run), 1, 1]);
            h0 += run;
        }
    }
}
/// Cached positions per `attn_decode` workgroup. The split count depends only on the context length.
pub const ATTN_CHUNK: usize = 256;

/// Output dimensions per `attn_combine` and `attn_combine_tokens` workgroup (`WG` in both shaders): a head's merge
/// spreads over `hd / ATTN_COMBINE_DIMS` workgroups rounded up, each dimension's sum one thread's in split order.
pub const ATTN_COMBINE_DIMS: usize = 64;

/// `attn_decode`'s chunks for `n_ctx` cached positions.
pub fn attn_splits(n_ctx: usize) -> usize {
    n_ctx.div_ceil(ATTN_CHUNK)
}

/// Past this many cached positions decode's attention runs `attn_decode_gqa`, one workgroup per (chunk, KV head) for
/// every query head reading that KV head, where `attn_decode` takes one per (chunk, query head) and so reads each cached
/// row once per query head. The two kernels are written operation for operation alike; neither's
/// sums are `precise`, so the same bits also rest on the compiler and driver, and
/// `the_gqa_attention_writes_attn_decodes_bits` (test 1's block) is what pins them on this driver. Measured on the RX
/// 9070 XT on the kernels before the K-tile rework (a call on the real model's shapes, best of three; the tiled GQA
/// kernel is about 1.5x faster at 4,096, so the crossover may now sit lower): 2,048 positions 38.8 µs for
/// `attn_decode` against 50.4 for the GQA kernel, 3,072 53.2 against 54.7, 4,096 74.2 against 58.4. Plan v012 measured
/// the tiled kernel (`decode_speed`, median of 1,000 with `attn_combine`): 512 positions 27.6 µs for `attn_decode`
/// against 38.4, 2,048 41.7 against 40.6, so the step keeps this threshold. The verify step does not: see
/// `attn_tokens_kernel`.
pub const ATTN_GQA_FROM: usize = 3_072;

/// The most query heads a KV head may serve for `attn_decode_gqa` (its `GMAX`); a model with more runs `attn_decode`.
pub const ATTN_GQA_GROUP: usize = 8;

/// `attn_decode_gqa`'s and `attn_decode_gqa_tokens`'s shared memory: the group's queries, then its weights, 8 heads ×
/// 256 floats (each shader's `qw`, 8,192 bytes), and one K tile of 32 positions × 128 f16 values (`attn_decode_gqa`) or
/// 64 positions × 64 (`attn_decode_gqa_tokens`) (`kt`, 8,192). Exactly Vulkan's 16,384, so not in
/// `BIG_SHARED_KERNELS`; change it with both shaders' `qw` (`GMAX`) and `kt`.
pub const ATTN_GQA_SHARED_BYTES: u32 = 16_384;

/// The shapes `attn_decode_gqa` serves: at most `ATTN_GQA_GROUP` query heads to a KV head, heads a multiple of 32
/// values.
fn gqa_takes(n_head: usize, n_kv: usize, hd: usize) -> bool {
    n_head / n_kv <= ATTN_GQA_GROUP && hd.is_multiple_of(32)
}

/// Which kernel computes decode attention's partials over `n_ctx` cached positions: `attn_decode_gqa` past
/// `ATTN_GQA_FROM` positions for heads of a multiple of 32 values and at most `ATTN_GQA_GROUP` query heads to a KV head,
/// `attn_decode` otherwise. Both write the same partials, bit for bit, on this driver, as
/// `the_gqa_attention_writes_attn_decodes_bits` pins (see `ATTN_GQA_FROM`).
pub fn attn_decode_kernel(n_ctx: usize, n_head: usize, n_kv: usize, hd: usize) -> Kernel {
    if n_ctx > ATTN_GQA_FROM && gqa_takes(n_head, n_kv, hd) {
        Kernel::AttnDecodeGqa
    } else {
        Kernel::AttnDecode
    }
}

/// Which kernel computes the verify step's attention partials (`attn_decode_tokens`): `attn_decode_gqa_tokens` at any
/// length for a shape `attn_decode_gqa` serves, `attn_decode_tokens` otherwise. Each query's partials still equal the
/// ones decode writes at its length with `attn_decode_kernel`'s choice, because `attn_decode_gqa` writes
/// `attn_decode`'s bits (`the_gqa_attention_writes_attn_decodes_bits`). `attn_decode_tokens` loops over the queries
/// in each workgroup and reads every cached row once per query head and query; the GQA form takes a workgroup per
/// query and reads a row once per query. On the real model's shapes (`decode_speed`, median of 1,000 with
/// `attn_combine_tokens`, plan v012): 512 positions at t = 2 47.7 against 32.2 µs and at t = 5 97.8 against 32.2;
/// 2,048 positions at t = 2 90.9 against 35.3 and at t = 5 225.5 against 41.2.
///
/// The verify step runs the GQA form at every length, so its rows match decode's only because the compiler treats
/// the two kernels alike; on another driver a difference shows at every length, not at one crossing.
pub fn attn_tokens_kernel(n_head: usize, n_kv: usize, hd: usize) -> Kernel {
    if gqa_takes(n_head, n_kv, hd) {
        Kernel::AttnDecodeGqaTokens
    } else {
        Kernel::AttnDecodeTokens
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct QkNormRopePush {
    pub qfull: u64,
    pub k: u64,
    pub q: u64,
    pub gate: u64,
    pub q_norm: u64,
    pub k_norm: u64,
    pub rope: u64,
    pub hd: u32,
    pub n_head: u32,
    pub n_rot: u32,
    pub eps: f32,
}
push_constants!(QkNormRopePush: u64, u64, u64, u64, u64, u64, u64, u32, u32, u32, f32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct KvWritePush {
    pub k: u64,
    pub v: u64,
    pub kc: u64,
    pub vc: u64,
    pub pos: u32,
    pub width: u32,
}
push_constants!(KvWritePush: u64, u64, u64, u64, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AttnDecodePush {
    pub q: u64,
    pub kc: u64,
    pub vc: u64,
    pub part: u64,
    pub n_ctx: u32,
    pub hd: u32,
    pub group: u32,
    pub kv_width: u32,
    pub n_split: u32,
    pub scale: f32,
}
push_constants!(AttnDecodePush: u64, u64, u64, u64, u32, u32, u32, u32, u32, f32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AttnCombinePush {
    pub part: u64,
    pub y: u64,
    pub hd: u32,
    pub n_split: u32,
}
push_constants!(AttnCombinePush: u64, u64, u32, u32);

/// Queries per `attn_prefill` workgroup, and the cached positions it takes at a time, in blocks aligned from
/// position 0. They are `QB` and `KB` in `attn_prefill.comp`: change them together.
pub const ATTN_QUERIES: usize = 16;
pub const ATTN_KEYS: usize = 32;

/// Rows a workgroup of `attn_prefill_cm` takes, and keys a block, aligned from position 0. They are `QB` and `KB` in its
/// shader: change them together (and `ATTN_CM_SHARED_BYTES` with the shader's arrays).
pub const ATTN_CM_QUERIES: usize = 64;
pub const ATTN_CM_KEYS: usize = 32;

/// Shared memory `attn_prefill.comp` declares, in bytes: kv 16,896 (32 rows × 33 uvec4 of f16s, keys and values in
/// turn) + ss 2,048 (16 × 32 floats) + ms, ls and as 192 (3 × 16 floats) = 19,136. Vulkan guarantees only 16,384, so
/// `Kernels::new` checks it against the device's `maxComputeSharedMemorySize` (32,768 on the RX 9070 XT). Change it
/// with the shader's arrays.
pub const ATTN_SHARED_BYTES: u32 = 19_136;

/// Shared memory `moe_group.comp` declares, in bytes: sid 16,384 (4,096 uints) + start 1,024 (256 uints) +
/// first_tile 1,024 (256 uints) = 18,432. Checked by `Kernels::new`; change it with the shader's arrays.
pub const MOE_GROUP_SHARED_BYTES: u32 = 18_432;

/// Shared memory `experts_matmul_gate_up.comp` declares, in bytes: wg, wu and xs, 3 x 2,048 floats = 24,576, the most
/// of the f32 kernels. Checked by `Kernels::new`; change it with the shader's arrays.
pub const EXPERTS_GATE_UP_SHARED_BYTES: u32 = 24_576;

/// Every kernel that declares more than Vulkan's guaranteed 16,384 bytes of shared memory, with what it declares.
pub const BIG_SHARED_KERNELS: [(&str, u32); 3] = [
    ("attn_prefill", ATTN_SHARED_BYTES),
    ("moe_group", MOE_GROUP_SHARED_BYTES),
    ("experts_matmul_gate_up", EXPERTS_GATE_UP_SHARED_BYTES),
];

/// Shared memory `matmul_cm.glsl` declares, in bytes: one array of 4 × 1,280 uints (the W and X tiles' hi and lo halves,
/// each 64 rows of 32 values padded to 40 f16s), whose first 4,352 uints then hold the [64][68] f32 output stage:
/// 20,480. Over Vulkan's 16,384, so it is in `COOP_SHARED_KERNELS`; change it with the shader's array.
pub const MATMUL_CM_SHARED_BYTES: u32 = 20_480;

/// Shared memory `experts_matmul_cm_gate_up.comp` declares, in bytes: 6 × 1,280 uints (the gate, up and x tiles' hi and
/// lo halves), whose first 4,352 uints then hold the output stage: 30,720, the most of any kernel. Change it with the
/// shader's array.
pub const EXPERTS_CM_GATE_UP_SHARED_BYTES: u32 = 30_720;

/// Shared memory `experts_matmul_cm_down.comp` declares, in bytes: 4 × 1,280 uints (the down and act tiles' hi and lo
/// halves), then the output stage: 20,480. Change it with the shader's array.
pub const EXPERTS_CM_DOWN_SHARED_BYTES: u32 = 20_480;

/// Shared memory `attn_prefill_cm.comp` declares, in bytes: one array of 7,488 uints: KV 16,896 (32 rows of K, then of
/// V, padded to 264 halves; before the key loop, a M tile's q hi and lo) + SP 10,240 (the block's scores, f32 [64][36],
/// then p hi and p lo, f16 [64][40] each) + PM and PS 2,048 (each row's four quarter maxima and sums) + MS, LS and AS
/// 768 (a running max, a running sum and a block's α a row) = 29,952; after the key loop KV and SP together hold the
/// [64][68] f32 output stage (17,408). Over Vulkan's 16,384, so it is in `COOP_SHARED_KERNELS`; change it with the
/// shader's array.
pub const ATTN_CM_SHARED_BYTES: u32 = 29_952;

/// Every cooperative-matrix kernel that declares more than Vulkan's guaranteed 16,384 bytes of shared memory, with what
/// it declares. They are built only where the device allows them all (`coop_fits`); otherwise prefill runs the f32
/// kernels, and nothing is refused.
pub const COOP_SHARED_KERNELS: [(&str, u32); 4] = [
    ("matmul_cm", MATMUL_CM_SHARED_BYTES),
    ("experts_matmul_cm_gate_up", EXPERTS_CM_GATE_UP_SHARED_BYTES),
    ("experts_matmul_cm_down", EXPERTS_CM_DOWN_SHARED_BYTES),
    ("attn_prefill_cm", ATTN_CM_SHARED_BYTES),
];

/// Whether a device whose workgroups may declare `limit` bytes of shared memory runs every cooperative-matrix kernel.
pub fn coop_fits(limit: u32) -> bool {
    COOP_SHARED_KERNELS.iter().all(|&(_, need)| need <= limit)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AttnPrefillPush {
    pub q: u64,
    pub kc: u64,
    pub vc: u64,
    pub y: u64,
    pub pos: u32,
    pub tokens: u32,
    pub hd: u32,
    pub group: u32,
    pub kv_width: u32,
    pub scale: f32,
}
push_constants!(AttnPrefillPush: u64, u64, u64, u64, u32, u32, u32, u32, u32, f32);

/// `attn_prefill_cm`'s push block: `AttnPrefillPush`'s fields, then the test hook and padding to 8 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AttnPrefillCmPush {
    pub q: u64,
    pub kc: u64,
    pub vc: u64,
    pub y: u64,
    pub pos: u32,
    pub tokens: u32,
    pub hd: u32,
    pub group: u32,
    pub kv_width: u32,
    pub scale: f32,
    /// 0 uses every subgroup; otherwise the kernel runs as if the workgroup had `min(sg_limit, gl_NumSubgroups)`
    /// subgroups, the rest idle at every barrier: tests only, so this GPU runs the one- and two-subgroup passes.
    pub sg_limit: u32,
    pub pad: u32,
}
push_constants!(AttnPrefillCmPush: u64, u64, u64, u64, u32, u32, u32, u32, u32, f32, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SigmoidMulPush {
    pub a: u64,
    pub g: u64,
    pub y: u64,
    pub n: u32,
    pub pad: u32,
}
push_constants!(SigmoidMulPush: u64, u64, u64, u32, u32);

/// Whether `attn_prefill_cm` takes heads of `hd` values: it walks the head dimension 32 values at a time in the scores
/// and 64 at a time in the output, so a multiple of 64, from 64 to 256. That admits 128 and 192, which no model in
/// scope uses and no GPU test runs (64 and 256 are tested).
pub fn attn_prefill_cm_fits(hd: usize) -> bool {
    hd.is_multiple_of(64) && (64..=256).contains(&hd)
}

impl Kernels {
    /// `q` and `gate` ← the interleaved projection `qfull` split per head, q normed with `q_norm` and rotated; the
    /// `n_kv` key heads at `k` normed with `k_norm` and rotated in place. `rope` holds `rope_table`'s n_rot values.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope(
        &self,
        r: &mut Recorder,
        qfull: u64,
        k: u64,
        q: u64,
        gate: u64,
        q_norm: u64,
        k_norm: u64,
        rope: u64,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        n_rot: usize,
        eps: f32,
    ) {
        self.qk_norm_rope_tokens(
            r, qfull, k, q, gate, q_norm, k_norm, rope, n_head, n_kv, hd, n_rot, eps, 1,
        );
    }

    /// `qk_norm_rope` for each of `tokens` tokens: `qfull`, `k`, `q`, `gate` and `rope` hold one row per token, each
    /// token's RoPE table its own position's.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_tokens(
        &self,
        r: &mut Recorder,
        qfull: u64,
        k: u64,
        q: u64,
        gate: u64,
        q_norm: u64,
        k_norm: u64,
        rope: u64,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        n_rot: usize,
        eps: f32,
        tokens: usize,
    ) {
        assert!(
            hd <= 256 && n_rot / 2 <= 64,
            "qk_norm_rope holds heads of at most 256"
        );
        let push = QkNormRopePush {
            qfull,
            k,
            q,
            gate,
            q_norm,
            k_norm,
            rope,
            hd: hd as u32,
            n_head: n_head as u32,
            n_rot: n_rot as u32,
            eps,
        };
        r.dispatch(
            self.pipe(Kernel::QkNormRope),
            &push,
            [count(n_head + n_kv), count(tokens), 1],
        );
    }

    /// Position `pos` of the f16 caches `kc` and `vc` (`[ctx][width]`) ← `k` and `v`.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_write(
        &self,
        r: &mut Recorder,
        k: u64,
        v: u64,
        kc: u64,
        vc: u64,
        pos: usize,
        width: usize,
    ) {
        self.kv_write_tokens(r, k, v, kc, vc, pos, width, 1);
    }

    /// Positions `pos .. pos + tokens` of the f16 caches ← the `[tokens][width]` keys and values at `k` and `v`.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_write_tokens(
        &self,
        r: &mut Recorder,
        k: u64,
        v: u64,
        kc: u64,
        vc: u64,
        pos: usize,
        width: usize,
        tokens: usize,
    ) {
        let push = KvWritePush {
            k,
            v,
            kc,
            vc,
            pos: pos as u32,
            width: width as u32,
        };
        r.dispatch(
            self.pipe(Kernel::KvWrite),
            &push,
            [groups(width, 256), count(tokens), 1],
        );
    }

    /// `part` ← each chunk's partial attention for every query head over `n_ctx` cached positions, on the kernel
    /// `attn_decode_kernel` chooses.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode(
        &self,
        r: &mut Recorder,
        q: u64,
        kc: u64,
        vc: u64,
        part: u64,
        n_ctx: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
    ) {
        let kernel = attn_decode_kernel(n_ctx, n_head, n_kv, hd);
        self.attn_decode_on(r, kernel, q, kc, vc, part, n_ctx, n_head, n_kv, hd);
    }

    /// `attn_decode` on `kernel`, `Kernel::AttnDecode` or `Kernel::AttnDecodeGqa`, whatever the context: for the test
    /// that pins their bits together.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_on(
        &self,
        r: &mut Recorder,
        kernel: Kernel,
        q: u64,
        kc: u64,
        vc: u64,
        part: u64,
        n_ctx: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
    ) {
        assert!(
            hd <= 256 && n_ctx >= 1,
            "attn_decode: heads of at most 256, at least one position"
        );
        let workgroups = match kernel {
            Kernel::AttnDecode => n_head,
            Kernel::AttnDecodeGqa => {
                assert!(
                    gqa_takes(n_head, n_kv, hd) && kc.is_multiple_of(16),
                    "attn_decode_gqa: at most {ATTN_GQA_GROUP} query heads a KV head, heads a multiple of 32 values, a 16-byte-aligned key cache"
                );
                n_kv
            }
            other => panic!(
                "attn_decode_on: {} isn't a decode attention kernel",
                other.name()
            ),
        };
        let n_split = attn_splits(n_ctx);
        let push = AttnDecodePush {
            q,
            kc,
            vc,
            part,
            n_ctx: n_ctx as u32,
            hd: hd as u32,
            group: (n_head / n_kv) as u32,
            kv_width: (n_kv * hd) as u32,
            n_split: n_split as u32,
            scale: 1.0 / (hd as f32).sqrt(),
        };
        r.dispatch(
            self.pipe(kernel),
            &push,
            [count(n_split), count(workgroups), 1],
        );
    }

    /// `y` ← every query head's attention output from `attn_decode`'s partials.
    pub fn attn_combine(
        &self,
        r: &mut Recorder,
        part: u64,
        y: u64,
        n_ctx: usize,
        n_head: usize,
        hd: usize,
    ) {
        let push = AttnCombinePush {
            part,
            y,
            hd: hd as u32,
            n_split: attn_splits(n_ctx) as u32,
        };
        r.dispatch(
            self.pipe(Kernel::AttnCombine),
            &push,
            [count(n_head), groups(hd, ATTN_COMBINE_DIMS), 1],
        );
    }

    /// `y` (`[tokens][n_head][hd]`) ← causal attention for the `tokens` queries at `q`, query i at position `pos` + i,
    /// over the f16 caches' positions 0 ..= pos + i. The batch's keys and values must already be in the caches.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill(
        &self,
        r: &mut Recorder,
        q: u64,
        kc: u64,
        vc: u64,
        y: u64,
        pos: usize,
        tokens: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
    ) {
        assert!(
            hd <= 256 && tokens >= 1,
            "attn_prefill: heads of at most 256, at least one query"
        );
        assert!(
            n_kv > 0 && n_head.is_multiple_of(n_kv),
            "attn_prefill: {n_head} query heads don't divide over {n_kv} KV heads"
        );
        assert!(
            kc.is_multiple_of(16) && vc.is_multiple_of(16),
            "attn_prefill: the caches' addresses ({kc:#x}, {vc:#x}) must be 16-byte aligned for the kernel's 16-byte loads"
        );
        assert!(
            q.is_multiple_of(16) && y.is_multiple_of(16),
            "attn_prefill: q and y ({q:#x}, {y:#x}) must be 16-byte aligned for the kernel's 16-byte loads and stores"
        );
        assert!(
            hd.is_multiple_of(8),
            "attn_prefill: the kernel reads the f16 caches 16 bytes (eight values) a load, so hd must be a multiple of 8"
        );
        let push = AttnPrefillPush {
            q,
            kc,
            vc,
            y,
            pos: pos as u32,
            tokens: tokens as u32,
            hd: hd as u32,
            group: (n_head / n_kv) as u32,
            kv_width: (n_kv * hd) as u32,
            scale: 1.0 / (hd as f32).sqrt(),
        };
        r.dispatch(
            self.pipe(Kernel::AttnPrefill),
            &push,
            [groups(tokens, ATTN_QUERIES), count(n_head), 1],
        );
    }

    /// `y` (`[tokens][n_head][hd]`) ← causal attention for the `tokens` queries at `q`, query i at position `pos` + i,
    /// over the f16 caches' positions 0 ..= pos + i, as `attn_prefill` computes it, on 16 × 16 × 16 f16 cooperative
    /// matrices with q and the softmax weights p each split into a hi and a lo f16 (K and V are exact f16 in the
    /// cache): a workgroup takes 64 queries of one head. The batch's keys and values must already be in the caches.
    /// Needs `coop()` (`Kernels::pipe` panics otherwise) and a head `attn_prefill_cm_fits`. Runs on any subgroup count
    /// (in passes, below four). `sg_limit` is 0 outside tests: otherwise the kernel runs as if the workgroup had
    /// `min(sg_limit, gl_NumSubgroups)` subgroups (a power of two), so this GPU runs the one- and two-subgroup passes.
    /// Within `tol::REDUCTION` of `twin::attention::attn_prefill_cm`, which has no subgroups: at most 7.44e-7 relative L2
    /// over the kernel tests' cases (hd 64 and 256, 7 to 130 queries from 0 to 3,000; measured 2026-10-18).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_cm(
        &self,
        r: &mut Recorder,
        q: u64,
        kc: u64,
        vc: u64,
        y: u64,
        pos: usize,
        tokens: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        sg_limit: u32,
    ) {
        assert!(
            attn_prefill_cm_fits(hd) && tokens >= 1,
            "attn_prefill_cm: heads of 64, 128, 192 or 256 values, at least one query"
        );
        assert!(
            n_kv > 0 && n_head.is_multiple_of(n_kv),
            "attn_prefill_cm: {n_head} query heads don't divide over {n_kv} KV heads"
        );
        assert!(
            kc.is_multiple_of(16) && vc.is_multiple_of(16),
            "attn_prefill_cm: the caches' addresses ({kc:#x}, {vc:#x}) must be 16-byte aligned for the kernel's 16-byte loads"
        );
        assert!(
            q.is_multiple_of(16) && y.is_multiple_of(16),
            "attn_prefill_cm: q and y ({q:#x}, {y:#x}) must be 16-byte aligned for the kernel's 16-byte loads and stores"
        );
        assert!(
            sg_limit == 0 || sg_limit.is_power_of_two(),
            "attn_prefill_cm: sg_limit {sg_limit} must be 0 or a power of two"
        );
        let push = AttnPrefillCmPush {
            q,
            kc,
            vc,
            y,
            pos: pos as u32,
            tokens: tokens as u32,
            hd: hd as u32,
            group: (n_head / n_kv) as u32,
            kv_width: (n_kv * hd) as u32,
            scale: 1.0 / (hd as f32).sqrt(),
            sg_limit,
            pad: 0,
        };
        r.dispatch(
            self.pipe(Kernel::AttnPrefillCm),
            &push,
            [groups(tokens, ATTN_CM_QUERIES), count(n_head), 1],
        );
    }

    /// `y` ← a ⊙ sigmoid(g) over `n` values.
    pub fn sigmoid_mul(&self, r: &mut Recorder, a: u64, g: u64, y: u64, n: usize) {
        let push = SigmoidMulPush {
            a,
            g,
            y,
            n: n as u32,
            pad: 0,
        };
        r.dispatch(self.pipe(Kernel::SigmoidMul), &push, [groups(n, 256), 1, 1]);
    }
}
/// Where a slot-table entry's expert lives.
pub const WHERE_VRAM: u32 = 0;
pub const WHERE_HOST: u32 = 1;
/// Computed by CPU workers through the miss path (see `mbox`); a kernel's `serve` mode decides whether such an entry is
/// refused.
pub const WHERE_CPU: u32 = 2;
/// Bytes per slot-table entry: { u64 addr; u32 where; u32 reserved }.
pub const ENTRY_BYTES: usize = 16;
/// The expert kernels' status buffer: [cpu, bad, -, -], each `layer << 16 | expert` of the lowest entry of that kind a
/// step reached, or all ones for none. `cpu` holds `CPU` entries; `bad` entries with an unknown `where` or no address.
/// Clear it to `STATUS_CLEAR` before a step and read it with `read_status`.
pub const STATUS_BYTES: usize = 16;
pub const STATUS_CLEAR: [u8; STATUS_BYTES] = [0xFF; STATUS_BYTES];

/// The entries a step refused, as (layer, expert): the lowest of each kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertStatus {
    pub cpu: Option<(u32, u32)>,
    pub bad: Option<(u32, u32)>,
}

pub fn read_status(b: &[u8; STATUS_BYTES]) -> ExpertStatus {
    let word = |i: usize| {
        let w = u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().expect("4 bytes"));
        (w != u32::MAX).then_some((w >> 16, w & 0xFFFF))
    };
    ExpertStatus {
        cpu: word(0),
        bad: word(1),
    }
}

/// The CPU miss path's mailbox, in `MemKind::AmdUncached` memory: the byte offsets `mbox.glsl` mirrors.
/// The engine's coordinator and the tests are the other readers of this module.
pub mod mbox {
    /// The flag area's size: every layer's publish and done flag, each on its own 128-byte line.
    pub const X0: usize = 16384;
    /// A layer's X area: a 256-byte header, then x (at most `N_EMBD` f32).
    pub const XSTRIDE: usize = 8448;
    /// The Y area's start (64 layers' X areas fit before it, on this layout's strides).
    pub const Y0: usize = 557056;
    /// A layer's Y area: the CPU's outputs, `[MAX_E][N_EMBD]` f32, one expert per routed index j.
    pub const YSTRIDE: usize = 65536;
    /// The most x values and routed experts a layer's areas hold.
    pub const N_EMBD: usize = 2048;
    pub const MAX_E: usize = 8;
    /// The wait's status words: [timeout count, skip count], at byte 12288 (layer 48's publish-flag line), hence
    /// `n_layer + 1 <= 48`.
    pub const STATUS_BYTE: usize = 12288;
    /// The mailbox's extent; 8 MiB is allocated, which leaves spare room.
    pub const BYTES: usize = Y0 + 64 * YSTRIDE;
    /// The most layers the flags serve: layer 48's publish-flag line holds the status words.
    pub const MAX_LAYERS: usize = 48;

    // The layout, pinned (`mbox.glsl` mirrors these numbers).
    const _: () = {
        assert!(BYTES == 4_751_360 && BYTES <= 8 << 20);
        assert!(STATUS_BYTE == 4 * pub_word(MAX_LAYERS));
        assert!(X0 == 64 * 256 && X0 > 4 * done_word(63));
        assert!(XSTRIDE == 256 + 4 * N_EMBD);
        assert!(Y0 == X0 + 64 * XSTRIDE);
        assert!(YSTRIDE == 4 * MAX_E * N_EMBD);
    };

    /// Layer `layer`'s publish flag, as a u32 word index.
    pub const fn pub_word(layer: usize) -> usize {
        64 * layer
    }
    /// Layer `layer`'s done flag, as a u32 word index.
    pub const fn done_word(layer: usize) -> usize {
        64 * layer + 32
    }
}

/// Refuses a VRAM or HOST entry whose slot isn't 16-byte aligned: the expert kernels read slots 16 bytes at a time
/// (`quant.glsl`'s `unit_dot`). Every slot the engine makes is (cache rows, staging buffers and pinned copies are
/// 256-byte-aligned arenas of whole slots, and every slot size is a multiple of 16); the engine checks each entry
/// decode's expert kernels read (`set_slot_entry`, `stage_slot_entry`, `weights::slot_table`). Prefill's staging
/// table, which `stage::plan` writes, isn't checked: only prefill's `experts_matmul*` kernels read it, and they don't
/// use `unit_dot`. `slot_entry` and `slot_update` only encode, so tests can write any entry.
pub fn check_slot_addr(addr: u64, loc: u32) {
    assert!(
        loc > WHERE_HOST || addr.is_multiple_of(16),
        "a slot at {addr:#x}: the expert kernels read 16-byte-aligned slots"
    );
}

pub fn slot_entry(addr: u64, loc: u32) -> [u8; ENTRY_BYTES] {
    let mut e = [0u8; ENTRY_BYTES];
    e[..8].copy_from_slice(&addr.to_le_bytes());
    e[8..12].copy_from_slice(&loc.to_le_bytes());
    e
}

/// Bytes per update-list entry: { u32 index; u32 where; u64 addr }.
pub const UPDATE_BYTES: usize = 16;

/// One update-list entry: slot-table entry `index` becomes { addr, where }.
pub fn slot_update(index: u32, loc: u32, addr: u64) -> [u8; UPDATE_BYTES] {
    let mut u = [0u8; UPDATE_BYTES];
    u[..4].copy_from_slice(&index.to_le_bytes());
    u[4..8].copy_from_slice(&loc.to_le_bytes());
    u[8..].copy_from_slice(&addr.to_le_bytes());
    u
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ApplySlotUpdatesPush {
    pub table: u64,
    pub updates: u64,
    pub count: u32,
    pub n_entries: u32,
}
push_constants!(ApplySlotUpdatesPush: u64, u64, u32, u32);

impl Kernels {
    /// `table[index]` ← { addr, where } for each of the `count` updates at `updates`. Indices must be
    /// distinct; one at or past `n_entries` is skipped. The engine skips the dispatch when there are no updates.
    pub fn apply_slot_updates(
        &self,
        r: &mut Recorder,
        table: u64,
        updates: u64,
        count: usize,
        n_entries: usize,
    ) {
        let push = ApplySlotUpdatesPush {
            table,
            updates,
            count: count as u32,
            n_entries: n_entries as u32,
        };
        r.dispatch(
            self.pipe(Kernel::ApplySlotUpdates),
            &push,
            [groups(count, 64), 1, 1],
        );
    }
}

/// One expert's packed slot: its gate rows, then its up rows, then its down rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotLayout {
    pub gate: WType,
    pub up: WType,
    pub down: WType,
    pub n_embd: usize,
    pub n_ff: usize,
}

impl SlotLayout {
    pub fn gate_bytes(&self) -> usize {
        self.n_ff * self.gate.row_bytes(self.n_embd)
    }

    pub fn up_bytes(&self) -> usize {
        self.n_ff * self.up.row_bytes(self.n_embd)
    }

    pub fn down_bytes(&self) -> usize {
        self.n_embd * self.down.row_bytes(self.n_ff)
    }

    pub fn up_offset(&self) -> usize {
        self.gate_bytes()
    }

    pub fn down_offset(&self) -> usize {
        self.gate_bytes() + self.up_bytes()
    }

    pub fn bytes(&self) -> usize {
        self.gate_bytes() + self.up_bytes() + self.down_bytes()
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RouterTopkPush {
    pub logits: u64,
    pub probs: u64,
    pub ids: u64,
    pub w: u64,
    pub n: u32,
    pub k: u32,
}
push_constants!(RouterTopkPush: u64, u64, u64, u64, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SwigluPush {
    pub g: u64,
    pub u: u64,
    pub y: u64,
    pub n: u32,
    pub pad: u32,
}
push_constants!(SwigluPush: u64, u64, u64, u32, u32);

/// Subgroups of 32 lanes in a workgroup of the routed experts' decode kernels (their shaders' EXPERT_WAVES).
pub const EXPERT_WAVES: usize = 8;

/// Rows a workgroup of the routed experts' decode kernels (`experts_gate_up`, `experts_down`) covers, for rows of `k`
/// inputs: EXPERT_WAVES subgroups of 32 lanes, each lane on one row; a row of at most 16 units (k <= 512) takes 16
/// lanes, so a subgroup carries two rows, and a longer row takes 32. The shaders make the same choice.
pub fn expert_rows(k: usize) -> usize {
    let per_subgroup = if k / 32 <= 16 { 2 } else { 1 };
    EXPERT_WAVES * per_subgroup
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertsGateUpPush {
    pub table: u64,
    pub ids: u64,
    pub x: u64,
    pub y: u64,
    pub status: u64,
    pub mbox: u64,
    pub layer_base: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub gate_type: u32,
    pub up_type: u32,
    pub up_off: u32,
    pub layer: u32,
    pub seq: u32,
    pub serve: u32,
    /// +0.0, the tree's added zero on rows of at most 16 units (`tree32.glsl`): from the host, so the driver can't
    /// fold the add away.
    pub zero: f32,
}
push_constants!(ExpertsGateUpPush: u64, u64, u64, u64, u64, u64, u32, u32, u32, u32, u32, u32, u32, u32, u32, f32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertsDownPush {
    pub table: u64,
    pub ids: u64,
    pub act: u64,
    pub y: u64,
    pub status: u64,
    pub mbox: u64,
    pub layer_base: u32,
    pub n_ff: u32,
    pub n_embd: u32,
    pub down_type: u32,
    pub down_off: u32,
    pub layer: u32,
    pub seq: u32,
    pub serve: u32,
    /// `ExpertsGateUpPush::zero`'s.
    pub zero: f32,
    pub pad: u32,
}
push_constants!(ExpertsDownPush: u64, u64, u64, u64, u64, u64, u32, u32, u32, u32, u32, u32, u32, u32, f32, u32);

/// `stamp`: tim[idx] = clockRealtimeEXT(). The calibration kernel: two stamps 300 ms apart give the
/// device clock's ticks per µs, so the wait's bound is passed in ticks, never hard-coded.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct StampPush {
    pub tim: u64,
    pub idx: u32,
    pub pad: u32,
}
push_constants!(StampPush: u64, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MissPublishPush {
    pub xv: u64,
    pub idsv: u64,
    pub table: u64,
    pub mbox: u64,
    pub layer_base: u32,
    pub layer: u32,
    pub seq: u32,
    pub token: u32,
    pub n_embd: u32,
    pub k: u32,
}
push_constants!(MissPublishPush: u64, u64, u64, u64, u32, u32, u32, u32, u32, u32);

// The miss path's pushes, and the two expert pushes it grew, carry no implicit padding: `Recorder` hands the
// struct's bytes to Vulkan, and padding bytes are uninitialised.
const _: () = {
    use std::mem::size_of;
    assert!(size_of::<ExpertsGateUpPush>() == 6 * 8 + 10 * 4);
    assert!(size_of::<ExpertsDownPush>() == 6 * 8 + 10 * 4);
    assert!(size_of::<StampPush>() == 8 + 2 * 4);
    assert!(size_of::<MissPublishPush>() == 4 * 8 + 6 * 4);
    assert!(size_of::<MissWaitPush>() == 3 * 8 + 6 * 4);
};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MissWaitPush {
    pub mbox: u64,
    pub ye: u64,
    pub status: u64,
    pub layer: u32,
    pub seq: u32,
    pub n_embd: u32,
    pub max_spins: u32,
    pub max_ticks: u32,
    pub pad: u32,
}
push_constants!(MissWaitPush: u64, u64, u64, u32, u32, u32, u32, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MoeCombinePush {
    pub ye: u64,
    pub w: u64,
    pub sh: u64,
    pub sg: u64,
    pub moe: u64,
    pub shexp: u64,
    pub n: u32,
    pub k: u32,
}
push_constants!(MoeCombinePush: u64, u64, u64, u64, u64, u64, u32, u32);

impl Kernels {
    /// `probs` ← softmax(logits) over `n` experts; `ids` and `w` ← the `k` chosen and their normalized weights.
    #[allow(clippy::too_many_arguments)]
    pub fn router_topk(
        &self,
        r: &mut Recorder,
        logits: u64,
        probs: u64,
        ids: u64,
        w: u64,
        n: usize,
        k: usize,
    ) {
        self.router_topk_tokens(r, logits, probs, ids, w, n, k, 1);
    }

    /// `router_topk` for each of `tokens` tokens, one workgroup each: `logits` and `probs` hold `[tokens][n]`, `ids`
    /// and `w` `[tokens][k]`.
    #[allow(clippy::too_many_arguments)]
    pub fn router_topk_tokens(
        &self,
        r: &mut Recorder,
        logits: u64,
        probs: u64,
        ids: u64,
        w: u64,
        n: usize,
        k: usize,
        tokens: usize,
    ) {
        assert!(
            n <= 256 && k <= 16 && k <= n,
            "router_topk: {k} of {n} experts"
        );
        let push = RouterTopkPush {
            logits,
            probs,
            ids,
            w,
            n: n as u32,
            k: k as u32,
        };
        r.dispatch(self.pipe(Kernel::RouterTopk), &push, [count(tokens), 1, 1]);
    }

    /// `y` ← silu(g) ⊙ u over `n` values.
    pub fn swiglu(&self, r: &mut Recorder, g: u64, u: u64, y: u64, n: usize) {
        let push = SwigluPush {
            g,
            u,
            y,
            n: n as u32,
            pad: 0,
        };
        r.dispatch(self.pipe(Kernel::Swiglu), &push, [groups(n, 256), 1, 1]);
    }

    /// `y` (`[k][n_ff]`) ← each routed expert's silu_det(gate·x) ⊙ (up·x), reading layer `layer`'s entries of `table`.
    ///
    /// `serve`: 0 refuses a CPU entry (NaN and the status, as before the CPU path); 1 skips it (writes
    /// +0, no refuse); 2 is the fallback pair, which computes a CPU entry from its host address only when this
    /// token's wait did not succeed (`status[0] != 0` at `mbox`, this token's timeout count: this layer's wait
    /// or an earlier one timed out; or the layer's done flag at `mbox` is not `seq`) and writes nothing otherwise (a
    /// late answer is wasted, never trusted). `mbox` and `seq` are read only when `serve == 2`; pass 0 for both
    /// otherwise.
    #[allow(clippy::too_many_arguments)]
    pub fn experts_gate_up(
        &self,
        r: &mut Recorder,
        table: u64,
        layer: usize,
        n_expert: usize,
        ids: u64,
        x: u64,
        y: u64,
        status: u64,
        lay: &SlotLayout,
        k: usize,
        serve: u32,
        mbox: u64,
        seq: u32,
    ) {
        assert!(
            x.is_multiple_of(16) && lay.up_offset().is_multiple_of(16),
            "experts_gate_up: x and the up matrix must be 16-byte aligned"
        );
        assert!(
            serve <= 2 && (serve != 2 || mbox != 0),
            "serve {serve}: 0, 1, or 2 with the mailbox"
        );
        let push = ExpertsGateUpPush {
            table,
            ids,
            x,
            y,
            status,
            mbox,
            layer_base: (layer * n_expert) as u32,
            n_embd: lay.n_embd as u32,
            n_ff: lay.n_ff as u32,
            gate_type: crate::wtype_code(lay.gate),
            up_type: crate::wtype_code(lay.up),
            up_off: lay.up_offset() as u32,
            layer: layer as u32,
            seq,
            serve,
            zero: 0.0,
        };
        r.dispatch(
            self.pipe(Kernel::ExpertsGateUp),
            &push,
            [groups(lay.n_ff, expert_rows(lay.n_embd)), count(k), 1],
        );
    }

    /// `y` (`[k][n_embd]`) ← each routed expert's down · act[j]. `serve`, `mbox` and `seq` are
    /// `experts_gate_up`'s.
    #[allow(clippy::too_many_arguments)]
    pub fn experts_down(
        &self,
        r: &mut Recorder,
        table: u64,
        layer: usize,
        n_expert: usize,
        ids: u64,
        act: u64,
        y: u64,
        status: u64,
        lay: &SlotLayout,
        k: usize,
        serve: u32,
        mbox: u64,
        seq: u32,
    ) {
        assert!(
            act.is_multiple_of(16) && lay.down_offset().is_multiple_of(16) && (4 * lay.n_ff).is_multiple_of(16),
            "experts_down: the activations, each expert's and the down matrix must be 16-byte aligned"
        );
        assert!(
            serve <= 2 && (serve != 2 || mbox != 0),
            "serve {serve}: 0, 1, or 2 with the mailbox"
        );
        let push = ExpertsDownPush {
            table,
            ids,
            act,
            y,
            status,
            mbox,
            layer_base: (layer * n_expert) as u32,
            n_ff: lay.n_ff as u32,
            n_embd: lay.n_embd as u32,
            down_type: crate::wtype_code(lay.down),
            down_off: lay.down_offset() as u32,
            layer: layer as u32,
            seq,
            serve,
            zero: 0.0,
            pad: 0,
        };
        r.dispatch(
            self.pipe(Kernel::ExpertsDown),
            &push,
            [groups(lay.n_embd, expert_rows(lay.n_ff)), count(k), 1],
        );
    }

    /// `tim[idx]` ← `clockRealtimeEXT()` (the calibration kernel: two stamps 300 ms apart give the device
    /// clock's ticks per µs, so the wait's bound is passed in ticks, never hard-coded).
    pub fn stamp(&self, r: &mut Recorder, tim: u64, idx: usize) {
        let push = StampPush {
            tim,
            idx: idx as u32,
            pad: 0,
        };
        r.dispatch(self.pipe(Kernel::Stamp), &push, [1, 1, 1]);
    }

    /// The CPU miss path's publish: the layer's `x`, its routed ids (the raw f32 bits the coordinator
    /// decodes) and the mask (bit `j` set when routed expert `j`'s entry in `table` is `WHERE_CPU`) into the
    /// mailbox, then the layer's publish flag, which the coordinator polls. `layer_base` is `layer * n_expert`.
    /// `n_embd <= 2048` and `k <= 8`, the mailbox's layout.
    #[allow(clippy::too_many_arguments)]
    pub fn miss_publish(
        &self,
        r: &mut Recorder,
        xv: u64,
        idsv: u64,
        table: u64,
        mbox: u64,
        layer: usize,
        n_expert: usize,
        seq: u32,
        token: u32,
        n_embd: usize,
        k: usize,
    ) {
        assert!(n_embd <= mbox::N_EMBD && n_embd.is_multiple_of(4) && k <= mbox::MAX_E);
        assert!(
            layer < mbox::MAX_LAYERS,
            "layer {layer}: the status words sit on layer 48's flag line"
        );
        assert!(
            xv.is_multiple_of(16),
            "the publish copies x 16 bytes at a time"
        );
        let push = MissPublishPush {
            xv,
            idsv,
            table,
            mbox,
            layer_base: (layer * n_expert) as u32,
            layer: layer as u32,
            seq,
            token,
            n_embd: n_embd as u32,
            k: k as u32,
        };
        r.dispatch(self.pipe(Kernel::MissPublish), &push, [1, 1, 1]);
    }

    /// The CPU miss path's wait: a bounded spin on the layer's done flag, then the CPU's `y[j]` into
    /// `ye[j]` for each set mask bit, and nothing on a timeout or a skip (the serve-2 fallback pair computes
    /// those experts from their host addresses). `status` points at the mailbox's two status words
    /// (`mbox::STATUS_BYTE`): the timeout and skip counts, per token.
    #[allow(clippy::too_many_arguments)]
    pub fn miss_wait(
        &self,
        r: &mut Recorder,
        mbox: u64,
        ye: u64,
        status: u64,
        layer: usize,
        seq: u32,
        n_embd: usize,
        max_spins: u32,
        max_ticks: u32,
    ) {
        assert!(n_embd <= mbox::N_EMBD && n_embd.is_multiple_of(4));
        assert!(
            layer < mbox::MAX_LAYERS,
            "layer {layer}: the status words sit on layer 48's flag line"
        );
        assert!(
            ye.is_multiple_of(16),
            "the wait copies y 16 bytes at a time"
        );
        assert_eq!(
            status,
            mbox + mbox::STATUS_BYTE as u64,
            "serve 2 reads the status words at the mailbox's STATUS_BYTE"
        );
        let push = MissWaitPush {
            mbox,
            ye,
            status,
            layer: layer as u32,
            seq,
            n_embd: n_embd as u32,
            max_spins,
            max_ticks,
            pad: 0,
        };
        r.dispatch(self.pipe(Kernel::MissWait), &push, [1, 1, 1]);
    }

    /// `moe` ← Σ_j w_j · ye[j] and `shexp` ← sigmoid(sg[0]) · sh, over `n` values and `k` experts.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_combine(
        &self,
        r: &mut Recorder,
        ye: u64,
        w: u64,
        sh: u64,
        sg: u64,
        moe: u64,
        shexp: u64,
        n: usize,
        k: usize,
    ) {
        self.moe_combine_tokens(r, ye, w, sh, sg, moe, shexp, n, k, 1);
    }

    /// `moe_combine` for each of `tokens` tokens: `ye` holds `[tokens][k][n]`, `w` `[tokens][k]`, `sg` `[tokens]`,
    /// and `sh`, `moe` and `shexp` `[tokens][n]`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_combine_tokens(
        &self,
        r: &mut Recorder,
        ye: u64,
        w: u64,
        sh: u64,
        sg: u64,
        moe: u64,
        shexp: u64,
        n: usize,
        k: usize,
        tokens: usize,
    ) {
        let push = MoeCombinePush {
            ye,
            w,
            sh,
            sg,
            moe,
            shexp,
            n: n as u32,
            k: k as u32,
        };
        r.dispatch(
            self.pipe(Kernel::MoeCombine),
            &push,
            [groups(n, 256), count(tokens), 1],
        );
    }
}

/// Entries per `moe_group` tile: one expert's tokens, as many as a grouped expert workgroup takes.
pub const MOE_TILE: usize = 64;

/// An upper bound on the tiles `moe_group` makes: at least as many as any list of `entries` over `n_expert` experts
/// can need, since every expert's run rounds up to whole tiles, so at most one part-full tile per expert that has any.
pub fn moe_tiles(entries: usize, n_expert: usize) -> usize {
    entries.div_ceil(MOE_TILE) + n_expert.min(entries)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MoeGroupPush {
    pub ids: u64,
    pub list: u64,
    pub tiles: u64,
    pub n: u32,
    pub entries: u32,
    pub max_tiles: u32,
    pub pad: u32,
}
push_constants!(MoeGroupPush: u64, u64, u64, u32, u32, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertsMatmulGateUpPush {
    pub table: u64,
    pub list: u64,
    pub tiles: u64,
    pub x: u64,
    pub act: u64,
    pub status: u64,
    pub layer_base: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub gate_type: u32,
    pub up_type: u32,
    pub up_off: u32,
    pub layer: u32,
    pub k: u32,
}
push_constants!(ExpertsMatmulGateUpPush: u64, u64, u64, u64, u64, u64, u32, u32, u32, u32, u32, u32, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertsMatmulDownPush {
    pub table: u64,
    pub list: u64,
    pub tiles: u64,
    pub act: u64,
    pub ye: u64,
    pub status: u64,
    pub layer_base: u32,
    pub n_ff: u32,
    pub n_embd: u32,
    pub down_type: u32,
    pub down_off: u32,
    pub layer: u32,
}
push_constants!(ExpertsMatmulDownPush: u64, u64, u64, u64, u64, u64, u32, u32, u32, u32, u32, u32);

impl Kernels {
    /// `list` (u32 per entry) and `tiles` (`max_tiles` × uvec4) ← the `entries` routed ids at `ids` (f32) grouped by
    /// expert: each expert's entries t·k + j in token order, experts in order, and each run cut into tiles of up to
    /// `MOE_TILE`, as (expert, first, count, 0), unused tiles empty. One workgroup; at most 256 experts.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_group(
        &self,
        r: &mut Recorder,
        ids: u64,
        list: u64,
        tiles: u64,
        n_expert: usize,
        entries: usize,
        max_tiles: usize,
    ) {
        assert!(n_expert <= 256, "moe_group: {n_expert} experts");
        assert!(
            max_tiles >= moe_tiles(entries, n_expert),
            "moe_group: {max_tiles} tiles are too few for {entries} entries over {n_expert} experts (writes would run past the tile buffer)"
        );
        let push = MoeGroupPush {
            ids,
            list,
            tiles,
            n: n_expert as u32,
            entries: entries as u32,
            max_tiles: max_tiles as u32,
            pad: 0,
        };
        r.dispatch(self.pipe(Kernel::MoeGroup), &push, [1, 1, 1]);
    }

    /// `act` (`[entries][n_ff]`, in `list`'s order) ← for each listed entry, silu_det(gate·x) ⊙ (up·x) of its expert in
    /// layer `layer`, read through `table`, and its token's row of `x` (`[tokens][n_embd]`), token = entry / `k`.
    /// One workgroup per tile of `tiles` and 64 rows.
    #[allow(clippy::too_many_arguments)]
    pub fn experts_matmul_gate_up(
        &self,
        r: &mut Recorder,
        table: u64,
        layer: usize,
        n_expert: usize,
        list: u64,
        tiles: u64,
        n_tiles: usize,
        x: u64,
        act: u64,
        status: u64,
        lay: &SlotLayout,
        k: usize,
    ) {
        self.record_experts_gate_up(
            r,
            Kernel::ExpertsMatmulGateUp,
            table,
            layer,
            n_expert,
            list,
            tiles,
            n_tiles,
            x,
            act,
            status,
            lay,
            k,
        );
    }

    /// `ye` (`[tokens][k][n_embd]`) ← for each listed entry, down · act[i] of its expert, written to row `list[i]`.
    #[allow(clippy::too_many_arguments)]
    pub fn experts_matmul_down(
        &self,
        r: &mut Recorder,
        table: u64,
        layer: usize,
        n_expert: usize,
        list: u64,
        tiles: u64,
        n_tiles: usize,
        act: u64,
        ye: u64,
        status: u64,
        lay: &SlotLayout,
    ) {
        self.record_experts_down(
            r,
            Kernel::ExpertsMatmulDown,
            table,
            layer,
            n_expert,
            list,
            tiles,
            n_tiles,
            act,
            ye,
            status,
            lay,
        );
    }

    /// `experts_matmul_gate_up` on cooperative matrices, every weight and input split into two f16s: the same
    /// parameters, grid and outputs. As in `matmul_cm`, the split bounds each input (hi + lo within about 2⁻²² of its
    /// f32 value), not the outputs, which measured within about 1e-6 relative L2 of the twin's (worst 8.5e-7 at k =
    /// 4,096, test 1), inside the kernel's 1e-5 tolerance. Each 32-value slice of k is summed from zero on the
    /// cooperative matrices and then added to the output with an ordinary f32 add, as `matmul_cm`'s are. Panics without
    /// `Kernels::coop`.
    #[allow(clippy::too_many_arguments)]
    pub fn experts_matmul_cm_gate_up(
        &self,
        r: &mut Recorder,
        table: u64,
        layer: usize,
        n_expert: usize,
        list: u64,
        tiles: u64,
        n_tiles: usize,
        x: u64,
        act: u64,
        status: u64,
        lay: &SlotLayout,
        k: usize,
    ) {
        self.record_experts_gate_up(
            r,
            Kernel::ExpertsMatmulCmGateUp,
            table,
            layer,
            n_expert,
            list,
            tiles,
            n_tiles,
            x,
            act,
            status,
            lay,
            k,
        );
    }

    /// `experts_matmul_down` on cooperative matrices, every weight and input split into two f16s: the same parameters,
    /// grid and outputs, each output summed per 32-value slice as `experts_matmul_cm_gate_up`'s. Its outputs measured
    /// within about 1e-6 relative L2 of the twin's (worst 9.5e-7 at k = 4,096, test 1), inside the
    /// kernel's 1e-5 tolerance. Panics without `Kernels::coop`.
    #[allow(clippy::too_many_arguments)]
    pub fn experts_matmul_cm_down(
        &self,
        r: &mut Recorder,
        table: u64,
        layer: usize,
        n_expert: usize,
        list: u64,
        tiles: u64,
        n_tiles: usize,
        act: u64,
        ye: u64,
        status: u64,
        lay: &SlotLayout,
    ) {
        self.record_experts_down(
            r,
            Kernel::ExpertsMatmulCmDown,
            table,
            layer,
            n_expert,
            list,
            tiles,
            n_tiles,
            act,
            ye,
            status,
            lay,
        );
    }

    /// The dispatch `experts_matmul_gate_up` and `experts_matmul_cm_gate_up` share: kernel `kern` (one of the two) over
    /// `n_ff` rows, `MATMUL_TILE` per workgroup, and `n_tiles` tiles. The assert names the kernel.
    #[allow(clippy::too_many_arguments)]
    fn record_experts_gate_up(
        &self,
        r: &mut Recorder,
        kern: Kernel,
        table: u64,
        layer: usize,
        n_expert: usize,
        list: u64,
        tiles: u64,
        n_tiles: usize,
        x: u64,
        act: u64,
        status: u64,
        lay: &SlotLayout,
        k: usize,
    ) {
        debug_assert!(
            matches!(
                kern,
                Kernel::ExpertsMatmulGateUp | Kernel::ExpertsMatmulCmGateUp
            ),
            "record_experts_gate_up: {} takes another push block",
            kern.name()
        );
        assert!(
            lay.n_embd.is_multiple_of(32) && lay.n_ff.is_multiple_of(32) && k > 0,
            "{}: n_embd {} and n_ff {} must be multiples of 32 (the tile loop steps 32 inputs at a time), and k {k} positive (token = entry / k)",
            kern.name(),
            lay.n_embd,
            lay.n_ff
        );
        let push = ExpertsMatmulGateUpPush {
            table,
            list,
            tiles,
            x,
            act,
            status,
            layer_base: (layer * n_expert) as u32,
            n_embd: lay.n_embd as u32,
            n_ff: lay.n_ff as u32,
            gate_type: wtype_code(lay.gate),
            up_type: wtype_code(lay.up),
            up_off: lay.up_offset() as u32,
            layer: layer as u32,
            k: k as u32,
        };
        r.dispatch(
            self.pipe(kern),
            &push,
            [groups(lay.n_ff, MATMUL_TILE), count(n_tiles), 1],
        );
    }

    /// The dispatch `experts_matmul_down` and `experts_matmul_cm_down` share: kernel `kern` (one of the two) over
    /// `n_embd` rows, `MATMUL_TILE` per workgroup, and `n_tiles` tiles. The assert names the kernel.
    #[allow(clippy::too_many_arguments)]
    fn record_experts_down(
        &self,
        r: &mut Recorder,
        kern: Kernel,
        table: u64,
        layer: usize,
        n_expert: usize,
        list: u64,
        tiles: u64,
        n_tiles: usize,
        act: u64,
        ye: u64,
        status: u64,
        lay: &SlotLayout,
    ) {
        debug_assert!(
            matches!(
                kern,
                Kernel::ExpertsMatmulDown | Kernel::ExpertsMatmulCmDown
            ),
            "record_experts_down: {} takes another push block",
            kern.name()
        );
        assert!(
            lay.n_embd.is_multiple_of(32) && lay.n_ff.is_multiple_of(32),
            "{}: n_embd {} and n_ff {} must be multiples of 32 (the tile loop steps 32 inputs at a time)",
            kern.name(),
            lay.n_embd,
            lay.n_ff
        );
        let push = ExpertsMatmulDownPush {
            table,
            list,
            tiles,
            act,
            ye,
            status,
            layer_base: (layer * n_expert) as u32,
            n_ff: lay.n_ff as u32,
            n_embd: lay.n_embd as u32,
            down_type: wtype_code(lay.down),
            down_off: lay.down_offset() as u32,
            layer: layer as u32,
        };
        r.dispatch(
            self.pipe(kern),
            &push,
            [groups(lay.n_embd, MATMUL_TILE), count(n_tiles), 1],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{
        attn_decode_kernel, attn_prefill_cm_fits, groups, ATTN_CM_KEYS, ATTN_CM_QUERIES,
        ATTN_GQA_FROM, MAX_WORKGROUPS,
    };
    use crate::Kernel;
    use qwen36::dims::Dims;

    #[test]
    fn attention_takes_the_cooperative_matrix_kernel_for_heads_it_fits() {
        assert!(attn_prefill_cm_fits(256) && attn_prefill_cm_fits(64));
        for hd in [0, 16, 32, 48, 72, 96, 264, 512] {
            assert!(!attn_prefill_cm_fits(hd), "hd={hd}");
        }
        assert_eq!((ATTN_CM_QUERIES, ATTN_CM_KEYS), (64, 32));
    }

    #[test]
    fn decode_attention_runs_the_gqa_kernel_past_its_threshold_only() {
        for d in [Dims::QWEN36, Dims::tiny()] {
            let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
            for (n_ctx, want) in [
                (1, Kernel::AttnDecode),
                (ATTN_GQA_FROM, Kernel::AttnDecode),
                (ATTN_GQA_FROM + 1, Kernel::AttnDecodeGqa),
                (65_536, Kernel::AttnDecodeGqa),
            ] {
                assert_eq!(
                    attn_decode_kernel(n_ctx, nh, nkv, hd),
                    want,
                    "hd={hd} n_ctx={n_ctx}"
                );
            }
        }
        // A model the GQA kernel can't take keeps attn_decode at any length: nine query heads to a KV head, or heads of
        // 48 values.
        assert_eq!(attn_decode_kernel(65_536, 18, 2, 256), Kernel::AttnDecode);
        assert_eq!(attn_decode_kernel(65_536, 16, 2, 48), Kernel::AttnDecode);
        assert_eq!(attn_decode_kernel(65_536, 16, 2, 64), Kernel::AttnDecodeGqa);
    }

    /// The expert kernels read slots 16 bytes at a time: a VRAM or HOST entry at a misaligned address is refused
    /// where the engine writes it. A null address passes (the kernels refuse it themselves, without reading), as do CPU
    /// entries and unknown kinds, which they never read.
    #[test]
    fn slot_addresses_the_expert_kernels_cant_read_are_refused() {
        use super::{check_slot_addr, WHERE_CPU, WHERE_HOST, WHERE_VRAM};
        for loc in [WHERE_VRAM, WHERE_HOST] {
            check_slot_addr(0x1_0000, loc);
            check_slot_addr(0, loc);
            for bad in [0x1_0008, 0x1_0001, 0x1_000E] {
                assert!(
                    std::panic::catch_unwind(|| check_slot_addr(bad, loc)).is_err(),
                    "{bad:#x} where {loc}"
                );
            }
        }
        check_slot_addr(0x1_0008, WHERE_CPU);
        check_slot_addr(u64::MAX, 7);
    }

    #[test]
    fn groups_allows_vulkans_limit() {
        assert_eq!(groups(MAX_WORKGROUPS as usize * 64, 64), MAX_WORKGROUPS);
    }

    #[test]
    #[should_panic(expected = "workgroups")]
    fn groups_refuses_more_than_vulkans_limit() {
        groups(MAX_WORKGROUPS as usize * 64 + 1, 64);
    }
}
