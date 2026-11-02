//! `matvec_tokens`: decode's matvec over up to `MAX_TOKENS` tokens, so a verify step reads each weight row once.
use super::MAX_TOKENS;
use crate::{dispatch::groups, dispatch::MatW, Kernel, Kernels, WType};
use vk::Recorder;

/// `matvec_tokens_{type}`'s push block (`matvec_tokens.glsl`). Strides are in f32 values.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MatvecTokensPush {
    pub w: u64,
    pub x: u64,
    pub y: u64,
    pub rows: u32,
    pub k: u32,
    pub tokens: u32,
    pub x_stride: u32,
    pub y_stride: u32,
    pub pad: u32,
}
push_constants!(MatvecTokensPush: u64, u64, u64, u32, u32, u32, u32, u32, u32);

impl Kernels {
    /// `y` ← W·x for `tokens` input vectors: token `i`'s `w.cols` inputs are at `x + 4·i·x_stride` and its `w.rows`
    /// outputs at `y + 4·i·y_stride` (strides in values). Each token's outputs are `matvec`'s, summed in its order
    /// (`unit_dot_n` per weight unit, then one tree per token); the weights are read once.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_tokens(
        &self,
        r: &mut Recorder,
        w: MatW,
        x: u64,
        y: u64,
        tokens: usize,
        x_stride: usize,
        y_stride: usize,
    ) {
        assert!(
            (1..=MAX_TOKENS).contains(&tokens),
            "a matvec over {tokens} tokens: 1 to {MAX_TOKENS} are supported"
        );
        let per = w.ty.block().0.max(32);
        assert!(
            w.cols.is_multiple_of(per),
            "a matvec over {} rows of {} values: rows hold whole {per}-value units",
            w.ty.name(),
            w.cols
        );
        // As `matvec`: `unit_dot_n` reads x, and F32, BF16, Q4_K and Q5_K rows, 16 bytes at a time; Q8_0 and Q6_K rows
        // two bytes at a time. Every token's input must be 16-byte aligned too, so the stride is a multiple of 4.
        let wide = matches!(w.ty, WType::F32 | WType::Bf16 | WType::Q4K | WType::Q5K);
        let need = if wide { 16 } else { 2 };
        assert!(
            x.is_multiple_of(16) && w.addr.is_multiple_of(need),
            "a matvec_tokens over {} at {:#x} with x at {x:#x}: x must be 16-byte aligned and the weights {need}-byte aligned",
            w.ty.name(),
            w.addr
        );
        assert!(
            x_stride.is_multiple_of(4) && x_stride >= w.cols,
            "a matvec_tokens over {} inputs with x_stride {x_stride}: a multiple of 4 values (16 bytes), at least the inputs",
            w.cols
        );
        // `unit_dot_n` forms a token's input offset in 32 bits (`4u * i * x_stride`), and the push holds both strides
        // as u32: the last token's inputs must end inside that.
        assert!(
            4 * (tokens - 1) * x_stride + 4 * w.cols <= u32::MAX as usize && y_stride <= u32::MAX as usize,
            "a matvec_tokens over {tokens} tokens with x_stride {x_stride} and y_stride {y_stride}: the strides overflow 32 bits"
        );
        assert!(
            y.is_multiple_of(4) && y_stride >= w.rows,
            "a matvec_tokens over {} outputs at {y:#x} with y_stride {y_stride}: y 4-byte aligned, the stride at least the outputs",
            w.rows
        );
        let push = MatvecTokensPush {
            w: w.addr,
            x,
            y,
            rows: w.rows as u32,
            k: w.cols as u32,
            tokens: tokens as u32,
            x_stride: x_stride as u32,
            y_stride: y_stride as u32,
            pad: 0,
        };
        r.dispatch(
            self.pipe(Kernel::matvec_tokens(w.ty)),
            &push,
            [groups(w.rows, 8), 1, 1],
        );
    }
}
