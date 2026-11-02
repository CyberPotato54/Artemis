//! CPU twins of the kernels. Each mirrors its kernel's arithmetic: the same f32 operations in the same
//! order, including every reduction's tree. Test 1 compares kernel and twin on the GPU; `tests/twins.rs` compares each
//! twin with `qwen36::ops` in CI.
pub mod argmax;
pub use argmax::argmax_row;
pub mod attention;
pub mod deltanet;
pub mod matvec;
pub mod moe;
pub mod silu_det;

use qwen36::quant::{self, WType};

/// A tree sum over `p.len()` lanes (a power of two) that halves its stride each step: `reduce.glsl`'s order.
pub fn tree(p: &mut [f32]) -> f32 {
    assert!(p.len().is_power_of_two(), "a tree over {} lanes", p.len());
    let mut s = p.len() / 2;
    while s > 0 {
        for i in 0..s {
            p[i] += p[i + s];
        }
        s /= 2;
    }
    p[0]
}

/// A tree max over `p.len()` lanes (a power of two): `reduce.glsl`'s `red_max`. Max is exact, so only the result
/// matters, but the twin keeps the shape.
pub fn tree_max(p: &mut [f32]) -> f32 {
    assert!(p.len().is_power_of_two(), "a tree over {} lanes", p.len());
    let mut s = p.len() / 2;
    while s > 0 {
        for i in 0..s {
            p[i] = p[i].max(p[i + s]);
        }
        s /= 2;
    }
    p[0]
}

/// Each of `lanes` lanes' in-order sum of f(i) over i = lane, lane + lanes, … below `n`: a strided loop's partials.
pub fn strided(n: usize, lanes: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..lanes)
        .map(|t| (t..n).step_by(lanes).fold(0f32, |s, i| s + f(i)))
        .collect()
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// log(1 + e) for e in [0, 1], as `common.glsl` computes it: 2·atanh(z), z = e / (2 + e).
pub fn log1p_small(e: f32) -> f32 {
    let z = e / (2.0 + e);
    let z2 = z * z;
    let mut p = 1.0f32 / 15.0;
    for c in [
        1.0f32 / 13.0,
        1.0 / 11.0,
        1.0 / 9.0,
        1.0 / 7.0,
        1.0 / 5.0,
        1.0 / 3.0,
        1.0,
    ] {
        p = p * z2 + c;
    }
    2.0 * z * p
}

/// llama.cpp's softplus, as `common.glsl` computes it: max(u, 0) + log1p(exp(−|u|)), or u above 20.
pub fn softplus(u: f32) -> f32 {
    if u > 20.0 {
        u
    } else {
        u.max(0.0) + log1p_small((-u.abs()).exp())
    }
}

/// f32 to f16 bits, rounding to nearest even: `common.glsl`'s `f16_bits`.
pub fn f16_bits(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

pub fn f16_value(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}

/// `embed_row`: the row dequantized, as `quant.glsl`'s `dq` does element by element.
pub fn embed_row(ty: WType, row: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    quant::dequant_row(ty, row, &mut out);
    out
}

/// `add`.
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// `rmsnorm`: 256 strided partial sums of squares, the tree, then (x · scale) · w.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut p = strided(x.len(), 256, |i| x[i] * x[i]);
    let total = tree(&mut p);
    let scale = 1.0 / (total / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| (v * scale) * w).collect()
}
