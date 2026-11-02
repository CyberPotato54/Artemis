//! Twins of the attention kernels, and the RoPE table the engine passes to `qk_norm_rope`.
use super::matvec::split16;
use super::{f16_bits, f16_value, sigmoid, strided, tree, tree_max};
use crate::dispatch::{ATTN_CHUNK, ATTN_CM_KEYS, ATTN_KEYS};

/// The step's rotation table: cos θ_i for i < n_rot/2, then sin θ_i, with θ_i = pos · base^(−2i/n_rot) computed in
/// f64 as `qwen36::ops::rope` does, then rounded to f32.
pub fn rope_table(pos: u32, n_rot: usize, base: f32) -> Vec<f32> {
    let half = n_rot / 2;
    let mut t = vec![0f32; n_rot];
    for i in 0..half {
        let theta = pos as f64 * (base as f64).powf(-2.0 * i as f64 / n_rot as f64);
        let (s, c) = theta.sin_cos();
        t[i] = c as f32;
        t[half + i] = s as f32;
    }
    t
}

/// NeoX RoPE on the first table.len() dims of `x`, in f32 with the table's values.
pub fn rotate(x: &mut [f32], table: &[f32]) {
    let half = table.len() / 2;
    for t in 0..half {
        let (c, s) = (table[t], table[half + t]);
        let (a, b) = (x[t], x[t + half]);
        x[t] = a * c - b * s;
        x[t + half] = a * s + b * c;
    }
}

/// One head: 64 strided partial sums of squares, the tree, (x · scale) · w, then the rotation.
fn norm_rope_head(x: &[f32], w: &[f32], table: &[f32], eps: f32) -> Vec<f32> {
    let hd = x.len();
    let mut p = strided(hd, 64, |i| x[i] * x[i]);
    let scale = 1.0 / (tree(&mut p) / hd as f32 + eps).sqrt();
    let mut xs: Vec<f32> = x.iter().zip(w).map(|(v, w)| (v * scale) * w).collect();
    rotate(&mut xs, table);
    xs
}

/// `qk_norm_rope`: (q, gate) from the interleaved projection; `k` normed and rotated in place.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope(
    qfull: &[f32],
    k: &mut [f32],
    q_norm: &[f32],
    k_norm: &[f32],
    table: &[f32],
    n_head: usize,
    hd: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let (mut q, mut gate) = (
        Vec::with_capacity(n_head * hd),
        Vec::with_capacity(n_head * hd),
    );
    for h in 0..n_head {
        let src = &qfull[2 * h * hd..2 * h * hd + hd];
        q.extend(norm_rope_head(src, q_norm, table, eps));
        gate.extend_from_slice(&qfull[2 * h * hd + hd..2 * (h + 1) * hd]);
    }
    for head in k.chunks_mut(hd) {
        let r = norm_rope_head(head, k_norm, table, eps);
        head.copy_from_slice(&r);
    }
    (q, gate)
}

/// `kv_write`: position `pos` of the `[ctx][width]` f16 caches.
pub fn kv_write(k: &[f32], v: &[f32], kc: &mut [u16], vc: &mut [u16], pos: usize, width: usize) {
    for i in 0..width {
        kc[pos * width + i] = f16_bits(k[i]);
        vc[pos * width + i] = f16_bits(v[i]);
    }
}

/// `attn_decode` then `attn_combine`, for every query head.
pub fn attention(
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    n_ctx: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
) -> Vec<f32> {
    let (group, width) = (n_head / n_kv, n_kv * hd);
    let scale = 1.0 / (hd as f32).sqrt();
    let n_split = n_ctx.div_ceil(ATTN_CHUNK);
    let mut out = vec![0f32; n_head * hd];
    for h in 0..n_head {
        let kvh = h / group;
        let qh = &q[h * hd..(h + 1) * hd];
        let parts: Vec<(f32, f32, Vec<f32>)> = (0..n_split)
            .map(|s| {
                let t0 = s * ATTN_CHUNK;
                let nt = ATTN_CHUNK.min(n_ctx - t0);
                let mut scores = vec![f32::NEG_INFINITY; ATTN_CHUNK];
                for (t, sc) in scores.iter_mut().enumerate().take(nt) {
                    let kr = &kc[(t0 + t) * width + kvh * hd..][..hd];
                    let acc = (0..hd).fold(0f32, |a, i| a + f16_value(kr[i]) * qh[i]);
                    *sc = acc * scale;
                }
                let m = tree_max(&mut scores.clone());
                let ws: Vec<f32> = (0..ATTN_CHUNK)
                    .map(|t| if t < nt { (scores[t] - m).exp() } else { 0.0 })
                    .collect();
                let l = tree(&mut ws.clone());
                let acc = (0..hd)
                    .map(|d| {
                        (0..nt).fold(0f32, |a, j| {
                            a + ws[j] * f16_value(vc[(t0 + j) * width + kvh * hd + d])
                        })
                    })
                    .collect();
                (m, l, acc)
            })
            .collect();
        let m = parts[1..].iter().fold(parts[0].0, |m, p| m.max(p.0));
        for d in 0..hd {
            let (mut l, mut acc) = (0f32, 0f32);
            for p in &parts {
                let f = (p.0 - m).exp();
                l += p.1 * f;
                acc += p.2[d] * f;
            }
            out[h * hd + d] = acc / l;
        }
    }
    out
}

/// `attn_decode_tokens` (or `attn_decode_gqa_tokens`) then `attn_combine_tokens`: `tokens` queries, `q` laid out
/// `[tokens][n_head·hd]`, query i attending the first `n_ctx0 + i` cached positions, each as `attention` does at that
/// context. The kernels give every query decode's kernel and split count for its own length, so the twin is
/// `attention` per query. Returns `[tokens][n_head·hd]`.
#[allow(clippy::too_many_arguments)]
pub fn attention_tokens(
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    n_ctx0: usize,
    tokens: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
) -> Vec<f32> {
    let width = n_kv * hd;
    let last = n_ctx0 + tokens - 1;
    assert!(
        tokens >= 1 && n_ctx0 >= 1 && kc.len() >= last * width && vc.len() >= last * width,
        "attention_tokens: the cache holds {} positions, the last of {tokens} queries at {n_ctx0} reads {last}",
        kc.len() / width
    );
    let mut y = Vec::with_capacity(tokens * n_head * hd);
    for i in 0..tokens {
        let qi = &q[i * n_head * hd..(i + 1) * n_head * hd];
        y.extend(attention(qi, kc, vc, n_ctx0 + i, n_head, n_kv, hd));
    }
    y
}

/// `attn_prefill`: each of the `tokens` queries, query i at position `pos` + i, over the cached positions 0 ..= its
/// own, in blocks of `ATTN_KEYS` from 0, with the kernel's online softmax: per block, scores in order of the head's
/// dims, m' = max(m, the block's max), α = exp(m − m'), l' = l·α + Σ p, acc' = acc·α + Σ p·v, each sum in key order.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill(
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    pos: usize,
    tokens: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
) -> Vec<f32> {
    let (group, width) = (n_head / n_kv, n_kv * hd);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut y = vec![0f32; tokens * n_head * hd];
    for i in 0..tokens {
        let last = pos + i;
        for h in 0..n_head {
            let kvh = h / group;
            let qh = &q[(i * n_head + h) * hd..][..hd];
            let at = |p: usize, d: usize| p * width + kvh * hd + d;
            let (mut m, mut l) = (f32::NEG_INFINITY, 0f32);
            let mut acc = vec![0f32; hd];
            for k0 in (0..=last).step_by(ATTN_KEYS) {
                let n = ATTN_KEYS.min(last + 1 - k0);
                let s: Vec<f32> = (0..n)
                    .map(|j| {
                        (0..hd).fold(0f32, |a, d| a + qh[d] * f16_value(kc[at(k0 + j, d)])) * scale
                    })
                    .collect();
                let mn = m.max(s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
                let alpha = (m - mn).exp();
                let p: Vec<f32> = s.iter().map(|&x| (x - mn).exp()).collect();
                l = l * alpha + p.iter().fold(0f32, |a, b| a + b);
                m = mn;
                for (d, ad) in acc.iter_mut().enumerate() {
                    *ad = (0..n).fold(*ad * alpha, |a, j| a + p[j] * f16_value(vc[at(k0 + j, d)]));
                }
            }
            for (d, ad) in acc.iter().enumerate() {
                y[(i * n_head + h) * hd + d] = ad / l;
            }
        }
    }
    y
}

/// Which operands of `attn_prefill_cm` keep their lo f16: q (`q_lo`) and the softmax weights p (`p_lo`). The kernel
/// keeps both (`BOTH`); the others exist so a test can measure what each split buys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CmSplit {
    pub q_lo: bool,
    pub p_lo: bool,
}

impl CmSplit {
    pub const BOTH: CmSplit = CmSplit {
        q_lo: true,
        p_lo: true,
    };
}

/// `attn_prefill_cm`: each of the `tokens` queries, query i at position `pos` + i, over the cached positions 0 ..=
/// its own, in blocks of `ATTN_CM_KEYS` from 0 (a block wholly past the query is skipped: in the kernel it is an exact
/// no-op), with the kernel's split-f16 arithmetic. q is split once, (hi, lo) = `split16`; K and V are the cache's
/// exact f16s. A score sums the head's dims 32 at a time: a slice's partial starts at 0 and takes, for each 16-dim
/// half in order, lo·k over its dims in order and then hi·k over the same dims, and the slice's partial is then added
/// to the score; the score is scaled by 1/√hd. A block's max m' = max(m, the block's max) and α (1 if m' = m, 0 if m
/// was −∞, else exp(m − m')) are per row, and p = exp(s − m'). The block's sum is four quarter-sums, over keys
/// 8c .. 8c + 8 for c in 0..4 (those the query sees), each in key order, added as ((q0 + q1) + q2) + q3; l' = l·α +
/// that sum, the product rounded before the add. p is split by `split16` too, and the block's PV is one partial a dim,
/// from 0, taking for each 16-key half in order p_lo·v over its keys in order and then p_hi·v; acc' = acc·α + partial,
/// again rounded apart. The output is acc / l. `split` can drop q's lo or p's lo (set to 0). The twin and the kernel
/// differ only in the order of the sums inside a slice (each f16 × f16 product is exact in f32) and in the GPU's
/// rounding of `exp` and of the division.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill_cm_split(
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    pos: usize,
    tokens: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    split: CmSplit,
) -> Vec<f32> {
    assert!(
        hd.is_multiple_of(32),
        "attn_prefill_cm: a score sums the head's dims 32 at a time, so hd ({hd}) must be a multiple of 32"
    );
    let (group, width) = (n_head / n_kv, n_kv * hd);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut y = vec![0f32; tokens * n_head * hd];
    for i in 0..tokens {
        let last = pos + i;
        for h in 0..n_head {
            let kvh = h / group;
            let (qhi, qlo): (Vec<f32>, Vec<f32>) = q[(i * n_head + h) * hd..][..hd]
                .iter()
                .map(|&v| {
                    let (hi, lo) = split16(v);
                    (hi, if split.q_lo { lo } else { 0.0 })
                })
                .unzip();
            let at = |p: usize, d: usize| p * width + kvh * hd + d;
            let (mut m, mut l) = (f32::NEG_INFINITY, 0f32);
            let mut acc = vec![0f32; hd];
            for k0 in (0..=last).step_by(ATTN_CM_KEYS) {
                let n = ATTN_CM_KEYS.min(last + 1 - k0);
                let x: Vec<f32> = (0..n)
                    .map(|j| {
                        let mut s = 0f32;
                        for d0 in (0..hd).step_by(32) {
                            let mut part = 0f32;
                            for half in [d0, d0 + 16] {
                                for d in half..half + 16 {
                                    part += qlo[d] * f16_value(kc[at(k0 + j, d)]);
                                }
                                for d in half..half + 16 {
                                    part += qhi[d] * f16_value(kc[at(k0 + j, d)]);
                                }
                            }
                            s += part;
                        }
                        s * scale
                    })
                    .collect();
                let mn = m.max(x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
                let alpha = if mn == m {
                    1.0
                } else if m == f32::NEG_INFINITY {
                    0.0
                } else {
                    (m - mn).exp()
                };
                let p: Vec<f32> = x.iter().map(|&v| (v - mn).exp()).collect();
                let quarter = |c: usize| (8 * c..(8 * c + 8).min(n)).fold(0f32, |a, j| a + p[j]);
                let sum = ((quarter(0) + quarter(1)) + quarter(2)) + quarter(3);
                l = l * alpha + sum;
                m = mn;
                let (phi, plo): (Vec<f32>, Vec<f32>) = p
                    .iter()
                    .map(|&v| {
                        let (hi, lo) = split16(v);
                        (hi, if split.p_lo { lo } else { 0.0 })
                    })
                    .unzip();
                for (d, ad) in acc.iter_mut().enumerate() {
                    let mut part = 0f32;
                    for half in [0, 16] {
                        let keys = (half..half + 16).take_while(|&j| j < n);
                        for j in keys.clone() {
                            part += plo[j] * f16_value(vc[at(k0 + j, d)]);
                        }
                        for j in keys {
                            part += phi[j] * f16_value(vc[at(k0 + j, d)]);
                        }
                    }
                    *ad *= alpha;
                    *ad += part;
                }
            }
            for (d, ad) in acc.iter().enumerate() {
                y[(i * n_head + h) * hd + d] = ad / l;
            }
        }
    }
    y
}

/// `attn_prefill_cm`: `attn_prefill_cm_split` with q and p both split, as the kernel runs.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill_cm(
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    pos: usize,
    tokens: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
) -> Vec<f32> {
    attn_prefill_cm_split(q, kc, vc, pos, tokens, n_head, n_kv, hd, CmSplit::BOTH)
}

/// `sigmoid_mul`.
pub fn sigmoid_mul(a: &[f32], g: &[f32]) -> Vec<f32> {
    a.iter().zip(g).map(|(x, g)| x * sigmoid(*g)).collect()
}
