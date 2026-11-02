//! The reference's operations (semantics reference §2 to §5): one function per step of the math, f32 values with f64
//! sums, written as plainly as possible. The CPU reference forward pass (`reference.rs`) is built from them, and every
//! GPU kernel's CPU twin is checked against them, so the GPU, the twins and the reference share
//! one definition of the math.
use crate::{dims::Dims, model::Weight, quant};

/// Matrices below this many weights run on one thread; spawning costs more than it saves.
const MIN_PARALLEL: usize = 1 << 20;

/// y = W·x, one f64 dot product per row, rows split across threads for large matrices.
pub fn matvec(w: &Weight, x: &[f32], threads: usize) -> Vec<f32> {
    assert_eq!(
        w.cols,
        x.len(),
        "matvec of a {}-column matrix with {} values",
        w.cols,
        x.len()
    );
    let mut y = vec![0f32; w.rows];
    let rows = |first: usize, ys: &mut [f32]| {
        let mut scratch = Vec::with_capacity(w.cols);
        for (j, yv) in ys.iter_mut().enumerate() {
            *yv = quant::dot_row(w.ty, w.row(first + j), x, &mut scratch) as f32;
        }
    };
    if threads <= 1 || w.rows * w.cols < MIN_PARALLEL {
        rows(0, &mut y);
        return y;
    }
    let chunk = w.rows.div_ceil(threads);
    std::thread::scope(|s| {
        for (i, ys) in y.chunks_mut(chunk).enumerate() {
            s.spawn(move || rows(i * chunk, ys));
        }
    });
    y
}

pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

/// ‖a − b‖ / ‖b‖ over one vector, or ‖a − b‖ when b is all zeros: the error measure of tests 1 and 2.
pub fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let d = *x as f64 - *y as f64;
        num += d * d;
        den += *y as f64 * *y as f64;
    }
    if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}

/// §2: x / sqrt(mean(x²) + eps) ⊙ w.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let s = 1.0 / (dot(x, x) / x.len() as f64 + eps as f64).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, w)| (*v as f64 * s) as f32 * w)
        .collect()
}

/// §4: x / sqrt(Σx² + eps).
pub fn l2norm(x: &[f32], eps: f32) -> Vec<f32> {
    let s = 1.0 / (dot(x, x) + eps as f64).sqrt();
    x.iter().map(|v| (*v as f64 * s) as f32).collect()
}

pub fn softmax(x: &[f32]) -> Vec<f32> {
    let mx = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let e: Vec<f64> = x.iter().map(|&v| (v as f64 - mx).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| (v / s) as f32).collect()
}

pub fn sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + (-(x as f64)).exp())) as f32
}

pub fn silu(x: f32) -> f32 {
    (x as f64 / (1.0 + (-(x as f64)).exp())) as f32
}

/// llama.cpp's softplus: x above 20, log(1 + eᵡ) otherwise.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (x as f64).exp().ln_1p() as f32
    }
}

/// §3: NeoX RoPE on the first `n_rot` dims: pairs (i, i + n_rot/2), θ = pos · base^(−2i/n_rot). The rest is
/// unchanged. For text, IMROPE with sections [11, 11, 10, 0] reduces to exactly this.
pub fn rope(x: &mut [f32], pos: u32, n_rot: usize, base: f32) {
    let half = n_rot / 2;
    for i in 0..half {
        let theta = pos as f64 * (base as f64).powf(-2.0 * i as f64 / n_rot as f64);
        let (s, c) = theta.sin_cos();
        let (a, b) = (x[i] as f64, x[i + half] as f64);
        x[i] = (a * c - b * s) as f32;
        x[i + half] = (a * s + b * c) as f32;
    }
}

/// §3: the Q projection's queries and gates, which are interleaved per head as `[q_h | gate_h]`.
pub fn split_q_gate(qf: &[f32], hd: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut q, mut gate) = (
        Vec::with_capacity(qf.len() / 2),
        Vec::with_capacity(qf.len() / 2),
    );
    for c in qf.chunks(2 * hd) {
        q.extend_from_slice(&c[..hd]);
        gate.extend_from_slice(&c[hd..]);
    }
    (q, gate)
}

/// §3: each head's RMSNorm with `w`, then RoPE at `pos`, in place.
pub fn norm_rope(
    heads: &mut [f32],
    w: &[f32],
    eps: f32,
    hd: usize,
    pos: u32,
    n_rot: usize,
    base: f32,
) {
    norm_heads(heads, w, eps, hd);
    rope_heads(heads, hd, pos, n_rot, base);
}

/// §3: each head's RMSNorm with `w`, in place (`norm_rope`'s first half; the MTP taps the heads between the two).
pub fn norm_heads(heads: &mut [f32], w: &[f32], eps: f32, hd: usize) {
    for x in heads.chunks_mut(hd) {
        let n = rmsnorm(x, w, eps);
        x.copy_from_slice(&n);
    }
}

/// §3: RoPE at `pos` on each head, in place (`norm_rope`'s second half).
pub fn rope_heads(heads: &mut [f32], hd: usize, pos: u32, n_rot: usize, base: f32) {
    for x in heads.chunks_mut(hd) {
        rope(x, pos, n_rot, base);
    }
}

/// §3: one query's softmax attention over `n_ctx` cached positions, exactly: f64 scores and sums. `kc` and `vc` hold
/// `[pos][n_head_kv × head_dim]`; query head i reads KV head i / (n_head / n_head_kv).
pub fn attention(q: &[f32], kc: &[f32], vc: &[f32], n_ctx: usize, d: &Dims) -> Vec<f32> {
    let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
    let (kvw, group) = (nkv * hd, nh / nkv);
    let scale = 1.0 / (hd as f64).sqrt();
    let mut pre = vec![0f32; nh * hd];
    for i in 0..nh {
        let kvh = i / group;
        let keys = |tp: usize| &kc[tp * kvw + kvh * hd..][..hd];
        let values = |tp: usize| &vc[tp * kvw + kvh * hd..][..hd];
        let qh = &q[i * hd..(i + 1) * hd];
        let scores: Vec<f64> = (0..n_ctx).map(|tp| dot(keys(tp), qh) * scale).collect();
        let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
        let sum: f64 = w.iter().sum();
        let mut acc = vec![0f64; hd];
        for (tp, wt) in w.iter().enumerate() {
            for (a, vv) in acc.iter_mut().zip(values(tp)) {
                *a += wt / sum * *vv as f64;
            }
        }
        for (o, a) in pre[i * hd..(i + 1) * hd].iter_mut().zip(&acc) {
            *o = *a as f32;
        }
    }
    pre
}

/// §4: β = sigmoid(b) and g = ssm_a ⊙ softplus(α + dt_bias), from the raw beta and alpha projections. `a` already
/// holds −exp(A_log).
pub fn gdn_gates(b: &[f32], alpha: &[f32], dt_bias: &[f32], a: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let beta = b.iter().map(|&x| sigmoid(x)).collect();
    let g = (0..alpha.len())
        .map(|i| a[i] * softplus(alpha[i] + dt_bias[i]))
        .collect();
    (beta, g)
}

/// §4: the causal conv for one token, post-SiLU. `state` holds each channel's last `kw − 1` raw inputs, oldest
/// first, `[channels][kw − 1]`; kernel tap `kw − 1` multiplies the current input, and the state keeps raw inputs.
pub fn conv_step(state: &mut [f32], w: &[f32], x: &[f32], kw: usize) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for (c, yc) in y.iter_mut().enumerate() {
        let st = &mut state[c * (kw - 1)..(c + 1) * (kw - 1)];
        let wc = &w[c * kw..(c + 1) * kw];
        let acc: f64 = st
            .iter()
            .chain(std::iter::once(&x[c]))
            .zip(wc)
            .map(|(x, w)| *x as f64 * *w as f64)
            .sum();
        *yc = silu(acc as f32);
        st.rotate_left(1);
        st[kw - 2] = x[c];
    }
    y
}

/// §4: the conv output split into q, k and v, with each key group of q and k L2-normalized.
pub fn split_qkv(y: &[f32], d: &Dims, eps: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (kd, dk) = (d.key_dim(), d.ssm_d_state);
    let q = y[..kd].chunks(dk).flat_map(|c| l2norm(c, eps)).collect();
    let k = y[kd..2 * kd]
        .chunks(dk)
        .flat_map(|c| l2norm(c, eps))
        .collect();
    (q, k, y[2 * kd..].to_vec())
}

/// §4: the gated delta rule for one token, every value head. `s` holds, per value head, `M[j][i] = S[i][j]` with the
/// key index contiguous, `[v_heads][v][k]`. Value head h uses key group h % n_group. Decay first, then the delta
/// update; the output reads the updated state.
pub fn delta_rule(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    d: &Dims,
) -> Vec<f32> {
    let (ng, nv, dk, dv) = (d.ssm_n_group, d.ssm_dt_rank, d.ssm_d_state, d.v_head_dim());
    let scale = 1.0 / (dk as f64).sqrt();
    let mut o = vec![0f32; nv * dv];
    for hd in 0..nv {
        let gi = hd % ng;
        let (qh, kh) = (&q[gi * dk..(gi + 1) * dk], &k[gi * dk..(gi + 1) * dk]);
        let sh = &mut s[hd * dv * dk..(hd + 1) * dv * dk];
        let decay = (g[hd] as f64).exp();
        for e in sh.iter_mut() {
            *e = (*e as f64 * decay) as f32;
        }
        for j in 0..dv {
            let col = &mut sh[j * dk..(j + 1) * dk]; // S[·][j]
            let u = dot(col, kh);
            let delta = (v[hd * dv + j] as f64 - u) * beta[hd] as f64;
            for (sv, kv) in col.iter_mut().zip(kh) {
                *sv = (*sv as f64 + *kv as f64 * delta) as f32;
            }
            o[hd * dv + j] = (dot(col, qh) * scale) as f32;
        }
    }
    o
}

/// §4: per value head, RMSNorm(o_h) ⊙ ssm_norm ⊙ silu(z_h). `w` is the raw HF weight, shared by all heads.
pub fn gated_norm(o: &[f32], z: &[f32], w: &[f32], eps: f32, dv: usize) -> Vec<f32> {
    let mut n = vec![0f32; o.len()];
    for hd in 0..o.len() / dv {
        let normed = rmsnorm(&o[hd * dv..(hd + 1) * dv], w, eps);
        for j in 0..dv {
            n[hd * dv + j] = normed[j] * silu(z[hd * dv + j]);
        }
    }
    n
}

/// §5: the `k` most probable experts, most probable first, ties to the lower id.
pub fn top_k(probs: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
    order.truncate(k);
    order
}

/// §5: the chosen experts' probabilities normalized to sum to 1 (with llama.cpp's floor on the sum), no scale.
pub fn route_weights(probs: &[f32], top: &[usize]) -> Vec<f32> {
    let sum: f64 = top.iter().map(|&e| probs[e] as f64).sum();
    top.iter()
        .map(|&e| (probs[e] as f64 / sum.max(6.103515625e-5)) as f32)
        .collect()
}

/// silu(g) ⊙ u.
pub fn swiglu(g: &[f32], u: &[f32]) -> Vec<f32> {
    g.iter().zip(u).map(|(g, u)| silu(*g) * u).collect()
}

/// §5: Σ_j w_j · y_j over the routed experts in routed order, in f64. `ys` holds `[k][n]`.
pub fn moe_sum(ys: &[f32], w: &[f32], n: usize) -> Vec<f32> {
    let mut acc = vec![0f64; n];
    for (j, &wj) in w.iter().enumerate() {
        for (a, y) in acc.iter_mut().zip(&ys[j * n..(j + 1) * n]) {
            *a += wj as f64 * *y as f64;
        }
    }
    acc.into_iter().map(|v| v as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{quant::WType, testing::Rng};

    #[test]
    fn rope_at_position_zero_is_the_identity() {
        let mut x: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 3.0).collect();
        let before = x.clone();
        rope(&mut x, 0, 16, 1e7);
        assert_eq!(x, before);
    }

    #[test]
    fn rope_rotates_pairs_and_keeps_their_length() {
        let mut x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let before = x.clone();
        rope(&mut x, 1234, 16, 1e7);
        for i in 0..8 {
            let n0 = before[i].hypot(before[i + 8]);
            let n1 = x[i].hypot(x[i + 8]);
            assert!((n0 - n1).abs() < 1e-5, "pair {i}");
        }
        assert_eq!(x[16..], before[16..], "dims past n_rot are unchanged");
        assert_ne!(x[0], before[0]);
    }

    #[test]
    fn threaded_matvec_matches_one_thread() {
        let (rows, cols) = (1024, 1024);
        let mut rng = Rng::new(7);
        let mut bytes = vec![0u8; rows * WType::Q8_0.row_bytes(cols)];
        rng.fill(&mut bytes);
        for b in bytes.chunks_mut(34) {
            b[0..2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
        }
        let w = Weight {
            ty: WType::Q8_0,
            cols,
            rows,
            bytes: &bytes,
        };
        let x: Vec<f32> = (0..cols).map(|_| rng.uniform(-1.0, 1.0)).collect();
        assert_eq!(matvec(&w, &x, 1), matvec(&w, &x, 7));
    }

    #[test]
    fn softplus_matches_its_definition() {
        assert_eq!(softplus(25.0), 25.0);
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        assert!((softplus(-30.0) - (-30f64).exp() as f32).abs() < 1e-20);
    }

    #[test]
    fn rel_l2_of_equal_vectors_is_zero_and_of_zeros_is_absolute() {
        assert_eq!(rel_l2(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        assert_eq!(rel_l2(&[3.0, 4.0], &[0.0, 0.0]), 5.0);
        assert!((rel_l2(&[1.1, 0.0], &[1.0, 0.0]) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn top_k_breaks_ties_toward_the_lower_id() {
        assert_eq!(top_k(&[0.1, 0.3, 0.3, 0.2, 0.1], 3), vec![1, 2, 3]);
        let w = route_weights(&[0.1, 0.3, 0.3, 0.2, 0.1], &[1, 2, 3]);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((w[0] - 0.375).abs() < 1e-6);
    }

    #[test]
    fn moe_sum_weights_each_expert() {
        let ys = [1.0, 2.0, 10.0, 20.0];
        assert_eq!(moe_sum(&ys, &[0.5, 0.25], 2), vec![3.0, 6.0]);
    }
}
