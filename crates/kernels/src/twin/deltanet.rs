//! Twins of the DeltaNet kernels.
use super::{sigmoid, silu, softplus, strided, tree};

/// `conv_step`: per channel, the window's products summed in order, then SiLU; the state shifts in the raw input.
pub fn conv_step(state: &mut [f32], w: &[f32], x: &[f32], kw: usize) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for (c, yc) in y.iter_mut().enumerate() {
        let st = &mut state[c * (kw - 1)..(c + 1) * (kw - 1)];
        let wc = &w[c * kw..(c + 1) * kw];
        let mut acc = 0f32;
        for t in 0..kw - 1 {
            acc += st[t] * wc[t];
        }
        acc += x[c] * wc[kw - 1];
        st.copy_within(1.., 0);
        st[kw - 2] = x[c];
        *yc = silu(acc);
    }
    y
}

/// `gdn_gates`: (β, g).
pub fn gdn_gates(b: &[f32], alpha: &[f32], dt_bias: &[f32], a: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let beta = b.iter().map(|&x| sigmoid(x)).collect();
    let g = (0..alpha.len())
        .map(|i| a[i] * softplus(alpha[i] + dt_bias[i]))
        .collect();
    (beta, g)
}

/// `l2norm`: each group of `n` values scaled in place by 1 / sqrt(Σx² + eps), the sum as 64 strided partials and the
/// tree.
pub fn l2norm(x: &mut [f32], n: usize, eps: f32) {
    for grp in x.chunks_mut(n) {
        let scale = {
            let mut p = strided(n, 64, |i| grp[i] * grp[i]);
            1.0 / (tree(&mut p) + eps).sqrt()
        };
        for v in grp.iter_mut() {
            *v *= scale;
        }
    }
}

/// `gdn_step`: every value head, each row of its state in order. Value head h uses key group h % n_group.
#[allow(clippy::too_many_arguments)]
pub fn gdn_step(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    n_group: usize,
    dk: usize,
    dv: usize,
) -> Vec<f32> {
    let scale = 1.0 / (dk as f32).sqrt();
    let mut o = vec![0f32; g.len() * dv];
    for h in 0..g.len() {
        let grp = h % n_group;
        let (qh, kh) = (&q[grp * dk..(grp + 1) * dk], &k[grp * dk..(grp + 1) * dk]);
        let decay = g[h].exp();
        for j in 0..dv {
            let m = &mut state[(h * dv + j) * dk..(h * dv + j + 1) * dk];
            let mut u = 0f32;
            for i in 0..dk {
                let e = m[i] * decay;
                m[i] = e;
                u += e * kh[i];
            }
            let delta = (v[h * dv + j] - u) * beta[h];
            let mut acc = 0f32;
            for i in 0..dk {
                let e = m[i] + kh[i] * delta;
                m[i] = e;
                acc += e * qh[i];
            }
            o[h * dv + j] = acc * scale;
        }
    }
    o
}

/// `conv_seq`: `conv_step` for each token of `x` (`[tokens][n]`) in order, the state carried. Without `store` the
/// outputs are the same and `state` is left as it was (the verify pass reads the state and never writes it).
pub fn conv_seq(
    state: &mut [f32],
    w: &[f32],
    x: &[f32],
    kw: usize,
    n: usize,
    store: bool,
) -> Vec<f32> {
    let mut carried = state.to_vec();
    let y = x
        .chunks(n)
        .flat_map(|xt| conv_step(&mut carried, w, xt, kw))
        .collect();
    if store {
        state.copy_from_slice(&carried);
    }
    y
}

/// `gdn_seq`: `gdn_step` for each token in order, the state carried. `q` and `k` hold `[tokens][n_group × dk]`, `v`
/// `[tokens][heads × dv]`, `g` and `beta` `[tokens][heads]`; returns `[tokens][heads × dv]`. Without `store` the
/// outputs are the same and `state` is left as it was.
#[allow(clippy::too_many_arguments)]
pub fn gdn_seq(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    n_group: usize,
    dk: usize,
    dv: usize,
    store: bool,
) -> Vec<f32> {
    let kd = n_group * dk;
    if q.is_empty() {
        // An empty chunk: no output, the state untouched, as the kernel leaves it.
        return Vec::new();
    }
    let heads = g.len() / (q.len() / kd);
    let mut o = Vec::with_capacity(v.len());
    let mut carried = state.to_vec();
    for (t, (qt, kt)) in q.chunks(kd).zip(k.chunks(kd)).enumerate() {
        o.extend(gdn_step(
            &mut carried,
            qt,
            kt,
            &v[t * heads * dv..(t + 1) * heads * dv],
            &g[t * heads..(t + 1) * heads],
            &beta[t * heads..(t + 1) * heads],
            n_group,
            dk,
            dv,
        ));
    }
    if store {
        state.copy_from_slice(&carried);
    }
    o
}

/// `gated_rmsnorm`: per head of `dv` values, 64 strided partial sums and the tree, then ((o · scale) · w) · silu(z).
pub fn gated_rmsnorm(o: &[f32], z: &[f32], w: &[f32], dv: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; o.len()];
    for h in 0..o.len() / dv {
        let oh = &o[h * dv..(h + 1) * dv];
        let mut p = strided(dv, 64, |i| oh[i] * oh[i]);
        let scale = 1.0 / (tree(&mut p) / dv as f32 + eps).sqrt();
        for i in 0..dv {
            y[h * dv + i] = ((oh[i] * scale) * w[i]) * silu(z[h * dv + i]);
        }
    }
    y
}
