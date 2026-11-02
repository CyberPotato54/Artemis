//! Twins of `matvec_{type}` and of `quant.glsl`'s `unit_dot`, which the expert kernels share.
use super::tree;
use qwen36::quant::WType;

fn f16_at(b: &[u8], i: usize) -> f32 {
    half::f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// ggml's get_scale_min_k4 over the 12 packed bytes `s`.
fn scale_min_k4(s: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        (
            (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4),
            (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
        )
    }
}

/// Σ w·x over 32-value unit `u` of `row`, with x[32u .. 32u + 32], in `unit_dot`'s operations.
pub fn unit_dot(t: WType, row: &[u8], u: usize, x: &[f32]) -> f32 {
    let x = &x[32 * u..32 * u + 32];
    match t {
        WType::F32 => {
            let w = &row[128 * u..128 * u + 128];
            (0..32).fold(0f32, |s, i| {
                s + f32::from_le_bytes([w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3]]) * x[i]
            })
        }
        WType::Bf16 => {
            let w = &row[64 * u..64 * u + 64];
            (0..32).fold(0f32, |s, i| {
                s + half::bf16::from_le_bytes([w[2 * i], w[2 * i + 1]]).to_f32() * x[i]
            })
        }
        WType::Q8_0 => {
            let b = &row[34 * u..34 * u + 34];
            let s = (0..32).fold(0f32, |s, i| s + (b[2 + i] as i8) as f32 * x[i]);
            f16_at(b, 0) * s
        }
        WType::Q4K | WType::Q5K => {
            let q5 = t == WType::Q5K;
            let bsz = if q5 { 176 } else { 144 };
            let b = &row[bsz * (u / 8)..bsz * (u / 8) + bsz];
            let (j, c) = (u % 8, (u % 8) / 2);
            let sh = 4 * (j % 2);
            let (sc, mn) = scale_min_k4(&b[4..16], j);
            let qs = &b[if q5 { 48 } else { 16 } + 32 * c..];
            let (mut sq, mut sx) = (0f32, 0f32);
            for l in 0..32 {
                let mut q = (qs[l] >> sh) & 0xF;
                if q5 && (b[16 + l] >> j) & 1 != 0 {
                    q += 16;
                }
                sq += q as f32 * x[l];
                sx += x[l];
            }
            (f16_at(b, 0) * sc as f32) * sq - (f16_at(b, 2) * mn as f32) * sx
        }
        WType::Q6K => {
            let b = &row[210 * (u / 8)..210 * (u / 8) + 210];
            let j = u % 8;
            let (n, m) = (j / 4, j % 4);
            let ql = &b[64 * n + 32 * (m & 1)..];
            let qh = &b[128 + 32 * n..];
            let (shl, shh) = (4 * (m >> 1), 2 * m);
            let (mut sa, mut sb) = (0f32, 0f32);
            for l in 0..32 {
                let lo = (ql[l] >> shl) & 0xF;
                let hi = (qh[l] >> shh) & 3;
                let prod = ((lo | (hi << 4)) as i32 - 32) as f32 * x[l];
                if l < 16 {
                    sa += prod;
                } else {
                    sb += prod;
                }
            }
            let sca = (b[192 + 8 * n + 2 * m] as i8) as f32;
            let scb = (b[192 + 8 * n + 2 * m + 1] as i8) as f32;
            f16_at(b, 208) * (sca * sa + scb * sb)
        }
    }
}

/// One row's dot product as a 32-lane row of a matvec workgroup computes it.
pub fn row_dot(t: WType, row: &[u8], x: &[f32]) -> f32 {
    let units = x.len() / 32;
    let mut p: Vec<f32> = (0..32)
        .map(|lane| {
            (lane..units)
                .step_by(32)
                .fold(0f32, |s, u| s + unit_dot(t, row, u, x))
        })
        .collect();
    tree(&mut p)
}

/// One output of `matmul_{type}`: the products of the row, dequantized as `dequant_row` gives it, and x, summed in
/// order in f32.
pub fn dot_in_order(t: WType, row: &[u8], x: &[f32]) -> f32 {
    let mut w = vec![0f32; x.len()];
    qwen36::quant::dequant_row(t, row, &mut w);
    w.iter().zip(x).fold(0f32, |s, (a, b)| s + a * b)
}

/// `matmul_{type}`: `y[t][r]` = Σ_i w[r][i]·x[t][i] for `tokens` vectors of x.len() / tokens values, as
/// `[tokens][rows]`. Each output adds its products in order, in f32, with the weights as `dequant_row` gives them.
pub fn matmul(t: WType, w: &[u8], rows: usize, x: &[f32], tokens: usize) -> Vec<f32> {
    assert!(
        tokens > 0 && x.len().is_multiple_of(tokens),
        "{} input values are not {tokens} whole vectors",
        x.len()
    );
    let k = x.len() / tokens;
    let rb = t.row_bytes(k);
    let mut y = vec![0f32; tokens * rows];
    let mut wr = vec![0f32; k];
    for r in 0..rows {
        qwen36::quant::dequant_row(t, &w[r * rb..(r + 1) * rb], &mut wr);
        for (tok, xs) in x.chunks_exact(k).enumerate() {
            y[tok * rows + r] = wr.iter().zip(xs).fold(0f32, |s, (a, b)| s + a * b);
        }
    }
    y
}

/// `matvec_{type}`: y = W·x for `rows` rows of x.len() values.
pub fn matvec(t: WType, w: &[u8], rows: usize, x: &[f32]) -> Vec<f32> {
    let rb = t.row_bytes(x.len());
    (0..rows)
        .map(|r| row_dot(t, &w[r * rb..(r + 1) * rb], x))
        .collect()
}

/// `matvec_tokens_{type}`: `matvec` once a token, as `[tokens][rows]` from `x` as `[tokens][k]`. The kernel gives every
/// token decode's matvec arithmetic (`unit_dot_n` sums each token's unit as `unit_dot` does, then one tree a token).
pub fn matvec_tokens(t: WType, w: &[u8], rows: usize, x: &[f32], tokens: usize) -> Vec<f32> {
    assert!(
        tokens > 0 && !x.is_empty() && x.len().is_multiple_of(tokens),
        "{} input values are not {tokens} whole vectors",
        x.len()
    );
    x.chunks_exact(x.len() / tokens)
        .flat_map(|xs| matvec(t, w, rows, xs))
        .collect()
}

/// `v` as `split2` (quant8.glsl) stores it, two f16s back in f32: hi = v rounded to nearest even, lo = v − hi rounded
/// the same way. hi + lo is within 2⁻²¹ of v, or 2⁻²⁵ absolute where lo is subnormal.
pub fn split16(v: f32) -> (f32, f32) {
    let hi = half::f16::from_f32(v).to_f32();
    (hi, half::f16::from_f32(v - hi).to_f32())
}

/// Every value of `v` through `split16`, as the (hi, lo) pairs `dot_split_pairs` takes. The split-f16 kernels' twins
/// share it.
pub fn split_all(v: &[f32]) -> Vec<(f32, f32)> {
    v.iter().map(|&a| split16(a)).collect()
}

/// `dot_split` over values already split by `split16`, as (hi, lo) pairs. The split-f16 kernels' twins share it.
pub fn dot_split_pairs(w: &[(f32, f32)], x: &[(f32, f32)]) -> f32 {
    assert_eq!(
        w.len(),
        x.len(),
        "a dot product over rows of different lengths"
    );
    let mut s = 0f32;
    for (ws, xs) in w.chunks(32).zip(x.chunks(32)) {
        // The slice's sum from zero, each 16-value half adding its lo·hi, hi·lo, then hi·hi products in k order.
        let mut part = 0f32;
        for (wb, xb) in ws.chunks(16).zip(xs.chunks(16)) {
            for (a, b) in wb.iter().zip(xb) {
                part += b.1 * a.0;
            }
            for (a, b) in wb.iter().zip(xb) {
                part += b.0 * a.1;
            }
            for (a, b) in wb.iter().zip(xb) {
                part += b.0 * a.0;
            }
        }
        s += part;
    }
    s
}

/// One output of a split-f16 cooperative-matrix kernel over weights `w` and inputs `x`, in the kernel's structure: for
/// each 32-value slice of k, a sum from zero over its two 16-value halves, each adding lo_x·hi_w, then hi_x·lo_w, then
/// hi_x·hi_w (the kernel's six products into a zeroed accumulator); then the slice's sum added to the running total in
/// f32. The kernel sums per slice because the hardware's accumulate inside `coopMatMulAdd` measured biased, its error
/// growing linearly with k, where a per-slice partial added with an ordinary f32 add grows as √k. Each product of two
/// f16s is exact in f32; the twin adds a slice's products in k order and the hardware in its own, so the two agree
/// within the kernel's tolerance, not bit for bit.
pub fn dot_split(w: &[f32], x: &[f32]) -> f32 {
    dot_split_pairs(&split_all(w), &split_all(x))
}

/// `matmul_cm_{type}`: `matmul`'s outputs, `[tokens][rows]`, from split-f16 weights (as `dequant_row` gives them) and
/// inputs, each output summed per 32-value slice as `dot_split` describes.
pub fn matmul_cm(t: WType, w: &[u8], rows: usize, x: &[f32], tokens: usize) -> Vec<f32> {
    assert!(
        tokens > 0 && x.len().is_multiple_of(tokens),
        "{} input values are not {tokens} whole vectors",
        x.len()
    );
    let k = x.len() / tokens;
    let rb = t.row_bytes(k);
    let xs = split_all(x);
    let mut y = vec![0f32; tokens * rows];
    let mut wr = vec![0f32; k];
    for r in 0..rows {
        qwen36::quant::dequant_row(t, &w[r * rb..(r + 1) * rb], &mut wr);
        let ws = split_all(&wr);
        for (tok, xt) in xs.chunks(k).enumerate() {
            y[tok * rows + r] = dot_split_pairs(&ws, xt);
        }
    }
    y
}
