//! The scalar expert: `twin::matvec::row_dot`'s operations, with Σx per unit computed once per call (it is the same
//! for every row). The reference the AVX-512 version is timed against.
use kernels::dispatch::{SlotLayout, WType};
use kernels::twin::silu_det::silu_det;

#[inline(always)]
fn f16_at(b: &[u8], i: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([b[i], b[i + 1]])).to_f32()
}

/// Σx over each 32-value unit, in order from 0.
pub fn unit_sums(x: &[f32]) -> Vec<f32> {
    x.as_chunks::<32>()
        .0
        .iter()
        .map(|c| c.iter().fold(0f32, |s, v| s + v))
        .collect()
}

#[inline(always)]
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

/// `unit_dot`, with the unit's Σx given.
#[inline(always)]
pub fn unit_dot(t: WType, row: &[u8], u: usize, x: &[f32], sx: f32) -> f32 {
    let x: &[f32; 32] = x[32 * u..32 * u + 32].try_into().unwrap();
    match t {
        WType::Q8_0 => {
            let b = &row[34 * u..34 * u + 34];
            let mut s = 0f32;
            for i in 0..32 {
                s += (b[2 + i] as i8) as f32 * x[i];
            }
            f16_at(b, 0) * s
        }
        WType::Q4K | WType::Q5K => {
            let q5 = t == WType::Q5K;
            let bsz = if q5 { 176 } else { 144 };
            let b = &row[bsz * (u / 8)..bsz * (u / 8) + bsz];
            let (j, c) = (u % 8, (u % 8) / 2);
            let sh = 4 * (j % 2);
            let (sc, mn) = scale_min_k4(&b[4..16], j);
            let qs: &[u8; 32] = b[if q5 { 48 } else { 16 } + 32 * c..][..32]
                .try_into()
                .unwrap();
            let mut sq = 0f32;
            if q5 {
                let qh: &[u8; 32] = b[16..48].try_into().unwrap();
                for l in 0..32 {
                    let q = ((qs[l] >> sh) & 0xF) | (((qh[l] >> j) & 1) << 4);
                    sq += q as f32 * x[l];
                }
            } else {
                for l in 0..32 {
                    sq += ((qs[l] >> sh) & 0xF) as f32 * x[l];
                }
            }
            (f16_at(b, 0) * sc as f32) * sq - (f16_at(b, 2) * mn as f32) * sx
        }
        WType::Q6K => {
            let b = &row[210 * (u / 8)..210 * (u / 8) + 210];
            let j = u % 8;
            let (n, m) = (j / 4, j % 4);
            let ql: &[u8; 32] = b[64 * n + 32 * (m & 1)..][..32].try_into().unwrap();
            let qh: &[u8; 32] = b[128 + 32 * n..][..32].try_into().unwrap();
            let (shl, shh) = (4 * (m >> 1), 2 * m);
            let (mut sa, mut sb) = (0f32, 0f32);
            for l in 0..16 {
                let v = ((ql[l] >> shl) & 0xF) | (((qh[l] >> shh) & 3) << 4);
                sa += (v as i32 - 32) as f32 * x[l];
            }
            for l in 16..32 {
                let v = ((ql[l] >> shl) & 0xF) | (((qh[l] >> shh) & 3) << 4);
                sb += (v as i32 - 32) as f32 * x[l];
            }
            let sca = (b[192 + 8 * n + 2 * m] as i8) as f32;
            let scb = (b[192 + 8 * n + 2 * m + 1] as i8) as f32;
            f16_at(b, 208) * (sca * sa + scb * sb)
        }
        _ => unimplemented!("F32/BF16 expert matrices are not supported"),
    }
}

/// `row_dot`: 32 lane partials (lane L adds units L, L + 32, … from 0), then the 32-lane tree.
#[inline]
pub fn row_dot(t: WType, row: &[u8], x: &[f32], sx: &[f32]) -> f32 {
    let mut p = [0f32; 32];
    for u in 0..x.len() / 32 {
        p[u % 32] += unit_dot(t, row, u, x, sx[u]);
    }
    tree32(&mut p)
}

#[inline(always)]
pub fn tree32(p: &mut [f32; 32]) -> f32 {
    let mut s = 16;
    while s > 0 {
        for i in 0..s {
            p[i] += p[i + s];
        }
        s /= 2;
    }
    p[0]
}

/// Rows `rows` of the matrix `m` (`k` columns) into `out[rows]`.
pub fn mat_rows(
    t: WType,
    m: &[u8],
    k: usize,
    x: &[f32],
    sx: &[f32],
    rows: std::ops::Range<usize>,
    out: &mut [f32],
) {
    let rb = t.row_bytes(k);
    for r in rows {
        out[r] = row_dot(t, &m[r * rb..(r + 1) * rb], x, sx);
    }
}

/// One expert: act[r] = silu_det(gate_r·x)·(up_r·x), y = down·act.
pub fn expert(slot: &[u8], lay: &SlotLayout, x: &[f32], act: &mut [f32], y: &mut [f32]) {
    let sx = unit_sums(x);
    let (rg, ru, rd) = (
        lay.gate.row_bytes(lay.n_embd),
        lay.up.row_bytes(lay.n_embd),
        lay.down.row_bytes(lay.n_ff),
    );
    let up = &slot[lay.up_offset()..];
    for r in 0..lay.n_ff {
        let g = row_dot(lay.gate, &slot[r * rg..(r + 1) * rg], x, &sx);
        let u = row_dot(lay.up, &up[r * ru..(r + 1) * ru], x, &sx);
        act[r] = silu_det(g) * u;
    }
    let sa = unit_sums(act);
    let down = &slot[lay.down_offset()..];
    for r in 0..lay.n_embd {
        y[r] = row_dot(lay.down, &down[r * rd..(r + 1) * rd], act, &sa);
    }
}

/// The twin-based reference: `twin::matvec::row_dot` for every dot product, `silu_det` for the activation. (The row
/// loops index act/y and compute their row offsets from the index; an iterator rewrite obscures it.)
#[allow(clippy::needless_range_loop)]
pub fn expert_twin(slot: &[u8], lay: &SlotLayout, x: &[f32], act: &mut [f32], y: &mut [f32]) {
    use kernels::twin::matvec::row_dot;
    let (rg, ru, rd) = (
        lay.gate.row_bytes(lay.n_embd),
        lay.up.row_bytes(lay.n_embd),
        lay.down.row_bytes(lay.n_ff),
    );
    for r in 0..lay.n_ff {
        let g = row_dot(lay.gate, &slot[r * rg..(r + 1) * rg], x);
        let at = lay.up_offset() + r * ru;
        act[r] = silu_det(g) * row_dot(lay.up, &slot[at..at + ru], x);
    }
    for r in 0..lay.n_embd {
        let at = lay.down_offset() + r * rd;
        y[r] = row_dot(lay.down, &slot[at..at + rd], act);
    }
}
