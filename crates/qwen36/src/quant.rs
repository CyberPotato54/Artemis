//! ggml's block formats for the six weight types Artemis accepts: dequantization and f64 dot
//! products. Checked against gguf-py on real rows of the model file (tests/quant_fixtures.rs).
use gguf::types;
use half::{bf16, f16};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WType {
    F32,
    Bf16,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

impl WType {
    pub const ALL: [WType; 6] = [
        WType::F32,
        WType::Bf16,
        WType::Q8_0,
        WType::Q4K,
        WType::Q5K,
        WType::Q6K,
    ];

    pub fn from_ggml(t: u32) -> Option<WType> {
        WType::ALL.into_iter().find(|w| w.ggml() == t)
    }

    pub fn ggml(self) -> u32 {
        match self {
            WType::F32 => types::F32,
            WType::Bf16 => types::BF16,
            WType::Q8_0 => types::Q8_0,
            WType::Q4K => types::Q4_K,
            WType::Q5K => types::Q5_K,
            WType::Q6K => types::Q6_K,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            WType::F32 => "F32",
            WType::Bf16 => "BF16",
            WType::Q8_0 => "Q8_0",
            WType::Q4K => "Q4_K",
            WType::Q5K => "Q5_K",
            WType::Q6K => "Q6_K",
        }
    }

    /// (elements, bytes) per block.
    pub fn block(self) -> (usize, usize) {
        match self {
            WType::F32 => (1, 4),
            WType::Bf16 => (1, 2),
            WType::Q8_0 => (32, 34),
            WType::Q4K => (256, 144),
            WType::Q5K => (256, 176),
            WType::Q6K => (256, 210),
        }
    }

    /// Bytes in one row of `k` values. `k` must be a whole number of blocks.
    pub fn row_bytes(self, k: usize) -> usize {
        let (per, bytes) = self.block();
        assert!(
            k.is_multiple_of(per),
            "{} rows hold whole {per}-value blocks, not {k} values",
            self.name()
        );
        k / per * bytes
    }
}

/// Dequantizes one row; `out.len()` is the row's length in values.
pub fn dequant_row(t: WType, row: &[u8], out: &mut [f32]) {
    assert_eq!(
        row.len(),
        t.row_bytes(out.len()),
        "{} row of {} values",
        t.name(),
        out.len()
    );
    match t {
        WType::F32 => {
            for (o, b) in out.iter_mut().zip(row.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*b);
            }
        }
        WType::Bf16 => {
            for (o, b) in out.iter_mut().zip(row.as_chunks::<2>().0) {
                *o = bf16::from_le_bytes(*b).to_f32();
            }
        }
        WType::Q8_0 => blocks::<34, 32>(row, out, q8_0_block),
        WType::Q4K => blocks::<144, 256>(row, out, q4k_block),
        WType::Q5K => blocks::<176, 256>(row, out, q5k_block),
        WType::Q6K => blocks::<210, 256>(row, out, q6k_block),
    }
}

/// Σ row·x in f64, dequantizing into `scratch`. The more accurate side of every comparison.
pub fn dot_row(t: WType, row: &[u8], x: &[f32], scratch: &mut Vec<f32>) -> f64 {
    scratch.resize(x.len(), 0.0);
    dequant_row(t, row, scratch);
    scratch
        .iter()
        .zip(x)
        .map(|(w, v)| *w as f64 * *v as f64)
        .sum()
}

/// Rounds activations the way llama.cpp's CPU backend does before multiplying them by weights of type `t` (semantics
/// reference §7): Q8_0 blocks for Q8_0 weights, Q8_K blocks for K-quants, BF16 for BF16, untouched for F32. Only the
/// reference's `Rounding::LlamaCpu` mode uses it, so test 2 compares at llama.cpp's own precision.
pub fn round_like_llama_cpu(t: WType, x: &[f32]) -> Vec<f32> {
    match t {
        WType::F32 => x.to_vec(),
        WType::Bf16 => x.iter().map(|&v| bf16::from_f32(v).to_f32()).collect(),
        // ggml's AVX2 quantize_row_q8_0: d = amax/127 stored as f16, q = round-half-even(x · 127/amax).
        WType::Q8_0 => x
            .chunks(32)
            .flat_map(|b| {
                let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
                let d = f16::from_f32(amax / 127.0).to_f32();
                b.iter().map(move |&v| (v * id).round_ties_even() * d)
            })
            .collect(),
        // ggml's quantize_row_q8_K_ref: the element of largest magnitude (first one wins) sets iscale = -127/max;
        // q = min(127, round-half-even(iscale · x)); d = 1/iscale, kept as f32.
        WType::Q4K | WType::Q5K | WType::Q6K => x
            .chunks(256)
            .flat_map(|b| {
                let (mut amax, mut max) = (0f32, 0f32);
                for &v in b {
                    if v.abs() > amax {
                        amax = v.abs();
                        max = v;
                    }
                }
                let iscale = if amax != 0.0 { -127.0 / max } else { 0.0 };
                let d = if amax != 0.0 { 1.0 / iscale } else { 0.0 };
                b.iter()
                    .map(move |&v| (iscale * v).round_ties_even().min(127.0) * d)
            })
            .collect(),
    }
}

fn blocks<const B: usize, const N: usize>(
    row: &[u8],
    out: &mut [f32],
    f: fn(&[u8; B], &mut [f32; N]),
) {
    for (b, o) in row
        .as_chunks::<B>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<N>().0)
    {
        f(b, o);
    }
}

fn half_at(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// f16 d, then 32 int8 quants.
fn q8_0_block(b: &[u8; 34], out: &mut [f32; 32]) {
    let d = half_at(b, 0);
    for (o, &q) in out.iter_mut().zip(&b[2..]) {
        *o = d * (q as i8) as f32;
    }
}

/// Scale and min for sub-block `j` (0..8) from the 12 packed bytes (ggml's get_scale_min_k4).
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// f16 d, f16 dmin, 12 bytes of packed 6-bit scales and mins, 128 bytes of nibbles.
fn q4k_block(b: &[u8; 144], out: &mut [f32; 256]) {
    let (d, dmin) = (half_at(b, 0), half_at(b, 2));
    let (scales, qs) = (&b[4..16], &b[16..144]);
    for c in 0..4 {
        let (sc1, m1) = scale_min_k4(2 * c, scales);
        let (sc2, m2) = scale_min_k4(2 * c + 1, scales);
        let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
        let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
        for l in 0..32 {
            let q = qs[32 * c + l];
            out[64 * c + l] = d1 * (q & 0xF) as f32 - mm1;
            out[64 * c + 32 + l] = d2 * (q >> 4) as f32 - mm2;
        }
    }
}

/// f16 d, f16 dmin, 12 bytes of scales and mins, 32 bytes of high bits, 128 bytes of low nibbles.
fn q5k_block(b: &[u8; 176], out: &mut [f32; 256]) {
    let (d, dmin) = (half_at(b, 0), half_at(b, 2));
    let (scales, qh, qs) = (&b[4..16], &b[16..48], &b[48..176]);
    for c in 0..4 {
        let (sc1, m1) = scale_min_k4(2 * c, scales);
        let (sc2, m2) = scale_min_k4(2 * c + 1, scales);
        let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
        let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
        let (u1, u2) = (1u8 << (2 * c), 2u8 << (2 * c));
        for l in 0..32 {
            let q = qs[32 * c + l];
            let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
            let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
            out[64 * c + l] = d1 * ((q & 0xF) + h1) as f32 - mm1;
            out[64 * c + 32 + l] = d2 * ((q >> 4) + h2) as f32 - mm2;
        }
    }
}

/// 128 bytes of low nibbles, 64 bytes of high 2-bit pairs, 16 int8 scales, f16 d.
fn q6k_block(b: &[u8; 210], out: &mut [f32; 256]) {
    let d = half_at(b, 208);
    for n in 0..2 {
        let (ql, qh, sc) = (
            &b[64 * n..64 * n + 64],
            &b[128 + 32 * n..128 + 32 * n + 32],
            &b[192 + 8 * n..192 + 8 * n + 8],
        );
        let y = &mut out[128 * n..128 * n + 128];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_bytes_per_type() {
        assert_eq!(WType::F32.row_bytes(4), 16);
        assert_eq!(WType::Bf16.row_bytes(3), 6);
        assert_eq!(WType::Q8_0.row_bytes(2048), 2176);
        assert_eq!(WType::Q4K.row_bytes(2048), 1152);
        assert_eq!(WType::Q5K.row_bytes(512), 352);
        assert_eq!(WType::Q6K.row_bytes(512), 420);
    }

    #[test]
    #[should_panic(expected = "whole 256-value blocks")]
    fn partial_k_quant_row_panics() {
        WType::Q4K.row_bytes(100);
    }

    #[test]
    fn ggml_ids_round_trip() {
        for w in WType::ALL {
            assert_eq!(WType::from_ggml(w.ggml()), Some(w));
        }
        assert_eq!(WType::from_ggml(2), None); // Q4_0 is not accepted
    }

    #[test]
    fn q8_0_hand_block() {
        let mut b = [0u8; 34];
        b[0..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        for (i, q) in b[2..].iter_mut().enumerate() {
            *q = (i as i8 - 16) as u8;
        }
        let mut out = [0f32; 32];
        dequant_row(WType::Q8_0, &b, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 0.5 * (i as f32 - 16.0));
        }
    }

    #[test]
    fn q5k_hand_block() {
        // d = 1, dmin = 0, every scale 1 and min 0, low nibbles 1 and 2, every high bit set: 17 and 18.
        let mut b = [0u8; 176];
        b[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        b[4..16].copy_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
        b[16..48].fill(0xFF);
        b[48..176].fill(0x21);
        let mut out = [0f32; 256];
        dequant_row(WType::Q5K, &b, &mut out);
        for c in 0..4 {
            assert!(
                out[64 * c..64 * c + 32].iter().all(|&v| v == 17.0),
                "chunk {c} low"
            );
            assert!(
                out[64 * c + 32..64 * c + 64].iter().all(|&v| v == 18.0),
                "chunk {c} high"
            );
        }
    }

    #[test]
    fn q6k_hand_block_extremes() {
        let mut b = [0u8; 210];
        b[192..208].fill(1);
        b[208..210].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        let mut out = [0f32; 256];
        dequant_row(WType::Q6K, &b, &mut out);
        assert!(out.iter().all(|&v| v == -32.0));
        b[0..192].fill(0xFF);
        dequant_row(WType::Q6K, &b, &mut out);
        assert!(out.iter().all(|&v| v == 31.0));
    }

    #[test]
    fn llama_rounding_q8_0_matches_ggml() {
        // amax 1: id = 127, d = f16(1/127); q = round-half-even(x * 127).
        let mut x = vec![0f32; 64];
        x[0] = 1.0;
        x[1] = 0.5;
        x[2] = -0.25;
        x[3] = 0.003_937; // just under half a step: rounds to 0
        let out = round_like_llama_cpu(WType::Q8_0, &x);
        let d = f16::from_f32(1.0 / 127.0).to_f32();
        assert_eq!(out[0], 127.0 * d);
        assert_eq!(out[1], 64.0 * d); // 63.5 rounds to 64
        assert_eq!(out[2], -32.0 * d); // -31.75 rounds to -32
        assert_eq!(out[3], 0.0 * d);
        assert!(
            out[32..].iter().all(|&v| v == 0.0),
            "an all-zero block stays zero"
        );
    }

    #[test]
    fn llama_rounding_q8_k_uses_the_signed_max() {
        // The largest magnitude is negative: iscale = -127 / -2 = 63.5, d = 1 / 63.5 (f32, not f16).
        let mut x = vec![0f32; 512];
        x[0] = -2.0;
        x[1] = 1.0;
        // Positive max in the second block: iscale = -63.5, so 1.0 maps to -64 -> clamped side is +127 only.
        x[256] = 2.0;
        x[257] = -1.0;
        let out = round_like_llama_cpu(WType::Q4K, &x);
        let d = 1.0f32 / 63.5;
        assert_eq!(out[0], -127.0 * d);
        assert_eq!(out[1], 64.0 * d); // 63.5 is a tie: half-even gives 64
        let d2 = 1.0f32 / -63.5;
        assert_eq!(out[256], -127.0 * d2);
        assert_eq!(out[257], 64.0 * d2);
        assert_eq!(
            round_like_llama_cpu(WType::Q6K, &x),
            out,
            "every K-quant uses Q8_K"
        );
    }

    #[test]
    fn llama_rounding_bf16_and_f32() {
        let x = [1.0f32 + 1.0 / 512.0, 3.0];
        assert_eq!(round_like_llama_cpu(WType::Bf16, &x), [1.0, 3.0]);
        assert_eq!(round_like_llama_cpu(WType::F32, &x), x);
    }

    #[test]
    fn bf16_and_f32_rows() {
        let mut out = [0f32; 2];
        dequant_row(WType::Bf16, &[0x80, 0x3F, 0x00, 0xC0], &mut out);
        assert_eq!(out, [1.0, -2.0]);
        let raw: Vec<u8> = [1.5f32, -0.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        dequant_row(WType::F32, &raw, &mut out);
        assert_eq!(out, [1.5, -0.25]);
    }
}
