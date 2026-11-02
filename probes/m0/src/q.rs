//! ggml's Q4_K and Q8_0 block formats: CPU dequantization and matvec, the reference the GPU kernels are checked against.
use crate::rng::Rng;
use half::f16;

pub const QK_K: usize = 256;
/// f16 d, f16 dmin, 12 bytes of packed 6-bit scales and mins, 128 bytes of 4-bit quants.
pub const Q4K_BLOCK_BYTES: usize = 144;
/// f16 d, 32 int8 quants.
pub const Q8_0_BLOCK_BYTES: usize = 34;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
pub enum Quant {
    #[value(name = "q4k")]
    #[serde(rename = "q4k")]
    Q4k,
    #[value(name = "q8_0")]
    #[serde(rename = "q8_0")]
    Q8_0,
}

impl Quant {
    pub fn row_bytes(self, k: usize) -> usize {
        match self {
            Quant::Q4k => k / QK_K * Q4K_BLOCK_BYTES,
            Quant::Q8_0 => k / 32 * Q8_0_BLOCK_BYTES,
        }
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

pub fn dequant_q4k_block(b: &[u8], out: &mut [f32]) {
    let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
    let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
    let scales = &b[4..16];
    let qs = &b[16..144];
    for c in 0..4 {
        let (sc1, mn1) = scale_min_k4(2 * c, scales);
        let (sc2, mn2) = scale_min_k4(2 * c + 1, scales);
        let (d1, m1) = (d * sc1 as f32, dmin * mn1 as f32);
        let (d2, m2) = (d * sc2 as f32, dmin * mn2 as f32);
        let (lo, hi) = out[64 * c..64 * c + 64].split_at_mut(32);
        for ((a, h), &byte) in lo
            .iter_mut()
            .zip(hi.iter_mut())
            .zip(&qs[32 * c..32 * c + 32])
        {
            *a = d1 * (byte & 0xF) as f32 - m1;
            *h = d2 * (byte >> 4) as f32 - m2;
        }
    }
}

pub fn dequant_q8_0_block(b: &[u8], out: &mut [f32]) {
    let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
    for (o, &q) in out.iter_mut().zip(&b[2..34]) {
        *o = d * (q as i8) as f32;
    }
}

pub fn dequant_row(quant: Quant, row: &[u8], k: usize) -> Vec<f32> {
    let mut out = vec![0f32; k];
    match quant {
        Quant::Q4k => {
            for (b, o) in row
                .as_chunks::<Q4K_BLOCK_BYTES>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<QK_K>().0)
            {
                dequant_q4k_block(b, o);
            }
        }
        Quant::Q8_0 => {
            for (b, o) in row
                .as_chunks::<Q8_0_BLOCK_BYTES>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<32>().0)
            {
                dequant_q8_0_block(b, o);
            }
        }
    }
    out
}

/// y = W·x, with W stored as `rows` quantized rows of length x.len(). Accumulates in f64, so it is the
/// more accurate side of any comparison.
pub fn matvec(quant: Quant, w: &[u8], x: &[f32], rows: usize) -> Vec<f32> {
    let rb = quant.row_bytes(x.len());
    (0..rows)
        .map(|r| {
            let row = dequant_row(quant, &w[r * rb..(r + 1) * rb], x.len());
            row.iter()
                .zip(x)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

/// Random but well-formed weights: f16 scales in [0.001, 0.02), any scale-table and quant bytes.
pub fn random_weights(quant: Quant, rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut w = vec![0u8; rows * quant.row_bytes(k)];
    rng.fill_bytes(&mut w);
    let (block, f16_fields) = match quant {
        Quant::Q4k => (Q4K_BLOCK_BYTES, 2),
        Quant::Q8_0 => (Q8_0_BLOCK_BYTES, 1),
    };
    for b in w.chunks_exact_mut(block) {
        for f in 0..f16_fields {
            let s = f16::from_f32(0.001 + 0.019 * (rng.next_u32() as f32 / u32::MAX as f32));
            b[2 * f..2 * f + 2].copy_from_slice(&s.to_le_bytes());
        }
    }
    w
}

pub fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding or invalid bit patterns, and u8 has alignment 1.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
