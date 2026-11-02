//! Dequantization must match gguf-py on one real row of each weight type in the model file.
use qwen36::quant::{dequant_row, dot_row, WType};

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn check(t: WType, raw: &[u8], want: &[u8]) {
    let want = f32s(want);
    assert_eq!(
        raw.len(),
        t.row_bytes(want.len()),
        "{} fixture size",
        t.name()
    );
    let mut got = vec![0f32; want.len()];
    dequant_row(t, raw, &mut got);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-6 * w.abs().max(1.0),
            "{} element {i}: got {g}, gguf-py {w}",
            t.name()
        );
    }
    // dot_row against ones is the f64 sum of the dequantized row.
    let ones = vec![1f32; want.len()];
    let sum: f64 = want.iter().map(|&v| v as f64).sum();
    let dot = dot_row(t, raw, &ones, &mut Vec::new());
    assert!(
        (dot - sum).abs() <= 1e-6 * sum.abs().max(1.0),
        "{} dot {dot} vs {sum}",
        t.name()
    );
}

macro_rules! fixture {
    ($test:ident, $t:expr, $stem:literal) => {
        #[test]
        fn $test() {
            check(
                $t,
                include_bytes!(concat!("fixtures/", $stem, ".bin")),
                include_bytes!(concat!("fixtures/", $stem, ".f32")),
            );
        }
    };
}

fixture!(f32_matches_gguf_py, WType::F32, "f32_row");
fixture!(bf16_matches_gguf_py, WType::Bf16, "bf16_row");
fixture!(q8_0_matches_gguf_py, WType::Q8_0, "q8_0_row");
fixture!(q4_k_matches_gguf_py, WType::Q4K, "q4_k_row");
fixture!(q5_k_matches_gguf_py, WType::Q5K, "q5_k_row");
fixture!(q6_k_matches_gguf_py, WType::Q6K, "q6_k_row");
