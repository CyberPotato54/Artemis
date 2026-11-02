//! The CPU dequantization must match gguf-py on real rows of the model file.
use m0::q::{dequant_row, random_weights, Quant};

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn check(quant: Quant, raw: &[u8], want: &[u8]) {
    let want = f32s(want);
    assert_eq!(raw.len(), quant.row_bytes(want.len()));
    let got = dequant_row(quant, raw, want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-6 * w.abs().max(1.0),
            "element {i}: got {g}, gguf-py {w}"
        );
    }
}

#[test]
fn q4k_matches_gguf_py() {
    check(
        Quant::Q4k,
        include_bytes!("fixtures/q4k_row.bin"),
        include_bytes!("fixtures/q4k_row.f32"),
    );
}

#[test]
fn q8_0_matches_gguf_py() {
    check(
        Quant::Q8_0,
        include_bytes!("fixtures/q8_0_row.bin"),
        include_bytes!("fixtures/q8_0_row.f32"),
    );
}

#[test]
fn random_weights_dequantize_to_finite_values() {
    for quant in [Quant::Q4k, Quant::Q8_0] {
        let w = random_weights(quant, 4, 2048, 1);
        for r in 0..4 {
            let row = &w[r * quant.row_bytes(2048)..(r + 1) * quant.row_bytes(2048)];
            assert!(dequant_row(quant, row, 2048).iter().all(|v| v.is_finite()));
        }
    }
}

#[test]
fn row_sizes_match_the_model_contract() {
    assert_eq!(Quant::Q4k.row_bytes(2048), 1152); // 512 rows of this = one expert's gate (589,824 bytes)
    assert_eq!(Quant::Q8_0.row_bytes(2048), 2176);
}
