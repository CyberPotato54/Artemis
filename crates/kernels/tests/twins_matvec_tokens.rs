//! CI's half of `matvec_tokens`: its twin is `matvec` once a token, bit for bit, because the kernel
//! gives every token decode's matvec arithmetic. The GPU half is `gpu_matvec_tokens.rs`.
use kernels::{testing::random_weights, twin::matvec as mv};
use qwen36::{quant::WType, testing::Rng};

#[test]
fn the_tokens_twin_is_matvec_per_token() {
    let mut rng = Rng::new(41);
    for ty in WType::ALL {
        for (rows, cols) in [(37, 512), (16, 2048), (5, 256)] {
            let w = random_weights(&mut rng, ty, rows, cols);
            for tokens in [1, 3, 16] {
                let x: Vec<f32> = (0..tokens * cols).map(|_| rng.uniform(-1.0, 1.0)).collect();
                let got = mv::matvec_tokens(ty, &w, rows, &x, tokens);
                assert_eq!(got.len(), tokens * rows, "{} {rows}x{cols}", ty.name());
                for (i, xs) in x.chunks_exact(cols).enumerate() {
                    let want = mv::matvec(ty, &w, rows, xs);
                    let differ = got[i * rows..(i + 1) * rows]
                        .iter()
                        .zip(&want)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    assert_eq!(
                        differ,
                        0,
                        "{} {rows}x{cols}, {tokens} tokens: token {i} differs from matvec in {differ} rows",
                        ty.name()
                    );
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "are not 3 whole vectors")]
fn the_tokens_twin_refuses_inputs_that_are_not_whole_vectors() {
    let mut rng = Rng::new(42);
    let w = random_weights(&mut rng, WType::F32, 4, 64);
    mv::matvec_tokens(WType::F32, &w, 4, &vec![0.5f32; 3 * 64 + 1], 3);
}

#[test]
#[should_panic(expected = "0 input values are not 3 whole vectors")]
fn the_tokens_twin_refuses_an_empty_input() {
    let mut rng = Rng::new(43);
    let w = random_weights(&mut rng, WType::F32, 4, 64);
    mv::matvec_tokens(WType::F32, &w, 4, &[], 3);
}

#[test]
fn the_push_block_is_the_shaders() {
    // matvec_tokens.glsl: three u64 addresses, then rows, k, tokens, x_stride, y_stride and a pad.
    assert_eq!(
        std::mem::size_of::<kernels::tokens::matvec::MatvecTokensPush>(),
        48
    );
}
