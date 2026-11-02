//! CI's half of the token attention's test 1: the twin of `attn_decode_tokens` and
//! `attn_combine_tokens` is decode's attention applied to each query at its own context length, bit for bit.
use kernels::twin::{attention as at, f16_bits};
use qwen36::{dims::Dims, testing::Rng};

#[test]
fn the_tokens_twin_is_attention_per_query() {
    let mut rng = Rng::new(61);
    // The real model's shapes at the short contexts; the tiny model's (heads of 64 values) at 3,070, where the twin's
    // cost in a debug build would otherwise dominate CI.
    let cases = [
        (Dims::QWEN36, 1),
        (Dims::QWEN36, 255),
        (Dims::tiny(), 1),
        (Dims::tiny(), 255),
        (Dims::tiny(), 3070),
    ];
    for (d, n_ctx0) in cases {
        let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
        for tokens in [1usize, 5, 16] {
            let q: Vec<f32> = (0..tokens * nh * hd)
                .map(|_| rng.uniform(-12.0, 12.0))
                .collect();
            let n = n_ctx0 + tokens - 1;
            let kc: Vec<u16> = (0..n * nkv * hd)
                .map(|_| f16_bits(rng.uniform(-1.0, 1.0)))
                .collect();
            let vc: Vec<u16> = (0..n * nkv * hd)
                .map(|_| f16_bits(rng.uniform(-1.0, 1.0)))
                .collect();
            let got = at::attention_tokens(&q, &kc, &vc, n_ctx0, tokens, nh, nkv, hd);
            assert_eq!(got.len(), tokens * nh * hd);
            for i in 0..tokens {
                let want = at::attention(
                    &q[i * nh * hd..(i + 1) * nh * hd],
                    &kc,
                    &vc,
                    n_ctx0 + i,
                    nh,
                    nkv,
                    hd,
                );
                let row = &got[i * nh * hd..(i + 1) * nh * hd];
                assert!(
                    row.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "hd={hd} n_ctx0={n_ctx0} tokens={tokens}: query {i} differs from attention() at {}",
                    n_ctx0 + i
                );
            }
            // A query sees only its own prefix: the same query against a longer cache gives the same row.
            let longer = [kc.clone(), vec![f16_bits(3.0); nkv * hd]].concat();
            let longer_v = [vc.clone(), vec![f16_bits(3.0); nkv * hd]].concat();
            let again = at::attention_tokens(&q, &longer, &longer_v, n_ctx0, tokens, nh, nkv, hd);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "hd={hd} n_ctx0={n_ctx0} tokens={tokens}: a position past the last query's changed an output"
            );
        }
    }
}

#[test]
#[should_panic(expected = "attention_tokens: the cache holds")]
fn the_tokens_twin_refuses_a_cache_shorter_than_its_last_query() {
    let d = Dims::tiny();
    let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
    let q = vec![0.5f32; 2 * nh * hd];
    let kc = vec![0u16; 2 * nkv * hd];
    let _ = at::attention_tokens(&q, &kc, &kc, 2, 2, nh, nkv, hd);
}
