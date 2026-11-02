//! Random expert slots for the bit-exactness tests: well-formed blocks (the kernels' test
//! generator) or raw random bytes (any f16 scale, including huge, tiny, denormal and non-finite
//! ones), and N(0, 1) x vectors. The GGUF half lives in the `#[ignore]`d real-expert test, behind a
//! dev-dependency.
use kernels::dispatch::{SlotLayout, WType};
use kernels::testing::Rng;

/// A random slot: well-formed blocks (kernels::testing::random_weights), or raw random bytes when `raw` (any f16
/// scale, including huge, tiny, denormal and non-finite ones).
pub fn random_slot(rng: &mut Rng, lay: &SlotLayout, raw: bool) -> Vec<u8> {
    let mut gen = |t: WType, rows: usize, cols: usize| {
        if raw {
            let mut v = vec![0u8; rows * t.row_bytes(cols)];
            rng.fill(&mut v);
            v
        } else {
            kernels::testing::random_weights(rng, t, rows, cols)
        }
    };
    let mut s = gen(lay.gate, lay.n_ff, lay.n_embd);
    s.extend(gen(lay.up, lay.n_ff, lay.n_embd));
    s.extend(gen(lay.down, lay.n_embd, lay.n_ff));
    assert_eq!(s.len(), lay.bytes());
    s
}

/// N(0, 1) samples (Box-Muller over the splitmix generator).
pub fn normal(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let u1 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            ((-2.0 * (u1.max(1e-300)).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        })
        .collect()
}
