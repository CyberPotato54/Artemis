//! The cooperative-matrix prefill attention's speed check against today's f32 kernel, on the real model's shape (16
//! query heads, 2 KV heads, head dimension 256): `attn_prefill` and `attn_prefill_cm` over the same inputs, for four
//! (position, queries) cases. A measurement, not a test (the runner's test-1 block runs every ignored test):
//! `cargo run --release -p kernels --example attn_speed`. Each figure is the best of five submits of the kernel alone,
//! inputs uploaded once, timed on the host around submit and wait, and a pair is one query-key pair over all heads, as
//! `engine::prefill::ATTN_PAIRS` counts them: t·pos + t·(t + 1)/2. `ATTN_PAIRS_CM` is set from the 25,600 line.
use kernels::{
    testing::{rand_f32, Gpu},
    twin, Kernels,
};
use qwen36::{dims::Dims, testing::Rng};
use std::time::Instant;

/// The best of five submits of `f`, in ms.
fn best(g: &Gpu, f: impl Fn(&Kernels, &mut vk::Recorder)) -> f64 {
    (0..5)
        .map(|_| {
            let t0 = Instant::now();
            g.run(&f);
            t0.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

/// Random f16 caches of `n_ctx` positions of `width` values, as bytes.
fn random_cache(rng: &mut Rng, n_ctx: usize, width: usize) -> Vec<u8> {
    (0..n_ctx * width)
        .flat_map(|_| twin::f16_bits(rng.uniform(-1.0, 1.0)).to_le_bytes())
        .collect()
}

fn main() {
    let g = Gpu::new();
    assert!(g.k.coop(), "this device runs no cooperative-matrix kernels");
    let d = Dims::QWEN36;
    let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
    let mut rng = Rng::new(51);
    for (pos, t) in [(0, 2048), (12_288, 2048), (25_600, 2048), (26_624, 792)] {
        let q = g.f32s(&rand_f32(&mut rng, t * nh * hd, -12.0, 12.0));
        let kc = g.bytes(&random_cache(&mut rng, pos + t, nkv * hd));
        let vc = g.bytes(&random_cache(&mut rng, pos + t, nkv * hd));
        let y = g.zeros(t * nh * hd);
        let pairs = (t * pos + t * (t + 1) / 2) as f64;
        let f = best(&g, |k, r| {
            k.attn_prefill(r, q.addr, kc.addr, vc.addr, y.addr, pos, t, nh, nkv, hd)
        });
        let c = best(&g, |k, r| {
            k.attn_prefill_cm(r, q.addr, kc.addr, vc.addr, y.addr, pos, t, nh, nkv, hd, 0)
        });
        println!(
            "{t} queries from {pos}: attn_prefill {f:.2} ms ({:.2} ns a pair), attn_prefill_cm {c:.2} ms ({:.2} ns a pair), {:.2}x",
            f * 1e6 / pairs,
            c * 1e6 / pairs,
            f / c
        );
    }
}
