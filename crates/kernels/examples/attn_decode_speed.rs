//! Decode attention on the GPU clock, on the real model's shape (16 query heads, 2 KV heads, head dimension 256), at
//! 4,096, 27,000 and 65,536 cached positions. A measurement, not a test:
//! `cargo run --release -p kernels --example attn_decode_speed -- --json FILE [--reps 1000]`.
//!
//! Method for a figure worth quoting: run it twice (the default is 1,000 repetitions) and compare the two runs' best
//! totals. If every one agrees within 1%, keep the second run. If not, run again, up to four runs in all, and keep the
//! second run of the closest pair.
//!
//! Per depth, two cases: one layer's call (`attn_decode_gqa`, a barrier, `attn_combine`), and the verify form at
//! t = 4 (`attn_decode_gqa_tokens`, a barrier, `attn_combine_tokens`), whose last query attends `n_ctx` positions
//! (the first attends `n_ctx - 3`), so both forms split their partials into the same number of chunks. Each dispatch
//! is timed by the device's timestamps (`Gpu::run_timed`). Each repetition's command buffer starts with a 256 MiB
//! buffer copy and a barrier, which evict the GPU caches as a decode step's weight reads do; the copy's own time is
//! left out, and putting it in the same buffer leaves the GPU no idle gap between it and the attention. Inputs are
//! seeded (`Rng::new(61)`): q in [-12, 12], K and V f16 in [-1, 1]. One untimed run comes first.
//!
//! The JSON holds `driver` and `reps`, then a `cases` array of `{kernel, n_ctx, t, gqa_us, combine_us, total_us,
//! kv_bytes, floor_us, bits}`. `gqa_us`, `combine_us` and `total_us` are `{best, median}` over the repetitions
//! (the median of an even count is the mean of the two middle values); `total_us` is taken over each repetition's
//! gqa + combine sum, which is not the sum of the two bests. `kv_bytes` is the K and V caches' size over `n_ctx`
//! positions (read once), `floor_us` that many bytes at 640 GB/s, and `bits` is the FNV-1a 64 of the outputs' f32 bits
//! (the partials, then `y`) as 16 hex digits. Every output is asserted finite, and the first and last repetitions'
//! outputs are asserted equal bit for bit.
use kernels::{
    dispatch::{attn_decode_kernel, attn_splits},
    testing::{f32_bytes, json_escaped, rand_f32, Gpu},
    twin, Kernel,
};
use qwen36::{dims::Dims, testing::fnv1a64, testing::Rng};
use std::fmt::Write as _;

/// The memory bandwidth the floor is taken at, in bytes per second.
const FLOOR_BYTES_PER_S: f64 = 640e9;
/// Verify-form queries.
const T: usize = 4;
const DEPTHS: [usize; 3] = [4_096, 27_000, 65_536];

/// Random f16 caches of `n_ctx` positions of `width` values, as bytes.
fn random_cache(rng: &mut Rng, n_ctx: usize, width: usize) -> Vec<u8> {
    (0..n_ctx * width)
        .flat_map(|_| twin::f16_bits(rng.uniform(-1.0, 1.0)).to_le_bytes())
        .collect()
}

fn best_and_median(v: &[f64]) -> (f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let median = if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    };
    (s[0], median)
}

/// Each repetition's gqa + combine sum.
fn totals(gqa: &[f64], combine: &[f64]) -> Vec<f64> {
    gqa.iter().zip(combine).map(|(a, b)| a + b).collect()
}

struct Case {
    kernel: &'static str,
    n_ctx: usize,
    t: usize,
    gqa: Vec<f64>,
    combine: Vec<f64>,
    kv_bytes: u64,
    bits: u64,
}

fn main() {
    let (mut json, mut reps) = (None, 1000usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--json" => json = Some(args.next().expect("--json needs a file")),
            "--reps" => {
                reps = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .filter(|&n| n >= 1)
                    .expect("--reps needs a number of at least 1")
            }
            other => panic!("unknown argument {other}; usage: --json FILE [--reps N]"),
        }
    }

    let g = Gpu::new();
    assert!(g.dev.can_profile(), "this device has no timestamps");
    let d = Dims::QWEN36;
    let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
    let evict_src = g.bytes(&vec![0x5a; 256 << 20]);
    let evict_dst = g.bytes(&vec![0; 256 << 20]);
    let mut rng = Rng::new(61);
    let mut cases = Vec::new();
    for n_ctx in DEPTHS {
        let kc = g.bytes(&random_cache(&mut rng, n_ctx, nkv * hd));
        let vc = g.bytes(&random_cache(&mut rng, n_ctx, nkv * hd));
        let kv_bytes = (2 * n_ctx * nkv * hd * 2) as u64;
        let n_split = attn_splits(n_ctx);
        for t in [1, T] {
            let q = g.f32s(&rand_f32(&mut rng, t * nh * hd, -12.0, 12.0));
            let part = g.zeros(t * nh * n_split * (hd + 2));
            let y = g.zeros(t * nh * hd);
            let n_ctx0 = n_ctx + 1 - t;
            assert_eq!(
                attn_decode_kernel(n_ctx0, nh, nkv, hd),
                Kernel::AttnDecodeGqa,
                "the GQA kernel is the one timed at {n_ctx0} positions"
            );
            // `evict` puts the cache-evicting copy and a barrier ahead of the attention in the same command buffer.
            let run = |evict: bool| {
                let evict_first = |r: &mut vk::Recorder| {
                    if evict {
                        r.copy(evict_src, evict_dst);
                        r.barrier();
                    }
                };
                if t == 1 {
                    g.run_timed(|k, r| {
                        evict_first(r);
                        k.attn_decode(r, q.addr, kc.addr, vc.addr, part.addr, n_ctx, nh, nkv, hd);
                        r.barrier();
                        k.attn_combine(r, part.addr, y.addr, n_ctx, nh, hd);
                    })
                } else {
                    g.run_timed(|k, r| {
                        evict_first(r);
                        k.attn_decode_tokens(
                            r, q.addr, kc.addr, vc.addr, part.addr, n_ctx0, t, nh, nkv, hd, n_split,
                        );
                        r.barrier();
                        k.attn_combine_tokens(r, part.addr, y.addr, n_ctx0, t, nh, hd, n_split);
                    })
                }
            };
            let outputs = || {
                let mut o = g.read(part);
                o.extend(g.read(y));
                assert!(
                    o.iter().all(|x| x.is_finite()),
                    "attn_decode_gqa at {n_ctx} positions, t = {t}: a non-finite output"
                );
                o
            };
            run(false);
            let first = outputs();
            let (mut gqa, mut combine) = (Vec::new(), Vec::new());
            for _ in 0..reps {
                let times = run(true);
                assert_eq!(times.len(), 2, "two dispatches, not {times:?}");
                gqa.push(times[0]);
                combine.push(times[1]);
            }
            let last = outputs();
            assert!(
                f32_bytes(&first) == f32_bytes(&last),
                "attn_decode_gqa at {n_ctx} positions, t = {t}: the outputs changed between repetitions"
            );
            let bits = fnv1a64(&f32_bytes(&last));
            let c = Case {
                kernel: if t == 1 {
                    "attn_decode_gqa"
                } else {
                    "attn_decode_gqa_tokens"
                },
                n_ctx,
                t,
                gqa,
                combine,
                kv_bytes,
                bits,
            };
            let (gb, gm) = best_and_median(&c.gqa);
            let (cb, cm) = best_and_median(&c.combine);
            let (tb, tm) = best_and_median(&totals(&c.gqa, &c.combine));
            println!(
                "{} n_ctx {n_ctx} t {t}: gqa best {gb:.1} median {gm:.1} us, combine best {cb:.1} median {cm:.1} us, \
                 total best {tb:.1} median {tm:.1} us, floor {:.1} us, bits {bits:016x}",
                c.kernel,
                kv_bytes as f64 / FLOOR_BYTES_PER_S * 1e6
            );
            cases.push(c);
        }
    }

    let mut out = format!(
        "{{\n  \"driver\": \"{}\",\n  \"reps\": {reps},\n  \"cases\": [\n",
        json_escaped(&g.dev.driver)
    );
    for (i, c) in cases.iter().enumerate() {
        let (gb, gm) = best_and_median(&c.gqa);
        let (cb, cm) = best_and_median(&c.combine);
        let (tb, tm) = best_and_median(&totals(&c.gqa, &c.combine));
        writeln!(
            out,
            "    {{\"kernel\": \"{}\", \"n_ctx\": {}, \"t\": {}, \
             \"gqa_us\": {{\"best\": {gb:.2}, \"median\": {gm:.2}}}, \
             \"combine_us\": {{\"best\": {cb:.2}, \"median\": {cm:.2}}}, \
             \"total_us\": {{\"best\": {tb:.2}, \"median\": {tm:.2}}}, \
             \"kv_bytes\": {}, \"floor_us\": {:.2}, \"bits\": \"{:016x}\"}}{}",
            c.kernel,
            c.n_ctx,
            c.t,
            c.kv_bytes,
            c.kv_bytes as f64 / FLOOR_BYTES_PER_S * 1e6,
            c.bits,
            if i + 1 < cases.len() { "," } else { "" }
        )
        .unwrap();
    }
    out.push_str("  ]\n}\n");
    match json {
        Some(path) => std::fs::write(&path, out).unwrap_or_else(|e| panic!("writing {path}: {e}")),
        None => print!("{out}"),
    }
}
