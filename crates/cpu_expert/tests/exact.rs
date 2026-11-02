//! Bit equality of the scalar and AVX-512 experts (and the pooled `expert_rows`) with the
//! twin-based reference: `twin::matvec::row_dot` for every dot product, `silu_det` for the
//! activation. Two checks go beyond bit equality: the FTZ gap on a tiny `x` whose activation is
//! denormal (the GPU's plain f32 shaders flush, so the worker must too) and the pool's jobs
//! carrying the FTZ bits. Skips, not fails, without AVX-512 (CI).
use cpu_expert::{avx, expert, pool, scalar, slots, with_ftz};
use kernels::dispatch::{SlotLayout, WType};
use kernels::testing::Rng;

const TYPES: [WType; 4] = [WType::Q4K, WType::Q5K, WType::Q6K, WType::Q8_0];

/// The AVX-512 worker needs AVX-512 (the test PC's CPU has it; CI's runner may not).
fn avx_or_skip() -> bool {
    let ok = avx::available();
    if !ok {
        eprintln!("no AVX-512 on this CPU; skipped");
    }
    ok
}

fn same(a: &[f32], b: &[f32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| !(x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())))
}

/// x vectors: N(0, 1), zeros (some negative), huge ±, and a mix of extremes (denormals, ±1e38, ±0, ±1, random).
fn xs(rng: &mut Rng, n: usize) -> Vec<(&'static str, Vec<f32>)> {
    let normal = slots::normal(rng, n);
    let zeros = (0..n)
        .map(|i| if i % 3 == 0 { -0.0 } else { 0.0 })
        .collect();
    let huge = (0..n).map(|_| rng.uniform(-1.0, 1.0) * 3.0e37).collect();
    let picks = [
        0.0f32, -0.0, 1e-40, -1e-40, 1.0, -1.0, 1e38, -1e38, 1e-30, 65504.0, -3.0e5,
    ];
    let mixed = (0..n)
        .map(|_| {
            let r = rng.next_u64();
            if r.is_multiple_of(4) {
                rng.uniform(-8.0, 8.0)
            } else {
                picks[(r >> 8) as usize % picks.len()]
            }
        })
        .collect();
    vec![
        ("normal", normal),
        ("zeros", zeros),
        ("huge", huge),
        ("mixed", mixed),
    ]
}

fn check_one(slot: &[u8], lay: &SlotLayout, x: &[f32], what: &str) -> usize {
    let (nf, ne) = (lay.n_ff, lay.n_embd);
    let (mut a0, mut y0) = (vec![0f32; nf], vec![0f32; ne]);
    scalar::expert_twin(slot, lay, x, &mut a0, &mut y0);
    let (mut a1, mut y1) = (vec![0f32; nf], vec![0f32; ne]);
    scalar::expert(slot, lay, x, &mut a1, &mut y1);
    assert_eq!(same(&a0, &a1), None, "scalar act, {what}");
    assert_eq!(same(&y0, &y1), None, "scalar y, {what}");
    let (mut a2, mut y2) = (vec![0f32; nf], vec![0f32; ne]);
    let (mut xs, mut as_) = (avx::XPrep::default(), avx::XPrep::default());
    unsafe { avx::expert(slot, lay, x, &mut a2, &mut y2, &mut xs, &mut as_) };
    if let Some(i) = same(&a0, &a2) {
        panic!("avx act[{i}] {:e} vs twin {:e}, {what}", a2[i], a0[i]);
    }
    if let Some(i) = same(&y0, &y2) {
        panic!("avx y[{i}] {:e} vs twin {:e}, {what}", y2[i], y0[i]);
    }
    a0.iter().filter(|v| v.is_nan()).count()
}

#[test]
fn random_slots_all_types() {
    if !avx_or_skip() {
        return;
    }
    let mut n = 0;
    let mut nan_acts = 0;
    for seed in [1u64, 2, 3] {
        for &g in &TYPES {
            for &u in &TYPES {
                for &d in &TYPES {
                    let lay = SlotLayout {
                        gate: g,
                        up: u,
                        down: d,
                        n_embd: 2048,
                        n_ff: 256,
                    };
                    let mut rng = Rng::new(seed * 1000 + n as u64);
                    // Well-formed blocks for seeds 1 and 2, raw random bytes (any f16 scale) for seed 3.
                    let slot = slots::random_slot(&mut rng, &lay, seed == 3);
                    for (name, x) in xs(&mut rng, lay.n_embd) {
                        nan_acts += check_one(
                            &slot,
                            &lay,
                            &x,
                            &format!("seed {seed} {g:?}/{u:?}/{d:?} x {name}"),
                        );
                        n += 1;
                    }
                }
            }
        }
    }
    eprintln!("{n} random experts bit-exact (64 type combinations x 3 seeds x 4 x kinds; {nan_acts} NaN acts)");
}

#[test]
fn random_slots_real_dims() {
    if !avx_or_skip() {
        return;
    }
    let mut n = 0;
    for (seed, d) in [
        (11u64, WType::Q5K),
        (12, WType::Q6K),
        (13, WType::Q4K),
        (14, WType::Q8_0),
    ] {
        let lay = SlotLayout {
            gate: WType::Q4K,
            up: WType::Q4K,
            down: d,
            n_embd: 2048,
            n_ff: 512,
        };
        let mut rng = Rng::new(seed);
        let slot = slots::random_slot(&mut rng, &lay, false);
        for (name, x) in xs(&mut rng, lay.n_embd) {
            check_one(
                &slot,
                &lay,
                &x,
                &format!("seed {seed} real dims down {d:?} x {name}"),
            );
            n += 1;
        }
    }
    eprintln!("{n} random experts at the real dims bit-exact");
}

#[test]
fn avx_expert_matches_the_twin_on_extreme_x() {
    if !avx_or_skip() {
        return;
    }
    // Four type combos on raw random bytes (any f16 scale, including denormal and non-finite
    // ones) and the mixed x (denormals, ±1e38, ±0): one broken link of the port shows here.
    // Both sides run under the same MXCSR (the twin and the worker read x alike); the FTZ
    // behavior against the GPU is the two dedicated tests below.
    for (g, u, d) in [
        (WType::Q8_0, WType::Q8_0, WType::Q8_0),
        (WType::Q4K, WType::Q4K, WType::Q4K),
        (WType::Q4K, WType::Q4K, WType::Q6K),
        (WType::Q5K, WType::Q5K, WType::Q5K),
    ] {
        let lay = SlotLayout {
            gate: g,
            up: u,
            down: d,
            n_embd: 256,
            n_ff: 256,
        };
        let mut rng = Rng::new(3);
        let slot = slots::random_slot(&mut rng, &lay, true);
        for (name, x) in xs(&mut rng, lay.n_embd) {
            check_one(&slot, &lay, &x, &format!("{g:?}/{u:?}/{d:?} x {name}"));
        }
    }
}

#[test]
fn ftz_is_required_for_a_denormal_x() {
    if !avx_or_skip() {
        return;
    }
    // A Q8_0 expert with normal scales and an all-1e-22 x: gate·x and up·x land normal (~1e-20),
    // but the activation silu(gate·x)·(up·x) lands DENORMAL (~2e-40), and FTZ flushes exactly
    // that. with_ftz (FTZ+DAZ, `MXCSR |= 0x8040`) and the default MXCSR must differ in at least
    // one y bit: the gap between the CPU's default and the GPU's flushing plain shaders.
    // (A denormal x cannot show it: both dots are so tiny that the act product underflows to 0
    // in both modes. This x is the case where the denormal act is what separates them.) Fails
    // if with_ftz is a no-op.
    let lay = SlotLayout {
        gate: WType::Q8_0,
        up: WType::Q8_0,
        down: WType::Q8_0,
        n_embd: 2048,
        n_ff: 512,
    };
    let mut rng = Rng::new(3);
    let slot = slots::random_slot(&mut rng, &lay, false);
    let x = vec![1e-22f32; lay.n_embd];
    let run = |y: &mut Vec<f32>| {
        let mut act = vec![0f32; lay.n_ff];
        expert(&slot, &lay, &x, &mut act, y);
    };
    let (mut a, mut b) = (vec![0f32; lay.n_embd], vec![0f32; lay.n_embd]);
    with_ftz(|| run(&mut a));
    run(&mut b);
    assert!(
        same(&a, &b).is_some(),
        "with_ftz changed no y bit where the act product is denormal"
    );
}

#[test]
fn pooled_matches() {
    if !avx_or_skip() {
        return;
    }
    let mut rng = Rng::new(5);
    let lay = SlotLayout {
        gate: WType::Q4K,
        up: WType::Q4K,
        down: WType::Q6K,
        n_embd: 2048,
        n_ff: 512,
    };
    let owned: Vec<Vec<u8>> = (0..5)
        .map(|_| slots::random_slot(&mut rng, &lay, false))
        .collect();
    let cases: Vec<(SlotLayout, Vec<Vec<u8>>)> = vec![(lay, owned)];
    for (lay, owned) in &cases {
        let slots_: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
        let x = slots::normal(&mut rng, lay.n_embd);
        let mut xs = avx::XPrep::default();
        unsafe { xs.build(&x, &[lay.gate, lay.up]) };
        for threads in [1usize, 3, 4, 7] {
            let p = pool::Pool::new(&vec![None; threads]);
            let (mut act, mut y) = (vec![0f32; 5 * lay.n_ff], vec![0f32; 5 * lay.n_embd]);
            pool::expert_rows(&p, &slots_, lay, &xs, &mut act, &mut y);
            for (e, s) in slots_.iter().enumerate() {
                let (mut a0, mut y0) = (vec![0f32; lay.n_ff], vec![0f32; lay.n_embd]);
                scalar::expert_twin(s, lay, &x, &mut a0, &mut y0);
                assert_eq!(
                    same(&a0, &act[e * lay.n_ff..(e + 1) * lay.n_ff]),
                    None,
                    "pool act {threads} {e}"
                );
                assert_eq!(
                    same(&y0, &y[e * lay.n_embd..(e + 1) * lay.n_embd]),
                    None,
                    "pool y {threads} {e}"
                );
            }
        }
    }
    eprintln!("1 pooled case x 4 thread counts x 5 experts bit-exact");

    // The tiny-x case: the pool's jobs carry the FTZ bits (Pool::run wraps every job), so the
    // pooled y equals a single-thread with_ftz expert's bit for bit. A pool that computes with
    // the default MXCSR differs on exactly the rows the workers took: the act product is
    // denormal at this x (see ftz_is_required_for_a_denormal_x). This is what pins the pool's
    // closures carrying the bits; ftz_is_required_for_a_denormal_x only pins that the two MXCSR
    // modes differ.
    let lay = SlotLayout {
        gate: WType::Q8_0,
        up: WType::Q8_0,
        down: WType::Q8_0,
        n_embd: 2048,
        n_ff: 512,
    };
    let mut rng = Rng::new(6);
    let slot = slots::random_slot(&mut rng, &lay, false);
    let x = vec![1e-22f32; lay.n_embd];
    let slots_ = vec![slot.as_slice()];
    let mut xs = avx::XPrep::default();
    unsafe { xs.build(&x, &[lay.gate, lay.up]) };
    let (mut act, mut y) = (vec![0f32; lay.n_ff], vec![0f32; lay.n_embd]);
    with_ftz(|| expert(slots_[0], &lay, &x, &mut act, &mut y));
    let p = pool::Pool::new(&[None; 3]);
    let (mut pact, mut py) = (vec![0f32; lay.n_ff], vec![0f32; lay.n_embd]);
    pool::expert_rows(&p, &slots_, &lay, &xs, &mut pact, &mut py);
    assert_eq!(
        same(&y, &py),
        None,
        "pooled denormal x differs from the single-thread with_ftz expert"
    );
}

/// The real model, for the `#[ignore]`d test (nothing in the production build reads it).
mod real {
    use gguf::Gguf;
    use kernels::dispatch::SlotLayout;
    use kernels::dispatch::WType;
    use std::path::PathBuf;

    pub fn model_path() -> PathBuf {
        std::env::var_os("ARTEMIS_MODEL")
            .expect("set ARTEMIS_MODEL to the Qwen3.6 GGUF")
            .into()
    }

    /// The real model's layer `l`: its slot layout and the three expert tensors' bytes.
    pub struct Layer<'a> {
        pub lay: SlotLayout,
        gate: &'a [u8],
        up: &'a [u8],
        down: &'a [u8],
    }

    impl<'a> Layer<'a> {
        pub fn new(g: &'a Gguf, l: usize) -> Layer<'a> {
            let t = |w: &str| {
                g.tensor(&format!("blk.{l}.ffn_{w}_exps.weight"))
                    .unwrap_or_else(|| panic!("no blk.{l}.ffn_{w}_exps.weight"))
            };
            let (gt, ut, dt) = (t("gate"), t("up"), t("down"));
            let ty = |v: &gguf::TensorView| WType::from_ggml(v.ggml_type).expect("a known type");
            let lay = SlotLayout {
                gate: ty(&gt),
                up: ty(&ut),
                down: ty(&dt),
                n_embd: gt.dims[0] as usize,
                n_ff: gt.dims[1] as usize,
            };
            assert_eq!(dt.dims[0] as usize, lay.n_ff);
            assert_eq!(dt.dims[1] as usize, lay.n_embd);
            Layer {
                lay,
                gate: gt.bytes,
                up: ut.bytes,
                down: dt.bytes,
            }
        }

        /// Expert e's slot bytes.
        pub fn slot(&self, e: usize) -> Vec<u8> {
            let (g, u, d) = (
                self.lay.gate_bytes(),
                self.lay.up_bytes(),
                self.lay.down_bytes(),
            );
            let mut v = Vec::with_capacity(self.lay.bytes());
            v.extend_from_slice(&self.gate[e * g..(e + 1) * g]);
            v.extend_from_slice(&self.up[e * u..(e + 1) * u]);
            v.extend_from_slice(&self.down[e * d..(e + 1) * d]);
            v
        }
    }
}

#[test]
#[ignore = "reads the 19 GB model; run on the test PC with ARTEMIS_MODEL"]
fn real_experts() {
    if !avx_or_skip() {
        return;
    }
    let path = real::model_path();
    let g = gguf::Gguf::open(&path).unwrap();
    let mut n = 0;
    let mut rng = Rng::new(77);
    let (mut order_diff, mut rows, mut silu_diff, mut acts) = (0, 0, 0, 0);
    for l in [0usize, 1, 17, 20, 34, 39] {
        let layer = real::Layer::new(&g, l);
        for e in [0usize, 1, 128, 255] {
            let slot = layer.slot(e);
            let x = slots::normal(&mut rng, layer.lay.n_embd);
            check_one(&slot, &layer.lay, &x, &format!("layer {l} expert {e}"));
            n += 1;
            // Sensitivity: how many of the down rows would differ if summed plainly in order
            // (dot_in_order), and how many acts would differ with today's silu: equality above
            // is not a coincidence of easy values.
            let lay = layer.lay;
            let (mut a0, mut y0) = (vec![0f32; lay.n_ff], vec![0f32; lay.n_embd]);
            scalar::expert_twin(&slot, &lay, &x, &mut a0, &mut y0);
            let rd = lay.down.row_bytes(lay.n_ff);
            order_diff += (0..lay.n_embd)
                .filter(|&r| {
                    let at = lay.down_offset() + r * rd;
                    kernels::twin::matvec::dot_in_order(lay.down, &slot[at..at + rd], &a0).to_bits()
                        != y0[r].to_bits()
                })
                .count();
            let rg = lay.gate.row_bytes(lay.n_embd);
            silu_diff += (0..lay.n_ff)
                .filter(|&r| {
                    let g =
                        kernels::twin::matvec::row_dot(lay.gate, &slot[r * rg..(r + 1) * rg], &x);
                    kernels::twin::silu(g).to_bits()
                        != kernels::twin::silu_det::silu_det(g).to_bits()
                })
                .count();
            rows += lay.n_embd;
            acts += lay.n_ff;
        }
        eprintln!("layer {l}: {:?}", layer.lay);
    }
    // The pooled real cases, on the same layers.
    for l in [3usize, 30] {
        let layer = real::Layer::new(&g, l);
        let lay = layer.lay;
        let owned: Vec<Vec<u8>> = (0..5).map(|e| layer.slot(e * 50)).collect();
        let slots_: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
        let x = slots::normal(&mut rng, lay.n_embd);
        let mut xs = avx::XPrep::default();
        unsafe { xs.build(&x, &[lay.gate, lay.up]) };
        let p = pool::Pool::new(&[None; 3]);
        let (mut act, mut y) = (vec![0f32; 5 * lay.n_ff], vec![0f32; 5 * lay.n_embd]);
        pool::expert_rows(&p, &slots_, &lay, &xs, &mut act, &mut y);
        for (e, s) in slots_.iter().enumerate() {
            let (mut a0, mut y0) = (vec![0f32; lay.n_ff], vec![0f32; lay.n_embd]);
            scalar::expert_twin(s, &lay, &x, &mut a0, &mut y0);
            assert_eq!(
                same(&a0, &act[e * lay.n_ff..(e + 1) * lay.n_ff]),
                None,
                "{l} {e}"
            );
            assert_eq!(
                same(&y0, &y[e * lay.n_embd..(e + 1) * lay.n_embd]),
                None,
                "{l} {e}"
            );
        }
        n += 5;
    }
    eprintln!("{n} real experts bit-exact");
    eprintln!("sensitivity: {order_diff}/{rows} down rows differ from an in-order sum; {silu_diff}/{acts} gate values have silu != silu_det");
}
