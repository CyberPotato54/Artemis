//! CI's half of test 1: every kernel's twin against the reference's operations (`qwen36::ops`), within the
//! kernel's stated tolerance. Test 1 on the GPU checks each kernel against its twin; together they bound each kernel
//! against the math.
use kernels::{twin, Kernel};
use qwen36::{ops, testing::Rng};
use std::path::Path;

type Unary = fn(f32) -> f32;

fn close(k: Kernel, case: &str, twin: &[f32], exact: &[f32]) {
    assert_eq!(twin.len(), exact.len(), "{} {case}", k.name());
    let e = ops::rel_l2(twin, exact);
    assert!(
        e <= k.tol(),
        "{} {case}: twin against ops {e:.3e}, stated {:.0e}",
        k.name(),
        k.tol()
    );
}

fn rand(rng: &mut Rng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.uniform(lo, hi)).collect()
}

#[test]
fn every_kernel_has_a_shader_and_every_shader_a_kernel() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension()? == "comp")
                .then(|| p.file_stem().unwrap().to_string_lossy().into_owned())
        })
        .collect();
    files.sort();
    let mut names: Vec<String> = Kernel::ALL.iter().map(|k| k.name().to_string()).collect();
    names.sort();
    assert_eq!(files, names);
}

#[test]
fn tree_halves_the_stride_each_step() {
    // Lanes 0 and 2 meet first: (1e8 + -1e8) + (1 + 1) = 2, where an in-order f32 sum gives 1.
    assert_eq!(twin::tree(&mut [1e8, 1.0, -1e8, 1.0]), 2.0);
    assert_eq!(
        twin::tree(&mut [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        36.0
    );
    assert_eq!(twin::strided(5, 2, |i| i as f32), vec![6.0, 4.0]);
}

#[test]
fn silu_det_matches_its_edges_and_makes_no_denormal() {
    // `silu_det` is silu from f32 add, sub and mul only, so a CPU that runs the same
    // function writes the GPU's bits. The range decisions are on the input's bits (a GPU that
    // flushes denormals takes the same branch), and no intermediate may be a denormal.
    let edges: &[u32] = &[
        0x0000_0000,
        0x8000_0000, // ±0
        0x7fc0_0000,
        0x7f80_0001, // NaN payloads, kept
        0x7f80_0000,
        0xff80_0000, // ±inf
        0x007f_ffff,
        0x0080_0000, // just below / at 2^-126
        0x00ff_ffff,
        0x0100_0000, // just below / at 2^-125
        0x337f_ffff,
        0x3380_0000, // just below / at 2^-24
        0x41ff_ffff,
        0x4200_0000, // just below / at 32
        0xc2ae_0000,
        0xc2ae_0001, // -87 and the next value down
        0x3f80_0000,
        0xbf80_0000, // ±1
    ];
    for &b in edges {
        let x = f32::from_bits(b);
        let mut saw_denormal = false;
        let y = twin::silu_det::silu_det_probe(x, &mut |v| {
            let a = v.to_bits() & 0x7fff_ffff;
            if a != 0 && a < 0x0080_0000 {
                saw_denormal = true;
            }
        });
        assert!(!saw_denormal, "{b:08x} produced a denormal");
        if x.is_nan() {
            assert!(y.is_nan() && y.to_bits() == b, "NaN payload {b:08x}");
        }
        // The range decisions, by bits.
        let a = b & 0x7fff_ffff;
        if a < 0x0100_0000 && !x.is_nan() {
            assert_eq!(y.to_bits(), 0, "{b:08x}");
        } else if a < 0x3380_0000 && !x.is_nan() {
            assert_eq!(y.to_bits(), (x * 0.5).to_bits(), "{b:08x}");
        } else if b < 0x8000_0000 && a >= 0x4200_0000 {
            assert_eq!(y.to_bits(), b, "{b:08x}");
        } else if b >= 0x8000_0000 && a > 0x42ae_0000 && !x.is_nan() {
            assert_eq!(y.to_bits(), 0, "{b:08x}");
        }
    }
}

#[test]
fn rmsnorm_twin_matches_ops() {
    let mut rng = Rng::new(1);
    for n in [2048, 256] {
        let x = rand(&mut rng, n, -2.0, 2.0);
        let w = rand(&mut rng, n, 0.8, 1.2);
        close(
            Kernel::Rmsnorm,
            &format!("n={n}"),
            &twin::rmsnorm(&x, &w, 1e-6),
            &ops::rmsnorm(&x, &w, 1e-6),
        );
    }
}

#[test]
fn add_twin_is_exact() {
    let mut rng = Rng::new(2);
    let (a, b) = (
        rand(&mut rng, 2048, -1.0, 1.0),
        rand(&mut rng, 2048, -1.0, 1.0),
    );
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_eq!(twin::add(&a, &b), want);
}

#[test]
fn elementwise_math_matches_ops_across_its_range() {
    let us: Vec<f32> = (-3000..=3000).map(|i| i as f32 * 0.01).collect();
    let pairs: [(&str, Unary, Unary); 3] = [
        ("sigmoid", twin::sigmoid, ops::sigmoid),
        ("silu", twin::silu, ops::silu),
        ("softplus", twin::softplus, ops::softplus),
    ];
    for (name, t, o) in pairs {
        let a: Vec<f32> = us.iter().map(|&u| t(u)).collect();
        let b: Vec<f32> = us.iter().map(|&u| o(u)).collect();
        let e = ops::rel_l2(&a, &b);
        assert!(e <= 1e-6, "{name}: {e:.3e}");
    }
    // softplus far below zero is about e^u: log1p_small keeps its relative precision where log(1 + e) can't.
    for u in [-25.0f32, -15.0, -8.0, -3.0, -0.5, 0.5, 3.0, 19.9] {
        let (t, o) = (twin::softplus(u), ops::softplus(u));
        assert!(((t - o) / o).abs() <= 1e-6, "softplus({u}): {t} vs {o}");
    }
}

#[test]
fn f16_rounding_is_to_nearest_even() {
    let one = 1.0f32;
    let ulp = 2f32.powi(-10);
    assert_eq!(
        twin::f16_value(twin::f16_bits(one + ulp / 2.0)),
        one,
        "a tie rounds to even"
    );
    assert_eq!(
        twin::f16_value(twin::f16_bits(one + 1.5 * ulp)),
        one + 2.0 * ulp,
        "a tie rounds to even"
    );
    assert_eq!(twin::f16_bits(65520.0), 0x7C00, "rounds up to infinity");
    assert_eq!(
        twin::f16_value(twin::f16_bits(2f32.powi(-24))),
        2f32.powi(-24),
        "the smallest subnormal"
    );
    assert_eq!(
        twin::f16_bits(2f32.powi(-26)),
        0,
        "below half the smallest subnormal"
    );
}
fn weights(rng: &mut Rng, ty: qwen36::quant::WType, rows: usize, cols: usize) -> Vec<u8> {
    kernels::testing::random_weights(rng, ty, rows, cols)
}

#[test]
fn matvec_twins_match_ops_for_every_type() {
    use qwen36::{model::Weight, quant::WType};
    let mut rng = Rng::new(3);
    for ty in WType::ALL {
        for (rows, cols) in [(64, 2048), (16, 512), (5, 256)] {
            let w = weights(&mut rng, ty, rows, cols);
            let x = rand(&mut rng, cols, -1.0, 1.0);
            let exact = ops::matvec(
                &Weight {
                    ty,
                    cols,
                    rows,
                    bytes: &w,
                },
                &x,
                1,
            );
            close(
                Kernel::matvec(ty),
                &format!("{} {rows}x{cols}", ty.name()),
                &twin::matvec::matvec(ty, &w, rows, &x),
                &exact,
            );
        }
    }
}

#[test]
fn unit_dot_covers_each_row_exactly_once() {
    // A row of ones against x = 0, 1, 2, …: the 16 units of a 512-value row together sum every x once.
    use qwen36::quant::WType;
    let cols = 512;
    let x: Vec<f32> = (0..cols).map(|i| i as f32).collect();
    let ones: Vec<u8> = (0..cols).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    let total: f32 = (0..cols / 32)
        .map(|u| twin::matvec::unit_dot(WType::F32, &ones, u, &x))
        .sum();
    assert_eq!(total, (cols * (cols - 1) / 2) as f32);
}
use qwen36::dims::Dims;

/// Random q and k, each key group L2-normalized as the conv output is before the delta rule.
fn unit_groups(rng: &mut Rng, n: usize, dk: usize) -> Vec<f32> {
    rand(rng, n, -1.0, 1.0)
        .chunks(dk)
        .flat_map(|c| ops::l2norm(c, 1e-6))
        .collect()
}

#[test]
fn conv_step_twin_matches_ops_over_steps() {
    let mut rng = Rng::new(4);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (n, kw) = (d.conv_dim(), d.ssm_conv);
        let w = rand(&mut rng, n * kw, -0.5, 0.5);
        let (mut st_t, mut st_o) = (vec![0f32; n * (kw - 1)], vec![0f32; n * (kw - 1)]);
        for step in 0..4 {
            let x = rand(&mut rng, n, -2.0, 2.0);
            let t = twin::deltanet::conv_step(&mut st_t, &w, &x, kw);
            let o = ops::conv_step(&mut st_o, &w, &x, kw);
            close(Kernel::ConvStep, &format!("n={n} step {step}"), &t, &o);
            assert_eq!(
                st_t, st_o,
                "the state holds raw inputs, so it matches exactly"
            );
        }
    }
}

#[test]
fn gdn_gates_twin_matches_ops() {
    let mut rng = Rng::new(5);
    let b = rand(&mut rng, 32, -6.0, 6.0);
    let alpha = rand(&mut rng, 32, -8.0, 8.0);
    let dt = rand(&mut rng, 32, -2.0, 2.0);
    let a = rand(&mut rng, 32, -1.5, -0.1);
    let (bt, gt) = twin::deltanet::gdn_gates(&b, &alpha, &dt, &a);
    let (bo, go) = ops::gdn_gates(&b, &alpha, &dt, &a);
    close(Kernel::GdnGates, "beta", &bt, &bo);
    close(Kernel::GdnGates, "g", &gt, &go);
}

#[test]
fn l2norm_twin_matches_ops() {
    let mut rng = Rng::new(6);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let y = rand(&mut rng, d.conv_dim(), -2.0, 2.0);
        let (q, k, _) = ops::split_qkv(&y, &d, d.rms_eps);
        let mut t = y.clone();
        twin::deltanet::l2norm(&mut t[..2 * d.key_dim()], d.ssm_d_state, d.rms_eps);
        close(Kernel::L2norm, "q", &t[..d.key_dim()], &q);
        close(Kernel::L2norm, "k", &t[d.key_dim()..2 * d.key_dim()], &k);
        assert_eq!(t[2 * d.key_dim()..], y[2 * d.key_dim()..], "v is untouched");
    }
}

#[test]
fn gdn_step_twin_matches_ops_over_steps() {
    let mut rng = Rng::new(7);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (nv, dk, dv, ng) = (d.ssm_dt_rank, d.ssm_d_state, d.v_head_dim(), d.ssm_n_group);
        let (mut st_t, mut st_o) = (vec![0f32; nv * dv * dk], vec![0f32; nv * dv * dk]);
        for step in 0..4 {
            let q = unit_groups(&mut rng, ng * dk, dk);
            let k = unit_groups(&mut rng, ng * dk, dk);
            let v = rand(&mut rng, nv * dv, -1.0, 1.0);
            let g = rand(&mut rng, nv, -2.0, 0.0);
            let beta = rand(&mut rng, nv, 0.05, 0.95);
            let t = twin::deltanet::gdn_step(&mut st_t, &q, &k, &v, &g, &beta, ng, dk, dv);
            let o = ops::delta_rule(&mut st_o, &q, &k, &v, &g, &beta, &d);
            close(Kernel::GdnStep, &format!("dk={dk} step {step}"), &t, &o);
            close(
                Kernel::GdnStep,
                &format!("dk={dk} state after step {step}"),
                &st_t,
                &st_o,
            );
        }
    }
}

#[test]
fn gated_rmsnorm_twin_matches_ops() {
    let mut rng = Rng::new(8);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let dv = d.v_head_dim();
        let o = rand(&mut rng, d.ssm_inner, -1.0, 1.0);
        let z = rand(&mut rng, d.ssm_inner, -4.0, 4.0);
        let w = rand(&mut rng, dv, 0.5, 1.5);
        close(
            Kernel::GatedRmsnorm,
            &format!("dv={dv}"),
            &twin::deltanet::gated_rmsnorm(&o, &z, &w, dv, d.rms_eps),
            &ops::gated_norm(&o, &z, &w, d.rms_eps, dv),
        );
    }
}
use kernels::twin::attention as at;

#[test]
fn rope_table_matches_the_reference_far_into_the_context() {
    let mut rng = Rng::new(9);
    for pos in [0u32, 7, 4096, 60_000] {
        let x = rand(&mut rng, 256, -1.0, 1.0);
        let mut want = x.clone();
        ops::rope(&mut want, pos, 64, 1e7);
        let mut got = x.clone();
        at::rotate(&mut got, &at::rope_table(pos, 64, 1e7));
        close(Kernel::QkNormRope, &format!("pos {pos}"), &got, &want);
    }
}

#[test]
fn qk_norm_rope_twin_matches_ops() {
    let mut rng = Rng::new(10);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let hd = d.head_dim;
        for pos in [0u32, 5, 60_000] {
            let qfull = rand(&mut rng, 2 * d.n_head * hd, -2.0, 2.0);
            let k = rand(&mut rng, d.n_head_kv * hd, -2.0, 2.0);
            let (qn, kn) = (rand(&mut rng, hd, 0.5, 1.5), rand(&mut rng, hd, 0.5, 1.5));
            let table = at::rope_table(pos, d.n_rot, d.rope_base);
            let mut kt = k.clone();
            let (qt, gt) =
                at::qk_norm_rope(&qfull, &mut kt, &qn, &kn, &table, d.n_head, hd, d.rms_eps);
            let (mut qo, go) = ops::split_q_gate(&qfull, hd);
            let mut ko = k.clone();
            ops::norm_rope(&mut qo, &qn, d.rms_eps, hd, pos, d.n_rot, d.rope_base);
            ops::norm_rope(&mut ko, &kn, d.rms_eps, hd, pos, d.n_rot, d.rope_base);
            close(
                Kernel::QkNormRope,
                &format!("q hd={hd} pos {pos}"),
                &qt,
                &qo,
            );
            close(
                Kernel::QkNormRope,
                &format!("k hd={hd} pos {pos}"),
                &kt,
                &ko,
            );
            assert_eq!(gt, go, "the gate is copied, not computed");
        }
    }
}

/// Caches of `n_ctx` positions holding f16 bits of random values, and the same values as f32 for `ops`.
fn caches(rng: &mut Rng, n_ctx: usize, width: usize) -> (Vec<u16>, Vec<u16>, Vec<f32>, Vec<f32>) {
    let kc: Vec<u16> = (0..n_ctx * width)
        .map(|_| twin::f16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let vc: Vec<u16> = (0..n_ctx * width)
        .map(|_| twin::f16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let kf = kc.iter().map(|&b| twin::f16_value(b)).collect();
    let vf = vc.iter().map(|&b| twin::f16_value(b)).collect();
    (kc, vc, kf, vf)
}

#[test]
fn attention_twin_matches_ops_across_split_boundaries() {
    let mut rng = Rng::new(11);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
        for n_ctx in [1, 256, 257, 1000] {
            // Queries scaled so scores spread over several units, as trained heads' do (score_range checks it).
            let q = rand(&mut rng, nh * hd, -12.0, 12.0);
            let (kc, vc, kf, vf) = caches(&mut rng, n_ctx, nkv * hd);
            if n_ctx >= 256 {
                let spread = kernels::testing::score_range(&q, &kf, n_ctx, nh, nkv, hd);
                assert!(
                    spread >= 8.0,
                    "hd={hd} n_ctx={n_ctx}: scores span only {spread:.2}"
                );
            }
            close(
                Kernel::AttnDecode,
                &format!("hd={hd} n_ctx={n_ctx}"),
                &at::attention(&q, &kc, &vc, n_ctx, nh, nkv, hd),
                &ops::attention(&q, &kf, &vf, n_ctx, &d),
            );
        }
    }
}

#[test]
fn kv_write_rounds_to_nearest_even_f16() {
    let k = [1.0f32, 1.0 + 2f32.powi(-11), 70_000.0, -0.0, 1e-7, 0.333];
    let v = [2.0f32; 6];
    let (mut kc, mut vc) = (vec![0u16; 18], vec![0u16; 18]);
    at::kv_write(&k, &v, &mut kc, &mut vc, 2, 6);
    assert_eq!(&kc[..12], &[0u16; 12], "only position 2 is written");
    assert_eq!(kc[12], 0x3C00);
    assert_eq!(kc[13], 0x3C00, "a tie rounds to even");
    assert_eq!(kc[14], 0x7C00, "overflow to infinity");
    assert_eq!(kc[15], 0x8000);
    assert_eq!(vc[12..], [0x4000u16; 6]);
}

#[test]
fn sigmoid_mul_twin_matches_ops() {
    let mut rng = Rng::new(12);
    let a = rand(&mut rng, 4096, -2.0, 2.0);
    let gte = rand(&mut rng, 4096, -8.0, 8.0);
    let want: Vec<f32> = a
        .iter()
        .zip(&gte)
        .map(|(x, g)| x * ops::sigmoid(*g))
        .collect();
    close(
        Kernel::SigmoidMul,
        "n=4096",
        &at::sigmoid_mul(&a, &gte),
        &want,
    );
}
use kernels::{dispatch::SlotLayout, twin::moe};
use qwen36::{model::Weight, quant::WType};

#[test]
fn slot_sizes_are_the_specs() {
    let lay = |down| SlotLayout {
        gate: WType::Q4K,
        up: WType::Q4K,
        down,
        n_embd: 2048,
        n_ff: 512,
    };
    // The slot size: 589,824 + 589,824 + 720,896 (Q5_K) or 860,160 (Q6_K).
    assert_eq!(lay(WType::Q5K).bytes(), 1_900_544);
    assert_eq!(lay(WType::Q6K).bytes(), 2_039_808);
    assert_eq!(lay(WType::Q5K).up_offset(), 589_824);
    assert_eq!(lay(WType::Q5K).down_offset(), 1_179_648);
    let e = kernels::dispatch::slot_entry(0x1234_5678_9abc, kernels::dispatch::WHERE_HOST);
    assert_eq!(
        u64::from_le_bytes(e[..8].try_into().unwrap()),
        0x1234_5678_9abc
    );
    assert_eq!(u32::from_le_bytes(e[8..12].try_into().unwrap()), 1);
}

#[test]
fn router_topk_twin_matches_ops() {
    let mut rng = Rng::new(13);
    for (n, k) in [(256, 8), (16, 4)] {
        let logits = rand(&mut rng, n, -8.0, 8.0);
        let (probs, ids, w) = moe::router_topk(&logits, k);
        let p_ops = ops::softmax(&logits);
        close(Kernel::RouterTopk, "probs", &probs, &p_ops);
        let top = ops::top_k(&p_ops, k);
        assert_eq!(
            ids,
            top.iter().map(|&e| e as f32).collect::<Vec<_>>(),
            "n={n}"
        );
        close(
            Kernel::RouterTopk,
            "weights",
            &w,
            &ops::route_weights(&p_ops, &top),
        );
    }
    // Exactly equal probabilities go to the lower id.
    let (_, ids, _) = moe::router_topk(&[1.0, 3.0, 3.0, 2.0], 2);
    assert_eq!(ids, [1.0, 2.0]);
}

/// `n` random experts packed as slots of `lay`.
fn slots(rng: &mut Rng, lay: &SlotLayout, n: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|_| {
            let mut s = kernels::testing::random_weights(rng, lay.gate, lay.n_ff, lay.n_embd);
            s.extend(kernels::testing::random_weights(
                rng, lay.up, lay.n_ff, lay.n_embd,
            ));
            s.extend(kernels::testing::random_weights(
                rng, lay.down, lay.n_embd, lay.n_ff,
            ));
            s
        })
        .collect()
}

#[test]
fn expert_twins_match_ops() {
    let mut rng = Rng::new(14);
    let lays = [
        SlotLayout {
            gate: WType::Q4K,
            up: WType::Q4K,
            down: WType::Q5K,
            n_embd: 2048,
            n_ff: 512,
        },
        SlotLayout {
            gate: WType::Q4K,
            up: WType::Q4K,
            down: WType::Q6K,
            n_embd: 256,
            n_ff: 256,
        },
        SlotLayout {
            gate: WType::Q8_0,
            up: WType::Q5K,
            down: WType::Q4K,
            n_embd: 2048,
            n_ff: 512,
        },
    ];
    for lay in lays {
        let s = slots(&mut rng, &lay, 3);
        let refs: Vec<&[u8]> = s.iter().map(|v| v.as_slice()).collect();
        let x = rand(&mut rng, lay.n_embd, -1.0, 1.0);
        let act = moe::experts_gate_up(&refs, &lay, &x);
        let mut want = Vec::new();
        for slot in &refs {
            let w = |ty, off: usize, rows, cols, len: usize| Weight {
                ty,
                cols,
                rows,
                bytes: &slot[off..off + len],
            };
            let g = ops::matvec(
                &w(lay.gate, 0, lay.n_ff, lay.n_embd, lay.gate_bytes()),
                &x,
                1,
            );
            let u = ops::matvec(
                &w(
                    lay.up,
                    lay.up_offset(),
                    lay.n_ff,
                    lay.n_embd,
                    lay.up_bytes(),
                ),
                &x,
                1,
            );
            want.extend(ops::swiglu(&g, &u));
        }
        let case = format!(
            "{}/{}/{} {}",
            lay.gate.name(),
            lay.up.name(),
            lay.down.name(),
            lay.n_embd
        );
        close(Kernel::ExpertsGateUp, &case, &act, &want);
        let ye = moe::experts_down(&refs, &lay, &act);
        let mut want = Vec::new();
        for (j, slot) in refs.iter().enumerate() {
            let w = Weight {
                ty: lay.down,
                cols: lay.n_ff,
                rows: lay.n_embd,
                bytes: &slot[lay.down_offset()..lay.down_offset() + lay.down_bytes()],
            };
            want.extend(ops::matvec(&w, &act[j * lay.n_ff..(j + 1) * lay.n_ff], 1));
        }
        close(Kernel::ExpertsDown, &case, &ye, &want);
    }
}

#[test]
fn moe_combine_and_swiglu_twins_match_ops() {
    let mut rng = Rng::new(15);
    let (n, k) = (2048, 8);
    let ye = rand(&mut rng, k * n, -1.0, 1.0);
    let mut w = rand(&mut rng, k, 0.0, 1.0);
    let s: f32 = w.iter().sum();
    w.iter_mut().for_each(|v| *v /= s);
    let sh = rand(&mut rng, n, -1.0, 1.0);
    let sg = 0.7f32;
    let (m, x) = moe::moe_combine(&ye, &w, &sh, sg, k);
    close(Kernel::MoeCombine, "moe", &m, &ops::moe_sum(&ye, &w, n));
    let want: Vec<f32> = sh.iter().map(|v| v * ops::sigmoid(sg)).collect();
    close(Kernel::MoeCombine, "shexp", &x, &want);
    let (g, u) = (
        rand(&mut rng, 512, -6.0, 6.0),
        rand(&mut rng, 512, -1.0, 1.0),
    );
    close(
        Kernel::Swiglu,
        "n=512",
        &moe::swiglu(&g, &u),
        &ops::swiglu(&g, &u),
    );
}

#[test]
fn status_names_the_lowest_refused_entry_of_each_kind() {
    use kernels::dispatch::{read_status, ExpertStatus, STATUS_CLEAR};
    let mut b = STATUS_CLEAR;
    assert_eq!(
        read_status(&b),
        ExpertStatus::default(),
        "a cleared status names nothing"
    );
    b[..4].copy_from_slice(&((3u32 << 16) | 200).to_le_bytes());
    b[4..8].copy_from_slice(&((39u32 << 16) | 7).to_le_bytes());
    assert_eq!(
        read_status(&b),
        ExpertStatus {
            cpu: Some((3, 200)),
            bad: Some((39, 7)),
        }
    );
}

#[test]
fn apply_slot_updates_twin_rewrites_only_the_named_entries() {
    // No `qwen36::ops` counterpart: the twin is checked against the entry format itself. The GPU kernel is checked
    // against the twin bit for bit (test 1).
    use kernels::dispatch::{
        slot_entry, slot_update, ENTRY_BYTES, WHERE_CPU, WHERE_HOST, WHERE_VRAM,
    };
    let n = 41 * 256;
    let mut table: Vec<u8> = (0..n)
        .flat_map(|i| slot_entry(0x1000 + i as u64, WHERE_HOST))
        .collect();
    let before = table.clone();
    let updates = [
        slot_update(0, WHERE_VRAM, 0xAB_0000),
        slot_update(5_000, WHERE_CPU, 0),
        // Past the end of the table: skipped.
        slot_update(n as u32, WHERE_VRAM, 0xDEAD),
        // Past `count`: not applied.
        slot_update(7, WHERE_VRAM, 0x77),
    ]
    .concat();
    moe::apply_slot_updates(&mut table, &updates, 3);
    let entry = |t: &[u8], i: usize| t[i * ENTRY_BYTES..(i + 1) * ENTRY_BYTES].to_vec();
    assert_eq!(entry(&table, 0), slot_entry(0xAB_0000, WHERE_VRAM));
    assert_eq!(entry(&table, 5_000), slot_entry(0, WHERE_CPU));
    for i in (0..n).filter(|&i| i != 0 && i != 5_000) {
        assert_eq!(entry(&table, i), entry(&before, i), "entry {i}");
    }
}

#[test]
fn argmax_skips_nan_and_ties_to_the_lower_id() {
    use twin::argmax_row;
    let nan = f32::NAN;
    assert_eq!(argmax_row(&[1.0, nan, 3.0, 3.0]), Some(2));
    assert_eq!(argmax_row(&[nan, 1.0]), Some(1), "a NaN at 0 doesn't win");
    assert_eq!(argmax_row(&[nan; 5]), None);
    assert_eq!(argmax_row(&[]), None);
    // Only the largest finite value counts: no infinity wins, and a row of -inf has no pick.
    assert_eq!(argmax_row(&[f32::NEG_INFINITY; 4]), None);
    assert_eq!(
        argmax_row(&[f32::NEG_INFINITY, 2.0, f32::INFINITY, nan]),
        Some(1)
    );
    assert_eq!(argmax_row(&[f32::INFINITY, f32::NEG_INFINITY]), None);
    // A finite -FLT_MAX still beats nothing.
    assert_eq!(argmax_row(&[f32::NEG_INFINITY, f32::MIN]), Some(1));
    // The equivalence with `engine::sample::greedy` on rows with no NaN and no infinity is tested in the engine crate.
}
