//! Test 1 on the GPU: every kernel against its twin on random inputs, at the real model's shapes and the
//! tiny model's, within the kernel's stated tolerance. `check` records each measurement when ARTEMIS_EVIDENCE is set.
use kernels::{
    testing::{check, rand_f32, random_weights, score_range, Gpu},
    twin, Kernel,
};
mod common;
use common::attention_shapes;
use qwen36::{quant::WType, testing::Rng};

#[test]
#[ignore = "needs the GPU"]
fn add() {
    let g = Gpu::new();
    let mut rng = Rng::new(10);
    let (a, b) = (
        rand_f32(&mut rng, 2048, -1.0, 1.0),
        rand_f32(&mut rng, 2048, -1.0, 1.0),
    );
    let (ga, gb, y) = (g.f32s(&a), g.f32s(&b), g.zeros(2048));
    g.run(|k, r| k.add(r, ga.addr, gb.addr, y.addr, 2048));
    check(Kernel::Add, "n=2048", &g.read(y), &twin::add(&a, &b));
}

#[test]
#[ignore = "needs the GPU"]
fn rmsnorm() {
    let g = Gpu::new();
    let mut rng = Rng::new(11);
    for n in [2048, 256] {
        let x = rand_f32(&mut rng, n, -2.0, 2.0);
        let w = rand_f32(&mut rng, n, 0.8, 1.2);
        let (gx, gw, y) = (g.f32s(&x), g.f32s(&w), g.zeros(n));
        g.run(|k, r| k.rmsnorm(r, gx.addr, gw.addr, y.addr, n, 1e-6));
        check(
            Kernel::Rmsnorm,
            &format!("n={n}"),
            &g.read(y),
            &twin::rmsnorm(&x, &w, 1e-6),
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn embed_row_every_type_from_host_memory() {
    let g = Gpu::new();
    let mut rng = Rng::new(12);
    let n = 2048;
    for ty in WType::ALL {
        let table = random_weights(&mut rng, ty, 4, n);
        let rb = ty.row_bytes(n);
        let host = g.host_bytes(&table);
        let y = g.zeros(n);
        g.run(|k, r| k.embed_row(r, host.addr + 2 * rb as u64, ty, n, y.addr));
        check(
            Kernel::EmbedRow,
            ty.name(),
            &g.read(y),
            &twin::embed_row(ty, &table[2 * rb..3 * rb], n),
        );
    }
}
use kernels::{dispatch::MatW, twin::matvec as mv};

/// Uploads `w` (VRAM or pinned host memory), runs the matvec, and returns its output.
fn gpu_matvec(g: &Gpu, ty: WType, w: &[u8], rows: usize, x: &[f32], host: bool) -> Vec<f32> {
    let gw = if host { g.host_bytes(w) } else { g.bytes(w) };
    let (gx, y) = (g.f32s(x), g.zeros(rows));
    let m = MatW {
        addr: gw.addr,
        ty,
        rows,
        cols: x.len(),
    };
    g.run(|k, r| k.matvec(r, m, gx.addr, y.addr));
    g.read(y)
}

#[test]
#[ignore = "needs the GPU"]
fn matvec_every_type_and_shape() {
    let g = Gpu::new();
    let mut rng = Rng::new(13);
    // Decode's shapes: one row (the shared-expert gate), a ragged count, expert gate/up and down, the tiny model's,
    // and a DeltaNet QKV projection.
    let shapes = [
        (1, 2048),
        (37, 2048),
        (512, 2048),
        (2048, 512),
        (256, 256),
        (8192, 2048),
    ];
    for ty in WType::ALL {
        for (rows, cols) in shapes {
            let w = random_weights(&mut rng, ty, rows, cols);
            let x = rand_f32(&mut rng, cols, -1.0, 1.0);
            let got = gpu_matvec(&g, ty, &w, rows, &x, false);
            check(
                Kernel::matvec(ty),
                &format!("{} {rows}x{cols}", ty.name()),
                &got,
                &mv::matvec(ty, &w, rows, &x),
            );
        }
    }
}

/// Decode's dot products are their twins', bit for bit: `unit_dot`'s sums are `precise`, so each
/// product is rounded and then added in the twin's order, and the 32 lanes meet in the twin's tree. Every type, from
/// VRAM and from pinned host memory, at the real model's shapes.
#[test]
#[ignore = "needs the GPU"]
fn matvec_equals_its_twin_bit_for_bit() {
    let g = Gpu::new();
    let mut rng = Rng::new(15);
    for ty in WType::ALL {
        for (rows, cols) in [(512, 2048), (2048, 512), (8192, 2048)] {
            let w = random_weights(&mut rng, ty, rows, cols);
            let x = rand_f32(&mut rng, cols, -1.0, 1.0);
            let want = mv::matvec(ty, &w, rows, &x);
            for host in [false, true] {
                let got = gpu_matvec(&g, ty, &w, rows, &x, host);
                let differ = got
                    .iter()
                    .zip(&want)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    differ,
                    0,
                    "{} {rows}x{cols} host={host}: {differ} of {rows} outputs differ from the twin",
                    ty.name()
                );
            }
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn matvec_from_host_memory_is_bit_identical_to_vram() {
    // HOST and VRAM weights run the same code at a different address, so results are identical.
    let g = Gpu::new();
    let mut rng = Rng::new(14);
    for ty in [WType::Q4K, WType::Q5K, WType::Q6K, WType::Q8_0] {
        let w = random_weights(&mut rng, ty, 512, 2048);
        let x = rand_f32(&mut rng, 2048, -1.0, 1.0);
        let vram = gpu_matvec(&g, ty, &w, 512, &x, false);
        let host = gpu_matvec(&g, ty, &w, 512, &x, true);
        assert_eq!(vram, host, "{}", ty.name());
    }
}
use kernels::twin::deltanet as dn;
use qwen36::dims::Dims;

fn unit_groups(rng: &mut Rng, n: usize, dk: usize) -> Vec<f32> {
    rand_f32(rng, n, -1.0, 1.0)
        .chunks(dk)
        .flat_map(|c| qwen36::ops::l2norm(c, 1e-6))
        .collect()
}

#[test]
#[ignore = "needs the GPU"]
fn conv_step_over_steps() {
    let g = Gpu::new();
    let mut rng = Rng::new(15);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (n, kw) = (d.conv_dim(), d.ssm_conv);
        let w = rand_f32(&mut rng, n * kw, -0.5, 0.5);
        let gw = g.f32s(&w);
        let gstate = g.zeros(n * (kw - 1));
        let mut state = vec![0f32; n * (kw - 1)];
        for step in 0..4 {
            let x = rand_f32(&mut rng, n, -2.0, 2.0);
            let (gx, y) = (g.f32s(&x), g.zeros(n));
            g.run(|k, r| k.conv_step(r, gx.addr, gw.addr, gstate.addr, y.addr, n, kw));
            let want = dn::conv_step(&mut state, &w, &x, kw);
            check(
                Kernel::ConvStep,
                &format!("n={n} step {step}"),
                &g.read(y),
                &want,
            );
            assert_eq!(
                g.read(gstate),
                state,
                "the conv state shifts in raw inputs exactly"
            );
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn gdn_gates() {
    let g = Gpu::new();
    let mut rng = Rng::new(16);
    let n = 32;
    let (b, alpha) = (
        rand_f32(&mut rng, n, -6.0, 6.0),
        rand_f32(&mut rng, n, -8.0, 8.0),
    );
    let (dt, a) = (
        rand_f32(&mut rng, n, -2.0, 2.0),
        rand_f32(&mut rng, n, -1.5, -0.1),
    );
    let ins = [g.f32s(&b), g.f32s(&alpha), g.f32s(&dt), g.f32s(&a)];
    let (beta, gg) = (g.zeros(n), g.zeros(n));
    g.run(|k, r| {
        k.gdn_gates(
            r,
            ins[0].addr,
            ins[1].addr,
            ins[2].addr,
            ins[3].addr,
            beta.addr,
            gg.addr,
            n,
        )
    });
    let (bt, gt) = dn::gdn_gates(&b, &alpha, &dt, &a);
    check(Kernel::GdnGates, "beta", &g.read(beta), &bt);
    check(Kernel::GdnGates, "g", &g.read(gg), &gt);
}

#[test]
#[ignore = "needs the GPU"]
fn l2norm() {
    let g = Gpu::new();
    let mut rng = Rng::new(17);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let x = rand_f32(&mut rng, 2 * d.key_dim(), -2.0, 2.0);
        let gx = g.f32s(&x);
        g.run(|k, r| k.l2norm(r, gx.addr, 2 * d.ssm_n_group, d.ssm_d_state, d.rms_eps));
        let mut want = x.clone();
        dn::l2norm(&mut want, d.ssm_d_state, d.rms_eps);
        check(
            Kernel::L2norm,
            &format!("dk={}", d.ssm_d_state),
            &g.read(gx),
            &want,
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn gdn_step_over_steps() {
    let g = Gpu::new();
    let mut rng = Rng::new(18);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (nv, dk, dv, ng) = (d.ssm_dt_rank, d.ssm_d_state, d.v_head_dim(), d.ssm_n_group);
        let gstate = g.zeros(nv * dv * dk);
        let mut state = vec![0f32; nv * dv * dk];
        for step in 0..4 {
            let q = unit_groups(&mut rng, ng * dk, dk);
            let kk = unit_groups(&mut rng, ng * dk, dk);
            let v = rand_f32(&mut rng, nv * dv, -1.0, 1.0);
            let gg = rand_f32(&mut rng, nv, -2.0, 0.0);
            let beta = rand_f32(&mut rng, nv, 0.05, 0.95);
            let ins = [
                g.f32s(&q),
                g.f32s(&kk),
                g.f32s(&v),
                g.f32s(&gg),
                g.f32s(&beta),
            ];
            let o = g.zeros(nv * dv);
            g.run(|k, r| {
                k.gdn_step(
                    r,
                    ins[0].addr,
                    ins[1].addr,
                    ins[2].addr,
                    ins[3].addr,
                    ins[4].addr,
                    gstate.addr,
                    o.addr,
                    nv,
                    dk,
                    dv,
                    ng,
                )
            });
            let want = dn::gdn_step(&mut state, &q, &kk, &v, &gg, &beta, ng, dk, dv);
            check(
                Kernel::GdnStep,
                &format!("dk={dk} step {step}"),
                &g.read(o),
                &want,
            );
            check(
                Kernel::GdnStep,
                &format!("dk={dk} state after step {step}"),
                &g.read(gstate),
                &state,
            );
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn gated_rmsnorm() {
    let g = Gpu::new();
    let mut rng = Rng::new(19);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let dv = d.v_head_dim();
        let o = rand_f32(&mut rng, d.ssm_inner, -1.0, 1.0);
        let z = rand_f32(&mut rng, d.ssm_inner, -4.0, 4.0);
        let w = rand_f32(&mut rng, dv, 0.5, 1.5);
        let (go, gz, gw, y) = (g.f32s(&o), g.f32s(&z), g.f32s(&w), g.zeros(d.ssm_inner));
        g.run(|k, r| {
            k.gated_rmsnorm(
                r,
                go.addr,
                gz.addr,
                gw.addr,
                y.addr,
                d.ssm_dt_rank,
                dv,
                d.rms_eps,
            )
        });
        check(
            Kernel::GatedRmsnorm,
            &format!("dv={dv}"),
            &g.read(y),
            &dn::gated_rmsnorm(&o, &z, &w, dv, d.rms_eps),
        );
    }
}
use kernels::{
    dispatch::{attn_decode_kernel, attn_splits},
    twin::attention as at,
};

fn read_u16(g: &Gpu, s: vk::Slice) -> Vec<u16> {
    g.dev
        .download(s)
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
#[ignore = "needs the GPU"]
fn qk_norm_rope_near_and_far_positions() {
    let g = Gpu::new();
    let mut rng = Rng::new(20);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let hd = d.head_dim;
        for pos in [5u32, 60_000] {
            let qfull = rand_f32(&mut rng, 2 * d.n_head * hd, -2.0, 2.0);
            let k = rand_f32(&mut rng, d.n_head_kv * hd, -2.0, 2.0);
            let (qn, kn) = (
                rand_f32(&mut rng, hd, 0.5, 1.5),
                rand_f32(&mut rng, hd, 0.5, 1.5),
            );
            let table = at::rope_table(pos, d.n_rot, d.rope_base);
            let (gqf, gk, gqn, gkn, gt) = (
                g.f32s(&qfull),
                g.f32s(&k),
                g.f32s(&qn),
                g.f32s(&kn),
                g.f32s(&table),
            );
            let (q, gate) = (g.zeros(d.n_head * hd), g.zeros(d.n_head * hd));
            g.run(|kk, r| {
                kk.qk_norm_rope(
                    r,
                    gqf.addr,
                    gk.addr,
                    q.addr,
                    gate.addr,
                    gqn.addr,
                    gkn.addr,
                    gt.addr,
                    d.n_head,
                    d.n_head_kv,
                    hd,
                    d.n_rot,
                    d.rms_eps,
                )
            });
            let mut kt = k.clone();
            let (qt, gtt) =
                at::qk_norm_rope(&qfull, &mut kt, &qn, &kn, &table, d.n_head, hd, d.rms_eps);
            check(
                Kernel::QkNormRope,
                &format!("q hd={hd} pos {pos}"),
                &g.read(q),
                &qt,
            );
            check(
                Kernel::QkNormRope,
                &format!("k hd={hd} pos {pos}"),
                &g.read(gk),
                &kt,
            );
            assert_eq!(g.read(gate), gtt, "the gate is copied exactly");
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn kv_write_is_bit_exact() {
    let g = Gpu::new();
    let mut rng = Rng::new(21);
    let width = 512;
    let mut k = rand_f32(&mut rng, width, -4.0, 4.0);
    // Ties, overflow, signed zero, subnormals and the smallest values f16 rounds to zero.
    k[..8].copy_from_slice(&[
        1.0 + 2f32.powi(-11),
        1.0 + 3.0 * 2f32.powi(-11),
        70_000.0,
        -0.0,
        1e-7,
        -3e-6,
        2f32.powi(-26),
        65_519.0,
    ]);
    let v = rand_f32(&mut rng, width, -4.0, 4.0);
    let (gk, gv) = (g.f32s(&k), g.f32s(&v));
    let (kc, vc) = (g.zeros(4 * width / 2), g.zeros(4 * width / 2));
    g.run(|kk, r| kk.kv_write(r, gk.addr, gv.addr, kc.addr, vc.addr, 3, width));
    let (mut tk, mut tv) = (vec![0u16; 4 * width], vec![0u16; 4 * width]);
    at::kv_write(&k, &v, &mut tk, &mut tv, 3, width);
    assert_eq!(read_u16(&g, kc), tk);
    assert_eq!(read_u16(&g, vc), tv);
}

#[test]
#[ignore = "needs the GPU"]
fn attention_across_split_boundaries() {
    let g = Gpu::new();
    let mut rng = Rng::new(22);
    for (nh, nkv, hd) in attention_shapes() {
        // 3,073 and 5,000 run attn_decode_gqa (past ATTN_GQA_FROM); the rest attn_decode.
        for n_ctx in [1, 256, 257, 1000, 3073, 5000] {
            // Queries scaled so scores spread over several units, as trained heads' do (score_range checks it).
            let q = rand_f32(&mut rng, nh * hd, -12.0, 12.0);
            let kc: Vec<u16> = (0..n_ctx * nkv * hd)
                .map(|_| twin::f16_bits(rng.uniform(-1.0, 1.0)))
                .collect();
            let vc: Vec<u16> = (0..n_ctx * nkv * hd)
                .map(|_| twin::f16_bits(rng.uniform(-1.0, 1.0)))
                .collect();
            let (gq, gkc, gvc) = (
                g.f32s(&q),
                g.bytes(&u16_bytes(&kc)),
                g.bytes(&u16_bytes(&vc)),
            );
            let part = g.zeros(nh * attn_splits(n_ctx) * (hd + 2));
            let y = g.zeros(nh * hd);
            g.run(|kk, r| {
                kk.attn_decode(
                    r, gq.addr, gkc.addr, gvc.addr, part.addr, n_ctx, nh, nkv, hd,
                );
                r.barrier();
                kk.attn_combine(r, part.addr, y.addr, n_ctx, nh, hd);
            });
            if n_ctx >= 256 {
                let kf: Vec<f32> = kc.iter().map(|&b| twin::f16_value(b)).collect();
                let spread = score_range(&q, &kf, n_ctx, nh, nkv, hd);
                assert!(
                    spread >= 8.0,
                    "hd={hd} n_ctx={n_ctx}: scores span only {spread:.2}"
                );
            }
            let want = at::attention(&q, &kc, &vc, n_ctx, nh, nkv, hd);
            check(
                attn_decode_kernel(n_ctx, nh, nkv, hd),
                &format!("hd={hd} n_ctx={n_ctx}"),
                &g.read(y),
                &want,
            );
        }
    }
}

/// attn_decode_gqa writes attn_decode's partials bit for bit, at any context: the engine picks
/// between them by length (`attn_decode_kernel`), so no result may depend on which ran. Partial chunks, one position,
/// split boundaries, both models' head shapes and heads of 96 and 224 values.
#[test]
#[ignore = "needs the GPU"]
fn the_gqa_attention_writes_attn_decodes_bits() {
    let g = Gpu::new();
    let mut rng = Rng::new(27);
    for (nh, nkv, hd) in attention_shapes() {
        for n_ctx in [1, 2, 255, 256, 257, 1000, 3073, 9000] {
            let q = rand_f32(&mut rng, nh * hd, -12.0, 12.0);
            let kc: Vec<u16> = (0..n_ctx * nkv * hd)
                .map(|_| twin::f16_bits(rng.uniform(-1.0, 1.0)))
                .collect();
            let vc: Vec<u16> = (0..n_ctx * nkv * hd)
                .map(|_| twin::f16_bits(rng.uniform(-1.0, 1.0)))
                .collect();
            let (gq, gkc, gvc) = (
                g.f32s(&q),
                g.bytes(&u16_bytes(&kc)),
                g.bytes(&u16_bytes(&vc)),
            );
            let n = nh * attn_splits(n_ctx) * (hd + 2);
            let (a, b) = (g.zeros(n), g.zeros(n));
            g.run(|kk, r| {
                for (kernel, part) in [(Kernel::AttnDecode, a), (Kernel::AttnDecodeGqa, b)] {
                    kk.attn_decode_on(
                        r, kernel, gq.addr, gkc.addr, gvc.addr, part.addr, n_ctx, nh, nkv, hd,
                    );
                }
            });
            let (a, b) = (g.read(a), g.read(b));
            let differ = a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(
                differ, 0,
                "hd={hd} n_ctx={n_ctx}: {differ} of {n} partial values differ"
            );
            assert!(
                a.iter().all(|v| v.is_finite()),
                "hd={hd} n_ctx={n_ctx}: a partial isn't finite"
            );
        }
    }
}

/// `attn_decode_on` refuses `attn_decode_gqa` for a shape it doesn't serve (heads of 48 values) before recording it,
/// after taking one it does (the tiny model's). Its buffers are sized for heads of 64 values, so a run without the
/// assert stays in bounds.
#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "attn_decode_gqa: at most")]
fn the_gqa_attention_refuses_a_shape_it_doesnt_serve() {
    let g = Gpu::new();
    let (nh, nkv, n_ctx) = (4, 1, 4);
    let (q, kc, vc) = (
        g.zeros(nh * 64),
        g.zeros(n_ctx * nkv * 64 / 2),
        g.zeros(n_ctx * nkv * 64 / 2),
    );
    let part = g.zeros(nh * attn_splits(n_ctx) * (64 + 2));
    for hd in [64, 48] {
        g.run(|kk, r| {
            kk.attn_decode_on(
                r,
                Kernel::AttnDecodeGqa,
                q.addr,
                kc.addr,
                vc.addr,
                part.addr,
                n_ctx,
                nh,
                nkv,
                hd,
            );
        });
    }
}

#[test]
#[ignore = "needs the GPU"]
fn sigmoid_mul() {
    let g = Gpu::new();
    let mut rng = Rng::new(23);
    let (a, gg) = (
        rand_f32(&mut rng, 4096, -2.0, 2.0),
        rand_f32(&mut rng, 4096, -8.0, 8.0),
    );
    let (ga, gg2, y) = (g.f32s(&a), g.f32s(&gg), g.zeros(4096));
    g.run(|k, r| k.sigmoid_mul(r, ga.addr, gg2.addr, y.addr, 4096));
    check(
        Kernel::SigmoidMul,
        "n=4096",
        &g.read(y),
        &at::sigmoid_mul(&a, &gg),
    );
}
use kernels::{
    dispatch::{
        read_status, slot_entry, ExpertStatus, SlotLayout, STATUS_BYTES, STATUS_CLEAR, WHERE_CPU,
        WHERE_HOST, WHERE_VRAM,
    },
    twin::moe,
};

#[test]
#[ignore = "needs the GPU"]
fn router_topk() {
    let g = Gpu::new();
    let mut rng = Rng::new(24);
    for (n, k) in [(256, 8), (16, 4)] {
        let logits = rand_f32(&mut rng, n, -8.0, 8.0);
        let gl = g.f32s(&logits);
        let (probs, ids, w) = (g.zeros(n), g.zeros(k), g.zeros(k));
        g.run(|kk, r| kk.router_topk(r, gl.addr, probs.addr, ids.addr, w.addr, n, k));
        let (tp, ti, tw) = moe::router_topk(&logits, k);
        check(
            Kernel::RouterTopk,
            &format!("probs n={n}"),
            &g.read(probs),
            &tp,
        );
        assert_eq!(g.read(ids), ti, "n={n}: the same experts in the same order");
        check(
            Kernel::RouterTopk,
            &format!("weights n={n}"),
            &g.read(w),
            &tw,
        );
    }
}

/// Packs `n` random experts of `lay` into one buffer (in VRAM or pinned host memory), writes a slot table for layer 1
/// of `n` experts whose entries point at them with `loc`, and returns (table, status, the slots' bytes).
fn expert_setup(
    g: &Gpu,
    rng: &mut Rng,
    lay: &SlotLayout,
    n: usize,
    loc: u32,
) -> (vk::Slice, vk::Slice, Vec<Vec<u8>>) {
    let slots: Vec<Vec<u8>> = (0..n)
        .map(|_| {
            let mut s = random_weights(rng, lay.gate, lay.n_ff, lay.n_embd);
            s.extend(random_weights(rng, lay.up, lay.n_ff, lay.n_embd));
            s.extend(random_weights(rng, lay.down, lay.n_embd, lay.n_ff));
            s
        })
        .collect();
    let packed: Vec<u8> = slots.concat();
    let at = if loc == WHERE_VRAM {
        g.bytes(&packed)
    } else {
        g.host_bytes(&packed)
    };
    // Layer 0's entries are garbage that must never be read; layer 1's point at the slots.
    let mut table = vec![0xEEu8; 2 * n * 16];
    for e in 0..n {
        let entry = slot_entry(at.addr + (e * lay.bytes()) as u64, loc);
        table[(n + e) * 16..(n + e + 1) * 16].copy_from_slice(&entry);
    }
    (g.bytes(&table), g.bytes(&STATUS_CLEAR), slots)
}

fn status_of(g: &Gpu, status: vk::Slice) -> ExpertStatus {
    let b: [u8; STATUS_BYTES] = g.dev.download(status).unwrap().try_into().unwrap();
    read_status(&b)
}

fn run_experts(
    g: &Gpu,
    table: vk::Slice,
    status: vk::Slice,
    lay: &SlotLayout,
    n: usize,
    ids: &[f32],
    x: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let k = ids.len();
    let (gids, gx) = (g.f32s(ids), g.f32s(x));
    let (act, ye) = (g.zeros(k * lay.n_ff), g.zeros(k * lay.n_embd));
    g.run(|kk, r| {
        kk.experts_gate_up(
            r,
            table.addr,
            1,
            n,
            gids.addr,
            gx.addr,
            act.addr,
            status.addr,
            lay,
            k,
            0,
            0,
            0,
        );
        r.barrier();
        kk.experts_down(
            r,
            table.addr,
            1,
            n,
            gids.addr,
            act.addr,
            ye.addr,
            status.addr,
            lay,
            k,
            0,
            0,
            0,
        );
    });
    (g.read(act), g.read(ye))
}

fn check_experts(g: &Gpu, rng: &mut Rng, lay: SlotLayout) {
    let n = 6;
    let ids = [4.0f32, 0.0, 5.0, 2.0];
    let x = rand_f32(rng, lay.n_embd, -1.0, 1.0);
    let case = format!(
        "{}/{}/{} n_embd {} n_ff {}",
        lay.gate.name(),
        lay.up.name(),
        lay.down.name(),
        lay.n_embd,
        lay.n_ff
    );
    let mut outs = Vec::new();
    for loc in [WHERE_HOST, WHERE_VRAM] {
        let place = if loc == WHERE_HOST { "HOST" } else { "VRAM" };
        let case = format!("{case} {place}");
        // The same seed for both, so the HOST and VRAM slots hold the same experts.
        let mut r2 = Rng::new(99);
        let (table, status, slots) = expert_setup(g, &mut r2, &lay, n, loc);
        let (act, ye) = run_experts(g, table, status, &lay, n, &ids, &x);
        let chosen: Vec<&[u8]> = ids.iter().map(|&e| slots[e as usize].as_slice()).collect();
        let t_act = moe::experts_gate_up(&chosen, &lay, &x);
        check(Kernel::ExpertsGateUp, &case, &act, &t_act);
        // gate_up is its twin bit for bit as well: its dots are `precise`, `tree32` adds in `red_sum`'s order (a 16-lane
        // row's first step adds +0.0, as the twin's idle lanes do), and `silu_det` is the twin's. This is the shader
        // half of AGENTS.md's HOST-path rule (the worker uses the twin's dot order and the same `silu_det`).
        assert!(
            act.iter()
                .zip(&t_act)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{case}: experts_gate_up differs from its twin"
        );
        let t_ye = moe::experts_down(&chosen, &lay, &act);
        check(Kernel::ExpertsDown, &case, &ye, &t_ye);
        // On the kernel's own activations, down is its twin bit for bit: `unit_dot`'s sums are `precise`.
        assert!(
            ye.iter()
                .zip(&t_ye)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{case}: experts_down differs from its twin"
        );
        assert_eq!(
            status_of(g, status),
            ExpertStatus::default(),
            "nothing was refused"
        );
        outs.push((act, ye));
    }
    assert!(
        outs[0] == outs[1],
        "{case}: HOST and VRAM experts must give bit-identical results"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn experts_from_host_and_vram_slots() {
    let g = Gpu::new();
    let mut rng = Rng::new(25);
    check_experts(
        &g,
        &mut rng,
        SlotLayout {
            gate: WType::Q4K,
            up: WType::Q4K,
            down: WType::Q5K,
            n_embd: 2048,
            n_ff: 512,
        },
    );
    check_experts(
        &g,
        &mut rng,
        SlotLayout {
            gate: WType::Q4K,
            up: WType::Q4K,
            down: WType::Q6K,
            n_embd: 2048,
            n_ff: 512,
        },
    );
    check_experts(
        &g,
        &mut rng,
        SlotLayout {
            gate: WType::Q4K,
            up: WType::Q4K,
            down: WType::Q6K,
            n_embd: 256,
            n_ff: 256,
        },
    );
}

#[test]
#[ignore = "needs the GPU"]
fn experts_down_lane_shapes() {
    // Down's rows of n_ff / 32 units: 64 (n_ff 2048, 32 lanes, two passes) and 24 (n_ff 768, 32 lanes with the last
    // 8 idle); gate_up's rows of n_embd / 32 units: 24 (768) and 64 (2048). check_experts holds down and gate_up to
    // their twins bit for bit, HOST and VRAM.
    let g = Gpu::new();
    let mut rng = Rng::new(28);
    for (gate, up, down, n_embd, n_ff) in [
        (WType::Q5K, WType::Q4K, WType::Q5K, 768, 2048),
        (WType::Q4K, WType::Q4K, WType::Q6K, 2048, 768),
        (WType::Q4K, WType::Q5K, WType::Q4K, 768, 1280),
    ] {
        check_experts(
            &g,
            &mut rng,
            SlotLayout {
                gate,
                up,
                down,
                n_embd,
                n_ff,
            },
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn experts_any_type_mix() {
    // Any of the six types is accepted per matrix; the expert kernels take them as push constants.
    let g = Gpu::new();
    let mut rng = Rng::new(26);
    check_experts(
        &g,
        &mut rng,
        SlotLayout {
            gate: WType::Q8_0,
            up: WType::Q5K,
            down: WType::Q4K,
            n_embd: 2048,
            n_ff: 512,
        },
    );
    check_experts(
        &g,
        &mut rng,
        SlotLayout {
            gate: WType::Bf16,
            up: WType::F32,
            down: WType::Q8_0,
            n_embd: 256,
            n_ff: 256,
        },
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_cpu_entry_is_refused_and_named() {
    let g = Gpu::new();
    let mut rng = Rng::new(27);
    let lay = SlotLayout {
        gate: WType::Q4K,
        up: WType::Q4K,
        down: WType::Q5K,
        n_embd: 2048,
        n_ff: 512,
    };
    let n = 6;
    let (table, _, _) = expert_setup(&g, &mut rng, &lay, n, WHERE_HOST);
    // Mark layer 1's experts 5 and 2 as computed on the CPU, with a null address that must never be read.
    let mut t = g.dev.download(table).unwrap();
    for e in [5, 2] {
        t[(n + e) * 16..(n + e + 1) * 16].copy_from_slice(&slot_entry(0, WHERE_CPU));
    }
    let table = g.bytes(&t);
    let x = rand_f32(&mut rng, lay.n_embd, -1.0, 1.0);
    // Both refused experts run in parallel; the status names the lower one, every time.
    for round in 0..16 {
        let status = g.bytes(&STATUS_CLEAR);
        let (act, ye) = run_experts(&g, table, status, &lay, n, &[4.0, 5.0, 2.0], &x);
        assert_eq!(
            status_of(&g, status),
            ExpertStatus {
                cpu: Some((1, 2)),
                bad: None,
            },
            "round {round}: status names layer 1's lowest CPU expert"
        );
        assert!(
            act[lay.n_ff..].iter().all(|v| v.is_nan()),
            "the refused experts' outputs are NaN"
        );
        assert!(
            act[..lay.n_ff].iter().all(|v| v.is_finite()),
            "the other expert still runs"
        );
        assert!(ye[lay.n_embd..].iter().all(|v| v.is_nan()));
    }
}

#[test]
#[ignore = "needs the GPU"]
fn entries_of_an_unknown_kind_or_without_an_address_are_refused() {
    let g = Gpu::new();
    let mut rng = Rng::new(29);
    let lay = SlotLayout {
        gate: WType::Q4K,
        up: WType::Q4K,
        down: WType::Q5K,
        n_embd: 2048,
        n_ff: 512,
    };
    let n = 6;
    let (table, _, _) = expert_setup(&g, &mut rng, &lay, n, WHERE_HOST);
    // Layer 1's expert 3 has a `where` nothing defines; expert 1 is HOST with no address.
    let mut t = g.dev.download(table).unwrap();
    let at3 = u64::from_le_bytes(t[(n + 3) * 16..(n + 3) * 16 + 8].try_into().unwrap());
    t[(n + 3) * 16..(n + 4) * 16].copy_from_slice(&slot_entry(at3, 7));
    t[(n + 1) * 16..(n + 2) * 16].copy_from_slice(&slot_entry(0, WHERE_HOST));
    let table = g.bytes(&t);
    let x = rand_f32(&mut rng, lay.n_embd, -1.0, 1.0);
    let status = g.bytes(&STATUS_CLEAR);
    let (act, _) = run_experts(&g, table, status, &lay, n, &[0.0, 3.0, 1.0], &x);
    assert_eq!(
        status_of(&g, status),
        ExpertStatus {
            cpu: None,
            bad: Some((1, 1)),
        }
    );
    assert!(act[..lay.n_ff].iter().all(|v| v.is_finite()));
    assert!(act[lay.n_ff..].iter().all(|v| v.is_nan()));
}

#[test]
#[ignore = "needs the GPU"]
fn moe_combine_and_swiglu() {
    let g = Gpu::new();
    let mut rng = Rng::new(28);
    let (n, k) = (2048, 8);
    let ye = rand_f32(&mut rng, k * n, -1.0, 1.0);
    let w = rand_f32(&mut rng, k, 0.0, 0.25);
    let sh = rand_f32(&mut rng, n, -1.0, 1.0);
    let sg = [0.7f32];
    let ins = [g.f32s(&ye), g.f32s(&w), g.f32s(&sh), g.f32s(&sg)];
    let (m, x) = (g.zeros(n), g.zeros(n));
    g.run(|kk, r| {
        kk.moe_combine(
            r,
            ins[0].addr,
            ins[1].addr,
            ins[2].addr,
            ins[3].addr,
            m.addr,
            x.addr,
            n,
            k,
        )
    });
    let (tm, tx) = moe::moe_combine(&ye, &w, &sh, sg[0], k);
    check(Kernel::MoeCombine, "moe", &g.read(m), &tm);
    check(Kernel::MoeCombine, "shexp", &g.read(x), &tx);
    let (gg, u) = (
        rand_f32(&mut rng, 512, -6.0, 6.0),
        rand_f32(&mut rng, 512, -1.0, 1.0),
    );
    let (ggg, gu, y) = (g.f32s(&gg), g.f32s(&u), g.zeros(512));
    g.run(|kk, r| kk.swiglu(r, ggg.addr, gu.addr, y.addr, 512));
    check(Kernel::Swiglu, "n=512", &g.read(y), &moe::swiglu(&gg, &u));
}

use kernels::dispatch::{slot_update, ENTRY_BYTES, UPDATE_BYTES};

#[test]
#[ignore = "needs the GPU"]
fn apply_slot_updates_is_bit_exact() {
    // The real table's shape, 41 blocks × 256 experts. 300 distinct entries change, from an update list in pinned host
    // memory as the engine keeps it, and one update past the end is skipped. Checked bit for bit, like kv_write.
    let g = Gpu::new();
    let mut rng = Rng::new(31);
    let n = 41 * 256;
    let table: Vec<u8> = (0..n)
        .flat_map(|_| slot_entry(rng.next_u64() & !0xFF, rng.next_u64() as u32 % 3))
        .collect();
    let mut order: Vec<u32> = (0..n as u32).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
    }
    let mut updates: Vec<u8> = order[..300]
        .iter()
        .flat_map(|&i| slot_update(i, rng.next_u64() as u32 % 3, rng.next_u64()))
        .collect();
    updates.extend(slot_update(n as u32, WHERE_VRAM, 0xDEAD_BEEF));
    let count = updates.len() / UPDATE_BYTES;
    let (gt, gu) = (g.bytes(&table), g.host_bytes(&updates));
    g.run(|k, r| k.apply_slot_updates(r, gt.addr, gu.addr, count, n));
    let mut want = table.clone();
    moe::apply_slot_updates(&mut want, &updates, count);
    let got = g.dev.download(gt).unwrap();
    assert_eq!(got.len(), n * ENTRY_BYTES);
    assert!(got == want, "the GPU's table differs from the twin's");
    assert!(got != table, "the updates changed the table");
}

// ---- The CPU miss path's handoff: publish, bounded wait, serve ----
use std::sync::atomic::{fence, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use kernels::dispatch::mbox;

/// The wait's bound: 20 ms in `clockRealtimeEXT` ticks, calibrated as `c_handoff`'s `calib_clock`: two stamps
/// 300 ms apart (measured at ~100.7 ticks/µs, so no hard-coding; ticks per 300 ms scale by 1/15).
fn max_ticks_20ms(g: &Gpu) -> u32 {
    let tim = g.bytes(&[0u8; 16]);
    g.run(|k, r| k.stamp(r, tim.addr, 0));
    std::thread::sleep(Duration::from_millis(300));
    g.run(|k, r| k.stamp(r, tim.addr, 1));
    let b = g.dev.download(tim).unwrap();
    let w = |i: usize| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap());
    ((w(1) - w(0)) as f64 / 15.0) as u32
}

/// The mailbox's sequence number for (token, layer): never 0.
fn seq_of(token: u32, layer: usize) -> u32 {
    token * 64 + layer as u32 + 1
}

/// An 8 MiB zeroed `AmdUncached` mailbox, as the engine's `CpuPath` allocates it.
fn test_mailbox(g: &Gpu) -> vk::Buffer {
    let b = g
        .dev
        .alloc(8 << 20, vk::MemKind::AmdUncached, "the test mailbox")
        .unwrap();
    b.write(0, &vec![0u8; 8 << 20]);
    fence(Ordering::SeqCst);
    b
}

/// The mailbox word `w` as an atomic (each flag and status word sits on its own 128-byte line).
fn mbox_word(b: &vk::Buffer, w: usize) -> &AtomicU32 {
    unsafe { &*(b.ptr().add(4 * w) as *const AtomicU32) }
}

/// The mailbox's status words: [timeout count, skip count], read after the submit.
fn mbox_status(b: &vk::Buffer) -> [u32; 2] {
    let p = unsafe { std::slice::from_raw_parts(b.ptr().add(mbox::STATUS_BYTE) as *const u32, 2) };
    [p[0], p[1]]
}

/// The handoff tests' table: `layers` layers of `n` entries each, expert `cpu` of every layer WHERE_CPU at its host
/// slot's address (the overlay keeps the host address), the rest WHERE_HOST at the same address (publish reads only
/// the `where`, and the wait reads no entry).
fn handoff_table(g: &Gpu, lay: &SlotLayout, n: usize, layers: usize, cpu: usize) -> vk::Slice {
    let slot: Vec<u8> = {
        let mut rng = Rng::new(42);
        let mut s = random_weights(&mut rng, lay.gate, lay.n_ff, lay.n_embd);
        s.extend(random_weights(&mut rng, lay.up, lay.n_ff, lay.n_embd));
        s.extend(random_weights(&mut rng, lay.down, lay.n_embd, lay.n_ff));
        s
    };
    let host = g.host_bytes(&slot);
    let mut table = vec![0u8; layers * n * 16];
    for (i, e) in table.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        let loc = if i % n == cpu { WHERE_CPU } else { WHERE_HOST };
        e.copy_from_slice(&slot_entry(host.addr, loc));
    }
    g.bytes(&table)
}

#[test]
#[ignore = "needs the GPU"]
fn a_missing_answer_costs_one_bound_and_writes_no_nan() {
    // Two layers, one command buffer: publish + wait each. A CPU thread checks what layer 0's publish wrote (the
    // header, the ids, x), answers it (the y, the protocol's fences, the done flag) and never answers layer 1. Layer 0: ye0[0] equals the thread's y bit for bit, and
    // the rest of ye0 stays 1.0 (the copy takes only the mask's bits). Layer 1: the wait hits the 20 ms bound,
    // writes NO NaN and copies nothing (the serve-2 fallback pair computes that expert from its host address),
    // counts one timeout, skips nothing. The submit costs one bound: 10..500 ms.
    let g = Gpu::new();
    let lay = SlotLayout {
        gate: WType::Q8_0,
        up: WType::Q8_0,
        down: WType::Q8_0,
        n_embd: 2048,
        n_ff: 512,
    };
    let n = 8;
    let table = handoff_table(&g, &lay, n, 2, 0); // expert 0 is the CPU entry; ids route it at j = 0
    let ids: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let x = rand_f32(&mut Rng::new(45), lay.n_embd, -1.0, 1.0);
    let (gx, gids) = (g.f32s(&x), g.f32s(&ids));
    let mbox = test_mailbox(&g);
    let (ye0, ye1) = (
        g.f32s(&vec![1.0f32; n * lay.n_embd]),
        g.f32s(&vec![1.0f32; n * lay.n_embd]),
    );
    let ticks = max_ticks_20ms(&g);
    let (seq0, seq1) = (seq_of(1, 0), seq_of(1, 1));
    let y0: Vec<f32> = (0..lay.n_embd).map(|i| i as f32 * 0.25).collect();
    let base = mbox.ptr() as usize;
    let pub0 = mbox_word(&mbox, mbox::pub_word(0));
    let done0 = mbox_word(&mbox, mbox::done_word(0));
    let status = mbox.addr() + mbox::STATUS_BYTE as u64;
    let (submit, published) = std::thread::scope(|sc| {
        let answer = sc.spawn(|| {
            let t0 = Instant::now();
            while pub0.load(Ordering::Acquire) != seq0 {
                if t0.elapsed() > Duration::from_secs(2) {
                    return Err("the GPU never published layer 0".to_string()); // the wait times out either way
                }
                std::hint::spin_loop();
            }
            // What publish wrote, read after the Acquire as the coordinator reads it: the header (seq, token,
            // layer, mask), the ids' raw f32 bits, and x's bits. A wrong offset, id word or mask, or a publish
            // flag that lands before its stores, shows here.
            let word = |w: usize| unsafe { std::ptr::read_volatile((base + 4 * w) as *const u32) };
            let hw = mbox::X0 / 4;
            let header = [word(hw), word(hw + 1), word(hw + 2), word(hw + 3)];
            let checked = if header != [seq0, 1, 0, 0b1] {
                Err(format!("header {header:?}"))
            } else if let Some(j) = (0..n).find(|&j| word(hw + 8 + j) != ids[j].to_bits()) {
                Err(format!("id word {j}: {:#x}", word(hw + 8 + j)))
            } else if let Some(i) = (0..lay.n_embd).find(|&i| word(hw + 64 + i) != x[i].to_bits()) {
                Err(format!("x[{i}]: {:#x}", word(hw + 64 + i)))
            } else {
                Ok(())
            };
            // Answer layer 0: y for the mask's bit 0, then the done flag with the protocol's fences.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    y0.as_ptr(),
                    (base + mbox::Y0) as *mut f32,
                    lay.n_embd,
                );
            }
            fence(Ordering::SeqCst);
            done0.store(seq0, Ordering::Release);
            fence(Ordering::SeqCst);
            checked
        });
        let t0 = Instant::now();
        g.run(|k, r| {
            k.miss_publish(
                r,
                gx.addr,
                gids.addr,
                table.addr,
                mbox.addr(),
                0,
                n,
                seq0,
                1,
                lay.n_embd,
                n,
            );
            r.barrier();
            k.miss_wait(
                r,
                mbox.addr(),
                ye0.addr,
                status,
                0,
                seq0,
                lay.n_embd,
                4_000_000,
                ticks,
            );
            r.barrier();
            k.miss_publish(
                r,
                gx.addr,
                gids.addr,
                table.addr,
                mbox.addr(),
                1,
                n,
                seq1,
                1,
                lay.n_embd,
                n,
            );
            r.barrier();
            k.miss_wait(
                r,
                mbox.addr(),
                ye1.addr,
                status,
                1,
                seq1,
                lay.n_embd,
                4_000_000,
                ticks,
            );
        });
        (t0.elapsed(), answer.join().unwrap())
    });
    published.expect("what publish wrote");
    let got0 = g.read(ye0);
    for i in 0..lay.n_embd {
        assert_eq!(got0[i].to_bits(), y0[i].to_bits(), "ye0[{i}]");
    }
    for v in &got0[lay.n_embd..] {
        assert_eq!(
            v.to_bits(),
            1.0f32.to_bits(),
            "only the mask's bits are copied"
        );
    }
    let got1 = g.read(ye1);
    assert!(
        got1.iter()
            .all(|v| !v.is_nan() && v.to_bits() == 1.0f32.to_bits()),
        "a timeout writes no NaN and copies nothing"
    );
    assert_eq!(mbox_status(&mbox), [1, 0], "one timeout, no skips");
    let ms = submit.as_millis();
    assert!(
        (10..500).contains(&ms),
        "one 20 ms bound, not zero and not two: {submit:?}"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn the_wait_skips_when_an_earlier_layer_timed_out() {
    // Three layers, no answerer: layer 0's wait hits the bound and counts one timeout; layers 1 and 2 see the
    // status set and give up at once (two skips). One bound a token: the counts and no NaN are what's asserted
    // (a time that separates one bound from two has calibration slack). The submit stays under 500 ms.
    let g = Gpu::new();
    let lay = SlotLayout {
        gate: WType::Q8_0,
        up: WType::Q8_0,
        down: WType::Q8_0,
        n_embd: 2048,
        n_ff: 512,
    };
    let n = 8;
    let table = handoff_table(&g, &lay, n, 3, 0);
    let ids: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let x = rand_f32(&mut Rng::new(45), lay.n_embd, -1.0, 1.0);
    let (gx, gids) = (g.f32s(&x), g.f32s(&ids));
    let mbox = test_mailbox(&g);
    let ye = [
        g.f32s(&vec![1.0f32; n * lay.n_embd]),
        g.f32s(&vec![1.0f32; n * lay.n_embd]),
        g.f32s(&vec![1.0f32; n * lay.n_embd]),
    ];
    let ticks = max_ticks_20ms(&g);
    let status = mbox.addr() + mbox::STATUS_BYTE as u64;
    let t0 = Instant::now();
    g.run(|k, r| {
        for (l, ye) in ye.iter().enumerate() {
            let seq = seq_of(1, l);
            k.miss_publish(
                r,
                gx.addr,
                gids.addr,
                table.addr,
                mbox.addr(),
                l,
                n,
                seq,
                1,
                lay.n_embd,
                n,
            );
            r.barrier();
            k.miss_wait(
                r,
                mbox.addr(),
                ye.addr,
                status,
                l,
                seq,
                lay.n_embd,
                4_000_000,
                ticks,
            );
            r.barrier();
        }
    });
    let submit = t0.elapsed();
    assert_eq!(mbox_status(&mbox), [1, 2], "one timeout, two skips");
    for ye in &ye {
        let got = g.read(*ye);
        assert!(
            got.iter()
                .all(|v| !v.is_nan() && v.to_bits() == 1.0f32.to_bits()),
            "no NaN anywhere"
        );
    }
    assert!(
        submit.as_millis() < 500,
        "one bound plus two skips: {submit:?}"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_zero_mask_does_not_spin() {
    // A layer whose routed set holds no CPU entry: the publish writes mask 0, and the wait returns at once
    // without polling: no answerer exists. No timeout counted, ye untouched, the submit fast.
    let g = Gpu::new();
    let lay = SlotLayout {
        gate: WType::Q8_0,
        up: WType::Q8_0,
        down: WType::Q8_0,
        n_embd: 2048,
        n_ff: 512,
    };
    let n = 8;
    let table = handoff_table(&g, &lay, n, 1, usize::MAX); // every entry HOST: the mask is 0
    let ids: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let x = rand_f32(&mut Rng::new(45), lay.n_embd, -1.0, 1.0);
    let (gx, gids) = (g.f32s(&x), g.f32s(&ids));
    let mbox = test_mailbox(&g);
    let ye = g.f32s(&vec![1.0f32; n * lay.n_embd]);
    let ticks = max_ticks_20ms(&g);
    let seq = seq_of(1, 0);
    let status = mbox.addr() + mbox::STATUS_BYTE as u64;
    let t0 = Instant::now();
    g.run(|k, r| {
        k.miss_publish(
            r,
            gx.addr,
            gids.addr,
            table.addr,
            mbox.addr(),
            0,
            n,
            seq,
            1,
            lay.n_embd,
            n,
        );
        r.barrier();
        k.miss_wait(
            r,
            mbox.addr(),
            ye.addr,
            status,
            0,
            seq,
            lay.n_embd,
            4_000_000,
            ticks,
        );
    });
    let submit = t0.elapsed();
    assert_eq!(mbox_status(&mbox), [0, 0], "no timeout, no skip");
    let got = g.read(ye);
    assert!(
        got.iter().all(|v| v.to_bits() == 1.0f32.to_bits()),
        "a zero mask copies nothing"
    );
    assert!(submit.as_millis() < 50, "no spin: {submit:?}");
}

/// The value the serve tests pre-fill `act` and `ye` with: a NaN payload no kernel writes, so a dispatch that writes
/// nothing is visible, and one that writes anything is too.
const UNWRITTEN: u32 = 0x7FC0_0123;

/// One Q8_0 expert, routed at j = 0 of layer 1 (layer_base 2), its entry WHERE_CPU at the host slot's address, run
/// through `experts_gate_up` + `experts_down` at `serve` against a fresh mailbox holding `done` in layer 1's done
/// flag and `timeouts` in the status's timeout count. Returns (act, ye, the expert status).
fn serve_pair(
    g: &Gpu,
    lay: &SlotLayout,
    host: u64,
    x: &[f32],
    serve: u32,
    done: u32,
    timeouts: u32,
) -> (Vec<f32>, Vec<f32>, ExpertStatus) {
    let mut table = vec![0xEEu8; 4 * 16];
    table[32..48].copy_from_slice(&slot_entry(host, WHERE_CPU));
    let table = g.bytes(&table);
    let (gx, gids) = (g.f32s(x), g.f32s(&[0.0]));
    let (act, ye) = (
        g.f32s(&vec![f32::from_bits(UNWRITTEN); lay.n_ff]),
        g.f32s(&vec![f32::from_bits(UNWRITTEN); lay.n_embd]),
    );
    let status = g.bytes(&STATUS_CLEAR);
    let mbox = test_mailbox(g);
    mbox_word(&mbox, mbox::done_word(1)).store(done, Ordering::SeqCst);
    mbox_word(&mbox, mbox::STATUS_BYTE / 4).store(timeouts, Ordering::SeqCst);
    let (seq, m) = (seq_of(1, 1), if serve == 2 { mbox.addr() } else { 0 });
    g.run(|k, r| {
        k.experts_gate_up(
            r,
            table.addr,
            1,
            2,
            gids.addr,
            gx.addr,
            act.addr,
            status.addr,
            lay,
            1,
            serve,
            m,
            seq,
        );
        r.barrier();
        k.experts_down(
            r,
            table.addr,
            1,
            2,
            gids.addr,
            act.addr,
            ye.addr,
            status.addr,
            lay,
            1,
            serve,
            m,
            seq,
        );
    });
    (g.read(act), g.read(ye), status_of(g, status))
}

fn q8_expert(seed: u64, lay: &SlotLayout) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut s = random_weights(&mut rng, lay.gate, lay.n_ff, lay.n_embd);
    s.extend(random_weights(&mut rng, lay.up, lay.n_ff, lay.n_embd));
    s.extend(random_weights(&mut rng, lay.down, lay.n_embd, lay.n_ff));
    s
}

const SERVED: ExpertStatus = ExpertStatus {
    cpu: None,
    bad: None,
};

#[test]
#[ignore = "needs the GPU"]
fn serve_two_matches_the_worker_on_a_denormal_x() {
    // The fallback pair's bits: the mailbox's done flag is not seq, so serve 2 computes the WHERE_CPU expert from
    // its host address, and ye must equal cpu_expert::with_ftz's expert bit for bit. Two x's: a normal one (the
    // GPU's silu_det against the worker's; a shader still calling silu differs by an ulp on some rows), and one
    // scaled so the activation products silu_det(gate·x)·(up·x) straddle the normal/denormal boundary, where
    // the GPU's plain shaders flush those and the worker matches only inside with_ftz. That second x
    // must make a default-MXCSR worker differ (else it tests nothing about FTZ), and both must give a y that isn't
    // all zero. (Not x[0] = 1e-40: a single denormal input rounds away in the row dot.)
    let g = Gpu::new();
    let lay = SlotLayout {
        gate: WType::Q8_0,
        up: WType::Q8_0,
        down: WType::Q8_0,
        n_embd: 2048,
        n_ff: 512,
    };
    let slot = q8_expert(46, &lay);
    let host = g.host_bytes(&slot);
    let normal = rand_f32(&mut Rng::new(47), lay.n_embd, -1.0, 1.0);
    let edge: Vec<f32> = normal.iter().map(|v| v * 1e-18).collect();
    for (name, x) in [("normal", &normal), ("edge", &edge)] {
        let (_, ye, st) = serve_pair(&g, &lay, host.addr, x, 2, 0, 0);
        assert_eq!(st, SERVED, "{name}: served, not refused");
        let (mut wact, mut wy) = (vec![0f32; lay.n_ff], vec![0f32; lay.n_embd]);
        cpu_expert::with_ftz(|| cpu_expert::expert(&slot, &lay, x, &mut wact, &mut wy));
        let nonzero = wy.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero > lay.n_embd / 2,
            "{name}: the fixture's y is mostly zero ({nonzero} non-zero)"
        );
        for i in 0..lay.n_embd {
            assert_eq!(ye[i].to_bits(), wy[i].to_bits(), "{name}: ye[{i}]");
        }
        if name == "edge" {
            let denormal = wact.iter().filter(|v| **v == 0.0).count();
            let (mut dact, mut dy) = (vec![0f32; lay.n_ff], vec![0f32; lay.n_embd]);
            cpu_expert::expert(&slot, &lay, x, &mut dact, &mut dy);
            let differ = (0..lay.n_embd)
                .filter(|&i| dy[i].to_bits() != wy[i].to_bits())
                .count();
            assert!(
                differ > 0,
                "edge: the default-MXCSR worker matches the FTZ one ({denormal} flushed acts): the x tests no FTZ"
            );
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn serve_two_writes_nothing_after_a_good_wait_and_ignores_a_late_flag() {
    // serve 2 after a wait that succeeded (done flag == seq, no timeout this token): the wait copied the CPU's y,
    // so the pair writes nothing: act and ye keep their pre-fill. With the done flag == seq but this token's
    // timeout count at 1 (the CPU caught up after a failed wait): a late answer is wasted, never trusted: the
    // pair computes from the host address, bit for bit with the worker. serve 1 on the same entry writes +0 into
    // act and ye and refuses nothing.
    let g = Gpu::new();
    let lay = SlotLayout {
        gate: WType::Q8_0,
        up: WType::Q8_0,
        down: WType::Q8_0,
        n_embd: 2048,
        n_ff: 512,
    };
    let slot = q8_expert(48, &lay);
    let host = g.host_bytes(&slot);
    let x = rand_f32(&mut Rng::new(49), lay.n_embd, -1.0, 1.0);
    let seq = seq_of(1, 1);

    let (act, ye, st) = serve_pair(&g, &lay, host.addr, &x, 2, seq, 0);
    assert_eq!(st, SERVED);
    assert!(
        act.iter().all(|v| v.to_bits() == UNWRITTEN),
        "a good wait: act untouched"
    );
    assert!(
        ye.iter().all(|v| v.to_bits() == UNWRITTEN),
        "a good wait: ye untouched"
    );

    let (_, ye, st) = serve_pair(&g, &lay, host.addr, &x, 2, seq, 1);
    assert_eq!(st, SERVED);
    let (mut wact, mut wy) = (vec![0f32; lay.n_ff], vec![0f32; lay.n_embd]);
    cpu_expert::with_ftz(|| cpu_expert::expert(&slot, &lay, &x, &mut wact, &mut wy));
    for i in 0..lay.n_embd {
        assert_eq!(
            ye[i].to_bits(),
            wy[i].to_bits(),
            "a late flag after a timeout: ye[{i}]"
        );
    }

    let (act, ye, st) = serve_pair(&g, &lay, host.addr, &x, 1, 0, 0);
    assert_eq!(st, SERVED, "serve 1 skips a CPU entry without refusing it");
    assert!(
        act.iter().chain(&ye).all(|v| v.to_bits() == 0),
        "serve 1 writes +0"
    );
}

// ---- argmax_rows ----
use kernels::dispatch::NO_DRAFT;

fn u32s_of(b: &[u8]) -> Vec<u32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn u64s_of(b: &[u8]) -> Vec<u64> {
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect()
}

#[test]
#[ignore = "needs the GPU"]
fn argmax_rows_matches_its_twin() {
    let g = Gpu::new();
    let mut rng = Rng::new(41);
    let (embd, row_bytes) = (0x3_0000_1000_u64, 1usize << 20);
    for n in [512usize, 248_320] {
        let mut rows: Vec<Vec<f32>> = Vec::new();
        for _ in 0..6 {
            rows.push(rand_f32(&mut rng, n, -8.0, 8.0));
        }
        // Planted ties at the maximum, across threads' strided shares (1,024 apart) and within one.
        let m = 9.0f32;
        for i in [n - 1, n / 2, 3, 3 + 1024.min(n - 4)] {
            rows[1][i] = m;
        }
        // NaNs, including at index 0 and on the maximum's side; infinities never win.
        rows[2][0] = f32::NAN;
        rows[2][n / 3] = f32::NAN;
        rows[2][7] = 8.5;
        rows[3][5] = f32::INFINITY;
        rows[3][9] = f32::NEG_INFINITY;
        rows[3][n - 2] = 8.25;
        // One all-NaN row, and one of -inf and +inf only.
        rows[4] = vec![f32::NAN; n];
        rows[5] = (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    f32::INFINITY
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        rows[0][n - 1] = rows[0].iter().cloned().fold(f32::MIN, f32::max); // a tie at the end
        for (ri, row) in rows.iter().enumerate() {
            let gl = g.f32s(row);
            let ids = g.bytes(&[0xAA; 4]);
            let out = g.bytes(&[0xBB; 8]);
            g.run(|k, r| k.argmax_rows(r, gl.addr, n, ids.addr, out.addr, embd, row_bytes, 0));
            let want = twin::argmax_row(row);
            let id = u32s_of(&g.dev.download(ids).unwrap())[0];
            let addr = u64s_of(&g.dev.download(out).unwrap())[0];
            assert_eq!(id, want.unwrap_or(NO_DRAFT), "n={n} row {ri}");
            assert!(addr >> 32 >= 3, "n={n} row {ri}: the high bits survive");
            assert_eq!(
                addr,
                embd + want.unwrap_or(0) as u64 * row_bytes as u64,
                "n={n} row {ri}: the next row's address"
            );
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn argmax_rows_writes_only_its_slot() {
    let g = Gpu::new();
    let mut rng = Rng::new(42);
    let n = 1000;
    let row = rand_f32(&mut rng, n, -3.0, 3.0);
    let gl = g.f32s(&row);
    let ids = g.bytes(&[0xAA; 16]);
    let out = g.bytes(&[0xBB; 32]);
    g.run(|k, r| k.argmax_rows(r, gl.addr, n, ids.addr, out.addr, 0x4_0000_2000, 1 << 20, 2));
    let (i, o) = (
        u32s_of(&g.dev.download(ids).unwrap()),
        u64s_of(&g.dev.download(out).unwrap()),
    );
    let want = twin::argmax_row(&row).unwrap();
    assert_eq!(i[2], want);
    assert_eq!(o[2], 0x4_0000_2000 + want as u64 * (1 << 20));
    assert!(o[2] >> 32 >= 4, "the high bits survive");
    for s in [0, 1, 3] {
        assert_eq!(i[s], 0xAAAA_AAAA, "ids[{s}] untouched");
        assert_eq!(o[s], 0xBBBB_BBBB_BBBB_BBBB, "rows[{s}] untouched");
    }
}
