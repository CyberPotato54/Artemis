//! Test 1 on the GPU for the prefill kernels: each kernel against its twin on random inputs, at the
//! real model's shapes and the tiny model's, within the kernel's stated tolerance. Many tokens at once, with counts
//! that leave the last tile part full.
use kernels::{
    dispatch::MatW,
    testing::{check, identity, rand_f32, random_weights, Gpu},
    twin, Kernel,
};
use qwen36::{quant::WType, testing::Rng};

/// Whether two f32 slices hold the same bits (so NaNs and signed zeros compare as what they are, not as `==` would).
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Runs `matmul_{ty}` on the GPU over `tokens` inputs and returns `[tokens][rows]`.
fn gpu_matmul(g: &Gpu, ty: WType, w: &[u8], rows: usize, x: &[f32], tokens: usize) -> Vec<f32> {
    let (gw, gx, y) = (g.bytes(w), g.f32s(x), g.zeros(tokens * rows));
    let m = MatW {
        addr: gw.addr,
        ty,
        rows,
        cols: x.len() / tokens,
    };
    g.run(|k, r| k.matmul(r, m, gx.addr, y.addr, tokens));
    g.read(y)
}

/// Runs `matmul_cm_{ty}` on the GPU over `tokens` inputs and returns `[tokens][rows]`.
fn gpu_matmul_cm(g: &Gpu, ty: WType, w: &[u8], rows: usize, x: &[f32], tokens: usize) -> Vec<f32> {
    let (gw, gx, y) = (g.bytes(w), g.f32s(x), g.zeros(tokens * rows));
    let m = MatW {
        addr: gw.addr,
        ty,
        rows,
        cols: x.len() / tokens,
    };
    g.run(|k, r| k.matmul_cm(r, m, gx.addr, y.addr, tokens));
    g.read(y)
}

#[test]
#[ignore = "needs the GPU"]
fn matmul_cm_every_type_and_shape() {
    // matmul_every_type_and_shape's shapes through the split-f16 cooperative-matrix kernel, against its twin, and two
    // of k = 4,096, the widest k prefill gives it (attn_output and ssm_out).
    let g = Gpu::new();
    assert!(
        g.k.coop(),
        "the RX 9070 XT runs the cooperative-matrix kernels"
    );
    let mut rng = Rng::new(46);
    let shapes = [
        (1, 2048, 70),
        (37, 2048, 64),
        (256, 2048, 130),
        (512, 2048, 33),
        (2048, 512, 65),
        (256, 256, 3),
        (8192, 2048, 64),
        (64, 4096, 33),
        (2048, 4096, 65),
    ];
    for ty in WType::ALL {
        for (rows, cols, tokens) in shapes {
            let w = random_weights(&mut rng, ty, rows, cols);
            let x = rand_f32(&mut rng, tokens * cols, -1.0, 1.0);
            check(
                Kernel::matmul_cm(ty),
                &format!("{} {rows}x{cols}, {tokens} tokens", ty.name()),
                &gpu_matmul_cm(&g, ty, &w, rows, &x, tokens),
                &twin::matvec::matmul_cm(ty, &w, rows, &x, tokens),
            );
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn a_matmul_cms_outputs_dont_depend_on_the_batch() {
    // For the cooperative-matrix kernel: each token's outputs in a batch of 130 (the last 64-token
    // tile part full), alone, and in batches of 64 from token 40 and of 50 from token 17 (so a token sits in another row
    // of its tile and of its 16-row block): bit-identical.
    let g = Gpu::new();
    let mut rng = Rng::new(47);
    let (rows, cols, tokens) = (300, 2048, 130);
    for ty in [WType::Q8_0, WType::Q6K, WType::F32] {
        let w = random_weights(&mut rng, ty, rows, cols);
        let x = rand_f32(&mut rng, tokens * cols, -1.0, 1.0);
        let all = gpu_matmul_cm(&g, ty, &w, rows, &x, tokens);
        for t in [0, 15, 16, 63, 64, 129] {
            let one = gpu_matmul_cm(&g, ty, &w, rows, &x[t * cols..(t + 1) * cols], 1);
            assert!(
                same_bits(&one, &all[t * rows..(t + 1) * rows]),
                "{} token {t} alone",
                ty.name()
            );
        }
        for (from, n) in [(40, 64), (17, 50)] {
            let part = gpu_matmul_cm(&g, ty, &w, rows, &x[from * cols..(from + n) * cols], n);
            assert!(
                same_bits(&part, &all[from * rows..(from + n) * rows]),
                "{}: tokens {from}..{} as a batch of their own",
                ty.name(),
                from + n
            );
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn matmul_cm_is_within_split_precision_of_the_f32_kernel() {
    // The split keeps prefill near f32 (decision of 2026-10-10): on the same inputs the cooperative-matrix kernel is within
    // 3e-5 of today's f32 kernel. Plain f16 inputs measured 2.7e-4, so dropping the lo products
    // fails this.
    let g = Gpu::new();
    let mut rng = Rng::new(48);
    let (rows, cols, tokens) = (2048, 2048, 64);
    for ty in [WType::Q8_0, WType::Q4K] {
        let w = random_weights(&mut rng, ty, rows, cols);
        let x = rand_f32(&mut rng, tokens * cols, -1.0, 1.0);
        let cm = gpu_matmul_cm(&g, ty, &w, rows, &x, tokens);
        let f32k = gpu_matmul(&g, ty, &w, rows, &x, tokens);
        let e = qwen36::ops::rel_l2(&cm, &f32k);
        assert!(e <= 3e-5, "{}: {e:.3e} from the f32 kernel", ty.name());
    }
}

#[test]
#[ignore = "needs the GPU"]
fn matmul_every_type_and_shape() {
    let g = Gpu::new();
    let mut rng = Rng::new(42);
    // Prefill's shapes, with token counts that leave tiles part full: one row (the shared-expert gate), a ragged row
    // count, the router, expert gate/up and down, the tiny model's, and a DeltaNet QKV projection.
    let shapes = [
        (1, 2048, 70),
        (37, 2048, 64),
        (256, 2048, 130),
        (512, 2048, 33),
        (2048, 512, 65),
        (256, 256, 3),
        (8192, 2048, 64),
    ];
    for ty in WType::ALL {
        for (rows, cols, tokens) in shapes {
            let w = random_weights(&mut rng, ty, rows, cols);
            let x = rand_f32(&mut rng, tokens * cols, -1.0, 1.0);
            check(
                Kernel::matmul(ty),
                &format!("{} {rows}x{cols}, {tokens} tokens", ty.name()),
                &gpu_matmul(&g, ty, &w, rows, &x, tokens),
                &twin::matvec::matmul(ty, &w, rows, &x, tokens),
            );
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn a_matmuls_outputs_dont_depend_on_the_batch() {
    // Each token's outputs, computed in a batch of 130 (three token tiles), alone, and in a batch of 64 of them:
    // bit-identical, so prefill's results don't depend on how a prompt is cut into chunks.
    let g = Gpu::new();
    let mut rng = Rng::new(43);
    let (rows, cols, tokens) = (300, 2048, 130);
    for ty in [WType::Q8_0, WType::Q6K, WType::F32] {
        let w = random_weights(&mut rng, ty, rows, cols);
        let x = rand_f32(&mut rng, tokens * cols, -1.0, 1.0);
        let all = gpu_matmul(&g, ty, &w, rows, &x, tokens);
        for t in [0, 63, 64, 129] {
            let one = gpu_matmul(&g, ty, &w, rows, &x[t * cols..(t + 1) * cols], 1);
            assert!(
                same_bits(&one, &all[t * rows..(t + 1) * rows]),
                "{} token {t} alone",
                ty.name()
            );
        }
        let part = gpu_matmul(&g, ty, &w, rows, &x[40 * cols..104 * cols], 64);
        assert!(
            same_bits(&part, &all[40 * rows..104 * rows]),
            "{}: tokens 40..104 as a batch of their own",
            ty.name()
        );
    }
}

use kernels::twin::{attention as at, deltanet as dn, moe};
use qwen36::dims::Dims;

#[test]
#[ignore = "needs the GPU"]
fn rmsnorm_per_token() {
    let g = Gpu::new();
    let mut rng = Rng::new(44);
    for (n, tokens) in [(2048, 130), (256, 5)] {
        let x = rand_f32(&mut rng, tokens * n, -2.0, 2.0);
        let w = rand_f32(&mut rng, n, 0.8, 1.2);
        let (gx, gw, y) = (g.f32s(&x), g.f32s(&w), g.zeros(tokens * n));
        g.run(|k, r| k.rmsnorm_tokens(r, gx.addr, gw.addr, y.addr, n, 1e-6, tokens));
        let want: Vec<f32> = x
            .chunks(n)
            .flat_map(|xt| twin::rmsnorm(xt, &w, 1e-6))
            .collect();
        check(
            Kernel::Rmsnorm,
            &format!("n={n}, {tokens} tokens"),
            &g.read(y),
            &want,
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn gdn_gates_per_token() {
    let g = Gpu::new();
    let mut rng = Rng::new(45);
    let (n, tokens) = (32, 130);
    let (b, alpha) = (
        rand_f32(&mut rng, tokens * n, -6.0, 6.0),
        rand_f32(&mut rng, tokens * n, -8.0, 8.0),
    );
    let (dt, a) = (
        rand_f32(&mut rng, n, -2.0, 2.0),
        rand_f32(&mut rng, n, -1.5, -0.1),
    );
    let ins = [g.f32s(&b), g.f32s(&alpha), g.f32s(&dt), g.f32s(&a)];
    let (beta, gg) = (g.zeros(tokens * n), g.zeros(tokens * n));
    g.run(|k, r| {
        k.gdn_gates_tokens(
            r,
            ins[0].addr,
            ins[1].addr,
            ins[2].addr,
            ins[3].addr,
            beta.addr,
            gg.addr,
            n,
            tokens,
        )
    });
    let (mut bt, mut gt) = (Vec::new(), Vec::new());
    for (bb, aa) in b.chunks(n).zip(alpha.chunks(n)) {
        let (x, y) = dn::gdn_gates(bb, aa, &dt, &a);
        bt.extend(x);
        gt.extend(y);
    }
    check(Kernel::GdnGates, "beta, 130 tokens", &g.read(beta), &bt);
    check(Kernel::GdnGates, "g, 130 tokens", &g.read(gg), &gt);
}

#[test]
#[ignore = "needs the GPU"]
fn l2norm_per_token_leaves_the_rest_of_each_row_alone() {
    // q's and k's key groups are the first 2 × n_group vectors of each token's conv row; v, after them, stays.
    let g = Gpu::new();
    let mut rng = Rng::new(46);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (stride, dk, groups, tokens) = (d.conv_dim(), d.ssm_d_state, 2 * d.ssm_n_group, 70);
        let x = rand_f32(&mut rng, tokens * stride, -1.0, 1.0);
        let gx = g.f32s(&x);
        g.run(|k, r| k.l2norm_tokens(r, gx.addr, groups, dk, 1e-6, tokens, stride));
        let mut want = x.clone();
        for row in want.chunks_mut(stride) {
            dn::l2norm(&mut row[..groups * dk], dk, 1e-6);
        }
        let got = g.read(gx);
        check(
            Kernel::L2norm,
            &format!("dk={dk}, {tokens} tokens"),
            &got,
            &want,
        );
        for (t, (a, b)) in got.chunks(stride).zip(want.chunks(stride)).enumerate() {
            assert!(
                same_bits(&a[groups * dk..], &b[groups * dk..]),
                "token {t}: v changed"
            );
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn qk_norm_rope_per_token_at_its_own_position() {
    let g = Gpu::new();
    let mut rng = Rng::new(47);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (hd, nh, nkv, tokens) = (d.head_dim, d.n_head, d.n_head_kv, 40);
        let pos0 = 60_000u32;
        let qfull = rand_f32(&mut rng, tokens * 2 * nh * hd, -2.0, 2.0);
        let k = rand_f32(&mut rng, tokens * nkv * hd, -2.0, 2.0);
        let (qn, kn) = (
            rand_f32(&mut rng, hd, 0.5, 1.5),
            rand_f32(&mut rng, hd, 0.5, 1.5),
        );
        let tables: Vec<f32> = (0..tokens as u32)
            .flat_map(|t| at::rope_table(pos0 + t, d.n_rot, d.rope_base))
            .collect();
        let (gqf, gk, gqn, gkn, gt) = (
            g.f32s(&qfull),
            g.f32s(&k),
            g.f32s(&qn),
            g.f32s(&kn),
            g.f32s(&tables),
        );
        let (q, gate) = (g.zeros(tokens * nh * hd), g.zeros(tokens * nh * hd));
        g.run(|kk, r| {
            kk.qk_norm_rope_tokens(
                r, gqf.addr, gk.addr, q.addr, gate.addr, gqn.addr, gkn.addr, gt.addr, nh, nkv, hd,
                d.n_rot, d.rms_eps, tokens,
            )
        });
        let (mut qw, mut kw, mut gw) = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..tokens {
            let mut kt = k[t * nkv * hd..(t + 1) * nkv * hd].to_vec();
            let (qt, gt) = at::qk_norm_rope(
                &qfull[t * 2 * nh * hd..(t + 1) * 2 * nh * hd],
                &mut kt,
                &qn,
                &kn,
                &tables[t * d.n_rot..(t + 1) * d.n_rot],
                nh,
                hd,
                d.rms_eps,
            );
            qw.extend(qt);
            kw.extend(kt);
            gw.extend(gt);
        }
        let case = |what: &str| format!("{what} hd={hd}, {tokens} tokens");
        check(Kernel::QkNormRope, &case("q"), &g.read(q), &qw);
        check(Kernel::QkNormRope, &case("k"), &g.read(gk), &kw);
        assert!(same_bits(&g.read(gate), &gw), "the gate is copied exactly");
    }
}

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

#[test]
#[ignore = "needs the GPU"]
fn kv_write_of_a_batch_is_bit_exact() {
    // 70 positions from 100 into a 256-position cache: the rows before and after stay as they were.
    let g = Gpu::new();
    let mut rng = Rng::new(48);
    for width in [512, 300] {
        let (ctx, pos, tokens) = (256, 100, 70);
        let k = rand_f32(&mut rng, tokens * width, -4.0, 4.0);
        let v = rand_f32(&mut rng, tokens * width, -4.0, 4.0);
        let (gk, gv) = (g.f32s(&k), g.f32s(&v));
        let (kc, vc) = (g.zeros(ctx * width / 2), g.zeros(ctx * width / 2));
        g.run(|kk, r| {
            kk.kv_write_tokens(r, gk.addr, gv.addr, kc.addr, vc.addr, pos, width, tokens)
        });
        let (mut tk, mut tv) = (vec![0u16; ctx * width], vec![0u16; ctx * width]);
        for t in 0..tokens {
            let at = t * width..(t + 1) * width;
            at::kv_write(&k[at.clone()], &v[at], &mut tk, &mut tv, pos + t, width);
        }
        assert_eq!(read_u16(&g, kc), tk);
        assert_eq!(read_u16(&g, vc), tv);
    }
}

#[test]
#[ignore = "needs the GPU"]
fn router_topk_per_token() {
    let g = Gpu::new();
    let mut rng = Rng::new(49);
    for (n, k, tokens) in [(256, 8, 130), (16, 4, 9)] {
        let logits = rand_f32(&mut rng, tokens * n, -4.0, 4.0);
        let gl = g.f32s(&logits);
        let (probs, ids, w) = (
            g.zeros(tokens * n),
            g.zeros(tokens * k),
            g.zeros(tokens * k),
        );
        g.run(|kk, r| {
            kk.router_topk_tokens(r, gl.addr, probs.addr, ids.addr, w.addr, n, k, tokens)
        });
        let (mut pt, mut it, mut wt) = (Vec::new(), Vec::new(), Vec::new());
        for lt in logits.chunks(n) {
            let (p, i, ww) = moe::router_topk(lt, k);
            pt.extend(p);
            it.extend(i);
            wt.extend(ww);
        }
        let case = |what: &str| format!("{what} n={n}, {tokens} tokens");
        check(Kernel::RouterTopk, &case("probs"), &g.read(probs), &pt);
        assert_eq!(g.read(ids), it, "n={n}: the chosen experts");
        check(Kernel::RouterTopk, &case("weights"), &g.read(w), &wt);
    }
}

#[test]
#[ignore = "needs the GPU"]
fn moe_combine_per_token() {
    let g = Gpu::new();
    let mut rng = Rng::new(50);
    for n in [2048, 300] {
        let (k, tokens) = (8, 70);
        let ye = rand_f32(&mut rng, tokens * k * n, -1.0, 1.0);
        let w = rand_f32(&mut rng, tokens * k, 0.0, 0.3);
        let sh = rand_f32(&mut rng, tokens * n, -1.0, 1.0);
        let sg = rand_f32(&mut rng, tokens, -3.0, 3.0);
        let (gye, gw, gsh, gsg) = (g.f32s(&ye), g.f32s(&w), g.f32s(&sh), g.f32s(&sg));
        let (m, s) = (g.zeros(tokens * n), g.zeros(tokens * n));
        g.run(|kk, r| {
            kk.moe_combine_tokens(
                r, gye.addr, gw.addr, gsh.addr, gsg.addr, m.addr, s.addr, n, k, tokens,
            )
        });
        let (mut mt, mut st) = (Vec::new(), Vec::new());
        for t in 0..tokens {
            let (a, b) = moe::moe_combine(
                &ye[t * k * n..(t + 1) * k * n],
                &w[t * k..(t + 1) * k],
                &sh[t * n..(t + 1) * n],
                sg[t],
                k,
            );
            mt.extend(a);
            st.extend(b);
        }
        check(
            Kernel::MoeCombine,
            &format!("moe n={n}, 70 tokens"),
            &g.read(m),
            &mt,
        );
        check(
            Kernel::MoeCombine,
            &format!("shexp n={n}, 70 tokens"),
            &g.read(s),
            &st,
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn embed_rows_every_type_from_host_memory() {
    // Rows 9, 2, 2 and 0 of a 12-row table: each token's row, from its own address.
    let g = Gpu::new();
    let mut rng = Rng::new(51);
    let n = 2048;
    let picks = [9usize, 2, 2, 0];
    for ty in WType::ALL {
        let table = random_weights(&mut rng, ty, 12, n);
        let rb = ty.row_bytes(n);
        let host = g.host_bytes(&table);
        let addrs: Vec<u8> = picks
            .iter()
            .flat_map(|&p| (host.addr + (p * rb) as u64).to_le_bytes())
            .collect();
        let list = g.host_bytes(&addrs);
        let y = g.zeros(picks.len() * n);
        g.run(|k, r| k.embed_rows(r, list.addr, ty, n, y.addr, picks.len()));
        let want: Vec<f32> = picks
            .iter()
            .flat_map(|&p| twin::embed_row(ty, &table[p * rb..(p + 1) * rb], n))
            .collect();
        check(Kernel::EmbedRows, ty.name(), &g.read(y), &want);
    }
}

#[test]
#[ignore = "needs the GPU"]
fn gated_rmsnorm_over_more_heads_than_one_dispatch_holds() {
    // 65,536 + 37 heads: more workgroups than Vulkan guarantees in one dispatch, so the helper splits the run.
    let g = Gpu::new();
    let mut rng = Rng::new(52);
    let (dv, heads) = (128, 65_536 + 37);
    let o = rand_f32(&mut rng, heads * dv, -1.0, 1.0);
    let z = rand_f32(&mut rng, heads * dv, -4.0, 4.0);
    let w = rand_f32(&mut rng, dv, 0.5, 1.5);
    let (go, gz, gw, y) = (g.f32s(&o), g.f32s(&z), g.f32s(&w), g.zeros(heads * dv));
    g.run(|k, r| k.gated_rmsnorm(r, go.addr, gz.addr, gw.addr, y.addr, heads, dv, 1e-6));
    check(
        Kernel::GatedRmsnorm,
        &format!("dv={dv}, {heads} heads"),
        &g.read(y),
        &dn::gated_rmsnorm(&o, &z, &w, dv, 1e-6),
    );
}

/// `n` random values with each consecutive group of `dk` L2-normalized.
fn unit_groups(rng: &mut Rng, n: usize, dk: usize) -> Vec<f32> {
    rand_f32(rng, n, -1.0, 1.0)
        .chunks(dk)
        .flat_map(|c| qwen36::ops::l2norm(c, 1e-6))
        .collect()
}

#[test]
#[ignore = "needs the GPU"]
fn conv_seq_over_a_batch_carries_the_state() {
    // 70 tokens from a state that isn't zero: each token's output, and the state left for the next chunk.
    let g = Gpu::new();
    let mut rng = Rng::new(54);
    // Both models' shapes, and a width that isn't a multiple of the workgroup's 256 (the last workgroup part full).
    let (q, t) = (Dims::QWEN36, Dims::tiny());
    for (n, kw) in [
        (q.conv_dim(), q.ssm_conv),
        (t.conv_dim(), t.ssm_conv),
        (300, 4),
    ] {
        let tokens = 70;
        let w = rand_f32(&mut rng, n * kw, -0.5, 0.5);
        let x = rand_f32(&mut rng, tokens * n, -2.0, 2.0);
        let mut state = rand_f32(&mut rng, n * (kw - 1), -2.0, 2.0);
        let (gw, gx, gstate, y) = (g.f32s(&w), g.f32s(&x), g.f32s(&state), g.zeros(tokens * n));
        g.run(|k, r| {
            k.conv_seq(
                r,
                gx.addr,
                gw.addr,
                gstate.addr,
                y.addr,
                n,
                kw,
                tokens,
                true,
            )
        });
        let want = dn::conv_seq(&mut state, &w, &x, kw, n, true);
        check(
            Kernel::ConvSeq,
            &format!("n={n}, {tokens} tokens"),
            &g.read(y),
            &want,
        );
        assert!(
            same_bits(&g.read(gstate), &state),
            "the state holds the last raw inputs"
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn gdn_seq_over_a_batch_carries_the_state() {
    // 40 tokens, q, k and v read from conv rows as prefill lays them out, from a state that isn't zero.
    let g = Gpu::new();
    let mut rng = Rng::new(55);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (nv, dk, dv, ng, tokens) = (
            d.ssm_dt_rank,
            d.ssm_d_state,
            d.v_head_dim(),
            d.ssm_n_group,
            40,
        );
        let (kd, stride) = (d.key_dim(), d.conv_dim());
        let q = unit_groups(&mut rng, tokens * kd, dk);
        let k = unit_groups(&mut rng, tokens * kd, dk);
        let v = rand_f32(&mut rng, tokens * nv * dv, -1.0, 1.0);
        let conv: Vec<f32> = (0..tokens)
            .flat_map(|t| {
                [
                    &q[t * kd..(t + 1) * kd],
                    &k[t * kd..(t + 1) * kd],
                    &v[t * nv * dv..(t + 1) * nv * dv],
                ]
                .concat()
            })
            .collect();
        let gg = rand_f32(&mut rng, tokens * nv, -2.0, 0.0);
        let beta = rand_f32(&mut rng, tokens * nv, 0.05, 0.95);
        let mut state = rand_f32(&mut rng, nv * dv * dk, -0.2, 0.2);
        let (gc, ggg, gb, gs) = (g.f32s(&conv), g.f32s(&gg), g.f32s(&beta), g.f32s(&state));
        let o = g.zeros(tokens * nv * dv);
        g.run(|kk, r| {
            kk.gdn_seq(
                r,
                gc.addr,
                gc.addr + 4 * kd as u64,
                gc.addr + 8 * kd as u64,
                ggg.addr,
                gb.addr,
                gs.addr,
                o.addr,
                nv,
                dk,
                dv,
                ng,
                tokens,
                stride,
                true,
            )
        });
        let want = dn::gdn_seq(&mut state, &q, &k, &v, &gg, &beta, ng, dk, dv, true);
        check(
            Kernel::GdnSeq,
            &format!("dk={dk}, {tokens} tokens"),
            &g.read(o),
            &want,
        );
        check(
            Kernel::GdnSeq,
            &format!("dk={dk}, the state"),
            &g.read(gs),
            &state,
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn the_scans_match_decodes_kernels_token_by_token() {
    // conv_seq and gdn_seq over 24 tokens against conv_step and gdn_step called once a token: the same arithmetic,
    // so prefill's DeltaNet differs from decode's only through its inputs.
    let g = Gpu::new();
    let mut rng = Rng::new(56);
    let d = Dims::QWEN36;
    let (n, kw, tokens) = (d.conv_dim(), d.ssm_conv, 24);
    let (nv, dk, dv, ng, kd) = (
        d.ssm_dt_rank,
        d.ssm_d_state,
        d.v_head_dim(),
        d.ssm_n_group,
        d.key_dim(),
    );
    let w = g.f32s(&rand_f32(&mut rng, n * kw, -0.5, 0.5));
    let x = rand_f32(&mut rng, tokens * n, -2.0, 2.0);
    let st0 = rand_f32(&mut rng, n * (kw - 1), -2.0, 2.0);
    let (gx, s_seq, s_step) = (g.f32s(&x), g.f32s(&st0), g.f32s(&st0));
    let (y_seq, y_step) = (g.zeros(tokens * n), g.zeros(tokens * n));
    g.run(|k, r| {
        k.conv_seq(
            r, gx.addr, w.addr, s_seq.addr, y_seq.addr, n, kw, tokens, true,
        )
    });
    for t in 0..tokens {
        let at = 4 * (t * n) as u64;
        g.run(|k, r| {
            k.conv_step(
                r,
                gx.addr + at,
                w.addr,
                s_step.addr,
                y_step.addr + at,
                n,
                kw,
            )
        });
    }
    let (a, b) = (g.read(y_seq), g.read(y_step));
    assert!(
        same_bits(&a, &b),
        "conv_seq's outputs are conv_step's, bit for bit"
    );
    assert!(
        same_bits(&g.read(s_seq), &g.read(s_step)),
        "and so is the state it leaves"
    );

    let conv = unit_groups(&mut rng, tokens * n, dk);
    let gg = g.f32s(&rand_f32(&mut rng, tokens * nv, -2.0, 0.0));
    let beta = g.f32s(&rand_f32(&mut rng, tokens * nv, 0.05, 0.95));
    let st0 = rand_f32(&mut rng, nv * dv * dk, -0.2, 0.2);
    let (gc, s_seq, s_step) = (g.f32s(&conv), g.f32s(&st0), g.f32s(&st0));
    let (o_seq, o_step) = (g.zeros(tokens * nv * dv), g.zeros(tokens * nv * dv));
    g.run(|k, r| {
        k.gdn_seq(
            r,
            gc.addr,
            gc.addr + 4 * kd as u64,
            gc.addr + 8 * kd as u64,
            gg.addr,
            beta.addr,
            s_seq.addr,
            o_seq.addr,
            nv,
            dk,
            dv,
            ng,
            tokens,
            n,
            true,
        )
    });
    for t in 0..tokens {
        let (row, gate, out) = (
            4 * (t * n) as u64,
            4 * (t * nv) as u64,
            4 * (t * nv * dv) as u64,
        );
        g.run(|k, r| {
            k.gdn_step(
                r,
                gc.addr + row,
                gc.addr + row + 4 * kd as u64,
                gc.addr + row + 8 * kd as u64,
                gg.addr + gate,
                beta.addr + gate,
                s_step.addr,
                o_step.addr + out,
                nv,
                dk,
                dv,
                ng,
            )
        });
    }
    // The same operations, in registers rather than memory: the compiler fuses a few multiply-adds differently.
    check(
        Kernel::GdnSeq,
        "against gdn_step, 24 tokens",
        &g.read(o_seq),
        &g.read(o_step),
    );
    check(
        Kernel::GdnSeq,
        "the state against gdn_step's",
        &g.read(s_seq),
        &g.read(s_step),
    );
    // Whether they are also equal bit for bit: reported, expected to differ, never asserted.
    identity(
        Kernel::GdnSeq.name(),
        "against gdn_step, 24 tokens",
        same_bits(&g.read(o_seq), &g.read(o_step)),
    );
    identity(
        Kernel::GdnSeq.name(),
        "the state against gdn_step's",
        same_bits(&g.read(s_seq), &g.read(s_step)),
    );
}

#[test]
#[ignore = "needs the GPU"]
fn the_scans_without_store_write_the_same_outputs_and_no_state() {
    // The verify pass reads state set A and never writes it. Over 24 tokens: store = false gives the
    // outputs store = true gives, bit for bit, and leaves the state buffer as it was.
    let g = Gpu::new();
    let mut rng = Rng::new(59);
    let d = Dims::QWEN36;
    let (n, kw, tokens) = (d.conv_dim(), d.ssm_conv, 24);
    let (nv, dk, dv, ng, kd) = (
        d.ssm_dt_rank,
        d.ssm_d_state,
        d.v_head_dim(),
        d.ssm_n_group,
        d.key_dim(),
    );
    let w = g.f32s(&rand_f32(&mut rng, n * kw, -0.5, 0.5));
    let gx = g.f32s(&rand_f32(&mut rng, tokens * n, -2.0, 2.0));
    let st0 = rand_f32(&mut rng, n * (kw - 1), -2.0, 2.0);
    let (s_store, s_keep) = (g.f32s(&st0), g.f32s(&st0));
    let (y_store, y_keep) = (g.zeros(tokens * n), g.zeros(tokens * n));
    g.run(|k, r| {
        k.conv_seq(
            r,
            gx.addr,
            w.addr,
            s_store.addr,
            y_store.addr,
            n,
            kw,
            tokens,
            true,
        )
    });
    g.run(|k, r| {
        k.conv_seq(
            r,
            gx.addr,
            w.addr,
            s_keep.addr,
            y_keep.addr,
            n,
            kw,
            tokens,
            false,
        )
    });
    assert!(
        same_bits(&g.read(y_keep), &g.read(y_store)),
        "conv_seq's outputs don't depend on store"
    );
    assert!(
        same_bits(&g.read(s_keep), &st0),
        "conv_seq left the state alone"
    );
    assert!(
        !same_bits(&g.read(s_store), &st0),
        "the same input does move the state with store"
    );

    let conv = g.f32s(&unit_groups(&mut rng, tokens * n, dk));
    let gg = g.f32s(&rand_f32(&mut rng, tokens * nv, -2.0, 0.0));
    let beta = g.f32s(&rand_f32(&mut rng, tokens * nv, 0.05, 0.95));
    let st0 = rand_f32(&mut rng, nv * dv * dk, -0.2, 0.2);
    let (s_store, s_keep) = (g.f32s(&st0), g.f32s(&st0));
    let (o_store, o_keep) = (g.zeros(tokens * nv * dv), g.zeros(tokens * nv * dv));
    for (state, o, store) in [(&s_store, &o_store, true), (&s_keep, &o_keep, false)] {
        g.run(|k, r| {
            k.gdn_seq(
                r,
                conv.addr,
                conv.addr + 4 * kd as u64,
                conv.addr + 8 * kd as u64,
                gg.addr,
                beta.addr,
                state.addr,
                o.addr,
                nv,
                dk,
                dv,
                ng,
                tokens,
                n,
                store,
            )
        });
    }
    assert!(
        same_bits(&g.read(o_keep), &g.read(o_store)),
        "gdn_seq's outputs don't depend on store"
    );
    assert!(
        same_bits(&g.read(s_keep), &st0),
        "gdn_seq left the state alone"
    );
    assert!(
        !same_bits(&g.read(s_store), &st0),
        "the same input does move the state with store"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn the_scans_dont_depend_on_where_a_chunk_is_cut() {
    // 17 tokens then 23, the state carried through memory, against 40 at once: bit for bit, outputs and final state.
    let g = Gpu::new();
    let mut rng = Rng::new(57);
    let d = Dims::QWEN36;
    let (n, kw, tokens, cut) = (d.conv_dim(), d.ssm_conv, 40, 17);
    let w = g.f32s(&rand_f32(&mut rng, n * kw, -0.5, 0.5));
    let x = g.f32s(&rand_f32(&mut rng, tokens * n, -2.0, 2.0));
    let st0 = rand_f32(&mut rng, n * (kw - 1), -2.0, 2.0);
    let (whole, split) = (g.f32s(&st0), g.f32s(&st0));
    let (y_whole, y_split) = (g.zeros(tokens * n), g.zeros(tokens * n));
    g.run(|k, r| {
        k.conv_seq(
            r,
            x.addr,
            w.addr,
            whole.addr,
            y_whole.addr,
            n,
            kw,
            tokens,
            true,
        )
    });
    g.run(|k, r| {
        k.conv_seq(
            r,
            x.addr,
            w.addr,
            split.addr,
            y_split.addr,
            n,
            kw,
            cut,
            true,
        )
    });
    let at = 4 * (cut * n) as u64;
    g.run(|k, r| {
        k.conv_seq(
            r,
            x.addr + at,
            w.addr,
            split.addr,
            y_split.addr + at,
            n,
            kw,
            tokens - cut,
            true,
        )
    });
    assert!(
        same_bits(&g.read(y_whole), &g.read(y_split)),
        "conv_seq's outputs"
    );
    assert!(
        same_bits(&g.read(whole), &g.read(split)),
        "conv_seq's state"
    );

    let (nv, dk, dv, ng, kd) = (
        d.ssm_dt_rank,
        d.ssm_d_state,
        d.v_head_dim(),
        d.ssm_n_group,
        d.key_dim(),
    );
    let conv = g.f32s(&unit_groups(&mut rng, tokens * n, dk));
    let gg = g.f32s(&rand_f32(&mut rng, tokens * nv, -2.0, 0.0));
    let beta = g.f32s(&rand_f32(&mut rng, tokens * nv, 0.05, 0.95));
    let st0 = rand_f32(&mut rng, nv * dv * dk, -0.2, 0.2);
    let (whole, split) = (g.f32s(&st0), g.f32s(&st0));
    let (o_whole, o_split) = (g.zeros(tokens * nv * dv), g.zeros(tokens * nv * dv));
    let scan = |state: u64, o: u64, from: usize, count: usize| {
        let (row, gate, out) = (
            4 * (from * n) as u64,
            4 * (from * nv) as u64,
            4 * (from * nv * dv) as u64,
        );
        g.run(|k, r| {
            k.gdn_seq(
                r,
                conv.addr + row,
                conv.addr + row + 4 * kd as u64,
                conv.addr + row + 8 * kd as u64,
                gg.addr + gate,
                beta.addr + gate,
                state,
                o + out,
                nv,
                dk,
                dv,
                ng,
                count,
                n,
                true,
            )
        });
    };
    scan(whole.addr, o_whole.addr, 0, tokens);
    scan(split.addr, o_split.addr, 0, cut);
    scan(split.addr, o_split.addr, cut, tokens - cut);
    assert!(
        same_bits(&g.read(o_whole), &g.read(o_split)),
        "gdn_seq's outputs"
    );
    assert!(same_bits(&g.read(whole), &g.read(split)), "gdn_seq's state");
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Random f16 caches of `n_ctx` positions of `width` values.
fn random_cache(rng: &mut Rng, n_ctx: usize, width: usize) -> Vec<u16> {
    (0..n_ctx * width)
        .map(|_| twin::f16_bits(rng.uniform(-1.0, 1.0)))
        .collect()
}

/// `attn_prefill` on the GPU for `tokens` queries from `pos`, over caches holding `kc` and `vc`.
#[allow(clippy::too_many_arguments)]
fn gpu_attn_prefill(
    g: &Gpu,
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    pos: usize,
    tokens: usize,
    d: &Dims,
) -> Vec<f32> {
    let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
    let (gq, gkc, gvc) = (g.f32s(q), g.bytes(&u16_bytes(kc)), g.bytes(&u16_bytes(vc)));
    let y = g.zeros(tokens * nh * hd);
    g.run(|k, r| {
        k.attn_prefill(
            r, gq.addr, gkc.addr, gvc.addr, y.addr, pos, tokens, nh, nkv, hd,
        )
    });
    g.read(y)
}

#[test]
#[ignore = "needs the GPU"]
fn attn_prefill_against_its_twin() {
    // From position 0 and from deep in the cache, with a query count that leaves the last workgroup part full. The
    // cache rows past the batch hold random values, which would move the result if they entered.
    let g = Gpu::new();
    let mut rng = Rng::new(58);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
        for (pos, tokens) in [(0, 70), (1000, 40)] {
            let q = rand_f32(&mut rng, tokens * nh * hd, -12.0, 12.0);
            let ctx = pos + tokens + 50;
            let (kc, vc) = (
                random_cache(&mut rng, ctx, nkv * hd),
                random_cache(&mut rng, ctx, nkv * hd),
            );
            check(
                Kernel::AttnPrefill,
                &format!("hd={hd}, {tokens} queries from {pos}"),
                &gpu_attn_prefill(&g, &q, &kc, &vc, pos, tokens, &d),
                &at::attn_prefill(&q, &kc, &vc, pos, tokens, nh, nkv, hd),
            );
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn attn_prefills_results_dont_depend_on_the_batch() {
    // 130 queries from 0 in one batch, and as batches of 50 and 80, of 51 and 79, and query 37 alone: the same blocks of
    // keys in the same order, so the same bits. The odd cuts move queries between even and odd slots of a workgroup,
    // whose value sums the kernel keeps in different registers. Prefill's chunk boundaries don't move a query's
    // attention.
    let g = Gpu::new();
    let mut rng = Rng::new(59);
    let d = Dims::QWEN36;
    let (nh, nkv, hd, tokens) = (d.n_head, d.n_head_kv, d.head_dim, 130);
    let q = rand_f32(&mut rng, tokens * nh * hd, -12.0, 12.0);
    let (kc, vc) = (
        random_cache(&mut rng, 200, nkv * hd),
        random_cache(&mut rng, 200, nkv * hd),
    );
    let all = gpu_attn_prefill(&g, &q, &kc, &vc, 0, tokens, &d);
    let w = nh * hd;
    for (from, n) in [(0, 50), (50, 80), (0, 51), (51, 79), (37, 1)] {
        let part = gpu_attn_prefill(&g, &q[from * w..(from + n) * w], &kc, &vc, from, n, &d);
        assert!(
            same_bits(&part, &all[from * w..(from + n) * w]),
            "queries {from}..{} as a batch from position {from}",
            from + n
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn attn_prefill_never_reads_cache_rows_past_the_batch() {
    // Rows past the last key hold f16 NaN (0x7E00): one that reached a sum would turn the result to NaN, so matching the
    // twin's finite result pins that rows past the batch's last position never reach the result. That is V's zeroing in
    // particular: K's rows there are past every query, so the mask discards their scores whatever the load does.
    let g = Gpu::new();
    let mut rng = Rng::new(60);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
        for (pos, tokens) in [(0, 20), (45, 7)] {
            let q = rand_f32(&mut rng, tokens * nh * hd, -12.0, 12.0);
            let ctx = pos + tokens + 70;
            let mut kc = random_cache(&mut rng, ctx, nkv * hd);
            let mut vc = random_cache(&mut rng, ctx, nkv * hd);
            for c in [&mut kc, &mut vc] {
                c[(pos + tokens) * nkv * hd..].fill(0x7E00);
            }
            let want = at::attn_prefill(&q, &kc, &vc, pos, tokens, nh, nkv, hd);
            assert!(want.iter().all(|x| x.is_finite()), "the twin's result");
            let got = gpu_attn_prefill(&g, &q, &kc, &vc, pos, tokens, &d);
            assert!(
                got.iter().all(|x| x.is_finite()),
                "hd={hd}: a NaN row entered"
            );
            check(
                Kernel::AttnPrefill,
                &format!("hd={hd}, {tokens} queries from {pos}, NaN past the batch"),
                &got,
                &want,
            );
        }
    }
}

/// `attn_prefill_cm` on the GPU for `tokens` queries from `pos`, over caches holding `kc` and `vc`, run as if the
/// workgroup had `sg_limit` subgroups (0: all of them). `y` is allocated with 64·w floats of sentinel past the
/// `tokens·w` the kernel owns, and they must be untouched afterwards: a padding row written would show here.
#[allow(clippy::too_many_arguments)]
fn gpu_attn_prefill_cm_sg(
    g: &Gpu,
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    pos: usize,
    tokens: usize,
    d: &Dims,
    sg_limit: u32,
) -> Vec<f32> {
    const SENTINEL: u32 = 0x5A5A_5A5A;
    assert!(
        g.k.coop(),
        "attn_prefill_cm needs the cooperative-matrix family"
    );
    let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
    let w = nh * hd;
    let (gq, gkc, gvc) = (g.f32s(q), g.bytes(&u16_bytes(kc)), g.bytes(&u16_bytes(vc)));
    let mut init = vec![0f32; tokens * w];
    init.extend(std::iter::repeat_n(f32::from_bits(SENTINEL), 64 * w));
    let y = g.f32s(&init);
    g.run(|k, r| {
        k.attn_prefill_cm(
            r, gq.addr, gkc.addr, gvc.addr, y.addr, pos, tokens, nh, nkv, hd, sg_limit,
        )
    });
    let mut out = g.read(y);
    let past = out.split_off(tokens * w);
    assert!(
        past.iter().all(|x| x.to_bits() == SENTINEL),
        "hd={hd}: {tokens} queries from {pos}, sg_limit {sg_limit}: the kernel wrote past its {tokens} rows"
    );
    out
}

/// `attn_prefill_cm` on every subgroup.
fn gpu_attn_prefill_cm(
    g: &Gpu,
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    pos: usize,
    tokens: usize,
    d: &Dims,
) -> Vec<f32> {
    gpu_attn_prefill_cm_sg(g, q, kc, vc, pos, tokens, d, 0)
}

#[test]
#[ignore = "needs the GPU"]
fn attn_prefill_cm_against_its_twin() {
    // From 0 and from deep in the cache, with query counts that leave the last 64-row workgroup part full (70, 130:
    // 6 and 2 rows; 65: one), and rows past the batch random, so reading them would move the result.
    let g = Gpu::new();
    let mut rng = Rng::new(63);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
        for (pos, tokens) in [(0, 70), (1000, 40), (0, 130), (3000, 65)] {
            let q = rand_f32(&mut rng, tokens * nh * hd, -12.0, 12.0);
            let ctx = pos + tokens + 50;
            let (kc, vc) = (
                random_cache(&mut rng, ctx, nkv * hd),
                random_cache(&mut rng, ctx, nkv * hd),
            );
            check(
                Kernel::AttnPrefillCm,
                &format!("hd={hd}, {tokens} queries from {pos}"),
                &gpu_attn_prefill_cm(&g, &q, &kc, &vc, pos, tokens, &d),
                &at::attn_prefill_cm(&q, &kc, &vc, pos, tokens, nh, nkv, hd),
            );
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn attn_prefill_cms_results_dont_depend_on_the_batch() {
    // 130 queries in one batch, from 0 and from 1,000 (8 positions into a 32-key block), against sub-batches that move
    // queries between 16-row tiles and 64-row workgroups (cuts at 50, 51, 64, 37), and single queries whose full-batch
    // workgroup walks blocks wholly past them (in the batch from 0, queries 0 and 5 walk the block 32..63 past them,
    // while query 37 sits inside it; in the batch from 1,000, query 1,037 walks the block 1,056..1,087 wholly past it):
    // the same bits. Every call allocates 64·w sentinel floats past its output and asserts them untouched, so a
    // single-query workgroup that wrote its 63 padding rows fails.
    let g = Gpu::new();
    let mut rng = Rng::new(64);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (nh, nkv, hd, tokens) = (d.n_head, d.n_head_kv, d.head_dim, 130);
        let w = nh * hd;
        let q = rand_f32(&mut rng, tokens * w, -12.0, 12.0);
        let (kc, vc) = (
            random_cache(&mut rng, 1200, nkv * hd),
            random_cache(&mut rng, 1200, nkv * hd),
        );
        for base in [0, 1000] {
            let all = gpu_attn_prefill_cm(&g, &q, &kc, &vc, base, tokens, &d);
            for (from, n) in [
                (0, 50),
                (50, 80),
                (0, 51),
                (51, 79),
                (0, 64),
                (64, 66),
                (37, 1),
                (5, 1),
                (0, 1),
            ] {
                let part = gpu_attn_prefill_cm(
                    &g,
                    &q[from * w..(from + n) * w],
                    &kc,
                    &vc,
                    base + from,
                    n,
                    &d,
                );
                assert!(
                    same_bits(&part, &all[from * w..(from + n) * w]),
                    "hd={hd}: queries {from}..{} from position {}",
                    from + n,
                    base + from
                );
            }
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn attn_prefill_cm_never_reads_cache_rows_past_the_batch() {
    // Rows past the last key hold f16 NaN (0x7E00): one that reached a product would turn the result to NaN (0·NaN is
    // NaN, so the mask can't save V), so a finite result matching the twin's pins that rows past the batch's last
    // position never reach the result.
    let g = Gpu::new();
    let mut rng = Rng::new(65);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
        for (pos, tokens) in [(0, 20), (45, 7), (0, 70)] {
            let q = rand_f32(&mut rng, tokens * nh * hd, -12.0, 12.0);
            let ctx = pos + tokens + 70;
            let mut kc = random_cache(&mut rng, ctx, nkv * hd);
            let mut vc = random_cache(&mut rng, ctx, nkv * hd);
            for c in [&mut kc, &mut vc] {
                c[(pos + tokens) * nkv * hd..].fill(0x7E00);
            }
            let want = at::attn_prefill_cm(&q, &kc, &vc, pos, tokens, nh, nkv, hd);
            assert!(want.iter().all(|x| x.is_finite()), "the twin's result");
            let got = gpu_attn_prefill_cm(&g, &q, &kc, &vc, pos, tokens, &d);
            assert!(
                got.iter().all(|x| x.is_finite()),
                "hd={hd}: a NaN row entered"
            );
            check(
                Kernel::AttnPrefillCm,
                &format!("hd={hd}, {tokens} queries from {pos}, NaN past the batch"),
                &got,
                &want,
            );
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn attn_prefill_cm_results_dont_depend_on_the_subgroup_count() {
    // With one or two subgroups the kernel runs its four M tiles in four or two passes. This GPU runs
    // four subgroups (wave64) or eight (wave32), never fewer, so sg_limit makes it run as one, two or four: every
    // row's bits must be those of the run on every subgroup. 130 queries from 0 and from 1,000 (a part-full last
    // workgroup: 2 rows), and cuts as the batch test's (a batch from inside a 64-row workgroup, one query), so passes
    // with no live row below nq run too. sg_limit 4 differs from 0 only at wave32.
    let g = Gpu::new();
    let mut rng = Rng::new(66);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
        let w = nh * hd;
        let q = rand_f32(&mut rng, 130 * w, -12.0, 12.0);
        let (kc, vc) = (
            random_cache(&mut rng, 1200, nkv * hd),
            random_cache(&mut rng, 1200, nkv * hd),
        );
        for (pos, from, n) in [(0, 0, 130), (1000, 0, 130), (1000, 51, 79), (0, 37, 1)] {
            let qs = &q[from * w..(from + n) * w];
            let all = gpu_attn_prefill_cm_sg(&g, qs, &kc, &vc, pos + from, n, &d, 0);
            for sg in [1, 2, 4] {
                let run = gpu_attn_prefill_cm_sg(&g, qs, &kc, &vc, pos + from, n, &d, sg);
                assert!(
                    same_bits(&run, &all),
                    "hd={hd}: {n} queries from {}, sg_limit {sg}",
                    pos + from
                );
            }
        }
    }
}

use kernels::dispatch::{
    moe_tiles, read_status, slot_entry, ExpertStatus, SlotLayout, STATUS_BYTES, STATUS_CLEAR,
    WHERE_CPU, WHERE_HOST, WHERE_VRAM,
};

fn read_u32(g: &Gpu, s: vk::Slice) -> Vec<u32> {
    g.dev
        .download(s)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// Routed ids for `tokens` tokens choosing `k` distinct experts of `n`, skewed so a few experts get most entries:
/// their runs fill several tiles.
fn skewed_ids(rng: &mut Rng, tokens: usize, k: usize, n: usize) -> Vec<f32> {
    let mut ids = Vec::with_capacity(tokens * k);
    for _ in 0..tokens {
        let mut chosen: Vec<usize> = Vec::with_capacity(k);
        while chosen.len() < k {
            let u = rng.uniform(0.0, 1.0);
            let e = ((u * u * u) * n as f32) as usize % n;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        ids.extend(chosen.iter().map(|&e| e as f32));
    }
    ids
}

/// moe_group on the GPU: (list, tiles as [u32; 4]).
fn gpu_moe_group(g: &Gpu, ids: &[f32], n: usize) -> (Vec<u32>, Vec<[u32; 4]>) {
    let max = moe_tiles(ids.len(), n);
    let (gids, list, tiles) = (g.f32s(ids), g.zeros(ids.len()), g.zeros(4 * max));
    g.run(|k, r| k.moe_group(r, gids.addr, list.addr, tiles.addr, n, ids.len(), max));
    let t = read_u32(g, tiles);
    (read_u32(g, list), t.as_chunks::<4>().0.to_vec())
}

#[test]
#[ignore = "needs the GPU"]
fn moe_group_matches_its_twin() {
    // A small batch, and a full chunk's 16,384 entries, which the kernel reads in four pieces.
    let g = Gpu::new();
    let mut rng = Rng::new(61);
    for (tokens, k, n) in [(130, 8, 256), (2048, 8, 256), (9, 4, 16), (1500, 8, 256)] {
        let ids = skewed_ids(&mut rng, tokens, k, n);
        let (list, tiles) = gpu_moe_group(&g, &ids, n);
        let (tl, tt) = moe::moe_group(&ids, n, moe_tiles(ids.len(), n));
        assert_eq!(list, tl, "{tokens} tokens: the list");
        assert_eq!(tiles, tt, "{tokens} tokens: the tiles");
        assert!(
            tokens < 100 || tt.iter().any(|t| t[2] == 64),
            "{tokens} tokens: some expert fills a tile"
        );
    }
}

/// `n` random experts of `lay` packed in one buffer, in VRAM and in pinned host memory, and their bytes.
fn packed_experts(g: &Gpu, rng: &mut Rng, lay: &SlotLayout, n: usize) -> (u64, u64, Vec<Vec<u8>>) {
    let slots: Vec<Vec<u8>> = (0..n)
        .map(|_| {
            let mut s = random_weights(rng, lay.gate, lay.n_ff, lay.n_embd);
            s.extend(random_weights(rng, lay.up, lay.n_ff, lay.n_embd));
            s.extend(random_weights(rng, lay.down, lay.n_embd, lay.n_ff));
            s
        })
        .collect();
    let packed = slots.concat();
    (g.bytes(&packed).addr, g.host_bytes(&packed).addr, slots)
}

/// A slot table for layer 1 of `n` experts: expert e's entry is `loc(e)`, pointing at its VRAM or host copy. Layer
/// 0's entries are garbage that must never be read.
fn table_for(
    g: &Gpu,
    lay: &SlotLayout,
    n: usize,
    vram: u64,
    host: u64,
    loc: impl Fn(usize) -> u32,
) -> vk::Slice {
    let mut t = vec![0xEEu8; 2 * n * 16];
    for e in 0..n {
        let base = if loc(e) == WHERE_VRAM { vram } else { host };
        let entry = slot_entry(base + (e * lay.bytes()) as u64, loc(e));
        t[(n + e) * 16..(n + e + 1) * 16].copy_from_slice(&entry);
    }
    g.bytes(&t)
}

/// moe_group and the two grouped expert kernels for layer 1, f32 or split-f16 cooperative-matrix (`cm`): (act in list
/// order, ye, status).
#[allow(clippy::too_many_arguments)]
fn run_grouped(
    g: &Gpu,
    table: vk::Slice,
    lay: &SlotLayout,
    n: usize,
    ids: &[f32],
    x: &[f32],
    k: usize,
    cm: bool,
) -> (Vec<f32>, Vec<f32>, ExpertStatus) {
    let entries = ids.len();
    let max = moe_tiles(entries, n);
    let (gids, gx) = (g.f32s(ids), g.f32s(x));
    let (list, tiles) = (g.zeros(entries), g.zeros(4 * max));
    let (act, ye) = (g.zeros(entries * lay.n_ff), g.zeros(entries * lay.n_embd));
    let status = g.bytes(&STATUS_CLEAR);
    g.run(|kk, r| {
        kk.moe_group(r, gids.addr, list.addr, tiles.addr, n, entries, max);
        r.barrier();
        if cm {
            kk.experts_matmul_cm_gate_up(
                r,
                table.addr,
                1,
                n,
                list.addr,
                tiles.addr,
                max,
                gx.addr,
                act.addr,
                status.addr,
                lay,
                k,
            );
        } else {
            kk.experts_matmul_gate_up(
                r,
                table.addr,
                1,
                n,
                list.addr,
                tiles.addr,
                max,
                gx.addr,
                act.addr,
                status.addr,
                lay,
                k,
            );
        }
        r.barrier();
        if cm {
            kk.experts_matmul_cm_down(
                r,
                table.addr,
                1,
                n,
                list.addr,
                tiles.addr,
                max,
                act.addr,
                ye.addr,
                status.addr,
                lay,
            );
        } else {
            kk.experts_matmul_down(
                r,
                table.addr,
                1,
                n,
                list.addr,
                tiles.addr,
                max,
                act.addr,
                ye.addr,
                status.addr,
                lay,
            );
        }
    });
    let st: [u8; STATUS_BYTES] = g.dev.download(status).unwrap().try_into().unwrap();
    (g.read(act), g.read(ye), read_status(&st))
}

#[test]
#[ignore = "needs the GPU"]
fn grouped_experts_against_their_twins_from_host_and_vram() {
    // Every expert in host memory, every expert in VRAM, and half of each: the twins' values, and the same bits
    // whichever memory an expert is read from. The 150-token layout makes full tiles and multi-tile runs (asserted).
    let g = Gpu::new();
    let mut rng = Rng::new(62);
    let lays = [
        (
            SlotLayout {
                gate: WType::Q4K,
                up: WType::Q4K,
                down: WType::Q5K,
                n_embd: 2048,
                n_ff: 512,
            },
            16,
            40,
            8,
        ),
        (
            SlotLayout {
                gate: WType::Q4K,
                up: WType::Q4K,
                down: WType::Q6K,
                n_embd: 256,
                n_ff: 256,
            },
            16,
            150,
            4,
        ),
    ];
    for (lay, n, tokens, k) in lays {
        let (vram, host, slots) = packed_experts(&g, &mut rng, &lay, n);
        let ids = skewed_ids(&mut rng, tokens, k, n);
        let x = rand_f32(&mut rng, tokens * lay.n_embd, -1.0, 1.0);
        let (list, tt) = moe::moe_group(&ids, n, moe_tiles(ids.len(), n));
        if tokens >= 100 {
            let mut per = vec![0; n];
            tt.iter()
                .filter(|t| t[2] > 0)
                .for_each(|t| per[t[0] as usize] += 1);
            assert!(
                tt.iter().any(|t| t[2] == 64),
                "{tokens} tokens: some tile is full"
            );
            assert!(
                per.iter().any(|&c| c > 1),
                "{tokens} tokens: some expert spans several tiles"
            );
        }
        let t_act = moe::experts_matmul_gate_up(|e| slots[e].as_slice(), &lay, &x, &ids, &list, k);
        let t_ye = moe::experts_matmul_down(|e| slots[e].as_slice(), &lay, &t_act, &ids, &list);
        let case = format!("{}/{} {}", lay.down.name(), lay.n_embd, tokens);
        let mut outs = Vec::new();
        for loc in [
            &(|_| WHERE_HOST) as &dyn Fn(usize) -> u32,
            &|_| WHERE_VRAM,
            &|e| if e % 2 == 0 { WHERE_VRAM } else { WHERE_HOST },
        ] {
            let table = table_for(&g, &lay, n, vram, host, loc);
            let (act, ye, st) = run_grouped(&g, table, &lay, n, &ids, &x, k, false);
            assert_eq!(st, ExpertStatus::default(), "{case}: nothing refused");
            check(Kernel::ExpertsMatmulGateUp, &case, &act, &t_act);
            check(Kernel::ExpertsMatmulDown, &case, &ye, &t_ye);
            outs.push((act, ye));
        }
        assert!(
            same_bits(&outs[0].0, &outs[1].0)
                && same_bits(&outs[1].0, &outs[2].0)
                && same_bits(&outs[0].1, &outs[1].1)
                && same_bits(&outs[1].1, &outs[2].1),
            "{case}: where an expert lives must not change a bit"
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn a_refused_expert_is_named_and_its_rows_are_nan() {
    // Expert 3's entry is CPU and expert 6's has no address: the status names each kind's lowest, their rows are NaN,
    // and the other experts' rows are computed.
    let g = Gpu::new();
    let mut rng = Rng::new(63);
    let lay = SlotLayout {
        gate: WType::Q4K,
        up: WType::Q4K,
        down: WType::Q6K,
        n_embd: 256,
        n_ff: 256,
    };
    let (n, k, tokens) = (8, 2, 6);
    let (vram, _, _) = packed_experts(&g, &mut rng, &lay, n);
    let mut t = vec![0xEEu8; 2 * n * 16];
    for e in 0..n {
        let entry = match e {
            3 => slot_entry(0, WHERE_CPU),
            6 => slot_entry(0, WHERE_HOST),
            _ => slot_entry(vram + (e * lay.bytes()) as u64, WHERE_VRAM),
        };
        t[(n + e) * 16..(n + e + 1) * 16].copy_from_slice(&entry);
    }
    let ids = [3.0, 0.0, 1.0, 6.0, 2.0, 4.0, 3.0, 6.0, 5.0, 7.0, 0.0, 1.0];
    let x = rand_f32(&mut rng, tokens * lay.n_embd, -1.0, 1.0);
    for cm in [false, true] {
        let (act, ye, st) = run_grouped(&g, g.bytes(&t), &lay, n, &ids, &x, k, cm);
        let (list, _) = moe::moe_group(&ids, n, moe_tiles(ids.len(), n));
        for (i, &entry) in list.iter().enumerate() {
            let e = ids[entry as usize];
            let row = &act[i * lay.n_ff..(i + 1) * lay.n_ff];
            if e == 3.0 || e == 6.0 {
                assert!(
                    row.iter().all(|v| v.is_nan()),
                    "cm {cm}: act row {i}, expert {e}"
                );
            } else {
                assert!(
                    row.iter().all(|v| v.is_finite()),
                    "cm {cm}: act row {i}, expert {e}"
                );
            }
        }
        assert_eq!(st.cpu, Some((1, 3)), "cm {cm}");
        assert_eq!(st.bad, Some((1, 6)), "cm {cm}");
        for (i, &e) in ids.iter().enumerate() {
            let row = &ye[i * lay.n_embd..(i + 1) * lay.n_embd];
            if e == 3.0 || e == 6.0 {
                assert!(
                    row.iter().all(|v| v.is_nan()),
                    "cm {cm}: entry {i}, expert {e}"
                );
            } else {
                assert!(
                    row.iter().all(|v| v.is_finite()),
                    "cm {cm}: entry {i}, expert {e}"
                );
            }
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn grouped_cm_experts_against_their_twins_from_host_and_vram() {
    // grouped_experts_against_their_twins_from_host_and_vram for the split-f16 kernels: the twins' values, and the same
    // bits whichever memory an expert is read from (test 5's rule: a staged expert is one more address). The last two
    // layouts take gate/up's k (n_embd) and then down's k (n_ff) to 4,096, so the per-32-value-slice sums
    // are pinned under the kernels' 1e-5: summing every product in the one
    // hardware accumulator measured biased, its error linear in k. Reverting a kernel to it (2026-10-11) failed these
    // layouts: gate/up 3.2e-5 at k = 4,096 (and 1.5e-5 on the first layout's 2,048), down 1.7e-5 at k = 4,096.
    let g = Gpu::new();
    assert!(g.k.coop());
    let mut rng = Rng::new(64);
    let lays = [
        (
            SlotLayout {
                gate: WType::Q4K,
                up: WType::Q4K,
                down: WType::Q5K,
                n_embd: 2048,
                n_ff: 512,
            },
            16,
            40,
            8,
        ),
        (
            SlotLayout {
                gate: WType::Q4K,
                up: WType::Q4K,
                down: WType::Q6K,
                n_embd: 256,
                n_ff: 256,
            },
            16,
            150,
            4,
        ),
        (
            SlotLayout {
                gate: WType::Q4K,
                up: WType::Q4K,
                down: WType::Q5K,
                n_embd: 4096,
                n_ff: 256,
            },
            16,
            40,
            8,
        ),
        (
            SlotLayout {
                gate: WType::Q4K,
                up: WType::Q4K,
                down: WType::Q6K,
                n_embd: 256,
                n_ff: 4096,
            },
            16,
            40,
            8,
        ),
    ];
    for (lay, n, tokens, k) in lays {
        let (vram, host, slots) = packed_experts(&g, &mut rng, &lay, n);
        let ids = skewed_ids(&mut rng, tokens, k, n);
        let x = rand_f32(&mut rng, tokens * lay.n_embd, -1.0, 1.0);
        let (list, tt) = moe::moe_group(&ids, n, moe_tiles(ids.len(), n));
        if tokens >= 100 {
            let mut per = vec![0; n];
            tt.iter()
                .filter(|t| t[2] > 0)
                .for_each(|t| per[t[0] as usize] += 1);
            assert!(
                tt.iter().any(|t| t[2] == 64),
                "{tokens} tokens: some tile is full"
            );
            assert!(
                per.iter().any(|&c| c > 1),
                "{tokens} tokens: some expert spans several tiles"
            );
        }
        let t_act =
            moe::experts_matmul_cm_gate_up(|e| slots[e].as_slice(), &lay, &x, &ids, &list, k);
        let t_ye = moe::experts_matmul_cm_down(|e| slots[e].as_slice(), &lay, &t_act, &ids, &list);
        let case = format!("{}/{}x{} {}", lay.down.name(), lay.n_embd, lay.n_ff, tokens);
        let mut outs = Vec::new();
        for loc in [
            &(|_| WHERE_HOST) as &dyn Fn(usize) -> u32,
            &|_| WHERE_VRAM,
            &|e| if e % 2 == 0 { WHERE_VRAM } else { WHERE_HOST },
        ] {
            let table = table_for(&g, &lay, n, vram, host, loc);
            let (act, ye, st) = run_grouped(&g, table, &lay, n, &ids, &x, k, true);
            assert_eq!(st, ExpertStatus::default(), "{case}: nothing refused");
            check(Kernel::ExpertsMatmulCmGateUp, &case, &act, &t_act);
            check(Kernel::ExpertsMatmulCmDown, &case, &ye, &t_ye);
            outs.push((act, ye));
        }
        assert!(
            same_bits(&outs[0].0, &outs[1].0)
                && same_bits(&outs[1].0, &outs[2].0)
                && same_bits(&outs[0].1, &outs[1].1)
                && same_bits(&outs[1].1, &outs[2].1),
            "{case}: where an expert lives must not change a bit"
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn a_grouped_cm_tokens_rows_dont_depend_on_its_tile() {
    // Chunk invariance for the grouped kernels (a token's bits are the same alone and in any batch): tokens
    // 37..120 of a 150-token batch, run on their own, fall in other tiles and other rows of their tiles (their experts'
    // runs start elsewhere and hold fewer entries), and tokens 0, 37 and 149 alone each make tiles of a single entry; yet
    // each token's ye rows are the same bits. Only ye is compared: the down kernel computes it from the gate/up kernel's
    // act rows, which feed it.
    let g = Gpu::new();
    let mut rng = Rng::new(65);
    let lay = SlotLayout {
        gate: WType::Q4K,
        up: WType::Q4K,
        down: WType::Q5K,
        n_embd: 2048,
        n_ff: 512,
    };
    let (n, tokens, k) = (16, 150, 8);
    let (vram, host, _) = packed_experts(&g, &mut rng, &lay, n);
    let table = table_for(&g, &lay, n, vram, host, |_| WHERE_VRAM);
    let ids = skewed_ids(&mut rng, tokens, k, n);
    let x = rand_f32(&mut rng, tokens * lay.n_embd, -1.0, 1.0);
    let (_, all, st) = run_grouped(&g, table, &lay, n, &ids, &x, k, true);
    assert_eq!(st, ExpertStatus::default(), "nothing refused");
    let (lo, hi) = (37, 120);
    let (_, part, _) = run_grouped(
        &g,
        table,
        &lay,
        n,
        &ids[lo * k..hi * k],
        &x[lo * lay.n_embd..hi * lay.n_embd],
        k,
        true,
    );
    assert!(
        same_bits(&part, &all[lo * k * lay.n_embd..hi * k * lay.n_embd]),
        "tokens {lo}..{hi} on their own"
    );
    for t in [0, 37, 149] {
        let (_, one, _) = run_grouped(
            &g,
            table,
            &lay,
            n,
            &ids[t * k..(t + 1) * k],
            &x[t * lay.n_embd..(t + 1) * lay.n_embd],
            k,
            true,
        );
        assert!(
            same_bits(&one, &all[t * k * lay.n_embd..(t + 1) * k * lay.n_embd]),
            "token {t} alone"
        );
    }
}
