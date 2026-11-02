//! CI's half of test 1 for the prefill kernels: each twin against the reference's operations
//! (`qwen36::ops`), applied token by token, within the kernel's stated tolerance.
use kernels::{twin, Kernel};
use qwen36::{ops, quant::WType, testing::Rng};

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
fn matmul_twins_match_ops_for_every_type() {
    // Each token's outputs against the reference's f64 matvec of the same rows: ragged row and token counts, and the
    // widths prefill uses.
    use qwen36::model::Weight;
    let mut rng = Rng::new(40);
    for ty in WType::ALL {
        for (rows, cols, tokens) in [(37, 256, 5), (64, 512, 3), (8, 2048, 2), (3, 4096, 1)] {
            let w = kernels::testing::random_weights(&mut rng, ty, rows, cols);
            let x = rand(&mut rng, tokens * cols, -1.0, 1.0);
            let weight = Weight {
                ty,
                cols,
                rows,
                bytes: &w,
            };
            let exact: Vec<f32> = x
                .chunks(cols)
                .flat_map(|xt| ops::matvec(&weight, xt, 1))
                .collect();
            close(
                Kernel::matmul(ty),
                &format!("{} {rows}x{cols}, {tokens} tokens", ty.name()),
                &twin::matvec::matmul(ty, &w, rows, &x, tokens),
                &exact,
            );
        }
    }
}

#[test]
fn a_matmul_twins_outputs_dont_depend_on_the_batch() {
    // Token t's outputs in a batch of 7 are bit for bit those of token t alone: each sum runs over its row in order.
    let mut rng = Rng::new(41);
    let (rows, cols, tokens) = (40, 512, 7);
    let w = kernels::testing::random_weights(&mut rng, WType::Q8_0, rows, cols);
    let x = rand(&mut rng, tokens * cols, -1.0, 1.0);
    let all = twin::matvec::matmul(WType::Q8_0, &w, rows, &x, tokens);
    for t in 0..tokens {
        let one = twin::matvec::matmul(WType::Q8_0, &w, rows, &x[t * cols..(t + 1) * cols], 1);
        assert!(one == all[t * rows..(t + 1) * rows], "token {t}");
    }
}

use qwen36::dims::Dims;

/// `n` random values of q (or k): each group of `dk` L2-normalized, as the conv output is before the delta rule.
fn unit_groups(rng: &mut Rng, n: usize, dk: usize) -> Vec<f32> {
    rand(rng, n, -1.0, 1.0)
        .chunks(dk)
        .flat_map(|c| ops::l2norm(c, 1e-6))
        .collect()
}

#[test]
fn conv_seq_twin_matches_ops_over_tokens() {
    // From a state that isn't zero, as a later chunk's is.
    let mut rng = Rng::new(52);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (n, kw, tokens) = (d.conv_dim(), d.ssm_conv, 9);
        let w = rand(&mut rng, n * kw, -0.5, 0.5);
        let x = rand(&mut rng, tokens * n, -2.0, 2.0);
        let mut st_t = rand(&mut rng, n * (kw - 1), -2.0, 2.0);
        let mut st_o = st_t.clone();
        let t = twin::deltanet::conv_seq(&mut st_t, &w, &x, kw, n, true);
        let o: Vec<f32> = x
            .chunks(n)
            .flat_map(|xt| ops::conv_step(&mut st_o, &w, xt, kw))
            .collect();
        close(Kernel::ConvSeq, &format!("n={n}, {tokens} tokens"), &t, &o);
        assert_eq!(st_t, st_o, "the state holds the last raw inputs");
    }
}

#[test]
fn gdn_seq_twin_matches_ops_over_tokens() {
    let mut rng = Rng::new(53);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (nv, dk, dv, ng, tokens) = (
            d.ssm_dt_rank,
            d.ssm_d_state,
            d.v_head_dim(),
            d.ssm_n_group,
            6,
        );
        let mut st_t = rand(&mut rng, nv * dv * dk, -0.2, 0.2);
        let mut st_o = st_t.clone();
        let q = unit_groups(&mut rng, tokens * ng * dk, dk);
        let k = unit_groups(&mut rng, tokens * ng * dk, dk);
        let v = rand(&mut rng, tokens * nv * dv, -1.0, 1.0);
        let g = rand(&mut rng, tokens * nv, -2.0, 0.0);
        let beta = rand(&mut rng, tokens * nv, 0.05, 0.95);
        let t = twin::deltanet::gdn_seq(&mut st_t, &q, &k, &v, &g, &beta, ng, dk, dv, true);
        let mut o = Vec::new();
        for i in 0..tokens {
            let (kd, vd) = (ng * dk, nv * dv);
            o.extend(ops::delta_rule(
                &mut st_o,
                &q[i * kd..(i + 1) * kd],
                &k[i * kd..(i + 1) * kd],
                &v[i * vd..(i + 1) * vd],
                &g[i * nv..(i + 1) * nv],
                &beta[i * nv..(i + 1) * nv],
                &d,
            ));
        }
        close(Kernel::GdnSeq, &format!("dk={dk}, {tokens} tokens"), &t, &o);
        close(Kernel::GdnSeq, &format!("dk={dk}, the state"), &st_t, &st_o);
    }
}

#[test]
fn the_scan_twins_take_an_empty_chunk() {
    // No tokens: no output, and the state is left as it was.
    let d = Dims::tiny();
    let (n, kw) = (d.conv_dim(), d.ssm_conv);
    let mut st = vec![1.5f32; n * (kw - 1)];
    assert!(twin::deltanet::conv_seq(&mut st, &vec![0.5; n * kw], &[], kw, n, true).is_empty());
    assert!(st.iter().all(|&v| v == 1.5));
    let (dk, dv) = (d.ssm_d_state, d.v_head_dim());
    let mut st = vec![0.25f32; d.ssm_dt_rank * dv * dk];
    let o = twin::deltanet::gdn_seq(
        &mut st,
        &[],
        &[],
        &[],
        &[],
        &[],
        d.ssm_n_group,
        dk,
        dv,
        true,
    );
    assert!(o.is_empty() && st.iter().all(|&v| v == 0.25));
}

#[test]
fn the_scan_twins_leave_the_state_alone_without_store() {
    // The verify pass reads the state and never writes it: without `store` the outputs are the ones
    // `store` gives, and the state is byte for byte what it was.
    let mut rng = Rng::new(58);
    let d = Dims::tiny();
    let (n, kw, tokens) = (d.conv_dim(), d.ssm_conv, 5);
    let w = rand(&mut rng, n * kw, -0.5, 0.5);
    let x = rand(&mut rng, tokens * n, -2.0, 2.0);
    let st0 = rand(&mut rng, n * (kw - 1), -2.0, 2.0);
    let (mut kept, mut stored) = (st0.clone(), st0.clone());
    let y_kept = twin::deltanet::conv_seq(&mut kept, &w, &x, kw, n, false);
    let y_stored = twin::deltanet::conv_seq(&mut stored, &w, &x, kw, n, true);
    assert!(
        y_kept == y_stored,
        "conv_seq's outputs don't depend on store"
    );
    assert!(kept == st0, "conv_seq left the state alone");
    assert!(stored != st0, "and with store the same input does move it");

    let (nv, dk, dv, ng) = (d.ssm_dt_rank, d.ssm_d_state, d.v_head_dim(), d.ssm_n_group);
    let q = unit_groups(&mut rng, tokens * ng * dk, dk);
    let k = unit_groups(&mut rng, tokens * ng * dk, dk);
    let v = rand(&mut rng, tokens * nv * dv, -1.0, 1.0);
    let g = rand(&mut rng, tokens * nv, -2.0, 0.0);
    let beta = rand(&mut rng, tokens * nv, 0.05, 0.95);
    let st0 = rand(&mut rng, nv * dv * dk, -0.2, 0.2);
    let (mut kept, mut stored) = (st0.clone(), st0.clone());
    let o_kept = twin::deltanet::gdn_seq(&mut kept, &q, &k, &v, &g, &beta, ng, dk, dv, false);
    let o_stored = twin::deltanet::gdn_seq(&mut stored, &q, &k, &v, &g, &beta, ng, dk, dv, true);
    assert!(
        o_kept == o_stored,
        "gdn_seq's outputs don't depend on store"
    );
    assert!(kept == st0, "gdn_seq left the state alone");
    assert!(stored != st0, "and with store the same input does move it");
}

#[test]
fn the_token_sizes() {
    use kernels::tokens::{attn_tokens_part_bytes, moe_unique_bytes, MAX_TOKENS};
    assert_eq!(MAX_TOKENS, 16);
    // [tokens][n_head][max_split][hd + 2] f32.
    assert_eq!(
        attn_tokens_part_bytes(16, 256, 256, 16),
        16 * 16 * 256 * 258 * 4
    );
    assert_eq!(attn_tokens_part_bytes(16, 256, 256, 1), 16 * 256 * 258 * 4);
    // [count: u32][per unique: expert u32, mask u32]: at most one entry for each (token, choice).
    assert_eq!(moe_unique_bytes(16, 8), 4 + 8 * 128);
    assert_eq!(moe_unique_bytes(1, 8), 4 + 8 * 8);
}

#[test]
fn the_shaders_max_tokens_is_the_rust_one() {
    // quant.glsl's unit_dot_n sizes its accumulators by a constant of its own, which must be tokens::MAX_TOKENS.
    let quant = include_str!("../shaders/quant.glsl");
    let want = format!("const uint MAX_TOKENS = {}u;", kernels::tokens::MAX_TOKENS);
    assert!(quant.contains(&want), "quant.glsl lacks `{want}`");
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

/// The exact attention (`ops::attention`) of each of `tokens` queries over its own prefix, query i at position
/// `pos` + i.
fn exact_attention(
    q: &[f32],
    kf: &[f32],
    vf: &[f32],
    pos: usize,
    tokens: usize,
    d: &Dims,
) -> Vec<f32> {
    let w = d.n_head * d.head_dim;
    (0..tokens)
        .flat_map(|i| ops::attention(&q[i * w..(i + 1) * w], kf, vf, pos + i + 1, d))
        .collect()
}

#[test]
fn attn_prefill_twin_matches_ops_for_every_query() {
    // Each query against the exact attention over its own prefix: batches that start at 0 and later, across block
    // boundaries, with scores spread over several units as trained heads' are.
    let mut rng = Rng::new(57);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
        for (pos, tokens) in [(0, 40), (290, 9)] {
            let n_ctx = pos + tokens;
            let q = rand(&mut rng, tokens * nh * hd, -12.0, 12.0);
            let (kc, vc, kf, vf) = caches(&mut rng, n_ctx, nkv * hd);
            close(
                Kernel::AttnPrefill,
                &format!("hd={hd}, {tokens} queries from {pos}"),
                &twin::attention::attn_prefill(&q, &kc, &vc, pos, tokens, nh, nkv, hd),
                &exact_attention(&q, &kf, &vf, pos, tokens, &d),
            );
        }
    }
}

#[test]
fn attn_prefill_cm_twin_matches_ops_for_every_query() {
    // The split twin against the exact attention over each query's own prefix: from 0 and later, across 32-key blocks
    // and 64-row workgroups, scores spread over several units.
    let mut rng = Rng::new(61);
    for d in [Dims::QWEN36, Dims::tiny()] {
        let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
        for (pos, tokens) in [(0, 40), (290, 9), (0, 70), (1000, 65)] {
            let q = rand(&mut rng, tokens * nh * hd, -12.0, 12.0);
            let (kc, vc, kf, vf) = caches(&mut rng, pos + tokens, nkv * hd);
            close(
                Kernel::AttnPrefillCm,
                &format!("cm, hd={hd}, {tokens} queries from {pos}"),
                &twin::attention::attn_prefill_cm(&q, &kc, &vc, pos, tokens, nh, nkv, hd),
                &exact_attention(&q, &kf, &vf, pos, tokens, &d),
            );
        }
    }
}

#[test]
fn the_split_keeps_attention_near_its_exact_value() {
    // What splitting q and p buys: the split twin is within 1e-5 of the exact attention, where plain f16 q is over
    // 1e-4 away (estimate 5e-4) and plain f16 p over 3e-5 (estimate 1e-4). Plain f16 failed the prefill-against-decode
    // bounds on the matmuls.
    use twin::attention::{attn_prefill_cm_split, CmSplit};
    let mut rng = Rng::new(62);
    let d = Dims::QWEN36;
    let (nh, nkv, hd, pos, tokens) = (d.n_head, d.n_head_kv, d.head_dim, 290, 9);
    let q = rand(&mut rng, tokens * nh * hd, -12.0, 12.0);
    let (kc, vc, kf, vf) = caches(&mut rng, pos + tokens, nkv * hd);
    let exact = exact_attention(&q, &kf, &vf, pos, tokens, &d);
    let err = |s: CmSplit| {
        ops::rel_l2(
            &attn_prefill_cm_split(&q, &kc, &vc, pos, tokens, nh, nkv, hd, s),
            &exact,
        )
    };
    let (both, plain_q, plain_p) = (
        err(CmSplit::BOTH),
        err(CmSplit {
            q_lo: false,
            p_lo: true,
        }),
        err(CmSplit {
            q_lo: true,
            p_lo: false,
        }),
    );
    eprintln!("split {both:.3e}, plain q {plain_q:.3e}, plain p {plain_p:.3e}");
    assert!(both <= 1e-5, "the split is {both:.3e} from exact");
    assert!(
        plain_q >= 1e-4,
        "plain f16 q is only {plain_q:.3e} from exact"
    );
    assert!(
        plain_p >= 3e-5,
        "plain f16 p is only {plain_p:.3e} from exact"
    );
}

use kernels::dispatch::{moe_tiles, SlotLayout};

#[test]
fn moe_group_lists_each_experts_entries_in_token_order() {
    // Three tokens choosing two of four experts: [2, 0], [2, 3], [0, 2].
    let ids = [2.0, 0.0, 2.0, 3.0, 0.0, 2.0];
    let max = moe_tiles(ids.len(), 4);
    assert_eq!(max, 5);
    let (list, tiles) = twin::moe::moe_group(&ids, 4, max);
    assert_eq!(list, [1, 4, 0, 2, 5, 3]);
    assert_eq!(
        tiles,
        [[0, 0, 2, 0], [2, 2, 3, 0], [3, 5, 1, 0], [0; 4], [0; 4]]
    );
    // 200 entries of expert 5 make three full tiles and a part-full one; an id that names no expert is dropped.
    let mut ids = vec![5.0; 200];
    ids[7] = 300.0;
    ids[8] = f32::NAN;
    let (list, tiles) = twin::moe::moe_group(&ids, 256, moe_tiles(200, 256));
    assert_eq!(list.len(), 198);
    assert!(!list.contains(&7) && !list.contains(&8));
    assert_eq!(
        tiles[..5],
        [
            [5, 0, 64, 0],
            [5, 64, 64, 0],
            [5, 128, 64, 0],
            [5, 192, 6, 0],
            [0; 4]
        ]
    );
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
fn grouped_expert_twins_match_ops() {
    // Five tokens choosing two of six experts, through the grouped twins and through the reference's operations one
    // entry at a time.
    use qwen36::model::Weight;
    let mut rng = Rng::new(60);
    let lay = SlotLayout {
        gate: WType::Q4K,
        up: WType::Q4K,
        down: WType::Q6K,
        n_embd: 256,
        n_ff: 256,
    };
    let (n, k, tokens) = (6, 2, 5);
    let s = slots(&mut rng, &lay, n);
    let x = rand(&mut rng, tokens * lay.n_embd, -1.0, 1.0);
    let ids: Vec<f32> = [1, 4, 0, 1, 5, 2, 1, 3, 4, 0]
        .iter()
        .map(|&e| e as f32)
        .collect();
    let (list, _) = twin::moe::moe_group(&ids, n, moe_tiles(ids.len(), n));
    let act = twin::moe::experts_matmul_gate_up(|e| s[e].as_slice(), &lay, &x, &ids, &list, k);
    let ye = twin::moe::experts_matmul_down(|e| s[e].as_slice(), &lay, &act, &ids, &list);
    let w = |e: usize, at: usize, ty: WType, rows: usize, cols: usize| Weight {
        ty,
        cols,
        rows,
        bytes: &s[e][at..at + rows * ty.row_bytes(cols)],
    };
    let (mut act_o, mut ye_o) = (Vec::new(), vec![0f32; ids.len() * lay.n_embd]);
    for &entry in &list {
        let e = ids[entry as usize] as usize;
        let xt = &x[entry as usize / k * lay.n_embd..][..lay.n_embd];
        let g = ops::matvec(&w(e, 0, lay.gate, lay.n_ff, lay.n_embd), xt, 1);
        let u = ops::matvec(&w(e, lay.up_offset(), lay.up, lay.n_ff, lay.n_embd), xt, 1);
        let a = ops::swiglu(&g, &u);
        let y = ops::matvec(
            &w(e, lay.down_offset(), lay.down, lay.n_embd, lay.n_ff),
            &a,
            1,
        );
        ye_o[entry as usize * lay.n_embd..][..lay.n_embd].copy_from_slice(&y);
        act_o.extend(a);
    }
    close(Kernel::ExpertsMatmulGateUp, "act", &act, &act_o);
    close(Kernel::ExpertsMatmulDown, "ye", &ye, &ye_o);
}

/// Asserts that `h` is an f16 value and that no f16 is nearer to `target` than `h` (a tie may go either way).
fn nearest_f16(target: f32, h: f32, what: &str) {
    assert_eq!(
        half::f16::from_f32(h).to_f32(),
        h,
        "{what} = {h} is not an f16"
    );
    let bits = half::f16::from_f32(h).to_bits();
    let (sign, mag) = (bits & 0x8000, bits & 0x7FFF);
    // h's two neighbours among the f16s: one step up and one down in magnitude, or, for a zero, the smallest subnormal
    // of either sign. A step up from 65,504 is infinity, which is never nearer.
    let neighbours = if mag == 0 {
        [0x0001, 0x8001]
    } else {
        [sign | (mag + 1), sign | (mag - 1)]
    };
    for nb in neighbours {
        let n = half::f16::from_bits(nb).to_f32();
        let (dh, dn) = (
            (target as f64 - h as f64).abs(),
            (target as f64 - n as f64).abs(),
        );
        assert!(
            dh <= dn,
            "{what}: the f16 {n} is nearer to {target} than {h}"
        );
    }
}

#[test]
fn split16_holds_a_value_to_about_2_to_the_minus_22() {
    // hi is v rounded to f16 (nearest even), lo the rest rounded the same way: hi + lo is within 2⁻²¹ of v, or 2⁻²⁵
    // absolute where lo is subnormal (inputs split into two f16s).
    let mut rng = Rng::new(43);
    for _ in 0..10_000 {
        for scale in [1e-4f32, 1e-2, 1.0, 100.0] {
            let v = rng.uniform(-1.0, 1.0) * scale;
            let (hi, lo) = twin::matvec::split16(v);
            // Independently of how split16 rounds: hi is a nearest f16 to v, and lo a nearest f16 to the rest, v − hi,
            // which f32 holds exactly (so the bound below can't pass a truncated lo).
            nearest_f16(v, hi, &format!("{v}: hi"));
            let rest = v - hi;
            assert_eq!(rest as f64, v as f64 - hi as f64, "{v}: v − hi is exact");
            nearest_f16(rest, lo, &format!("{v}: lo"));
            let err = ((hi as f64 + lo as f64) - v as f64).abs();
            assert!(
                err <= v.abs() as f64 * 2f64.powi(-21) + 2f64.powi(-25),
                "{v}: {hi} + {lo} is {err:e} away"
            );
        }
    }
}

#[test]
fn matmul_cm_twins_match_ops_on_their_split_inputs() {
    // The split is the kernel's design; what its twin must get right is the sum. Against the reference's f64 dot of the
    // same split values (hi + lo of every weight and input), each output agrees within the kernel's tolerance: the twin
    // drops lo·lo (under 2⁻²² of each product) and adds in f32.
    let mut rng = Rng::new(44);
    let back = |v: f32| {
        let (h, l) = twin::matvec::split16(v);
        h as f64 + l as f64
    };
    for ty in WType::ALL {
        for (rows, cols, tokens) in [(37, 256, 5), (64, 512, 3), (8, 2048, 2), (3, 4096, 1)] {
            let w = kernels::testing::random_weights(&mut rng, ty, rows, cols);
            let x = rand(&mut rng, tokens * cols, -1.0, 1.0);
            let rb = ty.row_bytes(cols);
            let mut wr = vec![0f32; cols];
            let mut exact = vec![0f32; tokens * rows];
            for r in 0..rows {
                qwen36::quant::dequant_row(ty, &w[r * rb..(r + 1) * rb], &mut wr);
                for t in 0..tokens {
                    let xt = &x[t * cols..(t + 1) * cols];
                    exact[t * rows + r] = wr
                        .iter()
                        .zip(xt)
                        .map(|(&a, &b)| back(a) * back(b))
                        .sum::<f64>() as f32;
                }
            }
            close(
                Kernel::matmul_cm(ty),
                &format!("{} {rows}x{cols}, {tokens} tokens", ty.name()),
                &twin::matvec::matmul_cm(ty, &w, rows, &x, tokens),
                &exact,
            );
        }
    }
}

#[test]
fn the_split_keeps_a_matmul_near_its_exact_value() {
    // What the split buys (decision of 2026-10-10): against the exact product of the unsplit values, the split twin is
    // within 3e-5 (8.2e-6 measured on the GPU for Q8_0), where plain f16 inputs are over 1e-4 away (2.7e-4
    // measured). Plain f16 failed the prefill-against-decode bounds on the real model; the split passed.
    let mut rng = Rng::new(45);
    let (rows, cols, tokens) = (64, 2048, 4);
    for ty in [WType::Q8_0, WType::Q4K] {
        let w = kernels::testing::random_weights(&mut rng, ty, rows, cols);
        let x = rand(&mut rng, tokens * cols, -1.0, 1.0);
        let rb = ty.row_bytes(cols);
        let mut wr = vec![0f32; cols];
        let (mut exact, mut plain) = (vec![0f32; tokens * rows], vec![0f32; tokens * rows]);
        let h = |v: f32| half::f16::from_f32(v).to_f32() as f64;
        for r in 0..rows {
            qwen36::quant::dequant_row(ty, &w[r * rb..(r + 1) * rb], &mut wr);
            for t in 0..tokens {
                let xt = &x[t * cols..(t + 1) * cols];
                exact[t * rows + r] = wr
                    .iter()
                    .zip(xt)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum::<f64>() as f32;
                plain[t * rows + r] =
                    wr.iter().zip(xt).map(|(&a, &b)| h(a) * h(b)).sum::<f64>() as f32;
            }
        }
        let split = twin::matvec::matmul_cm(ty, &w, rows, &x, tokens);
        let (es, ep) = (ops::rel_l2(&split, &exact), ops::rel_l2(&plain, &exact));
        assert!(
            es <= 3e-5,
            "{}: the split is {es:.3e} from exact",
            ty.name()
        );
        assert!(
            ep >= 1e-4,
            "{}: plain f16 is only {ep:.3e} from exact, so the check can't tell them apart",
            ty.name()
        );
    }
}

#[test]
fn grouped_cm_twins_match_ops_on_their_split_inputs() {
    // grouped_expert_twins_match_ops's entries through the split-f16 twins, against the reference's f64 products of the
    // same split values (every weight and input as hi + lo), entry by entry. Down's input is the twin's own act row, as
    // the down kernel's is the gate/up kernel's output.
    let mut rng = Rng::new(61);
    let lay = SlotLayout {
        gate: WType::Q4K,
        up: WType::Q4K,
        down: WType::Q6K,
        n_embd: 256,
        n_ff: 256,
    };
    let (n, k, tokens) = (6, 2, 5);
    let s = slots(&mut rng, &lay, n);
    let x = rand(&mut rng, tokens * lay.n_embd, -1.0, 1.0);
    let ids: Vec<f32> = [1, 4, 0, 1, 5, 2, 1, 3, 4, 0]
        .iter()
        .map(|&e| e as f32)
        .collect();
    let (list, _) = twin::moe::moe_group(&ids, n, moe_tiles(ids.len(), n));
    let act = twin::moe::experts_matmul_cm_gate_up(|e| s[e].as_slice(), &lay, &x, &ids, &list, k);
    let ye = twin::moe::experts_matmul_cm_down(|e| s[e].as_slice(), &lay, &act, &ids, &list);
    let back = |v: f32| {
        let (h, l) = twin::matvec::split16(v);
        h as f64 + l as f64
    };
    let dot = |ty: WType, row: &[u8], x: &[f32]| -> f32 {
        let mut w = vec![0f32; x.len()];
        qwen36::quant::dequant_row(ty, row, &mut w);
        w.iter()
            .zip(x)
            .map(|(&a, &b)| back(a) * back(b))
            .sum::<f64>() as f32
    };
    let (rg, ru, rd) = (
        lay.gate.row_bytes(lay.n_embd),
        lay.up.row_bytes(lay.n_embd),
        lay.down.row_bytes(lay.n_ff),
    );
    let (mut act_o, mut ye_o) = (Vec::new(), vec![0f32; ids.len() * lay.n_embd]);
    for (i, &entry) in list.iter().enumerate() {
        let e = ids[entry as usize] as usize;
        let xt = &x[entry as usize / k * lay.n_embd..][..lay.n_embd];
        let g: Vec<f32> = (0..lay.n_ff)
            .map(|r| dot(lay.gate, &s[e][r * rg..][..rg], xt))
            .collect();
        let u: Vec<f32> = (0..lay.n_ff)
            .map(|r| dot(lay.up, &s[e][lay.up_offset() + r * ru..][..ru], xt))
            .collect();
        act_o.extend(ops::swiglu(&g, &u));
        let a = &act[i * lay.n_ff..(i + 1) * lay.n_ff];
        for r in 0..lay.n_embd {
            ye_o[entry as usize * lay.n_embd + r] =
                dot(lay.down, &s[e][lay.down_offset() + r * rd..][..rd], a);
        }
    }
    close(Kernel::ExpertsMatmulCmGateUp, "act", &act, &act_o);
    close(Kernel::ExpertsMatmulCmDown, "ye", &ye, &ye_o);
    // The bound above pins the wiring, not the split: the f32 twins would pass it too. So the split twins must also give
    // other bits than the f32 twins on the same inputs (down's being the same act rows).
    let differ = |a: &[f32], b: &[f32]| a.iter().zip(b).any(|(p, q)| p.to_bits() != q.to_bits());
    let act32 = twin::moe::experts_matmul_gate_up(|e| s[e].as_slice(), &lay, &x, &ids, &list, k);
    let ye32 = twin::moe::experts_matmul_down(|e| s[e].as_slice(), &lay, &act, &ids, &list);
    assert!(
        differ(&act, &act32),
        "the split gate/up twin gives the f32 twin's bits"
    );
    assert!(
        differ(&ye, &ye32),
        "the split down twin gives the f32 twin's bits"
    );
}
