//! The MTP block (`blk.4` in the tiny model) in the CPU reference: its rows, its taps and what it
//! reads. Agreement with llama.cpp's MTP graph is test 2's MTP half.
use gguf::Gguf;
use qwen36::{
    dims::Dims,
    model::Model,
    ops, quant,
    reference::{MtpReference, Reference, Rounding},
    taps::{expected_mtp_taps, MtpKey, NullTap, Tap, TapName},
    testing::tiny_model,
};
use std::collections::HashMap;

#[derive(Default)]
struct Rec {
    last: HashMap<TapName, Vec<f32>>,
    order: Vec<TapName>,
}

impl Tap for Rec {
    fn tap(&mut self, name: TapName, _pos: u32, data: &[f32]) {
        self.order.push(name);
        self.last.insert(name, data.to_vec());
    }
}

const TOKENS: [u32; 8] = [10, 300, 7, 506, 42, 42, 1, 511];

fn tiny() -> Gguf {
    Gguf::from_bytes(tiny_model(1)).unwrap()
}

/// The main model's final hidden row at each position of `TOKENS`.
fn hiddens(m: &Model, rounding: Rounding) -> Vec<Vec<f32>> {
    let mut r = Reference::with_options(m, 4, rounding);
    TOKENS
        .iter()
        .map(|&t| {
            r.step(t, &mut NullTap).unwrap();
            r.hidden().unwrap().to_vec()
        })
        .collect()
}

/// The MTP's row at `q` takes `TOKENS[q]` and the main model's hidden row at `q − 1`.
fn row(
    mtp: &mut MtpReference,
    h: &[Vec<f32>],
    q: usize,
    tap: &mut dyn Tap,
) -> (Vec<f32>, Vec<f32>) {
    mtp.row(q as u32, TOKENS[q], &h[q - 1], tap)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn the_hidden_row_is_the_final_norm_tap() {
    let g = tiny();
    let m = Model::open(&g, &Dims::tiny()).unwrap();
    let mut r = Reference::with_threads(&m, 4);
    assert!(r.hidden().is_none(), "no hidden row before a step");
    for &t in &TOKENS[..3] {
        let mut rec = Rec::default();
        r.step(t, &mut rec).unwrap();
        assert_eq!(r.hidden().unwrap(), &rec.last[&TapName::FinalNorm][..]);
        assert_eq!(r.hidden().unwrap().len(), 256);
    }
}

#[test]
fn a_kv_only_row_writes_what_a_whole_row_writes() {
    let g = tiny();
    let m = Model::open(&g, &Dims::tiny()).unwrap();
    for rounding in [Rounding::Exact, Rounding::F16Kv, Rounding::LlamaCpu] {
        let h = hiddens(&m, rounding);
        let mut whole = MtpReference::new(&m, 4, rounding);
        for q in 1..=3 {
            row(&mut whole, &h, q, &mut NullTap);
        }
        let (want_logits, want_m) = row(&mut whole, &h, 4, &mut NullTap);
        let mut kv = MtpReference::new(&m, 4, rounding);
        for q in 1..=3 {
            kv.kv_row(q as u32, TOKENS[q], &h[q - 1]);
        }
        assert_eq!(kv.positions(), &[1, 2, 3]);
        let (logits, mrow) = row(&mut kv, &h, 4, &mut NullTap);
        assert_eq!(bits(&logits), bits(&want_logits), "{rounding:?}");
        assert_eq!(bits(&mrow), bits(&want_m), "{rounding:?}");
    }
}

#[test]
fn the_embedding_half_comes_first() {
    let d = Dims::tiny();
    let g = tiny();
    let m = Model::open(&g, &d).unwrap();
    let h = hiddens(&m, Rounding::Exact);
    let mut mtp = MtpReference::new(&m, 4, Rounding::Exact);
    let mut rec = Rec::default();
    let q = 3;
    row(&mut mtp, &h, q, &mut rec);
    let mut embd = vec![0f32; d.n_embd];
    quant::dequant_row(m.embd.ty, m.embd.row(TOKENS[q] as usize), &mut embd);
    let e = ops::rmsnorm(&embd, &m.mtp.enorm.to_f32(), d.rms_eps);
    let gh = ops::rmsnorm(&h[q - 1], &m.mtp.hnorm.to_f32(), d.rms_eps);
    assert_ne!(e, gh, "the two halves differ on this input");
    let cat = &rec.last[&TapName::Mtp(MtpKey::Concat)];
    assert_eq!(cat.len(), 2 * d.n_embd);
    assert_eq!(&cat[..d.n_embd], &e[..], "the embedding half first");
    assert_eq!(&cat[d.n_embd..], &gh[..], "the hidden half second");
    assert_eq!(&rec.last[&TapName::Mtp(MtpKey::Enorm)], &e);
    assert_eq!(&rec.last[&TapName::Mtp(MtpKey::Hnorm)], &gh);
    let u = ops::matvec(&m.mtp.eh_proj, cat, 4);
    assert_eq!(&rec.last[&TapName::Mtp(MtpKey::EhProj)], &u);
}

#[test]
fn every_mtp_tap_in_order_at_the_right_size() {
    let d = Dims::tiny();
    let g = tiny();
    let m = Model::open(&g, &d).unwrap();
    let h = hiddens(&m, Rounding::Exact);
    let mut mtp = MtpReference::new(&m, 4, Rounding::Exact);
    let mut rec = Rec::default();
    row(&mut mtp, &h, 2, &mut rec);
    assert_eq!(rec.order, expected_mtp_taps());
    assert_eq!(rec.order.len(), 25);
    for (name, v) in &rec.last {
        assert_eq!(v.len(), name.per_pos(&d), "{name}");
        assert!(v.iter().all(|x| x.is_finite()), "{name}");
    }
}

#[test]
fn the_mtp_attends_only_its_own_rows() {
    let d = Dims::tiny();
    let g = tiny();
    let m = Model::open(&g, &d).unwrap();
    let h = hiddens(&m, Rounding::Exact);
    // Rows at 2 and 4, a hole at 3, and the row at 5: three rows to attend.
    let mut mtp = MtpReference::new(&m, 4, Rounding::Exact);
    let mut taps = Vec::new();
    for q in [2usize, 4, 5] {
        let mut rec = Rec::default();
        row(&mut mtp, &h, q, &mut rec);
        taps.push((q, rec));
    }
    let (kvw, hd) = (d.n_head_kv * d.head_dim, d.head_dim);
    let (mut kc, mut vc) = (Vec::new(), Vec::new());
    for (q, rec) in &taps {
        // The K tap is after the norm and before RoPE (`mtp_Kcur_normed`); the cache holds it after.
        let mut k = rec.last[&TapName::Mtp(MtpKey::K)].clone();
        ops::rope_heads(&mut k, hd, *q as u32, d.n_rot, d.rope_base);
        kc.extend(k);
        vc.extend(rec.last[&TapName::Mtp(MtpKey::V)].clone());
    }
    assert_eq!(kc.len(), 3 * kvw);
    let (q5, rec5) = (&taps[2].0, &taps[2].1);
    let mut q = rec5.last[&TapName::Mtp(MtpKey::Q)].clone();
    ops::rope_heads(&mut q, hd, *q5 as u32, d.n_rot, d.rope_base);
    let want = ops::attention(&q, &kc, &vc, 3, &d);
    assert_eq!(
        bits(&rec5.last[&TapName::Mtp(MtpKey::AttnPregate)]),
        bits(&want),
        "the query attends the three rows, in order, and nothing at the hole"
    );
    // Its logits depend on the earlier rows written: one more row at 3 changes them.
    let (logits, _) = {
        let mut a = MtpReference::new(&m, 4, Rounding::Exact);
        for q in [2usize, 4] {
            row(&mut a, &h, q, &mut NullTap);
        }
        row(&mut a, &h, 5, &mut NullTap)
    };
    assert_eq!(
        bits(&logits),
        bits(&taps[2].1.last[&TapName::Mtp(MtpKey::Logits)])
    );
    let (with_three, _) = {
        let mut b = MtpReference::new(&m, 4, Rounding::Exact);
        for q in [2usize, 3, 4] {
            row(&mut b, &h, q, &mut NullTap);
        }
        row(&mut b, &h, 5, &mut NullTap)
    };
    assert_ne!(
        bits(&logits),
        bits(&with_three),
        "an extra earlier row matters"
    );
    let (without_two, _) = {
        let mut c = MtpReference::new(&m, 4, Rounding::Exact);
        row(&mut c, &h, 4, &mut NullTap);
        row(&mut c, &h, 5, &mut NullTap)
    };
    assert_ne!(
        bits(&logits),
        bits(&without_two),
        "a missing earlier row matters"
    );
}

#[test]
fn truncate_drops_the_rows_at_and_past_the_cut() {
    let g = tiny();
    let m = Model::open(&g, &Dims::tiny()).unwrap();
    let h = hiddens(&m, Rounding::Exact);
    let mut mtp = MtpReference::new(&m, 4, Rounding::Exact);
    for q in 1..=4 {
        row(&mut mtp, &h, q, &mut NullTap);
    }
    mtp.truncate(9);
    assert_eq!(
        mtp.positions(),
        &[1, 2, 3, 4],
        "a cut past the end drops nothing"
    );
    mtp.truncate(3);
    assert_eq!(mtp.positions(), &[1, 2]);
    // Redoing position 3 gives what a session that never ran 3 and 4 gives.
    let (a, am) = row(&mut mtp, &h, 3, &mut NullTap);
    let (b, bm) = {
        let mut f = MtpReference::new(&m, 4, Rounding::Exact);
        row(&mut f, &h, 1, &mut NullTap);
        row(&mut f, &h, 2, &mut NullTap);
        row(&mut f, &h, 3, &mut NullTap)
    };
    assert_eq!(bits(&a), bits(&b));
    assert_eq!(bits(&am), bits(&bm));
    mtp.truncate(0);
    assert!(mtp.positions().is_empty());
}

#[test]
fn the_mtp_reads_the_main_embedding_and_head() {
    let d = Dims::tiny();
    let g = tiny();
    // The GGUF carries no MTP-private embedding or head (semantics reference §6).
    assert!(g.tensor("blk.4.nextn.embed_tokens.weight").is_none());
    assert!(g.tensor("blk.4.nextn.shared_head_head.weight").is_none());
    let m = Model::open(&g, &d).unwrap();
    let h = hiddens(&m, Rounding::Exact);
    let mut mtp = MtpReference::new(&m, 4, Rounding::Exact);
    let mut rec = Rec::default();
    let (logits, mrow) = row(&mut mtp, &h, 1, &mut rec);
    // The head: the main `output` over m = RMSNorm(y) ⊙ shared_head_norm.
    assert_eq!(bits(&logits), bits(&ops::matvec(&m.output, &mrow, 4)));
    let y = &rec.last[&TapName::Mtp(MtpKey::PostFfn)];
    let want = ops::rmsnorm(y, &m.mtp.shared_head_norm.to_f32(), d.rms_eps);
    assert_eq!(bits(&mrow), bits(&want));
    assert_eq!(
        bits(&rec.last[&TapName::Mtp(MtpKey::HeadNorm)]),
        bits(&mrow)
    );
    assert_eq!(logits.len(), d.n_vocab);
    // The embedding: the main `token_embd` row, so another token's differs.
    let mut other = MtpReference::new(&m, 4, Rounding::Exact);
    let mut rec2 = Rec::default();
    other.row(1, TOKENS[1] + 1, &h[0], &mut rec2);
    assert_ne!(
        rec.last[&TapName::Mtp(MtpKey::Enorm)],
        rec2.last[&TapName::Mtp(MtpKey::Enorm)]
    );
}

#[test]
fn a_chained_row_takes_m_in_place_of_h() {
    let g = tiny();
    let m = Model::open(&g, &Dims::tiny()).unwrap();
    let h = hiddens(&m, Rounding::Exact);
    let mut mtp = MtpReference::new(&m, 4, Rounding::Exact);
    let (_, m1) = row(&mut mtp, &h, 1, &mut NullTap);
    let (l2, _) = mtp.row(2, TOKENS[2], &m1, &mut NullTap);
    let mut alt = MtpReference::new(&m, 4, Rounding::Exact);
    row(&mut alt, &h, 1, &mut NullTap);
    let (l2h, _) = row(&mut alt, &h, 2, &mut NullTap);
    assert_ne!(bits(&l2), bits(&l2h), "m and h are different rows");
}

#[test]
#[should_panic(expected = "outside the vocabulary")]
fn a_token_outside_the_vocabulary_panics() {
    let g = tiny();
    let m = Model::open(&g, &Dims::tiny()).unwrap();
    let h = hiddens(&m, Rounding::Exact);
    let mut mtp = MtpReference::new(&m, 4, Rounding::Exact);
    mtp.row(1, 512, &h[0], &mut NullTap);
}

#[test]
#[should_panic(expected = "ascend")]
fn a_row_not_past_the_last_panics() {
    let g = tiny();
    let m = Model::open(&g, &Dims::tiny()).unwrap();
    let h = hiddens(&m, Rounding::Exact);
    let mut mtp = MtpReference::new(&m, 4, Rounding::Exact);
    row(&mut mtp, &h, 2, &mut NullTap);
    row(&mut mtp, &h, 2, &mut NullTap);
}

/// Teacher forcing continues from the supplied value, as it does for the main layers.
#[test]
fn a_forced_mtp_tap_continues_from_the_forced_value() {
    struct Force(Vec<f32>);
    impl Tap for Force {
        fn tap(&mut self, _: TapName, _: u32, _: &[f32]) {}
        fn forced(&self, name: TapName, _: u32) -> Option<&[f32]> {
            (name == TapName::Mtp(MtpKey::EhProj)).then_some(&self.0[..])
        }
    }
    let d = Dims::tiny();
    let g = tiny();
    let m = Model::open(&g, &d).unwrap();
    let h = hiddens(&m, Rounding::Exact);
    let mut mtp = MtpReference::new(&m, 4, Rounding::Exact);
    let (free, _) = row(&mut mtp, &h, 1, &mut NullTap);
    let mut rec = Rec::default();
    row(
        &mut MtpReference::new(&m, 4, Rounding::Exact),
        &h,
        1,
        &mut rec,
    );
    let mut forced = rec.last[&TapName::Mtp(MtpKey::EhProj)].clone();
    forced[0] += 1.0;
    let (l, _) = row(
        &mut MtpReference::new(&m, 4, Rounding::Exact),
        &h,
        1,
        &mut Force(forced),
    );
    assert_ne!(bits(&l), bits(&free));
}

/// Each downstream tap recomputed from the taps upstream of it: the residual comes from `u` (the `eh_proj` output),
/// not from the normed row; the two norms aren't swapped; no residual is dropped.
#[test]
fn the_taps_chain_as_the_graph_does() {
    let d = Dims::tiny();
    let g = tiny();
    let m = Model::open(&g, &d).unwrap();
    let h = hiddens(&m, Rounding::Exact);
    let mut mtp = MtpReference::new(&m, 4, Rounding::Exact);
    let mut rec = Rec::default();
    row(&mut mtp, &h, 3, &mut rec);
    let t = |k| &rec.last[&TapName::Mtp(k)];
    let add = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
    let l = &m.mtp.layer;
    assert_eq!(
        bits(t(MtpKey::AttnNorm)),
        bits(&ops::rmsnorm(
            t(MtpKey::EhProj),
            &l.attn_norm.to_f32(),
            d.rms_eps
        ))
    );
    assert_eq!(
        bits(t(MtpKey::AttnResidual)),
        bits(&add(t(MtpKey::EhProj), t(MtpKey::AttnOut)))
    );
    assert_eq!(
        bits(t(MtpKey::PostNorm)),
        bits(&ops::rmsnorm(
            t(MtpKey::AttnResidual),
            &l.post_attention_norm.to_f32(),
            d.rms_eps
        ))
    );
    assert_eq!(
        bits(t(MtpKey::FfnOut)),
        bits(&add(t(MtpKey::MoeOut), t(MtpKey::ShexpGated)))
    );
    assert_eq!(
        bits(t(MtpKey::PostFfn)),
        bits(&add(t(MtpKey::AttnResidual), t(MtpKey::FfnOut)))
    );
    // The router intermediates chain too: probs from logits, the routed experts the top 8 of the probs.
    assert_eq!(
        bits(t(MtpKey::RouterProbs)),
        bits(&ops::softmax(t(MtpKey::RouterLogits)))
    );
    let ids: Vec<usize> = t(MtpKey::TopK).iter().map(|&e| e as usize).collect();
    assert_eq!(ids, ops::top_k(t(MtpKey::RouterProbs), d.n_expert_used));
    assert_eq!(
        bits(t(MtpKey::TopKW)),
        bits(&ops::route_weights(t(MtpKey::RouterProbs), &ids))
    );
    assert_eq!(
        bits(t(MtpKey::Logits)),
        bits(&ops::matvec(
            &m.output,
            &ops::rmsnorm(
                t(MtpKey::PostFfn),
                &m.mtp.shared_head_norm.to_f32(),
                d.rms_eps
            ),
            4
        ))
    );
}

/// In llama.cpp's CPU mode the shared-expert gate's BF16 matmul rounds its activation to BF16 first (blk.40's gate is
/// BF16; the main layers' are F32).
#[test]
fn the_llama_cpu_mode_rounds_the_bf16_shared_gate_activation() {
    let d = Dims::tiny();
    let g = tiny();
    let m = Model::open(&g, &d).unwrap();
    assert_eq!(m.mtp.layer.shared.gate_inp.ty, qwen36::quant::WType::Bf16);
    let h = hiddens(&m, Rounding::LlamaCpu);
    let mut mtp = MtpReference::new(&m, 4, Rounding::LlamaCpu);
    let mut rec = Rec::default();
    row(&mut mtp, &h, 2, &mut rec);
    let t = |k| &rec.last[&TapName::Mtp(k)];
    let l = &m.mtp.layer.shared;
    let f = t(MtpKey::PostNorm);
    let rounded = quant::round_like_llama_cpu(l.gate_inp.ty, f);
    assert_ne!(&rounded[..], &f[..], "the activation isn't already BF16");
    let sact = t(MtpKey::ShexpSwiglu);
    let sh = ops::matvec(&l.down, &quant::round_like_llama_cpu(l.down.ty, sact), 4);
    let gate = l.gate_inp.to_f32();
    let sg = ops::sigmoid(ops::dot(&gate, &rounded) as f32);
    let want: Vec<f32> = sh.iter().map(|v| v * sg).collect();
    assert_eq!(bits(t(MtpKey::ShexpGated)), bits(&want));
    let unrounded = ops::sigmoid(ops::dot(&gate, f) as f32);
    assert_ne!(
        sg.to_bits(),
        unrounded.to_bits(),
        "rounding changes the gate"
    );
}
