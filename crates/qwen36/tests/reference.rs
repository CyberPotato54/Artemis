//! The reference on the tiny model: it runs, taps everything, and obeys identities the math guarantees. Agreement
//! with llama.cpp is test 2.
use gguf::Gguf;
use qwen36::{
    dims::Dims,
    model::Model,
    reference::{Reference, Rounding, StepError},
    taps::{expected_taps, Key, NullTap, Tap, TapName},
    testing::tiny_model,
};
use std::collections::HashMap;

#[derive(Default)]
struct Rec(HashMap<TapName, Vec<Vec<f32>>>);

impl Tap for Rec {
    fn tap(&mut self, name: TapName, pos: u32, data: &[f32]) {
        let v = self.0.entry(name).or_default();
        assert_eq!(v.len(), pos as usize, "{name} arrived out of order");
        v.push(data.to_vec());
    }
}

const TOKENS: [u32; 8] = [10, 300, 7, 506, 42, 42, 1, 511];

fn run() -> (Rec, Vec<Vec<f32>>) {
    let g = Gguf::from_bytes(tiny_model(1)).unwrap();
    let m = Model::open(&g, &Dims::tiny()).unwrap();
    let mut r = Reference::with_threads(&m, 4);
    let mut rec = Rec::default();
    let logits = TOKENS
        .iter()
        .map(|&t| r.step(t, &mut rec).unwrap())
        .collect();
    assert_eq!(r.pos(), 8);
    (rec, logits)
}

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * y.abs().max(1.0),
            "{what}[{i}]: {x} vs {y}"
        );
    }
}

#[test]
fn every_tap_at_every_position() {
    let d = Dims::tiny();
    let (rec, logits) = run();
    for t in expected_taps(&d) {
        let v = rec.0.get(&t).unwrap_or_else(|| panic!("{t} never tapped"));
        assert_eq!(v.len(), TOKENS.len(), "{t}");
        assert!(v.iter().all(|p| p.len() == t.per_pos(&d)), "{t} size");
        assert!(
            v.iter().flatten().all(|x| x.is_finite()),
            "{t} has a non-finite value"
        );
    }
    assert_eq!(rec.0.len(), expected_taps(&d).len(), "no extra taps");
    assert!(logits
        .iter()
        .all(|l| l.len() == 512 && l.iter().all(|x| x.is_finite())));
    assert_eq!(&rec.0[&TapName::Logits][7], &logits[7]);
}

#[test]
fn deterministic() {
    let (_, a) = run();
    let (_, b) = run();
    assert_eq!(a, b);
}

#[test]
fn token_outside_the_vocabulary_is_an_error() {
    let g = Gguf::from_bytes(tiny_model(1)).unwrap();
    let m = Model::open(&g, &Dims::tiny()).unwrap();
    let mut r = Reference::new(&m);
    let e = r.step(512, &mut NullTap).unwrap_err();
    assert_eq!(
        e,
        StepError::TokenOutOfRange {
            token: 512,
            n_vocab: 512
        }
    );
    assert_eq!(
        e.to_string(),
        "token 512 is outside the vocabulary (512 tokens)"
    );
    assert_eq!(r.pos(), 0);
}

#[test]
fn first_attention_token_returns_v() {
    // At position 0 softmax has one entry, so each query head's output is its KV head's v.
    let d = Dims::tiny();
    let (rec, _) = run();
    let pre = &rec.0[&TapName::Layer(3, Key::AttnPregate)][0];
    let v = &rec.0[&TapName::Layer(3, Key::AttnV)][0];
    let group = d.n_head / d.n_head_kv;
    for i in 0..d.n_head {
        let kvh = i / group;
        close(
            &pre[i * d.head_dim..(i + 1) * d.head_dim],
            &v[kvh * d.head_dim..(kvh + 1) * d.head_dim],
            1e-6,
            "head",
        );
    }
}

#[test]
fn first_deltanet_token_closed_form() {
    // From a zero state: S = k̂ (β v)ᵀ, so o[h][j] = (k̂·q̂) β_h v[h][j] / sqrt(d_k), with group h % n_group.
    let d = Dims::tiny();
    let (rec, _) = run();
    let at = |k| &rec.0[&TapName::Layer(0, k)][0];
    let (q, k, v, beta, o) = (
        at(Key::GdnQ),
        at(Key::GdnK),
        at(Key::GdnV),
        at(Key::GdnBeta),
        at(Key::GdnO),
    );
    let (dk, dv) = (d.ssm_d_state, d.v_head_dim());
    let mut want = vec![0f32; d.ssm_dt_rank * dv];
    for h in 0..d.ssm_dt_rank {
        let g = h % d.ssm_n_group;
        let kq: f64 = (0..dk)
            .map(|i| k[g * dk + i] as f64 * q[g * dk + i] as f64)
            .sum();
        for j in 0..dv {
            want[h * dv + j] =
                (kq * beta[h] as f64 * v[h * dv + j] as f64 / (dk as f64).sqrt()) as f32;
        }
    }
    close(o, &want, 1e-5, "gdn.o");
}

#[test]
fn router_weights_are_a_distribution_over_distinct_experts() {
    let d = Dims::tiny();
    let (rec, _) = run();
    for l in 0..d.n_layer as u16 {
        for p in 0..TOKENS.len() {
            let ids = &rec.0[&TapName::Layer(l, Key::TopK)][p];
            let w = &rec.0[&TapName::Layer(l, Key::TopKW)][p];
            let probs = &rec.0[&TapName::Layer(l, Key::RouterProbs)][p];
            let sum: f32 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "layer {l} pos {p}: weights sum to {sum}"
            );
            let mut sorted = ids.clone();
            sorted.sort_by(f32::total_cmp);
            sorted.dedup();
            assert_eq!(sorted.len(), d.n_expert_used, "distinct experts");
            for pair in ids.windows(2) {
                assert!(
                    probs[pair[0] as usize] >= probs[pair[1] as usize],
                    "layer {l} pos {p}: sorted by probability"
                );
            }
        }
    }
}

fn run_rounding(rounding: Rounding) -> Rec {
    let g = Gguf::from_bytes(tiny_model(1)).unwrap();
    let m = Model::open(&g, &Dims::tiny()).unwrap();
    let mut r = Reference::with_options(&m, 4, rounding);
    let mut rec = Rec::default();
    for &t in &TOKENS {
        r.step(t, &mut rec).unwrap();
    }
    rec
}

#[test]
fn llama_rounding_first_attention_token_is_v_in_f16() {
    // Flash attention over one position: the f16 accumulator holds exactly f16(v), and the sum is 1.
    let d = Dims::tiny();
    let rec = run_rounding(Rounding::LlamaCpu);
    let pre = &rec.0[&TapName::Layer(3, Key::AttnPregate)][0];
    let v = &rec.0[&TapName::Layer(3, Key::AttnV)][0];
    let group = d.n_head / d.n_head_kv;
    for i in 0..d.n_head {
        for j in 0..d.head_dim {
            let want = half::f16::from_f32(v[(i / group) * d.head_dim + j]).to_f32();
            assert_eq!(pre[i * d.head_dim + j], want, "head {i} dim {j}");
        }
    }
}

#[test]
fn llama_rounding_only_perturbs_matmul_outputs() {
    // Same math, rounded inputs: layer 0's QKV moves by roughly Q8_0's activation error, not more.
    let exact = run_rounding(Rounding::Exact);
    let llama = run_rounding(Rounding::LlamaCpu);
    let t = TapName::Layer(0, Key::GdnQkv);
    for p in 0..TOKENS.len() {
        let (a, b) = (&llama.0[&t][p], &exact.0[&t][p]);
        let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
        let rel = (num / den).sqrt();
        assert!(
            rel > 1e-5 && rel < 3e-2,
            "position {p}: relative change {rel:e}"
        );
    }
    assert_eq!(
        exact.0[&TapName::Embd],
        llama.0[&TapName::Embd],
        "the embedding isn't a matmul"
    );
}

/// Supplies another run's values at every tap: teacher forcing.
struct Force<'a> {
    rec: Rec,
    from: &'a Rec,
}

impl Tap for Force<'_> {
    fn tap(&mut self, name: TapName, pos: u32, data: &[f32]) {
        self.rec.tap(name, pos, data);
    }

    fn forced(&self, name: TapName, pos: u32) -> Option<&[f32]> {
        self.from.0.get(&name).map(|v| v[pos as usize].as_slice())
    }
}

#[test]
fn teacher_forcing_continues_from_the_supplied_values() {
    // Run A on TOKENS; run B on other tokens, forced from A. B's embeddings are its own, but every later tap is
    // computed from A's forced values, so it equals A's exactly.
    let d = Dims::tiny();
    let g = Gguf::from_bytes(tiny_model(1)).unwrap();
    let m = Model::open(&g, &d).unwrap();
    let a = run_rounding(Rounding::Exact);
    let mut r = Reference::with_options(&m, 4, Rounding::Exact);
    let mut b = Force {
        rec: Rec::default(),
        from: &a,
    };
    for &t in &[3u32, 4, 5, 6, 7, 8, 9, 300] {
        r.step(t, &mut b).unwrap();
    }
    assert_ne!(b.rec.0[&TapName::Embd], a.0[&TapName::Embd]);
    for t in expected_taps(&d)
        .into_iter()
        .filter(|t| *t != TapName::Embd)
    {
        assert_eq!(b.rec.0[&t], a.0[&t], "{t}");
    }
}

/// Forces from `from`, except that tap `swap` at the last position takes `alt`'s value.
struct ForceSwap<'a> {
    rec: Rec,
    from: &'a Rec,
    alt: &'a Rec,
    swap: Option<TapName>,
}

impl Tap for ForceSwap<'_> {
    fn tap(&mut self, name: TapName, pos: u32, data: &[f32]) {
        self.rec.tap(name, pos, data);
    }

    fn forced(&self, name: TapName, pos: u32) -> Option<&[f32]> {
        let last = TOKENS.len() as u32 - 1;
        let src = if Some(name) == self.swap && pos == last {
            self.alt
        } else {
            self.from
        };
        src.0.get(&name).map(|v| v[pos as usize].as_slice())
    }
}

/// The taps computed directly from tap `x` within one position, in the tiny model (layers 0 to 2 DeltaNet, 3
/// attention).
fn consumers(x: TapName) -> Vec<TapName> {
    use Key::*;
    let TapName::Layer(l, k) = x else {
        return match x {
            TapName::Embd => vec![TapName::Layer(0, AttnNorm), TapName::Layer(0, Resid)],
            TapName::FinalNorm => vec![TapName::Logits],
            _ => vec![],
        };
    };
    let at = |keys: &[Key]| {
        keys.iter()
            .map(|&k| TapName::Layer(l, k))
            .collect::<Vec<_>>()
    };
    match k {
        AttnNorm if l == 3 => at(&[AttnQ, AttnK, AttnV, AttnGated]),
        AttnNorm => at(&[GdnQkv, GdnZ, GdnBeta, GdnG]),
        GdnQkv => at(&[GdnConv]),
        GdnZ | GdnO => at(&[GdnNormed]),
        GdnBeta | GdnG | GdnQ | GdnK | GdnV => at(&[GdnO]),
        GdnConv => at(&[GdnQ, GdnK, GdnV]),
        GdnNormed | AttnGated => at(&[MixerOut]),
        AttnQ | AttnK | AttnV => at(&[AttnPregate]),
        AttnPregate => at(&[AttnGated]),
        MixerOut => at(&[Resid]),
        Resid => at(&[FfnNorm, Out]),
        FfnNorm => at(&[RouterLogits, MoeSwiglu, ShexpSwiglu, ShexpOut]),
        RouterLogits => at(&[RouterProbs]),
        TopKW | MoeSwiglu => at(&[MoeOut]),
        MoeOut | ShexpOut => at(&[FfnOut]),
        ShexpSwiglu => at(&[ShexpOut]),
        FfnOut => at(&[Out]),
        Out if l == 3 => vec![TapName::FinalNorm],
        Out => vec![
            TapName::Layer(l + 1, AttnNorm),
            TapName::Layer(l + 1, Resid),
        ],
        // Swapping probabilities or ids may or may not change which experts win; not used as a probe.
        RouterProbs | TopK => unreachable!("not probed"),
    }
}

#[test]
fn teacher_forcing_feeds_every_tap_and_nothing_else() {
    // Forced from run A, swapping one tap's value at the last position for run C's must change exactly that tap's
    // direct consumers there. A tap whose force is missing changes nothing downstream; a missing force further on
    // lets the change leak to more taps. Probed for every tap kind in a DeltaNet layer and the attention layer.
    let d = Dims::tiny();
    let g = Gguf::from_bytes(tiny_model(1)).unwrap();
    let m = Model::open(&g, &d).unwrap();
    let a = run_rounding(Rounding::Exact);
    let c = {
        let mut r = Reference::with_options(&m, 4, Rounding::Exact);
        let mut rec = Rec::default();
        for &t in &[3u32, 4, 5, 6, 7, 8, 9, 300] {
            r.step(t, &mut rec).unwrap();
        }
        rec
    };
    let forced = |swap: Option<TapName>| {
        let mut r = Reference::with_options(&m, 4, Rounding::Exact);
        let mut f = ForceSwap {
            rec: Rec::default(),
            from: &a,
            alt: &c,
            swap,
        };
        for &t in &TOKENS {
            r.step(t, &mut f).unwrap();
        }
        f.rec
    };
    let base = forced(None);
    let last = TOKENS.len() - 1;
    let probes = expected_taps(&d).into_iter().filter(|t| match t {
        TapName::Layer(l, k) => (*l == 0 || *l == 3) && !matches!(k, Key::RouterProbs | Key::TopK),
        TapName::Logits => false,
        _ => true,
    });
    for x in probes {
        let b = forced(Some(x));
        let mut changed: Vec<(TapName, usize)> = Vec::new();
        for (t, per_pos) in &base.0 {
            for (p, v) in per_pos.iter().enumerate() {
                if b.0[t][p] != *v {
                    changed.push((*t, p));
                }
            }
        }
        changed.sort();
        let mut want: Vec<(TapName, usize)> = consumers(x).into_iter().map(|t| (t, last)).collect();
        want.sort();
        assert_eq!(changed, want, "swapping {x}");
    }
}

#[test]
fn f16_kv_first_attention_token_is_v_in_f16() {
    // One cached position: its softmax weight is exactly 1, so the output is the stored value, f16(v).
    let d = Dims::tiny();
    let rec = run_rounding(Rounding::F16Kv);
    let pre = &rec.0[&TapName::Layer(3, Key::AttnPregate)][0];
    let v = &rec.0[&TapName::Layer(3, Key::AttnV)][0];
    let group = d.n_head / d.n_head_kv;
    for i in 0..d.n_head {
        for j in 0..d.head_dim {
            let want = half::f16::from_f32(v[(i / group) * d.head_dim + j]).to_f32();
            assert_eq!(pre[i * d.head_dim + j], want, "head {i} dim {j}");
        }
    }
}

#[test]
fn f16_kv_differs_from_exact_only_where_the_cache_is_read() {
    // The GPU's tap rule compares against this mode: exact arithmetic, with K and V stored as f16 like the engine's
    // cache. At position 0, every tap before the first attention output is bit-identical to Exact's.
    let d = Dims::tiny();
    let (exact, f16) = (run_rounding(Rounding::Exact), run_rounding(Rounding::F16Kv));
    let first = TapName::Layer(3, Key::AttnPregate);
    for t in expected_taps(&d).into_iter().take_while(|&t| t != first) {
        assert_eq!(f16.0[&t][0], exact.0[&t][0], "{t} at position 0");
    }
    assert_ne!(
        f16.0[&first][0], exact.0[&first][0],
        "{first} reads the f16 cache"
    );
}
