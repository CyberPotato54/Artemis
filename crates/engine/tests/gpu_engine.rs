//! The engine on the GPU, with the tiny model.
use engine::{Engine, Options, Stats};
use gguf::Gguf;
use kernels::dispatch::{WHERE_CPU, WHERE_HOST, WHERE_VRAM};
use qwen36::{dims::Dims, model::Model, testing::tiny_model};

fn tiny() -> (Gguf, Dims) {
    (Gguf::from_bytes(tiny_model(1)).unwrap(), Dims::tiny())
}

#[test]
#[ignore = "needs the GPU"]
fn loads_the_tiny_model_with_every_expert_in_host_memory() {
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    let m = Model::open(&g, &d).unwrap();
    for l in 0..d.n_layer {
        let moe = &m.layers[l].moe;
        for ex in [0, 7, d.n_expert - 1] {
            let (addr, loc) = e.slot_entry(l, ex).unwrap();
            assert_eq!(loc, WHERE_HOST, "layer {l} expert {ex}");
            assert_ne!(addr, 0);
            let mut want = moe.gate_exps.expert(ex, d.n_ff_exp).bytes.to_vec();
            want.extend_from_slice(moe.up_exps.expert(ex, d.n_ff_exp).bytes);
            want.extend_from_slice(moe.down_exps.expert(ex, d.n_embd).bytes);
            assert!(
                e.host_slot(l, ex) == want,
                "layer {l} expert {ex}'s slot holds its gate, up and down"
            );
        }
    }
    for ex in 0..d.n_expert {
        assert_eq!(e.slot_entry(d.n_layer, ex).unwrap(), (0, WHERE_CPU));
    }
    assert_eq!((e.cache_slots(), e.stats().cache_bytes), (0, 0));
    let s = e.session(16).unwrap();
    assert_eq!((s.pos(), s.ctx(), s.tokens().len()), (0, 16, 0));
}
use engine::Error;
use qwen36::{
    ops::rel_l2,
    reference::{Reference, Rounding},
    taps::{expected_taps, Key, NullTap, Tap, TapName},
};
use std::collections::BTreeMap;

const TOKENS: [u32; 8] = [10, 300, 7, 506, 42, 42, 1, 511];

/// Every tap at every position, in order.
#[derive(Default)]
struct Collect(BTreeMap<TapName, Vec<Vec<f32>>>);

impl Tap for Collect {
    fn tap(&mut self, name: TapName, pos: u32, data: &[f32]) {
        let v = self.0.entry(name).or_default();
        assert_eq!(v.len(), pos as usize, "{name} arrived out of order");
        v.push(data.to_vec());
    }
}

fn run(e: &Engine, tokens: &[u32], tap: &mut dyn Tap) -> Vec<Vec<f32>> {
    let mut s = e.session(16).unwrap();
    tokens
        .iter()
        .map(|&t| s.decode_tapped(e, t, tap).unwrap())
        .collect()
}

#[test]
#[ignore = "needs the GPU"]
fn decode_tracks_the_exact_reference_on_the_tiny_model() {
    // Free-running, a sanity check on the wiring; the gate is the forced comparison.
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    let m = Model::open(&g, &d).unwrap();
    let mut r = Reference::with_options(&m, 4, Rounding::Exact);
    let mut s = e.session(16).unwrap();
    let mut worst = 0f64;
    for &t in &TOKENS {
        let want = r.step(t, &mut NullTap).unwrap();
        let got = s.decode(&e, t).unwrap();
        worst = worst.max(rel_l2(&got, &want));
    }
    eprintln!("worst free-running logit error against the exact reference: {worst:.3e}");
    assert!(worst < 1e-3, "{worst:.3e}");
    assert_eq!(s.tokens(), TOKENS);
}

#[test]
#[ignore = "needs the GPU"]
fn decode_is_deterministic_and_taps_change_nothing() {
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    let plain = |e: &Engine| -> Vec<Vec<f32>> {
        let mut s = e.session(16).unwrap();
        TOKENS.iter().map(|&t| s.decode(e, t).unwrap()).collect()
    };
    let a = plain(&e);
    assert!(a == plain(&e), "two sessions give bit-identical logits");
    let mut c = Collect::default();
    let tapped = run(&e, &TOKENS, &mut c);
    assert!(tapped == a, "tapping only adds copies");
    assert!(
        c.0[&TapName::Logits] == a,
        "the logits tap is the returned logits"
    );
    for t in expected_taps(&d) {
        assert_eq!(c.0.get(&t).map(Vec::len), Some(TOKENS.len()), "{t}");
    }
}

/// Forces from `from`, except tap `swap` at the last position, which takes `alt`'s value.
struct ForceSwap<'a> {
    rec: Collect,
    from: &'a Collect,
    alt: &'a Collect,
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

/// The taps the GPU computes directly from tap `x` within one position, in the tiny model (layers 0 to 2 DeltaNet, 3
/// attention): the reference's graph (`crates/qwen36/tests/reference.rs`), which the engine's kernels follow.
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
        // router_topk computes probabilities, ids and weights in one kernel from the logits; the ids change only if
        // the selection does (handled where this is used).
        RouterLogits => at(&[RouterProbs, TopKW]),
        TopKW | MoeSwiglu => at(&[MoeOut]),
        MoeOut | ShexpOut => at(&[FfnOut]),
        ShexpSwiglu => at(&[ShexpOut]),
        FfnOut => at(&[Out]),
        Out if l == 3 => vec![TapName::FinalNorm],
        Out => vec![
            TapName::Layer(l + 1, AttnNorm),
            TapName::Layer(l + 1, Resid),
        ],
        RouterProbs | TopK => unreachable!("not probed"),
    }
}

#[test]
#[ignore = "needs the GPU"]
fn forcing_feeds_every_tap_and_nothing_else() {
    // Forced from run A, swapping one tap's value at the last position for run C's must change exactly that tap's
    // direct consumers there. A tap whose force is missing, or lands after its consumer has read the output, changes
    // nothing downstream or leaks further. Probed for every tap kind in a DeltaNet layer and the attention layer.
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    let mut a = Collect::default();
    run(&e, &TOKENS, &mut a);
    let mut c = Collect::default();
    run(&e, &[3, 4, 5, 6, 7, 8, 9, 300], &mut c);
    let forced = |swap: Option<TapName>| {
        let mut f = ForceSwap {
            rec: Collect::default(),
            from: &a,
            alt: &c,
            swap,
        };
        run(&e, &TOKENS, &mut f);
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
        if let TapName::Layer(l, Key::RouterLogits) = x {
            let ids = (TapName::Layer(l, Key::TopK), last);
            if changed.contains(&ids) {
                want.push(ids);
            }
        }
        want.sort();
        assert_eq!(changed, want, "swapping {x}");
    }
}

#[test]
#[ignore = "needs the GPU"]
fn an_expert_marked_cpu_fails_the_step_and_names_it() {
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    for ex in 0..d.n_expert {
        e.set_slot_entry(2, ex, 0, kernels::dispatch::WHERE_CPU)
            .unwrap();
    }
    let mut s = e.session(4).unwrap();
    match s.decode(&e, 10) {
        Err(Error::ExpertOnCpu {
            pos: 0,
            layer: 2,
            expert,
        }) => {
            assert!((expert as usize) < d.n_expert);
            let msg = s.decode(&e, 10).unwrap_err().to_string();
            assert!(
                msg.contains("layer 2") && msg.contains("CPU miss path"),
                "{msg}"
            );
        }
        other => panic!("expected ExpertOnCpu at layer 2, got {other:?}"),
    }
}

#[test]
#[ignore = "needs the GPU"]
fn a_slot_entry_without_an_address_fails_the_step_and_names_it() {
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    for ex in 0..d.n_expert {
        e.set_slot_entry(1, ex, 0, WHERE_HOST).unwrap();
    }
    // Every expert is refused, so the status names layer 1's lowest routed expert: the same one every time.
    let named: Vec<u32> = (0..4)
        .map(|_| match e.session(4).unwrap().decode(&e, 10) {
            Err(Error::BadSlotEntry {
                pos: 0,
                layer: 1,
                expert,
            }) => expert,
            other => panic!("expected BadSlotEntry at layer 1, got {other:?}"),
        })
        .collect();
    assert!(named.iter().all(|&x| x == named[0]), "{named:?}");
}

#[test]
#[ignore = "needs the GPU"]
fn nan_logits_fail_the_step_and_name_the_position() {
    // Token 511's embedding gets a NaN scale, so the third position's logits are NaN.
    let mut bytes = tiny_model(1);
    let d = Dims::tiny();
    let off = {
        let g = Gguf::from_bytes(bytes.clone()).unwrap();
        let t = g.tensor("token_embd.weight").unwrap();
        assert_eq!(t.ggml_type, qwen36::quant::WType::Q8_0.ggml());
        t.offset as usize + 511 * qwen36::quant::WType::Q8_0.row_bytes(d.n_embd)
    };
    bytes[off..off + 2].copy_from_slice(&0x7E00u16.to_le_bytes());
    let g = Gguf::from_bytes(bytes).unwrap();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    let mut s = e.session(4).unwrap();
    s.decode(&e, 10).unwrap();
    s.decode(&e, 300).unwrap();
    match s.decode(&e, 511) {
        Err(Error::NonFinite { pos: 2, value, .. }) => assert!(!value.is_finite()),
        other => panic!("expected NonFinite at position 2, got {other:?}"),
    }
}

#[test]
#[ignore = "needs the GPU"]
fn context_full_and_bad_tokens_are_errors() {
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    let mut s = e.session(2).unwrap();
    s.decode(&e, 10).unwrap();
    assert!(matches!(
        s.decode(&e, d.n_vocab as u32),
        Err(Error::TokenOutOfRange {
            token: 512,
            n_vocab: 512
        })
    ));
    assert_eq!(s.pos(), 1, "a refused token doesn't move the position");
    s.decode(&e, 11).unwrap();
    assert!(matches!(
        s.decode(&e, 12),
        Err(Error::ContextFull { ctx: 2 })
    ));
}

#[test]
#[ignore = "needs the GPU"]
fn an_unforced_tap_holds_its_kernels_output_not_a_later_in_place_update() {
    // `l2norm` normalizes the conv output's q and k parts in place, right after the `gdn.conv` tap. The tap's copy
    // must finish first: each key group of `gdn.conv` is raw, so no group may equal its normalized `gdn.q`/`gdn.k`.
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    let (kd, dk) = (d.key_dim(), d.ssm_d_state);
    for round in 0..8 {
        let mut c = Collect::default();
        run(&e, &TOKENS, &mut c);
        for (&name, convs) in c.0.iter() {
            let TapName::Layer(l, Key::GdnConv) = name else {
                continue;
            };
            for (p, conv) in convs.iter().enumerate() {
                let q = &c.0[&TapName::Layer(l, Key::GdnQ)][p];
                let k = &c.0[&TapName::Layer(l, Key::GdnK)][p];
                let normed: Vec<f32> = q.iter().chain(k).copied().collect();
                for gi in 0..2 * kd / dk {
                    let span = gi * dk..(gi + 1) * dk;
                    assert_ne!(
                        conv[span.clone()],
                        normed[span],
                        "round {round}: l{l}.gdn.conv group {gi} at position {p} was copied after l2norm ran"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn a_session_needs_at_least_one_position() {
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    assert!(matches!(e.session(0), Err(Error::BadCtx { ctx: 0 })));
}

#[test]
#[ignore = "needs the GPU"]
fn a_context_too_large_for_vram_is_refused_before_loading() {
    let (g, d) = tiny();
    let r = Engine::from_gguf_with_ctx(&g, &d, None, 1 << 34);
    let Err(e @ Error::DoesNotFit { .. }) = r else {
        panic!("expected DoesNotFit");
    };
    assert!(e.to_string().contains("sequence state"), "{e}");
}

/// Forces layer 0's routed expert ids, and nothing else.
struct ForceIds(Vec<f32>);

impl Tap for ForceIds {
    fn tap(&mut self, _: TapName, _: u32, _: &[f32]) {}
    fn forced(&self, name: TapName, _: u32) -> Option<&[f32]> {
        matches!(name, TapName::Layer(0, Key::TopK)).then_some(self.0.as_slice())
    }
}

#[test]
#[ignore = "needs the GPU"]
fn forced_expert_ids_outside_the_router_are_refused_before_the_gpu_runs() {
    // A corrupt --force-from directory must be a clear error, never an index past the slot table.
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    for bad in [d.n_expert as f32, -1.0, f32::NAN, 1.5] {
        let mut ids: Vec<f32> = (0..d.n_expert_used).map(|i| i as f32).collect();
        ids[2] = bad;
        let mut s = e.session(4).unwrap();
        match s.decode_tapped(&e, 10, &mut ForceIds(ids)) {
            Err(Error::BadForcedExpert { tap, value, .. }) => {
                assert_eq!(tap, "l0.topk");
                assert!(value.to_bits() == bad.to_bits(), "{value} for {bad}");
            }
            other => panic!("{bad}: expected BadForcedExpert, got {other:?}"),
        }
    }
    // In range, the same forcing runs.
    let ids: Vec<f32> = (0..d.n_expert_used).map(|i| i as f32).collect();
    let mut s = e.session(4).unwrap();
    s.decode_tapped(&e, 10, &mut ForceIds(ids)).unwrap();
}

#[test]
#[ignore = "needs the GPU"]
fn a_session_runs_only_on_the_engine_that_made_it() {
    // Its state lives in that engine's device; another engine would read addresses that aren't its own.
    let (g, d) = tiny();
    let (a, b) = (
        Engine::from_gguf(&g, &d, None).unwrap(),
        Engine::from_gguf(&g, &d, None).unwrap(),
    );
    let mut s = a.session(4).unwrap();
    assert!(matches!(s.decode(&b, 10), Err(Error::WrongEngine)));
    assert_eq!(s.pos(), 0, "nothing ran");
    s.decode(&a, 10).unwrap();
}

#[test]
#[ignore = "needs the GPU"]
fn the_arenas_take_what_need_and_session_bytes_say() {
    // The fit check counts these numbers, so they must be what the engine really allocates.
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    let m = Model::open(&g, &d).unwrap();
    assert_eq!(e.vram_bytes(), engine::need(&m, false).vram);
    for ctx in [1, 4096, 1 << 20] {
        let s = e.session(ctx).unwrap();
        assert_eq!(s.vram_bytes(), engine::session_bytes(&d, ctx), "ctx {ctx}");
    }
    // With the MTP: the weights' arena takes `need(model, true)`, and a session takes its state and the
    // MTP's rows, ring and small buffers (`session_bytes_mtp`), as the cache's reserve counted them.
    let mtp = Engine::open(
        &g,
        &d,
        &Options {
            ctx: 1024,
            cache_slots: Some(0),
            speculation: true,
            mtp: Some(engine::MtpOptions {
                rows: 512,
                ring: 128,
                window: 256,
                head_rows: 0,
            }),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(mtp.vram_bytes(), engine::need(&m, true).vram);
    let o = mtp.mtp();
    assert!(o.is_some());
    for ctx in [1, 4096, 1 << 20] {
        let s = mtp.session(ctx).unwrap();
        assert_eq!(
            s.vram_bytes(),
            engine::session_bytes_mtp(&d, ctx, o),
            "ctx {ctx} with the MTP"
        );
        assert_eq!(
            s.vram_bytes(),
            engine::session_bytes(&d, ctx) + engine::mtp_session_bytes(&d, 512, 128),
            "ctx {ctx}: the MTP's buffers are the whole difference"
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn a_staged_entry_lands_at_the_next_step() {
    // Changes are staged in the update list and applied at the start of a step, so the table changes only
    // between steps.
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    let want = e.session(4).unwrap().decode(&e, 10).unwrap();
    let host: Vec<(u64, u32)> = (0..d.n_expert)
        .map(|ex| e.slot_entry(2, ex).unwrap())
        .collect();
    for ex in 0..d.n_expert {
        e.stage_slot_entry(2, ex, 0, WHERE_CPU).unwrap();
    }
    assert_eq!(e.slot_entry(2, 0).unwrap(), host[0], "staged, not applied");
    let mut s = e.session(4).unwrap();
    assert!(matches!(
        s.decode(&e, 10),
        Err(Error::ExpertOnCpu {
            pos: 0,
            layer: 2,
            ..
        })
    ));
    assert_eq!(e.slot_entry(2, 0).unwrap(), (0, WHERE_CPU));
    for (ex, &(addr, loc)) in host.iter().enumerate() {
        e.stage_slot_entry(2, ex, addr, loc).unwrap();
    }
    let got = e.session(4).unwrap().decode(&e, 10).unwrap();
    assert!(got == want, "the restored table gives the same logits");
    assert_eq!(e.slot_entry(2, 0).unwrap(), host[0]);
}

#[test]
#[ignore = "needs the GPU"]
fn restaging_an_entry_keeps_the_last_change() {
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    for ex in 0..d.n_expert {
        e.stage_slot_entry(1, ex, 0, WHERE_CPU).unwrap();
        // HOST with no address: refused as a corrupt entry, not as a CPU one.
        e.stage_slot_entry(1, ex, 0, WHERE_HOST).unwrap();
    }
    assert!(matches!(
        e.session(4).unwrap().decode(&e, 10),
        Err(Error::BadSlotEntry {
            pos: 0,
            layer: 1,
            ..
        })
    ));
    assert!(matches!(
        e.stage_slot_entry(d.n_layer + 1, 0, 0, WHERE_HOST),
        Err(Error::NoSuchEntry { .. })
    ));
    assert!(matches!(
        e.stage_slot_entry(0, d.n_expert, 0, WHERE_HOST),
        Err(Error::NoSuchEntry { .. })
    ));
}

/// Forces the embedding with a value of the wrong length: the step fails while recording, before its submit.
struct BadEmbd;

impl Tap for BadEmbd {
    fn tap(&mut self, _: TapName, _: u32, _: &[f32]) {}
    fn forced(&self, name: TapName, _: u32) -> Option<&[f32]> {
        matches!(name, TapName::Embd).then_some(&[1.0][..])
    }
}

#[test]
#[ignore = "needs the GPU"]
fn a_step_that_fails_before_its_submit_keeps_the_staged_changes() {
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    let host = e.slot_entry(2, 0).unwrap();
    for ex in 0..d.n_expert {
        e.stage_slot_entry(2, ex, 0, WHERE_CPU).unwrap();
    }
    let mut s = e.session(4).unwrap();
    assert!(matches!(
        s.decode_tapped(&e, 10, &mut BadEmbd),
        Err(Error::BadForce { .. })
    ));
    assert_eq!(e.slot_entry(2, 0).unwrap(), host, "nothing was submitted");
    assert!(matches!(
        e.session(4).unwrap().decode(&e, 10),
        Err(Error::ExpertOnCpu {
            pos: 0,
            layer: 2,
            ..
        })
    ));
}

/// The engine on the tiny model with `slots` expert slots per layer.
fn cached(g: &Gguf, d: &Dims, slots: usize) -> Engine {
    Engine::open(
        g,
        d,
        &Options {
            cache_slots: Some(slots),
            ..Options::default()
        },
    )
    .unwrap()
}

#[test]
#[ignore = "needs the GPU"]
fn a_cache_holds_the_lowest_ids_and_the_table_points_at_its_slots() {
    // The tiny model's profile is all zeros, so ties fill each layer's 5 slots with experts 0..5, slot s holding
    // expert s. The cache is one chunk of 5 rows: row s holds slot s of every layer in turn, so
    // layer l's slot s is at the chunk's start + s rows + the slots of the layers before l.
    let (g, d) = tiny();
    let e = cached(&g, &d, 5);
    assert_eq!(e.cache_slots(), 5);
    let sizes: Vec<u64> = (0..d.n_layer)
        .map(|l| e.host_slot(l, 0).len() as u64)
        .collect();
    let row: u64 = sizes.iter().sum();
    let (base, _) = e.slot_entry(0, 0).unwrap();
    for l in 0..d.n_layer {
        let before: u64 = sizes[..l].iter().sum();
        for ex in 0..d.n_expert {
            let (addr, loc) = e.slot_entry(l, ex).unwrap();
            if ex < 5 {
                assert_eq!(
                    (addr, loc),
                    (base + ex as u64 * row + before, WHERE_VRAM),
                    "layer {l} expert {ex}"
                );
                assert!(
                    e.cache_slot_bytes(l, ex).unwrap() == e.host_slot(l, ex),
                    "layer {l}: slot {ex} holds expert {ex}'s gate, up and down"
                );
            } else {
                assert_eq!(loc, WHERE_HOST, "layer {l} expert {ex}");
            }
        }
    }
    for ex in 0..d.n_expert {
        assert_eq!(e.slot_entry(d.n_layer, ex).unwrap(), (0, WHERE_CPU));
    }
    assert!(matches!(
        e.cache_slot_bytes(0, 5),
        Err(Error::NoSuchSlot {
            layer: 0,
            slot: 5,
            slots: 5
        })
    ));
    assert_eq!(e.stats().cache_bytes, 5 * row);

    // The same in chunks of 2 rows (2, 2 and 1): a slot is at its own chunk's start + its row in the chunk + the layers
    // before it, and the chunks need not be adjacent.
    let e = Engine::open(
        &g,
        &d,
        &Options {
            cache_slots: Some(5),
            chunk_rows: Some(2),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(e.cache_slots(), 5);
    for l in 0..d.n_layer {
        let before: u64 = sizes[..l].iter().sum();
        for s in 0..5usize {
            let (chunk_base, _) = e.slot_entry(0, 2 * (s / 2)).unwrap();
            let (addr, loc) = e.slot_entry(l, s).unwrap();
            assert_eq!(
                (addr, loc),
                (chunk_base + (s % 2) as u64 * row + before, WHERE_VRAM),
                "layer {l} slot {s}"
            );
            assert!(
                e.cache_slot_bytes(l, s).unwrap() == e.host_slot(l, s),
                "layer {l}: slot {s} holds expert {s}"
            );
        }
    }
    assert_eq!(e.stats().cache_bytes, 5 * row);
}

#[test]
#[ignore = "needs the GPU"]
fn logits_are_bit_identical_at_every_cache_size() {
    // VRAM and HOST experts run the same code at a different address: test 5 in miniature. Every step
    // reads 4 routed experts in each of 4 layers, 128 reads over the 8 tokens.
    let (g, d) = tiny();
    let reads = (TOKENS.len() * d.n_layer * d.n_expert_used) as u64;
    let run = |slots: usize| -> (Vec<Vec<f32>>, Stats) {
        let e = cached(&g, &d, slots);
        let mut s = e.session(16).unwrap();
        let logits = TOKENS.iter().map(|&t| s.decode(&e, t).unwrap()).collect();
        (logits, e.stats())
    };
    let (base, none) = run(0);
    assert_eq!((none.hits, none.misses), (0, reads));
    for slots in [1, 8, 16] {
        let (logits, st) = run(slots);
        assert!(
            logits == base,
            "{slots} slots: logits differ from a cache of zero"
        );
        assert_eq!(st.slots_per_layer, slots);
        assert_eq!(st.hits + st.misses, reads, "{slots} slots");
        if slots == 8 {
            assert!(st.hits > 0, "half the experts cached, yet no hits");
        }
        if slots == 16 {
            assert_eq!(st.misses, 0, "every expert cached, yet a miss");
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn the_cache_holds_what_cache_bytes_says() {
    // The fit check sizes the cache with cache_bytes, so the chunks the cache holds must add up to it.
    let (g, d) = tiny();
    let m = Model::open(&g, &d).unwrap();
    let sizes = engine::slot_sizes(&m);
    for slots in [0, 3, 16] {
        assert_eq!(
            cached(&g, &d, slots).stats().cache_bytes,
            engine::cache_bytes(&sizes, slots),
            "{slots} slots"
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn an_oversized_cache_request_is_clamped() {
    // All 16 of the tiny model's experts fit: asking for more, or for the default, gives 16 per layer.
    let (g, d) = tiny();
    assert_eq!(cached(&g, &d, 10_000).cache_slots(), d.n_expert);
    assert_eq!(
        Engine::open(&g, &d, &Options::default())
            .unwrap()
            .cache_slots(),
        d.n_expert
    );
}

/// The tiny model with `slots` cache slots per layer and, given a seed, a forced swap every step.
fn open_for_swaps(g: &Gguf, d: &Dims, slots: usize, force: Option<u64>) -> Result<Engine, Error> {
    Engine::open(
        g,
        d,
        &engine::Options {
            cache_slots: Some(slots),
            force_swap_seed: force,
            ..Default::default()
        },
    )
}

#[test]
#[ignore = "needs the GPU"]
fn forced_swaps_every_step_leave_the_logits_bit_identical() {
    // Test 5 in miniature: a swap starts every step, and the logits stay bit for bit those of an engine
    // that reads every expert from pinned host memory.
    let (g, d) = tiny();
    let plain = Engine::from_gguf(&g, &d, None).unwrap();
    let swapping = open_for_swaps(&g, &d, 6, Some(7)).unwrap();
    let tokens: Vec<u32> = (0..24u32)
        .map(|i| (i * 37 + 5) % d.n_vocab as u32)
        .collect();
    let (mut a, mut b) = (plain.session(32).unwrap(), swapping.session(32).unwrap());
    for (p, &t) in tokens.iter().enumerate() {
        let want = a.decode(&plain, t).unwrap();
        let got = b.decode(&swapping, t).unwrap();
        assert!(got == want, "position {p}: the logits differ");
    }
    let st = swapping.stats();
    assert_eq!(st.swaps_started, 24, "one swap starts every step");
    // A swap's copy goes out the step after it starts, and its VRAM entry the step after that: the last two swaps are
    // still in flight.
    assert_eq!(st.swaps_done, 22, "{st:?}");
    assert!(st.hits > 0 && st.misses > 0, "{st:?}");
}

#[test]
#[ignore = "needs the GPU"]
fn a_requested_swap_lands_in_vram() {
    // A cache of 4: the tiny model's profile is all zeros, so experts 0 to 3 of every layer fill slots 0 to 3. Swapping
    // 9 in for 2 in layer 1: the request changes nothing yet; the first step reads 2 from host memory; the
    // second copies 9 into slot 2; the third reads 9 from it. The logits never change.
    const VRAM: u32 = kernels::dispatch::WHERE_VRAM;
    let (g, d) = tiny();
    let plain = Engine::from_gguf(&g, &d, None).unwrap();
    let e = open_for_swaps(&g, &d, 4, None).unwrap();
    let (slot2, loc) = e.slot_entry(1, 2).unwrap();
    assert_eq!(loc, VRAM);
    let (host9, loc) = e.slot_entry(1, 9).unwrap();
    assert_eq!(loc, WHERE_HOST);
    let host2 = e.host_slot_addr(1, 2);
    assert_eq!(host9, e.host_slot_addr(1, 9));
    e.request_swap(1, 2, 9).unwrap();
    assert_eq!(
        e.slot_entry(1, 2).unwrap(),
        (slot2, VRAM),
        "nothing lands before a step"
    );
    // After each step: expert 2's entry, expert 9's entry, swaps done.
    let after = [
        ((host2, WHERE_HOST), (host9, WHERE_HOST), 0),
        ((host2, WHERE_HOST), (host9, WHERE_HOST), 0),
        ((host2, WHERE_HOST), (slot2, VRAM), 1),
    ];
    let (mut a, mut b) = (plain.session(8).unwrap(), e.session(8).unwrap());
    for (step, (t, (two, nine, done))) in [10u32, 300, 7].into_iter().zip(after).enumerate() {
        assert!(
            b.decode(&e, t).unwrap() == a.decode(&plain, t).unwrap(),
            "step {step}: the logits differ"
        );
        assert_eq!(e.slot_entry(1, 2).unwrap(), two, "step {step}: expert 2");
        assert_eq!(e.slot_entry(1, 9).unwrap(), nine, "step {step}: expert 9");
        assert_eq!(e.stats().swaps_done, done, "step {step}");
    }
    assert!(
        e.cache_slot_bytes(1, 2).unwrap() == e.host_slot(1, 9),
        "slot 2 holds expert 9's gate, up and down"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn the_gpu_table_matches_the_planner_after_forced_swaps() {
    // After 60 steps with a swap forced every step, every entry read back from the GPU's table is the planner's, in
    // address and location; and some routed expert was read from a swapped-in VRAM slot (not one of the initial fill).
    let (g, d) = tiny();
    let slots = 6;
    let e = open_for_swaps(&g, &d, slots, Some(11)).unwrap();
    let mut s = e.session(64).unwrap();
    let mut swapped_in_reads = 0;
    for p in 0..60u32 {
        s.decode(&e, (p * 37 + 5) % d.n_vocab as u32).unwrap();
        for (i, id) in e.last_routes().into_iter().enumerate() {
            let layer = i / d.n_expert_used;
            // The initial fill is experts 0..slots (the tiny model's profile is all zeros).
            if id >= slots && matches!(e.planner_entry(layer, id), Some(engine::Entry::Vram(_))) {
                swapped_in_reads += 1;
            }
        }
    }
    for layer in 0..d.n_layer {
        for ex in 0..d.n_expert {
            let want = match e.planner_entry(layer, ex).unwrap() {
                engine::Entry::Host => (e.host_slot_addr(layer, ex), WHERE_HOST),
                engine::Entry::Vram(slot) => (e.cache_slot_addr(layer, slot).unwrap(), WHERE_VRAM),
            };
            assert_eq!(
                e.slot_entry(layer, ex).unwrap(),
                want,
                "layer {layer} expert {ex}"
            );
        }
    }
    assert!(
        swapped_in_reads > 0,
        "no routed expert was read from a swapped-in slot"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_reset_session_gives_the_logits_of_a_fresh_one() {
    // Two sequences through one session, reset between, against a fresh session each.
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    let seqs: [Vec<u32>; 2] = [
        (0..12u32)
            .map(|i| (i * 37 + 5) % d.n_vocab as u32)
            .collect(),
        (0..9u32)
            .map(|i| (i * 53 + 11) % d.n_vocab as u32)
            .collect(),
    ];
    let mut reused = e.session(16).unwrap();
    for (n, seq) in seqs.iter().enumerate() {
        if n > 0 {
            reused.reset(&e).unwrap();
            assert_eq!((reused.pos(), reused.tokens().len()), (0, 0));
        }
        let mut fresh = e.session(16).unwrap();
        for (p, &t) in seq.iter().enumerate() {
            let want = fresh.decode(&e, t).unwrap();
            let got = reused.decode(&e, t).unwrap();
            assert!(got == want, "sequence {n} position {p}: the logits differ");
        }
    }
    let other = Engine::from_gguf(&g, &d, None).unwrap();
    assert!(matches!(reused.reset(&other), Err(Error::WrongEngine)));
}

#[test]
#[ignore = "needs the GPU"]
fn illegal_swaps_are_refused() {
    // A refused request changes nothing, and an expert in a swap can't join another: the table must never point at a
    // slot being written. A cache of 4: experts 0 to 3 sit in every layer's slots.
    use engine::SwapError;
    let (g, d) = tiny();
    let e = open_for_swaps(&g, &d, 4, None).unwrap();
    let refused = |layer, out, inn| match e.request_swap(layer, out, inn) {
        Err(Error::Swap(s)) => s,
        other => panic!("layer {layer}, {out} out, {inn} in: expected a SwapError, got {other:?}"),
    };
    assert_eq!(
        refused(0, 12, 13),
        SwapError::NotResident {
            layer: 0,
            expert: 12
        }
    );
    assert_eq!(
        refused(0, 0, 1),
        SwapError::AlreadyResident {
            layer: 0,
            expert: 1
        }
    );
    assert!(
        matches!(refused(4, 0, 9), SwapError::OutOfRange { .. }),
        "blk.4 is the MTP block, which has no cache"
    );
    assert!(matches!(refused(0, 0, 16), SwapError::OutOfRange { .. }));
    e.request_swap(1, 2, 9).unwrap();
    // While that swap is in flight, neither of its experts can be in another.
    for (out, inn, named) in [(2, 10, 2), (3, 9, 9)] {
        let m = refused(1, out, inn).to_string();
        assert!(
            m.contains(&format!("expert {named}")) && !m.contains('\n'),
            "{m}"
        );
    }
    assert_eq!(e.stats().swaps_started, 1, "refused requests start nothing");
    let mut s = e.session(8).unwrap();
    for t in [10, 300, 7] {
        s.decode(&e, t).unwrap();
    }
    assert_eq!(
        e.slot_entry(1, 9).unwrap().1,
        kernels::dispatch::WHERE_VRAM,
        "the legal swap still lands"
    );
    assert_eq!(e.stats().swaps_done, 1);
}

#[test]
#[ignore = "needs the GPU"]
fn forced_swaps_need_room_to_swap() {
    // A forced swap sends an expert out of the cache and brings another in. With no cache, or every expert cached,
    // there is none to make, so the load is refused rather than running without swaps.
    let (g, d) = tiny();
    for slots in [0, d.n_expert] {
        let Err(e @ Error::NoRoomToSwap { .. }) = open_for_swaps(&g, &d, slots, Some(1)) else {
            panic!("{slots} slots: expected NoRoomToSwap");
        };
        assert!(
            matches!(e, Error::NoRoomToSwap { slots: s, n_expert: 16 } if s == slots),
            "{e:?}"
        );
        assert!(e.to_string().contains("--force-swap-every-token"), "{e}");
        // Without the seed, the same cache loads.
        open_for_swaps(&g, &d, slots, None).unwrap();
    }
}

/// The base time: with `Options::base_time` decode reads, for every HOST expert, the resident
/// expert `base_table` names: the (e mod r)-th resident of its layer, residents in expert order. So it gives the logits
/// of an engine whose table points there itself. Two slots a layer under the static policy, so the table holds still; an
/// engine reading HOST shows the remap changed what was read.
#[test]
#[ignore = "needs the GPU"]
fn the_base_time_reads_a_resident_expert_for_every_host_one() {
    let (g, d) = tiny();
    let open = |base_time| {
        Engine::open(
            &g,
            &d,
            &Options {
                ctx: 64,
                cache_slots: Some(2),
                cache_policy: engine::CachePolicyKind::Static,
                base_time,
                ..Options::default()
            },
        )
        .unwrap()
    };
    let (based, remapped, plain) = (open(true), open(false), open(false));
    assert!(based.base_time() && !remapped.base_time());
    for l in 0..d.n_layer {
        let resident: Vec<(u64, u32)> = (0..d.n_expert)
            .map(|e| remapped.mirror_entry(l, e))
            .filter(|x| x.1 == WHERE_VRAM)
            .collect();
        assert_eq!(resident.len(), 2, "layer {l}: two resident experts");
        for e in 0..d.n_expert {
            if remapped.mirror_entry(l, e).1 == WHERE_HOST {
                let (addr, loc) = resident[e % 2];
                remapped.set_slot_entry(l, e, addr, loc).unwrap();
            }
        }
    }
    let run = |e: &Engine| {
        let mut s = e.session(64).unwrap();
        TOKENS
            .iter()
            .map(|&t| s.decode(e, t).unwrap())
            .collect::<Vec<_>>()
    };
    let (a, b, c) = (run(&based), run(&remapped), run(&plain));
    assert!(a == b, "the base time reads the experts base_table names");
    assert!(a != c, "the remap changed what decode read");
}

// ---- The CPU miss path on decode ----

/// The tiny model at `share`, with `slots` experts of every layer resident (the profile's ties: experts 0..slots) and
/// the static policy, so the mix of VRAM and HOST entries stays put for the whole run.
fn open_share(g: &Gguf, d: &Dims, slots: usize, share: f32) -> Engine {
    Engine::open(
        g,
        d,
        &Options {
            ctx: 64,
            cache_slots: Some(slots),
            cache_policy: engine::CachePolicyKind::Static,
            cpu_share: share,
            ..Options::default()
        },
    )
    .unwrap()
}

/// Every logit of `got` equals `want`'s, bit for bit.
fn same_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: logit count");
    if let Some(i) = (0..want.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
        panic!("{what}: logit {i} is {} against {}", got[i], want[i]);
    }
}

#[test]
#[ignore = "needs the GPU"]
fn cpu_share_matches_the_host_path_bit_for_bit() {
    // Two engines, the same weights and tokens (32 greedy), one at share 0 and one at share 1. Three experts of every
    // layer are resident and the rest HOST, which share 1 turns WHERE_CPU, so a routed set mixes VRAM and CPU
    // entries. Every logit of every step must match with to_bits. No wait may time out: the CPU's own y is what the
    // share-1 steps must have used (a timeout would fall back to the HOST path and test only the fallback). The
    // denormal-x case is the kernels test serve_two_matches_the_worker_on_a_denormal_x.
    let (g, d) = tiny();
    let (host, cpu) = (open_share(&g, &d, 3, 0.0), open_share(&g, &d, 3, 1.0));
    assert_eq!((host.cpu_share(), cpu.cpu_share()), (0.0, 1.0));
    for l in 0..d.n_layer {
        let resident = (0..d.n_expert)
            .filter(|&e| cpu.mirror_entry(l, e).1 == WHERE_VRAM)
            .count();
        assert_eq!(resident, 3, "layer {l}: three resident experts");
    }
    let (mut a, mut b) = (host.session(64).unwrap(), cpu.session(64).unwrap());
    let mut token = 10u32;
    for step in 0..32 {
        let want = a.decode(&host, token).unwrap();
        let got = b.decode(&cpu, token).unwrap();
        same_bits(&got, &want, &format!("step {step}"));
        token = engine::greedy(&want);
    }
    let st = cpu.stats();
    eprintln!(
        "share 1: {} VRAM reads, {} HOST misses, {} CPU experts, {} timeouts, longest submit {} us",
        st.hits, st.misses, st.cpu_misses, st.cpu_timeouts, st.longest_submit_us
    );
    assert!(
        st.hits > 0 && st.misses > 0,
        "the routed sets mixed VRAM and CPU entries"
    );
    assert!(st.cpu_misses > 0, "the CPU computed no expert");
    assert_eq!(
        st.cpu_timeouts, 0,
        "a wait timed out: the step fell back instead of using the CPU's y"
    );
    assert_eq!(host.stats().cpu_misses, 0, "share 0 starts no CPU path");
}

#[test]
#[ignore = "needs the GPU"]
fn a_dead_pool_falls_back_to_the_host_path() {
    // Share 1, every expert HOST (so CPU), and the coordinator told to answer nothing. One decode step: its logits
    // equal a share-0 engine's bit for bit, its submit costs one 20 ms bound (over 10 ms, under 500), and it counts a
    // timeout. The next step, the hook cleared, also matches and counts no new timeout: that pins the per-step status
    // zeroing (an uncleared count would make the next token skip every wait, and read back as a new timeout).
    let (g, d) = tiny();
    let (host, cpu) = (open_share(&g, &d, 0, 0.0), open_share(&g, &d, 0, 1.0));
    let (mut a, mut b) = (host.session(64).unwrap(), cpu.session(64).unwrap());
    cpu.cpu_fail_next(true);
    let want = a.decode(&host, 10).unwrap();
    let got = b.decode(&cpu, 10).unwrap();
    same_bits(&got, &want, "the dead pool's step");
    let st = cpu.stats();
    assert!(
        st.cpu_timeouts >= 1,
        "the dead pool's wait counted no timeout"
    );
    let ms = st.longest_submit_us / 1000;
    assert!(
        (10..500).contains(&ms),
        "one 20 ms bound, not none and not one per layer: {} us",
        st.longest_submit_us
    );
    cpu.cpu_fail_next(false);
    let token = engine::greedy(&want);
    let want = a.decode(&host, token).unwrap();
    let got = b.decode(&cpu, token).unwrap();
    same_bits(&got, &want, "the step after the dead pool");
    let next = cpu.stats();
    assert_eq!(
        next.cpu_timeouts, st.cpu_timeouts,
        "the next token timed out too"
    );
    assert!(
        next.cpu_misses > st.cpu_misses,
        "the CPU served nothing after the hook cleared"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_late_answer_is_wasted_not_trusted() {
    // Share 1, every expert HOST (so CPU). The coordinator answers layer 0 late: only once the GPU's wait has counted
    // its timeout does it store y and the done flag (a stall, not a death). It answers every later layer as usual, so
    // the CPU catches up while the token's later waits skip. What this pins: a token whose wait timed out has a
    // share-0 engine's logits bit for bit even though the CPU answers it late, and every wait after layer 0's skipped
    // (the step's skip count is n_layer - 1). Whether a late done store lands before serve 2 reads the flag is timing,
    // so this test does not by itself pin that serve 2 ignores a late flag; the deterministic pin is kernels'
    // serve_two_writes_nothing_after_a_good_wait_and_ignores_a_late_flag. The CPU must have answered every layer of
    // the token (else this was a death, the dead-pool test), the hook must have seen layer 0's timeout before it
    // answered (else no answer was late), and the next token is served as usual.
    let (g, d) = tiny();
    let (host, cpu) = (open_share(&g, &d, 0, 0.0), open_share(&g, &d, 0, 1.0));
    let (mut a, mut b) = (host.session(64).unwrap(), cpu.session(64).unwrap());
    cpu.cpu_stall_layer0_once();
    let want = a.decode(&host, 10).unwrap();
    let got = b.decode(&cpu, 10).unwrap();
    same_bits(&got, &want, "the late answer's step");
    let st = cpu.stats();
    assert_eq!(
        st.cpu_timeouts, 1,
        "layer 0's wait timed out once, the later ones skipped"
    );
    assert_eq!(
        cpu.cpu_last_skips(),
        Some(d.n_layer as u32 - 1),
        "every wait after layer 0's timeout skipped"
    );
    let seqs: Vec<u32> = (0..d.n_layer)
        .map(|l| engine::cpu_miss::seq_of(0, l))
        .collect();
    let t0 = std::time::Instant::now();
    while cpu.cpu_done_flags().unwrap() != seqs {
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(2),
            "the CPU never answered every layer: {:?} against {seqs:?}",
            cpu.cpu_done_flags()
        );
        std::hint::spin_loop();
    }
    // The catch-up itself: the stall hook saw layer 0's timeout inside its window and then answered. A hook that
    // missed the window (and answered on time, or never) fails here rather than passing without a late answer.
    assert_eq!(
        cpu.cpu_late_answered(),
        Some(true),
        "the stall hook never saw the GPU's timeout and answered late"
    );
    let token = engine::greedy(&want);
    let want = a.decode(&host, token).unwrap();
    let got = b.decode(&cpu, token).unwrap();
    same_bits(&got, &want, "the step after the late answer");
    assert_eq!(cpu.stats().cpu_timeouts, 1, "the next token timed out too");
}

/// The tiny model opened with the MTP's weights, with speculation on (the MTP needs it).
fn open_with_mtp(g: &Gguf, d: &Dims, mtp: bool) -> Engine {
    let opts = Options {
        ctx: 1024,
        cache_slots: Some(2),
        speculation: mtp,
        mtp: mtp.then(|| {
            engine::MtpOptions::for_session(&engine::SpecConfig::default(), 1024)
                .unwrap()
                .unwrap()
        }),
        ..Options::default()
    };
    Engine::open(g, d, &opts).unwrap()
}

#[test]
#[ignore = "needs the GPU"]
fn blk_n_layer_loads_host_only_when_the_mtp_is_asked_for() {
    let (g, d) = tiny();
    let m = Model::open(&g, &d).unwrap();
    let (off, on) = (open_with_mtp(&g, &d, false), open_with_mtp(&g, &d, true));
    // Host memory grows by exactly blk.4's experts; VRAM by its weights, in the arena and in the session's rows.
    let blk = &m.mtp.layer.moe;
    let experts =
        (blk.gate_exps.bytes.len() + blk.up_exps.bytes.len() + blk.down_exps.bytes.len()) as u64;
    assert_eq!(on.host_bytes() - off.host_bytes(), experts);
    assert_eq!(off.mtp_vram_bytes(), 0);
    assert!(on.mtp_vram_bytes() > 0);
    assert_eq!(on.vram_bytes(), engine::need(&m, true).vram);
    assert_eq!(off.vram_bytes(), engine::need(&m, false).vram);
    // Every one of the block's entries is HOST at its host copy, never a cache slot.
    for ex in 0..d.n_expert {
        let (addr, loc) = on.slot_entry(d.n_layer, ex).unwrap();
        assert_eq!(loc, WHERE_HOST, "expert {ex}");
        assert_ne!(addr, 0);
        let mut want = blk.gate_exps.expert(ex, d.n_ff_exp).bytes.to_vec();
        want.extend_from_slice(blk.up_exps.expert(ex, d.n_ff_exp).bytes);
        want.extend_from_slice(blk.down_exps.expert(ex, d.n_embd).bytes);
        assert!(on.host_slot(d.n_layer, ex) == want, "expert {ex}");
        assert_eq!(off.slot_entry(d.n_layer, ex).unwrap().1, WHERE_CPU);
    }
    // The main layers decode as before: the MTP changes nothing for them.
    let a = off.session(8).unwrap().decode(&off, 10).unwrap();
    let b = on.session(8).unwrap().decode(&on, 10).unwrap();
    assert!(a == b, "decode with the MTP loaded is the plain decode");
}
