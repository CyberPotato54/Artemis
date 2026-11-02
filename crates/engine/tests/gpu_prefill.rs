//! Prefill on the GPU, with the tiny model: against decode, whatever the chunk size and the
//! cache, and its errors.
use engine::{Engine, Error, Options, SessionState};
use gguf::Gguf;
use kernels::dispatch::{WHERE_CPU, WHERE_HOST};
use qwen36::{dims::Dims, ops::rel_l2, testing::tiny_model};

fn tiny() -> (Gguf, Dims) {
    (Gguf::from_bytes(tiny_model(1)).unwrap(), Dims::tiny())
}

/// 40 tokens of the tiny model's vocabulary.
fn tokens(d: &Dims) -> Vec<u32> {
    (0..40u32)
        .map(|i| (i * 211 + 17) % d.n_vocab as u32)
        .collect()
}

fn open(g: &Gguf, d: &Dims, slots: usize, chunk: usize, force: Option<u64>) -> Engine {
    Engine::open(
        g,
        d,
        &Options {
            cache_slots: Some(slots),
            prefill_chunk: Some(chunk),
            force_swap_seed: force,
            ..Options::default()
        },
    )
    .unwrap()
}

/// Every position's logits and the final state, through prefill.
fn prefilled(e: &Engine, toks: &[u32]) -> (Vec<Vec<f32>>, SessionState) {
    let mut s = e.session(64).unwrap();
    let mut all = Vec::new();
    s.prefill_each(e, toks, 0, |i, l| {
        assert_eq!(i, all.len(), "positions arrive in order");
        all.push(l.to_vec());
        Ok::<(), Error>(())
    })
    .unwrap();
    assert_eq!(all.len(), toks.len(), "a logits vector for every position");
    assert_eq!((s.pos(), s.tokens()), (toks.len(), toks));
    (all, s.state(e).unwrap())
}

/// Two values' f32s are the same bits (a +0 and a -0 differ, a NaN equals itself).
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn same_logits(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_bits(x, y))
}

fn same_state(a: &SessionState, b: &SessionState) -> bool {
    a.layers.len() == b.layers.len()
        && a.layers
            .iter()
            .zip(&b.layers)
            .all(|(x, y)| same_bits(&x.0, &y.0) && same_bits(&x.1, &y.1))
}

/// The same, one token at a time through decode.
fn decoded(e: &Engine, toks: &[u32]) -> (Vec<Vec<f32>>, SessionState) {
    let mut s = e.session(64).unwrap();
    let all = toks.iter().map(|&t| s.decode(e, t).unwrap()).collect();
    (all, s.state(e).unwrap())
}

/// The worst relative L2 between two states, layer by layer and buffer by buffer.
fn state_err(a: &SessionState, b: &SessionState) -> f64 {
    a.layers
        .iter()
        .zip(&b.layers)
        .flat_map(|(x, y)| [rel_l2(&x.0, &y.0), rel_l2(&x.1, &y.1)])
        // A NaN must win: `f64::max` would drop it and let a broken state pass.
        .fold(0f64, |m, e| if e.is_nan() || e > m { e } else { m })
}

#[test]
#[ignore = "needs the GPU"]
fn prefill_matches_decode_on_the_tiny_model() {
    // The prefill-against-decode check in miniature: chunks of 16 (three of them, the last part full), every position's
    // logits and the final DeltaNet, conv and KV state within 1e-3 of decode's.
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 16, None);
    let toks = tokens(&d);
    let (pl, ps) = prefilled(&e, &toks);
    let (dl, ds) = decoded(&e, &toks);
    let worst = pl
        .iter()
        .zip(&dl)
        .map(|(a, b)| rel_l2(a, b))
        .fold(0f64, f64::max);
    let st = state_err(&ps, &ds);
    eprintln!("prefill against decode: logits {worst:.3e}, state {st:.3e}");
    assert!(worst <= 1e-3, "logits {worst:.3e}");
    assert!(st <= 1e-3, "state {st:.3e}");
    assert_eq!(ps.layers.len(), d.n_layer);
    for (l, (a, b)) in ps.layers.iter().enumerate() {
        if d.is_attention(l) {
            assert_eq!(
                a.len(),
                toks.len() * d.n_head_kv * d.head_dim,
                "layer {l}'s keys"
            );
            assert_eq!(b.len(), a.len(), "layer {l}'s values");
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn prefill_returns_the_last_positions_logits_and_decode_goes_on_from_it() {
    // `prefill` gives what `prefill_each` gives for the last position; decode then takes the next positions.
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 16, None);
    let toks = tokens(&d);
    let (all, _) = prefilled(&e, &toks[..30]);
    let mut s = e.session(64).unwrap();
    let last = s.prefill(&e, &toks[..30]).unwrap();
    assert!(same_bits(&last, &all[29]), "the last position's logits");
    let mut d2 = e.session(64).unwrap();
    for &t in &toks[..30] {
        d2.decode(&e, t).unwrap();
    }
    for &t in &toks[30..] {
        let (a, b) = (s.decode(&e, t).unwrap(), d2.decode(&e, t).unwrap());
        assert!(rel_l2(&a, &b) <= 1e-3, "decode after prefill");
    }
    assert_eq!(s.tokens(), toks);
}

#[test]
#[ignore = "needs the GPU"]
fn prefills_results_dont_depend_on_the_chunk_size() {
    // Every kernel's arithmetic for a position depends only on that position and those before it, so chunks of 1, 7
    // and 64 give the same bits.
    let (g, d) = tiny();
    let toks = tokens(&d);
    let base = prefilled(&open(&g, &d, 0, 64, None), &toks);
    for chunk in [1, 7] {
        let other = prefilled(&open(&g, &d, 0, chunk, None), &toks);
        assert!(
            same_logits(&other.0, &base.0),
            "chunks of {chunk}: the logits"
        );
        assert!(
            same_state(&other.1, &base.1),
            "chunks of {chunk}: the state"
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn prefill_is_bit_identical_at_every_cache_size_and_with_swaps() {
    // Test 5 in miniature, through prefill: chunks of 5 read every expert through the slot table, from host memory or
    // VRAM, and a swap starts every step.
    let (g, d) = tiny();
    let toks = tokens(&d);
    let reads = (toks.len() * d.n_layer * d.n_expert_used) as u64;
    let none = open(&g, &d, 0, 5, None);
    let base = prefilled(&none, &toks);
    let st = none.stats();
    assert_eq!((st.hits, st.misses, st.steps), (0, reads, 8));
    for (slots, force) in [(6, None), (16, None), (6, Some(7))] {
        let e = open(&g, &d, slots, 5, force);
        let run = prefilled(&e, &toks);
        assert!(
            same_logits(&run.0, &base.0),
            "{slots} slots, forced {force:?}: the logits"
        );
        assert!(
            same_state(&run.1, &base.1),
            "{slots} slots, forced {force:?}: the state"
        );
        let st = e.stats();
        assert_eq!(st.hits + st.misses, reads, "{slots} slots");
        assert!(st.hits > 0, "{slots} slots: no hits");
        assert_eq!(st.steps, 8, "eight chunks of five");
        if force.is_some() {
            assert_eq!(st.swaps_started, st.steps, "a swap starts every step");
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn a_prefill_that_cant_run_is_refused_and_changes_nothing() {
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 16, None);
    let mut s = e.session(8).unwrap();
    assert!(matches!(s.prefill(&e, &[]), Err(Error::EmptyPrefill)));
    assert!(matches!(
        s.prefill(&e, &[1, 2, 600]),
        Err(Error::TokenOutOfRange { token: 600, .. })
    ));
    s.prefill(&e, &[1, 2, 3]).unwrap();
    match s.prefill(&e, &[4, 5, 6, 7, 8, 9]) {
        Err(Error::PrefillTooLong {
            pos: 3,
            tokens: 6,
            ctx: 8,
        }) => {}
        other => panic!("expected PrefillTooLong, got {:?}", other.err()),
    }
    assert_eq!((s.pos(), s.tokens()), (3, &[1, 2, 3][..]));
    let other = open(&g, &d, 0, 16, None);
    assert!(matches!(s.prefill(&other, &[4]), Err(Error::WrongEngine)));
    // The context fills exactly.
    s.prefill(&e, &[4, 5, 6, 7, 8]).unwrap();
    assert_eq!(s.pos(), 8);
}

#[test]
#[ignore = "needs the GPU"]
fn an_expert_marked_cpu_fails_the_prefill_step_and_names_it() {
    // Every expert of layer 2 goes down the refused CPU path: the chunk's first position is named.
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 8, None);
    let mut s = e.session(32).unwrap();
    s.prefill(&e, &tokens(&d)[..8]).unwrap();
    assert_eq!(
        e.last_prefill_routes().len(),
        d.n_layer * 8 * d.n_expert_used
    );
    for x in 0..d.n_expert {
        e.set_slot_entry(2, x, 0, WHERE_CPU).unwrap();
    }
    match s.prefill(&e, &tokens(&d)[8..16]) {
        Err(Error::PrefillExpertOnCpu {
            pos: 8,
            tokens: 8,
            layer: 2,
            ..
        }) => {}
        other => panic!(
            "expected PrefillExpertOnCpu at position 8, got {:?}",
            other.err()
        ),
    }
    // The failed step overwrote layers 0 to 2 of the routing log: it reads as empty, not as two chunks mixed.
    assert!(e.last_prefill_routes().is_empty());
    assert!(e.last_prefill_route_weights().is_empty());
}

#[test]
#[ignore = "needs the GPU"]
fn a_failed_prefill_step_stages_its_slot_changes_again() {
    // A step that fails partway, after its first submit applied the step's table changes, stages them again: the
    // next step applies them a second time, which changes nothing, and a change that hadn't landed lands then.
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 8, None);
    let mut s = e.session(32).unwrap();
    let host = |x| e.host_slot_addr(1, x);
    // Stage expert 3 of layer 1 to point at expert 4's copy, then make the step fail in layer 2.
    e.stage_slot_entry(1, 3, host(4), WHERE_HOST).unwrap();
    let saved: Vec<(u64, u32)> = (0..d.n_expert)
        .map(|x| e.slot_entry(2, x).unwrap())
        .collect();
    for x in 0..d.n_expert {
        e.set_slot_entry(2, x, 0, WHERE_CPU).unwrap();
    }
    assert!(s.prefill(&e, &tokens(&d)[..8]).is_err());
    assert_eq!(
        e.slot_entry(1, 3).unwrap(),
        (host(4), WHERE_HOST),
        "applied by layer 0's submit"
    );
    // Undo both, behind the planner's back: only a restaged change can put layer 1's back.
    e.set_slot_entry(1, 3, host(3), WHERE_HOST).unwrap();
    for (x, &(addr, loc)) in saved.iter().enumerate() {
        e.set_slot_entry(2, x, addr, loc).unwrap();
    }
    let mut s = e.session(32).unwrap();
    s.prefill(&e, &tokens(&d)[..8]).unwrap();
    assert_eq!(
        e.slot_entry(1, 3).unwrap(),
        (host(4), WHERE_HOST),
        "the next step applied it again"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn prefills_buffers_take_what_prefill_bytes_says() {
    let (g, d) = tiny();
    for chunk in [1, 16, 64] {
        let e = open(&g, &d, 0, chunk, None);
        assert_eq!(e.prefill_chunk(), chunk);
        assert_eq!(e.prefill_bytes(), engine::prefill_bytes(&d, chunk, None));
    }
    // A context shorter than the chunk cuts it.
    let e = Engine::open(
        &g,
        &d,
        &Options {
            ctx: 10,
            cache_slots: Some(0),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(e.prefill_chunk(), 10);
    // No chunk at all, or one past PREFILL_CHUNK, is refused before the model loads.
    for bad in [0, engine::PREFILL_CHUNK + 1] {
        match Engine::open(
            &g,
            &d,
            &Options {
                prefill_chunk: Some(bad),
                ..Options::default()
            },
        ) {
            Err(Error::BadPrefillChunk { chunk, max }) => {
                assert_eq!((chunk, max), (bad, engine::PREFILL_CHUNK))
            }
            other => panic!(
                "a chunk of {bad}: expected BadPrefillChunk, got {:?}",
                other.err()
            ),
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn more_than_64_wanted_rows_in_one_chunk_from_a_later_first() {
    // 120 tokens in one chunk of 128, logits wanted from token 37 on: 83 rows, so the output head runs twice in the
    // chunk (64 rows, then 19). Against chunks of 1, every reported position's logits are the same bits, and the
    // positions arrive in order, 37 to 119.
    let (g, d) = tiny();
    let toks: Vec<u32> = (0..120u32)
        .map(|i| (i * 211 + 17) % d.n_vocab as u32)
        .collect();
    let run = |chunk: usize| {
        let e = open(&g, &d, 0, chunk, None);
        let mut s = e.session(128).unwrap();
        let mut got: Vec<(usize, Vec<u32>)> = Vec::new();
        s.prefill_each(&e, &toks, 37, |i, l| {
            got.push((i, l.iter().map(|x| x.to_bits()).collect()));
            Ok::<(), Error>(())
        })
        .unwrap();
        assert_eq!(s.pos(), toks.len());
        let st = e.stats();
        assert_eq!(
            st.steps,
            engine::chunk_lengths(0, toks.len(), chunk, e.attn_pairs()).len() as u64
        );
        assert!(
            st.longest_submit_us > 0,
            "chunks of {chunk}: no submit was timed"
        );
        got
    };
    let (whole, ones) = (run(128), run(1));
    let at: Vec<usize> = whole.iter().map(|r| r.0).collect();
    assert_eq!(at, (37..120).collect::<Vec<_>>());
    assert_eq!(ones.len(), whole.len());
    for (a, b) in whole.iter().zip(&ones) {
        assert_eq!(a.0, b.0);
        assert!(
            a.1 == b.1,
            "position {}: chunks of 128 and of 1 differ",
            a.0
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn the_prefill_routing_log_holds_each_layers_experts_per_token() {
    // `[n_layer][tokens][n_expert_used]`: per layer and token, distinct experts in range whose weights sum to 1, and
    // the hit and miss counts cover them all.
    let (g, d) = tiny();
    let e = open(&g, &d, 6, 16, None);
    let mut s = e.session(32).unwrap();
    s.prefill(&e, &tokens(&d)[..9]).unwrap();
    let (ids, w) = (e.last_prefill_routes(), e.last_prefill_route_weights());
    let k = d.n_expert_used;
    assert_eq!((ids.len(), w.len()), (d.n_layer * 9 * k, d.n_layer * 9 * k));
    for (i, (e8, w8)) in ids.chunks(k).zip(w.chunks(k)).enumerate() {
        let mut sorted = e8.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), k, "layer {}, token {}: {e8:?}", i / 9, i % 9);
        assert!(e8.iter().all(|&x| x < d.n_expert), "{e8:?}");
        let sum: f32 = w8.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "layer {}, token {}: {sum}",
            i / 9,
            i % 9
        );
    }
    let st = e.stats();
    assert_eq!(st.hits + st.misses, ids.len() as u64);
}

#[test]
#[ignore = "needs the GPU"]
fn split_f16_prefill_stays_near_f32_prefill_and_decode() {
    // Prefill on the split-f16 cooperative-matrix kernels, in miniature, is within the per-position
    // bound of decode (1e-3, as prefill_matches_decode_on_the_tiny_model) and of prefill on today's f32
    // kernels, with the states as close, yet not the same bits as the f32 run, so the cooperative-matrix kernels ran.
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 16, None);
    assert_eq!(e.prefill_kernels(), "split-f16");
    let f = Engine::open(
        &g,
        &d,
        &Options {
            cache_slots: Some(0),
            prefill_chunk: Some(16),
            f32_prefill: true,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(f.prefill_kernels(), "f32");
    let toks = tokens(&d);
    let (sl, ss) = prefilled(&e, &toks);
    let (fl, fs) = prefilled(&f, &toks);
    let (dl, _) = decoded(&e, &toks);
    let worst = |a: &[Vec<f32>], b: &[Vec<f32>]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| rel_l2(x, y))
            .fold(0f64, |m, e| if e.is_nan() || e > m { e } else { m })
    };
    let (vs_f32, vs_decode, st) = (worst(&sl, &fl), worst(&sl, &dl), state_err(&ss, &fs));
    eprintln!("split-f16 prefill: against f32 prefill {vs_f32:.3e} (state {st:.3e}), against decode {vs_decode:.3e}");
    assert!(vs_decode <= 1e-3, "against decode {vs_decode:.3e}");
    assert!(vs_f32 <= 1e-3, "against f32 prefill {vs_f32:.3e}");
    assert!(st <= 1e-3, "state against f32 prefill {st:.3e}");
    assert!(
        !same_logits(&sl, &fl),
        "split-f16 and f32 prefill gave the same bits: the cooperative-matrix kernels didn't run"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn f32_prefill_runs_todays_kernels() {
    // With `f32_prefill`, prefill takes today's f32 kernels and stays invariant to the chunk size and
    // the cache.
    let (g, d) = tiny();
    let open32 = |slots: usize, chunk: usize| {
        Engine::open(
            &g,
            &d,
            &Options {
                cache_slots: Some(slots),
                prefill_chunk: Some(chunk),
                f32_prefill: true,
                ..Options::default()
            },
        )
        .unwrap()
    };
    let toks = tokens(&d);
    let base = prefilled(&open32(0, 16), &toks);
    for (slots, chunk) in [(6, 7), (16, 1)] {
        let e = open32(slots, chunk);
        assert_eq!(e.prefill_kernels(), "f32");
        let run = prefilled(&e, &toks);
        assert!(
            same_logits(&run.0, &base.0) && same_state(&run.1, &base.1),
            "{slots} slots, chunks of {chunk}"
        );
    }
    // The f32 run's logits differ from a split-f16 run's on the same tokens: the test proves which kernels ran, not
    // only the label.
    let split = open(&g, &d, 0, 16, None);
    if split.prefill_kernels() == "split-f16" {
        assert!(
            !same_logits(&base.0, &prefilled(&split, &toks).0),
            "f32 prefill gave the same bits as split-f16 prefill"
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn prefill_runs_the_cooperative_matrix_attention_only_on_the_split_f16_path() {
    // Split-f16 prefill runs attn_prefill_cm and never attn_prefill; f32 prefill the reverse.
    let (g, d) = tiny();
    let toks = tokens(&d);
    for (f32_prefill, want, not) in [
        (false, "attn_prefill_cm", "attn_prefill"),
        (true, "attn_prefill", "attn_prefill_cm"),
    ] {
        let e = Engine::open(
            &g,
            &d,
            &Options {
                cache_slots: Some(0),
                prefill_chunk: Some(16),
                f32_prefill,
                ..Options::default()
            },
        )
        .unwrap();
        e.set_timing(true);
        let mut s = e.session(128).unwrap();
        s.prefill(&e, &toks).unwrap();
        let names: std::collections::BTreeSet<&str> = e
            .take_timings()
            .iter()
            .flat_map(|t| t.gpu.iter().map(|g| g.name))
            .collect();
        assert!(
            names.contains(want) && !names.contains(not),
            "f32_prefill {f32_prefill}: {names:?}"
        );
    }
}
