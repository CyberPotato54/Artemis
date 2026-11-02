//! Prefill's staging on the GPU, with the tiny model.
use engine::{Engine, Error, Options, SessionState};
use gguf::Gguf;
use kernels::dispatch::WHERE_HOST;
use qwen36::{dims::Dims, model::Model, testing::tiny_model};

fn tiny() -> (Gguf, Dims) {
    (Gguf::from_bytes(tiny_model(1)).unwrap(), Dims::tiny())
}

fn open(g: &Gguf, d: &Dims, slots: usize, opts: Options) -> Engine {
    Engine::open(
        g,
        d,
        &Options {
            cache_slots: Some(slots),
            ..opts
        },
    )
    .unwrap()
}

#[test]
#[ignore = "needs the GPU"]
fn the_staging_area_takes_what_staging_bytes_says() {
    // The fit sizes the cache with staging_bytes beside it, so it must be what the staging area allocates: two buffers
    // of (16 − slots) experts and the prefill table. No staging with every expert cached, or with `no_staging`.
    let (g, d) = tiny();
    let m = Model::open(&g, &d).unwrap();
    let sizes = engine::slot_sizes(&m);
    for slots in [0, 6, 16] {
        let e = open(&g, &d, slots, Options::default());
        assert_eq!(
            e.stats().staging_bytes,
            engine::staging_bytes(&sizes, d.n_expert, slots),
            "{slots} slots"
        );
        assert_eq!(e.staging_on(), slots < d.n_expert, "{slots} slots");
    }
    let off = open(
        &g,
        &d,
        6,
        Options {
            no_staging: true,
            ..Options::default()
        },
    );
    assert!(!off.staging_on());
    assert_eq!(off.stats().staging_bytes, 0);
}

#[test]
#[ignore = "needs the GPU"]
fn the_mirror_is_the_slot_table() {
    // A staged chunk's prefill table is built from the engine's mirror of the slot table: after twenty steps with a
    // forced swap each, a test's direct write and a staged change, the mirror and the table in VRAM agree entry by entry
    // once a step has applied the staged change.
    let (g, d) = tiny();
    let e = open(
        &g,
        &d,
        6,
        Options {
            force_swap_seed: Some(3),
            ..Options::default()
        },
    );
    let mut s = e.session(64).unwrap();
    for p in 0..20u32 {
        s.decode(&e, (p * 37 + 5) % d.n_vocab as u32).unwrap();
    }
    e.set_slot_entry(1, 2, e.host_slot_addr(1, 5), WHERE_HOST)
        .unwrap();
    e.stage_slot_entry(2, 3, e.host_slot_addr(2, 7), WHERE_HOST)
        .unwrap();
    s.decode(&e, 9).unwrap();
    for l in 0..=d.n_layer {
        for x in 0..d.n_expert {
            assert_eq!(
                e.mirror_entry(l, x),
                e.slot_entry(l, x).unwrap(),
                "layer {l} expert {x}"
            );
        }
    }
}

/// 40 tokens of the tiny model's vocabulary.
fn tokens(d: &Dims) -> Vec<u32> {
    (0..40u32)
        .map(|i| (i * 211 + 17) % d.n_vocab as u32)
        .collect()
}

/// Every position's logits and the final state through prefill, as bits.
fn prefilled(e: &Engine, toks: &[u32]) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut s = e.session(64).unwrap();
    let mut all = Vec::new();
    s.prefill_each(e, toks, 0, |_, l| {
        all.push(l.iter().map(|x| x.to_bits()).collect());
        Ok::<(), Error>(())
    })
    .unwrap();
    let SessionState { layers } = s.state(e).unwrap();
    let state = layers
        .into_iter()
        .flat_map(|(a, b)| [a, b])
        .map(|v| v.iter().map(|x| x.to_bits()).collect())
        .collect();
    (all, state)
}

fn staged(slots: usize, chunk: usize, from: usize) -> Options {
    Options {
        cache_slots: Some(slots),
        prefill_chunk: Some(chunk),
        stage_from: Some(from),
        ..Options::default()
    }
}

#[test]
#[ignore = "needs the GPU"]
fn staged_and_unstaged_prefill_give_the_same_bits() {
    // Test 5's rule: a staged expert runs the same kernel on the same bytes at another
    // address. Chunks of 8, staged from 1 token and never, at 0, 6 and all 16 slots, on the split-f16 kernels and on
    // the f32 ones: the same logits and state; the staged runs read from the staging buffers, and every
    // read is counted once.
    let (g, d) = tiny();
    let toks = tokens(&d);
    let reads = (toks.len() * d.n_layer * d.n_expert_used) as u64;
    for f32_prefill in [false, true] {
        for slots in [0, 6, 16] {
            let open = |from: usize| {
                Engine::open(
                    &g,
                    &d,
                    &Options {
                        f32_prefill,
                        ..staged(slots, 8, from)
                    },
                )
                .unwrap()
            };
            let (on, off) = (open(1), open(usize::MAX));
            let case = format!("{} kernels, {slots} slots", on.prefill_kernels());
            assert_eq!(
                on.prefill_kernels(),
                if f32_prefill { "f32" } else { "split-f16" }
            );
            assert!(prefilled(&on, &toks) == prefilled(&off, &toks), "{case}");
            let (a, b) = (on.stats(), off.stats());
            assert_eq!(a.hits + a.staged + a.misses, reads, "{case}: {a:?}");
            assert_eq!(
                (b.staged, b.staged_steps),
                (0, 0),
                "{case}, unstaged: {b:?}"
            );
            if slots < d.n_expert {
                assert_eq!(
                    a.staged_steps, 5,
                    "{case}: five chunks of eight, all staged"
                );
                assert!(a.staged > 0 && a.misses == 0, "{case}: {a:?}");
            } else {
                assert!(!on.staging_on());
                assert_eq!(
                    a.staged_steps, 0,
                    "{case}: every expert cached, nothing to stage"
                );
            }
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn chunks_at_and_below_the_threshold_mix_and_change_no_bit() {
    // 40 tokens in chunks of 16 are 16, 16 and 8. From 16 tokens on, the first two are staged and the
    // last isn't; from 17, none is. Every mix gives the bits of an unstaged run.
    let (g, d) = tiny();
    let toks = tokens(&d);
    let base = prefilled(
        &Engine::open(&g, &d, &staged(6, 16, usize::MAX)).unwrap(),
        &toks,
    );
    for (from, want) in [(16, 2), (17, 0), (8, 3), (9, 2)] {
        let e = Engine::open(&g, &d, &staged(6, 16, from)).unwrap();
        assert!(prefilled(&e, &toks) == base, "staged from {from}");
        assert_eq!(e.stats().staged_steps, want, "staged from {from}");
    }
}

#[test]
#[ignore = "needs the GPU"]
fn staged_prefill_with_poisoned_late_copies_changes_no_bit() {
    // Every staging copy fills a 64 MiB scratch first and then poisons its regions with f16 NaNs, so
    // it lands long after the layer's compute was submitted; a layer that read its buffer without waiting would read
    // stale experts (another layer's) or NaN. With a swap forced every step as well, chunks of 5, all staged, on either
    // kernels: the bits of an engine on the same kernels reading every expert from host memory. Decoding goes on from it
    // unchanged.
    let (g, d) = tiny();
    let toks = tokens(&d);
    for f32_prefill in [false, true] {
        let plain = Engine::open(
            &g,
            &d,
            &Options {
                f32_prefill,
                ..staged(0, 5, usize::MAX)
            },
        )
        .unwrap();
        let e = Engine::open(
            &g,
            &d,
            &Options {
                f32_prefill,
                force_swap_seed: Some(7),
                poison_staging: true,
                ..staged(6, 5, 1)
            },
        )
        .unwrap();
        let kernels = e.prefill_kernels();
        assert!(
            prefilled(&e, &toks) == prefilled(&plain, &toks),
            "{kernels}: the prefill differs"
        );
        let st = e.stats();
        assert_eq!(
            (st.staged_steps, st.swaps_started),
            (8, 8),
            "{kernels}: {st:?}"
        );
        let (mut a, mut b) = (e.session(64).unwrap(), plain.session(64).unwrap());
        a.prefill(&e, &toks).unwrap();
        b.prefill(&plain, &toks).unwrap();
        for t in [3u32, 99, 250] {
            assert!(
                a.decode(&e, t).unwrap() == b.decode(&plain, t).unwrap(),
                "{kernels}: decoding after token {t}"
            );
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn a_layer_with_more_host_experts_than_the_buffer_holds_reads_the_rest_in_place() {
    // On the GPU, a cached expert of layer 1 is pointed at its own pinned copy behind the planner, so
    // the layer has 11 HOST experts for 10 places. The plan stages ten and leaves the highest HOST expert in place; the
    // bits are those of an engine with no cache.
    let (g, d) = tiny();
    let toks = tokens(&d);
    let plain = Engine::open(&g, &d, &staged(0, 8, usize::MAX)).unwrap();
    let e = Engine::open(&g, &d, &staged(6, 8, 1)).unwrap();
    let cached = (0..d.n_expert)
        .find(|&x| matches!(e.planner_entry(1, x), Some(engine::Entry::Vram(_))))
        .unwrap();
    e.set_slot_entry(1, cached, e.host_slot_addr(1, cached), WHERE_HOST)
        .unwrap();
    assert!(prefilled(&e, &toks) == prefilled(&plain, &toks));
    assert_eq!(
        e.stats().staged_steps,
        5,
        "five chunks of eight, all staged"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_failed_staging_copy_names_its_layer_and_the_next_step_runs_clean() {
    // A staging copy that fails fails the step with Error::Step, naming the chunk's first
    // position and the layer. Layer 2's copy fails once, after layer 0 has run (copies 0 and 1 have gone out). The
    // step's slot-table change is staged again and applied by the next step, and the next prefill, on a different
    // layer-1 layout, gives the no-cache engine's bits.
    //
    // What this doesn't exercise: the failed step's in-flight copies have landed before the retry. Each
    // `set_slot_entry` below is an upload the host waits for on the transfer queue, and a timeline signal's first
    // synchronization scope covers every earlier submit on that queue. So the ordering of a retry's copies after a
    // failed step's still-in-flight ones isn't tested here: it rests on `stage_async`'s leading transfer barrier,
    // checked by reading.
    let (g, d) = tiny();
    let toks = tokens(&d);
    let plain = Engine::open(&g, &d, &staged(0, 8, usize::MAX)).unwrap();
    let e = Engine::open(
        &g,
        &d,
        &Options {
            poison_staging: true,
            fail_staging_once: Some(2),
            ..staged(6, 8, 1)
        },
    )
    .unwrap();
    // A change that reads the same bytes: a cached expert of layer 1 at its own pinned copy.
    let x = (0..d.n_expert)
        .find(|&x| matches!(e.planner_entry(1, x), Some(engine::Entry::Vram(_))))
        .unwrap();
    let host = e.host_slot_addr(1, x);
    let saved = e.slot_entry(1, x).unwrap();
    assert_ne!(saved, (host, WHERE_HOST), "expert {x} is cached");
    e.stage_slot_entry(1, x, host, WHERE_HOST).unwrap();
    let mut s = e.session(64).unwrap();
    match s.prefill(&e, &toks[..8]) {
        Err(Error::Step {
            pos: 0,
            layer: Some(2),
            source: vk::Error::Timeout,
        }) => {}
        other => panic!("expected a failed step at layer 2, got {:?}", other.err()),
    }
    assert!(
        e.last_prefill_routes().is_empty(),
        "the routing log holds no chunk"
    );
    assert_eq!(
        e.slot_entry(1, x).unwrap(),
        (host, WHERE_HOST),
        "applied by layer 0's submit"
    );
    // Undo it behind the planner: only the restaged change can put it back.
    e.set_slot_entry(1, x, saved.0, saved.1).unwrap();
    // Another cached expert of layer 1, y, at its own pinned copy behind the planner: in the retry it takes one of
    // buffer 1's places ahead of some of the failed step's staged experts, which move to other places, so the retry's
    // layer-1 layout differs from the failed step's. It reads the same bytes, so the bits are still the no-cache
    // engine's.
    let cap = d.n_expert - e.cache_slots();
    let host_in_retry =
        |z: usize| z == x || matches!(e.planner_entry(1, z), Some(engine::Entry::Host));
    let y = (0..d.n_expert)
        .find(|&z| z != x && matches!(e.planner_entry(1, z), Some(engine::Entry::Vram(_))))
        .unwrap();
    assert!(
        (0..y).filter(|&z| host_in_retry(z)).count() < cap,
        "expert {y} takes one of buffer 1's {cap} places, so the staged experts after it move"
    );
    e.set_slot_entry(1, y, e.host_slot_addr(1, y), WHERE_HOST)
        .unwrap();
    assert!(
        prefilled(&e, &toks) == prefilled(&plain, &toks),
        "the next prefill differs"
    );
    assert_eq!(
        e.slot_entry(1, x).unwrap(),
        (host, WHERE_HOST),
        "the next step applied it again"
    );
}

/// The expert kernels read slots 16 bytes at a time: the engine refuses a HOST entry staged at a misaligned
/// address, the way the planner publishes every runtime entry. The aligned entry before it goes through, so the panic is
/// the check's.
#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "the expert kernels read 16-byte-aligned slots")]
fn staging_a_misaligned_slot_is_refused() {
    let (g, d) = tiny();
    let e = open(&g, &d, 6, Options::default());
    let host = e.host_slot_addr(1, 3);
    e.stage_slot_entry(1, 3, host, WHERE_HOST).unwrap();
    let _ = e.stage_slot_entry(1, 3, host + 8, WHERE_HOST);
}

/// The same for a direct write of the table.
#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "the expert kernels read 16-byte-aligned slots")]
fn writing_a_misaligned_slot_is_refused() {
    let (g, d) = tiny();
    let e = open(&g, &d, 6, Options::default());
    let host = e.host_slot_addr(1, 3);
    e.set_slot_entry(1, 3, host, WHERE_HOST).unwrap();
    let _ = e.set_slot_entry(1, 3, host + 8, WHERE_HOST);
}
