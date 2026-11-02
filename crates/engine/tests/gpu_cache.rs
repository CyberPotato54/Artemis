//! The adaptive cache on the GPU, with the tiny model: the policy's swaps, and
//! chunks retired and grown under a lowered budget. Wherever an expert lives, the logits are those of an engine reading
//! every expert from host memory (test 5's rule).
use engine::{CachePolicyKind, Engine, Entry, Error, Options, VRAM_MARGIN};
use gguf::Gguf;
use qwen36::{dims::Dims, testing::tiny_model};

fn tiny() -> (Gguf, Dims) {
    (Gguf::from_bytes(tiny_model(1)).unwrap(), Dims::tiny())
}

/// An engine reading every expert from host memory, prefilling in chunks of `chunk`, never staged.
fn plain(g: &Gguf, d: &Dims, chunk: usize) -> Engine {
    Engine::open(
        g,
        d,
        &Options {
            cache_slots: Some(0),
            prefill_chunk: Some(chunk),
            stage_from: Some(usize::MAX),
            ..Options::default()
        },
    )
    .unwrap()
}

fn tokens(d: &Dims, n: u32, mul: u32) -> Vec<u32> {
    (0..n).map(|i| (i * mul + 17) % d.n_vocab as u32).collect()
}

/// Every position's logits as bits: `prompt` through prefill, then `decode` tokens one at a time. `between` runs after
/// each decoded token (a budget change, say).
fn run(
    e: &Engine,
    prompt: &[u32],
    decode: &[u32],
    mut between: impl FnMut(usize),
) -> Vec<Vec<u32>> {
    let mut s = e.session(128).unwrap();
    let mut all = Vec::new();
    s.prefill_each(e, prompt, 0, |_, l| {
        all.push(l.iter().map(|x| x.to_bits()).collect());
        Ok::<(), Error>(())
    })
    .unwrap();
    for (i, &t) in decode.iter().enumerate() {
        all.push(
            s.decode(e, t)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect(),
        );
        between(i);
    }
    all
}

#[test]
#[ignore = "needs the GPU"]
fn the_policy_swaps_live_and_changes_no_logit() {
    // A cache of 6 slots per layer whose policy adapts swaps as the routes move, every copy
    // poisoned first (a read that overlapped a copy would see NaN and fail the step); the logits are those of no
    // cache. The static policy swaps nothing. The poison doesn't delay a copy, which lands within microseconds on the
    // tiny model: the ordering evidence is the planner fuzz, not this test.
    let (g, d) = tiny();
    let (prompt, rest) = (tokens(&d, 40, 211), tokens(&d, 40, 37));
    let want = run(&plain(&g, &d, 8), &prompt, &rest, |_| {});
    let mut hits = Vec::new();
    for policy in [CachePolicyKind::Adaptive, CachePolicyKind::Static] {
        let e = Engine::open(
            &g,
            &d,
            &Options {
                cache_slots: Some(6),
                prefill_chunk: Some(8),
                cache_policy: policy,
                poison_swaps: true,
                ..Options::default()
            },
        )
        .unwrap();
        assert!(run(&e, &prompt, &rest, |_| {}) == want, "{policy:?}");
        let st = e.stats();
        eprintln!("{policy:?}: {st:?}");
        assert_eq!(e.policy_adapts(), policy == CachePolicyKind::Adaptive);
        if policy == CachePolicyKind::Adaptive {
            assert!(st.swaps_started > 5 && st.swaps_done > 0, "{st:?}");
        } else {
            assert_eq!(st.swaps_started, 0, "{st:?}");
        }
        assert!(st.hits > 0, "{st:?}");
        hits.push(st.hits);
    }
    // The policy earns its swaps: the adaptive cache hits more than the static one (the run: 658 against 474).
    assert!(
        hits[0] > hits[1],
        "adaptive hits {} against static hits {}",
        hits[0],
        hits[1]
    );
}

// ---- The bank carried between steps ----

#[test]
#[ignore = "needs the GPU"]
fn staged_chunks_save_their_share_and_the_first_decode_step_spends_it() {
    // Every chunk of the prompt is staged, so the prefill makes no swap (`allowance`) and banks its share; a cap of
    // about one slot a token lets the first decode step spend the bank, which one token's share could not (a step that
    // lost the bank would start about one swap). The logits are those of no cache.
    let (g, d) = tiny();
    let (prompt, rest) = (tokens(&d, 40, 211), tokens(&d, 40, 37));
    let want = run(&plain(&g, &d, 8), &prompt, &rest, |_| {});
    let e = Engine::open(
        &g,
        &d,
        &Options {
            cache_slots: Some(6),
            prefill_chunk: Some(8),
            stage_from: Some(1),
            upload_cap: Some(200_000),
            cache_policy: CachePolicyKind::Adaptive,
            poison_swaps: true,
            ..Options::default()
        },
    )
    .unwrap();
    let mut s = e.session(128).unwrap();
    let mut got = Vec::new();
    s.prefill_each(&e, &prompt, 0, |_, l| {
        got.push(l.iter().map(|x| x.to_bits()).collect::<Vec<u32>>());
        Ok::<(), Error>(())
    })
    .unwrap();
    let after_prefill = e.stats();
    assert!(after_prefill.staged_steps > 0, "{after_prefill:?}");
    assert_eq!(after_prefill.swaps_started, 0, "{after_prefill:?}");
    got.push(
        s.decode(&e, rest[0])
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect(),
    );
    let after_decode = e.stats();
    eprintln!("after the first decode step: {after_decode:?}");
    assert!(
        after_decode.swaps_started > 3,
        "the staged chunks' share should be spent at once: {after_decode:?}"
    );
    for &t in &rest[1..] {
        got.push(
            s.decode(&e, t)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect(),
        );
    }
    assert!(got == want);
}

/// A cache of 6 slots per layer in chunks of 2 rows, checking the budget every step, with poisoned copies.
fn chunked(g: &Gguf, d: &Dims, policy: CachePolicyKind) -> Engine {
    Engine::open(
        g,
        d,
        &Options {
            cache_slots: Some(6),
            chunk_rows: Some(2),
            budget_every: Some(1),
            prefill_chunk: Some(8),
            cache_policy: policy,
            poison_swaps: true,
            ..Options::default()
        },
    )
    .unwrap()
}

/// The cut that leaves the budget `short` bytes under what the engine's usage and the margin need: the line a shrink
/// waits for (`BudgetWatch`).
fn cut_for(e: &Engine, short: u64) -> u64 {
    let h = e.vram_heap().unwrap();
    (h.budget + short).saturating_sub(h.usage + VRAM_MARGIN)
}

/// Every logit of `tokens` as bits, one decoded token at a time, in session `s`.
fn decoded(e: &Engine, s: &mut engine::Session, tokens: &[u32]) -> Vec<Vec<u32>> {
    tokens
        .iter()
        .map(|&t| {
            s.decode(e, t)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect()
        })
        .collect()
}

#[test]
#[ignore = "needs the GPU"]
fn a_short_budget_retires_the_last_chunk_and_a_freed_budget_grows_it_back() {
    // The shrink and grow tests, with both policies. The budget is cut to a byte short of the margin:
    // the next check retires rows 4 and 5, the chunk is freed once they drain, and the margin is back. While the cut
    // holds there is no room to grow: the chunk would bring usage back to where the cut leaves less than the margin. The
    // cut lifted, two checks in a row find room for the chunk, it is allocated again, and the policy fills its rows
    // (the static one in the profile's order).
    // Then between sessions: with the budget cut again, a new session's check retires the chunk and a reset's
    // check, the chunk still retiring, waits for it. Every logit is no cache's.
    let (g, d) = tiny();
    let (prompt, rest, more) = (tokens(&d, 20, 211), tokens(&d, 30, 37), tokens(&d, 3, 101));
    let base = plain(&g, &d, 8);
    let want = run(&base, &prompt, &rest, |_| {});
    let want_more = decoded(&base, &mut base.session(64).unwrap(), &more);
    for policy in [CachePolicyKind::Adaptive, CachePolicyKind::Static] {
        let e = chunked(&g, &d, policy);
        let full = e.stats().cache_bytes;
        let chunk = full / 3;
        let (cut, mut seen) = (std::cell::Cell::new(0), Vec::new());
        let got = run(&e, &prompt, &rest, |i| {
            let st = e.stats();
            match i {
                4 => {
                    cut.set(cut_for(&e, 1));
                    e.set_budget_cut(cut.get());
                }
                12 => {
                    // Retired and freed: the margin is back, under the cut budget.
                    let h = e.vram_heap().unwrap();
                    assert!(h.usage + VRAM_MARGIN <= h.budget - cut.get(), "{h:?}");
                    assert_eq!(
                        (st.shrinks, e.live_slots(), st.cache_bytes),
                        (1, 4, full - chunk),
                        "{st:?}"
                    );
                    // The last chunk went: rows 4 and 5 are dead, rows 0 to 3 still held.
                    assert!(e.cache_slot_bytes(0, 4).is_err() && e.cache_slot_bytes(0, 5).is_err());
                    assert!(e.cache_slot_bytes(3, 3).is_ok());
                    e.set_budget_cut(0);
                }
                _ => {}
            }
            seen.push((st.shrinks, st.grows, e.live_slots()));
        });
        assert!(got == want, "{policy:?}: the logits differ");
        let st = e.stats();
        eprintln!("{policy:?}: {st:?}\n{seen:?}");
        assert_eq!(
            (st.shrinks, st.grows, st.grows_refused),
            (1, 1, 0),
            "{policy:?}: {st:?}"
        );
        assert_eq!((e.live_slots(), st.cache_bytes), (6, full), "{policy:?}");
        assert!(
            st.fills >= 8,
            "{policy:?}: the grown rows' 8 slots filled and published: {st:?}"
        );
        assert_eq!(st.budget_short, 0);
        assert_eq!(
            st.swaps_started > 0,
            policy == CachePolicyKind::Adaptive,
            "{st:?}"
        );
        // Between sessions.
        e.set_budget_cut(cut_for(&e, 1));
        let mut s = e.session(64).unwrap();
        assert_eq!(
            (e.stats().shrinks, e.live_slots()),
            (2, 4),
            "{policy:?}: the new session's check retires the chunk"
        );
        s.reset(&e).unwrap();
        assert_eq!(
            e.stats().shrinks,
            2,
            "{policy:?}: the reset's check waits for the chunk still retiring"
        );
        assert!(
            decoded(&e, &mut s, &more) == want_more,
            "{policy:?}: the logits differ after the reset"
        );
        assert_eq!(
            (e.stats().shrinks, e.stats().cache_bytes),
            (2, full - chunk),
            "{policy:?}"
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn a_chunk_retired_in_the_middle_of_a_staged_prompt_changes_no_bit() {
    // The budget is cut before a prompt of five staged chunks; the first chunk's check retires rows 4 and
    // 5, whose experts go HOST: two a layer more than the staging buffer's 10 places, read in place. The chunk is freed
    // at the next chunk's start, once they drain. Staging and swap copies are late and poisoned. The bits are those of
    // an engine with no cache that never stages. (The engine waits for each step, so the rows drain at once here: that
    // a chunk is freed only once its rows are dead, copies into them landed, is the fuzz's check (d).)
    let (g, d) = tiny();
    let toks = tokens(&d, 40, 211);
    let prefilled = |e: &Engine, s: &mut engine::Session| -> Vec<Vec<u32>> {
        let mut all = Vec::new();
        s.prefill_each(e, &toks, 0, |_, l| {
            all.push(l.iter().map(|x| x.to_bits()).collect());
            Ok::<(), Error>(())
        })
        .unwrap();
        all
    };
    let base = plain(&g, &d, 8);
    let want = prefilled(&base, &mut base.session(64).unwrap());
    let e = Engine::open(
        &g,
        &d,
        &Options {
            cache_slots: Some(6),
            chunk_rows: Some(2),
            budget_every: Some(1),
            prefill_chunk: Some(8),
            stage_from: Some(1),
            poison_swaps: true,
            poison_staging: true,
            ..Options::default()
        },
    )
    .unwrap();
    let full = e.stats().cache_bytes;
    let mut s = e.session(64).unwrap();
    // A first staged chunk makes the driver allocate its working memory (`DRIVER_WORK`), far more than a chunk of the
    // tiny model's cache: the budget is cut once that is in.
    s.prefill(&e, &toks[..8]).unwrap();
    s.reset(&e).unwrap();
    e.set_budget_cut(cut_for(&e, 1));
    assert!(prefilled(&e, &mut s) == want, "the logits differ");
    let st = e.stats();
    eprintln!("{st:?}");
    assert_eq!(
        (st.shrinks, st.staged_steps, e.live_slots(), st.cache_bytes),
        (1, 6, 4, full / 3 * 2),
        "{st:?}"
    );
    assert!(
        st.misses > 0,
        "the experts past the staging buffer's places were read in place: {st:?}"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_budget_too_short_for_any_cache_retires_every_chunk_and_says_so() {
    // The budget cut far below what the engine uses retires the chunks one after another, each once the one
    // before has drained, and with none left each check reports the shortfall. Reads go to host memory; no logit moves.
    let (g, d) = tiny();
    let (prompt, rest) = (tokens(&d, 20, 211), tokens(&d, 20, 37));
    let want = run(&plain(&g, &d, 8), &prompt, &rest, |_| {});
    let e = chunked(&g, &d, CachePolicyKind::Adaptive);
    let got = run(&e, &prompt, &rest, |i| {
        if i == 0 {
            e.set_budget_cut(u64::MAX / 2);
        }
    });
    assert!(got == want, "the logits differ");
    let st = e.stats();
    eprintln!("{st:?}");
    assert_eq!(
        (st.shrinks, e.live_slots(), st.cache_bytes),
        (3, 0, 0),
        "{st:?}"
    );
    assert!(st.budget_short > 5, "{st:?}");
}

#[test]
#[ignore = "needs the GPU"]
fn the_ceiling_bounds_grown_rows_at_every_check() {
    // `--cache-slots` is the most slots the cache holds, grown rows included. With 6 slots of its
    // own and room in the budget, a ceiling of 8 grows a chunk of 2 rows and no more; with none, the cache grows a chunk
    // of the other 10. Checking every step, the cache reaches the ceiling while decoding; checking only between
    // sessions (new sessions, resets), it reaches it there. Neither grows past it, and the bits after the checks are no
    // cache's. Without staging buffers every row is the cache's own.
    let (g, d) = tiny();
    let rest = tokens(&d, 6, 37);
    let base = plain(&g, &d, 8);
    let want = decoded(&base, &mut base.session(64).unwrap(), &rest);
    for (every, ceiling, live) in [
        (1, Some(8), 8),
        (1, None, 16),
        (usize::MAX, Some(8), 8),
        (usize::MAX, None, 16),
    ] {
        let at = format!("checking every {every} tokens, ceiling {ceiling:?}");
        let e = Engine::open(
            &g,
            &d,
            &Options {
                cache_slots: ceiling,
                own_slots: Some(6),
                no_staging: true,
                budget_every: Some(every),
                poison_swaps: true,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!((e.cache_slots(), e.live_slots()), (6, 6), "{at}");
        let mut s = e.session(64).unwrap();
        assert!(
            decoded(&e, &mut s, &rest) == want,
            "{at}: the logits differ"
        );
        let st = e.stats();
        eprintln!("{at}: {st:?}");
        // Checking every step, two checks with room grow the chunk while decoding; only the new session's check runs
        // otherwise, and growing takes two.
        let grown = if every == 1 { (1, live) } else { (0, 6) };
        assert_eq!((st.grows, e.live_slots()), grown, "{at}: {st:?}");
        for _ in 0..4 {
            s.reset(&e).unwrap();
            drop(e.session(64).unwrap());
        }
        assert!(
            decoded(&e, &mut s, &rest) == want,
            "{at}: the logits differ after the resets"
        );
        assert_eq!(
            (e.stats().grows, e.live_slots()),
            (1, live),
            "{at}: one grow, to the ceiling and no further"
        );
    }
}

// ---- A grow the budget refuses ----

#[test]
#[ignore = "needs the GPU"]
fn a_grow_refused_once_its_chunk_is_allocated_is_given_back_and_tried_again_after_the_wait() {
    // The oscillation risk: the driver lowers the budget once usage passes 11.5 GB, so a chunk allocated with room
    // read below that line can leave less than the margin and `GROW_SLACK` (`set_grow_cut` lowers only the reading taken
    // with the chunk allocated). Such a grow is given back at once and counted, and growing waits 2 checks with room.
    // With 6 rows of its own and a ceiling of 8 (no staging buffers, so every row is the cache's own), the first grow
    // appends rows 6 and 7 after the last group and is refused: they read as dead, and the cache holds what it did. The
    // hook cleared, the grow after the wait reuses that group. Then the budget cut a byte short of the margin retires the
    // grown chunk and frees it; lifted, with the hook set again, the grow into the freed chunk is refused too, and comes
    // back after the wait. Every logit is no cache's.
    let (g, d) = tiny();
    let rest = tokens(&d, 24, 37);
    let base = plain(&g, &d, 8);
    let want = decoded(&base, &mut base.session(64).unwrap(), &rest);
    let e = Engine::open(
        &g,
        &d,
        &Options {
            cache_slots: Some(8),
            own_slots: Some(6),
            chunk_rows: Some(2),
            no_staging: true,
            budget_every: Some(1),
            poison_swaps: true,
            ..Options::default()
        },
    )
    .unwrap();
    let full = e.stats().cache_bytes;
    let chunk = full / 3;
    // (shrinks, grows, grows refused, live slots, cache bytes)
    let state = || {
        let st = e.stats();
        (
            st.shrinks,
            st.grows,
            st.grows_refused,
            e.live_slots(),
            st.cache_bytes,
        )
    };
    e.set_grow_cut(u64::MAX / 2);
    // The new session's check has room, and the first token's, the second in a row, grows rows 6 and 7: refused.
    let mut s = e.session(64).unwrap();
    let mut got = decoded(&e, &mut s, &rest[..1]);
    assert_eq!(
        state(),
        (0, 0, 1, 6, full),
        "the refused chunk is given back"
    );
    assert!(matches!(
        e.cache_slot_bytes(0, 6),
        Err(Error::DeadSlot { layer: 0, slot: 6 })
    ));
    assert!(matches!(
        e.cache_slot_addr(3, 7),
        Err(Error::DeadSlot { layer: 3, slot: 7 })
    ));
    assert!(matches!(
        e.cache_slot_bytes(0, 8),
        Err(Error::NoSuchSlot { slots: 6, .. })
    ));
    assert!(e.cache_slot_bytes(0, 5).is_ok());
    // Growing waits 2 checks with room; the third grows into the group the refusal left.
    e.set_grow_cut(0);
    got.extend(decoded(&e, &mut s, &rest[1..3]));
    assert_eq!(state(), (0, 0, 1, 6, full), "growing waits");
    got.extend(decoded(&e, &mut s, &rest[3..4]));
    assert_eq!(state(), (0, 1, 1, 8, full + chunk), "grown after the wait");
    assert!(e.cache_slot_bytes(0, 7).is_ok());
    // The budget cut a byte short of the margin retires the grown chunk, which is freed once its rows drain; while the
    // cut holds there is no room to grow it back.
    e.set_budget_cut(cut_for(&e, 1));
    got.extend(decoded(&e, &mut s, &rest[4..8]));
    assert_eq!(state(), (1, 1, 1, 6, full), "retired and freed");
    // The cut lifted and the hook set again: the grow into the freed chunk is refused, then comes back after the wait.
    e.set_budget_cut(0);
    e.set_grow_cut(u64::MAX / 2);
    got.extend(decoded(&e, &mut s, &rest[8..10]));
    assert_eq!(state(), (1, 1, 2, 6, full), "refused again");
    assert!(matches!(
        e.cache_slot_bytes(0, 6),
        Err(Error::DeadSlot { .. })
    ));
    e.set_grow_cut(0);
    got.extend(decoded(&e, &mut s, &rest[10..13]));
    assert_eq!(state(), (1, 2, 2, 8, full + chunk), "back after the wait");
    got.extend(decoded(&e, &mut s, &rest[13..]));
    let st = e.stats();
    eprintln!("{st:?}");
    assert!(got == want, "the logits differ");
    assert_eq!(
        (st.grows, st.grows_refused, e.live_slots()),
        (2, 2, 8),
        "{st:?}"
    );
    assert!(st.fills >= 8, "the grown rows' 8 slots filled: {st:?}");
}

/// The tiny model with 6 slots of its own (as a tight budget would fit), prefill chunks of 8 staged, poisoned copies.
fn lending(cache_slots: Option<usize>, policy: CachePolicyKind, fail: Option<usize>) -> Options {
    Options {
        cache_slots,
        own_slots: Some(6),
        prefill_chunk: Some(8),
        stage_from: Some(8),
        poison_staging: true,
        poison_swaps: true,
        fail_staging_once: fail,
        cache_policy: policy,
        ..Options::default()
    }
}

/// Starts a swap in every layer between an expert in the cache's own slots (below `own`) and one in host memory,
/// trying pairs until the planner takes one (an expert already in a move is refused).
fn swap_in_every_layer(e: &Engine, d: &Dims, own: usize) {
    for l in 0..d.n_layer {
        let at = |x| e.planner_entry(l, x);
        let outs: Vec<usize> = (0..d.n_expert)
            .filter(|&x| matches!(at(x), Some(Entry::Vram(s)) if s < own))
            .collect();
        let ins: Vec<usize> = (0..d.n_expert)
            .filter(|&x| at(x) == Some(Entry::Host))
            .collect();
        assert!(
            outs.iter()
                .any(|&o| ins.iter().any(|&i| e.request_swap(l, o, i).is_ok())),
            "layer {l}: no swap could start"
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn the_lent_rows_go_to_staging_for_a_staged_chunk_and_come_back() {
    // Lending, with both policies. With 6 slots of its own and a ceiling of 11, the cache gets the rows
    // the staging buffers can lend: two buffers of 10 experts hold 5 rows of the tiny model's 4 slots. A staged chunk
    // takes them back (their experts HOST, staged with the rest); the decode after it gets them back empty and fills
    // them (the static policy in the profile's order). Staging copies are late and poisoned, swap and fill copies
    // poisoned: a step that read a lent slot while staging wrote it would change the bits. Two prompts with decoding
    // between, every lent fill published before the second; the second starts with a swap in flight in every layer
    // (a new user turn while decoding's last swaps land), which leaves 11 experts a layer HOST against
    // the staging buffer's 10 places, so an extra one is read in place. Then fills into the lent rows that a staged
    // chunk's take cancels: an unstaged chunk returns the rows and picks their 20 fills, and the next staged chunk's
    // take finds them Copying (one decoded token between sent them; each is kept as an eviction until its copy lands)
    // or Draining (none between; never sent, they are dropped). Neither is ever published, and the rows come back after
    // the staged chunk and fill again. A fill's copy lands within microseconds on the tiny model, long before the late
    // staging copies, so that staging's copies land after a cancelled fill's rests on `stage_async`'s leading barrier,
    // checked by reading. Then a staged chunk that fails, and decoding after it.
    let (g, d) = tiny();
    let (p1, p2, rest) = (tokens(&d, 24, 211), tokens(&d, 16, 53), tokens(&d, 12, 37));
    // `rest` and then 16 more of the same sequence, for decoding until fills have published.
    let more = tokens(&d, 28, 37);
    let base = plain(&g, &d, 8);
    let bits = |l: Vec<f32>| l.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    for policy in [CachePolicyKind::Adaptive, CachePolicyKind::Static] {
        let e = Engine::open(&g, &d, &lending(Some(11), policy, None)).unwrap();
        assert_eq!(
            (e.cache_slots(), e.stats().lent_slots, e.live_slots()),
            (6, 5, 11)
        );
        let (mut a, mut b) = (base.session(128).unwrap(), e.session(128).unwrap());
        for (i, p) in [&p1, &p2].into_iter().enumerate() {
            let before = e.stats();
            if i == 1 {
                swap_in_every_layer(&e, &d, 6);
            }
            assert!(
                bits(b.prefill(&e, p).unwrap()) == bits(a.prefill(&base, p).unwrap()),
                "{policy:?}"
            );
            // The last chunk was staged: the lent rows are out of the cache.
            assert_eq!(e.live_slots(), 6, "{policy:?}");
            if i == 1 {
                let st = e.stats();
                assert!(
                    st.misses > before.misses,
                    "{policy:?}: an expert past the staging buffer's places was read in place: {st:?}"
                );
            }
            for &t in &rest {
                assert!(
                    bits(b.decode(&e, t).unwrap()) == bits(a.decode(&base, t).unwrap()),
                    "{policy:?}"
                );
            }
            assert_eq!(
                e.live_slots(),
                11,
                "{policy:?}: back after the staged chunks"
            );
        }
        let st = e.stats();
        eprintln!("{policy:?}: {st:?}");
        assert_eq!(st.staged_steps, 5);
        assert!(
            st.fills >= 40,
            "{policy:?}: both returns filled all 20 lent slots and published them: {st:?}"
        );
        assert_eq!(
            st.swaps_started > 4,
            policy == CachePolicyKind::Adaptive,
            "{st:?}"
        );
        // Fills into the lent rows cancelled by a staged chunk's take (`SwapPlanner::retire`): 11 tokens prefill as a
        // staged chunk of 8 and an unstaged one of 3, whose step returns the lent rows and picks their 20 fills with the
        // staged chunk's share. With one decoded token between, its poll sends them and the next staged chunk's take
        // finds them Copying; with none, Draining.
        for between in [1, 0] {
            let at = format!("{policy:?}, {between} decoded token(s) between");
            let e = Engine::open(&g, &d, &lending(Some(11), policy, None)).unwrap();
            let (mut a, mut b) = (base.session(128).unwrap(), e.session(128).unwrap());
            assert!(
                bits(b.prefill(&e, &p1[..11]).unwrap())
                    == bits(a.prefill(&base, &p1[..11]).unwrap()),
                "{at}"
            );
            assert_eq!(
                e.live_slots(),
                11,
                "{at}: the unstaged chunk returned the rows"
            );
            for &t in &more[..between] {
                assert!(
                    bits(b.decode(&e, t).unwrap()) == bits(a.decode(&base, t).unwrap()),
                    "{at}"
                );
            }
            // A copy a poll sends publishes at a later step's poll at the earliest, so none of the fills has yet.
            assert_eq!(e.stats().fills, 0, "{at}: the fills are still in flight");
            assert!(
                bits(b.prefill(&e, &p2[..8]).unwrap()) == bits(a.prefill(&base, &p2[..8]).unwrap()),
                "{at}"
            );
            assert_eq!(e.live_slots(), 6, "{at}: the staged chunk took the rows");
            // The rows come back at the first decoded token; their fills go out and publish at later steps' polls, as
            // the transfer queue lands them: decode until all 20 have published, at most 16 tokens.
            let mut took = 0;
            while e.stats().fills < 20 && took < 16 {
                let t = more[between + took];
                assert!(
                    bits(b.decode(&e, t).unwrap()) == bits(a.decode(&base, t).unwrap()),
                    "{at}"
                );
                took += 1;
            }
            let st = e.stats();
            eprintln!(
                "{at}: {} fills published after {took} decoded tokens",
                st.fills
            );
            assert_eq!(
                e.live_slots(),
                11,
                "{at}: back after the staged chunk ({took} decoded tokens)"
            );
            assert_eq!(
                (st.staged_steps, st.fills),
                (2, 20),
                "{at}: the cancelled fills never published; the returned rows' 20 slots filled and published within \
                 {took} decoded tokens: {st:?}"
            );
        }
        // A staged chunk whose layer-2 copy fails, its layer-1 copy sent ahead and possibly still landing: the decode
        // after it returns the lent rows and fills them, behind that copy on the transfer queue. That copy is late on
        // purpose, so after the 12 tokens decoding goes on until the 20 fills have published, at most 16 tokens more.
        let f = Engine::open(&g, &d, &lending(Some(11), policy, Some(2))).unwrap();
        let (mut a, mut b) = (base.session(128).unwrap(), f.session(128).unwrap());
        assert!(matches!(
            b.prefill(&f, &p1[..8]),
            Err(Error::Step { layer: Some(2), .. })
        ));
        let mut b = f.session(128).unwrap();
        let mut took = 0;
        while took < rest.len() || (f.stats().fills < 20 && took < rest.len() + 16) {
            let t = more[took];
            assert!(
                bits(b.decode(&f, t).unwrap()) == bits(a.decode(&base, t).unwrap()),
                "{policy:?}"
            );
            took += 1;
        }
        eprintln!(
            "{policy:?}, after the failed chunk: {} fills published after {took} decoded tokens",
            f.stats().fills
        );
        assert_eq!(f.live_slots(), 11);
        assert_eq!(
            f.stats().fills,
            20,
            "{policy:?}: after the failed chunk the returned rows' 20 slots (5 rows of 4 layers) filled and published \
             within {took} decoded tokens"
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn the_ceiling_bounds_lent_and_grown_rows() {
    // With lending, `--cache-slots` is the most slots the cache holds, lent and grown rows
    // included. With 6 of its own and a ceiling of 8, it borrows 2 of the 5 rows staging could lend and never grows,
    // though the budget has room; with no ceiling it borrows all 5 and grows a chunk of the other 5 after two checks.
    // The checks between sessions grow nothing past it, and the bits after them are no cache's. With `no_lend`
    // (`--no-lend`) and a ceiling of 11, nothing is lent: the cache keeps its own 6 rows through a staged
    // chunk and a decoded token, with no cache's bits.
    let (g, d) = tiny();
    let rest = tokens(&d, 6, 37);
    let base = plain(&g, &d, 8);
    let want = decoded(&base, &mut base.session(64).unwrap(), &rest);
    for (ceiling, lent, live) in [(Some(8), 2, 8), (None, 5, 16)] {
        let e = Engine::open(
            &g,
            &d,
            &Options {
                budget_every: Some(1),
                ..lending(ceiling, CachePolicyKind::Adaptive, None)
            },
        )
        .unwrap();
        let mut s = e.session(64).unwrap();
        assert!(
            decoded(&e, &mut s, &rest) == want,
            "{ceiling:?}: the logits differ"
        );
        let st = e.stats();
        assert_eq!(
            (st.lent_slots, e.live_slots()),
            (lent, live),
            "{ceiling:?}: {st:?}"
        );
        let grows = u64::from(ceiling.is_none());
        assert_eq!(st.grows, grows, "{ceiling:?}: {st:?}");
        for _ in 0..4 {
            s.reset(&e).unwrap();
            drop(e.session(64).unwrap());
        }
        assert!(
            decoded(&e, &mut s, &rest) == want,
            "{ceiling:?}: the logits differ after the resets"
        );
        assert_eq!(
            (e.stats().grows, e.live_slots()),
            (grows, live),
            "{ceiling:?}: no grow past the ceiling between sessions"
        );
    }
    let e = Engine::open(
        &g,
        &d,
        &Options {
            no_lend: true,
            ..lending(Some(11), CachePolicyKind::Adaptive, None)
        },
    )
    .unwrap();
    let state = || (e.cache_slots(), e.stats().lent_slots, e.live_slots());
    assert_eq!(state(), (6, 0, 6), "no_lend: nothing lent");
    let bits = |l: Vec<f32>| l.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let prompt = tokens(&d, 8, 211);
    let (mut a, mut s) = (base.session(64).unwrap(), e.session(64).unwrap());
    assert!(
        bits(s.prefill(&e, &prompt).unwrap()) == bits(a.prefill(&base, &prompt).unwrap()),
        "no_lend: the logits differ"
    );
    assert_eq!(e.stats().staged_steps, 1, "no_lend: the chunk was staged");
    assert_eq!(state(), (6, 0, 6), "no_lend: after the staged chunk");
    assert!(
        bits(s.decode(&e, rest[0]).unwrap()) == bits(a.decode(&base, rest[0]).unwrap()),
        "no_lend: the logits differ"
    );
    assert_eq!(state(), (6, 0, 6), "no_lend: after a decoded token");
}
