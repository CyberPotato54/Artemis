//! Snapshots in host RAM on the GPU, with the tiny model: a restore gives
//! an unbroken run's bits, whatever dirtied the KV in between; a copy still in flight is waited for; a pool too small
//! skips; a failed write leaves the session unusable; and parked snapshots come back across a reload.
use engine::{
    Engine, Error, Marks, Options, PoolLimit, Session, SessionState, Sizes, Snapshots, Source,
    Taken, Unparked,
};
use gguf::Gguf;
use kernels::dispatch::WHERE_CPU;
use qwen36::{dims::Dims, testing::tiny_model};

fn tiny() -> (Gguf, Dims) {
    (Gguf::from_bytes(tiny_model(1)).unwrap(), Dims::tiny())
}

/// 40 tokens of the tiny model's vocabulary.
fn tokens(d: &Dims) -> Vec<u32> {
    (0..40u32)
        .map(|i| (i * 211 + 17) % d.n_vocab as u32)
        .collect()
}

/// `n` tokens whose first `shared` are `tokens`' and whose next one differs from it.
fn diverging(d: &Dims, shared: usize, n: usize) -> Vec<u32> {
    let mut t = tokens(d);
    t.truncate(shared);
    t.extend((shared..n).map(|i| (i as u32 * 97 + 3) % d.n_vocab as u32));
    assert!(n > shared && t[shared] != tokens(d)[shared]);
    t
}

fn open(g: &Gguf, d: &Dims, slots: usize, chunk: usize) -> Engine {
    Engine::open(
        g,
        d,
        &Options {
            cache_slots: Some(slots),
            prefill_chunk: Some(chunk),
            ..Options::default()
        },
    )
    .unwrap()
}

fn limit(bytes: u64) -> PoolLimit {
    PoolLimit {
        bytes,
        requested: bytes,
        cut: false,
    }
}

fn snaps(e: &Engine) -> Snapshots {
    Snapshots::new(e, limit(64 << 20))
}

/// Two values' f32s are the same bits (a +0 and a -0 differ, a NaN equals itself).
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The tokens decode takes after a prefill, five and then five more.
const DECODE: [u32; 5] = [101, 7, 300, 42, 255];
const DECODE_MORE: [u32; 5] = [88, 9, 400, 151, 23];

/// What a run recorded: the logits of its last prefill position (when it prefilled) and of each decode step, and its
/// final state.
struct Run {
    logits: Vec<Vec<f32>>,
    state: SessionState,
}

/// Decodes `more` and reads the state.
fn finish(e: &Engine, s: &mut Session, mut logits: Vec<Vec<f32>>, more: &[u32]) -> Run {
    for &t in more {
        logits.push(s.decode(e, t).unwrap());
    }
    Run {
        logits,
        state: s.state(e).unwrap(),
    }
}

/// One session prefilling all of `toks`, then decoding five more.
fn unbroken(e: &Engine, toks: &[u32]) -> Run {
    let mut s = e.session(64).unwrap();
    let last = s.prefill(e, toks).unwrap();
    finish(e, &mut s, vec![last], &DECODE)
}

fn assert_same(got: &Run, want: &Run, what: &str) {
    assert_eq!(got.logits.len(), want.logits.len(), "{what}: logits count");
    for (i, (a, b)) in got.logits.iter().zip(&want.logits).enumerate() {
        assert!(same_bits(a, b), "{what}: logits {i} differ");
    }
    assert_eq!(got.state.layers.len(), want.state.layers.len());
    for (l, (a, b)) in got.state.layers.iter().zip(&want.state.layers).enumerate() {
        assert!(same_bits(&a.0, &b.0), "{what}: layer {l}'s first buffer");
        assert!(same_bits(&a.1, &b.1), "{what}: layer {l}'s second buffer");
    }
}

fn take(e: &Engine, sn: &mut Snapshots, s: &mut Session, marks: Marks) -> Taken {
    sn.take(e, s, marks).unwrap().expect("a snapshot was taken")
}

/// Restores `id`, prefills `toks` from where it ends and decodes five.
fn resume(
    e: &Engine,
    sn: &mut Snapshots,
    s: &mut Session,
    id: engine::SnapId,
    toks: &[u32],
) -> Run {
    sn.restore(e, s, id).unwrap();
    let last = s.prefill(e, &toks[s.pos()..]).unwrap();
    finish(e, s, vec![last], &DECODE)
}

/// Takes a snapshot at `at`, dirties the session with `dirt` prefilled from 0, restores and runs on to the end.
fn dirty_and_restore(e: &Engine, toks: &[u32], at: usize, dirt: &[u32]) -> Run {
    let mut sn = snaps(e);
    let mut s = e.session(64).unwrap();
    s.prefill(e, &toks[..at]).unwrap();
    let t = take(e, &mut sn, &mut s, Marks::PROMPT_END);
    s.reset(e).unwrap();
    s.prefill(e, dirt).unwrap();
    resume(e, &mut sn, &mut s, t.id, toks)
}

#[test]
#[ignore = "needs the GPU"]
fn a_restore_in_mid_chunk_gives_an_unbroken_runs_bits() {
    // Chunks are 16 tokens: the snapshot at 17 lands one token into the second chunk.
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 16);
    let toks = tokens(&d);
    let want = unbroken(&e, &toks);
    let mut sn = snaps(&e);
    let mut s = e.session(64).unwrap();
    s.prefill(&e, &toks[..17]).unwrap();
    let t = take(&e, &mut sn, &mut s, Marks::INTERVAL);
    // 13 unrelated tokens on top (the state and KV from 17 on change), then back.
    s.prefill(&e, &diverging(&d, 0, 30)[17..]).unwrap();
    let got = resume(&e, &mut sn, &mut s, t.id, &toks);
    assert_same(&got, &want, "mid-chunk restore");
}

#[test]
#[ignore = "needs the GPU"]
fn a_restore_at_a_chunk_boundary_gives_an_unbroken_runs_bits() {
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 16);
    let toks = tokens(&d);
    let want = unbroken(&e, &toks);
    let mut sn = snaps(&e);
    let mut s = e.session(64).unwrap();
    s.prefill(&e, &toks[..16]).unwrap();
    let t = take(&e, &mut sn, &mut s, Marks::INTERVAL);
    s.prefill(&e, &diverging(&d, 0, 29)[16..]).unwrap();
    let got = resume(&e, &mut sn, &mut s, t.id, &toks);
    assert_same(&got, &want, "chunk-boundary restore");
}

#[test]
#[ignore = "needs the GPU"]
fn a_run_dirtied_from_position_zero_and_restored_gives_the_same_bits() {
    // Every KV position below 30 is overwritten by a different prompt before the restore.
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 16);
    let toks = tokens(&d);
    let want = unbroken(&e, &toks);
    let got = dirty_and_restore(&e, &toks, 30, &diverging(&d, 0, 40));
    assert_same(&got, &want, "restore after a dirtying from 0");
}

#[test]
#[ignore = "needs the GPU"]
fn a_restore_copies_only_the_kv_past_the_common_prefix() {
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 16);
    let toks = tokens(&d);
    let want = unbroken(&e, &toks);
    let state = Sizes::of(&d).state;
    let mut sn = snaps(&e);
    let mut s = e.session(64).unwrap();
    s.prefill(&e, &toks[..30]).unwrap();
    let t = take(&e, &mut sn, &mut s, Marks::PROMPT_END);
    assert_eq!(t.bytes, state + 30 * 256, "the first snapshot's bytes");
    s.reset(&e).unwrap();
    // 40 tokens sharing the first 10: positions 10 to 30 differ.
    s.prefill(&e, &diverging(&d, 10, 40)).unwrap();
    let r = sn.restore(&e, &mut s, t.id).unwrap();
    assert_eq!((r.kv_from, r.bytes), (10, state + 20 * 256));
    let last = s.prefill(&e, &toks[30..]).unwrap();
    assert_same(&finish(&e, &mut s, vec![last], &DECODE), &want, "first");
    // At once again: the session's KV still starts with the snapshot's 30 tokens.
    let r = sn.restore(&e, &mut s, t.id).unwrap();
    assert_eq!((r.kv_from, r.bytes), (30, state));
    let last = s.prefill(&e, &toks[30..]).unwrap();
    assert_same(&finish(&e, &mut s, vec![last], &DECODE), &want, "second");
    let st = sn.stats();
    assert_eq!((st.restores, st.restore_bytes), (2, 2 * state + 20 * 256));
}

#[test]
#[ignore = "needs the GPU"]
fn a_restore_overwrites_every_position_the_poison_hit() {
    // `artemis warm` poisons the live KV between the dirtying and the restore, so a skipped copy cannot pass by luck.
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 16);
    let toks = tokens(&d);
    let want = unbroken(&e, &toks);
    let mut sn = snaps(&e);
    let mut s = e.session(64).unwrap();
    s.prefill(&e, &toks[..30]).unwrap();
    let t = take(&e, &mut sn, &mut s, Marks::PROMPT_END);
    s.reset(&e).unwrap();
    // 12 tokens sharing the first 10, from 0; positions 12 to 30 still hold the first run's bytes until poisoned.
    s.prefill(&e, &diverging(&d, 10, 12)).unwrap();
    s.poison_kv(&e, 12, 30).unwrap();
    let got = resume(&e, &mut sn, &mut s, t.id, &toks);
    assert_same(&got, &want, "restore over poison");
}

#[test]
#[ignore = "needs the GPU"]
fn a_snapshot_whose_parents_state_was_dropped_still_restores() {
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 16);
    let toks = tokens(&d);
    let want = unbroken(&e, &toks);
    let mut sn = snaps(&e);
    let mut s = e.session(64).unwrap();
    s.prefill(&e, &toks[..16]).unwrap();
    let p = take(&e, &mut sn, &mut s, Marks::INTERVAL);
    s.prefill(&e, &toks[16..30]).unwrap();
    let c = take(&e, &mut sn, &mut s, Marks::PROMPT_END);
    assert_eq!(sn.tree().node(c.id).unwrap().parent, Some(p.id));
    sn.drop_state(&e, p.id).unwrap();
    assert!(!sn.tree().node(p.id).unwrap().has_state);
    s.reset(&e).unwrap();
    s.prefill(&e, &diverging(&d, 0, 40)).unwrap();
    let got = resume(&e, &mut sn, &mut s, c.id, &toks);
    assert_same(&got, &want, "restore of a child of a stateless node");
    assert!(
        matches!(sn.restore(&e, &mut s, p.id), Err(Error::NoSuchSnapshot { id }) if id == p.id.0),
        "the stateless parent has nothing to restore"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_snapshot_built_by_decode_restores() {
    // 30 prefilled tokens and 5 decoded make the snapshot's 35; after the restore decode goes on with five more.
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 16);
    let toks = tokens(&d);
    let want = {
        let mut s = e.session(64).unwrap();
        s.prefill(&e, &toks[..30]).unwrap();
        for &t in &DECODE {
            s.decode(&e, t).unwrap();
        }
        finish(&e, &mut s, Vec::new(), &DECODE_MORE)
    };
    let mut sn = snaps(&e);
    let mut s = e.session(64).unwrap();
    s.prefill(&e, &toks[..30]).unwrap();
    for &t in &DECODE {
        s.decode(&e, t).unwrap();
    }
    let t = take(&e, &mut sn, &mut s, Marks::REPLY_END);
    s.reset(&e).unwrap();
    s.prefill(&e, &diverging(&d, 0, 40)).unwrap();
    sn.restore(&e, &mut s, t.id).unwrap();
    assert_eq!(s.pos(), 35);
    let got = finish(&e, &mut s, Vec::new(), &DECODE_MORE);
    assert_same(&got, &want, "decode after a restore of a decoded snapshot");
}

#[test]
#[ignore = "needs the GPU"]
fn a_copy_still_in_flight_is_waited_for_before_its_kv_is_overwritten() {
    // Every KV copy fills 1 GiB of VRAM first, so it lands late. The dirtying prompt writes from
    // position 0 at once, with no `settle`: the session must wait for the copy first (the counter, which does not
    // depend on timing), or the copy reads the new prompt's KV (the bits).
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 16);
    let toks = tokens(&d);
    let want = unbroken(&e, &toks);
    let mut sn = snaps(&e);
    sn.delay_copies(&e).unwrap();
    let mut s = e.session(64).unwrap();
    s.prefill(&e, &toks[..30]).unwrap();
    let t = take(&e, &mut sn, &mut s, Marks::PROMPT_END);
    assert_eq!(s.copy_waits(), 0, "nothing has written yet");
    s.reset(&e).unwrap();
    s.prefill(&e, &diverging(&d, 0, 40)).unwrap();
    assert!(
        s.copy_waits() >= 1,
        "the write below the pending copy's end did not wait for it"
    );
    let got = resume(&e, &mut sn, &mut s, t.id, &toks);
    assert_same(&got, &want, "restore of a delayed copy");
}

#[test]
#[ignore = "needs the GPU"]
fn a_pool_too_small_skips_the_snapshot() {
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 16);
    let toks = tokens(&d);
    let state = Sizes::of(&d).state;
    let mut sn = Snapshots::new(&e, limit(state - 1));
    let mut s = e.session(64).unwrap();
    s.prefill(&e, &toks[..20]).unwrap();
    assert!(sn.take(&e, &mut s, Marks::INTERVAL).unwrap().is_none());
    let st = sn.stats();
    assert_eq!(
        (st.skipped, st.snapshots_held, st.pool_bytes_used),
        (1, 0, 0)
    );
    s.prefill(&e, &toks[20..]).unwrap();
    for &t in &DECODE {
        s.decode(&e, t).unwrap();
    }
    assert_eq!(s.pos(), 45);
}

#[test]
#[ignore = "needs the GPU"]
fn a_pool_with_no_run_left_demotes_more_states_instead_of_skipping() {
    // The tree counts exact bytes, the pool hands out 256-byte runs: a one-token snapshot's KV is 128 B a buffer in the
    // tree and 256 in the pool. A pool of two states and 768 B fits two such snapshots in the tree (2 × (state + 256))
    // but not in the pool, and its one chunk is already open. The second take finds the tree with room and the pool
    // with no run, so it must demote the first snapshot (its slices with it), not skip.
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 16);
    let toks = tokens(&d);
    let state = Sizes::of(&d).state;
    let mut sn = Snapshots::new(&e, limit(2 * state + 768));
    let mut s = e.session(64).unwrap();
    s.prefill(&e, &toks[..1]).unwrap();
    let a = take(&e, &mut sn, &mut s, Marks::INTERVAL);
    s.reset(&e).unwrap();
    s.prefill(&e, &diverging(&d, 0, 1)).unwrap();
    let b = take(&e, &mut sn, &mut s, Marks::INTERVAL);
    let st = sn.stats();
    assert_eq!(
        (st.skipped, st.snapshots_held, st.evicted_unprotected),
        (0, 1, 1)
    );
    assert!(sn.tree().node(a.id).is_none(), "the first was demoted");
    // The survivor's memory is sound: it restores.
    s.reset(&e).unwrap();
    sn.restore(&e, &mut s, b.id).unwrap();
    assert_eq!(s.pos(), 1);
}

#[test]
#[ignore = "needs the GPU"]
fn a_failed_write_leaves_the_session_unusable_until_reset() {
    let (g, d) = tiny();
    let e = open(&g, &d, 0, 16);
    let toks = tokens(&d);
    let mut sn = snaps(&e);
    let mut s = e.session(64).unwrap();
    s.prefill(&e, &toks[..10]).unwrap();
    // A validation error changes nothing.
    assert!(matches!(
        s.prefill(&e, &[d.n_vocab as u32]),
        Err(Error::TokenOutOfRange { .. })
    ));
    assert!(s.usable());
    assert_eq!(s.kv_tokens(), &toks[..10]);
    // An expert on the refused CPU path fails the step past validation.
    let saved: Vec<(u64, u32)> = (0..d.n_expert)
        .map(|x| e.slot_entry(2, x).unwrap())
        .collect();
    for x in 0..d.n_expert {
        e.set_slot_entry(2, x, 0, WHERE_CPU).unwrap();
    }
    assert!(matches!(
        s.prefill(&e, &toks[10..18]),
        Err(Error::PrefillExpertOnCpu { .. })
    ));
    assert!(!s.usable());
    assert_eq!(s.kv_tokens().len(), 10);
    let c = sn.choose(&toks, &s);
    assert_eq!(c.source, Source::None, "an unusable session is no Live");
    assert!(sn.take(&e, &mut s, Marks::INTERVAL).unwrap().is_none());
    assert_eq!(sn.stats().skipped, 0, "an unusable session is not a skip");
    for (x, &(addr, loc)) in saved.iter().enumerate() {
        e.set_slot_entry(2, x, addr, loc).unwrap();
    }
    s.reset(&e).unwrap();
    assert!(s.usable());
    s.prefill(&e, &toks[..8]).unwrap();
}

#[test]
#[ignore = "needs the GPU"]
fn parked_snapshots_restore_after_a_reload() {
    let (g, d) = tiny();
    let toks = tokens(&d);
    let e1 = open(&g, &d, 0, 16);
    let want = unbroken(&e1, &toks);
    let mut sn = snaps(&e1);
    let mut s = e1.session(64).unwrap();
    s.prefill(&e1, &toks[..30]).unwrap();
    take(&e1, &mut sn, &mut s, Marks::PROMPT_END);
    // At once: the fence settles and makes the host read visible itself. The engine is gone before the pool is read, as
    // `Loaded::unload` does it: the pool's buffers hold the device.
    sn.fence_for_park(&e1).unwrap();
    drop(s);
    e1.unload();
    let parked = sn.into_parked();
    assert_eq!(parked.len(), 1);
    assert!(parked.bytes() > 0 && !parked.is_empty());
    let e2 = open(&g, &d, 0, 16);
    let (mut sn2, un) = Snapshots::unpark(&e2, limit(64 << 20), parked).unwrap();
    assert_eq!(
        un,
        Unparked {
            restored: 1,
            dropped: 0
        }
    );
    let id = sn2.tree().ids()[0];
    let mut s2 = e2.session(64).unwrap();
    let got = resume(&e2, &mut sn2, &mut s2, id, &toks);
    assert_same(&got, &want, "restore after a reload");
}

#[test]
#[ignore = "needs the GPU"]
fn parking_keeps_the_lru_and_taken_order_and_a_stateless_ancestor() {
    let (g, d) = tiny();
    let toks = tokens(&d);
    let e1 = open(&g, &d, 0, 16);
    let want = unbroken(&e1, &toks);
    let mut sn = snaps(&e1);
    let mut s = e1.session(64).unwrap();
    s.prefill(&e1, &toks[..16]).unwrap();
    let p = take(&e1, &mut sn, &mut s, Marks::INTERVAL);
    s.prefill(&e1, &toks[16..30]).unwrap();
    let c = take(&e1, &mut sn, &mut s, Marks::USER_QUERY);
    // A second conversation's snapshot, then S is used again.
    s.reset(&e1).unwrap();
    s.prefill(&e1, &diverging(&d, 0, 20)).unwrap();
    let q = take(&e1, &mut sn, &mut s, Marks::USER_QUERY);
    sn.restore(&e1, &mut s, c.id).unwrap();
    sn.drop_state(&e1, p.id).unwrap();
    // The nodes' positions (16 is P, 30 is S, 20 is Q) from the oldest to the newest by each tick.
    let ranks = |t: &engine::Tree| {
        let by = |key: fn(&engine::Node) -> u64| {
            let mut v: Vec<(u64, usize)> = t
                .ids()
                .iter()
                .map(|&id| {
                    let n = t.node(id).unwrap();
                    (key(n), n.pos())
                })
                .collect();
            v.sort_unstable();
            v.into_iter().map(|(_, pos)| pos).collect::<Vec<_>>()
        };
        (by(|n| n.taken), by(|n| n.last_used))
    };
    let before = ranks(sn.tree());
    assert_eq!(before, (vec![16, 30, 20], vec![16, 20, 30]));
    assert_eq!(sn.tree().node(q.id).unwrap().pos(), 20);
    let parked = sn.park(&e1).unwrap();
    assert_eq!(parked.len(), 3);
    drop(s);
    e1.unload();
    let e2 = open(&g, &d, 0, 16);
    let (mut sn2, un) = Snapshots::unpark(&e2, limit(64 << 20), parked).unwrap();
    assert_eq!(
        un,
        Unparked {
            restored: 3,
            dropped: 0
        }
    );
    let t = sn2.tree();
    assert_eq!(t.len(), 3);
    assert_eq!(ranks(t), before);
    let by_pos = |pos: usize| {
        t.ids()
            .into_iter()
            .find(|&id| t.node(id).unwrap().pos() == pos)
            .unwrap()
    };
    assert!(!t.node(by_pos(16)).unwrap().has_state, "P stays stateless");
    assert_eq!(t.node(by_pos(30)).unwrap().parent, Some(by_pos(16)));
    let id = by_pos(30);
    let mut s2 = e2.session(64).unwrap();
    let got = resume(&e2, &mut sn2, &mut s2, id, &toks);
    assert_same(&got, &want, "restore of a parked child of a stateless node");
}

/// The link measurement runs on this GPU and gives six positive, finite rates. A check that the code runs: 1 MiB is
/// too small for the figures to mean anything (`artemis snapshot-times` measures 256 MiB).
#[test]
#[ignore = "needs the GPU"]
fn link_rates_smoke() {
    let r = engine::link_rates(None, 1 << 20).unwrap();
    for (what, rate) in [
        ("cached d2h", r.cached_d2h),
        ("cached h2d", r.cached_h2d),
        ("pinned d2h", r.pinned_d2h),
        ("pinned h2d", r.pinned_h2d),
        ("cached cpu read", r.cached_cpu_read),
        ("pinned cpu read", r.pinned_cpu_read),
    ] {
        assert!(rate.is_finite() && rate > 0.0, "{what} is {rate} GB/s");
    }
}
