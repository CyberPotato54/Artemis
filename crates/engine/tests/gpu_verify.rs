//! `Session::verify` and `commit` on the GPU, with the tiny model.
//!
//! A verify step runs decode's kernels over `[x, d1 .. dk]` in one pass, and its DeltaNet layers run `gdn_seq` where
//! decode runs `gdn_step`. The two are not the same arithmetic, so a verify row is checked against decode's the way
//! prefill is (`prefill_real.rs`): the rows' relative L2 must average at most `FORCED_MEAN` and each stay within
//! `FORCED_WORST`, and the DeltaNet and conv state, and the KV, within `STATE`. Bit equality is reported
//! (`kernels::testing::identity`), never asserted, except that the replay reproduces the verify's own
//! arithmetic (`the_replay_reproduces_the_verifys_own_state`).
//!
//! Rows are compared where the verify routed as decode did. A row whose routes differ is a router near-tie: it is
//! reported, not judged, and neither are the rows after it, which ran from a different state.
use engine::{Engine, Error, Marks, Options, PoolLimit, Session, SessionState, Snapshots};
use gguf::Gguf;
use kernels::testing::identity;
use qwen36::{dims::Dims, ops::rel_l2, testing::tiny_model};

/// `prefill_real.rs`'s bounds, because the difference is the same kind (`gdn_seq` against `gdn_step`).
const FORCED_MEAN: f64 = 1e-3;
const FORCED_WORST: f64 = 5e-2;
const STATE: f64 = 1e-3;
/// Each session's context, and the engine's.
const CTX: usize = 64;

fn tiny() -> (Gguf, Dims) {
    (Gguf::from_bytes(tiny_model(1)).unwrap(), Dims::tiny())
}

fn open_with(g: &Gguf, d: &Dims, chunk: Option<usize>) -> Engine {
    Engine::open(
        g,
        d,
        &Options {
            ctx: CTX,
            cache_slots: Some(0),
            prefill_chunk: chunk,
            speculation: true,
            ..Options::default()
        },
    )
    .unwrap()
}

fn open(g: &Gguf, d: &Dims) -> Engine {
    open_with(g, d, None)
}

/// 10 tokens of the tiny model's vocabulary: the prompt every run starts from.
fn prompt(d: &Dims) -> Vec<u32> {
    (0..10u32)
        .map(|i| (i * 211 + 17) % d.n_vocab as u32)
        .collect()
}

/// `n` tokens to decode or verify after it (16 at most is what a verify takes).
fn run_of(d: &Dims, n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|i| (i * 97 + 3) % d.n_vocab as u32)
        .collect()
}

/// A session of `CTX` positions that has prefilled the prompt.
fn started(e: &Engine, d: &Dims) -> Session {
    let mut s = e.session(CTX).unwrap();
    s.prefill(e, &prompt(d)).unwrap();
    s
}

/// Two values' f32s are the same bits (a +0 and a -0 differ, a NaN equals itself).
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The worst relative L2 over every layer's two buffers (a NaN wins).
fn state_err(a: &SessionState, b: &SessionState) -> f64 {
    assert_eq!(a.layers.len(), b.layers.len());
    a.layers
        .iter()
        .zip(&b.layers)
        .flat_map(|(x, y)| [rel_l2(&x.0, &y.0), rel_l2(&x.1, &y.1)])
        .fold(0f64, |m, e| if e.is_nan() || e > m { e } else { m })
}

fn same_state(a: &SessionState, b: &SessionState) -> bool {
    a.layers.len() == b.layers.len()
        && a.layers
            .iter()
            .zip(&b.layers)
            .all(|(x, y)| same_bits(&x.0, &y.0) && same_bits(&x.1, &y.1))
}

/// What decoding a run one token at a time gave: each token's logits and routed experts (`[layer][used]` flattened).
struct Plain {
    logits: Vec<Vec<f32>>,
    routes: Vec<Vec<usize>>,
}

fn decode_run(e: &Engine, s: &mut Session, toks: &[u32]) -> Plain {
    let (mut logits, mut routes) = (Vec::new(), Vec::new());
    for &t in toks {
        logits.push(s.decode(e, t).unwrap());
        routes.push(e.last_routes());
    }
    Plain { logits, routes }
}

/// The routed experts of each row of the verify that just ran (`[layer][used]` flattened), from its routing log.
fn verify_routes(e: &Engine, d: &Dims, t: usize) -> Vec<Vec<usize>> {
    let (k, ids) = (d.n_expert_used, e.last_prefill_routes());
    assert_eq!(ids.len(), d.n_layer * t * k, "the log of a verify of {t}");
    (0..t)
        .map(|i| {
            (0..d.n_layer)
                .flat_map(|l| ids[(l * t + i) * k..(l * t + i + 1) * k].iter().copied())
                .collect()
        })
        .collect()
}

/// Whether two rows routed to the same experts in every layer.
fn same_routes(d: &Dims, a: &[usize], b: &[usize]) -> bool {
    let k = d.n_expert_used;
    a.len() == b.len()
        && a.chunks(k).zip(b.chunks(k)).all(|(x, y)| {
            let (mut x, mut y) = (x.to_vec(), y.to_vec());
            x.sort();
            y.sort();
            x == y
        })
}

/// Which leading rows routed as decode did: the rows to judge.
fn agreeing(d: &Dims, got: &[Vec<usize>], want: &[Vec<usize>]) -> usize {
    got.iter()
        .zip(want)
        .take_while(|(a, b)| same_routes(d, a, b))
        .count()
}

/// The bounds on rows: the first `judged` of them. Reports the rest, and every judged row's bits.
fn judge_rows(what: &str, got: &[Vec<f32>], want: &[Vec<f32>], judged: usize) {
    assert_eq!(got.len(), want.len(), "{what}: a row for every token");
    if judged < got.len() {
        eprintln!(
            "{what}: row {judged} routed differently from decode (a router near-tie): it and the {} after it are \
             reported, not judged",
            got.len() - judged - 1
        );
    }
    assert!(judged >= 1, "{what}: not even row 0 routed as decode did");
    let errs: Vec<f64> = got[..judged]
        .iter()
        .zip(&want[..judged])
        .map(|(a, b)| rel_l2(a, b))
        .collect();
    let mean = errs.iter().sum::<f64>() / errs.len() as f64;
    let worst = errs
        .iter()
        .copied()
        .fold(0f64, |m, e| if e.is_nan() || e > m { e } else { m });
    let equal = got[..judged]
        .iter()
        .zip(&want[..judged])
        .all(|(a, b)| same_bits(a, b));
    eprintln!("{what}: {judged} rows judged; mean {mean:.3e}, worst {worst:.3e}; bits {equal}");
    identity("verify_rows", what, equal);
    assert!(
        mean <= FORCED_MEAN,
        "{what}: the rows differ from decode's by {mean:.3e} on average, over {FORCED_MEAN:e}"
    );
    assert!(
        worst <= FORCED_WORST,
        "{what}: a row differs from decode's by {worst:.3e}, over {FORCED_WORST:e}"
    );
}

/// The bound on a state, with its bits reported.
fn judge_state(what: &str, got: &SessionState, want: &SessionState) {
    let err = state_err(got, want);
    eprintln!(
        "{what}: state differs by {err:.3e}; bits {}",
        same_state(got, want)
    );
    identity("verify_state", what, same_state(got, want));
    assert!(
        err <= STATE,
        "{what}: the state differs by {err:.3e}, over {STATE:e}"
    );
}

/// Runs a verify of `toks` and reads every row.
fn verified(e: &Engine, s: &mut Session, toks: &[u32]) -> Vec<Vec<f32>> {
    s.verify(e, toks).unwrap();
    assert!(s.pending());
    (0..toks.len()).map(|i| s.row(e, i).unwrap()).collect()
}

#[test]
#[ignore = "needs the GPU"]
fn verify_rows_match_decodes_logits() {
    let (g, d) = tiny();
    let e = open(&g, &d);
    for len in [1, 2, 5, 16] {
        let toks = run_of(&d, len);
        let mut plain_s = started(&e, &d);
        let plain = decode_run(&e, &mut plain_s, &toks);
        let mut s = started(&e, &d);
        let start = s.pos();
        s.verify(&e, &toks).unwrap();
        let routes = verify_routes(&e, &d, len);
        assert!(s.pending());
        // The session holds every token, as if all were kept.
        assert_eq!(s.pos(), start + len);
        assert_eq!(&s.tokens()[start..], &toks[..]);
        assert_eq!(&s.kv_tokens()[start..], &toks[..]);
        let rows: Vec<Vec<f32>> = (0..len).map(|i| s.row(&e, i).unwrap()).collect();
        assert!(rows.iter().all(|r| r.len() == d.n_vocab));
        let judged = agreeing(&d, &routes, &plain.routes);
        judge_rows(&format!("a verify of {len}"), &rows, &plain.logits, judged);
        s.commit(&e, len).unwrap();
        assert!(!s.pending());
        assert_eq!(
            (s.pos(), s.tokens(), s.kv_tokens()),
            (plain_s.pos(), plain_s.tokens(), plain_s.kv_tokens())
        );
        if judged == len {
            judge_state(
                &format!("the state after a verify of {len}"),
                &s.state(&e).unwrap(),
                &plain_s.state(&e).unwrap(),
            );
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn commit_keeps_the_state_of_the_kept_tokens() {
    let (g, d) = tiny();
    let e = open(&g, &d);
    let len = 6;
    let toks = run_of(&d, len);
    let next = 211u32;
    let mut judged_kept = 0;
    let (mut got_next, mut want_next) = (Vec::new(), Vec::new());
    for kept in 1..=len {
        let mut plain_s = started(&e, &d);
        let plain = decode_run(&e, &mut plain_s, &toks[..kept]);
        let mut s = started(&e, &d);
        s.verify(&e, &toks).unwrap();
        let routes = verify_routes(&e, &d, len);
        s.commit(&e, kept).unwrap();
        assert!(!s.pending());
        assert_eq!(
            (s.pos(), s.tokens(), s.kv_tokens()),
            (plain_s.pos(), plain_s.tokens(), plain_s.kv_tokens()),
            "kept {kept}"
        );
        if agreeing(&d, &routes[..kept], &plain.routes) < kept {
            eprintln!("kept {kept}: a row routed differently from decode (a near-tie): reported, not judged");
            continue;
        }
        judged_kept += 1;
        judge_state(
            &format!("kept {kept} of {len}"),
            &s.state(&e).unwrap(),
            &plain_s.state(&e).unwrap(),
        );
        // The next decode, from the committed state against the plain one.
        let a = s.decode(&e, next).unwrap();
        let a_routes = e.last_routes();
        let b = plain_s.decode(&e, next).unwrap();
        let b_routes = e.last_routes();
        assert_eq!(s.pos(), plain_s.pos());
        if !same_routes(&d, &a_routes, &b_routes) {
            eprintln!(
                "kept {kept}: the next decode routed differently (a near-tie): reported, not judged"
            );
            continue;
        }
        got_next.push(a);
        want_next.push(b);
    }
    assert!(judged_kept >= 1, "no kept count was judged");
    assert!(!got_next.is_empty(), "no decode after a commit was judged");
    let all = got_next.len();
    judge_rows("the decodes after commits", &got_next, &want_next, all);
}

#[test]
#[ignore = "needs the GPU"]
fn the_replay_reproduces_the_verifys_own_state() {
    // The replay rests on `gdn_seq`'s results not depending on the batch: a verify of 5 committed at 3 and a verify of
    // 3 committed at 3 leave the same DeltaNet and conv state, and the next verify gives the same rows.
    let (g, d) = tiny();
    let e = open(&g, &d);
    let toks = run_of(&d, 5);
    let after = [17u32, 300, 5, 450];
    let mut five = started(&e, &d);
    let rows_five = verified(&e, &mut five, &toks);
    five.commit(&e, 3).unwrap();
    let mut three = started(&e, &d);
    let rows_three = verified(&e, &mut three, &toks[..3]);
    let routes_three = verify_routes(&e, &d, 3);
    three.commit(&e, 3).unwrap();
    // The first three rows are the same in both.
    for i in 0..3 {
        assert!(
            same_bits(&rows_five[i], &rows_three[i]),
            "row {i} of the verify of 5 and of 3"
        );
    }
    assert_eq!((five.pos(), five.tokens()), (three.pos(), three.tokens()));
    let (sf, st) = (five.state(&e).unwrap(), three.state(&e).unwrap());
    for l in (0..d.n_layer).filter(|&l| !d.is_attention(l)) {
        assert!(
            same_bits(&sf.layers[l].0, &st.layers[l].0),
            "layer {l}'s conv state"
        );
        assert!(
            same_bits(&sf.layers[l].1, &st.layers[l].1),
            "layer {l}'s recurrent state"
        );
    }
    // The whole state (every layer's conv and recurrent state, and the KV the two wrote at the kept positions) is the
    // same too, so the next verify's attention reads the same.
    identity(
        "verify_state",
        "the verify of 5 and of 3: the whole session state",
        same_state(&sf, &st),
    );
    let more_five = verified(&e, &mut five, &after);
    let more_three = verified(&e, &mut three, &after);
    for (i, (a, b)) in more_five.iter().zip(&more_three).enumerate() {
        assert!(same_bits(a, b), "row {i} of the next verify");
    }
    five.commit(&e, after.len()).unwrap();
    three.commit(&e, after.len()).unwrap();
    // Against a plain session decoding the 3 tokens: within the bounds, bits reported.
    let mut plain_s = started(&e, &d);
    let plain = decode_run(&e, &mut plain_s, &toks[..3]);
    let judged = agreeing(&d, &routes_three, &plain.routes);
    judge_rows("the replay's rows", &rows_three, &plain.logits, judged);
    if judged == 3 {
        judge_state("the replayed state", &st, &plain_s.state(&e).unwrap());
    }
}

#[test]
#[ignore = "needs the GPU"]
fn a_snapshot_after_commit_restores_the_plain_state() {
    // The reply-end snapshot right after a partial accept holds the committed tokens, and restoring it
    // gives back the state the committed run went on from.
    let (g, d) = tiny();
    let e = open(&g, &d);
    let toks = run_of(&d, 5);
    let next = 123u32;
    let mut sn = snapshots(&e);
    let mut s = started(&e, &d);
    s.verify(&e, &toks).unwrap();
    let routes = verify_routes(&e, &d, 5);
    s.commit(&e, 3).unwrap();
    let committed: Vec<u32> = prompt(&d)
        .into_iter()
        .chain(toks[..3].iter().copied())
        .collect();
    assert_eq!(s.tokens(), &committed[..]);
    let id = sn
        .take(&e, &mut s, Marks::PROMPT_END)
        .unwrap()
        .expect("a snapshot of the committed tokens")
        .id;
    assert_eq!(sn.tree().node(id).unwrap().tokens, committed);
    // The run goes on from here; then something else dirties the session, and its KV over the rejected drafts'
    // positions is poisoned.
    let want = s.decode(&e, next).unwrap();
    s.reset(&e).unwrap();
    let other: Vec<u32> = (0..40u32)
        .map(|i| (i * 31 + 5) % d.n_vocab as u32)
        .collect();
    s.prefill(&e, &other).unwrap();
    s.poison_kv(&e, committed.len(), committed.len() + 4)
        .unwrap();
    sn.restore(&e, &mut s, id).unwrap();
    assert_eq!(
        (s.pos(), s.tokens(), s.kv_tokens()),
        (committed.len(), &committed[..], &committed[..])
    );
    let got = s.decode(&e, next).unwrap();
    assert!(
        same_bits(&got, &want),
        "the restored session's next decode differs from the run it was taken from"
    );
    // And that run is the plain one, within the bounds (where the routes agree).
    let got_routes = e.last_routes();
    let mut plain_s = started(&e, &d);
    let plain = decode_run(&e, &mut plain_s, &toks[..3]);
    let want_next = plain_s.decode(&e, next).unwrap();
    let want_routes = e.last_routes();
    if agreeing(&d, &routes[..3], &plain.routes) == 3 && same_routes(&d, &got_routes, &want_routes)
    {
        judge_state(
            "the restored state",
            &s.state(&e).unwrap(),
            &plain_s.state(&e).unwrap(),
        );
        judge_rows("the decode after the restore", &[got], &[want_next], 1);
    } else {
        eprintln!("a row routed differently from decode (a near-tie): reported, not judged");
    }
}

#[test]
#[ignore = "needs the GPU"]
fn a_failed_verify_cuts_the_kv_record_to_the_step_start() {
    // NaN in the 8 positions every row's attention reads fails the step at its first row.
    let (g, d) = tiny();
    let e = open(&g, &d);
    let mut s = started(&e, &d);
    let start = s.pos();
    s.poison_kv(&e, start - 8, start).unwrap();
    s.verify(&e, &run_of(&d, 3)).unwrap();
    assert!(s.pending());
    match s.row(&e, 0) {
        Err(Error::NonFinite { pos, .. }) => assert_eq!(pos, start),
        other => panic!("expected NonFinite, got {other:?}"),
    }
    assert_eq!(s.kv_tokens().len(), start);
    assert_eq!((s.pos(), s.tokens().len()), (start, start));
    assert!(!s.usable());
    assert!(!s.pending());
    // The next request starts from a reset.
    s.reset(&e).unwrap();
    assert!(s.usable());
    assert_eq!(s.decode(&e, 5).unwrap().len(), d.n_vocab);
}

#[test]
#[ignore = "needs the GPU"]
fn a_failed_commit_leaves_the_session_unusable_and_clear() {
    // The replay's failure follows decode's write rule.
    let (g, d) = tiny();
    let e = open(&g, &d);
    let mut s = started(&e, &d);
    let start = s.pos();
    s.fail_next_commit();
    s.verify(&e, &run_of(&d, 5)).unwrap();
    assert!(s.pending());
    let err = s.commit(&e, 2).unwrap_err();
    assert!(matches!(err, Error::Step { .. }), "{err:?}");
    assert_eq!(s.kv_tokens().len(), start);
    assert_eq!((s.pos(), s.tokens().len()), (start, start));
    assert!(!s.usable());
    assert!(!s.pending());
    // The hook fired once: after a reset, the same steps work.
    s.reset(&e).unwrap();
    assert!(s.usable());
    assert_eq!(s.decode(&e, 5).unwrap().len(), d.n_vocab);
    s.verify(&e, &run_of(&d, 3)).unwrap();
    s.commit(&e, 2).unwrap();
    assert!(!s.pending());
    assert_eq!(s.pos(), 3);
}

#[test]
#[ignore = "needs the GPU"]
fn rows_are_copied_only_when_read() {
    let (g, d) = tiny();
    let e = open(&g, &d);
    let mut s = started(&e, &d);
    assert_eq!(s.rows_read(), 0);
    s.verify(&e, &run_of(&d, 5)).unwrap();
    assert_eq!(s.rows_read(), 0, "a verify reads no row");
    s.row(&e, 0).unwrap();
    s.row(&e, 1).unwrap();
    s.commit(&e, 2).unwrap();
    assert_eq!(s.rows_read(), 2);
}

#[test]
#[ignore = "needs the GPU"]
fn verify_refuses_0_17_and_past_the_context() {
    let (g, d) = tiny();
    let e = open(&g, &d);
    let mut s = started(&e, &d);
    let start = s.pos();
    let unchanged = |s: &Session| {
        assert_eq!(
            (s.pos(), s.tokens().len(), s.kv_tokens().len()),
            (start, start, start)
        );
        assert!(s.usable() && !s.pending());
    };
    match s.verify(&e, &[]) {
        Err(Error::VerifyShape {
            tokens: 0,
            most: 16,
        }) => {}
        other => panic!("0 tokens: {other:?}"),
    }
    unchanged(&s);
    match s.verify(&e, &run_of(&d, 17)) {
        Err(Error::VerifyShape {
            tokens: 17,
            most: 16,
        }) => {}
        other => panic!("17 tokens: {other:?}"),
    }
    unchanged(&s);
    match s.verify(&e, &[1, d.n_vocab as u32]) {
        Err(Error::TokenOutOfRange { token, .. }) => assert_eq!(token, d.n_vocab as u32),
        other => panic!("a token past the vocabulary: {other:?}"),
    }
    unchanged(&s);
    // 54 + 11 > 64: past the context. Exactly to the context is fine.
    s.reset(&e).unwrap();
    s.prefill(&e, &(0..54u32).map(|i| i % 100).collect::<Vec<_>>())
        .unwrap();
    let at = s.pos();
    match s.verify(&e, &run_of(&d, 11)) {
        Err(Error::PrefillTooLong {
            pos,
            tokens: 11,
            ctx,
        }) => assert_eq!((pos, ctx), (at, CTX)),
        other => panic!("past the context: {other:?}"),
    }
    assert!(s.usable() && !s.pending() && s.pos() == at);
    s.verify(&e, &run_of(&d, 10)).unwrap();
    s.commit(&e, 10).unwrap();
    assert_eq!(s.pos(), CTX);
    // A chunk under 16 tokens caps the verify at the chunk: the step's rows, routes and RoPE tables are prefill's.
    let small = open_with(&g, &d, Some(4));
    let mut t = small.session(CTX).unwrap();
    match t.verify(&small, &run_of(&d, 5)) {
        Err(Error::VerifyShape { tokens: 5, most: 4 }) => {}
        other => panic!("5 tokens at chunk 4: {other:?}"),
    }
    t.verify(&small, &run_of(&d, 4)).unwrap();
    t.commit(&small, 4).unwrap();
    assert_eq!(t.pos(), 4);
}

#[test]
#[ignore = "needs the GPU"]
fn a_session_longer_than_the_engines_context_is_refused_not_overrun() {
    // The work area (1,246,336 bytes on the tiny model at ctx 64 and chunk 64) holds a verify's attention partials for
    // the engine's `ctx`. A 16-token verify in a session of 200,000 positions (782 splits) needs 14,442,112 bytes and
    // doesn't fit; a 1-token verify there needs only 902,704 and does, so the refusal is tested with 16 tokens (the
    // CPU test `a_session_deeper_than_the_engines_context_outgrows_the_work_area` pins those sizes).
    let (g, d) = tiny();
    let e = open(&g, &d);
    let mut s = e.session(200_000).unwrap();
    s.prefill(&e, &prompt(&d)).unwrap();
    match s.verify(&e, &run_of(&d, 16)) {
        Err(Error::VerifyBuffers {
            ctx: 200_000,
            need,
            have,
        }) => assert!(
            need > have && need == 14_442_112 && have == 1_246_336,
            "{need} {have}"
        ),
        other => panic!("{other:?}"),
    }
    assert!(s.usable() && !s.pending() && s.pos() == 10);
    assert_eq!(s.decode(&e, 5).unwrap().len(), d.n_vocab);
}

#[test]
#[ignore = "needs the GPU"]
fn a_session_dropped_with_a_snapshot_copy_in_flight_waits_for_it() {
    // `delay_copies` holds the KV copy on the transfer queue, so it is certainly running when the session drops; under
    // VK_LAYER_KHRONOS_validation a drop that doesn't wait raises VUID-vkDestroyBuffer-buffer-00922.
    let (g, d) = tiny();
    let e = open(&g, &d);
    let mut sn = snapshots(&e);
    sn.delay_copies(&e).unwrap();
    let mut s = started(&e, &d);
    sn.take(&e, &mut s, Marks::PROMPT_END)
        .unwrap()
        .expect("a snapshot");
    drop(s);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "verify step pending")]
fn decode_while_pending_asserts() {
    let (g, d) = tiny();
    let e = open(&g, &d);
    let mut s = started(&e, &d);
    s.verify(&e, &run_of(&d, 2)).unwrap();
    let _ = s.decode(&e, 5);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "verify step pending")]
fn reset_while_pending_asserts() {
    let (g, d) = tiny();
    let e = open(&g, &d);
    let mut s = started(&e, &d);
    s.verify(&e, &run_of(&d, 2)).unwrap();
    let _ = s.reset(&e);
}

#[test]
#[ignore = "needs the GPU"]
fn commit_and_row_refuse_another_engine_and_leave_the_step_pending() {
    let (g, d) = tiny();
    let (e, other) = (open(&g, &d), open(&g, &d));
    let mut s = started(&e, &d);
    s.verify(&e, &run_of(&d, 3)).unwrap();
    assert!(matches!(s.row(&other, 0), Err(Error::WrongEngine)));
    assert!(matches!(s.commit(&other, 2), Err(Error::WrongEngine)));
    // Nothing changed: the step is still pending, usable, and commits on its own engine.
    assert!(s.pending() && s.usable());
    s.commit(&e, 2).unwrap();
    assert!(!s.pending());
    assert_eq!(s.pos(), 12);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "verify step pending")]
fn prefill_while_pending_asserts() {
    let (g, d) = tiny();
    let e = open(&g, &d);
    let mut s = started(&e, &d);
    s.verify(&e, &run_of(&d, 2)).unwrap();
    let _ = s.prefill(&e, &[1, 2]);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "verify step pending")]
fn a_restore_into_the_session_while_pending_asserts() {
    let (g, d) = tiny();
    let e = open(&g, &d);
    let mut s = started(&e, &d);
    s.verify(&e, &run_of(&d, 2)).unwrap();
    let _ = s.set_restored(&e, &prompt(&d)[..4]);
}

fn snapshots(e: &Engine) -> Snapshots {
    Snapshots::new(
        e,
        PoolLimit {
            bytes: 64 << 20,
            requested: 64 << 20,
            cut: false,
        },
    )
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "verify step pending")]
fn a_snapshot_take_while_pending_asserts() {
    let (g, d) = tiny();
    let e = open(&g, &d);
    let mut sn = snapshots(&e);
    let mut s = started(&e, &d);
    s.verify(&e, &run_of(&d, 2)).unwrap();
    let _ = sn.take(&e, &mut s, Marks::PROMPT_END);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "verify step pending")]
fn a_snapshot_restore_while_pending_asserts() {
    let (g, d) = tiny();
    let e = open(&g, &d);
    let mut sn = snapshots(&e);
    let mut s = started(&e, &d);
    let id = sn
        .take(&e, &mut s, Marks::PROMPT_END)
        .unwrap()
        .expect("a snapshot")
        .id;
    s.verify(&e, &run_of(&d, 2)).unwrap();
    let _ = sn.restore(&e, &mut s, id);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "another verify pending")]
fn a_second_verify_while_pending_asserts() {
    let (g, d) = tiny();
    let e = open(&g, &d);
    let mut s = started(&e, &d);
    s.verify(&e, &run_of(&d, 2)).unwrap();
    let _ = s.verify(&e, &run_of(&d, 2));
}
