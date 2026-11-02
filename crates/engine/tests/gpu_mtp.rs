//! The MTP drafter on the GPU, with the tiny model's `blk.4` as the MTP block.
//!
//! A draft pass's logits are held to `MtpReference` (the CPU reference of the block) the way `gpu_engine.rs` holds
//! decode's to the main reference: free-running relative L2 under `BOUND`. The reference is fed the rows the session
//! has and, for each row's `h`, the hidden row the session itself computed: the ring's (`Session::mtp_hidden`: decodes,
//! commits) or the prefill window's (`Session::window_hidden`, after `capture_window_hidden`). So a miss measures the
//! MTP's arithmetic, not a gap between kernels or between a prefill and a decode. A drafted id is held to the
//! reference's argmax where the reference's top two are more than `TIE_MARGIN` apart.
mod common;

use common::{rel, top2, MtpOracle, TIE_MARGIN};
use engine::{
    Engine, Error, Marks, MtpOptions, Options, PoolLimit, Session, SessionState, Snapshots,
};
use gguf::Gguf;
use kernels::dispatch::{NO_DRAFT, WHERE_CPU, WHERE_HOST};
use qwen36::{dims::Dims, model::Model, quant::WType, testing::tiny_model};

/// `gpu_engine.rs`'s bound on decode's logits against the reference, free-running.
const BOUND: f64 = 1e-3;
/// The engine's context; sessions take this or less.
const CTX: usize = 1024;

fn tiny() -> (Gguf, Dims) {
    (Gguf::from_bytes(tiny_model(1)).unwrap(), Dims::tiny())
}

/// How a test opens its engine.
#[derive(Clone, Copy)]
struct Cfg {
    rows: usize,
    ring: usize,
    window: usize,
    chunk: Option<usize>,
    f32_prefill: bool,
}

impl Default for Cfg {
    fn default() -> Cfg {
        Cfg {
            rows: 512,
            ring: 128,
            window: 256,
            chunk: None,
            f32_prefill: false,
        }
    }
}

fn options(c: Cfg, mtp: bool) -> Options {
    Options {
        ctx: CTX,
        cache_slots: Some(0),
        speculation: true,
        f32_prefill: c.f32_prefill,
        prefill_chunk: c.chunk,
        mtp: mtp.then_some(MtpOptions {
            rows: c.rows,
            ring: c.ring,
            window: c.window,
            head_rows: 0,
        }),
        ..Options::default()
    }
}

fn open(g: &Gguf, d: &Dims, c: Cfg) -> Engine {
    Engine::open(g, d, &options(c, true)).unwrap()
}

/// A session of `CTX` positions that records the window's hidden rows for the reference.
fn capturing(e: &Engine) -> Session {
    let mut s = e.session(CTX).unwrap();
    s.capture_window_hidden(e).unwrap();
    s
}

/// `n` tokens of the tiny vocabulary, from `first`.
fn toks(d: &Dims, first: u32, n: usize) -> Vec<u32> {
    (first..first + n as u32)
        .map(|i| (i * 211 + 17) % d.n_vocab as u32)
        .collect()
}

fn one(d: &Dims, i: u32) -> u32 {
    (i * 97 + 3) % d.n_vocab as u32
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn same_state(a: &SessionState, b: &SessionState) -> bool {
    a.layers.len() == b.layers.len()
        && a.layers
            .iter()
            .zip(&b.layers)
            .all(|(x, y)| same_bits(&x.0, &y.0) && same_bits(&x.1, &y.1))
}

/// The hidden row of `q` the session itself computed: a decode's or a commit's from the ring, else the prefill window's.
fn hidden_of(e: &Engine, s: &Session, q: usize) -> Vec<f32> {
    s.mtp_hidden(e, q)
        .unwrap()
        .or_else(|| s.window_hidden(e, q))
        .unwrap_or_else(|| panic!("the session holds no hidden row of position {q}"))
}

/// The reference's draft row for `x` against the session's rows: its logits.
fn expected(o: &mut MtpOracle, e: &Engine, s: &Session, x: u32) -> Vec<f32> {
    let mut h = |q: usize| hidden_of(e, s, q);
    o.expect(&s.mtp_rows(), s.tokens(), s.pos(), x, &mut h).0
}

/// Judges `got` against `want`: under `BOUND`, and the same pick outside a near tie.
fn judge(what: &str, got: &[f32], id: u32, want: &[f32], gate: bool) -> f64 {
    let err = rel(got, want);
    let (top, gap) = top2(want);
    eprintln!("{what}: relative L2 {err:.3e}; draft {id}, reference {top} (lead {gap:.3})");
    if gate {
        assert!(
            err < BOUND,
            "{what}: logits off by {err:.3e}, over {BOUND:e}"
        );
        if gap > TIE_MARGIN {
            assert_eq!(id, top, "{what}: the draft differs outside a near tie");
        }
    }
    err
}

#[test]
#[ignore = "needs the GPU"]
fn mtp_logits_match_the_reference() {
    // The gating run is on the f32 prefill kernels; the default kernels' error is reported.
    logits_against_reference(true, true);
    logits_against_reference(false, false);
}

fn logits_against_reference(f32_prefill: bool, gate: bool) {
    let (g, d) = tiny();
    let e = open(
        &g,
        &d,
        Cfg {
            f32_prefill,
            ..Cfg::default()
        },
    );
    let m = Model::open(&g, &d).unwrap();
    let mut o = MtpOracle::new(&m);
    let mut s = capturing(&e);
    s.prefill(&e, &toks(&d, 0, 24)).unwrap();
    assert!(s.has_mtp());
    assert_eq!(s.mtp_rows(), (1..24).collect::<Vec<u32>>());
    let mut worst = 0f64;
    for i in 0..8 {
        let x = one(&d, 100 + i);
        let (id, logits) = s.mtp_draft_logits(&e, x).unwrap();
        let want = expected(&mut o, &e, &s, x);
        worst = worst.max(judge(
            &format!("step {i} (f32_prefill {f32_prefill})"),
            &logits,
            id,
            &want,
            gate,
        ));
        s.decode(&e, x).unwrap();
    }
    eprintln!("worst MTP logits error over 8 steps: {worst:.3e}");
}

#[test]
#[ignore = "needs the GPU"]
fn chained_drafts_match_the_reference() {
    chained_against_reference(true, true);
    chained_against_reference(false, false);
}

fn chained_against_reference(f32_prefill: bool, gate: bool) {
    let (g, d) = tiny();
    let e = open(
        &g,
        &d,
        Cfg {
            f32_prefill,
            ..Cfg::default()
        },
    );
    let m = Model::open(&g, &d).unwrap();
    let mut o = MtpOracle::new(&m);
    let mut s = capturing(&e);
    s.prefill(&e, &toks(&d, 0, 24)).unwrap();
    for i in 0..2 {
        s.decode(&e, one(&d, i)).unwrap();
    }
    for round in 0..3u32 {
        let x = one(&d, 50 + round);
        let x_pos = s.pos();
        let ids = s.mtp_draft(&e, x, 4).unwrap();
        assert_eq!(ids.len(), 4, "round {round}: four finite rows");
        let mut h = |q: usize| hidden_of(&e, &s, q);
        let chain = o.chain(&s.mtp_rows(), s.tokens(), x_pos, x, 4, &mut h);
        // The chain after a near tie follows another token than the engine's: compare only down to the first.
        let mut aligned = true;
        for (depth, ((top, gap, _), id)) in chain.iter().zip(&ids).enumerate() {
            eprintln!(
                "round {round} depth {}: draft {id}, reference {top} (lead {gap:.3})",
                depth + 1
            );
            if gate && aligned && *gap > TIE_MARGIN {
                assert_eq!(id, top, "round {round} depth {}", depth + 1);
            }
            aligned &= *gap > TIE_MARGIN && id == top;
        }
        // The chained rows are written and dropped: the rows end at x's position.
        assert_eq!(s.mtp_rows().last().copied(), Some(x_pos as u32));
        s.decode(&e, x).unwrap();
    }
}

/// A session that prefills `n` tokens and one that decodes them one at a time (catching up at the first draft) give the
/// same MTP rows and logits within `BOUND`.
fn window_against_catch_up(f32_prefill: bool, gate: bool) {
    let (g, d) = tiny();
    // 200 tokens in chunks of 144 and 56: the carry crosses a chunk. The ring holds 128 positions, so the decoding
    // session catches up on positions 73 ..= 199, the window of 127 the prefilling one writes.
    let cfg = Cfg {
        rows: 512,
        ring: 128,
        window: 127,
        chunk: Some(144),
        f32_prefill,
    };
    let e = open(&g, &d, cfg);
    let tokens = toks(&d, 0, 200);
    let x = one(&d, 7);
    let mut a = e.session(CTX).unwrap();
    a.prefill(&e, &tokens).unwrap();
    let mut b = e.session(CTX).unwrap();
    for &t in &tokens {
        b.decode(&e, t).unwrap();
    }
    let (ida, la) = a.mtp_draft_logits(&e, x).unwrap();
    let (idb, lb) = b.mtp_draft_logits(&e, x).unwrap();
    assert_eq!(a.mtp_rows(), b.mtp_rows(), "the same positions have rows");
    assert_eq!(a.mtp_rows(), (73..=200).collect::<Vec<u32>>());
    let err = rel(&la, &lb);
    eprintln!(
        "window rows against catch-up rows (f32_prefill {f32_prefill}): relative L2 {err:.3e}; drafts {ida} and {idb}"
    );
    if gate {
        assert!(
            err < BOUND,
            "the window's rows differ from catch-up's by {err:.3e}"
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn the_window_rows_equal_catch_up_rows() {
    // The window runs prefill's kernels (split-f16 by default), the catch-up rows decode's; the gating comparison
    // runs prefill on the f32 kernels. The default kernels' error is reported, not gated.
    window_against_catch_up(true, true);
    window_against_catch_up(false, false);
}

#[test]
#[ignore = "needs the GPU"]
fn mtp_after_a_restore_drafts_from_the_new_window_only() {
    // Request A runs, a snapshot is taken at its prompt, request B restores it and prefills 5 tokens.
    let (g, d) = tiny();
    let cfg = Cfg {
        ring: 16,
        chunk: Some(32),
        ..Cfg::default()
    };
    let e = open(&g, &d, cfg);
    let mut sn = Snapshots::new(
        &e,
        PoolLimit {
            bytes: 64 << 20,
            requested: 64 << 20,
            cut: false,
        },
    );
    let prompt = toks(&d, 0, 30);
    let suffix = toks(&d, 500, 5);
    // Request A on the session: a prompt, the snapshot, then decode and draft well past it.
    let mut s = e.session(CTX).unwrap();
    s.prefill(&e, &prompt).unwrap();
    let taken = sn
        .take(&e, &mut s, Marks::PROMPT_END)
        .unwrap()
        .expect("a snapshot was taken");
    for i in 0..8 {
        let x = one(&d, i);
        s.mtp_draft(&e, x, 2).unwrap();
        s.decode(&e, x).unwrap();
    }
    assert_eq!(
        s.mtp_rows().last().copied(),
        Some(37),
        "A's rows run past the snapshot"
    );
    // Request B: the restore clears the MTP, and nothing is drafted until the suffix's window gives it a row.
    sn.restore(&e, &mut s, taken.id).unwrap();
    assert!(s.mtp_rows().is_empty());
    assert!(
        s.mtp_draft(&e, one(&d, 1), 1).unwrap().is_empty(),
        "no hidden row before x"
    );
    assert!(s.mtp_rows().is_empty(), "a refused draft leaves no rows");
    s.prefill(&e, &suffix).unwrap();
    // The suffix starts at position 30: its first row has no hidden row before it (a hole), the rest are its window.
    assert_eq!(s.mtp_rows(), [31, 32, 33, 34]);
    let x = one(&d, 9);
    let (id, logits) = s.mtp_draft_logits(&e, x).unwrap();
    assert_eq!(s.mtp_rows(), [31, 32, 33, 34, 35]);
    // A fresh session restored from the same snapshot and prefilled the same suffix drafts the same.
    let mut f = e.session(CTX).unwrap();
    sn.restore(&e, &mut f, taken.id).unwrap();
    f.prefill(&e, &suffix).unwrap();
    let (idf, lf) = f.mtp_draft_logits(&e, x).unwrap();
    assert_eq!(f.mtp_rows(), s.mtp_rows());
    let err = rel(&logits, &lf);
    eprintln!(
        "a restored session against a fresh one: relative L2 {err:.3e}; drafts {id} and {idf}"
    );
    assert!(err < BOUND, "B drafts from A's rows: off by {err:.3e}");
    assert_eq!(id, idf);
}

#[test]
#[ignore = "needs the GPU"]
fn mtp_logits_after_a_slide_match_a_session_that_never_slid() {
    // `mtp_ctx` 512 and a ring of 128 slide the MTP's buffer once over 700 positions of steps (plain
    // decodes and verifies keeping 2 to 4 tokens, a draft pass before each). The comparison session has `mtp_ctx` 8,192
    // and prefills the same tokens with a window that gives it the slid session's rows. Run once more under
    // the validation layer: the slide's copies must not overlap (no VUID).
    slide_scenario(true, true);
    slide_scenario(false, false);
}

fn slide_scenario(f32_prefill: bool, gate: bool) {
    let (g, d) = tiny();
    let slid = Cfg {
        rows: 512,
        ring: 128,
        window: 256,
        chunk: None,
        f32_prefill,
    };
    let e = open(&g, &d, slid);
    let mut s = e.session(CTX).unwrap();
    s.prefill(&e, &toks(&d, 0, 8)).unwrap();
    let mut i = 0u32;
    let mut slides = 0;
    let mut last_first = 1;
    while s.pos() < 700 {
        let x = one(&d, 1000 + i);
        let k = 1 + (i as usize % 3);
        let drafts = s.mtp_draft(&e, x, k).unwrap();
        assert_eq!(drafts.len(), k, "step {i}");
        let rows = s.mtp_rows();
        if rows[0] > last_first {
            slides += 1;
            last_first = rows[0];
        }
        assert!(
            rows.len() <= 512,
            "step {i}: {} rows in a buffer of 512",
            rows.len()
        );
        if i % 3 == 1 {
            let kept = 2 + (i as usize / 3) % 3;
            let step: Vec<u32> = std::iter::once(x)
                .chain((0..3).map(|j| one(&d, 5000 + 4 * i + j)))
                .collect();
            s.verify(&e, &step).unwrap();
            s.commit(&e, kept).unwrap();
        } else {
            s.decode(&e, x).unwrap();
        }
        i += 1;
    }
    assert_eq!(slides, 1, "one slide over 700 positions");
    let x = one(&d, 9999);
    let (id, logits) = s.mtp_draft_logits(&e, x).unwrap();
    let rows = s.mtp_rows();
    let x_pos = s.pos() as u32;
    assert_eq!(rows.last().copied(), Some(x_pos));
    assert!(
        rows.windows(2).all(|w| w[1] == w[0] + 1),
        "a catch-up per step leaves no hole"
    );
    assert!(
        rows[0] > 1 && rows.len() >= 256,
        "the buffer slid: {} rows from {}",
        rows.len(),
        rows[0]
    );
    // The comparison session at `mtp_ctx` 8,192, its window the slid session's rows before x's.
    let window = rows.len() - 1;
    let big = Cfg {
        rows: 8192,
        ring: 128,
        window,
        chunk: None,
        f32_prefill,
    };
    let c = open(&g, &d, big);
    let mut t = c.session(CTX).unwrap();
    t.prefill(&c, s.tokens()).unwrap();
    let (idc, lc) = t.mtp_draft_logits(&c, x).unwrap();
    assert_eq!(t.mtp_rows(), rows, "the same positions have rows");
    let err = rel(&logits, &lc);
    eprintln!(
        "slid against never slid (f32_prefill {f32_prefill}, {} rows from {}): relative L2 {err:.3e}; drafts {id} and {idc}",
        rows.len(),
        rows[0]
    );
    if gate {
        assert!(err < BOUND, "after a slide the logits differ by {err:.3e}");
    }
}

#[test]
#[ignore = "needs the GPU"]
fn mtp_drafts_after_a_partial_accept_equal_a_fresh_session_over_the_same_rows() {
    // The session's rows come from catch-up and the fresh one's from prefill's window, so the gating comparison
    // runs prefill on the f32 kernels; the default kernels' error is reported.
    partial_accept(true, true);
    partial_accept(false, false);
}

fn partial_accept(f32_prefill: bool, gate: bool) {
    // A draft pass, a verify of x and its drafts, a commit keeping part of them, then another draft.
    let (g, d) = tiny();
    let e = open(
        &g,
        &d,
        Cfg {
            f32_prefill,
            ..Cfg::default()
        },
    );
    let mut s = e.session(CTX).unwrap();
    s.prefill(&e, &toks(&d, 0, 20)).unwrap();
    let x = one(&d, 1);
    let mut step = vec![x];
    step.extend(s.mtp_draft(&e, x, 3).unwrap());
    assert_eq!(step.len(), 4);
    s.verify(&e, &step).unwrap();
    s.commit(&e, 2).unwrap();
    // x's row stays, and nothing past the kept end is a row.
    assert_eq!(s.mtp_rows().last().copied(), Some(20));
    let x2 = one(&d, 2);
    let (id, logits) = s.mtp_draft_logits(&e, x2).unwrap();
    // x's row at 20, the kept draft's at 21 (caught up from the ring), x2's at 22.
    assert_eq!(s.mtp_rows(), (1..=22).collect::<Vec<u32>>());
    let mut f = e.session(CTX).unwrap();
    f.prefill(&e, s.tokens()).unwrap();
    let (idf, lf) = f.mtp_draft_logits(&e, x2).unwrap();
    assert_eq!(f.mtp_rows(), s.mtp_rows());
    let err = rel(&logits, &lf);
    eprintln!("after a partial accept, against a fresh session (f32_prefill {f32_prefill}): relative L2 {err:.3e}; drafts {id} and {idf}");
    if gate {
        assert!(
            err < BOUND,
            "the rejected drafts' rows were attended: off by {err:.3e}"
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn mtp_draft_refuses_k_past_the_context() {
    // A session of 512 positions holding 508.
    let (g, d) = tiny();
    let e = open(&g, &d, Cfg::default());
    let mut s = e.session(512).unwrap();
    s.prefill(&e, &toks(&d, 0, 508)).unwrap();
    let rows = s.mtp_rows();
    let x = one(&d, 1);
    // 3 positions follow x (509, 510, 511): at most a draft of 3.
    for k in [0usize, 4, 16] {
        match s.mtp_draft(&e, x, k) {
            Err(Error::MtpShape { k: got, most: 3 }) => assert_eq!(got, k),
            other => panic!("k {k}: expected MtpShape with most 3, got {other:?}"),
        }
    }
    assert_eq!((s.pos(), s.usable(), s.mtp_rows()), (508, true, rows));
    assert_eq!(s.mtp_draft(&e, x, 3).unwrap().len(), 3);
    s.decode(&e, x).unwrap();
    // At 16 positions of room the cap is the verify's: 15.
    let mut t = e.session(512).unwrap();
    t.prefill(&e, &toks(&d, 0, 40)).unwrap();
    assert!(matches!(
        t.mtp_draft(&e, x, 16),
        Err(Error::MtpShape { k: 16, most: 15 })
    ));
    assert_eq!(t.mtp_draft(&e, x, 15).unwrap().len(), 15);
    // The last position: no room for a chained row, and then none for x.
    for i in 0..2 {
        s.decode(&e, one(&d, 20 + i)).unwrap();
    }
    assert_eq!(s.pos(), 511);
    assert!(matches!(
        s.mtp_draft(&e, x, 1),
        Err(Error::MtpShape { k: 1, most: 0 })
    ));
    s.decode(&e, one(&d, 23)).unwrap();
    assert_eq!(s.pos(), 512);
    assert!(matches!(
        s.mtp_draft(&e, x, 1),
        Err(Error::ContextFull { ctx: 512 })
    ));
    assert!(s.usable());
    // A token outside the vocabulary is refused too, and changes nothing.
    let mut u = e.session(CTX).unwrap();
    u.prefill(&e, &toks(&d, 0, 20)).unwrap();
    let rows = u.mtp_rows();
    assert!(matches!(
        u.mtp_draft(&e, d.n_vocab as u32, 1),
        Err(Error::TokenOutOfRange { .. })
    ));
    assert_eq!((u.mtp_rows(), u.usable()), (rows, true));
}

#[test]
#[ignore = "needs the GPU"]
fn three_plain_steps_then_a_draft_catch_up_all_three() {
    // Plain decodes between draft passes leave no hole. Gated on the f32 prefill kernels.
    three_plain_steps(true, true);
    three_plain_steps(false, false);
}

fn three_plain_steps(f32_prefill: bool, gate: bool) {
    let (g, d) = tiny();
    let e = open(
        &g,
        &d,
        Cfg {
            f32_prefill,
            ..Cfg::default()
        },
    );
    let m = Model::open(&g, &d).unwrap();
    let mut o = MtpOracle::new(&m);
    let mut s = capturing(&e);
    s.prefill(&e, &toks(&d, 0, 10)).unwrap();
    let x = one(&d, 1);
    s.mtp_draft(&e, x, 1).unwrap();
    assert_eq!(s.mtp_rows(), (1..=10).collect::<Vec<u32>>());
    // x and three more, all plain: positions 10 to 13 are decoded, none drafted.
    for t in [x, one(&d, 2), one(&d, 3), one(&d, 4)] {
        s.decode(&e, t).unwrap();
    }
    assert_eq!(
        s.mtp_rows(),
        (1..=10).collect::<Vec<u32>>(),
        "plain steps add no rows"
    );
    let x = one(&d, 5);
    let (id, logits) = s.mtp_draft_logits(&e, x).unwrap();
    assert_eq!(
        s.mtp_rows(),
        (1..=14).collect::<Vec<u32>>(),
        "rows at 11, 12 and 13 and x's at 14"
    );
    let want = expected(&mut o, &e, &s, x);
    judge("after three plain steps", &logits, id, &want, gate);
}

#[test]
#[ignore = "needs the GPU"]
fn a_draft_pass_leaves_the_main_logits_bit_equal() {
    // The pass shares Scratch, Prefill::work and Session::part with the main steps. Two sessions run the
    // same decodes and verifies, one with a draft pass before every step; every main logit row and the final KV and
    // DeltaNet state are bit-equal.
    let (g, d) = tiny();
    let e = open(&g, &d, Cfg::default());
    let run = |draft: bool| -> (Vec<Vec<f32>>, SessionState) {
        let mut s = e.session(CTX).unwrap();
        let mut rows = vec![s.prefill(&e, &toks(&d, 0, 24)).unwrap()];
        for i in 0..12u32 {
            let x = one(&d, 300 + i);
            if draft {
                let k = 1 + (i as usize % 4);
                assert_eq!(s.mtp_draft(&e, x, k).unwrap().len(), k);
            }
            if i % 3 == 2 {
                let step: Vec<u32> = std::iter::once(x)
                    .chain((0..(1 + i % 4)).map(|j| one(&d, 700 + 4 * i + j)))
                    .collect();
                s.verify(&e, &step).unwrap();
                for r in 0..step.len() {
                    rows.push(s.row(&e, r).unwrap());
                }
                s.commit(&e, 1 + (i as usize % step.len())).unwrap();
            } else {
                rows.push(s.decode(&e, x).unwrap());
            }
        }
        let state = s.state(&e).unwrap();
        (rows, state)
    };
    let (plain, plain_state) = run(false);
    let (drafted, drafted_state) = run(true);
    assert_eq!(plain.len(), drafted.len());
    for (i, (a, b)) in plain.iter().zip(&drafted).enumerate() {
        assert!(
            same_bits(a, b),
            "main logit row {i} changed by the draft passes"
        );
    }
    assert!(
        same_state(&plain_state, &drafted_state),
        "the KV or the DeltaNet state changed by the draft passes"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_refused_mtp_slot_fails_the_pass_and_leaves_the_main_state() {
    let (g, d) = tiny();
    let e = open(&g, &d, Cfg::default());
    let mut s = e.session(CTX).unwrap();
    s.prefill(&e, &toks(&d, 0, 12)).unwrap();
    let x = one(&d, 1);
    assert_eq!(s.mtp_draft(&e, x, 1).unwrap().len(), 1);
    let saved: Vec<(u64, u32)> = (0..d.n_expert)
        .map(|ex| e.slot_entry(d.n_layer, ex).unwrap())
        .collect();
    assert!(saved
        .iter()
        .all(|&(addr, loc)| loc == WHERE_HOST && addr != 0));
    for ex in 0..d.n_expert {
        // HOST with no address: a corrupt entry, as `a_slot_entry_without_an_address_fails_the_step` makes one.
        e.set_slot_entry(d.n_layer, ex, 0, WHERE_HOST).unwrap();
    }
    let (pos, kv) = (s.pos(), s.kv_tokens().to_vec());
    match s.mtp_draft(&e, x, 2) {
        Err(Error::BadSlotEntry { layer, .. }) => assert_eq!(layer as usize, d.n_layer),
        other => panic!("expected BadSlotEntry at the MTP's layer, got {other:?}"),
    }
    assert_eq!((s.pos(), s.kv_tokens().to_vec()), (pos, kv));
    assert!(s.usable(), "a refused slot leaves the session usable");
    assert!(s.mtp_rows().is_empty(), "and the MTP's rows cleared");
    // The main model is untouched: a decode works.
    s.decode(&e, x).unwrap();
    // The entries back: the next pass catches up from the ring, which the refusal kept.
    for (ex, &(addr, loc)) in saved.iter().enumerate() {
        e.set_slot_entry(d.n_layer, ex, addr, loc).unwrap();
    }
    assert_eq!(s.mtp_draft(&e, one(&d, 2), 1).unwrap().len(), 1);
    assert_eq!(s.mtp_rows(), (12..=13).collect::<Vec<u32>>());
}

#[test]
#[ignore = "needs the GPU"]
fn the_mtp_router_never_writes_the_route_log() {
    let (g, d) = tiny();
    let e = open(&g, &d, Cfg::default());
    let mut s = e.session(CTX).unwrap();
    s.prefill(&e, &toks(&d, 0, 12)).unwrap();
    s.decode(&e, one(&d, 1)).unwrap();
    let before = e.last_routes();
    let misses = e.stats().misses;
    assert_eq!(s.mtp_draft(&e, one(&d, 2), 3).unwrap().len(), 3);
    assert_eq!(
        e.last_routes(),
        before,
        "the decode's routing log is still the decode's"
    );
    assert_eq!(e.stats().misses, misses, "the pass counts no misses");
}

#[test]
#[ignore = "needs the GPU"]
fn a_failed_mtp_submit_leaves_the_session_unusable_and_the_rows_clear() {
    let (g, d) = tiny();
    let e = open(&g, &d, Cfg::default());
    let mut s = e.session(CTX).unwrap();
    s.prefill(&e, &toks(&d, 0, 10)).unwrap();
    assert!(!s.mtp_rows().is_empty());
    s.fail_next_mtp_submit();
    let x = one(&d, 1);
    match s.mtp_draft(&e, x, 2) {
        Err(Error::Step { pos: 10, .. }) => {}
        other => panic!("expected the failed submit's Step error, got {other:?}"),
    }
    assert!(!s.usable());
    assert!(s.mtp_rows().is_empty());
    assert_eq!(s.pos(), 10);
    assert!(!s.pending());
    // After a reset the session decodes and drafts again.
    s.reset(&e).unwrap();
    assert!(s.usable());
    s.decode(&e, x).unwrap();
    assert_eq!(s.mtp_draft(&e, one(&d, 2), 1).unwrap().len(), 1);
}

#[test]
#[ignore = "needs the GPU"]
fn prefill_with_mtp_writes_the_same_main_state() {
    // The window only adds: the KV, the DeltaNet state and the logits are bit-equal to a session without the MTP.
    let (g, d) = tiny();
    let cfg = Cfg {
        ring: 16,
        chunk: Some(32),
        ..Cfg::default()
    };
    let with = open(&g, &d, cfg);
    let without = Engine::open(&g, &d, &options(cfg, false)).unwrap();
    let tokens = toks(&d, 0, 100);
    let mut a = with.session(CTX).unwrap();
    let la = a.prefill(&with, &tokens).unwrap();
    let mut b = without.session(CTX).unwrap();
    let lb = b.prefill(&without, &tokens).unwrap();
    assert!(a.has_mtp() && !b.has_mtp());
    assert!(same_bits(&la, &lb), "the last logits");
    assert!(
        same_state(&a.state(&with).unwrap(), &b.state(&without).unwrap()),
        "the KV and the DeltaNet state"
    );
    // The window (256) covers the whole 100-token call.
    assert_eq!(a.mtp_rows(), (1..100).collect::<Vec<u32>>());
    for i in 0..4 {
        let t = one(&d, i);
        assert!(same_bits(
            &a.decode(&with, t).unwrap(),
            &b.decode(&without, t).unwrap()
        ));
    }
}

#[test]
#[ignore = "needs the GPU"]
fn a_session_takes_what_the_mtp_sizes_count() {
    // The session's arena holds exactly `session_bytes_mtp`, and the engine reports `mtp_vram_bytes`.
    let (g, d) = tiny();
    let c = Cfg::default();
    let e = open(&g, &d, c);
    let o = e.mtp().expect("opened with the MTP");
    let s = e.session(CTX).unwrap();
    assert_eq!(s.vram_bytes(), engine::session_bytes_mtp(&d, CTX, Some(o)));
    assert_eq!(
        s.vram_bytes(),
        engine::session_bytes(&d, CTX) + engine::mtp_session_bytes(&d, o.rows, o.ring)
    );
    assert!(e.mtp_vram_bytes() > engine::mtp_session_bytes(&d, o.rows, o.ring));
    let plain = Engine::open(&g, &d, &options(c, false)).unwrap();
    assert_eq!(
        plain.session(CTX).unwrap().vram_bytes(),
        engine::session_bytes(&d, CTX)
    );
    assert!(!plain.session(CTX).unwrap().has_mtp());
}

#[test]
#[ignore = "needs the GPU"]
fn a_live_continuation_prefill_near_the_cap_slides_and_matches_a_session_that_never_slid() {
    // A second prefill call on a live session whose MTP buffer is nearly full. The window's slide
    // (through `WindowLayout`'s temps) and its first row reading `h` from the ring both run on the GPU, in a call of two
    // chunks (the second takes the carry). Under the validation layer it must report no VUID. Gating at
    // `f32_prefill: true`; the default kernels' error is reported.
    live_continuation(true, true);
    live_continuation(false, false);
}

fn live_continuation(f32_prefill: bool, gate: bool) {
    let (g, d) = tiny();
    let cfg = Cfg {
        rows: 512,
        ring: 128,
        window: 256,
        chunk: Some(144),
        f32_prefill,
    };
    let e = open(&g, &d, cfg);
    let mut s = e.session(CTX).unwrap();
    s.prefill(&e, &toks(&d, 0, 8)).unwrap();
    for i in 0..480u32 {
        let x = one(&d, 2000 + i);
        assert_eq!(s.mtp_draft(&e, x, 1).unwrap().len(), 1);
        s.decode(&e, x).unwrap();
    }
    assert_eq!(s.pos(), 488);
    assert_eq!(s.mtp_rows(), (1..=487).collect::<Vec<u32>>());
    // 200 more positions on the live session: 487 + 200 rows don't fit 512, so the window slides first.
    s.prefill(&e, &toks(&d, 3000, 200)).unwrap();
    assert_eq!(s.pos(), 688);
    assert_eq!(
        s.mtp_rows(),
        (232..=687).collect::<Vec<u32>>(),
        "the newest 256 rows kept, then the window's 200 (the first from the ring's h)"
    );
    let x = one(&d, 9999);
    let (id, logits) = s.mtp_draft_logits(&e, x).unwrap();
    let rows = s.mtp_rows();
    assert_eq!(rows, (232..=688).collect::<Vec<u32>>());
    // A session at `mtp_ctx` 8,192 prefilling the same tokens with the window that gives it the same rows.
    let big = Cfg {
        rows: 8192,
        window: rows.len() - 1,
        ..cfg
    };
    let c = open(&g, &d, big);
    let mut t = c.session(CTX).unwrap();
    t.prefill(&c, s.tokens()).unwrap();
    let (idc, lc) = t.mtp_draft_logits(&c, x).unwrap();
    assert_eq!(t.mtp_rows(), rows, "the same positions have rows");
    // The ring-fed row itself: position 488's key and value (its `h` came from the ring slot, the rest of the call's
    // from the chunk and the carry) against the same row in the session that prefilled everything in one call.
    let at = rows.iter().position(|&q| q == 488).unwrap();
    let (ks, vs) = s.mtp_kv_row(&e, at).unwrap().unwrap();
    let (kc, vc) = t.mtp_kv_row(&c, at).unwrap().unwrap();
    assert!(ks.iter().any(|v| *v != 0.0) && ks.iter().all(|v| v.is_finite()));
    let (errk, errv) = (rel(&ks, &kc), rel(&vs, &vc));
    eprintln!("position 488 (the ring-fed row): K off by {errk:.3e}, V by {errv:.3e}");
    if gate {
        assert!(
            errk < BOUND && errv < BOUND,
            "the ring-fed row differs: K {errk:.3e}, V {errv:.3e}"
        );
    }
    let err = rel(&logits, &lc);
    eprintln!(
        "live continuation against never slid (f32_prefill {f32_prefill}): relative L2 {err:.3e}; drafts {id} and {idc}"
    );
    if gate {
        assert!(
            err < BOUND,
            "after a live continuation's slide the logits differ by {err:.3e}"
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn an_mtp_ring_past_the_chunk_is_clamped_at_load() {
    // A ring of 2,048 in a prefill chunk of 2,048 loads with 2,032 (the chunk minus 16), and says so.
    let (g, d) = tiny();
    let opts = |ring| Options {
        ctx: 4096,
        cache_slots: Some(0),
        speculation: true,
        mtp: Some(MtpOptions {
            rows: 8192,
            ring,
            window: 256,
            head_rows: 0,
        }),
        ..Options::default()
    };
    let e = Engine::open(&g, &d, &opts(2048)).unwrap();
    assert_eq!(e.prefill_chunk(), 2048);
    assert_eq!(e.mtp().map(|o| o.ring), Some(2032));
    assert_eq!(e.mtp_ring_clamp(), Some((2048, 2032)));
    let mut s = e.session(4096).unwrap();
    s.prefill(&e, &toks(&d, 0, 40)).unwrap();
    assert_eq!(s.mtp_draft(&e, one(&d, 1), 3).unwrap().len(), 3);
    let fit = Engine::open(&g, &d, &opts(256)).unwrap();
    assert_eq!(
        (fit.mtp().map(|o| o.ring), fit.mtp_ring_clamp()),
        (Some(256), None)
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_truncated_head_drafts_the_argmax_of_its_rows() {
    // `mtp_head_rows` N reads the first N rows of the main head: the draft is the argmax of the full head's first N
    // logits, and so below N. The tiny vocabulary is 512, so N is 256.
    let (g, d) = tiny();
    let n = d.n_vocab / 2;
    let open_with = |head_rows| {
        let mut o = options(Cfg::default(), true);
        o.mtp = o.mtp.map(|m| MtpOptions { head_rows, ..m });
        Engine::open(&g, &d, &o).unwrap()
    };
    let (full, cut) = (open_with(0), open_with(n));
    assert_eq!(cut.mtp().map(|o| o.head_rows), Some(n));
    let mut a = full.session(CTX).unwrap();
    let mut b = cut.session(CTX).unwrap();
    a.prefill(&full, &toks(&d, 0, 24)).unwrap();
    b.prefill(&cut, &toks(&d, 0, 24)).unwrap();
    let mut decisive = 0;
    for i in 0..6 {
        let x = one(&d, 100 + i);
        let (_, all) = a.mtp_draft_logits(&full, x).unwrap();
        let (id, rows) = b.mtp_draft_logits(&cut, x).unwrap();
        assert_eq!((all.len(), rows.len()), (d.n_vocab, n));
        assert!(rel(&all[..n], &rows) < BOUND, "step {i}: the rows differ");
        assert!(
            (id as usize) < n,
            "step {i}: draft {id} is outside the first {n} rows"
        );
        let (best, gap) = top2(&all[..n]);
        if gap > TIE_MARGIN {
            assert_eq!(id, best, "step {i}");
            decisive += 1;
        }
        a.decode(&full, x).unwrap();
        b.decode(&cut, x).unwrap();
    }
    assert!(
        decisive > 0,
        "every step was a near tie: nothing was checked"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_row_with_no_finite_logit_gives_no_draft_and_clears_the_rows_not_the_ring() {
    // Token 511's embedding gets a NaN scale (as `nan_logits_fail_the_step...` does): a draft row for it is NaN all the
    // way, `argmax_rows` writes `NO_DRAFT`, and the pass returns no draft and no error. The rows go, the ring stays.
    let mut bytes = tiny_model(1);
    let d = Dims::tiny();
    let off = {
        let g = Gguf::from_bytes(bytes.clone()).unwrap();
        let t = g.tensor("token_embd.weight").unwrap();
        assert_eq!(t.ggml_type, WType::Q8_0.ggml());
        t.offset as usize + 511 * WType::Q8_0.row_bytes(d.n_embd)
    };
    bytes[off..off + 2].copy_from_slice(&0x7E00u16.to_le_bytes());
    let g = Gguf::from_bytes(bytes).unwrap();
    let e = open(&g, &d, Cfg::default());
    let mut s = e.session(CTX).unwrap();
    let prompt = toks(&d, 0, 12);
    assert!(!prompt.contains(&511));
    s.prefill(&e, &prompt).unwrap();
    assert!(!s.mtp_rows().is_empty());
    assert!(s.mtp_draft(&e, 511, 2).unwrap().is_empty());
    assert_eq!((s.pos(), s.usable(), s.mtp_rows()), (12, true, vec![]));
    let (id, _) = s.mtp_draft_logits(&e, 511).unwrap();
    assert_eq!(id, NO_DRAFT);
    // The ring held: the next pass at an ordinary token catches up from it and drafts.
    assert_eq!(s.mtp_draft(&e, one(&d, 1), 2).unwrap().len(), 2);
    assert_eq!(s.mtp_rows(), [12]);
}

#[test]
#[ignore = "needs the GPU"]
fn an_mtp_expert_on_the_cpu_fails_the_pass_and_leaves_the_session_usable() {
    // The test points the MTP block's entries HOST; a CPU entry is refused as in decode (`ExpertOnCpu`), after the
    // submit.
    let (g, d) = tiny();
    let e = open(&g, &d, Cfg::default());
    let mut s = e.session(CTX).unwrap();
    s.prefill(&e, &toks(&d, 0, 12)).unwrap();
    let saved: Vec<(u64, u32)> = (0..d.n_expert)
        .map(|ex| e.slot_entry(d.n_layer, ex).unwrap())
        .collect();
    for ex in 0..d.n_expert {
        e.set_slot_entry(d.n_layer, ex, 0, WHERE_CPU).unwrap();
    }
    match s.mtp_draft(&e, one(&d, 1), 2) {
        Err(Error::ExpertOnCpu { layer, .. }) => assert_eq!(layer as usize, d.n_layer),
        other => panic!("expected ExpertOnCpu at the MTP's layer, got {other:?}"),
    }
    assert_eq!((s.pos(), s.usable(), s.mtp_rows()), (12, true, vec![]));
    for (ex, &(addr, loc)) in saved.iter().enumerate() {
        e.set_slot_entry(d.n_layer, ex, addr, loc).unwrap();
    }
    assert_eq!(s.mtp_draft(&e, one(&d, 1), 2).unwrap().len(), 2);
}
