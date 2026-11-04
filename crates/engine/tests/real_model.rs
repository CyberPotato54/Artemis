//! The engine's VRAM plan on the real model file (no GPU). Reads ARTEMIS_MODEL.
mod common;

use gguf::Gguf;
use qwen36::{dims::Dims, model::Model};

#[test]
#[ignore = "needs ARTEMIS_MODEL"]
fn the_weights_arena_takes_the_weights_and_little_else() {
    // The arena packs 600-odd tensors, the slot table and the scratch buffers into chunks of at most 1 GiB, cut to
    // what they hold: no chunk tail is left over, whatever order the tensors come in.
    let path = std::env::var("ARTEMIS_MODEL").expect("set ARTEMIS_MODEL to the model's GGUF");
    let g = Gguf::open(std::path::Path::new(&path)).unwrap();
    let m = Model::open(&g, &Dims::QWEN36).unwrap();
    let mtp = format!("blk.{}.", Dims::QWEN36.n_layer);
    let weights: u64 = g
        .tensors()
        .filter(|t| {
            !t.name.starts_with(&mtp) && !t.name.contains("_exps") && t.name != "token_embd.weight"
        })
        .map(|t| t.bytes.len() as u64)
        .sum();
    let n = engine::need(&m, false);
    let over = n.vram - weights;
    eprintln!(
        "weights {} MiB; the arena {} MiB, {} KiB over",
        weights >> 20,
        n.vram >> 20,
        over >> 10
    );
    assert!(
        over < 8 << 20,
        "the arena takes {over} bytes more than the weights"
    );
}

#[test]
#[ignore = "needs ARTEMIS_MODEL"]
fn a_cache_row_of_the_real_model_is_76_439_552_bytes() {
    // 1,900,544-byte slots (Q5_K down), and 2,039,808 in layers 34, 38 and 39 (Q6_K down).
    let path = std::env::var("ARTEMIS_MODEL").expect("set ARTEMIS_MODEL to the model's GGUF");
    let g = Gguf::open(std::path::Path::new(&path)).unwrap();
    let m = Model::open(&g, &Dims::QWEN36).unwrap();
    let sizes = engine::slot_sizes(&m);
    for (l, &s) in sizes.iter().enumerate() {
        let want = if [34, 38, 39].contains(&l) {
            2_039_808
        } else {
            1_900_544
        };
        assert_eq!(s, want, "layer {l}");
    }
    assert_eq!(engine::cache_bytes(&sizes, 102), 102 * 76_439_552);
}

#[test]
#[ignore = "needs ARTEMIS_MODEL and the GPU"]
fn the_default_cache_loads_beside_its_session() {
    // The driver's VRAM budget falls from 15,416 MiB to about 14,340 once this process uses about 11.5 GB, so a cache
    // sized from the budget before it was allocated failed to load on the test PC. Sized against the budget at full
    // usage (`engine::fit_cache`), the default cache loads at generate's default context and at the KL's, its session
    // allocates, a token decodes, a full prefill chunk runs, and the margin is left.
    let path = std::env::var("ARTEMIS_MODEL").expect("set ARTEMIS_MODEL to the model's GGUF");
    let g = Gguf::open(std::path::Path::new(&path)).unwrap();
    for ctx in [65_536, 4_096] {
        let e = engine::Engine::open(
            &g,
            &Dims::QWEN36,
            &engine::Options {
                ctx,
                ..Default::default()
            },
        )
        .unwrap();
        let mut s = e.session(ctx).unwrap();
        let logits = s.decode(&e, 760).unwrap();
        // A full prefill chunk after it: prefill's buffers were counted before the cache was sized.
        let chunk: Vec<u32> = (0..e.prefill_chunk() as u32)
            .map(|i| 1000 + i % 5000)
            .collect();
        let last = s.prefill(&e, &chunk).unwrap();
        let h = e.vram_heap().unwrap();
        eprintln!(
            "ctx {ctx}: {} slots per layer ({} MiB), prefill's buffers {} MiB; usage {} MiB of a {} MiB budget with \
             the session; longest submit {:.1} ms",
            e.cache_slots(),
            e.stats().cache_bytes >> 20,
            e.prefill_bytes() >> 20,
            h.usage >> 20,
            h.budget >> 20,
            e.stats().longest_submit_us as f64 / 1e3
        );
        assert_eq!(logits.len(), Dims::QWEN36.n_vocab);
        assert_eq!(last.len(), Dims::QWEN36.n_vocab);
        assert_eq!(s.pos(), 1 + e.prefill_chunk());
        // What the prefill arena took is what `prefill_bytes` reserved for it before the cache was sized.
        assert_eq!(
            e.prefill_bytes(),
            engine::prefill_bytes(&Dims::QWEN36, e.prefill_chunk(), None)
        );
        assert!(
            e.cache_slots() > 100,
            "ctx {ctx}: only {} slots",
            e.cache_slots()
        );
        assert!(
            h.usage + engine::VRAM_MARGIN <= h.budget,
            "ctx {ctx}: the margin isn't kept: {h:?}"
        );
    }
}

#[test]
#[ignore = "needs the GPU and the model"]
fn swaps_overlapping_steps_change_no_logit_on_the_real_model() {
    // On the real model: 05-code-write-lru through prefill and 32 decoded tokens at 102 slots
    // with a poisoned swap every step, against no cache. Real slots are 1.9 MB, so each copy takes about 0.13 ms at
    // the transfer queue's measured 14 GB/s. The copy is issued before its step is recorded, so it may land before the
    // step's GPU work begins: the test checks that no step reads a slot mid-copy, not that the GPU work overlapped.
    let path = std::env::var("ARTEMIS_MODEL").expect("ARTEMIS_MODEL");
    let g = gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
    let prompt: Vec<u32> = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/golden/prompts/05-code-write-lru.tokens"
    ))
    .unwrap()
    .as_chunks::<4>()
    .0
    .iter()
    .map(|c| u32::from_le_bytes(*c))
    .collect();
    let run = |opts: engine::Options| -> (Vec<u64>, engine::Stats) {
        let e = engine::Engine::open(&g, &qwen36::dims::Dims::QWEN36, &opts).unwrap();
        let mut s = e.session(4096).unwrap();
        let fp = |l: &[f32]| {
            let b: Vec<u8> = l.iter().flat_map(|x| x.to_le_bytes()).collect();
            qwen36::testing::fnv1a64(&b)
        };
        let mut logits = s.prefill(&e, &prompt).unwrap();
        let mut prints = vec![fp(&logits)];
        for _ in 0..32 {
            let t = engine::greedy(&logits);
            logits = s.decode(&e, t).unwrap();
            prints.push(fp(&logits));
        }
        (prints, e.stats())
    };
    let (plain, _) = run(engine::Options {
        ctx: 4096,
        cache_slots: Some(0),
        ..Default::default()
    });
    let (swapped, st) = run(engine::Options {
        ctx: 4096,
        cache_slots: Some(102),
        force_swap_seed: Some(1),
        poison_swaps: true,
        ..Default::default()
    });
    eprintln!(
        "steps prepared while a swap's copy was in flight: {} of {}; {st:?}",
        st.steps_overlapping_copies, st.steps
    );
    assert!(swapped == plain, "the logits differ");
    assert_eq!(
        st.swaps_started, 33,
        "one swap per step: the prompt's chunk and 32 tokens"
    );
    assert!(
        st.steps_overlapping_copies > 0,
        "no step was prepared while a swap's copy was in flight: {st:?}"
    );
}

#[test]
#[ignore = "needs the GPU and the model"]
fn staging_changes_no_logit_on_the_real_model() {
    // On the real model: the long golden prompt's first 2,100 tokens (a 2,048-token chunk and a
    // 52-token one) through prefill at the default cache, staged and never staged, with every position's logits: the
    // same bits. The staged run stages the chunks of at least STAGE_FROM tokens; the other stages none.
    let path = std::env::var("ARTEMIS_MODEL").expect("ARTEMIS_MODEL");
    let g = gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
    let toks: Vec<u32> = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/golden/prompts/20-long-option-rs.tokens"
    ))
    .unwrap()
    .as_chunks::<4>()
    .0
    .iter()
    .take(2100)
    .map(|c| u32::from_le_bytes(*c))
    .collect();
    let run = |stage_from: Option<usize>| {
        let e = engine::Engine::open(
            &g,
            &qwen36::dims::Dims::QWEN36,
            &engine::Options {
                ctx: 4096,
                stage_from,
                ..Default::default()
            },
        )
        .unwrap();
        let mut s = e.session(4096).unwrap();
        let mut fp = Vec::new();
        s.prefill_each(&e, &toks, 0, |_, l| {
            let b: Vec<u8> = l.iter().flat_map(|x| x.to_le_bytes()).collect();
            fp.push(qwen36::testing::fnv1a64(&b));
            Ok::<(), engine::Error>(())
        })
        .unwrap();
        (fp, e.stats(), e.attn_pairs())
    };
    let (a, sa, cap) = run(None);
    let (b, sb, _) = run(Some(usize::MAX));
    let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    assert_eq!(
        differ, 0,
        "positions whose logits differ, staged against unstaged"
    );
    let want = engine::chunk_lengths(0, toks.len(), engine::PREFILL_CHUNK, cap)
        .iter()
        .filter(|&&t| t >= engine::STAGE_FROM)
        .count() as u64;
    assert_eq!(sa.staged_steps, want, "{sa:?}");
    assert!(sa.staged > 0 && sb.staged == 0, "{sa:?} {sb:?}");
    eprintln!("staged: {sa:?}\nunstaged: {sb:?}");
}

#[test]
#[ignore = "needs the GPU and the model"]
fn the_cache_lends_shrinks_and_grows_on_the_real_model() {
    // On the real model at the default cache for --ctx 65536: the staging area lends
    // its rows between staged chunks; the long prompt's first 2,100 tokens (a staged chunk of 2,048 and an unstaged 52)
    // take them back and return them; 24 tokens decode; the budget is cut short of the margin by half the cache's last
    // chunk, so that chunk retires and is freed; the cut lifted, it grows back and the policy fills it. The logits are
    // those of an engine with no cache throughout.
    let path = std::env::var("ARTEMIS_MODEL").expect("ARTEMIS_MODEL");
    let g = gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
    let toks: Vec<u32> = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/golden/prompts/20-long-option-rs.tokens"
    ))
    .unwrap()
    .as_chunks::<4>()
    .0
    .iter()
    .take(2100)
    .map(|c| u32::from_le_bytes(*c))
    .collect();
    let fp = |l: &[f32]| {
        let b: Vec<u8> = l.iter().flat_map(|x| x.to_le_bytes()).collect();
        qwen36::testing::fnv1a64(&b)
    };
    let run = |opts: engine::Options, watch: bool| {
        let e = engine::Engine::open(&g, &qwen36::dims::Dims::QWEN36, &opts).unwrap();
        let (own, lent) = (e.cache_slots(), e.stats().lent_slots);
        // The session the cache was fitted beside: a smaller one would leave room to grow a chunk at once.
        let mut s = e.session(opts.ctx).unwrap();
        let mut logits = s.prefill(&e, &toks).unwrap();
        if watch {
            // What the fit left, now that the driver has taken its first-use memory out of DRIVER_WORK: a grow back to
            // the size at load needs the margin and GROW_SLACK (32 MiB) free, so the grow back below needs this above
            // -32 MiB.
            let h = e.vram_heap().unwrap();
            let spare =
                h.budget as i64 - (h.usage + engine::VRAM_MARGIN + engine::DRIVER_WORK) as i64;
            eprintln!(
                "after the prompt: {} MiB spare beyond the margin and DRIVER_WORK",
                spare >> 20
            );
        }
        let mut prints = vec![fp(&logits)];
        let mut seen = vec![(0usize, e.live_slots(), e.stats())];
        let mut cut = 0;
        for i in 0..160 {
            if watch && i == 24 {
                let h = e.vram_heap().unwrap();
                // Half the last chunk short (chunks of 14 rows, the last partial): freeing it brings the margin back
                // with room to spare for the budget's drift between readings (a few MiB), and while the cut holds a grow
                // finds no room for the chunk, the margin and GROW_SLACK.
                let rows = match own % 14 {
                    0 => 14,
                    r => r,
                };
                cut = (h.budget + rows as u64 * 76_439_552 / 2)
                    .saturating_sub(h.usage + engine::VRAM_MARGIN);
                e.set_budget_cut(cut);
            }
            if watch && i == 40 {
                let h = e.vram_heap().unwrap();
                assert!(
                    h.usage + engine::VRAM_MARGIN <= h.budget - cut,
                    "the margin isn't back: {h:?}"
                );
                e.set_budget_cut(0);
            }
            logits = s.decode(&e, engine::greedy(&logits)).unwrap();
            prints.push(fp(&logits));
            seen.push((i, e.live_slots(), e.stats()));
        }
        (prints, own, lent, seen)
    };
    let (want, _, _, _) = run(
        engine::Options {
            ctx: 4096,
            cache_slots: Some(0),
            ..Default::default()
        },
        false,
    );
    let (got, own, lent, seen) = run(
        engine::Options {
            ctx: 65_536,
            budget_every: Some(1),
            ..Default::default()
        },
        true,
    );
    for (i, live, st) in seen.iter().step_by(8) {
        eprintln!(
            "step {i}: {live} live slots, {} MiB own; shrinks {} grows {} refused {} fills {} swaps {}; hits {} \
             staged {} misses {}",
            st.cache_bytes >> 20,
            st.shrinks,
            st.grows,
            st.grows_refused,
            st.fills,
            st.swaps_started,
            st.hits,
            st.staged,
            st.misses
        );
    }
    let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert_eq!(differ, 0, "positions whose logits differ from no cache's");
    let last = seen.last().unwrap();
    eprintln!("{own} slots of its own, {lent} lent; {:?}", last.2);
    // The staging area lends every whole row it holds: two buffers of (n_expert - own) experts, so the count falls as
    // the fit takes more slots of its own (6 rows at 137 slots, 5 at 146; the desktop's VRAM use moves the fit).
    assert!(lent >= 1, "{lent} lent rows");
    assert_eq!(
        lent,
        (last.2.staging_bytes / 76_439_552) as usize,
        "lent rows against the staging area's {} bytes",
        last.2.staging_bytes
    );
    assert_eq!(
        (last.2.shrinks, last.2.grows, last.2.grows_refused),
        (1, 1, 0),
        "{:?}",
        last.2
    );
    assert_eq!(last.1, own + lent, "every row live again");
}

#[test]
#[ignore = "needs the GPU and the real model"]
fn the_real_models_verify_rows_match_decode_across_3072() {
    // On the real model: the long golden prompt cut at 3,068 tokens, then its next 8 verified in one pass. The queries
    // see 3,069 to 3,076 positions, so decode's kernel changes at `ATTN_GQA_FROM` (3,072): the first four rows are
    // checked against `attn_decode` and the last four against `attn_decode_gqa`, while the verify step runs
    // `attn_decode_gqa_tokens` for all eight (`attn_tokens_kernel`). Each row is checked against the same tokens
    // decoded one at a time, with `prefill_real.rs`'s bounds, because the difference is the same kind (`gdn_seq`
    // against `gdn_step`): the rows' relative L2 averages at most 1e-3 and each is within 5e-2, and the committed state
    // within 1e-3. A row whose routes differ from decode's is a router near-tie: it and the rows after it are reported,
    // not judged. Bit equality is reported (`identity`), never asserted.
    const FORCED_MEAN: f64 = 1e-3;
    const FORCED_WORST: f64 = 5e-2;
    const STATE: f64 = 1e-3;
    const CUT: usize = 3068;
    const LEN: usize = 8;
    const CTX: usize = 4096;
    let path = std::env::var("ARTEMIS_MODEL").expect("set ARTEMIS_MODEL to the model's GGUF");
    let g = Gguf::open(std::path::Path::new(&path)).unwrap();
    let d = Dims::QWEN36;
    let all: Vec<u32> = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/golden/prompts/20-long-option-rs.tokens"
    ))
    .unwrap()
    .as_chunks::<4>()
    .0
    .iter()
    .map(|c| u32::from_le_bytes(*c))
    .collect();
    let (prompt, run) = (&all[..CUT], &all[CUT..CUT + LEN]);
    // Two sessions share the engine: its cache is sized for their contexts.
    let e = engine::Engine::open(
        &g,
        &d,
        &engine::Options {
            ctx: 3 * CTX,
            speculation: true,
            ..Default::default()
        },
    )
    .unwrap();
    let k = d.n_expert_used;
    let mut plain = e.session(CTX).unwrap();
    plain.prefill(&e, prompt).unwrap();
    let (mut want, mut want_routes) = (Vec::new(), Vec::new());
    for &t in run {
        want.push(plain.decode(&e, t).unwrap());
        want_routes.push(e.last_routes());
    }
    let mut s = e.session(CTX).unwrap();
    s.prefill(&e, prompt).unwrap();
    s.verify(&e, run).unwrap();
    let ids = e.last_prefill_routes();
    assert_eq!(ids.len(), d.n_layer * LEN * k);
    let rows: Vec<Vec<f32>> = (0..LEN).map(|i| s.row(&e, i).unwrap()).collect();
    s.commit(&e, LEN).unwrap();
    assert_eq!(
        (s.pos(), s.tokens(), s.kv_tokens()),
        (plain.pos(), plain.tokens(), plain.kv_tokens())
    );
    // The leading rows that routed as decode did, layer by layer, as sets.
    let routed_like_decode = |i: usize| {
        (0..d.n_layer).all(|l| {
            let mut a = ids[(l * LEN + i) * k..(l * LEN + i + 1) * k].to_vec();
            let mut b = want_routes[i][l * k..(l + 1) * k].to_vec();
            a.sort();
            b.sort();
            a == b
        })
    };
    let judged = (0..LEN).take_while(|&i| routed_like_decode(i)).count();
    if judged < LEN {
        eprintln!("row {judged} routed differently from decode (a near-tie): it and the rows after it are not judged");
    }
    assert!(judged >= 1, "not even row 0 routed as decode did");
    let errs: Vec<f64> = (0..judged)
        .map(|i| qwen36::ops::rel_l2(&rows[i], &want[i]))
        .collect();
    let mean = errs.iter().sum::<f64>() / judged as f64;
    let worst = errs
        .iter()
        .copied()
        .fold(0f64, |m, x| if x.is_nan() || x > m { x } else { m });
    let bits = (0..judged).all(|i| {
        rows[i].len() == want[i].len()
            && rows[i]
                .iter()
                .zip(&want[i])
                .all(|(a, b)| a.to_bits() == b.to_bits())
    });
    // The committed state is judged only when every row routed as decode did: after a near-tie the later tokens ran
    // different experts, and the state (the KV of the later positions in particular) is expected to differ.
    let (sv, sp) = (s.state(&e).unwrap(), plain.state(&e).unwrap());
    let state_judged = judged == LEN;
    let state = sv
        .layers
        .iter()
        .zip(&sp.layers)
        .flat_map(|(x, y)| {
            [
                qwen36::ops::rel_l2(&x.0, &y.0),
                qwen36::ops::rel_l2(&x.1, &y.1),
            ]
        })
        .fold(0f64, |m, x| if x.is_nan() || x > m { x } else { m });
    let state_bits = sv.layers.iter().zip(&sp.layers).all(|(x, y)| {
        let same = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits())
        };
        same(&x.0, &y.0) && same(&x.1, &y.1)
    });
    kernels::testing::identity("verify_rows", "the real model, 8 tokens across 3072", bits);
    kernels::testing::identity(
        "verify_state",
        "the real model, 8 tokens across 3072",
        state_bits,
    );
    let status = if judged >= 1
        && mean <= FORCED_MEAN
        && worst <= FORCED_WORST
        && (!state_judged || state <= STATE)
    {
        "pass"
    } else {
        "fail"
    };
    let num = |x: f64| {
        if x.is_finite() {
            serde_json::json!(x)
        } else {
            serde_json::json!(x.to_string())
        }
    };
    let verdict = serde_json::json!({
        "status": status,
        "input": "20-long-option-rs",
        "cut": CUT,
        "tokens": LEN,
        "rows_judged": judged,
        "mean": num(mean),
        "worst": num(worst),
        "state_judged": state_judged,
        "state": num(state),
        "rows_bit_equal": bits,
        "state_bit_equal": state_bits,
        "mean_bound": FORCED_MEAN,
        "worst_bound": FORCED_WORST,
        "state_bound": STATE,
    });
    eprintln!("{verdict:#}");
    if let Ok(dir) = std::env::var("ARTEMIS_EVIDENCE_DIR") {
        let p = std::path::Path::new(&dir).join("verify-rows.json");
        std::fs::write(&p, serde_json::to_vec_pretty(&verdict).unwrap())
            .unwrap_or_else(|e| panic!("writing {}: {e}", p.display()));
    }
    assert!(
        mean <= FORCED_MEAN,
        "the rows differ from decode's by {mean:.3e} on average, over {FORCED_MEAN:e}"
    );
    assert!(
        worst <= FORCED_WORST,
        "a row differs from decode's by {worst:.3e}, over {FORCED_WORST:e}"
    );
    if state_judged {
        assert!(
            state <= STATE,
            "the committed state differs by {state:.3e}, over {STATE:e}"
        );
    } else {
        eprintln!("the committed state differs by {state:.3e}: not judged after a near-tie");
    }
}

#[test]
#[ignore = "needs the GPU and the real model"]
fn the_real_models_mtp_drafts_agree_with_the_reference_at_eight_positions() {
    // The window rows come from prefill, the reference's from the session; the gating run is on the f32 prefill
    // kernels, and the default (split-f16) kernels' error is reported, not gated.
    mtp_against_the_reference(true, true);
    mtp_against_the_reference(false, false);
}

fn mtp_against_the_reference(f32_prefill: bool, gate: bool) {
    // On the real model: the long golden prompt cut at 512 tokens (prefilled, so its positions have
    // window rows), then 8 decode steps with a draft pass before each. Each pass's logits are held to `MtpReference`,
    // fed the rows the session has and the main model's hidden rows from the GPU: the ring's for the positions it decoded and
    // the prefill window's own final-norm rows (`Session::window_hidden`) for the prompt, so a miss is the MTP's
    // arithmetic alone. Relative L2 within decode's bound (1e-3, `gpu_engine.rs`),
    // and the drafted id equal to the reference's where its top two are more than `TIE_MARGIN` apart.
    use common::{rel, top2, MtpOracle, TIE_MARGIN};
    const BOUND: f64 = 1e-3;
    const CUT: usize = 512;
    const STEPS: usize = 8;
    const CTX: usize = 4096;
    let path = std::env::var("ARTEMIS_MODEL").expect("set ARTEMIS_MODEL to the model's GGUF");
    let g = Gguf::open(std::path::Path::new(&path)).unwrap();
    let d = Dims::QWEN36;
    let all: Vec<u32> = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/golden/prompts/20-long-option-rs.tokens"
    ))
    .unwrap()
    .as_chunks::<4>()
    .0
    .iter()
    .map(|c| u32::from_le_bytes(*c))
    .collect();
    let prompt = &all[..CUT];
    // The logits are compared with the whole vocabulary's: the full head, whatever the default is.
    let cfg = engine::SpecConfig {
        mtp_head_rows: 0,
        ..engine::SpecConfig::default()
    };
    let o = engine::MtpOptions::for_session(&cfg, CTX)
        .unwrap()
        .expect("the default sources include the MTP");
    let e = engine::Engine::open(
        &g,
        &d,
        &engine::Options {
            ctx: CTX,
            speculation: true,
            f32_prefill,
            mtp: Some(o),
            ..Default::default()
        },
    )
    .unwrap();
    let m = Model::open(&g, &d).unwrap();
    let mut s = e.session(CTX).unwrap();
    // The prefill window's final-norm rows, read back as the session computed them: the reference is fed those.
    s.capture_window_hidden(&e).unwrap();
    let mut logits = s.prefill(&e, prompt).unwrap();
    let mut oracle = MtpOracle::with_threads(&m, 16);
    let mut worst = 0f64;
    for step in 0..STEPS {
        let x = engine::greedy(&logits);
        let x_pos = s.pos();
        let (id, got) = s.mtp_draft_logits(&e, x).unwrap();
        let (want, _) = {
            let mut h = |q: usize| -> Vec<f32> {
                s.mtp_hidden(&e, q)
                    .unwrap()
                    .or_else(|| s.window_hidden(&e, q))
                    .unwrap_or_else(|| panic!("no hidden row of position {q}"))
            };
            oracle.expect(&s.mtp_rows(), s.tokens(), x_pos, x, &mut h)
        };
        let err = rel(&got, &want);
        let (top, gap) = top2(&want);
        eprintln!(
            "f32_prefill {f32_prefill}, step {step} at position {x_pos}: relative L2 {err:.3e}; draft {id}, reference {top} (lead {gap:.3}); {} rows",
            s.mtp_rows().len()
        );
        worst = worst.max(err);
        if gate {
            assert!(
                err < BOUND,
                "step {step}: the MTP's logits are off by {err:.3e}, over {BOUND:e}"
            );
            if gap > TIE_MARGIN {
                assert_eq!(id, top, "step {step}: the draft differs outside a near tie");
            }
        }
        logits = s.decode(&e, x).unwrap();
    }
    eprintln!("worst MTP logits error over {STEPS} steps: {worst:.3e}");
}
