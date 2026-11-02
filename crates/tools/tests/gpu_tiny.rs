//! The gate on the tiny model: the GPU engine, teacher-forced from the taps of the reference with an
//! f16 KV cache (`Rounding::F16Kv`), passes the GPU rule (`test2::gpu_forced`) on the committed tiny-model input. With
//! ARTEMIS_VERDICT set, the verdict is written there as JSON for tools/run-gpu-tests.ps1.
use artemis_tools::{
    tapdir::{Forcer, Recorder},
    test2::{gpu_forced, worst, Status},
};
use engine::Engine;
use gguf::Gguf;
use qwen36::{
    dims::Dims,
    model::Model,
    reference::{Reference, Rounding},
    testing::tiny_model,
};

const TOKENS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../testdata/test2/tiny/input.tokens"
);

#[test]
#[ignore = "needs the GPU"]
fn the_gpu_matches_the_exact_reference_tap_by_tap_on_the_tiny_model() {
    let d = Dims::tiny();
    let g = Gguf::from_bytes(tiny_model(1)).unwrap();
    let m = Model::open(&g, &d).unwrap();
    let toks: Vec<u32> = std::fs::read(TOKENS)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let mut r = Reference::with_options(&m, 4, Rounding::F16Kv);
    let mut reference = Recorder::default();
    for &t in &toks {
        r.step(t, &mut reference).unwrap();
    }
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    let mut s = e.session(toks.len()).unwrap();
    let mut f = Forcer::new(&reference.set, &d);
    for &t in &toks {
        s.decode_tapped(&e, t, &mut f).unwrap();
    }
    let v = gpu_forced(&reference.set, &f.rec.set, &d).unwrap();
    if let Ok(path) = std::env::var("ARTEMIS_VERDICT") {
        std::fs::write(&path, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
    }
    if let Some(w) = worst(&v) {
        eprintln!(
            "worst tap: {} at {:.3e} of a {:.3e} bound",
            w.tap, w.err, w.bound
        );
    }
    let failing: Vec<_> = v.rows.iter().filter(|r| !r.pass).collect();
    assert_eq!(
        v.status,
        Status::Pass,
        "failing taps: {failing:#?}\ntopk mismatches: {:?}\nties: {:?}\nmissing: {:?}",
        v.topk_mismatches,
        v.near_ties,
        v.missing
    );
}

/// `generate::run` with the speculating stepper over the real `Session::verify` and `commit`, on the
/// tiny model, against the plain loop. The verify step's DeltaNet layers are within tolerance of decode's, not bit
/// for bit, so the two runs may part at a near tie (test 3's rule: at most `TIE_MARGIN` apart); anywhere else they
/// must agree. Report only for the rest: the counts print.
#[test]
#[ignore = "needs the GPU"]
fn generate_with_speculation_follows_the_plain_loop_on_the_tiny_model() {
    use artemis_tools::generate::{self, Anyhow, Plain, Speculating, Stop};
    use engine::spec::{Drafter, SpecConfig, SpecState};
    use engine::{Options, SessionVerifier};

    /// Proposes the tokens after the ones generated so far.
    struct Script {
        tokens: Vec<u32>,
        primed: bool,
        seen: usize,
    }
    impl Drafter for Script {
        fn name(&self) -> &'static str {
            "script"
        }
        fn draft(&mut self, max: usize) -> Vec<u32> {
            self.tokens
                .iter()
                .skip(self.seen)
                .take(max)
                .copied()
                .collect()
        }
        fn observe(&mut self, tokens: &[u32]) {
            // The first call is the prompt; then each token generated, `x` first.
            if std::mem::replace(&mut self.primed, true) {
                self.seen += tokens.len();
            }
        }
    }

    let d = Dims::tiny();
    let g = Gguf::from_bytes(tiny_model(1)).unwrap();
    let e = Engine::open(
        &g,
        &d,
        &Options {
            ctx: 64,
            cache_slots: Some(0),
            speculation: true,
            ..Options::default()
        },
    )
    .unwrap();
    let prompt: Vec<u32> = (0..10u32)
        .map(|i| (i * 211 + 17) % d.n_vocab as u32)
        .collect();
    const TOKENS: usize = 32;
    let pick = |_: &[f32], top: u32| top;

    let mut s = e.session(64).unwrap();
    let logits = s.prefill(&e, &prompt).unwrap();
    let (plain, plain_stop) = generate::run(
        logits,
        TOKENS,
        64 - s.pos(),
        &[],
        pick,
        &mut Plain(|t| Ok(s.decode(&e, t)?)),
        |_| Ok(()),
    )
    .unwrap();

    let mut s = e.session(64).unwrap();
    let logits = s.prefill(&e, &prompt).unwrap();
    // A fixed draft length, and a drafter that proposes the plain run's own tokens: lookup could find no n-gram in
    // the tiny model's output and the run would never verify. Every step with a draft verifies four of them.
    let script: Vec<u32> = plain.iter().map(|s| s.token).collect();
    let mut state = SpecState::with_drafters(
        SpecConfig {
            max_draft: 4,
            fixed_draft: Some(4),
            ..SpecConfig::default()
        },
        Box::new(move || {
            Box::new(Script {
                tokens: script.clone(),
                primed: false,
                seen: 0,
            })
        }),
    );
    let room = 64 - s.pos();
    let (spec, spec_stop, delta) = {
        let mut stepper = Speculating::new(
            &mut state,
            Anyhow(SessionVerifier { e: &e, s: &mut s }),
            &prompt,
        );
        let (steps, stop) =
            generate::run(logits, TOKENS, room, &[], pick, &mut stepper, |_| Ok(())).unwrap();
        (steps, stop, stepper.delta)
    };
    eprintln!("speculation on the tiny model: {delta:?}");
    assert_eq!((plain_stop, spec_stop), (Stop::MaxTokens, Stop::MaxTokens));
    assert_eq!((plain.len(), spec.len()), (TOKENS, TOKENS));
    // Every row picked after the first is a committed position.
    assert_eq!(s.pos(), prompt.len() + TOKENS - 1);
    // The run made verify steps, and the drafter's tokens were the plain run's, so most were kept.
    assert!(delta.steps > 0 && delta.accepted > 0, "{delta:?}");
    // Identical, or parted at a near tie of the plain run's own top two.
    if let Some(i) = (0..TOKENS).find(|&i| plain[i].token != spec[i].token) {
        let margin = plain[i].logit - plain[i].second_logit;
        eprintln!("the runs part at step {i}, a margin of {margin}");
        assert!(
            margin <= artemis_tools::test3::TIE_MARGIN,
            "step {i}: {margin}"
        );
    }
}
