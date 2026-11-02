//! Swaps that overlap steps: every swap's copy first fills its slot with f16 NaNs, so a step
//! that read a slot mid-copy would produce NaN logits and fail. With precise waits no step waits for a copy it doesn't
//! read; the logits must still be bit for bit those of an engine reading every expert from host memory.
use engine::{Engine, Error, Options};
use gguf::Gguf;
use kernels::dispatch::{WHERE_HOST, WHERE_VRAM};
use qwen36::{
    dims::Dims,
    taps::{Tap, TapName},
    testing::tiny_model,
};

fn tiny() -> (Gguf, Dims) {
    (Gguf::from_bytes(tiny_model(1)).unwrap(), Dims::tiny())
}

fn swapping(g: &Gguf, d: &Dims, seed: u64, chunk: Option<usize>) -> Engine {
    Engine::open(
        g,
        d,
        &Options {
            cache_slots: Some(6),
            force_swap_seed: Some(seed),
            poison_swaps: true,
            prefill_chunk: chunk,
            ..Options::default()
        },
    )
    .unwrap()
}

#[test]
#[ignore = "needs the GPU"]
fn swaps_that_overlap_steps_never_change_a_logit() {
    let (g, d) = tiny();
    let plain = Engine::from_gguf(&g, &d, None).unwrap();
    let e = swapping(&g, &d, 3, None);
    let (mut a, mut b) = (plain.session(320).unwrap(), e.session(320).unwrap());
    for p in 0..300u32 {
        let t = (p * 37 + 5) % d.n_vocab as u32;
        assert!(
            b.decode(&e, t).unwrap() == a.decode(&plain, t).unwrap(),
            "position {p}: the logits differ"
        );
    }
    let st = e.stats();
    eprintln!(
        "steps prepared while a swap's copy was in flight: {} of {}; {st:?}",
        st.steps_overlapping_copies, st.steps
    );
    assert_eq!(st.swaps_started, 300, "one swap starts every step");
    assert!(
        st.steps_overlapping_copies > 0,
        "no step was prepared while a swap's copy was in flight: {st:?}"
    );
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
}

#[test]
#[ignore = "needs the GPU"]
fn prefill_with_swaps_overlapping_its_layers_changes_no_logit() {
    // Chunks of 7 over 60 tokens: 9 steps, each starting a swap whose copy may land while the chunk's layers run.
    let (g, d) = tiny();
    let plain = Engine::open(
        &g,
        &d,
        &Options {
            cache_slots: Some(0),
            prefill_chunk: Some(7),
            ..Options::default()
        },
    )
    .unwrap();
    let e = swapping(&g, &d, 5, Some(7));
    let tokens: Vec<u32> = (0..60u32)
        .map(|i| (i * 53 + 11) % d.n_vocab as u32)
        .collect();
    let all = |e: &Engine| {
        let mut s = e.session(64).unwrap();
        let mut out = Vec::new();
        s.prefill_each(e, &tokens, 0, |_, l| {
            out.push(l.to_vec());
            Ok::<(), Error>(())
        })
        .unwrap();
        out
    };
    assert!(all(&e) == all(&plain), "the logits differ");
    let st = e.stats();
    eprintln!(
        "steps prepared while a swap's copy was in flight: {} of {}; {st:?}",
        st.steps_overlapping_copies, st.steps
    );
    assert_eq!(st.swaps_started, 9, "one swap per chunk");
}

/// Forces the embedding tap with a value of the wrong length, so a step fails while recording, after `prepare_step`
/// has sent out the step's swap copies.
struct WrongEmbedding;

impl Tap for WrongEmbedding {
    fn tap(&mut self, _: TapName, _: u32, _: &[f32]) {}
    fn forced(&self, name: TapName, _: u32) -> Option<&[f32]> {
        matches!(name, TapName::Embd).then_some(&[0.0][..])
    }
}

#[test]
#[ignore = "needs the GPU"]
fn an_engine_dropped_with_a_copy_in_flight_waits_for_it() {
    // Each round's third step fails while recording, just after its prepare_step sent the second
    // step's swap copy; the engine is dropped at once. Drop must wait for the copy before freeing the slots and the
    // pinned copies, so the next engine runs on a healthy device. The validation run catches a buffer freed
    // in use.
    let (g, d) = tiny();
    for seed in 0..5 {
        let e = swapping(&g, &d, seed, None);
        let mut s = e.session(8).unwrap();
        s.decode(&e, 10).unwrap();
        s.decode(&e, 300).unwrap();
        let before = e.stats().steps_overlapping_copies;
        assert!(matches!(
            s.decode_tapped(&e, 7, &mut WrongEmbedding),
            Err(Error::BadForce { .. })
        ));
        drop(s);
        // The failing step's prepare_step must have left a copy in flight right before the drop.
        let after = e.stats().steps_overlapping_copies;
        eprintln!("round {seed}: steps_overlapping_copies {before} -> {after}");
        assert!(
            after > before,
            "round {seed}: steps_overlapping_copies did not rise across the failing step ({before} -> {after}); no copy in flight"
        );
        drop(e);
    }
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    e.session(4).unwrap().decode(&e, 10).unwrap();
}
