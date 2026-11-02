//! Step timings on the tiny model: every command timed under its layer, no result changed, and
//! a failed step still readable.
use engine::{Engine, Error, Options, SpanKind, StepKind, TAG_EMBED, TAG_OUTPUT};
use gguf::Gguf;
use kernels::dispatch::WHERE_CPU;
use qwen36::{dims::Dims, testing::tiny_model};
use std::collections::BTreeSet;

fn tiny() -> (Gguf, Dims) {
    (Gguf::from_bytes(tiny_model(1)).unwrap(), Dims::tiny())
}

/// The f32s are the same bits (a +0 and a -0 differ, a NaN equals itself).
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

const TOKENS: [u32; 8] = [10, 300, 7, 506, 42, 42, 1, 511];

#[test]
#[ignore = "needs the GPU"]
fn timings_change_no_logit_and_cover_every_layer() {
    let (g, d) = tiny();
    let e = Engine::open(
        &g,
        &d,
        &Options {
            cache_slots: Some(6),
            ..Options::default()
        },
    )
    .unwrap();
    // A 5-token prompt in one prefill chunk, then three decoded tokens.
    let run = |on: bool| {
        e.set_timing(on);
        let mut s = e.session(32).unwrap();
        let first = s.prefill(&e, &TOKENS[..5]).unwrap();
        let rest: Vec<Vec<f32>> = TOKENS[5..]
            .iter()
            .map(|&t| s.decode(&e, t).unwrap())
            .collect();
        (first, rest, e.take_timings())
    };
    let (a1, a2, none) = run(false);
    assert!(none.is_empty(), "timings off record nothing");
    let (b1, b2, steps) = run(true);
    assert!(
        same_bits(&a1, &b1)
            && a2.len() == b2.len()
            && a2.iter().zip(&b2).all(|(x, y)| same_bits(x, y)),
        "timings changed the logits"
    );
    let kinds: Vec<(StepKind, usize, usize)> =
        steps.iter().map(|s| (s.kind, s.pos, s.tokens)).collect();
    assert_eq!(
        kinds,
        [
            (StepKind::Prefill, 0, 5),
            (StepKind::Decode, 5, 1),
            (StepKind::Decode, 6, 1),
            (StepKind::Decode, 7, 1)
        ]
    );
    let want: BTreeSet<u32> = (0..d.n_layer as u32)
        .chain([TAG_EMBED, TAG_OUTPUT])
        .collect();
    for s in &steps {
        let tags: BTreeSet<u32> = s.gpu.iter().map(|t| t.tag).collect();
        assert_eq!(tags, want, "{:?} at {}", s.kind, s.pos);
        assert!(
            s.gpu.iter().any(|t| t.name.starts_with("experts_")),
            "{:?} at {}: no expert kernel timed",
            s.kind,
            s.pos
        );
        assert!(s.gpu.iter().map(|t| t.ns).sum::<u64>() > 0);
        assert!(s.cpu.submit_us > 0, "{:?}", s.cpu);
    }
    assert!(
        steps.iter().map(|s| s.cpu.record_us).sum::<u64>() > 0,
        "no record time: {:?}",
        steps.iter().map(|s| s.cpu).collect::<Vec<_>>()
    );
    assert!(
        e.take_timings().is_empty(),
        "taking the timings empties the list"
    );
    e.set_timing(false);
}

#[test]
#[ignore = "needs the GPU"]
fn a_failed_step_with_timings_on_reports_its_error_and_keeps_its_entry() {
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    for ex in 0..d.n_expert {
        e.set_slot_entry(2, ex, 0, kernels::dispatch::WHERE_CPU)
            .unwrap();
    }
    e.set_timing(true);
    let mut s = e.session(4).unwrap();
    match s.decode(&e, 10) {
        Err(Error::ExpertOnCpu {
            pos: 0, layer: 2, ..
        }) => {}
        other => panic!("expected ExpertOnCpu at layer 2, got {other:?}"),
    }
    let steps = e.take_timings();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].kind, StepKind::Decode);
    assert!(
        !steps[0].gpu.is_empty(),
        "the failed step's commands were timed"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn the_output_head_lands_in_its_own_chunks_entry() {
    let (g, d) = tiny();
    let e = Engine::open(
        &g,
        &d,
        &Options {
            cache_slots: Some(6),
            prefill_chunk: Some(3),
            ..Options::default()
        },
    )
    .unwrap();
    e.set_timing(true);
    let mut s = e.session(32).unwrap();
    s.prefill(&e, &TOKENS[..5]).unwrap();
    let steps = e.take_timings();
    e.set_timing(false);
    let kinds: Vec<(StepKind, usize, usize)> =
        steps.iter().map(|s| (s.kind, s.pos, s.tokens)).collect();
    assert_eq!(
        kinds,
        [(StepKind::Prefill, 0, 3), (StepKind::Prefill, 3, 2)]
    );
    let layers: BTreeSet<u32> = (0..d.n_layer as u32).chain([TAG_EMBED]).collect();
    let tags = |i: usize| -> BTreeSet<u32> { steps[i].gpu.iter().map(|t| t.tag).collect() };
    assert_eq!(tags(0), layers, "the first chunk has no output head");
    let mut last = layers;
    last.insert(TAG_OUTPUT);
    assert_eq!(
        tags(1),
        last,
        "the output head is in the last chunk's entry"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_refused_prefill_with_timings_on_keeps_its_entry() {
    let (g, d) = tiny();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    for ex in 0..d.n_expert {
        e.set_slot_entry(2, ex, 0, WHERE_CPU).unwrap();
    }
    e.set_timing(true);
    let mut s = e.session(8).unwrap();
    match s.prefill(&e, &TOKENS[..3]) {
        Err(Error::PrefillExpertOnCpu { .. }) => {}
        other => panic!("expected PrefillExpertOnCpu, got {other:?}"),
    }
    let steps = e.take_timings();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].kind, StepKind::Prefill);
    assert!(
        !steps[0].gpu.is_empty(),
        "the failed chunk's commands were timed"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_nan_step_with_timings_on_reports_its_error_and_keeps_its_entry() {
    // Token 511's embedding gets a NaN scale, so its logits are NaN (as in gpu_engine's NaN test).
    let mut bytes = tiny_model(1);
    let d = Dims::tiny();
    let off = {
        let g = Gguf::from_bytes(bytes.clone()).unwrap();
        let t = g.tensor("token_embd.weight").unwrap();
        t.offset as usize + 511 * qwen36::quant::WType::Q8_0.row_bytes(d.n_embd)
    };
    bytes[off..off + 2].copy_from_slice(&0x7E00u16.to_le_bytes());
    let g = Gguf::from_bytes(bytes).unwrap();
    let e = Engine::from_gguf(&g, &d, None).unwrap();
    e.set_timing(true);
    let mut s = e.session(4).unwrap();
    match s.decode(&e, 511) {
        Err(Error::NonFinite { pos: 0, .. }) => {}
        other => panic!("expected NonFinite at position 0, got {other:?}"),
    }
    let steps = e.take_timings();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].kind, StepKind::Decode);
    assert!(
        !steps[0].gpu.is_empty(),
        "the failed step's commands were timed"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_staged_steps_timings_hold_its_copies_beside_its_layers() {
    // With timings on, a staged prefill step records a compute span per submit and a
    // copy span per layer that staged experts (every layer of the tiny model at 6 slots), filed under its layer; each
    // layer's compute starts after its copy ends, because it waited for it. The logits don't change.
    let g = Gguf::from_bytes(tiny_model(1)).unwrap();
    let d = Dims::tiny();
    let opts = Options {
        cache_slots: Some(6),
        prefill_chunk: Some(8),
        stage_from: Some(1),
        ..Options::default()
    };
    let (e, plain) = (
        Engine::open(&g, &d, &opts).unwrap(),
        Engine::open(&g, &d, &opts).unwrap(),
    );
    e.set_timing(true);
    let toks: Vec<u32> = (0..16u32)
        .map(|i| (i * 211 + 17) % d.n_vocab as u32)
        .collect();
    let (mut a, mut b) = (e.session(32).unwrap(), plain.session(32).unwrap());
    assert!(a.prefill(&e, &toks).unwrap() == b.prefill(&plain, &toks).unwrap());
    let steps = e.take_timings();
    let prefill: Vec<_> = steps
        .iter()
        .filter(|s| s.kind == StepKind::Prefill)
        .collect();
    assert_eq!(prefill.len(), 2, "two chunks of eight");
    for s in prefill {
        let compute: Vec<_> = s
            .spans
            .iter()
            .filter(|x| x.kind == SpanKind::Compute)
            .collect();
        let copies: Vec<_> = s
            .spans
            .iter()
            .filter(|x| x.kind == SpanKind::Copy)
            .collect();
        let mut own: Vec<u32> = compute
            .iter()
            .map(|k| k.tag)
            .filter(|&t| t != TAG_OUTPUT)
            .collect();
        own.sort();
        assert_eq!(
            own,
            (0..d.n_layer as u32).collect::<Vec<_>>(),
            "one compute span per layer: {:?}",
            s.spans
        );
        let mut layers: Vec<u32> = copies.iter().map(|c| c.tag).collect();
        layers.sort();
        assert_eq!(
            layers,
            (0..d.n_layer as u32).collect::<Vec<_>>(),
            "{:?}",
            s.spans
        );
        for c in &copies {
            assert!(c.start_ns <= c.end_ns, "{c:?}");
            let k = compute
                .iter()
                .find(|k| k.tag == c.tag)
                .expect("the layer's compute span");
            assert!(
                k.start_ns >= c.end_ns,
                "layer {}'s compute began at {} before its copy ended at {}",
                c.tag,
                k.start_ns,
                c.end_ns
            );
        }
    }
}
