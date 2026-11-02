//! Test 2's first rung, in CI: the reference on the tiny model against llama.cpp b11509's committed taps for it.
use artemis_tools::{
    tapdir::{self, Forcer},
    test2::{test2_forced, Status},
};
use gguf::Gguf;
use qwen36::{
    dims::Dims,
    model::Model,
    reference::{Reference, Rounding},
    testing::{fnv1a64, tiny_model},
};
use std::path::Path;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/test2/tiny");

#[test]
fn the_reference_matches_llama_cpp_on_the_tiny_model() {
    let dir = Path::new(DIR);
    let d = Dims::tiny();
    let bytes = tiny_model(1);
    let want = std::fs::read_to_string(dir.join("model.fnv")).unwrap();
    assert_eq!(
        format!("{:016x}", fnv1a64(&bytes)),
        want.trim(),
        "the tiny model changed; regenerate testdata/test2/tiny"
    );
    let g = Gguf::from_bytes(bytes).unwrap();
    let m = Model::open(&g, &d).unwrap();
    let toks: Vec<u32> = std::fs::read(dir.join("input.tokens"))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    // llama.cpp's own rounding, so the Vulkan-vs-CPU floor measures what's left.
    let cpu = tapdir::read(&dir.join("llama-cpu"), &d).unwrap();
    // llama.cpp's rounding, teacher-forced with its values at every tap: each operation is checked on llama.cpp's
    // own inputs.
    let mut r = Reference::with_options(&m, 4, Rounding::LlamaCpu);
    let mut forcer = Forcer::new(&cpu, &d);
    for &t in &toks {
        r.step(t, &mut forcer).unwrap();
    }
    let v = test2_forced(&cpu, &forcer.rec.set, &d).unwrap();
    if let Some(w) = artemis_tools::test2::worst(&v) {
        eprintln!(
            "worst tap: {} at {:.3e} of a {:.3e} bound (floor {:.3e})",
            w.tap, w.err, w.bound, w.floor
        );
    }
    let failing: Vec<_> = v.rows.iter().filter(|r| !r.pass).collect();
    assert_eq!(
        v.status,
        Status::Pass,
        "failing taps: {failing:#?}\ntopk mismatches: {:?}\nnear ties: {:?}\nmissing: {:?}",
        v.topk_mismatches,
        v.near_ties,
        v.missing
    );
}
