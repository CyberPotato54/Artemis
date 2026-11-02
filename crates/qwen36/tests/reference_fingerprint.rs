//! The reference's output bits on the tiny model, pinned. Its operations later moved into `qwen36::ops`; these are
//! the fingerprints from before the move, so the move provably changed no arithmetic in either rounding mode.
use gguf::Gguf;
use qwen36::{
    dims::Dims,
    model::Model,
    reference::{Reference, Rounding},
    taps::{Tap, TapName},
    testing::{fnv1a64, tiny_model},
};
use std::collections::BTreeMap;

#[derive(Default)]
struct Collect(BTreeMap<TapName, Vec<f32>>);

impl Tap for Collect {
    fn tap(&mut self, name: TapName, _pos: u32, data: &[f32]) {
        self.0.entry(name).or_default().extend_from_slice(data);
    }
}

fn fingerprint(rounding: Rounding) -> (u64, usize) {
    let d = Dims::tiny();
    let g = Gguf::from_bytes(tiny_model(1)).unwrap();
    let m = Model::open(&g, &d).unwrap();
    let mut r = Reference::with_options(&m, 4, rounding);
    let mut c = Collect::default();
    for t in [5u32, 77, 300, 12, 499, 0, 256, 31] {
        r.step(t, &mut c).unwrap();
    }
    let bytes: Vec<u8> =
        c.0.values()
            .flat_map(|v| v.iter().flat_map(|x| x.to_le_bytes()))
            .collect();
    (fnv1a64(&bytes), c.0.len())
}

#[test]
fn the_exact_reference_is_bit_for_bit_plan_1as() {
    assert_eq!(fingerprint(Rounding::Exact), (0xc6fb_c701_8079_7b80, 94));
}

#[test]
fn the_llama_rounding_reference_is_bit_for_bit_plan_1as() {
    assert_eq!(fingerprint(Rounding::LlamaCpu), (0x6926_6926_4bdf_ca49, 94));
}
