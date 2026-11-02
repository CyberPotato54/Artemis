use gguf::{Gguf, Value};
use qwen36::{
    dims::Dims,
    model::Model,
    quant::WType,
    testing::{byte_to_unicode, fnv1a64, tiny_model, tiny_vocab, TINY_SPECIALS},
};
use std::collections::HashSet;

#[test]
fn the_tiny_model_meets_the_contract() {
    let bytes = tiny_model(1);
    // About 10 MB: 16 experts per block at the real model's per-expert layout dominate.
    assert!(bytes.len() < 16 << 20, "{} bytes", bytes.len());
    let g = Gguf::from_bytes(bytes).unwrap();
    let m = Model::open(&g, &Dims::tiny()).unwrap();
    assert_eq!(m.layers[2].moe.down_exps.ty, WType::Q6K);
    assert_eq!(m.layers[0].moe.down_exps.ty, WType::Q5K);
    assert_eq!(m.mtp.layer.moe.router.ty, WType::Bf16);
    assert_eq!(
        g.meta("tokenizer.ggml.eos_token_id")
            .and_then(Value::as_u64),
        Some(506)
    );
}

#[test]
fn the_tiny_model_is_deterministic() {
    assert_eq!(fnv1a64(&tiny_model(1)), fnv1a64(&tiny_model(1)));
    assert_ne!(fnv1a64(&tiny_model(1)), fnv1a64(&tiny_model(2)));
}

#[test]
fn the_tiny_vocabulary_is_well_formed() {
    let v = tiny_vocab(512);
    assert_eq!((v.tokens.len(), v.types.len()), (512, 512));
    assert_eq!(v.merges.len(), 512 - 256 - TINY_SPECIALS.len());
    let unique: HashSet<&String> = v.tokens.iter().collect();
    assert_eq!(unique.len(), 512, "token strings are unique");
    let b2u = byte_to_unicode();
    assert_eq!(v.tokens[b' ' as usize], "Ġ");
    assert_eq!(b2u[b'a' as usize], 'a');
    for (i, m) in v.merges.iter().enumerate() {
        let (a, b) = m.split_once(' ').expect("a merge is two tokens");
        assert!(
            unique.contains(&a.to_string()) && unique.contains(&b.to_string()),
            "merge {i} {m:?}"
        );
        assert_eq!(v.tokens[256 + i], format!("{a}{b}"));
    }
    for (j, (s, t)) in TINY_SPECIALS.iter().enumerate() {
        assert_eq!((v.tokens[504 + j].as_str(), v.types[504 + j]), (*s, *t));
    }
}

#[test]
fn rng_is_the_same_everywhere() {
    let mut r = qwen36::testing::Rng::new(42);
    // splitmix64's first output for seed 42.
    assert_eq!(r.next_u64(), 0xBDD7_3226_2FEB_6E95);
    let x = r.uniform(-1.0, 1.0);
    assert!((-1.0..1.0).contains(&x));
}
