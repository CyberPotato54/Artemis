//! Checks against the real model file. Run with ARTEMIS_MODEL set:
//! cargo test --release -p qwen36 --test real_model -- --ignored --test-threads=1
use gguf::Gguf;
use qwen36::{dims::Dims, model::Model, quant::WType};

pub fn model_path() -> std::path::PathBuf {
    std::env::var_os("ARTEMIS_MODEL")
        .expect("set ARTEMIS_MODEL to the Qwen3.6 GGUF")
        .into()
}

#[test]
#[ignore = "needs the model file"]
fn the_real_model_meets_the_contract() {
    let g = Gguf::open(&model_path()).unwrap();
    let m = Model::open(&g, &Dims::QWEN36).unwrap();
    assert_eq!(m.layers.len(), 40);
    assert_eq!((m.embd.ty, m.output.ty), (WType::Q8_0, WType::Q6K));
    for (l, layer) in m.layers.iter().enumerate() {
        let want = if [34, 38, 39].contains(&l) {
            WType::Q6K
        } else {
            WType::Q5K
        };
        assert_eq!(layer.moe.down_exps.ty, want, "layer {l} down");
        assert_eq!(layer.moe.gate_exps.ty, WType::Q4K, "layer {l} gate");
    }
    assert_eq!(m.mtp.layer.moe.router.ty, WType::Bf16);
}

#[test]
#[ignore = "needs the model file"]
fn the_reference_runs_on_the_real_model() {
    use qwen36::{reference::Reference, taps::NullTap};
    let g = Gguf::open(&model_path()).unwrap();
    let m = Model::open(&g, &Dims::QWEN36).unwrap();
    let mut r = Reference::new(&m);
    let start = std::time::Instant::now();
    for t in [785u32, 6722, 315] {
        let logits = r.step(t, &mut NullTap).unwrap();
        assert_eq!(logits.len(), 248_320);
        assert!(logits.iter().all(|x| x.is_finite()));
    }
    eprintln!("3 tokens in {:.1} s", start.elapsed().as_secs_f64());
}

#[test]
#[ignore = "needs the model file"]
fn the_committed_vocabulary_extract_matches_the_gguf() {
    use qwen36::tokenizer::Vocab;
    let g = Gguf::open(&model_path()).unwrap();
    let from_gguf = Vocab::from_gguf(&g).unwrap();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/tokenizer/qwen36-vocab.bin"
    );
    assert_eq!(
        Vocab::from_extract(&std::fs::read(path).unwrap()).unwrap(),
        from_gguf
    );
}

#[test]
#[ignore = "needs the model file"]
fn the_real_ggufs_template_and_vocabulary_match_the_pins() {
    use qwen36::pins::{template_digest, vocab_digest, CHAT_TEMPLATE_SHA256, VOCAB_SHA256};
    use qwen36::tokenizer::{chat_template, Vocab};
    let g = Gguf::open(&model_path()).unwrap();
    assert_eq!(
        template_digest(&chat_template(&g).unwrap()),
        CHAT_TEMPLATE_SHA256
    );
    assert_eq!(vocab_digest(&Vocab::from_gguf(&g).unwrap()), VOCAB_SHA256);
}

#[test]
#[ignore = "needs the model file"]
fn end_of_generation_matches_llama_cpp() {
    use qwen36::tokenizer::Vocab;
    let g = Gguf::open(&model_path()).unwrap();
    let ours = Vocab::from_gguf(&g).unwrap().eog();
    let json = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/tokenizer/eog.json"
    ))
    .unwrap();
    let list = json
        .split("\"eog\":[")
        .nth(1)
        .and_then(|r| r.split(']').next())
        .unwrap();
    let llama: Vec<u32> = list
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().unwrap())
        .collect();
    assert_eq!(ours, llama);
}
