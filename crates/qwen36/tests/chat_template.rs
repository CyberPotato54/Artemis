//! The committed copy of the chat template is the GGUF's own, byte for byte. Run with ARTEMIS_MODEL set
//! (it reads the header only): cargo test -p qwen36 --test chat_template -- --ignored
use gguf::Gguf;
use qwen36::tokenizer::chat_template;

const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/");

#[test]
#[ignore = "needs ARTEMIS_MODEL"]
fn the_committed_template_is_the_ggufs() {
    let model = std::env::var_os("ARTEMIS_MODEL").expect("set ARTEMIS_MODEL to the Qwen3.6 GGUF");
    let g = Gguf::open(std::path::Path::new(&model)).unwrap();
    let committed =
        std::fs::read(format!("{TESTDATA}template/qwen36-chat-template.jinja")).unwrap();
    assert_eq!(chat_template(&g).unwrap().as_bytes(), committed.as_slice());
}
