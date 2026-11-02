//! Token bytes: each token's raw bytes, so a consumer can hold back a partial UTF-8 character itself.
use qwen36::tokenizer::{Tokenizer, Vocab};

const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/");

fn tokenizer() -> Tokenizer {
    let b = std::fs::read(format!("{TESTDATA}tokenizer/qwen36-vocab.bin")).unwrap();
    Tokenizer::new(&Vocab::from_extract(&b).unwrap()).unwrap()
}

#[test]
fn token_bytes_concatenate_to_the_text() {
    let tk = tokenizer();
    for text in [
        "plain ASCII",
        "café 日本語 🎉",
        "<|im_start|>user\nhi<|im_end|>",
        "\r\n\t  x",
    ] {
        let ids = tk.encode(text, true);
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| tk.token_bytes(i)).collect();
        assert_eq!(bytes, text.as_bytes(), "{text}");
    }
    // A token that ends inside a character has the partial bytes, not U+FFFD.
    let ids = tk.encode("🎉", false);
    assert!(ids.len() > 1 && tk.token_bytes(ids[0]).len() < 4);
}
