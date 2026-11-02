//! Streaming text token by token, with the real vocabulary: decoding only the tokens since the last complete character
//! gives byte for byte what decoding every token so far gives, with Japanese and emoji split across tokens.
use qwen36::stream::TextStream;
use qwen36::tokenizer::{Tokenizer, Vocab};

const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/");

fn tokenizer() -> Tokenizer {
    let b = std::fs::read(format!("{TESTDATA}tokenizer/qwen36-vocab.bin")).unwrap();
    Tokenizer::new(&Vocab::from_extract(&b).unwrap()).unwrap()
}

/// Streams `ids` one token at a time both ways; the outputs must match at every step, and at the end.
fn same_both_ways(tk: &Tokenizer, ids: &[u32]) -> (Vec<String>, String) {
    let (mut old, mut new) = (TextStream::new(), TextStream::new());
    let mut steps = Vec::new();
    for n in 1..=ids.len() {
        let a = old.next(&tk.decode(&ids[..n])).to_string();
        let b = new.next_tokens(&ids[..n], |t| tk.decode(t));
        assert_eq!(a, b, "after {n} tokens");
        steps.push(b);
    }
    let a = old.finish(&tk.decode(ids)).to_string();
    let b = new.finish_tokens(ids, |t| tk.decode(t));
    assert_eq!(a, b, "the rest");
    (steps, b)
}

#[test]
fn incremental_decoding_streams_the_same_bytes() {
    let tk = tokenizer();
    for text in [
        "Hello, world. A plain ASCII line.",
        "日本語のテキストをストリームする。こんにちは世界！",
        "Mixed: café, naïve, 🎉🚀👨\u{200d}👩\u{200d}👧 and 漢字 done.",
        "🙂",
    ] {
        let ids = tk.encode(text, false);
        let (steps, rest) = same_both_ways(&tk, &ids);
        let whole: String = steps.concat() + &rest;
        assert_eq!(whole, tk.decode(&ids), "{text}");
        // A run that stops in the middle of a character prints the replacement at the end, as before.
        if ids.len() > 1 {
            same_both_ways(&tk, &ids[..ids.len() - 1]);
        }
    }
    // The emoji and the Japanese really are split across tokens: some step prints nothing.
    let ids = tk.encode("🎉🚀日本語", false);
    let (steps, _) = same_both_ways(&tk, &ids);
    assert!(steps.iter().any(String::is_empty), "{steps:?}");
}
