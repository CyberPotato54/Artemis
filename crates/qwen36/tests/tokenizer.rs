use gguf::Gguf;
use qwen36::{
    testing::{tiny_model, tiny_vocab},
    tokenizer::{Tokenizer, Vocab},
};

fn tiny() -> (Vocab, Tokenizer) {
    let g = Gguf::from_bytes(tiny_model(1)).unwrap();
    let v = Vocab::from_gguf(&g).unwrap();
    let t = Tokenizer::new(&v).unwrap();
    (v, t)
}

// The tiny vocabulary's specials: 504 <|endoftext|>, 505 <|im_start|>, 506 <|im_end|>, 507 <|vision_pad|>,
// 508 <tool_call>, 509 </tool_call>, 510 <think>, 511 </think>.

#[test]
fn the_vocabulary_comes_from_the_gguf() {
    let (v, _) = tiny();
    let tv = tiny_vocab(512);
    assert_eq!(
        (v.tokens, v.types, v.merges),
        (tv.tokens, tv.types, tv.merges)
    );
    assert_eq!((v.eos, v.bos), (506, 504));
}

#[test]
fn round_trips_any_text() {
    let (_, t) = tiny();
    for s in [
        "hello world",
        "The lighthouse keeper",
        "tabs\tand\r\nCRLF",
        "naïve café 東京 👍🏽",
        "   ",
        "<|im_end",
    ] {
        assert_eq!(t.decode(&t.encode(s, true)), s, "{s:?}");
        assert_eq!(t.decode(&t.encode(s, false)), s, "{s:?}");
    }
}

#[test]
fn empty_text_is_no_tokens() {
    let (_, t) = tiny();
    assert!(t.encode("", true).is_empty());
    assert!(t.encode("", false).is_empty());
}

#[test]
fn control_tokens_parse_only_with_special_parsing() {
    let (_, t) = tiny();
    let s = "<|im_start|>user\nhi<|im_end|>";
    let a = t.encode(s, true);
    assert_eq!((a.first(), a.last()), (Some(&505), Some(&506)));
    let b = t.encode(s, false);
    assert!(!b.contains(&505) && !b.contains(&506), "{b:?}");
}

#[test]
fn user_defined_tokens_parse_in_both_modes() {
    let (_, t) = tiny();
    for parse_special in [true, false] {
        let ids = t.encode("<think>x</think>", parse_special);
        assert_eq!(
            (ids.first(), ids.last()),
            (Some(&510), Some(&511)),
            "parse_special {parse_special}"
        );
    }
}

#[test]
fn partial_specials_are_plain_text() {
    let (_, t) = tiny();
    for s in ["<|im_end", "<|im_end|", "im_end|>", "<think"] {
        assert!(t.encode(s, true).iter().all(|&id| id < 504), "{s:?}");
    }
}

#[test]
fn merges_apply() {
    // The first merge of two single letters is one token.
    let (v, t) = tiny();
    let (i, m) = v
        .merges
        .iter()
        .enumerate()
        .find(|(_, m)| {
            m.split(' ')
                .all(|p| p.len() == 1 && p.chars().all(|c| c.is_ascii_alphabetic()))
        })
        .unwrap();
    let text = m.replace(' ', "");
    assert_eq!(t.encode(&text, true), vec![256 + i as u32]);
}

#[test]
fn the_extract_round_trips_and_refuses_truncation() {
    let (v, _) = tiny();
    let x = v.to_extract();
    assert_eq!(Vocab::from_extract(&x).unwrap(), v);
    for n in [0, 3, 4, 10, x.len() / 2, x.len() - 1] {
        assert!(Vocab::from_extract(&x[..n]).is_err(), "{n} bytes");
    }
}

#[test]
fn end_of_generation_tokens() {
    let (v, _) = tiny();
    assert_eq!(v.eog(), vec![504, 506]);
}

// A long run of whitespace trips Oniguruma's retry limit inside `tokenizers`. These use the real vocabulary.
mod whitespace_runs {
    use qwen36::tokenizer::{
        longest_whitespace_run, EncodeError, Tokenizer, Vocab, MAX_WHITESPACE_RUN,
    };

    const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/");

    fn real() -> Tokenizer {
        let b = std::fs::read(format!("{TESTDATA}tokenizer/qwen36-vocab.bin")).unwrap();
        Tokenizer::new(&Vocab::from_extract(&b).unwrap()).unwrap()
    }

    #[test]
    fn a_whitespace_run_past_the_cap_is_refused() {
        let tk = real();
        let text = format!("{}a", " ".repeat(MAX_WHITESPACE_RUN + 1));
        assert_eq!(
            tk.try_encode(&text, true),
            Err(EncodeError::WhitespaceRun { chars: 262_145 })
        );
        assert_eq!(
            tk.try_encode(&text, false),
            Err(EncodeError::WhitespaceRun { chars: 262_145 })
        );
        // A mixed run counts every character.
        let mixed = format!("x{}y", " \t\n\u{3000}".repeat(MAX_WHITESPACE_RUN / 4 + 1));
        assert_eq!(longest_whitespace_run(&mixed), MAX_WHITESPACE_RUN + 4);
        assert_eq!(
            tk.try_encode(&mixed, true),
            Err(EncodeError::WhitespaceRun {
                chars: MAX_WHITESPACE_RUN + 4
            })
        );
        assert_eq!(
            EncodeError::WhitespaceRun { chars: 262_145 }.to_string(),
            "the text holds a run of 262145 whitespace characters; the tokenizer takes at most 262,144 in a row"
        );
    }

    #[test]
    fn longest_whitespace_run_counts_runs() {
        assert_eq!(longest_whitespace_run(""), 0);
        assert_eq!(longest_whitespace_run("abc"), 0);
        assert_eq!(longest_whitespace_run("a  b   c \n\n"), 3);
        assert_eq!(longest_whitespace_run("  a\t\r\n\u{3000}b"), 4);
    }

    #[test]
    fn a_whitespace_run_at_the_cap_encodes() {
        let tk = real();
        let text = format!("{}a", " ".repeat(MAX_WHITESPACE_RUN));
        let ids = tk.try_encode(&text, true).unwrap();
        assert_eq!(tk.decode(&ids), text);
    }

    #[test]
    fn a_mixed_whitespace_run_at_the_cap_encodes() {
        let tk = real();
        // A newline then spaces: MAX_WHITESPACE_RUN whitespace characters in a row, the first a newline.
        let text = format!("a\n{}b", " ".repeat(MAX_WHITESPACE_RUN - 1));
        assert_eq!(longest_whitespace_run(&text), MAX_WHITESPACE_RUN);
        let ids = tk.try_encode(&text, true).unwrap();
        assert_eq!(tk.decode(&ids), text);
        // One more character is over.
        let over = format!("a\n{}b", " ".repeat(MAX_WHITESPACE_RUN));
        assert_eq!(
            tk.try_encode(&over, true),
            Err(EncodeError::WhitespaceRun {
                chars: MAX_WHITESPACE_RUN + 1
            })
        );
    }

    #[test]
    #[ignore = "slow: confirms N7"]
    fn the_raw_encoder_panics_on_twelve_million_spaces() {
        let tk = real();
        let text = format!("{}a", " ".repeat(12_000_000));
        let r = std::panic::catch_unwind(|| tk.encode(&text, true));
        assert!(r.is_err(), "the raw encoder did not panic");
    }
}
