//! Test 8: token ids equal llama.cpp b11509's on the committed 1 MB corpus, with and without special-token
//! parsing. The expected ids come from `artemis-oracle tokenize`; the vocabulary from the committed extract.
use qwen36::tokenizer::{Tokenizer, Vocab};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tokenizer/");

fn read(file: &str) -> Vec<u8> {
    std::fs::read(format!("{ROOT}{file}")).unwrap_or_else(|e| panic!("{file}: {e}"))
}

fn ids(file: &str) -> Vec<u32> {
    read(file)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn check(parse_special: bool, expected: &str) {
    let tk = Tokenizer::new(&Vocab::from_extract(&read("qwen36-vocab.bin")).unwrap()).unwrap();
    let text = String::from_utf8(read("corpus.txt")).unwrap();
    let got = tk.encode(&text, parse_special);
    let want = ids(expected);
    let first = (0..got.len().min(want.len())).find(|&i| got[i] != want[i]);
    let first = first.or((got.len() != want.len()).then(|| got.len().min(want.len())));
    if let Some(i) = first {
        let (lo, hi) = (i.saturating_sub(8), i + 8);
        let (g, w) = (&got[lo..hi.min(got.len())], &want[lo..hi.min(want.len())]);
        panic!(
            "parse_special {parse_special}: first difference at token {i} ({} tokens vs llama.cpp's {})\n  ours:  {g:?} = {:?}\n  llama: {w:?} = {:?}",
            got.len(),
            want.len(),
            tk.decode(g),
            tk.decode(w)
        );
    }
}

#[test]
fn corpus_matches_llama_cpp_with_special_parsing() {
    check(true, "corpus.special.ids");
}

#[test]
fn corpus_matches_llama_cpp_without_special_parsing() {
    check(false, "corpus.plain.ids");
}
