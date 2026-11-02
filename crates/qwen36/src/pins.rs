//! The chat template and vocabulary Artemis is tested with, pinned by digest.
//!
//! The server compares the GGUF it is given with these at start (`artemis serve`), so a file whose template or
//! vocabulary was edited is refused instead of served. The digests are of the committed copies in `testdata/`, which
//! the ignored real-model tests pin to the supported GGUF byte for byte.
use crate::sha256::{hex, sha256};
use crate::tokenizer::Vocab;

/// sha256 of `testdata/template/qwen36-chat-template.jinja`, the GGUF's `tokenizer.chat_template`.
pub const CHAT_TEMPLATE_SHA256: &str =
    "55d4931433fe502b794226ee7f4d206a6bdd436ac9f80eb7d8ebb4c639f9ea0c";

/// sha256 of `testdata/tokenizer/qwen36-vocab.bin`, `Vocab::to_extract`'s output for the supported GGUF.
pub const VOCAB_SHA256: &str = "f2d0daceeb491ad8b4bac90a09073b5e682951b6f5c8fe329a3648254f8749d5";

/// The lower-case hex sha256 of a chat template's text.
pub fn template_digest(source: &str) -> String {
    hex(&sha256(source.as_bytes()))
}

/// The lower-case hex sha256 of a vocabulary's extract (`Vocab::to_extract`).
pub fn vocab_digest(v: &Vocab) -> String {
    hex(&sha256(&v.to_extract()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn committed(rel: &str) -> Vec<u8> {
        let path = format!("{}/../../testdata/{rel}", env!("CARGO_MANIFEST_DIR"));
        std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    #[test]
    fn the_committed_template_matches_the_pin() {
        let bytes = committed("template/qwen36-chat-template.jinja");
        let source = String::from_utf8(bytes).unwrap();
        assert_eq!(template_digest(&source), CHAT_TEMPLATE_SHA256);
    }

    #[test]
    fn the_committed_vocabulary_matches_the_pin() {
        let bytes = committed("tokenizer/qwen36-vocab.bin");
        assert_eq!(hex(&sha256(&bytes)), VOCAB_SHA256);
        let v = Vocab::from_extract(&bytes).unwrap();
        assert_eq!(vocab_digest(&v), VOCAB_SHA256);
    }
}
