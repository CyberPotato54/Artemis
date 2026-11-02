//! The tokenizer, built from the GGUF's vocabulary the way llama.cpp b11509 reads it (semantics reference §8):
//! byte-level BPE, the `qwen35` pre-tokenizer, no BOS. With special parsing, CONTROL and USER_DEFINED tokens are
//! matched longest-first before BPE; without it, only USER_DEFINED tokens are.
use gguf::{Gguf, Value};
use std::sync::OnceLock;
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::BPE,
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
    },
    AddedToken, SplitDelimiterBehavior,
};

/// llama.cpp's `LLAMA_VOCAB_PRE_TYPE_QWEN35` regex, in its tokenizer.json form.
pub const QWEN35_PRETOKENIZER: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The longest run of whitespace characters `Tokenizer::try_encode` takes.
pub const MAX_WHITESPACE_RUN: usize = 262_144;

/// Why `Tokenizer::try_encode` refused a text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodeError {
    WhitespaceRun { chars: usize },
}

/// `n` in decimal with a comma between each group of three digits (`262144` gives `262,144`).
fn with_commas(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodeError::WhitespaceRun { chars } => write!(
                f,
                "the text holds a run of {chars} whitespace characters; the tokenizer takes at most {} in a row",
                with_commas(MAX_WHITESPACE_RUN)
            ),
        }
    }
}

impl std::error::Error for EncodeError {}

/// The length in characters of the longest run of `char::is_whitespace` characters (newlines included).
pub fn longest_whitespace_run(text: &str) -> usize {
    let (mut best, mut run) = (0, 0);
    for c in text.chars() {
        if c.is_whitespace() {
            run += 1;
            best = best.max(run);
        } else {
            run = 0;
        }
    }
    best
}

/// llama.cpp token types.
pub const CONTROL: i32 = 3;
pub const USER_DEFINED: i32 = 4;

/// The texts of the tokens that end generation, as llama.cpp b11509 derives them for this vocabulary: EOS, plus
/// the FIM pad, repo-name and file-separator tokens. `end_of_generation_matches_llama_cpp` checks it against
/// `artemis-oracle vocab`.
pub const EOG_TEXTS: &[&str] = &[
    "<|endoftext|>",
    "<|im_end|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
];

#[derive(Clone, Debug, PartialEq)]
pub struct Vocab {
    pub tokens: Vec<String>,
    pub types: Vec<i32>,
    /// `"a b"`: the two token strings a merge joins.
    pub merges: Vec<String>,
    pub eos: u32,
    pub bos: u32,
}

const EXTRACT_MAGIC: &[u8; 4] = b"AVX1";

impl Vocab {
    pub fn from_gguf(g: &Gguf) -> Result<Vocab, String> {
        let array = |key: &str| {
            g.meta(key)
                .and_then(Value::as_array)
                .ok_or(format!("{key} is missing"))
        };
        let strings = |key: &str| -> Result<Vec<String>, String> {
            array(key)?
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .ok_or(format!("{key} holds a non-string"))
                })
                .collect()
        };
        let types = array("tokenizer.ggml.token_type")?
            .iter()
            .map(|v| {
                v.as_i64()
                    .map(|t| t as i32)
                    .ok_or("tokenizer.ggml.token_type holds a non-integer".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let id = |key: &str| {
            g.meta(key)
                .and_then(Value::as_u64)
                .map(|v| v as u32)
                .ok_or(format!("{key} is missing"))
        };
        let v = Vocab {
            tokens: strings("tokenizer.ggml.tokens")?,
            types,
            merges: strings("tokenizer.ggml.merges")?,
            eos: id("tokenizer.ggml.eos_token_id")?,
            bos: id("tokenizer.ggml.bos_token_id")?,
        };
        if v.types.len() != v.tokens.len() {
            return Err(format!(
                "{} token types for {} tokens",
                v.types.len(),
                v.tokens.len()
            ));
        }
        Ok(v)
    }

    /// The compact form committed for CI (`testdata/tokenizer/qwen36-vocab.bin`), little-endian: magic `AVX1`;
    /// the token count, then each token's byte length, bytes and type; the merge count, then each merge's byte
    /// length and bytes; eos; bos.
    pub fn to_extract(&self) -> Vec<u8> {
        let mut out = EXTRACT_MAGIC.to_vec();
        let put = |out: &mut Vec<u8>, s: &str| {
            out.extend((s.len() as u32).to_le_bytes());
            out.extend(s.as_bytes());
        };
        out.extend((self.tokens.len() as u32).to_le_bytes());
        for (t, ty) in self.tokens.iter().zip(&self.types) {
            put(&mut out, t);
            out.extend(ty.to_le_bytes());
        }
        out.extend((self.merges.len() as u32).to_le_bytes());
        for m in &self.merges {
            put(&mut out, m);
        }
        out.extend(self.eos.to_le_bytes());
        out.extend(self.bos.to_le_bytes());
        out
    }

    pub fn from_extract(b: &[u8]) -> Result<Vocab, String> {
        let mut r = Reader { b, pos: 0 };
        if r.take(4)? != EXTRACT_MAGIC {
            return Err("not a vocabulary extract (bad magic)".into());
        }
        let n = r.u32()? as usize;
        let (mut tokens, mut types) = (Vec::new(), Vec::new());
        for _ in 0..n {
            tokens.push(r.string("a token")?);
            types.push(r.u32()? as i32);
        }
        let n = r.u32()? as usize;
        let mut merges = Vec::new();
        for _ in 0..n {
            merges.push(r.string("a merge")?);
        }
        Ok(Vocab {
            tokens,
            types,
            merges,
            eos: r.u32()?,
            bos: r.u32()?,
        })
    }

    /// Token ids whose text is in `EOG_TEXTS`.
    pub fn eog(&self) -> Vec<u32> {
        (0..self.tokens.len() as u32)
            .filter(|&i| EOG_TEXTS.contains(&self.tokens[i as usize].as_str()))
            .collect()
    }
}

/// The GGUF's chat template (`tokenizer.chat_template`), the Jinja source the model was trained with.
pub fn chat_template(g: &Gguf) -> Result<String, String> {
    g.meta("tokenizer.chat_template")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| "tokenizer.chat_template is missing".to_string())
}

/// Reads the extract, refusing truncation with the byte offset.
struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self
            .b
            .get(self.pos..self.pos + n)
            .ok_or_else(|| format!("vocabulary extract truncated at byte {}", self.pos))?;
        self.pos += n;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("take returned 4 bytes"),
        ))
    }

    fn string(&mut self, what: &str) -> Result<String, String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| format!("{what} is not UTF-8"))
    }
}

pub struct Tokenizer {
    /// Matches CONTROL and USER_DEFINED tokens before BPE.
    special: tokenizers::Tokenizer,
    /// Matches only USER_DEFINED tokens, as llama.cpp does with `parse_special = false`.
    plain: tokenizers::Tokenizer,
    /// Each token's bytes, by id.
    bytes: Vec<Vec<u8>>,
}

/// Whether GPT-2's `bytes_to_unicode` maps this byte to itself: `!`..`~`, `¡`..`¬`, `®`..`ÿ`.
fn byte_is_printable(b: u32) -> bool {
    (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b)
}

/// The byte a character of a byte-level BPE token string stands for: `bytes_to_unicode` inverted. The printable
/// bytes stand for themselves, the others for U+0100 onward in byte order.
fn byte_of(c: char) -> Option<u8> {
    /// Indexed by code point: the printable bytes sit below U+0100, the other 68 bytes at U+0100..U+0144.
    static TABLE: OnceLock<[Option<u8>; 0x200]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [None; 0x200];
        let mut next = 0x100;
        for b in 0..=255u32 {
            if byte_is_printable(b) {
                t[b as usize] = Some(b as u8);
            } else {
                t[next] = Some(b as u8);
                next += 1;
            }
        }
        t
    });
    table.get(c as usize).copied().flatten()
}

impl Tokenizer {
    pub fn new(v: &Vocab) -> Result<Tokenizer, String> {
        let bytes = v
            .tokens
            .iter()
            .zip(&v.types)
            .map(|(s, t)| {
                if *t == CONTROL || *t == USER_DEFINED {
                    Ok(s.as_bytes().to_vec())
                } else {
                    s.chars()
                        .map(|c| {
                            byte_of(c).ok_or_else(|| format!("token {s:?} holds {c:?}, not a byte"))
                        })
                        .collect()
                }
            })
            .collect::<Result<Vec<_>, String>>()?;
        let added = |kinds: &[i32]| -> Vec<AddedToken> {
            v.tokens
                .iter()
                .zip(&v.types)
                .filter(|(_, t)| kinds.contains(t))
                .map(|(s, t)| AddedToken::from(s.clone(), *t == CONTROL))
                .collect()
        };
        Ok(Tokenizer {
            special: build(v, added(&[CONTROL, USER_DEFINED]))?,
            plain: build(v, added(&[USER_DEFINED]))?,
            bytes,
        })
    }

    /// The bytes of one token: a CONTROL or USER_DEFINED token's text, or the bytes its byte-level string stands
    /// for. A token that ends inside a character has the partial bytes. Panics on an id outside the vocabulary.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        self.bytes
            .get(id as usize)
            .unwrap_or_else(|| panic!("token id {id} is outside the vocabulary"))
            .clone()
    }

    /// Encodes client-supplied text: a whitespace run over `MAX_WHITESPACE_RUN` characters is refused before
    /// `tokenizers` sees it, because its regex engine (Oniguruma) gives up on a run of about 12 million characters and
    /// `tokenizers` panics on that.
    pub fn try_encode(&self, text: &str, parse_special: bool) -> Result<Vec<u32>, EncodeError> {
        let chars = longest_whitespace_run(text);
        if chars > MAX_WHITESPACE_RUN {
            return Err(EncodeError::WhitespaceRun { chars });
        }
        Ok(self.encode(text, parse_special))
    }

    /// Encodes a fixed text or a test corpus. It panics on a whitespace run of about 12 million characters; client
    /// text goes through `try_encode`.
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        let tk = if parse_special {
            &self.special
        } else {
            &self.plain
        };
        tk.encode(text, false)
            .expect("encoding can't fail with a valid vocabulary")
            .get_ids()
            .to_vec()
    }

    /// Every token's text, special ones included. An id outside the vocabulary is dropped, as `tokenizers` does.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.special
            .decode(ids, false)
            .expect("decoding can't fail with a valid vocabulary")
    }
}

fn build(v: &Vocab, added: Vec<AddedToken>) -> Result<tokenizers::Tokenizer, String> {
    let vocab: tokenizers::models::bpe::Vocab = v
        .tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();
    let merges = v
        .merges
        .iter()
        .map(|m| {
            m.split_once(' ')
                .map(|(a, b)| (a.to_owned(), b.to_owned()))
                .ok_or_else(|| format!("merge {m:?} has no space"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| e.to_string())?;
    let mut tk = tokenizers::Tokenizer::new(bpe);
    let split = Split::new(
        SplitPattern::Regex(QWEN35_PRETOKENIZER.to_owned()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| e.to_string())?;
    tk.with_pre_tokenizer(Some(Sequence::new(vec![
        split.into(),
        ByteLevel::new(false, false, false).into(),
    ])));
    tk.with_decoder(Some(ByteLevelDecoder::default()));
    let (special, normal): (Vec<_>, Vec<_>) = added.into_iter().partition(|t| t.special);
    tk.add_special_tokens(&special);
    tk.add_tokens(&normal);
    Ok(tk)
}

#[cfg(test)]
mod tests {
    use super::byte_of;

    /// GPT-2's `bytes_to_unicode`, built forward: the printable bytes map to themselves, the rest to U+0100 on.
    #[test]
    fn byte_of_inverts_gpt2s_byte_table() {
        let mut next = 0x100u32;
        let mut seen = std::collections::HashSet::new();
        for b in 0..=255u32 {
            let c = if (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || b >= 0xae {
                char::from_u32(b).unwrap()
            } else {
                next += 1;
                char::from_u32(next - 1).unwrap()
            };
            assert!(seen.insert(c), "{c:?} is mapped twice");
            assert_eq!(byte_of(c), Some(b as u8), "byte {b:#04x} as {c:?}");
        }
        assert_eq!(next, 0x100 + 68);
        // Nothing past the table is a byte.
        assert_eq!(byte_of(char::from_u32(next).unwrap()), None);
        assert_eq!(byte_of('日'), None);
    }
}
