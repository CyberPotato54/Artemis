//! The tiny synthetic model: a valid `qwen35moe` GGUF of a few MB with random but well-formed weights and
//! a small working BPE vocabulary. CI runs the loader and the reference on it; llama.cpp loads it too, which gives
//! test 2 a cheap first rung. Test-only: `artemis` never opens a model with `Dims::tiny()`.
use crate::{
    contract::{expected_meta, expected_tensors},
    dims::Dims,
    quant::WType,
};
use gguf::{write::Writer, Value};
use half::{bf16, f16};
use std::collections::{BTreeMap, HashSet};

/// splitmix64: the same numbers on every platform.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [lo, hi), from 24 random bits.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / 16_777_216.0)
    }

    pub fn fill(&mut self, out: &mut [u8]) {
        for chunk in out.chunks_mut(8) {
            let v = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
    }
}

/// FNV-1a, 64-bit: a fingerprint for checking that the tiny model hasn't changed.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xCBF2_9CE4_8422_2325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01B3)
    })
}

/// GPT-2's byte-to-character map, which byte-level BPE vocabularies are written in.
pub fn byte_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .map(u32::from)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, c) in bs.iter().zip(&cs) {
        out[*b as usize] = char::from_u32(*c).expect("a valid char");
    }
    out
}

/// The tiny vocabulary's special tokens, after the bytes and merges: text and llama.cpp token type
/// (3 CONTROL, 4 USER_DEFINED), as in the real vocabulary.
pub const TINY_SPECIALS: [(&str, i32); 8] = [
    ("<|endoftext|>", 3),
    ("<|im_start|>", 3),
    ("<|im_end|>", 3),
    ("<|vision_pad|>", 3),
    ("<tool_call>", 4),
    ("</tool_call>", 4),
    ("<think>", 4),
    ("</think>", 4),
];

const TRAINING_TEXT: &str = "The quick brown fox jumps over the lazy dog. The lighthouse keeper climbed the \
spiral stairs each evening, counting the steps as his father had taught him, and the lamp turned slowly above the \
dark water. When the storm came in from the west, the ships that were still at sea could see the light long before \
they could hear the bell. In the morning the harbor was quiet again, and the fishermen mended their nets on the \
stones while the gulls argued over what the tide had left behind.\n\
fn main() {\n    let numbers: Vec<u32> = (0..10).collect();\n    let total: u32 = numbers.iter().sum();\n    \
println!(\"total = {}\", total);\n}\n\
def tokenize(text):\n    return [token for token in text.split() if token]\n\
for (int i = 0; i < count; i++) { sum += values[i]; }\n\
The weather in Paris is mild in spring, warm in summer and cool in the autumn evenings. Tourists walk along the \
river and the old streets, and the cafes fill up as soon as the sun comes out. There is always another museum to \
visit, another bridge to cross and another bakery to try before the day is over.\n";

pub struct TinyVocab {
    pub tokens: Vec<String>,
    pub types: Vec<i32>,
    pub merges: Vec<String>,
}

/// Split as a byte-level pre-tokenizer roughly would: each space or newline starts a new word.
fn split_words(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if i > start && (c == ' ' || c == '\n') {
            out.push(&text[start..i]);
            start = i;
        }
    }
    out.push(&text[start..]);
    out
}

/// 256 byte tokens, then merges learned from `TRAINING_TEXT`, then `TINY_SPECIALS`: `n_vocab` tokens in all.
pub fn tiny_vocab(n_vocab: usize) -> TinyVocab {
    let b2u = byte_to_unicode();
    let mut tokens: Vec<String> = (0..256).map(|b| b2u[b].to_string()).collect();
    let mut types = vec![1i32; 256];
    let n_merges = n_vocab - 256 - TINY_SPECIALS.len();
    let mut words: Vec<Vec<String>> = split_words(TRAINING_TEXT)
        .into_iter()
        .map(|w| w.bytes().map(|b| b2u[b as usize].to_string()).collect())
        .collect();
    let mut known: HashSet<String> = tokens.iter().cloned().collect();
    let mut merges = Vec::new();
    while merges.len() < n_merges {
        let mut counts: BTreeMap<(String, String), usize> = BTreeMap::new();
        for w in &words {
            for p in w.windows(2) {
                *counts.entry((p[0].clone(), p[1].clone())).or_default() += 1;
            }
        }
        // Most frequent first; ties go to the smallest pair, so the vocabulary is the same everywhere. A pair whose
        // result already exists is skipped, so token strings stay unique.
        let best = counts
            .into_iter()
            .filter(|((a, b), _)| !known.contains(&format!("{a}{b}")))
            .max_by(|(pa, ca), (pb, cb)| ca.cmp(cb).then(pb.cmp(pa)));
        let Some(((a, b), _)) = best else { break };
        let joined = format!("{a}{b}");
        for w in &mut words {
            let mut i = 0;
            while i + 1 < w.len() {
                if w[i] == a && w[i + 1] == b {
                    w[i] = joined.clone();
                    w.remove(i + 1);
                }
                i += 1;
            }
        }
        merges.push(format!("{a} {b}"));
        known.insert(joined.clone());
        tokens.push(joined);
        types.push(1);
    }
    assert_eq!(
        merges.len(),
        n_merges,
        "the training text is too small for {n_merges} merges"
    );
    for (s, t) in TINY_SPECIALS {
        tokens.push(s.to_string());
        types.push(t);
    }
    TinyVocab {
        tokens,
        types,
        merges,
    }
}

/// The type each tensor gets, mirroring the real file (semantics reference §7), with Q6_K down in layer 2 so the
/// tiny model has every expert type.
fn tiny_type(name: &str, d: &Dims) -> WType {
    let mtp = name.starts_with(&format!("blk.{}.", d.n_layer));
    if name.ends_with("ffn_gate_inp.weight") || name.ends_with("ffn_gate_inp_shexp.weight") {
        return if mtp { WType::Bf16 } else { WType::F32 };
    }
    if name.ends_with("ffn_down_exps.weight") {
        return if name.starts_with("blk.2.") {
            WType::Q6K
        } else {
            WType::Q5K
        };
    }
    if name.ends_with("_exps.weight") {
        return WType::Q4K;
    }
    if name == "output.weight" {
        return WType::Q6K;
    }
    const F32_SUFFIXES: [&str; 6] = [
        "norm.weight",
        "ssm_alpha.weight",
        "ssm_beta.weight",
        "ssm_conv1d.weight",
        "ssm_dt.bias",
        "ssm_a",
    ];
    if F32_SUFFIXES.iter().any(|s| name.ends_with(s)) {
        return WType::F32;
    }
    WType::Q8_0
}

/// Random values in each role's natural range; matrices get ±sqrt(3/cols), so outputs stay near unit size.
fn tensor_bytes(rng: &mut Rng, name: &str, ty: WType, dims: &[u64]) -> Vec<u8> {
    let cols = dims[0] as usize;
    let n = dims.iter().product::<u64>() as usize;
    let (lo, hi) = if name.ends_with("norm.weight") {
        (0.8, 1.2)
    } else if name.ends_with("ssm_a") {
        (-1.5, -0.1)
    } else if name.ends_with("ssm_dt.bias") || name.ends_with("ssm_conv1d.weight") {
        (-0.5, 0.5)
    } else if name.ends_with("ffn_gate_inp.weight") {
        // Routers 20× wider than other matrices: logits spread far enough apart that the 4th and 5th expert are
        // rarely within test 2's near-tie margin, as in a trained model.
        let s = 20.0 * (3.0 / cols as f32).sqrt();
        (-s, s)
    } else {
        let s = (3.0 / cols as f32).sqrt();
        (-s, s)
    };
    match ty {
        WType::F32 => (0..n)
            .flat_map(|_| rng.uniform(lo, hi).to_le_bytes())
            .collect(),
        WType::Bf16 => (0..n)
            .flat_map(|_| bf16::from_f32(rng.uniform(lo, hi)).to_le_bytes())
            .collect(),
        _ => random_quant(rng, ty, n, hi),
    }
}

/// Random quant bytes for `n` values of type `ty`, with each block's f16 scales set so values stay within about ±amp.
/// The tiny model's weights, and the kernels' random test matrices.
pub fn random_quant(rng: &mut Rng, ty: WType, n: usize, amp: f32) -> Vec<u8> {
    let (per, bytes) = ty.block();
    let mut out = vec![0u8; n / per * bytes];
    rng.fill(&mut out);
    let mut f = |scale: f32| f16::from_f32(scale * rng.uniform(0.5, 1.0)).to_le_bytes();
    for b in out.chunks_mut(bytes) {
        match ty {
            WType::Q8_0 => b[0..2].copy_from_slice(&f(amp / 127.0)),
            WType::Q4K => {
                b[0..2].copy_from_slice(&f(amp / (63.0 * 15.0)));
                b[2..4].copy_from_slice(&f(amp / (63.0 * 30.0)));
            }
            WType::Q5K => {
                b[0..2].copy_from_slice(&f(amp / (63.0 * 31.0)));
                b[2..4].copy_from_slice(&f(amp / (63.0 * 62.0)));
            }
            WType::Q6K => b[208..210].copy_from_slice(&f(amp / (127.0 * 32.0))),
            WType::F32 | WType::Bf16 => unreachable!("not a block type"),
        }
    }
    out
}

/// A valid `qwen35moe` GGUF for `Dims::tiny()`, the same bytes for the same seed on every platform.
pub fn tiny_model(seed: u64) -> Vec<u8> {
    let d = Dims::tiny();
    let v = tiny_vocab(d.n_vocab);
    let id = |s: &str| {
        Value::U32(
            v.tokens
                .iter()
                .position(|t| t == s)
                .expect("a special token") as u32,
        )
    };
    let s = |x: &str| Value::String(x.into());
    let mut w = Writer::new();
    for (k, val) in expected_meta(&d) {
        w.meta(&k, val);
    }
    w.meta("general.name", s("artemis-tiny"))
        .meta(
            "tokenizer.ggml.tokens",
            Value::Array(v.tokens.iter().map(|t| s(t)).collect()),
        )
        .meta(
            "tokenizer.ggml.token_type",
            Value::Array(v.types.iter().map(|&t| Value::I32(t)).collect()),
        )
        .meta(
            "tokenizer.ggml.merges",
            Value::Array(v.merges.iter().map(|m| s(m)).collect()),
        )
        .meta("tokenizer.ggml.eos_token_id", id("<|im_end|>"))
        .meta("tokenizer.ggml.bos_token_id", id("<|endoftext|>"))
        .meta("tokenizer.ggml.padding_token_id", id("<|vision_pad|>"))
        .meta("tokenizer.ggml.add_bos_token", Value::Bool(false));
    let mut rng = Rng::new(seed);
    for e in expected_tensors(&d) {
        let ty = tiny_type(&e.name, &d);
        let data = tensor_bytes(&mut rng, &e.name, ty, &e.dims);
        w.tensor(&e.name, &e.dims, ty.ggml(), data);
    }
    w.finish()
}
