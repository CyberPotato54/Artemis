//! What a GGUF must hold to be Qwen3.6-35B-A3B (or the tiny test model): metadata, and every tensor's name, shape
//! and allowed types. `validate` reports the first difference.
use crate::{dims::Dims, quant::WType};
use gguf::{Gguf, Value};
use std::{collections::HashSet, fmt};

#[derive(Clone, Debug, PartialEq)]
pub enum ContractError {
    Meta {
        key: String,
        expected: String,
        found: String,
    },
    /// The file is a Qwen3.6 GGUF without the MTP layer (the non-MTP repository's files).
    NoMtpLayer {
        key: String,
        found: String,
        expected: String,
    },
    MissingTensor(String),
    UnexpectedTensor(String),
    Shape {
        tensor: String,
        expected: Vec<u64>,
        found: Vec<u64>,
    },
    Type {
        tensor: String,
        allowed: Vec<&'static str>,
        found: String,
    },
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContractError::Meta {
                key,
                expected,
                found,
            } => write!(f, "metadata {key}: expected {expected}, found {found}"),
            ContractError::MissingTensor(t) => write!(f, "missing tensor {t}"),
            ContractError::NoMtpLayer {
                key,
                found,
                expected,
            } => write!(
                f,
                "this GGUF has no MTP layer ({key} is {found}, expected {expected}); Artemis needs a file from the -MTP-GGUF repository, unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
            ),
            ContractError::UnexpectedTensor(t) => write!(f, "unexpected tensor {t:?}"),
            ContractError::Shape {
                tensor,
                expected,
                found,
            } => {
                write!(
                    f,
                    "tensor {tensor}: shape expected {expected:?}, found {found:?}"
                )
            }
            ContractError::Type {
                tensor,
                allowed,
                found,
            } => {
                write!(
                    f,
                    "tensor {tensor}: type {found} is not one of [{}]",
                    allowed.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for ContractError {}

/// Weight matrices: any of the six types.
pub const MATRIX: &[WType] = &WType::ALL;
/// Norms, `ssm_*` vectors and the conv kernel.
pub const VECTOR: &[WType] = &[WType::F32];
/// Routers and the shared-expert gate: F32, or BF16 as in `blk.40`.
pub const ROUTER: &[WType] = &[WType::F32, WType::Bf16];

pub struct Expected {
    pub name: String,
    /// ne order: `dims[0]` is the contiguous dimension.
    pub dims: Vec<u64>,
    pub allowed: &'static [WType],
}

pub fn expected_meta(d: &Dims) -> Vec<(String, Value)> {
    let a = |k: &str| format!("qwen35moe.{k}");
    let u = |n: usize| Value::U32(n as u32);
    vec![
        (
            "general.architecture".into(),
            Value::String("qwen35moe".into()),
        ),
        (a("block_count"), u(d.n_layer + d.n_nextn)),
        (a("nextn_predict_layers"), u(d.n_nextn)),
        (a("context_length"), u(d.n_ctx_train)),
        (a("embedding_length"), u(d.n_embd)),
        (a("attention.head_count"), u(d.n_head)),
        (a("attention.head_count_kv"), u(d.n_head_kv)),
        (a("attention.key_length"), u(d.head_dim)),
        (a("attention.value_length"), u(d.head_dim)),
        (a("attention.layer_norm_rms_epsilon"), Value::F32(d.rms_eps)),
        (a("rope.dimension_count"), u(d.n_rot)),
        (
            a("rope.dimension_sections"),
            Value::Array(
                d.rope_sections
                    .iter()
                    .map(|&s| Value::I32(s as i32))
                    .collect(),
            ),
        ),
        (a("rope.freq_base"), Value::F32(d.rope_base)),
        (a("expert_count"), u(d.n_expert)),
        (a("expert_used_count"), u(d.n_expert_used)),
        (a("expert_feed_forward_length"), u(d.n_ff_exp)),
        (a("expert_shared_feed_forward_length"), u(d.n_ff_shexp)),
        (a("ssm.conv_kernel"), u(d.ssm_conv)),
        (a("ssm.state_size"), u(d.ssm_d_state)),
        (a("ssm.group_count"), u(d.ssm_n_group)),
        (a("ssm.time_step_rank"), u(d.ssm_dt_rank)),
        (a("ssm.inner_size"), u(d.ssm_inner)),
        (a("full_attention_interval"), u(d.full_attn_interval)),
        ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
        ("tokenizer.ggml.pre".into(), Value::String("qwen35".into())),
    ]
}

fn ex(name: String, dims: &[u64], allowed: &'static [WType]) -> Expected {
    Expected {
        name,
        dims: dims.to_vec(),
        allowed,
    }
}

/// Every tensor, in a fixed order: globals, then each block. 753 for the real model.
pub fn expected_tensors(d: &Dims) -> Vec<Expected> {
    let (e, v) = (d.n_embd as u64, d.n_vocab as u64);
    let mut out = vec![
        ex("token_embd.weight".into(), &[e, v], MATRIX),
        ex("output_norm.weight".into(), &[e], VECTOR),
        ex("output.weight".into(), &[e, v], MATRIX),
    ];
    for l in 0..d.n_layer + d.n_nextn {
        let p = |s: &str| format!("blk.{l}.{s}");
        out.push(ex(p("attn_norm.weight"), &[e], VECTOR));
        out.push(ex(p("post_attention_norm.weight"), &[e], VECTOR));
        if d.is_attention(l) {
            let (h, kv, hd) = (d.n_head as u64, d.n_head_kv as u64, d.head_dim as u64);
            out.push(ex(p("attn_q.weight"), &[e, 2 * h * hd], MATRIX));
            out.push(ex(p("attn_k.weight"), &[e, kv * hd], MATRIX));
            out.push(ex(p("attn_v.weight"), &[e, kv * hd], MATRIX));
            out.push(ex(p("attn_output.weight"), &[h * hd, e], MATRIX));
            out.push(ex(p("attn_q_norm.weight"), &[hd], VECTOR));
            out.push(ex(p("attn_k_norm.weight"), &[hd], VECTOR));
        } else {
            let (conv, inner, rank) = (
                d.conv_dim() as u64,
                d.ssm_inner as u64,
                d.ssm_dt_rank as u64,
            );
            out.push(ex(p("attn_qkv.weight"), &[e, conv], MATRIX));
            out.push(ex(p("attn_gate.weight"), &[e, inner], MATRIX));
            out.push(ex(p("ssm_beta.weight"), &[e, rank], MATRIX));
            out.push(ex(p("ssm_alpha.weight"), &[e, rank], MATRIX));
            out.push(ex(
                p("ssm_conv1d.weight"),
                &[d.ssm_conv as u64, conv],
                VECTOR,
            ));
            out.push(ex(p("ssm_dt.bias"), &[rank], VECTOR));
            out.push(ex(p("ssm_a"), &[rank], VECTOR));
            out.push(ex(p("ssm_norm.weight"), &[d.v_head_dim() as u64], VECTOR));
            out.push(ex(p("ssm_out.weight"), &[inner, e], MATRIX));
        }
        let (ne, ff, sff) = (d.n_expert as u64, d.n_ff_exp as u64, d.n_ff_shexp as u64);
        out.push(ex(p("ffn_gate_inp.weight"), &[e, ne], ROUTER));
        out.push(ex(p("ffn_gate_exps.weight"), &[e, ff, ne], MATRIX));
        out.push(ex(p("ffn_up_exps.weight"), &[e, ff, ne], MATRIX));
        out.push(ex(p("ffn_down_exps.weight"), &[ff, e, ne], MATRIX));
        out.push(ex(p("ffn_gate_inp_shexp.weight"), &[e], ROUTER));
        out.push(ex(p("ffn_gate_shexp.weight"), &[e, sff], MATRIX));
        out.push(ex(p("ffn_up_shexp.weight"), &[e, sff], MATRIX));
        out.push(ex(p("ffn_down_shexp.weight"), &[sff, e], MATRIX));
        if l >= d.n_layer {
            out.push(ex(p("nextn.eh_proj.weight"), &[2 * e, e], MATRIX));
            for n in [
                "nextn.enorm.weight",
                "nextn.hnorm.weight",
                "nextn.shared_head_norm.weight",
            ] {
                out.push(ex(p(n), &[e], VECTOR));
            }
        }
    }
    out
}

/// Numbers compare by value whatever their stored type; f32 values compare as stored.
fn same(want: &Value, found: &Value) -> bool {
    match (want, found) {
        (Value::String(a), Value::String(b)) => a == b,
        (Value::F32(a), _) => found.as_f64().is_some_and(|b| b as f32 == *a),
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same(x, y))
        }
        _ => want.as_i64().is_some() && want.as_i64() == found.as_i64(),
    }
}

/// How many characters of a file-provided value an error shows.
const SHOW_CHARS: usize = 200;

/// A value for an error message: strings escaped, and everything cut to about `SHOW_CHARS` characters so a crafted
/// value can't flood the terminal.
fn show(v: &Value) -> String {
    let s = show_full(v);
    match s.char_indices().nth(SHOW_CHARS) {
        Some((end, _)) => format!("{}...", &s[..end]),
        None => s,
    }
}

fn show_full(v: &Value) -> String {
    match v {
        Value::String(s) => {
            let head: String = s.chars().take(SHOW_CHARS).collect();
            let more = if head.len() < s.len() { "..." } else { "" };
            format!("{head:?}{more}")
        }
        Value::Array(a) => {
            let items: Vec<String> = a.iter().take(16).map(show).collect();
            let more = if a.len() > 16 { ", ..." } else { "" };
            format!("[{}{more}]", items.join(", "))
        }
        Value::Bool(b) => b.to_string(),
        Value::F32(x) => format!("{x:e}"),
        Value::F64(x) => format!("{x:e}"),
        other => other
            .as_i64()
            .map_or_else(|| format!("{other:?}"), |n| n.to_string()),
    }
}

pub fn validate(g: &Gguf, d: &Dims) -> Result<(), ContractError> {
    // Only a Qwen3.6 file can lack the MTP layer; any other architecture gets the architecture mismatch below.
    let is_qwen36 = g.meta("general.architecture").and_then(Value::as_str) == Some("qwen35moe");
    if is_qwen36 && d.n_nextn > 0 {
        let shown = |key: &str| g.meta(key).map_or_else(|| "missing".to_string(), show);
        let blocks = g.meta("qwen35moe.block_count");
        let nextn = g.meta("qwen35moe.nextn_predict_layers");
        if blocks.is_some_and(|v| v.as_u64() == Some(d.n_layer as u64)) {
            return Err(ContractError::NoMtpLayer {
                key: "qwen35moe.block_count".into(),
                found: shown("qwen35moe.block_count"),
                expected: (d.n_layer + d.n_nextn).to_string(),
            });
        }
        if nextn.and_then(Value::as_u64).is_none_or(|n| n == 0) {
            return Err(ContractError::NoMtpLayer {
                key: "qwen35moe.nextn_predict_layers".into(),
                found: shown("qwen35moe.nextn_predict_layers"),
                expected: d.n_nextn.to_string(),
            });
        }
    }
    for (key, want) in expected_meta(d) {
        let found = match g.meta(&key) {
            None => "missing".to_string(),
            Some(v) if !same(&want, v) => show(v),
            Some(_) => continue,
        };
        return Err(ContractError::Meta {
            key,
            expected: show(&want),
            found,
        });
    }
    let n_tokens = g
        .meta("tokenizer.ggml.tokens")
        .and_then(Value::as_array)
        .map_or(0, <[Value]>::len);
    if n_tokens != d.n_vocab {
        return Err(ContractError::Meta {
            key: "tokenizer.ggml.tokens".into(),
            expected: format!("{} entries", d.n_vocab),
            found: format!("{n_tokens} entries"),
        });
    }
    let expected = expected_tensors(d);
    for e in &expected {
        let t = g
            .tensor(&e.name)
            .ok_or_else(|| ContractError::MissingTensor(e.name.clone()))?;
        if t.dims != e.dims.as_slice() {
            return Err(ContractError::Shape {
                tensor: e.name.clone(),
                expected: e.dims.clone(),
                found: t.dims.to_vec(),
            });
        }
        let ty = WType::from_ggml(t.ggml_type);
        if !ty.is_some_and(|ty| e.allowed.contains(&ty)) {
            return Err(ContractError::Type {
                tensor: e.name.clone(),
                allowed: e.allowed.iter().map(|w| w.name()).collect(),
                found: ty.map_or_else(
                    || format!("ggml type {}", t.ggml_type),
                    |w| w.name().to_string(),
                ),
            });
        }
    }
    if g.tensor_count() != expected.len() {
        let names: HashSet<&str> = expected.iter().map(|e| e.name.as_str()).collect();
        let extra = g
            .tensors()
            .find(|t| !names.contains(t.name))
            .expect("more tensors than expected, so one is extra");
        return Err(ContractError::UnexpectedTensor(extra.name.to_string()));
    }
    Ok(())
}
