//! Tap names: one table naming every intermediate that the reference, the GPU engine and llama.cpp write,
//! so any two can be compared layer by layer. Each key maps to llama.cpp b11509's callback name (semantics reference
//! §9). Where llama.cpp names more than one tensor alike in a layer (`Qcur`, `Kcur`, `Vcur`), the last one is ours.
use crate::dims::Dims;
use std::{fmt, str::FromStr};

/// A layer's intermediates, in graph order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Key {
    AttnNorm,
    GdnQkv,
    GdnZ,
    GdnBeta,
    GdnG,
    GdnConv,
    GdnQ,
    GdnK,
    GdnV,
    GdnO,
    GdnNormed,
    AttnQ,
    AttnK,
    AttnV,
    AttnPregate,
    AttnGated,
    MixerOut,
    Resid,
    FfnNorm,
    RouterLogits,
    RouterProbs,
    TopK,
    TopKW,
    MoeSwiglu,
    MoeOut,
    ShexpSwiglu,
    ShexpOut,
    FfnOut,
    Out,
}

impl Key {
    pub const ALL: [Key; 29] = [
        Key::AttnNorm,
        Key::GdnQkv,
        Key::GdnZ,
        Key::GdnBeta,
        Key::GdnG,
        Key::GdnConv,
        Key::GdnQ,
        Key::GdnK,
        Key::GdnV,
        Key::GdnO,
        Key::GdnNormed,
        Key::AttnQ,
        Key::AttnK,
        Key::AttnV,
        Key::AttnPregate,
        Key::AttnGated,
        Key::MixerOut,
        Key::Resid,
        Key::FfnNorm,
        Key::RouterLogits,
        Key::RouterProbs,
        Key::TopK,
        Key::TopKW,
        Key::MoeSwiglu,
        Key::MoeOut,
        Key::ShexpSwiglu,
        Key::ShexpOut,
        Key::FfnOut,
        Key::Out,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Key::AttnNorm => "attn_norm",
            Key::GdnQkv => "gdn.qkv",
            Key::GdnZ => "gdn.z",
            Key::GdnBeta => "gdn.beta",
            Key::GdnG => "gdn.g",
            Key::GdnConv => "gdn.conv",
            Key::GdnQ => "gdn.q",
            Key::GdnK => "gdn.k",
            Key::GdnV => "gdn.v",
            Key::GdnO => "gdn.o",
            Key::GdnNormed => "gdn.normed",
            Key::AttnQ => "attn.q",
            Key::AttnK => "attn.k",
            Key::AttnV => "attn.v",
            Key::AttnPregate => "attn.pregate",
            Key::AttnGated => "attn.gated",
            Key::MixerOut => "mixer_out",
            Key::Resid => "resid",
            Key::FfnNorm => "ffn_norm",
            Key::RouterLogits => "router_logits",
            Key::RouterProbs => "router_probs",
            Key::TopK => "topk",
            Key::TopKW => "topk_w",
            Key::MoeSwiglu => "moe_swiglu",
            Key::ShexpSwiglu => "shexp_swiglu",
            Key::MoeOut => "moe_out",
            Key::ShexpOut => "shexp_out",
            Key::FfnOut => "ffn_out",
            Key::Out => "out",
        }
    }

    /// Whether a layer of this kind has this intermediate.
    pub fn applies(self, attention: bool) -> bool {
        use Key::*;
        match self {
            GdnQkv | GdnZ | GdnBeta | GdnG | GdnConv | GdnQ | GdnK | GdnV | GdnO | GdnNormed => {
                !attention
            }
            AttnQ | AttnK | AttnV | AttnPregate | AttnGated => attention,
            _ => true,
        }
    }

    /// llama.cpp b11509's callback name for this key, without the `-N` layer suffix.
    pub fn llama(self, attention: bool) -> Option<&'static str> {
        if !self.applies(attention) {
            return None;
        }
        Some(match self {
            Key::AttnNorm => "attn_norm",
            Key::GdnQkv => "linear_attn_qkv_mixed",
            Key::GdnZ => "z",
            Key::GdnBeta => "beta_sigmoid",
            Key::GdnG => "gate",
            Key::GdnConv => "conv_output_silu",
            Key::GdnQ => "q_conv_predelta",
            Key::GdnK => "k_conv_predelta",
            Key::GdnV => "v_conv_predelta",
            Key::GdnO => "attn_output",
            Key::GdnNormed => "final_output",
            Key::AttnQ => "Qcur",
            Key::AttnK => "Kcur",
            Key::AttnV => "Vcur",
            Key::AttnPregate => "attn_pregate",
            Key::AttnGated => "attn_gated",
            Key::MixerOut if attention => "attn_output",
            Key::MixerOut => "linear_attn_out",
            Key::Resid => "attn_residual",
            Key::FfnNorm => "attn_post_norm",
            Key::RouterLogits => "ffn_moe_logits",
            Key::RouterProbs => "ffn_moe_probs",
            Key::TopK => "ffn_moe_topk",
            Key::TopKW => "ffn_moe_weights_norm",
            Key::MoeSwiglu => "ffn_moe_swiglu",
            Key::ShexpSwiglu => "ffn_swiglu",
            Key::MoeOut => "ffn_moe_out",
            Key::ShexpOut => "ffn_shexp_gated",
            Key::FfnOut => "ffn_out",
            Key::Out => "l_out",
        })
    }
}

/// The MTP block's (`blk.40`) intermediates, in semantics order (§6: llama.cpp builds `mtp_hnorm` before `mtp_enorm`;
/// nothing compares by position). Each maps to a llama.cpp b11509 `mtp_*` callback name (`qwen35moe.cpp:605-738`; the
/// router, top-k and SwiGLU taps are `build_moe_ffn`/`build_ffn`'s own names, `llama-graph.cpp:1909, 2086, 2117, 2169,
/// 2202, 2307`). `Q` and `K` are llama.cpp's
/// `mtp_Qcur_normed` and `mtp_Kcur_normed`: after the per-head norm and **before** RoPE, unlike the main layers'
/// `AttnQ` and `AttnK`, which are named after RoPE (`build_attn` doesn't rename its inputs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MtpKey {
    Enorm,
    Hnorm,
    Concat,
    EhProj,
    AttnNorm,
    Q,
    K,
    V,
    AttnPregate,
    AttnGated,
    AttnOut,
    AttnResidual,
    PostNorm,
    RouterLogits,
    RouterProbs,
    TopK,
    TopKW,
    MoeSwiglu,
    MoeOut,
    ShexpSwiglu,
    ShexpGated,
    FfnOut,
    PostFfn,
    HeadNorm,
    Logits,
}

impl MtpKey {
    pub const ALL: [MtpKey; 25] = [
        MtpKey::Enorm,
        MtpKey::Hnorm,
        MtpKey::Concat,
        MtpKey::EhProj,
        MtpKey::AttnNorm,
        MtpKey::Q,
        MtpKey::K,
        MtpKey::V,
        MtpKey::AttnPregate,
        MtpKey::AttnGated,
        MtpKey::AttnOut,
        MtpKey::AttnResidual,
        MtpKey::PostNorm,
        MtpKey::RouterLogits,
        MtpKey::RouterProbs,
        MtpKey::TopK,
        MtpKey::TopKW,
        MtpKey::MoeSwiglu,
        MtpKey::MoeOut,
        MtpKey::ShexpSwiglu,
        MtpKey::ShexpGated,
        MtpKey::FfnOut,
        MtpKey::PostFfn,
        MtpKey::HeadNorm,
        MtpKey::Logits,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MtpKey::Enorm => "enorm",
            MtpKey::Hnorm => "hnorm",
            MtpKey::Concat => "concat",
            MtpKey::EhProj => "eh_proj",
            MtpKey::AttnNorm => "attn_norm",
            MtpKey::Q => "q",
            MtpKey::K => "k",
            MtpKey::V => "v",
            MtpKey::AttnPregate => "attn_pregate",
            MtpKey::AttnGated => "attn_gated",
            MtpKey::AttnOut => "attn_out",
            MtpKey::AttnResidual => "attn_residual",
            MtpKey::PostNorm => "post_norm",
            MtpKey::RouterLogits => "router_logits",
            MtpKey::RouterProbs => "router_probs",
            MtpKey::TopK => "topk",
            MtpKey::TopKW => "topk_w",
            MtpKey::MoeSwiglu => "moe_swiglu",
            MtpKey::MoeOut => "moe_out",
            MtpKey::ShexpSwiglu => "shexp_swiglu",
            MtpKey::ShexpGated => "shexp_gated",
            MtpKey::FfnOut => "ffn_out",
            MtpKey::PostFfn => "post_ffn",
            MtpKey::HeadNorm => "head_norm",
            MtpKey::Logits => "logits",
        }
    }

    /// llama.cpp's callback name. All but `h_nextn` and `result_output` carry the `-N` layer suffix (`N` =
    /// `n_layer`, the MTP block's index); those two are bare (`cb(..., -1)`).
    pub fn llama(self) -> &'static str {
        match self {
            MtpKey::Enorm => "mtp_enorm",
            MtpKey::Hnorm => "mtp_hnorm",
            MtpKey::Concat => "mtp_concat",
            MtpKey::EhProj => "mtp_eh_proj",
            MtpKey::AttnNorm => "mtp_attn_norm",
            MtpKey::Q => "mtp_Qcur_normed",
            MtpKey::K => "mtp_Kcur_normed",
            MtpKey::V => "mtp_Vcur",
            MtpKey::AttnPregate => "mtp_attn_pregate",
            // Oracle-assigned: llama.cpp leaves the product of `mtp_attn_pregate` and the gate unnamed
            // (`qwen35moe.cpp:667`); `artemis-oracle mtp-taps` names that node.
            MtpKey::AttnGated => "mtp_attn_gated",
            MtpKey::AttnOut => "mtp_attn_out",
            MtpKey::AttnResidual => "mtp_attn_residual",
            MtpKey::PostNorm => "mtp_attn_post_norm",
            MtpKey::RouterLogits => "ffn_moe_logits",
            MtpKey::RouterProbs => "ffn_moe_probs",
            MtpKey::TopK => "ffn_moe_topk",
            MtpKey::TopKW => "ffn_moe_weights_norm",
            MtpKey::MoeSwiglu => "ffn_moe_swiglu",
            MtpKey::MoeOut => "mtp_ffn_moe_out",
            MtpKey::ShexpSwiglu => "ffn_swiglu",
            MtpKey::ShexpGated => "mtp_ffn_shexp_gated",
            MtpKey::FfnOut => "mtp_ffn_out",
            MtpKey::PostFfn => "mtp_post_ffn",
            MtpKey::HeadNorm => "h_nextn",
            MtpKey::Logits => "result_output",
        }
    }

    /// Whether llama.cpp names this tensor without a `-N` layer suffix.
    pub fn bare(self) -> bool {
        matches!(self, MtpKey::HeadNorm | MtpKey::Logits)
    }

    /// Values per position.
    pub fn per_pos(self, d: &Dims) -> usize {
        match self {
            MtpKey::Concat => 2 * d.n_embd,
            MtpKey::Q | MtpKey::AttnPregate | MtpKey::AttnGated => d.n_head * d.head_dim,
            MtpKey::K | MtpKey::V => d.n_head_kv * d.head_dim,
            MtpKey::Logits => d.n_vocab,
            MtpKey::RouterLogits | MtpKey::RouterProbs => d.n_expert,
            MtpKey::TopK | MtpKey::TopKW => d.n_expert_used,
            MtpKey::MoeSwiglu => d.n_expert_used * d.n_ff_exp,
            MtpKey::ShexpSwiglu => d.n_ff_shexp,
            MtpKey::Enorm
            | MtpKey::Hnorm
            | MtpKey::EhProj
            | MtpKey::AttnNorm
            | MtpKey::AttnOut
            | MtpKey::AttnResidual
            | MtpKey::PostNorm
            | MtpKey::MoeOut
            | MtpKey::ShexpGated
            | MtpKey::FfnOut
            | MtpKey::PostFfn
            | MtpKey::HeadNorm => d.n_embd,
        }
    }
}

/// One intermediate. `Ord` is graph order: `Embd`, then layer by layer, then `FinalNorm` and `Logits`, then the MTP
/// block's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TapName {
    Embd,
    Layer(u16, Key),
    FinalNorm,
    Logits,
    Mtp(MtpKey),
}

const EMBD_LLAMA: &str = "model.input_embed";
const FINAL_NORM_LLAMA: &str = "result_norm";
const LOGITS_LLAMA: &str = "result_output";

impl TapName {
    /// Values per position. Position-major files hold `per_pos × n_tokens` values.
    pub fn per_pos(&self, d: &Dims) -> usize {
        match self {
            TapName::Mtp(k) => k.per_pos(d),
            TapName::Embd | TapName::FinalNorm => d.n_embd,
            TapName::Logits => d.n_vocab,
            TapName::Layer(_, k) => match k {
                Key::GdnQkv | Key::GdnConv => d.conv_dim(),
                Key::GdnZ | Key::GdnV | Key::GdnO | Key::GdnNormed => d.ssm_inner,
                Key::GdnBeta | Key::GdnG => d.ssm_dt_rank,
                Key::GdnQ | Key::GdnK => d.key_dim(),
                Key::AttnQ | Key::AttnPregate | Key::AttnGated => d.n_head * d.head_dim,
                Key::AttnK | Key::AttnV => d.n_head_kv * d.head_dim,
                Key::RouterLogits | Key::RouterProbs => d.n_expert,
                Key::TopK | Key::TopKW => d.n_expert_used,
                Key::MoeSwiglu => d.n_expert_used * d.n_ff_exp,
                Key::ShexpSwiglu => d.n_ff_shexp,
                Key::AttnNorm
                | Key::MixerOut
                | Key::Resid
                | Key::FfnNorm
                | Key::MoeOut
                | Key::ShexpOut
                | Key::FfnOut
                | Key::Out => d.n_embd,
            },
        }
    }

    /// llama.cpp's full callback name, such as `attn_norm-12`; None if this layer kind has no such tensor.
    pub fn llama(&self, d: &Dims) -> Option<String> {
        match self {
            TapName::Embd => Some(EMBD_LLAMA.into()),
            TapName::FinalNorm => Some(FINAL_NORM_LLAMA.into()),
            TapName::Logits => Some(LOGITS_LLAMA.into()),
            TapName::Layer(l, k) => k
                .llama(d.is_attention(*l as usize))
                .map(|n| format!("{n}-{l}")),
            TapName::Mtp(k) if k.bare() => Some(k.llama().into()),
            TapName::Mtp(k) => Some(format!("{}-{}", k.llama(), d.n_layer)),
        }
    }
}

impl fmt::Display for TapName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TapName::Embd => f.write_str("embd"),
            TapName::Layer(l, k) => write!(f, "l{l}.{}", k.name()),
            TapName::FinalNorm => f.write_str("final_norm"),
            TapName::Logits => f.write_str("logits"),
            TapName::Mtp(k) => write!(f, "mtp.{}", k.name()),
        }
    }
}

impl FromStr for TapName {
    type Err = String;

    fn from_str(s: &str) -> Result<TapName, String> {
        match s {
            "embd" => return Ok(TapName::Embd),
            "final_norm" => return Ok(TapName::FinalNorm),
            "logits" => return Ok(TapName::Logits),
            _ => {}
        }
        let bad = || format!("unknown tap name {s:?}");
        if let Some(key) = s.strip_prefix("mtp.") {
            return MtpKey::ALL
                .into_iter()
                .find(|k| k.name() == key)
                .map(TapName::Mtp)
                .ok_or_else(bad);
        }
        let (layer, key) = s
            .strip_prefix('l')
            .and_then(|r| r.split_once('.'))
            .ok_or_else(bad)?;
        let layer: u16 = layer.parse().map_err(|_| bad())?;
        let key = Key::ALL
            .into_iter()
            .find(|k| k.name() == key)
            .ok_or_else(bad)?;
        Ok(TapName::Layer(layer, key))
    }
}

/// Every tap the model has, in graph order.
pub fn expected_taps(d: &Dims) -> Vec<TapName> {
    let mut v = vec![TapName::Embd];
    for l in 0..d.n_layer {
        let attention = d.is_attention(l);
        v.extend(
            Key::ALL
                .into_iter()
                .filter(|k| k.applies(attention))
                .map(|k| TapName::Layer(l as u16, k)),
        );
    }
    v.push(TapName::FinalNorm);
    v.push(TapName::Logits);
    v
}

/// The `--filter` regex for `artemis-oracle taps`: every llama.cpp name the table uses, with an optional `-N`.
pub fn llama_filter() -> String {
    let mut names = vec![EMBD_LLAMA, FINAL_NORM_LLAMA, LOGITS_LLAMA];
    for k in Key::ALL {
        for attention in [false, true] {
            if let Some(n) = k.llama(attention) {
                if !names.contains(&n) {
                    names.push(n);
                }
            }
        }
    }
    let escaped: Vec<String> = names.iter().map(|n| n.replace('.', "\\.")).collect();
    format!("^({})(-[0-9]+)?$", escaped.join("|"))
}

/// Every tap the MTP block has, in graph order. Kept apart from `expected_taps`, which is the main model's.
pub fn expected_mtp_taps() -> Vec<TapName> {
    MtpKey::ALL.into_iter().map(TapName::Mtp).collect()
}

/// The `--filter` regex for `artemis-oracle` over the MTP graph: every llama.cpp name the MTP table uses (the `mtp_*`
/// ones, the shared `ffn_moe_*` and `ffn_swiglu` ones, `h_nextn` and `result_output`), each with an optional `-N`.
/// The shared names are also in the main graph, so the MTP oracle writes its own directory.
pub fn mtp_llama_filter() -> String {
    let names: Vec<&str> = MtpKey::ALL.into_iter().map(MtpKey::llama).collect();
    format!("^({})(-[0-9]+)?$", names.join("|"))
}

/// Receives intermediates as they're computed: `data` is tap `name` at position `pos`.
pub trait Tap {
    fn tap(&mut self, name: TapName, pos: u32, data: &[f32]);

    /// Teacher forcing: the value the computation continues from instead of its own, if this tap supplies
    /// one. Test 2 forces llama.cpp's values so each operation is checked on llama.cpp's own inputs.
    fn forced(&self, _name: TapName, _pos: u32) -> Option<&[f32]> {
        None
    }
}

/// Discards everything.
pub struct NullTap;

impl Tap for NullTap {
    fn tap(&mut self, _: TapName, _: u32, _: &[f32]) {}
}
