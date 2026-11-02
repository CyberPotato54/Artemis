//! The model's tensors by role, borrowed from the GGUF's bytes. `Model::open` validates the contract first.
use crate::{
    contract::{self, ContractError},
    dims::Dims,
    quant::{self, WType},
};
use gguf::Gguf;

/// `rows` rows of `cols` values each, stored as `ty`, with `cols` the contiguous dimension (ne0). Vectors have one row;
/// a 3-D expert tensor has `rows = rows_per_expert × n_expert`.
#[derive(Clone, Copy, Debug)]
pub struct Weight<'a> {
    pub ty: WType,
    pub cols: usize,
    pub rows: usize,
    pub bytes: &'a [u8],
}

impl<'a> Weight<'a> {
    pub fn row(&self, r: usize) -> &'a [u8] {
        let rb = self.ty.row_bytes(self.cols);
        &self.bytes[r * rb..(r + 1) * rb]
    }

    /// Expert `e` of a 3-D expert tensor, as its own `rows_per_expert × cols` matrix.
    pub fn expert(&self, e: usize, rows_per_expert: usize) -> Weight<'a> {
        let n = rows_per_expert * self.ty.row_bytes(self.cols);
        Weight {
            rows: rows_per_expert,
            bytes: &self.bytes[e * n..(e + 1) * n],
            ..*self
        }
    }

    /// Every value, dequantized. For vectors and small matrices.
    pub fn to_f32(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.rows * self.cols];
        for (r, o) in out.chunks_mut(self.cols).enumerate() {
            quant::dequant_row(self.ty, self.row(r), o);
        }
        out
    }
}

pub struct DeltaNet<'a> {
    pub qkv: Weight<'a>,
    /// `attn_gate`, the z projection.
    pub gate: Weight<'a>,
    pub beta: Weight<'a>,
    pub alpha: Weight<'a>,
    pub conv1d: Weight<'a>,
    pub dt_bias: Weight<'a>,
    /// `ssm_a`: already −exp(A_log).
    pub a: Weight<'a>,
    pub norm: Weight<'a>,
    pub out: Weight<'a>,
}

pub struct Attention<'a> {
    /// Query and gate, interleaved per head.
    pub q: Weight<'a>,
    pub k: Weight<'a>,
    pub v: Weight<'a>,
    pub out: Weight<'a>,
    pub q_norm: Weight<'a>,
    pub k_norm: Weight<'a>,
}

pub enum Mixer<'a> {
    DeltaNet(DeltaNet<'a>),
    Attention(Attention<'a>),
}

pub struct Moe<'a> {
    pub router: Weight<'a>,
    pub gate_exps: Weight<'a>,
    pub up_exps: Weight<'a>,
    pub down_exps: Weight<'a>,
}

pub struct Shared<'a> {
    /// `ffn_gate_inp_shexp`: one row, the shared expert's scalar gate.
    pub gate_inp: Weight<'a>,
    pub gate: Weight<'a>,
    pub up: Weight<'a>,
    pub down: Weight<'a>,
}

pub struct Layer<'a> {
    pub attn_norm: Weight<'a>,
    pub post_attention_norm: Weight<'a>,
    pub mixer: Mixer<'a>,
    pub moe: Moe<'a>,
    pub shared: Shared<'a>,
}

/// `blk.40`: validated and kept; the CPU reference runs it (`reference::MtpReference`, semantics reference §6).
pub struct Mtp<'a> {
    pub layer: Layer<'a>,
    pub eh_proj: Weight<'a>,
    pub enorm: Weight<'a>,
    pub hnorm: Weight<'a>,
    pub shared_head_norm: Weight<'a>,
}

pub struct Model<'a> {
    pub dims: Dims,
    pub embd: Weight<'a>,
    pub output_norm: Weight<'a>,
    pub output: Weight<'a>,
    pub layers: Vec<Layer<'a>>,
    pub mtp: Mtp<'a>,
}

impl<'a> Model<'a> {
    pub fn open(g: &'a Gguf, dims: &Dims) -> Result<Model<'a>, ContractError> {
        contract::validate(g, dims)?;
        let w = |name: &str| -> Weight<'a> {
            let t = g.tensor(name).expect("validated");
            Weight {
                ty: WType::from_ggml(t.ggml_type).expect("validated"),
                cols: t.dims[0] as usize,
                rows: t.dims[1..].iter().product::<u64>() as usize,
                bytes: t.bytes,
            }
        };
        let layer = |l: usize| -> Layer<'a> {
            let p = |s: &str| w(&format!("blk.{l}.{s}"));
            let mixer = if dims.is_attention(l) {
                Mixer::Attention(Attention {
                    q: p("attn_q.weight"),
                    k: p("attn_k.weight"),
                    v: p("attn_v.weight"),
                    out: p("attn_output.weight"),
                    q_norm: p("attn_q_norm.weight"),
                    k_norm: p("attn_k_norm.weight"),
                })
            } else {
                Mixer::DeltaNet(DeltaNet {
                    qkv: p("attn_qkv.weight"),
                    gate: p("attn_gate.weight"),
                    beta: p("ssm_beta.weight"),
                    alpha: p("ssm_alpha.weight"),
                    conv1d: p("ssm_conv1d.weight"),
                    dt_bias: p("ssm_dt.bias"),
                    a: p("ssm_a"),
                    norm: p("ssm_norm.weight"),
                    out: p("ssm_out.weight"),
                })
            };
            Layer {
                attn_norm: p("attn_norm.weight"),
                post_attention_norm: p("post_attention_norm.weight"),
                mixer,
                moe: Moe {
                    router: p("ffn_gate_inp.weight"),
                    gate_exps: p("ffn_gate_exps.weight"),
                    up_exps: p("ffn_up_exps.weight"),
                    down_exps: p("ffn_down_exps.weight"),
                },
                shared: Shared {
                    gate_inp: p("ffn_gate_inp_shexp.weight"),
                    gate: p("ffn_gate_shexp.weight"),
                    up: p("ffn_up_shexp.weight"),
                    down: p("ffn_down_shexp.weight"),
                },
            }
        };
        let m = dims.n_layer;
        Ok(Model {
            dims: dims.clone(),
            embd: w("token_embd.weight"),
            output_norm: w("output_norm.weight"),
            output: w("output.weight"),
            layers: (0..dims.n_layer).map(&layer).collect(),
            mtp: Mtp {
                layer: layer(m),
                eh_proj: w(&format!("blk.{m}.nextn.eh_proj.weight")),
                enorm: w(&format!("blk.{m}.nextn.enorm.weight")),
                hnorm: w(&format!("blk.{m}.nextn.hnorm.weight")),
                shared_head_norm: w(&format!("blk.{m}.nextn.shared_head_norm.weight")),
            },
        })
    }
}
