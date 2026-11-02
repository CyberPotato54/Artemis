//! The f32 CPU reference forward pass: the semantics reference's math, one token at a time, built from
//! `ops`. Weights are dequantized row by row from the mmap; dot products and reductions sum in f64. Test 2 checks it
//! against llama.cpp, and the GPU engine is checked against it.
use crate::{
    dims::Dims,
    model::{Attention, DeltaNet, Layer, Mixer, Model, Weight},
    ops, quant,
    taps::{Key, MtpKey, NullTap, Tap, TapName},
};
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum StepError {
    TokenOutOfRange { token: u32, n_vocab: usize },
}

impl fmt::Display for StepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepError::TokenOutOfRange { token, n_vocab } => {
                write!(
                    f,
                    "token {token} is outside the vocabulary ({n_vocab} tokens)"
                )
            }
        }
    }
}

impl std::error::Error for StepError {}

/// How the reference rounds.
///
/// `Exact` is the plain f32 reference. `F16Kv` is `Exact` with keys and values stored in an f16 cache, as the GPU
/// engine stores them: the GPU's tap rule compares against it, so the f16 rounding the engine shares with
/// it doesn't hide in attention's bound. `LlamaCpu` adds the rounding that
/// llama.cpp's CPU backend does: matmul activations converted to the weight type's dot-product type, f16 keys, values
/// and queries, and flash attention's f16 accumulator. Test 2 compares that mode against llama.cpp, because
/// llama.cpp's backends share the activation rounding, so their mutual noise floor can't see it. The flash-attention
/// part matches llama.cpp's CPU only for batches under 64 query tokens; from 64 it uses a tiled kernel with f32
/// queries and accumulators, and `artemis dump --llama-rounding` refuses such runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    Exact,
    F16Kv,
    LlamaCpu,
}

/// A layer's small vectors, dequantized once.
struct Vecs {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    shexp_gate: Vec<f32>,
    mixer: MixerVecs,
}

enum MixerVecs {
    DeltaNet {
        conv1d: Vec<f32>,
        dt_bias: Vec<f32>,
        a: Vec<f32>,
        norm: Vec<f32>,
    },
    Attention {
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    },
}

enum State {
    /// `conv`: each channel's last `ssm_conv − 1` raw inputs, oldest first, `[conv_dim][ssm_conv − 1]`.
    /// `s`: per value head, `M[j][i] = S[i][j]` with the key index contiguous, `[v_heads][v][k]`.
    DeltaNet { conv: Vec<f32>, s: Vec<f32> },
    /// Post-norm, post-RoPE keys and the values of every position so far, `[pos][n_head_kv × head_dim]`.
    Attention { k: Vec<f32>, v: Vec<f32> },
}

pub struct Reference<'a> {
    model: &'a Model<'a>,
    threads: usize,
    rounding: Rounding,
    pos: u32,
    vecs: Vec<Vecs>,
    state: Vec<State>,
    output_norm: Vec<f32>,
    /// The last step's final hidden row (empty before a step).
    hidden: Vec<f32>,
}

impl<'a> Reference<'a> {
    pub fn new(model: &'a Model<'a>) -> Reference<'a> {
        Reference::with_threads(model, 16)
    }

    pub fn with_threads(model: &'a Model<'a>, threads: usize) -> Reference<'a> {
        Reference::with_options(model, threads, Rounding::Exact)
    }

    pub fn with_options(model: &'a Model<'a>, threads: usize, rounding: Rounding) -> Reference<'a> {
        let d = &model.dims;
        let vecs = model
            .layers
            .iter()
            .map(|l| Vecs {
                attn_norm: l.attn_norm.to_f32(),
                ffn_norm: l.post_attention_norm.to_f32(),
                shexp_gate: l.shared.gate_inp.to_f32(),
                mixer: match &l.mixer {
                    Mixer::DeltaNet(dn) => MixerVecs::DeltaNet {
                        conv1d: dn.conv1d.to_f32(),
                        dt_bias: dn.dt_bias.to_f32(),
                        a: dn.a.to_f32(),
                        norm: dn.norm.to_f32(),
                    },
                    Mixer::Attention(at) => MixerVecs::Attention {
                        q_norm: at.q_norm.to_f32(),
                        k_norm: at.k_norm.to_f32(),
                    },
                },
            })
            .collect();
        let state = model
            .layers
            .iter()
            .map(|l| match l.mixer {
                Mixer::DeltaNet(_) => State::DeltaNet {
                    conv: vec![0.0; d.conv_dim() * (d.ssm_conv - 1)],
                    s: vec![0.0; d.ssm_dt_rank * d.v_head_dim() * d.ssm_d_state],
                },
                Mixer::Attention(_) => State::Attention {
                    k: Vec::new(),
                    v: Vec::new(),
                },
            })
            .collect();
        Reference {
            model,
            threads: threads.max(1),
            rounding,
            pos: 0,
            vecs,
            state,
            output_norm: model.output_norm.to_f32(),
            hidden: Vec::new(),
        }
    }

    /// The final hidden row of the last `step`: RMSNorm(x) ⊙ output_norm, the head's input (the MTP block's
    /// `h_prev`). If a tap forced `FinalNorm`, the forced value, which is what the head read. None before a step.
    pub fn hidden(&self) -> Option<&[f32]> {
        (!self.hidden.is_empty()).then_some(self.hidden.as_slice())
    }

    /// Positions processed so far.
    pub fn pos(&self) -> u32 {
        self.pos
    }

    /// Runs `token` at the next position and returns that position's logits.
    pub fn step(&mut self, token: u32, tap: &mut dyn Tap) -> Result<Vec<f32>, StepError> {
        let m: &'a Model<'a> = self.model;
        let d = &m.dims;
        if token as usize >= d.n_vocab {
            return Err(StepError::TokenOutOfRange {
                token,
                n_vocab: d.n_vocab,
            });
        }
        let p = self.pos;
        // §2: the embedding row, dequantized, not scaled.
        let mut x = vec![0f32; d.n_embd];
        quant::dequant_row(m.embd.ty, m.embd.row(token as usize), &mut x);
        emit(tap, TapName::Embd, p, &mut x);
        for (li, layer) in m.layers.iter().enumerate() {
            let t = |k| TapName::Layer(li as u16, k);
            // §5, residual structure: r = x + Mixer(norm(x)); x' = r + FFN(norm(r)).
            let mut h = ops::rmsnorm(&x, &self.vecs[li].attn_norm, d.rms_eps);
            emit(tap, t(Key::AttnNorm), p, &mut h);
            let mut mix = match &layer.mixer {
                Mixer::DeltaNet(dn) => self.deltanet(li, dn, &h, tap),
                Mixer::Attention(at) => self.attention(li, at, &h, tap),
            };
            emit(tap, t(Key::MixerOut), p, &mut mix);
            let mut r: Vec<f32> = x.iter().zip(&mix).map(|(a, b)| a + b).collect();
            emit(tap, t(Key::Resid), p, &mut r);
            let mut f = ops::rmsnorm(&r, &self.vecs[li].ffn_norm, d.rms_eps);
            emit(tap, t(Key::FfnNorm), p, &mut f);
            let names = |k| Some(t(k));
            let ffn = ffn_block(
                d,
                layer,
                &self.vecs[li].shexp_gate,
                &f,
                p,
                (self.threads, self.rounding),
                &names,
                tap,
            );
            x = r.iter().zip(&ffn).map(|(a, b)| a + b).collect();
            emit(tap, t(Key::Out), p, &mut x);
        }
        let mut fnorm = ops::rmsnorm(&x, &self.output_norm, d.rms_eps);
        emit(tap, TapName::FinalNorm, p, &mut fnorm);
        let logits = project(&m.output, &fnorm, self.threads, self.rounding);
        tap.tap(TapName::Logits, p, &logits);
        self.hidden = fnorm;
        self.pos += 1;
        Ok(logits)
    }

    /// §4: one Gated DeltaNet step.
    fn deltanet(&mut self, li: usize, dn: &DeltaNet, h: &[f32], tap: &mut dyn Tap) -> Vec<f32> {
        let m: &'a Model<'a> = self.model;
        let d = &m.dims;
        let (p, eps, threads, r) = (self.pos, d.rms_eps, self.threads, self.rounding);
        let t = |k| TapName::Layer(li as u16, k);
        let MixerVecs::DeltaNet {
            conv1d,
            dt_bias,
            a,
            norm,
        } = &self.vecs[li].mixer
        else {
            unreachable!("layer {li} is a DeltaNet layer")
        };
        let mut qkv = project(&dn.qkv, h, threads, r);
        emit(tap, t(Key::GdnQkv), p, &mut qkv);
        let mut z = project(&dn.gate, h, threads, r);
        emit(tap, t(Key::GdnZ), p, &mut z);
        let (mut beta, mut g) = ops::gdn_gates(
            &project(&dn.beta, h, threads, r),
            &project(&dn.alpha, h, threads, r),
            dt_bias,
            a,
        );
        emit(tap, t(Key::GdnBeta), p, &mut beta);
        emit(tap, t(Key::GdnG), p, &mut g);
        let State::DeltaNet { conv, s } = &mut self.state[li] else {
            unreachable!("layer {li} is a DeltaNet layer")
        };
        let mut y = ops::conv_step(conv, conv1d, &qkv, d.ssm_conv);
        emit(tap, t(Key::GdnConv), p, &mut y);
        let (mut q, mut k, mut v) = ops::split_qkv(&y, d, eps);
        emit(tap, t(Key::GdnQ), p, &mut q);
        emit(tap, t(Key::GdnK), p, &mut k);
        emit(tap, t(Key::GdnV), p, &mut v);
        let mut o = ops::delta_rule(s, &q, &k, &v, &g, &beta, d);
        emit(tap, t(Key::GdnO), p, &mut o);
        let mut n = ops::gated_norm(&o, &z, norm, eps, d.v_head_dim());
        emit(tap, t(Key::GdnNormed), p, &mut n);
        project(&dn.out, &n, threads, r)
    }

    /// §3: one full-attention step.
    fn attention(&mut self, li: usize, at: &Attention, h: &[f32], tap: &mut dyn Tap) -> Vec<f32> {
        let m: &'a Model<'a> = self.model;
        let (p, threads, r) = (self.pos, self.threads, self.rounding);
        let MixerVecs::Attention { q_norm, k_norm } = &self.vecs[li].mixer else {
            unreachable!("layer {li} is an attention layer")
        };
        let State::Attention { k: kc, v: vc } = &mut self.state[li] else {
            unreachable!("layer {li} is an attention layer")
        };
        let names = |k| Some(TapName::Layer(li as u16, k));
        attention_step(
            &m.dims,
            at,
            [q_norm, k_norm],
            h,
            p,
            (kc, vc),
            (threads, r),
            (&names, false, false),
            tap,
        )
    }
}

/// Maps a layer's key to the tap it is reported under, or None for no tap. The main layers and the MTP block share
/// `attention_step` and `ffn_block`; each brings its own names.
type Names<'n> = &'n dyn Fn(Key) -> Option<TapName>;

fn emit_key(tap: &mut dyn Tap, names: Names, key: Key, pos: u32, v: &mut [f32]) {
    if let Some(n) = names(key) {
        emit(tap, n, pos, v);
    }
}

/// Appends one position's key and value rows to a cache, rounded to f16 as the engine's cache is unless the mode is
/// `Exact`.
fn push_kv(kc: &mut Vec<f32>, vc: &mut Vec<f32>, k: &[f32], v: &[f32], r: Rounding) {
    if r != Rounding::Exact {
        // llama.cpp and the GPU engine keep K and V in an f16 cache.
        kc.extend(k.iter().map(|&x| f16r(x)));
        vc.extend(v.iter().map(|&x| f16r(x)));
    } else {
        kc.extend_from_slice(k);
        vc.extend_from_slice(v);
    }
}

/// §3: one full-attention step at position `p` over the rows in `(kc, vc)`, which it appends this position's key and
/// value to first: the query attends all `kc.len() / (n_head_kv × head_dim)` rows (RoPE is in the keys, so which
/// positions they hold doesn't matter here). `norms` is `[q_norm, k_norm]`, `cache` is `(kc, vc)`, `run` is
/// `(threads, rounding)`. `names` is `(map, tap_before_rope, kv_only)`: the map takes the keys this reports;
/// `tap_before_rope` reports `AttnQ` and `AttnK` after their norm and before RoPE (the MTP's `mtp_Qcur_normed`/`mtp_Kcur_normed`) instead of after it (the main layers').
/// `kv_only` stops after the cache write and returns nothing (a key/value-only row).
#[allow(clippy::too_many_arguments)]
fn attention_step(
    d: &Dims,
    at: &Attention,
    norms: [&[f32]; 2],
    h: &[f32],
    p: u32,
    cache: (&mut Vec<f32>, &mut Vec<f32>),
    run: (usize, Rounding),
    names: (Names, bool, bool),
    tap: &mut dyn Tap,
) -> Vec<f32> {
    let [q_norm, k_norm] = norms;
    let (kc, vc) = cache;
    let (threads, r) = run;
    let (names, tap_before_rope, kv_only) = names;
    let eps = d.rms_eps;
    let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
    // A key/value-only row (the MTP's) skips the query and everything after the cache write.
    let mut qg = (!kv_only).then(|| {
        let (mut q, gate) = ops::split_q_gate(&project(&at.q, h, threads, r), hd);
        ops::norm_heads(&mut q, q_norm, eps, hd);
        (q, gate)
    });
    let mut k = project(&at.k, h, threads, r);
    let mut v = project(&at.v, h, threads, r);
    ops::norm_heads(&mut k, k_norm, eps, hd);
    if tap_before_rope {
        if let Some((q, _)) = &mut qg {
            emit_key(tap, names, Key::AttnQ, p, q);
        }
        emit_key(tap, names, Key::AttnK, p, &mut k);
    }
    if let Some((q, _)) = &mut qg {
        ops::rope_heads(q, hd, p, d.n_rot, d.rope_base);
    }
    ops::rope_heads(&mut k, hd, p, d.n_rot, d.rope_base);
    if !tap_before_rope {
        if let Some((q, _)) = &mut qg {
            emit_key(tap, names, Key::AttnQ, p, q);
        }
        emit_key(tap, names, Key::AttnK, p, &mut k);
    }
    emit_key(tap, names, Key::AttnV, p, &mut v);
    push_kv(kc, vc, &k, &v, r);
    let Some((q, gate)) = qg else {
        return Vec::new();
    };
    let n_ctx = kc.len() / (nkv * hd);
    let mut pre = if r == Rounding::LlamaCpu {
        let (kvw, group) = (nkv * hd, nh / nkv);
        let mut pre = vec![0f32; nh * hd];
        for i in 0..nh {
            // Query head i reads KV head i / (n_head / n_head_kv): blocked, not modulo.
            let kvh = i / group;
            let keys = |tp: usize| &kc[tp * kvw + kvh * hd..][..hd];
            let values = |tp: usize| &vc[tp * kvw + kvh * hd..][..hd];
            let qh = &q[i * hd..(i + 1) * hd];
            flash_attention_f16(qh, n_ctx, keys, values, &mut pre[i * hd..(i + 1) * hd]);
        }
        pre
    } else {
        ops::attention(&q, kc, vc, n_ctx, d)
    };
    emit_key(tap, names, Key::AttnPregate, p, &mut pre);
    let mut gated: Vec<f32> = pre
        .iter()
        .zip(&gate)
        .map(|(a, g)| a * ops::sigmoid(*g))
        .collect();
    emit_key(tap, names, Key::AttnGated, p, &mut gated);
    project(&at.out, &gated, threads, r)
}

/// §5: the router, the 8 routed experts and the gated shared expert, for the normed row `f` at position `p`.
/// `shexp_gate` is the shared expert's dequantized gate row.
#[allow(clippy::too_many_arguments)]
fn ffn_block(
    d: &Dims,
    l: &Layer,
    shexp_gate: &[f32],
    f: &[f32],
    p: u32,
    run: (usize, Rounding),
    names: Names,
    tap: &mut dyn Tap,
) -> Vec<f32> {
    let (threads, r) = run;
    let mut logits = project(&l.moe.router, f, threads, r);
    emit_key(tap, names, Key::RouterLogits, p, &mut logits);
    let mut probs = ops::softmax(&logits);
    emit_key(tap, names, Key::RouterProbs, p, &mut probs);
    let mut ids: Vec<f32> = ops::top_k(&probs, d.n_expert_used)
        .into_iter()
        .map(|e| e as f32)
        .collect();
    emit_key(tap, names, Key::TopK, p, &mut ids);
    let top: Vec<usize> = ids.iter().map(|&e| e as usize).collect();
    let mut w = ops::route_weights(&probs, &top);
    emit_key(tap, names, Key::TopKW, p, &mut w);
    // Each routed expert's silu(gate · f) ⊙ (up · f), in the routed order; forced, the down projections start
    // from llama.cpp's values, so 8-bit rounding of an unforced intermediate can't flip.
    let ff = d.n_ff_exp;
    let mut act: Vec<f32> = top
        .iter()
        .flat_map(|&e| {
            gate_up(
                &l.moe.gate_exps.expert(e, ff),
                &l.moe.up_exps.expert(e, ff),
                f,
                threads,
                r,
            )
        })
        .collect();
    emit_key(tap, names, Key::MoeSwiglu, p, &mut act);
    let ys: Vec<f32> = top
        .iter()
        .enumerate()
        .flat_map(|(j, &e)| {
            project(
                &l.moe.down_exps.expert(e, d.n_embd),
                &act[j * ff..(j + 1) * ff],
                threads,
                r,
            )
        })
        .collect();
    let mut moe = ops::moe_sum(&ys, &w, d.n_embd);
    emit_key(tap, names, Key::MoeOut, p, &mut moe);
    let mut sact = gate_up(&l.shared.gate, &l.shared.up, f, threads, r);
    emit_key(tap, names, Key::ShexpSwiglu, p, &mut sact);
    let sh = project(&l.shared.down, &sact, threads, r);
    // The gate is a matmul too: llama.cpp's CPU rounds its activation to the gate's type (BF16 in blk.40; F32 is
    // untouched, so the main layers are unchanged).
    let fg = if r == Rounding::LlamaCpu {
        quant::round_like_llama_cpu(l.shared.gate_inp.ty, f)
    } else {
        f.to_vec()
    };
    let sg = ops::sigmoid(ops::dot(shexp_gate, &fg) as f32);
    let mut shexp: Vec<f32> = sh.iter().map(|v| v * sg).collect();
    emit_key(tap, names, Key::ShexpOut, p, &mut shexp);
    let mut out: Vec<f32> = moe.iter().zip(&shexp).map(|(a, b)| a + b).collect();
    emit_key(tap, names, Key::FfnOut, p, &mut out);
    out
}

/// The MTP block's (`blk.40`) map from the shared functions' keys to its taps (semantics reference §6). The 25 taps of
/// `MtpKey::ALL` include the attention gate's product, `MtpKey::AttnGated`: the oracle-assigned `mtp_attn_gated-40`
/// (llama.cpp leaves that node unnamed). Keys it doesn't name aren't tapped.
fn mtp_names(k: Key) -> Option<TapName> {
    Some(TapName::Mtp(match k {
        Key::AttnQ => MtpKey::Q,
        Key::AttnK => MtpKey::K,
        Key::AttnV => MtpKey::V,
        Key::AttnPregate => MtpKey::AttnPregate,
        Key::AttnGated => MtpKey::AttnGated,
        Key::RouterLogits => MtpKey::RouterLogits,
        Key::RouterProbs => MtpKey::RouterProbs,
        Key::TopK => MtpKey::TopK,
        Key::TopKW => MtpKey::TopKW,
        Key::MoeSwiglu => MtpKey::MoeSwiglu,
        Key::ShexpSwiglu => MtpKey::ShexpSwiglu,
        Key::MoeOut => MtpKey::MoeOut,
        Key::ShexpOut => MtpKey::ShexpGated,
        Key::FfnOut => MtpKey::FfnOut,
        _ => return None,
    }))
}

/// `blk.40` over the main model's embedding and head (semantics reference §6): the MTP drafter's CPU
/// reference. It keeps its own key/value rows, one per position it has run on, in position order, and a row attends
/// only those: a position it never ran on is a hole, not a zero row.
pub struct MtpReference<'a> {
    model: &'a Model<'a>,
    threads: usize,
    rounding: Rounding,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    head_norm: Vec<f32>,
    attn_norm: Vec<f32>,
    post_norm: Vec<f32>,
    shexp_gate: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    /// The position of each row, ascending.
    rows: Vec<u32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl<'a> MtpReference<'a> {
    pub fn new(model: &'a Model<'a>, threads: usize, rounding: Rounding) -> MtpReference<'a> {
        let mtp = &model.mtp;
        let Mixer::Attention(at) = &mtp.layer.mixer else {
            unreachable!("blk.{} is a full-attention block", model.dims.n_layer)
        };
        MtpReference {
            model,
            threads: threads.max(1),
            rounding,
            enorm: mtp.enorm.to_f32(),
            hnorm: mtp.hnorm.to_f32(),
            head_norm: mtp.shared_head_norm.to_f32(),
            attn_norm: mtp.layer.attn_norm.to_f32(),
            post_norm: mtp.layer.post_attention_norm.to_f32(),
            shexp_gate: mtp.layer.shared.gate_inp.to_f32(),
            q_norm: at.q_norm.to_f32(),
            k_norm: at.k_norm.to_f32(),
            rows: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
        }
    }

    /// The positions that have a row, ascending.
    pub fn positions(&self) -> &[u32] {
        &self.rows
    }

    /// A KV-only row at `pos`: writes its K and V and returns nothing. The key and value depend only on the
    /// row's `u`, so there's no query, no attention, no MoE and no head.
    pub fn kv_row(&mut self, pos: u32, token: u32, h_prev: &[f32]) {
        let m: &'a Model<'a> = self.model;
        let d = &m.dims;
        let (threads, r) = (self.threads, self.rounding);
        let (_, h) = self.front(pos, token, h_prev, &mut NullTap);
        let Mixer::Attention(at) = &m.mtp.layer.mixer else {
            unreachable!("blk.{} is a full-attention block", d.n_layer)
        };
        attention_step(
            d,
            at,
            [&self.q_norm, &self.k_norm],
            &h,
            pos,
            (&mut self.k, &mut self.v),
            (threads, r),
            (&mtp_names, true, true),
            &mut NullTap,
        );
        self.rows.push(pos);
    }

    /// A whole row at `pos` for `token` and `h_prev`, the main model's final hidden row (after `output_norm`) at
    /// `pos − 1`, or a previous row's `m`. Returns its logits and `m`, the shared-head-normed row a chained draft
    /// takes as `h_prev`. Taps each `MtpKey` in order at `pos`.
    ///
    /// Panics if `token` is outside the vocabulary or `h_prev` isn't `n_embd` long, or if `pos` isn't past every row
    /// already written (`truncate` first to redo a position): these are a caller's bugs, not inputs, since the
    /// engine's drafts are argmax ids under `n_vocab`.
    pub fn row(
        &mut self,
        pos: u32,
        token: u32,
        h_prev: &[f32],
        tap: &mut dyn Tap,
    ) -> (Vec<f32>, Vec<f32>) {
        let m: &'a Model<'a> = self.model;
        let d = &m.dims;
        let (threads, r) = (self.threads, self.rounding);
        let (u, h) = self.front(pos, token, h_prev, tap);
        let Mixer::Attention(at) = &m.mtp.layer.mixer else {
            unreachable!("blk.{} is a full-attention block", d.n_layer)
        };
        let t = |k| TapName::Mtp(k);
        let mut mix = attention_step(
            d,
            at,
            [&self.q_norm, &self.k_norm],
            &h,
            pos,
            (&mut self.k, &mut self.v),
            (threads, r),
            (&mtp_names, true, false),
            tap,
        );
        self.rows.push(pos);
        emit(tap, t(MtpKey::AttnOut), pos, &mut mix);
        let mut resid: Vec<f32> = u.iter().zip(&mix).map(|(a, b)| a + b).collect();
        emit(tap, t(MtpKey::AttnResidual), pos, &mut resid);
        let mut f = ops::rmsnorm(&resid, &self.post_norm, d.rms_eps);
        emit(tap, t(MtpKey::PostNorm), pos, &mut f);
        let ffn = ffn_block(
            d,
            &m.mtp.layer,
            &self.shexp_gate,
            &f,
            pos,
            (threads, r),
            &mtp_names,
            tap,
        );
        let mut y: Vec<f32> = resid.iter().zip(&ffn).map(|(a, b)| a + b).collect();
        emit(tap, t(MtpKey::PostFfn), pos, &mut y);
        let mut head = ops::rmsnorm(&y, &self.head_norm, d.rms_eps);
        emit(tap, t(MtpKey::HeadNorm), pos, &mut head);
        // The embedding and the head are the main model's: the GGUF has no `nextn.embed_tokens` or
        // `nextn.shared_head_head` (`qwen35moe.cpp:584, 734`).
        let logits = project(&m.output, &head, threads, r);
        tap.tap(t(MtpKey::Logits), pos, &logits);
        (logits, head)
    }

    /// Drops the rows at positions >= `pos` (the MTP's rollback).
    pub fn truncate(&mut self, pos: u32) {
        let keep = self.rows.partition_point(|&p| p < pos);
        let w = self.model.dims.n_head_kv * self.model.dims.head_dim;
        self.rows.truncate(keep);
        self.k.truncate(keep * w);
        self.v.truncate(keep * w);
    }

    /// `u` and `RMSNorm(u) ⊙ attn_norm` for the row at `pos`, with the taps up to `AttnNorm`
    /// (`qwen35moe.cpp:605-620`): `e = RMSNorm(embd(t)) ⊙ enorm`, `g = RMSNorm(h_prev) ⊙ hnorm`,
    /// `u = eh_proj · [e ; g]`, the embedding half first.
    fn front(
        &mut self,
        pos: u32,
        token: u32,
        h_prev: &[f32],
        tap: &mut dyn Tap,
    ) -> (Vec<f32>, Vec<f32>) {
        let m: &'a Model<'a> = self.model;
        let d = &m.dims;
        assert!(
            (token as usize) < d.n_vocab,
            "token {token} is outside the vocabulary ({} tokens)",
            d.n_vocab
        );
        assert_eq!(h_prev.len(), d.n_embd, "h_prev is one hidden row");
        assert!(
            self.rows.last().is_none_or(|&last| pos > last),
            "the MTP's rows ascend: position {pos} is not past {:?}; truncate first",
            self.rows.last()
        );
        let t = |k| TapName::Mtp(k);
        let mut x = vec![0f32; d.n_embd];
        quant::dequant_row(m.embd.ty, m.embd.row(token as usize), &mut x);
        let mut e = ops::rmsnorm(&x, &self.enorm, d.rms_eps);
        emit(tap, t(MtpKey::Enorm), pos, &mut e);
        let mut g = ops::rmsnorm(h_prev, &self.hnorm, d.rms_eps);
        emit(tap, t(MtpKey::Hnorm), pos, &mut g);
        let mut cat: Vec<f32> = e.into_iter().chain(g).collect();
        emit(tap, t(MtpKey::Concat), pos, &mut cat);
        let mut u = project(&m.mtp.eh_proj, &cat, self.threads, self.rounding);
        emit(tap, t(MtpKey::EhProj), pos, &mut u);
        let mut h = ops::rmsnorm(&u, &self.attn_norm, d.rms_eps);
        emit(tap, t(MtpKey::AttnNorm), pos, &mut h);
        (u, h)
    }
}

/// Reports `v` to `tap`; if the tap supplies a forced value (teacher forcing), the computation continues
/// from that instead.
fn emit(tap: &mut dyn Tap, name: TapName, pos: u32, v: &mut [f32]) {
    tap.tap(name, pos, v);
    if let Some(f) = tap.forced(name, pos) {
        v.copy_from_slice(f);
    }
}

/// silu(gate · x) ⊙ (up · x): an expert's activation, before its down projection.
fn gate_up(gate: &Weight, up: &Weight, x: &[f32], threads: usize, r: Rounding) -> Vec<f32> {
    ops::swiglu(&project(gate, x, threads, r), &project(up, x, threads, r))
}

/// A matmul with its activations rounded as `r` says.
fn project(w: &Weight, x: &[f32], threads: usize, r: Rounding) -> Vec<f32> {
    match r {
        Rounding::Exact | Rounding::F16Kv => ops::matvec(w, x, threads),
        Rounding::LlamaCpu => ops::matvec(w, &quant::round_like_llama_cpu(w.ty, x), threads),
    }
}

fn f16r(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

/// llama.cpp's CPU flash attention for one query head over f16 keys and values (ggml's
/// `flash_attn_ext_f16_one_chunk`): the query rounded to f16, an online softmax in f32, and a value accumulator
/// rounded to f16 after every rescale and multiply-add.
fn flash_attention_f16<'k>(
    q: &[f32],
    n_ctx: usize,
    keys: impl Fn(usize) -> &'k [f32],
    values: impl Fn(usize) -> &'k [f32],
    out: &mut [f32],
) {
    let q16: Vec<f32> = q.iter().map(|&x| f16r(x)).collect();
    let scale = 1.0 / (q.len() as f32).sqrt();
    let (mut m, mut sum) = (f32::NEG_INFINITY, 0f32);
    let mut acc = vec![0f32; q.len()];
    for tp in 0..n_ctx {
        let s = ops::dot(keys(tp), &q16) as f32 * scale;
        let (mut ms, mut vs) = (1f32, 1f32);
        if s > m {
            let old = m;
            m = s;
            ms = (old - m).exp();
            for a in acc.iter_mut() {
                *a = f16r(*a * ms);
            }
        } else {
            vs = (s - m).exp();
        }
        for (a, v) in acc.iter_mut().zip(values(tp)) {
            *a = f16r(v.mul_add(vs, *a));
        }
        sum = sum * ms + vs;
    }
    let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
    for (o, a) in out.iter_mut().zip(&acc) {
        *o = a * inv;
    }
}
