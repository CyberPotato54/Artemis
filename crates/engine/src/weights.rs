//! Getting the model onto the GPU: the non-expert weights uploaded to VRAM; the embedding and
//! every expert copied into host memory (host-cached), experts packed as slots; and the slot table, which points each expert
//! at its cache slot (VRAM) or its host copy (HOST). With `mtp`, `blk.40`'s non-expert weights go to VRAM too and
//! its experts to host memory, HOST in the table at load: only the MTP slots (`mtp_slots.rs`) point some of them at
//! VRAM later, through their own updates, never the trunk's cache.
use crate::{
    decode::Scratch,
    session::{session_bytes_mtp, MtpOptions},
    Error,
};
use kernels::dispatch::{
    check_slot_addr, slot_entry, MatW, SlotLayout, ENTRY_BYTES, WHERE_CPU, WHERE_HOST, WHERE_VRAM,
};
use qwen36::{
    dims::Dims,
    model::{Layer, Mixer, Model, Weight},
    quant::WType,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// VRAM left free beyond everything Artemis allocates.
pub const VRAM_MARGIN: u64 = 512 << 20;

/// VRAM the driver allocates on first use of the kernels, beyond everything Artemis allocates: measured on the
/// RX 9070 XT (Adrenalin 26.8.1) as about 25 MiB on the first prefill chunk and about 1 MiB on the first decode, then
/// flat. The cache is sized with it reserved beside `VRAM_MARGIN`, so the margin stays free once prefill has run.
pub const DRIVER_WORK: u64 = 64 << 20;

/// What a model needs: VRAM for the non-expert weights of layers 0..n_layer and the output head, and host memory
/// for the embedding and those layers' experts; with the MTP (`need`'s `mtp`), also `blk.40`'s non-expert weights and
/// experts. `session` is the VRAM of the sequence state the caller will ask for, if it said (`with_session`), the
/// MTP's rows and ring included. `prefill` and `prefill_host` are the VRAM and the host memory of prefill's buffers
/// (`with_prefill`); `check_fits` counts the host part with `host`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Need {
    pub vram: u64,
    pub host: u64,
    pub session: u64,
    pub prefill: u64,
    pub prefill_host: u64,
}

impl Need {
    /// This need, plus a session of `ctx` positions, with the MTP's rows and ring when `mtp` is set.
    pub fn with_session(self, d: &Dims, ctx: usize, mtp: Option<MtpOptions>) -> Need {
        Need {
            session: session_bytes_mtp(d, ctx, mtp),
            ..self
        }
    }

    /// This need, plus prefill's buffers for chunks of up to `chunk` tokens, in VRAM and in host memory. Like
    /// `with_session` it sets them, so asking twice counts them once. `verify_ctx` is `prefill_bytes`'s: the same value
    /// the allocation gets, or the plan and the allocation disagree.
    pub fn with_prefill(self, d: &Dims, chunk: usize, verify_ctx: Option<usize>) -> Need {
        Need {
            prefill: crate::prefill::prefill_bytes(d, chunk, verify_ctx),
            prefill_host: crate::prefill::prefill_host_bytes(d, chunk),
            ..self
        }
    }
}

fn bytes(w: &Weight) -> u64 {
    w.bytes.len() as u64
}

/// A layer's tensors by name, past `blk.{l}.`, in the order `Weights::upload` uploads them: the mixer's, then the norms,
/// the router and the shared expert.
const DELTANET: [&str; 9] = [
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_beta.weight",
    "ssm_alpha.weight",
    "ssm_conv1d.weight",
    "ssm_dt.bias",
    "ssm_a",
    "ssm_norm.weight",
    "ssm_out.weight",
];
const ATTENTION: [&str; 6] = [
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
];
const COMMON: [&str; 7] = [
    "attn_norm.weight",
    "post_attention_norm.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_inp_shexp.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
];
/// `blk.40`'s own tensors, uploaded after its layer's.
const NEXTN: [&str; 4] = [
    "nextn.eh_proj.weight",
    "nextn.enorm.weight",
    "nextn.hnorm.weight",
    "nextn.shared_head_norm.weight",
];

/// One layer's non-expert weights with their names, in upload order.
fn layer_weights<'a>(l: &Layer<'a>) -> Vec<(&'static str, Weight<'a>)> {
    let mut ws: Vec<(&'static str, Weight<'a>)> = match &l.mixer {
        Mixer::DeltaNet(dn) => DELTANET
            .into_iter()
            .zip([
                dn.qkv, dn.gate, dn.beta, dn.alpha, dn.conv1d, dn.dt_bias, dn.a, dn.norm, dn.out,
            ])
            .collect(),
        Mixer::Attention(at) => ATTENTION
            .into_iter()
            .zip([at.q, at.k, at.v, at.out, at.q_norm, at.k_norm])
            .collect(),
    };
    ws.extend(COMMON.into_iter().zip([
        l.attn_norm,
        l.post_attention_norm,
        l.moe.router,
        l.shared.gate_inp,
        l.shared.gate,
        l.shared.up,
        l.shared.down,
    ]));
    ws
}

/// The non-expert weights of layers 0..n_layer and the output head, in the order `Weights::upload` uploads them; with
/// `mtp`, `blk.40`'s after them.
pub(crate) fn vram_weights<'a>(m: &Model<'a>, mtp: bool) -> Vec<Weight<'a>> {
    named_vram_weights(m, mtp)
        .into_iter()
        .map(|(_, w)| w)
        .collect()
}

/// `vram_weights` with each weight's name.
fn named_vram_weights<'a>(m: &Model<'a>, mtp: bool) -> Vec<(String, Weight<'a>)> {
    let mut ws = Vec::new();
    for (l, layer) in m.layers.iter().enumerate() {
        for (n, w) in layer_weights(layer) {
            ws.push((format!("blk.{l}.{n}"), w));
        }
    }
    ws.push(("output_norm.weight".to_string(), m.output_norm));
    ws.push(("output.weight".to_string(), m.output));
    if mtp {
        let l = m.dims.n_layer;
        for (n, w) in layer_weights(&m.mtp.layer) {
            ws.push((format!("blk.{l}.{n}"), w));
        }
        let own = [
            m.mtp.eh_proj,
            m.mtp.enorm,
            m.mtp.hnorm,
            m.mtp.shared_head_norm,
        ];
        for (n, w) in NEXTN.into_iter().zip(own) {
            ws.push((format!("blk.{l}.{n}"), w));
        }
    }
    ws
}

/// The names `vram_weights` uploads, in its order, for a model of `d`.
fn vram_names(d: &Dims, mtp: bool) -> Vec<String> {
    let layer = |l: usize| -> Vec<String> {
        let mixer: &[&str] = if d.is_attention(l) {
            &ATTENTION
        } else {
            &DELTANET
        };
        mixer
            .iter()
            .chain(COMMON.iter())
            .map(|n| format!("blk.{l}.{n}"))
            .collect()
    };
    let mut names: Vec<String> = (0..d.n_layer).flat_map(layer).collect();
    names.push("output_norm.weight".into());
    names.push("output.weight".into());
    if mtp {
        names.extend(layer(d.n_layer));
        names.extend(NEXTN.iter().map(|n| format!("blk.{}.{n}", d.n_layer)));
    }
    names
}

/// One tensor as the byte arithmetic sees it: its name, type and dims in ne order (`dims[0]` contiguous), as
/// `contract::expected_tensors` lists them.
#[derive(Clone, Debug)]
pub(crate) struct TensorSpec {
    pub name: String,
    pub ty: WType,
    pub dims: Vec<u64>,
}

impl TensorSpec {
    fn bytes(&self) -> u64 {
        self.ty.row_bytes(self.dims[0] as usize) as u64 * self.dims[1..].iter().product::<u64>()
    }
}

/// The tensors of `m` that `need` counts, as specs.
fn model_specs(m: &Model, mtp: bool) -> Vec<TensorSpec> {
    let spec = |name: String, w: &Weight| TensorSpec {
        name,
        ty: w.ty,
        dims: vec![w.cols as u64, w.rows as u64],
    };
    let mut out: Vec<TensorSpec> = named_vram_weights(m, mtp)
        .iter()
        .map(|(n, w)| spec(n.clone(), w))
        .collect();
    out.push(spec("token_embd.weight".into(), &m.embd));
    let layers = m.layers.iter().chain(mtp.then_some(&m.mtp.layer));
    for (l, layer) in layers.enumerate() {
        for (n, w) in [
            ("gate", &layer.moe.gate_exps),
            ("up", &layer.moe.up_exps),
            ("down", &layer.moe.down_exps),
        ] {
            out.push(spec(format!("blk.{l}.ffn_{n}_exps.weight"), w));
        }
    }
    out
}

/// Every allocation of the weights' VRAM arena, in order: the non-expert weights, the slot table and the scratch
/// buffers.
pub(crate) fn vram_plan(m: &Model, mtp: bool) -> Vec<u64> {
    plan_with(&m.dims, vram_weights(m, mtp).iter().map(bytes).collect())
}

/// `sizes` (the weights' allocations), then the slot table and the scratch buffers.
fn plan_with(d: &Dims, mut sizes: Vec<u64>) -> Vec<u64> {
    sizes.push(table_bytes(d));
    sizes.extend(Scratch::bytes(d));
    sizes
}

/// The slot table's bytes: an entry for every expert of every block, the MTP block's included.
pub(crate) fn table_bytes(d: &Dims) -> u64 {
    ((d.n_layer + 1) * d.n_expert * ENTRY_BYTES) as u64
}

/// The byte arithmetic of `need`, over the file's tensors (name, type, dims) rather than a `Model`, so CI can run it
/// on the real model's shapes without the file.
pub(crate) fn need_from_specs(d: &Dims, specs: &[TensorSpec], mtp: bool) -> Need {
    let size: HashMap<&str, u64> = specs.iter().map(|s| (s.name.as_str(), s.bytes())).collect();
    let of = |name: &str| *size.get(name).unwrap_or_else(|| panic!("no tensor {name}"));
    let weights = vram_names(d, mtp).iter().map(|n| of(n)).collect();
    let layers = d.n_layer + usize::from(mtp);
    let host = of("token_embd.weight")
        + (0..layers)
            .flat_map(|l| ["gate", "up", "down"].map(|n| format!("blk.{l}.ffn_{n}_exps.weight")))
            .map(|n| of(&n))
            .sum::<u64>();
    Need {
        vram: vk::planned_bytes(&plan_with(d, weights)),
        host,
        session: 0,
        prefill: 0,
        prefill_host: 0,
    }
}

/// VRAM for the weights' arena, as it packs them, and host memory for the embedding (pinned) and the experts
/// (host-cached). `mtp` adds `blk.40`'s: its non-expert weights in VRAM and its experts in host memory.
pub fn need(m: &Model, mtp: bool) -> Need {
    need_from_specs(&m.dims, &model_specs(m, mtp), mtp)
}

/// Refuses, with the sizes, a model whose weights don't fit in what the budget leaves free. VRAM keeps
/// `VRAM_MARGIN` and `DRIVER_WORK` spare. The message names prefill's buffers and the sequence state only when they
/// are counted.
pub fn check_fits(need: &Need, vram_free: u64, host_free: u64) -> Result<(), Error> {
    let host = need.host + need.prefill_host;
    if host > host_free {
        return Err(Error::DoesNotFit {
            what: if need.prefill_host > 0 {
                "the embedding, the experts and prefill's host buffers in host memory"
            } else {
                "the embedding and the experts in host memory"
            },
            need: host,
            free: host_free,
        });
    }
    let vram = need.vram + need.prefill + need.session + VRAM_MARGIN + DRIVER_WORK;
    if vram > vram_free {
        return Err(Error::DoesNotFit {
            what: match (need.prefill > 0, need.session > 0) {
                (true, true) => {
                    "the non-expert weights, prefill's buffers and the sequence state in VRAM, with the 0.5 GiB margin and the driver's working memory,"
                }
                (true, false) => {
                    "the non-expert weights and prefill's buffers in VRAM, with the 0.5 GiB margin and the driver's working memory,"
                }
                (false, true) => {
                    "the non-expert weights and the sequence state in VRAM, with the 0.5 GiB margin and the driver's working memory,"
                }
                (false, false) => "the non-expert weights in VRAM, with the 0.5 GiB margin and the driver's working memory,",
            },
            need: vram,
            free: vram_free,
        });
    }
    Ok(())
}

/// The MTP block's experts (layer n_layer) are never in the trunk's cache. Panics on a VRAM entry for them: only the MTP
/// slots (`mtp_slots.rs`) point them at VRAM, with their own updates.
pub(crate) fn assert_not_cached(d: &Dims, layer: usize, loc: u32) {
    assert!(
        layer < d.n_layer || loc != WHERE_VRAM,
        "blk.{layer} (the MTP block) is never in the cache (C13)"
    );
}

/// The slot table: (n_layer + 1) × n_expert entries. Layer l < n_layer's entry for expert e is
/// `entry(l, e)`: (address, where), a cache slot in VRAM or the expert's host copy (HOST). The MTP block's entries
/// (layer n_layer) are CPU with no address, so the expert kernels refuse them; with `mtp` they are HOST at the
/// address `entry(n_layer, e)` gives, the expert's host copy, and never a cache slot: a VRAM entry there panics. (The
/// MTP slots start empty; their draft passes point the experts they take at VRAM later, `mtp_slots.rs`.)
pub fn slot_table(d: &Dims, entry: impl Fn(usize, usize) -> (u64, u32), mtp: bool) -> Vec<u8> {
    let mut t = Vec::with_capacity((d.n_layer + 1) * d.n_expert * ENTRY_BYTES);
    for l in 0..=d.n_layer {
        for e in 0..d.n_expert {
            t.extend(if l < d.n_layer || mtp {
                let (addr, loc) = entry(l, e);
                assert!(
                    l < d.n_layer || loc == WHERE_HOST,
                    "blk.{l} (the MTP block) is HOST only, never in the cache"
                );
                check_slot_addr(addr, loc);
                slot_entry(addr, loc)
            } else {
                slot_entry(0, WHERE_CPU)
            });
        }
    }
    t
}

/// Layer `l`'s expert slot layout: its gate, up and down weights' types.
pub(crate) fn slot_layout(l: &Layer, d: &Dims) -> SlotLayout {
    SlotLayout {
        gate: l.moe.gate_exps.ty,
        up: l.moe.up_exps.ty,
        down: l.moe.down_exps.ty,
        n_embd: d.n_embd,
        n_ff: d.n_ff_exp,
    }
}

pub(crate) enum GpuMixer {
    DeltaNet {
        qkv: MatW,
        z: MatW,
        beta: MatW,
        alpha: MatW,
        conv1d: u64,
        dt_bias: u64,
        a: u64,
        norm: u64,
        out: MatW,
    },
    Attention {
        q: MatW,
        k: MatW,
        v: MatW,
        out: MatW,
        q_norm: u64,
        k_norm: u64,
    },
}

/// One layer's weights on the GPU, and its experts' slot layout.
pub(crate) struct GpuLayer {
    pub attn_norm: u64,
    pub ffn_norm: u64,
    pub mixer: GpuMixer,
    pub router: MatW,
    pub sh_gate_inp: MatW,
    pub sh_gate: MatW,
    pub sh_up: MatW,
    pub sh_down: MatW,
    pub slot: SlotLayout,
}

/// `blk.40`'s weights on the GPU: its layer (a full-attention block with MoE, its experts HOST), the
/// projection of `[e ; g]` back to the model width, the two input norms and the head's norm.
pub(crate) struct GpuMtp {
    pub layer: GpuLayer,
    pub eh_proj: MatW,
    pub enorm: u64,
    pub hnorm: u64,
    pub head_norm: u64,
}

/// The non-expert weights in VRAM. The arena also holds the slot table and the scratch buffers.
pub(crate) struct Weights {
    pub arena: vk::Arena,
    pub layers: Vec<GpuLayer>,
    pub output_norm: u64,
    pub output: MatW,
    /// `blk.40`, when the load was asked for the MTP (read by the draft pass, `Session::mtp_draft`).
    pub mtp: Option<GpuMtp>,
    /// What `blk.40`'s weights take of the arena (the arena's alignment included); 0 without the MTP.
    pub mtp_bytes: u64,
}

/// Uploads one layer's non-expert weights in `layer_weights`' order through `put`.
fn upload_layer(
    put: &mut dyn FnMut(&Weight) -> Result<MatW, Error>,
    l: &Layer,
    d: &Dims,
) -> Result<GpuLayer, Error> {
    let mixer = match &l.mixer {
        Mixer::DeltaNet(dn) => GpuMixer::DeltaNet {
            qkv: put(&dn.qkv)?,
            z: put(&dn.gate)?,
            beta: put(&dn.beta)?,
            alpha: put(&dn.alpha)?,
            conv1d: put(&dn.conv1d)?.addr,
            dt_bias: put(&dn.dt_bias)?.addr,
            a: put(&dn.a)?.addr,
            norm: put(&dn.norm)?.addr,
            out: put(&dn.out)?,
        },
        Mixer::Attention(at) => GpuMixer::Attention {
            q: put(&at.q)?,
            k: put(&at.k)?,
            v: put(&at.v)?,
            out: put(&at.out)?,
            q_norm: put(&at.q_norm)?.addr,
            k_norm: put(&at.k_norm)?.addr,
        },
    };
    Ok(GpuLayer {
        attn_norm: put(&l.attn_norm)?.addr,
        ffn_norm: put(&l.post_attention_norm)?.addr,
        mixer,
        router: put(&l.moe.router)?,
        sh_gate_inp: put(&l.shared.gate_inp)?,
        sh_gate: put(&l.shared.gate)?,
        sh_up: put(&l.shared.up)?,
        sh_down: put(&l.shared.down)?,
        slot: slot_layout(l, d),
    })
}

/// Adds 1 to the load's progress counter, when there is one.
fn tick(ticks: Option<&AtomicU64>) {
    if let Some(t) = ticks {
        t.fetch_add(1, Ordering::Relaxed);
    }
}

impl Weights {
    /// Uploads the non-expert weights; `ticks` gains 1 after each tensor, the load's progress counter.
    pub fn upload(
        dev: &vk::Device,
        m: &Model,
        mtp: bool,
        ticks: Option<&AtomicU64>,
    ) -> Result<Weights, Error> {
        // Planned for the weights, then the slot table and the scratch buffers, which the engine allocates next.
        let mut arena = vk::Arena::planned(
            vk::MemKind::Vram,
            &vram_plan(m, mtp),
            "the non-expert weights",
        );
        let d = &m.dims;
        let order = vram_weights(m, mtp);
        let mut next = order.iter();
        let mut mtp_bytes = 0u64;
        let (layers, output_norm, output, mtp_weights) = {
            let mut put = |w: &Weight| -> Result<MatW, Error> {
                let want = next.next().expect("vram_weights lists every upload");
                assert!(
                    std::ptr::eq(want.bytes, w.bytes),
                    "uploads follow vram_weights' order"
                );
                let s = arena.alloc(dev, bytes(w))?;
                dev.upload(s, w.bytes)?;
                tick(ticks);
                Ok(MatW {
                    addr: s.addr,
                    ty: w.ty,
                    rows: w.rows,
                    cols: w.cols,
                })
            };
            let mut layers = Vec::with_capacity(m.layers.len());
            for l in &m.layers {
                layers.push(upload_layer(&mut put, l, d)?);
            }
            let output_norm = put(&m.output_norm)?.addr;
            let output = put(&m.output)?;
            let mtp_weights = if mtp {
                let layer = upload_layer(&mut put, &m.mtp.layer, d)?;
                Some(GpuMtp {
                    layer,
                    eh_proj: put(&m.mtp.eh_proj)?,
                    enorm: put(&m.mtp.enorm)?.addr,
                    hnorm: put(&m.mtp.hnorm)?.addr,
                    head_norm: put(&m.mtp.shared_head_norm)?.addr,
                })
            } else {
                None
            };
            (layers, output_norm, output, mtp_weights)
        };
        if mtp {
            let all = vram_weights(m, true);
            let own = &all[all.len() - mtp_weight_count(m)..];
            mtp_bytes = vk::planned_bytes(&own.iter().map(bytes).collect::<Vec<_>>());
        }
        Ok(Weights {
            arena,
            layers,
            output_norm,
            output,
            mtp: mtp_weights,
            mtp_bytes,
        })
    }
}

/// How many of `vram_weights(m, true)`'s last entries are `blk.40`'s.
fn mtp_weight_count(m: &Model) -> usize {
    layer_weights(&m.mtp.layer).len() + NEXTN.len()
}

/// The offset of the slot of `slot_bytes` that starts at device address `addr` in the buffer of `buf_size` bytes at
/// `buf_addr`: Some when `addr` is a whole number of slots past the buffer's start and the slot ends inside the buffer;
/// None before the buffer, mid-slot, or for a slot that would run past its end (`HostCopies::slot_at`).
fn slot_offset(buf_addr: u64, buf_size: u64, slot_bytes: u64, addr: u64) -> Option<u64> {
    let off = addr.checked_sub(buf_addr)?;
    (off.is_multiple_of(slot_bytes) && off.checked_add(slot_bytes)? <= buf_size).then_some(off)
}

/// The embedding in pinned host memory and the experts in host-cached memory (the CPU miss path reads them; the GPU
/// reads them over PCIe either way), in allocations of at most 1 GiB. Host-cached and pinned memory share host heap 0
/// on the RX 9070 XT, so `check_fits`' budget math reads the same heap.
pub(crate) struct HostCopies {
    embd: Vec<vk::Buffer>,
    embd_rows_per: usize,
    embd_row_bytes: usize,
    pub embd_ty: WType,
    /// Per layer: the buffers, slots per buffer, and bytes per slot.
    experts: Vec<(Vec<vk::Buffer>, usize, usize)>,
}

impl HostCopies {
    /// Copies the embedding and the experts to host memory; `ticks` gains 1 after the embedding and after each layer's
    /// experts, the load's progress counter.
    pub fn copy(
        dev: &vk::Device,
        m: &Model,
        mtp: bool,
        ticks: Option<&AtomicU64>,
    ) -> Result<HostCopies, Error> {
        let d = &m.dims;
        let rb = m.embd.ty.row_bytes(m.embd.cols);
        let per = (vk::MAX_ALLOCATION as usize / rb).max(1);
        let mut embd = Vec::new();
        for rows in m.embd.bytes.chunks(per * rb) {
            let b = dev.alloc(rows.len() as u64, vk::MemKind::HostPinned, "the embedding")?;
            b.write(0, rows);
            embd.push(b);
        }
        tick(ticks);
        let mut experts = Vec::with_capacity(m.layers.len() + usize::from(mtp));
        // With the MTP, `blk.40`'s experts follow the main layers' as layer n_layer: HOST copies, never cache slots.
        for (li, l) in m
            .layers
            .iter()
            .chain(mtp.then_some(&m.mtp.layer))
            .enumerate()
        {
            let lay = slot_layout(l, d);
            let sb = lay.bytes();
            let per = (vk::MAX_ALLOCATION as usize / sb).max(1);
            let what = format!("the experts of layer {li}");
            let mut bufs = Vec::new();
            for first in (0..d.n_expert).step_by(per) {
                let n = per.min(d.n_expert - first);
                let b = dev.alloc((n * sb) as u64, vk::MemKind::HostCached, &what)?;
                for e in first..first + n {
                    let at = ((e - first) * sb) as u64;
                    b.write(at, l.moe.gate_exps.expert(e, d.n_ff_exp).bytes);
                    b.write(
                        at + lay.up_offset() as u64,
                        l.moe.up_exps.expert(e, d.n_ff_exp).bytes,
                    );
                    b.write(
                        at + lay.down_offset() as u64,
                        l.moe.down_exps.expert(e, d.n_embd).bytes,
                    );
                }
                bufs.push(b);
            }
            experts.push((bufs, per, sb));
            tick(ticks);
        }
        Ok(HostCopies {
            embd,
            embd_rows_per: per,
            embd_row_bytes: rb,
            embd_ty: m.embd.ty,
            experts,
        })
    }

    /// Embedding row 0's device address and a row's bytes, for `argmax_rows` (`base + id * row_bytes`): the embedding must
    /// be one allocation. `assert!`s it is (the real model's is 540 MB, under the 1 GiB split).
    pub fn embd_base(&self) -> (u64, usize) {
        assert!(
            self.embd.len() == 1,
            "the MTP needs the embedding in one allocation, not {} (C21)",
            self.embd.len()
        );
        (self.embd[0].addr(), self.embd_row_bytes)
    }

    /// The device address of `token`'s embedding row.
    pub fn embd_row(&self, token: u32) -> u64 {
        let t = token as usize;
        self.embd[t / self.embd_rows_per].addr()
            + ((t % self.embd_rows_per) * self.embd_row_bytes) as u64
    }

    /// Expert `e`'s host slot in `layer`: what the cache copies from.
    pub fn slot_slice(&self, layer: usize, e: usize) -> vk::Slice {
        let (bufs, per, sb) = &self.experts[layer];
        bufs[e / per].slice(((e % per) * sb) as u64, *sb as u64)
    }

    /// The CPU address of expert `e`'s slot in `layer`, in the copies' host-cached mapping: what the CPU miss path's
    /// coordinator reads. Valid while these copies live; the coordinator is joined before they drop, or
    /// they are leaked (`leak_experts`) if it was detached.
    pub(crate) fn slot_ptr(&self, layer: usize, e: usize) -> *const u8 {
        let (bufs, per, sb) = &self.experts[layer];
        let b = &bufs[e / per];
        assert!(!b.ptr().is_null(), "the expert copies are mapped");
        // SAFETY: slot e % per of this buffer lies inside its mapping (`copy` sized the buffer for its slots).
        unsafe { b.ptr().add((e % per) * sb) }
    }

    /// Leaks the expert copies instead of freeing them: a detached CPU-path coordinator (`CpuPath::shutdown`) may
    /// still read them. Each leaked buffer keeps the device alive, so the mappings stay valid. Only as the engine
    /// drops: no slot remains to copy from afterwards.
    pub(crate) fn leak_experts(&mut self) {
        std::mem::forget(std::mem::take(&mut self.experts));
    }

    /// The device address of expert `e`'s slot in `layer`.
    pub fn slot_addr(&self, layer: usize, e: usize) -> u64 {
        self.slot_slice(layer, e).addr
    }

    /// The host slot of `layer` that starts at device address `addr`, if any: what a staging copy can copy from.
    /// An address anywhere else (a test's entry, another layer's copy) gives None.
    pub fn slot_at(&self, layer: usize, addr: u64) -> Option<vk::Slice> {
        let (bufs, _, sb) = &self.experts[layer];
        let sb = *sb as u64;
        bufs.iter()
            .find_map(|b| slot_offset(b.addr(), b.size(), sb, addr).map(|off| b.slice(off, sb)))
    }

    /// The bytes of expert `e`'s slot in `layer`.
    pub fn slot_bytes(&self, layer: usize, e: usize) -> Vec<u8> {
        let (bufs, per, sb) = &self.experts[layer];
        let mut out = vec![0u8; *sb];
        bufs[e / per].read(((e % per) * sb) as u64, &mut out);
        out
    }

    /// The host memory the copies hold: the embedding and every layer's experts, the MTP block's included when it was
    /// copied.
    pub fn bytes(&self) -> u64 {
        self.embd.iter().map(|b| b.size()).sum::<u64>()
            + self
                .experts
                .iter()
                .flat_map(|(bufs, _, _)| bufs)
                .map(|b| b.size())
                .sum::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gguf::Gguf;
    use kernels::dispatch::{WHERE_HOST, WHERE_VRAM};
    use qwen36::testing::tiny_model;

    /// The expert kernels read slots 16 bytes at a time: the table the engine builds at load refuses a slot
    /// that isn't 16-byte aligned.
    #[test]
    #[should_panic(expected = "16-byte-aligned slots")]
    fn the_slot_table_refuses_a_slot_the_expert_kernels_cant_read() {
        slot_table(&Dims::tiny(), |_, _| (0x1_0008, WHERE_HOST), false);
    }

    /// Every tensor of the tiny file the load places is placed exactly once: `mtp` includes `blk.4.*` (the MTP block),
    /// else none of it.
    fn check_every_tensor_is_placed_once(mtp: bool) {
        let d = Dims::tiny();
        let g = Gguf::from_bytes(tiny_model(1)).unwrap();
        let m = Model::open(&g, &d).unwrap();
        let n = need(&m, mtp);
        let block = format!("blk.{}.", d.n_layer);
        let total: u64 = g
            .tensors()
            .filter(|t| mtp || !t.name.starts_with(&block))
            .map(|t| t.bytes.len() as u64)
            .sum();
        let weights: u64 = vram_weights(&m, mtp).iter().map(bytes).sum();
        assert_eq!(weights + n.host, total);
        // The VRAM arena also holds the slot table and the scratch buffers, with each allocation 256-byte aligned.
        let (table, scratch) = (
            ((d.n_layer + 1) * d.n_expert * ENTRY_BYTES) as u64,
            crate::decode::Scratch::bytes(&d).iter().sum::<u64>(),
        );
        assert!(n.vram >= weights + table + scratch);
        assert!(n.vram < weights + table + scratch + 256 * vram_plan(&m, mtp).len() as u64);
        // `need` (the names' arithmetic) and the upload's plan (the structs') are one arena.
        assert_eq!(n.vram, vk::planned_bytes(&vram_plan(&m, mtp)));
        let layers = d.n_layer + usize::from(mtp);
        let exps = |l: &Layer| {
            (l.moe.gate_exps.bytes.len() + l.moe.up_exps.bytes.len() + l.moe.down_exps.bytes.len())
                as u64
        };
        assert_eq!(
            n.host,
            m.embd.bytes.len() as u64
                + m.layers
                    .iter()
                    .chain([&m.mtp.layer])
                    .take(layers)
                    .map(exps)
                    .sum::<u64>()
        );
        // The names `need` walks are the uploads' names, in the uploads' order and sizes.
        let by_name: Vec<(String, u64)> = named_vram_weights(&m, mtp)
            .iter()
            .map(|(n, w)| (n.clone(), bytes(w)))
            .collect();
        let names = vram_names(&d, mtp);
        assert_eq!(
            by_name.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
            names
        );
        for (n, b) in &by_name {
            assert_eq!(g.tensor(n).unwrap().bytes.len() as u64, *b, "{n}");
        }
    }

    #[test]
    fn need_places_every_tensor_but_the_mtp_blocks_exactly_once() {
        check_every_tensor_is_placed_once(false);
    }

    #[test]
    fn need_places_every_tensor_exactly_once_with_mtp() {
        check_every_tensor_is_placed_once(true);
    }

    /// The tiny model's `need`, as it was computed before the MTP existed: the embedding and the main layers' experts
    /// in host memory, the main layers' non-expert weights, the table and the scratch buffers in VRAM.
    #[test]
    fn mtp_is_not_uploaded_without_the_mtp_source_on_the_tiny_model() {
        let d = Dims::tiny();
        let g = Gguf::from_bytes(tiny_model(1)).unwrap();
        let m = Model::open(&g, &d).unwrap();
        // The formulas from before the MTP, transcribed.
        let old_host = bytes(&m.embd)
            + m.layers
                .iter()
                .map(|l| bytes(&l.moe.gate_exps) + bytes(&l.moe.up_exps) + bytes(&l.moe.down_exps))
                .sum::<u64>();
        let mut old_plan: Vec<u64> = Vec::new();
        for l in &m.layers {
            match &l.mixer {
                Mixer::DeltaNet(dn) => old_plan.extend(
                    [
                        dn.qkv, dn.gate, dn.beta, dn.alpha, dn.conv1d, dn.dt_bias, dn.a, dn.norm,
                        dn.out,
                    ]
                    .iter()
                    .map(bytes),
                ),
                Mixer::Attention(at) => old_plan.extend(
                    [at.q, at.k, at.v, at.out, at.q_norm, at.k_norm]
                        .iter()
                        .map(bytes),
                ),
            }
            old_plan.extend(
                [
                    l.attn_norm,
                    l.post_attention_norm,
                    l.moe.router,
                    l.shared.gate_inp,
                    l.shared.gate,
                    l.shared.up,
                    l.shared.down,
                ]
                .iter()
                .map(bytes),
            );
        }
        old_plan.extend([bytes(&m.output_norm), bytes(&m.output)]);
        old_plan.push(((d.n_layer + 1) * d.n_expert * ENTRY_BYTES) as u64);
        old_plan.extend(crate::decode::Scratch::bytes(&d));
        assert_eq!(vram_plan(&m, false), old_plan);
        let n = need(&m, false);
        assert_eq!((n.vram, n.host), (vk::planned_bytes(&old_plan), old_host));
        // The literals, so a change to either side shows.
        assert_eq!((n.vram, n.host), TINY_TODAY);
        assert_eq!(
            vram_weights(&m, false).len(),
            old_plan.len() - 1 - Scratch::bytes(&d).len()
        );
        // With the MTP the plan only grows, by blk.4's 17 tensors.
        assert_eq!(vram_plan(&m, true).len(), old_plan.len() + 17);
        assert!(need(&m, true).host > n.host && need(&m, true).vram > n.vram);
    }

    /// (vram, host) of `need(tiny, false)`, as the load computed them before the MTP existed.
    const TINY_TODAY: (u64, u64) = (3_086_080, 7_880_704);

    /// QWEN36's tensors as `contract::expected_tensors` lists them, typed as the real GGUF types them (the docs'
    /// reference §5): Q8_0 for the projections, the shared expert and the embedding, Q6_K for the head, K-quants for the
    /// routed experts (down Q5_K, Q6_K in layers 34, 38 and 39), F32 norms and routers (BF16 in `blk.40`).
    fn qwen36_specs() -> Vec<TensorSpec> {
        let d = Dims::QWEN36;
        qwen36::contract::expected_tensors(&d)
            .into_iter()
            .map(|t| {
                let (layer, rest) = match t.name.strip_prefix("blk.") {
                    Some(r) => {
                        let (l, rest) = r.split_once('.').unwrap();
                        (Some(l.parse::<usize>().unwrap()), rest.to_string())
                    }
                    None => (None, t.name.clone()),
                };
                let in_mtp = layer.is_some_and(|l| l >= d.n_layer);
                let ty = match rest.as_str() {
                    "output.weight" => WType::Q6K,
                    "ffn_gate_exps.weight" | "ffn_up_exps.weight" => WType::Q4K,
                    "ffn_down_exps.weight" if matches!(layer, Some(34 | 38 | 39)) => WType::Q6K,
                    "ffn_down_exps.weight" => WType::Q5K,
                    "ffn_gate_inp.weight" | "ffn_gate_inp_shexp.weight" if in_mtp => WType::Bf16,
                    "ffn_gate_inp.weight" | "ffn_gate_inp_shexp.weight" => WType::F32,
                    "ssm_conv1d.weight" | "ssm_dt.bias" | "ssm_a" | "ssm_beta.weight"
                    | "ssm_alpha.weight" => WType::F32,
                    n if n.ends_with("norm.weight") => WType::F32,
                    _ => WType::Q8_0,
                };
                TensorSpec {
                    name: t.name,
                    ty,
                    dims: t.dims,
                }
            })
            .collect()
    }

    #[test]
    fn mtp_is_not_uploaded_without_the_mtp_source() {
        // QWEN36's shapes (no GGUF in CI): without the MTP, `need` counts none of blk.40 and is today's.
        let d = Dims::QWEN36;
        let specs = qwen36_specs();
        let n = need_from_specs(&d, &specs, false);
        assert_eq!((n.vram, n.host), QWEN36_TODAY);
        // Dropping blk.40 from the file changes nothing: the arithmetic never reads it.
        let main_only: Vec<TensorSpec> = specs
            .iter()
            .filter(|s| !s.name.starts_with("blk.40."))
            .cloned()
            .collect();
        assert_eq!(need_from_specs(&d, &main_only, false), n);
    }

    /// (vram, host) of `need(QWEN36, false)` over `qwen36_specs`.
    const QWEN36_TODAY: (u64, u64) = (2_016_229_120, 20_108_869_632);

    #[test]
    fn the_mtp_adds_its_bytes() {
        let d = Dims::QWEN36;
        let specs = qwen36_specs();
        let (off, on) = (
            need_from_specs(&d, &specs, false),
            need_from_specs(&d, &specs, true),
        );
        // blk.40's non-expert tensors are 42,317,824 bytes, its experts 486,539,264 (1,900,544 an expert).
        let non_expert: u64 = specs
            .iter()
            .filter(|s| s.name.starts_with("blk.40.") && !s.name.contains("_exps"))
            .map(TensorSpec::bytes)
            .sum();
        assert_eq!(non_expert, 42_317_824);
        let experts: u64 = specs
            .iter()
            .filter(|s| s.name.starts_with("blk.40.") && s.name.contains("_exps"))
            .map(TensorSpec::bytes)
            .sum();
        assert_eq!(experts, 486_539_264);
        assert_eq!(experts / 256, 1_900_544);
        // Host memory grows by exactly the experts; VRAM by the weights plus the arena's alignment on 17 tensors.
        assert_eq!(on.host - off.host, 486_539_264);
        let grown = on.vram - off.vram;
        assert!(grown >= 42_317_824, "{grown}");
        assert!(grown <= 42_317_824 + 17 * 256, "{grown}");
    }

    #[test]
    fn check_fits_names_what_doesnt_fit_with_sizes() {
        let n = Need {
            vram: 2000 << 20,
            host: 19_000 << 20,
            session: 0,
            prefill: 0,
            prefill_host: 0,
        };
        assert!(check_fits(&n, 3000 << 20, 40_000 << 20).is_ok());
        let e = check_fits(&n, 2200 << 20, 40_000 << 20)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("VRAM") && e.contains("2576 MiB") && e.contains("2200 MiB"),
            "{e}"
        );
        // Nothing of prefill's is counted, so nothing of it is named.
        assert!(!e.contains("prefill"), "{e}");
        let e = check_fits(&n, 3000 << 20, 10_000 << 20)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("in host memory") && e.contains("19000 MiB") && e.contains("10000 MiB"),
            "{e}"
        );
        assert!(!e.contains("prefill"), "{e}");
    }

    #[test]
    fn the_slot_table_holds_each_layers_entries_and_refuses_the_mtp_block() {
        let d = Dims::tiny();
        // Expert 5 of every layer is cached in VRAM; the rest point at their host copies.
        let t = slot_table(
            &d,
            |l, e| {
                assert!(
                    l < d.n_layer,
                    "the closure is never asked for the MTP block without it"
                );
                if e == 5 {
                    (0x2000_0000 + (l * 0x1000) as u64, WHERE_VRAM)
                } else {
                    (0x1000_0000 + ((l * 16 + e) * 0x100) as u64, WHERE_HOST)
                }
            },
            false,
        );
        assert_eq!(t.len(), (d.n_layer + 1) * d.n_expert * ENTRY_BYTES);
        let entry = |l: usize, e: usize| {
            let at = (l * d.n_expert + e) * ENTRY_BYTES;
            (
                u64::from_le_bytes(t[at..at + 8].try_into().unwrap()),
                u32::from_le_bytes(t[at + 8..at + 12].try_into().unwrap()),
            )
        };
        assert_eq!(entry(0, 0), (0x1000_0000, WHERE_HOST));
        assert_eq!(
            entry(3, 15),
            (0x1000_0000 + (3 * 16 + 15) * 0x100, WHERE_HOST)
        );
        assert_eq!(entry(2, 5), (0x2000_2000, WHERE_VRAM));
        for e in 0..d.n_expert {
            assert_eq!(
                entry(d.n_layer, e),
                (0, WHERE_CPU),
                "blk.{}'s expert {e}",
                d.n_layer
            );
        }
    }

    #[test]
    fn the_slot_table_points_the_mtp_block_host_when_asked() {
        let d = Dims::tiny();
        // Every layer's expert 5 is cached; the MTP block's entries are its host copies.
        let entries = |l: usize, e: usize| {
            if e == 5 && l < d.n_layer {
                (0x2000_0000 + (l * 0x1000) as u64, WHERE_VRAM)
            } else {
                (0x1000_0000 + ((l * 16 + e) * 0x100) as u64, WHERE_HOST)
            }
        };
        let t = slot_table(&d, entries, true);
        assert_eq!(t.len(), (d.n_layer + 1) * d.n_expert * ENTRY_BYTES);
        let entry = |l: usize, e: usize| {
            let at = (l * d.n_expert + e) * ENTRY_BYTES;
            (
                u64::from_le_bytes(t[at..at + 8].try_into().unwrap()),
                u32::from_le_bytes(t[at + 8..at + 12].try_into().unwrap()),
            )
        };
        assert_eq!(
            entry(2, 5),
            (0x2000_2000, WHERE_VRAM),
            "main layers as before"
        );
        for e in 0..d.n_expert {
            assert_eq!(
                entry(d.n_layer, e),
                (
                    0x1000_0000 + ((d.n_layer * 16 + e) * 0x100) as u64,
                    WHERE_HOST
                ),
                "blk.{}'s expert {e}",
                d.n_layer
            );
        }
        // The main layers' entries are the same bytes with and without the MTP.
        let without = slot_table(&d, entries, false);
        let main = d.n_layer * d.n_expert * ENTRY_BYTES;
        assert_eq!(t[..main], without[..main]);
    }

    /// blk.40's experts never reach a cache slot, so the table refuses a VRAM entry for them.
    #[test]
    #[should_panic(expected = "HOST only, never in the cache")]
    fn the_slot_table_refuses_a_vram_entry_for_the_mtp_block() {
        slot_table(&Dims::tiny(), |_, _| (0x2000_0000, WHERE_VRAM), true);
    }

    /// The same rule for the engine's own entry setters (`set_slot_entry`, `stage_slot_entry`).
    #[test]
    fn a_vram_entry_is_refused_for_the_mtp_block_only() {
        let d = Dims::tiny();
        assert_not_cached(&d, d.n_layer - 1, WHERE_VRAM);
        assert_not_cached(&d, d.n_layer, WHERE_HOST);
        assert_not_cached(&d, d.n_layer, WHERE_CPU);
        let refused = std::panic::catch_unwind(|| assert_not_cached(&Dims::tiny(), 4, WHERE_VRAM));
        assert!(refused.is_err());
    }

    #[test]
    fn check_fits_counts_prefills_buffers() {
        // A 2,048-token chunk's buffers count against VRAM as the weights do (414 MiB), and against host memory as the
        // experts do (66 MiB: mostly the output head's 64 rows of logits, read back).
        let n = Need {
            vram: 2000 << 20,
            host: 19_000 << 20,
            session: 0,
            prefill: 0,
            prefill_host: 0,
        }
        .with_prefill(&Dims::QWEN36, 2048, None);
        assert_eq!(
            (n.prefill, n.prefill_host),
            (
                crate::prefill::prefill_bytes(&Dims::QWEN36, 2048, None),
                crate::prefill::prefill_host_bytes(&Dims::QWEN36, 2048)
            )
        );
        assert_eq!(
            n.with_prefill(&Dims::QWEN36, 2048, None),
            n,
            "counted once, however often asked"
        );
        assert!(check_fits(&n, 3000 << 20, 40_000 << 20).is_ok());
        let e = check_fits(&n, 2600 << 20, 40_000 << 20)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("prefill's buffers") && e.contains("2600 MiB") && !e.contains("sequence"),
            "{e}"
        );
        assert!(check_fits(&n, 3000 << 20, 19_100 << 20).is_ok());
        let e = check_fits(&n, 3000 << 20, 19_050 << 20)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("in host memory")
                && e.contains("prefill's")
                && e.contains("19067 MiB")
                && e.contains("19050 MiB"),
            "{e}"
        );
        // With a session as well, all three are named.
        let e = check_fits(
            &n.with_session(&Dims::QWEN36, 65_536, None),
            4000 << 20,
            40_000 << 20,
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("prefill's buffers") && e.contains("sequence state"),
            "{e}"
        );
    }

    #[test]
    fn slot_offset_finds_only_whole_slots_inside_the_buffer() {
        // What a staging copy may copy from: a buffer of four 0x100-byte slots and a partial
        // tail at 0x1000, and another buffer at 0x8000.
        let at = |addr| slot_offset(0x1000, 0x450, 0x100, addr);
        assert_eq!(at(0x1000), Some(0), "the first slot");
        assert_eq!(at(0x1300), Some(0x300), "the last whole slot");
        assert_eq!(at(0x1080), None, "mid-slot");
        assert_eq!(at(0x1004), None, "a few bytes in");
        assert_eq!(
            at(0x1400),
            None,
            "one past the end: the slot would run past the buffer"
        );
        assert_eq!(at(0x0F00), None, "below the buffer");
        assert_eq!(at(0x8000), None, "in another buffer");
        // A slot that would run past the end of the address space is refused, not an overflow.
        assert_eq!(slot_offset(0, u64::MAX, 0x100, u64::MAX - 0xFF), None);
        // A buffer that ends exactly at its last slot accepts it, and nothing past it.
        assert_eq!(slot_offset(0x1000, 0x400, 0x100, 0x1300), Some(0x300));
        assert_eq!(slot_offset(0x1000, 0x400, 0x100, 0x1400), None);
    }

    #[test]
    fn check_fits_counts_the_sequence_state() {
        // A 65,536-token session needs 1.3 GiB of VRAM on top of the weights: too big a --ctx fails before loading.
        let n = Need {
            vram: 2000 << 20,
            host: 19_000 << 20,
            session: 0,
            prefill: 0,
            prefill_host: 0,
        }
        .with_session(&Dims::QWEN36, 65_536, None);
        assert!(check_fits(&n, 4000 << 20, 40_000 << 20).is_ok());
        let e = check_fits(&n, 3000 << 20, 40_000 << 20)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("sequence state") && e.contains("3923 MiB") && e.contains("3000 MiB"),
            "{e}"
        );
        assert!(!e.contains("prefill"), "{e}");
    }
}
