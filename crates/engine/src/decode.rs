//! One token's decode: every kernel recorded into one command buffer, one submit, the logits read back.
use crate::{
    cpu_miss::{seq_of, step_table, CpuPath, MAX_SPINS},
    session::{is_validation, LayerState, TapArenas},
    timing::{CpuPhase, StepKind, TAG_EMBED, TAG_OUTPUT},
    weights::GpuMixer,
    Engine, Error, Session,
};
use kernels::{
    dispatch::{mbox, read_status, STATUS_BYTES, STATUS_CLEAR},
    twin::attention::rope_table,
};
use qwen36::{
    dims::Dims,
    taps::{Key, Tap, TapName},
};
use std::time::Instant;

/// The activations one step passes between kernels, in VRAM, reused by every layer.
pub(crate) struct Scratch {
    pub x: vk::Slice,
    pub h: vk::Slice,
    pub mix: vk::Slice,
    pub resid: vk::Slice,
    pub f: vk::Slice,
    pub qkv: vk::Slice,
    pub z: vk::Slice,
    pub braw: vk::Slice,
    pub araw: vk::Slice,
    pub beta: vk::Slice,
    pub g: vk::Slice,
    pub conv: vk::Slice,
    pub o: vk::Slice,
    pub nrm: vk::Slice,
    pub qfull: vk::Slice,
    pub q: vk::Slice,
    pub gate: vk::Slice,
    pub k: vk::Slice,
    pub v: vk::Slice,
    pub pre: vk::Slice,
    pub gated: vk::Slice,
    pub rlog: vk::Slice,
    pub probs: vk::Slice,
    pub ids: vk::Slice,
    pub w: vk::Slice,
    pub act: vk::Slice,
    pub ye: vk::Slice,
    pub sg_g: vk::Slice,
    pub sg_u: vk::Slice,
    pub sact: vk::Slice,
    pub sh: vk::Slice,
    pub sg: vk::Slice,
    pub moe_out: vk::Slice,
    pub shexp_out: vk::Slice,
    pub ffn_out: vk::Slice,
    pub fnorm: vk::Slice,
    pub logits: vk::Slice,
}

impl Scratch {
    /// Each buffer's f32 count, in field order.
    fn counts(d: &Dims) -> [usize; 37] {
        let (e, ad) = (d.n_embd, d.n_head * d.head_dim);
        let (kv, k, ff) = (d.n_head_kv * d.head_dim, d.n_expert_used, d.n_ff_shexp);
        [
            e,              // x
            e,              // h
            e,              // mix
            e,              // resid
            e,              // f
            d.conv_dim(),   // qkv
            d.ssm_inner,    // z
            d.ssm_dt_rank,  // braw
            d.ssm_dt_rank,  // araw
            d.ssm_dt_rank,  // beta
            d.ssm_dt_rank,  // g
            d.conv_dim(),   // conv
            d.ssm_inner,    // o
            d.ssm_inner,    // nrm
            2 * ad,         // qfull
            ad,             // q
            ad,             // gate
            kv,             // k
            kv,             // v
            ad,             // pre
            ad,             // gated
            d.n_expert,     // rlog
            d.n_expert,     // probs
            k,              // ids
            k,              // w
            k * d.n_ff_exp, // act
            k * e,          // ye
            ff,             // sg_g
            ff,             // sg_u
            ff,             // sact
            e,              // sh
            1,              // sg
            e,              // moe_out
            e,              // shexp_out
            e,              // ffn_out
            e,              // fnorm
            d.n_vocab,      // logits
        ]
    }

    /// Each buffer's bytes, in the order `new` allocates them.
    pub fn bytes(d: &Dims) -> Vec<u64> {
        Scratch::counts(d).iter().map(|&n| 4 * n as u64).collect()
    }

    pub fn new(dev: &vk::Device, arena: &mut vk::Arena, d: &Dims) -> Result<Scratch, Error> {
        let mut s = Vec::with_capacity(37);
        for b in Scratch::bytes(d) {
            s.push(arena.alloc(dev, b)?);
        }
        let [x, h, mix, resid, f, qkv, z, braw, araw, beta, g, conv, o, nrm, qfull, q, gate, k, v, pre, gated, rlog, probs, ids, w, act, ye, sg_g, sg_u, sact, sh, sg, moe_out, shexp_out, ffn_out, fnorm, logits] =
            s.try_into().expect("one slice per count");
        Ok(Scratch {
            x,
            h,
            mix,
            resid,
            f,
            qkv,
            z,
            braw,
            araw,
            beta,
            g,
            conv,
            o,
            nrm,
            qfull,
            q,
            gate,
            k,
            v,
            pre,
            gated,
            rlog,
            probs,
            ids,
            w,
            act,
            ye,
            sg_g,
            sg_u,
            sact,
            sh,
            sg,
            moe_out,
            shexp_out,
            ffn_out,
            fnorm,
            logits,
        })
    }
}

/// One step's taps: the sink, the host buffers, and where each tap landed.
struct TapPlan<'t> {
    sink: &'t mut dyn Tap,
    n_expert: usize,
    rb: &'t vk::Buffer,
    force: &'t vk::Buffer,
    used: u64,
    list: Vec<(TapName, u64, usize)>,
}

/// One step being recorded.
struct Step<'d, 't> {
    r: vk::Recorder<'d>,
    taps: Option<TapPlan<'t>>,
    pos: u32,
}

impl Step<'_, '_> {
    /// Separates kernels that don't end in a tap.
    fn barrier(&mut self) {
        self.r.barrier();
    }

    /// After the kernel that wrote `region`, for tap `name`. In tap mode: copy `region` out, and if the sink forces
    /// this tap, copy the forced value over `region` before anything reads it (teacher forcing).
    fn tap(&mut self, name: TapName, region: vk::Slice) -> Result<(), Error> {
        self.r.barrier();
        let Some(t) = self.taps.as_mut() else {
            return Ok(());
        };
        let n = (region.size / 4) as usize;
        let off = t.used;
        t.used += region.size;
        self.r.copy(region, t.rb.slice(off, region.size));
        // The copy out reads `region` before anything overwrites it: the forced value, or the next kernel, which may
        // update `region` in place (`l2norm` after `gdn.conv`).
        self.r.barrier();
        if let Some(f) = t.sink.forced(name, self.pos) {
            if f.len() != n {
                return Err(Error::BadForce {
                    tap: name.to_string(),
                    want: n,
                    got: f.len(),
                });
            }
            if let TapName::Layer(_, Key::TopK) = name {
                let bad = f
                    .iter()
                    .find(|&&v| !(v >= 0.0 && v < t.n_expert as f32 && v.fract() == 0.0));
                if let Some(&value) = bad {
                    return Err(Error::BadForcedExpert {
                        tap: name.to_string(),
                        value,
                        n_expert: t.n_expert,
                    });
                }
            }
            t.force.write_f32(off, f);
            self.r.copy(t.force.slice(off, region.size), region);
            self.r.barrier();
        }
        t.list.push((name, off, n));
        Ok(())
    }
}

impl Session {
    /// Decodes `token` at the next position and returns all its logits: one command buffer, one submit.
    /// After any error other than `TokenOutOfRange` or `ContextFull` the session's state is undefined; start a new
    /// session.
    pub fn decode(&mut self, e: &Engine, token: u32) -> Result<Vec<f32>, Error> {
        self.step(e, token, None)
    }

    /// `decode`, reporting every tap to `tap`, and continuing from any value `tap.forced()` supplies.
    /// Tapping only adds copies: the logits are bit-identical to `decode`'s.
    ///
    /// When the sink forces the `logits` tap, the returned logits are the forced values, and so is what the NaN check
    /// sees; the GPU's own logits are what the sink was given. A comparison of the taps (`artemis gpu-taps`) still
    /// fails on a NaN the GPU produced.
    ///
    /// The tap arenas are allocated before the token and context checks, so a failed allocation, unlike a refused
    /// token or a full context, marks the session unusable.
    pub fn decode_tapped(
        &mut self,
        e: &Engine,
        token: u32,
        tap: &mut dyn Tap,
    ) -> Result<Vec<f32>, Error> {
        self.step(e, token, Some(tap))
    }

    /// A step with the session's write rule around it: on success the KV record takes the token at the position it
    /// was written; on an error past validation (`is_validation`) the record is cut back to that position and the
    /// session is unusable until `reset`; a refused call changes nothing.
    fn step(
        &mut self,
        e: &Engine,
        token: u32,
        sink: Option<&mut dyn Tap>,
    ) -> Result<Vec<f32>, Error> {
        assert!(!self.pending(), "a decode with a verify step pending");
        let start = self.pos;
        let out = self.step_body(e, token, sink);
        match &out {
            Ok(_) => {
                self.kv.wrote(start, &[token]);
                if let Some(m) = &mut self.mtp {
                    m.on_decode(start);
                }
            }
            Err(err) if !is_validation(err) => {
                self.kv.failed(start);
                if let Some(m) = &mut self.mtp {
                    m.on_failed(start);
                }
                self.usable = false;
            }
            Err(_) => {}
        }
        out
    }

    fn step_body(
        &mut self,
        e: &Engine,
        token: u32,
        sink: Option<&mut dyn Tap>,
    ) -> Result<Vec<f32>, Error> {
        if self.engine != e.id {
            return Err(Error::WrongEngine);
        }
        // The tap buffers leave the session for the step, so the plan can borrow them while the step records.
        let arenas = match sink {
            Some(_) => {
                self.tap_arenas(e)?;
                self.taps.take()
            }
            None => None,
        };
        let out = self.run_step(e, token, sink, arenas.as_ref());
        if arenas.is_some() {
            self.taps = arenas;
        }
        out
    }

    fn run_step(
        &mut self,
        e: &Engine,
        token: u32,
        sink: Option<&mut dyn Tap>,
        arenas: Option<&TapArenas>,
    ) -> Result<Vec<f32>, Error> {
        let d = &e.dims;
        if token as usize >= d.n_vocab {
            return Err(Error::TokenOutOfRange {
                token,
                n_vocab: d.n_vocab,
            });
        }
        if self.pos >= self.ctx {
            return Err(Error::ContextFull { ctx: self.ctx });
        }
        // Validation has passed: a pending snapshot copy that still reads this position finishes first.
        self.wait_copy(e, self.pos)?;
        // Swaps move on between steps. A GPU failure here names the position, as a step's does.
        e.begin_timing(StepKind::Decode, self.pos, 1);
        let t0 = Instant::now();
        e.prepare_step(crate::PolicyStep::Decode, 1)
            .map_err(|err| match err {
                Error::Vk(source) => Error::Step {
                    pos: self.pos,
                    layer: None,
                    source,
                },
                other => other,
            })?;
        e.time(CpuPhase::Prepare, t0);
        let pos = self.pos;
        let t0 = Instant::now();
        e.rope
            .write_f32(0, &rope_table(pos as u32, d.n_rot, d.rope_base));
        e.status.write(0, &STATUS_CLEAR);
        let taps = match (sink, arenas) {
            (Some(sink), Some(a)) => Some(TapPlan {
                sink,
                n_expert: d.n_expert,
                rb: &a.rb,
                force: &a.force,
                used: 0,
                list: Vec::new(),
            }),
            _ => None,
        };
        let mut s = Step {
            r: e.record().map_err(|source| Error::Step {
                pos,
                layer: None,
                source,
            })?,
            taps,
            pos: pos as u32,
        };
        // The CPU miss path, on exactly while the share runs it (`path_on`): the step records the publish,
        // the wait and the fallback pair, and the coordinator answers during the submit. Off, the step is today's.
        let cpu = e.cpu.borrow();
        let cpu = cpu.as_ref();
        if let Some(c) = cpu {
            c.before_step(&e.dev, pos).map_err(|source| Error::Step {
                pos,
                layer: None,
                source,
            })?;
        }
        // The step's slot-table changes, applied first. A step that fails before its submit applies none,
        // so they wait for the next step.
        let staged = e.take_staged();
        if let Err(err) = self.record(e, &mut s, token, pos, staged.len(), cpu) {
            e.restage(staged);
            return Err(err);
        }
        // The step's final hidden row goes into the MTP's ring: one 8 KiB copy, only with `Options::mtp`.
        if let Some(g) = &self.mtp_gpu {
            g.copy_to_ring(&mut s.r, e.scratch.fnorm, pos, 1, d.n_embd);
        }
        s.r.copy(e.scratch.logits, e.logits.whole());
        s.r.host_barrier();
        e.time(CpuPhase::Record, t0);
        let Step { r, taps, .. } = s;
        // With the path on: the token announced to the coordinator, and the engine thread on worker 0's SMT sibling
        // at HIGHEST for the submit only (the guard restores it).
        let guard = cpu.map(|c| {
            c.announce(pos as u32);
            c.for_submit()
        });
        let submitted = e.submit_step(r);
        drop(guard);
        if let Some(c) = cpu {
            // This token's timeouts and the CPU's experts into the stats, then the status words zeroed: the wait's
            // no-spin is per token, so the next token's first wait must find them 0. After a failed submit the GPU may
            // still be running the step, so the words are left and the next step's `before_step` resets the mailbox.
            let (timeouts, computed) = c.after_step(submitted.is_ok());
            e.count_cpu(timeouts, computed);
        }
        if let Err(source) = submitted {
            // The planner already counts these changes as made, but a failed submit applied none of them. Each is an
            // absolute write of one entry, so the next step can apply it late; `restage` keeps any newer change
            // staged for the same entry.
            e.restage(staged);
            return Err(Error::Step {
                pos,
                layer: None,
                source,
            });
        }
        // Report the taps first: a failed step's taps are what explains it.
        if let Some(t) = taps {
            for &(name, off, n) in &t.list {
                let data = t.rb.read_f32(off, n);
                t.sink.tap(name, pos as u32, &data);
            }
        }
        let t0 = Instant::now();
        let mut st = [0u8; STATUS_BYTES];
        e.status.read(0, &mut st);
        let st = read_status(&st);
        e.time(CpuPhase::Read, t0);
        if let Some((layer, expert)) = st.bad {
            return Err(Error::BadSlotEntry { pos, layer, expert });
        }
        if let Some((layer, expert)) = st.cpu {
            return Err(Error::ExpertOnCpu { pos, layer, expert });
        }
        // Counted only for a step whose status checked out: its routing log is then sound.
        let t0 = Instant::now();
        let ids = e.routes.read_f32(0, d.n_layer * d.n_expert_used);
        e.count_routes(&ids, 1, None);
        let logits = e.logits.read_f32(0, d.n_vocab);
        let bad = crate::sample::first_non_finite(&logits);
        e.time(CpuPhase::Read, t0);
        e.observe_routes(&ids, 1);
        if let Some((index, value)) = bad {
            return Err(Error::NonFinite { pos, index, value });
        }
        self.pos += 1;
        self.tokens.push(token);
        Ok(logits)
    }

    /// Every kernel of one token, in graph order (semantics reference §2 to §5), with each tap after the kernel that
    /// computes it. The `updates` staged slot-table changes are applied first, before anything reads the table. With
    /// `cpu` (the CPU miss path on), the step reads the overlay table and each layer's experts go through the publish,
    /// the serve-1 pair, the wait and the serve-2 fallback pair; without it, today's dispatches.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        e: &Engine,
        s: &mut Step,
        token: u32,
        pos: usize,
        updates: usize,
        cpu: Option<&CpuPath>,
    ) -> Result<(), Error> {
        let (d, k, sc) = (&e.dims, &e.k, &e.scratch);
        s.r.tag(TAG_EMBED);
        if updates > 0 {
            k.apply_slot_updates(
                &mut s.r,
                e.table.addr,
                e.updates.addr(),
                updates,
                (d.n_layer + 1) * d.n_expert,
            );
            s.barrier();
        }
        if let Some(b) = &e.base {
            // The base time: the expert kernels read a copy of the table with every HOST entry
            // pointed at a resident expert, made from the mirror, which already holds this step's changes.
            b.write(&e.mirror.borrow(), d.n_layer, d.n_expert);
            s.r.copy(b.host.whole(), b.vram);
            s.barrier();
        }
        if let Some(c) = cpu {
            // The CPU path's overlay: the mirror, which already holds this step's changes, with the HOST
            // entries the share assigns (at their expert's own host copy) turned WHERE_CPU, at the share set since the
            // last token. Its own table, so prefill and the real table never
            // see it.
            let t = step_table(
                &e.mirror.borrow(),
                d.n_layer,
                d.n_expert,
                &c.host,
                c.share_cell(),
            );
            c.table.host.write(0, &t);
            s.r.copy(c.table.host.whole(), c.table.vram);
            s.barrier();
        }
        let table = e.expert_table();
        let eps = d.rms_eps;
        let (kd, dk, dv, nv) = (d.key_dim(), d.ssm_d_state, d.v_head_dim(), d.ssm_dt_rank);
        let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
        let status = e.status.addr();
        k.embed_row(
            &mut s.r,
            e.host.embd_row(token),
            e.host.embd_ty,
            d.n_embd,
            sc.x.addr,
        );
        s.tap(TapName::Embd, sc.x)?;
        for (l, gl) in e.w.layers.iter().enumerate() {
            s.r.tag(l as u32);
            let t = |key| TapName::Layer(l as u16, key);
            k.rmsnorm(&mut s.r, sc.x.addr, gl.attn_norm, sc.h.addr, d.n_embd, eps);
            s.tap(t(Key::AttnNorm), sc.h)?;
            match (&gl.mixer, &self.layers[l]) {
                (
                    GpuMixer::DeltaNet {
                        qkv,
                        z,
                        beta,
                        alpha,
                        conv1d,
                        dt_bias,
                        a,
                        norm,
                        out,
                    },
                    LayerState::DeltaNet { conv, s: state },
                ) => {
                    k.matvec(&mut s.r, *qkv, sc.h.addr, sc.qkv.addr);
                    s.tap(t(Key::GdnQkv), sc.qkv)?;
                    k.matvec(&mut s.r, *z, sc.h.addr, sc.z.addr);
                    s.tap(t(Key::GdnZ), sc.z)?;
                    k.matvec(&mut s.r, *beta, sc.h.addr, sc.braw.addr);
                    k.matvec(&mut s.r, *alpha, sc.h.addr, sc.araw.addr);
                    s.barrier();
                    k.gdn_gates(
                        &mut s.r,
                        sc.braw.addr,
                        sc.araw.addr,
                        *dt_bias,
                        *a,
                        sc.beta.addr,
                        sc.g.addr,
                        nv,
                    );
                    s.tap(t(Key::GdnBeta), sc.beta)?;
                    s.tap(t(Key::GdnG), sc.g)?;
                    k.conv_step(
                        &mut s.r,
                        sc.qkv.addr,
                        *conv1d,
                        conv.addr,
                        sc.conv.addr,
                        d.conv_dim(),
                        d.ssm_conv,
                    );
                    s.tap(t(Key::GdnConv), sc.conv)?;
                    k.l2norm(&mut s.r, sc.conv.addr, 2 * d.ssm_n_group, dk, eps);
                    let (cq, ck, cv) = (
                        sc.conv.f32s(0, kd),
                        sc.conv.f32s(kd, kd),
                        sc.conv.f32s(2 * kd, d.ssm_inner),
                    );
                    s.tap(t(Key::GdnQ), cq)?;
                    s.tap(t(Key::GdnK), ck)?;
                    s.tap(t(Key::GdnV), cv)?;
                    k.gdn_step(
                        &mut s.r,
                        cq.addr,
                        ck.addr,
                        cv.addr,
                        sc.g.addr,
                        sc.beta.addr,
                        state.addr,
                        sc.o.addr,
                        nv,
                        dk,
                        dv,
                        d.ssm_n_group,
                    );
                    s.tap(t(Key::GdnO), sc.o)?;
                    k.gated_rmsnorm(
                        &mut s.r,
                        sc.o.addr,
                        sc.z.addr,
                        *norm,
                        sc.nrm.addr,
                        nv,
                        dv,
                        eps,
                    );
                    s.tap(t(Key::GdnNormed), sc.nrm)?;
                    k.matvec(&mut s.r, *out, sc.nrm.addr, sc.mix.addr);
                }
                (
                    GpuMixer::Attention {
                        q,
                        k: kw,
                        v,
                        out,
                        q_norm,
                        k_norm,
                    },
                    LayerState::Attention { kc, vc },
                ) => {
                    k.matvec(&mut s.r, *q, sc.h.addr, sc.qfull.addr);
                    k.matvec(&mut s.r, *kw, sc.h.addr, sc.k.addr);
                    k.matvec(&mut s.r, *v, sc.h.addr, sc.v.addr);
                    s.barrier();
                    k.qk_norm_rope(
                        &mut s.r,
                        sc.qfull.addr,
                        sc.k.addr,
                        sc.q.addr,
                        sc.gate.addr,
                        *q_norm,
                        *k_norm,
                        e.rope.addr(),
                        nh,
                        nkv,
                        hd,
                        d.n_rot,
                        eps,
                    );
                    s.tap(t(Key::AttnQ), sc.q)?;
                    s.tap(t(Key::AttnK), sc.k)?;
                    s.tap(t(Key::AttnV), sc.v)?;
                    k.kv_write(
                        &mut s.r,
                        sc.k.addr,
                        sc.v.addr,
                        kc.addr,
                        vc.addr,
                        pos,
                        nkv * hd,
                    );
                    s.barrier();
                    k.attn_decode(
                        &mut s.r,
                        sc.q.addr,
                        kc.addr,
                        vc.addr,
                        self.part.addr,
                        pos + 1,
                        nh,
                        nkv,
                        hd,
                    );
                    s.barrier();
                    k.attn_combine(&mut s.r, self.part.addr, sc.pre.addr, pos + 1, nh, hd);
                    s.tap(t(Key::AttnPregate), sc.pre)?;
                    k.sigmoid_mul(&mut s.r, sc.pre.addr, sc.gate.addr, sc.gated.addr, nh * hd);
                    s.tap(t(Key::AttnGated), sc.gated)?;
                    k.matvec(&mut s.r, *out, sc.gated.addr, sc.mix.addr);
                }
                _ => unreachable!("layer {l}'s weights and state are of the same kind"),
            }
            s.tap(t(Key::MixerOut), sc.mix)?;
            k.add(&mut s.r, sc.x.addr, sc.mix.addr, sc.resid.addr, d.n_embd);
            s.tap(t(Key::Resid), sc.resid)?;
            k.rmsnorm(
                &mut s.r,
                sc.resid.addr,
                gl.ffn_norm,
                sc.f.addr,
                d.n_embd,
                eps,
            );
            s.tap(t(Key::FfnNorm), sc.f)?;
            k.matvec(&mut s.r, gl.router, sc.f.addr, sc.rlog.addr);
            s.tap(t(Key::RouterLogits), sc.rlog)?;
            k.router_topk(
                &mut s.r,
                sc.rlog.addr,
                sc.probs.addr,
                sc.ids.addr,
                sc.w.addr,
                d.n_expert,
                d.n_expert_used,
            );
            s.tap(t(Key::RouterProbs), sc.probs)?;
            s.tap(t(Key::TopK), sc.ids)?;
            s.tap(t(Key::TopKW), sc.w)?;
            // The routing log: this layer's routed ids, read back for the cache's statistics.
            let ids_bytes = 4 * d.n_expert_used as u64;
            s.r.copy(sc.ids, e.routes.slice(l as u64 * ids_bytes, ids_bytes));
            // Share 0: serve 0 (a CPU entry is refused). The path on: the layer's x, ids and mask to the coordinator,
            // then serve 1 (a CPU entry writes +0, its y comes from the wait or the fallback pair).
            let (serve, seq) = match cpu {
                Some(c) => {
                    let seq = seq_of(pos as u32, l);
                    k.miss_publish(
                        &mut s.r,
                        sc.f.addr,
                        sc.ids.addr,
                        table,
                        c.mbox_addr(),
                        l,
                        d.n_expert,
                        seq,
                        pos as u32,
                        d.n_embd,
                        d.n_expert_used,
                    );
                    s.barrier();
                    (1, seq)
                }
                None => (0, 0),
            };
            k.experts_gate_up(
                &mut s.r,
                table,
                l,
                d.n_expert,
                sc.ids.addr,
                sc.f.addr,
                sc.act.addr,
                status,
                &gl.slot,
                d.n_expert_used,
                serve,
                0,
                0,
            );
            s.tap(t(Key::MoeSwiglu), sc.act)?;
            k.experts_down(
                &mut s.r,
                table,
                l,
                d.n_expert,
                sc.ids.addr,
                sc.act.addr,
                sc.ye.addr,
                status,
                &gl.slot,
                d.n_expert_used,
                serve,
                0,
                0,
            );
            k.matvec(&mut s.r, gl.sh_gate, sc.f.addr, sc.sg_g.addr);
            k.matvec(&mut s.r, gl.sh_up, sc.f.addr, sc.sg_u.addr);
            s.barrier();
            k.swiglu(
                &mut s.r,
                sc.sg_g.addr,
                sc.sg_u.addr,
                sc.sact.addr,
                d.n_ff_shexp,
            );
            s.tap(t(Key::ShexpSwiglu), sc.sact)?;
            k.matvec(&mut s.r, gl.sh_down, sc.sact.addr, sc.sh.addr);
            k.matvec(&mut s.r, gl.sh_gate_inp, sc.f.addr, sc.sg.addr);
            s.barrier();
            if let Some(c) = cpu {
                // The shared expert above is the work that hides the wait. Then the CPU's y into ye (only on a
                // successful wait), and the serve-2 pair: each CPU expert whose wait did not succeed this token,
                // computed from its host address (a late answer is wasted, never trusted). The pair has no tap
                // between them, so its own barrier orders down after gate_up's writes to act.
                let mb = c.mbox_addr();
                k.miss_wait(
                    &mut s.r,
                    mb,
                    sc.ye.addr,
                    mb + mbox::STATUS_BYTE as u64,
                    l,
                    seq,
                    d.n_embd,
                    MAX_SPINS,
                    c.max_ticks,
                );
                s.barrier();
                k.experts_gate_up(
                    &mut s.r,
                    table,
                    l,
                    d.n_expert,
                    sc.ids.addr,
                    sc.f.addr,
                    sc.act.addr,
                    status,
                    &gl.slot,
                    d.n_expert_used,
                    2,
                    mb,
                    seq,
                );
                s.barrier();
                k.experts_down(
                    &mut s.r,
                    table,
                    l,
                    d.n_expert,
                    sc.ids.addr,
                    sc.act.addr,
                    sc.ye.addr,
                    status,
                    &gl.slot,
                    d.n_expert_used,
                    2,
                    mb,
                    seq,
                );
                s.barrier();
            }
            // ye[j] is routed index j's output wherever it came from (VRAM, HOST, the CPU or its fallback).
            k.moe_combine(
                &mut s.r,
                sc.ye.addr,
                sc.w.addr,
                sc.sh.addr,
                sc.sg.addr,
                sc.moe_out.addr,
                sc.shexp_out.addr,
                d.n_embd,
                d.n_expert_used,
            );
            s.tap(t(Key::MoeOut), sc.moe_out)?;
            s.tap(t(Key::ShexpOut), sc.shexp_out)?;
            k.add(
                &mut s.r,
                sc.moe_out.addr,
                sc.shexp_out.addr,
                sc.ffn_out.addr,
                d.n_embd,
            );
            s.tap(t(Key::FfnOut), sc.ffn_out)?;
            k.add(
                &mut s.r,
                sc.resid.addr,
                sc.ffn_out.addr,
                sc.x.addr,
                d.n_embd,
            );
            s.tap(t(Key::Out), sc.x)?;
        }
        s.r.tag(TAG_OUTPUT);
        k.rmsnorm(
            &mut s.r,
            sc.x.addr,
            e.w.output_norm,
            sc.fnorm.addr,
            d.n_embd,
            eps,
        );
        s.tap(TapName::FinalNorm, sc.fnorm)?;
        k.matvec(&mut s.r, e.w.output, sc.fnorm.addr, sc.logits.addr);
        s.tap(TapName::Logits, sc.logits)?;
        Ok(())
    }
}
