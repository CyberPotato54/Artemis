//! The verify step: the layout of its buffers in prefill's idle work area, `Session::verify` and
//! `commit`, the routes of the rows a step kept, and the options check that keeps speculation off the modes it can't
//! run beside.
//!
//! A verify step is a decode-shaped pass over `t` tokens (`t` at most `kernels::tokens::MAX_TOKENS`). Prefill's buffers
//! are idle during decode, so the step carves its own activations, the 30 DeltaNet layers' saved inputs and their replay
//! scratch, the attention partials and the expert lists from `Prefill::work` (`VerifyLayout`). The logit rows go in
//! `Prefill::logits`/`out`, the routes in `Prefill::routes`, the embedding rows in `Prefill::rows` and the RoPE tables
//! in `Prefill::rope`. At the production chunk (2,048) prefill's own need is larger than the layout's, so speculation
//! adds no VRAM (`prefill::prefill_sizes`).
//!
//! The pass reads DeltaNet state set A and never writes it: `conv_seq` and `gdn_seq` run with `store = false`, and
//! each layer saves the inputs they ran from. `commit(kept)` rewinds to the kept position by position (attention keeps
//! no other state) and replays the kept tokens through the saved inputs, in place, in one command buffer.
use crate::{
    session::LayerState,
    spec,
    timing::{CpuPhase, StepKind, TAG_EMBED, TAG_OUTPUT},
    weights::GpuMixer,
    Engine, Error, Options, PolicyStep, Session,
};
use kernels::{
    dispatch::{attn_splits, read_status, STATUS_BYTES, STATUS_CLEAR},
    tokens::{self, MAX_TOKENS},
    twin::attention::rope_table,
};
use qwen36::dims::Dims;
use std::time::Instant;

/// Where a DeltaNet layer's verify step keeps its saved inputs and its replay scratch, as byte offsets into
/// `Prefill::work`. Each layer has its own, so the replay's 2 x 30 independent dispatches need no barrier between
/// layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeltaNetRegions {
    /// The raw conv input, `[t][conv_dim]`: the qkv projection's output.
    pub raw: u64,
    /// The post-conv (then `l2norm`'d) q|k|v, `[t][conv_dim]`.
    pub post: u64,
    /// The gates, `[t][dt_rank]` each.
    pub g: u64,
    pub beta: u64,
    /// The replay's `conv_seq` output, `[t][conv_dim]`.
    pub replay_y: u64,
    /// The replay's `gdn_seq` output, `[t][ssm_inner]`.
    pub replay_o: u64,
}

/// A verify step of `tokens` tokens at a context of `ctx` positions: every buffer it works in, as 16-byte-aligned byte
/// offsets into `Prefill::work` (the matvec and expert dispatches assert the alignment), and the bytes they take.
/// Decode's `Scratch`, with a token dimension: each name is the `Scratch` field it widens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifyLayout {
    pub tokens: usize,
    /// The eight activations, `[t][n_embd]` each.
    pub x: u64,
    pub h: u64,
    pub mix: u64,
    pub resid: u64,
    pub f: u64,
    pub moe_out: u64,
    pub shexp_out: u64,
    pub ffn_out: u64,
    /// Attention: qfull `[t][2 * n_head * head_dim]`, q, gate, pre and gated `[t][n_head * head_dim]`, k and v
    /// `[t][n_head_kv * head_dim]`.
    pub qfull: u64,
    pub q: u64,
    pub gate: u64,
    pub k: u64,
    pub v: u64,
    pub pre: u64,
    pub gated: u64,
    /// DeltaNet's per-layer temporaries (the layers never live at once): z, o and nrm `[t][ssm_inner]`, the projections
    /// of beta and alpha before the gates, `[t][dt_rank]` each.
    pub z: u64,
    pub o: u64,
    pub nrm: u64,
    pub braw: u64,
    pub araw: u64,
    /// The MoE block's: rlog and probs `[t][n_expert]`, ids and w `[t][n_expert_used]`, act `[t][used][n_ff_exp]`, ye
    /// `[t][used][n_embd]`, and the shared expert's rows: sg_g, sg_u and sact `[t][n_ff_shexp]`, sh `[t][n_embd]`, sg
    /// `[t]`.
    pub rlog: u64,
    pub probs: u64,
    pub ids: u64,
    pub w: u64,
    pub act: u64,
    pub ye: u64,
    pub sg_g: u64,
    pub sg_u: u64,
    pub sact: u64,
    pub sh: u64,
    pub sg: u64,
    /// Per layer (`d.n_layer` entries): the list of the layer's distinct routed experts (`tokens::moe_unique_bytes`).
    pub unique: Vec<u64>,
    /// Per layer (`d.n_layer` entries): the DeltaNet layers' saved inputs and replay scratch; None for attention layers.
    pub deltanet: Vec<Option<DeltaNetRegions>>,
    /// The attention partials of `t` queries at `attn_splits(ctx)` splits each (`tokens::attn_tokens_part_bytes`).
    pub part: u64,
    /// The bytes all of it takes: the work area must hold at least this many.
    pub bytes: u64,
}

/// Hands out consecutive 16-byte-aligned regions.
struct Carve(u64);

impl Carve {
    fn take(&mut self, f32s: usize) -> u64 {
        self.bytes(4 * f32s as u64)
    }

    fn bytes(&mut self, n: u64) -> u64 {
        let at = self.0;
        self.0 += n.next_multiple_of(16);
        at
    }
}

impl VerifyLayout {
    /// The layout of a verify step of `tokens` tokens (1 to `tokens::MAX_TOKENS`) in a session of `ctx` positions, which
    /// sets the attention partials' split count.
    pub(crate) fn new(d: &Dims, ctx: usize, tokens: usize) -> VerifyLayout {
        assert!(
            (1..=MAX_TOKENS).contains(&tokens),
            "a verify of {tokens} tokens: 1 to {MAX_TOKENS}"
        );
        let t = tokens;
        let (e, ad) = (d.n_embd, d.n_head * d.head_dim);
        let (kv, used, ff) = (d.n_head_kv * d.head_dim, d.n_expert_used, d.n_ff_shexp);
        let (cd, inner, nv) = (d.conv_dim(), d.ssm_inner, d.ssm_dt_rank);
        let mut c = Carve(0);
        let x = c.take(t * e);
        let h = c.take(t * e);
        let mix = c.take(t * e);
        let resid = c.take(t * e);
        let f = c.take(t * e);
        let moe_out = c.take(t * e);
        let shexp_out = c.take(t * e);
        let ffn_out = c.take(t * e);
        let qfull = c.take(t * 2 * ad);
        let q = c.take(t * ad);
        let gate = c.take(t * ad);
        let k = c.take(t * kv);
        let v = c.take(t * kv);
        let pre = c.take(t * ad);
        let gated = c.take(t * ad);
        let z = c.take(t * inner);
        let o = c.take(t * inner);
        let nrm = c.take(t * inner);
        let braw = c.take(t * nv);
        let araw = c.take(t * nv);
        let rlog = c.take(t * d.n_expert);
        let probs = c.take(t * d.n_expert);
        let ids = c.take(t * used);
        let w = c.take(t * used);
        let act = c.take(t * used * d.n_ff_exp);
        let ye = c.take(t * used * e);
        let sg_g = c.take(t * ff);
        let sg_u = c.take(t * ff);
        let sact = c.take(t * ff);
        let sh = c.take(t * e);
        let sg = c.take(t);
        let unique = (0..d.n_layer)
            .map(|_| c.bytes(tokens::moe_unique_bytes(t, used)))
            .collect();
        let deltanet = (0..d.n_layer)
            .map(|l| {
                (!d.is_attention(l)).then(|| DeltaNetRegions {
                    raw: c.take(t * cd),
                    post: c.take(t * cd),
                    g: c.take(t * nv),
                    beta: c.take(t * nv),
                    replay_y: c.take(t * cd),
                    replay_o: c.take(t * inner),
                })
            })
            .collect();
        let part = c.bytes(tokens::attn_tokens_part_bytes(
            d.n_head,
            d.head_dim,
            attn_splits(ctx),
            t,
        ));
        VerifyLayout {
            tokens,
            x,
            h,
            mix,
            resid,
            f,
            moe_out,
            shexp_out,
            ffn_out,
            qfull,
            q,
            gate,
            k,
            v,
            pre,
            gated,
            z,
            o,
            nrm,
            braw,
            araw,
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
            unique,
            deltanet,
            part,
            bytes: c.0,
        }
    }
}

/// The routing log of the rows a step kept: `ids` is `[n_layer][t][n_used]` (the verify's log in `Prefill::routes`),
/// and the result is `[n_layer][kept][n_used]` with each layer's first `kept` rows, which is what
/// `CachePolicy::observe` takes for `kept` tokens. Rows past the stop were drafts nothing emitted.
pub(crate) fn gather_rows(
    ids: &[f32],
    n_layer: usize,
    t: usize,
    n_used: usize,
    kept: usize,
) -> Vec<f32> {
    assert!(
        (1..=t).contains(&kept),
        "kept {kept} rows of a verify of {t}"
    );
    assert_eq!(ids.len(), n_layer * t * n_used, "the routing log's size");
    let mut out = Vec::with_capacity(n_layer * kept * n_used);
    for layer in ids.chunks_exact(t * n_used) {
        out.extend_from_slice(&layer[..kept * n_used]);
    }
    out
}

/// Refuses speculation beside what it can't run with: the CPU miss path (its sequence numbers key on
/// the position, and a verify step's rows share one), and the base time (every HOST expert reads a resident one, so a
/// row's meaning is gone); and on a model whose verify step's expert pairs would not fit the kernels (`MAX_PAIRS`). Called by `Engine::open` before the model loads, after the share is checked as a share.
pub(crate) fn check_speculation(o: &Options, d: &Dims) -> Result<(), Error> {
    if o.mtp.is_some() && !o.speculation {
        return Err(Error::SpeculationRefused {
            why: "the MTP drafter needs speculation on",
        });
    }
    if !o.speculation {
        return Ok(());
    }
    // The expert kernels list a layer's (token, choice) pairs in one workgroup: a full verify must fit.
    if MAX_TOKENS * d.n_expert_used > tokens::MAX_PAIRS {
        return Err(Error::SpeculationRefused {
            why: "speculation needs the model's routed experts per token times 16 tokens to be at most 256",
        });
    }
    if crate::cpu_miss::path_on(o.cpu_share) {
        return Err(Error::SpeculationRefused {
            why: "speculation needs the CPU miss path off (--cpu-share 0)",
        });
    }
    if o.base_time {
        return Err(Error::SpeculationRefused {
            why: "speculation can't run with --base-time",
        });
    }
    Ok(())
}

/// A verify step that has run and not yet been committed: where it started, its routing log and the layout whose
/// saved DeltaNet inputs the replay reads. The session holds its every token meanwhile.
pub(crate) struct Pending {
    /// The position of the step's first token (`x`).
    pub start: usize,
    /// The step's routing log, `[n_layer][len][n_used]` (`Prefill::routes`'s layout), kept for `commit`.
    pub routes: Vec<f32>,
    pub layout: VerifyLayout,
}

impl Pending {
    /// The step's tokens.
    fn len(&self) -> usize {
        self.layout.tokens
    }
}

impl Session {
    /// Runs `tokens = [x, d1 .. dk]` (1 to `MAX_TOKENS` of them, and no more than the prefill chunk) at the next
    /// positions: every kernel of a decode step with a token dimension, in decode's order, one command buffer, one
    /// submit. On success the session holds all the tokens (`pos`, `tokens`, the KV record), as if all
    /// were kept, and is pending until `commit` keeps some of them; the logit rows are read with `row`. The DeltaNet
    /// and conv state are not touched.
    ///
    /// Fails without running anything, changing nothing, when the engine isn't this session's (`WrongEngine`), the
    /// tokens are none or too many (`VerifyShape`), one is out of the vocabulary (`TokenOutOfRange`), they pass the
    /// context (`PrefillTooLong`), or the work area wasn't sized for the session's context (`VerifyBuffers`). Any other
    /// error follows decode's write rule: the KV record is cut to the step's start and the session is unusable until
    /// `reset`. `assert!`s that no verify is pending, that the engine was opened with `Options::speculation` and that
    /// its CPU miss path and base time are off.
    ///
    /// The pending guard is per session, but the step's saved inputs, its rows and its routes live in the engine's
    /// prefill buffers: no other session may `prefill` or `verify` on this engine between this `verify` and its
    /// `commit`. The worker has one live session, so nothing enforces it.
    pub fn verify(&mut self, e: &Engine, toks: &[u32]) -> Result<(), Error> {
        assert!(!self.pending(), "a verify with another verify pending");
        let layout = self.check_verify(e, toks)?;
        let start = self.pos;
        match self.run_verify(e, toks, layout) {
            Ok(pending) => {
                self.pos += toks.len();
                self.tokens.extend_from_slice(toks);
                self.kv.wrote(start, toks);
                self.pending = Some(pending);
                Ok(())
            }
            Err(err) => {
                self.fail_step(start);
                Err(err)
            }
        }
    }

    /// Row `i` of the pending verify: the `n_vocab` logits after `tokens[i]`, read from the readback only now and
    /// checked for NaN alone, so a rejected row past the stop costs no copy and can't fail a step plain decode would
    /// have finished. A non-finite value fails the step: the KV record is cut to its start, the session is unusable and
    /// no longer pending, and `NonFinite` returns. `assert!`s that a verify is pending and `i` is one of its rows;
    /// another engine's is refused (`WrongEngine`).
    pub fn row(&mut self, e: &Engine, i: usize) -> Result<Vec<f32>, Error> {
        let Some(p) = &self.pending else {
            panic!("row {i} with no verify pending");
        };
        let (start, len) = (p.start, p.len());
        assert!(i < len, "row {i} of a verify of {len}");
        if self.engine != e.id {
            return Err(Error::WrongEngine);
        }
        let nv = e.dims.n_vocab;
        let t0 = Instant::now();
        let logits = e.prefill.out.read_f32(4 * (i * nv) as u64, nv);
        self.rows_read += 1;
        let bad = crate::sample::first_non_finite(&logits);
        e.time(CpuPhase::Read, t0);
        if let Some((index, value)) = bad {
            self.fail_step(start);
            return Err(Error::NonFinite {
                pos: start + i,
                index,
                value,
            });
        }
        Ok(logits)
    }

    /// Keeps `tokens[..kept]` of the pending verify (1 to its length) and settles it: rewinds to the kept
    /// position, replays the kept tokens through the DeltaNet layers' saved inputs (one command buffer, one submit),
    /// tells the cache policy the kept rows' routes and settles the allowance. This is the only call that clears
    /// `pending` on success. If the replay fails the step is failed as decode's write rule says (the KV record cut to the
    /// step's start, the session unusable, not pending) and the error returns. `assert!`s that a verify is pending; another engine's is refused
    /// (`WrongEngine`), the step still pending.
    pub fn commit(&mut self, e: &Engine, kept: usize) -> Result<(), Error> {
        let Some(p) = &self.pending else {
            panic!("commit with no verify pending");
        };
        let (start, len) = (p.start, p.len());
        assert!(
            (1..=len).contains(&kept),
            "kept {kept} of a verify of {len}"
        );
        // Another device would get this session's buffer addresses in its replay: refused, the step still pending.
        if self.engine != e.id {
            return Err(Error::WrongEngine);
        }
        let out = self.run_commit(e, kept);
        match &out {
            Ok(()) => {
                self.pending = None;
                if let Some(m) = &mut self.mtp {
                    m.on_commit(start, kept);
                }
            }
            Err(_) => self.fail_step(start),
        }
        out
    }

    /// A test hook: the next `commit`'s replay fails, as a device error in its submit would.
    #[doc(hidden)]
    pub fn fail_next_commit(&mut self) {
        self.fail_commit = true;
    }

    /// A test hook: how many rows `row` has copied out of the verify's logits.
    #[doc(hidden)]
    pub fn rows_read(&self) -> u64 {
        self.rows_read
    }

    /// Cuts the pending verify's tokens back to `to`, a position of the step (its start up to its end): `pos`, the
    /// token list and the KV record. The KV and the DeltaNet state are not touched: attention reads only positions below
    /// `pos`, and the state is the replay's (no GPU work).
    pub(crate) fn rewind(&mut self, to: usize) {
        let Some(p) = &self.pending else {
            panic!("rewind with no verify pending");
        };
        assert!(
            p.start <= to && to <= self.pos,
            "rewind to {to}, outside the step's {} to {}",
            p.start,
            self.pos
        );
        self.pos = to;
        self.tokens.truncate(to);
        self.kv.wrote(to, &[]);
        if let Some(m) = &mut self.mtp {
            m.rows.cut(to as u32);
        }
    }

    /// A step past validation failed: decode's write rule. The tokens and the KV record go back to the step's
    /// `start`, the session is unusable until `reset`, and nothing is pending.
    fn fail_step(&mut self, start: usize) {
        self.pending = None;
        self.pos = start;
        self.tokens.truncate(start);
        self.kv.failed(start);
        if let Some(m) = &mut self.mtp {
            m.on_failed(start);
        }
        self.usable = false;
    }

    /// The checks that refuse a verify before it touches anything, and the step's layout.
    fn check_verify(&self, e: &Engine, toks: &[u32]) -> Result<VerifyLayout, Error> {
        if self.engine != e.id {
            return Err(Error::WrongEngine);
        }
        assert!(
            e.speculation,
            "Session::verify on an engine opened without Options::speculation"
        );
        assert!(
            e.cpu.borrow().is_none() && e.base.is_none(),
            "Session::verify beside the CPU miss path or the base time (C12)"
        );
        let d = &e.dims;
        // The step's rows, routes, embeddings and RoPE tables live in prefill's buffers, sized by its chunk.
        let most = MAX_TOKENS.min(e.prefill.chunk);
        if toks.is_empty() || toks.len() > most {
            return Err(Error::VerifyShape {
                tokens: toks.len(),
                most,
            });
        }
        if let Some(&token) = toks.iter().find(|&&t| t as usize >= d.n_vocab) {
            return Err(Error::TokenOutOfRange {
                token,
                n_vocab: d.n_vocab,
            });
        }
        if self.pos + toks.len() > self.ctx {
            return Err(Error::PrefillTooLong {
                pos: self.pos,
                tokens: toks.len(),
                ctx: self.ctx,
            });
        }
        // The work area is sized for the engine's `ctx`; a longer session's attention partials may not fit.
        let layout = VerifyLayout::new(d, self.ctx, toks.len());
        if layout.bytes > e.prefill.work.size {
            return Err(Error::VerifyBuffers {
                ctx: self.ctx,
                need: layout.bytes,
                have: e.prefill.work.size,
            });
        }
        Ok(layout)
    }

    /// The step after validation: the host writes, the record, the submit and the checks of what it wrote.
    fn run_verify(
        &mut self,
        e: &Engine,
        toks: &[u32],
        layout: VerifyLayout,
    ) -> Result<Pending, Error> {
        let (d, p) = (&e.dims, &e.prefill);
        let (t, pos) = (toks.len(), self.pos);
        let step = |source| Error::Step {
            pos,
            layer: None,
            source,
        };
        // A pending snapshot copy that still reads these positions finishes first.
        self.wait_copy(e, pos)?;
        // The routing log is about to be overwritten: until the step finishes it holds no one step's routes.
        p.last.set(0);
        e.begin_timing(StepKind::Verify, pos, t);
        let t0 = Instant::now();
        e.prepare_step(PolicyStep::Verify, 1)
            .map_err(|err| match err {
                Error::Vk(source) => step(source),
                other => other,
            })?;
        e.time(CpuPhase::Prepare, t0);
        let t0 = Instant::now();
        let rows: Vec<u8> = toks
            .iter()
            .flat_map(|&tk| e.host.embd_row(tk).to_le_bytes())
            .collect();
        p.rows.write(0, &rows);
        let tables: Vec<f32> = (0..t)
            .flat_map(|i| rope_table((pos + i) as u32, d.n_rot, d.rope_base))
            .collect();
        p.rope.write_f32(0, &tables);
        e.status.write(0, &STATUS_CLEAR);
        let mut r = e.record().map_err(step)?;
        // The step's slot-table changes, applied first; a step that fails before its submit applies none.
        let staged = e.take_staged();
        self.record_verify(e, &mut r, &layout, pos, staged.len());
        e.time(CpuPhase::Record, t0);
        if let Err(source) = e.submit_step(r) {
            e.restage(staged);
            return Err(step(source));
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
        // Counted for a step whose status checked out: its routing log is then sound. Every row run counts, because
        // the reads happened; the policy hears only the kept rows' routes, at `commit`.
        let t0 = Instant::now();
        let ids = p.routes.read_f32(0, d.n_layer * t * d.n_expert_used);
        e.count_routes(&ids, t, None);
        e.time(CpuPhase::Read, t0);
        p.last.set(t);
        Ok(Pending {
            start: pos,
            routes: ids,
            layout,
        })
    }

    /// `commit` past its asserts: the rewind, the replay, the policy's view of the kept rows.
    fn run_commit(&mut self, e: &Engine, kept: usize) -> Result<(), Error> {
        let start = self.pending.as_ref().expect("a pending verify").start;
        self.rewind(start + kept);
        if std::mem::take(&mut self.fail_commit) {
            return Err(Error::Step {
                pos: start,
                layer: None,
                source: vk::Error::Timeout,
            });
        }
        self.replay(e, start, kept)?;
        let d = &e.dims;
        let p = self.pending.as_ref().expect("a pending verify");
        let ids = gather_rows(&p.routes, d.n_layer, p.len(), d.n_expert_used, kept);
        e.observe_routes(&ids, kept);
        e.settle_verify(kept);
        Ok(())
    }

    /// The kept tokens' DeltaNet state: per DeltaNet layer `conv_seq` and `gdn_seq` with `store = true` over the
    /// saved inputs of the pending verify, `kept` tokens from the state the verify read (A), in place. The layers are
    /// independent (each writes its own scratch): one command buffer, no barrier, one submit. The per-layer
    /// rollback is this `match`: an attention layer is rolled back by position and has nothing to do.
    fn replay(&self, e: &Engine, start: usize, kept: usize) -> Result<(), Error> {
        let (d, k) = (&e.dims, &e.k);
        let lay = &self.pending.as_ref().expect("a pending verify").layout;
        let step = |source| Error::Step {
            pos: start,
            layer: None,
            source,
        };
        let t0 = Instant::now();
        let mut r = e.record().map_err(step)?;
        let at = |off: u64| e.prefill.work.addr + off;
        let (kd, dk, dv, nv, ng, cd) = (
            d.key_dim(),
            d.ssm_d_state,
            d.v_head_dim(),
            d.ssm_dt_rank,
            d.ssm_n_group,
            d.conv_dim(),
        );
        for (l, gl) in e.w.layers.iter().enumerate() {
            match (&gl.mixer, &self.layers[l]) {
                (GpuMixer::DeltaNet { conv1d, .. }, LayerState::DeltaNet { conv, s }) => {
                    let rg = lay.deltanet[l].expect("a DeltaNet layer's saved inputs");
                    r.tag(l as u32);
                    k.conv_seq(
                        &mut r,
                        at(rg.raw),
                        *conv1d,
                        conv.addr,
                        at(rg.replay_y),
                        cd,
                        d.ssm_conv,
                        kept,
                        true,
                    );
                    k.gdn_seq(
                        &mut r,
                        at(rg.post),
                        at(rg.post) + 4 * kd as u64,
                        at(rg.post) + 8 * kd as u64,
                        at(rg.g),
                        at(rg.beta),
                        s.addr,
                        at(rg.replay_o),
                        nv,
                        dk,
                        dv,
                        ng,
                        kept,
                        cd,
                        true,
                    );
                }
                (GpuMixer::Attention { .. }, LayerState::Attention { .. }) => {}
                _ => unreachable!("layer {l}'s weights and state are of the same kind"),
            }
        }
        // The kept rows' final hidden rows go into the MTP's ring: the verify left them in
        // `Prefill::fnorm`, row `i` for position `start + i`.
        if let Some(g) = &self.mtp_gpu {
            g.copy_to_ring(&mut r, e.prefill.fnorm, start, kept, d.n_embd);
        }
        r.host_barrier();
        e.time(CpuPhase::Record, t0);
        e.trace_label.set(crate::timing::TRACE_REPLAY);
        e.submit_step(r).map_err(step)
    }

    /// Every kernel of a verify step, in `Session::record`'s order (decode's, with a token dimension) and with its
    /// barriers: a dependent kernel never starts before the one that feeds it ends. The token kernels do decode's
    /// arithmetic per token, bit for bit, except `gdn_seq`, which runs from the layer's raw conv input and saves
    /// what the replay needs. Every buffer is a region of the layout in `Prefill::work`.
    fn record_verify(
        &self,
        e: &Engine,
        r: &mut vk::Recorder,
        lay: &VerifyLayout,
        pos: usize,
        updates: usize,
    ) {
        let (d, k, p) = (&e.dims, &e.k, &e.prefill);
        let t = lay.tokens;
        let at = |off: u64| p.work.addr + off;
        r.tag(TAG_EMBED);
        if updates > 0 {
            k.apply_slot_updates(
                r,
                e.table.addr,
                e.updates.addr(),
                updates,
                (d.n_layer + 1) * d.n_expert,
            );
            r.barrier();
        }
        let table = e.expert_table();
        let eps = d.rms_eps;
        let (ne, nu, nff) = (d.n_expert, d.n_expert_used, d.n_ff_shexp);
        let (kd, dk, dv, nv, ng, cd) = (
            d.key_dim(),
            d.ssm_d_state,
            d.v_head_dim(),
            d.ssm_dt_rank,
            d.ssm_n_group,
            d.conv_dim(),
        );
        let (nh, nkv, hd, inner) = (d.n_head, d.n_head_kv, d.head_dim, d.ssm_inner);
        let (ad, kvw) = (nh * hd, nkv * hd);
        let status = e.status.addr();
        let max_split = attn_splits(self.ctx);
        k.embed_rows(r, p.rows.addr(), e.host.embd_ty, d.n_embd, at(lay.x), t);
        r.barrier();
        for (l, gl) in e.w.layers.iter().enumerate() {
            r.tag(l as u32);
            k.rmsnorm_tokens(r, at(lay.x), gl.attn_norm, at(lay.h), d.n_embd, eps, t);
            r.barrier();
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
                    let rg = lay.deltanet[l].expect("a DeltaNet layer's saved inputs");
                    // The qkv projection lands in the layer's saved raw slot: the replay's `conv_seq` input.
                    k.matvec_tokens(r, *qkv, at(lay.h), at(rg.raw), t, d.n_embd, cd);
                    k.matvec_tokens(r, *z, at(lay.h), at(lay.z), t, d.n_embd, inner);
                    k.matvec_tokens(r, *beta, at(lay.h), at(lay.braw), t, d.n_embd, nv);
                    k.matvec_tokens(r, *alpha, at(lay.h), at(lay.araw), t, d.n_embd, nv);
                    r.barrier();
                    k.gdn_gates_tokens(
                        r,
                        at(lay.braw),
                        at(lay.araw),
                        *dt_bias,
                        *a,
                        at(rg.beta),
                        at(rg.g),
                        nv,
                        t,
                    );
                    // State set A is read, never written: the post-conv slot is the saved copy the replay needs.
                    k.conv_seq(
                        r,
                        at(rg.raw),
                        *conv1d,
                        conv.addr,
                        at(rg.post),
                        cd,
                        d.ssm_conv,
                        t,
                        false,
                    );
                    r.barrier();
                    k.l2norm_tokens(r, at(rg.post), 2 * ng, dk, eps, t, cd);
                    r.barrier();
                    k.gdn_seq(
                        r,
                        at(rg.post),
                        at(rg.post) + 4 * kd as u64,
                        at(rg.post) + 8 * kd as u64,
                        at(rg.g),
                        at(rg.beta),
                        state.addr,
                        at(lay.o),
                        nv,
                        dk,
                        dv,
                        ng,
                        t,
                        cd,
                        false,
                    );
                    r.barrier();
                    k.gated_rmsnorm(r, at(lay.o), at(lay.z), *norm, at(lay.nrm), t * nv, dv, eps);
                    r.barrier();
                    k.matvec_tokens(r, *out, at(lay.nrm), at(lay.mix), t, inner, d.n_embd);
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
                    k.matvec_tokens(r, *q, at(lay.h), at(lay.qfull), t, d.n_embd, 2 * ad);
                    k.matvec_tokens(r, *kw, at(lay.h), at(lay.k), t, d.n_embd, kvw);
                    k.matvec_tokens(r, *v, at(lay.h), at(lay.v), t, d.n_embd, kvw);
                    r.barrier();
                    k.qk_norm_rope_tokens(
                        r,
                        at(lay.qfull),
                        at(lay.k),
                        at(lay.q),
                        at(lay.gate),
                        *q_norm,
                        *k_norm,
                        p.rope.addr(),
                        nh,
                        nkv,
                        hd,
                        d.n_rot,
                        eps,
                        t,
                    );
                    r.barrier();
                    k.kv_write_tokens(r, at(lay.k), at(lay.v), kc.addr, vc.addr, pos, kvw, t);
                    r.barrier();
                    // Query i sees `pos + 1 + i` positions, and its own K/V were just written.
                    k.attn_decode_tokens(
                        r,
                        at(lay.q),
                        kc.addr,
                        vc.addr,
                        at(lay.part),
                        pos + 1,
                        t,
                        nh,
                        nkv,
                        hd,
                        max_split,
                    );
                    r.barrier();
                    k.attn_combine_tokens(
                        r,
                        at(lay.part),
                        at(lay.pre),
                        pos + 1,
                        t,
                        nh,
                        hd,
                        max_split,
                    );
                    r.barrier();
                    k.sigmoid_mul(r, at(lay.pre), at(lay.gate), at(lay.gated), t * ad);
                    r.barrier();
                    k.matvec_tokens(r, *out, at(lay.gated), at(lay.mix), t, ad, d.n_embd);
                }
                _ => unreachable!("layer {l}'s weights and state are of the same kind"),
            }
            r.barrier();
            let n = t * d.n_embd;
            k.add(r, at(lay.x), at(lay.mix), at(lay.resid), n);
            r.barrier();
            k.rmsnorm_tokens(r, at(lay.resid), gl.ffn_norm, at(lay.f), d.n_embd, eps, t);
            r.barrier();
            k.matvec_tokens(r, gl.router, at(lay.f), at(lay.rlog), t, d.n_embd, ne);
            r.barrier();
            k.router_topk_tokens(
                r,
                at(lay.rlog),
                at(lay.probs),
                at(lay.ids),
                at(lay.w),
                ne,
                nu,
                t,
            );
            r.barrier();
            // The routing log: this layer's `[t][used]` ids, read back for the cache's statistics and `commit`.
            let ids_bytes = 4 * (t * nu) as u64;
            r.copy(
                p.work.sub(lay.ids, ids_bytes),
                p.routes.slice(l as u64 * ids_bytes, ids_bytes),
            );
            k.moe_unique(r, at(lay.ids), at(lay.unique[l]), t, nu);
            r.barrier();
            k.experts_gate_up_tokens(
                r,
                table,
                l,
                ne,
                at(lay.ids),
                at(lay.unique[l]),
                at(lay.f),
                at(lay.act),
                status,
                &gl.slot,
                nu,
                t,
            );
            r.barrier();
            k.experts_down_tokens(
                r,
                table,
                l,
                ne,
                at(lay.ids),
                at(lay.unique[l]),
                at(lay.act),
                at(lay.ye),
                status,
                &gl.slot,
                nu,
                t,
            );
            k.matvec_tokens(r, gl.sh_gate, at(lay.f), at(lay.sg_g), t, d.n_embd, nff);
            k.matvec_tokens(r, gl.sh_up, at(lay.f), at(lay.sg_u), t, d.n_embd, nff);
            r.barrier();
            k.swiglu(r, at(lay.sg_g), at(lay.sg_u), at(lay.sact), t * nff);
            r.barrier();
            k.matvec_tokens(r, gl.sh_down, at(lay.sact), at(lay.sh), t, nff, d.n_embd);
            k.matvec_tokens(r, gl.sh_gate_inp, at(lay.f), at(lay.sg), t, d.n_embd, 1);
            r.barrier();
            // Each token's routed outputs `ye[t][j]` weighted by its own `w[t][j]`, and its shared expert gated by its
            // own `sg[t]`: `moe_combine`'s order, one workgroup row a token.
            k.moe_combine_tokens(
                r,
                at(lay.ye),
                at(lay.w),
                at(lay.sh),
                at(lay.sg),
                at(lay.moe_out),
                at(lay.shexp_out),
                d.n_embd,
                nu,
                t,
            );
            r.barrier();
            k.add(r, at(lay.moe_out), at(lay.shexp_out), at(lay.ffn_out), n);
            r.barrier();
            k.add(r, at(lay.resid), at(lay.ffn_out), at(lay.x), n);
            r.barrier();
        }
        r.tag(TAG_OUTPUT);
        k.rmsnorm_tokens(
            r,
            at(lay.x),
            e.w.output_norm,
            p.fnorm.addr,
            d.n_embd,
            eps,
            t,
        );
        r.barrier();
        k.matvec_tokens(
            r,
            e.w.output,
            p.fnorm.addr,
            p.logits.addr,
            t,
            d.n_embd,
            d.n_vocab,
        );
        r.barrier();
        let bytes = 4 * (t * d.n_vocab) as u64;
        r.copy(p.logits.sub(0, bytes), p.out.slice(0, bytes));
        r.host_barrier();
    }
}

/// The step loop's view of a session on its engine (`spec::Verifier`): plain decode, and the verify pass with its
/// lazy rows and its commit.
pub struct SessionVerifier<'a> {
    pub e: &'a Engine,
    pub s: &'a mut Session,
}

impl spec::Verifier for SessionVerifier<'_> {
    type Error = Error;

    fn decode(&mut self, token: u32) -> Result<Vec<f32>, Error> {
        self.s.decode(self.e, token)
    }

    fn verify(&mut self, tokens: &[u32]) -> Result<(), Error> {
        self.s.verify(self.e, tokens)
    }

    fn row(&mut self, i: usize) -> Result<Vec<f32>, Error> {
        self.s.row(self.e, i)
    }

    fn commit(&mut self, kept: usize) -> Result<(), Error> {
        self.s.commit(self.e, kept)
    }

    fn n_vocab(&self) -> usize {
        self.e.dims.n_vocab
    }

    fn most(&self) -> usize {
        MAX_TOKENS.min(self.e.prefill.chunk)
    }

    fn mtp_draft(&mut self, x: u32, k: usize) -> Result<Vec<u32>, Error> {
        self.s.mtp_draft(self.e, x, k)
    }

    fn has_mtp(&self) -> bool {
        self.s.has_mtp()
    }

    fn position(&self) -> Option<usize> {
        Some(self.s.pos())
    }

    fn misses(&self) -> u64 {
        self.e.stats().misses
    }

    fn miss_bytes(&self) -> u64 {
        self.e.expert_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefill_bytes;

    #[test]
    fn gather_rows_keeps_the_first_kept_rows_of_each_layer() {
        // 3 layers, 4 tokens, 2 routed experts a token; layer l, token i, slot j holds 100 l + 10 i + j.
        let id = |l: usize, i: usize, j: usize| (100 * l + 10 * i + j) as f32;
        let ids: Vec<f32> = (0..3)
            .flat_map(|l| (0..4).flat_map(move |i| (0..2).map(move |j| id(l, i, j))))
            .collect();
        let got = gather_rows(&ids, 3, 4, 2, 2);
        let want: Vec<f32> = (0..3)
            .flat_map(|l| (0..2).flat_map(move |i| (0..2).map(move |j| id(l, i, j))))
            .collect();
        assert_eq!(got, want);
        assert_eq!(got.len(), 3 * 2 * 2);
        // Keeping every row is the log itself; keeping one is each layer's first token.
        assert_eq!(gather_rows(&ids, 3, 4, 2, 4), ids);
        assert_eq!(
            gather_rows(&ids, 3, 4, 2, 1),
            [0.0, 1.0, 100.0, 101.0, 200.0, 201.0]
        );
    }

    #[test]
    #[should_panic(expected = "kept 5 rows of a verify of 4")]
    fn gather_rows_refuses_more_rows_than_the_step_ran() {
        gather_rows(&[0.0; 24], 3, 4, 2, 5);
    }

    #[test]
    fn the_verify_layout_fits_in_prefills_work_at_the_production_chunk() {
        let d = Dims::QWEN36;
        let l = VerifyLayout::new(&d, 65_536, 16);
        // About 127 MiB against the work area's 225 MiB at the production chunk (the DeltaNet layers' temporaries).
        assert!(l.bytes < 225 << 20, "{} MiB", l.bytes >> 20);
        assert!(l.bytes > 120 << 20, "{} MiB", l.bytes >> 20);
        // Sized for it, the prefill buffers are the same 414 MiB: speculation adds no VRAM.
        assert_eq!(prefill_bytes(&d, 2048, Some(65_536)) >> 20, 414);
        assert_eq!(
            prefill_bytes(&d, 2048, Some(65_536)),
            prefill_bytes(&d, 2048, None)
        );
        // A 16-token chunk's work area (the DeltaNet layers' temporaries) is smaller than the layout, so it grows: the
        // figures of `the_prefill_buffers_on_the_real_model` hold without speculation.
        assert_eq!(prefill_bytes(&d, 16, None) >> 20, 18);
        assert!(prefill_bytes(&d, 16, Some(65_536)) > prefill_bytes(&d, 16, None));
    }

    #[test]
    fn the_layouts_regions_are_aligned_ordered_and_disjoint() {
        let d = Dims::QWEN36;
        let l = VerifyLayout::new(&d, 65_536, 16);
        // The real model's byte total at the production shape, pinned: a swapped dimension moves it.
        assert_eq!(l.bytes, 127_899_328);
        // Every region in the order the carve made them, with its size in bytes: each starts exactly where the one
        // before ends (rounded up to 16 bytes), so each takes the size it should and nothing overlaps.
        let t = 16u64;
        let f32s = |n: u64| 4 * n;
        let (e, ad, kv) = (2048, 4096, 512);
        let (inner, nv, cd) = (4096, 32, 8192);
        let mut regions = vec![
            (l.x, f32s(t * e)),
            (l.h, f32s(t * e)),
            (l.mix, f32s(t * e)),
            (l.resid, f32s(t * e)),
            (l.f, f32s(t * e)),
            (l.moe_out, f32s(t * e)),
            (l.shexp_out, f32s(t * e)),
            (l.ffn_out, f32s(t * e)),
            (l.qfull, f32s(t * 2 * ad)),
            (l.q, f32s(t * ad)),
            (l.gate, f32s(t * ad)),
            (l.k, f32s(t * kv)),
            (l.v, f32s(t * kv)),
            (l.pre, f32s(t * ad)),
            (l.gated, f32s(t * ad)),
            (l.z, f32s(t * inner)),
            (l.o, f32s(t * inner)),
            (l.nrm, f32s(t * inner)),
            (l.braw, f32s(t * nv)),
            (l.araw, f32s(t * nv)),
            (l.rlog, f32s(t * 256)),
            (l.probs, f32s(t * 256)),
            (l.ids, f32s(t * 8)),
            (l.w, f32s(t * 8)),
            (l.act, f32s(t * 8 * 512)),
            (l.ye, f32s(t * 8 * e)),
            (l.sg_g, f32s(t * 512)),
            (l.sg_u, f32s(t * 512)),
            (l.sact, f32s(t * 512)),
            (l.sh, f32s(t * e)),
            (l.sg, f32s(t)),
        ];
        regions.extend(l.unique.iter().map(|&u| (u, 4 + 8 * t * 8)));
        for r in l.deltanet.iter().flatten() {
            regions.extend([
                (r.raw, f32s(t * cd)),
                (r.post, f32s(t * cd)),
                (r.g, f32s(t * nv)),
                (r.beta, f32s(t * nv)),
                (r.replay_y, f32s(t * cd)),
                (r.replay_o, f32s(t * inner)),
            ]);
        }
        regions.push((l.part, 4 * t * 16 * 256 * (256 + 2)));
        let mut at = 0;
        for (i, &(start, size)) in regions.iter().enumerate() {
            assert_eq!(start % 16, 0, "region {i}");
            assert_eq!(start, at, "region {i} starts where the one before ends");
            at += size.next_multiple_of(16);
        }
        assert_eq!(at, l.bytes);
        // 30 DeltaNet layers each with its own regions; 40 lists, one a layer.
        assert_eq!(l.deltanet.iter().flatten().count(), 30);
        assert_eq!(l.deltanet.iter().filter(|r| r.is_none()).count(), 10);
        assert!(d.is_attention(3) && l.deltanet[3].is_none() && l.deltanet[0].is_some());
        assert_eq!(l.unique.len(), 40);
        // The partials are the last region and take exactly the kernels' size at 16 queries and 65,536 positions
        // (16 x 4.2 MiB).
        assert_eq!(
            l.bytes - l.part,
            tokens::attn_tokens_part_bytes(16, 256, attn_splits(65_536), 16)
        );
        // The step's sizes grow with the tokens and the context, and nothing else.
        assert!(VerifyLayout::new(&d, 65_536, 5).bytes < l.bytes);
        assert!(VerifyLayout::new(&d, 4_096, 16).bytes < l.bytes);
    }

    #[test]
    fn a_session_deeper_than_the_engines_context_outgrows_the_work_area() {
        // The tiny model at an engine context of 64 and chunk 64: a verify of 16 fits the work area, and a verify of 16
        // in a session of 200,000 positions (782 attention splits) doesn't, which `Session::verify` refuses as
        // `VerifyBuffers`.
        let d = Dims::tiny();
        let work = crate::prefill::prefill_sizes(&d, 64, Some(64))[8];
        assert!(VerifyLayout::new(&d, 64, 16).bytes <= work);
        assert!(VerifyLayout::new(&d, 200_000, 16).bytes > work);
        // The sizes the GPU test `a_session_longer_than_the_engines_context_is_refused_not_overrun` expects. A 1-token
        // verify in that session needs only 902,704 bytes and fits, which is why that test verifies 16 tokens.
        assert_eq!(work, 1_246_336);
        assert_eq!(VerifyLayout::new(&d, 200_000, 16).bytes, 14_442_112);
        assert_eq!(VerifyLayout::new(&d, 200_000, 1).bytes, 902_704);
        assert!(VerifyLayout::new(&d, 200_000, 1).bytes <= work);
        // The real model at the production shape has room to spare for a session as long as the engine's context.
        let prod = crate::prefill::prefill_sizes(&Dims::QWEN36, 2048, Some(65_536))[8];
        assert!(VerifyLayout::new(&Dims::QWEN36, 65_536, 16).bytes <= prod);
    }

    #[test]
    #[should_panic(expected = "a verify of 17 tokens")]
    fn the_layout_refuses_more_than_max_tokens() {
        VerifyLayout::new(&Dims::QWEN36, 1_024, 17);
    }

    #[test]
    fn speculation_on_a_model_with_too_many_experts_per_token_is_refused() {
        // 16 tokens of 16 choices fill the kernels' 256 pairs; 17 choices don't. Both real shapes are inside.
        let on = Options {
            speculation: true,
            ..Options::default()
        };
        assert!(check_speculation(&on, &Dims::QWEN36).is_ok());
        assert!(check_speculation(&on, &Dims::tiny()).is_ok());
        let sixteen = Dims {
            n_expert_used: 16,
            ..Dims::tiny()
        };
        assert!(check_speculation(&on, &sixteen).is_ok());
        let seventeen = Dims {
            n_expert_used: 17,
            ..Dims::tiny()
        };
        assert!(matches!(
            check_speculation(&on, &seventeen),
            Err(Error::SpeculationRefused { .. })
        ));
        // Off, any model is as it was.
        assert!(check_speculation(&Options::default(), &seventeen).is_ok());
    }

    #[test]
    fn speculation_with_the_cpu_path_or_base_time_is_refused() {
        let on = |cpu_share, base_time| Options {
            speculation: true,
            cpu_share,
            base_time,
            ..Options::default()
        };
        match check_speculation(&on(1.0, false), &Dims::QWEN36) {
            Err(Error::SpeculationRefused { why }) => {
                assert_eq!(
                    why,
                    "speculation needs the CPU miss path off (--cpu-share 0)"
                )
            }
            other => panic!("share 1.0: expected SpeculationRefused, got {other:?}"),
        }
        // 1/1024 assigns no expert (`path_on` is false below 1/512): the path is off.
        assert!(check_speculation(&on(1.0 / 1024.0, false), &Dims::QWEN36).is_ok());
        assert!(check_speculation(&on(0.0, false), &Dims::QWEN36).is_ok());
        match check_speculation(&on(0.0, true), &Dims::QWEN36) {
            Err(Error::SpeculationRefused { why }) => {
                assert_eq!(why, "speculation can't run with --base-time")
            }
            other => panic!("base time: expected SpeculationRefused, got {other:?}"),
        }
        // Without speculation both modes are as they were.
        let off = Options {
            cpu_share: 1.0,
            base_time: true,
            ..Options::default()
        };
        assert!(check_speculation(&off, &Dims::QWEN36).is_ok());
    }

    #[test]
    fn mtp_without_speculation_is_refused() {
        let opts = |speculation| Options {
            speculation,
            mtp: Some(crate::MtpOptions {
                rows: 1024,
                ring: 256,
                window: 512,
                head_rows: 0,
            }),
            ..Options::default()
        };
        match check_speculation(&opts(false), &Dims::QWEN36) {
            Err(Error::SpeculationRefused { why }) => assert!(why.contains("MTP"), "{why}"),
            other => panic!("expected SpeculationRefused, got {other:?}"),
        }
        // With speculation on, the MTP is as acceptable as speculation is.
        assert!(check_speculation(&opts(true), &Dims::QWEN36).is_ok());
        // And speculation alone, or neither, is as it was.
        let plain = |speculation| Options {
            speculation,
            ..Options::default()
        };
        assert!(check_speculation(&plain(false), &Dims::QWEN36).is_ok());
        assert!(check_speculation(&plain(true), &Dims::QWEN36).is_ok());
    }
}
