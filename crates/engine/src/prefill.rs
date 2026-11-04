//! Batched prefill: a prompt runs in chunks of up to `PREFILL_CHUNK` tokens. A chunk is one step, with one
//! submit per layer, so every submit stays well under the watchdog's limit. A chunk of at least `STAGE_FROM` tokens is
//! staged: each layer's non-resident experts are copied into a staging buffer while the layer
//! before computes, and the grouped kernels read them there through the prefill table; a shorter chunk reads them in
//! place through the slot table, as decode does. The dense projections and grouped experts run on split-f16
//! cooperative matrices where the device has them. The split has a limit the f32 kernels don't: an input above
//! f16's 65,504 makes its hi infinite (`quant8.glsl`'s `split2`), the outputs it reaches NaN, and prefill fails with
//! `Error::NonFinite` where the f32 kernels and decode would run. No golden or long prompt came near it; such a
//! prompt runs with `Options::f32_prefill` (`--f32-prefill`). Only the positions whose logits are wanted get the
//! output head, in sub-batches of `LOGIT_ROWS`.
use crate::{
    session::LayerState,
    timing::{CpuPhase, StepKind, TAG_EMBED, TAG_OUTPUT},
    verify::VerifyLayout,
    weights::GpuMixer,
    Engine, Error, Session,
};
use kernels::{
    dispatch::{moe_tiles, read_status, MatW, STATUS_BYTES, STATUS_CLEAR},
    twin::attention::rope_table,
};
use qwen36::dims::Dims;
use std::time::Instant;

/// Tokens per prefill chunk, at most, and the most `Options::prefill_chunk` may ask for: the buffers and the
/// kernels' grids are sized for it. At 4,096 tokens `sigmoid_mul` alone would need 65,536 workgroups on the real model,
/// past the 65,535 Vulkan guarantees.
pub const PREFILL_CHUNK: usize = 2048;

/// Positions whose logits one submit computes and reads back: one tile of `matmul`'s tokens, 64 MiB on the real model.
pub const LOGIT_ROWS: usize = 64;

/// Query-key pairs one chunk's attention may take in each attention layer, with the f32 kernel (`attn_prefill`, the
/// fallback and `--f32-prefill`). It takes about 1.4 to 2.1 ns a pair on the RX 9070 XT (2,048 queries from position
/// 25,600 in 74 to 115 ms), so this is about 35 to 55 ms, and a layer's submit stays well under 0.5 s with everything
/// else in it. The cap dates from a kernel about three times slower; it is kept for the watchdog's headroom, and the
/// results don't depend on the chunking. Deep in a long prompt, chunks get shorter to stay within it.
pub const ATTN_PAIRS: usize = 2048 * 12_288;

/// `ATTN_PAIRS` for the cooperative-matrix kernel (`attn_prefill_cm`): the same time a submit the f32 kernel has under
/// `ATTN_PAIRS`. Measured on the RX 9070 XT on 2026-10-18 at 2,048 queries from position 25,600:
/// `attn_prefill` 85.46 ms (1.57 ns a pair), `attn_prefill_cm`
/// 24.44 ms (0.45 ns a pair), 3.50 times as fast. So `2048 * K` with K = floor(12,288 * 85.46 / 24.44 / 1,024) * 1,024
/// = 41,984: 86.0M pairs, of which the 27,416-token prompt's fullest chunk (2,048 from 24,576) takes 52.4M.
pub const ATTN_PAIRS_CM: usize = 2048 * 41_984;

/// Chunks of at least this many tokens are staged: a layer's non-resident experts are copied
/// into a staging buffer while the layer before computes, and the grouped kernels read them from VRAM. Any past the
/// buffer's capacity (a swap in flight), and entries that don't point at one of the layer's pinned slots, are still read
/// in place. A shorter chunk reads the few experts it routes to in place, which costs less than copying them all.
/// Measured at 128 tokens: staging every non-resident
/// expert costs a fixed copy per chunk, about 710 ms here, which only a chunk this long repays; it changes where weights
/// are read from, never which kernel runs.
pub const STAGE_FROM: usize = 128;

/// Query-key pairs in a chunk of `t` tokens from position `pos`: query i sees pos + i + 1 positions.
fn pairs(pos: usize, t: usize) -> usize {
    t * pos + t * (t + 1) / 2
}

/// The lengths of the chunks that prefill `n` tokens from position `pos`: at most `max` each, and shorter where a
/// chunk's attention would take more than `cap` pairs (`ATTN_PAIRS` or `ATTN_PAIRS_CM`); never shorter than one token.
pub fn chunk_lengths(pos: usize, n: usize, max: usize, cap: usize) -> Vec<usize> {
    let (mut out, mut p, mut left) = (Vec::new(), pos, n);
    while left > 0 {
        let (mut lo, mut hi) = (1, max.min(left).max(1));
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if pairs(p, mid) <= cap {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        out.push(lo);
        p += lo;
        left -= lo;
    }
    out
}

/// Byte offsets, 256-aligned, of consecutive buffers of `counts` f32s (or u32s) each, and their total.
pub(crate) fn carve(counts: &[usize]) -> (Vec<u64>, u64) {
    let mut at = 0u64;
    let offs = counts
        .iter()
        .map(|&n| {
            let o = at;
            at += (4 * n as u64).next_multiple_of(256);
            o
        })
        .collect();
    (offs, at)
}

/// A DeltaNet layer's temporaries for a chunk of `t` tokens, in f32s: qkv, z, braw, araw, beta, g, conv, o, nrm.
fn deltanet_counts(d: &Dims, t: usize) -> [usize; 9] {
    let (nv, cd, inner) = (d.ssm_dt_rank, d.conv_dim(), d.ssm_inner);
    [
        t * cd,
        t * inner,
        t * nv,
        t * nv,
        t * nv,
        t * nv,
        t * cd,
        t * inner,
        t * inner,
    ]
}

/// An attention layer's: qfull, q, gate, k, v, pre, gated.
fn attention_counts(d: &Dims, t: usize) -> [usize; 7] {
    let (ad, kv) = (d.n_head * d.head_dim, d.n_head_kv * d.head_dim);
    [t * 2 * ad, t * ad, t * ad, t * kv, t * kv, t * ad, t * ad]
}

/// The MoE block's: rlog, probs, ids, w, list, tiles (4 u32s each), act, ye, sg_g, sg_u, sact, sh, sg.
fn moe_counts(d: &Dims, t: usize) -> [usize; 13] {
    let (k, e, ff) = (d.n_expert_used, d.n_embd, d.n_ff_shexp);
    [
        t * d.n_expert,
        t * d.n_expert,
        t * k,
        t * k,
        t * k,
        4 * moe_tiles(t * k, d.n_expert),
        t * k * d.n_ff_exp,
        t * k * e,
        t * ff,
        t * ff,
        t * ff,
        t * e,
        t,
    ]
}

/// The prefill buffers' VRAM allocations for chunks of up to `chunk` tokens, in the order `Prefill::new` makes them:
/// the activations every layer passes on (x, h, mix, resid, f, moe_out, shexp_out, ffn_out), the work area the mixer's
/// and the MoE block's temporaries share (they never live at once), and the output head's sub-batch (its normed
/// inputs and its logits). With `verify_ctx` (speculation on, a session of up to that many positions) the work area is
/// also at least a verify step's of `MAX_TOKENS` tokens (`VerifyLayout`): at the production chunk prefill's need is the
/// larger, so speculation adds no bytes there.
pub(crate) fn prefill_sizes(d: &Dims, chunk: usize, verify_ctx: Option<usize>) -> Vec<u64> {
    let work = [
        carve(&deltanet_counts(d, chunk)).1,
        carve(&attention_counts(d, chunk)).1,
        carve(&moe_counts(d, chunk)).1,
    ]
    .into_iter()
    .chain(verify_ctx.map(|ctx| VerifyLayout::new(d, ctx, kernels::tokens::MAX_TOKENS).bytes))
    .max()
    .expect("at least three layouts");
    let rows = LOGIT_ROWS.min(chunk) as u64;
    let mut sizes = vec![4 * (chunk * d.n_embd) as u64; 8];
    sizes.extend([
        work,
        4 * rows * d.n_embd as u64,
        4 * rows * d.n_vocab as u64,
    ]);
    sizes
}

/// The prefill work area's bytes (the ninth allocation of `prefill_sizes`): the temporaries a verify step and the MTP's
/// rows share.
pub(crate) fn work_bytes(d: &Dims, chunk: usize, verify_ctx: Option<usize>) -> u64 {
    prefill_sizes(d, chunk, verify_ctx)[8]
}

/// A dense projection over `t` rows, on the cooperative matrices where the device has them:
/// prefill's and the MTP window's share it.
pub(crate) fn dense(e: &Engine, r: &mut vk::Recorder, w: MatW, x: u64, y: u64, t: usize) {
    if e.coop {
        e.k.matmul_cm(r, w, x, y, t)
    } else {
        e.k.matmul(r, w, x, y, t)
    }
}

/// The VRAM the prefill buffers for chunks of up to `chunk` tokens take, as their arena packs them: 414 MiB at 2,048
/// tokens on the real model, with or without speculation (`verify_ctx`, as `prefill_sizes` takes it).
pub fn prefill_bytes(d: &Dims, chunk: usize, verify_ctx: Option<usize>) -> u64 {
    vk::planned_bytes(&prefill_sizes(d, chunk, verify_ctx))
}

/// The prefill buffers' host allocations for chunks of up to `chunk` tokens, in the order `Prefill::new` makes them:
/// the embedding rows' addresses and the RoPE tables (pinned), and the routing log, its weights and the output head's
/// sub-batch of logits (read back).
fn prefill_host_sizes(d: &Dims, chunk: usize) -> [u64; 5] {
    let log = 4 * (d.n_layer * chunk * d.n_expert_used) as u64;
    [
        8 * chunk as u64,
        4 * (chunk * d.n_rot) as u64,
        log,
        log,
        4 * LOGIT_ROWS.min(chunk) as u64 * d.n_vocab as u64,
    ]
}

/// The host memory the prefill buffers for chunks of up to `chunk` tokens take: 66 MiB at 2,048 tokens on the real
/// model, mostly the 64 rows of logits read back.
pub fn prefill_host_bytes(d: &Dims, chunk: usize) -> u64 {
    prefill_host_sizes(d, chunk).iter().sum()
}

/// The chunk an engine prefills with: `asked` (`Options::prefill_chunk`) or `PREFILL_CHUNK`, cut to `ctx` when that is
/// smaller (0 declares no context). A chunk of 0, or past `PREFILL_CHUNK`, is refused rather than clamped: the caller
/// asked for something the engine can't run, and quietly running another chunk would hide it.
pub(crate) fn chunk_for(asked: Option<usize>, ctx: usize) -> Result<usize, Error> {
    let chunk = asked.unwrap_or(PREFILL_CHUNK);
    if !(1..=PREFILL_CHUNK).contains(&chunk) {
        return Err(Error::BadPrefillChunk {
            chunk,
            max: PREFILL_CHUNK,
        });
    }
    Ok(if ctx > 0 { chunk.min(ctx) } else { chunk })
}

/// The buffers one chunk of prefill works in.
pub(crate) struct Prefill {
    /// Tokens per chunk, at most.
    pub chunk: usize,
    pub x: vk::Slice,
    pub h: vk::Slice,
    pub mix: vk::Slice,
    pub resid: vk::Slice,
    pub f: vk::Slice,
    pub moe_out: vk::Slice,
    pub shexp_out: vk::Slice,
    pub ffn_out: vk::Slice,
    pub work: vk::Slice,
    pub fnorm: vk::Slice,
    pub logits: vk::Slice,
    /// Each token's embedding row address, one u64 a token, in pinned host memory.
    pub rows: vk::Buffer,
    /// Each token's RoPE table (`rope_table`), in pinned host memory.
    pub rope: vk::Buffer,
    /// The routing log: each layer's routed ids for the chunk, `[n_layer][tokens][n_expert_used]`.
    pub routes: vk::Buffer,
    /// Their weights, laid out the same.
    pub route_weights: vk::Buffer,
    /// The output head's sub-batch of logits, read back.
    pub out: vk::Buffer,
    /// The tokens of the last chunk that finished, which the routing log holds. 0 from a step's start until it
    /// finishes: a failed step overwrote only some layers of the log, which then holds no one chunk's routes.
    pub last: std::cell::Cell<usize>,
    /// The VRAM the slices above live in: `Engine::prefill_bytes` reports what it took.
    pub arena: vk::Arena,
}

impl Prefill {
    pub fn new(
        dev: &vk::Device,
        d: &Dims,
        chunk: usize,
        verify_ctx: Option<usize>,
    ) -> Result<Prefill, Error> {
        let sizes = prefill_sizes(d, chunk, verify_ctx);
        let mut arena = vk::Arena::planned(vk::MemKind::Vram, &sizes, "the prefill buffers");
        let mut s = Vec::with_capacity(sizes.len());
        for &b in &sizes {
            s.push(arena.alloc(dev, b)?);
        }
        let [x, h, mix, resid, f, moe_out, shexp_out, ffn_out, work, fnorm, logits] =
            s.try_into().expect("one slice per size");
        let [rows, rope, routes, route_weights, out] = prefill_host_sizes(d, chunk);
        Ok(Prefill {
            chunk,
            x,
            h,
            mix,
            resid,
            f,
            moe_out,
            shexp_out,
            ffn_out,
            work,
            fnorm,
            logits,
            rows: dev.alloc(
                rows,
                vk::MemKind::HostPinned,
                "the prefill's embedding rows",
            )?,
            rope: dev.alloc(rope, vk::MemKind::HostPinned, "the prefill's RoPE tables")?,
            routes: dev.alloc(routes, vk::MemKind::Readback, "the prefill's routing log")?,
            route_weights: dev.alloc(
                route_weights,
                vk::MemKind::Readback,
                "the prefill's routing weights",
            )?,
            out: dev.alloc(out, vk::MemKind::Readback, "the prefill's logits")?,
            last: std::cell::Cell::new(0),
            arena,
        })
    }

    /// Slices of the work area, laid out by `counts`.
    fn parts<const N: usize>(&self, counts: [usize; N]) -> [vk::Slice; N] {
        let (offs, _) = carve(&counts);
        std::array::from_fn(|i| self.work.sub(offs[i], 4 * counts[i] as u64))
    }
}

/// What prefill hands each wanted position's logits to, with the position's index.
type Each<'a, E> = dyn FnMut(usize, &[f32]) -> Result<(), E> + 'a;

/// Which positions of a prefill get logits: the last only, or every one from an index into the tokens.
enum Wanted {
    Last,
    From(usize),
}

impl Session {
    /// Prefills `tokens` at the next positions and returns the last one's logits. Fails without running
    /// anything when there are no tokens, a token is out of range, or they don't fit the context. After any other
    /// error the session's state is undefined; start a new session. One error comes after a chunk is committed: with
    /// timings on (`Engine::set_timing`), failing to read the chunk's staging-copy timings (`collect_copy_spans`)
    /// means the device is failing, and the session already holds the chunk (its positions and tokens) though its
    /// logits were never delivered.
    pub fn prefill(&mut self, e: &Engine, tokens: &[u32]) -> Result<Vec<f32>, Error> {
        let mut last = Vec::new();
        self.run_prefill(e, tokens, Wanted::Last, &mut |_, l: &[f32]| {
            last = l.to_vec();
            Ok::<(), Error>(())
        })?;
        Ok(last)
    }

    /// Prefills `tokens` and hands `each` the logits of every position from token `first` on, in order, with the
    /// token's index. `artemis kl`, test 5 and the prefill-vs-decode check read them; they are computed and read back
    /// `LOGIT_ROWS` at a time. Errors as `prefill`'s: with timings on, a failed read of a chunk's staging-copy timings
    /// (`collect_copy_spans`) comes after the session holds the chunk, whose logits `each` never got; the device is
    /// failing.
    pub fn prefill_each<E: From<Error>>(
        &mut self,
        e: &Engine,
        tokens: &[u32],
        first: usize,
        mut each: impl FnMut(usize, &[f32]) -> Result<(), E>,
    ) -> Result<(), E> {
        self.run_prefill(e, tokens, Wanted::From(first), &mut each)
    }

    /// The checks that refuse a prefill before it touches anything.
    fn check_prefill(&self, e: &Engine, tokens: &[u32]) -> Result<(), Error> {
        if self.engine != e.id {
            return Err(Error::WrongEngine);
        }
        let d = &e.dims;
        if tokens.is_empty() {
            return Err(Error::EmptyPrefill);
        }
        if let Some(&token) = tokens.iter().find(|&&t| t as usize >= d.n_vocab) {
            return Err(Error::TokenOutOfRange {
                token,
                n_vocab: d.n_vocab,
            });
        }
        if self.pos + tokens.len() > self.ctx {
            return Err(Error::PrefillTooLong {
                pos: self.pos,
                tokens: tokens.len(),
                ctx: self.ctx,
            });
        }
        Ok(())
    }

    /// A prefill with the session's write rule around it: validation first (a refused call changes nothing and never
    /// waits), then the wait for a pending snapshot copy below the start, then the body. On success the KV record
    /// takes the tokens from the start; on any error past validation, including `each`'s own (the chunk it was handed
    /// is committed, but the call's contract is that the state is undefined after an error), the record is cut back
    /// to the start and the session is unusable until `reset`.
    fn run_prefill<E: From<Error>>(
        &mut self,
        e: &Engine,
        tokens: &[u32],
        wanted: Wanted,
        each: &mut Each<E>,
    ) -> Result<(), E> {
        assert!(!self.pending(), "a prefill with a verify step pending");
        self.check_prefill(e, tokens)?;
        let start = self.pos;
        let out = self
            .wait_copy(e, start)
            .map_err(E::from)
            .and_then(|()| self.prefill_body(e, tokens, wanted, each));
        match &out {
            Ok(()) => self.kv.wrote(start, tokens),
            Err(_) => {
                self.kv.failed(start);
                if let Some(m) = &mut self.mtp {
                    // A chunk may have slid the rows before it failed: nothing of them can be cut by position.
                    m.on_reset();
                }
                self.usable = false;
            }
        }
        out
    }

    fn prefill_body<E: From<Error>>(
        &mut self,
        e: &Engine,
        tokens: &[u32],
        wanted: Wanted,
        each: &mut Each<E>,
    ) -> Result<(), E> {
        let first = match wanted {
            Wanted::Last => tokens.len() - 1,
            Wanted::From(i) => i,
        };
        // The MTP's window rows: planned for the whole call before its first chunk, so the rows and the
        // ring lead the GPU's; a failure anywhere in the call resets them (`run_prefill`).
        let window = self.begin_window(e, tokens.len());
        let mut at = 0;
        for t in chunk_lengths(self.pos, tokens.len(), e.prefill.chunk, e.attn_pairs()) {
            self.run_chunk(e, &tokens[at..at + t], window.as_ref())?;
            // The chunk's positions are in the state now; their logits come from its last layer's output.
            let from = first.max(at);
            for i0 in (from..at + t).step_by(LOGIT_ROWS) {
                let n = LOGIT_ROWS.min(at + t - i0);
                self.logits(e, i0 - at, n, self.pos - t + (i0 - at), &mut |i, l| {
                    each(i0 + i, l)
                })?;
            }
            at += t;
        }
        Ok(())
    }

    /// One chunk, one step: every layer over the chunk's tokens, one submit each. Leaves the last layer's output in
    /// `x` for the output head.
    fn run_chunk(
        &mut self,
        e: &Engine,
        tokens: &[u32],
        window: Option<&crate::mtp::WindowPlan>,
    ) -> Result<(), Error> {
        let (d, p) = (&e.dims, &e.prefill);
        let (t, pos) = (tokens.len(), self.pos);
        let step = |source| Error::Step {
            pos,
            layer: None,
            source,
        };
        // The routing log is about to be overwritten: until the step finishes it holds no one chunk's routes.
        p.last.set(0);
        // Swaps move on between steps.
        e.begin_timing(StepKind::Prefill, pos, t);
        let t0 = Instant::now();
        let kind = if e.will_stage(t) {
            crate::PolicyStep::Staged
        } else {
            crate::PolicyStep::Prefill
        };
        e.prepare_step(kind, t).map_err(|err| match err {
            Error::Vk(source) => step(source),
            other => other,
        })?;
        e.time(CpuPhase::Prepare, t0);
        let t0 = Instant::now();
        let rows: Vec<u8> = tokens
            .iter()
            .flat_map(|&tk| e.host.embd_row(tk).to_le_bytes())
            .collect();
        p.rows.write(0, &rows);
        let tables: Vec<f32> = (0..t)
            .flat_map(|i| rope_table((pos + i) as u32, d.n_rot, d.rope_base))
            .collect();
        p.rope.write_f32(0, &tables);
        e.status.write(0, &STATUS_CLEAR);
        // The step's slot-table changes, applied first, in layer 0's submit. If the step fails in any
        // submit they are staged again: each is an absolute write of one entry, so applying one twice is harmless, and
        // one that never landed lands with the next step.
        let staged = e.take_staged();
        // A long enough chunk reads every expert from VRAM: the prefill table, built from the
        // table this step runs with, points each non-resident expert at its place in its layer's staging buffer.
        let plan = e.stage_plan(t);
        e.time(CpuPhase::Record, t0);
        let out = self.prefill_layers(e, tokens, staged.len(), plan.as_ref(), window);
        if out.is_err() {
            e.restage(staged);
        }
        out?;
        let t0 = Instant::now();
        p.last.set(t);
        let ids = p.routes.read_f32(0, d.n_layer * t * d.n_expert_used);
        e.count_routes(&ids, t, plan.as_ref().map(|s| s.staged.as_slice()));
        e.time(CpuPhase::Read, t0);
        e.observe_routes(&ids, t);
        self.pos += t;
        self.tokens.extend_from_slice(tokens);
        // After the commit: an error here (timings on only) means the device is failing, and the session already holds
        // the chunk (`Session::prefill`'s doc).
        e.collect_copy_spans()?;
        Ok(())
    }

    /// Every layer over the chunk's tokens, one submit each; the step's `updates` slot-table changes are applied first, in
    /// layer 0's submit. With a `plan` the chunk is staged: the prefill table is uploaded, and each layer's staging copy
    /// sent a layer ahead and waited for by its submit. The CPU time spent writing the prefill
    /// table and sending the copies counts as `CpuPhase::Record`, so a step's phases still sum to its host time.
    fn prefill_layers(
        &self,
        e: &Engine,
        tokens: &[u32],
        updates: usize,
        plan: Option<&crate::stage::StagePlan>,
        window: Option<&crate::mtp::WindowPlan>,
    ) -> Result<(), Error> {
        let (d, k, p) = (&e.dims, &e.k, &e.prefill);
        let (t, pos) = (tokens.len(), self.pos);
        let step = |layer: usize, source| Error::Step {
            pos,
            layer: Some(layer),
            source,
        };
        // The experts' table this chunk reads: the prefill table when staged, the slot table otherwise.
        let staging = plan.and(e.staging.as_ref());
        let table = staging.map_or(e.table.addr, |st| st.table.addr);
        // Layer l's staging copy, into buffer l % 2: the first two go out now, layer l + 2's as
        // soon as layer l, the last to read that buffer, has finished, so each copy runs while the layer before it
        // computes. Each copy names, as `after`, the compute submit that last read its buffer: for the first two, the
        // last compute submit so far. The previous staged chunk's last two layers read the buffers at or before it, and
        // so did every decode step and unstaged chunk that ran while the cache held the buffers' memory as its lent
        // rows, so it must stay the last submit, not the last staged chunk's. Copies into
        // the lent rows already sent land first, behind the transfer queue's leading barrier. For layer l + 2's copy,
        // `after` is layer l's. Already finished, they cost nothing to wait for, and the dependency doesn't rest on what
        // the host has seen finish (`Queue::synced`). copied[l] is the transfer value layer l waits for.
        let mut copied = vec![0u64; d.n_layer];
        if let (Some(s), Some(st)) = (plan, staging) {
            // Writing the prefill table and sending the copies counts as recording (`CpuTimes::record_us`).
            let t0 = Instant::now();
            st.table_host.write(0, &s.table);
            let after = e.dev.submitted(vk::QueueKind::Compute);
            let sent =
                copied
                    .iter_mut()
                    .enumerate()
                    .take(2)
                    .try_for_each(|(l, c)| -> Result<(), Error> {
                        *c = e.stage_layer(s, l, pos, after)?;
                        Ok(())
                    });
            e.time(CpuPhase::Record, t0);
            sent?;
        }
        for (l, gl) in e.w.layers.iter().enumerate() {
            let t0 = Instant::now();
            let mut r = e.record().map_err(|s| step(l, s))?;
            if l == 0 {
                r.tag(TAG_EMBED);
                if updates > 0 {
                    k.apply_slot_updates(
                        &mut r,
                        e.table.addr,
                        e.updates.addr(),
                        updates,
                        (d.n_layer + 1) * d.n_expert,
                    );
                    r.barrier();
                }
                if let Some(st) = staging {
                    // After the step's table changes, before any layer reads the prefill table.
                    r.copy(st.table_host.slice(0, st.table.size), st.table);
                    r.barrier();
                }
                k.embed_rows(&mut r, p.rows.addr(), e.host.embd_ty, d.n_embd, p.x.addr, t);
                r.barrier();
            }
            r.tag(l as u32);
            // This layer's experts must have landed in its staging buffer (a no-op wait when unstaged).
            r.wait(vk::QueueKind::Transfer, copied[l]);
            self.record_layer(e, &mut r, l, gl, t, pos, table);
            if l + 1 == d.n_layer {
                // The last layer's command buffer also writes the chunk's MTP window rows.
                if let Some(w) = window {
                    self.record_window(e, &mut r, w, pos, t);
                }
            }
            r.host_barrier();
            e.time(CpuPhase::Record, t0);
            e.submit_step(r).map_err(|s| step(l, s))?;
            let t0 = Instant::now();
            let mut st = [0u8; STATUS_BYTES];
            e.status.read(0, &mut st);
            let st = read_status(&st);
            e.time(CpuPhase::Read, t0);
            if let Some((layer, expert)) = st.bad {
                return Err(Error::PrefillBadSlotEntry {
                    pos,
                    tokens: t,
                    layer,
                    expert,
                });
            }
            if let Some((layer, expert)) = st.cpu {
                return Err(Error::PrefillExpertOnCpu {
                    pos,
                    tokens: t,
                    layer,
                    expert,
                });
            }
            if let Some(s) = plan {
                if l + 2 < d.n_layer {
                    // Layer l has finished (its submit was waited for): its buffer is free for layer l + 2. Sending
                    // the copy counts as recording.
                    let t0 = Instant::now();
                    let sent =
                        e.stage_layer(s, l + 2, pos, e.dev.submitted(vk::QueueKind::Compute));
                    e.time(CpuPhase::Record, t0);
                    copied[l + 2] = sent?;
                }
            }
        }
        Ok(())
    }

    /// Layer `l` over the chunk's `t` tokens at positions `pos ..`, in graph order (semantics reference §2 to §5).
    /// `table` is the experts' table: the prefill table in a staged chunk, the slot table otherwise.
    #[allow(clippy::too_many_arguments)]
    fn record_layer(
        &self,
        e: &Engine,
        r: &mut vk::Recorder,
        l: usize,
        gl: &crate::weights::GpuLayer,
        t: usize,
        pos: usize,
        table: u64,
    ) {
        let (d, k, p) = (&e.dims, &e.k, &e.prefill);
        let eps = d.rms_eps;
        // The dense projections and the grouped experts run on cooperative matrices with split-f16 inputs where the
        // device has them. The router, `ssm_alpha`, `ssm_beta` and the shared expert's gate
        // input stay on the f32 kernel, so near-ties route as decode routes them.
        let dense = |r: &mut vk::Recorder, w: MatW, x: u64, y: u64| dense(e, r, w, x, y, t);
        k.rmsnorm_tokens(r, p.x.addr, gl.attn_norm, p.h.addr, d.n_embd, eps, t);
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
                let [w_qkv, w_z, braw, araw, w_beta, w_g, w_conv, w_o, nrm] =
                    p.parts(deltanet_counts(d, p.chunk));
                let (nv, dk, dv, ng, kd, cd) = (
                    d.ssm_dt_rank,
                    d.ssm_d_state,
                    d.v_head_dim(),
                    d.ssm_n_group,
                    d.key_dim(),
                    d.conv_dim(),
                );
                dense(r, *qkv, p.h.addr, w_qkv.addr);
                dense(r, *z, p.h.addr, w_z.addr);
                k.matmul(r, *beta, p.h.addr, braw.addr, t);
                k.matmul(r, *alpha, p.h.addr, araw.addr, t);
                r.barrier();
                k.gdn_gates_tokens(
                    r,
                    braw.addr,
                    araw.addr,
                    *dt_bias,
                    *a,
                    w_beta.addr,
                    w_g.addr,
                    nv,
                    t,
                );
                k.conv_seq(
                    r,
                    w_qkv.addr,
                    *conv1d,
                    conv.addr,
                    w_conv.addr,
                    cd,
                    d.ssm_conv,
                    t,
                    true,
                );
                r.barrier();
                k.l2norm_tokens(r, w_conv.addr, 2 * ng, dk, eps, t, cd);
                r.barrier();
                k.gdn_seq(
                    r,
                    w_conv.addr,
                    w_conv.addr + 4 * kd as u64,
                    w_conv.addr + 8 * kd as u64,
                    w_g.addr,
                    w_beta.addr,
                    state.addr,
                    w_o.addr,
                    nv,
                    dk,
                    dv,
                    ng,
                    t,
                    cd,
                    true,
                );
                r.barrier();
                k.gated_rmsnorm(r, w_o.addr, w_z.addr, *norm, nrm.addr, t * nv, dv, eps);
                r.barrier();
                dense(r, *out, nrm.addr, p.mix.addr);
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
                let [qfull, wq, gate, wk, wv, pre, gated] = p.parts(attention_counts(d, p.chunk));
                let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
                dense(r, *q, p.h.addr, qfull.addr);
                dense(r, *kw, p.h.addr, wk.addr);
                dense(r, *v, p.h.addr, wv.addr);
                r.barrier();
                k.qk_norm_rope_tokens(
                    r,
                    qfull.addr,
                    wk.addr,
                    wq.addr,
                    gate.addr,
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
                k.kv_write_tokens(r, wk.addr, wv.addr, kc.addr, vc.addr, pos, nkv * hd, t);
                r.barrier();
                if e.coop_attn {
                    k.attn_prefill_cm(
                        r, wq.addr, kc.addr, vc.addr, pre.addr, pos, t, nh, nkv, hd, 0,
                    );
                } else {
                    k.attn_prefill(r, wq.addr, kc.addr, vc.addr, pre.addr, pos, t, nh, nkv, hd);
                }
                r.barrier();
                k.sigmoid_mul(r, pre.addr, gate.addr, gated.addr, t * nh * hd);
                r.barrier();
                dense(r, *out, gated.addr, p.mix.addr);
            }
            _ => unreachable!("layer {l}'s weights and state are of the same kind"),
        }
        r.barrier();
        let n = t * d.n_embd;
        k.add(r, p.x.addr, p.mix.addr, p.resid.addr, n);
        r.barrier();
        k.rmsnorm_tokens(r, p.resid.addr, gl.ffn_norm, p.f.addr, d.n_embd, eps, t);
        r.barrier();
        let [rlog, probs, ids, w, list, tiles, act, ye, sg_g, sg_u, sact, sh, sg] =
            p.parts(moe_counts(d, p.chunk));
        let (ke, ne) = (d.n_expert_used, d.n_expert);
        k.matmul(r, gl.router, p.f.addr, rlog.addr, t);
        r.barrier();
        k.router_topk_tokens(r, rlog.addr, probs.addr, ids.addr, w.addr, ne, ke, t);
        r.barrier();
        // The routing log: this layer's routed ids, read back for the cache's statistics.
        let ids_bytes = 4 * (t * ke) as u64;
        r.copy(
            ids.sub(0, ids_bytes),
            p.routes.slice(l as u64 * ids_bytes, ids_bytes),
        );
        r.copy(
            w.sub(0, ids_bytes),
            p.route_weights.slice(l as u64 * ids_bytes, ids_bytes),
        );
        let max_tiles = moe_tiles(t * ke, ne);
        k.moe_group(r, ids.addr, list.addr, tiles.addr, ne, t * ke, max_tiles);
        r.barrier();
        let status = e.status.addr();
        // One helper per grouped pair picks the kernel by `e.coop`; the table is the caller's argument, so
        // switching tables happens at the two calls below.
        let gate_up = |r: &mut vk::Recorder, table: u64| {
            if e.coop {
                k.experts_matmul_cm_gate_up(
                    r, table, l, ne, list.addr, tiles.addr, max_tiles, p.f.addr, act.addr, status,
                    &gl.slot, ke,
                );
            } else {
                k.experts_matmul_gate_up(
                    r, table, l, ne, list.addr, tiles.addr, max_tiles, p.f.addr, act.addr, status,
                    &gl.slot, ke,
                );
            }
        };
        let down = |r: &mut vk::Recorder, table: u64| {
            if e.coop {
                k.experts_matmul_cm_down(
                    r, table, l, ne, list.addr, tiles.addr, max_tiles, act.addr, ye.addr, status,
                    &gl.slot,
                );
            } else {
                k.experts_matmul_down(
                    r, table, l, ne, list.addr, tiles.addr, max_tiles, act.addr, ye.addr, status,
                    &gl.slot,
                );
            }
        };
        gate_up(r, table);
        dense(r, gl.sh_gate, p.f.addr, sg_g.addr);
        dense(r, gl.sh_up, p.f.addr, sg_u.addr);
        k.matmul(r, gl.sh_gate_inp, p.f.addr, sg.addr, t);
        r.barrier();
        down(r, table);
        k.swiglu(r, sg_g.addr, sg_u.addr, sact.addr, t * d.n_ff_shexp);
        r.barrier();
        dense(r, gl.sh_down, sact.addr, sh.addr);
        r.barrier();
        k.moe_combine_tokens(
            r,
            ye.addr,
            w.addr,
            sh.addr,
            sg.addr,
            p.moe_out.addr,
            p.shexp_out.addr,
            d.n_embd,
            ke,
            t,
        );
        r.barrier();
        k.add(r, p.moe_out.addr, p.shexp_out.addr, p.ffn_out.addr, n);
        r.barrier();
        k.add(r, p.resid.addr, p.ffn_out.addr, p.x.addr, n);
    }

    /// The logits of rows `i0 .. i0 + n` of the last chunk (positions `pos ..`), from its last layer's output: the
    /// final norm and the output head over the `n` rows, one submit, then `each` for every row in order.
    fn logits<E: From<Error>>(
        &self,
        e: &Engine,
        i0: usize,
        n: usize,
        pos: usize,
        each: &mut Each<E>,
    ) -> Result<(), E> {
        let (d, k, p) = (&e.dims, &e.k, &e.prefill);
        let step = |source| Error::Step {
            pos,
            layer: None,
            source,
        };
        let t0 = Instant::now();
        let mut r = e.record().map_err(step)?;
        r.tag(TAG_OUTPUT);
        let x = p.x.addr + 4 * (i0 * d.n_embd) as u64;
        k.rmsnorm_tokens(
            &mut r,
            x,
            e.w.output_norm,
            p.fnorm.addr,
            d.n_embd,
            d.rms_eps,
            n,
        );
        r.barrier();
        k.matmul(&mut r, e.w.output, p.fnorm.addr, p.logits.addr, n);
        r.barrier();
        let bytes = 4 * (n * d.n_vocab) as u64;
        r.copy(p.logits.sub(0, bytes), p.out.slice(0, bytes));
        r.host_barrier();
        e.time(CpuPhase::Record, t0);
        e.submit_step(r).map_err(step)?;
        for i in 0..n {
            let t0 = Instant::now();
            let logits = p.out.read_f32(4 * (i * d.n_vocab) as u64, d.n_vocab);
            let bad = crate::sample::first_non_finite(&logits);
            e.time(CpuPhase::Read, t0);
            if let Some((index, value)) = bad {
                return Err(Error::NonFinite {
                    pos: pos + i,
                    index,
                    value,
                }
                .into());
            }
            each(i, &logits)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_the_tokens_and_stay_within_the_attention_budget() {
        for cap in [ATTN_PAIRS, ATTN_PAIRS_CM] {
            for (pos, n, max) in [
                (0, 1, 2048),
                (0, 5000, 2048),
                (0, 27_416, 2048),
                (25_000, 3000, 2048),
                (60_000, 5000, 2048),
                (0, 40, 7),
            ] {
                let c = chunk_lengths(pos, n, max, cap);
                assert_eq!(c.iter().sum::<usize>(), n, "{pos} {n} {max}");
                let mut p = pos;
                for &t in &c {
                    assert!((1..=max).contains(&t), "{pos} {n} {max}: {t}");
                    assert!(pairs(p, t) <= cap || t == 1, "{p}: {t}");
                    p += t;
                }
            }
        }
        // Near the start, chunks are full; deep in the context, shorter.
        assert_eq!(chunk_lengths(0, 5000, 2048, ATTN_PAIRS), [2048, 2048, 904]);
        assert_eq!(chunk_lengths(0, 40, 7, ATTN_PAIRS), [7, 7, 7, 7, 7, 5]);
        let deep = chunk_lengths(25_000, 3000, 2048, ATTN_PAIRS);
        assert!(deep[0] < 2048 && deep[0] > 900, "{deep:?}");
        // The cooperative-matrix kernel's cap: the long prompt's chunks, and a deep chunk still cut short.
        assert_eq!(
            chunk_lengths(0, 27_416, 2048, ATTN_PAIRS_CM),
            [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 792]
        );
        assert!(
            chunk_lengths(0, 27_416, 2048, ATTN_PAIRS_CM).len()
                < chunk_lengths(0, 27_416, 2048, ATTN_PAIRS).len()
        );
        let deep = ATTN_PAIRS_CM / 2048 + 1;
        assert!(chunk_lengths(deep, 2048, 2048, ATTN_PAIRS_CM)[0] < 2048);
        const { assert!(ATTN_PAIRS_CM >= ATTN_PAIRS && ATTN_PAIRS_CM.is_multiple_of(2048)) };
    }

    #[test]
    fn the_prefill_buffers_on_the_real_model() {
        // At 2,048 tokens: eight [2048][2048] activations (128 MiB), the work area, sized for the DeltaNet layers'
        // temporaries (225 MiB; attention's take 200 and the MoE block's 192), and the output head's 64 rows
        // (62 MiB): 414 MiB.
        let b = prefill_bytes(&Dims::QWEN36, 2048, None);
        assert_eq!(b >> 20, 414, "{b}");
        // A 16-position context needs 16-token chunks, mostly the 16 rows of logits: 18 MiB.
        assert_eq!(prefill_bytes(&Dims::QWEN36, 16, None) >> 20, 18);
    }

    #[test]
    fn prefills_host_buffers_on_the_real_model() {
        // At 2,048 tokens, in host memory: the embedding rows' addresses (16 KiB), the RoPE tables (512 KiB), the
        // routing log and its weights (2.5 MiB each), and the output head's 64 rows of logits read back (60.6 MiB).
        assert_eq!(
            prefill_host_bytes(&Dims::QWEN36, 2048),
            16_384 + 524_288 + 2 * 2_621_440 + 63_569_920
        );
    }

    #[test]
    fn a_chunk_is_from_one_token_to_prefill_chunk_cut_to_the_context() {
        assert_eq!(chunk_for(None, 0).unwrap(), PREFILL_CHUNK);
        assert_eq!(chunk_for(None, 65_536).unwrap(), PREFILL_CHUNK);
        assert_eq!(chunk_for(None, 10).unwrap(), 10);
        assert_eq!(chunk_for(Some(500), 4096).unwrap(), 500);
        assert_eq!(chunk_for(Some(16), 8).unwrap(), 8);
        assert_eq!(chunk_for(Some(1), 0).unwrap(), 1);
        assert_eq!(chunk_for(Some(PREFILL_CHUNK), 0).unwrap(), PREFILL_CHUNK);
        // More than PREFILL_CHUNK would pass Vulkan's workgroup limit mid-step (`sigmoid_mul` at 4,096 tokens on the
        // real model), and 0 tokens is no chunk: both are refused, whatever the context.
        for (bad, ctx) in [(0, 0), (0, 10), (PREFILL_CHUNK + 1, 0), (4096, 1000)] {
            match chunk_for(Some(bad), ctx) {
                Err(Error::BadPrefillChunk { chunk, max }) => {
                    assert_eq!((chunk, max), (bad, PREFILL_CHUNK))
                }
                other => panic!("{bad} at ctx {ctx}: expected BadPrefillChunk, got {other:?}"),
            }
        }
    }
}
