//! A sequence's state: f16 KV for the attention layers, f32 DeltaNet and conv state for the others, the
//! position and the tokens so far. One session runs at a time on an engine.
use crate::{
    mtp::{MtpGpu, MtpState},
    reuse::KvRecord,
    Engine, Error,
};
use kernels::dispatch::attn_splits;
use qwen36::{dims::Dims, taps::expected_taps};

pub(crate) enum LayerState {
    DeltaNet { conv: vk::Slice, s: vk::Slice },
    Attention { kc: vk::Slice, vc: vk::Slice },
}

/// Host buffers for one step's taps: what the GPU copies out, and the forced values it copies back in.
pub(crate) struct TapArenas {
    pub rb: vk::Buffer,
    pub force: vk::Buffer,
}

/// A session's state, read back (`Session::state`): per layer, (conv, recurrent) state for a DeltaNet layer or (keys,
/// values) for an attention layer, as f32.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionState {
    pub layers: Vec<(Vec<f32>, Vec<f32>)>,
}

pub struct Session {
    /// The engine that made it (`Engine::id`).
    pub(crate) engine: u64,
    pub(crate) ctx: usize,
    pub(crate) pos: usize,
    pub(crate) tokens: Vec<u32>,
    pub(crate) layers: Vec<LayerState>,
    /// `attn_decode`'s partials, sized for `ctx`.
    pub(crate) part: vk::Slice,
    pub(crate) taps: Option<TapArenas>,
    /// The tokens behind the live KV: every write goes through the session, so no caller can
    /// forget to record it. `reset` keeps it; a restore sets it.
    pub(crate) kv: KvRecord,
    /// Whether the session's state is defined: a write that fails past validation clears it (the state is undefined
    /// after such an error), `reset` sets it again.
    pub(crate) usable: bool,
    /// The newest snapshot copy still reading KV, as (its end position, its transfer-queue value); a write below that
    /// position waits for it (`wait_copy`). Set by `Snapshots`.
    pub(crate) kv_copy: Option<(usize, u64)>,
    /// How many writes found a pending copy below their start and waited for it.
    pub(crate) copy_waits: u64,
    /// The verify step that ran and is not yet committed: `Session::verify` sets it, `commit` and every
    /// failure path clear it. While it is set the session holds the step's every token (`pos`, `tokens` and the KV
    /// record), and every other operation that moves them `assert!`s it is clear.
    pub(crate) pending: Option<crate::verify::Pending>,
    /// Tests only (`Session::fail_next_commit`): the next `commit`'s replay fails.
    pub(crate) fail_commit: bool,
    /// Tests only (`Session::rows_read`): rows `row` has copied out of the verify's logits.
    pub(crate) rows_read: u64,
    /// The MTP's rows and hidden-row ring: Some when the engine opened with `Options::mtp`. Every
    /// session operation updates it beside its own bookkeeping; `reset`, `set_restored` and every failure past
    /// validation clear or cut it. The GPU work is recorded beside each update (`mtp.rs`).
    pub(crate) mtp: Option<MtpState>,
    /// The MTP's VRAM buffers (its KV, the hidden-row ring, the carry row, a pass's ids and addresses), allocated in
    /// the session's arena with the rest (`mtp_session_sizes`): Some exactly when `mtp` is.
    pub(crate) mtp_gpu: Option<MtpGpu>,
    /// Tests only (`fail_next_mtp_submit`): the next draft pass's submit fails.
    pub(crate) fail_mtp_submit: bool,
    /// Tests only (`capture_window_hidden`): a readback of the final-norm rows the prefill window computes, row `p` at byte
    /// `p * 4 * n_embd`, and which positions hold one.
    pub(crate) capture: Option<(vk::Buffer, Vec<bool>)>,
    /// Waits for the device when the session drops (`Drop`): the session has no `&Engine` there.
    idle: vk::IdleHandle,
    arena: vk::Arena,
}

impl Drop for Session {
    /// The device goes idle before the session's buffers are destroyed, always. Commands that read or write them may
    /// still be running: a snapshot copy left running on the transfer queue (`kv_copy`, set by `Snapshots::take`; it is
    /// cleared only once a later write sees the copy finished), a step that failed past validation (`!usable`), and any
    /// blocking submit that touches session buffers and timed out without marking the session unusable (`reset`'s
    /// recurrent-state zeroing, `poison_kv`, `Snapshots::take`'s state copy, `Session::new`'s zero fill). It also covers
    /// an unwind: a `pending` assert panics with the copy in flight. Session drops are rare and off the hot path, so
    /// the wait is not made conditional on a flag that those cases could miss. A failed wait is ignored: the device may
    /// be lost, and then nothing is running.
    fn drop(&mut self) {
        let _ = self.idle.wait_idle();
    }
}

/// Whether `e` is one of the errors a write returns before it touches the session: it changes nothing, so the KV record
/// and the usable flag stay as they were. Any other error leaves the state undefined.
pub(crate) fn is_validation(e: &Error) -> bool {
    matches!(
        e,
        Error::WrongEngine
            | Error::EmptyPrefill
            | Error::TokenOutOfRange { .. }
            | Error::PrefillTooLong { .. }
            | Error::ContextFull { .. }
            | Error::VerifyShape { .. }
            | Error::VerifyBuffers { .. }
            | Error::MtpShape { .. }
    )
}

/// One session's allocations: each attention layer's K or V cache, each DeltaNet layer's conv and recurrent state,
/// and the attention partials, in bytes.
pub(crate) fn state_sizes(d: &Dims, ctx: usize) -> (u64, u64, u64, u64) {
    let kv = (ctx * d.n_head_kv * d.head_dim * 2) as u64;
    let conv = (4 * d.conv_dim() * (d.ssm_conv - 1)) as u64;
    let state = (4 * d.ssm_dt_rank * d.v_head_dim() * d.ssm_d_state) as u64;
    let part = (4 * d.n_head * attn_splits(ctx) * (d.head_dim + 2)) as u64;
    (kv, conv, state, part)
}

/// A session's allocations, in the order `Session::new` makes them: per layer the K and V caches (attention) or the
/// conv and recurrent state (DeltaNet), then the attention partials.
pub(crate) fn session_sizes(d: &Dims, ctx: usize) -> Vec<u64> {
    let (kv, conv, state, part) = state_sizes(d, ctx);
    let mut sizes: Vec<u64> = (0..d.n_layer)
        .flat_map(|l| {
            if d.is_attention(l) {
                [kv, kv]
            } else {
                [conv, state]
            }
        })
        .collect();
    sizes.push(part);
    sizes
}

/// The MTP's sizes for a session, clamped to the session. Built by `for_session` only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpOptions {
    /// The MTP KV's rows: `min(mtp_ctx, ctx)`.
    pub rows: usize,
    /// The hidden-row ring's rows: `min(mtp_ring, rows / 4)`, rounded down to 16.
    pub ring: usize,
    /// The prompt's last positions prefill writes into the MTP's KV: `min(mtp_window, rows / 2)`.
    pub window: usize,
    /// 0 drafts with the whole head; N drafts from the first N rows of the main head (`mtp_head_rows`). The load clamps
    /// it to the vocabulary.
    pub head_rows: usize,
}

impl MtpOptions {
    /// The shortest session MTP runs in: `rows / 4` must hold a 16-row ring and `make_room`'s slide must have room.
    pub const MIN_CTX: usize = 512;

    /// What `cfg` asks of a session of `ctx` positions: None when speculation is off or `"mtp"` isn't a source, an
    /// error for a context under 512 positions. The load (`Engine::load_dims`) and the tools' loads both call it.
    pub fn for_session(cfg: &crate::SpecConfig, ctx: usize) -> Result<Option<MtpOptions>, Error> {
        if !cfg.enabled || !cfg.has(crate::SpecSource::Mtp) {
            return Ok(None);
        }
        if ctx < MtpOptions::MIN_CTX {
            return Err(Error::SpeculationRefused {
                why: "MTP needs a session of at least 512 positions: set speculation.sources = [\"lookup\"] or a ctx of at least 512",
            });
        }
        let rows = cfg.mtp_ctx.min(ctx);
        Ok(Some(MtpOptions {
            rows,
            ring: cfg.mtp_ring.min(rows / 4) / 16 * 16,
            window: cfg.mtp_window.min(rows / 2),
            head_rows: cfg.mtp_head_rows,
        }))
    }
}

/// The rows a draft pass may write at once: the whole ring (catch-up) and a verify's worth of chained rows.
fn mtp_pass_rows(ring: usize) -> u64 {
    (ring + kernels::tokens::MAX_TOKENS) as u64
}

/// The session's MTP buffers, in allocation order: the K cache and the V cache of `rows` rows, the
/// hidden-row ring, the carry row (the last hidden row, kept across a prefill chunk), the pass's token ids and its
/// embedding-row addresses. Each is rounded up to 256 bytes, the alignment the arena places every allocation at, so the
/// sizes are what the arena really takes (no padding hides between them) and the MTP's bytes are the same in a session
/// of any context. They go before the attention partials (`session_sizes_mtp`): the arena doesn't pad its last
/// allocation, so the MTP's buffers sit where the difference to a session without them is exactly their sum.
pub(crate) fn mtp_session_sizes(d: &Dims, rows: usize, ring: usize) -> Vec<u64> {
    let align = |n: u64| n.next_multiple_of(256);
    let half = (rows * d.n_head_kv * d.head_dim * 2) as u64;
    vec![
        align(half),
        align(half),
        align((ring * d.n_embd * 4) as u64),
        align(4 * d.n_embd as u64),
        align(4 * mtp_pass_rows(ring)),
        align(8 * mtp_pass_rows(ring)),
    ]
}

/// The VRAM the MTP adds to a session of `rows` MTP rows and a ring of `ring` rows: `2 · rows · n_head_kv · head_dim · 2`
/// for K and V, `ring · n_embd · 4` for the ring, then the carry row, the ids and the addresses, each rounded up to 256
/// bytes (16 MiB and 2 MiB at 8,192 and 256, and about 11 KiB more). It is exactly `session_bytes_mtp` less
/// `session_bytes`, whatever the context.
pub fn mtp_session_bytes(d: &Dims, rows: usize, ring: usize) -> u64 {
    mtp_session_sizes(d, rows, ring).iter().sum()
}

/// A session's allocations with the MTP's before the attention partials (the last) when `mtp` is set.
pub(crate) fn session_sizes_mtp(d: &Dims, ctx: usize, mtp: Option<MtpOptions>) -> Vec<u64> {
    let mut sizes = session_sizes(d, ctx);
    if let Some(o) = mtp {
        let part = sizes.pop().expect("the attention partials are the last");
        sizes.extend(mtp_session_sizes(d, o.rows, o.ring));
        sizes.push(part);
    }
    sizes
}

/// `session_bytes`, with the MTP's rows and ring when `mtp` is set.
pub fn session_bytes_mtp(d: &Dims, ctx: usize, mtp: Option<MtpOptions>) -> u64 {
    vk::planned_bytes(&session_sizes_mtp(d, ctx, mtp))
}

/// Zeroes every DeltaNet layer's conv and recurrent state, in one submit.
fn zero_recurrent(e: &Engine, layers: &[LayerState]) -> Result<(), Error> {
    let mut r = e.dev.record()?;
    for st in layers {
        if let LayerState::DeltaNet { conv, s } = st {
            r.fill(*conv, 0);
            r.fill(*s, 0);
        }
    }
    e.dev.submit(r)?;
    Ok(())
}

/// The VRAM a session of `ctx` positions takes: f16 K and V for the attention layers, f32 conv and
/// recurrent state for the DeltaNet layers, and the attention partials, as its arena packs them.
pub fn session_bytes(d: &Dims, ctx: usize) -> u64 {
    vk::planned_bytes(&session_sizes(d, ctx))
}

impl Session {
    /// A fresh sequence of up to `ctx` positions: zero DeltaNet and conv state, an empty KV cache.
    pub(crate) fn new(e: &Engine, ctx: usize) -> Result<Session, Error> {
        if ctx == 0 {
            return Err(Error::BadCtx { ctx });
        }
        let d = &e.dims;
        let (kv, conv, state, part) = state_sizes(d, ctx);
        let mut arena = vk::Arena::planned(
            vk::MemKind::Vram,
            &session_sizes_mtp(d, ctx, e.mtp),
            "the sequence state",
        );
        let mut layers = Vec::with_capacity(d.n_layer);
        for l in 0..d.n_layer {
            layers.push(if d.is_attention(l) {
                LayerState::Attention {
                    kc: arena.alloc(&e.dev, kv)?,
                    vc: arena.alloc(&e.dev, kv)?,
                }
            } else {
                LayerState::DeltaNet {
                    conv: arena.alloc(&e.dev, conv)?,
                    s: arena.alloc(&e.dev, state)?,
                }
            });
        }
        // The MTP's buffers go before `part`, in `mtp_session_sizes'` order: the session takes exactly what `Need` counted.
        let mtp_gpu = e.mtp.map(|o| MtpGpu::new(e, &mut arena, o)).transpose()?;
        let part = arena.alloc(&e.dev, part)?;
        let mtp = e.mtp.map(|o| MtpState::new(o.rows, o.ring));
        if let (Some(m), Some(g)) = (&mtp, &mtp_gpu) {
            assert!(
                m.ring.cap() == g.ring_rows && m.rows.cap() as u64 * g.row_bytes <= g.k.size,
                "the MTP's rows and ring fit the buffers allocated for them"
            );
        }
        zero_recurrent(e, &layers)?;
        // A new session reuses the CPU path's sequence numbers: its flags start at zero.
        e.cpu_reset()?;
        Ok(Session {
            engine: e.id,
            ctx,
            pos: 0,
            tokens: Vec::new(),
            layers,
            part,
            taps: None,
            kv: KvRecord::default(),
            usable: true,
            kv_copy: None,
            copy_waits: 0,
            pending: None,
            fail_commit: false,
            rows_read: 0,
            mtp,
            mtp_gpu,
            fail_mtp_submit: false,
            capture: None,
            idle: e.dev.idle_handle(),
            arena,
        })
    }

    /// Starts a new sequence in the same allocations: the position and tokens go back to empty and the DeltaNet and
    /// conv state to zero. The KV caches are not cleared: attention reads only positions below `pos`, and a fresh
    /// session's aren't cleared either, so the KV record (`kv_tokens`) is kept: the KV still holds those tokens' values.
    /// The session is usable again. With the CPU miss path on, its mailbox's flags are zeroed. Equivalent to
    /// `e.session(self.ctx())` without the allocations, the budget check between sessions included.
    /// Refuses an engine other than the one that made the session.
    pub fn reset(&mut self, e: &Engine) -> Result<(), Error> {
        if self.engine != e.id {
            return Err(Error::WrongEngine);
        }
        assert!(!self.pending(), "reset with a verify step pending");
        zero_recurrent(e, &self.layers)?;
        // Position 0 again reuses the CPU path's sequence numbers: its flags go back to zero.
        e.cpu_reset()?;
        self.pos = 0;
        self.tokens.clear();
        self.usable = true;
        if let Some(m) = &mut self.mtp {
            m.on_reset();
        }
        if let Some((_, have)) = &mut self.capture {
            // Tests only: the captured window rows describe a session that no longer exists.
            have.fill(false);
        }
        e.check_budget()
    }

    /// Waits for the pending snapshot copy when a write at position `at` would overwrite KV it still reads (`at` below
    /// its end), counting the wait; clears the copy once the transfer queue has finished it. Writes at or above its end
    /// run on beside it. Called after a write's validation, so a refused call never waits.
    pub(crate) fn wait_copy(&mut self, e: &Engine, at: usize) -> Result<(), Error> {
        if let Some((end, v)) = self.kv_copy {
            if at < end {
                self.copy_waits += 1;
                e.dev.wait(vk::QueueKind::Transfer, v)?;
            }
            if e.dev.completed(vk::QueueKind::Transfer)? >= v {
                self.kv_copy = None;
            }
        }
        Ok(())
    }

    /// The tokens behind the KV the session holds, which can run ahead of `tokens` (after a `reset`, the KV is still
    /// valid for the tokens it held).
    pub fn kv_tokens(&self) -> &[u32] {
        self.kv.tokens()
    }

    /// Whether the session's state is defined: false after a write failed past validation, until `reset`.
    pub fn usable(&self) -> bool {
        self.usable
    }

    /// Whether a verify step has run and not yet been committed (`Session::verify`, `commit`). Every failure path of a
    /// step clears it before it returns, so it is set only between a successful `verify` and its `commit`.
    pub fn pending(&self) -> bool {
        self.pending.is_some()
    }

    /// A test hook: how many writes found a pending snapshot copy below their start and waited for it.
    pub fn copy_waits(&self) -> u64 {
        self.copy_waits
    }

    /// Each attention layer's K cache then V cache, layers in order (20 slices on the real model).
    pub fn kv_slices(&self) -> Vec<vk::Slice> {
        self.layers
            .iter()
            .filter_map(|st| match st {
                LayerState::Attention { kc, vc } => Some([*kc, *vc]),
                LayerState::DeltaNet { .. } => None,
            })
            .flatten()
            .collect()
    }

    /// Each DeltaNet layer's conv state then recurrent state, layers in order (60 slices on the real model).
    pub fn state_slices(&self) -> Vec<vk::Slice> {
        self.layers
            .iter()
            .filter_map(|st| match st {
                LayerState::DeltaNet { conv, s } => Some([*conv, *s]),
                LayerState::Attention { .. } => None,
            })
            .flatten()
            .collect()
    }

    /// After a restore (the caller has copied a snapshot's state and KV into the session's buffers): the session holds
    /// exactly `tokens`, its position is their count, and it is usable. The KV record is set to `tokens`, and the CPU
    /// path's flags go back to zero as `reset` does.
    pub fn set_restored(&mut self, e: &Engine, tokens: &[u32]) -> Result<(), Error> {
        if self.engine != e.id {
            return Err(Error::WrongEngine);
        }
        assert!(
            tokens.len() <= self.ctx,
            "a restore of {} tokens into a session of {}",
            tokens.len(),
            self.ctx
        );
        assert!(!self.pending(), "a restore with a verify step pending");
        e.cpu_reset()?;
        self.pos = tokens.len();
        self.tokens.clear();
        self.tokens.extend_from_slice(tokens);
        self.kv.set(tokens);
        self.usable = true;
        if let Some(m) = &mut self.mtp {
            m.on_reset();
        }
        if let Some((_, have)) = &mut self.capture {
            // Tests only: the captured window rows describe a session that no longer exists.
            have.fill(false);
        }
        Ok(())
    }

    /// Tests and `artemis warm` only: fills positions `[from, to)` of every KV buffer with the NaN word `0x7FC0_0000`,
    /// so a restore that skipped a KV copy cannot pass by luck. Waits for a pending snapshot copy whose end is above
    /// `from`, as a write from `from` does, then fills in one blocking compute submit. Nothing else changes: the
    /// record still names the tokens the session holds, so what it describes is no longer true of the KV until a
    /// restore or a write overwrites the range.
    pub fn poison_kv(&mut self, e: &Engine, from: usize, to: usize) -> Result<(), Error> {
        if self.engine != e.id {
            return Err(Error::WrongEngine);
        }
        assert!(
            from <= to && to <= self.ctx,
            "poison_kv({from}, {to}) on a session of {}",
            self.ctx
        );
        if from == to {
            return Ok(());
        }
        self.wait_copy(e, from)?;
        let b = (e.dims.n_head_kv * e.dims.head_dim * 2) as u64;
        let mut r = e.dev.record()?;
        for kv in self.kv_slices() {
            r.fill(kv.sub(from as u64 * b, (to - from) as u64 * b), 0x7FC0_0000);
        }
        e.dev.submit(r)?;
        Ok(())
    }

    /// The VRAM this session's state takes.
    pub fn vram_bytes(&self) -> u64 {
        self.arena.bytes()
    }

    /// Positions decoded so far.
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn ctx(&self) -> usize {
        self.ctx
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Reads the state back, as f32: for the check of prefill against decode, and diagnostics. Each layer in
    /// order gives (conv, recurrent) state if it is a DeltaNet layer, or (keys, values) for the positions so far,
    /// `[pos][n_head_kv × head_dim]`, if it is an attention layer. It downloads every buffer: slow, and only between
    /// steps.
    pub fn state(&self, e: &Engine) -> Result<SessionState, Error> {
        if self.engine != e.id {
            return Err(Error::WrongEngine);
        }
        let d = &e.dims;
        let f32s = |s: vk::Slice| -> Result<Vec<f32>, Error> {
            Ok(e.dev
                .download(s)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        };
        let f16s = |s: vk::Slice| -> Result<Vec<f32>, Error> {
            let n = (self.pos * d.n_head_kv * d.head_dim * 2) as u64;
            if n == 0 {
                return Ok(Vec::new());
            }
            Ok(e.dev
                .download(s.sub(0, n))?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| kernels::twin::f16_value(u16::from_le_bytes(*c)))
                .collect())
        };
        let layers = self
            .layers
            .iter()
            .map(|st| match st {
                LayerState::DeltaNet { conv, s } => Ok((f32s(*conv)?, f32s(*s)?)),
                LayerState::Attention { kc, vc } => Ok((f16s(*kc)?, f16s(*vc)?)),
            })
            .collect::<Result<_, Error>>()?;
        Ok(SessionState { layers })
    }

    /// The tap buffers, allocated on the first tapped step.
    pub(crate) fn tap_arenas(&mut self, e: &Engine) -> Result<&TapArenas, Error> {
        if self.taps.is_none() {
            let d = &e.dims;
            let bytes: u64 = expected_taps(d)
                .iter()
                .map(|t| 4 * t.per_pos(d) as u64)
                .sum();
            self.taps = Some(TapArenas {
                rb: e
                    .dev
                    .alloc(bytes, vk::MemKind::Readback, "the tap readback")?,
                force: e
                    .dev
                    .alloc(bytes, vk::MemKind::HostPinned, "the forced tap values")?,
            });
        }
        Ok(self.taps.as_ref().expect("allocated above"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen36::dims::Dims;

    #[test]
    fn the_write_rule_sorts_validation_errors_from_the_rest() {
        // A call refused by validation changes nothing; any other error leaves the session's state undefined, so its
        // KV record is cut back to the call's start and the session is unusable until reset.
        let refused = [
            Error::WrongEngine,
            Error::EmptyPrefill,
            Error::TokenOutOfRange {
                token: 9,
                n_vocab: 8,
            },
            Error::PrefillTooLong {
                pos: 4,
                tokens: 5,
                ctx: 8,
            },
            Error::ContextFull { ctx: 8 },
            Error::VerifyShape {
                tokens: 17,
                most: 16,
            },
            Error::VerifyBuffers {
                ctx: 262_144,
                need: 400 << 20,
                have: 225 << 20,
            },
            Error::MtpShape { k: 16, most: 15 },
        ];
        for err in &refused {
            assert!(is_validation(err), "{err:?}");
        }
        let past = [
            Error::NonFinite {
                pos: 3,
                index: 0,
                value: f32::NAN,
            },
            Error::BadCtx { ctx: 0 },
            Error::DoesNotFit {
                what: "x",
                need: 2,
                free: 1,
            },
            Error::Vk(vk::Error::Timeout),
        ];
        for err in &past {
            assert!(!is_validation(err), "{err:?}");
        }
    }

    #[test]
    fn session_bytes_counts_kv_recurrent_state_and_partials() {
        // At 65,536 positions: 10 attention layers' f16 K and V caches (64 MiB each), 30 DeltaNet layers' conv
        // (96 KiB) and recurrent (2 MiB) state, and the attention partials, packed with nothing to spare.
        assert_eq!(session_bytes(&Dims::QWEN36, 65_536), 1_412_268_032);
    }

    #[test]
    fn a_long_context_wastes_nothing_between_chunks() {
        // At 300,000 positions three K or V caches fill a chunk short of 1 GiB; the arena cuts each chunk to what it
        // holds, so the session takes its allocations' bytes exactly.
        let d = Dims::QWEN36;
        let sizes = session_sizes(&d, 300_000);
        assert_eq!(session_bytes(&d, 300_000), sizes.iter().sum::<u64>());
    }

    #[test]
    fn mtp_session_bytes_at_8192_rows() {
        // 16 MiB of K and V, the 2 MiB ring at 256 rows, and the small buffers, each rounded up to the arena's 256 bytes:
        // the carry row (8,192 B) and the pass's ids and addresses for 256 + 16 rows (1,088 B to 1,280, 2,176 B to 2,304).
        let d = Dims::QWEN36;
        assert_eq!(
            mtp_session_bytes(&d, 8192, 256),
            (16 << 20) + (2 << 20) + 8192 + 1280 + 2304
        );
        assert_eq!(mtp_session_bytes(&d, 8192, 256), 18_886_144);
        assert!(mtp_session_sizes(&d, 8192, 256)
            .iter()
            .all(|s| s % 256 == 0));
        // Not counted without the MTP: today's session, byte for byte.
        assert_eq!(session_bytes_mtp(&d, 4096, None), session_bytes(&d, 4096));
        assert_eq!(session_sizes_mtp(&d, 4096, None), session_sizes(&d, 4096));
        let o = MtpOptions {
            rows: 8192,
            ring: 256,
            window: 2048,
            head_rows: 0,
        };
        let with = session_sizes_mtp(&d, 4096, Some(o));
        // The MTP's six buffers go before the partials: the rest is today's session.
        let base = session_sizes(&d, 4096);
        let n = base.len();
        assert_eq!(&with[..n - 1], &base[..n - 1]);
        assert_eq!(with[with.len() - 1], base[n - 1]);
        assert_eq!(
            with[n - 1..with.len() - 1].iter().sum::<u64>(),
            mtp_session_bytes(&d, 8192, 256)
        );
        assert!(session_bytes_mtp(&d, 4096, Some(o)) > session_bytes(&d, 4096) + (18 << 20));
    }

    #[test]
    fn the_mtp_bytes_are_exactly_the_difference_to_a_session_without_them_at_any_context() {
        // The tiny model, 512 rows and a ring of 128: K and V 64 KiB each, the ring 128 KiB, the carry 1 KiB, the pass's ids
        // (144 rows: 576 B) and addresses (1,152 B) rounded up to 768 and 1,280.
        let d = Dims::tiny();
        let o = MtpOptions {
            rows: 512,
            ring: 128,
            window: 256,
            head_rows: 0,
        };
        assert_eq!(
            mtp_session_sizes(&d, 512, 128),
            [65_536, 65_536, 131_072, 1024, 768, 1280]
        );
        assert_eq!(mtp_session_bytes(&d, 512, 128), 265_216);
        // Whatever the context (so whatever the partials' size and the chunks), the arena takes exactly that more.
        for ctx in [1, 2, 63, 512, 4096, 300_000, 1 << 20] {
            assert_eq!(
                session_bytes_mtp(&d, ctx, Some(o)) - session_bytes(&d, ctx),
                265_216,
                "ctx {ctx}"
            );
        }
        let real = Dims::QWEN36;
        let o = MtpOptions {
            rows: 8192,
            ring: 256,
            window: 2048,
            head_rows: 0,
        };
        for ctx in [4096, 65_536] {
            assert_eq!(
                session_bytes_mtp(&real, ctx, Some(o)) - session_bytes(&real, ctx),
                mtp_session_bytes(&real, 8192, 256),
                "ctx {ctx}"
            );
        }
    }

    #[test]
    fn mtp_options_clamp_the_ring_with_the_rows() {
        // The full head, so the sizes are what is compared; the default config carries the 65,536-row head.
        let cfg = crate::SpecConfig {
            mtp_head_rows: 0,
            ..crate::SpecConfig::default()
        };
        let at = |ctx| MtpOptions::for_session(&cfg, ctx).unwrap().unwrap();
        assert_eq!(
            MtpOptions::for_session(&crate::SpecConfig::default(), 4096)
                .unwrap()
                .unwrap()
                .head_rows,
            65_536
        );
        // The defaults (mtp_ctx 8,192, mtp_ring 256, mtp_window 2,048) on a long session.
        assert_eq!(
            at(65_536),
            MtpOptions {
                rows: 8192,
                ring: 256,
                window: 2048,
                head_rows: 0,
            }
        );
        // ctx 1,024: rows 1,024, the ring 256 (a quarter of the rows), the window 512 (half).
        assert_eq!(
            at(1024),
            MtpOptions {
                rows: 1024,
                ring: 256,
                window: 512,
                head_rows: 0,
            }
        );
        // ctx 600: rows 600; a quarter is 150, rounded down to 144; the window is 300.
        assert_eq!(
            at(600),
            MtpOptions {
                rows: 600,
                ring: 144,
                window: 300,
                head_rows: 0,
            }
        );
        // make_room's arithmetic: a pass of a full ring plus a verify's 16 rows leaves room for a slide.
        for ctx in [512, 513, 600, 1024, 4096, 65_536] {
            let o = at(ctx);
            assert!(o.rows <= ctx && o.ring % 16 == 0 && o.ring >= 16, "{ctx}");
            assert!(o.ring + kernels::tokens::MAX_TOKENS <= o.rows / 2, "{ctx}");
            assert!(o.window >= 1 && o.window <= o.rows / 2, "{ctx}");
        }
    }

    #[test]
    fn a_context_under_512_refuses_mtp() {
        let cfg = crate::SpecConfig::default();
        for ctx in [0, 1, 511] {
            match MtpOptions::for_session(&cfg, ctx) {
                Err(Error::SpeculationRefused { why }) => {
                    assert!(
                        why.starts_with("MTP needs a session of at least 512 positions"),
                        "{why}"
                    );
                    assert!(why.contains("speculation.sources"), "{why}");
                }
                other => panic!("ctx {ctx}: expected SpeculationRefused, got {other:?}"),
            }
        }
        assert!(MtpOptions::for_session(&cfg, 512).unwrap().is_some());
    }

    #[test]
    fn no_mtp_options_without_the_mtp_source_or_with_speculation_off() {
        let cfg = crate::SpecConfig {
            sources: vec![crate::SpecSource::Lookup],
            ..crate::SpecConfig::default()
        };
        // Not even a short context is refused: nothing asks for the MTP.
        assert_eq!(MtpOptions::for_session(&cfg, 100).unwrap(), None);
        assert_eq!(MtpOptions::for_session(&cfg, 65_536).unwrap(), None);
        let off = crate::SpecConfig {
            enabled: false,
            ..crate::SpecConfig::default()
        };
        assert_eq!(MtpOptions::for_session(&off, 100).unwrap(), None);
        assert_eq!(MtpOptions::for_session(&off, 65_536).unwrap(), None);
    }
}
