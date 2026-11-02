//! The CPU miss path: which experts the CPU serves and the slot table that says so (pure functions of the
//! engine's mirror and the share, so the policy tests without a GPU), and `CpuPath`, the mailbox and the coordinator
//! thread that answers it while decode's submit runs.
use crate::{base::BaseTable, Error};
use cpu_expert::{expert_rows, with_ftz, Pool, XPrep};
use kernels::dispatch::{mbox, slot_entry, SlotLayout, WType, ENTRY_BYTES, WHERE_CPU, WHERE_HOST};
use kernels::Kernels;
use std::sync::atomic::{fence, AtomicBool, AtomicU32, AtomicU64, Ordering::*};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub use cpu_expert::pin_set;

/// The mailbox's sequence number for (token, layer): `token * 64 + layer + 1`, never 0 (context 65,536 fits a u32).
pub fn seq_of(token: u32, layer: usize) -> u32 {
    token.wrapping_mul(64).wrapping_add(layer as u32 + 1)
}

/// The wait's spin bound: 4,000,000 polls, beside the 20 ms clock bound.
pub const MAX_SPINS: u32 = 4_000_000;

/// The share a running path holds, as its f32 bits in an `AtomicU32`: `set_cpu_share` stores it, and each step reads
/// it once (`step_table`), so a share set between tokens is the next token's. The bits, not thousandths: the share is
/// read back exactly as set, so `share_q8` of it is `share_q8` of what was asked.
pub struct ShareCell(AtomicU32);

impl ShareCell {
    pub fn new(share: f32) -> ShareCell {
        ShareCell(AtomicU32::new(share.to_bits()))
    }
    pub fn set(&self, share: f32) {
        self.0.store(share.to_bits(), Release);
    }
    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Acquire))
    }
}

/// The table one decode step's expert kernels read with the path on: `overlay` of `mirror` (which already holds the
/// step's staged changes) at the share `cell` holds now, `host` being each main-layer expert's host copy (device
/// address, `[layer * n_expert + expert]`).
pub fn step_table(
    mirror: &[(u64, u32)],
    n_layer: usize,
    n_expert: usize,
    host: &[u64],
    cell: &ShareCell,
) -> Vec<u8> {
    overlay(mirror, n_layer, n_expert, host, share_q8(cell.get()))
}

/// Zeroes the mailbox's flag area, bytes `0..mbox::X0` (every publish and done flag, and the status words), then
/// `fence(SeqCst)`: a session reset or a new session reuses sequence numbers, and a leftover flag would satisfy them.
///
/// # Safety
/// `ptr` must be valid for writes of `mbox::X0` bytes, and no GPU work may be using the mailbox.
pub unsafe fn clear_flags(ptr: *mut u8) {
    unsafe { std::ptr::write_bytes(ptr, 0, mbox::X0) };
    fence(SeqCst);
}

/// Zeroes the wait's two status words at `mbox::STATUS_BYTE` (the token's timeout and skip counts), then
/// `fence(SeqCst)`. Called after every step: the wait's no-spin is per token.
///
/// # Safety
/// `ptr` must be valid for writes of `mbox::STATUS_BYTE + 8` bytes, and no GPU work may be using the mailbox.
pub unsafe fn clear_status(ptr: *mut u8) {
    unsafe { std::ptr::write_bytes(ptr.add(mbox::STATUS_BYTE), 0, 8) };
    fence(SeqCst);
}

/// The announce cell's token for "no token": the coordinator acknowledges it as idle (`CpuPath::quiesce`).
const IDLE: u32 = u32::MAX;

/// What the engine thread and the coordinator share.
struct Ctl {
    stop: AtomicBool,
    /// Bumped (Release) after `token` is stored, once per announce.
    gen: AtomicU32,
    token: AtomicU32,
    /// The last `gen` the coordinator acknowledged with `token == IDLE`.
    idle: AtomicU32,
    share: ShareCell,
    /// Tests only: skip every announced token (answer nothing).
    fail: AtomicBool,
    /// Tests only: answer the next token's layer 0 only once the GPU's wait has timed out.
    stall0: AtomicBool,
    /// Tests only: the stall hook saw the GPU's timeout (status word 0 non-zero) within its window and answered.
    late: AtomicBool,
    /// Routed experts the coordinator computed (mask bits), not yet taken into `Stats::cpu_misses`.
    computed: AtomicU64,
    /// Layers the coordinator gave up on after 2 s without their publish (a host-side bound).
    host_timeouts: AtomicU64,
}

/// What the coordinator needs, by value: it never borrows the engine.
struct Plan {
    base: usize,
    cpus: Vec<usize>,
    lays: Vec<SlotLayout>,
    /// `[layer][expert]`: each host copy's CPU address.
    slots: Vec<Vec<usize>>,
    n_embd: usize,
    k: usize,
}

/// The CPU miss path while the share runs it (`path_on`: at least 1/512): the 8 MiB `AmdUncached`
/// mailbox, the overlay's host/VRAM table pair, the wait's calibrated clock bound, and the coordinator thread (worker 0
/// of the pool, pinned inside the process mask, workers at `TIME_CRITICAL`). Dropping it stops and joins the
/// coordinator before the mailbox is freed.
pub(crate) struct CpuPath {
    ctl: Arc<Ctl>,
    coord: Option<std::thread::JoinHandle<()>>,
    /// The overlay's own table: written from the mirror each step, copied into VRAM by the step.
    pub table: BaseTable,
    /// 20 ms of `clockRealtimeEXT`, calibrated at start.
    pub max_ticks: u32,
    /// Worker 0's SMT sibling, when it is inside the process mask: the engine thread's CPU for a submit.
    sibling: Option<usize>,
    /// The position the last step decoded since the flags were last zeroed (`before_step`).
    last_pos: std::cell::Cell<Option<usize>>,
    /// The last step's skip count (the second status word, read before `clear_status`): tests and diagnostics.
    last_skips: std::cell::Cell<u32>,
    /// A step's submit failed, so the GPU may still be running it: the next `before_step` resets (`reset`).
    dirty: std::cell::Cell<bool>,
    /// `[layer * n_expert + expert]`: each main-layer expert's host copy, as the device address the slot table holds.
    /// The overlay turns a HOST entry CPU only at this address, the copy the coordinator computes from.
    pub host: Vec<u64>,
    /// The coordinator did not stop within `SHUTDOWN_BOUND` and was detached: the mailbox is leaked, not freed.
    detached: bool,
    /// Freed by `Drop` only once the coordinator is joined; leaked if it was detached (`shutdown`).
    mbox: std::mem::ManuallyDrop<vk::Buffer>,
}

/// How long `shutdown` waits for the coordinator to stop before it detaches it. Every coordinator loop checks stop,
/// so a healthy one stops within one layer's compute; this bounds only a wedged one.
const SHUTDOWN_BOUND: Duration = Duration::from_secs(5);

impl CpuPath {
    /// Allocates and zeroes the mailbox, calibrates the wait's 20 ms bound with two `stamp` dispatches 300 ms apart,
    /// and starts the coordinator on `workers(process_affinity_mask())`. `slots[l][e]` is expert `e`'s host copy in
    /// layer `l` (CPU address), `host[l * n_expert + e]` the same copy's device address, `lays[l]` its layout; the
    /// copies must outlive the path. The caller has checked the share and the machine (`check_share`, `check_path`);
    /// a mask that changed since is refused here as there.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        dev: &vk::Device,
        k: &Kernels,
        n_layer: usize,
        n_expert: usize,
        n_embd: usize,
        n_used: usize,
        lays: Vec<SlotLayout>,
        slots: Vec<Vec<usize>>,
        host: Vec<u64>,
        share: f32,
    ) -> Result<CpuPath, Error> {
        assert!(
            n_layer < mbox::MAX_LAYERS,
            "n_layer + 1 = {} > 48: the status words sit on layer 48's publish-flag line",
            n_layer + 1
        );
        assert!(n_embd <= mbox::N_EMBD && n_used <= mbox::MAX_E);
        assert!(
            lays.len() == n_layer && slots.len() == n_layer && host.len() == n_layer * n_expert
        );
        let mask = cpu_expert::process_affinity_mask();
        let cpus = workers(mask).map_err(|what| Error::CpuPathUnavailable { what })?;
        let mbox = dev.alloc(
            8 << 20,
            vk::MemKind::AmdUncached,
            "the CPU miss path's mailbox",
        )?;
        mbox.write(0, &vec![0u8; 8 << 20]);
        fence(SeqCst);
        // The wait's bound in device-clock ticks: two stamps 300 ms apart, scaled to 20 ms.
        let tim = dev.alloc(16, vk::MemKind::Readback, "the miss path's clock stamps")?;
        let stamp = |i: usize| -> Result<(), Error> {
            let mut r = dev.record()?;
            k.stamp(&mut r, tim.addr(), i);
            r.host_barrier();
            dev.submit(r)?;
            Ok(())
        };
        stamp(0)?;
        std::thread::sleep(Duration::from_millis(300));
        stamp(1)?;
        let mut b = [0u8; 16];
        tim.read(0, &mut b);
        let w = |i: usize| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap());
        let max_ticks = (w(1).wrapping_sub(w(0)) as f64 / 15.0).min(u32::MAX as f64) as u32;
        assert!(max_ticks > 0, "the device clock did not advance in 300 ms");
        let table = BaseTable::new(dev, n_layer, n_expert)?;
        let sibling = Some(cpus[0] + 1).filter(|&s| s < 64 && mask >> s & 1 == 1);
        let ctl = Arc::new(Ctl {
            stop: AtomicBool::new(false),
            gen: AtomicU32::new(0),
            token: AtomicU32::new(IDLE),
            idle: AtomicU32::new(0),
            share: ShareCell::new(share),
            fail: AtomicBool::new(false),
            stall0: AtomicBool::new(false),
            late: AtomicBool::new(false),
            computed: AtomicU64::new(0),
            host_timeouts: AtomicU64::new(0),
        });
        let plan = Plan {
            base: mbox.ptr() as usize,
            cpus,
            lays,
            slots,
            n_embd,
            k: n_used,
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let c = ctl.clone();
        let coord = std::thread::Builder::new()
            .name("cpu-miss coordinator".into())
            .spawn(move || coordinate(&c, &plan, tx))
            .map_err(|_| Error::CpuPathUnavailable {
                what: "a coordinator thread",
            })?;
        // The pool is built (and pinned, and at TIME_CRITICAL) before the first token is announced. A coordinator that
        // ended before saying so (its pool panicked), or a pool Windows would not pin, refuses the path: either way
        // the coordinator has returned (a refused pool drops first, joining its workers) and is joined here, so
        // nothing is left running and nothing reads the mailbox when it drops.
        let refused = match rx.recv() {
            Ok(true) => None,
            Ok(false) => Some("a pinned expert pool (Windows refused a worker's CPU)"),
            Err(_) => Some("a running coordinator (it ended before its pool was ready)"),
        };
        if let Some(what) = refused {
            let _ = coord.join();
            return Err(Error::CpuPathUnavailable { what });
        }
        Ok(CpuPath {
            ctl,
            coord: Some(coord),
            table,
            max_ticks,
            sibling,
            last_pos: std::cell::Cell::new(None),
            last_skips: std::cell::Cell::new(0),
            dirty: std::cell::Cell::new(false),
            host,
            detached: false,
            mbox: std::mem::ManuallyDrop::new(mbox),
        })
    }

    /// The share the step reads (once per step).
    pub fn share(&self) -> f32 {
        self.ctl.share.get()
    }

    pub fn set_share(&self, share: f32) {
        self.ctl.share.set(share);
    }

    /// The share cell itself, for `step_table`.
    pub fn share_cell(&self) -> &ShareCell {
        &self.ctl.share
    }

    /// The mailbox's device address.
    pub fn mbox_addr(&self) -> u64 {
        self.mbox.addr()
    }

    /// Tells the coordinator which token the next submit decodes (before the submit): it answers layer `l` at
    /// `seq_of(token, l)`, and abandons whatever token it was on.
    pub fn announce(&self, token: u32) {
        self.ctl.token.store(token, Relaxed);
        self.ctl.gen.fetch_add(1, Release);
    }

    /// Makes the coordinator abandon its token and waits until it says it is idle, so no write of an earlier token
    /// can land after this returns.
    fn quiesce(&self) {
        self.ctl.token.store(IDLE, Relaxed);
        let g = self.ctl.gen.fetch_add(1, Release).wrapping_add(1);
        let t0 = Instant::now();
        while self.ctl.idle.load(Acquire) != g {
            if self.coord.as_ref().is_none_or(|h| h.is_finished()) {
                return;
            }
            assert!(
                t0.elapsed() < Duration::from_secs(10),
                "the CPU miss path's coordinator did not go idle in 10 s"
            );
            std::hint::spin_loop();
        }
    }

    /// A session reset or a new session: the coordinator idle, the compute queue idle, then every flag zeroed
    /// (`clear_flags`), so a leftover done flag can't satisfy the new session's sequence numbers. Between steps only.
    /// Fails, leaving the flags as they were, if the queue's wait does.
    pub fn reset(&self, dev: &vk::Device) -> Result<(), vk::Error> {
        self.quiesce();
        // A step whose submit failed (its wait timed out) may still be running and reading the mailbox: wait for the
        // compute queue, as `set_cpu_share(0)` does before the mailbox drops.
        dev.wait(
            vk::QueueKind::Compute,
            dev.submitted(vk::QueueKind::Compute),
        )?;
        // SAFETY: the mailbox is 8 MiB and mapped; the compute queue is idle (just waited, and only this thread
        // submits) and the coordinator is idle (`quiesce`), so nothing else uses the flags.
        unsafe { clear_flags(self.mbox.ptr()) };
        self.last_pos.set(None);
        self.dirty.set(false);
        Ok(())
    }

    /// Before a decode step at `pos`: a position at or below the last step's means another session or a reset
    /// (one session's positions only grow), whose sequence numbers may repeat ones the flags still hold, so the flags
    /// are zeroed first (`reset`). `Session::reset` and a new session reset already; this also covers two sessions
    /// decoding in turn. So does a previous step whose submit failed (`after_step`): its work may still be running,
    /// and `reset` waits for the queue before it zeroes the flags and the status words.
    pub fn before_step(&self, dev: &vk::Device, pos: usize) -> Result<(), vk::Error> {
        if self.dirty.get() || self.last_pos.get().is_some_and(|p| pos <= p) {
            self.reset(dev)?;
        }
        self.last_pos.set(Some(pos));
        Ok(())
    }

    /// After a step's submit returned (`submitted`: whether it succeeded): (this token's GPU timeouts, the experts
    /// the coordinator computed since the last call), then the status words zeroed for the next token
    /// (`clear_status`). The skip count is kept for `last_skips`. After a failed submit the GPU may still be running
    /// the step and writing the status words: they are left alone (the token's timeouts count as 0), and the path is
    /// marked for the next `before_step` to reset, which waits for the queue first.
    pub fn after_step(&self, submitted: bool) -> (u64, u64) {
        let computed = self.ctl.computed.swap(0, AcqRel);
        if !submitted {
            self.dirty.set(true);
            self.last_skips.set(0);
            return (0, computed);
        }
        let p = self.mbox.ptr();
        // SAFETY: the status words lie inside the mapped mailbox, 4-byte aligned; the submit succeeded, so the step's
        // work has finished, and no other GPU work uses the mailbox (only this thread submits).
        let (timeouts, skips) = unsafe {
            (
                std::ptr::read_volatile(p.add(mbox::STATUS_BYTE) as *const u32),
                std::ptr::read_volatile(p.add(mbox::STATUS_BYTE + 4) as *const u32),
            )
        };
        self.last_skips.set(skips);
        // SAFETY: as above.
        unsafe { clear_status(p) };
        (timeouts as u64, computed)
    }

    /// The last step's skip count: its waits that did not spin because an earlier layer of the token had timed out.
    pub fn last_skips(&self) -> u32 {
        self.last_skips.get()
    }

    /// Experts computed and not yet taken (for the shutdown's last count).
    pub fn take_computed(&self) -> u64 {
        self.ctl.computed.swap(0, AcqRel)
    }

    /// Layer `layer`'s done flag: tests and diagnostics.
    pub fn done_flag(&self, layer: usize) -> u32 {
        assert!(layer < mbox::MAX_LAYERS, "layer {layer} has no done flag");
        let w = self.mbox.ptr() as usize + 4 * mbox::done_word(layer);
        // SAFETY: the word lies inside the mailbox's flag area (`done_word(layer) < X0 / 4` for layer < MAX_LAYERS),
        // which stays mapped while `self` lives, and is 4-byte aligned (the mapping is); `AtomicU32` has `u32`'s
        // layout, and the GPU and the coordinator touch the flag only atomically.
        unsafe { (*(w as *const AtomicU32)).load(Acquire) }
    }

    /// Layers the coordinator gave up on (2 s with no publish).
    pub fn host_timeouts(&self) -> u64 {
        self.ctl.host_timeouts.load(Relaxed)
    }

    /// Tests only: while true, the coordinator answers no token (a dead pool).
    pub fn fail_next(&self, fail: bool) {
        self.ctl.fail.store(fail, Release);
    }

    /// Tests only: the next token's layer 0 is answered late, once the GPU's wait has counted its timeout; every
    /// later layer of that token is answered as usual.
    pub fn stall_layer0_once(&self) {
        self.ctl.late.store(false, Release);
        self.ctl.stall0.store(true, Release);
    }

    /// Tests only: whether the last `stall_layer0_once` saw the GPU's wait time out within its 2 s window and then
    /// answered layer 0. False if it missed the window, or has not answered yet.
    pub fn late_answered(&self) -> bool {
        self.ctl.late.load(Acquire)
    }

    /// Pins the calling (engine) thread to worker 0's SMT sibling at `HIGHEST` for one submit; the guard restores
    /// its priority and unpins it. A sibling outside the process mask leaves the thread unpinned.
    pub fn for_submit(&self) -> SubmitGuard {
        let old = cpu_expert::pool::priority();
        cpu_expert::pool::set_priority(cpu_expert::pool::HIGHEST);
        let pinned = self.sibling.is_some_and(cpu_expert::pool::pin);
        SubmitGuard { old, pinned }
    }

    /// Stops the coordinator and joins it, waiting at most `SHUTDOWN_BOUND` (idempotent). True once it is joined.
    /// False if it did not stop in time (a wedged pool; a panicking one ends the thread): it is detached, logged, and may still read the
    /// mailbox and the host copies, so the mailbox is leaked when the path drops and the caller must leak the host
    /// copies (`HostCopies::leak_experts`). A leaked buffer keeps the device alive, so its mapping stays valid.
    /// Leaking is chosen over waiting forever: `Engine::drop` must end, and over freeing: a live thread may touch it.
    pub fn shutdown(&mut self) -> bool {
        self.ctl.stop.store(true, Release);
        let Some(h) = self.coord.take() else {
            return !self.detached;
        };
        let t0 = Instant::now();
        while !h.is_finished() {
            if t0.elapsed() > SHUTDOWN_BOUND {
                eprintln!(
                    "artemis: the CPU miss path's coordinator did not stop in {} s; detaching it and leaking the \
                     mailbox and the host expert copies it may still read",
                    SHUTDOWN_BOUND.as_secs()
                );
                self.detached = true;
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        // Finished, so this returns at once; a coordinator that panicked has nothing more to say.
        let _ = h.join();
        true
    }
}

impl Drop for CpuPath {
    fn drop(&mut self) {
        if self.shutdown() {
            // SAFETY: the coordinator is joined, so nothing reads the mailbox through `Plan::base`; the owner waited
            // for the compute queue before dropping the path; and this is the one drop of the field.
            unsafe { std::mem::ManuallyDrop::drop(&mut self.mbox) };
        }
    }
}

/// The engine thread's pin and priority for one submit; dropping it restores both.
pub(crate) struct SubmitGuard {
    old: i32,
    pinned: bool,
}

impl Drop for SubmitGuard {
    fn drop(&mut self) {
        if self.pinned {
            cpu_expert::pool::unpin();
        }
        cpu_expert::pool::set_priority(self.old);
    }
}

/// The mailbox word `w` as an atomic.
fn word(base: usize, w: usize) -> &'static AtomicU32 {
    // SAFETY: the mailbox outlives the coordinator (it is joined before the buffer drops, or the buffer is leaked if it
    // was detached: `CpuPath::shutdown`), and every flag word is 4-byte aligned inside it.
    unsafe { &*((base + 4 * w) as *const AtomicU32) }
}

/// The coordinator: builds the pool on this thread (worker 0) and the workers' priority, says whether the pool is
/// pinned (returning if not), then answers each announced token's layers in order until stopped.
fn coordinate(ctl: &Ctl, p: &Plan, ready: std::sync::mpsc::Sender<bool>) {
    let cpus: Vec<Option<usize>> = p.cpus.iter().map(|&c| Some(c)).collect();
    let pool = Pool::new(&cpus);
    pool.run(&|_| {
        cpu_expert::pool::set_priority(cpu_expert::pool::TIME_CRITICAL);
    });
    // A pool Windows would not pin refuses the path (`CpuPath::start`); returning drops it, joining its workers.
    let _ = ready.send(pool.pinned);
    if !pool.pinned {
        return;
    }
    let n_ff = p.lays.iter().map(|l| l.n_ff).max().unwrap_or(0);
    let (mut xs, mut x) = (XPrep::default(), vec![0f32; p.n_embd]);
    let (mut act, mut y) = (vec![0f32; p.k * n_ff], vec![0f32; p.k * p.n_embd]);
    let mut seen = ctl.gen.load(Acquire);
    loop {
        // The next announce.
        let g = loop {
            if ctl.stop.load(Acquire) {
                return;
            }
            let g = ctl.gen.load(Acquire);
            if g != seen {
                break g;
            }
            std::hint::spin_loop();
        };
        seen = g;
        let token = ctl.token.load(Relaxed);
        if token == IDLE {
            ctl.idle.store(g, Release);
            continue;
        }
        if ctl.fail.load(Acquire) {
            continue;
        }
        'layers: for (l, lay) in p.lays.iter().enumerate() {
            // A newer announce means the GPU has moved past this token (or the engine went idle): no wait will read
            // this token's remaining layers, and the next token's publish may be overwriting their x, so drop them.
            if ctl.gen.load(Acquire) != seen {
                break 'layers;
            }
            let seq = seq_of(token, l);
            let t0 = Instant::now();
            while word(p.base, mbox::pub_word(l)).load(Acquire) != seq {
                if ctl.stop.load(Acquire) {
                    return;
                }
                if ctl.gen.load(Acquire) != seen {
                    break 'layers; // a newer announce: abandon this token
                }
                if t0.elapsed() > Duration::from_secs(2) {
                    ctl.host_timeouts.fetch_add(1, Relaxed);
                    break 'layers;
                }
                std::hint::spin_loop();
            }
            let hdr = p.base + mbox::X0 + l * mbox::XSTRIDE;
            // SAFETY: `off < 256`, so `hdr + off` lies inside this layer's header in the mapped mailbox (which outlives
            // the coordinator), 4-byte aligned; publish wrote the header before its flag, which the Acquire load above
            // saw. Volatile: the GPU writes this memory behind the compiler's back.
            let rd = |off: usize| unsafe { std::ptr::read_volatile((hdr + off) as *const u32) };
            let mask = rd(12);
            if mask == 0 {
                continue; // the wait neither spins nor copies
            }
            let n_expert = p.slots[l].len();
            let mut sel = Vec::with_capacity(p.k);
            for j in 0..p.k {
                if mask >> j & 1 == 1 {
                    let e = f32::from_bits(rd(32 + 4 * j)) as usize;
                    if e >= n_expert {
                        // An id past the table: the layer goes unanswered, so its wait times out and the
                        // serve-2 fallback computes it.
                        continue 'layers;
                    }
                    sel.push((j, e));
                }
            }
            // SAFETY: x lies inside the mailbox's X area for this layer; publish wrote it before its flag (Acquire).
            unsafe {
                std::ptr::copy_nonoverlapping((hdr + 256) as *const f32, x.as_mut_ptr(), p.n_embd)
            };
            let mut late = false;
            if l == 0 && ctl.stall0.swap(false, AcqRel) {
                // Tests only: a stall past the wait's bound, then the answer anyway. `late` is what the loop saw,
                // not a re-read: after_step may zero the status between the loop's exit and a second load.
                let st = word(p.base, mbox::STATUS_BYTE / 4);
                let s0 = Instant::now();
                while s0.elapsed() < Duration::from_secs(2) {
                    if st.load(Acquire) != 0 {
                        late = true;
                        break;
                    }
                    if ctl.stop.load(Acquire) {
                        return;
                    }
                    if ctl.gen.load(Acquire) != seen {
                        break 'layers; // a newer announce: abandon this token
                    }
                    std::hint::spin_loop();
                }
            }
            let bytes = lay.bytes();
            let owned: Vec<&[u8]> = sel
                .iter()
                // SAFETY: each host copy is `bytes` long, mapped, and outlives the coordinator.
                .map(|&(_, e)| unsafe {
                    std::slice::from_raw_parts(p.slots[l][e] as *const u8, bytes)
                })
                .collect();
            let n = owned.len();
            // SAFETY: `check_path` refused a CPU without AVX-512. Built under FTZ+DAZ, as every pooled job runs.
            with_ftz(|| unsafe { xs.build(&x, &[lay.gate, lay.up]) });
            expert_rows(
                &pool,
                &owned,
                lay,
                &xs,
                &mut act[..n * lay.n_ff],
                &mut y[..n * p.n_embd],
            );
            let yl = p.base + mbox::Y0 + l * mbox::YSTRIDE;
            for (i, &(j, _)) in sel.iter().enumerate() {
                // SAFETY: y[j] lies inside this layer's Y area (j < 8, n_embd <= 2048).
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        y[i * p.n_embd..].as_ptr(),
                        (yl + 4 * j * p.n_embd) as *mut f32,
                        p.n_embd,
                    )
                };
            }
            fence(SeqCst);
            word(p.base, mbox::done_word(l)).store(seq, Release);
            fence(SeqCst);
            if late {
                ctl.late.store(true, Release);
            }
            ctl.computed.fetch_add(n as u64, Relaxed);
        }
    }
}

/// The share as a count of 256ths: `(share * 256).round()`, so 0.0 is 0, 0.5 is 128 and 1.0 is 256. A share below
/// 1/512 is 0.
pub fn share_q8(share: f32) -> u32 {
    (share * 256.0).round() as u32
}

/// Whether `share` runs the CPU path: `share_q8(share) > 0`. A share in (0, 1/512) assigns no expert, so it is taken
/// as 0: no path, so no publish, wait or serve-2 dispatch.
pub fn path_on(share: f32) -> bool {
    share_q8(share) > 0
}

/// Whether the CPU serves `expert` at a share of `q8` 256ths. 17 is coprime to 256, so `expert * 17 mod 256` visits
/// every residue once in 256 consecutive experts: q8 256 assigns every expert, q8 0 none, and q8 of them exactly q8 in
/// each 256. The same expert is the CPU's in every layer.
pub fn assigns_cpu(expert: u32, q8: u32) -> bool {
    (expert.wrapping_mul(17) % 256) < q8
}

/// The slot table's bytes for decode with the CPU path on: `mirror` (`(addr, where)` per entry, `n_layer + 1` layers of
/// `n_expert`, the last the MTP block) with each main layer's HOST entry whose expert `assigns_cpu` at `q8`, and whose
/// address is that expert's own host copy (`host[layer * n_expert + expert]`, the copy the coordinator computes from),
/// turned `WHERE_CPU` at the same address (slot size is unchanged, so the fallback's loads stay 16-byte aligned). A HOST
/// entry pointing anywhere else, VRAM entries, any other kind and the MTP block are copied through, and q8 0 is
/// `mirror` unchanged. Prefill reads the real table, never this one.
pub fn overlay(
    mirror: &[(u64, u32)],
    n_layer: usize,
    n_expert: usize,
    host: &[u64],
    q8: u32,
) -> Vec<u8> {
    assert_eq!(mirror.len(), (n_layer + 1) * n_expert, "a whole slot table");
    assert_eq!(
        host.len(),
        n_layer * n_expert,
        "a host copy per main-layer expert"
    );
    let mut t = Vec::with_capacity(mirror.len() * ENTRY_BYTES);
    for (i, &(addr, loc)) in mirror.iter().enumerate() {
        let (layer, expert) = (i / n_expert, i % n_expert);
        let loc = if layer < n_layer
            && loc == WHERE_HOST
            && addr == host[i]
            && assigns_cpu(expert as u32, q8)
        {
            WHERE_CPU
        } else {
            loc
        };
        t.extend(slot_entry(addr, loc));
    }
    t
}

/// Refuses a share outside `0.0..=1.0` (or NaN), then the base time with a share that runs the path (`path_on`; a
/// share below 1/512 is 0). Pure, and called by `Engine::open` before the model loads.
pub fn check_share(share: f32, base_time: bool) -> Result<(), Error> {
    if !(0.0..=1.0).contains(&share) {
        return Err(Error::BadCpuShare { share });
    }
    if base_time && path_on(share) {
        return Err(Error::BaseTimeAndCpuShare);
    }
    Ok(())
}

/// With a share that runs the path (`path_on`), refuses an expert matrix type the CPU worker has no kernel for: only
/// Q4_K, Q5_K, Q6_K and Q8_0 have one (`cpu_expert::avx`), and a worker that met F32 or BF16 would panic, leaving every
/// layer to fall back through the GPU's timeouts. Names the first such type of `types`. Pure, and called by
/// `Engine::open` before the model loads (and by `set_cpu_share`).
pub fn check_expert_types(share: f32, types: &[WType]) -> Result<(), Error> {
    if !path_on(share) {
        return Ok(());
    }
    match types
        .iter()
        .find(|t| !matches!(t, WType::Q4K | WType::Q5K | WType::Q6K | WType::Q8_0))
    {
        Some(t) => Err(Error::CpuShareExpertType { ty: t.name() }),
        None => Ok(()),
    }
}

/// What this machine offers the CPU path (`probe`), as plain flags so the refusals test without a GPU.
#[derive(Clone, Copy, Debug)]
pub struct Path {
    pub amd_uncached: bool,
    pub shader_clock: bool,
    pub avx512: bool,
    /// The process affinity mask (`process_affinity_mask`; 0 off Windows), which `workers` must accept.
    pub mask: u64,
}

impl Path {
    pub fn probe(dev: &vk::Device) -> Path {
        Path {
            amd_uncached: dev.has_kind(vk::MemKind::AmdUncached),
            shader_clock: dev.shader_clock,
            avx512: cpu_expert::available(),
            mask: cpu_expert::process_affinity_mask(),
        }
    }
}

/// The workers' CPUs for process mask `mask`: `pin_set(mask)`, refused (naming what is missing) when the mask has no
/// even CPU, or when the workers would take every CPU of it. Each worker spins at `TIME_CRITICAL`, so the engine
/// thread needs a CPU of the mask besides theirs (`spare_cpu`); without one a token takes seconds and every wait times
/// out. Pure.
pub fn workers(mask: u64) -> Result<Vec<usize>, &'static str> {
    let pins = pin_set(mask).ok_or("an even CPU in the process affinity mask")?;
    if !cpu_expert::spare_cpu(mask, &pins) {
        return Err("a CPU in the process mask besides the workers'");
    }
    Ok(pins)
}

/// With a share that runs the path (`path_on`), refuses (naming what is missing) a machine that lacks any of the
/// path's needs. A share of 0, or below 1/512, needs nothing. Pure.
pub fn check_path(share: f32, have: Path) -> Result<(), Error> {
    if !path_on(share) {
        return Ok(());
    }
    let missing = if !have.amd_uncached {
        "AmdUncached memory (the mailbox's type 5)"
    } else if !have.shader_clock {
        "VK_KHR_shader_clock"
    } else if !have.avx512 {
        "AVX-512 (the expert worker's four features)"
    } else if let Err(what) = workers(have.mask) {
        what
    } else {
        return Ok(());
    };
    Err(Error::CpuPathUnavailable { what: missing })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage::entries;
    use kernels::dispatch::{WHERE_CPU, WHERE_HOST, WHERE_VRAM};

    #[test]
    fn the_overlay_does_not_touch_vram_mtp_or_prefill() {
        // n_layer 2, n_expert 256. Layer 0: expert 0 HOST at 16, expert 1 VRAM at 32, the rest HOST; layer 1 all
        // HOST but expert 3, an unknown kind (7) at 0x3000, and expert 5, HOST at a foreign address (0x7000, not its
        // host copy); layer 2 (the MTP block) all WHERE_CPU at address 0 but expert 0, HOST at 0x2000, so a rewrite
        // of the MTP block, of anything that is not HOST, or of a HOST entry the coordinator would not compute from,
        // shows.
        let (n_layer, n_expert) = (2usize, 256usize);
        let copy = |l: u64, e: u64| {
            if l == 0 && e == 0 {
                16
            } else {
                0x10_0000 + 0x1_0000 * l + 16 * e
            }
        };
        let host: Vec<u64> = (0..n_layer as u64)
            .flat_map(|l| (0..n_expert as u64).map(move |e| copy(l, e)))
            .collect();
        let mut mirror = Vec::new();
        for l in 0..n_layer as u64 {
            for e in 0..n_expert as u64 {
                mirror.push(if l == 0 && e == 1 {
                    (32, WHERE_VRAM)
                } else if l == 1 && e == 3 {
                    (0x3000, 7)
                } else if l == 1 && e == 5 {
                    (0x7000, WHERE_HOST)
                } else {
                    (copy(l, e), WHERE_HOST)
                });
            }
        }
        mirror.push((0x2000, WHERE_HOST));
        mirror.extend(std::iter::repeat_n((0u64, WHERE_CPU), n_expert - 1));

        let t = entries(&overlay(&mirror, n_layer, n_expert, &host, 256));
        assert_eq!(t.len(), mirror.len());
        // Expert 0 is WHERE_CPU at the same address; expert 1 is still WHERE_VRAM at 32.
        assert_eq!(t[0], (16, WHERE_CPU));
        assert_eq!(t[1], (32, WHERE_VRAM));
        // The HOST entry at a foreign address stays HOST there, even at share 1.
        assert_eq!(t[n_expert + 5], (0x7000, WHERE_HOST));
        // Share 1 turns every HOST entry of the main layers at its expert's host copy to CPU at that address, and
        // nothing else.
        for (i, (&got, &was)) in t.iter().zip(&mirror).enumerate().take(n_layer * n_expert) {
            if was.1 == WHERE_HOST && was.0 == host[i] {
                assert_eq!(got, (was.0, WHERE_CPU), "entry {i}");
            } else {
                assert_eq!(got, was, "entry {i}");
            }
        }
        // The unknown kind is copied through, and the MTP block is the mirror's, its HOST entry included.
        assert_eq!(t[n_expert + 3], (0x3000, 7));
        assert_eq!(t[n_layer * n_expert..], mirror[n_layer * n_expert..]);
        assert_eq!(t[n_layer * n_expert], (0x2000, WHERE_HOST));

        // q8 0 leaves expert 0 HOST, and the whole table as the mirror's own bytes.
        let none = overlay(&mirror, n_layer, n_expert, &host, 0);
        assert_eq!(entries(&none)[0], (16, WHERE_HOST));
        assert_eq!(entries(&none), mirror);

        // A partial share assigns only the experts `assigns_cpu` names, in every main layer alike.
        let half = entries(&overlay(&mirror, n_layer, n_expert, &host, 128));
        for l in 0..n_layer {
            for e in 0..n_expert {
                let i = l * n_expert + e;
                let want = if mirror[i].1 == WHERE_HOST
                    && mirror[i].0 == host[i]
                    && assigns_cpu(e as u32, 128)
                {
                    WHERE_CPU
                } else {
                    mirror[i].1
                };
                assert_eq!(half[i], (mirror[i].0, want), "layer {l} expert {e}");
            }
        }
    }

    #[test]
    fn the_share_the_step_reads_is_the_one_set_between_tokens() {
        // The share cell a running path holds, read by the table each step builds: with the share at 0 the HOST entry
        // stays HOST; store 1.0 as `set_cpu_share` does and the next step's table makes it WHERE_CPU at the same
        // address. Fails if the step cached the share from open.
        let (n_layer, n_expert) = (2usize, 16usize);
        let mirror: Vec<(u64, u32)> = (0..(n_layer + 1) * n_expert)
            .map(|i| (0x10_0000 + 0x100 * i as u64, WHERE_HOST))
            .collect();
        let host: Vec<u64> = mirror[..n_layer * n_expert].iter().map(|m| m.0).collect();
        let cell = ShareCell::new(0.0);
        assert_eq!(cell.get(), 0.0);
        let before = entries(&step_table(&mirror, n_layer, n_expert, &host, &cell));
        assert_eq!(before[0], (0x10_0000, WHERE_HOST));
        assert_eq!(before, mirror, "share 0: the mirror's own table");
        cell.set(1.0);
        assert_eq!(cell.get(), 1.0);
        let after = entries(&step_table(&mirror, n_layer, n_expert, &host, &cell));
        assert_eq!(after[0], (0x10_0000, WHERE_CPU));
        // A share between the two is read exactly as set, not rounded through another unit.
        cell.set(0.0019);
        assert_eq!(cell.get(), 0.0019);
        assert_eq!(
            step_table(&mirror, n_layer, n_expert, &host, &cell),
            overlay(&mirror, n_layer, n_expert, &host, share_q8(0.0019))
        );
    }

    #[test]
    fn reset_clears_the_mailbox_flags() {
        // A leftover seq in a publish or done flag would satisfy the next session's first token.
        let mut p = vec![0x11u8; kernels::dispatch::mbox::X0];
        unsafe { clear_flags(p.as_mut_ptr()) };
        assert!(p.iter().all(|&b| b == 0), "every flag byte is 0");
    }

    #[test]
    fn the_status_words_zero_between_tokens() {
        // The per-step call: without it one timeout would make every later token of the session skip the wait (the
        // fallback would still fill ye, so no GPU test can catch it, only this call's existence and the dead-pool
        // test's frozen cpu_timeouts).
        use kernels::dispatch::mbox::STATUS_BYTE;
        let mut p = vec![0u8; kernels::dispatch::mbox::X0];
        p[STATUS_BYTE..STATUS_BYTE + 16].fill(0x11);
        unsafe { clear_status(p.as_mut_ptr()) };
        assert!(
            p[STATUS_BYTE..STATUS_BYTE + 8].iter().all(|&b| b == 0),
            "the timeout and skip counts are 0"
        );
        assert!(
            p[STATUS_BYTE + 8..STATUS_BYTE + 16]
                .iter()
                .all(|&b| b == 0x11),
            "only the two status words are written"
        );
    }

    #[test]
    fn share_one_assigns_every_expert_and_share_zero_none() {
        assert_eq!(share_q8(0.0), 0);
        assert_eq!(share_q8(1.0), 256);
        assert_eq!(share_q8(0.5), 128);
        assert!(assigns_cpu(0, 256) && assigns_cpu(255, 256));
        assert!(!assigns_cpu(0, 0) && !assigns_cpu(255, 0));
        let n = (0..256).filter(|&e| assigns_cpu(e, 128)).count();
        assert_eq!(n, 128);
        // Every q8 assigns exactly q8 of 256 experts (17 is coprime to 256), and larger experts wrap, not overflow.
        for q8 in [1, 7, 64, 255] {
            assert_eq!(
                (0..256).filter(|&e| assigns_cpu(e, q8)).count(),
                q8 as usize
            );
        }
        assert!(assigns_cpu(u32::MAX, 256));
        // The ×17 itself, not only the counts (any bijection gives those): 17 · 241 ≡ 1 (mod 256).
        assert!(assigns_cpu(241, 2) && !assigns_cpu(1, 2));
    }

    #[test]
    fn a_share_below_one_512th_is_zero() {
        // A share in (0, 1/512) assigns no expert: it runs no path (no publish, wait or serve 2), needs nothing of the
        // machine, and is no conflict with the base time. 1/512 rounds to one expert in 256.
        assert_eq!(share_q8(0.001), 0);
        assert!(!path_on(0.0) && !path_on(0.001) && !path_on(0.0019));
        assert!(path_on(1.0 / 512.0) && path_on(0.25) && path_on(1.0));
        let none = Path {
            amd_uncached: false,
            shader_clock: false,
            avx512: false,
            mask: 0,
        };
        assert!(check_path(0.001, none).is_ok());
        assert!(check_share(0.001, true).is_ok());
        assert!(matches!(
            check_path(1.0 / 512.0, none),
            Err(crate::Error::CpuPathUnavailable { .. })
        ));
    }

    #[test]
    fn a_share_outside_zero_to_one_is_refused_before_the_model_loads() {
        for ok in [0.0f32, 0.25, 1.0] {
            assert!(check_share(ok, false).is_ok(), "{ok}");
        }
        for bad in [-0.01f32, 1.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            match check_share(bad, false) {
                Err(crate::Error::BadCpuShare { share }) => {
                    assert!(share == bad || share.is_nan() && bad.is_nan())
                }
                other => panic!("{bad}: expected BadCpuShare, got {other:?}"),
            }
        }
        // A valid share with the base time is its own error; a share of 0 with the base time is fine.
        assert!(matches!(
            check_share(0.5, true),
            Err(crate::Error::BaseTimeAndCpuShare)
        ));
        assert!(check_share(0.0, true).is_ok());
        // An invalid share is refused as such even with the base time.
        assert!(matches!(
            check_share(2.0, true),
            Err(crate::Error::BadCpuShare { .. })
        ));
    }

    #[test]
    fn the_path_refuses_what_it_lacks_by_name() {
        // A CCD1 affinity mask: eight even workers, and the odd CPUs left for the engine thread.
        let all = Path {
            amd_uncached: true,
            shader_clock: true,
            avx512: true,
            mask: 0xFFFF_0000,
        };
        assert!(check_path(0.5, all).is_ok());
        // A share of 0 never needs the path.
        let none = Path {
            amd_uncached: false,
            shader_clock: false,
            avx512: false,
            mask: 0,
        };
        assert!(check_path(0.0, none).is_ok());
        for (what, lack) in [
            (
                "AmdUncached memory",
                Path {
                    amd_uncached: false,
                    ..all
                },
            ),
            (
                "VK_KHR_shader_clock",
                Path {
                    shader_clock: false,
                    ..all
                },
            ),
            (
                "AVX-512",
                Path {
                    avx512: false,
                    ..all
                },
            ),
            ("an even CPU", Path { mask: 0, ..all }),
            (
                "an even CPU",
                Path {
                    mask: 1 << 17,
                    ..all
                },
            ),
            // The workers would fill the mask, leaving the engine thread only CPUs that hold a TIME_CRITICAL spin:
            // the physical-cores-only mask (CCD1's even CPUs) and one even CPU.
            (
                "besides the workers'",
                Path {
                    mask: 0x5555_0000,
                    ..all
                },
            ),
            (
                "besides the workers'",
                Path {
                    mask: 1 << 18,
                    ..all
                },
            ),
        ] {
            match check_path(1.0, lack) {
                Err(crate::Error::CpuPathUnavailable { what: got }) => {
                    assert!(got.contains(what), "{what}: {got}")
                }
                other => panic!("{what}: expected CpuPathUnavailable, got {other:?}"),
            }
        }
    }

    #[test]
    fn check_expert_types_refuses_unquantized_experts() {
        // The path off (share 0, or below 1/512): any type passes.
        assert!(check_expert_types(0.0, &[WType::Bf16]).is_ok());
        assert!(check_expert_types(0.001, &[WType::F32]).is_ok());
        // On: one unquantized type among quantized ones is refused, and the message names it.
        match check_expert_types(1.0, &[WType::Q4K, WType::Bf16]) {
            Err(e @ crate::Error::CpuShareExpertType { ty: "BF16" }) => assert_eq!(
                e.to_string(),
                "--cpu-share needs quantized expert matrices (Q4_K, Q5_K, Q6_K or Q8_0); this model's experts include BF16"
            ),
            other => panic!("expected CpuShareExpertType for BF16, got {other:?}"),
        }
        assert!(matches!(
            check_expert_types(0.25, &[WType::F32]),
            Err(crate::Error::CpuShareExpertType { ty: "F32" })
        ));
        assert!(
            check_expert_types(1.0, &[WType::Q4K, WType::Q5K, WType::Q6K, WType::Q8_0]).is_ok()
        );
    }
}
