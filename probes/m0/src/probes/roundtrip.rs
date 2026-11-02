//! CPU↔GPU handoff latency. Three ways the GPU could wait for the CPU mid-token:
//! - spin: a running shader polls a flag in host memory that a CPU thread writes (gate 2);
//! - semaphore: a submission queued behind a timeline semaphore that the CPU signals;
//! - submit: a fresh submission per handoff.
//!
//! `--decode-shaped` runs the spin handoff inside a decode-shaped command buffer: each of `--layers` layers reads
//! `--layer-kib` of VRAM, as a layer's weights are read, then waits once for the CPU, as the CPU miss path would after
//! the router. The GPU times each wait with timestamps around the waiting dispatch.
use crate::{
    ctx::{is_device_lost, Buf, Ctx, Mem},
    record::{percentile, Record},
};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{
    mem::size_of,
    sync::atomic::{fence, AtomicBool, AtomicU32, Ordering},
    time::{Duration, Instant},
};

const SPIN_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spin.spv"));
const READ_BW_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/read_bw.spv"));
/// Workgroups of each decode-shaped layer's VRAM read (VRAM reads need at least 64 to saturate).
const READ_WGS: u32 = 512;
/// Most layers (so waits) to a decode-shaped command buffer.
const MAX_LAYERS: u32 = 128;
/// Bound on `--layer-kib` that only keeps the byte count (`layer_kib << 10`) from overflowing; the window check in
/// `decode_shaped_args` caps a layer at 524,288 KiB (512 MiB), so that a window of at least two layers fits in 1 GiB.
const MAX_LAYER_KIB: u64 = 1 << 20;
const PING_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ping.spv"));

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    Spin,
    Semaphore,
    Submit,
}

#[derive(clap::Args, Debug, serde::Serialize)]
pub struct Args {
    #[arg(long, value_enum, default_value_t = Mode::Spin)]
    pub mode: Mode,
    /// Memory holding the flags (amd-uncached needs VK_AMD_device_coherent_memory).
    #[arg(long, value_enum, default_value_t = Mem::Host)]
    pub flag_mem: Mem,
    /// Round trips in total.
    #[arg(long, default_value_t = 1_000_000)]
    pub n: u32,
    /// Spin mode: round trips per dispatch.
    #[arg(long, default_value_t = 1000)]
    pub batch: u32,
    /// Spin mode: polls before the GPU gives up on one answer and ends its dispatch.
    #[arg(long, default_value_t = 200_000)]
    pub max_spins: u32,
    /// Test hook: spin mode with no CPU responder, to prove the GPU gives up instead of hanging.
    #[arg(long)]
    pub no_responder: bool,
    /// Spin mode inside a decode-shaped command buffer: one handoff per layer, after the layer's VRAM read, timed on
    /// the GPU. `--n` counts handoffs; `--batch` doesn't apply.
    #[arg(long)]
    pub decode_shaped: bool,
    /// Decode-shaped: layers, so handoffs, per command buffer.
    #[arg(long, default_value_t = 40)]
    pub layers: u32,
    /// Decode-shaped: KiB of VRAM each layer reads before it waits (36 MiB is about a DeltaNet layer's dense weights).
    #[arg(long, default_value_t = 36_864)]
    pub layer_kib: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SpinPush {
    flags: u64,
    first: u32,
    count: u32,
    max_spins: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PingPush {
    flags: u64,
}

/// `read_bw.comp`'s push constants.
#[repr(C)]
#[derive(Clone, Copy)]
struct ReadPush {
    src: u64,
    dst: u64,
    n_vec4: u32,
    pad: u32,
}

struct Flags<'a> {
    gpu_seq: &'a AtomicU32,
    cpu_seq: &'a AtomicU32,
    timeouts: &'a AtomicU32,
}

fn flags(b: &Buf) -> Flags<'_> {
    let p = b.ptr as *const AtomicU32;
    // SAFETY: b is mapped, 4-byte aligned and at least 12 bytes long; the GPU accesses these words atomically too.
    unsafe {
        Flags {
            gpu_seq: &*p,
            cpu_seq: &*p.add(1),
            timeouts: &*p.add(2),
        }
    }
}

fn spin_until(a: &AtomicU32, target: u32, timeout: Duration) -> bool {
    let t0 = Instant::now();
    while a.load(Ordering::Acquire) != target {
        if t0.elapsed() > timeout {
            return false;
        }
        std::hint::spin_loop();
    }
    true
}

pub fn run(ctx: &Ctx, a: &Args) -> Result<Record> {
    ensure!(
        a.flag_mem != Mem::Vram,
        "the flags must be in host-visible memory"
    );
    ensure!(
        a.n >= 1 && a.batch >= 1,
        "--n and --batch must be at least 1"
    );
    if a.decode_shaped {
        decode_shaped_args(a)?;
    }
    let fb = ctx.buf(64, a.flag_mem)?;
    let f = flags(&fb);
    for w in [f.gpu_seq, f.cpu_seq, f.timeouts] {
        w.store(0, Ordering::SeqCst);
    }
    let (samples, gpu_timeouts, cpu_timeouts, device_lost) = match a.mode {
        Mode::Spin if a.decode_shaped => spin_decode_shaped(ctx, a, &fb)?,
        Mode::Spin => spin(ctx, a, &fb)?,
        Mode::Semaphore | Mode::Submit => queued(ctx, a, &fb)?,
    };
    ctx.free(fb);
    let mut s = samples;
    s.sort_by(f64::total_cmp);
    let results: Value = json!({
        "samples": s.len(),
        "median_us": percentile(&s, 0.5),
        "p99_us": percentile(&s, 0.99),
        "p999_us": percentile(&s, 0.999),
        "max_us": s.last().copied().unwrap_or(f64::NAN),
        "gpu_timeouts": gpu_timeouts,
        "cpu_timeouts": cpu_timeouts,
        "device_lost": device_lost,
        "timed_by": if a.decode_shaped { "gpu timestamps around each wait" } else { "the cpu" },
    });
    ctx.record("roundtrip", a, results)
}

fn spin(ctx: &Ctx, a: &Args, fb: &Buf) -> Result<(Vec<f64>, u32, u32, bool)> {
    let f = flags(fb);
    let pipe = ctx.pipe(SPIN_SPV, size_of::<SpinPush>() as u32)?;
    let stop = AtomicBool::new(false);
    let mut device_lost = false;
    let mut failure = None;
    let (samples, cpu_timeouts) = std::thread::scope(|s| {
        let responder = (!a.no_responder).then(|| s.spawn(|| respond(&f, a.n, a.batch, &stop)));
        for b in 0..a.n.div_ceil(a.batch) {
            let first = 1 + b * a.batch;
            let push = SpinPush {
                flags: fb.addr,
                first,
                count: a.batch.min(a.n + 1 - first),
                max_spins: a.max_spins,
                pad: 0,
            };
            let r = ctx.run(ctx.compute_queue, ctx.compute_pool, |cmd| {
                ctx.dispatch(cmd, &pipe, &push, 1);
                ctx.host_barrier(cmd);
            });
            if let Err(e) = r {
                device_lost = is_device_lost(&e);
                failure = (!device_lost).then_some(e);
                break;
            }
            if f.timeouts.load(Ordering::Acquire) > 0 {
                break;
            }
        }
        stop.store(true, Ordering::Release);
        responder
            .map(|h| h.join().expect("responder thread panicked"))
            .unwrap_or_default()
    });
    // Any other failure (a fence timeout, say) would leave a short run that looks clean: fail the probe instead.
    if let Some(e) = failure {
        return Err(e.context("spin run stopped early"));
    }
    Ok((
        samples,
        f.timeouts.load(Ordering::Acquire),
        cpu_timeouts,
        device_lost,
    ))
}

/// Limits of the decode-shaped mode, checked before anything is allocated: spin mode on the uncached flags the probe
/// is about; at most `MAX_LAYERS` layers to a submit (a missing CPU costs a buffer one wait's spins plus 1,024 polls
/// a wait, and the reads must keep the submit well under 0.5 s); a layer of at least 1 KiB, and the window of layers
/// it rotates through within the 1 GiB allocation rule.
fn decode_shaped_args(a: &Args) -> Result<()> {
    ensure!(
        a.mode == Mode::Spin,
        "--decode-shaped runs spin mode, not {:?}",
        a.mode
    );
    ensure!(
        a.flag_mem == Mem::AmdUncached,
        "--decode-shaped needs --flag-mem amd-uncached"
    );
    ensure!(
        (1..=MAX_LAYERS).contains(&a.layers),
        "--layers must be 1 to {MAX_LAYERS}, got {}",
        a.layers
    );
    ensure!(
        (1..=MAX_LAYER_KIB).contains(&a.layer_kib),
        "--layer-kib must be 1 to {MAX_LAYER_KIB} (that bound only keeps the byte count from overflowing; the window check caps a layer at 524,288 KiB, 512 MiB), got {}",
        a.layer_kib
    );
    let window = decode_shaped_window(a.layer_kib << 10);
    ensure!(
        window <= 1 << 30,
        "--layer-kib {} rotates through {window} bytes, over the 1 GiB allocation limit (at most 524,288 KiB)",
        a.layer_kib
    );
    Ok(())
}

/// Bytes of the buffer the layers' reads rotate through: 512 MiB or more, and at least two layers.
fn decode_shaped_window(layer: u64) -> u64 {
    (512u64 << 20).div_ceil(layer).max(2).saturating_mul(layer)
}

/// Spin mode inside a decode-shaped command buffer: per layer, a VRAM read of `layer_kib` (rotating through 512 MiB
/// or more, so it isn't served from the 64 MB cache), a barrier, then one handoff, timed by a timestamp before and one
/// after its dispatch. A missing answer ends its wait after `max_spins` polls (a GPU timeout), and every later wait of
/// the buffer within 1,024 polls more (`spin.comp`), so a buffer costs about one wait's spins when the CPU is gone; the
/// run stops at the first timeout. A failed submit or wait leaves the buffer possibly running: the device is idled
/// before the buffers it reads are freed.
fn spin_decode_shaped(ctx: &Ctx, a: &Args, fb: &Buf) -> Result<(Vec<f64>, u32, u32, bool)> {
    let f = flags(fb);
    let spin = ctx.pipe(SPIN_SPV, size_of::<SpinPush>() as u32)?;
    let read = ctx.pipe(READ_BW_SPV, size_of::<ReadPush>() as u32)?;
    let layer = a.layer_kib << 10;
    let windows = decode_shaped_window(layer) / layer;
    let busy = ctx.buf(windows * layer, Mem::Vram)?;
    let dst = match ctx.buf(4 * READ_WGS as u64, Mem::Vram) {
        Ok(b) => b,
        Err(e) => {
            ctx.free(busy);
            return Err(e);
        }
    };
    let stamps = match ctx.stamps(2 * a.layers) {
        Ok(s) => s,
        Err(e) => {
            ctx.free(busy);
            ctx.free(dst);
            return Err(e);
        }
    };
    let stop = AtomicBool::new(false);
    let mut samples = Vec::with_capacity(a.n as usize);
    let (mut device_lost, mut failure) = (false, None);
    let cpu_timeouts = std::thread::scope(|s| {
        // Batches of one: the responder answers and keeps no samples of its own.
        let responder = (!a.no_responder).then(|| s.spawn(|| respond(&f, a.n, 1, &stop)));
        let (mut k, mut w) = (1u32, 0u64);
        while k <= a.n {
            let count = a.layers.min(a.n + 1 - k);
            ctx.reset_stamps(&stamps);
            let r = ctx.run(ctx.compute_queue, ctx.compute_pool, |cmd| {
                for l in 0..count {
                    let read_push = ReadPush {
                        src: busy.addr + (w + l as u64) % windows * layer,
                        dst: dst.addr,
                        n_vec4: (layer / 16) as u32,
                        pad: 0,
                    };
                    ctx.dispatch(cmd, &read, &read_push, READ_WGS);
                    ctx.barrier(cmd);
                    ctx.stamp(cmd, &stamps, 2 * l);
                    let spin_push = SpinPush {
                        flags: fb.addr,
                        first: k + l,
                        count: 1,
                        max_spins: a.max_spins,
                        pad: 0,
                    };
                    ctx.dispatch(cmd, &spin, &spin_push, 1);
                    ctx.stamp(cmd, &stamps, 2 * l + 1);
                    ctx.barrier(cmd);
                }
                ctx.host_barrier(cmd);
            });
            if let Err(e) = r {
                device_lost = is_device_lost(&e);
                failure = (!device_lost).then_some(e);
                break;
            }
            if f.timeouts.load(Ordering::Acquire) > 0 {
                break;
            }
            match ctx.read_first_stamps_ns(&stamps, 2 * count) {
                Ok(ns) => samples.extend(ns.chunks(2).map(|p| (p[1] - p[0]) / 1e3)),
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
            k += count;
            w += count as u64;
        }
        stop.store(true, Ordering::Release);
        responder
            .map(|h| h.join().expect("responder thread panicked").1)
            .unwrap_or_default()
    });
    if failure.is_some() || device_lost {
        // SAFETY: waiting for all submitted work before its buffers are freed; on a lost device this returns at once.
        let _ = unsafe { ctx.device.device_wait_idle() };
    }
    ctx.free(busy);
    ctx.free(dst);
    if let Some(e) = failure {
        return Err(e.context("decode-shaped spin run stopped early"));
    }
    Ok((
        samples,
        f.timeouts.load(Ordering::Acquire),
        cpu_timeouts,
        device_lost,
    ))
}

/// The CPU side: answer each GPU ping with the same number, and time each round trip from our previous answer to
/// the GPU's next ping. The first ping of each dispatch is skipped, because its wait includes a submit.
fn respond(f: &Flags, n: u32, batch: u32, stop: &AtomicBool) -> (Vec<f64>, u32) {
    let mut samples = Vec::with_capacity(n as usize);
    let mut last = 0u32;
    let mut answered_at: Option<Instant> = None;
    let mut cpu_timeouts = 0;
    while last < n {
        let t0 = Instant::now();
        let seen = loop {
            let v = f.gpu_seq.load(Ordering::Acquire);
            if v != last {
                break Some(v);
            }
            if stop.load(Ordering::Acquire) {
                break None;
            }
            if t0.elapsed() > Duration::from_secs(2) {
                cpu_timeouts += 1;
                break None;
            }
            std::hint::spin_loop();
        };
        let Some(v) = seen else { break };
        let now = Instant::now();
        if let Some(t) = answered_at {
            if v == last + 1 && (v - 1) % batch != 0 {
                samples.push((now - t).as_secs_f64() * 1e6);
            }
        }
        f.cpu_seq.store(v, Ordering::Release);
        // A locked instruction drains the write-combining buffer, so the GPU sees the answer now, not eventually.
        fence(Ordering::SeqCst);
        answered_at = Some(Instant::now());
        last = v;
    }
    (samples, cpu_timeouts)
}

fn queued(ctx: &Ctx, a: &Args, fb: &Buf) -> Result<(Vec<f64>, u32, u32, bool)> {
    let f = flags(fb);
    let pipe = ctx.pipe(PING_SPV, size_of::<PingPush>() as u32)?;
    let cmd = ctx.reusable(|cmd| {
        ctx.dispatch(cmd, &pipe, &PingPush { flags: fb.addr }, 1);
        ctx.host_barrier(cmd);
    })?;
    let sem = ctx.timeline()?;
    let mut samples = Vec::with_capacity(a.n as usize);
    let (mut cpu_timeouts, mut device_lost) = (0, false);
    for k in 1..=a.n {
        let started = if a.mode == Mode::Semaphore {
            // Queue the work behind the semaphore first; the timed part is signal → GPU runs → CPU sees it.
            ctx.submit(cmd, Some((sem, k as u64))).and_then(|_| {
                let t0 = Instant::now();
                ctx.signal(sem, k as u64).map(|_| t0)
            })
        } else {
            let t0 = Instant::now();
            ctx.submit(cmd, None).map(|_| t0)
        };
        let t0 = match started {
            Ok(t) => t,
            Err(e) => {
                device_lost = is_device_lost(&e);
                break;
            }
        };
        if !spin_until(f.gpu_seq, k, Duration::from_secs(2)) {
            cpu_timeouts += 1;
            break;
        }
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    // SAFETY: waiting for all queued work before the flag buffer is freed.
    unsafe { ctx.device.device_wait_idle()? };
    Ok((samples, 0, cpu_timeouts, device_lost))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(layers: u32, layer_kib: u64) -> Args {
        Args {
            mode: Mode::Spin,
            flag_mem: Mem::AmdUncached,
            n: 1,
            batch: 1,
            max_spins: 1,
            no_responder: false,
            decode_shaped: true,
            layers,
            layer_kib,
        }
    }

    #[test]
    fn decode_shaped_arguments_are_bounded() {
        assert!(decode_shaped_args(&args(40, 36_864)).is_ok());
        assert!(decode_shaped_args(&args(1, 36_864)).is_ok());
        assert!(decode_shaped_args(&args(MAX_LAYERS, 1)).is_ok());
        assert!(decode_shaped_args(&args(0, 4_096)).is_err());
        assert!(decode_shaped_args(&args(MAX_LAYERS + 1, 4_096)).is_err());
        assert!(decode_shaped_args(&args(u32::MAX, 4_096)).is_err());
        assert!(decode_shaped_args(&args(40, 0)).is_err());
        // two windows of this layer are 1,026 MiB: over the 1 GiB allocation rule
        assert!(decode_shaped_args(&args(40, 513 << 10)).is_err());
        assert!(decode_shaped_args(&args(40, 512 << 10)).is_ok());
        assert!(decode_shaped_args(&args(40, u64::MAX)).is_err());
        let mut other = args(40, 4_096);
        other.flag_mem = Mem::Host;
        assert!(decode_shaped_args(&other).is_err());
        other = args(40, 4_096);
        other.mode = Mode::Submit;
        assert!(decode_shaped_args(&other).is_err());
    }
}
