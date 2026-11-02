//! Shader read bandwidth. The decode-shaped case (gate 1) is `--chain 40 --kib 4352`: 40 layers, each reading
//! about 2.3 missed experts, with a barrier between layers.
use crate::{
    ctx::{Ctx, Mem},
    record::{percentile, Record},
    rng::Rng,
};
use anyhow::{ensure, Result};
use serde_json::json;
use std::mem::size_of;

const SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/read_bw.spv"));

#[derive(clap::Args, Debug, serde::Serialize)]
pub struct Args {
    #[arg(long, value_enum, default_value_t = Mem::Host)]
    pub mem: Mem,
    /// KiB read per dispatch (4352 KiB is about 2.3 missed experts: one layer's misses at a 40% cache).
    #[arg(long, default_value_t = 4352)]
    pub kib: u64,
    /// Workgroups of 256 threads (at least 8).
    #[arg(long, default_value_t = 256)]
    pub wgs: u32,
    /// Dispatches per measurement, each reading a different window, with barriers between them.
    #[arg(long, default_value_t = 40)]
    pub chain: u32,
    /// Measurements, one submit each.
    #[arg(long, default_value_t = 30)]
    pub reps: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    src: u64,
    dst: u64,
    n_vec4: u32,
    pad: u32,
}

pub fn run(ctx: &Ctx, a: &Args) -> Result<Record> {
    let bytes = a.kib << 10;
    ensure!(
        a.wgs >= 8,
        "--wgs must be at least 8 (fewer risks a slow submit)"
    );
    ensure!(
        a.chain >= 1 && a.reps >= 1,
        "--chain and --reps must be at least 1"
    );
    // Enough windows to cover 512 MiB, so repeated reads can't come from the 64 MB Infinity Cache.
    let windows = ((512u64 << 20) / bytes).max(a.chain as u64);
    let total = windows * bytes;
    let mut data = vec![0u8; total as usize];
    Rng::new(7).fill_bytes(&mut data);
    let src = ctx.buf(total, a.mem)?;
    ctx.upload(&src, &data)?;
    let dst = ctx.buf(a.wgs as u64 * 4, Mem::Host)?;
    let pipe = ctx.pipe(SPV, size_of::<Push>() as u32)?;
    let push = |w: u64| Push {
        src: src.addr + (w % windows) * bytes,
        dst: dst.addr,
        n_vec4: (bytes / 16) as u32,
        pad: 0,
    };

    // Proof the loads happen: one dispatch over window 0, checked against the CPU's XOR of the same bytes.
    ctx.run(ctx.compute_queue, ctx.compute_pool, |cmd| {
        ctx.dispatch(cmd, &pipe, &push(0), a.wgs);
        ctx.host_barrier(cmd);
    })?;
    // SAFETY: dst is mapped and holds a.wgs u32s written by the finished dispatch.
    let gpu_xor = (0..a.wgs as usize).fold(0u32, |x, i| {
        x ^ unsafe { (dst.ptr as *const u32).add(i).read_volatile() }
    });
    let cpu_xor = data[..bytes as usize]
        .as_chunks::<4>()
        .0
        .iter()
        .fold(0u32, |x, c| x ^ u32::from_le_bytes(*c));
    ensure!(
        gpu_xor == cpu_xor,
        "checksum mismatch: GPU {gpu_xor:#x}, CPU {cpu_xor:#x}"
    );

    // Warm the clocks up with the same chain before timing it.
    ctx.warm_up(|cmd| {
        for c in 0..a.chain as u64 {
            ctx.dispatch(cmd, &pipe, &push(c), a.wgs);
            ctx.barrier(cmd);
        }
    })?;
    let stamps = ctx.stamps(2)?;
    let mut us = Vec::with_capacity(a.reps as usize);
    let mut w = 0u64;
    for _ in 0..a.reps {
        ctx.reset_stamps(&stamps);
        ctx.run(ctx.compute_queue, ctx.compute_pool, |cmd| {
            ctx.stamp(cmd, &stamps, 0);
            for _ in 0..a.chain {
                ctx.dispatch(cmd, &pipe, &push(w), a.wgs);
                ctx.barrier(cmd);
                w += 1;
            }
            ctx.stamp(cmd, &stamps, 1);
        })?;
        let t = ctx.read_stamps_ns(&stamps)?;
        us.push((t[1] - t[0]) / 1e3 / a.chain as f64);
    }
    us.sort_by(f64::total_cmp);
    let med = percentile(&us, 0.5);
    ctx.free(src);
    ctx.free(dst);
    ctx.record(
        "read-bw",
        a,
        json!({
            "median_us_per_dispatch": med,
            "p90_us_per_dispatch": percentile(&us, 0.9),
            "gbps": bytes as f64 / (med * 1e3),
            "checksum_ok": true,
        }),
    )
}
