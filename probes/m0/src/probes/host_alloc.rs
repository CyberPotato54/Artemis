//! Can 18.68 GiB of experts live in pinned host memory? Allocates 512 MiB chunks until the target or the first failure,
//! touching every page, then frees them.
use crate::{
    ctx::{Ctx, Mem},
    record::Record,
};
use anyhow::{ensure, Result};
use serde_json::json;
use std::time::Instant;

#[derive(clap::Args, Debug, serde::Serialize)]
pub struct Args {
    #[arg(long, default_value_t = 512)]
    pub chunk_mib: u64,
    /// 19,968 MiB = 39 chunks, a little more than the 18.68 GiB of experts.
    #[arg(long, default_value_t = 19968)]
    pub total_mib: u64,
    #[arg(long, value_enum, default_value_t = Mem::Host)]
    pub mem: Mem,
}

pub fn run(ctx: &Ctx, a: &Args) -> Result<Record> {
    ensure!(
        a.mem != Mem::Vram,
        "host-alloc measures host memory; use --mem host or host-cached"
    );
    let before = ctx.heap_budgets();
    let t0 = Instant::now();
    let mut bufs = Vec::new();
    let mut error = None;
    while (bufs.len() as u64) * a.chunk_mib < a.total_mib {
        match ctx.buf(a.chunk_mib << 20, a.mem) {
            Ok(b) => {
                for off in (0..b.size as usize).step_by(4096) {
                    // SAFETY: b is mapped and b.size bytes long.
                    unsafe { b.ptr.add(off).write_volatile(1) };
                }
                bufs.push(b);
            }
            Err(e) => {
                error = Some(format!("{e:#}"));
                break;
            }
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let during = ctx.heap_budgets();
    let allocated = bufs.len() as u64 * a.chunk_mib;
    for b in bufs {
        ctx.free(b);
    }
    ctx.record(
        "host-alloc",
        a,
        json!({
            "allocated_mib": allocated,
            "chunks": allocated / a.chunk_mib,
            "reached_total": allocated >= a.total_mib,
            "ms": ms,
            "error": error,
            "heaps_before": before,
            "heaps_during": during,
        }),
    )
}
