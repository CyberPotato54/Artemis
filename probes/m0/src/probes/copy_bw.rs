//! Host→VRAM copy bandwidth: what cache uploads and prefill staging can expect.
use crate::{
    ctx::{Ctx, Mem},
    record::{percentile, Record},
};
use anyhow::Result;
use ash::vk;
use serde_json::json;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Queue {
    Transfer,
    Compute,
}

#[derive(clap::Args, Debug, serde::Serialize)]
pub struct Args {
    #[arg(long, default_value_t = 256)]
    pub mib: u64,
    #[arg(long, default_value_t = 20)]
    pub reps: u32,
    #[arg(long, value_enum, default_value_t = Queue::Transfer)]
    pub queue: Queue,
}

pub fn run(ctx: &Ctx, a: &Args) -> Result<Record> {
    let bytes = a.mib << 20;
    let src = ctx.buf(bytes, Mem::Host)?;
    let dst = ctx.buf(bytes, Mem::Vram)?;
    let (queue, pool, gpu_ts) = match a.queue {
        Queue::Transfer => (
            ctx.transfer_queue,
            ctx.transfer_pool,
            ctx.transfer_timestamps,
        ),
        Queue::Compute => (ctx.compute_queue, ctx.compute_pool, true),
    };
    let stamps = ctx.stamps(2)?;
    let mut us = Vec::with_capacity(a.reps as usize);
    for _ in 0..a.reps {
        ctx.reset_stamps(&stamps);
        let t0 = Instant::now();
        ctx.run(queue, pool, |cmd| {
            if gpu_ts {
                ctx.stamp(cmd, &stamps, 0);
            }
            let region = [vk::BufferCopy::default().size(bytes)];
            // SAFETY: cmd is recording; both buffers are bytes long.
            unsafe { ctx.device.cmd_copy_buffer(cmd, src.buf, dst.buf, &region) };
            if gpu_ts {
                ctx.stamp(cmd, &stamps, 1);
            }
        })?;
        let cpu_us = t0.elapsed().as_secs_f64() * 1e6;
        us.push(if gpu_ts {
            let t = ctx.read_stamps_ns(&stamps)?;
            (t[1] - t[0]) / 1e3
        } else {
            cpu_us
        });
    }
    us.sort_by(f64::total_cmp);
    let med = percentile(&us, 0.5);
    ctx.free(src);
    ctx.free(dst);
    ctx.record(
        "copy-bw",
        a,
        json!({
            "median_us": med,
            "gbps": bytes as f64 / (med * 1e3),
            "timing": if gpu_ts { "gpu" } else { "cpu" },
        }),
    )
}
