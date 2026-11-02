//! Matvec speed and correctness. 8192 x 2048 Q4_K is 8 experts' gate and up projections (decode-shaped);
//! 8192 x 2048 Q8_0 is one DeltaNet QKV. With --mem host the weights are read over PCIe, as for a missed expert.
use crate::{
    ctx::{dispatches_per_submit, Ctx, Mem},
    q::{f32_bytes, matvec, random_weights, Quant, QK_K},
    record::{percentile, Record},
    rng::Rng,
};
use anyhow::{ensure, Result};
use serde_json::json;
use std::mem::size_of;

const Q4K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matvec_q4k.spv"));
const Q8_0_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matvec_q8_0.spv"));

#[derive(clap::Args, Debug, serde::Serialize)]
pub struct Args {
    #[arg(long, value_enum, default_value_t = Quant::Q4k)]
    pub quant: Quant,
    /// Where the weights live: vram, or host (read over PCIe).
    #[arg(long, value_enum, default_value_t = Mem::Vram)]
    pub mem: Mem,
    /// Rows; a multiple of 8.
    #[arg(long, default_value_t = 8192)]
    pub rows: u32,
    /// Row length; a multiple of 256.
    #[arg(long, default_value_t = 2048)]
    pub k: u32,
    #[arg(long, default_value_t = 200)]
    pub reps: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    w: u64,
    x: u64,
    y: u64,
    rows: u32,
    k: u32,
}

/// Largest absolute difference, relative to the largest reference value.
pub fn max_rel_err(got: &[f32], want: &[f32]) -> f64 {
    let scale = want
        .iter()
        .fold(0f64, |m, v| m.max(v.abs() as f64))
        .max(1e-30);
    got.iter()
        .zip(want)
        .fold(0f64, |m, (g, w)| m.max((*g as f64 - *w as f64).abs()))
        / scale
}

pub fn run(ctx: &Ctx, a: &Args) -> Result<Record> {
    ensure!(a.rows.is_multiple_of(8), "--rows must be a multiple of 8");
    ensure!(
        (a.k as usize).is_multiple_of(QK_K),
        "--k must be a multiple of 256"
    );
    let size = (a.quant.row_bytes(a.k as usize) * a.rows as usize) as u64;
    // Copies to rotate through (~512 MiB), so timing can't come from the 64 MB Infinity Cache.
    let copies = ((512u64 << 20) / size).max(1);
    let w_host = random_weights(a.quant, a.rows as usize, a.k as usize, 11);
    let w = ctx.buf(size * copies, a.mem)?;
    ctx.upload(&w, &w_host.repeat(copies as usize))?;
    let mut rng = Rng::new(5);
    let x_host: Vec<f32> = (0..a.k).map(|_| rng.next_f32()).collect();
    let x = ctx.buf(a.k as u64 * 4, Mem::Vram)?;
    ctx.upload(&x, f32_bytes(&x_host))?;
    let y = ctx.buf(a.rows as u64 * 4, Mem::Host)?;
    let spv = match a.quant {
        Quant::Q4k => Q4K_SPV,
        Quant::Q8_0 => Q8_0_SPV,
    };
    let pipe = ctx.pipe(spv, size_of::<Push>() as u32)?;
    let push = |c: u64| Push {
        w: w.addr + (c % copies) * size,
        x: x.addr,
        y: y.addr,
        rows: a.rows,
        k: a.k,
    };

    ctx.run(ctx.compute_queue, ctx.compute_pool, |cmd| {
        ctx.dispatch(cmd, &pipe, &push(0), a.rows / 8);
        ctx.host_barrier(cmd);
    })?;
    // SAFETY: y is mapped and holds a.rows f32s written by the finished dispatch.
    let got: Vec<f32> = (0..a.rows as usize)
        .map(|i| unsafe { (y.ptr as *const f32).add(i).read_volatile() })
        .collect();
    let rel = max_rel_err(&got, &matvec(a.quant, &w_host, &x_host, a.rows as usize));
    ensure!(
        rel <= 1e-4,
        "GPU matvec differs from the CPU reference: max relative error {rel:e}"
    );

    // Size submits from one timed dispatch, warm the clocks up, then time `reps` dispatches in ~20 ms submits.
    let one = ctx.stamps(2)?;
    ctx.reset_stamps(&one);
    ctx.run(ctx.compute_queue, ctx.compute_pool, |cmd| {
        ctx.stamp(cmd, &one, 0);
        ctx.dispatch(cmd, &pipe, &push(0), a.rows / 8);
        ctx.stamp(cmd, &one, 1);
    })?;
    let t = ctx.read_stamps_ns(&one)?;
    let per = dispatches_per_submit((t[1] - t[0]) / 1e3);
    ctx.warm_up(|cmd| {
        for i in 0..per {
            ctx.barrier(cmd);
            ctx.dispatch(cmd, &pipe, &push(i as u64), a.rows / 8);
        }
    })?;
    let stamps = ctx.stamps(2 * per)?;
    let mut us: Vec<f64> = Vec::with_capacity(a.reps as usize);
    let mut done = 0u32;
    while done < a.reps {
        let n = per.min(a.reps - done);
        ctx.reset_stamps(&stamps);
        ctx.run(ctx.compute_queue, ctx.compute_pool, |cmd| {
            for i in 0..n {
                ctx.barrier(cmd);
                ctx.stamp(cmd, &stamps, 2 * i);
                ctx.dispatch(cmd, &pipe, &push((done + i) as u64), a.rows / 8);
                ctx.stamp(cmd, &stamps, 2 * i + 1);
            }
        })?;
        let t = ctx.read_first_stamps_ns(&stamps, 2 * n)?;
        us.extend((0..n as usize).map(|i| (t[2 * i + 1] - t[2 * i]) / 1e3));
        done += n;
    }
    us.sort_by(f64::total_cmp);
    let med = percentile(&us, 0.5);
    let gbps = size as f64 / (med * 1e3);
    ctx.free(w);
    ctx.free(x);
    ctx.free(y);
    ctx.record(
        "matvec",
        a,
        json!({
            "median_us": med,
            "gbps": gbps,
            "fraction_of_640": gbps / 640.0,
            "max_rel_err": rel,
            "weight_mib": size as f64 / (1u64 << 20) as f64,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::max_rel_err;

    #[test]
    fn relative_error_is_scaled_by_the_largest_reference_value() {
        assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        assert!((max_rel_err(&[1.0, 4.1], &[1.0, 4.0]) - 0.025).abs() < 1e-6);
    }
}
