//! The speed check of the split-f16 cooperative-matrix kernels against today's f32 ones: prefill's widest dense
//! shape and one full chunk's grouped experts. A measurement, not a test (the runner's test-1 block runs every ignored
//! test): `cargo run --release -p kernels --example cm_speed`. Each figure is the best of five submits of the kernels
//! alone, inputs uploaded once, timed on the host around submit and wait.
use kernels::{
    dispatch::{moe_tiles, slot_entry, MatW, SlotLayout, STATUS_CLEAR, WHERE_VRAM},
    testing::{rand_f32, random_weights, Gpu},
    twin, Kernels,
};
use qwen36::{quant::WType, testing::Rng};
use std::time::Instant;

/// The best of five submits of `f`, in ms.
fn best(g: &Gpu, f: impl Fn(&Kernels, &mut vk::Recorder)) -> f64 {
    (0..5)
        .map(|_| {
            let t0 = Instant::now();
            g.run(&f);
            t0.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

/// DeltaNet's QKV projection over a full chunk: 8192 × 2048 Q8_0, 2,048 tokens. An earlier measurement gave 6.2 ms for
/// the f32 kernel and 2.6 ms for the split cooperative-matrix one, on the GPU's clock.
fn dense(g: &Gpu, rng: &mut Rng) {
    let (rows, cols, tokens) = (8192, 2048, 2048);
    let w = g.bytes(&random_weights(rng, WType::Q8_0, rows, cols));
    let x = g.f32s(&rand_f32(rng, tokens * cols, -1.0, 1.0));
    let y = g.zeros(tokens * rows);
    let m = MatW {
        addr: w.addr,
        ty: WType::Q8_0,
        rows,
        cols,
    };
    let f = best(g, |k, r| k.matmul(r, m, x.addr, y.addr, tokens));
    let c = best(g, |k, r| k.matmul_cm(r, m, x.addr, y.addr, tokens));
    println!(
        "dense, 8192x2048 Q8_0, 2,048 tokens: f32 {f:.2} ms, split cooperative matrices {c:.2} ms"
    );
}

/// One layer of a 2,048-token chunk: 16,384 entries over 256 experts in VRAM, Q4_K gate and up, Q5_K down, grouped once
/// on the CPU. An earlier measurement gave 20.6 ms for today's kernels and 9.4 ms for the split cooperative-matrix
/// ones, on the GPU's clock.
fn grouped(g: &Gpu, rng: &mut Rng) {
    let lay = SlotLayout {
        gate: WType::Q4K,
        up: WType::Q4K,
        down: WType::Q5K,
        n_embd: 2048,
        n_ff: 512,
    };
    let (n, tokens, k) = (256, 2048, 8);
    let mut packed = Vec::with_capacity(n * lay.bytes());
    for _ in 0..n {
        packed.extend(random_weights(rng, lay.gate, lay.n_ff, lay.n_embd));
        packed.extend(random_weights(rng, lay.up, lay.n_ff, lay.n_embd));
        packed.extend(random_weights(rng, lay.down, lay.n_embd, lay.n_ff));
    }
    let w = g.bytes(&packed);
    let table: Vec<u8> = (0..n)
        .flat_map(|e| slot_entry(w.addr + (e * lay.bytes()) as u64, WHERE_VRAM))
        .collect();
    let table = g.bytes(&table);
    let ids: Vec<f32> = (0..tokens)
        .flat_map(|t| (0..k).map(move |j| ((t * 37 + j * 31 + (t * j) % 7) % n) as f32))
        .collect();
    let max = moe_tiles(ids.len(), n);
    let (list, tiles) = twin::moe::moe_group(&ids, n, max);
    let u32s = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let (list, tiles) = (g.bytes(&u32s(&list)), g.bytes(&u32s(tiles.as_flattened())));
    let x = g.f32s(&rand_f32(rng, tokens * lay.n_embd, -1.0, 1.0));
    let (act, ye) = (
        g.zeros(ids.len() * lay.n_ff),
        g.zeros(ids.len() * lay.n_embd),
    );
    let status = g.bytes(&STATUS_CLEAR);
    let f = best(g, |kk, r| {
        kk.experts_matmul_gate_up(
            r,
            table.addr,
            0,
            n,
            list.addr,
            tiles.addr,
            max,
            x.addr,
            act.addr,
            status.addr,
            &lay,
            k,
        );
        r.barrier();
        kk.experts_matmul_down(
            r,
            table.addr,
            0,
            n,
            list.addr,
            tiles.addr,
            max,
            act.addr,
            ye.addr,
            status.addr,
            &lay,
        );
    });
    let c = best(g, |kk, r| {
        kk.experts_matmul_cm_gate_up(
            r,
            table.addr,
            0,
            n,
            list.addr,
            tiles.addr,
            max,
            x.addr,
            act.addr,
            status.addr,
            &lay,
            k,
        );
        r.barrier();
        kk.experts_matmul_cm_down(
            r,
            table.addr,
            0,
            n,
            list.addr,
            tiles.addr,
            max,
            act.addr,
            ye.addr,
            status.addr,
            &lay,
        );
    });
    println!("grouped experts, one 2,048-token layer: f32 {f:.1} ms, split cooperative matrices {c:.1} ms");
}

fn main() {
    let g = Gpu::new();
    assert!(g.k.coop(), "this device runs no cooperative-matrix kernels");
    let mut rng = Rng::new(49);
    dense(&g, &mut rng);
    grouped(&g, &mut rng);
}
