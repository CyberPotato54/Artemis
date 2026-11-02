//! The cost of `logprobs` over a full vocabulary: 248,320 logits from the engine's own RNG,
//! top 20, one untimed call and then 200 timed, the median printed in microseconds. It runs on the CPU and touches no
//! GPU. A measurement, not a test: `cargo run --release -p engine --example logprobs_cost`.
use engine::{logprobs, Rng};
use std::{hint::black_box, time::Instant};

/// Qwen3.6's vocabulary, in logits.
const VOCAB: usize = 248_320;
/// Timed calls, after one untimed.
const REPS: usize = 200;

fn main() {
    let mut rng = Rng::new(1);
    let logits: Vec<f32> = (0..VOCAB)
        .map(|_| (rng.unit() as f32 - 0.5) * 20.0)
        .collect();
    black_box(logprobs(black_box(&logits), 0, 20));
    let mut us: Vec<f64> = (0..REPS)
        .map(|_| {
            let t = Instant::now();
            black_box(logprobs(black_box(&logits), 0, 20));
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    us.sort_by(f64::total_cmp);
    println!(
        "logprobs over {VOCAB} logits, top 20: median {:.0} us of {REPS} (min {:.0}, max {:.0})",
        us[us.len() / 2],
        us[0],
        us[us.len() - 1]
    );
}
