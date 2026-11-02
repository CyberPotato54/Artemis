//! The short-chunk threshold: one prefill chunk of each length from a fresh session,
//! staged and read in place, on the real model at the default cache for `--ctx 65536`, as the gate and `generate` size
//! it (a smaller context leaves more slots and less to stage, which would bias the threshold low). A measurement, not a
//! test (the runner's test-1 block runs every ignored test): `cargo run --release -p engine --example stage_threshold --
//! OUT.json`, with ARTEMIS_MODEL set. The threshold is the shortest length from which staging is at least as fast at
//! every longer length tried.
use engine::{Engine, Options};
use gguf::Gguf;
use qwen36::dims::Dims;
use std::time::Instant;

/// Chunk lengths tried, in tokens. The tiny-model tests assume a threshold above 9, so the grid starts at 16.
const LENS: [usize; 14] = [
    16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048,
];
/// Timed runs of each length and mode, after one untimed.
const REPS: usize = 5;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args()
        .nth(1)
        .ok_or("give the output JSON's path")?;
    let path = std::env::var("ARTEMIS_MODEL")?;
    let g = Gguf::open(std::path::Path::new(&path))?;
    // The cache is sized for a 65,536-position session; the measurement's own session needs only 4,096.
    let (ctx, session) = (65_536, 4096);
    let e = Engine::open(
        &g,
        &Dims::QWEN36,
        &Options {
            ctx,
            ..Options::default()
        },
    )?;
    let toks: Vec<u32> = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/golden/prompts/20-long-option-rs.tokens"
    ))?
    .as_chunks::<4>()
    .0
    .iter()
    .map(|c| u32::from_le_bytes(*c))
    .collect();
    let mut s = e.session(session)?;
    let mut rows = Vec::new();
    for &n in &LENS {
        let mut ms = [Vec::new(), Vec::new()];
        for rep in 0..=REPS {
            for (i, from) in [1usize, usize::MAX].into_iter().enumerate() {
                e.set_stage_from(from);
                s.reset(&e)?;
                let t0 = Instant::now();
                s.prefill(&e, &toks[..n])?;
                if rep > 0 {
                    ms[i].push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
        let (staged, in_place) = (median(&mut ms[0]), median(&mut ms[1]));
        eprintln!("{n:>5} tokens: staged {staged:8.1} ms, in place {in_place:8.1} ms");
        rows.push(serde_json::json!({ "tokens": n, "staged_ms": staged, "in_place_ms": in_place }));
    }
    let faster: Vec<bool> = rows
        .iter()
        .map(|r| r["staged_ms"].as_f64().unwrap() <= r["in_place_ms"].as_f64().unwrap())
        .collect();
    let threshold = (0..LENS.len())
        .find(|&i| faster[i..].iter().all(|&f| f))
        .map(|i| LENS[i]);
    let v = serde_json::json!({
        "ctx": ctx,
        "session": session,
        "cache_slots": e.cache_slots(),
        "staging_mib": e.stats().staging_bytes >> 20,
        "prefill_kernels": e.prefill_kernels(),
        "reps": REPS,
        "rows": rows,
        "threshold": threshold,
    });
    std::fs::write(&out, serde_json::to_vec_pretty(&v)?)?;
    eprintln!("threshold: {threshold:?}");
    Ok(())
}
