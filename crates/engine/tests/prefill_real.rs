//! The prefill-against-decode check on the real model (reads ARTEMIS_MODEL): the long golden prompt's first 4,600
//! tokens, three prefill chunks, against token-by-token decode.
//!
//! Gating: decode is teacher-forced onto prefill's routing, each layer's routed experts and their
//! weights at every position, so the two paths compute the same function and differ only by their arithmetic. The
//! logits' relative L2 must average at most `FORCED_MEAN` over the positions and stay within `FORCED_WORST` at every
//! one, and the final DeltaNet, conv and KV state must be within `STATE`. Measured on 2026-10-09:
//! mean 7.0e-5, worst 5.6e-3 (9 positions above 1e-3), state 2.25e-4; in a dry run, state 3.0e-4, and over the
//! whole 27,416-token prompt mean 9.2e-5, worst 3.4e-2, state 3.2e-4.
//!
//! Two more checks: prefill gives the same bits with chunks of 2,048 and of 500 tokens (its arithmetic depends only on
//! the positions), and the whole 27,416-token prompt prefills with every submit well under the watchdog.
//!
//! Reported, not gating: decode free-running. A router near-tie (256 experts, gaps down to 1e-5) routes one position
//! differently, and the difference then spreads to every later position through attention and the DeltaNet state,
//! as llama.cpp's own backends differ.
//!
//! With ARTEMIS_EVIDENCE_DIR set, each test writes its measurements there, as `<test>.json`, for
//! tools/run-gpu-tests.ps1.
use engine::{chunk_lengths, Engine, Error, Options, SessionState};
use gguf::Gguf;
use qwen36::{
    dims::Dims,
    ops::rel_l2,
    taps::{Key, Tap, TapName},
};

/// Prompt positions compared: three chunks of the long golden prompt.
const N: usize = 4600;
/// Each session's context.
const CTX: usize = 4864;
/// The bounds on the logits, decode forced onto prefill's routing: their mean over the positions, and every one.
const FORCED_MEAN: f64 = 1e-3;
const FORCED_WORST: f64 = 5e-2;
/// The bound on the final state, layer by layer.
const STATE: f64 = 1e-3;
/// `Errs` counts the positions above this.
const OVER: f64 = 1e-3;

fn golden(id: &str) -> Vec<u32> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/golden/prompts")
        .join(format!("{id}.tokens"));
    std::fs::read(&p)
        .unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// Writes `v` to `$ARTEMIS_EVIDENCE_DIR/<name>.json`, when the runner set it.
fn evidence(name: &str, v: &serde_json::Value) {
    if let Ok(dir) = std::env::var("ARTEMIS_EVIDENCE_DIR") {
        let p = std::path::Path::new(&dir).join(format!("{name}.json"));
        std::fs::write(&p, serde_json::to_vec_pretty(v).unwrap())
            .unwrap_or_else(|e| panic!("writing {}: {e}", p.display()));
    }
}

/// Forces each layer's routed experts and weights to prefill's, position by position: `ids[i]` and `w[i]` hold
/// position `pos0 + i`'s, `[n_layer][n_expert_used]`.
struct ForceRoutes {
    ids: Vec<Vec<f32>>,
    w: Vec<Vec<f32>>,
    pos0: usize,
    k: usize,
}

impl Tap for ForceRoutes {
    fn tap(&mut self, _: TapName, _: u32, _: &[f32]) {}

    fn forced(&self, name: TapName, pos: u32) -> Option<&[f32]> {
        let i = pos as usize - self.pos0;
        let at = |l: u16| l as usize * self.k..(l as usize + 1) * self.k;
        match name {
            TapName::Layer(l, Key::TopK) => Some(&self.ids[i][at(l)]),
            TapName::Layer(l, Key::TopKW) => Some(&self.w[i][at(l)]),
            _ => None,
        }
    }
}

/// The worst relative L2 over every layer's two state buffers.
fn state_err(a: &SessionState, b: &SessionState) -> f64 {
    a.layers
        .iter()
        .zip(&b.layers)
        .flat_map(|(x, y)| [rel_l2(&x.0, &y.0), rel_l2(&x.1, &y.1)])
        // A NaN must win: `f64::max` would drop it and let a broken state pass.
        .fold(0f64, |m, e| if e.is_nan() || e > m { e } else { m })
}

/// A measurement for the evidence file: JSON has no NaN or infinity, so those are written as the strings "NaN" and
/// "inf" (not as `null`), and the record shows what happened.
fn num(x: f64) -> serde_json::Value {
    if x.is_finite() {
        serde_json::json!(x)
    } else {
        serde_json::json!(x.to_string())
    }
}

/// Worst, where, mean, and how many above `OVER`, of per-position errors. `worst_at` is the first NaN if there is one
/// (a NaN is the worst, and later ones don't replace it).
#[derive(Default)]
struct Errs {
    worst: f64,
    worst_at: usize,
    sum: f64,
    n: usize,
    over: usize,
    first_over: Option<usize>,
}

impl Errs {
    fn add(&mut self, pos: usize, e: f64) {
        if e > self.worst || (e.is_nan() && !self.worst.is_nan()) {
            self.worst = e;
            self.worst_at = pos;
        }
        self.sum += e;
        self.n += 1;
        if e > OVER {
            self.over += 1;
            self.first_over.get_or_insert(pos);
        }
    }

    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "worst": num(self.worst),
            "worst_at": self.worst_at,
            "mean": num(self.sum / self.n.max(1) as f64),
            "over_1e-3": self.over,
            "first_over_1e-3": self.first_over,
        })
    }
}

#[test]
#[ignore = "needs ARTEMIS_MODEL and the GPU"]
fn prefill_matches_decode_on_a_long_prompt() {
    let path = std::env::var("ARTEMIS_MODEL").expect("set ARTEMIS_MODEL to the model's GGUF");
    let g = Gguf::open(std::path::Path::new(&path)).unwrap();
    let d = Dims::QWEN36;
    let k = d.n_expert_used;
    let toks: Vec<u32> = golden("20-long-option-rs").into_iter().take(N).collect();
    assert_eq!(toks.len(), N);
    // Four sessions share the engine: the cache is sized for their contexts, so the engine is opened for six
    // contexts' worth: four sessions' KV plus their recurrent state (~63 MiB each, which `ctx` doesn't count).
    let e = Engine::open(
        &g,
        &d,
        &Options {
            ctx: 6 * CTX,
            ..Options::default()
        },
    )
    .unwrap();
    // `routing` prefills each chunk to read its routing; `pre` prefills it again with every position's logits, which
    // are the same bits, while `forced` and `free` decode the same positions.
    let mut routing = e.session(CTX).unwrap();
    let mut pre = e.session(CTX).unwrap();
    let mut forced = e.session(CTX).unwrap();
    let mut free = e.session(CTX).unwrap();
    let (mut ef, mut efree) = (Errs::default(), Errs::default());
    let mut flips = 0usize;
    let mut first_flip: Option<(usize, usize)> = None;
    let chunks = chunk_lengths(0, N, e.prefill_chunk(), e.attn_pairs());
    let t0 = std::time::Instant::now();
    let mut at = 0;
    for &t in &chunks {
        routing.prefill(&e, &toks[at..at + t]).unwrap();
        let (ids, w) = (e.last_prefill_routes(), e.last_prefill_route_weights());
        let per_pos = |i: usize, v: &dyn Fn(usize) -> f32| -> Vec<f32> {
            (0..d.n_layer)
                .flat_map(|l| (0..k).map(move |j| (l * t + i) * k + j))
                .map(v)
                .collect()
        };
        let mut force = ForceRoutes {
            ids: (0..t).map(|i| per_pos(i, &|x| ids[x] as f32)).collect(),
            w: (0..t).map(|i| per_pos(i, &|x| w[x])).collect(),
            pos0: at,
            k,
        };
        pre.prefill_each(&e, &toks[at..at + t], 0, |i, l| {
            let p = at + i;
            let lf = forced.decode_tapped(&e, toks[p], &mut force)?;
            ef.add(p, rel_l2(l, &lf));
            let lr = free.decode(&e, toks[p])?;
            efree.add(p, rel_l2(l, &lr));
            // Where free decode routed differently from prefill.
            let r = e.last_routes();
            for layer in 0..d.n_layer {
                let mut a: Vec<usize> = r[layer * k..(layer + 1) * k].to_vec();
                let mut b: Vec<usize> = ids[(layer * t + i) * k..(layer * t + i + 1) * k].to_vec();
                a.sort();
                b.sort();
                if a != b {
                    flips += 1;
                    first_flip.get_or_insert((p, layer));
                }
            }
            Ok::<(), Error>(())
        })
        .unwrap();
        at += t;
        eprintln!(
            "through position {at}: forced worst {:.3e}, free worst {:.3e}; {:.0} s",
            ef.worst,
            efree.worst,
            t0.elapsed().as_secs_f64()
        );
    }
    let ps = pre.state(&e).unwrap();
    let (sf, sr) = (
        state_err(&ps, &forced.state(&e).unwrap()),
        state_err(&ps, &free.state(&e).unwrap()),
    );
    assert_eq!(ef.n, N, "every position compared");
    let mean = ef.sum / ef.n as f64;
    let pass = mean <= FORCED_MEAN && ef.worst <= FORCED_WORST && sf <= STATE;
    let verdict = serde_json::json!({
        "status": if pass { "pass" } else { "fail" },
        "input": "20-long-option-rs",
        "positions": N,
        "chunks": chunks,
        "forced": ef.json(),
        "forced_state": num(sf),
        "mean_bound": FORCED_MEAN,
        "worst_bound": FORCED_WORST,
        "state_bound": STATE,
        "free": efree.json(),
        "free_state": num(sr),
        "free_routing_differences": flips,
        "free_first_routing_difference": first_flip,
        "seconds": t0.elapsed().as_secs_f64(),
    });
    eprintln!("{verdict:#}");
    evidence("vs-decode", &verdict);
    assert!(
        mean <= FORCED_MEAN,
        "forced onto prefill's routing, the logits differ by {mean:.3e} on average, over {FORCED_MEAN:e}"
    );
    assert!(
        ef.worst <= FORCED_WORST,
        "forced onto prefill's routing, position {}'s logits differ by {:.3e}, over {FORCED_WORST:e}",
        ef.worst_at,
        ef.worst
    );
    assert!(
        sf <= STATE,
        "the final state differs by {sf:.3e}, over {STATE:e}"
    );
}

/// Every submit must stay well under Windows' 2 s watchdog.
const LONGEST_SUBMIT_US: u64 = 300_000;

#[test]
#[ignore = "needs ARTEMIS_MODEL and the GPU"]
fn the_long_prompt_prefills_with_every_submit_well_under_the_watchdog() {
    // The 27,416-token golden prompt at the default cache: chunks shorten as attention's work grows deep in the
    // prompt (`ATTN_PAIRS` for the f32 kernel, `ATTN_PAIRS_CM` for `attn_prefill_cm`), so no layer's submit nears the
    // watchdog.
    let path = std::env::var("ARTEMIS_MODEL").expect("set ARTEMIS_MODEL to the model's GGUF");
    let g = Gguf::open(std::path::Path::new(&path)).unwrap();
    let toks = golden("20-long-option-rs");
    let ctx = 32_768;
    let e = Engine::open(
        &g,
        &Dims::QWEN36,
        &Options {
            ctx,
            ..Options::default()
        },
    )
    .unwrap();
    let mut s = e.session(ctx).unwrap();
    let t0 = std::time::Instant::now();
    s.prefill(&e, &toks).unwrap();
    let secs = t0.elapsed().as_secs_f64();
    let st = e.stats();
    let longest = st.longest_submit_us;
    let v = serde_json::json!({
        "status": if longest < LONGEST_SUBMIT_US { "pass" } else { "fail" },
        "input": "20-long-option-rs",
        "positions": toks.len(),
        "chunks": chunk_lengths(0, toks.len(), e.prefill_chunk(), e.attn_pairs()).len(),
        "seconds": secs,
        "tok_s": toks.len() as f64 / secs,
        "longest_submit_ms": longest as f64 / 1e3,
        "bound_ms": LONGEST_SUBMIT_US as f64 / 1e3,
        "cache_slots": e.cache_slots(),
        "from_vram": st.from_vram().map_or(serde_json::Value::Null, |f| serde_json::json!(f)),
        "staged_steps": st.staged_steps,
    });
    eprintln!("{v:#}");
    evidence("long-prompt", &v);
    assert!(
        longest < LONGEST_SUBMIT_US,
        "a submit took {} ms",
        longest / 1000
    );
}

#[test]
#[ignore = "needs ARTEMIS_MODEL and the GPU"]
fn prefill_is_bit_identical_whatever_the_chunk_size() {
    // The first 4,600 tokens with chunks of 2,048 and of 500: every position's logits (FNV-1a of their bytes) and the
    // final state, bit for bit. The two engines' caches differ too (prefill's buffers differ in size).
    let path = std::env::var("ARTEMIS_MODEL").expect("set ARTEMIS_MODEL to the model's GGUF");
    let g = Gguf::open(std::path::Path::new(&path)).unwrap();
    let toks: Vec<u32> = golden("20-long-option-rs").into_iter().take(N).collect();
    let run = |chunk: usize| {
        let e = Engine::open(
            &g,
            &Dims::QWEN36,
            &Options {
                ctx: CTX,
                prefill_chunk: Some(chunk),
                ..Options::default()
            },
        )
        .unwrap();
        let mut s = e.session(CTX).unwrap();
        let mut fp = Vec::with_capacity(N);
        s.prefill_each(&e, &toks, 0, |_, l| {
            let bytes: Vec<u8> = l.iter().flat_map(|x| x.to_le_bytes()).collect();
            fp.push(qwen36::testing::fnv1a64(&bytes));
            Ok::<(), Error>(())
        })
        .unwrap();
        (fp, s.state(&e).unwrap(), e.cache_slots())
    };
    let (a, sa, na) = run(2048);
    let (b, sb, nb) = run(500);
    assert_eq!(
        (a.len(), b.len()),
        (N, N),
        "a logits fingerprint for every position"
    );
    let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    let bits = |x: &[f32], y: &[f32]| {
        x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
    };
    let same_state = sa.layers.len() == sb.layers.len()
        && sa
            .layers
            .iter()
            .zip(&sb.layers)
            .all(|(x, y)| bits(&x.0, &y.0) && bits(&x.1, &y.1));
    let v = serde_json::json!({
        "status": if differ == 0 && same_state { "pass" } else { "fail" },
        "input": "20-long-option-rs",
        "positions": N,
        "chunk_sizes": [2048, 500],
        "cache_slots": [na, nb],
        "positions_differing": differ,
        "state_identical": same_state,
    });
    eprintln!("{v:#}");
    evidence("chunks", &v);
    assert_eq!(
        differ, 0,
        "positions whose logits differ between chunks of 2,048 and 500"
    );
    assert!(
        same_state,
        "the final state differs between chunks of 2,048 and 500"
    );
}
