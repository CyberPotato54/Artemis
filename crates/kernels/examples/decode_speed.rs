//! The non-expert decode kernels on the GPU clock, at the real model's shapes: every dense matvec of a layer by tensor
//! and type (and the output head), the DeltaNet step (`conv_step`, `gdn_step`), the router, `rmsnorm` and short-context
//! attention, each in its decode form (t = 1) and its token form (`matvec_tokens`, `conv_seq`, `gdn_seq`,
//! `router_topk_tokens`, `rmsnorm_tokens`, `attn_decode_tokens`) at the verify step's token counts. A measurement, not
//! a test: `cargo run --release -p kernels --example decode_speed -- --model GGUF --json FILE [--reps 1000]
//! [--tokens 1,2,3,4,5,16]` (`--model` defaults to `ARTEMIS_MODEL`).
//!
//! Method for a figure worth quoting (`expert_speed`'s): run it twice (the default is 1,000 repetitions) and compare
//! the two runs' best totals. If every one agrees within 1%, keep the second run. If not, run again, up to four runs in
//! all, and keep the second run of the closest pair.
//!
//! The weights are the GGUF's own: for each dense tensor the example groups the layers that hold it by type, prints the
//! groups, and times the first layer of each. The shapes come from `Dims::QWEN36` (`dense_table`) and are checked
//! against the GGUF's tensors. Inputs are seeded (`Rng::new(83)`). Each dispatch is timed by the device's timestamps
//! (`Gpu::run_timed`). Each repetition's command buffer starts with a 256 MiB buffer copy and a barrier, which evict
//! the GPU caches (the 64 MiB Infinity Cache included) as a decode step's other weight reads do; the copy's own time is
//! left out. Kernels that update a state (`conv_*`, `gdn_*`) start each repetition with a copy of the same untouched
//! state, so every repetition's outputs are equal. One untimed run comes first.
//!
//! The JSON holds `driver`, `reps`, `tokens`, then a `cases` array of `{group, name, types, layers, rows, cols, t,
//! n_ctx, kernels, total, bits}`. `kernels` holds per dispatch `{kernel, us: {best, median}, bytes, gbps: {best,
//! median}, pct_of_640: {best, median}, floor_us}`; `total` the same over each repetition's sum of the dispatches (not
//! the sum of the bests). `bytes` counts the dominant data a dispatch moves: the weights for a matvec (read once for
//! any t), the KV caches for attention, the state read and written for `gdn_*` and `conv_*`, the inputs and outputs for
//! `rmsnorm` and the router. GB/s is bytes over time, `pct_of_640` that over 640 GB/s, and `floor_us` the bytes at 640
//! GB/s. `bits` is the FNV-1a 64 of the outputs' bytes (and of the updated state). Every f32 output is asserted finite
//! and the first and last repetitions' outputs equal bit for bit.
use gguf::Gguf;
use kernels::{
    dispatch::{attn_splits, attn_tokens_kernel, MatW},
    testing::{f32_bytes, json_escaped, rand_f32, Gpu},
    tokens::MAX_TOKENS,
    twin, Kernels,
};
use qwen36::{
    dims::Dims,
    model::{Mixer, Model, Weight},
    quant::WType,
    testing::{fnv1a64, Rng},
};
use std::fmt::Write as _;
use vk::{Recorder, Slice};

/// The memory bandwidth the floor is taken at, in bytes per second.
const FLOOR_BYTES_PER_S: f64 = 640e9;
/// The cached positions the short-context attention cases take.
const SHORT_CTX: [usize; 2] = [512, 2048];

#[derive(Clone, Copy, PartialEq, Debug)]
enum Holder {
    /// Every main layer.
    Every,
    Delta,
    Attn,
    /// The output head, once.
    Head,
}

/// One dense matvec: its name, the layers that hold it and its shape (`rows` outputs from `cols` inputs).
#[derive(Clone, Copy, Debug)]
struct Dense {
    name: &'static str,
    holder: Holder,
    rows: usize,
    cols: usize,
}

/// Every dense tensor a decode step multiplies, in the order a layer runs them (semantics reference section 1).
fn dense_table(d: &Dims) -> Vec<Dense> {
    let t = |name, holder, rows, cols| Dense {
        name,
        holder,
        rows,
        cols,
    };
    let (e, ff) = (d.n_embd, d.n_ff_shexp);
    let attn_w = d.n_head * d.head_dim;
    vec![
        t("delta.qkv", Holder::Delta, d.conv_dim(), e),
        t("delta.gate", Holder::Delta, d.ssm_inner, e),
        t("delta.beta", Holder::Delta, d.ssm_dt_rank, e),
        t("delta.alpha", Holder::Delta, d.ssm_dt_rank, e),
        t("delta.out", Holder::Delta, e, d.ssm_inner),
        t("attn.q", Holder::Attn, 2 * attn_w, e),
        t("attn.k", Holder::Attn, d.n_head_kv * d.head_dim, e),
        t("attn.v", Holder::Attn, d.n_head_kv * d.head_dim, e),
        t("attn.out", Holder::Attn, e, attn_w),
        t("router", Holder::Every, d.n_expert, e),
        t("shared.gate_inp", Holder::Every, 1, e),
        t("shared.gate", Holder::Every, ff, e),
        t("shared.up", Holder::Every, ff, e),
        t("shared.down", Holder::Every, e, ff),
        t("output", Holder::Head, d.n_vocab, e),
    ]
}

/// `name`'s weight in main layer `l` (or the output head).
fn weight<'a>(m: &Model<'a>, l: usize, name: &str) -> Weight<'a> {
    let layer = &m.layers[l];
    match (name, &layer.mixer) {
        ("delta.qkv", Mixer::DeltaNet(x)) => x.qkv,
        ("delta.gate", Mixer::DeltaNet(x)) => x.gate,
        ("delta.beta", Mixer::DeltaNet(x)) => x.beta,
        ("delta.alpha", Mixer::DeltaNet(x)) => x.alpha,
        ("delta.out", Mixer::DeltaNet(x)) => x.out,
        ("attn.q", Mixer::Attention(x)) => x.q,
        ("attn.k", Mixer::Attention(x)) => x.k,
        ("attn.v", Mixer::Attention(x)) => x.v,
        ("attn.out", Mixer::Attention(x)) => x.out,
        ("router", _) => layer.moe.router,
        ("shared.gate_inp", _) => layer.shared.gate_inp,
        ("shared.gate", _) => layer.shared.gate,
        ("shared.up", _) => layer.shared.up,
        ("shared.down", _) => layer.shared.down,
        ("output", _) => m.output,
        _ => panic!("no tensor {name} in layer {l}"),
    }
}

fn holds(m: &Model, l: usize, h: Holder) -> bool {
    matches!(
        (h, &m.layers[l].mixer),
        (Holder::Every, _)
            | (Holder::Delta, Mixer::DeltaNet(_))
            | (Holder::Attn, Mixer::Attention(_))
    )
}

/// Random f16 caches of `n_ctx` positions of `width` values, as bytes.
fn random_cache(rng: &mut Rng, n_ctx: usize, width: usize) -> Vec<u8> {
    (0..n_ctx * width)
        .flat_map(|_| twin::f16_bits(rng.uniform(-1.0, 1.0)).to_le_bytes())
        .collect()
}

fn best_and_median(v: &[f64]) -> (f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let median = if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    };
    (s[0], median)
}

struct Case {
    group: &'static str,
    name: String,
    types: String,
    layers: Vec<usize>,
    rows: usize,
    cols: usize,
    t: usize,
    n_ctx: usize,
    /// Per dispatch: its kernel, the bytes it moves and each repetition's µs.
    kernels: Vec<(&'static str, u64, Vec<f64>)>,
    bits: u64,
}

impl Case {
    fn totals(&self) -> Vec<f64> {
        (0..self.kernels[0].2.len())
            .map(|i| self.kernels.iter().map(|k| k.2[i]).sum())
            .collect()
    }
    fn bytes(&self) -> u64 {
        self.kernels.iter().map(|k| k.1).sum()
    }
}

/// `{best, median}` of µs, and of GB/s and percent of 640 for `bytes`.
fn stats_json(times: &[f64], bytes: u64) -> String {
    let (b, m) = best_and_median(times);
    let gbps = |us: f64| bytes as f64 / us / 1e3;
    format!(
        "\"us\": {{\"best\": {b:.2}, \"median\": {m:.2}}}, \"bytes\": {bytes}, \
         \"gbps\": {{\"best\": {:.1}, \"median\": {:.1}}}, \
         \"pct_of_640\": {{\"best\": {:.1}, \"median\": {:.1}}}, \"floor_us\": {:.2}",
        gbps(b),
        gbps(m),
        gbps(b) / 6.4,
        gbps(m) / 6.4,
        bytes as f64 / FLOOR_BYTES_PER_S * 1e6
    )
}

/// What one case's timing needs from the harness.
struct Bench<'a> {
    g: &'a Gpu,
    reps: usize,
    evict: (Slice, Slice),
}

/// The outputs a case hashes: f32 buffers (asserted finite) and raw ones (ids).
struct Outs {
    f32s: Vec<Slice>,
    raw: Vec<Slice>,
}

impl Bench<'_> {
    /// Times `build`'s `n_disp` dispatches `reps` times (an untimed run first), each run after the cache-evicting copy
    /// and after `resets` (`(pristine, live)` state copies). Returns the per-dispatch times and the outputs' hash.
    fn run(
        &self,
        label: &str,
        resets: &[(Slice, Slice)],
        outs: &Outs,
        n_disp: usize,
        build: &dyn Fn(&Kernels, &mut Recorder),
    ) -> (Vec<Vec<f64>>, u64) {
        let g = self.g;
        let run = |evict: bool| {
            g.run_timed(|k, r| {
                if evict {
                    r.copy(self.evict.0, self.evict.1);
                }
                for &(from, to) in resets {
                    r.copy(from, to);
                }
                r.barrier();
                build(k, r);
            })
        };
        let outputs = || {
            let mut o = Vec::new();
            for &s in &outs.f32s {
                let v = g.read(s);
                assert!(
                    v.iter().all(|x| x.is_finite()),
                    "{label}: a non-finite output"
                );
                o.extend(f32_bytes(&v));
            }
            for &s in outs.raw.iter().chain(resets.iter().map(|(_, live)| live)) {
                o.extend(g.dev.download(s).unwrap());
            }
            o
        };
        run(false);
        let first = outputs();
        let mut times = vec![Vec::with_capacity(self.reps); n_disp];
        for _ in 0..self.reps {
            let ts = run(true);
            assert_eq!(ts.len(), n_disp, "{label}: {n_disp} dispatches, not {ts:?}");
            for (v, t) in times.iter_mut().zip(ts) {
                v.push(t);
            }
        }
        let last = outputs();
        assert!(
            first == last,
            "{label}: the outputs changed between repetitions"
        );
        (times, fnv1a64(&last))
    }
}

fn main() {
    let (mut json, mut reps, mut model) = (None, 1000usize, std::env::var("ARTEMIS_MODEL").ok());
    let mut tokens = vec![1usize, 2, 3, 4, 5, 16];
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--json" => json = Some(args.next().expect("--json needs a file")),
            "--model" => model = Some(args.next().expect("--model needs a GGUF")),
            "--reps" => {
                reps = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .filter(|&n| n >= 1)
                    .expect("--reps needs a number of at least 1")
            }
            "--tokens" => {
                tokens = args
                    .next()
                    .expect("--tokens needs a list such as 1,2,3")
                    .split(',')
                    .map(|v| v.parse().expect("--tokens holds numbers"))
                    .collect();
                assert!(
                    !tokens.is_empty() && tokens.iter().all(|t| (1..=MAX_TOKENS).contains(t)),
                    "--tokens takes counts from 1 to {MAX_TOKENS}"
                );
            }
            other => panic!(
                "unknown argument {other}; usage: --model GGUF --json FILE [--reps N] [--tokens 1,2,3]"
            ),
        }
    }
    let model = model.expect("name the model's GGUF with --model or ARTEMIS_MODEL");
    let gg = Gguf::open(std::path::Path::new(&model))
        .unwrap_or_else(|e| panic!("opening {model}: {e:?}"));
    let d = Dims::QWEN36;
    let m = Model::open(&gg, &d).unwrap_or_else(|e| panic!("{model} is not the model: {e:?}"));
    let max_t = *tokens.iter().max().unwrap();

    let g = Gpu::new();
    assert!(g.dev.can_profile(), "this device has no timestamps");
    let bench = Bench {
        g: &g,
        reps,
        evict: (
            g.bytes(&vec![0x5a; 256 << 20]),
            g.bytes(&vec![0; 256 << 20]),
        ),
    };
    let mut rng = Rng::new(83);
    let mut cases: Vec<Case> = Vec::new();
    let mut report = |c: Case| {
        let (tb, _) = best_and_median(&c.totals());
        println!(
            "{} {} {} t={} n_ctx={}: best {tb:.1} us, {:.0} GB/s, {:.1}% of 640",
            c.group,
            c.name,
            c.types,
            c.t,
            c.n_ctx,
            c.bytes() as f64 / tb / 1e3,
            c.bytes() as f64 / tb / 1e3 / 6.4
        );
        cases.push(c);
    };

    // Dense matvecs: the first layer of each (tensor, type) group, at each token count.
    let gx = g.f32s(&rand_f32(
        &mut rng,
        max_t * d.n_embd.max(d.ssm_inner),
        -1.0,
        1.0,
    ));
    for dn in dense_table(&d) {
        let layers: Vec<usize> = match dn.holder {
            Holder::Head => vec![0],
            h => (0..m.layers.len()).filter(|&l| holds(&m, l, h)).collect(),
        };
        let mut groups: Vec<(WType, Vec<usize>)> = Vec::new();
        for &l in &layers {
            let ty = weight(&m, l, dn.name).ty;
            match groups.iter_mut().find(|(t, _)| *t == ty) {
                Some((_, ls)) => ls.push(l),
                None => groups.push((ty, vec![l])),
            }
        }
        for (ty, ls) in groups {
            let w = weight(&m, ls[0], dn.name);
            assert_eq!(
                (w.rows, w.cols),
                (dn.rows, dn.cols),
                "{}: the GGUF's shape is not the table's",
                dn.name
            );
            let layers = if dn.holder == Holder::Head {
                vec![]
            } else {
                ls
            };
            println!("{} {}: layers {layers:?}", dn.name, ty.name());
            let mw = MatW {
                addr: g.bytes(w.bytes).addr,
                ty,
                rows: dn.rows,
                cols: dn.cols,
            };
            let bytes = (dn.rows * ty.row_bytes(dn.cols)) as u64;
            for &t in &tokens {
                let y = g.zeros(t * dn.rows);
                let label = format!("{} {} t={t}", dn.name, ty.name());
                let (times, bits) = bench.run(
                    &label,
                    &[],
                    &Outs {
                        f32s: vec![y],
                        raw: vec![],
                    },
                    1,
                    &|k, r| {
                        if t == 1 {
                            k.matvec(r, mw, gx.addr, y.addr);
                        } else {
                            k.matvec_tokens(r, mw, gx.addr, y.addr, t, dn.cols, dn.rows);
                        }
                    },
                );
                report(Case {
                    group: "dense",
                    name: dn.name.to_string(),
                    types: ty.name().to_string(),
                    layers: layers.clone(),
                    rows: dn.rows,
                    cols: dn.cols,
                    t,
                    n_ctx: 0,
                    kernels: vec![(
                        if t == 1 { "matvec" } else { "matvec_tokens" },
                        bytes,
                        times.into_iter().next().unwrap(),
                    )],
                    bits,
                });
            }
        }
    }

    // The router's top-k over a layer's logits (the router matvec is a dense case above).
    let (ne, nk) = (d.n_expert, d.n_expert_used);
    let logits = g.f32s(&rand_f32(&mut rng, max_t * ne, -4.0, 4.0));
    for &t in &tokens {
        let (probs, ids, wts) = (g.zeros(t * ne), g.zeros(t * nk), g.zeros(t * nk));
        let (times, bits) = bench.run(
            &format!("router_topk t={t}"),
            &[],
            &Outs {
                f32s: vec![probs, wts],
                raw: vec![ids],
            },
            1,
            &|k, r| {
                if t == 1 {
                    k.router_topk(r, logits.addr, probs.addr, ids.addr, wts.addr, ne, nk);
                } else {
                    k.router_topk_tokens(r, logits.addr, probs.addr, ids.addr, wts.addr, ne, nk, t);
                }
            },
        );
        report(Case {
            group: "router",
            name: "router_topk".into(),
            types: "f32".into(),
            layers: vec![],
            rows: ne,
            cols: nk,
            t,
            n_ctx: 0,
            kernels: vec![(
                if t == 1 {
                    "router_topk"
                } else {
                    "router_topk_tokens"
                },
                (t * (ne + ne + 2 * nk) * 4) as u64,
                times.into_iter().next().unwrap(),
            )],
            bits,
        });
    }

    // rmsnorm over the hidden row.
    let e = d.n_embd;
    let nw = g.f32s(&rand_f32(&mut rng, e, 0.5, 1.5));
    let nx = g.f32s(&rand_f32(&mut rng, max_t * e, -1.0, 1.0));
    for &t in &tokens {
        let y = g.zeros(t * e);
        let (times, bits) = bench.run(
            &format!("rmsnorm t={t}"),
            &[],
            &Outs {
                f32s: vec![y],
                raw: vec![],
            },
            1,
            &|k, r| k.rmsnorm_tokens(r, nx.addr, nw.addr, y.addr, e, d.rms_eps, t),
        );
        report(Case {
            group: "rmsnorm",
            name: "rmsnorm".into(),
            types: "f32".into(),
            layers: vec![],
            rows: t,
            cols: e,
            t,
            n_ctx: 0,
            kernels: vec![(
                if t == 1 { "rmsnorm" } else { "rmsnorm_tokens" },
                (t * 2 * e * 4) as u64,
                times.into_iter().next().unwrap(),
            )],
            bits,
        });
    }

    // The DeltaNet step: conv_step / conv_seq over the qkv channels, then gdn_step / gdn_seq.
    let (cd, kw, kd) = (d.conv_dim(), d.ssm_conv, d.key_dim());
    let (nv, dk, dv, ng) = (d.ssm_dt_rank, d.ssm_d_state, d.ssm_d_state, d.ssm_n_group);
    let conv_w = g.f32s(&rand_f32(&mut rng, cd * kw, -0.5, 0.5));
    let conv_x = g.f32s(&rand_f32(&mut rng, max_t * cd, -1.0, 1.0));
    let conv_state0 = g.f32s(&rand_f32(&mut rng, cd * (kw - 1), -1.0, 1.0));
    let conv_state = g.zeros(cd * (kw - 1));
    let gdn_state0 = g.f32s(&rand_f32(&mut rng, nv * dk * dv, -0.1, 0.1));
    let gdn_state = g.zeros(nv * dk * dv);
    let gdn_in = g.f32s(&rand_f32(&mut rng, max_t * cd, -0.1, 0.1));
    let gdn_g = g.f32s(&rand_f32(&mut rng, max_t * nv, -1.0, -0.1));
    let gdn_beta = g.f32s(&rand_f32(&mut rng, max_t * nv, 0.1, 0.9));
    for &t in &tokens {
        let y = g.zeros(t * cd);
        let (times, bits) = bench.run(
            &format!("conv t={t}"),
            &[(conv_state0, conv_state)],
            &Outs {
                f32s: vec![y],
                raw: vec![],
            },
            1,
            &|k, r| {
                if t == 1 {
                    k.conv_step(r, conv_x.addr, conv_w.addr, conv_state.addr, y.addr, cd, kw);
                } else {
                    k.conv_seq(
                        r,
                        conv_x.addr,
                        conv_w.addr,
                        conv_state.addr,
                        y.addr,
                        cd,
                        kw,
                        t,
                        true,
                    );
                }
            },
        );
        report(Case {
            group: "conv",
            name: "conv".into(),
            types: "f32".into(),
            layers: vec![],
            rows: cd,
            cols: kw,
            t,
            n_ctx: 0,
            kernels: vec![(
                if t == 1 { "conv_step" } else { "conv_seq" },
                ((cd * kw + 2 * cd * (kw - 1) + 2 * t * cd) * 4) as u64,
                times.into_iter().next().unwrap(),
            )],
            bits,
        });

        let o = g.zeros(t * nv * dv);
        let (times, bits) = bench.run(
            &format!("gdn t={t}"),
            &[(gdn_state0, gdn_state)],
            &Outs {
                f32s: vec![o],
                raw: vec![],
            },
            1,
            &|k, r| {
                let q = gdn_in.addr;
                let (kk, v) = (q + 4 * kd as u64, q + 8 * kd as u64);
                if t == 1 {
                    k.gdn_step(
                        r,
                        q,
                        kk,
                        v,
                        gdn_g.addr,
                        gdn_beta.addr,
                        gdn_state.addr,
                        o.addr,
                        nv,
                        dk,
                        dv,
                        ng,
                    );
                } else {
                    k.gdn_seq(
                        r,
                        q,
                        kk,
                        v,
                        gdn_g.addr,
                        gdn_beta.addr,
                        gdn_state.addr,
                        o.addr,
                        nv,
                        dk,
                        dv,
                        ng,
                        t,
                        cd,
                        true,
                    );
                }
            },
        );
        report(Case {
            group: "gdn",
            name: "gdn".into(),
            types: "f32".into(),
            layers: vec![],
            rows: nv,
            cols: dk,
            t,
            n_ctx: 0,
            kernels: vec![(
                if t == 1 { "gdn_step" } else { "gdn_seq" },
                (2 * nv * dk * dv * 4 + t * (cd + 2 * nv + nv * dv) * 4) as u64,
                times.into_iter().next().unwrap(),
            )],
            bits,
        });
    }

    // Short-context attention: the partials kernel, a barrier, the combine.
    let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
    for n_ctx in SHORT_CTX {
        let kc = g.bytes(&random_cache(&mut rng, n_ctx, nkv * hd));
        let vc = g.bytes(&random_cache(&mut rng, n_ctx, nkv * hd));
        let n_split = attn_splits(n_ctx);
        let q = g.f32s(&rand_f32(&mut rng, max_t * nh * hd, -12.0, 12.0));
        for &t in &tokens {
            let n_ctx0 = n_ctx + 1 - t;
            let part = g.zeros(t * nh * n_split * (hd + 2));
            let y = g.zeros(t * nh * hd);
            let (times, bits) = bench.run(
                &format!("attention n_ctx={n_ctx} t={t}"),
                &[],
                &Outs {
                    f32s: vec![part, y],
                    raw: vec![],
                },
                2,
                &|k, r| {
                    if t == 1 {
                        k.attn_decode(r, q.addr, kc.addr, vc.addr, part.addr, n_ctx, nh, nkv, hd);
                        r.barrier();
                        k.attn_combine(r, part.addr, y.addr, n_ctx, nh, hd);
                    } else {
                        k.attn_decode_tokens(
                            r, q.addr, kc.addr, vc.addr, part.addr, n_ctx0, t, nh, nkv, hd, n_split,
                        );
                        r.barrier();
                        k.attn_combine_tokens(r, part.addr, y.addr, n_ctx0, t, nh, hd, n_split);
                    }
                },
            );
            let kv_bytes = (2 * n_ctx * nkv * hd * 2) as u64;
            let mut it = times.into_iter();
            let (a, b) = (it.next().unwrap(), it.next().unwrap());
            let names = if t == 1 {
                ("attn_decode", "attn_combine")
            } else {
                (
                    attn_tokens_kernel(nh, nkv, hd).name(),
                    "attn_combine_tokens",
                )
            };
            report(Case {
                group: "attention",
                name: "attention".into(),
                types: "f16".into(),
                layers: vec![],
                rows: nh,
                cols: hd,
                t,
                n_ctx,
                kernels: vec![
                    (names.0, kv_bytes, a),
                    (names.1, (t * nh * n_split * (hd + 2) * 4) as u64, b),
                ],
                bits,
            });
        }
    }

    let mut out = format!(
        "{{\n  \"driver\": \"{}\",\n  \"reps\": {reps},\n  \"tokens\": {tokens:?},\n  \"cases\": [\n",
        json_escaped(&g.dev.driver)
    );
    for (i, c) in cases.iter().enumerate() {
        let kernels: Vec<String> = c
            .kernels
            .iter()
            .map(|(n, b, ts)| format!("{{\"kernel\": \"{n}\", {}}}", stats_json(ts, *b)))
            .collect();
        writeln!(
            out,
            "    {{\"group\": \"{}\", \"name\": \"{}\", \"types\": \"{}\", \"layers\": {:?}, \"rows\": {}, \
             \"cols\": {}, \"t\": {}, \"n_ctx\": {}, \"kernels\": [{}], \"total\": {{{}}}, \"bits\": \"{:016x}\"}}{}",
            c.group,
            c.name,
            c.types,
            c.layers,
            c.rows,
            c.cols,
            c.t,
            c.n_ctx,
            kernels.join(", "),
            stats_json(&c.totals(), c.bytes()),
            c.bits,
            if i + 1 < cases.len() { "," } else { "" }
        )
        .unwrap();
    }
    out.push_str("  ]\n}\n");
    match json {
        Some(path) => std::fs::write(&path, out).unwrap_or_else(|e| panic!("writing {path}: {e}")),
        None => print!("{out}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_example_times_every_dense_tensor_of_a_layer_once() {
        let table = dense_table(&Dims::QWEN36);
        let names: Vec<_> = table.iter().map(|t| t.name).collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "a tensor twice: {names:?}");
        let of = |h| table.iter().filter(|t| t.holder == h).count();
        // A DeltaNet layer: qkv, gate, beta, alpha, out; an attention layer: q, k, v, out; every layer: the router and
        // the shared expert's four; the output head once.
        assert_eq!(of(Holder::Delta) + of(Holder::Every), 10);
        assert_eq!(of(Holder::Attn) + of(Holder::Every), 9);
        assert_eq!(of(Holder::Head), 1);
        let shape = |n: &str| {
            let t = table.iter().find(|t| t.name == n).unwrap();
            (t.rows, t.cols)
        };
        assert_eq!(shape("delta.qkv"), (8192, 2048));
        assert_eq!(shape("delta.gate"), (4096, 2048));
        assert_eq!(shape("delta.beta"), (32, 2048));
        assert_eq!(shape("delta.alpha"), (32, 2048));
        assert_eq!(shape("delta.out"), (2048, 4096));
        assert_eq!(shape("attn.q"), (8192, 2048));
        assert_eq!(shape("attn.k"), (512, 2048));
        assert_eq!(shape("attn.v"), (512, 2048));
        assert_eq!(shape("attn.out"), (2048, 4096));
        assert_eq!(shape("router"), (256, 2048));
        assert_eq!(shape("shared.gate_inp"), (1, 2048));
        assert_eq!(shape("shared.gate"), (512, 2048));
        assert_eq!(shape("shared.up"), (512, 2048));
        assert_eq!(shape("shared.down"), (2048, 512));
        assert_eq!(shape("output"), (248_320, 2048));
    }
}
