//! The routed-expert kernels on the GPU clock, on the real model's slots: decode's `experts_gate_up` and
//! `experts_down` (one token, 8 experts) and the verify step's `moe_unique`, `experts_gate_up_tokens` and
//! `experts_down_tokens` at t = 2 to 5 tokens. A measurement, not a test:
//! `cargo run --release -p kernels --example expert_speed -- --model GGUF --json FILE [--reps 1000]`
//! (`--model` defaults to `ARTEMIS_MODEL`).
//!
//! Method for a figure worth quoting (`attn_decode_speed`'s): run it twice (the default is 1,000 repetitions) and
//! compare the two runs' best totals. If every one agrees within 1%, keep the second run. If not, run again, up to four
//! runs in all, and keep the second run of the closest pair.
//!
//! The weights are the GGUF's own: the example reads each main layer's routed-expert types (gate, up, down), prints how
//! many layers have each mix, and times the first layer of each mix. Five tokens' routes are drawn once per mix
//! (`Rng::new(71)`, each token 8 distinct experts of 256, independent of the others); decode takes the first token's
//! 8 and a t-token verify the first t tokens'. Only the routed experts' slots are uploaded, each in its own VRAM
//! allocation, and a one-layer table of 256 entries points at them; decode also runs once with the same 8 slots in
//! pinned host memory (`loc` "host"), what a miss costs. Inputs are seeded, x in [-1, 1].
//!
//! Each dispatch is timed by the device's timestamps (`Gpu::run_timed`). Each repetition's command buffer starts with a
//! 256 MiB buffer copy and a barrier, which evict the GPU caches (the 64 MiB Infinity Cache included) as a decode
//! step's other weight reads do; the copy's own time is left out. A barrier separates the kernels, as in a step. One
//! untimed run comes first.
//!
//! The JSON holds `driver`, `reps`, the GGUF's expert `types` (`{gate, up, down, layers}` per mix), then a `cases`
//! array of `{form, loc, layer, types, t, unique_experts, kernels, total, bits}`. `kernels` holds per dispatch
//! `{kernel, us: {best, median}, bytes, gbps: {best, median}, pct_of_640: {best, median}, floor_us}`; `total` the
//! same over each repetition's sum of the dispatches (not the sum of the bests). `bytes` counts the expert weights a
//! dispatch reads (each distinct expert once), not its inputs or outputs (under 1% of them); GB/s is bytes over time,
//! and `floor_us` the bytes at 640 GB/s. `bits` is the FNV-1a 64 of the outputs' f32 bits (activations, then expert
//! outputs). Every output is asserted finite, nothing refused, and the first and last repetitions' outputs equal bit
//! for bit.
use gguf::Gguf;
use kernels::{
    dispatch::{
        read_status, slot_entry, ExpertStatus, SlotLayout, STATUS_CLEAR, WHERE_HOST, WHERE_VRAM,
    },
    testing::{f32_bytes, json_escaped, rand_f32, Gpu},
    tokens::moe_unique_bytes,
};
use qwen36::{dims::Dims, model::Model, quant::WType, testing::fnv1a64, testing::Rng};
use std::fmt::Write as _;

/// The memory bandwidth the floor is taken at, in bytes per second.
const FLOOR_BYTES_PER_S: f64 = 640e9;
/// The most tokens a verify case takes.
const MAX_T: usize = 5;

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

/// `[tokens][k]` routed ids: each token's `k` distinct experts of `n_expert`, drawn independently of the others.
fn routes(rng: &mut Rng, tokens: usize, k: usize, n_expert: usize) -> Vec<usize> {
    let mut ids = Vec::new();
    for _ in 0..tokens {
        let mut pool: Vec<usize> = (0..n_expert).collect();
        for _ in 0..k {
            let at = (rng.next_u64() % pool.len() as u64) as usize;
            ids.push(pool.swap_remove(at));
        }
    }
    ids
}

/// The distinct experts of `ids`, in order of first appearance.
fn unique(ids: &[usize]) -> Vec<usize> {
    let mut u = Vec::new();
    for &e in ids {
        if !u.contains(&e) {
            u.push(e);
        }
    }
    u
}

struct Case {
    form: &'static str,
    loc: &'static str,
    layer: usize,
    types: String,
    t: usize,
    unique: usize,
    names: Vec<&'static str>,
    bytes: Vec<u64>,
    /// Per dispatch, each repetition's µs.
    times: Vec<Vec<f64>>,
    bits: u64,
}

impl Case {
    fn totals(&self) -> Vec<f64> {
        (0..self.times[0].len())
            .map(|i| self.times.iter().map(|t| t[i]).sum())
            .collect()
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

fn main() {
    let (mut json, mut reps, mut model) = (None, 1000usize, std::env::var("ARTEMIS_MODEL").ok());
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
            other => panic!("unknown argument {other}; usage: --model GGUF --json FILE [--reps N]"),
        }
    }
    let model = model.expect("name the model's GGUF with --model or ARTEMIS_MODEL");
    let gg = Gguf::open(std::path::Path::new(&model))
        .unwrap_or_else(|e| panic!("opening {model}: {e:?}"));
    let d = Dims::QWEN36;
    let m = Model::open(&gg, &d).unwrap_or_else(|e| panic!("{model} is not the model: {e:?}"));
    let k = d.n_expert_used;

    // The routed experts' types per main layer, grouped by mix.
    let mut mixes: Vec<((WType, WType, WType), Vec<usize>)> = Vec::new();
    for (l, layer) in m.layers.iter().enumerate() {
        let key = (
            layer.moe.gate_exps.ty,
            layer.moe.up_exps.ty,
            layer.moe.down_exps.ty,
        );
        match mixes.iter_mut().find(|(t, _)| *t == key) {
            Some((_, ls)) => ls.push(l),
            None => mixes.push((key, vec![l])),
        }
    }
    let mtp = &m.mtp.layer.moe;
    println!(
        "routed experts: {} (blk.{} MTP: {}/{}/{}, not timed: always HOST)",
        mixes
            .iter()
            .map(|((g, u, dn), ls)| format!(
                "{}/{}/{} in {} layers {ls:?}",
                g.name(),
                u.name(),
                dn.name(),
                ls.len()
            ))
            .collect::<Vec<_>>()
            .join("; "),
        d.n_layer,
        mtp.gate_exps.ty.name(),
        mtp.up_exps.ty.name(),
        mtp.down_exps.ty.name()
    );

    let g = Gpu::new();
    assert!(g.dev.can_profile(), "this device has no timestamps");
    let evict_src = g.bytes(&vec![0x5a; 256 << 20]);
    let evict_dst = g.bytes(&vec![0; 256 << 20]);
    // XS_CLEAN=1: after each eviction copy, a read-only matvec over 256 MiB leaves the caches clean, closer to what a
    // decode step leaves, so the timed kernels do not pay to write the copy's dirty lines back. The matvec gets the
    // first timestamp, which is dropped (`ts[1..]` below).
    let clean = std::env::var("XS_CLEAN").is_ok_and(|v| v == "1");
    let clean_w = g.bytes(&vec![0u8; if clean { 256 << 20 } else { 16 }]);
    let clean_x = g.zeros(2048);
    let clean_y = g.zeros(32768);
    let mut cases = Vec::new();
    for ((gt, ut, dt), layers) in &mixes {
        let l = layers[0];
        let moe = &m.layers[l].moe;
        let lay = SlotLayout {
            gate: *gt,
            up: *ut,
            down: *dt,
            n_embd: d.n_embd,
            n_ff: d.n_ff_exp,
        };
        let types = format!("{}/{}/{}", gt.name(), ut.name(), dt.name());
        let slot = |e: usize| -> Vec<u8> {
            let mut s = moe.gate_exps.expert(e, d.n_ff_exp).bytes.to_vec();
            s.extend_from_slice(moe.up_exps.expert(e, d.n_ff_exp).bytes);
            s.extend_from_slice(moe.down_exps.expert(e, d.n_embd).bytes);
            assert_eq!(s.len(), lay.bytes());
            s
        };
        let mut rng = Rng::new(71);
        let ids = routes(&mut rng, MAX_T, k, d.n_expert);
        let x = rand_f32(&mut rng, MAX_T * d.n_embd, -1.0, 1.0);
        let gx = g.f32s(&x);
        let gids = g.f32s(&ids.iter().map(|&e| e as f32).collect::<Vec<_>>());
        // A table whose routed entries point at `loc` copies of their slots; every other entry is an unusable slot.
        let table_for = |loc: u32, experts: &[usize]| -> vk::Slice {
            let mut t = vec![0u8; d.n_expert * 16];
            for e in 0..d.n_expert {
                t[e * 16..(e + 1) * 16].copy_from_slice(&slot_entry(0, 7));
            }
            for &e in experts {
                let s = slot(e);
                let at = if loc == WHERE_VRAM {
                    g.bytes(&s)
                } else {
                    g.host_bytes(&s)
                };
                t[e * 16..(e + 1) * 16].copy_from_slice(&slot_entry(at.addr, loc));
            }
            g.bytes(&t)
        };
        let vram_table = table_for(WHERE_VRAM, &unique(&ids));
        let host_table = table_for(WHERE_HOST, &unique(&ids[..k]));
        let per_expert_gu = (lay.gate_bytes() + lay.up_bytes()) as u64;
        let per_expert_down = lay.down_bytes() as u64;

        let mut plan: Vec<(&'static str, &'static str, vk::Slice, usize)> = vec![
            ("decode", "vram", vram_table, 1),
            ("decode", "host", host_table, 1),
        ];
        plan.extend((2..=MAX_T).map(|t| ("verify", "vram", vram_table, t)));
        for (form, loc, table, t) in plan {
            let u = unique(&ids[..t * k]).len();
            let status = g.bytes(&STATUS_CLEAR);
            let list = g.bytes(&vec![0u8; moe_unique_bytes(t, k) as usize]);
            let act = g.zeros(t * k * d.n_ff_exp);
            let ye = g.zeros(t * k * d.n_embd);
            let run = |evict: bool| {
                g.run_timed(|kk, r| {
                    if evict {
                        r.copy(evict_src, evict_dst);
                        r.barrier();
                        if clean {
                            kk.matvec(
                                r,
                                kernels::dispatch::MatW {
                                    addr: clean_w.addr,
                                    ty: qwen36::quant::WType::F32,
                                    rows: 32768,
                                    cols: 2048,
                                },
                                clean_x.addr,
                                clean_y.addr,
                            );
                            r.barrier();
                        }
                    }
                    let (n, tab, st) = (d.n_expert, table.addr, status.addr);
                    if form == "decode" {
                        kk.experts_gate_up(
                            r, tab, 0, n, gids.addr, gx.addr, act.addr, st, &lay, k, 0, 0, 0,
                        );
                        r.barrier();
                        kk.experts_down(
                            r, tab, 0, n, gids.addr, act.addr, ye.addr, st, &lay, k, 0, 0, 0,
                        );
                    } else {
                        kk.moe_unique(r, gids.addr, list.addr, t, k);
                        r.barrier();
                        kk.experts_gate_up_tokens(
                            r, tab, 0, n, gids.addr, list.addr, gx.addr, act.addr, st, &lay, k, t,
                        );
                        r.barrier();
                        kk.experts_down_tokens(
                            r, tab, 0, n, gids.addr, list.addr, act.addr, ye.addr, st, &lay, k, t,
                        );
                    }
                })
            };
            let case_name = format!("{form} {loc} blk.{l} {types} t={t}");
            let outputs = || {
                let mut o = g.read(act);
                o.extend(g.read(ye));
                assert!(
                    o.iter().all(|v| v.is_finite()),
                    "{case_name}: a non-finite output"
                );
                o
            };
            run(false);
            let first = outputs();
            let n_disp = if form == "decode" { 2 } else { 3 };
            let mut times = vec![Vec::with_capacity(reps); n_disp];
            for _ in 0..reps {
                let ts = run(true);
                let ts = if clean { ts[1..].to_vec() } else { ts };
                assert_eq!(
                    ts.len(),
                    n_disp,
                    "{case_name}: {n_disp} dispatches, not {ts:?}"
                );
                for (v, t) in times.iter_mut().zip(ts) {
                    v.push(t);
                }
            }
            let last = outputs();
            assert!(
                f32_bytes(&first) == f32_bytes(&last),
                "{case_name}: the outputs changed between repetitions"
            );
            let st: [u8; kernels::dispatch::STATUS_BYTES] =
                g.dev.download(status).unwrap().try_into().unwrap();
            assert_eq!(
                read_status(&st),
                ExpertStatus::default(),
                "{case_name}: nothing refused"
            );
            let (names, bytes) = if form == "decode" {
                (
                    vec!["experts_gate_up", "experts_down"],
                    vec![u as u64 * per_expert_gu, u as u64 * per_expert_down],
                )
            } else {
                (
                    vec![
                        "moe_unique",
                        "experts_gate_up_tokens",
                        "experts_down_tokens",
                    ],
                    vec![0, u as u64 * per_expert_gu, u as u64 * per_expert_down],
                )
            };
            let c = Case {
                form,
                loc,
                layer: l,
                types: types.clone(),
                t,
                unique: u,
                names,
                bytes,
                times,
                bits: fnv1a64(&f32_bytes(&last)),
            };
            let mut line = format!("{case_name} ({u} experts):");
            for ((n, ts), b) in c.names.iter().zip(&c.times).zip(&c.bytes) {
                let (best, med) = best_and_median(ts);
                write!(line, " {n} best {best:.1} median {med:.1} us").unwrap();
                if *b > 0 {
                    write!(line, " ({:.0} GB/s best)", *b as f64 / best / 1e3).unwrap();
                }
                line.push(',');
            }
            let tot_b: u64 = c.bytes.iter().sum();
            let (tb, tm) = best_and_median(&c.totals());
            println!(
                "{line} total best {tb:.1} median {tm:.1} us, {:.0} GB/s best, floor {:.1} us",
                tot_b as f64 / tb / 1e3,
                tot_b as f64 / FLOOR_BYTES_PER_S * 1e6
            );
            cases.push(c);
        }
    }

    let mut out = format!(
        "{{\n  \"driver\": \"{}\",\n  \"reps\": {reps},\n  \"types\": [\n",
        json_escaped(&g.dev.driver)
    );
    for (i, ((gt, ut, dt), ls)) in mixes.iter().enumerate() {
        writeln!(
            out,
            "    {{\"gate\": \"{}\", \"up\": \"{}\", \"down\": \"{}\", \"layers\": {ls:?}}}{}",
            gt.name(),
            ut.name(),
            dt.name(),
            if i + 1 < mixes.len() { "," } else { "" }
        )
        .unwrap();
    }
    out.push_str("  ],\n  \"cases\": [\n");
    for (i, c) in cases.iter().enumerate() {
        let kernels: Vec<String> = c
            .names
            .iter()
            .zip(&c.times)
            .zip(&c.bytes)
            .map(|((n, ts), b)| format!("{{\"kernel\": \"{n}\", {}}}", stats_json(ts, *b)))
            .collect();
        writeln!(
            out,
            "    {{\"form\": \"{}\", \"loc\": \"{}\", \"layer\": {}, \"types\": \"{}\", \"t\": {}, \
             \"unique_experts\": {}, \"kernels\": [{}], \"total\": {{{}}}, \"bits\": \"{:016x}\"}}{}",
            c.form,
            c.loc,
            c.layer,
            c.types,
            c.t,
            c.unique,
            kernels.join(", "),
            stats_json(&c.totals(), c.bytes.iter().sum()),
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
