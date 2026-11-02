//! The route simulator: routing traces recorded once (routing doesn't depend on the cache: test 5) replayed through the
//! real `SwapPlanner` and `CachePolicy`, stepping as `prepare_step` does (the lent rows taken before a staged chunk's
//! poll and returned after any other step's; `allowance` for the spend; every copy landed by the next step, as the
//! engine waits for each step), at given own and lent rows. Prints each trace's run and decode hit rates and each
//! group's average. A measurement, not a test:
//! `cargo run --release -p engine --example route_sim -- DIR OWN,LENT [OWN,LENT ...]`.
use engine::{
    allowance, CachePolicy, Entry, PolicyParams, PolicyStep, Profile, RowState, SwapPlanner,
    STAGE_FROM,
};
use qwen36::dims::Dims;
use std::collections::BTreeMap;

struct Step {
    prefill: bool,
    tokens: usize,
    ids: Vec<f32>,
}

fn read_trace(path: &std::path::Path) -> Vec<Step> {
    let b = std::fs::read(path).unwrap();
    let per = 40 * 8;
    let mut out = Vec::new();
    let mut at = 0;
    while at < b.len() {
        let kind = b[at];
        let t = u32::from_le_bytes(b[at + 1..at + 5].try_into().unwrap()) as usize;
        at += 5;
        let ids = b[at..at + t * per].iter().map(|&x| x as f32).collect();
        at += t * per;
        out.push(Step {
            prefill: kind == 1,
            tokens: t,
            ids,
        });
    }
    out
}

/// The real model's slot sizes: Q6_K down in layers 34, 38 and 39.
fn sizes() -> Vec<u64> {
    (0..40)
        .map(|l| {
            if [34, 38, 39].contains(&l) {
                2_039_808
            } else {
                1_900_544
            }
        })
        .collect()
}

#[derive(Default, Clone, Copy)]
struct Out {
    hits: u64,
    reads: u64,
    dhits: u64,
    dreads: u64,
}

fn run(trace: &[Step], own: usize, lent_n: usize) -> Out {
    let d = Dims::QWEN36;
    let prof = Profile::for_dims(&d);
    let sz = sizes();
    let live = own + lent_n;
    let resident: Vec<Vec<usize>> = (0..40).map(|l| prof.ranked(l)[..live].to_vec()).collect();
    let mut p = SwapPlanner::new(256, resident);
    let mut pol = CachePolicy::new(&prof, &d, true, PolicyParams::default());
    let (mut c_sub, mut t_sub) = (0u64, 0u64);
    let mut bank = 0u64;
    let mut o = Out::default();
    let lent = own..own + lent_n;
    for st in trace {
        let kind = if !st.prefill {
            PolicyStep::Decode
        } else if st.tokens >= STAGE_FROM {
            PolicyStep::Staged
        } else {
            PolicyStep::Prefill
        };
        let staged = kind == PolicyStep::Staged;
        if lent_n > 0 && staged && lent.clone().all(|r| p.row_state(r) == RowState::Live) {
            p.retire(lent.clone()).unwrap();
        }
        p.poll(c_sub, t_sub, |_| -> Result<u64, ()> {
            t_sub += 1;
            Ok(t_sub)
        })
        .unwrap();
        if lent_n > 0 && !staged && lent.clone().all(|r| p.row_state(r) == RowState::Dead) {
            p.add(lent.clone(), c_sub).unwrap();
        }
        let (spend, b) = allowance(kind, st.tokens, bank, pol.params().cap);
        bank = b;
        if spend > 0 {
            for pk in pol.choose(&p, spend, &sz) {
                match pk.out {
                    Some(out) => p.request(pk.layer, out, pk.inn).unwrap(),
                    None => {
                        p.fill(pk.layer, pk.inn).unwrap();
                    }
                }
            }
        }
        p.next_step(c_sub);
        let per = st.tokens * 8;
        for (j, &id) in st.ids.iter().enumerate() {
            let (l, e) = (j / per, id as usize);
            let hit = matches!(p.entry(l, e), Entry::Vram(_));
            o.reads += 1;
            o.hits += hit as u64;
            if !st.prefill {
                o.dreads += 1;
                o.dhits += hit as u64;
            }
        }
        pol.observe(&st.ids, st.tokens);
        c_sub += 1;
    }
    o
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&args[1]);
    let cfgs: Vec<(usize, usize)> = args[2..]
        .iter()
        .map(|s| {
            let (a, b) = s.split_once(',').unwrap();
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .collect();
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension()? == "routes")
                .then(|| p.file_stem().unwrap().to_string_lossy().into_owned())
        })
        .collect();
    names.sort();
    let traces: BTreeMap<String, Vec<Step>> = names
        .iter()
        .map(|n| (n.clone(), read_trace(&dir.join(format!("{n}.routes")))))
        .collect();
    let group = |n: &str| -> &'static str {
        let id: u32 = n.get(..2).and_then(|x| x.parse().ok()).unwrap_or(0);
        match id {
            1 | 5..=11 => "code",
            12..=15 => "prose",
            16..=18 => "tool",
            20 => "long",
            0 => "speed",
            _ => "other",
        }
    };
    let results: Vec<BTreeMap<String, Out>> = cfgs
        .iter()
        .map(|&(own, lent)| {
            traces
                .iter()
                .map(|(n, t)| (n.clone(), run(t, own, lent)))
                .collect()
        })
        .collect();
    print!("{:<24}", "prompt (run% decode%)");
    for c in &cfgs {
        print!(" | {:>14}", format!("{}+{}", c.0, c.1));
    }
    println!();
    for n in &names {
        print!("{:<24}", n);
        for r in &results {
            let o = r[n];
            print!(
                " | {:>6.2} {:>6.2} ",
                100.0 * o.hits as f64 / o.reads as f64,
                100.0 * o.dhits as f64 / o.dreads.max(1) as f64
            );
        }
        println!();
    }
    for g in ["code", "prose", "tool", "long", "speed"] {
        print!("{:<24}", format!("avg {g}"));
        for r in &results {
            let v: Vec<&Out> = names
                .iter()
                .filter(|n| group(n) == g)
                .map(|n| &r[n])
                .collect();
            let run: f64 = v
                .iter()
                .map(|o| o.hits as f64 / o.reads as f64)
                .sum::<f64>()
                / v.len() as f64;
            let dec: f64 = v
                .iter()
                .map(|o| o.dhits as f64 / o.dreads.max(1) as f64)
                .sum::<f64>()
                / v.len() as f64;
            print!(" | {:>6.2} {:>6.2} ", 100.0 * run, 100.0 * dec);
        }
        println!();
    }
}
