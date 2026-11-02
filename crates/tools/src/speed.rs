//! The speed gate. Over two runs each of the write, edit and explain prompts, the
//! median decode speed must reach 46.5 tok/s. The slower of two cold prefills of the 27,416-token golden prompt must
//! reach 1,000 tok/s. `artemis speed` reads `generate --json` records and writes this verdict. A record made any other
//! way than the gate's runs is refused by name, so a stray flag can't pass, and a budget that moved mid-gate is a
//! warning (the runner's desktop GPU samples catch a desktop change; see `HELD_BACK_SPREAD_MIB`).
use crate::{
    generate::{Record, Settings},
    test3::Status,
};
use qwen36::testing::fnv1a64;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const DECODE_BOUND: f64 = 46.5;
pub const PREFILL_BOUND: f64 = 1_000.0;
pub const PROMPTS: [&str; 3] = ["write", "edit", "explain"];
pub const SEEDS: [u64; 2] = [1234, 1235];
pub const MAX_TOKENS: usize = 700;
pub const CTX: usize = 65_536;
/// A larger change across the runs in what the driver holds back from the budget at load (heap minus budget) is a
/// warning. On this driver that figure is a fixed reserve (15,416.5 MiB of budget whatever the desktop holds), so the
/// warning only catches a budget the OS lowers; the runner's per-run samples of the desktop's own GPU memory
/// (`tools/run-gpu-tests.ps1`, speed block) are what catch a desktop change.
pub const HELD_BACK_SPREAD_MIB: f64 = 500.0;

/// A prompt's ids as `testdata/speed/made-from.json` records them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MadeFrom {
    pub tokens: usize,
    pub fnv1a64: String,
}

impl MadeFrom {
    /// The count and fingerprint of `ids`, as `tools/py/build_speed_prompts.py` records them.
    pub fn of(ids: &[u32]) -> MadeFrom {
        let bytes: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        MadeFrom {
            tokens: ids.len(),
            fnv1a64: format!("{:016x}", fnv1a64(&bytes)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecodeRow {
    pub prompt: String,
    pub seed: u64,
    pub decoded: usize,
    pub seconds: f64,
    pub tok_s: f64,
    /// The share of the run's routed experts read from VRAM, as a percentage: the cache's and the staging
    /// buffers' reads, prefill's routes included.
    pub from_vram_pct: f64,
    pub longest_submit_ms: f64,
    pub slots: usize,
    /// The heap's size minus the budget at load: what the driver held back from this
    /// process's budget (on this driver a fixed reserve, not the desktop's use, which the runner samples).
    /// None for a record without VRAM figures.
    pub held_back_mib: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PrefillRow {
    pub tokens: usize,
    pub seconds: f64,
    pub tok_s: f64,
    /// The share of the run's routed experts read from VRAM, as a percentage: the cache's and the staging
    /// buffers' reads, as in `DecodeRow`.
    pub from_vram_pct: f64,
    pub longest_submit_ms: f64,
    pub slots: usize,
    /// As in `DecodeRow`.
    pub held_back_mib: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Verdict {
    /// Pass when the records are the gate's runs and both speeds reach their bounds.
    pub status: Status,
    /// Whether a speed below a bound fails the run (`--gate`), or is only reported.
    pub gated: bool,
    pub decode: Vec<DecodeRow>,
    pub decode_median: f64,
    pub decode_bound: f64,
    pub prefill: Vec<PrefillRow>,
    /// The slower prefill's speed; 0 without one.
    pub prefill_slowest: f64,
    pub prefill_bound: f64,
    /// Records that aren't the gate's runs. Any problem fails the run, gated or not.
    pub problems: Vec<String>,
    pub warnings: Vec<String>,
    /// Decode's base time, from `artemis speed --base`: reported, never gated. None without a
    /// base-time record, or with one that isn't a base-time run of the write prompt (a warning says why).
    pub base: Option<BaseRow>,
}

/// A base-time run (`generate --base-time`): decode with every routed expert read from VRAM.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BaseRow {
    pub decoded: usize,
    pub seconds: f64,
    pub tok_s: f64,
    pub ms_per_token: f64,
    pub slots: usize,
}

/// The base time from `r`, which must be a base-time run of the write prompt with the gate's sampling and settings;
/// otherwise every reason it isn't.
pub fn base_row(r: &Record, made: &BTreeMap<String, MadeFrom>) -> Result<BaseRow, Vec<String>> {
    let mut why = Vec::new();
    if !r.settings.base_time {
        why.push("not a --base-time run".to_string());
    }
    if made.get("write") != Some(&MadeFrom::of(&r.prompt_tokens)) {
        why.push("its prompt isn't the write prompt".into());
    }
    let mut s = r.settings.clone();
    s.base_time = false;
    why.extend(decode_settings(&s));
    if r.timings.decoded == 0 || r.timings.generate_s <= 0.0 {
        why.push("no decode to time".into());
    }
    if !why.is_empty() {
        return Err(why);
    }
    Ok(BaseRow {
        decoded: r.timings.decoded,
        seconds: r.timings.generate_s,
        tok_s: r.timings.decoded as f64 / r.timings.generate_s,
        ms_per_token: 1e3 * r.timings.generate_s / r.timings.decoded as f64,
        slots: r.cache.slots_per_layer,
    })
}

impl Verdict {
    /// Whether `artemis speed` exits 1: on any problem, and, with `--gate`, on a speed below its bound.
    pub fn exits_nonzero(&self) -> bool {
        !self.problems.is_empty() || (self.gated && self.status == Status::Fail)
    }
}

/// The median of `v`; 0 when empty.
pub fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    match s.len() {
        0 => 0.0,
        n if !n.is_multiple_of(2) => s[n / 2],
        n => (s[n / 2 - 1] + s[n / 2]) / 2.0,
    }
}

/// Settings every gate run shares: the default context, the default cache, no forced swaps, no `--all-logits`.
fn shared(s: &Settings, bad: &mut Vec<String>) {
    if s.ctx != CTX {
        bad.push(format!("--ctx {}, not {CTX}", s.ctx));
    }
    if let Some(n) = s.cache_slots_asked {
        bad.push(format!(
            "--cache-slots {n}; the gate runs the default cache"
        ));
    }
    if s.force_swap_every_token {
        bad.push("--force-swap-every-token; the gate runs without it".into());
    }
    if s.all_logits {
        bad.push("--all-logits; the gate runs without it".into());
    }
    if s.profiled {
        bad.push("--profile; the gate's runs aren't profiled".into());
    }
    if s.fingerprints {
        bad.push("recorded with fingerprints; the gate's runs use --no-fingerprints".into());
    }
    if s.f32_prefill {
        bad.push("--f32-prefill; the gate runs the device's prefill kernels".into());
    }
    if s.no_staging {
        bad.push("--no-staging; the gate stages".into());
    }
    if let Some(n) = s.stage_from {
        bad.push(format!(
            "--stage-from {n}; the gate stages from the measured threshold"
        ));
    }
    if s.cache_policy != "adaptive" {
        bad.push(format!(
            "the {} cache policy; the gate runs the adaptive one",
            s.cache_policy
        ));
    }
    if let Some(n) = s.upload_cap_mib {
        bad.push(format!(
            "--upload-cap {n}; the gate runs the policy's own cap"
        ));
    }
    if s.no_lend {
        bad.push("--no-lend; the gate lends the staging area's VRAM".into());
    }
    if s.base_time {
        bad.push("--base-time; the gate's runs read the experts the model routes".into());
    }
    if s.cpu_share != 0.0 {
        bad.push(format!(
            "--cpu-share {}; the gate's runs read every miss over PCIe",
            s.cpu_share
        ));
    }
}

/// How a decode run's settings differ from the speed prompts' sampling.
fn decode_settings(s: &Settings) -> Vec<String> {
    let mut bad = Vec::new();
    if s.greedy {
        bad.push("greedy; the gate samples".into());
    }
    if s.temp != 0.6 {
        bad.push(format!("temperature {}, not 0.6", s.temp));
    }
    if s.top_p != 0.95 {
        bad.push(format!("top-p {}, not 0.95", s.top_p));
    }
    if s.top_k != 20 {
        bad.push(format!("top-k {}, not 20", s.top_k));
    }
    if s.min_p != 0.0 {
        bad.push(format!("min-p {}, not 0", s.min_p));
    }
    if s.presence_penalty != 0.0 {
        bad.push(format!("presence penalty {}, not 0", s.presence_penalty));
    }
    if s.max_tokens != MAX_TOKENS {
        bad.push(format!("--max-tokens {}, not {MAX_TOKENS}", s.max_tokens));
    }
    if !SEEDS.contains(&s.seed) {
        bad.push("a seed other than the gate's 1234 and 1235".into());
    }
    shared(s, &mut bad);
    bad
}

/// How a prefill run's settings differ from the gate's: greedy, one token, after the whole prompt.
fn prefill_settings(s: &Settings) -> Vec<String> {
    let mut bad = Vec::new();
    if !s.greedy {
        bad.push("not greedy; the gate's prefill runs are --greedy --max-tokens 1".into());
    }
    if s.max_tokens != 1 {
        bad.push(format!("--max-tokens {}, not 1", s.max_tokens));
    }
    shared(s, &mut bad);
    bad
}

/// The share of the run's routed-expert reads served from VRAM: the cache's and the staging buffers'.
fn from_vram_pct(r: &Record) -> f64 {
    let vram = r.cache.hits + r.cache.staged;
    100.0 * vram as f64 / (vram + r.cache.misses).max(1) as f64
}

/// What the driver held back from the run's budget at load, in MiB (heap minus budget), or None for a record without
/// VRAM figures. On this driver that is a fixed reserve, not the desktop's share: see `HELD_BACK_SPREAD_MIB`.
fn held_back(r: &Record) -> Option<f64> {
    (r.vram.heap_mib > 0.0).then_some(r.vram.heap_mib - r.vram.budget_at_open_mib)
}

/// Notes a run's held-back VRAM for the spread check and returns it; a record without VRAM figures is a warning.
fn note_vram(
    held: &mut Vec<(String, f64)>,
    warnings: &mut Vec<String>,
    name: &str,
    r: &Record,
) -> Option<f64> {
    let h = held_back(r);
    match h {
        Some(h) => held.push((name.to_string(), h)),
        None => warnings.push(format!(
            "{name}: no VRAM figures (a record from before plan 2a)"
        )),
    }
    h
}

/// The gate's verdict over `decode` (each labelled with its speed prompt) and `prefill` records. `made` is
/// `made-from.json`; `long` is the long golden prompt's ids; `gate` makes a speed below a bound fail the run.
///
/// A run with any problem is named in `problems` and left out of the rows, so out of the median and the slowest
/// prefill. Of repeated runs of one prompt and seed, the first is kept and the rest are left out.
pub fn judge(
    decode: &[(String, Record)],
    prefill: &[Record],
    made: &BTreeMap<String, MadeFrom>,
    long: &[u32],
    gate: bool,
) -> Verdict {
    let (mut problems, mut warnings) = (Vec::new(), Vec::new());
    let mut held: Vec<(String, f64)> = Vec::new();
    let mut seen: BTreeMap<(String, u64), usize> = BTreeMap::new();
    let mut rows = Vec::new();
    for (prompt, r) in decode {
        let name = format!("{prompt} seed {}", r.settings.seed);
        if !PROMPTS.contains(&prompt.as_str()) {
            problems.push(format!(
                "{name}: not a speed prompt; they are {}",
                PROMPTS.join(", ")
            ));
            continue;
        }
        let mut own = Vec::new();
        match made.get(prompt) {
            Some(want) => {
                let got = MadeFrom::of(&r.prompt_tokens);
                if &got != want {
                    own.push(format!(
                        "its prompt is {} tokens with fingerprint {}, not the speed prompt's {} tokens with {}",
                        got.tokens, got.fnv1a64, want.tokens, want.fnv1a64
                    ));
                }
            }
            None => own.push(format!("made-from.json has no entry for {prompt}")),
        }
        own.extend(decode_settings(&r.settings));
        if r.timings.decoded == 0 {
            own.push("decoded nothing, so it measures no speed".into());
        } else if r.timings.generate_s <= 0.0 {
            own.push("no decode time, so it measures no speed".into());
        }
        problems.extend(own.iter().map(|bad| format!("{name}: {bad}")));
        let count = seen.entry((prompt.clone(), r.settings.seed)).or_default();
        *count += 1;
        // A repeat is named in the summary below.
        if *count > 1 || !own.is_empty() {
            continue;
        }
        let held_back_mib = note_vram(&mut held, &mut warnings, &name, r);
        rows.push(DecodeRow {
            prompt: prompt.clone(),
            seed: r.settings.seed,
            decoded: r.timings.decoded,
            seconds: r.timings.generate_s,
            tok_s: r.timings.decoded as f64 / r.timings.generate_s,
            from_vram_pct: from_vram_pct(r),
            longest_submit_ms: r.cache.longest_submit_ms,
            slots: r.cache.slots_per_layer,
            held_back_mib,
        });
    }
    for p in PROMPTS {
        for seed in SEEDS {
            match seen.get(&(p.to_string(), seed)).copied().unwrap_or(0) {
                0 => problems.push(format!("{p} seed {seed}: no run")),
                1 => {}
                n => problems.push(format!("{p} seed {seed}: {n} runs; the gate takes one")),
            }
        }
    }
    if prefill.len() != 2 {
        problems.push(format!(
            "{} prefill runs; the gate takes two",
            prefill.len()
        ));
    }
    let mut prows = Vec::new();
    for (i, r) in prefill.iter().enumerate() {
        let name = format!("prefill run {}", i + 1);
        let mut own = Vec::new();
        if r.prompt_tokens != long {
            own.push(format!(
                "its prompt isn't the long golden prompt ({} tokens; it has {})",
                r.prompt_tokens.len(),
                long.len()
            ));
        }
        own.extend(prefill_settings(&r.settings));
        if r.timings.prompt_s <= 0.0 {
            own.push("no prefill time".into());
        }
        problems.extend(own.iter().map(|bad| format!("{name}: {bad}")));
        if !own.is_empty() {
            continue;
        }
        let held_back_mib = note_vram(&mut held, &mut warnings, &name, r);
        prows.push(PrefillRow {
            tokens: r.prompt_tokens.len(),
            seconds: r.timings.prompt_s,
            tok_s: r.prompt_tokens.len() as f64 / r.timings.prompt_s,
            from_vram_pct: from_vram_pct(r),
            longest_submit_ms: r.cache.longest_submit_ms,
            slots: r.cache.slots_per_layer,
            held_back_mib,
        });
    }
    let lo = held.iter().min_by(|a, b| a.1.total_cmp(&b.1));
    let hi = held.iter().max_by(|a, b| a.1.total_cmp(&b.1));
    if let (Some(lo), Some(hi)) = (lo, hi) {
        if hi.1 - lo.1 > HELD_BACK_SPREAD_MIB {
            warnings.push(format!(
                "the VRAM the driver held back at load ranged from {:.0} MiB ({}) to {:.0} MiB ({}): the driver's \
                 budget changed during the gate",
                lo.1, lo.0, hi.1, hi.0
            ));
        }
    }
    let decode_median = median(&rows.iter().map(|r| r.tok_s).collect::<Vec<_>>());
    let prefill_slowest = prows
        .iter()
        .map(|r| r.tok_s)
        .reduce(f64::min)
        .unwrap_or(0.0);
    let fast = decode_median >= DECODE_BOUND && prefill_slowest >= PREFILL_BOUND;
    Verdict {
        status: if problems.is_empty() && fast {
            Status::Pass
        } else {
            Status::Fail
        },
        gated: gate,
        decode: rows,
        decode_median,
        decode_bound: DECODE_BOUND,
        prefill: prows,
        prefill_slowest,
        prefill_bound: PREFILL_BOUND,
        problems,
        warnings,
        base: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::{CacheStats, Settings, Stop, Timings, Vram, FORMAT};

    fn prompt(i: usize) -> Vec<u32> {
        (0..10 + i as u32).collect()
    }

    fn long() -> Vec<u32> {
        (0..100).collect()
    }

    fn made() -> BTreeMap<String, MadeFrom> {
        PROMPTS
            .iter()
            .enumerate()
            .map(|(i, p)| (p.to_string(), MadeFrom::of(&prompt(i))))
            .collect()
    }

    /// A record as the runner's speed block makes them: sampled for a decode run, greedy for a prefill run.
    fn rec(
        tokens: Vec<u32>,
        greedy: bool,
        max_tokens: usize,
        seed: u64,
        generate_s: f64,
        prompt_s: f64,
    ) -> Record {
        let decoded = if greedy { 0 } else { 700 };
        Record {
            format: FORMAT.into(),
            prompt_tokens: tokens,
            settings: Settings {
                greedy,
                temp: if greedy { 0.0 } else { 0.6 },
                top_k: if greedy { 0 } else { 20 },
                top_p: if greedy { 1.0 } else { 0.95 },
                min_p: 0.0,
                presence_penalty: 0.0,
                seed,
                max_tokens,
                ctx: CTX,
                cache_slots: 130,
                force_swap_every_token: false,
                all_logits: false,
                cache_slots_asked: None,
                fingerprints: false,
                profiled: false,
                f32_prefill: false,
                no_staging: false,
                stage_from: None,
                cache_policy: "adaptive".into(),
                upload_cap_mib: None,
                no_lend: false,
                base_time: false,
                cpu_share: 0.0,
                speculation: None,
            },
            steps: Vec::new(),
            stop: Stop::MaxTokens,
            positions: Vec::new(),
            timings: Timings {
                load_s: 20.0,
                prompt_s,
                generate_s,
                decoded,
            },
            cache: CacheStats {
                slots_per_layer: 130,
                hits: 800,
                misses: 200,
                ..CacheStats::default()
            },
            memory: Default::default(),
            vram: Vram {
                heap_mib: 16_304.0,
                budget_at_open_mib: 15_416.0,
                usage_at_open_mib: 0.0,
                budget_mib: 14_339.0,
                usage_mib: 13_700.0,
            },
            prefill: Default::default(),
            warm: None,
            spec: None,
        }
    }

    fn decode_run(i: usize, seed: u64, tok_s: f64) -> (String, Record) {
        (
            PROMPTS[i].into(),
            rec(prompt(i), false, MAX_TOKENS, seed, 700.0 / tok_s, 0.5),
        )
    }

    fn prefill_run(tok_s: f64) -> Record {
        rec(long(), true, 1, 0, 0.0, 100.0 / tok_s)
    }

    /// The six decode runs, prompt by prompt, seed 1234 then 1235.
    fn six(speeds: [f64; 6]) -> Vec<(String, Record)> {
        (0..6)
            .map(|k| decode_run(k / 2, SEEDS[k % 2], speeds[k]))
            .collect()
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9 * b.abs().max(1.0)
    }

    #[test]
    fn the_median_of_six_is_the_mean_of_the_middle_two() {
        assert_eq!(median(&[50.0, 31.0, 47.0, 56.0, 33.0, 48.0]), 47.5);
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[]), 0.0);
    }

    #[test]
    fn fast_runs_pass_and_slow_ones_fail_and_exit_only_when_gated() {
        let fast = judge(
            &six([50.0, 52.0, 47.0, 48.0, 33.0, 31.0]),
            &[prefill_run(1_200.0), prefill_run(1_100.0)],
            &made(),
            &long(),
            true,
        );
        assert_eq!(fast.status, Status::Pass, "{fast:?}");
        assert!(near(fast.decode_median, 47.5), "{}", fast.decode_median);
        assert!(
            near(fast.prefill_slowest, 1_100.0),
            "{}",
            fast.prefill_slowest
        );
        assert!(
            fast.problems.is_empty() && fast.warnings.is_empty(),
            "{fast:?}"
        );
        assert!(near(fast.decode[0].from_vram_pct, 80.0));
        let slow = judge(
            &six([40.0; 6]),
            &[prefill_run(480.0), prefill_run(500.0)],
            &made(),
            &long(),
            false,
        );
        assert_eq!(slow.status, Status::Fail);
        assert!(
            slow.problems.is_empty(),
            "slow is not a problem: {:?}",
            slow.problems
        );
        assert!(!slow.gated);
        assert!(
            !slow.exits_nonzero(),
            "a slow run fails the run only when gated"
        );
        assert!(judge(
            &six([40.0; 6]),
            &[prefill_run(480.0), prefill_run(500.0)],
            &made(),
            &long(),
            true
        )
        .exits_nonzero());
    }

    #[test]
    fn staged_reads_count_as_from_vram() {
        // A staged prefill reads non-resident experts from a staging buffer in VRAM: the row's
        // share from VRAM counts them with the cache's hits.
        let mut r = prefill_run(1_200.0);
        r.cache.hits = 600;
        r.cache.staged = 300;
        r.cache.misses = 100;
        assert_eq!(from_vram_pct(&r), 90.0);
    }

    /// An older record has no cache policy: it reads as static, and the gate refuses it for that.
    #[test]
    fn a_record_from_before_plan_2c_is_refused_as_static() {
        let mut runs = six([50.0; 6]);
        let mut v = serde_json::to_value(&runs[3].1).unwrap();
        for k in ["cache_policy", "upload_cap_mib", "no_lend"] {
            assert!(
                v["settings"].as_object_mut().unwrap().remove(k).is_some(),
                "{k}"
            );
        }
        runs[3].1 = serde_json::from_value(v).unwrap();
        let v = judge(
            &runs,
            &[prefill_run(1_200.0), prefill_run(1_200.0)],
            &made(),
            &long(),
            false,
        );
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems
                .iter()
                .any(|p| p.starts_with("edit seed 1235") && p.contains("static cache policy")),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn the_base_time_is_read_from_a_base_time_run_of_the_write_prompt_only() {
        let mut r = decode_run(0, 1234, 100.0).1;
        r.settings.base_time = true;
        let row = base_row(&r, &made()).unwrap();
        assert_eq!((row.decoded, row.slots), (700, 130));
        assert!(
            near(row.ms_per_token, 10.0) && near(row.tok_s, 100.0),
            "{row:?}"
        );
        // Not a base-time run, another prompt, another sampling, no decode: each named, and no row.
        let mut plain = r.clone();
        plain.settings.base_time = false;
        let mut edit = decode_run(1, 1234, 100.0).1;
        edit.settings.base_time = true;
        let mut hot = r.clone();
        hot.settings.temp = 1.0;
        let mut none = r.clone();
        none.timings.decoded = 0;
        for (want, rec) in [
            ("not a --base-time run", plain),
            ("isn't the write prompt", edit),
            ("temperature", hot),
            ("no decode to time", none),
        ] {
            let why = base_row(&rec, &made()).unwrap_err();
            assert!(why.iter().any(|w| w.contains(want)), "{want}: {why:?}");
        }
        // The verdict carries no base row of its own: `artemis speed --base` adds it.
        let v = judge(
            &six([50.0; 6]),
            &[prefill_run(1_200.0), prefill_run(1_200.0)],
            &made(),
            &long(),
            true,
        );
        assert!(v.base.is_none());
    }

    /// An older record has no `cpu_share`: it reads as 0, and the gate does not refuse it for that.
    #[test]
    fn a_record_without_a_cpu_share_reads_as_zero_and_is_not_refused_for_it() {
        let mut runs = six([50.0; 6]);
        let mut v = serde_json::to_value(&runs[3].1).unwrap();
        v["settings"]["cpu_share"] = serde_json::json!(1.0);
        assert!(v["settings"]
            .as_object_mut()
            .unwrap()
            .remove("cpu_share")
            .is_some());
        runs[3].1 = serde_json::from_value(v).unwrap();
        assert_eq!(runs[3].1.settings.cpu_share, 0.0);
        let v = judge(
            &runs,
            &[prefill_run(1_200.0), prefill_run(1_200.0)],
            &made(),
            &long(),
            false,
        );
        assert!(
            !v.problems.iter().any(|p| p.contains("cpu-share")),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn records_that_arent_the_gates_runs_are_refused_by_name() {
        type Change = fn(&mut Record);
        let cases: [(&str, Change); 24] = [
            ("not the speed prompt", |r| r.prompt_tokens.push(7)),
            ("temperature", |r| r.settings.temp = 0.7),
            ("top-k", |r| r.settings.top_k = 40),
            ("top-p", |r| r.settings.top_p = 0.9),
            ("min-p", |r| r.settings.min_p = 0.05),
            ("presence penalty", |r| r.settings.presence_penalty = 1.0),
            ("--max-tokens", |r| r.settings.max_tokens = 256),
            ("--ctx", |r| r.settings.ctx = 4096),
            ("--cache-slots", |r| {
                r.settings.cache_slots_asked = Some(102)
            }),
            ("--force-swap-every-token", |r| {
                r.settings.force_swap_every_token = true
            }),
            ("--all-logits", |r| r.settings.all_logits = true),
            ("--profile", |r| r.settings.profiled = true),
            ("recorded with fingerprints", |r| {
                r.settings.fingerprints = true
            }),
            ("--f32-prefill", |r| r.settings.f32_prefill = true),
            ("--no-staging", |r| r.settings.no_staging = true),
            ("--stage-from", |r| r.settings.stage_from = Some(32)),
            ("cache policy", |r| {
                r.settings.cache_policy = "static".into()
            }),
            ("--upload-cap", |r| r.settings.upload_cap_mib = Some(16)),
            ("--no-lend", |r| r.settings.no_lend = true),
            ("--base-time", |r| r.settings.base_time = true),
            ("--cpu-share", |r| r.settings.cpu_share = 1.0),
            ("greedy", |r| r.settings.greedy = true),
            ("decoded nothing", |r| r.timings.decoded = 0),
            ("no decode time", |r| r.timings.generate_s = 0.0),
        ];
        for (want, change) in cases {
            let mut runs = six([50.0; 6]);
            // A wildly different speed, so a refused run that leaked into the median would show.
            runs[3].1.timings.generate_s = 700.0 / 5.0;
            change(&mut runs[3].1);
            let v = judge(
                &runs,
                &[prefill_run(1_200.0), prefill_run(1_200.0)],
                &made(),
                &long(),
                false,
            );
            assert_eq!(v.status, Status::Fail, "{want}");
            assert!(
                v.exits_nonzero(),
                "{want}: a problem fails the run, gated or not"
            );
            assert!(
                v.problems
                    .iter()
                    .any(|p| p.starts_with("edit seed 1235") && p.contains(want)),
                "{want}: {:?}",
                v.problems
            );
            assert_eq!(
                v.decode.len(),
                5,
                "{want}: the refused run is out of the rows"
            );
            assert!(near(v.decode_median, 50.0), "{want}: {}", v.decode_median);
        }
        // A seed other than the gate's is named by its own seed, and the seed that has no run is named too.
        let mut runs = six([50.0; 6]);
        runs[3].1.settings.seed = 9999;
        let v = judge(
            &runs,
            &[prefill_run(1_200.0), prefill_run(1_200.0)],
            &made(),
            &long(),
            false,
        );
        assert!(
            v.problems
                .iter()
                .any(|p| p.starts_with("edit seed 9999") && p.contains("a seed other")),
            "{:?}",
            v.problems
        );
        assert!(
            v.problems.contains(&"edit seed 1235: no run".to_string()),
            "{:?}",
            v.problems
        );
        assert_eq!(v.decode.len(), 5);
        // A prefill of another prompt, or with other settings, is refused and out of the slowest.
        let mut other = prefill_run(5.0);
        other.prompt_tokens.pop();
        let mut sampled = prefill_run(5.0);
        sampled.settings.greedy = false;
        let mut profiled = prefill_run(5.0);
        profiled.settings.profiled = true;
        let mut fingerprinted = prefill_run(5.0);
        fingerprinted.settings.fingerprints = true;
        let v = judge(
            &six([50.0; 6]),
            &[profiled, fingerprinted],
            &made(),
            &long(),
            false,
        );
        assert!(
            v.problems
                .iter()
                .any(|p| p.starts_with("prefill run 1") && p.contains("--profile")),
            "{:?}",
            v.problems
        );
        assert!(
            v.problems.iter().any(|p| p.starts_with("prefill run 2")
                && p.contains("recorded with fingerprints")),
            "{:?}",
            v.problems
        );
        assert!(v.prefill.is_empty(), "{:?}", v.prefill);
        let v = judge(
            &six([50.0; 6]),
            &[other.clone(), sampled],
            &made(),
            &long(),
            false,
        );
        assert!(
            v.problems
                .iter()
                .any(|p| p.starts_with("prefill run 1") && p.contains("long golden prompt")),
            "{:?}",
            v.problems
        );
        assert!(
            v.problems
                .iter()
                .any(|p| p.starts_with("prefill run 2") && p.contains("greedy")),
            "{:?}",
            v.problems
        );
        assert!(
            v.prefill.is_empty() && v.prefill_slowest == 0.0,
            "{:?}",
            v.prefill
        );
        let v = judge(
            &six([50.0; 6]),
            &[prefill_run(1_200.0), other],
            &made(),
            &long(),
            false,
        );
        assert_eq!(v.prefill.len(), 1);
        assert!(near(v.prefill_slowest, 1_200.0), "{}", v.prefill_slowest);
        assert!(near(v.prefill[0].from_vram_pct, 80.0));
        // A label that isn't a speed prompt.
        let mut runs = six([50.0; 6]);
        runs[0].0 = "prose".into();
        let v = judge(
            &runs,
            &[prefill_run(1_200.0), prefill_run(1_200.0)],
            &made(),
            &long(),
            false,
        );
        assert!(
            v.problems
                .iter()
                .any(|p| p.starts_with("prose") && p.contains("not a speed prompt")),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn missing_and_repeated_runs_are_problems() {
        let mut runs = six([50.0; 6]);
        runs.remove(1);
        runs.push(decode_run(2, 1234, 5.0));
        let v = judge(&runs, &[prefill_run(1_200.0)], &made(), &long(), false);
        assert!(
            v.problems.contains(&"write seed 1235: no run".to_string()),
            "{:?}",
            v.problems
        );
        assert!(
            v.problems
                .contains(&"explain seed 1234: 2 runs; the gate takes one".to_string()),
            "{:?}",
            v.problems
        );
        assert!(
            v.problems
                .contains(&"1 prefill runs; the gate takes two".to_string()),
            "{:?}",
            v.problems
        );
        // The first run of explain seed 1234 is kept; the repeat (5 tok/s) is out of the rows.
        assert_eq!(v.decode.len(), 5);
        assert!(v.decode.iter().all(|r| r.tok_s > 40.0), "{:?}", v.decode);
    }

    #[test]
    fn a_desktop_that_changes_mid_gate_is_a_warning() {
        let mut runs = six([50.0; 6]);
        // A browser opened before the fifth run: the driver held back 1.2 GiB more.
        runs[4].1.vram.budget_at_open_mib -= 1_200.0;
        let v = judge(
            &runs,
            &[prefill_run(1_200.0), prefill_run(1_200.0)],
            &made(),
            &long(),
            false,
        );
        assert!(v.problems.is_empty(), "{:?}", v.problems);
        assert!(near(v.decode[0].held_back_mib.unwrap(), 888.0));
        assert!(near(v.decode[4].held_back_mib.unwrap(), 2_088.0));
        assert!(
            v.warnings
                .iter()
                .any(|w| w.contains("888") && w.contains("2088")),
            "{:?}",
            v.warnings
        );
        // An older record has no VRAM figures: a warning, not a problem.
        let mut runs = six([50.0; 6]);
        runs[0].1.vram = Vram::default();
        let v = judge(
            &runs,
            &[prefill_run(1_200.0), prefill_run(1_200.0)],
            &made(),
            &long(),
            false,
        );
        assert!(v.problems.is_empty());
        assert!(
            v.warnings.iter().any(|w| w.contains("no VRAM figures")),
            "{:?}",
            v.warnings
        );
        assert_eq!(v.decode[0].held_back_mib, None);
    }
}
