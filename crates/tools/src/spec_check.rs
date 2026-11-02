//! `artemis spec-check`: speculation on against off, in one process.
//!
//! `golden` runs each golden prompt greedy for 256 tokens, free-running, first off, then on, and judges the pair by test
//! 3's near-tie rule. On and off are not expected to be identical: the verify pass's DeltaNet layers are within
//! tolerance of plain decode's, not equal to it. The runs may part where the greedy pick is a near tie, and anywhere
//! else they must not. `speed` times the speed prompts off and on: report only.
//!
//! The comparison is the gate's accuracy tool, so it fails safe: whatever it can't establish is a failure, never a pass.
//! A divergence is excused only by an off step it can read, whose top two logits are within `TIE_MARGIN`, and at which
//! the on run took one of those two tokens (the one the off run did not); a verdict with no prompts, in which
//! speculation never verified a step, or that covers only some of the golden set, is not a pass (nothing, or not
//! everything, was compared). Counts, ids and hashes only: no prompt, output or draft text is written anywhere.
use crate::{
    generate::{
        self, Anyhow, Asked, Fingerprints, Plain, Ran, Record, SpecFlags, SpecRecord, Speculating,
    },
    speed::{self, MadeFrom},
    test3::{self, TIE_MARGIN},
};
use anyhow::{bail, ensure, Context, Result};
use clap::{Subcommand, ValueEnum};
use engine::spec::{Source as SpecSource, SpecConfig, SpecState};
use engine::{Engine, Options, Sampler, Sampling, Session, SessionVerifier};
use gguf::Gguf;
use qwen36::{dims::Dims, testing::fnv1a64, tokenizer::Vocab};
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

/// The golden runs' length: the oracle's `--n 256`.
pub const GOLDEN_MAX_TOKENS: usize = test3::STEPS;
/// The context both commands open the engine and session with: production's.
pub const CTX: usize = speed::CTX;

/// Which side of the comparison a run is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// Plain decode, on the load without MTP (the gate's "off" process has no MTP memory either).
    Off,
    /// Speculation on with the configured sources. A speed run's "on" runs on the full load, MTP included.
    On,
    /// Speculation on with `--sources lookup`, on a second load without MTP (`Options::mtp` unset: no MTP VRAM, no
    /// ring copies), so it matches the gate's "lookup" process.
    Lookup,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Off => "off",
            Mode::On => "on",
            Mode::Lookup => "lookup",
        }
    }

    /// Whether the run speculates.
    pub fn speculates(self) -> bool {
        self != Mode::Off
    }

    /// The speculation config the load for this mode runs under: `cfg`, except that "off" and "lookup" drop MTP from
    /// the sources, so their load has none of its VRAM or ring copies (the gate's "off" and "lookup" processes).
    pub fn load_config(self, cfg: &SpecConfig) -> SpecConfig {
        match self {
            Mode::Off | Mode::Lookup => SpecConfig {
                sources: vec![SpecSource::Lookup],
                ..cfg.clone()
            },
            Mode::On => cfg.clone(),
        }
    }
}

// ----- the comparator ---------------------------------------------------------------------------------------------

/// How one golden prompt's off and on runs compare.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Divergence {
    pub id: String,
    /// The first step whose token differs (a run that ends sooner differs at its end-of-generation step); `None` when
    /// the runs agree on every step.
    pub at: Option<usize>,
    /// Whether the divergence is a near tie: the off run's top two logits at `at` were within `TIE_MARGIN`, and the
    /// on run's token there is one of those two. `None` without a divergence. A step that can't be read (the off run
    /// or the on run has none there, the margin is not a number from 0 up) is not a near tie.
    pub near_tie: Option<bool>,
    /// The off run's top logit minus its second at `at`, when it has that step.
    pub margin: Option<f32>,
    pub identical_tokens: bool,
    /// Whether the decoded positions' logit fingerprints agree (those before `at`, or all of them), `None` when the
    /// records hold none to compare. Report only: DeltaNet layers are expected to differ.
    pub identical_logits: Option<bool>,
    /// The on run's verify steps (`None`: its record has no speculation counts). An identical prompt with 0 is plain
    /// decode against plain decode, and proves nothing about the verify pass.
    pub verify_steps: Option<u64>,
}

impl Divergence {
    /// The runs agree on every step.
    fn is_identical(&self) -> bool {
        self.identical_tokens && self.at.is_none()
    }

    /// The runs part at a step of a near tie, which test 3's rule excuses.
    fn is_excused(&self) -> bool {
        !self.identical_tokens && self.at.is_some() && self.near_tie == Some(true)
    }

    /// One line for the console: the id and the step, never a token's text.
    pub fn line(&self) -> String {
        match (self.at, self.near_tie, self.margin) {
            _ if self.is_identical() => format!(
                "{}: identical{}",
                self.id,
                match self.verify_steps {
                    Some(0) => " (the on run made no verify step: nothing was checked)",
                    _ => "",
                }
            ),
            _ if self.is_excused() => format!(
                "{}: parts at step {} at a near tie (margin {:.4}); the rest is not compared",
                self.id,
                self.at.unwrap_or_default(),
                self.margin.unwrap_or_default()
            ),
            (Some(at), _, Some(m)) => format!(
                "{}: DIVERGES at step {at}, outside a near tie (margin {m:.4}, more than {TIE_MARGIN})",
                self.id
            ),
            (Some(at), _, None) => format!(
                "{}: DIVERGES at step {at}, where the off run has no step to judge it by",
                self.id
            ),
            _ => format!("{}: DIVERGES (inconsistent comparison)", self.id),
        }
    }
}

/// Compares an off run with an on run of the same prompt. They are the same run until the first step whose tokens
/// differ; from there the histories differ and the rest is not compared. The divergence is a near tie iff the off run's
/// top two logits at that step were at most `TIE_MARGIN` apart (test 3's rule, with the off run as the reference) and
/// the on run took one of those two (off is greedy, so the one it did not): a third-ranked token, or garbage, at a near
/// tie is a bug, not a tie.
pub fn compare(id: &str, off: &Record, on: &Record) -> Divergence {
    let (a, b) = (&off.steps, &on.steps);
    let common = a.len().min(b.len());
    let at = (0..common)
        .find(|&i| a[i].token != b[i].token)
        // One run ended where the other went on, with every step before it equal: they part at the shorter one's end.
        .or_else(|| (a.len() != b.len()).then_some(common));
    let margin = at.and_then(|i| a.get(i)).map(|s| s.logit - s.second_logit);
    // A margin that isn't a number from 0 up (a NaN, or a second above the top) is a damaged record, not a tie.
    let near_tie = at.map(|i| {
        let margin_ok = margin.is_some_and(|m| (0.0..=TIE_MARGIN).contains(&m));
        // The on run's token at `at`, which it may not have (it ended there).
        let in_top_two = match (a.get(i), b.get(i)) {
            (Some(o), Some(n)) => n.token == o.top || n.token == o.second,
            _ => false,
        };
        margin_ok && in_top_two
    });
    Divergence {
        id: id.to_string(),
        at,
        near_tie,
        margin,
        identical_tokens: at.is_none(),
        identical_logits: identical_logits(off, on, at),
        verify_steps: on.spec.as_ref().map(|s| s.steps),
    }
}

/// The decoded positions' fingerprints before `at` (every one without a divergence). A record's `positions[j]` is the
/// row `steps[j]` was picked from, so those before `at` were computed from the same history. `None` when either record
/// has no fingerprints, or there is nothing to compare.
fn identical_logits(off: &Record, on: &Record, at: Option<usize>) -> Option<bool> {
    let (a, b) = (&off.positions, &on.positions);
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let n = match at {
        Some(i) => i,
        // Identical steps: every position of both. Lists of different lengths are not the same logits.
        None if a.len() != b.len() => return Some(false),
        None => a.len(),
    };
    if n == 0 || n > a.len() || n > b.len() {
        return None;
    }
    Some(a[..n] == b[..n])
}

/// The verdict over a set of golden prompts.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GoldenVerdict {
    /// "pass" or "fail"; "subset" for a run of only some of the golden set that would otherwise pass (it is not the
    /// gate's pass).
    pub status: String,
    pub prompts: Vec<Divergence>,
    /// Prompts whose runs part at a step that is not a near tie (or that can't be judged).
    pub outside_near_tie: usize,
    /// Prompts whose runs agree on every step.
    pub identical: usize,
    /// The on runs' speculation counts, summed.
    pub spec: SpecRecord,
    /// Why a verdict with nothing outside a near tie still isn't a pass; absent when there is no reason.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub problems: Vec<String>,
}

/// Judges the prompts: "pass" iff none parts outside a near tie. Two more failures keep a vacuous comparison from
/// passing: no prompts at all, and on runs that never made a verify step (they were plain decode, so they match the off
/// runs whatever the verify pass does). `problems` says which.
pub fn judge(ds: Vec<Divergence>, spec: SpecRecord) -> GoldenVerdict {
    judge_expecting(ds, spec, false)
}

/// `judge`, where `expect_mtp` says the on runs' sources include MTP and the load had it: a third failure, no on run
/// drafted with MTP, so the gate would have tested lookup alone against plain decode.
pub fn judge_expecting(ds: Vec<Divergence>, spec: SpecRecord, expect_mtp: bool) -> GoldenVerdict {
    // A comparison that is neither identical nor excused counts against, including a damaged one.
    let outside_near_tie = ds
        .iter()
        .filter(|d| !d.is_identical() && !d.is_excused())
        .count();
    let identical = ds.iter().filter(|d| d.is_identical()).count();
    let mut problems = Vec::new();
    if ds.is_empty() {
        problems.push("no prompt was compared".to_string());
    }
    if spec.steps == 0 {
        problems.push(
            "speculation made no verify step in any on run, so nothing was checked against plain decode".to_string(),
        );
    }
    if expect_mtp && spec.drafted_mtp == 0 {
        problems.push(
            "MTP is a source and the load had it, but no on run drafted with MTP, so the comparison tested lookup alone"
                .to_string(),
        );
    }
    let pass = outside_near_tie == 0 && problems.is_empty();
    GoldenVerdict {
        status: if pass { "pass" } else { "fail" }.to_string(),
        prompts: ds,
        outside_near_tie,
        identical,
        spec,
        problems,
    }
}

/// `judge` over a run of `ds.len()` of the golden set's `total` prompts. Fewer is a subset run: its status is never
/// "pass" (a `--prompts` run, or a set missing a prompt, must not read as the gate's pass), and `problems` says so.
pub fn judge_of(
    ds: Vec<Divergence>,
    spec: SpecRecord,
    total: usize,
    expect_mtp: bool,
) -> GoldenVerdict {
    let n = ds.len();
    let mut v = judge_expecting(ds, spec, expect_mtp);
    if n != total {
        v.problems.push(format!(
            "only {n} of {total} golden prompts compared: a subset run is not the gate's pass"
        ));
        if v.status == "pass" {
            v.status = "subset".to_string();
        }
    }
    v
}

/// Which command's files `clear_out` may remove: the two commands can share `--out`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owner {
    Golden,
    Speed,
}

/// A speed run's record: `<write|edit|explain>-<seed>.<off|on|lookup>.json`. Golden ids are `NN-name`, so none matches.
fn is_speed_record(name: &str) -> bool {
    let stem = name
        .strip_suffix(".off.json")
        .or_else(|| name.strip_suffix(".on.json"))
        .or_else(|| name.strip_suffix(".lookup.json"));
    stem.and_then(|s| s.split_once('-'))
        .is_some_and(|(p, seed)| {
            speed::PROMPTS.contains(&p)
                && !seed.is_empty()
                && seed.bytes().all(|b| b.is_ascii_digit())
        })
}

/// Removes what an earlier run of `owner` left in `out`: golden's verdict and `<id>.{off,on}.json`, or speed's
/// `speed.json` and `<prompt>-<seed>.{off,on,lookup}.json`, so a run that fails or is refused leaves no stale pass or stale
/// records of a wider run beside its own. The other command's files and anything else in `out` are not touched.
pub fn clear_out(out: &Path, owner: Owner) -> Result<()> {
    let Ok(entries) = fs::read_dir(out) else {
        return Ok(());
    };
    for entry in entries {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        let record = name.ends_with(".off.json")
            || name.ends_with(".on.json")
            || name.ends_with(".lookup.json");
        let ours = match owner {
            Owner::Golden => name == "verdict.json" || (record && !is_speed_record(name)),
            Owner::Speed => name == "speed.json" || (record && is_speed_record(name)),
        };
        if ours && path.is_file() {
            fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
        }
    }
    Ok(())
}

/// Refuses a pair of records that can't be an off run and an on run of this prompt, before they are compared: a
/// comparison of the wrong things would pass for the wrong reasons.
fn check_pair(
    id: &str,
    tokens: &[u32],
    max_tokens: usize,
    off: &Record,
    on: &Record,
) -> Result<()> {
    for (mode, r) in [(Mode::Off, off), (Mode::On, on)] {
        let m = mode.name();
        ensure!(
            r.prompt_tokens == tokens,
            "{id}: the {m} run's prompt is not the golden prompt"
        );
        ensure!(
            r.settings.greedy && r.settings.max_tokens == max_tokens,
            "{id}: the {m} run was not greedy for {max_tokens} tokens"
        );
        ensure!(!r.steps.is_empty(), "{id}: the {m} run made no step");
        ensure!(
            r.settings.speculation.is_some() == (mode == Mode::On)
                && r.spec.is_some() == (mode == Mode::On),
            "{id}: the {m} run's record has speculation {}",
            if mode == Mode::On { "off" } else { "on" }
        );
    }
    same_but_speculation(id, off, on)
}

/// The two runs must differ in speculation only: the same sampling, seed, context and cache.
fn same_but_speculation(id: &str, off: &Record, on: &Record) -> Result<()> {
    let (mut a, mut b) = (off.settings.clone(), on.settings.clone());
    a.speculation = None;
    b.speculation = None;
    ensure!(
        a == b,
        "{id}: the off and on runs' settings differ beyond speculation"
    );
    Ok(())
}

/// Runs every prompt off then on through `run`, writes `<id>.off.json`, `<id>.on.json` and `verdict.json` in `out`, and
/// returns the verdict. `total` is the golden set's size (a run of fewer prompts is a subset, never a pass). `run` is
/// the engine's side (`Rig::run`) or a fake's. An earlier run's verdict and records in `out` are removed first. A pair
/// that can't be compared is an error after its records are written; the verdict is written whether it passes or not.
pub fn golden_with(
    prompts: &[(String, Vec<u32>)],
    total: usize,
    out: &Path,
    mut run: impl FnMut(Mode, &str, &[u32]) -> Result<Record>,
) -> Result<GoldenVerdict> {
    fs::create_dir_all(out).with_context(|| format!("making {}", out.display()))?;
    clear_out(out, Owner::Golden)?;
    let mut ds = Vec::with_capacity(prompts.len());
    let mut spec = SpecRecord::default();
    let mut expect_mtp = false;
    for (id, tokens) in prompts {
        let off = run(Mode::Off, id, tokens)?;
        let on = run(Mode::On, id, tokens)?;
        write_json(&out.join(format!("{id}.off.json")), &off)?;
        write_json(&out.join(format!("{id}.on.json")), &on)?;
        check_pair(id, tokens, GOLDEN_MAX_TOKENS, &off, &on)?;
        // MTP is expected to draft when the on run's sources name it and its load carried the MTP's VRAM.
        expect_mtp |= on.memory.mtp_vram_bytes > 0
            && on
                .settings
                .speculation
                .as_ref()
                .is_some_and(|s| s.sources.iter().any(|n| n == "mtp"));
        if let Some(s) = &on.spec {
            spec.steps += s.steps;
            spec.plain_steps += s.plain_steps;
            spec.drafted += s.drafted;
            spec.accepted += s.accepted;
            spec.replays += s.replays;
            spec.drafted_lookup += s.drafted_lookup;
            spec.accepted_lookup += s.accepted_lookup;
            spec.drafted_mtp += s.drafted_mtp;
            spec.accepted_mtp += s.accepted_mtp;
            spec.probes += s.probes;
        }
        ds.push(compare(id, &off, &on));
    }
    let verdict = judge_of(ds, spec, total, expect_mtp);
    write_json(&out.join("verdict.json"), &verdict)?;
    Ok(verdict)
}

/// The prompts the gate's golden set holds. A run of a set with fewer is never a pass (a deleted `expected/<id>.json`
/// or a smaller directory must not shrink the gate), unless a test asks for a small set.
pub const GOLDEN_SET_SIZE: usize = 20;

/// How many prompts a run must compare to be the gate's pass: the set's own size, at least `GOLDEN_SET_SIZE`
/// (`allow_small` lifts the floor, for tests).
pub fn required_prompts(total: usize, allow_small: bool) -> usize {
    if allow_small {
        total
    } else {
        total.max(GOLDEN_SET_SIZE)
    }
}

/// The golden ids to run: all of them, or the asked-for ones in the golden set's order. An id the set doesn't hold is an
/// error, so a typo can't shrink the gate.
pub fn select(all: &[String], asked: &[String]) -> Result<Vec<String>> {
    if asked.is_empty() {
        return Ok(all.to_vec());
    }
    if let Some(bad) = asked.iter().find(|a| !all.contains(a)) {
        bail!("--prompts: {bad:?} is not a golden prompt");
    }
    Ok(all
        .iter()
        .filter(|id| asked.contains(id))
        .cloned()
        .collect())
}

// ----- speed ------------------------------------------------------------------------------------------------------

/// FNV-1a 64 of a completion's token ids (their little-endian bytes), as 16 hex digits: the hash `generate::fingerprint`
/// takes of logits, applied to ids. Never of text.
pub fn completion_fnv(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
    format!("{:016x}", fnv1a64(&bytes))
}

/// One timed run.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SpeedRun {
    pub prompt: String,
    pub seed: u64,
    pub mode: &'static str,
    pub decoded: usize,
    pub generate_s: f64,
    /// `decoded / generate_s`.
    pub decode_tok_s: f64,
    pub completion_tokens: usize,
    pub completion_fnv: String,
    /// Routed-expert misses during decode (the record's `cache.decode_misses`).
    pub decode_misses: u64,
    /// The expert slots per layer the run's load had: the load with MTP's memory has about one fewer.
    pub slots_per_layer: usize,
    pub spec: Option<SpecRecord>,
}

/// A prompt and seed run off and on.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SpeedPair {
    pub prompt: String,
    pub seed: u64,
    pub off_tok_s: f64,
    pub on_tok_s: f64,
    /// on / off.
    pub ratio: f64,
    pub identical_completion: bool,
    /// The same prompt and seed with MTP dropped (the "lookup" mode), and on / lookup: MTP's gain. `None` without
    /// that mode.
    pub lookup_tok_s: Option<f64>,
    pub mtp_gain: Option<f64>,
}

/// `speed`'s output: report only, no verdict.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SpeedReport {
    pub status: &'static str,
    /// The modes asked for, in the order given, and whether each speculating run began with a fresh `SpecState`.
    pub modes: Vec<&'static str>,
    pub fresh_state: bool,
    pub runs: Vec<SpeedRun>,
    pub pairs: Vec<SpeedPair>,
    pub off_median: f64,
    pub on_median: f64,
    pub ratio: f64,
    /// The "lookup" mode's median, and `on_median / lookup_median`: whether MTP pays (the gate's bar is 1.02). `None`
    /// without that mode.
    pub lookup_median: Option<f64>,
    pub mtp_gain: Option<f64>,
}

fn ratio(on: f64, off: f64) -> f64 {
    if off > 0.0 {
        on / off
    } else {
        0.0
    }
}

/// A run's figures from its record. A run that decoded nothing has no speed.
fn speed_run(prompt: &str, seed: u64, mode: Mode, r: &Record) -> Result<SpeedRun> {
    ensure!(
        r.timings.decoded > 0 && r.timings.generate_s > 0.0,
        "{prompt} seed {seed}: the {} run decoded no token, so it has no speed",
        mode.name()
    );
    ensure!(
        r.settings.speculation.is_some() == mode.speculates()
            && r.spec.is_some() == mode.speculates(),
        "{prompt} seed {seed}: the {} run's record has speculation {}",
        mode.name(),
        if mode.speculates() { "off" } else { "on" }
    );
    if mode == Mode::Lookup {
        // The mode's whole point: a run that drafted with MTP, or loaded for it, would measure the wrong thing.
        let lookup_only = r
            .settings
            .speculation
            .as_ref()
            .is_some_and(|s| s.sources == ["lookup"]);
        ensure!(
            lookup_only
                && r.memory.mtp_vram_bytes == 0
                && r.spec
                    .as_ref()
                    .is_some_and(|s| s.drafted_mtp == 0 && s.probes == 0),
            "{prompt} seed {seed}: the lookup run used MTP; it must load and draft without it"
        );
    }
    let end = r.steps.len() - usize::from(r.stop == generate::Stop::EndOfGeneration);
    let ids: Vec<u32> = r.steps[..end].iter().map(|s| s.token).collect();
    Ok(SpeedRun {
        prompt: prompt.to_string(),
        seed,
        mode: mode.name(),
        decoded: r.timings.decoded,
        generate_s: r.timings.generate_s,
        decode_tok_s: r.timings.decoded as f64 / r.timings.generate_s,
        completion_tokens: ids.len(),
        completion_fnv: completion_fnv(&ids),
        decode_misses: r.cache.decode_misses,
        slots_per_layer: r.cache.slots_per_layer,
        spec: r.spec.clone(),
    })
}

/// Runs each prompt at each seed off then on through `run`: `speed_modes` with only the `Off` and `On` modes.
pub fn speed_with(
    prompts: &[(String, Vec<u32>)],
    seeds: &[u64],
    out: &Path,
    run: impl FnMut(Mode, &str, u64, &[u32]) -> Result<Record>,
) -> Result<SpeedReport> {
    speed_modes(
        prompts,
        seeds,
        out,
        &[Mode::Off, Mode::On],
        false,
        false,
        run,
    )
}

/// Runs each prompt at each seed in each of `modes` through `run`, writes each record as `<prompt>-<seed>.<mode>.json`
/// and the report as `speed.json` in `out`, and returns the report.
///
/// With `separate_loads` the modes run in two passes, because they use two loads and two engines never share the GPU:
/// "on" (the full load, MTP's memory included) for every prompt and seed, then "off" and "lookup" (the load without
/// MTP, as the gate's "off" and "lookup" processes have it) pair by pair, off first. `run` swaps loads at the pass
/// change. Without it there is one pass, off, on then lookup, pair by pair. An untimed run of a pass's first mode on
/// the first prompt and seed comes before the pass and is not reported, so the first timed run doesn't start with a
/// colder expert cache than the ones after it. `fresh_state` is only recorded (`run` builds the states). An earlier
/// run's report and records in `out` are removed first.
pub fn speed_modes(
    prompts: &[(String, Vec<u32>)],
    seeds: &[u64],
    out: &Path,
    modes: &[Mode],
    fresh_state: bool,
    separate_loads: bool,
    mut run: impl FnMut(Mode, &str, u64, &[u32]) -> Result<Record>,
) -> Result<SpeedReport> {
    ensure!(!modes.is_empty(), "--modes names no mode");
    for (i, m) in modes.iter().enumerate() {
        ensure!(!modes[..i].contains(m), "--modes names {} twice", m.name());
    }
    fs::create_dir_all(out).with_context(|| format!("making {}", out.display()))?;
    clear_out(out, Owner::Speed)?;
    let canonical = [Mode::Off, Mode::On, Mode::Lookup];
    let passes: Vec<Vec<Mode>> = if separate_loads {
        [vec![Mode::On], vec![Mode::Off, Mode::Lookup]]
            .into_iter()
            .map(|p| {
                p.into_iter()
                    .filter(|m| modes.contains(m))
                    .collect::<Vec<_>>()
            })
            .filter(|p| !p.is_empty())
            .collect()
    } else {
        vec![canonical
            .into_iter()
            .filter(|m| modes.contains(m))
            .collect()]
    };
    let mut runs: Vec<SpeedRun> = Vec::new();
    let mut records: Vec<(String, u64, Mode, Record)> = Vec::new();
    for pass in &passes {
        if let (Some((name, tokens)), Some(&seed)) = (prompts.first(), seeds.first()) {
            run(pass[0], name, seed, tokens)?;
        }
        for (name, tokens) in prompts {
            for &seed in seeds {
                for &mode in pass {
                    let r = run(mode, name, seed, tokens)?;
                    write_json(&out.join(format!("{name}-{seed}.{}.json", mode.name())), &r)?;
                    ensure!(
                        r.prompt_tokens == *tokens,
                        "{name} seed {seed}: the {} run's prompt is not the speed prompt",
                        mode.name()
                    );
                    runs.push(speed_run(name, seed, mode, &r)?);
                    records.push((name.clone(), seed, mode, r));
                }
            }
        }
    }
    // Every speculating run must differ from its prompt's off run in speculation only.
    for (name, seed, mode, r) in &records {
        if *mode == Mode::Off {
            continue;
        }
        if let Some((_, _, _, off)) = records
            .iter()
            .find(|(n, s, m, _)| n == name && s == seed && *m == Mode::Off)
        {
            // The loads differ in MTP's memory, so in their slots per layer: the other settings must agree.
            let (mut a, mut b) = (off.clone(), r.clone());
            a.settings.cache_slots = 0;
            b.settings.cache_slots = 0;
            same_but_speculation(&format!("{name} seed {seed}"), &a, &b)?;
        }
    }
    let tok_s = |name: &str, seed: u64, mode: Mode| {
        runs.iter()
            .find(|r| r.prompt == name && r.seed == seed && r.mode == mode.name())
    };
    let mut pairs = Vec::new();
    for (name, _) in prompts {
        for &seed in seeds {
            let (Some(off), Some(on)) = (tok_s(name, seed, Mode::Off), tok_s(name, seed, Mode::On))
            else {
                continue;
            };
            let lookup = tok_s(name, seed, Mode::Lookup).map(|r| r.decode_tok_s);
            pairs.push(SpeedPair {
                prompt: name.clone(),
                seed,
                off_tok_s: off.decode_tok_s,
                on_tok_s: on.decode_tok_s,
                ratio: ratio(on.decode_tok_s, off.decode_tok_s),
                identical_completion: on.completion_fnv == off.completion_fnv,
                lookup_tok_s: lookup,
                mtp_gain: lookup.map(|l| ratio(on.decode_tok_s, l)),
            });
        }
    }
    let med = |m: Mode| {
        let v: Vec<f64> = runs
            .iter()
            .filter(|r| r.mode == m.name())
            .map(|r| r.decode_tok_s)
            .collect();
        (!v.is_empty()).then(|| speed::median(&v))
    };
    let (off_median, on_median) = (med(Mode::Off).unwrap_or(0.0), med(Mode::On).unwrap_or(0.0));
    let lookup_median = med(Mode::Lookup);
    let report = SpeedReport {
        status: "report",
        modes: modes.iter().map(|m| m.name()).collect(),
        fresh_state,
        runs,
        pairs,
        off_median,
        on_median,
        ratio: ratio(on_median, off_median),
        lookup_median,
        mtp_gain: lookup_median
            .filter(|_| on_median > 0.0)
            .map(|l| ratio(on_median, l)),
    };
    write_json(&out.join("speed.json"), &report)?;
    Ok(report)
}

/// The speed prompts as the gate runs them, by name: `write` and `explain` from the committed token files, `edit` from
/// `edit_tokens` (the runner passes the committed `speed/prompts/edit.tokens`), each checked against `made-from.json` (a
/// stale prompt would time a different run).
pub fn speed_prompts(speed_dir: &Path, edit_tokens: &Path) -> Result<Vec<(String, Vec<u32>)>> {
    let made: std::collections::BTreeMap<String, MadeFrom> = serde_json::from_slice(
        &fs::read(speed_dir.join("made-from.json"))
            .with_context(|| format!("reading {}", speed_dir.join("made-from.json").display()))?,
    )
    .context("parsing made-from.json")?;
    let mut out = Vec::new();
    for name in speed::PROMPTS {
        let path = match name {
            "edit" => edit_tokens.to_path_buf(),
            other => speed_dir.join("prompts").join(format!("{other}.tokens")),
        };
        let ids = read_tokens(&path)?;
        let want = made
            .get(name)
            .with_context(|| format!("made-from.json has no {name} prompt"))?;
        ensure!(
            MadeFrom::of(&ids) == *want,
            "{}: not the {name} prompt made-from.json records ({} tokens, fnv {})",
            path.display(),
            want.tokens,
            want.fnv1a64
        );
        out.push((name.to_string(), ids));
    }
    Ok(out)
}

// ----- the engine's side ------------------------------------------------------------------------------------------

/// What one run generates.
pub struct Job {
    pub sampling: Sampling,
    pub greedy: bool,
    pub seed: u64,
    pub max_tokens: usize,
    /// Fingerprint every position's logits, for `identical_logits`. The hashing sits inside the timed region (about a
    /// millisecond a token), so speed runs leave it off.
    pub fingerprints: bool,
}

impl Job {
    /// Test 3's run: greedy, 256 tokens, with fingerprints.
    pub fn golden() -> Job {
        Job {
            sampling: Sampling::GREEDY,
            greedy: true,
            seed: 0,
            max_tokens: GOLDEN_MAX_TOKENS,
            fingerprints: true,
        }
    }

    /// A speed run: production's sampling, 700 tokens, no fingerprints.
    pub fn speed(seed: u64) -> Job {
        Job {
            sampling: Sampling::DEFAULT,
            greedy: false,
            seed,
            max_tokens: speed::MAX_TOKENS,
            fingerprints: false,
        }
    }
}

/// One loaded engine and its session: both modes run on it, a reset between runs. Speculation is on at the load (the
/// verify pass needs it); an off run makes plain decode calls all the same.
pub struct Rig {
    // The session drops before the engine that made it.
    s: Session,
    e: Engine,
    /// The speculation state the on runs use: one for the whole process, as the server's is engine-wide, so its
    /// controller's accept and cost estimates carry from one run to the next (the tokens' rule is unaffected, but a
    /// `--prompts` rerun of one prompt is not the run the full pass made). A test may replace it between runs.
    pub state: SpecState,
    /// Every speculating run starts with a new `SpecState` (`--fresh-state`), as a gate run, a fresh process, does.
    fresh: bool,
    eog: Vec<u32>,
    load_s: f64,
}

/// The options a rig opens its engine with for `cfg`: production's context and cache, speculation on (the verify pass
/// needs it), and the MTP exactly when `cfg` asks for it, so a config without it loads none of MTP's memory.
pub fn rig_options(cfg: &SpecConfig, ctx: usize, device_uuid: Option<[u8; 16]>) -> Result<Options> {
    Ok(Options {
        device_uuid,
        ctx,
        // What `artemis generate` opens with unless told otherwise.
        cache_policy: engine::CachePolicyKind::Adaptive,
        // `Session::verify` asserts it.
        speculation: true,
        mtp: generate::mtp_options_for_flags(Some(cfg), ctx)?,
        ..Options::default()
    })
}

/// Makes `state` a new one with the same config when `fresh`; says whether it did. A rig calls this before every
/// speculating run.
pub fn renew_state(state: &mut SpecState, fresh: bool) -> bool {
    if fresh {
        *state = SpecState::new(state.config().clone());
    }
    fresh
}

impl Rig {
    /// Loads `model`; the "on" runs speculate with `cfg`.
    pub fn open(model: &Path, device_uuid: Option<[u8; 16]>, cfg: SpecConfig) -> Result<Rig> {
        let g = Gguf::open(model).with_context(|| format!("opening {}", model.display()))?;
        let eog = Vocab::from_gguf(&g).map_err(anyhow::Error::msg)?.eog();
        Rig::from_gguf(&g, &Dims::QWEN36, device_uuid, CTX, cfg, eog)
    }

    pub fn from_gguf(
        g: &Gguf,
        dims: &Dims,
        device_uuid: Option<[u8; 16]>,
        ctx: usize,
        cfg: SpecConfig,
        eog: Vec<u32>,
    ) -> Result<Rig> {
        let opts = rig_options(&cfg, ctx, device_uuid)?;
        Rig::from_options(g, dims, opts, ctx, cfg, eog)
    }

    /// `from_gguf` with the engine's options given: setup opens the engine as the user's config would (`cache_slots`,
    /// `f32_prefill`), not as `rig_options` does. `opts.speculation` must be set, and `opts.mtp` is what `cfg` asks.
    pub fn from_options(
        g: &Gguf,
        dims: &Dims,
        opts: Options,
        ctx: usize,
        cfg: SpecConfig,
        eog: Vec<u32>,
    ) -> Result<Rig> {
        let t = Instant::now();
        let e = Engine::open(g, dims, &opts)?;
        let s = e.session(ctx)?;
        let load_s = t.elapsed().as_secs_f64();
        eprintln!(
            "loaded on {} in {load_s:.1} s; {ctx}-position session, {} expert slots per layer",
            e.device_name(),
            e.cache_slots()
        );
        Ok(Rig {
            s,
            e,
            state: SpecState::new(cfg),
            fresh: false,
            eog,
            load_s,
        })
    }

    /// The loaded engine: its device, driver and memory figures.
    pub fn engine(&self) -> &Engine {
        &self.e
    }

    /// Every speculating run starts with a new `SpecState` (the gate's runs are fresh processes).
    pub fn fresh_state(mut self, fresh: bool) -> Rig {
        self.fresh = fresh;
        self
    }

    /// Generates from `prompt` in a reset session, off (plain decode) or on (a speculating loop), and records it as
    /// `artemis generate --json` does.
    pub fn run(&mut self, mode: Mode, prompt: &[u32], job: &Job) -> Result<Record> {
        ensure!(!prompt.is_empty(), "the prompt is empty");
        ensure!(
            prompt.len() <= self.s.ctx(),
            "the prompt is longer than the context"
        );
        if mode == Mode::Lookup {
            // The mode loads without MTP's memory; a rig that has it would time the wrong thing.
            ensure!(
                self.e.mtp().is_none() && !self.state.config().has(SpecSource::Mtp),
                "a lookup run needs a load without MTP"
            );
        }
        if mode.speculates() {
            renew_state(&mut self.state, self.fresh);
        }
        self.s.reset(&self.e)?;
        let depths_before = self.state.totals().mtp_depth_counts;
        let (e, ctx) = (&self.e, self.s.ctx());
        let (at_open, loaded) = (e.vram_at_open(), e.vram_heap()?);
        let memory = generate::Memory::of(e);
        let mut positions = Fingerprints::new(job.fingerprints, job.max_tokens + 1);
        let t0 = Instant::now();
        let logits = self.s.prefill(e, prompt)?;
        positions.add(&logits);
        let prompt_s = t0.elapsed().as_secs_f64();
        let after_prefill = e.stats();
        let mut sampler = Sampler::new(job.sampling, job.seed);
        let plain_greedy = job.sampling.temp <= 0.0 && job.sampling.presence_penalty == 0.0;
        let room = ctx - self.s.pos();
        // The prompt's row was fingerprinted above; each later row is as it is picked from.
        let mut first_row = true;
        let mut pick = |l: &[f32], top: u32| {
            if !std::mem::take(&mut first_row) {
                positions.add(l);
            }
            if plain_greedy {
                top
            } else {
                sampler.sample(l)
            }
        };
        let speculation = mode.speculates().then(|| self.state.config().clone());
        let t1 = Instant::now();
        let (steps, stop, delta) = match mode {
            Mode::On | Mode::Lookup => {
                let mut stepper = Speculating::new(
                    &mut self.state,
                    Anyhow(SessionVerifier { e, s: &mut self.s }),
                    prompt,
                );
                let (steps, stop) = generate::run(
                    logits,
                    job.max_tokens,
                    room,
                    &self.eog,
                    &mut pick,
                    &mut stepper,
                    |_| Ok(()),
                )?;
                (steps, stop, Some(stepper.delta))
            }
            Mode::Off => {
                let s = &mut self.s;
                let (steps, stop) = generate::run(
                    logits,
                    job.max_tokens,
                    room,
                    &self.eog,
                    &mut pick,
                    &mut Plain(|t| Ok(s.decode(e, t)?)),
                    |_| Ok(()),
                )?;
                (steps, stop, None)
            }
        };
        let generate_s = t1.elapsed().as_secs_f64();
        let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
        let depths_after = self.state.totals().mtp_depth_counts;
        let record = generate::record(
            e,
            &Asked {
                greedy: job.greedy,
                sampling: job.sampling,
                seed: job.seed,
                max_tokens: job.max_tokens,
                ctx,
                cache_slots: None,
                force_swap_every_token: false,
                all_logits: false,
                fingerprints: job.fingerprints,
                profiled: false,
                f32_prefill: false,
                no_staging: false,
                stage_from: None,
                upload_cap_mib: None,
                no_lend: false,
                base_time: false,
                cpu_share: 0.0,
                speculation,
            },
            Ran {
                prompt: prompt.to_vec(),
                // Positions run: every row picked after the prompt's, a stop token's row included.
                decoded: steps.len().saturating_sub(1),
                steps,
                stop,
                positions: positions.into_list(),
                load_s: self.load_s,
                prompt_s,
                generate_s,
                stats: e.stats(),
                after_prefill,
                memory,
                vram: generate::Vram {
                    heap_mib: mib(at_open.size),
                    budget_at_open_mib: mib(at_open.budget),
                    usage_at_open_mib: mib(at_open.usage),
                    budget_mib: mib(loaded.budget),
                    usage_mib: mib(loaded.usage),
                },
                spec: delta,
            },
            None,
        );
        // MTP's counts per draft depth over this run's steps (the state's, less what it held before).
        Ok(Record {
            spec: record
                .spec
                .map(|s| s.with_depths(&depths_before, &depths_after)),
            ..record
        })
    }
}

/// The loads a `speed` command makes, one at a time (two engines never share the GPU): the full one for "on" and the
/// one without MTP for "off" and "lookup", each made when first needed, after the other is dropped. A config with no
/// MTP source needs only the one.
struct SpeedRigs {
    model: PathBuf,
    uuid: Option<[u8; 16]>,
    cfg: SpecConfig,
    fresh: bool,
    full: Option<Rig>,
    lookup: Option<Rig>,
}

impl SpeedRigs {
    fn rig(&mut self, mode: Mode) -> Result<&mut Rig> {
        let slot = if mode != Mode::On && self.cfg.has(SpecSource::Mtp) {
            // The VRAM of the full load is freed before the second is made.
            self.full = None;
            &mut self.lookup
        } else {
            self.lookup = None;
            &mut self.full
        };
        if slot.is_none() {
            let cfg = mode.load_config(&self.cfg);
            *slot = Some(Rig::open(&self.model, self.uuid, cfg)?.fresh_state(self.fresh));
        }
        Ok(slot.as_mut().expect("just made"))
    }
}

// ----- the command ------------------------------------------------------------------------------------------------

/// `artemis spec-check`'s subcommands.
#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// Run each golden prompt greedy for 256 tokens with speculation off, then on, in one process (one model load), and
    /// judge each pair by test 3's near-tie rule: they may part where the greedy pick is a near tie and nowhere else.
    /// Writes `<id>.off.json`, `<id>.on.json` and `verdict.json` in --out (an earlier run's are removed first); exits 1
    /// unless the verdict passes. The golden set is `expected/*.json`'s ids; a run of only some of them (--prompts) has
    /// status "subset", never "pass". One speculation state serves every on run, so the controller carries over from
    /// prompt to prompt.
    Golden {
        #[arg(long)]
        model: PathBuf,
        /// testdata/golden: `expected/<id>.json` names the prompts, `prompts/<id>.tokens` and `made-from.json` check them.
        #[arg(long, default_value = "testdata/golden")]
        golden: PathBuf,
        /// Only these golden prompts (default: all), by id.
        #[arg(long, value_delimiter = ',')]
        prompts: Vec<String>,
        #[arg(long)]
        out: PathBuf,
        /// The GPU's UUID, when there is more than one discrete GPU.
        #[arg(long)]
        device: Option<String>,
        /// For tests only: let a golden set of fewer than 20 prompts pass. Without it a smaller set is a subset run.
        #[arg(long, hide = true)]
        allow_small_golden: bool,
        /// The speculation flags, as `generate` takes them (--sources, --mtp-window ...): they set the on runs.
        #[command(flatten)]
        spec: SpecFlags,
    },
    /// Time the write, edit and explain speed prompts at seeds 1234 and 1235 (0.6/0.95/20, 700 tokens) with speculation
    /// off, then on, in one process. Prints and writes each run's decode tok/s, the on/off ratio, the completion's token
    /// hash and the speculation counts to `speed.json` in --out. Report only. The speculation flags set the on runs.
    /// With `--modes off,lookup,on` it also times speculation with `--sources lookup` on a second load without MTP, as
    /// the gate's "lookup" process has it, and reports MTP's gain (on over lookup).
    Speed {
        #[arg(long)]
        model: PathBuf,
        /// testdata/speed: `prompts/{write,edit,explain}.tokens` and `made-from.json`.
        #[arg(long, default_value = "testdata/speed")]
        speed: PathBuf,
        /// The edit prompt's token file (`speed/prompts/edit.tokens`, committed like the other two).
        #[arg(long)]
        edit_tokens: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// The GPU's UUID, when there is more than one discrete GPU.
        #[arg(long)]
        device: Option<String>,
        /// The modes to time, comma separated: `off`, `lookup` (speculation on with lookup alone, on a load without
        /// MTP) and `on`.
        #[arg(long, value_enum, value_delimiter = ',', default_value = "off,on")]
        modes: Vec<Mode>,
        /// Every speculating run starts with a new speculation controller, as each of the gate's runs (a fresh process)
        /// does. Without it one state serves the whole process, as a server's does.
        #[arg(long)]
        fresh_state: bool,
        #[command(flatten)]
        spec: SpecFlags,
    },
}

fn write_json<T: Serialize>(path: &Path, v: &T) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(v)?)
        .with_context(|| format!("writing {}", path.display()))
}

use generate::read_tokens;

/// The golden set's ids, sorted: the names of `expected/*.json`, the oracle files test 3 judges against.
fn golden_ids(golden: &Path) -> Result<Vec<String>> {
    let dir = golden.join("expected");
    let mut ids = Vec::new();
    for entry in fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|x| x == "json") {
            if let Some(id) = path.file_stem().and_then(|s| s.to_str()) {
                ids.push(id.to_string());
            }
        }
    }
    ids.sort();
    Ok(ids)
}

/// Every golden prompt, by id: an expected id with no `prompts/<id>.tokens` is an error, as is a token file other than
/// the one `made-from.json` says the oracle ran (`expected.prompt` is the file's FNV-1a 64), so a partial or stale set
/// can't be judged as the golden set.
pub fn golden_set(golden: &Path) -> Result<Vec<(String, Vec<u32>)>> {
    let made_path = golden.join("made-from.json");
    let made: serde_json::Value = serde_json::from_slice(
        &fs::read(&made_path).with_context(|| format!("reading {}", made_path.display()))?,
    )
    .with_context(|| format!("parsing {}", made_path.display()))?;
    // The oracle's files and the record of what they were made from must name the same prompts: a deleted
    // `expected/<id>.json` would otherwise drop that prompt from the run without a trace.
    let ids = golden_ids(golden)?;
    let named: Vec<&String> = made
        .as_object()
        .map_or_else(Vec::new, |m| m.keys().collect());
    let mut named_sorted = named.clone();
    named_sorted.sort();
    ensure!(
        named_sorted.iter().copied().eq(ids.iter()),
        "{}: names {} prompts but expected/ holds {}{}",
        made_path.display(),
        named.len(),
        ids.len(),
        {
            let mut diff: Vec<String> = ids
                .iter()
                .filter(|i| !named.contains(i))
                .map(|i| format!("{i} (in expected/ only)"))
                .chain(
                    named
                        .iter()
                        .filter(|n| !ids.contains(n))
                        .map(|n| format!("{n} (in made-from.json only)")),
                )
                .collect();
            diff.truncate(5);
            format!(": {}", diff.join(", "))
        }
    );
    ids.into_iter()
        .map(|id| {
            let path = golden.join("prompts").join(format!("{id}.tokens"));
            let ids = read_tokens(&path).with_context(|| {
                format!("golden prompt {id}: expected/{id}.json has no readable tokens file")
            })?;
            let want = made[&id]["expected"]["prompt"].as_str();
            ensure!(
                want == Some(completion_fnv(&ids).as_str()),
                "{}: not the prompt expected/{id}.json was made from (made-from.json)",
                path.display()
            );
            Ok((id, ids))
        })
        .collect()
}

fn device_uuid(s: Option<&str>) -> Result<Option<[u8; 16]>> {
    s.map(|s| engine::parse_uuid(s).with_context(|| format!("--device {s:?} is not a UUID")))
        .transpose()
}

/// Runs a subcommand and returns the process's exit code: 1 for a golden verdict that isn't a pass.
pub fn run(cmd: Cmd) -> Result<i32> {
    match cmd {
        Cmd::Golden {
            model,
            golden,
            prompts,
            out,
            device,
            allow_small_golden,
            spec,
        } => {
            // First of all, before any refusal: an earlier run's verdict must not outlive a run that doesn't finish.
            clear_out(&out, Owner::Golden)?;
            // Everything that can be checked without the GPU is, before the load.
            let set = golden_set(&golden)?;
            ensure!(
                !set.is_empty(),
                "{} holds no golden prompt",
                golden.display()
            );
            let all: Vec<String> = set.iter().map(|(id, _)| id.clone()).collect();
            let total = required_prompts(all.len(), allow_small_golden);
            let chosen = select(&all, &prompts)?;
            let prompts: Vec<_> = set
                .into_iter()
                .filter(|(id, _)| chosen.contains(id))
                .collect();
            let uuid = device_uuid(device.as_deref())?;
            let Some(cfg) = spec.config(0.0, false)? else {
                bail!(
                    "spec-check golden judges speculation on against off: drop --speculation off"
                );
            };
            let mut rig = Rig::open(&model, uuid, cfg)?;
            let job = Job::golden();
            let v = golden_with(&prompts, total, &out, |mode, id, tokens| {
                eprintln!("spec-check golden: {id} {}", mode.name());
                let r = rig.run(mode, tokens, &job)?;
                // A prompt whose on run never verified is plain decode against plain decode: say so per prompt.
                if let Some(s) = &r.spec {
                    eprintln!(
                        "spec-check golden: {id} on: {} verify steps, {} plain, {} drafted, {} kept",
                        s.steps, s.plain_steps, s.drafted, s.accepted
                    );
                }
                Ok(r)
            })?;
            for d in &v.prompts {
                println!("spec-check golden: {}", d.line());
            }
            println!(
                "spec-check golden: {}: {} of {} identical, {} outside a near tie; {} verify steps, {} plain, {} drafted, {} kept",
                v.status,
                v.identical,
                v.prompts.len(),
                v.outside_near_tie,
                v.spec.steps,
                v.spec.plain_steps,
                v.spec.drafted,
                v.spec.accepted
            );
            for p in &v.problems {
                println!("spec-check golden: problem: {p}");
            }
            Ok(i32::from(v.status != "pass"))
        }
        Cmd::Speed {
            model,
            speed: dir,
            edit_tokens,
            out,
            device,
            modes,
            fresh_state,
            spec,
        } => {
            clear_out(&out, Owner::Speed)?;
            let Some(cfg) = spec.config(0.0, false)? else {
                bail!("spec-check speed times speculation on against off: drop --speculation off");
            };
            let uuid = device_uuid(device.as_deref())?;
            let prompts = speed_prompts(&dir, &edit_tokens)?;
            let separate = cfg.has(SpecSource::Mtp);
            let mut rigs = SpeedRigs {
                model,
                uuid,
                cfg,
                fresh: fresh_state,
                full: None,
                lookup: None,
            };
            let report = speed_modes(
                &prompts,
                &speed::SEEDS,
                &out,
                &modes,
                fresh_state,
                separate,
                |mode, name, seed, tokens| {
                    eprintln!("spec-check speed: {name} seed {seed} {}", mode.name());
                    rigs.rig(mode)?.run(mode, tokens, &Job::speed(seed))
                },
            )?;
            for r in &report.runs {
                let spec = r.spec.as_ref().map_or(String::new(), |s| {
                    format!(
                        "; {} verify steps, {} plain, {} drafted, {} kept, {} replays; lookup {} drafted, {} kept; MTP {} drafted, {} kept, {} probes",
                        s.steps,
                        s.plain_steps,
                        s.drafted,
                        s.accepted,
                        s.replays,
                        s.drafted_lookup,
                        s.accepted_lookup,
                        s.drafted_mtp,
                        s.accepted_mtp,
                        s.probes
                    )
                });
                println!(
                    "spec-check speed: {} seed {} {}: {:.1} tok/s ({} tokens in {:.1} s); completion {}{spec}",
                    r.prompt, r.seed, r.mode, r.decode_tok_s, r.decoded, r.generate_s, r.completion_fnv
                );
            }
            for p in &report.pairs {
                println!(
                    "spec-check speed: {} seed {}: on/off {:.3} ({:.1} against {:.1} tok/s); completions {}",
                    p.prompt,
                    p.seed,
                    p.ratio,
                    p.on_tok_s,
                    p.off_tok_s,
                    if p.identical_completion { "identical" } else { "differ" }
                );
            }
            println!(
                "spec-check speed: median on {:.1} tok/s, off {:.1} tok/s, ratio {:.3}; reported, not gated",
                report.on_median, report.off_median, report.ratio
            );
            if let (Some(l), Some(g)) = (report.lookup_median, report.mtp_gain) {
                println!(
                    "spec-check speed: median lookup {l:.1} tok/s; MTP gain (on over lookup) {g:.3}; reported, not gated"
                );
            }
            Ok(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::{fingerprint, CacheStats, Settings, SpecSettings, Step, Stop, Timings};
    use clap::Parser;

    /// A step that took `token` over a second-best logit `margin` below it.
    fn step(token: u32, margin: f32) -> Step {
        Step {
            token,
            top: token,
            logit: margin,
            second: token + 1000,
            second_logit: 0.0,
        }
    }

    /// A greedy record of `steps` (token, margin) for a 3-token prompt, with `positions` as its fingerprints.
    fn record(steps: &[(u32, f32)], positions: &[&str], on: bool) -> Record {
        Record {
            format: generate::FORMAT.into(),
            prompt_tokens: vec![1, 2, 3],
            settings: Settings {
                greedy: true,
                temp: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                presence_penalty: 0.0,
                seed: 0,
                max_tokens: GOLDEN_MAX_TOKENS,
                ctx: CTX,
                cache_slots: 100,
                force_swap_every_token: false,
                all_logits: false,
                cache_slots_asked: None,
                fingerprints: !positions.is_empty(),
                profiled: false,
                f32_prefill: false,
                no_staging: false,
                stage_from: None,
                cache_policy: "adaptive".into(),
                upload_cap_mib: None,
                no_lend: false,
                base_time: false,
                cpu_share: 0.0,
                speculation: on.then(|| SpecSettings::from(&SpecConfig::default())),
            },
            steps: steps.iter().map(|&(t, m)| step(t, m)).collect(),
            stop: Stop::MaxTokens,
            positions: positions.iter().map(|p| p.to_string()).collect(),
            timings: Timings {
                decoded: steps.len().saturating_sub(1),
                generate_s: 2.0,
                ..Timings::default()
            },
            cache: CacheStats::default(),
            memory: Default::default(),
            vram: Default::default(),
            prefill: Default::default(),
            warm: None,
            spec: on.then_some(SpecRecord {
                steps: 2,
                plain_steps: 1,
                drafted: 6,
                accepted: 4,
                replays: 1,
                ..SpecRecord::default()
            }),
        }
    }

    fn off(steps: &[(u32, f32)]) -> Record {
        record(steps, &[], false)
    }

    fn on(steps: &[(u32, f32)]) -> Record {
        record(steps, &[], true)
    }

    fn some_spec() -> SpecRecord {
        SpecRecord {
            steps: 3,
            plain_steps: 1,
            drafted: 9,
            accepted: 6,
            replays: 1,
            ..SpecRecord::default()
        }
    }

    /// Four steps at a wide margin, the third (index 2) parted by the on run at the off run's `margin`.
    fn parted_at(margin: f32) -> (Record, Record) {
        (
            off(&[(5, 9.0), (6, 9.0), (7, margin), (8, 9.0)]),
            // The on run takes off's second-ranked token (`step` makes it token + 1000) and then goes its own way:
            // nothing after the part is compared, whatever it holds.
            on(&[(5, 9.0), (6, 9.0), (1007, 0.0), (80, 0.0), (90, 0.0)]),
        )
    }

    #[test]
    fn identical_runs_pass() {
        let steps = [(5, 1.0), (6, 2.0), (7, 3.0)];
        let d = compare("01-code", &off(&steps), &on(&steps));
        assert_eq!(d.at, None);
        assert_eq!(d.near_tie, None);
        assert_eq!(d.margin, None);
        assert!(d.identical_tokens);
        let v = judge(vec![d], some_spec());
        assert_eq!(v.status, "pass");
        assert_eq!((v.outside_near_tie, v.identical), (0, 1));
        assert!(v.problems.is_empty());
    }

    #[test]
    fn a_first_divergence_at_a_near_tie_passes_and_the_rest_is_not_compared() {
        // Exactly TIE_MARGIN apart is a near tie (the second logit is 0, so the difference is exact); 0.0501 is not.
        let (a, b) = parted_at(TIE_MARGIN);
        let d = compare("x", &a, &b);
        assert_eq!(
            (d.at, d.near_tie, d.margin),
            (Some(2), Some(true), Some(TIE_MARGIN))
        );
        assert!(!d.identical_tokens);
        let v = judge(vec![d], some_spec());
        assert_eq!(v.status, "pass");
        // The steps after the divergence are wildly different (and the on run is longer): not compared.
        assert_eq!((v.outside_near_tie, v.identical), (0, 0));

        let (a, b) = parted_at(0.0501);
        let d = compare("x", &a, &b);
        assert_eq!((d.at, d.near_tie), (Some(2), Some(false)));
        assert_eq!(judge(vec![d], some_spec()).status, "fail");
    }

    #[test]
    fn only_the_first_divergence_is_judged() {
        // A near tie at step 1, then a difference at a wide margin at step 3 that is the divergence's consequence.
        let a = off(&[(5, 9.0), (6, 0.01), (7, 9.0), (8, 9.0)]);
        let b = on(&[(5, 9.0), (1006, 9.0), (7, 9.0), (80, 9.0)]);
        let d = compare("x", &a, &b);
        assert_eq!((d.at, d.near_tie), (Some(1), Some(true)));
        assert_eq!(judge(vec![d], some_spec()).status, "pass");
    }

    #[test]
    fn a_divergence_outside_a_near_tie_fails() {
        let (a, b) = parted_at(3.0);
        let d = compare("06-code-write-sql", &a, &b);
        assert_eq!(
            (d.at, d.near_tie, d.margin),
            (Some(2), Some(false), Some(3.0))
        );
        let ok = compare("a", &a, &a);
        let v = judge(vec![ok, d], some_spec());
        assert_eq!(v.status, "fail");
        assert_eq!((v.outside_near_tie, v.identical), (1, 1));
        assert!(v.prompts[1].line().contains("DIVERGES at step 2"));
    }

    #[test]
    fn an_earlier_eog_is_a_divergence_at_its_step() {
        // The on run stops at step 2 (its end-of-generation token is the last step); the off run goes on.
        let long = [(5, 9.0), (6, 9.0), (7, 9.0), (8, 9.0)];
        let mut short = on(&[(5, 9.0), (6, 9.0), (1007, 9.0)]);
        short.stop = Stop::EndOfGeneration;
        let d = compare("x", &off(&long), &short);
        assert_eq!((d.at, d.near_tie), (Some(2), Some(false)));
        // At a near tie of the off run's step, where the stop token is off's second-ranked, it is excused.
        let tie = [(5, 9.0), (6, 9.0), (7, 0.02), (8, 9.0)];
        let d = compare("x", &off(&tie), &short);
        assert_eq!((d.at, d.near_tie), (Some(2), Some(true)));
        // But not when the stop token is not one of off's top two.
        let mut other = on(&[(5, 9.0), (6, 9.0), (99, 9.0)]);
        other.stop = Stop::EndOfGeneration;
        let d = compare("x", &off(&tie), &other);
        assert_eq!((d.at, d.near_tie), (Some(2), Some(false)));
        // The off run stopping sooner is a divergence at its end-of-generation step the same way.
        let mut stops = off(&[(5, 9.0), (6, 9.0), (99, 9.0)]);
        stops.stop = Stop::EndOfGeneration;
        let d = compare("x", &stops, &on(&long));
        assert_eq!(
            (d.at, d.near_tie, d.margin),
            (Some(2), Some(false), Some(9.0))
        );
    }

    #[test]
    fn a_run_that_is_a_prefix_of_the_other_diverges_where_it_ends() {
        // Cannot happen with a stop step recorded, but a damaged record must not read as identical.
        let long = [(5, 9.0), (6, 9.0), (7, 9.0)];
        let short = [(5, 9.0), (6, 9.0)];
        let d = compare("x", &off(&long), &on(&short));
        assert_eq!((d.at, d.near_tie), (Some(2), Some(false)));
        assert!(!d.identical_tokens);
        // The off run has no step there to excuse it by: not a near tie, however it is read.
        let d = compare("x", &off(&short), &on(&long));
        assert_eq!((d.at, d.near_tie, d.margin), (Some(2), Some(false), None));
        assert_eq!(judge(vec![d], some_spec()).status, "fail");
    }

    #[test]
    fn a_margin_that_is_not_a_number_from_zero_up_is_not_a_near_tie() {
        for bad in [f32::NAN, -0.01, f32::NEG_INFINITY, f32::INFINITY] {
            let (a, b) = parted_at(bad);
            let d = compare("x", &a, &b);
            assert_eq!((d.at, d.near_tie), (Some(2), Some(false)), "margin {bad}");
        }
        // A margin of exactly 0 (a true tie) is a near tie.
        let (a, b) = parted_at(0.0);
        assert_eq!(compare("x", &a, &b).near_tie, Some(true));
    }

    #[test]
    fn identical_logits_is_none_without_fingerprints() {
        let steps = [(5, 1.0), (6, 1.0)];
        let d = compare("x", &off(&steps), &on(&steps));
        assert_eq!(d.identical_logits, None);
        // One side without them is none too, as is a divergence at step 0 (nothing before it).
        let d = compare("x", &record(&steps, &["a", "b"], false), &on(&steps));
        assert_eq!(d.identical_logits, None);
        let a = record(&[(5, 1.0), (6, 1.0)], &["a", "b"], false);
        let b = record(&[(50, 1.0), (6, 1.0)], &["a", "b"], true);
        let d = compare("x", &a, &b);
        assert_eq!((d.at, d.identical_logits), (Some(0), None));
    }

    #[test]
    fn identical_logits_compares_the_positions_before_the_divergence() {
        let fp = |n: u32| fingerprint(&[n as f32]);
        let p = [fp(1), fp(2), fp(3), fp(4)];
        let q = [fp(1), fp(2), fp(9), fp(8)];
        fn refs(v: &[String; 4]) -> Vec<&str> {
            v.iter().map(String::as_str).collect()
        }
        let steps = [(5, 9.0), (6, 9.0), (7, 0.0), (8, 9.0)];
        // Parting at step 2: positions 0 and 1 agree, 2 and 3 are not looked at.
        let a = record(&steps, &refs(&p), false);
        let b = record(&[(5, 9.0), (6, 9.0), (70, 0.0), (80, 9.0)], &refs(&q), true);
        assert_eq!(compare("x", &a, &b).identical_logits, Some(true));
        // Parting at step 3: position 2 differs and is before it.
        let b = record(&[(5, 9.0), (6, 9.0), (7, 0.0), (80, 9.0)], &refs(&q), true);
        let d = compare("x", &a, &b);
        assert_eq!((d.at, d.identical_logits), (Some(3), Some(false)));
        // Same tokens: all positions, and they differ here; and identical ones agree.
        let b = record(&steps, &refs(&q), true);
        let d = compare("x", &a, &b);
        assert!(d.identical_tokens);
        assert_eq!(d.identical_logits, Some(false));
        let b = record(&steps, &refs(&p), true);
        assert_eq!(compare("x", &a, &b).identical_logits, Some(true));
    }

    #[test]
    fn the_logits_never_decide_the_verdict() {
        // Equal tokens with different logits (expected in the DeltaNet layers) pass; they are only reported.
        let steps = [(5, 9.0), (6, 9.0)];
        let a = record(&steps, &["a", "b"], false);
        let b = record(&steps, &["a", "c"], true);
        let d = compare("x", &a, &b);
        assert_eq!(d.identical_logits, Some(false));
        assert_eq!(judge(vec![d], some_spec()).status, "pass");
    }

    #[test]
    fn a_comparison_that_is_neither_identical_nor_excused_counts_against() {
        // Hand-built, inconsistent divergences: a pass has to be proved, not defaulted to.
        let base = Divergence {
            id: "x".into(),
            at: None,
            near_tie: None,
            margin: None,
            identical_tokens: true,
            identical_logits: None,
            verify_steps: Some(1),
        };
        let cases = [
            Divergence {
                identical_tokens: false,
                ..base.clone()
            },
            Divergence {
                at: Some(3),
                identical_tokens: false,
                near_tie: None,
                ..base.clone()
            },
            Divergence {
                at: Some(3),
                identical_tokens: true,
                near_tie: Some(true),
                ..base.clone()
            },
            Divergence {
                at: Some(3),
                identical_tokens: false,
                near_tie: Some(false),
                ..base.clone()
            },
        ];
        for c in cases {
            let v = judge(vec![c.clone()], some_spec());
            assert_eq!(
                (v.status.as_str(), v.outside_near_tie),
                ("fail", 1),
                "{c:?}"
            );
            assert!(c.line().contains("DIVERGES"));
        }
        assert_eq!(judge(vec![base], some_spec()).status, "pass");
    }

    #[test]
    fn an_mtp_load_whose_on_runs_never_drafted_with_mtp_is_not_a_pass() {
        let steps = [(5, 1.0), (6, 2.0)];
        let d = compare("x", &off(&steps), &on(&steps));
        let lookup_only = SpecRecord {
            drafted_lookup: 9,
            ..some_spec()
        };
        let v = judge_expecting(vec![d.clone()], lookup_only.clone(), true);
        assert_eq!((v.status.as_str(), v.problems.len()), ("fail", 1));
        assert!(
            v.problems[0].contains("no on run drafted with MTP"),
            "{v:?}"
        );
        // Not expected (no MTP source, or a load without it): the same counts pass.
        assert_eq!(
            judge_expecting(vec![d.clone()], lookup_only, false).status,
            "pass"
        );
        // Expected and drafted: pass.
        let mtp = SpecRecord {
            drafted_mtp: 4,
            ..some_spec()
        };
        assert_eq!(judge_expecting(vec![d], mtp, true).status, "pass");
    }

    #[test]
    fn golden_expects_mtp_when_the_on_runs_name_it_and_loaded_it() {
        let out = scratch("golden-mtp");
        let steps = [(5, 9.0), (6, 9.0)];
        let run = |sources: &[&str], mtp_vram: u64, drafted_mtp: u64| {
            let (sources, p) = (sources.to_vec(), prompts());
            golden_with(&p[..1], 1, &out, move |mode, _, _| {
                Ok(if mode == Mode::Off {
                    off(&steps)
                } else {
                    let mut r = on(&steps);
                    r.settings.speculation.as_mut().unwrap().sources =
                        sources.iter().map(|s| s.to_string()).collect();
                    r.memory.mtp_vram_bytes = mtp_vram;
                    r.spec = Some(SpecRecord {
                        drafted_mtp,
                        ..some_spec()
                    });
                    r
                })
            })
            .unwrap()
        };
        let v = run(&["lookup", "mtp"], 1 << 20, 0);
        assert_eq!(v.status, "fail", "{v:?}");
        assert!(v.problems.iter().any(|p| p.contains("lookup alone")));
        assert_eq!(run(&["lookup", "mtp"], 1 << 20, 3).status, "pass");
        assert_eq!(run(&["lookup"], 0, 0).status, "pass");
        assert_eq!(run(&["lookup", "mtp"], 0, 0).status, "pass");
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn nothing_compared_is_not_a_pass() {
        let v = judge(Vec::new(), some_spec());
        assert_eq!(v.status, "fail");
        assert_eq!(v.outside_near_tie, 0);
        assert_eq!(v.problems.len(), 1);
        // On runs that never verified a step were plain decode: they match the off runs by construction.
        let steps = [(5, 1.0)];
        let d = compare("x", &off(&steps), &on(&steps));
        let none = SpecRecord {
            steps: 0,
            ..some_spec()
        };
        let v = judge(vec![d.clone()], none);
        assert_eq!(
            (v.status.as_str(), v.outside_near_tie, v.problems.len()),
            ("fail", 0, 1)
        );
        assert_eq!(judge(vec![d], some_spec()).status, "pass");
    }

    #[test]
    fn the_verdict_serialises_as_the_runner_reads_it() {
        let steps = [(5, 1.0)];
        let d = compare("01-code", &off(&steps), &on(&steps));
        let v = serde_json::to_value(judge(vec![d], some_spec())).unwrap();
        assert_eq!(v["status"], "pass");
        assert_eq!(
            (v["outside_near_tie"].as_u64(), v["identical"].as_u64()),
            (Some(0), Some(1))
        );
        let p = &v["prompts"][0];
        for k in [
            "id",
            "at",
            "near_tie",
            "margin",
            "identical_tokens",
            "identical_logits",
        ] {
            assert!(p.get(k).is_some(), "missing {k}");
        }
        for k in ["steps", "plain_steps", "drafted", "accepted", "replays"] {
            assert!(v["spec"][k].is_u64(), "missing {k}");
        }
        // No reason, no key: a clean verdict has the shape the spec-golden record is built from.
        assert!(v.get("problems").is_none());
        let bad = serde_json::to_value(judge(Vec::new(), some_spec())).unwrap();
        assert_eq!(bad["problems"].as_array().map(Vec::len), Some(1));
    }

    // ----- the driver over a fake engine -----------------------------------------------------------------------

    fn prompts() -> Vec<(String, Vec<u32>)> {
        vec![
            ("01-code".into(), vec![1, 2, 3]),
            ("02-prose".into(), vec![1, 2, 3]),
        ]
    }

    fn scratch(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("artemis-spec-check-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn golden_runs_off_then_on_per_prompt_and_writes_the_records_and_the_verdict() {
        let out = scratch("golden");
        let mut calls = Vec::new();
        let v = golden_with(&prompts(), 2, &out, |mode, id, tokens| {
            calls.push((id.to_string(), mode));
            assert_eq!(tokens, [1, 2, 3]);
            let steps = [(5, 9.0), (6, 9.0)];
            Ok(match mode {
                Mode::Off => off(&steps),
                Mode::On | Mode::Lookup => on(&steps),
            })
        })
        .unwrap();
        assert_eq!(
            calls,
            [
                ("01-code".to_string(), Mode::Off),
                ("01-code".to_string(), Mode::On),
                ("02-prose".to_string(), Mode::Off),
                ("02-prose".to_string(), Mode::On),
            ]
        );
        assert_eq!(
            (v.status.as_str(), v.identical, v.prompts.len()),
            ("pass", 2, 2)
        );
        // The counts are the on runs' sums.
        assert_eq!((v.spec.steps, v.spec.drafted, v.spec.accepted), (4, 12, 8));
        for f in [
            "01-code.off.json",
            "01-code.on.json",
            "02-prose.off.json",
            "02-prose.on.json",
            "verdict.json",
        ] {
            assert!(out.join(f).exists(), "{f}");
        }
        let written: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("verdict.json")).unwrap()).unwrap();
        assert_eq!(written["status"], "pass");
        // The records read back as generate's.
        let r = generate::read_record(&out.join("01-code.on.json")).unwrap();
        assert!(r.spec.is_some());
        assert!(generate::read_record(&out.join("01-code.off.json"))
            .unwrap()
            .spec
            .is_none());
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn a_failing_verdict_is_still_written() {
        let out = scratch("failing");
        let v = golden_with(&prompts()[..1], 1, &out, |mode, _, _| {
            let (a, b) = parted_at(3.0);
            Ok(if mode == Mode::Off { a } else { b })
        })
        .unwrap();
        assert_eq!((v.status.as_str(), v.outside_near_tie), ("fail", 1));
        let written: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("verdict.json")).unwrap()).unwrap();
        assert_eq!(written["status"], "fail");
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn a_pair_that_cannot_be_compared_is_an_error_not_a_verdict() {
        let steps = [(5, 9.0), (6, 9.0)];
        let try_pair = |name: &str, f: &dyn Fn(Mode) -> Record| {
            let out = scratch(name);
            let r = golden_with(&prompts()[..1], 1, &out, |mode, _, _| Ok(f(mode)));
            let wrote = out.join("verdict.json").exists();
            let _ = fs::remove_dir_all(&out);
            assert!(!wrote, "{name}: a verdict was written");
            r.expect_err(name).to_string()
        };
        let pick = |m: Mode, steps: &[(u32, f32)]| match m {
            Mode::Off => off(steps),
            Mode::On | Mode::Lookup => on(steps),
        };
        // The "on" run that did not speculate (a run of off twice would otherwise match itself).
        let e = try_pair("off twice", &|_| off(&steps));
        assert!(e.contains("on run's record has speculation off"), "{e}");
        let e = try_pair("on twice", &|_| on(&steps));
        assert!(e.contains("off run's record has speculation on"), "{e}");
        // A run with no step compares equal to another with none.
        let e = try_pair("empty", &|m| pick(m, &[]));
        assert!(e.contains("made no step"), "{e}");
        // The wrong prompt, and a run that is not the golden run.
        let e = try_pair("prompt", &|m| {
            let mut r = pick(m, &steps);
            r.prompt_tokens = vec![9];
            r
        });
        assert!(e.contains("prompt is not the golden prompt"), "{e}");
        let e = try_pair("sampled", &|m| {
            let mut r = pick(m, &steps);
            r.settings.greedy = false;
            r
        });
        assert!(e.contains("was not greedy for 256 tokens"), "{e}");
        let e = try_pair("short", &|m| {
            let mut r = pick(m, &steps);
            r.settings.max_tokens = 32;
            r
        });
        assert!(e.contains("was not greedy for 256 tokens"), "{e}");
    }

    #[test]
    fn a_run_that_fails_ends_the_command() {
        let out = scratch("run-fails");
        let e = golden_with(&prompts(), 2, &out, |mode, _, _| {
            if mode == Mode::On {
                bail!("device lost")
            }
            Ok(off(&[(5, 9.0)]))
        })
        .unwrap_err();
        assert!(e.to_string().contains("device lost"));
        assert!(!out.join("verdict.json").exists());
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn select_takes_all_or_the_asked_ids_in_the_sets_order_and_refuses_an_unknown_one() {
        let all: Vec<String> = ["01-code", "02-prose", "03-chat"]
            .map(String::from)
            .to_vec();
        assert_eq!(select(&all, &[]).unwrap(), all);
        let asked = ["03-chat", "01-code"].map(String::from).to_vec();
        assert_eq!(select(&all, &asked).unwrap(), ["01-code", "03-chat"]);
        let e = select(&all, &["04-nope".to_string()])
            .unwrap_err()
            .to_string();
        assert!(e.contains("\"04-nope\" is not a golden prompt"), "{e}");
    }

    #[test]
    fn the_completion_hash_is_the_logit_fingerprints_hash_of_the_ids_bytes() {
        // `fingerprint` hashes f32s' little-endian bytes: the same bytes as ids' when the f32s carry the ids' bits.
        let ids = [0u32, 1, 248_319, 0xDEAD_BEEF];
        let floats: Vec<f32> = ids.iter().map(|&i| f32::from_bits(i)).collect();
        assert_eq!(completion_fnv(&ids), fingerprint(&floats));
        assert_eq!(completion_fnv(&ids).len(), 16);
        assert_ne!(completion_fnv(&[1, 2]), completion_fnv(&[2, 1]));
        // The empty completion is FNV's offset basis.
        assert_eq!(completion_fnv(&[]), "cbf29ce484222325");
    }

    /// A speed record of `tokens`, decoded in `secs`, on or off.
    fn timed(tokens: &[u32], secs: f64, speculated: bool) -> Record {
        let steps: Vec<(u32, f32)> = tokens.iter().map(|&t| (t, 1.0)).collect();
        let mut r = record(&steps, &[], speculated);
        r.settings.greedy = false;
        r.settings.max_tokens = 700;
        r.timings = Timings {
            decoded: tokens.len() - 1,
            generate_s: secs,
            ..Timings::default()
        };
        r.cache.decode_misses = 7;
        r
    }

    #[test]
    fn speed_runs_each_prompt_and_seed_off_then_on_and_reports_the_ratio() {
        let out = scratch("speed");
        let prompts = vec![
            ("write".to_string(), vec![1, 2, 3]),
            ("edit".to_string(), vec![1, 2, 3]),
        ];
        let mut calls = Vec::new();
        let r = speed_with(&prompts, &[1234, 1235], &out, |mode, name, seed, _| {
            calls.push(format!("{name}-{seed}-{}", mode.name()));
            // 100 decoded tokens in 1 s off, in 0.8 s on: 100 and 125 tok/s.
            let t: Vec<u32> = (0..101).collect();
            Ok(match mode {
                Mode::Off => timed(&t, 1.0, false),
                Mode::On | Mode::Lookup => timed(&t, 0.8, true),
            })
        })
        .unwrap();
        assert_eq!(
            calls,
            [
                // The untimed warm-up first, then each pair off then on.
                "write-1234-off",
                "write-1234-off",
                "write-1234-on",
                "write-1235-off",
                "write-1235-on",
                "edit-1234-off",
                "edit-1234-on",
                "edit-1235-off",
                "edit-1235-on"
            ]
        );
        assert_eq!((r.runs.len(), r.pairs.len(), r.status), (8, 4, "report"));
        assert_eq!((r.runs[0].mode, r.runs[1].mode), ("off", "on"));
        assert!((r.runs[0].decode_tok_s - 100.0).abs() < 1e-9);
        assert!((r.runs[1].decode_tok_s - 125.0).abs() < 1e-9);
        assert!((r.pairs[0].ratio - 1.25).abs() < 1e-9);
        assert!(r.pairs[0].identical_completion);
        assert_eq!((r.off_median, r.on_median), (100.0, 125.0));
        assert!((r.ratio - 1.25).abs() < 1e-9);
        assert!(r.runs[0].spec.is_none() && r.runs[1].spec.is_some());
        // 101 steps, none of them a stop: 101 completion tokens, hashed as ids.
        assert_eq!(r.runs[0].completion_tokens, 101);
        assert_eq!(
            r.runs[0].completion_fnv,
            completion_fnv(&(0..101).collect::<Vec<u32>>())
        );
        for f in [
            "write-1234.off.json",
            "write-1234.on.json",
            "edit-1235.on.json",
            "speed.json",
        ] {
            assert!(out.join(f).exists(), "{f}");
        }
        let _ = fs::remove_dir_all(&out);
    }

    /// A speculating record of `tokens` for mode `m`, as the engine writes it: `sources` and `drafted_mtp` say what ran.
    fn spec_run(tokens: &[u32], secs: f64, sources: &[&str], drafted_mtp: u64) -> Record {
        let mut r = timed(tokens, secs, true);
        r.settings.speculation.as_mut().unwrap().sources =
            sources.iter().map(|s| s.to_string()).collect();
        r.spec = Some(SpecRecord {
            steps: 3,
            drafted: 10,
            accepted: 6,
            drafted_lookup: 10 - drafted_mtp,
            drafted_mtp,
            accepted_mtp: drafted_mtp / 2,
            ..SpecRecord::default()
        });
        r
    }

    #[test]
    fn the_modes_run_off_and_on_pair_by_pair_then_lookup_after_them_and_report_mtps_gain() {
        let out = scratch("speed-modes");
        let prompts = vec![
            ("write".to_string(), vec![1, 2, 3]),
            ("edit".to_string(), vec![1, 2, 3]),
        ];
        let mut calls = Vec::new();
        let t: Vec<u32> = (0..101).collect();
        let r = speed_modes(
            &prompts,
            &[1234],
            &out,
            // The order asked for doesn't matter: the load a mode needs decides when it runs.
            &[Mode::Lookup, Mode::On, Mode::Off],
            true,
            true,
            |mode, name, seed, _| {
                calls.push(format!("{name}-{seed}-{}", mode.name()));
                // 100 decoded tokens in 1 s off, 0.8 s on (125 tok/s), 0.9 s lookup (111.1 tok/s).
                let mut r = match mode {
                    Mode::Off => timed(&t, 1.0, false),
                    Mode::On => spec_run(&t, 0.8, &["lookup", "mtp"], 4),
                    Mode::Lookup => spec_run(&t, 0.9, &["lookup"], 0),
                };
                // MTP's memory costs the full load a slot a layer.
                let slots = if mode == Mode::On { 134 } else { 135 };
                r.settings.cache_slots = slots;
                r.cache.slots_per_layer = slots;
                Ok(r)
            },
        )
        .unwrap();
        assert_eq!(
            calls,
            [
                // The full load first: "on" alone, after its untimed warm-up...
                "write-1234-on",
                "write-1234-on",
                "edit-1234-on",
                // ...then the load without MTP, as the gate's "off" and "lookup" processes have it: its own warm-up,
                // then each prompt's off and lookup, off first.
                "write-1234-off",
                "write-1234-off",
                "write-1234-lookup",
                "edit-1234-off",
                "edit-1234-lookup",
            ]
        );
        assert_eq!(r.modes, ["lookup", "on", "off"]);
        assert!(r.fresh_state);
        assert_eq!((r.runs.len(), r.pairs.len()), (6, 2));
        assert_eq!(r.runs[3].mode, "lookup");
        // The loads differ in slots, which the comparison allows and the report shows.
        assert_eq!(
            r.runs.iter().map(|x| x.slots_per_layer).collect::<Vec<_>>(),
            [134, 134, 135, 135, 135, 135]
        );
        assert_eq!(r.runs[0].mode, "on");
        assert!((r.pairs[0].lookup_tok_s.unwrap() - 100.0 / 0.9).abs() < 1e-9);
        assert!((r.pairs[0].mtp_gain.unwrap() - 1.125).abs() < 1e-9);
        assert!((r.lookup_median.unwrap() - 100.0 / 0.9).abs() < 1e-9);
        assert!((r.mtp_gain.unwrap() - 125.0 / (100.0 / 0.9)).abs() < 1e-9);
        for f in [
            "write-1234.off.json",
            "write-1234.on.json",
            "write-1234.lookup.json",
            "edit-1234.lookup.json",
            "speed.json",
        ] {
            assert!(out.join(f).exists(), "{f}");
        }
        // The default modes are `Off` and `On`: no lookup figure.
        let none = speed_with(&prompts, &[1234], &out, |mode, _, _, _| {
            Ok(match mode {
                Mode::Off => timed(&t, 1.0, false),
                _ => timed(&t, 0.8, true),
            })
        })
        .unwrap();
        assert_eq!(none.modes, ["off", "on"]);
        assert!(!none.fresh_state);
        assert!(none.lookup_median.is_none() && none.mtp_gain.is_none());
        assert!(none.pairs.iter().all(|p| p.lookup_tok_s.is_none()));
        // Each rerun clears the lookup records too, so none of an earlier, wider run survives.
        speed_modes(
            &prompts,
            &[1234],
            &out,
            &[Mode::Off, Mode::On],
            false,
            false,
            |mode, _, _, _| {
                Ok(match mode {
                    Mode::Off => timed(&t, 1.0, false),
                    _ => timed(&t, 0.8, true),
                })
            },
        )
        .unwrap();
        assert!(!out.join("write-1234.lookup.json").exists());
        // On alone has no pair; no mode, or one twice, is refused before any run.
        let only = speed_modes(
            &prompts,
            &[1234],
            &out,
            &[Mode::On],
            false,
            false,
            |_, _, _, _| Ok(timed(&t, 0.8, true)),
        )
        .unwrap();
        assert!(only.pairs.is_empty() && only.off_median == 0.0);
        let e = speed_modes(
            &prompts,
            &[1234],
            &out,
            &[],
            false,
            false,
            |_, _, _, _| unreachable!(),
        )
        .unwrap_err();
        assert!(e.to_string().contains("names no mode"), "{e}");
        let e = speed_modes(
            &prompts,
            &[1234],
            &out,
            &[Mode::On, Mode::On],
            false,
            false,
            |_, _, _, _| unreachable!(),
        )
        .unwrap_err();
        assert!(e.to_string().contains("on twice"), "{e}");
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn a_lookup_run_that_used_mtp_is_a_mode_problem() {
        let prompts = vec![("write".to_string(), vec![1, 2, 3])];
        let t: Vec<u32> = (0..11).collect();
        let lookup_with = |sources: &'static [&'static str], mtp: u64, vram: u64| {
            let out = scratch("speed-lookup-bad");
            let e = speed_modes(
                &prompts,
                &[1],
                &out,
                &[Mode::Lookup],
                false,
                false,
                |_, _, _, _| {
                    let mut r = spec_run(&t, 0.8, sources, mtp);
                    r.memory.mtp_vram_bytes = vram;
                    Ok(r)
                },
            );
            let _ = fs::remove_dir_all(&out);
            e
        };
        // A run that drafted with MTP, or whose record names MTP as a source, timed the wrong thing.
        let e = lookup_with(&["lookup"], 3, 0).unwrap_err();
        assert!(e.to_string().contains("the lookup run used MTP"), "{e}");
        let e = lookup_with(&["lookup", "mtp"], 0, 0).unwrap_err();
        assert!(e.to_string().contains("the lookup run used MTP"), "{e}");
        assert!(lookup_with(&["lookup"], 0, 0).is_ok());
    }

    #[test]
    fn the_lookup_mode_loads_without_mtp() {
        let cfg = SpecConfig::default();
        assert!(cfg.has(SpecSource::Mtp));
        // The modes' loads: "on" has the full config; "off" and "lookup" drop MTP from the sources, keeping the rest
        // (the gate's off process has no MTP memory).
        assert_eq!(Mode::On.load_config(&cfg), cfg);
        assert_eq!(Mode::Off.load_config(&cfg), Mode::Lookup.load_config(&cfg));
        let lookup = Mode::Lookup.load_config(&cfg);
        assert_eq!(lookup.sources, [SpecSource::Lookup]);
        assert_eq!(
            lookup,
            SpecConfig {
                sources: vec![SpecSource::Lookup],
                ..cfg.clone()
            }
        );
        // The engine options follow the config. The full load asks for the MTP (its VRAM, its ring copies); the
        // lookup load leaves `Options::mtp` unset, so it has neither.
        let full = rig_options(&Mode::On.load_config(&cfg), CTX, None).unwrap();
        let none = rig_options(&lookup, CTX, None).unwrap();
        let mtp = full.mtp.expect("the full load has the MTP");
        assert_eq!((mtp.rows, mtp.ring, mtp.window), (8192, 256, 2048));
        assert!(none.mtp.is_none());
        // Both speculate (the verify pass needs it), at production's context and the adaptive cache.
        assert!(full.speculation && none.speculation);
        assert_eq!((full.ctx, none.ctx), (CTX, CTX));
        // A session too short for the MTP is refused for the full load, and fine for lookup alone.
        assert!(rig_options(&cfg, 256, None).is_err());
        assert!(rig_options(&lookup, 256, None).is_ok());
        // The speed command's rigs: "on" uses the full load, "off" and "lookup" the one without MTP (the rig for a
        // lookup run checks that it has none; see `Rig::run`).
        assert!(Mode::Lookup.speculates() && Mode::On.speculates() && !Mode::Off.speculates());
    }

    /// A verifier that decodes a constant row: one plain step a call.
    struct Plain1;

    impl engine::spec::Verifier for Plain1 {
        type Error = std::fmt::Error;
        fn decode(&mut self, _: u32) -> Result<Vec<f32>, std::fmt::Error> {
            Ok(vec![0.0, 1.0])
        }
        fn verify(&mut self, _: &[u32]) -> Result<(), std::fmt::Error> {
            unreachable!("a fixed draft of 0 never verifies")
        }
        fn row(&mut self, _: usize) -> Result<Vec<f32>, std::fmt::Error> {
            unreachable!()
        }
        fn commit(&mut self, _: usize) -> Result<(), std::fmt::Error> {
            unreachable!()
        }
        fn n_vocab(&self) -> usize {
            2
        }
        fn misses(&self) -> u64 {
            0
        }
        fn miss_bytes(&self) -> u64 {
            0
        }
    }

    /// Runs one plain step on `state` and returns the engine-wide plain steps it has counted so far.
    fn one_step(state: &mut SpecState) -> u64 {
        let mut drafter = state.drafter();
        let mut delta = engine::spec::SpecDelta::default();
        state
            .step(
                &mut Plain1,
                drafter.as_mut(),
                0,
                0,
                &mut |_| Ok(engine::spec::Next::Go(1)),
                &mut delta,
            )
            .unwrap();
        state.totals().delta.plain_steps
    }

    #[test]
    fn fresh_state_builds_a_spec_state_per_run() {
        let cfg = SpecConfig {
            fixed_draft: Some(0),
            max_draft: 4,
            ..SpecConfig::default()
        };
        // One state for the whole process (no --fresh-state): its totals and its controller carry from run to run.
        let mut state = SpecState::new(cfg.clone());
        for run in 1..=3 {
            assert!(!renew_state(&mut state, false));
            assert_eq!(one_step(&mut state), run, "run {run}");
        }
        // --fresh-state: each run starts from a new controller and zero totals, with the same config.
        let mut state = SpecState::new(cfg.clone());
        for run in 1..=3 {
            assert!(renew_state(&mut state, true));
            assert_eq!(state.totals().delta.plain_steps, 0, "run {run}");
            assert_eq!(state.config(), &cfg);
            assert_eq!(one_step(&mut state), 1, "run {run}");
        }
    }

    #[test]
    fn the_speed_command_line_takes_modes_and_fresh_state_and_golden_takes_the_spec_flags() {
        let base = [
            "speed",
            "--model",
            "m",
            "--out",
            "o",
            "--edit-tokens",
            "e.tokens",
        ];
        match parse(&base).unwrap() {
            Cmd::Speed {
                modes, fresh_state, ..
            } => {
                assert_eq!(modes, [Mode::Off, Mode::On]);
                assert!(!fresh_state);
            }
            _ => panic!("not speed"),
        }
        let mut v = base.to_vec();
        v.extend([
            "--modes",
            "off,lookup,on",
            "--fresh-state",
            "--sources",
            "lookup,mtp",
            "--mtp-window",
            "1024",
        ]);
        match parse(&v).unwrap() {
            Cmd::Speed {
                modes,
                fresh_state,
                spec,
                ..
            } => {
                assert_eq!(modes, [Mode::Off, Mode::Lookup, Mode::On]);
                assert!(fresh_state);
                let c = spec.config(0.0, false).unwrap().unwrap();
                assert_eq!(c.mtp_window, 1024);
            }
            _ => panic!("not speed"),
        }
        let mut v = base.to_vec();
        v.extend(["--modes", "mtp"]);
        assert!(parse(&v).is_err());
        // `golden` takes the same flags, and refuses speculation off before any load.
        match parse(&[
            "golden",
            "--model",
            "m",
            "--out",
            "o",
            "--sources",
            "lookup",
        ])
        .unwrap()
        {
            Cmd::Golden { spec, .. } => {
                let c = spec.config(0.0, false).unwrap().unwrap();
                assert_eq!(c.sources, [SpecSource::Lookup]);
            }
            _ => panic!("not golden"),
        }
    }

    #[test]
    fn a_stop_token_is_not_part_of_the_completion_and_differing_completions_are_reported() {
        let out = scratch("speed-stop");
        let prompts = vec![("write".to_string(), vec![1, 2, 3])];
        let r = speed_with(&prompts, &[1234], &out, |mode, _, _, _| {
            Ok(match mode {
                Mode::Off => {
                    let mut r = timed(&[5, 6, 7, 99], 1.0, false);
                    r.stop = Stop::EndOfGeneration;
                    r
                }
                Mode::On | Mode::Lookup => timed(&[5, 6, 8, 99], 1.0, true),
            })
        })
        .unwrap();
        assert_eq!(r.runs[0].completion_tokens, 3);
        assert_eq!(r.runs[0].completion_fnv, completion_fnv(&[5, 6, 7]));
        assert_eq!(r.runs[1].completion_tokens, 4);
        assert!(!r.pairs[0].identical_completion);
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn a_speed_run_that_decoded_nothing_or_is_the_wrong_mode_is_an_error() {
        let prompts = vec![("write".to_string(), vec![1, 2, 3])];
        let out = scratch("speed-bad");
        let e = speed_with(&prompts, &[1], &out, |mode, _, _, _| {
            Ok(match mode {
                Mode::Off => timed(&[5, 6], 1.0, false),
                Mode::On | Mode::Lookup => timed(&[5], 1.0, true),
            })
        })
        .unwrap_err();
        assert!(e.to_string().contains("decoded no token"), "{e}");
        let e = speed_with(&prompts, &[1], &out, |_, _, _, _| {
            Ok(timed(&[5, 6], 1.0, false))
        })
        .unwrap_err();
        assert!(
            e.to_string()
                .contains("the on run's record has speculation off"),
            "{e}"
        );
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn the_speed_prompts_must_match_made_from() {
        let dir = scratch("prompts");
        fs::create_dir_all(dir.join("prompts")).unwrap();
        let bytes = |ids: &[u32]| {
            ids.iter()
                .flat_map(|t| t.to_le_bytes())
                .collect::<Vec<u8>>()
        };
        let (w, x, ed) = ([1u32, 2, 3], [4u32, 5], [6u32, 7, 8, 9]);
        fs::write(dir.join("prompts").join("write.tokens"), bytes(&w)).unwrap();
        fs::write(dir.join("prompts").join("explain.tokens"), bytes(&x)).unwrap();
        fs::write(dir.join("edit.tokens"), bytes(&ed)).unwrap();
        let made = serde_json::json!({
            "write": MadeFrom::of(&w), "explain": MadeFrom::of(&x), "edit": MadeFrom::of(&ed)
        });
        fs::write(
            dir.join("made-from.json"),
            serde_json::to_vec(&made).unwrap(),
        )
        .unwrap();
        let got = speed_prompts(&dir, &dir.join("edit.tokens")).unwrap();
        assert_eq!(
            got.iter()
                .map(|(n, t)| (n.as_str(), t.len()))
                .collect::<Vec<_>>(),
            [("write", 3), ("edit", 4), ("explain", 2)]
        );
        // A stale edit prompt is refused by name.
        fs::write(dir.join("edit.tokens"), bytes(&[6, 7, 8, 10])).unwrap();
        let e = speed_prompts(&dir, &dir.join("edit.tokens"))
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("not the edit prompt made-from.json records"),
            "{e}"
        );
        // A file that is not whole ids.
        fs::write(dir.join("edit.tokens"), [1u8, 2, 3]).unwrap();
        assert!(speed_prompts(&dir, &dir.join("edit.tokens")).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_committed_speed_prompts_match_made_from() {
        // All three prompts are committed (`edit` with its PSF-licensed source, `speed/sources/`).
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/speed");
        let made: std::collections::BTreeMap<String, MadeFrom> =
            serde_json::from_slice(&fs::read(dir.join("made-from.json")).unwrap()).unwrap();
        for name in ["write", "edit", "explain"] {
            let ids = read_tokens(&dir.join("prompts").join(format!("{name}.tokens"))).unwrap();
            assert_eq!(MadeFrom::of(&ids), made[name], "{name}");
        }
    }

    #[test]
    fn a_near_tie_needs_the_on_token_to_be_one_of_the_off_runs_top_two() {
        let off_steps = [(5, 9.0), (6, 9.0), (7, 0.01), (8, 9.0)];
        // `step` makes off's second-ranked token at index 2 equal to 1007: taking it is the tie.
        let second = on(&[(5, 9.0), (6, 9.0), (1007, 0.0)]);
        let d = compare("x", &off(&off_steps), &second);
        assert_eq!((d.at, d.near_tie), (Some(2), Some(true)));
        assert_eq!(judge(vec![d], some_spec()).status, "pass");
        // A third-ranked token (or garbage) at the same near tie is a bug, not a tie.
        let third = on(&[(5, 9.0), (6, 9.0), (4321, 0.0)]);
        let d = compare("x", &off(&off_steps), &third);
        assert_eq!(
            (d.at, d.near_tie, d.margin),
            (Some(2), Some(false), Some(0.01))
        );
        let v = judge(vec![d.clone()], some_spec());
        assert_eq!((v.status.as_str(), v.outside_near_tie), ("fail", 1));
        assert!(d.line().contains("DIVERGES at step 2"));
        // The on run having no step there (it ended first, with every step before it equal) is outside too, at a
        // margin of 0.
        let ended = on(&[(5, 9.0), (6, 9.0)]);
        let d = compare("x", &off(&[(5, 9.0), (6, 9.0), (7, 0.0)]), &ended);
        assert_eq!((d.at, d.near_tie), (Some(2), Some(false)));
    }

    #[test]
    fn the_verdict_records_each_prompts_verify_steps() {
        let steps = [(5, 1.0)];
        let mut quiet = on(&steps);
        quiet.spec = Some(SpecRecord::default());
        let d = compare("a", &off(&steps), &quiet);
        assert_eq!(d.verify_steps, Some(0));
        assert!(d.line().contains("no verify step"), "{}", d.line());
        let d = compare("b", &off(&steps), &on(&steps));
        assert_eq!(d.verify_steps, Some(2));
        assert!(!d.line().contains("no verify step"));
        // The aggregate guard stays: one verifying prompt clears the others, and the key is in the verdict.
        let v = judge(vec![compare("a", &off(&steps), &quiet), d], some_spec());
        assert_eq!(v.status, "pass");
        let j = serde_json::to_value(&v).unwrap();
        assert_eq!(j["prompts"][0]["verify_steps"], 0);
        assert_eq!(j["prompts"][1]["verify_steps"], 2);
    }

    #[test]
    fn an_earlier_runs_verdict_and_records_do_not_survive_a_failed_run() {
        let out = scratch("stale");
        fs::create_dir_all(&out).unwrap();
        fs::write(out.join("verdict.json"), br#"{"status":"pass"}"#).unwrap();
        fs::write(out.join("19-old.off.json"), b"{}").unwrap();
        fs::write(out.join("19-old.on.json"), b"{}").unwrap();
        fs::write(out.join("speed.json"), b"{}").unwrap();
        fs::write(out.join("notes.txt"), b"keep").unwrap();
        let e = golden_with(&prompts(), 2, &out, |mode, _, _| {
            if mode == Mode::On {
                bail!("device lost")
            }
            Ok(off(&[(5, 9.0)]))
        })
        .unwrap_err();
        assert!(e.to_string().contains("device lost"));
        for f in ["verdict.json", "19-old.off.json", "19-old.on.json"] {
            assert!(!out.join(f).exists(), "{f} survived");
        }
        // Golden does not clear speed's report, nor unrelated files.
        assert!(out.join("speed.json").exists());
        assert!(
            out.join("notes.txt").exists(),
            "an unrelated file was removed"
        );
        // The speed report is cleared the same way.
        let e = speed_with(
            &[("write".to_string(), vec![1, 2, 3])],
            &[1],
            &out,
            |_, _, _, _| bail!("device lost"),
        )
        .unwrap_err();
        assert!(e.to_string().contains("device lost"));
        assert!(!out.join("speed.json").exists());
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn a_subset_run_is_never_a_pass() {
        let out = scratch("subset");
        let ok = |mode: Mode, _: &str, _: &[u32]| -> Result<Record> {
            let steps = [(5, 9.0), (6, 9.0)];
            Ok(if mode == Mode::Off {
                off(&steps)
            } else {
                on(&steps)
            })
        };
        let v = golden_with(&prompts()[..1], 2, &out, ok).unwrap();
        assert_eq!(v.status, "subset");
        assert_eq!(v.outside_near_tie, 0);
        assert!(v
            .problems
            .iter()
            .any(|p| p.contains("only 1 of 2 golden prompts")));
        let written: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("verdict.json")).unwrap()).unwrap();
        assert_eq!(written["status"], "subset");
        // The whole set passes; a subset that fails still says fail.
        assert_eq!(golden_with(&prompts(), 2, &out, ok).unwrap().status, "pass");
        let v = golden_with(&prompts()[..1], 2, &out, |m, _, _| {
            let (a, b) = parted_at(3.0);
            Ok(if m == Mode::Off { a } else { b })
        })
        .unwrap();
        assert_eq!(v.status, "fail");
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn the_off_and_on_runs_must_differ_in_speculation_only() {
        let steps = [(5, 9.0), (6, 9.0)];
        let out = scratch("settings");
        let e = golden_with(&prompts()[..1], 1, &out, |mode, _, _| {
            let mut r = if mode == Mode::Off {
                off(&steps)
            } else {
                on(&steps)
            };
            if mode == Mode::On {
                r.settings.ctx = 4096;
            }
            Ok(r)
        })
        .unwrap_err()
        .to_string();
        assert!(e.contains("settings differ beyond speculation"), "{e}");
        let e = speed_with(
            &[("write".to_string(), vec![1, 2, 3])],
            &[1],
            &out,
            |mode, _, _, _| {
                let mut r = timed(&[5, 6], 1.0, mode == Mode::On);
                r.settings.seed = u64::from(mode == Mode::On);
                Ok(r)
            },
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("settings differ beyond speculation"), "{e}");
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn the_speed_warm_up_is_not_reported_or_written() {
        let out = scratch("warm-up");
        let mut n = 0;
        let r = speed_with(
            &[("write".to_string(), vec![1, 2, 3])],
            &[7],
            &out,
            |mode, _, _, _| {
                n += 1;
                // The warm-up is slow, as a cold cache is; only the pair is reported.
                let secs = if n == 1 { 50.0 } else { 1.0 };
                Ok(timed(&[5, 6, 7], secs, mode == Mode::On))
            },
        )
        .unwrap();
        assert_eq!((n, r.runs.len(), r.pairs.len()), (3, 2, 1));
        assert!(r.runs.iter().all(|x| x.generate_s == 1.0));
        assert_eq!(r.runs[0].decode_misses, 7);
        let _ = fs::remove_dir_all(&out);
    }

    /// The golden set in a temp dir: `expected/<id>.json` for each id, tokens for the first `with_tokens`.
    fn mini_golden(name: &str, ids: &[&str], with_tokens: usize) -> PathBuf {
        let dir = scratch(name);
        fs::create_dir_all(dir.join("expected")).unwrap();
        fs::create_dir_all(dir.join("prompts")).unwrap();
        let mut made = serde_json::Map::new();
        for (i, id) in ids.iter().enumerate() {
            fs::write(dir.join("expected").join(format!("{id}.json")), b"{}").unwrap();
            let t = [i as u32 + 1, 2, 3];
            if i < with_tokens {
                let b: Vec<u8> = t.iter().flat_map(|x| x.to_le_bytes()).collect();
                fs::write(dir.join("prompts").join(format!("{id}.tokens")), b).unwrap();
            }
            made.insert(
                id.to_string(),
                serde_json::json!({"expected": {"prompt": completion_fnv(&t)}}),
            );
        }
        fs::write(
            dir.join("made-from.json"),
            serde_json::to_vec(&made).unwrap(),
        )
        .unwrap();
        dir
    }

    #[test]
    fn the_golden_set_is_the_expected_ids_and_each_needs_its_tokens() {
        let dir = mini_golden("set", &["01-a", "02-b"], 2);
        let set = golden_set(&dir).unwrap();
        assert_eq!(
            set.iter()
                .map(|(i, t)| (i.as_str(), t[0]))
                .collect::<Vec<_>>(),
            [("01-a", 1), ("02-b", 2)]
        );
        // A tokens file that was never made is refused, naming the prompt, whatever --prompts asks for.
        let dir2 = mini_golden("set-missing", &["01-a", "02-b"], 1);
        let e = format!("{:#}", golden_set(&dir2).unwrap_err());
        assert!(
            e.contains("02-b") && e.contains("no readable tokens file"),
            "{e}"
        );
        // A tokens file other than the one the oracle ran is refused.
        let b: Vec<u8> = [9u32, 2, 3].iter().flat_map(|x| x.to_le_bytes()).collect();
        fs::write(dir.join("prompts").join("01-a.tokens"), b).unwrap();
        let e = golden_set(&dir).unwrap_err().to_string();
        assert!(
            e.contains("not the prompt expected/01-a.json was made from"),
            "{e}"
        );
        // A tokens file with no expected file is not part of the set.
        let dir3 = mini_golden("set-extra", &["01-a"], 1);
        fs::write(dir3.join("prompts").join("99-extra.tokens"), [0u8; 4]).unwrap();
        assert_eq!(golden_set(&dir3).unwrap().len(), 1);
        for d in [dir, dir2, dir3] {
            let _ = fs::remove_dir_all(d);
        }
    }

    #[test]
    fn the_committed_golden_set_is_twenty_prompts_matching_made_from() {
        let set = golden_set(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden"))
            .unwrap();
        assert_eq!(set.len(), 20);
        assert_eq!(set[0].0, "01-code");
    }

    #[test]
    fn a_failing_start_clears_an_earlier_verdict_before_the_load() {
        let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden");
        let out = scratch("cleared");
        fs::create_dir_all(&out).unwrap();
        fs::write(out.join("verdict.json"), br#"{"status":"pass"}"#).unwrap();
        let e = run(Cmd::Golden {
            model: PathBuf::from("no-such-model.gguf"),
            golden,
            prompts: vec!["01-code".to_string()],
            out: out.clone(),
            device: None,
            allow_small_golden: false,
            spec: SpecFlags::default(),
        })
        .unwrap_err()
        .to_string();
        assert!(e.contains("no-such-model.gguf"), "{e}");
        assert!(!out.join("verdict.json").exists(), "a stale pass survived");
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn each_command_clears_only_its_own_files_from_a_shared_out() {
        let seed = |out: &Path| {
            fs::create_dir_all(out).unwrap();
            for f in [
                "verdict.json",
                "01-code.off.json",
                "01-code.on.json",
                "speed.json",
                "write-1234.off.json",
                "edit-1235.on.json",
                "notes.txt",
            ] {
                fs::write(out.join(f), b"{}").unwrap();
            }
        };
        let left = |out: &Path| {
            let mut v: Vec<String> = fs::read_dir(out)
                .unwrap()
                .map(|e| e.unwrap().file_name().into_string().unwrap())
                .collect();
            v.sort();
            v
        };
        let out = scratch("shared");
        seed(&out);
        clear_out(&out, Owner::Speed).unwrap();
        assert_eq!(
            left(&out),
            [
                "01-code.off.json",
                "01-code.on.json",
                "notes.txt",
                "verdict.json"
            ]
        );
        let _ = fs::remove_dir_all(&out);
        seed(&out);
        clear_out(&out, Owner::Golden).unwrap();
        assert_eq!(
            left(&out),
            [
                "edit-1235.on.json",
                "notes.txt",
                "speed.json",
                "write-1234.off.json"
            ]
        );
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn a_refused_run_leaves_no_old_pass_on_disk() {
        let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden");
        let out = scratch("refused");
        let seed = || {
            fs::create_dir_all(&out).unwrap();
            fs::write(out.join("verdict.json"), br#"{"status":"pass"}"#).unwrap();
            fs::write(out.join("speed.json"), b"{}").unwrap();
        };
        // A refused --prompts: the old verdict is gone, speed's report is not golden's to remove.
        seed();
        let e = run(Cmd::Golden {
            model: PathBuf::from("no-such-model.gguf"),
            golden: golden.clone(),
            prompts: vec!["99-nope".to_string()],
            out: out.clone(),
            device: None,
            allow_small_golden: false,
            spec: SpecFlags::default(),
        })
        .unwrap_err()
        .to_string();
        assert!(e.contains("not a golden prompt"), "{e}");
        assert!(!out.join("verdict.json").exists(), "a stale pass survived");
        assert!(out.join("speed.json").exists());
        // A missing golden directory, too.
        seed();
        run(Cmd::Golden {
            model: PathBuf::from("no-such-model.gguf"),
            golden: golden.join("nowhere"),
            prompts: Vec::new(),
            out: out.clone(),
            device: None,
            allow_small_golden: false,
            spec: SpecFlags::default(),
        })
        .unwrap_err();
        assert!(!out.join("verdict.json").exists());
        // A refused speed run clears speed's report and leaves golden's verdict.
        seed();
        let c = parse(&[
            "speed",
            "--model",
            "m",
            "--out",
            out.to_str().unwrap(),
            "--edit-tokens",
            "e.tokens",
            "--speculation",
            "off",
        ])
        .unwrap();
        assert!(run(c).is_err());
        assert!(!out.join("speed.json").exists());
        assert!(out.join("verdict.json").exists());
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn made_from_and_expected_must_name_the_same_prompts() {
        // A deleted expected file: made-from.json still names it.
        let dir = mini_golden("made-extra", &["01-a", "02-b", "03-c"], 3);
        fs::remove_file(dir.join("expected").join("02-b.json")).unwrap();
        let e = golden_set(&dir).unwrap_err().to_string();
        assert!(e.contains("02-b (in made-from.json only)"), "{e}");
        // An expected file made-from.json doesn't know.
        let dir2 = mini_golden("made-missing", &["01-a", "02-b"], 2);
        fs::write(dir2.join("expected").join("03-c.json"), b"{}").unwrap();
        fs::write(dir2.join("prompts").join("03-c.tokens"), [0u8; 12]).unwrap();
        let e = golden_set(&dir2).unwrap_err().to_string();
        assert!(e.contains("03-c (in expected/ only)"), "{e}");
        // Both agree: fine.
        assert_eq!(
            golden_set(&mini_golden("made-ok", &["01-a"], 1))
                .unwrap()
                .len(),
            1
        );
        for d in [dir, dir2] {
            let _ = fs::remove_dir_all(d);
        }
        let _ = fs::remove_dir_all(scratch("made-ok"));
    }

    #[test]
    fn a_golden_set_of_fewer_than_twenty_prompts_is_never_a_pass() {
        assert_eq!(GOLDEN_SET_SIZE, 20);
        assert_eq!(required_prompts(19, false), 20);
        assert_eq!(required_prompts(20, false), 20);
        assert_eq!(required_prompts(21, false), 21);
        assert_eq!(required_prompts(2, true), 2);
        // The committed set is the gate's 20.
        let set = golden_set(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden"))
            .unwrap();
        assert_eq!(required_prompts(set.len(), false), set.len());
        // Two prompts, all compared and clean: a subset under the floor, a pass only when the test flag lifts it.
        let ok = |mode: Mode, _: &str, _: &[u32]| -> Result<Record> {
            let steps = [(5, 9.0), (6, 9.0)];
            Ok(if mode == Mode::Off {
                off(&steps)
            } else {
                on(&steps)
            })
        };
        let out = scratch("small");
        let v = golden_with(&prompts(), required_prompts(2, false), &out, ok).unwrap();
        assert_eq!(v.status, "subset");
        assert!(v.problems.iter().any(|p| p.contains("only 2 of 20")));
        let v = golden_with(&prompts(), required_prompts(2, true), &out, ok).unwrap();
        assert_eq!(v.status, "pass");
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn the_small_set_flag_is_hidden_and_off_by_default() {
        match parse(&["golden", "--model", "m", "--out", "o"]).unwrap() {
            Cmd::Golden {
                allow_small_golden, ..
            } => assert!(!allow_small_golden),
            _ => panic!("not golden"),
        }
        match parse(&[
            "golden",
            "--model",
            "m",
            "--out",
            "o",
            "--allow-small-golden",
        ])
        .unwrap()
        {
            Cmd::Golden {
                allow_small_golden, ..
            } => assert!(allow_small_golden),
            _ => panic!("not golden"),
        }
        let help = Wrap::try_parse_from(["spec-check", "golden", "--help"])
            .err()
            .expect("--help ends parsing")
            .to_string();
        assert!(!help.contains("allow-small-golden"));
    }

    // ----- the command line -------------------------------------------------------------------------------------

    #[derive(Parser)]
    struct Wrap {
        #[command(subcommand)]
        cmd: Cmd,
    }

    fn parse(args: &[&str]) -> Result<Cmd, clap::Error> {
        let mut v = vec!["spec-check"];
        v.extend(args);
        Wrap::try_parse_from(v).map(|w| w.cmd)
    }

    #[test]
    fn the_golden_command_line() {
        match parse(&["golden", "--model", "m.gguf", "--out", "o"]).unwrap() {
            Cmd::Golden {
                golden, prompts, ..
            } => {
                assert_eq!(golden, PathBuf::from("testdata/golden"));
                assert!(prompts.is_empty());
            }
            _ => panic!("not golden"),
        }
        let all = [
            "golden",
            "--model",
            "m",
            "--out",
            "o",
            "--prompts",
            "01-code,03-chat",
            "--golden",
            "g",
        ];
        match parse(&all).unwrap() {
            Cmd::Golden {
                golden, prompts, ..
            } => {
                assert_eq!(golden, PathBuf::from("g"));
                assert_eq!(prompts, ["01-code", "03-chat"]);
            }
            _ => panic!("not golden"),
        }
        assert!(parse(&["golden", "--model", "m"]).is_err());
    }

    #[test]
    fn the_speed_command_line_takes_generates_speculation_flags() {
        assert!(parse(&["speed", "--model", "m", "--out", "o"]).is_err());
        let base = [
            "speed",
            "--model",
            "m",
            "--out",
            "o",
            "--edit-tokens",
            "e.tokens",
        ];
        match parse(&base).unwrap() {
            Cmd::Speed { speed, spec, .. } => {
                assert_eq!(speed, PathBuf::from("testdata/speed"));
                let c = spec.config(0.0, false).unwrap().unwrap();
                assert_eq!(c.fixed_draft, None);
                assert_eq!(c.max_draft, SpecConfig::default().max_draft);
            }
            _ => panic!("not speed"),
        }
        let mut v = base.to_vec();
        v.extend(["--draft-len", "8", "--accept-prior", "0.7"]);
        match parse(&v).unwrap() {
            Cmd::Speed { spec, .. } => {
                let c = spec.config(0.0, false).unwrap().unwrap();
                assert_eq!(
                    (c.fixed_draft, c.max_draft, c.accept_prior),
                    (Some(8), 8, 0.7)
                );
            }
            _ => panic!("not speed"),
        }
        // `--speculation off` parses (it is generate's flag) and `run` refuses it before any load.
        let mut v = base.to_vec();
        v.extend(["--speculation", "off"]);
        let e = run(parse(&v).unwrap()).unwrap_err().to_string();
        assert!(e.contains("drop --speculation off"), "{e}");
    }

    #[test]
    fn an_unknown_golden_prompt_or_directory_is_refused_before_the_load() {
        let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden");
        let out = scratch("unused");
        let cmd = |prompts: &[&str], golden: &Path| Cmd::Golden {
            model: PathBuf::from("no-such-model.gguf"),
            golden: golden.to_path_buf(),
            prompts: prompts.iter().map(|s| s.to_string()).collect(),
            out: out.clone(),
            device: None,
            allow_small_golden: false,
            spec: SpecFlags::default(),
        };
        let e = run(cmd(&["99-nope"], &golden)).unwrap_err().to_string();
        assert!(e.contains("not a golden prompt"), "{e}");
        let e = run(cmd(&[], &golden.join("nowhere")))
            .unwrap_err()
            .to_string();
        assert!(e.contains("reading"), "{e}");
        // With a real prompt the checks pass and the model is the first thing opened.
        let e = run(cmd(&["01-code"], &golden)).unwrap_err().to_string();
        assert!(e.contains("no-such-model.gguf"), "{e}");
        assert!(!out.exists(), "nothing is written before the model loads");
    }

    // ----- the engine's side ----------------------------------------------------------------------------------

    /// Proposes the tokens after the ones generated so far (the tiny model's output holds no n-gram lookup could
    /// find, so a lookup drafter would never verify).
    struct Script {
        tokens: Vec<u32>,
        primed: bool,
        seen: usize,
    }

    impl engine::spec::Drafter for Script {
        fn name(&self) -> &'static str {
            "script"
        }
        fn draft(&mut self, max: usize) -> Vec<u32> {
            self.tokens
                .iter()
                .skip(self.seen)
                .take(max)
                .copied()
                .collect()
        }
        fn observe(&mut self, tokens: &[u32]) {
            // The first call is the prompt; then each token generated, `x` first.
            if std::mem::replace(&mut self.primed, true) {
                self.seen += tokens.len();
            }
        }
    }

    /// The real engine on the tiny model, off then on through `Rig`: the records are generate's, the off run made no
    /// verify step, the on run (a drafter proposing the off run's own tokens, a fixed draft of 4) made verify steps
    /// and kept drafts, and the pair goes through the comparator and passes.
    #[test]
    #[ignore = "needs the GPU"]
    fn a_rig_runs_the_tiny_model_off_and_on() {
        use qwen36::testing::tiny_model;
        let d = Dims::tiny();
        let g = Gguf::from_bytes(tiny_model(1)).unwrap();
        // The default sources, so the rig loads the MTP: its session is the shortest it holds (512 positions).
        let cfg = SpecConfig {
            max_draft: 4,
            fixed_draft: Some(4),
            ..SpecConfig::default()
        };
        let mut rig = Rig::from_gguf(&g, &d, None, 512, cfg.clone(), Vec::new()).unwrap();
        assert!(rig.e.mtp().is_some() && rig.s.has_mtp());
        let prompt: Vec<u32> = (0..10u32)
            .map(|i| (i * 211 + 17) % d.n_vocab as u32)
            .collect();
        let job = Job {
            max_tokens: 24,
            ..Job::golden()
        };
        let a = rig.run(Mode::Off, &prompt, &job).unwrap();
        assert!(a.spec.is_none() && a.settings.speculation.is_none());
        assert_eq!((a.steps.len(), a.positions.len()), (24, 24));
        let script: Vec<u32> = a.steps.iter().map(|s| s.token).collect();
        rig.state = SpecState::with_drafters(
            cfg.clone(),
            Box::new(move || {
                Box::new(Script {
                    tokens: script.clone(),
                    primed: false,
                    seen: 0,
                })
            }),
        );
        let b = rig.run(Mode::On, &prompt, &job).unwrap();
        assert!(b.settings.speculation.is_some());
        let spec = b.spec.clone().expect("an on run has counts");
        assert!(spec.steps > 0 && spec.accepted > 0, "{spec:?}");
        assert_eq!(b.positions.len(), b.steps.len());
        // Two off runs on one rig are the same run: the reset between runs leaves nothing behind.
        let a2 = rig.run(Mode::Off, &prompt, &job).unwrap();
        assert_eq!(a.steps, a2.steps);
        assert_eq!(a.positions, a2.positions);
        // The verify pass may part from decode at a near tie and nowhere else.
        let v = judge(vec![compare("tiny", &a, &b)], spec);
        assert_eq!(v.status, "pass", "{v:?}");
        // The real lookup drafter finds nothing in the tiny model's output, so MTP drafts every step: through
        // `SessionVerifier`, `Anyhow` and `Speculating`, the tokens are the plain loop's up to a near tie.
        rig.state = SpecState::new(cfg);
        let c = rig.run(Mode::On, &prompt, &job).unwrap();
        let mtp = c.spec.clone().expect("an on run has counts");
        assert!(mtp.drafted_mtp > 0 && mtp.drafted_lookup == 0, "{mtp:?}");
        assert!(!mtp.mtp_depth_counts.is_empty(), "{mtp:?}");
        let v = judge(vec![compare("tiny-mtp", &a, &c)], mtp);
        assert_eq!(v.status, "pass", "{v:?}");
    }
}
