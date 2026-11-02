//! Test 5's pass rule: the same prompt at several cache sizes, and with a swap forced every
//! step, gives the same tokens and bit-identical logits at every position, the prompt's through prefill and the rest
//! through decode. HOST and VRAM experts run the same code, so where an expert lives must never change a bit.
use crate::{generate::Record, test3::Status};
use serde::Serialize;

/// One run, as the verdict reports it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunSummary {
    pub label: String,
    pub cache_slots: usize,
    pub forced_swaps: bool,
    pub positions: usize,
    pub hits: u64,
    pub misses: u64,
    /// Routed-expert reads served from a staging buffer (`CacheStats::staged`).
    pub staged: u64,
    pub swaps_started: u64,
    pub steps: u64,
    /// The cache policy the run's record names: "adaptive" or "static".
    pub cache_policy: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Verdict {
    pub status: Status,
    pub runs: Vec<RunSummary>,
    /// The first run, in the order given, whose logits differ from the first run's, and the first position where they
    /// do.
    pub first_difference: Option<(String, usize)>,
    pub problems: Vec<String>,
    /// The `--report-run` runs, each compared with the first run: reported, never in the
    /// status. The speculation-off run is one: it is expected to differ in the DeltaNet layers.
    pub reports: Vec<Report>,
}

/// One report run against the first gating run.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Report {
    pub label: String,
    /// The same prompt and the same generated tokens.
    pub identical_tokens: bool,
    /// The same prompt and the same logits fingerprints at every position, and there are some.
    pub identical_logits: bool,
}

/// How `report` compares with `base`. A run of another prompt, or without fingerprints, is not identical.
fn report(label: &str, report: &Record, base: Option<&Record>) -> Report {
    let same_prompt = base.is_some_and(|b| b.prompt_tokens == report.prompt_tokens);
    let tokens = |r: &Record| r.steps.iter().map(|s| s.token).collect::<Vec<u32>>();
    Report {
        label: label.to_string(),
        identical_tokens: same_prompt && base.is_some_and(|b| tokens(b) == tokens(report)),
        identical_logits: same_prompt
            && !report.positions.is_empty()
            && base.is_some_and(|b| b.positions == report.positions),
    }
}

/// The five runs, by label (decision of 2026-10-09): the cache slots each must have after clamping, and whether
/// it is forced. `cache-max` is the clamped maximum, so it only has to exceed 102.
const LABELS: [&str; 5] = [
    "cache-0",
    "cache-26",
    "cache-102",
    "cache-max",
    "cache-102-swaps",
];

fn configurations(runs: &[(String, Record)], problems: &mut Vec<String>) {
    let mut seen: Vec<&str> = Vec::new();
    for (label, r) in runs {
        let label = label.as_str();
        if !LABELS.contains(&label) {
            problems.push(format!(
                "{label}: not one of the five runs ({})",
                LABELS.join(", ")
            ));
            continue;
        }
        if seen.contains(&label) {
            problems.push(format!("{label}: given more than once"));
            continue;
        }
        seen.push(label);
        let (slots, forced) = (r.settings.cache_slots, r.settings.force_swap_every_token);
        let (slots_ok, want) = match label {
            "cache-0" => (slots == 0, "0 slots".to_string()),
            "cache-26" => (slots == 26, "26 slots".to_string()),
            "cache-102" | "cache-102-swaps" => (slots == 102, "102 slots".to_string()),
            _ => (slots > 102, "more than 102 slots".to_string()),
        };
        if !slots_ok {
            problems.push(format!(
                "{label}: its record has {slots} slots per layer; this run needs {want}"
            ));
        }
        if label == "cache-102-swaps" && !forced {
            problems.push(format!(
                "{label}: its record wasn't run with --force-swap-every-token"
            ));
        }
        // The runs with a cache swap live under the adaptive policy; the forced run's one swap a
        // step is the only one, so its policy is static.
        let policy = r.settings.cache_policy.as_str();
        let want = match label {
            "cache-26" | "cache-102" | "cache-max" => Some("adaptive"),
            "cache-102-swaps" => Some("static"),
            _ => None,
        };
        if let Some(want) = want.filter(|&w| w != policy) {
            problems.push(format!(
                "{label}: its record ran the {policy} cache policy; this run needs the {want} one"
            ));
        }
        if label != "cache-102-swaps" && forced {
            problems.push(format!(
                "{label}: its record was run with --force-swap-every-token, which only cache-102-swaps does"
            ));
        }
    }
    let missing: Vec<&str> = LABELS
        .iter()
        .copied()
        .filter(|l| !seen.contains(l))
        .collect();
    if !missing.is_empty() {
        problems.push(format!("missing runs: {}", missing.join(", ")));
    }
}

/// Test 5 runs the speculative path, and runs the same steps at every cache size: each gating run must
/// have speculated with a fixed draft length, the same in all of them, and made at least one verify step. Without this
/// a set of runs with `--speculation off` would pass and say nothing about verify. A set served by the CPU miss path
/// (every run with `cpu_share` above 0, which the engine refuses with speculation) is the one exception: it runs plain
/// decode throughout, and has no speculation to test. The exemption is the whole set's: one such run among speculating
/// ones is a plain run that should have speculated.
///
/// With MTP a source, at least one gating run must have drafted with it: MTP drafts where lookup has none, so
/// a set in which none did never ran MTP's rows at any cache size, and would pass and say nothing about them.
fn speculation_problems(runs: &[(String, Record)]) -> Vec<String> {
    let mut problems = Vec::new();
    let mut first_len: Option<(&str, usize)> = None;
    if !runs.is_empty()
        && runs
            .iter()
            .all(|(_, r)| r.settings.speculation.is_none() && r.settings.cpu_share > 0.0)
    {
        return problems;
    }
    for (label, r) in runs {
        let (spec, counts) = (&r.settings.speculation, &r.spec);
        let Some(len) = spec.as_ref().and_then(|s| s.draft_len) else {
            problems.push(match spec {
                None => format!(
                    "{label}: its record ran with speculation off; test 5's gating runs speculate with --draft-len 4"
                ),
                Some(_) => format!(
                    "{label}: its record ran the draft-length controller; test 5's gating runs fix the draft length \
                     (--draft-len 4), so every cache size runs the same steps"
                ),
            });
            continue;
        };
        match first_len {
            None => first_len = Some((label, len)),
            Some((base, n)) if n != len => problems.push(format!(
                "{label}: its record fixed the draft length at {len}, {base}'s at {n}; the runs must run the same steps"
            )),
            Some(_) => {}
        }
        if counts.as_ref().is_none_or(|c| c.steps == 0) {
            problems.push(format!(
                "{label}: its record made no verify step, so it didn't test the speculative path"
            ));
        }
    }
    let mtp_source = runs.iter().any(|(_, r)| {
        r.settings
            .speculation
            .as_ref()
            .is_some_and(|s| s.sources.iter().any(|n| n == "mtp"))
    });
    if mtp_source
        && !runs
            .iter()
            .any(|(_, r)| r.spec.as_ref().is_some_and(|c| c.drafted_mtp > 0))
    {
        problems.push(
            "MTP is a source, but no gating run drafted with it, so test 5 didn't run MTP's rows at any cache size"
                .to_string(),
        );
    }
    problems
}

/// Test 5. Every run's prompt tokens, generated tokens and every position's logits fingerprint must equal the first
/// run's. Guards, so the comparison can't pass by not using the cache: a run without
/// a cache reads no expert from the cache, every run with one reads some, and a run with forced swaps started one swap per
/// step. The runs at 26, 102 and the most slots run the adaptive policy and must
/// swap live, and the forced run's policy is static. The prompt runs through prefill, so a step is a prefill
/// chunk or a decoded position, and every run must fingerprint every prompt position (`--all-logits`): prefill reads
/// experts through the slot table too, or, in a staged chunk, through the prefill table built from it.
pub fn judge(runs: &[(String, Record)]) -> Verdict {
    judge_with_reports(runs, &[])
}

/// `judge`, and each of `reports` compared with the first run and listed in the verdict's `reports`. A report run is
/// never a gating run: whatever it shows, the status is the five runs'.
pub fn judge_with_reports(runs: &[(String, Record)], reports: &[(String, Record)]) -> Verdict {
    let mut problems = Vec::new();
    let mut first_difference = None;
    if runs.len() < 2 {
        problems.push(format!(
            "test 5 compares at least two runs; {} given",
            runs.len()
        ));
    }
    configurations(runs, &mut problems);
    let summaries: Vec<RunSummary> = runs
        .iter()
        .map(|(label, r)| RunSummary {
            label: label.clone(),
            cache_slots: r.settings.cache_slots,
            forced_swaps: r.settings.force_swap_every_token,
            positions: r.positions.len(),
            hits: r.cache.hits,
            misses: r.cache.misses,
            staged: r.cache.staged,
            swaps_started: r.cache.swaps_started,
            steps: r.cache.steps,
            cache_policy: r.settings.cache_policy.clone(),
        })
        .collect();
    problems.extend(speculation_problems(runs));
    for ((s, label), (_, r)) in summaries.iter().zip(runs.iter().map(|r| &r.0)).zip(runs) {
        if !r.settings.fingerprints {
            problems.push(format!(
                "{label}: recorded with --no-fingerprints, so it has no logits to compare"
            ));
            continue;
        }
        if s.positions == 0 {
            problems.push(format!("{label}: no position was decoded"));
        } else if s.positions != r.prompt_tokens.len() + r.timings.decoded {
            problems.push(format!(
                "{label}: {} positions fingerprinted, not the prompt's {} and the {} decoded; run it with --all-logits",
                s.positions,
                r.prompt_tokens.len(),
                r.timings.decoded
            ));
        }
        if s.cache_slots == 0 && s.hits > 0 {
            problems.push(format!(
                "{label}: a run without a cache read {} routed experts from the cache",
                s.hits
            ));
        }
        if s.cache_slots > 0 && s.hits == 0 {
            problems.push(format!(
                "{label}: a run with {} slots per layer read no routed expert from the cache, so the cache wasn't used",
                s.cache_slots
            ));
        }
        if s.cache_slots > 0 && s.cache_policy == "adaptive" && s.swaps_started == 0 {
            problems.push(format!(
                "{label}: the adaptive policy started no swap, so nothing was swapped live"
            ));
        }
        if s.forced_swaps && s.swaps_started != s.steps {
            problems.push(format!(
                "{label}: {} swaps started over {} steps; a forced run starts one per step",
                s.swaps_started, s.steps
            ));
        }
    }
    if let Some((base_label, base)) = runs.first() {
        for (label, r) in &runs[1..] {
            if r.prompt_tokens != base.prompt_tokens {
                problems.push(format!(
                    "{label}: its prompt tokens differ from {base_label}'s, so the runs can't be compared"
                ));
                continue;
            }
            let ours: Vec<u32> = r.steps.iter().map(|s| s.token).collect();
            let theirs: Vec<u32> = base.steps.iter().map(|s| s.token).collect();
            if ours != theirs {
                let at = ours
                    .iter()
                    .zip(&theirs)
                    .position(|(a, b)| a != b)
                    .unwrap_or(ours.len().min(theirs.len()));
                problems.push(format!(
                    "{label}: generated token {at} differs from {base_label}'s ({} steps against {})",
                    ours.len(),
                    theirs.len()
                ));
                // Token `at` is chosen from the logits of position `prompt - 1 + at`.
                if first_difference.is_none() {
                    first_difference = Some((
                        label.clone(),
                        (r.prompt_tokens.len() + at).saturating_sub(1),
                    ));
                }
            }
            if r.positions != base.positions {
                let at = r
                    .positions
                    .iter()
                    .zip(&base.positions)
                    .position(|(a, b)| a != b)
                    .unwrap_or(r.positions.len().min(base.positions.len()));
                problems.push(format!(
                    "{label}: the logits at position {at} differ from {base_label}'s"
                ));
                // The earlier of the two, when a token difference was noted above.
                if first_difference.as_ref().is_none_or(|f| f.1 > at) {
                    first_difference = Some((label.clone(), at));
                }
            }
        }
    }
    Verdict {
        status: if problems.is_empty() {
            Status::Pass
        } else {
            Status::Fail
        },
        runs: summaries,
        first_difference,
        problems,
        reports: reports
            .iter()
            .map(|(label, r)| report(label, r, runs.first().map(|b| &b.1)))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::{
        fingerprint, CacheStats, Settings, SpecRecord, SpecSettings, Step, Stop, Timings, FORMAT,
    };

    /// A greedy run at `slots` slots per layer: a 4-token prompt prefilled in one step, and 3 generated tokens, of
    /// which 2 are decoded: 6 positions in 3 steps.
    fn run(slots: usize, forced: bool) -> Record {
        let positions: Vec<String> = (0..6).map(|p| fingerprint(&[p as f32, 1.0])).collect();
        Record {
            format: FORMAT.into(),
            prompt_tokens: vec![1, 2, 3, 4],
            settings: Settings {
                greedy: true,
                temp: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                presence_penalty: 0.0,
                seed: 0,
                max_tokens: 3,
                ctx: 4096,
                cache_slots: slots,
                force_swap_every_token: forced,
                all_logits: true,
                cache_slots_asked: None,
                fingerprints: true,
                profiled: false,
                f32_prefill: false,
                no_staging: false,
                stage_from: None,
                cache_policy: if forced { "static" } else { "adaptive" }.into(),
                upload_cap_mib: None,
                no_lend: false,
                base_time: false,
                cpu_share: 0.0,
                // Lookup alone. The MTP source is added by `with_mtp`.
                speculation: Some(SpecSettings::from(&engine::spec::SpecConfig {
                    max_draft: 4,
                    fixed_draft: Some(4),
                    sources: vec![engine::spec::Source::Lookup],
                    ..engine::spec::SpecConfig::default()
                })),
            },
            steps: [7, 8, 9]
                .iter()
                .map(|&t| Step {
                    token: t,
                    top: t,
                    logit: 1.0,
                    second: 0,
                    second_logit: 0.5,
                })
                .collect(),
            stop: Stop::MaxTokens,
            timings: Timings {
                decoded: 2,
                ..Timings::default()
            },
            cache: CacheStats {
                slots_per_layer: slots,
                mib: 0.0,
                // Six positions × 40 layers × 8 routed experts.
                hits: if slots == 0 {
                    0
                } else {
                    (slots as u64).min(1_920)
                },
                misses: if slots == 0 {
                    1_920
                } else {
                    1_920 - (slots as u64).min(1_920)
                },
                // A forced run swaps once a step; the adaptive policy swaps live wherever there is a cache.
                swaps_started: if forced {
                    3
                } else if slots > 0 {
                    5
                } else {
                    0
                },
                swaps_done: if forced { 1 } else { 0 },
                steps: 3,
                longest_submit_ms: 0.0,
                steps_overlapping_copies: 0,
                staged: 0,
                ..CacheStats::default()
            },
            memory: Default::default(),
            vram: Default::default(),
            prefill: Default::default(),
            warm: None,
            spec: Some(SpecRecord {
                steps: 1,
                plain_steps: 0,
                drafted: 4,
                accepted: 3,
                replays: 0,
                drafted_lookup: 4,
                accepted_lookup: 3,
                ..SpecRecord::default()
            }),
            positions,
        }
    }

    /// The five runs of test 5.
    fn five() -> Vec<(String, Record)> {
        vec![
            ("cache-0".into(), run(0, false)),
            ("cache-26".into(), run(26, false)),
            ("cache-102".into(), run(102, false)),
            ("cache-max".into(), run(140, false)),
            ("cache-102-swaps".into(), run(102, true)),
        ]
    }

    /// The five runs of a set whose config has MTP as a source; `drafts[i]` is run i's drafted-by-MTP count (its
    /// drafts are then MTP's, in place of lookup's).
    fn with_mtp(drafts: [u64; 5]) -> Vec<(String, Record)> {
        let mut runs = five();
        for ((_, r), d) in runs.iter_mut().zip(drafts) {
            let s = r.settings.speculation.as_mut().unwrap();
            s.sources = vec!["lookup".into(), "mtp".into()];
            let c = r.spec.as_mut().unwrap();
            c.drafted_lookup = c.drafted - d;
            c.accepted_lookup = c.accepted.saturating_sub(d);
            c.drafted_mtp = d;
            c.accepted_mtp = d.min(c.accepted);
        }
        runs
    }

    #[test]
    fn with_mtp_a_gating_set_that_never_drafted_with_mtp_fails() {
        // MTP drafted in one run of the five: the set ran MTP's rows, and passes.
        let v = judge(&with_mtp([0, 0, 2, 0, 0]));
        assert_eq!(v.status, Status::Pass, "{v:?}");
        // MTP is a source and no run drafted with it: the set tested nothing of MTP.
        let v = judge(&with_mtp([0; 5]));
        assert_eq!(v.status, Status::Fail, "{v:?}");
        assert!(
            v.problems
                .iter()
                .any(|p| p.contains("MTP is a source")
                    && p.contains("no gating run drafted with it")),
            "{:?}",
            v.problems
        );
        // Lookup alone: no MTP draft is required.
        assert_eq!(judge(&five()).status, Status::Pass);
        // A run that records no MTP source at all (a record from before MTP) is lookup alone too.
        let mut runs = five();
        for r in &mut runs {
            r.1.settings.speculation.as_mut().unwrap().sources = vec!["lookup".into()];
        }
        assert_eq!(judge(&runs).status, Status::Pass);
    }

    #[test]
    fn a_run_recorded_without_fingerprints_is_refused() {
        let mut runs = five();
        runs[2].1.settings.fingerprints = false;
        runs[2].1.positions.clear();
        let v = judge(&runs);
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems
                .iter()
                .any(|p| p.contains("cache-102") && p.contains("--no-fingerprints")),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn identical_runs_pass() {
        let v = judge(&five());
        assert_eq!(v.status, Status::Pass, "{v:?}");
        assert_eq!(v.runs.len(), 5);
        assert_eq!((v.runs[4].swaps_started, v.runs[4].steps), (3, 3));
    }

    #[test]
    fn the_gating_runs_must_have_made_verify_steps_with_a_fixed_draft_length() {
        let problems = |runs: &[(String, Record)]| {
            let v = judge(runs);
            assert_eq!(v.status, Status::Fail, "{v:?}");
            v.problems.join("; ")
        };
        // Speculation off in every run: the comparison would pass and test nothing of verify.
        let mut runs = five();
        for r in &mut runs {
            r.1.settings.speculation = None;
            r.1.spec = None;
        }
        let p = problems(&runs);
        assert!(
            p.contains("cache-0: its record ran with speculation off"),
            "{p}"
        );
        assert!(
            p.contains("cache-102-swaps: its record ran with speculation off"),
            "{p}"
        );
        // The controller, not a fixed length.
        let mut runs = five();
        runs[2].1.settings.speculation.as_mut().unwrap().draft_len = None;
        assert!(problems(&runs).contains("cache-102: its record ran the draft-length controller"));
        // A fixed length that differs from the others'.
        let mut runs = five();
        runs[3].1.settings.speculation.as_mut().unwrap().draft_len = Some(2);
        assert!(problems(&runs)
            .contains("cache-max: its record fixed the draft length at 2, cache-0's at 4"));
        // Speculation on and no verify step ran.
        for none in [None, Some(SpecRecord::default())] {
            let mut runs = five();
            runs[1].1.spec = none;
            assert!(problems(&runs).contains("cache-26: its record made no verify step"));
        }
        // The CPU-share set runs plain decode, which the engine requires: exempt, and it passes with the share on.
        let mut runs = five();
        for r in &mut runs {
            r.1.settings.speculation = None;
            r.1.spec = None;
            r.1.settings.cpu_share = 0.5;
        }
        let v = judge(&runs);
        assert_eq!(v.status, Status::Pass, "{v:?}");
        // The exemption is the whole set's: one plain CPU-share run among speculating ones is still a plain run.
        let mut runs = five();
        runs[2].1.settings.speculation = None;
        runs[2].1.spec = None;
        runs[2].1.settings.cpu_share = 0.5;
        assert!(problems(&runs).contains("cache-102: its record ran with speculation off"));
        // Report runs are not gating runs: a report run with speculation off is fine.
        let mut off = run(102, false);
        off.settings.speculation = None;
        off.spec = None;
        assert_eq!(
            judge_with_reports(&five(), &[("plain".into(), off)]).status,
            Status::Pass
        );
    }

    #[test]
    fn a_report_run_is_compared_not_gated() {
        // The speculation-off run: same tokens, different logits in some DeltaNet layer.
        let mut off = run(102, false);
        off.positions[4] = fingerprint(&[4.0, 1.000_000_1]);
        let v = judge_with_reports(&five(), &[("plain".into(), off.clone())]);
        assert_eq!(v.status, Status::Pass, "{v:?}");
        assert!(
            v.problems.is_empty() && v.first_difference.is_none(),
            "{v:?}"
        );
        assert_eq!(
            v.reports,
            [Report {
                label: "plain".into(),
                identical_tokens: true,
                identical_logits: false
            }]
        );
        // Different tokens too; the status is still the five runs'.
        off.steps[2].token = 10;
        let v = judge_with_reports(&five(), &[("plain".into(), off.clone())]);
        assert_eq!(v.status, Status::Pass);
        assert_eq!(
            (v.reports[0].identical_tokens, v.reports[0].identical_logits),
            (false, false)
        );
        // An identical one, and the exact shape pinned for the runner.
        let v = judge_with_reports(&five(), &[("same".into(), run(102, false))]);
        assert_eq!(
            serde_json::to_value(&v.reports).unwrap(),
            serde_json::json!([{"label": "same", "identical_tokens": true, "identical_logits": true}])
        );
        // A report run never rescues or fails the gating runs: a differing gating run still fails.
        let mut runs = five();
        runs[3].1.positions[4] = fingerprint(&[4.0, 1.000_000_1]);
        assert_eq!(
            judge_with_reports(&runs, &[("same".into(), run(102, false))]).status,
            Status::Fail
        );
        // Without report runs the list is empty, and `judge` is the same verdict.
        assert!(judge(&five()).reports.is_empty());
        assert_eq!(judge(&five()), judge_with_reports(&five(), &[]));
    }

    #[test]
    fn a_report_run_of_another_prompt_or_without_fingerprints_is_not_identical() {
        let mut other = run(102, false);
        other.prompt_tokens[0] = 99;
        let mut bare = run(102, false);
        bare.settings.fingerprints = false;
        bare.positions.clear();
        let v = judge_with_reports(&five(), &[("other".into(), other), ("bare".into(), bare)]);
        assert_eq!(v.status, Status::Pass);
        assert_eq!(
            v.reports
                .iter()
                .map(|r| (r.identical_tokens, r.identical_logits))
                .collect::<Vec<_>>(),
            [(false, false), (true, false)]
        );
    }

    #[test]
    fn one_differing_position_names_the_run_and_the_position() {
        let mut runs = five();
        runs[3].1.positions[4] = fingerprint(&[4.0, 1.000_000_1]);
        let v = judge(&runs);
        assert_eq!(v.status, Status::Fail);
        assert_eq!(v.first_difference, Some(("cache-max".to_string(), 4)));
        assert!(v.problems[0].contains("position 4"), "{:?}", v.problems);
    }

    #[test]
    fn a_token_only_difference_sets_the_first_difference() {
        let mut runs = five();
        runs[1].1.steps[2].token = 10;
        let v = judge(&runs);
        // Token 2 of a 4-token prompt is chosen from position 5's logits.
        assert_eq!(v.first_difference, Some(("cache-26".to_string(), 5)));
    }

    #[test]
    fn a_run_with_no_positions_fails() {
        let mut runs = five();
        runs[2].1.positions.clear();
        let v = judge(&runs);
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems
                .iter()
                .any(|p| p.starts_with("cache-102: no position was decoded")),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn a_run_that_stops_early_or_late_fails_naming_the_step() {
        for (label_at, steps) in [(1usize, 2usize), (3, 4)] {
            let mut runs = five();
            runs[label_at].1.steps.resize(
                steps,
                Step {
                    token: 9,
                    top: 9,
                    logit: 1.0,
                    second: 0,
                    second_logit: 0.5,
                },
            );
            let v = judge(&runs);
            assert_eq!(v.status, Status::Fail);
            let label = runs[label_at].0.clone();
            assert!(
                v.problems.iter().any(|p| p.starts_with(&label)
                    && p.contains(&format!("token {}", steps.min(3)))
                    && p.contains(&format!("{steps} steps against 3"))),
                "{:?}",
                v.problems
            );
        }
    }

    #[test]
    fn a_differing_token_fails() {
        let mut runs = five();
        runs[1].1.steps[2].token = 10;
        let v = judge(&runs);
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems[0].contains("cache-26") && v.problems[0].contains("token 2"),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn a_cache_run_that_reads_nothing_from_vram_fails() {
        let mut runs = five();
        runs[2].1.cache.hits = 0;
        let v = judge(&runs);
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems[0].contains("cache-102") && v.problems[0].contains("wasn't used"),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn a_run_without_a_cache_that_reads_from_vram_fails() {
        let mut runs = five();
        runs[0].1.cache.hits = 8;
        let v = judge(&runs);
        assert_eq!(v.status, Status::Fail);
        assert!(v.problems[0].contains("cache-0"), "{:?}", v.problems);
    }

    #[test]
    fn a_forced_run_with_fewer_swaps_than_steps_fails() {
        let mut runs = five();
        runs[4].1.cache.swaps_started = 2;
        let v = judge(&runs);
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems[0].contains("2 swaps started over 3 steps"),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn a_run_without_every_prompt_position_fails() {
        // Without --all-logits, prefill fingerprints only the prompt's last position: 3 of the 6.
        let mut runs = five();
        runs[2].1.positions.drain(..3);
        let v = judge(&runs);
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems
                .iter()
                .any(|p| p.starts_with("cache-102: 3 positions fingerprinted")
                    && p.contains("--all-logits")),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn a_missing_run_fails_and_is_named() {
        let mut runs = five();
        runs.remove(0);
        let v = judge(&runs);
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems.iter().any(|p| p == "missing runs: cache-0"),
            "{:?}",
            v.problems
        );
        let mut runs = five();
        runs.pop();
        let v = judge(&runs);
        assert!(
            v.problems
                .iter()
                .any(|p| p == "missing runs: cache-102-swaps"),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn a_duplicate_label_fails() {
        let mut runs = five();
        runs.push(("cache-26".into(), run(26, false)));
        let v = judge(&runs);
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems
                .iter()
                .any(|p| p.starts_with("cache-26: given more than once")),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn an_unknown_label_fails() {
        let mut runs = five();
        runs[2].0 = "cache-103".into();
        let v = judge(&runs);
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems
                .iter()
                .any(|p| p.starts_with("cache-103: not one of")),
            "{:?}",
            v.problems
        );
        assert!(
            v.problems.iter().any(|p| p == "missing runs: cache-102"),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn a_record_whose_slots_do_not_match_its_label_fails() {
        for (i, wrong) in [(0, 5), (1, 25), (2, 101), (3, 102), (3, 0), (4, 26)] {
            let mut runs = five();
            runs[i].1.settings.cache_slots = wrong;
            let v = judge(&runs);
            assert_eq!(v.status, Status::Fail, "{i} {wrong}");
            let label = &runs[i].0;
            assert!(
                v.problems
                    .iter()
                    .any(|p| p.starts_with(&format!("{label}: ")) && p.contains("slots")),
                "{i} {wrong}: {:?}",
                v.problems
            );
        }
    }

    #[test]
    fn a_forced_run_whose_record_is_not_forced_fails() {
        let mut runs = five();
        runs[4].1.settings.force_swap_every_token = false;
        let v = judge(&runs);
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems.iter().any(
                |p| p == "cache-102-swaps: its record wasn't run with --force-swap-every-token"
            ),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn an_unforced_run_whose_record_is_forced_fails() {
        let mut runs = five();
        runs[1].1.settings.force_swap_every_token = true;
        let v = judge(&runs);
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems
                .iter()
                .any(|p| p.starts_with("cache-26: ") && p.contains("--force-swap-every-token")),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn the_cached_runs_swap_live_under_the_adaptive_policy() {
        for (i, label) in [(1, "cache-26"), (2, "cache-102"), (3, "cache-max")] {
            let mut runs = five();
            runs[i].1.settings.cache_policy = "static".into();
            let v = judge(&runs);
            assert_eq!(v.status, Status::Fail, "{label}");
            assert!(
                v.problems
                    .iter()
                    .any(|p| p.starts_with(&format!("{label}: "))
                        && p.contains("static cache policy")),
                "{label}: {:?}",
                v.problems
            );
            let mut runs = five();
            runs[i].1.cache.swaps_started = 0;
            let v = judge(&runs);
            let want = format!(
                "{label}: the adaptive policy started no swap, so nothing was swapped live"
            );
            assert!(v.problems.contains(&want), "{label}: {:?}", v.problems);
        }
        let mut runs = five();
        runs[4].1.settings.cache_policy = "adaptive".into();
        assert!(judge(&runs)
            .problems
            .iter()
            .any(|p| p.starts_with("cache-102-swaps: ") && p.contains("needs the static one")));
    }

    #[test]
    fn different_prompts_and_a_single_run_fail() {
        let mut runs = five();
        runs[1].1.prompt_tokens[0] = 99;
        assert_eq!(judge(&runs).status, Status::Fail);
        let v = judge(&five()[..1]);
        assert_eq!(v.status, Status::Fail);
        assert!(v.problems[0].contains("at least two"), "{:?}", v.problems);
    }

    #[test]
    fn test5_refuses_a_record_of_another_format() {
        let mut r = run(100, false);
        r.format = "artemis-generate-9".into();
        let p = std::env::temp_dir().join(format!("artemis-t5-fmt-{}.json", std::process::id()));
        std::fs::write(&p, serde_json::to_vec(&r).unwrap()).unwrap();
        let e = crate::generate::read_record(&p).unwrap_err().to_string();
        std::fs::remove_file(&p).unwrap();
        assert!(
            e.contains("artemis-t5-fmt-") && e.contains("artemis-generate-9"),
            "{e}"
        );
    }
}
