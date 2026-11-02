//! The hit-rate gate: the golden set through `generate` with 128 greedy tokens, the cache
//! policy on against `--cache-policy static`. The code prompts (01, 05 to 11), averaged, may be at most 1 point below
//! static; prose (12 to 15) and tool use (16 to 18), averaged, must be above it. A prompt's hit rate is the cache's
//! share of its run's routed-expert reads, `hits / (hits + staged + misses)`, prefill included, as test 5 counts it;
//! the decode hit rate, reported beside it, counts decode's reads alone.
use crate::{generate::Record, test3::Status};
use serde::Serialize;

/// The golden prompts the gate compares, by group.
pub const GROUPS: [(&str, &[&str]); 3] = [
    (
        "code",
        &[
            "01-code",
            "05-code-write-lru",
            "06-code-write-sql",
            "07-code-write-ts",
            "08-code-edit-bugfix",
            "09-code-edit-refactor",
            "10-code-explain-regex",
            "11-code-explain-rust",
        ],
    ),
    (
        "prose",
        &[
            "12-prose-story",
            "13-prose-email",
            "14-prose-summary",
            "15-prose-explain",
        ],
    ),
    (
        "tool",
        &["16-tool-weather", "17-tool-result", "18-tool-multi"],
    ),
];

/// How far the code prompts' average may fall below static's, in points.
pub const CODE_SLACK: f64 = 1.0;

/// The tokens each run generates.
pub const TOKENS: usize = 128;

/// A golden prompt's two runs, the policy's and the static one's, and the prompt's committed tokens
/// (testdata/golden/prompts/<id>.tokens).
#[derive(Clone, Debug)]
pub struct Pair {
    pub id: String,
    pub policy: Record,
    pub fixed: Record,
    pub prompt: Vec<u32>,
}

/// One prompt's runs.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Row {
    pub id: String,
    pub group: String,
    pub policy_pct: f64,
    pub static_pct: f64,
    pub policy_decode_pct: Option<f64>,
    pub static_decode_pct: Option<f64>,
    /// The policy run's swaps.
    pub swaps: u64,
}

/// A group's averages and its rule.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Group {
    pub name: String,
    pub policy_pct: f64,
    pub static_pct: f64,
    pub policy_decode_pct: f64,
    pub static_decode_pct: f64,
    pub rule: String,
    pub pass: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Verdict {
    pub status: Status,
    pub rows: Vec<Row>,
    pub groups: Vec<Group>,
    pub problems: Vec<String>,
}

/// The cache's share of a run's routed-expert reads, in percent: prefill's (staged reads count as reads, not hits) and
/// decode's.
pub fn hit_pct(r: &Record) -> f64 {
    let c = &r.cache;
    let total = c.hits + c.staged + c.misses;
    if total == 0 {
        0.0
    } else {
        100.0 * c.hits as f64 / total as f64
    }
}

/// The cache's share of decode's reads alone, in percent; None when nothing was decoded.
pub fn decode_pct(r: &Record) -> Option<f64> {
    let c = &r.cache;
    let total = c.decode_hits + c.decode_misses;
    (total > 0).then(|| 100.0 * c.decode_hits as f64 / total as f64)
}

/// Why a pair of runs can't be compared, if it can't: the two must be the same run of the committed prompt, with the
/// defaults but for the policy.
fn mismatch(id: &str, p: &Record, s: &Record, prompt: &[u32]) -> Vec<String> {
    let mut bad = Vec::new();
    for (which, r) in [("policy", p), ("static", s)] {
        let st = &r.settings;
        if !st.greedy {
            bad.push(format!("{id} ({which}): not greedy"));
        }
        if st.max_tokens != TOKENS {
            bad.push(format!(
                "{id} ({which}): --max-tokens {}, not {TOKENS}",
                st.max_tokens
            ));
        }
        // One message a flag, each naming it: the gate runs the defaults.
        for (refused, flag) in [
            (
                st.force_swap_every_token,
                "forced swaps (--force-swap-every-token)",
            ),
            (st.profiled, "--profile"),
            (st.upload_cap_mib.is_some(), "--upload-cap"),
            (st.no_lend, "--no-lend"),
            (st.no_staging, "--no-staging"),
            (st.stage_from.is_some(), "--stage-from"),
            (st.f32_prefill, "--f32-prefill"),
            (st.base_time, "--base-time"),
            (st.cpu_share != 0.0, "--cpu-share"),
        ] {
            if refused {
                bad.push(format!(
                    "{id} ({which}): {flag}; the gate runs the defaults"
                ));
            }
        }
    }
    if p.settings.cache_policy != "adaptive" {
        bad.push(format!(
            "{id}: the policy run's cache policy is {}, not adaptive",
            p.settings.cache_policy
        ));
    }
    if s.settings.cache_policy != "static" {
        bad.push(format!(
            "{id}: the static run's cache policy is {}, not static",
            s.settings.cache_policy
        ));
    }
    if p.prompt_tokens != s.prompt_tokens {
        bad.push(format!("{id}: the two runs' prompts differ"));
    }
    if p.prompt_tokens != prompt {
        bad.push(format!(
            "{id}: the policy run's prompt isn't the committed one (testdata/golden/prompts/{id}.tokens)"
        ));
    }
    let toks = |r: &Record| r.steps.iter().map(|s| s.token).collect::<Vec<_>>();
    if toks(p) != toks(s) {
        bad.push(format!(
            "{id}: the two runs generated different tokens; where an expert is read changes no bit (test 5)"
        ));
    }
    let (a, b) = (&p.settings, &s.settings);
    if a.ctx != b.ctx {
        bad.push(format!(
            "{id}: the runs' --ctx differ ({} and {})",
            a.ctx, b.ctx
        ));
    }
    if a.cache_slots_asked != b.cache_slots_asked {
        bad.push(format!(
            "{id}: the runs' --cache-slots differ ({:?} and {:?})",
            a.cache_slots_asked, b.cache_slots_asked
        ));
    }
    if a.cache_slots != b.cache_slots {
        bad.push(format!(
            "{id}: the runs' cache size differ ({} and {} slots)",
            a.cache_slots, b.cache_slots
        ));
    }
    if p.cache.lent_slots != s.cache.lent_slots {
        bad.push(format!(
            "{id}: the runs' lent rows differ ({} and {})",
            p.cache.lent_slots, s.cache.lent_slots
        ));
    }
    bad
}

fn mean(v: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = v.collect();
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// The gate over `runs`, every prompt of `GROUPS` exactly once.
pub fn judge(runs: &[Pair]) -> Verdict {
    let mut problems = Vec::new();
    let mut rows = Vec::new();
    for Pair {
        id,
        policy: p,
        fixed: s,
        prompt,
    } in runs
    {
        let Some((group, _)) = GROUPS.iter().find(|(_, ids)| ids.contains(&id.as_str())) else {
            problems.push(format!("{id}: not one of the gate's prompts"));
            continue;
        };
        if rows.iter().any(|r: &Row| &r.id == id) {
            problems.push(format!("{id}: given more than once"));
            continue;
        }
        problems.extend(mismatch(id, p, s, prompt));
        rows.push(Row {
            id: id.clone(),
            group: group.to_string(),
            policy_pct: hit_pct(p),
            static_pct: hit_pct(s),
            policy_decode_pct: decode_pct(p),
            static_decode_pct: decode_pct(s),
            swaps: p.cache.swaps_started,
        });
    }
    let mut groups = Vec::new();
    for (name, ids) in GROUPS {
        let missing: Vec<&str> = ids
            .iter()
            .copied()
            .filter(|i| !rows.iter().any(|r| r.id == *i))
            .collect();
        if !missing.is_empty() {
            problems.push(format!("{name}: missing {}", missing.join(", ")));
            continue;
        }
        let of = |f: fn(&Row) -> f64| mean(rows.iter().filter(|r| r.group == name).map(f));
        let (pol, sta) = (of(|r| r.policy_pct), of(|r| r.static_pct));
        let (rule, pass) = if name == "code" {
            (
                format!("at most {CODE_SLACK} point below static"),
                pol >= sta - CODE_SLACK,
            )
        } else {
            ("above static".to_string(), pol > sta)
        };
        groups.push(Group {
            name: name.to_string(),
            policy_pct: pol,
            static_pct: sta,
            policy_decode_pct: of(|r| r.policy_decode_pct.unwrap_or(f64::NAN)),
            static_decode_pct: of(|r| r.static_decode_pct.unwrap_or(f64::NAN)),
            rule,
            pass,
        });
    }
    let failed = groups.iter().any(|g| !g.pass);
    Verdict {
        status: if problems.is_empty() && !failed {
            Status::Pass
        } else {
            Status::Fail
        },
        rows,
        groups,
        problems,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::{CacheStats, Settings, Step, Stop, Timings, FORMAT};

    /// A 128-token greedy run whose cache served `pct` percent of 1,000 reads (decode's 640 of them at the same rate).
    fn run(policy: &str, pct: f64) -> Record {
        let hits = (pct * 10.0).round() as u64;
        let dh = (pct * 6.4).round() as u64;
        Record {
            format: FORMAT.into(),
            prompt_tokens: vec![1, 2, 3],
            settings: Settings {
                greedy: true,
                temp: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                presence_penalty: 0.0,
                seed: 0,
                max_tokens: TOKENS,
                ctx: 65_536,
                cache_slots: 124,
                force_swap_every_token: false,
                all_logits: false,
                cache_slots_asked: None,
                fingerprints: false,
                profiled: false,
                f32_prefill: false,
                no_staging: false,
                stage_from: None,
                cache_policy: policy.into(),
                upload_cap_mib: None,
                no_lend: false,
                base_time: false,
                cpu_share: 0.0,
                speculation: None,
            },
            steps: vec![Step {
                token: 9,
                top: 9,
                logit: 1.0,
                second: 0,
                second_logit: 0.5,
            }],
            stop: Stop::MaxTokens,
            positions: Vec::new(),
            timings: Timings::default(),
            cache: CacheStats {
                hits,
                misses: 1_000 - hits,
                decode_hits: dh,
                decode_misses: 640 - dh,
                swaps_started: if policy == "adaptive" { 300 } else { 0 },
                ..CacheStats::default()
            },
            memory: Default::default(),
            vram: Default::default(),
            prefill: Default::default(),
            warm: None,
            spec: None,
        }
    }

    fn pair(id: &str, pct: f64) -> Pair {
        Pair {
            id: id.into(),
            policy: run("adaptive", pct),
            fixed: run("static", 70.0),
            prompt: vec![1, 2, 3],
        }
    }

    /// Every gate prompt, the policy run at `code`, `prose` and `tool` percent and the static runs at 70.
    fn all(code: f64, prose: f64, tool: f64) -> Vec<Pair> {
        GROUPS
            .iter()
            .flat_map(|(g, ids)| {
                let pct = match *g {
                    "code" => code,
                    "prose" => prose,
                    _ => tool,
                };
                ids.iter().map(move |id| pair(id, pct))
            })
            .collect()
    }

    #[test]
    fn the_rules_per_group() {
        let v = judge(&all(75.0, 72.0, 71.0));
        assert_eq!(v.status, Status::Pass, "{v:?}");
        assert_eq!(v.rows.len(), 15);
        assert_eq!(
            (v.groups[0].policy_pct, v.groups[0].static_pct),
            (75.0, 70.0)
        );
        assert!((v.groups[0].policy_decode_pct - 75.0).abs() < 0.1);
        // Code may fall a point below static, not more; prose and tool must be above, not level.
        assert_eq!(judge(&all(69.0, 72.0, 71.0)).status, Status::Pass);
        let v = judge(&all(68.9, 72.0, 71.0));
        assert_eq!(v.status, Status::Fail);
        assert!(!v.groups[0].pass && v.groups[1].pass && v.groups[2].pass);
        for (prose, tool) in [(70.0, 71.0), (72.0, 70.0)] {
            assert_eq!(
                judge(&all(75.0, prose, tool)).status,
                Status::Fail,
                "{prose} {tool}"
            );
        }
    }

    #[test]
    fn runs_that_cant_be_compared_fail_by_name() {
        type Change = fn(&mut Record, &mut Record);
        let cases: [(&str, Change); 20] = [
            ("different tokens", |p, _| p.steps[0].token = 10),
            ("prompts differ", |p, _| p.prompt_tokens.push(4)),
            ("not adaptive", |p, _| {
                p.settings.cache_policy = "static".into()
            }),
            ("not static", |_, s| {
                s.settings.cache_policy = "adaptive".into()
            }),
            ("--max-tokens", |p, _| p.settings.max_tokens = 64),
            ("not greedy", |_, s| s.settings.greedy = false),
            ("cache size differ", |p, _| p.settings.cache_slots = 100),
            ("--ctx", |p, _| p.settings.ctx = 4096),
            ("--cache-slots", |_, s| {
                s.settings.cache_slots_asked = Some(124)
            }),
            ("forced swaps", |p, _| {
                p.settings.force_swap_every_token = true
            }),
            ("--profile", |_, s| s.settings.profiled = true),
            ("--upload-cap", |p, _| p.settings.upload_cap_mib = Some(16)),
            ("--no-lend", |p, _| p.settings.no_lend = true),
            ("--no-staging", |_, s| s.settings.no_staging = true),
            ("--stage-from", |p, _| p.settings.stage_from = Some(1)),
            ("--f32-prefill", |p, s| {
                p.settings.f32_prefill = true;
                s.settings.f32_prefill = true;
            }),
            ("--base-time", |p, _| p.settings.base_time = true),
            ("--cpu-share", |_, s| s.settings.cpu_share = 0.5),
            ("lent rows differ", |_, s| s.cache.lent_slots = 7),
            ("isn't the committed one", |p, s| {
                p.prompt_tokens = vec![1, 2, 4];
                s.prompt_tokens = vec![1, 2, 4];
            }),
        ];
        for (want, change) in cases {
            let mut runs = all(80.0, 80.0, 80.0);
            let Pair { policy, fixed, .. } = &mut runs[4];
            change(policy, fixed);
            let v = judge(&runs);
            assert_eq!(v.status, Status::Fail, "{want}");
            assert!(
                v.problems
                    .iter()
                    .any(|m| m.starts_with("08-code-edit-bugfix") && m.contains(want)),
                "{want}: {:?}",
                v.problems
            );
        }
    }

    /// Sets a run's reads: the cache's, the staged ones, the host's; then decode's alone.
    fn reads(r: &mut Record, hits: u64, staged: u64, misses: u64, dh: u64, dm: u64) {
        r.cache.hits = hits;
        r.cache.staged = staged;
        r.cache.misses = misses;
        r.cache.decode_hits = dh;
        r.cache.decode_misses = dm;
    }

    /// Decision 3: a prompt's rate is hits / (hits + staged + misses), prefill's staged reads counting as reads.
    #[test]
    fn staged_reads_count_in_the_runs_rate() {
        let mut r = run("adaptive", 0.0);
        reads(&mut r, 600, 200, 200, 0, 0);
        assert_eq!(hit_pct(&r), 60.0);
        reads(&mut r, 0, 0, 0, 0, 0);
        assert_eq!(hit_pct(&r), 0.0);
        // Code: 640 of 1,000 reads from the cache is 6 points under static's 70, though 80% of the unstaged reads were.
        let mut runs = all(80.0, 80.0, 80.0);
        for p in &mut runs[..8] {
            reads(&mut p.policy, 640, 200, 160, 512, 128);
        }
        let v = judge(&runs);
        assert!(!v.groups[0].pass, "{v:?}");
        assert_eq!(v.groups[0].policy_pct, 64.0);
        // Static's own staged reads lower its rate: 700 of 1,000 is 70, not 93.3. Prose at 75 beats it; 69 doesn't, and
        // would beat 93.3.
        let mut runs = all(80.0, 75.0, 80.0);
        for p in &mut runs[8..12] {
            reads(&mut p.fixed, 700, 250, 50, 448, 192);
            reads(&mut p.policy, 750, 0, 250, 480, 160);
        }
        assert!(judge(&runs).groups[1].pass);
        for p in &mut runs[8..12] {
            reads(&mut p.policy, 690, 0, 310, 480, 160);
        }
        assert!(!judge(&runs).groups[1].pass);
    }

    /// Decision 3: the rule judges the run's rate; decode's is reported beside it and judges nothing.
    #[test]
    fn the_run_rate_is_gated_and_decodes_only_reported() {
        let mut runs = all(80.0, 80.0, 80.0);
        // Code: the run is 2 points below static (68 against 70) while decode is above (80 against 70): fails.
        // Prose: the run is above (75 against 70) while decode is below (60 against 70): passes.
        for p in &mut runs[..8] {
            reads(&mut p.policy, 680, 0, 320, 512, 128);
        }
        for p in &mut runs[8..12] {
            reads(&mut p.policy, 750, 0, 250, 384, 256);
        }
        let v = judge(&runs);
        assert!(!v.groups[0].pass, "{v:?}");
        assert!(v.groups[1].pass, "{v:?}");
        assert_eq!(v.status, Status::Fail);
        assert!((v.groups[0].policy_decode_pct - 80.0).abs() < 0.1);
        assert!((v.groups[1].policy_decode_pct - 60.0).abs() < 0.1);
    }

    /// A group's rate is its prompts' average: not the first, not the lowest, not the highest.
    #[test]
    fn a_group_is_averaged_over_its_prompts() {
        // The first code prompt far below, the rest above: the mean (73.25) passes.
        let mut runs = all(78.0, 80.0, 80.0);
        reads(&mut runs[0].policy, 400, 0, 600, 256, 384);
        let v = judge(&runs);
        assert!(v.groups[0].pass, "{v:?}");
        assert_eq!(v.groups[0].policy_pct, (40.0 + 7.0 * 78.0) / 8.0);
        // The first far above, the rest below static: the mean (63.75) fails.
        let mut runs = all(60.0, 80.0, 80.0);
        reads(&mut runs[0].policy, 900, 0, 100, 576, 64);
        let v = judge(&runs);
        assert!(!v.groups[0].pass, "{v:?}");
        // Prose: one high prompt among three at static's rate carries the mean above it; one low among three above
        // it sinks the mean to static's.
        let mut runs = all(80.0, 70.0, 80.0);
        reads(&mut runs[8].policy, 990, 0, 10, 634, 6);
        assert!(judge(&runs).groups[1].pass);
        let mut runs = all(80.0, 71.0, 80.0);
        reads(&mut runs[11].policy, 600, 0, 400, 384, 256);
        assert!(!judge(&runs).groups[1].pass);
    }

    /// The code slack is exactly one point: 69.00 against static's 70.00 passes, 68.99 fails (10,000 reads a run).
    #[test]
    fn the_code_slack_is_one_point_exactly() {
        for (hits, pass) in [(6_900, true), (6_899, false), (6_901, true)] {
            let mut runs = all(80.0, 80.0, 80.0);
            for p in &mut runs[..8] {
                reads(&mut p.policy, hits, 0, 10_000 - hits, 0, 0);
                reads(&mut p.fixed, 7_000, 0, 3_000, 0, 0);
            }
            let v = judge(&runs);
            assert_eq!(v.groups[0].pass, pass, "{hits}: {v:?}");
        }
    }

    #[test]
    fn a_missing_extra_or_repeated_prompt_fails() {
        let mut runs = all(80.0, 80.0, 80.0);
        runs.remove(13);
        let v = judge(&runs);
        assert!(
            v.problems
                .iter()
                .any(|m| m == "tool: missing 17-tool-result"),
            "{:?}",
            v.problems
        );
        let mut runs = all(80.0, 80.0, 80.0);
        runs.push(pair("02-prose", 80.0));
        runs.push(runs[0].clone());
        let v = judge(&runs);
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems
                .iter()
                .any(|m| m.starts_with("02-prose: not one")),
            "{:?}",
            v.problems
        );
        assert!(
            v.problems
                .iter()
                .any(|m| m == "01-code: given more than once"),
            "{:?}",
            v.problems
        );
    }
}
