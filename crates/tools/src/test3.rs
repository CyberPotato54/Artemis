//! Test 3's pass rule, teacher-forced against a floor (decision of 2026-10-09). The
//! oracle is llama.cpp b11509's greedy run on each golden prompt (Vulkan, f16 KV, 256 steps). An engine is fed the
//! prompt and then the oracle's tokens one at a time, and at every step its top token is compared with the oracle's.
//! A flip is a step where they differ and the oracle's own top two logits are more than TIE_MARGIN apart. Pooled over
//! the prompts run, Artemis may flip at most FLIP_FACTOR times as often as llama.cpp's CPU backend does along the same
//! tokens (the floor, `testdata/golden/forced-cpu`). The first, free-running comparison is reported, not
//! gating: a free greedy run is fed the oracle's tokens until its top first differs, so it would diverge exactly where
//! the forced run first differs.
use crate::generate::Record;
use serde::{Deserialize, Serialize};

/// A step is counted only where the oracle's top two logits are more than this apart.
pub const TIE_MARGIN: f32 = 0.05;
/// Artemis's pooled flips may be at most this many times llama.cpp CPU's, as test 4's KL_FACTOR bounds its KL.
pub const FLIP_FACTOR: f64 = 1.5;
/// Steps per prompt: the oracle ran `--n 256`, so our run uses `--max-tokens 256`.
pub const STEPS: usize = 256;

/// One step of `artemis-oracle greedy`'s output.
#[derive(Clone, Debug, Deserialize)]
pub struct OracleStep {
    pub token: u32,
    pub logit: f32,
    pub second: u32,
    pub second_logit: f32,
}

/// `artemis-oracle greedy`'s output: `testdata/golden/expected/<id>.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct Oracle {
    pub prompt_tokens: usize,
    pub stopped_at_eog: bool,
    pub steps: Vec<OracleStep>,
}

/// One step of `artemis-oracle forced`'s output: the token fed there, the run's own top two, and the fed token's logit.
#[derive(Clone, Debug, Deserialize)]
pub struct FloorStep {
    pub token: u32,
    pub top: u32,
    pub logit: f32,
    pub second: u32,
    pub second_logit: f32,
    pub token_logit: f32,
}

/// `artemis-oracle forced`'s output along the oracle's tokens: `testdata/golden/forced-cpu/<id>.json`, the floor.
#[derive(Clone, Debug, Deserialize)]
pub struct Floor {
    pub backend: String,
    pub prompt_tokens: usize,
    pub steps: Vec<FloorStep>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    Pass,
    Fail,
}

/// A counted step where an engine's top token isn't the oracle's.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Flip {
    pub step: usize,
    /// The engine's top token.
    pub top: u32,
    pub oracle: u32,
    /// The oracle's top logit minus its second there: more than TIE_MARGIN.
    pub oracle_gap: f32,
}

/// Where a free greedy run would first leave the oracle's tokens: the forced run's first step whose top token isn't
/// the oracle's, counted or not. Reported, not gating.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Divergence {
    pub step: usize,
    pub top: u32,
    pub oracle: u32,
    pub oracle_gap: f32,
    /// Whether the first rule excused it: the oracle's gap is at most TIE_MARGIN.
    pub excused: bool,
}

/// One golden prompt's comparison.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PromptResult {
    pub id: String,
    /// Fail only when the runs can't be compared (`problem`). The flips are judged pooled, by `judge`.
    pub status: Status,
    /// The oracle's steps, every one compared.
    pub steps: usize,
    /// Steps where the oracle's top two logits are more than TIE_MARGIN apart.
    pub counted: usize,
    /// Artemis's flips.
    pub flips: Vec<Flip>,
    /// llama.cpp CPU's flips along the same tokens.
    pub floor_flips: Vec<Flip>,
    pub free_divergence: Option<Divergence>,
    pub floor_free_divergence: Option<Divergence>,
    pub problem: Option<String>,
}

/// Test 3's verdict over the prompts run.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Verdict {
    pub status: Status,
    pub counted: usize,
    /// Artemis's flips, pooled.
    pub flips: usize,
    /// llama.cpp CPU's flips, pooled: the floor.
    pub floor_flips: usize,
    /// FLIP_FACTOR × floor_flips.
    pub bound: f64,
    pub prompts: Vec<PromptResult>,
    pub problems: Vec<String>,
}

fn gap(s: &OracleStep) -> f32 {
    s.logit - s.second_logit
}

/// An engine's flips along the oracle's steps, from its top token at each, and its first difference counted or not.
fn count_flips(
    tops: impl Iterator<Item = u32>,
    oracle: &[OracleStep],
) -> (Vec<Flip>, Option<Divergence>) {
    let (mut flips, mut first) = (Vec::new(), None);
    for (step, (top, o)) in tops.zip(oracle).enumerate() {
        if top == o.token {
            continue;
        }
        let g = gap(o);
        if first.is_none() {
            first = Some(Divergence {
                step,
                top,
                oracle: o.token,
                oracle_gap: g,
                excused: g <= TIE_MARGIN,
            });
        }
        if g > TIE_MARGIN {
            flips.push(Flip {
                step,
                top,
                oracle: o.token,
                oracle_gap: g,
            });
        }
    }
    (flips, first)
}

/// Test 3 for one golden prompt. `want_prompt` is its committed token file; `oracle` and `floor` are its committed
/// expected output and floor; `ours` is the record of `artemis generate --greedy --max-tokens 256 --force-tokens <the
/// oracle's tokens> --json`. Every oracle step is compared, the end-of-generation step included. Fails, naming the
/// mismatch, if the oracle or the floor was made from another prompt, if the floor isn't llama.cpp CPU's run along the
/// oracle's tokens, if our prompt tokens differ from the committed ones (the tokenizer is checked end to end), or if
/// our run isn't greedy with 256 tokens and fed exactly the oracle's tokens. Otherwise it counts the flips.
pub fn compare(
    id: &str,
    want_prompt: &[u32],
    oracle: &Oracle,
    floor: &Floor,
    ours: &Record,
) -> PromptResult {
    let fail = |problem: String| PromptResult {
        id: id.into(),
        status: Status::Fail,
        steps: oracle.steps.len(),
        counted: 0,
        flips: Vec::new(),
        floor_flips: Vec::new(),
        free_divergence: None,
        floor_free_divergence: None,
        problem: Some(problem),
    };
    let n = oracle.steps.len();
    if oracle.prompt_tokens != want_prompt.len() {
        return fail(format!(
            "the expected output was made from a {}-token prompt, but the committed prompt has {} tokens: \
             regenerate testdata/golden",
            oracle.prompt_tokens,
            want_prompt.len()
        ));
    }
    if floor.backend != "cpu" {
        return fail(format!(
            "the floor was made on llama.cpp's {} backend; test 3's floor is its CPU backend",
            floor.backend
        ));
    }
    if floor.prompt_tokens != want_prompt.len() {
        return fail(format!(
            "the floor was made from a {}-token prompt, but the committed prompt has {} tokens: regenerate it \
             (build_golden_set.py --stage floor)",
            floor.prompt_tokens,
            want_prompt.len()
        ));
    }
    if let Some(k) =
        (0..n.min(floor.steps.len())).find(|&k| floor.steps[k].token != oracle.steps[k].token)
    {
        return fail(format!(
            "the floor was fed {} at step {k}, but the expected output's token there is {}: regenerate it \
             (build_golden_set.py --stage floor)",
            floor.steps[k].token, oracle.steps[k].token
        ));
    }
    if floor.steps.len() != n {
        return fail(format!(
            "the floor has {} steps and the expected output {n}: regenerate it (build_golden_set.py --stage floor)",
            floor.steps.len()
        ));
    }
    if let Some(at) = (0..ours.prompt_tokens.len().max(want_prompt.len()))
        .find(|&i| ours.prompt_tokens.get(i) != want_prompt.get(i))
    {
        let show = |v: &[u32]| v.get(at).map_or("the end".to_string(), |t| t.to_string());
        return fail(format!(
            "our prompt tokens differ from the committed ones at position {at}: ours {}, committed {}",
            show(&ours.prompt_tokens),
            show(want_prompt)
        ));
    }
    if !ours.settings.greedy {
        return fail("the run must be greedy (--greedy)".into());
    }
    if ours.settings.max_tokens != STEPS {
        return fail(format!(
            "the run must allow {STEPS} tokens (--max-tokens {STEPS}), as the oracle's --n {STEPS}; it allowed {}",
            ours.settings.max_tokens
        ));
    }
    if let Some(k) =
        (0..n.min(ours.steps.len())).find(|&k| ours.steps[k].token != oracle.steps[k].token)
    {
        return fail(format!(
            "our run was fed {} at step {k}, but the oracle's token there is {}: test 3's run is generate \
             --force-tokens with the oracle's tokens",
            ours.steps[k].token, oracle.steps[k].token
        ));
    }
    if ours.steps.len() != n {
        return fail(format!(
            "our run has {} steps (it stopped at {}); the oracle took {n} (it stopped at {})",
            ours.steps.len(),
            ours.stop.describe(),
            if oracle.stopped_at_eog {
                "an end-of-generation token"
            } else {
                "--n"
            }
        ));
    }
    let (flips, free_divergence) = count_flips(ours.steps.iter().map(|s| s.top), &oracle.steps);
    let (floor_flips, floor_free_divergence) =
        count_flips(floor.steps.iter().map(|s| s.top), &oracle.steps);
    PromptResult {
        id: id.into(),
        status: Status::Pass,
        steps: n,
        counted: oracle.steps.iter().filter(|s| gap(s) > TIE_MARGIN).count(),
        flips,
        floor_flips,
        free_divergence,
        floor_free_divergence,
        problem: None,
    }
}

/// Splits `ID=FILE` arguments of the option `flag` (`--run`, `--cold`, ...), refusing one that isn't ID=FILE and an id
/// given twice.
pub fn split_runs<'a>(flag: &str, runs: &'a [String]) -> anyhow::Result<Vec<(&'a str, &'a str)>> {
    let mut out: Vec<(&str, &str)> = Vec::with_capacity(runs.len());
    for r in runs {
        let Some((id, file)) = r.split_once('=') else {
            anyhow::bail!("{flag} {r:?} is not ID=FILE");
        };
        if out.iter().any(|(i, _)| *i == id) {
            anyhow::bail!("{flag} {id} is given twice");
        }
        out.push((id, file));
    }
    Ok(out)
}

/// Test 3's verdict: every prompt compared without a problem, and Artemis's flips, pooled over the prompts, at most
/// FLIP_FACTOR times llama.cpp CPU's. A floor of no flips allows none.
pub fn judge(prompts: Vec<PromptResult>) -> Verdict {
    let mut problems: Vec<String> = prompts
        .iter()
        .filter_map(|p| p.problem.as_ref().map(|e| format!("{}: {e}", p.id)))
        .collect();
    if prompts.is_empty() {
        problems.push("test 3 compares at least one prompt; none given".into());
    }
    let counted = prompts.iter().map(|p| p.counted).sum();
    let flips: usize = prompts.iter().map(|p| p.flips.len()).sum();
    let floor_flips: usize = prompts.iter().map(|p| p.floor_flips.len()).sum();
    let bound = FLIP_FACTOR * floor_flips as f64;
    if flips as f64 > bound {
        problems.push(format!(
            "{flips} flips over {} prompts, more than {FLIP_FACTOR} × llama.cpp CPU's {floor_flips} = {bound}",
            prompts.len()
        ));
    }
    Verdict {
        status: if problems.is_empty() {
            Status::Pass
        } else {
            Status::Fail
        },
        counted,
        flips,
        floor_flips,
        bound,
        prompts,
        problems,
    }
}

/// A free run's first divergence for the console: where, the oracle's gap there, and whether it was a tie.
fn describe(d: &Option<Divergence>) -> String {
    match d {
        None => "never".into(),
        Some(d) => format!(
            "at step {} (gap {:.4}{})",
            d.step,
            d.oracle_gap,
            if d.excused { ", a tie" } else { "" }
        ),
    }
}

impl PromptResult {
    /// One line for the console.
    pub fn line(&self) -> String {
        if let Some(p) = &self.problem {
            return format!("{}: Fail; {p}", self.id);
        }
        let steps: Vec<String> = self.flips.iter().map(|f| f.step.to_string()).collect();
        format!(
            "{}: {} steps, {} counted; flips: ours {}{}, llama.cpp CPU's {}; a free run would first differ: ours {}, \
             llama.cpp CPU's {}",
            self.id,
            self.steps,
            self.counted,
            self.flips.len(),
            if steps.is_empty() {
                String::new()
            } else {
                let s = if steps.len() == 1 { "" } else { "s" };
                format!(" (at step{s} {})", steps.join(", "))
            },
            self.floor_flips.len(),
            describe(&self.free_divergence),
            describe(&self.floor_free_divergence)
        )
    }
}

impl Verdict {
    /// The verdict's line for the console.
    pub fn line(&self) -> String {
        format!(
            "{:?}; {} flips in {} counted steps over {} prompt{}, against a bound of {FLIP_FACTOR} × llama.cpp \
             CPU's {} = {}{}",
            self.status,
            self.flips,
            self.counted,
            self.prompts.len(),
            if self.prompts.len() == 1 { "" } else { "s" },
            self.floor_flips,
            self.bound,
            if self.problems.is_empty() {
                String::new()
            } else {
                format!("; {}", self.problems.join("; "))
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::{CacheStats, Settings, Step, Stop, Timings, FORMAT};

    const EOG: u32 = 248_046;
    const PROMPT: [u32; 3] = [10, 20, 30];

    /// The oracle's run: `tokens` in order, each with a top-two gap of 1.0 except where `gaps` says.
    fn oracle(prompt: usize, tokens: &[u32], gaps: &[(usize, f32)]) -> Oracle {
        Oracle {
            prompt_tokens: prompt,
            stopped_at_eog: tokens.last() == Some(&EOG),
            steps: tokens
                .iter()
                .enumerate()
                .map(|(i, &t)| {
                    let gap = gaps.iter().find(|g| g.0 == i).map_or(1.0, |g| g.1);
                    OracleStep {
                        token: t,
                        logit: 20.0,
                        second: t + 1,
                        second_logit: 20.0 - gap,
                    }
                })
                .collect(),
        }
    }

    /// An engine's top tokens along `tokens`: each token itself, except at the steps in `flips`: token + 100.
    fn tops(tokens: &[u32], flips: &[usize]) -> Vec<u32> {
        let mut t = tokens.to_vec();
        for &i in flips {
            t[i] += 100;
        }
        t
    }

    /// llama.cpp CPU's run along `tokens`, its top token differing at `flips`.
    fn floor(prompt: usize, tokens: &[u32], flips: &[usize]) -> Floor {
        Floor {
            backend: "cpu".into(),
            prompt_tokens: prompt,
            steps: tokens
                .iter()
                .zip(tops(tokens, flips))
                .map(|(&t, top)| FloorStep {
                    token: t,
                    top,
                    logit: 20.0,
                    second: t + 1,
                    second_logit: 19.0,
                    token_logit: if top == t { 20.0 } else { 19.0 },
                })
                .collect(),
        }
    }

    /// Our run fed `tokens` after `prompt`, with --greedy --max-tokens 256, its top token differing at `flips`.
    fn ours(prompt: &[u32], tokens: &[u32], flips: &[usize]) -> Record {
        let stop = if tokens.last() == Some(&EOG) {
            Stop::EndOfGeneration
        } else {
            Stop::MaxTokens
        };
        Record {
            format: FORMAT.into(),
            prompt_tokens: prompt.to_vec(),
            settings: Settings {
                greedy: true,
                temp: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                presence_penalty: 0.0,
                seed: 0,
                max_tokens: STEPS,
                ctx: 65_536,
                cache_slots: 102,
                force_swap_every_token: false,
                all_logits: false,
                cache_slots_asked: None,
                fingerprints: true,
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
            steps: tokens
                .iter()
                .zip(tops(tokens, flips))
                .map(|(&t, top)| Step {
                    token: t,
                    top,
                    logit: 20.0,
                    second: t + 1,
                    second_logit: 19.0,
                })
                .collect(),
            stop,
            positions: Vec::new(),
            timings: Timings::default(),
            cache: CacheStats::default(),
            memory: Default::default(),
            vram: Default::default(),
            prefill: Default::default(),
            warm: None,
            spec: None,
        }
    }

    /// One prompt's comparison: the oracle with `gaps`, the floor flipping at `floor_flips`, ours at `our_flips`.
    fn one(
        tokens: &[u32],
        gaps: &[(usize, f32)],
        floor_flips: &[usize],
        our_flips: &[usize],
    ) -> PromptResult {
        compare(
            "p",
            &PROMPT,
            &oracle(3, tokens, gaps),
            &floor(3, tokens, floor_flips),
            &ours(&PROMPT, tokens, our_flips),
        )
    }

    #[test]
    fn identical_runs_pass() {
        let toks: Vec<u32> = (100..356).collect();
        let p = one(&toks, &[], &[], &[]);
        assert_eq!(
            (p.status, p.steps, p.counted),
            (Status::Pass, 256, 256),
            "{p:?}"
        );
        assert_eq!(
            (p.flips.len(), &p.free_divergence, &p.problem),
            (0, &None, &None)
        );
        let v = judge(vec![p]);
        assert_eq!(
            (v.status, v.flips, v.floor_flips, v.bound),
            (Status::Pass, 0, 0, 0.0),
            "{v:?}"
        );
    }

    #[test]
    fn flips_count_only_where_the_oracle_is_decided() {
        // The oracle's top two are 0.05 apart at step 1 and 0.06 at step 3. Ours differs there and at step 4.
        let p = one(&[5, 6, 7, 8, 9], &[(1, 0.05), (3, 0.06)], &[], &[1, 3, 4]);
        assert_eq!((p.status, p.counted), (Status::Pass, 4), "{p:?}");
        let steps: Vec<usize> = p.flips.iter().map(|f| f.step).collect();
        assert_eq!(steps, [3, 4]);
        assert_eq!((p.flips[0].top, p.flips[0].oracle), (108, 8));
        assert!((p.flips[0].oracle_gap - 0.06).abs() < 1e-6);
        // A free run would have left the oracle's tokens at step 1, where the first rule excused it.
        let d = p.free_divergence.unwrap();
        assert_eq!((d.step, d.top, d.oracle, d.excused), (1, 106, 6, true));
    }

    #[test]
    fn pooled_flips_may_reach_one_and_a_half_times_the_floors() {
        // Two prompts: ours flips 3 times in all, llama.cpp CPU twice, so the bound is 3.
        let toks: Vec<u32> = (100..120).collect();
        let a = one(&toks, &[], &[2], &[1, 5]);
        let b = one(&toks, &[], &[7], &[9]);
        let v = judge(vec![a.clone(), b]);
        assert_eq!(
            (v.status, v.counted, v.flips, v.floor_flips, v.bound),
            (Status::Pass, 40, 3, 2, 3.0),
            "{v:?}"
        );
        // One flip more fails, and the verdict gives the numbers.
        let v = judge(vec![a, one(&toks, &[], &[7], &[9, 10])]);
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems[0].contains("4 flips") && v.problems[0].contains("llama.cpp CPU's 2"),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn a_floor_of_no_flips_allows_none() {
        let v = judge(vec![one(&[5, 6, 7], &[], &[], &[2])]);
        assert_eq!(
            (v.status, v.flips, v.bound),
            (Status::Fail, 1, 0.0),
            "{v:?}"
        );
    }

    #[test]
    fn the_free_running_comparison_is_reported_not_gated() {
        // Ours leaves the oracle's tokens at step 0, where the oracle was sure, so the first rule fails it. But
        // llama.cpp CPU flips there too, so test 3 passes.
        let p = one(&[5, 6, 7], &[], &[0], &[0]);
        let d = p.free_divergence.clone().unwrap();
        assert_eq!((d.step, d.excused), (0, false));
        assert_eq!(p.floor_free_divergence.as_ref().map(|d| d.step), Some(0));
        assert_eq!(judge(vec![p]).status, Status::Pass);
    }

    #[test]
    fn an_immediate_end_of_generation_matches() {
        let p = one(&[EOG], &[], &[], &[]);
        assert_eq!(
            (p.status, p.steps, p.counted),
            (Status::Pass, 1, 1),
            "{p:?}"
        );
        assert!(p.flips.is_empty());
        // Ours would have gone on where the oracle stopped: a flip at the end-of-generation step.
        let p = one(&[EOG], &[], &[], &[0]);
        assert_eq!(p.flips.iter().map(|f| f.oracle).collect::<Vec<_>>(), [EOG]);
    }

    #[test]
    fn a_prompt_that_tokenizes_differently_fails_at_its_first_difference() {
        let (o, f) = (oracle(3, &[5], &[]), floor(3, &[5], &[]));
        let p = compare("03-chat", &PROMPT, &o, &f, &ours(&[10, 21, 30], &[5], &[]));
        assert_eq!(p.status, Status::Fail);
        let e = p.problem.clone().unwrap();
        assert!(
            e.contains("position 1") && e.contains("ours 21") && e.contains("committed 20"),
            "{e}"
        );
        // The pool fails with it, naming the prompt.
        let v = judge(vec![p]);
        assert!(v.problems[0].starts_with("03-chat: "), "{v:?}");
        // A shorter prompt differs where it ends.
        let e = compare("p", &PROMPT, &o, &f, &ours(&[10, 20], &[5], &[]))
            .problem
            .unwrap();
        assert!(
            e.contains("position 2") && e.contains("ours the end"),
            "{e}"
        );
    }

    #[test]
    fn an_expected_output_from_another_prompt_fails() {
        let e = compare(
            "p",
            &PROMPT,
            &oracle(4, &[5], &[]),
            &floor(4, &[5], &[]),
            &ours(&PROMPT, &[5], &[]),
        )
        .problem
        .unwrap();
        assert!(
            e.contains("4-token prompt") && e.contains("regenerate"),
            "{e}"
        );
    }

    #[test]
    fn a_stale_floor_fails_naming_the_mismatch() {
        let toks = [5, 6, 7];
        let (o, r) = (oracle(3, &toks, &[]), ours(&PROMPT, &toks, &[]));
        let problem = |f: &Floor| compare("p", &PROMPT, &o, f, &r).problem.unwrap();
        let e = problem(&floor(4, &toks, &[]));
        assert!(e.contains("floor") && e.contains("4-token prompt"), "{e}");
        let e = problem(&floor(3, &[5, 60, 7], &[]));
        assert!(
            e.contains("floor was fed 60 at step 1") && e.contains("token there is 6"),
            "{e}"
        );
        let e = problem(&floor(3, &[5, 6], &[]));
        assert!(
            e.contains("the floor has 2 steps and the expected output 3:"),
            "{e}"
        );
        let mut f = floor(3, &toks, &[]);
        f.backend = "vulkan".into();
        let e = problem(&f);
        assert!(e.contains("vulkan") && e.contains("CPU"), "{e}");
    }

    #[test]
    fn a_run_not_forced_along_the_oracles_tokens_fails() {
        let toks = [5, 6, 7];
        let (o, f) = (oracle(3, &toks, &[]), floor(3, &toks, &[]));
        // A free greedy run: it took its own token at step 1.
        let e = compare("p", &PROMPT, &o, &f, &ours(&PROMPT, &[5, 60, 7], &[]))
            .problem
            .unwrap();
        assert!(
            e.contains("fed 60 at step 1") && e.contains("--force-tokens"),
            "{e}"
        );
        // A run that stopped early, when the context filled.
        let mut r = ours(&PROMPT, &[5, 6], &[]);
        r.stop = Stop::ContextFull;
        let e = compare("p", &PROMPT, &o, &f, &r).problem.unwrap();
        assert!(
            e.contains("2 steps")
                && e.contains("a full context")
                && e.contains("the oracle took 3"),
            "{e}"
        );
    }

    #[test]
    fn a_run_that_isnt_greedy_with_256_tokens_fails() {
        let (o, f) = (oracle(3, &[5], &[]), floor(3, &[5], &[]));
        let mut r = ours(&PROMPT, &[5], &[]);
        r.settings.greedy = false;
        let e = compare("p", &PROMPT, &o, &f, &r).problem.unwrap();
        assert!(e.contains("--greedy"), "{e}");
        let mut r = ours(&PROMPT, &[5], &[]);
        r.settings.max_tokens = 64;
        let e = compare("p", &PROMPT, &o, &f, &r).problem.unwrap();
        assert!(e.contains("--max-tokens 256") && e.contains("64"), "{e}");
    }

    #[test]
    fn test3_refuses_a_record_of_another_format() {
        let mut r = ours(&PROMPT, &[5, 6], &[]);
        r.format = "artemis-generate-9".into();
        let p = std::env::temp_dir().join(format!("artemis-t3-fmt-{}.json", std::process::id()));
        std::fs::write(&p, serde_json::to_vec(&r).unwrap()).unwrap();
        let e = crate::generate::read_record(&p).unwrap_err().to_string();
        std::fs::remove_file(&p).unwrap();
        assert!(
            e.contains("artemis-t3-fmt-") && e.contains("artemis-generate-9"),
            "{e}"
        );
    }

    #[test]
    fn run_arguments_are_id_equals_file_and_ids_are_unique() {
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            split_runs("--run", &args(&["a=x.json", "b=y.json"])).unwrap(),
            [("a", "x.json"), ("b", "y.json")]
        );
        let e = split_runs("--run", &args(&["a=x.json", "a=y.json"]))
            .unwrap_err()
            .to_string();
        assert_eq!(e, "--run a is given twice");
        let e = split_runs("--path1", &args(&["nofile"]))
            .unwrap_err()
            .to_string();
        assert_eq!(e, "--path1 \"nofile\" is not ID=FILE");
    }

    #[test]
    fn no_prompts_is_a_fail_not_a_pass() {
        let v = judge(vec![]);
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems.iter().any(|p| p.contains("at least one prompt")),
            "{:?}",
            v.problems
        );
    }
}
