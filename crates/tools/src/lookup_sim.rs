//! `artemis lookup-sim`: replays recorded token streams through the prompt-lookup drafter and
//! the acceptance rule, and predicts the accept rate and the decode speed with its cost model. CPU only.
//!
//! A recorded stream is the output for its seed whatever is drafted, so a draft token is accepted exactly when it
//! equals the recorded token at its place; exact up to near-ties, where a verify's `gdn_seq` rows and plain
//! decode's `gdn_step` rows may pick differently. The simulated controller is fed the cost model as its measurements,
//! and each stream starts a fresh controller (at the accept prior), as each of the gate's speed runs is a fresh
//! process; a long-lived server's controller carries its rate across requests, so the prior matters less there.
//! The cost model is a fixed estimate, `c(t)` independent of the context length, unless the `--c1-ms` and
//! `--slope-ms` figures come from `verify-cost`.
use crate::{generate, speed};
use anyhow::{bail, Context, Result};
use clap::Args;
use engine::spec::{next_source_for, score_against, Controller, Drafter, Lookup, Source};
use engine::{SourceChoice, SpecConfig};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The speed bar the six speed records' predicted median is shown beside (done-when 1).
pub const SPEED_BAR_TOK_S: f64 = 110.0;

/// The golden set's `max_tokens` (`expected/*.json` holds 256 steps unless generation stopped earlier).
const GOLDEN_MAX_TOKENS: usize = 256;

/// The cost of a step: `c(t) = c1_ms + (t - 1) * slope_ms`, or, with `--verify-cost`, the measured rows: plain
/// steps cost the `decode` row and a verify step of `t` tokens the `verify` row for `t`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Cost {
    pub c1_ms: f64,
    pub slope_ms: f64,
    /// The plain decode step's measured time; `c1_ms` without it.
    pub decode_ms: Option<f64>,
    /// The verify step's measured time for `t = 2, 3, ...` (index `t - 2`); `c1_ms + (t - 1) * slope_ms` without it.
    pub verify_ms: Option<Vec<f64>>,
}

impl Default for Cost {
    fn default() -> Cost {
        Cost::flat(10.2, 4.4)
    }
}

impl Cost {
    /// The linear cost.
    pub fn flat(c1_ms: f64, slope_ms: f64) -> Cost {
        Cost {
            c1_ms,
            slope_ms,
            decode_ms: None,
            verify_ms: None,
        }
    }

    /// The time of a plain decode step, in ms.
    pub fn plain_step_ms(&self) -> f64 {
        self.decode_ms.unwrap_or(self.c1_ms)
    }

    /// The time of a step of `t` tokens, in ms.
    pub fn step_ms(&self, t: usize) -> f64 {
        if t <= 1 {
            return self.plain_step_ms();
        }
        if let Some(rows) = self.verify_ms.as_deref().filter(|r| !r.is_empty()) {
            return match rows.get(t - 2) {
                Some(&ms) => ms,
                // Past the last row (t = rows.len() + 1): the slope carries on.
                None => rows[rows.len() - 1] + (t - 1 - rows.len()) as f64 * self.slope_ms,
            };
        }
        self.c1_ms + (t - 1) as f64 * self.slope_ms
    }

    /// The decode and verify rows of `verify-cost`'s `--verify` JSON at context `ctx`. Verify sizes the file lacks (it
    /// has 2, 3, 4, 5, 8 and 16) are interpolated linearly. The slope is the least-squares slope of the verify rows
    /// from t = 2 to 5, the range the step loop uses.
    pub fn from_verify_cost(doc: &serde_json::Value, ctx: u64) -> Result<Cost> {
        let records = doc["records"]
            .as_array()
            .context("verify-cost JSON: no records")?;
        let row = |path: &str, t: u64| {
            records.iter().find_map(|r| {
                (r["path"] == path && r["ctx"] == ctx && r["t"] == t)
                    .then(|| r["mean_ms"].as_f64())
                    .flatten()
            })
        };
        let decode = row("decode", 1);
        let Some(decode) = decode else {
            let have: Vec<u64> = records.iter().filter_map(|r| r["ctx"].as_u64()).collect();
            bail!("verify-cost JSON: no decode row at ctx {ctx} (it has ctx {have:?})");
        };
        let mut points: Vec<(usize, f64)> = (2..=16u64)
            .filter_map(|t| row("verify", t).map(|ms| (t as usize, ms)))
            .collect();
        points.sort_by_key(|p| p.0);
        if points.len() < 2 {
            bail!("verify-cost JSON: fewer than two verify rows at ctx {ctx}");
        }
        if points[0].0 != 2 {
            bail!("verify-cost JSON: no verify row for t = 2 at ctx {ctx}");
        }
        let max_t = points.last().map_or(2, |p| p.0);
        let mut rows = Vec::new();
        for t in 2..=max_t {
            let ms = match points.iter().find(|p| p.0 == t) {
                Some(p) => p.1,
                None => {
                    let lo = points
                        .iter()
                        .rev()
                        .find(|p| p.0 < t)
                        .copied()
                        .unwrap_or(points[0]);
                    let hi = points.iter().find(|p| p.0 > t).copied().unwrap_or(lo);
                    if hi.0 == lo.0 {
                        lo.1
                    } else {
                        lo.1 + (hi.1 - lo.1) * (t - lo.0) as f64 / (hi.0 - lo.0) as f64
                    }
                }
            };
            rows.push(ms);
        }
        let near: Vec<(f64, f64)> = points
            .iter()
            .filter(|p| p.0 <= 5)
            .map(|p| (p.0 as f64, p.1))
            .collect();
        let slope = if near.len() >= 2 {
            let n = near.len() as f64;
            let (mt, mm) = (
                near.iter().map(|p| p.0).sum::<f64>() / n,
                near.iter().map(|p| p.1).sum::<f64>() / n,
            );
            let spread: f64 = near.iter().map(|p| (p.0 - mt).powi(2)).sum();
            let cross: f64 = near.iter().map(|p| (p.0 - mt) * (p.1 - mm)).sum();
            (cross / spread).max(0.0)
        } else {
            0.0
        };
        Ok(Cost {
            c1_ms: decode,
            slope_ms: slope,
            decode_ms: Some(decode),
            verify_ms: Some(rows),
        })
    }
}

/// What one stream's replay came to.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SimResult {
    /// Tokens the decode steps emitted: the stream's, less the first, which came from prefill.
    pub tokens: usize,
    pub verify_steps: usize,
    pub plain_steps: usize,
    /// Draft tokens verified, over all verify steps.
    pub drafted: usize,
    /// Of those, the ones that matched.
    pub accepted: usize,
    /// `drafted / verify_steps`; 0 without a verify step.
    pub mean_draft: f64,
    pub tokens_per_step: f64,
    /// The steps' time in the cost model.
    pub predicted_ms: f64,
    /// `tokens` plain steps' time: the same stream without speculation.
    pub plain_ms: f64,
    /// Of `verify_steps`, the steps whose draft was MTP's, with their drafted and accepted tokens (in `drafted` and
    /// `accepted` too) and how many were probes.
    pub mtp_steps: usize,
    pub mtp_drafted: usize,
    pub mtp_accepted: usize,
    pub mtp_probes: usize,
    /// Lookup's verified draft tokens and the kept ones, by `LookupMatch::class` (a draft with no class counts as 0).
    pub lookup_class_drafted: [usize; engine::spec::LookupMatch::CLASSES],
    pub lookup_class_accepted: [usize; engine::spec::LookupMatch::CLASSES],
}

impl SimResult {
    /// `accepted / drafted`; 0 without drafts.
    pub fn accept_rate(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }

    /// `tokens / predicted_ms * 1000`.
    pub fn predicted_tok_s(&self) -> f64 {
        rate(self.tokens, self.predicted_ms)
    }

    pub fn plain_tok_s(&self) -> f64 {
        rate(self.tokens, self.plain_ms)
    }
}

fn rate(tokens: usize, ms: f64) -> f64 {
    if ms > 0.0 {
        tokens as f64 / ms * 1000.0
    } else {
        0.0
    }
}

/// Replays `stream`, the tokens emitted after `prompt` (the first came from prefill), through the lookup drafter and
/// the draft-length rule: `cfg.fixed_draft` if set, else the controller fed `cost` as its measurements. A step drafts
/// from the history ending at `x`, the last committed token, verifies up to `k` drafts against the stream, and emits
/// the matching ones and one more. `k` is capped so the step ends within the stream, as the turn caps it at
/// `max_tokens`. Each call starts a fresh controller. The end of `stream` is taken as the length cap; see
/// `simulate_capped` for a stream that a stop (the end-of-generation token) cut short.
pub fn simulate(prompt: &[u32], stream: &[u32], cfg: &SpecConfig, cost: &Cost) -> SimResult {
    simulate_capped(prompt, stream, stream.len(), cfg, cost)
}

/// `simulate` for a request whose `max_tokens` is `max_tokens` (at least `stream.len()`). A stream shorter than that
/// ended at a stop: the turn still caps `k` at `max_tokens`, so a verify near the end can carry drafts past the stop
/// (and pays for them), and the stop cuts the burst. The drafts past the stop are never compared: they count as
/// drafted, not as rejected.
pub fn simulate_capped(
    prompt: &[u32],
    stream: &[u32],
    max_tokens: usize,
    cfg: &SpecConfig,
    cost: &Cost,
) -> SimResult {
    run_sim(
        prompt,
        stream,
        max_tokens,
        cfg,
        cost,
        None,
        None,
        Variant::default(),
    )
}

/// MTP's measured behaviour, the simulator's model of it.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpModel {
    /// `a_i`: the chance that draft `i` is kept given drafts `1 .. i - 1` were; a depth past the list takes its last
    /// value.
    pub depth_rates: Vec<f64>,
    /// The draft pass's time per draft token, in ms.
    pub ms_per_draft: f64,
    /// The draft pass's fixed time, in ms: `d(k) = ms_per_pass + k * ms_per_draft`.
    pub ms_per_pass: f64,
}

impl MtpModel {
    fn rate(&self, depth: usize) -> f64 {
        self.depth_rates
            .get(depth)
            .or(self.depth_rates.last())
            .copied()
            .unwrap_or(0.0)
    }

    /// The draft pass for `k` drafts, in ms.
    pub fn draft_ms(&self, k: usize) -> f64 {
        self.ms_per_pass + k as f64 * self.ms_per_draft
    }
}

/// splitmix64: the draws behind an MTP step's accepted count.
struct SplitMix(u64);

impl SplitMix {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `simulate_capped` with MTP as a second source. The engine's own `Controller` runs fresh per stream, with `cfg`'s
/// `mtp_accept_prior`, `probe_every` and `max_draft`, fed `cost`'s times and `d(k) = ms_per_pass + k * ms_per_draft` as
/// its measurements. Lookup drafts as `simulate_capped` does. `engine::spec::next_source` decides, as the step loop
/// does; a step where it gives MTP the draft draws its accepted count depth by depth from `mtp.depth_rates` (splitmix64
/// from `seed`). MTP's draft tokens are not in the recording, so a step that takes them emits the recorded tokens.
pub fn simulate_with_mtp(
    prompt: &[u32],
    stream: &[u32],
    cfg: &SpecConfig,
    cost: &Cost,
    mtp: &MtpModel,
    seed: u64,
) -> SimResult {
    simulate_with_mtp_capped(prompt, stream, stream.len(), cfg, cost, mtp, seed)
}

/// `simulate_with_mtp` for a request whose `max_tokens` is `max_tokens`, as `simulate_capped`.
pub fn simulate_with_mtp_capped(
    prompt: &[u32],
    stream: &[u32],
    max_tokens: usize,
    cfg: &SpecConfig,
    cost: &Cost,
    mtp: &MtpModel,
    seed: u64,
) -> SimResult {
    run_sim(
        prompt,
        stream,
        max_tokens,
        cfg,
        cost,
        Some((mtp, SplitMix(seed))),
        None,
        Variant::default(),
    )
}

/// How a simulation departs from the step loop, to measure what a controller change is worth. The default is the step
/// loop as it is (`Variant::STEP_LOOP`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Variant {
    /// Lookup's drafts carry no class, so the controller scores them all with its one pooled rate (before the per-class
    /// rates).
    pub pooled_lookup: bool,
    /// On a step MTP drafted, lookup's unverified draft is scored on its first token only (before whole-draft scoring).
    pub first_token_only: bool,
    /// An MTP step keeps at least as many drafts as lookup's draft, on the stream, would have: MTP is never worse than
    /// lookup where lookup has a draft. Draws are independent of the stream otherwise, which credits a lookup draft
    /// with every step MTP would have got wrong; this bounds that from the other side.
    pub mtp_follows_lookup: bool,
}

impl Variant {
    /// The step loop before per-class lookup rates and whole-draft scoring.
    pub const BEFORE: Variant = Variant {
        pooled_lookup: true,
        first_token_only: true,
        mtp_follows_lookup: false,
    };
    /// Per-class lookup rates and whole-draft scoring.
    pub const AFTER: Variant = Variant {
        pooled_lookup: false,
        first_token_only: false,
        mtp_follows_lookup: false,
    };
    /// What `SpecState::step` does.
    pub const STEP_LOOP: Variant = Variant::AFTER;
}

impl Default for Variant {
    fn default() -> Variant {
        Variant::STEP_LOOP
    }
}

/// `simulate_with_mtp_capped` (or `simulate_capped` without `mtp`) as `variant` says.
pub fn simulate_variant(
    prompt: &[u32],
    stream: &[u32],
    max_tokens: usize,
    cfg: &SpecConfig,
    cost: &Cost,
    mtp: Option<(&MtpModel, u64)>,
    variant: Variant,
) -> SimResult {
    run_sim(
        prompt,
        stream,
        max_tokens,
        cfg,
        cost,
        mtp.map(|(m, seed)| (m, SplitMix(seed))),
        None,
        variant,
    )
}

/// `simulate_with_mtp`, and each step's `(source, k, probe)`: what the step loop must choose too (tests).
#[cfg(test)]
fn simulate_with_mtp_traced(
    prompt: &[u32],
    stream: &[u32],
    cfg: &SpecConfig,
    cost: &Cost,
    mtp: &MtpModel,
    seed: u64,
) -> (SimResult, Vec<(Source, usize, bool)>) {
    let mut trace = Vec::new();
    let r = run_sim(
        prompt,
        stream,
        stream.len(),
        cfg,
        cost,
        Some((mtp, SplitMix(seed))),
        Some(&mut trace),
        Variant::default(),
    );
    (r, trace)
}

#[allow(clippy::too_many_arguments)] // the replay's inputs, spelled out
fn run_sim(
    prompt: &[u32],
    stream: &[u32],
    max_tokens: usize,
    cfg: &SpecConfig,
    cost: &Cost,
    mut mtp: Option<(&MtpModel, SplitMix)>,
    mut trace: Option<&mut Vec<(Source, usize, bool)>>,
    variant: Variant,
) -> SimResult {
    let max_tokens = max_tokens.max(stream.len());
    let mut lookup = Lookup::new(cfg.ngram_min, cfg.ngram_max);
    let mut controller = Controller::new(cfg);
    let plain_ms = cost.plain_step_ms();
    let mut r = SimResult {
        tokens: stream.len().saturating_sub(1),
        verify_steps: 0,
        plain_steps: 0,
        drafted: 0,
        accepted: 0,
        mean_draft: 0.0,
        tokens_per_step: 0.0,
        predicted_ms: 0.0,
        plain_ms: stream.len().saturating_sub(1) as f64 * plain_ms,
        mtp_steps: 0,
        mtp_drafted: 0,
        mtp_accepted: 0,
        mtp_probes: 0,
        lookup_class_drafted: [0; engine::spec::LookupMatch::CLASSES],
        lookup_class_accepted: [0; engine::spec::LookupMatch::CLASSES],
    };
    let Some(&first) = stream.first() else {
        return r;
    };
    lookup.observe(prompt);
    lookup.observe(&[first]);
    // `at` is the stream index of `x`.
    let mut at = 0;
    while at + 1 < stream.len() {
        let left = stream.len() - 1 - at;
        // The turn caps `k` at `max_tokens - completion - 1`, with `at + 1` tokens emitted. No room, no draft.
        let k_max = max_tokens.saturating_sub(at + 2);
        let cap = cfg.max_draft.min(k_max);
        let draft = if cfg.has(Source::Lookup) {
            lookup.draft(cap)
        } else {
            Vec::new()
        };
        // The draft's class as the lookup reports it; the controller sees it unless the variant pools.
        let seen_class = lookup.last_match().filter(|_| !draft.is_empty());
        let class = seen_class.filter(|_| !variant.pooled_lookup);
        let (source, k, probe) =
            next_source_for(&controller, cfg, draft.len(), class, mtp.is_some(), cap);
        if let Some(t) = trace.as_deref_mut() {
            t.push((source, k, probe));
        }
        let next = &stream[at + 1..];
        if k == 0 {
            r.plain_steps += 1;
            r.predicted_ms += plain_ms;
            controller.observe_cost(1, plain_ms);
            controller.observe_plain();
            // Lookup scores itself for free at k = 0: its first draft token against the token picked.
            if let Some(&d) = draft.first() {
                controller.observe_accept_as(Source::Lookup, class, 1, usize::from(d == next[0]));
            }
            lookup.observe(&next[..1]);
            at += 1;
            continue;
        }
        // The stream's last token is the pick of the step that reaches it (a stop ends the burst there), so drafts
        // never accept past `left - 1`; a draft cut off by the stop was never compared.
        let (accepted, draft_ms) = match (source, mtp.as_mut()) {
            (Source::Mtp, Some((model, rng))) => {
                let mut kept = 0;
                while kept < k && rng.next_f64() < model.rate(kept) {
                    kept += 1;
                }
                if variant.mtp_follows_lookup {
                    let lookup_kept = draft.iter().zip(next).take_while(|(d, t)| d == t).count();
                    kept = kept.max(lookup_kept.min(k));
                }
                (kept.min(left - 1), model.draft_ms(k))
            }
            _ => (
                draft[..k]
                    .iter()
                    .zip(next)
                    .take_while(|(d, t)| d == t)
                    .count()
                    .min(left - 1),
                0.0,
            ),
        };
        let rejected = usize::from(accepted < k && accepted < left - 1);
        let step_ms = cost.step_ms(k + 1);
        r.verify_steps += 1;
        r.drafted += k;
        r.accepted += accepted;
        r.predicted_ms += draft_ms + step_ms;
        controller.observe_cost(k + 1, step_ms);
        match source {
            Source::Lookup => {
                let c = seen_class.map_or(0, |m| m.class());
                r.lookup_class_drafted[c] += k;
                r.lookup_class_accepted[c] += accepted;
                controller.observe_accept_as(Source::Lookup, class, accepted + rejected, accepted)
            }
            Source::Mtp => {
                r.mtp_steps += 1;
                r.mtp_drafted += k;
                r.mtp_accepted += accepted;
                r.mtp_probes += usize::from(probe);
                controller.observe_draft_cost(k, draft_ms);
                controller.observe_mtp_drafted();
                controller.observe_accept(Source::Mtp, accepted + rejected, accepted);
                // Lookup scores its draft against the tokens the step picked, as the step loop does.
                if let Some(&d) = draft.first() {
                    let (tested, hit) = if variant.first_token_only {
                        (1, usize::from(d == next[0]))
                    } else {
                        score_against(&draft, &next[..accepted + 1])
                    };
                    if tested > 0 {
                        controller.observe_accept_as(Source::Lookup, class, tested, hit);
                    }
                }
            }
        }
        // The accepted drafts, then the token the verify picked after them.
        lookup.observe(&next[..accepted + 1]);
        at += accepted + 1;
    }
    if r.verify_steps > 0 {
        r.mean_draft = r.drafted as f64 / r.verify_steps as f64;
    }
    r.tokens_per_step = r.tokens as f64 / (r.verify_steps + r.plain_steps).max(1) as f64;
    r
}

/// Lookup's drafts of one class, replayed with lookup verifying every draft it has (`bucket_rates`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct BucketCount {
    /// Drafts made.
    pub drafts: usize,
    /// Their tokens.
    pub drafted: usize,
    /// Tokens kept.
    pub accepted: usize,
    /// Tokens compared: the kept ones and at most one rejection a draft, as the controller counts them.
    pub tested: usize,
    /// Drafts whose first token was kept.
    pub first_kept: usize,
    /// Drafts kept whole at the full length (`max_draft` tokens).
    pub full: usize,
}

impl BucketCount {
    fn add(&mut self, o: &BucketCount) {
        self.drafts += o.drafts;
        self.drafted += o.drafted;
        self.accepted += o.accepted;
        self.tested += o.tested;
        self.first_kept += o.first_kept;
        self.full += o.full;
    }
}

/// `bucket_rates`' result: lookup's drafts by `LookupMatch::class` (0: a 2-token match, 1: 3 tokens, 2: 4 or more,
/// 3: a continuing run), by match length, and by whether the previous draft was kept whole at the full length (the
/// proxy's split: `[not, kept whole]`).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct BucketReport {
    pub by_class: [BucketCount; engine::spec::LookupMatch::CLASSES],
    pub by_n: BTreeMap<usize, BucketCount>,
    pub after_full: [BucketCount; 2],
    /// Tokens emitted after the first, steps and plain steps of the replay.
    pub tokens: usize,
    pub steps: usize,
    pub plain: usize,
}

impl BucketReport {
    pub fn add(&mut self, o: &BucketReport) {
        for (a, b) in self.by_class.iter_mut().zip(&o.by_class) {
            a.add(b);
        }
        for (n, c) in &o.by_n {
            self.by_n.entry(*n).or_default().add(c);
        }
        for (a, b) in self.after_full.iter_mut().zip(&o.after_full) {
            a.add(b);
        }
        self.tokens += o.tokens;
        self.steps += o.steps;
        self.plain += o.plain;
    }
}

/// Replays `stream` after `prompt` with lookup verifying every draft it has (up to `max_draft` tokens, cut to the
/// stream as `simulate_capped` cuts it) and no controller, and counts the drafts by what produced them. The rates are
/// the drafts' own, whichever source the controller would have chosen.
pub fn bucket_rates(
    prompt: &[u32],
    stream: &[u32],
    max_draft: usize,
    ngram_min: usize,
    ngram_max: usize,
) -> BucketReport {
    let mut r = BucketReport {
        tokens: stream.len().saturating_sub(1),
        ..BucketReport::default()
    };
    let Some(&first) = stream.first() else {
        return r;
    };
    let mut lookup = Lookup::new(ngram_min, ngram_max);
    lookup.observe(prompt);
    lookup.observe(&[first]);
    let (mut at, mut prev_full) = (0, false);
    while at + 1 < stream.len() {
        let left = stream.len() - 1 - at;
        let cap = max_draft.min(stream.len().saturating_sub(at + 2));
        let draft = lookup.draft(cap);
        let next = &stream[at + 1..];
        r.steps += 1;
        let Some(m) = lookup.last_match().filter(|_| !draft.is_empty()) else {
            r.plain += 1;
            prev_full = false;
            lookup.observe(&next[..1]);
            at += 1;
            continue;
        };
        let accepted = draft
            .iter()
            .zip(next)
            .take_while(|(d, t)| d == t)
            .count()
            .min(left - 1);
        let rejected = usize::from(accepted < draft.len() && accepted < left - 1);
        let full = draft.len() == max_draft && accepted == max_draft;
        let count = BucketCount {
            drafts: 1,
            drafted: draft.len(),
            accepted,
            tested: accepted + rejected,
            first_kept: usize::from(draft[0] == next[0]),
            full: usize::from(full),
        };
        r.by_class[m.class()].add(&count);
        r.by_n.entry(m.n).or_default().add(&count);
        r.after_full[usize::from(prev_full)].add(&count);
        prev_full = full;
        lookup.observe(&next[..accepted + 1]);
        at += accepted + 1;
    }
    r
}

/// Each measured run's predicted figure: its measured "on" tok/s times (the MTP simulation's rate over the lookup-only
/// simulation's), so the harness's bias cancels; and the lookup-only simulation's error against the measured "on",
/// `(simulated - measured) / measured`. `sims[i]` is `(MTP simulation, lookup-only simulation)` for run `i`. A run
/// whose lookup-only simulation has no rate predicts 0 with error 0.
pub fn predict_runs(sims: &[(SimResult, SimResult)], measured_on: &[f64]) -> Vec<(f64, f64)> {
    sims.iter()
        .zip(measured_on)
        .map(|((with_mtp, lookup_only), &measured)| {
            let base = lookup_only.predicted_tok_s();
            if base <= 0.0 || measured <= 0.0 {
                return (0.0, 0.0);
            }
            (
                measured * with_mtp.predicted_tok_s() / base,
                (base - measured) / measured,
            )
        })
        .collect()
}

/// Several seeds' results as one: the counts rounded to means, the time the mean time. `rs` is not empty.
fn mean_result(rs: &[SimResult]) -> SimResult {
    let n = rs.len() as f64;
    let avg = |f: fn(&SimResult) -> usize| {
        (rs.iter().map(|r| f(r) as f64).sum::<f64>() / n).round() as usize
    };
    let mut out = rs[0].clone();
    out.verify_steps = avg(|r| r.verify_steps);
    out.plain_steps = avg(|r| r.plain_steps);
    out.drafted = avg(|r| r.drafted);
    out.accepted = avg(|r| r.accepted);
    out.mtp_steps = avg(|r| r.mtp_steps);
    out.mtp_drafted = avg(|r| r.mtp_drafted);
    out.mtp_accepted = avg(|r| r.mtp_accepted);
    out.mtp_probes = avg(|r| r.mtp_probes);
    for c in 0..engine::spec::LookupMatch::CLASSES {
        out.lookup_class_drafted[c] = (rs
            .iter()
            .map(|r| r.lookup_class_drafted[c] as f64)
            .sum::<f64>()
            / n)
            .round() as usize;
        out.lookup_class_accepted[c] = (rs
            .iter()
            .map(|r| r.lookup_class_accepted[c] as f64)
            .sum::<f64>()
            / n)
            .round() as usize;
    }
    out.predicted_ms = rs.iter().map(|r| r.predicted_ms).sum::<f64>() / n;
    out.mean_draft = if out.verify_steps > 0 {
        out.drafted as f64 / out.verify_steps as f64
    } else {
        0.0
    };
    out.tokens_per_step = out.tokens as f64 / (out.verify_steps + out.plain_steps).max(1) as f64;
    out
}

/// One recorded stream.
pub struct Stream {
    pub id: String,
    /// The request's `max_tokens`; above `tokens.len()` when a stop ended the stream.
    pub max_tokens: usize,
    pub prompt: Vec<u32>,
    pub tokens: Vec<u32>,
}

use crate::generate::read_tokens;

/// The golden set: `expected/<id>.json`'s `steps[].token` and `prompts/<id>.tokens`, sorted by id. A prompt whose
/// length is not the expected file's `prompt_tokens` is refused.
pub fn load_golden(dir: &Path) -> Result<Vec<Stream>> {
    let expected = dir.join("expected");
    let mut ids: Vec<String> = std::fs::read_dir(&expected)
        .with_context(|| format!("reading {}", expected.display()))?
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()? == "json").then(|| p.file_stem()?.to_str().map(str::to_string))?
        })
        .collect();
    ids.sort();
    if ids.is_empty() {
        bail!("{} holds no expected/*.json", dir.display());
    }
    ids.into_iter()
        .map(|id| {
            let path = expected.join(format!("{id}.json"));
            let v: serde_json::Value = serde_json::from_slice(
                &std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
            )
            .with_context(|| format!("parsing {}", path.display()))?;
            let tokens = v["steps"]
                .as_array()
                .with_context(|| format!("{}: no steps", path.display()))?
                .iter()
                .map(|s| {
                    s["token"]
                        .as_u64()
                        .and_then(|t| u32::try_from(t).ok())
                        .with_context(|| format!("{}: a step has no token", path.display()))
                })
                .collect::<Result<Vec<u32>>>()?;
            let prompt = read_tokens(&dir.join("prompts").join(format!("{id}.tokens")))?;
            let count = v["prompt_tokens"].as_u64();
            if count != Some(prompt.len() as u64) {
                bail!(
                    "{id}: prompts/{id}.tokens has {} tokens; {} says prompt_tokens is {}",
                    prompt.len(),
                    path.display(),
                    count.map_or("missing".to_string(), |c| c.to_string())
                );
            }
            // The golden runs were `--max-tokens 256`; one that did not stop at the end-of-generation token has 256.
            let eog = v["stopped_at_eog"].as_bool().unwrap_or(false);
            let max_tokens = if eog { GOLDEN_MAX_TOKENS } else { tokens.len() };
            Ok(Stream {
                id,
                max_tokens,
                prompt,
                tokens,
            })
        })
        .collect()
}

/// A `generate --json` record as a stream named `id`.
pub fn load_record(id: &str, path: &Path) -> Result<Stream> {
    let rec = generate::read_record(path)?;
    Ok(Stream {
        id: id.to_string(),
        max_tokens: rec.settings.max_tokens,
        prompt: rec.prompt_tokens,
        tokens: rec.steps.iter().map(|s| s.token).collect(),
    })
}

#[derive(Args, Clone, Debug)]
pub struct LookupSimArgs {
    /// testdata/golden: replays every `expected/<id>.json` with its `prompts/<id>.tokens`.
    #[arg(long)]
    golden: Option<PathBuf>,
    /// ID=FILE: replays a `generate --json` record as stream ID. The speed streams: six records.
    #[arg(long = "record")]
    records: Vec<String>,
    /// Simulates a fixed draft length instead of the controller (lifts max_draft to N when N is above 4, as
    /// `artemis generate --draft-len N` does, unless it is given a smaller --max-draft).
    #[arg(long)]
    draft_len: Option<usize>,
    /// The controller's max_draft (default: SpecConfig's).
    #[arg(long)]
    max_draft: Option<usize>,
    /// The accept rate each stream's fresh controller starts from, 0 to 1; a table at 0 is printed too
    /// (default: SpecConfig's).
    #[arg(long)]
    accept_prior: Option<f64>,
    /// Which source drafts when both have a draft: `priority` or `best` (default: SpecConfig's, `priority`). Only the
    /// MTP predictions use it.
    #[arg(long)]
    source_choice: Option<String>,
    /// c(1), the plain step's time in ms. The default is the model's estimate; pass `verify-cost`'s figure.
    #[arg(long, default_value_t = Cost::default().c1_ms)]
    c1_ms: f64,
    /// The cost per extra token of a step, in ms. The default is the model's estimate; pass `verify-cost`'s slope.
    #[arg(long, default_value_t = Cost::default().slope_ms)]
    slope_ms: f64,
    /// Also writes the results as JSON.
    #[arg(long)]
    json: Option<PathBuf>,
    /// MTP's accept rate at draft depths 1, 2, ... (A1,A2,A3,A4 from `mtp_probe.py`'s pooled depth rates): predicts the
    /// six `--record` streams with MTP as a second source. Needs --measured.
    #[arg(long, value_delimiter = ',')]
    mtp_rates: Vec<f64>,
    /// The MTP draft pass's time per draft token, in ms. The prediction is also printed at 1.2 and 1.8; the checkpoint
    /// is judged on 1.8.
    #[arg(long, default_value_t = MTP_MS_PER_DRAFT)]
    mtp_ms_per_draft: f64,
    /// The MTP draft pass's fixed time, in ms.
    #[arg(long, default_value_t = MTP_MS_PER_PASS)]
    mtp_ms_per_pass: f64,
    /// `verify-cost --verify`'s JSON: plain steps cost its decode row and verify steps its verify rows, at --cost-ctx.
    /// Replaces --c1-ms and --slope-ms.
    #[arg(long)]
    verify_cost: Option<PathBuf>,
    /// The context of --verify-cost's rows (the speed runs sit at or below 2,048).
    #[arg(long, default_value_t = 2048)]
    cost_ctx: u64,
    /// The measured "on" figures: the `serve-speed-spec` aggregate record in `results.jsonl`, its `runs[]` of mode "on"
    /// and their `decode_steps_tok_s` (the figures the gate judged). A file with several aggregates is refused unless
    /// --measured-commit picks one.
    #[arg(long)]
    measured: Option<PathBuf>,
    /// The `commit` of the `serve-speed-spec` aggregate to read from --measured.
    #[arg(long)]
    measured_commit: Option<String>,
    /// `mtp_probe.py`'s output: its `depth1_per_prompt` gives the sensitivity line (each stream predicted with its own
    /// prompt's depth-1 rate, a2.. scaled by that prompt's a1 over the pooled a1, capped at 1).
    #[arg(long)]
    mtp_probe: Option<PathBuf>,
    /// MTP's draws per stream: the prediction is the mean time over this many seeds.
    #[arg(long, default_value_t = 16)]
    seeds: u64,
    /// Prints lookup's accept rates by class of match (2, 3, 4 tokens, a continuing run) and by whether the previous
    /// draft was kept whole, with lookup verifying every draft it has and no controller; then exits.
    #[arg(long)]
    buckets: bool,
    /// Compares the controller before per-class lookup rates and whole-draft scoring with after, and `ngram_min` 3 and
    /// 4, on the `--record` streams with MTP modelled from `--mtp-rates-for` (else `--mtp-rates`); with --measured,
    /// scales each run's measured "on" figure by the simulated ratio. Then exits.
    #[arg(long)]
    variants: bool,
    /// PROMPT=A1,A2,..: MTP's depth rates for the `--record` streams whose id starts with PROMPT and a dash
    /// (`write=0.89,0.85` for `write-1234`).
    #[arg(long = "mtp-rates-for")]
    mtp_rates_for: Vec<String>,
}

/// The breakdown of the estimate: eight HOST experts, the head and the block's rest, per draft token.
pub const MTP_MS_PER_DRAFT: f64 = 1.8;
/// MTP must gain at least this over lookup alone, on the median of six, to go on.
const CHECKPOINT_MIN_GAIN: f64 = 0.03;
/// The draft pass's fixed time.
pub const MTP_MS_PER_PASS: f64 = 0.3;
/// The other `ms_per_draft` the prediction is printed at (the low end of that estimate).
const MTP_MS_PER_DRAFT_LOW: f64 = 1.2;

/// One simulation setting's results over all the streams.
#[derive(Serialize)]
struct Table {
    accept_prior: f64,
    draft_len: Option<usize>,
    rows: Vec<Row>,
    /// The median predicted tok/s over the `--record` streams; `None` without any.
    record_median_tok_s: Option<f64>,
}

#[derive(Serialize)]
struct Row {
    id: String,
    source: &'static str,
    prompt_tokens: usize,
    #[serde(flatten)]
    result: SimResult,
    accept_rate: f64,
    predicted_tok_s: f64,
    plain_tok_s: f64,
}

fn run_table(streams: &[(Stream, &'static str)], cfg: &SpecConfig, cost: &Cost) -> Table {
    let rows: Vec<Row> = streams
        .iter()
        .map(|(s, source)| {
            let result = simulate_capped(&s.prompt, &s.tokens, s.max_tokens, cfg, cost);
            Row {
                id: s.id.clone(),
                source,
                prompt_tokens: s.prompt.len(),
                accept_rate: result.accept_rate(),
                predicted_tok_s: result.predicted_tok_s(),
                plain_tok_s: result.plain_tok_s(),
                result,
            }
        })
        .collect();
    let rec: Vec<f64> = rows
        .iter()
        .filter(|r| r.source == "record")
        .map(|r| r.predicted_tok_s)
        .collect();
    Table {
        accept_prior: cfg.accept_prior,
        draft_len: cfg.fixed_draft,
        record_median_tok_s: (!rec.is_empty()).then(|| speed::median(&rec)),
        rows,
    }
}

fn print_table(t: &Table) {
    match t.draft_len {
        Some(n) => println!("\n### fixed draft length {n}\n"),
        None => println!("\n### controller, accept prior {}\n", t.accept_prior),
    }
    println!(
        "| stream | prompt | tokens | verify | plain | drafted | accepted | accept % | mean draft | tok/step | \
         predicted tok/s | plain tok/s |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for r in &t.rows {
        let s = &r.result;
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {:.1} | {:.2} | {:.3} | {:.1} | {:.1} |",
            r.id,
            r.prompt_tokens,
            s.tokens,
            s.verify_steps,
            s.plain_steps,
            s.drafted,
            s.accepted,
            r.accept_rate * 100.0,
            s.mean_draft,
            s.tokens_per_step,
            r.predicted_tok_s,
            r.plain_tok_s
        );
    }
    let all = t.rows.iter().map(|r| &r.result);
    let (tokens, ms, plain_ms) = all.fold((0, 0.0, 0.0), |a, s| {
        (a.0 + s.tokens, a.1 + s.predicted_ms, a.2 + s.plain_ms)
    });
    println!(
        "\nAll streams together: {} tokens, {:.1} tok/s predicted, {:.1} plain.",
        tokens,
        rate(tokens, ms),
        rate(tokens, plain_ms)
    );
    if let Some(m) = t.record_median_tok_s {
        let n = t.rows.iter().filter(|r| r.source == "record").count();
        println!(
            "Median of {n} record streams: {m:.1} tok/s predicted, against {SPEED_BAR_TOK_S:.1} ({}).",
            if m >= SPEED_BAR_TOK_S {
                "meets the bar"
            } else {
                "below the bar"
            }
        );
    }
}

/// The simulated config: the defaults, the cost model's slope as the controller's first guess, the flags' prior and
/// fixed length. A fixed length above the controller's `max_draft` would be refused, so it lifts `max_draft` to match.
fn config(a: &LookupSimArgs, cost: &Cost) -> Result<SpecConfig> {
    let base = SpecConfig::default();
    if let Some(p) = a.accept_prior {
        if !(0.0..=1.0).contains(&p) {
            bail!("--accept-prior must be from 0 to 1");
        }
    }
    let source_choice = match &a.source_choice {
        None => base.source_choice,
        Some(name) => SourceChoice::parse(name).ok_or_else(|| {
            anyhow::anyhow!(
                "--source-choice: an unknown choice {name:?}, expected priority or best"
            )
        })?,
    };
    let cfg = SpecConfig {
        cost_slope_ms: cost.slope_ms,
        source_choice,
        accept_prior: a.accept_prior.unwrap_or(base.accept_prior),
        max_draft: a
            .max_draft
            .unwrap_or(base.max_draft)
            .max(a.draft_len.unwrap_or(0)),
        fixed_draft: a.draft_len,
        ..base
    };
    cfg.check()
        .map_err(|(key, why)| anyhow::anyhow!("{key}: {why}"))?;
    Ok(cfg)
}

/// The measured "on" tok/s per run, keyed `write-1234`, from the `serve-speed-spec` aggregate record in
/// `results.jsonl` (the figures the gate judged): its `runs[]` of mode "on", `decode_steps_tok_s`. With `commit` only
/// that commit's aggregate counts. Zero or several aggregates are refused, so a later record (a rerun, or a later gate
/// run with MTP loaded) never wins silently.
pub fn load_measured(path: &Path, commit: Option<&str>) -> Result<BTreeMap<String, f64>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut found: Vec<serde_json::Value> = Vec::new();
    for (i, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let v: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("{}: line {}", path.display(), i + 1))?;
        if v["test"] == "serve-speed-spec"
            && v["runs"].is_array()
            && commit.is_none_or(|c| v["commit"] == c)
        {
            found.push(v);
        }
    }
    let [agg] = found.as_slice() else {
        let who: Vec<String> = found
            .iter()
            .map(|v| {
                format!(
                    "{} {}",
                    v["commit"].as_str().unwrap_or("?"),
                    v["date"].as_str().unwrap_or("?")
                )
            })
            .collect();
        bail!(
            "{}: {} \"serve-speed-spec\" aggregates{} ({who:?}); pass --measured-commit to pick one",
            path.display(),
            found.len(),
            commit.map_or(String::new(), |c| format!(" for commit {c}"))
        );
    };
    let mut out = BTreeMap::new();
    for r in agg["runs"].as_array().into_iter().flatten() {
        if r["mode"] != "on" {
            continue;
        }
        if let (Some(run), Some(tok_s)) = (r["run"].as_str(), r["decode_steps_tok_s"].as_f64()) {
            if out.insert(run.replace(':', "-"), tok_s).is_some() {
                bail!("{}: run {run} is in the aggregate twice", path.display());
            }
        }
    }
    Ok(out)
}

/// One stream's lookup-only simulation, its MTP simulations (the mean over the seeds) and its measured figure.
#[derive(Serialize)]
struct MtpRow {
    id: String,
    measured_on_tok_s: f64,
    lookup_sim_tok_s: f64,
    /// `(lookup_sim - measured) / measured`.
    calibration_error: f64,
    /// By `ms_per_draft`: the predicted tok/s and the simulation's counts.
    at: Vec<MtpAt>,
}

#[derive(Serialize)]
struct MtpAt {
    ms_per_draft: f64,
    predicted_tok_s: f64,
    sim: SimResult,
}

/// Prints and returns the MTP prediction for the `--record` streams.
fn mtp_report(
    a: &LookupSimArgs,
    streams: &[(Stream, &'static str)],
    base: &SpecConfig,
    cost: &Cost,
) -> Result<serde_json::Value> {
    let Some(path) = &a.measured else {
        bail!("--mtp-rates needs --measured");
    };
    if !(1..engine::spec::MAX_VERIFY).contains(&a.mtp_rates.len())
        || !a.mtp_rates.iter().all(|r| (0.0..=1.0).contains(r))
    {
        bail!("--mtp-rates takes 1 to 15 values, each from 0 to 1");
    }
    for (flag, v) in [
        ("--mtp-ms-per-draft", a.mtp_ms_per_draft),
        ("--mtp-ms-per-pass", a.mtp_ms_per_pass),
    ] {
        if !(v.is_finite() && v >= 0.0) {
            bail!("{flag} must be a finite number, 0 or more");
        }
    }
    if a.seeds < 1 {
        bail!("--seeds must be at least 1");
    }
    let measured = load_measured(path, a.measured_commit.as_deref())?;
    let records: Vec<&Stream> = streams
        .iter()
        .filter(|(_, src)| *src == "record")
        .map(|(s, _)| s)
        .collect();
    if records.is_empty() {
        bail!("--mtp-rates predicts the --record streams; give some");
    }
    let mut on = Vec::new();
    for s in &records {
        match measured.get(&s.id.replace(':', "-")) {
            Some(&m) => on.push(m),
            None => bail!(
                "{}: no \"on\" run for it in the aggregate in {} (it has {:?})",
                s.id,
                path.display(),
                measured.keys().collect::<Vec<_>>()
            ),
        }
    }
    let lookup_only: Vec<SimResult> = records
        .iter()
        .map(|s| simulate_capped(&s.prompt, &s.tokens, s.max_tokens, base, cost))
        .collect();
    let mut drafts = vec![MTP_MS_PER_DRAFT_LOW, MTP_MS_PER_DRAFT];
    if !drafts.contains(&a.mtp_ms_per_draft) {
        drafts.push(a.mtp_ms_per_draft);
    }
    drafts.sort_by(f64::total_cmp);
    // One prediction per stream at `ms_per_draft`, each stream with its own rates.
    let predict = |ms_per_draft: f64, rates: &[Vec<f64>]| -> Result<Vec<(f64, SimResult)>> {
        let mut sims = Vec::new();
        for (s, rates) in records.iter().zip(rates) {
            let cfg = SpecConfig {
                mtp_accept_prior: rates.clone(),
                mtp_cost_ms: ms_per_draft,
                ..base.clone()
            };
            cfg.check()
                .map_err(|(key, why)| anyhow::anyhow!("{key}: {why}"))?;
            let model = MtpModel {
                depth_rates: rates.clone(),
                ms_per_draft,
                ms_per_pass: a.mtp_ms_per_pass,
            };
            let runs: Vec<SimResult> = (0..a.seeds)
                .map(|seed| {
                    simulate_with_mtp_capped(
                        &s.prompt,
                        &s.tokens,
                        s.max_tokens,
                        &cfg,
                        cost,
                        &model,
                        seed,
                    )
                })
                .collect();
            sims.push(mean_result(&runs));
        }
        let pairs: Vec<(SimResult, SimResult)> = sims
            .iter()
            .cloned()
            .zip(lookup_only.iter().cloned())
            .collect();
        let figures = predict_runs(&pairs, &on);
        Ok(sims
            .into_iter()
            .zip(figures)
            .map(|(sim, (tok_s, _))| (tok_s, sim))
            .collect())
    };
    let pooled: Vec<Vec<f64>> = vec![a.mtp_rates.clone(); records.len()];
    let mut predicted: Vec<(f64, Vec<(f64, SimResult)>)> = Vec::new();
    for &ms_per_draft in &drafts {
        predicted.push((ms_per_draft, predict(ms_per_draft, &pooled)?));
    }
    // Sensitivity: the pooled rates are measured with MTP drafting everywhere, the simulator uses them where lookup
    // declines. Each stream with its own prompt's depth-1 rate (a2.. scaled by a1 / pooled a1, at most 1).
    let sensitivity = match &a.mtp_probe {
        None => None,
        Some(probe) => {
            let doc: serde_json::Value = serde_json::from_slice(
                &std::fs::read(probe).with_context(|| format!("reading {}", probe.display()))?,
            )
            .with_context(|| format!("parsing {}", probe.display()))?;
            let mut own = Vec::new();
            for s in &records {
                let prompt = s.id.split(['-', ':']).next().unwrap_or(&s.id);
                let Some(a1) = doc["depth1_per_prompt"][prompt].as_f64() else {
                    bail!(
                        "{}: no depth1_per_prompt for {prompt:?} (stream {})",
                        probe.display(),
                        s.id
                    );
                };
                let scale = if a.mtp_rates[0] > 0.0 {
                    a1 / a.mtp_rates[0]
                } else {
                    1.0
                };
                let mut rates = vec![a1.clamp(0.0, 1.0)];
                rates.extend(a.mtp_rates[1..].iter().map(|r| (r * scale).clamp(0.0, 1.0)));
                own.push(rates);
            }
            Some((predict(MTP_MS_PER_DRAFT, &own)?, own))
        }
    };
    let errors: Vec<f64> = predict_runs(
        &lookup_only
            .iter()
            .cloned()
            .zip(lookup_only.iter().cloned())
            .collect::<Vec<_>>(),
        &on,
    )
    .iter()
    .map(|f| f.1)
    .collect();
    println!(
        "\n### MTP prediction (depth rates {:?}, d(k) = {} + k * ms_per_draft ms, {} seeds)\n",
        a.mtp_rates, a.mtp_ms_per_pass, a.seeds
    );
    println!("Calibration: the lookup-only simulation against the measured \"on\" run.\n");
    println!("| stream | measured on tok/s | lookup-only sim tok/s | error % |");
    println!("|---|---:|---:|---:|");
    for (i, s) in records.iter().enumerate() {
        println!(
            "| {} | {:.1} | {:.1} | {:+.1} |",
            s.id,
            on[i],
            lookup_only[i].predicted_tok_s(),
            errors[i] * 100.0
        );
    }
    println!("\nPrediction: measured on x (MTP simulation / lookup-only simulation).\n");
    let head: Vec<String> = drafts.iter().map(|d| format!("MTP @ {d}")).collect();
    println!("| stream | measured on | {} |", head.join(" | "));
    println!("|---|---:|{}", "---:|".repeat(drafts.len()));
    for (i, s) in records.iter().enumerate() {
        let cells: Vec<String> = predicted
            .iter()
            .map(|(_, rows)| format!("{:.1}", rows[i].0))
            .collect();
        println!("| {} | {:.1} | {} |", s.id, on[i], cells.join(" | "));
    }
    let lookup_median = speed::median(&on);
    let mut medians = Vec::new();
    for (ms, rows) in &predicted {
        let med = speed::median(&rows.iter().map(|r| r.0).collect::<Vec<_>>());
        medians.push((*ms, med));
        println!(
            "\nAt ms_per_draft {ms}: predicted median of {}: {med:.1} tok/s, lookup-only measured median \
             {lookup_median:.1}, bar {SPEED_BAR_TOK_S:.1} ({}); MTP's predicted gain over lookup alone on the median \
             {:+.1}%.",
            rows.len(),
            if med >= SPEED_BAR_TOK_S {
                "meets the bar"
            } else {
                "below the bar"
            },
            (med / lookup_median - 1.0) * 100.0
        );
    }
    let at_checkpoint = medians
        .iter()
        .find(|m| m.0 == MTP_MS_PER_DRAFT)
        .map_or(0.0, |m| m.1);
    let gain = at_checkpoint / lookup_median - 1.0;
    // Under 3% MTP would not pay its memory.
    let verdict = if gain >= CHECKPOINT_MIN_GAIN {
        format!("gain {:.1}% >= 3%: go on", gain * 100.0)
    } else {
        format!(
            "gain {:.1}% < 3%: STOP and take the figures to the maintainer",
            gain * 100.0
        )
    };
    println!(
        "\nCheckpoint (judged at ms_per_draft {MTP_MS_PER_DRAFT}): predicted median {at_checkpoint:.1} tok/s against \
         {SPEED_BAR_TOK_S:.1}; gain over lookup alone {:+.1}%. {verdict}.",
        gain * 100.0
    );
    let sens_median = sensitivity.as_ref().map(|(rows, _)| {
        let med = speed::median(&rows.iter().map(|r| r.0).collect::<Vec<_>>());
        println!(
            "Sensitivity (each stream with its own prompt's depth-1 rate, at {MTP_MS_PER_DRAFT}): predicted median \
             {med:.1} tok/s, gain {:+.1}% (pooled: {at_checkpoint:.1}, {:+.1}%).",
            (med / lookup_median - 1.0) * 100.0,
            gain * 100.0
        );
        med
    });
    let rows: Vec<MtpRow> = records
        .iter()
        .enumerate()
        .map(|(i, s)| MtpRow {
            id: s.id.clone(),
            measured_on_tok_s: on[i],
            lookup_sim_tok_s: lookup_only[i].predicted_tok_s(),
            calibration_error: errors[i],
            at: predicted
                .iter()
                .map(|(ms, rows)| MtpAt {
                    ms_per_draft: *ms,
                    predicted_tok_s: rows[i].0,
                    sim: rows[i].1.clone(),
                })
                .collect(),
        })
        .collect();
    Ok(serde_json::json!({
        "depth_rates": a.mtp_rates,
        "ms_per_pass": a.mtp_ms_per_pass,
        "seeds": a.seeds,
        "rows": rows,
        "lookup_measured_median": lookup_median,
        "medians": medians
            .iter()
            .map(|(ms, med)| serde_json::json!({
                "ms_per_draft": ms,
                "predicted_median": med,
                "gain_over_lookup": med / lookup_median - 1.0,
            }))
            .collect::<Vec<_>>(),
        "bar": SPEED_BAR_TOK_S,
        "checkpoint_ms_per_draft": MTP_MS_PER_DRAFT,
        "checkpoint_predicted_median": at_checkpoint,
        "checkpoint_gain_over_lookup": gain,
        "checkpoint_verdict": verdict,
        "sensitivity": sensitivity.as_ref().zip(sens_median).map(|((rows, rates), med)| serde_json::json!({
            "predicted_median": med,
            "gain_over_lookup": med / lookup_median - 1.0,
            "per_stream": records.iter().zip(rows).zip(rates).map(|((s, r), rates)| serde_json::json!({
                "id": s.id, "depth_rates": rates, "predicted_tok_s": r.0,
            })).collect::<Vec<_>>(),
        })),
    }))
}

/// The prompt a stream id names: `write-1234` is `write`; a golden stream is `golden`.
fn prompt_of(id: &str, source: &str) -> String {
    if source == "golden" {
        return "golden".to_string();
    }
    id.split(['-', ':']).next().unwrap_or(id).to_string()
}

fn per_token(c: &BucketCount) -> f64 {
    if c.tested == 0 {
        0.0
    } else {
        c.accepted as f64 / c.tested as f64
    }
}

/// `--buckets`: lookup's accept rates by class, per prompt and over all the streams.
fn buckets_report(
    a: &LookupSimArgs,
    streams: &[(Stream, &'static str)],
    cfg: &SpecConfig,
) -> Result<()> {
    let mut groups: BTreeMap<String, BucketReport> = BTreeMap::new();
    for (s, source) in streams {
        let r = bucket_rates(
            &s.prompt,
            &s.tokens,
            cfg.max_draft,
            cfg.ngram_min,
            cfg.ngram_max,
        );
        groups.entry(prompt_of(&s.id, source)).or_default().add(&r);
        groups.entry("all".to_string()).or_default().add(&r);
    }
    println!(
        "Lookup verifying every draft (up to {} tokens, n-grams {} to {}), no controller. Per token: kept over \
         compared, as the controller counts (at most one rejection a draft).",
        cfg.max_draft, cfg.ngram_min, cfg.ngram_max
    );
    const NAMES: [&str; 4] = [
        "2-token match",
        "3-token match",
        "4-token match",
        "continuing run",
    ];
    for (g, r) in &groups {
        println!(
            "\n### {g}: {} tokens, {} steps ({} plain)\n",
            r.tokens, r.steps, r.plain
        );
        println!("| drafts by | drafts | tokens | kept / drafted | per token | first kept | kept whole |");
        println!("|---|---:|---:|---:|---:|---:|---:|");
        let row = |name: &str, c: &BucketCount| {
            if c.drafts == 0 {
                return;
            }
            println!(
                "| {name} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} |",
                c.drafts,
                c.drafted,
                c.accepted as f64 / c.drafted.max(1) as f64,
                per_token(c),
                c.first_kept as f64 / c.drafts as f64,
                c.full as f64 / c.drafts as f64
            );
        };
        let mut all = BucketCount::default();
        for (name, c) in NAMES.iter().zip(&r.by_class) {
            row(name, c);
            all.add(c);
        }
        row("after a draft kept whole", &r.after_full[1]);
        row("after anything else", &r.after_full[0]);
        row("all", &all);
    }
    if let Some(path) = &a.json {
        let doc = serde_json::json!({
            "format": "artemis-lookup-sim-buckets-1",
            "max_draft": cfg.max_draft,
            "ngram_min": cfg.ngram_min,
            "ngram_max": cfg.ngram_max,
            "groups": groups,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&doc)?)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

/// One controller setting `--variants` simulates.
struct Setting {
    name: &'static str,
    variant: Variant,
    ngram_min: usize,
    lookup_only: bool,
}

const fn setting(
    name: &'static str,
    variant: Variant,
    ngram_min: usize,
    lookup_only: bool,
) -> Setting {
    Setting {
        name,
        variant,
        ngram_min,
        lookup_only,
    }
}

/// `--variants`: the controller before and after, and `ngram_min`, per `--record` stream.
fn variants_report(
    a: &LookupSimArgs,
    streams: &[(Stream, &'static str)],
    base: &SpecConfig,
    cost: &Cost,
) -> Result<()> {
    let records: Vec<&Stream> = streams
        .iter()
        .filter(|(_, src)| *src == "record")
        .map(|(s, _)| s)
        .collect();
    if records.is_empty() {
        bail!("--variants compares the --record streams; give some");
    }
    if a.seeds < 1 {
        bail!("--seeds must be at least 1");
    }
    let mut rates_for: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for item in &a.mtp_rates_for {
        let Some((prompt, list)) = item.split_once('=') else {
            bail!("--mtp-rates-for {item:?} is not PROMPT=A1,A2,..");
        };
        let rates: Vec<f64> = list
            .split(',')
            .map(|v| v.trim().parse::<f64>())
            .collect::<std::result::Result<_, _>>()
            .with_context(|| format!("--mtp-rates-for {item:?}"))?;
        if !(1..engine::spec::MAX_VERIFY).contains(&rates.len())
            || !rates.iter().all(|r| (0.0..=1.0).contains(r))
        {
            bail!("--mtp-rates-for {item:?}: 1 to 15 rates, each from 0 to 1");
        }
        rates_for.insert(prompt.to_string(), rates);
    }
    let measured = match &a.measured {
        Some(path) => Some(load_measured(path, a.measured_commit.as_deref())?),
        None => None,
    };
    let before = Variant::BEFORE;
    let after = Variant::AFTER;
    let n2 = base.ngram_min;
    let settings = [
        setting("before", before, n2, false),
        setting(
            "whole-draft scoring",
            Variant {
                first_token_only: false,
                ..before
            },
            n2,
            false,
        ),
        setting(
            "classes",
            Variant {
                pooled_lookup: false,
                ..before
            },
            n2,
            false,
        ),
        setting("after (both)", after, n2, false),
        setting("before, ngram_min 3", before, 3, false),
        setting("before, ngram_min 4", before, 4, false),
        setting("after, ngram_min 4", after, 4, false),
        setting("lookup only, before", before, n2, true),
        setting("lookup only, after", after, n2, true),
        setting("lookup only, before, ngram_min 4", before, 4, true),
        setting("lookup only, after, ngram_min 4", after, 4, true),
    ];
    println!(
        "Cost: c(t) = {} + (t - 1) * {} ms; MTP pass d(k) = {} + k * {} ms; {} seeds; the controller starts at the \
         config's priors (fresh per stream).",
        cost.c1_ms, cost.slope_ms, a.mtp_ms_per_pass, a.mtp_ms_per_draft, a.seeds
    );
    let mut json_rows = Vec::new();
    // `ratios[setting][stream]`: simulated tok/s over the "before" setting's (lookup-only settings over "lookup only,
    // before"), MTP independent of the stream.
    let mut ratios = vec![Vec::new(); settings.len()];
    for s in &records {
        let prompt = prompt_of(&s.id, "record");
        let rates = rates_for
            .get(&prompt)
            .cloned()
            .or_else(|| (!a.mtp_rates.is_empty()).then(|| a.mtp_rates.clone()));
        let Some(rates) = rates else {
            bail!(
                "{}: no --mtp-rates-for {prompt}=.. and no --mtp-rates",
                s.id
            );
        };
        let model = MtpModel {
            depth_rates: rates.clone(),
            ms_per_draft: a.mtp_ms_per_draft,
            ms_per_pass: a.mtp_ms_per_pass,
        };
        println!("\n### {} (MTP depth rates {:?})\n", s.id, rates);
        println!(
            "| setting | MTP model | tok/s | ratio | lookup drafted by class (2/3/4/run) | lookup kept | MTP drafted | \
             MTP kept | steps |"
        );
        println!("|---|---|---:|---:|---|---:|---:|---:|---:|");
        let mut base_tok_s = [0.0f64; 2];
        let mut lookup_base = 0.0f64;
        for (i, st) in settings.iter().enumerate() {
            for (m, follows) in [false, true].into_iter().enumerate() {
                if st.lookup_only && follows {
                    continue;
                }
                let cfg = SpecConfig {
                    ngram_min: st.ngram_min,
                    sources: if st.lookup_only {
                        vec![Source::Lookup]
                    } else {
                        base.sources.clone()
                    },
                    ..base.clone()
                };
                cfg.check()
                    .map_err(|(key, why)| anyhow::anyhow!("{key}: {why}"))?;
                let variant = Variant {
                    mtp_follows_lookup: follows,
                    ..st.variant
                };
                let runs: Vec<SimResult> = (0..a.seeds)
                    .map(|seed| {
                        simulate_variant(
                            &s.prompt,
                            &s.tokens,
                            s.max_tokens,
                            &cfg,
                            cost,
                            (!st.lookup_only).then_some((&model, seed)),
                            variant,
                        )
                    })
                    .collect();
                let r = mean_result(&runs);
                let tok_s = r.predicted_tok_s();
                let ratio = if st.lookup_only {
                    if lookup_base == 0.0 {
                        lookup_base = tok_s;
                    }
                    tok_s / lookup_base
                } else {
                    if i == 0 {
                        base_tok_s[m] = tok_s;
                    }
                    tok_s / base_tok_s[m]
                };
                if !follows {
                    ratios[i].push(ratio);
                }
                let kept: usize = r.lookup_class_accepted.iter().sum();
                let model_name = if st.lookup_only {
                    "none"
                } else if follows {
                    "follows lookup"
                } else {
                    "independent"
                };
                println!(
                    "| {} | {model_name} | {tok_s:.1} | {ratio:.4} | {:?} | {kept} | {} | {} | {} |",
                    st.name,
                    r.lookup_class_drafted,
                    r.mtp_drafted,
                    r.mtp_accepted,
                    r.verify_steps + r.plain_steps
                );
                json_rows.push(serde_json::json!({
                    "stream": s.id, "setting": st.name, "mtp_follows_lookup": follows,
                    "lookup_only": st.lookup_only, "ngram_min": st.ngram_min, "variant": variant,
                    "predicted_tok_s": tok_s, "ratio": ratio, "sim": r,
                }));
            }
        }
    }
    let mut medians = Vec::new();
    if let Some(on) = &measured {
        let mut figures = Vec::new();
        for s in &records {
            match on.get(&s.id.replace(':', "-")) {
                Some(&m) => figures.push(m),
                None => bail!("{}: no measured \"on\" run for it", s.id),
            }
        }
        println!(
            "\nMeasured \"on\" figures scaled by each setting's simulated ratio (MTP independent of the stream); \
             lookup-only settings are scaled the same way and compare only among themselves.\n"
        );
        let ids: Vec<&str> = records.iter().map(|s| s.id.as_str()).collect();
        println!("| setting | {} | median |", ids.join(" | "));
        println!("|---|{}---:|", "---:|".repeat(records.len()));
        for (st, rs) in settings.iter().zip(&ratios) {
            let scaled: Vec<f64> = figures.iter().zip(rs).map(|(m, r)| m * r).collect();
            let med = speed::median(&scaled);
            let cells: Vec<String> = scaled.iter().map(|v| format!("{v:.1}")).collect();
            println!("| {} | {} | {med:.2} |", st.name, cells.join(" | "));
            medians.push(serde_json::json!({"setting": st.name, "figures": scaled, "median": med}));
        }
    }
    if let Some(path) = &a.json {
        let doc = serde_json::json!({
            "format": "artemis-lookup-sim-variants-1",
            "cost": cost,
            "mtp_ms_per_pass": a.mtp_ms_per_pass,
            "mtp_ms_per_draft": a.mtp_ms_per_draft,
            "seeds": a.seeds,
            "rows": json_rows,
            "scaled_medians": medians,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&doc)?)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

/// `artemis lookup-sim`.
pub fn run(a: LookupSimArgs) -> Result<()> {
    if a.golden.is_none() && a.records.is_empty() {
        bail!("give --golden DIR, --record ID=FILE, or both");
    }
    if !(a.c1_ms.is_finite() && a.c1_ms > 0.0) {
        bail!("--c1-ms must be a finite number above 0");
    }
    if !(a.slope_ms.is_finite() && a.slope_ms >= 0.0) {
        bail!("--slope-ms must be a finite number, 0 or more");
    }
    if a.draft_len.is_some_and(|n| n >= engine::spec::MAX_VERIFY) {
        bail!(
            "--draft-len must be at most {}",
            engine::spec::MAX_VERIFY - 1
        );
    }
    let mut streams: Vec<(Stream, &'static str)> = Vec::new();
    if let Some(dir) = &a.golden {
        streams.extend(load_golden(dir)?.into_iter().map(|s| (s, "golden")));
    }
    for r in &a.records {
        let Some((id, file)) = r
            .split_once('=')
            .filter(|(id, f)| !id.is_empty() && !f.is_empty())
        else {
            bail!("--record {r:?} is not ID=FILE");
        };
        streams.push((load_record(id, Path::new(file))?, "record"));
    }
    let (cost, source) = match &a.verify_cost {
        Some(path) => {
            let doc: serde_json::Value = serde_json::from_slice(
                &std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
            )
            .with_context(|| format!("parsing {}", path.display()))?;
            (
                Cost::from_verify_cost(&doc, a.cost_ctx)?,
                format!("verify-cost rows at ctx {}", a.cost_ctx),
            )
        }
        None => {
            let cost = Cost::flat(a.c1_ms, a.slope_ms);
            let source = if cost == Cost::default() {
                "spec 5.3 estimate; pass verify-cost's c(1) and slope"
            } else {
                "given"
            };
            (cost, source.to_string())
        }
    };
    let mut cfg = config(&a, &cost)?;
    if a.buckets {
        return buckets_report(&a, &streams, &cfg);
    }
    if a.variants {
        return variants_report(&a, &streams, &cfg, &cost);
    }
    let base_cfg = cfg.clone();
    if cost.verify_ms.is_some() {
        println!(
            "Cost model: plain step {:.2} ms, verify steps {:?} ms for t = 2, 3, ... ({source}); the controller's \
             slope guess is {:.2}. Each stream starts a fresh controller.",
            cost.plain_step_ms(),
            cost.verify_ms.as_deref().unwrap_or(&[]),
            cost.slope_ms
        );
    } else {
        println!(
            "Cost model: c(t) = {} + (t - 1) * {} ms ({source}); it ignores the context length. Each stream starts a \
             fresh controller.",
            cost.c1_ms, cost.slope_ms
        );
    }
    let mut tables = vec![run_table(&streams, &cfg, &cost)];
    // The controller's start is moot with a fixed length; otherwise the table is also made at prior 0.
    if cfg.fixed_draft.is_none() && cfg.accept_prior != 0.0 {
        cfg.accept_prior = 0.0;
        tables.push(run_table(&streams, &cfg, &cost));
    }
    for t in &tables {
        print_table(t);
    }
    let mtp = if a.mtp_rates.is_empty() {
        None
    } else {
        Some(mtp_report(&a, &streams, &base_cfg, &cost)?)
    };
    if let Some(path) = &a.json {
        let doc = serde_json::json!({
            "format": "artemis-lookup-sim-1",
            "cost": cost,
            "cost_source": source,
            "cost_depends_on_context": false,
            "fresh_controller_per_stream": true,
            "tables": tables,
            "mtp": mtp,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&doc)?)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed(n: usize) -> SpecConfig {
        SpecConfig {
            max_draft: n.max(4),
            fixed_draft: Some(n),
            ..SpecConfig::default()
        }
    }

    fn golden_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden")
    }

    #[test]
    fn the_simulator_counts_a_tiny_stream() {
        // The first step has no draft: plain, emits 2. Then [1,2] recurs and the draft [3,4,...] is cut to the 2 that
        // keep the step inside the stream: one verify of [x,3,4], both accepted, and the bonus 5.
        let r = simulate(&[1, 2, 3, 4], &[1, 2, 3, 4, 5], &fixed(4), &Cost::default());
        assert_eq!(r.tokens, 4);
        assert_eq!((r.verify_steps, r.plain_steps), (1, 1));
        assert_eq!((r.drafted, r.accepted), (2, 2));
        assert_eq!(r.mean_draft, 2.0);
        assert_eq!(r.tokens_per_step, 2.0);
        let c = Cost::default();
        assert_eq!(r.predicted_ms, c.step_ms(1) + c.step_ms(3));
        assert_eq!(r.plain_ms, 4.0 * c.c1_ms);
    }

    #[test]
    fn a_rejected_draft_emits_the_pick_and_counts_as_drafted() {
        // After the plain step the draft is [3,7,..]; the stream has 3 then 8: one accepted, one rejected, and the
        // verify still emits both 3 and the picked 8. The last token is a plain step (no room for a draft).
        let r = simulate(&[1, 2, 3, 7], &[1, 2, 3, 8, 0], &fixed(4), &Cost::default());
        assert_eq!(r.tokens, 4);
        assert_eq!((r.verify_steps, r.plain_steps), (1, 2));
        assert_eq!((r.drafted, r.accepted), (2, 1));
        let c = Cost::default();
        assert_eq!(r.predicted_ms, 2.0 * c.step_ms(1) + c.step_ms(3));
        assert_eq!(r.tokens, r.plain_steps + r.accepted + r.verify_steps);
    }

    #[test]
    fn a_stream_with_no_repeats_runs_plain() {
        let prompt: Vec<u32> = (0..50).collect();
        let stream: Vec<u32> = (100..160).collect();
        let c = Cost::default();
        for cfg in [fixed(4), SpecConfig::default()] {
            let r = simulate(&prompt, &stream, &cfg, &c);
            assert_eq!(r.tokens, 59);
            assert_eq!((r.verify_steps, r.plain_steps, r.drafted), (0, 59, 0));
            assert_eq!(r.predicted_ms, r.plain_ms);
            assert_eq!(r.mean_draft, 0.0);
            assert_eq!(r.tokens_per_step, 1.0);
        }
    }

    #[test]
    fn a_repetitive_stream_gains_under_the_controller() {
        let prompt: Vec<u32> = (0..40).collect();
        // The stream replays the prompt twice: after a few plain steps the drafts hit.
        let stream: Vec<u32> = (0..40).chain(0..40).collect();
        let r = simulate(&prompt, &stream, &SpecConfig::default(), &Cost::default());
        assert!(r.verify_steps > 5, "{r:?}");
        assert!(r.accept_rate() > 0.9, "{r:?}");
        assert!(r.predicted_ms < r.plain_ms, "{r:?}");
        assert!(r.predicted_tok_s() > r.plain_tok_s());
        assert_eq!(r.tokens, r.plain_steps + r.accepted + r.verify_steps);
    }

    #[test]
    fn the_controller_stops_verifying_drafts_that_keep_failing() {
        // `1, 2` recurs every third token and lookup drafts what followed it last time, but each follower is new:
        // every draft is wrong. A fixed length keeps paying for them; the controller's accept rate falls and it stops.
        let prompt: Vec<u32> = [1, 2, 3].repeat(20);
        let stream: Vec<u32> = (0..100u32).flat_map(|i| [1, 2, 100 + i]).collect();
        let c = Cost::default();
        let ctl = simulate(&prompt, &stream, &SpecConfig::default(), &c);
        let fix = simulate(&prompt, &stream, &fixed(4), &c);
        // Only the first draft, from the prompt's own cycle, can match.
        assert!(ctl.accepted <= 1 && fix.accepted <= 1, "{ctl:?} {fix:?}");
        assert!(fix.verify_steps > 90, "{fix:?}");
        assert!(fix.predicted_ms > fix.plain_ms, "{fix:?}");
        assert!(ctl.verify_steps < fix.verify_steps / 2, "{ctl:?} {fix:?}");
        assert!(ctl.predicted_ms < fix.predicted_ms, "{ctl:?} {fix:?}");
    }

    #[test]
    fn an_empty_or_one_token_stream_has_no_steps() {
        let c = Cost::default();
        for s in [&[][..], &[5][..]] {
            let r = simulate(&[1, 2], s, &fixed(4), &c);
            assert_eq!((r.tokens, r.verify_steps, r.plain_steps), (0, 0, 0));
            assert_eq!(
                (r.predicted_ms, r.plain_ms, r.tokens_per_step),
                (0.0, 0.0, 0.0)
            );
            assert_eq!(r.predicted_tok_s(), 0.0);
        }
    }

    #[test]
    fn the_golden_files_load() {
        let g = load_golden(&golden_dir()).unwrap();
        assert_eq!(g.len(), 20);
        assert_eq!(g[0].id, "01-code");
        assert_eq!(g[19].id, "20-long-option-rs");
        assert_eq!(g[0].prompt.len(), 23);
        assert_eq!(g[19].prompt.len(), 27_416);
        // 256 tokens, fewer when generation stopped at the end-of-generation token.
        assert!(g.iter().all(|s| (1..=256).contains(&s.tokens.len())));
        assert_eq!(g.iter().filter(|s| s.tokens.len() == 256).count(), 15);
        assert_eq!(g.iter().map(|s| s.tokens.len()).sum::<usize>(), 4_498);
        // Every golden request had `max_tokens` 256, the stopped ones included.
        assert!(g.iter().all(|s| s.max_tokens == 256));
    }

    #[test]
    fn a_prompt_of_the_wrong_length_is_refused() {
        let dir = std::env::temp_dir().join(format!("artemis-lookup-sim-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("expected")).unwrap();
        std::fs::create_dir_all(dir.join("prompts")).unwrap();
        std::fs::write(
            dir.join("expected/a.json"),
            r#"{"prompt_tokens": 3, "steps": [{"token": 7}]}"#,
        )
        .unwrap();
        std::fs::write(dir.join("prompts/a.tokens"), [1u8, 0, 0, 0, 2, 0, 0, 0]).unwrap();
        let err = load_golden(&dir).err().expect("refused").to_string();
        assert!(
            err.contains("a: ") && err.contains("2 tokens") && err.contains("is 3"),
            "{err}"
        );
        std::fs::write(
            dir.join("expected/a.json"),
            r#"{"prompt_tokens": 2, "steps": [{"token": 7}]}"#,
        )
        .unwrap();
        let g = load_golden(&dir).unwrap();
        assert_eq!(
            (g[0].prompt.clone(), g[0].tokens.clone()),
            (vec![1, 2], vec![7])
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_golden_streams_simulate_consistently() {
        let g = load_golden(&golden_dir()).unwrap();
        let c = Cost::default();
        for s in &g {
            let r = simulate_capped(
                &s.prompt,
                &s.tokens,
                s.max_tokens,
                &SpecConfig::default(),
                &c,
            );
            assert_eq!(r.tokens, s.tokens.len() - 1, "{}", s.id);
            assert_eq!(
                r.tokens,
                r.plain_steps + r.accepted + r.verify_steps,
                "{}",
                s.id
            );
            assert!(r.accepted <= r.drafted, "{}", s.id);
            // Replays are deterministic.
            assert_eq!(
                r,
                simulate_capped(
                    &s.prompt,
                    &s.tokens,
                    s.max_tokens,
                    &SpecConfig::default(),
                    &c
                )
            );
        }
    }

    fn args(golden: bool, records: &[&str]) -> LookupSimArgs {
        LookupSimArgs {
            golden: golden.then(golden_dir),
            records: records.iter().map(|r| r.to_string()).collect(),
            draft_len: None,
            max_draft: None,
            accept_prior: None,
            source_choice: None,
            c1_ms: Cost::default().c1_ms,
            slope_ms: Cost::default().slope_ms,
            json: None,
            mtp_rates: Vec::new(),
            mtp_ms_per_draft: MTP_MS_PER_DRAFT,
            mtp_ms_per_pass: MTP_MS_PER_PASS,
            verify_cost: None,
            cost_ctx: 2048,
            measured: None,
            measured_commit: None,
            mtp_probe: None,
            seeds: 16,
            buckets: false,
            variants: false,
            mtp_rates_for: Vec::new(),
        }
    }

    #[test]
    fn run_refuses_no_streams_a_bad_record_and_a_bad_cost() {
        let err = |a: LookupSimArgs| run(a).unwrap_err().to_string();
        assert!(err(args(false, &[])).contains("--golden"));
        assert!(err(args(false, &["nofile"])).contains("ID=FILE"));
        assert!(err(args(false, &["=x"])).contains("ID=FILE"));
        let mut a = args(true, &[]);
        a.c1_ms = 0.0;
        assert!(err(a).contains("--c1-ms"));
        let mut a = args(true, &[]);
        a.draft_len = Some(16);
        assert!(err(a).contains("--draft-len"));
    }

    #[test]
    fn run_writes_the_controller_tables_at_the_prior_and_at_zero() {
        let json =
            std::env::temp_dir().join(format!("artemis-lookup-sim-{}.json", std::process::id()));
        let mut a = args(true, &[]);
        a.json = Some(json.clone());
        run(a).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&json).unwrap()).unwrap();
        std::fs::remove_file(&json).unwrap();
        assert_eq!(v["format"], "artemis-lookup-sim-1");
        let tables = v["tables"].as_array().unwrap();
        assert_eq!(tables.len(), 2);
        assert_eq!(tables[0]["accept_prior"], 0.5);
        assert_eq!(tables[1]["accept_prior"], 0.0);
        assert_eq!(tables[0]["rows"].as_array().unwrap().len(), 20);
        assert!(tables[0]["record_median_tok_s"].is_null());
        // A fixed length makes one table: the prior doesn't enter.
        let mut a = args(true, &[]);
        a.draft_len = Some(4);
        a.json = Some(json.clone());
        run(a).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&json).unwrap()).unwrap();
        std::fs::remove_file(&json).unwrap();
        assert_eq!(v["tables"].as_array().unwrap().len(), 1);
        assert_eq!(v["tables"][0]["draft_len"], 4);
    }

    #[test]
    fn a_stream_cut_by_a_stop_still_verifies_drafts_past_it() {
        // The same tiny stream, but it ended at a stop with 15 tokens of room: the turn caps k at max_tokens, so the
        // verify carries all 4 drafts, pays for them, and the stop cuts the burst after the 2 that matched.
        let c = Cost::default();
        let r = simulate_capped(&[1, 2, 3, 4], &[1, 2, 3, 4, 5], 20, &fixed(4), &c);
        assert_eq!(r.tokens, 4);
        assert_eq!((r.verify_steps, r.plain_steps), (1, 1));
        assert_eq!((r.drafted, r.accepted), (4, 2));
        assert_eq!(r.predicted_ms, c.step_ms(1) + c.step_ms(5));
        // Without the room the cap is the stream's end, as `simulate` has it.
        let capped = simulate_capped(&[1, 2, 3, 4], &[1, 2, 3, 4, 5], 5, &fixed(4), &c);
        assert_eq!(
            capped,
            simulate(&[1, 2, 3, 4], &[1, 2, 3, 4, 5], &fixed(4), &c)
        );
        assert_eq!(capped.drafted, 2);
        // A smaller cap than the stream is taken as the stream's length.
        assert_eq!(
            simulate_capped(&[1, 2, 3, 4], &[1, 2, 3, 4, 5], 2, &fixed(4), &c),
            capped
        );
    }

    #[test]
    fn a_fixed_length_above_four_verifies_that_many_drafts() {
        let prompt: Vec<u32> = (0..40).collect();
        let stream: Vec<u32> = (0..40).chain(0..40).collect();
        let r = simulate(&prompt, &stream, &fixed(8), &Cost::default());
        assert!(r.mean_draft > 6.0, "{r:?}");
        assert!(r.accept_rate() > 0.99, "{r:?}");
    }

    #[test]
    fn the_config_lifts_max_draft_to_a_longer_fixed_length_only() {
        let cost = Cost::default();
        let cfg = config(
            &LookupSimArgs {
                draft_len: Some(8),
                ..args(true, &[])
            },
            &cost,
        )
        .unwrap();
        assert_eq!((cfg.max_draft, cfg.fixed_draft), (8, Some(8)));
        let cfg = config(
            &LookupSimArgs {
                draft_len: Some(3),
                ..args(true, &[])
            },
            &cost,
        )
        .unwrap();
        assert_eq!((cfg.max_draft, cfg.fixed_draft), (4, Some(3)));
        let cfg = config(
            &LookupSimArgs {
                draft_len: Some(5),
                ..args(true, &[])
            },
            &cost,
        )
        .unwrap();
        assert_eq!((cfg.max_draft, cfg.fixed_draft), (5, Some(5)));
        let cfg = config(&args(true, &[]), &cost).unwrap();
        assert_eq!(
            (cfg.max_draft, cfg.fixed_draft, cfg.accept_prior),
            (4, None, 0.5)
        );
    }

    #[test]
    fn a_bad_accept_prior_is_refused_under_the_flags_name() {
        for p in [1.5, -0.1, f64::NAN] {
            let a = LookupSimArgs {
                accept_prior: Some(p),
                ..args(true, &[])
            };
            let err = run(a).unwrap_err().to_string();
            assert!(
                err.contains("--accept-prior") && !err.contains("accept_prior"),
                "{err}"
            );
        }
        let ok = LookupSimArgs {
            accept_prior: Some(1.0),
            ..args(true, &[])
        };
        assert!(config(&ok, &Cost::default()).is_ok());
    }

    #[test]
    fn the_json_labels_the_cost_model_and_the_fresh_controllers() {
        let json = std::env::temp_dir().join(format!(
            "artemis-lookup-sim-label-{}.json",
            std::process::id()
        ));
        let read = |a: LookupSimArgs| {
            run(a).unwrap();
            let v: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&json).unwrap()).unwrap();
            std::fs::remove_file(&json).unwrap();
            v
        };
        let v = read(LookupSimArgs {
            json: Some(json.clone()),
            ..args(true, &[])
        });
        assert!(v["cost_source"]
            .as_str()
            .unwrap()
            .starts_with("spec 5.3 estimate"));
        assert_eq!(v["cost_depends_on_context"], false);
        assert_eq!(v["fresh_controller_per_stream"], true);
        let v = read(LookupSimArgs {
            json: Some(json.clone()),
            c1_ms: 9.7,
            ..args(true, &[])
        });
        assert_eq!(v["cost_source"], "given");
        assert_eq!(v["cost"]["c1_ms"], 9.7);
    }

    // ---- MTP ----

    /// The engine's measured costs for the MTP tests: c(1) 10.4, c(t) = 10.4 + 4.4 (t - 1), d(k) = 0.3 + 1.8 k.
    fn measured_cost() -> Cost {
        Cost::flat(10.4, 4.4)
    }

    fn mtp(rates: &[f64]) -> MtpModel {
        MtpModel {
            depth_rates: rates.to_vec(),
            ms_per_draft: 1.8,
            ms_per_pass: 0.3,
        }
    }

    fn mtp_cfg(prior: &[f64]) -> SpecConfig {
        SpecConfig {
            mtp_accept_prior: prior.to_vec(),
            source_choice: SourceChoice::Priority,
            ..SpecConfig::default()
        }
    }

    fn no_repeat(len: u32) -> (Vec<u32>, Vec<u32>) {
        ((0..50).collect(), (1000..1000 + len).collect())
    }

    #[test]
    fn mtp_at_rate_1_emits_k_plus_1_a_step() {
        // 1 + 5 * 20 tokens: a plain first step (c(1) unknown), then twenty steps of max_draft 4 + 1.
        let (prompt, stream) = no_repeat(102);
        let cfg = mtp_cfg(&[1.0]);
        let r = simulate_with_mtp(
            &prompt,
            &stream,
            &cfg,
            &measured_cost(),
            &mtp(&[1.0, 1.0, 1.0]),
            7,
        );
        assert_eq!(r.tokens, 101);
        assert_eq!((r.plain_steps, r.verify_steps), (1, 20));
        assert_eq!((r.drafted, r.accepted), (80, 80));
        assert_eq!((r.mtp_steps, r.mtp_drafted, r.mtp_accepted), (20, 80, 80));
        assert_eq!(
            (r.tokens - r.plain_steps) as f64 / r.verify_steps as f64,
            5.0
        );
        // A depth past the list takes its last value: depth 4 kept at rate 1 too.
        let c = measured_cost();
        let d = 0.3 + 1.8 * 4.0;
        assert!((r.predicted_ms - (c.step_ms(1) + 20.0 * (d + c.step_ms(5)))).abs() < 1e-9);
        assert!(r.predicted_tok_s() > r.plain_tok_s());
    }

    #[test]
    fn a_fresh_stream_drafts_with_mtp_at_the_measured_prior() {
        let c = measured_cost();
        // Prior [0.75, 0.70]: after the plain first step MTP drafts one token from the second step on.
        let (prompt, stream) = no_repeat(4);
        let r = simulate_with_mtp(
            &prompt,
            &stream,
            &mtp_cfg(&[0.75, 0.70]),
            &c,
            &mtp(&[1.0]),
            1,
        );
        assert_eq!((r.plain_steps, r.verify_steps), (1, 1));
        assert_eq!((r.mtp_steps, r.mtp_drafted, r.mtp_probes), (1, 1, 0));
        // A single prior of 0.5 drafts only on probes: none in a short stream at probe_every 64 ...
        let (prompt, stream) = no_repeat(41);
        let cold = simulate_with_mtp(&prompt, &stream, &mtp_cfg(&[0.5]), &c, &mtp(&[1.0]), 1);
        assert_eq!((cold.verify_steps, cold.plain_steps), (0, 40));
        // ... and k = 1 probes, one every probe_every plain steps, at probe_every 4.
        let probing = SpecConfig {
            probe_every: 4,
            ..mtp_cfg(&[0.5])
        };
        let p = simulate_with_mtp(&prompt, &stream, &probing, &c, &mtp(&[1.0]), 1);
        assert!(p.mtp_steps > 0, "{p:?}");
        assert_eq!((p.mtp_drafted, p.mtp_probes), (p.mtp_steps, p.mtp_steps));
        assert_eq!(p.mtp_accepted, p.mtp_steps);
        // The first probe comes after 4 plain steps, then 4 more before the next.
        assert!(p.plain_steps >= 4 * p.mtp_steps, "{p:?}");
    }

    #[test]
    fn sim_lookup_drafts_take_priority_over_mtp() {
        // The stream replays the prompt: lookup has a draft from the second step on. MTP's prior is high and its
        // measured rate is 0, so any MTP step would show as a rejected draft.
        let prompt: Vec<u32> = (0..40).collect();
        let stream: Vec<u32> = (0..40).chain(0..40).collect();
        let cfg = mtp_cfg(&[1.0]);
        let r = simulate_with_mtp(&prompt, &stream, &cfg, &measured_cost(), &mtp(&[0.0]), 3);
        assert!(r.verify_steps > 5, "{r:?}");
        assert_eq!(r.mtp_steps, 0, "{r:?}");
        assert_eq!(r.accepted, r.drafted, "{r:?}");
        // The same as lookup alone.
        let alone = simulate_capped(&prompt, &stream, stream.len(), &cfg, &measured_cost());
        assert_eq!(
            (r.drafted, r.accepted, r.predicted_ms),
            (alone.drafted, alone.accepted, alone.predicted_ms)
        );
    }

    #[test]
    fn mtp_takes_a_lookup_draft_the_controller_declines_and_lookup_scores_its_first_token() {
        // Lookup drafts `1, 2` -> a wrong token every third position, so its rate falls and its choice becomes 0.
        // MTP (prior and measured rate 1) then drafts, and lookup keeps scoring its first draft token.
        let prompt: Vec<u32> = [1, 2, 3].repeat(20);
        let stream: Vec<u32> = (0..100u32).flat_map(|i| [1, 2, 100 + i]).collect();
        let cfg = mtp_cfg(&[1.0]);
        let r = simulate_with_mtp(&prompt, &stream, &cfg, &measured_cost(), &mtp(&[1.0]), 5);
        assert!(r.mtp_steps > 10, "{r:?}");
        let lookup_only = simulate_capped(&prompt, &stream, stream.len(), &cfg, &measured_cost());
        assert!(
            r.predicted_ms < lookup_only.predicted_ms,
            "{r:?} {lookup_only:?}"
        );
    }

    #[test]
    fn plain_steps_cost_the_decode_row() {
        let cost = Cost {
            c1_ms: 10.4,
            slope_ms: 4.4,
            decode_ms: Some(9.0),
            verify_ms: Some(vec![14.0, 17.0, 20.0]),
        };
        assert_eq!(cost.plain_step_ms(), 9.0);
        assert_eq!(cost.step_ms(1), 9.0);
        assert_eq!(
            (cost.step_ms(2), cost.step_ms(3), cost.step_ms(4)),
            (14.0, 17.0, 20.0)
        );
        // Past the last row the slope carries on.
        assert!((cost.step_ms(6) - (20.0 + 2.0 * 4.4)).abs() < 1e-12);
        // No repeats, lookup only: every step is the decode row.
        let (prompt, stream) = no_repeat(60);
        let r = simulate(&prompt, &stream, &SpecConfig::default(), &cost);
        assert_eq!(r.plain_steps, 59);
        assert!((r.predicted_ms - 59.0 * 9.0).abs() < 1e-9);
        assert!((r.plain_ms - 59.0 * 9.0).abs() < 1e-9);
        // The tiny stream: one plain step and one verify of t = 3.
        let r = simulate(&[1, 2, 3, 4], &[1, 2, 3, 4, 5], &fixed(4), &cost);
        assert!((r.predicted_ms - (9.0 + 17.0)).abs() < 1e-9);
        // MTP's plain steps too.
        let m = simulate_with_mtp(&prompt, &stream, &mtp_cfg(&[0.1]), &cost, &mtp(&[1.0]), 1);
        assert!((m.predicted_ms - 59.0 * 9.0).abs() < 1e-9);
    }

    #[test]
    fn bucket_rates_count_each_class_with_lookup_verifying_every_draft() {
        // The prompt holds the run 10..=21; the stream copies it after [10, 11, 12, 13] and then leaves it.
        let run: Vec<u32> = (10..22).collect();
        let prompt: Vec<u32> = [run.as_slice(), &[99, 10, 11, 12]].concat();
        let stream: Vec<u32> = [&run[3..], &[50, 51, 52, 53]].concat();
        let b = bucket_rates(&prompt, &stream, 4, 2, 4);
        // Step 1 drafts [14..=17] from a fresh 4-token match, all kept; step 2 carries the run on: [19..=21, 99]
        // keeps three and the fourth is rejected (the stream has 50). Then nothing matches.
        let fresh = b.by_class[2];
        let cont = b.by_class[3];
        assert_eq!(
            (fresh.drafts, fresh.drafted, fresh.accepted, fresh.tested),
            (1, 4, 4, 4)
        );
        assert_eq!(
            (cont.drafts, cont.drafted, cont.accepted, cont.tested),
            (1, 4, 3, 4)
        );
        assert_eq!((cont.first_kept, cont.full), (1, 0));
        assert_eq!(fresh.full, 1);
        assert_eq!(b.by_class[0].drafts + b.by_class[1].drafts, 0);
        assert_eq!(b.steps, b.plain + 2);
        assert_eq!(b.tokens, stream.len() - 1);
    }

    #[test]
    fn on_qwens_golden_tokens_a_continuing_run_is_lookups_best_class() {
        // A measured property of the model's own output, not a rule of the code: drafts that carry on a run are kept
        // far more often than 2-token matches (`lookup-sim --buckets` prints the figures).
        let mut total = BucketReport::default();
        for s in load_golden(&golden_dir()).unwrap() {
            total.add(&bucket_rates(&s.prompt, &s.tokens, 4, 2, 4));
        }
        let rate = |c: &BucketCount| c.accepted as f64 / c.tested as f64;
        let [short, three, long, cont] = &total.by_class;
        assert!(rate(cont) > rate(long) && rate(long) > rate(three) && rate(three) > rate(short));
        assert!(rate(cont) > 0.85 && rate(short) < 0.6, "{total:?}");
    }

    /// Continuing runs that are right, between 2-token matches that are wrong: lookup's classes tell them apart.
    fn runs_and_noise() -> (Vec<u32>, Vec<u32>) {
        let mut prompt = Vec::new();
        let mut stream = Vec::new();
        for r in 0..16u32 {
            let run: Vec<u32> = (0..32).map(|i| 1000 + r * 100 + i).collect();
            prompt.extend_from_slice(&run);
            prompt.extend_from_slice(&[7, 8, 9 + r]);
            // The stream copies the run, then [7, 8] three times, each followed by a new token: a 2-token match
            // that drafts what followed [7, 8] last time, wrongly.
            stream.extend_from_slice(&run);
            stream.push(5000 + r);
            for j in 0..3 {
                stream.extend_from_slice(&[7, 8, 9000 + r * 10 + j]);
            }
        }
        (prompt, stream)
    }

    #[test]
    fn per_class_rates_give_lookup_the_steps_its_good_drafts_earn() {
        let (prompt, stream) = runs_and_noise();
        let cfg = SpecConfig {
            source_choice: SourceChoice::Best,
            ..mtp_cfg(&[1.0])
        };
        // MTP always right, but its pass costs: lookup's free drafts win only where its rate is near 1 too.
        let m = mtp(&[1.0]);
        let sim = |v: Variant| {
            simulate_variant(
                &prompt,
                &stream,
                stream.len(),
                &cfg,
                &Cost::default(),
                Some((&m, 7)),
                v,
            )
        };
        let before = sim(Variant::BEFORE);
        let after = sim(Variant::AFTER);
        let lookup = |r: &SimResult| r.drafted - r.mtp_drafted;
        assert!(
            lookup(&after) > lookup(&before) + 50,
            "{before:?} {after:?}"
        );
        assert!(
            after.predicted_ms < before.predicted_ms,
            "{before:?} {after:?}"
        );
    }

    #[test]
    fn predict_runs_applies_the_ratio_to_the_measured_figure() {
        let sim = |tokens: usize, ms: f64| SimResult {
            tokens,
            predicted_ms: ms,
            ..run_sim(
                &[],
                &[],
                0,
                &SpecConfig::default(),
                &Cost::default(),
                None,
                None,
                Variant::default(),
            )
        };
        // Lookup-only simulates 100 tok/s, MTP 120: a run measured at 90 predicts 108, and the calibration error is
        // (100 - 90) / 90.
        let out = predict_runs(
            &[
                (sim(120, 1000.0), sim(100, 1000.0)),
                (sim(80, 1000.0), sim(100, 1000.0)),
            ],
            &[90.0, 110.0],
        );
        assert!((out[0].0 - 108.0).abs() < 1e-9, "{out:?}");
        assert!((out[0].1 - 10.0 / 90.0).abs() < 1e-12);
        assert!((out[1].0 - 88.0).abs() < 1e-9);
        assert!((out[1].1 - (-10.0 / 110.0)).abs() < 1e-12);
        // A run without a simulated rate predicts nothing.
        assert_eq!(
            predict_runs(&[(sim(1, 1.0), sim(0, 0.0))], &[90.0]),
            vec![(0.0, 0.0)]
        );
    }

    #[test]
    fn simulate_with_mtp_is_deterministic_per_seed() {
        let (prompt, stream) = no_repeat(200);
        let cfg = mtp_cfg(&[0.7, 0.6, 0.5]);
        let m = mtp(&[0.7, 0.5, 0.4, 0.3]);
        let c = measured_cost();
        let a = simulate_with_mtp(&prompt, &stream, &cfg, &c, &m, 11);
        assert_eq!(a, simulate_with_mtp(&prompt, &stream, &cfg, &c, &m, 11));
        assert_eq!(a.tokens, a.plain_steps + a.accepted + a.verify_steps);
        assert!(a.mtp_accepted <= a.mtp_drafted && a.mtp_steps > 0, "{a:?}");
        let other = simulate_with_mtp(&prompt, &stream, &cfg, &c, &m, 12);
        assert_ne!(a, other);
    }

    fn verify_cost_doc() -> serde_json::Value {
        let row = |path: &str, ctx: u64, t: u64, ms: f64| serde_json::json!({"path": path, "ctx": ctx, "t": t, "mean_ms": ms, "min_ms": ms, "misses_per_step": 1.0});
        let mut records = vec![row("decode", 2048, 1, 10.0), row("decode", 64, 1, 9.0)];
        for (t, ms) in [
            (2, 14.0),
            (3, 18.0),
            (4, 22.0),
            (5, 26.0),
            (8, 41.0),
            (16, 81.0),
        ] {
            records.push(row("verify", 2048, t, ms));
            records.push(row("prefill", 2048, t, 99.0));
        }
        records.push(row("verify", 64, 2, 1.0));
        serde_json::json!({"ctx": 65536, "records": records})
    }

    #[test]
    fn the_verify_cost_rows_at_a_context_make_the_cost() {
        let c = Cost::from_verify_cost(&verify_cost_doc(), 2048).unwrap();
        assert_eq!(c.decode_ms, Some(10.0));
        assert_eq!(c.c1_ms, 10.0);
        assert!((c.slope_ms - 4.0).abs() < 1e-9, "{}", c.slope_ms);
        // t = 6 and 7 are interpolated between 5 and 8; the table runs to 16.
        assert_eq!(c.step_ms(5), 26.0);
        assert_eq!((c.step_ms(6), c.step_ms(7)), (31.0, 36.0));
        assert_eq!((c.step_ms(8), c.step_ms(16)), (41.0, 81.0));
        assert_eq!(c.verify_ms.as_ref().map(Vec::len), Some(15));
        let err = Cost::from_verify_cost(&verify_cost_doc(), 4096)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no decode row at ctx 4096"), "{err}");
    }

    fn aggregate(commit: &str, runs: &[(&str, &str, f64)]) -> String {
        let runs: Vec<_> = runs
            .iter()
            .map(|(run, mode, tok_s)| {
                serde_json::json!({"run": run, "mode": mode, "decode_steps_tok_s": tok_s})
            })
            .collect();
        serde_json::json!({"test": "serve-speed-spec", "commit": commit, "date": "d", "runs": runs})
            .to_string()
    }

    fn write_tmp(name: &str, lines: &[String]) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("artemis-{name}-{}.jsonl", std::process::id()));
        std::fs::write(&path, lines.join("\n")).unwrap();
        path
    }

    #[test]
    fn the_measured_figures_come_from_the_gates_aggregate() {
        let per_run =
            serde_json::json!({"test": "serve-speed-spec run", "mode": "on", "run": "write:1234",
                                         "figures": {"decode_tok_s": 999.0}})
            .to_string();
        let path = write_tmp(
            "measured",
            &[
                per_run,
                aggregate(
                    "c1",
                    &[
                        ("write-1234", "off", 90.0),
                        ("write-1234", "on", 95.5),
                        ("edit-1235", "on", 130.0),
                    ],
                ),
                String::new(),
            ],
        );
        let m = load_measured(&path, None).unwrap();
        std::fs::remove_file(&path).unwrap();
        // The aggregate's decode_steps_tok_s of mode "on", not a per-run line's figure.
        assert_eq!(m.len(), 2);
        assert_eq!((m["write-1234"], m["edit-1235"]), (95.5, 130.0));
    }

    #[test]
    fn two_aggregates_are_refused_unless_a_commit_picks_one() {
        let path = write_tmp(
            "two-aggs",
            &[
                aggregate("c1", &[("write-1234", "on", 95.0)]),
                aggregate("c2", &[("write-1234", "on", 120.0)]),
            ],
        );
        let err = load_measured(&path, None).unwrap_err().to_string();
        assert!(
            err.contains("2 ") && err.contains("--measured-commit") && err.contains("c1"),
            "{err}"
        );
        assert_eq!(
            load_measured(&path, Some("c1")).unwrap()["write-1234"],
            95.0
        );
        assert_eq!(
            load_measured(&path, Some("c2")).unwrap()["write-1234"],
            120.0
        );
        let err = load_measured(&path, Some("c3")).unwrap_err().to_string();
        std::fs::remove_file(&path).unwrap();
        assert!(err.contains("0 ") && err.contains("c3"), "{err}");
    }

    #[test]
    fn the_real_5a_aggregate_reads_when_the_evidence_is_there() {
        // `ARTEMIS_LOOKUP_RESULTS` names the file; the private tree's copy is the fallback.
        let named = std::env::var_os("ARTEMIS_LOOKUP_RESULTS").map(std::path::PathBuf::from);
        // A path somebody set is a request to run the test, so a missing file is a failure, not a skip.
        if let Some(p) = named.as_ref().filter(|p| !p.exists()) {
            panic!(
                "ARTEMIS_LOOKUP_RESULTS names {}, which does not exist",
                p.display()
            );
        }
        let path = named.unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../docs/superpowers/evidence/m5/results.jsonl")
        });
        if !path.exists() {
            eprintln!(
                "skipped: {} does not exist; set ARTEMIS_LOOKUP_RESULTS to the m5 results.jsonl to run this test",
                path.display()
            );
            return;
        }
        let m = load_measured(&path, Some("462cdf5")).unwrap();
        assert_eq!(m.len(), 6);
        let med = speed::median(&m.values().copied().collect::<Vec<_>>());
        assert!((med - 98.139).abs() < 0.01, "{med}");
    }

    #[test]
    fn at_the_real_ctx_2048_costs_a_fresh_prior_drafts_one_token() {
        // Decode 11.27 ms, verify t = 2..5 16.23, 19.39, 23.61, 27.76; prior [0.75, 0.70]. Preflight's arithmetic
        // (k = 2 above the bar) used the measured c(3); the controller only knows c(1) plus its slope guess at the
        // first MTP step (k = 1: 0.1037, k = 2: 0.1012), and once c(2) is measured k = 2 scores 0.0906. So k = 1.
        let mut cost = Cost::flat(11.27, 3.8);
        cost.decode_ms = Some(11.27);
        cost.verify_ms = Some(vec![16.23, 19.39, 23.61, 27.76]);
        let cfg = SpecConfig {
            cost_slope_ms: cost.slope_ms,
            ..mtp_cfg(&[0.75, 0.70])
        };
        let (prompt, stream) = no_repeat(5);
        let r = simulate_with_mtp(&prompt, &stream, &cfg, &cost, &mtp(&[1.0]), 1);
        assert_eq!((r.mtp_steps, r.mtp_drafted), (1, 1), "{r:?}");
        // Prior 0.5 does not draft at all.
        let r = simulate_with_mtp(&prompt, &stream, &mtp_cfg(&[0.5]), &cost, &mtp(&[1.0]), 1);
        assert_eq!(r.mtp_steps, 0, "{r:?}");
    }

    #[test]
    fn the_mtp_report_judges_the_checkpoint_at_1_8() {
        let path =
            std::env::temp_dir().join(format!("artemis-mtp-report-{}.jsonl", std::process::id()));
        std::fs::write(&path, aggregate("c1", &[("a-1", "on", 90.0)])).unwrap();
        let (prompt, tokens) = no_repeat(300);
        let streams = vec![(
            Stream {
                id: "a-1".into(),
                max_tokens: 300,
                prompt,
                tokens,
            },
            "record",
        )];
        let a = LookupSimArgs {
            mtp_rates: vec![0.8, 0.7, 0.6, 0.5],
            measured: Some(path.clone()),
            seeds: 4,
            ..args(false, &[])
        };
        let cost = measured_cost();
        let v = mtp_report(&a, &streams, &config(&a, &cost).unwrap(), &cost).unwrap();
        // A missing run is named.
        let missing = vec![(
            Stream {
                id: "b-2".into(),
                max_tokens: 3,
                prompt: vec![1],
                tokens: vec![1, 2, 3],
            },
            "record",
        )];
        let err = mtp_report(&a, &missing, &config(&a, &cost).unwrap(), &cost)
            .unwrap_err()
            .to_string();
        std::fs::remove_file(&path).unwrap();
        assert!(err.contains("b-2"), "{err}");
        assert_eq!(v["checkpoint_ms_per_draft"], 1.8);
        let ms: Vec<f64> = v["medians"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["ms_per_draft"].as_f64().unwrap())
            .collect();
        assert_eq!(ms, vec![1.2, 1.8]);
        // No lookup draft in a no-repeat stream, so lookup alone is plain (the calibration is a ratio of 1) and MTP
        // at these rates pays.
        assert!(
            v["checkpoint_gain_over_lookup"].as_f64().unwrap() > 0.05,
            "{v}"
        );
        assert!(
            v["medians"][0]["predicted_median"].as_f64()
                > v["medians"][1]["predicted_median"].as_f64()
        );
        // Bad rates are refused.
        for bad in [vec![], vec![1.5]] {
            let a = LookupSimArgs {
                mtp_rates: bad,
                ..args(false, &[])
            };
            let err = mtp_report(&a, &streams, &SpecConfig::default(), &cost)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("--measured") || err.contains("--mtp-rates"),
                "{err}"
            );
        }
    }

    #[test]
    fn the_sensitivity_line_uses_each_prompts_own_depth_1_rate() {
        let measured = write_tmp(
            "sens-measured",
            &[aggregate("c1", &[("write-1234", "on", 90.0)])],
        );
        let probe = write_tmp(
            "sens-probe",
            &[serde_json::json!({"depth1_per_prompt": {"write": 0.4}}).to_string()],
        );
        let (prompt, tokens) = no_repeat(300);
        let streams = vec![(
            Stream {
                id: "write-1234".into(),
                max_tokens: 300,
                prompt,
                tokens,
            },
            "record",
        )];
        let a = LookupSimArgs {
            mtp_rates: vec![0.8, 0.7, 0.6, 0.5],
            measured: Some(measured.clone()),
            mtp_probe: Some(probe.clone()),
            seeds: 4,
            ..args(false, &[])
        };
        let cost = measured_cost();
        let v = mtp_report(&a, &streams, &config(&a, &cost).unwrap(), &cost).unwrap();
        // A prompt the probe lacks is refused.
        let other = vec![(
            Stream {
                id: "edit-1234".into(),
                max_tokens: 3,
                prompt: vec![1],
                tokens: vec![1, 2, 3],
            },
            "record",
        )];
        let a2 = LookupSimArgs {
            measured: Some(write_tmp(
                "sens-m2",
                &[aggregate("c1", &[("edit-1234", "on", 90.0)])],
            )),
            ..a.clone()
        };
        let err = mtp_report(&a2, &other, &config(&a2, &cost).unwrap(), &cost)
            .unwrap_err()
            .to_string();
        for p in [&measured, &probe] {
            std::fs::remove_file(p).unwrap();
        }
        assert!(err.contains("edit"), "{err}");
        let sens = &v["sensitivity"];
        // a1 = 0.4 against a pooled 0.8: a2.. halved, and the prediction falls below the pooled one.
        assert_eq!(
            sens["per_stream"][0]["depth_rates"],
            serde_json::json!([0.4, 0.35, 0.3, 0.25])
        );
        assert!(
            sens["predicted_median"].as_f64().unwrap()
                < v["checkpoint_predicted_median"].as_f64().unwrap(),
            "{v}"
        );
        assert!(
            v["checkpoint_verdict"].as_str().unwrap().contains("go on"),
            "{v}"
        );
    }

    #[test]
    fn the_checkpoint_says_stop_when_mtp_gains_under_3_percent() {
        let path = write_tmp("stop-measured", &[aggregate("c1", &[("a-1", "on", 90.0)])]);
        let (prompt, tokens) = no_repeat(300);
        let streams = vec![(
            Stream {
                id: "a-1".into(),
                max_tokens: 300,
                prompt,
                tokens,
            },
            "record",
        )];
        // MTP that is never right and never drafts (prior 0): no gain.
        let a = LookupSimArgs {
            mtp_rates: vec![0.0],
            measured: Some(path.clone()),
            seeds: 2,
            ..args(false, &[])
        };
        let cost = measured_cost();
        let v = mtp_report(&a, &streams, &config(&a, &cost).unwrap(), &cost).unwrap();
        std::fs::remove_file(&path).unwrap();
        let verdict = v["checkpoint_verdict"].as_str().unwrap();
        assert!(verdict.contains("< 3%") && verdict.contains("STOP"), "{v}");
        assert!(v["checkpoint_gain_over_lookup"].as_f64().unwrap() < 0.03);
    }

    /// A `Verifier` over a recorded stream: every row's greedy pick is the stream's next token, and its MTP draws its
    /// kept count from the simulator's own model and generator, so the same seed gives the same outcomes. It logs what
    /// each step did.
    struct StreamVerifier<'a> {
        stream: &'a [u32],
        /// The stream index of the token the state is one short of.
        at: usize,
        mtp: &'a MtpModel,
        rng: SplitMix,
        drafted_by_mtp: bool,
        log: Vec<(Source, usize)>,
    }

    const PIN_VOCAB: usize = 4096;

    impl StreamVerifier<'_> {
        fn row_at(&self, i: usize) -> Vec<f32> {
            let mut row = vec![0.0; PIN_VOCAB];
            row[self.stream[(self.at + 1 + i).min(self.stream.len() - 1)] as usize] = 1.0;
            row
        }
    }

    impl engine::spec::Verifier for StreamVerifier<'_> {
        type Error = ();
        fn decode(&mut self, _token: u32) -> Result<Vec<f32>, ()> {
            self.log.push((Source::Lookup, 0));
            let row = self.row_at(0);
            self.at += 1;
            Ok(row)
        }
        fn verify(&mut self, tokens: &[u32]) -> Result<(), ()> {
            let source = if std::mem::take(&mut self.drafted_by_mtp) {
                Source::Mtp
            } else {
                Source::Lookup
            };
            self.log.push((source, tokens.len() - 1));
            Ok(())
        }
        fn row(&mut self, i: usize) -> Result<Vec<f32>, ()> {
            Ok(self.row_at(i))
        }
        fn commit(&mut self, kept: usize) -> Result<(), ()> {
            self.at += kept;
            Ok(())
        }
        fn n_vocab(&self) -> usize {
            PIN_VOCAB
        }
        fn misses(&self) -> u64 {
            0
        }
        fn miss_bytes(&self) -> u64 {
            0
        }
        fn has_mtp(&self) -> bool {
            true
        }
        fn mtp_draft(&mut self, _x: u32, k: usize) -> Result<Vec<u32>, ()> {
            self.drafted_by_mtp = true;
            let mut kept = 0;
            while kept < k && self.rng.next_f64() < self.mtp.rate(kept) {
                kept += 1;
            }
            // `kept` right tokens, then a wrong one (never compared where the stream ends first).
            Ok((0..k)
                .map(|i| {
                    let right = self.stream.get(self.at + 1 + i).copied().unwrap_or(0);
                    if i < kept {
                        right
                    } else {
                        right + 1000
                    }
                })
                .collect())
        }
    }

    #[test]
    fn the_simulator_and_the_step_loop_choose_the_same_sources_and_ks() {
        use engine::spec::{Next, SpecDelta, SpecState, StepTimed};
        // Repeats lookup finds, runs it loses, and fresh tokens only MTP can be asked about; probes at 4.
        let prompt: Vec<u32> = [1, 2, 3].repeat(10);
        let stream: Vec<u32> = (0..70u32)
            .flat_map(|i| match i % 7 {
                0..=2 => vec![1, 2, 3],
                3 => vec![40 + i, 50 + i],
                _ => vec![60 + i],
            })
            .collect();
        let cost = measured_cost();
        let model = MtpModel {
            depth_rates: vec![0.85, 0.6, 0.5],
            ms_per_draft: 1.8,
            ms_per_pass: 0.0,
        };
        // Steps each source took, over all of a choice's runs: the scenario is not trivial for either choice.
        let mut taken = std::collections::HashMap::new();
        for (choice, prior) in [SourceChoice::Priority, SourceChoice::Best]
            .into_iter()
            .flat_map(|c| [vec![0.89, 0.81, 0.74, 0.78], vec![0.3]].map(|p| (c, p)))
        {
            for seed in [1u64, 2, 3] {
                let cfg = SpecConfig {
                    probe_every: 4,
                    source_choice: choice,
                    ..mtp_cfg(&prior)
                };
                let (sim, want) =
                    simulate_with_mtp_traced(&prompt, &stream, &cfg, &cost, &model, seed);

                let mut v = StreamVerifier {
                    stream: &stream,
                    at: 0,
                    mtp: &model,
                    rng: SplitMix(seed),
                    drafted_by_mtp: false,
                    log: Vec::new(),
                };
                let (c, m) = (cost.clone(), model.clone());
                let mut state =
                    SpecState::new(cfg.clone()).with_clock(Box::new(move |t| match t {
                        StepTimed::Plain => c.plain_step_ms(),
                        StepTimed::Verify { tokens } => c.step_ms(tokens),
                        StepTimed::MtpDraft { k } => m.draft_ms(k),
                    }));
                let mut lookup = state.drafter();
                lookup.observe(&prompt);
                lookup.observe(&stream[..1]);
                let mut delta = SpecDelta::default();
                let mut out = 1;
                while out < stream.len() {
                    let k_max = stream.len().saturating_sub(out + 1);
                    let stepped = state
                        .step(
                            &mut v,
                            &mut *lookup,
                            stream[out - 1],
                            k_max,
                            &mut |row| {
                                let y = row.iter().position(|&p| p > 0.5).unwrap() as u32;
                                assert_eq!(y, stream[out]);
                                out += 1;
                                Ok(if out >= stream.len() {
                                    Next::End
                                } else {
                                    Next::Go(y)
                                })
                            },
                            &mut delta,
                        )
                        .unwrap();
                    if stepped.ended {
                        break;
                    }
                }
                let got: Vec<(Source, usize)> = want.iter().map(|&(s, k, _)| (s, k)).collect();
                assert_eq!(v.log, got, "{choice:?} prior {prior:?} seed {seed}");
                // The counts agree too.
                let n = |s: Source| got.iter().filter(|&&(g, k)| g == s && k > 0).count();
                let t: &mut (usize, usize) = taken.entry(choice as u8).or_default();
                *t = (t.0 + n(Source::Lookup), t.1 + n(Source::Mtp));
                assert_eq!(delta.steps as usize, sim.verify_steps);
                assert_eq!(delta.plain_steps as usize, sim.plain_steps);
                assert_eq!(delta.drafted as usize, sim.drafted);
                assert_eq!(delta.accepted as usize, sim.accepted);
                assert_eq!(delta.drafted_mtp as usize, sim.mtp_drafted);
                assert_eq!(delta.accepted_mtp as usize, sim.mtp_accepted);
                assert_eq!(delta.probes as usize, sim.mtp_probes);
                assert_eq!(
                    want.iter().filter(|w| w.2).count(),
                    sim.mtp_probes,
                    "probes"
                );
            }
        }
        assert_eq!(taken.len(), 2);
        for (choice, (lookup, mtp)) in taken {
            assert!(
                lookup > 0 && mtp > 0,
                "choice {choice}: lookup {lookup} mtp {mtp}"
            );
        }
    }
}
