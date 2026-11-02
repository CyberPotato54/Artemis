//! The draft-length controller: per source, from the accept rate and the measured step costs, how
//! many draft tokens the next step should verify. 0 is plain decode.
use super::{LookupMatch, Source, SpecConfig, MAX_VERIFY};

/// The half-life, in steps, of each step size's average time.
const COST_HALF_LIFE: f64 = 16.0;

/// How many steps a measured size's average stays in use after it was last observed: 4 half-lives. A size is only
/// re-measured when it is chosen, so a stale pessimistic value (a cold sample) would never correct itself; past this
/// it falls back to the line, and an optimistic line gets the size chosen and measured again.
const COST_FRESH_STEPS: u64 = 64;

/// What the controller has measured of its own, for the first-run setup: `None` while only the config's priors stand.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeasuredCosts {
    /// The least-squares cost per extra token of a step, once two step sizes are fresh.
    pub slope_ms: Option<f64>,
    /// The MTP draft pass's time at `k = 1`, while it is fresh.
    pub mtp_d1_ms: Option<f64>,
}

pub struct Controller {
    max_draft: usize,
    min_gain: f64,
    cost_slope_ms: f64,
    /// The weight an old accept outcome keeps after one more draft token.
    accept_keep: f64,
    cost_keep: f64,
    /// Lookup's per-token accept rate over all its drafts: the totals' figure, and the rate of a draft whose source
    /// gave no class.
    accept: f64,
    /// Lookup's per-token accept rate by `LookupMatch::class`; `None` until the class's first outcome, and until then
    /// the pooled rate stands in (a class starts from what lookup has shown so far, not from the prior).
    lookup_rates: [Option<f64>; LookupMatch::CLASSES],
    /// MTP's accept rate at draft depth `i + 1`, given the drafts before it were kept.
    mtp_rates: [f64; MAX_VERIFY - 1],
    /// The prior for the draft pass's time per draft token.
    mtp_cost_ms: f64,
    /// `draft_cost[k]`: the average time of the MTP's pass for `k` drafts.
    draft_cost: [Option<f64>; MAX_VERIFY + 1],
    /// The step at which each `draft_cost[k]` was last observed.
    draft_seen: [u64; MAX_VERIFY + 1],
    probe_every: usize,
    /// Plain steps since MTP last drafted.
    plain_since_mtp: usize,
    /// `cost[t]`: the average time of a step of `t` tokens; `cost[0]` is never used.
    cost: [Option<f64>; MAX_VERIFY + 1],
    /// The step at which each `cost[t]` was last observed.
    cost_seen: [u64; MAX_VERIFY + 1],
    /// Steps observed (valid `observe_cost` calls), the clock the averages expire on.
    steps: u64,
}

impl Controller {
    /// `cfg` must pass `SpecConfig::check`.
    pub fn new(cfg: &SpecConfig) -> Controller {
        let mut mtp_rates = [0.0; MAX_VERIFY - 1];
        let last = cfg.mtp_accept_prior.last().copied().unwrap_or(0.0);
        for (i, r) in mtp_rates.iter_mut().enumerate() {
            *r = cfg.mtp_accept_prior.get(i).copied().unwrap_or(last);
        }
        Controller {
            max_draft: cfg.max_draft,
            min_gain: cfg.min_gain,
            cost_slope_ms: cfg.cost_slope_ms,
            accept_keep: 0.5f64.powf(1.0 / cfg.accept_half_life),
            cost_keep: 0.5f64.powf(1.0 / COST_HALF_LIFE),
            accept: cfg.accept_prior,
            lookup_rates: [None; LookupMatch::CLASSES],
            mtp_rates,
            mtp_cost_ms: cfg.mtp_cost_ms,
            draft_cost: [None; MAX_VERIFY + 1],
            draft_seen: [0; MAX_VERIFY + 1],
            probe_every: cfg.probe_every,
            plain_since_mtp: 0,
            cost: [None; MAX_VERIFY + 1],
            cost_seen: [0; MAX_VERIFY + 1],
            steps: 0,
        }
    }

    /// How many draft tokens to verify: up to `available`, `k_max` and `max_draft`, whichever is smallest. 0 until a
    /// plain step has been timed, and when even the best length gains less than `min_gain`. Ties go to the smaller.
    /// Lookup's score is `E(k) / c(k + 1)` with one geometric rate; MTP's is `E_m(k) / (d(k) + c(k + 1))` with a rate
    /// per depth.
    pub fn choose(&self, source: Source, available: usize, k_max: usize) -> usize {
        self.choose_scored(source, available, k_max).0
    }

    /// `choose`, with the winning length's score `E(k) / (d(k) + c(k + 1))` (0.0 when it chooses 0). The best-of
    /// source choice compares these scores.
    pub fn choose_scored(&self, source: Source, available: usize, k_max: usize) -> (usize, f64) {
        self.choose_scored_as(source, None, available, k_max)
    }

    /// `choose_scored` for a lookup draft of class `class`: lookup's rate is that class's (`lookup_rate`). MTP's
    /// choice ignores `class`.
    pub fn choose_scored_as(
        &self,
        source: Source,
        class: Option<LookupMatch>,
        available: usize,
        k_max: usize,
    ) -> (usize, f64) {
        let lookup_rate = self.lookup_rate(class);
        let Some(c1) = self.cost_ms(1) else {
            return (0, 0.0);
        };
        let top = available.min(k_max).min(self.max_draft);
        let (mut best_k, mut best) = (0, 0.0);
        // E(k) = 1 + a + ... + a^k; for MTP, a is the running product of the depth rates.
        let (mut expected, mut power) = (1.0, 1.0);
        for k in 1..=top {
            let (a, draft_ms) = match source {
                Source::Lookup => (lookup_rate, 0.0),
                Source::Mtp => (self.mtp_rates[k - 1], self.draft_cost_ms(k)),
            };
            power *= a;
            expected += power;
            let Some(ck) = self.cost_ms(k + 1) else {
                break;
            };
            let score = expected / (draft_ms + ck);
            if score > best {
                (best_k, best) = (k, score);
            }
        }
        if best < self.min_gain / c1 {
            (0, 0.0)
        } else {
            (best_k, best)
        }
    }

    /// A step compared `tested` draft tokens with the tokens picked and `accepted` of them matched. `tested` is
    /// `accepted`, plus 1 if a draft token mismatched. Draft tokens after the mismatch, or cut off by an `End` (a
    /// stop), were never compared and are untested. At most one rejection counts, so a larger `tested` than that
    /// counts as `accepted + 1`. For MTP the outcome at draft `i` updates depth `i`'s rate alone.
    pub fn observe_accept(&mut self, source: Source, tested: usize, accepted: usize) {
        self.observe_accept_as(source, None, tested, accepted);
    }

    /// `observe_accept` for a lookup draft of class `class`: the outcome moves the pooled rate and, with a class, that
    /// class's rate too. MTP's outcomes ignore `class`.
    pub fn observe_accept_as(
        &mut self,
        source: Source,
        class: Option<LookupMatch>,
        tested: usize,
        accepted: usize,
    ) {
        let accepted = accepted.min(tested);
        let rejected = (tested - accepted).min(1);
        let keep = self.accept_keep;
        let blend = |rate: &mut f64, hit: f64| *rate = keep * *rate + (1.0 - keep) * hit;
        if source == Source::Lookup && accepted + rejected > 0 {
            if let Some(m) = class {
                let pooled = self.accept;
                self.lookup_rates[m.class()].get_or_insert(pooled);
            }
        }
        for i in 0..accepted + rejected {
            let hit = if i < accepted { 1.0 } else { 0.0 };
            match source {
                Source::Lookup => {
                    blend(&mut self.accept, hit);
                    if let Some(rate) = class.and_then(|m| self.lookup_rates[m.class()].as_mut()) {
                        blend(rate, hit);
                    }
                }
                Source::Mtp => match self.mtp_rates.get_mut(i) {
                    Some(r) => blend(r, hit),
                    None => break,
                },
            }
        }
    }

    /// Lookup's per-token accept rate for a draft of class `class`; the pooled rate without one.
    pub fn lookup_rate(&self, class: Option<LookupMatch>) -> f64 {
        class
            .and_then(|m| self.lookup_rates[m.class()])
            .unwrap_or(self.accept)
    }

    /// The MTP's draft pass for `k` drafts took `ms`.
    pub fn observe_draft_cost(&mut self, k: usize, ms: f64) {
        if !(1..=MAX_VERIFY).contains(&k) || !ms.is_finite() || ms < 0.0 {
            return;
        }
        // An expired value is absent: the new sample replaces it whole, not a blend into the stale average.
        self.draft_cost[k] = Some(match self.fresh_draft_cost(k) {
            Some(old) => self.cost_keep * old + (1.0 - self.cost_keep) * ms,
            None => ms,
        });
        self.draft_seen[k] = self.steps;
    }

    /// `draft_cost[k]`, while it is fresh.
    fn fresh_draft_cost(&self, k: usize) -> Option<f64> {
        let ms = (*self.draft_cost.get(k)?)?;
        (self.steps - self.draft_seen[k] <= COST_FRESH_STEPS).then_some(ms)
    }

    /// The draft pass for `k` drafts' time: measured and fresh, else the least-squares line through the fresh measured
    /// `k`s once there are two (never below 0), else `k * mtp_cost_ms`.
    pub fn draft_cost_ms(&self, k: usize) -> f64 {
        if let Some(ms) = self.fresh_draft_cost(k) {
            return ms;
        }
        let points =
            || (0..=MAX_VERIFY).filter_map(|k| self.fresh_draft_cost(k).map(|ms| (k as f64, ms)));
        let count = points().count();
        if count < 2 {
            return k as f64 * self.mtp_cost_ms;
        }
        let n = count as f64;
        let mean_k = points().map(|p| p.0).sum::<f64>() / n;
        let mean_ms = points().map(|p| p.1).sum::<f64>() / n;
        let spread: f64 = points().map(|p| (p.0 - mean_k).powi(2)).sum();
        let cross: f64 = points().map(|p| (p.0 - mean_k) * (p.1 - mean_ms)).sum();
        let slope = cross / spread;
        (mean_ms + slope * (k as f64 - mean_k)).max(0.0)
    }

    /// Whether a probe is due: `probe_every` plain steps have run since MTP last drafted. The step loop asks only when
    /// MTP's choice is 0.
    pub fn probe_due(&self) -> bool {
        self.plain_since_mtp >= self.probe_every
    }

    /// A plain step ran.
    pub fn observe_plain(&mut self) {
        self.plain_since_mtp = self.plain_since_mtp.saturating_add(1);
    }

    /// MTP drafted: the probe count starts again.
    pub fn observe_mtp_drafted(&mut self) {
        self.plain_since_mtp = 0;
    }

    /// A step of `t` tokens took `ms`, from the decode or verify call to the end of the step.
    pub fn observe_cost(&mut self, t: usize, ms: f64) {
        if !(1..=MAX_VERIFY).contains(&t) || !ms.is_finite() || ms < 0.0 {
            return;
        }
        self.steps += 1;
        // An expired value is absent: the new sample replaces it whole, not a blend into the stale average.
        self.cost[t] = Some(match self.fresh_cost(t) {
            Some(old) => self.cost_keep * old + (1.0 - self.cost_keep) * ms,
            None => ms,
        });
        self.cost_seen[t] = self.steps;
    }

    /// `cost[t]`, while it is fresh; `cost[1]` never expires.
    fn fresh_cost(&self, t: usize) -> Option<f64> {
        let ms = (*self.cost.get(t)?)?;
        (t == 1 || self.steps - self.cost_seen[t] <= COST_FRESH_STEPS).then_some(ms)
    }

    /// The current per-token accept probability; MTP's is its depth 1.
    pub fn accept_rate(&self, source: Source) -> f64 {
        match source {
            Source::Lookup => self.accept,
            Source::Mtp => self.mtp_rates[0],
        }
    }

    /// MTP's accept rates at depths 1 to 15.
    pub fn depth_rates(&self) -> [f64; MAX_VERIFY - 1] {
        self.mtp_rates
    }

    /// A step of `t` tokens' time: measured and fresh, or `c(1) + (t - 1) * m`. `None` until `c(1)` is measured.
    pub fn cost_ms(&self, t: usize) -> Option<f64> {
        let c1 = self.cost[1]?;
        if let Some(ms) = self.fresh_cost(t) {
            return Some(ms);
        }
        Some(c1 + t.saturating_sub(1) as f64 * self.slope())
    }

    /// `m`: the least-squares slope over the fresh measured sizes once there are two, else the config's; never below 0.
    fn slope(&self) -> f64 {
        self.measured_slope().unwrap_or(self.cost_slope_ms)
    }

    /// The least-squares slope over the fresh measured sizes, once there are two; never below 0.
    fn measured_slope(&self) -> Option<f64> {
        let points =
            || (0..=MAX_VERIFY).filter_map(|t| self.fresh_cost(t).map(|ms| (t as f64, ms)));
        let count = points().count();
        if count < 2 {
            return None;
        }
        let n = count as f64;
        let mean_t = points().map(|p| p.0).sum::<f64>() / n;
        let mean_ms = points().map(|p| p.1).sum::<f64>() / n;
        let spread: f64 = points().map(|p| (p.0 - mean_t).powi(2)).sum();
        let cross: f64 = points().map(|p| (p.0 - mean_t) * (p.1 - mean_ms)).sum();
        Some((cross / spread).max(0.0))
    }

    /// The costs this controller measured: the slope once two step sizes are fresh, and the `k = 1` draft cost.
    pub fn measured(&self) -> MeasuredCosts {
        MeasuredCosts {
            slope_ms: self.measured_slope(),
            mtp_d1_ms: self.fresh_draft_cost(1),
        }
    }

    #[cfg(test)]
    fn set_accept_rate(&mut self, a: f64) {
        self.accept = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SourceChoice;

    /// An explicit config: the controller's tests never lean on the defaults, which may be retuned.
    fn cfg() -> SpecConfig {
        SpecConfig {
            enabled: true,
            max_draft: 4,
            ngram_min: 2,
            ngram_max: 4,
            min_gain: 1.03,
            accept_half_life: 32.0,
            cost_slope_ms: 2.5,
            accept_prior: 0.5,
            fixed_draft: None,
            sources: vec![Source::Lookup, Source::Mtp],
            probe_every: 64,
            mtp_window: 2048,
            mtp_ctx: 8192,
            mtp_ring: 256,
            mtp_accept_prior: vec![0.7, 0.6, 0.5],
            mtp_cost_ms: 1.8,
            source_choice: SourceChoice::Priority,
            mtp_head_rows: 0,
        }
    }

    fn timed(rate: f64) -> Controller {
        let mut c = Controller::new(&cfg());
        c.observe_cost(1, 10.0);
        c.set_accept_rate(rate);
        c
    }

    #[test]
    fn measured_costs_need_two_sizes() {
        let mut c = Controller::new(&cfg());
        assert_eq!(
            c.measured(),
            MeasuredCosts {
                slope_ms: None,
                mtp_d1_ms: None
            }
        );
        // One size is a point, not a line: the config's slope still stands.
        c.observe_cost(1, 10.0);
        assert_eq!(c.measured().slope_ms, None);
        // Two sizes give the slope between them: 3 ms per extra token.
        c.observe_cost(3, 16.0);
        let m = c.measured();
        assert!((m.slope_ms.unwrap() - 3.0).abs() < 1e-9, "{m:?}");
        assert_eq!(m.mtp_d1_ms, None);
        // The draft cost at k = 1 shows once it is observed; other depths do not stand in for it.
        c.observe_draft_cost(2, 3.0);
        assert_eq!(c.measured().mtp_d1_ms, None);
        c.observe_draft_cost(1, 1.25);
        assert_eq!(c.measured().mtp_d1_ms, Some(1.25));
        // A size that has not been seen for COST_FRESH_STEPS steps no longer counts: the line is back to one point.
        for _ in 0..=COST_FRESH_STEPS {
            c.observe_cost(1, 10.0);
        }
        assert_eq!(c.measured().slope_ms, None);
    }

    #[test]
    fn nothing_is_drafted_before_a_plain_step_is_timed() {
        let c = Controller::new(&cfg());
        assert_eq!(c.choose(Source::Lookup, 4, 4), 0);
        assert_eq!(c.cost_ms(1), None);
        assert_eq!(c.cost_ms(3), None);
        // A verify step's time alone doesn't unlock it: c(1) is the anchor.
        let mut v = Controller::new(&cfg());
        v.observe_cost(3, 14.0);
        v.set_accept_rate(0.9);
        assert_eq!(v.choose(Source::Lookup, 4, 4), 0);
        assert_eq!(v.cost_ms(3), None);
    }

    #[test]
    fn the_accept_rate_starts_at_the_prior() {
        assert_eq!(Controller::new(&cfg()).accept_rate(Source::Lookup), 0.5);
        let mut other = cfg();
        other.accept_prior = 0.8;
        assert_eq!(Controller::new(&other).accept_rate(Source::Lookup), 0.8);
    }

    #[test]
    fn the_accept_rate_has_a_half_life_of_32_draft_tokens() {
        let mut c = Controller::new(&cfg());
        c.set_accept_rate(0.0);
        c.observe_accept(Source::Lookup, 32, 32);
        assert!(
            (c.accept_rate(Source::Lookup) - 0.5).abs() < 1e-9,
            "{}",
            c.accept_rate(Source::Lookup)
        );
        // The same 32 tokens in four calls give the same rate.
        let mut d = Controller::new(&cfg());
        d.set_accept_rate(0.0);
        for _ in 0..4 {
            d.observe_accept(Source::Lookup, 8, 8);
        }
        assert!((d.accept_rate(Source::Lookup) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_rejection_pulls_the_rate_down_and_nothing_tested_leaves_it() {
        let mut c = Controller::new(&cfg());
        c.set_accept_rate(1.0);
        c.observe_accept(Source::Lookup, 0, 0);
        assert_eq!(c.accept_rate(Source::Lookup), 1.0);
        // One accepted and the first rejection: two tested tokens. A 1.0 stays 1.0 on a hit, then loses one step's weight.
        c.observe_accept(Source::Lookup, 2, 1);
        let w = 0.5f64.powf(1.0 / 32.0);
        assert!(
            (c.accept_rate(Source::Lookup) - w).abs() < 1e-12,
            "{} vs {w}",
            c.accept_rate(Source::Lookup)
        );
        let mut zero = Controller::new(&cfg());
        zero.set_accept_rate(1.0);
        for _ in 0..32 {
            zero.observe_accept(Source::Lookup, 1, 0);
        }
        assert!((zero.accept_rate(Source::Lookup) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_mid_draft_mismatch_counts_the_accepted_ones_and_one_rejection() {
        // Four drafted, a mismatch at the third: two accepted, one rejection, the fourth untested.
        let mut mid = Controller::new(&cfg());
        mid.observe_accept(Source::Lookup, 3, 2);
        let w = 0.5f64.powf(1.0 / 32.0);
        let mut a = 0.5;
        for hit in [1.0, 1.0, 0.0] {
            a = w * a + (1.0 - w) * hit;
        }
        assert!((mid.accept_rate(Source::Lookup) - a).abs() < 1e-12);
        // A caller that passes every draft proposed gets the same count.
        let mut all = Controller::new(&cfg());
        all.observe_accept(Source::Lookup, 4, 2);
        assert_eq!(
            all.accept_rate(Source::Lookup),
            mid.accept_rate(Source::Lookup)
        );
    }

    #[test]
    fn a_draft_cut_off_by_a_stop_tests_only_the_tokens_reached() {
        // Four drafted, a stop after two accepted: two tests, no rejection.
        let mut cut = Controller::new(&cfg());
        cut.observe_accept(Source::Lookup, 2, 2);
        let w = 0.5f64.powf(1.0 / 32.0);
        let mut a = 0.5;
        for _ in 0..2 {
            a = w * a + (1.0 - w);
        }
        assert!((cut.accept_rate(Source::Lookup) - a).abs() < 1e-12);
        assert!(cut.accept_rate(Source::Lookup) > 0.5);
    }

    #[test]
    fn a_high_accept_rate_drafts_the_most() {
        let c = timed(0.9);
        assert_eq!(c.choose(Source::Lookup, 4, 4), 4);
        assert_eq!(c.choose(Source::Lookup, 2, 4), 2);
        assert_eq!(c.choose(Source::Lookup, 4, 1), 1);
        assert_eq!(c.choose(Source::Lookup, 0, 4), 0);
        assert_eq!(c.choose(Source::Lookup, 4, 0), 0);
    }

    #[test]
    fn max_draft_caps_the_choice() {
        let mut small = cfg();
        small.max_draft = 2;
        let mut c = Controller::new(&small);
        c.observe_cost(1, 10.0);
        c.set_accept_rate(0.9);
        assert_eq!(c.choose(Source::Lookup, 4, 4), 2);
    }

    #[test]
    fn a_gain_inside_the_margin_runs_plain() {
        // k = 1: (1 + a) / 12.5 against 1.03 / 10.
        assert_eq!(timed(0.28).choose(Source::Lookup, 4, 4), 0);
        assert_eq!(timed(0.30).choose(Source::Lookup, 4, 4), 1);
    }

    #[test]
    fn a_steep_cost_runs_plain() {
        let mut steep = cfg();
        steep.cost_slope_ms = 10.0;
        let mut c = Controller::new(&steep);
        c.observe_cost(1, 10.0);
        c.set_accept_rate(0.5);
        assert_eq!(c.choose(Source::Lookup, 4, 4), 0);
    }

    #[test]
    fn ties_go_to_the_smaller_k() {
        // A flat cost and a rate of 0 make every k score 0.1: a tie, and below the 1.03 margin, so plain.
        let mut flat = cfg();
        flat.cost_slope_ms = 0.0;
        let mut f = Controller::new(&flat);
        f.observe_cost(1, 10.0);
        f.set_accept_rate(0.0);
        assert_eq!(f.choose(Source::Lookup, 4, 4), 0);
        // With the margin off (1.0) the tie is accepted, and the smaller k takes it.
        let mut open = flat.clone();
        open.min_gain = 1.0;
        let mut o = Controller::new(&open);
        o.observe_cost(1, 10.0);
        o.set_accept_rate(0.0);
        assert_eq!(o.choose(Source::Lookup, 4, 4), 1);
        // A flat cost with a rate of 1 makes the score grow with k: the largest wins.
        f.set_accept_rate(1.0);
        assert_eq!(f.choose(Source::Lookup, 4, 4), 4);
    }

    #[test]
    fn the_slope_comes_from_the_config_until_there_are_two_measured_steps() {
        let mut c = Controller::new(&cfg());
        c.observe_cost(1, 10.0);
        assert_eq!(c.cost_ms(1), Some(10.0));
        assert_eq!(c.cost_ms(5), Some(20.0));
    }

    #[test]
    fn the_slope_comes_from_measured_steps_once_there_are_two() {
        let mut c = Controller::new(&cfg());
        c.observe_cost(1, 10.0);
        c.observe_cost(3, 14.0);
        assert_eq!(c.cost_ms(5), Some(18.0));
        assert_eq!(c.cost_ms(3), Some(14.0));
        assert_eq!(c.cost_ms(2), Some(12.0));
    }

    #[test]
    fn a_falling_slope_is_never_below_zero() {
        let mut c = Controller::new(&cfg());
        c.observe_cost(1, 10.0);
        c.observe_cost(3, 8.0);
        assert_eq!(c.cost_ms(5), Some(10.0));
    }

    #[test]
    fn the_step_cost_is_an_average_with_a_half_life_of_16_steps() {
        let mut c = Controller::new(&cfg());
        c.observe_cost(1, 10.0);
        for _ in 0..16 {
            c.observe_cost(1, 20.0);
        }
        // After 16 steps at 20, half the weight of the first 10 is gone.
        assert!((c.cost_ms(1).unwrap() - 15.0).abs() < 1e-9);
    }

    #[test]
    fn bad_timings_and_sizes_are_ignored() {
        let mut c = Controller::new(&cfg());
        c.observe_cost(1, 10.0);
        c.observe_cost(1, f64::NAN);
        c.observe_cost(1, -3.0);
        c.observe_cost(0, 5.0);
        c.observe_cost(1000, 5.0);
        assert_eq!(c.cost_ms(1), Some(10.0));
        // Still one measured size, so the slope is the config's.
        assert_eq!(c.cost_ms(2), Some(12.5));
    }

    fn mtp_cfg(prior: &[f64], c1: f64, slope: f64, mtp_cost: f64) -> Controller {
        let mut config = cfg();
        config.mtp_accept_prior = prior.to_vec();
        config.cost_slope_ms = slope;
        config.mtp_cost_ms = mtp_cost;
        let mut c = Controller::new(&config);
        c.observe_cost(1, c1);
        c
    }

    #[test]
    fn lookups_choice_is_5as() {
        // The lookup-only cases through the source argument: the MTP keys change nothing for lookup.
        let c = timed(0.9);
        assert_eq!(c.choose(Source::Lookup, 4, 4), 4);
        assert_eq!(c.choose(Source::Lookup, 2, 4), 2);
        assert_eq!(c.choose(Source::Lookup, 4, 1), 1);
        assert_eq!(timed(0.28).choose(Source::Lookup, 4, 4), 0);
        assert_eq!(timed(0.30).choose(Source::Lookup, 4, 4), 1);
        let mut other = cfg();
        other.mtp_accept_prior = vec![0.99];
        other.mtp_cost_ms = 0.0;
        let mut o = Controller::new(&other);
        o.observe_cost(1, 10.0);
        o.set_accept_rate(0.28);
        assert_eq!(o.choose(Source::Lookup, 4, 4), 0);
        // Lookup outcomes leave MTP's rates alone, and the reverse.
        o.observe_accept(Source::Lookup, 2, 2);
        assert_eq!(o.accept_rate(Source::Mtp), 0.99);
        let before = o.accept_rate(Source::Lookup);
        o.observe_accept(Source::Mtp, 2, 2);
        assert_eq!(o.accept_rate(Source::Lookup), before);
    }

    #[test]
    fn mtps_depth_rates_start_at_their_prior_and_move_independently() {
        let mut config = cfg();
        config.mtp_accept_prior = vec![0.6, 0.4];
        let mut c = Controller::new(&config);
        let rates = c.depth_rates();
        assert_eq!(rates.len(), MAX_VERIFY - 1);
        assert_eq!(
            (rates[0], rates[1], rates[2], rates[14]),
            (0.6, 0.4, 0.4, 0.4)
        );
        assert_eq!(c.accept_rate(Source::Mtp), 0.6);
        assert_eq!(c.accept_rate(Source::Lookup), 0.5);
        for _ in 0..32 {
            // Draft 1 kept, draft 2 rejected, draft 3 untested.
            c.observe_accept(Source::Mtp, 2, 1);
        }
        let r = c.depth_rates();
        assert!((r[0] - (1.0 - 0.5 * (1.0 - 0.6))).abs() < 1e-9, "{}", r[0]);
        assert!((r[1] - 0.5 * 0.4).abs() < 1e-9, "{}", r[1]);
        assert_eq!(r[2], 0.4);
        assert_eq!(c.accept_rate(Source::Lookup), 0.5);
    }

    #[test]
    fn a_fresh_controller_drafts_with_mtp_at_the_measured_prior() {
        let c = mtp_cfg(&[0.75, 0.70], 10.4, 4.4, 1.8);
        assert_eq!(c.choose(Source::Mtp, 3, 3), 1);
        let c = mtp_cfg(&[0.5], 10.4, 4.4, 1.8);
        assert_eq!(c.choose(Source::Mtp, 3, 3), 0);
        // Not timed: plain first.
        let mut config = cfg();
        config.mtp_accept_prior = vec![0.9];
        assert_eq!(Controller::new(&config).choose(Source::Mtp, 3, 3), 0);
    }

    #[test]
    fn a_deep_mtp_draft_is_chosen_only_when_its_depth_rates_pay() {
        let deep = mtp_cfg(&[0.8, 0.8], 10.0, 2.5, 1.2);
        assert_eq!(deep.choose(Source::Mtp, 2, 4), 2);
        let shallow = mtp_cfg(&[0.8, 0.1], 10.0, 2.5, 1.2);
        assert_eq!(shallow.choose(Source::Mtp, 2, 4), 1);
    }

    #[test]
    fn mtps_cost_includes_its_draft_pass() {
        let free = mtp_cfg(&[0.8, 0.5], 10.0, 2.5, 0.0);
        assert_eq!(free.choose(Source::Mtp, 2, 4), 2);
        let dear = mtp_cfg(&[0.8, 0.5], 10.0, 2.5, 3.0);
        assert_eq!(dear.choose(Source::Mtp, 2, 4), 1);
    }

    #[test]
    fn the_draft_cost_line_comes_from_two_measured_ks() {
        let mut c = mtp_cfg(&[0.7], 10.0, 2.5, 1.8);
        assert_eq!(c.draft_cost_ms(3), 5.4);
        c.observe_draft_cost(1, 2.0);
        // One measured k is not a line: the others stay at k times the prior.
        assert_eq!(c.draft_cost_ms(1), 2.0);
        assert!((c.draft_cost_ms(3) - 5.4).abs() < 1e-12);
        c.observe_draft_cost(3, 4.0);
        assert_eq!(c.draft_cost_ms(3), 4.0);
        assert!((c.draft_cost_ms(2) - 3.0).abs() < 1e-9);
        assert!((c.draft_cost_ms(5) - 6.0).abs() < 1e-9);
        // A falling line is never below 0.
        let mut f = mtp_cfg(&[0.7], 10.0, 2.5, 1.8);
        f.observe_draft_cost(1, 4.0);
        f.observe_draft_cost(2, 1.0);
        assert_eq!(f.draft_cost_ms(4), 0.0);
        // Bad timings are ignored.
        f.observe_draft_cost(1, f64::NAN);
        f.observe_draft_cost(0, 1.0);
        assert_eq!(f.draft_cost_ms(1), 4.0);
    }

    #[test]
    fn a_stale_cost_expires_to_the_line() {
        let mut c = Controller::new(&cfg());
        c.observe_cost(1, 10.0);
        c.observe_cost(2, 14.0);
        c.observe_cost(3, 40.0);
        for _ in 0..63 {
            c.observe_cost(2, 14.0);
        }
        // 63 and 64 steps on, the outlier is still the measured average.
        assert_eq!(c.cost_ms(3), Some(40.0));
        c.observe_cost(2, 14.0);
        assert_eq!(c.cost_ms(3), Some(40.0));
        c.observe_cost(2, 14.0);
        // 65 steps on: the line through sizes 1 and 2 (m = 4).
        assert!((c.cost_ms(3).unwrap() - 18.0).abs() < 1e-9);
        // Size 1 never expires.
        assert_eq!(c.cost_ms(1), Some(10.0));
    }

    #[test]
    fn a_stale_cost_no_longer_traps_lookup_at_one_token() {
        let mut c = Controller::new(&cfg());
        c.observe_cost(1, 19.4);
        c.observe_cost(2, 21.0);
        c.observe_cost(3, 32.9);
        // 14 fully accepted drafts of 4: 56 tokens, a rate of about 0.85.
        for _ in 0..14 {
            c.observe_accept(Source::Lookup, 4, 4);
        }
        c.observe_cost(2, 21.0);
        let a = c.accept_rate(Source::Lookup);
        assert!((0.84..0.86).contains(&a), "{a}");
        assert_eq!(c.choose(Source::Lookup, 4, 4), 1);
        for _ in 0..65 {
            c.observe_cost(2, 21.0);
        }
        assert!(c.choose(Source::Lookup, 4, 4) >= 2);
    }

    #[test]
    fn a_re_measured_expired_size_takes_the_new_sample_and_stays_chosen() {
        let mut c = Controller::new(&cfg());
        c.observe_cost(1, 19.4);
        c.observe_cost(2, 21.0);
        c.observe_cost(3, 32.9);
        for _ in 0..14 {
            c.observe_accept(Source::Lookup, 4, 4);
        }
        for _ in 0..65 {
            c.observe_cost(2, 21.0);
        }
        // c(3) has expired: the line gives about 22.6 and a 2-token draft picks k = 2.
        assert_eq!(c.choose(Source::Lookup, 2, 4), 2);
        // One warm 3-token step at 21 ms replaces the stale 32.9 rather than blending into it.
        c.observe_cost(3, 21.0);
        let c3 = c.cost_ms(3).unwrap();
        assert!((c3 - 21.0).abs() < 1e-9, "{c3}");
        assert_eq!(c.choose(Source::Lookup, 2, 4), 2);
    }

    #[test]
    fn a_stale_draft_cost_expires_to_the_prior() {
        let mut c = mtp_cfg(&[0.7], 10.0, 2.5, 1.8);
        c.observe_draft_cost(1, 2.0);
        c.observe_draft_cost(3, 40.0);
        assert_eq!(c.draft_cost_ms(3), 40.0);
        for _ in 0..64 {
            c.observe_cost(1, 10.0);
        }
        assert_eq!(c.draft_cost_ms(3), 40.0);
        c.observe_cost(1, 10.0);
        // Both points are 65 steps old: back to k times the prior.
        assert!((c.draft_cost_ms(3) - 5.4).abs() < 1e-12);
        // One fresh point is not a line either.
        c.observe_draft_cost(1, 2.0);
        assert_eq!(c.draft_cost_ms(1), 2.0);
        assert!((c.draft_cost_ms(3) - 5.4).abs() < 1e-12);
    }

    const SHORT: LookupMatch = LookupMatch {
        n: 2,
        continues: false,
    };
    const LONG: LookupMatch = LookupMatch {
        n: 4,
        continues: false,
    };
    const RUN: LookupMatch = LookupMatch {
        n: 4,
        continues: true,
    };

    #[test]
    fn each_lookup_class_starts_from_the_pooled_rate_and_learns_alone() {
        let mut c = Controller::new(&cfg());
        for m in [SHORT, LONG, RUN] {
            assert_eq!(c.lookup_rate(Some(m)), 0.5);
        }
        // 32 rejections of 2-token matches halve that class's rate and the pooled rate.
        for _ in 0..32 {
            c.observe_accept_as(Source::Lookup, Some(SHORT), 1, 0);
        }
        assert!((c.lookup_rate(Some(SHORT)) - 0.25).abs() < 1e-9);
        assert!((c.accept_rate(Source::Lookup) - 0.25).abs() < 1e-9);
        assert_eq!(c.lookup_rate(None), c.accept_rate(Source::Lookup));
        // A class not yet seen stands on the pooled rate; its first outcome starts it from there, and from then on
        // it moves alone.
        assert_eq!(c.lookup_rate(Some(RUN)), c.accept_rate(Source::Lookup));
        c.observe_accept_as(Source::Lookup, Some(RUN), 1, 1);
        let w = 0.5f64.powf(1.0 / 32.0);
        let run = w * 0.25 + (1.0 - w);
        assert!((c.lookup_rate(Some(RUN)) - run).abs() < 1e-12);
        for _ in 0..8 {
            c.observe_accept_as(Source::Lookup, Some(SHORT), 1, 0);
        }
        assert!((c.lookup_rate(Some(RUN)) - run).abs() < 1e-12);
        assert!(
            c.lookup_rate(Some(LONG)) < 0.25,
            "LONG still stands on the pooled rate"
        );
        // A 3-token match is its own class; 5 tokens or more share the 4-token class.
        let three = LookupMatch {
            n: 3,
            continues: false,
        };
        let five = LookupMatch {
            n: 5,
            continues: false,
        };
        let mut d = Controller::new(&cfg());
        d.observe_accept_as(Source::Lookup, Some(SHORT), 1, 0);
        d.observe_accept_as(Source::Lookup, Some(three), 4, 4);
        d.observe_accept_as(Source::Lookup, Some(five), 1, 0);
        assert!(d.lookup_rate(Some(three)) > 0.5);
        assert!(d.lookup_rate(Some(LONG)) < d.lookup_rate(Some(three)));
        assert_eq!(d.lookup_rate(Some(LONG)), d.lookup_rate(Some(five)));
    }

    #[test]
    fn an_outcome_with_no_class_moves_only_the_pooled_rate() {
        let mut c = Controller::new(&cfg());
        c.observe_accept_as(Source::Lookup, Some(RUN), 1, 0);
        let run = c.lookup_rate(Some(RUN));
        c.observe_accept(Source::Lookup, 8, 8);
        assert!(c.accept_rate(Source::Lookup) > 0.5);
        assert_eq!(c.lookup_rate(Some(RUN)), run);
        // MTP outcomes never touch lookup's classes, nor start one.
        c.observe_accept_as(Source::Mtp, Some(RUN), 2, 2);
        assert_eq!(c.lookup_rate(Some(RUN)), run);
        c.observe_accept_as(Source::Mtp, Some(SHORT), 2, 0);
        assert_eq!(c.lookup_rate(Some(SHORT)), c.accept_rate(Source::Lookup));
        // Nothing tested starts nothing.
        c.observe_accept_as(Source::Lookup, Some(SHORT), 0, 0);
        c.observe_accept(Source::Lookup, 4, 0);
        assert_eq!(c.lookup_rate(Some(SHORT)), c.accept_rate(Source::Lookup));
    }

    #[test]
    fn the_class_rate_sets_lookups_draft_length() {
        let mut c = timed(0.5);
        // A long run of fully kept continuing drafts, and short matches that fail.
        for _ in 0..40 {
            c.observe_accept_as(Source::Lookup, Some(RUN), 4, 4);
            c.observe_accept_as(Source::Lookup, Some(SHORT), 1, 0);
        }
        assert_eq!(c.choose_scored_as(Source::Lookup, Some(RUN), 4, 4).0, 4);
        assert_eq!(c.choose_scored_as(Source::Lookup, Some(SHORT), 4, 4).0, 0);
        // `choose` with no class is the pooled rate's choice, as before.
        let pooled = c.accept_rate(Source::Lookup);
        assert_eq!(
            c.choose(Source::Lookup, 4, 4),
            timed(pooled).choose(Source::Lookup, 4, 4)
        );
        // MTP's choice ignores a lookup class.
        let m = mtp_cfg(&[0.75, 0.70], 10.4, 4.4, 1.8);
        assert_eq!(
            m.choose_scored_as(Source::Mtp, Some(RUN), 3, 3),
            m.choose_scored(Source::Mtp, 3, 3)
        );
    }

    #[test]
    fn the_probe_counts_plain_steps_since_mtp_last_drafted() {
        let mut config = cfg();
        config.probe_every = 4;
        let mut c = Controller::new(&config);
        c.observe_plain();
        // A lookup verify step in between: its calls must not move the probe count.
        c.observe_cost(3, 15.0);
        c.observe_accept(Source::Lookup, 2, 2);
        c.observe_plain();
        c.observe_plain();
        assert!(!c.probe_due());
        c.observe_plain();
        assert!(c.probe_due());
        c.observe_mtp_drafted();
        assert!(!c.probe_due());
        for _ in 0..3 {
            c.observe_plain();
        }
        assert!(!c.probe_due());
        c.observe_plain();
        assert!(c.probe_due());
    }
}
