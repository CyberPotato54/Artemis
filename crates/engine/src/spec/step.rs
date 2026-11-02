//! The speculation step loop: draft, choose how many drafts to verify, run one decode-shaped
//! pass, pick each position's token with the caller's sampler in order, keep the drafts that equal the picks, commit.
//! It is one loop for the turn, `artemis generate` and test 6's warm path; the GPU session reaches it through
//! `Verifier`.
use super::{
    next_source_for, Controller, Drafter, Lookup, LookupMatch, MeasuredCosts, Source, SourceChoice,
    SpecConfig, MAX_VERIFY,
};
use std::time::Instant;

/// What one step needs of the engine: a plain decode, or a verify pass over `[x, d1 .. dk]` whose rows are read one at
/// a time and then kept in part.
pub trait Verifier {
    type Error;
    /// One plain decode step of `token`; returns the row after it.
    fn decode(&mut self, token: u32) -> Result<Vec<f32>, Self::Error>;
    /// Runs `tokens = [x, d1 .. dk]` (1 to `MAX_VERIFY` of them) as if all were kept. The step's rows are then read
    /// with `row`, and the step is settled by `commit`.
    fn verify(&mut self, tokens: &[u32]) -> Result<(), Self::Error>;
    /// Row `i` of the pending verify: the logits after `tokens[i]`. Read lazily, as `pick` reaches it.
    fn row(&mut self, i: usize) -> Result<Vec<f32>, Self::Error>;
    /// Keeps `tokens[..kept]` of the pending verify. On an error from `verify`, `row` or `commit` the verifier has
    /// already failed the step: `step` doesn't commit, clear or repair anything after one. When `step` commits
    /// after a `pick` error and that commit fails too, the `pick` error is the one returned; the session's `usable`
    /// flag tells the caller the state is gone.
    fn commit(&mut self, kept: usize) -> Result<(), Self::Error>;
    fn n_vocab(&self) -> usize;
    /// The most tokens one verify may take: `MAX_VERIFY`, or the engine's prefill chunk where that is smaller (the
    /// verify's rows live in prefill's buffers). `step` drafts at most one fewer, so a small chunk never reaches
    /// `VerifyShape` mid-turn.
    fn most(&self) -> usize {
        MAX_VERIFY
    }
    /// MTP's drafts after `x`, up to `k` of them (`Session::mtp_draft`). `x` is the token the step is one short
    /// of, not yet decoded. It may return fewer than `k` (down to none): the step then verifies what it got. An error
    /// leaves nothing pending, and `step` returns it at once. Default: no drafts.
    fn mtp_draft(&mut self, x: u32, k: usize) -> Result<Vec<u32>, Self::Error> {
        let _ = (x, k);
        Ok(Vec::new())
    }
    /// Whether the session can draft with MTP (the load allocated it).
    fn has_mtp(&self) -> bool {
        false
    }
    /// The engine's routed-expert misses so far.
    fn misses(&self) -> u64;
    /// Bytes one miss reads (`expert_bytes`).
    fn miss_bytes(&self) -> u64;
}

/// What the caller's `pick` says about the token it chose from a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Next {
    /// Emit and continue with this token as the next `x`. It is also compared with the next draft token.
    Go(u32),
    /// The turn ends here (a stop token, `max_tokens`, the context's end, a cancel, an unload): no later row is read.
    End,
}

/// The speculation counts of a request, or of the whole process. Sums, so deltas add.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecDelta {
    /// Verify steps (`k >= 1`).
    pub steps: u64,
    /// Steps the controller (or an empty draft) ran at `k = 0`.
    pub plain_steps: u64,
    /// The sum of `k`.
    pub drafted: u64,
    /// Draft tokens kept.
    pub accepted: u64,
    /// `drafted` and `accepted` by source: `drafted == drafted_lookup + drafted_mtp`, and likewise `accepted`.
    pub drafted_lookup: u64,
    pub accepted_lookup: u64,
    pub drafted_mtp: u64,
    pub accepted_mtp: u64,
    /// MTP's drafts that were probes, a subset of the MTP steps.
    pub probes: u64,
    /// Verify steps that kept fewer than all their drafts, a cut by the caller's stop included.
    pub replays: u64,
}

/// What steps of one size cost. Sums; row `i` is the size `t = i + 1` (`x` and `t - 1` drafts), so row 0 holds the
/// plain steps.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SizeRow {
    pub steps: u64,
    /// Milliseconds, from the `decode` or `verify` call to the end of the step.
    pub ms: f64,
    /// The routed-expert misses of these steps, in bytes.
    pub miss_bytes: u64,
    /// Positions kept.
    pub tokens: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecTotals {
    pub delta: SpecDelta,
    pub by_size: [SizeRow; MAX_VERIFY],
    /// Lookup's current per-token accept probability.
    pub accept_rate: f64,
    /// MTP's draft passes by size: row `k - 1` holds the passes that returned `k` drafts (`steps`), their time (`ms`)
    /// and the drafts kept (`tokens`). `miss_bytes` stays 0.
    pub mtp_by_k: [SizeRow; MAX_VERIFY],
    /// MTP's current accept rate at depth 1.
    pub mtp_accept_rate: f64,
    /// MTP's current accept rates at depths 1 to 15.
    pub depth_rates: [f64; MAX_VERIFY - 1],
    /// Per MTP draft depth, `(drafts tested, drafts accepted)`: sums, so runs pool into measured rates.
    pub mtp_depth_counts: [(u64, u64); MAX_VERIFY - 1],
}

/// What a clock is asked to time.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepTimed {
    /// A plain decode step.
    Plain,
    /// A verify step of `tokens` tokens (`x` and its drafts).
    Verify { tokens: usize },
    /// An MTP draft pass that returned `k` drafts, after the cut to the vocabulary and to the `k` asked for. A pass
    /// that returned none is timed as `k = 1`.
    MtpDraft { k: usize },
}

/// How a step ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stepped {
    /// Positions kept: `x` and the drafts whose rows `pick` read (all `pick` reached, a rejected draft's row included).
    pub kept: usize,
    /// `pick` returned `End`.
    pub ended: bool,
}

/// A draft that was not verified, scored against the tokens a step picked: `(tested, accepted)` as
/// `Controller::observe_accept` counts them. The draft's tokens are compared in order while picks last; the first
/// mismatch is tested and ends it, and draft tokens past the last pick (a stop, or a shorter step) are untested.
pub fn score_against(draft: &[u32], picks: &[u32]) -> (usize, usize) {
    let accepted = draft.iter().zip(picks).take_while(|(d, p)| d == p).count();
    let mismatched = accepted < draft.len() && accepted < picks.len();
    (accepted + usize::from(mismatched), accepted)
}

/// Speculation's engine-wide state: the settings, the controller and the totals. Built at each load.
pub struct SpecState {
    cfg: SpecConfig,
    controller: Controller,
    totals: SpecTotals,
    make: Box<dyn Fn() -> Box<dyn Drafter> + Send>,
    /// Replaces the wall clock when set (tests): called once per timed region, with what it was, for its ms.
    #[allow(clippy::type_complexity)]
    clock: Option<Box<dyn FnMut(StepTimed) -> f64 + Send>>,
}

impl SpecState {
    /// Prompt lookup as the drafter. `cfg` must pass `SpecConfig::check`.
    pub fn new(cfg: SpecConfig) -> SpecState {
        let (min, max) = (cfg.ngram_min, cfg.ngram_max);
        SpecState::with_drafters(cfg, Box::new(move || Box::new(Lookup::new(min, max))))
    }

    /// `make` builds a request's drafter (tests, and other sources).
    pub fn with_drafters(
        cfg: SpecConfig,
        make: Box<dyn Fn() -> Box<dyn Drafter> + Send>,
    ) -> SpecState {
        SpecState {
            controller: Controller::new(&cfg),
            totals: SpecTotals {
                delta: SpecDelta::default(),
                by_size: [SizeRow::default(); MAX_VERIFY],
                accept_rate: cfg.accept_prior,
                mtp_by_k: [SizeRow::default(); MAX_VERIFY],
                mtp_accept_rate: 0.0,
                depth_rates: [0.0; MAX_VERIFY - 1],
                mtp_depth_counts: [(0, 0); MAX_VERIFY - 1],
            },
            cfg,
            make,
            clock: None,
        }
    }

    /// Takes the time of each step and each MTP draft pass from `clock`, in ms, instead of the wall clock. For tests;
    /// production never calls it.
    #[doc(hidden)]
    pub fn with_clock(mut self, clock: Box<dyn FnMut(StepTimed) -> f64 + Send>) -> SpecState {
        self.clock = Some(clock);
        self
    }

    fn ms_since(&mut self, start: Instant, what: StepTimed) -> f64 {
        match self.clock.as_mut() {
            Some(clock) => clock(what),
            None => start.elapsed().as_secs_f64() * 1000.0,
        }
    }

    pub fn config(&self) -> &SpecConfig {
        &self.cfg
    }

    /// A fresh drafter for one request: `Lookup::new(ngram_min, ngram_max)` unless `with_drafters` said otherwise.
    pub fn drafter(&self) -> Box<dyn Drafter> {
        (self.make)()
    }

    /// What the controller has measured of its own step and draft costs; `None` while only the priors stand.
    pub fn measured(&self) -> MeasuredCosts {
        self.controller.measured()
    }

    pub fn totals(&self) -> SpecTotals {
        SpecTotals {
            accept_rate: self.controller.accept_rate(Source::Lookup),
            mtp_accept_rate: self.controller.accept_rate(Source::Mtp),
            depth_rates: self.controller.depth_rates(),
            ..self.totals
        }
    }

    /// One step. `x` is the token the state is one short of (already emitted, and already `observe`d by `drafter`).
    /// It drafts at most `k_max` tokens, picks `k` (the fixed length, or the controller's), runs a decode or a verify
    /// of `[x, d1 .. dk]`, and calls `pick` on each row in order: only while every earlier pick equalled the draft
    /// token at its position, and not past an `End`. Every `Go(y)` is passed to `drafter.observe`. Then it commits
    /// the rows read.
    ///
    /// An error from `decode`, `verify`, `row` or `commit` returns at once: the verifier has failed the step itself.
    /// An error from `pick` after a verify commits the rows read first, then returns (the commit's own error, if
    /// it fails too, is dropped: the verifier has failed the step and the pick's came first). A step that returns an
    /// error counts nothing. `delta` is added to, here and in the state's totals.
    #[allow(clippy::type_complexity)] // `pick`'s signature, spelled out
    pub fn step<V: Verifier>(
        &mut self,
        v: &mut V,
        drafter: &mut dyn Drafter,
        x: u32,
        k_max: usize,
        pick: &mut dyn FnMut(&[f32]) -> Result<Next, V::Error>,
        delta: &mut SpecDelta,
    ) -> Result<Stepped, V::Error> {
        let cap = k_max
            .min(self.cfg.max_draft)
            .min(v.most().min(MAX_VERIFY).saturating_sub(1));
        // Lookup's draft: a guess, but its ids go to the embedding lookup, so keep to the vocabulary and to `cap`.
        let mut draft = if self.cfg.has(Source::Lookup) {
            drafter.draft(cap)
        } else {
            Vec::new()
        };
        draft.truncate(cap);
        if let Some(bad) = draft.iter().position(|&t| t as usize >= v.n_vocab()) {
            draft.truncate(bad);
        }
        // What produced lookup's draft: the controller keeps an accept rate per class of match.
        let class: Option<LookupMatch> = if draft.is_empty() {
            None
        } else {
            drafter.last_match()
        };
        // The one decision the simulator shares: lookup first, then MTP, then MTP's probe; or, with
        // `source_choice = best`, the source with the higher score.
        let (mut source, mut k, mut probe) = next_source_for(
            &self.controller,
            &self.cfg,
            draft.len(),
            class,
            v.has_mtp(),
            cap,
        );
        // The tokens this step verifies after `x`.
        let mut chosen: Vec<u32> = draft[..k.min(draft.len())].to_vec();
        // The MTP pass's time, counted once the step has succeeded (an error counts nothing).
        let mut mtp_pass_ms = None;
        if source == Source::Mtp {
            let pass = Instant::now();
            let mut got = v.mtp_draft(x, k)?;
            got.truncate(k);
            if let Some(bad) = got.iter().position(|&t| t as usize >= v.n_vocab()) {
                got.truncate(bad);
            }
            let pass_ms = self.ms_since(
                pass,
                StepTimed::MtpDraft {
                    k: got.len().max(1),
                },
            );
            if got.is_empty() {
                // Nothing drafted, so the step is plain and MTP has not drafted. `probe_due` therefore stays
                // true and the next lookup-less step asks again. That is bounded: this plain step refills the
                // hidden-row ring, so the next pass has its input. Resetting the count here would put the first
                // real probe `probe_every` steps off after every restore.
                (source, k, probe) = (Source::Lookup, 0, false);
                chosen.clear();
                // `source_choice = best` may have passed over a draft lookup's controller took: it goes ahead.
                if self.cfg.source_choice == SourceChoice::Best && !draft.is_empty() {
                    k = self
                        .controller
                        .choose_scored_as(Source::Lookup, class, draft.len(), cap)
                        .0;
                    chosen = draft[..k].to_vec();
                }
            } else {
                k = got.len();
                mtp_pass_ms = Some(pass_ms);
                chosen = got;
            }
        }
        debug_assert!(k == chosen.len() && k < MAX_VERIFY);

        let start = Instant::now();
        let misses_before = v.misses();
        let (kept, ended, tested, accepted);
        // Lookup's draft that this step did not verify, scored against the tokens the step picked: `(tested,
        // accepted)` as `observe_accept` counts them.
        let mut lookup_unverified = None;
        if k == 0 {
            let row = v.decode(x)?;
            let next = pick(&row)?;
            kept = 1;
            ended = next == Next::End;
            // Lookup scores itself for free: its first draft token against the token picked. An `End` leaves
            // it untested.
            let first = draft.first().copied();
            accepted = usize::from(matches!((next, first), (Next::Go(y), Some(d)) if y == d));
            tested = usize::from(first.is_some() && !ended);
            if let Next::Go(y) = next {
                drafter.observe(&[y]);
            }
        } else {
            let mut tokens = Vec::with_capacity(k + 1);
            tokens.push(x);
            tokens.extend_from_slice(&chosen);
            v.verify(&tokens)?;
            let (mut read, mut stopped, mut mismatched) = (0, false, false);
            let mut failure = None;
            let mut picks = Vec::with_capacity(k + 1);
            for i in 0..=k {
                let row = v.row(i)?;
                read = i + 1;
                match pick(&row) {
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                    Ok(Next::End) => {
                        stopped = true;
                        break;
                    }
                    Ok(Next::Go(y)) => {
                        drafter.observe(&[y]);
                        picks.push(y);
                        // Row k has no draft after it to compare with.
                        if chosen.get(i).is_some_and(|&d| d != y) {
                            mismatched = true;
                            break;
                        }
                    }
                }
            }
            if let Some(e) = failure {
                let _ = v.commit(read);
                return Err(e);
            }
            v.commit(read)?;
            (kept, ended) = (read, stopped);
            accepted = read - 1;
            tested = accepted + usize::from(mismatched);
            if source == Source::Mtp && !draft.is_empty() {
                lookup_unverified = Some(score_against(&draft, &picks));
            }
        }
        let ms = self.ms_since(
            start,
            if k == 0 {
                StepTimed::Plain
            } else {
                StepTimed::Verify { tokens: k + 1 }
            },
        );
        let miss_bytes = v.misses().saturating_sub(misses_before) * v.miss_bytes();

        if tested > 0 {
            // At `k == 0` the source is lookup (scoring its own first token) and the class is its draft's.
            let as_class = if source == Source::Lookup {
                class
            } else {
                None
            };
            self.controller
                .observe_accept_as(source, as_class, tested, accepted);
        }
        if let Some((tested, hit)) = lookup_unverified.filter(|&(tested, _)| tested > 0) {
            self.controller
                .observe_accept_as(Source::Lookup, class, tested, hit);
        }
        if k == 0 {
            self.controller.observe_plain();
        }
        self.controller.observe_cost(k + 1, ms);
        if let Some(pass_ms) = mtp_pass_ms {
            self.controller.observe_mtp_drafted();
            self.controller.observe_draft_cost(k, pass_ms);
            let row = &mut self.totals.mtp_by_k[k - 1];
            row.steps += 1;
            row.ms += pass_ms;
            for i in 0..tested {
                let counts = &mut self.totals.mtp_depth_counts[i];
                counts.0 += 1;
                counts.1 += u64::from(i < accepted);
            }
            row.tokens += accepted as u64;
        }
        let mut step = SpecDelta::default();
        if k == 0 {
            step.plain_steps = 1;
        } else {
            step.steps = 1;
            step.drafted = k as u64;
            step.accepted = accepted as u64;
            step.replays = u64::from(kept < k + 1);
            match source {
                Source::Lookup => {
                    step.drafted_lookup = step.drafted;
                    step.accepted_lookup = step.accepted;
                }
                Source::Mtp => {
                    step.drafted_mtp = step.drafted;
                    step.accepted_mtp = step.accepted;
                    step.probes = u64::from(probe);
                }
            }
        }
        let row = &mut self.totals.by_size[k];
        row.steps += 1;
        row.ms += ms;
        row.miss_bytes += miss_bytes;
        row.tokens += kept as u64;
        for d in [&mut self.totals.delta, delta] {
            d.steps += step.steps;
            d.plain_steps += step.plain_steps;
            d.drafted += step.drafted;
            d.accepted += step.accepted;
            d.drafted_lookup += step.drafted_lookup;
            d.accepted_lookup += step.accepted_lookup;
            d.drafted_mtp += step.drafted_mtp;
            d.accepted_mtp += step.accepted_mtp;
            d.probes += step.probes;
            d.replays += step.replays;
        }
        Ok(Stepped { kept, ended })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::{greedy, Sampler, Sampling};
    use std::time::Duration;

    const VOCAB: usize = 24;
    const PROMPT: [u32; 5] = [3, 1, 4, 1, 5];

    /// A deterministic row: the logits after a history, from its last three tokens (so greedy runs cycle, as real
    /// text repeats, and lookup has something to find).
    fn logits(history: &[u32]) -> Vec<f32> {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &t in &history[history.len().saturating_sub(3)..] {
            h = (h ^ u64::from(t)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        (0..VOCAB as u64)
            .map(|j| {
                let mut z =
                    (h ^ j.wrapping_mul(0x9e37_79b9_7f4a_7c15)).wrapping_add(0x9e37_79b9_7f4a_7c15);
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 31;
                (z >> 40) as f32 / (1u64 << 24) as f32 * 8.0
            })
            .collect()
    }

    #[derive(Clone, Debug, PartialEq)]
    enum Call {
        Decode(u32),
        Verify(Vec<u32>),
        Row(usize),
        Commit(usize),
        /// `mtp_draft(x, k)`.
        Mtp(u32, usize),
    }

    #[derive(Default)]
    struct Faults {
        verify: bool,
        row: Option<usize>,
        commit: bool,
    }

    /// The engine as a pure function of the history: `decode` and `verify` append, `commit` cuts, and a fault fails
    /// the step the way the session does: the pending verify is dropped and the history cut back to its start.
    struct ScriptedVerifier {
        history: Vec<u32>,
        pending: Option<usize>,
        calls: Vec<Call>,
        faults: Faults,
        misses: u64,
        commit_sleep: Duration,
        /// What `Verifier::most` says: a verify of more tokens than this is a `VerifyShape` error on the real engine.
        most: usize,
        /// The scripted MTP; `None` is a session without one (`has_mtp()` false).
        mtp: Option<MtpScript>,
    }

    /// What the scripted MTP returns.
    enum MtpMode {
        /// By call: up to `len` tokens of the true stream after `x`, the one at `wrong_at` replaced; the last repeats.
        Plans(Vec<Plan>),
        /// These tokens, whatever `x` is (cut to `k`).
        Fixed(Vec<u32>),
        /// An error.
        Fail,
    }

    struct MtpScript {
        truth: Vec<u32>,
        mode: MtpMode,
        made: usize,
    }

    impl ScriptedVerifier {
        fn new(prompt: &[u32]) -> ScriptedVerifier {
            ScriptedVerifier {
                history: prompt.to_vec(),
                pending: None,
                calls: Vec::new(),
                faults: Faults::default(),
                misses: 0,
                commit_sleep: Duration::ZERO,
                most: MAX_VERIFY,
                mtp: None,
            }
        }

        fn with_mtp(mut self, truth: &[u32], mode: MtpMode) -> ScriptedVerifier {
            self.mtp = Some(MtpScript {
                truth: truth.to_vec(),
                mode,
                made: 0,
            });
            self
        }

        fn mtp_asks(&self) -> Vec<usize> {
            self.calls
                .iter()
                .filter_map(|c| match c {
                    Call::Mtp(_, k) => Some(*k),
                    _ => None,
                })
                .collect()
        }

        fn prefill_row(&self) -> Vec<f32> {
            logits(&self.history)
        }

        fn fail(&mut self, start: usize, why: &'static str) -> &'static str {
            self.pending = None;
            self.history.truncate(start);
            why
        }

        /// The last verify call's size and the commit that followed it.
        fn last_burst(&self) -> (usize, usize) {
            let verify = self
                .calls
                .iter()
                .rposition(|c| matches!(c, Call::Verify(_)));
            let Some(Call::Verify(t)) = verify.map(|i| &self.calls[i]) else {
                panic!("no verify call");
            };
            let Some(Call::Commit(kept)) = self.calls.last() else {
                panic!("no commit last");
            };
            (t.len(), *kept)
        }

        fn verify_sizes(&self) -> Vec<usize> {
            self.calls
                .iter()
                .filter_map(|c| match c {
                    Call::Verify(t) => Some(t.len()),
                    _ => None,
                })
                .collect()
        }
    }

    impl Verifier for ScriptedVerifier {
        type Error = &'static str;

        fn decode(&mut self, token: u32) -> Result<Vec<f32>, &'static str> {
            assert!(self.pending.is_none(), "decode while a verify is pending");
            self.calls.push(Call::Decode(token));
            self.misses += 2;
            self.history.push(token);
            Ok(logits(&self.history))
        }

        fn verify(&mut self, tokens: &[u32]) -> Result<(), &'static str> {
            assert!(self.pending.is_none(), "verify while a verify is pending");
            assert!((1..=self.most).contains(&tokens.len()));
            assert!(tokens.iter().all(|&t| (t as usize) < VOCAB));
            self.calls.push(Call::Verify(tokens.to_vec()));
            let start = self.history.len();
            if self.faults.verify {
                return Err(self.fail(start, "verify"));
            }
            self.misses += 5;
            self.pending = Some(start);
            self.history.extend_from_slice(tokens);
            Ok(())
        }

        fn row(&mut self, i: usize) -> Result<Vec<f32>, &'static str> {
            let start = self.pending.expect("row without a pending verify");
            self.calls.push(Call::Row(i));
            if self.faults.row == Some(i) {
                return Err(self.fail(start, "row"));
            }
            Ok(logits(&self.history[..start + i + 1]))
        }

        fn commit(&mut self, kept: usize) -> Result<(), &'static str> {
            let start = self.pending.expect("commit without a pending verify");
            self.calls.push(Call::Commit(kept));
            std::thread::sleep(self.commit_sleep);
            if self.faults.commit {
                return Err(self.fail(start, "commit"));
            }
            self.history.truncate(start + kept);
            self.pending = None;
            Ok(())
        }

        fn n_vocab(&self) -> usize {
            VOCAB
        }

        fn most(&self) -> usize {
            self.most
        }

        fn misses(&self) -> u64 {
            self.misses
        }

        fn miss_bytes(&self) -> u64 {
            100
        }

        fn mtp_draft(&mut self, x: u32, k: usize) -> Result<Vec<u32>, &'static str> {
            assert!(
                self.pending.is_none(),
                "mtp_draft while a verify is pending"
            );
            self.calls.push(Call::Mtp(x, k));
            let at = self.history.len() + 1;
            let Some(m) = self.mtp.as_mut() else {
                return Ok(Vec::new());
            };
            match &m.mode {
                MtpMode::Fail => Err("mtp"),
                MtpMode::Fixed(t) => Ok(t.iter().take(k).copied().collect()),
                MtpMode::Plans(plans) => {
                    let plan = plans[m.made.min(plans.len() - 1)];
                    m.made += 1;
                    let mut out: Vec<u32> = m
                        .truth
                        .get(at..)
                        .unwrap_or(&[])
                        .iter()
                        .take(plan.len.min(k))
                        .copied()
                        .collect();
                    if let Some(w) = plan.wrong_at.filter(|&w| w < out.len()) {
                        out[w] = (out[w] + 1) % VOCAB as u32;
                    }
                    Ok(out)
                }
            }
        }

        fn has_mtp(&self) -> bool {
            self.mtp.is_some()
        }
    }

    /// One call of a `ScriptedDrafter`: up to `len` tokens of the true stream, the one at `wrong_at` replaced.
    #[derive(Clone, Copy)]
    struct Plan {
        len: usize,
        wrong_at: Option<usize>,
    }

    const fn plan(len: usize, wrong_at: Option<usize>) -> Plan {
        Plan { len, wrong_at }
    }

    /// A drafter that knows the stream a plain loop produces, and drafts it right, wrong or half right by plan; the
    /// last plan repeats. It counts what it observed, so it knows where in the stream the request is.
    struct ScriptedDrafter {
        truth: Vec<u32>,
        seen: usize,
        plans: Vec<Plan>,
        calls: usize,
    }

    impl ScriptedDrafter {
        fn new(truth: &[u32], plans: &[Plan]) -> ScriptedDrafter {
            ScriptedDrafter {
                truth: truth.to_vec(),
                seen: 0,
                plans: plans.to_vec(),
                calls: 0,
            }
        }
        fn right(truth: &[u32]) -> ScriptedDrafter {
            Self::new(truth, &[plan(15, None)])
        }
        fn wrong(truth: &[u32]) -> ScriptedDrafter {
            Self::new(truth, &[plan(15, Some(0))])
        }
        fn half(truth: &[u32]) -> ScriptedDrafter {
            Self::new(truth, &[plan(15, Some(2))])
        }
        fn none(truth: &[u32]) -> ScriptedDrafter {
            Self::new(truth, &[plan(0, None)])
        }
        /// Already told the prompt and `x`, as the caller does before the first step.
        fn primed(mut self, x: u32) -> ScriptedDrafter {
            self.observe(&PROMPT);
            self.observe(&[x]);
            self
        }
    }

    impl Drafter for ScriptedDrafter {
        fn name(&self) -> &'static str {
            "scripted"
        }
        fn draft(&mut self, max: usize) -> Vec<u32> {
            let plan = self.plans[self.calls.min(self.plans.len() - 1)];
            self.calls += 1;
            let from = self.seen.min(self.truth.len());
            let to = (self.seen + plan.len.min(max)).min(self.truth.len());
            let mut d = self.truth[from..to].to_vec();
            if let Some(w) = plan.wrong_at.filter(|&w| w < d.len()) {
                d[w] = (d[w] + 1) % VOCAB as u32;
            }
            d
        }
        fn observe(&mut self, tokens: &[u32]) {
            self.seen += tokens.len();
        }
    }

    /// Drafts these tokens whatever `max` says, as a careless source might.
    struct FixedDrafter(Vec<u32>);

    impl Drafter for FixedDrafter {
        fn name(&self) -> &'static str {
            "fixed"
        }
        fn draft(&mut self, _max: usize) -> Vec<u32> {
            self.0.clone()
        }
        fn observe(&mut self, _tokens: &[u32]) {}
    }

    fn cfg(fixed: Option<usize>) -> SpecConfig {
        SpecConfig {
            enabled: true,
            max_draft: 4,
            ngram_min: 2,
            ngram_max: 4,
            min_gain: 1.03,
            accept_half_life: 32.0,
            cost_slope_ms: 2.5,
            accept_prior: 0.5,
            fixed_draft: fixed,
            ..SpecConfig::default()
        }
    }

    struct Run {
        out: Vec<u32>,
        verifier: ScriptedVerifier,
        sampler: Sampler,
        state: SpecState,
        delta: SpecDelta,
    }

    /// The plain loop: sample, emit, decode; the last token is never decoded.
    fn plain(sampling: Sampling, seed: u64, n: usize) -> Run {
        let mut v = ScriptedVerifier::new(&PROMPT);
        let mut s = Sampler::new(sampling, seed);
        let mut x = s.sample(&v.prefill_row());
        let mut out = vec![x];
        while out.len() < n {
            let row = v.decode(x).unwrap();
            x = s.sample(&row);
            out.push(x);
        }
        Run {
            out,
            verifier: v,
            sampler: s,
            state: SpecState::new(cfg(None)),
            delta: SpecDelta::default(),
        }
    }

    /// The same loop through `step`: `pick` samples, emits, and ends at `n` tokens.
    fn speculate(
        state: SpecState,
        drafter: &mut dyn Drafter,
        sampling: Sampling,
        seed: u64,
        n: usize,
        k_max: usize,
    ) -> Run {
        speculate_on(
            state,
            ScriptedVerifier::new(&PROMPT),
            drafter,
            sampling,
            seed,
            n,
            k_max,
        )
    }

    /// `speculate` on a given verifier (one with an MTP script, say).
    fn speculate_on(
        mut state: SpecState,
        mut v: ScriptedVerifier,
        drafter: &mut dyn Drafter,
        sampling: Sampling,
        seed: u64,
        n: usize,
        k_max: usize,
    ) -> Run {
        let mut s = Sampler::new(sampling, seed);
        let mut delta = SpecDelta::default();
        let mut x = s.sample(&v.prefill_row());
        let mut out = vec![x];
        drafter.observe(&PROMPT);
        drafter.observe(&[x]);
        while out.len() < n {
            let mut next = None;
            let stepped = state
                .step(
                    &mut v,
                    drafter,
                    x,
                    k_max,
                    &mut |row| {
                        let y = s.sample(row);
                        out.push(y);
                        if out.len() >= n {
                            return Ok(Next::End);
                        }
                        next = Some(y);
                        Ok(Next::Go(y))
                    },
                    &mut delta,
                )
                .unwrap();
            if stepped.ended {
                break;
            }
            x = next.expect("a step that didn't end went on");
        }
        Run {
            out,
            verifier: v,
            sampler: s,
            state,
            delta,
        }
    }

    /// The prompt and the tokens a plain loop emits: what a right drafter drafts from.
    fn stream(sampling: Sampling, seed: u64, n: usize) -> Vec<u32> {
        [PROMPT.as_slice(), &plain(sampling, seed, n).out].concat()
    }

    fn greedy_x() -> u32 {
        greedy(&logits(&PROMPT))
    }

    /// One step whose `pick` takes greedy's token and always goes on.
    fn one_step(
        state: &mut SpecState,
        v: &mut ScriptedVerifier,
        drafter: &mut dyn Drafter,
        x: u32,
        k_max: usize,
        delta: &mut SpecDelta,
    ) -> Result<Stepped, &'static str> {
        state.step(
            v,
            drafter,
            x,
            k_max,
            &mut |row| Ok(Next::Go(greedy(row))),
            delta,
        )
    }

    /// One greedy step of `[x, d1 .. d4]` from the prompt, whose `pick` goes on for `go` rows, then ends.
    fn end_after(
        go: usize,
        plans: &[Plan],
    ) -> (
        ScriptedVerifier,
        SpecState,
        ScriptedDrafter,
        SpecDelta,
        Stepped,
    ) {
        let x = greedy_x();
        let mut v = ScriptedVerifier::new(&PROMPT);
        let mut state = SpecState::new(cfg(Some(4)));
        let mut drafter = ScriptedDrafter::new(&stream(Sampling::GREEDY, 0, 64), plans).primed(x);
        let mut delta = SpecDelta::default();
        let mut calls = 0;
        let stepped = state
            .step(
                &mut v,
                &mut drafter,
                x,
                15,
                &mut |row| {
                    calls += 1;
                    if calls > go {
                        return Ok(Next::End);
                    }
                    Ok(Next::Go(greedy(row)))
                },
                &mut delta,
            )
            .unwrap();
        (v, state, drafter, delta, stepped)
    }

    #[test]
    fn any_draft_gives_the_plain_loops_tokens() {
        for (sampling, seed) in [(Sampling::GREEDY, 0), (Sampling::DEFAULT, 1234)] {
            let truth = stream(sampling, seed, 64);
            let want = plain(sampling, seed, 64);
            let drafters: Vec<(&str, Box<dyn Drafter>)> = vec![
                ("right", Box::new(ScriptedDrafter::right(&truth))),
                ("wrong", Box::new(ScriptedDrafter::wrong(&truth))),
                ("half", Box::new(ScriptedDrafter::half(&truth))),
                ("none", Box::new(ScriptedDrafter::none(&truth))),
                ("lookup", Box::new(Lookup::new(2, 4))),
            ];
            for (name, mut drafter) in drafters {
                let state = SpecState::new(cfg(Some(4)));
                let got = speculate(state, &mut *drafter, sampling, seed, 64, 15);
                assert_eq!(got.out, want.out, "{name} {sampling:?}: the tokens");
                assert_eq!(
                    got.verifier.history, want.verifier.history,
                    "{name} {sampling:?}: the committed history"
                );
                if name != "none" {
                    assert!(got.delta.steps > 0, "{name}: no verify step ran");
                }
                if name == "right" {
                    assert!(got.delta.accepted > 0);
                }
            }
        }
    }

    #[test]
    fn rows_past_the_stop_never_reach_the_sampler() {
        // The stop (20 tokens) falls at row 3 of a five-token burst: the sampler must never see row 4.
        let truth = stream(Sampling::DEFAULT, 1234, 64);
        let mut want = plain(Sampling::DEFAULT, 1234, 20);
        let mut drafter = ScriptedDrafter::right(&truth);
        let state = SpecState::new(cfg(Some(4)));
        let mut got = speculate(state, &mut drafter, Sampling::DEFAULT, 1234, 20, 15);
        assert_eq!(got.out, want.out);
        assert_eq!(
            got.verifier.last_burst(),
            (5, 4),
            "the stop should cut a full burst"
        );
        let fixed = logits(&[7, 7, 7]);
        for i in 0..8 {
            assert_eq!(
                got.sampler.sample(&fixed),
                want.sampler.sample(&fixed),
                "draw {i} after the stop"
            );
        }
    }

    #[test]
    fn presence_penalty_inside_a_burst_matches_the_plain_loop() {
        // `seen` holds the emitted tokens only; a rejected draft's row never reaches the sampler.
        let sampled = Sampling {
            presence_penalty: 0.5,
            ..Sampling::DEFAULT
        };
        let greedy_penalized = Sampling {
            presence_penalty: 0.5,
            ..Sampling::GREEDY
        };
        for sampling in [sampled, greedy_penalized] {
            let want = plain(sampling, 1234, 64);
            let free = plain(
                Sampling {
                    presence_penalty: 0.0,
                    ..sampling
                },
                1234,
                64,
            );
            assert_ne!(want.out, free.out, "the penalty should matter here");
            let truth = stream(sampling, 1234, 64);
            for mut drafter in [
                ScriptedDrafter::right(&truth),
                ScriptedDrafter::half(&truth),
                ScriptedDrafter::wrong(&truth),
            ] {
                let state = SpecState::new(cfg(Some(4)));
                let got = speculate(state, &mut drafter, sampling, 1234, 64, 15);
                assert_eq!(got.out, want.out, "{sampling:?}");
                assert_eq!(got.verifier.history, want.verifier.history);
            }
        }
    }

    #[test]
    fn a_draft_never_runs_past_k_max() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        let want = plain(Sampling::GREEDY, 0, 64);
        for k_max in 0..=4 {
            // The scripted drafter honours `max`; the fixed one drafts ten tokens whatever it is asked.
            let ten = truth[PROMPT.len() + 1..PROMPT.len() + 11].to_vec();
            let drafters: Vec<Box<dyn Drafter>> = vec![
                Box::new(ScriptedDrafter::right(&truth)),
                Box::new(FixedDrafter(ten)),
            ];
            for mut drafter in drafters {
                let state = SpecState::new(cfg(Some(4)));
                let got = speculate(state, &mut *drafter, Sampling::GREEDY, 0, 64, k_max);
                assert_eq!(got.out, want.out, "k_max {k_max}");
                let sizes = got.verifier.verify_sizes();
                assert!(sizes.iter().all(|&s| s <= k_max + 1), "{k_max}: {sizes:?}");
                if k_max == 0 {
                    assert!(sizes.is_empty());
                    assert_eq!(got.delta.steps, 0);
                }
            }
        }
        // And k_max is reached when the draft allows it.
        let mut drafter = ScriptedDrafter::right(&truth);
        let state = SpecState::new(cfg(Some(4)));
        let got = speculate(state, &mut drafter, Sampling::GREEDY, 0, 64, 2);
        assert_eq!(got.verifier.verify_sizes()[0], 3);
    }

    #[test]
    fn a_draft_token_outside_the_vocabulary_never_reaches_verify() {
        let mut v = ScriptedVerifier::new(&PROMPT);
        let mut state = SpecState::new(cfg(Some(4)));
        let mut drafter = FixedDrafter(vec![2, VOCAB as u32, 5]);
        let mut delta = SpecDelta::default();
        one_step(&mut state, &mut v, &mut drafter, 9, 15, &mut delta).unwrap();
        assert_eq!(v.calls[0], Call::Verify(vec![9, 2]));
    }

    #[test]
    fn a_step_never_verifies_more_than_max_verify_tokens() {
        // A config that skipped `check` (max_draft 20) still can't index past the table or overfill a verify.
        let mut v = ScriptedVerifier::new(&PROMPT);
        let mut state = SpecState::new(SpecConfig {
            max_draft: 20,
            fixed_draft: Some(20),
            ..cfg(None)
        });
        let mut drafter = FixedDrafter(vec![1; 20]);
        let mut delta = SpecDelta::default();
        one_step(&mut state, &mut v, &mut drafter, 9, 99, &mut delta).unwrap();
        assert_eq!(v.verify_sizes(), vec![MAX_VERIFY]);
        assert_eq!(state.totals().by_size[MAX_VERIFY - 1].steps, 1);
    }

    #[test]
    fn a_small_prefill_chunk_caps_the_draft_below_the_verifiers_most() {
        // A chunk of 6 takes a verify of at most 6 tokens, so a draft of at most 5: the step never asks the engine
        // for a shape it refuses (`VerifyShape`), whatever the config and the caller's cap say.
        let mut v = ScriptedVerifier::new(&PROMPT);
        v.most = 6;
        let mut state = SpecState::new(SpecConfig {
            max_draft: 15,
            fixed_draft: Some(15),
            ..cfg(None)
        });
        let mut drafter = FixedDrafter(vec![1; 15]);
        let mut delta = SpecDelta::default();
        one_step(&mut state, &mut v, &mut drafter, 9, 99, &mut delta).unwrap();
        assert_eq!(v.verify_sizes(), vec![6]);
        assert_eq!(state.totals().by_size[5].steps, 1);
        // The adaptive controller is held to the same cap.
        let mut v = ScriptedVerifier::new(&PROMPT);
        v.most = 6;
        let mut state = SpecState::new(SpecConfig {
            max_draft: 15,
            ..cfg(None)
        });
        let mut delta = SpecDelta::default();
        for _ in 0..20 {
            let mut drafter = FixedDrafter(vec![1; 15]);
            one_step(&mut state, &mut v, &mut drafter, 9, 99, &mut delta).unwrap();
        }
        assert!(v.verify_sizes().iter().all(|&t| t <= 6));
        // A chunk of 1 verifies nothing: every step is a plain decode.
        let mut v = ScriptedVerifier::new(&PROMPT);
        v.most = 1;
        let mut state = SpecState::new(SpecConfig {
            max_draft: 15,
            fixed_draft: Some(4),
            ..cfg(None)
        });
        let mut drafter = FixedDrafter(vec![1; 15]);
        let mut delta = SpecDelta::default();
        one_step(&mut state, &mut v, &mut drafter, 9, 99, &mut delta).unwrap();
        assert!(v.verify_sizes().is_empty());
        assert_eq!(v.calls[0], Call::Decode(9));
        assert_eq!(delta.plain_steps, 1);
    }

    #[test]
    fn k_zero_is_one_decode_call() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        let mut drafter = ScriptedDrafter::right(&truth);
        let state = SpecState::new(cfg(Some(0)));
        let got = speculate(state, &mut drafter, Sampling::GREEDY, 0, 64, 15);
        assert!(got
            .verifier
            .calls
            .iter()
            .all(|c| matches!(c, Call::Decode(_))));
        // 63 steps: the first token came from the prefill, and the last was never decoded.
        assert_eq!(got.verifier.calls.len(), 63);
        assert_eq!(got.delta.plain_steps, 63);
        assert_eq!(got.delta.steps, 0);
        assert_eq!(got.state.totals().by_size[0].steps, 63);
        // The controller's own k = 0: an accept rate of 0 against a wrong drafter never gains, whatever the timing
        // (E(k) = 1 and c(k + 1) >= c(1), so the best score is below min_gain / c(1)).
        let mut drafter = ScriptedDrafter::wrong(&truth);
        let state = SpecState::new(SpecConfig {
            accept_prior: 0.0,
            ..cfg(None)
        });
        let got = speculate(state, &mut drafter, Sampling::GREEDY, 0, 64, 15);
        assert!(got
            .verifier
            .calls
            .iter()
            .all(|c| matches!(c, Call::Decode(_))));
        assert_eq!(got.delta.plain_steps, 63);
        assert_eq!(got.delta.steps, 0);
        assert_eq!(got.state.totals().accept_rate, 0.0);
        // A controller with no timed plain step yet also starts at k = 0.
        let mut drafter = ScriptedDrafter::right(&truth);
        let first = speculate(
            SpecState::new(cfg(None)),
            &mut drafter,
            Sampling::GREEDY,
            0,
            3,
            15,
        );
        assert!(matches!(first.verifier.calls[0], Call::Decode(_)));
    }

    #[test]
    fn fixed_draft_ignores_the_controller() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        let x = greedy_x();
        // The controller has timed nothing and would pick 0; the fixed length picks 3, or what the draft has.
        for (fixed, draft_len, want) in [(3, 15, 4), (4, 2, 3), (4, 0, 1)] {
            let mut v = ScriptedVerifier::new(&PROMPT);
            let mut state = SpecState::new(cfg(Some(fixed)));
            let mut drafter = ScriptedDrafter::new(&truth, &[plan(draft_len, None)]).primed(x);
            let mut delta = SpecDelta::default();
            one_step(&mut state, &mut v, &mut drafter, x, 15, &mut delta).unwrap();
            match &v.calls[0] {
                Call::Verify(t) => assert_eq!(t.len(), want),
                Call::Decode(_) => assert_eq!(want, 1),
                c => panic!("{c:?}"),
            }
        }
        let mut v = ScriptedVerifier::new(&PROMPT);
        let mut state = SpecState::new(cfg(None));
        let mut drafter = ScriptedDrafter::right(&truth).primed(x);
        let mut delta = SpecDelta::default();
        one_step(&mut state, &mut v, &mut drafter, x, 15, &mut delta).unwrap();
        assert_eq!(
            v.calls[0],
            Call::Decode(x),
            "the controller alone starts plain"
        );
    }

    #[test]
    fn an_end_inside_a_burst_commits_the_rows_read() {
        let (v, state, drafter, delta, stepped) = end_after(2, &[plan(4, None)]);
        assert_eq!(
            stepped,
            Stepped {
                kept: 3,
                ended: true
            }
        );
        let x = greedy_x();
        let drafted = stream(Sampling::GREEDY, 0, 64)[PROMPT.len() + 1..PROMPT.len() + 5].to_vec();
        assert_eq!(
            v.calls,
            vec![
                Call::Verify([&[x], drafted.as_slice()].concat()),
                Call::Row(0),
                Call::Row(1),
                Call::Row(2),
                Call::Commit(3),
            ]
        );
        assert_eq!(v.history, [PROMPT.as_slice(), &[x], &drafted[..2]].concat());
        let want = SpecDelta {
            steps: 1,
            plain_steps: 0,
            drafted: 4,
            accepted: 2,
            drafted_lookup: 4,
            accepted_lookup: 2,
            replays: 1,
            ..SpecDelta::default()
        };
        assert_eq!(delta, want);
        // The prompt, x and the two `Go` picks were passed to the drafter; the End row's token was not.
        assert_eq!(drafter.seen, PROMPT.len() + 1 + 2);
        // Both drafts were tested and matched; the End row tested none.
        let mut rate = Controller::new(&cfg(Some(4)));
        rate.observe_accept(Source::Lookup, 2, 2);
        assert_eq!(state.totals().accept_rate, rate.accept_rate(Source::Lookup));
    }

    #[test]
    fn a_mismatch_tests_the_rejected_draft_and_an_end_before_a_draft_tests_nothing() {
        let (v, state, drafter, delta, stepped) = end_after(10, &[plan(4, Some(1))]);
        assert_eq!(
            stepped,
            Stepped {
                kept: 2,
                ended: false
            }
        );
        assert_eq!(v.calls.last(), Some(&Call::Commit(2)));
        assert_eq!(drafter.seen, PROMPT.len() + 1 + 2);
        assert_eq!((delta.accepted, delta.replays), (1, 1));
        let mut rate = Controller::new(&cfg(Some(4)));
        rate.observe_accept(Source::Lookup, 2, 1);
        assert_eq!(state.totals().accept_rate, rate.accept_rate(Source::Lookup));
        // An End on row 0: nothing was compared.
        let (v, state, drafter, delta, stepped) = end_after(0, &[plan(4, None)]);
        assert_eq!(
            stepped,
            Stepped {
                kept: 1,
                ended: true
            }
        );
        assert_eq!(v.calls.last(), Some(&Call::Commit(1)));
        assert_eq!(drafter.seen, PROMPT.len() + 1);
        assert_eq!((delta.accepted, delta.replays), (0, 1));
        assert_eq!(state.totals().accept_rate, cfg(None).accept_prior);
    }

    #[test]
    fn at_k_zero_the_draft_scores_itself_unless_the_step_ends() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        let x = greedy_x();
        let run = |p: Plan, end: bool| {
            let mut v = ScriptedVerifier::new(&PROMPT);
            let mut state = SpecState::new(cfg(Some(0)));
            let mut drafter = ScriptedDrafter::new(&truth, &[p]).primed(x);
            let mut delta = SpecDelta::default();
            let mut pick = |row: &[f32]| {
                Ok(if end {
                    Next::End
                } else {
                    Next::Go(greedy(row))
                })
            };
            state
                .step(&mut v, &mut drafter, x, 15, &mut pick, &mut delta)
                .unwrap();
            (state.totals().accept_rate, delta)
        };
        let prior = cfg(None).accept_prior;
        let rate = |tested, accepted| {
            let mut c = Controller::new(&cfg(None));
            c.observe_accept(Source::Lookup, tested, accepted);
            c.accept_rate(Source::Lookup)
        };
        // The draft's first token matches: one hit. It doesn't: one miss. No draft, or an End: nothing tested.
        assert_eq!(run(plan(4, None), false).0, rate(1, 1));
        assert_eq!(run(plan(4, Some(0)), false).0, rate(1, 0));
        assert_eq!(run(plan(0, None), false).0, prior);
        let (r, d) = run(plan(4, None), true);
        assert_eq!(r, prior);
        assert_eq!(
            d,
            SpecDelta {
                plain_steps: 1,
                ..SpecDelta::default()
            }
        );
    }

    #[test]
    fn the_step_table_counts_each_size() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        let mut x = greedy_x();
        let mut v = ScriptedVerifier::new(&PROMPT);
        let mut state = SpecState::new(cfg(Some(4)));
        // Four drafts, all right; three with the second wrong; none.
        let plans = [plan(4, None), plan(3, Some(1)), plan(0, None)];
        let mut drafter = ScriptedDrafter::new(&truth, &plans).primed(x);
        let mut delta = SpecDelta::default();
        let mut kept = Vec::new();
        for _ in 0..3 {
            let mut last = 0;
            let s = state
                .step(
                    &mut v,
                    &mut drafter,
                    x,
                    15,
                    &mut |row| {
                        last = greedy(row);
                        Ok(Next::Go(last))
                    },
                    &mut delta,
                )
                .unwrap();
            kept.push(s.kept);
            x = last;
        }
        assert_eq!(kept, vec![5, 2, 1]);
        let want = SpecDelta {
            steps: 2,
            plain_steps: 1,
            drafted: 7,
            accepted: 5,
            drafted_lookup: 7,
            accepted_lookup: 5,
            replays: 1,
            ..SpecDelta::default()
        };
        assert_eq!(delta, want);
        let t = state.totals();
        assert_eq!(t.delta, want);
        for (i, row) in t.by_size.iter().enumerate() {
            // (steps, tokens kept, misses): a verify costs 5 misses, a decode 2, at 100 bytes each.
            let (steps, tokens, misses) = match i {
                4 => (1, 5, 5),
                3 => (1, 2, 5),
                0 => (1, 1, 2),
                _ => (0, 0, 0),
            };
            assert_eq!((row.steps, row.tokens), (steps, tokens), "size {}", i + 1);
            assert_eq!(row.miss_bytes, misses * 100, "size {}", i + 1);
            assert_eq!(row.ms > 0.0, steps > 0, "size {}", i + 1);
        }
        // The controller got each size's time.
        assert!(state.controller.cost_ms(1).is_some());
        // The caller's delta is its own: a fresh one counts only its steps, and the totals keep adding.
        let mut other = SpecDelta::default();
        let mut idle = ScriptedDrafter::none(&truth);
        one_step(&mut state, &mut v, &mut idle, x, 15, &mut other).unwrap();
        assert_eq!(
            other,
            SpecDelta {
                plain_steps: 1,
                ..SpecDelta::default()
            }
        );
        assert_eq!(state.totals().delta.plain_steps, 2);
    }

    #[test]
    fn the_steps_time_runs_to_its_end_picks_and_commit_included() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        let x = greedy_x();
        let mut v = ScriptedVerifier::new(&PROMPT);
        v.commit_sleep = Duration::from_millis(5);
        let mut state = SpecState::new(cfg(Some(2)));
        // A plain step first (no draft), then a verify of three tokens.
        let mut drafter = ScriptedDrafter::new(&truth, &[plan(0, None), plan(15, None)]).primed(x);
        let mut delta = SpecDelta::default();
        let mut x = x;
        for _ in 0..2 {
            let mut last = 0;
            // Each row picked takes 5 ms; the verify's commit takes another 5.
            state
                .step(
                    &mut v,
                    &mut drafter,
                    x,
                    15,
                    &mut |row| {
                        std::thread::sleep(Duration::from_millis(5));
                        last = greedy(row);
                        Ok(Next::Go(last))
                    },
                    &mut delta,
                )
                .unwrap();
            x = last;
        }
        let totals = state.totals();
        assert_eq!((totals.by_size[0].steps, totals.by_size[2].steps), (1, 1));
        assert!(totals.by_size[0].ms >= 5.0, "{} ms", totals.by_size[0].ms);
        assert!(totals.by_size[2].ms >= 20.0, "{} ms", totals.by_size[2].ms);
        // The controller got the same figures.
        assert_eq!(state.controller.cost_ms(1), Some(totals.by_size[0].ms));
        assert_eq!(state.controller.cost_ms(3), Some(totals.by_size[2].ms));
    }

    #[test]
    fn rows_past_the_stop_are_never_read() {
        // The draft's second token is wrong: a match fails at row 1 of 5.
        let (v, _, _, _, stepped) = end_after(10, &[plan(4, Some(1))]);
        assert_eq!(stepped.kept, 2);
        let rows: Vec<_> = v
            .calls
            .iter()
            .filter(|c| matches!(c, Call::Row(_)))
            .collect();
        assert_eq!(rows, vec![&Call::Row(0), &Call::Row(1)]);
    }

    /// One step of four right drafts against a verifier with these faults.
    fn failing(
        faults: Faults,
    ) -> (
        ScriptedVerifier,
        Result<Stepped, &'static str>,
        SpecDelta,
        SpecState,
    ) {
        let x = greedy_x();
        let mut v = ScriptedVerifier::new(&PROMPT);
        v.faults = faults;
        let mut state = SpecState::new(cfg(Some(4)));
        let mut drafter = ScriptedDrafter::right(&stream(Sampling::GREEDY, 0, 64)).primed(x);
        let mut delta = SpecDelta::default();
        let r = one_step(&mut state, &mut v, &mut drafter, x, 15, &mut delta);
        (v, r, delta, state)
    }

    #[test]
    fn a_row_error_returns_without_a_commit() {
        let (v, r, delta, state) = failing(Faults {
            row: Some(1),
            ..Faults::default()
        });
        assert_eq!(r, Err("row"));
        assert!(v.calls.iter().all(|c| !matches!(c, Call::Commit(_))));
        assert_eq!(v.calls.last(), Some(&Call::Row(1)));
        // A failed step counts nothing, and the verifier cut itself back.
        assert_eq!(delta, SpecDelta::default());
        assert_eq!(state.totals().delta, SpecDelta::default());
        assert_eq!(v.history, PROMPT);
    }

    #[test]
    fn a_verify_error_returns_at_once() {
        let (v, r, delta, _) = failing(Faults {
            verify: true,
            ..Faults::default()
        });
        assert_eq!(r, Err("verify"));
        assert_eq!(v.calls.len(), 1);
        assert_eq!(delta, SpecDelta::default());
    }

    #[test]
    fn a_commit_error_is_returned_and_counts_nothing() {
        let (v, r, delta, state) = failing(Faults {
            commit: true,
            ..Faults::default()
        });
        assert_eq!(r, Err("commit"));
        assert_eq!(v.calls.last(), Some(&Call::Commit(5)));
        assert_eq!(delta, SpecDelta::default());
        assert_eq!(state.totals().by_size[4].steps, 0);
    }

    #[test]
    fn a_pick_error_after_a_verify_commits_the_rows_read_and_a_plain_one_just_returns() {
        let x = greedy_x();
        let truth = stream(Sampling::GREEDY, 0, 64);
        let mut v = ScriptedVerifier::new(&PROMPT);
        let mut state = SpecState::new(cfg(Some(4)));
        let mut drafter = ScriptedDrafter::right(&truth).primed(x);
        let mut delta = SpecDelta::default();
        // A pick that fails on its second row.
        let mut calls = 0;
        let r = state.step(
            &mut v,
            &mut drafter,
            x,
            15,
            &mut |row| {
                calls += 1;
                if calls == 2 {
                    return Err("pick");
                }
                Ok(Next::Go(greedy(row)))
            },
            &mut delta,
        );
        assert_eq!(r, Err("pick"));
        assert_eq!(v.calls.last(), Some(&Call::Commit(2)));
        assert_eq!(v.history.len(), PROMPT.len() + 2);
        assert_eq!(delta, SpecDelta::default());
        // The commit that follows a pick error fails too: the pick's error is still the one returned, and the
        // verifier has cut itself back. A failed step counts nothing, though it did run: the request fails.
        let mut v = ScriptedVerifier::new(&PROMPT);
        v.faults.commit = true;
        let mut state = SpecState::new(cfg(Some(4)));
        let mut drafter = ScriptedDrafter::right(&truth).primed(x);
        let mut delta = SpecDelta::default();
        let mut calls = 0;
        let r = state.step(
            &mut v,
            &mut drafter,
            x,
            15,
            &mut |row| {
                calls += 1;
                if calls == 2 {
                    return Err("pick");
                }
                Ok(Next::Go(greedy(row)))
            },
            &mut delta,
        );
        assert_eq!(r, Err("pick"));
        assert_eq!(v.calls.last(), Some(&Call::Commit(2)));
        assert_eq!(v.history, PROMPT);
        assert_eq!(delta, SpecDelta::default());
        assert_eq!(state.totals().delta, SpecDelta::default());
        // The same error from a plain step's pick.
        let mut delta = SpecDelta::default();
        let mut drafter = ScriptedDrafter::right(&truth).primed(x);
        let mut v = ScriptedVerifier::new(&PROMPT);
        let mut state = SpecState::new(cfg(Some(0)));
        let r = state.step(
            &mut v,
            &mut drafter,
            x,
            15,
            &mut |_| Err("pick"),
            &mut delta,
        );
        assert_eq!(r, Err("pick"));
        assert_eq!(v.calls, vec![Call::Decode(x)]);
    }

    #[test]
    fn the_drafter_factory_builds_each_requests_drafter() {
        let state =
            SpecState::with_drafters(cfg(None), Box::new(|| Box::new(FixedDrafter(vec![1]))));
        assert_eq!(state.drafter().name(), "fixed");
        assert_eq!(SpecState::new(cfg(None)).drafter().name(), "lookup");
        assert_eq!(state.config(), &cfg(None));
    }

    // ---- two sources ----

    /// The measured costs: decode 10.4 ms, verify c(t) = 10.4 + 4.4 (t - 1), the draft pass d(k) = 1.8 k.
    fn measured_clock() -> Box<dyn FnMut(StepTimed) -> f64 + Send> {
        Box::new(|t| match t {
            StepTimed::Plain => 10.4,
            StepTimed::Verify { tokens } => 10.4 + 4.4 * (tokens - 1) as f64,
            StepTimed::MtpDraft { k } => 1.8 * k as f64,
        })
    }

    fn mtp_cfg(prior: &[f64]) -> SpecConfig {
        SpecConfig {
            mtp_accept_prior: prior.to_vec(),
            source_choice: SourceChoice::Priority,
            ..SpecConfig::default()
        }
    }

    fn timed_state(cfg: SpecConfig) -> SpecState {
        SpecState::new(cfg).with_clock(measured_clock())
    }

    fn right_mtp() -> MtpMode {
        MtpMode::Plans(vec![plan(15, None)])
    }

    /// Lookup's drafter that drafts (as `inner` does) on every third call only, as a stream with matches now and then.
    struct EveryThird {
        inner: ScriptedDrafter,
        calls: usize,
    }

    impl Drafter for EveryThird {
        fn name(&self) -> &'static str {
            "every-third"
        }
        fn draft(&mut self, max: usize) -> Vec<u32> {
            self.calls += 1;
            let d = self.inner.draft(max);
            if self.calls.is_multiple_of(3) {
                d
            } else {
                Vec::new()
            }
        }
        fn observe(&mut self, tokens: &[u32]) {
            self.inner.observe(tokens);
        }
    }

    /// `n` greedy steps from the prompt that never end; returns the last `x`.
    fn greedy_steps(
        state: &mut SpecState,
        v: &mut ScriptedVerifier,
        drafter: &mut dyn Drafter,
        n: usize,
        k_max: usize,
        delta: &mut SpecDelta,
    ) -> Result<u32, &'static str> {
        let mut x = greedy_x();
        drafter.observe(&PROMPT);
        drafter.observe(&[x]);
        for _ in 0..n {
            let mut last = x;
            state.step(
                v,
                drafter,
                x,
                k_max,
                &mut |row| {
                    last = greedy(row);
                    Ok(Next::Go(last))
                },
                delta,
            )?;
            x = last;
        }
        Ok(x)
    }

    #[test]
    fn any_mtp_draft_gives_the_plain_loops_tokens() {
        for (sampling, seed) in [(Sampling::GREEDY, 0), (Sampling::DEFAULT, 1234)] {
            let truth = stream(sampling, seed, 64);
            let want = plain(sampling, seed, 64);
            let plans = [
                ("right", plan(15, None)),
                ("wrong", plan(15, Some(0))),
                ("half", plan(15, Some(2))),
                ("short", plan(1, None)),
                ("none", plan(0, None)),
            ];
            for (name, p) in plans {
                for fixed in [Some(4), None] {
                    let state = timed_state(SpecConfig {
                        fixed_draft: fixed,
                        ..mtp_cfg(&[0.89, 0.81, 0.74, 0.78])
                    });
                    let v =
                        ScriptedVerifier::new(&PROMPT).with_mtp(&truth, MtpMode::Plans(vec![p]));
                    let mut none = ScriptedDrafter::none(&truth);
                    let got = speculate_on(state, v, &mut none, sampling, seed, 64, 15);
                    assert_eq!(got.out, want.out, "{name} {fixed:?} {sampling:?}");
                    assert_eq!(got.verifier.history, want.verifier.history, "{name}");
                    // The controller path drafts too: the prior wants MTP, and `right` is the plan it should keep.
                    if name != "none" && (fixed.is_some() || name == "right") {
                        assert!(
                            got.delta.drafted_mtp > 0,
                            "{name} {fixed:?}: MTP never drafted"
                        );
                    }
                    assert_eq!(got.delta.drafted_lookup, 0);
                }
            }
        }
    }

    #[test]
    fn bad_mtp_drafts_give_the_plain_loops_tokens() {
        let nv = VOCAB as u32;
        for (sampling, seed) in [(Sampling::GREEDY, 0), (Sampling::DEFAULT, 1234)] {
            let truth = stream(sampling, seed, 64);
            let want = plain(sampling, seed, 64);
            let bad: [(&str, Vec<u32>); 4] = [
                ("zero", vec![0; 4]),
                ("last", vec![nv - 1; 4]),
                ("repeat", vec![7, 7, 7, 7]),
                // Outside the vocabulary: cut before verify, never reaches the embedding lookup.
                ("outside", vec![3, nv, 5, 5]),
            ];
            for (name, tokens) in bad {
                let state = timed_state(SpecConfig {
                    fixed_draft: Some(4),
                    ..mtp_cfg(&[0.89])
                });
                let v = ScriptedVerifier::new(&PROMPT).with_mtp(&truth, MtpMode::Fixed(tokens));
                let mut none = ScriptedDrafter::none(&truth);
                let got = speculate_on(state, v, &mut none, sampling, seed, 64, 15);
                assert_eq!(got.out, want.out, "{name} {sampling:?}");
                assert_eq!(got.verifier.history, want.verifier.history, "{name}");
                assert!(got.delta.drafted_mtp > 0, "{name}");
            }
        }
    }

    #[test]
    fn lookup_drafts_take_priority_over_mtp() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        for fixed in [Some(4), None] {
            let state = timed_state(SpecConfig {
                fixed_draft: fixed,
                ..mtp_cfg(&[1.0])
            });
            // MTP would draft garbage: any step it took would show as a call.
            let v = ScriptedVerifier::new(&PROMPT).with_mtp(&truth, MtpMode::Fixed(vec![0; 4]));
            let mut lookup = ScriptedDrafter::right(&truth);
            let got = speculate_on(state, v, &mut lookup, Sampling::GREEDY, 0, 64, 15);
            assert!(got.delta.drafted_lookup > 0, "{fixed:?}");
            assert_eq!(got.delta.drafted_mtp, 0, "{fixed:?}");
            assert!(got.verifier.mtp_asks().is_empty(), "{fixed:?}");
            assert_eq!(got.delta.probes, 0);
        }
    }

    #[test]
    fn best_of_falls_back_to_lookups_draft_when_mtp_drafts_nothing() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        // Lookup at a rate of 0.7 against MTP at 0.95 a depth: best-of picks MTP, which returns nothing.
        let run = |choice| {
            let state = timed_state(SpecConfig {
                accept_prior: 0.7,
                source_choice: choice,
                ..mtp_cfg(&[0.95, 0.95, 0.95, 0.95])
            });
            let v = ScriptedVerifier::new(&PROMPT).with_mtp(&truth, MtpMode::Fixed(Vec::new()));
            let mut lookup = ScriptedDrafter::right(&truth);
            speculate_on(state, v, &mut lookup, Sampling::GREEDY, 0, 64, 15)
        };
        let want = plain(Sampling::GREEDY, 0, 64);
        let best = run(SourceChoice::Best);
        assert_eq!(best.out, want.out);
        assert!(!best.verifier.mtp_asks().is_empty());
        assert!(best.delta.drafted_lookup > 0, "{:?}", best.delta);
        // The step where MTP was asked and returned nothing verified lookup's draft rather than going plain.
        let calls = &best.verifier.calls;
        assert!(
            calls
                .windows(2)
                .any(|w| matches!((&w[0], &w[1]), (Call::Mtp(..), Call::Verify(_)))),
            "{calls:?}"
        );
        // Priority takes lookup's draft in the first place and never asks MTP.
        let priority = run(SourceChoice::Priority);
        assert!(priority.delta.drafted_lookup > 0 && priority.verifier.mtp_asks().is_empty());
    }

    #[test]
    fn mtp_drafts_when_lookup_has_nothing() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        let state = timed_state(mtp_cfg(&[0.89, 0.81, 0.74, 0.78]));
        let v = ScriptedVerifier::new(&PROMPT).with_mtp(&truth, right_mtp());
        let mut none = ScriptedDrafter::none(&truth);
        let got = speculate_on(state, v, &mut none, Sampling::GREEDY, 0, 64, 15);
        let want = plain(Sampling::GREEDY, 0, 64);
        assert_eq!(got.out, want.out);
        assert!(got.delta.drafted_mtp > 0 && got.delta.accepted_mtp > 0);
        assert_eq!(
            (got.delta.drafted_lookup, got.delta.accepted_lookup),
            (0, 0)
        );
        assert!(got.delta.steps < 64 / 2, "{:?}", got.delta);
        // The first step is plain: nothing is timed yet.
        assert!(matches!(got.verifier.calls[0], Call::Decode(_)));
        assert!(matches!(got.verifier.calls[1], Call::Mtp(_, _)));
    }

    #[test]
    fn an_mtp_draft_never_runs_past_k_max() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        let cases = [
            (1, MAX_VERIFY, 1),
            (2, MAX_VERIFY, 2),
            (15, MAX_VERIFY, 4),
            (15, 3, 2),
            (0, MAX_VERIFY, 0),
        ];
        for (k_max, most, want_max) in cases {
            let state = timed_state(SpecConfig {
                fixed_draft: Some(15),
                ..mtp_cfg(&[0.89])
            });
            let mut v = ScriptedVerifier::new(&PROMPT).with_mtp(&truth, right_mtp());
            v.most = most;
            let mut none = ScriptedDrafter::none(&truth);
            let got = speculate_on(state, v, &mut none, Sampling::GREEDY, 0, 40, k_max);
            let asks = got.verifier.mtp_asks();
            assert!(asks.iter().all(|&k| k <= want_max), "{k_max}: {asks:?}");
            assert_eq!(asks.is_empty(), want_max == 0, "{k_max}: {asks:?}");
            assert_eq!(got.out, plain(Sampling::GREEDY, 0, 40).out);
        }
        // The controller's choice too, with a rate that wants the longest draft.
        let state = timed_state(mtp_cfg(&[1.0]));
        let v = ScriptedVerifier::new(&PROMPT).with_mtp(&truth, right_mtp());
        let mut none = ScriptedDrafter::none(&truth);
        let got = speculate_on(state, v, &mut none, Sampling::GREEDY, 0, 40, 3);
        let asks = got.verifier.mtp_asks();
        assert!(!asks.is_empty() && asks.iter().all(|&k| k <= 3), "{asks:?}");
    }

    #[test]
    fn mtp_is_probed_every_probe_every_plain_steps_across_lookup_steps() {
        let truth = stream(Sampling::GREEDY, 0, 96);
        // Rate 0 everywhere: MTP's choice is 0 and only probes draft. Its probes are wrong, so the rate stays near 0.
        let state = timed_state(SpecConfig {
            probe_every: 4,
            ..mtp_cfg(&[0.0])
        });
        let v = ScriptedVerifier::new(&PROMPT)
            .with_mtp(&truth, MtpMode::Plans(vec![plan(15, Some(0))]));
        let mut lookup = EveryThird {
            inner: ScriptedDrafter::right(&truth),
            calls: 0,
        };
        let got = speculate_on(state, v, &mut lookup, Sampling::GREEDY, 0, 96, 15);
        assert_eq!(got.out, plain(Sampling::GREEDY, 0, 96).out);
        let asks = got.verifier.mtp_asks();
        assert!(asks.len() >= 4, "{asks:?}");
        assert!(asks.iter().all(|&k| k == 1), "{asks:?}");
        assert_eq!(got.delta.probes, asks.len() as u64);
        // One mtp_draft after each 4 plain steps since the last MTP draft; lookup's verifies don't count or reset.
        let (mut plain_steps, mut lookup_verifies) = (0, 0);
        let mut after_mtp = false;
        for c in &got.verifier.calls {
            match c {
                Call::Decode(_) => plain_steps += 1,
                Call::Mtp(..) => {
                    assert_eq!(plain_steps, 4, "{:?}", got.verifier.calls);
                    plain_steps = 0;
                    after_mtp = true;
                }
                Call::Verify(_) => {
                    if !after_mtp {
                        lookup_verifies += 1;
                    }
                    after_mtp = false;
                }
                _ => {}
            }
        }
        assert!(lookup_verifies >= 4, "lookup verified {lookup_verifies}");
    }

    #[test]
    fn a_fresh_state_drafts_with_mtp_at_the_prior() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        let run = |prior: &[f64]| {
            let mut state = timed_state(mtp_cfg(prior));
            let mut v = ScriptedVerifier::new(&PROMPT).with_mtp(&truth, right_mtp());
            let mut none = ScriptedDrafter::none(&truth);
            let mut delta = SpecDelta::default();
            greedy_steps(&mut state, &mut v, &mut none, 6, 15, &mut delta).unwrap();
            v
        };
        // [0.75, 0.70] with the measured costs: k = 1 scores 1.75 / 16.6 = 0.1054, over the bar 1.03 / 10.4 = 0.0990.
        let v = run(&[0.75, 0.70]);
        assert!(matches!(v.calls[0], Call::Decode(_)), "{:?}", v.calls);
        assert!(matches!(v.calls[1], Call::Mtp(_, 1)), "{:?}", v.calls);
        // A prior of 0.5 scores 1.5 / 16.6 = 0.0904: no draft before the first probe.
        let v = run(&[0.5]);
        assert!(v.mtp_asks().is_empty(), "{:?}", v.calls);
    }

    #[test]
    fn the_counts_split_by_source() {
        let truth = stream(Sampling::GREEDY, 0, 96);
        let state = timed_state(mtp_cfg(&[0.89, 0.81, 0.74, 0.78]));
        let v = ScriptedVerifier::new(&PROMPT)
            .with_mtp(&truth, MtpMode::Plans(vec![plan(15, Some(2))]));
        let mut lookup = EveryThird {
            inner: ScriptedDrafter::half(&truth),
            calls: 0,
        };
        let got = speculate_on(state, v, &mut lookup, Sampling::GREEDY, 0, 96, 15);
        let d = got.delta;
        assert!(d.drafted_lookup > 0 && d.drafted_mtp > 0, "{d:?}");
        assert_eq!(d.drafted, d.drafted_lookup + d.drafted_mtp);
        assert_eq!(d.accepted, d.accepted_lookup + d.accepted_mtp);
        assert!(d.accepted_lookup <= d.drafted_lookup && d.accepted_mtp <= d.drafted_mtp);
        let t = got.state.totals();
        assert_eq!(t.delta, d);
        let passes: u64 = t.mtp_by_k.iter().map(|r| r.steps).sum();
        let drafted: u64 = t
            .mtp_by_k
            .iter()
            .enumerate()
            .map(|(i, r)| r.steps * (i as u64 + 1))
            .sum();
        assert!(passes > 0);
        assert_eq!(drafted, d.drafted_mtp);
    }

    #[test]
    fn fixed_draft_uses_mtp_when_lookup_has_nothing() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        let cfg = SpecConfig {
            fixed_draft: Some(3),
            ..mtp_cfg(&[0.0])
        };
        // No lookup draft: MTP drafts N = 3 on the very first step, with no controller.
        let mut state = SpecState::new(cfg.clone());
        let mut v = ScriptedVerifier::new(&PROMPT).with_mtp(&truth, right_mtp());
        let mut none = ScriptedDrafter::none(&truth);
        let mut delta = SpecDelta::default();
        greedy_steps(&mut state, &mut v, &mut none, 1, 15, &mut delta).unwrap();
        assert_eq!(v.mtp_asks(), vec![3]);
        assert_eq!(v.verify_sizes(), vec![4]);
        assert_eq!(
            (delta.drafted_mtp, delta.accepted_mtp, delta.probes),
            (3, 3, 0)
        );
        // A lookup draft is taken instead, and MTP isn't asked.
        let mut state = SpecState::new(cfg);
        let mut v = ScriptedVerifier::new(&PROMPT).with_mtp(&truth, right_mtp());
        let mut lookup = ScriptedDrafter::right(&truth);
        let mut delta = SpecDelta::default();
        greedy_steps(&mut state, &mut v, &mut lookup, 1, 15, &mut delta).unwrap();
        assert!(v.mtp_asks().is_empty());
        assert_eq!((delta.drafted_lookup, delta.drafted_mtp), (3, 0));
    }

    #[test]
    fn an_mtp_error_returns_without_a_verify() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        let mut state = SpecState::new(SpecConfig {
            fixed_draft: Some(3),
            ..mtp_cfg(&[0.9])
        });
        let mut v = ScriptedVerifier::new(&PROMPT).with_mtp(&truth, MtpMode::Fail);
        let mut none = ScriptedDrafter::none(&truth);
        let mut delta = SpecDelta::default();
        let x = greedy_x();
        none.observe(&PROMPT);
        none.observe(&[x]);
        let err = state
            .step(&mut v, &mut none, x, 15, &mut |_| Ok(Next::End), &mut delta)
            .unwrap_err();
        assert_eq!(err, "mtp");
        assert_eq!(v.calls, vec![Call::Mtp(x, 3)]);
        assert_eq!(v.history, PROMPT.to_vec());
        assert_eq!(delta, SpecDelta::default());
        assert_eq!(state.totals().delta, SpecDelta::default());
        assert!(state.totals().mtp_by_k.iter().all(|r| r.steps == 0));
    }

    #[test]
    fn without_mtp_the_step_is_5as() {
        let truth = stream(Sampling::GREEDY, 0, 96);
        let lookup_only = SpecConfig {
            sources: vec![Source::Lookup],
            ..SpecConfig::default()
        };
        let both = SpecConfig::default();
        let calls = |cfg: SpecConfig, mtp: bool| {
            let mut v = ScriptedVerifier::new(&PROMPT);
            if mtp {
                v = v.with_mtp(&truth, right_mtp());
            }
            let mut lookup = ScriptedDrafter::new(&truth, &[plan(15, Some(2)), plan(15, None)]);
            let got = speculate_on(
                timed_state(cfg),
                v,
                &mut lookup,
                Sampling::GREEDY,
                0,
                96,
                15,
            );
            (got.verifier.calls, got.delta, got.out)
        };
        let base = calls(lookup_only.clone(), false);
        assert!(base.1.steps > 0);
        // sources = ["lookup"] with an MTP session, and MTP as a source without one: both are the lookup-only loop.
        assert_eq!(calls(lookup_only, true), base);
        assert_eq!(calls(both, false), base);
        assert_eq!(base.1.drafted_mtp + base.1.probes, 0);
        assert_eq!(base.1.drafted, base.1.drafted_lookup);

        // A fixed draft too, against a log taken from the lookup-only loop on this script: 24 tokens, a draft of 3.
        let fixed = |cfg: SpecConfig, mtp: bool| {
            let mut v = ScriptedVerifier::new(&PROMPT);
            if mtp {
                v = v.with_mtp(&truth, right_mtp());
            }
            let mut lookup = ScriptedDrafter::new(&truth, &[plan(15, Some(2)), plan(15, None)]);
            let cfg = SpecConfig {
                fixed_draft: Some(3),
                ..cfg
            };
            let got = speculate_on(
                timed_state(cfg),
                v,
                &mut lookup,
                Sampling::GREEDY,
                0,
                24,
                15,
            );
            let log: Vec<String> = got
                .verifier
                .calls
                .iter()
                .map(|c| format!("{c:?}"))
                .collect();
            (log.join(" "), got.delta)
        };
        let from_5a = (
            "Verify([9, 12, 16, 6]) Row(0) Row(1) Row(2) Commit(3) Verify([5, 8, 12, 0]) Row(0) Row(1) Row(2) Row(3) Commit(4) Verify([13, 20, 18, 13]) Row(0) Row(1) Row(2) Row(3) Commit(4) Verify([3, 15, 21, 15]) Row(0) Row(1) Row(2) Row(3) Commit(4) Verify([21, 15, 21, 15]) Row(0) Row(1) Row(2) Row(3) Commit(4) Verify([21, 15, 21, 15]) Row(0) Row(1) Row(2) Row(3) Commit(4)",
            SpecDelta {
                steps: 6,
                plain_steps: 0,
                drafted: 18,
                accepted: 17,
                drafted_lookup: 18,
                accepted_lookup: 17,
                replays: 1,
                ..SpecDelta::default()
            },
        );
        for (cfg, mtp) in [
            (lookup_only_cfg(), true),
            (lookup_only_cfg(), false),
            (SpecConfig::default(), false),
        ] {
            let (log, delta) = fixed(cfg, mtp);
            assert_eq!((log.as_str(), delta), from_5a);
        }
    }

    fn lookup_only_cfg() -> SpecConfig {
        SpecConfig {
            sources: vec![Source::Lookup],
            ..SpecConfig::default()
        }
    }

    #[test]
    fn the_depth_counts_add_up() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        let mut state = SpecState::new(SpecConfig {
            fixed_draft: Some(4),
            ..mtp_cfg(&[0.5])
        });
        // Two passes of 4 with the third wrong: depths 1 and 2 kept, depth 3 rejected, depth 4 untested. Then one pass
        // of 4 all right.
        let mut v = ScriptedVerifier::new(&PROMPT).with_mtp(
            &truth,
            MtpMode::Plans(vec![plan(4, Some(2)), plan(4, Some(2)), plan(4, None)]),
        );
        let mut none = ScriptedDrafter::none(&truth);
        let mut delta = SpecDelta::default();
        greedy_steps(&mut state, &mut v, &mut none, 3, 15, &mut delta).unwrap();
        let t = state.totals();
        assert_eq!(
            &t.mtp_depth_counts[..5],
            &[(3, 3), (3, 3), (3, 1), (1, 1), (0, 0)]
        );
        assert!(t.mtp_depth_counts[5..].iter().all(|&c| c == (0, 0)));
        assert_eq!((delta.drafted_mtp, delta.accepted_mtp), (12, 8));
        // The passes by size: three of 4, with the drafts kept.
        assert_eq!(t.mtp_by_k[3].steps, 3);
        assert_eq!(t.mtp_by_k[3].tokens, 8);
        assert!(t
            .mtp_by_k
            .iter()
            .enumerate()
            .all(|(i, r)| i == 3 || r.steps == 0));
        // The rates moved at the depths tested, and the others kept the prior.
        assert!(t.depth_rates[0] > 0.5 && t.depth_rates[2] < 0.5);
        assert_eq!(t.depth_rates[5], 0.5);
        assert_eq!(t.mtp_accept_rate, t.depth_rates[0]);
    }

    #[test]
    fn a_pass_that_returns_fewer_drafts_sets_k_and_an_empty_one_makes_the_step_plain() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        let mut state = timed_state(SpecConfig {
            fixed_draft: Some(4),
            ..mtp_cfg(&[0.9])
        });
        let mut v = ScriptedVerifier::new(&PROMPT)
            .with_mtp(&truth, MtpMode::Plans(vec![plan(2, None), plan(0, None)]));
        let mut none = ScriptedDrafter::none(&truth);
        let mut delta = SpecDelta::default();
        greedy_steps(&mut state, &mut v, &mut none, 2, 15, &mut delta).unwrap();
        // Step 1: 2 drafts, verify of 3. Step 2: nothing back, a plain decode.
        assert_eq!(v.mtp_asks(), vec![4, 4]);
        assert_eq!(v.verify_sizes(), vec![3]);
        assert!(matches!(v.calls.last(), Some(Call::Decode(_))));
        let t = state.totals();
        assert_eq!(t.mtp_by_k[1].steps, 1);
        assert_eq!(
            (delta.steps, delta.plain_steps, delta.drafted_mtp),
            (1, 1, 2)
        );
    }

    #[test]
    fn an_unverified_draft_is_scored_against_the_picks_in_order() {
        // All four kept by the picks (the fifth pick is the step's own token).
        assert_eq!(score_against(&[1, 2, 3, 4], &[1, 2, 3, 4, 9]), (4, 4));
        // A mismatch at the third: two kept and one rejection; the fourth is untested.
        assert_eq!(score_against(&[1, 2, 3, 4], &[1, 2, 7, 4, 9]), (3, 2));
        // Fewer picks than drafts (a stop, or MTP kept less): only the picks' positions are tested.
        assert_eq!(score_against(&[1, 2, 3, 4], &[1, 2]), (2, 2));
        assert_eq!(score_against(&[1, 2, 3, 4], &[5]), (1, 0));
        assert_eq!(score_against(&[1, 2], &[]), (0, 0));
        assert_eq!(score_against(&[], &[1, 2]), (0, 0));
    }

    /// Lookup's drafter whose calls alternate: a right draft from a continuing run, then a wrong one from a 2-token
    /// match. With `classes` off it hides the class, as a drafter that can't say does.
    struct Classed {
        inner: ScriptedDrafter,
        calls: usize,
        last: Option<LookupMatch>,
        classes: bool,
    }

    impl Drafter for Classed {
        fn name(&self) -> &'static str {
            "classed"
        }
        fn draft(&mut self, max: usize) -> Vec<u32> {
            self.calls += 1;
            let mut d = self.inner.draft(max);
            let run = self.calls % 2 == 1;
            if !run {
                if let Some(t) = d.first_mut() {
                    *t = (*t + 1) % VOCAB as u32;
                }
            }
            self.last = (!d.is_empty()).then_some(LookupMatch {
                n: if run { 4 } else { 2 },
                continues: run,
            });
            d
        }
        fn observe(&mut self, tokens: &[u32]) {
            self.inner.observe(tokens);
        }
        fn last_match(&self) -> Option<LookupMatch> {
            self.last.filter(|_| self.classes)
        }
    }

    #[test]
    fn the_draft_class_steers_lookup_against_mtp_and_the_tokens_stay_the_plain_loops() {
        let truth = stream(Sampling::GREEDY, 0, 600);
        let want = plain(Sampling::GREEDY, 0, 600);
        let run = |classes: bool| {
            let cfg = SpecConfig {
                source_choice: SourceChoice::Best,
                ..mtp_cfg(&[0.95, 0.95, 0.95, 0.95])
            };
            let v = ScriptedVerifier::new(&PROMPT).with_mtp(&truth, right_mtp());
            let mut lookup = Classed {
                inner: ScriptedDrafter::right(&truth),
                calls: 0,
                last: None,
                classes,
            };
            speculate_on(
                timed_state(cfg),
                v,
                &mut lookup,
                Sampling::GREEDY,
                0,
                600,
                15,
            )
        };
        let pooled = run(false);
        let split = run(true);
        assert_eq!(pooled.out, want.out);
        assert_eq!(split.out, want.out);
        // One rate for both kinds sits near a half: MTP, always right, takes the steps.
        assert!(
            split.delta.drafted_lookup > pooled.delta.drafted_lookup + 40,
            "split {:?} pooled {:?}",
            split.delta,
            pooled.delta
        );
        // The continuing drafts are right, the 2-token ones wrong: what lookup verifies with classes is kept.
        let kept = split.delta.accepted_lookup as f64 / split.delta.drafted_lookup as f64;
        assert!(kept > 0.9, "{:?}", split.delta);
        // Its classes are its own: the pooled rate is lookup's figure in the totals either way.
        let c = &split.state.controller;
        assert!(
            c.lookup_rate(Some(LookupMatch {
                n: 4,
                continues: true
            })) > 0.9
        );
        assert!(
            c.lookup_rate(Some(LookupMatch {
                n: 2,
                continues: false
            })) < 0.2
        );
        assert_eq!(
            split.state.totals().accept_rate,
            c.accept_rate(Source::Lookup)
        );
    }

    #[test]
    fn lookup_keeps_scoring_its_draft_on_steps_mtp_verifies() {
        // Lookup drafts wrong for 12 calls, so its rate falls and its choice becomes 0: MTP takes over. Lookup
        // then drafts right, and scored against every token the MTP step picked it earns its rate back and takes
        // steps again.
        let truth = stream(Sampling::GREEDY, 0, 200);
        let mut plans = vec![plan(15, Some(0)); 12];
        plans.push(plan(15, None));
        let state = timed_state(mtp_cfg(&[0.89, 0.81, 0.74, 0.78]));
        let v = ScriptedVerifier::new(&PROMPT).with_mtp(&truth, right_mtp());
        let mut lookup = ScriptedDrafter::new(&truth, &plans);
        let got = speculate_on(state, v, &mut lookup, Sampling::GREEDY, 0, 200, 15);
        assert!(got.delta.drafted_mtp > 0);
        let (mut after_mtp, mut mtp_seen, mut late_lookup) = (false, false, 0);
        for c in &got.verifier.calls {
            match c {
                Call::Mtp(..) => {
                    mtp_seen = true;
                    after_mtp = true;
                }
                Call::Verify(_) => {
                    if mtp_seen && !after_mtp {
                        late_lookup += 1;
                    }
                    after_mtp = false;
                }
                _ => {}
            }
        }
        assert!(late_lookup > 0, "lookup never came back: {:?}", got.delta);
    }

    /// A lookup drafter that always reports one class of match for the draft `inner` makes.
    struct Tagged {
        inner: ScriptedDrafter,
        class: LookupMatch,
    }

    impl Drafter for Tagged {
        fn name(&self) -> &'static str {
            "tagged"
        }
        fn draft(&mut self, max: usize) -> Vec<u32> {
            self.inner.draft(max)
        }
        fn observe(&mut self, tokens: &[u32]) {
            self.inner.observe(tokens);
        }
        fn last_match(&self) -> Option<LookupMatch> {
            Some(self.class)
        }
    }

    const RUN4: LookupMatch = LookupMatch {
        n: 4,
        continues: true,
    };

    /// One greedy step from `x` on `state`; returns the next `x` (the step must not end).
    fn greedy_step(
        state: &mut SpecState,
        v: &mut ScriptedVerifier,
        drafter: &mut dyn Drafter,
        x: u32,
        delta: &mut SpecDelta,
    ) -> u32 {
        let mut s = Sampler::new(Sampling::GREEDY, 0);
        let mut next = None;
        state
            .step(
                v,
                drafter,
                x,
                15,
                &mut |row| {
                    let y = s.sample(row);
                    next = Some(y);
                    Ok(Next::Go(y))
                },
                delta,
            )
            .unwrap();
        next.unwrap()
    }

    #[test]
    fn an_mtp_step_scores_lookups_whole_draft_against_every_pick() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        let cfg = SpecConfig {
            source_choice: SourceChoice::Best,
            ..mtp_cfg(&[0.95, 0.95, 0.95, 0.95])
        };
        let mut state = timed_state(cfg);
        // MTP's drafts are right for two tokens and wrong at the third: it keeps 2 and the step picks 3 tokens.
        let mut v = ScriptedVerifier::new(&PROMPT)
            .with_mtp(&truth, MtpMode::Plans(vec![plan(15, Some(2))]));
        // Lookup's draft is right throughout: all 3 picks are hits for it, whatever MTP kept.
        let mut lookup = Tagged {
            inner: ScriptedDrafter::right(&truth),
            class: RUN4,
        };
        let mut delta = SpecDelta::default();
        let mut x = greedy(&v.prefill_row());
        lookup.observe(&PROMPT);
        lookup.observe(&[x]);
        let keep = 0.5f64.powf(1.0 / 32.0);
        for _ in 0..8 {
            let before = state.controller.lookup_rate(Some(RUN4));
            let (calls, accepted_mtp) = (v.calls.len(), delta.accepted_mtp);
            x = greedy_step(&mut state, &mut v, &mut lookup, x, &mut delta);
            if !v.calls[calls..].iter().any(|c| matches!(c, Call::Mtp(..))) {
                continue;
            }
            assert_eq!(delta.accepted_mtp - accepted_mtp, 2, "MTP kept 2");
            // The class has a rate by now (the first, plain step scored it), so three hits blend into it.
            let mut want = before;
            for _ in 0..3 {
                want = keep * want + (1.0 - keep);
            }
            let got = state.controller.lookup_rate(Some(RUN4));
            assert!(
                (got - want).abs() < 1e-12,
                "got {got} want {want} from {before}"
            );
            return;
        }
        panic!("no MTP step ran: {:?}", delta);
    }

    #[test]
    fn the_draft_class_goes_with_lookups_draft_when_mtp_drafts_nothing() {
        let truth = stream(Sampling::GREEDY, 0, 64);
        let cfg = SpecConfig {
            source_choice: SourceChoice::Best,
            ..mtp_cfg(&[0.95, 0.95, 0.95, 0.95])
        };
        let mut state = timed_state(cfg);
        // The run class is kept 60% of the time while the pooled rate is far lower: only the class makes lookup's
        // draft worth verifying, and MTP, at 0.95 a depth, outscores it so is asked first.
        for _ in 0..10 {
            state
                .controller
                .observe_accept_as(Source::Lookup, Some(RUN4), 1, 1);
        }
        for _ in 0..60 {
            state.controller.observe_accept(Source::Lookup, 1, 0);
        }
        assert!(state.controller.lookup_rate(Some(RUN4)) > 0.55);
        assert!(state.controller.accept_rate(Source::Lookup) < 0.2);
        let mut v = ScriptedVerifier::new(&PROMPT).with_mtp(&truth, MtpMode::Fixed(Vec::new()));
        let mut lookup = Tagged {
            inner: ScriptedDrafter::right(&truth),
            class: RUN4,
        };
        let mut delta = SpecDelta::default();
        let mut x = greedy(&v.prefill_row());
        lookup.observe(&PROMPT);
        lookup.observe(&[x]);
        for _ in 0..8 {
            let calls = v.calls.len();
            x = greedy_step(&mut state, &mut v, &mut lookup, x, &mut delta);
            let new = &v.calls[calls..];
            if let Some(i) = new.iter().position(|c| matches!(c, Call::Mtp(..))) {
                assert!(
                    matches!(new.get(i + 1), Some(Call::Verify(_))),
                    "MTP drafted nothing and the step should verify lookup's draft: {new:?}"
                );
                return;
            }
        }
        panic!("MTP was never asked: {:?}", delta);
    }
}
