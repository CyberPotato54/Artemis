//! Test 6's pure side: where the snapshot is cut, which prompt dirties the
//! engine, the order of calls that builds each warm path (`drive`, through `WarmOps`, so CI runs it on a fake), and the
//! pass rule over the cold run and both paths. `artemis warm` and `artemis test6` put the real engine and the files
//! under it.
//!
//! Each golden prompt runs twice, once per path, each in a fresh process. Path 1 prefills the first `cut` tokens and
//! snapshots; path 2 prefills the whole prompt, decodes the oracle's first tokens and snapshots. Then another golden
//! prompt dirties the engine, the snapshot is restored, and the oracle's tokens are forced as in test 3. Each path's
//! flips, pooled over the prompts, may be at most `BOUND`, and prompt 20's prefilled positions after the restore must
//! equal the cold run's bit for bit.
use crate::generate::{self, Fingerprints, Record, Step, Stepper, Stop};
use crate::test3::{self, PromptResult, Status};
use anyhow::{bail, ensure, Result};
use engine::spec::{Drafter, Next, SpecDelta, SpecState, Verifier};
use serde::Serialize;

/// Each path's pooled flips may be at most this many: test 3's bound, 1.5 × the floor's 32.
pub const BOUND: usize = 48;
/// The flip count test 3 gave when the speed work closed, reported beside the cold run's.
pub const M2_COLD_FLIPS: usize = 29;
/// The long golden prompt, whose cut is chosen for the bit-for-bit gate.
pub const LONG_ID: &str = "20-long-option-rs";
/// Its cut: a multiple of 2,048 and 4,096 near the middle of its 27,416 tokens, so the rest of the prompt
/// prefills in the cold run's chunks (pinned by a test below).
pub const LONG_CUT: usize = 12_288;

/// Path 1's cut: half the prompt, in mid-chunk, but `LONG_CUT` for the long prompt.
pub fn cut_for(id: &str, n: usize) -> usize {
    if id == LONG_ID {
        LONG_CUT
    } else {
        n / 2
    }
}

/// Path 2's split: the judged steps before the snapshot, half the oracle's steps:
/// 128 for a full 256, and half for the prompts whose oracle stopped early.
pub fn split_for(steps: usize) -> usize {
    steps / 2
}

/// The prompt that dirties the engine for `id`: the next id in sorted order, the first after the last. `None` when
/// there is no other prompt.
pub fn dirtier<'a>(ids: &'a [String], id: &str) -> Option<&'a str> {
    let next = ids
        .iter()
        .filter(|i| i.as_str() > id)
        .min()
        .or_else(|| ids.iter().min())?;
    (next != id).then_some(next.as_str())
}

/// Where a warm path's snapshot is taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    /// Path 1: after prefilling the first `cut` tokens of the prompt.
    Prefill { cut: usize },
    /// Path 2: after prefilling the prompt and decoding `split - 1` forced tokens, so the logits of the first `split`
    /// steps come from before the restore.
    Decode { split: usize },
}

/// What a restore did, for the record.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RestoreFacts {
    pub bytes: u64,
    /// The first position of the live KV the restore copied back.
    pub kv_from: usize,
    pub ms: f64,
}

/// The engine as `drive` needs it: a session and the snapshot machinery over it. The real one is `artemis warm`'s;
/// CI's records the order of calls.
pub trait WarmOps {
    /// Positions left in the session.
    fn room(&self) -> usize;
    /// Prefills `tokens` at the next positions and returns the last position's logits; `each` gets every position's
    /// logits when `all`, else only the last's.
    fn prefill(
        &mut self,
        tokens: &[u32],
        all: bool,
        each: &mut dyn FnMut(&[f32]),
    ) -> Result<Vec<f32>>;
    fn decode(&mut self, token: u32) -> Result<Vec<f32>>;
    /// A verify pass over `tokens = [x, d1 .. dk]` (`Session::verify`): the rows are then read with `row` and the step
    /// is settled by `commit`.
    fn verify(&mut self, tokens: &[u32]) -> Result<()>;
    /// Row `i` of the pending verify.
    fn row(&mut self, i: usize) -> Result<Vec<f32>>;
    /// Keeps the first `kept` tokens of the pending verify.
    fn commit(&mut self, kept: usize) -> Result<()>;
    /// The model's vocabulary size: a draft id past it is dropped.
    fn n_vocab(&self) -> usize;
    /// MTP's drafts after `x` (`Verifier::mtp_draft`); none by default.
    fn mtp_draft(&mut self, x: u32, k: usize) -> Result<Vec<u32>> {
        let _ = (x, k);
        Ok(Vec::new())
    }
    /// Whether the session can draft with MTP.
    fn has_mtp(&self) -> bool {
        false
    }
    /// The engine's routed-expert misses so far, and the bytes one reads (the step table's; this test reports neither).
    fn misses(&self) -> u64 {
        0
    }
    fn miss_bytes(&self) -> u64 {
        0
    }
    /// Snapshots the live state and returns its bytes.
    fn snapshot(&mut self) -> Result<u64>;
    /// Resets and prefills `tokens` from position 0, then (the real engine) poisons the live KV from `tokens.len()` up
    /// to the snapshot's position, so a restore that skips the KV copy can't pass by luck.
    fn dirty(&mut self, tokens: &[u32]) -> Result<()>;
    fn restore(&mut self) -> Result<RestoreFacts>;
}

/// One warm run: the oracle's tokens forced, the engine's own top two at each step, every position's fingerprint.
#[derive(Clone, Debug)]
pub struct Drive {
    pub steps: Vec<Step>,
    pub stop: Stop,
    /// As `Record::positions`: the prompt's positions with `all_logits` (else its last), then every decoded one.
    pub positions: Vec<String>,
    pub snapshot_bytes: u64,
    pub restore: RestoreFacts,
    /// The speculative steps' counts; `None` for a run without speculation.
    pub spec: Option<SpecDelta>,
}

/// `WarmOps` as the step loop's verifier, errors in anyhow's.
struct WarmVerifier<'a, O: WarmOps>(&'a mut O);

impl<O: WarmOps> Verifier for WarmVerifier<'_, O> {
    type Error = anyhow::Error;

    fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        self.0.decode(token)
    }

    fn verify(&mut self, tokens: &[u32]) -> Result<()> {
        self.0.verify(tokens)
    }

    fn row(&mut self, i: usize) -> Result<Vec<f32>> {
        self.0.row(i)
    }

    fn commit(&mut self, kept: usize) -> Result<()> {
        self.0.commit(kept)
    }

    fn n_vocab(&self) -> usize {
        self.0.n_vocab()
    }

    fn mtp_draft(&mut self, x: u32, k: usize) -> Result<Vec<u32>> {
        self.0.mtp_draft(x, k)
    }

    fn has_mtp(&self) -> bool {
        self.0.has_mtp()
    }

    fn misses(&self) -> u64 {
        self.0.misses()
    }

    fn miss_bytes(&self) -> u64 {
        self.0.miss_bytes()
    }
}

/// `drive`'s stepper: plain decode, or speculation, with path 2's snapshot, dirtying and restore between the steps
/// that straddle the split. The snapshot is taken when exactly `split - 1` positions are decoded, so a burst never runs
/// past it: the steps before it are capped at the split.
struct Warmer<'a, O: WarmOps> {
    ops: &'a mut O,
    spec: Option<(&'a mut SpecState, Box<dyn Drafter>, SpecDelta)>,
    dirty: &'a [u32],
    /// Path 2's split; `None` for path 1, whose snapshot is taken before any decode.
    split: Option<usize>,
    decoded: usize,
    warmed: bool,
    snapshot_bytes: u64,
    restore: RestoreFacts,
}

impl<O: WarmOps> Stepper for Warmer<'_, O> {
    fn begin(&mut self, first: u32) {
        if let Some((_, drafter, _)) = &mut self.spec {
            drafter.observe(&[first]);
        }
    }

    fn step(
        &mut self,
        x: u32,
        most: usize,
        pick: &mut dyn FnMut(&[f32]) -> Result<Next>,
    ) -> Result<usize> {
        let mut most = most;
        if let Some(split) = self.split.filter(|_| !self.warmed) {
            if self.decoded == split - 1 {
                self.snapshot_bytes = self.ops.snapshot()?;
                self.ops.dirty(self.dirty)?;
                self.restore = self.ops.restore()?;
                self.warmed = true;
            } else {
                // The step ends at the split: `split - 1` positions are decoded before the snapshot, exactly.
                most = most.min(split - 1 - self.decoded);
            }
        }
        let ran = match &mut self.spec {
            None => {
                let row = self.ops.decode(x)?;
                pick(&row)?;
                1
            }
            Some((state, drafter, delta)) => {
                state
                    .step(
                        &mut WarmVerifier(&mut *self.ops),
                        drafter.as_mut(),
                        x,
                        most.saturating_sub(1),
                        pick,
                        delta,
                    )?
                    .kept
            }
        };
        self.decoded += ran;
        Ok(ran)
    }
}

/// Builds the warm state along `path` and decodes `forced` from it, in `generate::run`'s order. With `spec`, the
/// decoding speculates (the oracle's tokens are what the drafts are checked against).
///
/// Path 1: prefill `prompt[..cut]`, snapshot, dirty, restore, prefill `prompt[cut..]`, then decode the forced tokens
/// (all but the last, which `run` never decodes). Path 2: prefill `prompt`, decode `forced[..split - 1]`, snapshot,
/// dirty, restore, then decode `forced[split - 1..]`. The state at the snapshot holds no logits, so the
/// restore comes before the decode of `forced[split - 1]`, whose logits (the step after) are the first from the
/// restored state; steps `0..split` are judged from the logits before it. A speculative step never runs past the
/// split: the snapshot is taken at a step boundary, with exactly `split - 1` positions decoded.
#[allow(clippy::too_many_arguments)]
pub fn drive(
    ops: &mut impl WarmOps,
    prompt: &[u32],
    dirty: &[u32],
    path: Path,
    forced: &[u32],
    eog: &[u32],
    all_logits: bool,
    spec: Option<&mut SpecState>,
) -> Result<Drive> {
    ensure!(
        !prompt.is_empty(),
        "test 6 drives a prompt of at least one token"
    );
    ensure!(
        !dirty.is_empty(),
        "test 6 dirties with a prompt of at least one token"
    );
    generate::check_forced(forced, forced.len(), eog)?;
    let mut positions = Fingerprints::new(true, prompt.len() + forced.len());
    let (mut snapshot_bytes, mut restore) = (0, RestoreFacts::default());
    let (logits, split) = match path {
        Path::Prefill { cut } => {
            ensure!(
                cut > 0 && cut < prompt.len(),
                "path 1's cut {cut} is not inside the {}-token prompt",
                prompt.len()
            );
            ops.prefill(&prompt[..cut], all_logits, &mut |l| {
                if all_logits {
                    positions.add(l);
                }
            })?;
            snapshot_bytes = ops.snapshot()?;
            ops.dirty(dirty)?;
            restore = ops.restore()?;
            let last = ops.prefill(&prompt[cut..], all_logits, &mut |l| positions.add(l))?;
            (last, None)
        }
        Path::Decode { split } => {
            ensure!(split > 0, "path 2's split is at least one step");
            let last = ops.prefill(prompt, all_logits, &mut |l| positions.add(l))?;
            (last, Some(split))
        }
    };
    let room = ops.room();
    let spec = spec.map(|state| {
        let mut drafter = state.drafter();
        drafter.observe(prompt);
        (state, drafter, SpecDelta::default())
    });
    let mut warmer = Warmer {
        ops,
        spec,
        dirty,
        split,
        decoded: 0,
        warmed: split.is_none(),
        snapshot_bytes,
        restore,
    };
    // The prompt's last position was fingerprinted by the prefill: every later row `run` picks from is a decoded one.
    let mut first = true;
    let mut force = generate::forced(forced);
    let (steps, stop) = generate::run(
        logits,
        forced.len(),
        room,
        eog,
        |l, top| {
            if !std::mem::take(&mut first) {
                positions.add(l);
            }
            force(l, top)
        },
        &mut warmer,
        |_| Ok(()),
    )?;
    if !warmer.warmed {
        bail!(
            "the run decoded {} tokens and ended before path 2's split at step {}",
            warmer.decoded,
            split.unwrap_or(0)
        );
    }
    Ok(Drive {
        steps,
        stop,
        positions: positions.into_list(),
        snapshot_bytes: warmer.snapshot_bytes,
        restore: warmer.restore,
        spec: warmer.spec.map(|(_, _, delta)| delta),
    })
}

/// The bit-for-bit gate: the cold and warm runs' fingerprints of positions `from..to`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BitCheck {
    pub id: String,
    pub from: usize,
    pub to: usize,
    pub equal: bool,
    /// The first position in `from..to` whose fingerprints differ.
    pub first_difference: Option<usize>,
    /// Why the two records can't be compared (which also fails the gate).
    pub problem: Option<String>,
}

/// Compares the cold and warm records' fingerprints of positions `from..to`, which `--all-logits` records for every
/// prompt position, at index = position.
pub fn bit_check(id: &str, cold: &Record, warm: &Record, from: usize, to: usize) -> BitCheck {
    let problem = (|| {
        if !cold.settings.all_logits || !warm.settings.all_logits {
            return Some(format!(
                "both runs need --all-logits to fingerprint the prompt's positions (cold: {}, warm: {})",
                cold.settings.all_logits, warm.settings.all_logits
            ));
        }
        if cold.prompt_tokens != warm.prompt_tokens {
            return Some("the cold and warm runs are of different prompts".into());
        }
        if from >= to {
            return Some(format!("the range {from}..{to} is empty"));
        }
        // Past the prompt, the fingerprints are decode positions, not prefilled ones.
        if to > cold.prompt_tokens.len() {
            return Some(format!(
                "the range {from}..{to} runs past the prompt's {} tokens",
                cold.prompt_tokens.len()
            ));
        }
        for (what, r) in [("cold", cold), ("warm", warm)] {
            if r.positions.len() < to {
                return Some(format!(
                    "the {what} run has {} fingerprints, fewer than the {to} positions to compare",
                    r.positions.len()
                ));
            }
        }
        None
    })();
    let first_difference = match problem {
        Some(_) => None,
        None => (from..to).find(|&i| cold.positions[i] != warm.positions[i]),
    };
    BitCheck {
        id: id.into(),
        from,
        to,
        equal: problem.is_none() && first_difference.is_none(),
        first_difference,
        problem,
    }
}

/// Whether every decode step of the two runs is bit for bit the same: each step's top two (ids and logits, compared
/// as bits) and each decoded position's fingerprint. Reported per prompt, not gating.
pub fn decode_bits_equal(cold: &Record, warm: &Record) -> bool {
    // The fingerprints after the prompt's: all of its positions with `--all-logits`, else its last.
    fn decoded(r: &Record) -> Option<&[String]> {
        let prompt = if r.settings.all_logits {
            r.prompt_tokens.len()
        } else {
            1
        };
        r.positions.get(prompt..)
    }
    let step = |s: &Step| {
        (
            s.token,
            s.top,
            s.logit.to_bits(),
            s.second,
            s.second_logit.to_bits(),
        )
    };
    cold.steps.len() == warm.steps.len()
        && cold.steps.iter().map(step).eq(warm.steps.iter().map(step))
        && decoded(cold).is_some()
        && decoded(cold) == decoded(warm)
}

/// One prompt's line of the report.
#[derive(Clone, Debug, Serialize)]
pub struct PerPrompt {
    pub id: String,
    pub cold_flips: usize,
    pub path1_flips: usize,
    pub path2_flips: usize,
    /// Path 1's prefilled positions after the restore against the cold run's, where both recorded them.
    pub path1_prefill_bits: Option<bool>,
    /// Path 1's decode steps against the cold run's; `None` when there is no path-1 record.
    pub path1_decode_bits: Option<bool>,
    /// The same for path 2.
    pub path2_decode_bits: Option<bool>,
}

/// Test 6's verdict. Each `test3::Verdict`'s own status (1.5 × the floor of the prompts run) is reported, not used.
#[derive(Debug, Serialize)]
pub struct Verdict6 {
    pub status: Status,
    pub bound: usize,
    pub m2_cold_flips: usize,
    pub cold: test3::Verdict,
    pub path1: test3::Verdict,
    pub path2: test3::Verdict,
    pub bits: Option<BitCheck>,
    pub per_prompt: Vec<PerPrompt>,
    pub problems: Vec<String>,
}

/// Test 6's verdict: each path's flips, pooled, at most `BOUND`; the bit check present and equal; and no problem,
/// which includes any prompt run cold but missing from a path.
pub fn judge(
    cold: Vec<PromptResult>,
    path1: Vec<PromptResult>,
    path2: Vec<PromptResult>,
    bits: Option<BitCheck>,
    per_prompt: Vec<PerPrompt>,
) -> Verdict6 {
    let (cold, path1, path2) = (test3::judge(cold), test3::judge(path1), test3::judge(path2));
    let mut problems = Vec::new();
    for (name, v) in [("path 1", &path1), ("path 2", &path2)] {
        if v.prompts.is_empty() {
            problems.push(format!("{name}: no prompts were run"));
        }
        for p in &v.prompts {
            if let Some(e) = &p.problem {
                problems.push(format!("{name}: {}: {e}", p.id));
            }
        }
        for c in &cold.prompts {
            if !v.prompts.iter().any(|p| p.id == c.id) {
                problems.push(format!("{name}: {} was run cold but not on {name}", c.id));
            }
        }
        if v.flips > BOUND {
            problems.push(format!(
                "{name}: {} flips over {} prompts, more than the bound of {BOUND}",
                v.flips,
                v.prompts.len()
            ));
        }
    }
    match &bits {
        None => problems.push("no bit check was run".into()),
        Some(b) => match (&b.problem, b.first_difference) {
            (Some(p), _) => problems.push(format!("bit check, {}: {p}", b.id)),
            (None, Some(at)) => problems.push(format!(
                "bit check, {}: the logits of positions {}..{} are not the cold run's bit for bit, first at {at}",
                b.id, b.from, b.to
            )),
            (None, None) if !b.equal => problems.push(format!(
                "bit check, {}: the check does not report the positions {}..{} equal",
                b.id, b.from, b.to
            )),
            (None, None) => {}
        },
    }
    Verdict6 {
        status: if problems.is_empty() {
            Status::Pass
        } else {
            Status::Fail
        },
        bound: BOUND,
        m2_cold_flips: M2_COLD_FLIPS,
        cold,
        path1,
        path2,
        bits,
        per_prompt,
        problems,
    }
}

impl Verdict6 {
    /// The verdict's line for the console.
    pub fn line(&self) -> String {
        let bits = match &self.bits {
            None => "none run".to_string(),
            Some(b) => match (&b.problem, b.first_difference) {
                (Some(p), _) => format!("not comparable ({p})"),
                (None, Some(at)) => format!("positions {}..{} differ, first at {at}", b.from, b.to),
                (None, None) if !b.equal => {
                    format!("positions {}..{} not reported equal", b.from, b.to)
                }
                (None, None) => format!("positions {}..{} equal", b.from, b.to),
            },
        };
        format!(
            "{:?}; flips over {} prompts: path 1 {}, path 2 {}, each at most {}; cold {}, milestone 2's closing {}; \
             bit check: {bits}{}",
            self.status,
            self.path1.prompts.len().max(self.path2.prompts.len()),
            self.path1.flips,
            self.path2.flips,
            self.bound,
            self.cold.flips,
            self.m2_cold_flips,
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
    use crate::generate::{fingerprint, CacheStats, Record, Settings, Step, Stop, Timings, FORMAT};
    use crate::test3::{
        compare, Flip, Floor, FloorStep, Oracle, OracleStep, PromptResult, Status, STEPS,
    };

    const EOG: u32 = 248_046;

    #[derive(Debug, Clone, PartialEq)]
    enum Call {
        Prefill(usize, bool),
        Decode(u32),
        Verify(Vec<u32>),
        Commit(usize),
        Snapshot,
        Dirty(usize),
        Restore,
        Mtp(u32, usize),
    }

    /// Records the calls `drive` makes; every logits row has argmax 0.
    struct FakeOps {
        calls: Vec<Call>,
        room: usize,
        /// The drafts its MTP gives; empty is a session without one.
        mtp: Vec<u32>,
    }

    fn fake() -> FakeOps {
        FakeOps {
            calls: Vec::new(),
            room: 65_536,
            mtp: Vec::new(),
        }
    }

    fn row() -> Vec<f32> {
        vec![2.0, 1.0, 0.0, 0.0]
    }

    impl WarmOps for FakeOps {
        fn room(&self) -> usize {
            self.room
        }
        fn prefill(
            &mut self,
            tokens: &[u32],
            all: bool,
            each: &mut dyn FnMut(&[f32]),
        ) -> anyhow::Result<Vec<f32>> {
            self.calls.push(Call::Prefill(tokens.len(), all));
            for _ in 0..if all { tokens.len() } else { 1 } {
                each(&row());
            }
            Ok(row())
        }
        fn decode(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
            self.calls.push(Call::Decode(token));
            Ok(row())
        }
        fn verify(&mut self, tokens: &[u32]) -> anyhow::Result<()> {
            self.calls.push(Call::Verify(tokens.to_vec()));
            Ok(())
        }
        fn row(&mut self, _i: usize) -> anyhow::Result<Vec<f32>> {
            Ok(row())
        }
        fn commit(&mut self, kept: usize) -> anyhow::Result<()> {
            self.calls.push(Call::Commit(kept));
            Ok(())
        }
        fn n_vocab(&self) -> usize {
            1 << 20
        }
        fn mtp_draft(&mut self, x: u32, k: usize) -> anyhow::Result<Vec<u32>> {
            self.calls.push(Call::Mtp(x, k));
            Ok(self.mtp.iter().copied().take(k).collect())
        }
        fn has_mtp(&self) -> bool {
            !self.mtp.is_empty()
        }
        fn snapshot(&mut self) -> anyhow::Result<u64> {
            self.calls.push(Call::Snapshot);
            Ok(1_000)
        }
        fn dirty(&mut self, tokens: &[u32]) -> anyhow::Result<()> {
            self.calls.push(Call::Dirty(tokens.len()));
            Ok(())
        }
        fn restore(&mut self) -> anyhow::Result<RestoreFacts> {
            self.calls.push(Call::Restore);
            Ok(RestoreFacts {
                bytes: 7,
                kv_from: 5,
                ms: 1.5,
            })
        }
    }

    #[test]
    fn the_warm_verifier_forwards_the_mtp() {
        // Without the forwards a warm run would load the MTP and never draft with it.
        let mut ops = fake();
        assert!(!WarmVerifier(&mut ops).has_mtp());
        assert!(WarmVerifier(&mut ops).mtp_draft(3, 2).unwrap().is_empty());
        let mut ops = fake();
        ops.mtp = vec![7, 8, 9];
        let mut v = WarmVerifier(&mut ops);
        assert!(v.has_mtp());
        assert_eq!(v.mtp_draft(3, 2).unwrap(), [7, 8]);
        assert_eq!(ops.calls.last(), Some(&Call::Mtp(3, 2)));
    }

    fn prompt() -> Vec<u32> {
        (1000..1040).collect()
    }

    fn dirty() -> Vec<u32> {
        (5000..5023).collect()
    }

    fn forced256() -> Vec<u32> {
        (100..356).collect()
    }

    fn decodes(tokens: &[u32]) -> Vec<Call> {
        tokens.iter().map(|&t| Call::Decode(t)).collect()
    }

    /// Drafts the oracle's own next tokens, so every draft is accepted and every burst runs to its cap.
    struct Ahead {
        forced: Vec<u32>,
        primed: bool,
        seen: usize,
    }

    impl Drafter for Ahead {
        fn name(&self) -> &'static str {
            "ahead"
        }
        fn draft(&mut self, max: usize) -> Vec<u32> {
            self.forced
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

    fn drafting_ahead(forced: &[u32], fixed: usize) -> SpecState {
        let forced = forced.to_vec();
        SpecState::with_drafters(
            engine::spec::SpecConfig {
                max_draft: fixed.max(4),
                fixed_draft: Some(fixed),
                ..engine::spec::SpecConfig::default()
            },
            Box::new(move || {
                Box::new(Ahead {
                    forced: forced.clone(),
                    primed: false,
                    seen: 0,
                })
            }),
        )
    }

    #[test]
    fn the_split_lands_on_a_step_boundary() {
        let (p, forced) = (prompt(), forced256());
        let split = split_for(forced.len());
        assert_eq!(split, 128);
        let mut st = drafting_ahead(&forced, 4);
        let mut ops = fake();
        let d = drive(
            &mut ops,
            &p,
            &dirty(),
            Path::Decode { split },
            &forced,
            &[EOG],
            true,
            Some(&mut st),
        )
        .unwrap();
        // The positions decoded before the snapshot: a decode is one, a commit is the rows it kept.
        let at = ops.calls.iter().position(|c| *c == Call::Snapshot).unwrap();
        let before: usize = ops.calls[..at]
            .iter()
            .map(|c| match c {
                Call::Decode(_) => 1,
                Call::Commit(k) => *k,
                _ => 0,
            })
            .sum();
        assert_eq!(before, split - 1, "{:?}", &ops.calls[..at]);
        // 25 bursts of 5 and then the cap: 127 = 25 * 5 + 2, so the last verify before the split is two tokens
        // (x and one draft), not the five a free burst would run across it.
        assert_eq!(
            &ops.calls[at - 2..=at],
            [
                Call::Verify(forced[125..127].to_vec()),
                Call::Commit(2),
                Call::Snapshot
            ]
        );
        assert_eq!(
            &ops.calls[at + 1..at + 3],
            [Call::Dirty(23), Call::Restore],
            "the dirtying and the restore come right after the snapshot"
        );
        // After the restore the bursts go on, and the run is the plain run's: the same steps and fingerprints.
        assert!(matches!(&ops.calls[at + 3], Call::Verify(v) if v.len() == 5));
        let counts = d.spec.expect("speculation ran");
        assert!(counts.steps > 25 && counts.plain_steps == 0, "{counts:?}");
        let plain = drive(
            &mut fake(),
            &p,
            &dirty(),
            Path::Decode { split },
            &forced,
            &[EOG],
            true,
            None,
        )
        .unwrap();
        assert!(plain.spec.is_none());
        assert_eq!(d.steps, plain.steps);
        assert_eq!(d.stop, plain.stop);
        assert_eq!(d.positions, plain.positions);
        // Path 1 speculates too: nothing to cap, the same steps again.
        let mut st = drafting_ahead(&forced, 4);
        let one = drive(
            &mut fake(),
            &p,
            &dirty(),
            Path::Prefill { cut: 20 },
            &forced,
            &[EOG],
            true,
            Some(&mut st),
        )
        .unwrap();
        assert_eq!(one.steps, plain.steps);
        assert!(one.spec.is_some_and(|s| s.steps > 0));
    }

    #[test]
    fn a_split_that_the_run_ends_before_is_still_refused_with_speculation() {
        // Eight oracle tokens, a split at 10: the run ends first, whatever the bursts do.
        let (p, forced) = (prompt(), (100..108).collect::<Vec<u32>>());
        let mut st = drafting_ahead(&forced, 4);
        let e = drive(
            &mut fake(),
            &p,
            &dirty(),
            Path::Decode { split: 10 },
            &forced,
            &[EOG],
            false,
            Some(&mut st),
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("ended before path 2's split at step 10"), "{e}");
    }

    #[test]
    fn the_cut_is_half_the_prompt_and_12288_for_the_long_one() {
        assert_eq!(cut_for("03-chat", 20), 10);
        assert_eq!(cut_for("05-code-write-lru", 96), 48);
        assert_eq!(cut_for(LONG_ID, 27_416), 12_288);
        assert_eq!(LONG_CUT, 12_288);
    }

    #[test]
    fn the_split_is_half_the_oracles_steps() {
        for (steps, split) in [
            (256, 128),
            (152, 76),
            (58, 29),
            (149, 74),
            (174, 87),
            (125, 62),
        ] {
            assert_eq!(split_for(steps), split, "{steps} steps");
        }
    }

    #[test]
    fn the_dirtier_is_the_next_prompt_and_01_follows_20() {
        let ids: Vec<String> = ["01-code", "02-prose", "03-chat", "20-long-option-rs"]
            .map(String::from)
            .to_vec();
        assert_eq!(dirtier(&ids, "01-code"), Some("02-prose"));
        assert_eq!(dirtier(&ids, "02-prose"), Some("03-chat"));
        assert_eq!(dirtier(&ids, "03-chat"), Some("20-long-option-rs"));
        assert_eq!(dirtier(&ids, "20-long-option-rs"), Some("01-code"));
        // Unsorted input gives the same answer, and a lone prompt has no other to dirty with.
        let shuffled: Vec<String> = ["20-long-option-rs", "01-code", "03-chat", "02-prose"]
            .map(String::from)
            .to_vec();
        assert_eq!(dirtier(&shuffled, "02-prose"), Some("03-chat"));
        assert_eq!(dirtier(&shuffled, "20-long-option-rs"), Some("01-code"));
        assert_eq!(dirtier(&ids[..1], "01-code"), None);
        assert_eq!(dirtier(&[], "01-code"), None);
    }

    #[test]
    fn the_restore_cut_leaves_the_cold_runs_chunks() {
        for cap in [engine::ATTN_PAIRS, engine::ATTN_PAIRS_CM] {
            let mut warm = engine::chunk_lengths(0, LONG_CUT, 2048, cap);
            warm.extend(engine::chunk_lengths(
                LONG_CUT,
                27_416 - LONG_CUT,
                2048,
                cap,
            ));
            assert_eq!(warm, engine::chunk_lengths(0, 27_416, 2048, cap), "{cap}");
        }
    }

    #[test]
    fn path_1_prefills_the_cut_dirties_restores_and_prefills_the_rest() {
        let (p, forced) = (prompt(), forced256());
        let mut ops = fake();
        let d = drive(
            &mut ops,
            &p,
            &dirty(),
            Path::Prefill { cut: 20 },
            &forced,
            &[EOG],
            true,
            None,
        )
        .unwrap();
        let mut want = vec![
            Call::Prefill(20, true),
            Call::Snapshot,
            Call::Dirty(23),
            Call::Restore,
            Call::Prefill(20, true),
        ];
        want.extend(decodes(&forced[..255]));
        assert_eq!(ops.calls, want);
        assert_eq!(d.steps.len(), 256);
        for (k, s) in d.steps.iter().enumerate() {
            assert_eq!(s.token, forced[k], "step {k}");
        }
        assert_eq!(d.stop, Stop::MaxTokens);
        assert_eq!(d.positions.len(), 40 + 255);
        assert_eq!(d.snapshot_bytes, 1_000);
        assert_eq!(
            d.restore,
            RestoreFacts {
                bytes: 7,
                kv_from: 5,
                ms: 1.5
            }
        );
    }

    #[test]
    fn path_1_without_all_logits_records_the_prompts_last_position_only() {
        let (p, forced) = (prompt(), forced256());
        let mut ops = fake();
        let d = drive(
            &mut ops,
            &p,
            &dirty(),
            Path::Prefill { cut: 20 },
            &forced,
            &[EOG],
            false,
            None,
        )
        .unwrap();
        assert_eq!(ops.calls[0], Call::Prefill(20, false));
        assert_eq!(ops.calls[4], Call::Prefill(20, false));
        assert_eq!(d.positions.len(), 1 + 255);
    }

    #[test]
    fn path_2_decodes_split_minus_one_before_the_snapshot_and_the_split_token_after_the_restore() {
        let (p, forced) = (prompt(), forced256());
        let mut ops = fake();
        let d = drive(
            &mut ops,
            &p,
            &dirty(),
            Path::Decode { split: 128 },
            &forced,
            &[EOG],
            true,
            None,
        )
        .unwrap();
        let mut want = vec![Call::Prefill(40, true)];
        want.extend(decodes(&forced[..127]));
        want.extend([Call::Snapshot, Call::Dirty(23), Call::Restore]);
        want.extend(decodes(&forced[127..255]));
        assert_eq!(ops.calls, want);
        assert_eq!(d.steps.len(), 256);
        for (k, s) in d.steps.iter().enumerate() {
            assert_eq!(s.token, forced[k], "step {k}");
        }
        assert_eq!(d.positions.len(), 40 + 255);
    }

    #[test]
    fn path_2_with_an_early_stop_splits_at_half_and_never_decodes_the_end() {
        let p = prompt();
        let mut forced: Vec<u32> = (100..157).collect();
        forced.push(EOG);
        assert_eq!(forced.len(), 58);
        let mut ops = fake();
        let d = drive(
            &mut ops,
            &p,
            &dirty(),
            Path::Decode {
                split: split_for(58),
            },
            &forced,
            &[EOG],
            true,
            None,
        )
        .unwrap();
        let mut want = vec![Call::Prefill(40, true)];
        want.extend(decodes(&forced[..28]));
        want.extend([Call::Snapshot, Call::Dirty(23), Call::Restore]);
        want.extend(decodes(&forced[28..57]));
        assert_eq!(ops.calls, want);
        assert_eq!(d.stop, Stop::EndOfGeneration);
        assert_eq!(d.steps.len(), 58);
        assert!(!ops.calls.contains(&Call::Decode(EOG)));
        assert_eq!(d.positions.len(), 40 + 57);
    }

    #[test]
    fn a_path_that_cannot_happen_is_an_error() {
        let (p, forced) = (prompt(), forced256());
        let go = |path| {
            drive(
                &mut fake(),
                &p,
                &dirty(),
                path,
                &forced,
                &[EOG],
                false,
                None,
            )
        };
        assert!(go(Path::Prefill { cut: 0 }).is_err());
        assert!(go(Path::Prefill { cut: 40 }).is_err());
        assert!(go(Path::Decode { split: 0 }).is_err());
        // The run ends (256 steps, 255 decodes) before a split of 300 comes up.
        assert!(go(Path::Decode { split: 300 }).is_err());
    }

    fn settings() -> Settings {
        Settings {
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
        }
    }

    fn record(prompt: &[u32], d: Drive, all_logits: bool) -> Record {
        Record {
            format: FORMAT.into(),
            prompt_tokens: prompt.to_vec(),
            settings: Settings {
                all_logits,
                ..settings()
            },
            steps: d.steps,
            stop: d.stop,
            positions: d.positions,
            timings: Timings::default(),
            cache: CacheStats::default(),
            memory: Default::default(),
            vram: Default::default(),
            prefill: Default::default(),
            warm: None,
            spec: None,
        }
    }

    #[test]
    fn a_driven_record_passes_test3s_compare() {
        let (p, forced) = (prompt(), forced256());
        let oracle = Oracle {
            prompt_tokens: p.len(),
            stopped_at_eog: false,
            steps: forced
                .iter()
                .map(|&t| OracleStep {
                    token: t,
                    logit: 20.0,
                    second: t + 1,
                    second_logit: 20.0,
                })
                .collect(),
        };
        let floor = Floor {
            backend: "cpu".into(),
            prompt_tokens: p.len(),
            steps: forced
                .iter()
                .map(|&t| FloorStep {
                    token: t,
                    top: t,
                    logit: 20.0,
                    second: t + 1,
                    second_logit: 19.0,
                    token_logit: 20.0,
                })
                .collect(),
        };
        for path in [Path::Prefill { cut: 20 }, Path::Decode { split: 128 }] {
            let d = drive(
                &mut fake(),
                &p,
                &dirty(),
                path,
                &forced,
                &[EOG],
                false,
                None,
            )
            .unwrap();
            let r = compare("03-chat", &p, &oracle, &floor, &record(&p, d, false));
            assert_eq!(r.problem, None, "{path:?}");
            assert_eq!(r.status, Status::Pass);
            assert_eq!(r.steps, 256);
        }
    }

    /// `n` prompts `p0`, `p1`, …, each with `flips` flips; the first has all of them.
    fn results(n: usize, flips: usize) -> Vec<PromptResult> {
        (0..n)
            .map(|i| PromptResult {
                id: format!("p{i}"),
                status: Status::Pass,
                steps: 256,
                counted: 256,
                flips: (0..if i == 0 { flips } else { 0 })
                    .map(|step| Flip {
                        step,
                        top: 1,
                        oracle: 2,
                        oracle_gap: 1.0,
                    })
                    .collect(),
                floor_flips: Vec::new(),
                free_divergence: None,
                floor_free_divergence: None,
                problem: None,
            })
            .collect()
    }

    fn good_bits() -> Option<BitCheck> {
        Some(BitCheck {
            id: LONG_ID.into(),
            from: LONG_CUT,
            to: 27_416,
            equal: true,
            first_difference: None,
            problem: None,
        })
    }

    fn per_prompt(n: usize) -> Vec<PerPrompt> {
        (0..n)
            .map(|i| PerPrompt {
                id: format!("p{i}"),
                cold_flips: 0,
                path1_flips: 0,
                path2_flips: 0,
                path1_prefill_bits: None,
                path1_decode_bits: Some(true),
                path2_decode_bits: Some(true),
            })
            .collect()
    }

    fn verdict(cold: usize, p1: usize, p2: usize, bits: Option<BitCheck>) -> Verdict6 {
        judge(
            results(3, cold),
            results(3, p1),
            results(3, p2),
            bits,
            per_prompt(3),
        )
    }

    #[test]
    fn test6_passes_at_48_and_fails_at_49_on_either_path() {
        let v = verdict(29, 48, 48, good_bits());
        assert_eq!(v.status, Status::Pass, "{v:?}");
        assert_eq!((v.path1.flips, v.path2.flips), (48, 48));
        assert!(v.problems.is_empty(), "{:?}", v.problems);
        // Test 3's own bound (1.5 × a floor of 0 here) is reported, not used.
        assert_eq!(v.path1.status, Status::Fail);

        let v = verdict(29, 49, 48, good_bits());
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems.iter().any(|p| p.contains("49")),
            "{:?}",
            v.problems
        );
        let v = verdict(29, 48, 49, good_bits());
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems.iter().any(|p| p.contains("49")),
            "{:?}",
            v.problems
        );
    }

    fn long_record(all_logits: bool) -> Record {
        let n = 27_416;
        let p = vec![1u32; n];
        Record {
            format: FORMAT.into(),
            prompt_tokens: p,
            settings: Settings {
                all_logits,
                ..settings()
            },
            steps: vec![Step {
                token: 5,
                top: 5,
                logit: 1.0,
                second: 6,
                second_logit: 0.5,
            }],
            stop: Stop::MaxTokens,
            positions: (0..n).map(|i| format!("{i:016x}")).collect(),
            timings: Timings::default(),
            cache: CacheStats::default(),
            memory: Default::default(),
            vram: Default::default(),
            prefill: Default::default(),
            warm: None,
            spec: None,
        }
    }

    #[test]
    fn the_bit_check_gates_on_the_long_prompts_restored_positions() {
        let cold = long_record(true);
        let warm = long_record(true);
        let b = bit_check(LONG_ID, &cold, &warm, LONG_CUT, 27_416);
        assert_eq!(
            b,
            BitCheck {
                id: LONG_ID.into(),
                from: 12_288,
                to: 27_416,
                equal: true,
                first_difference: None,
                problem: None
            }
        );
        assert_eq!(verdict(29, 10, 10, Some(b)).status, Status::Pass);

        // One fingerprint differs at 20,000; one before the cut does not count.
        let mut bad = long_record(true);
        bad.positions[20_000] = fingerprint(&[1.0]);
        bad.positions[100] = fingerprint(&[2.0]);
        let b = bit_check(LONG_ID, &cold, &bad, LONG_CUT, 27_416);
        assert_eq!((b.equal, b.first_difference), (false, Some(20_000)));
        assert_eq!(b.problem, None);
        let v = verdict(29, 10, 10, Some(b));
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems.iter().any(|p| p.contains("20000")),
            "{:?}",
            v.problems
        );

        // A record without --all-logits has no fingerprints of the prompt's positions.
        for (c, w) in [(true, false), (false, true)] {
            let b = bit_check(LONG_ID, &long_record(c), &long_record(w), LONG_CUT, 27_416);
            assert!(!b.equal && b.first_difference.is_none(), "{b:?}");
            assert!(
                b.problem
                    .as_deref()
                    .is_some_and(|p| p.contains("--all-logits")),
                "{b:?}"
            );
            assert_eq!(verdict(29, 10, 10, Some(b)).status, Status::Fail);
        }
        // Records that stop short of the range, and runs of different prompts, are problems too.
        let mut short = long_record(true);
        short.positions.truncate(20_000);
        assert!(bit_check(LONG_ID, &cold, &short, LONG_CUT, 27_416)
            .problem
            .is_some());
        let mut other = long_record(true);
        other.prompt_tokens[0] = 2;
        assert!(bit_check(LONG_ID, &cold, &other, LONG_CUT, 27_416)
            .problem
            .is_some());
    }

    #[test]
    fn a_bit_check_that_does_not_say_equal_is_a_fail() {
        let mut b = good_bits().unwrap();
        b.equal = false;
        let v = verdict(29, 10, 10, Some(b));
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems.iter().any(|p| p.contains("bit check")),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn a_bit_check_range_past_the_prompt_is_a_problem() {
        // The records' fingerprints run to 27,416 (the prompt) and the range asks for one more.
        let mut cold = long_record(true);
        let mut warm = long_record(true);
        cold.positions.push("x".into());
        warm.positions.push("x".into());
        let b = bit_check(LONG_ID, &cold, &warm, LONG_CUT, 27_417);
        assert!(!b.equal && b.first_difference.is_none(), "{b:?}");
        assert!(
            b.problem
                .as_deref()
                .is_some_and(|p| p.contains("past the prompt")),
            "{b:?}"
        );
    }

    #[test]
    fn no_bit_check_is_a_fail() {
        let v = verdict(29, 10, 10, None);
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems.iter().any(|p| p.contains("bit")),
            "{:?}",
            v.problems
        );
    }

    #[test]
    fn a_missing_path_run_is_a_problem() {
        let mut p1 = results(3, 0);
        p1.remove(1);
        let v = judge(results(3, 0), p1, results(3, 0), good_bits(), per_prompt(3));
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems
                .iter()
                .any(|p| p.contains("p1") && p.contains("path 1")),
            "{:?}",
            v.problems
        );
        // A prompt whose comparison had a problem fails the verdict through its path.
        let mut p2 = results(3, 0);
        p2[2].problem = Some("our run has 3 steps".into());
        let v = judge(results(3, 0), results(3, 0), p2, good_bits(), per_prompt(3));
        assert_eq!(v.status, Status::Fail);
        assert!(
            v.problems
                .iter()
                .any(|p| p.contains("path 2") && p.contains("our run has 3 steps")),
            "{:?}",
            v.problems
        );
        // An empty path is a problem, not a pass.
        let v = judge(
            results(3, 0),
            Vec::new(),
            results(3, 0),
            good_bits(),
            per_prompt(3),
        );
        assert_eq!(v.status, Status::Fail);
    }

    #[test]
    fn test6_reports_the_cold_flips_and_m2s_29() {
        let v = verdict(29, 30, 31, good_bits());
        assert_eq!((v.cold.flips, v.m2_cold_flips, v.bound), (29, 29, 48));
        assert_eq!(v.per_prompt.len(), 3);
        let line = v.line();
        for want in ["Pass", "30", "31", "48", "29"] {
            assert!(line.contains(want), "{want} in {line}");
        }
        let json = serde_json::to_value(&v).unwrap();
        assert_eq!(json["status"], "pass");
        assert_eq!(json["m2_cold_flips"], 29);
        assert_eq!(json["bits"]["equal"], true);
    }

    #[test]
    fn decode_bits_compare_every_step_and_every_decoded_fingerprint() {
        let mk = |all: bool| {
            let mut r = long_record(all);
            r.prompt_tokens.truncate(3);
            r.positions = (0..if all { 3 } else { 1 } + 2)
                .map(|i| format!("{i:016x}"))
                .collect();
            r.steps = (0..3)
                .map(|t| Step {
                    token: t,
                    top: t,
                    logit: 1.5,
                    second: t + 1,
                    second_logit: 0.5,
                })
                .collect();
            r
        };
        for all in [true, false] {
            let (a, b) = (mk(all), mk(all));
            assert!(decode_bits_equal(&a, &b));
            let mut c = mk(all);
            c.steps[1].logit = f32::from_bits(1.5f32.to_bits() + 1);
            assert!(!decode_bits_equal(&a, &c));
            let mut d = mk(all);
            let last = d.positions.len() - 1;
            d.positions[last] = "x".into();
            assert!(!decode_bits_equal(&a, &d));
            // A prompt fingerprint differing is not a decode difference.
            let mut e = mk(all);
            e.positions[0] = "x".into();
            assert!(decode_bits_equal(&a, &e));
            let mut f = mk(all);
            f.positions.pop();
            assert!(!decode_bits_equal(&a, &f));
        }
    }
}
