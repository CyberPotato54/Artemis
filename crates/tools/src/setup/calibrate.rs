//! The calibration's order and time limit, driven through a `Bench` so a fake can stand in for the model.
use super::choose::{priors_differ, Calibration, Score, SpecCandidate, SETUP_LIMIT_S};
use engine::spec::MeasuredCosts;
use engine::{SpecConfig, SpecSource};
use std::collections::{BTreeMap, BTreeSet};

/// Which model load a group of candidates runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Load {
    WithMtp,
    WithoutMtp,
}

/// What a load reports.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadInfo {
    pub seconds: f64,
    pub cache_slots: usize,
    /// The MTP block's VRAM expert slots the load got and their bytes (0 without the MTP or without room).
    pub mtp_slots: usize,
    pub mtp_slot_bytes: u64,
}

impl LoadInfo {
    /// The line setup prints after a load: the expert cache, and the MTP slots in `generate`'s wording when it got any.
    pub fn summary(&self) -> String {
        let mtp = crate::generate::mtp_slots_note(&engine::Stats {
            mtp_slots: self.mtp_slots,
            mtp_slot_bytes: self.mtp_slot_bytes,
            ..engine::Stats::default()
        });
        format!(
            "Loaded in {:.1} s with {} expert slots per layer{mtp}.",
            self.seconds, self.cache_slots
        )
    }
}

/// One timed pass: the tokens it decoded and the seconds it took.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pass {
    pub decoded: usize,
    pub seconds: f64,
}

/// What the calibration drives. Each candidate keeps its own speculation state across its passes, as a server's
/// engine-wide state does; a `spec` of `None` is plain decode.
pub trait Bench {
    fn load(&mut self, load: Load) -> anyhow::Result<LoadInfo>;
    fn unload(&mut self);
    fn warm_up(&mut self) -> anyhow::Result<()>;
    fn pass(&mut self, c: SpecCandidate, spec: Option<&SpecConfig>) -> anyhow::Result<Pass>;
    fn measured(&self, c: SpecCandidate) -> MeasuredCosts;
}

/// Seconds since setup began.
pub trait Clock {
    fn elapsed_s(&self) -> f64;
}

/// The seconds before the limit that setup keeps for writing the result and printing it.
const LIMIT_SLACK_S: f64 = 60.0;

/// The speculation config a candidate runs with; `None` is plain decode.
fn spec_for(c: SpecCandidate, base: &SpecConfig, priors: Option<(f64, f64)>) -> Option<SpecConfig> {
    match c {
        SpecCandidate::Base => Some(base.clone()),
        SpecCandidate::LookupOnly => Some(SpecConfig {
            sources: vec![SpecSource::Lookup],
            ..base.clone()
        }),
        SpecCandidate::Off => None,
        SpecCandidate::MeasuredPriors => {
            let (slope, d1) = priors?;
            Some(SpecConfig {
                cost_slope_ms: slope,
                mtp_cost_ms: d1,
                ..base.clone()
            })
        }
    }
}

/// What the passes so far have found.
struct State {
    scores: BTreeMap<SpecCandidate, Score>,
    priors: Option<(f64, f64)>,
    measured: MeasuredCosts,
    limit_hit: bool,
    /// The seconds the last pass (the warm-up, before a load's first) took: what the next one is expected to take.
    last_s: f64,
}

/// The calibration: loads, warm-ups and passes in the planned order, within `SETUP_LIMIT_S`. Of `cands`, `Base` with
/// MTP runs on the load with MTP; the rest run on a load without it, and the two loads never overlap.
pub fn calibrate(
    b: &mut dyn Bench,
    cands: &[SpecCandidate],
    base: &SpecConfig,
    user: &BTreeSet<String>,
    clock: &dyn Clock,
) -> anyhow::Result<(Calibration, Vec<LoadInfo>)> {
    let want = |c| cands.contains(&c);
    let mut with_mtp = Vec::new();
    let mut without = Vec::new();
    if want(SpecCandidate::Base) {
        if base.has(SpecSource::Mtp) {
            with_mtp.push(SpecCandidate::Base);
        } else {
            without.push(SpecCandidate::Base);
        }
    }
    for c in [SpecCandidate::LookupOnly, SpecCandidate::Off] {
        if want(c) {
            without.push(c);
        }
    }
    let mut st = State {
        scores: BTreeMap::new(),
        priors: None,
        measured: MeasuredCosts {
            slope_ms: None,
            mtp_d1_ms: None,
        },
        limit_hit: false,
        last_s: 0.0,
    };
    let mut loads: Vec<LoadInfo> = Vec::new();
    for (group, load) in [(with_mtp, Load::WithMtp), (without, Load::WithoutMtp)] {
        if group.is_empty() || st.limit_hit {
            continue;
        }
        // A second load costs what the first did, and then a pass.
        if let Some(first) = loads.first() {
            if clock.elapsed_s() + first.seconds + st.last_s > SETUP_LIMIT_S - LIMIT_SLACK_S {
                st.limit_hit = true;
                continue;
            }
        }
        loads.push(b.load(load)?);
        let ran = run_loaded(
            b,
            &group,
            base,
            user,
            clock,
            cands == [SpecCandidate::Off],
            &mut st,
        );
        b.unload();
        ran?;
    }
    Ok((
        Calibration {
            scores: st.scores,
            priors: st.priors,
            measured: st.measured,
            limit_hit: st.limit_hit,
        },
        loads,
    ))
}

/// The warm-up and the rounds of `group` on the load that is up.
fn run_loaded(
    b: &mut dyn Bench,
    group: &[SpecCandidate],
    base: &SpecConfig,
    user: &BTreeSet<String>,
    clock: &dyn Clock,
    only_off: bool,
    st: &mut State,
) -> anyhow::Result<()> {
    let warm_from = clock.elapsed_s();
    b.warm_up()?;
    st.last_s = clock.elapsed_s() - warm_from;
    // Plain decode is one pass only when it is the single candidate (the user turned speculation off): it then only
    // gives the expected speed. Beside `Base` it is compared under the margin and gets its two rounds.
    let rounds = if only_off { 1 } else { 2 };
    for round in 0..rounds {
        let mut order = group.to_vec();
        if round > 0 && st.priors.is_some() {
            if let Some(i) = order.iter().position(|&c| c == SpecCandidate::Base) {
                order.insert(i + 1, SpecCandidate::MeasuredPriors);
            }
        }
        for c in order {
            if clock.elapsed_s() + st.last_s > SETUP_LIMIT_S - LIMIT_SLACK_S {
                st.limit_hit = true;
                return Ok(());
            }
            let p = b.pass(c, spec_for(c, base, st.priors).as_ref())?;
            st.last_s = p.seconds;
            let s = st.scores.entry(c).or_insert(Score {
                decoded: 0,
                seconds: 0.0,
                passes: 0,
            });
            s.decoded += p.decoded;
            s.seconds += p.seconds;
            s.passes += 1;
            if c == SpecCandidate::Base {
                st.measured = b.measured(c);
                if round == 0 {
                    st.priors = priors_differ(base, &st.measured, user);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use SpecCandidate::*;

    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn the_load_summary_names_the_mtp_slots_only_when_the_load_got_some() {
        let with = LoadInfo {
            seconds: 20.04,
            cache_slots: 112,
            mtp_slots: 64,
            mtp_slot_bytes: 121_634_816,
        };
        assert_eq!(
            with.summary(),
            "Loaded in 20.0 s with 112 expert slots per layer; 64 MTP expert slots (116 MiB)."
        );
        // The load without the MTP, and a load whose slots were refused, say nothing about them.
        let without = LoadInfo {
            mtp_slots: 0,
            mtp_slot_bytes: 0,
            ..with
        };
        assert_eq!(
            without.summary(),
            "Loaded in 20.0 s with 112 expert slots per layer."
        );
    }

    struct FakeClock(Rc<Cell<f64>>);

    impl Clock for FakeClock {
        fn elapsed_s(&self) -> f64 {
            self.0.get()
        }
    }

    /// A bench that logs its calls and advances the clock by what each takes.
    struct Fake {
        clock: Rc<Cell<f64>>,
        log: Vec<String>,
        loaded: bool,
        load_s: f64,
        warm_s: f64,
        pass_s: f64,
        fail_load: Option<usize>,
        /// The warm-up (counted from 1) that fails.
        fail_warm: Option<usize>,
        /// The pass (counted from 1) that fails.
        fail_pass: Option<usize>,
        warms: usize,
        passes: usize,
        loads: usize,
        measured: MeasuredCosts,
        specs: Vec<(SpecCandidate, Option<SpecConfig>)>,
    }

    impl Fake {
        fn new(load_s: f64, warm_s: f64, pass_s: f64) -> (Fake, FakeClock) {
            let clock = Rc::new(Cell::new(0.0));
            (
                Fake {
                    clock: clock.clone(),
                    log: Vec::new(),
                    loaded: false,
                    load_s,
                    warm_s,
                    pass_s,
                    fail_load: None,
                    fail_warm: None,
                    fail_pass: None,
                    warms: 0,
                    passes: 0,
                    loads: 0,
                    measured: MeasuredCosts {
                        slope_ms: Some(4.4),
                        mtp_d1_ms: Some(1.6),
                    },
                    specs: Vec::new(),
                },
                FakeClock(clock),
            )
        }

        fn tick(&self, s: f64) {
            self.clock.set(self.clock.get() + s);
        }
    }

    impl Bench for Fake {
        fn load(&mut self, load: Load) -> anyhow::Result<LoadInfo> {
            assert!(!self.loaded, "two loads held at once");
            self.log.push(format!("load {load:?}"));
            self.loads += 1;
            if self.fail_load == Some(self.loads) {
                anyhow::bail!("the model did not load");
            }
            self.tick(self.load_s);
            self.loaded = true;
            Ok(LoadInfo {
                seconds: self.load_s,
                cache_slots: 150,
                mtp_slots: 0,
                mtp_slot_bytes: 0,
            })
        }

        fn unload(&mut self) {
            self.log.push("unload".into());
            self.loaded = false;
        }

        fn warm_up(&mut self) -> anyhow::Result<()> {
            assert!(self.loaded);
            self.log.push("warm".into());
            self.warms += 1;
            if self.fail_warm == Some(self.warms) {
                anyhow::bail!("the warm-up failed");
            }
            self.tick(self.warm_s);
            Ok(())
        }

        fn pass(&mut self, c: SpecCandidate, spec: Option<&SpecConfig>) -> anyhow::Result<Pass> {
            assert!(self.loaded);
            self.log.push(format!("pass {c:?}"));
            self.passes += 1;
            if self.fail_pass == Some(self.passes) {
                anyhow::bail!("the pass failed");
            }
            self.specs.push((c, spec.cloned()));
            self.tick(self.pass_s);
            Ok(Pass {
                decoded: 256 * 3,
                seconds: self.pass_s,
            })
        }

        fn measured(&self, _: SpecCandidate) -> MeasuredCosts {
            self.measured
        }
    }

    fn no_keys() -> BTreeSet<String> {
        BTreeSet::new()
    }

    fn all() -> [SpecCandidate; 3] {
        [Base, LookupOnly, Off]
    }

    #[test]
    fn runs_the_planned_order() {
        let (mut b, clock) = Fake::new(30.0, 5.0, 20.0);
        let base = SpecConfig::default();
        let (cal, loads) = calibrate(&mut b, &all(), &base, &no_keys(), &clock).unwrap();
        assert_eq!(
            b.log,
            [
                "load WithMtp",
                "warm",
                "pass Base",
                "pass Base",
                "unload",
                "load WithoutMtp",
                "warm",
                "pass LookupOnly",
                "pass Off",
                "pass LookupOnly",
                "pass Off",
                "unload",
            ]
        );
        assert_eq!(loads.len(), 2);
        assert!(!cal.limit_hit);
        assert_eq!(cal.priors, None);
        assert_eq!(cal.scores.len(), 3);
        for c in all() {
            let s = cal.scores[&c];
            assert_eq!((s.passes, s.decoded), (2, 1536), "{c:?}");
            assert_eq!(s.seconds, 40.0);
        }
        // The specs each candidate ran with: the base, lookup alone, and plain decode.
        let spec_of = |c| b.specs.iter().find(|(k, _)| *k == c).unwrap().1.clone();
        assert_eq!(spec_of(Base), Some(base.clone()));
        assert_eq!(
            spec_of(LookupOnly).unwrap().sources,
            vec![SpecSource::Lookup]
        );
        assert_eq!(spec_of(Off), None);
    }

    #[test]
    fn shifted_priors_add_measured_priors_to_the_second_round() {
        let (mut b, clock) = Fake::new(30.0, 5.0, 20.0);
        b.measured = MeasuredCosts {
            slope_ms: Some(2.0),
            mtp_d1_ms: None,
        };
        let base = SpecConfig::default();
        let (cal, _) = calibrate(&mut b, &all(), &base, &no_keys(), &clock).unwrap();
        assert_eq!(
            &b.log[..6],
            [
                "load WithMtp",
                "warm",
                "pass Base",
                "pass Base",
                "pass MeasuredPriors",
                "unload"
            ]
        );
        assert_eq!(cal.priors, Some((2.0, base.mtp_cost_ms)));
        assert_eq!(cal.scores[&MeasuredPriors].passes, 1);
        let spec = b
            .specs
            .iter()
            .find(|(k, _)| *k == MeasuredPriors)
            .unwrap()
            .1
            .clone()
            .unwrap();
        assert_eq!(spec.cost_slope_ms, 2.0);
        assert_eq!(spec.mtp_cost_ms, base.mtp_cost_ms);
        // The user's own cost keys keep the priors out.
        let (mut b, clock) = Fake::new(30.0, 5.0, 20.0);
        b.measured = MeasuredCosts {
            slope_ms: Some(2.0),
            mtp_d1_ms: None,
        };
        let user: BTreeSet<String> = ["speculation.cost_slope_ms".to_string()].into();
        let (cal, _) = calibrate(&mut b, &all(), &base, &user, &clock).unwrap();
        assert_eq!(cal.priors, None);
        assert!(!b.log.iter().any(|l| l == "pass MeasuredPriors"));
    }

    #[test]
    fn plain_decode_alone_is_one_pass() {
        let (mut b, clock) = Fake::new(30.0, 5.0, 20.0);
        let (cal, loads) =
            calibrate(&mut b, &[Off], &SpecConfig::default(), &no_keys(), &clock).unwrap();
        assert_eq!(b.log, ["load WithoutMtp", "warm", "pass Off", "unload"]);
        assert_eq!(loads.len(), 1);
        assert_eq!(cal.scores[&Off].passes, 1);
    }

    #[test]
    fn stops_at_the_limit_and_says_so() {
        // 30 s load + 20 s warm-up = 50 s. A pass of 250 s: the first starts at 50 (50 + 20 <= 540) and ends at 300;
        // the second would end at 550, past the 540 s line.
        let (mut b, clock) = Fake::new(30.0, 20.0, 250.0);
        let (cal, loads) =
            calibrate(&mut b, &[Base], &SpecConfig::default(), &no_keys(), &clock).unwrap();
        assert!(cal.limit_hit);
        assert_eq!(cal.scores[&Base].passes, 1);
        assert_eq!(loads.len(), 1);
        assert_eq!(b.log, ["load WithMtp", "warm", "pass Base", "unload"]);
        assert!(!b.loaded);
    }

    #[test]
    fn skips_the_second_load_when_it_would_pass_the_limit() {
        // Base's two passes end at 70 + 150 + 150 = 370 s; another load (60 s) and a pass (150 s) would end at 580.
        let (mut b, clock) = Fake::new(60.0, 10.0, 150.0);
        let (cal, loads) =
            calibrate(&mut b, &all(), &SpecConfig::default(), &no_keys(), &clock).unwrap();
        assert!(cal.limit_hit);
        assert_eq!(loads.len(), 1);
        assert_eq!(b.loads, 1);
        assert_eq!(cal.scores[&Base].passes, 2);
        assert!(!cal.scores.contains_key(&LookupOnly));
        assert_eq!(b.log.last().map(String::as_str), Some("unload"));
    }

    #[test]
    fn a_failed_load_is_an_error() {
        let (mut b, clock) = Fake::new(30.0, 5.0, 20.0);
        b.fail_load = Some(1);
        let e = calibrate(&mut b, &all(), &SpecConfig::default(), &no_keys(), &clock).unwrap_err();
        assert!(e.to_string().contains("did not load"), "{e}");
        // The second load failing is an error too, and the first was unloaded before it.
        let (mut b, clock) = Fake::new(30.0, 5.0, 20.0);
        b.fail_load = Some(2);
        let e = calibrate(&mut b, &all(), &SpecConfig::default(), &no_keys(), &clock).unwrap_err();
        assert!(e.to_string().contains("did not load"), "{e}");
        assert_eq!(
            b.log,
            [
                "load WithMtp",
                "warm",
                "pass Base",
                "pass Base",
                "unload",
                "load WithoutMtp",
            ]
        );
        assert!(!b.loaded);
    }

    #[test]
    fn a_failed_warm_up_or_pass_unloads_and_returns_the_error() {
        let (mut b, clock) = Fake::new(30.0, 5.0, 20.0);
        b.fail_warm = Some(1);
        let e = calibrate(&mut b, &all(), &SpecConfig::default(), &no_keys(), &clock).unwrap_err();
        assert!(e.to_string().contains("warm-up failed"), "{e}");
        assert_eq!(b.log, ["load WithMtp", "warm", "unload"]);
        assert!(!b.loaded);
        // A pass failing in the second group: the first group's unload came before, this one's follows the failure.
        let (mut b, clock) = Fake::new(30.0, 5.0, 20.0);
        b.fail_pass = Some(4);
        let e = calibrate(&mut b, &all(), &SpecConfig::default(), &no_keys(), &clock).unwrap_err();
        assert!(e.to_string().contains("pass failed"), "{e}");
        assert_eq!(b.log.last().map(String::as_str), Some("unload"));
        assert_eq!(b.log.iter().filter(|l| *l == "unload").count(), 2);
        assert!(!b.loaded);
    }

    #[test]
    fn off_beside_base_gets_its_two_rounds_and_alone_gets_one() {
        // The user named the sources: Base (with MTP) and Off are compared, and Off is on the second load alone.
        let (mut b, clock) = Fake::new(30.0, 5.0, 20.0);
        let (cal, _) = calibrate(
            &mut b,
            &[Base, Off],
            &SpecConfig::default(),
            &no_keys(),
            &clock,
        )
        .unwrap();
        assert_eq!(
            b.log,
            [
                "load WithMtp",
                "warm",
                "pass Base",
                "pass Base",
                "unload",
                "load WithoutMtp",
                "warm",
                "pass Off",
                "pass Off",
                "unload",
            ]
        );
        assert_eq!(cal.scores[&Off].passes, 2);
    }
}
