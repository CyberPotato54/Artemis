//! What setup measures and what it keeps: the setup constants, the candidates a user's config leaves open, and the
//! rules that turn the measured speeds into the settings the tuned file records. Pure logic: no model, no GPU.
use engine::spec::MeasuredCosts;
use engine::{SpecConfig, SpecSource};
use std::collections::{BTreeMap, BTreeSet};

/// The most seconds the whole setup, loads included, may take; the calibration stops a minute short of it.
pub const SETUP_LIMIT_S: f64 = 600.0;
/// Tokens each calibration prompt decodes.
pub const CAL_TOKENS: usize = 256;
/// The sampling seed of every calibration pass.
pub const CAL_SEED: u64 = 1234;
/// MTP stays on when its speed is at least this multiple of prompt lookup alone.
pub const MTP_PAYS: f64 = 1.02;
/// A candidate replaces the current pick only when it is this multiple faster.
pub const SWITCH_MARGIN: f64 = 1.03;
/// A measured cost replaces its prior when they differ by more than this fraction of the prior.
pub const PRIOR_SHIFT: f64 = 0.25;
/// The target card's VRAM: 15 GiB or more counts as a 16 GB card.
pub const TARGET_VRAM_BYTES: u64 = 15 << 30;
/// Setup warns when another program holds more than this much of the GPU's memory.
pub const VRAM_HELD_WARN: u64 = 2 << 30;

/// The user's config keys that setup also chooses. A key in the user's `artemis.toml` or flags wins and is not
/// measured.
const KEY_ENABLED: &str = "speculation.enabled";
const KEY_SOURCES: &str = "speculation.sources";
pub const KEY_SLOPE: &str = "speculation.cost_slope_ms";
pub const KEY_MTP_COST: &str = "speculation.mtp_cost_ms";

/// One configuration setup measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SpecCandidate {
    /// The config loaded without the tuned file: lookup and MTP unless the user set otherwise.
    Base,
    /// Prompt lookup alone.
    LookupOnly,
    /// Plain decode.
    Off,
    /// `Base` with the measured costs in place of the built-in priors.
    MeasuredPriors,
}

/// The candidates the user's config leaves open, in the order they are measured. `user` holds the dotted keys the user
/// set (`speculation.enabled`); `base_enabled` is the loaded config's `speculation.enabled`.
pub fn candidates(
    user: &BTreeSet<String>,
    base: &SpecConfig,
    base_enabled: bool,
) -> Vec<SpecCandidate> {
    if !base_enabled {
        return vec![SpecCandidate::Off];
    }
    let mut out = vec![SpecCandidate::Base];
    if !user.contains(KEY_SOURCES) && base.has(SpecSource::Mtp) {
        out.push(SpecCandidate::LookupOnly);
    }
    if !user.contains(KEY_ENABLED) {
        out.push(SpecCandidate::Off);
    }
    out
}

/// The measured slope and MTP draft cost (each falls back to its prior) when either differs from its prior by more
/// than `PRIOR_SHIFT`, and the user set neither `speculation.cost_slope_ms` nor `speculation.mtp_cost_ms`.
pub fn priors_differ(
    base: &SpecConfig,
    m: &MeasuredCosts,
    user: &BTreeSet<String>,
) -> Option<(f64, f64)> {
    if user.contains(KEY_SLOPE) || user.contains(KEY_MTP_COST) {
        return None;
    }
    let slope = m.slope_ms.unwrap_or(base.cost_slope_ms);
    let d1 = m.mtp_d1_ms.unwrap_or(base.mtp_cost_ms);
    let shifted = |measured: f64, prior: f64| {
        if prior > 0.0 {
            (measured - prior).abs() / prior > PRIOR_SHIFT
        } else {
            measured > 0.0
        }
    };
    (shifted(slope, base.cost_slope_ms) || shifted(d1, base.mtp_cost_ms)).then_some((slope, d1))
}

/// What one candidate decoded over all its passes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    pub decoded: usize,
    pub seconds: f64,
    pub passes: u32,
}

impl Score {
    /// Pooled over the passes: tokens decoded over seconds taken.
    pub fn tok_s(&self) -> f64 {
        if self.seconds > 0.0 {
            self.decoded as f64 / self.seconds
        } else {
            0.0
        }
    }
}

/// Everything the calibration measured.
#[derive(Clone, Debug)]
pub struct Calibration {
    pub scores: BTreeMap<SpecCandidate, Score>,
    /// The measured (slope, MTP draft cost) the `MeasuredPriors` candidate ran with, if it was planned.
    pub priors: Option<(f64, f64)>,
    /// What `Base`'s controller measured by the end.
    pub measured: MeasuredCosts,
    /// The time limit stopped the calibration early.
    pub limit_hit: bool,
}

/// What setup keeps.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub enabled: bool,
    pub sources: Vec<String>,
    pub priors: Option<(f64, f64)>,
    pub expected_tok_s: f64,
    /// One sentence per chosen value with the figures behind it, keyed by the config names the tuned file's renderer
    /// looks up: `speculation.enabled`, `speculation.sources`, and the priors sentence under both
    /// `speculation.cost_slope_ms` and `speculation.mtp_cost_ms`.
    pub why: BTreeMap<String, String>,
}

/// The `why` keys `choose` writes.
pub const WHY_ENABLED: &str = KEY_ENABLED;
pub const WHY_SOURCES: &str = KEY_SOURCES;

/// Puts the priors sentence under both cost keys.
fn why_priors(why: &mut BTreeMap<String, String>, text: String) {
    why.insert(KEY_SLOPE.to_string(), text.clone());
    why.insert(KEY_MTP_COST.to_string(), text);
}

/// The settings to keep from `cal`.
pub fn choose(cal: &Calibration, base: &SpecConfig, base_enabled: bool) -> Choice {
    let speed = |c: SpecCandidate| {
        cal.scores
            .get(&c)
            .filter(|s| s.passes > 0)
            .map(|s| s.tok_s())
    };
    let base_names: Vec<String> = base.sources.iter().map(|s| s.name().to_string()).collect();
    let mut why = BTreeMap::new();
    let (base_speed, look, off) = (
        speed(SpecCandidate::Base),
        speed(SpecCandidate::LookupOnly),
        speed(SpecCandidate::Off),
    );

    // The user turned speculation off: one plain pass, and its figure is the expected speed.
    if !base_enabled {
        let expected = off.unwrap_or(0.0);
        why.insert(
            WHY_ENABLED.to_string(),
            match off {
                Some(v) => format!(
                    "Speculation is off in your settings; plain decode runs at {v:.1} tok/s here."
                ),
                None => {
                    "Speculation is off in your settings; plain decode was not measured.".into()
                }
            },
        );
        return Choice {
            enabled: false,
            sources: base_names,
            priors: None,
            expected_tok_s: expected,
            why,
        };
    }

    // The speculating pick: with MTP (the base sources) or prompt lookup alone.
    let (mut sources, mut pick) = match (base_speed, look) {
        (Some(b), Some(l)) => {
            let ratio = b / l;
            if b >= MTP_PAYS * l {
                why.insert(
                    WHY_SOURCES.to_string(),
                    format!(
                        "MTP pays here: {b:.1} tok/s with it against {l:.1} with prompt lookup alone \
                         ({ratio:.2}x; it stays at {MTP_PAYS:.2}x or more)."
                    ),
                );
                (base_names.clone(), Some(b))
            } else {
                why.insert(
                    WHY_SOURCES.to_string(),
                    format!(
                        "MTP does not pay here: {b:.1} tok/s with it against {l:.1} with prompt lookup alone \
                         ({ratio:.2}x; it stays at {MTP_PAYS:.2}x or more), so it is off."
                    ),
                );
                (vec![SpecSource::Lookup.name().to_string()], Some(l))
            }
        }
        (Some(b), None) => {
            // Prompt lookup alone is not a candidate when the user named the sources or the model has no MTP, and is
            // unmeasured when the time limit stopped the calibration first.
            let why_not = if !base.has(SpecSource::Mtp) {
                "the model has no MTP, so there is no lookup-only run to compare"
            } else if cal.limit_hit {
                "the time limit stopped the measuring before prompt lookup alone ran"
            } else {
                "your settings name the sources, so prompt lookup alone was not compared"
            };
            why.insert(
                WHY_SOURCES.to_string(),
                format!(
                    "Kept the base sources ({}) at {b:.1} tok/s: {why_not}.",
                    base_names.join(", ")
                ),
            );
            (base_names.clone(), Some(b))
        }
        (None, Some(l)) => {
            why.insert(
                WHY_SOURCES.to_string(),
                format!(
                    "Prompt lookup alone ran at {l:.1} tok/s; the run with MTP was not measured."
                ),
            );
            (vec![SpecSource::Lookup.name().to_string()], Some(l))
        }
        (None, None) => {
            why.insert(
                WHY_SOURCES.to_string(),
                "Kept the base sources: nothing was measured.".to_string(),
            );
            (base_names.clone(), None)
        }
    };

    // Measured costs in place of the priors, only on the base sources' line.
    let mut priors = None;
    if let (Some(b), Some(mp), Some(p)) =
        (base_speed, speed(SpecCandidate::MeasuredPriors), cal.priors)
    {
        if sources == base_names && pick == Some(b) {
            if mp >= SWITCH_MARGIN * b {
                why_priors(
                    &mut why,
                    format!(
                        "The measured costs ({:.2} ms per extra token, {:.2} ms per MTP draft) gave {mp:.1} tok/s \
                         against {b:.1} with the built-in ones ({:.2}x; they need {SWITCH_MARGIN:.2}x or more).",
                        p.0,
                        p.1,
                        mp / b
                    ),
                );
                priors = Some(p);
                pick = Some(mp);
            } else {
                why_priors(
                    &mut why,
                    format!(
                        "The measured costs gave {mp:.1} tok/s against {b:.1} with the built-in ones ({:.2}x; \
                         they need {SWITCH_MARGIN:.2}x or more), so the built-in ones stay.",
                        mp / b
                    ),
                );
            }
        }
    }

    // Plain decode replaces the pick only by the margin.
    let mut enabled = true;
    let mut expected = pick.unwrap_or(0.0);
    match (pick, off) {
        (Some(p), Some(o)) => {
            if o >= SWITCH_MARGIN * p {
                enabled = false;
                expected = o;
                sources = base_names.clone();
                priors = None;
                // The sources and priors lines described a speculating pick that is not kept.
                why.remove(KEY_SLOPE);
                why.remove(KEY_MTP_COST);
                why.insert(
                    WHY_SOURCES.to_string(),
                    format!(
                        "Speculation is off, so the sources are left as they were ({}).",
                        base_names.join(", ")
                    ),
                );
                why.insert(
                    WHY_ENABLED.to_string(),
                    format!(
                        "Speculation is off: plain decode ran at {o:.1} tok/s against {p:.1} with it ({:.2}x; \
                         plain decode needs {SWITCH_MARGIN:.2}x or more).",
                        o / p
                    ),
                );
            } else {
                why.insert(
                    WHY_ENABLED.to_string(),
                    format!(
                        "Speculation stays on: {p:.1} tok/s with it against {o:.1} for plain decode ({:.2}x).",
                        p / o
                    ),
                );
            }
        }
        _ => {
            why.insert(
                WHY_ENABLED.to_string(),
                "Speculation stays on: no speculating run was measured to compare plain decode with.".into(),
            );
        }
    }

    Choice {
        enabled,
        sources,
        priors,
        expected_tok_s: expected,
        why,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use SpecCandidate::*;

    fn keys(k: &[&str]) -> BTreeSet<String> {
        k.iter().map(|s| s.to_string()).collect()
    }

    /// Scores of two passes each, with the speeds given.
    fn cal(speeds: &[(SpecCandidate, f64)]) -> Calibration {
        Calibration {
            scores: speeds
                .iter()
                .map(|&(c, v)| {
                    (
                        c,
                        Score {
                            decoded: (v * 10.0).round() as usize,
                            seconds: 10.0,
                            passes: 2,
                        },
                    )
                })
                .collect(),
            priors: Some((2.0, 1.0)),
            measured: MeasuredCosts {
                slope_ms: Some(2.0),
                mtp_d1_ms: Some(1.0),
            },
            limit_hit: false,
        }
    }

    fn names(c: &Choice) -> Vec<&str> {
        c.sources.iter().map(String::as_str).collect()
    }

    #[test]
    fn mtp_stays_when_it_pays() {
        let c = choose(
            &cal(&[(Base, 128.0), (LookupOnly, 103.0), (Off, 101.0)]),
            &SpecConfig::default(),
            true,
        );
        assert!(c.enabled);
        assert_eq!(names(&c), ["lookup", "mtp"]);
        assert_eq!(c.priors, None);
        assert!((c.expected_tok_s - 128.0).abs() < 1e-9);
        assert_eq!(
            c.why["speculation.sources"],
            "MTP pays here: 128.0 tok/s with it against 103.0 with prompt lookup alone \
             (1.24x; it stays at 1.02x or more)."
        );
    }

    #[test]
    fn mtp_goes_under_1_02() {
        let c = choose(
            &cal(&[(Base, 101.0), (LookupOnly, 100.0), (Off, 90.0)]),
            &SpecConfig::default(),
            true,
        );
        assert!(c.enabled);
        assert_eq!(names(&c), ["lookup"]);
        assert!((c.expected_tok_s - 100.0).abs() < 1e-9);
        assert!(c.why["speculation.sources"].starts_with("MTP does not pay here: 101.0 tok/s"));
        assert!(c.why["speculation.sources"].contains("1.01x"));
    }

    #[test]
    fn off_needs_3_percent() {
        let base = SpecConfig::default();
        let kept = choose(&cal(&[(Base, 100.0), (Off, 102.0)]), &base, true);
        assert!(kept.enabled);
        assert!((kept.expected_tok_s - 100.0).abs() < 1e-9);
        let off = choose(&cal(&[(Base, 100.0), (Off, 104.0)]), &base, true);
        assert!(!off.enabled);
        assert!((off.expected_tok_s - 104.0).abs() < 1e-9);
        assert!(
            off.why["speculation.enabled"].contains("104.0"),
            "{:?}",
            off.why
        );
        // The pick is the lookup-only line when MTP does not pay: Off is held to that figure.
        let held = choose(
            &cal(&[(Base, 100.0), (LookupOnly, 110.0), (Off, 112.0)]),
            &base,
            true,
        );
        assert!(held.enabled);
        assert_eq!(names(&held), ["lookup"]);
    }

    #[test]
    fn priors_need_3_percent() {
        let base = SpecConfig::default();
        let left = choose(&cal(&[(Base, 100.0), (MeasuredPriors, 102.0)]), &base, true);
        assert_eq!(left.priors, None);
        assert!((left.expected_tok_s - 100.0).abs() < 1e-9);
        let taken = choose(&cal(&[(Base, 100.0), (MeasuredPriors, 104.0)]), &base, true);
        assert_eq!(taken.priors, Some((2.0, 1.0)));
        assert!((taken.expected_tok_s - 104.0).abs() < 1e-9);
        assert_eq!(names(&taken), ["lookup", "mtp"]);
        assert_eq!(
            taken.why["speculation.cost_slope_ms"],
            taken.why["speculation.mtp_cost_ms"]
        );
        assert!(taken.why["speculation.cost_slope_ms"].contains("104.0"));
        // The measured costs belong to MTP's line: with lookup alone as the pick they are not kept.
        let lookup = choose(
            &cal(&[(Base, 100.0), (LookupOnly, 110.0), (MeasuredPriors, 120.0)]),
            &base,
            true,
        );
        assert_eq!(lookup.priors, None);
    }

    #[test]
    fn unmeasured_never_wins_and_nothing_measured_keeps_the_base() {
        let base = SpecConfig::default();
        // Nothing measured: the base stands.
        let none = choose(&cal(&[]), &base, true);
        assert!(none.enabled);
        assert_eq!(names(&none), ["lookup", "mtp"]);
        assert_eq!(none.priors, None);
        assert_eq!(none.expected_tok_s, 0.0);
        // A candidate with no pass never wins, whatever its figure says.
        let mut c = cal(&[(Base, 100.0), (Off, 500.0), (LookupOnly, 300.0)]);
        for k in [Off, LookupOnly] {
            c.scores.get_mut(&k).unwrap().passes = 0;
        }
        let kept = choose(&c, &base, true);
        assert!(kept.enabled);
        assert_eq!(names(&kept), ["lookup", "mtp"]);
        assert!((kept.expected_tok_s - 100.0).abs() < 1e-9);
        // Only lookup measured (the MTP load never ran): lookup alone.
        let look = choose(&cal(&[(LookupOnly, 90.0)]), &base, true);
        assert_eq!(names(&look), ["lookup"]);
        // Only Off measured while speculation is on in the base: nothing to compare with, the base stands.
        let off = choose(&cal(&[(Off, 90.0)]), &base, true);
        assert!(off.enabled);
        assert_eq!(names(&off), ["lookup", "mtp"]);
    }

    #[test]
    fn user_set_keys_are_not_candidates() {
        let base = SpecConfig::default();
        assert_eq!(candidates(&keys(&[]), &base, true), [Base, LookupOnly, Off]);
        // enabled = true: Off is not measured.
        assert_eq!(
            candidates(&keys(&[KEY_ENABLED]), &base, true),
            [Base, LookupOnly]
        );
        // enabled = false: only Off.
        assert_eq!(candidates(&keys(&[KEY_ENABLED]), &base, false), [Off]);
        // sources set: lookup alone would change the user's list.
        assert_eq!(candidates(&keys(&[KEY_SOURCES]), &base, true), [Base, Off]);
        // A base without MTP has no lookup-only line to compare.
        let no_mtp = SpecConfig {
            sources: vec![SpecSource::Lookup],
            ..SpecConfig::default()
        };
        assert_eq!(candidates(&keys(&[]), &no_mtp, true), [Base, Off]);
        // Unrelated keys change nothing.
        assert_eq!(
            candidates(&keys(&["ctx", "speculation.max_draft"]), &base, true),
            [Base, LookupOnly, Off]
        );
        // MeasuredPriors is never in the first list.
        assert!(!candidates(&keys(&[]), &base, true).contains(&MeasuredPriors));
    }

    #[test]
    fn speculation_off_by_the_user_gives_one_plain_pass_and_its_figure() {
        let base = SpecConfig::default();
        assert_eq!(candidates(&keys(&[KEY_ENABLED]), &base, false), [Off]);
        let c = choose(&cal(&[(Off, 99.0)]), &base, false);
        assert!(!c.enabled);
        assert!((c.expected_tok_s - 99.0).abs() < 1e-9);
        assert_eq!(c.priors, None);
    }

    #[test]
    fn priors_differ_rules() {
        let base = SpecConfig::default();
        assert_eq!(base.cost_slope_ms, 4.4);
        let m = |slope, d1| MeasuredCosts {
            slope_ms: slope,
            mtp_d1_ms: d1,
        };
        // The same as the prior, and nothing measured: none.
        assert_eq!(priors_differ(&base, &m(Some(4.4), None), &keys(&[])), None);
        assert_eq!(priors_differ(&base, &m(None, None), &keys(&[])), None);
        // Within a quarter of the prior: none.
        assert_eq!(
            priors_differ(&base, &m(Some(4.4 * 1.2), Some(1.6 * 0.8)), &keys(&[])),
            None
        );
        // The slope differs: the d1 falls back to its prior.
        assert_eq!(
            priors_differ(&base, &m(Some(2.0), None), &keys(&[])),
            Some((2.0, base.mtp_cost_ms))
        );
        // The d1 differs: the slope falls back to its prior.
        assert_eq!(
            priors_differ(&base, &m(None, Some(0.8)), &keys(&[])),
            Some((base.cost_slope_ms, 0.8))
        );
        // The user set either key: none.
        assert_eq!(
            priors_differ(&base, &m(Some(2.0), Some(0.8)), &keys(&[KEY_SLOPE])),
            None
        );
        assert_eq!(
            priors_differ(&base, &m(Some(2.0), Some(0.8)), &keys(&[KEY_MTP_COST])),
            None
        );
    }

    #[test]
    fn every_why_key_is_a_config_name_the_tuned_file_renders() {
        const KEYS: [&str; 4] = [
            "speculation.enabled",
            "speculation.sources",
            "speculation.cost_slope_ms",
            "speculation.mtp_cost_ms",
        ];
        let base = SpecConfig::default();
        let no_mtp = SpecConfig {
            sources: vec![SpecSource::Lookup],
            ..SpecConfig::default()
        };
        let mut seen = BTreeSet::new();
        let cases: Vec<(Calibration, &SpecConfig, bool)> = vec![
            (
                cal(&[(Base, 128.0), (LookupOnly, 103.0), (Off, 101.0)]),
                &base,
                true,
            ),
            (cal(&[(Base, 101.0), (LookupOnly, 100.0)]), &base, true),
            (cal(&[(Base, 100.0), (MeasuredPriors, 104.0)]), &base, true),
            (cal(&[(Base, 100.0), (MeasuredPriors, 102.0)]), &base, true),
            (cal(&[(Base, 100.0), (Off, 104.0)]), &base, true),
            (
                cal(&[(Base, 100.0), (MeasuredPriors, 120.0), (Off, 130.0)]),
                &base,
                true,
            ),
            (cal(&[(Base, 100.0)]), &no_mtp, true),
            (cal(&[(LookupOnly, 90.0)]), &base, true),
            (cal(&[(Off, 90.0)]), &base, true),
            (cal(&[]), &base, true),
            (cal(&[(Off, 99.0)]), &base, false),
            (cal(&[]), &base, false),
        ];
        for (c, b, enabled) in &cases {
            let choice = choose(c, b, *enabled);
            for k in choice.why.keys() {
                assert!(KEYS.contains(&k.as_str()), "{k}");
                seen.insert(k.clone());
            }
            assert!(choice.why.contains_key("speculation.enabled"));
        }
        assert_eq!(seen.len(), 4, "{seen:?}");
    }

    #[test]
    fn the_why_lines_say_why_lookup_alone_was_not_compared() {
        let base = SpecConfig::default();
        let c = cal(&[(Base, 100.0)]);
        let named = choose(&c, &base, true);
        assert!(
            named.why["speculation.sources"].contains("your settings name the sources"),
            "{:?}",
            named.why
        );
        let no_mtp = SpecConfig {
            sources: vec![SpecSource::Lookup],
            ..SpecConfig::default()
        };
        let none = choose(&c, &no_mtp, true);
        assert!(none.why["speculation.sources"].contains("no MTP"));
        let mut cut = cal(&[(Base, 100.0)]);
        cut.limit_hit = true;
        let limit = choose(&cut, &base, true);
        assert!(limit.why["speculation.sources"].contains("time limit"));
    }

    #[test]
    fn when_off_wins_the_lines_agree_with_what_is_written() {
        let base = SpecConfig::default();
        let c = choose(
            &cal(&[(Base, 100.0), (MeasuredPriors, 110.0), (Off, 140.0)]),
            &base,
            true,
        );
        assert!(!c.enabled);
        assert_eq!(names(&c), ["lookup", "mtp"]);
        assert_eq!(c.priors, None);
        assert!(
            c.why["speculation.sources"].starts_with("Speculation is off, so the sources are left")
        );
        assert!(!c.why.contains_key("speculation.cost_slope_ms"));
        assert!(!c.why.contains_key("speculation.mtp_cost_ms"));
    }

    #[test]
    fn a_score_pools_its_passes() {
        let s = Score {
            decoded: 600,
            seconds: 5.0,
            passes: 3,
        };
        assert_eq!(s.tok_s(), 120.0);
        assert_eq!(
            Score {
                decoded: 0,
                seconds: 0.0,
                passes: 0
            }
            .tok_s(),
            0.0
        );
    }
}
