//! The source decision in one pure function: which source drafts the next step, how many tokens, and whether the step
//! is a probe. `SpecState::step` and the lookup simulator both call it, so they cannot disagree.
use super::{Controller, LookupMatch, Source, SourceChoice, SpecConfig};

/// Which source drafts, how many draft tokens (`k`) and whether MTP's draft is a probe. `k == 0` is plain decode (the
/// source is then `Lookup`, unused).
/// - `lookup_len`: tokens in lookup's draft (0 for none or when lookup isn't a source).
/// - `has_mtp`: the session can draft with MTP (the load allocated it).
/// - `cap`: the most tokens either source may draft, `min(k_max, max_draft, most - 1)`.
///
/// Lookup comes first. If its draft is empty, or its controller choice is 0, MTP's choice comes next. When that is 0
/// too, MTP drafts one token as a probe once `probe_every` plain steps have run since it last drafted. With
/// `fixed_draft` (`--draft-len N`) lookup's draft is taken whole when it has one, else MTP drafts `N`, with no
/// controller and no probe.
///
/// With `source_choice = best` (and no `fixed_draft`) both sources are scored, each by its controller's best
/// `E(k) / (d(k) + c(k + 1))`: the higher score drafts, a tie goes to lookup. When both choose 0 the probe rule is the
/// same as above.
pub fn next_source(
    controller: &Controller,
    cfg: &SpecConfig,
    lookup_len: usize,
    has_mtp: bool,
    cap: usize,
) -> (Source, usize, bool) {
    next_source_for(controller, cfg, lookup_len, None, has_mtp, cap)
}

/// `next_source` with the class of lookup's draft (`Drafter::last_match`): lookup is scored with that class's accept
/// rate (`Controller::lookup_rate`), the pooled rate when `lookup_class` is `None`.
pub fn next_source_for(
    controller: &Controller,
    cfg: &SpecConfig,
    lookup_len: usize,
    lookup_class: Option<LookupMatch>,
    has_mtp: bool,
    cap: usize,
) -> (Source, usize, bool) {
    let use_lookup = cfg.has(Source::Lookup) && lookup_len > 0;
    let use_mtp = cfg.has(Source::Mtp) && has_mtp;
    if let Some(n) = cfg.fixed_draft {
        if use_lookup {
            return (Source::Lookup, n.min(lookup_len).min(cap), false);
        }
        if use_mtp && n.min(cap) > 0 {
            return (Source::Mtp, n.min(cap), false);
        }
        return (Source::Lookup, 0, false);
    }
    if cfg.source_choice == SourceChoice::Best {
        let (k_l, s_l) = if use_lookup {
            controller.choose_scored_as(Source::Lookup, lookup_class, lookup_len, cap)
        } else {
            (0, 0.0)
        };
        let (k_m, s_m) = if use_mtp && cap > 0 {
            controller.choose_scored(Source::Mtp, cap, cap)
        } else {
            (0, 0.0)
        };
        if k_l > 0 && (k_m == 0 || s_l >= s_m) {
            return (Source::Lookup, k_l, false);
        }
        if k_m > 0 {
            return (Source::Mtp, k_m, false);
        }
        if use_mtp && cap > 0 && controller.probe_due() {
            return (Source::Mtp, 1, true);
        }
        return (Source::Lookup, 0, false);
    }
    if use_lookup {
        let k = controller
            .choose_scored_as(Source::Lookup, lookup_class, lookup_len, cap)
            .0;
        if k > 0 {
            return (Source::Lookup, k, false);
        }
    }
    if use_mtp && cap > 0 {
        let k = controller.choose(Source::Mtp, cap, cap);
        if k > 0 {
            return (Source::Mtp, k, false);
        }
        if controller.probe_due() {
            return (Source::Mtp, 1, true);
        }
    }
    (Source::Lookup, 0, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(prior: &[f64]) -> SpecConfig {
        SpecConfig {
            mtp_accept_prior: prior.to_vec(),
            source_choice: SourceChoice::Priority,
            probe_every: 3,
            ..SpecConfig::default()
        }
    }

    /// A controller that has timed a plain step (10.4 ms).
    fn timed(cfg: &SpecConfig) -> Controller {
        let mut c = Controller::new(cfg);
        c.observe_cost(1, 10.4);
        c
    }

    #[test]
    fn lookup_goes_first_and_mtp_takes_what_lookup_declines() {
        let config = cfg(&[0.9, 0.9, 0.9]);
        let mut c = timed(&config);
        // Lookup has a draft and a good rate: its choice.
        let (s, k, probe) = next_source(&c, &config, 4, true, 4);
        assert_eq!((s, probe), (Source::Lookup, false));
        assert!(k > 0);
        // No lookup draft: MTP.
        let (s, k, probe) = next_source(&c, &config, 0, true, 4);
        assert_eq!((s, probe), (Source::Mtp, false));
        assert!(k > 0);
        // A lookup draft its controller declines (a rate of 0): MTP, not plain.
        for _ in 0..200 {
            c.observe_accept(Source::Lookup, 1, 0);
        }
        let (s, k, _) = next_source(&c, &config, 4, true, 4);
        assert_eq!(s, Source::Mtp);
        assert!(k > 0);
        // No MTP in the session: plain.
        assert_eq!(
            next_source(&c, &config, 4, false, 4),
            (Source::Lookup, 0, false)
        );
        // MTP not a source: plain.
        let lookup_only = SpecConfig {
            sources: vec![Source::Lookup],
            ..config.clone()
        };
        assert_eq!(
            next_source(&c, &lookup_only, 4, true, 4),
            (Source::Lookup, 0, false)
        );
        // Lookup not a source: its draft is ignored.
        let mtp_only = SpecConfig {
            sources: vec![Source::Mtp],
            ..config.clone()
        };
        assert_eq!(next_source(&c, &mtp_only, 4, true, 4).0, Source::Mtp);
    }

    #[test]
    fn nothing_drafts_before_a_plain_step_is_timed() {
        let config = cfg(&[0.99]);
        let c = Controller::new(&config);
        assert_eq!(
            next_source(&c, &config, 4, true, 4),
            (Source::Lookup, 0, false)
        );
    }

    #[test]
    fn a_cap_of_0_is_plain() {
        let config = cfg(&[0.9]);
        let c = timed(&config);
        assert_eq!(
            next_source(&c, &config, 4, true, 0),
            (Source::Lookup, 0, false)
        );
    }

    #[test]
    fn mtp_probes_after_probe_every_plain_steps_since_it_last_drafted() {
        // A prior of 0.5 at c(1) 10.4 and the default slope: MTP's choice is 0.
        let config = cfg(&[0.5]);
        let mut c = timed(&config);
        assert_eq!(
            next_source(&c, &config, 0, true, 4),
            (Source::Lookup, 0, false)
        );
        c.observe_plain();
        c.observe_plain();
        assert_eq!(
            next_source(&c, &config, 0, true, 4),
            (Source::Lookup, 0, false)
        );
        c.observe_plain();
        assert_eq!(next_source(&c, &config, 0, true, 4), (Source::Mtp, 1, true));
        // A probe is k = 1 even where the room is larger, and not past the cap.
        assert_eq!(next_source(&c, &config, 0, true, 1), (Source::Mtp, 1, true));
        // After MTP drafted the count starts again.
        c.observe_mtp_drafted();
        assert_eq!(
            next_source(&c, &config, 0, true, 4),
            (Source::Lookup, 0, false)
        );
    }

    /// Lookup at a rate of 0.7 (a draft of 4), MTP at 0.95 a depth: MTP's score is the higher.
    fn lookup_low_mtp_high(choice: SourceChoice) -> (SpecConfig, Controller) {
        let config = SpecConfig {
            accept_prior: 0.7,
            source_choice: choice,
            ..cfg(&[0.95, 0.95, 0.95, 0.95])
        };
        let c = timed(&config);
        (config, c)
    }

    #[test]
    fn best_of_takes_mtp_when_its_score_is_higher() {
        let (config, c) = lookup_low_mtp_high(SourceChoice::Priority);
        assert_eq!(next_source(&c, &config, 4, true, 4).0, Source::Lookup);
        let (config, c) = lookup_low_mtp_high(SourceChoice::Best);
        let (s, k, probe) = next_source(&c, &config, 4, true, 4);
        assert_eq!((s, probe), (Source::Mtp, false));
        assert!(k > 0);
        // Without a lookup draft it is MTP's either way.
        assert_eq!(next_source(&c, &config, 0, true, 4).0, Source::Mtp);
    }

    #[test]
    fn best_of_keeps_lookup_when_its_score_is_higher_or_equal() {
        let config = SpecConfig {
            accept_prior: 0.95,
            source_choice: SourceChoice::Best,
            ..cfg(&[0.7, 0.7, 0.7, 0.7])
        };
        let c = timed(&config);
        let (s, k, probe) = next_source(&c, &config, 4, true, 4);
        assert_eq!((s, probe), (Source::Lookup, false));
        assert!(k > 0);
        // Equal scores: the same rate and no draft cost for MTP is impossible, so a zero draft cost and equal rates tie.
        let tie = SpecConfig {
            accept_prior: 0.8,
            mtp_cost_ms: 0.0,
            source_choice: SourceChoice::Best,
            ..cfg(&[0.8, 0.8, 0.8, 0.8])
        };
        let c = timed(&tie);
        assert_eq!(next_source(&c, &tie, 4, true, 4).0, Source::Lookup);
        // Lookup's controller declines (a rate of 0): MTP takes the step.
        let (config, mut c) = lookup_low_mtp_high(SourceChoice::Best);
        for _ in 0..200 {
            c.observe_accept(Source::Lookup, 1, 0);
        }
        assert_eq!(next_source(&c, &config, 4, true, 4).0, Source::Mtp);
    }

    #[test]
    fn a_continuing_run_beats_mtp_where_the_pooled_rate_loses() {
        let (config, mut c) = lookup_low_mtp_high(SourceChoice::Best);
        let run = LookupMatch {
            n: 4,
            continues: true,
        };
        let short = LookupMatch {
            n: 2,
            continues: false,
        };
        // Continuing drafts are kept whole; 2-token matches fail, three for each run.
        for _ in 0..40 {
            c.observe_accept_as(Source::Lookup, Some(run), 4, 4);
            for _ in 0..3 {
                c.observe_accept_as(Source::Lookup, Some(short), 1, 0);
            }
        }
        // The pooled rate still loses to MTP's 0.95, so a draft with no class goes to MTP...
        assert!(c.accept_rate(Source::Lookup) < 0.95);
        assert_eq!(next_source(&c, &config, 4, true, 4).0, Source::Mtp);
        assert_eq!(
            next_source_for(&c, &config, 4, None, true, 4).0,
            Source::Mtp
        );
        // ...and so does a 2-token match, but a continuing run drafts 4 with lookup.
        assert_eq!(
            next_source_for(&c, &config, 4, Some(short), true, 4).0,
            Source::Mtp
        );
        assert_eq!(
            next_source_for(&c, &config, 4, Some(run), true, 4),
            (Source::Lookup, 4, false)
        );
        // `priority` asks lookup's class rate first: a poor class falls through to MTP rather than plain.
        let prio = SpecConfig {
            source_choice: SourceChoice::Priority,
            ..config.clone()
        };
        assert_eq!(
            next_source_for(&c, &prio, 4, Some(short), true, 4).0,
            Source::Mtp
        );
        assert_eq!(
            next_source_for(&c, &prio, 4, Some(run), true, 4).0,
            Source::Lookup
        );
    }

    #[test]
    fn best_of_probes_as_priority_does() {
        let config = SpecConfig {
            accept_prior: 0.3,
            source_choice: SourceChoice::Best,
            ..cfg(&[0.5])
        };
        let mut c = timed(&config);
        assert_eq!(
            next_source(&c, &config, 0, true, 4),
            (Source::Lookup, 0, false)
        );
        c.observe_plain();
        c.observe_plain();
        c.observe_plain();
        assert_eq!(next_source(&c, &config, 0, true, 4), (Source::Mtp, 1, true));
        // A lookup draft both decline does not stop the probe.
        assert_eq!(next_source(&c, &config, 4, true, 4), (Source::Mtp, 1, true));
        c.observe_mtp_drafted();
        assert_eq!(
            next_source(&c, &config, 0, true, 4),
            (Source::Lookup, 0, false)
        );
    }

    #[test]
    fn a_fixed_draft_takes_lookup_whole_else_mtp_at_n() {
        let config = SpecConfig {
            fixed_draft: Some(3),
            ..cfg(&[0.1])
        };
        let c = Controller::new(&config);
        assert_eq!(
            next_source(&c, &config, 2, true, 4),
            (Source::Lookup, 2, false)
        );
        assert_eq!(
            next_source(&c, &config, 5, true, 4),
            (Source::Lookup, 3, false)
        );
        assert_eq!(
            next_source(&c, &config, 0, true, 4),
            (Source::Mtp, 3, false)
        );
        assert_eq!(
            next_source(&c, &config, 0, true, 2),
            (Source::Mtp, 2, false)
        );
        assert_eq!(
            next_source(&c, &config, 0, false, 4),
            (Source::Lookup, 0, false)
        );
    }
}
