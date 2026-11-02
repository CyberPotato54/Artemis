//! Speculative decoding's CPU side: the config, the prompt-lookup drafter, the draft-length
//! controller and the step loop. Nothing here touches the GPU.
mod controller;
mod lookup;
mod next_source;
mod step;

pub use controller::{Controller, MeasuredCosts};
pub use lookup::Lookup;
pub use next_source::{next_source, next_source_for};
pub use step::{
    score_against, Next, SizeRow, SpecDelta, SpecState, SpecTotals, StepTimed, Stepped, Verifier,
};

/// The most tokens one verify step runs (`x` and up to 15 drafts): the token kernels' cap.
pub const MAX_VERIFY: usize = kernels::tokens::MAX_TOKENS;

/// Where a draft comes from: the prompt-lookup drafter, or the model's own MTP block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Lookup,
    Mtp,
}

impl Source {
    /// The name the config, the counters and the records spell.
    pub fn name(self) -> &'static str {
        match self {
            Source::Lookup => "lookup",
            Source::Mtp => "mtp",
        }
    }

    /// The source named `s`: "lookup" or "mtp".
    pub fn parse(s: &str) -> Option<Source> {
        match s {
            "lookup" => Some(Source::Lookup),
            "mtp" => Some(Source::Mtp),
            _ => None,
        }
    }

    /// A `sources` list from its names, refused by the rules of `check`, and for an unknown name.
    pub fn parse_list(names: &[&str]) -> Result<Vec<Source>, (&'static str, String)> {
        let mut out: Vec<Source> = Vec::new();
        for name in names {
            let Some(s) = Source::parse(name) else {
                return Err(("sources", format!("an unknown source {name:?}")));
            };
            out.push(s);
        }
        check_sources(&out)?;
        Ok(out)
    }
}

fn check_sources(sources: &[Source]) -> Result<(), (&'static str, String)> {
    if sources.is_empty() {
        return Err((
            "sources",
            "needs a source; to turn speculation off set enabled = false".to_string(),
        ));
    }
    for (i, s) in sources.iter().enumerate() {
        if sources[..i].contains(s) {
            return Err(("sources", format!("{:?} is listed twice", s.name())));
        }
    }
    Ok(())
}

/// How `next_source` picks between lookup's draft and MTP's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceChoice {
    /// Lookup's draft when its controller takes it, else MTP's.
    Priority,
    /// The source whose draft scores higher (tokens per ms); a tie goes to lookup.
    Best,
}

impl SourceChoice {
    /// The name the config spells.
    pub fn name(self) -> &'static str {
        match self {
            SourceChoice::Priority => "priority",
            SourceChoice::Best => "best",
        }
    }

    /// The choice named `s`: "priority" or "best".
    pub fn parse(s: &str) -> Option<SourceChoice> {
        match s {
            "priority" => Some(SourceChoice::Priority),
            "best" => Some(SourceChoice::Best),
            _ => None,
        }
    }
}

/// How speculation behaves. The server's `[speculation]` table and `artemis generate`'s flags
/// map onto it.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecConfig {
    /// `false` is today's decode path, with nothing allocated for speculation.
    pub enabled: bool,
    /// The most draft tokens a step verifies, from 1 to 15.
    pub max_draft: usize,
    /// Prompt lookup's shortest n-gram.
    pub ngram_min: usize,
    /// Prompt lookup's longest n-gram, at most 16.
    pub ngram_max: usize,
    /// The predicted speed-up below which a step runs plain decode.
    pub min_gain: f64,
    /// Draft tokens per halving of an old outcome's weight in the accept rate.
    pub accept_half_life: f64,
    /// The controller's cost per extra token of a step, in ms, until it has measured its own.
    pub cost_slope_ms: f64,
    /// The accept rate a fresh controller starts from.
    pub accept_prior: f64,
    /// `artemis generate --draft-len`: bypasses the controller. `None` everywhere else.
    pub fixed_draft: Option<usize>,
    /// The draft sources in use. The list names which run; its order does nothing. `source_choice` says how a
    /// step picks between them (`priority`: lookup first; `best`: the higher predicted tokens/ms).
    pub sources: Vec<Source>,
    /// MTP drafts at `k = 1` once this many plain steps have run since it last drafted.
    pub probe_every: usize,
    /// The last positions of the prompt that prefill writes into the MTP's KV.
    pub mtp_window: usize,
    /// The MTP KV's rows (positions), before the load clamps them to the session.
    pub mtp_ctx: usize,
    /// The hidden-row ring's length, before the load clamps it.
    pub mtp_ring: usize,
    /// MTP's accept rate at each draft depth until it has measured its own; a depth past the list takes the last value.
    pub mtp_accept_prior: Vec<f64>,
    /// The prior for the draft pass's time per draft token, in ms, until it is measured.
    pub mtp_cost_ms: f64,
    /// Which source drafts when both have a draft: lookup first (`priority`) or the higher score (`best`, the default,
    /// as measured).
    pub source_choice: SourceChoice,
    /// 0 for MTP's full head; N drafts from the first N rows of the main head (a prefix view, no VRAM), 1024 or more. A
    /// vocabulary smaller than N gives the full head. The default is the measured choice.
    pub mtp_head_rows: usize,
}

impl Default for SpecConfig {
    fn default() -> SpecConfig {
        SpecConfig {
            enabled: true,
            max_draft: 4,
            ngram_min: 2,
            ngram_max: 4,
            min_gain: 1.03,
            accept_half_life: 32.0,
            cost_slope_ms: 4.4,
            accept_prior: 0.5,
            fixed_draft: None,
            sources: vec![Source::Lookup, Source::Mtp],
            probe_every: 64,
            mtp_window: 2048,
            mtp_ctx: 8192,
            mtp_ring: 256,
            mtp_accept_prior: vec![0.79, 0.81, 0.74, 0.78],
            mtp_cost_ms: 1.6,
            source_choice: SourceChoice::Best,
            mtp_head_rows: 65536,
        }
    }
}

impl SpecConfig {
    /// Whether `s` is one of the sources in use.
    pub fn has(&self, s: Source) -> bool {
        self.sources.contains(&s)
    }

    /// The rules for `sources` and the MTP keys.
    fn check_mtp_keys(&self) -> Result<(), (&'static str, String)> {
        let refuse = |key: &'static str, why: &str| Err((key, why.to_string()));
        check_sources(&self.sources)?;
        if self.probe_every < 1 {
            return refuse("probe_every", "must be at least 1");
        }
        if !(512..=65536).contains(&self.mtp_ctx) || !self.mtp_ctx.is_multiple_of(256) {
            return refuse("mtp_ctx", "must be from 512 to 65536 and a multiple of 256");
        }
        if !(1..=self.mtp_ctx / 2).contains(&self.mtp_window) {
            return Err((
                "mtp_window",
                format!("must be from 1 to mtp_ctx / 2 ({})", self.mtp_ctx / 2),
            ));
        }
        if !(16..=self.mtp_ctx / 4).contains(&self.mtp_ring) || !self.mtp_ring.is_multiple_of(16) {
            return Err((
                "mtp_ring",
                format!(
                    "must be from 16 to mtp_ctx / 4 ({}) and a multiple of 16",
                    self.mtp_ctx / 4
                ),
            ));
        }
        if !(1..MAX_VERIFY).contains(&self.mtp_accept_prior.len())
            || !self
                .mtp_accept_prior
                .iter()
                .all(|a| (0.0..=1.0).contains(a))
        {
            return Err((
                "mtp_accept_prior",
                format!("must be 1 to {} values, each from 0 to 1", MAX_VERIFY - 1),
            ));
        }
        if !(self.mtp_cost_ms.is_finite() && self.mtp_cost_ms >= 0.0) {
            return refuse("mtp_cost_ms", "must be a finite number, 0 or more");
        }
        // The upper bound (the vocabulary) is a clamp at load (`mtp::head_rows`), where the vocabulary is known.
        if self.mtp_head_rows != 0 && self.mtp_head_rows < 1024 {
            return refuse(
                "mtp_head_rows",
                "must be 0 (the full head) or at least 1024",
            );
        }
        Ok(())
    }

    /// Refuses a setting the controller or the drafter can't honour: the key's name, and why.
    pub fn check(&self) -> Result<(), (&'static str, String)> {
        let refuse = |key: &'static str, why: &str| Err((key, why.to_string()));
        if !(1..MAX_VERIFY).contains(&self.max_draft) {
            return refuse(
                "max_draft",
                &format!("must be from 1 to {}", MAX_VERIFY - 1),
            );
        }
        if self.ngram_min < 1 {
            return refuse("ngram_min", "must be at least 1");
        }
        if self.ngram_max > 16 {
            return refuse("ngram_max", "must be at most 16");
        }
        if self.ngram_min > self.ngram_max {
            return refuse("ngram_min", "must be at most ngram_max");
        }
        if !(self.min_gain.is_finite() && self.min_gain >= 1.0) {
            return refuse("min_gain", "must be a finite number, 1 or more");
        }
        if !(self.accept_half_life.is_finite() && self.accept_half_life > 0.0) {
            return refuse("accept_half_life", "must be a finite number above 0");
        }
        if !(self.cost_slope_ms.is_finite() && self.cost_slope_ms >= 0.0) {
            return refuse("cost_slope_ms", "must be a finite number, 0 or more");
        }
        if !(0.0..=1.0).contains(&self.accept_prior) {
            return refuse("accept_prior", "must be from 0 to 1");
        }
        if self.fixed_draft.is_some_and(|n| n >= MAX_VERIFY) {
            return refuse(
                "fixed_draft",
                &format!("must be at most {}", MAX_VERIFY - 1),
            );
        }
        if let Some(n) = self.fixed_draft.filter(|&n| n > self.max_draft) {
            return Err((
                "fixed_draft",
                format!("{n} is above max_draft {}", self.max_draft),
            ));
        }
        self.check_mtp_keys()?;
        Ok(())
    }
}

/// A source of draft tokens. MTP's drafts don't come through this trait: the verifier makes them
/// (`Verifier::mtp_draft`).
pub trait Drafter: Send {
    /// The source's name, as the counters and records spell it.
    fn name(&self) -> &'static str;
    /// Up to `max` tokens that probably follow the tokens seen so far; empty for no draft.
    fn draft(&mut self, max: usize) -> Vec<u32>;
    /// Appends tokens the request has committed (the prompt, then each token emitted, `x` included).
    fn observe(&mut self, tokens: &[u32]);
    /// What produced the last `draft`, when the source can say: the controller keeps a separate accept rate per
    /// `LookupMatch::class`. `None` (the default, and after an empty draft) uses the one pooled rate.
    fn last_match(&self) -> Option<LookupMatch> {
        None
    }
}

/// What produced a prompt-lookup draft. Lookup's drafts differ a lot in quality by how they were found: on Qwen's
/// own tokens (the speed and golden streams) a draft from a 2-token match keeps about half its tokens, one from a
/// 4-token match nearly nine in ten, and one that carries on the run the previous draft came from more than nine
/// in ten. One rate over all of them under-drafts the good ones and over-drafts the poor ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookupMatch {
    /// The length of the suffix that matched (`ngram_min` to `ngram_max`).
    pub n: usize,
    /// The match is the previous draft's source moved on by exactly the tokens committed since: the request is still
    /// copying the same run.
    pub continues: bool,
}

impl LookupMatch {
    /// How many classes `class` returns.
    pub const CLASSES: usize = 4;

    /// The rate class: 3 for a continuing run, else 2 for a match of 4 tokens or more, 1 for 3 tokens, 0 for 2 or
    /// fewer.
    pub fn class(self) -> usize {
        if self.continues {
            3
        } else {
            match self.n {
                0..=2 => 0,
                3 => 1,
                _ => 2,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_passes_check() {
        assert_eq!(SpecConfig::default().check(), Ok(()));
        // The measured defaults: max_draft maximises E(k)/c(k+1) at the edit prompt's accept rate; the slope is
        // the verify step's, per extra token, at 8,192 positions.
        let d = SpecConfig::default();
        assert_eq!(
            (d.max_draft, d.cost_slope_ms, d.accept_prior),
            (4, 4.4, 0.5)
        );
        // The pooled MTP depth rates (a2 .. a4 = 0.812, 0.744, 0.776, rounded to 0.01) and the edit-prompt depth 1
        // (0.79).
        assert_eq!(d.mtp_accept_prior, [0.79, 0.81, 0.74, 0.78]);
        // Best-of sources, and the head cut to 65,536 rows (median 125.9 against 115.4 tok/s).
        assert_eq!(
            (d.source_choice, d.mtp_head_rows),
            (SourceChoice::Best, 65536)
        );
    }

    #[test]
    fn check_names_each_bad_key() {
        let bad = |edit: &dyn Fn(&mut SpecConfig)| {
            let mut c = SpecConfig::default();
            edit(&mut c);
            c.check().expect_err("the config should be refused").0
        };
        assert_eq!(bad(&|c| c.max_draft = 0), "max_draft");
        assert_eq!(bad(&|c| c.max_draft = 16), "max_draft");
        assert_eq!(bad(&|c| c.ngram_min = 0), "ngram_min");
        assert_eq!(bad(&|c| c.ngram_max = 17), "ngram_max");
        assert_eq!(
            bad(&|c| {
                c.ngram_min = 3;
                c.ngram_max = 2;
            }),
            "ngram_min"
        );
        assert_eq!(bad(&|c| c.min_gain = 0.9), "min_gain");
        assert_eq!(bad(&|c| c.min_gain = f64::NAN), "min_gain");
        assert_eq!(bad(&|c| c.accept_half_life = 0.0), "accept_half_life");
        assert_eq!(
            bad(&|c| c.accept_half_life = f64::INFINITY),
            "accept_half_life"
        );
        assert_eq!(bad(&|c| c.cost_slope_ms = f64::NAN), "cost_slope_ms");
        assert_eq!(bad(&|c| c.cost_slope_ms = -0.1), "cost_slope_ms");
        assert_eq!(bad(&|c| c.accept_prior = 1.5), "accept_prior");
        assert_eq!(bad(&|c| c.accept_prior = -0.1), "accept_prior");
        assert_eq!(bad(&|c| c.fixed_draft = Some(16)), "fixed_draft");
        // A fixed length above max_draft would silently run shorter: refused, naming both values.
        let c = SpecConfig {
            fixed_draft: Some(8),
            ..SpecConfig::default()
        };
        let (key, why) = c.check().expect_err("8 is above max_draft 4");
        assert_eq!(key, "fixed_draft");
        assert!(why.contains('8') && why.contains("max_draft 4"), "{why}");
    }

    #[test]
    fn check_names_each_new_bad_key() {
        let bad = |edit: &dyn Fn(&mut SpecConfig)| {
            let mut c = SpecConfig::default();
            edit(&mut c);
            c.check().expect_err("the config should be refused")
        };
        let (key, why) = bad(&|c| c.sources = vec![]);
        assert_eq!(key, "sources");
        assert!(why.contains("needs a source; to turn speculation off set enabled = false"));
        let (key, why) = bad(&|c| c.sources = vec![Source::Lookup, Source::Lookup]);
        assert_eq!(
            (key, why.as_str()),
            ("sources", "\"lookup\" is listed twice")
        );
        assert_eq!(
            Source::parse_list(&["lookup", "x"]),
            Err(("sources", "an unknown source \"x\"".to_string()))
        );
        assert_eq!(
            Source::parse_list(&["mtp", "mtp"]).unwrap_err().1,
            "\"mtp\" is listed twice"
        );
        assert_eq!(
            Source::parse_list(&["mtp", "lookup"]),
            Ok(vec![Source::Mtp, Source::Lookup])
        );
        assert_eq!(bad(&|c| c.probe_every = 0).0, "probe_every");
        assert_eq!(bad(&|c| c.mtp_ctx = 500).0, "mtp_ctx");
        assert_eq!(bad(&|c| c.mtp_ctx = 769).0, "mtp_ctx");
        assert_eq!(bad(&|c| c.mtp_ctx = 65792).0, "mtp_ctx");
        assert_eq!(bad(&|c| c.mtp_window = 0).0, "mtp_window");
        assert_eq!(bad(&|c| c.mtp_window = 4097).0, "mtp_window");
        assert_eq!(bad(&|c| c.mtp_ring = 8).0, "mtp_ring");
        assert_eq!(bad(&|c| c.mtp_ring = 24).0, "mtp_ring");
        assert_eq!(bad(&|c| c.mtp_ring = 2064).0, "mtp_ring");
        assert_eq!(bad(&|c| c.mtp_accept_prior = vec![]).0, "mtp_accept_prior");
        assert_eq!(
            bad(&|c| c.mtp_accept_prior = vec![0.5; 16]).0,
            "mtp_accept_prior"
        );
        assert_eq!(
            bad(&|c| c.mtp_accept_prior = vec![0.5, 1.1]).0,
            "mtp_accept_prior"
        );
        assert_eq!(bad(&|c| c.mtp_cost_ms = f64::NAN).0, "mtp_cost_ms");
        assert_eq!(bad(&|c| c.mtp_cost_ms = -1.0).0, "mtp_cost_ms");
        assert_eq!(bad(&|c| c.mtp_head_rows = 512).0, "mtp_head_rows");
        assert_eq!(bad(&|c| c.mtp_head_rows = 1023).0, "mtp_head_rows");
        assert!(bad(&|c| c.mtp_head_rows = 512).1.contains("1024"));
        for rows in [0, 1024, 32768] {
            let c = SpecConfig {
                mtp_head_rows: rows,
                ..SpecConfig::default()
            };
            assert_eq!(c.check(), Ok(()), "{rows}");
        }
        assert_eq!(SourceChoice::parse("best"), Some(SourceChoice::Best));
        assert_eq!(SourceChoice::parse("Best"), None);
        assert_eq!(SourceChoice::Priority.name(), "priority");
        // Each key's message names its limit.
        let why = |edit: &dyn Fn(&mut SpecConfig)| bad(edit).1;
        assert!(why(&|c| c.probe_every = 0).contains("at least 1"));
        assert!(why(&|c| c.mtp_ctx = 769).contains("multiple of 256"));
        assert!(why(&|c| c.mtp_window = 4097).contains("mtp_ctx / 2 (4096)"));
        assert!(why(&|c| c.mtp_ring = 2064).contains("mtp_ctx / 4 (2048)"));
        assert!(why(&|c| c.mtp_accept_prior = vec![]).contains("1 to 15 values"));
        assert!(why(&|c| c.mtp_cost_ms = f64::NAN).contains("finite"));
        // The edges pass.
        let edge = SpecConfig {
            mtp_ctx: 512,
            mtp_window: 256,
            mtp_ring: 128,
            mtp_accept_prior: vec![0.0; 15],
            mtp_cost_ms: 0.0,
            sources: vec![Source::Mtp],
            ..SpecConfig::default()
        };
        assert_eq!(edge.check(), Ok(()));
        assert!(edge.has(Source::Mtp) && !edge.has(Source::Lookup));
        assert_eq!(Source::parse("lookup").map(Source::name), Some("lookup"));
        assert_eq!(Source::parse("Mtp"), None);
    }

    #[test]
    fn the_edges_of_each_range_pass() {
        let ok = |edit: &dyn Fn(&mut SpecConfig)| {
            let mut c = SpecConfig::default();
            edit(&mut c);
            assert_eq!(c.check(), Ok(()));
        };
        ok(&|c| c.max_draft = 1);
        ok(&|c| c.max_draft = 15);
        ok(&|c| c.ngram_min = 1);
        ok(&|c| c.ngram_max = 16);
        ok(&|c| {
            c.ngram_min = 4;
            c.ngram_max = 4;
        });
        ok(&|c| c.min_gain = 1.0);
        ok(&|c| c.cost_slope_ms = 0.0);
        ok(&|c| c.accept_prior = 0.0);
        ok(&|c| c.accept_prior = 1.0);
        ok(&|c| c.fixed_draft = Some(0));
        ok(&|c| c.fixed_draft = Some(c.max_draft));
        ok(&|c| {
            c.max_draft = 15;
            c.fixed_draft = Some(15);
        });
    }
}
