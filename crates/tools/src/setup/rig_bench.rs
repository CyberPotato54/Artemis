//! The calibration's bench over a real engine (`RigBench`) and the engine options setup measures with. Setup opens the
//! engine as `artemis serve` will: the user's `ctx`, `cache_slots`, `device_uuid`, `f32_prefill` and `[speculation]`
//! table, so what it measures is what the server runs.
use super::calibrate::{Bench, Load, LoadInfo, Pass};
use super::choose::{SpecCandidate, CAL_SEED, CAL_TOKENS};
use super::prompts;
use crate::serve::backend::spec_config;
use crate::spec_check::{rig_options, Job, Mode, Rig};
use anyhow::{anyhow, bail, Context, Result};
use engine::spec::{MeasuredCosts, SpecState};
use engine::{Options, Sampling, SpecConfig, SpecSource};
use gguf::Gguf;
use qwen36::{
    dims::Dims,
    tokenizer::{chat_template, Tokenizer, Vocab},
};
use server::template::Template;
use std::collections::BTreeMap;
use std::time::Instant;

/// The tokens the warm-up decodes (untimed): enough to fill the expert cache's first swaps and to settle the clocks.
const WARM_TOKENS: usize = 32;

/// The positions a prompt leaves free beyond its decode, so a short `ctx` does not stop a pass half way.
const CTX_SLACK: usize = 32;

/// The engine options, speculation config and context for a load of the model under the user's config. `WithoutMtp`
/// sets `sources = [Lookup]`, which loads none of MTP's memory.
pub fn bench_options(cfg: &server::Config, load: Load) -> Result<(Options, SpecConfig, usize)> {
    let uuid = match &cfg.device_uuid {
        Some(text) => Some(engine::parse_uuid(text).ok_or_else(|| {
            anyhow!("device_uuid is not a UUID: it is 32 hex digits, with or without dashes")
        })?),
        None => None,
    };
    let ctx = cfg.ctx as usize;
    let spec = spec_config(&cfg.speculation);
    let spec = match load {
        Load::WithMtp => spec,
        Load::WithoutMtp => Mode::Lookup.load_config(&spec),
    };
    let mut opts = rig_options(&spec, ctx, uuid)?;
    opts.cache_slots = cfg.cache_slots.map(|n| n as usize);
    opts.mtp_slots = Some(cfg.mtp_expert_slots as usize);
    opts.f32_prefill = cfg.f32_prefill;
    Ok((opts, spec, ctx))
}

/// What one load reported, kept for the facts setup writes.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadRecord {
    pub load: Load,
    pub info: LoadInfo,
    pub gpu_uuid: [u8; 16],
    /// The host-cached heap's usage by this process after the load (the pool's input, `pool_on_machine`).
    pub host_heap_usage: Option<u64>,
}

/// A `Bench` over real engines. At most one engine is up at a time: `load` drops the old one before it opens the new.
pub struct RigBench<'a> {
    cfg: server::Config,
    g: &'a Gguf,
    eog: Vec<u32>,
    /// The calibration prompts that fit the context, as token ids.
    prompts: Vec<Vec<u32>>,
    /// The warm-up's prompt: the edit prompt, or the longest that fits.
    warm: Vec<u32>,
    rig: Option<Rig>,
    /// Each candidate's speculation state, kept across its passes and dropped with the load.
    states: BTreeMap<SpecCandidate, SpecState>,
    passes: BTreeMap<SpecCandidate, usize>,
    loads_planned: usize,
    only_off: bool,
    pub loads: Vec<LoadRecord>,
}

/// The three calibration prompts as token ids by the model's own template and vocabulary, and the vocabulary's
/// end-of-generation ids.
fn calibration_prompts(g: &Gguf) -> Result<(Vec<Vec<u32>>, Vec<u32>)> {
    let vocab = Vocab::from_gguf(g).map_err(anyhow::Error::msg)?;
    let tokenizer = Tokenizer::new(&vocab).map_err(anyhow::Error::msg)?;
    let source = chat_template(g).map_err(anyhow::Error::msg)?;
    let template = Template::new(source).map_err(|e| anyhow!("the chat template: {e}"))?;
    let mut out = Vec::new();
    for text in [
        prompts::WRITE.to_string(),
        prompts::EXPLAIN.to_string(),
        prompts::edit(),
    ] {
        out.push(prompts::render(&template, &tokenizer, &text).map_err(anyhow::Error::msg)?);
    }
    Ok((out, vocab.eog()))
}

/// The prompts of `all` that fit a session of `ctx` positions with `CAL_TOKENS` decoded after them.
pub fn fitting(all: Vec<Vec<u32>>, ctx: usize) -> Vec<Vec<u32>> {
    all.into_iter()
        .filter(|p| p.len() + CAL_TOKENS + CTX_SLACK <= ctx)
        .collect()
}

impl<'a> RigBench<'a> {
    /// A bench for the model `g` under `cfg` (its device already chosen). `loads_planned` is the number of loads the
    /// calibration will make, for the progress lines; `only_off` is set when plain decode is the only candidate.
    pub fn new(
        cfg: server::Config,
        g: &'a Gguf,
        loads_planned: usize,
        only_off: bool,
    ) -> Result<RigBench<'a>> {
        let (all, eog) = calibration_prompts(g)?;
        let ctx = cfg.ctx as usize;
        // The edit prompt is last and longest: it warms up.
        let edit = all.last().cloned().unwrap_or_default();
        let prompts = fitting(all, ctx);
        let Some(longest) = prompts.iter().max_by_key(|p| p.len()).cloned() else {
            bail!(
                "ctx is {ctx}, which is too small to measure speculation on; setup needs room for a prompt of about \
                 40 tokens and {CAL_TOKENS} generated ones"
            );
        };
        let warm = if edit.len() + WARM_TOKENS + CTX_SLACK <= ctx {
            edit
        } else {
            longest
        };
        Ok(RigBench {
            cfg,
            g,
            eog,
            prompts,
            warm,
            rig: None,
            states: BTreeMap::new(),
            passes: BTreeMap::new(),
            loads_planned,
            only_off,
            loads: Vec::new(),
        })
    }

    fn rig(&mut self) -> Result<&mut Rig> {
        self.rig
            .as_mut()
            .ok_or_else(|| anyhow!("no model is loaded"))
    }
}

/// What a candidate's measurement is called in the progress line.
pub fn what(c: SpecCandidate, base: &SpecConfig) -> &'static str {
    match c {
        SpecCandidate::Base if base.has(SpecSource::Mtp) => "speculation with MTP",
        SpecCandidate::Base | SpecCandidate::LookupOnly => "prompt lookup alone",
        SpecCandidate::MeasuredPriors => "measured cost priors",
        SpecCandidate::Off => "plain decode",
    }
}

/// The passes a candidate gets: one for the measured priors (the second round only) and for plain decode alone, else
/// two.
pub fn rounds(c: SpecCandidate, only_off: bool) -> usize {
    if c == SpecCandidate::MeasuredPriors || only_off {
        1
    } else {
        2
    }
}

impl Bench for RigBench<'_> {
    fn load(&mut self, load: Load) -> Result<LoadInfo> {
        // The old engine goes first: two never share the GPU.
        self.rig = None;
        self.states.clear();
        self.passes.clear();
        println!(
            "Loading the model ({} of {})...",
            self.loads.len() + 1,
            self.loads_planned
        );
        let (opts, spec, ctx) = bench_options(&self.cfg, load)?;
        let started = Instant::now();
        let rig = Rig::from_options(self.g, &Dims::QWEN36, opts, ctx, spec, self.eog.clone())
            .context("loading the model")?;
        let e = rig.engine();
        let info = LoadInfo {
            seconds: started.elapsed().as_secs_f64(),
            cache_slots: e.stats().slots_per_layer,
            mtp_slots: e.stats().mtp_slots,
            mtp_slot_bytes: e.stats().mtp_slot_bytes,
        };
        println!("{}", info.summary());
        self.loads.push(LoadRecord {
            load,
            info: info.clone(),
            gpu_uuid: e.device_uuid(),
            host_heap_usage: e.host_heap_usage().ok(),
        });
        self.rig = Some(rig);
        Ok(info)
    }

    fn unload(&mut self) {
        self.rig = None;
        self.states.clear();
    }

    fn warm_up(&mut self) -> Result<()> {
        let warm = self.warm.clone();
        let job = Job {
            sampling: Sampling::DEFAULT,
            greedy: false,
            seed: CAL_SEED,
            max_tokens: WARM_TOKENS,
            fingerprints: false,
        };
        self.rig()?
            .run(Mode::Off, &warm, &job)
            .context("warming up")?;
        Ok(())
    }

    fn pass(&mut self, c: SpecCandidate, spec: Option<&SpecConfig>) -> Result<Pass> {
        let done = self.passes.entry(c).or_insert(0);
        *done += 1;
        let (i, n) = (*done, rounds(c, self.only_off));
        let base = spec_config(&self.cfg.speculation);
        println!("Measuring {} ({i} of {n})...", what(c, &base));
        let job = Job {
            sampling: Sampling::DEFAULT,
            greedy: false,
            seed: CAL_SEED,
            max_tokens: CAL_TOKENS,
            fingerprints: false,
        };
        let prompts = self.prompts.clone();
        // A candidate keeps its own speculation state across its passes, as the server's engine-wide state does: the
        // rig runs with it in place, and it is put back afterwards.
        let mut held = spec.map(|s| {
            self.states
                .remove(&c)
                .unwrap_or_else(|| SpecState::new(s.clone()))
        });
        let rig = self.rig()?;
        let mode = match c {
            SpecCandidate::Off => Mode::Off,
            SpecCandidate::LookupOnly if rig.engine().mtp().is_none() => Mode::Lookup,
            _ => Mode::On,
        };
        if let Some(s) = held.as_mut() {
            std::mem::swap(&mut rig.state, s);
        }
        let mut total = Pass {
            decoded: 0,
            seconds: 0.0,
        };
        let mut result = Ok(());
        for prompt in &prompts {
            match rig.run(mode, prompt, &job) {
                Ok(r) => {
                    total.decoded += r.timings.decoded;
                    total.seconds += r.timings.generate_s;
                }
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        if let Some(mut s) = held {
            std::mem::swap(&mut rig.state, &mut s);
            self.states.insert(c, s);
        }
        result.with_context(|| format!("measuring {}", what(c, &base)))?;
        Ok(total)
    }

    fn measured(&self, c: SpecCandidate) -> MeasuredCosts {
        self.states.get(&c).map_or(
            MeasuredCosts {
                slope_ms: None,
                mtp_d1_ms: None,
            },
            SpecState::measured,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_options_follow_the_users_config() {
        let cfg = server::Config {
            ctx: 32_768,
            cache_slots: Some(100),
            mtp_expert_slots: 16,
            f32_prefill: true,
            device_uuid: Some("00000000-4800-0000-0000-000000000000".into()),
            ..server::Config::default()
        };
        let (opts, spec, ctx) = bench_options(&cfg, Load::WithMtp).unwrap();
        assert_eq!(ctx, 32_768);
        assert_eq!(opts.ctx, 32_768);
        assert_eq!(opts.cache_slots, Some(100));
        assert_eq!(opts.mtp_slots, Some(16));
        assert!(opts.f32_prefill);
        let mut uuid = [0u8; 16];
        uuid[4] = 0x48;
        assert_eq!(opts.device_uuid, Some(uuid));
        assert!(opts.speculation);
        assert!(opts.mtp.is_some(), "the load with MTP asks for it");
        assert!(spec.has(SpecSource::Mtp) && spec.has(SpecSource::Lookup));

        // The load without MTP: lookup alone, and none of MTP's memory.
        let (opts, spec, ctx) = bench_options(&cfg, Load::WithoutMtp).unwrap();
        assert_eq!(ctx, 32_768);
        assert_eq!(spec.sources, [SpecSource::Lookup]);
        assert!(opts.mtp.is_none());
        assert_eq!(opts.cache_slots, Some(100));
        assert!(opts.f32_prefill);

        // Defaults come through as the server's own.
        let (opts, _, ctx) = bench_options(&server::Config::default(), Load::WithMtp).unwrap();
        assert_eq!(
            (ctx, opts.cache_slots, opts.f32_prefill),
            (65_536, None, false)
        );
        assert_eq!(opts.device_uuid, None);
    }

    #[test]
    fn the_users_speculation_table_reaches_the_load() {
        let mut cfg = server::Config::default();
        cfg.speculation.sources = vec!["lookup".into()];
        cfg.speculation.cost_slope_ms = 7.5;
        let (opts, spec, _) = bench_options(&cfg, Load::WithMtp).unwrap();
        assert!(opts.mtp.is_none(), "no MTP source, no MTP memory");
        assert_eq!(spec.cost_slope_ms, 7.5);
        // Speculation turned off in the user's file loads no MTP either.
        let mut cfg = server::Config::default();
        cfg.speculation.enabled = false;
        let (opts, spec, _) = bench_options(&cfg, Load::WithMtp).unwrap();
        assert!(opts.mtp.is_none() && !spec.enabled);
    }

    #[test]
    fn a_bad_device_uuid_is_refused_naming_the_key() {
        let cfg = server::Config {
            device_uuid: Some("not-a-uuid".into()),
            ..server::Config::default()
        };
        let e = bench_options(&cfg, Load::WithMtp).unwrap_err().to_string();
        assert!(e.contains("device_uuid"), "{e}");
    }

    #[test]
    fn a_short_context_keeps_only_the_prompts_that_fit() {
        let prompts = vec![vec![1; 40], vec![1; 45], vec![1; 3000]];
        assert_eq!(fitting(prompts.clone(), 65_536).len(), 3);
        // 3000 + 256 + 32 does not fit 2048 positions; the short ones do.
        assert_eq!(fitting(prompts.clone(), 2048).len(), 2);
        assert!(fitting(prompts, 300).is_empty());
    }

    #[test]
    fn the_progress_names_and_rounds() {
        let with = SpecConfig::default();
        let without = SpecConfig {
            sources: vec![SpecSource::Lookup],
            ..SpecConfig::default()
        };
        assert_eq!(what(SpecCandidate::Base, &with), "speculation with MTP");
        assert_eq!(what(SpecCandidate::Base, &without), "prompt lookup alone");
        assert_eq!(
            what(SpecCandidate::MeasuredPriors, &with),
            "measured cost priors"
        );
        assert_eq!(
            what(SpecCandidate::LookupOnly, &with),
            "prompt lookup alone"
        );
        assert_eq!(what(SpecCandidate::Off, &with), "plain decode");
        assert_eq!(rounds(SpecCandidate::Base, false), 2);
        assert_eq!(rounds(SpecCandidate::Off, false), 2);
        assert_eq!(rounds(SpecCandidate::Off, true), 1);
        assert_eq!(rounds(SpecCandidate::MeasuredPriors, false), 1);
    }
}
