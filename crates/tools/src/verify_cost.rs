//! `artemis verify-cost`: what a `t`-token pass costs on the prefill path, `c(t)`.
//!
//! One session runs the prompt in ascending contexts. At each `--at` the prompt is prefilled up to that position (a
//! long fill is staged, and staging lends cache rows that come back empty), then `warmup` untimed decode steps, then
//! `reps` rounds. A round is one timed decode step (`t = 1`) and one timed `prefill_each` call over the prompt's next
//! `t` tokens for every `t >= 2` (every row's logits are read back, as a verify pass would). The rounds interleave the
//! `t`s, so whatever the cache and the clock are still doing spreads over all of them alike.
//!
//! With `--verify` each round also times verify steps of every `t` in `--t`, `t = 1` included: `Session::verify` over
//! the prompt's next `t` tokens, every row read as the step loop reads them, then `commit(1)` (`path: "verify"`; the
//! position advances by one, as a decode's does, so the replay costs its one-token minimum) and, for `t >= 2`, a second
//! step that keeps all `t` (`path: "verify_all"`, `commit(t)`), so the replay's growth with the tokens kept shows. The engine opens
//! with speculation on for it, which `Session::verify` asserts. The rows sit beside the prefill rows of the same `t`,
//! measured in the same rounds.
//!
//! With `--mtp` each round also times the MTP's draft pass, `Session::mtp_draft`, for every `k` in `--t` that is at
//! most 15 (`path: "mtp_draft"`, `t = k`: `d(k)`). Every timed pass has the same prelude, so each
//! catches up exactly one position, as a draft after a plain step does: an untimed pass at `k = 1` (which writes `x`'s
//! row and absorbs whatever the round's other steps left without one, the decode, the prefill passes and the verify
//! steps), then two untimed decode steps (the first commits `x`, whose row the pass kept; the second commits a position
//! with no row), then the timed pass for the next token. The pass moves nothing of the main state; its two decodes
//! are the only move. The engine opens with the MTP (`MtpOptions::for_session` of the default speculation config),
//! which adds the ring's copy to every step.
//!
//! The warm-up: the policy refills the returned rows at up to 8 MiB of copies a token (`policy::UPLOAD_CAP`). Staging
//! lends about 7 slots a layer, about 0.5 GB for the real model, so the refill takes about 70 tokens; 96 covers it with
//! margin (`--warmup`). Without it the first timed steps run on the emptiest cache.
//!
//! The positions run past `--at` by `warmup + reps * (1 + the sum of the t's)`, 246 for the defaults. That is noise
//! against 2,048 and more; the 64 row is a short-context figure (its passes run at positions 64 to 310), not a
//! 64-position one.
//!
//! The record names counts, milliseconds and the build; no token or text goes into it.
use anyhow::{bail, ensure, Context, Result};
use engine::{Engine, Options, Session};
use gguf::Gguf;
use qwen36::dims::Dims;
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

/// One timed pass, as `summarize` takes it: the path (`"decode"` or `"prefill"`), the context it was taken at, its
/// token count `t`, its milliseconds and the host-miss reads it made (`Stats::misses`, counted per routed (token,
/// expert) pair served from host memory, so a `t`-token pass counts a host expert shared by its tokens `t` times).
pub type Sample = (String, usize, usize, f64, u64);

/// One row of the record: the passes of one (path, context, `t`). `misses_per_step` is the mean host reads per pass, as
/// `Sample` counts them: routes, not experts crossing PCIe.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CostRow {
    pub path: String,
    pub ctx: usize,
    pub t: usize,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub misses_per_step: f64,
}

/// The mean and minimum milliseconds and the mean misses of each (path, context, `t`), in the order each first
/// appears in `samples`.
pub fn summarize(samples: &[Sample]) -> Vec<CostRow> {
    // (row so far, passes, sum of ms, sum of misses)
    let mut acc: Vec<(CostRow, usize, f64, u64)> = Vec::new();
    for (path, ctx, t, ms, misses) in samples {
        match acc
            .iter_mut()
            .find(|(r, ..)| r.path == *path && r.ctx == *ctx && r.t == *t)
        {
            Some((r, n, sum, miss)) => {
                *n += 1;
                *sum += ms;
                *miss += misses;
                r.min_ms = r.min_ms.min(*ms);
            }
            None => acc.push((
                CostRow {
                    path: path.clone(),
                    ctx: *ctx,
                    t: *t,
                    mean_ms: 0.0,
                    min_ms: *ms,
                    misses_per_step: 0.0,
                },
                1,
                *ms,
                *misses,
            )),
        }
    }
    acc.into_iter()
        .map(|(mut r, n, sum, miss)| {
            r.mean_ms = sum / n as f64;
            r.misses_per_step = miss as f64 / n as f64;
            r
        })
        .collect()
}

/// The run's schedule, checked before the model loads.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub at: Vec<usize>,
    /// The `t >= 2` passes, ascending, once each. `t = 1` is the decode step, always timed.
    pub passes: Vec<usize>,
    pub reps: usize,
    /// Untimed decode steps after each fill.
    pub warmup: usize,
    /// How far the positions run past a context: `warmup + reps * (the positions a round moves the session)`.
    pub drift: usize,
    /// The `t`s of the verify steps (`--verify`), ascending, once each, `t = 1` included; empty without `--verify`.
    /// A verify step of each `t` keeps one position (`commit(1)`), and for `t >= 2` another keeps all `t`; the drift
    /// counts what they keep.
    pub verify: Vec<usize>,
    /// The `k`s of the MTP's draft passes (`--mtp`), ascending, once each: the `t`s from 1 to `MAX_VERIFY - 1`; empty
    /// without `--mtp`. Each follows an untimed pass and two untimed decodes, which the drift counts.
    pub mtp: Vec<usize>,
    /// The most positions of the session (counted from a context) the plan reads or moves to: the drift, or further if
    /// a step reads prompt tokens beyond where it leaves the session (a verify step reads `t` and keeps one).
    pub extent: usize,
}

/// `--warmup`'s default; see the module doc.
pub const WARMUP: usize = 96;

/// One round's steps as (positions the step moves the session, prompt tokens it reads): the decode, the prefill passes,
/// the verify steps that keep one position, then those that keep all `t` (`t >= 2`; at `t = 1` it is the same step).
fn round_steps(passes: &[usize], verify: &[usize], mtp: &[usize]) -> Vec<(usize, usize)> {
    let mut round = vec![(1, 1)];
    // Each MTP pass is preceded by two untimed decodes (its prelude), which read three prompt tokens in all: the first
    // pass's `x`, the second decode's token and the timed pass's `x`.
    round.extend(mtp.iter().map(|_| (2, 3)));
    round.extend(passes.iter().map(|&t| (t, t)));
    round.extend(verify.iter().map(|&t| (1, t)));
    round.extend(verify.iter().filter(|&&t| t >= 2).map(|&t| (t, t)));
    round
}

impl Plan {
    pub fn new(
        prompt_len: usize,
        at: &[usize],
        t: &[usize],
        reps: usize,
        warmup: usize,
    ) -> Result<Plan> {
        Plan::with_verify(prompt_len, at, t, reps, warmup, false)
    }

    /// `new`, and with `verify` also a verify step of each `t` (`--verify`): `Session::verify`, every row read,
    /// `commit(1)`. A verify is 1 to `MAX_VERIFY` tokens, and keeps one position, so each costs the drift one.
    pub fn with_verify(
        prompt_len: usize,
        at: &[usize],
        t: &[usize],
        reps: usize,
        warmup: usize,
        verify: bool,
    ) -> Result<Plan> {
        Plan::with_mtp(prompt_len, at, t, reps, warmup, verify, false)
    }

    /// `with_verify`, and with `mtp` also the MTP's draft pass at each `k` in `t` (`--mtp`): a draft is 1 to
    /// `MAX_VERIFY - 1` tokens.
    pub fn with_mtp(
        prompt_len: usize,
        at: &[usize],
        t: &[usize],
        reps: usize,
        warmup: usize,
        verify: bool,
        mtp: bool,
    ) -> Result<Plan> {
        if reps == 0 {
            bail!("--reps is 0");
        }
        if at.is_empty() || t.is_empty() {
            bail!("--at and --t each need at least one value");
        }
        if t.contains(&0) {
            bail!("--t has a 0: a pass is at least one token");
        }
        let mut passes: Vec<usize> = t.iter().copied().filter(|&t| t >= 2).collect();
        passes.sort_unstable();
        passes.dedup();
        let mut verify_ts: Vec<usize> = if verify { t.to_vec() } else { Vec::new() };
        verify_ts.sort_unstable();
        verify_ts.dedup();
        let most = engine::spec::MAX_VERIFY;
        if let Some(bad) = verify_ts.iter().find(|&&t| t > most) {
            bail!("--verify times steps of 1 to {most} tokens; --t has {bad}");
        }
        // A draft is 1 to `most - 1` tokens: the `t`s past that are verify lengths only.
        let mut mtp_ks: Vec<usize> = if mtp {
            t.iter().copied().filter(|&k| k < most).collect()
        } else {
            Vec::new()
        };
        mtp_ks.sort_unstable();
        mtp_ks.dedup();
        if mtp && mtp_ks.is_empty() {
            bail!(
                "--mtp times draft passes of 1 to {} tokens; --t has only {}",
                most - 1,
                t.iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
        let round = round_steps(&passes, &verify_ts, &mtp_ks);
        let per_round: usize = round.iter().map(|s| s.0).sum();
        let drift = warmup + reps * per_round;
        // The last round's furthest read: where a step starts, plus the prompt tokens it reads.
        let mut at_step = warmup + (reps - 1) * per_round;
        let mut extent = drift;
        for (kept, read) in &round {
            extent = extent.max(at_step + read);
            at_step += kept;
        }
        if at[0] == 0 {
            bail!("--at has a 0: a context needs at least one prompt token");
        }
        for w in at.windows(2) {
            if w[1] < w[0] + drift {
                bail!(
                    "--at must ascend by at least {drift} (the positions a context's passes use): {} then {}",
                    w[0],
                    w[1]
                );
            }
        }
        let end = at[at.len() - 1] + extent;
        if prompt_len < end {
            bail!(
                "the prompt is {prompt_len} tokens; the last context needs {end} (--at {} plus {extent})",
                at[at.len() - 1]
            );
        }
        Ok(Plan {
            at: at.to_vec(),
            passes,
            reps,
            warmup,
            drift,
            verify: verify_ts,
            mtp: mtp_ks,
            extent,
        })
    }

    /// The positions the plan uses, and the session must hold: a verify step is refused past the session's end.
    pub fn ctx(&self) -> usize {
        self.at[self.at.len() - 1] + self.extent
    }

    /// Refuses a session of `ctx` positions that the plan doesn't fit.
    pub fn check_ctx(&self, ctx: usize) -> Result<()> {
        if ctx < self.ctx() {
            bail!(
                "--ctx {ctx} is below the {} positions the plan uses",
                self.ctx()
            );
        }
        Ok(())
    }
}

/// What the timed passes run on: the GPU engine's session, or a fake in the tests.
trait Machine {
    fn pos(&self) -> usize;
    /// Cumulative host-miss reads.
    fn misses(&self) -> u64;
    /// One decode step.
    fn decode(&mut self, token: u32) -> Result<()>;
    /// A pass over `tokens` with every row's logits read back.
    fn pass(&mut self, tokens: &[u32]) -> Result<()>;
    /// A verify step over `tokens`: every row read back, then the first `keep` positions kept.
    fn verify(&mut self, tokens: &[u32], keep: usize) -> Result<()>;
    /// Prefills `tokens` untimed, to reach a context.
    fn fill(&mut self, tokens: &[u32]) -> Result<()>;
    /// The MTP's draft pass for `x` at `k` drafts; returns how many it made.
    fn mtp_draft(&mut self, x: u32, k: usize) -> Result<usize>;
}

/// Times `call` on `m`: its milliseconds and the miss reads it made, pushed to `out`.
fn timed<M: Machine>(
    m: &mut M,
    out: &mut Vec<Sample>,
    (path, ctx, t): (&str, usize, usize),
    call: impl FnOnce(&mut M) -> Result<()>,
) -> Result<()> {
    let m0 = m.misses();
    let t0 = Instant::now();
    call(m)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    out.push((path.to_string(), ctx, t, ms, m.misses() - m0));
    Ok(())
}

/// Runs the plan and returns every timed pass: after each fill, `warmup` untimed decode steps, then `reps` rounds of
/// one timed decode and one timed pass per `t`, in that order, so the `t`s share the cache and the clock.
fn measure(m: &mut impl Machine, prompt: &[u32], plan: &Plan) -> Result<Vec<Sample>> {
    let mut out = Vec::new();
    for &ctx in &plan.at {
        let pos = m.pos();
        if pos < ctx {
            m.fill(&prompt[pos..ctx])?;
        }
        for _ in 0..plan.warmup {
            let tok = prompt[m.pos()];
            m.decode(tok)?;
        }
        for _ in 0..plan.reps {
            let tok = prompt[m.pos()];
            timed(m, &mut out, ("decode", ctx, 1), |m| m.decode(tok))?;
            for &k in &plan.mtp {
                // The prelude: whatever is outstanding is caught up and `x`'s row written, then two decodes leave
                // exactly one position (the second's) without a row, whatever came before.
                let x = prompt[m.pos()];
                let n = m.mtp_draft(x, 1)?;
                ensure!(
                    n == 1,
                    "the MTP drafted {n} of 1 tokens before a timed pass"
                );
                for _ in 0..2 {
                    let tok = prompt[m.pos()];
                    m.decode(tok)?;
                }
                let x = prompt[m.pos()];
                timed(m, &mut out, ("mtp_draft", ctx, k), |m| {
                    let n = m.mtp_draft(x, k)?;
                    ensure!(
                        n == k,
                        "the MTP drafted {n} of {k} tokens, so its time is not a {k}-token pass's"
                    );
                    Ok(())
                })?;
            }
            for &t in &plan.passes {
                let p = m.pos();
                timed(m, &mut out, ("prefill", ctx, t), |m| {
                    m.pass(&prompt[p..p + t])
                })?;
            }
            for &t in &plan.verify {
                let p = m.pos();
                timed(m, &mut out, ("verify", ctx, t), |m| {
                    m.verify(&prompt[p..p + t], 1)
                })?;
            }
            // Keeping all `t` replays `t` tokens: the same step as above at t = 1.
            for &t in plan.verify.iter().filter(|&&t| t >= 2) {
                let p = m.pos();
                timed(m, &mut out, ("verify_all", ctx, t), |m| {
                    m.verify(&prompt[p..p + t], t)
                })?;
            }
        }
    }
    Ok(out)
}

/// The engine's session.
struct Gpu<'a> {
    e: &'a Engine,
    s: Session,
}

impl Machine for Gpu<'_> {
    fn pos(&self) -> usize {
        self.s.pos()
    }
    fn misses(&self) -> u64 {
        self.e.stats().misses
    }
    fn decode(&mut self, token: u32) -> Result<()> {
        self.s.decode(self.e, token)?;
        Ok(())
    }
    fn pass(&mut self, tokens: &[u32]) -> Result<()> {
        self.s
            .prefill_each(self.e, tokens, 0, |_, _| Ok::<(), engine::Error>(()))?;
        Ok(())
    }
    fn fill(&mut self, tokens: &[u32]) -> Result<()> {
        self.s.prefill(self.e, tokens)?;
        Ok(())
    }
    fn verify(&mut self, tokens: &[u32], keep: usize) -> Result<()> {
        self.s.verify(self.e, tokens)?;
        for i in 0..tokens.len() {
            self.s.row(self.e, i)?;
        }
        self.s.commit(self.e, keep)?;
        Ok(())
    }
    fn mtp_draft(&mut self, x: u32, k: usize) -> Result<usize> {
        Ok(self.s.mtp_draft(self.e, x, k)?.len())
    }
}

/// The `--json` file: made (with its directory) before the model loads, any earlier file at the path removed, written
/// under a temporary name and renamed into place by `finish`, so a failed run leaves no file at the path: no empty
/// one, and no old one for a later reader to take as a result.
pub struct Output {
    path: PathBuf,
    tmp: PathBuf,
    file: Option<fs::File>,
}

impl Output {
    pub fn create(path: &Path) -> Result<Output> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            fs::create_dir_all(dir)
                .with_context(|| format!("--json: can't make {}", dir.display()))?;
        }
        // A failed run leaves no file, and no earlier run's either: a stale record is taken for this run's.
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(e).with_context(|| format!("--json: can't replace {}", path.display()))
            }
        }
        let mut name = path.file_name().unwrap_or_default().to_os_string();
        name.push(".tmp");
        let tmp = path.with_file_name(name);
        let file = fs::File::create(&tmp)
            .with_context(|| format!("--json: can't write {}", path.display()))?;
        Ok(Output {
            path: path.to_path_buf(),
            tmp,
            file: Some(file),
        })
    }

    fn write(&mut self, v: &impl Serialize) -> Result<()> {
        let f = self.file.as_mut().expect("written before finish");
        serde_json::to_writer_pretty(f, v)
            .with_context(|| format!("writing {}", self.path.display()))
    }

    /// Puts the written file at its name.
    pub fn finish(mut self) -> Result<()> {
        drop(self.file.take());
        fs::rename(&self.tmp, &self.path).map_err(|e| {
            let _ = fs::remove_file(&self.tmp);
            anyhow::Error::new(e).context(format!("moving the record to {}", self.path.display()))
        })
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        // Only a finished output has given its file up by renaming it; every other end removes the temporary one.
        if self.file.take().is_some() {
            let _ = fs::remove_file(&self.tmp);
        }
    }
}

/// The record `--json` holds. `ctx` is the session's position count (the cache is sized for it), not a row's context.
#[derive(Serialize)]
struct Record<'a> {
    commit: &'a str,
    dirty: Option<bool>,
    device: &'a str,
    driver: &'a str,
    prefill_kernels: &'static str,
    ctx: usize,
    cache_slots: usize,
    lent_slots: usize,
    prompt_tokens: usize,
    reps: usize,
    warmup: usize,
    at: &'a [usize],
    t: Vec<usize>,
    /// The `t`s timed through `Session::verify` (rows `path: "verify"`); empty without `--verify`.
    verify: &'a [usize],
    /// The `k`s of the MTP's draft passes (`--mtp`); empty without it.
    mtp: &'a [usize],
    /// The draft head's rows the MTP loaded with (0 the full head); null without `--mtp`.
    mtp_head_rows: Option<usize>,
    records: &'a [CostRow],
}

/// The engine options for `plan`: speculation on exactly when it times verify steps or MTP passes (`Session::verify`
/// asserts it), and the MTP loaded exactly when it times draft passes. `mtp_head_rows` (`--mtp-head-rows`) replaces the
/// default config's draft head rows (0 is the full head); a value the config refuses fails here, before the load.
pub fn options(
    plan: &Plan,
    ctx: usize,
    device_uuid: Option<[u8; 16]>,
    mtp_head_rows: Option<usize>,
) -> Result<Options> {
    let mtp = !plan.mtp.is_empty();
    let spec = mtp
        .then(|| {
            let mut c = engine::spec::SpecConfig::default();
            if let Some(rows) = mtp_head_rows {
                c.mtp_head_rows = rows;
            }
            c.check()
                .map_err(|(key, why)| anyhow::anyhow!("--mtp-head-rows ({key}): {why}"))?;
            Ok::<_, anyhow::Error>(c)
        })
        .transpose()?;
    Ok(Options {
        device_uuid,
        ctx,
        speculation: !plan.verify.is_empty() || mtp,
        mtp: crate::generate::mtp_options(spec.as_ref(), ctx)?,
        ..Options::default()
    })
}

/// `artemis verify-cost`. `repo_hint` is the directory of the prompt file, where the record's commit is read.
/// `prompt` is the token file's ids; `opts` are `options(plan, ...)`, whose `ctx` is the session's size, which sizes
/// the expert cache (`Plan::check_ctx` was checked); `out` was made by the caller before this runs and is `finish`ed by
/// it after.
pub fn run(
    model: &Path,
    prompt: &[u32],
    plan: &Plan,
    opts: &Options,
    out: &mut Output,
    repo_hint: &Path,
) -> Result<()> {
    let ctx = opts.ctx;
    let head_rows = opts.mtp.as_ref().map(|m| m.head_rows);
    let g = Gguf::open(model).with_context(|| format!("opening {}", model.display()))?;
    let e = Engine::open(&g, &Dims::QWEN36, opts)?;
    let s = e.session(ctx)?;
    let (cache_slots, lent_slots) = (e.cache_slots(), e.stats().lent_slots);
    let build = Build::read(repo_hint);
    eprintln!(
        concat!(
            "loaded on {} (driver {}, build {}); prompt {} tokens; {}-position session, ",
            "{} expert slots per layer and {} lent, prefill on {} kernels; contexts {:?}, ",
            "passes of {:?} tokens, verify steps of {:?} tokens, {} warm-up steps, {} reps each"
        ),
        e.device_name(),
        e.driver(),
        build.label(),
        prompt.len(),
        ctx,
        cache_slots,
        lent_slots,
        e.prefill_kernels(),
        plan.at,
        plan.passes,
        plan.verify,
        plan.warmup,
        plan.reps
    );
    let samples = measure(&mut Gpu { e: &e, s }, prompt, plan)?;
    let rows = summarize(&samples);
    for r in &rows {
        eprintln!(
            "{:>8} ctx {:>6} t {:>2}: mean {:>8.3} ms, min {:>8.3} ms, {:>6.1} host reads (routes) per step",
            r.path, r.ctx, r.t, r.mean_ms, r.min_ms, r.misses_per_step
        );
    }
    let mut t = vec![1];
    t.extend(&plan.passes);
    out.write(&Record {
        commit: &build.commit,
        dirty: build.dirty,
        device: e.device_name(),
        driver: e.driver(),
        prefill_kernels: e.prefill_kernels(),
        ctx,
        cache_slots,
        lent_slots,
        prompt_tokens: prompt.len(),
        reps: plan.reps,
        warmup: plan.warmup,
        at: &plan.at,
        t,
        verify: &plan.verify,
        mtp: &plan.mtp,
        mtp_head_rows: head_rows,
        records: &rows,
    })
}

/// The checkout's commit and whether it has changes, read when the tool runs. Neither a build-time commit nor a build
/// path may go into the binary: either would change it with every commit or checkout, and the gate runs one binary
/// (`artemis_sha256`). The checkout is the one holding the prompt file the run reads,
/// else the process's current directory. Best effort: `unknown` and no `dirty` when git or a repo isn't there.
struct Build {
    commit: String,
    dirty: Option<bool>,
}

impl Build {
    /// Reads the commit of the repo around `hint` (a directory), else around the current directory.
    fn read(hint: &Path) -> Build {
        [hint, Path::new(".")]
            .into_iter()
            .find_map(Build::at)
            .unwrap_or(Build {
                commit: "unknown".to_string(),
                dirty: None,
            })
    }

    fn at(dir: &Path) -> Option<Build> {
        let git = |args: &[&str]| {
            let o = std::process::Command::new("git")
                .arg("-C")
                .arg(dir)
                .args(args)
                .output()
                .ok()?;
            o.status
                .success()
                .then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
        };
        let commit = git(&["rev-parse", "HEAD"]).filter(|c| !c.is_empty())?;
        Some(Build {
            commit,
            dirty: git(&["status", "--porcelain"]).map(|s| !s.is_empty()),
        })
    }

    fn label(&self) -> String {
        match self.dirty {
            Some(true) => format!("{} (dirty)", self.commit),
            _ => self.commit.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(path: &str, ctx: usize, t: usize, ms: f64, misses: u64) -> Sample {
        (path.to_string(), ctx, t, ms, misses)
    }

    #[test]
    fn summarize_takes_the_mean_and_min_per_path_ctx_and_t() {
        let rows = summarize(&[
            s("decode", 64, 1, 2.0, 10),
            s("prefill", 64, 2, 5.0, 0),
            s("decode", 64, 1, 4.0, 20),
            s("prefill", 64, 2, 3.0, 4),
            s("prefill", 64, 2, 4.0, 2),
            s("prefill", 2048, 2, 8.0, 1),
            s("prefill", 64, 3, 6.5, 0),
        ]);
        let want = |path: &str, ctx, t, mean_ms, min_ms, misses_per_step| CostRow {
            path: path.to_string(),
            ctx,
            t,
            mean_ms,
            min_ms,
            misses_per_step,
        };
        // Four groups, in the order each first appears.
        assert_eq!(
            rows,
            vec![
                want("decode", 64, 1, 3.0, 2.0, 15.0),
                want("prefill", 64, 2, 4.0, 3.0, 2.0),
                want("prefill", 2048, 2, 8.0, 8.0, 1.0),
                want("prefill", 64, 3, 6.5, 6.5, 0.0),
            ]
        );
        assert!(summarize(&[]).is_empty());
    }

    #[test]
    fn a_row_serializes_with_the_records_field_names() {
        let v = serde_json::to_value(&summarize(&[s("decode", 64, 1, 2.0, 3)])[0]).unwrap();
        assert_eq!(
            v,
            serde_json::json!({"path": "decode", "ctx": 64, "t": 1, "mean_ms": 2.0, "min_ms": 2.0,
                "misses_per_step": 3.0})
        );
    }

    #[test]
    fn the_plan_refuses_what_the_prompt_or_the_contexts_cannot_hold() {
        let t = [1, 2, 3, 4, 5];
        // 96 warm-up steps, then 10 reps of (1 + 2 + 3 + 4 + 5) positions.
        let p = Plan::new(30_000, &[64, 2048, 8192, 20_000], &t, 10, WARMUP).unwrap();
        assert_eq!((p.drift, p.ctx()), (246, 20_246));
        assert_eq!(p.passes, vec![2, 3, 4, 5]);
        // A duplicate or a 1 doesn't add passes.
        assert_eq!(
            Plan::new(1000, &[64], &[5, 1, 2, 2], 1, 0).unwrap().passes,
            vec![2, 5]
        );
        let err = |r: Result<Plan>| r.unwrap_err().to_string();
        assert!(err(Plan::new(20_245, &[64, 20_000], &t, 10, WARMUP)).contains("needs 20246"));
        assert!(
            err(Plan::new(30_000, &[64, 300], &t, 10, WARMUP)).contains("ascend by at least 246")
        );
        assert!(err(Plan::new(30_000, &[2048, 64], &t, 10, WARMUP)).contains("ascend"));
        assert!(err(Plan::new(30_000, &[0], &t, 10, WARMUP)).contains("--at has a 0"));
        assert!(err(Plan::new(30_000, &[64], &[0, 2], 10, WARMUP)).contains("--t has a 0"));
        assert!(err(Plan::new(30_000, &[64], &t, 0, WARMUP)).contains("--reps"));
        assert!(err(Plan::new(30_000, &[], &t, 1, WARMUP)).contains("at least one"));
        assert!(err(Plan::new(30_000, &[64], &[], 1, WARMUP)).contains("at least one"));
    }

    #[test]
    fn the_session_must_hold_the_plan() {
        let p = Plan::new(30_000, &[64, 20_000], &[1, 2], 10, WARMUP).unwrap();
        assert_eq!(p.ctx(), 20_000 + 96 + 30);
        assert!(p.check_ctx(65_536).is_ok());
        assert!(p.check_ctx(p.ctx()).is_ok());
        assert!(p
            .check_ctx(p.ctx() - 1)
            .unwrap_err()
            .to_string()
            .contains("--ctx 20125 is below the 20126"));
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("artemis-vc-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn the_json_is_made_with_its_directory_and_only_a_finished_run_leaves_it() {
        let dir = scratch("out");
        let path = dir.join("runs").join("m5a").join("cost.json");
        // A failed run: the directory is made, no file stays, empty or otherwise, and an earlier run's goes too.
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "old").unwrap();
        let out = Output::create(&path).unwrap();
        assert!(!path.exists());
        assert!(path.with_file_name("cost.json.tmp").exists());
        drop(out);
        assert!(!path.exists() && !path.with_file_name("cost.json.tmp").exists());
        // A finished run: the record, whole, under its name.
        let mut out = Output::create(&path).unwrap();
        out.write(&serde_json::json!({"a": 1})).unwrap();
        out.finish().unwrap();
        let v: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(v["a"], 1);
        assert!(!path.with_file_name("cost.json.tmp").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_record_names_its_build_and_cache() {
        let r = Record {
            commit: "abc",
            dirty: Some(false),
            device: "gpu",
            driver: "d",
            prefill_kernels: "coop",
            ctx: 65_536,
            cache_slots: 135,
            lent_slots: 7,
            prompt_tokens: 3,
            reps: 1,
            warmup: 0,
            at: &[1],
            t: vec![1],
            verify: &[],
            mtp: &[],
            mtp_head_rows: Some(0),
            records: &[],
        };
        let v = serde_json::to_value(&r).unwrap();
        for k in [
            "commit",
            "dirty",
            "device",
            "driver",
            "prefill_kernels",
            "ctx",
            "cache_slots",
            "lent_slots",
            "warmup",
            "mtp_head_rows",
        ] {
            assert!(v.get(k).is_some(), "{k}");
        }
        // Read at run time, best effort: a commit hash or `unknown`, never empty.
        let b = Build::read(Path::new(env!("CARGO_MANIFEST_DIR")));
        assert!(!b.commit.is_empty());
        assert_eq!(b.commit == "unknown", b.dirty.is_none());
    }

    /// A machine that records what the schedule asked of it.
    #[derive(Default)]
    struct Fake {
        pos: usize,
        /// The furthest prompt position any call read to (its start plus its length).
        reach: usize,
        calls: Vec<(char, usize, usize)>, // ('f'ill, 'd'ecode or 'p'ass), first token, length
        /// How many drafts short of the ask each `mtp_draft` falls.
        short: usize,
        /// The positions below this have an MTP row or are holes (a fill's); `x`'s row, kept by a draft, counts.
        rows_upto: usize,
        /// The positions each `mtp_draft` call had to catch up, in call order.
        catchups: Vec<usize>,
    }

    impl Machine for Fake {
        fn pos(&self) -> usize {
            self.pos
        }
        fn misses(&self) -> u64 {
            self.calls.len() as u64
        }
        fn decode(&mut self, token: u32) -> Result<()> {
            self.calls.push(('d', token as usize, 1));
            self.reach = self.reach.max(self.pos + 1);
            self.pos += 1;
            Ok(())
        }
        fn pass(&mut self, tokens: &[u32]) -> Result<()> {
            self.calls.push(('p', tokens[0] as usize, tokens.len()));
            self.reach = self.reach.max(self.pos + tokens.len());
            self.pos += tokens.len();
            Ok(())
        }
        fn fill(&mut self, tokens: &[u32]) -> Result<()> {
            self.calls.push(('f', tokens[0] as usize, tokens.len()));
            self.reach = self.reach.max(self.pos + tokens.len());
            self.pos += tokens.len();
            // A fill's positions are holes, never caught up.
            self.rows_upto = self.pos;
            Ok(())
        }
        fn mtp_draft(&mut self, x: u32, k: usize) -> Result<usize> {
            // 'm': a draft pass moves nothing; a fake that drafts fewer than asked says so through `short`.
            self.calls.push(('m', x as usize, k));
            self.reach = self.reach.max(self.pos + 1);
            // The positions committed since the last row, then `x`'s own row (a pass keeps it).
            self.catchups.push(self.pos.saturating_sub(self.rows_upto));
            self.rows_upto = self.rows_upto.max(self.pos + 1);
            Ok(k.saturating_sub(self.short))
        }
        fn verify(&mut self, tokens: &[u32], keep: usize) -> Result<()> {
            // 'v' keeps one position (`commit(1)`), 'w' keeps all of a longer step.
            let c = if keep == 1 { 'v' } else { 'w' };
            self.calls.push((c, tokens[0] as usize, tokens.len()));
            self.reach = self.reach.max(self.pos + tokens.len());
            self.pos += keep;
            Ok(())
        }
    }

    #[test]
    fn the_schedule_times_verify_steps_that_keep_one_and_that_keep_all() {
        let prompt: Vec<u32> = (0..300).collect();
        let plan = Plan::with_verify(prompt.len(), &[10], &[3, 1, 2, 2], 2, 0, true).unwrap();
        assert_eq!(plan.verify, vec![1, 2, 3]);
        assert_eq!(plan.passes, vec![2, 3]);
        // A round moves the session by 1 (decode) + 2 + 3 (passes) + 3 (three verify steps keeping one) + 2 + 3 (the
        // steps of 2 and 3 tokens that keep them all).
        assert_eq!(plan.drift, 2 * (1 + 5 + 3 + 5));
        let mut m = Fake::default();
        let samples = measure(&mut m, &prompt, &plan).unwrap();
        // Per round: decode at 10, the 2- and 3-token prefill passes, verify steps of 1, 2 and 3 tokens keeping one,
        // then the 2- and 3-token ones keeping all, each starting where the last left the session.
        assert_eq!(
            m.calls,
            vec![
                ('f', 0, 10),
                ('d', 10, 1),
                ('p', 11, 2),
                ('p', 13, 3),
                ('v', 16, 1),
                ('v', 17, 2),
                ('v', 18, 3),
                ('w', 19, 2),
                ('w', 21, 3),
                ('d', 24, 1),
                ('p', 25, 2),
                ('p', 27, 3),
                ('v', 30, 1),
                ('v', 31, 2),
                ('v', 32, 3),
                ('w', 33, 2),
                ('w', 35, 3),
            ]
        );
        let got: Vec<(&str, usize)> = samples.iter().map(|s| (s.0.as_str(), s.2)).collect();
        assert_eq!(
            got[..9],
            [
                ("decode", 1),
                ("prefill", 2),
                ("prefill", 3),
                ("verify", 1),
                ("verify", 2),
                ("verify", 3),
                ("verify_all", 2),
                ("verify_all", 3),
                ("decode", 1)
            ]
        );
        // Rows come out per (path, ctx, t): the verify rows beside the verify_all rows (not at t = 1) and the prefill rows.
        let rows = summarize(&samples);
        assert_eq!(rows.iter().filter(|r| r.path == "verify").count(), 3);
        assert_eq!(rows.iter().filter(|r| r.path == "verify_all").count(), 2);
        // The drift is exactly the positions the rounds moved the session.
        assert_eq!(m.pos, plan.at[0] + plan.drift);
    }

    #[test]
    fn the_schedule_times_the_mtp_pass_after_a_decode_that_primes_the_ring() {
        let prompt: Vec<u32> = (0..100).collect();
        let plan = Plan::with_mtp(prompt.len(), &[10], &[1, 3, 2, 2], 2, 0, false, true).unwrap();
        assert_eq!(plan.mtp, vec![1, 2, 3]);
        assert_eq!(plan.passes, vec![2, 3]);
        // A round moves 1 (the timed decode) + 3 * 2 (the two decodes before each MTP pass) + 2 + 3 (the prefill
        // passes).
        assert_eq!(plan.drift, 2 * (1 + 6 + 5));
        let mut m = Fake::default();
        let samples = measure(&mut m, &prompt, &plan).unwrap();
        assert_eq!(
            m.calls[..15],
            [
                ('f', 0, 10),
                ('d', 10, 1),
                // k = 1: the untimed pass, two decodes, the timed pass for the token after them.
                ('m', 11, 1),
                ('d', 11, 1),
                ('d', 12, 1),
                ('m', 13, 1),
                // k = 2 and 3, each with the same prelude.
                ('m', 13, 1),
                ('d', 13, 1),
                ('d', 14, 1),
                ('m', 15, 2),
                ('m', 15, 1),
                ('d', 15, 1),
                ('d', 16, 1),
                ('m', 17, 3),
                ('p', 17, 2),
            ]
        );
        // The calls alternate prelude, timed pass: every timed pass (the odd ones) caught up exactly one position,
        // whatever k it is and whatever the round did before it.
        assert_eq!(m.catchups.len(), 2 * 3 * 2);
        for (i, c) in m.catchups.iter().enumerate().filter(|(i, _)| i % 2 == 1) {
            assert_eq!(*c, 1, "timed pass {i}: {:?}", m.catchups);
        }
        let got: Vec<(&str, usize)> = samples.iter().map(|s| (s.0.as_str(), s.2)).collect();
        assert_eq!(
            got[..7],
            [
                ("decode", 1),
                ("mtp_draft", 1),
                ("mtp_draft", 2),
                ("mtp_draft", 3),
                ("prefill", 2),
                ("prefill", 3),
                ("decode", 1)
            ]
        );
        assert_eq!(m.pos, plan.at[0] + plan.drift);
        // A pass that drafts fewer tokens than asked is no k-token pass: the run stops.
        let mut short = Fake {
            short: 1,
            ..Fake::default()
        };
        let e = measure(&mut short, &prompt, &plan).unwrap_err().to_string();
        assert!(e.contains("drafted 0 of 1"), "{e}");
        // Without --mtp none is timed; a draft is 1 to 15 tokens, so a 16 in --t is a verify length only.
        let plain = Plan::new(1000, &[64], &[1, 2], 1, 0).unwrap();
        assert!(plain.mtp.is_empty());
        let wide = Plan::with_mtp(1000, &[64], &[1, 16], 1, 0, true, true).unwrap();
        assert_eq!((wide.mtp, wide.verify), (vec![1], vec![1, 16]));
        let e = Plan::with_mtp(1000, &[64], &[16], 1, 0, false, true)
            .unwrap_err()
            .to_string();
        assert!(e.contains("1 to 15 tokens") && e.contains("only 16"), "{e}");
        // The engine loads the MTP (and speculation, which it needs) exactly for a plan that times it.
        let o = options(&plan, 65_536, None, None).unwrap();
        assert!(o.speculation && o.mtp.is_some());
        let default_rows = o.mtp.unwrap().head_rows;
        let o = options(&plain, 65_536, None, None).unwrap();
        assert!(!o.speculation && o.mtp.is_none());
        let verifying = Plan::with_verify(1000, &[64], &[1, 2], 1, 0, true).unwrap();
        let o = options(&verifying, 65_536, None, None).unwrap();
        assert!(o.speculation && o.mtp.is_none());
        // A session under 512 positions can't hold the MTP.
        assert!(options(&plan, 256, None, None).is_err());
        // --mtp-head-rows sets the draft head's rows (0 the full head); a value the config refuses fails before the
        // load.
        let full = options(&plan, 65_536, None, Some(0)).unwrap();
        assert_eq!(full.mtp.unwrap().head_rows, 0);
        assert_ne!(default_rows, 0);
        let e = options(&plan, 65_536, None, Some(512))
            .unwrap_err()
            .to_string();
        assert!(e.contains("mtp_head_rows"), "{e}");
    }

    #[test]
    fn the_plan_holds_every_prompt_token_and_position_a_step_reads_before_the_load() {
        // A verify step reads `t` prompt tokens but may keep one: the prompt and the session must hold what it reads,
        // not only where it leaves the session. Each plan's prompt is the shortest it accepts; the schedule must run on
        // it (an out-of-range slice would panic) and reach exactly the plan's extent.
        for (at, t, reps, warmup) in [
            (vec![10usize], vec![1usize, 2, 3], 2usize, 0usize),
            (vec![10], vec![16], 1, 0),
            (vec![10, 100], vec![1], 3, 4),
            (vec![5, 200], vec![2, 5, 16], 2, 7),
        ] {
            let plan = Plan::with_verify(10_000, &at, &t, reps, warmup, true).unwrap();
            let need = plan.ctx();
            let short = Plan::with_verify(need - 1, &at, &t, reps, warmup, true);
            assert!(
                short
                    .unwrap_err()
                    .to_string()
                    .contains(&format!("needs {need}")),
                "{at:?} {t:?}"
            );
            let prompt: Vec<u32> = (0..need as u32).collect();
            assert!(Plan::with_verify(need, &at, &t, reps, warmup, true).is_ok());
            let mut m = Fake::default();
            measure(&mut m, &prompt, &plan).unwrap();
            assert_eq!(m.reach, need, "{at:?} {t:?}");
            // The session must hold them too.
            assert!(plan.check_ctx(need).is_ok() && plan.check_ctx(need - 1).is_err());
        }
        // The shortest prompt for the first plan: 10 + 2 * 14 (a round moves 14).
        assert!(Plan::with_verify(37, &[10], &[1, 2, 3], 2, 0, true).is_err());
        assert!(Plan::with_verify(38, &[10], &[1, 2, 3], 2, 0, true).is_ok());
    }

    #[test]
    fn a_plan_without_verify_times_none_and_a_verify_is_one_to_sixteen_tokens() {
        let plan = Plan::new(1000, &[64], &[1, 2], 1, 0).unwrap();
        assert!(plan.verify.is_empty());
        let mut m = Fake::default();
        let samples = measure(&mut m, &(0..1000).collect::<Vec<u32>>(), &plan).unwrap();
        assert!(samples.iter().all(|s| s.0 != "verify") && m.calls.iter().all(|c| c.0 != 'v'));
        let verifying = |t: &[usize]| Plan::with_verify(1000, &[64], t, 1, 0, true);
        let e = verifying(&[1, 17]).unwrap_err().to_string();
        assert!(e.contains("1 to 16 tokens") && e.contains("has 17"), "{e}");
        assert_eq!(verifying(&[1, 16]).unwrap().verify, vec![1, 16]);
        // Without --verify a long prefill pass is fine.
        assert!(Plan::new(1000, &[64], &[1, 17], 1, 0).is_ok());
    }

    #[test]
    fn the_schedule_warms_up_after_each_fill_then_interleaves_the_ts_round_by_round() {
        // The prompt's token i is i, so a call's first token is its position.
        let prompt: Vec<u32> = (0..200).collect();
        let plan = Plan::new(prompt.len(), &[10, 100], &[1, 2, 3], 2, 3).unwrap();
        assert_eq!(plan.drift, 3 + 2 * (1 + 2 + 3));
        let mut m = Fake::default();
        let samples = measure(&mut m, &prompt, &plan).unwrap();
        // Context 10: fill 0..10, three untimed decodes at 10, 11, 12; then two rounds of decode, 2-token pass and
        // 3-token pass from 13, 19. Context 100: fill 25..100 and the same again.
        assert_eq!(
            m.calls,
            vec![
                ('f', 0, 10),
                ('d', 10, 1),
                ('d', 11, 1),
                ('d', 12, 1),
                ('d', 13, 1),
                ('p', 14, 2),
                ('p', 16, 3),
                ('d', 19, 1),
                ('p', 20, 2),
                ('p', 22, 3),
                ('f', 25, 75),
                ('d', 100, 1),
                ('d', 101, 1),
                ('d', 102, 1),
                ('d', 103, 1),
                ('p', 104, 2),
                ('p', 106, 3),
                ('d', 109, 1),
                ('p', 110, 2),
                ('p', 112, 3),
            ]
        );
        // Only the rounds are timed, in the same order: 2 contexts x 2 rounds x 3 lengths.
        let got: Vec<(&str, usize, usize)> = samples
            .iter()
            .map(|(p, c, t, _, _)| (p.as_str(), *c, *t))
            .collect();
        assert_eq!(got.len(), 12);
        assert_eq!(
            got[..6],
            [
                ("decode", 10, 1),
                ("prefill", 10, 2),
                ("prefill", 10, 3),
                ("decode", 10, 1),
                ("prefill", 10, 2),
                ("prefill", 10, 3)
            ]
        );
        assert_eq!(got[6], ("decode", 100, 1));
        // Each sample's misses are the delta across its own call (the fake adds one call, so 1).
        assert!(samples.iter().all(|s| s.4 == 1));
        let rows = summarize(&samples);
        assert_eq!(rows.len(), 6);
    }

    #[test]
    fn the_commit_comes_from_the_hint_or_the_current_directory() {
        // A directory that doesn't exist falls back to the current one (the tests run inside the repo).
        let here = Build::read(Path::new("."));
        let missing = Build::read(Path::new("no/such/dir/for/verify-cost"));
        assert_eq!(missing.commit, here.commit);
        assert_eq!(missing.dirty.is_none(), here.dirty.is_none());
    }

    /// No build path goes into the binary: `env!("CARGO_MANIFEST_DIR")` appears in tests only. Each file's tests
    /// start at its `#[cfg(test)]`, the last item of every file in the crate that has any.
    #[test]
    fn the_tools_source_has_no_build_path_outside_tests() {
        fn walk(dir: &Path, hits: &mut Vec<String>) {
            for e in fs::read_dir(dir).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    walk(&p, hits);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    let src = fs::read_to_string(&p).unwrap();
                    let code = src.split("#[cfg(test)]").next().unwrap();
                    if code.contains(concat!("env!(\"CARGO", "_MANIFEST_DIR\")")) {
                        hits.push(p.display().to_string());
                    }
                }
            }
        }
        let mut hits = Vec::new();
        walk(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
            &mut hits,
        );
        assert!(hits.is_empty(), "build path compiled into: {hits:?}");
    }
}
