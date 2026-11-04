//! `artemis generate`'s loop and its `--json` record, apart from the engine so CI tests them. Tests 3 and 5 read the
//! record.
use anyhow::{bail, Result};
use engine::spec::{
    Drafter, Next, Source as SpecSource, SourceChoice, SpecConfig, SpecDelta, SpecState, Verifier,
};
use engine::{Engine, Sampling};
use serde::{Deserialize, Serialize};

/// The record's format name. Version 2 runs the prompt through prefill: it adds `settings.all_logits`,
/// `cache.steps` and `cache.longest_submit_ms` (which reads as 0 when absent), and `positions` holds the prompt's
/// positions only with `--all-logits`.
pub const FORMAT: &str = "artemis-generate-2";

/// `--force-swap-every-token`'s seed: fixed, independent of `--seed`, so a forced run repeats exactly.
pub const FORCE_SWAP_SEED: u64 = 1;

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Stop {
    MaxTokens,
    EndOfGeneration,
    ContextFull,
}

impl Stop {
    pub fn describe(self) -> &'static str {
        match self {
            Stop::MaxTokens => "--max-tokens",
            Stop::EndOfGeneration => "an end-of-generation token",
            Stop::ContextFull => "a full context",
        }
    }
}

/// One generated step: the token chosen, and the two highest logits it was chosen from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub token: u32,
    pub top: u32,
    pub logit: f32,
    pub second: u32,
    pub second_logit: f32,
}

/// The two highest logits as (id, logit, id, logit), ties to the lower id. The oracle's greedy mode records the same.
/// Needs at least two logits.
pub fn top_two(logits: &[f32]) -> (u32, f32, u32, f32) {
    assert!(logits.len() >= 2, "top_two needs at least two logits");
    let (mut best, mut second) = (0usize, usize::MAX);
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > logits[best] {
            second = best;
            best = i;
        } else if second == usize::MAX || v > logits[second] {
            second = i;
        }
    }
    (best as u32, logits[best], second as u32, logits[second])
}

/// FNV-1a 64 of the logits' little-endian bytes, as 16 hex digits: test 5's fingerprint of a position.
pub fn fingerprint(logits: &[f32]) -> String {
    // The same FNV-1a 64 as `qwen36::testing::fnv1a64`, fed four bytes a logit, so no copy of the logits is made.
    let hash = logits
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .fold(0xCBF2_9CE4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x0100_0000_01B3)
        });
    format!("{hash:016x}")
}

/// The fingerprints of the decoded positions, kept only when something will record them (`--json`). Hashing the
/// 248,320 logits costs about a millisecond a token, inside the timed region, so a run without a record skips it.
pub struct Fingerprints {
    on: bool,
    list: Vec<String>,
}

impl Fingerprints {
    pub fn new(on: bool, capacity: usize) -> Self {
        Fingerprints {
            on,
            list: Vec::with_capacity(if on { capacity } else { 0 }),
        }
    }

    /// Fingerprints `logits` if enabled; otherwise does nothing.
    pub fn add(&mut self, logits: &[f32]) {
        if self.on {
            self.list.push(fingerprint(logits));
        }
    }

    pub fn into_list(self) -> Vec<String> {
        self.list
    }
}

/// Reads a `generate --json` record, refusing one of another format by naming the file and the format found.
pub fn read_record(path: &std::path::Path) -> Result<Record> {
    use anyhow::Context;
    let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_slice(&b).with_context(|| format!("parsing {}", path.display()))?;
    match v.get("format").and_then(|f| f.as_str()) {
        Some(FORMAT) => {}
        found => bail!(
            "{}: the record's format is {}, not \"{FORMAT}\"",
            path.display(),
            found.map_or("missing".to_string(), |f| format!("\"{f}\""))
        ),
    }
    serde_json::from_value(v).with_context(|| format!("parsing {}", path.display()))
}

/// The load line's slot count: a clamped `--cache-slots N` says what was asked for.
pub fn slots_note(asked: Option<usize>, got: usize) -> String {
    match asked {
        Some(n) if got < n => format!("{got} (asked for {n}; {got} fit)"),
        _ => got.to_string(),
    }
}

/// The load line's tail when a split-f16 kernel failed to build on a device that lists their shape
/// (`Engine::coop_error`), so a run on the f32 kernels says why; empty otherwise, leaving the line as it was.
pub fn coop_note(error: Option<&str>) -> String {
    error
        .map(|e| format!(", split-f16 kernels unavailable: {e}"))
        .unwrap_or_default()
}

/// The ids of a little-endian u32 token file.
pub fn read_tokens(path: &std::path::Path) -> Result<Vec<u32>> {
    let b = std::fs::read(path).map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
    if !b.len().is_multiple_of(4) {
        bail!("{} is not a whole number of u32 token ids", path.display());
    }
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

/// The MTP's load options for a tool's own load: `Options::mtp` for a run speculating with `spec`, `None`
/// for one that doesn't speculate or has no `mtp` source. A session under 512 positions can't hold MTP's ring and slide.
pub fn mtp_options(spec: Option<&SpecConfig>, ctx: usize) -> Result<Option<engine::MtpOptions>> {
    match spec {
        None => Ok(None),
        Some(c) => Ok(engine::MtpOptions::for_session(c, ctx)?),
    }
}

/// `mtp_options` for a command that takes the speculation flags: a refusal names the way out.
pub fn mtp_options_for_flags(
    spec: Option<&SpecConfig>,
    ctx: usize,
) -> Result<Option<engine::MtpOptions>> {
    mtp_options(spec, ctx)
        .map_err(|e| anyhow::anyhow!("{e} (--sources lookup speculates without it)"))
}

/// The submit trace as `artemis generate` writes it (`ARTEMIS_SUBMIT_TRACE=FILE`): `{"submits": [...]}`, one object a
/// traced compute submit in order, with its label's name (`decode`, `prefill`, `verify`, `mtp`, `replay`, `other`) and
/// its host and GPU times in ns on one clock (`vk::SubmitTrace`). `tools/py/submit_gaps.py` reads it.
pub fn submit_trace_json(t: &[vk::SubmitTrace]) -> serde_json::Value {
    let name = |l: u32| match l {
        1 => "decode",
        2 => "prefill",
        3 => "verify",
        4 => "mtp",
        engine::TRACE_REPLAY => "replay",
        _ => "other",
    };
    serde_json::json!({
        "submits": t
            .iter()
            .map(|s| serde_json::json!({
                "kind": name(s.label),
                "record_ns": s.record_ns,
                "submit_ns": s.submit_ns,
                "submitted_ns": s.submitted_ns,
                "woke_ns": s.woke_ns,
                "gpu_start_ns": s.gpu_start_ns,
                "gpu_end_ns": s.gpu_end_ns,
                "cal_err_ns": s.cal_err_ns,
            }))
            .collect::<Vec<_>>()
    })
}

/// The profile line for the MTP's draft passes (and its window rows, which a prefill chunk's time carries under their own
/// tag): empty without them.
pub fn mtp_note(steps: usize, tokens: usize, gpu_us: f64) -> String {
    if steps == 0 {
        return String::new();
    }
    format!(
        "; MTP {:.2} ms of GPU time a draft pass over {steps} passes ({tokens} draft rows)",
        gpu_us / 1e3 / steps as f64
    )
}

/// The load line's note when the load cut the MTP's ring to the prefill chunk (`Engine::mtp_ring_clamp`).
pub fn mtp_ring_note(clamp: Option<(usize, usize)>) -> String {
    match clamp {
        Some((asked, used)) => format!(
            "; the MTP ring cut from {asked} to {used} rows (the prefill chunk minus 16, a multiple of 16)"
        ),
        None => String::new(),
    }
}

/// The load line's note on the MTP block's VRAM slots (K-64): empty when the load gave none.
pub fn mtp_slots_note(st: &engine::Stats) -> String {
    match st.mtp_slots {
        0 => String::new(),
        n => format!("; {n} MTP expert slots ({} MiB)", st.mtp_slot_bytes >> 20),
    }
}

/// The profile line's verify clause: empty without verify steps, else the verify steps' GPU time a step.
pub fn verify_note(steps: usize, tokens: usize, gpu_us: f64) -> String {
    if steps == 0 {
        return String::new();
    }
    format!(
        "; verify {:.2} ms of GPU time a step over {steps} steps ({tokens} positions)",
        gpu_us / 1e3 / steps as f64
    )
}

/// The settings a run used.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    pub greedy: bool,
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub seed: u64,
    pub max_tokens: usize,
    pub ctx: usize,
    /// Expert slots per layer, after clamping to what fits.
    pub cache_slots: usize,
    pub force_swap_every_token: bool,
    /// Prefill computed every prompt position's logits (`--all-logits`), not only the last's.
    pub all_logits: bool,
    /// `--cache-slots` as given (None: the default cache). Absent from older records, which read as None.
    #[serde(default)]
    pub cache_slots_asked: Option<usize>,
    /// Whether `positions` holds the logits' fingerprints; false with `--no-fingerprints`. Older records read as true.
    #[serde(default = "yes")]
    pub fingerprints: bool,
    /// Whether the run was profiled (`--profile`): timestamp queries slow a step, so such a record is not a timed
    /// one. Absent from older records, which read as false.
    #[serde(default)]
    pub profiled: bool,
    /// `--f32-prefill`: prefill ran today's f32 kernels even where the split-f16 ones exist.
    /// Absent from older records, which read as false.
    #[serde(default)]
    pub f32_prefill: bool,
    /// `--no-staging`: no staging buffers. Absent from older records, which read as false.
    #[serde(default)]
    pub no_staging: bool,
    /// `--stage-from` as given (None: the measured threshold). Absent from older records, which read as None.
    #[serde(default)]
    pub stage_from: Option<usize>,
    /// The cache policy that ran: "adaptive", or "static" (`--cache-policy static`, or a
    /// forced swap a step). Records written before the adaptive policy existed ran a static cache and read as "static".
    #[serde(default = "static_policy")]
    pub cache_policy: String,
    /// `--upload-cap` as given, in MiB of copies per token (None: the engine's `UPLOAD_CAP`). Absent from older
    /// records, which read as None.
    #[serde(default)]
    pub upload_cap_mib: Option<u64>,
    /// `--no-lend`. Absent from older records, which read as false.
    #[serde(default)]
    pub no_lend: bool,
    /// `--base-time`: decode read a resident expert in place of every HOST one, for the base
    /// time, so its output is meaningless. Absent from older records, which read as false.
    #[serde(default)]
    pub base_time: bool,
    /// `--cpu-share`: the share of HOST misses the CPU served, 0.0 to 1.0. Absent from older records, which
    /// read as 0.0. The speed and hit-rate gates refuse a record whose share is not 0.
    #[serde(default)]
    pub cpu_share: f32,
    /// The speculation settings the run used; `None` is speculation off. Absent from older
    /// records, which read as off.
    #[serde(default)]
    pub speculation: Option<SpecSettings>,
}

/// The settings of a run that speculated: `SpecConfig`'s keys, and `--draft-len` (`None`: the controller chose how
/// many drafts a step verified).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpecSettings {
    pub max_draft: usize,
    pub ngram_min: usize,
    pub ngram_max: usize,
    pub min_gain: f64,
    pub accept_half_life: f64,
    pub accept_prior: f64,
    pub cost_slope_ms: f64,
    pub draft_len: Option<usize>,
    /// The draft sources that ran, by name (`source_choice` says how a step picked between them). Absent from records
    /// written before MTP, which read as lookup alone, and so do the keys below as zeros: such a run had no MTP.
    #[serde(default = "lookup_only")]
    pub sources: Vec<String>,
    #[serde(default)]
    pub probe_every: usize,
    #[serde(default)]
    pub mtp_window: usize,
    #[serde(default)]
    pub mtp_ctx: usize,
    #[serde(default)]
    pub mtp_ring: usize,
    #[serde(default)]
    pub mtp_accept_prior: Vec<f64>,
    #[serde(default)]
    pub mtp_cost_ms: f64,
    /// `priority` or `best`. Absent from older records, which read as `priority`.
    #[serde(default = "priority")]
    pub source_choice: String,
    /// 0 for MTP's full head (also what an older record reads as).
    #[serde(default)]
    pub mtp_head_rows: usize,
}

fn lookup_only() -> Vec<String> {
    vec!["lookup".into()]
}

fn priority() -> String {
    "priority".into()
}

impl From<&SpecConfig> for SpecSettings {
    fn from(c: &SpecConfig) -> SpecSettings {
        SpecSettings {
            max_draft: c.max_draft,
            ngram_min: c.ngram_min,
            ngram_max: c.ngram_max,
            min_gain: c.min_gain,
            accept_half_life: c.accept_half_life,
            accept_prior: c.accept_prior,
            cost_slope_ms: c.cost_slope_ms,
            draft_len: c.fixed_draft,
            sources: c.sources.iter().map(|s| s.name().to_string()).collect(),
            probe_every: c.probe_every,
            mtp_window: c.mtp_window,
            mtp_ctx: c.mtp_ctx,
            mtp_ring: c.mtp_ring,
            mtp_accept_prior: c.mtp_accept_prior.clone(),
            mtp_cost_ms: c.mtp_cost_ms,
            source_choice: c.source_choice.name().to_string(),
            mtp_head_rows: c.mtp_head_rows,
        }
    }
}

/// What speculation did in a run, in counts: verify steps, steps at `k = 0`, draft tokens verified and kept, and
/// the verify steps that kept fewer than all their drafts; also each source's drafted and kept tokens
/// (`drafted` and `accepted` stay the sums), MTP's probes and, where the caller has the state's totals, MTP's counts
/// per draft depth. Absent keys read as 0 (an older record had no MTP).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SpecRecord {
    pub steps: u64,
    pub plain_steps: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub replays: u64,
    #[serde(default)]
    pub drafted_lookup: u64,
    #[serde(default)]
    pub accepted_lookup: u64,
    #[serde(default)]
    pub drafted_mtp: u64,
    #[serde(default)]
    pub accepted_mtp: u64,
    #[serde(default)]
    pub probes: u64,
    /// Per MTP draft depth (the first entry is depth 1): `[drafts tested, drafts kept]`, to the last depth tested.
    /// Sums, so runs pool into measured rates. Filled by `spec-check` (`with_depths`); empty elsewhere.
    #[serde(default)]
    pub mtp_depth_counts: Vec<[u64; 2]>,
}

impl SpecRecord {
    /// The record with MTP's per-depth counts of the steps between two readings of a state's `mtp_depth_counts`.
    pub fn with_depths(mut self, before: &[(u64, u64)], after: &[(u64, u64)]) -> SpecRecord {
        let mut counts: Vec<[u64; 2]> = after
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let b = before.get(i).copied().unwrap_or((0, 0));
                [a.0.saturating_sub(b.0), a.1.saturating_sub(b.1)]
            })
            .collect();
        while counts.last() == Some(&[0, 0]) {
            counts.pop();
        }
        self.mtp_depth_counts = counts;
        self
    }
}

impl From<SpecDelta> for SpecRecord {
    fn from(d: SpecDelta) -> SpecRecord {
        SpecRecord {
            steps: d.steps,
            plain_steps: d.plain_steps,
            drafted: d.drafted,
            accepted: d.accepted,
            replays: d.replays,
            drafted_lookup: d.drafted_lookup,
            accepted_lookup: d.accepted_lookup,
            drafted_mtp: d.drafted_mtp,
            accepted_mtp: d.accepted_mtp,
            probes: d.probes,
            mtp_depth_counts: Vec::new(),
        }
    }
}

fn yes() -> bool {
    true
}

fn static_policy() -> String {
    "static".into()
}

/// The VRAM heap at load, in MiB: its size, the budget the driver gave this process and its
/// usage when the engine began loading, then once the model and session were in. The heap's size minus the budget at
/// open is what the driver held back from this process's budget at load (on this driver a fixed reserve, not the
/// desktop's use, which the runner samples).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vram {
    pub heap_mib: f64,
    pub budget_at_open_mib: f64,
    pub usage_at_open_mib: f64,
    pub budget_mib: f64,
    pub usage_mib: f64,
}

/// The host memory and the MTP's share of the load: what the 32 GB check and the gate read.
/// Absent from older records, which read as zeros.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    /// The embedding and every expert copy in host memory: `blk.40`'s experts included when the MTP is loaded.
    pub host_bytes: u64,
    /// The host-cached heap's usage after the load (`heap_budget(HostCached).usage`); null when the driver
    /// wouldn't say, which no check may read as 0 used.
    pub host_heap_usage: Option<u64>,
    /// The VRAM the MTP takes: `blk.40`'s weights and the session's rows and ring; 0 without it.
    pub mtp_vram_bytes: u64,
    /// The snapshot pool a 32 GB machine (26 GiB free before the load) would get with 8 GiB asked, in bytes
    /// (`engine::pool_on_machine(8, 26 GiB, host_heap_usage)`).
    /// Null when the heap's usage is unknown.
    pub pool_at_26gib: Option<u64>,
}

impl Memory {
    /// What `e` holds right after its load (sampled by the caller then, not when the record is built).
    pub fn of(e: &Engine) -> Memory {
        Memory::new(e.host_bytes(), e.host_heap_usage().ok(), e.mtp_vram_bytes())
    }

    /// The block from its parts. An unreadable heap (`None`) gives no usage and no pool figure: it can't pass the
    /// 32 GB check as an empty heap would.
    pub fn new(host_bytes: u64, host_heap_usage: Option<u64>, mtp_vram_bytes: u64) -> Memory {
        Memory {
            host_bytes,
            host_heap_usage,
            mtp_vram_bytes,
            pool_at_26gib: host_heap_usage.map(|u| engine::pool_on_machine(8, 26 << 30, u).bytes),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Timings {
    pub load_s: f64,
    pub prompt_s: f64,
    pub generate_s: f64,
    /// Positions decoded while generating (the prompt's aren't counted).
    pub decoded: usize,
}

/// How the run prefilled: reported, as the engine ran it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PrefillRun {
    /// "split-f16" or "f32" (`Engine::prefill_kernels`); empty in records written before staging existed.
    pub kernels: String,
    /// Whether the engine had staging buffers.
    pub staging: bool,
    /// Chunks of at least this many tokens were staged.
    pub stage_from: usize,
    /// The staging area's VRAM.
    pub staging_mib: f64,
    /// Prefill chunks that were staged.
    pub staged_steps: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CacheStats {
    /// The slots per layer the cache loaded with in chunks of its own. The lent rows are filled at load too, and
    /// `lent_slots` reports them.
    pub slots_per_layer: usize,
    /// The MiB the cache's own chunks hold when the record is read (`slots_per_layer` is the size at load).
    pub mib: f64,
    pub hits: u64,
    pub misses: u64,
    pub swaps_started: u64,
    pub swaps_done: u64,
    /// Steps run: each prefill chunk of the prompt, then each decoded position.
    pub steps: u64,
    /// The run's longest single submit in milliseconds (`Stats::longest_submit_us`), the watchdog margin's
    /// measure. Absent from records written before it was added, which read as 0.
    #[serde(default)]
    pub longest_submit_ms: f64,
    /// Steps prepared while a swap's copy was in flight (`Stats::steps_overlapping_copies`): a copy still outstanding
    /// when the step was prepared, not GPU overlap measured. Absent from older records, which read as 0.
    #[serde(default)]
    pub steps_overlapping_copies: u64,
    /// Routed-expert reads served from a staging buffer (`Stats::staged`); `hits` and `misses` don't count them. Absent
    /// from older records, which read as 0.
    #[serde(default)]
    pub staged: u64,
    /// Decode's reads alone, from the cache and from host memory (decode is never staged): generation's hit rate.
    /// Absent from older records, which read as 0.
    #[serde(default)]
    pub decode_hits: u64,
    #[serde(default)]
    pub decode_misses: u64,
    /// Rows the staging area lent the cache (`Stats::lent_slots`), fills published (`Stats::fills`), and the budget
    /// checks' shrinks, grows, refused grows and shortfalls. Absent from older records,
    /// which read as 0.
    #[serde(default)]
    pub lent_slots: usize,
    #[serde(default)]
    pub fills: u64,
    #[serde(default)]
    pub shrinks: u64,
    #[serde(default)]
    pub grows: u64,
    #[serde(default)]
    pub grows_refused: u64,
    #[serde(default)]
    pub budget_short: u64,
    /// The CPU miss path's decode tokens whose wait hit its bound, and the routed experts the CPU computed
    /// (`Stats::cpu_timeouts`, `cpu_misses`). A CPU miss is a miss: neither enters the hit rate. Absent from
    /// older records, which read as 0.
    #[serde(default)]
    pub cpu_timeouts: u64,
    #[serde(default)]
    pub cpu_misses: u64,
    /// The MTP block's VRAM slots the load gave (K-64; 0 without the MTP), its routed-expert reads in draft passes
    /// served from them and from host memory, and the experts copied into them (`Stats::mtp_slots`, `mtp_hits`,
    /// `mtp_misses`, `mtp_fills`). Absent from older records, which read as 0.
    #[serde(default)]
    pub mtp_slots: usize,
    #[serde(default)]
    pub mtp_slot_hits: u64,
    #[serde(default)]
    pub mtp_slot_misses: u64,
    #[serde(default)]
    pub mtp_slot_fills: u64,
}

/// What `generate --json` writes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub format: String,
    pub prompt_tokens: Vec<u32>,
    pub settings: Settings,
    /// Every generated step, the end-of-generation token's included when generation stopped there.
    pub steps: Vec<Step>,
    pub stop: Stop,
    /// `fingerprint` of the logits of every position computed, in order: every prompt position with `--all-logits`,
    /// else the prompt's last; then every decoded position.
    pub positions: Vec<String>,
    pub timings: Timings,
    pub cache: CacheStats,
    /// The VRAM heap at load. Absent from older records, which read as zeros.
    #[serde(default)]
    pub vram: Vram,
    /// The load's host memory and the MTP's share. Absent from older records, which read as zeros.
    #[serde(default)]
    pub memory: Memory,
    /// How the prompt was prefilled. Absent from older records, which read as `PrefillRun::default()`.
    #[serde(default)]
    pub prefill: PrefillRun,
    /// How `artemis warm` got this run's state. Absent from every other record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warm: Option<WarmInfo>,
    /// What speculation did; `None` when the run didn't speculate. Absent from older records, which read as `None`.
    #[serde(default)]
    pub spec: Option<SpecRecord>,
}

/// What `artemis warm` did to reach the state a record's run decoded from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WarmInfo {
    /// 1: the snapshot was built by prefill at `cut`; 2: by decode, at `split`.
    pub path: u8,
    /// The golden prompt run, and the one that dirtied the engine between the snapshot and the restore.
    pub id: String,
    pub dirty_id: String,
    /// Path 1's cut: the prompt's tokens before the snapshot.
    pub cut: Option<usize>,
    /// Path 2's split: the judged steps before the snapshot.
    pub split: Option<usize>,
    pub snapshot_bytes: u64,
    pub restore_bytes: u64,
    /// The first position of the live KV the restore copied back.
    pub restore_kv_from: usize,
    pub restore_ms: f64,
    pub pool_limit_bytes: u64,
}

/// `--speculation on|off`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Switch {
    #[default]
    On,
    Off,
}

/// `artemis generate`'s speculation flags. Speculation is on by default, as the server's is.
/// Each flag after `--speculation` is a `[speculation]` key; the others are the config file's defaults. Any of them
/// with `--speculation off` is refused before the load, so a record never claims a setting that didn't run.
#[derive(Clone, Debug, Default, clap::Args)]
pub struct SpecFlags {
    /// Draft tokens from the request's own history and verify them in one decode-shaped pass (on), or decode one token
    /// a step as before (off).
    #[arg(long, value_enum, default_value_t = Switch::On)]
    pub speculation: Switch,
    /// The most draft tokens a step verifies, 1 to 15 (default 4).
    #[arg(long)]
    pub max_draft: Option<usize>,
    /// Prompt lookup's shortest n-gram (default 2).
    #[arg(long)]
    pub ngram_min: Option<usize>,
    /// Prompt lookup's longest n-gram, at most 16 (default 4).
    #[arg(long)]
    pub ngram_max: Option<usize>,
    /// The predicted speed-up below which a step runs plain decode, 1 or more (default 1.03).
    #[arg(long, allow_negative_numbers = true)]
    pub min_gain: Option<f64>,
    /// Draft tokens per halving of an old outcome's weight in the accept rate (default 32).
    #[arg(long, allow_negative_numbers = true)]
    pub accept_half_life: Option<f64>,
    /// The accept rate the controller starts from, 0 to 1 (default 0.5).
    #[arg(long, allow_negative_numbers = true)]
    pub accept_prior: Option<f64>,
    /// The controller's cost per extra token of a step in milliseconds, until it has measured its own (default 4.4).
    #[arg(long, allow_negative_numbers = true)]
    pub cost_slope_ms: Option<f64>,
    /// Bypass the controller: verify min(N, the drafts found, the room left) drafts every step, as test 5's runs do so
    /// that every cache size runs the same steps. Above --max-draft's default of 4 it lifts max-draft to N (as
    /// `lookup-sim --draft-len` does); with an explicit smaller --max-draft it is refused.
    #[arg(long)]
    pub draft_len: Option<usize>,
    /// The draft sources that run, comma separated: `lookup`, `mtp` or `lookup,mtp` (default `lookup,mtp`); their order
    /// does nothing, `--source-choice` picks between them.
    /// (`artemis serve` spells this `--speculation-sources`.)
    #[arg(long, value_delimiter = ',')]
    pub sources: Option<Vec<String>>,
    /// MTP drafts one token as a probe once this many plain steps have run since it last drafted, 1 or more (default
    /// 64).
    #[arg(long)]
    pub probe_every: Option<usize>,
    /// The last positions of the prompt that prefill writes into the MTP's KV, 1 to --mtp-ctx / 2 (default 2048).
    #[arg(long)]
    pub mtp_window: Option<usize>,
    /// The MTP KV's rows, 512 to 65536 and a multiple of 256 (default 8192).
    #[arg(long)]
    pub mtp_ctx: Option<usize>,
    /// The hidden-row ring's rows, 16 to --mtp-ctx / 4 and a multiple of 16 (default 256).
    #[arg(long)]
    pub mtp_ring: Option<usize>,
    /// MTP's accept rate at each draft depth until it has measured its own: 1 to 15 comma separated values from 0 to 1.
    #[arg(long, value_delimiter = ',', allow_negative_numbers = true)]
    pub mtp_accept_prior: Option<Vec<f64>>,
    /// The prior for the MTP pass's time per draft token in milliseconds, 0 or more (default 1.6).
    #[arg(long, allow_negative_numbers = true)]
    pub mtp_cost_ms: Option<f64>,
    /// Which source drafts when both have a draft: `priority` (lookup first) or `best` (the higher predicted tokens per
    /// ms) (default `best`).
    #[arg(long)]
    pub source_choice: Option<String>,
    /// Draft MTP tokens from the first N rows of the output head (a prefix view, no VRAM), 1024 or more; 0 is the full
    /// head (default 65536; a smaller vocabulary is the full head).
    #[arg(long)]
    pub mtp_head_rows: Option<usize>,
}

impl SpecFlags {
    /// The flags `--speculation off` refuses, by name, in the order they are declared.
    fn given(&self) -> [(&'static str, bool); 17] {
        [
            ("--max-draft", self.max_draft.is_some()),
            ("--ngram-min", self.ngram_min.is_some()),
            ("--ngram-max", self.ngram_max.is_some()),
            ("--min-gain", self.min_gain.is_some()),
            ("--accept-half-life", self.accept_half_life.is_some()),
            ("--accept-prior", self.accept_prior.is_some()),
            ("--cost-slope-ms", self.cost_slope_ms.is_some()),
            ("--draft-len", self.draft_len.is_some()),
            ("--sources", self.sources.is_some()),
            ("--probe-every", self.probe_every.is_some()),
            ("--mtp-window", self.mtp_window.is_some()),
            ("--mtp-ctx", self.mtp_ctx.is_some()),
            ("--mtp-ring", self.mtp_ring.is_some()),
            ("--mtp-accept-prior", self.mtp_accept_prior.is_some()),
            ("--mtp-cost-ms", self.mtp_cost_ms.is_some()),
            ("--source-choice", self.source_choice.is_some()),
            ("--mtp-head-rows", self.mtp_head_rows.is_some()),
        ]
    }

    /// The config the flags ask for, `None` for speculation off. Checked before the model loads: a speculation flag
    /// with `--speculation off` ("--draft-len needs --speculation on"), the CPU share or the base time with speculation
    /// on (the engine refuses them too, `Engine::open`; the share by `cpu_miss::path_on`, so one under 1/512 that runs
    /// no path passes), and every setting `SpecConfig::check` refuses, named by its flag.
    pub fn config(&self, cpu_share: f32, base_time: bool) -> Result<Option<SpecConfig>> {
        if self.speculation == Switch::Off {
            if let Some((flag, _)) = self.given().iter().find(|f| f.1) {
                bail!("{flag} needs --speculation on");
            }
            return Ok(None);
        }
        if engine::cpu_miss::path_on(cpu_share) {
            bail!("--cpu-share needs --speculation off");
        }
        if base_time {
            bail!("--base-time needs --speculation off");
        }
        let mut c = SpecConfig::default();
        let d = c.clone();
        c.max_draft = self.max_draft.unwrap_or(d.max_draft);
        c.ngram_min = self.ngram_min.unwrap_or(d.ngram_min);
        c.ngram_max = self.ngram_max.unwrap_or(d.ngram_max);
        c.min_gain = self.min_gain.unwrap_or(d.min_gain);
        c.accept_half_life = self.accept_half_life.unwrap_or(d.accept_half_life);
        c.accept_prior = self.accept_prior.unwrap_or(d.accept_prior);
        c.cost_slope_ms = self.cost_slope_ms.unwrap_or(d.cost_slope_ms);
        if let Some(names) = &self.sources {
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            c.sources = SpecSource::parse_list(&names)
                .map_err(|(_, why)| anyhow::anyhow!("--sources: {why}"))?;
        }
        c.probe_every = self.probe_every.unwrap_or(d.probe_every);
        c.mtp_window = self.mtp_window.unwrap_or(d.mtp_window);
        c.mtp_ctx = self.mtp_ctx.unwrap_or(d.mtp_ctx);
        c.mtp_ring = self.mtp_ring.unwrap_or(d.mtp_ring);
        c.mtp_accept_prior = self.mtp_accept_prior.clone().unwrap_or(d.mtp_accept_prior);
        c.mtp_cost_ms = self.mtp_cost_ms.unwrap_or(d.mtp_cost_ms);
        if let Some(name) = &self.source_choice {
            c.source_choice = SourceChoice::parse(name).ok_or_else(|| {
                anyhow::anyhow!(
                    "--source-choice: an unknown choice {name:?}, expected priority or best"
                )
            })?;
        }
        c.mtp_head_rows = self.mtp_head_rows.unwrap_or(d.mtp_head_rows);
        if let Some(n) = self.draft_len {
            let most = engine::spec::MAX_VERIFY - 1;
            if n > most {
                bail!("--draft-len must be at most {most}");
            }
            match self.max_draft {
                // The same lift as `lookup-sim --draft-len`: a fixed length above the default max_draft raises it.
                None => c.max_draft = c.max_draft.max(n),
                Some(m) if n > m => bail!("--draft-len {n} is above --max-draft {m}"),
                Some(_) => {}
            }
            c.fixed_draft = Some(n);
        }
        c.check().map_err(|(key, why)| {
            let flag = match key {
                "max_draft" => "--max-draft",
                "ngram_min" => "--ngram-min",
                "ngram_max" => "--ngram-max",
                "min_gain" => "--min-gain",
                "accept_half_life" => "--accept-half-life",
                "accept_prior" => "--accept-prior",
                "cost_slope_ms" => "--cost-slope-ms",
                "fixed_draft" => "--draft-len",
                "sources" => "--sources",
                "probe_every" => "--probe-every",
                "mtp_window" => "--mtp-window",
                "mtp_ctx" => "--mtp-ctx",
                "mtp_ring" => "--mtp-ring",
                "mtp_accept_prior" => "--mtp-accept-prior",
                "mtp_cost_ms" => "--mtp-cost-ms",
                "mtp_head_rows" => "--mtp-head-rows",
                // A key `check` learns later is named as it is, not blamed on a flag that isn't its own.
                other => other,
            };
            // The reasons name config keys; this is the command line.
            let why = why
                .replace("ngram_max", "--ngram-max")
                .replace("max_draft", "--max-draft")
                .replace("mtp_ctx", "--mtp-ctx");
            anyhow::anyhow!("{flag}: {why}")
        })?;
        Ok(Some(c))
    }
}

/// What a run was asked for: the command-line side of a record's `settings`.
pub struct Asked {
    pub greedy: bool,
    pub sampling: Sampling,
    pub seed: u64,
    pub max_tokens: usize,
    pub ctx: usize,
    pub cache_slots: Option<usize>,
    pub force_swap_every_token: bool,
    pub all_logits: bool,
    pub fingerprints: bool,
    pub profiled: bool,
    pub f32_prefill: bool,
    pub no_staging: bool,
    pub stage_from: Option<usize>,
    pub upload_cap_mib: Option<u64>,
    pub no_lend: bool,
    pub base_time: bool,
    pub cpu_share: f32,
    /// The config the run speculated with; `None` for a run that didn't.
    pub speculation: Option<SpecConfig>,
}

/// What a run did, for its record.
pub struct Ran {
    pub prompt: Vec<u32>,
    pub steps: Vec<Step>,
    pub stop: Stop,
    pub positions: Vec<String>,
    pub load_s: f64,
    pub prompt_s: f64,
    pub generate_s: f64,
    pub decoded: usize,
    /// The engine's stats at the end, and at the end of the prompt (decode's cache reads are the difference).
    pub stats: engine::Stats,
    pub after_prefill: engine::Stats,
    pub vram: Vram,
    /// The load's host memory, sampled right after it.
    pub memory: Memory,
    /// The steps' counts when the run speculated.
    pub spec: Option<SpecDelta>,
}

/// The record `generate --json`, `warm` and `spec-check` write: one builder, so they agree field for field.
pub fn record(e: &Engine, asked: &Asked, ran: Ran, warm: Option<WarmInfo>) -> Record {
    let Asked {
        greedy,
        sampling,
        seed,
        max_tokens,
        ctx,
        cache_slots,
        force_swap_every_token,
        all_logits,
        fingerprints,
        profiled,
        f32_prefill,
        no_staging,
        stage_from,
        upload_cap_mib,
        no_lend,
        base_time,
        cpu_share,
        speculation,
    } = asked;
    let Ran {
        prompt,
        steps,
        stop,
        positions,
        load_s,
        prompt_s,
        generate_s,
        decoded,
        stats: st,
        after_prefill,
        vram,
        memory,
        spec,
    } = ran;
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    Record {
        format: FORMAT.into(),
        prompt_tokens: prompt,
        settings: Settings {
            greedy: *greedy,
            temp: sampling.temp,
            top_k: sampling.top_k,
            top_p: sampling.top_p,
            min_p: sampling.min_p,
            presence_penalty: sampling.presence_penalty,
            seed: *seed,
            max_tokens: *max_tokens,
            ctx: *ctx,
            cache_slots: e.cache_slots(),
            force_swap_every_token: *force_swap_every_token,
            all_logits: *all_logits,
            cache_slots_asked: *cache_slots,
            fingerprints: *fingerprints,
            profiled: *profiled,
            f32_prefill: *f32_prefill,
            no_staging: *no_staging,
            stage_from: *stage_from,
            cache_policy: if e.policy_adapts() {
                "adaptive"
            } else {
                "static"
            }
            .into(),
            upload_cap_mib: *upload_cap_mib,
            no_lend: *no_lend,
            base_time: *base_time,
            // What the engine ran: a share below 1/512 runs no path, so it is recorded as 0.
            cpu_share: if engine::cpu_miss::path_on(*cpu_share) {
                *cpu_share
            } else {
                0.0
            },
            speculation: speculation.as_ref().map(SpecSettings::from),
        },
        steps,
        stop,
        positions,
        timings: Timings {
            load_s,
            prompt_s,
            generate_s,
            decoded,
        },
        cache: CacheStats {
            slots_per_layer: st.slots_per_layer,
            mib: st.cache_bytes as f64 / (1u64 << 20) as f64,
            hits: st.hits,
            misses: st.misses,
            swaps_started: st.swaps_started,
            swaps_done: st.swaps_done,
            steps: st.steps,
            longest_submit_ms: st.longest_submit_us as f64 / 1e3,
            steps_overlapping_copies: st.steps_overlapping_copies,
            staged: st.staged,
            decode_hits: st.hits - after_prefill.hits,
            decode_misses: st.misses - after_prefill.misses,
            lent_slots: st.lent_slots,
            fills: st.fills,
            shrinks: st.shrinks,
            grows: st.grows,
            grows_refused: st.grows_refused,
            budget_short: st.budget_short,
            cpu_timeouts: st.cpu_timeouts,
            cpu_misses: st.cpu_misses,
            mtp_slots: st.mtp_slots,
            mtp_slot_hits: st.mtp_hits,
            mtp_slot_misses: st.mtp_misses,
            mtp_slot_fills: st.mtp_fills,
        },
        vram,
        memory,
        prefill: PrefillRun {
            kernels: e.prefill_kernels().into(),
            staging: e.staging_on(),
            stage_from: e.stage_from(),
            staging_mib: mib(st.staging_bytes),
            staged_steps: st.staged_steps,
        },
        warm,
        spec: spec.map(SpecRecord::from),
    }
}

/// How the generation loop gets from one position to the next: plain decode, or a speculative step.
/// `run` hands `step` the token `x` the state is one short of, and it runs from `x`, at most `most` positions
/// (`x` and the drafts it verifies), calling `pick` with each row it uses, in order, until `pick` says `Next::End` or the
/// rows run out. It returns the positions run, which is the number of rows it gave `pick`.
pub trait Stepper {
    /// Called once with the first token, chosen from the prompt's logits, before any step: a drafter's history ends
    /// with it. Nothing for a stepper that doesn't draft.
    fn begin(&mut self, _first: u32) {}

    /// Runs from `x`, at most `most` positions, calling `pick` with each row used in order; returns the positions run.
    fn step(
        &mut self,
        x: u32,
        most: usize,
        pick: &mut dyn FnMut(&[f32]) -> Result<Next>,
    ) -> Result<usize>;
}

/// Today's loop: one decode of `x` and one row. `F` decodes a token and returns the next logits.
pub struct Plain<F: FnMut(u32) -> Result<Vec<f32>>>(pub F);

impl<F: FnMut(u32) -> Result<Vec<f32>>> Stepper for Plain<F> {
    fn step(
        &mut self,
        x: u32,
        _most: usize,
        pick: &mut dyn FnMut(&[f32]) -> Result<Next>,
    ) -> Result<usize> {
        let row = (self.0)(x)?;
        pick(&row)?;
        Ok(1)
    }
}

/// Maps a verifier's errors into anyhow's, so `SessionVerifier` (`Error = engine::Error`) fits `Speculating`.
pub struct Anyhow<V>(pub V);

impl<V: Verifier> Verifier for Anyhow<V>
where
    V::Error: std::error::Error + Send + Sync + 'static,
{
    type Error = anyhow::Error;

    fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        self.0.decode(token).map_err(anyhow::Error::new)
    }

    fn verify(&mut self, tokens: &[u32]) -> Result<()> {
        self.0.verify(tokens).map_err(anyhow::Error::new)
    }

    fn row(&mut self, i: usize) -> Result<Vec<f32>> {
        self.0.row(i).map_err(anyhow::Error::new)
    }

    fn commit(&mut self, kept: usize) -> Result<()> {
        self.0.commit(kept).map_err(anyhow::Error::new)
    }

    fn n_vocab(&self) -> usize {
        self.0.n_vocab()
    }

    fn most(&self) -> usize {
        self.0.most()
    }

    fn mtp_draft(&mut self, x: u32, k: usize) -> Result<Vec<u32>> {
        self.0.mtp_draft(x, k).map_err(anyhow::Error::new)
    }

    fn has_mtp(&self) -> bool {
        self.0.has_mtp()
    }

    fn position(&self) -> Option<usize> {
        self.0.position()
    }

    fn misses(&self) -> u64 {
        self.0.misses()
    }

    fn miss_bytes(&self) -> u64 {
        self.0.miss_bytes()
    }
}

/// Speculation's loop: `SpecState::step` over a verifier, with this run's drafter. The drafter's history must hold the
/// prompt before the first step (`new` does that); `begin` adds the first token.
pub struct Speculating<'a, V: Verifier<Error = anyhow::Error>> {
    pub state: &'a mut SpecState,
    pub drafter: Box<dyn Drafter>,
    pub v: V,
    /// The counts of this run's steps.
    pub delta: SpecDelta,
}

impl<'a, V: Verifier<Error = anyhow::Error>> Speculating<'a, V> {
    /// A run's stepper: a fresh drafter that has seen `prompt`.
    pub fn new(state: &'a mut SpecState, v: V, prompt: &[u32]) -> Self {
        let mut drafter = state.drafter();
        drafter.observe(prompt);
        Speculating {
            state,
            drafter,
            v,
            delta: SpecDelta::default(),
        }
    }
}

impl<V: Verifier<Error = anyhow::Error>> Stepper for Speculating<'_, V> {
    fn begin(&mut self, first: u32) {
        self.drafter.observe(&[first]);
    }

    fn step(
        &mut self,
        x: u32,
        most: usize,
        pick: &mut dyn FnMut(&[f32]) -> Result<Next>,
    ) -> Result<usize> {
        // `x` takes one of the `most` positions: at most `most - 1` drafts.
        let k_max = most.saturating_sub(1);
        let stepped = self.state.step(
            &mut self.v,
            self.drafter.as_mut(),
            x,
            k_max,
            pick,
            &mut self.delta,
        )?;
        Ok(stepped.kept)
    }
}

/// One generation's state and per-position rules: what `run` does with each row, whoever ran it.
struct Loop<'a, P, E> {
    eog: &'a [u32],
    max_tokens: usize,
    /// Positions left in the context.
    room: usize,
    pick: P,
    emit: E,
    steps: Vec<Step>,
    out: Vec<u32>,
    stop: Option<Stop>,
}

impl<P: FnMut(&[f32], u32) -> u32, E: FnMut(&[u32]) -> Result<()>> Loop<'_, P, E> {
    /// Chooses the token from `row`'s logits, records the step and says whether the run goes on. One scan gives the
    /// step's top two, and the top is what a greedy pick takes.
    fn take(&mut self, row: &[f32]) -> Result<Next> {
        if self.out.len() >= self.max_tokens {
            self.stop = Some(Stop::MaxTokens);
            return Ok(Next::End);
        }
        let (top, logit, second, second_logit) = top_two(row);
        let token = (self.pick)(row, top);
        self.steps.push(Step {
            token,
            top,
            logit,
            second,
            second_logit,
        });
        if self.eog.contains(&token) {
            self.stop = Some(Stop::EndOfGeneration);
            return Ok(Next::End);
        }
        self.out.push(token);
        (self.emit)(&self.out)?;
        if self.out.len() >= self.max_tokens {
            self.stop = Some(Stop::MaxTokens);
            return Ok(Next::End);
        }
        if self.room == 0 {
            self.stop = Some(Stop::ContextFull);
            return Ok(Next::End);
        }
        Ok(Next::Go(token))
    }
}

/// The generation loop. `logits` are the prompt's last position's. `pick` chooses each token from its logits and their
/// top token (what `--greedy` takes, ties to the lower id); `stepper` runs from the last token and gives the loop the
/// next positions' logits, one row at a time: a plain decode, or a burst of a speculative step; `emit` gets the generated
/// tokens so far, never the end-of-generation token. It stops after `max_tokens` tokens (without running the last), at
/// an end-of-generation token (recorded as a step, not emitted), or once `room` more positions have been run. A burst
/// never runs past any of those: `stepper` is given at most `min(max_tokens - tokens so far, room)` positions, and
/// the loop's rules end it at the same token a plain run ends at, so the steps, the stop and the emitted tokens are
/// the plain run's whatever the stepper drafts.
pub fn run(
    logits: Vec<f32>,
    max_tokens: usize,
    room: usize,
    eog: &[u32],
    pick: impl FnMut(&[f32], u32) -> u32,
    stepper: &mut dyn Stepper,
    emit: impl FnMut(&[u32]) -> Result<()>,
) -> Result<(Vec<Step>, Stop)> {
    let mut l = Loop {
        eog,
        max_tokens,
        room,
        pick,
        emit,
        steps: Vec::new(),
        out: Vec::new(),
        stop: None,
    };
    let Next::Go(mut x) = l.take(&logits)? else {
        let stop = l.stop.expect("a loop that ends says why");
        return Ok((l.steps, stop));
    };
    stepper.begin(x);
    loop {
        // Both are above 0 here: the loop ends at a full count or a full context.
        let most = (l.max_tokens - l.out.len()).min(l.room);
        let mut delivered = 0usize;
        let ran = stepper.step(x, most, &mut |row| {
            // `End` is final: a row after it would add steps past the stop (and could outrun `room`).
            anyhow::ensure!(
                l.stop.is_none(),
                "the stepper gave the loop a row after the run ended"
            );
            delivered += 1;
            l.room -= 1;
            let next = l.take(row)?;
            if let Next::Go(t) = next {
                x = t;
            }
            Ok(next)
        })?;
        anyhow::ensure!(
            ran == delivered && (1..=most).contains(&ran),
            "the stepper reports {ran} positions run, gave the loop {delivered} rows, and was allowed {most}"
        );
        if let Some(stop) = l.stop {
            return Ok((l.steps, stop));
        }
    }
}

/// The sampling settings from the command line: `--greedy`, or the sampling flags over production's defaults.
/// `--greedy` with any sampling flag is an error, as is a setting the sampler can't honour.
pub fn sampling(
    greedy: bool,
    temp: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    presence_penalty: Option<f32>,
) -> Result<Sampling> {
    let flags = [
        ("--temp", temp.is_some()),
        ("--top-k", top_k.is_some()),
        ("--top-p", top_p.is_some()),
        ("--min-p", min_p.is_some()),
        ("--presence-penalty", presence_penalty.is_some()),
    ];
    if greedy {
        if let Some((flag, _)) = flags.iter().find(|f| f.1) {
            bail!("--greedy always takes the top token, so {flag} has no effect: drop one of them");
        }
        return Ok(Sampling::GREEDY);
    }
    let d = Sampling::DEFAULT;
    let s = Sampling {
        temp: temp.unwrap_or(d.temp),
        top_k: top_k.unwrap_or(d.top_k),
        top_p: top_p.unwrap_or(d.top_p),
        min_p: min_p.unwrap_or(d.min_p),
        presence_penalty: presence_penalty.unwrap_or(d.presence_penalty),
    };
    s.check().map_err(anyhow::Error::msg)?;
    Ok(s)
}

/// `--force-tokens`' pick: the given tokens in order, whatever the logits say. `run` records each as its step's
/// `token`, beside the engine's own top two, so test 3 sees the engine's choice along the oracle's tokens.
pub fn forced(tokens: &[u32]) -> impl FnMut(&[f32], u32) -> u32 + '_ {
    let mut next = tokens.iter().copied();
    move |_, _| {
        next.next()
            .expect("check_forced keeps a run within its forced tokens")
    }
}

/// Checks that every `--force-tokens` id is a token of the model, before it loads.
pub fn check_forced_vocab(tokens: &[u32], n_vocab: usize) -> Result<()> {
    if let Some(i) = tokens.iter().position(|&t| t as usize >= n_vocab) {
        bail!(
            "--force-tokens: token {i} is {}, not below the vocabulary size {n_vocab}",
            tokens[i]
        );
    }
    Ok(())
}

/// Checks `--force-tokens`' tokens so that `run` takes exactly them: at least one, at most `max_tokens`, an
/// end-of-generation token only as the last, and fewer than `max_tokens` only when the last is one, which ends the run
/// as it ended the oracle's.
pub fn check_forced(tokens: &[u32], max_tokens: usize, eog: &[u32]) -> Result<()> {
    let n = tokens.len();
    let Some((&last, before)) = tokens.split_last() else {
        bail!("--force-tokens: the file holds no tokens");
    };
    if let Some(i) = before.iter().position(|t| eog.contains(t)) {
        bail!(
            "--force-tokens: token {i} of {n} is an end-of-generation token, which would end the run there; only \
             the last may be one"
        );
    }
    if n > max_tokens {
        bail!("--force-tokens holds {n} tokens, more than --max-tokens {max_tokens}");
    }
    if n < max_tokens && !eog.contains(&last) {
        bail!(
            "--force-tokens holds {n} tokens and the last doesn't end generation, so the run would go on past them: \
             give --max-tokens {n}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Logits whose argmax is `t`, over a vocabulary of 16.
    fn pick(t: u32) -> Vec<f32> {
        (0..16).map(|i| if i == t { 1.0 } else { 0.0 }).collect()
    }

    /// Greedy generation from logits picking 3, with a decode that answers token `t` with logits picking `t + 1`.
    /// Returns the tokens, why it stopped, the decodes made and the emitted lengths.
    fn drive(max_tokens: usize, room: usize, eog: &[u32]) -> (Vec<u32>, Stop, usize, Vec<usize>) {
        let (mut decodes, mut emitted) = (0, Vec::new());
        let (steps, stop) = run(
            pick(3),
            max_tokens,
            room,
            eog,
            |l, top| {
                assert_eq!(top, engine::greedy(l));
                top
            },
            &mut Plain(|t| {
                decodes += 1;
                Ok(pick(t + 1))
            }),
            |so_far| {
                emitted.push(so_far.len());
                Ok(())
            },
        )
        .unwrap();
        (
            steps.iter().map(|s| s.token).collect(),
            stop,
            decodes,
            emitted,
        )
    }

    #[test]
    fn zero_max_tokens_generates_nothing() {
        assert_eq!(drive(0, 10, &[]), (vec![], Stop::MaxTokens, 0, vec![]));
    }

    #[test]
    fn max_tokens_counts_tokens_and_skips_the_last_decode() {
        assert_eq!(
            drive(3, 10, &[]),
            (vec![3, 4, 5], Stop::MaxTokens, 2, vec![1, 2, 3])
        );
    }

    #[test]
    fn an_end_of_generation_token_is_recorded_not_emitted() {
        // Token 5 ends generation: it is the last step, but never emitted and never decoded.
        assert_eq!(
            drive(10, 10, &[5]),
            (vec![3, 4, 5], Stop::EndOfGeneration, 2, vec![1, 2])
        );
    }

    #[test]
    fn a_full_context_stops_after_the_last_position() {
        // Two positions left: the first token comes from the prompt's logits, then two more are decoded.
        assert_eq!(
            drive(10, 2, &[]),
            (vec![3, 4, 5], Stop::ContextFull, 2, vec![1, 2, 3])
        );
    }

    #[test]
    fn each_step_records_the_pick_and_the_top_two() {
        // A sampler that takes the second-best token: the step names it, and the logits it came from.
        let logits = vec![0.1, 0.9, 0.5, 0.7];
        let (steps, stop) = run(
            logits,
            1,
            10,
            &[],
            |l, _| top_two(l).2,
            &mut Plain(|_| unreachable!("the last token isn't decoded")),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(stop, Stop::MaxTokens);
        assert_eq!(
            steps,
            [Step {
                token: 3,
                top: 1,
                logit: 0.9,
                second: 3,
                second_logit: 0.7
            }]
        );
    }

    #[test]
    fn the_top_of_the_scan_is_the_greedy_pick_even_on_ties() {
        // Coarse values make ties common: the one scan must choose as `greedy` does.
        for seed in 0..200u32 {
            let l: Vec<f32> = (0..40u32)
                .map(|i| ((i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97) >> 7) % 5) as f32)
                .collect();
            assert_eq!(top_two(&l).0, engine::greedy(&l), "{l:?}");
        }
    }

    #[test]
    fn top_two_takes_the_lower_id_on_a_tie() {
        assert_eq!(top_two(&[0.5, 2.0, -1.0, 2.0]), (1, 2.0, 3, 2.0));
        assert_eq!(top_two(&[1.0, 1.0, 1.0]), (0, 1.0, 1, 1.0));
        // A new maximum makes the old one second, even after a lower runner-up.
        assert_eq!(top_two(&[1.0, 3.0, 2.0, 5.0]), (3, 5.0, 1, 3.0));
    }

    #[test]
    fn a_fingerprint_is_fnv1a_of_the_little_endian_bytes() {
        assert_eq!(fingerprint(&[1.0, -2.0, 0.5]), "c598e74ad8b1c9b5");
        assert_eq!(fingerprint(&[]), "cbf29ce484222325");
        // One ulp differs.
        assert_eq!(fingerprint(&[1.0, -2.0, 0.500_000_1]), "058e135b851cf813");
    }

    #[test]
    fn a_fingerprint_equals_fnv1a_of_the_bytes_on_any_vector() {
        // xorshift64: the same values every run; any bit pattern, NaNs and -0.0 included.
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut random = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    f32::from_bits((x >> 16) as u32)
                })
                .collect()
        };
        let vectors = [
            vec![],
            vec![0.0],
            vec![-0.0, f32::NAN, f32::INFINITY, f32::MIN_POSITIVE],
            random(1_001),
            random(248_320),
        ];
        for v in vectors {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            assert_eq!(
                fingerprint(&v),
                format!("{:016x}", qwen36::testing::fnv1a64(&bytes)),
                "{} values",
                v.len()
            );
        }
    }

    #[test]
    fn greedy_with_a_sampling_flag_is_refused() {
        assert_eq!(
            sampling(true, None, None, None, None, None).unwrap(),
            Sampling::GREEDY
        );
        let e = sampling(true, Some(0.7), None, None, None, None)
            .unwrap_err()
            .to_string();
        assert!(e.contains("--greedy") && e.contains("--temp"), "{e}");
        let e = sampling(true, None, None, None, None, Some(0.5))
            .unwrap_err()
            .to_string();
        assert!(e.contains("--presence-penalty"), "{e}");
    }

    #[test]
    fn sampling_flags_override_productions_defaults() {
        assert_eq!(
            sampling(false, None, None, None, None, None).unwrap(),
            Sampling::DEFAULT
        );
        let s = sampling(false, Some(1.0), Some(40), None, Some(0.05), None).unwrap();
        assert_eq!(
            s,
            Sampling {
                temp: 1.0,
                top_k: 40,
                min_p: 0.05,
                ..Sampling::DEFAULT
            }
        );
        let e = sampling(false, None, None, Some(0.0), None, None).unwrap_err();
        assert!(e.to_string().contains("--top-p"), "{e}");
    }

    #[test]
    fn a_record_round_trips_through_json() {
        let r = sample();
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"stop\":\"end-of-generation\""), "{json}");
        assert_eq!(serde_json::from_str::<Record>(&json).unwrap(), r);
    }

    #[test]
    fn a_record_with_warm_info_round_trips_and_an_old_one_still_reads() {
        let mut r = sample();
        r.warm = Some(WarmInfo {
            path: 1,
            id: "20-long-option-rs".into(),
            dirty_id: "01-code".into(),
            cut: Some(12_288),
            split: None,
            snapshot_bytes: 680_263_680,
            restore_bytes: 300_000_000,
            restore_kv_from: 23,
            restore_ms: 21.5,
            pool_limit_bytes: 4 << 30,
        });
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"warm\":{\"path\":1"), "{json}");
        assert_eq!(serde_json::from_str::<Record>(&json).unwrap(), r);
        // A record without warm info serializes without the key, and a record from before it reads as `None`.
        let old = serde_json::to_string(&sample()).unwrap();
        assert!(!old.contains("warm"), "{old}");
        assert_eq!(serde_json::from_str::<Record>(&old).unwrap().warm, None);
    }

    fn sample() -> Record {
        Record {
            format: FORMAT.into(),
            prompt_tokens: vec![1, 2, 3],
            settings: Settings {
                greedy: true,
                temp: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                presence_penalty: 0.0,
                seed: 0,
                max_tokens: 256,
                ctx: 4096,
                cache_slots: 102,
                force_swap_every_token: false,
                all_logits: false,
                cache_slots_asked: Some(102),
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
            },
            steps: vec![Step {
                token: 9,
                top: 9,
                logit: 12.5,
                second: 4,
                second_logit: 11.25,
            }],
            stop: Stop::EndOfGeneration,
            positions: vec![fingerprint(&[1.0, -2.0, 0.5]); 3],
            timings: Timings {
                load_s: 20.5,
                prompt_s: 0.1,
                generate_s: 0.0,
                decoded: 0,
            },
            cache: CacheStats {
                slots_per_layer: 102,
                mib: 7436.0,
                hits: 2000,
                misses: 880,
                swaps_started: 0,
                swaps_done: 0,
                steps: 1,
                longest_submit_ms: 147.5,
                steps_overlapping_copies: 0,
                staged: 0,
                ..CacheStats::default()
            },
            memory: Default::default(),
            vram: Vram {
                heap_mib: 16_304.0,
                budget_at_open_mib: 15_416.0,
                usage_at_open_mib: 0.0,
                budget_mib: 14_339.0,
                usage_mib: 13_761.0,
            },
            prefill: Default::default(),
            warm: None,
            spec: None,
        }
    }

    #[test]
    fn the_record_says_which_kernels_and_staging_ran() {
        // A record carries the prefill kernels and the staging that ran, and the flags that chose them;
        // a record written before staging existed reads with empty kernels, no staging and no flags.
        let mut r = sample();
        r.settings.f32_prefill = true;
        r.settings.stage_from = Some(32);
        r.prefill = PrefillRun {
            kernels: "f32".into(),
            staging: true,
            stage_from: 32,
            staging_mib: 463.1,
            staged_steps: 3,
        };
        r.cache.staged = 4_000;
        let json = serde_json::to_string(&r).unwrap();
        for key in [
            "\"prefill\":{\"kernels\":\"f32\"",
            "\"f32_prefill\":true",
            "\"stage_from\":32",
            "\"staged\":4000",
        ] {
            assert!(json.contains(key), "{key} in {json}");
        }
        assert_eq!(serde_json::from_str::<Record>(&json).unwrap(), r);
        let mut old: serde_json::Value = serde_json::to_value(sample()).unwrap();
        let o = old.as_object_mut().unwrap();
        o.remove("prefill");
        for k in ["f32_prefill", "no_staging", "stage_from"] {
            o["settings"].as_object_mut().unwrap().remove(k);
        }
        o["cache"].as_object_mut().unwrap().remove("staged");
        let back: Record = serde_json::from_value(old).unwrap();
        assert_eq!(back.prefill, PrefillRun::default());
        assert_eq!(back.prefill.kernels, "");
        assert!(
            !back.settings.f32_prefill
                && !back.settings.no_staging
                && back.settings.stage_from.is_none()
        );
        assert_eq!(back.cache.staged, 0);
    }

    #[test]
    fn a_record_from_before_plan_2c_reads_as_static_with_zero_counts() {
        let mut v = serde_json::to_value(sample()).unwrap();
        // Give the new fields values, so reading zeros back shows the defaults, then remove them as old records lack them.
        v["settings"]["upload_cap_mib"] = serde_json::json!(16);
        v["settings"]["no_lend"] = serde_json::json!(true);
        v["settings"]["base_time"] = serde_json::json!(true);
        v["settings"]["cpu_share"] = serde_json::json!(1.0);
        let counts = [
            "decode_hits",
            "decode_misses",
            "lent_slots",
            "fills",
            "shrinks",
            "grows",
            "grows_refused",
            "budget_short",
            "cpu_timeouts",
            "cpu_misses",
        ];
        for k in counts {
            v["cache"][k] = serde_json::json!(5);
        }
        for k in [
            "cache_policy",
            "upload_cap_mib",
            "no_lend",
            "base_time",
            "cpu_share",
        ] {
            assert!(
                v["settings"].as_object_mut().unwrap().remove(k).is_some(),
                "{k}"
            );
        }
        for k in counts {
            assert!(
                v["cache"].as_object_mut().unwrap().remove(k).is_some(),
                "{k}"
            );
        }
        let back: Record = serde_json::from_value(v).unwrap();
        assert_eq!(back.settings.cache_policy, "static");
        assert_eq!(back.settings.upload_cap_mib, None);
        assert!(!back.settings.no_lend);
        assert!(!back.settings.base_time);
        assert_eq!(back.settings.cpu_share, 0.0);
        let c = &back.cache;
        assert_eq!(
            [
                c.decode_hits,
                c.decode_misses,
                c.fills,
                c.shrinks,
                c.grows,
                c.grows_refused,
                c.budget_short,
                c.cpu_timeouts,
                c.cpu_misses
            ],
            [0; 9]
        );
        assert_eq!(c.lent_slots, 0);
    }

    #[test]
    fn a_record_from_before_the_speed_fields_reads_with_their_defaults() {
        // Older records have no cache_slots_asked, fingerprints, profiled or vram: they read as no --cache-slots,
        // fingerprinted, and no VRAM figures.
        let mut v = serde_json::to_value(sample()).unwrap();
        let s = v["settings"].as_object_mut().unwrap();
        s.remove("cache_slots_asked");
        s.remove("fingerprints");
        s.remove("profiled");
        v.as_object_mut().unwrap().remove("vram");
        let back: Record = serde_json::from_value(v).unwrap();
        assert_eq!(back.settings.cache_slots_asked, None);
        assert!(back.settings.fingerprints);
        assert!(!back.settings.profiled);
        assert_eq!(back.vram, Vram::default());
    }

    #[test]
    fn the_memory_block_round_trips_and_an_older_record_reads_nothing() {
        let mut r = sample();
        r.memory = Memory::new(20_595_408_896, Some(20_700_000_000), 61_000_000);
        let v = serde_json::to_value(&r).unwrap();
        let m = v["memory"].as_object().unwrap();
        for k in [
            "host_bytes",
            "host_heap_usage",
            "mtp_vram_bytes",
            "pool_at_26gib",
        ] {
            assert!(m[k].is_u64(), "{k}");
        }
        assert_eq!(m.len(), 4);
        let back: Record = serde_json::from_value(v.clone()).unwrap();
        assert_eq!(back.memory, r.memory);
        // A record from before MTP has no block.
        let mut old = v;
        old.as_object_mut().unwrap().remove("memory");
        let back: Record = serde_json::from_value(old).unwrap();
        assert_eq!(back.memory, Memory::default());
    }

    #[test]
    fn an_unreadable_heap_gives_null_not_a_passing_pool() {
        // 21 GiB used leaves a 1 GiB pool at 26 GiB free; a failed read must not look like an empty heap (8 GiB).
        let known = Memory::new(1, Some(21 << 30), 0);
        assert_eq!(known.pool_at_26gib, Some(1 << 30));
        let unknown = Memory::new(1, None, 0);
        assert_eq!(
            (unknown.host_heap_usage, unknown.pool_at_26gib),
            (None, None)
        );
        let mut r = sample();
        r.memory = unknown;
        let v = serde_json::to_value(&r).unwrap();
        assert!(v["memory"]["host_heap_usage"].is_null());
        assert!(v["memory"]["pool_at_26gib"].is_null());
        assert_eq!(
            serde_json::from_value::<Record>(v).unwrap().memory,
            r.memory
        );
    }

    #[test]
    fn a_forced_run_feeds_the_given_tokens_and_records_its_own_top_two() {
        // The decode answers token t with logits picking t + 1. The run is fed 7, 2 and 9, an end-of-generation token.
        let mut fed = Vec::new();
        let (steps, stop) = run(
            pick(3),
            256,
            10,
            &[9],
            forced(&[7, 2, 9]),
            &mut Plain(|t| {
                fed.push(t);
                Ok(pick(t + 1))
            }),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(stop, Stop::EndOfGeneration);
        assert_eq!(fed, [7, 2]);
        // Each step names the token fed and the engine's own top token there.
        let fed_and_top: Vec<(u32, u32)> = steps.iter().map(|s| (s.token, s.top)).collect();
        assert_eq!(fed_and_top, [(7, 3), (2, 8), (9, 3)]);
    }

    #[test]
    fn forced_tokens_must_be_exactly_the_runs_steps() {
        let eog = [9];
        assert!(check_forced(&[1, 2, 3], 3, &eog).is_ok());
        assert!(check_forced(&[1, 9], 256, &eog).is_ok());
        let refused: [(&[u32], usize, &str); 4] = [
            (&[], 256, "no tokens"),
            (&[1, 9, 2], 256, "token 1 of 3"),
            (&[1, 2, 3], 2, "3 tokens, more than --max-tokens 2"),
            (&[1, 2], 256, "give --max-tokens 2"),
        ];
        for (tokens, max_tokens, want) in refused {
            let e = check_forced(tokens, max_tokens, &eog)
                .unwrap_err()
                .to_string();
            assert!(e.contains(want), "{e}");
        }
    }

    #[test]
    fn without_json_no_fingerprints_are_made() {
        let mut off = Fingerprints::new(false, 8);
        off.add(&[1.0, 2.0]);
        assert!(off.into_list().is_empty());
        let mut on = Fingerprints::new(true, 8);
        on.add(&[1.0, 2.0]);
        assert_eq!(on.into_list(), [fingerprint(&[1.0, 2.0])]);
    }

    #[test]
    fn forced_ids_must_be_in_the_vocabulary() {
        assert!(check_forced_vocab(&[0, 9], 10).is_ok());
        let e = check_forced_vocab(&[1, 10, 99], 10)
            .unwrap_err()
            .to_string();
        assert!(e.contains("token 1 is 10") && e.contains("10"), "{e}");
    }

    #[test]
    fn a_clamped_slot_count_says_what_was_asked() {
        assert_eq!(slots_note(None, 102), "102");
        assert_eq!(slots_note(Some(50), 50), "50");
        assert_eq!(slots_note(Some(200), 102), "102 (asked for 200; 102 fit)");
    }

    #[test]
    fn a_failed_split_f16_build_is_named_on_the_load_line() {
        assert_eq!(coop_note(None), "");
        assert_eq!(
            coop_note(Some("building matmul_cm_q4_k: VK_ERROR_UNKNOWN")),
            ", split-f16 kernels unavailable: building matmul_cm_q4_k: VK_ERROR_UNKNOWN"
        );
    }

    #[test]
    fn a_record_of_another_format_is_refused_naming_file_and_format() {
        let p = std::env::temp_dir().join(format!("artemis-rec-{}.json", std::process::id()));
        std::fs::write(&p, r#"{"format":"artemis-generate-1"}"#).unwrap();
        let e = read_record(&p).unwrap_err().to_string();
        std::fs::remove_file(&p).unwrap();
        assert!(
            e.contains("artemis-rec-") && e.contains("artemis-generate-1"),
            "{e}"
        );
    }

    /// The engine as a function of the token fed: the row after token `t` picks `t + 1` (a vocabulary of 16, so the
    /// stream 3, 4, 5 ... ends at 15). `decode` and `verify` record their calls; `commit` adds up the positions kept.
    struct Mock {
        pending: Vec<u32>,
        decodes: usize,
        verifies: Vec<usize>,
        committed: usize,
    }

    impl Mock {
        fn new() -> Mock {
            Mock {
                pending: Vec::new(),
                decodes: 0,
                verifies: Vec::new(),
                committed: 0,
            }
        }
    }

    impl Verifier for &mut Mock {
        type Error = std::fmt::Error;

        fn decode(&mut self, token: u32) -> Result<Vec<f32>, std::fmt::Error> {
            self.decodes += 1;
            Ok(pick(token + 1))
        }

        fn verify(&mut self, tokens: &[u32]) -> Result<(), std::fmt::Error> {
            assert!(self.pending.is_empty(), "a verify while one is pending");
            self.pending = tokens.to_vec();
            self.verifies.push(tokens.len());
            Ok(())
        }

        fn row(&mut self, i: usize) -> Result<Vec<f32>, std::fmt::Error> {
            Ok(pick(self.pending[i] + 1))
        }

        fn commit(&mut self, kept: usize) -> Result<(), std::fmt::Error> {
            assert!((1..=self.pending.len()).contains(&kept));
            self.committed += kept;
            self.pending.clear();
            Ok(())
        }

        fn n_vocab(&self) -> usize {
            16
        }

        fn misses(&self) -> u64 {
            0
        }

        fn miss_bytes(&self) -> u64 {
            0
        }
    }

    /// A drafter that knows the future: `Right` drafts the engine's own continuation (`x + 1 ...`), `Wrong` drafts
    /// tokens it never picks, `Script` drafts the given tokens after the ones observed so far.
    enum Guess {
        Right,
        Wrong,
        Script(Vec<u32>),
    }

    struct Guesser {
        guess: Guess,
        primed: bool,
        seen: usize,
        last: u32,
    }

    impl Drafter for Guesser {
        fn name(&self) -> &'static str {
            "guess"
        }

        fn draft(&mut self, max: usize) -> Vec<u32> {
            let last = self.last;
            match &self.guess {
                Guess::Right => (1..=max as u32).map(|i| (last + i) % 16).collect(),
                Guess::Wrong => (1..=max as u32).map(|i| (last + i + 5) % 16).collect(),
                Guess::Script(s) => s.iter().skip(self.seen).take(max).copied().collect(),
            }
        }

        fn observe(&mut self, tokens: &[u32]) {
            // The first call is the prompt; the rest are the generated tokens, `x` first.
            if !std::mem::replace(&mut self.primed, true) {
                return;
            }
            self.seen += tokens.len();
            self.last = *tokens.last().expect("observe is never empty");
        }
    }

    fn state(guess: impl Fn() -> Guess + Send + 'static) -> SpecState {
        let cfg = SpecConfig {
            max_draft: 4,
            fixed_draft: Some(4),
            ..SpecConfig::default()
        };
        SpecState::with_drafters(
            cfg,
            Box::new(move || {
                Box::new(Guesser {
                    guess: guess(),
                    primed: false,
                    seen: 0,
                    last: 0,
                })
            }),
        )
    }

    /// What a run gave: the steps, the stop, the emitted lengths and the engine's positions (decodes and kept rows).
    type Outcome = (Vec<Step>, Stop, Vec<usize>, usize);

    fn plain_run(
        max_tokens: usize,
        room: usize,
        eog: &[u32],
        pick_with: &mut dyn FnMut(&[f32], u32) -> u32,
    ) -> Outcome {
        let (mut decodes, mut emitted) = (0, Vec::new());
        let (steps, stop) = run(
            pick(3),
            max_tokens,
            room,
            eog,
            pick_with,
            &mut Plain(|t| {
                decodes += 1;
                Ok(pick(t + 1))
            }),
            |so_far| {
                emitted.push(so_far.len());
                Ok(())
            },
        )
        .unwrap();
        (steps, stop, emitted, decodes)
    }

    fn speculating_run(
        st: &mut SpecState,
        max_tokens: usize,
        room: usize,
        eog: &[u32],
        pick_with: &mut dyn FnMut(&[f32], u32) -> u32,
    ) -> (Outcome, SpecDelta, Vec<usize>) {
        let (mut mock, mut emitted) = (Mock::new(), Vec::new());
        let mut stepper = Speculating::new(st, Anyhow(&mut mock), &[1, 2]);
        let (steps, stop) = run(
            pick(3),
            max_tokens,
            room,
            eog,
            pick_with,
            &mut stepper,
            |so_far| {
                emitted.push(so_far.len());
                Ok(())
            },
        )
        .unwrap();
        let delta = stepper.delta;
        drop(stepper);
        let positions = mock.decodes + mock.committed;
        ((steps, stop, emitted, positions), delta, mock.verifies)
    }

    #[test]
    fn run_with_a_speculating_stepper_records_the_plain_runs_steps() {
        let greedy = |l: &[f32], top: u32| {
            assert_eq!(top, engine::greedy(l));
            top
        };
        // (max_tokens, room, eog): a length stop, a context end, a stop token inside a burst and at its edge, a
        // length that cuts a burst, and the shortest runs.
        let cases: [(usize, usize, &[u32]); 8] = [
            (10, 100, &[]),
            (3, 100, &[]),
            (10, 2, &[]),
            (10, 100, &[7]),
            (10, 100, &[8]),
            (10, 100, &[4]),
            (1, 100, &[]),
            (0, 100, &[]),
        ];
        for (name, guess) in [
            ("right", (|| Guess::Right) as fn() -> Guess),
            ("wrong", || Guess::Wrong),
        ] {
            for (max_tokens, room, eog) in cases {
                let want = plain_run(max_tokens, room, eog, &mut { greedy });
                let mut st = state(guess);
                let (got, delta, verifies) =
                    speculating_run(&mut st, max_tokens, room, eog, &mut { greedy });
                let label = format!("{name} {max_tokens} {room} {eog:?}");
                assert_eq!(got.0, want.0, "{label}");
                assert_eq!(got.1, want.1, "{label}");
                assert_eq!(got.2, want.2, "{label}");
                // Every decoded position (the plain run's decodes) was run exactly once, as a decode or a kept row.
                assert_eq!(got.3, want.3, "{label}");
                // No burst runs past the length or the context: a step may run what is left, and no more. With right
                // drafts every step is as long as the room allows (x and four drafts), so the sizes are exact: the
                // last one is cut to what was left, and a stop token inside a burst doesn't shorten the verify.
                if name == "right" {
                    let want: &[usize] = match (max_tokens, room, eog) {
                        (10, 100, []) => &[5, 4],
                        (3, 100, []) | (10, 2, []) => &[2],
                        (10, 100, [7]) | (10, 100, [8]) | (10, 100, [4]) => &[5],
                        _ => &[],
                    };
                    assert_eq!(verifies, want, "{label}");
                }
                if name == "right" && max_tokens == 10 && room == 100 && eog.is_empty() {
                    // The right drafts ran as bursts (the test would pass vacuously without any).
                    assert!(delta.steps > 0 && delta.accepted > 0, "{label}: {delta:?}");
                    assert_eq!(delta.replays, 0, "{label}");
                }
                if name == "wrong" && max_tokens >= 3 {
                    assert!(delta.steps > 0 && delta.accepted == 0, "{label}: {delta:?}");
                }
            }
        }
    }

    #[test]
    fn forced_tokens_with_speculation_match_without() {
        // Test 3's pick: the given tokens whatever the logits say, drafted ahead by a drafter that knows them.
        let tokens = [7u32, 2, 9, 4, 11, 6];
        for (eog, max_tokens) in [(&[11u32][..], 256), (&[][..], 6)] {
            let want = plain_run(max_tokens, 100, eog, &mut forced(&tokens[..]));
            let mut st = state(|| Guess::Script(vec![7, 2, 9, 4, 11, 6]));
            let (got, delta, _) =
                speculating_run(&mut st, max_tokens, 100, eog, &mut forced(&tokens[..]));
            assert_eq!(got.0, want.0, "{eog:?}");
            assert_eq!(got.1, want.1);
            assert_eq!(got.3, want.3);
            assert!(delta.steps > 0 && delta.accepted > 0, "{delta:?}");
            // Each step names the token forced and the engine's own top token at its row (the fed token's + 1).
            let top = |s: &Step| (s.token, s.top);
            assert_eq!(
                got.0.iter().map(top).collect::<Vec<_>>(),
                want.0.iter().map(top).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn the_stepper_must_report_the_rows_it_gave() {
        struct Liar;
        impl Stepper for Liar {
            fn step(
                &mut self,
                _x: u32,
                _most: usize,
                give: &mut dyn FnMut(&[f32]) -> Result<Next>,
            ) -> Result<usize> {
                give(&pick(4))?;
                Ok(2)
            }
        }
        let e = run(pick(3), 10, 10, &[], |_, t| t, &mut Liar, |_| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(e.contains("2 positions run") && e.contains("1 rows"), "{e}");
    }

    #[test]
    fn a_row_after_end_is_an_error_not_a_longer_run() {
        // A stepper that ignores `End`: the stop token's row, then one more. The loop must not record a step past it.
        struct Deaf;
        impl Stepper for Deaf {
            fn step(
                &mut self,
                _x: u32,
                _most: usize,
                give: &mut dyn FnMut(&[f32]) -> Result<Next>,
            ) -> Result<usize> {
                assert_eq!(give(&pick(9))?, Next::End);
                give(&pick(5))?;
                Ok(2)
            }
        }
        let mut emitted = 0;
        let e = run(
            pick(3),
            10,
            10,
            &[9],
            |_, t| t,
            &mut Deaf,
            |_| {
                emitted += 1;
                Ok(())
            },
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("a row after the run ended"), "{e}");
        assert_eq!(emitted, 1, "only the first token was emitted");
    }

    #[test]
    fn an_old_record_reads_as_speculation_off() {
        let mut v = serde_json::to_value(sample()).unwrap();
        // Today's record says so explicitly...
        assert!(v["settings"]["speculation"].is_null() && v["spec"].is_null());
        // ...and one from before speculation has neither key.
        v["settings"].as_object_mut().unwrap().remove("speculation");
        v.as_object_mut().unwrap().remove("spec");
        let back: Record = serde_json::from_value(v).unwrap();
        assert_eq!((back.settings.speculation, back.spec), (None, None));
        // A speculating run's record round-trips with its settings and counts.
        let mut r = sample();
        r.settings.speculation = Some(SpecSettings::from(&SpecConfig {
            max_draft: 4,
            fixed_draft: Some(4),
            ..SpecConfig::default()
        }));
        r.spec = Some(SpecRecord {
            steps: 9,
            plain_steps: 1,
            drafted: 36,
            accepted: 30,
            replays: 3,
            ..SpecRecord::default()
        });
        let json = serde_json::to_string(&r).unwrap();
        assert!(
            json.contains("\"draft_len\":4") && json.contains("\"replays\":3"),
            "{json}"
        );
        assert_eq!(serde_json::from_str::<Record>(&json).unwrap(), r);
    }

    #[test]
    fn an_old_record_reads_with_no_mtp() {
        // A record with MTP's settings and counts...
        let mut r = sample();
        r.settings.speculation = Some(SpecSettings::from(&SpecConfig::default()));
        r.spec = Some(SpecRecord {
            steps: 9,
            drafted: 36,
            accepted: 30,
            drafted_lookup: 20,
            accepted_lookup: 18,
            drafted_mtp: 16,
            accepted_mtp: 12,
            probes: 2,
            mtp_depth_counts: vec![[10, 8], [6, 4]],
            ..SpecRecord::default()
        });
        let json = serde_json::to_string(&r).unwrap();
        assert!(
            json.contains("\"sources\":[\"lookup\",\"mtp\"]")
                && json.contains("\"drafted_mtp\":16")
                && json.contains("\"mtp_depth_counts\":[[10,8],[6,4]]"),
            "{json}"
        );
        assert_eq!(serde_json::from_str::<Record>(&json).unwrap(), r);
        // ...and one from before MTP has none of those keys: lookup alone, and no MTP counts.
        let mut v = serde_json::to_value(&r).unwrap();
        for k in [
            "sources",
            "probe_every",
            "mtp_window",
            "mtp_ctx",
            "mtp_ring",
            "mtp_accept_prior",
            "mtp_cost_ms",
            "source_choice",
            "mtp_head_rows",
        ] {
            assert!(v["settings"]["speculation"]
                .as_object_mut()
                .unwrap()
                .remove(k)
                .is_some());
        }
        for k in [
            "drafted_lookup",
            "accepted_lookup",
            "drafted_mtp",
            "accepted_mtp",
            "probes",
            "mtp_depth_counts",
        ] {
            assert!(v["spec"].as_object_mut().unwrap().remove(k).is_some());
        }
        let old: Record = serde_json::from_value(v).unwrap();
        let s = old.settings.speculation.unwrap();
        assert_eq!(s.sources, ["lookup"]);
        assert_eq!(
            (
                s.probe_every,
                s.mtp_window,
                s.mtp_ctx,
                s.mtp_ring,
                s.mtp_cost_ms
            ),
            (0, 0, 0, 0, 0.0)
        );
        assert!(s.mtp_accept_prior.is_empty());
        assert_eq!((s.source_choice.as_str(), s.mtp_head_rows), ("priority", 0));
        let c = old.spec.unwrap();
        assert_eq!((c.steps, c.drafted, c.accepted), (9, 36, 30));
        assert_eq!(
            (
                c.drafted_lookup,
                c.accepted_lookup,
                c.drafted_mtp,
                c.accepted_mtp,
                c.probes
            ),
            (0, 0, 0, 0, 0)
        );
        assert!(c.mtp_depth_counts.is_empty());
    }

    /// A verifier with an MTP: its drafts are `x + 1`, `x + 2` ...
    struct WithMtp;

    impl Verifier for WithMtp {
        type Error = std::fmt::Error;
        fn decode(&mut self, _: u32) -> Result<Vec<f32>, std::fmt::Error> {
            unreachable!()
        }
        fn verify(&mut self, _: &[u32]) -> Result<(), std::fmt::Error> {
            unreachable!()
        }
        fn row(&mut self, _: usize) -> Result<Vec<f32>, std::fmt::Error> {
            unreachable!()
        }
        fn commit(&mut self, _: usize) -> Result<(), std::fmt::Error> {
            unreachable!()
        }
        fn n_vocab(&self) -> usize {
            16
        }
        fn mtp_draft(&mut self, x: u32, k: usize) -> Result<Vec<u32>, std::fmt::Error> {
            Ok((1..=k as u32).map(|i| x + i).collect())
        }
        fn has_mtp(&self) -> bool {
            true
        }
        fn misses(&self) -> u64 {
            0
        }
        fn miss_bytes(&self) -> u64 {
            0
        }
    }

    #[test]
    fn the_anyhow_adapter_forwards_the_mtp() {
        let mut v = Anyhow(WithMtp);
        assert!(v.has_mtp());
        assert_eq!(v.mtp_draft(5, 3).unwrap(), [6, 7, 8]);
        // A verifier without one stays without.
        let mut mock = Mock::new();
        let mut v = Anyhow(&mut mock);
        assert!(!v.has_mtp() && v.mtp_draft(5, 3).unwrap().is_empty());
    }

    #[test]
    fn the_depth_counts_are_the_runs_own_and_stop_at_the_last_depth_tested() {
        let before = [(10, 8), (4, 3), (0, 0)];
        let after = [(15, 11), (9, 5), (3, 1), (0, 0), (0, 0)];
        let r = SpecRecord::default().with_depths(&before, &after);
        assert_eq!(r.mtp_depth_counts, [[5, 3], [5, 2], [3, 1]]);
        // A fresh state (nothing before) is the totals themselves; no drafts is an empty list.
        assert_eq!(
            SpecRecord::default()
                .with_depths(&[], &after)
                .mtp_depth_counts,
            [[15, 11], [9, 5], [3, 1]]
        );
        assert!(SpecRecord::default()
            .with_depths(&after, &after)
            .mtp_depth_counts
            .is_empty());
    }

    /// The flags as the command line gives them.
    fn flags(args: &[&str]) -> SpecFlags {
        #[derive(clap::Parser)]
        struct T {
            #[command(flatten)]
            f: SpecFlags,
        }
        let mut a = vec!["t"];
        a.extend(args);
        <T as clap::Parser>::try_parse_from(a).unwrap().f
    }

    fn refused(args: &[&str], cpu_share: f32, base_time: bool) -> String {
        flags(args)
            .config(cpu_share, base_time)
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn speculation_is_on_by_default_and_the_flags_set_its_config() {
        assert_eq!(
            flags(&[]).config(0.0, false).unwrap(),
            Some(SpecConfig::default())
        );
        let c = flags(&[
            "--max-draft",
            "6",
            "--ngram-min",
            "3",
            "--ngram-max",
            "5",
            "--min-gain",
            "1.1",
            "--accept-half-life",
            "16",
            "--accept-prior",
            "0.25",
            "--cost-slope-ms",
            "3",
            "--draft-len",
            "2",
        ])
        .config(0.0, false)
        .unwrap()
        .unwrap();
        assert_eq!(
            c,
            SpecConfig {
                enabled: true,
                max_draft: 6,
                ngram_min: 3,
                ngram_max: 5,
                min_gain: 1.1,
                accept_half_life: 16.0,
                accept_prior: 0.25,
                cost_slope_ms: 3.0,
                fixed_draft: Some(2),
                ..SpecConfig::default()
            }
        );
        assert_eq!(
            flags(&["--speculation", "off"]).config(0.0, false).unwrap(),
            None
        );
        // The settings a record claims are the ones that ran.
        let s = SpecSettings::from(&c);
        assert_eq!(
            (s.max_draft, s.draft_len, s.accept_prior),
            (6, Some(2), 0.25)
        );
    }

    #[test]
    fn a_speculation_flag_with_speculation_off_is_refused() {
        for (flag, value) in [
            ("--max-draft", "4"),
            ("--ngram-min", "2"),
            ("--ngram-max", "4"),
            ("--min-gain", "1.03"),
            ("--accept-half-life", "32"),
            ("--accept-prior", "0.5"),
            ("--cost-slope-ms", "2.5"),
            ("--draft-len", "4"),
        ] {
            // Even a value that equals the default is a claim the record would make.
            assert_eq!(
                refused(&["--speculation", "off", flag, value], 0.0, false),
                format!("{flag} needs --speculation on")
            );
        }
        // The share and base time are fine with speculation off.
        assert!(flags(&["--speculation", "off"])
            .config(0.5, true)
            .unwrap()
            .is_none());
    }

    #[test]
    fn the_mtp_flags_are_refused_with_speculation_off() {
        for (flag, value) in [
            ("--sources", "lookup,mtp"),
            ("--probe-every", "64"),
            ("--mtp-window", "2048"),
            ("--mtp-ctx", "8192"),
            ("--mtp-ring", "256"),
            ("--mtp-accept-prior", "0.9,0.8"),
            ("--mtp-cost-ms", "1.6"),
            ("--source-choice", "priority"),
            ("--mtp-head-rows", "0"),
        ] {
            // Even a value that equals the default is a claim the record would make.
            assert_eq!(
                refused(&["--speculation", "off", flag, value], 0.0, false),
                format!("{flag} needs --speculation on")
            );
        }
        // And on, they set the config, each under its key.
        let c = flags(&[
            "--sources",
            "mtp",
            "--probe-every",
            "16",
            "--mtp-window",
            "1024",
            "--mtp-ctx",
            "4096",
            "--mtp-ring",
            "128",
            "--mtp-accept-prior",
            "0.9,0.5",
            "--mtp-cost-ms",
            "2.5",
            "--source-choice",
            "best",
            "--mtp-head-rows",
            "32768",
        ])
        .config(0.0, false)
        .unwrap()
        .unwrap();
        assert_eq!(
            c,
            SpecConfig {
                sources: vec![SpecSource::Mtp],
                probe_every: 16,
                mtp_window: 1024,
                mtp_ctx: 4096,
                mtp_ring: 128,
                mtp_accept_prior: vec![0.9, 0.5],
                mtp_cost_ms: 2.5,
                source_choice: SourceChoice::Best,
                mtp_head_rows: 32768,
                ..SpecConfig::default()
            }
        );
        // The record claims the sources and keys that ran.
        let s = SpecSettings::from(&c);
        assert_eq!(
            (s.sources, s.probe_every, s.mtp_ring, s.mtp_accept_prior),
            (vec!["mtp".to_string()], 16, 128, vec![0.9, 0.5])
        );
        assert_eq!((s.source_choice.as_str(), s.mtp_head_rows), ("best", 32768));
        // A bad value is refused under its flag's name.
        for (args, want) in [
            (
                &["--probe-every", "0"][..],
                "--probe-every: must be at least 1",
            ),
            (
                &["--mtp-ctx", "300"],
                "--mtp-ctx: must be from 512 to 65536 and a multiple of 256",
            ),
            (
                &["--mtp-window", "5000"],
                "--mtp-window: must be from 1 to --mtp-ctx / 2 (4096)",
            ),
            (
                &["--mtp-ring", "100"],
                "--mtp-ring: must be from 16 to --mtp-ctx / 4 (2048) and a multiple of 16",
            ),
            (
                &["--mtp-accept-prior", "0.5,1.5"],
                "--mtp-accept-prior: must be 1 to 15 values, each from 0 to 1",
            ),
            (
                &["--mtp-cost-ms", "-1"],
                "--mtp-cost-ms: must be a finite number, 0 or more",
            ),
            (
                &["--mtp-head-rows", "512"],
                "--mtp-head-rows: must be 0 (the full head) or at least 1024",
            ),
            (
                &["--source-choice", "fastest"],
                "--source-choice: an unknown choice \"fastest\", expected priority or best",
            ),
        ] {
            assert_eq!(refused(args, 0.0, false), want, "{args:?}");
        }
    }

    #[test]
    fn sources_parse_and_refuse_by_name() {
        let sources = |a: &[&str]| {
            flags(a)
                .config(0.0, false)
                .map(|c| c.unwrap().sources)
                .map_err(|e| e.to_string())
        };
        assert_eq!(sources(&[]), Ok(vec![SpecSource::Lookup, SpecSource::Mtp]));
        assert_eq!(
            sources(&["--sources", "lookup"]),
            Ok(vec![SpecSource::Lookup])
        );
        assert_eq!(sources(&["--sources", "mtp"]), Ok(vec![SpecSource::Mtp]));
        // The order given is the priority.
        assert_eq!(
            sources(&["--sources", "mtp,lookup"]),
            Ok(vec![SpecSource::Mtp, SpecSource::Lookup])
        );
        let e = sources(&["--sources", "lookup,draft"]).unwrap_err();
        assert!(
            e.starts_with("--sources: ") && e.contains("\"draft\""),
            "{e}"
        );
        let e = sources(&["--sources", "mtp,mtp"]).unwrap_err();
        assert!(
            e.starts_with("--sources: ") && e.contains("listed twice"),
            "{e}"
        );
        let e = sources(&["--sources", ""]).unwrap_err();
        assert!(e.starts_with("--sources: "), "{e}");
    }

    #[test]
    fn the_mtp_loads_exactly_when_the_run_speculates_with_it() {
        let cfg = |sources: &[&str]| {
            flags(&["--sources", &sources.join(",")])
                .config(0.0, false)
                .unwrap()
        };
        // The tools' loads ask for the MTP when the config has it (default: lookup and mtp)...
        let o = mtp_options(flags(&[]).config(0.0, false).unwrap().as_ref(), 65_536)
            .unwrap()
            .expect("the default sources include mtp");
        assert_eq!((o.rows, o.ring, o.window), (8192, 256, 2048));
        assert!(mtp_options(cfg(&["mtp"]).as_ref(), 4096).unwrap().is_some());
        // ...not with lookup alone, and not with speculation off.
        assert!(mtp_options(cfg(&["lookup"]).as_ref(), 65_536)
            .unwrap()
            .is_none());
        assert!(mtp_options(None, 65_536).unwrap().is_none());
        // A session under 512 positions can't hold it, and the refusal names the way out.
        let default = flags(&[]).config(0.0, false).unwrap();
        let e = mtp_options(default.as_ref(), 256).unwrap_err().to_string();
        assert!(e.contains("512") && !e.contains("--sources"), "{e}");
        // The commands that take the flags add the way out; verify-cost and mtp-check have no such flag.
        let e = mtp_options_for_flags(default.as_ref(), 256)
            .unwrap_err()
            .to_string();
        assert!(e.contains("512") && e.contains("--sources lookup"), "{e}");
    }

    #[test]
    fn the_cpu_share_and_the_base_time_with_speculation_are_refused() {
        assert_eq!(
            refused(&[], 0.5, false),
            "--cpu-share needs --speculation off"
        );
        assert_eq!(
            refused(&[], 1.0, false),
            "--cpu-share needs --speculation off"
        );
        assert_eq!(
            refused(&[], 0.0, true),
            "--base-time needs --speculation off"
        );
        // As the engine decides it: a share under 1/512 runs no CPU path, so speculation may run.
        assert!(flags(&[]).config(0.001, false).unwrap().is_some());
        assert!(!engine::cpu_miss::path_on(0.001) && engine::cpu_miss::path_on(1.0 / 512.0));
    }

    #[test]
    fn a_draft_len_above_max_draft_lifts_it_unless_max_draft_was_given() {
        // As `lookup-sim --draft-len` does: N above the default 4 raises max_draft to N.
        let c = flags(&["--draft-len", "8"])
            .config(0.0, false)
            .unwrap()
            .unwrap();
        assert_eq!((c.max_draft, c.fixed_draft), (8, Some(8)));
        let c = flags(&["--draft-len", "3"])
            .config(0.0, false)
            .unwrap()
            .unwrap();
        assert_eq!((c.max_draft, c.fixed_draft), (4, Some(3)));
        let c = flags(&["--draft-len", "5"])
            .config(0.0, false)
            .unwrap()
            .unwrap();
        assert_eq!((c.max_draft, c.fixed_draft), (5, Some(5)));
        let c = flags(&["--draft-len", "15"])
            .config(0.0, false)
            .unwrap()
            .unwrap();
        assert_eq!((c.max_draft, c.fixed_draft), (15, Some(15)));
        // An explicit --max-draft is not overridden: the conflict is named.
        assert_eq!(
            refused(&["--max-draft", "4", "--draft-len", "8"], 0.0, false),
            "--draft-len 8 is above --max-draft 4"
        );
        assert!(flags(&["--max-draft", "8", "--draft-len", "8"])
            .config(0.0, false)
            .is_ok());
        assert_eq!(
            refused(&["--draft-len", "16"], 0.0, false),
            "--draft-len must be at most 15"
        );
    }

    #[test]
    fn a_bad_setting_is_refused_under_its_flags_name() {
        for (args, want) in [
            (
                &["--max-draft", "0"][..],
                "--max-draft: must be from 1 to 15",
            ),
            (&["--max-draft", "16"], "--max-draft: must be from 1 to 15"),
            (&["--ngram-min", "0"], "--ngram-min: must be at least 1"),
            (&["--ngram-max", "17"], "--ngram-max: must be at most 16"),
            (
                &["--ngram-min", "5", "--ngram-max", "3"],
                "--ngram-min: must be at most --ngram-max",
            ),
            (
                &["--min-gain", "0.5"],
                "--min-gain: must be a finite number, 1 or more",
            ),
            (
                &["--accept-half-life", "0"],
                "--accept-half-life: must be a finite number above 0",
            ),
            (
                &["--accept-prior", "1.5"],
                "--accept-prior: must be from 0 to 1",
            ),
            (
                &["--cost-slope-ms", "-1"],
                "--cost-slope-ms: must be a finite number, 0 or more",
            ),
        ] {
            assert_eq!(refused(args, 0.0, false), want, "{args:?}");
        }
    }

    #[test]
    fn the_profile_line_counts_verify_steps() {
        assert_eq!(mtp_note(0, 0, 0.0), "");
        assert_eq!(
            mtp_note(4, 10, 6_000.0),
            "; MTP 1.50 ms of GPU time a draft pass over 4 passes (10 draft rows)"
        );
        assert_eq!(mtp_ring_note(None), "");
        assert!(mtp_ring_note(Some((2048, 2032))).contains("2048 to 2032"));
        assert_eq!(mtp_slots_note(&engine::Stats::default()), "");
        assert_eq!(
            mtp_slots_note(&engine::Stats {
                mtp_slots: 64,
                mtp_slot_bytes: 121_634_816,
                ..engine::Stats::default()
            }),
            "; 64 MTP expert slots (116 MiB)"
        );
        assert_eq!(verify_note(0, 0, 0.0), "");
        assert_eq!(
            verify_note(4, 14, 80_000.0),
            "; verify 20.00 ms of GPU time a step over 4 steps (14 positions)"
        );
    }

    #[test]
    fn forced_token_messages_are_single_spaced() {
        let eog = [9];
        for (tokens, max_tokens) in [(&[1u32, 9, 2][..], 256), (&[1, 2][..], 256)] {
            let e = check_forced(tokens, max_tokens, &eog)
                .unwrap_err()
                .to_string();
            assert!(!e.contains("  "), "{e:?}");
        }
    }
}
