//! The Artemis engine: the model on the GPU, a session's decode, one token per submit, and its
//! batched prefill, a chunk of up to 2,048 tokens per step. The non-expert weights live in VRAM; the
//! embedding and every expert in host memory. An expert cache holds experts in VRAM: filled from the
//! routing profile at load with each layer's most-used, then adapted to the session's routes as it runs.
//! `cache.rs` holds its chunked memory and the rows the staging area lends, `prepare_step` with the policy,
//! and the budget watch. The slot table points every expert either at its cache slot (VRAM) or at its host copy,
//! which the GPU reads over PCIe (HOST).
mod base;
mod cache;
pub mod cpu_miss;
mod decode;
mod error;
#[cfg(any(test, feature = "fake"))]
pub mod fake;
mod loaded;
mod logprobs;
mod mtp;
mod policy;
mod pool;
mod prefill;
mod profile;
mod request;
mod reuse;
pub mod rng;
mod sample;
mod session;
mod snapshot;
pub mod spec;
mod stage;
pub mod swap;
mod timing;
mod turn;
mod verify;
mod watch;
mod weights;
mod worker;

pub use cache::{
    cache_bytes, cache_plan, choose_slots, chunk_rows, fit_cache, fit_slots, row_bytes, slot_sizes,
    staging_bytes, staging_plan, Stats, SWAP_POISON,
};
pub use error::{Error, ErrorClass};
pub use loaded::{EngineLoader, Loaded};
pub use logprobs::logprobs;
pub use policy::{
    allowance, earned, CachePolicy, Pick, PolicyParams, PolicyStep, HALF_LIFE, MARGIN,
    PRIOR_TOKENS, RESHAPE_BYTES, UPLOAD_CAP,
};
pub use pool::{
    add_reclaimable, limit_at_load, os_available_bytes, parse_meminfo, pool_limit, pool_on_machine,
    FreeList, Pool, PoolLimit, PoolSlice, POOL_ALIGN, POOL_CHUNK, POOL_MARGIN,
};
pub use prefill::{
    chunk_lengths, prefill_bytes, prefill_host_bytes, ATTN_PAIRS, ATTN_PAIRS_CM, LOGIT_ROWS,
    PREFILL_CHUNK, STAGE_FROM,
};
pub use profile::{Profile, QWEN36_PROFILE};
pub use request::{
    CacheDelta, ChannelSink, Counts, Cut, ErrorKind, Event, EventSink, Finish, Lifecycle, Request,
    Source, ThinkLimit, Timings, TokenLogprobs, THINK_ENDS,
};
pub use reuse::{
    kv_copies, lcp, split_points, unpark_order, Choice, Freed, KvRecord, Marks, Node, Plan,
    ReuseStats, Sizes, Skip, SnapId, Tree, INTERVAL,
};
pub use rng::Rng;
pub use sample::{greedy, Sampler, Sampling};
pub use session::{
    mtp_session_bytes, session_bytes, session_bytes_mtp, MtpOptions, Session, SessionState,
};
pub use snapshot::{link_rates, LinkRates, Parked, Restore, Snapshots, Taken, Unparked};
pub use spec::Source as SpecSource;
pub use spec::SourceChoice;
pub use spec::SpecConfig;
pub use swap::{
    Change, CopyOrder, Entry, Kind, Move, Phase, Residency, RowState, SwapError, SwapPlanner,
};
pub use timing::{
    CpuTimes, Span, SpanKind, StepKind, StepTiming, Timing, TAG_EMBED, TAG_MTP_WINDOW, TAG_OUTPUT,
    TRACE_REPLAY, UNTAGGED,
};
pub use turn::{cache_delta, prefill_points, run_turn, Ending, Live, LiveVerifier, Restored};
pub use verify::SessionVerifier;
pub use vk::parse_uuid;
pub use watch::{BudgetWatch, Verdict, BUDGET_EVERY};
pub use weights::{check_fits, need, slot_table, Need, DRIVER_WORK, VRAM_MARGIN};
pub use worker::{
    host_short_message, vram_short_message, Command, Gone, LoadError, LoadFailure, LoadInfo,
    Loader, Notify, ParkedStats, Reply, UnloadCause, UnloadInfo, Worker, WorkerConfig, WorkerEvent,
    WorkerParams, WorkerStats,
};

use gguf::Gguf;
use kernels::{
    dispatch::{
        check_slot_addr, slot_entry, slot_update, ENTRY_BYTES, STATUS_BYTES, UPDATE_BYTES,
        WHERE_HOST, WHERE_VRAM,
    },
    Kernels,
};
use qwen36::{dims::Dims, model::Model};
use std::{
    cell::{Cell, RefCell},
    path::PathBuf,
    sync::{atomic::AtomicU64, Arc},
};

pub struct Config {
    pub model: PathBuf,
    /// The GPU to use, when there is more than one discrete GPU.
    pub device_uuid: Option<[u8; 16]>,
    /// The context the caller will ask for, so a session that can't fit is refused before the load; 0 to skip.
    pub ctx: usize,
    /// Expert slots per layer (`--cache-slots`); None sizes the cache to the budget. Some(n) is also the ceiling for lent
    /// and grown rows; without it, the cache may grow toward every expert as the budget allows.
    pub cache_slots: Option<usize>,
    /// Start one seeded pseudo-random swap per step (`--force-swap-every-token`).
    pub force_swap_seed: Option<u64>,
    /// The host-RAM snapshot pool's size in GiB, at most; the load cuts it to what the machine
    /// has free, and under 1 GiB it is 0 (reuse off).
    pub snapshot_pool_gib: u32,
    /// Prefill on the f32 kernels even where the split-f16 ones run (`Options::f32_prefill`).
    pub f32_prefill: bool,
    /// Speculative decoding's settings; `SpecConfig::default()` is on.
    pub speculation: SpecConfig,
    /// A counter the load adds 1 to after each weight tensor it uploads and each layer of expert copies, so a caller
    /// can tell a slow load from a stuck one. None (the default) counts nothing.
    pub load_ticks: Option<Arc<AtomicU64>>,
}

impl Config {
    /// The server's defaults for `model`: any GPU, a context of 65,536, the cache sized to the budget, no forced
    /// swaps, an 8 GiB snapshot pool, the default prefill kernels and speculation on.
    pub fn new(model: PathBuf) -> Config {
        Config {
            model,
            device_uuid: None,
            ctx: 65_536,
            cache_slots: None,
            force_swap_seed: None,
            snapshot_pool_gib: 8,
            f32_prefill: false,
            speculation: SpecConfig::default(),
            load_ticks: None,
        }
    }
}

/// How to load.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// The GPU, when there is more than one discrete GPU.
    pub device_uuid: Option<[u8; 16]>,
    /// The context the caller will ask for: its session is counted before the cache is sized. 0 sizes the cache with
    /// no session counted, so a later `session(n)` can fail out of budget; callers set `ctx`.
    pub ctx: usize,
    /// Expert slots per layer: None sizes the cache to the budget; Some(n) asks for n, clamped to what
    /// fits. n is also the ceiling for lent and grown rows; without it, the cache may grow toward every expert as the
    /// budget allows.
    pub cache_slots: Option<usize>,
    /// Start one seeded pseudo-random swap per step (`--force-swap-every-token`). Before each step
    /// the planner picks a random layer and swaps one of its cached experts for one outside the cache, when no swap of
    /// that layer is in flight. It needs some experts in the cache and some out of it (`NoRoomToSwap`).
    pub force_swap_seed: Option<u64>,
    /// Tokens per prefill chunk, at most: the prefill buffers are sized for it, and their VRAM comes out of
    /// the cache's. None takes `PREFILL_CHUNK`; it is cut to `ctx` when that is smaller. It must be from 1 to
    /// `PREFILL_CHUNK`, which the kernels' grids are sized for: anything else is refused before the model loads
    /// (`BadPrefillChunk`).
    pub prefill_chunk: Option<usize>,
    /// Tests only: every swap's copy first fills its slot with `SWAP_POISON`, so a step that
    /// reads a slot mid-copy gets NaN logits and fails instead of passing by luck.
    pub poison_swaps: bool,
    /// Run prefill's dense projections, grouped experts and attention on today's f32 kernels even where the device has
    /// the split-f16 cooperative-matrix ones (`--f32-prefill`): for comparisons and
    /// diagnostics, and the speed gate refuses such runs. A device without the cooperative-matrix kernels runs f32
    /// whatever this says. It is also the way around the split's limit: a prefill input above f16's 65,504 fails the
    /// split-f16 kernels with `Error::NonFinite` where the f32 kernels run (`prefill`'s module doc).
    pub f32_prefill: bool,
    /// Prefill without staging buffers (`--no-staging`): the cache keeps the slots staging would
    /// take (about 7 a layer at `--ctx 65536`: 131 → 124 measured on 2026-10-11), and every chunk reads non-resident
    /// experts in place.
    /// For comparisons; the speed gate refuses such runs.
    pub no_staging: bool,
    /// Chunks of at least this many tokens are staged; shorter ones read non-resident experts in place.
    /// None takes `STAGE_FROM`, the measured threshold; tests use 1 (stage every chunk) and `usize::MAX`
    /// (stage none). It changes where weights are read from, never which kernel runs, so no result depends on it.
    pub stage_from: Option<usize>,
    /// Tests only: every staging copy first fills a 64 MiB scratch buffer and then poisons its regions with
    /// `SWAP_POISON`, so it lands late and a layer that read its staging buffer without waiting for its copy fails. The
    /// scratch is allocated after the cache is sized, so it comes out of the VRAM margin.
    pub poison_staging: bool,
    /// Tests only: the staging copy of this layer fails, once, as a failed submit would.
    pub fail_staging_once: Option<usize>,
    /// Rows per chunk of cache memory: None takes `chunk_rows` (14 on the real model). Tests use
    /// small chunks.
    pub chunk_rows: Option<usize>,
    /// The cache policy: adaptive by default; static (`--cache-policy static`) keeps the
    /// profile's cache, only filling slots that come empty, in the profile's order. A forced swap a
    /// step (`force_swap_seed`) runs static whatever this says, so its swap is the step's only one.
    pub cache_policy: CachePolicyKind,
    /// Bytes of copies per token the policy may make (`--upload-cap`): None takes `UPLOAD_CAP`.
    /// For tuning; the speed and hit-rate gates refuse a run given `--upload-cap`, whatever its value. A cap outside
    /// 1 ..= `RESHAPE_BYTES` (2 GiB) is refused before the model loads (`Error::BadUploadCap`).
    pub upload_cap: Option<u64>,
    /// Tokens between budget checks: None takes `BUDGET_EVERY`, and 0 is taken as 1. Tests use 1.
    pub budget_every: Option<usize>,
    /// Tests only: the cache's own rows at most this many, as a tighter budget would fit, so that lent rows and grown
    /// chunks have room below `cache_slots` on a model whose cache otherwise fits whole.
    pub own_slots: Option<usize>,
    /// Don't lend the staging buffers' VRAM to the cache between staged prefill chunks (`--no-lend`).
    /// For comparisons; the speed gate refuses such runs.
    pub no_lend: bool,
    /// Measurement only (`--base-time`): decode's expert kernels read a copy of the slot table in
    /// which every HOST entry points at a resident expert of its layer (`base::base_table`), so every routed expert is
    /// read from VRAM. The output is meaningless; the per-token time is decode's base time: every routed expert read
    /// from VRAM, plus the table's per-step rebuild and copy. Prefill reads the real tables. The speed and hit-rate
    /// gates refuse such runs.
    pub base_time: bool,
    /// The share of HOST misses the CPU serves, 0.0 to 1.0 (`--cpu-share`): `cpu_miss::share_q8` of 256
    /// experts, the same ones in every layer. 0.0 (the default) leaves the HOST path as it was, and so does a share
    /// below 1/512, which assigns no expert (`cpu_miss::path_on`): it is taken as 0. A value outside 0.0 ..= 1.0 is
    /// refused before the model loads (`Error::BadCpuShare`), as is a share that runs the path (`path_on`, at least
    /// 1/512) with `base_time` (`BaseTimeAndCpuShare`) or on a machine without what the path needs
    /// (`CpuPathUnavailable`), or with experts that are not all Q4_K, Q5_K, Q6_K or Q8_0 (`CpuShareExpertType`). The
    /// speed and hit-rate gates refuse a run given a non-zero share. Expert copies are
    /// host-cached memory whatever the share.
    pub cpu_share: f32,
    /// Speculative decoding will run on this engine: the prefill buffers' work area is sized for the larger of
    /// prefill's need and a verify step's (`verify::VerifyLayout` at `ctx`, 16 tokens), where the verify step's buffers
    /// live, which at the production chunk costs no VRAM. It is refused before the model loads with the CPU miss path
    /// on (`cpu_miss::path_on(cpu_share)`), with `base_time`, or on a model that routes more than 16 experts a token
    /// (`Error::SpeculationRefused`). Set it only where
    /// `Session::verify` runs: the loader (`Config::speculation`) and the tools that verify. False by default.
    ///
    /// The work area covers sessions of up to `ctx` positions (the attention partials grow with the context): `ctx` 0
    /// sizes it for a 1-position context, and a session opened for more than `ctx` can outgrow it, which
    /// `Session::verify` refuses (`Error::VerifyBuffers`) rather than write past it.
    pub speculation: bool,
    /// The MTP drafter will run: the load also puts `blk.40`'s non-expert weights in VRAM and its experts in
    /// host memory (always HOST in the slot table), and counts the session's MTP rows and ring. Some only with
    /// `speculation` (`Error::SpeculationRefused` otherwise); build it with `MtpOptions::for_session`. None by default:
    /// the load is then the one without the MTP, byte for byte.
    pub mtp: Option<MtpOptions>,
    /// A counter the load adds 1 to after each weight tensor it uploads and each layer of expert copies
    /// (`Config::load_ticks`). None counts nothing.
    pub load_ticks: Option<Arc<AtomicU64>>,
}

/// Which cache policy runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CachePolicyKind {
    /// Scores follow the routes; swaps under the upload cap.
    #[default]
    Adaptive,
    /// The static profile's cache (`--cache-policy static`).
    Static,
}

/// Each engine's identity, so a session runs only on the engine that made it.
static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub struct Engine {
    pub(crate) id: u64,
    /// The VRAM heap when the engine began loading.
    pub(crate) at_open: vk::HeapBudget,
    pub(crate) dims: Dims,
    pub(crate) k: Kernels,
    pub(crate) w: weights::Weights,
    pub(crate) host: weights::HostCopies,
    /// The slot table, in the weights' VRAM arena.
    pub(crate) table: vk::Slice,
    /// The expert cache's slots in VRAM, in chunks of whole rows.
    pub(crate) cache: RefCell<cache::Cache>,
    /// Prefill's staging buffers and prefill table; None with `no_staging`, when every expert is
    /// cached, or when the budget had no room for them.
    pub(crate) staging: Option<stage::Staging>,
    /// What the slot table holds, entry by entry, as (addr, where): written wherever the table is (at load, by
    /// `set_slot_entry`, and by each step's staged changes as the step takes them), so a staged chunk's prefill table is
    /// built from the table it runs with. If a step fails before applying its changes, the mirror leads the table until
    /// the next step's `take_staged` puts them back; so a staged chunk builds its prefill table after `take_staged`.
    pub(crate) mirror: RefCell<Vec<(u64, u32)>>,
    /// Chunks of at least this many tokens are staged (`Options::stage_from`, else `STAGE_FROM`).
    pub(crate) stage_from: Cell<usize>,
    /// Tests only (`Options::poison_staging`): staging copies poison their regions before copying, after filling
    /// `stage_delay`.
    pub(crate) poison_staging: bool,
    /// Tests only (`Options::poison_staging`): the 64 MiB scratch each staging copy fills first, so it lands late; its
    /// arena keeps it alive. Allocated after `fit_cache` has sized the cache, so it comes out of the VRAM margin. None
    /// without `poison_staging` or without staging buffers.
    pub(crate) stage_delay: Option<(vk::Arena, vk::Slice)>,
    /// Tests only (`Options::fail_staging_once`): the layer whose staging copy fails next.
    pub(crate) fail_staging_once: Cell<Option<usize>>,
    /// Timed staging copies not yet read back (`collect_copy_spans`).
    pub(crate) copy_tags: RefCell<Vec<timing::CopyTag>>,
    /// The planner's view of the table, and the cache's statistics.
    pub(crate) swaps: RefCell<cache::Swaps>,
    /// The routing log: each layer's routed expert ids for the last decoded token (a prefill step doesn't write it),
    /// `[n_layer][n_expert_used]` f32.
    pub(crate) routes: vk::Buffer,
    pub(crate) scratch: decode::Scratch,
    /// Prefill's buffers, sized for its chunks.
    pub(crate) prefill: prefill::Prefill,
    /// The expert kernels' status (`kernels::dispatch::read_status`).
    pub(crate) status: vk::Buffer,
    /// The step's RoPE table (`kernels::twin::attention::rope_table`).
    pub(crate) rope: vk::Buffer,
    /// Every logit, read back each step.
    pub(crate) logits: vk::Buffer,
    /// The slot-table update list: the changes the next step applies, in pinned host memory.
    pub(crate) updates: vk::Buffer,
    /// Changes waiting for the next step, as (entry index, where, address), at most one per entry.
    pub(crate) staged: RefCell<Vec<(u32, u32, u64)>>,
    /// Step timings while `set_timing(true)` is in force; None when off.
    pub(crate) timing: RefCell<Option<Vec<timing::StepTiming>>>,
    /// The submit trace's label for the step being run (`timing::TRACE_*`), which `submit_step` hands the device.
    pub(crate) trace_label: std::cell::Cell<u32>,
    /// The word each swap's copy fills its slot with first (`Options::poison_swaps`); None for a plain copy.
    pub(crate) poison: Option<u32>,
    /// Whether prefill runs the split-f16 cooperative-matrix kernels (the device has them and `f32_prefill` is off).
    pub(crate) coop: bool,
    /// Whether prefill's attention runs `attn_prefill_cm`: `coop`, and the model's head fits the kernel
    /// (`kernels::dispatch::attn_prefill_cm_fits`).
    pub(crate) coop_attn: bool,
    /// Each main layer's slot bytes: what a copy into the cache costs against the upload cap.
    pub(crate) slot_bytes: Vec<u64>,
    /// The most slots per layer the cache may hold (`Options::cache_slots`, else every expert): lent rows and grown
    /// chunks stay within it.
    pub(crate) ceiling: usize,
    /// The base time's table (`Options::base_time`); None otherwise.
    pub(crate) base: Option<base::BaseTable>,
    /// Tokens between budget checks (`Options::budget_every`, else `BUDGET_EVERY`).
    pub(crate) budget_every: usize,
    /// Tests only (`set_budget_cut`): bytes taken off every budget reading, as if another process had taken them.
    pub(crate) budget_cut: Cell<u64>,
    /// Tests only (`set_grow_cut`): bytes taken off the budget reading a grow takes with its chunk allocated, as if the
    /// driver had lowered the budget once the chunk was allocated.
    pub(crate) grow_cut: Cell<u64>,
    /// The CPU miss path, running exactly while the share runs it (`path_on`, `set_cpu_share`): at share 0 no
    /// coordinator or worker thread exists and decode records today's dispatches.
    pub(crate) cpu: RefCell<Option<cpu_miss::CpuPath>>,
    /// A CPU-path coordinator was detached (`CpuPath::shutdown` timed out): it may still read the host expert copies,
    /// so `Drop` leaks them instead of freeing them.
    pub(crate) cpu_detached: Cell<bool>,
    /// Opened with `Options::speculation`: the prefill work area holds a verify step's buffers, up to `ctx` positions
    /// deep, and `Session::verify` may run.
    pub(crate) speculation: bool,
    /// Opened with `Options::mtp`: `blk.40` is on the GPU (`w.mtp`) and in host memory (`host`).
    pub(crate) mtp: Option<MtpOptions>,
    /// (asked, used) when the load cut `mtp_ring` to fit the prefill chunk (`mtp::clamp_ring`).
    pub(crate) mtp_ring_clamp: Option<(usize, usize)>,
    pub(crate) dev: vk::Device,
}

impl Drop for Engine {
    /// Waits for both queues before any field is freed. Fields drop in declaration order, so `host` (the host
    /// copies), `cache`, `table` and the scratch would free their memory before `dev`'s own wait-idle runs; a swap's
    /// async copy, or a step whose wait timed out, could still be using them. A wait that failed any way but a
    /// device loss is followed by a device-wide wait with no timeout (`must_wait_idle`).
    fn drop(&mut self) {
        // The coordinator first: it reads the host copies and the mailbox, so it is joined before either is freed
        // (the mailbox itself drops with the field, after the queues are idle). One that does not stop in time is
        // detached, and what it may still read is leaked instead (`CpuPath::shutdown`).
        if let Some(c) = self.cpu.get_mut() {
            if !c.shutdown() {
                self.cpu_detached.set(true);
            }
        }
        if self.cpu_detached.get() {
            self.host.leak_experts();
        }
        let mut idle = true;
        for q in [vk::QueueKind::Transfer, vk::QueueKind::Compute] {
            if let Err(e) = self.dev.wait(q, self.dev.submitted(q)) {
                idle &= !must_wait_idle(&e);
            }
        }
        // A queue wait that timed out (or failed any way but a device loss) leaves work that may still run, and a
        // freed buffer under it is a GPU use-after-free. `vkDeviceWaitIdle` has no timeout, so a hung GPU holds the
        // process here instead of having its memory freed and reused under it. Under `serve` the stall watchdog then
        // ends the process (exit 4); the watchdog exists only there, so any other caller waits for the GPU. A lost
        // device has nothing running, so it is not waited for.
        if !idle {
            let _ = self.dev.wait_idle();
        }
    }
}

/// Whether a failed queue wait leaves the device possibly busy, so the engine must wait for it to go idle before it
/// frees anything: true for everything but a device loss, after which nothing runs.
fn must_wait_idle(e: &vk::Error) -> bool {
    !matches!(e, vk::Error::DeviceLost { .. })
}

impl Engine {
    /// Opens and validates the model file as Qwen3.6-35B-A3B, then loads it.
    pub fn load(cfg: &Config) -> Result<Engine, Error> {
        Engine::load_dims(cfg, &Dims::QWEN36)
    }

    /// `load` for a model of `dims`: the server's worker test loads the tiny model through it. Only tests pass dims
    /// other than `Dims::QWEN36`.
    pub fn load_dims(cfg: &Config, dims: &Dims) -> Result<Engine, Error> {
        let g = Gguf::open(&cfg.model)?;
        Engine::open(
            &g,
            dims,
            &Options {
                device_uuid: cfg.device_uuid,
                ctx: cfg.ctx,
                cache_slots: cfg.cache_slots,
                force_swap_seed: cfg.force_swap_seed,
                prefill_chunk: None,
                poison_swaps: false,
                f32_prefill: cfg.f32_prefill,
                no_staging: false,
                stage_from: None,
                poison_staging: false,
                fail_staging_once: None,
                chunk_rows: None,
                cache_policy: CachePolicyKind::Adaptive,
                upload_cap: None,
                budget_every: None,
                own_slots: None,
                no_lend: false,
                base_time: false,
                cpu_share: 0.0,
                speculation: cfg.speculation.enabled,
                mtp: MtpOptions::for_session(&cfg.speculation, cfg.ctx)?,
                load_ticks: cfg.load_ticks.clone(),
            },
        )
    }

    /// Drops the engine, which waits for both queues (`Drop`). Every other holder of the device is dropped or finished
    /// with first: the sessions, and the snapshots' GPU work, since each `vk::Buffer` holds the
    /// device `Arc` and the device is destroyed only when the last one drops. Parking drops the session and this
    /// engine before the pool is copied into host RAM (`Loaded::unload`), so the engine's host expert copies are freed
    /// first; the pool's buffers outlive the engine on purpose, holding the device's `Arc`. Without parking the
    /// snapshots drop first.
    pub fn unload(self) {
        drop(self);
    }

    /// Loads `g` as a model of `dims` with a cache of zero: every expert starts in host memory. With the default
    /// options it also allocates the staging area, which prefill's chunks of at least `STAGE_FROM` tokens read from.
    /// Only tests pass dims other than `Dims::QWEN36`.
    pub fn from_gguf(
        g: &Gguf,
        dims: &Dims,
        device_uuid: Option<[u8; 16]>,
    ) -> Result<Engine, Error> {
        Engine::from_gguf_with_ctx(g, dims, device_uuid, 0)
    }

    /// `from_gguf`, refusing up front, with the sizes, when the weights and a session of `ctx` positions can't both
    /// fit. `ctx` 0 checks the weights alone.
    pub fn from_gguf_with_ctx(
        g: &Gguf,
        dims: &Dims,
        device_uuid: Option<[u8; 16]>,
        ctx: usize,
    ) -> Result<Engine, Error> {
        Engine::open(
            g,
            dims,
            &Options {
                device_uuid,
                ctx,
                cache_slots: Some(0),
                ..Options::default()
            },
        )
    }

    /// Loads `g` as a model of `dims`. Refuses up front, with the sizes, when the weights and a session of
    /// `opts.ctx` positions can't both fit. The cache gets what the budget leaves, or the slots asked for
    /// clamped to that, and is filled with each layer's most-routed experts.
    pub fn open(g: &Gguf, dims: &Dims, opts: &Options) -> Result<Engine, Error> {
        let chunk = prefill::chunk_for(opts.prefill_chunk, opts.ctx)?;
        // The MTP's ring is also cut to the prefill chunk (a pass keeps a RoPE table per row in the chunk's buffers): the
        // load line says so (`Engine::mtp_ring_clamp`).
        let clamped;
        let mut mtp_ring_clamp = None;
        let opts = match opts.mtp {
            Some(o) if mtp::clamp_ring(o.ring, chunk) != o.ring => {
                let ring = mtp::clamp_ring(o.ring, chunk);
                mtp_ring_clamp = Some((o.ring, ring));
                clamped = Options {
                    mtp: Some(MtpOptions { ring, ..o }),
                    ..opts.clone()
                };
                &clamped
            }
            _ => opts,
        };
        let upload_cap = policy::cap_for(opts.upload_cap)?;
        cpu_miss::check_share(opts.cpu_share, opts.base_time)?;
        verify::check_speculation(opts, dims)?;
        let m = Model::open(g, dims)?;
        // The CPU worker has kernels for four quantized types only; refuse before the device opens.
        let expert_types: Vec<_> = m
            .layers
            .iter()
            .flat_map(|l| [l.moe.gate_exps.ty, l.moe.up_exps.ty, l.moe.down_exps.ty])
            .collect();
        cpu_miss::check_expert_types(opts.cpu_share, &expert_types)?;
        let dev = vk::Device::open(opts.device_uuid)?;
        cpu_miss::check_path(opts.cpu_share, cpu_miss::Path::probe(&dev))?;
        // The expert copies are host-cached memory (the CPU reads them at full speed): a device without that type
        // refuses here, whatever the share.
        if !dev.has_kind(vk::MemKind::HostCached) {
            return Err(vk::Error::NoMemoryType(format!("{:?}", vk::MemKind::HostCached)).into());
        }
        let at_open = dev.heap_budget(vk::MemKind::Vram)?;
        // Speculation sizes prefill's work area for a verify step too (`prefill_sizes`); a context of 0 counts 1 position.
        let verify_ctx = opts.speculation.then(|| opts.ctx.max(1));
        if let Some(o) = opts.mtp {
            mtp::check_prefill(dims, chunk, verify_ctx, o)?;
        }
        let mut n = need(&m, opts.mtp.is_some()).with_prefill(dims, chunk, verify_ctx);
        if opts.ctx > 0 {
            n = n.with_session(dims, opts.ctx, opts.mtp);
        }
        check_fits(
            &n,
            dev.free_bytes(vk::MemKind::Vram)?,
            dev.free_bytes(vk::MemKind::HostPinned)?,
        )?;
        // A forced swap needs experts in the cache and out of it: --cache-slots 0 is refused before the weights load,
        // and any other size once the cache is fitted.
        if opts.cache_slots == Some(0) {
            cache::forced_swaps(opts.force_swap_seed, 0, dims.n_expert)?;
        }
        let k = Kernels::new(&dev)?;
        let coop = k.coop() && !opts.f32_prefill;
        let coop_attn = coop && kernels::dispatch::attn_prefill_cm_fits(dims.head_dim);
        let mut w =
            weights::Weights::upload(&dev, &m, opts.mtp.is_some(), opts.load_ticks.as_deref())?;
        let host =
            weights::HostCopies::copy(&dev, &m, opts.mtp.is_some(), opts.load_ticks.as_deref())?;
        if opts.mtp.is_some() {
            // `argmax_rows` addresses embedding rows as `base + id * row_bytes`.
            host.embd_base();
        }
        // The rest of the weights' arena, in its plan's order: the slot table, written once the cache is filled, and
        // the scratch buffers.
        let table = w.arena.alloc(&dev, weights::table_bytes(dims))?;
        let scratch = decode::Scratch::new(&dev, &mut w.arena, dims)?;
        let prefill = prefill::Prefill::new(&dev, dims, chunk, verify_ctx)?;
        // The cache comes last, so the driver's usage counts everything else Artemis holds in VRAM. It is sized against
        // the budget the driver gives once the cache and the session are both allocated.
        let sizes = slot_sizes(&m);
        let session = if opts.ctx > 0 {
            session::session_sizes_mtp(dims, opts.ctx, opts.mtp)
        } else {
            Vec::new()
        };
        let per_chunk = opts.chunk_rows.unwrap_or_else(|| cache::chunk_rows(&sizes));
        let (slots, _, (mut cache, staging_vram, _)) = fit_cache(
            dev.heap_budget(vk::MemKind::Vram)?,
            n.session,
            &sizes,
            dims.n_expert,
            opts.cache_slots.map_or(opts.own_slots, |n| {
                Some(opts.own_slots.map_or(n, |o| o.min(n)))
            }),
            !opts.no_staging,
            |slots, staged| {
                let cache = cache::Cache::new(&dev, sizes.clone(), slots, per_chunk)?;
                let staging = if staged {
                    stage::StagingVram::new(&dev, &sizes, dims.n_expert, slots)?
                } else {
                    None
                };
                // A cache of none is asked for, not sized, so nothing is held back for the session, unless the
                // staging area is sized beside it.
                let held = if opts.cache_slots == Some(0) && !staged {
                    None
                } else {
                    cache::reserve(&dev, &session)?
                };
                Ok((cache, staging, held))
            },
            || dev.heap_budget(vk::MemKind::Vram),
        )?;
        // The prefill table's pinned host buffer is allocated once the fit has settled, not in its rounds.
        let staging = staging_vram.map(|s| s.finish(&dev)).transpose()?;
        // Between staged prefill chunks the staging buffers lend their VRAM to the cache as whole rows,
        // up to the ceiling `--cache-slots` sets.
        let ceiling = opts.cache_slots.unwrap_or(dims.n_expert).min(dims.n_expert);
        if let (Some(st), false) = (&staging, opts.no_lend) {
            let places: Vec<Vec<vk::Slice>> =
                stage::lend_plan(&sizes, st.bufs[0].size, ceiling.saturating_sub(slots))
                    .into_iter()
                    .map(|row| {
                        row.into_iter()
                            .enumerate()
                            .map(|(l, (b, at))| st.bufs[b].sub(at, sizes[l]))
                            .collect()
                    })
                    .collect();
            cache.lend(places);
        }
        let live = cache.rows();
        let forced = cache::forced_swaps(opts.force_swap_seed, live, dims.n_expert)?;
        // A forced swap a step is the step's only one (test 5's guard): the policy is static with it.
        let adaptive = opts.cache_policy == CachePolicyKind::Adaptive && forced.is_none();
        let profile = Profile::for_dims(dims);
        let policy = CachePolicy::new(
            &profile,
            dims,
            adaptive,
            PolicyParams {
                cap: upload_cap,
                ..PolicyParams::default()
            },
        );
        let resident: Vec<Vec<usize>> = (0..dims.n_layer)
            .map(|l| profile.ranked(l)[..live].to_vec())
            .collect();
        cache.fill(&dev, &host, &resident)?;
        let planner = SwapPlanner::new(dims.n_expert, resident);
        // Layer n_layer, the MTP block, is not the planner's: its entries are its host copies.
        let table_bytes = slot_table(
            dims,
            |l, e| {
                if l == dims.n_layer {
                    return (host.slot_addr(l, e), WHERE_HOST);
                }
                match planner.entry(l, e) {
                    Entry::Vram(s) => (cache.slot(l, s).addr, WHERE_VRAM),
                    Entry::Host => (host.slot_addr(l, e), WHERE_HOST),
                }
            },
            opts.mtp.is_some(),
        );
        dev.upload(table, &table_bytes)?;
        let mirror = stage::entries(&table_bytes);
        let status = dev.alloc(
            STATUS_BYTES as u64,
            vk::MemKind::HostPinned,
            "the expert status",
        )?;
        let rope = dev.alloc(
            4 * dims.n_rot as u64,
            vk::MemKind::HostPinned,
            "the RoPE table",
        )?;
        let logits = dev.alloc(4 * dims.n_vocab as u64, vk::MemKind::Readback, "the logits")?;
        let updates = dev.alloc(
            ((dims.n_layer + 1) * dims.n_expert * UPDATE_BYTES) as u64,
            vk::MemKind::HostPinned,
            "the slot-table update list",
        )?;
        let routes = dev.alloc(
            4 * (dims.n_layer * dims.n_expert_used) as u64,
            vk::MemKind::Readback,
            "the routing log",
        )?;
        let stats = Stats {
            slots_per_layer: slots,
            cache_bytes: cache.bytes(),
            staging_bytes: staging.as_ref().map_or(0, |s| s.bytes),
            lent_slots: live - slots,
            ..Stats::default()
        };
        // Tests only: a 64 MiB scratch each staging copy fills first, so the copies land late. Allocated after the fit,
        // it comes out of the VRAM margin.
        let stage_delay = if opts.poison_staging && staging.is_some() {
            let mut a = vk::Arena::new(vk::MemKind::Vram, 64 << 20, "the staging delay (tests)");
            let s = a.alloc(&dev, 64 << 20)?;
            Some((a, s))
        } else {
            None
        };
        // Measurement only: the base time's table, also out of the margin.
        let base = if opts.base_time {
            Some(base::BaseTable::new(&dev, dims.n_layer, dims.n_expert)?)
        } else {
            None
        };
        let e = Engine {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            at_open,
            dims: dims.clone(),
            k,
            w,
            host,
            table,
            cache: RefCell::new(cache),
            staging,
            mirror: RefCell::new(mirror),
            stage_from: Cell::new(opts.stage_from.unwrap_or(STAGE_FROM)),
            poison_staging: opts.poison_staging,
            stage_delay,
            fail_staging_once: Cell::new(opts.fail_staging_once),
            copy_tags: RefCell::new(Vec::new()),
            swaps: RefCell::new(cache::Swaps {
                planner,
                stats,
                forced,
                policy,
                bank: 0,
                watch: watch::BudgetWatch::default(),
                since_check: 0,
                copies_since: 0,
                copies_last: 0,
            }),
            routes,
            scratch,
            prefill,
            status,
            rope,
            logits,
            updates,
            staged: RefCell::new(Vec::new()),
            timing: RefCell::new(None),
            trace_label: std::cell::Cell::new(0),
            poison: opts.poison_swaps.then_some(SWAP_POISON),
            coop,
            coop_attn,
            slot_bytes: sizes,
            ceiling,
            base,
            budget_every: opts.budget_every.unwrap_or(BUDGET_EVERY).max(1),
            budget_cut: Cell::new(0),
            grow_cut: Cell::new(0),
            cpu: RefCell::new(None),
            cpu_detached: Cell::new(false),
            speculation: opts.speculation,
            mtp: opts.mtp,
            mtp_ring_clamp,
            dev,
        };
        // The CPU miss path starts last, after the checks above refused a share the machine can't run.
        if cpu_miss::path_on(opts.cpu_share) {
            e.set_cpu_share(opts.cpu_share)?;
        }
        Ok(e)
    }

    /// The VRAM heap's size, budget and this process's usage when the engine began loading, before Artemis allocated
    /// anything: the heap's size minus the budget then is what the driver held back from this process (on this
    /// driver a fixed reserve, not the desktop's use).
    pub fn vram_at_open(&self) -> vk::HeapBudget {
        self.at_open
    }

    pub fn dims(&self) -> &Dims {
        &self.dims
    }

    pub fn device_name(&self) -> &str {
        &self.dev.name
    }

    /// The Vulkan driver's name and version, for the evidence.
    pub fn driver(&self) -> &str {
        &self.dev.driver
    }

    /// The device's UUID: what `Options::device_uuid` names and the tuned file records.
    pub fn device_uuid(&self) -> [u8; 16] {
        self.dev.uuid
    }

    /// The host memory the expert copies and the embedding hold: with the MTP, `blk.40`'s experts too.
    pub fn host_bytes(&self) -> u64 {
        self.host.bytes()
    }

    /// The host-cached heap's usage by this process now (the driver's budget reading): what `pool_on_machine` takes
    /// as `used_after_load` once the load is done.
    pub fn host_heap_usage(&self) -> Result<u64, Error> {
        Ok(self.dev.heap_budget(vk::MemKind::HostCached)?.usage)
    }

    /// The VRAM the MTP takes beyond the load without the MTP: `blk.40`'s non-expert weights and the session's MTP
    /// rows and ring (`mtp_session_bytes`, which the cache was sized around). 0 without `Options::mtp`.
    pub fn mtp_vram_bytes(&self) -> u64 {
        self.mtp.map_or(0, |o| {
            self.w.mtp_bytes + mtp_session_bytes(&self.dims, o.rows, o.ring)
        })
    }

    /// (asked, used) when the load cut `mtp_ring` to fit the prefill chunk: a pass needs a RoPE table per row in the
    /// chunk's buffers, so the ring is at most the chunk minus 16, rounded down to 16.
    pub fn mtp_ring_clamp(&self) -> Option<(usize, usize)> {
        self.mtp_ring_clamp
    }

    /// The MTP options the engine was opened with, if any.
    pub fn mtp(&self) -> Option<MtpOptions> {
        self.mtp
    }

    /// The VRAM the weights' arena takes: the non-expert weights, the slot table and the scratch buffers. Equals
    /// `need(model, mtp).vram`.
    pub fn vram_bytes(&self) -> u64 {
        self.w.arena.bytes()
    }

    /// The VRAM prefill's buffers take: what their arena allocated, which `prefill_bytes` predicts.
    pub fn prefill_bytes(&self) -> u64 {
        self.prefill.arena.bytes()
    }

    /// Tokens per prefill chunk, at most.
    pub fn prefill_chunk(&self) -> usize {
        self.prefill.chunk
    }

    /// Query-key pairs one prefill chunk's attention may take: `ATTN_PAIRS_CM` where it runs `attn_prefill_cm`, else
    /// `ATTN_PAIRS`. `chunk_lengths` takes it as its cap.
    pub fn attn_pairs(&self) -> usize {
        if self.coop_attn {
            ATTN_PAIRS_CM
        } else {
            ATTN_PAIRS
        }
    }

    /// The kernels prefill's dense projections, grouped experts and attention run on: "split-f16" (cooperative matrices
    /// with every input split into two f16s) or "f32" (a device without them, or
    /// `Options::f32_prefill`).
    /// Where it returns "split-f16", attention runs `attn_prefill_cm` only where the head fits
    /// (`kernels::dispatch::attn_prefill_cm_fits`), else `attn_prefill`.
    pub fn prefill_kernels(&self) -> &'static str {
        if self.coop {
            "split-f16"
        } else {
            "f32"
        }
    }

    /// Why prefill isn't on the split-f16 kernels on a device that lists their shape: the error building one of them
    /// (`Kernels::coop_error`). None when they were built, or when the device lacks the shape or
    /// the shared memory.
    pub fn coop_error(&self) -> Option<&str> {
        self.k.coop_error()
    }

    /// Whether prefill has staging buffers.
    pub fn staging_on(&self) -> bool {
        self.staging.is_some()
    }

    /// Whether decode reads every routed expert from VRAM, for the base time (`Options::base_time`).
    pub fn base_time(&self) -> bool {
        self.base.is_some()
    }

    /// The slot table decode's expert kernels read: the CPU path's overlay while the share runs the path (`path_on`),
    /// the base time's remapped copy with `Options::base_time` (never both: `check_share`), else the slot table.
    pub(crate) fn expert_table(&self) -> u64 {
        if let Some(c) = self.cpu.borrow().as_ref() {
            return c.table.vram.addr;
        }
        self.base.as_ref().map_or(self.table.addr, |b| b.vram.addr)
    }

    /// Sets the share of HOST misses the CPU serves from the next token on (`Options::cpu_share`). Refuses a
    /// share outside 0.0 ..= 1.0 (`BadCpuShare`), one that runs the path with the base time (`BaseTimeAndCpuShare`), or
    /// one on a machine without what the path needs (`CpuPathUnavailable`), or one on a model whose experts are not all
    /// Q4_K, Q5_K, Q6_K or Q8_0 (`CpuShareExpertType`), keeping the previous share. From 0 to a
    /// share that runs the path starts the path (the mailbox, the clock calibration, the coordinator and its workers);
    /// → 0 stops it and joins its threads, so nothing spins while the GPU idles. A share below 1/512 assigns no expert
    /// and is taken as 0 (`cpu_miss::path_on`). Between steps only; it rebuilds no table the GPU reads.
    pub fn set_cpu_share(&self, share: f32) -> Result<(), Error> {
        cpu_miss::check_share(share, self.base.is_some())?;
        cpu_miss::check_path(share, cpu_miss::Path::probe(&self.dev))?;
        let expert_types: Vec<_> = self
            .w
            .layers
            .iter()
            .flat_map(|gl| [gl.slot.gate, gl.slot.up, gl.slot.down])
            .collect();
        cpu_miss::check_expert_types(share, &expert_types)?;
        let mut cpu = self.cpu.borrow_mut();
        let running = cpu.is_some();
        match (cpu_miss::path_on(share), running) {
            (true, true) => cpu.as_ref().expect("running").set_share(share),
            (true, false) => {
                let d = &self.dims;
                let lays = self.w.layers.iter().map(|gl| gl.slot).collect();
                let slots = (0..d.n_layer)
                    .map(|l| {
                        (0..d.n_expert)
                            .map(|e| self.host.slot_ptr(l, e) as usize)
                            .collect()
                    })
                    .collect();
                let host = (0..d.n_layer)
                    .flat_map(|l| (0..d.n_expert).map(move |e| (l, e)))
                    .map(|(l, e)| self.host.slot_addr(l, e))
                    .collect();
                *cpu = Some(cpu_miss::CpuPath::start(
                    &self.dev,
                    &self.k,
                    d.n_layer,
                    d.n_expert,
                    d.n_embd,
                    d.n_expert_used,
                    lays,
                    slots,
                    host,
                    share,
                )?);
            }
            (false, true) => {
                let mut c = cpu.take().expect("running");
                if !c.shutdown() {
                    // Detached: its mailbox leaks with it, and the host copies when the engine drops.
                    self.cpu_detached.set(true);
                }
                let computed = c.take_computed();
                self.swaps.borrow_mut().stats.cpu_misses += computed;
                // No submit is running between steps; a failed one may still be, so the mailbox waits for it. If the
                // wait fails too, that step may still read the mailbox and the overlay table: both are leaked with the
                // path, as a detached coordinator's mailbox is, instead of freed under it.
                let idle = self.dev.wait(
                    vk::QueueKind::Compute,
                    self.dev.submitted(vk::QueueKind::Compute),
                );
                if idle.is_ok() {
                    drop(c);
                } else {
                    std::mem::forget(c);
                }
            }
            (false, false) => {}
        }
        Ok(())
    }

    /// The share the CPU serves (`set_cpu_share`); 0 while the path is off.
    pub fn cpu_share(&self) -> f32 {
        self.cpu.borrow().as_ref().map_or(0.0, |c| c.share())
    }

    /// Tests only: while `fail` holds, the CPU path's coordinator answers no token, as a dead pool would. No effect
    /// with the path off.
    pub fn cpu_fail_next(&self, fail: bool) {
        if let Some(c) = self.cpu.borrow().as_ref() {
            c.fail_next(fail);
        }
    }

    /// Tests only: the CPU path answers the next token's layer 0 late, once the GPU's wait has counted its timeout,
    /// and every later layer on time. No effect with the path off.
    pub fn cpu_stall_layer0_once(&self) {
        if let Some(c) = self.cpu.borrow().as_ref() {
            c.stall_layer0_once();
        }
    }

    /// Tests and diagnostics: each main layer's done flag in the CPU path's mailbox (the last seq the CPU answered);
    /// None with the path off.
    pub fn cpu_done_flags(&self) -> Option<Vec<u32>> {
        let cpu = self.cpu.borrow();
        let c = cpu.as_ref()?;
        Some((0..self.dims.n_layer).map(|l| c.done_flag(l)).collect())
    }

    /// Tests and diagnostics: the last decode step's skip count (the mailbox's second status word, read before the
    /// step zeroes it): the waits that did not spin because an earlier layer of the token had timed out. None with
    /// the path off.
    pub fn cpu_last_skips(&self) -> Option<u32> {
        self.cpu.borrow().as_ref().map(|c| c.last_skips())
    }

    /// Tests only: whether the last `cpu_stall_layer0_once` saw the GPU's wait time out within its window and then
    /// answered layer 0 late. None with the path off.
    pub fn cpu_late_answered(&self) -> Option<bool> {
        self.cpu.borrow().as_ref().map(|c| c.late_answered())
    }

    /// Diagnostics: layers the CPU path's coordinator gave up on after 2 s without their publish (a host-side bound;
    /// the GPU's 20 ms bound fires first). 0 with the path off.
    pub fn cpu_host_timeouts(&self) -> u64 {
        self.cpu.borrow().as_ref().map_or(0, |c| c.host_timeouts())
    }

    /// A session reset or a new session: with the CPU path on, its coordinator idle and the mailbox's flags zeroed,
    /// so the new session's sequence numbers start clean. Fails if waiting for the compute queue does.
    pub(crate) fn cpu_reset(&self) -> Result<(), Error> {
        if let Some(c) = self.cpu.borrow().as_ref() {
            c.reset(&self.dev)?;
        }
        Ok(())
    }

    /// Stage chunks of at least `tokens` tokens from the next step on (`Options::stage_from`); for measuring the
    /// threshold (`examples/stage_threshold.rs`). No result depends on it.
    pub fn set_stage_from(&self, tokens: usize) {
        self.stage_from.set(tokens);
    }

    /// Chunks of at least this many tokens are staged (`Options::stage_from`, `set_stage_from`).
    pub fn stage_from(&self) -> usize {
        self.stage_from.get()
    }

    /// The prefill table and copies for a chunk of `tokens`, from the table the chunk runs with (call it after
    /// `take_staged`), or None when the chunk isn't staged: no staging buffers, or fewer tokens than `stage_from`.
    pub(crate) fn stage_plan(&self, tokens: usize) -> Option<stage::StagePlan> {
        if !self.will_stage(tokens) {
            return None;
        }
        let st = self.staging.as_ref()?;
        let d = &self.dims;
        Some(stage::plan(
            &self.mirror.borrow(),
            d.n_layer,
            d.n_expert,
            [st.bufs[0].addr, st.bufs[1].addr],
            st.stride,
            st.cap,
            |l, a| self.host.slot_at(l, a).is_some(),
        ))
    }

    /// Whether a prefill chunk of `tokens` is staged: the engine has staging buffers and the chunk is at least
    /// `stage_from` long.
    pub(crate) fn will_stage(&self, tokens: usize) -> bool {
        self.staging.is_some() && tokens >= self.stage_from.get()
    }

    /// Layer `l`'s staging copy for a chunk at `pos`: every expert `s` stages in the layer, from
    /// its host copy into its place in buffer `l % 2`, in one transfer submit that first waits for compute value
    /// `after`, the submit that last read that buffer. Returns the transfer value the layer's compute waits for; 0 when
    /// the layer stages nothing. A failure names the position and the layer.
    pub(crate) fn stage_layer(
        &self,
        s: &stage::StagePlan,
        l: usize,
        pos: usize,
        after: u64,
    ) -> Result<u64, Error> {
        let fail = |source| Error::Step {
            pos,
            layer: Some(l),
            source,
        };
        if self.fail_staging_once.get() == Some(l) {
            self.fail_staging_once.set(None);
            return Err(fail(vk::Error::Timeout));
        }
        let st = self
            .staging
            .as_ref()
            .expect("a staged chunk has staging buffers");
        let regions: Vec<(vk::Slice, vk::Slice)> = s.copies[l]
            .iter()
            .map(|&(src, i)| {
                let from = self
                    .host
                    .slot_at(l, src)
                    .expect("the plan stages only host slots");
                (from, st.bufs[l % 2].sub(i as u64 * st.stride, from.size))
            })
            .collect();
        if regions.is_empty() {
            return Ok(0);
        }
        let opts = vk::StageOpts {
            poison: self.poison_staging.then_some(SWAP_POISON),
            delay: self.stage_delay.as_ref().map(|(_, d)| *d),
            timed: self.timing_on() && self.dev.can_time_copies(),
        };
        let v = self.dev.stage_async(&regions, after, opts).map_err(fail)?;
        if opts.timed {
            let step = self
                .timing
                .borrow()
                .as_ref()
                .map_or(0, |v| v.len().saturating_sub(1));
            self.copy_tags.borrow_mut().push(timing::CopyTag {
                value: v,
                step,
                pos,
                layer: l as u32,
            });
        }
        Ok(v)
    }

    /// What the engine holds as `expert`'s slot-table entry in `layer`, (address, where): the mirror staged chunks
    /// build their prefill table from. For tests and diagnostics; `slot_entry` reads the table itself.
    pub fn mirror_entry(&self, layer: usize, expert: usize) -> (u64, u32) {
        self.mirror.borrow()[layer * self.dims.n_expert + expert]
    }

    /// The mean bytes of one expert's slot over the main layers: what a miss costs in host-to-GPU traffic, the figure
    /// behind "MiB of misses per token".
    pub fn expert_bytes(&self) -> u64 {
        let n = self.slot_bytes.len() as u64;
        self.slot_bytes
            .iter()
            .sum::<u64>()
            .checked_div(n)
            .unwrap_or(0)
    }

    /// A fresh sequence of up to `ctx` positions. The budget is checked once it is allocated (between sessions).
    pub fn session(&self, ctx: usize) -> Result<Session, Error> {
        let s = Session::new(self, ctx)?;
        self.check_budget()?;
        Ok(s)
    }

    /// `expert`'s slot-table entry in `layer`: (address, where). For tests and diagnostics.
    pub fn slot_entry(&self, layer: usize, expert: usize) -> Result<(u64, u32), Error> {
        let at = ((layer * self.dims.n_expert + expert) * ENTRY_BYTES) as u64;
        let b = self.dev.download(self.table.sub(at, ENTRY_BYTES as u64))?;
        Ok(stage::entry_at(
            b.as_slice().try_into().expect("the download is one entry"),
        ))
    }

    /// Overwrites one slot-table entry at once, outside any step. Tests use this to send an expert down the refused CPU
    /// path; the engine itself changes the table only through `stage_slot_entry`.
    pub fn set_slot_entry(
        &self,
        layer: usize,
        expert: usize,
        addr: u64,
        loc: u32,
    ) -> Result<(), Error> {
        weights::assert_not_cached(&self.dims, layer, loc);
        check_slot_addr(addr, loc);
        let at = ((layer * self.dims.n_expert + expert) * ENTRY_BYTES) as u64;
        self.dev.upload(
            self.table.sub(at, ENTRY_BYTES as u64),
            &slot_entry(addr, loc),
        )?;
        self.mirror.borrow_mut()[layer * self.dims.n_expert + expert] = (addr, loc);
        Ok(())
    }

    /// Stages a change to `expert`'s slot-table entry in `layer`. It lands at the start of the next step,
    /// so no step sees a half-updated table. Restaging an entry before then replaces the earlier change. `layer` may be
    /// the MTP block's, `n_layer`.
    pub fn stage_slot_entry(
        &self,
        layer: usize,
        expert: usize,
        addr: u64,
        loc: u32,
    ) -> Result<(), Error> {
        if layer > self.dims.n_layer || expert >= self.dims.n_expert {
            return Err(Error::NoSuchEntry { layer, expert });
        }
        weights::assert_not_cached(&self.dims, layer, loc);
        check_slot_addr(addr, loc);
        let index = (layer * self.dims.n_expert + expert) as u32;
        let mut staged = self.staged.borrow_mut();
        match staged.iter_mut().find(|u| u.0 == index) {
            Some(u) => *u = (index, loc, addr),
            None => staged.push((index, loc, addr)),
        }
        Ok(())
    }

    /// Takes the staged changes for the step being recorded and writes them into the update list.
    pub(crate) fn take_staged(&self) -> Vec<(u32, u32, u64)> {
        let staged = std::mem::take(&mut *self.staged.borrow_mut());
        let list: Vec<u8> = staged
            .iter()
            .flat_map(|&(index, loc, addr)| slot_update(index, loc, addr))
            .collect();
        self.updates.write(0, &list);
        // The step about to be recorded applies these in its first submit: from it on, the table holds them. (If the
        // step fails first, the mirror leads the table until the next step's `take_staged`: build a staged chunk's
        // prefill table after this call.) `stage_slot_entry` bounds every index.
        let mut mirror = self.mirror.borrow_mut();
        for &(index, loc, addr) in &staged {
            mirror[index as usize] = (addr, loc);
        }
        staged
    }

    /// Puts back the changes of a step that failed (at any of its submits, which may have applied them), so the next
    /// step applies them. A change staged since then wins.
    pub(crate) fn restage(&self, back: Vec<(u32, u32, u64)>) {
        let mut staged = self.staged.borrow_mut();
        for u in back {
            if !staged.iter().any(|s| s.0 == u.0) {
                staged.push(u);
            }
        }
    }

    /// The bytes of `expert`'s host slot in `layer`. For tests and diagnostics.
    pub fn host_slot(&self, layer: usize, expert: usize) -> Vec<u8> {
        self.host.slot_bytes(layer, expert)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_new_has_the_servers_defaults() {
        let c = Config::new(PathBuf::from("model.gguf"));
        assert_eq!(c.model, PathBuf::from("model.gguf"));
        assert_eq!(c.device_uuid, None);
        assert_eq!(c.ctx, 65_536);
        assert_eq!(c.cache_slots, None);
        assert_eq!(c.force_swap_seed, None);
        assert_eq!(c.snapshot_pool_gib, 8);
        assert!(!c.f32_prefill);
        assert_eq!(c.speculation, SpecConfig::default());
    }

    #[test]
    fn must_wait_idle_is_false_only_for_a_device_loss() {
        assert!(!must_wait_idle(&vk::Error::DeviceLost { in_flight: None }));
        assert!(!must_wait_idle(&vk::Error::DeviceLost {
            in_flight: Some("matvec".into())
        }));
        assert!(must_wait_idle(&vk::Error::Timeout));
        assert!(must_wait_idle(&vk::Error::Loader("x".into())));
        assert!(must_wait_idle(&vk::Error::NoTimestamps));
    }
}
