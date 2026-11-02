//! The adapter from the engine's worker to the server's `Backend`: the
//! conversions between the two crates' mirrored types, the sink that hands the worker's events to the server, the
//! observer that writes the engine's lifecycle lines, and the counting that keeps one device loss one loss.
//!
//! Everything here is a channel send or an atomic: no method blocks, as the `Backend` contract requires. The worker
//! answers on its own thread, so the replies complete tokio `oneshot`s.
use engine::{Notify, UnloadCause, Worker, WorkerConfig, WorkerEvent};
use server::backend as seam;
use server::logs::{stats_line, LifecycleLine, Logger};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

const MIB: f64 = (1u64 << 20) as f64;
const GIB: f64 = (1u64 << 30) as f64;
/// How often the test hook's park looks at `stopping`.
const TEST_STALL_PERIOD: Duration = Duration::from_millis(50);

/// Called for a device loss that no request carried: `App::device_lost`, once the app exists.
pub type LossHook = Box<dyn Fn(&str) + Send + Sync>;

/// What `shutdown` started: the thread that stops the worker, and a channel it completes when it has.
struct Stopper {
    handle: JoinHandle<()>,
    done: Receiver<()>,
}

/// What the adapter counts for the watchdog and the stats route, all from callbacks the worker
/// already makes on its own thread. The engine keeps no counter of its own.
#[derive(Default)]
struct Gauge {
    /// Bumped whenever the worker calls the adapter: a sink is asked or sent to, a reply or an observer runs.
    ticks: AtomicU64,
    /// The engine's load progress (`engine::Config::load_ticks`): +1 per weight tensor uploaded and per layer of expert
    /// copies, so a load that is slow but moving is not a stall. `progress` adds it to `ticks`.
    load_ticks: Arc<AtomicU64>,
    /// Requests whose sink the worker has not yet dropped, and commands whose reply has not run or been dropped.
    pending: AtomicU32,
    /// The last answer to a `stats` command, kept whether or not its asker still waited.
    latest: Mutex<Option<seam::StatsCopy>>,
}

impl Gauge {
    fn tick(&self) {
        self.ticks.fetch_add(1, Ordering::SeqCst);
    }

    fn progress(&self) -> seam::Progress {
        seam::Progress {
            ticks: self
                .ticks
                .load(Ordering::SeqCst)
                .wrapping_add(self.load_ticks.load(Ordering::SeqCst)),
            pending: self.pending.load(Ordering::SeqCst),
        }
    }
}

/// One piece of pending work. It counts up when it is given, with no tick, and down, with a tick, on `Drop`, so a sink
/// the worker skips or drops, a reply that runs and one that never will (the command is dropped with the worker) all
/// end their pending.
struct PendingGuard(Arc<Gauge>);

impl PendingGuard {
    fn new(gauge: &Arc<Gauge>) -> PendingGuard {
        gauge.pending.fetch_add(1, Ordering::SeqCst);
        // No tick: work arriving says nothing of the worker, and under steady traffic a tick here would reset the
        // watchdog's clock for a worker stuck mid-decode. The watch starts its clock when pending leaves zero.
        PendingGuard(gauge.clone())
    }
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        self.0.pending.fetch_sub(1, Ordering::SeqCst);
        self.0.tick();
    }
}

/// The engine's worker as the server's `Backend`.
pub struct EngineBackend {
    /// `None` once `shutdown` has taken it. Locked only for the call it makes, which is a channel send.
    worker: Mutex<Option<Worker>>,
    /// The `Error { DeviceLost }` events a request's sink accepted that the observer has not yet matched with the
    /// worker's `DeviceLost` notice. Only `EngineSink::send` raises it and only the observer lowers it: a request
    /// carried that loss, so the server counts it, and the observer must not count it again. A failed `Load` notifies
    /// `LoadFailed`, never `DeviceLost`, so the load path can't raise it either.
    carried: Arc<AtomicU64>,
    /// Set when a shutdown begins: every `EngineSink` then reads closed, so the turn in flight ends `Cancelled` at its
    /// next token and the requests behind it are skipped.
    stopping: Arc<AtomicBool>,
    /// The unloads this adapter sent and the worker has not yet answered; `load` doesn't take its fast path past one.
    pending_unloads: Arc<AtomicUsize>,
    /// Whether the worker unloaded a model for the shutdown (`Unloaded { cause: Shutdown }`), which it sends only when
    /// something was loaded.
    unloaded_at_shutdown: Arc<AtomicBool>,
    hook: Arc<OnceLock<LossHook>>,
    stopper: Mutex<Option<Stopper>>,
    gauge: Arc<Gauge>,
    /// The test hook (`set_test_stall`): the token at which each request's sink parks the worker, 0 for no hook.
    test_stall: Arc<AtomicU32>,
    /// What the tuned file measured on: a load that cannot open the GPU it names says so.
    tuned: Option<TunedGpu>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl EngineBackend {
    /// Spawns the real worker (`Worker::spawn`) from the server's config. The worker loads on the first request or
    /// `load`.
    pub fn spawn(cfg: &server::Config, logger: Logger) -> anyhow::Result<EngineBackend> {
        let config = engine_config(cfg)?;
        let tuned = tuned_gpu(cfg);
        // The load's progress counter is the gauge's.
        let load_ticks = config.load_ticks.clone().unwrap_or_default();
        let mut worker = WorkerConfig::new(config);
        worker.idle_unload = idle_unload(cfg);
        worker.park_snapshots = cfg.park_snapshots;
        Ok(EngineBackend::build(
            |notify| {
                worker.notify = Some(notify);
                Worker::spawn(worker)
            },
            logger,
            load_ticks,
            // The real engine warns about an untested GPU, once for this backend (one per process).
            Some(Arc::new(AtomicBool::new(false))),
            tuned,
        ))
    }

    /// Any worker the caller builds, given the observer to install; the tests pass `Worker::spawn_with(FakeLoader ..)`.
    pub fn with_worker(make: impl FnOnce(Notify) -> Worker, logger: Logger) -> EngineBackend {
        EngineBackend::build(make, logger, Arc::default(), None, None)
    }

    /// `with_worker` with the counter the worker's loads add to (`engine::Config::load_ticks`) and, for the real
    /// engine, the flag that makes the observer warn once about an untested GPU (`None`: never warn).
    fn build(
        make: impl FnOnce(Notify) -> Worker,
        logger: Logger,
        load_ticks: Arc<AtomicU64>,
        untested_gpu_warned: Option<Arc<AtomicBool>>,
        tuned: Option<TunedGpu>,
    ) -> EngineBackend {
        // The `uptime_s` of the `stats` line the observer writes after an idle unload.
        let started = Instant::now();
        let carried = Arc::new(AtomicU64::new(0));
        let hook = Arc::new(OnceLock::new());
        let unloaded_at_shutdown = Arc::new(AtomicBool::new(false));
        let gauge = Arc::new(Gauge {
            load_ticks,
            ..Gauge::default()
        });
        let notify = observer(
            logger,
            started,
            carried.clone(),
            hook.clone(),
            unloaded_at_shutdown.clone(),
            gauge.clone(),
            untested_gpu_warned,
            tuned.clone(),
        );
        EngineBackend {
            worker: Mutex::new(Some(make(notify))),
            carried,
            stopping: Arc::new(AtomicBool::new(false)),
            pending_unloads: Arc::new(AtomicUsize::new(0)),
            unloaded_at_shutdown,
            hook,
            stopper: Mutex::new(None),
            gauge,
            test_stall: Arc::new(AtomicU32::new(0)),
            tuned,
        }
    }

    /// The hidden test hook (`--test-stall-at-token`): from now on, every request's sink parks the worker
    /// thread at the request's `at_token`th `Token` event, before it forwards the token or ticks, until a shutdown
    /// begins. To the watchdog that is a stall. `None`, or 0, turns the hook off.
    pub fn set_test_stall(&self, at_token: Option<u32>) {
        self.test_stall
            .store(at_token.unwrap_or(0), Ordering::SeqCst);
    }

    /// Called once, after `App::new`: the losses no request carried go here (`app.device_lost`).
    pub fn set_loss_hook(&self, hook: LossHook) {
        if self.hook.set(hook).is_err() {
            tracing::warn!("the loss hook was already set; the first one stays");
        }
    }

    /// After `serve` returns: waits, at most `cap`, for the worker thread a `shutdown` started. When no shutdown
    /// came (`serve` returns without its stop when its listener fails) it starts one.
    pub fn finish(&self, cap: Duration) {
        // Always: it starts a stop when none came and does nothing when the worker is already taken, and a stop in
        // progress has stored its stopper by the time this returns (`shutdown` stores it under the worker's lock).
        drop(seam::Backend::shutdown(self));
        let Some(stopper) = lock(&self.stopper).take() else {
            return;
        };
        // `JoinHandle` has no timed join: the stopper completes a channel, and the handle is joined once it has. A
        // stopper that died without completing it disconnects the channel, which is as good as done.
        match stopper.done.recv_timeout(cap) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                let _ = stopper.handle.join();
            }
            Err(RecvTimeoutError::Timeout) => {
                tracing::warn!(
                    "the engine's worker did not stop within {} s; leaving it",
                    cap.as_secs()
                );
            }
        }
    }
}

impl seam::Backend for EngineBackend {
    fn submit(&self, req: seam::GenRequest, sink: seam::Sink) -> Result<(), seam::Full> {
        // Counted from here: a sink that is dropped unused (no worker, one the worker skips) counts down too.
        let sink = EngineSink {
            sink,
            carried: self.carried.clone(),
            stopping: self.stopping.clone(),
            test_stall: self.test_stall.clone(),
            tokens: 0,
            pending: PendingGuard::new(&self.gauge),
        };
        let req = to_request(req);
        match lock(&self.worker).as_ref() {
            Some(w) => w.submit(req, Box::new(sink)).map_err(|_| seam::Full),
            None => Err(seam::Full),
        }
    }

    fn state(&self) -> seam::Lifecycle {
        lock(&self.worker)
            .as_ref()
            .map_or(seam::Lifecycle::Idle, |w| lifecycle(w.state()))
    }

    fn load(&self) -> seam::Reply<()> {
        let (tx, rx) = oneshot::channel();
        let guard = lock(&self.worker);
        let Some(w) = guard.as_ref() else {
            // The sender drops: the server reads a worker that is gone.
            return rx;
        };
        // The worker answers a `Load` in order, behind a running request. A loaded model needs no such wait
        // unless an unload this adapter sent is still on its way: the state reads `Ready` until it ends.
        if w.state() == engine::Lifecycle::Ready && self.pending_unloads.load(Ordering::SeqCst) == 0
        {
            let _ = tx.send(Ok(()));
            return rx;
        }
        // On `Gone` the command, and the sender and the guard in it, is dropped.
        let pending = PendingGuard::new(&self.gauge);
        let tuned = self.tuned.clone();
        let _ = w.load(Box::new(move |answer| {
            drop(pending);
            let _ = tx.send(match answer {
                Ok(_) => Ok(()),
                Err(e) => Err(seam::BackendError {
                    kind: error_kind(e.kind),
                    message: tuned_load_failure(tuned.as_ref(), e.message, false),
                }),
            });
        }));
        rx
    }

    fn unload(&self) -> seam::Reply<bool> {
        let (tx, rx) = oneshot::channel();
        let guard = lock(&self.worker);
        let Some(w) = guard.as_ref() else {
            // A worker that was stopped has nothing loaded; a dropped reply would read as an engine that died.
            let _ = tx.send(Ok(false));
            return rx;
        };
        self.pending_unloads.fetch_add(1, Ordering::SeqCst);
        let pending = self.pending_unloads.clone();
        let counted = PendingGuard::new(&self.gauge);
        let sent = w.unload(Box::new(move |info| {
            pending.fetch_sub(1, Ordering::SeqCst);
            drop(counted);
            let _ = tx.send(Ok(info.was_loaded));
        }));
        if sent.is_err() {
            // The command, and the reply in it, never ran.
            self.pending_unloads.fetch_sub(1, Ordering::SeqCst);
        }
        rx
    }

    fn stats(&self) -> seam::Reply<seam::BackendStats> {
        let (tx, rx) = oneshot::channel();
        let guard = lock(&self.worker);
        match guard.as_ref() {
            None => {
                let _ = tx.send(Ok(seam::BackendStats::default()));
            }
            Some(w) => {
                let pending = PendingGuard::new(&self.gauge);
                let gauge = self.gauge.clone();
                let _ = w.stats(Box::new(move |s| {
                    let stats = to_stats(&s);
                    // Kept whether or not anyone still waits for the answer: `at` is when the worker gave it.
                    *lock(&gauge.latest) = Some(seam::StatsCopy {
                        stats: stats.clone(),
                        at: Instant::now(),
                    });
                    drop(pending);
                    let _ = tx.send(Ok(stats));
                }));
            }
        }
        rx
    }

    fn progress(&self) -> Option<seam::Progress> {
        Some(self.gauge.progress())
    }

    fn latest_stats(&self) -> Option<seam::StatsCopy> {
        lock(&self.gauge.latest).clone()
    }

    /// Unloads without parking and joins the worker's thread, on a thread of its own: `Worker::shutdown`
    /// joins, and no `Backend` method may block. Setting `stopping` first ends the turn in flight at its next token;
    /// the worker's own `Shutdown` is a command behind that turn.
    fn shutdown(&self) -> seam::Reply<bool> {
        let (tx, rx) = oneshot::channel();
        self.stopping.store(true, Ordering::SeqCst);
        let mut guard = lock(&self.worker);
        let Some(worker) = guard.take() else {
            let _ = tx.send(Ok(false));
            return rx;
        };
        let unloaded = self.unloaded_at_shutdown.clone();
        let (done_tx, done) = channel();
        let handle = std::thread::Builder::new()
            .name("artemis-stop".into())
            .spawn(move || {
                // A worker that panicked answers `Gone`; either way it has ended.
                let _ = worker.shutdown();
                // The reply says whether the shutdown unloaded a model: a worker that was mid-way through an idle
                // park has none left, and the idle line is already written.
                let _ = tx.send(Ok(unloaded.load(Ordering::SeqCst)));
                let _ = done_tx.send(());
            })
            .expect("spawn the stop thread");
        // Stored before the worker's lock is released, so a `finish` that finds the worker gone finds the stopper.
        *lock(&self.stopper) = Some(Stopper { handle, done });
        drop(guard);
        rx
    }
}

/// What the worker's `EventSink` is: the server's `Sink`, with the loss counting and the stop.
struct EngineSink {
    sink: seam::Sink,
    carried: Arc<AtomicU64>,
    stopping: Arc<AtomicBool>,
    /// The adapter's test hook, read at each token.
    test_stall: Arc<AtomicU32>,
    /// The `Token` events this request has sent so far, for the hook.
    tokens: u32,
    /// Dropped with the sink, which the worker does after the reply-end snapshot; it holds the gauge the sink ticks.
    pending: PendingGuard,
}

impl EngineSink {
    /// The hook: at the request's Nth token, holds the worker thread until `stopping` is set. The wait is a timed park
    /// that nothing unparks, so it ends within one period of the flag and the stop can join the thread.
    fn test_stall_at(&mut self) {
        self.tokens += 1;
        let at = self.test_stall.load(Ordering::SeqCst);
        if at == 0 || self.tokens != at {
            return;
        }
        while !self.stopping.load(Ordering::SeqCst) {
            std::thread::park_timeout(TEST_STALL_PERIOD);
        }
    }
}

impl engine::EventSink for EngineSink {
    fn send(&mut self, event: engine::Event) -> bool {
        // Before the tick: a worker held here makes no progress.
        if matches!(event, engine::Event::Token { .. }) {
            self.test_stall_at();
        }
        self.pending.0.tick();
        let lost = matches!(
            event,
            engine::Event::Error {
                kind: engine::ErrorKind::DeviceLost,
                ..
            }
        );
        let sent = self.sink.send(to_event(event));
        // A loss a request's receiver was there for is the server's to count.
        if lost && sent {
            self.carried.fetch_add(1, Ordering::SeqCst);
        }
        sent
    }

    fn closed(&self) -> bool {
        // Asked before the request starts and before each prefill segment, so it is a tick for each.
        self.pending.0.tick();
        self.sink.closed() || self.stopping.load(Ordering::SeqCst)
    }
}

/// The observer the worker calls, on its own thread, with each lifecycle event. Each argument is state it keeps for
/// one backend, handed over once at `build`.
#[allow(clippy::too_many_arguments)]
fn observer(
    logger: Logger,
    started: Instant,
    carried: Arc<AtomicU64>,
    hook: Arc<OnceLock<LossHook>>,
    unloaded_at_shutdown: Arc<AtomicBool>,
    gauge: Arc<Gauge>,
    untested_gpu_warned: Option<Arc<AtomicBool>>,
    tuned: Option<TunedGpu>,
) -> Notify {
    let kept = gauge.clone();
    let mut handle: Notify = Box::new(move |event| match event {
        WorkerEvent::Loaded(info) => {
            logger.lifecycle(&load_line(&info));
            let warning = untested_gpu_warned
                .as_deref()
                .and_then(|warned| untested_gpu_warning_once(warned, &info.device));
            if let Some(message) = warning {
                logger.lifecycle(&LifecycleLine::ConfigWarning { message });
            }
            // Each load: the tuned values were measured on one GPU and driver.
            let differs = tuned
                .as_ref()
                .and_then(|t| tuned_gpu_warning(t, &info.device_uuid, &info.driver));
            if let Some(message) = differs {
                logger.lifecycle(&LifecycleLine::ConfigWarning { message });
            }
        }
        WorkerEvent::Unloaded { cause } => match cause {
            UnloadCause::Idle => {
                logger.lifecycle(&LifecycleLine::Unload {
                    cause: "idle".into(),
                });
                // From the copy this adapter holds: the worker is the thread running this, so it can't ask itself.
                // `uptime_s` counts from `with_worker`; the server's own lines count from `App::new`, a little later.
                let copy = lock(&kept.latest).clone();
                logger.lifecycle(&stats_line(
                    copy.as_ref(),
                    started.elapsed().as_secs(),
                    logger.write_failures(),
                ));
            }
            // The server writes the lines for the unloads it asked for.
            UnloadCause::Shutdown => unloaded_at_shutdown.store(true, Ordering::SeqCst),
            UnloadCause::Requested => {}
        },
        WorkerEvent::DeviceLost => {
            // The worker sends a request's `Error` before this notice, on the same thread, so the count and the notice
            // can't cross. A loss a request carried is already counted: use up its credit.
            let carried_before =
                carried.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1));
            if carried_before.is_err() {
                match hook.get() {
                    Some(hook) => hook("the GPU was lost after a reply; no request carried it"),
                    None => tracing::warn!("the GPU was lost, and no loss hook is set to count it"),
                }
            }
        }
        // The engine's own messages, never a request's text. The logger's warning is the stderr twin.
        WorkerEvent::LoadFailed { message } => logger.lifecycle(&LifecycleLine::LoadFailed {
            message: tuned_load_failure(tuned.as_ref(), message, true),
        }),
        WorkerEvent::ReloadFailed { message } => logger.lifecycle(&LifecycleLine::ReloadFailed {
            message: tuned_load_failure(tuned.as_ref(), message, true),
        }),
    });
    // A load, an unload or a loss that ends is movement: a load shows no other progress.
    Box::new(move |event| {
        gauge.tick();
        handle(event);
    })
}

/// The engine's config from the server's: `device_uuid` through `engine::parse_uuid`, the unsigned counts widened.
/// A bad UUID is an error that names the key and never shows the value.
pub fn engine_config(cfg: &server::Config) -> anyhow::Result<engine::Config> {
    let Some(model) = cfg.model.clone() else {
        anyhow::bail!("model is not set: name the GGUF in the config or with --model");
    };
    let mut e = engine::Config::new(model);
    if let Some(uuid) = &cfg.device_uuid {
        e.device_uuid = Some(engine::parse_uuid(uuid).ok_or_else(|| {
            anyhow::anyhow!(
                "device_uuid is not a UUID: it is 32 hex digits, with or without dashes"
            )
        })?);
    }
    e.ctx = cfg.ctx as usize;
    e.cache_slots = cfg.cache_slots.map(|n| n as usize);
    e.snapshot_pool_gib = cfg.snapshot_pool_gib;
    e.f32_prefill = cfg.f32_prefill;
    e.speculation = spec_config(&cfg.speculation);
    e.load_ticks = Some(Arc::default());
    Ok(e)
}

/// The engine's draft sources from the table's names, refused by the engine's own rules.
pub fn spec_sources(
    t: &server::config::SpeculationConfig,
) -> Result<Vec<engine::spec::Source>, (&'static str, String)> {
    let names: Vec<&str> = t.sources.iter().map(String::as_str).collect();
    engine::spec::Source::parse_list(&names)
}

/// The engine's speculation settings from the server's `[speculation]` table (which `Config::load` has checked, so
/// `sources` and `source_choice` parse; they `expect` that rather than fall back to a default).
pub fn spec_config(t: &server::config::SpeculationConfig) -> engine::SpecConfig {
    engine::SpecConfig {
        enabled: t.enabled,
        max_draft: t.max_draft as usize,
        ngram_min: t.ngram_min as usize,
        ngram_max: t.ngram_max as usize,
        min_gain: t.min_gain,
        accept_half_life: t.accept_half_life,
        cost_slope_ms: t.cost_slope_ms,
        accept_prior: t.accept_prior,
        fixed_draft: None,
        sources: spec_sources(t)
            .expect("Config::load checks the table, and its rules for sources are the engine's"),
        probe_every: t.probe_every as usize,
        mtp_window: t.mtp_window as usize,
        mtp_ctx: t.mtp_ctx as usize,
        mtp_ring: t.mtp_ring as usize,
        mtp_accept_prior: t.mtp_accept_prior.clone(),
        mtp_cost_ms: t.mtp_cost_ms,
        source_choice: engine::SourceChoice::parse(&t.source_choice).expect(
            "Config::load checks the table, and its rule for source_choice is the engine's",
        ),
        mtp_head_rows: t.mtp_head_rows as usize,
    }
}

/// `idle_unload_s`: 0 is never.
pub fn idle_unload(cfg: &server::Config) -> Option<Duration> {
    (cfg.idle_unload_s > 0).then(|| Duration::from_secs(cfg.idle_unload_s))
}

pub fn to_request(r: seam::GenRequest) -> engine::Request {
    engine::Request {
        prompt: r.prompt,
        cut: r.cut.map(|c| engine::Cut {
            at: c.at,
            user_query: c.user_query,
        }),
        sampling: engine::Sampling {
            temp: r.sampling.temp,
            top_k: r.sampling.top_k,
            top_p: r.sampling.top_p,
            min_p: r.sampling.min_p,
            presence_penalty: r.sampling.presence_penalty,
        },
        seed: r.seed,
        max_tokens: r.max_tokens,
        stop_tokens: r.stop_tokens,
        think: r.think.map(|t| engine::ThinkLimit {
            open: t.open,
            budget: t.budget,
            ends: t.ends.to_vec(),
            close: t.close,
        }),
        logprobs: r.logprobs,
    }
}

/// A duration as the log's milliseconds.
fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

pub fn to_event(e: engine::Event) -> seam::GenEvent {
    match e {
        engine::Event::Started {
            prompt_tokens,
            reused_tokens,
            restored,
            diverged_at,
        } => seam::GenEvent::Started {
            prompt_tokens,
            reused_tokens,
            restored: source(restored),
            diverged_at,
        },
        engine::Event::Token {
            id,
            forced,
            logprobs,
        } => seam::GenEvent::Token {
            id,
            forced,
            logprobs: logprobs.map(|l| seam::TokenLogprobs {
                logprob: l.logprob,
                top: l.top,
            }),
        },
        engine::Event::Done {
            finish,
            counts,
            timings,
            cache,
            restore_bytes,
            pool_bytes_used,
            spec,
        } => seam::GenEvent::Done {
            finish: finish_of(finish),
            counts: seam::Counts {
                prompt: counts.prompt,
                reused: counts.reused,
                prefilled: counts.prefilled,
                completion: counts.completion,
                reasoning: counts.reasoning,
            },
            // `first_token` is queued to first token, so it includes the queue and the load; the fake's is measured
            // from the dequeue. It is for the logs only.
            timings: seam::Timings {
                queue_ms: ms(timings.queue),
                load_ms: ms(timings.load),
                restore_ms: ms(timings.restore),
                prefill_ms: ms(timings.prefill),
                decode_ms: ms(timings.decode),
                ttft_ms: ms(timings.first_token),
            },
            // The cache's other counts have no counterpart and are dropped.
            cache: seam::CacheReport {
                hit_rate: cache.hit_rate,
                miss_mib_per_token: cache.miss_mib_per_token,
                restore_bytes,
                pool_bytes_used,
                decode_misses: Some(cache.decode_misses),
            },
            spec: spec.map(|d| seam::SpecReport {
                steps: d.steps,
                plain_steps: d.plain_steps,
                drafted: d.drafted,
                accepted: d.accepted,
                replays: d.replays,
                drafted_lookup: d.drafted_lookup,
                accepted_lookup: d.accepted_lookup,
                drafted_mtp: d.drafted_mtp,
                accepted_mtp: d.accepted_mtp,
            }),
        },
        engine::Event::Error { kind, message } => seam::GenEvent::Error {
            kind: error_kind(kind),
            message,
        },
    }
}

fn source(s: engine::Source) -> seam::Source {
    match s {
        engine::Source::None => seam::Source::None,
        engine::Source::Live => seam::Source::Live,
        engine::Source::Interval => seam::Source::Interval,
        engine::Source::PromptEnd => seam::Source::PromptEnd,
        engine::Source::UserQuery => seam::Source::UserQuery,
        engine::Source::ReplyEnd => seam::Source::ReplyEnd,
    }
}

fn finish_of(f: engine::Finish) -> seam::Finish {
    match f {
        engine::Finish::Stop(id) => seam::Finish::Stop(id),
        engine::Finish::Length => seam::Finish::Length,
        engine::Finish::ContextFull => seam::Finish::ContextFull,
        engine::Finish::Cancelled => seam::Finish::Cancelled,
        engine::Finish::Unloaded => seam::Finish::Unloaded,
    }
}

fn error_kind(k: engine::ErrorKind) -> seam::ErrorKind {
    match k {
        engine::ErrorKind::DeviceLost => seam::ErrorKind::DeviceLost,
        engine::ErrorKind::NonFinite => seam::ErrorKind::NonFinite,
        engine::ErrorKind::OutOfMemory => seam::ErrorKind::OutOfMemory,
        engine::ErrorKind::LoadFailed => seam::ErrorKind::LoadFailed,
        engine::ErrorKind::Internal => seam::ErrorKind::Internal,
    }
}

fn lifecycle(l: engine::Lifecycle) -> seam::Lifecycle {
    match l {
        engine::Lifecycle::Idle => seam::Lifecycle::Idle,
        engine::Lifecycle::Loading => seam::Lifecycle::Loading,
        engine::Lifecycle::Ready => seam::Lifecycle::Ready,
        engine::Lifecycle::Recovering => seam::Lifecycle::Recovering,
    }
}

/// The nine reuse counters of the same names, the two the seam gained for the engine's `park_errors` and the pool's
/// `chunks_refused`, and the worker's own counts and the expert cache's (`None` while unloaded).
pub fn to_stats(s: &engine::WorkerStats) -> seam::BackendStats {
    let r = &s.reuse;
    seam::BackendStats {
        snapshots_held: r.snapshots_held,
        pool_bytes_used: r.pool_bytes_used,
        pool_bytes_limit: r.pool_bytes_limit,
        restores: r.restores,
        restore_bytes: r.restore_bytes,
        reused_tokens: r.reused_tokens,
        prefilled_tokens: r.prefilled_tokens,
        evicted_protected: r.evicted_protected,
        evicted_unprotected: r.evicted_unprotected,
        park_errors: s.park_errors,
        chunks_refused: r.chunks_refused,
        loads: s.loads,
        device_losses: s.device_losses,
        after_done_errors: s.after_done_errors,
        parked_snapshots: u64::try_from(s.parked).unwrap_or(u64::MAX),
        spec: s.spec.as_ref().map(spec_counters),
        cache: s.cache.map(|c| seam::CacheCounters {
            slots_per_layer: u64::try_from(c.slots_per_layer).unwrap_or(u64::MAX),
            hits: c.hits,
            misses: c.misses,
            staged: c.staged,
            cpu_misses: c.cpu_misses,
            cpu_timeouts: c.cpu_timeouts,
            longest_submit_us: c.longest_submit_us,
        }),
    }
}

/// The speculation totals as the seam's counters. `by_size` keeps the sizes that ran: row `i` of the engine's table is
/// the step size `t = i + 1`. The means are per step. "lookup" is lookup's accept-rate estimate. The MTP figures are
/// listed once MTP has drafted (before that they would only be its prior): "mtp" is its depth-1 estimate,
/// `mtp_depth_rates` the estimates for depths 1 up to the deepest it has tested, and `mtp_draft` the draft passes by
/// `k` (row `k - 1` of the engine's table), with the mean time per pass.
fn spec_counters(s: &engine::spec::SpecTotals) -> seam::SpecCounters {
    let d = &s.delta;
    let mtp_ran = s.mtp_by_k.iter().any(|r| r.steps > 0);
    seam::SpecCounters {
        steps: d.steps,
        plain_steps: d.plain_steps,
        drafted: d.drafted,
        accepted: d.accepted,
        replays: d.replays,
        accept_rate: std::iter::once(("lookup".to_string(), s.accept_rate))
            .chain(mtp_ran.then(|| ("mtp".to_string(), s.mtp_accept_rate)))
            .collect(),
        mtp_depth_rates: if mtp_ran {
            let deepest = s
                .mtp_depth_counts
                .iter()
                .rposition(|c| c.0 > 0)
                .map_or(1, |i| i + 1);
            s.depth_rates[..deepest].to_vec()
        } else {
            Vec::new()
        },
        mtp_draft: s
            .mtp_by_k
            .iter()
            .enumerate()
            .filter(|(_, r)| r.steps > 0)
            .map(|(i, r)| seam::MtpDraftRow {
                k: u32::try_from(i + 1).unwrap_or(u32::MAX),
                passes: r.steps,
                mean_ms: r.ms / r.steps as f64,
            })
            .collect(),
        by_size: s
            .by_size
            .iter()
            .enumerate()
            .filter(|(_, r)| r.steps > 0)
            .map(|(i, r)| {
                let steps = r.steps as f64;
                seam::StepSize {
                    t: u32::try_from(i + 1).unwrap_or(u32::MAX),
                    steps: r.steps,
                    mean_ms: r.ms / steps,
                    mean_miss_mib: r.miss_bytes as f64 / MIB / steps,
                    tokens: r.tokens,
                }
            })
            .collect(),
    }
}

/// What the tuned file measured on, from `TunedState::Loaded`: the GPU's name, UUID and driver, which each load is
/// compared with, and whether the file names the GPU (`device_uuid` among the keys it applied).
#[derive(Clone, Debug, PartialEq)]
pub struct TunedGpu {
    pub path: PathBuf,
    pub gpu_name: String,
    pub gpu_uuid: String,
    pub driver: String,
    pub names_device: bool,
}

/// The tuned file's GPU, when `cfg` loaded one.
pub fn tuned_gpu(cfg: &server::Config) -> Option<TunedGpu> {
    match &cfg.tuned {
        server::TunedState::Loaded {
            path,
            applied,
            gpu_name,
            gpu_uuid,
            driver,
            ..
        } => Some(TunedGpu {
            path: path.clone(),
            gpu_name: gpu_name.clone(),
            gpu_uuid: gpu_uuid.clone(),
            driver: driver.clone(),
            names_device: applied.iter().any(|k| k == "device_uuid"),
        }),
        _ => None,
    }
}

/// The `config_warning` text when a load's device or driver is not the one the tuned file was measured on.
pub fn tuned_gpu_warning(t: &TunedGpu, uuid: &[u8; 16], driver: &str) -> Option<String> {
    let same_gpu = engine::parse_uuid(&t.gpu_uuid).is_some_and(|u| &u == uuid);
    (!same_gpu || t.driver != driver).then(|| {
        format!(
            "The tuned settings were measured on {} with driver {}; this GPU or driver differs. Run artemis setup \
             --force to measure again.",
            t.gpu_name, t.driver
        )
    })
}

/// The words of the engine's error for a GPU that is not there (`vk::Error::NoDevice`).
const NO_DEVICE: &str = "no usable discrete GPU";

/// A load's failure `message`, with the way out added when the tuned file names the GPU and that GPU cannot be opened:
/// the file is the thing to measure again. The log names the file's path (`in_log`); a reply to an API client says
/// "the tuned settings file" and keeps the local path to itself.
pub fn tuned_load_failure(t: Option<&TunedGpu>, message: String, in_log: bool) -> String {
    match t {
        Some(t) if t.names_device && message.contains(NO_DEVICE) => {
            let file = if in_log {
                format!("the tuned file {}", t.path.display())
            } else {
                "the tuned settings file".to_string()
            };
            format!(
                "{message}; the GPU is named by device_uuid in {file}. Run artemis setup --force to measure again"
            )
        }
        _ => message,
    }
}

/// The GPUs whose names the engine's tests have run on (`LoadInfo::device`).
pub const TESTED_GPUS: &[&str] = &["AMD Radeon RX 9070 XT"];

/// Whether a load on `device` warrants a warning: any GPU not in `TESTED_GPUS`.
pub fn untested_gpu_warning(device: &str) -> Option<String> {
    (!TESTED_GPUS.contains(&device)).then(|| {
        format!(
            "this GPU ({device}) has not run Artemis's tests; the tested GPU is the AMD Radeon RX 9070 XT"
        )
    })
}

/// `untested_gpu_warning`, once per `warned` flag, so a reload after a device loss does not repeat it.
fn untested_gpu_warning_once(warned: &AtomicBool, device: &str) -> Option<String> {
    let message = untested_gpu_warning(device)?;
    (!warned.swap(true, Ordering::SeqCst)).then_some(message)
}

/// The `load` line: the time and the device, and the load's figures.
pub fn load_line(info: &engine::LoadInfo) -> LifecycleLine {
    let retried = match info.retried_with_slots {
        Some(n) => format!("retried with {n} slots"),
        None => "first try".to_string(),
    };
    LifecycleLine::Load {
        seconds: info.time.as_secs_f64(),
        detail: format!("{}; {retried}", info.device),
        slots_per_layer: Some(u32::try_from(info.slots_per_layer).unwrap_or(u32::MAX)),
        vram_used_mib: info.vram_used.map(|b| b as f64 / MIB),
        pool_limit_gib: Some(info.pool.bytes as f64 / GIB),
        pool_cut: Some(info.pool.cut),
        unparked: Some(info.unparked.restored as u64),
    }
}

// The adapter is shared with the server's tasks as `Arc<dyn Backend>`.
const _: fn() = || {
    fn is<T: Send + Sync + 'static>() {}
    is::<EngineBackend>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{PoolLimit, Unparked};
    use server::backend::{Cut, GenRequest, Sampling, ThinkLimit};

    #[test]
    fn requests_convert_field_for_field() {
        let seam_req = GenRequest {
            prompt: vec![5, 6, 7, 8],
            cut: Some(Cut {
                at: 3,
                user_query: true,
            }),
            sampling: Sampling {
                temp: 0.25,
                top_k: 17,
                top_p: 0.5,
                min_p: 0.125,
                presence_penalty: 1.5,
            },
            seed: 0xDEAD_BEEF_0102,
            max_tokens: 99,
            stop_tokens: vec![11, 12],
            think: Some(ThinkLimit {
                open: true,
                budget: Some(40),
                ends: [21, 22],
                close: vec![31, 32, 33],
            }),
            logprobs: Some(4),
        };
        assert_eq!(
            to_request(seam_req),
            engine::Request {
                prompt: vec![5, 6, 7, 8],
                cut: Some(engine::Cut {
                    at: 3,
                    user_query: true,
                }),
                sampling: engine::Sampling {
                    temp: 0.25,
                    top_k: 17,
                    top_p: 0.5,
                    min_p: 0.125,
                    presence_penalty: 1.5,
                },
                seed: 0xDEAD_BEEF_0102,
                max_tokens: 99,
                stop_tokens: vec![11, 12],
                think: Some(engine::ThinkLimit {
                    open: true,
                    budget: Some(40),
                    ends: vec![21, 22],
                    close: vec![31, 32, 33],
                }),
                logprobs: Some(4),
            }
        );
        // The optional parts, absent.
        let bare = GenRequest {
            cut: None,
            think: None,
            logprobs: None,
            ..GenRequest {
                prompt: vec![1],
                cut: None,
                sampling: Sampling {
                    temp: 0.0,
                    top_k: 0,
                    top_p: 1.0,
                    min_p: 0.0,
                    presence_penalty: 0.0,
                },
                seed: 1,
                max_tokens: 1,
                stop_tokens: vec![],
                think: None,
                logprobs: None,
            }
        };
        let r = to_request(bare);
        assert_eq!((r.cut, r.think, r.logprobs), (None, None, None));
    }

    #[test]
    fn events_convert_and_build_the_cache_report() {
        use engine::{CacheDelta, Counts, Event, Finish, Source, Timings};
        assert_eq!(
            to_event(Event::Started {
                prompt_tokens: 9,
                reused_tokens: 4,
                restored: Source::UserQuery,
                diverged_at: 6,
            }),
            seam::GenEvent::Started {
                prompt_tokens: 9,
                reused_tokens: 4,
                restored: seam::Source::UserQuery,
                diverged_at: 6,
            }
        );
        assert_eq!(
            to_event(Event::Token {
                id: 70,
                forced: true,
                logprobs: Some(engine::TokenLogprobs {
                    logprob: -0.5,
                    top: vec![(70, -0.5), (71, -1.5)],
                }),
            }),
            seam::GenEvent::Token {
                id: 70,
                forced: true,
                logprobs: Some(seam::TokenLogprobs {
                    logprob: -0.5,
                    top: vec![(70, -0.5), (71, -1.5)],
                }),
            }
        );
        let done = to_event(Event::Done {
            finish: Finish::Stop(2),
            counts: Counts {
                prompt: 10,
                reused: 4,
                prefilled: 6,
                completion: 8,
                reasoning: 3,
            },
            timings: Timings {
                queue: Duration::from_millis(1),
                load: Duration::from_millis(2),
                restore: Duration::from_millis(3),
                prefill: Duration::from_millis(4),
                decode: Duration::from_millis(5),
                first_token: Duration::from_millis(12),
            },
            // The counts the seam has no place for are dropped; the rates and the two byte figures carry over.
            cache: CacheDelta {
                hits: 100,
                staged: 101,
                misses: 102,
                decode_misses: 103,
                decoded: 104,
                hit_rate: Some(0.75),
                miss_mib_per_token: Some(2.5),
            },
            restore_bytes: 4096,
            pool_bytes_used: 8192,
            spec: Some(engine::spec::SpecDelta {
                steps: 5,
                plain_steps: 2,
                drafted: 12,
                accepted: 9,
                replays: 3,
                drafted_lookup: 7,
                accepted_lookup: 6,
                drafted_mtp: 5,
                accepted_mtp: 3,
                probes: 1,
            }),
        });
        let seam::GenEvent::Done {
            finish,
            counts,
            timings,
            cache,
            spec,
        } = done
        else {
            panic!("not Done: {done:?}");
        };
        assert_eq!(finish, seam::Finish::Stop(2));
        assert_eq!(
            counts,
            seam::Counts {
                prompt: 10,
                reused: 4,
                prefilled: 6,
                completion: 8,
                reasoning: 3,
            }
        );
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(close(timings.queue_ms, 1.0) && close(timings.load_ms, 2.0));
        assert!(close(timings.restore_ms, 3.0) && close(timings.prefill_ms, 4.0));
        assert!(close(timings.decode_ms, 5.0));
        assert!(close(timings.ttft_ms, 12.0), "{}", timings.ttft_ms);
        assert_eq!(
            cache,
            seam::CacheReport {
                hit_rate: Some(0.75),
                miss_mib_per_token: Some(2.5),
                restore_bytes: 4096,
                pool_bytes_used: 8192,
                decode_misses: Some(103),
            }
        );
        assert_eq!(
            spec,
            Some(seam::SpecReport {
                steps: 5,
                plain_steps: 2,
                drafted: 12,
                accepted: 9,
                replays: 3,
                drafted_lookup: 7,
                accepted_lookup: 6,
                drafted_mtp: 5,
                accepted_mtp: 3,
            })
        );
        assert_eq!(
            to_event(Event::Error {
                kind: engine::ErrorKind::DeviceLost,
                message: "lost".into(),
            }),
            seam::GenEvent::Error {
                kind: seam::ErrorKind::DeviceLost,
                message: "lost".into(),
            }
        );
    }

    #[test]
    fn to_event_carries_decode_misses() {
        // The cache's counters differ, so a crossed pair shows; a request with none still has a figure, zero.
        let done = |misses, decode_misses| {
            let seam::GenEvent::Done { cache, .. } = to_event(engine::Event::Done {
                finish: engine::Finish::Length,
                counts: engine::Counts::default(),
                timings: engine::Timings::default(),
                cache: engine::CacheDelta {
                    misses,
                    decode_misses,
                    ..engine::CacheDelta::default()
                },
                restore_bytes: 0,
                pool_bytes_used: 0,
                spec: None,
            }) else {
                panic!("not Done");
            };
            cache.decode_misses
        };
        assert_eq!(done(500, 41), Some(41));
        assert_eq!(done(500, 0), Some(0));
    }

    #[test]
    fn stats_take_the_reuse_counters_and_park_errors() {
        // Distinct values in, so a crossed pair shows; the fields the seam doesn't carry hold others.
        let stats = engine::WorkerStats {
            state: engine::Lifecycle::Ready,
            cache: Some(engine::Stats {
                slots_per_layer: 101,
                cache_bytes: 190,
                staging_bytes: 191,
                hits: 102,
                misses: 103,
                staged: 104,
                staged_steps: 192,
                swaps_started: 193,
                swaps_done: 194,
                steps: 195,
                steps_overlapping_copies: 196,
                longest_submit_us: 105,
                fills: 197,
                shrinks: 198,
                grows: 199,
                grows_refused: 200,
                budget_short: 201,
                lent_slots: 202,
                cpu_timeouts: 106,
                cpu_misses: 107,
            }),
            reuse: engine::ReuseStats {
                snapshots_held: 1,
                pool_bytes_used: 2,
                pool_bytes_limit: 3,
                restores: 4,
                restore_bytes: 5,
                reused_tokens: 6,
                prefilled_tokens: 7,
                evicted_protected: 8,
                evicted_unprotected: 9,
                skipped: 90,
                pool_cut: true,
                chunks_refused: 11,
            },
            parked: 12,
            loads: 13,
            device_losses: 14,
            after_done_errors: 15,
            park_errors: 10,
            spec: None,
        };
        assert_eq!(
            to_stats(&stats),
            seam::BackendStats {
                snapshots_held: 1,
                pool_bytes_used: 2,
                pool_bytes_limit: 3,
                restores: 4,
                restore_bytes: 5,
                reused_tokens: 6,
                prefilled_tokens: 7,
                evicted_protected: 8,
                evicted_unprotected: 9,
                park_errors: 10,
                chunks_refused: 11,
                loads: 13,
                device_losses: 14,
                after_done_errors: 15,
                parked_snapshots: 12,
                cache: Some(seam::CacheCounters {
                    slots_per_layer: 101,
                    hits: 102,
                    misses: 103,
                    staged: 104,
                    cpu_misses: 107,
                    cpu_timeouts: 106,
                    longest_submit_us: 105,
                }),
                spec: None,
            }
        );
        // No model loaded, no cache counters.
        let unloaded = engine::WorkerStats {
            cache: None,
            ..stats
        };
        assert_eq!(to_stats(&unloaded).cache, None);
        assert_eq!(to_stats(&unloaded).spec, None);
    }

    #[test]
    fn stats_take_the_speculation_totals_by_step_size() {
        use engine::spec::{SizeRow, SpecDelta, SpecTotals, MAX_VERIFY};
        let mib = 1024 * 1024;
        let mut by_size = [SizeRow::default(); MAX_VERIFY];
        // Row 0 is t = 1 (the plain steps), row 2 is t = 3. Distinct values, so a crossed pair shows.
        by_size[0] = SizeRow {
            steps: 4,
            ms: 72.0,
            miss_bytes: 8 * mib,
            tokens: 4,
        };
        by_size[2] = SizeRow {
            steps: 5,
            ms: 107.5,
            miss_bytes: 200 * mib,
            tokens: 11,
        };
        let totals = SpecTotals {
            delta: SpecDelta {
                steps: 5,
                plain_steps: 4,
                drafted: 10,
                accepted: 6,
                replays: 2,
                ..SpecDelta::default()
            },
            by_size,
            accept_rate: 0.625,
            mtp_by_k: [SizeRow::default(); MAX_VERIFY],
            mtp_accept_rate: 0.9,
            depth_rates: [0.5; MAX_VERIFY - 1],
            mtp_depth_counts: [(0, 0); MAX_VERIFY - 1],
        };
        let stats = engine::WorkerStats {
            state: engine::Lifecycle::Ready,
            cache: None,
            reuse: engine::ReuseStats::default(),
            parked: 0,
            loads: 1,
            device_losses: 0,
            after_done_errors: 0,
            park_errors: 0,
            spec: Some(totals),
        };
        let spec = to_stats(&stats).spec.expect("loaded with speculation on");
        assert_eq!((spec.steps, spec.plain_steps, spec.drafted), (5, 4, 10));
        assert_eq!((spec.accepted, spec.replays), (6, 2));
        // MTP never drafted: its prior is not reported.
        assert_eq!(spec.accept_rate, [("lookup".to_string(), 0.625)].into());
        assert!(spec.mtp_depth_rates.is_empty() && spec.mtp_draft.is_empty());
        // Only the sizes that ran, in order; the means are per step.
        assert_eq!(
            spec.by_size,
            vec![
                seam::StepSize {
                    t: 1,
                    steps: 4,
                    mean_ms: 18.0,
                    mean_miss_mib: 2.0,
                    tokens: 4,
                },
                seam::StepSize {
                    t: 3,
                    steps: 5,
                    mean_ms: 21.5,
                    mean_miss_mib: 40.0,
                    tokens: 11,
                },
            ]
        );
    }

    #[test]
    fn each_mtp_figure_reaches_the_stats() {
        use engine::spec::{SizeRow, SpecDelta, SpecTotals, MAX_VERIFY};
        let mut mtp_by_k = [SizeRow::default(); MAX_VERIFY];
        // Passes of k = 1 and k = 3 ran; distinct values, so a crossed pair shows.
        mtp_by_k[0] = SizeRow {
            steps: 4,
            ms: 8.0,
            miss_bytes: 0,
            tokens: 3,
        };
        mtp_by_k[2] = SizeRow {
            steps: 2,
            ms: 11.0,
            miss_bytes: 0,
            tokens: 4,
        };
        let mut depth_rates = [0.5; MAX_VERIFY - 1];
        depth_rates[..4].copy_from_slice(&[0.9, 0.8, 0.7, 0.6]);
        let mut mtp_depth_counts = [(0, 0); MAX_VERIFY - 1];
        mtp_depth_counts[..3].copy_from_slice(&[(6, 5), (3, 2), (2, 1)]);
        let totals = SpecTotals {
            delta: SpecDelta {
                steps: 6,
                drafted: 10,
                accepted: 7,
                drafted_lookup: 4,
                accepted_lookup: 3,
                drafted_mtp: 6,
                accepted_mtp: 4,
                ..SpecDelta::default()
            },
            by_size: [SizeRow::default(); MAX_VERIFY],
            accept_rate: 0.625,
            mtp_by_k,
            mtp_accept_rate: 0.9,
            depth_rates,
            mtp_depth_counts,
        };
        let spec = spec_counters(&totals);
        assert_eq!(
            spec.accept_rate,
            [("lookup".to_string(), 0.625), ("mtp".to_string(), 0.9)].into()
        );
        // Depths 1 to the deepest tested (3), not the rest at their prior.
        assert_eq!(spec.mtp_depth_rates, vec![0.9, 0.8, 0.7]);
        assert_eq!(
            spec.mtp_draft,
            vec![
                seam::MtpDraftRow {
                    k: 1,
                    passes: 4,
                    mean_ms: 2.0
                },
                seam::MtpDraftRow {
                    k: 3,
                    passes: 2,
                    mean_ms: 5.5
                },
            ]
        );
    }

    fn info() -> engine::LoadInfo {
        engine::LoadInfo {
            time: Duration::from_millis(9_500),
            device: "AMD Radeon RX 9070 XT".into(),
            device_uuid: [7; 16],
            driver: "2.0.355 (0x800163)".into(),
            slots_per_layer: 128,
            vram_weights: 1,
            vram_session: 2,
            vram_used: Some(15 << 30),
            pool: PoolLimit {
                bytes: 6 << 30,
                requested: 8 << 30,
                cut: true,
            },
            unparked: Unparked {
                restored: 7,
                dropped: 1,
            },
            retried_with_slots: None,
        }
    }

    #[test]
    fn progress_counts_load_ticks() {
        let gauge = Gauge::default();
        let before = gauge.progress();
        gauge.load_ticks.fetch_add(3, Ordering::SeqCst);
        let after = gauge.progress();
        assert_eq!(after.ticks, before.ticks + 3);
        assert_eq!(after.pending, before.pending);
        // The load's ticks and the adapter's own both move it.
        gauge.tick();
        assert_eq!(gauge.progress().ticks, before.ticks + 4);
    }

    #[test]
    fn engine_config_hands_the_load_a_progress_counter() {
        let cfg = server::Config {
            model: Some("m.gguf".into()),
            ..server::Config::default()
        };
        let ticks = engine_config(&cfg).unwrap().load_ticks.expect("a counter");
        assert_eq!(ticks.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn only_the_tested_gpu_has_no_warning() {
        assert_eq!(untested_gpu_warning("AMD Radeon RX 9070 XT"), None);
        let w = untested_gpu_warning("NVIDIA GeForce RTX 4090").expect("a warning");
        assert_eq!(
            w,
            "this GPU (NVIDIA GeForce RTX 4090) has not run Artemis's tests; the tested GPU is the AMD Radeon RX 9070 XT"
        );
        // Once per process: the second load on the same untested GPU is quiet, and a tested one never warns.
        let warned = AtomicBool::new(false);
        assert_eq!(
            untested_gpu_warning_once(&warned, "AMD Radeon RX 9070 XT"),
            None
        );
        assert!(!warned.load(Ordering::SeqCst));
        assert_eq!(
            untested_gpu_warning_once(&warned, "X"),
            untested_gpu_warning("X")
        );
        assert_eq!(untested_gpu_warning_once(&warned, "X"), None);
    }

    /// The `kind` of each line the logger wrote to `dir`, in order.
    fn kinds(dir: &std::path::Path) -> Vec<String> {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        files.sort();
        files
            .iter()
            .flat_map(|f| {
                std::fs::read_to_string(f)
                    .unwrap()
                    .lines()
                    .map(String::from)
                    .collect::<Vec<_>>()
            })
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(&l).unwrap();
                v["event"]
                    .as_str()
                    .or_else(|| v["kind"].as_str())
                    .unwrap_or("?")
                    .to_string()
            })
            .collect()
    }

    fn tuned_file() -> TunedGpu {
        TunedGpu {
            path: "T/tuned.toml".into(),
            gpu_name: "AMD Radeon RX 9070 XT".into(),
            gpu_uuid: vk::format_uuid(&[7; 16]),
            driver: "2.0.355 (0x800163)".into(),
            names_device: true,
        }
    }

    const DIFFERS: &str = "The tuned settings were measured on AMD Radeon RX 9070 XT with driver 2.0.355 (0x800163); \
                           this GPU or driver differs. Run artemis setup --force to measure again.";

    #[test]
    fn a_load_on_another_gpu_or_driver_is_compared_with_the_tuned_file() {
        let t = tuned_file();
        let driver = "2.0.355 (0x800163)";
        assert_eq!(tuned_gpu_warning(&t, &[7; 16], driver), None);
        assert_eq!(
            tuned_gpu_warning(&t, &[8; 16], driver).as_deref(),
            Some(DIFFERS)
        );
        assert_eq!(
            tuned_gpu_warning(&t, &[7; 16], "2.0.400 (0x800190)").as_deref(),
            Some(DIFFERS)
        );
        // The file may spell the UUID without dashes.
        let plain = TunedGpu {
            gpu_uuid: "07070707070707070707070707070707".into(),
            ..tuned_file()
        };
        assert_eq!(tuned_gpu_warning(&plain, &[7; 16], driver), None);
        // A UUID that does not parse never matches.
        let bad = TunedGpu {
            gpu_uuid: "x".into(),
            ..tuned_file()
        };
        assert_eq!(
            tuned_gpu_warning(&bad, &[7; 16], driver).as_deref(),
            Some(DIFFERS)
        );
    }

    #[test]
    fn the_tuned_gpu_comes_from_a_loaded_file_only() {
        use server::TunedState;
        let cfg = |tuned| server::Config {
            tuned,
            ..server::Config::default()
        };
        assert_eq!(tuned_gpu(&cfg(TunedState::Off)), None);
        assert_eq!(tuned_gpu(&cfg(TunedState::Missing("T".into()))), None);
        let loaded = |applied: &[&str]| TunedState::Loaded {
            path: "T/tuned.toml".into(),
            applied: applied.iter().map(|s| s.to_string()).collect(),
            ignored: vec![],
            stale: vec![],
            gpu_name: "AMD Radeon RX 9070 XT".into(),
            gpu_uuid: vk::format_uuid(&[7; 16]),
            driver: "2.0.355 (0x800163)".into(),
        };
        assert_eq!(
            tuned_gpu(&cfg(loaded(&["device_uuid"]))),
            Some(tuned_file())
        );
        let t = tuned_gpu(&cfg(loaded(&["speculation.enabled"]))).unwrap();
        assert!(!t.names_device);
    }

    /// A3: a GPU the tuned file names that cannot be opened is reported with the file and the way out.
    #[test]
    fn a_named_gpu_that_cannot_be_opened_names_the_tuned_file() {
        let gone = "no usable discrete GPU; found: Other GPU (discrete, uuid 1)".to_string();
        let t = tuned_file();
        let got = tuned_load_failure(Some(&t), gone.clone(), true);
        assert!(got.starts_with(&gone), "{got}");
        assert!(
            got.contains("T/tuned.toml") && got.contains("device_uuid"),
            "{got}"
        );
        assert!(
            got.ends_with("Run artemis setup --force to measure again"),
            "{got}"
        );
        // A reply to an API client names the file without its local path.
        let reply = tuned_load_failure(Some(&t), gone.clone(), false);
        assert!(
            reply.contains("the tuned settings file") && !reply.contains("T/tuned.toml"),
            "{reply}"
        );
        assert!(
            reply.ends_with("Run artemis setup --force to measure again"),
            "{reply}"
        );
        // The file does not name the GPU, no file at all, or another failure: the message stands.
        let unnamed = TunedGpu {
            names_device: false,
            ..tuned_file()
        };
        assert_eq!(tuned_load_failure(Some(&unnamed), gone.clone(), true), gone);
        assert_eq!(tuned_load_failure(None, gone.clone(), true), gone);
        let other = "the GPU was lost".to_string();
        assert_eq!(tuned_load_failure(Some(&t), other.clone(), true), other);
    }

    #[test]
    fn the_observer_checks_each_load_against_the_tuned_file_and_explains_a_missing_gpu() {
        let run = |tuned: Option<TunedGpu>, events: Vec<WorkerEvent>, tag: &str| {
            let dir = std::env::temp_dir().join(format!(
                "artemis-backend-tuned-{}-{tag}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            let logger = Logger::new(dir.clone(), 0).unwrap();
            let mut notify = observer(
                logger,
                Instant::now(),
                Arc::default(),
                Arc::default(),
                Arc::default(),
                Arc::default(),
                None,
                tuned,
            );
            for e in events {
                notify(e);
            }
            let got = log_lines(&dir);
            let _ = std::fs::remove_dir_all(&dir);
            got
        };
        let same = || WorkerEvent::Loaded(info());
        let other_gpu = || {
            WorkerEvent::Loaded(engine::LoadInfo {
                device_uuid: [9; 16],
                ..info()
            })
        };
        let other_driver = || {
            WorkerEvent::Loaded(engine::LoadInfo {
                driver: "2.0.400 (0x800190)".into(),
                ..info()
            })
        };
        let line_kinds = |lines: &[serde_json::Value]| -> Vec<String> {
            lines
                .iter()
                .map(|l| l["event"].as_str().unwrap().to_string())
                .collect()
        };
        // The same GPU and driver: only the load line.
        let got = run(Some(tuned_file()), vec![same()], "same");
        assert_eq!(line_kinds(&got), ["load"]);
        // Another GPU, then another driver: a warning after each load line, with the file's words.
        let got = run(
            Some(tuned_file()),
            vec![other_gpu(), other_driver()],
            "differs",
        );
        assert_eq!(
            line_kinds(&got),
            ["load", "config_warning", "load", "config_warning"]
        );
        assert_eq!(got[1]["message"], DIFFERS);
        assert_eq!(got[3]["message"], DIFFERS);
        // No tuned file: nothing to compare.
        assert_eq!(line_kinds(&run(None, vec![other_gpu()], "none")), ["load"]);
        // A load that cannot open the GPU the file names says which file to measure again.
        let gone = "no usable discrete GPU; found: Other GPU".to_string();
        let got = run(
            Some(tuned_file()),
            vec![
                WorkerEvent::LoadFailed {
                    message: gone.clone(),
                },
                WorkerEvent::ReloadFailed { message: gone },
            ],
            "gone",
        );
        assert_eq!(line_kinds(&got), ["load_failed", "reload_failed"]);
        for l in &got {
            let m = l["message"].as_str().unwrap();
            assert!(
                m.contains("T/tuned.toml") && m.contains("artemis setup --force"),
                "{m}"
            );
        }
    }

    fn log_lines(dir: &std::path::Path) -> Vec<serde_json::Value> {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        files.sort();
        files
            .iter()
            .flat_map(|f| {
                std::fs::read_to_string(f)
                    .unwrap()
                    .lines()
                    .map(|l| serde_json::from_str(l).unwrap())
                    .collect::<Vec<serde_json::Value>>()
            })
            .collect()
    }

    #[test]
    fn the_observer_warns_once_about_an_untested_gpu_after_the_load_line() {
        let loaded = |device: &str| {
            WorkerEvent::Loaded(engine::LoadInfo {
                device: device.into(),
                ..info()
            })
        };
        let run = |tag: &str, flag: Option<Arc<AtomicBool>>, devices: &[&str]| {
            let dir = std::env::temp_dir()
                .join(format!("artemis-backend-warn-{}-{tag}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let logger = Logger::new(dir.clone(), 0).unwrap();
            let mut notify = observer(
                logger,
                Instant::now(),
                Arc::default(),
                Arc::default(),
                Arc::default(),
                Arc::default(),
                flag,
                None,
            );
            for d in devices {
                notify(loaded(d));
            }
            let got = kinds(&dir);
            let _ = std::fs::remove_dir_all(&dir);
            got
        };
        let on = || Some(Arc::new(AtomicBool::new(false)));
        // On: one warning after the first load line, none after the second.
        assert_eq!(
            run("on", on(), &["Some Other GPU", "Some Other GPU"]),
            ["load", "config_warning", "load"]
        );
        // The tested GPU never warns, and a backend with the flag off (the fake loaders') never does either.
        assert_eq!(run("tested", on(), &["AMD Radeon RX 9070 XT"]), ["load"]);
        assert_eq!(run("off", None, &["Some Other GPU"]), ["load"]);
    }

    #[test]
    fn the_load_line_carries_the_figures() {
        assert_eq!(
            load_line(&info()),
            LifecycleLine::Load {
                seconds: 9.5,
                detail: "AMD Radeon RX 9070 XT; first try".into(),
                slots_per_layer: Some(128),
                vram_used_mib: Some(15360.0),
                pool_limit_gib: Some(6.0),
                pool_cut: Some(true),
                unparked: Some(7),
            }
        );
        // No figure from the driver is no figure, and a retry says so.
        let retried = engine::LoadInfo {
            vram_used: None,
            retried_with_slots: Some(64),
            ..info()
        };
        let LifecycleLine::Load {
            detail,
            vram_used_mib,
            ..
        } = load_line(&retried)
        else {
            panic!("not a load line");
        };
        assert_eq!(detail, "AMD Radeon RX 9070 XT; retried with 64 slots");
        assert_eq!(vram_used_mib, None);
    }

    #[test]
    fn idle_unload_zero_is_never() {
        let cfg = |s| server::Config {
            idle_unload_s: s,
            ..server::Config::default()
        };
        assert_eq!(idle_unload(&cfg(0)), None);
        assert_eq!(idle_unload(&cfg(30)), Some(Duration::from_secs(30)));
    }

    #[test]
    fn engine_config_parses_the_device_uuid() {
        let cfg = |uuid: Option<&str>| server::Config {
            model: Some("m.gguf".into()),
            ctx: 4096,
            cache_slots: Some(33),
            snapshot_pool_gib: 3,
            f32_prefill: true,
            device_uuid: uuid.map(String::from),
            ..server::Config::default()
        };
        let e = engine_config(&cfg(Some("00112233-4455-6677-8899-aabbccddeeff"))).unwrap();
        assert_eq!(
            e.device_uuid,
            Some([
                0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
                0xee, 0xff
            ])
        );
        assert_eq!(
            (e.ctx, e.cache_slots, e.snapshot_pool_gib, e.f32_prefill),
            (4096, Some(33), 3, true)
        );
        assert_eq!(e.model, std::path::PathBuf::from("m.gguf"));
        assert_eq!(engine_config(&cfg(None)).unwrap().device_uuid, None);

        let bad = "not-a-uuid-at-all-0123";
        let err = engine_config(&cfg(Some(bad))).err().unwrap().to_string();
        assert!(err.contains("device_uuid"), "{err}");
        assert!(!err.contains(bad), "the message shows the value: {err}");
        assert!(!err.contains("m.gguf"), "{err}");
    }

    #[test]
    fn speculation_defaults_match_the_engines() {
        assert_eq!(
            spec_config(&server::config::SpeculationConfig::default()),
            engine::SpecConfig::default()
        );
        // The measured defaults, pinned on both sides.
        let d = engine::SpecConfig::default();
        assert_eq!(
            (d.max_draft, d.cost_slope_ms, d.accept_prior),
            (4, 4.4, 0.5)
        );
    }

    #[test]
    fn engine_config_carries_the_speculation_table() {
        let table = server::config::SpeculationConfig {
            enabled: false,
            sources: vec!["lookup".into()],
            max_draft: 7,
            ngram_min: 3,
            ngram_max: 9,
            min_gain: 1.2,
            accept_half_life: 10.5,
            cost_slope_ms: 1.25,
            accept_prior: 0.75,
            probe_every: 8,
            mtp_window: 100,
            mtp_ctx: 1024,
            mtp_ring: 32,
            mtp_accept_prior: vec![0.9, 0.1],
            mtp_cost_ms: 2.5,
            source_choice: "best".into(),
            mtp_head_rows: 32768,
        };
        let cfg = server::Config {
            model: Some("m.gguf".into()),
            speculation: table,
            ..server::Config::default()
        };
        assert_eq!(
            engine_config(&cfg).unwrap().speculation,
            engine::SpecConfig {
                enabled: false,
                max_draft: 7,
                ngram_min: 3,
                ngram_max: 9,
                min_gain: 1.2,
                accept_half_life: 10.5,
                cost_slope_ms: 1.25,
                accept_prior: 0.75,
                sources: vec![engine::spec::Source::Lookup],
                fixed_draft: None,
                probe_every: 8,
                mtp_window: 100,
                mtp_ctx: 1024,
                mtp_ring: 32,
                mtp_accept_prior: vec![0.9, 0.1],
                mtp_cost_ms: 2.5,
                source_choice: engine::SourceChoice::Best,
                mtp_head_rows: 32768,
            }
        );
        let on = server::Config {
            model: Some("m.gguf".into()),
            ..server::Config::default()
        };
        assert_eq!(
            engine_config(&on).unwrap().speculation,
            engine::SpecConfig::default()
        );
    }

    /// The server cannot depend on the engine, so its `[speculation]` rules are a copy of `SpecConfig::check`. Each
    /// case is set in both, and both must refuse it with the same key and words, or accept it.
    /// A rule added to the engine's check needs a case here, or this test cannot see it drift.
    #[test]
    fn the_servers_speculation_rules_agree_with_the_engines() {
        use server::config::SpeculationConfig as Table;
        type Edit = fn(&mut Table);
        let cases: Vec<(&str, Edit)> = vec![
            ("defaults", |_| {}),
            ("disabled", |t| t.enabled = false),
            ("max_draft 0", |t| t.max_draft = 0),
            ("max_draft 1", |t| t.max_draft = 1),
            ("max_draft 15", |t| t.max_draft = 15),
            ("max_draft 16", |t| t.max_draft = 16),
            ("max_draft huge", |t| t.max_draft = u32::MAX),
            ("ngram_min 0", |t| t.ngram_min = 0),
            ("ngram_min 1", |t| t.ngram_min = 1),
            ("ngram_min above ngram_max", |t| t.ngram_min = 5),
            ("ngram_max 16", |t| t.ngram_max = 16),
            ("ngram_max 17", |t| t.ngram_max = 17),
            ("ngram_max 0", |t| t.ngram_max = 0),
            ("min_gain 1", |t| t.min_gain = 1.0),
            ("min_gain 0.99", |t| t.min_gain = 0.99),
            ("min_gain nan", |t| t.min_gain = f64::NAN),
            ("min_gain inf", |t| t.min_gain = f64::INFINITY),
            ("half_life 0", |t| t.accept_half_life = 0.0),
            ("half_life negative", |t| t.accept_half_life = -1.0),
            ("half_life tiny", |t| t.accept_half_life = 1e-9),
            ("half_life nan", |t| t.accept_half_life = f64::NAN),
            ("half_life inf", |t| t.accept_half_life = f64::INFINITY),
            ("cost_slope 0", |t| t.cost_slope_ms = 0.0),
            ("cost_slope negative", |t| t.cost_slope_ms = -0.5),
            ("cost_slope nan", |t| t.cost_slope_ms = f64::NAN),
            ("cost_slope inf", |t| t.cost_slope_ms = f64::INFINITY),
            ("accept_prior 0", |t| t.accept_prior = 0.0),
            ("accept_prior 1", |t| t.accept_prior = 1.0),
            ("accept_prior below 0", |t| t.accept_prior = -0.1),
            ("accept_prior above 1", |t| t.accept_prior = 1.1),
            ("accept_prior nan", |t| t.accept_prior = f64::NAN),
            ("sources empty", |t| t.sources = vec![]),
            ("sources lookup", |t| t.sources = vec!["lookup".into()]),
            ("sources mtp", |t| t.sources = vec!["mtp".into()]),
            ("sources mtp first", |t| {
                t.sources = vec!["mtp".into(), "lookup".into()]
            }),
            ("sources duplicate lookup", |t| {
                t.sources = vec!["lookup".into(), "lookup".into()]
            }),
            ("sources duplicate mtp", |t| {
                t.sources = vec!["mtp".into(), "lookup".into(), "mtp".into()]
            }),
            ("sources duplicate before unknown", |t| {
                t.sources = vec!["lookup".into(), "lookup".into(), "x".into()]
            }),
            ("sources empty is not unknown", |t| t.sources = vec![]),
            ("sources unknown", |t| t.sources = vec!["x".into()]),
            ("sources unknown after lookup", |t| {
                t.sources = vec!["lookup".into(), "suffix".into()]
            }),
            ("probe_every 0", |t| t.probe_every = 0),
            ("probe_every 1", |t| t.probe_every = 1),
            ("mtp_ctx 256", |t| t.mtp_ctx = 256),
            ("mtp_ctx 511", |t| t.mtp_ctx = 511),
            ("mtp_ctx 512", |t| {
                t.mtp_ctx = 512;
                t.mtp_window = 256;
                t.mtp_ring = 128;
            }),
            ("mtp_ctx 512 keeps the default window", |t| t.mtp_ctx = 512),
            ("mtp_ctx 65536", |t| {
                t.mtp_ctx = 65536;
                t.mtp_window = 32768;
                t.mtp_ring = 16384;
            }),
            ("mtp_ctx 65792", |t| t.mtp_ctx = 65792),
            ("mtp_ctx not a multiple of 256", |t| t.mtp_ctx = 8000),
            ("mtp_ctx huge", |t| t.mtp_ctx = u32::MAX),
            ("mtp_window 0", |t| t.mtp_window = 0),
            ("mtp_window 1", |t| t.mtp_window = 1),
            ("mtp_window half the ctx", |t| t.mtp_window = 4096),
            ("mtp_window above half", |t| t.mtp_window = 4097),
            ("mtp_ring 8", |t| t.mtp_ring = 8),
            ("mtp_ring 16", |t| t.mtp_ring = 16),
            ("mtp_ring 24", |t| t.mtp_ring = 24),
            ("mtp_ring a quarter of the ctx", |t| t.mtp_ring = 2048),
            ("mtp_ring above a quarter", |t| t.mtp_ring = 2064),
            ("mtp_accept_prior empty", |t| t.mtp_accept_prior = vec![]),
            ("mtp_accept_prior one", |t| t.mtp_accept_prior = vec![0.5]),
            ("mtp_accept_prior 15", |t| {
                t.mtp_accept_prior = vec![0.5; 15]
            }),
            ("mtp_accept_prior 16", |t| {
                t.mtp_accept_prior = vec![0.5; 16]
            }),
            ("mtp_accept_prior 0 and 1", |t| {
                t.mtp_accept_prior = vec![0.0, 1.0]
            }),
            ("mtp_accept_prior below 0", |t| {
                t.mtp_accept_prior = vec![0.5, -0.1]
            }),
            ("mtp_accept_prior above 1", |t| {
                t.mtp_accept_prior = vec![1.1]
            }),
            ("mtp_accept_prior nan", |t| {
                t.mtp_accept_prior = vec![f64::NAN]
            }),
            ("mtp_cost_ms 0", |t| t.mtp_cost_ms = 0.0),
            ("mtp_cost_ms negative", |t| t.mtp_cost_ms = -0.1),
            ("mtp_cost_ms nan", |t| t.mtp_cost_ms = f64::NAN),
            ("mtp_cost_ms inf", |t| t.mtp_cost_ms = f64::INFINITY),
            ("mtp_head_rows 0", |t| t.mtp_head_rows = 0),
            ("mtp_head_rows 512", |t| t.mtp_head_rows = 512),
            ("mtp_head_rows 1023", |t| t.mtp_head_rows = 1023),
            ("mtp_head_rows 1024", |t| t.mtp_head_rows = 1024),
            ("mtp_head_rows 32768", |t| t.mtp_head_rows = 32768),
        ];
        // `source_choice` is a string here, so the engine's rule is `SourceChoice::parse`.
        for (name, want) in [
            ("priority", true),
            ("best", true),
            ("fastest", false),
            ("", false),
        ] {
            let table = Table {
                source_choice: name.into(),
                ..Table::default()
            };
            assert_eq!(
                table.check().is_ok(),
                engine::SourceChoice::parse(name).is_some() && want,
                "{name}"
            );
        }
        let mut refused = 0;
        for (name, edit) in cases {
            let mut table = Table::default();
            edit(&mut table);
            // The engine's `sources` rules live in `parse_list`; its `check` takes the rest.
            let engine = spec_sources(&table).and_then(|_| spec_config(&table).check());
            assert_eq!(table.check(), engine, "{name}");
            refused += engine.is_err() as usize;
        }
        assert!(refused >= 35, "the table refuses too little: {refused}");
    }
}
