//! The engine worker thread: one thread owns the loaded engine and runs the
//! commands the server queues, in order. It loads lazily and again after a device loss, parks the snapshot pool on an
//! unload so a reload can bring it back, and drives `run_turn` for each request. The engine behind it is a `Loader`:
//! the real one is `EngineLoader`, the CI's is `fake::FakeLoader`. There is no tokio here: std channels and
//! one thread.
use crate::loaded::EngineLoader;
use crate::request::{
    CacheDelta, Counts, ErrorKind, Event, EventSink, Finish, Lifecycle, Request, Timings,
};
use crate::reuse::{Marks, ReuseStats};
use crate::spec::{SpecConfig, SpecState, SpecTotals};
use crate::turn::{run_turn, Ending, Live};
use crate::{Config, Error, ErrorClass, Parked, PoolLimit, Stats, Unparked};
use qwen36::dims::Dims;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// A callback the worker calls once, on its own thread, with a command's answer (no async runtime in `engine`).
pub type Reply<T> = Box<dyn FnOnce(T) + Send>;

/// The message of a load that did not fit in host memory; retrying with a smaller cache would not help (the cache is in
/// VRAM). It keeps the engine's figures.
pub fn host_short_message(e: &Error) -> String {
    format!(
        "{e}. Close other programs or add RAM; the snapshot pool is sized after this check and is not part of these figures"
    )
}

/// The message of a load that failed for GPU memory twice: once with the engine's own slot count and once
/// with half of it, each with the engine's figures.
pub fn vram_short_message(slots: usize, first: &Error, half: usize, second: &Error) -> String {
    format!(
        "not enough GPU memory: with {slots} cache slots per layer, {first}; with {half}, {second}. Close programs that use the GPU, or lower ctx or set cache_slots"
    )
}

/// A failed `Loader::load`.
#[derive(Debug)]
pub struct LoadFailure {
    pub error: Error,
    /// The slot count `fit_cache` chose for the failed attempt, when `Engine::open` got that far; `None` otherwise.
    pub slots: Option<usize>,
}

/// What a successful load reports.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadInfo {
    pub time: Duration,
    pub device: String,
    /// The device's UUID and the driver's name and version: the tuned file records both, and `serve` compares them at
    /// each load.
    pub device_uuid: [u8; 16],
    pub driver: String,
    pub slots_per_layer: usize,
    /// K-64's MTP slots the load gave `blk.40` and their bytes (`Stats::mtp_slots`, `mtp_slot_bytes`); 0 without the
    /// MTP or without room.
    pub mtp_slots: usize,
    pub mtp_slot_bytes: u64,
    pub vram_weights: u64,
    pub vram_session: u64,
    /// This process's VRAM use after the load, the device-local heap's `usage` (`VK_EXT_memory_budget`); `None` when
    /// the driver reports none.
    pub vram_used: Option<u64>,
    pub pool: PoolLimit,
    pub unparked: Unparked,
    /// The slot count of the retry that worked, when the first attempt failed for VRAM.
    pub retried_with_slots: Option<usize>,
}

/// Why a load failed, as the server reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadError {
    pub kind: ErrorKind,
    pub message: String,
}

/// What an `Unload` did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnloadInfo {
    pub was_loaded: bool,
    /// The snapshots parked in host memory.
    pub parked: usize,
    pub parked_bytes: u64,
}

/// What the `Stats` command answers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkerStats {
    pub state: Lifecycle,
    /// The expert cache's counters; `None` while unloaded.
    pub cache: Option<Stats>,
    /// The reuse counters. `reused_tokens` and `prefilled_tokens` are the worker's own, reset at each load.
    pub reuse: ReuseStats,
    /// The snapshots held parked, waiting for the next load.
    pub parked: usize,
    /// Loads that succeeded, reloads after a device loss included.
    pub loads: u64,
    pub device_losses: u64,
    /// Errors in the reply-end snapshot or the budget check after a `Done`, which no client sees, and commit faults
    /// after a burst had already ended.
    pub after_done_errors: u64,
    /// Unloads whose park failed: the snapshots were lost. A park that failed with a device loss also counts in
    /// `device_losses`.
    pub park_errors: u64,
    /// Speculation's counts since the last load; `None` while unloaded and with speculation off.
    pub spec: Option<SpecTotals>,
}

/// Why the model was unloaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnloadCause {
    /// The idle timer ran out.
    Idle,
    /// An `Unload` command.
    Requested,
    /// A `Shutdown` command, or the `Worker` was dropped.
    Shutdown,
}

/// What `WorkerConfig.notify` is told, on the worker thread. A device loss sends `DeviceLost`, then `Loaded` or
/// `ReloadFailed`; no `Unloaded`.
#[derive(Clone, Debug, PartialEq)]
pub enum WorkerEvent {
    Loaded(LoadInfo),
    Unloaded { cause: UnloadCause },
    LoadFailed { message: String },
    DeviceLost,
    ReloadFailed { message: String },
}

/// The observer of the worker's lifecycle.
pub type Notify = Box<dyn FnMut(WorkerEvent) + Send>;

/// How to run the real worker (`Worker::spawn`).
pub struct WorkerConfig {
    pub engine: Config,
    pub dims: Dims,
    /// Unload after this long with no request; `None` keeps the model loaded.
    pub idle_unload: Option<Duration>,
    /// Whether an unload parks the snapshots in host memory for the next load.
    pub park_snapshots: bool,
    pub notify: Option<Notify>,
}

impl WorkerConfig {
    /// The server's defaults for `engine`: the Qwen3.6 dims, an idle unload after 10 minutes
    /// and parked snapshots.
    pub fn new(engine: Config) -> WorkerConfig {
        WorkerConfig {
            engine,
            dims: Dims::QWEN36,
            idle_unload: Some(Duration::from_secs(600)),
            park_snapshots: true,
            notify: None,
        }
    }
}

/// What the worker thread needs besides its loader.
pub struct WorkerParams {
    pub idle_unload: Option<Duration>,
    pub park_snapshots: bool,
    /// The session's positions and the vocabulary's size, for `Request::check`.
    pub ctx: usize,
    pub n_vocab: usize,
    pub notify: Option<Notify>,
    /// Speculation's settings, when it is on: the worker builds a `SpecState` from them at each load and drops it at
    /// each unload. `None` is today's decode loop.
    pub speculation: Option<SpecConfig>,
}

/// The speculation settings the worker runs under for `engine`: the one source is `Config::speculation`, `Some`
/// only when it is enabled.
fn speculation_of(engine: &Config) -> Option<SpecConfig> {
    engine
        .speculation
        .enabled
        .then(|| engine.speculation.clone())
}

/// What a parked set of snapshots reports about itself, for `UnloadInfo` and `WorkerStats`.
pub trait ParkedStats {
    /// The snapshots held.
    fn count(&self) -> usize;
    /// The bytes of KV and state held.
    fn bytes(&self) -> u64;
}

impl ParkedStats for Parked {
    fn count(&self) -> usize {
        self.len()
    }

    fn bytes(&self) -> u64 {
        Parked::bytes(self)
    }
}

/// The engine as the worker loads and unloads it.
pub trait Loader {
    type Live: Live;
    type Parked: ParkedStats;

    /// Loads the model. `cache_slots` overrides the configured slot count (the retry's half); `None` keeps the
    /// configured one. `parked` is taken only at the last step, so a failed attempt leaves it for the retry and for the
    /// next request.
    fn load(
        &mut self,
        parked: &mut Option<Self::Parked>,
        cache_slots: Option<usize>,
    ) -> Result<(Self::Live, LoadInfo), LoadFailure>;
    /// Parks the snapshots in host memory and unloads. On `Err` the live state is gone and nothing is parked.
    fn park(&mut self, live: Self::Live) -> Result<Self::Parked, Error>;
    /// Unloads without parking.
    fn unload(&mut self, live: Self::Live);
}

/// What a `Worker` queues for its thread, in one FIFO.
pub enum Command {
    Generate {
        req: Request,
        sink: Box<dyn EventSink>,
        queued: Instant,
    },
    Load(Reply<Result<LoadInfo, LoadError>>),
    Unload(Reply<UnloadInfo>),
    Stats(Reply<WorkerStats>),
    Shutdown,
}

/// The worker's thread has ended.
#[derive(Debug)]
pub struct Gone;

/// The handle to the worker thread.
pub struct Worker {
    tx: Sender<Command>,
    state: Arc<AtomicU8>,
    /// Unloads queued and not yet processed: non-zero ends a running turn `Unloaded`.
    unload_now: Arc<AtomicUsize>,
    thread: JoinHandle<()>,
}

impl Worker {
    /// Starts the worker thread over the real engine: `EngineLoader`, with the parameters `cfg` names. The thread loads
    /// on the first request or `Load`.
    pub fn spawn(cfg: WorkerConfig) -> Worker {
        let params = WorkerParams {
            idle_unload: cfg.idle_unload,
            park_snapshots: cfg.park_snapshots,
            ctx: cfg.engine.ctx,
            n_vocab: cfg.dims.n_vocab,
            notify: cfg.notify,
            speculation: speculation_of(&cfg.engine),
        };
        Worker::spawn_with(EngineLoader::new(cfg.engine, cfg.dims), params)
    }

    /// Starts the worker thread over `loader`. The thread loads on the first request or `Load`.
    pub fn spawn_with<L: Loader + Send + 'static>(loader: L, p: WorkerParams) -> Worker {
        let (tx, rx) = channel();
        let state = Arc::new(AtomicU8::new(Lifecycle::Idle as u8));
        let unload_now = Arc::new(AtomicUsize::new(0));
        let thread = std::thread::Builder::new()
            .name("artemis-worker".into())
            .spawn({
                let (state, unload_now) = (Arc::clone(&state), Arc::clone(&unload_now));
                move || serve(loader, p, rx, state, unload_now)
            })
            .expect("spawn the worker thread");
        Worker {
            tx,
            state,
            unload_now,
            thread,
        }
    }

    /// Queues a request. `Err(Gone)` if the thread ended.
    pub fn submit(&self, req: Request, sink: Box<dyn EventSink>) -> Result<(), Gone> {
        self.send(Command::Generate {
            req,
            sink,
            queued: Instant::now(),
        })
    }

    /// Queues a load; answers with the last `LoadInfo` if the model is already loaded, in FIFO order (behind a running
    /// request).
    pub fn load(&self, reply: Reply<Result<LoadInfo, LoadError>>) -> Result<(), Gone> {
        self.send(Command::Load(reply))
    }

    /// Queues an unload. Requests running or queued before it end `Unloaded` rather than finish.
    pub fn unload(&self, reply: Reply<UnloadInfo>) -> Result<(), Gone> {
        self.unload_now.fetch_add(1, Ordering::SeqCst);
        let sent = self.send(Command::Unload(reply));
        if sent.is_err() {
            self.unload_now.fetch_sub(1, Ordering::SeqCst);
        }
        sent
    }

    /// Queues a stats read; like every command it waits behind a running request.
    pub fn stats(&self, reply: Reply<WorkerStats>) -> Result<(), Gone> {
        self.send(Command::Stats(reply))
    }

    pub fn state(&self) -> Lifecycle {
        Lifecycle::from_u8(self.state.load(Ordering::SeqCst))
    }

    /// Unloads without parking and joins the thread. `Err(Gone)` if the thread had panicked.
    pub fn shutdown(self) -> Result<(), Gone> {
        let _ = self.tx.send(Command::Shutdown);
        self.thread.join().map_err(|_| Gone)
    }

    fn send(&self, c: Command) -> Result<(), Gone> {
        self.tx.send(c).map_err(|_| Gone)
    }
}

/// Forwards to the request's sink and keeps the `Done`'s counts, which `run_turn` does not return.
struct Tap<'a> {
    inner: &'a mut dyn EventSink,
    counts: Option<Counts>,
}

impl EventSink for Tap<'_> {
    fn send(&mut self, event: Event) -> bool {
        if let Event::Done { counts, .. } = &event {
            self.counts = Some(*counts);
        }
        self.inner.send(event)
    }

    fn closed(&self) -> bool {
        self.inner.closed()
    }
}

/// A load error as the server reports it.
fn load_error(e: &Error) -> LoadError {
    let message = if e.is_host_memory() {
        host_short_message(e)
    } else {
        e.to_string()
    };
    LoadError {
        kind: e.kind().event_kind(),
        message,
    }
}

/// The worker thread's state: the loader, the loaded engine if any, and the queue in front of the channel.
struct Run<L: Loader> {
    loader: L,
    live: Option<L::Live>,
    /// The snapshots of the last unload, waiting for the next load. Empty whenever the model is loaded.
    parked: Option<L::Parked>,
    p: WorkerParams,
    /// Speculation's engine-wide state: built at each load, dropped at each unload. `None` while unloaded and
    /// with speculation off.
    spec: Option<SpecState>,
    rx: Receiver<Command>,
    /// Commands taken off the channel ahead of their turn (after a failed load), in order.
    queue: VecDeque<Command>,
    state: Arc<AtomicU8>,
    unload_now: Arc<AtomicUsize>,
    /// The last successful load's report, for a `Load` that finds the model loaded.
    info: Option<LoadInfo>,
    /// When the last request or load finished: the idle timer counts from here.
    last_active: Instant,
    reused: u64,
    prefilled: u64,
    loads: u64,
    device_losses: u64,
    after_done_errors: u64,
    park_errors: u64,
}

fn serve<L: Loader>(
    loader: L,
    p: WorkerParams,
    rx: Receiver<Command>,
    state: Arc<AtomicU8>,
    unload_now: Arc<AtomicUsize>,
) {
    let mut run = Run {
        loader,
        live: None,
        parked: None,
        p,
        spec: None,
        rx,
        queue: VecDeque::new(),
        state,
        unload_now,
        info: None,
        last_active: Instant::now(),
        reused: 0,
        prefilled: 0,
        loads: 0,
        device_losses: 0,
        after_done_errors: 0,
        park_errors: 0,
    };
    run.run();
}

impl<L: Loader> Run<L> {
    /// Runs commands until a `Shutdown` or until the `Worker` is gone, then unloads without parking. Requests and
    /// replies still queued are dropped with the thread.
    fn run(&mut self) {
        while let Some(command) = self.next() {
            match command {
                Command::Generate { req, sink, queued } => self.generate(req, sink, queued),
                Command::Load(reply) => {
                    let answer = self.load_command();
                    // A load that did the loading set this already; a load of a loaded model counts as activity too.
                    self.last_active = Instant::now();
                    // The queue is failed before the reply: a client that retries on seeing it must not be drained.
                    if let Err(e) = &answer {
                        self.fail_queue(&e.message);
                    }
                    reply(answer);
                }
                Command::Unload(reply) => {
                    let _ = self
                        .unload_now
                        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1));
                    reply(self.unload_live(UnloadCause::Requested, true));
                }
                Command::Stats(reply) => reply(self.stats()),
                Command::Shutdown => break,
            }
        }
        self.unload_live(UnloadCause::Shutdown, false);
        self.set_state(Lifecycle::Idle);
    }

    /// The next command: the local queue first, then the channel, blocking, with the idle timer running while the
    /// model is loaded. `None` when the `Worker` was dropped.
    fn next(&mut self) -> Option<Command> {
        loop {
            if let Some(c) = self.queue.pop_front() {
                return Some(c);
            }
            let Some(idle) = self.p.idle_unload.filter(|_| self.live.is_some()) else {
                return self.rx.recv().ok();
            };
            let left = idle.saturating_sub(self.last_active.elapsed());
            match self.rx.recv_timeout(left) {
                Ok(c) => return Some(c),
                Err(RecvTimeoutError::Timeout) => {
                    self.unload_live(UnloadCause::Idle, true);
                }
                Err(RecvTimeoutError::Disconnected) => return None,
            }
        }
    }

    fn set_state(&self, s: Lifecycle) {
        self.state.store(s as u8, Ordering::SeqCst);
    }

    fn notify(&mut self, e: WorkerEvent) {
        if let Some(n) = &mut self.p.notify {
            n(e);
        }
    }

    /// Unloads the model if one is loaded, parking its snapshots when asked and configured. A failed park has already
    /// dropped the live state; nothing is parked then, and `park_errors` counts it.
    fn unload_live(&mut self, cause: UnloadCause, park: bool) -> UnloadInfo {
        let Some(live) = self.live.take() else {
            return UnloadInfo {
                was_loaded: false,
                parked: 0,
                parked_bytes: 0,
            };
        };
        self.spec = None;
        let (parked, parked_bytes) = if park && self.p.park_snapshots {
            match self.loader.park(live) {
                Ok(held) => {
                    let (n, bytes) = (held.count(), held.bytes());
                    self.parked = (n > 0).then_some(held);
                    (n, bytes)
                }
                Err(e) => {
                    self.park_errors += 1;
                    if e.kind() == ErrorClass::DeviceLost {
                        self.device_losses += 1;
                    }
                    (0, 0)
                }
            }
        } else {
            self.loader.unload(live);
            (0, 0)
        };
        self.set_state(Lifecycle::Idle);
        self.notify(WorkerEvent::Unloaded { cause });
        UnloadInfo {
            was_loaded: true,
            parked,
            parked_bytes,
        }
    }

    fn stats(&self) -> WorkerStats {
        let mut reuse = self
            .live
            .as_ref()
            .map_or_else(ReuseStats::default, |l| l.reuse_stats());
        reuse.reused_tokens = self.reused;
        reuse.prefilled_tokens = self.prefilled;
        WorkerStats {
            state: Lifecycle::from_u8(self.state.load(Ordering::SeqCst)),
            cache: self.live.as_ref().map(|l| l.cache_stats()),
            reuse,
            parked: self.parked.as_ref().map_or(0, |p| p.count()),
            loads: self.loads,
            device_losses: self.device_losses,
            after_done_errors: self.after_done_errors,
            park_errors: self.park_errors,
            spec: self.spec.as_ref().map(|s| s.totals()),
        }
    }

    /// One load attempt, and the retry with half the slots when the first failed for VRAM and knows its count.
    fn attempt(&mut self) -> Result<(L::Live, LoadInfo), LoadError> {
        let failure = match self.loader.load(&mut self.parked, None) {
            Ok(loaded) => return Ok(loaded),
            Err(f) => f,
        };
        let vram =
            failure.error.kind() == ErrorClass::OutOfMemory && !failure.error.is_host_memory();
        if let Some(n) = failure.slots.filter(|&n| vram && n >= 2) {
            let half = n / 2;
            return match self.loader.load(&mut self.parked, Some(half)) {
                Ok((live, mut info)) => {
                    info.retried_with_slots = Some(half);
                    Ok((live, info))
                }
                Err(f)
                    if f.error.kind() == ErrorClass::OutOfMemory && !f.error.is_host_memory() =>
                {
                    Err(LoadError {
                        kind: f.error.kind().event_kind(),
                        message: vram_short_message(n, &failure.error, half, &f.error),
                    })
                }
                Err(f) => Err(load_error(&f.error)),
            };
        }
        Err(load_error(&failure.error))
    }

    /// Loads (or reloads after a device loss) and tells the observer. A failure leaves the worker `Idle`; the caller
    /// fails the requests queued behind it (`fail_queue`) and only then answers its own request or command, so a
    /// client that retries on seeing its error finds the queue empty.
    fn load_model(&mut self, reload: bool) -> Result<LoadInfo, LoadError> {
        if !reload {
            self.set_state(Lifecycle::Loading);
        }
        match self.attempt() {
            Ok((live, info)) => {
                self.live = Some(live);
                self.spec = self.p.speculation.clone().map(SpecState::new);
                self.info = Some(info.clone());
                self.reused = 0;
                self.prefilled = 0;
                self.loads += 1;
                self.last_active = Instant::now();
                self.set_state(Lifecycle::Ready);
                self.notify(WorkerEvent::Loaded(info.clone()));
                Ok(info)
            }
            Err(e) => {
                self.set_state(Lifecycle::Idle);
                let message = e.message.clone();
                self.notify(if reload {
                    WorkerEvent::ReloadFailed { message }
                } else {
                    WorkerEvent::LoadFailed { message }
                });
                Err(e)
            }
        }
    }

    /// A load that failed fails every request waiting behind it, not a ~9 s load each. The other commands keep their
    /// order and run next.
    fn fail_queue(&mut self, message: &str) {
        while let Ok(command) = self.rx.try_recv() {
            match command {
                Command::Generate { mut sink, .. } => {
                    sink.send(Event::Error {
                        kind: ErrorKind::LoadFailed,
                        message: message.to_string(),
                    });
                }
                other => self.queue.push_back(other),
            }
        }
    }

    fn load_command(&mut self) -> Result<LoadInfo, LoadError> {
        match (&self.live, &self.info) {
            (Some(_), Some(info)) => Ok(info.clone()),
            _ => self.load_model(false),
        }
    }

    /// A request whose turn it is.
    fn generate(&mut self, req: Request, mut sink: Box<dyn EventSink>, queued: Instant) {
        let dequeued = Instant::now();
        if sink.closed() {
            return;
        }
        if let Err(message) = req.check(self.p.ctx, self.p.n_vocab) {
            sink.send(Event::Error {
                kind: ErrorKind::Internal,
                message,
            });
            return;
        }
        if self.unload_pending() {
            self.end_unloaded(&req, sink.as_mut(), queued, dequeued);
            return;
        }
        if self.live.is_none() {
            if let Err(e) = self.load_model(false) {
                // The queue first: a client that retries on seeing its error must not be drained by this failure.
                self.fail_queue(&e.message);
                sink.send(Event::Error {
                    kind: ErrorKind::LoadFailed,
                    message: e.message,
                });
                return;
            }
            // An unload or a close that landed during the load is seen before `Started`.
            if sink.closed() {
                return;
            }
            if self.unload_pending() {
                self.end_unloaded(&req, sink.as_mut(), queued, dequeued);
                return;
            }
        }
        let live = self.live.as_mut().expect("loaded above");
        let mut tap = Tap {
            inner: sink.as_mut(),
            counts: None,
        };
        // Everything since `dequeued` that is not the restore: the load and the checks around it.
        let load = dequeued.elapsed();
        let ending = run_turn(
            live,
            &req,
            &mut tap,
            &self.unload_now,
            queued,
            dequeued,
            load,
            self.spec.as_mut(),
        );
        let counts = tap.counts;
        match ending {
            Ending::Done(finish) => {
                if let Some(c) = counts {
                    self.reused += u64::from(c.reused);
                    self.prefilled += u64::from(c.prefilled);
                }
                self.after_done(finish != Finish::Unloaded);
            }
            // The client got its `Done`; the last burst's replay faulted, so the live state is gone. Counted with the
            // errors after a `Done` that no client sees, no snapshot is taken of it, and the next request starts fresh.
            Ending::Unsettled(_, e) => {
                if let Some(c) = counts {
                    self.reused += u64::from(c.reused);
                    self.prefilled += u64::from(c.prefilled);
                }
                self.after_done_errors += 1;
                if e.kind() == ErrorClass::DeviceLost {
                    self.recover();
                } else {
                    live.mark_unusable();
                }
            }
            Ending::Failed(e) => match e.kind() {
                ErrorClass::DeviceLost => self.recover(),
                // The request was refused before it wrote anything: the live state is as it was.
                ErrorClass::BadRequest => {}
                _ => live.mark_unusable(),
            },
        }
        self.last_active = Instant::now();
    }

    fn unload_pending(&self) -> bool {
        self.unload_now.load(Ordering::SeqCst) > 0
    }

    /// `Done { Unloaded }` for a request an unload overtook before it started.
    fn end_unloaded(
        &self,
        req: &Request,
        sink: &mut dyn EventSink,
        queued: Instant,
        dequeued: Instant,
    ) {
        sink.send(Event::Done {
            finish: Finish::Unloaded,
            counts: Counts {
                prompt: u32::try_from(req.prompt.len()).unwrap_or(u32::MAX),
                ..Counts::default()
            },
            timings: Timings {
                queue: dequeued.saturating_duration_since(queued),
                load: dequeued.elapsed(),
                ..Timings::default()
            },
            cache: CacheDelta::default(),
            restore_bytes: 0,
            pool_bytes_used: self
                .live
                .as_ref()
                .map_or(0, |l| l.reuse_stats().pool_bytes_used),
            spec: None,
        });
    }

    /// After a `Done`: the reply-end snapshot, then the budget check unless `budget` is false (an `Unloaded` ending:
    /// the park and unload come instead). The client has its answer, so an error here reaches
    /// nobody: a device loss recovers, any other marks the live state unusable and is counted.
    fn after_done(&mut self, budget: bool) {
        let live = self.live.as_mut().expect("a turn just ran");
        let result = live.snapshot(Marks::REPLY_END).and_then(|()| {
            if budget {
                live.check_budget()
            } else {
                Ok(())
            }
        });
        if let Err(e) = result {
            if e.kind() == ErrorClass::DeviceLost {
                self.recover();
            } else {
                live.mark_unusable();
                self.after_done_errors += 1;
            }
        }
    }

    /// A lost device: unload without parking (the snapshots may sit in lost memory) and reload at once.
    fn recover(&mut self) {
        self.device_losses += 1;
        self.set_state(Lifecycle::Recovering);
        self.notify(WorkerEvent::DeviceLost);
        self.spec = None;
        if let Some(live) = self.live.take() {
            self.loader.unload(live);
        }
        // A failure leaves the worker `Idle`, and the requests queued behind it fail with it.
        if let Err(e) = self.load_model(true) {
            self.fail_queue(&e.message);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::{Call, FakeLoader, LoaderCall, LoaderLog, VOCAB};
    use crate::sample::Sampling;
    use crate::spec::SpecDelta;
    use crate::Source;
    use std::sync::atomic::AtomicBool;
    use std::sync::Mutex;

    const WAIT: Duration = Duration::from_secs(5);
    const CTX: usize = 64;

    /// A sink that tags what it is sent with its id into a channel shared by several, and closes when told.
    struct ProbeSink {
        id: usize,
        tx: Sender<(usize, Event)>,
        closed: Arc<AtomicBool>,
    }

    impl EventSink for ProbeSink {
        fn send(&mut self, event: Event) -> bool {
            !self.closed.load(Ordering::SeqCst) && self.tx.send((self.id, event)).is_ok()
        }

        fn closed(&self) -> bool {
            self.closed.load(Ordering::SeqCst)
        }
    }

    /// The receiving end of one or more `ProbeSink`s.
    struct Probe {
        rx: Receiver<(usize, Event)>,
        tx: Sender<(usize, Event)>,
        closed: Vec<Arc<AtomicBool>>,
    }

    impl Probe {
        fn new() -> Probe {
            let (tx, rx) = channel();
            Probe {
                rx,
                tx,
                closed: Vec::new(),
            }
        }

        /// The next sink; its id is the order it was made in.
        fn sink(&mut self) -> Box<dyn EventSink> {
            let closed = Arc::new(AtomicBool::new(false));
            self.closed.push(Arc::clone(&closed));
            Box::new(ProbeSink {
                id: self.closed.len() - 1,
                tx: self.tx.clone(),
                closed,
            })
        }

        fn close(&self, id: usize) {
            self.closed[id].store(true, Ordering::SeqCst);
        }

        fn next(&self) -> (usize, Event) {
            self.rx.recv_timeout(WAIT).expect("an event within 5 s")
        }

        /// Events of sink `id` up to and including its `Done` or `Error`.
        fn until_end(&self, id: usize) -> Vec<Event> {
            let mut events = Vec::new();
            loop {
                let (from, e) = self.next();
                assert_eq!(from, id, "an event of another request: {e:?}");
                let end = matches!(e, Event::Done { .. } | Event::Error { .. });
                events.push(e);
                if end {
                    return events;
                }
            }
        }

        /// Nothing more will come: every sink is dropped and the channel is empty.
        fn assert_silent(&mut self) {
            // Drop our own sender so the channel can disconnect once the worker's sinks are gone.
            let (tx, _) = channel();
            self.tx = tx;
            match self.rx.recv_timeout(WAIT) {
                Err(RecvTimeoutError::Disconnected) => {}
                other => panic!("expected silence, got {other:?}"),
            }
        }
    }

    /// A reply into a channel.
    fn reply<T: Send + 'static>() -> (Reply<T>, Receiver<T>) {
        let (tx, rx) = channel();
        (
            Box::new(move |v| {
                let _ = tx.send(v);
            }),
            rx,
        )
    }

    fn recv<T>(rx: &Receiver<T>) -> T {
        rx.recv_timeout(WAIT).expect("an answer within 5 s")
    }

    /// Blocks the thread that calls `pass` until the test calls `release`; the test learns the thread arrived through
    /// `arrived`.
    struct Gate {
        arrived: Receiver<()>,
        release: Sender<()>,
    }

    /// The worker's end of a `Gate`; clones share the one gate.
    #[derive(Clone)]
    struct GatePass(Arc<Mutex<(Sender<()>, Receiver<()>)>>);

    fn gate() -> (Gate, GatePass) {
        let (a_tx, a_rx) = channel();
        let (r_tx, r_rx) = channel();
        (
            Gate {
                arrived: a_rx,
                release: r_tx,
            },
            GatePass(Arc::new(Mutex::new((a_tx, r_rx)))),
        )
    }

    impl Gate {
        fn wait_arrived(&self) {
            self.arrived.recv_timeout(WAIT).expect("the gate reached");
        }

        fn release(&self) {
            self.release.send(()).expect("the gate is waiting");
        }
    }

    impl GatePass {
        /// Blocks until released, or until the test is gone.
        fn pass(&self) {
            let g = self.0.lock().unwrap_or_else(|e| e.into_inner());
            let _ = g.0.send(());
            let _ = g.1.recv_timeout(WAIT);
        }
    }

    fn params(ctx: usize) -> WorkerParams {
        WorkerParams {
            idle_unload: None,
            park_snapshots: true,
            ctx,
            n_vocab: VOCAB,
            notify: None,
            speculation: None,
        }
    }

    /// The model picks 7 whatever the history.
    fn sevens() -> FakeLoader {
        FakeLoader::new(CTX, |_| 7)
    }

    fn spawn(loader: FakeLoader) -> (Worker, Arc<LoaderLog>) {
        let log = loader.log();
        (Worker::spawn_with(loader, params(CTX)), log)
    }

    fn prompt(n: usize) -> Vec<u32> {
        (0..n).map(|i| (i * 7 % 50) as u32).collect()
    }

    /// A greedy request for `prompt(n)`, 3 tokens, stop token 63, no cut.
    fn req(n: usize) -> Request {
        Request {
            prompt: prompt(n),
            cut: None,
            sampling: Sampling::GREEDY,
            seed: 1,
            max_tokens: 3,
            stop_tokens: vec![63],
            think: None,
            logprobs: None,
        }
    }

    fn sync(w: &Worker) -> WorkerStats {
        let (r, rx) = reply();
        w.stats(r).unwrap();
        recv(&rx)
    }

    /// Runs one request to its end.
    fn run(w: &Worker, r: Request) -> Vec<Event> {
        let mut p = Probe::new();
        w.submit(r, p.sink()).unwrap();
        p.until_end(0)
    }

    fn unload(w: &Worker) -> UnloadInfo {
        let (r, rx) = reply();
        w.unload(r).unwrap();
        recv(&rx)
    }

    fn load(w: &Worker) -> Result<LoadInfo, LoadError> {
        let (r, rx) = reply();
        w.load(r).unwrap();
        recv(&rx)
    }

    fn finish(events: &[Event]) -> Finish {
        match events.last() {
            Some(Event::Done { finish, .. }) => *finish,
            other => panic!("not Done: {other:?}"),
        }
    }

    fn started(events: &[Event]) -> (u32, Source) {
        match events.first() {
            Some(Event::Started {
                reused_tokens,
                restored,
                ..
            }) => (*reused_tokens, *restored),
            other => panic!("not Started: {other:?}"),
        }
    }

    fn error(events: &[Event]) -> (ErrorKind, String) {
        match events.last() {
            Some(Event::Error { kind, message }) => (*kind, message.clone()),
            other => panic!("not Error: {other:?}"),
        }
    }

    fn device_lost() -> Error {
        Error::Vk(vk::Error::DeviceLost { in_flight: None })
    }

    fn non_finite() -> Error {
        Error::NonFinite {
            pos: 3,
            index: 1,
            value: f32::NAN,
        }
    }

    /// A VRAM `DoesNotFit`.
    fn vram_short() -> Error {
        Error::DoesNotFit {
            what: "the expert cache",
            need: 2 << 30,
            free: 1 << 30,
        }
    }

    fn host_short() -> Error {
        Error::DoesNotFit {
            what: "the expert copies in host memory",
            need: 30 << 30,
            free: 10 << 30,
        }
    }

    /// A host heap's budget refusing an allocation during the load.
    fn host_budget() -> Error {
        Error::Vk(vk::Error::OutOfBudget {
            what: "the experts of layer 7".into(),
            want: 400 << 20,
            budget: 20_000 << 20,
            usage: 19_900 << 20,
            host: true,
        })
    }

    fn fail(error: Error, slots: Option<usize>) -> Result<(), LoadFailure> {
        Err(LoadFailure { error, slots })
    }

    /// Polls up to 5 s for the worker to read `want`.
    fn wait_state(w: &Worker, want: Lifecycle) {
        let deadline = Instant::now() + WAIT;
        while w.state() != want {
            assert!(Instant::now() < deadline, "state stayed {:?}", w.state());
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// The decodes the fakes ran, in order.
    fn decodes(log: &LoaderLog) -> Vec<u32> {
        log.live_calls()
            .into_iter()
            .filter_map(|c| match c {
                Call::Decode(t) => Some(t),
                _ => None,
            })
            .collect()
    }

    fn prefills(log: &LoaderLog) -> usize {
        log.live_calls()
            .iter()
            .filter(|c| matches!(c, Call::Prefill(..)))
            .count()
    }

    /// Blocks the worker thread in the prefill the fake runs.
    fn block_prefill(live: &mut crate::fake::FakeLive, pass: &GatePass) {
        let pass = pass.clone();
        live.on_call(move |c| {
            if matches!(c, Call::Prefill(..)) {
                pass.pass();
            }
        });
    }

    #[test]
    fn requests_run_in_arrival_order() {
        let (w, _) = spawn(sevens());
        let mut p = Probe::new();
        for _ in 0..3 {
            let s = p.sink();
            w.submit(req(5), s).unwrap();
        }
        let mut order = Vec::new();
        let mut done = 0;
        while done < 3 {
            match p.next() {
                (id, Event::Started { .. }) => {
                    assert_eq!(id, done, "a request started before the last was done");
                    order.push(id);
                }
                (id, Event::Done { .. }) => {
                    assert_eq!(id, done);
                    done += 1;
                }
                _ => {}
            }
        }
        assert_eq!(order, vec![0, 1, 2]);
        w.shutdown().unwrap();
    }

    #[test]
    fn a_request_whose_sink_closed_in_the_queue_never_runs() {
        let (g, pass) = gate();
        let loader = sevens().tweak(move |_, live| block_prefill(live, &pass));
        let (w, log) = spawn(loader);
        let mut p = Probe::new();
        w.submit(req(5), p.sink()).unwrap();
        g.wait_arrived();
        w.submit(req(5), p.sink()).unwrap();
        p.close(1);
        g.release();
        assert_eq!(finish(&p.until_end(0)), Finish::Length);
        sync(&w);
        assert_eq!(prefills(&log), 1, "the closed request prefilled");
        p.assert_silent();
    }

    #[test]
    fn a_request_that_fails_its_check_gets_an_internal_error_without_touching_the_state() {
        let (w, log) = spawn(sevens());
        let mut bad = req(5);
        bad.prompt = vec![99];
        let (kind, message) = error(&run(&w, bad));
        assert_eq!(kind, ErrorKind::Internal);
        assert!(message.contains("prompt holds 99"), "{message}");
        assert_eq!(w.state(), Lifecycle::Idle);
        assert!(log.calls().is_empty(), "the check loaded the model");
    }

    #[test]
    fn a_reply_end_snapshot_follows_every_ending() {
        // The live state's tokens at the end are the prompt and every token decoded; the snapshot is at that length.
        fn check(log: &LoaderLog, w: &Worker, prompt_len: usize, want: bool) {
            let info = unload(w);
            let n = prompt_len + decodes(log).len();
            let taken = log
                .live_calls()
                .contains(&Call::Snapshot(n, Marks::REPLY_END));
            let nodes = log.parks().pop().expect("the unload parked");
            let held = nodes.iter().any(|(t, m)| t.len() == n && m.reply_end);
            assert_eq!((taken, held), (want, want), "{nodes:?}");
            assert_eq!(info.parked, nodes.len());
        }
        // Stop: the model picks 63 once the history reaches 7 tokens.
        let (w, log) = spawn(FakeLoader::new(CTX, |h| if h.len() < 7 { 7 } else { 63 }));
        assert!(matches!(finish(&run(&w, req(5))), Finish::Stop(63)));
        check(&log, &w, 5, true);
        // Length.
        let (w, log) = spawn(sevens());
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        check(&log, &w, 5, true);
        // ContextFull: 8 positions, a prompt of 5.
        let loader = FakeLoader::new(8, |_| 7);
        let log = loader.log();
        let w = Worker::spawn_with(loader, params(8));
        let mut long = req(5);
        long.max_tokens = 10;
        assert_eq!(finish(&run(&w, long)), Finish::ContextFull);
        check(&log, &w, 5, true);
        // Cancelled: the sink closes during the first decode.
        let (g, pass) = gate();
        let loader = sevens().tweak(move |_, live| {
            let pass = pass.clone();
            live.on_call(move |c| {
                if matches!(c, Call::Decode(_)) {
                    pass.pass();
                }
            });
        });
        let (w, log) = spawn(loader);
        let mut p = Probe::new();
        w.submit(req(5), p.sink()).unwrap();
        g.wait_arrived();
        p.close(0);
        g.release();
        sync(&w);
        // Cancelled, not Length: the first decode was the last (a Length run decodes twice).
        assert_eq!(decodes(&log).len(), 1);
        check(&log, &w, 5, true);
        // Unloaded: an unload lands during the prefill.
        let (g, pass) = gate();
        let (w, log) = spawn(sevens().tweak(move |_, live| block_prefill(live, &pass)));
        let mut p = Probe::new();
        w.submit(req(5), p.sink()).unwrap();
        g.wait_arrived();
        let (r, rx) = reply();
        w.unload(r).unwrap();
        g.release();
        assert_eq!(finish(&p.until_end(0)), Finish::Unloaded);
        recv(&rx);
        let nodes = log.parks().pop().expect("the unload parked");
        assert!(
            nodes.iter().any(|(t, m)| t.len() == 5 && m.reply_end),
            "{nodes:?}"
        );
        // Error: no snapshot.
        let (w, log) = spawn(sevens().tweak(|_, live| live.fail_at = Some((1, non_finite()))));
        assert_eq!(error(&run(&w, req(5))).0, ErrorKind::NonFinite);
        assert!(
            !log.live_calls()
                .iter()
                .any(|c| matches!(c, Call::Snapshot(..))),
            "an Error took a snapshot"
        );
        check(&log, &w, 5, false);
    }

    #[test]
    fn unload_now_in_prefill_ends_unloaded_parks_and_the_next_request_resumes() {
        let (g, pass) = gate();
        let loader = sevens().tweak(move |i, live| {
            if i == 0 {
                block_prefill(live, &pass);
            }
        });
        let (w, log) = spawn(loader);
        let mut p = Probe::new();
        w.submit(req(10), p.sink()).unwrap();
        g.wait_arrived();
        let (r, rx) = reply();
        w.unload(r).unwrap();
        g.release();
        assert_eq!(finish(&p.until_end(0)), Finish::Unloaded);
        let info = recv(&rx);
        assert!(info.was_loaded);
        assert_eq!(info.parked, 1, "the reply-end snapshot parks with the rest");
        assert!(info.parked_bytes > 0);
        wait_state(&w, Lifecycle::Idle);
        let mut again = req(10);
        again.prompt.extend([9, 9]);
        let events = run(&w, again);
        assert_eq!(started(&events), (10, Source::ReplyEnd));
        assert_eq!(log.loads(), vec![(None, false), (None, true)]);
    }

    #[test]
    fn an_unloaded_ending_takes_the_snapshot_but_runs_no_budget_check() {
        // The park and unload come instead of the budget check.
        let (g, pass) = gate();
        let loader = sevens().tweak(move |i, live| {
            if i == 0 {
                block_prefill(live, &pass);
                live.fail_budget = Some(Error::WrongEngine);
            }
        });
        let (w, log) = spawn(loader);
        let mut p = Probe::new();
        w.submit(req(10), p.sink()).unwrap();
        g.wait_arrived();
        let (r, rx) = reply();
        w.unload(r).unwrap();
        g.release();
        assert_eq!(finish(&p.until_end(0)), Finish::Unloaded);
        assert_eq!(recv(&rx).parked, 1, "the snapshot was taken");
        assert_eq!(sync(&w).after_done_errors, 0, "a budget check ran");
        assert_eq!(log.unusable_marks(), 0);
    }

    #[test]
    fn a_request_queued_before_the_unload_ends_unloaded_without_loading() {
        let (g, pass) = gate();
        let (w, log) = spawn(sevens().tweak(move |_, live| block_prefill(live, &pass)));
        let mut p = Probe::new();
        w.submit(req(5), p.sink()).unwrap();
        g.wait_arrived();
        w.submit(req(5), p.sink()).unwrap();
        let (r, rx) = reply();
        w.unload(r).unwrap();
        g.release();
        assert_eq!(finish(&p.until_end(0)), Finish::Unloaded);
        let second = p.until_end(1);
        assert_eq!(second.len(), 1, "no Started: {second:?}");
        assert_eq!(finish(&second), Finish::Unloaded);
        assert!(recv(&rx).was_loaded);
        assert_eq!(log.loads().len(), 1);
        assert_eq!(prefills(&log), 1);
    }

    #[test]
    fn two_unloads_in_a_row_both_answer_and_leave_the_counter_at_zero() {
        let (w, _) = spawn(sevens());
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        let (r1, rx1) = reply();
        let (r2, rx2) = reply();
        w.unload(r1).unwrap();
        w.unload(r2).unwrap();
        assert!(recv(&rx1).was_loaded);
        assert!(!recv(&rx2).was_loaded);
        // A pending count left over would end this request Unloaded.
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
    }

    #[test]
    fn an_unload_during_a_lazy_load_ends_unloaded_before_started() {
        let (g, pass) = gate();
        let (w, log) = spawn(sevens().on_load(move |_| pass.pass()));
        let mut p = Probe::new();
        w.submit(req(5), p.sink()).unwrap();
        g.wait_arrived();
        let (r, rx) = reply();
        w.unload(r).unwrap();
        g.release();
        let events = p.until_end(0);
        assert_eq!(events.len(), 1, "no Started: {events:?}");
        assert_eq!(finish(&events), Finish::Unloaded);
        assert!(recv(&rx).was_loaded, "the Unload unloads what just loaded");
        assert_eq!(log.loads().len(), 1);
        assert_eq!(prefills(&log), 0);
        wait_state(&w, Lifecycle::Idle);
    }

    #[test]
    fn a_sink_closed_during_the_load_is_dropped_before_started() {
        let (g, pass) = gate();
        let (w, log) = spawn(sevens().on_load(move |_| pass.pass()));
        let mut p = Probe::new();
        w.submit(req(5), p.sink()).unwrap();
        g.wait_arrived();
        p.close(0);
        g.release();
        sync(&w);
        assert_eq!(prefills(&log), 0);
        assert_eq!(w.state(), Lifecycle::Ready, "the model stays loaded");
        p.assert_silent();
    }

    #[test]
    fn the_idle_timer_parks_and_unloads_between_requests() {
        let loader = sevens().tweak(|i, live| {
            if i == 0 {
                // Three times the timer: the request must not be cut short by it.
                live.on_call(|c| {
                    if matches!(c, Call::Prefill(..)) {
                        std::thread::sleep(Duration::from_millis(150));
                    }
                });
            }
        });
        let log = loader.log();
        let mut cfg = params(CTX);
        cfg.idle_unload = Some(Duration::from_millis(50));
        let w = Worker::spawn_with(loader, cfg);
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        assert!(
            log.wait_for(|c| c.iter().any(|c| matches!(c, LoaderCall::Park(_)))),
            "the timer never parked"
        );
        wait_state(&w, Lifecycle::Idle);
        assert_eq!(sync(&w).parked, 1);
        // The next request loads again, with the parked snapshots.
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        assert_eq!(log.loads(), vec![(None, false), (None, true)]);
    }

    #[test]
    fn a_vram_shortage_with_a_known_slot_count_retries_once_with_half_of_it() {
        let loader = sevens().script_loads(vec![fail(vram_short(), Some(128)), Ok(())]);
        let (w, log) = spawn(loader);
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        assert_eq!(log.loads(), vec![(None, false), (Some(64), false)]);
        let info = load(&w).expect("loaded");
        assert_eq!(info.retried_with_slots, Some(64));
        assert_eq!(info.slots_per_layer, 64);

        // The retry runs with the parked snapshots still held.
        let loader = sevens().script_loads(vec![Ok(()), fail(vram_short(), Some(128)), Ok(())]);
        let (w, log) = spawn(loader);
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        assert_eq!(unload(&w).parked, 1);
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        assert_eq!(
            log.loads(),
            vec![(None, false), (None, true), (Some(64), true)]
        );
        assert_eq!(sync(&w).parked, 0);

        // A memory class again: the message keeps both attempts' figures, and there is no third call.
        let loader = sevens().script_loads(vec![
            fail(vram_short(), Some(128)),
            fail(vram_short(), Some(64)),
            fail(vram_short(), Some(128)),
            fail(vram_short(), Some(64)),
        ]);
        let (w, log) = spawn(loader);
        let (kind, message) = error(&run(&w, req(5)));
        let expected = vram_short_message(128, &vram_short(), 64, &vram_short());
        assert_eq!(
            (kind, message.as_str()),
            (ErrorKind::LoadFailed, expected.as_str())
        );
        for part in ["128", "64", "2048 MiB", "Close programs that use the GPU"] {
            assert!(message.contains(part), "{part} missing from {message}");
        }
        assert_eq!(log.loads(), vec![(None, false), (Some(64), false)]);
        assert_eq!(w.state(), Lifecycle::Idle);
        assert_eq!(
            load(&w),
            Err(LoadError {
                kind: ErrorKind::OutOfMemory,
                message: expected
            })
        );
        assert_eq!(log.loads().len(), 4);
    }

    #[test]
    fn no_slot_count_means_no_retry() {
        let (w, log) = spawn(sevens().script_loads(vec![fail(vram_short(), None)]));
        assert_eq!(
            load(&w),
            Err(LoadError {
                kind: ErrorKind::OutOfMemory,
                message: vram_short().to_string()
            })
        );
        assert_eq!(log.loads().len(), 1);
    }

    #[test]
    fn a_slot_count_of_one_is_not_halved() {
        let (w, log) = spawn(sevens().script_loads(vec![fail(vram_short(), Some(1))]));
        assert_eq!(
            load(&w),
            Err(LoadError {
                kind: ErrorKind::OutOfMemory,
                message: vram_short().to_string()
            })
        );
        assert_eq!(log.loads().len(), 1);
    }

    #[test]
    fn a_host_memory_shortage_gets_its_own_message_and_no_retry() {
        let (w, log) = spawn(sevens().script_loads(vec![fail(host_short(), Some(128))]));
        let got = load(&w);
        assert_eq!(
            got,
            Err(LoadError {
                kind: ErrorKind::OutOfMemory,
                message: host_short_message(&host_short())
            })
        );
        let message = got.unwrap_err().message;
        for part in ["30720 MiB", "10240 MiB", "Close other programs"] {
            assert!(message.contains(part), "{part} missing from {message}");
        }
        assert_eq!(log.loads().len(), 1);
        // Any other class keeps the engine's message and is not retried either.
        let (w, log) = spawn(sevens().script_loads(vec![fail(Error::WrongEngine, Some(128))]));
        assert_eq!(
            load(&w),
            Err(LoadError {
                kind: ErrorKind::Internal,
                message: Error::WrongEngine.to_string()
            })
        );
        assert_eq!(log.loads().len(), 1);
    }

    #[test]
    fn a_host_budget_refusal_at_load_gets_the_host_message_and_no_retry() {
        let (w, log) = spawn(sevens().script_loads(vec![fail(host_budget(), Some(128))]));
        let got = load(&w);
        assert_eq!(
            got,
            Err(LoadError {
                kind: ErrorKind::OutOfMemory,
                message: host_short_message(&host_budget())
            })
        );
        let message = got.unwrap_err().message;
        assert!(message.contains("Close other programs"), "{message}");
        assert!(!message.contains("GPU"), "{message}");
        assert_eq!(log.loads().len(), 1);
        // A VRAM-kind budget refusal still retries with half the slots.
        let vram_budget = || {
            Error::Vk(vk::Error::OutOfBudget {
                what: "the expert cache".into(),
                want: 2 << 30,
                budget: 1 << 30,
                usage: 0,
                host: false,
            })
        };
        let (w, log) = spawn(sevens().script_loads(vec![
            fail(vram_budget(), Some(128)),
            fail(vram_budget(), Some(64)),
        ]));
        let got = load(&w).unwrap_err();
        assert!(
            got.message.starts_with("not enough GPU memory"),
            "{}",
            got.message
        );
        assert_eq!(log.loads().len(), 2);
    }

    #[test]
    fn a_failed_load_fails_every_queued_request_and_keeps_the_other_commands_in_order() {
        let (g, pass) = gate();
        let loader = sevens()
            .script_loads(vec![fail(Error::WrongEngine, None)])
            .on_load(move |i| {
                if i == 0 {
                    pass.pass();
                }
            });
        let (w, log) = spawn(loader);
        let mut p = Probe::new();
        w.submit(req(5), p.sink()).unwrap();
        g.wait_arrived();
        // Queued behind the load: a request, an unload, a request, a stats.
        let (order_tx, order_rx) = channel();
        w.submit(req(5), p.sink()).unwrap();
        let tx = order_tx.clone();
        w.unload(Box::new(move |u| {
            tx.send(("unload", Some(u), None)).unwrap();
        }))
        .unwrap();
        w.submit(req(5), p.sink()).unwrap();
        let tx = order_tx;
        w.stats(Box::new(move |s| {
            tx.send(("stats", None, Some(s))).unwrap();
        }))
        .unwrap();
        g.release();
        let message = Error::WrongEngine.to_string();
        // The queue is failed before the request that failed the load is answered, so a retry can't be drained.
        for id in [1, 2, 0] {
            let (kind, m) = error(&p.until_end(id));
            assert_eq!((kind, m), (ErrorKind::LoadFailed, message.clone()), "{id}");
        }
        let (name, unloaded, _) = recv(&order_rx);
        assert_eq!(name, "unload");
        assert!(!unloaded.unwrap().was_loaded);
        let (name, _, stats) = recv(&order_rx);
        assert_eq!(name, "stats");
        assert_eq!(stats.unwrap().state, Lifecycle::Idle);
        assert_eq!(log.loads().len(), 1);
    }

    #[test]
    fn the_next_request_after_a_failed_load_retries() {
        let loader = sevens().script_loads(vec![Ok(()), fail(Error::WrongEngine, None), Ok(())]);
        let (w, _) = spawn(loader);
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        assert_eq!(unload(&w).parked, 1);
        let (kind, _) = error(&run(&w, req(5)));
        assert_eq!(kind, ErrorKind::LoadFailed);
        assert_eq!(w.state(), Lifecycle::Idle);
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        assert_eq!(w.state(), Lifecycle::Ready);
    }

    #[test]
    fn a_failed_load_keeps_the_parked_snapshots_for_the_next_try() {
        let loader = sevens().script_loads(vec![Ok(()), fail(Error::WrongEngine, None), Ok(())]);
        let (w, log) = spawn(loader);
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        assert_eq!(unload(&w).parked, 1);
        assert_eq!(error(&run(&w, req(5))).0, ErrorKind::LoadFailed);
        assert_eq!(sync(&w).parked, 1, "the failed load dropped the snapshots");
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        assert_eq!(sync(&w).parked, 0, "the load that worked took them");
        assert_eq!(log.loads(), vec![(None, false), (None, true), (None, true)]);
    }

    #[test]
    fn a_device_loss_unloads_without_parking_and_reloads() {
        let (g, pass) = gate();
        let loader = sevens()
            .tweak(|i, live| {
                if i == 0 {
                    live.fail_at = Some((2, device_lost()));
                }
            })
            .on_load(move |i| {
                if i == 1 {
                    pass.pass();
                }
            });
        let (w, log) = spawn(loader);
        let mut long = req(5);
        long.max_tokens = 6;
        let events = run(&w, long);
        assert_eq!(error(&events).0, ErrorKind::DeviceLost);
        g.wait_arrived();
        assert_eq!(w.state(), Lifecycle::Recovering);
        g.release();
        wait_state(&w, Lifecycle::Ready);
        let stats = sync(&w);
        assert_eq!(stats.device_losses, 1);
        assert_eq!(stats.loads, 2);
        assert!(log.parks().is_empty(), "a lost device parked");
        assert_eq!(log.unloads(), 1);
        assert_eq!(log.loads(), vec![(None, false), (None, false)]);

        // A reload that fails leaves the worker idle, and the next request loads again.
        let loader = sevens()
            .script_loads(vec![Ok(()), fail(Error::WrongEngine, None)])
            .tweak(|i, live| {
                if i == 0 {
                    live.fail_at = Some((0, device_lost()));
                }
            });
        let (w, _) = spawn(loader);
        assert_eq!(error(&run(&w, req(5))).0, ErrorKind::DeviceLost);
        wait_state(&w, Lifecycle::Idle);
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
    }

    #[test]
    fn an_error_marks_the_live_state_unusable_and_the_next_request_restores() {
        // r1 takes steps 0-2 and leaves a reply end at 7 tokens; r2 continues it live (prefill: step 3) and fails in
        // its first decode (step 4).
        let (w, log) = spawn(sevens().tweak(|_, live| live.fail_at = Some((4, non_finite()))));
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        assert_eq!(sync(&w).after_done_errors, 0);
        assert_eq!(log.unusable_marks(), 0);
        let mut next = req(5);
        next.prompt.extend([7, 7, 9]);
        let events = run(&w, next.clone());
        assert_eq!(started(&events), (7, Source::Live));
        assert_eq!(error(&events).0, ErrorKind::NonFinite);
        // The worker marks the state itself after an Error, whatever the engine did.
        sync(&w);
        assert_eq!(log.unusable_marks(), 1);
        let events = run(&w, next);
        assert_eq!(started(&events), (7, Source::ReplyEnd), "{events:?}");
    }

    #[test]
    fn a_device_loss_in_the_reply_end_snapshot_recovers() {
        let loader = sevens().tweak(|i, live| {
            if i == 0 {
                live.fail_snapshot = Some(device_lost());
            }
        });
        let (w, log) = spawn(loader);
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        let stats = sync(&w);
        assert_eq!(stats.device_losses, 1);
        assert_eq!(stats.after_done_errors, 0);
        assert_eq!(stats.state, Lifecycle::Ready);
        assert_eq!(log.unloads(), 1);
        assert_eq!(log.loads().len(), 2);
        assert!(log.parks().is_empty());
    }

    #[test]
    fn another_error_in_the_reply_end_snapshot_marks_the_state_unusable_and_counts() {
        let loader = sevens().tweak(|_, live| live.fail_snapshot = Some(Error::WrongEngine));
        let (w, log) = spawn(loader);
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        assert_eq!(sync(&w).after_done_errors, 1);
        assert_eq!(log.unusable_marks(), 1);
        // The snapshot failed, so there is nothing to restore, and the live state is not trusted.
        let mut next = req(5);
        next.prompt.extend([7, 7, 9]);
        assert_eq!(started(&run(&w, next)), (0, Source::None));
    }

    #[test]
    fn a_failed_budget_check_after_done_is_treated_the_same() {
        let loader = sevens().tweak(|_, live| live.fail_budget = Some(Error::WrongEngine));
        let (w, log) = spawn(loader);
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        assert_eq!(sync(&w).after_done_errors, 1);
        assert_eq!(log.unusable_marks(), 1);
        // The reply-end snapshot was taken before the check; the live state is not trusted.
        let mut next = req(5);
        next.prompt.extend([7, 7, 9]);
        assert_eq!(started(&run(&w, next)), (7, Source::ReplyEnd));
    }

    #[test]
    fn a_dropped_worker_shuts_down() {
        let (w, log) = spawn(sevens());
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        drop(w);
        assert!(
            log.wait_for(|c| c.contains(&LoaderCall::Unload)),
            "the dropped worker never unloaded"
        );
        assert!(log.parks().is_empty());
    }

    #[test]
    fn shutdown_unloads_without_parking_and_joins() {
        let (w, log) = spawn(sevens());
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        w.shutdown().unwrap();
        assert_eq!(log.unloads(), 1);
        assert!(log.parks().is_empty());
    }

    #[test]
    fn a_worker_that_panicked_answers_gone() {
        let loader = sevens().tweak(|_, live| {
            live.on_call(|c| {
                if matches!(c, Call::Prefill(..)) {
                    panic!("scripted panic in a prefill");
                }
            });
        });
        let (w, _) = spawn(loader);
        let mut p = Probe::new();
        w.submit(req(5), p.sink()).unwrap();
        let deadline = Instant::now() + WAIT;
        while w.submit(req(5), p.sink()).is_ok() {
            assert!(Instant::now() < deadline, "the worker never ended");
            std::thread::sleep(Duration::from_millis(2));
        }
        let (r, _) = reply::<WorkerStats>();
        assert!(w.stats(r).is_err());
        assert!(w.shutdown().is_err());
    }

    #[test]
    fn park_snapshots_false_unloads_without_parking() {
        let loader = sevens();
        let log = loader.log();
        let mut cfg = params(CTX);
        cfg.park_snapshots = false;
        let w = Worker::spawn_with(loader, cfg);
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        let info = unload(&w);
        assert_eq!(
            (info.was_loaded, info.parked, info.parked_bytes),
            (true, 0, 0)
        );
        assert!(log.parks().is_empty());
        assert_eq!(log.unloads(), 1);
    }

    #[test]
    fn stats_answer_without_loading() {
        let (w, log) = spawn(sevens());
        let s = sync(&w);
        assert_eq!(s.state, Lifecycle::Idle);
        assert!(s.cache.is_none());
        assert_eq!((s.parked, s.loads, s.device_losses), (0, 0, 0));
        assert!(log.calls().is_empty());
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        let s = sync(&w);
        assert_eq!(s.state, Lifecycle::Ready);
        assert!(s.cache.is_some());
        assert_eq!(s.loads, 1);
    }

    #[test]
    fn load_twice_loads_once() {
        let (w, log) = spawn(sevens().load_time(Duration::from_millis(9)));
        let first = load(&w).expect("loaded");
        assert_eq!(first.time, Duration::from_millis(9));
        let second = load(&w).expect("still loaded");
        assert_eq!(first, second);
        assert_eq!(log.loads().len(), 1);
        assert_eq!(sync(&w).loads, 1);
    }

    #[test]
    fn lifecycle_reads_loading_during_a_slow_load() {
        let (g, pass) = gate();
        let (w, _) = spawn(sevens().on_load(move |_| pass.pass()));
        let (r, rx) = reply();
        w.load(r).unwrap();
        g.wait_arrived();
        assert_eq!(w.state(), Lifecycle::Loading);
        g.release();
        assert!(recv(&rx).is_ok());
        assert_eq!(w.state(), Lifecycle::Ready);
    }

    #[test]
    fn a_load_that_ends_with_a_closed_sink_starts_the_idle_timer() {
        let (g, pass) = gate();
        let loader = sevens().on_load(move |_| pass.pass());
        let log = loader.log();
        let mut cfg = params(CTX);
        cfg.idle_unload = Some(Duration::from_secs(1));
        let w = Worker::spawn_with(loader, cfg);
        let mut p = Probe::new();
        w.submit(req(5), p.sink()).unwrap();
        g.wait_arrived();
        p.close(0);
        // The load outlasts the timer. A timer counted from the spawn would unload the fresh model at once.
        std::thread::sleep(Duration::from_millis(1200));
        g.release();
        sync(&w);
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(w.state(), Lifecycle::Ready);
        assert_eq!(log.calls().len(), 1, "the fresh model was unloaded");
        // The timer still runs from the load.
        assert!(log.wait_for(|c| c.iter().any(|c| matches!(c, LoaderCall::Park(_)))));
    }

    #[test]
    fn a_failed_park_is_counted_and_the_unload_still_completes() {
        let loader = sevens().fail_parks(vec![Error::WrongEngine, device_lost()]);
        let (w, log) = spawn(loader);
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        let info = unload(&w);
        assert_eq!(
            (info.was_loaded, info.parked, info.parked_bytes),
            (true, 0, 0)
        );
        wait_state(&w, Lifecycle::Idle);
        let s = sync(&w);
        assert_eq!((s.park_errors, s.device_losses, s.parked), (1, 0, 0));
        // A device loss in the park counts as one, too.
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        assert!(unload(&w).was_loaded);
        let s = sync(&w);
        assert_eq!(
            (s.park_errors, s.device_losses, s.state),
            (2, 1, Lifecycle::Idle)
        );
        let failed = log
            .calls()
            .iter()
            .filter(|c| **c == LoaderCall::ParkFailed)
            .count();
        assert_eq!(failed, 2);
        // Nothing was parked, so the next load has nothing to bring back.
        assert_eq!(finish(&run(&w, req(5))), Finish::Length);
        assert_eq!(log.loads().last(), Some(&(None, false)));
    }

    #[test]
    fn the_observer_sees_the_lifecycle() {
        fn tag(e: &WorkerEvent) -> String {
            match e {
                WorkerEvent::Loaded(_) => "Loaded".into(),
                WorkerEvent::Unloaded { cause } => format!("Unloaded({cause:?})"),
                WorkerEvent::LoadFailed { .. } => "LoadFailed".into(),
                WorkerEvent::DeviceLost => "DeviceLost".into(),
                WorkerEvent::ReloadFailed { .. } => "ReloadFailed".into(),
            }
        }
        fn observer() -> (Notify, Receiver<(Option<String>, WorkerEvent)>) {
            let (tx, rx) = channel();
            (
                Box::new(move |e| {
                    let name = std::thread::current().name().map(String::from);
                    let _ = tx.send((name, e));
                }),
                rx,
            )
        }
        // The timer.
        let (notify, rx) = observer();
        let mut cfg = params(CTX);
        cfg.idle_unload = Some(Duration::from_millis(50));
        cfg.notify = Some(notify);
        let w = Worker::spawn_with(sevens(), cfg);
        load(&w).expect("loaded");
        let (thread, e) = recv(&rx);
        assert_eq!(tag(&e), "Loaded");
        assert_eq!(thread.as_deref(), Some("artemis-worker"));
        let (_, e) = recv(&rx);
        assert_eq!(tag(&e), "Unloaded(Idle)");
        w.shutdown().unwrap();
        assert!(
            rx.try_iter().next().is_none(),
            "nothing was loaded to unload"
        );

        // The rest, in one script.
        let (notify, rx) = observer();
        let mut cfg = params(CTX);
        cfg.notify = Some(notify);
        let loader = sevens()
            .script_loads(vec![
                fail(Error::WrongEngine, None),
                Ok(()),
                Ok(()),
                fail(Error::WrongEngine, None),
            ])
            .tweak(|i, live| {
                if i == 1 || i == 2 {
                    live.fail_at = Some((0, device_lost()));
                }
            });
        let w = Worker::spawn_with(loader, cfg);
        assert!(load(&w).is_err());
        assert_eq!(error(&run(&w, req(5))).0, ErrorKind::DeviceLost);
        assert_eq!(error(&run(&w, req(5))).0, ErrorKind::DeviceLost);
        load(&w).expect("loaded");
        unload(&w);
        load(&w).expect("loaded");
        w.shutdown().unwrap();
        let seen: Vec<(Option<String>, WorkerEvent)> = rx.iter().collect();
        let tags: Vec<String> = seen.iter().map(|(_, e)| tag(e)).collect();
        assert_eq!(
            tags,
            [
                "LoadFailed",
                "Loaded",
                "DeviceLost",
                "Loaded",
                "DeviceLost",
                "ReloadFailed",
                "Loaded",
                "Unloaded(Requested)",
                "Loaded",
                "Unloaded(Shutdown)"
            ]
        );
        assert!(
            seen.iter()
                .all(|(n, _)| n.as_deref() == Some("artemis-worker")),
            "an event came from another thread"
        );
        match &seen[0].1 {
            WorkerEvent::LoadFailed { message } => {
                assert_eq!(*message, Error::WrongEngine.to_string())
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn loaded_reports_the_vram_used() {
        let (tx, rx) = channel();
        let mut cfg = params(CTX);
        cfg.notify = Some(Box::new(move |e| {
            let _ = tx.send(e);
        }));
        let w = Worker::spawn_with(sevens().vram_used(|n| Some(100 + n as u64)), cfg);
        load(&w).expect("loaded");
        unload(&w);
        load(&w).expect("loaded");
        let used: Vec<Option<u64>> = rx
            .try_iter()
            .filter_map(|e| match e {
                WorkerEvent::Loaded(info) => Some(info.vram_used),
                _ => None,
            })
            .collect();
        assert_eq!(used, [Some(100), Some(101)]);
        w.shutdown().unwrap();
    }

    #[test]
    fn park_time_delays_the_next_command() {
        let (tx, rx) = channel();
        let mut cfg = params(CTX);
        cfg.idle_unload = Some(Duration::from_millis(50));
        cfg.notify = Some(Box::new(move |e| {
            let _ = tx.send(e);
        }));
        let park = Duration::from_millis(300);
        let w = Worker::spawn_with(sevens().park_time(park), cfg);
        load(&w).expect("loaded");
        let loaded = Instant::now();
        // The idle timer fires 50 ms after the load and the park then takes 300 ms; ask in the middle of it.
        std::thread::sleep(Duration::from_millis(150));
        let sent = Instant::now();
        let s = sync(&w);
        // A stall that put the command after the park would pass the checks below without testing the delay.
        assert!(
            sent.duration_since(loaded) < park,
            "the stats went out after the park"
        );
        assert!(
            loaded.elapsed() >= park,
            "the stats were answered {:?} after the load, during the park",
            loaded.elapsed()
        );
        let events: Vec<WorkerEvent> = rx.try_iter().collect();
        assert!(
            matches!(
                events.as_slice(),
                [
                    WorkerEvent::Loaded(_),
                    WorkerEvent::Unloaded {
                        cause: UnloadCause::Idle
                    }
                ]
            ),
            "{events:?}"
        );
        assert_eq!(s.state, Lifecycle::Idle);
        w.shutdown().unwrap();
    }

    #[test]
    fn the_worker_accumulates_reused_and_prefilled_tokens() {
        let (w, _) = spawn(sevens());
        assert_eq!(finish(&run(&w, req(10))), Finish::Length);
        let s = sync(&w).reuse;
        assert_eq!((s.reused_tokens, s.prefilled_tokens), (0, 10));
        // The second continues the first live: 12 tokens reused, 3 prefilled.
        let mut next = req(10);
        next.prompt.extend([7, 7, 9, 9, 9]);
        assert_eq!(finish(&run(&w, next)), Finish::Length);
        let s = sync(&w).reuse;
        assert_eq!((s.reused_tokens, s.prefilled_tokens), (12, 13));
        // A load starts the counts again.
        unload(&w);
        assert_eq!(finish(&run(&w, req(10))), Finish::Length);
        let s = sync(&w).reuse;
        assert_eq!((s.reused_tokens, s.prefilled_tokens), (0, 10));
    }

    // ---- speculation ----

    /// Speculation on, with a fixed draft length (the wall-clock controller would pick plain steps on a fake).
    fn spec_params(ctx: usize) -> WorkerParams {
        WorkerParams {
            speculation: Some(SpecConfig {
                max_draft: 4,
                fixed_draft: Some(4),
                ..SpecConfig::default()
            }),
            ..params(ctx)
        }
    }

    /// The model counts round `10 11 12 13 14 15 10 ...`, so prompt lookup finds every draft it needs; `stop_at` makes it
    /// pick the stop token 63 once the history reaches that many tokens.
    fn cycling(stop_at: usize) -> FakeLoader {
        FakeLoader::new(CTX, move |h| match h.last() {
            _ if h.len() >= stop_at => 63,
            Some(&t) if (10..16).contains(&t) => 10 + (t - 10 + 1) % 6,
            _ => 10,
        })
    }

    /// A request for a prompt of 14 tokens of the cycle, `max_tokens` long.
    fn cyclic(max_tokens: u32) -> Request {
        Request {
            prompt: (0..14).map(|i| 10 + i % 6).collect(),
            max_tokens,
            ..req(0)
        }
    }

    fn done_spec(events: &[Event]) -> Option<SpecDelta> {
        match events.last() {
            Some(Event::Done { spec, .. }) => *spec,
            other => panic!("not Done: {other:?}"),
        }
    }

    #[test]
    fn the_spec_totals_are_per_load_and_none_while_unloaded() {
        let loader = cycling(usize::MAX);
        let log = loader.log();
        let w = Worker::spawn_with(loader, spec_params(CTX));
        // Nothing loaded yet: no state, so no totals.
        assert_eq!(sync(&w).spec, None);

        let first = run(&w, cyclic(12));
        assert_eq!(finish(&first), Finish::Length);
        let first_delta = done_spec(&first).expect("the request speculated");
        assert!(first_delta.steps > 0, "{first_delta:?}");
        assert!(
            log.live_calls()
                .iter()
                .any(|c| matches!(c, Call::Verify(_))),
            "no verify ran"
        );
        let totals = sync(&w).spec.expect("loaded with speculation on");
        assert_eq!(totals.delta, first_delta);

        // A second request adds to the engine-wide totals; its own `Done` has its own counts.
        let second = run(&w, cyclic(12));
        let second_delta = done_spec(&second).expect("the request speculated");
        let totals = sync(&w).spec.expect("loaded");
        assert_eq!(totals.delta.steps, first_delta.steps + second_delta.steps);
        assert_eq!(
            totals.delta.drafted,
            first_delta.drafted + second_delta.drafted
        );

        // An unload drops them; the next load starts again from zero.
        unload(&w);
        assert_eq!(sync(&w).spec, None);
        let third = run(&w, cyclic(12));
        let third_delta = done_spec(&third).expect("the request speculated");
        assert_eq!(sync(&w).spec.expect("reloaded").delta, third_delta);
        w.shutdown().unwrap();
    }

    #[test]
    fn mtp_drafts_reach_the_worker_and_its_totals_count_them_per_source() {
        // Prompt lookup has nothing to find in a prompt that never repeats, so MTP drafts every step.
        let script = |h: &[u32]| (h.len() as u32 + 1) % 50;
        let tokens = |events: &[Event]| -> Vec<u32> {
            events
                .iter()
                .filter_map(|e| match e {
                    Event::Token { id, .. } => Some(*id),
                    _ => None,
                })
                .collect()
        };
        let plain_loader = FakeLoader::new(CTX, script);
        let plain = Worker::spawn_with(plain_loader, params(CTX));
        let request = Request {
            prompt: (0..14).map(|i| (i * 3 + 100) % 50).collect(),
            max_tokens: 12,
            ..req(0)
        };
        let want = tokens(&run(&plain, request.clone()));
        plain.shutdown().unwrap();

        let loader = FakeLoader::new(CTX, script).tweak(|_, live| {
            live.set_mtp(crate::fake::FakeMtp::Right);
        });
        let log = loader.log();
        let w = Worker::spawn_with(loader, spec_params(CTX));
        let events = run(&w, request);
        // Output does not depend on MTP.
        assert_eq!(tokens(&events), want);
        let delta = done_spec(&events).expect("the request speculated");
        assert!(delta.drafted_mtp > 0 && delta.accepted_mtp > 0, "{delta:?}");
        assert_eq!(delta.drafted_lookup, 0, "{delta:?}");
        assert!(log
            .live_calls()
            .iter()
            .any(|c| matches!(c, Call::MtpDraft(..))));
        let totals = sync(&w).spec.expect("loaded with speculation on");
        assert_eq!(totals.delta, delta);
        assert_eq!(totals.delta.drafted, totals.delta.drafted_mtp);
        w.shutdown().unwrap();
    }

    #[test]
    fn speculation_off_gives_no_totals_and_no_counts() {
        let loader = cycling(usize::MAX);
        let log = loader.log();
        let w = Worker::spawn_with(loader, params(CTX));
        let events = run(&w, cyclic(12));
        assert_eq!(finish(&events), Finish::Length);
        assert_eq!(done_spec(&events), None);
        assert_eq!(sync(&w).spec, None);
        assert!(log
            .live_calls()
            .iter()
            .all(|c| !matches!(c, Call::Verify(_) | Call::Commit(_))));
        w.shutdown().unwrap();
    }

    #[test]
    fn the_workers_speculation_has_one_source_the_engine_config() {
        let mut cfg = Config::new(Default::default());
        assert!(cfg.speculation.enabled);
        assert_eq!(speculation_of(&cfg), Some(cfg.speculation.clone()));
        cfg.speculation.enabled = false;
        assert_eq!(speculation_of(&cfg), None);
        cfg.speculation = SpecConfig {
            enabled: true,
            max_draft: 7,
            ..SpecConfig::default()
        };
        assert_eq!(speculation_of(&cfg).map(|s| s.max_draft), Some(7));
    }

    #[test]
    fn the_reply_end_snapshot_after_a_partial_accept_holds_the_committed_tokens() {
        // The prompt is 14 tokens; the model gives 12, 13, 14, then stops: the draft [13, 14, 15, 10] is cut after the
        // third row, so the step commits three of its five tokens.
        let loader = cycling(17);
        let log = loader.log();
        let w = Worker::spawn_with(loader, spec_params(CTX));
        let events = run(&w, cyclic(12));
        assert_eq!(finish(&events), Finish::Stop(63));
        // The worker sends `Done` first and takes the reply-end snapshot after it; `sync` returns once the worker has
        // finished that, so the call log below is complete (without it this read raced the snapshot).
        sync(&w);
        let calls = log.live_calls();
        assert!(
            calls.contains(&Call::Verify(vec![12, 13, 14, 15, 10])),
            "{calls:?}"
        );
        assert!(calls.contains(&Call::Commit(3)), "{calls:?}");
        // The reply-end snapshot is at the committed 14 + 3 tokens, and it was taken (not skipped as a stale KV record).
        assert!(
            calls.contains(&Call::Snapshot(17, Marks::REPLY_END)),
            "{calls:?}"
        );
        unload(&w);
        let nodes = log.parks().pop().expect("the unload parked");
        let want = [cyclic(1).prompt, vec![12, 13, 14]].concat();
        assert!(
            nodes.iter().any(|(t, m)| *t == want && m.reply_end),
            "{nodes:?}"
        );
        w.shutdown().unwrap();
    }

    #[test]
    fn a_failed_commit_ends_the_request_and_the_next_one_starts_fresh() {
        let loader = cycling(usize::MAX).tweak(|i, live| {
            if i == 0 {
                live.fail_commit = Some(non_finite());
            }
        });
        let log = loader.log();
        let w = Worker::spawn_with(loader, spec_params(CTX));
        let events = run(&w, cyclic(12));
        assert_eq!(error(&events).0, ErrorKind::NonFinite);
        sync(&w);
        // The state is unusable: the worker took no reply-end snapshot and nothing panicked.
        assert!(
            !log.live_calls()
                .iter()
                .any(|c| matches!(c, Call::Snapshot(..))),
            "a failed step took a snapshot"
        );
        assert_eq!(sync(&w).after_done_errors, 0);

        // The next request starts from a reset, not from the half-committed state.
        let events = run(&w, cyclic(12));
        assert_eq!(finish(&events), Finish::Length);
        assert_eq!(started(&events), (0, Source::None));
        w.shutdown().unwrap();
    }

    #[test]
    fn a_failed_verify_ends_the_request_and_the_next_one_starts_fresh() {
        let loader = cycling(usize::MAX).tweak(|i, live| {
            if i == 0 {
                live.fail_verify = Some(non_finite());
            }
        });
        let w = Worker::spawn_with(loader, spec_params(CTX));
        let events = run(&w, cyclic(12));
        assert_eq!(error(&events).0, ErrorKind::NonFinite);
        let events = run(&w, cyclic(12));
        assert_eq!(finish(&events), Finish::Length);
        assert_eq!(started(&events), (0, Source::None));
        w.shutdown().unwrap();
    }

    #[test]
    fn a_commit_failing_after_the_burst_ended_gives_done_and_the_next_request_starts_fresh() {
        // The model stops inside the first burst (the stop token's row is the last read), and then the replay faults.
        let loader = cycling(17).tweak(|i, live| {
            if i == 0 {
                live.fail_commit = Some(non_finite());
            }
        });
        let log = loader.log();
        let w = Worker::spawn_with(loader, spec_params(CTX));
        let events = run(&w, cyclic(12));
        // The client's answer is whole: tokens, then `Done`, no `Error`.
        assert_eq!(finish(&events), Finish::Stop(63));
        assert!(!events.iter().any(|e| matches!(e, Event::Error { .. })));
        let stats = sync(&w);
        // Counted with the faults no client sees, the state marked unusable, no snapshot of it.
        assert_eq!(stats.after_done_errors, 1);
        assert_eq!(log.unusable_marks(), 1);
        assert!(
            !log.live_calls()
                .iter()
                .any(|c| matches!(c, Call::Snapshot(..))),
            "a lost state took a snapshot"
        );
        assert_eq!(stats.reuse.prefilled_tokens, 14);
        let events = run(&w, cyclic(12));
        assert_eq!(started(&events), (0, Source::None));
        w.shutdown().unwrap();
    }

    #[test]
    fn a_failed_row_ends_the_request_and_the_next_one_starts_fresh() {
        let loader = cycling(usize::MAX).tweak(|i, live| {
            if i == 0 {
                live.fail_row = Some((1, non_finite()));
            }
        });
        let log = loader.log();
        let w = Worker::spawn_with(loader, spec_params(CTX));
        let events = run(&w, cyclic(12));
        assert_eq!(error(&events).0, ErrorKind::NonFinite);
        sync(&w);
        assert_eq!(log.unusable_marks(), 1);
        assert!(
            !log.live_calls()
                .iter()
                .any(|c| matches!(c, Call::Snapshot(..) | Call::Commit(_))),
            "a failed step was committed or snapshotted"
        );
        let events = run(&w, cyclic(12));
        assert_eq!(finish(&events), Finish::Length);
        assert_eq!(started(&events), (0, Source::None));
        w.shutdown().unwrap();
    }
}
