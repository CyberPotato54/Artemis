//! The scripted fake engine: plays the engine worker on its own thread, so the HTTP side meets the
//! same concurrency. One `std::thread` takes commands from a `std::sync::mpsc` FIFO one at a time; the lifecycle state
//! is an atomic the server reads without the worker. The `tools` crate adapts the real worker to the same `Backend`
//! seam.
use crate::backend::*;
use crate::logs::{LifecycleLine, Logger};
use crate::tokens::Specials;
use qwen36::tokenizer::Tokenizer;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FakeConfig {
    /// A load (lazy or asked for) takes this long, state `Loading`.
    pub load_time: Duration,
    pub unload_time: Duration,
    /// How long a `DeviceLost` shows `Recovering`.
    pub recover_time: Duration,
    /// When `Ready`, unload after this long without a command (the worker's `recv_timeout`).
    pub idle_unload: Option<Duration>,
    pub start_loaded: bool,
}

/// How a script's request fails at `fail_at`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fail {
    /// `Error { DeviceLost }`, then `Recovering` for `recover_time`, then `Ready`.
    DeviceLost,
    NonFinite,
    OutOfMemory,
    Internal,
    /// The sink is dropped with no terminal event and the worker stops: `submit` then answers `Err(Full)`.
    Vanish,
}

/// One scripted reply. The scripts are used in request order; a request that finds none gets `"ok"`. A request that is
/// skipped (closed sink) or ended `Unloaded` before it starts, one that raced its own lazy load included, takes none. A
/// request whose load fails takes the script that failed it; an explicit `load()` takes the front script only when it
/// has `load_fails`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Script {
    /// The sampled tokens. When they run out the request ends with the model's `<|im_end|>` as its stop token.
    pub tokens: Vec<u32>,
    pub before_started: Duration,
    pub prefill: Duration,
    pub per_token: Duration,
    pub reused: u32,
    pub source: Source,
    pub diverged_at: u32,
    /// The restore's time, as `Done` reports it.
    pub restore_ms: f64,
    /// What `Done` reports of the cache; `None` is the fake's zeros and no rates.
    pub cache: Option<CacheReport>,
    /// What `Done` reports of speculation; `None` is speculation off.
    pub spec: Option<SpecReport>,
    /// Fail when about to send sampled token number `.0` (zero-based; forced tokens don't count).
    pub fail_at: Option<(usize, Fail)>,
    /// Fail the lazy load this request triggers (or the explicit load that meets this script) with this message.
    pub load_fails: Option<String>,
}

impl Script {
    /// `s` as the model's tokens (special tokens parsed), zero delays, no restore.
    pub fn text(tk: &Tokenizer, s: &str) -> Script {
        Script {
            tokens: tk.encode(s, true),
            ..Script::default()
        }
    }
}

/// One request that ran (it got `Started`). `token_times[i]` is the `Instant` token `i` was sent (forced tokens
/// included; the stop token, never sent, excluded). Tests assert cancellation in tokens: `run.tokens_after(t) <= 1`.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    pub started: Instant,
    pub request: GenRequest,
    pub token_times: Vec<Instant>,
    /// `None` for a request that ended in an `Error` or vanished.
    pub finish: Option<Finish>,
}

impl Run {
    /// The tokens sent after `t`.
    pub fn tokens_after(&self, t: Instant) -> usize {
        self.token_times.iter().filter(|&&s| s > t).count()
    }
}

enum Command {
    Run(GenRequest, Sink, Instant, Held),
    Load(oneshot::Sender<Result<(), BackendError>>, Held),
    Unload(oneshot::Sender<Result<bool, BackendError>>, Held),
    Stats(oneshot::Sender<Result<BackendStats, BackendError>>, Held),
}

/// One piece of pending work (`Progress::pending`): up, with no tick, when it is given; down, with a tick, when it is dropped. A
/// command holds it until its reply, a request until its sink is dropped, so a command that is skipped or dies with the
/// worker counts down too.
struct Held(Arc<Shared>);

impl Held {
    fn new(shared: &Arc<Shared>) -> Held {
        shared.pending.fetch_add(1, Ordering::AcqRel);
        // No tick: arrival says nothing of the worker (see the adapter's `PendingGuard`).
        Held(shared.clone())
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        self.0.pending.fetch_sub(1, Ordering::AcqRel);
        self.0.tick();
    }
}

#[derive(Default)]
struct Records {
    runs: Vec<Run>,
    cancellations: Vec<(Instant, usize)>,
    loads: Vec<Instant>,
    unloads: Vec<Instant>,
}

struct Shared {
    state: AtomicU8,
    /// Unloads asked for and not yet done: up from `unload()` at once, down when its `Unload` command completes.
    pending_unloads: AtomicUsize,
    /// Idle unloads the worker has begun (up before its park, so a test can wait for one to be under way).
    idle_unloads_begun: AtomicUsize,
    /// A `Vanish` stopped the worker.
    gone: AtomicBool,
    scripts: Mutex<VecDeque<Script>>,
    records: Mutex<Records>,
    stats: Mutex<BackendStats>,
    /// The last `Stats` answer.
    latest: Mutex<Option<StatsCopy>>,
    /// `Progress`: a tick for each token sent, each phase start and each command answered.
    ticks: AtomicU64,
    pending: AtomicU32,
    /// Where the loads and idle unloads are written, as the real worker's adapter writes them.
    logger: Mutex<Option<Logger>>,
}

impl Shared {
    fn tick(&self) {
        self.ticks.fetch_add(1, Ordering::AcqRel);
    }

    fn state(&self) -> Lifecycle {
        match self.state.load(Ordering::Acquire) {
            0 => Lifecycle::Idle,
            1 => Lifecycle::Loading,
            2 => Lifecycle::Ready,
            _ => Lifecycle::Recovering,
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn code(l: Lifecycle) -> u8 {
    match l {
        Lifecycle::Idle => 0,
        Lifecycle::Loading => 1,
        Lifecycle::Ready => 2,
        Lifecycle::Recovering => 3,
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

pub struct FakeBackend {
    shared: Arc<Shared>,
    tx: mpsc::Sender<Command>,
}

impl FakeBackend {
    pub fn new(cfg: FakeConfig, tk: Arc<Tokenizer>, sp: Specials) -> Arc<FakeBackend> {
        let start = if cfg.start_loaded {
            Lifecycle::Ready
        } else {
            Lifecycle::Idle
        };
        let shared = Arc::new(Shared {
            state: AtomicU8::new(code(start)),
            pending_unloads: AtomicUsize::new(0),
            idle_unloads_begun: AtomicUsize::new(0),
            gone: AtomicBool::new(false),
            scripts: Mutex::new(VecDeque::new()),
            records: Mutex::new(Records::default()),
            stats: Mutex::new(BackendStats::default()),
            latest: Mutex::new(None),
            ticks: AtomicU64::new(0),
            pending: AtomicU32::new(0),
            logger: Mutex::new(None),
        });
        let (tx, rx) = mpsc::channel();
        let worker = Worker {
            shared: shared.clone(),
            cfg,
            tk,
            sp,
            rx,
            backlog: VecDeque::new(),
            last_active: Instant::now(),
        };
        std::thread::Builder::new()
            .name("fake-engine".into())
            .spawn(move || worker.run())
            .expect("spawning the fake engine's thread");
        Arc::new(FakeBackend { shared, tx })
    }

    /// Queues a script for a later request; they are used in request order.
    pub fn push(&self, s: Script) {
        lock(&self.shared.scripts).push_back(s);
    }

    /// The requests that ran, with when each started (a skipped one is not listed).
    pub fn requests(&self) -> Vec<(Instant, GenRequest)> {
        lock(&self.shared.records)
            .runs
            .iter()
            .map(|r| (r.started, r.request.clone()))
            .collect()
    }

    /// The same, with each token's send time.
    pub fn runs(&self) -> Vec<Run> {
        lock(&self.shared.records).runs.clone()
    }

    /// `(when the request saw closed(), tokens sent before it)`.
    pub fn cancellations(&self) -> Vec<(Instant, usize)> {
        lock(&self.shared.records).cancellations.clone()
    }

    pub fn loads(&self) -> Vec<Instant> {
        lock(&self.shared.records).loads.clone()
    }

    pub fn unloads(&self) -> Vec<Instant> {
        lock(&self.shared.records).unloads.clone()
    }

    /// Unloads asked for and not yet done (up at the call to `unload`, down when the worker has done it).
    pub fn pending_unloads(&self) -> usize {
        self.shared.pending_unloads.load(Ordering::Acquire)
    }

    /// Idle unloads begun: up when the worker starts one, before `unload_time` has passed and the state has left
    /// `Ready`.
    pub fn idle_unloads_begun(&self) -> usize {
        self.shared.idle_unloads_begun.load(Ordering::Acquire)
    }

    pub fn set_stats(&self, s: BackendStats) {
        *lock(&self.shared.stats) = s;
    }

    /// Writes the loads and the idle unloads to `l`. The server writes the other lifecycle lines (an unload it asked
    /// for, a device loss, the exit).
    pub fn set_logger(&self, l: Logger) {
        *lock(&self.shared.logger) = Some(l);
    }

    /// Sends a command whose reply the caller holds; a dead worker drops it, which the receiver sees as closed.
    fn command<T>(
        &self,
        make: impl FnOnce(oneshot::Sender<Result<T, BackendError>>, Held) -> Command,
    ) -> (Reply<T>, bool) {
        let (tx, rx) = oneshot::channel();
        (rx, self.tx.send(make(tx, Held::new(&self.shared))).is_ok())
    }
}

impl Backend for FakeBackend {
    fn submit(&self, req: GenRequest, sink: Sink) -> Result<(), Full> {
        if self.shared.gone.load(Ordering::Acquire) {
            return Err(Full);
        }
        self.tx
            .send(Command::Run(
                req,
                sink,
                Instant::now(),
                Held::new(&self.shared),
            ))
            .map_err(|_| Full)
    }

    fn state(&self) -> Lifecycle {
        self.shared.state()
    }

    fn load(&self) -> Reply<()> {
        self.command(Command::Load).0
    }

    fn unload(&self) -> Reply<bool> {
        // Up at once, so the request in flight sees it; the worker brings it down when this unload is done.
        self.shared.pending_unloads.fetch_add(1, Ordering::AcqRel);
        let (reply, sent) = self.command(Command::Unload);
        if !sent {
            self.shared.pending_unloads.fetch_sub(1, Ordering::AcqRel);
        }
        reply
    }

    fn stats(&self) -> Reply<BackendStats> {
        self.command(Command::Stats).0
    }

    fn progress(&self) -> Option<Progress> {
        Some(Progress {
            ticks: self.shared.ticks.load(Ordering::Acquire),
            pending: self.shared.pending.load(Ordering::Acquire),
        })
    }

    fn latest_stats(&self) -> Option<StatsCopy> {
        lock(&self.shared.latest).clone()
    }
}

/// How a request's decode loop ended.
enum End {
    Done(Finish),
    Fail(Fail),
}

struct Worker {
    shared: Arc<Shared>,
    cfg: FakeConfig,
    tk: Arc<Tokenizer>,
    sp: Specials,
    rx: mpsc::Receiver<Command>,
    /// Commands a failed load set aside, in order: the unloads and stats that proceed.
    backlog: VecDeque<Command>,
    /// When the last request or load finished: the idle timer counts from here, not from any command.
    last_active: Instant,
}

impl Worker {
    fn run(mut self) {
        loop {
            let command = match self.backlog.pop_front() {
                Some(c) => c,
                None => match self.next_command() {
                    Some(c) => c,
                    None => return,
                },
            };
            if !self.handle(command) {
                return;
            }
        }
    }

    /// The next command, or None once every sender is gone. When `Ready` with an idle timeout, waiting until `idle`
    /// after the last request or load with nothing queued unloads (the real worker's `recv_timeout`).
    fn next_command(&mut self) -> Option<Command> {
        loop {
            let Some(idle) = self
                .cfg
                .idle_unload
                .filter(|_| self.state() == Lifecycle::Ready)
            else {
                return self.rx.recv().ok();
            };
            let left = idle.saturating_sub(self.last_active.elapsed());
            match self.rx.recv_timeout(left) {
                Ok(c) => return Some(c),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    self.shared
                        .idle_unloads_begun
                        .fetch_add(1, Ordering::AcqRel);
                    self.finish_unload();
                    // The real adapter follows its idle `unload` line with a `stats` line from its own copy; the fake
                    // writes none, and the server's log tests do not expect one.
                    self.log(&LifecycleLine::Unload {
                        cause: "idle".into(),
                    });
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => return None,
            }
        }
    }

    fn state(&self) -> Lifecycle {
        self.shared.state()
    }

    fn set_state(&self, l: Lifecycle) {
        // The tick first, so a reader that sees the new state also sees the tick.
        self.shared.tick();
        self.shared.state.store(code(l), Ordering::Release);
    }

    fn unload_pending(&self) -> bool {
        self.shared.pending_unloads.load(Ordering::Acquire) != 0
    }

    /// False when the worker should stop (a `Vanish`).
    fn handle(&mut self, command: Command) -> bool {
        self.shared.tick();
        match command {
            Command::Run(req, sink, queued_at, held) => {
                let go = self.serve(req, sink, queued_at);
                drop(held);
                return go;
            }
            Command::Load(reply, held) => {
                let result = if self.state() == Lifecycle::Idle {
                    let fails = self.take_load_failure();
                    self.load_now(fails.as_deref())
                } else {
                    Ok(())
                };
                drop(held);
                let _ = reply.send(result);
            }
            Command::Unload(reply, held) => {
                let was_loaded = self.state() != Lifecycle::Idle;
                if was_loaded {
                    self.finish_unload();
                }
                self.shared.pending_unloads.fetch_sub(1, Ordering::AcqRel);
                drop(held);
                let _ = reply.send(Ok(was_loaded));
            }
            Command::Stats(reply, held) => {
                let stats = lock(&self.shared.stats).clone();
                *lock(&self.shared.latest) = Some(StatsCopy {
                    stats: stats.clone(),
                    at: Instant::now(),
                });
                drop(held);
                let _ = reply.send(Ok(stats));
            }
        }
        true
    }

    fn log(&self, l: &LifecycleLine) {
        if let Some(logger) = lock(&self.shared.logger).as_ref() {
            logger.lifecycle(l);
        }
    }

    /// The unload itself: `unload_time`, then `Idle`, recorded.
    fn finish_unload(&self) {
        std::thread::sleep(self.cfg.unload_time);
        self.set_state(Lifecycle::Idle);
        lock(&self.shared.records).unloads.push(Instant::now());
    }

    /// Takes the front script if it fails the load, for a load nobody's request triggered.
    fn take_load_failure(&self) -> Option<String> {
        let mut scripts = lock(&self.shared.scripts);
        if scripts.front()?.load_fails.is_some() {
            scripts.pop_front()?.load_fails
        } else {
            None
        }
    }

    /// Loads an idle model: `Loading` for `load_time`, then `Ready` (recorded), or back to `Idle` after failing the
    /// queue. The failing load answers its own request or command only after that, as the engine's worker does.
    fn load_now(&mut self, fails: Option<&str>) -> Result<(), BackendError> {
        self.set_state(Lifecycle::Loading);
        let began = Instant::now();
        std::thread::sleep(self.cfg.load_time);
        match fails {
            None => {
                self.set_state(Lifecycle::Ready);
                self.last_active = Instant::now();
                lock(&self.shared.records).loads.push(Instant::now());
                self.log(&LifecycleLine::Load {
                    seconds: began.elapsed().as_secs_f64(),
                    detail: "fake engine".into(),
                    slots_per_layer: None,
                    vram_used_mib: None,
                    pool_limit_gib: None,
                    pool_cut: None,
                    unparked: None,
                });
                Ok(())
            }
            Some(message) => {
                self.set_state(Lifecycle::Idle);
                self.fail_queue(message);
                Err(BackendError {
                    kind: ErrorKind::LoadFailed,
                    message: message.to_string(),
                })
            }
        }
    }

    /// Every command already queued fails with a load: each `Run` gets the error, each `Load` an `Err`. The other
    /// commands keep their order and run next.
    fn fail_queue(&mut self, message: &str) {
        while let Ok(command) = self.rx.try_recv() {
            match command {
                Command::Run(_, sink, _, _) => {
                    sink.send(GenEvent::Error {
                        kind: ErrorKind::LoadFailed,
                        message: message.to_string(),
                    });
                }
                Command::Load(reply, held) => {
                    drop(held);
                    let _ = reply.send(Err(BackendError {
                        kind: ErrorKind::LoadFailed,
                        message: message.to_string(),
                    }));
                }
                other => self.backlog.push_back(other),
            }
        }
    }

    fn next_script(&self) -> Script {
        lock(&self.shared.scripts)
            .pop_front()
            .unwrap_or_else(|| Script::text(&self.tk, "ok"))
    }

    fn end_unloaded(&self, req: &GenRequest, sink: &Sink, queue_ms: f64, load_ms: f64) {
        sink.send(GenEvent::Done {
            finish: Finish::Unloaded,
            counts: Counts {
                prompt: req.prompt.len() as u32,
                reused: 0,
                prefilled: 0,
                completion: 0,
                reasoning: 0,
            },
            timings: Timings {
                queue_ms,
                load_ms,
                restore_ms: 0.0,
                prefill_ms: 0.0,
                decode_ms: 0.0,
                ttft_ms: 0.0,
            },
            cache: cache_report(),
            spec: None,
        });
    }

    /// A request whose turn it is. False when the worker should stop.
    fn serve(&mut self, req: GenRequest, sink: Sink, queued_at: Instant) -> bool {
        let dequeued = Instant::now();
        let queue_ms = ms(dequeued.duration_since(queued_at));
        if sink.closed() {
            return true;
        }
        if self.unload_pending() {
            self.end_unloaded(&req, &sink, queue_ms, 0.0);
            return true;
        }
        let mut load_ms = 0.0;
        if self.state() == Lifecycle::Idle {
            // The script is only looked at until the request is past the load and its checks.
            let fails = lock(&self.shared.scripts)
                .front()
                .and_then(|s| s.load_fails.clone());
            let t = Instant::now();
            if let Err(e) = self.load_now(fails.as_deref()) {
                if fails.is_some() {
                    self.next_script();
                }
                // The queue was answered first (in `load_now`): a client that retries on seeing this error finds it
                // empty.
                sink.send(GenEvent::Error {
                    kind: e.kind,
                    message: e.message,
                });
                return true;
            }
            load_ms = ms(t.elapsed());
            // An unload or a close that landed during the load is seen before `Started`.
            if sink.closed() {
                return true;
            }
            if self.unload_pending() {
                self.end_unloaded(&req, &sink, queue_ms, load_ms);
                return true;
            }
        }
        let script = self.next_script();
        let go = self.generate(req, sink, script, (queue_ms, load_ms), dequeued);
        self.last_active = Instant::now();
        go
    }

    fn generate(
        &mut self,
        req: GenRequest,
        sink: Sink,
        script: Script,
        (queue_ms, load_ms): (f64, f64),
        dequeued: Instant,
    ) -> bool {
        let sh = self.shared.clone();
        std::thread::sleep(script.before_started);
        let prompt = req.prompt.len() as u32;
        let run = {
            let mut records = lock(&sh.records);
            records.runs.push(Run {
                started: Instant::now(),
                request: req.clone(),
                token_times: vec![],
                finish: None,
            });
            records.runs.len() - 1
        };
        sink.send(GenEvent::Started {
            prompt_tokens: prompt,
            reused_tokens: script.reused,
            restored: script.source,
            diverged_at: script.diverged_at,
        });
        sh.tick();
        let t = Instant::now();
        std::thread::sleep(script.prefill);
        let prefill_ms = ms(t.elapsed());
        let decode_start = Instant::now();
        sh.tick();

        let mut open = req.think.as_ref().is_some_and(|t| t.open);
        let mut forcing: VecDeque<u32> = VecDeque::new();
        let mut sampled = 0usize;
        let mut completion = 0u32;
        let mut sent = 0usize;
        let mut ttft_ms = 0.0;
        let end = loop {
            if completion >= req.max_tokens {
                break End::Done(Finish::Length);
            }
            // The next token: a forced close token, the budget's first forced one, or the script's (its end is the
            // model's own stop).
            let budget_hit = open
                && req.think.as_ref().is_some_and(|t| {
                    t.budget.is_some_and(|b| sampled as u32 >= b) && !t.close.is_empty()
                });
            let (id, forced, exhausted) = if let Some(id) = forcing.pop_front() {
                (id, true, false)
            } else if budget_hit {
                open = false;
                forcing = req
                    .think
                    .iter()
                    .flat_map(|t| t.close.iter().copied())
                    .collect();
                (
                    forcing.pop_front().expect("close is not empty"),
                    true,
                    false,
                )
            } else if let Some(&id) = script.tokens.get(sampled) {
                (id, false, false)
            } else {
                (self.sp.im_end, false, true)
            };
            std::thread::sleep(script.per_token);
            if sink.closed() {
                lock(&sh.records).cancellations.push((Instant::now(), sent));
                break End::Done(Finish::Cancelled);
            }
            if self.unload_pending() {
                break End::Done(Finish::Unloaded);
            }
            if !forced {
                if let Some((_, fail)) = script.fail_at.filter(|&(at, _)| at == sampled) {
                    break End::Fail(fail);
                }
                if exhausted || req.stop_tokens.contains(&id) {
                    completion += 1;
                    break End::Done(Finish::Stop(id));
                }
            }
            let logprobs = req.logprobs.filter(|_| !forced).map(|n| TokenLogprobs {
                logprob: -0.25,
                top: (0..n)
                    .map(|k| (id + u32::from(k), -0.25 - f32::from(k)))
                    .collect(),
            });
            if !sink.send(GenEvent::Token {
                id,
                forced,
                logprobs,
            }) {
                lock(&sh.records).cancellations.push((Instant::now(), sent));
                break End::Done(Finish::Cancelled);
            }
            sh.tick();
            let now = Instant::now();
            lock(&sh.records).runs[run].token_times.push(now);
            if sent == 0 {
                ttft_ms = ms(now.duration_since(dequeued));
            }
            sent += 1;
            completion += 1;
            if !forced {
                sampled += 1;
                if open && req.think.as_ref().is_some_and(|t| t.ends.contains(&id)) {
                    open = false;
                }
            }
        };

        match end {
            End::Done(finish) => {
                lock(&sh.records).runs[run].finish = Some(finish);
                sink.send(GenEvent::Done {
                    finish,
                    counts: Counts {
                        prompt,
                        reused: script.reused,
                        prefilled: prompt.saturating_sub(script.reused),
                        completion,
                        reasoning: 0,
                    },
                    timings: Timings {
                        queue_ms,
                        load_ms,
                        restore_ms: script.restore_ms,
                        prefill_ms,
                        decode_ms: ms(decode_start.elapsed()),
                        ttft_ms,
                    },
                    cache: script.cache.clone().unwrap_or_else(cache_report),
                    spec: script.spec,
                });
            }
            End::Fail(Fail::Vanish) => {
                // The sink drops on return with no terminal event; `submit` says `Full` from now on.
                sh.gone.store(true, Ordering::Release);
                return false;
            }
            End::Fail(fail) => {
                let kind = match fail {
                    Fail::DeviceLost => ErrorKind::DeviceLost,
                    Fail::NonFinite => ErrorKind::NonFinite,
                    Fail::OutOfMemory => ErrorKind::OutOfMemory,
                    Fail::Internal | Fail::Vanish => ErrorKind::Internal,
                };
                sink.send(GenEvent::Error {
                    kind,
                    message: format!("scripted failure: {fail:?}"),
                });
                if fail == Fail::DeviceLost {
                    self.set_state(Lifecycle::Recovering);
                    std::thread::sleep(self.cfg.recover_time);
                    self.set_state(Lifecycle::Ready);
                }
            }
        }
        true
    }
}

/// The fake has no cache: zeros, and no figure for the two rates.
fn cache_report() -> CacheReport {
    CacheReport {
        hit_rate: None,
        miss_mib_per_token: None,
        restore_bytes: 0,
        pool_bytes_used: 0,
        decode_misses: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen36::tokenizer::{Tokenizer, Vocab};
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};
    use tokio::sync::mpsc::error::TryRecvError;
    use tokio::sync::mpsc::UnboundedReceiver;

    const MS: Duration = Duration::from_millis(1);

    fn fixtures() -> &'static (Arc<Tokenizer>, Specials) {
        static T: OnceLock<(Arc<Tokenizer>, Specials)> = OnceLock::new();
        T.get_or_init(|| {
            let b = std::fs::read(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/tokenizer/qwen36-vocab.bin"
            ))
            .unwrap();
            let v = Vocab::from_extract(&b).unwrap();
            let tk = Tokenizer::new(&v).unwrap();
            let sp = Specials::new(&v, &tk).unwrap();
            (Arc::new(tk), sp)
        })
    }

    fn loaded() -> FakeConfig {
        FakeConfig {
            start_loaded: true,
            ..FakeConfig::default()
        }
    }

    fn backend(cfg: FakeConfig) -> Arc<FakeBackend> {
        let (tk, sp) = fixtures();
        FakeBackend::new(cfg, tk.clone(), sp.clone())
    }

    fn script(tokens: &[u32], per_token: Duration) -> Script {
        Script {
            tokens: tokens.to_vec(),
            per_token,
            ..Script::default()
        }
    }

    /// A request whose prompt is `n` copies of `n`, so the records tell requests apart.
    fn req(n: usize, max_tokens: u32) -> GenRequest {
        GenRequest {
            prompt: vec![n as u32; n],
            cut: None,
            sampling: Sampling {
                temp: 0.0,
                top_k: 1,
                top_p: 1.0,
                min_p: 0.0,
                presence_penalty: 0.0,
            },
            seed: 1,
            max_tokens,
            stop_tokens: vec![fixtures().1.im_end],
            think: None,
            logprobs: None,
        }
    }

    fn submit(b: &FakeBackend, r: GenRequest) -> (UnboundedReceiver<GenEvent>, CancelHandle) {
        let (sink, rx, cancel) = Sink::channel();
        assert!(b.submit(r, sink).is_ok());
        (rx, cancel)
    }

    /// Polls `f` until it answers, for at most 30 s (these are plain tests: a thread sleep is fine).
    fn wait<T>(mut f: impl FnMut() -> Option<T>) -> T {
        let end = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(v) = f() {
                return v;
            }
            assert!(Instant::now() < end, "timed out");
            std::thread::sleep(2 * MS);
        }
    }

    /// Every event up to and including the terminal one, or up to the sink being dropped.
    fn events(rx: &mut UnboundedReceiver<GenEvent>) -> Vec<GenEvent> {
        let mut out = vec![];
        loop {
            let ev = wait(|| match rx.try_recv() {
                Ok(ev) => Some(Some(ev)),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => Some(None),
            });
            let Some(ev) = ev else { return out };
            let end = matches!(ev, GenEvent::Done { .. } | GenEvent::Error { .. });
            out.push(ev);
            if end {
                return out;
            }
        }
    }

    fn next(rx: &mut UnboundedReceiver<GenEvent>) -> GenEvent {
        wait(|| rx.try_recv().ok())
    }

    fn ids(evs: &[GenEvent]) -> Vec<u32> {
        evs.iter()
            .filter_map(|e| match e {
                GenEvent::Token { id, .. } => Some(*id),
                _ => None,
            })
            .collect()
    }

    fn finish(evs: &[GenEvent]) -> Finish {
        match evs.last() {
            Some(GenEvent::Done { finish, .. }) => *finish,
            other => panic!("no Done last: {other:?}"),
        }
    }

    fn started(evs: &[GenEvent]) -> bool {
        evs.iter().any(|e| matches!(e, GenEvent::Started { .. }))
    }

    fn done_of(evs: &[GenEvent]) -> (Counts, Timings, CacheReport) {
        match evs.last() {
            Some(GenEvent::Done {
                counts,
                timings,
                cache,
                ..
            }) => (counts.clone(), timings.clone(), cache.clone()),
            other => panic!("no Done last: {other:?}"),
        }
    }

    fn reply<T>(mut r: Reply<T>) -> Result<T, BackendError> {
        wait(|| match r.try_recv() {
            Ok(v) => Some(v),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => None,
            Err(e) => panic!("the reply was dropped: {e:?}"),
        })
    }

    #[test]
    fn requests_run_in_order_one_at_a_time() {
        let b = backend(loaded());
        for _ in 0..3 {
            b.push(script(&[100, 101], 20 * MS));
        }
        let mut rxs: Vec<_> = (1..=3).map(|n| submit(&b, req(n, 10)).0).collect();
        for rx in &mut rxs {
            let evs = events(rx);
            assert_eq!(ids(&evs), [100, 101]);
            assert!(matches!(finish(&evs), Finish::Stop(_)));
        }
        let runs = b.runs();
        let lens: Vec<_> = runs.iter().map(|r| r.request.prompt.len()).collect();
        assert_eq!(lens, [1, 2, 3]);
        let reqs = b.requests();
        assert_eq!(reqs.len(), 3);
        assert_eq!(reqs[1].1.prompt.len(), 2);
        for pair in runs.windows(2) {
            let last = *pair[0].token_times.last().unwrap();
            assert!(last <= pair[1].started, "one at a time");
        }
    }

    #[test]
    fn a_closed_sink_is_skipped() {
        let b = backend(loaded());
        b.push(script(&[100, 101, 102], 50 * MS));
        let (mut rx1, _c1) = submit(&b, req(1, 10));
        // Cancelled before it is submitted, so no timing decides whether the worker saw it open.
        let (sink2, mut rx2, c2) = Sink::channel();
        c2.cancel();
        assert!(b.submit(req(2, 10), sink2).is_ok());
        let (mut rx3, _c3) = submit(&b, req(3, 10));
        events(&mut rx1);
        let evs3 = events(&mut rx3);
        assert!(started(&evs3));
        assert!(matches!(finish(&evs3), Finish::Stop(_)));
        // The skipped request got nothing at all (its sink was dropped unused), and is not listed.
        assert_eq!(rx2.try_recv(), Err(TryRecvError::Disconnected));
        let lens: Vec<_> = b.requests().iter().map(|r| r.1.prompt.len()).collect();
        assert_eq!(lens, [1, 3]);
    }

    #[test]
    fn cancel_ends_within_one_token() {
        let b = backend(loaded());
        b.push(script(&[100; 20], 100 * MS));
        let (mut rx, cancel) = submit(&b, req(5, 50));
        // Two tokens in, then cancel.
        let mut seen = vec![];
        while ids(&seen).len() < 2 {
            seen.push(next(&mut rx));
        }
        let t = Instant::now();
        cancel.cancel();
        let mut evs = seen;
        evs.extend(events(&mut rx));
        assert_eq!(finish(&evs), Finish::Cancelled);
        assert!(ids(&evs).len() < 20);
        let run = &b.runs()[0];
        assert!(run.tokens_after(t) <= 1, "{}", run.tokens_after(t));
        assert_eq!(run.finish, Some(Finish::Cancelled));
        let cancels = b.cancellations();
        assert_eq!(cancels.len(), 1);
        assert_eq!(cancels[0].1, run.token_times.len());
        assert!(cancels[0].0 >= t);
    }

    #[test]
    fn unload_ends_the_request_in_flight() {
        let b = backend(FakeConfig {
            unload_time: 50 * MS,
            ..loaded()
        });
        b.push(script(&[100; 40], 50 * MS));
        let (mut rx, _c) = submit(&b, req(5, 100));
        let mut evs = vec![];
        while ids(&evs).is_empty() {
            evs.push(next(&mut rx));
        }
        let u = b.unload();
        evs.extend(events(&mut rx));
        assert_eq!(finish(&evs), Finish::Unloaded);
        assert!(started(&evs));
        assert!(reply(u).is_ok());
        assert_eq!(b.state(), Lifecycle::Idle);
        assert_eq!(b.unloads().len(), 1);
        assert_eq!(b.runs()[0].finish, Some(Finish::Unloaded));
    }

    #[test]
    fn a_request_queued_before_an_unload_is_unloaded_not_run() {
        let b = backend(FakeConfig {
            unload_time: 1500 * MS,
            ..loaded()
        });
        b.push(script(&[100; 40], 50 * MS));
        let (mut rx1, _c1) = submit(&b, req(1, 100));
        let (mut rx2, _c2) = submit(&b, req(2, 100));
        let mut evs1 = vec![];
        while ids(&evs1).is_empty() {
            evs1.push(next(&mut rx1));
        }
        let mut u = b.unload();
        evs1.extend(events(&mut rx1));
        assert_eq!(finish(&evs1), Finish::Unloaded);
        assert!(started(&evs1));
        let evs2 = events(&mut rx2);
        assert_eq!(evs2.len(), 1, "{evs2:?}");
        assert_eq!(finish(&evs2), Finish::Unloaded);
        assert!(!started(&evs2));
        // The unload completes only after both ended.
        assert!(u.try_recv().is_err());
        assert!(reply(u).is_ok());
        assert_eq!(b.unloads().len(), 1);
        assert_eq!(b.requests().len(), 1);
    }

    #[test]
    fn a_request_submitted_after_an_unload_runs_after_it() {
        let b = backend(FakeConfig {
            unload_time: 50 * MS,
            ..loaded()
        });
        b.push(script(&[100; 40], 50 * MS));
        let (mut rx1, _c1) = submit(&b, req(1, 100));
        let mut evs1 = vec![];
        while ids(&evs1).is_empty() {
            evs1.push(next(&mut rx1));
        }
        let u = b.unload();
        let (mut rx2, _c2) = submit(&b, req(2, 100));
        evs1.extend(events(&mut rx1));
        assert_eq!(finish(&evs1), Finish::Unloaded);
        let evs2 = events(&mut rx2);
        assert!(started(&evs2), "{evs2:?}");
        assert!(matches!(finish(&evs2), Finish::Stop(_)));
        assert!(reply(u).is_ok());
        // It triggered a new lazy load, after the unload was recorded.
        let (loads, unloads) = (b.loads(), b.unloads());
        assert_eq!((loads.len(), unloads.len()), (1, 1));
        assert!(loads[0] >= unloads[0]);
        assert!(b.runs()[1].started >= loads[0]);
        assert_eq!(b.state(), Lifecycle::Ready);
    }

    #[test]
    fn lazy_load_shows_loading_then_ready() {
        let b = backend(FakeConfig {
            load_time: 1500 * MS,
            ..FakeConfig::default()
        });
        assert_eq!(b.state(), Lifecycle::Idle);
        let (mut rx, _c) = submit(&b, req(5, 10));
        wait(|| (b.state() == Lifecycle::Loading).then_some(()));
        assert!(rx.try_recv().is_err(), "no Started during the load");
        let evs = events(&mut rx);
        assert!(started(&evs));
        assert_eq!(b.state(), Lifecycle::Ready);
        assert_eq!(b.loads().len(), 1);
        assert!(b.runs()[0].started >= b.loads()[0]);
        // An explicit load when loaded is Ok at once, and does not load again.
        assert!(reply(b.load()).is_ok());
        assert_eq!(b.loads().len(), 1);
    }

    #[test]
    fn an_unload_racing_a_lazy_load_ends_the_request_unloaded() {
        let b = backend(FakeConfig {
            load_time: 1500 * MS,
            ..FakeConfig::default()
        });
        // A script for the request after: the one that raced the load takes none.
        b.push(script(&[100, 101], MS));
        let (mut rx, _c) = submit(&b, req(5, 10));
        wait(|| (b.state() == Lifecycle::Loading).then_some(()));
        let u = b.unload();
        let evs = events(&mut rx);
        assert_eq!(evs.len(), 1, "{evs:?}");
        assert_eq!(finish(&evs), Finish::Unloaded);
        assert!(reply(u).is_ok());
        assert_eq!(b.state(), Lifecycle::Idle);
        assert!(b.requests().is_empty());
        assert_eq!((b.loads().len(), b.unloads().len()), (1, 1));
        let (mut rx2, _c2) = submit(&b, req(6, 10));
        assert_eq!(ids(&events(&mut rx2)), [100, 101], "the script survived");
    }

    #[test]
    fn a_closed_sink_after_a_lazy_load_is_skipped() {
        let b = backend(FakeConfig {
            load_time: 1500 * MS,
            ..FakeConfig::default()
        });
        b.push(script(&[100, 101], MS));
        let (mut rx, cancel) = submit(&b, req(5, 10));
        wait(|| (b.state() == Lifecycle::Loading).then_some(()));
        cancel.cancel();
        wait(|| (b.state() == Lifecycle::Ready).then_some(()));
        // The load finished, the request was dropped unrun and took no script.
        let (mut rx2, _c2) = submit(&b, req(6, 10));
        assert_eq!(ids(&events(&mut rx2)), [100, 101], "the script survived");
        assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
        let lens: Vec<_> = b.requests().iter().map(|r| r.1.prompt.len()).collect();
        assert_eq!(lens, [6]);
        assert_eq!(b.loads().len(), 1);
    }

    #[test]
    fn a_load_failure_fails_the_queue_and_returns_to_idle() {
        let b = backend(FakeConfig {
            load_time: 1500 * MS,
            ..FakeConfig::default()
        });
        b.push(Script {
            load_fails: Some("no gpu".into()),
            ..Script::default()
        });
        let (mut trigger, _c0) = submit(&b, req(1, 10));
        wait(|| (b.state() == Lifecycle::Loading).then_some(()));
        let (mut q1, _c1) = submit(&b, req(2, 10));
        let (mut q2, _c2) = submit(&b, req(3, 10));
        let mut l = b.load();
        let u = b.unload();
        let want = GenEvent::Error {
            kind: ErrorKind::LoadFailed,
            message: "no gpu".into(),
        };
        // The trigger is answered last: when its error is seen, the queued ones are already answered.
        assert_eq!(next(&mut trigger), want);
        assert_eq!(q1.try_recv(), Ok(want.clone()));
        assert_eq!(q2.try_recv(), Ok(want.clone()));
        let queued_load = l.try_recv().expect("the queued load is already answered");
        assert_eq!(
            queued_load.unwrap_err(),
            BackendError {
                kind: ErrorKind::LoadFailed,
                message: "no gpu".into()
            }
        );
        // The queued unload proceeds (nothing is loaded: Ok).
        assert!(reply(u).is_ok());
        assert_eq!(b.state(), Lifecycle::Idle);
        assert!(b.loads().is_empty() && b.requests().is_empty());
        // A later request retries the load.
        let (mut rx, _c) = submit(&b, req(4, 10));
        let evs = events(&mut rx);
        assert!(started(&evs));
        assert_eq!((b.loads().len(), b.state()), (1, Lifecycle::Ready));
    }

    #[test]
    fn queue_ms_excludes_the_load() {
        let b = backend(FakeConfig {
            load_time: 400 * MS,
            ..FakeConfig::default()
        });
        let t_submit = Instant::now();
        let (mut rx, _c) = submit(&b, req(5, 10));
        let (_, t, _) = done_of(&events(&mut rx));
        let total = ms(t_submit.elapsed());
        assert!(t.load_ms >= 400.0, "{t:?}");
        // Queue and load are disjoint spans of the request's life: had queue_ms included the load, the two would
        // add up to more than the whole.
        assert!(t.queue_ms + t.load_ms <= total, "{t:?} within {total}");
        // Waiting behind another request is queue time.
        let b = backend(loaded());
        b.push(script(&[100; 4], 50 * MS));
        let (mut rx1, _c1) = submit(&b, req(1, 10));
        let (mut rx2, _c2) = submit(&b, req(2, 10));
        events(&mut rx1);
        let (_, t2, _) = done_of(&events(&mut rx2));
        assert!(t2.queue_ms >= 150.0, "{t2:?}");
        assert_eq!(t2.load_ms, 0.0);
    }

    #[test]
    fn two_unloads_both_reply() {
        let b = backend(FakeConfig {
            unload_time: 100 * MS,
            ..loaded()
        });
        let (u1, u2) = (b.unload(), b.unload());
        // The first one unloaded the model; the second found nothing loaded.
        assert_eq!((reply(u1).unwrap(), reply(u2).unwrap()), (true, false));
        assert_eq!(b.state(), Lifecycle::Idle);
        assert_eq!(b.unloads().len(), 1);
        // And an unload when idle answers at once.
        assert!(!reply(b.unload()).unwrap());
        assert_eq!(b.unloads().len(), 1);
    }

    #[test]
    fn device_loss_recovers() {
        let b = backend(FakeConfig {
            recover_time: 1500 * MS,
            ..loaded()
        });
        b.push(Script {
            fail_at: Some((1, Fail::DeviceLost)),
            ..script(&[100, 101, 102], 10 * MS)
        });
        let (mut rx, _c) = submit(&b, req(5, 10));
        let evs = events(&mut rx);
        assert_eq!(ids(&evs), [100]);
        assert!(matches!(
            evs.last(),
            Some(GenEvent::Error {
                kind: ErrorKind::DeviceLost,
                ..
            })
        ));
        wait(|| (b.state() == Lifecycle::Recovering).then_some(()));
        wait(|| (b.state() == Lifecycle::Ready).then_some(()));
        // The worker serves the next request.
        let (mut rx2, _c2) = submit(&b, req(6, 10));
        assert!(matches!(finish(&events(&mut rx2)), Finish::Stop(_)));
        assert_eq!(b.runs()[0].finish, None);
    }

    #[test]
    fn idle_unload_fires_after_the_timeout() {
        let b = backend(FakeConfig {
            idle_unload: Some(200 * MS),
            ..loaded()
        });
        assert_eq!(b.state(), Lifecycle::Ready);
        wait(|| (b.unloads().len() == 1).then_some(()));
        assert_eq!(b.state(), Lifecycle::Idle);
        // Idle: no timer runs, and a request loads it again.
        let (mut rx, _c) = submit(&b, req(5, 10));
        assert!(started(&events(&mut rx)));
        assert_eq!(b.loads().len(), 1);
    }

    #[test]
    fn a_stats_command_does_not_restart_the_idle_timer() {
        let idle = 1000 * MS;
        let b = backend(FakeConfig {
            idle_unload: Some(idle),
            ..loaded()
        });
        std::thread::sleep(800 * MS);
        let t_stats = Instant::now();
        assert!(reply(b.stats()).is_ok());
        wait(|| (b.unloads().len() == 1).then_some(()));
        // The timer counts from the last request or load, not from any command: it fires `idle` after the start, some
        // 200 ms after the stats command; a timer the command restarted would fire `idle` after it.
        assert!(b.unloads()[0] < t_stats + idle / 2, "the timer restarted");
        // A finished request does restart it.
        let (mut rx, _c) = submit(&b, req(5, 10));
        events(&mut rx);
        let t_done = Instant::now();
        wait(|| (b.unloads().len() == 2).then_some(()));
        assert!(b.unloads()[1] >= t_done);
    }

    #[test]
    fn the_think_budget_forces_the_close_tokens() {
        let b = backend(loaded());
        let think = ThinkLimit {
            open: true,
            budget: Some(2),
            ends: [90, 91],
            close: vec![7, 8, 9],
        };
        b.push(script(&[100, 101, 102, 103], MS));
        let mut r = req(5, 50);
        r.think = Some(think.clone());
        r.logprobs = Some(2);
        let (mut rx, _c) = submit(&b, r);
        let evs = events(&mut rx);
        assert_eq!(ids(&evs), [100, 101, 7, 8, 9, 102, 103]);
        let toks: Vec<_> = evs
            .iter()
            .filter_map(|e| match e {
                GenEvent::Token {
                    id,
                    forced,
                    logprobs,
                } => Some((*id, *forced, logprobs.clone())),
                _ => None,
            })
            .collect();
        for (id, forced, lp) in &toks {
            if *forced {
                assert!(lp.is_none(), "{id}");
            } else {
                let lp = lp.as_ref().expect("a sampled token carries logprobs");
                assert_eq!(lp.logprob, -0.25);
                assert_eq!(lp.top, vec![(*id, -0.25), (*id + 1, -1.25)]);
            }
        }
        assert_eq!(
            toks.iter().map(|t| t.1).collect::<Vec<_>>(),
            [false, false, true, true, true, false, false]
        );
        // 7 sent tokens plus the stop token the script ends with.
        let (counts, _, cache) = done_of(&evs);
        assert_eq!((counts.completion, counts.prompt), (8, 5));
        assert_eq!(cache.hit_rate, None);

        // A reasoning end sampled before the budget closes the budget's reach: nothing is forced.
        b.push(script(&[100, 90, 101, 102], MS));
        let mut r = req(5, 50);
        r.think = Some(think);
        let (mut rx, _c) = submit(&b, r);
        assert_eq!(ids(&events(&mut rx)), [100, 90, 101, 102]);
    }

    #[test]
    fn vanish_ends_the_stream_without_a_terminal_event() {
        let b = backend(loaded());
        b.push(Script {
            fail_at: Some((1, Fail::Vanish)),
            ..script(&[100, 101, 102], 10 * MS)
        });
        let (mut rx, _c) = submit(&b, req(5, 10));
        let evs = events(&mut rx);
        assert!(started(&evs));
        assert_eq!(ids(&evs), [100]);
        assert!(!matches!(
            evs.last(),
            Some(GenEvent::Done { .. } | GenEvent::Error { .. })
        ));
        // The worker is gone.
        let (sink, _rx2, _c2) = Sink::channel();
        assert_eq!(b.submit(req(6, 10), sink), Err(Full));
    }

    #[test]
    fn done_reports_counts_and_a_zeroed_cache_report() {
        let b = backend(loaded());
        b.push(Script {
            reused: 3,
            source: Source::PromptEnd,
            diverged_at: 4,
            ..script(&[100, 101], MS)
        });
        let (mut rx, _c) = submit(&b, req(10, 50));
        let evs = events(&mut rx);
        assert_eq!(
            evs[0],
            GenEvent::Started {
                prompt_tokens: 10,
                reused_tokens: 3,
                restored: Source::PromptEnd,
                diverged_at: 4
            }
        );
        let (counts, timings, cache) = done_of(&evs);
        assert_eq!(
            counts,
            Counts {
                prompt: 10,
                reused: 3,
                prefilled: 7,
                completion: 3,
                reasoning: 0
            }
        );
        assert!(timings.decode_ms >= 0.0);
        assert_eq!(
            cache,
            CacheReport {
                hit_rate: None,
                miss_mib_per_token: None,
                restore_bytes: 0,
                pool_bytes_used: 0,
                decode_misses: None
            }
        );
        let stats = BackendStats {
            snapshots_held: 2,
            ..BackendStats::default()
        };
        b.set_stats(stats.clone());
        assert_eq!(reply(b.stats()).unwrap(), stats);
    }

    #[test]
    fn the_fake_reports_progress() {
        let b = backend(loaded());
        assert_eq!(b.progress(), Some(Progress::default()));
        assert_eq!(b.latest_stats(), None);

        // A request counts as pending from its submit, and a tick for each phase, token and answer follows.
        b.push(script(&[100, 101, 102], 20 * MS));
        let (mut rx, _c) = submit(&b, req(4, 10));
        assert_eq!(b.progress().unwrap().pending, 1);
        events(&mut rx);
        let done = wait(|| {
            let p = b.progress().unwrap();
            (p.pending == 0).then_some(p)
        });
        // Three tokens, the start, the decode and the answer, at the least.
        assert!(done.ticks >= 6, "{done:?}");

        // A command is pending until its reply, and the answer is kept as the latest copy, whether or not its asker
        // still waits.
        let stats = BackendStats {
            snapshots_held: 7,
            ..BackendStats::default()
        };
        b.set_stats(stats.clone());
        let before = Instant::now();
        let ticks = b.progress().unwrap().ticks;
        assert_eq!(reply(b.stats()).unwrap(), stats);
        let copy = b.latest_stats().expect("the answer is kept");
        assert_eq!(copy.stats, stats);
        assert!(copy.at >= before);
        wait(|| (b.progress().unwrap().pending == 0).then_some(()));
        assert!(b.progress().unwrap().ticks > ticks);
        drop(b.stats());
        wait(|| (b.latest_stats().unwrap().at > copy.at).then_some(()));

        // A request the worker skips ends its pending too.
        let (sink, _rx, c) = Sink::channel();
        c.cancel();
        assert!(b.submit(req(5, 10), sink).is_ok());
        wait(|| (b.progress().unwrap().pending == 0).then_some(()));
    }

    #[test]
    fn work_given_to_a_busy_fake_does_not_move_the_ticks() {
        // The worker is held in its load, where it ticks nowhere, so only the submit could move the count. It must
        // not: under steady traffic an arrival that ticked would hide a worker stuck mid-decode from the watchdog.
        let b = backend(FakeConfig {
            load_time: 300 * MS,
            ..FakeConfig::default()
        });
        let load = b.load();
        wait(|| (b.state() == Lifecycle::Loading).then_some(()));
        let t0 = b.progress().unwrap().ticks;
        let (mut rx, _c) = submit(&b, req(2, 3));
        let p = b.progress().unwrap();
        assert_eq!((p.ticks, p.pending), (t0, 2), "a request");
        let _stats = b.stats();
        let p = b.progress().unwrap();
        assert_eq!((p.ticks, p.pending), (t0, 3), "a command");
        assert!(reply(load).is_ok());
        events(&mut rx);
    }

    #[test]
    fn script_text_encodes_with_specials() {
        let (tk, _) = fixtures();
        let s = Script::text(tk, "hello <|im_end|>");
        assert_eq!(s.tokens, tk.encode("hello <|im_end|>", true));
        assert_eq!(s.per_token, Duration::ZERO);
        assert_eq!(s.source, Source::None);
    }
}
