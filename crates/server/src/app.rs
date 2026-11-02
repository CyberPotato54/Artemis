//! The HTTP side: `App`, its router and middleware (the Host check and auth, both on headers alone), `/health`,
//! `/v1/models`, and `serve`.
use crate::backend::{Backend, Lifecycle};
use crate::config::{ApiKey, Config};
use crate::error::{ApiError, ErrorType, Flavour};
use crate::handlers::chat::chat;
use crate::handlers::count::count_tokens;
use crate::handlers::lifecycle::{load, unload};
use crate::handlers::messages::messages;
use crate::ids::{iso_utc, new_id, unix_now};
use crate::logs::{stats_line, LifecycleLine, Logger, RequestLine};
use crate::queue::Queue;
use crate::template::Template;
use crate::tokens::Specials;
use crate::watchdog;
use axum::body::Bytes;
use axum::extract::rejection::{BytesRejection, FailedToBufferBody, PathRejection};
use axum::extract::{ConnectInfo, DefaultBodyLimit, FromRequest, Path, Request, State};
use axum::http::header::AUTHORIZATION;
use axum::http::{HeaderMap, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use qwen36::tokenizer::Tokenizer;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::{oneshot, watch, OwnedSemaphorePermit, Semaphore};
use tokio::task::{AbortHandle, JoinHandle};
use tokio::time::{interval_at, MissedTickBehavior};

/// A request body is read up to this many bytes; one byte more is a 413.
pub const MAX_BODY: usize = 16 << 20;

/// The body of `/v1/load` and `/v1/unload` is read up to this many bytes (they take none, so 4 KiB is plenty); one byte
/// more is a 413.
pub const LIFECYCLE_BODY: usize = 4096;

/// At most this many requests are in preparation at once: the body read's result, its parse, the template renders and
/// the tokenization. Each holds the body and its parsed form, whose size the value cap bounds (at most
/// `MAX_JSON_VALUES` nodes). A request leaves the stage when its prompt build ends, before it waits for the engine.
pub const PREP_PERMITS: usize = 4;

/// At most this many requests wait for the engine; the next gets 503 at once.
pub const QUEUE_LIMIT: usize = 64;

/// Time since an arbitrary start, so a test can drive the device-loss window by hand.
pub trait Clock: Send + Sync {
    fn now(&self) -> Duration;
}

/// The real clock: time since it was made.
pub struct SystemClock(Instant);

impl SystemClock {
    pub fn new() -> SystemClock {
        SystemClock(Instant::now())
    }
}

impl Default for SystemClock {
    fn default() -> SystemClock {
        SystemClock::new()
    }
}

impl Clock for SystemClock {
    fn now(&self) -> Duration {
        self.0.elapsed()
    }
}

/// A clock a test moves by hand: it starts at zero and only `advance` changes it.
#[derive(Default)]
pub struct ManualClock(Mutex<Duration>);

impl ManualClock {
    pub fn new() -> ManualClock {
        ManualClock::default()
    }

    pub fn advance(&self, by: Duration) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) += by;
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Duration {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// What the watchdog calls to end the process, with the exit code (4 for a stall). It must not return in
/// production; tests record the code instead.
pub type Halt = Arc<dyn Fn(i32) + Send + Sync>;

/// The timings a test may change.
#[derive(Clone)]
pub struct Tuning {
    /// How often a stream sends a keep-alive comment (10 s).
    pub keepalive: Duration,
    /// Two device losses within this long end the process (600 s).
    pub loss_window: Duration,
    pub clock: Arc<dyn Clock>,
    /// Once `serve` is told to stop, how long it lets the unload and the open responses finish before it returns
    /// anyway (30 s): a stream that never ends must not hold the process.
    pub shutdown_grace: Duration,
    /// How often the watchdog looks at the backend's progress (5 s).
    pub watchdog_tick: Duration,
    /// How often the `stats` heartbeat line is written (600 s).
    pub stats_every: Duration,
    /// How the watchdog ends the process (`std::process::exit`).
    pub halt: Halt,
    /// How long a client may take to send the head of a request, and to wait idle on a kept-alive connection for the
    /// next one (30 s). It does not limit a response: a stream or a long prefill is not cut by it.
    pub header_timeout: Duration,
    /// How long a request body may take to arrive, from the handler's first read to its last byte (60 s).
    pub body_timeout: Duration,
    /// At most this many connections are served at once (128); the rest wait in the OS backlog. It stays above
    /// `QUEUE_LIMIT` + 1 (one running, the queue full) so the queue's "full" 503 is reachable, with room for idle
    /// keep-alive connections.
    pub max_connections: usize,
    /// How long a request waits for one of the `PREP_PERMITS` before it is answered 503 (10 s).
    pub prep_wait: Duration,
}

impl Default for Tuning {
    fn default() -> Tuning {
        Tuning {
            keepalive: Duration::from_secs(10),
            loss_window: Duration::from_secs(600),
            clock: Arc::new(SystemClock::new()),
            shutdown_grace: Duration::from_secs(30),
            watchdog_tick: Duration::from_secs(5),
            stats_every: Duration::from_secs(600),
            halt: Arc::new(|code| std::process::exit(code)),
            header_timeout: Duration::from_secs(30),
            body_timeout: Duration::from_secs(60),
            max_connections: 128,
            prep_wait: Duration::from_secs(10),
        }
    }
}

/// Everything an `App` is built from.
pub struct AppParts {
    pub config: Config,
    pub key: Option<ApiKey>,
    pub backend: Arc<dyn Backend>,
    pub tokenizer: Arc<Tokenizer>,
    pub specials: Specials,
    pub template: Template,
    pub logger: Logger,
    pub tuning: Tuning,
    /// What `/health` reports as `version`: the binary's `CARGO_PKG_VERSION`.
    pub version: &'static str,
}

/// Why `serve` returned without being asked to stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    DeviceLostTwice,
    EngineGone,
}

struct Inner {
    config: Config,
    key: Option<ApiKey>,
    backend: Arc<dyn Backend>,
    tokenizer: Arc<Tokenizer>,
    specials: Specials,
    template: Template,
    logger: Logger,
    tuning: Tuning,
    version: &'static str,
    /// When this process built its app: `created` in `/v1/models`.
    started: u64,
    /// The `Host` values accepted when there is no key (lower case, no port).
    hosts: Vec<String>,
    exit: watch::Sender<Option<Exit>>,
    /// When the engine last lost the device, by the tuning's clock.
    last_loss: Mutex<Option<Duration>>,
    queue: Queue,
    /// The `PREP_PERMITS` preparation slots (`App::prep_permit`).
    prep: Arc<Semaphore>,
    /// Set when `serve` is stopping: no generation or load is taken from then on.
    stopping: AtomicBool,
    /// Held across a queue entry and its submit, so the engine meets requests in the queue's order.
    submit_order: Mutex<()>,
    /// The ids of the requests between `Exchange::begin` and the end of their `Exchange`, in the order requests entered
    /// `begin` (which can differ from submit order by a hair).
    in_flight: Mutex<InFlightSet>,
    /// When the app was built: the `uptime_s` of the `stats` lines and `/v1/stats`.
    booted: Instant,
    /// Set by the watchdog when it has decided to halt: the process is about to end with the watchdog's own `exit`
    /// line, so `serve` and the heartbeat write nothing more.
    halting: AtomicBool,
}

#[derive(Default)]
struct InFlightSet {
    next: u64,
    ids: BTreeMap<u64, String>,
}

/// A request's place in the in-flight registry, taken first by `Exchange::begin` and given up when the guard drops: at
/// every early return of `begin`, and with the `Exchange` it moves into until the engine reports the end of the
/// request. A request that reached the engine leaves the stats copy to be refreshed as it goes.
pub(crate) struct InFlight {
    app: App,
    key: u64,
    /// The request was handed to the engine.
    submitted: bool,
}

impl InFlight {
    /// The request is with the engine now, so its end is worth a refresh of the stats copy.
    pub(crate) fn submitted(&mut self) {
        self.submitted = true;
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.app.inner.flights().ids.remove(&self.key);
        if self.submitted {
            // Fire and forget: the backend keeps the answer whether or not anyone waits for it (`latest_stats`). A
            // request the server refused before the engine saw it (queue full, stopping) asks for nothing, so a flood
            // of them can't flood the engine's command queue.
            drop(self.app.backend().stats());
        }
    }
}

/// The server's shared state; cheap to clone.
#[derive(Clone)]
pub struct App {
    inner: Arc<Inner>,
}

impl Inner {
    fn flights(&self) -> MutexGuard<'_, InFlightSet> {
        self.in_flight
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// The host as it appears in a `Host` header: lower case, an IPv6 address in brackets.
fn host_form(host: &str) -> String {
    let h = host.trim().to_ascii_lowercase();
    if h.contains(':') && !h.starts_with('[') {
        format!("[{h}]")
    } else {
        h
    }
}

impl App {
    pub fn new(p: AppParts) -> Result<App, String> {
        if p.config.host.trim().is_empty() {
            return Err("host is empty".into());
        }
        if p.config.ctx == 0 {
            return Err("ctx must be above 0".into());
        }
        if p.config.model_id.is_empty() {
            return Err("model_id is empty".into());
        }
        let mut hosts: Vec<String> = ["localhost", "127.0.0.1", "[::1]"].map(String::from).into();
        hosts.push(host_form(&p.config.host));
        let (exit, _) = watch::channel(None);
        Ok(App {
            inner: Arc::new(Inner {
                config: p.config,
                key: p.key,
                backend: p.backend,
                tokenizer: p.tokenizer,
                specials: p.specials,
                template: p.template,
                logger: p.logger,
                tuning: p.tuning,
                version: p.version,
                started: unix_now(),
                hosts,
                exit,
                last_loss: Mutex::new(None),
                queue: Queue::new(QUEUE_LIMIT),
                prep: Arc::new(Semaphore::new(PREP_PERMITS)),
                stopping: AtomicBool::new(false),
                submit_order: Mutex::new(()),
                in_flight: Mutex::new(InFlightSet::default()),
                booted: Instant::now(),
                halting: AtomicBool::new(false),
            }),
        })
    }

    pub fn config(&self) -> &Config {
        &self.inner.config
    }

    pub fn key(&self) -> Option<&ApiKey> {
        self.inner.key.as_ref()
    }

    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.inner.backend
    }

    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.inner.tokenizer
    }

    pub fn specials(&self) -> &Specials {
        &self.inner.specials
    }

    pub fn template(&self) -> &Template {
        &self.inner.template
    }

    pub fn logger(&self) -> &Logger {
        &self.inner.logger
    }

    pub fn queue(&self) -> &Queue {
        &self.inner.queue
    }

    /// A slot for a request's preparation: the permit, held until the prompt is built. Past `Tuning::prep_wait`
    /// without one, a 503. The body is already read when a handler asks, so what the permits bound is the parsed form,
    /// the renders and the tokenization, which cost many times the body.
    pub(crate) async fn prep_permit(&self) -> Result<OwnedSemaphorePermit, ApiError> {
        let busy = || {
            ApiError::new(
                503,
                ErrorType::Unavailable,
                "the server is busy preparing other requests; try again",
            )
        };
        let wait = self.inner.prep.clone().acquire_owned();
        match tokio::time::timeout(self.inner.tuning.prep_wait, wait).await {
            Ok(Ok(permit)) => Ok(permit),
            // The semaphore is never closed; if it were, the server is not preparing anything.
            Ok(Err(_)) | Err(_) => Err(busy()),
        }
    }

    /// Writes the line of a request refused before any handler: its status and error kind, the route it matched (or
    /// `unknown`, never the raw path), the client's IP when the guard refused it, and nothing else.
    fn log_refusal(&self, path: &str, peer: Option<IpAddr>, e: &ApiError, flavour: Flavour) {
        let mut line = RequestLine::new(new_id("req_"), route_of(path));
        line.peer = peer.map(|ip| ip.to_string());
        line.status = e.status;
        line.set_error(e, flavour);
        self.inner.logger.request(&line);
    }

    /// Taken by `Exchange::begin` around `Queue::enter` and `Backend::submit`, neither of which waits.
    pub(crate) fn submit_order(&self) -> MutexGuard<'_, ()> {
        self.inner
            .submit_order
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Registers request `id` as in flight until the returned guard drops.
    pub(crate) fn enter_flight(&self, id: &str) -> InFlight {
        let mut set = self.inner.flights();
        let key = set.next;
        set.next += 1;
        set.ids.insert(key, id.to_string());
        InFlight {
            app: self.clone(),
            key,
            submitted: false,
        }
    }

    /// Whole seconds since the app was built.
    pub fn uptime_s(&self) -> u64 {
        self.inner.booted.elapsed().as_secs()
    }

    /// The `stats` line for now: the backend's latest copy (never a question to the engine), the uptime and the log's
    /// write failures.
    pub fn stats_line(&self) -> LifecycleLine {
        stats_line(
            self.backend().latest_stats().as_ref(),
            self.uptime_s(),
            self.logger().write_failures(),
        )
    }

    /// The watchdog has decided to end the process; true for the first caller. From here `serve` and the heartbeat
    /// write no line, so the watchdog's `exit` is the log's last and the only one.
    pub(crate) fn begin_halt(&self) -> bool {
        !self.inner.halting.swap(true, Ordering::AcqRel)
    }

    /// The watchdog is halting the process (`begin_halt` was called).
    pub fn halting(&self) -> bool {
        self.inner.halting.load(Ordering::Acquire)
    }

    /// The id of the request that has been in flight longest, if any: the worker is FIFO, so that is the one it runs,
    /// or the one stuck behind the stall.
    pub fn oldest_in_flight(&self) -> Option<String> {
        self.inner.flights().ids.values().next().cloned()
    }

    /// Why `serve` has been asked to exit, if it has.
    pub fn exited(&self) -> Option<Exit> {
        *self.inner.exit.borrow()
    }

    pub fn tuning(&self) -> &Tuning {
        &self.inner.tuning
    }

    /// The engine lost the device (`ErrorKind::DeviceLost` reached a request or a `/v1/load`), with `message`, which is
    /// the engine's and never carries request text; it is written to the log. The loss is recorded at
    /// the tuning's clock; one within `loss_window` of the previous loss makes `serve` return `Err(DeviceLostTwice)`.
    /// One source per loss: if the engine adapter also reports losses from the worker's lifecycle observer, it
    /// must not report one that a request already carried here, and must never deliver one loss to several sinks or
    /// replies as `DeviceLost`, or a single loss counts twice and ends the process. The adapter calls it for a loss
    /// after a reply.
    pub fn device_lost(&self, message: &str) {
        self.inner.logger.lifecycle(&LifecycleLine::DeviceLost {
            message: message.to_string(),
        });
        let now = self.inner.tuning.clock.now();
        let window = self.inner.tuning.loss_window;
        let mut last = self
            .inner
            .last_loss
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let twice = last.is_some_and(|before| now.saturating_sub(before) < window);
        *last = Some(now);
        drop(last);
        if twice {
            tracing::error!(
                "the GPU was lost twice within {} s: the server is exiting",
                window.as_secs()
            );
            self.exit(Exit::DeviceLostTwice);
        }
    }

    /// Stops taking generations and loads (they answer 503), and waits out any that is entering the queue: what
    /// `abort_all` then takes is everything that will ever be queued.
    pub(crate) fn begin_stopping(&self) {
        let _order = self.submit_order();
        self.inner.stopping.store(true, Ordering::Release);
    }

    /// `serve` has been asked to stop or exit.
    pub(crate) fn is_stopping(&self) -> bool {
        self.inner.stopping.load(Ordering::Acquire)
    }

    /// Makes `serve` return `Err(why)`, once the connections in flight have finished or `Tuning::shutdown_grace` has
    /// passed, whichever is first. The first reason stays.
    pub fn exit(&self, why: Exit) {
        self.inner.exit.send_if_modified(|e| {
            let first = e.is_none();
            if first {
                *e = Some(why);
            }
            first
        });
    }

    /// The router, with the Host check and auth in front of every route, the fallbacks included.
    pub fn router(&self) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(models))
            .route("/v1/models/{id}", get(model))
            .route("/v1/stats", get(stats))
            .route("/v1/chat/completions", post(chat))
            .route("/v1/messages", post(messages))
            .route("/v1/messages/count_tokens", post(count_tokens))
            // A route layer overrides the router's `DefaultBodyLimit` below.
            .route(
                "/v1/load",
                post(load).layer(DefaultBodyLimit::max(LIFECYCLE_BODY)),
            )
            .route(
                "/v1/unload",
                post(unload).layer(DefaultBodyLimit::max(LIFECYCLE_BODY)),
            )
            .fallback(not_found)
            .method_not_allowed_fallback(method_not_allowed)
            // After the fallbacks, so an unknown path and a wrong method are checked too (axum 0.8's `Router::layer`
            // wraps the path routes, the fallback router and the catch-all fallback).
            .layer(middleware::from_fn_with_state(self.clone(), guard))
            // The limit `read_body` (axum's `Bytes` extractor) enforces; the guard above it never reads a body.
            .layer(DefaultBodyLimit::max(MAX_BODY))
            .with_state(self.clone())
    }

    /// One entry of `/v1/models`.
    fn model_entry(&self, id: &str) -> Value {
        json!({
            "id": id, "object": "model", "created": self.inner.started, "owned_by": "artemis",
            "type": "model", "display_name": id, "created_at": iso_utc(self.inner.started),
            "context_length": self.config().ctx, "max_model_len": self.config().ctx,
        })
    }
}

/// Runs the server on `listener` until `shutdown` resolves or `App::exit` is called, then `Ok(())` for a shutdown and
/// `Err(why)` for an exit.
///
/// Stopping first answers the queued requests 503 and unloads, which ends the request in flight `Unloaded`;
/// both happen inside the future that gates the accept loop's graceful shutdown (`listen::run`), while the
/// connections are still open, so a streaming client gets its error event and the stream ends. Only then do the
/// connections in flight finish. The whole of it is held to `Tuning::shutdown_grace`: past that `serve` returns
/// anyway, so a stream that never ends (or an unload that never answers) can't hold the process.
pub async fn serve(
    app: App,
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), Exit> {
    let mut exit = app.inner.exit.subscribe();
    app.logger().lifecycle(&LifecycleLine::Start {
        version: app.inner.version.to_string(),
        bind: listener
            .local_addr()
            .map_or_else(|_| app.config().host.clone(), |a| a.to_string()),
        model_id: app.config().model_id.clone(),
        log_dir: app.logger().dir().display().to_string(),
    });
    let (end_watchdog, watchdog_done) = keep_watch(&app, "watchdog", watchdog::spawn(&app));
    let (end_heartbeat, heartbeat_done) =
        keep_watch(&app, "heartbeat", tokio::spawn(heartbeat(app.clone())));
    let (stopped_tx, stopped_rx) = oneshot::channel();
    let stopping = app.clone();
    let ends = [end_watchdog.clone(), end_heartbeat.clone()];
    let stop = async move {
        tokio::select! {
            () = shutdown => {}
            () = async { let _ = exit.wait_for(Option::is_some).await; } => {}
        }
        // Before anything else, so a stopping server never writes `stall` or `stats` after its `exit`.
        for end in &ends {
            end.abort();
        }
        let _ = stopped_tx.send(());
        // The door first, then the queue: nothing enters behind the unload.
        stopping.begin_stopping();
        stopping.queue().abort_all();
        // A worker that is gone answers with a closed reply, which is as good as done.
        // A halting process is the watchdog's: its lines are the last. (The check and the write are not one step: a
        // line can still land after the watchdog's `exit`, in the microseconds before the process ends.)
        if let Ok(Ok(true)) = stopping.backend().shutdown().await {
            if !stopping.halting() {
                stopping.logger().lifecycle(&LifecycleLine::Unload {
                    cause: "shutdown".into(),
                });
            }
        }
    };
    let server = crate::listen::run(listener, app.router(), &app.inner.tuning, stop);
    let grace = app.inner.tuning.shutdown_grace;
    let out_of_time = async {
        if stopped_rx.await.is_ok() {
            tokio::time::sleep(grace).await;
        } else {
            // The server ended without being stopped.
            std::future::pending::<()>().await;
        }
    };
    tokio::select! {
        () = server => {}
        () = out_of_time => {
            tracing::warn!("shutdown did not finish within {} s: closing anyway", grace.as_secs());
        }
    }
    // The same for a server that ended without being stopped. Both tasks are gone before the last lines are written.
    end_watchdog.abort();
    end_heartbeat.abort();
    let _ = tokio::join!(watchdog_done, heartbeat_done);
    let why = *app.inner.exit.borrow();
    // The process's last words: its figures, then why it ended. A halting watchdog writes its own.
    if !app.halting() {
        app.logger().lifecycle(&app.stats_line());
        app.logger().lifecycle(&LifecycleLine::Exit {
            reason: match why {
                None => "shutdown",
                Some(Exit::DeviceLostTwice) => "device_lost_twice",
                Some(Exit::EngineGone) => "engine_gone",
            }
            .into(),
        });
    }
    why.map_or(Ok(()), Err)
}

/// Watches `task`, which `serve` counts on until its stop: if it ends before then, or panics, the log says so, because
/// a task that is gone does its job no more and nothing else shows it. Returns the handle that ends the task (a
/// cancelled task says nothing) and the watcher, which finishes once the task has.
fn keep_watch(
    app: &App,
    name: &'static str,
    task: JoinHandle<()>,
) -> (AbortHandle, JoinHandle<()>) {
    let end = task.abort_handle();
    let app = app.clone();
    let watcher = tokio::spawn(async move {
        match task.await {
            // The watchdog returns after it halts the process, which is not a loss.
            Ok(()) if app.halting() => {}
            Ok(()) => tracing::warn!("the {name} task ended before the server was stopped"),
            Err(e) if e.is_panic() => tracing::error!("the {name} task panicked"),
            Err(_) => {}
        }
    });
    (end, watcher)
}

/// The `stats` heartbeat: every `Tuning::stats_every`, starting one period after `start`, it asks the
/// engine for its figures, waits at most `HEARTBEAT_WAIT` for the answer, and writes the line from the latest copy and
/// that copy's age. Because the line is written whether or not the engine answered, a gap of more than a period between
/// `stats` lines means the server's own runtime has stopped.
async fn heartbeat(app: App) {
    let every = app.tuning().stats_every.max(Duration::from_millis(1));
    let mut tick = interval_at(tokio::time::Instant::now() + every, every);
    tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        // The backend stores the answer itself: a timeout drops only the wait for it.
        let _ = tokio::time::timeout(HEARTBEAT_WAIT, app.backend().stats()).await;
        // Check, then write: not one step, so a line can land after the watchdog's `exit` in the microseconds before
        // the process ends.
        if !app.halting() {
            app.logger().lifecycle(&app.stats_line());
        }
    }
}

/// How long the heartbeat waits for the engine's answer before it writes the line from the copy it has.
const HEARTBEAT_WAIT: Duration = Duration::from_secs(2);

/// The runtime the server runs on: multi-thread, 2 workers, which is enough for one sequence.
pub fn runtime() -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("artemis-http")
        .enable_all()
        .build()
}

/// Reads a request's body, which `DefaultBodyLimit` (set by the router) caps at `MAX_BODY`, within
/// `Tuning::body_timeout`. Past `limit` is a 413, whether the length was announced, understated or absent (chunked); a
/// body that has not fully arrived in time is a 408; any other failure (the client left, a bad chunk) a 400.
pub async fn read_body(app: &App, req: Request, limit: usize) -> Result<Bytes, ApiError> {
    let within = app.tuning().body_timeout;
    let read = tokio::time::timeout(within, Bytes::from_request(req, &()));
    match read.await {
        Ok(Ok(bytes)) => Ok(bytes),
        Ok(Err(BytesRejection::FailedToBufferBody(FailedToBufferBody::LengthLimitError(_)))) => {
            Err(ApiError::new(
                413,
                ErrorType::TooLarge,
                format!("the request body is over {}", size_text(limit)),
            ))
        }
        Ok(Err(_)) => Err(ApiError::new(
            400,
            ErrorType::InvalidRequest,
            "the request body could not be read",
        )),
        Err(_) => Err(ApiError::new(
            408,
            ErrorType::InvalidRequest,
            format!(
                "the request body did not arrive within {} s",
                within.as_secs().max(1)
            ),
        )),
    }
}

/// A byte limit as the 413 message says it: `16 MiB`, `4 KiB`, or bytes when it is neither.
fn size_text(bytes: usize) -> String {
    if bytes != 0 && bytes.is_multiple_of(1 << 20) {
        format!("{} MiB", bytes >> 20)
    } else if bytes != 0 && bytes.is_multiple_of(1 << 10) {
        format!("{} KiB", bytes >> 10)
    } else {
        format!("{bytes} bytes")
    }
}

/// The route a path matches, as the request line names it: a constant, so no part of the path itself is logged.
fn route_of(path: &str) -> &'static str {
    const ROUTES: [&str; 8] = [
        "/health",
        "/v1/models",
        "/v1/stats",
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/load",
        "/v1/unload",
    ];
    if let Some(route) = ROUTES.iter().find(|r| **r == path) {
        route
    } else if path
        .strip_prefix("/v1/models/")
        .is_some_and(|id| !id.is_empty() && !id.contains('/'))
    {
        "/v1/models/{id}"
    } else {
        "unknown"
    }
}

/// `path` is under `/v1/`.
fn is_v1(path: &str) -> bool {
    path == "/v1" || path.starts_with("/v1/")
}

/// The host of a `Host` header, lower case, without its port; `None` when it is not a host with an optional numeric
/// port (`[::1]:8090` gives `[::1]`; `127.0.0.1:80@x`, `::1` and `localhost:abc` give `None`).
fn host_of(value: &str) -> Option<String> {
    // `port` keeps its colon.
    let (host, port) = if value.starts_with('[') {
        value.split_at(value.find(']')? + 1)
    } else {
        value.split_at(value.rfind(':').unwrap_or(value.len()))
    };
    let port_ok = match port.strip_prefix(':') {
        None => port.is_empty(),
        Some(digits) => digits.bytes().all(|b| b.is_ascii_digit()),
    };
    (port_ok && !host.is_empty() && (host.starts_with('[') || !host.contains(':')))
        .then(|| host.to_ascii_lowercase())
}

/// `origin` without a trailing `/`, lower case.
fn origin_form(origin: &[u8]) -> Vec<u8> {
    let origin = origin.strip_suffix(b"/").unwrap_or(origin);
    origin.to_ascii_lowercase()
}

/// A browser page cannot reach the server unless its origin is in `allowed_origins`. Any `Origin` header must
/// match an entry (ASCII case-insensitive, a trailing `/` ignored); without an allowed `Origin`, a `Sec-Fetch-Site` of
/// `cross-site` or `same-site` is refused. Requests with neither header (curl, SDKs, agents) pass.
fn origin_allowed(app: &App, headers: &HeaderMap) -> Result<(), ApiError> {
    let allowed = &app.config().allowed_origins;
    let mut origins = headers
        .get_all(axum::http::header::ORIGIN)
        .iter()
        .peekable();
    if origins.peek().is_some() {
        let all_allowed = origins.all(|o| {
            let o = origin_form(o.as_bytes());
            allowed.iter().any(|a| origin_form(a.as_bytes()) == o)
        });
        return if all_allowed {
            Ok(())
        } else {
            Err(ApiError::new(
                403,
                ErrorType::Permission,
                "the Origin header is not allowed; add the origin to allowed_origins to permit a browser-based client",
            ))
        };
    }
    let cross = headers.get_all("sec-fetch-site").iter().any(|v| {
        let v = v.as_bytes().trim_ascii();
        v.eq_ignore_ascii_case(b"cross-site") || v.eq_ignore_ascii_case(b"same-site")
    });
    if cross {
        return Err(ApiError::new(
            403,
            ErrorType::Permission,
            "a cross-site browser request is not allowed; add its origin to allowed_origins",
        ));
    }
    Ok(())
}

/// A POST with a body must say it is JSON: `application/json` or `application/*+json`, parameters ignored. A form or
/// plain-text post is the one a web page can send without a CORS preflight, so it never reaches a handler. A POST
/// with no body (load, unload) passes with any type or none.
fn json_body_ok(method: &Method, headers: &HeaderMap) -> Result<(), ApiError> {
    if method != Method::POST {
        return Ok(());
    }
    let has_body = headers.contains_key(axum::http::header::TRANSFER_ENCODING)
        || headers
            .get_all(axum::http::header::CONTENT_LENGTH)
            .iter()
            .any(|v| {
                v.to_str()
                    .ok()
                    .and_then(|v| v.trim().parse::<u64>().ok())
                    .is_none_or(|n| n > 0)
            });
    if !has_body {
        return Ok(());
    }
    let json_type = |v: &[u8]| {
        let media = v.split(|&b| b == b';').next().unwrap_or_default();
        let media = media.trim_ascii().to_ascii_lowercase();
        media == b"application/json"
            || (media.starts_with(b"application/")
                && media.ends_with(b"+json")
                && media.len() > b"application/+json".len())
    };
    let mut types = headers
        .get_all(axum::http::header::CONTENT_TYPE)
        .iter()
        .peekable();
    if types.peek().is_some() && types.all(|v| json_type(v.as_bytes())) {
        Ok(())
    } else {
        Err(ApiError::new(
            415,
            ErrorType::InvalidRequest,
            "the request body must be JSON, sent with Content-Type: application/json",
        ))
    }
}

/// The guard's refusal as a response that also closes the connection: a refused client (a wrong key, a foreign
/// origin) gets no kept-alive connection, so it cannot hold one of the `max_connections` slots until the header
/// timeout.
fn refused(e: ApiError, flavour: Flavour) -> Response {
    let mut response = e.into_response(flavour);
    response.headers_mut().insert(
        axum::http::header::CONNECTION,
        axum::http::HeaderValue::from_static("close"),
    );
    response
}

/// The Origin and Content-Type checks, then the Host check (no key) or auth (a key): headers only, before any body is
/// read.
async fn guard(State(app): State<App>, req: Request, next: Next) -> Response {
    let path = req.uri().path();
    let flavour = Flavour::of(path);
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0.ip());
    if let Err(e) =
        origin_allowed(&app, req.headers()).and_then(|()| json_body_ok(req.method(), req.headers()))
    {
        app.log_refusal(path, peer, &e, flavour);
        return refused(e, flavour);
    }
    match app.key() {
        None => {
            let ok = req
                .headers()
                .get(axum::http::header::HOST)
                .and_then(|h| h.to_str().ok())
                .and_then(host_of)
                .is_some_and(|h| app.inner.hosts.contains(&h));
            if !ok {
                let e = ApiError::new(403, ErrorType::Permission, "the Host header is not allowed");
                app.log_refusal(path, peer, &e, flavour);
                return refused(e, flavour);
            }
        }
        Some(key) if is_v1(path) => {
            if !authorized(key, req.headers()) {
                let e = ApiError::new(401, ErrorType::Authentication, "missing or invalid API key");
                app.log_refusal(path, peer, &e, flavour);
                return refused(e, flavour);
            }
        }
        Some(_) => {}
    }
    next.run(req).await
}

/// `Authorization: Bearer <key>` (the scheme in any case) or `x-api-key: <key>`, compared in constant time.
fn authorized(key: &ApiKey, headers: &HeaderMap) -> bool {
    let bearer = headers.get_all(AUTHORIZATION).iter().fold(false, |ok, v| {
        let v = v.as_bytes();
        let split = v
            .iter()
            .position(u8::is_ascii_whitespace)
            .unwrap_or(v.len());
        let (scheme, rest) = v.split_at(split);
        let token = rest.trim_ascii_start();
        // Both are evaluated, so a match does not show in the time taken.
        let matches = key.matches(token);
        ok | (scheme.eq_ignore_ascii_case(b"bearer") & matches)
    });
    let x_api_key = headers
        .get_all("x-api-key")
        .iter()
        .fold(false, |ok, v| ok | key.matches(v.as_bytes()));
    bearer | x_api_key
}

/// The state as `/health` and `/v1/stats` name it.
fn lifecycle_name(state: Lifecycle) -> &'static str {
    match state {
        Lifecycle::Idle => "idle",
        Lifecycle::Ready => "ready",
        Lifecycle::Loading => "loading",
        Lifecycle::Recovering => "recovering",
    }
}

async fn health(State(app): State<App>) -> Response {
    let state = app.backend().state();
    let name = lifecycle_name(state);
    let status = match state {
        Lifecycle::Idle | Lifecycle::Ready => StatusCode::OK,
        Lifecycle::Loading | Lifecycle::Recovering => StatusCode::SERVICE_UNAVAILABLE,
    };
    let c = app.config();
    let body = json!({
        "service": c.health_service,
        "status": name,
        "version": app.inner.version,
        "loaded": name == "ready",
        "model": c.model_id,
        "max_context": c.ctx,
        "api_key": app.key().is_some(),
    });
    (status, Json(body)).into_response()
}

/// `GET /v1/stats`: the latest copy of the engine's figures with its age, the progress counter and the
/// state. It reads what the backend keeps and never asks the engine, so it answers while the engine is stuck. A 200
/// writes no request line (a poll every few seconds would drown the log); the guard's refusals still do.
async fn stats(State(app): State<App>) -> Json<Value> {
    let copy = app.backend().latest_stats();
    Json(json!({
        "object": "stats",
        "uptime_s": app.uptime_s(),
        "state": lifecycle_name(app.backend().state()),
        "progress": app.backend().progress(),
        "age_s": copy.as_ref().map(|c| c.at.elapsed().as_secs_f64()),
        "log_write_failures": app.logger().write_failures(),
        "stats": copy.as_ref().map(|c| &c.stats),
    }))
}

async fn models(State(app): State<App>) -> Json<Value> {
    let id = &app.config().model_id;
    Json(json!({
        "object": "list", "data": [app.model_entry(id)], "has_more": false, "first_id": id, "last_id": id,
    }))
}

/// Any id is accepted and echoed.
async fn model(State(app): State<App>, id: Result<Path<String>, PathRejection>) -> Response {
    match id {
        Ok(Path(id)) => Json(app.model_entry(&id)).into_response(),
        // A percent-encoded id that is not UTF-8 is axum's plain-text rejection otherwise.
        Err(_) => ApiError::invalid("id", "the model id is not valid UTF-8")
            .into_response(Flavour::OpenAi),
    }
}

async fn not_found(State(app): State<App>, req: Request) -> Response {
    let flavour = Flavour::of(req.uri().path());
    let e = ApiError::new(404, ErrorType::NotFound, "no such endpoint");
    app.log_refusal(req.uri().path(), None, &e, flavour);
    e.into_response(flavour)
}

async fn method_not_allowed(State(app): State<App>, req: Request) -> Response {
    let flavour = Flavour::of(req.uri().path());
    let e = ApiError::new(
        405,
        ErrorType::InvalidRequest,
        "method not allowed for this endpoint",
    );
    app.log_refusal(req.uri().path(), None, &e, flavour);
    e.into_response(flavour)
}

/// An app over a fake engine for the unit tests: its `halt` records the codes it is called with. Returns the logger and
/// the log directory too (which the caller removes).
#[cfg(test)]
pub(crate) fn test_app(halts: Arc<Mutex<Vec<i32>>>) -> (Logger, App, std::path::PathBuf) {
    test_app_tuned(halts, |_| {})
}

/// `test_app` with `tune` applied to the default tuning.
#[cfg(test)]
pub(crate) fn test_app_tuned(
    halts: Arc<Mutex<Vec<i32>>>,
    tune: impl FnOnce(&mut Tuning),
) -> (Logger, App, std::path::PathBuf) {
    use crate::fake::{FakeBackend, FakeConfig};
    use qwen36::tokenizer::Vocab;
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/");
    let b = std::fs::read(format!("{data}tokenizer/qwen36-vocab.bin")).unwrap();
    let v = Vocab::from_extract(&b).unwrap();
    let tk = Arc::new(Tokenizer::new(&v).unwrap());
    let sp = Specials::new(&v, &tk).unwrap();
    let src =
        std::fs::read_to_string(format!("{data}template/qwen36-chat-template.jinja")).unwrap();
    let n = N.fetch_add(1, Ordering::Relaxed) as usize;
    let dir = std::env::temp_dir().join(format!("artemis-app-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let logger = Logger::new(dir.clone(), 0).unwrap();
    let mut tuning = Tuning {
        halt: Arc::new(move |code| halts.lock().unwrap().push(code)),
        ..Tuning::default()
    };
    tune(&mut tuning);
    let app = App::new(AppParts {
        config: Config {
            model: Some("m.gguf".into()),
            ctx: 4096,
            ..Config::default()
        },
        key: None,
        backend: FakeBackend::new(FakeConfig::default(), tk.clone(), sp.clone()),
        tokenizer: tk,
        specials: sp,
        template: Template::new(src).unwrap(),
        logger: logger.clone(),
        tuning,
        version: "test",
    })
    .unwrap();
    (logger, app, dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A tracing writer that keeps what it is given.
    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<u8>>>);

    impl Write for Captured {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// `tracing` goes to the returned buffer until the guard drops (the test runtime is single-threaded, so the spawned
    /// watchers log on this thread).
    fn captured() -> (Captured, tracing::subscriber::DefaultGuard) {
        let out = Captured::default();
        let writer = out.clone();
        let sub = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        (out, tracing::subscriber::set_default(sub))
    }

    fn text(out: &Captured) -> String {
        String::from_utf8_lossy(&out.0.lock().unwrap()).into_owned()
    }

    #[tokio::test]
    async fn prep_permits_bound_the_preparation() {
        let wait = Duration::from_millis(100);
        let (_logger, app, dir) =
            test_app_tuned(Arc::new(Mutex::new(Vec::new())), |t| t.prep_wait = wait);
        let mut held = Vec::new();
        for _ in 0..PREP_PERMITS {
            held.push(app.prep_permit().await.unwrap());
        }
        let started = Instant::now();
        let e = app.prep_permit().await.unwrap_err();
        let waited = started.elapsed();
        assert_eq!((e.status, e.kind), (503, ErrorType::Unavailable));
        assert_eq!(
            e.message,
            "the server is busy preparing other requests; try again"
        );
        assert!(
            waited >= Duration::from_millis(100) && waited < Duration::from_secs(1),
            "{waited:?}"
        );
        // A permit given up is the next request's, at once.
        held.pop();
        let started = Instant::now();
        let permit = app.prep_permit().await.unwrap();
        assert!(started.elapsed() < Duration::from_millis(50));
        drop(permit);
        drop(held);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A client that leaves mid-build drops the handler's future, but the blocking build goes on: its slot stays taken
    /// until the build ends.
    #[tokio::test]
    async fn a_prep_permit_outlives_an_abandoned_build() {
        let (_logger, app, dir) = test_app(Arc::new(Mutex::new(Vec::new())));
        let (begun_tx, begun_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let permit = app.prep_permit().await.unwrap();
        assert_eq!(app.inner.prep.available_permits(), PREP_PERMITS - 1);
        let handler = tokio::spawn(crate::prompt::run_holding(permit, move || {
            begun_tx.send(()).unwrap();
            go_rx.recv().unwrap();
        }));
        tokio::task::spawn_blocking(move || begun_rx.recv().unwrap())
            .await
            .unwrap();
        // The client leaves: the handler's future is dropped while the build runs.
        handler.abort();
        let _ = handler.await;
        assert_eq!(app.inner.prep.available_permits(), PREP_PERMITS - 1);
        go_tx.send(()).unwrap();
        let until = Instant::now() + Duration::from_secs(5);
        while app.inner.prep.available_permits() != PREP_PERMITS {
            assert!(
                Instant::now() < until,
                "the slot was not given back when the build ended"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_task_that_dies_before_the_stop_is_logged() {
        let (out, _guard) = captured();
        let (_logger, app, dir) = test_app(Arc::new(Mutex::new(Vec::new())));
        let (_end, ended) = keep_watch(&app, "heartbeat", tokio::spawn(async {}));
        ended.await.unwrap();
        let (_end, panicked) = keep_watch(
            &app,
            "watchdog",
            tokio::spawn(async { panic!("the test's own") }),
        );
        panicked.await.unwrap();
        let log = text(&out);
        assert!(
            log.contains("the heartbeat task ended before the server was stopped"),
            "{log}"
        );
        assert!(log.contains("the watchdog task panicked"), "{log}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_task_that_serve_ends_says_nothing() {
        let (out, _guard) = captured();
        let (_logger, app, dir) = test_app(Arc::new(Mutex::new(Vec::new())));
        let (end, watcher) = keep_watch(&app, "heartbeat", tokio::spawn(std::future::pending()));
        end.abort();
        watcher.await.unwrap();
        // And the watchdog's return after it halts the process is not a loss.
        assert!(app.begin_halt());
        let (_end, halted) = keep_watch(&app, "watchdog", tokio::spawn(async {}));
        halted.await.unwrap();
        assert_eq!(text(&out), "");
        let _ = std::fs::remove_dir_all(dir);
    }
}
