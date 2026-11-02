//! One generation as the HTTP side sees it: take a place in the queue, submit, wait for `Started`, read
//! the events. `run_whole` drives them through a `Pipeline` to the end for the non-streaming endpoints.
use crate::app::{App, Exit, InFlight};
use crate::backend::{
    CacheReport, CancelHandle, Counts, ErrorKind, Finish, GenEvent, GenRequest, Sink, Source,
    SpecReport, Timings,
};
use crate::error::{ApiError, ErrorType};
use crate::pipeline::{Out, Pipeline, Summary};
use std::time::Duration;
use tokio::sync::mpsc::UnboundedReceiver;

/// What `Started` said.
#[derive(Clone, Debug, PartialEq)]
pub struct Started {
    pub prompt_tokens: u32,
    pub reused_tokens: u32,
    pub restored: Source,
    pub diverged_at: u32,
}

/// A request the engine has taken: its events, and the means to stop it. Dropping it cancels the request, so a handler
/// that is dropped (the client left) stops the engine within a token.
pub struct Exchange {
    pub started: Started,
    app: App,
    rx: UnboundedReceiver<GenEvent>,
    cancel: CancelHandle,
    /// Whether logprobs were asked for.
    logprobs: bool,
    /// `Done` or `Error` has been read: the channel closing after that is the engine's ordinary end.
    ended: bool,
    /// The request stays in the in-flight registry until the engine reports its end (`next` reads `Done` or an error, or
    /// finds the channel closed), or the exchange is dropped: a stream that has ended but whose client has stopped reading
    /// is not in flight, and giving up the place is what refreshes the stats copy.
    in_flight: Option<InFlight>,
}

/// 503, in the API's `service_unavailable` / `overloaded_error` kind.
fn unavailable(message: &str) -> ApiError {
    ApiError::new(503, ErrorType::Unavailable, message)
}

/// 503 once `serve` is stopping.
pub(crate) fn shutting_down() -> ApiError {
    unavailable("the server is shutting down")
}

/// The engine could not continue. A lost device and a failed load are 503 (the client may retry), the rest
/// 500. The message is the engine's, which never carries request text.
pub fn engine_error(kind: ErrorKind, message: &str) -> ApiError {
    match kind {
        ErrorKind::DeviceLost | ErrorKind::LoadFailed => {
            ApiError::new(503, ErrorType::Unavailable, message)
        }
        ErrorKind::NonFinite | ErrorKind::OutOfMemory | ErrorKind::Internal => {
            ApiError::new(500, ErrorType::Server, message)
        }
    }
}

impl Exchange {
    /// `id` is the request line's id: the request is in flight (`App::oldest_in_flight`) from here until the exchange
    /// it ends in is dropped. Enters the queue, submits `req` and waits for the engine to take it. Ends with 503 when
    /// the queue is full, the wait passes `queue_timeout_s` (the engine will skip the request), or an unload takes it;
    /// with the engine's own error, or 503 when the request was ended `Unloaded` before it started; with 500 when the
    /// engine is gone (`submit` answered `Full`, or the channel closed; `serve` is asked to exit).
    pub async fn begin(app: &App, id: &str, req: GenRequest) -> Result<Exchange, ApiError> {
        // First, so every early return below gives the place up.
        let mut in_flight = app.enter_flight(id);
        let logprobs = req.logprobs.is_some();
        let (sink, mut rx, cancel) = Sink::channel();
        // Neither call waits, so holding the lock keeps the engine's order the queue's.
        let ticket = {
            let _order = app.submit_order();
            if app.is_stopping() {
                return Err(shutting_down());
            }
            let ticket = app
                .queue()
                .enter(cancel.clone())
                .map_err(|_| unavailable("the queue is full"))?;
            // `Full` means the worker's thread has ended, as a closed event channel does: the process is to exit.
            app.backend().submit(req, sink).map_err(|_| gone(app))?;
            in_flight.submitted();
            ticket
        };
        let timeout = tokio::time::sleep(Duration::from_secs(app.config().queue_timeout_s));
        let first = tokio::select! {
            biased;
            () = ticket.aborted() => {
                cancel.cancel();
                return Err(unavailable("the model was unloaded"));
            }
            ev = rx.recv() => ev,
            () = timeout => {
                cancel.cancel();
                return Err(unavailable("timed out waiting in the queue"));
            }
        };
        match first {
            Some(GenEvent::Started {
                prompt_tokens,
                reused_tokens,
                restored,
                diverged_at,
            }) => {
                ticket.started();
                Ok(Exchange {
                    started: Started {
                        prompt_tokens,
                        reused_tokens,
                        restored,
                        diverged_at,
                    },
                    app: app.clone(),
                    rx,
                    cancel,
                    logprobs,
                    ended: false,
                    in_flight: Some(in_flight),
                })
            }
            Some(GenEvent::Error { kind, message }) => {
                if kind == ErrorKind::DeviceLost {
                    app.device_lost(&message);
                }
                Err(engine_error(kind, &message))
            }
            // The engine ended the request without starting it: the model was unloaded under it.
            Some(GenEvent::Done { .. }) => Err(unavailable("the model was unloaded")),
            Some(GenEvent::Token { .. }) => Err(ApiError::new(
                500,
                ErrorType::Server,
                "the engine sent output before it started the request",
            )),
            None => Err(lost(app, &cancel)),
        }
    }

    /// The next event; `None` once the channel has ended. A channel that ends before `Done` or `Error` means the
    /// engine's worker is gone: `serve` is asked to exit with `Exit::EngineGone`.
    pub async fn next(&mut self) -> Option<GenEvent> {
        let ev = self.rx.recv().await;
        match &ev {
            Some(GenEvent::Done { .. }) => self.ended = true,
            Some(GenEvent::Error { kind, message }) => {
                self.ended = true;
                if *kind == ErrorKind::DeviceLost {
                    self.app.device_lost(message);
                }
            }
            Some(_) => {}
            None if !self.ended => {
                self.ended = true;
                let _ = gone(&self.app);
            }
            None => {}
        }
        if self.ended {
            self.in_flight = None;
        }
        ev
    }

    /// Asks the engine to stop the request; the events up to `Done` still arrive.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

impl Drop for Exchange {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// The channel ended before any event. If the request was cancelled (an unload took it, and the engine met that by
/// skipping it and dropping its sink) that is the unload; otherwise the worker is gone.
fn lost(app: &App, cancel: &CancelHandle) -> ApiError {
    if cancel.is_cancelled() {
        unavailable("the model was unloaded")
    } else {
        gone(app)
    }
}

/// The engine's worker is gone: 500, and the process is to exit.
pub(crate) fn gone(app: &App) -> ApiError {
    app.exit(Exit::EngineGone);
    ApiError::new(500, ErrorType::Server, "the engine stopped")
}

/// A finished generation, as the pipeline made it.
#[derive(Clone, Debug, PartialEq)]
pub struct Generation {
    pub outs: Vec<Out>,
    pub finish: Finish,
    pub counts: Counts,
    /// What the engine measured, for the request's log line.
    pub timings: Timings,
    pub cache: CacheReport,
    /// Speculation's counts for the request; `None` with speculation off.
    pub spec: Option<SpecReport>,
    pub summary: Summary,
    /// Whether the request asked for logprobs (a reply that asked and has no content tokens still carries the field).
    pub logprobs: bool,
}

/// What the response reports as usage: the server's own reasoning count, the engine's others.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usage {
    pub prompt: u32,
    pub completion: u32,
    pub reused: u32,
    pub reasoning: u32,
}

/// Why the reply ended, in the APIs' common terms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ending {
    Stop,
    StopString(String),
    Length,
    ToolCalls,
}

/// The rule: a stop string or a stop token ends with `Stop` (`StopString` when a stop string did), the token
/// budget or a full context with `Length` (a call cut off mid-way included), and at least one valid call with
/// `ToolCalls`, a stop string after the call included.
pub fn ending(finish: &Finish, s: &Summary) -> Ending {
    match (&s.stopped, finish) {
        (Some(_), _) if s.calls > 0 => Ending::ToolCalls,
        (Some(m), _) => Ending::StopString(m.clone()),
        (None, Finish::Length | Finish::ContextFull) => Ending::Length,
        (None, _) if s.calls > 0 => Ending::ToolCalls,
        (None, _) => Ending::Stop,
    }
}

/// The usage a finished request reports.
pub fn usage_of(counts: &Counts, s: &Summary) -> Usage {
    Usage {
        prompt: counts.prompt,
        completion: counts.completion,
        reused: counts.reused,
        reasoning: s.reasoning_tokens,
    }
}

impl Generation {
    pub fn usage(&self) -> Usage {
        usage_of(&self.counts, &self.summary)
    }

    pub fn ending(&self) -> Ending {
        ending(&self.finish, &self.summary)
    }
}

/// What one engine event comes to, through the pipeline. `Done` is once per request, so boxing its parts would only
/// change every pattern that reads it.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Step {
    /// Output so far (empty for `Started`); the request goes on.
    Pieces(Vec<Out>),
    /// The request ended as the engine finished it: `tail` is what the pipeline held back, `summary` what it saw.
    Done {
        finish: Finish,
        counts: Counts,
        timings: Timings,
        cache: CacheReport,
        spec: Option<SpecReport>,
        summary: Summary,
        tail: Vec<Out>,
    },
    /// It cannot go on: an unload took it, the engine failed, or the engine is gone (`ev` was `None`).
    Fail(ApiError),
}

/// The one place an event is read, for the whole and the streamed replies of both APIs. A token goes through `p`; a
/// stop-string match cancels the engine, and the events up to `Done` are still read. An unload (`Finish::Unloaded`) is
/// 503 unless a stop string had matched (the reply is then complete), an engine error its status, and a channel that
/// ended (`None`) 500.
pub fn step(ex: &Exchange, p: &mut Pipeline, ev: Option<GenEvent>) -> Step {
    match ev {
        Some(GenEvent::Started { .. }) => Step::Pieces(vec![]),
        Some(GenEvent::Token {
            id,
            forced,
            logprobs,
        }) => {
            if forced {
                p.note_forced();
            }
            let pieces = p.push(id, logprobs.as_ref());
            if pieces.iter().any(|o| matches!(o, Out::Stopped { .. })) {
                ex.cancel();
            }
            Step::Pieces(pieces)
        }
        Some(GenEvent::Done {
            finish,
            counts,
            timings,
            cache,
            spec,
        }) => {
            let tail = p.finish(&finish);
            let summary = p.summary();
            // A matched stop string means the reply is complete, whether the engine ended it `Cancelled` (our cancel)
            // or `Unloaded` (an unload landed in the same token). Without a match, either end is the unload's.
            if matches!(finish, Finish::Cancelled | Finish::Unloaded) && summary.stopped.is_none() {
                return Step::Fail(unavailable("the model was unloaded"));
            }
            Step::Done {
                finish,
                counts,
                timings,
                cache,
                spec,
                summary,
                tail,
            }
        }
        Some(GenEvent::Error { kind, message }) => Step::Fail(engine_error(kind, &message)),
        None => Step::Fail(ApiError::new(500, ErrorType::Server, "the engine stopped")),
    }
}

/// Reads `ex` to its `Done`, through `p`, for the non-streaming endpoints.
pub async fn run_whole(mut ex: Exchange, mut p: Pipeline) -> Result<Generation, ApiError> {
    let mut outs = Vec::new();
    loop {
        let ev = ex.next().await;
        match step(&ex, &mut p, ev) {
            Step::Pieces(pieces) => outs.extend(pieces),
            Step::Done {
                finish,
                counts,
                timings,
                cache,
                spec,
                summary,
                tail,
            } => {
                outs.extend(tail);
                return Ok(Generation {
                    outs,
                    finish,
                    counts,
                    timings,
                    cache,
                    spec,
                    summary,
                    logprobs: ex.logprobs,
                });
            }
            Step::Fail(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{AppParts, Tuning};
    use crate::backend::{Backend, BackendStats, Full, Lifecycle, Reply, Sampling};
    use crate::config::Config;
    use crate::logs::Logger;
    use crate::template::Template;
    use crate::tokens::Specials;
    use qwen36::tokenizer::{Tokenizer, Vocab};
    use std::sync::{Arc, OnceLock};

    fn summary(calls: usize, stopped: Option<&str>) -> Summary {
        Summary {
            calls,
            malformed_calls: 0,
            budget_closed: false,
            reasoning_tokens: 0,
            repairs: vec![],
            stopped: stopped.map(String::from),
        }
    }

    #[test]
    fn the_ending_follows_spec_6_8() {
        let stop = Finish::Stop(1);
        assert_eq!(ending(&stop, &summary(0, None)), Ending::Stop);
        assert_eq!(ending(&stop, &summary(2, None)), Ending::ToolCalls);
        assert_eq!(ending(&Finish::Length, &summary(0, None)), Ending::Length);
        assert_eq!(
            ending(&Finish::ContextFull, &summary(0, None)),
            Ending::Length
        );
        // A call cut off mid-way is still `length`.
        assert_eq!(ending(&Finish::Length, &summary(1, None)), Ending::Length);
        // A stop string: `stop`, with the string, unless a valid call came first.
        let cancelled = Finish::Cancelled;
        assert_eq!(
            ending(&cancelled, &summary(0, Some("END"))),
            Ending::StopString("END".into())
        );
        assert_eq!(
            ending(&cancelled, &summary(1, Some("END"))),
            Ending::ToolCalls
        );
        // The match and the length limit on the same token: the match wins.
        assert_eq!(
            ending(&Finish::Length, &summary(0, Some("END"))),
            Ending::StopString("END".into())
        );
    }

    /// An engine that, on `submit`, runs a hook (an unload landing) and drops the sink at once, as a worker that pops a
    /// cancelled request does.
    struct Skipper {
        hook: Box<dyn Fn() + Send + Sync>,
    }

    fn ready<T: Send + 'static>(v: T) -> Reply<T> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = tx.send(Ok(v));
        rx
    }

    impl Backend for Skipper {
        fn submit(&self, _req: GenRequest, sink: Sink) -> Result<(), Full> {
            (self.hook)();
            drop(sink);
            Ok(())
        }
        fn state(&self) -> Lifecycle {
            Lifecycle::Ready
        }
        fn load(&self) -> Reply<()> {
            ready(())
        }
        fn unload(&self) -> Reply<bool> {
            ready(false)
        }
        fn stats(&self) -> Reply<BackendStats> {
            ready(BackendStats::default())
        }
    }

    /// A log directory under the temp dir, unique per process and call, removed when it drops.
    struct Scratch(std::path::PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The log directory and the app over `backend` that writes to it. Bind them in this order (`let (_logs, app)`):
    /// the app then drops first and the directory is removed after the logger's file is closed.
    fn app_over(backend: Arc<dyn Backend>) -> (Scratch, App) {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let data = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/");
        let b = std::fs::read(format!("{data}tokenizer/qwen36-vocab.bin")).unwrap();
        let v = Vocab::from_extract(&b).unwrap();
        let tk = Tokenizer::new(&v).unwrap();
        let sp = Specials::new(&v, &tk).unwrap();
        let src =
            std::fs::read_to_string(format!("{data}template/qwen36-chat-template.jinja")).unwrap();
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = Scratch(
            std::env::temp_dir().join(format!("artemis-exchange-{}-{n}", std::process::id())),
        );
        let app = App::new(AppParts {
            config: Config {
                model: Some("m.gguf".into()),
                ctx: 4096,
                ..Config::default()
            },
            key: None,
            backend,
            tokenizer: Arc::new(tk),
            specials: sp,
            template: Template::new(src).unwrap(),
            logger: Logger::new(dir.0.clone(), 0).unwrap(),
            tuning: Tuning::default(),
            version: "test",
        })
        .unwrap();
        (dir, app)
    }

    fn request() -> GenRequest {
        GenRequest {
            prompt: vec![1, 2, 3],
            cut: None,
            sampling: Sampling {
                temp: 0.0,
                top_k: 1,
                top_p: 1.0,
                min_p: 0.0,
                presence_penalty: 0.0,
            },
            seed: 0,
            max_tokens: 8,
            stop_tokens: vec![],
            think: None,
            logprobs: None,
        }
    }

    #[tokio::test]
    async fn an_unload_during_submit_is_503_and_not_a_lost_engine() {
        let slot: Arc<OnceLock<App>> = Arc::new(OnceLock::new());
        let hook = {
            let slot = slot.clone();
            // The unload lands between the submit and the first event; the engine drops the sink of the cancelled
            // request before the handler looks.
            move || {
                assert_eq!(slot.get().unwrap().queue().abort_all(), 1);
            }
        };
        let (_logs, app) = app_over(Arc::new(Skipper {
            hook: Box::new(hook),
        }));
        slot.set(app.clone()).ok();
        let Err(e) = Exchange::begin(&app, "req_test", request()).await else {
            panic!("the request should not have started");
        };
        assert_eq!(
            (e.status, e.message.as_str()),
            (503, "the model was unloaded")
        );
        assert_eq!(app.exited(), None, "an unload is not a lost engine");
        assert_eq!(app.queue().waiting(), 0);
        assert_eq!(
            app.oldest_in_flight(),
            None,
            "an early return gives the place up"
        );
    }

    #[tokio::test]
    async fn a_closed_channel_is_the_unload_when_cancelled_and_a_lost_engine_when_not() {
        let (_logs, app) = app_over(Arc::new(Skipper {
            hook: Box::new(|| {}),
        }));
        let (_sink, _rx, cancel) = Sink::channel();
        cancel.cancel();
        let e = lost(&app, &cancel);
        assert_eq!(
            (e.status, e.message.as_str()),
            (503, "the model was unloaded")
        );
        assert_eq!(app.exited(), None);
        let (_sink, _rx, live) = Sink::channel();
        let e = lost(&app, &live);
        assert_eq!(e.status, 500);
        assert_eq!(app.exited(), Some(Exit::EngineGone));
        // And with nothing cancelling, `begin` itself reports the engine gone.
        let Err(e) = Exchange::begin(&app, "req_test", request()).await else {
            panic!("the request should not have started");
        };
        assert_eq!(e.status, 500);
        assert_eq!(app.oldest_in_flight(), None);
    }

    /// An engine whose worker has ended: `submit` answers `Err(Full)`.
    struct Vanished;

    impl Backend for Vanished {
        fn submit(&self, _req: GenRequest, _sink: Sink) -> Result<(), Full> {
            Err(Full)
        }
        fn state(&self) -> Lifecycle {
            Lifecycle::Ready
        }
        fn load(&self) -> Reply<()> {
            ready(())
        }
        fn unload(&self) -> Reply<bool> {
            ready(false)
        }
        fn stats(&self) -> Reply<BackendStats> {
            ready(BackendStats::default())
        }
    }

    #[tokio::test]
    async fn a_submit_that_finds_the_worker_gone_is_500_and_exits() {
        let (_logs, app) = app_over(Arc::new(Vanished));
        let Err(e) = Exchange::begin(&app, "req_test", request()).await else {
            panic!("the request should not have started");
        };
        assert_eq!((e.status, e.message.as_str()), (500, "the engine stopped"));
        assert_eq!(app.exited(), Some(Exit::EngineGone));
        assert_eq!(
            app.queue().waiting(),
            0,
            "the place in the queue was given back"
        );
        assert_eq!(app.oldest_in_flight(), None);
    }

    /// An engine that takes the request and says `Started`, and nothing more.
    struct Starter;

    impl Backend for Starter {
        fn submit(&self, _req: GenRequest, sink: Sink) -> Result<(), Full> {
            sink.send(GenEvent::Started {
                prompt_tokens: 3,
                reused_tokens: 0,
                restored: Source::None,
                diverged_at: 0,
            });
            Ok(())
        }
        fn state(&self) -> Lifecycle {
            Lifecycle::Ready
        }
        fn load(&self) -> Reply<()> {
            ready(())
        }
        fn unload(&self) -> Reply<bool> {
            ready(false)
        }
        fn stats(&self) -> Reply<BackendStats> {
            ready(BackendStats::default())
        }
    }

    fn done(finish: Finish) -> GenEvent {
        GenEvent::Done {
            finish,
            counts: Counts {
                prompt: 3,
                reused: 0,
                prefilled: 3,
                completion: 1,
                reasoning: 0,
            },
            timings: Timings {
                queue_ms: 0.0,
                load_ms: 0.0,
                restore_ms: 0.0,
                prefill_ms: 0.0,
                decode_ms: 0.0,
                ttft_ms: 0.0,
            },
            cache: CacheReport {
                hit_rate: None,
                miss_mib_per_token: None,
                restore_bytes: 0,
                pool_bytes_used: 0,
                decode_misses: None,
            },
            spec: None,
        }
    }

    /// An engine that answers a request in full at once, and counts the `stats` it is asked for.
    struct Ender(std::sync::atomic::AtomicUsize);

    impl Backend for Ender {
        fn submit(&self, _req: GenRequest, sink: Sink) -> Result<(), Full> {
            sink.send(GenEvent::Started {
                prompt_tokens: 3,
                reused_tokens: 0,
                restored: Source::None,
                diverged_at: 0,
            });
            sink.send(done(Finish::Length));
            Ok(())
        }
        fn state(&self) -> Lifecycle {
            Lifecycle::Ready
        }
        fn load(&self) -> Reply<()> {
            ready(())
        }
        fn unload(&self) -> Reply<bool> {
            ready(false)
        }
        fn stats(&self) -> Reply<BackendStats> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            ready(BackendStats::default())
        }
    }

    #[tokio::test]
    async fn a_request_the_engine_has_ended_is_no_longer_in_flight_and_refreshes_the_stats_once() {
        let ender = Arc::new(Ender(std::sync::atomic::AtomicUsize::new(0)));
        let (_logs, app) = app_over(ender.clone());
        let asked = || ender.0.load(std::sync::atomic::Ordering::Acquire);
        let mut ex = Exchange::begin(&app, "req_test", request()).await.unwrap();
        assert_eq!(app.oldest_in_flight().as_deref(), Some("req_test"));
        assert_eq!(asked(), 0, "nothing is refreshed while the request runs");
        // The engine's `Done` is read, but the exchange lives on: a stream still has its last frames to send, to a client
        // that may have stopped reading. The request is over for the engine, so it is out of the registry.
        assert!(matches!(ex.next().await, Some(GenEvent::Done { .. })));
        assert_eq!(app.oldest_in_flight(), None);
        assert_eq!(asked(), 1, "the copy is refreshed when the request ends");
        drop(ex);
        assert_eq!(asked(), 1, "and only once");
    }

    #[tokio::test]
    async fn a_request_refused_before_the_engine_saw_it_refreshes_nothing() {
        let ender = Arc::new(Ender(std::sync::atomic::AtomicUsize::new(0)));
        let (_logs, app) = app_over(ender.clone());
        app.begin_stopping();
        let Err(e) = Exchange::begin(&app, "req_test", request()).await else {
            panic!("a stopping server takes no request");
        };
        assert_eq!(e.status, 503);
        assert_eq!(ender.0.load(std::sync::atomic::Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn an_unload_in_the_token_of_a_stop_match_leaves_the_reply_complete() {
        let (_logs, app) = app_over(Arc::new(Starter));
        let ex = Exchange::begin(&app, "req_test", request()).await.unwrap();
        // The place moved into the exchange, and goes with it; the oldest request is named first.
        assert_eq!(app.oldest_in_flight().as_deref(), Some("req_test"));
        let second = app.enter_flight("req_later");
        assert_eq!(app.oldest_in_flight().as_deref(), Some("req_test"));
        let mk = |stop: &[&str]| {
            Pipeline::new(
                crate::pipeline::PipelineSetup {
                    thinking: false,
                    tools: None,
                    stop: stop.iter().map(|s| s.to_string()).collect(),
                },
                app.tokenizer().clone(),
                app.specials(),
            )
        };
        let x = app.tokenizer().encode("x", false)[0];
        // The stop string matched, then the engine ended the request `Unloaded`: the reply is whole.
        let mut p = mk(&["x"]);
        let Step::Pieces(pieces) = step(
            &ex,
            &mut p,
            Some(GenEvent::Token {
                id: x,
                forced: false,
                logprobs: None,
            }),
        ) else {
            panic!("a token is pieces");
        };
        assert!(pieces.iter().any(|o| matches!(o, Out::Stopped { .. })));
        match step(&ex, &mut p, Some(done(Finish::Unloaded))) {
            Step::Done { summary, .. } => assert_eq!(summary.stopped.as_deref(), Some("x")),
            other => panic!("a complete reply was failed: {other:?}"),
        }
        // Without a match, an unload is still 503.
        let mut p = mk(&["x"]);
        match step(&ex, &mut p, Some(done(Finish::Unloaded))) {
            Step::Fail(e) => assert_eq!(e.status, 503),
            other => panic!("expected 503, got {other:?}"),
        }
        drop(ex);
        assert_eq!(app.oldest_in_flight().as_deref(), Some("req_later"));
        drop(second);
        assert_eq!(app.oldest_in_flight(), None);
    }

    #[test]
    fn engine_errors_have_the_statuses_of_spec_8_3() {
        let status = |k| engine_error(k, "m").status;
        assert_eq!(status(ErrorKind::DeviceLost), 503);
        assert_eq!(status(ErrorKind::LoadFailed), 503);
        assert_eq!(status(ErrorKind::NonFinite), 500);
        assert_eq!(status(ErrorKind::OutOfMemory), 500);
        assert_eq!(status(ErrorKind::Internal), 500);
    }
}
