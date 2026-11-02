//! The adapter from the engine's worker to the server's `Backend`, over the engine's scripted
//! `FakeLoader`: the request path, the lifecycle, the device-loss counting and the shutdown.
//! Timing follows the server tests' rule: a cancel is asserted in tokens, with the fake at 100 ms a token through a
//! hook on the worker's own thread, and no test sleeps a tokio thread.
mod support;

use artemis_tools::serve::backend::EngineBackend;
use artemis_tools::test7::http::{Client, SseItem};
use engine::fake::{Call, FakeLive, FakeLoader, LoaderLog, VOCAB};
use engine::{LoadFailure, SpecConfig, Worker, WorkerParams};
use server::app::{Exit, ManualClock, Tuning};
use server::backend::{
    Backend, BackendError, BackendStats, CancelHandle, Counts, ErrorKind, Finish, Full, GenEvent,
    GenRequest, Lifecycle, Progress, Reply, Sampling, Sink, Source,
};
use server::config::Config;
use server::logs::{LifecycleLine, Logger};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use support::{
    events, lifecycle_lines, parts, scratch_dir, start, tokenizer, vocab_size, wait_until, Running,
};
use tokio::sync::mpsc::UnboundedReceiver;

const WAIT: Duration = Duration::from_secs(10);
const CTX: usize = 512;
/// The context of the tests that send a prompt of several prefill segments (the engine cuts one at every 4,096 tokens).
const LONG_CTX: usize = 20_480;
const PER_TOKEN: Duration = Duration::from_millis(100);
/// How long an idle unload's park holds the worker, from 300 ms after the load: wide enough that a stalled test
/// finds the park over and fails its check (`assert_in_the_park`) instead of passing without testing the race.
const PARK: Duration = Duration::from_millis(1500);

type Rx = UnboundedReceiver<GenEvent>;

/// A scripted device loss, as the engine's own tests build one.
fn lost() -> engine::Error {
    engine::Error::Vk(vk::Error::DeviceLost { in_flight: None })
}

/// The model picks 7 whatever the history.
fn sevens() -> FakeLoader {
    FakeLoader::new(CTX, |_| 7)
}

/// Makes each decode of `live` take `PER_TOKEN`, on the worker's thread.
fn slow(live: &mut FakeLive) {
    live.on_call(|c| {
        if matches!(c, Call::Decode(_)) {
            std::thread::sleep(PER_TOKEN);
        }
    });
}

fn params(idle_unload: Option<Duration>, notify: engine::Notify) -> WorkerParams {
    WorkerParams {
        idle_unload,
        park_snapshots: true,
        ctx: CTX,
        n_vocab: VOCAB,
        notify: Some(notify),
        speculation: None,
    }
}

/// An adapter over a fake engine, its log directory and a loss hook that counts its calls.
struct Rig {
    backend: Arc<EngineBackend>,
    log: Arc<LoaderLog>,
    dir: PathBuf,
    hooked: Arc<AtomicUsize>,
}

impl Rig {
    fn new(loader: FakeLoader, idle_unload: Option<Duration>) -> Rig {
        Rig::build(loader, idle_unload, false, CTX)
    }

    /// A rig whose worker takes `LONG_CTX` (see `long_ctx_parts`).
    fn with_long_ctx(loader: FakeLoader) -> Rig {
        Rig::build(loader, None, false, LONG_CTX)
    }

    /// A rig whose loss hook writes the `device_lost` line, as the server's does.
    fn logging_losses(loader: FakeLoader) -> Rig {
        Rig::build(loader, None, true, CTX)
    }

    fn build(
        loader: FakeLoader,
        idle_unload: Option<Duration>,
        log_losses: bool,
        ctx: usize,
    ) -> Rig {
        let log = loader.log();
        let dir = scratch_dir("backend");
        let logger = Logger::new(dir.clone(), 0).unwrap();
        let backend = EngineBackend::with_worker(
            |notify| {
                Worker::spawn_with(
                    loader,
                    WorkerParams {
                        ctx,
                        ..params(idle_unload, notify)
                    },
                )
            },
            logger.clone(),
        );
        let hooked = Arc::new(AtomicUsize::new(0));
        let h = hooked.clone();
        backend.set_loss_hook(Box::new(move |message| {
            if log_losses {
                logger.lifecycle(&LifecycleLine::DeviceLost {
                    message: message.to_string(),
                });
            }
            h.fetch_add(1, Ordering::SeqCst);
        }));
        Rig {
            backend: Arc::new(backend),
            log,
            dir,
            hooked,
        }
    }

    fn submit(&self, req: GenRequest) -> (Rx, CancelHandle) {
        let (sink, rx, cancel) = Sink::channel();
        self.backend.submit(req, sink).expect("the worker is there");
        (rx, cancel)
    }

    /// The lifecycle lines' events, oldest first.
    fn events(&self) -> Vec<String> {
        events(&self.dir)
    }

    /// Waits for the worker to answer a command sent after everything before it, so what a request did to the
    /// observer is done.
    async fn sync(&self) {
        reply(self.backend.stats()).await.unwrap();
    }

    /// Fails when the idle park is already over (the fake logs `Park` after its sleep, and the worker reads `Idle`
    /// only after the park): the test stalled past its window, and what follows would test nothing.
    fn assert_in_the_park(&self) {
        assert!(
            self.log.parks().is_empty() && self.backend.state() == Lifecycle::Ready,
            "the park ended first: the test stalled"
        );
    }

    /// `finish(5 s)` on a blocking thread, and how long it took.
    async fn finish(&self) -> Duration {
        self.finish_within(Duration::from_secs(5)).await
    }

    /// `finish(cap)` on a blocking thread, and how long it took.
    async fn finish_within(&self, cap: Duration) -> Duration {
        let b = self.backend.clone();
        tokio::task::spawn_blocking(move || {
            let t = Instant::now();
            b.finish(cap);
            t.elapsed()
        })
        .await
        .unwrap()
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

/// A greedy request of a 6-token prompt (ids under 64, as the fake's vocabulary needs), stopping on 63.
fn request(max_tokens: u32) -> GenRequest {
    GenRequest {
        prompt: (0..6).map(|i| i * 7).collect(),
        cut: None,
        sampling: Sampling {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            presence_penalty: 0.0,
        },
        seed: 1,
        max_tokens,
        stop_tokens: vec![63],
        think: None,
        logprobs: None,
    }
}

/// The loader and the server config for a prompt of several prefill segments: both take `LONG_CTX`, which the
/// file's `CTX` does not hold. The model picks 7 whatever the history.
fn long_ctx_parts() -> (FakeLoader, Config) {
    let cfg = Config {
        model: Some("m.gguf".into()),
        ctx: LONG_CTX as u32,
        ..Config::default()
    };
    (FakeLoader::new(LONG_CTX, |_| 7), cfg)
}

/// A reply that arrives, and is not a dropped one.
async fn reply<T>(r: Reply<T>) -> Result<T, BackendError> {
    tokio::time::timeout(WAIT, r)
        .await
        .expect("an answer within 10 s")
        .expect("the reply was not dropped")
}

async fn next(rx: &mut Rx) -> GenEvent {
    tokio::time::timeout(WAIT, rx.recv())
        .await
        .expect("an event within 10 s")
        .expect("the request's events end with Done or Error")
}

/// The events up to and including the `Done` or `Error`.
async fn until_end(rx: &mut Rx) -> Vec<GenEvent> {
    let mut events = Vec::new();
    loop {
        let e = next(rx).await;
        let end = matches!(e, GenEvent::Done { .. } | GenEvent::Error { .. });
        events.push(e);
        if end {
            return events;
        }
    }
}

fn finish_of(events: &[GenEvent]) -> Finish {
    match events.last() {
        Some(GenEvent::Done { finish, .. }) => *finish,
        other => panic!("not Done: {other:?}"),
    }
}

/// Reads tokens until `n` have come and returns what came with them.
async fn tokens(rx: &mut Rx, n: usize) -> Vec<GenEvent> {
    let mut seen = Vec::new();
    let mut count = 0;
    while count < n {
        let e = next(rx).await;
        assert!(
            !matches!(e, GenEvent::Done { .. } | GenEvent::Error { .. }),
            "the request ended after {count} tokens: {e:?}"
        );
        count += usize::from(matches!(e, GenEvent::Token { .. }));
        seen.push(e);
    }
    seen
}

fn token_count(events: &[GenEvent]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, GenEvent::Token { .. }))
        .count()
}

#[test]
fn the_backend_is_send_and_sync() {
    fn is<T: Send + Sync + 'static>() {}
    is::<EngineBackend>();
}

// ---- the request path ----

#[tokio::test]
async fn a_request_runs_and_ends_done() {
    // The prompt has 6 tokens: the model picks 7 twice, then 63, which is a stop token.
    let rig = Rig::new(
        FakeLoader::new(CTX, |h| if h.len() >= 8 { 63 } else { 7 }),
        None,
    );
    let (mut rx, _cancel) = rig.submit(request(10));
    let events = until_end(&mut rx).await;
    assert_eq!(
        events[0],
        GenEvent::Started {
            prompt_tokens: 6,
            reused_tokens: 0,
            restored: Source::None,
            diverged_at: 0,
        }
    );
    let ids: Vec<u32> = events
        .iter()
        .filter_map(|e| match e {
            GenEvent::Token {
                id,
                forced: false,
                logprobs: None,
            } => Some(*id),
            _ => None,
        })
        .collect();
    assert_eq!(ids, [7, 7]);
    assert_eq!(events.len(), 4);
    match events.last().unwrap() {
        GenEvent::Done {
            finish,
            counts,
            timings,
            cache,
            spec,
        } => {
            assert_eq!(*finish, Finish::Stop(63));
            // Speculation is off in this rig: no counts.
            assert_eq!(*spec, None);
            // The stop token is counted though it is never sent.
            assert_eq!(
                *counts,
                Counts {
                    prompt: 6,
                    reused: 0,
                    prefilled: 6,
                    completion: 3,
                    reasoning: 0,
                }
            );
            assert!(timings.ttft_ms >= 0.0 && timings.decode_ms >= 0.0);
            assert_eq!(cache.restore_bytes, 0);
        }
        other => panic!("not Done: {other:?}"),
    }
}

#[tokio::test]
async fn a_cancel_ends_cancelled_within_one_token() {
    let rig = Rig::new(sevens().tweak(|_, l| slow(l)), None);
    let (mut rx, cancel) = rig.submit(request(50));
    tokens(&mut rx, 3).await;
    cancel.cancel();
    let rest = until_end(&mut rx).await;
    assert!(token_count(&rest) <= 1, "{rest:?}");
    assert_eq!(finish_of(&rest), Finish::Cancelled);
}

#[tokio::test]
async fn submit_after_the_worker_is_gone_is_full() {
    let rig = Rig::new(sevens(), None);
    assert_eq!(reply(rig.backend.shutdown()).await, Ok(false));
    let (sink, _rx, _cancel) = Sink::channel();
    assert_eq!(rig.backend.submit(request(3), sink), Err(Full));
    // The unload is answered, not dropped: a dropped reply reads as an engine that is gone.
    assert_eq!(reply(rig.backend.unload()).await, Ok(false));
    assert_eq!(
        reply(rig.backend.stats()).await,
        Ok(BackendStats::default())
    );
    assert_eq!(rig.backend.state(), Lifecycle::Idle);
    rig.finish().await;
}

// ---- load and unload ----

#[tokio::test]
async fn load_answers_at_once_when_ready() {
    let rig = Rig::new(sevens().tweak(|_, l| slow(l)), None);
    let (mut rx, cancel) = rig.submit(request(30));
    assert!(matches!(next(&mut rx).await, GenEvent::Started { .. }));
    assert_eq!(rig.backend.state(), Lifecycle::Ready);
    // The worker answers a `Load` behind the running request; the adapter does not ask it.
    assert_eq!(reply(rig.backend.load()).await, Ok(()));
    while let Ok(e) = rx.try_recv() {
        assert!(
            !matches!(e, GenEvent::Done { .. }),
            "the load waited for the request's end"
        );
    }
    cancel.cancel();
    until_end(&mut rx).await;
}

#[tokio::test]
async fn load_waits_behind_a_pending_unload() {
    let rig = Rig::new(sevens().tweak(|_, l| slow(l)), None);
    let (mut rx, _cancel) = rig.submit(request(50));
    assert!(matches!(next(&mut rx).await, GenEvent::Started { .. }));
    let unload = rig.backend.unload();
    let load = rig.backend.load();
    // The model still reads `Ready` while the request winds down, but a load sent after an unload must not take the
    // fast path: it is the worker's to answer, after the unload and in order.
    assert_eq!(reply(load).await, Ok(()));
    let mut unload = unload;
    assert_eq!(
        unload.try_recv().expect("the unload answered first"),
        Ok(true)
    );
    assert_eq!(rig.log.loads().len(), 2, "the load loaded again");
    assert_eq!(rig.log.parks().len(), 1);
    assert_eq!(finish_of(&until_end(&mut rx).await), Finish::Unloaded);
}

#[tokio::test]
async fn unload_says_whether_it_unloaded() {
    let rig = Rig::new(sevens(), None);
    assert_eq!(reply(rig.backend.unload()).await, Ok(false));
    assert_eq!(reply(rig.backend.load()).await, Ok(()));
    assert_eq!(rig.backend.state(), Lifecycle::Ready);
    assert_eq!(reply(rig.backend.unload()).await, Ok(true));
    assert_eq!(rig.backend.state(), Lifecycle::Idle);
    assert_eq!(reply(rig.backend.unload()).await, Ok(false));
}

#[tokio::test]
async fn the_idle_unload_writes_one_line_and_its_stats() {
    let rig = Rig::new(sevens(), Some(Duration::from_millis(300)));
    assert_eq!(reply(rig.backend.load()).await, Ok(()));
    assert!(wait_until(WAIT, || rig.events().contains(&"stats".to_string())).await);
    // Nothing loads or unloads again: the model is gone and nothing asks for it.
    tokio::time::sleep(Duration::from_millis(700)).await;
    let lines = lifecycle_lines(&rig.dir);
    let names: Vec<_> = lines.iter().map(|l| l["event"].as_str().unwrap()).collect();
    assert_eq!(names, ["load", "unload", "stats"]);
    assert_eq!(lines[1]["cause"], "idle");
    assert_eq!(lines[0]["slots_per_layer"], 128);
    // The `stats` line is the adapter's own: its uptime from `with_worker`, and the copy it holds, none before any
    // `stats` answer.
    let stats = &lines[2];
    assert!(stats["uptime_s"].is_u64(), "{stats}");
    assert_eq!(stats["log_write_failures"], 0);
    assert!(
        stats["stats"].is_null() && stats["age_s"].is_null(),
        "{stats}"
    );
}

#[tokio::test]
async fn the_idle_unloads_stats_carry_the_latest_copy() {
    let rig = Rig::new(sevens(), Some(Duration::from_millis(400)));
    assert_eq!(reply(rig.backend.load()).await, Ok(()));
    rig.sync().await;
    assert!(wait_until(WAIT, || rig.events().contains(&"stats".to_string())).await);
    let lines = lifecycle_lines(&rig.dir);
    let stats = lines.iter().find(|l| l["event"] == "stats").unwrap();
    assert_eq!(stats["stats"]["loads"], 1, "{stats}");
    assert!(stats["age_s"].as_f64().unwrap() < 5.0, "{stats}");
}

#[tokio::test]
async fn a_request_during_the_idle_park_is_served_after_a_reload() {
    // The model unloads 300 ms after the load, and the park holds the worker for 500 ms from then.
    let rig = Rig::new(sevens().park_time(PARK), Some(Duration::from_millis(300)));
    assert_eq!(reply(rig.backend.load()).await, Ok(()));
    tokio::time::sleep(Duration::from_millis(400)).await;
    rig.assert_in_the_park();
    let (mut rx, _cancel) = rig.submit(request(3));
    let events = until_end(&mut rx).await;
    assert!(matches!(events[0], GenEvent::Started { .. }));
    assert_eq!(finish_of(&events), Finish::Length);
    // /health reads this state: the request ended with the model loaded again.
    assert_eq!(rig.backend.state(), Lifecycle::Ready);
    let lines = lifecycle_lines(&rig.dir);
    let names: Vec<_> = lines.iter().map(|l| l["event"].as_str().unwrap()).collect();
    assert_eq!(names[..4], ["load", "unload", "stats", "load"], "{names:?}");
    assert_eq!(lines[1]["cause"], "idle");
    assert_eq!(rig.log.parks().len(), 1);
}

#[tokio::test]
async fn a_failed_load_writes_load_failed() {
    let rig = Rig::new(
        sevens().script_loads(vec![Err(LoadFailure {
            error: engine::Error::WrongEngine,
            slots: None,
        })]),
        None,
    );
    assert!(reply(rig.backend.load()).await.is_err());
    rig.sync().await;
    let lines = lifecycle_lines(&rig.dir);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert_eq!(lines[0]["event"], "load_failed");
    assert_eq!(lines[0]["message"], engine::Error::WrongEngine.to_string());
}

#[tokio::test]
async fn a_failed_reload_writes_reload_failed() {
    let rig = Rig::logging_losses(
        sevens()
            .script_loads(vec![
                Ok(()),
                Err(LoadFailure {
                    error: engine::Error::WrongEngine,
                    slots: None,
                }),
            ])
            .tweak(|i, l| {
                if i == 0 {
                    // The reply-end snapshot loses the device after the reply, so the observer's hook counts it.
                    l.fail_snapshot = Some(lost());
                }
            }),
    );
    let (mut rx, _cancel) = rig.submit(request(3));
    assert_eq!(finish_of(&until_end(&mut rx).await), Finish::Length);
    rig.sync().await;
    assert_eq!(rig.hooked.load(Ordering::SeqCst), 1);
    let lines = lifecycle_lines(&rig.dir);
    let names: Vec<_> = lines.iter().map(|l| l["event"].as_str().unwrap()).collect();
    assert_eq!(names, ["load", "device_lost", "reload_failed"], "{names:?}");
    assert_eq!(lines[2]["message"], engine::Error::WrongEngine.to_string());
}

// ---- device loss: one source per loss ----

#[tokio::test]
async fn a_carried_loss_is_not_counted_again_by_the_observer() {
    let rig = Rig::new(
        sevens().tweak(|i, l| {
            if i == 0 {
                // The first decode fails.
                l.fail_at = Some((1, lost()));
            }
        }),
        None,
    );
    let (mut rx, _cancel) = rig.submit(request(10));
    let events = until_end(&mut rx).await;
    assert!(
        matches!(
            events.last(),
            Some(GenEvent::Error {
                kind: ErrorKind::DeviceLost,
                ..
            })
        ),
        "{events:?}"
    );
    rig.sync().await;
    assert_eq!(rig.hooked.load(Ordering::SeqCst), 0);
    // The worker reloaded after the loss.
    assert_eq!(rig.events(), ["load", "load"]);
}

#[tokio::test]
async fn a_carried_loss_leaves_no_credit_behind() {
    let rig = Rig::new(
        sevens().tweak(|i, l| match i {
            0 => l.fail_at = Some((1, lost())),
            // The reload's first reply-end budget check fails: a loss no request carries.
            1 => l.fail_budget = Some(lost()),
            _ => {}
        }),
        None,
    );
    let (mut rx, _cancel) = rig.submit(request(10));
    until_end(&mut rx).await;
    rig.sync().await;
    assert_eq!(rig.hooked.load(Ordering::SeqCst), 0);
    let (mut rx, _cancel) = rig.submit(request(3));
    assert_eq!(finish_of(&until_end(&mut rx).await), Finish::Length);
    rig.sync().await;
    assert_eq!(rig.hooked.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_loss_on_a_closed_sink_is_counted_by_the_hook() {
    // The receiver goes in the decode that fails: the hooks run before `fail_at` applies, so the step fails, `send`
    // finds the channel closed and nothing is carried. Dropped from this thread it would end the turn `Cancelled`
    // before that step.
    let slot: Arc<Mutex<Option<Rx>>> = Arc::new(Mutex::new(None));
    let s = slot.clone();
    let rig = Rig::new(
        sevens().tweak(move |i, l| {
            if i == 0 {
                l.fail_at = Some((1, lost()));
                let s = s.clone();
                l.on_call(move |c| {
                    if matches!(c, Call::Decode(_)) {
                        drop(s.lock().unwrap().take());
                    }
                });
            }
        }),
        None,
    );
    let (sink, rx, _cancel) = Sink::channel();
    *slot.lock().unwrap() = Some(rx);
    rig.backend.submit(request(10), sink).unwrap();
    rig.sync().await;
    assert!(slot.lock().unwrap().is_none(), "the decode ran");
    assert_eq!(rig.hooked.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_loss_after_a_reply_is_counted_by_the_hook() {
    let rig = Rig::new(
        sevens().tweak(|i, l| {
            if i == 0 {
                // The reply-end snapshot, after `Done`.
                l.fail_snapshot = Some(lost());
            }
        }),
        None,
    );
    let (mut rx, _cancel) = rig.submit(request(3));
    assert_eq!(finish_of(&until_end(&mut rx).await), Finish::Length);
    rig.sync().await;
    assert_eq!(rig.hooked.load(Ordering::SeqCst), 1);
    assert_eq!(rig.events(), ["load", "load"]);
}

#[tokio::test]
async fn a_load_loss_then_a_loss_after_a_reply_calls_the_hook_once() {
    let rig = Rig::new(
        sevens()
            .script_loads(vec![Err(LoadFailure {
                error: lost(),
                slots: None,
            })])
            .tweak(|i, l| {
                if i == 1 {
                    l.fail_snapshot = Some(lost());
                }
            }),
        None,
    );
    // A `/v1/load` that loses the device is counted by the server from its reply, never by the observer.
    let err = reply(rig.backend.load()).await.unwrap_err();
    assert_eq!(err.kind, ErrorKind::DeviceLost);
    rig.sync().await;
    assert_eq!(rig.hooked.load(Ordering::SeqCst), 0);
    // The next loss is a loss after a reply: the failed load left no credit behind to swallow it.
    let (mut rx, _cancel) = rig.submit(request(3));
    assert_eq!(finish_of(&until_end(&mut rx).await), Finish::Length);
    rig.sync().await;
    assert_eq!(rig.hooked.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_hooked_loss_and_a_carried_loss_within_the_window_exit() {
    // The worker takes the committed vocabulary's size and the config's context: the server's prompts hold ids far
    // past the fake's 64, and `Request::check` refuses them otherwise. The fake's logits stay 64 wide.
    let loader = FakeLoader::new(65_536, |_| 7).tweak(|i, l| match i {
        // The reply-end budget check fails: a loss after the reply. (The snapshot is no use here: the server's
        // request has a cut, and the prompt-end snapshot would take the failure inside the turn.)
        0 => l.fail_budget = Some(lost()),
        // The reload's first prefill fails: a loss the request carries.
        1 => l.fail_at = Some((0, lost())),
        _ => {}
    });
    let dir = scratch_dir("app");
    let backend = Arc::new(EngineBackend::with_worker(
        |notify| {
            Worker::spawn_with(
                loader,
                WorkerParams {
                    ctx: 65_536,
                    n_vocab: vocab_size(),
                    ..params(None, notify)
                },
            )
        },
        Logger::new(dir.clone(), 0).unwrap(),
    ));
    let clock = Arc::new(ManualClock::new());
    let tuning = Tuning {
        clock: clock.clone(),
        shutdown_grace: Duration::from_secs(5),
        ..Tuning::default()
    };
    let cfg = Config {
        model: Some("m.gguf".into()),
        ctx: 65_536,
        ..Config::default()
    };
    let mut running = start(parts(backend.clone(), cfg, None, &dir), tuning).await;
    let app = running.app.clone();
    backend.set_loss_hook(Box::new(move |m| app.device_lost(m)));
    let lines = |dir: &PathBuf| {
        lifecycle_lines(dir)
            .iter()
            .filter(|l| l["event"] == "device_lost")
            .count()
    };

    let body = br#"{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none","max_tokens":4}"#;
    let first = running
        .send("POST", "/v1/chat/completions", &[], body)
        .await;
    assert_eq!(
        first.status,
        200,
        "{}",
        String::from_utf8_lossy(&first.body)
    );
    // The loss comes after the reply, from the worker thread.
    assert!(wait_until(WAIT, || lines(&dir) == 1).await);
    assert_eq!(running.app.exited(), None);

    clock.advance(Duration::from_secs(60));
    let second = running
        .send("POST", "/v1/chat/completions", &[], body)
        .await;
    assert_eq!(second.status, 503);
    assert_eq!(lines(&dir), 2, "each loss is one line");
    assert_eq!(running.app.exited(), Some(Exit::DeviceLostTwice));
    assert_eq!(
        running.stop.join(WAIT).await,
        Some(Err(Exit::DeviceLostTwice))
    );
    let b = backend.clone();
    tokio::task::spawn_blocking(move || b.finish(Duration::from_secs(5)))
        .await
        .unwrap();
}

// ---- the progress seam ----

#[tokio::test]
async fn progress_counts_tokens_and_prefill_segments() {
    let (loader, cfg) = long_ctx_parts();
    assert_eq!(cfg.ctx as usize, LONG_CTX);
    let rig = Rig::with_long_ctx(loader.tweak(|_, l| slow(l)));
    assert_eq!(rig.backend.progress(), Some(Progress::default()));

    // Loaded first, so neither request's count holds the load's.
    assert_eq!(reply(rig.backend.load()).await, Ok(()));
    let ticks = || rig.backend.progress().unwrap().ticks;

    // A short prompt is one segment: the ticks of its five tokens and the rest of the request.
    let began = ticks();
    let (mut rx, _cancel) = rig.submit(request(5));
    assert_eq!(finish_of(&until_end(&mut rx).await), Finish::Length);
    assert!(wait_until(WAIT, || rig.backend.progress().unwrap().pending == 0).await);
    let short = ticks() - began;

    // 9,000 tokens are three segments (4,096, 4,096 and the rest), and the worker asks the sink before each.
    let mut big = request(5);
    big.prompt = (0..9_000).map(|i| i % 64).collect();
    let began = ticks();
    let (mut rx, _cancel) = rig.submit(big);
    // The request is pending from the submit, before the worker has seen it, and until its sink is dropped.
    assert_eq!(rig.backend.progress().unwrap().pending, 1);
    tokens(&mut rx, 2).await;
    assert_eq!(rig.backend.progress().unwrap().pending, 1);
    assert_eq!(finish_of(&until_end(&mut rx).await), Finish::Length);
    // The worker drops the sink after the reply-end snapshot, a moment after `Done`.
    assert!(wait_until(WAIT, || rig.backend.progress().unwrap().pending == 0).await);
    let long = ticks() - began;
    assert!(short >= 8, "{short}");
    assert!(
        long >= short + 2,
        "the two extra segments are two ticks: {short} ticks for one, {long} for three"
    );
}

#[tokio::test]
async fn pending_counts_commands_until_their_reply() {
    // The fake sleeps in the load call and in the park, on the worker's thread, so each command is under way when
    // the pending count is read.
    let rig = Rig::new(
        sevens()
            .load_time(Duration::from_millis(300))
            .on_load(|_| std::thread::sleep(Duration::from_millis(300)))
            .park_time(Duration::from_millis(300)),
        None,
    );
    let ticks = rig.backend.progress().unwrap().ticks;
    let load = rig.backend.load();
    assert_eq!(rig.backend.progress().unwrap().pending, 1);
    assert_eq!(reply(load).await, Ok(()));
    // The reply counts down before it is sent, so an answer is never read next to a pending command.
    let after = rig.backend.progress().unwrap();
    assert_eq!(after.pending, 0);
    assert!(after.ticks > ticks, "{after:?}");

    // The same for an unload and a stats command.
    let unload = rig.backend.unload();
    assert_eq!(rig.backend.progress().unwrap().pending, 1);
    assert_eq!(reply(unload).await, Ok(true));
    assert_eq!(rig.backend.progress().unwrap().pending, 0);
    reply(rig.backend.stats()).await.unwrap();
    assert_eq!(rig.backend.progress().unwrap().pending, 0);
}

#[tokio::test]
async fn work_given_to_a_busy_worker_does_not_move_the_ticks() {
    // A watchdog measures the time since the last tick while there is work, so an arrival must not tick: under steady
    // traffic it would hide a worker stuck mid-decode. The worker is held in its load (it ticks nowhere in it), so
    // only a submit could move the count.
    let rig = Rig::new(
        sevens().on_load(|_| std::thread::sleep(Duration::from_millis(300))),
        None,
    );
    let ticks = || rig.backend.progress().unwrap().ticks;
    let load = rig.backend.load();
    assert!(wait_until(WAIT, || rig.backend.state() == Lifecycle::Loading).await);
    let t0 = ticks();
    let (mut rx, _cancel) = rig.submit(request(2));
    assert_eq!(ticks(), t0, "a request");
    assert_eq!(rig.backend.progress().unwrap().pending, 2);
    let _stats = rig.backend.stats();
    assert_eq!(ticks(), t0, "a command");
    assert_eq!(rig.backend.progress().unwrap().pending, 3);
    assert_eq!(reply(load).await, Ok(()));
    until_end(&mut rx).await;
}

#[tokio::test]
async fn a_request_the_worker_skips_still_ends_its_pending() {
    let rig = Rig::new(sevens().tweak(|_, l| slow(l)), None);
    let (mut first, cancel) = rig.submit(request(50));
    // Closed before it is submitted, so the worker finds the sink closed whenever it gets to it.
    let (sink, mut second, closed) = Sink::channel();
    closed.cancel();
    rig.backend.submit(request(3), sink).unwrap();
    assert_eq!(rig.backend.progress().unwrap().pending, 2);
    tokens(&mut first, 1).await;
    cancel.cancel();
    until_end(&mut first).await;
    assert!(wait_until(WAIT, || rig.backend.progress().unwrap().pending == 0).await);
    // It was skipped: its sink was dropped with no event.
    assert!(second.try_recv().is_err());
}

#[tokio::test]
async fn latest_stats_is_the_last_answer() {
    let rig = Rig::new(sevens(), None);
    assert_eq!(rig.backend.latest_stats(), None);
    assert_eq!(reply(rig.backend.load()).await, Ok(()));
    // The load is no stats command.
    assert_eq!(rig.backend.latest_stats(), None);
    let before = Instant::now();
    let answer = reply(rig.backend.stats()).await.unwrap();
    let copy = rig.backend.latest_stats().expect("the answer is kept");
    assert_eq!(copy.stats, answer);
    assert!(copy.at >= before);
    assert_eq!(copy.stats.loads, 1);
    assert!(copy.stats.cache.is_some(), "{:?}", copy.stats);
    // A stats command nobody waits for is kept too (and no other stats command is sent to make it so).
    drop(rig.backend.stats());
    assert!(wait_until(WAIT, || rig.backend.latest_stats().unwrap().at > copy.at).await);
    // Unloaded, the cache has no counters.
    assert_eq!(reply(rig.backend.unload()).await, Ok(true));
    reply(rig.backend.stats()).await.unwrap();
    assert_eq!(rig.backend.latest_stats().unwrap().stats.cache, None);
}

// ---- shutdown ----

#[tokio::test]
async fn shutdown_during_a_generation_ends_it_within_one_token_and_parks_nothing() {
    let rig = Rig::new(sevens().tweak(|_, l| slow(l)), None);
    let (mut rx, _cancel) = rig.submit(request(50));
    tokens(&mut rx, 3).await;
    let stop = rig.backend.shutdown();
    let rest = until_end(&mut rx).await;
    assert!(token_count(&rest) <= 1, "{rest:?}");
    assert_eq!(finish_of(&rest), Finish::Cancelled);
    assert_eq!(reply(stop).await, Ok(true));
    let took = rig.finish().await;
    assert!(took < Duration::from_secs(4), "finish took {took:?}");
    assert!(rig.log.parks().is_empty());
    assert_eq!(rig.log.unloads(), 1);
}

#[tokio::test]
async fn shutdown_during_a_load_parks_nothing_and_joins() {
    let rig = Rig::new(
        sevens().on_load(|_| std::thread::sleep(Duration::from_secs(1))),
        None,
    );
    let (mut rx, _cancel) = rig.submit(request(10));
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        rig.backend.state(),
        Lifecycle::Loading,
        "the load ended first: the test stalled"
    );
    let stop = rig.backend.shutdown();
    assert!(reply(stop).await.is_ok());
    let took = rig.finish().await;
    assert!(took < Duration::from_secs(4), "finish took {took:?}");
    assert!(rig.log.parks().is_empty());
    // The request got its `Done`, or nothing: never a hang.
    let end = tokio::time::timeout(WAIT, rx.recv())
        .await
        .expect("the request's channel ends");
    assert!(matches!(end, None | Some(GenEvent::Done { .. })), "{end:?}");
}

#[tokio::test]
async fn shutdown_during_an_idle_park_writes_no_shutdown_line() {
    let rig = Rig::new(sevens().park_time(PARK), Some(Duration::from_millis(300)));
    assert_eq!(reply(rig.backend.load()).await, Ok(()));
    // 100 ms into the park, which begins 300 ms after the load.
    tokio::time::sleep(Duration::from_millis(400)).await;
    rig.assert_in_the_park();
    // The model is already on its way out, so the shutdown unloads nothing and the server logs no second line.
    assert_eq!(reply(rig.backend.shutdown()).await, Ok(false));
    rig.finish().await;
    let lines = lifecycle_lines(&rig.dir);
    let names: Vec<_> = lines.iter().map(|l| l["event"].as_str().unwrap()).collect();
    assert_eq!(names, ["load", "unload", "stats"]);
    assert_eq!(lines[1]["cause"], "idle");
}

#[tokio::test]
async fn a_second_shutdown_replies_false() {
    let rig = Rig::new(sevens(), None);
    assert_eq!(reply(rig.backend.load()).await, Ok(()));
    assert_eq!(reply(rig.backend.shutdown()).await, Ok(true));
    assert_eq!(reply(rig.backend.shutdown()).await, Ok(false));
    rig.finish().await;
    // The shutdown unloaded without parking, and the observer wrote no line for it.
    assert!(rig.log.parks().is_empty());
    assert_eq!(rig.events(), ["load"]);
}

#[tokio::test]
async fn finish_shuts_the_worker_down_when_no_stop_came() {
    let rig = Rig::new(sevens(), None);
    assert_eq!(reply(rig.backend.load()).await, Ok(()));
    let took = rig.finish().await;
    assert!(took < Duration::from_secs(4), "finish took {took:?}");
    assert_eq!(rig.log.unloads(), 1);
    let (sink, _rx, _cancel) = Sink::channel();
    assert_eq!(rig.backend.submit(request(3), sink), Err(Full));
}

#[tokio::test]
async fn a_finished_unload_leaves_the_fast_path_open() {
    let rig = Rig::new(sevens().tweak(|_, l| slow(l)), None);
    assert_eq!(reply(rig.backend.load()).await, Ok(()));
    assert_eq!(reply(rig.backend.unload()).await, Ok(true));
    let (mut rx, cancel) = rig.submit(request(30));
    assert!(matches!(next(&mut rx).await, GenEvent::Started { .. }));
    // The unload is answered, so its count is back to 0: the load is answered at once, ahead of the request's end.
    assert_eq!(reply(rig.backend.load()).await, Ok(()));
    while let Ok(e) = rx.try_recv() {
        assert!(!matches!(e, GenEvent::Done { .. }), "the load waited");
    }
    cancel.cancel();
    until_end(&mut rx).await;
}

#[tokio::test]
async fn finish_returns_at_its_cap_when_the_stop_outlasts_it() {
    // The worker can't interrupt a load: the stop waits for these 2 s.
    let rig = Rig::new(
        sevens().on_load(|_| std::thread::sleep(Duration::from_secs(2))),
        None,
    );
    let (_rx, _cancel) = rig.submit(request(10));
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        rig.backend.state(),
        Lifecycle::Loading,
        "the load ended first: the test stalled"
    );
    let stop = rig.backend.shutdown();
    let took = rig.finish_within(Duration::from_millis(300)).await;
    // It gave up at the cap, with the worker still loading, and did not wait for the join.
    assert!(
        took >= Duration::from_millis(250) && took < Duration::from_millis(1500),
        "finish took {took:?}"
    );
    assert!(rig.log.unloads() == 0, "the worker was still loading");
    // The stop completes in its own time: nothing is left running behind the test.
    assert!(reply(stop).await.is_ok());
}

// ---- the watchdog over the real adapter ----

#[tokio::test]
async fn a_long_prefill_in_segments_does_not_stall() {
    // The prefill as a whole takes more than twice the bound, and each gap between segments is under half of it: the
    // watchdog stays quiet only because the worker's sink check before every segment is a tick.
    const STALL_S: u64 = 2;
    const SEGMENT: Duration = Duration::from_millis(700);
    let (loader, mut cfg) = long_ctx_parts();
    cfg.stall_s = STALL_S;
    let segments = Arc::new(AtomicUsize::new(0));
    let prefilled = Arc::new(AtomicUsize::new(0));
    let (seen, tokens_seen) = (segments.clone(), prefilled.clone());
    let loader = loader.tweak(move |_, live| {
        let (seen, tokens_seen) = (seen.clone(), tokens_seen.clone());
        live.on_call(move |c| {
            if let Call::Prefill(_, n) = c {
                seen.fetch_add(1, Ordering::SeqCst);
                tokens_seen.fetch_add(*n, Ordering::SeqCst);
                std::thread::sleep(SEGMENT);
            }
        });
    });
    let dir = scratch_dir("app");
    let backend = Arc::new(EngineBackend::with_worker(
        |notify| {
            Worker::spawn_with(
                loader,
                WorkerParams {
                    ctx: LONG_CTX,
                    n_vocab: vocab_size(),
                    ..params(None, notify)
                },
            )
        },
        Logger::new(dir.clone(), 0).unwrap(),
    ));
    let tuning = Tuning {
        watchdog_tick: Duration::from_millis(50),
        shutdown_grace: Duration::from_secs(5),
        ..Tuning::default()
    };
    let mut running = start(parts(backend.clone(), cfg, None, &dir), tuning).await;

    // About 17,000 tokens of prompt: "hello " is one token each.
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hello ".repeat(17_000)}],
        "reasoning_effort": "none",
        "max_tokens": 4,
    })
    .to_string();
    let resp = running
        .send("POST", "/v1/chat/completions", &[], body.as_bytes())
        .await;
    assert_eq!(resp.status, 200, "{}", String::from_utf8_lossy(&resp.body));
    // Four full segments and the rest (the engine also parts the prompt at the cut): six prefill calls of `SEGMENT`.
    let calls = u32::try_from(segments.load(Ordering::SeqCst)).unwrap();
    assert!(calls >= 6, "{calls} prefill calls");
    assert!(prefilled.load(Ordering::SeqCst) > 16_500);
    assert!(
        SEGMENT * calls >= Duration::from_secs(2 * STALL_S),
        "the prefill ({:?}) must outlast twice the bound",
        SEGMENT * calls
    );
    assert!(
        SEGMENT * 2 <= Duration::from_secs(STALL_S),
        "each gap is under half the bound"
    );
    assert!(running.halts().is_empty());
    assert!(!events(&dir).contains(&"stall".to_string()));
    assert_eq!(running.stop.stop().await, Ok(()));
    let b = backend.clone();
    tokio::task::spawn_blocking(move || b.finish(Duration::from_secs(5)))
        .await
        .unwrap();
}

#[tokio::test]
async fn the_test_hook_stalls_the_worker_and_the_watchdog_halts() {
    // The hook parks the worker at the third token of a request, before the sink forwards it or ticks. The fake engine
    // is not stuck: it is the adapter's sink that holds the worker, as it does on the real engine.
    let mut cfg = Config {
        model: Some("m.gguf".into()),
        ctx: CTX as u32,
        ..Config::default()
    };
    cfg.stall_s = 2;
    let dir = scratch_dir("app");
    let backend = Arc::new(EngineBackend::with_worker(
        |notify| {
            Worker::spawn_with(
                sevens(),
                WorkerParams {
                    n_vocab: vocab_size(),
                    ..params(None, notify)
                },
            )
        },
        Logger::new(dir.clone(), 0).unwrap(),
    ));
    backend.set_test_stall(Some(3));
    let tuning = Tuning {
        watchdog_tick: Duration::from_millis(50),
        shutdown_grace: Duration::from_secs(2),
        ..Tuning::default()
    };
    let mut running = start(parts(backend.clone(), cfg, None, &dir), tuning).await;

    let port: u16 = running
        .base_url
        .rsplit(':')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let client = Client::new("127.0.0.1", port, "k").unwrap();
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": "none",
        "max_tokens": 20,
        "stream": true,
    });
    let mut stream = client.stream("/v1/chat/completions", &body).await.unwrap();
    assert_eq!(stream.status, 200);

    // The id of the stream and its content chunks, until the halt is recorded and then what was already on the wire.
    let (mut id, mut chunks) = (None, 0);
    let mut read = |item: SseItem| {
        let SseItem::Event { data, .. } = item else {
            return;
        };
        if data != "[DONE]" {
            let chunk: serde_json::Value = serde_json::from_str(&data).unwrap();
            id = chunk["id"].as_str().map(String::from).or(id.take());
            let content = &chunk["choices"][0]["delta"]["content"];
            chunks += usize::from(content.as_str().is_some_and(|c| !c.is_empty()));
        }
    };
    // Generous: this waits for the event and ends when it comes. The checks that mean something are below.
    let patience = Duration::from_secs(20);
    let end = Instant::now() + patience;
    while running.halts().is_empty() {
        assert!(Instant::now() < end, "no halt within {patience:?}");
        if let Some(item) = stream.next(Duration::from_millis(100)).await {
            read(item);
        }
    }
    while let Some(item) = stream.next(Duration::from_millis(200)).await {
        read(item);
    }
    assert_eq!(running.halts(), [4]);
    assert_eq!(chunks, 2, "the third token never left the sink");

    let lines = lifecycle_lines(&dir);
    let names: Vec<_> = lines.iter().map(|l| l["event"].as_str().unwrap()).collect();
    assert_eq!(names, ["start", "load", "stall", "stats", "exit"]);
    assert_eq!(lines[2]["doing"], "request");
    assert_eq!(lines[2]["request"].as_str(), id.as_deref());
    assert_eq!(lines[2]["bound_s"], 2);
    assert_eq!(lines[4]["reason"], "stall");

    // The recorder does not end the process, so the worker is still parked: the stop sets `stopping`, which ends the
    // park, and the worker joins instead of being leaked. A leaked worker is one `finish` gives up on at its cap, so
    // the join is told apart from that by finishing well inside a cap that a loaded machine cannot reach.
    let t = Instant::now();
    running.stop.fire();
    assert!(running.stop.join(Duration::from_secs(20)).await.is_some());
    let b = backend.clone();
    tokio::task::spawn_blocking(move || b.finish(Duration::from_secs(20)))
        .await
        .unwrap();
    assert!(t.elapsed() < Duration::from_secs(15), "{:?}", t.elapsed());
    drop(stream);
    // `exit` is still the log's last line.
    assert_eq!(events(&dir).last().map(String::as_str), Some("exit"));
}

// ---- speculation ----

/// The text the fake model copies: eleven distinct capitals, each its own token even side by side (found by search:
/// most runs of capitals merge), with ids under the fake's 64-token output vocabulary (its logits row has 64 entries,
/// so it can only pick ids below that).
const COPY: &str = "IYBQDZCJFVH";

fn copy_ids() -> Vec<u32> {
    let ids: Vec<u32> = COPY
        .chars()
        .map(|c| {
            let t = tokenizer().encode(&c.to_string(), false);
            assert_eq!(t.len(), 1, "{c:?} is one token");
            assert!((t[0] as usize) < VOCAB, "{c:?} is id {}", t[0]);
            t[0]
        })
        .collect();
    // In a prompt too, the letters stay one token each (no merge), so lookup finds them there.
    assert_eq!(tokenizer().encode(COPY, false), ids);
    ids
}

/// The model copies `ids` in order, whatever the prompt: its next token is the one after the run of copied tokens it
/// has emitted so far (the template's tokens end every prompt, and none is a letter of `ids`).
fn copying(ids: Vec<u32>) -> FakeLoader {
    FakeLoader::new(CTX, move |h| {
        let run = h.iter().rev().take_while(|t| ids.contains(t)).count();
        ids[run % ids.len()]
    })
}

/// A server over the adapter with speculation on at a fixed draft of 4 (the wall-clock controller would
/// pick plain steps on a fake, and no burst would ever run).
async fn spec_server(loader: FakeLoader) -> (Running, Arc<EngineBackend>) {
    let dir = scratch_dir("spec");
    let backend = Arc::new(EngineBackend::with_worker(
        |notify| {
            Worker::spawn_with(
                loader,
                WorkerParams {
                    n_vocab: vocab_size(),
                    speculation: Some(SpecConfig {
                        max_draft: 4,
                        fixed_draft: Some(4),
                        ..SpecConfig::default()
                    }),
                    ..params(None, notify)
                },
            )
        },
        Logger::new(dir.clone(), 0).unwrap(),
    ));
    let cfg = Config {
        model: Some("m.gguf".into()),
        ctx: CTX as u32,
        ..Config::default()
    };
    let running = start(parts(backend.clone(), cfg, None, &dir), Tuning::default()).await;
    (running, backend)
}

async fn stop_spec_server(mut running: Running, backend: Arc<EngineBackend>) {
    assert_eq!(running.stop.stop().await, Ok(()));
    tokio::task::spawn_blocking(move || backend.finish(Duration::from_secs(5)))
        .await
        .unwrap();
}

fn chat(content: &str, extra: serde_json::Value) -> Vec<u8> {
    let mut b = serde_json::json!({
        "messages": [{"role": "user", "content": content}],
        "reasoning_effort": "none",
    });
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b.to_string().into_bytes()
}

fn request_lines(running: &Running) -> Vec<serde_json::Value> {
    running
        .log_lines()
        .into_iter()
        .filter(|l| l.get("event").is_none())
        .collect()
}

#[tokio::test]
async fn the_adapter_maps_the_spec_counts() {
    let (running, backend) = spec_server(copying(copy_ids())).await;
    // The prompt holds the text the model copies, so prompt lookup drafts it from the third token on: the first token
    // is the prefill's, the second a plain step, then two bursts of x and 4 drafts.
    let r = running
        .send(
            "POST",
            "/v1/chat/completions",
            &[],
            &chat(COPY, serde_json::json!({"max_tokens": 12})),
        )
        .await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    assert!(wait_until(WAIT, || !request_lines(&running).is_empty()).await);

    let line = &request_lines(&running)[0];
    let n = |k: &str| line[k].as_u64().unwrap_or_else(|| panic!("{k} in {line}"));
    assert_eq!(n("spec_steps"), 2, "{line}");
    assert_eq!(n("spec_plain_steps"), 1, "{line}");
    assert_eq!(n("spec_drafted"), 8, "{line}");
    // The model copies what lookup drafts: every draft is right.
    assert_eq!(n("spec_accepted"), 8, "{line}");
    assert_eq!(line["spec_accept_rate"].as_f64(), Some(1.0), "{line}");
    assert_eq!(n("spec_replays"), 0, "{line}");

    // The stats carry the same steps, by size; the answer is the latest copy `/v1/stats` reads.
    reply(backend.stats()).await.unwrap();
    let stats = running.send("GET", "/v1/stats", &[], b"").await.json();
    let spec = &stats["stats"]["spec"];
    for (stat, logged) in [
        ("steps", "spec_steps"),
        ("plain_steps", "spec_plain_steps"),
        ("drafted", "spec_drafted"),
        ("accepted", "spec_accepted"),
        ("replays", "spec_replays"),
    ] {
        assert_eq!(spec[stat], line[logged], "{stat}: {stats}");
    }
    assert!(spec["accept_rate"]["lookup"].is_number(), "{stats}");
    let rows = spec["by_size"].as_array().expect("by_size is a list");
    // Only the sizes that ran: the plain step (t = 1) and the bursts (t = 5), with every step counted once.
    let sizes: Vec<(u64, u64)> = rows
        .iter()
        .map(|r| (r["t"].as_u64().unwrap(), r["steps"].as_u64().unwrap()))
        .collect();
    assert_eq!(sizes, [(1, 1), (5, 2)], "{stats}");
    // The plain step kept one position, the bursts five each.
    assert_eq!(rows[0]["tokens"], 1);
    assert_eq!(rows[1]["tokens"], 10);
    assert!(rows.iter().all(|r| r["mean_ms"].is_number()));
    assert!(rows.iter().all(|r| r["mean_miss_mib"].is_number()));
    stop_spec_server(running, backend).await;
}

/// A stop string that matches inside an accepted burst ends the client's text there, and the request
/// after it, with that reply appended, restores and answers.
#[tokio::test]
async fn a_stop_string_inside_a_burst_ends_the_text_and_the_next_request_reuses() {
    let loader = copying(copy_ids());
    let log = loader.log();
    let (running, backend) = spec_server(loader).await;
    // The first burst is x = "Y" with the drafts "B" "Q" "D" "Z"; its rows pick "B" "Q" "D" "Z" "C". "Q" is the 2nd of
    // them, and the first "Q" in the reply.
    let first = running
        .send(
            "POST",
            "/v1/chat/completions",
            &[],
            &chat(COPY, serde_json::json!({"max_tokens": 12, "stop": ["Q"]})),
        )
        .await
        .json();
    let text = first["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(first["choices"][0]["finish_reason"], "stop", "{first}");
    assert_eq!(text, "IYB", "the text ends before the stop string");

    // The first burst carried the stop string's token as the 2nd of its 4 drafts, and the engine accepted them. The
    // worker is far faster than the HTTP side, so how much of the burst it commits before the server's cancel arrives
    // is a race (the cut inside a burst is pinned by `a_cancel_inside_a_burst_commits_the_rows_reached`):
    // two rows at least, five at most. The client's text ends at the stop either way.
    let calls = log.live_calls();
    let burst = calls
        .iter()
        .position(|c| matches!(c, Call::Verify(v) if v.len() == 5))
        .unwrap_or_else(|| panic!("no 5-token verify in {calls:?}"));
    assert!(
        matches!(&calls[burst], Call::Verify(v) if v[2] == copy_ids()[3]),
        "{calls:?}"
    );
    assert!(
        matches!(calls[burst + 1], Call::Commit(n) if (2..=5).contains(&n)),
        "{calls:?}"
    );
    assert!(wait_until(WAIT, || !request_lines(&running).is_empty()).await);
    let line = &request_lines(&running)[0];
    assert!(line["spec_steps"].as_u64().unwrap() >= 1, "{line}");
    assert!(line["spec_drafted"].as_u64().unwrap() >= 4, "{line}");

    // The next request is the conversation with that reply appended.
    let again = serde_json::json!({
        "messages": [
            {"role": "user", "content": COPY},
            {"role": "assistant", "content": text},
            {"role": "user", "content": "again"},
        ],
        "reasoning_effort": "none",
        "max_tokens": 4,
    })
    .to_string();
    let r = running
        .send("POST", "/v1/chat/completions", &[], again.as_bytes())
        .await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    assert!(wait_until(WAIT, || request_lines(&running).len() == 2).await);
    let next = &request_lines(&running)[1];
    assert_eq!(next["status"], 200, "{next}");
    assert_ne!(next["restored"], "none", "{next}");
    assert!(next["reused_tokens"].as_u64().unwrap() > 0, "{next}");
    stop_spec_server(running, backend).await;
}
