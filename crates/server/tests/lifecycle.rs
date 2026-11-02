//! The lifecycle from the HTTP side.
mod common;

use common::{tokenizer, wait_until, Resp, Setup, Sse, SseItem, TestServer};
use serde_json::{json, Value};
use server::app::{Exit, ManualClock};
use server::backend::{Backend, Lifecycle};
use server::fake::{Fail, Script};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const KEY: &str = "test-key-0123456789";
const CHAT: &str = "/v1/chat/completions";
const MSGS: &str = "/v1/messages";
const WAIT: Duration = Duration::from_secs(20);

fn chat_body(stream: bool) -> Vec<u8> {
    json!({"messages": [{"role": "user", "content": "hi"}], "stream": stream, "reasoning_effort": "none"})
        .to_string()
        .into_bytes()
}

fn msgs_body(stream: bool) -> Vec<u8> {
    json!({"model": "m", "max_tokens": 1024, "stream": stream,
        "messages": [{"role": "user", "content": "hi"}]})
    .to_string()
    .into_bytes()
}

/// A script of about `n` words at `per_token` each: long enough to be cut short.
fn long(n: usize, per_token: Duration) -> Script {
    Script {
        per_token,
        ..Script::text(
            tokenizer(),
            &format!("x\n</think>\n\n{}", "word ".repeat(n)),
        )
    }
}

fn auth() -> [(&'static str, &'static str); 1] {
    [("Authorization", "Bearer test-key-0123456789")]
}

async fn post(srv: &TestServer, path: &str, body: &[u8]) -> Resp {
    srv.send("POST", path, &[], body).await
}

async fn health(srv: &TestServer) -> (u16, Value) {
    let r = srv.send("GET", "/health", &[], b"").await;
    (r.status, r.json())
}

/// Every item of a stream until it ends.
async fn read_all(sse: &mut Sse) -> Vec<SseItem> {
    let mut items = Vec::new();
    loop {
        match sse.next(WAIT).await {
            Some(i) => items.push(i),
            None => {
                assert!(sse.ended(), "the stream stalled for {WAIT:?}");
                return items;
            }
        }
    }
}

/// The in-band error of an OpenAI stream: the last `data:` object that has an `error` key. `[DONE]` must follow it.
fn openai_error(items: &[SseItem]) -> Value {
    let data: Vec<&str> = items
        .iter()
        .filter_map(|i| match i {
            SseItem::Event { data, .. } => Some(data.as_str()),
            SseItem::Comment(_) => None,
        })
        .collect();
    assert_eq!(data.last(), Some(&"[DONE]"), "{data:?}");
    data.iter()
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .rfind(|v| v.get("error").is_some())
        .unwrap_or_else(|| panic!("no error event in {data:?}"))["error"]
        .clone()
}

/// The `error` event of an Anthropic stream, which must be the last one.
fn anthropic_error(items: &[SseItem]) -> Value {
    let events: Vec<(&Option<String>, &String)> = items
        .iter()
        .filter_map(|i| match i {
            SseItem::Event { name, data } => Some((name, data)),
            SseItem::Comment(_) => None,
        })
        .collect();
    let (name, data) = events.last().expect("some events");
    assert_eq!(name.as_deref(), Some("error"), "{events:?}");
    serde_json::from_str::<Value>(data).unwrap()["error"].clone()
}

/// Waits until the fake has started request number `n`.
async fn started(srv: &TestServer, n: usize) {
    assert!(
        wait_until(WAIT, || srv.fake.requests().len() == n).await,
        "the engine never started request {n}"
    );
}

async fn waiting_is(srv: &TestServer, n: usize) {
    assert!(
        wait_until(WAIT, || srv.waiting() == n).await,
        "waiting() is {}, expected {n}",
        srv.waiting()
    );
}

async fn state_is(srv: &TestServer, want: Lifecycle) {
    assert!(
        wait_until(WAIT, || srv.fake.state() == want).await,
        "the state is {:?}, expected {want:?}",
        srv.fake.state()
    );
}

#[tokio::test]
async fn h5_health_answers_within_a_second_mid_work() {
    let mut s = Setup::new();
    s.fake.start_loaded = true;
    let srv = TestServer::start(s).await;
    // A 30 s prefill, then a 30 s decode: the stop token takes a step like any other.
    let tokens = tokenizer().encode("hello", true);
    let steps = u32::try_from(tokens.len()).unwrap() + 1;
    srv.fake.push(Script {
        tokens,
        prefill: Duration::from_secs(30),
        per_token: Duration::from_secs(30) / steps,
        ..Script::default()
    });
    // Streamed, so the reply's headers (after `Started`) beat the harness's 60 s wait for a whole reply.
    let mut sse = srv.stream(CHAT, &[], &chat_body(true)).await;
    assert_eq!(sse.status(), 200);
    let done = AtomicBool::new(false);
    let read = async {
        let items = read_all(&mut sse).await;
        done.store(true, Ordering::Release);
        items
    };
    let probe = async {
        let mut times = Vec::new();
        // A probe every 2 s on the clock, not 2 s after the last answer: a slow machine stretches each answer
        // (which the 1 s bound below still catches) without thinning the count of probes in the minute of work.
        let mut tick = tokio::time::interval(Duration::from_secs(2));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        while !done.load(Ordering::Acquire) {
            tick.tick().await;
            let t = Instant::now();
            let (status, body) = health(&srv).await;
            let took = t.elapsed();
            assert_eq!((status, body["status"].as_str()), (200, Some("ready")));
            assert!(took < Duration::from_secs(1), "/health took {took:?}");
            times.push(took);
        }
        times
    };
    let (items, times) = tokio::join!(read, probe);
    assert!(
        times.len() >= 25,
        "only {} probes in a minute of work",
        times.len()
    );
    assert!(matches!(items.last(), Some(SseItem::Event { data, .. }) if data == "[DONE]"));
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn h6_idle_unload() {
    let mut s = Setup::new();
    s.fake.start_loaded = true;
    s.fake.idle_unload = Some(Duration::from_secs(2));
    let srv = TestServer::start(s).await;
    assert_eq!(post(&srv, CHAT, &chat_body(false)).await.status, 200);
    assert_eq!(health(&srv).await.1["status"], "ready");
    assert!(
        wait_until(Duration::from_secs(4), || srv.fake.unloads().len() == 1).await,
        "no idle unload within 4 s"
    );
    // Measured on the fake's own instants: the client's clock starts after the reply reached it.
    let last_token = *srv.fake.runs()[0]
        .token_times
        .last()
        .expect("the request sent a token");
    let idle = srv.fake.unloads()[0].duration_since(last_token);
    assert!(
        idle >= Duration::from_secs(2),
        "unloaded after {idle:?} idle"
    );
    let (status, body) = health(&srv).await;
    assert_eq!(
        (status, &body["status"], &body["loaded"]),
        (200, &json!("idle"), &json!(false))
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn h8_device_loss_recovering_and_exit() {
    let clock = Arc::new(ManualClock::new());
    let mut s = Setup::new();
    s.fake.start_loaded = true;
    s.fake.recover_time = Duration::from_secs(2);
    s.tuning.clock = clock.clone();
    let mut srv = TestServer::start(s).await;
    let lose = |per_token: Duration| Script {
        per_token,
        fail_at: Some((0, Fail::DeviceLost)),
        ..Script::text(tokenizer(), "lost")
    };

    // The first loss, non-streaming: 503; a request that arrives meanwhile waits through the recovery and is served.
    srv.fake.push(lose(Duration::from_secs(1)));
    let body = chat_body(false);
    let first = post(&srv, CHAT, &body);
    let queued = async {
        started(&srv, 1).await;
        post(&srv, MSGS, &msgs_body(false)).await
    };
    let probe = async {
        state_is(&srv, Lifecycle::Recovering).await;
        // The second request was already waiting when the recovery began, and keeps waiting through it.
        assert_eq!(srv.waiting(), 1);
        assert_eq!(srv.fake.requests().len(), 1);
        let (status, body) = health(&srv).await;
        assert_eq!((status, &body["status"]), (503, &json!("recovering")));
        state_is(&srv, Lifecycle::Ready).await;
        let (status, body) = health(&srv).await;
        assert_eq!(
            (status, &body["status"], &body["loaded"]),
            (200, &json!("ready"), &json!(true))
        );
    };
    let (first, queued, ()) = tokio::join!(first, queued, probe);
    assert_eq!(first.status, 503);
    assert_eq!(first.json()["error"]["type"], "service_unavailable");
    assert_eq!(
        queued.status,
        200,
        "{}",
        String::from_utf8_lossy(&queued.body)
    );
    assert_eq!(srv.app.exited(), None);

    // A second loss 11 minutes on, streaming: an error event, and no exit.
    clock.advance(Duration::from_secs(11 * 60));
    srv.fake.push(lose(Duration::ZERO));
    let mut sse = srv.stream(MSGS, &[], &msgs_body(true)).await;
    assert_eq!(sse.status(), 200);
    let err = anthropic_error(&read_all(&mut sse).await);
    assert_eq!(err["type"], "overloaded_error");
    assert_eq!(srv.app.exited(), None);
    state_is(&srv, Lifecycle::Ready).await;
    assert!(srv.exited(Duration::from_millis(100)).await.is_none());

    // A third, a minute after the second: the process is to exit.
    clock.advance(Duration::from_secs(60));
    srv.fake.push(lose(Duration::ZERO));
    let third = post(&srv, CHAT, &chat_body(false)).await;
    assert_eq!(third.status, 503);
    assert_eq!(srv.app.exited(), Some(Exit::DeviceLostTwice));
    assert_eq!(srv.exited(WAIT).await, Some(Err(Exit::DeviceLostTwice)));
}

#[tokio::test]
async fn h9_load_and_unload() {
    let mut s = Setup::new();
    s.key = Some(KEY);
    s.fake.load_time = Duration::from_millis(200);
    let srv = Arc::new(TestServer::start(s).await);

    // No key: 401 on both, and nothing loaded.
    for path in ["/v1/load", "/v1/unload"] {
        assert_eq!(srv.send("POST", path, &[], b"").await.status, 401, "{path}");
    }
    assert!(srv.fake.loads().is_empty());

    // Idle: unload answers 200 at once and the fake has nothing to unload.
    let r = srv.send("POST", "/v1/unload", &auth(), b"").await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json(), json!({"status": "idle", "loaded": false}));
    assert!(srv.fake.unloads().is_empty());

    // Load: 200 ready, and a second load answers 200 without loading again.
    let r = srv.send("POST", "/v1/load", &auth(), b"").await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json(), json!({"status": "ready", "loaded": true}));
    assert_eq!(srv.fake.loads().len(), 1);
    let r = srv.send("POST", "/v1/load", &auth(), b"").await;
    assert_eq!(
        (r.status, r.json()),
        (200, json!({"status": "ready", "loaded": true}))
    );
    assert_eq!(srv.fake.loads().len(), 1);

    // An unload during a generation, non-streaming: the request in flight and two queued ones get 503.
    srv.fake.push(long(200, Duration::from_millis(100)));
    let spawn = |path: &'static str, body: Vec<u8>| {
        let srv = srv.clone();
        tokio::spawn(async move { srv.send("POST", path, &auth(), &body).await })
    };
    let running = spawn(CHAT, chat_body(false));
    started(&srv, 1).await;
    let q1 = spawn(CHAT, chat_body(false));
    waiting_is(&srv, 1).await;
    let q2 = spawn(MSGS, msgs_body(false));
    waiting_is(&srv, 2).await;
    let r = srv.send("POST", "/v1/unload", &auth(), b"").await;
    assert_eq!(
        (r.status, r.json()),
        (200, json!({"status": "idle", "loaded": false}))
    );
    let (running, q1, q2) = (running.await.unwrap(), q1.await.unwrap(), q2.await.unwrap());
    for r in [&running, &q1] {
        assert_eq!(r.status, 503);
        let e = &r.json()["error"];
        assert_eq!(
            (&e["type"], &e["message"]),
            (
                &json!("service_unavailable"),
                &json!("the model was unloaded")
            )
        );
    }
    assert_eq!(q2.status, 503);
    assert_eq!(q2.json()["error"]["type"], "overloaded_error");
    assert_eq!(srv.fake.unloads().len(), 1, "the fake recorded the unload");
    assert_eq!(
        srv.fake.requests().len(),
        1,
        "the queued requests never ran"
    );
    assert_eq!(srv.waiting(), 0);
    let (status, body) = health(&srv).await;
    assert_eq!((status, &body["status"]), (200, &json!("idle")));

    // The next chat loads again.
    let r = srv.send("POST", CHAT, &auth(), &chat_body(false)).await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    assert_eq!(srv.fake.loads().len(), 2);

    // An unload during a stream: an error event, then the end.
    srv.fake.push(long(200, Duration::from_millis(100)));
    let mut sse = srv.stream(CHAT, &auth(), &chat_body(true)).await;
    assert_eq!(sse.status(), 200);
    let headers = auth();
    let unload = srv.send("POST", "/v1/unload", &headers, b"");
    let (r, items) = tokio::join!(unload, read_all(&mut sse));
    assert_eq!(r.status, 200);
    let e = openai_error(&items);
    assert_eq!(
        (&e["type"], &e["message"]),
        (
            &json!("service_unavailable"),
            &json!("the model was unloaded")
        )
    );
    assert_eq!(srv.fake.unloads().len(), 2);
    assert_eq!(health(&srv).await.1["status"], "idle");

    // A load that fails is 503 with the engine's message, and the server goes back to idle and can load again.
    srv.fake.push(Script {
        load_fails: Some("no GPU memory".into()),
        ..Script::default()
    });
    let r = srv.send("POST", "/v1/load", &auth(), b"").await;
    assert_eq!(r.status, 503);
    assert_eq!(r.json()["error"]["message"], "no GPU memory");
    assert_eq!(r.json()["error"]["type"], "service_unavailable");
    assert_eq!(health(&srv).await.1["status"], "idle");
    let r = srv.send("POST", "/v1/load", &auth(), b"").await;
    assert_eq!(
        (r.status, r.json()),
        (200, json!({"status": "ready", "loaded": true}))
    );
    Arc::into_inner(srv).unwrap().shutdown().await.unwrap();
}

#[tokio::test]
async fn unload_during_a_lazy_load_and_twice_at_once() {
    let mut s = Setup::new();
    s.fake.load_time = Duration::from_secs(2);
    let srv = TestServer::start(s).await;
    let body = chat_body(false);
    let chat = post(&srv, CHAT, &body);
    let unloads = async {
        state_is(&srv, Lifecycle::Loading).await;
        let both = async {
            tokio::join!(
                srv.send("POST", "/v1/unload", &[], b""),
                srv.send("POST", "/v1/unload", &[], b"")
            )
        };
        // Both are asked for while the load is still going: they overlapped, queued behind it.
        let overlapped = async {
            wait_until(WAIT, || srv.fake.pending_unloads() == 2).await
                && srv.fake.state() == Lifecycle::Loading
        };
        let (pair, overlapped) = tokio::join!(both, overlapped);
        assert!(overlapped, "the two unloads were never pending together");
        pair
    };
    let (chat, (u1, u2)) = tokio::join!(chat, unloads);
    for u in [&u1, &u2] {
        assert_eq!(
            (u.status, u.json()),
            (200, json!({"status": "idle", "loaded": false}))
        );
    }
    // The request that triggered the load is the one that failed.
    assert_eq!(chat.status, 503);
    assert_eq!(chat.json()["error"]["message"], "the model was unloaded");
    state_is(&srv, Lifecycle::Idle).await;
    let (status, body) = health(&srv).await;
    assert_eq!(
        (status, &body["status"], &body["loaded"]),
        (200, &json!("idle"), &json!(false))
    );
    assert_eq!(srv.fake.requests().len(), 0, "the request never ran");
    // Nothing deadlocked: the next request loads and runs.
    assert_eq!(post(&srv, CHAT, &chat_body(false)).await.status, 200);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn engine_gone_is_500_and_exits() {
    for (path, body, kind) in [
        (CHAT, chat_body(false), "server_error"),
        (MSGS, msgs_body(false), "api_error"),
    ] {
        let mut s = Setup::new();
        s.fake.start_loaded = true;
        let mut srv = TestServer::start(s).await;
        srv.fake.push(Script {
            fail_at: Some((0, Fail::Vanish)),
            ..Script::text(tokenizer(), "gone")
        });
        let r = post(&srv, path, &body).await;
        assert_eq!(r.status, 500, "{path}");
        let j = r.json();
        let e = &j["error"];
        assert_eq!(
            (&e["type"], &e["message"]),
            (&json!(kind), &json!("the engine stopped"))
        );
        assert_eq!(
            srv.exited(WAIT).await,
            Some(Err(Exit::EngineGone)),
            "{path}"
        );
    }
}

#[tokio::test]
async fn shutdown_cancels_unloads_and_exits_cleanly() {
    let mut s = Setup::new();
    s.fake.start_loaded = true;
    // No assertion here depends on how long a round trip takes: the token interval only has to outlast the test's few
    // requests so the unload is still pending when they are made, and the grace is raised so that no CI stall can make
    // the cap, not the drain, end the stream. That the drain ended by itself is read off the events below: the stream
    // got the unload's error, which a cut connection at the cap would not have delivered.
    s.tuning.shutdown_grace = Duration::from_secs(60);
    let mut srv = TestServer::start(s).await;
    srv.fake.push(long(200, Duration::from_secs(3)));
    let mut sse = srv.stream(MSGS, &[], &msgs_body(true)).await;
    assert_eq!(sse.status(), 200);
    let loads = srv.fake.loads().len();
    let body = chat_body(false);
    let queued = post(&srv, CHAT, &body);
    let stop = async {
        waiting_is(&srv, 1).await;
        srv.stop();
        // The unload is asked for once the server has stopped taking work and answered the queued request.
        assert!(
            wait_until(WAIT, || srv.fake.pending_unloads() == 1).await,
            "the unload was never asked for"
        );
        // Work that arrives now is refused, in both shapes, and never loads or queues.
        let late = [
            post(&srv, CHAT, &chat_body(false)).await,
            post(&srv, MSGS, &msgs_body(false)).await,
            srv.send("POST", "/v1/load", &[], b"").await,
        ];
        late
    };
    let (queued, late, items) = tokio::join!(queued, stop, read_all(&mut sse));
    for (r, anthropic) in late.iter().zip([false, true, false]) {
        check(r, anthropic, 503, "service_unavailable", "overloaded_error");
        assert_eq!(r.json()["error"]["message"], "the server is shutting down");
    }
    assert_eq!(
        srv.fake.loads().len(),
        loads,
        "a late request loaded the model"
    );
    // The stream in flight got its error event and ended; the queued request got 503.
    let e = anthropic_error(&items);
    assert_eq!(
        (&e["type"], &e["message"]),
        (&json!("overloaded_error"), &json!("the model was unloaded"))
    );
    check(
        &queued,
        false,
        503,
        "service_unavailable",
        "overloaded_error",
    );
    assert_eq!(queued.json()["error"]["message"], "the model was unloaded");
    // The stream's error comes from the request ending `Unloaded`, which the fake sends before it does the unload and
    // records it, so the record may still be empty here.
    assert!(
        wait_until(WAIT, || srv.fake.unloads().len() == 1).await,
        "the unload was never recorded"
    );
    assert_eq!(srv.fake.unloads().len(), 1);
    assert_eq!(srv.fake.requests().len(), 1);
    assert_eq!(srv.exited(WAIT).await, Some(Ok(())));
}

#[tokio::test]
async fn shutdown_does_not_wait_forever_for_a_stream_that_never_ends() {
    let mut s = Setup::new();
    s.fake.start_loaded = true;
    s.tuning.shutdown_grace = Duration::from_millis(500);
    let mut srv = TestServer::start(s).await;
    // The engine's first token takes an hour, so neither the stream nor the unload behind it can finish.
    srv.fake.push(Script {
        per_token: Duration::from_secs(3600),
        ..Script::text(tokenizer(), "stuck")
    });
    let sse = srv.stream(CHAT, &[], &chat_body(true)).await;
    assert_eq!(sse.status(), 200);
    srv.stop();
    assert_eq!(
        srv.exited(WAIT).await,
        Some(Ok(())),
        "serve held by the open stream"
    );
    drop(sse);
}

/// An error body in the path's shape: `status`, and the `type` the flavour names it by.
fn check(r: &Resp, anthropic: bool, status: u16, openai_type: &str, anthropic_type: &str) {
    assert_eq!(r.status, status, "{}", String::from_utf8_lossy(&r.body));
    let j = r.json();
    if anthropic {
        assert_eq!(j["type"], "error", "{j}");
        assert_eq!(j["error"]["type"], anthropic_type, "{j}");
    } else {
        assert!(j.get("type").is_none(), "{j}");
        assert_eq!(j["error"]["type"], openai_type, "{j}");
    }
    let m = j["error"]["message"].as_str().unwrap();
    assert!(!m.is_empty() && m.len() <= 500 && !m.contains('\n'), "{m}");
}

/// Both paths, each with its own body.
fn flavours() -> [(bool, &'static str, Vec<u8>); 2] {
    [
        (false, CHAT, chat_body(false)),
        (true, MSGS, msgs_body(false)),
    ]
}

#[tokio::test]
async fn errors_map_per_8_3() {
    // The engine's failures, one script each (a lazy load for the first row): 503 for a failed load and a lost device,
    // 500 for the rest.
    let mut s = Setup::new();
    // Each loss is a "first" one.
    s.tuning.loss_window = Duration::ZERO;
    let srv = TestServer::start(s).await;
    let rows: [(&str, Script, u16, &str, &str); 5] = [
        (
            "LoadFailed",
            Script {
                load_fails: Some("no GPU".into()),
                ..Script::default()
            },
            503,
            "service_unavailable",
            "overloaded_error",
        ),
        (
            "DeviceLost",
            fail(Fail::DeviceLost),
            503,
            "service_unavailable",
            "overloaded_error",
        ),
        (
            "NonFinite",
            fail(Fail::NonFinite),
            500,
            "server_error",
            "api_error",
        ),
        (
            "OutOfMemory",
            fail(Fail::OutOfMemory),
            500,
            "server_error",
            "api_error",
        ),
        (
            "Internal",
            fail(Fail::Internal),
            500,
            "server_error",
            "api_error",
        ),
    ];
    for (_name, script, status, openai, anthropic) in rows {
        for (is_anthropic, path, body) in flavours() {
            srv.fake.push(script.clone());
            let r = post(&srv, path, &body).await;
            check(&r, is_anthropic, status, openai, anthropic);
        }
    }
    srv.shutdown().await.unwrap();

    // 413: over 16 MiB.
    let srv = TestServer::start(Setup::new()).await;
    let big = vec![b' '; (16 << 20) + 1];
    for (is_anthropic, path, _) in flavours() {
        let r = post(&srv, path, &big).await;
        check(
            &r,
            is_anthropic,
            413,
            "invalid_request_error",
            "request_too_large",
        );
    }
    srv.shutdown().await.unwrap();

    // 401: a key is set and the request has none. 403: no key, and a Host that is not local.
    let mut s = Setup::new();
    s.key = Some(KEY);
    let srv = TestServer::start(s).await;
    for (is_anthropic, path, body) in flavours() {
        let r = post(&srv, path, &body).await;
        check(
            &r,
            is_anthropic,
            401,
            "authentication_error",
            "authentication_error",
        );
    }
    srv.shutdown().await.unwrap();
    let srv = TestServer::start(Setup::new()).await;
    for (is_anthropic, path, body) in flavours() {
        let r = srv
            .send("POST", path, &[("host", "example.com")], &body)
            .await;
        check(
            &r,
            is_anthropic,
            403,
            "permission_error",
            "permission_error",
        );
    }
    srv.shutdown().await.unwrap();

    // 400: a prompt past the context, before it queues.
    let mut s = Setup::new();
    s.config.ctx = 64;
    let srv = TestServer::start(s).await;
    let text = "word ".repeat(400);
    let openai = json!({"messages": [{"role": "user", "content": text}]}).to_string();
    let anthropic = json!({"model": "m", "max_tokens": 16,
        "messages": [{"role": "user", "content": text}]})
    .to_string();
    let r = post(&srv, CHAT, openai.as_bytes()).await;
    check(&r, false, 400, "invalid_request_error", "");
    assert_eq!(r.json()["error"]["code"], "context_length_exceeded");
    let r = post(&srv, MSGS, anthropic.as_bytes()).await;
    check(&r, true, 400, "", "invalid_request_error");
    assert_eq!(srv.fake.requests().len(), 0);
    srv.shutdown().await.unwrap();

    // 503: the queue times out.
    let mut s = Setup::new();
    s.fake.start_loaded = true;
    s.config.queue_timeout_s = 1;
    let srv = Arc::new(TestServer::start(s).await);
    srv.fake.push(Script {
        prefill: Duration::from_secs(3),
        ..Script::text(tokenizer(), "slow")
    });
    let blocker = spawn_post(&srv, CHAT, chat_body(false));
    started(&srv, 1).await;
    for (is_anthropic, path, body) in flavours() {
        let r = post(&srv, path, &body).await;
        check(
            &r,
            is_anthropic,
            503,
            "service_unavailable",
            "overloaded_error",
        );
        assert_eq!(
            r.json()["error"]["message"],
            "timed out waiting in the queue"
        );
    }
    assert_eq!(blocker.await.unwrap().status, 200);
    Arc::into_inner(srv).unwrap().shutdown().await.unwrap();

    // 503: the queue is full (64 wait, the 65th is refused).
    let mut s = Setup::new();
    s.fake.start_loaded = true;
    let srv = Arc::new(TestServer::start(s).await);
    srv.fake.push(Script {
        prefill: Duration::from_secs(2),
        ..Script::text(tokenizer(), "slow")
    });
    let blocker = spawn_post(&srv, CHAT, chat_body(false));
    started(&srv, 1).await;
    let waiters: Vec<_> = (0..64)
        .map(|_| spawn_post(&srv, CHAT, chat_body(false)))
        .collect();
    waiting_is(&srv, 64).await;
    for (is_anthropic, path, body) in flavours() {
        let r = post(&srv, path, &body).await;
        check(
            &r,
            is_anthropic,
            503,
            "service_unavailable",
            "overloaded_error",
        );
        assert_eq!(r.json()["error"]["message"], "the queue is full");
    }
    assert_eq!(blocker.await.unwrap().status, 200);
    for w in waiters {
        assert_eq!(w.await.unwrap().status, 200);
    }
    Arc::into_inner(srv).unwrap().shutdown().await.unwrap();
}

fn fail(f: Fail) -> Script {
    Script {
        fail_at: Some((0, f)),
        ..Script::text(tokenizer(), "oops")
    }
}

fn spawn_post(
    srv: &Arc<TestServer>,
    path: &'static str,
    body: Vec<u8>,
) -> tokio::task::JoinHandle<Resp> {
    let srv = srv.clone();
    tokio::spawn(async move { srv.send("POST", path, &[], &body).await })
}

// ---- what an unload says it did ----

/// The `cause` of every `unload` line the server has logged (the fake writes the idle ones), oldest first.
fn unload_causes(srv: &TestServer) -> Vec<String> {
    let mut files: Vec<_> = std::fs::read_dir(&srv.log_dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .collect();
    files.sort();
    files
        .iter()
        .flat_map(|p| {
            std::fs::read_to_string(p)
                .unwrap()
                .lines()
                .map(String::from)
                .collect::<Vec<_>>()
        })
        .map(|l| serde_json::from_str::<Value>(&l).unwrap())
        .filter(|l| l["event"] == "unload")
        .map(|l| l["cause"].as_str().unwrap().to_string())
        .collect()
}

/// A server whose idle unload takes 2 s to park (the state stays Ready meanwhile), with one request served, and the
/// moment the worker has begun that idle unload.
async fn server_parking_an_idle_unload() -> TestServer {
    let mut s = Setup::new();
    s.fake_logs = true;
    s.fake.start_loaded = true;
    s.fake.idle_unload = Some(Duration::from_secs(1));
    s.fake.unload_time = Duration::from_secs(2);
    let srv = TestServer::start(s).await;
    assert_eq!(post(&srv, CHAT, &chat_body(false)).await.status, 200);
    // Gated on the worker's own record of it, not on a sleep: a starved worker thread is waited for.
    assert!(
        wait_until(WAIT, || srv.fake.idle_unloads_begun() == 1).await,
        "the idle unload never began"
    );
    srv
}

#[tokio::test]
async fn an_unload_racing_the_idle_unload_logs_one_line() {
    // The request that lands in the middle of the park finds nothing left to unload once the worker gets to it.
    let mut srv = server_parking_an_idle_unload().await;
    let r = post(&srv, "/v1/unload", b"").await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json(), json!({"status": "idle", "loaded": false}));
    assert_eq!(srv.fake.unloads().len(), 1);
    srv.stop();
    assert_eq!(srv.exited(WAIT).await, Some(Ok(())));
    assert_eq!(unload_causes(&srv), ["idle"]);
}

#[tokio::test]
async fn a_stop_racing_the_idle_unload_logs_one_line() {
    // The stop lands in the middle of the park (2 s, well inside `shutdown_grace`'s 5 s): its shutdown finds nothing
    // loaded once the idle unload is done, so it writes no `shutdown` line.
    let mut srv = server_parking_an_idle_unload().await;
    srv.stop();
    assert_eq!(srv.exited(WAIT).await, Some(Ok(())));
    assert_eq!(srv.fake.unloads().len(), 1);
    assert_eq!(unload_causes(&srv), ["idle"]);
}

#[tokio::test]
async fn unload_while_idle_writes_no_unload_line() {
    let mut s = Setup::new();
    s.fake_logs = true;
    let mut srv = TestServer::start(s).await;
    let r = post(&srv, "/v1/unload", b"").await;
    assert_eq!(
        (r.status, r.json()),
        (200, json!({"status": "idle", "loaded": false}))
    );
    // Asked of the backend itself: nothing was loaded, so nothing was unloaded.
    assert_eq!(srv.fake.unload().await.unwrap(), Ok(false));
    srv.stop();
    assert_eq!(srv.exited(WAIT).await, Some(Ok(())));
    assert_eq!(unload_causes(&srv), Vec::<String>::new());
}

#[tokio::test]
async fn the_stop_logs_shutdown_only_when_it_unloaded() {
    // Idle: the stop has nothing to unload, so it writes no `unload` line.
    let mut s = Setup::new();
    s.fake_logs = true;
    let mut srv = TestServer::start(s).await;
    srv.stop();
    assert_eq!(srv.exited(WAIT).await, Some(Ok(())));
    assert_eq!(unload_causes(&srv), Vec::<String>::new());

    // Loaded: one, with the cause `shutdown`.
    let mut s = Setup::new();
    s.fake_logs = true;
    s.fake.start_loaded = true;
    let mut srv = TestServer::start(s).await;
    srv.stop();
    assert_eq!(srv.exited(WAIT).await, Some(Ok(())));
    assert_eq!(unload_causes(&srv), ["shutdown"]);
}

#[tokio::test]
async fn a_lifecycle_body_over_4_kib_is_413() {
    let srv = TestServer::start(Setup::new()).await;
    let big = format!(r#"{{"x": "{}"}}"#, "a".repeat(4_991));
    assert_eq!(big.len(), 5_000);
    for path in ["/v1/unload", "/v1/load"] {
        let r = post(&srv, path, big.as_bytes()).await;
        assert_eq!(r.status, 413, "{path}");
        let message = r.json()["error"]["message"].as_str().unwrap().to_string();
        assert!(message.contains("4 KiB"), "{message}");
    }
    assert!(srv.fake.loads().is_empty());
    // An empty body, and one under the limit, are served; the chat body limit is untouched.
    assert_eq!(post(&srv, "/v1/unload", b"").await.status, 200);
    assert_eq!(post(&srv, "/v1/unload", br#"{"x": 1}"#).await.status, 200);
    assert_eq!(post(&srv, CHAT, &chat_body(false)).await.status, 200);
}
