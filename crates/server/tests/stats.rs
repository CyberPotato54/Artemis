//! `GET /v1/stats`, the `stats` heartbeat and the `stats` lines. The route reads the backend's
//! latest copy and never asks the engine; the heartbeat asks, waits a little, and writes the line from the copy.
mod common;

use common::{tokenizer, wait_until, Setup, TestServer};
use serde_json::{json, Value};
use server::backend::{Backend, BackendStats, Lifecycle};
use server::fake::{FakeConfig, Script};
use std::sync::Arc;
use std::time::{Duration, Instant};

const CHAT: &str = "/v1/chat/completions";
const STATS: &str = "/v1/stats";
const KEY: &str = "test-key-0123456789"; // gitleaks:allow
const WAIT: Duration = Duration::from_secs(15);

fn body() -> Vec<u8> {
    json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "none"})
        .to_string()
        .into_bytes()
}

/// A loaded fake and a heartbeat that is not going to fire during the test.
fn setup() -> Setup {
    let mut s = Setup::new();
    s.fake.start_loaded = true;
    s
}

/// Every line the server has logged, oldest first.
fn all_lines(srv: &TestServer) -> Vec<Value> {
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
        .map(|l| serde_json::from_str(&l).unwrap_or_else(|e| panic!("a bad log line ({e}): {l}")))
        .collect()
}

/// The lifecycle lines' events, oldest first.
fn events(srv: &TestServer) -> Vec<String> {
    all_lines(srv)
        .iter()
        .filter_map(|l| l["event"].as_str().map(String::from))
        .collect()
}

fn lines_of(srv: &TestServer, event: &str) -> Vec<Value> {
    all_lines(srv)
        .into_iter()
        .filter(|l| l["event"] == event)
        .collect()
}

async fn get_stats(srv: &TestServer) -> Value {
    let r = srv.send("GET", STATS, &[], b"").await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    r.json()
}

/// Asks `/v1/stats` until `done` holds of its answer or `WAIT` has passed; the last answer, and whether it held.
async fn stats_until(srv: &TestServer, done: impl Fn(&Value) -> bool) -> (bool, Value) {
    let end = Instant::now() + WAIT;
    loop {
        let v = get_stats(srv).await;
        if done(&v) || Instant::now() >= end {
            return (done(&v), v);
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn stats_answers_at_once_while_the_engine_is_stuck() {
    let srv = Arc::new(TestServer::start(setup()).await);
    srv.fake.push(Script {
        before_started: Duration::from_secs(30),
        ..Script::text(tokenizer(), "stuck")
    });
    let post = {
        let srv = srv.clone();
        tokio::spawn(async move { srv.send("POST", CHAT, &[], &body()).await })
    };
    assert!(
        wait_until(WAIT, || srv.app.oldest_in_flight().is_some()).await,
        "the request never arrived"
    );
    let began = Instant::now();
    let v = get_stats(&srv).await;
    // Not waiting on the engine is the property: it is stuck for 30 s, so an answer that waited would take that long.
    // The bound is wide enough for a loaded machine.
    assert!(
        began.elapsed() < Duration::from_secs(2),
        "took {:?}",
        began.elapsed()
    );
    assert_eq!(v["progress"]["pending"], 1, "{v}");
    assert!(v["progress"]["ticks"].is_u64(), "{v}");
    post.abort();
}

#[tokio::test]
async fn stats_has_the_documented_shape_before_any_answer() {
    let srv = TestServer::start(setup()).await;
    let v = get_stats(&srv).await;
    let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "age_s",
            "log_write_failures",
            "object",
            "progress",
            "state",
            "stats",
            "uptime_s"
        ]
    );
    assert_eq!(v["object"], "stats");
    assert_eq!(v["state"], "ready");
    assert!(v["uptime_s"].is_u64(), "{v}");
    assert_eq!(v["log_write_failures"], 0);
    // The engine has not answered `stats` yet, so there is no copy and no age.
    assert!(v["stats"].is_null() && v["age_s"].is_null(), "{v}");
    assert_eq!(v["progress"]["pending"], 0);
    // The route never asks the engine, and a 200 writes no request line.
    assert_eq!(srv.fake.latest_stats(), None);
    assert!(
        all_lines(&srv).iter().all(|l| l.get("event").is_some()),
        "a request line was written"
    );
}

#[tokio::test]
async fn stats_needs_the_key() {
    let mut s = setup();
    s.key = Some(KEY);
    let srv = TestServer::start(s).await;
    let r = srv.send("GET", STATS, &[], b"").await;
    assert_eq!(r.status, 401);
    assert_eq!(r.json()["error"]["type"], "authentication_error");
    let line = all_lines(&srv)
        .into_iter()
        .find(|l| l.get("event").is_none())
        .expect("the refusal left a line");
    assert_eq!(
        (&line["endpoint"], &line["status"]),
        (&json!(STATS), &json!(401))
    );

    let bearer = format!("Bearer {KEY}");
    let r = srv
        .send("GET", STATS, &[("Authorization", &bearer)], b"")
        .await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["object"], "stats");
}

#[tokio::test]
async fn the_copy_is_refreshed_after_each_request() {
    let srv = TestServer::start(setup()).await;
    srv.fake.set_stats(BackendStats {
        loads: 1,
        ..BackendStats::default()
    });
    let r = srv.send("POST", CHAT, &[], &body()).await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    // The refresh is fire-and-forget, so it lands just after the response.
    let (ok, last) = stats_until(&srv, |v| v["stats"]["loads"] == 1).await;
    assert!(ok, "the copy never arrived: {last}");
    assert!(last["age_s"].as_f64().unwrap() < 5.0, "{last}");
}

#[tokio::test]
async fn a_request_that_ends_inside_begin_still_refreshes_the_copy() {
    // A lazy load that an unload takes: the request never starts, so no exchange ever holds its place.
    let mut s = Setup::new();
    s.fake = FakeConfig {
        load_time: Duration::from_millis(500),
        ..FakeConfig::default()
    };
    let srv = Arc::new(TestServer::start(s).await);
    srv.fake.set_stats(BackendStats {
        loads: 7,
        ..BackendStats::default()
    });
    let post = {
        let srv = srv.clone();
        tokio::spawn(async move { srv.send("POST", CHAT, &[], &body()).await })
    };
    assert!(wait_until(WAIT, || srv.fake.state() == Lifecycle::Loading).await);
    assert_eq!(srv.send("POST", "/v1/unload", &[], b"").await.status, 200);
    assert_eq!(post.await.unwrap().status, 503);
    let (ok, last) = stats_until(&srv, |v| v["stats"]["loads"] == 7).await;
    assert!(ok, "no refresh after the request ended in `begin`: {last}");
}

#[tokio::test]
async fn the_heartbeat_writes_stats_lines() {
    let mut s = setup();
    s.tuning.stats_every = Duration::from_millis(100);
    // The log's `time` has whole-second resolution, so the lower bound is measured on the test's clock, from before the
    // server started: `serve` begins after `began`, so its first tick cannot come before `began + 100 ms`.
    let began = Instant::now();
    let srv = TestServer::start(s).await;
    srv.fake.set_stats(BackendStats {
        loads: 3,
        ..BackendStats::default()
    });
    // No `stats` line follows `start` at once: the first comes one period on.
    assert!(wait_until(WAIT, || !lines_of(&srv, "stats").is_empty()).await);
    assert!(
        began.elapsed() >= Duration::from_millis(100),
        "the first stats line came after {:?}",
        began.elapsed()
    );
    assert_eq!(events(&srv)[..2], ["start", "stats"]);
    // Three come within 5 s of the start with a 100 ms period: the period is honoured, with fifty times its length to
    // spare for a loaded machine. (The lower bound on the first line is checked above.)
    assert!(
        wait_until(
            Duration::from_secs(5).saturating_sub(began.elapsed()),
            || lines_of(&srv, "stats").len() >= 3
        )
        .await,
        "{:?} after {:?}",
        events(&srv),
        began.elapsed()
    );
    let lines = lines_of(&srv, "stats");
    for l in &lines {
        assert!(l["uptime_s"].is_u64(), "{l}");
        assert_eq!(l["log_write_failures"], 0);
    }
    // The heartbeat asked the engine, so a later line carries what it answered.
    let last = lines.last().unwrap();
    assert_eq!(last["stats"]["loads"], 3, "{last}");
    assert!(last["age_s"].as_f64().unwrap() < 5.0, "{last}");
}

#[tokio::test]
async fn the_heartbeat_still_writes_while_the_engine_is_stuck() {
    let mut s = setup();
    s.tuning.stats_every = Duration::from_millis(100);
    let srv = Arc::new(TestServer::start(s).await);
    srv.fake.push(Script {
        before_started: Duration::from_secs(30),
        ..Script::text(tokenizer(), "stuck")
    });
    let _post = {
        let srv = srv.clone();
        tokio::spawn(async move { srv.send("POST", CHAT, &[], &body()).await })
    };
    assert!(
        wait_until(WAIT, || srv.app.oldest_in_flight().is_some()).await,
        "the request never arrived"
    );
    // A beat that began before the engine was stuck may still be answered, and the first beat can come before the
    // request does, which leaves a copy. So the lines to look at are the ones after those: the third new line began its
    // beat with the engine stuck, and the heartbeat waits at most 2 s for it, then writes from the copy it has.
    let before = lines_of(&srv, "stats").len();
    // Each beat waits for the engine at most 2 s after its 100 ms period: three of them take about 6.3 s, and a beat
    // that waited far past its cap would not make this bound.
    let bound = 3 * (Duration::from_millis(100) + Duration::from_secs(2)) + Duration::from_secs(3);
    assert!(
        wait_until(bound, || lines_of(&srv, "stats").len() >= before + 3).await,
        "{:?}",
        events(&srv)
    );
    let line = &lines_of(&srv, "stats")[before + 2];
    // The copy and its age are there together, or both are not; an age is never negative.
    assert_eq!(line["stats"].is_null(), line["age_s"].is_null(), "{line}");
    if !line["stats"].is_null() {
        assert!(line["age_s"].as_f64().unwrap() >= 0.0, "{line}");
    }
}

#[tokio::test]
async fn the_heartbeat_ends_before_exit() {
    let mut s = setup();
    s.tuning.stats_every = Duration::from_millis(100);
    let mut srv = TestServer::start(s).await;
    assert!(wait_until(WAIT, || !lines_of(&srv, "stats").is_empty()).await);
    srv.stop();
    assert_eq!(srv.exited(WAIT).await, Some(Ok(())));
    // Several periods on, nothing follows the exit.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let events = events(&srv);
    assert_eq!(
        events.last().map(String::as_str),
        Some("exit"),
        "{events:?}"
    );
    // One `stats` line precedes the `exit`, written by `serve` itself.
    assert_eq!(events[events.len() - 2], "stats", "{events:?}");
}

#[tokio::test]
async fn a_stop_writes_one_stats_line_and_it_follows_the_unload() {
    // `stats` goes before `exit`, and none follows the shutdown's `unload` besides: that one is it.
    let mut srv = TestServer::start(setup()).await;
    srv.stop();
    assert_eq!(srv.exited(WAIT).await, Some(Ok(())));
    assert_eq!(events(&srv), ["start", "unload", "stats", "exit"]);
}
