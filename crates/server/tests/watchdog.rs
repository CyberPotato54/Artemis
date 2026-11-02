//! The stall watchdog over the fake engine: work that stops moving for `stall_s` writes a
//! `stall` line and an `exit` line and halts the process with 4, without waiting for the stuck engine or `serve`'s stop.
//! The test server's `halt` records the code instead of exiting. Timing is shrunk: `stall_s` 1 and a 50 ms tick.
mod common;

use common::{tokenizer, wait_until, Resp, Setup, TestServer};
use serde_json::{json, Value};
use server::fake::Script;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

const CHAT: &str = "/v1/chat/completions";
/// How long a test waits for something to happen. An event is polled for and the wait ends when it comes, so this only
/// matters to a test that fails, or to a machine so loaded that a 1 s watchdog is late: the timing checks that mean
/// something are the lower bounds and the "must not happen" assertions, which do not use it.
const WAIT: Duration = Duration::from_secs(15);

fn body() -> Vec<u8> {
    json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "none"})
        .to_string()
        .into_bytes()
}

/// A loaded fake, `stall_s` as given and a 50 ms watchdog tick.
fn setup(stall_s: u64) -> Setup {
    let mut s = Setup::new();
    s.config.stall_s = stall_s;
    s.tuning.watchdog_tick = Duration::from_millis(50);
    s.fake.start_loaded = true;
    s
}

/// A request the engine takes `d` to start.
fn stuck(d: Duration) -> Script {
    Script {
        before_started: d,
        ..Script::text(tokenizer(), "stuck")
    }
}

fn spawn_post(srv: &Arc<TestServer>) -> JoinHandle<Resp> {
    let srv = srv.clone();
    tokio::spawn(async move { srv.send("POST", CHAT, &[], &body()).await })
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

/// The lifecycle lines of `event`.
fn lines_of(srv: &TestServer, event: &str) -> Vec<Value> {
    all_lines(srv)
        .into_iter()
        .filter(|l| l["event"] == event)
        .collect()
}

fn events(srv: &TestServer) -> Vec<String> {
    all_lines(srv)
        .iter()
        .filter_map(|l| l["event"].as_str().map(String::from))
        .collect()
}

async fn halted(srv: &TestServer, within: Duration) -> bool {
    wait_until(within, || !srv.halts().is_empty()).await
}

#[tokio::test]
async fn a_request_that_never_starts_stalls_and_halts_with_4() {
    let srv = Arc::new(TestServer::start(setup(1)).await);
    srv.fake.push(stuck(Duration::from_secs(30)));
    let post = spawn_post(&srv);
    assert!(halted(&srv, WAIT).await, "no halt in {WAIT:?}");
    assert_eq!(srv.halts(), [4]);

    // The stall, its stats, then the exit, and nothing else after the start.
    assert_eq!(events(&srv), ["start", "stall", "stats", "exit"]);
    let stall = &lines_of(&srv, "stall")[0];
    assert_eq!(stall["doing"], "request");
    assert_eq!(stall["bound_s"], 1);
    // The bound less the watchdog's slack (it decides a stall up to 50 ms early, never late).
    assert!(stall["quiet_s"].as_f64().unwrap() >= 0.95, "{stall}");
    assert!(stall["ticks"].is_u64(), "{stall}");
    assert_eq!(lines_of(&srv, "exit")[0]["reason"], "stall");
    let id = stall["request"]
        .as_str()
        .expect("the stall names its request");

    // The request never started, so the stop's `abort_all` answers it, and its line carries the id the stall named.
    srv.stop();
    let resp = post.await.unwrap();
    assert_eq!(resp.status, 503);
    let line = {
        let mut found = None;
        let ok = wait_until(WAIT, || {
            found = all_lines(&srv).into_iter().find(|l| l["id"] == id);
            found.is_some()
        })
        .await;
        assert!(ok, "no request line with id {id}");
        found.unwrap()
    };
    assert_eq!(line["status"], 503);
    assert_eq!(line["error_message"], "the model was unloaded");
}

#[tokio::test]
async fn a_slow_request_that_keeps_moving_does_not_stall() {
    // Three seconds in all against a two-second bound, a token every 300 ms: a loaded machine has to starve the worker
    // for most of the bound before this stalls.
    let srv = Arc::new(TestServer::start(setup(2)).await);
    let mut script = Script::text(tokenizer(), &"hello ".repeat(10));
    script.tokens.truncate(10);
    assert_eq!(script.tokens.len(), 10);
    script.per_token = Duration::from_millis(300);
    srv.fake.push(script);
    let resp = spawn_post(&srv).await.unwrap();
    assert_eq!(resp.status, 200, "{}", String::from_utf8_lossy(&resp.body));
    assert!(srv.halts().is_empty());
    assert!(lines_of(&srv, "stall").is_empty());
}

#[tokio::test]
async fn a_long_load_gets_the_load_bound() {
    // A four-second load under a ten-second load bound, and a two-second request bound that would have stopped it.
    let mut s = setup(2);
    s.fake.start_loaded = false;
    s.fake.load_time = Duration::from_secs(4);
    s.config.stall_load_s = 10;
    let srv = Arc::new(TestServer::start(s).await);
    let resp = spawn_post(&srv).await.unwrap();
    assert_eq!(resp.status, 200, "{}", String::from_utf8_lossy(&resp.body));
    assert!(srv.halts().is_empty());
    assert!(lines_of(&srv, "stall").is_empty());

    // A five-second load under a one-second load bound is a stall, named a load.
    let mut s = setup(60);
    s.fake.start_loaded = false;
    // Long enough that the load is still running when a late watchdog looks.
    s.fake.load_time = Duration::from_secs(30);
    s.config.stall_load_s = 1;
    let srv = Arc::new(TestServer::start(s).await);
    let _post = spawn_post(&srv);
    assert!(halted(&srv, WAIT).await, "no halt in {WAIT:?}");
    assert_eq!(srv.halts(), [4]);
    let stall = &lines_of(&srv, "stall")[0];
    assert_eq!(
        (&stall["doing"], &stall["bound_s"]),
        (&json!("load"), &json!(1))
    );
}

#[tokio::test]
async fn an_idle_server_never_stalls() {
    let srv = TestServer::start(setup(1)).await;
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert!(srv.halts().is_empty());
    assert!(lines_of(&srv, "stall").is_empty());
}

#[tokio::test]
async fn stall_s_zero_never_halts() {
    let srv = Arc::new(TestServer::start(setup(0)).await);
    srv.fake.push(stuck(Duration::from_secs(3)));
    let _post = spawn_post(&srv);
    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert!(srv.halts().is_empty());
    assert!(lines_of(&srv, "stall").is_empty());
}

#[tokio::test]
async fn the_halt_does_not_wait_for_a_stuck_backend() {
    let srv = Arc::new(TestServer::start(setup(1)).await);
    srv.fake.push(stuck(Duration::from_secs(30)));
    let _post = spawn_post(&srv);
    // The bound is the documented one (`stall_s` and about 2 s for the lines), and it is about the stuck work, so it
    // runs from the request reaching the server's side, not from the test's start: a loaded machine can be slow to take
    // the POST.
    assert!(
        wait_until(WAIT, || srv.app.oldest_in_flight().is_some()).await,
        "the request never arrived"
    );
    let stuck_since = Instant::now();
    assert!(halted(&srv, WAIT).await, "no halt in {WAIT:?}");
    assert!(
        stuck_since.elapsed() < Duration::from_millis(3500),
        "the halt came {:?} after the request",
        stuck_since.elapsed()
    );
    // `serve` was never asked to stop, and an unload would have waited on the worker: neither happened.
    assert!(lines_of(&srv, "unload").is_empty(), "{:?}", events(&srv));
}

#[tokio::test]
async fn steady_arrivals_do_not_hide_a_stuck_worker() {
    let srv = Arc::new(TestServer::start(setup(1)).await);
    srv.fake.push(stuck(Duration::from_secs(30)));
    let _stuck = spawn_post(&srv);
    // New requests keep arriving behind it every 300 ms. An arrival is not movement.
    let mut arrivals = Vec::new();
    let end = Instant::now() + WAIT;
    while srv.halts().is_empty() && Instant::now() < end {
        arrivals.push(spawn_post(&srv));
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    assert_eq!(
        srv.halts(),
        [4],
        "no halt within {WAIT:?} of steady arrivals"
    );
}

#[tokio::test]
async fn a_normal_stop_writes_no_stall_line() {
    // The engine is held for 5 s by a request that is in flight when `serve` stops. The stop has to wait for the engine,
    // so it runs past the two seconds at which the watchdog would have fired: it must be gone by then. (Two seconds, not
    // one, so that a loaded machine delaying the stop cannot let a legitimate stall come first.)
    let srv = Arc::new(TestServer::start(setup(2)).await);
    srv.fake.push(stuck(Duration::from_secs(5)));
    let post = spawn_post(&srv);
    assert!(
        wait_until(WAIT, || srv.app.oldest_in_flight().is_some()).await,
        "the request never arrived"
    );
    srv.stop();
    assert_eq!(post.await.unwrap().status, 503);
    let mut srv = Arc::try_unwrap(srv).ok().expect("the post task is done");
    let result = srv.exited(WAIT * 2).await;
    assert_eq!(result, Some(Ok(())));
    assert!(srv.halts().is_empty());
    let events = events(&srv);
    assert!(!events.contains(&"stall".to_string()), "{events:?}");
    assert_eq!(
        events.last().map(String::as_str),
        Some("exit"),
        "{events:?}"
    );
    assert_eq!(lines_of(&srv, "exit")[0]["reason"], "shutdown");
}

#[tokio::test]
async fn a_stop_that_begins_at_once_cannot_cancel_the_halt() {
    // The stall is decided and its lines are being written when the stop begins. The stop aborts the watchdog task, but
    // the halt is in a closure it started and can no longer cancel: the recorder still gets 4, and `serve`, which sees the
    // halt under way, writes no exit line of its own.
    let srv = Arc::new(TestServer::start(setup(1)).await);
    srv.fake.push(stuck(Duration::from_secs(30)));
    let post = spawn_post(&srv);
    assert!(
        wait_until(WAIT, || !lines_of(&srv, "stall").is_empty()).await,
        "no stall line in {WAIT:?}"
    );
    srv.stop();
    assert!(halted(&srv, WAIT).await, "no halt in {WAIT:?}");
    assert_eq!(srv.halts(), [4]);
    // The stop answers the request that never started.
    assert_eq!(post.await.unwrap().status, 503);
    let mut srv = Arc::try_unwrap(srv).ok().expect("the post task is done");
    assert_eq!(srv.exited(WAIT * 2).await, Some(Ok(())));
    let exits = lines_of(&srv, "exit");
    assert_eq!(exits.len(), 1, "{:?}", events(&srv));
    assert_eq!(exits[0]["reason"], "stall");
    assert_eq!(events(&srv).last().map(String::as_str), Some("exit"));
}
