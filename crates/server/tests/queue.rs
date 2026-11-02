//! The HTTP-side queue. Arrival order is made deterministic by waiting for
//! `TestServer::waiting()` to rise after each request, never by sleeping.
mod common;

use common::{tokenizer, wait_until, Resp, Setup, TestServer};
use serde_json::{json, Value};
use server::fake::Script;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::task::JoinHandle;

const CHAT: &str = "/v1/chat/completions";
const WAIT: Duration = Duration::from_secs(20);

/// A request whose user message is `tag`, so the fake's record says which request ran when.
fn body(tag: &str) -> Vec<u8> {
    json!({"messages": [{"role": "user", "content": tag}], "reasoning_effort": "none"})
        .to_string()
        .into_bytes()
}

/// A script that holds the engine for `d` before it answers.
fn slow(d: Duration) -> Script {
    Script {
        prefill: d,
        ..Script::text(tokenizer(), "slow")
    }
}

fn spawn_post(srv: &Arc<TestServer>, tag: &str) -> JoinHandle<Resp> {
    let srv = srv.clone();
    let body = body(tag);
    tokio::spawn(async move { srv.send("POST", CHAT, &[], &body).await })
}

/// Waits for the engine to be running request number `n` and the queue to be empty: the running request's ticket
/// leaves the queue just after the engine records it.
async fn running(srv: &TestServer, n: usize) {
    assert!(
        wait_until(WAIT, || srv.fake.requests().len() == n
            && srv.waiting() == 0)
        .await,
        "the engine never ran request {n}"
    );
}

async fn waiting_is(srv: &TestServer, n: usize) {
    assert!(
        wait_until(WAIT, || srv.waiting() == n).await,
        "waiting() is {}, expected {n}",
        srv.waiting()
    );
}

/// The user messages of the requests the fake ran, in order.
fn ran(srv: &TestServer) -> Vec<String> {
    srv.fake
        .requests()
        .iter()
        .map(|(_, r)| {
            let text = tokenizer().decode(&r.prompt);
            let after = text.split("<|im_start|>user\n").last().unwrap();
            after.split("<|im_end|>").next().unwrap().to_string()
        })
        .collect()
}

#[tokio::test]
async fn h7_requests_run_in_arrival_order() {
    let srv = Arc::new(TestServer::start(Setup::new()).await);
    srv.fake.push(slow(Duration::from_millis(1500)));
    let finished = Arc::new(Mutex::new(Vec::new()));
    let mut tasks = Vec::new();
    for (i, tag) in ["r1", "r2", "r3"].into_iter().enumerate() {
        let (s, order) = (srv.clone(), finished.clone());
        let body = body(tag);
        tasks.push(tokio::spawn(async move {
            let r = s.send("POST", CHAT, &[], &body).await;
            order.lock().unwrap().push(tag);
            r
        }));
        if i == 0 {
            running(&srv, 1).await;
        } else {
            // Request i+1 is the i-th to wait.
            waiting_is(&srv, i).await;
        }
    }
    for t in tasks {
        assert_eq!(t.await.unwrap().status, 200);
    }
    assert_eq!(ran(&srv), ["r1", "r2", "r3"]);
    assert_eq!(*finished.lock().unwrap(), ["r1", "r2", "r3"]);
    assert_eq!(srv.waiting(), 0);
    Arc::into_inner(srv).unwrap().shutdown().await.unwrap();
}

#[tokio::test]
async fn h7_a_waiting_request_times_out_and_never_runs() {
    let mut s = Setup::new();
    s.config.queue_timeout_s = 1;
    let srv = Arc::new(TestServer::start(s).await);
    srv.fake.push(slow(Duration::from_secs(5)));
    let first = spawn_post(&srv, "r1");
    running(&srv, 1).await;
    let second = spawn_post(&srv, "r2");
    waiting_is(&srv, 1).await;
    let r = second.await.unwrap();
    assert_eq!(r.status, 503);
    let e = &r.json()["error"];
    assert_eq!(e["type"], "service_unavailable");
    assert_eq!(e["message"], "timed out waiting in the queue");
    assert_eq!(srv.waiting(), 0, "the timed-out request left the queue");
    assert_eq!(first.await.unwrap().status, 200);
    // The engine meets the dead request after the first one and skips it: the next one runs.
    let third = spawn_post(&srv, "r3").await.unwrap();
    assert_eq!(third.status, 200);
    assert_eq!(ran(&srv), ["r1", "r3"]);
    Arc::into_inner(srv).unwrap().shutdown().await.unwrap();
}

#[tokio::test]
async fn h7_the_65th_waiting_request_is_503_at_once() {
    let srv = Arc::new(TestServer::start(Setup::new()).await);
    // Long enough that filling 64 places, one observed at a time, cannot overrun it on a loaded machine.
    srv.fake.push(slow(Duration::from_secs(10)));
    let mut tasks = vec![spawn_post(&srv, "run")];
    running(&srv, 1).await;
    for n in 1..=64 {
        tasks.push(spawn_post(&srv, &format!("w{n}")));
        waiting_is(&srv, n).await;
    }
    let r = srv.send("POST", CHAT, &[], &body("one too many")).await;
    assert_eq!(r.status, 503);
    let e = &r.json()["error"];
    assert_eq!(e["type"], "service_unavailable");
    assert_eq!(e["message"], "the queue is full");
    assert_eq!(srv.waiting(), 64);
    for t in tasks {
        assert_eq!(t.await.unwrap().status, 200);
    }
    let order = ran(&srv);
    assert_eq!(order.len(), 65, "the refused request never ran");
    assert_eq!(order[0], "run");
    let want: Vec<String> = (1..=64).map(|n| format!("w{n}")).collect();
    assert_eq!(order[1..], want[..]);
    Arc::into_inner(srv).unwrap().shutdown().await.unwrap();
}

#[tokio::test]
async fn queue_a_request_dropped_while_waiting_never_runs() {
    let srv = Arc::new(TestServer::start(Setup::new()).await);
    srv.fake.push(slow(Duration::from_secs(4)));
    let first = spawn_post(&srv, "r1");
    running(&srv, 1).await;
    // The second request's connection is aborted after 1.5 s, which is long after it queued and well before the
    // engine is free.
    let dropped = {
        let srv = srv.clone();
        tokio::spawn(async move {
            srv.send_then_drop("POST", CHAT, &[], &body("r2"), Duration::from_millis(1500))
                .await;
        })
    };
    waiting_is(&srv, 1).await;
    dropped.await.unwrap();
    waiting_is(&srv, 0).await;
    assert_eq!(
        srv.fake.requests().len(),
        1,
        "still the first request alone"
    );
    assert_eq!(first.await.unwrap().status, 200);
    let third = spawn_post(&srv, "r3").await.unwrap();
    assert_eq!(third.status, 200);
    assert_eq!(ran(&srv), ["r1", "r3"]);
    let b: Value = third.json();
    assert_eq!(b["choices"][0]["finish_reason"], "stop");
    Arc::into_inner(srv).unwrap().shutdown().await.unwrap();
}
