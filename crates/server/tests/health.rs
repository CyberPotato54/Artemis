//! `/health`.
mod common;

use common::{wait_until, Setup, TestServer};
use serde_json::{json, Value};
use server::backend::{Backend, Lifecycle};
use std::time::Duration;

const KEY: &str = "test-key-0123456789";
const FIELDS: [&str; 7] = [
    "service",
    "status",
    "version",
    "loaded",
    "model",
    "max_context",
    "api_key",
];

fn keys(v: &Value) -> Vec<&str> {
    let mut k: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    k.sort_unstable();
    k
}

#[tokio::test]
async fn h1_health_needs_no_auth() {
    let mut s = Setup::new();
    s.key = Some(KEY);
    let srv = TestServer::start(s).await;
    let bearer = format!("Bearer {KEY}");
    let none = srv.send("GET", "/health", &[], b"").await;
    let wrong = srv
        .send("GET", "/health", &[("Authorization", "Bearer wrong")], b"")
        .await;
    let right = srv
        .send("GET", "/health", &[("Authorization", &bearer)], b"")
        .await;
    for r in [&none, &wrong, &right] {
        assert_eq!(r.status, 200);
        assert_eq!(r.body, none.body, "the same body whatever the header says");
    }
    let body = none.json();
    let mut want = FIELDS.to_vec();
    want.sort_unstable();
    assert_eq!(keys(&body), want);
    assert_eq!(
        body,
        json!({
            "service": "artemis",
            "status": "idle",
            "version": env!("CARGO_PKG_VERSION"),
            "loaded": false,
            "model": "qwen3.6-35b-a3b",
            "max_context": 65536,
            "api_key": true,
        })
    );
    assert!(body["loaded"].is_boolean());
    assert!(body["max_context"].is_u64());
    // The field order is the documented one.
    let text = String::from_utf8(none.body.clone()).unwrap();
    let at: Vec<usize> = FIELDS
        .iter()
        .map(|f| text.find(&format!("\"{f}\"")).unwrap())
        .collect();
    assert!(at.windows(2).all(|w| w[0] < w[1]), "{text}");
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn h1_without_a_key_says_so() {
    let srv = TestServer::start(Setup::new()).await;
    let r = srv.send("GET", "/health", &[], b"").await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["api_key"], json!(false));
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn h1_ready_when_the_model_is_loaded() {
    let mut s = Setup::new();
    s.fake.start_loaded = true;
    let srv = TestServer::start(s).await;
    let r = srv.send("GET", "/health", &[], b"").await;
    assert_eq!(r.status, 200);
    let body = r.json();
    assert_eq!(body["status"], "ready");
    assert_eq!(body["loaded"], json!(true));
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn h1_loading_is_503_with_the_same_body() {
    let mut s = Setup::new();
    // Long enough that the load is still going whenever the test looks; it is never awaited.
    s.fake.load_time = Duration::from_secs(30);
    let srv = TestServer::start(s).await;
    let _loading = srv.fake.load();
    assert!(
        wait_until(Duration::from_secs(5), || srv.fake.state()
            == Lifecycle::Loading)
        .await
    );
    let r = srv.send("GET", "/health", &[], b"").await;
    assert_eq!(r.status, 503);
    let body = r.json();
    let mut want = FIELDS.to_vec();
    want.sort_unstable();
    assert_eq!(keys(&body), want);
    assert_eq!(body["status"], "loading");
    assert_eq!(body["loaded"], json!(false));
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn h2_the_service_name_changes_only_service() {
    let artemis = TestServer::start(Setup::new()).await;
    let mut s = Setup::new();
    s.config.health_service = "compat".into();
    let compat = TestServer::start(s).await;
    let a = artemis.send("GET", "/health", &[], b"").await;
    let b = compat.send("GET", "/health", &[], b"").await;
    assert_eq!((a.status, b.status), (200, 200));
    let raw = String::from_utf8(b.body.clone()).unwrap();
    assert!(
        has_service_compat(&raw),
        "the raw body must match \"service\"\\s*:\\s*\"compat\": {raw}"
    );
    let (mut a, mut b) = (a.json(), b.json());
    assert_eq!(a["service"], "artemis");
    assert_eq!(b["service"], "compat");
    a["service"] = json!("x");
    b["service"] = json!("x");
    assert_eq!(a, b, "every other field is as with artemis");
    artemis.shutdown().await.unwrap();
    compat.shutdown().await.unwrap();
}

/// `"service"\s*:\s*"compat"` without a regex crate.
fn has_service_compat(raw: &str) -> bool {
    let Some(at) = raw.find("\"service\"") else {
        return false;
    };
    let rest = raw[at + "\"service\"".len()..].trim_start();
    let Some(rest) = rest.strip_prefix(':') else {
        return false;
    };
    rest.trim_start().starts_with("\"compat\"")
}

#[tokio::test]
async fn h4_idle_loads_on_demand() {
    let srv = TestServer::start(Setup::new()).await;
    let r = srv.send("GET", "/health", &[], b"").await;
    assert_eq!(r.status, 200);
    let body = r.json();
    assert_eq!(
        (&body["status"], &body["loaded"]),
        (&json!("idle"), &json!(false))
    );
    assert!(srv.fake.loads().is_empty());

    let chat = json!({"messages": [{"role": "user", "content": "hi"}]}).to_string();
    let r = srv
        .send("POST", "/v1/chat/completions", &[], chat.as_bytes())
        .await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    assert_eq!(srv.fake.loads().len(), 1, "the request loaded the model");

    let body = srv.send("GET", "/health", &[], b"").await.json();
    assert_eq!(
        (&body["status"], &body["loaded"]),
        (&json!("ready"), &json!(true))
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn h3_health_during_a_load() {
    let mut s = Setup::new();
    s.fake.load_time = Duration::from_secs(3);
    let srv = TestServer::start(s).await;
    let chat = json!({"messages": [{"role": "user", "content": "hi"}]}).to_string();
    // The first request loads the model; `/health` is read while it does and again once it is done.
    let request = srv.send("POST", "/v1/chat/completions", &[], chat.as_bytes());
    let probe = async {
        assert!(
            wait_until(Duration::from_secs(5), || srv.fake.state()
                == Lifecycle::Loading)
            .await
        );
        let r = srv.send("GET", "/health", &[], b"").await;
        assert_eq!(r.status, 503);
        let body = r.json();
        let mut want = FIELDS.to_vec();
        want.sort_unstable();
        assert_eq!(keys(&body), want, "the same keys as when idle");
        assert_eq!(body["status"], "loading");
        assert_eq!(body["loaded"], json!(false));
        assert!(
            wait_until(Duration::from_secs(20), || srv.fake.state()
                == Lifecycle::Ready)
            .await
        );
        let r = srv.send("GET", "/health", &[], b"").await;
        assert_eq!(r.status, 200);
        assert_eq!(r.json()["status"], "ready");
    };
    let (r, ()) = tokio::join!(request, probe);
    assert_eq!(r.status, 200);
    srv.shutdown().await.unwrap();
}

/// Building a big prompt (two renders and a tokenize, seconds for a 16 MiB body) runs off the async workers, so
/// `/health` still answers "within milliseconds" while one is under way. On the test's single-threaded
/// runtime a render on the runtime itself would hold every probe until it finished.
#[tokio::test]
async fn health_answers_while_a_large_prompt_is_built() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let srv = TestServer::start(Setup::new()).await;
    let text = "word ".repeat(1 << 19);
    let body = json!({"model": "m", "max_tokens": 16,
        "messages": [{"role": "user", "content": text}]})
    .to_string();
    let done = AtomicBool::new(false);
    let counting = async {
        let r = srv
            .send("POST", "/v1/messages/count_tokens", &[], body.as_bytes())
            .await;
        done.store(true, Ordering::SeqCst);
        r
    };
    let probing = async {
        let (mut probes, mut worst) = (0, Duration::ZERO);
        while !done.load(Ordering::SeqCst) {
            let t = std::time::Instant::now();
            let r = srv.send("GET", "/health", &[], b"").await;
            worst = worst.max(t.elapsed());
            assert_eq!(r.status, 200);
            probes += 1;
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        (probes, worst)
    };
    let (counted, (probes, worst)) = tokio::join!(counting, probing);
    assert_eq!(counted.status, 200);
    assert!(probes >= 3, "only {probes} probes ran during the count");
    assert!(
        worst < Duration::from_secs(1),
        "/health took {worst:?} while a prompt was built"
    );
}
