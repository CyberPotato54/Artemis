//! `/v1/models` and the 404, 405 and 413 answers.
mod common;

use common::{Setup, TestServer};
use serde_json::{json, Value};
use server::ids::{iso_utc, unix_now};

const LIMIT: usize = 16 << 20;

#[tokio::test]
async fn e1_models_shapes() {
    let srv = TestServer::start(Setup::new()).await;
    let r = srv.send("GET", "/v1/models", &[], b"").await;
    assert_eq!(r.status, 200);
    let ct = r.headers["content-type"].to_str().unwrap();
    assert!(ct.starts_with("application/json"), "{ct}");
    let body = r.json();
    let created = body["data"][0]["created"].as_u64().unwrap();
    assert!(
        created <= unix_now() && unix_now() - created < 120,
        "{created}"
    );
    let m = "qwen3.6-35b-a3b";
    let entry = json!({
        "id": m, "object": "model", "created": created, "owned_by": "artemis", "type": "model",
        "display_name": m, "created_at": iso_utc(created), "context_length": 65536, "max_model_len": 65536,
    });
    assert_eq!(
        body,
        json!({"object": "list", "data": [entry], "has_more": false, "first_id": m, "last_id": m})
    );
    // Any id is accepted and echoed; `created` is the process start in both.
    let r = srv.send("GET", "/v1/models/claude-x", &[], b"").await;
    assert_eq!(r.status, 200);
    let one = r.json();
    assert_eq!(
        one,
        json!({
            "id": "claude-x", "object": "model", "created": created, "owned_by": "artemis",
            "type": "model", "display_name": "claude-x", "created_at": iso_utc(created),
            "context_length": 65536, "max_model_len": 65536,
        })
    );
    // A percent-encoded id is decoded; one with a slash is no route.
    let r = srv.send("GET", "/v1/models/a%2Eb", &[], b"").await;
    assert_eq!(r.json()["id"], "a.b");
    // An id that is not UTF-8 once decoded is a JSON 400 in the OpenAI shape, not axum's plain text.
    let r = srv.send("GET", "/v1/models/%FF", &[], b"").await;
    assert_eq!(r.status, 400);
    check_shape("/v1/models/%FF", &r.json(), "invalid_request_error");
    assert!(r.headers["content-type"]
        .to_str()
        .unwrap()
        .starts_with("application/json"));
    assert_eq!(
        srv.send("GET", "/v1/models/a/b", &[], b"").await.status,
        404
    );
    // A query string is ignored, and no CORS header is sent.
    let r = srv
        .send("GET", "/v1/models?limit=1&beta=true", &[], b"")
        .await;
    assert_eq!(r.status, 200);
    for (name, _) in r.headers.iter() {
        assert!(!name.as_str().starts_with("access-control-"), "{name}");
    }
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn models_carry_the_context_length() {
    let srv = TestServer::start(Setup::new()).await;
    let list = srv.send("GET", "/v1/models", &[], b"").await.json();
    assert_eq!(list["data"][0]["context_length"], 65536);
    assert_eq!(list["data"][0]["max_model_len"], 65536);
    let one = srv
        .send("GET", "/v1/models/anything", &[], b"")
        .await
        .json();
    assert_eq!(one["context_length"], 65536);
    assert_eq!(one["max_model_len"], 65536);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn models_follow_the_configured_context() {
    let mut s = Setup::new();
    s.config.ctx = 4096;
    let srv = TestServer::start(s).await;
    let list = srv.send("GET", "/v1/models", &[], b"").await.json();
    assert_eq!(list["data"][0]["context_length"], 4096);
    assert_eq!(list["data"][0]["max_model_len"], 4096);
    let one = srv
        .send("GET", "/v1/models/anything", &[], b"")
        .await
        .json();
    assert_eq!(one["context_length"], 4096);
    assert_eq!(one["max_model_len"], 4096);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn e1_models_follow_the_configured_id() {
    let mut s = Setup::new();
    s.config.model_id = "my-model".into();
    let srv = TestServer::start(s).await;
    let body = srv.send("GET", "/v1/models", &[], b"").await.json();
    assert_eq!(body["data"][0]["id"], "my-model");
    assert_eq!(body["first_id"], "my-model");
    assert_eq!(
        srv.send("GET", "/health", &[], b"").await.json()["model"],
        "my-model"
    );
    srv.shutdown().await.unwrap();
}

/// The error's type and message, in the shape the path decides.
fn check_shape(path: &str, body: &Value, want_type: &str) {
    if path == "/v1/messages" || path.starts_with("/v1/messages/") {
        assert_eq!(body["type"], "error", "{path}: {body}");
        assert_eq!(body["error"]["type"], want_type, "{path}: {body}");
        assert!(body["error"]["message"].is_string());
        assert!(body["error"].get("param").is_none() && body["error"].get("code").is_none());
    } else {
        assert!(body.get("type").is_none(), "{path}: {body}");
        assert_eq!(body["error"]["type"], want_type, "{path}: {body}");
        assert!(body["error"]["message"].is_string());
        assert!(body["error"].get("param").is_some() && body["error"].get("code").is_some());
    }
}

#[tokio::test]
async fn e1_404_405_413_in_each_paths_shape() {
    let srv = TestServer::start(Setup::new()).await;
    let r = srv.send("GET", "/v1/chat/completions", &[], b"").await;
    assert_eq!(r.status, 405);
    check_shape("/v1/chat/completions", &r.json(), "invalid_request_error");
    let r = srv.send("GET", "/v1/messages", &[], b"").await;
    assert_eq!(r.status, 405);
    check_shape("/v1/messages", &r.json(), "invalid_request_error");
    for (method, path) in [
        ("POST", "/health"),
        ("DELETE", "/v1/models"),
        ("GET", "/v1/load"),
    ] {
        let r = srv.send(method, path, &[], b"").await;
        assert_eq!(r.status, 405, "{method} {path}");
        check_shape(path, &r.json(), "invalid_request_error");
    }
    let r = srv.send("POST", "/v1/nope", &[], b"{}").await;
    assert_eq!(r.status, 404);
    check_shape("/v1/nope", &r.json(), "not_found_error");
    let r = srv.send("POST", "/v1/messages/nope", &[], b"{}").await;
    assert_eq!(r.status, 404);
    check_shape("/v1/messages/nope", &r.json(), "not_found_error");
    // `/v1/messagesx` is not under /v1/messages.
    let r = srv.send("POST", "/v1/messagesx", &[], b"{}").await;
    assert_eq!(r.status, 404);
    check_shape("/v1/messagesx", &r.json(), "not_found_error");
    let r = srv.send("GET", "/elsewhere", &[], b"").await;
    assert_eq!(r.status, 404);
    check_shape("/elsewhere", &r.json(), "not_found_error");

    // One byte over the limit to each POST route (16 MiB; 4 KiB for the two lifecycle routes): 413 in its shape, once
    // with a Content-Length and once chunked.
    for (path, limit) in [
        ("/v1/chat/completions", LIMIT),
        ("/v1/messages", LIMIT),
        ("/v1/messages/count_tokens", LIMIT),
        ("/v1/load", server::app::LIFECYCLE_BODY),
        ("/v1/unload", server::app::LIFECYCLE_BODY),
    ] {
        let big = vec![b' '; limit + 1];
        let want = if path.starts_with("/v1/messages") {
            "request_too_large"
        } else {
            "invalid_request_error"
        };
        let r = srv.send("POST", path, &[], &big).await;
        assert_eq!(r.status, 413, "{path} with a length");
        check_shape(path, &r.json(), want);
        let r = srv.send_chunked("POST", path, &[], &big).await;
        assert_eq!(r.status, 413, "{path} chunked");
        check_shape(path, &r.json(), want);
    }
    // Exactly 16 MiB is read.
    let ok = vec![b' '; LIMIT];
    let r = srv.send("POST", "/v1/chat/completions", &[], &ok).await;
    assert_ne!(r.status, 413);
    let r = srv.send_chunked("POST", "/v1/messages", &[], &ok).await;
    assert_ne!(r.status, 413);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn serve_returns_why_it_exited() {
    use server::app::Exit;
    use std::time::Duration;
    let mut srv = TestServer::start(Setup::new()).await;
    assert!(srv.exited(Duration::from_millis(50)).await.is_none());
    srv.app.exit(Exit::DeviceLostTwice);
    srv.app.exit(Exit::EngineGone); // the first reason stays
    let got = srv.exited(Duration::from_secs(10)).await;
    assert_eq!(got, Some(Err(Exit::DeviceLostTwice)));
}
