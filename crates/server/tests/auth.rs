//! Auth and the Host check.
mod common;

use common::{Setup, TestServer};
use serde_json::Value;

const KEY: &str = "test-key-0123456789";

fn keyed() -> Setup {
    let mut s = Setup::new();
    s.key = Some(KEY);
    s
}

/// The error's `type`, from either body shape.
fn error_type(v: &Value) -> &str {
    v.get("error")
        .and_then(|e| e.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("")
}

#[tokio::test]
async fn a1_missing_or_wrong_key_is_401_before_the_body() {
    let srv = TestServer::start(keyed()).await;
    let bearer_wrong = [("Authorization", "Bearer wrong")];
    let key_wrong = [("x-api-key", "wrong")];
    let none: [(&str, &str); 0] = [];
    let cases: [&[(&str, &str)]; 3] = [&none, &bearer_wrong, &key_wrong];
    for path in ["/v1/chat/completions", "/v1/messages"] {
        for headers in cases {
            for body in [&b"{}"[..], &b"{not json"[..]] {
                let r = srv.send("POST", path, headers, body).await;
                assert_eq!(r.status, 401, "{path} {headers:?}");
                let j = r.json();
                assert_eq!(error_type(&j), "authentication_error", "{j}");
                if path == "/v1/messages" {
                    assert_eq!(j["type"], "error", "the Anthropic shape: {j}");
                } else {
                    assert!(j["error"].get("message").is_some(), "the OpenAI shape: {j}");
                    assert!(j.get("type").is_none(), "{j}");
                }
            }
        }
    }
    // The other /v1 paths need the key too, the lifecycle ones included.
    for (method, path) in [
        ("POST", "/v1/messages/count_tokens"),
        ("POST", "/v1/load"),
        ("POST", "/v1/unload"),
        ("GET", "/v1/models"),
        ("GET", "/v1/models/x"),
        ("POST", "/v1/nope"),
    ] {
        let r = srv.send(method, path, &[], b"").await;
        assert_eq!(r.status, 401, "{method} {path}");
    }
    assert!(
        srv.fake.loads().is_empty(),
        "an unauthorized /v1/load must not load"
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a1_the_error_never_carries_the_key() {
    let srv = TestServer::start(keyed()).await;
    let bearer = "Bearer a-wrong-secret-ABC";
    let r = srv
        .send("POST", "/v1/messages", &[("Authorization", bearer)], b"{}")
        .await;
    let text = String::from_utf8(r.body).unwrap();
    assert!(
        !text.contains(KEY) && !text.contains("a-wrong-secret-ABC"),
        "{text}"
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a2_bearer_and_x_api_key_are_accepted() {
    // The key file is written "<key>\r\n".
    let srv = TestServer::start(keyed()).await;
    let bearer = format!("Bearer {KEY}");
    let lower = format!("bearer {KEY}");
    let upper = format!("BEARER  {KEY}");
    for headers in [
        vec![("Authorization", bearer.as_str())],
        vec![("Authorization", lower.as_str())],
        vec![("Authorization", upper.as_str())],
        vec![("x-api-key", KEY)],
        // Claude Code sends one of the two, with its version headers.
        vec![
            ("x-api-key", KEY),
            ("anthropic-version", "2023-06-01"),
            (
                "anthropic-beta",
                "claude-code-20250219,interleaved-thinking-2025-05-14",
            ),
        ],
    ] {
        let r = srv.send("GET", "/v1/models", &headers, b"").await;
        assert_eq!(r.status, 200, "{headers:?}");
    }
    // A wrong scheme or a key with extra text is no key.
    for bad in [
        format!("Basic {KEY}"),
        format!("Bearer {KEY}x"),
        KEY.to_string(),
    ] {
        let r = srv
            .send("GET", "/v1/models", &[("Authorization", &bad)], b"")
            .await;
        assert_eq!(r.status, 401, "{bad}");
    }
    // A bare scheme and an empty `x-api-key` are no key.
    for bad in [
        vec![("Authorization", "Bearer")],
        vec![("Authorization", "Bearer ")],
        vec![("x-api-key", "")],
        vec![("Authorization", "Bearer"), ("x-api-key", "")],
    ] {
        let r = srv.send("GET", "/v1/models", &bad, b"").await;
        assert_eq!(r.status, 401, "{bad:?}");
    }
    // Either header is enough: a wrong one beside a right one does not reject it.
    for headers in [
        vec![("Authorization", "Bearer wrong"), ("x-api-key", KEY)],
        vec![("Authorization", bearer.as_str()), ("x-api-key", "wrong")],
    ] {
        let r = srv.send("GET", "/v1/models", &headers, b"").await;
        assert_eq!(r.status, 200, "{headers:?}");
    }
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a2_the_clients_requests_pass_auth() {
    // What the capture shows: Claude Code posts /v1/messages?beta=true with a Bearer or an x-api-key, OpenCode posts
    // /v1/chat/completions with a Bearer. Past auth they are not 401, 403 or 404.
    let srv = TestServer::start(keyed()).await;
    let bearer = format!("Bearer {KEY}");
    let claude = [
        ("Authorization", bearer.as_str()),
        ("anthropic-version", "2023-06-01"),
        ("anthropic-beta", "claude-code-20250219"),
        ("x-app", "cli"),
    ];
    let claude_key = [("x-api-key", KEY), ("anthropic-version", "2023-06-01")];
    let opencode = [("Authorization", bearer.as_str())];
    for (path, headers) in [
        ("/v1/messages?beta=true", &claude[..]),
        ("/v1/messages?beta=true", &claude_key[..]),
        ("/v1/messages/count_tokens?beta=true", &claude[..]),
        ("/v1/chat/completions", &opencode[..]),
    ] {
        let r = srv.send("POST", path, headers, b"{}").await;
        assert!(
            ![401, 403, 404, 405].contains(&r.status),
            "{path}: {}",
            r.status
        );
    }
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a3_host_check_without_a_key() {
    let srv = TestServer::start(Setup::new()).await;
    for path in ["/health", "/v1/models"] {
        let r = srv.send("GET", path, &[("Host", "example.com")], b"").await;
        assert_eq!(r.status, 403, "{path}");
        assert_eq!(r.json()["error"]["type"], "permission_error");
        for host in ["localhost:8090", "127.0.0.1:8090", "[::1]:8090"] {
            let r = srv.send("GET", path, &[("Host", host)], b"").await;
            assert_eq!(r.status, 200, "{path} {host}");
        }
    }
    // The 403 is in the path's shape and never echoes the Host.
    let r = srv
        .send("POST", "/v1/messages", &[("Host", "evil.example")], b"{}")
        .await;
    assert_eq!(r.status, 403);
    let j = r.json();
    assert_eq!(
        (j["type"].as_str(), error_type(&j)),
        (Some("error"), "permission_error")
    );
    assert!(!String::from_utf8(r.body).unwrap().contains("evil.example"));
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a3_host_variants() {
    let srv = TestServer::start(Setup::new()).await;
    for host in [
        "LOCALHOST:8090",
        "[::1]",
        "Localhost",
        "127.0.0.1",
        "[::1]:1",
    ] {
        let r = srv.send("GET", "/health", &[("Host", host)], b"").await;
        assert_eq!(r.status, 200, "{host}");
    }
    for host in [
        "localhost.",
        "localhost.:8090",
        "127.0.0.1.example.com",
        "127.0.0.1:8090@evil.example",
        "localhost:abc",
        "[::2]",
        "::1",
        "",
    ] {
        // "" leaves the header out (the harness's rule).
        for (method, path) in [
            ("GET", "/health"),
            ("POST", "/v1/nope"),
            ("GET", "/v1/chat/completions"),
        ] {
            let r = srv.send(method, path, &[("Host", host)], b"").await;
            assert_eq!(r.status, 403, "{method} {path} Host {host:?}");
        }
    }
    // With a good Host the same requests reach their 404 and 405.
    assert_eq!(srv.send("POST", "/v1/nope", &[], b"").await.status, 404);
    assert_eq!(
        srv.send("GET", "/v1/chat/completions", &[], b"")
            .await
            .status,
        405
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a3_host_may_be_the_bind_address() {
    let mut s = Setup::new();
    s.bind_host = "127.0.0.2";
    let srv = TestServer::start(s).await;
    let port = srv.addr.port();
    let ok = format!("127.0.0.2:{port}");
    let other = format!("127.0.0.3:{port}");
    assert_eq!(
        srv.send("GET", "/health", &[("Host", &ok)], b"")
            .await
            .status,
        200
    );
    assert_eq!(
        srv.send("GET", "/health", &[("Host", "127.0.0.2")], b"")
            .await
            .status,
        200
    );
    assert_eq!(
        srv.send("GET", "/health", &[("Host", &other)], b"")
            .await
            .status,
        403
    );
    assert_eq!(
        srv.send("GET", "/health", &[("Host", "127.0.0.3")], b"")
            .await
            .status,
        403
    );
    // The fixed list still passes.
    assert_eq!(
        srv.send("GET", "/health", &[("Host", "localhost")], b"")
            .await
            .status,
        200
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a3_with_a_key_any_host_passes() {
    let srv = TestServer::start(keyed()).await;
    let bearer = format!("Bearer {KEY}");
    for host in ["example.com", "192.0.2.7:9", "localhost."] {
        assert_eq!(
            srv.send("GET", "/health", &[("Host", host)], b"")
                .await
                .status,
            200
        );
        let r = srv
            .send(
                "GET",
                "/v1/models",
                &[("Host", host), ("Authorization", &bearer)],
                b"",
            )
            .await;
        assert_eq!(r.status, 200, "{host}");
    }
    srv.shutdown().await.unwrap();
}

// ---- the Origin, Sec-Fetch-Site and Content-Type checks ----

const CHAT: &str = "/v1/chat/completions";

fn chat_body() -> Vec<u8> {
    serde_json::json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "none"})
        .to_string()
        .into_bytes()
}

/// The request lines in the server's log: the ones without an `event`.
fn request_lines(srv: &TestServer) -> Vec<Value> {
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
        .filter(|l| l.get("event").is_none())
        .collect()
}

fn with_origins(origins: &[&str]) -> Setup {
    let mut s = Setup::new();
    s.config.allowed_origins = origins.iter().map(|o| o.to_string()).collect();
    s
}

#[tokio::test]
async fn a_foreign_origin_is_403_and_logged() {
    let srv = TestServer::start(Setup::new()).await;
    let r = srv
        .send(
            "POST",
            CHAT,
            &[("origin", "https://evil.example")],
            &chat_body(),
        )
        .await;
    assert_eq!(r.status, 403);
    let j = r.json();
    assert_eq!(error_type(&j), "permission_error", "{j}");
    assert!(
        j["error"]["message"]
            .as_str()
            .unwrap()
            .contains("allowed_origins"),
        "{j}"
    );
    assert!(!String::from_utf8(r.body).unwrap().contains("evil.example"));
    let lines = request_lines(&srv);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert_eq!(lines[0]["status"], 403);
    assert_eq!(lines[0]["endpoint"], CHAT);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn health_with_a_foreign_origin_is_403() {
    let srv = TestServer::start(Setup::new()).await;
    let r = srv
        .send("GET", "/health", &[("origin", "https://evil.example")], b"")
        .await;
    assert_eq!(r.status, 403);
    assert_eq!(error_type(&r.json()), "permission_error");
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn no_origin_behaves_as_today() {
    let srv = TestServer::start(Setup::new()).await;
    assert_eq!(srv.send("GET", "/health", &[], b"").await.status, 200);
    assert_eq!(srv.send("POST", CHAT, &[], &chat_body()).await.status, 200);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn an_allowed_origin_passes() {
    let srv = TestServer::start(with_origins(&["http://localhost:3000"])).await;
    for origin in ["http://localhost:3000", "HTTP://LOCALHOST:3000/"] {
        let r = srv.send("GET", "/health", &[("origin", origin)], b"").await;
        assert_eq!(r.status, 200, "{origin}");
    }
    for origin in ["http://localhost:3001", "https://localhost:3000"] {
        let r = srv.send("GET", "/health", &[("origin", origin)], b"").await;
        assert_eq!(r.status, 403, "{origin}");
    }
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_listed_non_http_origin_passes_and_an_unlisted_one_is_403() {
    // Webview, app and extension clients send these; a web page cannot.
    let srv = TestServer::start(with_origins(&[
        "vscode-webview://abc",
        "app://local",
        "chrome-extension://extid",
    ]))
    .await;
    for origin in [
        "vscode-webview://abc",
        "VSCODE-WEBVIEW://ABC",
        "app://local/",
        "chrome-extension://extid",
    ] {
        let r = srv.send("GET", "/health", &[("origin", origin)], b"").await;
        assert_eq!(r.status, 200, "{origin}");
    }
    for origin in [
        "vscode-webview://other",
        "app://local:8080",
        "https://local",
        "file://",
        "moz-extension://extid",
    ] {
        let r = srv.send("GET", "/health", &[("origin", origin)], b"").await;
        assert_eq!(r.status, 403, "{origin}");
        assert_eq!(error_type(&r.json()), "permission_error");
    }
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_null_origin_is_403_even_with_an_allowlist() {
    // Sandboxed frames and some redirects send `Origin: null`; no entry can match it.
    let srv = TestServer::start(with_origins(&["http://localhost:3000"])).await;
    for origin in ["null", "NULL"] {
        let r = srv.send("GET", "/health", &[("origin", origin)], b"").await;
        assert_eq!(r.status, 403, "{origin}");
        assert_eq!(error_type(&r.json()), "permission_error");
    }
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn sec_fetch_site_cross_and_same_site_are_403() {
    let srv = TestServer::start(Setup::new()).await;
    for site in ["cross-site", "same-site"] {
        let r = srv
            .send("GET", "/health", &[("sec-fetch-site", site)], b"")
            .await;
        assert_eq!(r.status, 403, "{site}");
        let j = r.json();
        assert_eq!(error_type(&j), "permission_error", "{j}");
        assert!(
            j["error"]["message"]
                .as_str()
                .unwrap()
                .contains("allowed_origins"),
            "{j}"
        );
    }
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn sec_fetch_site_same_origin_and_none_pass() {
    let srv = TestServer::start(Setup::new()).await;
    for site in ["same-origin", "none"] {
        let r = srv
            .send("GET", "/health", &[("sec-fetch-site", site)], b"")
            .await;
        assert_eq!(r.status, 200, "{site}");
    }
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn an_allowed_origin_lifts_a_cross_site_fetch() {
    let srv = TestServer::start(with_origins(&["http://localhost:3000"])).await;
    let r = srv
        .send(
            "GET",
            "/health",
            &[
                ("origin", "http://localhost:3000"),
                ("sec-fetch-site", "same-site"),
            ],
            b"",
        )
        .await;
    assert_eq!(r.status, 200);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_key_does_not_lift_the_origin_check() {
    let srv = TestServer::start(keyed()).await;
    let bearer = format!("Bearer {KEY}");
    let r = srv
        .send(
            "GET",
            "/v1/models",
            &[
                ("authorization", &bearer),
                ("origin", "https://evil.example"),
            ],
            b"",
        )
        .await;
    assert_eq!(r.status, 403);
    assert_eq!(error_type(&r.json()), "permission_error");
    let r = srv
        .send("GET", "/v1/models", &[("authorization", &bearer)], b"")
        .await;
    assert_eq!(r.status, 200);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn the_content_type_check_runs_before_the_key_check() {
    let srv = TestServer::start(keyed()).await;
    // A wrong key and a wrong type: the 415 comes first, as docs/server.md says.
    let r = srv
        .send(
            "POST",
            CHAT,
            &[
                ("authorization", "Bearer wrong"),
                ("content-type", "text/plain"),
            ],
            &chat_body(),
        )
        .await;
    assert_eq!(r.status, 415);
    // A wrong key with the right type is the 401.
    let r = srv
        .send(
            "POST",
            CHAT,
            &[("authorization", "Bearer wrong")],
            &chat_body(),
        )
        .await;
    assert_eq!(r.status, 401);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_text_plain_body_is_415() {
    let srv = TestServer::start(Setup::new()).await;
    let r = srv
        .send(
            "POST",
            CHAT,
            &[("content-type", "text/plain")],
            &chat_body(),
        )
        .await;
    assert_eq!(r.status, 415);
    let j = r.json();
    assert_eq!(error_type(&j), "invalid_request_error", "{j}");
    assert!(
        j["error"]["message"]
            .as_str()
            .unwrap()
            .contains("application/json"),
        "{j}"
    );
    for ct in [
        "application/json; charset=utf-8",
        "Application/JSON",
        "application/vnd.api+json",
    ] {
        let r = srv
            .send("POST", CHAT, &[("content-type", ct)], &chat_body())
            .await;
        assert_eq!(r.status, 200, "{ct}");
    }
    // A body with an empty content type is not JSON either.
    let r = srv
        .send("POST", "/v1/unload", &[("content-type", "")], b"x=y")
        .await;
    assert_eq!(r.status, 415);
    // Nor is one with no Content-Type header at all.
    let r = srv
        .send("POST", "/v1/unload", &[("no-content-type", "")], b"x=y")
        .await;
    assert_eq!(r.status, 415);
    let r = srv
        .send("POST", CHAT, &[("no-content-type", "")], &chat_body())
        .await;
    assert_eq!(r.status, 415);
    // A chunked body (no Content-Length) is a body: `text/plain` is refused there too.
    let r = srv
        .send_chunked(
            "POST",
            CHAT,
            &[("content-type", "text/plain")],
            &chat_body(),
        )
        .await;
    assert_eq!(r.status, 415);
    let r = srv
        .send_chunked(
            "POST",
            CHAT,
            &[("content-type", "application/json")],
            &chat_body(),
        )
        .await;
    assert_eq!(r.status, 200);
    // No body: load and unload pass with any type or none.
    let r = srv.send("POST", "/v1/unload", &[], b"").await;
    assert_eq!(r.status, 200);
    let r = srv
        .send("POST", "/v1/unload", &[("content-type", "text/plain")], b"")
        .await;
    assert_eq!(r.status, 200);
    // A form post is the classic cross-site body.
    let r = srv
        .send(
            "POST",
            "/v1/unload",
            &[("content-type", "application/x-www-form-urlencoded")],
            b"x=y",
        )
        .await;
    assert_eq!(r.status, 415);
    srv.shutdown().await.unwrap();
}
