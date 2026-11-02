//! Two cases: the oracle's request bodies over HTTP, and every request OpenCode and Claude Code were captured
//! sending, replayed against the server on the fake engine. The fixtures are `testdata/clients/` and
//! `testdata/template/requests/`.
mod common;

use common::{Setup, Sse, SseItem, TestServer, TESTDATA};
use serde_json::Value;
use server::fake::Script;
use std::time::Duration;

const KEY: &str = "test-key-for-t2";
const WAIT: Duration = Duration::from_secs(60);

/// Paths a client probed that are not part of the API: each is asserted to be a 404.
/// The capture recorded none: both clients sent only their generation requests.
const OUTSIDE_THE_API: &[&str] = &[];

/// The captured headers the harness sets itself.
const HARNESS_HEADERS: &[&str] = &[
    "content-length",
    "host",
    "transfer-encoding",
    "connection",
    "accept-encoding",
];

struct Captured {
    id: String,
    method: String,
    /// The path and query as recorded.
    target: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    json: Value,
}

/// Every `*.json` under `dir`, at any depth, sorted.
fn json_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let path = e.unwrap().path();
        if path.is_dir() {
            json_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "json") {
            out.push(path);
        }
    }
    out.sort();
}

/// Every fixture of `testdata/clients/**`, sorted: the headers without the harness's own, the auth replaced by the
/// test key.
fn captured() -> Vec<Captured> {
    let mut out = Vec::new();
    let root = std::path::PathBuf::from(format!("{TESTDATA}clients"));
    let mut files = Vec::new();
    json_files(&root, &mut files);
    for path in files {
        let id = path
            .strip_prefix(&root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let c: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let path_s = c["path"].as_str().unwrap().to_string();
        let query = c["query"].as_str().unwrap();
        let target = if query.is_empty() {
            path_s.clone()
        } else {
            format!("{path_s}?{query}")
        };
        let headers = c["headers"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(k, _)| !HARNESS_HEADERS.contains(&k.to_ascii_lowercase().as_str()))
            .map(|(k, v)| {
                let v = if k.eq_ignore_ascii_case("authorization") {
                    assert_eq!(v, "<masked>", "{id}");
                    format!("Bearer {KEY}")
                } else {
                    v.as_str().unwrap().to_string()
                };
                (k.clone(), v)
            })
            .collect();
        out.push(Captured {
            id,
            method: c["method"].as_str().unwrap().to_string(),
            target,
            path: path_s,
            headers,
            body: serde_json::to_vec(&c["body"]).unwrap(),
            json: c["body"].clone(),
        });
    }
    out
}

/// Reads a stream to its end: `[DONE]` for OpenAI's, `message_stop` for Anthropic's. Returns the events' names (the
/// data for an unnamed one).
async fn read_to_end(mut sse: Sse, id: &str) -> Vec<String> {
    let mut seen = Vec::new();
    while let Some(item) = sse.next(WAIT).await {
        if let SseItem::Event { name, data } = item {
            let label = name.unwrap_or_else(|| data.clone());
            let last = label == "[DONE]" || label == "message_stop" || label == "error";
            seen.push(label);
            if last {
                break;
            }
        }
    }
    let last = seen.last().map(String::as_str);
    assert!(
        matches!(last, Some("[DONE]") | Some("message_stop")),
        "{id}: the stream did not end cleanly: {seen:?}"
    );
    seen
}

#[tokio::test]
async fn t2_every_captured_request_is_served() {
    let mut setup = Setup::new();
    setup.key = Some(KEY);
    let ctx = setup.config.ctx as usize;
    let srv = TestServer::start(setup).await;
    let fixtures = captured();
    assert_eq!(fixtures.len(), 9, "four Claude Code and five OpenCode");
    let (mut generated, mut counted, mut outside) = (0, 0, Vec::new());
    for f in &fixtures {
        let headers: Vec<(&str, &str)> = f
            .headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let path = f.path.as_str();
        match (f.method.as_str(), path) {
            ("POST", "/v1/chat/completions") | ("POST", "/v1/messages") => {
                srv.fake.push(Script::text(common::tokenizer(), "ok"));
                let streams = f.json["stream"] == true;
                if streams {
                    let sse = srv.stream(&f.target, &headers, &f.body).await;
                    assert_eq!(sse.status(), 200, "{}", f.id);
                    let seen = read_to_end(sse, &f.id).await;
                    assert!(seen.len() > 2, "{}: {seen:?}", f.id);
                } else {
                    let r = srv.send("POST", &f.target, &headers, &f.body).await;
                    assert_eq!(
                        r.status,
                        200,
                        "{}: {}",
                        f.id,
                        String::from_utf8_lossy(&r.body)
                    );
                }
                generated += 1;
                // The prompt the fake got: rendered with the cut, and within the context.
                let (_, req) = srv.fake.requests().into_iter().last().unwrap();
                assert_eq!(srv.fake.requests().len(), generated, "{}", f.id);
                assert!(req.cut.is_some(), "{}: no cut", f.id);
                assert!(
                    !req.prompt.is_empty() && req.prompt.len() < ctx,
                    "{}: {} prompt tokens in {ctx}",
                    f.id,
                    req.prompt.len()
                );
                if path == "/v1/messages" {
                    // Claude Code also asks for the count before a turn; the same body counts.
                    let target = f
                        .target
                        .replacen("/v1/messages", "/v1/messages/count_tokens", 1);
                    let r = srv.send("POST", &target, &headers, &f.body).await;
                    assert_eq!(
                        r.status,
                        200,
                        "{}: {}",
                        f.id,
                        String::from_utf8_lossy(&r.body)
                    );
                    let n = r.json()["input_tokens"].as_u64().unwrap();
                    assert_eq!(n as usize, req.prompt.len(), "{}", f.id);
                    counted += 1;
                }
            }
            ("POST", "/v1/messages/count_tokens") => {
                let r = srv.send("POST", &f.target, &headers, &f.body).await;
                assert_eq!(r.status, 200, "{}", f.id);
                counted += 1;
            }
            ("GET", "/v1/models") => {
                assert_eq!(srv.send("GET", &f.target, &headers, b"").await.status, 200);
            }
            ("GET", p) if p.starts_with("/v1/models/") => {
                assert_eq!(srv.send("GET", &f.target, &headers, b"").await.status, 200);
            }
            // The health route needs no key: send it without one.
            ("GET", "/health") => {
                assert_eq!(srv.send("GET", &f.target, &[], b"").await.status, 200);
            }
            (_, p) => {
                assert!(
                    OUTSIDE_THE_API.contains(&p),
                    "{}: {} {p} is neither an API route nor a listed probe",
                    f.id,
                    f.method
                );
                let r = srv.send(&f.method, &f.target, &headers, &f.body).await;
                assert_eq!(r.status, 404, "{}: {} {p}", f.id, f.method);
                outside.push(p.to_string());
            }
        }
    }
    assert_eq!(generated, 9);
    assert_eq!(counted, 4, "each Claude Code body counts");
    assert_eq!(outside.len(), OUTSIDE_THE_API.len());
    // The 404 branch itself: a path no route has, with the key, answers 404.
    let r = srv
        .send(
            "GET",
            "/api/event_logging/batch",
            &[("authorization", &format!("Bearer {KEY}"))],
            b"",
        )
        .await;
    assert_eq!(r.status, 404);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn t1_oracle_fixtures_through_http() {
    let srv = TestServer::start(Setup::new()).await;
    let dir = format!("{TESTDATA}template/requests");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 18, "{paths:?}");
    for (n, path) in paths.iter().enumerate() {
        let id = path.file_stem().unwrap().to_string_lossy().into_owned();
        let body = std::fs::read(path).unwrap();
        let raw = std::fs::read(format!("{TESTDATA}template/prompts/{id}.tokens")).unwrap();
        assert_eq!(raw.len() % 4, 0, "{id}: a .tokens file is whole u32s");
        let want: Vec<u32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        srv.fake.push(Script::text(common::tokenizer(), "ok"));
        let r = srv.send("POST", "/v1/chat/completions", &[], &body).await;
        assert_eq!(r.status, 200, "{id}: {}", String::from_utf8_lossy(&r.body));
        let runs = srv.fake.requests();
        assert_eq!(runs.len(), n + 1, "{id}");
        let got = &runs[n].1.prompt;
        let at = got
            .iter()
            .zip(&want)
            .position(|(a, b)| a != b)
            .unwrap_or(got.len().min(want.len()));
        assert!(
            *got == want,
            "{id}: first token difference at {at} ({} got, {} want)",
            got.len(),
            want.len()
        );
    }
    srv.shutdown().await.unwrap();
}
