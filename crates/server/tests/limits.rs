//! The accept loop's limits: the header and body timeouts, the connection cap, the peer address on a
//! refusal, and that none of them cuts a healthy long response.
mod common;

use common::{tokenizer, wait_until, Setup, SseItem, TestServer};
use serde_json::{json, Value};
use server::fake::Script;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tokio::net::TcpStream;

const CHAT: &str = "/v1/chat/completions";
const WAIT: Duration = Duration::from_secs(20);

/// `tokio::io::{AsyncReadExt, AsyncWriteExt}` for a `TcpStream`, over its readiness methods: the `io-util` feature would
/// add a package to `Cargo.lock`.
trait Io {
    async fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize>;
    async fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()>;
}

impl Io for TcpStream {
    async fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        loop {
            self.readable().await?;
            match self.try_read(buf) {
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                done => return done,
            }
        }
    }

    async fn write_all(&mut self, mut bytes: &[u8]) -> std::io::Result<()> {
        while !bytes.is_empty() {
            self.writable().await?;
            match self.try_write(bytes) {
                Ok(n) => bytes = &bytes[n..],
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

async fn raw(addr: SocketAddr) -> TcpStream {
    TcpStream::connect(addr).await.unwrap()
}

/// A `GET /health` request, keep-alive.
fn get_health(addr: SocketAddr) -> String {
    format!(
        "GET /health HTTP/1.1\r\nHost: 127.0.0.1:{}\r\n\r\n",
        addr.port()
    )
}

/// Reads one response with a `Content-Length` body: its status and its body. `None` when the stream ends or `within`
/// passes before the response is complete.
async fn read_response(s: &mut TcpStream, within: Duration) -> Option<(u16, String)> {
    let end = Instant::now() + within;
    let mut buf = Vec::new();
    loop {
        if let Some(head_end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
            let len = head
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    k.eq_ignore_ascii_case("content-length")
                        .then(|| v.trim().parse::<usize>().ok())?
                })
                .unwrap_or(0);
            if buf.len() >= head_end + 4 + len {
                let status = head.split(' ').nth(1)?.parse().ok()?;
                let body =
                    String::from_utf8_lossy(&buf[head_end + 4..head_end + 4 + len]).to_string();
                return Some((status, body));
            }
        }
        let left = end.saturating_duration_since(Instant::now());
        let mut chunk = [0u8; 4096];
        match tokio::time::timeout(left, s.read(&mut chunk)).await {
            Ok(Ok(n)) if n > 0 => buf.extend_from_slice(&chunk[..n]),
            _ => return None,
        }
    }
}

/// True when the next read returns 0 bytes (the server closed) within `within`; false on data, an error other than a
/// reset, or the time passing.
async fn closed_within(s: &mut TcpStream, within: Duration) -> bool {
    let mut chunk = [0u8; 1024];
    match tokio::time::timeout(within, s.read(&mut chunk)).await {
        Ok(Ok(0)) => true,
        Ok(Err(e)) => e.kind() == std::io::ErrorKind::ConnectionReset,
        _ => false,
    }
}

/// The request lines in the server's log (the ones without an `event`).
fn request_lines(srv: &TestServer) -> Vec<Value> {
    let mut lines = Vec::new();
    for e in std::fs::read_dir(&srv.log_dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|e| e == "jsonl") {
            for l in std::fs::read_to_string(p).unwrap().lines() {
                let v: Value = serde_json::from_str(l).unwrap();
                if v.get("event").is_none() {
                    lines.push(v);
                }
            }
        }
    }
    lines
}

#[tokio::test]
async fn a_connection_that_never_finishes_its_headers_is_closed() {
    let mut setup = Setup::new();
    setup.tuning.header_timeout = Duration::from_millis(300);
    let srv = TestServer::start(setup).await;
    let mut s = raw(srv.addr).await;
    s.write_all(
        format!(
            "POST {CHAT} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\n",
            srv.addr.port()
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    assert!(
        closed_within(&mut s, Duration::from_secs(2)).await,
        "the connection should be closed with no response"
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_body_that_stalls_is_408() {
    let mut setup = Setup::new();
    setup.tuning.body_timeout = Duration::from_millis(300);
    let srv = TestServer::start(setup).await;
    let mut s = raw(srv.addr).await;
    s.write_all(
        format!(
            "POST {CHAT} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\ncontent-type: application/json\r\ncontent-length: 100\r\n\r\n0123456789",
            srv.addr.port()
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    let (status, body) = read_response(&mut s, Duration::from_secs(3))
        .await
        .expect("a 408 response");
    assert_eq!(status, 408, "{body}");
    let j: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(j["error"]["type"], "invalid_request_error", "{j}");
    assert!(
        j["error"]["message"]
            .as_str()
            .unwrap()
            .contains("did not arrive"),
        "{j}"
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn connections_past_the_cap_wait_for_a_free_one() {
    let mut setup = Setup::new();
    setup.tuning.max_connections = 2;
    setup.tuning.header_timeout = Duration::from_secs(5);
    let srv = TestServer::start(setup).await;
    // Both are served and kept alive, so both hold a slot.
    let mut first = raw(srv.addr).await;
    let mut second = raw(srv.addr).await;
    for s in [&mut first, &mut second] {
        s.write_all(get_health(srv.addr).as_bytes()).await.unwrap();
        let (status, _) = read_response(s, Duration::from_secs(2)).await.unwrap();
        assert_eq!(status, 200);
    }
    let mut third = raw(srv.addr).await;
    third
        .write_all(get_health(srv.addr).as_bytes())
        .await
        .unwrap();
    assert!(
        read_response(&mut third, Duration::from_millis(300))
            .await
            .is_none(),
        "the third connection must wait while two are open"
    );
    drop(first);
    let (status, _) = read_response(&mut third, Duration::from_secs(2))
        .await
        .expect("served once a slot is free");
    assert_eq!(status, 200);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn an_idle_keep_alive_connection_is_closed_cleanly() {
    let mut setup = Setup::new();
    setup.tuning.header_timeout = Duration::from_millis(300);
    let srv = TestServer::start(setup).await;
    let mut s = raw(srv.addr).await;
    s.write_all(get_health(srv.addr).as_bytes()).await.unwrap();
    let (status, _) = read_response(&mut s, Duration::from_secs(2)).await.unwrap();
    assert_eq!(status, 200);
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(
        closed_within(&mut s, Duration::from_secs(2)).await,
        "an idle connection is closed with no partial response"
    );
    let mut fresh = raw(srv.addr).await;
    fresh
        .write_all(get_health(srv.addr).as_bytes())
        .await
        .unwrap();
    let (status, _) = read_response(&mut fresh, Duration::from_secs(2))
        .await
        .unwrap();
    assert_eq!(status, 200);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_401_line_carries_the_peer_address() {
    let mut setup = Setup::new();
    setup.key = Some("test-key-0123456789");
    let srv = TestServer::start(setup).await;
    let r = srv
        .send("POST", CHAT, &[("Authorization", "Bearer wrong")], b"{}")
        .await;
    assert_eq!(r.status, 401);
    let r = srv
        .send(
            "GET",
            "/v1/models",
            &[("Authorization", "Bearer test-key-0123456789")],
            b"",
        )
        .await;
    assert_eq!(r.status, 200);
    // Only the refusal writes a line for these two requests (a 200 /v1/models writes none), so a 200 with a line is a
    // chat request.
    srv.fake.push(Script::text(tokenizer(), "ok"));
    let ok = json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "none"});
    let r = srv
        .send(
            "POST",
            CHAT,
            &[("Authorization", "Bearer test-key-0123456789")],
            ok.to_string().as_bytes(),
        )
        .await;
    assert_eq!(r.status, 200);
    let lines = || request_lines(&srv);
    assert!(wait_until(WAIT, || lines().len() == 2).await);
    let lines = lines();
    let refused = lines.iter().find(|l| l["status"] == 401).unwrap();
    assert_eq!(refused["peer"], "127.0.0.1", "{refused}");
    let served = lines.iter().find(|l| l["status"] == 200).unwrap();
    assert!(served["peer"].is_null(), "{served}");
    assert!(served.as_object().unwrap().contains_key("peer"), "{served}");
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_403_and_a_415_line_carry_the_peer_address() {
    let srv = TestServer::start(Setup::new()).await;
    let r = srv
        .send("POST", CHAT, &[("host", "evil.example")], b"{}")
        .await;
    assert_eq!(r.status, 403);
    let r = srv
        .send("POST", CHAT, &[("content-type", "text/plain")], b"hi")
        .await;
    assert_eq!(r.status, 415);
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 2).await);
    for l in request_lines(&srv) {
        assert_eq!(l["peer"], "127.0.0.1", "{l}");
    }
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_refusal_closes_its_connection() {
    let srv = TestServer::start(Setup::new()).await;
    let mut s = raw(srv.addr).await;
    // A Host the server does not serve: refused at the door, 403.
    s.write_all(
        b"GET /v1/models HTTP/1.1
Host: evil.example

",
    )
    .await
    .unwrap();
    let mut seen = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match tokio::time::timeout(Duration::from_secs(2), s.read(&mut chunk)).await {
            Ok(Ok(0)) => break,
            Ok(Ok(n)) => seen.extend_from_slice(&chunk[..n]),
            other => panic!("the connection stayed open after a refusal: {other:?}"),
        }
    }
    let text = String::from_utf8_lossy(&seen).to_ascii_lowercase();
    assert!(text.starts_with("http/1.1 403"), "{text}");
    assert!(text.contains("connection: close"), "{text}");
    srv.shutdown().await.unwrap();
}

/// A reply of `n` words at `per_token` each.
fn slow(n: usize, per_token: Duration) -> Script {
    Script {
        per_token,
        ..Script::text(
            tokenizer(),
            &format!("x\n</think>\n\n{}", "word ".repeat(n)),
        )
    }
}

#[tokio::test]
async fn a_stream_longer_than_the_header_timeout_is_not_cut() {
    let mut setup = Setup::new();
    setup.tuning.header_timeout = Duration::from_millis(300);
    setup.tuning.body_timeout = Duration::from_millis(300);
    let srv = TestServer::start(setup).await;
    srv.fake.push(slow(30, Duration::from_millis(50)));
    let body = json!({"messages": [{"role": "user", "content": "hi"}], "stream": true, "reasoning_effort": "none"});
    let started = Instant::now();
    let mut sse = srv.stream(CHAT, &[], body.to_string().as_bytes()).await;
    assert_eq!(sse.status(), 200);
    let mut done = false;
    while let Some(item) = sse.next(WAIT).await {
        if matches!(&item, SseItem::Event { data, .. } if data == "[DONE]") {
            done = true;
            break;
        }
    }
    assert!(done, "the stream must end with [DONE]");
    assert!(
        started.elapsed() > Duration::from_millis(1200),
        "the stream should have outlasted the header timeout several times over: {:?}",
        started.elapsed()
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_long_prefill_is_not_cut_by_the_timeouts() {
    let mut setup = Setup::new();
    setup.tuning.header_timeout = Duration::from_millis(300);
    setup.tuning.body_timeout = Duration::from_millis(300);
    let srv = TestServer::start(setup).await;
    srv.fake.push(Script {
        prefill: Duration::from_millis(1500),
        ..Script::text(tokenizer(), "done")
    });
    let body = json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "none"});
    let r = srv
        .send("POST", CHAT, &[], body.to_string().as_bytes())
        .await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    srv.shutdown().await.unwrap();
}

#[test]
fn the_default_cap_leaves_room_for_a_full_queue() {
    // One running request and a full queue are 65 connections; idle keep-alive connections come on top.
    assert!(server::Tuning::default().max_connections > server::app::QUEUE_LIMIT + 1);
}
