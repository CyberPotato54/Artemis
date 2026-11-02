//! The driver's HTTP/1 client: one hyper connection per call over a `tokio::net::TcpStream`, modelled on
//! `crates/server/tests/common/mod.rs` (copied, not shared: `common` is a per-binary test module). Every request
//! carries the key as `x-api-key` and as `Authorization: Bearer`, in headers marked sensitive. The key is never in an
//! error, a log or a verdict: no message here names a header value.
use anyhow::{anyhow, bail, Context, Result};
use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::client::conn::http1::{handshake, SendRequest};
use hyper::header::{HeaderMap, HeaderValue};
use hyper_util::rt::TokioIo;
use serde_json::Value;
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use tokio::task::JoinHandle;

/// How long a connect, or a wait for headers, or a whole non-streaming reply, may take: the original Test-Messages
/// script's own bound.
const REPLY: Duration = Duration::from_secs(600);
const CONNECT: Duration = Duration::from_secs(10);

#[derive(Clone)]
pub struct Client {
    host: String,
    port: u16,
    x_api_key: HeaderValue,
    bearer: HeaderValue,
    timeout: Duration,
}

/// A whole reply. `sent_at` is taken just before the request is written, `headers_at` when its headers arrived and
/// `done_at` when its body ended.
pub struct Resp {
    pub status: u16,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
    pub sent_at: Instant,
    pub headers_at: Instant,
    pub done_at: Instant,
}

impl Resp {
    pub fn json(&self) -> Result<Value> {
        serde_json::from_slice(&self.body)
            .with_context(|| format!("status {}: the body is not JSON", self.status))
    }
}

type Open = (
    SendRequest<Full<Bytes>>,
    JoinHandle<()>,
    hyper::Request<Full<Bytes>>,
);

impl Client {
    /// A client for `host:port` (the caller has checked them, see `check_base_url`) holding `key`.
    pub fn new(host: &str, port: u16, key: &str) -> Result<Client> {
        let value = |v: String| -> Result<HeaderValue> {
            let mut v = HeaderValue::from_str(&v)
                .map_err(|_| anyhow!("the key is not a valid header value"))?;
            v.set_sensitive(true);
            Ok(v)
        };
        Ok(Client {
            host: host.to_string(),
            port,
            x_api_key: value(key.to_string())?,
            bearer: value(format!("Bearer {key}"))?,
            timeout: REPLY,
        })
    }

    async fn open(&self, method: &str, path: &str, body: Option<&Value>) -> Result<Open> {
        let stream =
            tokio::time::timeout(CONNECT, TcpStream::connect((self.host.as_str(), self.port)))
                .await
                .map_err(|_| anyhow!("connecting to {}:{} timed out", self.host, self.port))?
                .with_context(|| format!("cannot connect to {}:{}", self.host, self.port))?;
        let (sender, conn) = handshake(TokioIo::new(stream))
            .await
            .context("the HTTP handshake failed")?;
        let conn = tokio::spawn(async move {
            let _ = conn.await;
        });
        let mut req = hyper::Request::builder()
            .method(method)
            .uri(path)
            .header("host", format!("{}:{}", self.host, self.port))
            .header("x-api-key", self.x_api_key.clone())
            .header("authorization", self.bearer.clone());
        if path.starts_with("/v1/messages") {
            req = req.header("anthropic-version", "2023-06-01");
        }
        let payload = match body {
            Some(b) => {
                req = req.header("content-type", "application/json");
                Bytes::from(serde_json::to_vec(b)?)
            }
            None => Bytes::new(),
        };
        let req = req
            .body(Full::new(payload))
            .context("building the request")?;
        Ok((sender, conn, req))
    }

    /// One request on its own connection; the whole reply.
    pub async fn send(&self, method: &str, path: &str, body: Option<&Value>) -> Result<Resp> {
        let (mut sender, conn, req) = self.open(method, path, body).await?;
        let sent_at = Instant::now();
        let reply = async {
            let res = sender.send_request(req).await?;
            let headers_at = Instant::now();
            let (parts, body) = res.into_parts();
            let body = body.collect().await?.to_bytes().to_vec();
            Ok::<_, hyper::Error>(Resp {
                status: parts.status.as_u16(),
                headers: parts.headers,
                body,
                sent_at,
                headers_at,
                done_at: Instant::now(),
            })
        };
        let result = tokio::time::timeout(self.timeout, reply).await;
        conn.abort();
        match result {
            Err(_) => bail!("no reply to {method} {path} in {:?}", self.timeout),
            Ok(r) => r.with_context(|| format!("{method} {path} failed")),
        }
    }

    /// A POST whose reply is read as it comes: returns once the headers have arrived.
    pub async fn stream(&self, path: &str, body: &Value) -> Result<Sse> {
        let (mut sender, conn, req) = self.open("POST", path, Some(body)).await?;
        let sent_at = Instant::now();
        let res = match tokio::time::timeout(self.timeout, sender.send_request(req)).await {
            Err(_) => {
                conn.abort();
                bail!("no reply to POST {path} in {:?}", self.timeout)
            }
            Ok(Err(e)) => {
                conn.abort();
                return Err(e).with_context(|| format!("POST {path} failed"));
            }
            Ok(Ok(res)) => res,
        };
        let headers_at = Instant::now();
        let (parts, body) = res.into_parts();
        Ok(Sse {
            status: parts.status.as_u16(),
            headers: parts.headers,
            sent_at,
            headers_at,
            body,
            buf: Vec::new(),
            ended: false,
            failed: false,
            conn,
            _sender: sender,
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum SseItem {
    Comment(String),
    Event { name: Option<String>, data: String },
}

/// A streaming reply, parsed by hand from the body's frames.
pub struct Sse {
    pub status: u16,
    pub headers: HeaderMap,
    pub sent_at: Instant,
    pub headers_at: Instant,
    body: Incoming,
    buf: Vec<u8>,
    ended: bool,
    failed: bool,
    conn: JoinHandle<()>,
    _sender: SendRequest<Full<Bytes>>,
}

impl Sse {
    /// True once the server has ended the body.
    pub fn ended(&self) -> bool {
        self.ended
    }

    /// True when the body ended in a transport error (a reset, a truncated chunk) rather than cleanly.
    pub fn failed(&self) -> bool {
        self.failed
    }

    /// The next comment or event, or `None` when `within` passes first or the body has ended.
    pub async fn next(&mut self, within: Duration) -> Option<SseItem> {
        let end = Instant::now() + within;
        loop {
            if let Some(item) = self.parse() {
                return Some(item);
            }
            if self.ended {
                return None;
            }
            if !self.fill(end).await {
                return None;
            }
        }
    }

    /// Reads one more frame into the buffer; false when `end` passed first. A finished body sets `ended`.
    async fn fill(&mut self, end: Instant) -> bool {
        let left = end.saturating_duration_since(Instant::now());
        match tokio::time::timeout(left, self.body.frame()).await {
            Err(_) => false,
            Ok(None) => {
                self.ended = true;
                true
            }
            // A body that broke mid-way is an end, but not a clean one: the caller can tell.
            Ok(Some(Err(_))) => {
                self.failed = true;
                self.ended = true;
                true
            }
            Ok(Some(Ok(frame))) => {
                if let Ok(data) = frame.into_data() {
                    self.buf.extend_from_slice(&data);
                }
                true
            }
        }
    }

    /// Everything not yet taken from the body, buffered bytes first, until it ends or `within` passes: for a status
    /// that isn't a stream (an error body).
    pub async fn rest(&mut self, within: Duration) -> Vec<u8> {
        let end = Instant::now() + within;
        while !self.ended && self.fill(end).await {}
        std::mem::take(&mut self.buf)
    }

    /// The first complete, non-empty block of the buffer, removed from it.
    fn parse(&mut self) -> Option<SseItem> {
        let text = loop {
            let end = self.buf.windows(2).position(|w| w == b"\n\n")?;
            let block: Vec<u8> = self.buf.drain(..end + 2).collect();
            let text = String::from_utf8_lossy(&block[..end]).into_owned();
            if !text.trim().is_empty() {
                break text;
            }
        };
        let (mut comments, mut name, mut data) = (Vec::new(), None, Vec::new());
        for line in text.lines() {
            let value = |v: &str| v.strip_prefix(' ').unwrap_or(v).to_string();
            if let Some(c) = line.strip_prefix(':') {
                comments.push(value(c));
            } else if let Some(n) = line.strip_prefix("event:") {
                name = Some(value(n));
            } else if let Some(d) = line.strip_prefix("data:") {
                data.push(value(d));
            }
        }
        if name.is_none() && data.is_empty() {
            Some(SseItem::Comment(comments.join("\n")))
        } else {
            Some(SseItem::Event {
                name,
                data: data.join("\n"),
            })
        }
    }

    /// Aborts the hyper connection task so the socket closes; merely dropping the body would leave the connection
    /// open, and the server could not tell a disconnect from a slow reader. Returns the instant taken just before.
    pub fn disconnect(self) -> Instant {
        let at = Instant::now();
        self.conn.abort();
        at
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    /// What the canned server writes after the request: a chunked event stream of three events.
    const HEAD: &str =
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";

    async fn write_all(s: &TcpStream, data: &[u8]) -> bool {
        let mut off = 0;
        while off < data.len() {
            if s.writable().await.is_err() {
                return false;
            }
            match s.try_write(&data[off..]) {
                Ok(n) => off += n,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(_) => return false,
            }
        }
        true
    }

    /// Reads until `done` is true of what has arrived; false when the peer closed first.
    async fn read_until(s: &TcpStream, done: impl Fn(&[u8]) -> bool) -> bool {
        let mut got = Vec::new();
        loop {
            if s.readable().await.is_err() {
                return false;
            }
            let mut b = [0u8; 1024];
            match s.try_read(&mut b) {
                Ok(0) => return false,
                Ok(n) => {
                    got.extend_from_slice(&b[..n]);
                    if done(&got) {
                        return true;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(_) => return false,
            }
        }
    }

    /// A raw listener (hyper's `server` feature is not among the dependencies) that answers one request with the
    /// head and three SSE events, then stalls until the client closes (or, with `truncate`, breaks its last chunk and closes); `closed` fires when it does.
    async fn canned(truncate: bool) -> (u16, oneshot::Receiver<()>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (closed_tx, closed_rx) = oneshot::channel();
        tokio::spawn(async move {
            let (s, _) = listener.accept().await.unwrap();
            if !read_until(&s, |b| b.windows(4).any(|w| w == b"\r\n\r\n")).await {
                return;
            }
            let mut out = HEAD.as_bytes().to_vec();
            for i in 1..=3 {
                let event = format!("data: {{\"n\":{i}}}\n\n");
                out.extend_from_slice(format!("{:x}\r\n{event}\r\n", event.len()).as_bytes());
            }
            if truncate {
                // A chunk that promises 255 bytes and delivers three, then the socket closes.
                out.extend_from_slice(
                    b"ff
abc",
                );
            }
            if !write_all(&s, &out).await {
                return;
            }
            if truncate {
                return;
            }
            read_until(&s, |_| false).await;
            let _ = closed_tx.send(());
        });
        (port, closed_rx)
    }

    #[tokio::test]
    async fn sse_yields_three_events_then_times_out_and_disconnect_returns() {
        let (port, closed) = canned(false).await;
        let client = Client::new("127.0.0.1", port, "k").unwrap();
        let mut sse = client
            .stream("/v1/chat/completions", &serde_json::json!({"stream": true}))
            .await
            .unwrap();
        assert_eq!(sse.status, 200);
        assert!(sse.headers_at >= sse.sent_at);
        for i in 1..=3 {
            let item = sse.next(Duration::from_secs(5)).await.unwrap();
            assert_eq!(
                item,
                SseItem::Event {
                    name: None,
                    data: format!("{{\"n\":{i}}}")
                }
            );
        }
        let wait = Duration::from_millis(300);
        let began = Instant::now();
        assert_eq!(sse.next(wait).await, None);
        let took = began.elapsed();
        assert!(took >= wait && took < Duration::from_secs(5), "{took:?}");
        assert!(!sse.ended(), "the server never ended the body");
        let at = sse.disconnect();
        assert!(at >= began);
        tokio::time::timeout(Duration::from_secs(5), closed)
            .await
            .expect("the server saw no close")
            .unwrap();
    }

    #[tokio::test]
    async fn a_body_that_breaks_mid_chunk_ends_the_stream_as_failed_not_clean() {
        let (port, _closed) = canned(true).await;
        let client = Client::new("127.0.0.1", port, "k").unwrap();
        let mut sse = client
            .stream("/v1/chat/completions", &serde_json::json!({"stream": true}))
            .await
            .unwrap();
        for _ in 0..3 {
            assert!(sse.next(Duration::from_secs(5)).await.is_some());
        }
        assert_eq!(sse.next(Duration::from_secs(5)).await, None);
        assert!(sse.ended() && sse.failed());
    }

    #[tokio::test]
    async fn a_refused_connection_is_an_error_naming_the_address() {
        let port = {
            let l = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
            l.local_addr().unwrap().port()
        };
        let client = Client::new("127.0.0.1", port, "k").unwrap();
        let err = client.send("GET", "/health", None).await.err().unwrap();
        assert!(format!("{err:#}").contains(&port.to_string()), "{err:#}");
    }
}
