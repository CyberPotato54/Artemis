//! Helpers shared by the test binaries, each built once per binary.
#![allow(dead_code)]
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request};
use http_body_util::BodyExt;
use hyper::body::{Body, Bytes, Frame, Incoming, SizeHint};
use hyper_util::rt::TokioIo;
use qwen36::tokenizer::{Tokenizer, Vocab};
use serde_json::Value;
use server::app::{serve, App, AppParts, Exit, Tuning};
use server::config::Config;
use server::fake::{FakeBackend, FakeConfig};
use server::logs::Logger;
use server::template::Template;
use server::tokens::Specials;
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

pub const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/");

fn tokenizer_arc() -> &'static Arc<Tokenizer> {
    static T: OnceLock<Arc<Tokenizer>> = OnceLock::new();
    T.get_or_init(|| {
        let b = std::fs::read(format!("{TESTDATA}tokenizer/qwen36-vocab.bin")).unwrap();
        Arc::new(Tokenizer::new(&Vocab::from_extract(&b).unwrap()).unwrap())
    })
}

/// The model's tokenizer, from the committed vocabulary extract.
pub fn tokenizer() -> &'static Tokenizer {
    tokenizer_arc()
}

/// The committed copy of the model's chat template.
pub fn template() -> &'static Template {
    static T: OnceLock<Template> = OnceLock::new();
    T.get_or_init(|| {
        let src = std::fs::read_to_string(format!("{TESTDATA}template/qwen36-chat-template.jinja"))
            .unwrap();
        Template::new(src).unwrap()
    })
}

/// The ids the server finds by token text.
pub fn specials() -> &'static Specials {
    static T: OnceLock<Specials> = OnceLock::new();
    T.get_or_init(|| {
        let b = std::fs::read(format!("{TESTDATA}tokenizer/qwen36-vocab.bin")).unwrap();
        Specials::new(&Vocab::from_extract(&b).unwrap(), tokenizer()).unwrap()
    })
}

/// The last-query flag computed independently of `is_user_query`: the committed template, with a line appended that
/// prints its own `last_query_index`, renders `messages` (the template context's messages); returns that index and
/// whether the message there is a `user` one with only `tool` messages and the system ones the template drops after it.
/// The template merges the first message if it is a system or developer one, and the second too if it is another; it
/// renders no other system message.
pub fn template_user_query(messages: &Value) -> (usize, bool) {
    const MARK: &str = "\u{1f}LQI=";
    static T: OnceLock<Template> = OnceLock::new();
    let t = T.get_or_init(|| {
        let src = std::fs::read_to_string(format!("{TESTDATA}template/qwen36-chat-template.jinja"))
            .unwrap();
        Template::new(format!("{src}{{{{- '{MARK}' ~ ns.last_query_index }}}}")).unwrap()
    });
    let out = t
        .render_json(&serde_json::json!({"messages": messages, "add_generation_prompt": true}))
        .unwrap();
    let lqi: usize = out.rsplit_once(MARK).unwrap().1.parse().unwrap();
    let ms = messages.as_array().unwrap();
    let role = |m: &Value| m["role"].as_str().unwrap().to_string();
    let is_system = |i: usize| matches!(role(&ms[i]).as_str(), "system" | "developer");
    let merged = match (is_system(0), ms.len() > 1 && is_system(1)) {
        (false, _) => 0,
        (true, false) => 1,
        (true, true) => 2,
    };
    // `i >= merged` mirrors the template's rule but cannot decide: ms[lqi] is a `user`, so no index after it is a
    // merged leading system.
    let user_query = role(&ms[lqi]) == "user"
        && (lqi + 1..ms.len()).all(|i| role(&ms[i]) == "tool" || (is_system(i) && i >= merged));
    (lqi, user_query)
}

/// What `TestServer::start` builds the server from.
pub struct Setup {
    pub config: Config,
    /// Written to a key file as `<key>\r\n`, as a Windows editor leaves it.
    pub key: Option<&'static str>,
    pub fake: FakeConfig,
    pub tuning: Tuning,
    /// The address the listener binds, port 0. `config.host` is set to it.
    pub bind_host: &'static str,
    /// The fake writes its loads and idle unloads to the server's log, as the real engine's adapter does. Off, the
    /// log holds only what the server writes.
    pub fake_logs: bool,
    /// A chat template to use instead of the committed one.
    pub template: Option<&'static str>,
    /// The exit codes the watchdog's `halt` was called with (`TestServer::halts`), so a stall does not end the test
    /// process. `tuning.halt` records here; a test that replaces it does so on its own.
    pub halts: Arc<Mutex<Vec<i32>>>,
}

impl Setup {
    /// No key, the model `m.gguf` with 65536 tokens of context, a fake that starts idle with zero delays.
    pub fn new() -> Setup {
        let halts = Arc::new(Mutex::new(Vec::new()));
        let recorder = halts.clone();
        Setup {
            config: Config {
                model: Some("m.gguf".into()),
                ctx: 65_536,
                ..Config::default()
            },
            key: None,
            fake: FakeConfig::default(),
            // Shorter than production's 30 s, so a drain that hangs fails a test instead of passing late.
            tuning: Tuning {
                shutdown_grace: Duration::from_secs(5),
                halt: Arc::new(move |code| recorder.lock().unwrap().push(code)),
                ..Tuning::default()
            },
            bind_host: "127.0.0.1",
            fake_logs: false,
            template: None,
            halts,
        }
    }
}

/// A server on the test's runtime, bound to an ephemeral port, over a fake engine.
pub struct TestServer {
    pub addr: SocketAddr,
    pub fake: Arc<FakeBackend>,
    pub app: App,
    /// Removed with the server (see `Drop`): read the logs before `shutdown`, which consumes the server.
    pub log_dir: PathBuf,
    scratch: Vec<PathBuf>,
    halts: Arc<Mutex<Vec<i32>>>,
    stop: Mutex<Option<oneshot::Sender<()>>>,
    task: Option<JoinHandle<Result<(), Exit>>>,
}

/// A new, empty directory under the system temp dir, unique per process and call. The process id alone is not unique
/// across runs: Windows reuses ids, and the directories are never cleaned, so a name from an earlier process with the
/// same id may exist already and hold its log lines. `create_dir` fails on an existing name, so a taken one is skipped.
pub fn scratch_dir(tag: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join("artemis-server-tests");
    std::fs::create_dir_all(&root).unwrap();
    loop {
        let dir = root.join(format!(
            "{}-{}-{tag}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        match std::fs::create_dir(&dir) {
            Ok(()) => return dir,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => panic!("cannot create {}: {e}", dir.display()),
        }
    }
}

/// Polls `f` every 10 ms until it is true or `within` has passed; the last answer.
pub async fn wait_until(within: Duration, mut f: impl FnMut() -> bool) -> bool {
    let end = Instant::now() + within;
    loop {
        if f() {
            return true;
        }
        if Instant::now() >= end {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// A request body that goes out whole (with a length) or in pieces (chunked, no length).
struct Payload {
    chunks: VecDeque<Bytes>,
    len: Option<u64>,
}

impl Body for Payload {
    type Data = Bytes;
    type Error = std::io::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        Poll::Ready(self.chunks.pop_front().map(|b| Ok(Frame::data(b))))
    }

    fn is_end_stream(&self) -> bool {
        self.chunks.is_empty()
    }

    fn size_hint(&self) -> SizeHint {
        match self.len {
            Some(n) => SizeHint::with_exact(n),
            None => SizeHint::default(),
        }
    }
}

type Sender = hyper::client::conn::http1::SendRequest<Payload>;

/// How long a test waits for a reply before it fails instead of hanging.
const REPLY: Duration = Duration::from_secs(60);

pub struct Resp {
    pub status: u16,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

impl Resp {
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|e| {
            panic!(
                "status {}: the body is not JSON ({e}): {}",
                self.status,
                String::from_utf8_lossy(&self.body)
            )
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SseItem {
    Comment(String),
    Event { name: Option<String>, data: String },
}

/// A streaming reply, parsed by hand from the body's frames.
pub struct Sse {
    status: u16,
    headers: HeaderMap,
    body: Incoming,
    buf: Vec<u8>,
    ended: bool,
    conn: JoinHandle<()>,
    _sender: Sender,
}

impl Sse {
    pub fn status(&self) -> u16 {
        self.status
    }

    pub fn headers(&self) -> &HeaderMap {
        &self.headers
    }

    /// True once the server has ended the body.
    pub fn ended(&self) -> bool {
        self.ended
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
            let left = end.saturating_duration_since(Instant::now());
            match tokio::time::timeout(left, self.body.frame()).await {
                Err(_) => return None,
                Ok(None) | Ok(Some(Err(_))) => self.ended = true,
                Ok(Some(Ok(frame))) => {
                    if let Ok(data) = frame.into_data() {
                        self.buf.extend_from_slice(&data);
                    }
                }
            }
        }
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

    /// Aborts the hyper connection task so the socket closes (the OS may send FIN or RST); merely dropping the body
    /// would leave the connection open and the server could not tell a disconnect from a slow reader.
    pub fn disconnect(self) {
        self.conn.abort();
    }
}

/// The server's scratch directories go with it, unless the test is failing: then they stay for a look.
impl Drop for TestServer {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            for dir in &self.scratch {
                let _ = std::fs::remove_dir_all(dir);
            }
        }
    }
}

impl TestServer {
    pub async fn start(s: Setup) -> TestServer {
        let log_dir = scratch_dir("logs");
        let mut scratch = vec![log_dir.clone()];
        let mut config = s.config;
        if let Some(key) = s.key {
            let dir = scratch_dir("key");
            scratch.push(dir.clone());
            let file = dir.join("api-key");
            std::fs::write(&file, format!("{key}\r\n")).unwrap();
            config.api_key_file = Some(file);
        }
        let key = config.read_key().unwrap();
        let listener = TcpListener::bind((s.bind_host, 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        config.host = s.bind_host.to_string();
        config.port = addr.port();
        config.log_dir = Some(log_dir.clone());
        let logger = Logger::new(log_dir.clone(), 0).unwrap();
        let fake = FakeBackend::new(s.fake, tokenizer_arc().clone(), specials().clone());
        if s.fake_logs {
            fake.set_logger(logger.clone());
        }
        let src = s.template.map(String::from).unwrap_or_else(|| {
            std::fs::read_to_string(format!("{TESTDATA}template/qwen36-chat-template.jinja"))
                .unwrap()
        });
        let app = App::new(AppParts {
            config,
            key,
            backend: fake.clone(),
            tokenizer: tokenizer_arc().clone(),
            specials: specials().clone(),
            template: Template::new(src).unwrap(),
            logger,
            tuning: s.tuning,
            version: env!("CARGO_PKG_VERSION"),
        })
        .unwrap();
        let (stop, stopped) = oneshot::channel::<()>();
        let task = tokio::spawn(serve(app.clone(), listener, async move {
            let _ = stopped.await;
        }));
        TestServer {
            addr,
            fake,
            app,
            log_dir,
            scratch,
            halts: s.halts,
            stop: Mutex::new(Some(stop)),
            task: Some(task),
        }
    }

    /// One connection per call. `Host` is `127.0.0.1:<port>` unless the headers give one; a `host` header of `""`
    /// leaves it out. A body gets `content-type: application/json` unless the headers give one; the pseudo-header
    /// `no-content-type` (sent to no one) leaves the type out altogether, which `("content-type", "")` does not: that
    /// sends an empty one.
    async fn open(
        &self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: Payload,
    ) -> (Sender, JoinHandle<()>, Request<Payload>) {
        let stream = TcpStream::connect(self.addr).await.unwrap();
        let (sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
            .await
            .unwrap();
        let conn = tokio::spawn(async move {
            let _ = conn.await;
        });
        let mut req = Request::builder()
            .method(Method::from_bytes(method.as_bytes()).unwrap())
            .uri(path);
        let has = |name: &str| headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(name));
        if !has("host") {
            req = req.header("host", format!("127.0.0.1:{}", self.addr.port()));
        }
        if !body.is_end_stream() && !has("content-type") && !has("no-content-type") {
            req = req.header("content-type", "application/json");
        }
        for (k, v) in headers {
            if (k.eq_ignore_ascii_case("host") && v.is_empty())
                || k.eq_ignore_ascii_case("no-content-type")
            {
                continue;
            }
            req = req.header(
                HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        (sender, conn, req.body(body).unwrap())
    }

    fn whole(body: &[u8]) -> Payload {
        Payload {
            chunks: if body.is_empty() {
                VecDeque::new()
            } else {
                VecDeque::from([Bytes::copy_from_slice(body)])
            },
            len: Some(body.len() as u64),
        }
    }

    async fn exchange(
        &self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: Payload,
    ) -> Resp {
        let (mut sender, conn, req) = self.open(method, path, headers, body).await;
        let reply = async {
            let res = sender.send_request(req).await.unwrap();
            let (parts, body) = res.into_parts();
            Resp {
                status: parts.status.as_u16(),
                headers: parts.headers,
                body: body.collect().await.unwrap().to_bytes().to_vec(),
            }
        };
        let resp = tokio::time::timeout(REPLY, reply)
            .await
            .unwrap_or_else(|_| panic!("no reply to {method} {path} in {REPLY:?}"));
        conn.abort();
        resp
    }

    pub async fn send(
        &self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Resp {
        self.exchange(method, path, headers, Self::whole(body))
            .await
    }

    /// Like `send`, but the body goes out in 64 KiB chunks with no `Content-Length`.
    pub async fn send_chunked(
        &self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Resp {
        let chunks = body.chunks(64 << 10).map(Bytes::copy_from_slice).collect();
        self.exchange(method, path, headers, Payload { chunks, len: None })
            .await
    }

    /// A POST whose reply is read as it comes.
    pub async fn stream(&self, path: &str, headers: &[(&str, &str)], body: &[u8]) -> Sse {
        let (mut sender, conn, req) = self.open("POST", path, headers, Self::whole(body)).await;
        let res = tokio::time::timeout(REPLY, sender.send_request(req))
            .await
            .unwrap_or_else(|_| panic!("no reply to POST {path} in {REPLY:?}"))
            .unwrap();
        let (parts, body) = res.into_parts();
        Sse {
            status: parts.status.as_u16(),
            headers: parts.headers,
            body,
            buf: Vec::new(),
            ended: false,
            conn,
            _sender: sender,
        }
    }

    /// Sends the request, waits `after`, then aborts the connection task (the socket closes) without reading.
    /// Returns the instant taken just before the abort.
    pub async fn send_then_drop(
        &self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: &[u8],
        after: Duration,
    ) -> Instant {
        let (mut sender, conn, req) = self.open(method, path, headers, Self::whole(body)).await;
        // `send_request` queues the request at once; the connection task writes it while this sleeps.
        let pending = sender.send_request(req);
        tokio::time::sleep(after).await;
        let at = Instant::now();
        conn.abort();
        drop(pending);
        at
    }

    /// The codes the watchdog has halted with, oldest first (the test server never exits the process).
    pub fn halts(&self) -> Vec<i32> {
        self.halts.lock().unwrap().clone()
    }

    /// How many requests wait in the HTTP-side queue (not the one the engine is running). Tests order arrivals by
    /// waiting for this to rise, never by sleeping.
    pub fn waiting(&self) -> usize {
        self.app.queue().waiting()
    }

    /// Resolves `serve`'s shutdown future, as a signal would, without waiting for `serve`; `exited` collects it.
    pub fn stop(&self) {
        let stop = self.stop.lock().unwrap().take();
        if let Some(stop) = stop {
            let _ = stop.send(());
        }
    }

    /// Stops the server as a signal would and returns what `serve` returned.
    pub async fn shutdown(mut self) -> Result<(), Exit> {
        self.stop();
        match self.task.take() {
            Some(task) => tokio::time::timeout(REPLY, task)
                .await
                .expect("the server did not stop")
                .unwrap(),
            None => Ok(()),
        }
    }

    /// What `serve` returned, if it has returned within `within`.
    pub async fn exited(&mut self, within: Duration) -> Option<Result<(), Exit>> {
        let task = self.task.as_mut()?;
        let result = tokio::time::timeout(within, task).await.ok()?;
        self.task = None;
        Some(result.unwrap())
    }
}
