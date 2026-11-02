//! The harness the `artemis-tools` test binaries share (modelled on `crates/server/tests/common/mod.rs`, which a
//! test module can't share across crates): an `App` over any `Backend`, the committed vocabulary and template, and an
//! in-process server on `127.0.0.1:0`. Each binary builds the parts it uses once.
#![allow(dead_code)]
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use qwen36::tokenizer::{Tokenizer, Vocab};
use serde_json::Value;
use server::app::{serve, App, AppParts, Exit, Tuning};
use server::backend::Backend;
use server::config::Config;
use server::fake::{FakeBackend, FakeConfig};
use server::logs::Logger;
use server::template::Template;
use server::tokens::Specials;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

pub const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/");

/// How long a test waits for a reply before it fails instead of hanging.
pub const REPLY: Duration = Duration::from_secs(60);

fn vocab() -> &'static Vocab {
    static V: OnceLock<Vocab> = OnceLock::new();
    V.get_or_init(|| {
        let b = std::fs::read(format!("{TESTDATA}tokenizer/qwen36-vocab.bin")).unwrap();
        Vocab::from_extract(&b).unwrap()
    })
}

fn tokenizer_arc() -> &'static Arc<Tokenizer> {
    static T: OnceLock<Arc<Tokenizer>> = OnceLock::new();
    T.get_or_init(|| Arc::new(Tokenizer::new(vocab()).unwrap()))
}

/// The model's tokenizer, from the committed vocabulary extract.
pub fn tokenizer() -> &'static Tokenizer {
    tokenizer_arc()
}

/// How many tokens the committed vocabulary has: the `n_vocab` a worker over a fake engine takes, so
/// `Request::check` accepts every id the real tokenizer gives.
pub fn vocab_size() -> usize {
    vocab().tokens.len()
}

/// The committed copy of the model's chat template, compiled: a `Template` is not `Clone`, so each caller gets its
/// own, from the source read once.
pub fn template() -> Template {
    static SRC: OnceLock<String> = OnceLock::new();
    let src = SRC.get_or_init(|| {
        std::fs::read_to_string(format!("{TESTDATA}template/qwen36-chat-template.jinja")).unwrap()
    });
    Template::new(src.clone()).unwrap()
}

/// The ids the server finds by token text.
pub fn specials() -> &'static Specials {
    static T: OnceLock<Specials> = OnceLock::new();
    T.get_or_init(|| Specials::new(vocab(), tokenizer()).unwrap())
}

/// A new, empty directory under the system temp dir, unique per process and call. The process id alone is not unique
/// across runs: Windows reuses ids, and the directories are never cleaned, so a name from an earlier process with the
/// same id may exist already and hold its log lines. `create_dir` fails on an existing name, so a taken one is skipped.
pub fn scratch_dir(tag: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join("artemis-tools-tests");
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

/// `backend`, `cfg` and the logs in `log_dir` as the parts of an `App`. A `key` is written to a key file as
/// `<key>\r\n`, as a Windows editor leaves it, and named by `cfg.api_key_file`. The host, the port and the tuning are
/// `start`'s to set.
pub fn parts(
    backend: Arc<dyn Backend>,
    cfg: Config,
    key: Option<&str>,
    log_dir: &Path,
) -> AppParts {
    let mut config = cfg;
    if let Some(key) = key {
        let file = scratch_dir("key").join("api-key");
        std::fs::write(&file, format!("{key}\r\n")).unwrap();
        config.api_key_file = Some(file);
    }
    let key = config.read_key().unwrap();
    config.log_dir = Some(log_dir.to_path_buf());
    AppParts {
        config,
        key,
        backend,
        tokenizer: tokenizer_arc().clone(),
        specials: specials().clone(),
        template: template(),
        logger: Logger::new(log_dir.to_path_buf(), 0).unwrap(),
        tuning: Tuning::default(),
        version: env!("CARGO_PKG_VERSION"),
    }
}

/// The server's fake engine, over the committed vocabulary.
pub fn fake(cfg: FakeConfig) -> Arc<FakeBackend> {
    FakeBackend::new(cfg, tokenizer_arc().clone(), specials().clone())
}

/// How a `Running` server is stopped and collected: `serve`'s shutdown future and its task.
pub struct Stop {
    tx: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<Result<(), Exit>>>,
}

impl Stop {
    /// Resolves `serve`'s shutdown future, as a signal would, without waiting for `serve`.
    pub fn fire(&mut self) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(());
        }
    }

    /// What `serve` returned, if it has returned within `within`.
    pub async fn join(&mut self, within: Duration) -> Option<Result<(), Exit>> {
        let task = self.task.as_mut()?;
        let result = tokio::time::timeout(within, task).await.ok()?;
        self.task = None;
        Some(result.unwrap())
    }

    /// Stops the server as a signal would and returns what `serve` returned.
    pub async fn stop(&mut self) -> Result<(), Exit> {
        self.fire();
        self.join(REPLY).await.expect("the server did not stop")
    }
}

/// A server on the test's runtime, bound to an ephemeral loopback port. Its scratch files (the log directory, a key
/// file) go with it, unless the test is failing: then they stay for a look.
pub struct Running {
    pub base_url: String,
    pub key_file: Option<PathBuf>,
    pub log_dir: PathBuf,
    pub app: App,
    pub stop: Stop,
    addr: SocketAddr,
    halts: Arc<Mutex<Vec<i32>>>,
}

/// Serves `parts` on `127.0.0.1:0` with `tuning`, except its `halt`: that records the codes the watchdog halts with
/// (`Running::halts`), so a stall does not end the test process.
pub async fn start(parts: AppParts, tuning: Tuning) -> Running {
    let mut parts = parts;
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    parts.config.host = "127.0.0.1".into();
    parts.config.port = addr.port();
    let halts = Arc::new(Mutex::new(Vec::new()));
    let recorder = halts.clone();
    parts.tuning = Tuning {
        halt: Arc::new(move |code| recorder.lock().unwrap().push(code)),
        ..tuning
    };
    let key_file = parts.config.api_key_file.clone();
    let log_dir = parts
        .config
        .log_dir
        .clone()
        .expect("`parts` names the log directory");
    let app = App::new(parts).unwrap();
    let (tx, stopped) = oneshot::channel::<()>();
    let task = tokio::spawn(serve(app.clone(), listener, async move {
        let _ = stopped.await;
    }));
    Running {
        base_url: format!("http://127.0.0.1:{}", addr.port()),
        key_file,
        log_dir,
        app,
        stop: Stop {
            tx: Some(tx),
            task: Some(task),
        },
        addr,
        halts,
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        let _ = std::fs::remove_dir_all(&self.log_dir);
        if let Some(dir) = self.key_file.as_deref().and_then(Path::parent) {
            if dir.starts_with(std::env::temp_dir().join("artemis-tools-tests")) {
                let _ = std::fs::remove_dir_all(dir);
            }
        }
    }
}

pub struct Resp {
    pub status: u16,
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

impl Running {
    /// The codes the watchdog has halted with, oldest first.
    pub fn halts(&self) -> Vec<i32> {
        self.halts.lock().unwrap().clone()
    }

    /// One request on its own connection; the whole reply. A body gets `content-type: application/json`.
    pub async fn send(
        &self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Resp {
        let stream = TcpStream::connect(self.addr).await.unwrap();
        let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
            .await
            .unwrap();
        let conn = tokio::spawn(async move {
            let _ = conn.await;
        });
        let mut req = hyper::Request::builder()
            .method(method)
            .uri(path)
            .header("host", format!("127.0.0.1:{}", self.addr.port()));
        if !body.is_empty() {
            req = req.header("content-type", "application/json");
        }
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let req = req.body(Full::new(Bytes::copy_from_slice(body))).unwrap();
        let reply = async {
            let res = sender.send_request(req).await.unwrap();
            let (parts, body) = res.into_parts();
            Resp {
                status: parts.status.as_u16(),
                body: body.collect().await.unwrap().to_bytes().to_vec(),
            }
        };
        let resp = tokio::time::timeout(REPLY, reply)
            .await
            .unwrap_or_else(|_| panic!("no reply to {method} {path} in {REPLY:?}"));
        conn.abort();
        resp
    }

    /// The lines in this server's log directory, oldest first.
    pub fn log_lines(&self) -> Vec<Value> {
        log_lines(&self.log_dir)
    }
}

/// Every line in the `.jsonl` files of `dir`, oldest first. A last line with no newline yet is a write in progress and
/// is left out; a complete line that is not JSON is a failure.
pub fn log_lines(dir: &Path) -> Vec<Value> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .collect();
    files.sort();
    files
        .iter()
        .flat_map(|p| {
            let text = std::fs::read_to_string(p).unwrap();
            let whole = text.rfind('\n').map_or("", |i| &text[..=i]);
            whole.lines().map(String::from).collect::<Vec<_>>()
        })
        .map(|l| serde_json::from_str(&l).unwrap_or_else(|e| panic!("a bad log line ({e}): {l}")))
        .collect()
}

/// The lifecycle lines of `dir` (the ones with an `event`), oldest first.
pub fn lifecycle_lines(dir: &Path) -> Vec<Value> {
    log_lines(dir)
        .into_iter()
        .filter(|l| l.get("event").is_some())
        .collect()
}

/// The `event` names of the lifecycle lines of `dir`, oldest first.
pub fn events(dir: &Path) -> Vec<String> {
    lifecycle_lines(dir)
        .iter()
        .map(|l| l["event"].as_str().unwrap().to_string())
        .collect()
}
