//! The stall watchdog on the real engine: the server is started from the production config
//! template on port 8090 with the hidden `--test-stall-at-token` hook, one streamed request stalls the engine's worker
//! at its 16th token, and the process must log `stall`, `stats` and `exit`, and end with code 4 within `stall_s` plus
//! ten seconds. The test needs the GPU and the real model, so it is `#[ignore]`d; the runner's `serve-stall` block runs
//! it, one process on the GPU at a time.
//!
//! What it reads from the environment:
//! - `ARTEMIS_MODEL` (required): the GGUF.
//! - `ARTEMIS_KEY_FILE` (optional): the runner's per-run key file. The test reads it and never prints or deletes it.
//!   Without it the test writes a per-run key into its scratch directory, and deletes that file when it ends.
//! - `ARTEMIS_RUN_DIR` (optional): the runner's run directory, which then IS the scratch directory (the filled config at
//!   `artemis.toml`, the logs at `logs\`, the child's stderr at `stderr.txt`) and which the test never deletes. The test
//!   also writes its figures to `stall.json` there. Without it, a temporary directory that is deleted after a pass.
mod support;

use artemis_tools::test7::http::{Client, SseItem};
use serde_json::{json, Value};
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use support::lifecycle_lines;

/// The port every runner block uses; the test refuses to run when it is busy.
const PORT: u16 = 8090;
/// The request stalls at this token of its reply.
const STALL_AT_TOKEN: u32 = 16;
/// How long `/health` may take to answer after the process starts.
const HEALTH_WITHIN: Duration = Duration::from_secs(120);
/// From the request to its first content chunk: the lazy load (95.3 s worst cold) and a short prefill.
const FIRST_CHUNK_WITHIN: Duration = Duration::from_secs(300);
/// The template, relative to this crate.
const TEMPLATE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/config/artemis-production.toml.in"
);

/// True when `port` appears in `text` as a whole number: not preceded or followed by another digit.
fn names_port(text: &str, port: &str) -> bool {
    let bytes = text.as_bytes();
    text.match_indices(port).any(|(at, _)| {
        let end = at + port.len();
        (at == 0 || !bytes[at - 1].is_ascii_digit())
            && (end >= bytes.len() || !bytes[end].is_ascii_digit())
    })
}

/// The template filled for `port`, with no other change.
fn fill_config(model: &Path, port: u16, key_file: &Path, log_dir: &Path) -> String {
    let template = std::fs::read_to_string(TEMPLATE).expect("the config template");
    let (model, port, key_file, log_dir) = (
        model.display().to_string(),
        port.to_string(),
        key_file.display().to_string(),
        log_dir.display().to_string(),
    );
    server::config::fill_template(
        &template,
        &[
            ("MODEL", model.as_str()),
            ("PORT", port.as_str()),
            ("API_KEY_FILE", key_file.as_str()),
            ("LOG_DIR", log_dir.as_str()),
        ],
    )
    .expect("the template fills")
}

/// A whole-number setting of the filled config (`name = 120`).
fn setting(config: &str, name: &str) -> u64 {
    config
        .lines()
        .find_map(|l| l.strip_prefix(name)?.trim_start().strip_prefix('='))
        .unwrap_or_else(|| panic!("the config sets no {name}"))
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("{name} is not a whole number"))
}

#[test]
fn the_template_fills_without_the_production_port() {
    let config = fill_config(
        Path::new("m.gguf"),
        PORT,
        Path::new("key"),
        Path::new("logs"),
    );
    assert!(!names_port(&config, "8080"), "{config}");
    assert!(names_port(&config, "8090"), "{config}");
    assert_eq!(setting(&config, "stall_s"), 120);
    // The check is on whole numbers: a longer number that holds the digits is not the port.
    assert!(names_port("port = 8080", "8080"));
    assert!(names_port("8080", "8080"));
    assert!(!names_port("port = 28080", "8080"));
    assert!(!names_port("port = 80801", "8080"));
    assert!(names_port("x 80801 and 8080", "8080"));
}

/// A key no one chose: the hasher's random keys (the standard library seeds them from the system), twice.
fn fresh_key() -> String {
    let word = || RandomState::new().build_hasher().finish();
    format!("{:016x}{:016x}", word(), word())
}

/// The first non-empty line of a key file, without a byte-order mark.
fn read_key(file: &Path) -> String {
    let text = std::fs::read_to_string(file).expect("the key file is readable");
    text.trim_start_matches('\u{feff}')
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .expect("the key file holds a key")
        .to_string()
}

/// The scratch directory, and the files this test made in it that hold the key.
struct Scratch {
    dir: PathBuf,
    /// Deleted at the end of the test when it made the directory.
    temporary: bool,
    /// The key file this test wrote, deleted at the end whether the test passed or not. Never the runner's.
    own_key_file: Option<PathBuf>,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Some(file) = &self.own_key_file {
            let _ = std::fs::remove_file(file);
        }
        // A failing test leaves its logs for a look; the key file is already gone.
        if self.temporary && !std::thread::panicking() {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

/// What the streamed request showed, and how the process ended. Whatever the run did not reach stays `None` or 0.
#[derive(Default)]
struct Run {
    /// The stream's id, from its first data event.
    id: Option<String>,
    /// Content chunks (a non-empty `delta.content`) seen.
    chunks: usize,
    /// The arrival of the last data event (keep-alive comments don't count).
    t_last: Option<Instant>,
    /// When the process was found ended.
    t_end: Option<Instant>,
    /// Its exit code.
    code: Option<i32>,
    /// Why the run stopped short: the request did not start, or the process did not end in time.
    problem: Option<String>,
}

/// The child process, killed if the test ends first.
struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        if let Ok(None) = self.0.try_wait() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

/// Every file under `dir`, except `skip`.
fn files_under(dir: &Path, skip: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap().map(Result::unwrap) {
        let path = entry.path();
        if path.is_dir() {
            out.extend(files_under(&path, skip));
        } else if !skip.iter().any(|s| same_file(s, &path)) {
            out.push(path);
        }
    }
    out
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// Something holds the port: a bind to it fails, or a connection to it is accepted. The connect matters on Windows,
/// where a bind to 127.0.0.1 can succeed while another process listens on 0.0.0.0 on the same port.
fn port_is_busy(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpListener::bind(addr).is_err()
        || TcpStream::connect_timeout(&addr, Duration::from_secs(1)).is_ok()
}

/// Nothing holds the port (`port_is_busy`). Waits a few seconds for the system to let go of it.
fn port_is_free(port: u16) -> bool {
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        if !port_is_busy(port) {
            return true;
        }
        if Instant::now() >= end {
            return false;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Polls `GET /health` until it answers 200. `ended` says whether the process has ended (with what to show for it);
/// each poll gets what is left of `within`, not the client's own 600 s, and the polls are `pause` apart. The error is
/// the problem, for the caller to fail on. All of its tokio calls are inside this `async fn`, so they run on the
/// caller's runtime: a `timeout` or a `sleep` built outside `block_on` has no reactor to register with.
async fn wait_for_health(
    client: &Client,
    within: Duration,
    pause: Duration,
    mut ended: impl FnMut() -> Option<String>,
) -> Result<(), String> {
    let started = Instant::now();
    loop {
        if let Some(status) = ended() {
            return Err(format!(
                "artemis ended before it answered /health: {status}"
            ));
        }
        let left = within.saturating_sub(started.elapsed());
        if left.is_zero() {
            return Err(format!("no /health within {within:?}"));
        }
        let health = tokio::time::timeout(left, client.send("GET", "/health", None)).await;
        if matches!(health, Ok(Ok(ref r)) if r.status == 200) {
            return Ok(());
        }
        tokio::time::sleep(pause).await;
    }
}

/// Sends the streamed request and follows it until the process ends. `t_last` is the arrival of the last data event
/// (the keep-alive comments don't count). `ended` gives the process's exit code once it has ended. The reply's headers
/// come within `first_chunk_within`, and so does the first data event; once there is one, the process must end within
/// `last_within` of the latest. Nothing in here asserts: a request that never starts, or a process that does not end
/// in time, comes back as a `problem`, so the figures are written (with what is missing as null) before the test fails
/// on it.
async fn stream_until_end(
    client: &Client,
    body: &Value,
    first_chunk_within: Duration,
    last_within: Duration,
    mut ended: impl FnMut() -> Option<Option<i32>>,
) -> Run {
    let sent = Instant::now();
    let mut run = Run::default();
    // The wait for the reply's headers, which come after the lazy load, has its own bound.
    let stream = tokio::time::timeout(
        first_chunk_within,
        client.stream("/v1/chat/completions", body),
    )
    .await;
    let mut stream = match stream {
        Err(_) => {
            run.problem = Some(format!("no reply headers within {first_chunk_within:?}"));
            return run;
        }
        Ok(Err(e)) => {
            run.problem = Some(format!("the request failed: {e:#}"));
            return run;
        }
        Ok(Ok(stream)) => stream,
    };
    if stream.status != 200 {
        run.problem = Some(format!("the request got status {}", stream.status));
        return run;
    }
    loop {
        if let Some(code) = ended() {
            run.t_end = Some(Instant::now());
            run.code = code;
            return run;
        }
        let limit = match run.t_last {
            Some(t) => t + last_within,
            None => sent + first_chunk_within,
        };
        if Instant::now() >= limit {
            run.problem = Some("the process did not end in time".to_string());
            return run;
        }
        match stream.next(Duration::from_millis(250)).await {
            Some(SseItem::Event { data, .. }) if data != "[DONE]" => {
                run.t_last = Some(Instant::now());
                let chunk: Value = serde_json::from_str(&data).unwrap_or(Value::Null);
                run.id = chunk["id"].as_str().map(String::from).or(run.id.take());
                let content = &chunk["choices"][0]["delta"]["content"];
                run.chunks += usize::from(content.as_str().is_some_and(|c| !c.is_empty()));
            }
            Some(_) => {}
            // The connection broke with the process: the exit shows at the next `ended`.
            None if stream.ended() => tokio::time::sleep(Duration::from_millis(50)).await,
            None => {}
        }
    }
}

#[test]
#[ignore = "needs the GPU and the real model"]
fn a_stuck_engine_logs_stall_and_exits_4() {
    let model = PathBuf::from(
        std::env::var("ARTEMIS_MODEL").expect("ARTEMIS_MODEL names the model to load"),
    );
    assert!(model.is_file(), "ARTEMIS_MODEL is not a file");
    assert!(!port_is_busy(PORT), "127.0.0.1:{PORT} is busy");

    // The scratch directory, the key and the filled config.
    let run_dir = std::env::var_os("ARTEMIS_RUN_DIR").map(PathBuf::from);
    let mut scratch = Scratch {
        dir: run_dir
            .clone()
            .unwrap_or_else(|| support::scratch_dir("stall-gpu")),
        temporary: run_dir.is_none(),
        own_key_file: None,
    };
    std::fs::create_dir_all(&scratch.dir).unwrap();
    let (key_file, key) = match std::env::var_os("ARTEMIS_KEY_FILE").map(PathBuf::from) {
        Some(file) => {
            let key = read_key(&file);
            (file, key)
        }
        None => {
            let file = scratch.dir.join("api-key");
            let key = fresh_key();
            std::fs::write(&file, format!("{key}\n")).unwrap();
            scratch.own_key_file = Some(file.clone());
            (file, key)
        }
    };
    let logs = scratch.dir.join("logs");
    let config = fill_config(&model, PORT, &key_file, &logs);
    assert!(
        !names_port(&config, "8080"),
        "the filled config names the production port"
    );
    let stall_s = setting(&config, "stall_s");
    assert!(stall_s > 0, "the template switches the watchdog off");
    let config_file = scratch.dir.join("artemis.toml");
    std::fs::write(&config_file, &config).unwrap();

    // The key is in the file, never on the command line. Standard input stays open until the child has ended: closing
    // it would stop the server cleanly, with exit code 0.
    let stderr = std::fs::File::create(scratch.dir.join("stderr.txt")).unwrap();
    let mut child = Guard(
        Command::new(env!("CARGO_BIN_EXE_artemis"))
            // A serve without this runs setup and writes the real tuned file.
            .env("ARTEMIS_TUNED", "off")
            .arg("serve")
            .arg("--config")
            .arg(&config_file)
            .args(["--port", &PORT.to_string()])
            .args(["--test-stall-at-token", &STALL_AT_TOKEN.to_string()])
            .arg("--stop-on-stdin-close")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(stderr)
            .spawn()
            .expect("artemis starts"),
    );
    let stdin = child.0.stdin.take();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let client = Client::new("127.0.0.1", PORT, &key).unwrap();

    // `/health` answers 200 within 120 s of the start (the model loads on the first request, not before).
    // Every tokio future is built inside the runtime: `wait_for_health` is an `async fn`, so `block_on` is the first
    // thing to poll it.
    let health = runtime.block_on(wait_for_health(
        &client,
        HEALTH_WITHIN,
        Duration::from_millis(500),
        || child.0.try_wait().unwrap().map(|s| s.to_string()),
    ));
    if let Err(problem) = health {
        panic!("{problem}");
    }

    // One streamed request, thinking off. `t_last` is the arrival of the last data event before the silence (the
    // keep-alive comments don't count), and the process ends with the watchdog's 4.
    let body = json!({
        "messages": [{"role": "user", "content": "Count from one to forty, one number per line."}],
        "reasoning_effort": "none",
        "max_tokens": 64,
        "stream": true,
    });
    // Nothing in here asserts: a request that never starts, or a process that does not end in time, comes back as a
    // `problem`, so the figures below are written (with what is missing as null) before the test fails on it.
    let run = runtime.block_on(stream_until_end(
        &client,
        &body,
        FIRST_CHUNK_WITHIN,
        Duration::from_secs(stall_s + 60),
        || child.0.try_wait().unwrap().map(|s| s.code()),
    ));
    // A process still running (a problem above) is killed here, before its log is read.
    drop(stdin);
    drop(child);
    let silence = run
        .t_last
        .zip(run.t_end)
        .map(|(last, end)| end.saturating_duration_since(last));

    // The log's lifecycle lines (a run that never got as far has no directory), and the `stall` line of this run: the
    // one that starts the log's last three lines, `stall`, `stats` and `exit`.
    let lines = if logs.is_dir() {
        lifecycle_lines(&logs)
    } else {
        Vec::new()
    };
    let names: Vec<&str> = lines
        .iter()
        .map(|l| l["event"].as_str().unwrap_or(""))
        .collect();
    let tail_ok = names.ends_with(&["stall", "stats", "exit"]);
    let stall_at = tail_ok.then(|| lines.len() - 3);
    let stall = stall_at.map(|at| &lines[at]);
    // The last `load` before the stall line: a reused run directory holds the loads of earlier runs before it.
    let load_s = stall_at.and_then(|at| {
        lines[..at]
            .iter()
            .rev()
            .find(|l| l["event"] == "load")
            .and_then(|l| l["seconds"].as_f64())
    });
    let quiet_s = stall.and_then(|l| l["quiet_s"].as_f64());

    // The figures first, whatever came of the run, so a failed check below leaves them for the report. What the run
    // never produced is null.
    if let Some(dir) = &run_dir {
        let figures = json!({
            "load_s": load_s,
            "stall_s": stall_s,
            "quiet_s": quiet_s,
            "t_end_minus_t_last_s": silence.map(|s| s.as_secs_f64()),
            "content_chunks": run.chunks,
            "exit_code": run.code,
            "request": stall.map(|l| l["request"].clone()),
            "problem": run.problem,
        });
        std::fs::write(
            dir.join("stall.json"),
            serde_json::to_string_pretty(&figures).unwrap() + "\n",
        )
        .unwrap();
    }

    if let Some(problem) = &run.problem {
        panic!("{problem}");
    }
    assert_eq!(run.code, Some(4));
    let silence = silence.expect("the stream gave at least one data event");
    assert!(
        silence <= Duration::from_secs(stall_s + 10),
        "the process ended {silence:?} after the last data event"
    );
    assert!(
        (1..=15).contains(&run.chunks),
        "{} content chunks came before the stall at token {STALL_AT_TOKEN}",
        run.chunks
    );
    assert!(tail_ok, "the log ends with {names:?}");
    let stall = stall.unwrap();
    let quiet_s = quiet_s.unwrap_or(f64::NAN);
    assert_eq!(stall["doing"], "request");
    assert!(run.id.is_some(), "the stream carried no id");
    assert_eq!(stall["request"].as_str(), run.id.as_deref());
    assert_eq!(stall["bound_s"], stall_s);
    // The watchdog decides a stall `STALL_SLACK` (50 ms) early at most, to absorb tick jitter.
    assert!(quiet_s >= stall_s as f64 - 0.05, "quiet_s was {quiet_s}");
    assert_eq!(lines[lines.len() - 1]["reason"], "stall");

    // No file the run wrote holds the key (the key file itself is the one place it lives), and the port is free.
    let mut skip = vec![key_file.clone()];
    skip.extend(scratch.own_key_file.clone());
    for file in files_under(&scratch.dir, &skip) {
        let text = std::fs::read(&file).unwrap();
        assert!(
            !String::from_utf8_lossy(&text).contains(&key),
            "the key is in {}",
            file.display()
        );
    }
    assert!(port_is_free(PORT), "127.0.0.1:{PORT} is still taken");
}

// The harness above on the CPU: the same helpers, driven by the same kind of current-thread runtime and `block_on`,
// against the in-process server over the fake engine. A tokio call made outside the runtime panics here, in CI,
// rather than on the test PC after the model has loaded.
mod harness {
    use super::*;
    use server::app::Tuning;
    use server::config::Config;
    use server::fake::Script;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use support::{fake, parts, scratch_dir, start, tokenizer};

    /// An invented key-shaped literal, not a secret.
    const KEY: &str = "stall-harness-key-not-secret"; // gitleaks:allow

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    /// A server over a fake that says "one two three four five", and a client for it.
    fn rig(runtime: &tokio::runtime::Runtime) -> (support::Running, Client) {
        let fake = fake(Default::default());
        fake.push(Script::text(tokenizer(), "one two three four five"));
        let log_dir = scratch_dir("stall-harness");
        let cfg = Config {
            model: Some("m.gguf".into()),
            ctx: 65_536,
            ..Config::default()
        };
        let running = runtime.block_on(start(
            parts(fake, cfg, Some(KEY), &log_dir),
            Tuning::default(),
        ));
        let port: u16 = running
            .base_url
            .rsplit(':')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let client = Client::new("127.0.0.1", port, KEY).unwrap();
        (running, client)
    }

    fn body() -> Value {
        json!({
            "messages": [{"role": "user", "content": "Count."}],
            "reasoning_effort": "none",
            "max_tokens": 64,
            "stream": true,
        })
    }

    #[test]
    fn health_poll_answers_on_the_runtime() {
        let runtime = runtime();
        let (_running, client) = rig(&runtime);
        let polled = runtime.block_on(wait_for_health(
            &client,
            Duration::from_secs(30),
            Duration::from_millis(10),
            || None,
        ));
        assert_eq!(polled, Ok(()));
    }

    #[test]
    fn health_poll_stops_when_the_process_ended_or_the_budget_is_spent() {
        let runtime = runtime();
        // A port nothing listens on: every poll is refused at once.
        let free = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = free.local_addr().unwrap().port();
        drop(free);
        let client = Client::new("127.0.0.1", port, KEY).unwrap();
        let ended = runtime.block_on(wait_for_health(
            &client,
            Duration::from_secs(30),
            Duration::from_millis(10),
            || Some("exit code: 1".to_string()),
        ));
        assert_eq!(
            ended,
            Err("artemis ended before it answered /health: exit code: 1".to_string())
        );
        let spent = runtime.block_on(wait_for_health(
            &client,
            Duration::from_millis(300),
            Duration::from_millis(20),
            || None,
        ));
        assert_eq!(spent, Err("no /health within 300ms".to_string()));
    }

    #[test]
    fn the_stream_is_followed_until_the_process_ends() {
        let runtime = runtime();
        let (_running, client) = rig(&runtime);
        // The process "ends" with code 4 once the loop has looked at it a number of times: by then the stream is read.
        let looks = AtomicUsize::new(0);
        let run = runtime.block_on(stream_until_end(
            &client,
            &body(),
            Duration::from_secs(30),
            Duration::from_secs(30),
            || (looks.fetch_add(1, Ordering::SeqCst) >= 40).then_some(Some(4)),
        ));
        assert_eq!(run.problem, None);
        assert_eq!(run.code, Some(4));
        assert!(run.chunks >= 1, "{} content chunks", run.chunks);
        assert!(run.id.is_some());
        assert!(run.t_last.is_some() && run.t_end.is_some());
    }

    #[test]
    fn a_process_that_does_not_end_is_a_problem() {
        let runtime = runtime();
        let (_running, client) = rig(&runtime);
        let run = runtime.block_on(stream_until_end(
            &client,
            &body(),
            Duration::from_secs(30),
            Duration::from_millis(300),
            || None,
        ));
        assert_eq!(
            run.problem.as_deref(),
            Some("the process did not end in time")
        );
        assert_eq!(run.code, None);
    }

    #[test]
    fn a_request_that_cannot_start_is_a_problem() {
        let runtime = runtime();
        let free = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = free.local_addr().unwrap().port();
        drop(free);
        let client = Client::new("127.0.0.1", port, KEY).unwrap();
        let run = runtime.block_on(stream_until_end(
            &client,
            &body(),
            Duration::from_secs(30),
            Duration::from_secs(30),
            || None,
        ));
        assert!(
            run.problem
                .as_deref()
                .is_some_and(|p| p.starts_with("the request failed")),
            "{:?}",
            run.problem
        );
    }
}
