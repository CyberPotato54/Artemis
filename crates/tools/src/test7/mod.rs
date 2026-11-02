//! `artemis test7`: the driver that runs the real-engine cases against a running server.
//! Each case writes `OUT/<case>.json` (`{case, status, checks, figures}`), which the runner `Record`s. The
//! command exits 0 when it wrote every verdict and 2 on a harness error (an unreachable server, an unreadable log, a
//! refused URL). The key is read from a file, sent as `x-api-key` and as a bearer token, and never printed, logged
//! or written to a verdict. The URL is checked before any connection: a mistyped one must never send `/v1/unload`
//! and a key to the production server on 8080.
pub mod cases;
pub mod http;
pub mod judge;
pub mod log;
pub mod report;

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The cases `--case` names, in the order the runner's blocks run them.
pub const CASES: &[&str] = &[
    "load",
    "requests",
    "budget",
    "reuse",
    "disconnect",
    "speed",
    "speed-fresh",
    "chat-tools",
    "idle",
    "unload-stream",
    "queue",
];

/// The window for `queue`: Q answers 503 this long after it was sent (server C's queue timeout is 5 s).
pub const QUEUE_WINDOW: (Duration, Duration) = (Duration::from_secs(5), Duration::from_secs(8));
/// The window for `idle`: the idle unload comes this long after C's request ended (server B's is 30 s).
pub const IDLE_WINDOW: (Duration, Duration) = (Duration::from_secs(30), Duration::from_secs(45));

#[derive(clap::Args, Debug)]
pub struct Test7Args {
    /// The server's base URL: `http://127.0.0.1:PORT` or `http://localhost:PORT`, never port 8080.
    #[arg(long)]
    pub base_url: String,
    /// The file holding the server's API key.
    #[arg(long)]
    pub key_file: PathBuf,
    /// The server's log directory (`--log-dir` of `artemis serve`).
    #[arg(long)]
    pub log_dir: PathBuf,
    /// The cases to run, in order: load, requests, budget, reuse, disconnect, speed, speed-fresh, chat-tools, idle,
    /// unload-stream, queue.
    #[arg(long, value_delimiter = ',', required = true)]
    pub case: Vec<String>,
    /// The run `speed-fresh` makes, one per server process: `<write|edit|explain>:<1234|1235>`, or `long`. Given with
    /// `--case speed-fresh` alone, and required by it.
    #[arg(long)]
    pub speed_run: Option<String>,
    /// Where the verdicts go, one `<case>.json` each.
    #[arg(long)]
    pub out: PathBuf,
    /// The repository root, for the cases that read its test data.
    #[arg(long, default_value = ".")]
    pub prompts_root: PathBuf,
}

/// The one request `speed-fresh` makes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpeedRun {
    /// A prompt of `testdata/speed/prompts.json` (`write`, `edit` or `explain`) with one of its seeds.
    Decode { prompt: String, seed: u64 },
    /// Golden prompt 20, unwrapped, with one token asked for: a prefill alone.
    Long,
}

/// The prompts and seeds `--speed-run` accepts: those of `testdata/speed/prompts.json`.
const SPEED_PROMPTS: [&str; 3] = ["write", "edit", "explain"];
const SPEED_SEEDS: [u64; 2] = [1234, 1235];

/// `--speed-run`'s value: `long`, or `<prompt>:<seed>`. A refusal never repeats the value.
pub fn parse_speed_run(text: &str) -> Result<SpeedRun> {
    if text == "long" {
        return Ok(SpeedRun::Long);
    }
    let shape = || {
        anyhow!(
            "--speed-run takes <{}>:<{}> or long",
            SPEED_PROMPTS.join("|"),
            SPEED_SEEDS.map(|s| s.to_string()).join("|")
        )
    };
    let (prompt, seed) = text.split_once(':').ok_or_else(shape)?;
    let seed: u64 = seed.parse().map_err(|_| shape())?;
    if !SPEED_PROMPTS.contains(&prompt) || !SPEED_SEEDS.contains(&seed) {
        return Err(shape());
    }
    Ok(SpeedRun::Decode {
        prompt: prompt.to_string(),
        seed,
    })
}

/// What a case runs with.
pub struct Ctx {
    pub http: http::Client,
    pub log_dir: PathBuf,
    pub prompts_root: PathBuf,
    /// Test hooks: the fake-backed tests set them, the command never does. `queue_window` and `idle_window` scale the
    /// default durations to a fake's shorter timeouts (`QUEUE_WINDOW`, `IDLE_WINDOW` by default).
    pub queue_window: (Duration, Duration),
    pub idle_window: (Duration, Duration),
    /// Lets a load line's missing figures (`vram_used_mib`, `slots_per_layer`, `pool_limit_gib`: the fake writes
    /// `null`) pass instead of fail. `false` by default.
    pub allow_missing_vram: bool,
    /// Lets the `speed` case skip the `edit` prompt when its pinned source is not on disk (the committed `encoder.py`,
    /// absent only from a fake's scratch root) with a note, instead of failing. Any other prompt's missing source still
    /// fails. `false` by default; only the fake test sets it.
    pub allow_missing_edit: bool,
    /// The run `speed-fresh` makes; `None` for every other case.
    pub speed_run: Option<SpeedRun>,
}

impl Ctx {
    /// A context for `base_url` (checked here) holding `key`.
    pub fn new(base_url: &str, key: &str, log_dir: PathBuf, prompts_root: PathBuf) -> Result<Ctx> {
        let (host, port) = check_base_url(base_url)?;
        Ok(Ctx {
            http: http::Client::new(&host, port, key)?,
            log_dir,
            prompts_root,
            queue_window: QUEUE_WINDOW,
            idle_window: IDLE_WINDOW,
            allow_missing_vram: false,
            allow_missing_edit: false,
            speed_run: None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Pass,
    Fail,
    /// Run and reported, with nothing gated.
    Ran,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Check {
    pub name: String,
    pub pass: bool,
    pub detail: String,
}

impl Check {
    pub fn new(name: impl Into<String>, pass: bool, detail: impl Into<String>) -> Check {
        Check {
            name: name.into(),
            pass,
            detail: detail.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Verdict {
    pub case: String,
    pub status: Status,
    pub checks: Vec<Check>,
    pub figures: serde_json::Map<String, serde_json::Value>,
}

impl Verdict {
    /// A gated case: `pass` when every check passed.
    pub fn judged(
        case: &str,
        checks: Vec<Check>,
        figures: serde_json::Map<String, serde_json::Value>,
    ) -> Verdict {
        let status = if checks.iter().all(|c| c.pass) {
            Status::Pass
        } else {
            Status::Fail
        };
        Verdict {
            case: case.to_string(),
            status,
            checks,
            figures,
        }
    }

    /// A case that is reported, not gated.
    pub fn ran(
        case: &str,
        checks: Vec<Check>,
        figures: serde_json::Map<String, serde_json::Value>,
    ) -> Verdict {
        Verdict {
            status: Status::Ran,
            ..Verdict::judged(case, checks, figures)
        }
    }

    /// The names of the checks that failed.
    pub fn failed(&self) -> Vec<&str> {
        self.checks
            .iter()
            .filter(|c| !c.pass)
            .map(|c| c.name.as_str())
            .collect()
    }
}

/// Runs the case `name`. A failed check is a `fail` verdict; an `Err` is a harness error (exit 2).
pub async fn run_case(ctx: &Ctx, name: &str) -> Result<Verdict> {
    match name {
        "load" => cases::load(ctx).await,
        "requests" => cases::requests(ctx).await,
        "budget" => cases::budget(ctx).await,
        "reuse" => cases::reuse(ctx).await,
        "disconnect" => cases::disconnect(ctx).await,
        "speed" => cases::speed(ctx).await,
        "speed-fresh" => cases::speed_fresh(ctx).await,
        "chat-tools" => cases::chat_tools(ctx).await,
        "idle" => cases::idle(ctx).await,
        "unload-stream" => cases::unload_stream(ctx).await,
        "queue" => cases::queue(ctx).await,
        other => bail!("unknown case {other:?}; the cases are {}", CASES.join(", ")),
    }
}

/// Writes `verdict` to `dir/<case>.json` (made if missing) and returns the path.
pub fn write_verdict(dir: &Path, verdict: &Verdict) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    let path = dir.join(format!("{}.json", verdict.case));
    let mut text = serde_json::to_string_pretty(verdict)?;
    text.push('\n');
    std::fs::write(&path, text).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(path)
}

/// The host and port of `url`, if it is `http://127.0.0.1:PORT` or `http://localhost:PORT` (a trailing `/` allowed)
/// and the port is not 8080, where the production server listens. Anything else is refused.
/// A refusal may name the host, which holds no secret, but never the rest of the URL (a path, a user, a query).
pub fn check_base_url(url: &str) -> Result<(String, u16)> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| anyhow!("--base-url must start with http://"))?;
    let rest = rest.strip_suffix('/').unwrap_or(rest);
    if rest.is_empty() || rest.contains(['/', '@', '?', '#', '\\', ' ']) {
        bail!("--base-url must be http://127.0.0.1:PORT or http://localhost:PORT, with no path or user");
    }
    let (host, port) = rest
        .rsplit_once(':')
        .ok_or_else(|| anyhow!("--base-url names no port"))?;
    let host = host.to_ascii_lowercase();
    if host != "127.0.0.1" && host != "localhost" {
        bail!("--base-url's host must be 127.0.0.1 or localhost, not {host}");
    }
    if port.is_empty() || !port.bytes().all(|b| b.is_ascii_digit()) {
        bail!("--base-url's port is not a number");
    }
    let port: u16 = port
        .parse()
        .map_err(|_| anyhow!("--base-url's port is out of range"))?;
    if port == 8080 {
        bail!("--base-url names port 8080, where the production server listens; the driver never talks to it");
    }
    Ok((host, port))
}

/// The key in `path`, read as the server reads its key file: the first non-empty line, trimmed (a BOM is not part of
/// it). The error names the file, never its contents.
fn read_key(path: &Path) -> Result<String> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read the key file {}", path.display()))?;
    let key = text
        .strip_prefix('\u{feff}')
        .unwrap_or(&text)
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty());
    match key {
        Some(key) => Ok(key.to_string()),
        None => bail!("the key file {} is empty", path.display()),
    }
}

/// The run `--speed-run` names: given exactly when `--case` is `speed-fresh` alone (one process, one run), and valid.
fn speed_run_of(args: &Test7Args) -> Result<Option<SpeedRun>> {
    let fresh = args.case == ["speed-fresh"];
    match (&args.speed_run, fresh) {
        (Some(text), true) => parse_speed_run(text).map(Some),
        (Some(_), false) => bail!("--speed-run is only for --case speed-fresh, by itself"),
        (None, true) => bail!("the case speed-fresh needs --speed-run <prompt>:<seed> or long"),
        (None, false) => {
            if args.case.iter().any(|c| c == "speed-fresh") {
                bail!("the case speed-fresh needs --speed-run, and runs by itself");
            }
            Ok(None)
        }
    }
}

/// Runs the cases of `args` in order, writing each verdict as it ends. 0 when every verdict was written (whatever
/// they say), and an `Err` for a harness error, which `main` turns into exit 2.
#[inline(never)]
pub fn run(args: Test7Args) -> Result<i32> {
    // Before anything else is read or opened: the URL, then the case names.
    check_base_url(&args.base_url)?;
    for c in &args.case {
        if !CASES.contains(&c.as_str()) {
            bail!("unknown case {c:?}; the cases are {}", CASES.join(", "));
        }
    }
    let speed_run = speed_run_of(&args)?;
    let key = read_key(&args.key_file)?;
    let mut ctx = Ctx::new(
        &args.base_url,
        &key,
        args.log_dir.clone(),
        args.prompts_root.clone(),
    )?;
    ctx.speed_run = speed_run;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("cannot start the runtime")?;
    rt.block_on(async {
        for name in &args.case {
            let verdict = run_case(&ctx, name)
                .await
                .with_context(|| format!("case {name}"))?;
            let path = write_verdict(&args.out, &verdict)?;
            println!(
                "test7: {name}: {:?}{} ({})",
                verdict.status,
                if verdict.failed().is_empty() {
                    String::new()
                } else {
                    format!(", failed: {}", verdict.failed().join(", "))
                },
                path.display()
            );
        }
        Ok::<i32, anyhow::Error>(0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_must_be_loopback_and_not_8080() {
        assert_eq!(
            check_base_url("http://127.0.0.1:8090").unwrap(),
            ("127.0.0.1".to_string(), 8090)
        );
        assert_eq!(
            check_base_url("http://localhost:8090").unwrap(),
            ("localhost".to_string(), 8090)
        );
        assert_eq!(check_base_url("http://127.0.0.1:8090/").unwrap().1, 8090);
        for bad in [
            "http://127.0.0.1:8080",
            "http://localhost:8080",
            "http://127.0.0.1:08080",
            "http://10.0.0.5:8090",
            "http://example.com:8090",
            "https://127.0.0.1:8090",
            "http://127.0.0.1",
            "http://127.0.0.1:",
            "http://127.0.0.1:99999",
            "http://127.0.0.1:8090/v1",
            "http://127.0.0.1@example.com:8090",
            "http://[::1]:8090",
            "127.0.0.1:8090",
        ] {
            assert!(check_base_url(bad).is_err(), "{bad} was accepted");
        }
    }

    /// The URL is refused before the key file is read or a connection opened: the error is the URL's.
    #[test]
    fn a_refused_url_is_refused_before_anything_else() {
        let err = run(Test7Args {
            base_url: "http://127.0.0.1:8080".into(),
            key_file: PathBuf::from("no-such-key-file"),
            log_dir: PathBuf::from("no-such-log-dir"),
            case: vec!["load".into()],
            out: PathBuf::from("no-such-out"),
            prompts_root: PathBuf::from("."),
            speed_run: None,
        })
        .err()
        .unwrap();
        assert!(err.to_string().contains("8080"), "{err}");
    }

    /// The key file is read as the server reads it: the first non-empty line, trimmed; an empty file is an error.
    #[test]
    fn the_key_is_the_first_non_empty_line_trimmed() {
        let dir =
            std::env::temp_dir().join(format!("artemis-test7-readkey-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let read = |name: &str, text: &str| {
            let path = dir.join(name);
            std::fs::write(&path, text).unwrap();
            read_key(&path)
        };
        assert_eq!(read("plain", "abc\n").unwrap(), "abc");
        assert_eq!(
            read("blank-first", "\r\n  \n  abc \r\nsecond\n").unwrap(),
            "abc"
        );
        assert_eq!(read("two-lines", "abc\nsecond line\n").unwrap(), "abc");
        assert_eq!(read("bom", "\u{feff}abc\n").unwrap(), "abc");
        for empty in ["", "\n", " \r\n\t\n"] {
            let err = read("empty", empty).unwrap_err().to_string();
            assert!(err.contains("empty"), "{err}");
        }
        let err = read_key(&dir.join("absent")).unwrap_err().to_string();
        assert!(err.contains("cannot read"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_unknown_case_is_refused_before_the_key_is_read() {
        let err = run(Test7Args {
            base_url: "http://127.0.0.1:8090".into(),
            key_file: PathBuf::from("no-such-key-file"),
            log_dir: PathBuf::from("no-such-log-dir"),
            case: vec!["load".into(), "nonesuch".into()],
            out: PathBuf::from("no-such-out"),
            prompts_root: PathBuf::from("."),
            speed_run: None,
        })
        .err()
        .unwrap();
        assert!(err.to_string().contains("nonesuch"), "{err}");
    }

    #[test]
    fn a_verdict_is_pass_only_when_every_check_passes() {
        let ok = Check::new("a", true, "");
        let bad = Check::new("b", false, "no");
        let m = serde_json::Map::new;
        assert_eq!(
            Verdict::judged("c", vec![ok.clone()], m()).status,
            Status::Pass
        );
        let v = Verdict::judged("c", vec![ok.clone(), bad.clone()], m());
        assert_eq!((v.status, v.failed()), (Status::Fail, vec!["b"]));
        assert_eq!(Verdict::ran("c", vec![ok, bad], m()).status, Status::Ran);
        let json = serde_json::to_value(Verdict::judged("c", vec![], m())).unwrap();
        assert_eq!(json["status"], "pass");
    }

    /// `Test7Args` for `cases` and `speed_run`, with nothing on disk: only what `run` refuses before it reads the key.
    fn args(cases: &[&str], speed_run: Option<&str>) -> Test7Args {
        Test7Args {
            base_url: "http://127.0.0.1:8090".into(),
            key_file: PathBuf::from("no-such-key-file"),
            log_dir: PathBuf::from("no-such-log-dir"),
            case: cases.iter().map(|c| c.to_string()).collect(),
            out: PathBuf::from("no-such-out"),
            prompts_root: PathBuf::from("."),
            speed_run: speed_run.map(str::to_string),
        }
    }

    #[test]
    fn speed_run_is_refused_for_other_cases() {
        for cases in [
            &["speed"][..],
            &["load", "speed-fresh"],
            &["speed-fresh", "speed"],
        ] {
            let err = run(args(cases, Some("write:1234")))
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("--speed-run") && err.contains("speed-fresh"),
                "{err}"
            );
            assert!(
                !err.contains("no-such-key-file"),
                "the key file was read first: {err}"
            );
        }
    }

    #[test]
    fn speed_fresh_needs_a_speed_run() {
        let err = run(args(&["speed-fresh"], None)).unwrap_err().to_string();
        assert!(err.contains("--speed-run"), "{err}");
        assert!(
            !err.contains("no-such-key-file"),
            "the key file was read first: {err}"
        );
    }

    #[test]
    fn a_speed_run_names_a_prompt_and_a_seed_or_long() {
        assert_eq!(parse_speed_run("long").unwrap(), SpeedRun::Long);
        for (text, prompt, seed) in [
            ("write:1234", "write", 1234),
            ("edit:1235", "edit", 1235),
            ("explain:1234", "explain", 1234),
        ] {
            assert_eq!(
                parse_speed_run(text).unwrap(),
                SpeedRun::Decode {
                    prompt: prompt.into(),
                    seed
                }
            );
        }
        for bad in [
            "",
            "write",
            "write:",
            "write:1236",
            "write:0",
            "nonesuch:1234",
            ":1234",
            "Long",
            "long:1234",
            "write:1234:1",
            "write: 1234",
        ] {
            assert!(parse_speed_run(bad).is_err(), "{bad:?} was accepted");
        }
        // A refusal names the value's shape, not an arbitrary long text.
        let err = parse_speed_run(&"x".repeat(500)).unwrap_err().to_string();
        assert!(err.len() < 300, "{err}");
    }
}
