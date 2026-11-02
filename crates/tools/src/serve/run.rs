//! `artemis serve`: the command (milestone-3 spec 7, 8.1 and 8.3). It reads the config, checks the model against the
//! contract, builds the app over the engine's adapter and runs the server until a stop or an exit.
//!
//! Nothing here opens the port until the config, the key and the model have all passed, so a start that is refused
//! leaves no `start` line in the log. A refusal is an `anyhow` error: its message names the path or the key's file and
//! never the key's value, and `main` exits 1.
use super::backend::EngineBackend;
use super::exit_code;
use anyhow::{anyhow, Context};
use gguf::Gguf;
use qwen36::contract;
use qwen36::dims::Dims;
use qwen36::pins::{template_digest, vocab_digest, CHAT_TEMPLATE_SHA256, VOCAB_SHA256};
use qwen36::tokenizer::{chat_template, Tokenizer, Vocab};
use server::logs::{init_tracing, LifecycleLine, Logger};
use server::template::Template;
use server::tokens::Specials;
use server::{key_strength, runtime, serve, App, AppParts, Exit, ServeArgs, Tuning};
use std::future::{pending, Future};
use std::io::Read;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

/// How long `finish` waits for the engine's worker after `serve` returns. With `serve`'s own 30 s grace this stays
/// inside the 90 s the runner's `Stop-ArtemisServer` waits.
const FINISH_CAP: Duration = Duration::from_secs(20);

/// The flags of `artemis serve`: the server's, and a switch for scripts.
#[derive(clap::Args, Debug)]
pub struct ServeCmd {
    #[command(flatten)]
    pub args: ServeArgs,
    /// Stop as on Ctrl-C when standard input closes (for scripts on Windows, which can't send Ctrl-C to a child).
    #[arg(long, hide = true)]
    pub stop_on_stdin_close: bool,
    /// A test hook for the GPU stall test: every request stalls the engine's worker at its Nth token, so the watchdog
    /// fires. Not a config key, so no file sets it; refused on port 8080.
    #[arg(long, hide = true, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
    pub test_stall_at_token: Option<u32>,
}

/// Runs the server and returns the process's exit code (spec 8.1, 8.3): 0 for a requested stop, 2 after a second
/// device loss, 3 when the engine's thread is gone, 1 after a panic in the serve task. A refusal to start, or a
/// listener that won't bind, is the `Err` (exit 1 in `main`).
///
/// The body stays out of the caller's frame (`inline(never)`), so a single-call-site function is not folded into it.
#[inline(never)]
pub fn run(cmd: ServeCmd) -> anyhow::Result<i32> {
    let env = |k: &str| std::env::var(k).ok();
    let cfg = server::Config::load(&cmd.args, &env, cfg!(windows)).map_err(anyhow::Error::msg)?;
    // Before the log, the key or the model: the production port is never one a stall may be set up on.
    if cmd.test_stall_at_token.is_some() && cfg.port == 8080 {
        return Err(anyhow!("--test-stall-at-token is refused on port 8080"));
    }
    init_tracing(&cfg.log_level).map_err(anyhow::Error::msg)?;
    // Before the log: opening the `Logger` prunes old log files, and a refused start prunes nothing.
    let key = cfg.read_key().map_err(anyhow::Error::msg)?;
    let addr = cfg.bind(key.as_ref()).map_err(anyhow::Error::msg)?;
    let key_warning = match &key {
        Some(key) => key_strength(key, addr).map_err(anyhow::Error::msg)?,
        None => None,
    };

    let log_dir = cfg.log_dir(&env, cfg!(windows));
    let logger = Logger::new(log_dir.clone(), cfg.log_keep_days)
        .with_context(|| format!("cannot create the log directory {}", log_dir.display()))?;
    super::panic::install(logger.clone());

    let model = cfg
        .model
        .as_deref()
        .ok_or_else(|| anyhow!("model is required"))?;
    let (tokenizer, specials, template, pin_warning) = {
        let Checked {
            vocab,
            tokenizer,
            template: source,
            pin_warning,
            ..
        } = check_model(model, cfg.allow_custom_template)?;
        let template = Template::new(source)
            .map_err(|e| anyhow!("{}: the chat template: {e}", model.display()))?;
        let specials =
            Specials::new(&vocab, &tokenizer).map_err(|e| anyhow!("{}: {e}", model.display()))?;
        (tokenizer, specials, template, pin_warning)
    };

    // The first start measures this PC before anything is served, now that the model has passed. The port opens only
    // after this, and the config is read again so that the tuned values apply.
    let cfg = match crate::setup::first_run_path(&cfg) {
        Some(path) => {
            crate::setup::run_first(&cfg, &path, &logger)?;
            server::Config::load(&cmd.args, &env, cfg!(windows)).map_err(anyhow::Error::msg)?
        }
        None => cfg,
    };

    for message in config_warnings(&cfg, cmd.test_stall_at_token) {
        warn(&logger, &message);
    }
    for line in tuned_lines(&cfg, &env, cfg!(windows)) {
        logger.lifecycle(&line);
    }
    if let Some(message) = key_warning {
        warn(&logger, &message);
    }
    if let Some(message) = pin_warning {
        warn(&logger, &message);
    }

    let backend = Arc::new(EngineBackend::spawn(&cfg, logger.clone())?);
    backend.set_test_stall(cmd.test_stall_at_token);
    let lazy_load = cfg.lazy_load;
    let app = match App::new(AppParts {
        config: cfg,
        key,
        backend: backend.clone(),
        tokenizer: Arc::new(tokenizer),
        specials,
        template,
        logger,
        tuning: Tuning::default(),
        version: env!("CARGO_PKG_VERSION"),
    }) {
        Ok(app) => app,
        Err(e) => {
            backend.finish(FINISH_CAP);
            return Err(anyhow::Error::msg(e));
        }
    };
    // The cycle App -> backend -> hook -> App lasts as long as the process.
    let hooked = app.clone();
    backend.set_loss_hook(Box::new(move |m| hooked.device_lost(m)));

    // The future runs on a 2 MiB runtime worker, not on the calling thread. Past this point the worker is the
    // adapter's to stop: the way out of an error is `finish`, as for every other end.
    let rt = match runtime() {
        Ok(rt) => rt,
        Err(e) => {
            backend.finish(FINISH_CAP);
            return Err(anyhow::Error::new(e).context("cannot start the runtime"));
        }
    };
    let joined = rt.block_on(rt.spawn(serve_on(app, addr, lazy_load, cmd.stop_on_stdin_close)));
    drop(rt);
    // Whatever ended the serve future, the worker is stopped and joined (a bounded wait) before the process exits.
    backend.finish(FINISH_CAP);
    match joined {
        Ok(Ok(result)) => Ok(exit_code(&result)),
        Ok(Err(e)) => Err(e),
        Err(e) => {
            // The kind only: a panic's message is not for the log.
            tracing::error!(
                "the server task {}",
                if e.is_panic() {
                    "panicked"
                } else {
                    "was cancelled"
                }
            );
            Ok(1)
        }
    }
}

/// A model that passed the contract and the pins: what `serve` and `setup` both start from.
pub struct Checked {
    pub g: Gguf,
    pub vocab: Vocab,
    pub tokenizer: Tokenizer,
    /// The chat template's source.
    pub template: String,
    /// The `config_warning` text for an allowed custom template.
    pub pin_warning: Option<String>,
}

/// Opens `model` and checks it against the contract and the pins (`check_pins`), as `serve` does before it starts and
/// `setup` before it measures. Every refusal names the model's path.
pub fn check_model(model: &Path, allow_custom_template: bool) -> anyhow::Result<Checked> {
    let g =
        Gguf::open(model).with_context(|| format!("cannot open the model {}", model.display()))?;
    contract::validate(&g, &Dims::QWEN36)
        .map_err(|e| anyhow!("{} is not the supported model: {e}", model.display()))?;
    let vocab = Vocab::from_gguf(&g).map_err(|e| anyhow!("{}: {e}", model.display()))?;
    let tokenizer = Tokenizer::new(&vocab).map_err(|e| anyhow!("{}: {e}", model.display()))?;
    let template = chat_template(&g).map_err(|e| anyhow!("{}: {e}", model.display()))?;
    let pin_warning = check_pins(&template, &vocab, allow_custom_template)
        .map_err(|e| anyhow!("{}: {e}", model.display()))?;
    Ok(Checked {
        g,
        vocab,
        tokenizer,
        template,
        pin_warning,
    })
}

/// Checks the GGUF's vocabulary and chat template against the ones Artemis is tested with (`qwen36::pins`). A
/// different vocabulary is always refused. A different template is refused unless `allow_custom_template` is set, in
/// which case the `Ok` carries the `config_warning` text `run` writes.
pub fn check_pins(
    template: &str,
    vocab: &Vocab,
    allow_custom_template: bool,
) -> Result<Option<String>, String> {
    let found = vocab_digest(vocab);
    if found != VOCAB_SHA256 {
        return Err(format!(
            "the vocabulary differs from Qwen3.6-35B-A3B's (sha256 {found}); Artemis supports only that vocabulary"
        ));
    }
    let found = template_digest(template);
    if found == CHAT_TEMPLATE_SHA256 {
        return Ok(None);
    }
    if allow_custom_template {
        return Ok(Some(format!(
            "the GGUF's chat template differs from the one Artemis is tested with (sha256 {found}); \
             allow_custom_template is set, so it is used"
        )));
    }
    Err(format!(
        "the chat template differs from the one Artemis is tested with (sha256 {found}, expected \
         {CHAT_TEMPLATE_SHA256}); set allow_custom_template = true to use it"
    ))
}

/// The `config_warning` texts for a config, in the order `run` writes them: one for each setting that switches a
/// safeguard off, one for a load bound below the request bound (both nonzero: `stall_load_s = 0` only lifts the load
/// bound), then one for the test hook when it is set. Pure, because `run` reaches its warnings only after opening a
/// real GGUF.
pub fn config_warnings(cfg: &server::Config, test_stall: Option<u32>) -> Vec<String> {
    let mut out = Vec::new();
    if cfg.snapshot_pool_gib == 0 {
        out.push("snapshot_pool_gib is 0: prompt reuse is off".to_string());
    }
    if cfg.idle_unload_s == 0 {
        out.push("idle_unload_s is 0: the model never unloads when idle".to_string());
    }
    if cfg.stall_s == 0 {
        out.push("stall_s is 0: the stall watchdog is off".to_string());
    }
    if cfg.stall_s != 0 && cfg.stall_load_s != 0 && cfg.stall_load_s < cfg.stall_s {
        out.push(format!(
            "stall_load_s ({} s) is below stall_s ({} s): a load is given less time than a request",
            cfg.stall_load_s, cfg.stall_s
        ));
    }
    if let Some(n) = test_stall {
        out.push(format!(
            "a test hook is set: every request stalls at token {n}"
        ));
    }
    out
}

/// The tuned layer's lines for a start: the `tuned` line (the file's path, the keys it applied and the ones the user's
/// own settings outranked; empty lists for a file that is missing), then a `config_warning` for each sentence the layer
/// owes (an unplaced file, or the ways a loaded one is out of date). Nothing when the layer is off by choice.
pub fn tuned_lines(
    cfg: &server::Config,
    env: server::config::Env<'_>,
    windows: bool,
) -> Vec<LifecycleLine> {
    let mut out = Vec::new();
    match &cfg.tuned {
        server::TunedState::Loaded {
            path,
            applied,
            ignored,
            ..
        } => out.push(LifecycleLine::Tuned {
            path: path.display().to_string(),
            applied: applied.clone(),
            ignored: ignored.clone(),
        }),
        server::TunedState::Missing(path) => out.push(LifecycleLine::Tuned {
            path: path.display().to_string(),
            applied: Vec::new(),
            ignored: Vec::new(),
        }),
        server::TunedState::Off => {}
    }
    for message in cfg.tuned_warnings(env, windows) {
        out.push(LifecycleLine::ConfigWarning { message });
    }
    out
}

fn warn(logger: &Logger, message: &str) {
    logger.lifecycle(&LifecycleLine::ConfigWarning {
        message: message.into(),
    });
}

/// Opens the port. A port another program holds is an error that names the address, which `main` prints and exits 1.
pub async fn bind_listener(addr: SocketAddr) -> anyhow::Result<TcpListener> {
    TcpListener::bind(addr)
        .await
        .with_context(|| format!("cannot listen on {addr}"))
}

/// Binds, starts the load when the config asks for it at start, and serves until a stop or an exit.
async fn serve_on(
    app: App,
    addr: SocketAddr,
    lazy_load: bool,
    stop_on_stdin_close: bool,
) -> anyhow::Result<Result<(), Exit>> {
    let listener = bind_listener(addr).await?;
    if !lazy_load {
        // A failed load leaves the server idle, and the next request tries again.
        let reply = app.backend().load();
        tokio::spawn(async move {
            match reply.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!("the load at start failed: {}", e.message),
                Err(_) => tracing::warn!("the load at start failed: the engine stopped"),
            }
        });
    }
    Ok(serve(app, listener, shutdown(stop_on_stdin_close)).await)
}

/// Resolves on the first of Ctrl-C, Ctrl-Break (Windows) and, when asked, standard input closing.
async fn shutdown(stop_on_stdin_close: bool) {
    tokio::select! {
        () = ctrl_c() => {}
        () = ctrl_break() => {}
        () = stdin_closed(stop_on_stdin_close) => {}
    }
}

/// A signal that can't be registered never fires: it must not stop the server at once.
async fn or_never<T>(what: &str, r: std::io::Result<T>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                "cannot listen for {what} ({:?}): that stop is off",
                e.kind()
            );
            pending().await
        }
    }
}

async fn ctrl_c() {
    or_never("Ctrl-C", tokio::signal::ctrl_c().await).await;
}

#[cfg(windows)]
async fn ctrl_break() {
    let mut signal = or_never("Ctrl-Break", tokio::signal::windows::ctrl_break()).await;
    if signal.recv().await.is_none() {
        pending::<()>().await;
    }
}

#[cfg(not(windows))]
fn ctrl_break() -> impl Future<Output = ()> {
    pending()
}

/// With the switch, resolves when standard input reaches its end (or can no longer be read): the script that started
/// the server is gone or has asked it to stop. Without it, never.
fn stdin_closed(on: bool) -> impl Future<Output = ()> {
    let closed = on.then(stdin_watch).flatten();
    async move {
        match closed {
            Some(rx) => {
                // The sender is dropped only after it sends.
                let _ = rx.await;
            }
            None => pending().await,
        }
    }
}

/// A thread that reads standard input to its end and then completes the receiver. `None` when the thread can't start.
fn stdin_watch() -> Option<oneshot::Receiver<()>> {
    let (tx, rx) = oneshot::channel();
    let spawned = std::thread::Builder::new()
        .name("artemis-stdin".into())
        .spawn(move || {
            let mut buf = [0u8; 256];
            let mut stdin = std::io::stdin().lock();
            loop {
                match stdin.read(&mut buf) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(_) => break,
                }
            }
            let _ = tx.send(());
        });
    match spawned {
        Ok(_) => Some(rx),
        Err(e) => {
            tracing::warn!(
                "cannot watch standard input ({:?}): that stop is off",
                e.kind()
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_busy_port_is_refused_naming_it() {
        let held = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = held.local_addr().unwrap();
        let err = bind_listener(addr).await.unwrap_err();
        assert!(
            format!("{err:#}").contains(&format!("cannot listen on 127.0.0.1:{}", addr.port())),
            "{err:#}"
        );
        assert!(
            err.to_string()
                .contains(&format!("cannot listen on {addr}")),
            "the outer message names the address: {err}"
        );
        assert!(
            err.chain().count() >= 2,
            "the OS error stays as the cause: {err:#}"
        );
        drop(held);
        assert!(bind_listener(addr).await.is_ok(), "a freed port binds");
    }

    /// Each setting that switches a safeguard off is named once, in this order; the defaults name none.
    #[test]
    fn config_warnings_name_each_off_switch() {
        let default = server::Config::default();
        assert_eq!(config_warnings(&default, None), Vec::<String>::new());

        let mut cfg = server::Config {
            snapshot_pool_gib: 0,
            ..default.clone()
        };
        assert_eq!(
            config_warnings(&cfg, None),
            ["snapshot_pool_gib is 0: prompt reuse is off"]
        );
        cfg.idle_unload_s = 0;
        cfg.stall_s = 0;
        assert_eq!(
            config_warnings(&cfg, None),
            [
                "snapshot_pool_gib is 0: prompt reuse is off",
                "idle_unload_s is 0: the model never unloads when idle",
                "stall_s is 0: the stall watchdog is off",
            ]
        );
        let only_idle = server::Config {
            idle_unload_s: 0,
            ..default.clone()
        };
        assert_eq!(
            config_warnings(&only_idle, None),
            ["idle_unload_s is 0: the model never unloads when idle"]
        );
        let only_stall = server::Config {
            stall_s: 0,
            ..default
        };
        assert_eq!(
            config_warnings(&only_stall, None),
            ["stall_s is 0: the stall watchdog is off"]
        );
    }

    /// A load bound below the request bound is named, unless either is 0 (off, or only the load bound lifted); the
    /// warning follows the off-switch ones.
    #[test]
    fn config_warnings_name_a_load_bound_below_the_request_bound() {
        let default = server::Config::default();
        let with = |stall_s, stall_load_s| server::Config {
            stall_s,
            stall_load_s,
            ..default.clone()
        };
        assert_eq!(
            config_warnings(&with(120, 60), None),
            ["stall_load_s (60 s) is below stall_s (120 s): a load is given less time than a request"]
        );
        // Equal or above, and either at 0, name nothing about the pair.
        for (s, l) in [(120, 120), (120, 300), (0, 60), (120, 0), (0, 0)] {
            let got = config_warnings(&with(s, l), None);
            assert!(
                !got.iter().any(|m| m.contains("stall_load_s")),
                "{s} {l}: {got:?}"
            );
        }
        let cfg = server::Config {
            idle_unload_s: 0,
            ..with(120, 60)
        };
        assert_eq!(
            config_warnings(&cfg, Some(3)),
            [
                "idle_unload_s is 0: the model never unloads when idle",
                "stall_load_s (60 s) is below stall_s (120 s): a load is given less time than a request",
                "a test hook is set: every request stalls at token 3",
            ]
        );
    }

    /// The hook's text comes last, after any off-switch warnings, and only when the hook is set.
    #[test]
    fn config_warnings_name_the_test_hook_last() {
        let default = server::Config::default();
        assert_eq!(
            config_warnings(&default, Some(3)),
            ["a test hook is set: every request stalls at token 3"]
        );
        let cfg = server::Config {
            stall_s: 0,
            ..default
        };
        assert_eq!(
            config_warnings(&cfg, Some(16)),
            [
                "stall_s is 0: the stall watchdog is off",
                "a test hook is set: every request stalls at token 16",
            ]
        );
    }

    fn warning(m: impl Into<String>) -> LifecycleLine {
        LifecycleLine::ConfigWarning { message: m.into() }
    }

    /// A1: the tuned layer's lines are the `tuned` line and one `config_warning` for each sentence it owes.
    #[test]
    fn the_tuned_lines_name_the_file_and_write_each_warning() {
        use server::TunedState;
        let with = |tuned| server::Config {
            tuned,
            ..server::Config::default()
        };
        let no_env = |_: &str| None::<String>;
        // No place for the file: one warning, no `tuned` line.
        assert_eq!(
            tuned_lines(&with(TunedState::Off), &no_env, true),
            [warning(server::tuned::unplaced_warning(true))]
        );
        assert_eq!(
            tuned_lines(&with(TunedState::Off), &no_env, false),
            [warning(server::tuned::unplaced_warning(false))]
        );
        // The layer off by choice: nothing.
        let off = |k: &str| (k == "ARTEMIS_TUNED").then(|| "off".to_string());
        assert!(tuned_lines(&with(TunedState::Off), &off, true).is_empty());
        // A file that is missing: the `tuned` line with nothing applied.
        let at = std::path::PathBuf::from("T/tuned.toml");
        let some = |k: &str| (k == "LOCALAPPDATA").then(|| "C:/L".to_string());
        assert_eq!(
            tuned_lines(&with(TunedState::Missing(at.clone())), &some, true),
            [LifecycleLine::Tuned {
                path: "T/tuned.toml".into(),
                applied: vec![],
                ignored: vec![],
            }]
        );
        // A loaded file: the `tuned` line, then one warning for each stale sentence.
        let loaded = TunedState::Loaded {
            path: at,
            applied: vec!["speculation.enabled".into()],
            ignored: vec!["device_uuid".into()],
            stale: vec!["first sentence".into(), "second sentence".into()],
            gpu_name: "g".into(),
            gpu_uuid: "u".into(),
            driver: "d".into(),
        };
        assert_eq!(
            tuned_lines(&with(loaded), &some, true),
            [
                LifecycleLine::Tuned {
                    path: "T/tuned.toml".into(),
                    applied: vec!["speculation.enabled".into()],
                    ignored: vec!["device_uuid".into()],
                },
                warning("first sentence"),
                warning("second sentence"),
            ]
        );
    }

    fn committed(rel: &str) -> Vec<u8> {
        let path = format!("{}/../../testdata/{rel}", env!("CARGO_MANIFEST_DIR"));
        std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    #[test]
    fn check_pins_refuses_a_changed_vocabulary_or_template() {
        let template = String::from_utf8(committed("template/qwen36-chat-template.jinja")).unwrap();
        let vocab = Vocab::from_extract(&committed("tokenizer/qwen36-vocab.bin")).unwrap();
        assert_eq!(check_pins(&template, &vocab, false), Ok(None));
        assert_eq!(check_pins(&template, &vocab, true), Ok(None));

        // One more space: both digests are named, and the key lets it through with a warning.
        let edited = format!("{template} ");
        let found = template_digest(&edited);
        let e = check_pins(&edited, &vocab, false).unwrap_err();
        assert_eq!(
            e,
            format!(
                "the chat template differs from the one Artemis is tested with (sha256 {found}, expected {CHAT_TEMPLATE_SHA256}); set allow_custom_template = true to use it"
            )
        );
        let w = check_pins(&edited, &vocab, true).unwrap().unwrap();
        assert_eq!(
            w,
            format!(
                "the GGUF's chat template differs from the one Artemis is tested with (sha256 {found}); allow_custom_template is set, so it is used"
            )
        );

        // One token changed: refused, with the key as well.
        let mut other = vocab.clone();
        other.tokens[1000].push('x');
        let found = vocab_digest(&other);
        for allow in [false, true] {
            let e = check_pins(&template, &other, allow).unwrap_err();
            assert_eq!(
                e,
                format!(
                    "the vocabulary differs from Qwen3.6-35B-A3B's (sha256 {found}); Artemis supports only that vocabulary"
                )
            );
        }
        // The vocabulary is checked first.
        assert!(check_pins(&edited, &other, true).is_err());
    }
}
