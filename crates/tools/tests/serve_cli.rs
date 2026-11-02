//! `artemis serve` refuses to start on bad input, before it binds. Each test runs the
//! binary with `--port 0` and a scratch `--config`, so no user config file is read and no fixed port is touched, and
//! asserts the exit code, that the message names the path or the key's file, and that the log directory holds no
//! `start` line: `serve` writes `start` first, so without one the port never opened. No test loads a model: the
//! contract check refuses the tiny one.
mod support;

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use support::{events, scratch_dir};

/// An invented key-shaped literal, not a secret.
const KEY: &str = "test-key-not-secret-4242"; // gitleaks:allow

/// A scratch directory holding an empty config file, and the log directory `serve` is told to use.
struct Scratch {
    dir: PathBuf,
    config: PathBuf,
    logs: PathBuf,
}

fn scratch(tag: &str) -> Scratch {
    let dir = scratch_dir(tag);
    let config = dir.join("artemis.toml");
    std::fs::write(&config, "").unwrap();
    Scratch {
        logs: dir.join("logs"),
        config,
        dir,
    }
}

/// `artemis serve --port 0 --config <scratch> --log-dir <scratch> --stop-on-stdin-close` and `extra`; stdin is closed,
/// so a server that did start (a regression) stops at once, and fails the exit code assertion, rather than hang CI.
fn serve(s: &Scratch, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_artemis"))
        .env("ARTEMIS_TUNED", "off")
        .args(["serve", "--port", "0", "--config"])
        .arg(&s.config)
        .arg("--log-dir")
        .arg(&s.logs)
        .arg("--stop-on-stdin-close")
        .args(extra)
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

/// A path as a command-line argument (the scratch paths are UTF-8).
fn arg(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// No `start` line: the port never opened. A log directory that was never made holds none either.
fn assert_never_started(s: &Scratch) {
    if s.logs.exists() {
        let events = events(&s.logs);
        assert!(
            !events.iter().any(|e| e == "start"),
            "serve started: {events:?}"
        );
    }
}

fn write_tiny(s: &Scratch) -> PathBuf {
    let model = s.dir.join("tiny.gguf");
    std::fs::write(&model, qwen36::testing::tiny_model(1)).unwrap();
    model
}

#[test]
fn a_missing_model_refuses_to_start() {
    let s = scratch("serve-missing-model");
    let model = s.dir.join("no-such-model.gguf");
    let out = serve(&s, &["--model", arg(&model)]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("no-such-model.gguf"),
        "{}",
        stderr(&out)
    );
    assert_never_started(&s);
}

#[test]
fn a_contract_mismatch_refuses_to_start() {
    let s = scratch("serve-contract");
    let model = write_tiny(&s);
    let out = serve(&s, &["--model", arg(&model)]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    // The first difference, as `ContractError` words it.
    let err = stderr(&out);
    assert!(
        err.contains("metadata ") && err.contains("expected"),
        "{err}"
    );
    assert_never_started(&s);
}

#[test]
fn a_missing_config_file_refuses_to_start() {
    let mut s = scratch("serve-missing-config");
    s.config = s.dir.join("no-such-config.toml");
    let model = write_tiny(&s);
    let out = serve(&s, &["--model", arg(&model)]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("no-such-config.toml"),
        "{}",
        stderr(&out)
    );
    assert_never_started(&s);
}

/// The key is read before the model is checked, so a valid key file and a refused model put the key in the process's
/// memory and then fail: nothing the process prints or logs may hold it. A key file that can't be read is named.
#[test]
fn a_startup_error_never_prints_the_key() {
    let s = scratch("serve-key");
    let key_file = s.dir.join("api-key");
    std::fs::write(&key_file, format!("{KEY}\r\n")).unwrap();
    let model = write_tiny(&s);
    let out = serve(
        &s,
        &["--model", arg(&model), "--api-key-file", arg(&key_file)],
    );
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(stderr(&out).contains("metadata "), "{}", stderr(&out));
    assert!(!stderr(&out).contains(KEY), "the key is on stderr");
    assert!(!String::from_utf8_lossy(&out.stdout).contains(KEY));
    if s.logs.exists() {
        for entry in std::fs::read_dir(&s.logs).unwrap() {
            let text = std::fs::read_to_string(entry.unwrap().path()).unwrap();
            assert!(!text.contains(KEY), "the key is in the log directory");
        }
    }
    assert_never_started(&s);

    let missing = s.dir.join("no-such-key");
    let out = serve(
        &s,
        &["--model", arg(&model), "--api-key-file", arg(&missing)],
    );
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(stderr(&out).contains("no-such-key"), "{}", stderr(&out));
    assert!(!stderr(&out).contains(KEY));
    assert_never_started(&s);
}

#[test]
fn no_key_and_a_non_loopback_host_refuse() {
    let s = scratch("serve-open-host");
    let model = write_tiny(&s);
    let out = serve(&s, &["--host", "0.0.0.0", "--model", arg(&model)]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(stderr(&out).contains("api_key_file"), "{}", stderr(&out));
    assert_never_started(&s);
}

/// Opening the log prunes old files, so a start the key checks refuse must not have opened it.
#[test]
fn a_refused_key_start_prunes_no_logs() {
    let s = scratch("serve-refused-prunes-nothing");
    std::fs::create_dir_all(&s.logs).unwrap();
    let old = s.logs.join("artemis-2000-01-01.jsonl");
    std::fs::write(&old, b"{}\n").unwrap();
    let model = write_tiny(&s);
    let out = serve(&s, &["--host", "0.0.0.0", "--model", arg(&model)]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(stderr(&out).contains("api_key_file"), "{}", stderr(&out));
    assert!(old.exists(), "a refused start pruned the log directory");
}

/// The hook is for the GPU stall test: `--help` does not list it.
#[test]
fn the_test_hook_is_hidden_from_help() {
    let out = Command::new(env!("CARGO_BIN_EXE_artemis"))
        .args(["serve", "--help"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    let help = String::from_utf8_lossy(&out.stdout);
    // The help printed, and lists the options that are not hidden.
    assert!(help.contains("--port"), "{help}");
    assert!(!help.contains("test-stall"), "{help}");
}

/// The refusal comes right after the config loads: before the model is opened (the missing file is never named), and
/// so before the log directory is made or any port is bound. No `--port 0` here, because the port is the point.
#[test]
fn the_test_hook_is_refused_on_8080() {
    let s = scratch("serve-hook-8080");
    let model = s.dir.join("no-such-model.gguf");
    let out = Command::new(env!("CARGO_BIN_EXE_artemis"))
        .env("ARTEMIS_TUNED", "off")
        .args(["serve", "--config"])
        .arg(&s.config)
        .arg("--log-dir")
        .arg(&s.logs)
        .args(["--stop-on-stdin-close", "--port", "8080"])
        .args(["--test-stall-at-token", "3", "--model"])
        .arg(&model)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(
        err.contains("--test-stall-at-token is refused on port 8080"),
        "{err}"
    );
    assert!(!err.contains("no-such-model"), "{err}");
    assert!(!s.logs.exists(), "the refusal came after the log opened");
    assert_never_started(&s);
}
