//! `artemis setup` and the first `artemis serve`, as far as a CPU test reaches: every refusal that comes before the
//! GPU, the report of an existing tuned file, the lock, `--help` and a bare `artemis`. Each test points
//! `ARTEMIS_TUNED` at a scratch path and the per-user folders at the scratch directory, so the real tuned file, config
//! and logs are never touched, and none loads a model: the contract check refuses the tiny one.
mod support;

use server::tuned::{render, SetupFacts, Tuned, TunedSpeculation, ARTEMIS_VERSION, TUNED_SCHEMA};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use support::{events, scratch_dir};

/// The words of a file that was valid when it was written.
const GPU: &str = "Scratch GPU 9000";

struct Scratch {
    dir: PathBuf,
    config: PathBuf,
    logs: PathBuf,
    tuned: PathBuf,
}

impl Scratch {
    fn lock(&self) -> PathBuf {
        PathBuf::from(format!("{}.lock", self.tuned.display()))
    }
}

fn scratch(tag: &str) -> Scratch {
    let dir = scratch_dir(tag);
    let config = dir.join("artemis.toml");
    let logs = dir.join("logs");
    // The setup command opens the log in the config's folder; a literal string keeps a Windows path's backslashes.
    std::fs::write(&config, format!("log_dir = '{}'\n", logs.display())).unwrap();
    Scratch {
        tuned: dir.join("state").join("tuned.toml"),
        logs,
        config,
        dir,
    }
}

fn artemis(s: &Scratch) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_artemis"));
    c.env("ARTEMIS_TUNED", &s.tuned)
        .env("LOCALAPPDATA", &s.dir)
        .env("APPDATA", &s.dir)
        .env("HOME", &s.dir)
        .env_remove("XDG_STATE_HOME")
        .env_remove("XDG_CONFIG_HOME")
        .stdin(Stdio::null());
    c
}

fn setup(s: &Scratch, extra: &[&str]) -> Output {
    artemis(s)
        .arg("setup")
        .arg("--config")
        .arg(&s.config)
        .args(extra)
        .output()
        .unwrap()
}

fn arg(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn write_tiny(s: &Scratch) -> PathBuf {
    let model = s.dir.join("tiny.gguf");
    std::fs::write(&model, qwen36::testing::tiny_model(1)).unwrap();
    model
}

/// A tuned file that parses.
fn valid_file() -> String {
    let t = Tuned {
        device_uuid: None,
        speculation: TunedSpeculation {
            enabled: Some(true),
            sources: Some(vec!["lookup".into(), "mtp".into()]),
            cost_slope_ms: None,
            mtp_cost_ms: None,
        },
        setup: SetupFacts {
            schema: TUNED_SCHEMA,
            artemis_version: ARTEMIS_VERSION.into(),
            finished: "2026-11-02T10:11:12Z".into(),
            seconds: 187.5,
            limit_hit: false,
            target: true,
            gpu_name: GPU.into(),
            gpu_uuid: "00000000-4800-0000-0000-000000000000".into(),
            driver: "2.0.355 (0x800163)".into(),
            vram_bytes: 17_095_000_000,
            budget_bytes: 16_106_127_360,
            cache_slots: 112,
            link_h2d_gbps: 25.5,
            host_free_bytes: 20_000_000_000,
            pool_gib: 8,
            model_bytes: 22_000_000_000,
            expected_tok_s: 137.0,
            measured_slope_ms: Some(4.5),
            measured_mtp_d1_ms: None,
            scores: BTreeMap::from([("lookup+mtp".to_string(), 137.0)]),
        },
    };
    render(&t, &BTreeMap::new())
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn no_setup_line(s: &Scratch) {
    if s.logs.exists() {
        let seen = events(&s.logs);
        assert!(
            !seen.iter().any(|e| e.starts_with("setup_")),
            "a setup line was written: {seen:?}"
        );
    }
}

#[test]
fn setup_refuses_a_model_the_contract_refuses_and_writes_nothing() {
    let s = scratch("setup-contract");
    let model = write_tiny(&s);
    let out = setup(&s, &["--model", arg(&model)]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(
        err.contains("tiny.gguf") && err.contains("metadata ") && err.contains("expected"),
        "{err}"
    );
    assert!(!s.tuned.exists(), "a refused setup wrote the tuned file");
    assert!(!s.lock().exists(), "a refused setup left its lock");
    let folder = s.tuned.parent().unwrap();
    assert!(
        !folder.exists() || std::fs::read_dir(folder).unwrap().next().is_none(),
        "a refused setup left a file beside the tuned file"
    );
    no_setup_line(&s);
}

#[test]
fn an_empty_model_file_is_refused_the_same_way() {
    let s = scratch("setup-empty-model");
    let model = s.dir.join("empty.gguf");
    std::fs::write(&model, b"").unwrap();
    let out = setup(&s, &["--model", arg(&model)]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(stderr(&out).contains("empty.gguf"), "{}", stderr(&out));
    assert!(!s.tuned.exists() && !s.lock().exists());
}

#[test]
fn serve_refuses_before_setup_runs() {
    let s = scratch("serve-before-setup");
    let model = write_tiny(&s);
    let out = artemis(&s)
        .args(["serve", "--port", "0", "--config"])
        .arg(&s.config)
        .arg("--log-dir")
        .arg(&s.logs)
        .arg("--stop-on-stdin-close")
        .args(["--model", arg(&model)])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(stderr(&out).contains("metadata "), "{}", stderr(&out));
    assert!(!s.tuned.exists(), "the refused start wrote a tuned file");
    assert!(!s.lock().exists(), "the refused start left a lock");
    no_setup_line(&s);
    if s.logs.exists() {
        assert!(!events(&s.logs).iter().any(|e| e == "start"));
    }
}

#[test]
fn setup_with_tuned_off_is_refused() {
    let s = scratch("setup-off");
    let model = write_tiny(&s);
    for off in ["off", "OFF"] {
        let out = artemis(&s)
            .env("ARTEMIS_TUNED", off)
            .arg("setup")
            .arg("--config")
            .arg(&s.config)
            .args(["--model", arg(&model)])
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
        assert!(
            stderr(&out).contains(
                "ARTEMIS_TUNED is off, so setup has nowhere to write. Unset it, or set it to a file path."
            ),
            "{}",
            stderr(&out)
        );
    }
    assert!(!s.tuned.exists());
}

#[test]
fn setup_with_no_place_for_the_file_is_refused() {
    let s = scratch("setup-unplaced");
    let model = write_tiny(&s);
    let out = artemis(&s)
        .env_remove("ARTEMIS_TUNED")
        .env_remove("LOCALAPPDATA")
        .env_remove("HOME")
        .arg("setup")
        .arg("--config")
        .arg(&s.config)
        .args(["--model", arg(&model)])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("is not set, so the tuned settings are not read or written"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn setup_without_force_reports_an_existing_file() {
    let s = scratch("setup-report");
    std::fs::create_dir_all(s.tuned.parent().unwrap()).unwrap();
    let text = valid_file();
    std::fs::write(&s.tuned, &text).unwrap();
    // The model is never opened: a path that does not exist is not an error here.
    let missing = s.dir.join("no-such-model.gguf");
    let out = setup(&s, &["--model", arg(&missing)]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    let said = stdout(&out);
    assert!(said.contains(GPU), "{said}");
    assert!(said.contains("artemis setup --force"), "{said}");
    assert!(said.contains(&s.tuned.display().to_string()), "{said}");
    assert!(said.contains("137"), "{said}");
    assert_eq!(std::fs::read_to_string(&s.tuned).unwrap(), text);
    assert!(!s.lock().exists());
    no_setup_line(&s);
}

#[test]
fn a_failed_setup_leaves_an_existing_file_alone() {
    let s = scratch("setup-force-fails");
    std::fs::create_dir_all(s.tuned.parent().unwrap()).unwrap();
    let text = valid_file();
    std::fs::write(&s.tuned, &text).unwrap();
    let model = write_tiny(&s);
    let out = setup(&s, &["--force", "--model", arg(&model)]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(stderr(&out).contains("metadata "), "{}", stderr(&out));
    assert_eq!(std::fs::read_to_string(&s.tuned).unwrap(), text);
    assert!(!s.lock().exists());
}

#[test]
fn a_corrupt_file_does_not_stop_a_forced_setup_or_a_plain_one() {
    // `serve` refuses a corrupt file and names `artemis setup --force`; setup itself never reads it with --force, and
    // without --force it finds the file unreadable and measures again (here the model is refused first).
    let s = scratch("setup-corrupt");
    std::fs::create_dir_all(s.tuned.parent().unwrap()).unwrap();
    std::fs::write(&s.tuned, "speculation = [").unwrap();
    let model = write_tiny(&s);
    for extra in [&["--force"][..], &[][..]] {
        let mut args = extra.to_vec();
        args.extend(["--model", arg(&model)]);
        let out = setup(&s, &args);
        assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
        assert!(stderr(&out).contains("metadata "), "{}", stderr(&out));
        assert_eq!(
            std::fs::read_to_string(&s.tuned).unwrap(),
            "speculation = ["
        );
    }
}

#[test]
fn a_live_lock_refuses_and_a_stale_one_is_replaced() {
    let s = scratch("setup-lock");
    let model = write_tiny(&s);
    std::fs::create_dir_all(s.tuned.parent().unwrap()).unwrap();

    // A lock written a moment ago by a process that is running belongs to a setup that is running (this test's own
    // process stands for it).
    let me = std::process::id();
    let live = format!("pid={me}\ntime={}\n", unix_now());
    std::fs::write(s.lock(), &live).unwrap();
    let out = setup(&s, &["--model", arg(&model)]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(
        stderr(&out).contains(&format!(
            "Another artemis setup is running (pid {me}). If it is not, delete "
        )),
        "{}",
        stderr(&out)
    );
    assert_eq!(std::fs::read_to_string(s.lock()).unwrap(), live);
    // The refusal came before the model was looked at.
    assert!(!stderr(&out).contains("metadata "), "{}", stderr(&out));

    // A lock older than twice the time limit is stale: setup replaces it, goes on to the model (which it refuses), and
    // leaves no lock behind.
    let old = format!("pid=4242\ntime={}\n", unix_now() - 2 * 600 - 60);
    std::fs::write(s.lock(), old).unwrap();
    let out = setup(&s, &["--model", arg(&model)]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(stderr(&out).contains("metadata "), "{}", stderr(&out));
    assert!(!s.lock().exists(), "the stale lock was not released");
}

#[test]
fn help_lists_setup() {
    let s = scratch("setup-help");
    let out = artemis(&s).arg("--help").output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    let help = stdout(&out);
    assert!(help.contains("setup"), "{help}");
    assert!(help.contains("serve"), "{help}");
    let out = artemis(&s).args(["setup", "--help"]).output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    let help = stdout(&out);
    for flag in ["--config", "--model", "--device-uuid", "--force", "--json"] {
        assert!(help.contains(flag), "{flag} in {help}");
    }
}

#[test]
fn bare_artemis_names_serve_and_setup() {
    let s = scratch("setup-bare");
    let out = artemis(&s).output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    assert_eq!(
        stdout(&out).trim_end(),
        "Start the server with: artemis serve --model <path to the GGUF>. Its first start measures this PC; \
         artemis setup measures again. artemis --help lists every command."
    );
}

#[test]
fn an_unwritable_json_path_is_refused_before_anything_is_measured() {
    let s = scratch("setup-json");
    let model = write_tiny(&s);
    let blocker = s.dir.join("blocker");
    std::fs::write(&blocker, "a file where a folder should be").unwrap();
    let report = blocker.join("report.json");
    let out = setup(&s, &["--model", arg(&model), "--json", arg(&report)]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    let err = stderr(&out);
    // The report path is found before the lock and the model are looked at.
    assert!(err.contains("cannot write the --json report"), "{err}");
    assert!(!err.contains("metadata "), "{err}");
    assert!(!s.tuned.exists() && !s.lock().exists());
    no_setup_line(&s);
}
