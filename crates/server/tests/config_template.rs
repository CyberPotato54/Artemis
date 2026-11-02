//! The production config template: `tools/config/artemis-production.toml.in` and `fill_template`.
use server::config::{fill_template, Config, ServeArgs, SpeculationConfig};
use std::path::PathBuf;

const TEMPLATE: &str = include_str!("../../../tools/config/artemis-production.toml.in");

fn no_env(_: &str) -> Option<String> {
    None
}

// Absolute paths on the platform the test runs on: the config resolves a relative path against its own folder, and
// `C:\...` is relative on Linux.
#[cfg(windows)]
const MODEL: &str = r"C:\models\m.gguf";
#[cfg(windows)]
const KEY: &str = r"C:\keys\key.txt";
#[cfg(windows)]
const LOGS: &str = r"C:\artemis\logs";
#[cfg(not(windows))]
const MODEL: &str = "/models/m.gguf";
#[cfg(not(windows))]
const KEY: &str = "/keys/key.txt";
#[cfg(not(windows))]
const LOGS: &str = "/artemis/logs";

fn values() -> Vec<(&'static str, &'static str)> {
    vec![
        ("MODEL", MODEL),
        ("PORT", "8080"),
        ("API_KEY_FILE", KEY),
        ("LOG_DIR", LOGS),
    ]
}

/// A scratch file under the temp dir, deleted when it drops.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str, content: &str) -> Scratch {
        let path = std::env::temp_dir().join(format!(
            "artemis-config-template-{}-{name}.toml",
            std::process::id()
        ));
        std::fs::write(&path, content).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn the_filled_template_is_the_production_config() {
    let text = fill_template(TEMPLATE, &values()).unwrap();
    let file = Scratch::new("filled", &text);
    let args = ServeArgs {
        config: Some(file.0.clone()),
        ..ServeArgs::default()
    };
    let c = Config::load(&args, &no_env, false).unwrap();
    assert_eq!(c.port, 8080);
    assert_eq!(c.host, "127.0.0.1");
    assert_eq!(c.health_service, "artemis");
    assert!(c.lazy_load);
    assert_eq!(c.idle_unload_s, 600);
    assert_eq!(c.queue_timeout_s, 300);
    assert_eq!(c.stall_s, 120);
    assert_eq!(c.stall_load_s, 300);
    assert_eq!(c.snapshot_pool_gib, 8);
    assert!(c.park_snapshots);
    // Speculation is on by default, so the template carries no line for it.
    assert_eq!(c.speculation, SpeculationConfig::default());
    assert!(c.speculation.enabled);
    assert_eq!(c.model, Some(PathBuf::from(MODEL)));
    assert_eq!(c.api_key_file, Some(PathBuf::from(KEY)));
    assert_eq!(c.log_dir, Some(PathBuf::from(LOGS)));
    // The template's eleven keys, written out so a change of default cannot move them; nothing else is set.
    let expected = Config {
        model: c.model.clone(),
        host: "127.0.0.1".into(),
        port: 8080,
        api_key_file: c.api_key_file.clone(),
        health_service: "artemis".into(),
        lazy_load: true,
        idle_unload_s: 600,
        queue_timeout_s: 300,
        log_dir: c.log_dir.clone(),
        stall_s: 120,
        stall_load_s: 300,
        ..Config::default()
    };
    // The keys the file set are recorded beside the values; they are not a setting. The template sets exactly these.
    assert_eq!(
        c.user_keys.iter().map(String::as_str).collect::<Vec<_>>(),
        [
            "api_key_file",
            "health_service",
            "host",
            "idle_unload_s",
            "lazy_load",
            "log_dir",
            "model",
            "port",
            "queue_timeout_s",
            "stall_load_s",
            "stall_s",
        ]
    );
    let c = Config {
        user_keys: Default::default(),
        ..c
    };
    assert_eq!(c, expected);
}

#[test]
fn the_template_names_no_machine_path() {
    for forbidden in [r":\", "/Users", "192.168.", "18080", "8085", "8080"] {
        assert!(
            !TEMPLATE.contains(forbidden),
            "the template names {forbidden:?}"
        );
    }
    let mut names = Vec::new();
    let mut rest = TEMPLATE;
    while let Some(at) = rest.find("{{") {
        let after = &rest[at + 2..];
        let end = after.find("}}").expect("an unclosed placeholder");
        names.push(&after[..end]);
        rest = &after[end + 2..];
    }
    assert_eq!(names, ["MODEL", "PORT", "API_KEY_FILE", "LOG_DIR"]);
}

#[test]
fn fill_refuses_a_quote_a_newline_a_leftover_and_an_unknown_name() {
    let with = |name: &str, value: &'static str| -> Vec<(&'static str, &'static str)> {
        values()
            .into_iter()
            .map(|(n, v)| if n == name { (n, value) } else { (n, v) })
            .collect()
    };
    assert!(fill_template(TEMPLATE, &values()).is_ok());
    // A value holding a quote, a newline or an opening brace pair.
    assert!(fill_template(TEMPLATE, &with("MODEL", "C:\\it's.gguf")).is_err());
    assert!(fill_template(TEMPLATE, &with("LOG_DIR", "C:\\a\nb")).is_err());
    assert!(fill_template(TEMPLATE, &with("LOG_DIR", "C:\\a\rb")).is_err());
    assert!(fill_template(TEMPLATE, &with("API_KEY_FILE", "C:\\{{PORT}}")).is_err());
    // A name the template does not hold, and a name the values leave out.
    let mut unknown = values();
    unknown.push(("EXTRA", "x"));
    assert!(fill_template(TEMPLATE, &unknown).is_err());
    let mut missing = values();
    missing.retain(|(n, _)| *n != "PORT");
    assert!(fill_template(TEMPLATE, &missing).is_err());
    // A placeholder the values do not name is a leftover.
    let extra = format!("{TEMPLATE}other = '{{{{OTHER}}}}'\n");
    assert!(fill_template(&extra, &values()).is_err());
}
