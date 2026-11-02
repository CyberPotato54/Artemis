//! The example config shipped in the release archive: `tools/config/artemis.toml.example`.
use server::config::{Config, ServeArgs};
use std::path::PathBuf;

const EXAMPLE: &str = include_str!("../../../tools/config/artemis.toml.example");

fn no_env(_: &str) -> Option<String> {
    None
}

/// A scratch directory under the temp dir, deleted when it drops.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!(
            "artemis-config-example-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn the_example_config_loads() {
    let s = Scratch::new("loads");
    let model = s.0.join("model.gguf");
    std::fs::write(&model, b"x").unwrap();

    // The user edits the `model` line; do the same with a real file's path in a TOML literal string.
    let mut replaced = 0;
    let text: String = EXAMPLE
        .lines()
        .map(|line| {
            if line.starts_with("model = ") {
                replaced += 1;
                format!("model = '{}'\n", model.display())
            } else {
                format!("{line}\n")
            }
        })
        .collect();
    assert_eq!(replaced, 1, "the example has exactly one model line");

    let file = s.0.join("artemis.toml");
    std::fs::write(&file, text).unwrap();
    let args = ServeArgs {
        config: Some(file),
        ..ServeArgs::default()
    };
    let c = Config::load(&args, &no_env, false).unwrap();
    assert_eq!(c.model, Some(model));
    assert_eq!(c.port, 8080);
    assert_eq!(c.host, "127.0.0.1");
    assert_eq!(c.api_key_file, None);
}

#[test]
fn the_example_names_no_machine_path() {
    // The drive-T path is built here so this file does not hold the text it forbids.
    let drive_t = format!("{}:\\", 'T');
    for forbidden in ["/Users", r"C:\Users", drive_t.as_str()] {
        assert!(
            !EXAMPLE.contains(forbidden),
            "the example names {forbidden:?}"
        );
    }
}

/// The sample values the example's comments show must be valid TOML: a commented-out `key = value` line, and the
/// double-quoted sample in the note about a single quote.
#[test]
fn the_examples_commented_samples_are_valid_toml() {
    let mut parsed = 0;
    for line in EXAMPLE.lines() {
        let Some(rest) = line.strip_prefix("# ") else {
            continue;
        };
        if rest.split_once(" = ").is_some_and(|(k, _)| {
            !k.is_empty() && k.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
        }) {
            let table: toml::Table = rest.parse().unwrap_or_else(|e| panic!("{rest}: {e}"));
            assert_eq!(table.len(), 1, "{rest}");
            parsed += 1;
        }
    }
    assert!(parsed >= 1, "the example has a commented key line");
    let sample = EXAMPLE
        .split_once("(\"")
        .and_then(|(_, rest)| rest.split_once("\")"))
        .map(|(sample, _)| sample)
        .expect("the example shows a double-quoted sample");
    let table: toml::Table = format!("model = \"{sample}\"")
        .parse()
        .unwrap_or_else(|e| panic!("{sample}: {e}"));
    assert_eq!(
        table["model"].as_str(),
        Some(r"C:\models\it's.gguf"),
        "the doubled backslashes read back as single ones"
    );
}
