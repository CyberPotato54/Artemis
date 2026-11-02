//! `tools/test-messages.ps1` against the server's fake engine, scripted as the server's test `m1_test_messages_flow` is
//! (`crates/server/tests/anthropic.rs`). It needs PowerShell 7 (`pwsh`). Linux CI has no `pwsh`; the GitHub Windows job
//! runs it. It is `#[ignore]`d and runs where `pwsh` is installed:
//! `cargo test -p artemis-tools --test test_messages_pwsh -- --ignored`. The server runs on its own multi-thread
//! runtime, so `pwsh` runs from this plain test thread: a blocking `Command::output()` there can't starve the server it
//! is talking to.
mod support;

use server::app::Tuning;
use server::config::Config;
use server::fake::{FakeConfig, Script};
use std::path::{Path, PathBuf};
use std::process::Command;
use support::{fake, parts, scratch_dir, start, tokenizer};

/// An invented key-shaped literal, not a secret.
const KEY: &str = "test-messages-key-not-secret-4242"; // gitleaks:allow

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn call_text() -> String {
    "I should read the file.\n</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\npyproject.toml\n</parameter>\n</function>\n</tool_call>"
        .to_string()
}

/// `pwsh -NoProfile -File tools/test-messages.ps1 -BaseUrl <base> -KeyFile <key>`: the exit code and the output.
fn run_script(base_url: &str, key_file: &Path) -> (i32, String) {
    let out = Command::new("pwsh")
        .current_dir(repo_root())
        .args(["-NoProfile", "-File", "tools/test-messages.ps1", "-BaseUrl"])
        .arg(base_url)
        .arg("-KeyFile")
        .arg(key_file)
        .output()
        .expect("pwsh runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code().expect("pwsh exited"), text)
}

#[ignore = "needs pwsh"]
#[test]
fn test_messages_passes_against_the_fake() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let fake = fake(FakeConfig::default());
    let log_dir = scratch_dir("test-messages-logs");
    let cfg = Config {
        model: Some("m.gguf".into()),
        ctx: 65_536,
        ..Config::default()
    };
    let mut running = rt.block_on(start(
        parts(fake.clone(), cfg, Some(KEY), &log_dir),
        Tuning::default(),
    ));
    let key_file = running.key_file.clone().unwrap();

    // The model's replies: step 2 calls read_file; step 3 answers, its prompt served in part from the cache.
    fake.push(Script::text(tokenizer(), &call_text()));
    fake.push(Script {
        reused: 24,
        ..Script::text(
            tokenizer(),
            "Done.\n</think>\n\nThe project is named example.",
        )
    });
    let (code, text) = run_script(&running.base_url, &key_file);
    assert_eq!(code, 0, "{text}");
    assert!(!text.contains(KEY), "the output holds the key: {text}");
    assert!(text.contains("cache_read_input_tokens: 24"), "{text}");
    // Step 1's `{}` never reaches the engine; steps 2 and 3 do.
    assert_eq!(fake.requests().len(), 2, "{text}");

    // Step 2 answering in text, not with a call: the script fails there and never sends step 3.
    fake.push(Script::text(
        tokenizer(),
        "I will not read it.\n</think>\n\nNo.",
    ));
    let (code, text) = run_script(&running.base_url, &key_file);
    assert_eq!(code, 1, "{text}");
    assert!(!text.contains(KEY), "the output holds the key: {text}");
    assert!(text.contains("step 2"), "{text}");
    assert_eq!(fake.requests().len(), 3, "{text}");

    // The key file is read as the server reads it: the first non-empty line, trimmed. A copy with blank lines, padding
    // and a second line still authenticates.
    let padded = scratch_dir("test-messages-padded-key").join("key.txt");
    std::fs::write(&padded, format!("\r\n  {KEY}  \r\nnot the key\r\n")).unwrap();
    fake.push(Script::text(tokenizer(), &call_text()));
    fake.push(Script {
        reused: 24,
        ..Script::text(
            tokenizer(),
            "Done.\n</think>\n\nThe project is named example.",
        )
    });
    let (code, text) = run_script(&running.base_url, &padded);
    assert_eq!(code, 0, "{text}");
    assert!(!text.contains(KEY), "the output holds the key: {text}");

    rt.block_on(running.stop.stop()).unwrap();
}

/// The URL rules are the driver's (`check_base_url`): `http://` only, a loopback host, an explicit port that is not 8080,
/// and no path, user, query or fragment (one trailing `/` is allowed). Each refusal is before any request, so no server
/// runs; the key file is real, to show the URL is what is refused.
#[ignore = "needs pwsh"]
#[test]
fn test_messages_refuses_what_the_driver_refuses() {
    let dir = scratch_dir("test-messages-refusals");
    let key_file = dir.join("key.txt");
    std::fs::write(&key_file, format!("{KEY}\n")).unwrap();
    // An empty key file, or one of blank lines, is refused too.
    for empty in ["", "\n", " \r\n\t\n"] {
        let blank = dir.join("blank.txt");
        std::fs::write(&blank, empty).unwrap();
        let (code, text) = run_script("http://127.0.0.1:8090", &blank);
        assert_eq!(code, 1, "{text}");
        assert!(text.contains("empty"), "{text}");
    }
    for bad in [
        "https://127.0.0.1:8090",
        "http://127.0.0.1:8090/v1",
        "http://127.0.0.1:8090/v1/",
        "http://127.0.0.1:8090?x=1",
        "http://127.0.0.1:8090#f",
        "http://127.0.0.1@example.com:8090",
        "http://127.0.0.1",
        "http://127.0.0.1:",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:08080",
        "http://localhost:8080",
        "http://10.0.0.5:8090",
        "http://example.com:8090",
        "http://[::1]:8090",
        "http://127.0.0.1:99999",
        "127.0.0.1:8090",
    ] {
        let (code, text) = run_script(bad, &key_file);
        assert_eq!(code, 1, "{bad}: {text}");
        assert!(text.contains("refused"), "{bad}: {text}");
        assert!(
            !text.contains(KEY),
            "{bad}: the output holds the key: {text}"
        );
    }
}
