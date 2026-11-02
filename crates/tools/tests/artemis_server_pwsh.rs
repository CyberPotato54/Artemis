//! `tools/ps/Test-ArtemisServer.ps1`: `Start-ArtemisServer -Config` past the config fill with a stub executable, so a
//! PowerShell name clash (the local `$config` that once overwrote the `-Config` parameter) fails here and not on the
//! test PC with the model loading. It needs Windows and PowerShell 7 (`pwsh`). Linux CI has no `pwsh`; the GitHub
//! Windows job runs it. It is `#[ignore]`d and runs where `pwsh` is installed:
//! `cargo test -p artemis-tools --test artemis_server_pwsh -- --ignored`. It starts no `artemis` and binds no port; it
//! reports a skip (and passes) when port 8090 is busy or an `artemis` process runs.
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[ignore = "needs pwsh on Windows"]
#[test]
fn start_artemis_server_fills_the_config_it_was_given() {
    let out = Command::new("pwsh")
        .current_dir(repo_root())
        .args(["-NoProfile", "-File", "tools/ps/Test-ArtemisServer.ps1"])
        .output()
        .expect("pwsh runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(
        text.contains("all checks passed") || text.contains("SKIP"),
        "{text}"
    );
}

/// `New-ArtemisKey` makes the key file owner-only (no inherited access rules, one rule for the current user). The
/// server that the harness starts runs as the same user, so it must still read the key: this reads the file with the
/// server's own `Config::read_key`.
#[ignore = "needs pwsh on Windows"]
#[test]
fn the_server_reads_a_key_file_new_artemis_key_made() {
    // Not canonicalize(): on Windows it returns a `\\?\C:\...` path, and PowerShell's execution-policy check fails on
    // that form ("AuthorizationManager check failed") under every policy but Bypass.
    let module = repo_root().join("tools/ps/ArtemisServer.psm1");
    assert!(module.is_file(), "the module exists: {}", module.display());
    let dir = std::env::temp_dir().join(format!("artemis-key-read-{}", std::process::id()));
    let script = format!(
        "Import-Module '{}' -Force; New-ArtemisKey -Dir '{}'",
        module.display(),
        dir.display()
    );
    let out = Command::new("pwsh")
        .args(["-NoProfile", "-Command", &script])
        .output()
        .expect("pwsh runs");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(out.status.code(), Some(0), "{stdout}{stderr}");
    let path = PathBuf::from(stdout.trim());
    let cfg = server::config::Config {
        api_key_file: Some(path.clone()),
        ..Default::default()
    };
    let key = cfg.read_key();
    let _ = std::fs::remove_dir_all(&dir);
    let key = key
        .expect("the server reads the key file")
        .expect("the config names a key file");
    assert_eq!(key.chars(), 43);
}
