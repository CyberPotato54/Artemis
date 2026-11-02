//! `tools/ps/Test-Hygiene.ps1`: the runner's affinity helpers (the system mask, set and read back, restored in a
//! `finally`) and the PCIe link check with stub interpreters. It needs Windows and PowerShell 7 (`pwsh`). Linux CI has
//! no `pwsh`; the GitHub Windows job runs it. It is `#[ignore]`d and runs where `pwsh` is installed:
//! `cargo test -p artemis-tools --test hygiene_pwsh -- --ignored`. It starts no `artemis` and uses no GPU.
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[ignore = "needs pwsh on Windows"]
#[test]
fn the_hygiene_module_sets_and_reads_back_affinity_and_judges_the_link() {
    let out = Command::new("pwsh")
        .current_dir(repo_root())
        .args(["-NoProfile", "-File", "tools/ps/Test-Hygiene.ps1"])
        .output()
        .expect("pwsh runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(text.contains("all checks passed"), "{text}");
}
