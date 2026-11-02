//! `tools/scan-secrets.ps1` and `tools/git-hooks/pre-push` (plan m6q, Task 21): a local run of the pinned gitleaks the
//! forge's secret-scan job runs. The tests need PowerShell 7 (`pwsh`) and `git`, and the first run downloads gitleaks
//! from the forge release into `%LOCALAPPDATA%\artemis-tools` (git's own credential for the forge; later runs use the
//! cached, hash-checked copy). CI has no `pwsh`, so they are `#[ignore]`d:
//! `cargo test -p artemis-tools --test scan_secrets_pwsh -- --ignored --test-threads=1`.
//! The fake key is assembled at run time from pieces, so no literal secret sits in this file for the forge's own scan.
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("scan-secrets")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// An AWS-style access key id (the shape gitleaks' aws-access-token rule reads), built from pieces.
fn fake_key() -> String {
    ["AKIA", "QYLPMN5H", "HKQPZ4TR"].concat()
}

/// A config line that holds the key, with `suffix` after it.
fn key_line(suffix: &str) -> String {
    format!("aws_access_key_id = {}{suffix}\n", fake_key())
}

fn git_in(dir: &Path, args: &[&str]) {
    let st = Command::new("git")
        .current_dir(dir)
        .args([
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.autocrlf=false",
        ])
        .args(args)
        .status()
        .expect("git runs");
    assert!(st.success(), "git {args:?} in {dir:?}");
}

/// A repository with a first commit of a readme, then (when `leaky` is some) a second commit of `config.txt`.
fn scratch_repo(name: &str, leaky: Option<&str>) -> PathBuf {
    let dir = scratch(name).join("repo");
    std::fs::create_dir_all(&dir).unwrap();
    git_in(&dir, &["init", "-q"]);
    std::fs::write(dir.join("README.txt"), "nothing to see\n").unwrap();
    git_in(&dir, &["add", "-A"]);
    git_in(&dir, &["commit", "-q", "-m", "first"]);
    if let Some(text) = leaky {
        std::fs::write(dir.join("config.txt"), text).unwrap();
        git_in(&dir, &["add", "-A"]);
        git_in(&dir, &["commit", "-q", "-m", "config"]);
    }
    dir
}

/// Runs the script from this repository with the current folder `cwd`: the exit code and the output.
fn scan(cwd: &Path, tools_dir: Option<&Path>, args: &[&OsStr]) -> (i32, String) {
    scan_env(cwd, tools_dir, &[], args)
}

/// `scan` with extra environment variables for the script.
fn scan_env(
    cwd: &Path,
    tools_dir: Option<&Path>,
    env: &[(&str, &OsStr)],
    args: &[&OsStr],
) -> (i32, String) {
    let mut cmd = Command::new("pwsh");
    cmd.envs(env.iter().copied());
    cmd.current_dir(cwd)
        .arg("-NoProfile")
        .arg("-File")
        .arg(repo_root().join("tools/scan-secrets.ps1"))
        .args(args);
    if let Some(dir) = tools_dir {
        cmd.env("ARTEMIS_TOOLS_DIR", dir);
    }
    let out = cmd.output().expect("pwsh runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code().expect("pwsh exited"), text)
}

/// The key must not appear in a report, whole or in its middle piece.
fn assert_redacted(text: &str) {
    assert!(
        !text.contains(&fake_key()),
        "the key is in the output: {text}"
    );
    assert!(
        !text.contains("QYLPMN5H"),
        "part of the key is in the output: {text}"
    );
}

#[ignore = "needs pwsh and the pinned gitleaks"]
#[test]
fn a_committed_key_is_found_and_redacted() {
    let repo = scratch_repo("range-leak", Some(&key_line("")));
    let (code, text) = scan(
        &repo,
        None,
        &[
            OsStr::new("-Mode"),
            OsStr::new("range"),
            OsStr::new("-Range"),
            OsStr::new("HEAD~1..HEAD"),
        ],
    );
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("config.txt"), "the file is named: {text}");
    assert!(text.contains("REDACTED"), "{text}");
    assert_redacted(&text);
}

#[ignore = "needs pwsh and the pinned gitleaks"]
#[test]
fn a_gitleaks_allow_comment_passes() {
    let repo = scratch_repo("range-allow", Some(&key_line(" # gitleaks:allow")));
    let (code, text) = scan(
        &repo,
        None,
        &[
            OsStr::new("-Mode"),
            OsStr::new("range"),
            OsStr::new("-Range"),
            OsStr::new("HEAD~1..HEAD"),
        ],
    );
    assert_eq!(code, 0, "{text}");
    assert_redacted(&text);
}

#[ignore = "needs pwsh and the pinned gitleaks"]
#[test]
fn the_unpushed_mode_finds_a_key_no_remote_holds() {
    let repo = scratch_repo("unpushed", Some(&key_line("")));
    let (code, text) = scan(&repo, None, &[OsStr::new("-Mode"), OsStr::new("unpushed")]);
    assert_eq!(code, 1, "{text}");
    assert_redacted(&text);
}

#[ignore = "needs pwsh and the pinned gitleaks"]
#[test]
fn the_tree_mode_scans_untracked_files_and_skips_ignored_ones() {
    let repo = scratch_repo("tree", None);
    std::fs::write(repo.join(".gitignore"), "ignored/\n").unwrap();
    std::fs::create_dir_all(repo.join("ignored")).unwrap();
    std::fs::write(repo.join("ignored/k.txt"), key_line("")).unwrap();
    let (code, text) = scan(&repo, None, &[]);
    assert_eq!(code, 0, "an ignored file is not scanned: {text}");
    std::fs::write(repo.join("new.txt"), key_line("")).unwrap();
    let (code, text) = scan(&repo, None, &[]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("new.txt"), "{text}");
    assert_redacted(&text);
}

#[ignore = "needs pwsh and the pinned gitleaks"]
#[test]
fn the_dir_mode_passes_a_clean_folder_and_finds_a_key() {
    let dir = scratch("dir-clean");
    std::fs::write(dir.join("a.txt"), "plain text\n").unwrap();
    let (code, text) = scan(
        &dir,
        None,
        &[
            OsStr::new("-Mode"),
            OsStr::new("dir"),
            OsStr::new("-Path"),
            dir.as_os_str(),
        ],
    );
    assert_eq!(code, 0, "{text}");

    std::fs::write(dir.join("b.txt"), key_line("")).unwrap();
    let (code, text) = scan(&dir, None, &dir_args(&dir));
    assert_eq!(code, 1, "{text}");
    assert_redacted(&text);
}

fn dir_args(dir: &Path) -> Vec<&OsStr> {
    vec![
        OsStr::new("-Mode"),
        OsStr::new("dir"),
        OsStr::new("-Path"),
        dir.as_os_str(),
    ]
}

/// The fingerprint gitleaks prints for the key in a file `b.txt`.
const IGNORE_B: &str = "b.txt:aws-access-token:1\n";

#[ignore = "needs pwsh and the pinned gitleaks"]
#[test]
fn the_dir_mode_reads_only_the_scanned_folders_ignore_file() {
    // The fingerprint suppresses the finding when the scanned folder holds the ignore file (this proves the
    // fingerprint is right) and does nothing when only the current folder holds it.
    let own = scratch("dir-ignore-own");
    std::fs::write(own.join("b.txt"), key_line("")).unwrap();
    std::fs::write(own.join(".gitleaksignore"), IGNORE_B).unwrap();
    let (code, text) = scan(&own, None, &dir_args(&own));
    assert_eq!(code, 0, "{text}");

    let target = scratch("dir-ignore-target");
    std::fs::write(target.join("b.txt"), key_line("")).unwrap();
    let cwd = scratch("dir-ignore-cwd");
    std::fs::write(cwd.join(".gitleaksignore"), IGNORE_B).unwrap();
    let (code, text) = scan(&cwd, None, &dir_args(&target));
    assert_eq!(code, 1, "{text}");
}

/// A rule file with one dummy rule and no default rules: if gitleaks read it, the key would go unseen.
const PLANTED: &str =
    "title = \"planted\"\n[[rules]]\nid = \"dummy\"\nregex = \'\'\'zzzzzzzz\'\'\'\n";

#[ignore = "needs pwsh and the pinned gitleaks"]
#[test]
fn a_planted_rule_file_or_variable_does_not_replace_the_rules() {
    let dir = scratch("planted-dir");
    std::fs::write(dir.join("b.txt"), key_line("")).unwrap();
    std::fs::write(dir.join(".gitleaks.toml"), PLANTED).unwrap();
    let (code, text) = scan(&dir, None, &dir_args(&dir));
    assert_eq!(code, 1, "a .gitleaks.toml in the folder: {text}");

    let other = scratch("planted-env");
    std::fs::write(other.join("b.txt"), key_line("")).unwrap();
    let toml = other.join("one-rule.toml");
    std::fs::write(&toml, PLANTED).unwrap();
    let (code, text) = scan_env(
        &other,
        None,
        &[
            ("GITLEAKS_CONFIG", toml.as_os_str()),
            ("GITLEAKS_CONFIG_TOML", OsStr::new(PLANTED)),
        ],
        &dir_args(&other),
    );
    assert_eq!(code, 1, "GITLEAKS_CONFIG: {text}");

    // In git mode the repository's working tree holds the planted file.
    let repo = scratch_repo("planted-repo", Some(&key_line("")));
    std::fs::write(repo.join(".gitleaks.toml"), PLANTED).unwrap();
    let (code, text) = scan(&repo, None, &range_args("HEAD~1..HEAD"));
    assert_eq!(code, 1, "a .gitleaks.toml in the repository: {text}");
}

fn range_args(range: &str) -> Vec<&OsStr> {
    vec![
        OsStr::new("-Mode"),
        OsStr::new("range"),
        OsStr::new("-Range"),
        OsStr::new(range),
    ]
}

#[ignore = "needs pwsh"]
#[test]
fn a_range_git_cannot_walk_is_an_error_and_never_clean() {
    let repo = scratch_repo("bad-range", None);
    for range in ["nonexistentref..alsonot", "--output=x"] {
        let (code, text) = scan(&repo, None, &range_args(range));
        assert_eq!(code, 3, "{range}: {text}");
        assert!(!text.contains("no leaks found"), "{range}: {text}");
        assert!(text.contains("scan-secrets: "), "{range}: {text}");
    }
}

#[ignore = "needs pwsh and the pinned gitleaks"]
#[test]
fn a_replaced_exe_is_restored_from_the_verified_zip() {
    // Warm the default cache, then copy its zip into a scratch cache with a bad exe next to it.
    let dir = scratch("exe-source");
    std::fs::write(dir.join("a.txt"), "plain text\n").unwrap();
    let (code, text) = scan(&dir, None, &dir_args(&dir));
    assert_eq!(code, 0, "{text}");
    let cached = Path::new(&std::env::var_os("LOCALAPPDATA").expect("LOCALAPPDATA"))
        .join("artemis-tools/gitleaks-8.30.1");
    let tools = scratch("tampered-exe");
    let cache = tools.join("gitleaks-8.30.1");
    std::fs::create_dir_all(&cache).unwrap();
    let zip = "gitleaks_8.30.1_windows_x64.zip";
    std::fs::copy(cached.join(zip), cache.join(zip)).unwrap();
    let good_len = std::fs::metadata(cached.join("gitleaks.exe"))
        .unwrap()
        .len();
    std::fs::write(cache.join("gitleaks.exe"), b"not a program").unwrap();
    let (code, text) = scan(&dir, Some(&tools), &dir_args(&dir));
    assert_eq!(code, 0, "{text}");
    assert_eq!(
        std::fs::metadata(cache.join("gitleaks.exe")).unwrap().len(),
        good_len,
        "the exe was replaced by the zip's"
    );
}

#[ignore = "needs pwsh"]
#[test]
fn without_a_download_url_the_scan_stops_and_names_the_config_key() {
    let tools = scratch("no-url");
    let repo = scratch_repo("no-url-repo", None);
    // An empty value in the environment's git config overrides the clone's artemis.gitleaksUrl.
    let (code, text) = scan_env(
        &repo,
        Some(&tools),
        &[
            ("GIT_CONFIG_COUNT", OsStr::new("1")),
            ("GIT_CONFIG_KEY_0", OsStr::new("artemis.gitleaksUrl")),
            ("GIT_CONFIG_VALUE_0", OsStr::new("")),
        ],
        &[],
    );
    assert_eq!(code, 3, "{text}");
    assert!(text.contains("artemis.gitleaksUrl"), "{text}");
}

#[ignore = "needs pwsh"]
#[test]
fn a_corrupted_cached_zip_is_refused_and_removed() {
    let tools = scratch("corrupt-cache");
    let cache = tools.join("gitleaks-8.30.1");
    std::fs::create_dir_all(&cache).unwrap();
    let zip = cache.join("gitleaks_8.30.1_windows_x64.zip");
    std::fs::write(&zip, b"this is not the pinned archive").unwrap();
    let repo = scratch_repo("corrupt-repo", None);
    let (code, text) = scan(&repo, Some(&tools), &[]);
    assert_eq!(code, 3, "{text}");
    assert!(
        text.contains("gitleaks download does not match its pinned SHA-256"),
        "{text}"
    );
    assert!(!zip.exists(), "the bad zip is deleted");
}

/// Runs `git push` in `repo`: whether it succeeded and the output.
fn push(repo: &Path, args: &[&str]) -> (bool, String) {
    push_env(repo, &[], args)
}

fn push_env(repo: &Path, env: &[(&str, &OsStr)], args: &[&str]) -> (bool, String) {
    let out = Command::new("git")
        .envs(env.iter().copied())
        .current_dir(repo)
        .args(["-c", "core.autocrlf=false", "push"])
        .args(args)
        .output()
        .expect("git runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

#[ignore = "needs pwsh, sh and the pinned gitleaks"]
#[test]
fn the_pre_push_hook_stops_a_key_and_lets_a_clean_push_through() {
    let work = scratch("hook");
    let remote = work.join("remote.git");
    let st = Command::new("git")
        .args(["init", "-q", "--bare"])
        .arg(&remote)
        .status()
        .unwrap();
    assert!(st.success());
    let repo = scratch_repo("hook-repo", None);
    git_in(&repo, &["branch", "-M", "main"]);
    git_in(
        &repo,
        &["remote", "add", "origin", remote.to_str().unwrap()],
    );
    let hooks = repo_root().join("tools/git-hooks");
    git_in(
        &repo,
        &["config", "core.hooksPath", hooks.to_str().unwrap()],
    );

    // A clean first push (a new branch) goes through.
    let (ok, text) = push(&repo, &["origin", "main"]);
    assert!(ok, "{text}");

    // A commit with a key is refused, with the hook's message and no secret in the output.
    std::fs::write(repo.join("config.txt"), key_line("")).unwrap();
    git_in(&repo, &["add", "-A"]);
    git_in(&repo, &["commit", "-q", "-m", "config"]);
    let (ok, text) = push(&repo, &["origin", "main"]);
    assert!(!ok, "{text}");
    assert!(
        text.contains("gitleaks found a possible secret; the push is stopped."),
        "{text}"
    );
    assert_redacted(&text);

    // A new branch holding the same commit is refused too.
    let (ok, text) = push(&repo, &["origin", "main:other"]);
    assert!(!ok, "{text}");

    // Drop the commit, commit a clean file, and the push goes through.
    git_in(&repo, &["reset", "-q", "--hard", "HEAD~1"]);
    std::fs::write(repo.join("config.txt"), "aws_access_key_id = removed\n").unwrap();
    git_in(&repo, &["add", "-A"]);
    git_in(&repo, &["commit", "-q", "-m", "config without a key"]);
    let (ok, text) = push(&repo, &["origin", "main"]);
    assert!(ok, "{text}");

    // A deletion sends no commits and is not scanned.
    let (ok, text) = push(&repo, &["origin", "main:tmp"]);
    assert!(ok, "{text}");
    let (ok, text) = push(&repo, &["origin", "--delete", "tmp"]);
    assert!(ok, "{text}");
}

#[ignore = "needs pwsh and sh"]
#[test]
fn the_pre_push_hook_says_so_when_the_scan_could_not_run() {
    let work = scratch("hook-error");
    let remote = work.join("remote.git");
    let st = Command::new("git")
        .args(["init", "-q", "--bare"])
        .arg(&remote)
        .status()
        .unwrap();
    assert!(st.success());
    let repo = scratch_repo("hook-error-repo", None);
    git_in(&repo, &["branch", "-M", "main"]);
    git_in(
        &repo,
        &["remote", "add", "origin", remote.to_str().unwrap()],
    );
    let hooks = repo_root().join("tools/git-hooks");
    git_in(
        &repo,
        &["config", "core.hooksPath", hooks.to_str().unwrap()],
    );

    // A corrupted cached zip: the scan cannot run, and the hook does not claim a secret was found.
    let tools = scratch("hook-error-cache");
    let cache = tools.join("gitleaks-8.30.1");
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::write(
        cache.join("gitleaks_8.30.1_windows_x64.zip"),
        b"this is not the pinned archive",
    )
    .unwrap();
    let (ok, text) = push_env(
        &repo,
        &[("ARTEMIS_TOOLS_DIR", tools.as_os_str())],
        &["origin", "main"],
    );
    assert!(!ok, "{text}");
    assert!(text.contains("the secret scan could not run: "), "{text}");
    assert!(text.contains("does not match its pinned SHA-256"), "{text}");
    assert!(text.contains("git push --no-verify"), "{text}");
    assert!(!text.contains("found a possible secret"), "{text}");
}

/// Copies `tools/install-hooks.ps1` and `tools/scan-secrets.ps1` into a scratch repository whose origin is `origin`,
/// runs the installer there (an empty tools folder, so no zip is cached) and returns the output and the repository's
/// local `core.hooksPath` and `artemis.gitleaksUrl` ("" when unset). Only the scratch repository's config changes.
fn install_hooks_in_scratch(name: &str, origin: &str) -> (String, String, String) {
    let repo = scratch(name).join("repo");
    std::fs::create_dir_all(repo.join("tools")).unwrap();
    for f in ["install-hooks.ps1", "scan-secrets.ps1"] {
        std::fs::copy(
            repo_root().join("tools").join(f),
            repo.join("tools").join(f),
        )
        .unwrap();
    }
    git_in(&repo, &["init", "-q"]);
    git_in(&repo, &["remote", "add", "origin", origin]);
    let empty_tools = scratch(&format!("{name}-tools"));
    let out = Command::new("pwsh")
        .current_dir(&repo)
        .env("ARTEMIS_TOOLS_DIR", &empty_tools)
        .args(["-NoProfile", "-File", "tools/install-hooks.ps1"])
        .output()
        .expect("pwsh runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(0), "{text}");
    let get = |key: &str| {
        let o = Command::new("git")
            .current_dir(&repo)
            .args(["config", "--local", "--get", key])
            .output()
            .unwrap();
        String::from_utf8_lossy(&o.stdout).trim().to_string()
    };
    (text, get("core.hooksPath"), get("artemis.gitleaksUrl"))
}

#[ignore = "needs pwsh"]
#[test]
fn the_installer_leaves_the_hook_off_in_a_clone_of_the_github_repository() {
    let (text, hooks, url) =
        install_hooks_in_scratch("install-github", "https://github.com/example/artemis.git");
    assert_eq!(hooks, "", "{text}");
    assert_eq!(url, "", "{text}");
    assert!(text.contains("the pre-push hook stays off"), "{text}");
    assert!(text.contains("ARTEMIS_TOOLS_DIR"), "{text}");
    assert!(text.contains("secrets job"), "{text}");
}

#[ignore = "needs pwsh"]
#[test]
fn the_installer_turns_the_hook_on_when_origin_gives_a_forge_release_url() {
    let (text, hooks, url) = install_hooks_in_scratch(
        "install-forge",
        "https://forge.example.invalid/owner/artemis.git",
    );
    assert_eq!(hooks, "tools/git-hooks", "{text}");
    assert_eq!(
        url,
        "https://forge.example.invalid/owner/artemis/releases/download/ci-tools%2Fgitleaks-v8.30.1",
        "{text}"
    );
}
