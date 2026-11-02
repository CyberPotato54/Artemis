//! `tools/check-public-snapshot.ps1`: it exports the publish set from `HEAD` and scans every file
//! for the markers in a file. It needs PowerShell 7 (`pwsh`) and `git`. Linux CI has no `pwsh`; the GitHub Windows job
//! runs them. They are `#[ignore]`d and run where `pwsh` is installed:
//! `cargo test -p artemis-tools --test public_snapshot_pwsh -- --ignored`.
//! One test runs the script on this repository's `HEAD`. The others build a scratch repository and point `-Root` at it,
//! so the exclusions, the `allow` syntax, UTF-16LE and the user-profile pattern are pinned on known files.
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("public-snapshot")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Runs the script from this repository with extra arguments: the exit code and the output.
fn run(args: &[&OsStr]) -> (i32, String) {
    let out = Command::new("pwsh")
        .current_dir(repo_root())
        .args(["-NoProfile", "-File", "tools/check-public-snapshot.ps1"])
        .args(args)
        .output()
        .expect("pwsh runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code().expect("pwsh exited"), text)
}

/// Makes a repository under the scratch folder holding `files` (path, bytes), committed, and returns its folder.
fn scratch_repo(name: &str, files: &[(&str, Vec<u8>)]) -> PathBuf {
    let dir = scratch(name).join("repo");
    std::fs::create_dir_all(&dir).unwrap();
    let git = |args: &[&str]| {
        let st = Command::new("git")
            .current_dir(&dir)
            .args([
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.invalid",
            ])
            .args(["-c", "commit.gpgsign=false", "-c", "core.autocrlf=false"])
            .args(args)
            .status()
            .expect("git runs");
        assert!(st.success(), "git {args:?}");
    };
    git(&["init", "-q"]);
    for (path, bytes) in files {
        let full = dir.join(path);
        std::fs::create_dir_all(full.parent().unwrap()).unwrap();
        std::fs::write(full, bytes).unwrap();
    }
    git(&["add", "-A"]);
    git(&["commit", "-q", "-m", "files"]);
    dir
}

/// Runs the script on a scratch repository with the markers file text `lines`: the exit code and the sorted output lines.
fn run_on(repo: &Path, lines: &str) -> (i32, Vec<String>) {
    let markers = repo.parent().unwrap().join("markers.txt");
    std::fs::write(&markers, lines).unwrap();
    let (code, text) = run(&[
        OsStr::new("-Root"),
        repo.as_os_str(),
        OsStr::new("-Markers"),
        markers.as_os_str(),
    ]);
    let mut out: Vec<String> = text.lines().map(str::to_string).collect();
    out.sort();
    (code, out)
}

fn utf16le(s: &str) -> Vec<u8> {
    s.encode_utf16().flat_map(u16::to_le_bytes).collect()
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[ignore = "needs pwsh"]
#[test]
fn a_marker_in_the_tree_fails_and_names_a_file() {
    let dir = scratch("present");
    let markers = dir.join("markers.txt");
    std::fs::write(&markers, "# a test's markers\nartemis\n").unwrap();
    let (code, text) = run(&[OsStr::new("-Markers"), markers.as_os_str()]);
    assert_eq!(code, 1, "{text}");
    let first = text.lines().next().unwrap_or("");
    // "<file>:<line>: <marker>", and the file is in this repository.
    let (file, rest) = first.split_once(':').expect("a file name then a colon");
    assert!(repo_root().join(file).is_file(), "no file {file:?}: {text}");
    let (line, marker) = rest
        .split_once(": ")
        .expect("a line number then the marker");
    assert!(line.parse::<usize>().is_ok(), "{first}");
    assert_eq!(marker, "artemis", "{first}");
    assert!(!text.contains("clean:"), "{text}");
}

#[ignore = "needs pwsh"]
#[test]
fn a_marker_not_in_the_tree_passes() {
    // Built at run time so this file does not hold the marker itself.
    let marker = ["zzqx", "not", "in", "the", "tree"].join("-");
    let repo = scratch_repo(
        "absent",
        &[
            ("a.txt", b"some text\n".to_vec()),
            ("b/c.txt", b"more text\n".to_vec()),
        ],
    );
    let (code, out) = run_on(&repo, &format!("{marker}\n"));
    assert_eq!(code, 0, "{out:?}");
    assert_eq!(out, strings(&["clean: 2 files"]));
}

#[ignore = "needs pwsh"]
#[test]
fn excluded_paths_are_skipped_and_the_rest_is_scanned() {
    let repo = scratch_repo(
        "exclusions",
        &[
            ("kept.txt", b"zzqxsecret in a kept file\n".to_vec()),
            ("docs/superpowers/a.txt", b"zzqxsecret\n".to_vec()),
            ("docs/github/notes.md", b"zzqxsecret\n".to_vec()),
            ("docs/github/about.md", b"zzqxsecret\n".to_vec()),
            (".gitea/workflows/ci.yml", b"zzqxsecret\n".to_vec()),
            ("CLAUDE.md", b"zzqxsecret\n".to_vec()),
            (".gitleaksignore", b"zzqxsecret\n".to_vec()),
            ("data.bin", utf16le("zzqxsecret")),
        ],
    );
    let (code, out) = run_on(&repo, "zzqxsecret\n");
    assert_eq!(code, 1, "{out:?}");
    assert_eq!(
        out,
        strings(&[
            "data.bin:1: zzqxsecret",
            "docs/github/about.md:1: zzqxsecret",
            "kept.txt:1: zzqxsecret",
        ])
    );
}

#[ignore = "needs pwsh"]
#[test]
fn an_allow_line_suppresses_a_marker_only_in_matching_files() {
    let repo = scratch_repo(
        "allow",
        &[
            ("a.txt", b"zzqxone\nzzqxtwo\n".to_vec()),
            ("sub/a.txt", b"zzqxone\n".to_vec()),
            ("b.txt", b"zzqxone and zzqxtwo\n".to_vec()),
        ],
    );
    let (code, out) = run_on(
        &repo,
        "zzqxone\nzzqxtwo\nallow a.txt zzqxone\nallow sub/* zzqxtwo\n",
    );
    assert_eq!(code, 1, "{out:?}");
    // a.txt keeps its zzqxtwo hit (the allow names zzqxone), sub/a.txt keeps its zzqxone hit (the allow names zzqxtwo),
    // and the glob `a.txt` is anchored, so it does not cover sub/a.txt.
    assert_eq!(
        out,
        strings(&[
            "a.txt:2: zzqxtwo",
            "b.txt:1: zzqxone",
            "b.txt:1: zzqxtwo",
            "sub/a.txt:1: zzqxone",
        ])
    );
    let (code, out) = run_on(
        &repo,
        "zzqxone\nzzqxtwo\nallow * zzqxone\nallow * zzqxtwo\n",
    );
    assert_eq!(code, 0, "{out:?}");
    assert_eq!(out, strings(&["clean: 3 files"]));
}

#[ignore = "needs pwsh"]
#[test]
fn utf16le_text_is_found_at_either_byte_alignment_and_case_is_ignored() {
    let mut aligned = vec![0xff, 0xfe];
    aligned.extend(utf16le("a line\nthe ZZQXWIDE marker\n"));
    let mut odd = vec![0x00];
    odd.extend(utf16le("zzqxwide"));
    let repo = scratch_repo(
        "utf16",
        &[
            ("aligned.txt", aligned),
            ("odd.bin", odd),
            ("plain.txt", b"nothing here\n".to_vec()),
        ],
    );
    let (code, out) = run_on(&repo, "zzqxwide\n");
    assert_eq!(code, 1, "{out:?}");
    assert_eq!(
        out,
        strings(&["aligned.txt:2: zzqxwide", "odd.bin:1: zzqxwide"])
    );
}

#[ignore = "needs pwsh"]
#[test]
fn a_user_profile_path_is_found_in_every_spelling_and_for_every_drive() {
    // The paths are put together here so that this file does not hold a profile path itself.
    let (bs, u) = ('\\', "Users");
    let single = format!("see Q:{bs}{u}{bs}someone{bs}x\n");
    let double = format!("\"E:{bs}{bs}{}{bs}{bs}someone\"\n", u.to_lowercase());
    let slash = format!("d:/{u}/someone/x\n");
    let fine = format!("C:{bs}{u} alone, /{u}/x and D:/Program Files\n");
    let repo = scratch_repo(
        "profile",
        &[
            ("single.txt", single.into_bytes()),
            ("double.txt", double.into_bytes()),
            ("slash.txt", slash.into_bytes()),
            ("fine.txt", fine.into_bytes()),
        ],
    );
    let (code, out) = run_on(&repo, "# no markers\n");
    assert_eq!(code, 1, "{out:?}");
    assert_eq!(
        out,
        strings(&[
            "double.txt:1: user-profile path",
            "single.txt:1: user-profile path",
            "slash.txt:1: user-profile path",
        ])
    );
}

#[ignore = "needs pwsh and the pinned gitleaks"]
#[test]
fn a_secret_in_the_export_fails_the_check_and_is_redacted() {
    // Built at run time so this file holds no literal key; see tools/scan-secrets.ps1.
    let key = ["AKIA", "QYLPMN5H", "HKQPZ4TR"].concat();
    let repo = scratch_repo(
        "secret",
        &[
            ("a.txt", b"some text\n".to_vec()),
            (
                "conf.txt",
                format!("aws_access_key_id = {key}\n").into_bytes(),
            ),
        ],
    );
    let (code, out) = run_on(&repo, "zzqx-not-a-marker\n");
    let text = out.join("\n");
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("conf.txt"), "{text}");
    assert!(!text.contains(&key), "the key is in the output: {text}");
    assert!(!text.contains("clean:"), "{text}");
}

#[ignore = "needs pwsh"]
#[test]
fn an_allowed_longer_marker_covers_the_shorter_markers_inside_its_own_match() {
    // `zzqx` sits inside the allowed `zzqxlong-fixed`, the way the drive marker sits inside the oracle root's path.
    let repo = scratch_repo(
        "allow-span",
        &[
            (
                "a.txt",
                b"zzqxlong-fixed, and again zzqxlong-fixed\n".to_vec(),
            ),
            (
                "b.txt",
                b"zzqxlong-fixed\nzzqxlong-fixed and zzqxother\n".to_vec(),
            ),
            ("c.txt", b"zzqxlong-fixed\n".to_vec()),
        ],
    );
    let (code, out) = run_on(
        &repo,
        "zzqx\nzzqxlong-fixed\nallow a.txt zzqxlong-fixed\nallow b.txt zzqxlong-fixed\n",
    );
    // a.txt is clean. b.txt keeps the `zzqx` of `zzqxother`, outside any allowed match. c.txt has no allow line, so
    // both of its markers hit.
    assert_eq!(code, 1, "{out:?}");
    assert_eq!(
        out,
        strings(&["b.txt:2: zzqx", "c.txt:1: zzqx", "c.txt:1: zzqxlong-fixed",])
    );
}

/// The repository's own markers file: the fixed oracle root is allowed in the runner and the smoke files, and no other
/// path on that drive is.
#[ignore = "needs pwsh"]
#[test]
fn the_oracle_root_is_allowed_in_the_runner_and_the_smoke_files_but_no_other_t_path_is() {
    // The markers file is not in the publish set, so a checkout of the public repository has nothing to check here.
    let Ok(markers) =
        std::fs::read_to_string(repo_root().join("docs/superpowers/publish/markers.txt"))
    else {
        return;
    };
    let bs = '\\';
    let fixed = format!("T:{bs}artemis-oracle{bs}runs{bs}m3c-smoke\n");
    let other = format!("T:{bs}other{bs}dir\n");
    let files = [
        "tools/run-gpu-tests.ps1",
        "tools/smoke/Smoke.psm1",
        "tools/smoke/opencode.ps1",
        "tools/smoke/claude-code.ps1",
    ];
    let clean: Vec<(&str, Vec<u8>)> = files
        .iter()
        .map(|f| (*f, fixed.clone().into_bytes()))
        .collect();
    let (code, out) = run_on(&scratch_repo("oracle-root-ok", &clean), &markers);
    assert_eq!(code, 0, "{out:?}");
    assert_eq!(out, strings(&["clean: 4 files"]));
    for f in files {
        let text = format!("{fixed}{other}");
        let (code, out) = run_on(
            &scratch_repo("oracle-root-other", &[(f, text.into_bytes())]),
            &markers,
        );
        assert_eq!(code, 1, "{f}: {out:?}");
        assert_eq!(out, vec![format!("{f}:2: T:{bs}")], "{f}");
    }
}
