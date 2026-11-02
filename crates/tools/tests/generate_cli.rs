//! `artemis generate` refuses a speculation flag that can't run before it reads the prompt or opens the model.
//! No test loads a model: each names one that doesn't exist, so a refusal that came after the
//! load would say the model can't be opened instead.
use std::process::{Command, Output, Stdio};

const MISSING: &str = "no-such-model-for-generate-cli.gguf";

fn generate(extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_artemis"))
        .args(["generate", "--model", MISSING, "--prompt", "hello"])
        .args(extra)
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn the_cpu_share_with_speculation_is_refused_before_the_load() {
    // Speculation is on by default, so a share alone is the conflict.
    for extra in [
        &["--cpu-share", "0.5"][..],
        &["--cpu-share", "1", "--speculation", "on"],
    ] {
        let o = generate(extra);
        let e = stderr(&o);
        assert_eq!(o.status.code(), Some(1), "{extra:?}: {e}");
        assert!(e.contains("--cpu-share needs --speculation off"), "{e}");
        assert!(!e.contains("opening"), "the model was opened first: {e}");
    }
    // The same share with speculation off gets past the check, to the model that isn't there.
    let o = generate(&["--cpu-share", "0.5", "--speculation", "off"]);
    let e = stderr(&o);
    assert_eq!(o.status.code(), Some(1), "{e}");
    assert!(
        e.contains("opening") && !e.contains("needs --speculation"),
        "{e}"
    );
    // And a share under 1/512, which the engine takes as no CPU path, is not a conflict.
    let e = stderr(&generate(&["--cpu-share", "0.001"]));
    assert!(
        e.contains("opening") && !e.contains("needs --speculation"),
        "{e}"
    );
}

#[test]
fn the_base_time_with_speculation_is_refused_before_the_load() {
    let o = generate(&["--base-time"]);
    let e = stderr(&o);
    assert_eq!(o.status.code(), Some(1), "{e}");
    assert!(
        e.contains("--base-time needs --speculation off") && !e.contains("opening"),
        "{e}"
    );
}

#[test]
fn a_speculation_flag_with_speculation_off_is_refused() {
    for (flag, value) in [
        ("--max-draft", "4"),
        ("--ngram-min", "2"),
        ("--ngram-max", "4"),
        ("--min-gain", "1.03"),
        ("--accept-half-life", "32"),
        ("--accept-prior", "0.5"),
        ("--cost-slope-ms", "2.5"),
        ("--draft-len", "4"),
    ] {
        let o = generate(&["--speculation", "off", flag, value]);
        let e = stderr(&o);
        assert_eq!(o.status.code(), Some(1), "{flag}: {e}");
        assert!(
            e.contains(&format!("{flag} needs --speculation on")),
            "{flag}: {e}"
        );
        assert!(
            !e.contains("opening"),
            "{flag}: the model was opened first: {e}"
        );
    }
}

#[test]
fn a_bad_speculation_setting_is_refused_under_its_flags_name_before_the_load() {
    let o = generate(&["--draft-len", "16"]);
    let e = stderr(&o);
    assert_eq!(o.status.code(), Some(1), "{e}");
    assert!(
        e.contains("--draft-len must be at most 15") && !e.contains("opening"),
        "{e}"
    );
    let e = stderr(&generate(&["--max-draft", "0"]));
    assert!(
        e.contains("--max-draft: must be from 1 to 15") && !e.contains("opening"),
        "{e}"
    );
}
