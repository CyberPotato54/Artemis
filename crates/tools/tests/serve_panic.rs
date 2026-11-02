//! The panic hook. A panic hook is process-wide, so this is its own test binary and one
//! test: the hook is installed, the panics run, the hook is taken back, and only then are the lines checked, so a
//! failing assertion prints its own message.
mod support;

use artemis_tools::serve::panic::{install, panic_line};
use serde_json::Value;
use server::logs::{LifecycleLine, Logger};
use std::path::Path;
use std::time::{Duration, Instant};
use support::{lifecycle_lines, scratch_dir};

/// Runs `f` on a thread called `name` and waits for it; whether it panicked. It asserts nothing, so a failure never
/// panics inside the redacting hook: the caller checks after taking the hook back.
fn panics_on(name: &str, f: impl FnOnce() + Send + 'static) -> bool {
    std::thread::Builder::new()
        .name(name.into())
        .spawn(f)
        .is_ok_and(|h| h.join().is_err())
}

/// Every byte of every file under `dir`.
fn all_text(dir: &Path) -> String {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| String::from_utf8_lossy(&std::fs::read(e.unwrap().path()).unwrap()).into_owned())
        .collect()
}

fn panics(dir: &Path) -> Vec<Value> {
    lifecycle_lines(dir)
        .into_iter()
        .filter(|l| l["event"] == "panic")
        .collect()
}

/// Waits until `dir` holds `n` panic lines, for at most 30 s. The hook gives its write 2 s (`WRITE_BOUND`) and goes on
/// without it, so on a loaded machine the line can land after the panicking thread has been joined; the bound is only
/// the hook's, not the line's. Never panics: a short count is left for the assertions to print.
fn wait_for_panics(dir: &Path, n: usize) {
    let end = Instant::now() + Duration::from_secs(30);
    while panics(dir).len() < n && Instant::now() < end {
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn the_panic_hook() {
    let dir = scratch_dir("panic");
    install(Logger::new(dir.clone(), 0).unwrap());

    // A fixed text is a `&'static str` payload: it is compiled into the binary, so the line may carry it.
    let mut panicked = vec![panics_on("artemis-worker", || panic!("a fixed text"))];
    wait_for_panics(&dir, 1);
    let after_static = lifecycle_lines(&dir);

    // Formatted payloads are `String`s and can hold request text: the line withholds them. The marker is built at run
    // time, so `format_args!` cannot fold it into a static string.
    let marker = format!("MARKER-{}", std::process::id());
    let m = marker.clone();
    panicked.push(panics_on("artemis-http", move || panic!("{}", m)));
    let m = marker.clone();
    panicked.push(panics_on("artemis-http", move || {
        std::hint::black_box(Err::<(), String>(m)).unwrap();
    }));

    wait_for_panics(&dir, 3);
    let _ = std::panic::take_hook();

    assert_eq!(panicked, [true; 3], "every thread should have panicked");

    assert_eq!(after_static.len(), 1, "{after_static:?}");
    let first = &after_static[0];
    assert_eq!(first["event"], "panic");
    assert_eq!(first["thread"], "artemis-worker");
    assert!(first["location"]
        .as_str()
        .unwrap()
        .contains("serve_panic.rs"));
    assert_eq!(first["message"], "a fixed text");

    let lines = panics(&dir);
    assert_eq!(lines.len(), 3, "{lines:?}");
    for l in &lines[1..] {
        assert_eq!(l["thread"], "artemis-http");
        assert!(l["location"].as_str().unwrap().contains("serve_panic.rs"));
        assert!(l["message"].is_null(), "{l}");
    }
    assert!(!all_text(&dir).contains(&marker));
    assert!(
        lifecycle_lines(&dir).iter().all(|l| l["event"] != "exit"),
        "a panic is not an exit"
    );

    // A `String` payload gives no message; an unnamed thread is "unnamed"; `location` is `file:line:col`.
    let payload: Box<dyn std::any::Any + Send> = Box::new(String::from("formatted"));
    let line = panic_line(None, Some(std::panic::Location::caller()), &*payload);
    let LifecycleLine::Panic {
        thread,
        location,
        message,
    } = line
    else {
        panic!("not a panic line");
    };
    assert_eq!(thread, "unnamed");
    assert_eq!(message, None);
    let location = location.unwrap();
    assert_eq!(location.split(':').count(), 3, "{location}");
}
