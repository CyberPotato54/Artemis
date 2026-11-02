//! The panic hook of `artemis serve`. A panic writes one `panic` line: the thread, the location,
//! and the message only when the payload is a `&'static str`. A formatted payload (a `Debug` of an error, `expect`'s
//! text, an index error) can carry request text, so neither the line nor stderr shows it; `prompt.rs` keeps payloads
//! out of the logs for the same reason.
//!
//! The hook never panics: a panic inside it aborts the process. So it uses no `unwrap`, `expect` or `eprintln!`.
use server::logs::{LifecycleLine, Logger};
use std::any::Any;
use std::io::Write;
use std::panic::Location;
use std::time::Duration;

/// How long the hook waits for the line to be written: the log's lock may be held by the panicking thread itself, which
/// holds it until the hook returns and it unwinds.
const WRITE_BOUND: Duration = Duration::from_secs(2);

/// The name of the logger's helper thread (`Logger::lifecycle_within`). A panic there must not call the logger.
const LOG_THREAD: &str = "artemis-log";

/// What of the payload may be shown: its text for a `&'static str`, nothing for anything else. Not
/// `PanicHookInfo::payload_as_str`, which also returns `String` payloads. The line and stderr both use this, through
/// `panic_line`.
fn redacted(payload: &(dyn Any + Send)) -> Option<String> {
    payload
        .downcast_ref::<&'static str>()
        .map(|s| (*s).to_string())
}

/// The `panic` line for a panic on `thread` (`None` for an unnamed thread) at `location`.
pub fn panic_line(
    thread: Option<&str>,
    location: Option<&Location<'_>>,
    payload: &(dyn Any + Send),
) -> LifecycleLine {
    LifecycleLine::Panic {
        thread: thread.unwrap_or("unnamed").to_string(),
        location: location.map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column())),
        message: redacted(payload),
    }
}

/// Replaces the default hook with one that writes the `panic` line through `logger` and prints a backtrace when one is
/// captured (`RUST_BACKTRACE` set). It does not call the default hook, whose stderr text would show the message.
///
/// The line reaches stderr through the logger's warn-level `tracing` arm, which runs before the file lock is taken, so
/// the hook adds no stderr line of its own and each panic prints once. On the logger's own helper thread the hook can't
/// call the logger, so it writes the line's text to stderr itself.
pub fn install(logger: Logger) {
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let line = panic_line(thread.name(), info.location(), info.payload());
        if thread.name() == Some(LOG_THREAD) {
            let _ = writeln!(std::io::stderr(), "{}", line.human());
        } else {
            let _ = logger.lifecycle_within(&line, WRITE_BOUND);
        }
        let bt = std::backtrace::Backtrace::capture();
        if bt.status() == std::backtrace::BacktraceStatus::Captured {
            let _ = writeln!(std::io::stderr(), "{bt}");
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only a `&'static str` payload shows its text; the stderr text is the line's `human()`, so it withholds the same.
    #[test]
    fn redaction_keeps_static_text_only() {
        let fixed: Box<dyn Any + Send> = Box::new("a fixed text");
        let formatted: Box<dyn Any + Send> = Box::new(String::from("request text"));
        let other: Box<dyn Any + Send> = Box::new(7_u32);
        assert_eq!(redacted(&*fixed).as_deref(), Some("a fixed text"));
        assert_eq!(redacted(&*formatted), None);
        assert_eq!(redacted(&*other), None);

        let shown = panic_line(Some("t"), Some(Location::caller()), &*fixed).human();
        assert!(shown.starts_with("panic in t at "), "{shown}");
        assert!(shown.ends_with(": a fixed text"), "{shown}");
        let withheld = panic_line(Some("t"), None, &*formatted).human();
        assert_eq!(
            withheld,
            "panic in t at an unknown place: (message withheld: it may carry request text)"
        );
    }
}
