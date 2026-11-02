//! The request log, read back: the server's `artemis-YYYY-MM-DD.jsonl` files in its log directory. A request's line
//! carries the figures the cases judge (`cut_at`, `user_query`, `reused_tokens`, `finish`, ...); a lifecycle line
//! carries an `event`. The log's `id` is the response's `id` (`handlers/chat.rs`, `handlers/messages.rs`), and a
//! streamed request's line is written when its stream ends.
use anyhow::{Context, Result};
use serde_json::Value;
use std::ops::Deref;
use std::path::Path;
use std::time::{Duration, Instant};

/// How often `request` reads the directory again.
const POLL: Duration = Duration::from_millis(50);

/// A request's line did not appear in time: the one error of `request` that is not a harness error (a server that
/// wrote none is a finding for the case). Unreadable directories and malformed lines are other errors.
#[derive(Debug)]
pub struct NoLine(pub String);

impl std::fmt::Display for NoLine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NoLine {}

/// One request's log line.
#[derive(Clone, Debug)]
pub struct RequestLine(pub Value);

impl Deref for RequestLine {
    type Target = Value;
    fn deref(&self) -> &Value {
        &self.0
    }
}

/// Every complete line of every `artemis-*.jsonl` in `dir`, oldest file first. A last line with no newline yet is a
/// write in progress and is left out; a complete line that is not JSON, or a directory that can't be read, is an error.
pub fn read_lines(dir: &Path) -> Result<Vec<Value>> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read the log directory {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| e == "jsonl")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("artemis-"))
        })
        .collect();
    files.sort();
    let mut lines = Vec::new();
    for p in files {
        let text = std::fs::read_to_string(&p)
            .with_context(|| format!("cannot read the log {}", p.display()))?;
        let whole = text.rfind('\n').map_or("", |i| &text[..=i]);
        for l in whole.lines().filter(|l| !l.trim().is_empty()) {
            lines.push(
                serde_json::from_str(l)
                    .with_context(|| format!("a line of {} is not JSON", p.display()))?,
            );
        }
    }
    Ok(lines)
}

/// The line of the request whose `id` is `id`, polled every 50 ms until `within` has passed.
pub async fn request(dir: &Path, id: &str, within: Duration) -> Result<RequestLine> {
    let end = Instant::now() + within;
    loop {
        if let Some(l) = read_lines(dir)?
            .into_iter()
            .find(|l| l.get("event").is_none() && l["id"] == id)
        {
            return Ok(RequestLine(l));
        }
        if Instant::now() >= end {
            return Err(NoLine(format!(
                "no log line for request {id} in {} within {within:?}",
                dir.display()
            ))
            .into());
        }
        tokio::time::sleep(POLL).await;
    }
}

/// The lifecycle lines named `event`, oldest first.
pub fn lifecycle(dir: &Path, event: &str) -> Result<Vec<Value>> {
    Ok(read_lines(dir)?
        .into_iter()
        .filter(|l| l["event"] == event)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("artemis-test7-log-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Two dated files, a foreign file, and a last line still being written.
    fn write(d: &Path) {
        std::fs::write(
            d.join("artemis-2026-10-25.jsonl"),
            "{\"event\":\"start\"}\n{\"id\":\"chatcmpl-a\",\"status\":200,\"cut_at\":7}\n",
        )
        .unwrap();
        std::fs::write(
            d.join("artemis-2026-10-26.jsonl"),
            "{\"event\":\"load\",\"seconds\":9.5}\n{\"id\":\"msg_b\",\"status\":503}\n{\"id\":\"msg_c\",\"sta",
        )
        .unwrap();
        std::fs::write(d.join("other.jsonl"), "{\"id\":\"chatcmpl-x\"}\n").unwrap();
    }

    #[tokio::test]
    async fn request_finds_a_line_by_id_and_times_out_naming_the_id() {
        let d = scratch("request");
        write(&d);
        let l = request(&d, "chatcmpl-a", Duration::from_secs(1))
            .await
            .unwrap();
        assert_eq!(
            (l["status"].as_u64(), l["cut_at"].as_u64()),
            (Some(200), Some(7))
        );
        let l = request(&d, "msg_b", Duration::from_secs(1)).await.unwrap();
        assert_eq!(l["status"], 503);
        // The unfinished line and the foreign file are not lines of the log.
        for id in ["msg_c", "chatcmpl-x"] {
            let err = request(&d, id, Duration::from_millis(120))
                .await
                .err()
                .unwrap();
            assert!(err.to_string().contains(id), "{err}");
            assert!(err.downcast_ref::<NoLine>().is_some(), "{err}");
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn request_sees_a_line_written_while_it_polls() {
        let d = scratch("late");
        write(&d);
        let late = d.join("artemis-2026-10-26.jsonl");
        let writer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let mut t = std::fs::read_to_string(&late).unwrap();
            t.push_str("tus\":200}\n");
            std::fs::write(&late, t).unwrap();
        });
        let l = request(&d, "msg_c", Duration::from_secs(5)).await.unwrap();
        assert_eq!(l["status"], 200);
        writer.await.unwrap();
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn lifecycle_returns_the_named_events_in_order() {
        let d = scratch("lifecycle");
        write(&d);
        let load = lifecycle(&d, "load").unwrap();
        assert_eq!(load.len(), 1);
        assert_eq!(load[0]["seconds"], 9.5);
        assert!(lifecycle(&d, "exit").unwrap().is_empty());
        assert_eq!(read_lines(&d).unwrap().len(), 4);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn a_complete_line_that_is_not_json_and_a_missing_directory_are_errors() {
        let d = scratch("bad");
        std::fs::write(d.join("artemis-2026-10-26.jsonl"), "not json\n").unwrap();
        assert!(read_lines(&d).is_err());
        // `request` passes both on as they are: neither is a `NoLine`, so a case treats them as harness errors.
        let err = request(&d, "x", Duration::from_millis(100))
            .await
            .err()
            .unwrap();
        assert!(err.downcast_ref::<NoLine>().is_none(), "{err:#}");
        let _ = std::fs::remove_dir_all(&d);
        assert!(read_lines(&d).is_err());
        let err = request(&d, "x", Duration::from_millis(100))
            .await
            .err()
            .unwrap();
        assert!(err.downcast_ref::<NoLine>().is_none(), "{err:#}");
    }
}
