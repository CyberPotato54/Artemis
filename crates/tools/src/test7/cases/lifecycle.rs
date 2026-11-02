//! `idle`, `unload-stream` and `queue`: the cases that act on the server's own state.
use super::{fig, line_of, named, plain_body, poll, read_deltas, read_events, CHAT};
use crate::test7::judge::{self, Poll};
use crate::test7::{log, Check, Ctx, Verdict};
use anyhow::{Context, Result};
use serde_json::{json, Map, Value};
use std::time::{Duration, Instant};

/// How often `idle` reads the log for the unload line.
const IDLE_POLL: Duration = Duration::from_millis(50);
/// How long past the window's end `idle` goes on waiting, so a late unload is reported with its time, not as missing.
const IDLE_SLACK: Duration = Duration::from_secs(5);

/// `idle` writes two load lines on a fresh server: its own `/v1/load`'s and the reload's. With one, the
/// `/v1/load` found the model loaded and wrote none.
fn load_lines_check(new: usize) -> Check {
    let detail = match new {
        2.. => format!("{new} new load lines (the preload and the reload)"),
        1 => "the server was already loaded: run idle first on a fresh server".to_string(),
        0 => "no load line was written, by /v1/load or by C's next turn".to_string(),
    };
    Check::new("load_lines", new >= 2, detail)
}

/// `/health` reads 200 `idle` with `loaded: false`.
fn health_idle_check(health: &Poll) -> Check {
    Check::new(
        "health_idle",
        health.status == 200 && health.body["status"] == "idle" && health.body["loaded"] == false,
        format!(
            "/health read {} {} loaded {}",
            health.status, health.body["status"], health.body["loaded"]
        ),
    )
}

/// The VRAM of two load lines within 64 MiB (`judge::vram_figures`: no figure fails unless the hook allows it); no
/// line to compare fails.
fn vram_check(ctx: &Ctx, first: Option<&Value>, later: Option<&Value>) -> Check {
    let vram = |l: Option<&Value>| l.and_then(|l| l["vram_used_mib"].as_f64());
    match (first, later) {
        (Some(_), Some(_)) => judge::vram_figures(vram(first), vram(later), ctx.allow_missing_vram),
        _ => Check::new(
            "vram_within_64_mib",
            false,
            "a load line to compare is missing",
        ),
    }
}

/// How many idle unloads the log holds.
fn idle_unloads(ctx: &Ctx) -> Result<usize> {
    Ok(log::lifecycle(&ctx.log_dir, "unload")?
        .iter()
        .filter(|l| l["cause"] == "idle")
        .count())
}

/// `idle` (on server B, idle unload 30 s): `/v1/load` (line L1); one request in conversation C; wait for
/// the `unload` line with cause `idle`, which comes 30 to 45 s after C's request ended; `/health` reads
/// `idle`; C's next turn (line L2) answers, and the reload took the same VRAM (within 64 MiB) and slots as the first.
///
/// Precondition: a fresh server, with the model not loaded (the case adds no unload of its own, so L1 is the line
/// its own `/v1/load` writes). On a loaded server `/v1/load` writes no line and `load_lines` fails.
pub async fn idle(ctx: &Ctx) -> Result<Verdict> {
    let loads_before = log::lifecycle(&ctx.log_dir, "load")?.len();
    let idles_before = idle_unloads(ctx)?;
    let load = ctx.http.send("POST", "/v1/load", None).await?;

    let system = json!({"role": "system", "content": "You are a terse assistant."});
    let ask = json!({"role": "user", "content": "Name three prime numbers."});
    let r = ctx
        .http
        .send(
            "POST",
            CHAT,
            Some(&plain_body(json!([system, ask]), 64, false)),
        )
        .await?;
    let ended = r.done_at;
    let first = judge::openai_reply(r.status, &r.json().unwrap_or(Value::Null));

    // The unload line, polled until the window's end and a little past it.
    let (lo, hi) = ctx.idle_window;
    let give_up = ended + hi + IDLE_SLACK;
    let mut seen = None;
    loop {
        if idle_unloads(ctx)? > idles_before {
            seen = Some(Instant::now());
            break;
        }
        if Instant::now() >= give_up {
            break;
        }
        tokio::time::sleep(IDLE_POLL).await;
    }
    let after = seen.map(|t| t.saturating_duration_since(ended));

    let (health, _) = poll(ctx, Instant::now()).await?;

    let again = json!({"role": "user", "content": "And two more?"});
    let messages = json!([system, ask, {"role": "assistant", "content": first.text}, again]);
    let r2 = ctx
        .http
        .send("POST", CHAT, Some(&plain_body(messages, 64, false)))
        .await?;
    let second = judge::openai_reply(r2.status, &r2.json().unwrap_or(Value::Null));
    let second_line = line_of(ctx, &second.id).await?;

    let loads = log::lifecycle(&ctx.log_dir, "load")?;
    let (l1, l2) = (loads.get(loads_before), loads.get(loads_before + 1));
    let mut checks = vec![
        Check::new(
            "load_answers",
            load.status == 200 && first.status == 200,
            format!(
                "/v1/load {}, C's first request {}",
                load.status, first.status
            ),
        ),
        match after {
            Some(t) => named("idle_unload_in_window", judge::within(t, lo, hi)),
            None => Check::new(
                "idle_unload_in_window",
                false,
                format!(
                    "no idle unload line within {:.1} s of C's request",
                    (hi + IDLE_SLACK).as_secs_f64()
                ),
            ),
        },
        health_idle_check(&health),
        Check::new(
            "c_next_turn_answers",
            second.status == 200,
            format!("status {}", second.status),
        ),
        load_lines_check(loads.len().saturating_sub(loads_before)),
    ];
    let slots = |l: Option<&Value>| l.and_then(|l| l["slots_per_layer"].as_u64());
    checks.push(vram_check(ctx, l1, l2));
    checks.push(judge::slots_check(
        slots(l1),
        slots(l2),
        ctx.allow_missing_vram,
    ));

    let mut figures = Map::new();
    figures.insert(
        "unload_after_s".into(),
        json!(after.map(|t| t.as_secs_f64())),
    );
    figures.insert("l1_vram_used_mib".into(), fig(l1, "vram_used_mib"));
    figures.insert("l2_vram_used_mib".into(), fig(l2, "vram_used_mib"));
    figures.insert("l1_slots_per_layer".into(), fig(l1, "slots_per_layer"));
    figures.insert("l2_slots_per_layer".into(), fig(l2, "slots_per_layer"));
    figures.insert(
        "c_next_reused".into(),
        fig(second_line.as_deref(), "reused_tokens"),
    );
    figures.insert(
        "c_next_restored".into(),
        fig(second_line.as_deref(), "restored"),
    );
    Ok(Verdict::judged("idle", checks, figures))
}

/// `unload-stream` (on server B): a streaming request (`max_tokens` 1,000); after 20 deltas `POST
/// /v1/unload`; the stream ends in an in-band error event; the unload answers 200 `{"status":"idle","loaded":false}`;
/// `/health` reads `idle`; `/v1/load` answers 200 `ready`; and the VRAM of that load is within 64 MiB of the
/// first `load` line in the log, so the case runs alone too.
pub async fn unload_stream(ctx: &Ctx) -> Result<Verdict> {
    let ask = json!([{"role": "user", "content": "Write a long story about a lighthouse keeper."}]);
    let mut sse = ctx.http.stream(CHAT, &plain_body(ask, 1000, true)).await?;
    let mut figures = Map::new();
    if sse.status != 200 {
        let check = Check::new("stream_opens", false, format!("status {}", sse.status));
        return Ok(Verdict::judged("unload-stream", vec![check], figures));
    }
    let read = read_deltas(&mut sse, 20).await;
    if read.count < 20 {
        let check = Check::new(
            "twenty_deltas",
            false,
            format!(
                "the stream gave {} content deltas, then ended or stalled",
                read.count
            ),
        );
        return Ok(Verdict::judged("unload-stream", vec![check], figures));
    }
    // The unload waits for the engine, and the stream is read meanwhile.
    let http = ctx.http.clone();
    let unload = tokio::spawn(async move { http.send("POST", "/v1/unload", None).await });
    let (events, stalled) = read_events(&mut sse).await;
    let unload = unload.await.context("the unload task failed")??;
    let (health, _) = poll(ctx, Instant::now()).await?;
    let load = ctx.http.send("POST", "/v1/load", None).await?;

    let last = events.iter().rev().find(|e| e.data.trim() != "[DONE]");
    let errored = last
        .and_then(|e| serde_json::from_str::<Value>(&e.data).ok())
        .is_some_and(|v| !v["error"].is_null());
    let clean = !stalled && !sse.failed();
    let unload_body = unload.json().unwrap_or(Value::Null);
    let load_body = load.json().unwrap_or(Value::Null);
    let loads = log::lifecycle(&ctx.log_dir, "load")?;
    let (l1, l3) = (
        loads.first(),
        loads
            .len()
            .checked_sub(1)
            .filter(|&i| i > 0)
            .and_then(|i| loads.get(i)),
    );
    let mut checks = vec![
        Check::new(
            "stream_ends_in_error",
            errored && clean,
            format!(
                "{} events; the last is {}an error event; the body ended {}",
                events.len(),
                if errored { "" } else { "not " },
                if clean { "cleanly" } else { "badly" }
            ),
        ),
        Check::new(
            "unload_answers_idle",
            unload.status == 200 && unload_body == json!({"status": "idle", "loaded": false}),
            format!("status {}", unload.status),
        ),
        health_idle_check(&health),
        Check::new(
            "load_answers_ready",
            load.status == 200 && load_body["status"] == "ready",
            format!("status {}", load.status),
        ),
    ];
    checks.push(vram_check(ctx, l1, l3));
    figures.insert("deltas_before_unload".into(), json!(read.count));
    figures.insert("events_after_unload".into(), json!(events.len()));
    figures.insert("l1_vram_used_mib".into(), fig(l1, "vram_used_mib"));
    figures.insert("l3_vram_used_mib".into(), fig(l3, "vram_used_mib"));
    Ok(Verdict::judged("unload-stream", checks, figures))
}

/// `queue` (on server C, queue timeout 5 s): `/v1/load`; a long streaming request L (thinking off, "Count
/// from 1 to 2000...", `max_tokens` 1,500, about 16 s); once L's headers have arrived, a non-streaming request Q.
/// Q answers 503 "timed out waiting in the queue" 5.0 to 8.0 s after it was sent, and L answers 200.
pub async fn queue(ctx: &Ctx) -> Result<Verdict> {
    let load = ctx.http.send("POST", "/v1/load", None).await?;
    let long = json!([{"role": "user", "content": "Count from 1 to 2000, one number per line."}]);
    let mut l = ctx.http.stream(CHAT, &plain_body(long, 1500, true)).await?;
    let mut figures = Map::new();
    if l.status != 200 {
        figures.insert("l_status".into(), json!(l.status));
        let check = Check::new("l_answers", false, format!("status {}", l.status));
        return Ok(Verdict::judged("queue", vec![check], figures));
    }
    // L's headers are in, so the engine has taken L: Q waits behind it.
    let hello = json!([{"role": "user", "content": "Say hello."}]);
    let q = ctx
        .http
        .send("POST", CHAT, Some(&plain_body(hello, 16, false)))
        .await?;
    let waited = q.done_at.duration_since(q.sent_at);
    let message = q
        .json()
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_string));
    let (events, stalled) = read_events(&mut l).await;
    let reply = judge::openai_stream(l.status, &events, false);

    let (lo, hi) = ctx.queue_window;
    let checks = vec![
        Check::new(
            "load_answers",
            load.status == 200,
            format!("status {}", load.status),
        ),
        Check::new(
            "q_times_out_in_the_queue",
            q.status == 503
                && message
                    .as_deref()
                    .is_some_and(|m| m.contains("timed out waiting in the queue")),
            format!("status {}, message {:?}", q.status, message),
        ),
        named("q_in_window", judge::within(waited, lo, hi)),
        Check::new(
            "l_answers",
            l.status == 200 && reply.ended && !stalled && !l.failed(),
            format!(
                "status {}, the stream {}",
                l.status,
                if reply.ended {
                    "ended in [DONE]"
                } else {
                    "did not end in [DONE]"
                }
            ),
        ),
    ];
    figures.insert("q_status".into(), json!(q.status));
    figures.insert("q_seconds".into(), json!(waited.as_secs_f64()));
    figures.insert("l_status".into(), json!(l.status));
    Ok(Verdict::judged("queue", checks, figures))
}
