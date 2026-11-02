//! The cases. Every gating request uses `temperature: 0` (greedy) and
//! `seed: 7`, so a run is repeatable. Prompts are built here from fixed strings: no fixture, and no client prompt text.
//! A case returns a `Verdict`; a failed check is a `fail`, an `Err` a harness error.
//!
//! `requests` holds `load`, `requests` and `budget`; `tool_loop` the two cases that run a conversation with a tool and
//! read the cache's reuse back from the log (`reuse`, `disconnect`); `chat_tools` the client-shaped tool loop
//! (`chat-tools`); `lifecycle` the three that act on the server's own state (`idle`, `unload-stream`, `queue`); `speed`
//! the reported speed prompts, and `speed-fresh` one of them (or the long prefill) per server process. What they share
//! (the request bodies, `exchange`, the log helpers) is here.
mod chat_tools;
mod lifecycle;
mod requests;
mod speed;
mod tool_loop;

pub use chat_tools::chat_tools;
pub use lifecycle::{idle, queue, unload_stream};
pub use requests::{budget, load, requests, BUDGET_PROMPT, LOG_GATED, REQUESTS_STEPS};
pub use speed::{speed, speed_fresh, unwrap_chatml_user};
pub use tool_loop::{disconnect, reuse};

use super::http::Sse;
use super::judge::{self, Api, Call, Ev, Poll, Reply};
use super::{log, Check, Ctx};
use anyhow::Result;
use serde_json::{json, Map, Value};
use std::time::{Duration, Instant};

/// The model name every request carries; the server echoes it.
const MODEL: &str = "qwen3.6-35b-a3b";
/// A `/health` poll that takes longer than this is a harness error: the server answers it within a second.
const POLL_WITHIN: Duration = Duration::from_secs(10);
/// The longest silence in a stream (the server's keep-alives come every 10 s) before it counts as stalled.
const GAP: Duration = Duration::from_secs(120);
/// How long a request's log line may take to appear after its reply.
const LOG_WITHIN: Duration = Duration::from_secs(5);

/// One `/health` read, and when it was answered (ms since `began`).
async fn poll(ctx: &Ctx, began: Instant) -> Result<(Poll, f64)> {
    let r = tokio::time::timeout(POLL_WITHIN, ctx.http.send("GET", "/health", None))
        .await
        .map_err(|_| anyhow::anyhow!("/health did not answer within {POLL_WITHIN:?}"))??;
    let at = r.done_at.duration_since(began).as_secs_f64() * 1000.0;
    let body = r.json().unwrap_or(Value::Null);
    Ok((
        Poll {
            status: r.status,
            body,
        },
        at,
    ))
}

/// The call the echo carries: the reply's `read_file` call if it has one with a string path, else a scripted one
/// (`{"path": "notes.txt"}`) and `true`.
fn usable_call(reply: &Reply, scripted_id: &str) -> (Call, bool) {
    match reply.calls.as_slice() {
        [c] if c.name == "read_file" && c.path().is_some() => (c.clone(), false),
        _ => (
            Call {
                id: scripted_id.to_string(),
                name: "read_file".into(),
                arguments: "{\"path\":\"notes.txt\"}".into(),
                args: Some(json!({"path": "notes.txt"})),
            },
            true,
        ),
    }
}

/// A canned tool result: the path and 20 lines.
fn tool_result(path: &str) -> String {
    let mut s = format!("contents of {path}:\n");
    for i in 1..=20 {
        s.push_str(&format!(
            "line {i}: the quick brown fox jumps over the lazy dog\n"
        ));
    }
    s
}

fn openai_tool() -> Value {
    json!({"type": "function", "function": {"name": "read_file", "description": "Reads a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
            "required": ["path"]}}})
}

fn anthropic_tool() -> Value {
    json!({"name": "read_file", "description": "Reads a file.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
            "required": ["path"]}})
}

/// A request body for either API (the two share their fields): greedy, seed 7, `extra`'s fields on top.
pub fn body(messages: Value, extra: Value, max_tokens: u32, stream: bool) -> Value {
    let mut m = Map::new();
    m.insert("model".into(), json!(MODEL));
    m.insert("messages".into(), messages);
    m.insert("max_tokens".into(), json!(max_tokens));
    m.insert("temperature".into(), json!(0));
    m.insert("seed".into(), json!(7));
    if stream {
        m.insert("stream".into(), json!(true));
    }
    if let Value::Object(e) = extra {
        m.extend(e);
    }
    Value::Object(m)
}

/// Sends `body` to the API's endpoint and reads the whole reply, as a stream if the body asks for one.
async fn exchange(ctx: &Ctx, api: Api, body: Value) -> Result<Reply> {
    let path = match api {
        Api::OpenAi => "/v1/chat/completions",
        Api::Anthropic => "/v1/messages",
    };
    if body["stream"] != true {
        let r = ctx.http.send("POST", path, Some(&body)).await?;
        let v = r.json().unwrap_or(Value::Null);
        return Ok(match api {
            Api::OpenAi => judge::openai_reply(r.status, &v),
            Api::Anthropic => judge::anthropic_reply(r.status, &v),
        });
    }
    let mut sse = ctx.http.stream(path, &body).await?;
    if sse.status != 200 {
        let v = serde_json::from_slice(&sse.rest(GAP).await).unwrap_or(Value::Null);
        return Ok(Reply::failed(api, sse.status, &v));
    }
    let (events, stalled) = read_events(&mut sse).await;
    let want_usage = body["stream_options"]["include_usage"] == true;
    let mut reply = match api {
        Api::OpenAi => judge::openai_stream(sse.status, &events, want_usage),
        Api::Anthropic => judge::anthropic_stream(sse.status, &events),
    };
    if stalled {
        reply
            .problems
            .push(format!("the stream stalled for {GAP:?}"));
    }
    if sse.failed() {
        reply
            .problems
            .push("the stream's body ended in a transport error".into());
    }
    Ok(reply)
}

/// A stream's events until its body ends; `true` too when it went `GAP` without a byte, which ends the read.
pub async fn read_events(sse: &mut Sse) -> (Vec<Ev>, bool) {
    let mut events = Vec::new();
    loop {
        match sse.next(GAP).await {
            Some(super::http::SseItem::Event { name, data }) => events.push(Ev { name, data }),
            Some(super::http::SseItem::Comment(_)) => {}
            None => return (events, !sse.ended()),
        }
    }
}

const CHAT: &str = "/v1/chat/completions";

/// The pool gate for the reuse verdicts, over every `load` line in the log (each server has its own log directory, and
/// `reuse` spans two loads): the check (first among the case's) and the figures to add to its own: the smallest
/// `pool_limit_gib`, every load's `pool_limit_gib` and `pool_cut` in order, and `inconclusive`.
fn pool_gate(ctx: &Ctx) -> Result<(Check, Map<String, Value>)> {
    let loads = log::lifecycle(&ctx.log_dir, "load")?;
    let limits: Vec<Option<f64>> = loads.iter().map(|l| l["pool_limit_gib"].as_f64()).collect();
    let smallest = limits.iter().flatten().copied().reduce(f64::min);
    let mut figures = Map::new();
    figures.insert("pool_limit_gib".into(), json!(smallest));
    figures.insert(
        "pool_loads".into(),
        json!(loads
            .iter()
            .map(|l| json!({"pool_limit_gib": l["pool_limit_gib"], "pool_cut": l["pool_cut"]}))
            .collect::<Vec<_>>()),
    );
    figures.insert(
        "inconclusive".into(),
        json!(smallest.is_some_and(|g| g < 2.0)),
    );
    Ok((judge::pool_check(&limits, ctx.allow_missing_vram), figures))
}

/// `check` under `name`.
fn named(name: &str, check: Check) -> Check {
    Check {
        name: name.to_string(),
        ..check
    }
}

/// A request's log line, or `None` when the server wrote none in time (a finding for the case; an unreadable log is
/// a harness error).
async fn line_of(ctx: &Ctx, id: &str) -> Result<Option<log::RequestLine>> {
    if id.is_empty() {
        return Ok(None);
    }
    match log::request(&ctx.log_dir, id, LOG_WITHIN).await {
        Ok(l) => Ok(Some(l)),
        Err(e) if e.downcast_ref::<log::NoLine>().is_some() => Ok(None),
        Err(e) => Err(e),
    }
}

/// The figure `k` of a log line, `null` where there is none.
fn fig(line: Option<&Value>, k: &str) -> Value {
    line.map_or(Value::Null, |l| l[k].clone())
}

/// What `read_deltas` found.
#[derive(Default)]
struct Deltas {
    /// The stream's `id` (the log's too).
    id: String,
    /// The content read, joined.
    text: String,
    /// How many content deltas: fewer than asked when the stream ended or stalled first.
    count: usize,
}

/// Reads a chat stream until `n` content deltas (the non-empty `delta.content` chunks) have come.
async fn read_deltas(sse: &mut Sse, n: usize) -> Deltas {
    let mut d = Deltas::default();
    while d.count < n {
        match sse.next(GAP).await {
            Some(super::http::SseItem::Event { data, .. }) => {
                let Ok(v) = serde_json::from_str::<Value>(&data) else {
                    continue;
                };
                if d.id.is_empty() {
                    d.id = v["id"].as_str().unwrap_or("").to_string();
                }
                if let Some(s) = v["choices"][0]["delta"]["content"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                {
                    d.text.push_str(s);
                    d.count += 1;
                }
            }
            Some(super::http::SseItem::Comment(_)) => {}
            None => break,
        }
    }
    d
}

/// A chat request of the two-message kind the idle and side calls use: thinking off.
fn plain_body(messages: Value, max_tokens: u32, stream: bool) -> Value {
    body(
        messages,
        json!({"reasoning_effort": "none"}),
        max_tokens,
        stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every gating request is greedy with the fixed seed (so a run is repeatable).
    #[test]
    fn bodies_are_greedy_with_seed_7() {
        for b in [
            body(json!([]), json!({"reasoning_effort": "low"}), 9, true),
            body(
                json!([]),
                json!({"thinking": {"type": "disabled"}}),
                9,
                false,
            ),
        ] {
            assert_eq!(
                (&b["temperature"], &b["seed"], &b["max_tokens"]),
                (&json!(0), &json!(7), &json!(9))
            );
        }
        assert_eq!(body(json!([]), json!({}), 1, true)["stream"], true);
        assert!(body(json!([]), json!({}), 1, false).get("stream").is_none());
    }

    #[test]
    fn a_reply_without_a_call_gets_the_scripted_one() {
        let none = judge::openai_reply(
            200,
            &json!({"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "x"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "completion_tokens_details": {"reasoning_tokens": 0}}}),
        );
        let (call, scripted) = usable_call(&none, "call_scripted");
        assert!(scripted);
        assert_eq!(
            (call.id.as_str(), call.path()),
            ("call_scripted", Some("notes.txt"))
        );
    }

    #[test]
    fn the_canned_tool_result_is_the_path_and_20_lines() {
        let r = tool_result("a.py");
        assert!(r.starts_with("contents of a.py:\n"));
        assert_eq!(r.lines().count(), 21);
    }
}
