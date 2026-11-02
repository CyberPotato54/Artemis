//! `reuse` and `disconnect`: conversations whose cache reuse is read back from the request log.
use super::{
    body, exchange, fig, line_of, named, openai_tool, plain_body, pool_gate, read_deltas,
    read_events, tool_result, usable_call, CHAT,
};
use crate::test7::judge::{self, Api, Turn};
use crate::test7::{Check, Ctx, Verdict};
use anyhow::Result;
use serde_json::{json, Map, Value};
use std::time::Duration;

/// `disconnect`'s second request must have its headers this soon after the client left.
const RESUME_HEADERS: Duration = Duration::from_millis(250);

const RULE_SENTENCES: [&str; 10] = [
    "Name every variable for what it holds, and keep functions short enough to read at once.",
    "Return early on the error path so the main path stays flat and easy to follow.",
    "Write a test before the fix, and keep the test when the fix is in.",
    "Never swallow an error; either handle it where it happens or pass it up with context.",
    "Prefer a plain loop over a clever chain when the loop reads more clearly.",
    "Keep each module's public surface small, and document every item that is public.",
    "Do not repeat a constant; give it a name and use the name everywhere.",
    "Comment the reason for a choice, never the line that makes it.",
    "Format the code with the project's formatter before every commit.",
    "Check a bound before the index is used, not after the access has failed.",
];

const NOTE_SENTENCES: [&str; 8] = [
    "Write in short sentences and keep the verbs active throughout.",
    "Use the plain word, not the long one, when both mean the same thing.",
    "Put the point of a paragraph in its first line.",
    "Give a number with its unit, and say where it came from.",
    "Keep a list to parallel items, each beginning with the same kind of word.",
    "Cut every word that carries no meaning, then read the sentence aloud.",
    "Spell out a term the first time it is used, then use the short form.",
    "End a section when its point is made, without a summary of what was said.",
];

/// `head`, then `n` lines `<label> <i>: <sentence>`, the sentences taken in turn.
fn numbered(head: &str, label: &str, sentences: &[&str], n: usize) -> String {
    let mut s = head.to_string();
    for i in 1..=n {
        s.push_str(&format!(
            "\n{label} {i}: {}",
            sentences[(i - 1) % sentences.len()]
        ));
    }
    s
}

/// A's system prompt: numbered coding rules, about 3,000 tokens.
fn rules_prompt() -> String {
    numbered(
        "You are a careful coding assistant. Follow every rule below.",
        "Rule",
        &RULE_SENTENCES,
        125,
    )
}

/// B's system prompt: numbered style notes, about 2,000 tokens.
fn notes_prompt() -> String {
    numbered(
        "You are an editor. Apply every style note below.",
        "Style note",
        &NOTE_SENTENCES,
        90,
    )
}

/// One request of `reuse` that has a log line to read back.
struct Sent {
    name: &'static str,
    id: String,
    status: u16,
    /// The reply had no `read_file` call, so the driver put the scripted one in the history.
    scripted: bool,
}

/// One turn of conversation A: sends it, then adds to `convo` the reply's `read_file` call (the scripted one when the
/// reply had none, with the reply's reasoning) and its canned result, so the next turn has a tool result to answer.
async fn a_turn(
    ctx: &Ctx,
    convo: &mut Vec<Value>,
    name: &'static str,
    sent: &mut Vec<Sent>,
) -> Result<()> {
    let extra = json!({"reasoning_effort": "none", "tools": [openai_tool()]});
    let reply = exchange(ctx, Api::OpenAi, body(json!(convo), extra, 512, false)).await?;
    let (call, scripted) = usable_call(&reply, &format!("call_scripted_{name}"));
    let mut assistant = json!({"role": "assistant", "content": null, "tool_calls": [{"id": call.id,
        "type": "function", "function": {"name": call.name, "arguments": call.arguments}}]});
    if !reply.reasoning.is_empty() {
        assistant["reasoning_content"] = json!(reply.reasoning);
    }
    convo.push(assistant);
    convo.push(json!({"role": "tool", "tool_call_id": call.id,
        "content": tool_result(call.path().unwrap_or("notes.txt"))}));
    sent.push(Sent {
        name,
        id: reply.id,
        status: reply.status,
        scripted,
    });
    Ok(())
}

/// A turn's figures from its log line, for the floors; `None` when the line lacks the cut or the reuse.
fn turn_of(l: &Value) -> Option<Turn> {
    Some(Turn {
        cut_at: u32::try_from(l["cut_at"].as_u64()?).ok()?,
        user_query: l["user_query"].as_bool().unwrap_or(false),
        reused: u32::try_from(l["reused_tokens"].as_u64()?).ok()?,
        restored: l["restored"].as_str().unwrap_or("none").to_string(),
        restore_bytes: l["restore_bytes"].as_u64().unwrap_or(0),
    })
}

/// `reuse` (on server A): conversation A (about 3,000 tokens of system prompt and a `read_file` tool) over
/// a tool loop (t1, t2, t3), a new user turn (t4), then a conversation B of about 2,000 tokens from position 0, a side
/// call, and A's next turn (a5); then `/v1/unload`, `/v1/load` and A's next turn (a6). The floors are read from the
/// requests' log lines (`judge::floors_a_b_c`), so the driver computes nothing from the template. A pool below 2 GiB
/// in any of the log's loads makes the verdict inconclusive: its first check says so and the figures carry
/// `inconclusive: true`.
pub async fn reuse(ctx: &Ctx) -> Result<Verdict> {
    let system = json!({"role": "system", "content": rules_prompt()});
    let mut convo = vec![
        system,
        json!({"role": "user", "content": "Read notes.txt with the read_file tool, then follow the rules in what you do next."}),
    ];
    let mut sent: Vec<Sent> = Vec::new();
    for name in ["t1", "t2", "t3"] {
        a_turn(ctx, &mut convo, name, &mut sent).await?;
    }
    convo
        .push(json!({"role": "user", "content": "Now read notes.txt again and say what changed."}));
    a_turn(ctx, &mut convo, "t4", &mut sent).await?;

    // Conversation B from position 0, then a side call: the live state is no longer A's.
    let mut other: Vec<(&str, u16)> = Vec::new();
    let b_messages = json!([{"role": "system", "content": notes_prompt()},
        {"role": "user", "content": "Say in one sentence what the style notes ask for."}]);
    let b = exchange(ctx, Api::OpenAi, plain_body(b_messages, 64, false)).await?;
    other.push(("b", b.status));
    let side = json!([{"role": "user", "content": "Reply with the single word ready."}]);
    let s = exchange(ctx, Api::OpenAi, plain_body(side, 30, false)).await?;
    other.push(("side", s.status));
    let b_line = line_of(ctx, &b.id).await?;

    a_turn(ctx, &mut convo, "a5", &mut sent).await?;
    let unload = ctx.http.send("POST", "/v1/unload", None).await?;
    other.push(("unload", unload.status));
    let load = ctx.http.send("POST", "/v1/load", None).await?;
    other.push(("load", load.status));
    a_turn(ctx, &mut convo, "a6", &mut sent).await?;

    // What the server's log says of each A turn.
    let mut turns: Vec<Turn> = Vec::new();
    let mut unreadable: Vec<&str> = Vec::new();
    let mut figs: Vec<Value> = Vec::new();
    let mut reply_end_hits: Vec<&str> = Vec::new();
    for s in &sent {
        let line = line_of(ctx, &s.id).await?;
        let l = line.as_deref();
        match l.and_then(turn_of) {
            Some(t) => {
                if matches!(t.restored.as_str(), "reply_end" | "live") {
                    reply_end_hits.push(s.name);
                }
                turns.push(t);
            }
            None => unreadable.push(s.name),
        }
        figs.push(
            json!({"name": s.name, "status": s.status, "scripted": s.scripted,
            "prompt_tokens": fig(l, "prompt_tokens"), "cut_at": fig(l, "cut_at"),
            "user_query": fig(l, "user_query"), "reused": fig(l, "reused_tokens"),
            "restored": fig(l, "restored"), "restore_bytes": fig(l, "restore_bytes"),
            "restore_ms": fig(l, "restore_ms"), "cause": fig(l, "cause"),
            "prefilled_tokens": fig(l, "prefilled_tokens")}),
        );
    }

    let (pool_check, pool_figures) = pool_gate(ctx)?;
    let mut checks = vec![pool_check];
    let bad: Vec<String> = sent
        .iter()
        .map(|s| (s.name, s.status))
        .chain(other.iter().copied())
        .filter(|&(_, status)| status != 200)
        .map(|(n, status)| format!("{n} {status}"))
        .collect();
    checks.push(Check::new(
        "requests_answer_200",
        bad.is_empty(),
        if bad.is_empty() {
            "every request answered 200".to_string()
        } else {
            bad.join(", ")
        },
    ));
    if unreadable.is_empty() {
        checks.extend(judge::floors_a_b_c(&turns));
    } else {
        checks.push(Check::new(
            "turn_lines",
            false,
            format!(
                "no cut_at or reused_tokens in the log line of {}",
                unreadable.join(", ")
            ),
        ));
    }

    let mut figures = Map::new();
    figures.insert("turns".into(), json!(figs));
    figures.insert(
        "scripted_calls".into(),
        json!(sent
            .iter()
            .filter(|s| s.scripted)
            .map(|s| s.name)
            .collect::<Vec<_>>()),
    );
    figures.insert("reply_end_hits".into(), json!(reply_end_hits));
    figures.insert("a_prompt_tokens".into(), figs[0]["prompt_tokens"].clone());
    figures.insert(
        "b_prompt_tokens".into(),
        fig(b_line.as_deref(), "prompt_tokens"),
    );
    figures.extend(pool_figures);
    Ok(Verdict::judged("reuse", checks, figures))
}

/// `disconnect` (on server A): thinking off, "Count from 1 to 400, one number per line.", streaming,
/// `max_tokens` 2,000. After 40 content deltas the client disconnects (`t0`) and at once sends the same conversation
/// plus the partial reply as the assistant turn and "Continue." (streaming). The first request's line says 200 and
/// `cancelled`; the second's headers come within 250 ms of `t0`; the second reuses at least the first's `cut_at`.
pub async fn disconnect(ctx: &Ctx) -> Result<Verdict> {
    let user = json!({"role": "user", "content": "Count from 1 to 400, one number per line."});
    let mut first = ctx
        .http
        .stream(CHAT, &plain_body(json!([user]), 2000, true))
        .await?;
    let mut figures = Map::new();
    if first.status != 200 {
        figures.insert("first_status".into(), json!(first.status));
        let check = Check::new("first_streams", false, format!("status {}", first.status));
        return Ok(Verdict::judged("disconnect", vec![check], figures));
    }
    let read = read_deltas(&mut first, 40).await;
    if read.count < 40 {
        let check = Check::new(
            "forty_deltas",
            false,
            format!(
                "the stream gave {} content deltas, then ended or stalled",
                read.count
            ),
        );
        return Ok(Verdict::judged("disconnect", vec![check], figures));
    }
    let t0 = first.disconnect();

    // At once: the same conversation, the partial reply and "Continue.".
    let messages = json!([user, {"role": "assistant", "content": read.text},
        {"role": "user", "content": "Continue."}]);
    let mut second = ctx
        .http
        .stream(CHAT, &plain_body(messages, 200, true))
        .await?;
    let headers = second.headers_at.saturating_duration_since(t0);
    let status = second.status;
    let mut second_id = String::new();
    if status == 200 {
        let (events, _) = read_events(&mut second).await;
        second_id = judge::openai_stream(status, &events, false).id;
    }

    let first_line = line_of(ctx, &read.id).await?;
    let second_line = line_of(ctx, &second_id).await?;
    let first_cut = first_line.as_ref().and_then(|l| l["cut_at"].as_u64());
    let reused = second_line
        .as_ref()
        .and_then(|l| l["reused_tokens"].as_u64());

    let (pool_check, pool_figures) = pool_gate(ctx)?;
    let mut checks = vec![pool_check];
    checks.push(Check::new(
        "second_answers",
        status == 200,
        format!("status {status}"),
    ));
    let (logged_status, finish) = match first_line.as_deref() {
        Some(l) => (
            l["status"].as_u64(),
            l["finish"].as_str().map(str::to_string),
        ),
        None => (None, None),
    };
    checks.push(Check::new(
        "first_cancelled",
        logged_status == Some(200) && finish.as_deref() == Some("cancelled"),
        match first_line {
            Some(_) => format!("logged status {logged_status:?}, finish {finish:?}"),
            None => "the server wrote no log line for the first request".to_string(),
        },
    ));
    checks.push(named(
        "second_headers_by_250_ms",
        judge::within(headers, Duration::ZERO, RESUME_HEADERS),
    ));
    checks.push(match (reused, first_cut) {
        (Some(r), Some(c)) => judge::floor(
            "second_reused_floor",
            u32::try_from(r).unwrap_or(u32::MAX),
            u32::try_from(c).unwrap_or(u32::MAX),
            "the first request's cut_at",
        ),
        _ => Check::new(
            "second_reused_floor",
            false,
            "no reused_tokens or cut_at in the log lines",
        ),
    });

    figures.insert("deltas_read".into(), json!(read.count));
    figures.insert("first_cut_at".into(), json!(first_cut));
    figures.insert("second_reused".into(), json!(reused));
    figures.insert(
        "second_restored".into(),
        fig(second_line.as_deref(), "restored"),
    );
    figures.insert(
        "reused_minus_cut".into(),
        match (reused, first_cut) {
            (Some(r), Some(c)) => json!(r as i64 - c as i64),
            _ => Value::Null,
        },
    );
    figures.insert(
        "t1_minus_t0_ms".into(),
        json!(headers.as_secs_f64() * 1000.0),
    );
    figures.extend(pool_figures);
    Ok(Verdict::judged("disconnect", checks, figures))
}
