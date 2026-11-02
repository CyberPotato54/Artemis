//! The pass rules of the test7 cases, as pure functions from what the driver saw to checks. Nothing here reads
//! the network or the log, so each rule's edges are unit-tested. A detail names counts and shapes, never the model's
//! text.
use super::Check;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Duration;

/// The keys every `/health` body has, whatever the status.
pub const HEALTH_KEYS: [&str; 7] = [
    "service",
    "status",
    "version",
    "loaded",
    "model",
    "max_context",
    "api_key",
];

/// One `/health` answer.
#[derive(Clone, Debug)]
pub struct Poll {
    pub status: u16,
    pub body: Value,
}

fn health_shape(body: &Value) -> bool {
    body.as_object().is_some_and(|o| {
        o.len() == HEALTH_KEYS.len() && HEALTH_KEYS.iter().all(|k| o.contains_key(*k))
    })
}

fn reads(p: &Poll, status: u16, name: &str) -> bool {
    p.status == status && p.body["status"] == name && health_shape(&p.body)
}

/// `load`: the chat answers 200; at least one poll reads 503 `loading` with the same seven keys; the last poll reads
/// 200 `ready`.
pub fn judge_load(chat_status: u16, polls: &[Poll]) -> Vec<Check> {
    let loading = polls.iter().filter(|p| reads(p, 503, "loading")).count();
    let last = match polls.last() {
        None => "there was no poll".to_string(),
        Some(p) => format!("the last poll read {} {}", p.status, p.body["status"]),
    };
    vec![
        Check::new(
            "chat_answers",
            chat_status == 200,
            format!("status {chat_status}"),
        ),
        Check::new(
            "loading_seen",
            loading >= 1,
            format!(
                "{loading} of {} polls read 503 loading with the seven keys",
                polls.len()
            ),
        ),
        Check::new(
            "ready_last",
            polls.last().is_some_and(|p| reads(p, 200, "ready")),
            last,
        ),
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    OpenAi,
    Anthropic,
}

/// A tool call in a reply. `arguments` is the JSON text as the reply carried it (OpenAI's own form; Anthropic's
/// `input` serialised), and `args` its parse.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Call {
    pub id: String,
    pub name: String,
    pub arguments: String,
    pub args: Option<Value>,
}

impl Call {
    fn new(id: &str, name: &str, arguments: String) -> Call {
        let args = serde_json::from_str(&arguments).ok();
        Call {
            id: id.to_string(),
            name: name.to_string(),
            arguments,
            args,
        }
    }

    /// The `path` argument, if it is a string.
    pub fn path(&self) -> Option<&str> {
        self.args.as_ref()?["path"].as_str()
    }
}

/// A reply normalised across the two APIs and across streaming and not. `problems` lists every departure from the
/// expected shape that the parse found.
#[derive(Clone, Debug)]
pub struct Reply {
    pub api: Api,
    pub status: u16,
    pub id: String,
    pub text: String,
    pub reasoning: String,
    /// A reasoning field or a thinking block was present, empty or not.
    pub reasoning_present: bool,
    pub calls: Vec<Call>,
    /// OpenAI's `finish_reason` or Anthropic's `stop_reason`.
    pub finish: String,
    pub reasoning_tokens: Option<u64>,
    /// A stream ended in `[DONE]` or `message_stop`; a whole reply always counts as ended.
    pub ended: bool,
    pub problems: Vec<String>,
}

impl Reply {
    fn new(api: Api, status: u16) -> Reply {
        Reply {
            api,
            status,
            id: String::new(),
            text: String::new(),
            reasoning: String::new(),
            reasoning_present: false,
            calls: Vec::new(),
            finish: String::new(),
            reasoning_tokens: None,
            ended: false,
            problems: Vec::new(),
        }
    }

    /// A reply with a status other than 200: its error message (never more than 160 characters) is the problem.
    pub fn failed(api: Api, status: u16, body: &Value) -> Reply {
        let mut r = Reply::new(api, status);
        let message: String = body["error"]["message"]
            .as_str()
            .unwrap_or("no error message")
            .chars()
            .take(160)
            .collect();
        r.problems.push(format!("status {status}: {message}"));
        r
    }

    fn problem(&mut self, p: impl Into<String>) {
        self.problems.push(p.into());
    }
}

fn is_count(v: &Value) -> bool {
    v.is_u64()
}

/// An OpenAI `chat.completion` body (non-streaming).
pub fn openai_reply(status: u16, body: &Value) -> Reply {
    if status != 200 {
        return Reply::failed(Api::OpenAi, status, body);
    }
    let mut r = Reply::new(Api::OpenAi, status);
    r.id = body["id"].as_str().unwrap_or("").to_string();
    if !r.id.starts_with("chatcmpl-") {
        r.problem("the id is not chatcmpl-…");
    }
    if body["object"] != "chat.completion" {
        r.problem("object is not chat.completion");
    }
    if !is_count(&body["created"]) || !body["model"].is_string() {
        r.problem("created or model is missing");
    }
    let choices = body["choices"].as_array().map_or(&[][..], Vec::as_slice);
    if choices.len() != 1 {
        r.problem(format!("{} choices, not one", choices.len()));
        return r;
    }
    let c = &choices[0];
    if c["index"] != 0 {
        r.problem("the choice's index is not 0");
    }
    let m = &c["message"];
    if m["role"] != "assistant" {
        r.problem("the message's role is not assistant");
    }
    let calls = m["tool_calls"].as_array().map_or(&[][..], Vec::as_slice);
    match &m["content"] {
        Value::String(s) => r.text = s.clone(),
        Value::Null if !calls.is_empty() => {}
        _ => r.problem("content is neither a string nor null beside tool calls"),
    }
    match &m["reasoning_content"] {
        Value::Null => {}
        Value::String(s) => {
            // The server omits the field when there is no reasoning (`openai/response.rs`), so any string,
            // empty or not, is a reasoning field.
            r.reasoning = s.clone();
            r.reasoning_present = true;
        }
        _ => r.problem("reasoning_content is not a string"),
    }
    for t in calls {
        let f = &t["function"];
        if !t["id"].as_str().is_some_and(|i| i.starts_with("call_")) || t["type"] != "function" {
            r.problem("a tool call has no call_ id or is not a function");
        }
        let args = f["arguments"].as_str().unwrap_or("").to_string();
        let call = Call::new(
            t["id"].as_str().unwrap_or(""),
            f["name"].as_str().unwrap_or(""),
            args,
        );
        if call.args.is_none() {
            r.problem("a call's arguments are not JSON text");
        }
        r.calls.push(call);
    }
    match c["finish_reason"].as_str() {
        Some(f) => r.finish = f.to_string(),
        None => r.problem("finish_reason is not a string"),
    }
    let u = &body["usage"];
    if !(is_count(&u["prompt_tokens"])
        && is_count(&u["completion_tokens"])
        && is_count(&u["total_tokens"])
        && is_count(&u["prompt_tokens_details"]["cached_tokens"]))
    {
        r.problem("usage lacks its token counts");
    }
    match u["completion_tokens_details"]["reasoning_tokens"].as_u64() {
        Some(n) => r.reasoning_tokens = Some(n),
        None => r.problem("usage lacks reasoning_tokens"),
    }
    r.ended = true;
    r
}

/// Whether no choice of the `chat.completion` body `reply` carries a `function_call` in its message: the key absent,
/// `null` or `[]`, the three forms a client that refuses the legacy field accepts. A body with no choices has nothing
/// to flag (`one_choice_each` is the check that fails it).
pub fn function_call_absent(reply: &Value) -> bool {
    reply["choices"].as_array().is_none_or(|choices| {
        choices
            .iter()
            .all(|c| match &c["message"]["function_call"] {
                Value::Null => true,
                Value::Array(a) => a.is_empty(),
                _ => false,
            })
    })
}

/// One SSE event: its `event:` name, if it had one, and its `data:`.
#[derive(Clone, Debug)]
pub struct Ev {
    pub name: Option<String>,
    pub data: String,
}

/// An OpenAI stream's events: the role chunk first, deltas, a `finish_reason` chunk, then `[DONE]`. The
/// stream carries usage only when the request asked for it (`stream_options.include_usage`): with `want_usage`, the
/// usage chunk (`choices: []`) must be there with the counts of the non-streaming body; without it, a usage chunk
/// is not judged. The Anthropic stream's usage is judged in `anthropic_stream`.
pub fn openai_stream(status: u16, events: &[Ev], want_usage: bool) -> Reply {
    let mut r = Reply::new(Api::OpenAi, status);
    let mut done = false;
    let mut first = true;
    let mut usage_seen = false;
    let mut calls: BTreeMap<u64, Call> = BTreeMap::new();
    for ev in events {
        if done {
            r.problem("an event after [DONE]");
            break;
        }
        if ev.data.trim() == "[DONE]" {
            done = true;
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(&ev.data) else {
            r.problem("a chunk is not JSON");
            continue;
        };
        if !v["error"].is_null() {
            r.problem("an in-band error");
            continue;
        }
        if v["object"] != "chat.completion.chunk" {
            r.problem("a chunk's object is not chat.completion.chunk");
        }
        let id = v["id"].as_str().unwrap_or("");
        if r.id.is_empty() {
            r.id = id.to_string();
            if !id.starts_with("chatcmpl-") {
                r.problem("the id is not chatcmpl-…");
            }
        } else if r.id != id {
            r.problem("the chunks' ids differ");
        }
        let Some(choice) = v["choices"].as_array().and_then(|c| c.first()) else {
            // A usage chunk: `choices: []`.
            if want_usage {
                usage_seen = true;
                let u = &v["usage"];
                if !(is_count(&u["prompt_tokens"])
                    && is_count(&u["completion_tokens"])
                    && is_count(&u["total_tokens"])
                    && is_count(&u["prompt_tokens_details"]["cached_tokens"]))
                {
                    r.problem("the usage chunk lacks its token counts");
                }
                match u["completion_tokens_details"]["reasoning_tokens"].as_u64() {
                    Some(n) => r.reasoning_tokens = Some(n),
                    None => r.problem("the usage chunk lacks reasoning_tokens"),
                }
            } else if let Some(n) =
                v["usage"]["completion_tokens_details"]["reasoning_tokens"].as_u64()
            {
                r.reasoning_tokens = Some(n);
            }
            continue;
        };
        let d = &choice["delta"];
        if first {
            first = false;
            if d["role"] != "assistant" || d["content"] != "" {
                r.problem("the first chunk is not the role chunk");
            }
        }
        if let Some(s) = d["content"].as_str() {
            r.text.push_str(s);
        }
        if let Some(s) = d["reasoning_content"].as_str() {
            r.reasoning.push_str(s);
            r.reasoning_present = true;
        }
        for t in d["tool_calls"].as_array().map_or(&[][..], Vec::as_slice) {
            let call = calls.entry(t["index"].as_u64().unwrap_or(0)).or_default();
            if let Some(i) = t["id"].as_str() {
                call.id = i.to_string();
            }
            if let Some(n) = t["function"]["name"].as_str() {
                call.name = n.to_string();
            }
            if let Some(a) = t["function"]["arguments"].as_str() {
                call.arguments.push_str(a);
            }
        }
        if let Some(f) = choice["finish_reason"].as_str() {
            r.finish = f.to_string();
        }
    }
    r.calls = calls
        .into_values()
        .map(|c| Call::new(&c.id, &c.name, c.arguments))
        .collect();
    if r.calls.iter().any(|c| c.args.is_none()) {
        r.problem("a call's arguments are not JSON text");
    }
    if events.is_empty() {
        r.problem("the stream had no events");
    }
    if !done {
        r.problem("the stream did not end in [DONE]");
    }
    if r.finish.is_empty() {
        r.problem("no chunk carried a finish_reason");
    }
    if want_usage && !usage_seen {
        r.problem("the stream had no usage chunk, though it was asked for");
    }
    r.ended = done;
    r
}

/// An Anthropic `message` body (non-streaming).
pub fn anthropic_reply(status: u16, body: &Value) -> Reply {
    if status != 200 {
        return Reply::failed(Api::Anthropic, status, body);
    }
    let mut r = Reply::new(Api::Anthropic, status);
    r.id = body["id"].as_str().unwrap_or("").to_string();
    if !r.id.starts_with("msg_") {
        r.problem("the id is not msg_…");
    }
    if body["type"] != "message" || body["role"] != "assistant" || !body["model"].is_string() {
        r.problem("type, role or model is wrong");
    }
    match body["content"].as_array() {
        None => r.problem("content is not a list of blocks"),
        Some(blocks) => {
            for b in blocks {
                match b["type"].as_str() {
                    Some("thinking") => {
                        r.reasoning.push_str(b["thinking"].as_str().unwrap_or(""));
                        r.reasoning_present = true;
                    }
                    Some("text") => r.text.push_str(b["text"].as_str().unwrap_or("")),
                    Some("tool_use") => {
                        let call = Call::new(
                            b["id"].as_str().unwrap_or(""),
                            b["name"].as_str().unwrap_or(""),
                            b["input"].to_string(),
                        );
                        if !call.id.starts_with("toolu_") || !b["input"].is_object() {
                            r.problem("a tool_use block has no toolu_ id or no input object");
                        }
                        r.calls.push(call);
                    }
                    _ => r.problem("a block of an unknown type"),
                }
            }
        }
    }
    match body["stop_reason"].as_str() {
        Some(s) => r.finish = s.to_string(),
        None => r.problem("stop_reason is not a string"),
    }
    let u = &body["usage"];
    if !(is_count(&u["input_tokens"])
        && is_count(&u["output_tokens"])
        && is_count(&u["cache_read_input_tokens"])
        && is_count(&u["cache_creation_input_tokens"]))
    {
        r.problem("usage lacks its token counts");
    }
    r.ended = true;
    r
}

/// An Anthropic stream's events: `message_start`, blocks that each close before the next opens, then
/// `message_delta` and `message_stop`. The API puts the stream's usage in two places, both judged: `input_tokens` in
/// `message_start` and `output_tokens` in `message_delta`. It defines no cache counts for a stream, so none are judged.
pub fn anthropic_stream(status: u16, events: &[Ev]) -> Reply {
    let mut r = Reply::new(Api::Anthropic, status);
    let mut stopped = false;
    // The open block: its index and its type.
    let mut open: Option<(u64, String)> = None;
    let mut calls: Vec<Call> = Vec::new();
    for (i, ev) in events.iter().enumerate() {
        let Ok(v) = serde_json::from_str::<Value>(&ev.data) else {
            r.problem("an event is not JSON");
            continue;
        };
        if ev.name.as_deref() != v["type"].as_str() {
            r.problem("an event's name and type differ");
        }
        match v["type"].as_str().unwrap_or("") {
            "ping" => {}
            "message_start" => {
                if i != 0 {
                    r.problem("message_start is not first");
                }
                r.id = v["message"]["id"].as_str().unwrap_or("").to_string();
                if !r.id.starts_with("msg_") {
                    r.problem("the id is not msg_…");
                }
                if !is_count(&v["message"]["usage"]["input_tokens"]) {
                    r.problem("message_start's usage has no input_tokens");
                }
            }
            "content_block_start" => {
                if open.is_some() {
                    r.problem("a block opened inside another");
                }
                let b = &v["content_block"];
                let kind = b["type"].as_str().unwrap_or("").to_string();
                match kind.as_str() {
                    "thinking" => r.reasoning_present = true,
                    "tool_use" => calls.push(Call {
                        id: b["id"].as_str().unwrap_or("").to_string(),
                        name: b["name"].as_str().unwrap_or("").to_string(),
                        ..Call::default()
                    }),
                    "text" => {}
                    _ => r.problem("a block of an unknown type"),
                }
                open = Some((v["index"].as_u64().unwrap_or(u64::MAX), kind));
            }
            "content_block_delta" => {
                let d = &v["delta"];
                match &open {
                    Some((idx, _)) if v["index"].as_u64() == Some(*idx) => {}
                    _ => r.problem("a delta outside its block"),
                }
                match d["type"].as_str().unwrap_or("") {
                    "thinking_delta" => r.reasoning.push_str(d["thinking"].as_str().unwrap_or("")),
                    "text_delta" => r.text.push_str(d["text"].as_str().unwrap_or("")),
                    "input_json_delta" => {
                        if let Some(c) = calls.last_mut() {
                            c.arguments
                                .push_str(d["partial_json"].as_str().unwrap_or(""));
                        }
                    }
                    "signature_delta" => {}
                    _ => r.problem("a delta of an unknown type"),
                }
            }
            "content_block_stop" => {
                if open.take().is_none() {
                    r.problem("a block stopped that was never opened");
                }
            }
            "message_delta" => {
                if open.is_some() {
                    r.problem("message_delta with a block open");
                }
                if let Some(s) = v["delta"]["stop_reason"].as_str() {
                    r.finish = s.to_string();
                }
                if !is_count(&v["usage"]["output_tokens"]) {
                    r.problem("message_delta's usage has no output_tokens");
                }
            }
            "message_stop" => stopped = true,
            "error" => r.problem("an in-band error event"),
            _ => r.problem("an event of an unknown type"),
        }
    }
    r.calls = calls
        .into_iter()
        .map(|c| Call::new(&c.id, &c.name, c.arguments))
        .collect();
    if r.calls.iter().any(|c| c.args.is_none()) {
        r.problem("a call's input is not JSON");
    }
    if events.first().and_then(|e| e.name.as_deref()) != Some("message_start") {
        r.problem("the stream did not start with message_start");
    }
    if !stopped {
        r.problem("the stream did not end in message_stop");
    }
    if r.finish.is_empty() {
        r.problem("no message_delta carried a stop_reason");
    }
    r.ended = stopped;
    r
}

/// What a step of `requests` must show.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Want {
    /// A 200 of the right shape with non-empty text.
    Text,
    /// `Text`, and no reasoning: `reasoning_effort: "none"` or `thinking: {type: "disabled"}`.
    TextNoReasoning,
    /// One `read_file` call with a string `path`, and the finish reason of a call.
    Call,
    /// The reply to a tool result: text, and `stop` or `end_turn`.
    Echo,
}

/// The pass rule of one `requests` step, as one check named `name`.
pub fn judge_reply(name: &str, r: &Reply, want: Want) -> Check {
    let mut why = r.problems.clone();
    let (call_finish, stop_finish) = match r.api {
        Api::OpenAi => ("tool_calls", "stop"),
        Api::Anthropic => ("tool_use", "end_turn"),
    };
    let has_text = !r.text.trim().is_empty();
    if r.status == 200 {
        match want {
            Want::Text | Want::TextNoReasoning => {
                if !has_text {
                    why.push(format!("no text (finish {})", r.finish));
                }
            }
            Want::Call => {
                if r.calls.len() != 1 {
                    why.push(format!("{} tool calls, not one", r.calls.len()));
                } else if r.calls[0].name != "read_file" {
                    why.push("the call is not read_file".into());
                } else if r.calls[0].path().is_none() {
                    why.push("the call's path is not a string".into());
                }
                if r.finish != call_finish {
                    why.push(format!("finish {:?}, not {call_finish}", r.finish));
                }
            }
            Want::Echo => {
                if !has_text {
                    why.push(format!("no text (finish {})", r.finish));
                }
                if r.finish != stop_finish {
                    why.push(format!("finish {:?}, not {stop_finish}", r.finish));
                }
                if !r.calls.is_empty() {
                    why.push("the reply to a tool result called a tool".into());
                }
            }
        }
        if want == Want::TextNoReasoning {
            if r.reasoning_present || !r.reasoning.is_empty() {
                why.push("the reply carried reasoning".into());
            }
            if r.reasoning_tokens.unwrap_or(0) != 0 {
                why.push(format!(
                    "reasoning_tokens is {}",
                    r.reasoning_tokens.unwrap_or(0)
                ));
            }
            // No think tag leaks into the text.
            if r.text.contains("<think>") || r.text.contains("</think>") {
                why.push("the text holds a think tag".into());
            }
        }
    }
    let detail = if why.is_empty() {
        format!(
            "ok: finish {}, {} chars of text, {} calls",
            r.finish,
            r.text.chars().count(),
            r.calls.len()
        )
    } else {
        why.join("; ")
    };
    Check::new(name, why.is_empty(), detail)
}

/// The log-side rule of a no-reasoning step (`openai_effort_none`, `anthropic_thinking_disabled`): the request's log
/// line reports `reasoning_tokens` 0. For Anthropic it is the only token-level evidence, since the reply has no count.
pub fn judge_log_reasoning(name: &str, logged: Option<u64>) -> Check {
    match logged {
        Some(0) => Check::new(name, true, "the log reports 0 reasoning tokens"),
        Some(n) => Check::new(name, false, format!("the log reports {n} reasoning tokens")),
        None => Check::new(name, false, "the log line has no reasoning_tokens"),
    }
}

/// What a turn of the `reuse` case left in its request's log line: the cut and what the engine reused of the prompt.
#[derive(Clone, Debug, PartialEq)]
pub struct Turn {
    pub cut_at: u32,
    /// The turn before the cut is the user's query.
    pub user_query: bool,
    pub reused: u32,
    /// Where the restored state came from (`none`, `live`, `interval`, `prompt_end`, `user_query`, `reply_end`).
    pub restored: String,
    pub restore_bytes: u64,
}

/// One floor: the turn reused at least `floor` tokens (`what` names where the floor comes from).
pub fn floor(name: &str, reused: u32, floor: u32, what: &str) -> Check {
    Check::new(
        name,
        reused >= floor,
        format!(
            "reused {reused} {} {what} {floor}",
            if reused >= floor { ">=" } else { "<" }
        ),
    )
}

/// The `reuse` case's floors, over its six turns in order: `t1`, `t2`, `t3` (a tool loop), `t4` (a new user
/// turn), `a5` (after conversation B and a side call) and `a6` (after the unload and the load).
/// - (a) `t2.reused >= t1.cut_at` and `t3.reused >= t2.cut_at`;
/// - (b) `t1.user_query`, and `t4.reused >= t1.cut_at`;
/// - (c) `a5.reused >= t4.cut_at`, and the restore copied KV back: `a5.restore_bytes > 0` with `a5.restored` neither
///   `none` nor `live`;
/// - resume: `a6.reused >= a5.cut_at`.
pub fn floors_a_b_c(turns: &[Turn]) -> Vec<Check> {
    let [t1, t2, t3, t4, a5, a6] = turns else {
        return vec![Check::new(
            "turns",
            false,
            format!("{} turns, not 6", turns.len()),
        )];
    };
    vec![
        floor("a_t2_reused", t2.reused, t1.cut_at, "t1.cut_at"),
        floor("a_t3_reused", t3.reused, t2.cut_at, "t2.cut_at"),
        Check::new(
            "b_t1_user_query",
            t1.user_query,
            format!("t1.user_query is {}", t1.user_query),
        ),
        floor("b_t4_reused", t4.reused, t1.cut_at, "t1.cut_at"),
        floor("c_a5_reused", a5.reused, t4.cut_at, "t4.cut_at"),
        Check::new(
            "c_a5_restores_kv",
            a5.restore_bytes > 0 && !matches!(a5.restored.as_str(), "none" | "live"),
            format!(
                "restored {}, {} restore bytes",
                a5.restored, a5.restore_bytes
            ),
        ),
        floor("resume_a6_reused", a6.reused, a5.cut_at, "a5.cut_at"),
    ]
}

/// `budget`: with `reasoning_effort: low` (budget 1,024) the reasoning is cut at the budget and closed by the forced
/// `\n</think>\n\n`, so `reasoning` counts from 1,024 to 1,024 + `close_len`, and the run did not hit `max_tokens`
/// (`finish` is not `length`). A run that closes its reasoning earlier fails with its count: the author picks a
/// harder prompt rather than weakening this.
pub fn budget_check(reasoning: u32, close_len: u32, finish: &str) -> Check {
    let (lo, hi) = (1024, 1024 + close_len);
    let in_range = (lo..=hi).contains(&reasoning);
    let mut why = Vec::new();
    if !in_range {
        why.push(format!(
            "{reasoning} reasoning tokens, outside {lo} to {hi}"
        ));
    }
    if finish == "length" {
        why.push("finish_reason is length".to_string());
    }
    let detail = if why.is_empty() {
        format!("{reasoning} reasoning tokens, within {lo} to {hi}; finish {finish}")
    } else {
        why.join("; ")
    };
    Check::new("reasoning_budget", why.is_empty(), detail)
}

/// `idle` and `unload-stream`: the VRAM after a reload is within 64 MiB of the first load's.
pub fn vram_check(first: f64, later: f64) -> Check {
    let diff = (later - first).abs();
    Check::new(
        "vram_within_64_mib",
        diff <= 64.0,
        format!("{first:.1} MiB then {later:.1} MiB, a difference of {diff:.1} (limit 64)"),
    )
}

/// `vram_check` over the load lines' figures, which are `null` from the fake. No figure fails, unless the caller allows
/// it: only the fake-backed tests do, never the command.
pub fn vram_figures(first: Option<f64>, later: Option<f64>, allow_missing: bool) -> Check {
    match (first, later) {
        (Some(a), Some(b)) => vram_check(a, b),
        _ => Check::new(
            "vram_within_64_mib",
            allow_missing,
            "no figure: a load line has no vram_used_mib",
        ),
    }
}

/// `idle`: the cache holds the same slots per layer after the reload. No figure fails as `vram_figures` does.
pub fn slots_check(first: Option<u64>, later: Option<u64>, allow_missing: bool) -> Check {
    match (first, later) {
        (Some(a), Some(b)) => Check::new(
            "slots_equal",
            a == b,
            format!("{a} slots per layer, then {b}"),
        ),
        _ => Check::new(
            "slots_equal",
            allow_missing,
            "no figure: a load line has no slots_per_layer",
        ),
    }
}

/// `queue` and `idle`: `t` lies in `lo..=hi`.
pub fn within(t: Duration, lo: Duration, hi: Duration) -> Check {
    Check::new(
        "within",
        (lo..=hi).contains(&t),
        format!(
            "{:.2} s, window {:.2} to {:.2} s",
            t.as_secs_f64(),
            lo.as_secs_f64(),
            hi.as_secs_f64()
        ),
    )
}

/// The pool the reuse floors were measured at, over every `load` line of the server's log (a block loads more
/// than once, and each load's pool can be cut differently): if any `pool_limit_gib` is below 2, the block's reuse
/// verdicts are inconclusive. That is a failing check with this exact detail, which the runner reads as
/// stop-and-report, never a pass. A line with no figure is left out of the minimum; when none has one the check
/// fails too, unless the caller allows it (the fake-backed tests).
pub fn pool_check(limits: &[Option<f64>], allow_missing: bool) -> Check {
    let smallest = limits.iter().flatten().copied().reduce(f64::min);
    match smallest {
        Some(g) if g < 2.0 => Check::new(
            "pool_at_least_2_gib",
            false,
            "inconclusive: pool below 2 GiB",
        ),
        Some(g) => Check::new(
            "pool_at_least_2_gib",
            true,
            format!("the smallest pool of {} loads is {g:.2} GiB", limits.len()),
        ),
        None => Check::new(
            "pool_at_least_2_gib",
            allow_missing,
            "no figure: no load line has pool_limit_gib",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn health(status: &str) -> Value {
        json!({"service": "artemis", "status": status, "version": "0", "loaded": status == "ready",
            "model": "m", "max_context": 1, "api_key": true})
    }

    fn poll(status: u16, name: &str) -> Poll {
        Poll {
            status,
            body: health(name),
        }
    }

    fn pass(c: &[Check]) -> Vec<(&str, bool)> {
        c.iter().map(|c| (c.name.as_str(), c.pass)).collect()
    }

    #[test]
    fn load_needs_a_loading_poll_with_the_seven_keys_and_a_ready_last_poll() {
        let good = [poll(200, "idle"), poll(503, "loading"), poll(200, "ready")];
        let c = judge_load(200, &good);
        assert!(c.iter().all(|c| c.pass), "{c:?}");
        // Already loaded: no loading poll.
        let c = judge_load(200, &[poll(200, "ready"), poll(200, "ready")]);
        assert_eq!(
            pass(&c),
            [
                ("chat_answers", true),
                ("loading_seen", false),
                ("ready_last", true)
            ]
        );
        // A 503 that isn't `loading`, and a loading body missing a key.
        let mut short = health("loading");
        short.as_object_mut().unwrap().remove("api_key");
        let c = judge_load(
            200,
            &[
                poll(503, "recovering"),
                Poll {
                    status: 503,
                    body: short,
                },
                poll(200, "ready"),
            ],
        );
        assert!(!c[1].pass, "{c:?}");
        // An extra key is not "the same seven keys".
        let mut extra = health("loading");
        extra["more"] = json!(1);
        assert!(
            !judge_load(
                200,
                &[
                    Poll {
                        status: 503,
                        body: extra
                    },
                    poll(200, "ready")
                ]
            )[1]
            .pass
        );
        // The last poll must be ready; the chat must answer 200; no polls at all fails.
        assert!(!judge_load(200, &[poll(503, "loading")])[2].pass);
        assert!(!judge_load(500, &good)[0].pass);
        let c = judge_load(200, &[]);
        assert_eq!(
            pass(&c),
            [
                ("chat_answers", true),
                ("loading_seen", false),
                ("ready_last", false)
            ]
        );
    }

    fn oa_call() -> Value {
        json!({"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "logprobs": null,
                "message": {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "read_file", "arguments": "{\"path\":\"notes.txt\"}"}}]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0}}})
    }

    fn oa_text(text: &str) -> Value {
        let mut v = oa_call();
        v["choices"][0]["message"] = json!({"role": "assistant", "content": text});
        v["choices"][0]["finish_reason"] = json!("stop");
        v
    }

    #[test]
    fn an_openai_call_turn_is_one_read_file_call_with_a_string_path() {
        let r = openai_reply(200, &oa_call());
        let c = judge_reply("call", &r, Want::Call);
        assert!(c.pass, "{c:?}");
        assert_eq!(r.calls[0].path(), Some("notes.txt"));
        // Text where the call should be.
        let c = judge_reply("call", &openai_reply(200, &oa_text("no")), Want::Call);
        assert!(!c.pass && c.detail.contains("0 tool calls"), "{c:?}");
        // Two calls, another tool, a path that isn't a string, the wrong finish.
        let mut two = oa_call();
        let one = two["choices"][0]["message"]["tool_calls"][0].clone();
        two["choices"][0]["message"]["tool_calls"] = json!([one.clone(), one]);
        assert!(!judge_reply("c", &openai_reply(200, &two), Want::Call).pass);
        let mut other = oa_call();
        other["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = json!("write_file");
        assert!(!judge_reply("c", &openai_reply(200, &other), Want::Call).pass);
        let mut num = oa_call();
        num["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] =
            json!("{\"path\":3}");
        assert!(!judge_reply("c", &openai_reply(200, &num), Want::Call).pass);
        let mut stop = oa_call();
        stop["choices"][0]["finish_reason"] = json!("stop");
        assert!(!judge_reply("c", &openai_reply(200, &stop), Want::Call).pass);
        // Arguments that are not JSON text.
        let mut bad = oa_call();
        bad["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json!("{");
        assert!(!judge_reply("c", &openai_reply(200, &bad), Want::Call).pass);
    }

    #[test]
    fn an_openai_echo_is_text_with_stop() {
        assert!(judge_reply("e", &openai_reply(200, &oa_text("It says hi.")), Want::Echo).pass);
        assert!(!judge_reply("e", &openai_reply(200, &oa_text("  ")), Want::Echo).pass);
        let mut length = oa_text("cut");
        length["choices"][0]["finish_reason"] = json!("length");
        assert!(!judge_reply("e", &openai_reply(200, &length), Want::Echo).pass);
        // The reply to a tool result may not call a tool.
        assert!(!judge_reply("e", &openai_reply(200, &oa_call()), Want::Echo).pass);
    }

    #[test]
    fn an_openai_body_must_have_the_spec_shape() {
        let ok = openai_reply(200, &oa_text("x"));
        assert!(ok.problems.is_empty(), "{:?}", ok.problems);
        let mut two = oa_text("x");
        let c = two["choices"][0].clone();
        two["choices"] = json!([c.clone(), c]);
        assert!(!openai_reply(200, &two).problems.is_empty());
        let mut nocontent = oa_text("x");
        nocontent["choices"][0]["message"]["content"] = Value::Null;
        assert!(!openai_reply(200, &nocontent).problems.is_empty());
        let mut noreason = oa_text("x");
        noreason["usage"]["completion_tokens_details"] = json!({});
        assert!(!openai_reply(200, &noreason).problems.is_empty());
        let mut id = oa_text("x");
        id["id"] = json!("msg_1");
        assert!(!openai_reply(200, &id).problems.is_empty());
        // Not 200: the error message, cut to 160 characters, is the problem.
        let long = "e".repeat(400);
        let err = openai_reply(503, &json!({"error": {"message": long}}));
        let c = judge_reply("x", &err, Want::Text);
        assert!(!c.pass && c.detail.len() < 220, "{c:?}");
    }

    #[test]
    fn no_reasoning_means_no_field_and_no_reasoning_tokens() {
        let none = |v: &Value| judge_reply("n", &openai_reply(200, v), Want::TextNoReasoning);
        assert!(none(&oa_text("x")).pass);
        let mut field = oa_text("x");
        field["choices"][0]["message"]["reasoning_content"] = json!("hmm");
        assert!(!none(&field).pass);
        let mut tokens = oa_text("x");
        tokens["usage"]["completion_tokens_details"]["reasoning_tokens"] = json!(3);
        assert!(!none(&tokens).pass);
        // The server omits the field when there is no reasoning, so an empty `reasoning_content` is still the field.
        let mut empty = oa_text("x");
        empty["choices"][0]["message"]["reasoning_content"] = json!("");
        assert!(!none(&empty).pass);
        // A think tag in the text is a leak, whichever side of the reasoning it names.
        for leak in [
            "<think>
hm",
            "ok
</think>

Paris.",
        ] {
            let c = none(&oa_text(leak));
            assert!(!c.pass && c.detail.contains("think tag"), "{c:?}");
        }
        // The same for Anthropic; a stream's `[DONE]`-less tags aside, text with the word `think` is fine.
        let mut a = an_text("<think>x</think>");
        assert!(!judge_reply("d", &anthropic_reply(200, &a), Want::TextNoReasoning).pass);
        a = an_text("I think so.");
        assert!(judge_reply("d", &anthropic_reply(200, &a), Want::TextNoReasoning).pass);
        // Text with reasoning beside it is fine for `Text`.
        assert!(judge_reply("t", &openai_reply(200, &field), Want::Text).pass);
    }

    fn ev(name: Option<&str>, data: Value) -> Ev {
        Ev {
            name: name.map(String::from),
            data: data.to_string(),
        }
    }

    fn chunk(delta: Value, finish: Value) -> Ev {
        ev(
            None,
            json!({"id": "chatcmpl-2", "object": "chat.completion.chunk", "created": 1, "model": "m",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}),
        )
    }

    fn done() -> Ev {
        Ev {
            name: None,
            data: "[DONE]".into(),
        }
    }

    fn oa_stream() -> Vec<Ev> {
        vec![
            chunk(json!({"role": "assistant", "content": ""}), Value::Null),
            chunk(json!({"reasoning_content": "think"}), Value::Null),
            chunk(json!({"content": "Par"}), Value::Null),
            chunk(json!({"content": "is."}), Value::Null),
            chunk(json!({}), json!("stop")),
            done(),
        ]
    }

    #[test]
    fn an_openai_stream_joins_its_deltas_and_must_end_in_done() {
        let r = openai_stream(200, &oa_stream(), false);
        assert!(r.problems.is_empty(), "{:?}", r.problems);
        assert_eq!(
            (r.text.as_str(), r.reasoning.as_str(), r.finish.as_str()),
            ("Paris.", "think", "stop")
        );
        assert!(r.ended && r.reasoning_present);
        assert!(judge_reply("s", &r, Want::Text).pass);
        // Cut before [DONE].
        let cut = openai_stream(200, &oa_stream()[..5], false);
        assert!(!cut.ended && !judge_reply("s", &cut, Want::Text).pass);
        // No role chunk first.
        let mut late = oa_stream();
        late.remove(0);
        assert!(!openai_stream(200, &late, false).problems.is_empty());
        // An in-band error, and no events at all.
        let mut err = oa_stream();
        err.insert(2, ev(None, json!({"error": {"message": "unloaded"}})));
        assert!(!openai_stream(200, &err, false).problems.is_empty());
        assert!(!openai_stream(200, &[], false).problems.is_empty());
        // A usage chunk (`choices: []`), when the request asked for one, is judged like the body's usage.
        let usage_chunk = |u: Value| {
            ev(
                None,
                json!({"id": "chatcmpl-2", "object": "chat.completion.chunk", "choices": [], "usage": u}),
            )
        };
        let good = json!({"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 12}});
        let mut usage = oa_stream();
        usage.insert(5, usage_chunk(good.clone()));
        let r = openai_stream(200, &usage, true);
        assert!(r.problems.is_empty(), "{:?}", r.problems);
        assert_eq!(r.reasoning_tokens, Some(12));
        // Asked for and missing; present but short of counts or of reasoning_tokens.
        assert!(!openai_stream(200, &oa_stream(), true).problems.is_empty());
        let mut short = oa_stream();
        short.insert(5, usage_chunk(json!({"prompt_tokens": 5})));
        assert!(!openai_stream(200, &short, true).problems.is_empty());
        let mut no_reasoning = good.clone();
        no_reasoning["completion_tokens_details"] = json!({});
        let mut nr = oa_stream();
        nr.insert(5, usage_chunk(no_reasoning));
        assert!(!openai_stream(200, &nr, true).problems.is_empty());
        // Not asked for: a usage chunk is not judged.
        assert!(openai_stream(200, &usage, false).problems.is_empty());
    }

    #[test]
    fn an_openai_stream_assembles_a_call_from_its_fragments() {
        let events = vec![
            chunk(json!({"role": "assistant", "content": ""}), Value::Null),
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "call_9", "type": "function",
                    "function": {"name": "read_file", "arguments": ""}}]}),
                Value::Null,
            ),
            chunk(
                json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"path\":"}}]}),
                Value::Null,
            ),
            chunk(
                json!({"tool_calls": [{"index": 0, "function": {"arguments": "\"a.txt\"}"}}]}),
                Value::Null,
            ),
            chunk(json!({}), json!("tool_calls")),
            done(),
        ];
        let r = openai_stream(200, &events, false);
        assert!(r.problems.is_empty(), "{:?}", r.problems);
        assert_eq!(r.calls[0].id, "call_9");
        assert!(judge_reply("c", &r, Want::Call).pass);
    }

    fn an_text(text: &str) -> Value {
        json!({"id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                "output_tokens": 3}})
    }

    fn an_call() -> Value {
        let mut v = an_text("");
        v["content"] = json!([{"type": "tool_use", "id": "toolu_1", "name": "read_file",
            "input": {"path": "notes.txt"}}]);
        v["stop_reason"] = json!("tool_use");
        v
    }

    #[test]
    fn an_anthropic_reply_is_judged_by_its_blocks() {
        assert!(judge_reply("c", &anthropic_reply(200, &an_call()), Want::Call).pass);
        assert!(!judge_reply("c", &anthropic_reply(200, &an_text("no")), Want::Call).pass);
        assert!(
            judge_reply(
                "e",
                &anthropic_reply(200, &an_text("It says hi.")),
                Want::Echo
            )
            .pass
        );
        let mut tool_use = an_text("x");
        tool_use["stop_reason"] = json!("tool_use");
        assert!(!judge_reply("e", &anthropic_reply(200, &tool_use), Want::Echo).pass);
        // A thinking block fails `disabled`, not a plain text request.
        let mut thinking = an_text("x");
        thinking["content"] = json!([{"type": "thinking", "thinking": "", "signature": ""},
            {"type": "text", "text": "x"}]);
        let r = anthropic_reply(200, &thinking);
        assert!(r.problems.is_empty(), "{:?}", r.problems);
        assert!(judge_reply("t", &r, Want::Text).pass);
        assert!(!judge_reply("d", &r, Want::TextNoReasoning).pass);
        assert!(
            judge_reply(
                "d",
                &anthropic_reply(200, &an_text("x")),
                Want::TextNoReasoning
            )
            .pass
        );
        // Shape: usage keys, the id's prefix.
        let mut usage = an_text("x");
        usage["usage"]
            .as_object_mut()
            .unwrap()
            .remove("cache_read_input_tokens");
        assert!(!anthropic_reply(200, &usage).problems.is_empty());
        let mut id = an_text("x");
        id["id"] = json!("chatcmpl-1");
        assert!(!anthropic_reply(200, &id).problems.is_empty());
    }

    fn an(name: &str, data: Value) -> Ev {
        let mut d = data;
        d["type"] = json!(name);
        ev(Some(name), d)
    }

    fn an_stream() -> Vec<Ev> {
        vec![
            an(
                "message_start",
                json!({"message": {"id": "msg_2", "usage": {"input_tokens": 3}}}),
            ),
            an(
                "content_block_start",
                json!({"index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            ),
            an(
                "content_block_delta",
                json!({"index": 0, "delta": {"type": "thinking_delta", "thinking": "hm"}}),
            ),
            an(
                "content_block_delta",
                json!({"index": 0, "delta": {"type": "signature_delta", "signature": ""}}),
            ),
            an("content_block_stop", json!({"index": 0})),
            an(
                "content_block_start",
                json!({"index": 1, "content_block": {"type": "text", "text": ""}}),
            ),
            an("ping", json!({})),
            an(
                "content_block_delta",
                json!({"index": 1, "delta": {"type": "text_delta", "text": "Par"}}),
            ),
            an(
                "content_block_delta",
                json!({"index": 1, "delta": {"type": "text_delta", "text": "is."}}),
            ),
            an("content_block_stop", json!({"index": 1})),
            an(
                "message_delta",
                json!({"delta": {"stop_reason": "end_turn", "stop_sequence": null},
                "usage": {"output_tokens": 4}}),
            ),
            an("message_stop", json!({})),
        ]
    }

    #[test]
    fn an_anthropic_stream_closes_each_block_before_the_next_and_ends_in_message_stop() {
        let r = anthropic_stream(200, &an_stream());
        assert!(r.problems.is_empty(), "{:?}", r.problems);
        assert_eq!(
            (r.text.as_str(), r.reasoning.as_str(), r.finish.as_str()),
            ("Paris.", "hm", "end_turn")
        );
        assert!(judge_reply("s", &r, Want::Text).pass);
        // Cut before message_stop.
        let cut = anthropic_stream(200, &an_stream()[..11]);
        assert!(!cut.ended && !judge_reply("s", &cut, Want::Text).pass);
        // A block opened inside another: the thinking block's stop is missing.
        let mut nested = an_stream();
        nested.remove(4);
        assert!(!anthropic_stream(200, &nested).problems.is_empty());
        // message_start not first; an in-band error.
        let mut late = an_stream();
        late.swap(0, 1);
        assert!(!anthropic_stream(200, &late).problems.is_empty());
        let mut err = an_stream();
        err.insert(
            6,
            an("error", json!({"error": {"type": "overloaded_error"}})),
        );
        assert!(!anthropic_stream(200, &err).problems.is_empty());
        // An event whose name and type differ.
        let mut name = an_stream();
        name[6] = ev(Some("pong"), json!({"type": "ping"}));
        assert!(!anthropic_stream(200, &name).problems.is_empty());
        // The usage the API puts in a stream: `input_tokens` in message_start, `output_tokens` in message_delta.
        let mut no_in = an_stream();
        no_in[0] = an(
            "message_start",
            json!({"message": {"id": "msg_2", "usage": {}}}),
        );
        assert!(!anthropic_stream(200, &no_in).problems.is_empty());
        let mut no_out = an_stream();
        no_out[10] = an(
            "message_delta",
            json!({"delta": {"stop_reason": "end_turn"}}),
        );
        assert!(!anthropic_stream(200, &no_out).problems.is_empty());
    }

    #[test]
    fn an_anthropic_stream_assembles_a_tool_use_from_its_json_fragments() {
        let events = vec![
            an(
                "message_start",
                json!({"message": {"id": "msg_3", "usage": {"input_tokens": 8}}}),
            ),
            an(
                "content_block_start",
                json!({"index": 0, "content_block":
                {"type": "tool_use", "id": "toolu_5", "name": "read_file", "input": {}}}),
            ),
            an(
                "content_block_delta",
                json!({"index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"path\":"}}),
            ),
            an(
                "content_block_delta",
                json!({"index": 0, "delta": {"type": "input_json_delta", "partial_json": "\"b.txt\"}"}}),
            ),
            an("content_block_stop", json!({"index": 0})),
            an(
                "message_delta",
                json!({"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
            ),
            an("message_stop", json!({})),
        ];
        let r = anthropic_stream(200, &events);
        assert!(r.problems.is_empty(), "{:?}", r.problems);
        assert_eq!(r.calls[0].path(), Some("b.txt"));
        assert!(judge_reply("c", &r, Want::Call).pass);
    }

    #[test]
    fn the_log_must_report_zero_reasoning_tokens_for_a_no_reasoning_step() {
        assert!(judge_log_reasoning("n", Some(0)).pass);
        assert!(!judge_log_reasoning("n", Some(3)).pass);
        assert!(!judge_log_reasoning("n", None).pass);
    }

    #[test]
    fn a_status_other_than_200_fails_whatever_is_wanted() {
        for api in [Api::OpenAi, Api::Anthropic] {
            let r = Reply::failed(api, 503, &json!({"error": {"message": "timed out"}}));
            for want in [Want::Text, Want::TextNoReasoning, Want::Call, Want::Echo] {
                let c = judge_reply("x", &r, want);
                assert!(!c.pass && c.detail.contains("503"), "{c:?}");
            }
        }
    }

    /// Six turns of the `reuse` case: a1..a3 of the tool loop, the new user turn, the turn after conversation B and the
    /// side call, and the turn after the unload and the load. Each `reused` is its floor's `cut_at`, so every floor
    /// holds at equality.
    fn turns() -> Vec<Turn> {
        let t = |cut_at, reused, restored: &str, restore_bytes| Turn {
            cut_at,
            user_query: true,
            reused,
            restored: restored.into(),
            restore_bytes,
        };
        vec![
            t(3000, 0, "none", 0),
            t(3100, 3000, "prompt_end", 0),
            t(3200, 3100, "live", 0),
            t(3400, 3000, "user_query", 0),
            t(3500, 3400, "user_query", 63_000_000),
            t(3600, 3500, "prompt_end", 63_000_000),
        ]
    }

    fn failed(c: &[Check]) -> Vec<&str> {
        c.iter()
            .filter(|c| !c.pass)
            .map(|c| c.name.as_str())
            .collect()
    }

    #[test]
    fn floors_hold_at_equality_and_fail_one_below() {
        let c = floors_a_b_c(&turns());
        assert_eq!(failed(&c), Vec::<&str>::new(), "{c:?}");
        assert_eq!(
            c.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            [
                "a_t2_reused",
                "a_t3_reused",
                "b_t1_user_query",
                "b_t4_reused",
                "c_a5_reused",
                "c_a5_restores_kv",
                "resume_a6_reused"
            ]
        );
        // One token below each floor fails that floor and no other: (a) twice, (b), (c) and the resume.
        for (turn, name) in [
            (1, "a_t2_reused"),
            (2, "a_t3_reused"),
            (3, "b_t4_reused"),
            (4, "c_a5_reused"),
            (5, "resume_a6_reused"),
        ] {
            let mut t = turns();
            t[turn].reused -= 1;
            assert_eq!(failed(&floors_a_b_c(&t)), [name], "turn {turn}");
        }
        // The user-query flag is (b)'s own condition.
        let mut t = turns();
        t[0].user_query = false;
        assert_eq!(failed(&floors_a_b_c(&t)), ["b_t1_user_query"]);
        // A wrong number of turns is one failing check, not a panic.
        let c = floors_a_b_c(&turns()[..5]);
        assert_eq!(failed(&c), ["turns"], "{c:?}");
    }

    #[test]
    fn c_needs_a_kv_copy() {
        // No bytes copied back, a restore that was the live state, and no restore at all each fail (c) alone.
        for (what, restored, bytes) in [
            ("no bytes", "user_query", 0),
            ("live", "live", 63_000_000),
            ("none", "none", 63_000_000),
        ] {
            let mut t = turns();
            t[4].restored = restored.into();
            t[4].restore_bytes = bytes;
            assert_eq!(failed(&floors_a_b_c(&t)), ["c_a5_restores_kv"], "{what}");
        }
        // Any other source with bytes counts: a reply-end snapshot, an interval one, a prompt-end one.
        for restored in ["reply_end", "interval", "prompt_end", "user_query"] {
            let mut t = turns();
            t[4].restored = restored.into();
            assert!(floors_a_b_c(&t).iter().all(|c| c.pass), "{restored}");
        }
    }

    #[test]
    fn budget_range_is_1024_to_1024_plus_close() {
        // The close is 4 tokens: 1,024 to 1,028 inclusive.
        for (reasoning, finish, pass) in [
            (1023, "stop", false),
            (1024, "stop", true),
            (1028, "stop", true),
            (1029, "stop", false),
            (1024, "length", false),
            (1026, "length", false),
        ] {
            let c = budget_check(reasoning, 4, finish);
            assert_eq!(c.pass, pass, "{reasoning} {finish}: {c:?}");
        }
        // The detail names the count and the range, for an author who must pick a harder prompt.
        let c = budget_check(700, 4, "stop");
        assert!(
            c.detail.contains("700") && c.detail.contains("1024"),
            "{c:?}"
        );
    }

    #[test]
    fn vram_within_64_mib() {
        assert!(vram_check(1000.0, 1064.0).pass);
        assert!(vram_check(1064.0, 1000.0).pass);
        assert!(vram_check(1000.0, 1000.0).pass);
        assert!(!vram_check(1000.0, 1064.1).pass);
        assert!(!vram_check(1064.1, 1000.0).pass);
    }

    #[test]
    fn within_holds_at_both_edges() {
        let s = Duration::from_secs_f64;
        let (lo, hi) = (s(5.0), s(8.0));
        for (t, pass) in [
            (4.99, false),
            (5.0, true),
            (6.5, true),
            (8.0, true),
            (8.01, false),
        ] {
            assert_eq!(within(s(t), lo, hi).pass, pass, "{t}");
        }
        let c = within(s(9.0), lo, hi);
        assert!(
            c.detail.contains("9.0") && c.detail.contains("5.0"),
            "{c:?}"
        );
    }

    #[test]
    fn a_missing_figure_passes_only_under_the_hook() {
        assert!(vram_figures(Some(100.0), Some(160.0), false).pass);
        assert!(!vram_figures(Some(100.0), Some(170.0), true).pass);
        for (a, b) in [(None, None), (Some(1.0), None), (None, Some(1.0))] {
            let c = vram_figures(a, b, false);
            assert!(!c.pass && c.detail.starts_with("no figure"), "{c:?}");
            assert!(vram_figures(a, b, true).pass);
        }
        assert!(slots_check(Some(128), Some(128), false).pass);
        assert!(!slots_check(Some(128), Some(127), true).pass);
        assert!(!slots_check(None, None, false).pass);
        assert!(slots_check(None, None, true).pass);
    }

    #[test]
    fn a_pool_below_2_gib_is_inconclusive() {
        let c = pool_check(&[Some(2.0)], false);
        assert!(c.pass, "{c:?}");
        assert!(pool_check(&[Some(8.0), Some(2.0)], false).pass);
        let c = pool_check(&[Some(1.9)], false);
        assert!(
            !c.pass && c.detail == "inconclusive: pool below 2 GiB",
            "{c:?}"
        );
        // Every load counts: a first load below 2 makes the verdict inconclusive whatever the reload had, and so does
        // a reload below 2 after a good first load.
        for loads in [[Some(1.5), Some(8.0)], [Some(8.0), Some(1.5)]] {
            let c = pool_check(&loads, true);
            assert!(
                !c.pass && c.detail == "inconclusive: pool below 2 GiB",
                "{c:?}"
            );
        }
        // A line without the figure is left out; none with it is no figure, a failure unless the caller (a
        // fake-backed test) allows it.
        assert!(pool_check(&[None, Some(8.0)], false).pass);
        assert!(!pool_check(&[None, Some(1.0)], true).pass);
        assert!(!pool_check(&[None], false).pass);
        assert!(!pool_check(&[], false).pass);
        assert!(pool_check(&[None], true).pass);
    }

    /// The client refuses a message with a `function_call`; absent, `null` and `[]` are the three ways to have none.
    #[test]
    fn judge_flags_a_function_call_field() {
        let with = |message: Value| json!({"choices": [{"index": 0, "message": message}]});
        let base = json!({"role": "assistant", "content": "x"});
        assert!(function_call_absent(&with(base.clone())));
        for none in [json!(null), json!([])] {
            let mut m = base.clone();
            m["function_call"] = none;
            assert!(function_call_absent(&with(m)));
        }
        for some in [
            json!({"name": "lookup", "arguments": "{}"}),
            json!([{"name": "lookup"}]),
            json!(""),
            json!({}),
            json!(0),
        ] {
            let mut m = base.clone();
            m["function_call"] = some.clone();
            assert!(!function_call_absent(&with(m)), "{some} passed");
        }
        // Any choice counts, not only the first.
        let two = json!({"choices": [{"message": base.clone()},
            {"message": {"role": "assistant", "function_call": {"name": "x"}}}]});
        assert!(!function_call_absent(&two));
        // A body with no choices has nothing to flag (`one_choice_each` is the check that fails it).
        assert!(function_call_absent(&json!({})));
        assert!(function_call_absent(&json!({"choices": []})));
    }
}
