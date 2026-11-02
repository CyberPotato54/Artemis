//! `chat-tools`: a client's non-streaming tool loop. The shape is the one a chat bridge in
//! production uses, none of its names: `POST /v1/chat/completions` without streaming, `reasoning_effort: "low"`,
//! `max_tokens` 4,096, two tools that each take one required string and nothing else, at most 4 requests and 3 tool
//! calls, and each round's calls echoed back as the assistant message with its `tool_calls` and one `role: "tool"`
//! message per call. A tool call on the 4th request, or a 4th call, is the client's error.
//!
//! A detail names counts, lengths and our own tool and key names, never a call's argument, a key or tool name the
//! model chose, or its text.
use super::{body, line_of, CHAT};
use crate::test7::judge::{self, Call};
use crate::test7::{Check, Ctx, Verdict};
use anyhow::Result;
use serde_json::{json, Map, Value};
use std::time::{Duration, Instant};

/// The most requests the loop makes, and the most tool calls it serves.
const MAX_REQUESTS: usize = 4;
const MAX_TOOL_CALLS: usize = 3;
const MAX_OUTPUT: u32 = 4096;
/// The client's deadline over all its calls, and the loop's bound against it: the model is loaded by now.
const LOOP_BOUND: Duration = Duration::from_secs(150);

/// A tool: its one argument, required, a string of 1 to `max` characters.
struct Tool {
    name: &'static str,
    description: &'static str,
    arg: &'static str,
    arg_description: &'static str,
    max: usize,
    /// What the driver answers a call with: ours, never the call's argument.
    result: &'static str,
}

const TOOLS: [Tool; 2] = [
    Tool {
        name: "lookup",
        description: "Searches the notes and returns the matching entries, each with its id.",
        arg: "query",
        arg_description: "What to search for.",
        max: 200,
        result: "2 matches:\n[C1] scheduler-retry: how failed jobs are retried\n[C2] scheduler-queue: how jobs are queued\n",
    },
    Tool {
        name: "read_entry",
        description: "Reads one entry in full, by the id a search result gives.",
        arg: "entry",
        arg_description: "The id of the entry to read.",
        max: 300,
        result: "scheduler-retry: A failed job is retried three times, 30 seconds apart. After the third failure it is parked until someone releases it.\n",
    },
];

const SYSTEM: &str = "You answer questions from a set of notes. You have two tools: lookup searches the notes, and read_entry reads one entry by the id a search result gives. Search first, read the entry the search points to, then answer in one sentence that names the entry you used.";
const USER: &str =
    "How does the scheduler retry a job that failed? Search the notes, read the entry that covers it, and answer in one sentence.";

fn tool_defs() -> Value {
    json!(TOOLS
        .iter()
        .map(|t| json!({"type": "function", "function": {"name": t.name, "description": t.description,
            "parameters": {"type": "object",
                "properties": {t.arg: {"type": "string", "description": t.arg_description,
                    "minLength": 1, "maxLength": t.max}},
                "required": [t.arg], "additionalProperties": false}}}))
        .collect::<Vec<_>>())
}

/// What `call` gets wrong against its tool's schema: the key it must have and no other, a string of 1 to the tool's
/// limit in characters. Each problem names counts and our own names.
fn schema_problems(call: &Call) -> Vec<String> {
    let Some(tool) = TOOLS.iter().find(|t| t.name == call.name) else {
        return vec!["a tool that was not offered".to_string()];
    };
    let Some(args) = call.args.as_ref().and_then(Value::as_object) else {
        return vec![format!(
            "{}: the arguments are not a JSON object",
            tool.name
        )];
    };
    let mut problems = Vec::new();
    let extra = args.keys().filter(|k| k.as_str() != tool.arg).count();
    if extra > 0 {
        problems.push(format!("{}: {extra} extra key(s)", tool.name));
    }
    match args.get(tool.arg) {
        None => problems.push(format!(
            "{}: required key {} is missing",
            tool.name, tool.arg
        )),
        Some(Value::String(s)) => {
            let n = s.chars().count();
            if n < 1 || n > tool.max {
                problems.push(format!(
                    "{}: {} has length {n}, outside 1..={}",
                    tool.name, tool.arg, tool.max
                ));
            }
        }
        Some(_) => problems.push(format!("{}: {} is not a string", tool.name, tool.arg)),
    }
    problems
}

/// `loop_within_150_s`: the loop's wall time, from the first request sent to the last reply, against the client's
/// deadline.
fn loop_check(wall: Duration) -> Check {
    Check::new(
        "loop_within_150_s",
        wall <= LOOP_BOUND,
        format!(
            "{:.1} s against the client's {} s",
            wall.as_secs_f64(),
            LOOP_BOUND.as_secs()
        ),
    )
}

/// One request of the loop and what its reply said.
struct Round {
    status: u16,
    id: String,
    sent_at: Instant,
    done_at: Instant,
    choices: usize,
    no_function_call: bool,
    calls: Vec<Call>,
    text: String,
    /// Figures of a 200 reply: its `finish_reason` (judged by `finish_reasons_match`), and the report-only shape
    /// problems `openai_reply` found and usage counts. A reply that was not 200 keeps only its status (its error
    /// message may hold request text).
    finish: Option<String>,
    shape_problems: Vec<String>,
    completion_tokens: Option<u64>,
    reasoning_tokens: Option<u64>,
}

/// `finish_reasons_match`: every round that carried tool calls finished `tool_calls`, and the final reply, when the
/// loop ended `Ok`, finished `stop`. A loop that did not end fails it without naming the cause again
/// (`ends_in_text_within_4` has it).
fn finish_check(rounds: &[Round], outcome: &Result<(), String>) -> Check {
    const NAME: &str = "finish_reasons_match";
    if outcome.is_err() {
        return Check::new(NAME, false, "the loop did not end in text");
    }
    let mut seen = Vec::new();
    // The loop goes on only after a reply with calls, so every round but the last carried calls: a round's position
    // says which finish it owes.
    for (i, r) in rounds.iter().enumerate() {
        let want = if i + 1 == rounds.len() {
            "stop"
        } else {
            "tool_calls"
        };
        if r.finish.as_deref() != Some(want) {
            let read = r.finish.as_deref().map_or_else(
                || "no finish_reason".to_string(),
                |f| format!("finished {f}"),
            );
            return Check::new(
                NAME,
                false,
                format!("request {} {read}, want {want}", i + 1),
            );
        }
        seen.push(want);
    }
    // An `Ok` loop always has a round; the empty case is kept so the check can never pass on nothing.
    Check::new(
        NAME,
        !seen.is_empty(),
        if seen.is_empty() {
            "no replies".to_string()
        } else {
            seen.join(", ")
        },
    )
}

/// `chat-tools` (on the server of `prod-check`'s first block, after `load`): the loop above, judged by
/// `one_choice_each`, `calls_parse`, `no_function_call` and `ends_in_text_within_4`, `finish_reasons_match`
/// (a round with calls finished `tool_calls`, the final reply `stop`), `at_least_one_call`, `arguments_match_schema`,
/// `final_text_non_empty` and `no_malformed_calls`, and `loop_within_150_s`. Gated. The figures are
/// `requests`, `tool_calls`, `loop_ms` and, one entry per request each, `request_ms`, `finish_reasons` (the figure
/// behind `finish_reasons_match`) and the report-only `shape_problems` (the departures from the response shape that
/// `openai_reply` finds), `completion_tokens` and `reasoning_tokens`.
pub async fn chat_tools(ctx: &Ctx) -> Result<Verdict> {
    let mut messages = vec![
        json!({"role": "system", "content": SYSTEM}),
        json!({"role": "user", "content": USER}),
    ];
    let mut rounds: Vec<Round> = Vec::new();
    let mut tool_calls = 0;
    // How the loop ended: `Ok` in a reply without calls, else why the client would have failed it.
    let outcome: Result<(), String> = loop {
        let n = rounds.len() + 1;
        let extra = json!({"reasoning_effort": "low", "tools": tool_defs()});
        let sent = body(json!(messages), extra, MAX_OUTPUT, false);
        let r = ctx.http.send("POST", CHAT, Some(&sent)).await?;
        let v = r.json().unwrap_or(Value::Null);
        let reply = judge::openai_reply(r.status, &v);
        let choices = v["choices"].as_array().map_or(0, Vec::len);
        let message = v["choices"][0]["message"].clone();
        let calls = reply.calls.clone();
        let ok = r.status == 200;
        rounds.push(Round {
            status: r.status,
            id: reply.id,
            sent_at: r.sent_at,
            done_at: r.done_at,
            choices,
            no_function_call: judge::function_call_absent(&v),
            calls: reply.calls,
            text: reply.text,
            finish: Some(reply.finish).filter(|f| ok && !f.is_empty()),
            shape_problems: if ok {
                reply.problems
            } else {
                vec![format!("status {}", r.status)]
            },
            completion_tokens: v["usage"]["completion_tokens"].as_u64().filter(|_| ok),
            reasoning_tokens: reply.reasoning_tokens.filter(|_| ok),
        });
        if r.status != 200 {
            break Err(format!("request {n} answered status {}", r.status));
        }
        if choices != 1 {
            break Err(format!("request {n} had {choices} choices"));
        }
        if calls.is_empty() {
            break Ok(());
        }
        tool_calls += calls.len();
        if calls
            .iter()
            .any(|c| !c.args.as_ref().is_some_and(Value::is_object))
        {
            break Err(format!(
                "request {n} had a call whose arguments are not a JSON object"
            ));
        }
        if n == MAX_REQUESTS {
            break Err(format!("a tool call on request {n}"));
        }
        if tool_calls > MAX_TOOL_CALLS {
            break Err(format!("{tool_calls} tool calls by request {n}"));
        }
        // The echo: the assistant message with its calls, then one tool message per call.
        messages.push(json!({"role": "assistant", "content": message["content"],
            "tool_calls": message["tool_calls"]}));
        for c in &calls {
            let result = TOOLS
                .iter()
                .find(|t| t.name == c.name)
                .map_or("unknown tool", |t| t.result);
            messages.push(json!({"role": "tool", "tool_call_id": c.id, "content": result}));
        }
    };

    let answered: Vec<&Round> = rounds.iter().filter(|r| r.status == 200).collect();
    let all_calls: Vec<&Call> = answered.iter().flat_map(|r| r.calls.iter()).collect();
    let mut checks = Vec::new();

    let wrong_choices = answered.iter().filter(|r| r.choices != 1).count();
    checks.push(Check::new(
        "one_choice_each",
        !answered.is_empty() && wrong_choices == 0,
        if answered.is_empty() {
            "no reply answered 200".to_string()
        } else if wrong_choices == 0 {
            format!("{} replies, one choice each", answered.len())
        } else {
            format!(
                "{wrong_choices} of {} replies did not have exactly one choice",
                answered.len()
            )
        },
    ));

    let unparsed = all_calls
        .iter()
        .filter(|c| !c.args.as_ref().is_some_and(Value::is_object))
        .count();
    checks.push(Check::new(
        "calls_parse",
        unparsed == 0,
        if all_calls.is_empty() {
            "0 calls (no calls)".to_string()
        } else if unparsed == 0 {
            format!("{} calls, every arguments a JSON object", all_calls.len())
        } else {
            format!(
                "{unparsed} of {} calls had arguments that are not a JSON object",
                all_calls.len()
            )
        },
    ));

    let flagged = answered.iter().filter(|r| !r.no_function_call).count();
    checks.push(Check::new(
        "no_function_call",
        flagged == 0,
        if answered.is_empty() {
            "0 replies (no replies)".to_string()
        } else if flagged == 0 {
            "no reply carries function_call".to_string()
        } else {
            format!("{flagged} replies carry function_call")
        },
    ));

    checks.push(Check::new(
        "ends_in_text_within_4",
        outcome.is_ok(),
        match &outcome {
            Ok(()) => format!(
                "text on request {}, after {tool_calls} tool calls",
                rounds.len()
            ),
            Err(why) => why.clone(),
        },
    ));

    checks.push(finish_check(&rounds, &outcome));

    checks.push(Check::new(
        "at_least_one_call",
        tool_calls >= 1,
        format!("{tool_calls} tool calls"),
    ));

    let schema: Vec<String> = all_calls
        .iter()
        .enumerate()
        .flat_map(|(i, c)| {
            schema_problems(c)
                .into_iter()
                .map(move |p| format!("call {}: {p}", i + 1))
        })
        .collect();
    checks.push(Check::new(
        "arguments_match_schema",
        schema.is_empty(),
        if all_calls.is_empty() {
            "0 calls (no calls)".to_string()
        } else if schema.is_empty() {
            format!("{} calls match the schema", all_calls.len())
        } else {
            schema.join("; ")
        },
    ));

    let final_chars = match (&outcome, rounds.last()) {
        (Ok(()), Some(r)) => Some(r.text.trim().chars().count()),
        _ => None,
    };
    checks.push(Check::new(
        "final_text_non_empty",
        final_chars.is_some_and(|n| n > 0),
        match final_chars {
            Some(n) if n > 0 => format!("{n} characters"),
            Some(_) => "the final text is empty".to_string(),
            None => "the loop did not end in text".to_string(),
        },
    ));

    // The log's `malformed_calls` of each request: calls the parser sent as content, which no client can use.
    let mut malformed = Vec::new();
    let mut unlogged = Vec::new();
    for (i, r) in rounds.iter().enumerate().filter(|(_, r)| r.status == 200) {
        let line = line_of(ctx, &r.id).await?;
        match line.as_ref().and_then(|l| l["malformed_calls"].as_u64()) {
            Some(0) => {}
            Some(n) => malformed.push(format!("request {} has malformed_calls {n}", i + 1)),
            None => unlogged.push((i + 1).to_string()),
        }
    }
    checks.push(Check::new(
        "no_malformed_calls",
        malformed.is_empty() && unlogged.is_empty(),
        if !malformed.is_empty() || !unlogged.is_empty() {
            let mut parts = malformed;
            if !unlogged.is_empty() {
                parts.push(format!(
                    "no malformed_calls in the log line of request {}",
                    unlogged.join(", ")
                ));
            }
            parts.join("; ")
        } else if answered.is_empty() {
            "0 requests (no replies)".to_string()
        } else {
            format!("malformed_calls 0 on {} requests", answered.len())
        },
    ));

    // From the first request sent to the last reply.
    let wall = match (rounds.first(), rounds.last()) {
        (Some(a), Some(b)) => b.done_at.saturating_duration_since(a.sent_at),
        _ => Duration::ZERO,
    };
    checks.push(loop_check(wall));

    let mut figures = Map::new();
    figures.insert("requests".into(), json!(rounds.len()));
    figures.insert("tool_calls".into(), json!(tool_calls));
    figures.insert("loop_ms".into(), json!(wall.as_secs_f64() * 1000.0));
    figures.insert(
        "request_ms".into(),
        json!(rounds
            .iter()
            .map(|r| r.done_at.saturating_duration_since(r.sent_at).as_secs_f64() * 1000.0)
            .collect::<Vec<_>>()),
    );
    // One entry per request, `null` where the reply was not 200; `finish_reasons_match` judges it.
    figures.insert(
        "finish_reasons".into(),
        json!(rounds.iter().map(|r| &r.finish).collect::<Vec<_>>()),
    );
    figures.insert(
        "shape_problems".into(),
        json!(rounds.iter().map(|r| &r.shape_problems).collect::<Vec<_>>()),
    );
    figures.insert(
        "completion_tokens".into(),
        json!(rounds
            .iter()
            .map(|r| r.completion_tokens)
            .collect::<Vec<_>>()),
    );
    figures.insert(
        "reasoning_tokens".into(),
        json!(rounds
            .iter()
            .map(|r| r.reasoning_tokens)
            .collect::<Vec<_>>()),
    );
    Ok(Verdict::judged("chat-tools", checks, figures))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, arguments: &str) -> Call {
        Call {
            id: "call_1".into(),
            name: name.into(),
            arguments: arguments.into(),
            args: serde_json::from_str(arguments).ok(),
        }
    }

    /// The loop's wall time is held to the client's deadline: 150 s over all its calls, the bound included.
    #[test]
    fn chat_tools_fails_a_loop_over_150_s() {
        for ok in [
            Duration::ZERO,
            Duration::from_millis(149_900),
            Duration::from_secs(150),
        ] {
            let c = loop_check(ok);
            assert!(c.pass && c.name == "loop_within_150_s", "{c:?}");
        }
        for over in [Duration::from_millis(150_001), Duration::from_secs(600)] {
            let c = loop_check(over);
            assert!(!c.pass, "{c:?}");
        }
        let c = loop_check(Duration::from_millis(151_240));
        assert!(
            c.detail.contains("151.2") && c.detail.contains("150"),
            "{c:?}"
        );
    }

    fn round(finish: Option<&str>, calls: Vec<Call>) -> Round {
        let now = Instant::now();
        Round {
            status: 200,
            id: "chatcmpl-1".into(),
            sent_at: now,
            done_at: now,
            choices: 1,
            no_function_call: true,
            calls,
            text: String::new(),
            finish: finish.map(String::from),
            shape_problems: Vec::new(),
            completion_tokens: None,
            reasoning_tokens: None,
        }
    }

    fn rounds_of(finishes: &[&str]) -> Vec<Round> {
        let last = finishes.len() - 1;
        finishes
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let calls = if i < last {
                    vec![call("lookup", r#"{"query":"x"}"#)]
                } else {
                    Vec::new()
                };
                round(Some(f), calls)
            })
            .collect()
    }

    #[test]
    fn finish_reasons_pass_on_tool_calls_then_stop() {
        let c = finish_check(&rounds_of(&["tool_calls", "tool_calls", "stop"]), &Ok(()));
        assert!(c.pass && c.name == "finish_reasons_match", "{c:?}");
        assert_eq!(c.detail, "tool_calls, tool_calls, stop");
    }

    #[test]
    fn finish_reasons_fail_a_final_reply_cut_at_the_limit() {
        let c = finish_check(&rounds_of(&["tool_calls", "tool_calls", "length"]), &Ok(()));
        assert!(!c.pass, "{c:?}");
        assert_eq!(c.detail, "request 3 finished length, want stop");
    }

    #[test]
    fn finish_reasons_fail_a_call_round_that_reads_stop() {
        let c = finish_check(&rounds_of(&["tool_calls", "stop", "stop"]), &Ok(()));
        assert!(!c.pass, "{c:?}");
        assert_eq!(c.detail, "request 2 finished stop, want tool_calls");
        // A round with no finish_reason at all is named too.
        let mut rounds = rounds_of(&["tool_calls", "stop"]);
        rounds[0].finish = None;
        let c = finish_check(&rounds, &Ok(()));
        assert!(
            !c.pass && c.detail == "request 1 no finish_reason, want tool_calls",
            "{c:?}"
        );
    }

    #[test]
    fn finish_reasons_fail_a_loop_that_did_not_end() {
        let c = finish_check(
            &rounds_of(&["tool_calls", "tool_calls"]),
            &Err("a tool call on request 2".into()),
        );
        assert!(!c.pass && c.detail.contains("did not end"), "{c:?}");
        assert!(!c.detail.contains("tool call on request"), "{c:?}");
    }

    #[test]
    fn arguments_are_held_to_the_schema() {
        let ok = |name, args: &str| schema_problems(&call(name, args));
        assert!(ok("lookup", r#"{"query":"x"}"#).is_empty());
        assert!(ok("read_entry", r#"{"entry":"e"}"#).is_empty());
        // The limits are inclusive: 1 to 200 characters for a query, 1 to 300 for an entry.
        assert!(ok("lookup", &json!({"query": "q".repeat(200)}).to_string()).is_empty());
        assert!(ok("read_entry", &json!({"entry": "e".repeat(300)}).to_string()).is_empty());
        assert!(ok("lookup", &json!({"query": "é".repeat(200)}).to_string()).is_empty());
        for (name, args, want) in [
            ("lookup", r#"{"query":""}"#, "length 0"),
            (
                "lookup",
                &json!({"query": "q".repeat(201)}).to_string(),
                "length 201",
            ),
            (
                "read_entry",
                &json!({"entry": "e".repeat(301)}).to_string(),
                "length 301",
            ),
            ("lookup", r#"{}"#, "missing"),
            ("lookup", r#"{"entry":"x"}"#, "missing"),
            ("lookup", r#"{"query":7}"#, "not a string"),
            ("lookup", r#"{"query":"x","limit":5}"#, "1 extra key"),
            ("lookup", r#"{"query":"x","a":1,"b":2}"#, "2 extra key"),
            ("lookup", r#"["x"]"#, "not a JSON object"),
            ("nonesuch", r#"{"query":"x"}"#, "not offered"),
        ] {
            let p = ok(name, args);
            assert!(
                p.iter().any(|s| s.contains(want)),
                "{name} {args}: {p:?} lacks {want}"
            );
        }
    }
}
