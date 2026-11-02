//! `load`, `requests` and `budget`: the cases that send plain requests and read the replies.
use super::{
    anthropic_tool, body, exchange, fig, line_of, openai_tool, poll, tool_result, usable_call,
    LOG_WITHIN,
};
use crate::test7::judge::{self, Api, Call, Poll, Reply, Want};
use crate::test7::{log, Check, Ctx, Verdict};
use anyhow::{anyhow, Context, Result};
use serde_json::{json, Map, Value};
use std::path::Path;
use std::time::{Duration, Instant};

const ROUND_TRIP_PROMPT: &str =
    "Read the file notes.txt with the read_file tool, then say in one sentence what it is about.";
const PLAIN_PROMPT: &str = "Answer in one short sentence: what is the capital of France?";

/// How often `/health` is read while the load case's chat runs.
const POLL: Duration = Duration::from_millis(250);

/// The steps whose request's log line must report `reasoning_tokens` 0: each adds a check named `<step>_log_reasoning_tokens`
/// after the steps' own. For Anthropic it is the only token-level evidence of no reasoning.
pub const LOG_GATED: [&str; 2] = ["openai_effort_none", "anthropic_thinking_disabled"];

/// The steps of `requests`, in the order it sends them. Each is the name of its check; a test that scripts a fake
/// engine reads the order from here.
pub const REQUESTS_STEPS: [&str; 13] = [
    "openai_round_trip_call",
    "openai_round_trip_echo",
    "anthropic_round_trip_call",
    "anthropic_round_trip_echo",
    "openai_nonstream",
    "openai_stream",
    "anthropic_nonstream",
    "anthropic_stream",
    "openai_effort_none",
    "openai_effort_low",
    "openai_effort_medium",
    "openai_effort_high",
    "anthropic_thinking_disabled",
];

/// `load`: an idle server, one chat request (thinking off, `max_tokens` 16), `/health` read every 250 ms while it
/// runs and once more after. The load's own figures come from the `load` line the server wrote meanwhile.
pub async fn load(ctx: &Ctx) -> Result<Verdict> {
    let lines_before = log::lifecycle(&ctx.log_dir, "load")?.len();
    let body = body(
        json!([{"role": "user", "content": "Say hello."}]),
        json!({"reasoning_effort": "none"}),
        16,
        false,
    );
    let began = Instant::now();
    let http = ctx.http.clone();
    let chat =
        tokio::spawn(async move { http.send("POST", "/v1/chat/completions", Some(&body)).await });
    let mut polls: Vec<(Poll, f64)> = Vec::new();
    let mut tick = tokio::time::interval(POLL);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    while !chat.is_finished() {
        tick.tick().await;
        polls.push(poll(ctx, began).await?);
    }
    let resp = chat.await.context("the chat task failed")??;
    // The last poll reads the state after the chat has answered.
    polls.push(poll(ctx, began).await?);

    let plain: Vec<Poll> = polls.iter().map(|(p, _)| p.clone()).collect();
    let checks = judge::judge_load(resp.status, &plain);
    let loading_ms: Vec<f64> = polls
        .iter()
        .filter(|(p, _)| p.status == 503 && p.body["status"] == "loading")
        .map(|(_, at)| *at)
        .collect();
    let mut figures = Map::new();
    figures.insert("polls".into(), json!(polls.len()));
    figures.insert("loading_polls".into(), json!(loading_ms.len()));
    figures.insert("first_loading_ms".into(), json!(loading_ms.first()));
    figures.insert("last_loading_ms".into(), json!(loading_ms.last()));
    figures.insert(
        "chat_ms".into(),
        json!(resp.done_at.duration_since(began).as_secs_f64() * 1000.0),
    );
    // The load line of this case, if the server wrote one (an already loaded model writes none).
    let lines = log::lifecycle(&ctx.log_dir, "load")?;
    let line = lines.get(lines_before..).and_then(<[Value]>::last);
    for k in [
        "seconds",
        "detail",
        "slots_per_layer",
        "vram_used_mib",
        "pool_limit_gib",
        "pool_cut",
        "unparked",
    ] {
        let name = if k == "seconds" { "load_seconds" } else { k };
        figures.insert(name.into(), line.map_or(Value::Null, |l| l[k].clone()));
    }
    Ok(Verdict::judged("load", checks, figures))
}

/// `requests`: OpenAI and Anthropic, each non-streaming and streaming; a `read_file` round trip on each API;
/// OpenAI `reasoning_effort` `none`, `low`, `medium` and `high`; Anthropic `thinking: {type: "disabled"}`. The
/// round trips come first (thinking off, so the first reply is the call). The four plain steps carry a think budget
/// (OpenAI `low`, Anthropic `enabled` at 1,024 tokens) so a greedy reasoning loop can't starve their content; `high` is
/// the unbounded one. When a first reply has no call, the echo
/// still goes out with a scripted one, so the steps after it stay in their order and are judged on their own.
pub async fn requests(ctx: &Ctx) -> Result<Verdict> {
    let mut run = Steps::default();

    // OpenAI round trip.
    let q = json!({"role": "user", "content": ROUND_TRIP_PROMPT});
    let off = json!({"reasoning_effort": "none", "tools": [openai_tool()]});
    let reply = exchange(ctx, Api::OpenAi, body(json!([q]), off.clone(), 512, false)).await?;
    let call = run.step(&reply, Want::Call, "call_scripted");
    let messages = json!([
        q,
        {"role": "assistant", "content": null, "tool_calls": [{"id": call.id, "type": "function",
            "function": {"name": call.name, "arguments": call.arguments}}]},
        {"role": "tool", "tool_call_id": call.id, "content": tool_result(call.path().unwrap_or("notes.txt"))},
    ]);
    let reply = exchange(ctx, Api::OpenAi, body(messages, off, 512, false)).await?;
    run.step(&reply, Want::Echo, "");

    // Anthropic round trip.
    let q = json!({"role": "user", "content": ROUND_TRIP_PROMPT});
    let off = json!({"thinking": {"type": "disabled"}, "tools": [anthropic_tool()]});
    let reply = exchange(
        ctx,
        Api::Anthropic,
        body(json!([q]), off.clone(), 512, false),
    )
    .await?;
    let call = run.step(&reply, Want::Call, "toolu_scripted");
    let messages = json!([
        q,
        {"role": "assistant", "content": [{"type": "tool_use", "id": call.id, "name": call.name,
            "input": call.args.clone().unwrap_or_else(|| json!({}))}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call.id,
            "content": tool_result(call.path().unwrap_or("notes.txt"))}]},
    ]);
    let reply = exchange(ctx, Api::Anthropic, body(messages, off, 512, false)).await?;
    run.step(&reply, Want::Echo, "");

    // Plain requests, with a think budget, so a greedy reasoning loop can't starve their content: OpenAI `low`,
    // Anthropic `enabled` at 1,024 tokens.
    let plain = json!([{"role": "user", "content": PLAIN_PROMPT}]);
    for stream in [false, true] {
        let mut extra = json!({"reasoning_effort": "low"});
        if stream {
            extra["stream_options"] = json!({"include_usage": true});
        }
        let reply = exchange(ctx, Api::OpenAi, body(plain.clone(), extra, 2048, stream)).await?;
        run.step(&reply, Want::Text, "");
    }
    for stream in [false, true] {
        let extra = json!({"thinking": {"type": "enabled", "budget_tokens": 1024}});
        let reply = exchange(
            ctx,
            Api::Anthropic,
            body(plain.clone(), extra, 2048, stream),
        )
        .await?;
        run.step(&reply, Want::Text, "");
    }

    // Each reasoning effort, and thinking off.
    for (effort, max_tokens, want) in [
        ("none", 64, Want::TextNoReasoning),
        ("low", 2048, Want::Text),
        ("medium", 6000, Want::Text),
        ("high", 6000, Want::Text),
    ] {
        let body = body(
            plain.clone(),
            json!({"reasoning_effort": effort}),
            max_tokens,
            false,
        );
        let reply = exchange(ctx, Api::OpenAi, body).await?;
        run.step(&reply, want, "");
    }
    let body = body(plain, json!({"thinking": {"type": "disabled"}}), 64, false);
    let reply = exchange(ctx, Api::Anthropic, body).await?;
    run.step(&reply, Want::TextNoReasoning, "");

    // What the server's own log says of each request: reported, except the two no-reasoning steps' reasoning
    // tokens, which gate. Only a missing line is "no line" (a finding about the server); an unreadable directory or a
    // malformed line is a harness error.
    let mut steps = Map::new();
    let mut logged_reasoning: Vec<(&str, Option<u64>)> = Vec::new();
    for (name, id) in &run.ids {
        let fact = if id.is_empty() {
            json!({"log": "no id to look up"})
        } else {
            match log::request(&ctx.log_dir, id, LOG_WITHIN).await {
                Ok(l) => {
                    logged_reasoning.push((name, l["reasoning_tokens"].as_u64()));
                    json!({
                        "status": l["status"], "finish": l["finish"], "stream": l["stream"],
                        "tool_calls": l["tool_calls"], "malformed_calls": l["malformed_calls"],
                        "think_budget": l["think_budget"], "reasoning_tokens": l["reasoning_tokens"],
                        "prompt_tokens": l["prompt_tokens"], "completion_tokens": l["completion_tokens"],
                    })
                }
                Err(e) if e.downcast_ref::<log::NoLine>().is_some() => json!({"log": "no line"}),
                Err(e) => return Err(e.context(format!("reading the log for {name}"))),
            }
        };
        steps.insert(name.to_string(), fact);
    }
    for step in LOG_GATED {
        let logged = logged_reasoning
            .iter()
            .find(|(n, _)| *n == step)
            .and_then(|(_, t)| *t);
        run.checks.push(judge::judge_log_reasoning(
            &format!("{step}_log_reasoning_tokens"),
            logged,
        ));
    }
    let mut figures = Map::new();
    figures.insert("steps".into(), Value::Object(steps));
    figures.insert("scripted_calls".into(), json!(run.scripted));
    Ok(Verdict::judged("requests", run.checks, figures))
}

/// What `requests` has judged so far: the checks, each request's id (for the log) and the call steps that had to be
/// scripted.
#[derive(Default)]
struct Steps {
    checks: Vec<Check>,
    ids: Vec<(&'static str, String)>,
    scripted: Vec<&'static str>,
}

impl Steps {
    /// Judges the next step's reply. For a call step, returns the call its echo carries: the reply's `read_file`
    /// call if it has a string `path`, else a scripted one (`{"path": "notes.txt"}`) under `scripted_id`, and the
    /// step is noted as scripted. The echo still goes out, so the steps after it stay in their order.
    fn step(&mut self, reply: &Reply, want: Want, scripted_id: &str) -> Call {
        let name = REQUESTS_STEPS[self.checks.len()];
        self.checks.push(judge::judge_reply(name, reply, want));
        self.ids.push((name, reply.id.clone()));
        let (call, scripted) = usable_call(reply, scripted_id);
        if want == Want::Call && scripted {
            self.scripted.push(name);
        }
        call
    }
}

/// The prompt that must drive `reasoning_effort: low` past its 1,024-token budget under greedy decoding.
pub const BUDGET_PROMPT: &str = "Find every pair of positive integers (a, b) with a \u{2264} b \u{2264} 60 such that a\u{b2} + b\u{b2} is a perfect square. Check each candidate carefully before answering.";

/// The length of the server's forced close (`\n</think>\n\n`), from the committed vocabulary under `root` (the
/// repository, as `--prompts-root` names it): what `Specials::new` makes of it.
fn close_len(root: &Path) -> Result<u32> {
    let path = root.join("testdata/tokenizer/qwen36-vocab.bin");
    let bytes = std::fs::read(&path).with_context(|| {
        format!(
            "cannot read {} (--prompts-root is the repository root)",
            path.display()
        )
    })?;
    let vocab = qwen36::tokenizer::Vocab::from_extract(&bytes)
        .map_err(|e| anyhow!("{}: {e}", path.display()))?;
    let tk = qwen36::tokenizer::Tokenizer::new(&vocab)
        .map_err(|e| anyhow!("{}: {e}", path.display()))?;
    let sp = server::tokens::Specials::new(&vocab, &tk).map_err(|e| anyhow!(e))?;
    Ok(u32::try_from(sp.think_close.len())?)
}

/// `budget`: `reasoning_effort: low` (a budget of 1,024), greedy, streaming, `max_tokens` 4,096, on `BUDGET_PROMPT`.
/// The reasoning count is the usage chunk's, so it is the server's own.
pub async fn budget(ctx: &Ctx) -> Result<Verdict> {
    let close = close_len(&ctx.prompts_root)?;
    let extra = json!({"reasoning_effort": "low", "stream_options": {"include_usage": true}});
    let messages = json!([{"role": "user", "content": BUDGET_PROMPT}]);
    let reply = exchange(ctx, Api::OpenAi, body(messages, extra, 4096, true)).await?;
    let line = line_of(ctx, &reply.id).await?;
    let mut checks = vec![Check::new(
        "answers",
        reply.status == 200,
        format!("status {}", reply.status),
    )];
    checks.push(match reply.reasoning_tokens {
        Some(n) => judge::budget_check(u32::try_from(n).unwrap_or(u32::MAX), close, &reply.finish),
        None => Check::new(
            "reasoning_budget",
            false,
            "the reply reports no reasoning_tokens",
        ),
    });
    let chars = reply.text.trim().chars().count();
    checks.push(Check::new(
        "has_content",
        chars > 0,
        format!(
            "{chars} chars of content beside {} of reasoning",
            reply.reasoning.chars().count()
        ),
    ));
    let mut figures = Map::new();
    figures.insert("reasoning_tokens".into(), json!(reply.reasoning_tokens));
    figures.insert("close_len".into(), json!(close));
    figures.insert("finish".into(), json!(reply.finish));
    figures.insert("content_chars".into(), json!(chars));
    for k in [
        "think_budget",
        "budget_closed",
        "completion_tokens",
        "reasoning_tokens",
    ] {
        let name = if k == "reasoning_tokens" {
            "logged_reasoning_tokens"
        } else {
            k
        };
        figures.insert(name.into(), fig(line.as_deref(), k));
    }
    Ok(Verdict::judged("budget", checks, figures))
}
