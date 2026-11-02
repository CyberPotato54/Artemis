//! The request and lifecycle logs: every field, the divergence cause, and no text.
mod common;

use common::{specials, tokenizer, wait_until, Setup, Sse, SseItem, TestServer, TESTDATA};
use serde_json::{json, Value};
use server::backend::{CacheReport, Cut, MtpDraftRow, Source, SpecCounters, SpecReport, StepSize};
use server::fake::{Fail, FakeConfig, Script};
use server::logs::{classify_divergence, Cause};
use std::io::Write;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

const CHAT: &str = "/v1/chat/completions";
const MSGS: &str = "/v1/messages";
const WAIT: Duration = Duration::from_secs(20);

/// The fields, in the order the line writes them.
const FIELDS: [&str; 54] = [
    "time",
    "id",
    "endpoint",
    "peer",
    "model",
    "status",
    "prompt_tokens",
    "cut_at",
    "user_query",
    "stream",
    "tool_calls",
    "malformed_calls",
    "think_budget",
    "budget_closed",
    "reused_tokens",
    "restored",
    "restore_bytes",
    "restore_ms",
    "diverged_at",
    "cause",
    "prefilled_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "queue_ms",
    "load_ms",
    "prefill_ms",
    "ttft_ms",
    "total_ms",
    "prefill_tok_s",
    "decode_tok_s",
    "decode_ms",
    "decode_tokens",
    "decode_misses",
    "hit_rate",
    "miss_mib_per_token",
    "spec_steps",
    "spec_plain_steps",
    "spec_drafted",
    "spec_accepted",
    "spec_accept_rate",
    "spec_replays",
    "spec_drafted_lookup",
    "spec_accepted_lookup",
    "spec_drafted_mtp",
    "spec_accepted_mtp",
    "pool_bytes_used",
    "finish",
    "error_kind",
    "error_message",
    "ignored",
    "ignored_more",
    "repairs",
    "repairs_more",
    "dropped_messages",
];

// ---- the log files ----

/// Every line the server has logged, oldest first.
fn all_lines(srv: &TestServer) -> Vec<Value> {
    let mut files: Vec<_> = std::fs::read_dir(&srv.log_dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .collect();
    files.sort();
    files
        .iter()
        .flat_map(|p| {
            std::fs::read_to_string(p)
                .unwrap()
                .lines()
                .map(String::from)
                .collect::<Vec<_>>()
        })
        .map(|l| serde_json::from_str(&l).unwrap_or_else(|e| panic!("a bad log line ({e}): {l}")))
        .collect()
}

/// The request lines: the ones without an `event`.
fn request_lines(srv: &TestServer) -> Vec<Value> {
    all_lines(srv)
        .into_iter()
        .filter(|l| l.get("event").is_none())
        .collect()
}

/// The lifecycle lines, as `(event, line)`.
fn events(srv: &TestServer) -> Vec<(String, Value)> {
    all_lines(srv)
        .into_iter()
        .filter_map(|l| Some((l.get("event")?.as_str()?.to_string(), l)))
        .collect()
}

fn keys(line: &Value) -> Vec<&str> {
    line.as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect()
}

// ---- prompts and their divergence ----

/// The request body of an oracle fixture whose file name starts with `prefix`.
fn fixture(prefix: &str) -> Value {
    let dir = format!("{TESTDATA}template/requests");
    let path = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.file_name().unwrap().to_string_lossy().starts_with(prefix))
        .unwrap_or_else(|| panic!("no fixture {prefix}"));
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// The prompt of an OpenAI body (the tokens, with special tokens parsed) and its cut.
fn prompt_of(body: &Value) -> (Vec<u32>, Option<Cut>) {
    let defaults = server::params::Defaults::new("artemis", 65536);
    let convo = server::openai::request::parse(&serde_json::to_vec(body).unwrap())
        .and_then(|mut r| {
            let f = r.fields(&defaults)?;
            r.conversation(&f)
        })
        .unwrap();
    let p =
        server::prompt::build_prompt(&convo, common::template(), tokenizer(), specials(), 65536)
            .unwrap();
    (p.tokens, p.cut)
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The end of the text before token `d`, for a failure message.
fn tail(tokens: &[u32], d: usize) -> String {
    let text = tokenizer().decode(&tokens[..d]);
    let from = text.char_indices().rev().nth(39).map_or(0, |(i, _)| i);
    text[from..].to_string()
}

/// A divergence case: what a snapshot held (`prev`, the tokens it covers), what the next request's prompt is (`new`),
/// and where they part. The expected cause is derived from the rendered text, not from the classifier: `before_ends`
/// is what the prompt says just before the parting point, and the comment on each case says why that is this cause.
fn check(name: &str, prev: &[u32], new: &Value, restored: Source, before_ends: &str, want: Cause) {
    let (tokens, cut) = prompt_of(new);
    let d = lcp(prev, &tokens);
    let before = tokenizer().decode(&tokens[..d]);
    assert!(
        before.ends_with(before_ends),
        "{name}: the prompt before the parting point (token {d}) ends {:?}, not {before_ends:?}",
        tail(&tokens, d)
    );
    let got = classify_divergence(
        &tokens,
        cut.as_ref(),
        d as u32,
        &restored,
        specials(),
        tokenizer(),
    );
    assert_eq!(
        got, want,
        "{name}: parted at token {d}, after {before_ends:?}"
    );
}

/// The index of the `n`th (from 0) token `id` in `tokens`.
fn nth(tokens: &[u32], id: u32, n: usize) -> usize {
    tokens
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == id)
        .nth(n)
        .map(|(i, _)| i)
        .unwrap()
}

fn set_text(body: &mut Value, message: usize, key: &str, text: &str) {
    body["messages"][message][key] = json!(text);
}

#[test]
fn cause_by_turn() {
    let sp = specials();
    let t18 = fixture("t18");
    let (t18_tokens, t18_cut) = prompt_of(&t18);
    let t01 = fixture("t01");
    let t03 = fixture("t03");
    let t04 = fixture("t04");
    let t05 = fixture("t05");
    let (t04_tokens, _) = prompt_of(&t04);
    let (t05_tokens, _) = prompt_of(&t05);

    // t18 is user, assistant (reasoning and content, kept by default), user. t01 is the same history with
    // `preserve_thinking: false`; t04 a tool loop whose assistant turn has no reasoning; t05 a loop with its reasoning
    // followed by a new user turn.
    // The tokens before the `n`th `<|im_end|>`: a snapshot taken when that turn ended (the stop token isn't decoded).
    let reply_end = |tokens: &[u32], n: usize| tokens[..nth(tokens, sp.im_end, n)].to_vec();

    // A new system prompt: the two prompts part inside the system turn, after the words they share.
    let mut a = t18.clone();
    a["messages"]
        .as_array_mut()
        .unwrap()
        .insert(0, json!({"role": "system", "content": "You are helpful."}));
    let mut b = t18.clone();
    b["messages"]
        .as_array_mut()
        .unwrap()
        .insert(0, json!({"role": "system", "content": "You are terse."}));
    let (a_tokens, _) = prompt_of(&a);
    check(
        "a new system prompt",
        &a_tokens,
        &b,
        Source::PromptEnd,
        "<|im_start|>system\nYou are",
        Cause::NewConversation,
    );

    // The opening turn differs: t04 opens with the tools' system turn, t18 with the user; or the first message does.
    check(
        "another conversation",
        &t04_tokens,
        &t18,
        Source::PromptEnd,
        "",
        Cause::NewConversation,
    );
    let mut c = t18.clone();
    set_text(&mut c, 0, "content", "What is 17 plus 3?");
    check(
        "another first message",
        &t18_tokens,
        &c,
        Source::PromptEnd,
        "<|im_start|>user\nWhat is 17",
        Cause::NewConversation,
    );
    // A first message or a system prompt cut short parts at its own `<|im_end|>`. That is a turn's end, but the prompt
    // still does not open as the cache did: a new conversation. A later user turn cut short is new input.
    let mut c = t18.clone();
    set_text(&mut c, 0, "content", "What is 17");
    check(
        "a first message cut short",
        &t18_tokens,
        &c,
        Source::PromptEnd,
        "<|im_start|>user\nWhat is 17",
        Cause::NewConversation,
    );
    let mut c = b.clone();
    c["messages"][0]["content"] = json!("You are");
    check(
        "a system prompt cut short",
        &a_tokens,
        &c,
        Source::PromptEnd,
        "<|im_start|>system\nYou are",
        Cause::NewConversation,
    );
    let mut c = t18.clone();
    set_text(&mut c, 2, "content", "Now");
    check(
        "a later user turn cut short",
        &t18_tokens,
        &c,
        Source::PromptEnd,
        "<|im_start|>user\nNow",
        Cause::NewInput,
    );

    // The engine never decodes the stop token: the reply-end snapshot ends at the reply's last content token. The
    // request adds the next user turn, so the prompt parts at the `<|im_end|>` itself: the history matches, and what
    // follows is new input.
    check(
        "a new user turn after a reply (t18)",
        &reply_end(&t18_tokens, 1),
        &t18,
        Source::ReplyEnd,
        "17 times 3 is 51.",
        Cause::NewInput,
    );
    // The same at a tool loop's results: the cache ends at the call's last token, the `</tool_call>` the call ends
    // with.
    check(
        "tool results after a call (t04)",
        &reply_end(&t04_tokens, 2),
        &t04,
        Source::ReplyEnd,
        "</tool_call>",
        Cause::NewInput,
    );
    // With parallel calls off the stop token is `</tool_call>` and the cache ends before it: the prompt parts at the
    // `</tool_call>`, which the `<|im_end|>` follows.
    let call_cut = nth(&t04_tokens, sp.im_end, 2) - 1;
    assert_eq!(t04_tokens[call_cut], sp.end_tool_call);
    check(
        "tool results after a call, parallel off (t04)",
        &t04_tokens[..call_cut],
        &t04,
        Source::ReplyEnd,
        "</function>\n",
        Cause::NewInput,
    );
    // And a new user turn after the loop's final reply (t05).
    check(
        "a new user turn after a tool loop (t05)",
        &reply_end(&t05_tokens, 4),
        &t05,
        Source::ReplyEnd,
        "It is snowing lightly in Oslo at -2 C.",
        Cause::NewInput,
    );
    // A cache that ran on through the `<|im_end|>` parts at the `\n` after it: new input as well.
    check(
        "a new user turn, the stop token held (t18)",
        &t18_tokens[..=nth(&t18_tokens, sp.im_end, 1)],
        &t18,
        Source::ReplyEnd,
        "17 times 3 is 51.<|im_end|>",
        Cause::NewInput,
    );

    // The user's text changed: the prompts part inside the last user turn.
    let mut c = t18.clone();
    set_text(&mut c, 2, "content", "Now triple that.");
    check(
        "a user turn that changed",
        &t18_tokens,
        &c,
        Source::PromptEnd,
        "<|im_start|>user\nNow",
        Cause::NewInput,
    );
    // A tool result that changed: it is inside a user turn too.
    let mut c = t04.clone();
    set_text(&mut c, 2, "content", "Light snow, -3 C.");
    check(
        "a tool result that changed",
        &t04_tokens,
        &c,
        Source::PromptEnd,
        "<tool_response>\nLight snow, -",
        Cause::NewInput,
    );

    // The reply's text changed after the same reasoning: the prompts part inside the assistant turn after its
    // `</think>`.
    let mut c = t18.clone();
    set_text(&mut c, 1, "content", "17 x 3 = 51.");
    check(
        "a reply that differs (t18)",
        &t18_tokens,
        &c,
        Source::PromptEnd,
        "</think>\n\n17",
        Cause::ReplyRetokenized,
    );

    // A client that sends no reasoning back in a tool loop: t03 echoes it, t04 does not, so t04 renders an empty
    // think block where the cache holds the reasoning. They part right after `<think>`, inside the think block.
    let (t03_tokens, _) = prompt_of(&t03);
    check(
        "reasoning not echoed in a tool loop (t04 against t03)",
        &t03_tokens,
        &t04,
        Source::PromptEnd,
        "<|im_start|>assistant\n<think>",
        Cause::ReasoningDropped,
    );

    // Inside a turn the template rendered with no think block, past its start, it is the reply that differs.
    let (t01_tokens, _) = prompt_of(&t01);
    let mut c = t01.clone();
    set_text(&mut c, 1, "content", "17 x 3 = 51.");
    check(
        "a reply that differs, no think block (t01)",
        &t01_tokens,
        &c,
        Source::PromptEnd,
        "<|im_start|>assistant\n17",
        Cause::ReplyRetokenized,
    );

    // A new query drops the reasoning of the turns before it (`preserve_thinking: false`): the template renders the
    // earlier assistant turn with no think block at all, so t01 parts from t18 (reasoning kept) right after
    // `assistant\n`, where t18 has `<think>`.
    check(
        "reasoning dropped by a new query (t01 against t18)",
        &t18_tokens,
        &t01,
        Source::PromptEnd,
        "<|im_start|>assistant\n",
        Cause::ReasoningDropped,
    );
    let mut dropped = t05.clone();
    dropped["chat_template_kwargs"] = json!({"preserve_thinking": false});
    check(
        "reasoning dropped by a new query after a tool loop (t05)",
        &t05_tokens,
        &dropped,
        Source::UserQuery,
        "What is the weather in Oslo?<|im_end|>\n<|im_start|>assistant\n",
        Cause::ReasoningDropped,
    );

    // No snapshot in reach: the engine restored nothing and the prompts still share a prefix.
    let mut c = t18.clone();
    set_text(&mut c, 2, "content", "Now triple that.");
    check(
        "no snapshot",
        &t18_tokens,
        &c,
        Source::None,
        "<|im_start|>user\nNow",
        Cause::NoSnapshot,
    );

    // A parting point of 0 is a new conversation, restored or not.
    let (tokens, _) = prompt_of(&t18);
    for restored in [Source::None, Source::PromptEnd] {
        let got = classify_divergence(&tokens, None, 0, &restored, sp, tokenizer());
        assert_eq!(got, Cause::NewConversation);
    }

    // Past the cut there is no divergence to speak of: the generation prompt is always new.
    let cut = t18_cut.unwrap();
    let (tokens, _) = prompt_of(&t18);
    for (name, d) in [
        ("the whole prompt matched", tokens.len() as u32),
        ("the cut", cut.at),
        ("past the cut", cut.at + 2),
    ] {
        let got = classify_divergence(&tokens, Some(&cut), d, &Source::PromptEnd, sp, tokenizer());
        assert_eq!(got, Cause::None, "{name}");
    }
    // One token before the cut is the `\n` after the last user turn's `<|im_end|>`: new input.
    let got = classify_divergence(
        &tokens,
        Some(&cut),
        cut.at - 1,
        &Source::PromptEnd,
        sp,
        tokenizer(),
    );
    assert_eq!(got, Cause::NewInput);
    // A prompt with no cut classifies by its turns alone.
    let got = classify_divergence(
        &tokens,
        None,
        tokens.len() as u32,
        &Source::PromptEnd,
        sp,
        tokenizer(),
    );
    assert_eq!(got, Cause::None);
}

// ---- the request line ----

fn chat_body(extra: Value) -> Vec<u8> {
    let mut b =
        json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "none"});
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b.to_string().into_bytes()
}

fn msgs_body(extra: Value) -> Vec<u8> {
    let mut b = json!({"model": "claude-x", "max_tokens": 256, "messages": [{"role": "user", "content": "hi"}]});
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b.to_string().into_bytes()
}

/// Reads a stream to its end.
async fn drain(sse: &mut Sse) {
    while sse.next(WAIT).await.is_some() {}
    assert!(sse.ended(), "the stream stalled");
}

fn num(line: &Value, key: &str) -> f64 {
    line[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{key} is not a number in {line}"))
}

#[tokio::test]
async fn request_lines_carry_every_field() {
    let mut setup = Setup::new();
    setup.fake_logs = true;
    let srv = TestServer::start(setup).await;

    // A 503: the model fails to load, before the engine takes the request.
    srv.fake.push(Script {
        load_fails: Some("scripted load failure".into()),
        ..Script::default()
    });
    let r = srv.send("POST", CHAT, &[], &chat_body(json!({}))).await;
    assert_eq!(r.status, 503, "{}", String::from_utf8_lossy(&r.body));

    // A 400: the messages are not a list.
    let r = srv
        .send("POST", CHAT, &[], br#"{"messages": "nope"}"#)
        .await;
    assert_eq!(r.status, 400);

    // An OpenAI request that restored part of its prompt, with the engine's figures carried to the line.
    let cache = CacheReport {
        hit_rate: Some(0.75),
        miss_mib_per_token: Some(1.5),
        restore_bytes: 4096,
        pool_bytes_used: 123_456,
        decode_misses: Some(11),
    };
    // Thinking is on (the default), so the reply's first words are reasoning.
    let body = json!({"model": "gpt-x", "frequency_penalty": 0.5, "messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"}]});
    // The request's third turn starts at the third `<|im_start|>`: the prompt parts at its text.
    let (prompt_tokens, _) = prompt_of(&body);
    let diverged_at = nth(&prompt_tokens, specials().im_start, 2) as u32 + 3;
    srv.fake.push(Script {
        diverged_at,
        reused: 5,
        source: Source::PromptEnd,
        restore_ms: 12.5,
        cache: Some(cache),
        prefill: Duration::from_millis(5),
        per_token: Duration::from_millis(2),
        ..Script::text(tokenizer(), "Thinking.\n</think>\n\nHello there.")
    });
    let r = srv
        .send("POST", CHAT, &[], body.to_string().as_bytes())
        .await;
    assert_eq!(r.status, 200);

    // An Anthropic stream with the fake's default figures.
    let mut s = srv
        .stream(MSGS, &[], &msgs_body(json!({"stream": true})))
        .await;
    assert_eq!(s.status(), 200);
    drain(&mut s).await;
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 4).await);

    let lines = request_lines(&srv);
    for l in &lines {
        assert_eq!(keys(l), FIELDS, "{l}");
        assert!(l["time"].as_str().unwrap().ends_with('Z'));
        assert!(l["ignored"].is_array() && l["repairs"].is_array());
        // Every line, a refusal and a stream included, says how long its request ran.
        assert!(num(l, "total_ms") >= 0.0, "{l}");
    }

    let [unavailable, bad, openai, anthropic] = &lines[..] else {
        panic!("{lines:?}")
    };
    // The request ran at least through its prefill and first token.
    assert!(
        num(openai, "total_ms") >= num(openai, "ttft_ms"),
        "{openai}"
    );
    // The 503 had its prompt tokenized and nothing more.
    assert_eq!(unavailable["status"], 503);
    assert_eq!(unavailable["error_kind"], "service_unavailable");
    assert!(num(unavailable, "prompt_tokens") > 0.0);
    for k in [
        "reused_tokens",
        "restored",
        "cause",
        "completion_tokens",
        "finish",
        "ttft_ms",
        "hit_rate",
    ] {
        assert!(unavailable[k].is_null(), "{k}: {unavailable}");
    }
    // The 400 never got to a prompt.
    assert_eq!(bad["status"], 400);
    assert_eq!(bad["error_kind"], "invalid_request_error");
    assert_eq!(bad["endpoint"], CHAT);
    for k in ["model", "prompt_tokens", "restored", "finish", "queue_ms"] {
        assert!(bad[k].is_null(), "{k}: {bad}");
    }

    // The OpenAI request, field by field.
    let prompt = srv.fake.requests()[0].1.prompt.len() as f64;
    assert!(openai["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(openai["endpoint"], CHAT);
    assert_eq!(openai["model"], "gpt-x");
    assert_eq!(openai["status"], 200);
    assert_eq!(num(openai, "prompt_tokens"), prompt);
    assert_eq!(num(openai, "reused_tokens"), 5.0);
    assert_eq!(openai["restored"], "prompt_end");
    assert_eq!(num(openai, "restore_bytes"), 4096.0);
    assert_eq!(num(openai, "restore_ms"), 12.5);
    assert_eq!(num(openai, "diverged_at"), f64::from(diverged_at));
    // The parting point is in the last user turn.
    assert_eq!(openai["cause"], "new_input");
    assert_eq!(num(openai, "prefilled_tokens"), prompt - 5.0);
    assert!(num(openai, "completion_tokens") > 0.0);
    assert!(num(openai, "reasoning_tokens") > 0.0);
    assert!(num(openai, "queue_ms") >= 0.0);
    assert!(num(openai, "load_ms") >= 0.0);
    assert!(num(openai, "prefill_ms") >= 5.0);
    assert!(num(openai, "ttft_ms") > 0.0);
    assert!(num(openai, "prefill_tok_s") > 0.0 && num(openai, "decode_tok_s") > 0.0);
    assert_eq!(num(openai, "hit_rate"), 0.75);
    assert_eq!(num(openai, "miss_mib_per_token"), 1.5);
    assert_eq!(num(openai, "pool_bytes_used"), 123_456.0);
    assert_eq!(num(openai, "decode_misses"), 11.0);
    assert_eq!(openai["finish"], "stop");
    assert!(openai["error_kind"].is_null());
    assert_eq!(openai["ignored"], json!(["frequency_penalty"]));
    // The new keys: a whole reply, no calls, thinking on with no budget.
    assert_eq!(openai["stream"], false);
    assert_eq!(
        (&openai["tool_calls"], &openai["malformed_calls"]),
        (&json!(0), &json!(0))
    );
    assert!(openai["think_budget"].is_null() && openai["budget_closed"] == false);
    assert!(openai["cut_at"].as_u64().is_some() && openai["user_query"].is_boolean());
    // The 400 never got to a prompt; the 503 built its prompt, so it has the cut, before the engine refused it.
    assert_eq!(bad["stream"], false);
    assert!(bad["cut_at"].is_null() && bad["user_query"].is_null());
    assert!(unavailable["cut_at"].as_u64().is_some() && unavailable["think_budget"].is_null());

    // The Anthropic stream: the engine reported no rates, so they are null, not zero.
    assert!(anthropic["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(anthropic["endpoint"], MSGS);
    assert_eq!(anthropic["model"], "claude-x");
    assert_eq!(anthropic["status"], 200);
    assert_eq!(anthropic["restored"], "none");
    assert_eq!(num(anthropic, "restore_bytes"), 0.0);
    assert_eq!(num(anthropic, "restore_ms"), 0.0);
    assert!(anthropic["hit_rate"].is_null() && anthropic["miss_mib_per_token"].is_null());
    assert_eq!(num(anthropic, "pool_bytes_used"), 0.0);
    assert_eq!(anthropic["cause"], "new_conversation");
    assert!(num(anthropic, "completion_tokens") > 0.0);
    assert_eq!(anthropic["finish"], "end_turn");
    assert_eq!(anthropic["stream"], true);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn the_budget_and_its_close_are_logged() {
    let srv = TestServer::start(Setup::new()).await;
    let word = tokenizer().encode("word", true)[0];

    // `low` is a 1024-token budget; a reply that reasons past it has the close forced.
    srv.fake.push(Script {
        tokens: vec![word; 1100],
        ..Script::default()
    });
    let r = srv
        .send(
            "POST",
            CHAT,
            &[],
            &chat_body(json!({"reasoning_effort": "low"})),
        )
        .await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    // A reply that closes its own thinking inside the budget has the budget but no forced close.
    srv.fake
        .push(Script::text(tokenizer(), "Short.\n</think>\n\nDone."));
    let r = srv
        .send(
            "POST",
            CHAT,
            &[],
            &chat_body(json!({"reasoning_effort": "low"})),
        )
        .await;
    assert_eq!(r.status, 200);
    // No thinking: no budget.
    let r = srv.send("POST", CHAT, &[], &chat_body(json!({}))).await;
    assert_eq!(r.status, 200);
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 3).await);

    let lines = request_lines(&srv);
    let want = [
        (json!(1024), true),
        (json!(1024), false),
        (Value::Null, false),
    ];
    for (l, (budget, closed)) in lines.iter().zip(want) {
        assert_eq!(l["think_budget"], budget, "{l}");
        assert_eq!(l["budget_closed"], closed, "{l}");
    }
    srv.shutdown().await.unwrap();
}

/// A template whose generation prompt is empty: the prompt has no cut.
const NO_CUT_TEMPLATE: &str =
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}";

#[tokio::test]
async fn the_cut_and_flag_are_logged() {
    let srv = TestServer::start(Setup::new()).await;
    // A chat and a tool loop from the golden prompts: whatever cut the fake was handed is the line's.
    for prefix in ["t18", "t04"] {
        let r = srv
            .send("POST", CHAT, &[], fixture(prefix).to_string().as_bytes())
            .await;
        assert_eq!(
            r.status,
            200,
            "{prefix}: {}",
            String::from_utf8_lossy(&r.body)
        );
    }
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 2).await);
    let runs = srv.fake.requests();
    let lines = request_lines(&srv);
    let mut flags = Vec::new();
    for (l, (_, req)) in lines.iter().zip(&runs) {
        let cut = req.cut.as_ref().expect("a golden prompt has a cut");
        assert_eq!(num(l, "cut_at"), f64::from(cut.at), "{l}");
        assert_eq!(l["user_query"], cut.user_query, "{l}");
        flags.push(cut.user_query);
    }
    // The two prompts differ in the flag: it is not a constant.
    assert_eq!(flags.len(), 2);
    assert_ne!(flags[0], flags[1], "{flags:?}");
    srv.shutdown().await.unwrap();

    // A prompt with no cut logs null for both.
    let mut setup = Setup::new();
    setup.template = Some(NO_CUT_TEMPLATE);
    let srv = TestServer::start(setup).await;
    let r = srv.send("POST", CHAT, &[], &chat_body(json!({}))).await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    assert!(srv.fake.requests()[0].1.cut.is_none());
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 1).await);
    let l = &request_lines(&srv)[0];
    assert!(l["cut_at"].is_null() && l["user_query"].is_null(), "{l}");
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn stream_and_call_counts_are_logged() {
    let srv = TestServer::start(Setup::new()).await;
    let tools = json!([{"type": "function", "function": {
        "name": "get_weather", "description": "weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]);
    let call = |name: &str| {
        format!("<tool_call>\n<function={name}>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>")
    };
    let pair = format!("{}\n{}", call("get_weather"), call("get_weather"));
    // A reply whose one call names a function the request did not declare: it comes back as content.
    let undeclared = call("teleport");

    // A streamed reply with two valid calls, and the same whole.
    srv.fake.push(Script::text(tokenizer(), &pair));
    let mut s = srv
        .stream(
            CHAT,
            &[],
            &chat_body(json!({"stream": true, "tools": tools})),
        )
        .await;
    assert_eq!(s.status(), 200);
    drain(&mut s).await;
    srv.fake.push(Script::text(tokenizer(), &pair));
    let r = srv
        .send("POST", CHAT, &[], &chat_body(json!({"tools": tools})))
        .await;
    assert_eq!(r.status, 200);
    // The malformed one, whole and streamed.
    srv.fake.push(Script::text(tokenizer(), &undeclared));
    let r = srv
        .send("POST", CHAT, &[], &chat_body(json!({"tools": tools})))
        .await;
    assert_eq!(r.status, 200);
    srv.fake.push(Script::text(tokenizer(), &undeclared));
    let mut s = srv
        .stream(
            CHAT,
            &[],
            &chat_body(json!({"stream": true, "tools": tools})),
        )
        .await;
    assert_eq!(s.status(), 200);
    drain(&mut s).await;
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 4).await);

    // (stream, valid calls, malformed calls), in the order they were sent.
    let got: Vec<(Value, Value, Value)> = request_lines(&srv)
        .iter()
        .map(|l| {
            (
                l["stream"].clone(),
                l["tool_calls"].clone(),
                l["malformed_calls"].clone(),
            )
        })
        .collect();
    let want = [
        (true, 2, 0),  // streamed, valid
        (false, 2, 0), // whole, valid
        (false, 0, 1), // whole, malformed
        (true, 0, 1),  // streamed, malformed
    ];
    let want: Vec<(Value, Value, Value)> = want
        .iter()
        .map(|&(s, t, m)| (json!(s), json!(t), json!(m)))
        .collect();
    assert_eq!(got, want);

    // A stream the client leaves after a valid call: the line is written as it stands, with the counts so far.
    srv.fake.push(Script {
        per_token: Duration::from_millis(20),
        ..Script::text(
            tokenizer(),
            &format!("{}\n\n{}", call("get_weather"), "word ".repeat(400)),
        )
    });
    let mut s = srv
        .stream(
            CHAT,
            &[],
            &chat_body(json!({"stream": true, "tools": tools})),
        )
        .await;
    assert_eq!(s.status(), 200);
    // Content after the call means the call has ended.
    loop {
        match s.next(WAIT).await {
            Some(SseItem::Event { data, .. }) if data.contains("word") => break,
            Some(_) => {}
            None => panic!("the stream ended before its content"),
        }
    }
    s.disconnect();
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 5).await);
    let line = request_lines(&srv).remove(4);
    assert_eq!(line["finish"], "cancelled", "{line}");
    assert_eq!(line["stream"], true, "{line}");
    assert_eq!(line["tool_calls"], 1, "{line}");
    assert_eq!(line["malformed_calls"], 0, "{line}");
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_client_that_leaves_still_leaves_a_line() {
    let mut setup = Setup::new();
    setup.fake.start_loaded = true;
    let srv = TestServer::start(setup).await;
    let slow = || Script {
        per_token: Duration::from_millis(20),
        ..Script::text(
            tokenizer(),
            &format!("x\n</think>\n\n{}", "word ".repeat(400)),
        )
    };

    // A stream: its headers went out with 200, and it ends `cancelled` with the tokens it had.
    srv.fake.push(slow());
    let mut s = srv
        .stream(CHAT, &[], &chat_body(json!({"stream": true})))
        .await;
    assert_eq!(s.status(), 200);
    for _ in 0..4 {
        assert!(s.next(WAIT).await.is_some());
    }
    s.disconnect();
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 1).await);
    let line = request_lines(&srv).remove(0);
    assert_eq!(keys(&line), FIELDS);
    assert_eq!(line["status"], 200);
    assert_eq!(line["finish"], "cancelled");
    assert!(num(&line, "prompt_tokens") > 0.0);
    assert!(num(&line, "completion_tokens") >= 1.0, "{line}");

    // A whole reply: nothing was sent, so the line says 499 (the client closed the request), with what was known when
    // it left.
    srv.fake.push(slow());
    srv.send_then_drop(
        "POST",
        CHAT,
        &[],
        &chat_body(json!({})),
        Duration::from_millis(200),
    )
    .await;
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 2).await);
    let line = request_lines(&srv).remove(1);
    assert_eq!(keys(&line), FIELDS);
    assert_eq!(line["status"], 499);
    assert_eq!(line["finish"], "cancelled");
    assert!(num(&line, "prompt_tokens") > 0.0);
    assert!(line["error_kind"].is_null());
    srv.shutdown().await.unwrap();
}

// ---- the lifecycle lines ----

fn lines_of(srv: &TestServer, event: &str) -> Vec<Value> {
    events(srv)
        .into_iter()
        .filter(|(e, _)| e == event)
        .map(|(_, l)| l)
        .collect()
}

#[tokio::test]
async fn lifecycle_lines() {
    // The idle unload and a load, written by the fake as the engine's adapter writes them. (The adapter follows its idle
    // `unload` with a `stats` line; the fake writes none, which this test does not look for.)
    let mut setup = Setup::new();
    setup.fake_logs = true;
    setup.fake = FakeConfig {
        idle_unload: Some(Duration::from_millis(200)),
        ..FakeConfig::default()
    };
    let srv = TestServer::start(setup).await;
    assert_eq!(srv.send("POST", "/v1/load", &[], b"").await.status, 200);
    assert!(wait_until(WAIT, || !lines_of(&srv, "unload").is_empty()).await);
    let load = lines_of(&srv, "load").remove(0);
    assert!(load["seconds"].as_f64().unwrap() >= 0.0);
    assert!(!load["detail"].as_str().unwrap().is_empty());
    assert!(load["time"].as_str().unwrap().ends_with('Z'));
    assert_eq!(lines_of(&srv, "unload")[0]["cause"], "idle");
    srv.shutdown().await.unwrap();

    // The server's own: an unload it was asked for and its stats, a device loss, the unload at shutdown, and the stats
    // and the exit that end the log.
    let mut setup = Setup::new();
    setup.fake_logs = true;
    let mut srv = TestServer::start(setup).await;
    assert_eq!(srv.send("POST", "/v1/load", &[], b"").await.status, 200);
    srv.fake.push(Script {
        fail_at: Some((0, Fail::DeviceLost)),
        ..Script::default()
    });
    let r = srv.send("POST", CHAT, &[], &chat_body(json!({}))).await;
    assert_eq!(r.status, 503);
    assert_eq!(srv.send("POST", "/v1/unload", &[], b"").await.status, 200);
    // An unload of a model that is not loaded is not an unload.
    assert_eq!(srv.send("POST", "/v1/unload", &[], b"").await.status, 200);
    assert_eq!(srv.send("POST", "/v1/load", &[], b"").await.status, 200);
    srv.stop();
    assert_eq!(srv.exited(WAIT).await, Some(Ok(())));
    let seen: Vec<(String, Option<String>)> = events(&srv)
        .into_iter()
        .map(|(e, l)| (e, l["cause"].as_str().map(String::from)))
        .collect();
    let named = |e: &str, c: Option<&str>| (e.to_string(), c.map(String::from));
    assert_eq!(
        seen,
        [
            named("start", None),
            named("load", None),
            named("device_lost", None),
            named("unload", Some("request")),
            named("stats", None),
            named("load", None),
            named("unload", Some("shutdown")),
            named("stats", None),
            named("exit", None),
        ]
    );
    let start = lines_of(&srv, "start").remove(0);
    assert_eq!(start["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(start["model_id"], "qwen3.6-35b-a3b");
    assert_eq!(start["bind"], srv.addr.to_string());
    let lost = lines_of(&srv, "device_lost").remove(0);
    assert!(lost["message"].as_str().unwrap().contains("DeviceLost"));
    assert_eq!(lines_of(&srv, "exit")[0]["reason"], "shutdown");
}

#[tokio::test]
async fn requests_refused_at_the_door_leave_a_line() {
    const PATH_A: &str = "/v1/MARKER-PATH-1c3d/x";
    const PATH_B: &str = "/v1/MARKER-PATH-2e4f";
    let mut setup = Setup::new();
    setup.key = Some(KEY);
    let srv = TestServer::start(setup).await;
    let auth = [("Authorization", format!("Bearer {KEY}"))];
    let auth: Vec<(&str, &str)> = auth.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let wrong = [("Authorization", "Bearer not-the-key")];
    let big = vec![b' '; (16 << 20) + 1];

    assert_eq!(srv.send("POST", CHAT, &wrong, b"{}").await.status, 401);
    assert_eq!(srv.send("POST", PATH_A, &[], b"{}").await.status, 401);
    assert_eq!(srv.send("POST", PATH_B, &auth, b"{}").await.status, 404);
    assert_eq!(srv.send("GET", CHAT, &auth, b"").await.status, 405);
    assert_eq!(srv.send("POST", CHAT, &auth, &big).await.status, 413);
    assert_eq!(srv.send("POST", "/v1/load", &auth, b"").await.status, 200);
    assert_eq!(srv.send("POST", "/v1/unload", &auth, b"").await.status, 200);
    assert_eq!(srv.send("POST", "/v1/load", &wrong, b"").await.status, 401);

    let got: Vec<(String, u64, Option<String>)> = request_lines(&srv)
        .iter()
        .map(|l| {
            assert_eq!(keys(l), FIELDS, "{l}");
            for k in FIELDS {
                let held = [
                    "time",
                    "id",
                    "endpoint",
                    "peer",
                    "status",
                    "total_ms",
                    "stream",
                    "tool_calls",
                    "malformed_calls",
                    "budget_closed",
                    "error_kind",
                    "error_message",
                    "ignored",
                    "ignored_more",
                    "repairs",
                    "repairs_more",
                    "dropped_messages",
                ];
                assert!(held.contains(&k) || l[k].is_null(), "{k} in {l}");
            }
            assert_eq!(l["error_kind"].is_string(), l["error_message"].is_string());
            // Nothing of the request was read: the flag and the counts are their defaults.
            assert_eq!((&l["stream"], &l["tool_calls"]), (&json!(false), &json!(0)));
            let rest = (&l["malformed_calls"], &l["budget_closed"]);
            assert_eq!(rest, (&json!(0), &json!(false)));
            assert!(
                l["id"].as_str().unwrap().starts_with("req_")
                    || l["id"].as_str().unwrap().starts_with("chatcmpl-")
            );
            (
                l["endpoint"].as_str().unwrap().to_string(),
                l["status"].as_u64().unwrap(),
                l["error_kind"].as_str().map(String::from),
            )
        })
        .collect();
    let line =
        |e: &str, status, kind: Option<&str>| (e.to_string(), status, kind.map(String::from));
    assert_eq!(
        got,
        [
            line(CHAT, 401, Some("authentication_error")),
            line("unknown", 401, Some("authentication_error")),
            line("unknown", 404, Some("not_found_error")),
            line(CHAT, 405, Some("invalid_request_error")),
            line(CHAT, 413, Some("invalid_request_error")),
            line("/v1/load", 200, None),
            line("/v1/unload", 200, None),
            line("/v1/load", 401, Some("authentication_error")),
        ]
    );
    for path in std::fs::read_dir(&srv.log_dir).unwrap() {
        let text = std::fs::read_to_string(path.unwrap().path()).unwrap();
        assert!(
            !text.contains("MARKER-PATH") && !text.contains(KEY),
            "{text}"
        );
    }

    // A Host the server does not serve (no key) is refused at the door too.
    let srv = TestServer::start(Setup::new()).await;
    let r = srv
        .send("POST", MSGS, &[("host", "evil.example")], b"{}")
        .await;
    assert_eq!(r.status, 403);
    let lines = request_lines(&srv);
    assert_eq!(lines.len(), 1);
    assert_eq!(
        (
            lines[0]["endpoint"].as_str(),
            lines[0]["status"].as_u64(),
            lines[0]["error_kind"].as_str()
        ),
        (Some(MSGS), Some(403), Some("permission_error"))
    );
}

// ---- no text ----

/// What the global tracing subscriber wrote.
fn traced() -> &'static Arc<Mutex<Vec<u8>>> {
    static BUF: OnceLock<Arc<Mutex<Vec<u8>>>> = OnceLock::new();
    BUF.get_or_init(|| {
        struct Capture(Arc<Mutex<Vec<u8>>>);
        impl Write for Capture {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let buf = Arc::new(Mutex::new(Vec::new()));
        let writer = buf.clone();
        tracing_subscriber::fmt()
            .with_writer(move || Capture(writer.clone()))
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .try_init()
            .expect("the first subscriber of this test binary");
        buf
    })
}

const KEY: &str = "sk-key-MARKER-9d1e-secretvalue";
const MARKERS: [&str; 14] = [
    "MARKER-DESC-1b72",
    "MARKER-ENUM-3d07",
    "MARKER-ENUM-3d08",
    "MARKER-KWARG-2c91",
    "MARKER-META-8e44",
    "MARKER-STOP-6a15",
    "MARKER-PROMPT-7f3a",
    "MARKER-THINK-91c2",
    "MARKER-OUT-55de",
    "MARKER-ARG-0b6e",
    "MARKER-BAD-4411",
    "MARKER-FIELD-52a8",
    "MARKER-TYPE-77e0",
    "MARKER-TYPE-91b4",
];

fn assert_clean(what: &str, text: &str) {
    for m in MARKERS.iter().chain([&KEY]) {
        assert!(!text.contains(m), "{what} holds {m}: {text}");
    }
}

#[tokio::test]
async fn logs_hold_no_prompt_output_or_key() {
    let trace = traced().clone();
    let mut setup = Setup::new();
    setup.key = Some(KEY);
    setup.fake_logs = true;
    let srv = TestServer::start(setup).await;
    let auth = [("Authorization", format!("Bearer {KEY}"))];
    let auth: Vec<(&str, &str)> = auth.iter().map(|(k, v)| (*k, v.as_str())).collect();

    let tools = json!([{"type": "function", "function": {
        "name": "get_weather", "description": "weather",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer"}}}}}]);
    let anthropic_tools = json!([{"name": "get_weather", "description": "weather",
        "input_schema": {"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer"}}}}]);
    // Reasoning, content and a call whose `days` is not an integer (a repair), all carrying markers.
    let output = "MARKER-THINK-91c2\n</think>\n\nMARKER-OUT-55de\n\n<tool_call>\n<function=get_weather>\n\
        <parameter=city>\nMARKER-ARG-0b6e\n</parameter>\n<parameter=days>\nMARKER-BAD-4411\n</parameter>\n\
        </function>\n</tool_call>";
    let script = || Script::text(tokenizer(), output);
    let prompt = "MARKER-PROMPT-7f3a";

    let mut errors: Vec<(u16, String)> = Vec::new();
    let mut note = |r: common::Resp| {
        errors.push((r.status, String::from_utf8_lossy(&r.body).into_owned()));
        r.status
    };

    // A 503 whose prompt holds the marker: the load fails.
    srv.fake.push(Script {
        load_fails: Some("scripted load failure".into()),
        ..Script::default()
    });
    let openai = |extra: Value| {
        let mut b = json!({"messages": [{"role": "user", "content": prompt}], "tools": tools,
            "reasoning_effort": "none", "zz_unknown": "MARKER-FIELD-52a8",
            "metadata": {"k": "MARKER-FIELD-52a8"}});
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        b.to_string().into_bytes()
    };
    let anthropic = |extra: Value| {
        let mut b = json!({"model": "claude-x", "max_tokens": 256, "tools": anthropic_tools,
            "messages": [{"role": "user", "content": prompt}], "zz_unknown": "MARKER-FIELD-52a8"});
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        b.to_string().into_bytes()
    };
    assert_eq!(
        note(srv.send("POST", CHAT, &auth, &openai(json!({}))).await),
        503
    );

    // The four that produce the output: whole and streamed, on both APIs.
    for stream in [false, true] {
        srv.fake.push(script());
        if stream {
            let mut s = srv
                .stream(CHAT, &auth, &openai(json!({"stream": true})))
                .await;
            drain(&mut s).await;
        } else {
            assert_eq!(
                srv.send("POST", CHAT, &auth, &openai(json!({})))
                    .await
                    .status,
                200
            );
        }
        srv.fake.push(script());
        if stream {
            let mut s = srv
                .stream(MSGS, &auth, &anthropic(json!({"stream": true})))
                .await;
            drain(&mut s).await;
        } else {
            assert_eq!(
                srv.send("POST", MSGS, &auth, &anthropic(json!({})))
                    .await
                    .status,
                200
            );
        }
    }

    // Wrongly typed fields: the 400s name the field and the type, never the value.
    let wrong = [
        (CHAT, openai(json!({"tools": "MARKER-TYPE-77e0"}))),
        (
            CHAT,
            openai(json!({"messages": [{"role": "user", "content": {"MARKER-TYPE-91b4": 1}}]})),
        ),
        (MSGS, anthropic(json!({"tools": "MARKER-TYPE-77e0"}))),
        (
            MSGS,
            anthropic(json!({"messages": [{"role": "user", "content": {"MARKER-TYPE-91b4": 1}}]})),
        ),
    ];
    for (path, body) in &wrong {
        assert_eq!(
            note(srv.send("POST", path, &auth, body).await),
            400,
            "{path}"
        );
    }
    // A wrong key, and a body that is not JSON at all.
    let bad = [("Authorization", "Bearer MARKER-PROMPT-7f3a")];
    assert_eq!(
        note(srv.send("POST", CHAT, &bad, &openai(json!({}))).await),
        401
    );
    assert_eq!(
        note(srv.send("POST", CHAT, &auth, b"MARKER-PROMPT-7f3a {").await),
        400
    );

    // One line each for the ten that reached a handler, and the 401 that stopped at the door.
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 11).await);
    let lines = request_lines(&srv);
    let repairs: Vec<&Value> = lines
        .iter()
        .filter(|l| !l["repairs"].as_array().unwrap().is_empty())
        .collect();
    assert_eq!(
        repairs.len(),
        4,
        "the bad value was repaired in each reply that carried it"
    );
    assert!(lines.iter().all(|l| l["ignored"].as_array().is_some()));

    // The file, the tracing output and every error body.
    for path in std::fs::read_dir(&srv.log_dir).unwrap() {
        let path = path.unwrap().path();
        assert_clean("a log file", &std::fs::read_to_string(path).unwrap());
    }
    srv.stop();
    for (status, body) in &errors {
        assert_clean(&format!("the {status} body"), body);
    }
    let text = String::from_utf8_lossy(&trace.lock().unwrap()).into_owned();
    assert!(
        text.contains("fields ignored"),
        "the subscriber saw the server: {text}"
    );
    assert_clean("the tracing output", &text);
}

// ---- the error message and the decode figures ----

/// The first 300 characters of `s`, as the line cuts a message.
fn cut300(s: &str) -> String {
    s.chars().take(server::logs::ERROR_MESSAGE_CHARS).collect()
}

/// The request lines that carry an `error_kind`.
fn error_lines(srv: &TestServer) -> Vec<Value> {
    request_lines(srv)
        .into_iter()
        .filter(|l| !l["error_kind"].is_null())
        .collect()
}

#[tokio::test]
async fn every_error_line_carries_its_message() {
    let mut setup = Setup::new();
    setup.key = Some(KEY);
    setup.fake_logs = true;
    let srv = TestServer::start(setup).await;
    let auth = [("Authorization", format!("Bearer {KEY}"))];
    let auth: Vec<(&str, &str)> = auth.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let message = |r: &common::Resp| r.json()["error"]["message"].as_str().unwrap().to_string();

    // A wrong-typed field (400), a missing key (401), a path nobody serves (404), a failed load (503).
    let r400 = srv
        .send("POST", CHAT, &auth, br#"{"messages": "nope"}"#)
        .await;
    assert_eq!(r400.status, 400);
    let r401 = srv.send("POST", CHAT, &[], &chat_body(json!({}))).await;
    assert_eq!(r401.status, 401);
    let r404 = srv.send("GET", "/v1/nowhere", &auth, b"").await;
    assert_eq!(r404.status, 404);
    srv.fake.push(Script {
        load_fails: Some("scripted load failure".into()),
        ..Script::default()
    });
    let r503 = srv.send("POST", CHAT, &auth, &chat_body(json!({}))).await;
    assert_eq!(r503.status, 503);

    // A failure after the headers went out: the line keeps status 200.
    srv.fake.push(Script {
        fail_at: Some((2, Fail::Internal)),
        ..Script::text(tokenizer(), "one two three four five")
    });
    let mut s = srv
        .stream(CHAT, &auth, &chat_body(json!({"stream": true})))
        .await;
    assert_eq!(s.status(), 200);
    drain(&mut s).await;

    assert!(wait_until(WAIT, || request_lines(&srv).len() == 5).await);
    let lines = request_lines(&srv);
    let [l400, l401, l404, l503, mid] = &lines[..] else {
        panic!("{lines:?}")
    };
    for (line, r, status) in [
        (l400, &r400, 400),
        (l401, &r401, 401),
        (l404, &r404, 404),
        (l503, &r503, 503),
    ] {
        assert_eq!(line["status"], status, "{line}");
        assert_eq!(line["error_message"], cut300(&message(r)), "{line}");
    }
    assert_eq!(l503["error_message"], "scripted load failure");
    assert_eq!(mid["status"], 200);
    assert_eq!(mid["error_kind"], "server_error");
    assert_eq!(mid["error_message"], "scripted failure: Internal");

    // A request the client abandons has neither field.
    srv.fake.push(Script {
        per_token: Duration::from_millis(20),
        ..Script::text(tokenizer(), &"word ".repeat(400))
    });
    srv.send_then_drop(
        "POST",
        CHAT,
        &auth,
        &chat_body(json!({})),
        Duration::from_millis(200),
    )
    .await;
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 6).await);
    let gone = request_lines(&srv).remove(5);
    assert_eq!(gone["status"], 499);
    assert!(gone["error_kind"].is_null() && gone["error_message"].is_null());
    // The lines that were not errors have no message either.
    assert_eq!(error_lines(&srv).len(), 5);
    srv.shutdown().await.unwrap();

    // 503: the queue is full (64 wait, the 65th is refused).
    let mut setup = Setup::new();
    setup.fake.start_loaded = true;
    let srv = Arc::new(TestServer::start(setup).await);
    srv.fake.push(Script {
        // Long enough that filling 64 places, one observed at a time, cannot overrun it on a loaded machine.
        prefill: Duration::from_secs(10),
        ..Script::text(tokenizer(), "slow")
    });
    let post = |srv: &Arc<TestServer>| {
        let srv = srv.clone();
        tokio::spawn(async move { srv.send("POST", CHAT, &[], &chat_body(json!({}))).await })
    };
    let blocker = post(&srv);
    assert!(wait_until(WAIT, || srv.fake.requests().len() == 1).await);
    let waiters: Vec<_> = (0..64).map(|_| post(&srv)).collect();
    assert!(wait_until(WAIT, || srv.waiting() == 64).await);
    let full = srv.send("POST", CHAT, &[], &chat_body(json!({}))).await;
    assert_eq!(full.status, 503);
    assert_eq!(message(&full), "the queue is full");
    assert_eq!(blocker.await.unwrap().status, 200);
    for w in waiters {
        assert_eq!(w.await.unwrap().status, 200);
    }
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 66).await);
    let errors = error_lines(&srv);
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0]["status"], 503);
    assert_eq!(errors[0]["error_message"], "the queue is full");
    Arc::into_inner(srv).unwrap().shutdown().await.unwrap();
}

#[tokio::test]
async fn decode_figures_are_logged() {
    let mut setup = Setup::new();
    setup.fake_logs = true;
    let srv = TestServer::start(setup).await;
    let word = tokenizer().encode("hi", false)[0];
    // Nine sampled tokens and the stop token: ten in all, nine of them decoded after the first.
    let script = |cache| Script {
        tokens: vec![word; 9],
        per_token: Duration::from_millis(3),
        cache,
        ..Script::default()
    };
    srv.fake.push(script(None));
    let r = srv.send("POST", CHAT, &[], &chat_body(json!({}))).await;
    assert_eq!(r.status, 200);
    srv.fake.push(script(Some(CacheReport {
        hit_rate: Some(0.5),
        miss_mib_per_token: Some(1.0),
        restore_bytes: 0,
        pool_bytes_used: 0,
        decode_misses: Some(7),
    })));
    let r = srv.send("POST", CHAT, &[], &chat_body(json!({}))).await;
    assert_eq!(r.status, 200);
    // A failed request had no decode to measure.
    let r = srv
        .send("POST", CHAT, &[], br#"{"messages": "nope"}"#)
        .await;
    assert_eq!(r.status, 400);
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 3).await);

    let lines = request_lines(&srv);
    for (line, misses) in lines[..2].iter().zip([Value::Null, json!(7)]) {
        assert_eq!(keys(line), FIELDS);
        assert_eq!(num(line, "completion_tokens"), 10.0, "{line}");
        assert_eq!(line["decode_tokens"], 9, "{line}");
        assert_eq!(line["decode_misses"], misses, "{line}");
        // The fake sleeps 3 ms for each of the ten tokens.
        let ms = num(line, "decode_ms");
        assert!(ms >= 27.0, "{line}");
        // `decode_tok_s` is unchanged: the completion over the decode time.
        let rate = num(line, "decode_tok_s");
        assert!((rate - 10.0 / (ms / 1000.0)).abs() < 1e-6 * rate, "{line}");
    }
    for k in ["decode_ms", "decode_tokens", "decode_misses"] {
        assert!(lines[2][k].is_null(), "{k}: {}", lines[2]);
    }
    srv.shutdown().await.unwrap();
}

const SPEC_KEYS: [&str; 10] = [
    "spec_steps",
    "spec_plain_steps",
    "spec_drafted",
    "spec_accepted",
    "spec_accept_rate",
    "spec_replays",
    "spec_drafted_lookup",
    "spec_accepted_lookup",
    "spec_drafted_mtp",
    "spec_accepted_mtp",
];

/// The speculation counts reach the line on both paths (whole and streamed), and a request that drafted nothing has
/// no accept rate (`spec_accept_rate` is accepted over drafted).
#[tokio::test]
async fn per_source_counts_are_logged() {
    let mut setup = Setup::new();
    setup.fake_logs = true;
    let srv = TestServer::start(setup).await;
    let script = |spec| Script {
        spec,
        ..Script::text(tokenizer(), "ok")
    };
    srv.fake.push(script(Some(SpecReport {
        steps: 5,
        plain_steps: 2,
        drafted: 12,
        accepted: 9,
        replays: 3,
        drafted_lookup: 7,
        accepted_lookup: 6,
        drafted_mtp: 5,
        accepted_mtp: 3,
    })));
    let r = srv.send("POST", CHAT, &[], &chat_body(json!({}))).await;
    assert_eq!(r.status, 200);
    // The streamed path reads the same `Done`; this request ran steps and drafted none.
    srv.fake.push(script(Some(SpecReport {
        steps: 0,
        plain_steps: 4,
        drafted: 0,
        accepted: 0,
        replays: 0,
        ..SpecReport::default()
    })));
    let mut s = srv
        .stream(MSGS, &[], &msgs_body(json!({"stream": true})))
        .await;
    assert_eq!(s.status(), 200);
    drain(&mut s).await;
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 2).await);

    let lines = request_lines(&srv);
    assert_eq!(keys(&lines[0]), FIELDS);
    assert_eq!(lines[0]["spec_steps"], 5);
    assert_eq!(lines[0]["spec_plain_steps"], 2);
    assert_eq!(lines[0]["spec_drafted"], 12);
    assert_eq!(lines[0]["spec_accepted"], 9);
    assert_eq!(num(&lines[0], "spec_accept_rate"), 0.75);
    assert_eq!(lines[0]["spec_replays"], 3);
    assert_eq!(lines[0]["spec_drafted_lookup"], 7);
    assert_eq!(lines[0]["spec_accepted_lookup"], 6);
    assert_eq!(lines[0]["spec_drafted_mtp"], 5);
    assert_eq!(lines[0]["spec_accepted_mtp"], 3);
    assert_eq!(lines[1]["spec_drafted_mtp"], 0);
    assert_eq!(lines[1]["spec_plain_steps"], 4);
    assert_eq!(lines[1]["spec_drafted"], 0);
    assert!(lines[1]["spec_accept_rate"].is_null(), "{}", lines[1]);
    srv.shutdown().await.unwrap();
}

/// With speculation off the engine sends no counts: all six are `null`, a refusal's line included.
#[tokio::test]
async fn per_source_counts_are_null_with_speculation_off() {
    let mut setup = Setup::new();
    setup.fake_logs = true;
    let srv = TestServer::start(setup).await;
    let r = srv.send("POST", CHAT, &[], &chat_body(json!({}))).await;
    assert_eq!(r.status, 200);
    let r = srv
        .send("POST", CHAT, &[], br#"{"messages": "nope"}"#)
        .await;
    assert_eq!(r.status, 400);
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 2).await);
    for line in request_lines(&srv) {
        assert_eq!(keys(&line), FIELDS);
        for k in SPEC_KEYS {
            assert!(line[k].is_null(), "{k}: {line}");
        }
    }
    srv.shutdown().await.unwrap();
}

/// Every leaf under `v` is a number or null (a key's name is not a leaf).
fn only_numbers(v: &Value, at: &str) {
    match v {
        Value::Null | Value::Number(_) => {}
        Value::Array(a) => a.iter().for_each(|x| only_numbers(x, at)),
        Value::Object(o) => o
            .iter()
            .for_each(|(k, x)| only_numbers(x, &format!("{at}.{k}"))),
        other => panic!("{at} is {other}, not a number"),
    }
}

/// The marker test, extended to the speculation counts: a request line's `spec_*` fields and the stats'
/// `SpecCounters` hold numbers and nothing else, so no draft text can be in them.
#[test]
fn the_spec_report_carries_numbers_only() {
    use server::backend::{Counts, Timings};
    use server::logs::RequestLine;
    let counts = Counts {
        prompt: 10,
        reused: 0,
        prefilled: 10,
        completion: 5,
        reasoning: 0,
    };
    let timings = Timings {
        queue_ms: 1.0,
        load_ms: 0.0,
        restore_ms: 0.0,
        prefill_ms: 5.0,
        decode_ms: 50.0,
        ttft_ms: 6.0,
    };
    let cache = CacheReport {
        hit_rate: None,
        miss_mib_per_token: None,
        restore_bytes: 0,
        pool_bytes_used: 0,
        decode_misses: None,
    };
    let report = SpecReport {
        steps: 5,
        plain_steps: 2,
        drafted: 12,
        accepted: 9,
        replays: 3,
        drafted_lookup: 7,
        accepted_lookup: 6,
        drafted_mtp: 5,
        accepted_mtp: 3,
    };
    let mut line = RequestLine::new("x", "/v1/x");
    line.measured(&counts, &timings, &cache, Some(&report));
    let v = serde_json::to_value(&line).unwrap();
    for k in SPEC_KEYS {
        assert!(v[k].is_number(), "{k}: {v}");
        only_numbers(&v[k], k);
    }
    let mut off = RequestLine::new("y", "/v1/x");
    off.measured(&counts, &timings, &cache, None);
    let v = serde_json::to_value(&off).unwrap();
    for k in SPEC_KEYS {
        only_numbers(&v[k], k);
    }

    let counters = SpecCounters {
        steps: 5,
        plain_steps: 2,
        drafted: 12,
        accepted: 9,
        replays: 3,
        accept_rate: [("lookup".to_string(), 0.75), ("mtp".to_string(), 0.7)].into(),
        by_size: vec![StepSize {
            t: 3,
            steps: 5,
            mean_ms: 21.5,
            mean_miss_mib: 40.0,
            tokens: 11,
        }],
        mtp_depth_rates: vec![0.7, 0.6],
        mtp_draft: vec![MtpDraftRow {
            k: 2,
            passes: 4,
            mean_ms: 3.5,
        }],
    };
    let v = serde_json::to_value(&counters).unwrap();
    only_numbers(&v, "spec");
    assert_eq!(v["accept_rate"]["lookup"], 0.75);
    assert_eq!(v["accept_rate"]["mtp"], 0.7);
    assert_eq!(v["mtp_depth_rates"], json!([0.7, 0.6]));
    assert_eq!(
        v["mtp_draft"],
        json!([{"k": 2, "passes": 4, "mean_ms": 3.5}])
    );
}

#[tokio::test]
async fn an_ignored_name_is_cut_to_40_characters() {
    let trace = traced().clone();
    let srv = TestServer::start(Setup::new()).await;
    let long = "k".repeat(200);
    let mut body: Value =
        serde_json::from_slice(&chat_body(json!({"frequency_penalty": 0.5}))).unwrap();
    body[&long] = json!(1);
    let r = srv
        .send("POST", CHAT, &[], body.to_string().as_bytes())
        .await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 1).await);
    let ignored = request_lines(&srv).remove(0)["ignored"].clone();
    assert_eq!(server::params::IGNORED_NAME_CHARS, 40);
    assert_eq!(ignored, json!(["frequency_penalty", "k".repeat(40)]));
    // The request's stderr line carries the count of ignored fields and no name: one line, whatever the keys.
    let text = String::from_utf8_lossy(&trace.lock().unwrap()).into_owned();
    assert!(text.contains("(2 fields ignored)"), "{text}");
    assert!(!text.contains("accepted and ignored"), "{text}");
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_long_model_name_is_cut_to_64_characters() {
    let srv = TestServer::start(Setup::new()).await;
    let model = "é".repeat(100);
    let r = srv
        .send("POST", CHAT, &[], &chat_body(json!({"model": model})))
        .await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let r = srv
        .send("POST", MSGS, &[], &msgs_body(json!({"model": model})))
        .await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let counted = srv
        .send(
            "POST",
            "/v1/messages/count_tokens",
            &[],
            &msgs_body(json!({"model": model})),
        )
        .await;
    assert_eq!(counted.status, 200);
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 3).await);
    assert_eq!(server::logs::MODEL_CHARS, 64);
    for line in request_lines(&srv) {
        let logged = line["model"].as_str().unwrap();
        assert_eq!(logged.chars().count(), 64, "{line}");
        assert_eq!(logged, "é".repeat(64));
    }
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_request_line_stays_small_whatever_the_request() {
    let srv = TestServer::start(Setup::new()).await;
    let mut body: serde_json::Map<String, Value> = serde_json::from_slice(&chat_body(json!({
        "model": "m".repeat(16_000)
    })))
    .unwrap();
    for i in 0..100_000 {
        body.insert(format!("{i:0>40}"), json!(0));
    }
    let body = Value::Object(body).to_string().into_bytes();
    let r = srv.send("POST", CHAT, &[], &body).await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    // The same keys on the endpoint that never queues.
    let mut body: serde_json::Map<String, Value> = serde_json::from_slice(&msgs_body(json!({
        "model": "m".repeat(16_000)
    })))
    .unwrap();
    for i in 0..100_000 {
        body.insert(format!("{i:0>40}"), json!(0));
    }
    let body = Value::Object(body).to_string().into_bytes();
    let r = srv
        .send("POST", "/v1/messages/count_tokens", &[], &body)
        .await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 2).await);
    // The files, not the parsed lines: the bytes the log holds for each request.
    for path in std::fs::read_dir(&srv.log_dir).unwrap() {
        let text = std::fs::read_to_string(path.unwrap().path()).unwrap();
        for line in text.lines().filter(|l| !l.contains("\"event\"")) {
            assert!(line.len() < 8192, "a request line of {} bytes", line.len());
        }
    }
    for line in request_lines(&srv) {
        assert_eq!(line["ignored"].as_array().unwrap().len(), 32, "{line}");
        assert_eq!(line["ignored_more"], 99_968, "{line}");
        assert_eq!(line["model"].as_str().unwrap().len(), 64);
    }
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_late_system_message_is_counted_as_dropped() {
    use server::convo::{dropped_by_template, Content, Message};
    let message = |role: &str| Message {
        role: role.into(),
        content: Content::Text("x".into()),
        reasoning_content: None,
        tool_calls: Vec::new(),
    };
    let count = |roles: &[&str]| {
        let messages: Vec<Message> = roles.iter().map(|r| message(r)).collect();
        dropped_by_template(&messages)
    };
    assert_eq!(count(&["system", "user", "assistant", "system", "user"]), 1);
    assert_eq!(count(&["system", "system", "user"]), 0);
    assert_eq!(count(&["user", "narrator"]), 1);
    assert_eq!(count(&["developer", "system", "user", "developer"]), 1);
    assert_eq!(count(&["user", "system", "user"]), 1);
    assert_eq!(count(&["system", "system", "system", "user"]), 1);
    assert_eq!(count(&["user", "assistant", "tool"]), 0);
    assert_eq!(count(&[]), 0);

    // Through the server: the reply is a normal 200, and the line and the stderr line say what was dropped.
    let trace = traced().clone();
    let srv = TestServer::start(Setup::new()).await;
    let body = json!({"reasoning_effort": "none", "messages": [
        {"role": "system", "content": "a"}, {"role": "user", "content": "hi"},
        {"role": "system", "content": "b"}, {"role": "narrator", "content": "c"},
        {"role": "user", "content": "again"}]});
    let r = srv
        .send("POST", CHAT, &[], body.to_string().as_bytes())
        .await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let r = srv.send("POST", CHAT, &[], &chat_body(json!({}))).await;
    assert_eq!(r.status, 200);
    assert!(wait_until(WAIT, || request_lines(&srv).len() == 2).await);
    let lines = request_lines(&srv);
    assert_eq!(lines[0]["dropped_messages"], 2, "{}", lines[0]);
    assert_eq!(lines[1]["dropped_messages"], 0, "{}", lines[1]);
    let text = String::from_utf8_lossy(&trace.lock().unwrap()).into_owned();
    assert!(
        text.contains("(2 messages dropped by the template)"),
        "{text}"
    );
    srv.shutdown().await.unwrap();
}

/// Markers in the values of requests that fail, one body for each place a request's value could reach a message. No
/// marker is ever an object key: the log carries unknown key names by design.
#[tokio::test]
async fn error_messages_never_carry_request_text() {
    let mut setup = Setup::new();
    setup.fake_logs = true;
    setup.config.ctx = 2048;
    let srv = TestServer::start(setup).await;
    let tools = json!([{"type": "function", "function": {
        "name": "lookup", "description": "MARKER-DESC-1b72",
        "parameters": {"type": "object", "properties": {
            "mode": {"type": "string", "enum": ["MARKER-ENUM-3d07", "MARKER-ENUM-3d08"]}}}}}]);
    let anthropic_tools = json!([{"name": "lookup", "description": "MARKER-DESC-1b72",
        "input_schema": {"type": "object", "properties": {
            "mode": {"type": "string", "enum": ["MARKER-ENUM-3d07", "MARKER-ENUM-3d08"]}}}}]);
    let openai = |extra: Value| {
        let mut b = json!({"messages": [{"role": "user", "content": "MARKER-PROMPT-7f3a"}],
            "tools": tools, "reasoning_effort": "none", "zz_unknown": "MARKER-FIELD-52a8",
            "metadata": {"k": "MARKER-META-8e44"}});
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        b.to_string().into_bytes()
    };
    let anthropic = |extra: Value| {
        let mut b = json!({"model": "claude-x", "max_tokens": 256, "tools": anthropic_tools,
            "messages": [{"role": "user", "content": "MARKER-PROMPT-7f3a"}],
            "zz_unknown": "MARKER-FIELD-52a8", "metadata": {"user_id": "MARKER-META-8e44"}});
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        b.to_string().into_bytes()
    };
    let long = json!([{"role": "user", "content": format!("MARKER-PROMPT-7f3a {}", "word ".repeat(3000))}]);

    // Requests that 400: a wrong-typed value in a string field, in `chat_template_kwargs`, in a stop list, in a
    // tool_choice, in a content part and in a system prompt; too many stop strings; and a prompt past the context,
    // with the tools and the markers in it.
    let wrong = [
        (CHAT, openai(json!({"model": {"k": "MARKER-FIELD-52a8"}}))),
        (
            CHAT,
            openai(json!({"reasoning_effort": {"k": "MARKER-TYPE-77e0"}})),
        ),
        (
            CHAT,
            openai(json!({"chat_template_kwargs": {"enable_thinking": "MARKER-KWARG-2c91"}})),
        ),
        (
            CHAT,
            openai(json!({"stop": ["MARKER-STOP-6a15", {"k": "MARKER-STOP-6a15"}]})),
        ),
        (CHAT, openai(json!({"stop": vec!["MARKER-STOP-6a15"; 17]}))),
        (CHAT, openai(json!({"tool_choice": "MARKER-FIELD-52a8"}))),
        (
            CHAT,
            openai(json!({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "MARKER-TYPE-91b4"}}]}]})),
        ),
        (CHAT, openai(json!({"messages": long}))),
        // A tool's own fields: its type, a wrong-typed description and name, a missing name; a schema that is no object
        // (taken as is) with a prompt past the context.
        (
            CHAT,
            openai(json!({"tools": [{"type": "MARKER-TYPE-77e0", "function": {
                "name": "lookup", "description": "MARKER-DESC-1b72"}}]})),
        ),
        (
            CHAT,
            openai(json!({"tools": [{"type": "function", "function": {
                "name": "lookup", "description": {"k": "MARKER-DESC-1b72"}}}]})),
        ),
        (
            CHAT,
            openai(json!({"tools": [{"type": "function", "function": {
                "name": {"k": "MARKER-DESC-1b72"}, "description": "MARKER-DESC-1b72"}}]})),
        ),
        (
            CHAT,
            openai(json!({"tools": [{"type": "function", "function": {
                "name": "lookup", "description": "MARKER-DESC-1b72",
                "parameters": "MARKER-ENUM-3d07"}}], "messages": long})),
        ),
        (
            MSGS,
            anthropic(json!({"tools": [{"name": "lookup", "description": ["MARKER-DESC-1b72"]}]})),
        ),
        (
            MSGS,
            anthropic(json!({"tools": [{"description": "MARKER-DESC-1b72",
                "input_schema": {"type": "object", "properties": {
                    "mode": {"enum": ["MARKER-ENUM-3d07"]}}}}]})),
        ),
        (
            MSGS,
            anthropic(
                json!({"tools": [{"name": "lookup", "description": "MARKER-DESC-1b72",
                "input_schema": "MARKER-ENUM-3d07"}], "messages": long}),
            ),
        ),
        (
            MSGS,
            anthropic(json!({"tool_choice": {"type": "MARKER-FIELD-52a8"}})),
        ),
        (
            MSGS,
            anthropic(json!({"system": {"k": "MARKER-TYPE-91b4"}})),
        ),
        (
            MSGS,
            anthropic(json!({"stop_sequences": ["MARKER-STOP-6a15", 5]})),
        ),
        (MSGS, anthropic(json!({"messages": long}))),
    ];
    let mut not_400 = Vec::new();
    for (i, (path, body)) in wrong.iter().enumerate() {
        let r = srv.send("POST", path, &[], body).await;
        if r.status != 400 {
            not_400.push((i, r.status));
        }
    }
    assert!(
        not_400.is_empty(),
        "the requests (index, status) that did not 400: {not_400:?}"
    );

    // A failed load, and a failure in the middle of a stream.
    srv.fake.push(Script {
        load_fails: Some("scripted load failure".into()),
        ..Script::default()
    });
    let r = srv.send("POST", CHAT, &[], &openai(json!({}))).await;
    assert_eq!(r.status, 503);
    srv.fake.push(Script {
        fail_at: Some((2, Fail::Internal)),
        ..Script::text(
            tokenizer(),
            "MARKER-OUT-55de MARKER-THINK-91c2 more words follow",
        )
    });
    let mut s = srv
        .stream(CHAT, &[], &openai(json!({"stream": true})))
        .await;
    assert_eq!(s.status(), 200);
    drain(&mut s).await;

    let n = wrong.len() + 2;
    assert!(wait_until(WAIT, || request_lines(&srv).len() == n).await);
    for path in std::fs::read_dir(&srv.log_dir).unwrap() {
        let path = path.unwrap().path();
        assert_clean("a log file", &std::fs::read_to_string(path).unwrap());
    }
    let lines = request_lines(&srv);
    assert_eq!(error_lines(&srv).len(), n);
    for l in &lines {
        let message = l["error_message"].as_str().unwrap_or_else(|| panic!("{l}"));
        assert!(!message.is_empty() && message.chars().count() <= 300, "{l}");
    }
    srv.shutdown().await.unwrap();
}
