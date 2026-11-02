//! `/v1/chat/completions`, non-streaming, end to end on the fake engine.
mod common;

use common::{specials, template, tokenizer, Resp, Setup, TestServer};
use serde_json::{json, Value};
use server::backend::Finish;
use server::config::Config;
use server::fake::Script;
use server::openai::request::parse;
use server::params::Defaults;
use server::prompt::build_prompt;
use std::time::Duration;

const CHAT: &str = "/v1/chat/completions";
const MODEL: &str = "qwen3.6-35b-a3b";

fn say(srv: &TestServer, text: &str) {
    srv.fake.push(Script::text(tokenizer(), text));
}

async fn post(srv: &TestServer, body: &Value) -> Resp {
    srv.send("POST", CHAT, &[], body.to_string().as_bytes())
        .await
}

fn hi() -> Value {
    json!([{"role": "user", "content": "hi"}])
}

/// The request lines the server has logged, oldest first (the lifecycle lines have an `event`).
fn log_lines(srv: &TestServer) -> Vec<Value> {
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
        .map(|l| serde_json::from_str::<Value>(&l).unwrap())
        .filter(|l| l.get("event").is_none())
        .collect()
}

fn prompt_text(srv: &TestServer, i: usize) -> String {
    tokenizer().decode(&srv.fake.requests()[i].1.prompt)
}

#[tokio::test]
async fn o1_non_streaming_shape() {
    let srv = TestServer::start(Setup::new()).await;
    say(&srv, "Reasoning.\n</think>\n\nHello there.");
    let body = json!({
        "messages": [
            {"role": "system", "content": "You are brief."},
            {"role": "user", "content": "Say hello."},
        ],
        "max_tokens": 64,
        "reasoning_effort": "low",
    });
    let r = post(&srv, &body).await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    assert!(r.headers.contains_key("content-length"));
    assert!(!r.headers.contains_key("transfer-encoding"));
    let ct = r.headers["content-type"].to_str().unwrap();
    assert!(ct.starts_with("application/json"), "{ct}");
    let b = r.json();
    assert!(b["id"].as_str().unwrap().starts_with("chatcmpl-"), "{b}");
    assert_eq!(b["id"].as_str().unwrap().len(), "chatcmpl-".len() + 24);
    assert_eq!(b["object"], "chat.completion");
    assert!(b["created"].as_u64().unwrap() > 1_700_000_000);
    assert_eq!(b["model"], MODEL);
    assert_eq!(b["choices"].as_array().unwrap().len(), 1);
    let c = &b["choices"][0];
    assert_eq!(c["index"], 0);
    assert_eq!(c["message"]["role"], "assistant");
    assert_eq!(c["message"]["content"], "Hello there.");
    assert_eq!(c["message"]["reasoning_content"], "Reasoning.");
    assert!(c["message"].get("tool_calls").is_none());
    assert!(c.get("logprobs").is_some() && c["logprobs"].is_null());
    assert_eq!(c["finish_reason"], "stop");
    let reqs = srv.fake.requests();
    assert_eq!(reqs.len(), 1);
    let u = &b["usage"];
    let prompt = reqs[0].1.prompt.len() as u64;
    assert_eq!(u["prompt_tokens"], prompt);
    let want_completion = tokenizer()
        .encode("Reasoning.\n</think>\n\nHello there.", true)
        .len() as u64
        + 1;
    assert_eq!(u["completion_tokens"], want_completion);
    assert_eq!(u["total_tokens"], prompt + want_completion);
    assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 0);
    // The server's own count: every token before `</think>` (the held newline included), the tag excluded.
    let ids = tokenizer().encode("Reasoning.\n</think>\n\nHello there.", true);
    let before_tag = ids
        .iter()
        .position(|&id| id == specials().end_think)
        .unwrap() as u64;
    assert!(before_tag >= 2);
    assert_eq!(
        u["completion_tokens_details"]["reasoning_tokens"],
        before_tag
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o1_reused_tokens_are_cached_tokens() {
    let srv = TestServer::start(Setup::new()).await;
    srv.fake.push(Script {
        reused: 7,
        ..Script::text(tokenizer(), "Hi")
    });
    let r = post(&srv, &json!({"messages": hi()})).await;
    assert_eq!(r.status, 200);
    assert_eq!(
        r.json()["usage"]["prompt_tokens_details"]["cached_tokens"],
        7
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o2_effort_none_renders_thinking_off() {
    let srv = TestServer::start(Setup::new()).await;
    // The reference prompt: the same body, thinking off by the template's own variable.
    let reference = |extra: Value| {
        let mut body = json!({"messages": hi()});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let mut req = parse(body.to_string().as_bytes()).unwrap();
        let d = Defaults::new(MODEL, 65_536);
        let f = req.fields(&d).unwrap();
        let c = req.conversation(&f).unwrap();
        build_prompt(&c, template(), tokenizer(), specials(), 65_536)
            .unwrap()
            .tokens
    };
    let off = reference(json!({"chat_template_kwargs": {"enable_thinking": false}}));
    let on = reference(json!({}));
    assert_ne!(off, on, "the two renders differ, or this proves nothing");

    say(&srv, "Hello");
    let r = post(&srv, &json!({"messages": hi(), "reasoning_effort": "none"})).await;
    assert_eq!(r.status, 200);
    let b = r.json();
    let content = b["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(content, "Hello");
    assert!(!content.contains("<think>"));
    assert!(b["choices"][0]["message"]
        .get("reasoning_content")
        .is_none());
    assert_eq!(srv.fake.requests()[0].1.prompt, off);
    assert!(srv.fake.requests()[0].1.think.is_none());

    // The default thinks: the generation prompt differs.
    say(&srv, "Plan\n</think>\n\nHello");
    let r = post(&srv, &json!({"messages": hi()})).await;
    assert_eq!(r.status, 200);
    assert_eq!(srv.fake.requests()[1].1.prompt, on);

    // One token of room: the answer is cut and says so.
    let r = post(
        &srv,
        &json!({"messages": hi(), "reasoning_effort": "none", "max_tokens": 1}),
    )
    .await;
    assert_eq!(r.status, 200);
    let b = r.json();
    assert_eq!(b["choices"][0]["finish_reason"], "length");
    assert_eq!(b["usage"]["completion_tokens"], 1);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o3_budgets_and_the_split_on_the_wire() {
    let srv = TestServer::start(Setup::new()).await;
    for (i, (effort, want)) in [
        ("low", Some(Some(1024))),
        ("medium", Some(Some(4096))),
        ("high", Some(None)),
        ("none", None),
    ]
    .into_iter()
    .enumerate()
    {
        let r = post(&srv, &json!({"messages": hi(), "reasoning_effort": effort})).await;
        assert_eq!(r.status, 200, "{effort}");
        let think = srv.fake.requests()[i].1.think.clone();
        assert_eq!(think.map(|t| t.budget), want, "{effort}");
    }

    // Thinking on: the reasoning is the text before `</think>`.
    say(&srv, "plan\n</think>\n\nAnswer");
    let b = post(&srv, &json!({"messages": hi()})).await.json();
    assert_eq!(b["choices"][0]["message"]["reasoning_content"], "plan");
    assert_eq!(b["choices"][0]["message"]["content"], "Answer");

    // Thinking off: a `<think>` the model opens anyway still splits.
    say(&srv, "Hi <think>secret</think> there");
    let b = post(&srv, &json!({"messages": hi(), "reasoning_effort": "none"}))
        .await
        .json();
    let m = &b["choices"][0]["message"];
    assert_eq!(m["reasoning_content"], "secret");
    let content = m["content"].as_str().unwrap();
    for bad in ["<think>", "</think>", "secret"] {
        assert!(!content.contains(bad), "{content:?}");
    }
    assert!(
        content.contains("Hi") && content.contains("there"),
        "{content:?}"
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o4_models_unknown_fields_rejections() {
    let srv = TestServer::start(Setup::new()).await;
    for model in ["some-model", "qwen-coder", "claude-x", ""] {
        let r = post(&srv, &json!({"model": model, "messages": hi()})).await;
        assert_eq!(r.status, 200, "{model}");
        assert_eq!(r.json()["model"], model);
    }
    let r = post(&srv, &json!({"messages": hi()})).await;
    assert_eq!(r.json()["model"], MODEL);

    let ok = [
        json!({"foo": 1}),
        json!({"frequency_penalty": 0.5}),
        // OpenCode 2.0.30's top-level fields, but for `stream`: all accepted.
        json!({
            "model": MODEL, "stream": false, "stream_options": {"include_usage": true},
            "max_completion_tokens": 16384, "store": false, "reasoning_effort": "low",
        }),
    ];
    for extra in ok {
        let mut body = json!({"messages": hi()});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let r = post(&srv, &body).await;
        assert_eq!(
            r.status,
            200,
            "{extra}: {}",
            String::from_utf8_lossy(&r.body)
        );
    }

    let sent = srv.fake.requests().len();
    let bad = [
        (json!({"n": 2}), "n"),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}}),
            "response_format",
        ),
        (
            json!({"logprobs": true, "top_logprobs": 21}),
            "top_logprobs",
        ),
    ];
    for (extra, field) in bad {
        let mut body = json!({"messages": hi()});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let r = post(&srv, &body).await;
        assert_eq!(r.status, 400, "{field}");
        let e = &r.json()["error"];
        assert_eq!(e["type"], "invalid_request_error");
        assert_eq!(e["param"], field);
        assert!(e["message"].as_str().unwrap().contains(field), "{e}");
    }
    assert_eq!(
        srv.fake.requests().len(),
        sent,
        "a 400 never reaches the engine"
    );
    srv.shutdown().await.unwrap();
}

fn weather() -> Value {
    json!([{"type": "function", "function": {
        "name": "get_weather", "description": "The weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }}])
}

#[tokio::test]
async fn o9_tool_choice_on_the_wire() {
    let srv = TestServer::start(Setup::new()).await;
    let end_tool_call = specials().end_tool_call;
    let with = |extra: Value| {
        let mut body = json!({"messages": hi(), "tools": weather()});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        body
    };
    // auto: the tools are rendered.
    assert_eq!(post(&srv, &with(json!({}))).await.status, 200);
    assert!(prompt_text(&srv, 0).contains("<tools>"));
    let auto = srv.fake.requests()[0].1.clone();
    assert!(!auto.stop_tokens.contains(&end_tool_call));

    // none: the tools are left out of the prompt.
    assert_eq!(
        post(&srv, &with(json!({"tool_choice": "none"})))
            .await
            .status,
        200
    );
    assert!(!prompt_text(&srv, 1).contains("<tools>"));

    // required and a named function render as auto, and the log names the ignored field.
    let named = json!({"type": "function", "function": {"name": "get_weather"}});
    for (i, choice) in [json!("required"), named].into_iter().enumerate() {
        let r = post(&srv, &with(json!({"tool_choice": choice}))).await;
        assert_eq!(r.status, 200);
        assert_eq!(srv.fake.requests()[2 + i].1.prompt, auto.prompt);
    }
    let lines = log_lines(&srv);
    assert_eq!(lines.len(), 4);
    for (i, l) in lines.iter().enumerate() {
        let ignored: Vec<&str> = l["ignored"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(ignored.contains(&"tool_choice"), i >= 2, "line {i}: {l}");
    }

    // parallel_tool_calls false: `</tool_call>` stops the request, when tools were rendered.
    assert_eq!(
        post(&srv, &with(json!({"parallel_tool_calls": false})))
            .await
            .status,
        200
    );
    assert!(srv.fake.requests()[4]
        .1
        .stop_tokens
        .contains(&end_tool_call));
    // With `tool_choice: none` nothing was rendered, so there is no call to stop at.
    let body = with(json!({"parallel_tool_calls": false, "tool_choice": "none"}));
    assert_eq!(post(&srv, &body).await.status, 200);
    assert!(!srv.fake.requests()[5]
        .1
        .stop_tokens
        .contains(&end_tool_call));
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o10_over_context_is_400_with_both_counts() {
    let mut s = Setup::new();
    s.config = Config {
        ctx: 64,
        ..s.config
    };
    let srv = TestServer::start(s).await;
    let long = "word ".repeat(200);
    let body = json!({"messages": [{"role": "user", "content": long}]});
    let r = post(&srv, &body).await;
    assert_eq!(r.status, 400);
    assert!(r.body.len() <= 500, "{} bytes", r.body.len());
    let e = &r.json()["error"];
    assert_eq!(e["code"], "context_length_exceeded");
    assert_eq!(e["type"], "invalid_request_error");
    let message = e["message"].as_str().unwrap();
    let mut req = parse(body.to_string().as_bytes()).unwrap();
    let d = Defaults::new(MODEL, u32::MAX);
    let f = req.fields(&d).unwrap();
    let c = req.conversation(&f).unwrap();
    let tokens = build_prompt(&c, template(), tokenizer(), specials(), u32::MAX)
        .unwrap()
        .tokens
        .len();
    assert!(tokens > 64);
    assert!(message.contains(&tokens.to_string()), "{message}");
    assert!(message.contains("64"), "{message}");
    assert!(srv.fake.requests().is_empty());
    assert!(
        srv.fake.loads().is_empty(),
        "nothing was queued, so nothing loaded"
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o12_logprobs_non_streaming() {
    let srv = TestServer::start(Setup::new()).await;
    say(&srv, "Thinking\n</think>\n\nHi there");
    let r = post(
        &srv,
        &json!({"messages": hi(), "logprobs": true, "top_logprobs": 20}),
    )
    .await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let c = &r.json()["choices"][0];
    let content = c["message"]["content"].as_str().unwrap().to_string();
    assert_eq!(content, "Hi there");
    let entries = c["logprobs"]["content"].as_array().unwrap();
    assert!(!entries.is_empty());
    // Entries for the content tokens only: their text is the content, and the reasoning has none.
    let text: String = entries
        .iter()
        .map(|e| e["token"].as_str().unwrap())
        .collect();
    assert_eq!(text, content);
    for e in entries {
        assert!(e["logprob"].is_number());
        let bytes: Vec<u8> = e["bytes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_u64().unwrap() as u8)
            .collect();
        assert_eq!(bytes, e["token"].as_str().unwrap().as_bytes());
        let top = e["top_logprobs"].as_array().unwrap();
        assert_eq!(top.len(), 20);
        for t in top {
            assert!(t["token"].is_string() && t["logprob"].is_number() && t["bytes"].is_array());
        }
    }
    // Without logprobs the field is present and null.
    let c = &post(&srv, &json!({"messages": hi()})).await.json()["choices"][0];
    assert!(c.get("logprobs").is_some() && c["logprobs"].is_null());
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o1_tool_calls_and_the_finish_reason() {
    let srv = TestServer::start(Setup::new()).await;
    let call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
    say(&srv, &format!("Look.\n</think>\n\n{call}"));
    let body = json!({"messages": hi(), "tools": weather()});
    let r = post(&srv, &body).await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let c = &r.json()["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls");
    // No text beside the call: `content` is null, not "".
    assert!(c["message"]["content"].is_null(), "{c}");
    let calls = c["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1);
    assert!(calls[0]["id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(calls[0]["id"].as_str().unwrap().len(), "call_".len() + 24);
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let args = calls[0]["function"]["arguments"].as_str().unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(args).unwrap(),
        json!({"city": "Paris"})
    );

    // Text before the call stays content, with the calls beside it.
    say(&srv, &format!("Look.\n</think>\n\nChecking. {call}"));
    let c = post(&srv, &body).await.json()["choices"][0].clone();
    assert_eq!(c["finish_reason"], "tool_calls");
    assert_eq!(c["message"]["content"], "Checking. ");
    assert_eq!(c["message"]["tool_calls"].as_array().unwrap().len(), 1);

    // The same text without declared tools is content, and an ordinary stop.
    say(&srv, &format!("Look.\n</think>\n\n{call}"));
    let c = post(&srv, &json!({"messages": hi()})).await.json()["choices"][0].clone();
    assert_eq!(c["finish_reason"], "stop");
    assert!(c["message"].get("tool_calls").is_none());
    assert!(c["message"]["content"]
        .as_str()
        .unwrap()
        .contains("<tool_call>"));
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o11_a_stop_string_ends_the_answer() {
    let srv = TestServer::start(Setup::new()).await;
    // A tail of ten words after the match, at 100 ms a token: the engine must be cancelled, not read to the end.
    let text = "x\n</think>\n\nOne two STOP three four five six seven eight nine ten eleven twelve";
    srv.fake.push(Script {
        per_token: Duration::from_millis(100),
        ..Script::text(tokenizer(), text)
    });
    let b = post(&srv, &json!({"messages": hi(), "stop": ["STOP"]}))
        .await
        .json();
    let c = &b["choices"][0];
    assert_eq!(c["message"]["content"], "One two ");
    assert_eq!(c["finish_reason"], "stop");
    // The token that completes the match; at most the one already in flight follows it.
    let ids = tokenizer().encode(text, true);
    let k = (1..=ids.len())
        .find(|&k| tokenizer().decode(&ids[..k]).contains("STOP"))
        .unwrap();
    assert!(ids.len() >= k + 8, "a long tail after the match");
    let runs = srv.fake.runs();
    assert_eq!(runs[0].finish, Some(Finish::Cancelled));
    assert!(
        runs[0].token_times.len() <= k + 1,
        "{} tokens",
        runs[0].token_times.len()
    );
    assert!(!srv.fake.cancellations().is_empty());
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_failed_request_still_leaves_a_log_line() {
    let srv = TestServer::start(Setup::new()).await;
    say(&srv, "Hello");
    let r = post(&srv, &json!({"model": "m", "messages": hi()})).await;
    assert_eq!(r.status, 200);
    let bad = post(&srv, &json!({"messages": "no"})).await;
    assert_eq!(bad.status, 400);
    let lines = log_lines(&srv);
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["endpoint"], CHAT);
    assert_eq!(lines[0]["status"], 200);
    assert_eq!(lines[0]["model"], "m");
    assert_eq!(lines[0]["finish"], "stop");
    assert!(lines[0]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(lines[0]["completion_tokens"].as_u64().unwrap() > 0);
    assert_eq!(lines[1]["status"], 400);
    assert!(lines[1]["prompt_tokens"].is_null());
    // No prompt, no output.
    let text = lines.iter().map(Value::to_string).collect::<String>();
    assert!(!text.contains("Hello") && !text.contains("\"hi\""));
    srv.shutdown().await.unwrap();
}

/// The value cap and the preparation permits must let a large legitimate agent request through. The
/// context is raised so the prompts fit; a refusal for the context's sake would say so, and the cap's says "JSON
/// values".
#[tokio::test]
async fn a_large_legitimate_request_passes() {
    let mut setup = Setup::new();
    setup.config.ctx = 8_000_000;
    let srv = TestServer::start(setup).await;

    // About 15 MiB: 300 messages of 50,000 characters each.
    let text = "a".repeat(50_000);
    let messages: Vec<Value> = (0..300)
        .map(|i| json!({"role": if i % 2 == 0 {"user"} else {"assistant"}, "content": text}))
        .collect();
    let mut messages = messages;
    messages.push(json!({"role": "user", "content": "go"}));
    say(&srv, "ok");
    let body = json!({"messages": messages, "reasoning_effort": "none"});
    let bytes = body.to_string();
    assert!(
        bytes.len() > 14 << 20 && bytes.len() < 16 << 20,
        "{}",
        bytes.len()
    );
    let r = srv.send("POST", CHAT, &[], bytes.as_bytes()).await;
    let shown = String::from_utf8_lossy(&r.body)
        .chars()
        .take(300)
        .collect::<String>();
    assert_eq!(r.status, 200, "{shown}");

    // One tool whose parameters hold 200,000 values.
    let options: Vec<u32> = (0..200_000).collect();
    let tools = json!([{"type": "function", "function": {
        "name": "pick", "description": "Picks.",
        "parameters": {"type": "object", "properties": {"n": {"type": "integer", "enum": options}}},
    }}]);
    say(&srv, "ok");
    let r = post(
        &srv,
        &json!({"messages": hi(), "tools": tools, "reasoning_effort": "none"}),
    )
    .await;
    let shown = String::from_utf8_lossy(&r.body)
        .chars()
        .take(300)
        .collect::<String>();
    assert!(!shown.contains("JSON values"), "{shown}");
    assert_eq!(r.status, 200, "{shown}");
}

#[tokio::test]
async fn a_body_over_the_value_cap_is_a_400_on_the_wire() {
    let srv = TestServer::start(Setup::new()).await;
    let mut body = String::from(r#"{"messages": [{"role": "user", "content": "hi"}], "x": ["#);
    body.push_str(&"0,".repeat(1_000_000));
    body.push_str("0]}");
    let r = srv.send("POST", CHAT, &[], body.as_bytes()).await;
    assert_eq!(r.status, 400);
    assert!(r.json()["error"]["message"]
        .as_str()
        .unwrap()
        .contains("1,000,000"));
    assert!(srv.fake.requests().is_empty());
}

#[tokio::test]
async fn a_huge_whitespace_run_is_a_400_invalid_request() {
    let srv = TestServer::start(Setup::new()).await;
    let r = post(
        &srv,
        // The template trims the ends of a message, so the run sits between two words.
        &json!({"messages": [{"role": "user", "content": format!("a{}b", " ".repeat(300_000))}]}),
    )
    .await;
    assert_eq!(r.status, 400, "{}", String::from_utf8_lossy(&r.body));
    let e = &r.json()["error"];
    assert_eq!(e["type"], "invalid_request_error");
    assert!(e["message"].as_str().unwrap().contains("whitespace"));
    assert!(srv.fake.requests().is_empty());
}
