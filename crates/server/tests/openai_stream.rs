//! `/v1/chat/completions` streaming (SSE), tool calls on the wire, keep-alives and disconnects, end to end on the fake
//! engine.
mod common;

use common::{specials, tokenizer, wait_until, Resp, Setup, Sse, SseItem, TestServer, TESTDATA};
use serde_json::{json, Value};
use server::app::Exit;
use server::backend::{Backend, Finish};
use server::fake::{Fail, Script};
use std::time::{Duration, Instant};

const CHAT: &str = "/v1/chat/completions";
const MODEL: &str = "qwen3.6-35b-a3b";
const WAIT: Duration = Duration::from_secs(60);

fn say(srv: &TestServer, text: &str) {
    srv.fake.push(Script::text(tokenizer(), text));
}

fn hi() -> Value {
    json!([{"role": "user", "content": "hi"}])
}

/// Adds `stream: true` to `body` and opens the stream.
async fn open(srv: &TestServer, body: &Value) -> Sse {
    let mut body = body.clone();
    body["stream"] = json!(true);
    srv.stream(CHAT, &[], body.to_string().as_bytes()).await
}

async fn post(srv: &TestServer, body: &Value) -> Resp {
    srv.send("POST", CHAT, &[], body.to_string().as_bytes())
        .await
}

/// A whole stream, read to `[DONE]`.
struct Streamed {
    /// The `data:` JSON of every event before `[DONE]`.
    chunks: Vec<Value>,
    /// How many `: keep-alive` comments came before the first chunk that carries `content`.
    keepalives_before_content: usize,
    keepalives: usize,
    /// Events after `[DONE]`, and whether it was seen at all.
    done: bool,
    after_done: usize,
}

/// Reads `sse` until `[DONE]` or the end of the body.
async fn drain(sse: &mut Sse) -> Streamed {
    let mut s = Streamed {
        chunks: vec![],
        keepalives_before_content: 0,
        keepalives: 0,
        done: false,
        after_done: 0,
    };
    let mut content_seen = false;
    loop {
        match sse.next(WAIT).await {
            None => {
                assert!(sse.ended(), "the stream stalled for {WAIT:?}");
                return s;
            }
            Some(SseItem::Comment(c)) => {
                assert_eq!(c, "keep-alive");
                s.keepalives += 1;
                if !content_seen {
                    s.keepalives_before_content += 1;
                }
            }
            Some(SseItem::Event { name, data }) => {
                assert_eq!(name, None);
                if s.done {
                    s.after_done += 1;
                } else if data == "[DONE]" {
                    s.done = true;
                } else {
                    let v: Value = serde_json::from_str(&data).unwrap_or_else(|e| {
                        panic!("a data line that is not JSON ({e}): {data}");
                    });
                    if delta_of(&v).is_some_and(|d| d.get("content").is_some_and(|c| c != "")) {
                        content_seen = true;
                    }
                    s.chunks.push(v);
                }
            }
        }
    }
}

fn delta_of(chunk: &Value) -> Option<&Value> {
    chunk["choices"].get(0).map(|c| &c["delta"])
}

/// One streamed call, rebuilt from its deltas.
#[derive(Debug, PartialEq)]
struct Call {
    id: String,
    name: String,
    arguments: String,
}

/// What a stream said, folded.
#[derive(Debug)]
struct Folded {
    content: String,
    reasoning: String,
    calls: Vec<Call>,
    finish: Option<String>,
    usage: Option<Value>,
}

impl Streamed {
    fn fold(&self) -> Folded {
        let mut f = Folded {
            content: String::new(),
            reasoning: String::new(),
            calls: vec![],
            finish: None,
            usage: None,
        };
        for c in &self.chunks {
            if let Some(u) = c.get("usage").filter(|u| !u.is_null()) {
                f.usage = Some(u.clone());
            }
            let Some(choice) = c["choices"].get(0) else {
                continue;
            };
            if let Some(r) = choice["finish_reason"].as_str() {
                f.finish = Some(r.to_string());
            }
            let d = &choice["delta"];
            if let Some(t) = d["content"].as_str() {
                f.content.push_str(t);
            }
            if let Some(t) = d["reasoning_content"].as_str() {
                f.reasoning.push_str(t);
            }
            for tc in d["tool_calls"].as_array().into_iter().flatten() {
                let index = tc["index"].as_u64().unwrap() as usize;
                if tc.get("id").is_some() {
                    assert_eq!(index, f.calls.len(), "calls open in order");
                    f.calls.push(Call {
                        id: tc["id"].as_str().unwrap().to_string(),
                        name: tc["function"]["name"].as_str().unwrap().to_string(),
                        arguments: String::new(),
                    });
                } else {
                    assert!(!tc["function"].as_object().unwrap().contains_key("name"));
                }
                f.calls[index]
                    .arguments
                    .push_str(tc["function"]["arguments"].as_str().unwrap());
            }
        }
        f
    }
}

/// Opens, drains and checks the framing every stream shares.
async fn streamed(srv: &TestServer, body: &Value) -> Streamed {
    let mut sse = open(srv, body).await;
    assert_eq!(sse.status(), 200);
    let s = drain(&mut sse).await;
    assert!(s.done, "no [DONE]");
    assert_eq!(s.after_done, 0);
    s
}

fn weather() -> Value {
    json!({"type": "function", "function": {
        "name": "get_weather", "description": "The weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }})
}

/// A tool whose parameters need a `$ref`, an `enum` without a type and an `anyOf`.
fn lookup() -> Value {
    json!({"type": "function", "function": {
        "name": "lookup", "description": "Looks something up.",
        "parameters": {"type": "object",
            "properties": {
                "id": {"$ref": "#/$defs/Id"},
                "mode": {"enum": ["fast", "slow"]},
                "tag": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
            },
            "$defs": {"Id": {"type": "integer"}}},
    }})
}

fn call_text(name: &str, params: &[(&str, &str)]) -> String {
    let mut s = format!("<tool_call>\n<function={name}>\n");
    for (k, v) in params {
        s.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
    }
    s.push_str("</function>\n</tool_call>");
    s
}

#[tokio::test]
async fn o5_sse_sequence_and_keepalive() {
    let mut setup = Setup::new();
    setup.tuning.keepalive = Duration::from_secs(1);
    let srv = TestServer::start(setup).await;
    srv.fake.push(Script {
        prefill: Duration::from_secs(25),
        ..Script::text(tokenizer(), "Hi.")
    });
    let body = json!({
        "messages": hi(), "reasoning_effort": "none",
        "stream_options": {"include_usage": true},
    });
    let mut sse = open(&srv, &body).await;
    assert_eq!(sse.status(), 200);
    let ct = sse.headers()["content-type"].to_str().unwrap().to_string();
    assert!(ct.starts_with("text/event-stream"), "{ct}");
    assert!(!sse.headers().contains_key("content-length"));
    assert_eq!(sse.headers()["cache-control"], "no-cache");
    let s = drain(&mut sse).await;
    assert!(s.done && s.after_done == 0);
    assert!(
        s.keepalives_before_content >= 20,
        "{} keep-alives in a 25 s prefill",
        s.keepalives_before_content
    );
    // The first chunk: the role and an empty content.
    let first = &s.chunks[0];
    assert_eq!(first["object"], "chat.completion.chunk");
    assert!(first["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(first["model"], MODEL);
    assert!(first["created"].as_u64().unwrap() > 1_700_000_000);
    assert_eq!(first["choices"][0]["index"], 0);
    assert_eq!(
        first["choices"][0]["delta"],
        json!({"role": "assistant", "content": ""})
    );
    assert!(first["choices"][0]["finish_reason"].is_null());
    // Every chunk has the first one's id, model and creation time.
    for c in &s.chunks {
        for k in ["id", "model", "created", "object"] {
            assert_eq!(c[k], first[k], "{c}");
        }
    }
    let f = s.fold();
    assert_eq!(f.content, "Hi.");
    assert_eq!(f.finish.as_deref(), Some("stop"));
    // The final chunk is an empty delta with the finish reason; the usage chunk after it has no choices.
    let n = s.chunks.len();
    let last = &s.chunks[n - 2];
    assert_eq!(last["choices"][0]["delta"], json!({}));
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert!(last.get("usage").is_none_or(Value::is_null));
    let usage = &s.chunks[n - 1];
    assert_eq!(usage["choices"], json!([]));
    let u = &usage["usage"];
    let prompt = srv.fake.requests()[0].1.prompt.len() as u64;
    let completion = tokenizer().encode("Hi.", true).len() as u64 + 1;
    assert_eq!(u["prompt_tokens"], prompt);
    assert_eq!(u["completion_tokens"], completion);
    assert_eq!(u["total_tokens"], prompt + completion);
    assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 0);
    assert_eq!(u["completion_tokens_details"]["reasoning_tokens"], 0);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o5_usage_only_when_asked_and_only_when_streaming() {
    let srv = TestServer::start(Setup::new()).await;
    say(&srv, "Hi.");
    let body = json!({"messages": hi(), "reasoning_effort": "none"});
    let s = streamed(&srv, &body).await;
    assert!(s.fold().usage.is_none(), "no usage chunk unless asked");
    assert!(s
        .chunks
        .iter()
        .all(|c| !c["choices"].as_array().unwrap().is_empty()));
    assert_eq!(s.fold().finish.as_deref(), Some("stop"));
    // `stream_options` on a request that does not stream changes nothing: one JSON body.
    say(&srv, "Hi.");
    let mut body = body;
    body["stream_options"] = json!({"include_usage": true});
    let r = post(&srv, &body).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["object"], "chat.completion");
    // The comment is a keep-alive and nothing else comes out of a stream that was never kept waiting.
    assert_eq!(s.keepalives, 0);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o5_errors_before_started_are_real_statuses() {
    let mut setup = Setup::new();
    setup.config.queue_timeout_s = 1;
    let srv = std::sync::Arc::new(TestServer::start(setup).await);
    // A validation error with `stream: true` is a 400 with a JSON body, not an event.
    let body = json!({"messages": "no", "stream": true});
    let r = post(&srv, &body).await;
    assert_eq!(r.status, 400);
    assert!(r.json()["error"]["message"].is_string());
    // A request that waits past the queue timeout is a 503 before any header.
    srv.fake.push(Script {
        prefill: Duration::from_secs(4),
        ..Script::text(tokenizer(), "slow")
    });
    let first = {
        let srv = srv.clone();
        tokio::spawn(async move { post(&srv, &json!({"messages": hi()})).await })
    };
    assert!(wait_until(WAIT, || srv.fake.requests().len() == 1).await);
    let r = post(&srv, &json!({"messages": hi(), "stream": true})).await;
    assert_eq!(r.status, 503);
    assert_eq!(r.json()["error"]["type"], "service_unavailable");
    assert_eq!(first.await.unwrap().status, 200);
    std::sync::Arc::into_inner(srv)
        .unwrap()
        .shutdown()
        .await
        .unwrap();
}

#[tokio::test]
async fn o5_reasoning_and_content_deltas() {
    let srv = TestServer::start(Setup::new()).await;
    say(&srv, "Let me think.\n</think>\n\nThe answer.");
    let s = streamed(&srv, &json!({"messages": hi()})).await;
    let f = s.fold();
    assert_eq!(f.reasoning, "Let me think.");
    assert_eq!(f.content, "The answer.");
    assert_eq!(f.finish.as_deref(), Some("stop"));
    // Reasoning deltas carry no `content` and content deltas no `reasoning_content`.
    for c in &s.chunks[1..] {
        let d = delta_of(c).unwrap();
        assert!(
            d.get("content").is_none() || d.get("reasoning_content").is_none(),
            "{d}"
        );
    }
    let first_content = s.chunks.iter().position(|c| {
        delta_of(c).is_some_and(|d| d["content"].as_str().is_some_and(|t| !t.is_empty()))
    });
    let last_reasoning = s
        .chunks
        .iter()
        .rposition(|c| delta_of(c).is_some_and(|d| d.get("reasoning_content").is_some()));
    assert!(last_reasoning.unwrap() < first_content.unwrap());
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o6_streamed_tool_calls() {
    let srv = TestServer::start(Setup::new()).await;
    let tools = json!([lookup(), weather()]);
    for calls in [
        vec![
            call_text("lookup", &[("id", "42"), ("mode", "fast"), ("tag", "7")]),
            call_text("get_weather", &[("city", "Paris \"quoted\"\nnext")]),
        ],
        vec![
            call_text("get_weather", &[("city", "Oslo")]),
            call_text("lookup", &[("id", "1"), ("tag", "x y")]),
            call_text("lookup", &[]),
        ],
    ] {
        let script = format!("Look.\n</think>\n\n{}", calls.join("\n"));
        let body = json!({"messages": hi(), "tools": tools});
        say(&srv, &script);
        let s = streamed(&srv, &body).await;
        // Each call opens with its index, id, type, name and empty arguments.
        let mut opened = 0;
        for c in &s.chunks {
            for tc in delta_of(c).unwrap()["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
            {
                if tc.get("id").is_some() {
                    assert_eq!(tc["index"], opened);
                    opened += 1;
                    assert!(tc["id"].as_str().unwrap().starts_with("call_"));
                    assert_eq!(tc["id"].as_str().unwrap().len(), "call_".len() + 24);
                    assert_eq!(tc["type"], "function");
                    assert!(tc["function"]["name"].is_string());
                    assert_eq!(tc["function"]["arguments"], "");
                    assert_eq!(tc["function"].as_object().unwrap().len(), 2);
                } else {
                    // A later delta: the index and the fragment, no more.
                    assert_eq!(tc.as_object().unwrap().len(), 2, "{tc}");
                    assert_eq!(tc["function"].as_object().unwrap().len(), 1, "{tc}");
                }
            }
        }
        assert_eq!(opened, calls.len());
        let f = s.fold();
        assert_eq!(f.finish.as_deref(), Some("tool_calls"));
        assert_eq!(f.content, "", "no text beside the calls");
        assert_eq!(f.reasoning, "Look.");
        assert_eq!(f.calls.len(), calls.len());
        // The fragments of each call join into JSON that is the non-streaming reply's.
        say(&srv, &script);
        let whole = post(&srv, &body).await.json();
        let wcalls = whole["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        for (streamed, whole) in f.calls.iter().zip(wcalls) {
            assert_eq!(streamed.name, whole["function"]["name"]);
            let a: Value = serde_json::from_str(&streamed.arguments).unwrap_or_else(|e| {
                panic!("arguments that do not parse ({e}): {}", streamed.arguments)
            });
            let b: Value =
                serde_json::from_str(whole["function"]["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(a, b);
            assert_eq!(
                streamed.arguments,
                whole["function"]["arguments"].as_str().unwrap()
            );
        }
        let ids: Vec<_> = f.calls.iter().map(|c| c.id.clone()).collect();
        let mut dedup = ids.clone();
        dedup.sort();
        dedup.dedup();
        assert_eq!(dedup.len(), ids.len(), "ids are unique");
    }
    // The values themselves, typed by the schema.
    say(
        &srv,
        &format!(
            "x\n</think>\n\n{}",
            call_text("lookup", &[("id", "42"), ("mode", "fast"), ("tag", "7")])
        ),
    );
    let f = streamed(&srv, &json!({"messages": hi(), "tools": tools}))
        .await
        .fold();
    let args: Value = serde_json::from_str(&f.calls[0].arguments).unwrap();
    assert_eq!(args, json!({"id": 42, "mode": "fast", "tag": 7}));
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o7_history_round_trip_renders_the_fixture() {
    let srv = TestServer::start(Setup::new()).await;
    let name = "t03-toolloop-reasoning-echoed";
    let bytes = std::fs::read(format!("{TESTDATA}template/requests/{name}.json")).unwrap();
    let mut body: Value = serde_json::from_slice(&bytes).unwrap();
    let want: Vec<u32> = std::fs::read(format!("{TESTDATA}template/prompts/{name}.tokens"))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    say(&srv, "It is snowing.");
    body["stream"] = json!(true);
    let mut sse = srv.stream(CHAT, &[], body.to_string().as_bytes()).await;
    assert_eq!(sse.status(), 200);
    let s = drain(&mut sse).await;
    assert!(s.done);
    assert_eq!(s.fold().finish.as_deref(), Some("stop"));
    let got = &srv.fake.requests()[0].1.prompt;
    let at = got.iter().zip(&want).position(|(a, b)| a != b);
    assert!(
        got == &want,
        "the prompt differs from the oracle's at token {at:?} ({} against {})",
        got.len(),
        want.len()
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o8_malformed_calls_are_content() {
    let srv = TestServer::start(Setup::new()).await;
    let tools = json!([
        weather(),
        json!({"type": "function", "function": {
            "name": "count", "description": "Counts.",
            "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}},
        }})
    ]);
    let run = |text: String, extra: Value| {
        let (srv, tools) = (&srv, &tools);
        async move {
            let mut body = json!({"messages": hi(), "tools": tools, "reasoning_effort": "none"});
            for (k, v) in extra.as_object().unwrap() {
                body[k] = v.clone();
            }
            say(srv, &text);
            streamed(srv, &body).await.fold()
        }
    };
    // An unknown function: all content, an ordinary stop.
    let text = "<tool_call>\n<function=nothing>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>";
    let f = run(text.to_string(), json!({})).await;
    assert_eq!(f.content, text);
    assert!(f.calls.is_empty());
    assert_eq!(f.finish.as_deref(), Some("stop"));
    // `<tool_call>` without `<function=`.
    let text = "<tool_call>\nplain words\n</tool_call>";
    let f = run(text.to_string(), json!({})).await;
    assert_eq!(f.content, text);
    assert!(f.calls.is_empty());
    assert_eq!(f.finish.as_deref(), Some("stop"));
    // A stray `<function=` in content is only text.
    let text = "Use <function=get_weather> like so.";
    let f = run(text.to_string(), json!({})).await;
    assert_eq!(f.content, text);
    assert!(f.calls.is_empty());
    // A bad integer is repaired to a string and the arguments still parse.
    let text = call_text("count", &[("n", "twelve")]);
    let f = run(text.clone(), json!({})).await;
    assert_eq!(f.finish.as_deref(), Some("tool_calls"));
    let args: Value = serde_json::from_str(&f.calls[0].arguments).unwrap();
    assert_eq!(args, json!({"n": "twelve"}));
    // A call cut off by `max_tokens`: `length`, and what was written parses.
    let text = call_text(
        "get_weather",
        &[("city", "A long name that does not end soon")],
    );
    let ids = tokenizer().encode(&text, true);
    let cut = ids
        .iter()
        .position(|&id| id == specials().end_tool_call)
        .unwrap()
        - 4;
    let f = run(text.clone(), json!({"max_tokens": cut})).await;
    assert_eq!(f.finish.as_deref(), Some("length"));
    assert_eq!(f.calls.len(), 1);
    let args: Value = serde_json::from_str(&f.calls[0].arguments).unwrap();
    assert!(args["city"].as_str().unwrap().starts_with("A long name"));
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o8_streamed_and_whole_content_agree_for_malformed_calls() {
    let srv = TestServer::start(Setup::new()).await;
    let long_header = format!(
        "<tool_call>\n<function=get_weather>{}\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>",
        " ".repeat(300)
    );
    let undeclared = "<tool_call>\n<function=teleport>\n<parameter=to>\nMars\n</parameter>\n</function>\n</tool_call>";
    let declared = call_text("get_weather", &[("city", "Oslo")]);
    let cases = [
        // No tools at all: the text is content.
        ("tools absent", declared.clone(), None),
        (
            "undeclared function",
            undeclared.to_string(),
            Some(json!([weather()])),
        ),
        (
            "header past 256 bytes",
            long_header,
            Some(json!([weather()])),
        ),
        (
            "text after a malformed call",
            format!("Before. {undeclared} After."),
            Some(json!([weather()])),
        ),
    ];
    for (name, text, tools) in cases {
        let mut body = json!({"messages": hi(), "reasoning_effort": "none"});
        if let Some(t) = tools {
            body["tools"] = t;
        }
        say(&srv, &text);
        let whole = post(&srv, &body).await.json();
        let whole_content = whole["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_else(|| panic!("{name}: no content in {whole}"))
            .to_string();
        say(&srv, &text);
        let f = streamed(&srv, &body).await.fold();
        assert_eq!(f.content, whole_content, "{name}");
        assert!(!whole_content.is_empty(), "{name}");
        assert!(f.calls.is_empty() && whole["choices"][0]["message"].get("tool_calls").is_none());
        assert_eq!(
            f.finish.as_deref(),
            whole["choices"][0]["finish_reason"].as_str(),
            "{name}"
        );
    }
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o11_stop_string_on_the_wire() {
    let srv = TestServer::start(Setup::new()).await;
    // "END" arrives in two tokens, "E" and "ND", with a long tail behind it.
    let tk = tokenizer();
    let mut tokens = tk.encode("one ", true);
    tokens.extend(tk.encode("E", true));
    tokens.extend(tk.encode("ND", true));
    let k = tokens.len();
    tokens.extend(tk.encode(
        " two three four five six seven eight nine ten eleven twelve",
        true,
    ));
    assert!(tokens.len() >= k + 8);
    srv.fake.push(Script {
        tokens,
        per_token: Duration::from_millis(100),
        ..Script::default()
    });
    let body = json!({"messages": hi(), "reasoning_effort": "none", "stop": ["END"]});
    let s = streamed(&srv, &body).await;
    let f = s.fold();
    assert_eq!(f.content, "one ");
    assert_eq!(f.finish.as_deref(), Some("stop"));
    let runs = srv.fake.runs();
    assert_eq!(runs[0].finish, Some(Finish::Cancelled));
    // The cancel flag is set when the token that completes the match is processed, a whole token interval before the
    // next send: at most that token and the one already in flight were sent.
    assert!(
        runs[0].token_times.len() <= k + 1,
        "{} tokens sent, the match completes at {k}",
        runs[0].token_times.len()
    );
    assert!(!srv.fake.cancellations().is_empty());
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn o12_logprobs_streaming() {
    let srv = TestServer::start(Setup::new()).await;
    say(&srv, "Thinking\n</think>\n\nHi there");
    let body = json!({"messages": hi(), "logprobs": true, "top_logprobs": 2});
    let s = streamed(&srv, &body).await;
    let f = s.fold();
    assert_eq!(f.content, "Hi there");
    let mut text = String::new();
    let mut entries = 0;
    for c in &s.chunks {
        let choice = &c["choices"][0];
        assert!(choice.get("logprobs").is_some(), "{c}");
        let d = &choice["delta"];
        if d.get("reasoning_content").is_some() {
            // Reasoning tokens have no entries.
            assert!(
                choice["logprobs"].is_null()
                    || choice["logprobs"]["content"].as_array().unwrap().is_empty(),
                "{c}"
            );
        }
        for e in choice["logprobs"]["content"]
            .as_array()
            .into_iter()
            .flatten()
        {
            entries += 1;
            text.push_str(e["token"].as_str().unwrap());
            assert!(e["logprob"].is_number());
            assert_eq!(e["top_logprobs"].as_array().unwrap().len(), 2);
            let bytes: Vec<u8> = e["bytes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b.as_u64().unwrap() as u8)
                .collect();
            assert_eq!(bytes, e["token"].as_str().unwrap().as_bytes());
        }
    }
    assert!(entries > 0);
    assert_eq!(text, f.content);
    // Without logprobs the field is null on every chunk.
    say(&srv, "Hi there");
    let s = streamed(&srv, &json!({"messages": hi(), "reasoning_effort": "none"})).await;
    for c in &s.chunks {
        let choice = &c["choices"][0];
        assert!(
            choice.get("logprobs").is_some() && choice["logprobs"].is_null(),
            "{c}"
        );
    }
    srv.shutdown().await.unwrap();
}

/// A script of `n` tokens, one word each, at `per_token`.
fn long_script(n: usize, per_token: Duration) -> Script {
    let text = "word ".repeat(n);
    let mut s = Script::text(tokenizer(), &format!("x\n</think>\n\n{text}"));
    s.per_token = per_token;
    s
}

#[tokio::test]
async fn o13_disconnects_cancel_within_one_token() {
    let srv = TestServer::start(Setup::new()).await;
    // Streaming: leave after the first content chunk.
    srv.fake.push(long_script(200, Duration::from_millis(150)));
    let mut sse = open(&srv, &json!({"messages": hi()})).await;
    assert_eq!(sse.status(), 200);
    loop {
        let item = sse
            .next(WAIT)
            .await
            .expect("a chunk before the stream ends");
        if let SseItem::Event { data, .. } = item {
            let v: Value = serde_json::from_str(&data).unwrap();
            if delta_of(&v).is_some_and(|d| d["content"].as_str().is_some_and(|t| !t.is_empty())) {
                break;
            }
        }
    }
    let t = Instant::now();
    sse.disconnect();
    assert!(
        wait_until(WAIT, || !srv.fake.cancellations().is_empty()).await,
        "the engine was never told to stop"
    );
    let run = srv.fake.runs()[0].clone();
    assert!(
        run.token_times.len() < 150,
        "{} tokens",
        run.token_times.len()
    );
    assert!(
        run.tokens_after(t) <= 1,
        "{} tokens after the disconnect",
        run.tokens_after(t)
    );
    assert!(
        wait_until(WAIT, || srv.fake.runs()[0].finish
            == Some(Finish::Cancelled))
        .await
    );
    // And the server serves the next request.
    say(&srv, "again");
    let r = post(&srv, &json!({"messages": hi(), "reasoning_effort": "none"})).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["choices"][0]["message"]["content"], "again");

    // Non-streaming: the client leaves 600 ms into a 200-token script.
    srv.fake.push(long_script(200, Duration::from_millis(150)));
    let body = json!({"messages": hi()}).to_string();
    let t = srv
        .send_then_drop(
            "POST",
            CHAT,
            &[],
            body.as_bytes(),
            Duration::from_millis(600),
        )
        .await;
    assert!(
        wait_until(WAIT, || srv.fake.cancellations().len() == 2).await,
        "the engine was never told to stop"
    );
    let run = srv.fake.runs()[2].clone();
    assert!(
        run.token_times.len() < 150,
        "{} tokens",
        run.token_times.len()
    );
    assert!(
        run.tokens_after(t) <= 1,
        "{} tokens after the disconnect",
        run.tokens_after(t)
    );
    say(&srv, "once more");
    let r = post(&srv, &json!({"messages": hi(), "reasoning_effort": "none"})).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["choices"][0]["message"]["content"], "once more");
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_device_loss_mid_stream_is_an_in_band_error() {
    let srv = TestServer::start(Setup::new()).await;
    let mut script = Script::text(tokenizer(), "x\n</think>\n\nOne two three four five six");
    script.fail_at = Some((6, Fail::DeviceLost));
    srv.fake.push(script);
    let mut sse = open(&srv, &json!({"messages": hi()})).await;
    // Headers had gone out: the status is 200 and the failure is an event.
    assert_eq!(sse.status(), 200);
    let s = drain(&mut sse).await;
    assert!(s.done && s.after_done == 0);
    let last = s.chunks.last().unwrap();
    assert_eq!(last["error"]["type"], "service_unavailable");
    assert!(last["error"]["message"].is_string());
    assert!(last.get("choices").is_none());
    // Text that came before the failure was sent, and no finish chunk follows.
    let f = s.fold();
    assert!(!f.content.is_empty() || !f.reasoning.is_empty());
    assert_eq!(f.finish, None);
    // The server recovers: the next request is served.
    say(&srv, "back");
    let r = post(&srv, &json!({"messages": hi()})).await;
    assert_eq!(r.status, 200);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn an_engine_failure_mid_stream_is_an_in_band_error_without_the_text() {
    let srv = TestServer::start(Setup::new()).await;
    let mut script = Script::text(
        tokenizer(),
        "x\n</think>\n\nZqxmarker two three four five six",
    );
    script.fail_at = Some((5, Fail::OutOfMemory));
    srv.fake.push(script);
    let body = json!({"messages": [{"role": "user", "content": "Wvkpromptmarker"}]});
    let s = streamed(&srv, &body).await;
    let last = s.chunks.last().unwrap();
    assert_eq!(last["error"]["type"], "server_error");
    // The error names nothing of the prompt or the output.
    let text = last.to_string();
    assert!(
        !text.contains("Zqxmarker") && !text.contains("Wvkpromptmarker"),
        "{text}"
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn an_unload_mid_stream_is_an_in_band_error() {
    let srv = TestServer::start(Setup::new()).await;
    srv.fake.push(long_script(200, Duration::from_millis(100)));
    let mut sse = open(&srv, &json!({"messages": hi()})).await;
    loop {
        let item = sse
            .next(WAIT)
            .await
            .expect("a chunk before the stream ends");
        if let SseItem::Event { data, .. } = item {
            let v: Value = serde_json::from_str(&data).unwrap();
            if delta_of(&v).is_some_and(|d| d["content"].as_str().is_some_and(|t| !t.is_empty())) {
                break;
            }
        }
    }
    drop(srv.fake.unload());
    let s = drain(&mut sse).await;
    assert!(s.done && s.after_done == 0);
    let last = s.chunks.last().unwrap();
    assert_eq!(last["error"]["type"], "service_unavailable");
    assert_eq!(last["error"]["message"], "the model was unloaded");
    assert_eq!(s.fold().finish, None);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_vanished_engine_mid_stream_is_an_in_band_error_and_an_exit() {
    let srv = TestServer::start(Setup::new()).await;
    let mut script = Script::text(tokenizer(), "x\n</think>\n\nOne two three four five six");
    script.fail_at = Some((5, Fail::Vanish));
    srv.fake.push(script);
    let s = streamed(&srv, &json!({"messages": hi()})).await;
    let last = s.chunks.last().unwrap();
    assert_eq!(last["error"]["type"], "server_error");
    assert_eq!(last["error"]["message"], "the engine stopped");
    assert_eq!(srv.app.exited(), Some(Exit::EngineGone));
    assert_eq!(srv.shutdown().await, Err(Exit::EngineGone));
}

#[tokio::test]
async fn keep_alives_go_out_between_slow_tokens() {
    let mut setup = Setup::new();
    setup.tuning.keepalive = Duration::from_millis(200);
    let srv = TestServer::start(setup).await;
    // Each token takes longer than the keep-alive interval.
    srv.fake.push(Script {
        per_token: Duration::from_millis(700),
        ..Script::text(tokenizer(), "one two three")
    });
    let body = json!({"messages": hi(), "reasoning_effort": "none"});
    let mut sse = open(&srv, &body).await;
    let s = drain(&mut sse).await;
    assert!(s.done);
    assert_eq!(s.fold().content, "one two three");
    let after_first_content = s.keepalives - s.keepalives_before_content;
    assert!(
        s.keepalives_before_content >= 2 && after_first_content >= 2,
        "{} before the first content, {after_first_content} after",
        s.keepalives_before_content
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_streamed_request_leaves_one_log_line() {
    let srv = TestServer::start(Setup::new()).await;
    say(&srv, "Hello streamed");
    let body = json!({"messages": hi(), "reasoning_effort": "none"});
    let s = streamed(&srv, &body).await;
    assert_eq!(s.fold().content, "Hello streamed");
    // The line is written when the stream ends, which may be just after the client saw `[DONE]`.
    let read = || -> Vec<Value> {
        let mut files: Vec<_> = std::fs::read_dir(&srv.log_dir)
            .unwrap()
            .map(|e| e.unwrap().path())
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
    };
    assert!(wait_until(WAIT, || read().len() == 1).await);
    let lines = read();
    assert_eq!(lines[0]["endpoint"], CHAT);
    assert_eq!(lines[0]["status"], 200);
    assert_eq!(lines[0]["finish"], "stop");
    assert!(lines[0]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(lines[0]["completion_tokens"].as_u64().unwrap() > 0);
    let text = lines[0].to_string();
    assert!(!text.contains("Hello streamed"));
    srv.shutdown().await.unwrap();
}
