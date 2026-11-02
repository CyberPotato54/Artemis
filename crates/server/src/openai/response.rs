//! The OpenAI response bodies: the non-streaming `chat.completion` and the streaming chunks.
use crate::error::{ApiError, Flavour};
use crate::exchange::{Ending, Generation, Usage};
use crate::ids::new_id;
use crate::pipeline::{LogprobEntry, Out};
use serde_json::{json, Map, Number, Value};

/// `finish_reason`.
pub fn finish_reason(e: &Ending) -> &'static str {
    match e {
        Ending::Stop | Ending::StopString(_) => "stop",
        Ending::Length => "length",
        Ending::ToolCalls => "tool_calls",
    }
}

/// A log-probability as JSON: the shortest decimal of the `f32` (not its `f64` widening), and OpenAI's -9999 for a
/// value JSON cannot hold.
fn number(x: f32) -> Value {
    let n = if x.is_finite() {
        x.to_string().parse::<f64>().ok().and_then(Number::from_f64)
    } else {
        None
    };
    Value::Number(n.unwrap_or_else(|| Number::from_f64(-9999.0).expect("finite")))
}

fn logprob_entry(e: &LogprobEntry) -> Value {
    let top: Vec<Value> = e
        .top
        .iter()
        .map(|(token, logprob, bytes)| {
            json!({"token": token, "logprob": number(*logprob), "bytes": bytes})
        })
        .collect();
    json!({
        "token": e.token, "logprob": number(e.logprob), "bytes": e.bytes, "top_logprobs": top,
    })
}

/// The `chat.completion` body: one choice, `Content-Length`-framed by the caller. `model` is what the request named
/// (or `model_id`); call ids are `call_` and 24 base-62 characters.
pub fn openai_body(g: &Generation, model: &str, id: &str, created: u64) -> Value {
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut calls: Vec<(String, String)> = Vec::new();
    let mut entries = Vec::new();
    for out in &g.outs {
        match out {
            Out::Content(t) => content.push_str(t),
            Out::Reasoning(t) => reasoning.push_str(t),
            Out::CallStart { name, .. } => calls.push((name.clone(), String::new())),
            Out::CallArgs { index, fragment } => {
                if let Some(call) = calls.get_mut(*index) {
                    call.1.push_str(fragment);
                }
            }
            Out::CallEnd { .. } | Out::Stopped { .. } => {}
            Out::Logprob(e) => entries.push(logprob_entry(e)),
        }
    }
    let mut message = Map::new();
    message.insert("role".into(), "assistant".into());
    // `content` is null only with calls and no text.
    message.insert(
        "content".into(),
        if content.is_empty() && !calls.is_empty() {
            Value::Null
        } else {
            Value::String(content)
        },
    );
    if !reasoning.is_empty() {
        message.insert("reasoning_content".into(), Value::String(reasoning));
    }
    if !calls.is_empty() {
        let calls = calls.into_iter().map(|(name, arguments)| {
            json!({"id": new_id("call_"), "type": "function",
                "function": {"name": name, "arguments": arguments}})
        });
        message.insert("tool_calls".into(), Value::Array(calls.collect()));
    }
    let logprobs = if g.logprobs || !entries.is_empty() {
        json!({"content": entries})
    } else {
        Value::Null
    };
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": logprobs,
            "finish_reason": finish_reason(&g.ending()),
        }],
        "usage": usage_json(&g.usage()),
    })
}

/// The `usage` object, in a body and in the stream's last chunk.
fn usage_json(u: &Usage) -> Value {
    json!({
        "prompt_tokens": u.prompt,
        "completion_tokens": u.completion,
        "total_tokens": u.prompt + u.completion,
        "prompt_tokens_details": {"cached_tokens": u.reused},
        "completion_tokens_details": {"reasoning_tokens": u.reasoning},
    })
}

/// The streaming reply's `chat.completion.chunk`s, one value per `data:` line. Every chunk carries the
/// reply's id, model and creation time. Call ids are made here, when a call's header goes out.
pub struct ChunkWriter {
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
    /// Whether the request asked for logprobs: a content chunk then carries a `logprobs.content` list, which may be
    /// empty.
    logprobs: bool,
}

impl ChunkWriter {
    pub fn new(
        id: &str,
        model: &str,
        created: u64,
        include_usage: bool,
        logprobs: bool,
    ) -> ChunkWriter {
        ChunkWriter {
            id: id.to_string(),
            model: model.to_string(),
            created,
            include_usage,
            logprobs,
        }
    }

    fn chunk(&self, choices: Value) -> Value {
        json!({
            "id": self.id, "object": "chat.completion.chunk", "created": self.created,
            "model": self.model, "choices": choices,
        })
    }

    fn choice(&self, delta: Value, logprobs: Value, finish: Value) -> Value {
        self.chunk(json!([{
            "index": 0, "delta": delta, "logprobs": logprobs, "finish_reason": finish,
        }]))
    }

    /// The first chunk: the role and an empty `content`.
    pub fn first(&self) -> Value {
        self.choice(
            json!({"role": "assistant", "content": ""}),
            Value::Null,
            Value::Null,
        )
    }

    /// The chunk for one piece of output; none for the pieces that only mark a boundary. A log-probability entry
    /// follows the text it belongs to (`Pipeline`), and is a chunk of its own here: `push_all` joins it to its text.
    pub fn push(&mut self, out: &Out) -> Vec<Value> {
        self.push_all(std::slice::from_ref(out))
    }

    /// The chunks for a run of pieces. An entry goes into the content chunk before it when there is one in the run,
    /// so a client that reads a chunk's `logprobs` beside its `content` finds them together.
    pub fn push_all(&mut self, outs: &[Out]) -> Vec<Value> {
        let mut chunks: Vec<Value> = Vec::new();
        // Whether the last chunk is a content one an entry may join.
        let mut joinable = false;
        for out in outs {
            match out {
                Out::Content(t) => {
                    let logprobs = if self.logprobs {
                        json!({"content": []})
                    } else {
                        Value::Null
                    };
                    chunks.push(self.choice(json!({"content": t}), logprobs, Value::Null));
                    joinable = true;
                    continue;
                }
                Out::Reasoning(t) => chunks.push(self.choice(
                    json!({"reasoning_content": t}),
                    Value::Null,
                    Value::Null,
                )),
                Out::CallStart { index, name } => chunks.push(self.choice(
                    json!({"tool_calls": [{
                        "index": index, "id": new_id("call_"), "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }]}),
                    Value::Null,
                    Value::Null,
                )),
                Out::CallArgs { index, fragment } => chunks.push(self.choice(
                    json!({"tool_calls": [{"index": index, "function": {"arguments": fragment}}]}),
                    Value::Null,
                    Value::Null,
                )),
                Out::Logprob(e) => {
                    let entry = logprob_entry(e);
                    match chunks.last_mut().filter(|_| joinable) {
                        Some(c) => {
                            let lp = &mut c["choices"][0]["logprobs"];
                            if lp.is_null() {
                                *lp = json!({"content": []});
                            }
                            if let Some(list) = lp["content"].as_array_mut() {
                                list.push(entry);
                            }
                        }
                        None => chunks.push(self.choice(
                            json!({}),
                            json!({"content": [entry]}),
                            Value::Null,
                        )),
                    }
                    continue;
                }
                Out::CallEnd { .. } | Out::Stopped { .. } => continue,
            }
            joinable = false;
        }
        chunks
    }

    /// The end: the chunk with an empty delta and the finish reason, then the usage chunk (`choices: []`) when asked.
    pub fn finish(&mut self, e: &Ending, u: &Usage) -> Vec<Value> {
        let mut chunks = vec![self.choice(
            json!({}),
            Value::Null,
            Value::String(finish_reason(e).to_string()),
        )];
        if self.include_usage {
            let mut usage = self.chunk(json!([]));
            usage["usage"] = usage_json(u);
            chunks.push(usage);
        }
        chunks
    }

    /// A failure after the headers went out: the OpenAI error body, as an event.
    pub fn error(&self, e: &ApiError) -> Value {
        e.body(Flavour::OpenAi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{CacheReport, Counts, Finish, Timings};
    use crate::pipeline::Summary;

    fn generation(outs: Vec<Out>, finish: Finish, calls: usize) -> Generation {
        Generation {
            outs,
            finish,
            counts: Counts {
                prompt: 10,
                reused: 4,
                prefilled: 6,
                completion: 5,
                reasoning: 99,
            },
            timings: Timings {
                queue_ms: 0.0,
                load_ms: 0.0,
                restore_ms: 0.0,
                prefill_ms: 0.0,
                decode_ms: 0.0,
                ttft_ms: 0.0,
            },
            cache: CacheReport {
                hit_rate: None,
                miss_mib_per_token: None,
                restore_bytes: 0,
                pool_bytes_used: 0,
                decode_misses: None,
            },
            spec: None,
            summary: Summary {
                calls,
                malformed_calls: 0,
                budget_closed: false,
                reasoning_tokens: 2,
                repairs: vec![],
                stopped: None,
            },
            logprobs: false,
        }
    }

    fn call(index: usize, name: &str, args: &[&str]) -> Vec<Out> {
        let mut v = vec![Out::CallStart {
            index,
            name: name.into(),
        }];
        v.extend(args.iter().map(|a| Out::CallArgs {
            index,
            fragment: (*a).into(),
        }));
        v.push(Out::CallEnd { index });
        v
    }

    #[test]
    fn a_text_reply() {
        let g = generation(
            vec![Out::Reasoning("why".into()), Out::Content("Hi".into())],
            Finish::Stop(1),
            0,
        );
        let b = openai_body(&g, "m", "chatcmpl-x", 7);
        assert_eq!(
            b,
            json!({
                "id": "chatcmpl-x", "object": "chat.completion", "created": 7, "model": "m",
                "choices": [{"index": 0, "logprobs": null, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hi", "reasoning_content": "why"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                    "prompt_tokens_details": {"cached_tokens": 4},
                    "completion_tokens_details": {"reasoning_tokens": 2}},
            })
        );
    }

    #[test]
    fn calls_with_and_without_text_and_without_reasoning() {
        let mut outs = call(0, "a", &["{\"x\":", "1}"]);
        outs.extend(call(1, "b", &["{}"]));
        let b = openai_body(&generation(outs.clone(), Finish::Stop(1), 2), "m", "i", 0);
        let m = &b["choices"][0]["message"];
        assert!(m["content"].is_null());
        assert!(m.get("reasoning_content").is_none());
        let calls = m["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "a");
        assert_eq!(calls[0]["function"]["arguments"], "{\"x\":1}");
        assert_eq!(calls[1]["function"]["arguments"], "{}");
        assert_ne!(calls[0]["id"], calls[1]["id"]);
        assert_eq!(b["choices"][0]["finish_reason"], "tool_calls");

        outs.insert(0, Out::Content("Checking".into()));
        let b = openai_body(&generation(outs, Finish::Stop(1), 2), "m", "i", 0);
        assert_eq!(b["choices"][0]["message"]["content"], "Checking");

        // No calls and no text: an empty string, not null.
        let b = openai_body(&generation(vec![], Finish::Stop(1), 0), "m", "i", 0);
        assert_eq!(b["choices"][0]["message"]["content"], "");
    }

    #[test]
    fn a_cut_off_call_is_length() {
        let g = generation(call(0, "a", &["{"]), Finish::Length, 1);
        let b = openai_body(&g, "m", "i", 0);
        assert_eq!(b["choices"][0]["finish_reason"], "length");
        assert_eq!(
            b["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{"
        );
    }

    #[test]
    fn logprobs_print_as_f32_and_survive_an_infinity() {
        assert_eq!(number(-0.1).to_string(), "-0.1");
        assert_eq!(number(f32::NEG_INFINITY).to_string(), "-9999.0");
        let entry = LogprobEntry {
            token: "a".into(),
            logprob: -0.25,
            bytes: vec![97],
            top: vec![("b".into(), -1.5, vec![98])],
        };
        let mut g = generation(vec![Out::Logprob(entry)], Finish::Stop(1), 0);
        let b = openai_body(&g, "m", "i", 0);
        assert_eq!(
            b["choices"][0]["logprobs"],
            json!({"content": [{"token": "a", "logprob": -0.25, "bytes": [97],
                "top_logprobs": [{"token": "b", "logprob": -1.5, "bytes": [98]}]}]})
        );
        // Asked for, but no content token: the field is there, empty.
        g.outs.clear();
        g.logprobs = true;
        let b = openai_body(&g, "m", "i", 0);
        assert_eq!(b["choices"][0]["logprobs"], json!({"content": []}));
    }

    fn writer(usage: bool, logprobs: bool) -> ChunkWriter {
        ChunkWriter::new("chatcmpl-x", "m", 7, usage, logprobs)
    }

    fn delta(c: &Value) -> &Value {
        &c["choices"][0]["delta"]
    }

    #[test]
    fn the_first_chunk_and_the_shape_every_chunk_shares() {
        let mut w = writer(false, false);
        let first = w.first();
        assert_eq!(
            first,
            json!({"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 7, "model": "m",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                    "logprobs": null, "finish_reason": null}]})
        );
        let c = w.push(&Out::Content("Hi".into()));
        assert_eq!(c.len(), 1);
        assert_eq!(delta(&c[0]), &json!({"content": "Hi"}));
        assert!(c[0]["choices"][0]["logprobs"].is_null());
        let r = w.push(&Out::Reasoning("why".into()));
        assert_eq!(delta(&r[0]), &json!({"reasoning_content": "why"}));
        for out in [
            Out::CallEnd { index: 0 },
            Out::Stopped {
                matched: "x".into(),
            },
        ] {
            assert!(w.push(&out).is_empty());
        }
    }

    #[test]
    fn a_calls_first_delta_names_it_and_the_rest_carry_fragments() {
        let mut w = writer(false, false);
        let start = w.push(&Out::CallStart {
            index: 1,
            name: "f".into(),
        });
        let tc = &delta(&start[0])["tool_calls"][0];
        assert_eq!(tc["index"], 1);
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"], json!({"name": "f", "arguments": ""}));
        assert!(tc["id"].as_str().unwrap().starts_with("call_"));
        let args = w.push(&Out::CallArgs {
            index: 1,
            fragment: "{}".into(),
        });
        assert_eq!(
            delta(&args[0]),
            &json!({"tool_calls": [{"index": 1, "function": {"arguments": "{}"}}]})
        );
    }

    #[test]
    fn the_end_is_an_empty_delta_then_usage_when_asked() {
        let u = Usage {
            prompt: 10,
            completion: 5,
            reused: 4,
            reasoning: 2,
        };
        let end = writer(false, false).finish(&Ending::ToolCalls, &u);
        assert_eq!(end.len(), 1);
        assert_eq!(delta(&end[0]), &json!({}));
        assert_eq!(end[0]["choices"][0]["finish_reason"], "tool_calls");
        let end = writer(true, false).finish(&Ending::Length, &u);
        assert_eq!(end.len(), 2);
        assert_eq!(end[0]["choices"][0]["finish_reason"], "length");
        assert_eq!(end[1]["choices"], json!([]));
        assert_eq!(end[1]["usage"]["total_tokens"], 15);
        assert_eq!(end[1]["usage"]["prompt_tokens_details"]["cached_tokens"], 4);
        assert_eq!(
            end[1]["usage"]["completion_tokens_details"]["reasoning_tokens"],
            2
        );
        assert!(end[0].get("usage").is_none());
    }

    #[test]
    fn an_entry_joins_the_content_before_it_or_stands_alone() {
        let entry = |t: &str| {
            Out::Logprob(LogprobEntry {
                token: t.into(),
                logprob: -0.5,
                bytes: t.bytes().collect(),
                top: vec![],
            })
        };
        let mut w = writer(false, true);
        let c = w.push_all(&[Out::Content("ab".into()), entry("a"), entry("b")]);
        assert_eq!(c.len(), 1);
        assert_eq!(
            c[0]["choices"][0]["logprobs"]["content"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        // Content with no entry yet still has the list.
        let c = w.push_all(&[Out::Content("c".into())]);
        assert_eq!(c[0]["choices"][0]["logprobs"], json!({"content": []}));
        // An entry with no content before it in the run is a chunk of its own.
        let c = w.push_all(&[entry("c")]);
        assert_eq!(c.len(), 1);
        assert_eq!(delta(&c[0]), &json!({}));
        assert_eq!(c[0]["choices"][0]["logprobs"]["content"][0]["token"], "c");
        // Reasoning between breaks the join.
        let c = w.push_all(&[
            Out::Content("d".into()),
            Out::Reasoning("r".into()),
            entry("d"),
        ]);
        assert_eq!(c.len(), 3);
        assert_eq!(c[0]["choices"][0]["logprobs"], json!({"content": []}));
    }

    #[test]
    fn an_error_event_is_the_openai_error_body() {
        let e = ApiError::new(
            503,
            crate::error::ErrorType::Unavailable,
            "the model was unloaded",
        );
        let v = writer(false, false).error(&e);
        assert_eq!(v["error"]["type"], "service_unavailable");
        assert_eq!(v["error"]["message"], "the model was unloaded");
        assert!(v.get("choices").is_none());
    }
}
