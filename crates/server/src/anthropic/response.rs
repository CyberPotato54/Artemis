//! The Anthropic response bodies: the non-streaming `message` and the streaming events.
//!
//! The pipeline's segments become content blocks in the order they occur, streaming and not: a reasoning segment is a
//! `thinking` block, a content segment a `text` block, a call a `tool_use` block. A later segment of the same kind is
//! a block of its own whenever something else came between, and adjacent pieces of one segment are one block.
use crate::error::{ApiError, Flavour};
use crate::exchange::{Ending, Generation, Usage};
use crate::ids::new_id;
use crate::pipeline::Out;
use serde_json::{json, Value};

/// What a `thinking` block's `signature` holds. Claude Code keeps a thinking block with an empty signature and echoes
/// it back (the capture), so the oracle's empty string is enough.
pub const THINKING_SIGNATURE: &str = "";

/// `stop_reason`.
pub fn stop_reason(e: &Ending) -> &'static str {
    match e {
        Ending::Stop => "end_turn",
        Ending::StopString(_) => "stop_sequence",
        Ending::Length => "max_tokens",
        Ending::ToolCalls => "tool_use",
    }
}

/// `stop_sequence`: the string that matched, else null.
fn stop_sequence(e: &Ending) -> Value {
    match e {
        Ending::StopString(s) => Value::String(s.clone()),
        _ => Value::Null,
    }
}

/// The usage Anthropic means: `input_tokens` excludes what was read from the cache.
fn usage_json(prompt: u32, reused: u32, output: u32) -> Value {
    json!({
        "input_tokens": prompt.saturating_sub(reused),
        "cache_read_input_tokens": reused,
        "cache_creation_input_tokens": 0,
        "output_tokens": output,
    })
}

/// A block of the non-streaming reply.
enum Block {
    Thinking(String),
    Text(String),
    /// A call's name and its arguments' text.
    Tool(String, String),
}

/// The `message` body. `model` is what the request named (or `model_id`); call ids are `toolu_` and 24 base-62
/// characters. A call's `input` is its arguments parsed (`{}` for a call cut off before they were whole).
pub fn anthropic_body(g: &Generation, model: &str, id: &str) -> Value {
    let mut blocks: Vec<Block> = Vec::new();
    // The block of each call, by the call's index.
    let mut calls: Vec<usize> = Vec::new();
    for out in &g.outs {
        match out {
            Out::Reasoning(t) if !t.is_empty() => match blocks.last_mut() {
                Some(Block::Thinking(s)) => s.push_str(t),
                _ => blocks.push(Block::Thinking(t.clone())),
            },
            Out::Content(t) if !t.is_empty() => match blocks.last_mut() {
                Some(Block::Text(s)) => s.push_str(t),
                _ => blocks.push(Block::Text(t.clone())),
            },
            Out::CallStart { name, .. } => {
                calls.push(blocks.len());
                blocks.push(Block::Tool(name.clone(), String::new()));
            }
            Out::CallArgs { index, fragment } => {
                if let Some(Block::Tool(_, args)) =
                    calls.get(*index).and_then(|&b| blocks.get_mut(b))
                {
                    args.push_str(fragment);
                }
            }
            _ => {}
        }
    }
    let content: Vec<Value> = if blocks.is_empty() {
        vec![json!({"type": "text", "text": ""})]
    } else {
        blocks
            .into_iter()
            .map(|b| match b {
                Block::Thinking(thinking) => {
                    json!({"type": "thinking", "thinking": thinking, "signature": THINKING_SIGNATURE})
                }
                Block::Text(text) => json!({"type": "text", "text": text}),
                Block::Tool(name, args) => {
                    let input = serde_json::from_str::<Value>(&args)
                        .ok()
                        .filter(Value::is_object)
                        .unwrap_or_else(|| json!({}));
                    json!({"type": "tool_use", "id": new_id("toolu_"), "name": name, "input": input})
                }
            })
            .collect()
    };
    let e = g.ending();
    let u = g.usage();
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason(&e),
        "stop_sequence": stop_sequence(&e),
        "usage": usage_json(u.prompt, u.reused, u.completion),
    })
}

/// The block the stream has open.
enum Open {
    None,
    Thinking,
    Text,
    /// A call, and whether any argument text has gone out.
    Tool(bool),
}

/// One streamed event: its name, and its `data:` JSON (whose `type` is the name).
pub type Event = (String, Value);

fn event(v: Value) -> Event {
    let name = v["type"].as_str().expect("an event has a type").to_string();
    (name, v)
}

/// The streaming reply's events. Each block is closed before the next opens, and `index` counts blocks in
/// opening order.
pub struct EventWriter {
    id: String,
    model: String,
    /// The prompt's tokens and how many of them were reused, from `Started`.
    usage_in: (u32, u32),
    /// How many blocks have been opened.
    opened: usize,
    open: Open,
}

impl EventWriter {
    /// `usage_in` is `(prompt_tokens, reused_tokens)`.
    pub fn new(id: &str, model: &str, usage_in: (u32, u32)) -> EventWriter {
        EventWriter {
            id: id.to_string(),
            model: model.to_string(),
            usage_in,
            opened: 0,
            open: Open::None,
        }
    }

    /// `message_start`.
    pub fn start(&self) -> Event {
        event(json!({
            "type": "message_start",
            "message": {
                "id": self.id, "type": "message", "role": "assistant", "model": self.model,
                "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": usage_json(self.usage_in.0, self.usage_in.1, 0),
            },
        }))
    }

    /// `ping`.
    pub fn ping(&self) -> Event {
        event(json!({"type": "ping"}))
    }

    /// The index of the open block.
    fn index(&self) -> usize {
        self.opened - 1
    }

    fn delta(&self, delta: Value) -> Event {
        event(json!({"type": "content_block_delta", "index": self.index(), "delta": delta}))
    }

    /// Opens a block: closes the open one first.
    fn begin(&mut self, block: Value, open: Open, ev: &mut Vec<Event>) {
        self.close(ev);
        self.opened += 1;
        self.open = open;
        ev.push(event(json!({
            "type": "content_block_start", "index": self.index(), "content_block": block,
        })));
    }

    /// Closes the open block, if any: a thinking block with its signature first, a call that sent no arguments with
    /// `{}`.
    fn close(&mut self, ev: &mut Vec<Event>) {
        let open = std::mem::replace(&mut self.open, Open::None);
        match open {
            Open::None => return,
            Open::Thinking => ev.push(
                self.delta(json!({"type": "signature_delta", "signature": THINKING_SIGNATURE})),
            ),
            Open::Text => {}
            Open::Tool(sent) => {
                if !sent {
                    ev.push(self.delta(json!({"type": "input_json_delta", "partial_json": "{}"})));
                }
            }
        }
        ev.push(event(
            json!({"type": "content_block_stop", "index": self.index()}),
        ));
    }

    /// The events for one piece of output; none for the pieces that only mark a boundary.
    pub fn push(&mut self, out: &Out) -> Vec<Event> {
        let mut ev = Vec::new();
        match out {
            Out::Reasoning(t) if !t.is_empty() => {
                if !matches!(self.open, Open::Thinking) {
                    self.begin(
                        json!({"type": "thinking", "thinking": ""}),
                        Open::Thinking,
                        &mut ev,
                    );
                }
                ev.push(self.delta(json!({"type": "thinking_delta", "thinking": t})));
            }
            Out::Content(t) if !t.is_empty() => {
                if !matches!(self.open, Open::Text) {
                    self.begin(json!({"type": "text", "text": ""}), Open::Text, &mut ev);
                }
                ev.push(self.delta(json!({"type": "text_delta", "text": t})));
            }
            Out::CallStart { name, .. } => self.begin(
                json!({"type": "tool_use", "id": new_id("toolu_"), "name": name, "input": {}}),
                Open::Tool(false),
                &mut ev,
            ),
            Out::CallArgs { fragment, .. } if !fragment.is_empty() => {
                if let Open::Tool(sent) = &mut self.open {
                    *sent = true;
                    ev.push(
                        self.delta(json!({"type": "input_json_delta", "partial_json": fragment})),
                    );
                }
            }
            Out::CallEnd { .. } => {
                if matches!(self.open, Open::Tool(_)) {
                    self.close(&mut ev);
                }
            }
            _ => {}
        }
        ev
    }

    /// The events for a run of pieces.
    pub fn push_all(&mut self, outs: &[Out]) -> Vec<Event> {
        outs.iter().flat_map(|o| self.push(o)).collect()
    }

    /// The end: the open block closes (a reply with no block at all gets one empty text block), then `message_delta`
    /// and `message_stop`.
    pub fn finish(&mut self, e: &Ending, u: &Usage) -> Vec<Event> {
        let mut ev = Vec::new();
        self.close(&mut ev);
        if self.opened == 0 {
            self.begin(json!({"type": "text", "text": ""}), Open::Text, &mut ev);
            self.close(&mut ev);
        }
        ev.push(event(json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason(e), "stop_sequence": stop_sequence(e)},
            "usage": {"output_tokens": u.completion},
        })));
        ev.push(event(json!({"type": "message_stop"})));
        ev
    }

    /// A failure after the headers went out: the Anthropic error body, as the `error` event.
    pub fn error(&self, e: &ApiError) -> Event {
        ("error".to_string(), e.body(Flavour::Anthropic))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{CacheReport, Counts, Finish, Timings};
    use crate::error::ErrorType;
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

    fn usage() -> Usage {
        Usage {
            prompt: 10,
            completion: 5,
            reused: 4,
            reasoning: 2,
        }
    }

    #[test]
    fn a_text_reply_with_the_usage_anthropic_means() {
        let g = generation(
            vec![Out::Reasoning("why".into()), Out::Content("Hi".into())],
            Finish::Stop(1),
            0,
        );
        assert_eq!(
            anthropic_body(&g, "m", "msg_x"),
            json!({
                "id": "msg_x", "type": "message", "role": "assistant", "model": "m",
                "content": [{"type": "thinking", "thinking": "why", "signature": ""},
                    {"type": "text", "text": "Hi"}],
                "stop_reason": "end_turn", "stop_sequence": null,
                "usage": {"input_tokens": 6, "cache_read_input_tokens": 4,
                    "cache_creation_input_tokens": 0, "output_tokens": 5},
            })
        );
    }

    #[test]
    fn segments_are_blocks_in_order() {
        let mut outs = vec![
            Out::Reasoning("a".into()),
            Out::Reasoning("a2".into()),
            Out::Content("Hi ".into()),
            Out::Content("there".into()),
        ];
        outs.extend(call(0, "f", &["{\"x\":", "1}"]));
        outs.push(Out::Content("then".into()));
        outs.push(Out::Reasoning("b".into()));
        outs.push(Out::Content("bye".into()));
        outs.extend(call(1, "g", &[]));
        let b = anthropic_body(&generation(outs, Finish::Stop(1), 2), "m", "i");
        let c = b["content"].as_array().unwrap();
        let kinds: Vec<&str> = c.iter().map(|x| x["type"].as_str().unwrap()).collect();
        assert_eq!(
            kinds,
            ["thinking", "text", "tool_use", "text", "thinking", "text", "tool_use"]
        );
        assert_eq!(c[0]["thinking"], "aa2");
        assert_eq!(c[1]["text"], "Hi there");
        assert_eq!(c[2]["input"], json!({"x": 1}));
        assert_eq!(c[6]["input"], json!({}));
        assert!(c[2]["id"].as_str().unwrap().starts_with("toolu_"));
        assert_ne!(c[2]["id"], c[6]["id"]);
        assert_eq!(b["stop_reason"], "tool_use");
    }

    #[test]
    fn an_empty_reply_is_one_empty_text_block_and_a_cut_off_call_is_max_tokens() {
        let b = anthropic_body(&generation(vec![], Finish::Stop(1), 0), "m", "i");
        assert_eq!(b["content"], json!([{"type": "text", "text": ""}]));
        let g = generation(call(0, "a", &["{\"x"]), Finish::Length, 1);
        let b = anthropic_body(&g, "m", "i");
        assert_eq!(b["stop_reason"], "max_tokens");
        assert_eq!(b["content"][0]["input"], json!({}));
    }

    fn names(ev: &[Event]) -> Vec<&str> {
        ev.iter().map(|e| e.0.as_str()).collect()
    }

    #[test]
    fn blocks_close_before_the_next_opens_and_indexes_count_up() {
        let mut w = EventWriter::new("msg_x", "m", (10, 4));
        let (n, start) = w.start();
        assert_eq!(n, "message_start");
        assert_eq!(start["message"]["usage"]["input_tokens"], 6);
        assert_eq!(start["message"]["usage"]["cache_read_input_tokens"], 4);
        let mut ev = Vec::new();
        ev.extend(w.push(&Out::Reasoning("a".into())));
        ev.extend(w.push(&Out::Content("b".into())));
        ev.extend(w.push_all(&call(0, "f", &["{}"])));
        ev.extend(w.push(&Out::Content("c".into())));
        ev.extend(w.finish(&Ending::Stop, &usage()));
        assert_eq!(
            names(&ev),
            [
                "content_block_start",
                "content_block_delta", // thinking 0
                "content_block_delta",
                "content_block_stop", // its signature, stop
                "content_block_start",
                "content_block_delta",
                "content_block_stop", // text 1
                "content_block_start",
                "content_block_delta",
                "content_block_stop", // call 2
                "content_block_start",
                "content_block_delta",
                "content_block_stop", // text 3
                "message_delta",
                "message_stop",
            ]
        );
        let idx: Vec<u64> = ev.iter().filter_map(|e| e.1["index"].as_u64()).collect();
        assert_eq!(idx, [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
        assert_eq!(
            ev[2].1["delta"],
            json!({"type": "signature_delta", "signature": ""})
        );
        let last = &ev[ev.len() - 2].1;
        assert_eq!(last["usage"], json!({"output_tokens": 5}));
        for e in &ev {
            assert_eq!(e.1["type"], e.0.as_str());
        }
    }

    #[test]
    fn a_call_starts_with_an_empty_input_and_sends_braces_when_it_has_no_arguments() {
        let mut w = EventWriter::new("i", "m", (1, 0));
        let ev = w.push_all(&call(0, "now", &[]));
        assert_eq!(
            ev[0].1["content_block"],
            json!({"type": "tool_use", "id": ev[0].1["content_block"]["id"], "name": "now", "input": {}})
        );
        assert_eq!(
            ev[1].1["delta"],
            json!({"type": "input_json_delta", "partial_json": "{}"})
        );
        assert_eq!(
            names(&ev),
            [
                "content_block_start",
                "content_block_delta",
                "content_block_stop"
            ]
        );
        // A call that did send arguments does not get another.
        let ev = w.push_all(&call(1, "f", &["{\"a\":", "1}"]));
        assert_eq!(names(&ev).len(), 4);
        // One cut off before its end is closed by the finish.
        let mut ev = w.push(&Out::CallStart {
            index: 2,
            name: "g".into(),
        });
        ev.extend(w.finish(&Ending::Length, &usage()));
        assert_eq!(ev[1].1["delta"]["partial_json"], "{}");
        assert_eq!(ev[ev.len() - 2].1["delta"]["stop_reason"], "max_tokens");
    }

    #[test]
    fn an_empty_reply_streams_one_empty_text_block() {
        let mut w = EventWriter::new("i", "m", (1, 0));
        let ev = w.finish(&Ending::StopString("END".into()), &usage());
        assert_eq!(
            names(&ev),
            [
                "content_block_start",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        assert_eq!(ev[0].1["index"], 0);
        assert_eq!(
            ev[0].1["content_block"],
            json!({"type": "text", "text": ""})
        );
        assert_eq!(
            ev[2].1["delta"],
            json!({"stop_reason": "stop_sequence", "stop_sequence": "END"})
        );
    }

    #[test]
    fn markers_and_empty_pieces_make_no_events() {
        let mut w = EventWriter::new("i", "m", (1, 0));
        for out in [
            Out::Content(String::new()),
            Out::Reasoning(String::new()),
            Out::CallEnd { index: 0 },
            Out::Stopped {
                matched: "x".into(),
            },
        ] {
            assert!(w.push(&out).is_empty());
        }
    }

    #[test]
    fn an_error_event_is_the_anthropic_error_body() {
        let w = EventWriter::new("i", "m", (1, 0));
        let (name, v) = w.error(&ApiError::new(
            503,
            ErrorType::Unavailable,
            "the model was unloaded",
        ));
        assert_eq!(name, "error");
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "overloaded_error");
        assert_eq!(v["error"]["message"], "the model was unloaded");
    }
}
