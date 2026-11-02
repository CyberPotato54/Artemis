//! `POST /v1/messages` bodies: parsed field by field, so a type error names its path and never quotes the value,
//! and converted to the internal conversation as llama.cpp b11509 converts them.
use crate::convo::{Content, Conversation, Message, ToolCallIn, ToolDef};
use crate::error::{take_field, ApiError, ErrorType};
use crate::params::{Fields, ToolChoice};
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::fmt;

/// The request, as far as the prompt goes.
#[derive(Clone, Debug)]
pub struct MessagesRequest {
    /// `None` when absent; an empty string or an array without text blocks is still a (empty) system message.
    pub system: Option<RawSystem>,
    pub messages: Vec<RawMessage>,
    pub tools: Option<Vec<RawTool>>,
    /// Every other key, in the body's order.
    pub extra: Map<String, Value>,
}

/// `system`: a string, or an array of blocks of which the text ones count.
#[derive(Clone, Debug)]
pub enum RawSystem {
    Text(String),
    Blocks(Vec<RawBlock>),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(expecting = "a message object")]
pub struct RawMessage {
    pub role: String,
    /// `None` when the key is absent, `Some(None)` when it is null: llama.cpp skips an `assistant` message without
    /// the key, rejects another role's, and takes a null as empty text.
    #[serde(default, deserialize_with = "present")]
    pub content: Option<Option<RawContent>>,
}

/// A message's content: a string, or an array of blocks.
#[derive(Clone, Debug)]
pub enum RawContent {
    Text(String),
    Blocks(Vec<RawBlock>),
}

/// A content block of any type. Each type interprets (and type-checks) the fields it uses and the rest are ignored
/// (`signature`, `cache_control`, `is_error`, `tool_use_id`, a `web_search_tool_result`'s object `content`, ...), so a
/// field is held as it came.
#[derive(Clone, Debug, Deserialize)]
#[serde(expecting = "a content block object")]
pub struct RawBlock {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub text: Option<Value>,
    #[serde(default)]
    pub thinking: Option<Value>,
    #[serde(default)]
    pub name: Option<Value>,
    #[serde(default)]
    pub input: Option<Value>,
    /// A `tool_result`'s content: a string, or an array of blocks.
    #[serde(default)]
    pub content: Option<Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(expecting = "a tool object")]
pub struct RawTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// `None` when absent (`{}`), `Some(None)` when null (llama.cpp passes the null on).
    #[serde(default, deserialize_with = "present")]
    pub input_schema: Option<Option<Value>>,
}

/// For a field whose absence differs from its null: `Some(None)` is null, `Some(Some(v))` a value.
fn present<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d).map(Some)
}

/// A string, or an array of `T`; any other JSON type is a type error naming `what`.
fn text_or_array<'de, D, T>(d: D, what: &'static str) -> Result<Result<String, Vec<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct V<T>(&'static str, std::marker::PhantomData<T>);
    impl<'de, T: Deserialize<'de>> Visitor<'de> for V<T> {
        type Value = Result<String, Vec<T>>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }

        fn visit_str<E: de::Error>(self, s: &str) -> Result<Self::Value, E> {
            Ok(Ok(s.to_string()))
        }

        fn visit_string<E: de::Error>(self, s: String) -> Result<Self::Value, E> {
            Ok(Ok(s))
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut items = Vec::new();
            while let Some(b) = seq.next_element::<T>()? {
                items.push(b);
            }
            Ok(Err(items))
        }
    }
    d.deserialize_any(V(what, std::marker::PhantomData))
}

impl<'de> Deserialize<'de> for RawContent {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<RawContent, D::Error> {
        Ok(
            match text_or_array(d, "a string or an array of content blocks")? {
                Ok(t) => RawContent::Text(t),
                Err(b) => RawContent::Blocks(b),
            },
        )
    }
}

impl<'de> Deserialize<'de> for RawSystem {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<RawSystem, D::Error> {
        Ok(
            match text_or_array(d, "a string or an array of text blocks")? {
                Ok(t) => RawSystem::Text(t),
                Err(b) => RawSystem::Blocks(b),
            },
        )
    }
}

/// Parses a request body. Known fields are taken out with `take_field`, what is left is `extra`.
pub fn parse(body: &[u8]) -> Result<MessagesRequest, ApiError> {
    crate::jsonlimit::check(body)?;
    let value: Value = serde_json::from_slice(body).map_err(|e| {
        // serde_json's syntax errors are a fixed phrase and a position; they never quote the input.
        ApiError::new(
            400,
            ErrorType::InvalidRequest,
            format!("the body is not valid JSON: {e}"),
        )
    })?;
    let Value::Object(mut map) = value else {
        return Err(ApiError::new(
            400,
            ErrorType::InvalidRequest,
            "the body must be a JSON object",
        ));
    };
    let messages = take_field::<Vec<RawMessage>>(&mut map, "messages")?
        .ok_or_else(|| ApiError::invalid("messages", "messages is required"))?;
    let system = take_field(&mut map, "system")?;
    let tools = take_field(&mut map, "tools")?;
    if !map.is_empty() {
        // Bounded like the request line's list: a body of a million unknown keys is one short line.
        let names: Vec<String> = map
            .keys()
            .take(crate::params::IGNORED_SHOWN)
            .map(|k| crate::params::cut_name(k))
            .collect();
        let more = map.len() - names.len();
        tracing::debug!(?names, more, "request fields the prompt doesn't use");
    }
    Ok(MessagesRequest {
        system,
        messages,
        tools,
        extra: map,
    })
}

const BILLING_HEADER: &str = "x-anthropic-billing-header:";

/// Claude Code's billing header (a system text starting `x-anthropic-billing-header:`) carries a `cch=` value that
/// changes on every request and would defeat prefix reuse; its 5 bytes become `fffff` when a `;` follows them. Any
/// other text is left alone. llama.cpp's rule, to the byte: the first `cch=` after the prefix, counted in bytes.
pub fn normalize_billing_header(s: &mut String) {
    if !s.starts_with(BILLING_HEADER) {
        return;
    }
    let Some(at) = s[BILLING_HEADER.len()..].find("cch=") else {
        return;
    };
    let value = BILLING_HEADER.len() + at + "cch=".len();
    // The `;` is ASCII and the value starts after `=`, so both ends are character boundaries.
    if s.as_bytes().get(value + 5) == Some(&b';') {
        s.replace_range(value..value + 5, "fffff");
    } else {
        // `debug`: a request can hold a block for every few bytes of its body, so this is no line for the default
        // level.
        tracing::debug!(
            "the billing header's cch value is not 5 bytes before a `;`; left as it is"
        );
    }
}

const NO_IMAGES: &str = "image input is not supported";

/// The string a block field holds: `None` when absent or null; a 400 naming the field when it is another type.
fn string_field<'a>(
    at: &str,
    field: &str,
    v: &'a Option<Value>,
) -> Result<Option<&'a str>, ApiError> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => {
            let param = format!("{at}.{field}");
            let msg = format!("{param} has the wrong type, expected a string");
            Err(ApiError::invalid(param, msg))
        }
    }
}

impl MessagesRequest {
    /// The internal conversation, thinking as `f` says and without tools when `f`'s tool choice is `none`. Mirrors
    /// llama.cpp's `server_chat_convert_anthropic_to_oai`: one leading `system` message when `system` is given; a
    /// message's text blocks one message of its role (parts), its thinking blocks that message's `reasoning_content`,
    /// its `tool_use` blocks its calls; its `tool_result` blocks `tool` messages after it.
    pub fn conversation(&self, f: &Fields) -> Result<Conversation, ApiError> {
        let mut messages = Vec::new();
        if let Some(system) = &self.system {
            let mut text = String::new();
            match system {
                RawSystem::Text(t) => {
                    text = t.clone();
                    normalize_billing_header(&mut text);
                }
                RawSystem::Blocks(blocks) => {
                    for b in blocks.iter().filter(|b| b.kind.as_deref() == Some("text")) {
                        // As llama.cpp's `json_value`: a missing or non-string text is empty.
                        let mut t = b
                            .text
                            .as_ref()
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        normalize_billing_header(&mut t);
                        text.push_str(&t);
                    }
                }
            }
            messages.push(plain("system", Content::Text(text)));
        }
        for (i, m) in self.messages.iter().enumerate() {
            convert(i, m, &mut messages)?;
        }
        let tools = self
            .tools
            .iter()
            .flatten()
            .filter(|_| f.tool_choice != ToolChoice::None)
            .map(|t| ToolDef {
                name: t.name.clone(),
                description: t.description.clone().unwrap_or_default(),
                parameters: match &t.input_schema {
                    None => Value::Object(Map::new()),
                    Some(v) => v.clone().unwrap_or(Value::Null),
                },
            })
            .collect();
        Ok(Conversation {
            messages,
            tools,
            enable_thinking: f.thinking.on,
            preserve_thinking: None,
        })
    }
}

fn plain(role: &str, content: Content) -> Message {
    Message {
        role: role.to_string(),
        content,
        reasoning_content: None,
        tool_calls: Vec::new(),
    }
}

/// Appends message `i` as the messages it becomes.
fn convert(i: usize, m: &RawMessage, out: &mut Vec<Message>) -> Result<(), ApiError> {
    let blocks = match &m.content {
        // As llama.cpp: an assistant message without the key is skipped, another role's is an error, and a null
        // is empty text.
        None if m.role == "assistant" => return Ok(()),
        None => {
            let at = format!("messages[{i}]");
            return Err(ApiError::invalid(&at, format!("{at}.content is missing")));
        }
        Some(None) => {
            out.push(plain(&m.role, Content::Text(String::new())));
            return Ok(());
        }
        Some(Some(RawContent::Text(t))) => {
            out.push(plain(&m.role, Content::Text(t.clone())));
            return Ok(());
        }
        Some(Some(RawContent::Blocks(b))) => b,
    };
    let mut parts = Vec::new();
    let mut reasoning = String::new();
    let mut calls = Vec::new();
    let mut results = Vec::new();
    for (j, b) in blocks.iter().enumerate() {
        let at = format!("messages[{i}].content[{j}]");
        match b.kind.as_deref() {
            Some("text") => match string_field(&at, "text", &b.text)? {
                Some(t) => parts.push(t.to_string()),
                None => {
                    return Err(ApiError::invalid(
                        format!("{at}.text"),
                        format!("{at}.text is missing"),
                    ))
                }
            },
            Some("thinking") => {
                reasoning.push_str(string_field(&at, "thinking", &b.thinking)?.unwrap_or_default())
            }
            Some("image") => return Err(ApiError::invalid(&at, format!("{at}: {NO_IMAGES}"))),
            Some("tool_use") => {
                let Some(name) = string_field(&at, "name", &b.name)? else {
                    return Err(ApiError::invalid(
                        format!("{at}.name"),
                        format!("{at}.name is missing"),
                    ));
                };
                let arguments = match &b.input {
                    None | Some(Value::Null) => Value::Object(Map::new()),
                    Some(v @ Value::Object(_)) => v.clone(),
                    Some(_) => {
                        return Err(ApiError::invalid(
                            format!("{at}.input"),
                            format!("{at}.input must be an object"),
                        ))
                    }
                };
                calls.push(ToolCallIn {
                    name: name.to_string(),
                    arguments,
                });
            }
            Some("tool_result") => {
                results.push(plain("tool", Content::Text(result_text(&at, &b.content)?)));
            }
            _ => tracing::debug!(
                "ignoring message {i}'s content block {j}: its type is not supported"
            ),
        }
    }
    if !parts.is_empty() || !calls.is_empty() || !reasoning.is_empty() {
        out.push(Message {
            role: m.role.clone(),
            content: if parts.is_empty() {
                Content::Text(String::new())
            } else {
                Content::Parts(parts)
            },
            reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
            tool_calls: calls,
        });
    }
    out.append(&mut results);
    Ok(())
}

/// A `tool_result`'s content as its `tool` message's string: the string, or the text blocks concatenated with no
/// separator (other block types are ignored, an image is a 400); anything else is `""`, as llama.cpp has it.
fn result_text(at: &str, content: &Option<Value>) -> Result<String, ApiError> {
    match content {
        Some(Value::String(t)) => Ok(t.clone()),
        Some(Value::Array(items)) => {
            let mut text = String::new();
            for (k, item) in items.iter().enumerate() {
                match item.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        text.push_str(item.get("text").and_then(Value::as_str).unwrap_or_default())
                    }
                    Some("image") => {
                        let at = format!("{at}.content[{k}]");
                        return Err(ApiError::invalid(&at, format!("{at}: {NO_IMAGES}")));
                    }
                    _ => {}
                }
            }
            Ok(text)
        }
        _ => Ok(String::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Defaults;
    use serde_json::json;

    fn conv(body: serde_json::Value) -> Result<Conversation, ApiError> {
        let mut r = parse(&serde_json::to_vec(&body).unwrap())?;
        let f = r.fields(&Defaults::new("artemis", 65536), true)?;
        r.conversation(&f)
    }

    fn ok(body: serde_json::Value) -> Conversation {
        conv(body).unwrap()
    }

    fn err(body: serde_json::Value) -> ApiError {
        conv(body).unwrap_err()
    }

    fn msg(role: &str, content: Content) -> Message {
        Message {
            role: role.into(),
            content,
            reasoning_content: None,
            tool_calls: vec![],
        }
    }

    fn text(t: &str) -> Content {
        Content::Text(t.into())
    }

    fn parts(p: &[&str]) -> Content {
        Content::Parts(p.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn many_odd_billing_headers_leave_at_most_one_stderr_line() {
        use std::io::Write;
        use std::sync::{Arc, Mutex};
        struct Capture(Arc<Mutex<Vec<u8>>>);
        impl Write for Capture {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let buf = Arc::new(Mutex::new(Vec::new()));
        let writer = buf.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || Capture(writer.clone()))
            .with_max_level(tracing::Level::INFO)
            .with_ansi(false)
            .finish();
        let blocks: Vec<_> = (0..1_000)
            .map(|_| json!({"type": "text", "text": "x-anthropic-billing-header: cch=abc;"}))
            .collect();
        tracing::subscriber::with_default(subscriber, || {
            ok(json!({"model": "m", "max_tokens": 1, "system": blocks,
                "messages": [{"role": "user", "content": "hi"}]}));
        });
        let text = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert!(
            text.lines().count() <= 1,
            "{} lines: {text}",
            text.lines().count()
        );
    }

    #[test]
    fn m2_string_and_block_content() {
        let c = ok(json!({"messages": [
            {"role": "user", "content": "plain"},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "assistant"},
            {"role": "user", "content": null},
            {"role": "assistant", "content": null},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "f", "input": {"x": [1, {"y": null}]}},
                {"type": "tool_use", "id": "t2", "name": "g"}]},
            {"role": "user", "content": [{"type": "text", "text": "q", "cache_control": {"type": "ephemeral"}}]}]}));
        assert!(c.enable_thinking);
        assert_eq!(c.preserve_thinking, None);
        assert!(c.tools.is_empty());
        let call = |name: &str, arguments| ToolCallIn {
            name: name.into(),
            arguments,
        };
        assert_eq!(
            c.messages,
            vec![
                msg("user", text("plain")),
                msg("assistant", parts(&["a", "b"])),
                // An assistant message without the key is skipped; a null content is empty text for any role.
                msg("user", text("")),
                msg("assistant", text("")),
                Message {
                    tool_calls: vec![
                        call("f", json!({"x": [1, {"y": null}]})),
                        call("g", json!({}))
                    ],
                    ..msg("assistant", text(""))
                },
                msg("user", parts(&["q"])),
            ]
        );
    }

    #[test]
    fn m2_a_message_other_than_assistant_needs_the_content_key() {
        for role in ["user", "tool", "system"] {
            let e = err(json!({"messages": [{"role": "user", "content": "q"}, {"role": role}]}));
            assert_eq!(e.status(), 400);
            assert_eq!(e.param.as_deref(), Some("messages[1]"));
            assert!(
                e.message.contains("messages[1].content is missing"),
                "{e:?}"
            );
        }
    }

    #[test]
    fn m2_system_blocks_concatenate_with_cache_control() {
        let tool = json!({"name": "f", "description": "d", "input_schema": {"type": "object"},
            "cache_control": {"type": "ephemeral"}});
        let user = json!({"role": "user", "content": "q"});
        let c = ok(json!({
            "system": [
                {"type": "text", "text": "one.", "cache_control": {"type": "ephemeral"}},
                {"type": "image", "source": {"type": "url", "url": "x"}},
                {"type": "text", "text": "two."},
                {"type": "text", "text": " three."}],
            "tools": [tool],
            "messages": [user]}));
        assert_eq!(
            c.messages,
            vec![
                msg("system", text("one.two. three.")),
                msg("user", text("q"))
            ]
        );
        let c = ok(json!({"system": "plain", "messages": [user]}));
        assert_eq!(c.messages[0], msg("system", text("plain")));
        assert_eq!(c.messages.len(), 2);
        // Absent or null: no system message. Present but empty: an empty one, as llama.cpp has it.
        assert_eq!(ok(json!({"messages": [user]})).messages.len(), 1);
        assert_eq!(
            ok(json!({"system": null, "messages": [user]}))
                .messages
                .len(),
            1
        );
        assert_eq!(
            ok(json!({"system": [], "messages": [user]})).messages[0],
            msg("system", text(""))
        );
        assert_eq!(
            ok(json!({"system": "", "messages": [user]})).messages[0],
            msg("system", text(""))
        );
        // The billing header is normalized in a block and in a string.
        for system in [
            json!([{"type": "text", "text": "x-anthropic-billing-header: cch=12345;"}, {"type": "text",
                "text": "cch=12345;"}]),
            json!("x-anthropic-billing-header: cch=12345;cch=12345;"),
        ] {
            let c = ok(json!({"system": system, "messages": [user]}));
            assert_eq!(
                c.messages[0],
                msg(
                    "system",
                    text("x-anthropic-billing-header: cch=fffff;cch=12345;")
                )
            );
        }
    }

    #[test]
    fn m2_tools_map_to_definitions() {
        let c = ok(
            json!({"messages": [{"role": "user", "content": "q"}], "tools": [
            {"name": "f", "description": "d", "input_schema": {"type": "object", "properties": {}}, "extra": 1},
            {"name": "g"},
            {"name": "h", "input_schema": null}]}),
        );
        let def = |name: &str, description: &str, parameters| ToolDef {
            name: name.into(),
            description: description.into(),
            parameters,
        };
        assert_eq!(
            c.tools,
            vec![
                def("f", "d", json!({"type": "object", "properties": {}})),
                def("g", "", json!({})),
                // Null stays null, as llama.cpp passes it; absent is `{}`.
                def("h", "", Value::Null),
            ]
        );
        let ctx = c.template_context(true);
        assert_eq!(
            serde_json::to_string(&ctx["tools"][1]).unwrap(),
            r#"{"type":"function","function":{"name":"g","description":"","parameters":{}}}"#
        );
        // An empty list renders no tools block.
        let c = ok(json!({"messages": [{"role": "user", "content": "q"}], "tools": []}));
        assert!(c.template_context(true).get("tools").is_none());
        let e = err(
            json!({"messages": [{"role": "user", "content": "q"}], "tools": [{"name": "f"}, {"description": "d"}]}),
        );
        assert_eq!(e.param.as_deref(), Some("tools[1]"));
        assert!(e.message.contains("tools[1].name is missing"), "{e:?}");
    }

    #[test]
    fn m2_tool_result_string_and_blocks_and_is_error() {
        let c = ok(json!({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "a", "name": "f", "input": {}},
                {"type": "tool_use", "id": "b", "name": "f", "input": {}},
                {"type": "tool_use", "id": "c", "name": "f", "input": {}},
                {"type": "tool_use", "id": "d", "name": "f", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "string result"},
                {"type": "tool_result", "tool_use_id": "b", "is_error": true, "content": [
                    {"type": "text", "text": "Light snow."}, {"type": "text", "text": " -2 C.",
                        "cache_control": {"type": "ephemeral"}},
                    {"type": "document"}]},
                {"type": "tool_result", "tool_use_id": "c"},
                {"type": "tool_result", "tool_use_id": "d", "content": []}]}]}));
        let tools: Vec<_> = c.messages[2..]
            .iter()
            .map(|m| (&m.role[..], m.content.clone()))
            .collect();
        assert_eq!(
            tools,
            vec![
                ("tool", text("string result")),
                ("tool", text("Light snow. -2 C.")),
                ("tool", text("")),
                ("tool", text("")),
            ]
        );
        // The user message held only results: it isn't emitted.
        assert_eq!(c.messages.len(), 6);
    }

    #[test]
    fn m2_thinking_blocks_with_signature_become_reasoning() {
        let c = ok(json!({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "first ", "signature": "c2ln"},
                {"type": "text", "text": "answer"},
                {"type": "thinking", "thinking": "second", "signature": "c2ln"}]},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "only thoughts"}]},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": ""}, {"type": "redacted_thinking",
                "data": "x"}]},
            {"role": "user", "content": "q4"}]}));
        assert_eq!(c.messages.len(), 6);
        assert_eq!(
            c.messages[1].reasoning_content.as_deref(),
            Some("first second")
        );
        assert_eq!(c.messages[1].content, parts(&["answer"]));
        // Thoughts alone keep the message, its content empty; empty thoughts don't.
        assert_eq!(
            c.messages[3].reasoning_content.as_deref(),
            Some("only thoughts")
        );
        assert_eq!(c.messages[3].content, text(""));
        assert_eq!(c.messages[4], msg("user", text("q3")));
        let ctx = c.template_context(true);
        assert_eq!(ctx["messages"][1]["reasoning_content"], "first second");
        assert!(ctx["messages"][0].get("reasoning_content").is_none());
    }

    #[test]
    fn m2_an_image_is_400() {
        let e = err(json!({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "a"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]}));
        assert_eq!(e.status(), 400);
        assert_eq!(e.param.as_deref(), Some("messages[0].content[1]"));
        assert!(e.message.contains("image input is not supported"), "{e:?}");
        let e = err(json!({"messages": [
            {"role": "user", "content": "q"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": [{"type": "text", "text": "t"}, {"type": "image",
                    "source": {"type": "url", "url": "u"}}]}]}]}));
        assert_eq!(e.status(), 400);
        assert_eq!(
            e.param.as_deref(),
            Some("messages[1].content[0].content[1]")
        );
        assert!(e.message.contains("image input is not supported"), "{e:?}");
    }

    #[test]
    fn m2_text_beside_tool_results_is_a_user_message_before_the_tool_messages() {
        let c = ok(json!({"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "A", "name": "f", "input": {"p": "a"}},
                {"type": "tool_use", "id": "B", "name": "f", "input": {"p": "b"}}]},
            {"role": "user", "content": [
                {"type": "text", "text": "Also check b."},
                {"type": "tool_result", "tool_use_id": "A", "content": "ra"},
                {"type": "tool_result", "tool_use_id": "B", "content": "rb"}]}]}));
        let rest: Vec<_> = c.messages[2..].to_vec();
        assert_eq!(
            rest,
            vec![
                msg("user", parts(&["Also check b."])),
                msg("tool", text("ra")),
                msg("tool", text("rb")),
            ]
        );
        // Results first, then the text: the tool messages still follow the user message.
        let c = ok(json!({"messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "A", "content": "ra"},
                {"type": "text", "text": "after"}]}]}));
        assert_eq!(
            c.messages,
            vec![msg("user", parts(&["after"])), msg("tool", text("ra"))]
        );
    }

    #[test]
    fn m2_a_block_is_read_only_by_its_own_type() {
        // An unknown type with object content (web_search_tool_result), name, text or input of any type is ignored.
        let c = ok(json!({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [
                {"type": "web_search_tool_result", "tool_use_id": "w", "content": {"type": "web_search_result_location",
                    "url": "u"}},
                {"type": "server_tool_use", "id": "s", "name": 5, "input": "x"},
                {"type": "document", "text": 3, "thinking": [], "content": 7},
                {"type": "text", "text": "kept", "content": {"a": 1}, "name": 4, "input": 9}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": 7},
                {"type": "tool_result", "tool_use_id": "b", "content": {"type": "text", "text": "x"}},
                {"type": "tool_result", "tool_use_id": "c", "content": [3, {"type": "text", "text": "t"},
                    {"type": "text", "text": 5}, {"type": "x", "text": "no"}]}]}]}));
        assert_eq!(c.messages[1], msg("assistant", parts(&["kept"])));
        let results: Vec<_> = c.messages[2..].iter().map(|m| m.content.clone()).collect();
        assert_eq!(results, vec![text(""), text(""), text("t")]);
        // The same wrong types where the block's own type reads them are 400s.
        for (block, param) in [
            (
                json!({"type": "text", "text": 3}),
                "messages[1].content[0].text",
            ),
            (
                json!({"type": "thinking", "thinking": 3}),
                "messages[1].content[0].thinking",
            ),
            (
                json!({"type": "tool_use", "name": 5, "input": {}}),
                "messages[1].content[0].name",
            ),
            (
                json!({"type": "tool_use", "name": "f", "input": 9}),
                "messages[1].content[0].input",
            ),
        ] {
            let e = err(
                json!({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": [block]}]}),
            );
            assert_eq!(e.param.as_deref(), Some(param), "{e:?}");
        }
    }

    #[test]
    fn m6_cch_is_normalized() {
        let sys = |s: &str| {
            let c = ok(json!({"system": s, "messages": [{"role": "user", "content": "q"}]}));
            c.messages[0].content.text()
        };
        assert_eq!(
            sys("x-anthropic-billing-header: cc_version=1.2.3; cch=ab12c; rest"),
            "x-anthropic-billing-header: cc_version=1.2.3; cch=fffff; rest"
        );
        assert_eq!(
            sys("x-anthropic-billing-header: cc_version=1.2.3; cch=ab12c; rest"),
            sys("x-anthropic-billing-header: cc_version=1.2.3; cch=zz99y; rest")
        );
        // Not a 5-character value followed by `;`, no header prefix, no `cch=`, or a `cch=` in the prefix's place.
        for same in [
            "x-anthropic-billing-header: cc_version=1.2.3; cch=abc;",
            "x-anthropic-billing-header: cch=abcdef;",
            "x-anthropic-billing-header: cch=abcde",
            "x-anthropic-billing-header: cch=",
            "x-anthropic-billing-header: no value here;",
            "X-anthropic-billing-header: cch=ab12c;",
            " x-anthropic-billing-header: cch=ab12c;",
            "cch=ab12c;",
            "You are x-anthropic-billing-header: cch=ab12c;",
        ] {
            assert_eq!(sys(same), same);
            let mut s = same.to_string();
            normalize_billing_header(&mut s);
            assert_eq!(s, same);
        }
        // The first `cch=` after the prefix is the one; a multi-byte value is 5 bytes, as llama.cpp counts.
        let mut s = "x-anthropic-billing-header:cch=\u{e9}\u{e9}a;cch=12345;".to_string();
        normalize_billing_header(&mut s);
        assert_eq!(s, "x-anthropic-billing-header:cch=fffff;cch=12345;");
        // Two requests that differ only in `cch` give equal conversations.
        let body = |cch: &str| {
            json!({"system": [{"type": "text",
                "text": format!("x-anthropic-billing-header: cc_version=2.1.1; cch={cch};")},
                {"type": "text", "text": "You are an assistant."}],
                "messages": [{"role": "user", "content": "q"}]})
        };
        assert_eq!(ok(body("ab12c")), ok(body("0ff1e")));
    }

    #[test]
    fn unknown_fields_land_in_extra_in_order() {
        let r = parse(
            br#"{"zeta": 1, "messages": [{"role": "user", "content": "q"}], "model": "m", "alpha": null,
                "max_tokens": 5}"#,
        )
        .unwrap();
        let keys: Vec<_> = r.extra.keys().map(String::as_str).collect();
        assert_eq!(keys, ["zeta", "model", "alpha", "max_tokens"]);
        assert_eq!(r.messages.len(), 1);
    }

    #[test]
    fn fields_takes_extra_without_copying() {
        let mut r = parse(
            br#"{"messages": [{"role": "user", "content": "q"}], "model": "m", "zeta": [1, 2], "max_tokens": 5}"#,
        )
        .unwrap();
        let f = r.fields(&Defaults::new("artemis", 65536), false).unwrap();
        assert!(r.extra.is_empty());
        assert_eq!(f.model.as_deref(), Some("m"));
        assert_eq!(f.max_tokens, Some(5));
        assert_eq!(f.ignored, ["zeta"]);
    }

    #[test]
    fn a_body_over_the_value_cap_is_a_400_at_parse() {
        let mut body = String::from(r#"{"messages": [], "max_tokens": 5, "x": ["#);
        body.push_str(&"0,".repeat(crate::jsonlimit::MAX_JSON_VALUES));
        body.push_str("0]}");
        let e = parse(body.as_bytes()).unwrap_err();
        assert_eq!(e.status, 400);
        assert!(e.message.contains("1,000,000"), "{}", e.message);
    }

    #[test]
    fn a_type_error_names_the_full_path() {
        let user = json!({"role": "user", "content": "q"});
        let e = err(json!({"model": "m"}));
        assert_eq!((e.status(), e.param.as_deref()), (400, Some("messages")));
        let e = err(json!({"messages": [user, {"role": "user", "content": 5}]}));
        assert_eq!(e.param.as_deref(), Some("messages[1].content"));
        let e = err(
            json!({"messages": [user, {"role": "user", "content": [{"type": "text", "text": 5}]}]}),
        );
        assert_eq!(e.param.as_deref(), Some("messages[1].content[0].text"));
        let e = err(json!({"messages": [user, {"role": "user", "content": [3]}]}));
        assert!(
            e.message.ends_with("expected a content block object"),
            "{e:?}"
        );
        let e = err(json!({"messages": [user, {"content": "x"}]}));
        assert!(e.message.contains("messages[1].role is missing"), "{e:?}");
        let e = err(json!({"messages": ["x"]}));
        assert!(e.message.ends_with("expected a message object"), "{e:?}");
        let e = err(json!({"system": 5, "messages": [user]}));
        assert_eq!(e.param.as_deref(), Some("system"));
        let e = err(json!({"messages": [user], "tools": 7}));
        assert_eq!(e.param.as_deref(), Some("tools"));
        let e =
            err(json!({"messages": [user, {"role": "assistant", "content": [{"type": "text"}]}]}));
        assert_eq!(e.param.as_deref(), Some("messages[1].content[0].text"));
        assert!(e.message.contains("is missing"), "{e:?}");
        let e = err(
            json!({"messages": [user, {"role": "assistant", "content": [{"type": "tool_use", "name": "f",
                "input": "s"}]}]}),
        );
        assert_eq!(e.param.as_deref(), Some("messages[1].content[0].input"));
        let e = err(
            json!({"messages": [user, {"role": "assistant", "content": [{"type": "tool_use", "input": {}}]}]}),
        );
        assert_eq!(e.param.as_deref(), Some("messages[1].content[0].name"));
        let e = parse(b"[1]").unwrap_err();
        assert_eq!(e.status(), 400);
    }

    #[test]
    fn a_400_never_echoes_the_value() {
        const M: &str = "MARKER-ECHO-3c1d";
        let user = json!({"role": "user", "content": "q"});
        let bodies = [
            json!({"messages": [{"role": "user", "content": {M: 1}}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "text", "text": {M: 1}}]}]}),
            json!({"messages": [{"role": M, "content": [M]}]}),
            json!({"messages": [user], "system": {M: 1}}),
            json!({"messages": [user], "system": [M]}),
            json!({"messages": [user], "tools": M}),
            json!({"messages": [user], "tools": [M]}),
            json!({"messages": M}),
            json!({"messages": [{"role": "assistant", "content": [{"type": "tool_use", "name": M, "input": M}]}]}),
            json!({"messages": [{"role": "assistant", "content": [{"type": "tool_use", "name": {M: 1},
                "input": {}}]}]}),
            json!({"messages": [{"role": "user", "content": [{"type": M, "source": {"type": M}}, {"type": "image",
                "source": {"url": M}}]}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "tool_result", "content": [{"type": "image",
                "source": M}]}]}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "thinking", "thinking": {M: 1}}]}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "tool_result", "content": M}, {"type": "text",
                "text": [M]}]}]}),
        ];
        for b in bodies {
            let e = err(b.clone());
            assert_eq!(e.status(), 400, "{b}");
            assert!(!e.message.contains(M), "{b}: {e:?}");
            assert!(
                !e.param.clone().unwrap_or_default().contains(M),
                "{b}: {e:?}"
            );
        }
        let e = parse(format!("{{\"messages\": {M}").as_bytes()).unwrap_err();
        assert!(!e.message.contains(M), "{e:?}");
    }
}
