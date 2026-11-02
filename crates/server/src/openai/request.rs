//! `POST /v1/chat/completions` bodies: parsed field by field, so a type error names its path and never quotes the
//! value, and converted to the internal conversation.
use crate::convo::{Content, Conversation, Message, ToolCallIn, ToolDef};
use crate::error::{take_field, ApiError};
use crate::params::{Fields, ToolChoice};
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::fmt;

/// The request, as far as the prompt goes.
#[derive(Clone, Debug)]
pub struct ChatRequest {
    pub messages: Vec<RawMessage>,
    pub tools: Option<Vec<RawTool>>,
    pub chat_template_kwargs: Option<Map<String, Value>>,
    /// Every other key, in the body's order.
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(expecting = "a message object")]
pub struct RawMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<RawContent>,
    #[serde(default)]
    pub reasoning_content: Option<Value>,
    #[serde(default)]
    pub reasoning: Option<Value>,
    #[serde(default)]
    pub tool_calls: Option<Vec<RawToolCall>>,
}

/// A message's content: a string, or an array of parts.
#[derive(Clone, Debug)]
pub enum RawContent {
    Text(String),
    Parts(Vec<RawPart>),
}

/// A content part; only its type and text matter.
#[derive(Clone, Debug, Deserialize)]
#[serde(expecting = "a content part object")]
pub struct RawPart {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(expecting = "a tool call object")]
pub struct RawToolCall {
    pub function: RawCall,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(expecting = "a tool call's function object")]
pub struct RawCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(expecting = "a tool object")]
pub struct RawTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: RawToolFunction,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(expecting = "a tool's function object")]
pub struct RawToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
}

impl<'de> Deserialize<'de> for RawContent {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<RawContent, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = RawContent;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a string or an array of content parts")
            }

            fn visit_str<E: de::Error>(self, s: &str) -> Result<RawContent, E> {
                Ok(RawContent::Text(s.to_string()))
            }

            fn visit_string<E: de::Error>(self, s: String) -> Result<RawContent, E> {
                Ok(RawContent::Text(s))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<RawContent, A::Error> {
                let mut parts = Vec::new();
                while let Some(p) = seq.next_element::<RawPart>()? {
                    parts.push(p);
                }
                Ok(RawContent::Parts(parts))
            }
        }
        d.deserialize_any(V)
    }
}

/// Parses a request body. Known fields are taken out with `take_field`, what is left is `extra`.
pub fn parse(body: &[u8]) -> Result<ChatRequest, ApiError> {
    crate::jsonlimit::check(body)?;
    let value: Value = serde_json::from_slice(body).map_err(|e| {
        // serde_json's syntax errors are a fixed phrase and a position; they never quote the input.
        ApiError::new(
            400,
            crate::error::ErrorType::InvalidRequest,
            format!("the body is not valid JSON: {e}"),
        )
    })?;
    let Value::Object(mut map) = value else {
        return Err(ApiError::new(
            400,
            crate::error::ErrorType::InvalidRequest,
            "the body must be a JSON object",
        ));
    };
    let messages = take_field::<Vec<RawMessage>>(&mut map, "messages")?
        .ok_or_else(|| ApiError::invalid("messages", "messages is required"))?;
    let tools = take_field(&mut map, "tools")?;
    let chat_template_kwargs = take_field(&mut map, "chat_template_kwargs")?;
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
    Ok(ChatRequest {
        messages,
        tools,
        chat_template_kwargs,
        extra: map,
    })
}

/// The content part types that carry an image or a video.
const MEDIA: [&str; 4] = ["image_url", "image", "input_image", "video"];

impl ChatRequest {
    /// The internal conversation: roles kept (`developer` is `system`), contents as text, history arguments as
    /// objects, tools normalized and left out when `f`'s tool choice is `none`, thinking as `f` says.
    pub fn conversation(&self, f: &Fields) -> Result<Conversation, ApiError> {
        let messages = self
            .messages
            .iter()
            .enumerate()
            .map(|(i, m)| message(i, m))
            .collect::<Result<Vec<_>, _>>()?;
        let tools = self
            .tools
            .iter()
            .flatten()
            .filter(|_| f.tool_choice != ToolChoice::None)
            .enumerate()
            .map(|(i, t)| {
                if t.kind != "function" {
                    return Err(ApiError::invalid(
                        format!("tools[{i}].type"),
                        format!("tools[{i}].type must be \"function\""),
                    ));
                }
                Ok(ToolDef {
                    name: t.function.name.clone(),
                    description: t.function.description.clone().unwrap_or_default(),
                    parameters: t
                        .function
                        .parameters
                        .clone()
                        .unwrap_or_else(|| Value::Object(Map::new())),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        // llama.cpp sets `preserve_thinking` from `preserve_reasoning` (a boolean one), and a `preserve_thinking`
        // in the same kwargs overrides that (chat.cpp, `common_chat_template_direct_apply_impl`). The template keeps
        // history reasoning only for a `preserve_thinking` that is true, so any other value means false.
        let mut preserve_thinking = None;
        if let Some(kwargs) = &self.chat_template_kwargs {
            for key in ["preserve_reasoning", "preserve_thinking"] {
                match kwargs.get(key) {
                    Some(Value::Bool(b)) => preserve_thinking = Some(*b),
                    Some(_) => {
                        tracing::info!(
                            "chat_template_kwargs.{key} is not a boolean; taken as false"
                        );
                        preserve_thinking = Some(false);
                    }
                    None => {}
                }
            }
        }
        Ok(Conversation {
            messages,
            tools,
            enable_thinking: f.thinking.on,
            preserve_thinking,
        })
    }
}

fn message(i: usize, m: &RawMessage) -> Result<Message, ApiError> {
    let role = if m.role == "developer" {
        "system".to_string()
    } else {
        m.role.clone()
    };
    let content = match &m.content {
        None => Content::Text(String::new()),
        Some(RawContent::Text(t)) => Content::Text(t.clone()),
        Some(RawContent::Parts(parts)) => {
            let mut texts = Vec::new();
            for (j, p) in parts.iter().enumerate() {
                let kind = p.kind.as_deref();
                if kind.is_some_and(|k| MEDIA.contains(&k)) {
                    return Err(ApiError::invalid(
                        format!("messages[{i}].content[{j}]"),
                        format!("messages[{i}].content[{j}]: image input is not supported"),
                    ));
                }
                if kind.is_none_or(|k| k == "text") {
                    match (&p.text, kind) {
                        (Some(t), _) => texts.push(t.clone()),
                        (None, Some(_)) => {
                            return Err(ApiError::invalid(
                                format!("messages[{i}].content[{j}].text"),
                                format!("messages[{i}].content[{j}].text is missing"),
                            ))
                        }
                        (None, None) => tracing::debug!(
                            "ignoring message {i}'s content part {j}: no type or text"
                        ),
                    }
                } else {
                    tracing::debug!(
                        "ignoring message {i}'s content part {j}: its type is not supported"
                    );
                }
            }
            Content::Parts(texts)
        }
    };
    let reasoning_content =
        [&m.reasoning_content, &m.reasoning]
            .into_iter()
            .find_map(|r| match r {
                Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
                _ => None,
            });
    let tool_calls = m
        .tool_calls
        .iter()
        .flatten()
        .enumerate()
        .map(|(j, c)| {
            Ok(ToolCallIn {
                name: c.function.name.clone(),
                arguments: arguments(i, j, c.function.arguments.as_ref())?,
            })
        })
        .collect::<Result<Vec<_>, ApiError>>()?;
    Ok(Message {
        role,
        content,
        reasoning_content,
        tool_calls,
    })
}

/// A history call's arguments as an object: a JSON string is parsed (an empty or bad one is a 400, as the oracle
/// refuses it), an object is taken as is, an absent or null one is `{}`.
fn arguments(i: usize, j: usize, a: Option<&Value>) -> Result<Value, ApiError> {
    let param = format!("messages[{i}].tool_calls[{j}].function.arguments");
    match a {
        None | Some(Value::Null) => Ok(Value::Object(Map::new())),
        Some(v @ Value::Object(_)) => Ok(v.clone()),
        Some(Value::String(s)) => serde_json::from_str(s).map_err(|_| {
            let msg = format!("{param} is not valid JSON");
            ApiError::invalid(param, msg)
        }),
        Some(_) => {
            let msg = format!("{param} has the wrong type, expected a string or an object");
            Err(ApiError::invalid(param, msg))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Defaults;
    use serde_json::json;

    fn conv(body: serde_json::Value) -> Result<Conversation, ApiError> {
        let mut r = parse(&serde_json::to_vec(&body).unwrap())?;
        let f = r.fields(&Defaults::new("artemis", 65536))?;
        r.conversation(&f)
    }

    fn ok(body: serde_json::Value) -> Conversation {
        conv(body).unwrap()
    }

    fn err(body: serde_json::Value) -> ApiError {
        conv(body).unwrap_err()
    }

    #[test]
    fn string_arguments_are_parsed_before_rendering() {
        let c = ok(json!({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"b\": 1, \"a\": [true]}"}},
                {"id": "c2", "type": "function", "function": {"name": "g", "arguments": {"x": 2}}}]}]}));
        let calls = &c.messages[1].tool_calls;
        assert_eq!(calls[0].name, "f");
        assert_eq!(
            serde_json::to_string(&calls[0].arguments).unwrap(),
            r#"{"b":1,"a":[true]}"#
        );
        assert_eq!(calls[1].arguments, json!({"x": 2}));
        assert_eq!(c.messages[1].content, Content::Text(String::new()));
    }

    #[test]
    fn empty_or_bad_arguments_are_400_naming_the_field() {
        for args in ["", "{"] {
            let e = err(json!({"messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "tool_calls": [
                    {"type": "function", "function": {"name": "f", "arguments": args}}]}]}));
            assert_eq!(e.status(), 400);
            assert_eq!(
                e.param.as_deref(),
                Some("messages[1].tool_calls[0].function.arguments")
            );
        }
    }

    #[test]
    fn an_image_part_is_400() {
        for kind in ["image_url", "image", "input_image", "video"] {
            let e = err(json!({"messages": [
                {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": kind,
                    "image_url": {"url": "x"}}]}]}));
            assert_eq!(e.status(), 400);
            assert_eq!(e.param.as_deref(), Some("messages[0].content[1]"));
            assert!(e.message.contains("image input is not supported"), "{e:?}");
        }
        // Another part type is ignored.
        let c = ok(json!({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "audio"}, {"type": "text",
                "text": "b"}]}]}));
        assert_eq!(
            c.messages[0].content,
            Content::Parts(vec!["a".into(), "b".into()])
        );
    }

    #[test]
    fn a_wrong_type_names_an_object_not_a_rust_type() {
        let e = err(json!({"messages": ["x"]}));
        assert_eq!(e.param.as_deref(), Some("messages[0]"));
        assert!(e.message.ends_with("expected a message object"), "{e:?}");
        let e = err(json!({"messages": [{"role": "user", "content": [3]}]}));
        assert!(
            e.message.ends_with("expected a content part object"),
            "{e:?}"
        );
        for body in [
            json!({"messages": [{"role": "user"}], "tools": [1]}),
            json!({"messages": [{"role": "user", "tool_calls": [1]}]}),
            json!({"messages": [{"role": "user", "tool_calls": [{"function": 1}]}]}),
            json!({"messages": [{"role": "user"}], "tools": [{"type": "function", "function": 1}]}),
        ] {
            let e = err(body);
            assert!(!e.message.contains("struct"), "{e:?}");
            assert!(e.message.contains("object"), "{e:?}");
        }
    }

    #[test]
    fn developer_is_system() {
        let c = ok(json!({"messages": [
            {"role": "developer", "content": "d"},
            {"role": "user", "content": "q"},
            {"role": "narrator", "content": "n"}]}));
        let roles: Vec<_> = c.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user", "narrator"]);
    }

    #[test]
    fn empty_reasoning_content_is_omitted() {
        let c = ok(json!({"messages": [
            {"role": "assistant", "content": "a", "reasoning_content": ""},
            {"role": "assistant", "content": "b", "reasoning": "why"},
            {"role": "assistant", "content": "c", "reasoning_content": "because", "reasoning": "no"},
            {"role": "assistant", "content": "d", "reasoning": 5}]}));
        let r: Vec<_> = c
            .messages
            .iter()
            .map(|m| m.reasoning_content.as_deref())
            .collect();
        assert_eq!(r, [None, Some("why"), Some("because"), None]);
        // An empty reasoning_content doesn't hide a non-empty reasoning.
        let both = ok(
            json!({"messages": [{"role": "assistant", "reasoning_content": "", "reasoning": "x"}]}),
        );
        assert_eq!(both.messages[0].reasoning_content.as_deref(), Some("x"));
        let ctx = c.template_context(true);
        assert!(ctx["messages"][0].get("reasoning_content").is_none());
        assert_eq!(ctx["messages"][1]["reasoning_content"], "why");
    }

    #[test]
    fn tools_are_normalized_and_strict_dropped() {
        let c = ok(
            json!({"messages": [{"role": "user", "content": "q"}], "tools": [
            {"type": "function", "function": {"strict": true, "parameters": {"type": "object"}, "name": "f",
                "description": "d"}, "extra": 1},
            {"type": "function", "function": {"name": "g"}}]}),
        );
        let ctx = c.template_context(true);
        assert_eq!(
            serde_json::to_string(&ctx["tools"]).unwrap(),
            concat!(
                r#"[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"}}},"#,
                r#"{"type":"function","function":{"name":"g","description":"","parameters":{}}}]"#
            )
        );
        let e = err(
            json!({"messages": [{"role": "user", "content": "q"}], "tools": [
            {"type": "function", "function": {"name": "f"}}, {"type": "custom", "function": {"name": "g"}}]}),
        );
        assert_eq!(e.param.as_deref(), Some("tools[1].type"));
        // An empty list renders no tools block.
        let c = ok(json!({"messages": [{"role": "user", "content": "q"}], "tools": []}));
        assert!(c.template_context(true).get("tools").is_none());
    }

    #[test]
    fn enable_thinking_must_be_boolean() {
        let base = |k: serde_json::Value| {
            json!({"messages": [{"role": "user", "content": "q"}],
            "chat_template_kwargs": k})
        };
        assert!(ok(json!({"messages": [{"role": "user", "content": "q"}]})).enable_thinking);
        assert!(!ok(base(json!({"enable_thinking": false}))).enable_thinking);
        let e = err(base(json!({"enable_thinking": "false"})));
        assert_eq!(e.status(), 400);
        assert_eq!(
            e.param.as_deref(),
            Some("chat_template_kwargs.enable_thinking")
        );
        let c = ok(base(json!({"preserve_thinking": true})));
        assert_eq!(c.preserve_thinking, Some(true));
        assert_eq!(c.template_context(true)["preserve_thinking"], true);
        // The oracle keeps the history's reasoning unless the client says otherwise, and any value that isn't `true`
        // says otherwise.
        let plain = ok(json!({"messages": [{"role": "user", "content": "q"}]}));
        assert_eq!(plain.preserve_thinking, None);
        assert_eq!(plain.template_context(true)["preserve_thinking"], true);
        let c = ok(base(json!({"preserve_thinking": false})));
        assert_eq!(c.template_context(true)["preserve_thinking"], false);
        assert_eq!(
            ok(base(json!({"preserve_thinking": "yes"}))).preserve_thinking,
            Some(false)
        );
        // `preserve_reasoning` sets it too, and an explicit `preserve_thinking` wins when both are given.
        let pt = |k: serde_json::Value| ok(base(k)).preserve_thinking;
        assert_eq!(pt(json!({"preserve_reasoning": false})), Some(false));
        assert_eq!(pt(json!({"preserve_reasoning": true})), Some(true));
        assert_eq!(pt(json!({"preserve_reasoning": "x"})), Some(false));
        assert_eq!(
            pt(json!({"preserve_reasoning": false, "preserve_thinking": true})),
            Some(true)
        );
        assert_eq!(
            pt(json!({"preserve_reasoning": true, "preserve_thinking": false})),
            Some(false)
        );
    }

    #[test]
    fn a_type_error_names_the_full_path() {
        let e = err(json!({"messages": [
            {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": 5}]}));
        assert_eq!(
            (e.status(), e.param.as_deref()),
            (400, Some("messages[2].content"))
        );
        let e = err(json!({"messages": [{"role": "user", "content": "a"}], "tools": 7}));
        assert_eq!(e.param.as_deref(), Some("tools"));
        let e = err(
            json!({"messages": [{"role": "user", "content": "a"}], "chat_template_kwargs": [1]}),
        );
        assert_eq!(e.param.as_deref(), Some("chat_template_kwargs"));
        let e = err(json!({"messages": [{"role": "user", "content": "a"}, {"content": "x"}]}));
        assert_eq!(e.param.as_deref(), Some("messages[1]"));
        assert!(e.message.contains("messages[1].role is missing"), "{e:?}");
        let e = err(
            json!({"messages": [{"role": "user", "content": "a", "tool_calls": [{"function": {"arguments": {}}}]}]}),
        );
        assert_eq!(
            e.param.as_deref(),
            Some("messages[0].tool_calls[0].function")
        );
        let e = err(json!({"model": "m"}));
        assert_eq!(e.param.as_deref(), Some("messages"));
    }

    #[test]
    fn unknown_fields_land_in_extra_in_order() {
        let r = parse(
            br#"{"zeta": 1, "messages": [{"role": "user", "content": "q"}], "model": "m", "alpha": null,
                "stream": true}"#,
        )
        .unwrap();
        let keys: Vec<_> = r.extra.keys().map(String::as_str).collect();
        assert_eq!(keys, ["zeta", "model", "alpha", "stream"]);
        assert_eq!(r.messages.len(), 1);
    }

    #[test]
    fn fields_takes_extra_without_copying() {
        let mut r = parse(
            br#"{"messages": [{"role": "user", "content": "q"}], "model": "m", "zeta": [1, 2], "stream": true}"#,
        )
        .unwrap();
        let f = r.fields(&Defaults::new("artemis", 65536)).unwrap();
        assert!(r.extra.is_empty());
        assert_eq!(f.model.as_deref(), Some("m"));
        assert!(f.stream);
        assert_eq!(f.ignored, ["zeta"]);
    }

    #[test]
    fn a_body_over_the_value_cap_is_a_400_at_parse() {
        let mut body = String::from(r#"{"messages": [], "x": ["#);
        body.push_str(&"0,".repeat(crate::jsonlimit::MAX_JSON_VALUES));
        body.push_str("0]}");
        let e = parse(body.as_bytes()).unwrap_err();
        assert_eq!(e.status, 400);
        assert!(e.message.contains("1,000,000"), "{}", e.message);
    }

    #[test]
    fn a_400_never_echoes_the_value() {
        const M: &str = "MARKER-ECHO-3c1d";
        let bodies = [
            json!({"messages": [{"role": "user", "content": {M: 1}}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "text", "text": {M: 1}}]}]}),
            json!({"messages": [{"role": M, "content": [M]}]}),
            json!({"messages": [{"role": "user", "content": "q"}], "tools": M}),
            json!({"messages": [{"role": "user", "content": "q"}], "tools": [M]}),
            json!({"messages": M}),
            json!({"messages": [{"role": "user", "content": "q"}], "chat_template_kwargs": M}),
            json!({"messages": [{"role": "user", "content": "q"}], "chat_template_kwargs": {"enable_thinking": M}}),
            json!({"messages": [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": M}}]}]}),
            json!({"messages": [{"role": "assistant", "tool_calls": [{"function": {"name": M, "arguments": 5}}]}]}),
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
        assert_eq!(e.status(), 400);
        assert!(!e.message.contains(M), "{e:?}");
        let e = parse(format!("[\"{M}\"]").as_bytes()).unwrap_err();
        assert!(!e.message.contains(M), "{e:?}");
    }
}
