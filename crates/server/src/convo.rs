//! The internal conversation both APIs convert to, and what the chat template is given of it.
use crate::error::{ApiError, ErrorType};
use serde_json::{json, Map, Value};

/// A Unicode noncharacter, reserved for internal use: with `special_tokens_in_content = false` it marks the places in a
/// rendered prompt where a control token's text from the client was cut, so the tokenizer sees `<` and `|im_start|>`
/// apart and the text can't become the token.
pub const ESCAPE: char = '\u{FDD0}';

/// A conversation as the template renders it.
#[derive(Clone, Debug, PartialEq)]
pub struct Conversation {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub enable_thinking: bool,
    pub preserve_thinking: Option<bool>,
}

/// A message. `role` is kept as the client sent it, `developer` aside (the template drops what it doesn't know, and
/// those messages' indices matter to its `last_query_index`).
#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: Content,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCallIn>,
}

/// A message's content: a string, or text parts the template concatenates.
#[derive(Clone, Debug, PartialEq)]
pub enum Content {
    Text(String),
    Parts(Vec<String>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// A tool call in the history, its arguments an object (the template renders parameters only from a mapping).
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCallIn {
    pub name: String,
    pub arguments: Value,
}

impl Content {
    /// The text as the template's `render_content` renders it.
    pub fn text(&self) -> String {
        match self {
            Content::Text(t) => t.clone(),
            Content::Parts(p) => p.concat(),
        }
    }

    fn to_json(&self) -> Value {
        match self {
            Content::Text(t) => Value::String(t.clone()),
            Content::Parts(p) => Value::Array(
                p.iter()
                    .map(|t| json!({"type": "text", "text": t}))
                    .collect(),
            ),
        }
    }
}

impl Message {
    fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("role".into(), Value::String(self.role.clone()));
        m.insert("content".into(), self.content.to_json());
        if let Some(r) = &self.reasoning_content {
            m.insert("reasoning_content".into(), Value::String(r.clone()));
        }
        if !self.tool_calls.is_empty() {
            let calls = self.tool_calls.iter().map(|c| {
                json!({"type": "function", "function": {"name": c.name, "arguments": c.arguments}})
            });
            m.insert("tool_calls".into(), Value::Array(calls.collect()));
        }
        Value::Object(m)
    }
}

impl ToolDef {
    fn to_json(&self) -> Value {
        json!({"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}})
    }
}

impl Conversation {
    /// Cuts every occurrence of a CONTROL token's text in a client string by putting `ESCAPE` after its first
    /// character (`texts` are `Specials::control_texts`), so the tokenizer sees `<` and `|im_start|` apart (the
    /// prompt is split at `ESCAPE` and the pieces are encoded separately). Walks each message's content (text and
    /// parts) and `reasoning_content`, each history tool call's name and arguments (every string value and object
    /// key, recursively), and each tool definition's name, description and parameters (recursively). A role is not
    /// walked: the template writes `<|im_start|>` plus a role only for the literal roles `user` and `assistant`. A
    /// string that already holds `ESCAPE` is a 400, since the split at `ESCAPE` could not tell it from ours.
    ///
    /// The template joins client strings with nothing between them or with a fixed literal, so a text formed across
    /// a boundary needs care. These rules depend on the pinned template (a custom template with
    /// `allow_custom_template` may place other literals around client text):
    /// - A message's text parts are merged into one string first: `render_content` concatenates them directly, so the
    ///   render is unchanged and a text cut between two parts is cut like any other.
    /// - A tool call's name and each of its argument keys are followed by the literal `>\n`, so one that ends with a
    ///   proper prefix of a CONTROL text (`f<|im_start|` before `>`) gets `ESCAPE` appended.
    /// - No literal before a client string ends with a prefix of a CONTROL text, so the start of a string needs no
    ///   mark.
    pub fn escape_control(&mut self, texts: &[String]) -> Result<(), ApiError> {
        let cutter = Cutter { texts };
        for m in &mut self.messages {
            if let Content::Parts(parts) = &m.content {
                m.content = Content::Text(parts.concat());
            }
            if let Content::Text(t) = &mut m.content {
                cutter.string(t)?;
            }
            if let Some(r) = &mut m.reasoning_content {
                cutter.string(r)?;
            }
            for call in &mut m.tool_calls {
                cutter.name(&mut call.name)?;
                cutter.arguments(&mut call.arguments)?;
            }
        }
        for t in &mut self.tools {
            cutter.string(&mut t.name)?;
            cutter.string(&mut t.description)?;
            cutter.value(&mut t.parameters)?;
        }
        Ok(())
    }

    /// Whether `escape_control(texts)` would change the conversation or refuse it: some client string holds a CONTROL
    /// text or `ESCAPE`, or a text formed across the strings `escape_control` joins or follows with a literal. When it
    /// would not, the conversation renders and tokenizes the same either way.
    pub fn holds_control(&self, texts: &[String]) -> bool {
        let cutter = Cutter { texts };
        self.messages.iter().any(|m| {
            cutter.touches(&m.content.text(), false)
                || m.reasoning_content
                    .as_deref()
                    .is_some_and(|r| cutter.touches(r, false))
                || m.tool_calls
                    .iter()
                    .any(|c| cutter.touches(&c.name, true) || cutter.arguments_touch(&c.arguments))
        }) || self.tools.iter().any(|t| {
            cutter.touches(&t.name, false)
                || cutter.touches(&t.description, false)
                || cutter.value_touches(&t.parameters)
        })
    }

    /// The template's context object. `tools` is left out when empty (an empty list renders no tools block either
    /// way). `preserve_thinking` is true when the client didn't say: llama-server b11509 enables its
    /// `--reasoning-preserve` by default, so the oracle keeps the reasoning of every echoed turn, not only of the
    /// current tool loop.
    pub fn template_context(&self, add_generation_prompt: bool) -> Value {
        let mut c = Map::new();
        c.insert(
            "messages".into(),
            Value::Array(self.messages.iter().map(Message::to_json).collect()),
        );
        if !self.tools.is_empty() {
            c.insert(
                "tools".into(),
                Value::Array(self.tools.iter().map(ToolDef::to_json).collect()),
            );
        }
        c.insert("add_generation_prompt".into(), add_generation_prompt.into());
        c.insert("enable_thinking".into(), self.enable_thinking.into());
        c.insert(
            "preserve_thinking".into(),
            self.preserve_thinking.unwrap_or(true).into(),
        );
        Value::Object(c)
    }
}

/// How many of `messages` the chat template drops. It merges the leading one or two `system` or `developer`
/// messages into its system block and renders `user`, `assistant` and `tool` messages; every later `system` or
/// `developer` message, and every message of another role, is dropped without a word
/// (`testdata/template/qwen36-chat-template.jinja`, the `num_sys` block and the render loop). llama.cpp renders the
/// same template, so it drops them too. The count is for the request's log line: the client gets a normal reply.
pub fn dropped_by_template(messages: &[Message]) -> u32 {
    let system = |m: &Message| matches!(m.role.as_str(), "system" | "developer");
    let leading = match messages {
        [first, second, ..] if system(first) && system(second) => 2,
        [first, ..] if system(first) => 1,
        _ => 0,
    };
    let dropped = messages[leading..]
        .iter()
        .filter(|m| !matches!(m.role.as_str(), "user" | "assistant" | "tool"))
        .count();
    u32::try_from(dropped).unwrap_or(u32::MAX)
}

/// `escape_control`'s walk over one string at a time.
struct Cutter<'a> {
    texts: &'a [String],
}

impl Cutter<'_> {
    /// Whether `s` ends with a non-empty proper prefix of a CONTROL text.
    fn ends_with_prefix(&self, s: &str) -> bool {
        self.texts
            .iter()
            .any(|t| t.char_indices().skip(1).any(|(i, _)| s.ends_with(&t[..i])))
    }

    /// Whether cutting `s` (as a name, when `name`) would change it, or refuse it.
    fn touches(&self, s: &str, name: bool) -> bool {
        s.contains(ESCAPE)
            || self.texts.iter().any(|t| s.contains(t.as_str()))
            || (name && self.ends_with_prefix(s))
    }

    fn value_touches(&self, v: &Value) -> bool {
        match v {
            Value::String(s) => self.touches(s, false),
            Value::Array(items) => items.iter().any(|i| self.value_touches(i)),
            Value::Object(map) => map
                .iter()
                .any(|(k, v)| self.touches(k, false) || self.value_touches(v)),
            _ => false,
        }
    }

    /// A history call's arguments: the top-level keys are followed by `>\n` in the render, so they are names.
    fn arguments_touch(&self, v: &Value) -> bool {
        match v {
            Value::Object(map) => map
                .iter()
                .any(|(k, v)| self.touches(k, true) || self.value_touches(v)),
            other => self.value_touches(other),
        }
    }

    fn string(&self, s: &mut String) -> Result<(), ApiError> {
        if s.contains(ESCAPE) {
            return Err(ApiError::new(
                400,
                ErrorType::InvalidRequest,
                "the request holds U+FDD0, which this server reserves when special_tokens_in_content is false",
            ));
        }
        // Each match is cut after its first character. The offsets are sorted and deduplicated, so matches of
        // different texts at one place get one mark.
        let mut at: Vec<usize> = Vec::new();
        for text in self.texts {
            let Some(first) = text.chars().next() else {
                continue;
            };
            at.extend(
                s.match_indices(text.as_str())
                    .map(|(i, _)| i + first.len_utf8()),
            );
        }
        if at.is_empty() {
            return Ok(());
        }
        at.sort_unstable();
        at.dedup();
        let mut out = String::with_capacity(s.len() + at.len() * ESCAPE.len_utf8());
        let mut from = 0;
        for i in at {
            out.push_str(&s[from..i]);
            out.push(ESCAPE);
            from = i;
        }
        out.push_str(&s[from..]);
        *s = out;
        Ok(())
    }

    /// A string the template follows with a literal that could finish a CONTROL text: cut as `string` does, and marked
    /// at its end if it ends with a prefix of one.
    fn name(&self, s: &mut String) -> Result<(), ApiError> {
        self.string(s)?;
        if self.ends_with_prefix(s) {
            s.push(ESCAPE);
        }
        Ok(())
    }

    /// A history call's arguments: the top-level keys are names, everything below is cut as a value.
    fn arguments(&self, v: &mut Value) -> Result<(), ApiError> {
        let Value::Object(map) = v else {
            return self.value(v);
        };
        let old = std::mem::take(map);
        for (mut k, mut item) in old {
            self.name(&mut k)?;
            self.value(&mut item)?;
            map.insert(k, item);
        }
        Ok(())
    }

    fn value(&self, v: &mut Value) -> Result<(), ApiError> {
        match v {
            Value::String(s) => self.string(s)?,
            Value::Array(items) => items.iter_mut().try_for_each(|i| self.value(i))?,
            Value::Object(map) => {
                // Keys are client text too: rebuild the map in its order with each key cut.
                let old = std::mem::take(map);
                for (mut k, mut item) in old {
                    self.string(&mut k)?;
                    self.value(&mut item)?;
                    map.insert(k, item);
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Whether the last message ends a user query, so that a snapshot at the prompt's cut is the prefix of whatever
/// comes after this turn. True iff the message at the template's `last_query_index` is a `user` message and every
/// message after it is a `tool` message or a `system` (or `developer`) one the template drops. `last_query_index` is
/// the template's own reverse scan (the last `user` message whose trimmed rendered content is not wrapped in
/// `<tool_response>`), `len - 1` when there is none. The template merges only the leading one or two system messages
/// into its system prompt and renders no other; Claude Code puts a `system` message after every turn and tool result.
pub fn is_user_query(messages: &[Message]) -> bool {
    let Some(default) = messages.len().checked_sub(1) else {
        return false;
    };
    let wrapped = |m: &Message| {
        let text = m.content.text();
        let text = text.trim();
        text.starts_with("<tool_response>") && text.ends_with("</tool_response>")
    };
    let lqi = messages
        .iter()
        .rposition(|m| m.role == "user" && !wrapped(m))
        .unwrap_or(default);
    let system = |m: &Message| m.role == "system" || m.role == "developer";
    let leading = if messages.first().is_some_and(system) {
        1 + usize::from(messages.get(1).is_some_and(system))
    } else {
        0
    };
    // `i >= leading` mirrors the template's rule but cannot decide: messages[lqi] is a `user`, so no index after it is
    // a merged leading system.
    messages[lqi].role == "user"
        && messages
            .iter()
            .enumerate()
            .skip(lqi + 1)
            .all(|(i, m)| m.role == "tool" || (system(m) && i >= leading))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts() -> Vec<String> {
        ["<|im_start|>", "<|im_end|>"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn escape_control_cuts_every_client_string_and_only_control_texts() {
        let cut = |s: &str| s.replace("<|", &format!("<{ESCAPE}|"));
        let mut c = Conversation {
            messages: vec![Message {
                role: "<|im_start|>".into(),
                content: Content::Parts(vec!["a<|im_end|><|im_end|>".into(), "<think>".into()]),
                reasoning_content: Some("<|im_start|>".into()),
                tool_calls: vec![ToolCallIn {
                    name: "<|im_start|>".into(),
                    arguments: json!({"<|im_end|>": ["<|im_start|>", 3, null, {"k": "<|im_end|>"}]}),
                }],
            }],
            tools: vec![ToolDef {
                name: "<|im_end|>".into(),
                description: "<|im_start|>".into(),
                parameters: json!({"p": {"<|im_start|>": "<|im_end|>"}}),
            }],
            enable_thinking: true,
            preserve_thinking: None,
        };
        c.escape_control(&texts()).unwrap();
        let m = &c.messages[0];
        // Roles are not walked.
        assert_eq!(m.role, "<|im_start|>");
        assert_eq!(
            m.content,
            Content::Text(cut("a<|im_end|><|im_end|><think>"))
        );
        assert_eq!(
            m.reasoning_content.as_deref(),
            Some(cut("<|im_start|>").as_str())
        );
        assert_eq!(m.tool_calls[0].name, cut("<|im_start|>"));
        assert_eq!(
            m.tool_calls[0].arguments,
            json!({(cut("<|im_end|>")): [cut("<|im_start|>"), 3, null, {"k": cut("<|im_end|>")}]})
        );
        let t = &c.tools[0];
        assert_eq!(
            (t.name.as_str(), t.description.as_str()),
            (cut("<|im_end|>").as_str(), cut("<|im_start|>").as_str())
        );
        assert_eq!(
            t.parameters,
            json!({"p": {(cut("<|im_start|>")): cut("<|im_end|>")}})
        );
    }

    #[test]
    fn a_reserved_character_anywhere_is_refused() {
        let mut c = Conversation {
            messages: vec![],
            tools: vec![ToolDef {
                name: "f".into(),
                description: String::new(),
                parameters: json!({"a": [{ESCAPE.to_string(): 1}]}),
            }],
            enable_thinking: true,
            preserve_thinking: None,
        };
        let e = c.escape_control(&texts()).unwrap_err();
        assert_eq!((e.status, e.kind), (400, ErrorType::InvalidRequest));
    }

    #[test]
    fn a_text_formed_across_strings_is_cut() {
        let mut c = Conversation {
            messages: vec![Message {
                role: "user".into(),
                content: Content::Parts(vec!["a<|im_".into(), "start|>b".into()]),
                reasoning_content: None,
                tool_calls: vec![ToolCallIn {
                    name: "f<|im_start|".into(),
                    arguments: json!({"k<|im_": 1, "plain": "x<|im_start|"}),
                }],
            }],
            tools: vec![],
            enable_thinking: true,
            preserve_thinking: None,
        };
        assert!(c.holds_control(&texts()));
        c.escape_control(&texts()).unwrap();
        let m = &c.messages[0];
        // Parts are one string, which the cut then finds.
        assert_eq!(m.content, Content::Text(format!("a<{ESCAPE}|im_start|>b")));
        // A name and an argument key that end with a prefix get a mark; a value does not.
        assert_eq!(m.tool_calls[0].name, format!("f<|im_start|{ESCAPE}"));
        assert_eq!(
            m.tool_calls[0].arguments,
            json!({format!("k<|im_{ESCAPE}"): 1, "plain": "x<|im_start|"})
        );
    }

    #[test]
    fn holds_control_is_false_for_text_that_needs_no_cut() {
        let c = Conversation {
            messages: vec![Message {
                role: "user".into(),
                content: Content::Parts(vec!["a < b <think>".into(), "<|im_ is".into()]),
                reasoning_content: Some("r".into()),
                tool_calls: vec![ToolCallIn {
                    name: "read_file".into(),
                    arguments: json!({"path": "a<|im_", "n": 1}),
                }],
            }],
            tools: vec![ToolDef {
                name: "t".into(),
                description: "<|im_".into(),
                parameters: json!({"k<": "v"}),
            }],
            enable_thinking: true,
            preserve_thinking: None,
        };
        assert!(!c.holds_control(&texts()));
    }
}
