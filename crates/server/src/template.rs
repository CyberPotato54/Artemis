//! The chat template renderer: minijinja with our own `tojson` and `string` filters and `raise_exception`.
use crate::convo::Conversation;
use crate::pyfmt;
use minijinja::value::Serde;
use minijinja::{AutoEscape, Environment, Error, ErrorKind, UndefinedBehavior, Value};
use serde_json::Value as Json;
use std::error::Error as _;
use std::fmt;

/// The name the template is registered under.
const NAME: &str = "chat_template";

/// The compiled chat template.
pub struct Template {
    env: Environment<'static>,
}

/// Why a render failed: the template raised (`raise_exception`), or minijinja did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderError {
    /// The error's detail: no template line text. It goes into the 400 body for a raised error only (the template's own
    /// words); a render error's message is never logged (`prompt.rs` logs `kind` and `line`), and a compile error's is
    /// returned to the caller.
    pub message: String,
    /// The template's own `raise_exception`, which the server answers as a client error.
    pub raised: bool,
    /// minijinja's name for the kind of error: fixed text, safe to log.
    pub kind: String,
    /// The template line it happened on, if minijinja knows it.
    pub line: Option<usize>,
}

/// The source of a `raise_exception` error. minijinja raises `InvalidOperation` for its own errors too, so the kind
/// can't tell the two apart.
#[derive(Debug)]
struct Raised;

impl fmt::Display for Raised {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("raised by the template")
    }
}

impl std::error::Error for Raised {}

/// The instructions one render may run (minijinja's fuel). The committed template renders a 5,000-message
/// conversation well inside it; a template that loops without end, or nests loops, runs out and the render fails
/// (a 500, `render_error`) instead of holding a preparation permit and a CPU for as long as it likes.
pub const TEMPLATE_FUEL: u64 = 50_000_000;

impl Template {
    /// Compiles `source`, the GGUF's `tokenizer.chat_template`.
    pub fn new(source: String) -> Result<Template, String> {
        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.set_fuel(Some(TEMPLATE_FUEL));
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // Python's `json.dumps` and `str`, not minijinja's: the prompt must be the oracle's bytes.
        env.add_filter("tojson", |v: Value| -> Result<Value, Error> {
            let j = serde_json::to_value(&v)
                .map_err(|e| Error::new(ErrorKind::InvalidOperation, format!("tojson: {e}")))?;
            Ok(Value::from_safe_string(pyfmt::dumps(&j)))
        });
        env.add_filter("string", |v: Value| pyfmt::py_str(&v));
        env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg).with_source(Raised))
        });
        env.add_template_owned(NAME, source)
            .map_err(|e| error(e).message)?;
        Ok(Template { env })
    }

    /// Renders `c`, with or without the generation prompt.
    pub fn render(
        &self,
        c: &Conversation,
        add_generation_prompt: bool,
    ) -> Result<String, RenderError> {
        self.render_json(&c.template_context(add_generation_prompt))
    }

    /// Renders with `ctx`, the template's context object: `messages`, `tools`, `add_generation_prompt`,
    /// `enable_thinking` and `preserve_thinking`.
    pub fn render_json(&self, ctx: &Json) -> Result<String, RenderError> {
        let template = self.env.get_template(NAME).map_err(error)?;
        template.render(Value::from(Serde(ctx))).map_err(error)
    }
}

fn error(e: Error) -> RenderError {
    let mut raised = false;
    let mut source = e.source();
    while let Some(s) = source {
        raised |= s.is::<Raised>();
        source = s.source();
    }
    let message = e
        .detail()
        .map_or_else(|| e.kind().to_string(), str::to_string);
    RenderError {
        message,
        raised,
        kind: e.kind().to_string(),
        line: e.line(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn render(src: &str, ctx: &Json) -> Result<String, RenderError> {
        Template::new(src.to_string()).unwrap().render_json(ctx)
    }

    #[test]
    fn tojson_is_the_oracles_dumps_and_unescaped() {
        let ctx = json!({"v": {"z": "<é> & 'q'", "a": [1, 1.0, null, true]}});
        // Floats are the oracle's C++ text: `1.0` is `1`.
        assert_eq!(
            render("{{ v | tojson }}", &ctx).unwrap(),
            r#"{"z": "<é> & 'q'", "a": [1, 1, null, true]}"#
        );
    }

    #[test]
    fn string_is_the_oracles_str() {
        let ctx = json!({"t": true, "n": null, "f": 1e-7, "i": 42, "s": "x<y"});
        let src =
            "{{ t | string }} {{ n | string }} {{ f | string }} {{ i | string }} {{ s | string }}";
        assert_eq!(render(src, &ctx).unwrap(), "True None 1e-07 42 x<y");
    }

    #[test]
    fn a_compile_error_is_one_line_of_detail() {
        let e = Template::new("a\n{% if %}\nb".to_string()).err().unwrap();
        assert!(!e.is_empty() && !e.contains('\n'), "{e:?}");
    }

    #[test]
    fn raise_exception_is_told_from_minijinjas_own_errors() {
        let e = render("a{{ raise_exception('nope') }}", &json!({})).unwrap_err();
        assert_eq!((e.message.as_str(), e.raised), ("nope", true));
        let e = render("{{ missing_function() }}", &json!({})).unwrap_err();
        assert!(!e.raised, "{e:?}");
        assert!(!e.message.contains("missing_function()"), "{e:?}");
    }

    #[test]
    fn a_looping_template_runs_out_of_fuel() {
        let src =
            "{% for i in range(100000) %}{% for j in range(100000) %}{% endfor %}{% endfor %}";
        let e = render(src, &json!({})).unwrap_err();
        assert!(!e.raised, "{e:?}");
        assert_eq!(e.kind, "engine ran out of fuel", "{e:?}");
    }

    #[test]
    fn the_real_template_renders_a_long_conversation_within_fuel() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/template/qwen36-chat-template.jinja"
        ))
        .unwrap();
        let words = vec!["word"; 20].join(" ");
        let messages: Vec<Json> = (0..5_000)
            .map(|i| json!({"role": if i % 2 == 0 { "user" } else { "assistant" }, "content": words}))
            .collect();
        let tool = |name: &str| {
            json!({"type": "function", "function": {"name": name, "description": "a tool",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}})
        };
        let ctx = json!({
            "messages": messages,
            "tools": [tool("search"), tool("fetch")],
            "add_generation_prompt": true,
            "enable_thinking": true,
            "preserve_thinking": false,
        });
        let out = render(&source, &ctx).unwrap();
        assert!(out.len() > 100_000, "{}", out.len());
    }
}
