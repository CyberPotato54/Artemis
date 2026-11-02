//! The tool-call parser: the text between `<tool_call>` and `</tool_call>` becomes a call's name and its arguments as
//! compact JSON, incrementally.
//!
//! The text arrives in pieces cut anywhere, so every decision waits until the text decides it: the pieces' boundaries
//! never change the output or the repairs.
use super::Out;
use crate::schema::{convert, Kind, ParamTypes, ToolSet};
use serde_json::Value;
use std::sync::Arc;

/// The header (`<function=NAME>\n`) must end within this many bytes of the `<tool_call>` token.
const HEADER_LIMIT: usize = 256;
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<parameter=";
/// A parameter's key (up to its `>`) is at most this long, else the tag is dropped text.
const KEY_LIMIT: usize = 256;
/// A tool's name is cut to this many characters in a repair line.
const NAME_SHOWN: usize = 64;
/// What ends a value, as llama.cpp's `until("\n</parameter>\n")` has it: a newline, the parameter's closer and a
/// newline (so `\n</parameter>X` stays in the value), or `\n</function>` (a repair that ends the body too).
const VALUE_END: &str = "\n</parameter>\n";
const VALUE_END_FUNCTION: &str = "\n</function>";
/// A value the call ends in (`</tool_call>`, or the length limit) ends here without its last newline.
const VALUE_CLOSE: &str = "\n</parameter>";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// Between calls.
    Idle,
    /// Before a whole `<function=NAME>\n`.
    Header,
    /// The header failed: the rest of the call is content.
    Malformed,
    /// In the body, between parameters.
    Between,
    /// Inside a parameter's value.
    Value,
    /// After `</function>`, before `</tool_call>`.
    Done,
}

/// How the current value goes out.
#[derive(Debug)]
enum Mode {
    /// Exactly `string`: escaped as it arrives.
    Stream,
    /// Any other type: held until the value ends, then converted.
    Buffer(ParamTypes),
    /// A repeated key: dropped.
    Drop,
}

/// The parser of one `<tool_call>` at a time. `open` starts a call; `text` feeds it; `close` or `cut_off` ends it.
pub struct ToolCalls {
    tools: Arc<ToolSet>,
    state: State,
    /// Text received and not yet consumed (in a value: the held-back suffix and what can't be released yet).
    buf: String,
    /// The current call's name and index.
    name: String,
    index: usize,
    /// Valid calls started so far.
    calls: usize,
    /// Calls that were not valid and went out as content.
    malformed: usize,
    /// Parameters written in the current call, and every key seen (a repeat is dropped).
    emitted: usize,
    keys: Vec<String>,
    /// The current value's key, mode and (buffered) text.
    key: String,
    mode: Mode,
    raw: String,
    /// A repair was written for the run of dropped text now under way.
    dropping: bool,
    repairs: Vec<String>,
}

impl ToolCalls {
    pub fn new(tools: Arc<ToolSet>) -> ToolCalls {
        ToolCalls {
            tools,
            state: State::Idle,
            buf: String::new(),
            name: String::new(),
            index: 0,
            calls: 0,
            malformed: 0,
            emitted: 0,
            keys: Vec::new(),
            key: String::new(),
            mode: Mode::Drop,
            raw: String::new(),
            dropping: false,
            repairs: Vec::new(),
        }
    }

    /// The `<tool_call>` token.
    pub fn open(&mut self) {
        self.state = State::Header;
        self.buf.clear();
    }

    /// Text inside the call.
    pub fn text(&mut self, s: &str) -> Vec<Out> {
        let mut out = Vec::new();
        match self.state {
            State::Idle => {}
            State::Malformed => {
                if !s.is_empty() {
                    out.push(Out::Content(s.to_string()));
                }
            }
            _ => {
                self.buf.push_str(s);
                loop {
                    let more = match self.state {
                        State::Header => self.header(&mut out),
                        State::Between => self.between(&mut out),
                        State::Value => self.value(&mut out),
                        State::Done => self.done(),
                        State::Idle | State::Malformed => false,
                    };
                    if !more {
                        break;
                    }
                }
            }
        }
        out
    }

    /// `</tool_call>`, or the end after `Stop(248059)`.
    pub fn close(&mut self) -> Vec<Out> {
        let mut out = Vec::new();
        match self.state {
            State::Idle => {}
            State::Header => self.malformed(&mut out, "</tool_call>"),
            State::Malformed => out.push(Out::Content("</tool_call>".into())),
            State::Between => {
                if !self.buf.trim().is_empty() {
                    self.drop_text("text in the body dropped");
                }
                self.end_call(&mut out);
            }
            State::Value => {
                self.stop_value(&mut out, true);
                self.end_call(&mut out);
            }
            State::Done => self.end_call(&mut out),
        }
        self.state = State::Idle;
        out
    }

    /// The end in the middle of a call (Length, ContextFull): what the call has is closed as valid JSON.
    pub fn cut_off(&mut self) -> Vec<Out> {
        let mut out = Vec::new();
        match self.state {
            State::Idle | State::Malformed => {}
            State::Header => self.malformed(&mut out, ""),
            State::Between | State::Done | State::Value => {
                if self.state == State::Value {
                    self.stop_value(&mut out, false);
                }
                self.repair("call cut off, closed as valid JSON".into());
                self.end_call(&mut out);
            }
        }
        self.state = State::Idle;
        out
    }

    pub fn repairs(&self) -> &[String] {
        &self.repairs
    }

    /// The valid calls started so far.
    pub fn calls(&self) -> usize {
        self.calls
    }

    /// The calls that went out as content.
    pub fn malformed_count(&self) -> usize {
        self.malformed
    }

    /// A repair line: the tool (its name cut to `NAME_SHOWN` characters) and the repair, never the value's text.
    fn repair(&mut self, what: String) {
        let name: String = self.name.chars().take(NAME_SHOWN).collect();
        self.repairs.push(format!("tool {name:?}: {what}"));
    }

    /// A repair line about the parameter `key`, which the model wrote: the key's byte length and never its text, which
    /// a prompt injection could fill with anything from the context.
    fn repair_key(&mut self, key: &str, what: &str) {
        self.repair(format!("parameter of {} bytes: {what}", key.len()));
    }

    /// The call ends inside a value. One that ends exactly in `\n</parameter>` ends there; any other is a repair if
    /// `repair` (the `</tool_call>` came early).
    fn stop_value(&mut self, out: &mut Vec<Out>, repair: bool) {
        if self.buf.ends_with(VALUE_CLOSE) {
            self.buf.truncate(self.buf.len() - VALUE_CLOSE.len());
        } else if repair {
            let key = self.key.clone();
            self.repair_key(&key, "value closed by the end of the call");
        }
        self.finish_value(out);
    }

    /// The text so far is not a call: it goes out as content, `<tool_call>` first and `tail` last.
    fn malformed(&mut self, out: &mut Vec<Out>, tail: &str) {
        out.push(Out::Content(format!(
            "<tool_call>{}{tail}",
            std::mem::take(&mut self.buf)
        )));
        self.malformed += 1;
        self.state = State::Malformed;
    }

    /// The header: skip whitespace, then `<function=NAME>\n` with a declared NAME, ending within the limit.
    fn header(&mut self, out: &mut Vec<Out>) -> bool {
        let t = self.buf.trim_start();
        let lead = self.buf.len() - t.len();
        let too_long = self.buf.len() >= HEADER_LIMIT;
        let wait = |me: &mut ToolCalls, out: &mut Vec<Out>| {
            if too_long {
                me.malformed(out, "");
            }
            false
        };
        if t.is_empty() {
            return wait(self, out);
        }
        let Some(rest) = t.strip_prefix(FUNCTION_OPEN) else {
            if FUNCTION_OPEN.starts_with(t) {
                return wait(self, out);
            }
            self.malformed(out, "");
            return false;
        };
        let Some(p) = rest.find(['>', '\n', '<']) else {
            return wait(self, out);
        };
        if rest.as_bytes()[p] != b'>' || !self.tools.has(&rest[..p]) {
            self.malformed(out, "");
            return false;
        }
        let after = &rest[p + 1..];
        if after.is_empty() {
            return wait(self, out);
        }
        let end = lead + FUNCTION_OPEN.len() + p + 2;
        if !after.starts_with('\n') || end > HEADER_LIMIT {
            self.malformed(out, "");
            return false;
        }
        self.name = rest[..p].to_string();
        self.buf.drain(..end);
        self.index = self.calls;
        self.calls += 1;
        self.emitted = 0;
        self.keys.clear();
        self.dropping = false;
        self.state = State::Between;
        out.push(Out::CallStart {
            index: self.index,
            name: self.name.clone(),
        });
        true
    }

    /// Between parameters: `<parameter=KEY>\n` opens a value, `</function>` ends the body, other text is dropped.
    fn between(&mut self, out: &mut Vec<Out>) -> bool {
        let t = self.buf.trim_start();
        let lead = self.buf.len() - t.len();
        if t.is_empty() {
            self.buf.clear();
            return false;
        }
        if t.starts_with(FUNCTION_CLOSE) {
            self.buf.drain(..lead + FUNCTION_CLOSE.len());
            self.dropping = false;
            self.state = State::Done;
            return true;
        }
        if let Some(rest) = t.strip_prefix(PARAM_OPEN) {
            match rest.find(['>', '\n', '<']) {
                // a key's end can't be past the limit: no `>` yet within it is no key at all
                None if rest.len() <= KEY_LIMIT => return false,
                Some(p) if p <= KEY_LIMIT && rest.as_bytes()[p] == b'>' => {
                    let after = &rest[p + 1..];
                    if after.is_empty() {
                        return false;
                    }
                    if after.starts_with('\n') {
                        let key = rest[..p].to_string();
                        self.buf.drain(..lead + PARAM_OPEN.len() + p + 2);
                        self.open_value(key, out);
                        return true;
                    }
                }
                _ => {}
            }
        } else if PARAM_OPEN.starts_with(t) || FUNCTION_CLOSE.starts_with(t) {
            return false;
        }
        // Anything else: dropped up to the next `<`.
        let first = t.chars().next().map_or(0, char::len_utf8);
        let cut = lead + first + t[first..].find('<').unwrap_or(t.len() - first);
        self.buf.drain(..cut);
        self.drop_text("text between parameters dropped");
        true
    }

    /// Text after `</function>`: whitespace is silent, anything else is dropped with one repair line.
    fn done(&mut self) -> bool {
        if !self.buf.trim().is_empty() {
            self.drop_text("text after </function> dropped");
        }
        self.buf.clear();
        false
    }

    /// One repair line per run of dropped text.
    fn drop_text(&mut self, what: &str) {
        if !self.dropping {
            self.dropping = true;
            self.repair(what.into());
        }
    }

    fn open_value(&mut self, key: String, out: &mut Vec<Out>) {
        self.dropping = false;
        self.raw.clear();
        self.mode = if self.keys.contains(&key) {
            self.repair_key(&key, "repeated, the first value kept");
            Mode::Drop
        } else {
            self.keys.push(key.clone());
            match self.tools.param(&self.name, &key) {
                ParamTypes::Only(set) if set.len() == 1 && set.contains(&Kind::String) => {
                    let head = format!("{}{}:\"", self.separator(), Value::String(key.clone()));
                    self.emitted += 1;
                    self.args(out, head);
                    Mode::Stream
                }
                types => Mode::Buffer(types),
            }
        };
        self.key = key;
        self.state = State::Value;
    }

    /// `{` before a call's first parameter, `,` before the others.
    fn separator(&self) -> &'static str {
        if self.emitted == 0 {
            "{"
        } else {
            ","
        }
    }

    fn args(&self, out: &mut Vec<Out>, fragment: String) {
        if !fragment.is_empty() {
            out.push(Out::CallArgs {
                index: self.index,
                fragment,
            });
        }
    }

    /// A value: its text up to `\n</parameter>\n` (or `\n</function>`, a repair that ends the body too). The longest
    /// suffix that could start either is held back.
    fn value(&mut self, out: &mut Vec<Out>) -> bool {
        let found = [(VALUE_END, false), (VALUE_END_FUNCTION, true)]
            .into_iter()
            .filter_map(|(d, f)| self.buf.find(d).map(|i| (i, d, f)))
            .min_by_key(|&(i, ..)| i);
        let Some((i, delim, function)) = found else {
            let b = self.buf.as_bytes();
            let hold = (1..VALUE_END.len())
                .rev()
                .find(|&k| {
                    k <= b.len()
                        && [VALUE_END, VALUE_END_FUNCTION]
                            .iter()
                            .any(|d| d.as_bytes().starts_with(&b[b.len() - k..]))
                })
                .unwrap_or(0);
            let release: String = self.buf.drain(..self.buf.len() - hold).collect();
            self.release(out, &release);
            return false;
        };
        let release: String = self.buf.drain(..i).collect();
        self.release(out, &release);
        self.buf.drain(..delim.len());
        self.finish_value_only(out);
        self.dropping = false;
        if function {
            let key = self.key.clone();
            self.repair_key(&key, "value closed by </function>");
            self.state = State::Done;
        } else {
            self.state = State::Between;
        }
        true
    }

    /// Part of the value's text: escaped and out, or held for the conversion.
    fn release(&mut self, out: &mut Vec<Out>, chunk: &str) {
        match self.mode {
            Mode::Stream => {
                let q = Value::String(chunk.to_string()).to_string();
                self.args(out, q[1..q.len() - 1].to_string());
            }
            Mode::Buffer(_) => self.raw.push_str(chunk),
            Mode::Drop => {}
        }
    }

    /// The value ends here, with the held-back text part of it (the call ended inside it).
    fn finish_value(&mut self, out: &mut Vec<Out>) {
        let rest = std::mem::take(&mut self.buf);
        self.release(out, &rest);
        self.finish_value_only(out);
    }

    /// Close the value: the closing quote of a streamed string, or a buffered value's conversion.
    fn finish_value_only(&mut self, out: &mut Vec<Out>) {
        match std::mem::replace(&mut self.mode, Mode::Drop) {
            Mode::Stream => self.args(out, "\"".into()),
            Mode::Buffer(types) => {
                let (v, repair) = convert(&self.raw, &types);
                if let Some(r) = repair {
                    let key = self.key.clone();
                    self.repair_key(&key, &r);
                }
                let head = format!(
                    "{}{}:{v}",
                    self.separator(),
                    Value::String(self.key.clone())
                );
                self.emitted += 1;
                self.args(out, head);
            }
            Mode::Drop => {}
        }
        self.raw.clear();
    }

    /// The end of the arguments and of the call.
    fn end_call(&mut self, out: &mut Vec<Out>) {
        let tail = if self.emitted == 0 { "{}" } else { "}" };
        self.args(out, tail.into());
        out.push(Out::CallEnd { index: self.index });
        self.buf.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convo::ToolDef;
    use serde_json::{json, Value};

    const SEEDS: u64 = 200;

    /// A run's pieces and its repairs.
    type Run = (Vec<Out>, Vec<String>);

    /// splitmix64: a seeded stream, so a failing split reproduces.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// `text` cut at random char boundaries: seed `s` cuts after a char with probability 1/2, 1/4, 1/8 or 1/16
    /// (by `s % 4`), so the chunks run from one character to long runs.
    fn split(text: &str, seed: u64) -> Vec<&str> {
        let mut rng = Rng(seed);
        let denom = 2u64 << (seed % 4);
        let mut chunks = Vec::new();
        let mut start = 0;
        for (i, c) in text.char_indices() {
            let end = i + c.len_utf8();
            if end < text.len() && rng.next().is_multiple_of(denom) {
                chunks.push(&text[start..end]);
                start = end;
            }
        }
        chunks.push(&text[start..]);
        chunks
    }

    #[derive(Clone, Copy)]
    enum End {
        Close,
        CutOff,
        /// Neither: the pieces so far.
        None,
    }

    fn run_chunks(tools: &Arc<ToolSet>, chunks: &[&str], end: End) -> (Vec<Out>, Vec<String>) {
        let mut tc = ToolCalls::new(tools.clone());
        tc.open();
        let mut outs = Vec::new();
        for c in chunks {
            outs.extend(tc.text(c));
        }
        match end {
            End::Close => outs.extend(tc.close()),
            End::CutOff => outs.extend(tc.cut_off()),
            End::None => {}
        }
        (outs, tc.repairs().to_vec())
    }

    fn run(tools: &Arc<ToolSet>, text: &str, seed: u64, end: End) -> (Vec<Out>, Vec<String>) {
        run_chunks(tools, &split(text, seed), end)
    }

    /// Adjacent pieces of one kind (and one call) joined.
    fn merged(outs: Vec<Out>) -> Vec<Out> {
        let mut m: Vec<Out> = Vec::new();
        for o in outs {
            match (m.last_mut(), &o) {
                (Some(Out::Content(a)), Out::Content(b)) => a.push_str(b),
                (Some(Out::Reasoning(a)), Out::Reasoning(b)) => a.push_str(b),
                (
                    Some(Out::CallArgs {
                        index: i,
                        fragment: a,
                    }),
                    Out::CallArgs {
                        index: j,
                        fragment: b,
                    },
                ) if i == j => a.push_str(b),
                _ => m.push(o),
            }
        }
        m
    }

    /// The merged result of `text` for every seed (and one char per chunk, and the whole text in one): all the
    /// same, with the same repairs.
    fn all_seeds(tools: &Arc<ToolSet>, text: &str, end: End) -> (Vec<Out>, Vec<String>) {
        let (first, repairs) = run(tools, text, 0, end);
        let first = merged(first);
        let mut variants: Vec<(String, Run)> = (1..SEEDS)
            .map(|s| (format!("seed {s}"), run(tools, text, s, end)))
            .collect();
        let chars: Vec<String> = text.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        variants.push(("one char per chunk".into(), run_chunks(tools, &chars, end)));
        variants.push(("one chunk".into(), run_chunks(tools, &[text], end)));
        for (what, (outs, r)) in variants {
            assert_eq!(merged(outs), first, "{what}");
            assert_eq!(r, repairs, "{what}: repairs");
        }
        (first, repairs)
    }

    fn tools(defs: &[(&str, Value)]) -> Arc<ToolSet> {
        let defs: Vec<ToolDef> = defs
            .iter()
            .map(|(name, parameters)| ToolDef {
                name: name.to_string(),
                description: String::new(),
                parameters: parameters.clone(),
            })
            .collect();
        Arc::new(ToolSet::new(&defs))
    }

    fn string_tool(name: &str, keys: &[&str]) -> Arc<ToolSet> {
        let props: serde_json::Map<String, Value> = keys
            .iter()
            .map(|k| (k.to_string(), json!({"type": "string"})))
            .collect();
        tools(&[(name, json!({"type": "object", "properties": props}))])
    }

    fn start(index: usize, name: &str) -> Out {
        Out::CallStart {
            index,
            name: name.into(),
        }
    }

    fn args(index: usize, fragment: &str) -> Out {
        Out::CallArgs {
            index,
            fragment: fragment.into(),
        }
    }

    /// A call's pieces from `merged`: (name, arguments text); panics on anything else.
    fn single_call(outs: &[Out]) -> (String, String) {
        match outs {
            [Out::CallStart { index: 0, name }, Out::CallArgs { index: 0, fragment }, Out::CallEnd { index: 0 }] => {
                (name.clone(), fragment.clone())
            }
            other => panic!("not one whole call: {other:?}"),
        }
    }

    #[test]
    fn a_call_streams_its_string_arguments() {
        let ts = string_tool("read_file", &["path"]);
        let text =
            "\n<function=read_file>\n<parameter=path>\nsrc/a b.rs\n</parameter>\n</function>\n";
        let (outs, repairs) = all_seeds(&ts, text, End::Close);
        assert_eq!(
            outs,
            vec![
                start(0, "read_file"),
                args(0, r#"{"path":"src/a b.rs"}"#),
                Out::CallEnd { index: 0 },
            ]
        );
        assert!(repairs.is_empty(), "{repairs:?}");
        for seed in 0..SEEDS {
            let (raw, _) = run(&ts, text, seed, End::Close);
            let first = raw
                .iter()
                .find_map(|o| match o {
                    Out::CallArgs { fragment, .. } => Some(fragment.as_str()),
                    _ => None,
                })
                .unwrap();
            assert!(first.starts_with(r#"{"path":""#), "seed {seed}: {first}");
        }
        // the name goes out before the first argument, and the string streams as it arrives
        let (raw, _) = run_chunks(
            &ts,
            &["\n<function=read_file>\n<parameter=path>\nsr", "c/a"],
            End::None,
        );
        assert_eq!(
            raw,
            vec![
                start(0, "read_file"),
                args(0, r#"{"path":""#),
                args(0, "sr"),
                args(0, "c/a"),
            ]
        );
    }

    #[test]
    fn o6_typed_values_follow_the_schema() {
        let ts = tools(&[(
            "f",
            json!({
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "f": {"type": "number"},
                    "b": {"type": "boolean"},
                    "o": {"type": "object"},
                    "s": {"type": "string"},
                    "m": {"type": ["string", "integer"]},
                    "u": {"description": "no type"},
                    "r": {"$ref": "#/$defs/Mode"},
                    "e": {"enum": [1, 2]},
                    "a": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
                },
                "$defs": {"Mode": {"type": "string"}}
            }),
        )]);
        let values = [
            ("n", "3"),
            ("f", "0.5"),
            ("b", "True"),
            ("o", r#"{"k":1}"#),
            ("s", "123"),
            ("m", "42"),
            ("u", "hello"),
            ("r", "fast"),
            ("e", "2"),
            ("a", "None"),
        ];
        let mut text = String::from("\n<function=f>\n");
        for (k, v) in values {
            text.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
        }
        text.push_str("</function>\n");
        let (outs, repairs) = all_seeds(&ts, &text, End::Close);
        let (name, got) = single_call(&outs);
        assert_eq!(name, "f");
        assert_eq!(
            got,
            r#"{"n":3,"f":0.5,"b":true,"o":{"k":1},"s":"123","m":42,"u":"hello","r":"fast","e":2,"a":null}"#
        );
        assert!(repairs.is_empty(), "{repairs:?}");
    }

    #[test]
    fn o8_a_bad_header_is_content() {
        let ts = string_tool("read_file", &["path"]);
        let long = format!("\n{}", "x".repeat(300));
        let cases = [
            // an undeclared name
            "\n<function=nope>\n<parameter=path>\nx\n</parameter>\n</function>\n".to_string(),
            // no "<function="
            "\nI will read it now.\n".to_string(),
            "\n<functio read_file>\n".to_string(),
            // the name's line doesn't end
            "\n<function=read_file> trailing\n<parameter=path>\nx\n</parameter>\n".to_string(),
            // 300 bytes of text before the header
            format!("{long}<function=read_file>\n<parameter=path>\nx\n</parameter>\n</function>\n"),
            // the header ends past byte 256
            format!(
                "{}<function=read_file>\n<parameter=path>\nx\n</parameter>\n</function>\n",
                " ".repeat(240)
            ),
            // the header ends at byte 257, one past the limit
            format!(
                "{}<function=read_file>\n</function>\n",
                " ".repeat(257 - "<function=read_file>\n".len())
            ),
        ];
        for text in cases {
            let (outs, repairs) = all_seeds(&ts, &text, End::None);
            assert_eq!(outs, vec![Out::Content(format!("<tool_call>{text}"))]);
            assert!(repairs.is_empty(), "{repairs:?}");
            // the closing tag is content too
            let (outs, _) = all_seeds(&ts, &text, End::Close);
            assert_eq!(
                outs,
                vec![Out::Content(format!("<tool_call>{text}</tool_call>"))]
            );
        }
        // a header that ends at byte 256 is whole
        let pad = " ".repeat(256 - "<function=read_file>\n".len());
        let text = format!("{pad}<function=read_file>\n</function>\n");
        let (outs, _) = all_seeds(&ts, &text, End::Close);
        assert_eq!(single_call(&outs), ("read_file".into(), "{}".into()));
        let mut tc = ToolCalls::new(ts);
        tc.open();
        tc.text(&"x".repeat(300));
        assert_eq!(tc.calls(), 0);
    }

    #[test]
    fn o8_bad_values_are_repaired() {
        let ts = tools(&[(
            "g",
            json!({"type": "object", "properties": {"n": {"type": "integer"}, "s": {"type": "string"}}}),
        )]);
        let text = "\n<function=g>\n<parameter=n>\nthree\n</parameter>\n<parameter=n>\n5\n</parameter>\n\
                    <parameter=s>\nfirst\n</parameter>\n<parameter=s>\nsecond\n</parameter>\n</function>\n";
        let (outs, repairs) = all_seeds(&ts, text, End::Close);
        let (_, got) = single_call(&outs);
        assert_eq!(got, r#"{"n":"three","s":"first"}"#);
        // a bad integer, then two repeats, one line each: the tool and the parameter, never the value
        assert_eq!(repairs.len(), 3, "{repairs:?}");
        for line in &repairs {
            assert!(
                line.starts_with("tool \"g\": parameter of 1 bytes: "),
                "{line}"
            );
            assert!(
                !line.contains("three") && !line.contains("second"),
                "{line}"
            );
            assert!(!line.contains('\n'), "{line}");
        }
        // an undeclared parameter is kept, typed any
        let text = "\n<function=g>\n<parameter=zz>\n7\n</parameter>\n<parameter=yy>\nhi\n</parameter>\n</function>";
        let (outs, repairs) = all_seeds(&ts, text, End::Close);
        assert_eq!(single_call(&outs).1, r#"{"zz":7,"yy":"hi"}"#);
        assert!(repairs.is_empty());
    }

    #[test]
    fn a_repair_names_the_parameter_by_its_length() {
        // A key the model wrote may be anything from the prompt's context: the line carries its length only.
        let ts = tools(&[(
            "f",
            json!({"type": "object", "properties": {"secretkey": {"type": "integer"}}}),
        )]);
        let text = "
<function=f>
<parameter=secretkey>
abc
</parameter>
</function>
";
        let (_, repairs) = all_seeds(&ts, text, End::Close);
        assert_eq!(repairs.len(), 1, "{repairs:?}");
        assert!(repairs[0].contains("parameter of 9 bytes"), "{repairs:?}");
        assert!(!repairs[0].contains("secretkey"), "{repairs:?}");
        // The tool's name is cut to 64 characters.
        let long = "t".repeat(100);
        let ts = tools(&[(
            long.as_str(),
            json!({"type": "object", "properties": {"n": {"type": "integer"}}}),
        )]);
        let text = format!(
            "
<function={long}>
<parameter=n>
abc
</parameter>
</function>
"
        );
        let (_, repairs) = all_seeds(&ts, &text, End::Close);
        assert_eq!(repairs.len(), 1, "{repairs:?}");
        assert!(
            repairs[0].starts_with(&format!("tool \"{}\": ", "t".repeat(64))),
            "{repairs:?}"
        );
    }

    #[test]
    fn o8_text_between_parameters_is_dropped_with_a_repair() {
        let ts = string_tool("f", &["a", "b"]);
        let text = "\n<function=f>\nchatter <b>here</b>\n<parameter=a>\n1\n</parameter>\nmore\n\
                    <parameter=b>\n2\n</parameter>\n</function>\n";
        let (outs, repairs) = all_seeds(&ts, text, End::Close);
        assert_eq!(single_call(&outs).1, r#"{"a":"1","b":"2"}"#);
        assert_eq!(repairs.len(), 2, "{repairs:?}");
        assert!(!repairs.iter().any(|l| l.contains("chatter")));
    }

    #[test]
    fn o8_a_cut_off_call_closes_as_valid_json() {
        let ts = tools(&[(
            "f",
            json!({"type": "object", "properties": {"a": {"type": "string"}, "n": {"type": "integer"}}}),
        )]);
        let (outs, repairs) = all_seeds(&ts, "\n<function=f>\n<parameter=a>\nabc", End::CutOff);
        assert_eq!(single_call(&outs).1, r#"{"a":"abc"}"#);
        assert_eq!(repairs.len(), 1, "{repairs:?}");
        // the held-back suffix is part of the value
        let (outs, _) = all_seeds(
            &ts,
            "\n<function=f>\n<parameter=a>\nabc\n</par",
            End::CutOff,
        );
        let args = single_call(&outs).1;
        assert_eq!(
            serde_json::from_str::<Value>(&args).unwrap(),
            json!({"a": "abc\n</par"})
        );
        let (outs, _) = all_seeds(&ts, "\n<function=f>\n<parameter=n>\n12", End::CutOff);
        assert_eq!(single_call(&outs).1, r#"{"n":12}"#);
        // a buffered value that doesn't convert becomes a string
        let (outs, _) = all_seeds(&ts, "\n<function=f>\n<parameter=n>\n12x", End::CutOff);
        assert_eq!(single_call(&outs).1, r#"{"n":"12x"}"#);
        // after a whole parameter, and with no parameter
        let text = "\n<function=f>\n<parameter=a>\nx\n</parameter>\n";
        let (outs, _) = all_seeds(&ts, text, End::CutOff);
        assert_eq!(single_call(&outs).1, r#"{"a":"x"}"#);
        let (outs, _) = all_seeds(&ts, "\n<function=f>\n", End::CutOff);
        assert_eq!(single_call(&outs).1, "{}");
        // in the header: the text as content, with no closing tag
        let (outs, _) = all_seeds(&ts, "\n<function=", End::CutOff);
        assert_eq!(outs, vec![Out::Content("<tool_call>\n<function=".into())]);
        let (outs, _) = all_seeds(&ts, "", End::CutOff);
        assert_eq!(outs, vec![Out::Content("<tool_call>".into())]);
    }

    #[test]
    fn a_call_without_parameters_is_an_empty_object() {
        let ts = string_tool("f", &["a"]);
        let (outs, repairs) = all_seeds(&ts, "\n<function=f>\n</function>\n", End::Close);
        assert_eq!(single_call(&outs), ("f".into(), "{}".into()));
        assert!(repairs.is_empty());
        let (outs, _) = all_seeds(&ts, "\n<function=f>\n", End::Close);
        assert_eq!(single_call(&outs).1, "{}");
    }

    #[test]
    fn o8_the_close_tag_inside_a_value_closes_it() {
        let ts = tools(&[(
            "f",
            json!({"type": "object", "properties": {"a": {"type": "string"}, "n": {"type": "integer"}}}),
        )]);
        let (outs, repairs) = all_seeds(&ts, "\n<function=f>\n<parameter=a>\nabc", End::Close);
        assert_eq!(single_call(&outs).1, r#"{"a":"abc"}"#);
        assert_eq!(repairs.len(), 1, "{repairs:?}");
        assert!(
            repairs[0].starts_with("tool \"f\": parameter of 1 bytes: "),
            "{repairs:?}"
        );
        let (outs, repairs) = all_seeds(&ts, "\n<function=f>\n<parameter=n>\n12", End::Close);
        assert_eq!(single_call(&outs).1, r#"{"n":12}"#);
        assert_eq!(repairs.len(), 1, "{repairs:?}");
        // the held-back suffix is part of the value
        let (outs, _) = all_seeds(&ts, "\n<function=f>\n<parameter=a>\nabc\n</par", End::Close);
        assert_eq!(single_call(&outs).1, r#"{"a":"abc\n</par"}"#);
        // after a whole parameter, an empty value
        let text = "\n<function=f>\n<parameter=a>\nx\n</parameter>\n<parameter=n>\n";
        let (outs, repairs) = all_seeds(&ts, text, End::Close);
        assert_eq!(single_call(&outs).1, r#"{"a":"x","n":""}"#);
        assert_eq!(repairs.len(), 2, "{repairs:?}");
    }

    #[test]
    fn o8_the_function_close_inside_a_value_closes_it_and_the_body() {
        let ts = string_tool("f", &["a", "b"]);
        let text = "\n<function=f>\n<parameter=a>\nabc\n</function>\n";
        let (outs, repairs) = all_seeds(&ts, text, End::Close);
        assert_eq!(single_call(&outs).1, r#"{"a":"abc"}"#);
        assert_eq!(repairs.len(), 1, "{repairs:?}");
    }

    #[test]
    fn o8_the_close_tag_in_the_header_is_content() {
        let ts = string_tool("re", &[]);
        let (outs, repairs) = all_seeds(&ts, "\n<function=re", End::Close);
        assert_eq!(
            outs,
            vec![Out::Content("<tool_call>\n<function=re</tool_call>".into())]
        );
        assert!(repairs.is_empty());
        let mut tc = ToolCalls::new(ts);
        tc.open();
        tc.text("\n<function=re");
        tc.close();
        assert_eq!(tc.calls(), 0);
    }

    #[test]
    fn whitespace_after_function_is_not_a_repair() {
        let ts = string_tool("f", &["a"]);
        let body = "\n<function=f>\n<parameter=a>\nx\n</parameter>\n</function>";
        let (outs, repairs) = all_seeds(&ts, &format!("{body}\n  \n"), End::Close);
        assert_eq!(single_call(&outs).1, r#"{"a":"x"}"#);
        assert!(repairs.is_empty(), "{repairs:?}");
        let (outs, repairs) = all_seeds(&ts, &format!("{body}\nnote"), End::Close);
        assert_eq!(single_call(&outs).1, r#"{"a":"x"}"#);
        assert_eq!(repairs.len(), 1, "{repairs:?}");
        assert!(!repairs[0].contains("note"));
        let (outs, repairs) = all_seeds(&ts, &format!("{body}\nsome note\nand more"), End::Close);
        assert_eq!(single_call(&outs).1, r#"{"a":"x"}"#);
        assert_eq!(repairs.len(), 1, "{repairs:?}");
    }

    #[test]
    fn python_literals_follow_the_allowed_kinds() {
        let ts = tools(&[(
            "p",
            json!({"type": "object", "properties": {
                "b": {"type": "boolean"},
                "n": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "u": {},
                "m": {"type": ["string", "boolean"]},
                "s": {"type": "string"},
                "i": {"type": "integer"}
            }}),
        )]);
        let mut text = String::from("\n<function=p>\n");
        for (k, v) in [
            ("b", "True"),
            ("n", "None"),
            ("u", "False"),
            ("m", "True"),
            ("s", "True"),
            ("i", "True"),
        ] {
            text.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
        }
        text.push_str("</function>\n");
        let (outs, repairs) = all_seeds(&ts, &text, End::Close);
        assert_eq!(
            single_call(&outs).1,
            r#"{"b":true,"n":null,"u":false,"m":true,"s":"True","i":"True"}"#
        );
        assert_eq!(repairs.len(), 1, "{repairs:?}");
        assert!(
            repairs[0].starts_with("tool \"p\": parameter of 1 bytes: "),
            "{repairs:?}"
        );
    }

    #[test]
    fn a_value_ending_in_an_emoji_streams_whole() {
        let ts = string_tool("f", &["a"]);
        let text = "\n<function=f>\n<parameter=a>\nok 🎉\n</parameter>\n</function>\n";
        let (outs, repairs) = all_seeds(&ts, text, End::Close);
        assert_eq!(single_call(&outs).1, "{\"a\":\"ok 🎉\"}");
        assert!(repairs.is_empty());
        // non-ASCII at the delimiter and in a key
        let text = "\n<function=f>\n<parameter=a>\n🎉🎉\n</parameter>\n<parameter=é>\n终止。\n</parameter>\n</function>";
        let (outs, _) = all_seeds(&ts, text, End::Close);
        assert_eq!(single_call(&outs).1, "{\"a\":\"🎉🎉\",\"é\":\"终止。\"}");
        // every two-chunk split of the first text, the cut at each char boundary
        let text = "\n<function=f>\n<parameter=a>\nok 🎉\n</parameter>\n</function>\n";
        for (i, _) in text.char_indices().skip(1) {
            let (outs, repairs) = run_chunks(&ts, &[&text[..i], &text[i..]], End::Close);
            assert_eq!(
                single_call(&merged(outs)).1,
                "{\"a\":\"ok 🎉\"}",
                "cut at {i}"
            );
            assert!(repairs.is_empty(), "cut at {i}: {repairs:?}");
        }
    }

    #[test]
    fn a_value_ends_at_the_newline_after_its_close_tag() {
        // `\n</parameter>X` stays in the value (llama.cpp's until("\n</parameter>\n"))
        let ts = tools(&[(
            "f",
            json!({"type": "object", "properties": {"a": {"type": "string"}, "u": {}}}),
        )]);
        let text = "\n<function=f>\n<parameter=a>\nabc\n</parameter>X and\n</parameter><x>\n</parameter>\n\
                    <parameter=u>\n1\n</parameter>2\n</parameter>\n</function>\n";
        let (outs, repairs) = all_seeds(&ts, text, End::Close);
        let got: Value = serde_json::from_str(&single_call(&outs).1).unwrap();
        assert_eq!(
            got,
            json!({"a": "abc\n</parameter>X and\n</parameter><x>", "u": "1\n</parameter>2"})
        );
        assert!(repairs.is_empty(), "{repairs:?}");
    }

    #[test]
    fn a_call_ending_in_a_close_tag_ends_the_value_cleanly() {
        let ts = tools(&[(
            "f",
            json!({"type": "object", "properties": {"a": {"type": "string"}, "n": {"type": "integer"}}}),
        )]);
        for (text, want) in [
            (
                "\n<function=f>\n<parameter=a>\nabc\n</parameter>",
                r#"{"a":"abc"}"#,
            ),
            (
                "\n<function=f>\n<parameter=n>\n12\n</parameter>",
                r#"{"n":12}"#,
            ),
            (
                "\n<function=f>\n<parameter=a>\n\n</parameter>",
                r#"{"a":""}"#,
            ),
            // not the whole close tag: part of the value, and a repair
            (
                "\n<function=f>\n<parameter=a>\nabc\n</parameter",
                "{\"a\":\"abc\\n</parameter\"}",
            ),
        ] {
            let (outs, repairs) = all_seeds(&ts, text, End::Close);
            assert_eq!(single_call(&outs).1, want, "{text:?}");
            assert_eq!(
                repairs.len(),
                usize::from(want.contains("</param")),
                "{repairs:?}"
            );
            // a cut-off is one repair for the call, none for the value
            let (outs, repairs) = all_seeds(&ts, text, End::CutOff);
            assert_eq!(single_call(&outs).1, want, "{text:?}");
            assert_eq!(repairs.len(), 1, "{repairs:?}");
        }
    }

    #[test]
    fn a_long_key_is_shown_by_its_length_in_the_repair_lines() {
        let ts = tools(&[("g", json!({"type": "object", "properties": {}}))]);
        let key = format!("{}k", "é".repeat(100));
        let text = format!(
            "\n<function=g>\n<parameter={key}>\n1\n</parameter>\n<parameter={key}>\n2\n</parameter>\n</function>\n"
        );
        let (outs, repairs) = all_seeds(&ts, &text, End::Close);
        assert_eq!(single_call(&outs).1, format!("{{\"{key}\":1}}"));
        assert_eq!(repairs.len(), 1, "{repairs:?}");
        // The key's byte length (100 two-byte characters and a `k`), never its text.
        assert!(
            repairs[0].contains("parameter of 201 bytes"),
            "{}",
            repairs[0]
        );
        assert!(repairs[0].len() < 140, "{}", repairs[0]);
        assert!(!repairs[0].contains('é') && !repairs[0].contains(&key));
    }

    #[test]
    fn a_tag_with_no_key_end_within_the_limit_is_dropped_text() {
        let ts = string_tool("f", &["a"]);
        let long = "k".repeat(300);
        let text = format!(
            "\n<function=f>\n<parameter={long}>\nx\n</parameter>\n<parameter=a>\ny\n</parameter>\n</function>\n"
        );
        let (outs, repairs) = all_seeds(&ts, &text, End::Close);
        assert_eq!(single_call(&outs).1, r#"{"a":"y"}"#);
        assert_eq!(repairs.len(), 1, "{repairs:?}");
        // a tag that never ends: dropped, with no wait for it
        let mut tc = ToolCalls::new(ts.clone());
        tc.open();
        tc.text("\n<function=f>\n");
        tc.text(&format!("<parameter={long}"));
        assert_eq!(tc.repairs().len(), 1);
        // a key of exactly the limit is a key
        let key = "k".repeat(256);
        let text = format!("\n<function=f>\n<parameter={key}>\nx\n</parameter>\n</function>\n");
        let (outs, repairs) = all_seeds(&ts, &text, End::Close);
        assert_eq!(single_call(&outs).1, format!("{{\"{key}\":\"x\"}}"));
        assert!(repairs.is_empty());
    }

    #[test]
    fn the_value_keeps_its_newlines() {
        let ts = string_tool("f", &["s"]);
        let text = "\n<function=f>\n<parameter=s>\na\n\nb\n\n</parameter>\n</function>\n";
        let (outs, repairs) = all_seeds(&ts, text, End::Close);
        let got: Value = serde_json::from_str(&single_call(&outs).1).unwrap();
        assert_eq!(got, json!({"s": "a\n\nb\n"}));
        assert!(repairs.is_empty());
        // leading and trailing spaces, tabs, quotes and backslashes survive escaping
        let text =
            "\n<function=f>\n<parameter=s>\n  \t\"q\" \\ x\u{1}\n</parameter>\n</function>\n";
        let (outs, _) = all_seeds(&ts, text, End::Close);
        let got: Value = serde_json::from_str(&single_call(&outs).1).unwrap();
        assert_eq!(got, json!({"s": "  \t\"q\" \\ x\u{1}"}));
        // an empty value
        let text = "\n<function=f>\n<parameter=s>\n\n</parameter>\n</function>\n";
        let (outs, _) = all_seeds(&ts, text, End::Close);
        assert_eq!(single_call(&outs).1, r#"{"s":""}"#);
    }

    #[test]
    fn several_calls_count_from_zero_and_bad_ones_do_not_count() {
        let ts = string_tool("f", &["a"]);
        let mut tc = ToolCalls::new(ts);
        let mut outs = Vec::new();
        for text in [
            "\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n",
            "\n<function=nope>\n",
            "\n<function=f>\n<parameter=a>\n2\n</parameter>\n</function>\n",
        ] {
            tc.open();
            outs.extend(tc.text(text));
            outs.extend(tc.close());
        }
        assert_eq!(tc.calls(), 2);
        let outs = merged(outs);
        assert_eq!(
            outs,
            vec![
                start(0, "f"),
                args(0, r#"{"a":"1"}"#),
                Out::CallEnd { index: 0 },
                Out::Content("<tool_call>\n<function=nope>\n</tool_call>".into()),
                start(1, "f"),
                args(1, r#"{"a":"2"}"#),
                Out::CallEnd { index: 1 },
            ]
        );
    }
}
