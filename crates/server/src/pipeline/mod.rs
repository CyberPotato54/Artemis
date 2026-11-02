//! The output pipeline: the pieces a request's tokens become.
pub mod stop;
pub mod toolcall;

use crate::backend::{Finish, TokenLogprobs};
use crate::schema::ToolSet;
use crate::tokens::Specials;
use qwen36::stream::TextStream;
use qwen36::tokenizer::Tokenizer;
use std::collections::VecDeque;
use std::sync::Arc;
use stop::StopMatcher;
use toolcall::ToolCalls;

/// One piece of a request's output, in the order it goes out.
#[derive(Clone, Debug, PartialEq)]
pub enum Out {
    Reasoning(String),
    Content(String),
    CallStart { index: usize, name: String },
    CallArgs { index: usize, fragment: String },
    CallEnd { index: usize },
    Logprob(LogprobEntry),
    Stopped { matched: String },
}

/// The log-probability of one sampled token and its top alternatives.
#[derive(Clone, Debug, PartialEq)]
pub struct LogprobEntry {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top: Vec<(String, f32, Vec<u8>)>,
}

/// What a request needs of its pipeline.
#[derive(Clone)]
pub struct PipelineSetup {
    /// The generation prompt opened `<think>`: the output starts inside reasoning.
    pub thinking: bool,
    /// The request's tools: `<tool_call>` opens a call. None: `<tool_call>` is content text.
    pub tools: Option<Arc<ToolSet>>,
    pub stop: Vec<String>,
}

/// What the pipeline saw, for the response's usage and the request's log line.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    /// Valid calls started.
    pub calls: usize,
    /// `<tool_call>` texts returned as content: not valid calls.
    pub malformed_calls: usize,
    /// The engine forced the think budget's close (a forced token came through).
    pub budget_closed: bool,
    /// Tokens put in reasoning, tags excluded, forced tokens included.
    pub reasoning_tokens: u32,
    /// Repair lines, never carrying output text.
    pub repairs: Vec<String>,
    /// The stop string that matched.
    pub stopped: Option<String>,
}

/// The tag token ids the split works on.
struct Tags {
    think: u32,
    end_think: u32,
    tool_call: u32,
    end_tool_call: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Reasoning,
    /// Content, and AfterCall (content that starts by dropping whitespace: `Strip::Space`).
    Content,
    Call,
}

/// What the next content text starts by dropping.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Strip {
    No,
    /// The newlines after `</think>`.
    Newlines,
    /// The whitespace after a call.
    Space,
}

/// One segment's token ids and their `TextStream`: a split character never crosses segments, and a tag never enters
/// the text.
struct Seg {
    ids: Vec<u32>,
    text: TextStream,
}

impl Seg {
    fn new() -> Seg {
        Seg {
            ids: Vec::new(),
            text: TextStream::new(),
        }
    }

    fn push(&mut self, id: u32, tk: &Tokenizer) -> String {
        self.ids.push(id);
        self.text.next_tokens(&self.ids, |ids| tk.decode(ids))
    }

    /// The rest of the segment, a dangling partial character included.
    fn finish(&mut self, tk: &Tokenizer) -> String {
        self.text.finish_tokens(&self.ids, |ids| tk.decode(ids))
    }
}

/// A logprob entry waiting for its text: it goes out once `end` bytes of the stream it counts in have gone out.
struct Held {
    entry: LogprobEntry,
    end: u64,
}

/// Turns one request's tokens into output pieces: the reasoning split on token ids, the tool-call parser
/// in a call, the stop strings on content text, and the logprob entries that belong to the text released.
///
/// Two byte counts place a content token's entry. `g` counts the raw bytes of every content token, wherever the
/// token's text goes; a token's `end` in `g` is its last byte. `s` counts the content text after the dropped leading
/// whitespace, which is what the stop matcher sees and releases. An entry waits first for `TextStream` to release
/// its last byte (`g`), then for the matcher to (`s`).
pub struct Pipeline {
    tk: Arc<Tokenizer>,
    tags: Tags,
    calls: Option<ToolCalls>,
    stops: StopMatcher,
    mode: Mode,
    seg: Seg,
    /// Reasoning: text held back (a trailing `\n`, or everything while it is whitespace only), and whether any
    /// non-whitespace text has come.
    held_text: String,
    started: bool,
    /// The segment follows a `<think>` token in content: it drops its leading whitespace (llama.cpp opens such a
    /// segment with `<think>` and a space, not the template's newline).
    lead_strip: bool,
    reasoning_tokens: u32,
    /// Content: what the next text drops.
    strip: Strip,
    /// Raw bytes of content tokens fed, and the part `TextStream` has released.
    g_fed: u64,
    g_rel: u64,
    /// The last (up to three) raw bytes fed, to tell whether the last token ends inside a character.
    tail: Vec<u8>,
    /// Content text handed to the matcher, and released by it.
    s_in: u64,
    s_out: u64,
    /// Entries `TextStream` has not released yet (`end` in `g`), and those the matcher has not (`end` in `s`).
    unreleased: VecDeque<Held>,
    held: VecDeque<Held>,
    repairs: Vec<String>,
    stopped: Option<String>,
    budget_closed: bool,
    finished: bool,
}

impl Pipeline {
    pub fn new(setup: PipelineSetup, tk: Arc<Tokenizer>, sp: &Specials) -> Pipeline {
        let mut p = Pipeline {
            tk,
            tags: Tags {
                think: sp.think,
                end_think: sp.end_think,
                tool_call: sp.tool_call,
                end_tool_call: sp.end_tool_call,
            },
            calls: setup.tools.map(ToolCalls::new),
            stops: StopMatcher::new(setup.stop),
            mode: Mode::Content,
            seg: Seg::new(),
            held_text: String::new(),
            started: false,
            lead_strip: false,
            reasoning_tokens: 0,
            strip: Strip::No,
            g_fed: 0,
            g_rel: 0,
            tail: Vec::new(),
            s_in: 0,
            s_out: 0,
            unreleased: VecDeque::new(),
            held: VecDeque::new(),
            repairs: Vec::new(),
            stopped: None,
            budget_closed: false,
            finished: false,
        };
        if setup.thinking {
            p.enter_reasoning(false);
        }
        p
    }

    /// One sampled or forced token. `logprobs` is None for a forced token.
    pub fn push(&mut self, id: u32, logprobs: Option<&TokenLogprobs>) -> Vec<Out> {
        let mut out = Vec::new();
        if self.stopped.is_some() || self.finished {
            return out;
        }
        let t = &self.tags;
        let (think, end_think, tool_call, end_tool_call) =
            (t.think, t.end_think, t.tool_call, t.end_tool_call);
        let tools = self.calls.is_some();
        match self.mode {
            Mode::Reasoning if id == end_think => {
                self.end_reasoning(true, &mut out);
                self.enter_content(Strip::Newlines);
            }
            Mode::Reasoning if id == tool_call => {
                self.end_reasoning(false, &mut out);
                if tools {
                    self.enter_call();
                } else {
                    self.enter_content(Strip::No);
                    self.content_token(id, logprobs, &mut out);
                }
            }
            Mode::Reasoning if id == think => self.repair("<think> in reasoning dropped"),
            Mode::Reasoning => {
                self.reasoning_tokens += 1;
                let text = self.seg.push(id, &self.tk);
                self.reasoning_text(&text, &mut out);
            }
            Mode::Content if id == think => {
                self.end_content(&mut out);
                if self.stopped.is_none() {
                    self.enter_reasoning(true);
                }
            }
            Mode::Content if id == end_think => self.repair("stray </think> in content dropped"),
            Mode::Content if id == tool_call && tools => {
                self.end_content(&mut out);
                if self.stopped.is_none() {
                    self.enter_call();
                }
            }
            Mode::Content => self.content_token(id, logprobs, &mut out),
            Mode::Call if id == end_tool_call => {
                // Whitespace after a call is dropped; after a malformed one (all content) it is text.
                let ended = self.end_call(true, &mut out);
                if self.stopped.is_none() {
                    self.enter_content(if ended { Strip::Space } else { Strip::No });
                }
            }
            Mode::Call if id == think || id == end_think => {
                self.repair("think tag in a tool call dropped");
            }
            Mode::Call => {
                let text = self.seg.push(id, &self.tk);
                self.call_text(&text, &mut out);
            }
        }
        out
    }

    /// The end of the request: what is held back goes out, and an open call is closed (`Stop(</tool_call>)`) or cut
    /// off (any other end).
    pub fn finish(&mut self, finish: &Finish) -> Vec<Out> {
        let mut out = Vec::new();
        if self.stopped.is_some() || self.finished {
            return out;
        }
        self.finished = true;
        match self.mode {
            Mode::Reasoning => self.end_reasoning(false, &mut out),
            Mode::Content => self.end_content(&mut out),
            Mode::Call => {
                let closed = matches!(finish, Finish::Stop(id) if *id == self.tags.end_tool_call);
                self.end_call(closed, &mut out);
                self.flush_stops(&mut out);
            }
        }
        out
    }

    /// A token the engine forced came through: the think budget closed the reasoning.
    /// The think-budget close is the only sequence the engine forces today, so `budget_closed` keys on any forced token.
    pub fn note_forced(&mut self) {
        self.budget_closed = true;
    }

    pub fn summary(&self) -> Summary {
        let mut repairs = self.repairs.clone();
        if let Some(c) = &self.calls {
            repairs.extend(c.repairs().iter().cloned());
        }
        Summary {
            calls: self.calls.as_ref().map_or(0, ToolCalls::calls),
            malformed_calls: self.calls.as_ref().map_or(0, ToolCalls::malformed_count),
            budget_closed: self.budget_closed,
            reasoning_tokens: self.reasoning_tokens,
            repairs,
            stopped: self.stopped.clone(),
        }
    }

    /// A repair line: what was dropped, never its text.
    fn repair(&mut self, what: &str) {
        self.repairs.push(what.to_string());
    }

    fn enter_reasoning(&mut self, lead_strip: bool) {
        self.mode = Mode::Reasoning;
        self.seg = Seg::new();
        self.held_text.clear();
        self.started = false;
        self.lead_strip = lead_strip;
    }

    fn enter_content(&mut self, strip: Strip) {
        self.mode = Mode::Content;
        self.seg = Seg::new();
        self.strip = strip;
    }

    fn enter_call(&mut self) {
        self.mode = Mode::Call;
        self.seg = Seg::new();
        if let Some(c) = &mut self.calls {
            c.open();
        }
    }

    /// Reasoning text: everything is held while it is whitespace only; after that one trailing `\n` is held, in case
    /// `</think>` follows.
    fn reasoning_text(&mut self, text: &str, out: &mut Vec<Out>) {
        if text.is_empty() {
            return;
        }
        self.held_text.push_str(text);
        if !self.started {
            if self.held_text.trim().is_empty() {
                return;
            }
            self.started = true;
            if self.lead_strip {
                self.held_text = self.held_text.trim_start().to_string();
            }
        }
        let keep = usize::from(self.held_text.ends_with('\n'));
        let cut = self.held_text.len() - keep;
        if cut > 0 {
            let rest = self.held_text.split_off(cut);
            out.push(Out::Reasoning(std::mem::replace(&mut self.held_text, rest)));
        }
    }

    /// The reasoning segment ends. `</think>` drops the held `\n`; any other end (`<tool_call>`, the end of the
    /// output) emits it. Whitespace-only reasoning is never emitted.
    fn end_reasoning(&mut self, close_think: bool, out: &mut Vec<Out>) {
        let text = self.seg.finish(&self.tk);
        self.reasoning_text(&text, out);
        let held = std::mem::take(&mut self.held_text);
        if self.started && !close_think && !held.is_empty() {
            out.push(Out::Reasoning(held));
        }
        self.started = false;
    }

    /// A token in content: its entry waits for its text.
    fn content_token(&mut self, id: u32, logprobs: Option<&TokenLogprobs>, out: &mut Vec<Out>) {
        let bytes = self.tk.token_bytes(id);
        self.g_fed += bytes.len() as u64;
        self.tail.extend(&bytes);
        let drop = self.tail.len().saturating_sub(3);
        self.tail.drain(..drop);
        if let Some(lp) = logprobs {
            let entry = self.entry(bytes, lp);
            self.unreleased.push_back(Held {
                entry,
                end: self.g_fed,
            });
        }
        let text = self.seg.push(id, &self.tk);
        let g_rel = self.g_fed - incomplete_tail(&self.tail);
        self.content_text(&text, g_rel, out);
    }

    fn entry(&self, bytes: Vec<u8>, lp: &TokenLogprobs) -> LogprobEntry {
        let lossy = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
        LogprobEntry {
            token: lossy(&bytes),
            logprob: lp.logprob,
            top: lp
                .top
                .iter()
                .map(|&(id, p)| {
                    let b = self.tk.token_bytes(id);
                    (lossy(&b), p, b)
                })
                .collect(),
            bytes,
        }
    }

    /// Text `TextStream` released, with the raw bytes released so far (`g_rel`): drop the leading whitespace the mode
    /// asks for, hand the rest to the matcher, and send what it releases with the entries it completes.
    fn content_text(&mut self, text: &str, g_rel: u64, out: &mut Vec<Out>) {
        let n = match self.strip {
            Strip::No => 0,
            Strip::Newlines => text.len() - text.trim_start_matches('\n').len(),
            Strip::Space => text.len() - text.trim_start().len(),
        };
        let pass = &text[n..];
        if !pass.is_empty() {
            self.strip = Strip::No;
        }
        // The released text starts at `g_rel`'s old value; the dropped part is its first `n` bytes.
        let base = self.g_rel + n as u64;
        while self.unreleased.front().is_some_and(|h| h.end <= g_rel) {
            let mut h = self.unreleased.pop_front().expect("checked");
            if h.end <= base {
                continue;
            }
            h.end = self.s_in + (h.end - base).min(pass.len() as u64);
            self.held.push_back(h);
        }
        self.g_rel = g_rel.max(self.g_rel);
        self.s_in += pass.len() as u64;
        let (released, matched) = self.stops.push(pass);
        self.release(&released, matched, out);
    }

    /// Matcher output: the text, then the entries whose last byte is in what has gone out. A match drops the rest.
    fn release(&mut self, text: &str, matched: Option<String>, out: &mut Vec<Out>) {
        if !text.is_empty() {
            out.push(Out::Content(text.to_string()));
            self.s_out += text.len() as u64;
        }
        while self.held.front().is_some_and(|h| h.end <= self.s_out) {
            let h = self.held.pop_front().expect("checked");
            out.push(Out::Logprob(h.entry));
        }
        if let Some(m) = matched {
            self.held.clear();
            self.unreleased.clear();
            self.stopped = Some(m.clone());
            out.push(Out::Stopped { matched: m });
        }
    }

    /// The content segment ends (a `<think>` or `<tool_call>` token, or the end of the output): everything held in
    /// it goes out, so a stop string never spans two segments.
    fn end_content(&mut self, out: &mut Vec<Out>) {
        let text = self.seg.finish(&self.tk);
        self.tail.clear();
        self.content_text(&text, self.g_fed, out);
        self.flush_stops(out);
        self.unreleased.clear();
    }

    /// Text inside a call goes to the parser. What the parser writes goes out as it is, except its content (a
    /// malformed call's text, which is returned as content): that meets the stop strings like any content, so a
    /// stop fires the same whether or not tools were declared. It has no logprob entries: the tokens of a call carry
    /// none, so OpenAI's `logprobs.content` may not cover a malformed call's text (a known limit).
    fn call_text(&mut self, text: &str, out: &mut Vec<Out>) {
        if text.is_empty() {
            return;
        }
        if let Some(c) = &mut self.calls {
            let pieces = c.text(text);
            self.route(pieces, out);
        }
    }

    /// Parser output to the response. After a stop match nothing more is forwarded (a malformed call never started
    /// a call, so nothing is left open).
    fn route(&mut self, pieces: Vec<Out>, out: &mut Vec<Out>) {
        for piece in pieces {
            if self.stopped.is_some() {
                break;
            }
            match piece {
                Out::Content(s) => {
                    self.s_in += s.len() as u64;
                    let (released, matched) = self.stops.push(&s);
                    self.release(&released, matched, out);
                }
                other => out.push(other),
            }
        }
    }

    /// The call ends: `</tool_call>` or the stop token close it; any other end cuts it off, closed as valid JSON.
    /// Whether it ended a valid call (it wrote a `CallEnd`).
    fn end_call(&mut self, closed: bool, out: &mut Vec<Out>) -> bool {
        let text = self.seg.finish(&self.tk);
        self.call_text(&text, out);
        let Some(c) = &mut self.calls else {
            return false;
        };
        let pieces = if closed { c.close() } else { c.cut_off() };
        let ended = pieces.iter().any(|p| matches!(p, Out::CallEnd { .. }));
        self.route(pieces, out);
        ended
    }

    /// What the matcher holds back is not a stop string: it goes out.
    fn flush_stops(&mut self, out: &mut Vec<Out>) {
        if self.stopped.is_none() {
            let rest = self.stops.flush();
            self.release(&rest, None, out);
        }
    }
}

/// How many bytes at the end of `tail` (the last bytes fed) are the start of a character still to come.
///
/// Limit: `g` counts raw token bytes and the text counts decoded bytes. They agree for valid UTF-8, which is what the
/// model's tokens spell; where the decoder substitutes they differ (one stray invalid byte decodes to the three bytes
/// of `U+FFFD`). A real `U+FFFD` the model sampled does the same, since `TextStream` holds any trailing `U+FFFD` back
/// as if it were a partial character, while this function counts its bytes as whole. Past such a byte an entry can go
/// out a piece early, and one for a token a stop string then cuts can still be listed. The error is bounded by the
/// substitutions in the output; no entry is lost or reordered.
fn incomplete_tail(tail: &[u8]) -> u64 {
    let Some(i) = tail.iter().rposition(|&b| b & 0xC0 != 0x80) else {
        return 0;
    };
    let need = match tail[i] {
        0xF0..=0xF7 => 4,
        0xE0..=0xEF => 3,
        0xC0..=0xDF => 2,
        _ => 1,
    };
    let have = tail.len() - i;
    if have < need {
        have as u64
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convo::ToolDef;
    use qwen36::tokenizer::Vocab;
    use serde_json::json;
    use std::sync::OnceLock;

    const SEEDS: u64 = 100;

    struct World {
        tk: Arc<Tokenizer>,
        sp: Specials,
        /// The id of each single-byte token.
        byte_ids: Vec<u32>,
        vocab_len: u32,
    }

    /// The tokenizer and ids from the committed vocabulary extract, once per binary.
    fn world() -> &'static World {
        static W: OnceLock<World> = OnceLock::new();
        W.get_or_init(|| {
            let b = std::fs::read(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/tokenizer/qwen36-vocab.bin"
            ))
            .unwrap();
            let v = Vocab::from_extract(&b).unwrap();
            let tk = Tokenizer::new(&v).unwrap();
            let sp = Specials::new(&v, &tk).unwrap();
            let vocab_len = v.tokens.len() as u32;
            let mut byte_ids = vec![u32::MAX; 256];
            for id in 0..vocab_len {
                let t = tk.token_bytes(id);
                if t.len() == 1 && byte_ids[t[0] as usize] == u32::MAX {
                    byte_ids[t[0] as usize] = id;
                }
            }
            assert!(
                byte_ids.iter().all(|&i| i != u32::MAX),
                "a byte has no token"
            );
            World {
                tk: Arc::new(tk),
                sp,
                byte_ids,
                vocab_len,
            }
        })
    }

    /// xorshift64*, seeded: a failing split reproduces.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Rng {
            Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }

        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 16
        }
    }

    fn enc(s: &str) -> Vec<u32> {
        if s.is_empty() {
            return Vec::new();
        }
        world().tk.encode(s, false)
    }

    /// A run of bytes as single-byte tokens (a character cut anywhere).
    fn bytes_ids(b: &[u8]) -> Vec<u32> {
        b.iter().map(|&b| world().byte_ids[b as usize]).collect()
    }

    /// A plain run (no tags) in seeded random chunks. With `bytewise`, a non-ASCII character becomes its bytes as
    /// single tokens half the time, so tokens end inside characters.
    fn plain_ids(run: &str, rng: &mut Rng, seed: u64, bytewise: bool) -> Vec<u32> {
        let denom = 2u64 << (seed % 4);
        let mut chunks: Vec<&str> = Vec::new();
        let mut start = 0;
        for (i, c) in run.char_indices() {
            let end = i + c.len_utf8();
            if end < run.len() && rng.next().is_multiple_of(denom) {
                chunks.push(&run[start..end]);
                start = end;
            }
        }
        chunks.push(&run[start..]);
        let mut ids = Vec::new();
        for chunk in chunks {
            let mut pending = String::new();
            for c in chunk.chars() {
                if bytewise && !c.is_ascii() && rng.next().is_multiple_of(2) {
                    ids.extend(enc(&std::mem::take(&mut pending)));
                    ids.extend(bytes_ids(c.to_string().as_bytes()));
                } else {
                    pending.push(c);
                }
            }
            ids.extend(enc(&pending));
        }
        ids
    }

    /// `text` as the model would sample it: each tag its single special id, each plain run in random chunks.
    fn ids_of(text: &str, seed: u64, bytewise: bool) -> Vec<u32> {
        let sp = &world().sp;
        let tags = [
            ("<think>", sp.think),
            ("</think>", sp.end_think),
            ("<tool_call>", sp.tool_call),
            ("</tool_call>", sp.end_tool_call),
        ];
        let mut rng = Rng::new(seed);
        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let next = tags
                .iter()
                .filter_map(|(t, id)| rest.find(t).map(|i| (i, t.len(), *id)))
                .min();
            let (run, tag) = match next {
                Some((i, len, id)) => (&rest[..i], Some((len, id))),
                None => (rest, None),
            };
            ids.extend(plain_ids(run, &mut rng, seed, bytewise));
            rest = &rest[run.len()..];
            if let Some((len, id)) = tag {
                ids.push(id);
                rest = &rest[len..];
            }
        }
        ids
    }

    fn tool_f() -> Option<Arc<ToolSet>> {
        Some(Arc::new(ToolSet::new(&[ToolDef {
            name: "f".into(),
            description: String::new(),
            parameters: json!({"type": "object", "properties": {"a": {"type": "string"}}}),
        }])))
    }

    fn setup(thinking: bool, tools: Option<Arc<ToolSet>>, stop: &[&str]) -> PipelineSetup {
        PipelineSetup {
            thinking,
            tools,
            stop: stop.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn lp_of(id: u32) -> TokenLogprobs {
        TokenLogprobs {
            logprob: -0.5,
            top: vec![(id, -0.5), (id, -1.5)],
        }
    }

    /// The pieces and summary of a request whose sampled tokens are `ids`, each with logprobs if `lp`.
    fn drive(s: &PipelineSetup, ids: &[u32], lp: bool, fin: Finish) -> (Vec<Out>, Summary) {
        let w = world();
        let mut p = Pipeline::new(s.clone(), w.tk.clone(), &w.sp);
        let mut outs = Vec::new();
        for &id in ids {
            let l = lp.then(|| lp_of(id));
            outs.extend(p.push(id, l.as_ref()));
        }
        outs.extend(p.finish(&fin));
        (outs, p.summary())
    }

    fn end() -> Finish {
        Finish::Stop(world().sp.im_end)
    }

    /// Without the logprob entries, adjacent pieces of one kind (and one call) joined.
    fn merged(outs: &[Out]) -> Vec<Out> {
        let mut m: Vec<Out> = Vec::new();
        for o in outs {
            match (m.last_mut(), o) {
                (_, Out::Logprob(_)) => {}
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
                _ => m.push(o.clone()),
            }
        }
        m
    }

    fn reasoning(s: &str) -> Out {
        Out::Reasoning(s.into())
    }

    fn content(s: &str) -> Out {
        Out::Content(s.into())
    }

    fn call_start(index: usize, name: &str) -> Out {
        Out::CallStart {
            index,
            name: name.into(),
        }
    }

    fn call_args(index: usize, fragment: &str) -> Out {
        Out::CallArgs {
            index,
            fragment: fragment.into(),
        }
    }

    fn call_end(index: usize) -> Out {
        Out::CallEnd { index }
    }

    fn stopped(matched: &str) -> Out {
        Out::Stopped {
            matched: matched.into(),
        }
    }

    /// The entry the pipeline writes for a token sampled with `lp_of`.
    fn entry(id: u32) -> Out {
        let b = world().tk.token_bytes(id);
        let t = String::from_utf8_lossy(&b).into_owned();
        Out::Logprob(LogprobEntry {
            token: t.clone(),
            logprob: -0.5,
            bytes: b.clone(),
            top: vec![(t.clone(), -0.5, b.clone()), (t, -1.5, b)],
        })
    }

    /// `text` through the pipeline for every seed (and the byte-split variants), merged and checked against `want`;
    /// returns the summary of the last run.
    fn check(s: &PipelineSetup, text: &str, want: &[Out]) -> Summary {
        let mut last = None;
        for seed in 0..SEEDS {
            for bytewise in [false, true] {
                let (outs, sum) = drive(s, &ids_of(text, seed, bytewise), false, end());
                assert_eq!(merged(&outs), want, "seed {seed} bytewise {bytewise}");
                last = Some(sum);
            }
        }
        last.unwrap()
    }

    #[test]
    fn o3_reasoning_then_answer() {
        let sum = check(
            &setup(true, None, &[]),
            "Let me think.\n</think>\n\nThe answer.",
            &[reasoning("Let me think."), content("The answer.")],
        );
        assert!(sum.repairs.is_empty());
        assert_eq!(sum.stopped, None);
    }

    #[test]
    fn o3_a_think_tag_in_content_goes_to_reasoning() {
        let sum = check(
            &setup(false, None, &[]),
            "Hi <think>secret</think> there",
            &[content("Hi "), reasoning("secret"), content(" there")],
        );
        assert!(sum.repairs.iter().all(|r| !r.contains("secret")));
    }

    #[test]
    fn a_think_tag_in_content_drops_the_reasoning_s_leading_whitespace() {
        // llama.cpp opens such a segment with `<think>` and a space: the newline after the tag is not reasoning.
        check(
            &setup(false, None, &[]),
            "Hi <think>\n  secret</think> there",
            &[content("Hi "), reasoning("secret"), content(" there")],
        );
        // Whitespace inside the text is kept.
        check(
            &setup(false, None, &[]),
            "<think>\n\none\n\ntwo</think>x",
            &[reasoning("one\n\ntwo"), content("x")],
        );
    }

    #[test]
    fn whitespace_only_reasoning_is_omitted() {
        check(
            &setup(true, None, &[]),
            "\n\n</think>\n\nAnswer",
            &[content("Answer")],
        );
        // Ended by a call, or by the end of the output, the same.
        check(
            &setup(true, tool_f(), &[]),
            " \n<tool_call>\n<function=f>\n</function>\n</tool_call>",
            &[call_start(0, "f"), call_args(0, "{}"), call_end(0)],
        );
        for seed in 0..SEEDS {
            let (outs, _) = drive(
                &setup(true, None, &[]),
                &ids_of("\n\n", seed, false),
                false,
                end(),
            );
            assert_eq!(outs, vec![], "seed {seed}");
        }
    }

    #[test]
    fn a_reasoning_newline_held_at_the_end_is_emitted() {
        for seed in 0..SEEDS {
            let (outs, _) = drive(
                &setup(true, None, &[]),
                &ids_of("think\n", seed, false),
                false,
                end(),
            );
            assert_eq!(merged(&outs), vec![reasoning("think\n")], "seed {seed}");
        }
    }

    #[test]
    fn tool_call_ends_reasoning() {
        let sum = check(
            &setup(true, tool_f(), &[]),
            "plan\n<tool_call>\n<function=f>\n</function>\n</tool_call>",
            &[
                reasoning("plan\n"),
                call_start(0, "f"),
                call_args(0, "{}"),
                call_end(0),
            ],
        );
        assert_eq!(sum.calls, 1);
    }

    #[test]
    fn without_tools_a_tool_call_is_content() {
        let call = "<tool_call>\n<function=f>\n</function>\n</tool_call>";
        let sum = check(
            &setup(true, None, &[]),
            &format!("plan\n{call}"),
            &[reasoning("plan\n"), content(call)],
        );
        assert_eq!(sum.calls, 0);
        check(&setup(false, None, &[]), call, &[content(call)]);
    }

    #[test]
    fn o11_stop_strings_split_across_tokens() {
        // Two tokens that split the stop string.
        let mut ids = enc("abc ST");
        ids.extend(enc("OP def"));
        let (outs, sum) = drive(&setup(false, None, &["STOP"]), &ids, false, end());
        assert_eq!(merged(&outs), vec![content("abc "), stopped("STOP")]);
        assert_eq!(sum.stopped.as_deref(), Some("STOP"));
        // And any other split.
        for seed in 0..SEEDS {
            let (outs, sum) = drive(
                &setup(false, None, &["STOP"]),
                &ids_of("abc STOP def", seed, false),
                false,
                Finish::Cancelled,
            );
            assert_eq!(
                merged(&outs),
                vec![content("abc "), stopped("STOP")],
                "seed {seed}"
            );
            assert_eq!(sum.stopped.as_deref(), Some("STOP"));
        }
    }

    #[test]
    fn stop_strings_find_non_ascii_across_splits() {
        let s = setup(false, None, &["。", "终止"]);
        let cases = [
            ("你好。再见", "你好", "。"),
            ("好的终止吧", "好的", "终止"),
            ("终止", "", "终止"),
            ("a终止", "a", "终止"),
        ];
        for (text, want, hit) in cases {
            for seed in 0..SEEDS {
                for bytewise in [false, true] {
                    let (outs, sum) = drive(&s, &ids_of(text, seed, bytewise), false, end());
                    let mut expect = Vec::new();
                    if !want.is_empty() {
                        expect.push(content(want));
                    }
                    expect.push(stopped(hit));
                    assert_eq!(
                        merged(&outs),
                        expect,
                        "{text} seed {seed} bytewise {bytewise}"
                    );
                    assert_eq!(sum.stopped.as_deref(), Some(hit));
                }
            }
        }
        // Every character cut into its bytes in turn.
        let text = "你好。再见";
        for (at, c) in text.char_indices() {
            let mut ids = enc(&text[..at]);
            ids.extend(bytes_ids(c.to_string().as_bytes()));
            ids.extend(enc(&text[at + c.len_utf8()..]));
            let (outs, _) = drive(&s, &ids, false, end());
            assert_eq!(
                merged(&outs),
                vec![content("你好"), stopped("。")],
                "cut at {at}"
            );
        }
    }

    #[test]
    fn stop_strings_ignore_reasoning_and_calls() {
        let call = "<tool_call>\n<function=f>\n<parameter=a>\nxSTOPy\n</parameter>\n</function>\n</tool_call>";
        for seed in 0..SEEDS {
            let (outs, sum) = drive(
                &setup(true, tool_f(), &["STOP"]),
                &ids_of("think STOP\n</think>\n\nfine", seed, false),
                false,
                end(),
            );
            assert_eq!(
                merged(&outs),
                vec![reasoning("think STOP"), content("fine")],
                "seed {seed}"
            );
            assert_eq!(sum.stopped, None);
            let (outs, sum) = drive(
                &setup(false, tool_f(), &["STOP"]),
                &ids_of(call, seed, false),
                false,
                end(),
            );
            assert_eq!(
                merged(&outs),
                vec![
                    call_start(0, "f"),
                    call_args(0, "{\"a\":\"xSTOPy\"}"),
                    call_end(0)
                ],
                "seed {seed}"
            );
            assert_eq!(sum.stopped, None);
        }
    }

    #[test]
    fn a_stop_string_does_not_span_a_reasoning_segment() {
        // The held "ST" is released when the content is interrupted, so the segments keep their order.
        for seed in 0..SEEDS {
            let (outs, sum) = drive(
                &setup(false, None, &["STOP"]),
                &ids_of("ST<think>r</think>OP", seed, false),
                false,
                end(),
            );
            assert_eq!(
                merged(&outs),
                vec![content("ST"), reasoning("r"), content("OP")],
                "seed {seed}"
            );
            assert_eq!(sum.stopped, None);
        }
    }

    #[test]
    fn after_a_stop_push_returns_nothing() {
        let w = world();
        let mut p = Pipeline::new(setup(false, None, &["x"]), w.tk.clone(), &w.sp);
        let first = p.push(enc("x")[0], None);
        assert_eq!(first, vec![stopped("x")]);
        for id in enc("more text") {
            assert_eq!(p.push(id, Some(&lp_of(id))), vec![]);
        }
        assert_eq!(p.finish(&Finish::Cancelled), vec![]);
        assert_eq!(p.summary().stopped.as_deref(), Some("x"));
    }

    #[test]
    fn o12_logprobs_are_for_content_tokens_only() {
        // Every token carries logprobs here, forced or not: only the content ones get an entry.
        let text = concat!(
            "r\n</think>\n\nHi<tool_call>\n<function=f>\n<parameter=a>\nv\n</parameter>\n",
            "</function>\n</tool_call>\n"
        );
        for seed in 0..SEEDS {
            let (outs, _) = drive(
                &setup(true, tool_f(), &[]),
                &ids_of(text, seed, false),
                true,
                end(),
            );
            let got: Vec<u8> = outs
                .iter()
                .filter_map(|o| match o {
                    Out::Logprob(e) => Some(e.bytes.clone()),
                    _ => None,
                })
                .flatten()
                .collect();
            assert_eq!(got, b"Hi", "seed {seed}");
        }
        // And a content token without logprobs (a forced one) has none.
        let (outs, _) = drive(&setup(false, None, &[]), &enc("Hello"), false, end());
        assert!(outs.iter().all(|o| !matches!(o, Out::Logprob(_))));
    }

    #[test]
    fn o12_logprob_entries_follow_the_text_they_release() {
        let w = world();
        let hello = enc("Hello");
        let space = enc(" ");
        let world_tok = enc(" world");
        assert_eq!((hello.len(), space.len(), world_tok.len()), (1, 1, 1));
        let party = bytes_ids("🎉".as_bytes());
        let mut ids = enc("r\n");
        ids.push(w.sp.end_think);
        ids.extend(enc("\n\n"));
        ids.extend(&hello);
        ids.extend(&space);
        ids.extend(&party);
        ids.extend(&world_tok);
        let mut want = vec![
            reasoning("r"),
            content("Hello"),
            entry(hello[0]),
            content(" "),
            entry(space[0]),
            content("🎉"),
        ];
        want.extend(party.iter().map(|&i| entry(i)));
        let (outs, _) = drive(&setup(true, None, &[]), &ids, true, end());
        let mut full = want.clone();
        full.extend([content(" world"), entry(world_tok[0])]);
        assert_eq!(outs, full);
        // The stop string cuts " world": its token has no entry, and the text is "Hello 🎉 ".
        let (outs, sum) = drive(&setup(true, None, &["wor"]), &ids, true, end());
        want.push(content(" "));
        want.push(stopped("wor"));
        assert_eq!(outs, want);
        assert_eq!(sum.stopped.as_deref(), Some("wor"));
    }

    #[test]
    fn o12_entries_never_precede_their_text() {
        let text = "Hello 🎉 world, 终止 ok";
        for seed in 0..SEEDS {
            for stop in [None, Some("wor"), Some("止 o")] {
                let stops: Vec<&str> = stop.into_iter().collect();
                let ids = ids_of(text, seed, true);
                let (outs, _) = drive(&setup(false, None, &stops), &ids, true, end());
                let mut shown = String::new();
                let mut listed: Vec<u8> = Vec::new();
                for o in &outs {
                    match o {
                        Out::Content(c) => shown.push_str(c),
                        Out::Logprob(e) => {
                            listed.extend(&e.bytes);
                            assert!(
                                shown.len() >= listed.len(),
                                "seed {seed} {stop:?}: an entry came before its text"
                            );
                        }
                        _ => {}
                    }
                }
                match stop {
                    None => {
                        assert_eq!(shown, text);
                        assert_eq!(listed, text.as_bytes(), "seed {seed}");
                    }
                    Some(s) => {
                        assert_eq!(shown, text[..text.find(s).unwrap()], "seed {seed}");
                        // Whole tokens before the match, in order; the one the match cuts has none.
                        assert!(
                            shown.as_bytes().starts_with(&listed),
                            "seed {seed} {stop:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_partly_stripped_token_keeps_its_entry() {
        let w = world();
        // A token that starts with a newline and has more after it.
        let id = (0..w.vocab_len)
            .find(|&i| {
                let b = w.tk.token_bytes(i);
                b.len() > 1 && b[0] == b'\n' && b.iter().any(|&c| c != b'\n') && b.is_ascii()
            })
            .expect("a token of a newline and more");
        let b = w.tk.token_bytes(id);
        let first = b.iter().position(|&c| c != b'\n').unwrap();
        let rest = String::from_utf8(b[first..].to_vec()).unwrap();
        let mut ids = vec![w.sp.end_think, id];
        ids.extend(enc("x"));
        let (outs, _) = drive(&setup(true, None, &[]), &ids, true, end());
        assert_eq!(merged(&outs), vec![content(&format!("{rest}x"))]);
        let entries: Vec<&Out> = outs
            .iter()
            .filter(|o| matches!(o, Out::Logprob(_)))
            .collect();
        assert_eq!(entries[0], &entry(id));
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn mode_table_rows_not_in_the_happy_path() {
        let tools = tool_f();
        let call = "<tool_call>\n<function=f>\n</function>\n</tool_call>";
        for seed in 0..SEEDS {
            // Reasoning: a <think> token is dropped and reasoning goes on.
            let (outs, sum) = drive(
                &setup(true, None, &[]),
                &ids_of("a<think>b</think>c", seed, false),
                false,
                end(),
            );
            assert_eq!(
                merged(&outs),
                vec![reasoning("ab"), content("c")],
                "seed {seed}"
            );
            assert_eq!(sum.repairs.len(), 1);
            // Content: a stray </think> is dropped.
            let (outs, sum) = drive(
                &setup(false, None, &[]),
                &ids_of("x</think>y", seed, false),
                false,
                end(),
            );
            assert_eq!(merged(&outs), vec![content("xy")], "seed {seed}");
            assert_eq!(sum.repairs.len(), 1);
            // Call: </think> and <think> are dropped from the call's text.
            let text = concat!(
                "<tool_call>\n<function=f>\n<parameter=a>\nx</think>y<think>z\n</parameter>\n",
                "</function>\n</tool_call>"
            );
            let (outs, sum) = drive(
                &setup(false, tools.clone(), &[]),
                &ids_of(text, seed, false),
                false,
                end(),
            );
            assert_eq!(
                merged(&outs),
                vec![
                    call_start(0, "f"),
                    call_args(0, "{\"a\":\"xyz\"}"),
                    call_end(0)
                ],
                "seed {seed}"
            );
            assert_eq!(sum.repairs.len(), 2);
            // AfterCall: a second <tool_call> opens call 1; whitespace between is dropped.
            let text = format!("{call}\n\n{call}\n");
            let (outs, sum) = drive(
                &setup(false, tools.clone(), &[]),
                &ids_of(&text, seed, false),
                false,
                end(),
            );
            assert_eq!(
                merged(&outs),
                vec![
                    call_start(0, "f"),
                    call_args(0, "{}"),
                    call_end(0),
                    call_start(1, "f"),
                    call_args(1, "{}"),
                    call_end(1),
                ],
                "seed {seed}"
            );
            assert_eq!(sum.calls, 2);
            // AfterCall: <think>...</think> then text is reasoning then content.
            let text = format!("{call}\n <think>r</think>\n\nT");
            let (outs, _) = drive(
                &setup(false, tools.clone(), &[]),
                &ids_of(&text, seed, false),
                false,
                end(),
            );
            assert_eq!(
                merged(&outs),
                vec![
                    call_start(0, "f"),
                    call_args(0, "{}"),
                    call_end(0),
                    reasoning("r"),
                    content("T"),
                ],
                "seed {seed}"
            );
            // AfterCall: the first text that isn't whitespace is content.
            let text = format!("{call}\n  Done");
            let (outs, _) = drive(
                &setup(false, tools.clone(), &[]),
                &ids_of(&text, seed, false),
                false,
                end(),
            );
            assert_eq!(merged(&outs).last(), Some(&content("Done")), "seed {seed}");
        }
    }

    #[test]
    fn forced_close_ends_reasoning() {
        let w = world();
        let mut ids = enc("abc");
        let n = ids.len() as u32;
        ids.extend(&w.sp.think_close);
        ids.extend(enc("Done"));
        let (outs, sum) = drive(&setup(true, None, &[]), &ids, false, end());
        assert_eq!(merged(&outs), vec![reasoning("abc"), content("Done")]);
        // The forced "\n" is in reasoning; the tag and what follows are not.
        assert_eq!(sum.reasoning_tokens, n + 1);
    }

    #[test]
    fn length_mid_call_closes_it() {
        let text = "<tool_call>\n<function=f>\n<parameter=a>\nabc";
        for seed in 0..SEEDS {
            for fin in [Finish::Length, Finish::ContextFull] {
                let (outs, sum) = drive(
                    &setup(false, tool_f(), &[]),
                    &ids_of(text, seed, false),
                    false,
                    fin,
                );
                assert_eq!(
                    merged(&outs),
                    vec![
                        call_start(0, "f"),
                        call_args(0, "{\"a\":\"abc\"}"),
                        call_end(0)
                    ],
                    "seed {seed}"
                );
                assert_eq!(sum.calls, 1);
            }
        }
    }

    #[test]
    fn parallel_off_closes_the_call_at_the_stop_token() {
        let text = "<tool_call>\n<function=f>\n</function>\n";
        for seed in 0..SEEDS {
            let (outs, sum) = drive(
                &setup(false, tool_f(), &[]),
                &ids_of(text, seed, false),
                false,
                Finish::Stop(world().sp.end_tool_call),
            );
            assert_eq!(
                merged(&outs),
                vec![call_start(0, "f"), call_args(0, "{}"), call_end(0)],
                "seed {seed}"
            );
            assert_eq!(sum.calls, 1);
        }
    }

    #[test]
    fn a_malformed_call_is_content_and_holds_no_call() {
        let text = "<tool_call>\n<function=nope>\n</function>\n</tool_call>";
        for seed in 0..SEEDS {
            let (outs, sum) = drive(
                &setup(false, tool_f(), &[]),
                &ids_of(text, seed, false),
                false,
                end(),
            );
            assert_eq!(merged(&outs), vec![content(text)], "seed {seed}");
            assert_eq!(sum.calls, 0);
        }
    }

    #[test]
    fn a_stop_string_in_a_malformed_calls_content_matches() {
        let text = "<tool_call>junk STOP more</tool_call>";
        let want = [content("<tool_call>junk "), stopped("STOP")];
        // Tools declared or not, the same text gives the same content and the same stop.
        for tools in [tool_f(), None] {
            let sum = check(&setup(false, tools, &["STOP"]), text, &want);
            assert_eq!(sum.stopped.as_deref(), Some("STOP"));
            assert_eq!(sum.calls, 0);
        }
        // A stop string split between the tag's text and the rest, and one that ends the output half-written.
        for seed in 0..SEEDS {
            let s = setup(false, tool_f(), &["junk STOP"]);
            let (outs, _) = drive(&s, &ids_of(text, seed, false), false, end());
            assert_eq!(
                merged(&outs),
                vec![content("<tool_call>"), stopped("junk STOP")],
                "seed {seed}"
            );
            let (outs, sum) = drive(
                &setup(false, tool_f(), &["STOP"]),
                &ids_of("<tool_call>junk ST", seed, false),
                false,
                Finish::Length,
            );
            assert_eq!(
                merged(&outs),
                vec![content("<tool_call>junk ST")],
                "seed {seed}"
            );
            assert_eq!(sum.stopped, None);
        }
    }

    #[test]
    fn whitespace_after_a_malformed_call_is_kept() {
        let text = "<tool_call>junk</tool_call>\n\nMore";
        for tools in [tool_f(), None] {
            check(&setup(false, tools, &[]), text, &[content(text)]);
        }
        // After a valid call it is still dropped.
        check(
            &setup(false, tool_f(), &[]),
            "<tool_call>\n<function=f>\n</function>\n</tool_call>\n\nMore",
            &[
                call_start(0, "f"),
                call_args(0, "{}"),
                call_end(0),
                content("More"),
            ],
        );
    }

    #[test]
    fn a_stop_token_that_is_the_call_end_with_no_call_open_changes_nothing() {
        let fin = || Finish::Stop(world().sp.end_tool_call);
        let call = "<tool_call>\n<function=f>\n</function>\n</tool_call>\n";
        for seed in 0..SEEDS {
            let (outs, sum) = drive(
                &setup(false, tool_f(), &[]),
                &ids_of("abc", seed, false),
                false,
                fin(),
            );
            assert_eq!(merged(&outs), vec![content("abc")], "seed {seed}");
            assert_eq!(sum.calls, 0);
            let (outs, _) = drive(
                &setup(true, tool_f(), &[]),
                &ids_of("think\n", seed, false),
                false,
                fin(),
            );
            assert_eq!(merged(&outs), vec![reasoning("think\n")], "seed {seed}");
            // After a closed call, too.
            let (outs, _) = drive(
                &setup(false, tool_f(), &[]),
                &ids_of(call, seed, false),
                false,
                fin(),
            );
            assert_eq!(
                merged(&outs),
                vec![call_start(0, "f"), call_args(0, "{}"), call_end(0)],
                "seed {seed}"
            );
        }
    }

    #[test]
    fn a_held_stop_prefix_is_flushed_before_a_call() {
        let text = "ST<tool_call>\n<function=f>\n</function>\n</tool_call>";
        check(
            &setup(false, tool_f(), &["STOP"]),
            text,
            &[
                content("ST"),
                call_start(0, "f"),
                call_args(0, "{}"),
                call_end(0),
            ],
        );
    }

    #[test]
    fn a_split_character_at_the_end_is_printed_lossy() {
        // The last token ends inside a character: it is printed (lossy) at the end, entry included.
        let mut ids = enc("ok ");
        ids.extend(bytes_ids(&"🎉".as_bytes()[..2]));
        let (outs, _) = drive(&setup(false, None, &[]), &ids, true, end());
        let shown: String = outs
            .iter()
            .filter_map(|o| match o {
                Out::Content(c) => Some(c.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(shown, "ok \u{FFFD}");
        let entries = outs.iter().filter(|o| matches!(o, Out::Logprob(_))).count();
        assert_eq!(entries, ids.len());
    }
}
