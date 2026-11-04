//! Thinking control and the request field rules for both APIs, assembled into a `GenRequest`.
//! A field is taken from the body with `take_field`, so a type error names its path and never quotes the value.
use crate::anthropic::request::MessagesRequest;
use crate::backend::{GenRequest, Sampling, ThinkLimit};
use crate::error::{take_field, ApiError};
use crate::openai::request::ChatRequest;
use crate::prompt::Prompt;
use crate::tokens::Specials;
use serde::Deserialize;
use serde_json::{Map, Number, Value};

/// What a request that says nothing gets: the config's values, and the model's context.
#[derive(Clone, Debug, PartialEq)]
pub struct Defaults {
    pub thinking: bool,
    pub thinking_budget: Option<u32>,
    /// Whether a `/v1/messages` request that says nothing of thinking thinks by default (`thinking` decides the rest):
    /// `true` is llama.cpp's behaviour, `false` is Anthropic's.
    pub anthropic_thinking: bool,
    pub sampling: Sampling,
    pub model_id: String,
    pub ctx: u32,
}

/// Whether the template thinks, and for how many tokens at most.
#[derive(Clone, Debug, PartialEq)]
pub struct Thinking {
    pub on: bool,
    pub budget: Option<u32>,
}

/// `Unenforced` holds what the client asked for (`required`, `any`, `function`, `tool`): it renders as `auto`.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolChoice {
    Auto,
    None,
    Unenforced(String),
}

/// A request's fields as the rules make them, in the order the server uses them.
#[derive(Clone, Debug, PartialEq)]
pub struct Fields {
    pub thinking: Thinking,
    pub sampling: Sampling,
    pub seed: Option<u64>,
    pub max_tokens: Option<u32>,
    pub tool_choice: ToolChoice,
    pub parallel: bool,
    pub stop: Vec<String>,
    pub logprobs: Option<u8>,
    pub stream: bool,
    pub include_usage: bool,
    pub model: Option<String>,
    /// The fields that were accepted and ignored, and the unknown ones, by name: the first `IGNORED_SHOWN`, each cut
    /// to `IGNORED_NAME_CHARS` characters.
    pub ignored: Vec<String>,
    /// How many more there were past `IGNORED_SHOWN`, counted and not named.
    pub ignored_more: u32,
}

impl Defaults {
    /// Production's sampling, thinking on without a budget.
    pub fn new(model_id: impl Into<String>, ctx: u32) -> Defaults {
        Defaults {
            thinking: true,
            thinking_budget: None,
            anthropic_thinking: true,
            sampling: Sampling {
                temp: 0.6,
                top_k: 20,
                top_p: 0.95,
                min_p: 0.0,
                presence_penalty: 0.0,
            },
            model_id: model_id.into(),
            ctx,
        }
    }
}

/// A `/v1/messages` request that says nothing of thinking and asks for fewer tokens than this has its default reasoning
/// capped at half of `max_tokens`: a title or a quota probe would otherwise spend the whole allowance thinking.
const SMALL_MAX_TOKENS: u32 = 1024;
/// The budget of `low` and `minimal`, and of `medium`.
const LOW: u32 = 1024;
const MEDIUM: u32 = 4096;
/// The most stop strings a request may carry.
const MAX_STOP: usize = 16;
/// The longest stop string, in UTF-8 bytes (the matcher holds back that much text).
const MAX_STOP_BYTES: usize = 1024;
/// The most alternatives `top_logprobs` may ask for.
const MAX_TOP_LOGPROBS: i128 = 20;

type Body = Map<String, Value>;

/// What one field says about thinking.
enum Say {
    Off,
    /// On, with this budget (`None` for no limit).
    On(Option<u32>),
}

/// The fields' sayings: any off wins, else the first on, else the defaults.
#[derive(Default)]
struct Says {
    off: bool,
    on: Option<Option<u32>>,
}

impl Says {
    fn add(&mut self, s: Say) {
        match s {
            Say::Off => self.off = true,
            Say::On(b) => {
                self.on.get_or_insert(b);
            }
        }
    }

    fn resolve(self, d: &Defaults) -> Thinking {
        let off = Thinking {
            on: false,
            budget: None,
        };
        if self.off {
            return off;
        }
        match self.on {
            Some(budget) => Thinking { on: true, budget },
            None if d.thinking => Thinking {
                on: true,
                budget: d.thinking_budget,
            },
            None => off,
        }
    }
}

/// `low`, `medium` and `high` (and `minimal`, treated as `low`).
fn effort(s: &str) -> Option<Say> {
    match s {
        "minimal" | "low" => Some(Say::On(Some(LOW))),
        "medium" => Some(Say::On(Some(MEDIUM))),
        "high" => Some(Say::On(None)),
        _ => None,
    }
}

#[derive(Deserialize)]
#[serde(expecting = "a reasoning object")]
struct Reasoning {
    #[serde(default)]
    effort: Option<String>,
}

#[derive(Deserialize)]
#[serde(expecting = "a thinking object")]
struct ThinkingIn {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    budget_tokens: Option<Number>,
}

#[derive(Deserialize)]
#[serde(expecting = "an output_config object")]
struct OutputConfig {
    #[serde(default)]
    effort: Option<String>,
    /// Only whether it is there matters; its value is never read, so never echoed.
    #[serde(default)]
    format: Option<Value>,
}

#[derive(Deserialize)]
#[serde(expecting = "a stream_options object")]
struct StreamOptions {
    #[serde(default)]
    include_usage: Option<bool>,
}

#[derive(Deserialize)]
#[serde(expecting = "a response_format object")]
struct ResponseFormat {
    #[serde(rename = "type", default)]
    kind: Option<String>,
}

#[derive(Deserialize)]
#[serde(expecting = "a tool_choice object")]
struct ToolChoiceIn {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    disable_parallel_tool_use: Option<bool>,
}

/// How many characters of an ignored field's name the log keeps, so a request can't put a long text into the log
/// through a key.
pub const IGNORED_NAME_CHARS: usize = 40;

/// A request's field name as the logs may carry it: cut to `IGNORED_NAME_CHARS` characters.
pub fn cut_name(name: &str) -> String {
    name.chars().take(IGNORED_NAME_CHARS).collect()
}

/// How many ignored field names a request keeps. The list is held twice through the queue and the generation (in the
/// fields and in the request's log line) and is written to the log, so a body of a million unknown keys must not grow
/// it: the rest are counted.
pub const IGNORED_SHOWN: usize = 32;

/// The fields a request had that were accepted and ignored, and the unknown ones: the first `IGNORED_SHOWN` names and
/// the count of the rest.
#[derive(Default)]
struct Ignored {
    names: Vec<String>,
    more: u32,
}

/// Notes `name` as accepted and ignored: in the request's list (cut to `IGNORED_NAME_CHARS` characters, never its
/// value) while fewer than `IGNORED_SHOWN` are held, and in the count of the rest after that. It writes no log
/// line: the request's stderr line carries the count.
fn ignore(ig: &mut Ignored, name: &str) {
    if ig.names.len() < IGNORED_SHOWN {
        ig.names.push(cut_name(name));
    } else {
        ig.more = ig.more.saturating_add(1);
    }
}

/// A 400 about `name`, its message `name` and `what`.
fn invalid(name: &str, what: &str) -> ApiError {
    ApiError::invalid(name, format!("{name} {what}"))
}

/// A number field as a float; a 400 naming `key` when it is another type.
fn number(m: &mut Body, key: &str) -> Result<Option<f64>, ApiError> {
    Ok(take_field::<Number>(m, key)?.map(|n| n.as_f64().unwrap_or(f64::NAN)))
}

/// An integer field (a float with no fraction counts); a 400 naming `key` otherwise.
fn int(m: &mut Body, key: &str) -> Result<Option<i128>, ApiError> {
    take_field::<Number>(m, key)?
        .map(|n| whole(&n, key))
        .transpose()
}

/// A number that is a whole number, as an integer; a 400 naming `key` for a fraction or a value out of range.
fn whole(n: &Number, key: &str) -> Result<i128, ApiError> {
    match (n.as_i64(), n.as_u64(), n.as_f64()) {
        (Some(i), _, _) => Ok(i128::from(i)),
        (_, Some(u), _) => Ok(i128::from(u)),
        (_, _, Some(f)) if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e18 => Ok(f as i128),
        (_, _, Some(f)) if f.is_finite() && f.fract() == 0.0 => {
            Err(invalid(key, "is out of range"))
        }
        _ => Err(invalid(key, "must be an integer")),
    }
}

/// `max_tokens` or `max_completion_tokens`: at least 1; above `u32::MAX` it is `u32::MAX` (the room clamps it).
fn count(m: &mut Body, key: &str) -> Result<Option<u32>, ApiError> {
    match int(m, key)? {
        None => Ok(None),
        Some(n) if n < 1 => Err(invalid(key, "must be at least 1")),
        Some(n) => Ok(Some(u32::try_from(n).unwrap_or(u32::MAX))),
    }
}

/// The request's sampling over the defaults, each setting checked as `engine::Sampling::check` checks it
/// (`Sampling::check`, which the config's `[sampling]` table shares).
fn sampling(m: &mut Body, d: &Defaults) -> Result<Sampling, ApiError> {
    let mut s = d.sampling.clone();
    if let Some(v) = number(m, "temperature")? {
        s.temp = v as f32;
    }
    if let Some(v) = number(m, "top_p")? {
        s.top_p = v as f32;
    }
    if let Some(v) = int(m, "top_k")? {
        s.top_k = usize::try_from(v).map_err(|_| invalid("top_k", "must be 0 or more"))?;
    }
    if let Some(v) = number(m, "min_p")? {
        s.min_p = v as f32;
    }
    if let Some(v) = number(m, "presence_penalty")? {
        s.presence_penalty = v as f32;
    }
    s.check().map_err(|(key, why)| invalid(key, why))?;
    Ok(s)
}

/// `seed`: a negative one means "random", as llama.cpp's clients send it, so it is no seed.
fn seed(m: &mut Body) -> Result<Option<u64>, ApiError> {
    Ok(int(m, "seed")?.and_then(|n| u64::try_from(n).ok()))
}

/// The settings that change style, not the contract: removed, and listed in the request's `ignored` when not at their
/// defaults.
fn style(m: &mut Body, ig: &mut Ignored) {
    const DEFAULTS: [(&str, f64); 6] = [
        ("frequency_penalty", 0.0),
        ("repetition_penalty", 1.0),
        ("typical_p", 1.0),
        ("mirostat", 0.0),
        ("mirostat_tau", 5.0),
        ("mirostat_eta", 0.1),
    ];
    for (key, default) in DEFAULTS {
        let at_default = match m.shift_remove(key) {
            None | Some(Value::Null) => true,
            Some(Value::Number(n)) => n.as_f64() == Some(default),
            Some(_) => false,
        };
        if !at_default {
            ignore(ig, key);
        }
    }
    let at_default = match m.shift_remove("logit_bias") {
        None | Some(Value::Null) => true,
        Some(Value::Object(o)) => o.is_empty(),
        Some(_) => false,
    };
    if !at_default {
        ignore(ig, "logit_bias");
    }
}

/// Fields accepted and ignored without a word.
fn silent(m: &mut Body, keys: &[&str]) {
    for k in keys {
        m.shift_remove(*k);
    }
}

/// What is left of the body is unknown: listed in `ignored` by name (the first `IGNORED_SHOWN`, then counted).
fn unknown(m: &Body, ig: &mut Ignored) {
    for k in m.keys() {
        ignore(ig, k);
    }
}

/// The stop strings: at most 16, each at most 1,024 bytes (400 past either); an empty one is ignored with a log line.
/// `one` also allows a lone string (OpenAI's `stop`).
fn stop_list(m: &mut Body, key: &str, one: bool) -> Result<Vec<String>, ApiError> {
    let mut scalar = false;
    let items = match take_field::<Value>(m, key)? {
        None => return Ok(Vec::new()),
        Some(Value::String(s)) if one => {
            scalar = true;
            vec![Value::String(s)]
        }
        Some(Value::Array(a)) => a,
        Some(_) => {
            let what = if one {
                "has the wrong type, expected a string or an array of strings"
            } else {
                "has the wrong type, expected an array of strings"
            };
            return Err(invalid(key, what));
        }
    };
    if items.len() > MAX_STOP {
        return Err(invalid(key, "holds at most 16 strings"));
    }
    let mut out = Vec::new();
    for (i, v) in items.into_iter().enumerate() {
        match v {
            Value::String(s) if s.is_empty() => tracing::info!("{key}[{i}] is empty; ignored"),
            Value::String(s) if s.len() > MAX_STOP_BYTES => {
                // A lone string has no index to name.
                let at = if scalar {
                    key.to_string()
                } else {
                    format!("{key}[{i}]")
                };
                return Err(invalid(&at, "holds at most 1,024 bytes"));
            }
            Value::String(s) => out.push(s),
            _ => {
                let at = format!("{key}[{i}]");
                return Err(invalid(&at, "has the wrong type, expected a string"));
            }
        }
    }
    Ok(out)
}

/// `tool_choice`, and whether parallel calls are allowed (Anthropic's `disable_parallel_tool_use`). `required`,
/// `any` and named tools are accepted and rendered as `auto`, and listed in `ignored` as `tool_choice`; an unknown
/// string (OpenAI) or type (Anthropic) is a 400, as llama.cpp answers.
fn tool_choice(
    m: &mut Body,
    anthropic: bool,
    ig: &mut Ignored,
) -> Result<(ToolChoice, bool), ApiError> {
    // Recorded in the request's `ignored` list (and so in its stderr line's count), not in a line of its own.
    let unenforced = |what: &str| ToolChoice::Unenforced(what.to_string());
    let wrong = || {
        let what = if anthropic {
            "an object"
        } else {
            "a string or an object"
        };
        invalid(
            "tool_choice",
            &format!("has the wrong type, expected {what}"),
        )
    };
    let Some(v) = take_field::<Value>(m, "tool_choice")? else {
        return Ok((ToolChoice::Auto, true));
    };
    match v {
        Value::String(s) if !anthropic => Ok((
            match s.as_str() {
                "none" => ToolChoice::None,
                "auto" => ToolChoice::Auto,
                "required" => unenforced("required"),
                _ => {
                    return Err(invalid(
                        "tool_choice",
                        "must be one of none, auto, required or an object",
                    ));
                }
            },
            true,
        )),
        Value::Object(_) => {
            let mut one = Body::new();
            one.insert("tool_choice".into(), v);
            let t = take_field::<ToolChoiceIn>(&mut one, "tool_choice")?.ok_or_else(wrong)?;
            let choice = match t.kind.as_deref() {
                Some("none") if anthropic => ToolChoice::None,
                Some("auto") if anthropic => ToolChoice::Auto,
                Some("any") if anthropic => unenforced("any"),
                Some("tool") if anthropic => unenforced("tool"),
                Some("function") if !anthropic => unenforced("function"),
                _ if anthropic => {
                    return Err(invalid(
                        "tool_choice.type",
                        "must be one of auto, any, tool, none",
                    ));
                }
                _ => {
                    ignore(ig, "tool_choice");
                    ToolChoice::Auto
                }
            };
            Ok((choice, !t.disable_parallel_tool_use.unwrap_or(false)))
        }
        _ => Err(wrong()),
    }
}

/// A thinking budget: a whole number of at least 1, else a 400 naming `at`.
fn budget(n: &Number, at: &str) -> Result<u32, ApiError> {
    match whole(n, at)? {
        v if v >= 1 => Ok(u32::try_from(v).unwrap_or(u32::MAX)),
        _ => Err(invalid(at, "must be at least 1")),
    }
}

/// What a `/v1/messages` request that says nothing of thinking gets, given what the defaults resolved: off when
/// `anthropic_thinking` is false (Anthropic's meaning of an absent field), else the default, with its budget cut to
/// half of a small `max_tokens` (or the configured budget when that is smaller).
fn unstated_thinking(default: Thinking, d: &Defaults, max_tokens: Option<u32>) -> Thinking {
    if !default.on {
        return default;
    }
    if !d.anthropic_thinking {
        return Thinking {
            on: false,
            budget: None,
        };
    }
    match max_tokens {
        Some(n) if n < SMALL_MAX_TOKENS => {
            let half = (n / 2).max(1);
            Thinking {
                on: true,
                budget: Some(default.budget.map_or(half, |b| b.min(half))),
            }
        }
        _ => default,
    }
}

impl ChatRequest {
    /// The request's fields under the API's rules. The prompt's own fields (`messages`, `tools`,
    /// `chat_template_kwargs`) were taken at parse; the rest are taken here by value (`extra` is left empty, so the
    /// unknown fields are not copied: a body full of them is as large again as it was).
    pub fn fields(&mut self, d: &Defaults) -> Result<Fields, ApiError> {
        let mut m = std::mem::take(&mut self.extra);
        let mut ig = Ignored::default();
        let mut says = Says::default();
        // Effort first, so its budget wins over a bare `enable_thinking: true`; `reasoning_effort` before
        // `reasoning.effort`, so it wins when both are on (`Says` keeps the first).
        let efforts = [
            (
                "reasoning_effort",
                take_field::<String>(&mut m, "reasoning_effort")?,
            ),
            (
                "reasoning.effort",
                take_field::<Reasoning>(&mut m, "reasoning")?.and_then(|r| r.effort),
            ),
        ];
        for (key, e) in efforts {
            match e.as_deref() {
                None => {}
                Some("none") => says.add(Say::Off),
                Some(e) => match effort(e) {
                    Some(s) => says.add(s),
                    None => {
                        return Err(invalid(
                            key,
                            "must be one of none, minimal, low, medium, high",
                        ));
                    }
                },
            }
        }
        if let Some(kwargs) = &self.chat_template_kwargs {
            match kwargs.get("enable_thinking") {
                None => {}
                Some(Value::Bool(false)) => says.add(Say::Off),
                Some(Value::Bool(true)) => says.add(Say::On(d.thinking_budget)),
                Some(_) => {
                    return Err(invalid(
                        "chat_template_kwargs.enable_thinking",
                        "must be a boolean",
                    ));
                }
            }
        }
        let thinking = says.resolve(d);
        let sampling = sampling(&mut m, d)?;
        let seed = seed(&mut m)?;
        let completion = count(&mut m, "max_completion_tokens")?;
        let max_tokens = completion.or(count(&mut m, "max_tokens")?);
        if int(&mut m, "n")?.is_some_and(|n| n != 1) {
            return Err(invalid("n", "must be 1: several choices are not supported"));
        }
        if let Some(f) = take_field::<ResponseFormat>(&mut m, "response_format")? {
            match f.kind.as_deref() {
                Some("json_object") => {
                    return Err(invalid(
                        "response_format",
                        "json_object is not supported: there are no grammars",
                    ));
                }
                Some("json_schema") => {
                    return Err(invalid(
                        "response_format",
                        "json_schema is not supported: there are no grammars",
                    ));
                }
                _ => {}
            }
        }
        let top = int(&mut m, "top_logprobs")?;
        if top.is_some_and(|t| !(0..=MAX_TOP_LOGPROBS).contains(&t)) {
            return Err(invalid("top_logprobs", "must be from 0 to 20"));
        }
        let logprobs = take_field::<bool>(&mut m, "logprobs")?.unwrap_or(false);
        if top.is_some() && !logprobs {
            return Err(invalid("top_logprobs", "requires logprobs to be true"));
        }
        let logprobs = logprobs.then(|| top.unwrap_or(0) as u8);
        let (tool_choice, mut parallel) = tool_choice(&mut m, false, &mut ig)?;
        if take_field::<bool>(&mut m, "parallel_tool_calls")? == Some(false) {
            parallel = false;
        }
        let stop = stop_list(&mut m, "stop", true)?;
        let stream = take_field::<bool>(&mut m, "stream")?.unwrap_or(false);
        let include_usage = take_field::<StreamOptions>(&mut m, "stream_options")?
            .and_then(|o| o.include_usage)
            .unwrap_or(false);
        let model = take_field::<String>(&mut m, "model")?;
        style(&mut m, &mut ig);
        silent(&mut m, &["user", "metadata", "store", "service_tier"]);
        unknown(&m, &mut ig);
        Ok(Fields {
            thinking,
            sampling,
            seed,
            max_tokens,
            tool_choice,
            parallel,
            stop,
            logprobs,
            stream,
            include_usage,
            model,
            ignored: ig.names,
            ignored_more: ig.more,
        })
    }
}

impl MessagesRequest {
    /// As `ChatRequest::fields`, for `/v1/messages`. `max_tokens` is required, but not by `count_tokens`.
    pub fn fields(&mut self, d: &Defaults, count_only: bool) -> Result<Fields, ApiError> {
        let mut m = std::mem::take(&mut self.extra);
        let mut ig = Ignored::default();
        let mut says = Says::default();
        // `adaptive` (Claude Code's) is thinking on, its budget from `output_config.effort` as when `thinking` is
        // absent; `enabled` and `disabled` are the budget's and the switch's own, and an unknown type is ignored.
        let thinking_in = take_field::<ThinkingIn>(&mut m, "thinking")?;
        let mut adaptive = false;
        if let Some(t) = &thinking_in {
            match t.kind.as_deref() {
                Some("enabled") => {
                    let b = match &t.budget_tokens {
                        Some(n) => Some(budget(n, "thinking.budget_tokens")?),
                        None => d.thinking_budget,
                    };
                    says.add(Say::On(b));
                }
                Some("disabled") => says.add(Say::Off),
                Some("adaptive") => {
                    adaptive = true;
                    if t.budget_tokens.is_some() {
                        // The effort decides the budget; one beside `adaptive` is not used.
                        ignore(&mut ig, "thinking.budget_tokens");
                    }
                }
                _ => ignore(&mut ig, "thinking.type"),
            }
        }
        let output = take_field::<OutputConfig>(&mut m, "output_config")?;
        // Structured outputs need a grammar, and there are none: as OpenAI's
        // `response_format`.
        if output.as_ref().is_some_and(|o| o.format.is_some()) {
            return Err(ApiError::invalid(
                "output_config.format",
                "structured outputs (output_config.format) are not supported",
            ));
        }
        if thinking_in.is_none() || adaptive {
            match output.as_ref().and_then(|o| o.effort.as_deref()) {
                None => {}
                Some("max") => {
                    ignore(&mut ig, "output_config.effort");
                    says.add(Say::On(None));
                }
                Some(e) => match effort(e) {
                    Some(s) => says.add(s),
                    None => ignore(&mut ig, "output_config.effort"),
                },
            }
            if adaptive {
                says.add(Say::On(d.thinking_budget));
            }
        } else if output.is_some_and(|o| o.effort.is_some()) {
            // Beside `enabled`, `disabled` or an unknown type the effort counts for nothing.
            ignore(&mut ig, "output_config.effort");
        }
        // Whether the request says anything of thinking: a known `thinking` type or a known `effort`. An unknown
        // type or effort is ignored and counts as saying nothing.
        let stated = says.off || says.on.is_some();
        let thinking = says.resolve(d);
        let sampling = sampling(&mut m, d)?;
        let seed = seed(&mut m)?;
        let max_tokens = count(&mut m, "max_tokens")?;
        if max_tokens.is_none() && !count_only {
            return Err(invalid("max_tokens", "is required"));
        }
        let thinking = if stated {
            thinking
        } else {
            unstated_thinking(thinking, d, max_tokens)
        };
        let (tool_choice, parallel) = tool_choice(&mut m, true, &mut ig)?;
        let stop = stop_list(&mut m, "stop_sequences", false)?;
        let stream = take_field::<bool>(&mut m, "stream")?.unwrap_or(false);
        let model = take_field::<String>(&mut m, "model")?;
        style(&mut m, &mut ig);
        silent(&mut m, &["metadata", "service_tier", "user"]);
        unknown(&m, &mut ig);
        Ok(Fields {
            thinking,
            sampling,
            seed,
            max_tokens,
            tool_choice,
            parallel,
            stop,
            logprobs: None,
            stream,
            include_usage: false,
            model,
            ignored: ig.names,
            ignored_more: ig.more,
        })
    }
}

/// The engine's request: the prompt, the sampling, the stop tokens and the reasoning limit. `max_tokens` is the
/// request's, or the room left, whichever is less; `</tool_call>` stops a request that rendered tools and
/// allows one call at most.
pub fn assemble(
    f: &Fields,
    p: &Prompt,
    sp: &Specials,
    tools_rendered: bool,
    d: &Defaults,
    seed: u64,
) -> GenRequest {
    let room = d.ctx.saturating_sub(p.tokens.len() as u32);
    let mut stop_tokens = sp.eog.clone();
    if !f.parallel && tools_rendered && !stop_tokens.contains(&sp.end_tool_call) {
        stop_tokens.push(sp.end_tool_call);
    }
    GenRequest {
        prompt: p.tokens.clone(),
        cut: p.cut.clone(),
        sampling: f.sampling.clone(),
        seed: f.seed.unwrap_or(seed),
        max_tokens: f.max_tokens.map_or(room, |n| n.min(room)),
        stop_tokens,
        think: f.thinking.on.then(|| ThinkLimit {
            open: true,
            budget: f.thinking.budget,
            ends: [sp.end_think, sp.tool_call],
            close: sp.think_close.clone(),
        }),
        logprobs: f.logprobs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::request::parse as anthropic_parse;
    use crate::error::ErrorType;
    use crate::openai::request::parse as openai_parse;
    use serde_json::{json, Value};

    fn defaults() -> Defaults {
        Defaults::new("artemis", 1000)
    }

    fn oa(extra: Value) -> Result<Fields, ApiError> {
        let mut body = json!({"messages": [{"role": "user", "content": "q"}]});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        openai_parse(&serde_json::to_vec(&body).unwrap())?.fields(&defaults())
    }

    fn an(extra: Value) -> Result<Fields, ApiError> {
        let mut body = json!({"messages": [{"role": "user", "content": "q"}], "max_tokens": 100});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        anthropic_parse(&serde_json::to_vec(&body).unwrap())?.fields(&defaults(), false)
    }

    fn think(on: bool, budget: Option<u32>) -> Thinking {
        Thinking { on, budget }
    }

    #[test]
    fn unknown_keys_past_32_are_counted_not_named() {
        assert_eq!(IGNORED_SHOWN, 32);
        let keys: serde_json::Map<String, Value> = (0..40)
            .map(|i| (format!("unknown{i:02}"), json!(i)))
            .collect();
        for f in [
            oa(Value::Object(keys.clone())).unwrap(),
            an(Value::Object(keys.clone())).unwrap(),
        ] {
            assert_eq!(f.ignored.len(), 32);
            assert_eq!(f.ignored_more, 8);
            assert_eq!(f.ignored[0], "unknown00");
            assert_eq!(f.ignored[31], "unknown31");
        }
        // The names that count against the limit include the style fields listed ahead of the unknown ones.
        let mut keys = keys;
        keys.insert("frequency_penalty".into(), json!(0.5));
        let f = oa(Value::Object(keys)).unwrap();
        assert_eq!((f.ignored.len(), f.ignored_more), (32, 9));
        assert_eq!(f.ignored[0], "frequency_penalty");
        // A request under the limit has no count.
        let f = oa(json!({"zzz": 1})).unwrap();
        assert_eq!((f.ignored.len(), f.ignored_more), (1, 0));
    }

    #[test]
    fn defaults_are_the_production_values() {
        let f = oa(json!({})).unwrap();
        assert_eq!(f.thinking, think(true, None));
        assert_eq!(
            f.sampling,
            Sampling {
                temp: 0.6,
                top_k: 20,
                top_p: 0.95,
                min_p: 0.0,
                presence_penalty: 0.0
            }
        );
        assert_eq!(f.tool_choice, ToolChoice::Auto);
        assert!(f.parallel && !f.stream && !f.include_usage);
        assert_eq!(
            (f.seed, f.max_tokens, f.logprobs, f.model),
            (None, None, None, None)
        );
        assert!(f.stop.is_empty() && f.ignored.is_empty());
    }

    #[test]
    fn o2_reasoning_effort_none_turns_thinking_off() {
        assert_eq!(
            oa(json!({"reasoning_effort": "none"})).unwrap().thinking,
            think(false, None)
        );
        assert_eq!(
            oa(json!({"reasoning": {"effort": "none"}}))
                .unwrap()
                .thinking,
            think(false, None)
        );
        assert_eq!(
            oa(json!({"chat_template_kwargs": {"enable_thinking": false}}))
                .unwrap()
                .thinking,
            think(false, None)
        );
    }

    #[test]
    fn o3_effort_budgets_are_1024_4096_and_none() {
        for (effort, want) in [
            ("minimal", Some(1024)),
            ("low", Some(1024)),
            ("medium", Some(4096)),
            ("high", None),
        ] {
            for body in [
                json!({"reasoning_effort": effort}),
                json!({"reasoning": {"effort": effort}}),
            ] {
                assert_eq!(oa(body).unwrap().thinking, think(true, want), "{effort}");
            }
        }
        // An effort's budget wins over the config's, and a bare enable_thinking true takes the config's.
        let mut d = defaults();
        d.thinking_budget = Some(500);
        let mut r = openai_parse(
            br#"{"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "high"}"#,
        )
        .unwrap();
        assert_eq!(r.fields(&d).unwrap().thinking, think(true, None));
        let mut r = openai_parse(
            br#"{"messages": [{"role": "user", "content": "q"}], "chat_template_kwargs": {"enable_thinking": true}}"#,
        )
        .unwrap();
        assert_eq!(r.fields(&d).unwrap().thinking, think(true, Some(500)));
    }

    #[test]
    fn any_off_wins() {
        let f = oa(
            json!({"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": false}}),
        )
        .unwrap();
        assert_eq!(f.thinking, think(false, None));
        let f = oa(json!({"reasoning_effort": "none", "reasoning": {"effort": "high"}})).unwrap();
        assert_eq!(f.thinking, think(false, None));
        let f = an(json!({"thinking": {"type": "disabled"}, "output_config": {"effort": "high"}}))
            .unwrap();
        assert_eq!(f.thinking, think(false, None));
    }

    #[test]
    fn two_openai_efforts_the_first_field_wins() {
        let f = oa(json!({"reasoning_effort": "low", "reasoning": {"effort": "high"}})).unwrap();
        assert_eq!(f.thinking, think(true, Some(1024)));
        let f = oa(json!({"reasoning_effort": "high", "reasoning": {"effort": "low"}})).unwrap();
        assert_eq!(f.thinking, think(true, None));
    }

    #[test]
    fn an_effort_beside_thinking_is_listed_as_ignored() {
        let f = an(
            json!({"thinking": {"type": "enabled", "budget_tokens": 2000},
            "output_config": {"effort": "low"}}),
        )
        .unwrap();
        assert_eq!(f.thinking, think(true, Some(2000)));
        assert_eq!(f.ignored, ["output_config.effort"]);
        let f = an(json!({"thinking": {"type": "disabled"}, "output_config": {}})).unwrap();
        assert!(f.ignored.is_empty());
        // Beside `adaptive` it is used, not ignored.
        let f = an(json!({"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}))
            .unwrap();
        assert_eq!(f.thinking, think(true, Some(1024)));
        assert!(f.ignored.is_empty());
    }

    #[test]
    fn whole_numbers_may_be_floats_and_a_huge_one_is_out_of_range() {
        let f = an(json!({"thinking": {"type": "enabled", "budget_tokens": 2000.0}})).unwrap();
        assert_eq!(f.thinking, think(true, Some(2000)));
        let e = an(json!({"thinking": {"type": "enabled", "budget_tokens": 1.5}})).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("thinking.budget_tokens"));
        assert!(e.message.contains("integer"), "{e:?}");
        let e = oa(json!({"max_tokens": 1e30})).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("max_tokens"));
        assert!(e.message.contains("out of range"), "{e:?}");
        let e = oa(json!({"max_tokens": 2.5})).unwrap_err();
        assert!(e.message.contains("integer"), "{e:?}");
        assert_eq!(oa(json!({"seed": 7.0})).unwrap().seed, Some(7));
    }

    #[test]
    fn an_unknown_tool_choice_is_400() {
        let e = oa(json!({"tool_choice": "sometimes"})).unwrap_err();
        assert_eq!((e.status(), e.param.as_deref()), (400, Some("tool_choice")));
        assert!(!e.message.contains("sometimes"));
        for t in ["sometimes", "function"] {
            let e = an(json!({"tool_choice": {"type": t}})).unwrap_err();
            assert_eq!(e.param.as_deref(), Some("tool_choice.type"), "{t}");
            assert!(!e.message.contains("sometimes"), "{e:?}");
        }
        assert_eq!(an(json!({"tool_choice": {}})).unwrap_err().status(), 400);
        // The OpenAI forms that name a tool or require one stay accepted (and render as `auto`).
        assert!(oa(json!({"tool_choice": "required"})).is_ok());
        assert!(
            oa(json!({"tool_choice": {"type": "function", "function": {"name": "f"}}})).is_ok()
        );
    }

    #[test]
    fn an_unknown_effort_is_400() {
        let e = oa(json!({"reasoning_effort": "ultra"})).unwrap_err();
        assert_eq!(
            (e.status(), e.param.as_deref()),
            (400, Some("reasoning_effort"))
        );
        assert!(!e.message.contains("ultra"));
        let e = oa(json!({"reasoning": {"effort": "ultra"}})).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("reasoning.effort"));
        let e = oa(json!({"reasoning_effort": 5})).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("reasoning_effort"));
        let e = oa(json!({"chat_template_kwargs": {"enable_thinking": "no"}})).unwrap_err();
        assert_eq!(
            e.param.as_deref(),
            Some("chat_template_kwargs.enable_thinking")
        );
    }

    #[test]
    fn an_unknown_anthropic_effort_is_logged_never_400() {
        let f = an(json!({"output_config": {"effort": "max"}})).unwrap();
        assert_eq!(f.thinking, think(true, None));
        assert!(
            f.ignored.contains(&"output_config.effort".to_string()),
            "{:?}",
            f.ignored
        );
        // An unknown effort says nothing, so the request is an unstated one: max_tokens is 100 here, so the default's
        // budget is cut to half of it.
        let f = an(json!({"output_config": {"effort": "ultra"}})).unwrap();
        assert_eq!(f.thinking, think(true, Some(50)));
        assert!(f.ignored.contains(&"output_config.effort".to_string()));
        // The config's defaults, not a fixed answer.
        let mut d = defaults();
        d.thinking = false;
        let mut r = anthropic_parse(
            br#"{"messages": [{"role": "user", "content": "q"}], "max_tokens": 9,
                "output_config": {"effort": "ultra"}}"#,
        )
        .unwrap();
        assert_eq!(r.fields(&d, false).unwrap().thinking, think(false, None));
        let mut r = anthropic_parse(
            br#"{"messages": [{"role": "user", "content": "q"}], "max_tokens": 9,
                "output_config": {"effort": "max"}}"#,
        )
        .unwrap();
        assert_eq!(r.fields(&d, false).unwrap().thinking, think(true, None));
    }

    #[test]
    fn m4_anthropic_thinking_maps_per_6_6() {
        let f = an(json!({"thinking": {"type": "enabled", "budget_tokens": 2000}})).unwrap();
        assert_eq!(f.thinking, think(true, Some(2000)));
        let f = an(json!({"thinking": {"type": "disabled"}})).unwrap();
        assert_eq!(f.thinking, think(false, None));
        for (effort, want) in [("low", Some(1024)), ("medium", Some(4096)), ("high", None)] {
            let f = an(json!({"output_config": {"effort": effort}})).unwrap();
            assert_eq!(f.thinking, think(true, want), "{effort}");
        }
        // max_tokens is 100 here: a default-thinking request that small has its budget capped at half of it.
        assert_eq!(an(json!({})).unwrap().thinking, think(true, Some(50)));
        // `adaptive` is thinking on with the budget `output_config.effort` gives, as when `thinking` is absent; its
        // other keys (`display`) are ignored; no effort takes the config's.
        for (effort, want) in [("low", Some(1024)), ("medium", Some(4096)), ("high", None)] {
            let f = an(
                json!({"thinking": {"type": "adaptive", "display": "omitted"},
                "output_config": {"effort": effort}}),
            )
            .unwrap();
            assert_eq!(f.thinking, think(true, want), "{effort}");
            assert!(f.ignored.is_empty(), "{effort}: {:?}", f.ignored);
        }
        let f = an(json!({"thinking": {"type": "adaptive"}})).unwrap();
        assert_eq!(f.thinking, think(true, None));
        assert!(f.ignored.is_empty(), "{:?}", f.ignored);
        // A budget beside `adaptive` counts for nothing (the effort decides): listed, not dropped silently.
        let f = an(
            json!({"thinking": {"type": "adaptive", "budget_tokens": 3000},
            "output_config": {"effort": "low"}}),
        )
        .unwrap();
        assert_eq!(f.thinking, think(true, Some(1024)));
        assert_eq!(f.ignored, ["thinking.budget_tokens"]);
        let mut d = defaults();
        d.thinking_budget = Some(500);
        d.thinking = false;
        let body = |extra: &str| {
            format!(
                r#"{{"messages": [{{"role": "user", "content": "q"}}], "max_tokens": 9,
                    "thinking": {{"type": "adaptive"}}{extra}}}"#
            )
        };
        let mut r = anthropic_parse(body("").as_bytes()).unwrap();
        assert_eq!(
            r.fields(&d, false).unwrap().thinking,
            think(true, Some(500))
        );
        // `max` is `high` with a log; another unknown effort takes the config's, thinking on, with a log.
        let mut r =
            anthropic_parse(body(r#", "output_config": {"effort": "max"}"#).as_bytes()).unwrap();
        let f = r.fields(&d, false).unwrap();
        assert_eq!(f.thinking, think(true, None));
        assert_eq!(f.ignored, ["output_config.effort"]);
        let mut r =
            anthropic_parse(body(r#", "output_config": {"effort": "ultra"}"#).as_bytes()).unwrap();
        let f = r.fields(&d, false).unwrap();
        assert_eq!(f.thinking, think(true, Some(500)));
        assert_eq!(f.ignored, ["output_config.effort"]);
        // An unknown type is ignored, and its effort with it: nothing is stated, so max_tokens 100 gets the half cap.
        let f = an(json!({"thinking": {"type": "sideways"}, "output_config": {"effort": "low"}}))
            .unwrap();
        assert_eq!(f.thinking, think(true, Some(50)));
        assert_eq!(f.ignored, ["thinking.type", "output_config.effort"]);
        // An enabled thinking without a budget takes the config's; a budget below 1 is 400.
        assert_eq!(
            an(json!({"thinking": {"type": "enabled"}}))
                .unwrap()
                .thinking,
            think(true, None)
        );
        let e = an(json!({"thinking": {"type": "enabled", "budget_tokens": 0}})).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("thinking.budget_tokens"));
        let e = an(json!({"thinking": "on"})).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("thinking"));
    }

    #[test]
    fn o4_rejections_and_ignores() {
        for (body, param) in [
            (json!({"n": 2}), "n"),
            (
                json!({"response_format": {"type": "json_schema", "json_schema": {}}}),
                "response_format",
            ),
            (
                json!({"response_format": {"type": "json_object"}}),
                "response_format",
            ),
            (
                json!({"logprobs": true, "top_logprobs": 21}),
                "top_logprobs",
            ),
            (json!({"max_tokens": 0}), "max_tokens"),
            (json!({"max_completion_tokens": 0}), "max_completion_tokens"),
            (
                json!({"stop": (0..17).map(|i| format!("s{i}")).collect::<Vec<_>>()}),
                "stop",
            ),
        ] {
            let e = oa(body.clone()).unwrap_err();
            assert_eq!(
                (e.status(), e.param.as_deref()),
                (400, Some(param)),
                "{body}"
            );
        }
        assert!(oa(json!({"n": 1, "response_format": {"type": "text"}})).is_ok());
        let f = oa(json!({"frequency_penalty": 0.5, "repetition_penalty": 1.0, "logit_bias": {"5": 1},
            "mirostat": 0, "user": "u", "store": true, "metadata": {}, "service_tier": "auto", "zzz": 1}))
        .unwrap();
        assert_eq!(f.ignored, ["frequency_penalty", "logit_bias", "zzz"]);
        // At their defaults they are not listed.
        let f = oa(json!({"frequency_penalty": 0, "typical_p": 1, "mirostat_tau": 5.0, "mirostat_eta": 0.1,
            "logit_bias": {}, "parallel_tool_calls": true, "stream_options": {"include_usage": true}}))
        .unwrap();
        assert!(f.ignored.is_empty(), "{:?}", f.ignored);
        assert!(f.include_usage);
        let f = oa(json!({"stop": ["a", "", "b"]})).unwrap();
        assert_eq!(f.stop, ["a", "b"]);
        let f = oa(json!({"stop": "x", "stream": true, "model": "m"})).unwrap();
        assert_eq!(
            (f.stop, f.stream, f.model),
            (vec!["x".to_string()], true, Some("m".into()))
        );
        let f = oa(json!({"stop": (0..16).map(|i| format!("s{i}")).collect::<Vec<_>>()})).unwrap();
        assert_eq!(f.stop.len(), 16);
        let f = oa(json!({"logprobs": true})).unwrap();
        assert_eq!(f.logprobs, Some(0));
        let f = oa(json!({"logprobs": true, "top_logprobs": 20})).unwrap();
        assert_eq!(f.logprobs, Some(20));
        let f = oa(json!({"logprobs": false})).unwrap();
        assert_eq!(f.logprobs, None);
        // top_logprobs without logprobs: true is 400, as llama.cpp answers.
        for body in [
            json!({"top_logprobs": 5}),
            json!({"logprobs": false, "top_logprobs": 5}),
        ] {
            let e = oa(body.clone()).unwrap_err();
            assert_eq!(
                (e.status(), e.param.as_deref()),
                (400, Some("top_logprobs")),
                "{body}"
            );
        }
        for n in [0, -1] {
            let e = oa(json!({ "n": n })).unwrap_err();
            assert_eq!(e.param.as_deref(), Some("n"));
        }
        assert!(oa(json!({"n": 1.0})).is_ok());
    }

    #[test]
    fn o9_tool_choice_and_parallel() {
        let body = |extra: Value| {
            let mut b = json!({"messages": [{"role": "user", "content": "q"}],
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]});
            b.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            openai_parse(&serde_json::to_vec(&b).unwrap()).unwrap()
        };
        let mut none = body(json!({"tool_choice": "none"}));
        let f = none.fields(&defaults()).unwrap();
        assert_eq!(f.tool_choice, ToolChoice::None);
        assert!(none.conversation(&f).unwrap().tools.is_empty());
        let mut auto = body(json!({"tool_choice": "auto"}));
        let f = auto.fields(&defaults()).unwrap();
        assert_eq!(f.tool_choice, ToolChoice::Auto);
        assert_eq!(auto.conversation(&f).unwrap().tools.len(), 1);
        let mut req = body(json!({"tool_choice": "required"}));
        let f = req.fields(&defaults()).unwrap();
        assert_eq!(f.tool_choice, ToolChoice::Unenforced("required".into()));
        assert_eq!(req.conversation(&f).unwrap().tools.len(), 1);
        let f =
            oa(json!({"tool_choice": {"type": "function", "function": {"name": "f"}}})).unwrap();
        assert_eq!(f.tool_choice, ToolChoice::Unenforced("function".into()));
        assert!(f.parallel);
        let f = oa(json!({"parallel_tool_calls": false})).unwrap();
        assert!(!f.parallel);
        let f =
            an(json!({"tool_choice": {"type": "any", "disable_parallel_tool_use": true}})).unwrap();
        assert_eq!(f.tool_choice, ToolChoice::Unenforced("any".into()));
        assert!(!f.parallel);
        let f = an(json!({"tool_choice": {"type": "tool", "name": "f"}})).unwrap();
        assert_eq!(f.tool_choice, ToolChoice::Unenforced("tool".into()));
        assert_eq!(
            an(json!({"tool_choice": {"type": "none"}}))
                .unwrap()
                .tool_choice,
            ToolChoice::None
        );
        assert_eq!(
            an(json!({"tool_choice": {"type": "auto"}}))
                .unwrap()
                .tool_choice,
            ToolChoice::Auto
        );
        // The stop token: parallel false with tools rendered adds </tool_call>.
        let sp = specials();
        let p = prompt(100);
        let mut f = oa(json!({"parallel_tool_calls": false})).unwrap();
        let g = assemble(&f, &p, &sp, true, &defaults(), 1);
        assert_eq!(g.stop_tokens, [7, 8, 248059]);
        let g = assemble(&f, &p, &sp, false, &defaults(), 1);
        assert_eq!(g.stop_tokens, [7, 8]);
        f.parallel = true;
        let g = assemble(&f, &p, &sp, true, &defaults(), 1);
        assert_eq!(g.stop_tokens, [7, 8]);
    }

    fn specials() -> Specials {
        Specials {
            im_start: 1,
            im_end: 2,
            think: 3,
            end_think: 4,
            tool_call: 5,
            end_tool_call: 248059,
            eog: vec![7, 8],
            think_close: vec![9, 4, 10],
            control_texts: vec![],
        }
    }

    fn prompt(n: usize) -> Prompt {
        Prompt {
            tokens: vec![0; n],
            cut: None,
            text: String::new(),
        }
    }

    #[test]
    fn max_tokens_is_clamped_to_the_room_left() {
        let sp = specials();
        let p = prompt(100);
        let d = defaults();
        let f = oa(json!({"max_tokens": 32000})).unwrap();
        assert_eq!(f.max_tokens, Some(32000));
        assert_eq!(assemble(&f, &p, &sp, false, &d, 1).max_tokens, 900);
        let f = oa(json!({})).unwrap();
        assert_eq!(assemble(&f, &p, &sp, false, &d, 1).max_tokens, 900);
        let f = oa(json!({"max_tokens": 50})).unwrap();
        assert_eq!(assemble(&f, &p, &sp, false, &d, 1).max_tokens, 50);
        // max_completion_tokens is the same field.
        let f = oa(json!({"max_completion_tokens": 60})).unwrap();
        assert_eq!(f.max_tokens, Some(60));
    }

    #[test]
    fn assemble_carries_the_think_limit_the_seed_and_the_sampling() {
        let sp = specials();
        let p = Prompt {
            tokens: vec![1, 2, 3],
            cut: Some(crate::backend::Cut {
                at: 2,
                user_query: true,
            }),
            text: String::new(),
        };
        let d = defaults();
        let f = oa(json!({"reasoning_effort": "low", "logprobs": true, "top_logprobs": 3, "temperature": 0})).unwrap();
        let g = assemble(&f, &p, &sp, false, &d, 77);
        assert_eq!(g.prompt, [1, 2, 3]);
        assert_eq!(g.cut, p.cut);
        assert_eq!(g.seed, 77);
        assert_eq!(g.logprobs, Some(3));
        assert_eq!(g.sampling.temp, 0.0);
        assert_eq!(
            g.think,
            Some(ThinkLimit {
                open: true,
                budget: Some(1024),
                ends: [4, 5],
                close: vec![9, 4, 10]
            })
        );
        let f = oa(json!({"reasoning_effort": "none", "seed": 5})).unwrap();
        let g = assemble(&f, &p, &sp, false, &d, 77);
        assert_eq!((g.think, g.seed), (None, 5));
        // A negative seed means "random", as llama.cpp's clients send it.
        assert_eq!(oa(json!({"seed": -1})).unwrap().seed, None);
        assert_eq!(
            oa(json!({"seed": 18446744073709551615u64})).unwrap().seed,
            Some(u64::MAX)
        );
        assert_eq!(
            oa(json!({"seed": 1.5})).unwrap_err().param.as_deref(),
            Some("seed")
        );
    }

    #[test]
    fn anthropic_requires_max_tokens() {
        let body = br#"{"messages": [{"role": "user", "content": "q"}]}"#;
        let mut r = anthropic_parse(body).unwrap();
        let e = r.fields(&defaults(), false).unwrap_err();
        assert_eq!((e.status(), e.param.as_deref()), (400, Some("max_tokens")));
        // count_tokens takes none.
        let mut r = anthropic_parse(body).unwrap();
        assert_eq!(r.fields(&defaults(), true).unwrap().max_tokens, None);
        assert_eq!(
            an(json!({"max_tokens": 0})).unwrap_err().param.as_deref(),
            Some("max_tokens")
        );
        let f = an(json!({"max_tokens": 5, "stop_sequences": ["a", "b"], "stream": true})).unwrap();
        assert_eq!(
            (f.max_tokens, f.stop, f.stream),
            (Some(5), vec!["a".to_string(), "b".into()], true)
        );
        assert!(!f.include_usage);
        let e = an(json!({"stop_sequences": "a"})).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("stop_sequences"));
    }

    #[test]
    fn a_stop_string_of_1024_bytes_is_accepted() {
        let s = "é".repeat(512);
        assert_eq!(s.len(), 1024);
        assert_eq!(
            oa(json!({"stop": s.clone()})).unwrap().stop,
            vec![s.clone()]
        );
        assert_eq!(
            oa(json!({"stop": [s.clone()]})).unwrap().stop,
            vec![s.clone()]
        );
        assert_eq!(
            an(json!({"stop_sequences": [s.clone()]})).unwrap().stop,
            vec![s]
        );
    }

    #[test]
    fn a_stop_string_of_1025_bytes_is_refused() {
        let s = "é".repeat(512) + "x";
        for (e, key) in [
            (oa(json!({"stop": s.clone()})).unwrap_err(), "stop"),
            (
                oa(json!({"stop": ["a", s.clone()]})).unwrap_err(),
                "stop[1]",
            ),
            (
                an(json!({"stop_sequences": [s.clone()]})).unwrap_err(),
                "stop_sequences[0]",
            ),
        ] {
            assert_eq!(e.param.as_deref(), Some(key));
            assert!(e.message.contains("1,024 bytes"), "{}", e.message);
        }
    }

    #[test]
    fn a_three_byte_character_at_the_stop_cap_is_counted_in_bytes() {
        // 1,021 bytes of ASCII and a 3-byte character end exactly at 1,024; 1,022 and the character end at 1,025.
        let fits = "x".repeat(1021) + "€";
        assert_eq!(fits.len(), 1024);
        assert_eq!(
            oa(json!({"stop": [fits.clone()]})).unwrap().stop,
            vec![fits]
        );
        // 341 of them are 1,023 bytes; one more byte is the cap, one more past it.
        let full = "€".repeat(341) + "x";
        assert_eq!(full.len(), 1024);
        assert_eq!(
            oa(json!({"stop": [full.clone()]})).unwrap().stop,
            vec![full]
        );
        for over in ["x".repeat(1022) + "€", "€".repeat(342)] {
            assert!(over.len() >= 1025);
            let e = oa(json!({"stop": [over]})).unwrap_err();
            assert_eq!(e.param.as_deref(), Some("stop[0]"));
            assert!(e.message.contains("1,024 bytes"), "{}", e.message);
        }
    }

    #[test]
    fn sampling_is_checked_by_field() {
        for (body, param) in [
            (json!({"temperature": -1}), "temperature"),
            (json!({"top_p": 0}), "top_p"),
            (json!({"top_p": 1.5}), "top_p"),
            (json!({"min_p": 2}), "min_p"),
            (json!({"min_p": -0.1}), "min_p"),
            (json!({"top_k": -1}), "top_k"),
            (json!({"top_k": 2.5}), "top_k"),
            (json!({"presence_penalty": "x"}), "presence_penalty"),
            (json!({"temperature": 1e300}), "temperature"),
        ] {
            let e = oa(body.clone()).unwrap_err();
            assert_eq!(
                (e.status(), e.param.as_deref()),
                (400, Some(param)),
                "{body}"
            );
            let e = an(body.clone()).unwrap_err();
            assert_eq!(e.param.as_deref(), Some(param), "{body}");
        }
        let f = oa(
            json!({"temperature": 0.2, "top_p": 1, "top_k": 0, "min_p": 0.05,
            "presence_penalty": 1.5}),
        )
        .unwrap();
        assert_eq!(
            f.sampling,
            Sampling {
                temp: 0.2,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.05,
                presence_penalty: 1.5
            }
        );
        // A field not sent keeps the config's value.
        let f = oa(json!({"temperature": 0})).unwrap();
        assert_eq!((f.sampling.top_k, f.sampling.top_p), (20, 0.95));
    }

    #[test]
    fn m4_output_config_format_is_400() {
        const M: &str = "MARKER-ECHO-5c1d";
        for format in [
            json!({"type": "json_schema", "schema": {}}),
            json!({"type": "json_schema", "schema": {M: {"type": M}}}),
            json!({"type": M}),
            json!(M),
            json!({}),
        ] {
            let body = json!({"output_config": {"format": format}});
            let e = an(body.clone()).unwrap_err();
            assert_eq!(e.status(), 400, "{body}");
            assert_eq!(e.kind, ErrorType::InvalidRequest, "{body}");
            assert_eq!(e.param.as_deref(), Some("output_config.format"), "{body}");
            assert_eq!(
                e.message,
                "structured outputs (output_config.format) are not supported"
            );
            assert!(e.message.len() <= 500 && !e.message.contains(M), "{e:?}");
        }
        // Beside an effort, and whatever `thinking` says, it is still a 400.
        let e = an(json!({"thinking": {"type": "disabled"},
            "output_config": {"effort": "high", "format": {"type": "json_schema"}}}))
        .unwrap_err();
        assert_eq!(e.param.as_deref(), Some("output_config.format"));
        // A null format is an absent one.
        assert!(an(json!({"output_config": {"format": null}})).is_ok());
    }

    /// An Anthropic body with `max_tokens` and `extra`, under `d`.
    fn an_with(d: &Defaults, max_tokens: u32, extra: Value) -> Thinking {
        let mut body =
            json!({"messages": [{"role": "user", "content": "q"}], "max_tokens": max_tokens});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        anthropic_parse(&serde_json::to_vec(&body).unwrap())
            .unwrap()
            .fields(d, false)
            .unwrap()
            .thinking
    }

    #[test]
    fn anthropic_thinking_false_follows_anthropic() {
        let mut d = defaults();
        d.anthropic_thinking = false;
        // Nothing said: off, as Anthropic's API means it; the OpenAI side keeps the config's thinking.
        assert_eq!(an_with(&d, 4096, json!({})), think(false, None));
        assert_eq!(
            an_with(&d, 4096, json!({"thinking": {"type": "enabled"}})),
            think(true, None)
        );
        assert_eq!(
            an_with(&d, 4096, json!({"output_config": {"effort": "low"}})),
            think(true, Some(LOW))
        );
        assert_eq!(
            an_with(&d, 4096, json!({"thinking": {"type": "adaptive"}})),
            think(true, None)
        );
        let mut r = openai_parse(br#"{"messages": [{"role": "user", "content": "q"}]}"#).unwrap();
        assert_eq!(r.fields(&d).unwrap().thinking, think(true, None));
        // The default is llama.cpp's: on.
        assert!(defaults().anthropic_thinking);
        assert_eq!(an_with(&defaults(), 4096, json!({})), think(true, None));
    }

    #[test]
    fn a_small_max_tokens_caps_default_thinking_at_half() {
        let d = defaults();
        assert_eq!(an_with(&d, 200, json!({})), think(true, Some(100)));
        assert_eq!(an_with(&d, 1023, json!({})), think(true, Some(511)));
        assert_eq!(an_with(&d, 1024, json!({})), think(true, None));
        assert_eq!(an_with(&d, 1, json!({})), think(true, Some(1)));
        // A request that states thinking is untouched.
        assert_eq!(
            an_with(&d, 200, json!({"thinking": {"type": "enabled"}})),
            think(true, None)
        );
        let mut c = defaults();
        c.thinking_budget = Some(500);
        assert_eq!(
            an_with(&c, 200, json!({"thinking": {"type": "enabled"}})),
            think(true, Some(500))
        );
        assert_eq!(
            an_with(&c, 200, json!({"output_config": {"effort": "high"}})),
            think(true, None)
        );
        // The configured budget wins when it is smaller; the half wins when it is.
        let mut c = defaults();
        c.thinking_budget = Some(64);
        assert_eq!(an_with(&c, 200, json!({})), think(true, Some(64)));
        assert_eq!(an_with(&c, 100, json!({})), think(true, Some(50)));
        // A budget with a large max_tokens is the configured one.
        assert_eq!(an_with(&c, 4096, json!({})), think(true, Some(64)));
        // Thinking off in the config stays off.
        let mut c = defaults();
        c.thinking = false;
        assert_eq!(an_with(&c, 200, json!({})), think(false, None));
        // count_tokens has no max_tokens, so nothing to cap.
        let mut r =
            anthropic_parse(br#"{"messages": [{"role": "user", "content": "q"}]}"#).unwrap();
        assert_eq!(
            r.fields(&defaults(), true).unwrap().thinking,
            think(true, None)
        );
    }

    #[test]
    fn an_unknown_thinking_type_still_gets_the_half_cap() {
        let d = defaults();
        // An unknown type is ignored, so it says nothing and the default's small-request cap applies.
        assert_eq!(
            an_with(&d, 200, json!({"thinking": {"type": "mystery"}})),
            think(true, Some(100))
        );
        assert_eq!(
            an_with(&d, 200, json!({"thinking": {}})),
            think(true, Some(100))
        );
        assert_eq!(
            an_with(&d, 200, json!({"output_config": {"effort": "mystery"}})),
            think(true, Some(100))
        );
    }

    #[test]
    fn an_unknown_effort_does_not_turn_thinking_on_under_anthropic_thinking_false() {
        let mut d = defaults();
        d.anthropic_thinking = false;
        assert_eq!(
            an_with(&d, 4096, json!({"output_config": {"effort": "mystery"}})),
            think(false, None)
        );
        assert_eq!(
            an_with(&d, 4096, json!({"thinking": {"type": "mystery"}})),
            think(false, None)
        );
    }

    #[test]
    fn output_config_effort_alone_still_passes() {
        let f = an(json!({"output_config": {"effort": "low"}})).unwrap();
        assert_eq!(f.thinking, think(true, Some(LOW)));
        assert!(f.ignored.is_empty(), "{:?}", f.ignored);
        let f = an(json!({"output_config": {"effort": "medium"}})).unwrap();
        assert_eq!(f.thinking, think(true, Some(MEDIUM)));
        let f = an(json!({"thinking": {"type": "disabled"}, "output_config": {"effort": "high"}}))
            .unwrap();
        assert_eq!(f.thinking, think(false, None));
        assert_eq!(f.ignored, ["output_config.effort"]);
    }

    #[test]
    fn a_wrong_type_never_echoes_the_value() {
        const M: &str = "MARKER-ECHO-8e2a";
        // Every case must be a 400 that names a field and does not quote the value.
        let mut cases: Vec<(&str, Value, Value)> = Vec::new();
        for key in [
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "seed",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "logprobs",
            "top_logprobs",
            "parallel_tool_calls",
            "stream",
        ] {
            cases.push(("openai", json!({ key: M }), json!(key)));
            cases.push(("openai", json!({ key: {M: M} }), json!(key)));
        }
        for key in ["stream_options", "reasoning", "response_format"] {
            cases.push(("openai", json!({ key: M }), json!(key)));
        }
        for key in ["model", "stop", "reasoning_effort"] {
            cases.push(("openai", json!({ key: {M: M} }), json!(key)));
        }
        cases.push((
            "openai",
            json!({"reasoning_effort": M}),
            json!("reasoning_effort"),
        ));
        cases.push((
            "openai",
            json!({"reasoning": {"effort": M}}),
            json!("reasoning.effort"),
        ));
        cases.push((
            "openai",
            json!({"reasoning": {"effort": {M: M}}}),
            json!("reasoning.effort"),
        ));
        cases.push(("openai", json!({"tool_choice": M}), json!("tool_choice")));
        cases.push(("openai", json!({"tool_choice": [M]}), json!("tool_choice")));
        cases.push(("openai", json!({"stop": [M, 5]}), json!("stop[1]")));
        cases.push(("openai", json!({"stop": [5, M]}), json!("stop[0]")));
        cases.push((
            "openai",
            json!({"stream_options": {"include_usage": M}}),
            json!("stream_options.include_usage"),
        ));
        cases.push((
            "openai",
            json!({"chat_template_kwargs": {"enable_thinking": M}}),
            json!("chat_template_kwargs.enable_thinking"),
        ));
        for key in [
            "temperature",
            "max_tokens",
            "stream",
            "thinking",
            "output_config",
            "tool_choice",
        ] {
            cases.push(("anthropic", json!({ key: M }), json!(key)));
        }
        for key in ["temperature", "max_tokens", "model", "stop_sequences"] {
            cases.push(("anthropic", json!({ key: {M: M} }), json!(key)));
        }
        cases.push((
            "anthropic",
            json!({"stop_sequences": M}),
            json!("stop_sequences"),
        ));
        cases.push((
            "anthropic",
            json!({"stop_sequences": [M, 5]}),
            json!("stop_sequences[1]"),
        ));
        cases.push((
            "anthropic",
            json!({"thinking": {"type": "enabled", "budget_tokens": M}}),
            json!("thinking.budget_tokens"),
        ));
        cases.push((
            "anthropic",
            json!({"thinking": {"type": {M: M}}}),
            json!("thinking.type"),
        ));
        cases.push((
            "anthropic",
            json!({"output_config": {"effort": {M: M}}}),
            json!("output_config.effort"),
        ));
        cases.push((
            "anthropic",
            json!({"tool_choice": {"type": M}}),
            json!("tool_choice.type"),
        ));
        cases.push((
            "anthropic",
            json!({"tool_choice": {"type": "auto", "disable_parallel_tool_use": M}}),
            json!("tool_choice.disable_parallel_tool_use"),
        ));
        for (api, body, param) in cases {
            let r = if api == "openai" {
                oa(body.clone())
            } else {
                an(body.clone())
            };
            let e = r.expect_err(&format!("{api} {body} must be 400"));
            assert_eq!(e.status(), 400, "{api} {body}");
            assert!(!e.message.contains(M), "{api} {body}: {e:?}");
            let got = e.param.clone().unwrap_or_default();
            assert!(!got.contains(M), "{api} {body}: {e:?}");
            assert_eq!(got, param.as_str().unwrap(), "{api} {body}");
        }
    }
}
