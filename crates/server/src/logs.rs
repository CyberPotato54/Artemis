//! The request log: one JSON line per request in `log_dir/artemis-YYYY-MM-DD.jsonl` (UTC), the lifecycle lines beside
//! them, and a human-readable twin of each on stderr through `tracing`. No prompt or output text and no key
//! reaches a line: every field here is a count, a time, a name the server chose or an id.
use crate::backend::{
    BackendStats, CacheReport, Counts, Cut, Source, SpecReport, StatsCopy, Timings,
};
use crate::error::{ApiError, Flavour};
use crate::ids::{iso_utc, parse_date, unix_now, utc_date};
use crate::tokens::Specials;
use qwen36::tokenizer::Tokenizer;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// Why a request's cache restore stopped short of its prompt. The line carries it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Cause {
    None,
    NewConversation,
    ReasoningDropped,
    ReplyRetokenized,
    NewInput,
    NoSnapshot,
}

/// Why a request's restore stopped where it did, from the prompt's own tokens: the turn `diverged_at` falls
/// in (the last `<|im_start|>` at or before it, its role the text of the token after) and whether it lies in a think
/// block.
///
/// - At or past the cut there is no divergence: the generation prompt is always new. Nothing restored and a parting
///   point past 0 is `NoSnapshot`; a parting point of 0 is a new conversation.
/// - At a turn's end, the cache held the history up to it and what follows is new input, whatever turn the next token
///   belongs to. The engine never decodes the stop token, so the reply-end snapshot ends at the reply's last content
///   token and a chat that continues parts at the `<|im_end|>` itself (or at a `</tool_call>` closing a call, with the
///   `<|im_end|>` after it); right after an `<|im_end|>` counts too. The `<|im_end|>` of the first turn or of a
///   `system` turn is not a turn's end in this sense: a first message or a system prompt cut short is the next rule's.
/// - In the first turn, or in a `system` or `developer` turn: `NewConversation` (the prompt does not open as anything
///   held did). In a `user` turn (tool results are `user` turns), or in a turn of any other role: `NewInput`.
/// - In an `assistant` turn: `ReasoningDropped` in its think block (the `<think>` to the `</think>`, both included)
///   or, at the start of its body (right after `assistant\n`), when the turn has none, which is the template
///   dropping the reasoning of a turn at or before the last query (a model's reply always opens one);
///   `ReplyRetokenized` anywhere else in it.
pub fn classify_divergence(
    prompt: &[u32],
    cut: Option<&Cut>,
    diverged_at: u32,
    restored: &Source,
    sp: &Specials,
    tk: &Tokenizer,
) -> Cause {
    let d = diverged_at as usize;
    if cut.is_some_and(|c| diverged_at >= c.at) || d >= prompt.len() {
        return Cause::None;
    }
    if d == 0 {
        return Cause::NewConversation;
    }
    if *restored == Source::None {
        return Cause::NoSnapshot;
    }
    let ends_call = prompt[d] == sp.end_tool_call && prompt.get(d + 1) == Some(&sp.im_end);
    let turn = prompt[..=d].iter().rposition(|&t| t == sp.im_start);
    let role_of = |start: usize| {
        prompt
            .get(start + 1)
            .map(|&t| tk.decode(&[t]))
            .unwrap_or_default()
    };
    // A turn's own end is new input only past the first turn and the system turn: a first message or a system prompt
    // cut short still parts the conversation from what the cache held.
    let ends_turn = prompt[d] == sp.im_end
        && turn.is_some_and(|start| {
            start != 0 && matches!(role_of(start).trim(), "user" | "assistant")
        });
    if ends_turn || prompt[d - 1] == sp.im_end || ends_call {
        return Cause::NewInput;
    }
    let Some(start) = turn else {
        return Cause::NewConversation;
    };
    let role = role_of(start);
    match role.trim() {
        _ if start == 0 => Cause::NewConversation,
        "system" | "developer" => Cause::NewConversation,
        "assistant" => {
            let end = prompt[start + 1..]
                .iter()
                .position(|&t| t == sp.im_start)
                .map_or(prompt.len(), |n| start + 1 + n);
            let turn = &prompt[start..end];
            let Some(open) = turn.iter().position(|&t| t == sp.think) else {
                // No think block: the template dropped it, which shows where the body starts.
                let newline = prompt
                    .get(start + 2)
                    .is_some_and(|&t| tk.decode(&[t]) == "\n");
                return if d == start + 2 + usize::from(newline) {
                    Cause::ReasoningDropped
                } else {
                    Cause::ReplyRetokenized
                };
            };
            let close = turn[open..]
                .iter()
                .position(|&t| t == sp.end_think)
                .map_or(end, |n| start + open + n);
            if (start + open..=close).contains(&d) {
                Cause::ReasoningDropped
            } else {
                Cause::ReplyRetokenized
            }
        }
        _ => Cause::NewInput,
    }
}

/// How many characters of the request's `model` the line keeps: the name is the client's text.
pub const MODEL_CHARS: usize = 64;

/// How many repairs a request's line names; the rest are counted in `repairs_more`.
pub const REPAIRS_SHOWN: usize = 16;

/// The first `MODEL_CHARS` characters of a model name, as the line may carry it.
pub fn cut_model(s: &str) -> String {
    s.chars().take(MODEL_CHARS).collect()
}

/// The line one request leaves. A field the request never reached is `null` (a 401 has no token counts), and `ignored`
/// and `repairs` are lists of field names and repair names, never values: at most `IGNORED_SHOWN` and `REPAIRS_SHOWN`
/// of them, with the rest counted in `ignored_more` and `repairs_more`, so a line stays small whatever the request.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RequestLine {
    /// `YYYY-MM-DDTHH:MM:SSZ`; the file is named by its date.
    pub time: String,
    pub id: String,
    pub endpoint: String,
    /// The client's IP address (no port), set only on the guard's refusals (401, 403, 415) so a run of failed keys
    /// shows where it came from; `null` otherwise. An address is not prompt text.
    pub peer: Option<String>,
    /// The model the request named, cut to `MODEL_CHARS` characters.
    pub model: Option<String>,
    pub status: u16,
    pub prompt_tokens: Option<u32>,
    /// The token the generation prompt starts at, and whether the turn before it is the user's query;
    /// `null` for a prompt with no cut, or a request that never got a prompt.
    pub cut_at: Option<u32>,
    pub user_query: Option<bool>,
    /// Whether the reply was asked for as a stream.
    pub stream: bool,
    /// Valid calls the reply emitted, and `<tool_call>` texts returned as content because they were not valid calls.
    pub tool_calls: u32,
    pub malformed_calls: u32,
    /// The request's thinking budget; `null` with thinking off or no budget. `budget_closed` is true when the engine
    /// forced the close at that budget.
    pub think_budget: Option<u32>,
    pub budget_closed: bool,
    pub reused_tokens: Option<u32>,
    /// Where the restored state came from (`none`, `live`, `interval`, `prompt_end`, `user_query`, `reply_end`).
    pub restored: Option<String>,
    pub restore_bytes: Option<u64>,
    pub restore_ms: Option<f64>,
    pub diverged_at: Option<u32>,
    pub cause: Option<Cause>,
    pub prefilled_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub reasoning_tokens: Option<u32>,
    pub queue_ms: Option<f64>,
    pub load_ms: Option<f64>,
    pub prefill_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    /// How long the request ran, from the line being made (`time` is that moment) to its being written.
    /// `Logger::request` fills it when the caller has not, so every line carries it: a refusal, a stream that failed
    /// mid-way, a cancel.
    pub total_ms: Option<f64>,
    pub prefill_tok_s: Option<f64>,
    pub decode_tok_s: Option<f64>,
    /// The decode phase: its time, the tokens decoded after the first (`completion - 1`), and the expert cache's
    /// misses during it.
    pub decode_ms: Option<f64>,
    pub decode_tokens: Option<u32>,
    pub decode_misses: Option<u64>,
    pub hit_rate: Option<f64>,
    pub miss_mib_per_token: Option<f64>,
    /// Speculation's counts for the request; all `null` with speculation off or no decode step. Counts
    /// only: no draft or output text reaches the line. `spec_accept_rate` is `spec_accepted / spec_drafted`, `null`
    /// when nothing was drafted.
    pub spec_steps: Option<u64>,
    pub spec_plain_steps: Option<u64>,
    pub spec_drafted: Option<u64>,
    pub spec_accepted: Option<u64>,
    pub spec_accept_rate: Option<f64>,
    pub spec_replays: Option<u64>,
    /// `spec_drafted` and `spec_accepted` by source (they sum to those two); `null` as they are.
    pub spec_drafted_lookup: Option<u64>,
    pub spec_accepted_lookup: Option<u64>,
    pub spec_drafted_mtp: Option<u64>,
    pub spec_accepted_mtp: Option<u64>,
    pub pool_bytes_used: Option<u64>,
    pub finish: Option<String>,
    pub error_kind: Option<String>,
    /// The error's message, cut to `ERROR_MESSAGE_CHARS` characters; `null` when `error_kind` is.
    pub error_message: Option<String>,
    pub ignored: Vec<String>,
    /// Ignored fields past the names `ignored` holds.
    pub ignored_more: u32,
    pub repairs: Vec<String>,
    /// Repairs past the ones `repairs` holds.
    pub repairs_more: u32,
    /// Messages the chat template drops: a `system` or `developer` message after the leading one or two, and a message
    /// of a role the template does not know (`convo::dropped_by_template`). The client gets a normal reply.
    pub dropped_messages: u32,
    /// When the line was made, for `total_ms`; not written.
    #[serde(skip)]
    started: Instant,
}

impl RequestLine {
    /// A line stamped now, with nothing else known yet.
    pub fn new(id: impl Into<String>, endpoint: impl Into<String>) -> RequestLine {
        RequestLine {
            time: iso_utc(unix_now()),
            id: id.into(),
            endpoint: endpoint.into(),
            peer: None,
            model: None,
            status: 0,
            prompt_tokens: None,
            cut_at: None,
            user_query: None,
            stream: false,
            tool_calls: 0,
            malformed_calls: 0,
            think_budget: None,
            budget_closed: false,
            reused_tokens: None,
            restored: None,
            restore_bytes: None,
            restore_ms: None,
            diverged_at: None,
            cause: None,
            prefilled_tokens: None,
            completion_tokens: None,
            reasoning_tokens: None,
            queue_ms: None,
            load_ms: None,
            prefill_ms: None,
            ttft_ms: None,
            total_ms: None,
            prefill_tok_s: None,
            decode_tok_s: None,
            decode_ms: None,
            decode_tokens: None,
            decode_misses: None,
            hit_rate: None,
            miss_mib_per_token: None,
            spec_steps: None,
            spec_plain_steps: None,
            spec_drafted: None,
            spec_accepted: None,
            spec_accept_rate: None,
            spec_replays: None,
            spec_drafted_lookup: None,
            spec_accepted_lookup: None,
            spec_drafted_mtp: None,
            spec_accepted_mtp: None,
            pool_bytes_used: None,
            finish: None,
            error_kind: None,
            error_message: None,
            ignored: Vec::new(),
            ignored_more: 0,
            repairs: Vec::new(),
            repairs_more: 0,
            dropped_messages: 0,
            started: Instant::now(),
        }
    }

    /// What the engine measured: its timings and the cache's figures. The two rates are the engine's, or `null` when
    /// it had none; the token rates are the counts over their phase's time, `null` when either is nothing. `spec` is
    /// the speculation report, `None` when the engine sent none.
    pub fn measured(
        &mut self,
        counts: &Counts,
        t: &Timings,
        c: &CacheReport,
        spec: Option<&SpecReport>,
    ) {
        self.queue_ms = Some(t.queue_ms);
        self.load_ms = Some(t.load_ms);
        self.restore_ms = Some(t.restore_ms);
        self.prefill_ms = Some(t.prefill_ms);
        self.ttft_ms = Some(t.ttft_ms);
        self.prefill_tok_s = rate(counts.prefilled, t.prefill_ms);
        self.decode_tok_s = rate(counts.completion, t.decode_ms);
        // The first token comes out of the prefill; the decode steps are the rest (the engine tool's count).
        self.decode_ms = Some(t.decode_ms);
        self.decode_tokens = Some(counts.completion.saturating_sub(1));
        self.decode_misses = c.decode_misses;
        self.hit_rate = c.hit_rate;
        self.miss_mib_per_token = c.miss_mib_per_token;
        self.restore_bytes = Some(c.restore_bytes);
        self.pool_bytes_used = Some(c.pool_bytes_used);
        self.spec_steps = spec.map(|s| s.steps);
        self.spec_plain_steps = spec.map(|s| s.plain_steps);
        self.spec_drafted = spec.map(|s| s.drafted);
        self.spec_accepted = spec.map(|s| s.accepted);
        self.spec_accept_rate = spec
            .filter(|s| s.drafted > 0)
            .map(|s| s.accepted as f64 / s.drafted as f64);
        self.spec_replays = spec.map(|s| s.replays);
        self.spec_drafted_lookup = spec.map(|s| s.drafted_lookup);
        self.spec_accepted_lookup = spec.map(|s| s.accepted_lookup);
        self.spec_drafted_mtp = spec.map(|s| s.drafted_mtp);
        self.spec_accepted_mtp = spec.map(|s| s.accepted_mtp);
    }

    /// Records the reply's repairs: the first `REPAIRS_SHOWN` by name and the count of the rest.
    pub fn note_repairs(&mut self, repairs: &[String]) {
        self.repairs = repairs.iter().take(REPAIRS_SHOWN).cloned().collect();
        self.repairs_more =
            u32::try_from(repairs.len().saturating_sub(REPAIRS_SHOWN)).unwrap_or(u32::MAX);
    }

    /// Records an error the request ended in: its kind in the flavour's vocabulary, and its message cut to
    /// `ERROR_MESSAGE_CHARS` characters (a message holds no request text).
    pub fn set_error(&mut self, e: &ApiError, flavour: Flavour) {
        self.error_kind = Some(e.kind.name(flavour).to_string());
        self.error_message = Some(e.message.chars().take(ERROR_MESSAGE_CHARS).collect());
    }
}

/// How many characters of an error's message the request line keeps.
pub const ERROR_MESSAGE_CHARS: usize = 300;

/// Tokens per second over `ms`; `None` for no tokens or no time.
fn rate(tokens: u32, ms: f64) -> Option<f64> {
    (tokens > 0 && ms.is_finite() && ms > 0.0).then(|| f64::from(tokens) / (ms / 1000.0))
}

/// A request's line, held while the request runs. If the holder is dropped before the line is taken (the client left
/// and the handler's future went with it), the line is written as it stands: `status` 499 (the client closed the
/// request; no reply was sent, so no HTTP status applies) and `finish` `cancelled`, with whatever was known.
pub struct LineGuard {
    logger: Logger,
    line: Option<RequestLine>,
}

impl LineGuard {
    pub fn new(logger: &Logger, line: RequestLine) -> LineGuard {
        LineGuard {
            logger: logger.clone(),
            line: Some(line),
        }
    }

    pub fn line(&mut self) -> &mut RequestLine {
        self.line.as_mut().expect("the line is held until taken")
    }

    /// The line, for its holder to write; the guard then writes nothing.
    pub fn take(mut self) -> RequestLine {
        self.line.take().expect("the line is held until taken")
    }
}

impl Drop for LineGuard {
    fn drop(&mut self) {
        if let Some(mut line) = self.line.take() {
            line.status = 499;
            line.finish = Some("cancelled".to_string());
            self.logger.request(&line);
        }
    }
}

/// What the server or the engine did to its own state: start, config warnings, loads and unloads and their
/// cause, device loss and exit. In the file each carries `event` (the variant, snake case) beside `time`. `Stats` is
/// written every ten minutes, so the size of the counters it carries costs nothing worth an indirection.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
pub enum LifecycleLine {
    Start {
        version: String,
        bind: String,
        model_id: String,
        /// The directory the logs go to.
        log_dir: String,
    },
    ConfigWarning {
        message: String,
    },
    /// The figures are the engine's, `null` where it gave none (the fake gives none): the slots each layer's expert
    /// cache holds, the VRAM the load took, the pool's limit and whether it was cut below the config's, and the
    /// parked snapshots restored into the pool.
    Load {
        seconds: f64,
        detail: String,
        slots_per_layer: Option<u32>,
        vram_used_mib: Option<f64>,
        pool_limit_gib: Option<f64>,
        pool_cut: Option<bool>,
        unparked: Option<u64>,
    },
    /// `cause` is `idle`, `request` or `shutdown`.
    Unload {
        cause: String,
    },
    DeviceLost {
        message: String,
    },
    /// A load the engine could not finish, with its message (the engine's, never a request's text).
    LoadFailed {
        message: String,
    },
    /// A reload after a device loss that the engine could not finish, with its message.
    ReloadFailed {
        message: String,
    },
    /// A panic on any thread (from the panic hook). `thread` is `unnamed` for a thread with no name and `location` is
    /// `file:line:col`. `message` is the payload's text only when it is a `&'static str`, which is compiled into the
    /// binary; a formatted payload can carry request text, so it is `null`.
    Panic {
        thread: String,
        location: Option<String>,
        message: Option<String>,
    },
    /// Work that stopped moving (from the watchdog): what it was (`request`, `load`, `reload` or `command`), the oldest
    /// request in flight, how long it was quiet against its bound, and the progress counter it stopped at.
    Stall {
        doing: String,
        request: Option<String>,
        quiet_s: f64,
        bound_s: u64,
        ticks: u64,
    },
    /// The heartbeat, and the line after each `unload` and before each `exit`. The figures are the engine's latest
    /// answer to `stats` and its age in seconds, `null` before there is one; `log_write_failures` is `Logger::write_failures`.
    Stats {
        uptime_s: u64,
        age_s: Option<f64>,
        log_write_failures: u64,
        stats: Option<BackendStats>,
    },
    Exit {
        reason: String,
    },
    /// `artemis setup` began measuring this PC: `reason` is `first run` (`serve` found no tuned file) or `asked`, and
    /// `path` is the tuned file it will write.
    SetupStart {
        reason: String,
        path: String,
    },
    /// Setup wrote the tuned file. `enabled`, `sources` and `priors` are what it chose; `scores` are the measured
    /// decode speeds in tok/s by candidate; `measured_slope_ms` and `measured_mtp_d1_ms` are what the controller
    /// measured, `null` where it measured nothing; `expected_tok_s` is 0 when nothing was measured.
    SetupDone {
        seconds: f64,
        limit_hit: bool,
        target: bool,
        gpu_name: String,
        vram_bytes: u64,
        cache_slots: u32,
        scores: BTreeMap<String, f64>,
        enabled: bool,
        sources: Vec<String>,
        priors: Option<SetupPriors>,
        measured_slope_ms: Option<f64>,
        measured_mtp_d1_ms: Option<f64>,
        expected_tok_s: f64,
        path: String,
    },
    /// Setup stopped without writing the tuned file; `message` is the sentence it printed.
    SetupFailed {
        message: String,
    },
    /// At every start: the tuned file's `path`, the keys it `applied` and the ones it `ignored` because the user set
    /// them. Both lists are empty when the file is missing.
    Tuned {
        path: String,
        applied: Vec<String>,
        ignored: Vec<String>,
    },
}

/// The measured costs a setup run chose in place of the built-in ones.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SetupPriors {
    pub cost_slope_ms: f64,
    pub mtp_cost_ms: f64,
}

/// The `stats` line for the backend's latest copy (`Backend::latest_stats`), or for none yet.
pub fn stats_line(copy: Option<&StatsCopy>, uptime_s: u64, write_failures: u64) -> LifecycleLine {
    LifecycleLine::Stats {
        uptime_s,
        age_s: copy.map(|c| c.at.elapsed().as_secs_f64()),
        log_write_failures: write_failures,
        stats: copy.map(|c| c.stats.clone()),
    }
}

impl LifecycleLine {
    /// The line as a person reads it on stderr.
    pub fn human(&self) -> String {
        match self {
            LifecycleLine::Start {
                version,
                bind,
                model_id,
                log_dir,
            } => format!(
                "artemis {version} listening on {bind}, model {model_id}, logs in {log_dir}"
            ),
            LifecycleLine::ConfigWarning { message } => format!("config: {message}"),
            LifecycleLine::Load {
                seconds, detail, ..
            } => {
                format!("model loaded in {seconds:.1} s: {detail}")
            }
            LifecycleLine::Unload { cause } => format!("model unloaded ({cause})"),
            LifecycleLine::DeviceLost { message } => format!("device lost: {message}"),
            LifecycleLine::LoadFailed { message } => format!("the model did not load: {message}"),
            LifecycleLine::ReloadFailed { message } => {
                format!("the model did not reload after the GPU was lost: {message}")
            }
            LifecycleLine::Panic {
                thread,
                location,
                message,
            } => format!(
                "panic in {thread} at {}: {}",
                location.as_deref().unwrap_or("an unknown place"),
                message
                    .as_deref()
                    .unwrap_or("(message withheld: it may carry request text)")
            ),
            LifecycleLine::Stall {
                doing,
                quiet_s,
                bound_s,
                ..
            } if doing == "load" || doing == "reload" => format!(
                "stalled: the model load made no progress for {quiet_s:.0} s (bound {bound_s} s); \
                 if the model file is on a slow disk, raise stall_load_s"
            ),
            LifecycleLine::Stall {
                doing,
                request,
                quiet_s,
                bound_s,
                ..
            } => format!(
                "stalled: {doing}{} made no progress for {quiet_s:.0} s (bound {bound_s} s)",
                request.as_ref().map_or(String::new(), |r| format!(" {r}"))
            ),
            LifecycleLine::Stats {
                uptime_s,
                age_s,
                log_write_failures,
                stats,
            } => {
                let figures = match (stats, age_s) {
                    (Some(s), Some(age)) => format!(
                        "{} loads, {} device losses, {} snapshots held, {} restores (copy {age:.0} s old)",
                        s.loads, s.device_losses, s.snapshots_held, s.restores
                    ),
                    _ => "no figures from the engine yet".to_string(),
                };
                format!(
                    "stats: up {uptime_s} s, {figures}, {log_write_failures} log write failures"
                )
            }
            LifecycleLine::Exit { reason } => format!("exiting: {reason}"),
            LifecycleLine::SetupStart { reason, path } => {
                format!("setup ({reason}): measuring this PC, then writing {path}")
            }
            LifecycleLine::SetupDone {
                seconds,
                gpu_name,
                enabled,
                sources,
                expected_tok_s,
                path,
                ..
            } => {
                let speed = if *expected_tok_s > 0.0 {
                    format!(", about {expected_tok_s:.0} tok/s")
                } else {
                    String::new()
                };
                let choice = if *enabled {
                    format!("speculation on with {}", sources.join(" and "))
                } else {
                    "speculation off".to_string()
                };
                format!("setup done in {seconds:.0} s on {gpu_name}: {choice}{speed}; wrote {path}")
            }
            LifecycleLine::SetupFailed { message } => format!("setup failed: {message}"),
            LifecycleLine::Tuned {
                path,
                applied,
                ignored,
            } => format!(
                "tuned settings {path}: {} applied, {} ignored because you set them",
                applied.len(),
                ignored.len()
            ),
        }
    }
}

/// A lifecycle line with its time.
#[derive(Serialize)]
struct Stamped<'a> {
    time: String,
    #[serde(flatten)]
    line: &'a LifecycleLine,
}

struct Open {
    date: String,
    file: File,
}

struct Files {
    dir: PathBuf,
    /// Days of files kept (`prune`); 0 keeps all.
    keep_days: u32,
    open: Option<Open>,
    /// A failed write is reported once, not on every request.
    warned: bool,
}

/// The JSONL writer. Cheap to clone; every clone appends to the same file under one lock, one `write_all` per line.
#[derive(Clone)]
pub struct Logger {
    files: Arc<Mutex<Files>>,
    /// The directory as the start line names it (absolute where the platform can say), kept beside the lock.
    dir: Arc<PathBuf>,
    /// Lines that could not be written (a failed serialize, open or write), kept beside the lock so a reader never
    /// waits on a write.
    failures: Arc<AtomicU64>,
}

impl Logger {
    /// Creates `dir` if it is missing, and removes the day files older than `keep_days` days (`prune`; 0 keeps all).
    pub fn new(dir: PathBuf, keep_days: u32) -> std::io::Result<Logger> {
        std::fs::create_dir_all(&dir)?;
        prune(&dir, &utc_date(unix_now()), keep_days);
        Ok(Logger {
            dir: Arc::new(std::path::absolute(&dir).unwrap_or_else(|_| dir.clone())),
            files: Arc::new(Mutex::new(Files {
                dir,
                keep_days,
                open: None,
                warned: false,
            })),
            failures: Arc::new(AtomicU64::new(0)),
        })
    }

    /// The directory the logs go to, as the start line names it.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Holds the writer's lock, as a write stuck on a slow disk would.
    #[cfg(test)]
    pub(crate) fn hold_the_lock(&self) -> impl Drop + '_ {
        self.files.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// How many lines were dropped because they could not be serialized, or their file opened or written.
    pub fn write_failures(&self) -> u64 {
        self.failures.load(Ordering::Relaxed)
    }

    /// Writes the request's line to the file of its date and a summary to stderr: the one stderr line a request
    /// leaves, with the counts of what the server ignored and what the template dropped when there were any.
    pub fn request(&self, l: &RequestLine) {
        tracing::info!("{}", stderr_line(l));
        let total_ms = l
            .total_ms
            .unwrap_or_else(|| l.started.elapsed().as_secs_f64() * 1000.0);
        let line = RequestLine {
            total_ms: Some(total_ms),
            ..l.clone()
        };
        self.write(l.time.get(..10), serde_json::to_string(&line));
    }

    pub fn lifecycle(&self, l: &LifecycleLine) {
        match l {
            LifecycleLine::ConfigWarning { .. }
            | LifecycleLine::DeviceLost { .. }
            | LifecycleLine::LoadFailed { .. }
            | LifecycleLine::ReloadFailed { .. }
            | LifecycleLine::Panic { .. }
            | LifecycleLine::SetupFailed { .. }
            | LifecycleLine::Stall { .. } => tracing::warn!("{}", l.human()),
            _ => tracing::info!("{}", l.human()),
        }
        let now = unix_now();
        let time = iso_utc(now);
        let stamped = Stamped {
            time: time.clone(),
            line: l,
        };
        self.write(Some(&utc_date(now)), serde_json::to_string(&stamped));
    }

    /// `lifecycle` on a helper thread named `artemis-log`, waited on for at most `within`: for a caller that must not
    /// hang on the log's lock or a stuck disk, the panic hook above all. Returns whether the line was written in
    /// time; a line that is late is still written when the lock frees. The thread is started with `Builder::spawn`,
    /// not `thread::spawn`, which panics on failure: inside the panic hook that is a double panic, which aborts. A
    /// failed spawn writes the line's text to stderr instead and returns `false`. Each call that times out leaves its
    /// `artemis-log` thread blocked on the lock until the lock frees, so this is for the panic and halt paths, not a
    /// hot path.
    pub fn lifecycle_within(&self, l: &LifecycleLine, within: Duration) -> bool {
        let (done, wait) = mpsc::channel();
        let (logger, line) = (self.clone(), l.clone());
        let spawned = std::thread::Builder::new()
            .name("artemis-log".into())
            .spawn(move || {
                logger.lifecycle(&line);
                let _ = done.send(());
            });
        match spawned {
            Ok(_) => wait.recv_timeout(within).is_ok(),
            Err(_) => {
                // Not `eprintln!`, which panics when stderr is gone.
                let _ = writeln!(std::io::stderr(), "{}", l.human());
                false
            }
        }
    }

    /// Appends `line` to `artemis-<date>.jsonl`, opening it first if the date changed. A line that cannot be written
    /// is dropped with one warning: logging never fails a request.
    fn write(&self, date: Option<&str>, line: serde_json::Result<String>) {
        let mut files = self.files.lock().unwrap_or_else(PoisonError::into_inner);
        let date = date
            .filter(|d| d.len() == 10)
            .map_or_else(|| utc_date(unix_now()), str::to_string);
        let result = line.map_err(std::io::Error::from).and_then(|mut line| {
            line.push('\n');
            if files.open.as_ref().is_none_or(|o| o.date != date) {
                // A new day's file: the old one is closed first, then the days past retention go.
                files.open = None;
                prune(&files.dir, &utc_date(unix_now()), files.keep_days);
                let path = files.dir.join(format!("artemis-{date}.jsonl"));
                let file = OpenOptions::new().create(true).append(true).open(path)?;
                files.open = Some(Open {
                    date: date.clone(),
                    file,
                });
            }
            match files.open.as_mut() {
                Some(o) => o.file.write_all(line.as_bytes()),
                None => Ok(()),
            }
        });
        if let Err(e) = result {
            self.failures.fetch_add(1, Ordering::Relaxed);
            if !files.warned {
                files.warned = true;
                tracing::warn!("cannot write the request log: {:?}", e.kind());
            }
        }
    }
}

/// The request's line on stderr: `{id} {endpoint} -> {status} {finish}`, then ` ({n} fields ignored)` and
/// ` ({m} messages dropped by the template)` when there were any.
fn stderr_line(l: &RequestLine) -> String {
    let mut text = format!("{} {} -> {}", l.id, l.endpoint, l.status);
    if let Some(finish) = l.finish.as_deref().filter(|f| !f.is_empty()) {
        text.push(' ');
        text.push_str(finish);
    }
    let ignored = l.ignored.len() as u64 + u64::from(l.ignored_more);
    if ignored > 0 {
        text.push_str(&format!(" ({ignored} fields ignored)"));
    }
    if l.dropped_messages > 0 {
        text.push_str(&format!(
            " ({} messages dropped by the template)",
            l.dropped_messages
        ));
    }
    text
}

/// The date in an Artemis day file's name: `artemis-YYYY-MM-DD.jsonl` exactly, and a date that exists.
fn day_file_date(name: &str) -> Option<i64> {
    parse_date(name.strip_prefix("artemis-")?.strip_suffix(".jsonl")?)
}

/// Removes the day files in `dir` dated `keep_days` days before `today` (`YYYY-MM-DD`) or earlier; with 14 on
/// 2026-11-14 the files from 2026-11-01 on stay. The unit is the calendar day (UTC): with 1 only today's file stays,
/// so yesterday's goes as the day changes. Only a name of exactly the form `artemis-YYYY-MM-DD.jsonl` with a
/// valid date is touched, so a shared folder's other files, a `.bak` copy and `artemis-latest.jsonl` stay. 0 keeps every
/// file. A file that cannot be removed (open elsewhere, no right) is left; nothing here fails a request. Returns the
/// files it removed.
pub fn prune(dir: &Path, today: &str, keep_days: u32) -> Vec<PathBuf> {
    let (Some(today), true) = (parse_date(today), keep_days > 0) else {
        return Vec::new();
    };
    let cutoff = today - i64::from(keep_days);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut removed = Vec::new();
    for entry in entries.flatten() {
        let old = entry
            .file_name()
            .to_str()
            .and_then(day_file_date)
            .is_some_and(|d| d <= cutoff);
        let is_file = entry.file_type().is_ok_and(|t| t.is_file());
        if old && is_file && std::fs::remove_file(entry.path()).is_ok() {
            removed.push(entry.path());
        }
    }
    removed
}

fn level_filter(level: &str) -> Result<tracing::level_filters::LevelFilter, String> {
    level
        .trim()
        .parse()
        .map_err(|_| format!("log_level {level:?} is not error, warn, info, debug or trace"))
}

/// Sends `tracing` to stderr as text at `level` (`error`, `warn`, `info`, `debug` or `trace`, any case); the level
/// comes from the config and nothing else, no environment filter. Errors if the level is unknown or a subscriber is
/// already installed.
pub fn init_tracing(level: &str) -> Result<(), String> {
    use std::io::IsTerminal;
    let level = level_filter(level)?;
    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .try_init()
        .map_err(|e| format!("tracing is already set up: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::SpecReport;
    use crate::error::ErrorType;
    use serde_json::Value;

    fn day_number(date: &str) -> i64 {
        crate::ids::parse_date(date).unwrap()
    }

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("artemis-logs-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn a_load_stall_says_how_to_raise_the_bound() {
        let stall = |doing: &str| LifecycleLine::Stall {
            doing: doing.into(),
            request: Some("req-1".into()),
            quiet_s: 300.4,
            bound_s: 300,
            ticks: 7,
        };
        for doing in ["load", "reload"] {
            assert_eq!(
                stall(doing).human(),
                "stalled: the model load made no progress for 300 s (bound 300 s); if the model file is on a slow disk, raise stall_load_s"
            );
        }
        // The other stalls keep their text.
        assert_eq!(
            stall("request").human(),
            "stalled: request req-1 made no progress for 300 s (bound 300 s)"
        );
        assert!(!stall("command").human().contains("stall_load_s"));
    }

    #[test]
    fn a_request_line_has_every_field_of_the_spec() {
        let d = dir("request");
        let logger = Logger::new(d.clone(), 0).unwrap();
        let mut l = RequestLine::new("chatcmpl-x", "/v1/chat/completions");
        l.time = "2026-10-18T23:59:59Z".into();
        l.status = 401;
        logger.request(&l);
        l.time = "2026-10-19T00:00:00Z".into();
        l.cause = Some(Cause::NewInput);
        l.ignored = vec!["metadata".into()];
        logger.request(&l);
        let first = std::fs::read_to_string(d.join("artemis-2026-10-18.jsonl")).unwrap();
        let second = std::fs::read_to_string(d.join("artemis-2026-10-19.jsonl")).unwrap();
        assert_eq!((first.lines().count(), second.lines().count()), (1, 1));
        let a: Value = serde_json::from_str(first.trim_end()).unwrap();
        let keys: Vec<&str> = a.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "time",
                "id",
                "endpoint",
                "peer",
                "model",
                "status",
                "prompt_tokens",
                "cut_at",
                "user_query",
                "stream",
                "tool_calls",
                "malformed_calls",
                "think_budget",
                "budget_closed",
                "reused_tokens",
                "restored",
                "restore_bytes",
                "restore_ms",
                "diverged_at",
                "cause",
                "prefilled_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "queue_ms",
                "load_ms",
                "prefill_ms",
                "ttft_ms",
                "total_ms",
                "prefill_tok_s",
                "decode_tok_s",
                "decode_ms",
                "decode_tokens",
                "decode_misses",
                "hit_rate",
                "miss_mib_per_token",
                "spec_steps",
                "spec_plain_steps",
                "spec_drafted",
                "spec_accepted",
                "spec_accept_rate",
                "spec_replays",
                "spec_drafted_lookup",
                "spec_accepted_lookup",
                "spec_drafted_mtp",
                "spec_accepted_mtp",
                "pool_bytes_used",
                "finish",
                "error_kind",
                "error_message",
                "ignored",
                "ignored_more",
                "repairs",
                "repairs_more",
                "dropped_messages"
            ]
        );
        assert_eq!(a["status"], 401);
        assert!(a["prompt_tokens"].is_null() && a["hit_rate"].is_null());
        let b: Value = serde_json::from_str(second.trim_end()).unwrap();
        assert_eq!(b["cause"], "new_input");
        assert_eq!(b["ignored"], serde_json::json!(["metadata"]));
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn repairs_past_16_are_counted() {
        let repairs: Vec<String> = (0..20).map(|i| format!("repair {i}")).collect();
        let mut l = RequestLine::new("a", "/v1/x");
        l.note_repairs(&repairs);
        assert_eq!(l.repairs.len(), REPAIRS_SHOWN);
        assert_eq!(l.repairs[15], "repair 15");
        assert_eq!(l.repairs_more, 4);
        l.note_repairs(&repairs[..16]);
        assert_eq!((l.repairs.len(), l.repairs_more), (16, 0));
        l.note_repairs(&[]);
        assert_eq!((l.repairs.len(), l.repairs_more), (0, 0));
    }

    #[test]
    fn a_model_is_cut_to_64_characters() {
        assert_eq!(MODEL_CHARS, 64);
        assert_eq!(cut_model(&"é".repeat(100)).chars().count(), 64);
        assert_eq!(cut_model("qwen"), "qwen");
    }

    #[test]
    fn the_stderr_line_counts_what_was_ignored_and_dropped() {
        let mut l = RequestLine::new("chatcmpl-x", "/v1/chat/completions");
        l.status = 200;
        l.finish = Some("stop".into());
        assert_eq!(
            stderr_line(&l),
            "chatcmpl-x /v1/chat/completions -> 200 stop"
        );
        // No finish reason: no trailing or doubled space.
        l.finish = None;
        assert_eq!(stderr_line(&l), "chatcmpl-x /v1/chat/completions -> 200");
        l.ignored = vec!["a".into()];
        assert_eq!(
            stderr_line(&l),
            "chatcmpl-x /v1/chat/completions -> 200 (1 fields ignored)"
        );
        l.finish = Some("stop".into());
        l.ignored = vec!["a".into(), "b".into()];
        l.ignored_more = 8;
        assert_eq!(
            stderr_line(&l),
            "chatcmpl-x /v1/chat/completions -> 200 stop (10 fields ignored)"
        );
        l.dropped_messages = 2;
        assert_eq!(
            stderr_line(&l),
            "chatcmpl-x /v1/chat/completions -> 200 stop (10 fields ignored) (2 messages dropped by the template)"
        );
        l.ignored.clear();
        l.ignored_more = 0;
        assert_eq!(
            stderr_line(&l),
            "chatcmpl-x /v1/chat/completions -> 200 stop (2 messages dropped by the template)"
        );
    }

    #[test]
    fn the_setup_and_tuned_lines_are_typed() {
        let d = dir("setup-lines");
        let logger = Logger::new(d.clone(), 0).unwrap();
        let done_with = |expected_tok_s: f64, enabled: bool| LifecycleLine::SetupDone {
            seconds: 187.5,
            limit_hit: false,
            target: true,
            gpu_name: "AMD Radeon RX 9070 XT".into(),
            vram_bytes: 17_095_000_000,
            cache_slots: 112,
            scores: BTreeMap::from([("lookup+mtp".to_string(), 137.0)]),
            enabled,
            sources: vec!["lookup".into(), "mtp".into()],
            priors: Some(SetupPriors {
                cost_slope_ms: 4.5,
                mtp_cost_ms: 1.5,
            }),
            measured_slope_ms: Some(4.5),
            measured_mtp_d1_ms: None,
            expected_tok_s,
            path: "/state/tuned.toml".into(),
        };
        let done = done_with(137.0, true);
        let lines = [
            LifecycleLine::SetupStart {
                reason: "first run".into(),
                path: "/state/tuned.toml".into(),
            },
            done.clone(),
            LifecycleLine::SetupFailed {
                message: "cannot write /state/tuned.toml: disk full".into(),
            },
            LifecycleLine::Tuned {
                path: "/state/tuned.toml".into(),
                applied: vec!["speculation.enabled".into()],
                ignored: vec!["device_uuid".into()],
            },
        ];
        for l in &lines {
            logger.lifecycle(l);
        }
        let file = std::fs::read_dir(&d)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let got: Vec<Value> = std::fs::read_to_string(file)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let events: Vec<&str> = got.iter().map(|l| l["event"].as_str().unwrap()).collect();
        assert_eq!(
            events,
            ["setup_start", "setup_done", "setup_failed", "tuned"]
        );
        assert_eq!(got[0]["reason"], "first run");
        assert_eq!(got[1]["scores"]["lookup+mtp"], 137.0);
        assert_eq!(got[1]["priors"]["mtp_cost_ms"], 1.5);
        assert!(got[1]["measured_mtp_d1_ms"].is_null());
        assert_eq!(got[1]["sources"][1], "mtp");
        assert_eq!(got[3]["ignored"][0], "device_uuid");
        for l in &lines {
            let h = l.human();
            assert!(!h.contains('\u{2014}') && !h.contains('\u{2013}'), "{h}");
        }
        assert_eq!(
            done.human(),
            "setup done in 188 s on AMD Radeon RX 9070 XT: speculation on with lookup and mtp, about 137 tok/s; \
             wrote /state/tuned.toml"
        );
        // Nothing measured: no speed is claimed.
        let none = done_with(0.0, false);
        assert!(!none.human().contains("tok/s"), "{}", none.human());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn the_start_line_names_the_log_directory() {
        let start = LifecycleLine::Start {
            version: "0.1.0".into(),
            bind: "127.0.0.1:8080".into(),
            model_id: "m".into(),
            log_dir: "/var/log/artemis".into(),
        };
        assert_eq!(
            start.human(),
            "artemis 0.1.0 listening on 127.0.0.1:8080, model m, logs in /var/log/artemis"
        );
        assert!(start.human().contains("logs in "));
        let d = dir("start-dir");
        let logger = Logger::new(d.clone(), 0).unwrap();
        assert!(logger.dir().is_absolute());
        assert!(logger.dir().ends_with(d.file_name().unwrap()));
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn prune_removes_only_old_artemis_day_files() {
        let d = dir("prune");
        std::fs::create_dir_all(&d).unwrap();
        let names = [
            "artemis-2026-10-01.jsonl",
            "artemis-2026-10-31.jsonl",
            "artemis-2026-11-01.jsonl",
            "artemis-2026-11-14.jsonl",
            "artemis-2026-10-01.jsonl.bak",
            "artemis-latest.jsonl",
            "artemis-2026-13-01.jsonl",
            "notes.txt",
        ];
        for n in names {
            std::fs::write(d.join(n), b"x").unwrap();
        }
        // A directory with a day file's name is not a file to remove.
        std::fs::create_dir(d.join("artemis-2026-09-01.jsonl")).unwrap();
        // 0 keeps every file.
        assert!(prune(&d, "2026-11-14", 0).is_empty());
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), names.len() + 1);
        let mut removed: Vec<String> = prune(&d, "2026-11-14", 14)
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        removed.sort();
        assert_eq!(
            removed,
            ["artemis-2026-10-01.jsonl", "artemis-2026-10-31.jsonl"]
        );
        let mut left: Vec<String> = std::fs::read_dir(&d)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        let mut want: Vec<&str> = names[2..].to_vec();
        want.push("artemis-2026-09-01.jsonl");
        want.sort_unstable();
        assert_eq!(left, want);
        // A bad "today" removes nothing.
        assert!(prune(&d, "not a date", 14).is_empty());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_logger_prunes_at_start_and_when_a_new_days_file_opens() {
        let d = dir("prune-logger");
        std::fs::create_dir_all(&d).unwrap();
        let today = utc_date(unix_now());
        let ago = |days: i64| {
            let t = day_number(&today) - days;
            utc_date(t as u64 * 86_400)
        };
        let old = format!("artemis-{}.jsonl", ago(30));
        let recent = format!("artemis-{}.jsonl", ago(3));
        std::fs::write(d.join(&old), b"x").unwrap();
        std::fs::write(d.join(&recent), b"x").unwrap();
        let logger = Logger::new(d.clone(), 14).unwrap();
        assert!(!d.join(&old).exists() && d.join(&recent).exists());
        // A file that ages out while the server runs goes when the log moves to another day's file.
        let stale = format!("artemis-{}.jsonl", ago(40));
        std::fs::write(d.join(&stale), b"x").unwrap();
        let mut l = RequestLine::new("a", "/v1/x");
        logger.request(&l);
        assert!(!d.join(&stale).exists());
        l.time = format!("{}T00:00:00Z", ago(1));
        std::fs::write(d.join(&stale), b"x").unwrap();
        logger.request(&l);
        assert!(!d.join(&stale).exists(), "a new date's file prunes");
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn total_ms_is_the_time_from_the_line_being_made_to_its_writing() {
        let d = dir("total");
        let logger = Logger::new(d.clone(), 0).unwrap();
        let mut l = RequestLine::new("a", "/v1/x");
        // A request that began 1.5 s ago: the line's `time` is its arrival, `total_ms` how long it ran.
        l.started = Instant::now() - Duration::from_millis(1500);
        logger.request(&l);
        // A figure already set is kept.
        let mut set = RequestLine::new("b", "/v1/x");
        set.total_ms = Some(7.0);
        logger.request(&set);
        // A line dropped unfinished still gets one.
        drop(LineGuard::new(&logger, RequestLine::new("c", "/v1/x")));
        let file = std::fs::read_dir(&d)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let text = std::fs::read_to_string(file).unwrap();
        let lines: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let total = |i: usize| lines[i]["total_ms"].as_f64().unwrap();
        assert!((1500.0..5000.0).contains(&total(0)), "{}", total(0));
        assert_eq!(total(1), 7.0);
        assert!((0.0..1000.0).contains(&total(2)), "{}", total(2));
        // The caller's line is not changed by the writing.
        assert_eq!(l.total_ms, None);
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_lifecycle_line_names_its_event() {
        let d = dir("lifecycle");
        let logger = Logger::new(d.clone(), 0).unwrap();
        logger.lifecycle(&LifecycleLine::Unload {
            cause: "idle".into(),
        });
        logger.clone().lifecycle(&LifecycleLine::Exit {
            reason: "signal".into(),
        });
        let file = std::fs::read_dir(&d)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let text = std::fs::read_to_string(file).unwrap();
        let lines: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(
            (lines[0]["event"].as_str(), lines[0]["cause"].as_str()),
            (Some("unload"), Some("idle"))
        );
        assert_eq!(lines[1]["event"], "exit");
        assert!(lines[1]["time"].as_str().unwrap().ends_with('Z'));
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn the_load_line_carries_its_figures() {
        let d = dir("load");
        let logger = Logger::new(d.clone(), 0).unwrap();
        logger.lifecycle(&LifecycleLine::Load {
            seconds: 12.5,
            detail: "device".into(),
            slots_per_layer: Some(64),
            vram_used_mib: Some(13_312.5),
            pool_limit_gib: Some(2.0),
            pool_cut: Some(true),
            unparked: Some(40),
        });
        logger.lifecycle(&LifecycleLine::Load {
            seconds: 0.1,
            detail: "fake engine".into(),
            slots_per_layer: None,
            vram_used_mib: None,
            pool_limit_gib: None,
            pool_cut: None,
            unparked: None,
        });
        let file = std::fs::read_dir(&d)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let text = std::fs::read_to_string(file).unwrap();
        let lines: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let figures = [
            "slots_per_layer",
            "vram_used_mib",
            "pool_limit_gib",
            "pool_cut",
            "unparked",
        ];
        for l in &lines {
            let keys: Vec<&str> = l.as_object().unwrap().keys().map(String::as_str).collect();
            assert_eq!(
                keys,
                [&["time", "event", "seconds", "detail"][..], &figures[..]].concat()
            );
        }
        assert_eq!(lines[0]["slots_per_layer"], 64);
        assert_eq!(lines[0]["vram_used_mib"], 13_312.5);
        assert_eq!(lines[0]["pool_limit_gib"], 2.0);
        assert_eq!(lines[0]["pool_cut"], true);
        assert_eq!(lines[0]["unparked"], 40);
        assert!(figures.iter().all(|k| lines[1][*k].is_null()));
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn the_token_rates_are_null_without_tokens_or_time() {
        let counts = |prefilled, completion| Counts {
            prompt: 10,
            reused: 0,
            prefilled,
            completion,
            reasoning: 0,
        };
        let timings = |prefill_ms, decode_ms| Timings {
            queue_ms: 1.0,
            load_ms: 0.0,
            restore_ms: 2.0,
            prefill_ms,
            decode_ms,
            ttft_ms: 3.0,
        };
        let cache = CacheReport {
            hit_rate: None,
            miss_mib_per_token: None,
            restore_bytes: 7,
            pool_bytes_used: 9,
            decode_misses: Some(3),
        };
        let mut l = RequestLine::new("x", "/v1/x");
        l.measured(&counts(10, 5), &timings(500.0, 250.0), &cache, None);
        assert_eq!((l.prefill_tok_s, l.decode_tok_s), (Some(20.0), Some(20.0)));
        // The decode figures: the time as measured, the tokens after the first, the cache's misses.
        assert_eq!(
            (l.decode_ms, l.decode_tokens, l.decode_misses),
            (Some(250.0), Some(4), Some(3))
        );
        assert_eq!((l.hit_rate, l.miss_mib_per_token), (None, None));
        assert_eq!((l.restore_bytes, l.pool_bytes_used), (Some(7), Some(9)));
        l.measured(&counts(0, 5), &timings(500.0, 0.0), &cache, None);
        assert_eq!((l.prefill_tok_s, l.decode_tok_s), (None, None));
        // No completion at all is no decoded token, not an underflow.
        l.measured(&counts(0, 0), &timings(500.0, 0.0), &cache, None);
        assert_eq!((l.decode_ms, l.decode_tokens), (Some(0.0), Some(0)));
    }

    #[test]
    fn the_spec_counts_are_the_reports_and_the_rate_is_accepted_over_drafted() {
        let counts = Counts {
            prompt: 10,
            reused: 0,
            prefilled: 10,
            completion: 5,
            reasoning: 0,
        };
        let timings = Timings {
            queue_ms: 1.0,
            load_ms: 0.0,
            restore_ms: 0.0,
            prefill_ms: 5.0,
            decode_ms: 50.0,
            ttft_ms: 6.0,
        };
        let cache = CacheReport {
            hit_rate: None,
            miss_mib_per_token: None,
            restore_bytes: 0,
            pool_bytes_used: 0,
            decode_misses: None,
        };
        let spec = |drafted, accepted| SpecReport {
            steps: 5,
            plain_steps: 2,
            drafted,
            accepted,
            replays: 3,
            drafted_lookup: drafted - drafted / 3,
            accepted_lookup: accepted / 2,
            drafted_mtp: drafted / 3,
            accepted_mtp: accepted - accepted / 2,
        };
        let mut l = RequestLine::new("x", "/v1/x");
        l.measured(&counts, &timings, &cache, Some(&spec(12, 9)));
        assert_eq!(
            (l.spec_steps, l.spec_plain_steps, l.spec_drafted),
            (Some(5), Some(2), Some(12))
        );
        assert_eq!((l.spec_accepted, l.spec_replays), (Some(9), Some(3)));
        assert_eq!(l.spec_accept_rate, Some(0.75));
        assert_eq!(
            (l.spec_drafted_lookup, l.spec_accepted_lookup),
            (Some(8), Some(4))
        );
        assert_eq!(
            (l.spec_drafted_mtp, l.spec_accepted_mtp),
            (Some(4), Some(5))
        );
        // Nothing drafted is no rate, not a division by zero.
        l.measured(&counts, &timings, &cache, Some(&spec(0, 0)));
        assert_eq!((l.spec_drafted, l.spec_accept_rate), (Some(0), None));
        // No report (speculation off, or the request never decoded): all six are null again.
        l.measured(&counts, &timings, &cache, None);
        assert_eq!(
            (l.spec_steps, l.spec_plain_steps, l.spec_drafted),
            (None, None, None)
        );
        assert_eq!(
            (l.spec_accepted, l.spec_accept_rate, l.spec_replays),
            (None, None, None)
        );
        assert_eq!(
            (
                l.spec_drafted_lookup,
                l.spec_accepted_lookup,
                l.spec_drafted_mtp,
                l.spec_accepted_mtp
            ),
            (None, None, None, None)
        );
    }

    #[test]
    fn a_stats_line_carries_the_copy_and_its_age_or_nulls() {
        let d = dir("stats");
        let logger = Logger::new(d.clone(), 0).unwrap();
        let copy = StatsCopy {
            stats: BackendStats {
                loads: 2,
                spec: Some(crate::backend::SpecCounters {
                    steps: 7,
                    drafted: 28,
                    accept_rate: [("lookup".to_string(), 0.5)].into(),
                    by_size: vec![crate::backend::StepSize {
                        t: 5,
                        steps: 7,
                        ..crate::backend::StepSize::default()
                    }],
                    ..crate::backend::SpecCounters::default()
                }),
                ..BackendStats::default()
            },
            at: std::time::Instant::now(),
        };
        logger.lifecycle(&stats_line(Some(&copy), 600, 3));
        logger.lifecycle(&stats_line(None, 601, 0));
        let text: String = std::fs::read_dir(&d)
            .unwrap()
            .map(|e| std::fs::read_to_string(e.unwrap().path()).unwrap())
            .collect();
        let lines: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let keys = |v: &Value| -> Vec<String> { v.as_object().unwrap().keys().cloned().collect() };
        let want = [
            "time",
            "event",
            "uptime_s",
            "age_s",
            "log_write_failures",
            "stats",
        ];
        assert_eq!(keys(&lines[0]).len(), want.len(), "{}", lines[0]);
        for k in want {
            assert!(lines[0].get(k).is_some(), "{k}: {}", lines[0]);
        }
        assert_eq!(lines[0]["event"], "stats");
        assert_eq!(
            (&lines[0]["uptime_s"], &lines[0]["log_write_failures"]),
            (&Value::from(600), &Value::from(3))
        );
        assert_eq!(lines[0]["stats"]["loads"], 2);
        // The heartbeat carries the speculation block as `/v1/stats` does.
        assert_eq!(lines[0]["stats"]["spec"]["steps"], 7);
        assert_eq!(lines[0]["stats"]["spec"]["accept_rate"]["lookup"], 0.5);
        assert_eq!(lines[0]["stats"]["spec"]["by_size"][0]["t"], 5);
        assert!(lines[0]["age_s"].as_f64().unwrap() < 5.0);
        assert!(
            lines[1]["stats"].is_null() && lines[1]["age_s"].is_null(),
            "{}",
            lines[1]
        );
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn error_message_is_cut_to_300_characters_on_a_char_boundary() {
        let error = |m: String| ApiError::new(400, ErrorType::InvalidRequest, m);
        let mut l = RequestLine::new("x", "/v1/x");
        // 400 ASCII characters are cut to 300.
        l.set_error(&error("a".repeat(400)), Flavour::OpenAi);
        assert_eq!(l.error_kind.as_deref(), Some("invalid_request_error"));
        assert_eq!(l.error_message, Some("a".repeat(ERROR_MESSAGE_CHARS)));
        // 400 two-byte characters: `ApiError` keeps 500 bytes, 250 characters, and the line keeps them whole.
        l.set_error(&error("é".repeat(400)), Flavour::Anthropic);
        assert_eq!(l.error_message, Some("é".repeat(250)));
        // 400 three-byte characters in a message `ApiError` has not capped (its fields are public): the first 300 stay.
        let mut e = error(String::new());
        e.message = "€".repeat(400);
        l.set_error(&e, Flavour::OpenAi);
        let m = l.error_message.as_deref().unwrap();
        assert_eq!(m.chars().count(), ERROR_MESSAGE_CHARS);
        assert!(m.chars().all(|c| c == '€'));
        // A short message is kept whole; the kind is the flavour's name.
        l.set_error(
            &ApiError::new(503, ErrorType::Unavailable, "the queue is full"),
            Flavour::Anthropic,
        );
        assert_eq!(l.error_kind.as_deref(), Some("overloaded_error"));
        assert_eq!(l.error_message.as_deref(), Some("the queue is full"));
        let json = serde_json::to_value(&l).unwrap();
        assert_eq!(json["error_message"], "the queue is full");
    }

    #[test]
    fn a_guard_dropped_with_its_line_writes_it_as_cancelled() {
        let d = dir("guard");
        let logger = Logger::new(d.clone(), 0).unwrap();
        let mut guard = LineGuard::new(&logger, RequestLine::new("a", "/v1/x"));
        guard.line().prompt_tokens = Some(12);
        drop(guard);
        // A taken line is its taker's to write.
        let guard = LineGuard::new(&logger, RequestLine::new("b", "/v1/x"));
        let _line = guard.take();
        let file = std::fs::read_dir(&d)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let text = std::fs::read_to_string(file).unwrap();
        let lines: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 1);
        assert_eq!(
            (
                lines[0]["id"].as_str(),
                lines[0]["status"].as_u64(),
                lines[0]["finish"].as_str()
            ),
            (Some("a"), Some(499), Some("cancelled"))
        );
        assert_eq!(lines[0]["prompt_tokens"], 12);
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn failed_writes_are_counted() {
        let d = dir("failed");
        let logger = Logger::new(d.clone(), 0).unwrap();
        assert_eq!(logger.write_failures(), 0);
        // Nothing is open yet, so the directory goes (Windows too) and a plain file takes its name: every open fails.
        std::fs::remove_dir(&d).unwrap();
        std::fs::write(&d, b"not a directory").unwrap();
        let mut l = RequestLine::new("a", "/v1/x");
        l.time = "2026-10-18T12:00:00Z".into();
        for _ in 0..3 {
            logger.request(&l);
        }
        assert_eq!(logger.write_failures(), 3);
        // The count is the logger's, shared by its clones.
        logger.clone().lifecycle(&LifecycleLine::Exit {
            reason: "shutdown".into(),
        });
        assert_eq!(logger.write_failures(), 4);
        std::fs::remove_file(d).unwrap();
    }

    #[test]
    fn a_line_written_within_a_bound_does_not_wait_on_a_held_lock() {
        let d = dir("held");
        let logger = Logger::new(d.clone(), 0).unwrap();
        let line = LifecycleLine::Exit {
            reason: "stall".into(),
        };
        let held = logger.hold_the_lock();
        let t = std::time::Instant::now();
        assert!(!logger.lifecycle_within(&line, Duration::from_millis(100)));
        assert!(t.elapsed() < Duration::from_secs(1), "{:?}", t.elapsed());
        drop(held);
        // The late line is still written once the lock is free.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let written = loop {
            let text = std::fs::read_dir(&d)
                .unwrap()
                .filter_map(|e| std::fs::read_to_string(e.unwrap().path()).ok())
                .collect::<String>();
            if !text.is_empty() || std::time::Instant::now() > deadline {
                break text;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(!written.is_empty(), "the late line was never written");
        let v: Value = serde_json::from_str(written.trim_end()).unwrap();
        assert_eq!(
            (v["event"].as_str(), v["reason"].as_str()),
            (Some("exit"), Some("stall"))
        );
        // With the lock free the line is written in time.
        assert!(logger.lifecycle_within(&line, Duration::from_secs(10)));
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn the_load_failure_lines_carry_the_message() {
        let d = dir("loadfail");
        let logger = Logger::new(d.clone(), 0).unwrap();
        logger.lifecycle(&LifecycleLine::LoadFailed {
            message: "no room".into(),
        });
        logger.lifecycle(&LifecycleLine::ReloadFailed {
            message: "still no room".into(),
        });
        let text = std::fs::read_to_string(
            std::fs::read_dir(&d)
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path(),
        )
        .unwrap();
        let lines: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(
            (lines[0]["event"].as_str(), lines[0]["message"].as_str()),
            (Some("load_failed"), Some("no room"))
        );
        assert_eq!(
            (lines[1]["event"].as_str(), lines[1]["message"].as_str()),
            (Some("reload_failed"), Some("still no room"))
        );
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn the_log_level_is_one_of_tracings() {
        for ok in ["error", "WARN", " info ", "Debug", "trace", "off"] {
            assert!(level_filter(ok).is_ok(), "{ok}");
        }
        assert!(init_tracing("loud").unwrap_err().contains("log_level"));
    }
}
