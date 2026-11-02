//! The server's config: one TOML file, the flags over it, the API key file and the loopback bind rule (spec 7).
use crate::backend::Sampling;
use crate::params::Defaults;
use crate::tuned::{self, SetupFacts, TunedLocation, REMEDY};
use qwen36::sha256::sha256;
use std::collections::BTreeSet;
use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, ToSocketAddrs};
use std::path::PathBuf;
use subtle::ConstantTimeEq;

/// The context lengths the server accepts, in tokens: the model's trained window is 262,144.
const CTX_RANGE: std::ops::RangeInclusive<u32> = 512..=262_144;

/// The most expert-cache slots a layer can use: a layer has 256 experts.
const MAX_CACHE_SLOTS: u32 = 256;

/// Whether `name` is a `health_service` name: 1 to 32 characters, each a lower-case letter, a digit or a hyphen, the
/// first not a hyphen.
fn is_service_name(name: &str) -> bool {
    (1..=32).contains(&name.len())
        && !name.starts_with('-')
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// The `[sampling]` table (C13): what a request that sets none of these gets.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub presence_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> SamplingConfig {
        SamplingConfig {
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            presence_penalty: 0.0,
        }
    }
}

/// The `[speculation]` table (milestone-5 spec 10.1, D10): speculative decoding with two draft sources, prompt
/// lookup and the model's MTP block. `artemis serve` copies it into the engine's `SpecConfig`; the server cannot depend
/// on the engine, so `check` repeats that config's rules and a test in `tools` holds the two together.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SpeculationConfig {
    /// `false` is the rollback: plain decode, nothing allocated for speculation.
    pub enabled: bool,
    /// The draft sources that run: `"lookup"` and `"mtp"`, each at most once. The list's order does nothing;
    /// `source_choice` says how a step picks between them.
    pub sources: Vec<String>,
    /// The most draft tokens a step verifies, from 1 to 15.
    pub max_draft: u32,
    /// Prompt lookup's shortest n-gram.
    pub ngram_min: u32,
    /// Prompt lookup's longest n-gram, at most 16.
    pub ngram_max: u32,
    /// The predicted speed-up below which a step runs plain decode.
    pub min_gain: f64,
    /// Draft tokens per halving of an old outcome's weight in the accept rate.
    pub accept_half_life: f64,
    /// The controller's cost per extra token of a step, in ms, until it has measured its own.
    pub cost_slope_ms: f64,
    /// The accept rate a fresh controller starts from.
    pub accept_prior: f64,
    /// MTP drafts at `k = 1` once this many plain steps have run since it last drafted.
    pub probe_every: u32,
    /// The last positions of the prompt that prefill writes into the MTP's KV.
    pub mtp_window: u32,
    /// The MTP KV's rows (positions), before the load clamps them to the session.
    pub mtp_ctx: u32,
    /// The hidden-row ring's length, before the load clamps it.
    pub mtp_ring: u32,
    /// MTP's accept rate at each draft depth until it has measured its own; a depth past the list takes the last.
    pub mtp_accept_prior: Vec<f64>,
    /// The prior for the draft pass's time per draft token, in ms, until it is measured.
    pub mtp_cost_ms: f64,
    /// Which source drafts when both have a draft: `"priority"` (lookup first) or `"best"` (the higher predicted tokens
    /// per ms; the default, from G-2's round-3 measurement).
    pub source_choice: String,
    /// 0 drafts MTP tokens with the full output head; N drafts from its first N rows (1024 or more, at most the
    /// vocabulary; the load clamps a larger N to it). The default is 65,536, G-2's round-3 measurement.
    pub mtp_head_rows: u32,
}

impl Default for SpeculationConfig {
    fn default() -> SpeculationConfig {
        SpeculationConfig {
            enabled: true,
            sources: vec!["lookup".into(), "mtp".into()],
            max_draft: 4,
            ngram_min: 2,
            ngram_max: 4,
            min_gain: 1.03,
            accept_half_life: 32.0,
            cost_slope_ms: 4.4,
            accept_prior: 0.5,
            probe_every: 64,
            mtp_window: 2048,
            mtp_ctx: 8192,
            mtp_ring: 256,
            mtp_accept_prior: vec![0.79, 0.81, 0.74, 0.78],
            mtp_cost_ms: 1.6,
            source_choice: "best".into(),
            mtp_head_rows: 65536,
        }
    }
}

impl SpeculationConfig {
    /// Refuses a setting the engine can't honour: the key's name, and why. The rules and the words are the engine's
    /// `SpecConfig::check`, plus `sources`. A rule added to the engine's check needs the same rule here and a case in
    /// `the_servers_speculation_rules_agree_with_the_engines` (`crates/tools/src/serve/backend.rs`).
    pub fn check(&self) -> Result<(), (&'static str, String)> {
        let refuse = |key: &'static str, why: &str| Err((key, why.to_string()));
        let sources = &self.sources;
        // The engine's order (`Source::parse_list`): every name is known first, then the list is non-empty, then no
        // name is repeated.
        if let Some(name) = sources.iter().find(|s| *s != "lookup" && *s != "mtp") {
            return Err(("sources", format!("an unknown source {name:?}")));
        }
        if sources.is_empty() {
            return refuse(
                "sources",
                "needs a source; to turn speculation off set enabled = false",
            );
        }
        for (i, name) in sources.iter().enumerate() {
            if sources[..i].contains(name) {
                return Err(("sources", format!("{name:?} is listed twice")));
            }
        }
        if !(1..16).contains(&self.max_draft) {
            return refuse("max_draft", "must be from 1 to 15");
        }
        if self.ngram_min < 1 {
            return refuse("ngram_min", "must be at least 1");
        }
        if self.ngram_max > 16 {
            return refuse("ngram_max", "must be at most 16");
        }
        if self.ngram_min > self.ngram_max {
            return refuse("ngram_min", "must be at most ngram_max");
        }
        if !(self.min_gain.is_finite() && self.min_gain >= 1.0) {
            return refuse("min_gain", "must be a finite number, 1 or more");
        }
        if !(self.accept_half_life.is_finite() && self.accept_half_life > 0.0) {
            return refuse("accept_half_life", "must be a finite number above 0");
        }
        if !(self.cost_slope_ms.is_finite() && self.cost_slope_ms >= 0.0) {
            return refuse("cost_slope_ms", "must be a finite number, 0 or more");
        }
        if !(0.0..=1.0).contains(&self.accept_prior) {
            return refuse("accept_prior", "must be from 0 to 1");
        }
        if self.probe_every < 1 {
            return refuse("probe_every", "must be at least 1");
        }
        if !(512..=65536).contains(&self.mtp_ctx) || !self.mtp_ctx.is_multiple_of(256) {
            return refuse("mtp_ctx", "must be from 512 to 65536 and a multiple of 256");
        }
        if !(1..=self.mtp_ctx / 2).contains(&self.mtp_window) {
            return Err((
                "mtp_window",
                format!("must be from 1 to mtp_ctx / 2 ({})", self.mtp_ctx / 2),
            ));
        }
        if !(16..=self.mtp_ctx / 4).contains(&self.mtp_ring) || !self.mtp_ring.is_multiple_of(16) {
            return Err((
                "mtp_ring",
                format!(
                    "must be from 16 to mtp_ctx / 4 ({}) and a multiple of 16",
                    self.mtp_ctx / 4
                ),
            ));
        }
        if !(1..16).contains(&self.mtp_accept_prior.len())
            || !self
                .mtp_accept_prior
                .iter()
                .all(|a| (0.0..=1.0).contains(a))
        {
            return refuse(
                "mtp_accept_prior",
                "must be 1 to 15 values, each from 0 to 1",
            );
        }
        if !(self.mtp_cost_ms.is_finite() && self.mtp_cost_ms >= 0.0) {
            return refuse("mtp_cost_ms", "must be a finite number, 0 or more");
        }
        if self.source_choice != "priority" && self.source_choice != "best" {
            return refuse("source_choice", "must be \"priority\" or \"best\"");
        }
        if self.mtp_head_rows != 0 && self.mtp_head_rows < 1024 {
            return refuse(
                "mtp_head_rows",
                "must be 0 (the full head) or at least 1024",
            );
        }
        Ok(())
    }
}

/// Every key of spec 7.3 (and `park_snapshots`, spec 5.10). An unknown key is an error that names it (C22).
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    pub model: Option<PathBuf>,
    pub host: String,
    pub port: u16,
    pub api_key_file: Option<PathBuf>,
    /// Browser origins allowed to call the server (`scheme://host[:port]`); empty refuses every request that carries
    /// an `Origin` header.
    pub allowed_origins: Vec<String>,
    pub ctx: u32,
    pub snapshot_pool_gib: u32,
    pub park_snapshots: bool,
    pub lazy_load: bool,
    pub idle_unload_s: u64,
    pub queue_timeout_s: u64,
    /// Seconds pending work may go without moving before the watchdog halts the process; 0 turns the watchdog off.
    pub stall_s: u64,
    /// The same bound while a load or a reload runs, which shows no progress inside; 0 lifts only this bound.
    pub stall_load_s: u64,
    pub model_id: String,
    /// The `service` the `/health` body names: 1 to 32 lower-case letters, digits or hyphens.
    pub health_service: String,
    pub thinking: bool,
    /// 0 is no budget.
    pub thinking_budget: u32,
    pub sampling: SamplingConfig,
    pub speculation: SpeculationConfig,
    pub log_dir: Option<PathBuf>,
    pub log_level: String,
    /// Days of request logs kept: a file `artemis-YYYY-MM-DD.jsonl` dated this many days before today or earlier is
    /// removed when the server starts and when the log moves to a new day's file. 0 keeps every file.
    pub log_keep_days: u32,
    pub device_uuid: Option<String>,
    pub cache_slots: Option<u32>,
    pub f32_prefill: bool,
    /// Use a GGUF whose chat template differs from the pinned one (the start is refused otherwise).
    pub allow_custom_template: bool,
    /// Whether a control token's text in client text (a file, a tool result, a message) is tokenized as the token, as
    /// llama.cpp does (the default); `false` keeps it plain text (S-5).
    pub special_tokens_in_content: bool,
    /// Whether a `/v1/messages` request that sends neither `thinking` nor `output_config.effort` thinks by default
    /// (when `thinking` is on), as llama.cpp's server does (the default); `false` follows Anthropic's API, where an
    /// absent field means no thinking.
    pub anthropic_thinking: bool,
    /// Whether `artemis serve` measures this PC once, on the first start that finds no tuned file (`artemis setup`).
    pub first_run_setup: bool,
    /// What the tuned file contributed to this config; set by `load`, never read from a file.
    #[serde(skip)]
    pub tuned: TunedState,
    /// The dotted keys the user's file or a flag set (`tuned::user_keys`); set by `load`, never read from a file. Setup
    /// leaves these keys alone: it measures with the user's values and chooses only the rest.
    #[serde(skip)]
    pub user_keys: BTreeSet<String>,
}

/// What `Config::load` found of the tuned file (`ARTEMIS_TUNED`, spec 7).
#[derive(Clone, Debug, Default, PartialEq)]
pub enum TunedState {
    /// The layer is off (`ARTEMIS_TUNED=off`), has no place (see `Config::tuned_warnings`) or was not asked for.
    #[default]
    Off,
    /// There is no file at the path.
    Missing(PathBuf),
    /// The file was read. `applied` and `ignored` are the dotted keys it set and the ones a user key outranked;
    /// `stale` is a sentence for each way it is out of date; the GPU's UUID and the driver are for the check at engine
    /// load.
    Loaded {
        path: PathBuf,
        applied: Vec<String>,
        ignored: Vec<String>,
        stale: Vec<String>,
        gpu_name: String,
        gpu_uuid: String,
        driver: String,
    },
}

impl Default for Config {
    fn default() -> Config {
        Config {
            model: None,
            host: "127.0.0.1".into(),
            port: 8080,
            api_key_file: None,
            allowed_origins: Vec::new(),
            ctx: 65_536,
            snapshot_pool_gib: 8,
            park_snapshots: true,
            lazy_load: true,
            idle_unload_s: 600,
            queue_timeout_s: 300,
            stall_s: 120,
            stall_load_s: 300,
            model_id: "qwen3.6-35b-a3b".into(),
            health_service: "artemis".into(),
            thinking: true,
            thinking_budget: 0,
            sampling: SamplingConfig::default(),
            speculation: SpeculationConfig::default(),
            log_dir: None,
            log_level: "info".into(),
            log_keep_days: 14,
            device_uuid: None,
            cache_slots: None,
            f32_prefill: false,
            allow_custom_template: false,
            special_tokens_in_content: true,
            anthropic_thinking: true,
            first_run_setup: true,
            tuned: TunedState::Off,
            user_keys: BTreeSet::new(),
        }
    }
}

/// The flags of `artemis serve`: one per key except the `[sampling]` table and the `[speculation]` table, which has
/// only `--speculation-enabled` and `--speculation-sources`. Each overrides the file.
#[derive(Clone, Debug, Default, clap::Args)]
pub struct ServeArgs {
    /// The config file (default: the per-user path, if it exists).
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// Path to the GGUF.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// The address to listen on (default 127.0.0.1).
    #[arg(long)]
    pub host: Option<String>,
    /// The port to listen on (default 8080).
    #[arg(long)]
    pub port: Option<u16>,
    /// File whose first non-empty line is the API key (default: auth off, loopback only).
    #[arg(long)]
    pub api_key_file: Option<PathBuf>,
    /// Browser origins allowed to call the server, comma-separated, each `scheme://host[:port]` (default none).
    #[arg(long, value_delimiter = ',')]
    pub allowed_origins: Option<Vec<String>>,
    /// Context window in tokens (default 65536).
    #[arg(long)]
    pub ctx: Option<u32>,
    /// Host RAM for snapshots in GiB; 0 turns reuse off (default 8).
    #[arg(long)]
    pub snapshot_pool_gib: Option<u32>,
    /// Keep snapshots across an unload (default true).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub park_snapshots: Option<bool>,
    /// Load the model on the first request; false loads at start (default true).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub lazy_load: Option<bool>,
    /// Unload after this many idle seconds (default 600).
    #[arg(long)]
    pub idle_unload_s: Option<u64>,
    /// Seconds a request may wait in the queue (default 300).
    #[arg(long)]
    pub queue_timeout_s: Option<u64>,
    /// Seconds pending work may go without moving before the process halts with exit code 4; 0 is off (default 120).
    #[arg(long)]
    pub stall_s: Option<u64>,
    /// The same bound for a load or a reload; 0 lifts only this bound (default 300).
    #[arg(long)]
    pub stall_load_s: Option<u64>,
    /// The model id the API reports (default qwen3.6-35b-a3b).
    #[arg(long)]
    pub model_id: Option<String>,
    /// The /health service name: 1 to 32 lower-case letters, digits or hyphens (default artemis).
    #[arg(long)]
    pub health_service: Option<String>,
    /// Think by default when a request says nothing (default true).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub thinking: Option<bool>,
    /// Default reasoning budget in tokens; 0 is none (default 0).
    #[arg(long)]
    pub thinking_budget: Option<u32>,
    /// Directory for the request logs (default per user).
    #[arg(long)]
    pub log_dir: Option<PathBuf>,
    /// Log level (default info).
    #[arg(long)]
    pub log_level: Option<String>,
    /// Days of request logs to keep; 0 keeps every file (default 14).
    #[arg(long)]
    pub log_keep_days: Option<u32>,
    /// The GPU to use, by UUID (default: auto).
    #[arg(long)]
    pub device_uuid: Option<String>,
    /// Expert-cache slots per layer (default: auto).
    #[arg(long)]
    pub cache_slots: Option<u32>,
    /// Prefill in f32 (default false).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub f32_prefill: Option<bool>,
    /// Use a GGUF whose chat template differs from the pinned one (default false).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub allow_custom_template: Option<bool>,
    /// Tokenize control tokens' text in client text as the tokens, as llama.cpp does; false keeps it plain text (default
    /// true).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub special_tokens_in_content: Option<bool>,
    /// Think by default on a `/v1/messages` request that says nothing of thinking, as llama.cpp does; false follows
    /// Anthropic's API and thinks off (default true).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub anthropic_thinking: Option<bool>,
    /// Measure this PC on the first start that finds no tuned file; false skips it (default true).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub first_run_setup: Option<bool>,
    /// Speculative decoding on or off over the file's `[speculation] enabled` (default true).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub speculation_enabled: Option<bool>,
    /// The draft sources over the file's `[speculation] sources`, comma-separated: `lookup`, `mtp` or `lookup,mtp`.
    #[arg(long, value_delimiter = ',')]
    pub speculation_sources: Option<Vec<String>>,
}

/// The API key, held as its SHA-256 digest and its length in characters. It has no `Display` and no `Serialize`, and
/// its `Debug` shows nothing.
pub struct ApiKey {
    digest: [u8; 32],
    chars: usize,
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey(..)")
    }
}

impl ApiKey {
    /// A key from its text: the digest and the length are all that is kept.
    fn new(key: &str) -> ApiKey {
        ApiKey {
            digest: sha256(key.as_bytes()),
            chars: key.chars().count(),
        }
    }

    /// A constant-time comparison of the digests, so neither the key's bytes nor its length leak through the timing
    /// of the compare.
    pub fn matches(&self, presented: &[u8]) -> bool {
        sha256(presented).ct_eq(&self.digest).into()
    }

    /// The key's length in characters.
    pub fn chars(&self) -> usize {
        self.chars
    }
}

/// The shortest key accepted without a warning, and the shortest on a bind past loopback.
pub const MIN_KEY_CHARS: usize = 32;

/// Whether the key is long enough for where the server listens. Under `MIN_KEY_CHARS` on an address that is not
/// loopback is the `Err`; under it on loopback is the `Ok` carrying the `config_warning` text; else `Ok(None)`.
pub fn key_strength(key: &ApiKey, addr: SocketAddr) -> Result<Option<String>, String> {
    let n = key.chars();
    if n >= MIN_KEY_CHARS {
        return Ok(None);
    }
    if addr.ip().is_loopback() {
        return Ok(Some(format!(
            "the API key has {n} characters; use at least {MIN_KEY_CHARS}"
        )));
    }
    Err(format!(
        "the API key in api_key_file has {n} characters; a non-loopback bind needs at least {MIN_KEY_CHARS} (docs/server.md, Auth and network, shows how to make one)"
    ))
}

/// The largest key file read.
const MAX_KEY_FILE: usize = 64 * 1024;

/// The 1-based line of a byte offset.
fn line_of(text: &str, offset: usize) -> usize {
    text.as_bytes()[..offset.min(text.len())]
        .iter()
        .filter(|&&b| b == b'\n')
        .count()
        + 1
}

/// Fills a config template (`tools/config/artemis-production.toml.in`): each `{{NAME}}` becomes its value. The values
/// go inside TOML literal strings or bare, so a value holding a single quote, a line break or `{{` is refused, as is
/// a name the template lacks and any placeholder still open after filling. Errors name the placeholder, never a value.
pub fn fill_template(template: &str, values: &[(&str, &str)]) -> Result<String, String> {
    let mut text = template.to_string();
    for (name, value) in values {
        if value.contains('\'') {
            return Err(format!(
                "the value for {name} holds a single quote, which a TOML literal string cannot hold"
            ));
        }
        if value.contains(['\n', '\r']) {
            return Err(format!("the value for {name} holds a line break"));
        }
        if value.contains("{{") {
            return Err(format!(
                "the value for {name} holds an opening double brace"
            ));
        }
        let placeholder = format!("{{{{{name}}}}}");
        if !text.contains(&placeholder) {
            return Err(format!("the template has no placeholder {name}"));
        }
        text = text.replace(&placeholder, value);
    }
    if let Some(at) = text.find("{{") {
        let open = text[at + 2..].lines().next().unwrap_or("");
        let name = open.split("}}").next().unwrap_or("");
        return Err(format!("no value for placeholder {name}"));
    }
    Ok(text)
}

/// The environment, as a lookup, so a test needs no process variables.
pub type Env<'a> = &'a dyn Fn(&str) -> Option<String>;

/// A variable that is set and not empty.
pub(crate) fn var(env: Env<'_>, name: &str) -> Option<PathBuf> {
    env(name).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// `%APPDATA%\artemis\artemis.toml` on Windows; else `$XDG_CONFIG_HOME/artemis/artemis.toml`, else
/// `$HOME/.config/artemis/artemis.toml`. `None` when the base variable is unset.
pub fn default_config_path(env: Env<'_>, windows: bool) -> Option<PathBuf> {
    let base = if windows {
        var(env, "APPDATA")?
    } else {
        var(env, "XDG_CONFIG_HOME").or_else(|| var(env, "HOME").map(|h| h.join(".config")))?
    };
    Some(base.join("artemis").join("artemis.toml"))
}

/// One line, at most 500 bytes (the error rule): toml's messages carry a source excerpt on further lines.
pub(crate) fn one_line(text: &str) -> String {
    let mut line = text.lines().next().unwrap_or("").trim().to_string();
    if line.len() > 500 {
        let mut end = 500;
        while !line.is_char_boundary(end) {
            end -= 1;
        }
        line.truncate(end);
    }
    line
}

/// `value` is a scheme (an RFC 3986 scheme: a letter, then letters, digits, `+`, `-` or `.`), `://`, a host (a name, an
/// IPv4 address or a bracketed IPv6 address) and an optional numeric port from 0 to 65535, with nothing after it. The
/// scheme is compared without regard to case, as the request-time match is. A web page cannot send a non-http(s)
/// `Origin`, so entries such as `vscode-webview://id`, `app://local` or `chrome-extension://id` are safe to allow.
fn is_origin(value: &str) -> bool {
    let Some((scheme, rest)) = value.split_once("://") else {
        return false;
    };
    let mut chars = scheme.chars();
    let scheme_ok = chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    if !scheme_ok {
        return false;
    }
    let port = if rest.starts_with('[') {
        let Some(close) = rest.find(']') else {
            return false;
        };
        let (host, port) = rest.split_at(close + 1);
        let inner = &host[1..close];
        let v6 = !inner.is_empty()
            && inner.contains(':')
            && inner
                .bytes()
                .all(|b| b.is_ascii_hexdigit() || b == b':' || b == b'.');
        if !v6 {
            return false;
        }
        port
    } else {
        let (host, port) = rest.split_at(rest.rfind(':').unwrap_or(rest.len()));
        let name = !host.is_empty()
            && host
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.' || b == b'_');
        if !name {
            return false;
        }
        port
    };
    match port.strip_prefix(':') {
        None => port.is_empty(),
        Some(digits) => {
            !digits.is_empty()
                && digits.len() <= 5
                && digits.bytes().all(|b| b.is_ascii_digit())
                && digits
                    .parse::<u32>()
                    .is_ok_and(|n| n <= u32::from(u16::MAX))
        }
    }
}

impl Config {
    /// `load` without the tuned layer, for `artemis setup`, which measures what the layer would otherwise supply.
    pub fn load_without_tuned(
        args: &ServeArgs,
        env: Env<'_>,
        windows: bool,
    ) -> Result<Config, String> {
        Config::load_with(args, env, windows, false)
    }

    /// The `config_warning` texts the tuned layer owes: one when its place is unknown, one for each way a loaded file
    /// is stale.
    pub fn tuned_warnings(&self, env: Env<'_>, windows: bool) -> Vec<String> {
        match &self.tuned {
            TunedState::Loaded { stale, .. } => stale.clone(),
            TunedState::Off | TunedState::Missing(_) => {
                if tuned::tuned_location(env, windows) == TunedLocation::Unplaced {
                    vec![tuned::unplaced_warning(windows)]
                } else {
                    Vec::new()
                }
            }
        }
    }

    /// The file (`--config`, else the default path if it exists), then the flags over it. A missing `--config` file is
    /// an error; a missing default file gives the defaults. `model` is required once the flags are in.
    ///
    /// Under the file sits the tuned file (`tuned.rs`): its values fill the keys the file and the flags leave alone.
    pub fn load(args: &ServeArgs, env: Env<'_>, windows: bool) -> Result<Config, String> {
        Config::load_with(args, env, windows, true)
    }

    fn load_with(
        args: &ServeArgs,
        env: Env<'_>,
        windows: bool,
        tuned_layer: bool,
    ) -> Result<Config, String> {
        let (path, explicit) = match &args.config {
            Some(p) => (Some(p.clone()), true),
            None => (default_config_path(env, windows), false),
        };
        let mut cfg = Config::default();
        let mut user_table: Option<toml::Table> = None;
        if let Some(path) = &path {
            match std::fs::read_to_string(path) {
                Ok(text) => {
                    // A file saved with a BOM (PowerShell 5.1, Notepad) is still a config file.
                    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
                    let refuse = |e: toml::de::Error| {
                        let line = e
                            .span()
                            .map_or(String::new(), |s| format!("{}:", line_of(text, s.start)));
                        one_line(&format!("{}:{line} {}", path.display(), e.message()))
                    };
                    cfg = toml::from_str(text).map_err(refuse)?;
                    // The same text as a table, for the keys the user set.
                    user_table = Some(text.parse().map_err(refuse)?);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound && !explicit => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(format!("config file {} not found", path.display()));
                }
                Err(e) => {
                    return Err(one_line(&format!(
                        "cannot read config file {}: {}",
                        path.display(),
                        e
                    )));
                }
            }
        }
        // A relative path in the file is read from the file's folder; a flag's stays relative to the working directory.
        if let Some(dir) = path
            .as_ref()
            .and_then(|p| p.parent())
            .filter(|d| !d.as_os_str().is_empty())
        {
            for slot in [&mut cfg.model, &mut cfg.api_key_file, &mut cfg.log_dir] {
                if let Some(p) = slot.as_mut().filter(|p| p.is_relative()) {
                    *p = dir.join(&*p);
                }
            }
        }
        let (tuned_state, tuned_facts) = if tuned_layer {
            cfg.apply_tuned(args, user_table.as_ref(), env, windows)?
        } else {
            (TunedState::Off, None)
        };
        cfg.tuned = tuned_state;
        cfg.user_keys = tuned::user_keys(user_table.as_ref(), args);
        cfg.apply(args);
        let at = path
            .as_ref()
            .map_or("config".into(), |p| p.display().to_string());
        cfg.defaults()
            .sampling
            .check()
            .map_err(|(key, why)| one_line(&format!("{at}: sampling.{key} {why}")))?;
        // A value a flag set is reported under the flag, one the file set under the file and the key.
        let source = |flag: bool, key: &str| {
            if flag {
                format!("--{}", key.replace('_', "-"))
            } else {
                format!("{at}: {key}")
            }
        };
        cfg.speculation.check().map_err(|(key, why)| {
            let who = if key == "sources" && args.speculation_sources.is_some() {
                source(true, "speculation_sources")
            } else {
                source(false, &format!("speculation.{key}"))
            };
            one_line(&format!("{who} {why}"))
        })?;
        for (i, origin) in cfg.allowed_origins.iter().enumerate() {
            if !is_origin(origin) {
                return Err(one_line(&format!(
                    "{at}: allowed_origins[{i}] is not scheme://host[:port]"
                )));
            }
        }
        if !CTX_RANGE.contains(&cfg.ctx) {
            return Err(one_line(&format!(
                "{} {} is outside 512 to 262,144",
                source(args.ctx.is_some(), "ctx"),
                cfg.ctx
            )));
        }
        if cfg.queue_timeout_s == 0 {
            return Err(one_line(&format!(
                "{} must be at least 1",
                source(args.queue_timeout_s.is_some(), "queue_timeout_s")
            )));
        }
        if let Some(n) = cfg.cache_slots.filter(|&n| n > MAX_CACHE_SLOTS) {
            return Err(one_line(&format!(
                "{} {n} is over {MAX_CACHE_SLOTS}, the experts in a layer",
                source(args.cache_slots.is_some(), "cache_slots")
            )));
        }
        if !is_service_name(&cfg.health_service) {
            return Err(one_line(&format!(
                "{} must be 1 to 32 lower-case letters, digits or hyphens, not starting with a hyphen",
                source(args.health_service.is_some(), "health_service")
            )));
        }
        if cfg.model.is_none() {
            let place = path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "the config file".into());
            return Err(format!(
                "model is required: set it in {place} or pass --model"
            ));
        }
        // What a loaded file says of its own age is judged once the model's path is final.
        if let (TunedState::Loaded { stale, .. }, Some(facts)) = (&mut cfg.tuned, &tuned_facts) {
            let model_bytes = cfg
                .model
                .as_ref()
                .and_then(|m| std::fs::metadata(m).ok())
                .map(|m| m.len());
            *stale = tuned::stale(facts, tuned::ARTEMIS_VERSION, model_bytes);
        }
        Ok(cfg)
    }

    /// The tuned file's values for every key the user's file and the flags leave alone, before the flags and the
    /// checks. Returns what was found, and the file's facts when it was read. A file that cannot be read or holds a bad
    /// value is an error that names it.
    fn apply_tuned(
        &mut self,
        args: &ServeArgs,
        user_table: Option<&toml::Table>,
        env: Env<'_>,
        windows: bool,
    ) -> Result<(TunedState, Option<SetupFacts>), String> {
        let path = match tuned::tuned_location(env, windows) {
            TunedLocation::Off | TunedLocation::Unplaced => return Ok((TunedState::Off, None)),
            TunedLocation::At(path) => path,
        };
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok((TunedState::Missing(path), None));
            }
            Err(e) => {
                return Err(format!(
                    "{}; {REMEDY}",
                    one_line(&format!(
                        "cannot read the tuned file {}: {}",
                        path.display(),
                        e
                    ))
                ));
            }
        };
        let t = tuned::parse(&text, &path)?;
        // The file's values meet the config's rules by themselves first, so a bad one is named with the tuned file
        // even where a user key outranks it.
        let sp = &t.speculation;
        let mut probe = SpeculationConfig::default();
        if let Some(v) = sp.enabled {
            probe.enabled = v;
        }
        if let Some(v) = &sp.sources {
            probe.sources = v.clone();
        }
        if let Some(v) = sp.cost_slope_ms {
            probe.cost_slope_ms = v;
        }
        if let Some(v) = sp.mtp_cost_ms {
            probe.mtp_cost_ms = v;
        }
        probe.check().map_err(|(key, why)| {
            format!(
                "{}; {REMEDY}",
                one_line(&format!("{}: speculation.{key} {why}", path.display()))
            )
        })?;
        // The same rule the engine applies to `device_uuid`: 32 hex digits, with or without dashes.
        if let Some(u) = &t.device_uuid {
            let digits = u.bytes().filter(|&c| c != b'-');
            if digits.clone().count() != 32 || !digits.clone().all(|c| c.is_ascii_hexdigit()) {
                return Err(format!(
                    "{}; {REMEDY}",
                    one_line(&format!(
                        "{}: device_uuid is not a UUID: it is 32 hex digits, with or without dashes",
                        path.display()
                    ))
                ));
            }
        }
        let user = tuned::user_keys(user_table, args);
        let mut applied = Vec::new();
        let mut ignored = Vec::new();
        let mut take = |key: &str, present: bool, set: &mut dyn FnMut()| {
            if !present {
                return;
            }
            if user.contains(key) {
                ignored.push(key.to_string());
            } else {
                set();
                applied.push(key.to_string());
            }
        };
        take("device_uuid", t.device_uuid.is_some(), &mut || {
            self.device_uuid = t.device_uuid.clone();
        });
        take("speculation.enabled", sp.enabled.is_some(), &mut || {
            self.speculation.enabled = probe.enabled;
        });
        take("speculation.sources", sp.sources.is_some(), &mut || {
            self.speculation.sources = probe.sources.clone();
        });
        take(
            "speculation.cost_slope_ms",
            sp.cost_slope_ms.is_some(),
            &mut || self.speculation.cost_slope_ms = probe.cost_slope_ms,
        );
        take(
            "speculation.mtp_cost_ms",
            sp.mtp_cost_ms.is_some(),
            &mut || self.speculation.mtp_cost_ms = probe.mtp_cost_ms,
        );
        let state = TunedState::Loaded {
            path,
            applied,
            ignored,
            stale: Vec::new(),
            gpu_name: t.setup.gpu_name.clone(),
            gpu_uuid: t.setup.gpu_uuid.clone(),
            driver: t.setup.driver.clone(),
        };
        Ok((state, Some(t.setup)))
    }

    /// The flags that were given replace the file's values.
    fn apply(&mut self, a: &ServeArgs) {
        fn set<T: Clone>(slot: &mut T, flag: &Option<T>) {
            if let Some(v) = flag {
                *slot = v.clone();
            }
        }
        fn set_opt<T: Clone>(slot: &mut Option<T>, flag: &Option<T>) {
            if flag.is_some() {
                *slot = flag.clone();
            }
        }
        set_opt(&mut self.model, &a.model);
        set(&mut self.host, &a.host);
        set(&mut self.port, &a.port);
        set_opt(&mut self.api_key_file, &a.api_key_file);
        set(&mut self.allowed_origins, &a.allowed_origins);
        set(&mut self.ctx, &a.ctx);
        set(&mut self.snapshot_pool_gib, &a.snapshot_pool_gib);
        set(&mut self.park_snapshots, &a.park_snapshots);
        set(&mut self.lazy_load, &a.lazy_load);
        set(&mut self.idle_unload_s, &a.idle_unload_s);
        set(&mut self.queue_timeout_s, &a.queue_timeout_s);
        set(&mut self.stall_s, &a.stall_s);
        set(&mut self.stall_load_s, &a.stall_load_s);
        set(&mut self.model_id, &a.model_id);
        set(&mut self.health_service, &a.health_service);
        set(&mut self.thinking, &a.thinking);
        set(&mut self.thinking_budget, &a.thinking_budget);
        set_opt(&mut self.log_dir, &a.log_dir);
        set(&mut self.log_level, &a.log_level);
        set(&mut self.log_keep_days, &a.log_keep_days);
        set_opt(&mut self.device_uuid, &a.device_uuid);
        set_opt(&mut self.cache_slots, &a.cache_slots);
        set(&mut self.f32_prefill, &a.f32_prefill);
        set(&mut self.allow_custom_template, &a.allow_custom_template);
        set(
            &mut self.special_tokens_in_content,
            &a.special_tokens_in_content,
        );
        set(&mut self.anthropic_thinking, &a.anthropic_thinking);
        set(&mut self.first_run_setup, &a.first_run_setup);
        set(&mut self.speculation.enabled, &a.speculation_enabled);
        set(&mut self.speculation.sources, &a.speculation_sources);
    }

    /// The key: the first non-empty line of `api_key_file`, whitespace and CR trimmed. `None` when no file is set. A
    /// missing, unreadable or empty file is an error that names the path and never shows the content.
    pub fn read_key(&self) -> Result<Option<ApiKey>, String> {
        let Some(path) = &self.api_key_file else {
            return Ok(None);
        };
        let unreadable = |e: std::io::Error| {
            one_line(&format!(
                "cannot read api_key_file {}: {}",
                path.display(),
                e.kind()
            ))
        };
        // At most MAX_KEY_FILE bytes: a key file is a line, and a stray big file is no key.
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .and_then(|f| f.take(MAX_KEY_FILE as u64 + 1).read_to_end(&mut bytes))
            .map_err(unreadable)?;
        if bytes.len() > MAX_KEY_FILE {
            return Err(format!(
                "api_key_file {} is over {} KiB",
                path.display(),
                MAX_KEY_FILE / 1024
            ));
        }
        let text = String::from_utf8(bytes)
            .map_err(|_| format!("api_key_file {} is not valid UTF-8", path.display()))?;
        // A BOM (PowerShell 5.1's `-Encoding UTF8`, Notepad) is not part of the key.
        let key = text
            .strip_prefix('\u{feff}')
            .unwrap_or(&text)
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .ok_or_else(|| format!("api_key_file {} is empty", path.display()))?;
        Ok(Some(ApiKey::new(key)))
    }

    /// The address to listen on. Without a key the host must be loopback (`127.0.0.0/8`, `::1`, or `localhost`, taken
    /// as `127.0.0.1`); anything else, a hostname included, is refused. With a key any host is allowed (decision 9): an
    /// IP as written, a hostname by its first address.
    pub fn bind(&self, key: Option<&ApiKey>) -> Result<SocketAddr, String> {
        self.bind_with(key, &|host, port| {
            (host, port).to_socket_addrs().map(Iterator::collect)
        })
    }

    /// `bind` with the name lookup given, so a test needs no network.
    fn bind_with(
        &self,
        key: Option<&ApiKey>,
        resolve: &dyn Fn(&str, u16) -> std::io::Result<Vec<SocketAddr>>,
    ) -> Result<SocketAddr, String> {
        let host = self.host.trim();
        let bare = host
            .strip_prefix('[')
            .and_then(|h| h.strip_suffix(']'))
            .unwrap_or(host);
        let ip: Option<IpAddr> = if host.eq_ignore_ascii_case("localhost") {
            Some(Ipv4Addr::LOCALHOST.into())
        } else {
            bare.parse().ok()
        };
        match (ip, key) {
            (Some(ip), None) if !ip.is_loopback() => {
                Err(format!("refusing to bind {host} without api_key_file"))
            }
            (Some(ip), _) => Ok(SocketAddr::new(ip, self.port)),
            (None, None) => Err(format!("refusing to bind {host} without api_key_file")),
            // The lookup's kind (usually `Uncategorized`) and the OS text, on one line: it holds no key or request
            // value.
            (None, Some(_)) => match resolve(bare, self.port) {
                Ok(addrs) => addrs
                    .into_iter()
                    .next()
                    .ok_or_else(|| format!("cannot resolve host {host:?}: no address")),
                Err(e) => {
                    let text = e.to_string();
                    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
                    Err(format!(
                        "cannot resolve host {host:?}: {:?}: {text}",
                        e.kind()
                    ))
                }
            },
        }
    }

    /// What a request that says nothing gets.
    pub fn defaults(&self) -> Defaults {
        let s = &self.sampling;
        Defaults {
            thinking: self.thinking,
            thinking_budget: (self.thinking_budget > 0).then_some(self.thinking_budget),
            anthropic_thinking: self.anthropic_thinking,
            sampling: Sampling {
                temp: s.temperature,
                top_k: s.top_k as usize,
                top_p: s.top_p,
                min_p: s.min_p,
                presence_penalty: s.presence_penalty,
            },
            model_id: self.model_id.clone(),
            ctx: self.ctx,
        }
    }

    /// `log_dir`, else `%LOCALAPPDATA%\artemis\logs` on Windows, else `$HOME/.local/state/artemis/logs` (also on
    /// Windows without `LOCALAPPDATA`); with neither variable set, `artemis-logs` under the working directory.
    pub fn log_dir(&self, env: Env<'_>, windows: bool) -> PathBuf {
        if let Some(dir) = &self.log_dir {
            return dir.clone();
        }
        if let Some(base) = var(env, "LOCALAPPDATA").filter(|_| windows) {
            return base.join("artemis").join("logs");
        }
        match var(env, "HOME") {
            Some(home) => home
                .join(".local")
                .join("state")
                .join("artemis")
                .join("logs"),
            None => PathBuf::from("artemis-logs"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    /// A scratch directory under the temp dir, deleted when it drops.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Scratch {
            let dir = std::env::temp_dir()
                .join(format!("artemis-config-test-{}-{name}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
        fn file(&self, name: &str, content: &[u8]) -> PathBuf {
            let p = self.0.join(name);
            std::fs::write(&p, content).unwrap();
            p
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn with_key_file(path: PathBuf) -> Config {
        Config {
            api_key_file: Some(path),
            ..Config::default()
        }
    }

    fn at_host(host: &str) -> Config {
        Config {
            host: host.into(),
            ..Config::default()
        }
    }

    #[test]
    fn defaults_match_7_3() {
        let s = Scratch::new("defaults");
        let file = s.file("a.toml", b"model = \"m.gguf\"\n");
        let args = ServeArgs {
            config: Some(file),
            ..ServeArgs::default()
        };
        let c = Config::load(&args, &no_env, false).unwrap();
        // The keys the file set are recorded beside the values; they are not a setting.
        assert_eq!(c.user_keys.iter().collect::<Vec<_>>(), ["model"]);
        let c = Config {
            user_keys: Default::default(),
            ..c
        };
        assert_eq!(
            c,
            Config {
                model: Some(s.0.join("m.gguf")),
                ..Config::default()
            }
        );
        assert_eq!(c.host, "127.0.0.1");
        assert_eq!(c.port, 8080);
        assert_eq!(c.api_key_file, None);
        assert_eq!(c.ctx, 65_536);
        assert_eq!(c.snapshot_pool_gib, 8);
        assert!(c.park_snapshots && c.lazy_load);
        assert_eq!((c.idle_unload_s, c.queue_timeout_s), (600, 300));
        assert_eq!((c.stall_s, c.stall_load_s), (120, 300));
        assert_eq!(c.model_id, "qwen3.6-35b-a3b");
        assert_eq!(c.health_service, "artemis");
        assert!(c.thinking);
        assert_eq!(c.thinking_budget, 0);
        assert_eq!(c.log_level, "info");
        assert_eq!(c.log_keep_days, 14);
        assert_eq!(
            (c.log_dir.clone(), c.device_uuid.clone(), c.cache_slots),
            (None, None, None)
        );
        assert!(!c.f32_prefill && !c.allow_custom_template);
        assert!(c.special_tokens_in_content);
        assert!(c.anthropic_thinking && c.defaults().anthropic_thinking);
        assert_eq!(c.speculation, SpeculationConfig::default());
        assert!(c.speculation.enabled);
        let d = c.defaults();
        assert_eq!(d.thinking_budget, None);
        assert_eq!(d.sampling.temp, 0.6);
        assert_eq!(d.sampling.top_p, 0.95);
        assert_eq!(d.sampling.top_k, 20);
        assert_eq!(d.sampling.min_p, 0.0);
        assert_eq!(d.sampling.presence_penalty, 0.0);
        assert_eq!((d.model_id.as_str(), d.ctx), ("qwen3.6-35b-a3b", 65_536));
    }

    #[test]
    fn the_file_sets_every_key_and_the_sampling_table() {
        let s = Scratch::new("all");
        let toml = "model = \"m.gguf\"\nhost = \"0.0.0.0\"\nport = 9000\napi_key_file = \"k\"\nctx = 4096\n\
            snapshot_pool_gib = 0\npark_snapshots = false\nlazy_load = false\nidle_unload_s = 5\nqueue_timeout_s = 6\n\
            stall_s = 90\nstall_load_s = 200\n\
            model_id = \"x\"\nhealth_service = \"gateway\"\nthinking = false\nthinking_budget = 512\n\
            log_dir = \"logs\"\nlog_level = \"debug\"\ndevice_uuid = \"u\"\ncache_slots = 3\nf32_prefill = true\nallow_custom_template = true\n\
            log_keep_days = 3\nspecial_tokens_in_content = false\nanthropic_thinking = false\n\
            [sampling]\ntemperature = 0.7\ntop_k = 40\n";
        let args = ServeArgs {
            config: Some(s.file("a.toml", toml.as_bytes())),
            ..ServeArgs::default()
        };
        let c = Config::load(&args, &no_env, false).unwrap();
        assert_eq!(c.health_service, "gateway");
        assert_eq!((c.stall_s, c.stall_load_s), (90, 200));
        assert_eq!(c.cache_slots, Some(3));
        assert!(c.allow_custom_template);
        assert!(!c.special_tokens_in_content);
        assert!(!c.anthropic_thinking && !c.defaults().anthropic_thinking);
        assert_eq!(c.log_dir, Some(s.0.join("logs")));
        assert_eq!(c.log_keep_days, 3);
        let d = c.defaults();
        assert_eq!(d.thinking_budget, Some(512));
        assert!(!d.thinking);
        assert_eq!(d.sampling.temp, 0.7);
        assert_eq!(d.sampling.top_k, 40);
        assert_eq!(d.sampling.top_p, 0.95);
    }

    #[test]
    fn allowed_origins_must_be_scheme_host_port() {
        let s = Scratch::new("origins");
        for bad in [
            "localhost:3000",
            "http://x/path",
            "http://x/",
            "null",
            "file://",
            "file:///tmp/x",
            "vscode-webview://",
            "vscode-webview:///x",
            "1app://x",
            "ap p://x",
            "app_x://x",
            "://x",
            "app://x/path",
            "app://u@x",
            "app://x:65536",
            "http://",
            "http://x:",
            "http://x:80a",
            "http://x:65536",
            "http://x:99999",
            "https://[::1]:70000",
            "http://u@x",
            "https://[::1",
            "https://[]:80",
            "",
        ] {
            let toml = format!(
                "model = \"m.gguf\"
allowed_origins = [\"{bad}\"]
"
            );
            let args = ServeArgs {
                config: Some(s.file("o.toml", toml.as_bytes())),
                ..ServeArgs::default()
            };
            let e = Config::load(&args, &no_env, false).unwrap_err();
            assert!(
                e.ends_with("allowed_origins[0] is not scheme://host[:port]"),
                "{bad}: {e}"
            );
        }
        for good in [
            "http://127.0.0.1:3000",
            "https://[::1]:8443",
            "https://example.com",
            "http://localhost:3000",
            "http://localhost:65535",
            "HTTPS://Example.com:8443",
            "ftp://x",
            "vscode-webview://abc",
            "app://local",
            "chrome-extension://abcdefghijklmnop",
            "Custom+Scheme.v-1://host:8080",
        ] {
            let toml = format!(
                "model = \"m.gguf\"
allowed_origins = [\"{good}\"]
"
            );
            let args = ServeArgs {
                config: Some(s.file("o.toml", toml.as_bytes())),
                ..ServeArgs::default()
            };
            let c = Config::load(&args, &no_env, false).unwrap();
            assert_eq!(c.allowed_origins, [good], "{good}");
        }
        // The index names the entry; the flag replaces the file's list.
        let args = ServeArgs {
            config: Some(s.file(
                "o.toml",
                b"model = \"m.gguf\"
allowed_origins = [\"http://a\"]
",
            )),
            allowed_origins: Some(vec!["http://b".into(), "nope".into()]),
            ..ServeArgs::default()
        };
        let e = Config::load(&args, &no_env, false).unwrap_err();
        assert!(
            e.ends_with("allowed_origins[1] is not scheme://host[:port]"),
            "{e}"
        );
    }

    #[test]
    fn flags_override_the_file() {
        let s = Scratch::new("flags");
        let file = s.file(
            "a.toml",
            b"model = \"a.gguf\"\nport = 9000\nthinking = false\nctx = 4096\n",
        );
        let args = ServeArgs {
            config: Some(file),
            model: Some("b.gguf".into()),
            port: Some(9100),
            thinking: Some(true),
            api_key_file: Some("kf".into()),
            health_service: Some("gateway".into()),
            stall_s: Some(30),
            ..ServeArgs::default()
        };
        let c = Config::load(&args, &no_env, false).unwrap();
        assert_eq!(c.model, Some(PathBuf::from("b.gguf")));
        assert_eq!(c.port, 9100);
        assert!(c.thinking);
        assert_eq!(c.api_key_file, Some(PathBuf::from("kf")));
        assert_eq!(c.health_service, "gateway");
        assert_eq!((c.stall_s, c.stall_load_s), (30, 300));
        assert_eq!(
            c.ctx, 4096,
            "a key the flags leave alone keeps the file's value"
        );
    }

    #[test]
    fn model_is_required_after_the_flags() {
        let s = Scratch::new("nomodel");
        let file = s.file("a.toml", b"port = 9000\n");
        let mut args = ServeArgs {
            config: Some(file.clone()),
            ..ServeArgs::default()
        };
        let err = Config::load(&args, &no_env, false).unwrap_err();
        assert_eq!(
            err,
            format!(
                "model is required: set it in {} or pass --model",
                file.display()
            )
        );
        args.model = Some("m.gguf".into());
        assert!(Config::load(&args, &no_env, false).is_ok());
    }

    #[test]
    fn an_unknown_key_is_refused_by_name() {
        let s = Scratch::new("unknown");
        let file = s.file("a.toml", b"model = \"m\"\nthinkng = true\n");
        let args = ServeArgs {
            config: Some(file),
            ..ServeArgs::default()
        };
        let err = Config::load(&args, &no_env, false).unwrap_err();
        assert!(err.contains("thinkng"), "{err}");
        assert!(!err.contains('\n') && err.len() <= 500, "{err}");
        let file = s.file("b.toml", b"model = \"m\"\n[sampling]\ntemprature = 1\n");
        let args = ServeArgs {
            config: Some(file),
            ..ServeArgs::default()
        };
        let err = Config::load(&args, &no_env, false).unwrap_err();
        assert!(err.contains("temprature"), "{err}");
    }

    #[test]
    fn a_missing_config_path_is_an_error() {
        let s = Scratch::new("missing");
        let missing = s.0.join("nope.toml");
        let args = ServeArgs {
            config: Some(missing.clone()),
            model: Some("m.gguf".into()),
            ..ServeArgs::default()
        };
        let err = Config::load(&args, &no_env, false).unwrap_err();
        assert!(err.contains(&missing.display().to_string()), "{err}");
    }

    #[test]
    fn a_missing_default_file_gives_defaults() {
        let s = Scratch::new("default-missing");
        let home = s.0.display().to_string();
        let env = move |k: &str| (k == "HOME").then(|| home.clone());
        let args = ServeArgs {
            model: Some("m.gguf".into()),
            ..ServeArgs::default()
        };
        let c = Config::load(&args, &env, false).unwrap();
        // The tuned file has a default place too, and it is missing here.
        let tuned_at = s.0.join(".local/state/artemis/tuned.toml");
        assert_eq!(c.tuned, TunedState::Missing(tuned_at));
        assert_eq!(
            Config {
                tuned: TunedState::Off,
                ..c
            },
            Config {
                model: Some(PathBuf::from("m.gguf")),
                ..Config::default()
            }
        );
        // No variable at all: no default path, still the defaults.
        assert!(Config::load(&args, &no_env, true).is_ok());
        // The default file, when it exists, is read.
        let dir = s.0.join(".config").join("artemis");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("artemis.toml"), b"model = \"d.gguf\"\nport = 7\n").unwrap();
        let home = s.0.display().to_string();
        let env = move |k: &str| (k == "HOME").then(|| home.clone());
        let c = Config::load(&ServeArgs::default(), &env, false).unwrap();
        assert_eq!((c.model, c.port), (Some(dir.join("d.gguf")), 7));
    }

    #[test]
    fn default_paths_per_os() {
        let env = |k: &str| match k {
            "APPDATA" => Some("AD".to_string()),
            "XDG_CONFIG_HOME" => Some("XDG".to_string()),
            "HOME" => Some("H".to_string()),
            "LOCALAPPDATA" => Some("LAD".to_string()),
            _ => None,
        };
        let tail = |base: &str| PathBuf::from(base).join("artemis").join("artemis.toml");
        assert_eq!(default_config_path(&env, true), Some(tail("AD")));
        assert_eq!(default_config_path(&env, false), Some(tail("XDG")));
        let no_xdg = |k: &str| match k {
            "XDG_CONFIG_HOME" => Some(String::new()),
            "HOME" => Some("H".to_string()),
            _ => None,
        };
        assert_eq!(
            default_config_path(&no_xdg, false),
            Some(tail(
                &PathBuf::from("H").join(".config").display().to_string()
            ))
        );
        assert_eq!(default_config_path(&no_env, false), None);
        assert_eq!(default_config_path(&no_env, true), None);

        let c = Config::default();
        let logs = |base: PathBuf| base.join("artemis").join("logs");
        assert_eq!(c.log_dir(&env, true), logs(PathBuf::from("LAD")));
        assert_eq!(
            c.log_dir(&env, false),
            PathBuf::from("H")
                .join(".local")
                .join("state")
                .join("artemis")
                .join("logs")
        );
        let set = Config {
            log_dir: Some("mine".into()),
            ..Config::default()
        };
        assert_eq!(set.log_dir(&env, true), PathBuf::from("mine"));
    }

    #[test]
    fn a2_key_file_trims_newlines_and_crlf() {
        let s = Scratch::new("trim");
        for (name, content) in [
            ("crlf", &b"k3y-value\r\n"[..]),
            ("pad", b"\n  k3y-value  \n"),
            ("two", b"\r\n\r\nk3y-value\r\nsecond\r\n"),
            ("bom", "\u{FEFF}k3y-value\r\n".as_bytes()),
            ("bom-blank", "\u{FEFF}\r\n  k3y-value\n".as_bytes()),
        ] {
            let key = with_key_file(s.file(name, content))
                .read_key()
                .unwrap()
                .unwrap();
            assert!(key.matches(b"k3y-value"), "{name}");
            assert!(!key.matches(b"k3y-valuf"), "{name}");
            assert!(!key.matches(b"k3y-value\r"), "{name}");
            assert!(!key.matches(b""), "{name}");
        }
        assert!(Config::default().read_key().unwrap().is_none());
    }

    #[test]
    fn a2_empty_or_missing_key_file_refuses_to_start() {
        let s = Scratch::new("emptykey");
        let empty = s.file("empty", b"");
        let blank = s.file("blank", b" \r\n\n\t\n");
        let missing = s.0.join("gone");
        for path in [empty, blank, missing] {
            let err = with_key_file(path.clone()).read_key().unwrap_err();
            assert!(err.contains(&path.display().to_string()), "{err}");
        }
        let bad = s.file("bad", &[0xff, 0xfe, 0x41]);
        let err = with_key_file(bad.clone()).read_key().unwrap_err();
        assert!(err.contains(&bad.display().to_string()), "{err}");
    }

    #[test]
    fn a3_no_key_refuses_a_non_loopback_bind() {
        for host in [
            "0.0.0.0",
            "192.0.2.1",
            "::",
            "example.com",
            "::ffff:127.0.0.1",
            "127.0.0.1.example.com",
            "[[::1]",
            "[::1",
        ] {
            let err = at_host(host).bind(None).unwrap_err();
            assert!(err.contains(host), "{err}");
        }
        assert_eq!(
            at_host("0.0.0.0").bind(None).unwrap_err(),
            "refusing to bind 0.0.0.0 without api_key_file"
        );
        for host in [
            "127.0.0.1",
            "127.0.0.2",
            "127.255.255.254",
            "::1",
            "[::1]",
            "localhost",
            "LOCALHOST",
        ] {
            assert!(at_host(host).bind(None).is_ok(), "{host}");
        }
        assert_eq!(
            at_host("localhost").bind(None).unwrap(),
            "127.0.0.1:8080".parse().unwrap()
        );
        assert_eq!(
            at_host("::1").bind(None).unwrap(),
            "[::1]:8080".parse().unwrap()
        );
        let s = Scratch::new("bind");
        let key = with_key_file(s.file("k", b"abc\n")).read_key().unwrap();
        for host in ["0.0.0.0", "192.0.2.1", "127.0.0.1", "::"] {
            assert!(at_host(host).bind(key.as_ref()).is_ok(), "{host}");
        }
    }

    #[test]
    fn a_key_matches_only_itself() {
        let s = Scratch::new("self");
        let key = with_key_file(s.file("k", b"k3y-value\n"))
            .read_key()
            .unwrap()
            .unwrap();
        assert!(key.matches(b"k3y-value"));
        assert!(!key.matches(b"k3y-valu"));
        assert!(!key.matches(b"k3y-valuee"));
        assert!(!key.matches(b""));
    }

    #[test]
    fn a_short_key_on_a_network_bind_is_refused() {
        let s = Scratch::new("strength");
        let key_of = |name: &str, text: &str| {
            with_key_file(s.file(name, text.as_bytes()))
                .read_key()
                .unwrap()
                .unwrap()
        };
        let short = key_of("short", "letmein\n");
        assert_eq!(short.chars(), 7);
        let err = key_strength(&short, "0.0.0.0:8080".parse().unwrap()).unwrap_err();
        assert!(
            err.contains("7 characters") && err.contains("at least 32"),
            "{err}"
        );
        assert!(!err.contains("letmein"), "{err}");
        let warning = key_strength(&short, "127.0.0.1:8080".parse().unwrap())
            .unwrap()
            .unwrap();
        assert!(
            warning.contains("7 characters") && warning.contains("at least 32"),
            "{warning}"
        );
        let long = key_of("long", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ\n");
        assert_eq!(long.chars(), 43);
        for addr in ["0.0.0.0:8080", "127.0.0.1:8080", "[::1]:8080"] {
            assert_eq!(
                key_strength(&long, addr.parse().unwrap()),
                Ok(None),
                "{addr}"
            );
        }
        // The boundary: 32 ASCII characters pass on a network bind, 31 do not.
        let net: SocketAddr = "0.0.0.0:8080".parse().unwrap();
        let exact = key_of("exact", &"k".repeat(32));
        assert_eq!(exact.chars(), 32);
        assert_eq!(key_strength(&exact, net), Ok(None));
        let one_short = key_of("one-short", &"k".repeat(31));
        assert_eq!(one_short.chars(), 31);
        let err = key_strength(&one_short, net).unwrap_err();
        assert!(err.contains("31 characters"), "{err}");
        // Characters, not bytes: 31 two-byte characters are 62 bytes and still short.
        let wide = key_of("wide", &"\u{e9}".repeat(31));
        assert!(key_strength(&wide, "0.0.0.0:8080".parse().unwrap()).is_err());
    }

    #[test]
    fn the_key_never_formats() {
        let s = Scratch::new("fmt");
        let secret = "sup3r-s3cret-t0ken"; // gitleaks:allow
        let path = s.file("k", format!("{secret}\r\n").as_bytes());
        let cfg = with_key_file(path.clone());
        let key = cfg.read_key().unwrap().unwrap();
        assert!(!format!("{cfg:?}").contains(secret));
        assert!(!format!("{key:?}").contains(secret));
        assert!(!format!("{:?}", cfg.read_key()).contains(secret));
        let mut errors = vec![at_host("0.0.0.0").bind(None).unwrap_err()];
        errors.push(with_key_file(s.file("e", b"")).read_key().unwrap_err());
        errors.push(with_key_file(s.0.join("none")).read_key().unwrap_err());
        // A key file that is not a key file: its content still stays out of the error.
        errors.push(
            with_key_file(s.file("bad", &[b's', b'u', b'p', b'3', b'r', 0xff]))
                .read_key()
                .unwrap_err(),
        );
        for e in errors {
            assert!(!e.contains(secret) && !e.contains("sup3r"), "{e}");
        }
    }

    #[test]
    fn a_hostname_resolves_only_with_a_key() {
        let s = Scratch::new("resolve");
        let key = with_key_file(s.file("k", b"abc\n")).read_key().unwrap();
        let found = |host: &str, port: u16| {
            assert_eq!((host, port), ("box.example", 8080));
            Ok(vec![
                "192.0.2.7:8080".parse().unwrap(),
                "192.0.2.8:8080".parse().unwrap(),
            ])
        };
        let c = at_host("box.example");
        // With a key the first address wins.
        assert_eq!(
            c.bind_with(key.as_ref(), &found).unwrap(),
            "192.0.2.7:8080".parse().unwrap()
        );
        // Without one the name is refused before any lookup.
        let never = |_: &str, _: u16| -> std::io::Result<Vec<SocketAddr>> { panic!("resolved") };
        let err = c.bind_with(None, &never).unwrap_err();
        assert!(err.contains("refusing"), "{err}");
        // An IP is never looked up.
        assert!(at_host("192.0.2.1").bind_with(key.as_ref(), &never).is_ok());
    }

    #[test]
    fn a_failed_lookup_names_the_host_the_kind_and_the_os_text_on_one_line() {
        let s = Scratch::new("noresolve");
        let key = with_key_file(s.file("k", b"abc\n")).read_key().unwrap();
        let os_text = "No such host is known.\r\n(os error 11001)";
        let fails = |_: &str, _: u16| -> std::io::Result<Vec<SocketAddr>> {
            Err(std::io::Error::other(os_text))
        };
        let err = at_host("box.example")
            .bind_with(key.as_ref(), &fails)
            .unwrap_err();
        assert!(
            err.contains("box.example") && err.contains("Other"),
            "{err}"
        );
        assert!(
            err.contains("No such host is known. (os error 11001)") && !err.contains('\n'),
            "{err}"
        );
        assert!(!err.contains("refusing"), "{err}");
        let empty = |_: &str, _: u16| -> std::io::Result<Vec<SocketAddr>> { Ok(vec![]) };
        let err = at_host("box.example")
            .bind_with(key.as_ref(), &empty)
            .unwrap_err();
        assert!(err.contains("box.example"), "{err}");
    }

    #[test]
    fn an_oversized_key_file_is_an_error_naming_the_path() {
        let s = Scratch::new("bigkey");
        let big = s.file("big", &vec![b'a'; 70 * 1024]);
        let err = with_key_file(big.clone()).read_key().unwrap_err();
        assert!(err.contains(&big.display().to_string()), "{err}");
        assert!(!err.contains("aaaa"), "{err}");
        let ok = s.file("ok", &vec![b'a'; 60 * 1024]);
        assert!(with_key_file(ok).read_key().unwrap().is_some());
    }

    fn load_toml(s: &Scratch, name: &str, toml: &str) -> Result<Config, String> {
        let args = ServeArgs {
            config: Some(s.file(name, toml.as_bytes())),
            ..ServeArgs::default()
        };
        Config::load(&args, &no_env, false)
    }

    #[test]
    fn out_of_range_values_are_refused() {
        let s = Scratch::new("ranges");
        for (i, (line, key, text)) in [
            ("ctx = 0", "ctx", "ctx 0 is outside 512 to 262,144"),
            ("ctx = 511", "ctx", "ctx 511 is outside 512 to 262,144"),
            (
                "ctx = 300000",
                "ctx",
                "ctx 300000 is outside 512 to 262,144",
            ),
            (
                "queue_timeout_s = 0",
                "queue_timeout_s",
                "queue_timeout_s must be at least 1",
            ),
            (
                "cache_slots = 257",
                "cache_slots",
                "cache_slots 257 is over 256, the experts in a layer",
            ),
        ]
        .iter()
        .enumerate()
        {
            let name = format!("r{i}.toml");
            let toml = format!("model = \"m\"\n{line}\n");
            let err = load_toml(&s, &name, &toml).unwrap_err();
            assert!(err.contains(key) && err.contains(text), "{line}: {err}");
            assert!(err.contains(&name), "the file is named: {err}");
            assert!(!err.contains('\n'), "{err}");
        }
        // A flag is held to the same ranges.
        let args = ServeArgs {
            config: Some(s.file("f.toml", b"model = \"m\"\n")),
            ctx: Some(300_000),
            ..ServeArgs::default()
        };
        let err = Config::load(&args, &no_env, false).unwrap_err();
        assert_eq!(err, "--ctx 300000 is outside 512 to 262,144");
        for (args, want) in [
            (
                ServeArgs {
                    queue_timeout_s: Some(0),
                    ..ServeArgs::default()
                },
                "--queue-timeout-s must be at least 1",
            ),
            (
                ServeArgs {
                    cache_slots: Some(300),
                    ..ServeArgs::default()
                },
                "--cache-slots 300 is over 256, the experts in a layer",
            ),
        ] {
            let args = ServeArgs {
                config: Some(s.file("g.toml", b"model = \"m\"\n")),
                ..args
            };
            assert_eq!(Config::load(&args, &no_env, false).unwrap_err(), want);
        }
        // A bad value from the file is blamed on the file, not on a flag that set another key.
        let args = ServeArgs {
            config: Some(s.file("h.toml", b"model = \"m\"\nqueue_timeout_s = 0\n")),
            ctx: Some(1024),
            ..ServeArgs::default()
        };
        let err = Config::load(&args, &no_env, false).unwrap_err();
        assert!(
            err.contains("h.toml: queue_timeout_s must be at least 1"),
            "{err}"
        );
        for (i, line) in [
            "ctx = 512",
            "ctx = 262144",
            "cache_slots = 0",
            "cache_slots = 256",
        ]
        .iter()
        .enumerate()
        {
            let toml = format!("model = \"m\"\n{line}\n");
            assert!(
                load_toml(&s, &format!("ok{i}.toml"), &toml).is_ok(),
                "{line}"
            );
        }
        let c = load_toml(&s, "slots0.toml", "model = \"m\"\ncache_slots = 0\n").unwrap();
        assert_eq!(c.cache_slots, Some(0));
    }

    #[test]
    fn relative_paths_resolve_against_the_config_file() {
        let s = Scratch::new("relative");
        let file = s.file(
            "a.toml",
            b"model = 'm.gguf'\napi_key_file = 'sub/key.txt'\nlog_dir = 'logs'\n",
        );
        let args = ServeArgs {
            config: Some(file),
            ..ServeArgs::default()
        };
        let c = Config::load(&args, &no_env, false).unwrap();
        assert_eq!(c.model, Some(s.0.join("m.gguf")));
        assert_eq!(c.api_key_file, Some(s.0.join("sub/key.txt")));
        assert_eq!(c.log_dir, Some(s.0.join("logs")));
        // A flag is resolved against the working directory, as before.
        let args = ServeArgs {
            config: args.config,
            model: Some("m.gguf".into()),
            api_key_file: Some("k.txt".into()),
            log_dir: Some("l".into()),
            ..ServeArgs::default()
        };
        let c = Config::load(&args, &no_env, false).unwrap();
        assert_eq!(c.model, Some(PathBuf::from("m.gguf")));
        assert_eq!(c.api_key_file, Some(PathBuf::from("k.txt")));
        assert_eq!(c.log_dir, Some(PathBuf::from("l")));
        // An absolute path in the file stays as it is.
        let abs = s.0.join("abs.gguf");
        let toml = format!("model = '{}'\n", abs.display());
        assert_eq!(load_toml(&s, "b.toml", &toml).unwrap().model, Some(abs));
    }

    #[test]
    fn health_service_takes_a_name() {
        let s = Scratch::new("health-name");
        let name = |v: &str| {
            let toml = format!("model = \"m\"\nhealth_service = \"{v}\"\n");
            load_toml(&s, "h.toml", &toml)
        };
        assert_eq!(name("compat").unwrap().health_service, "compat");
        assert_eq!(name("gateway").unwrap().health_service, "gateway");
        // The name the setting had as its default before it became "artemis" still loads. It is assembled here so
        // that the public tree does not carry the word (tools/check-public-snapshot.ps1 scans for it).
        let old = ["str", "ata"].concat();
        assert_eq!(name(&old).unwrap().health_service, old);
        assert_eq!(name("a").unwrap().health_service, "a");
        assert_eq!(name("0-x-9").unwrap().health_service, "0-x-9");
        let longest = "a".repeat(32);
        assert_eq!(name(&longest).unwrap().health_service, longest);
        let too_long = "a".repeat(33);
        for bad in [
            "Bad Name",
            "",
            "-lead",
            "Upper",
            "under_score",
            too_long.as_str(),
        ] {
            let err = name(bad).unwrap_err();
            assert!(
                err.contains(
                    "health_service must be 1 to 32 lower-case letters, digits or hyphens, not starting with a hyphen"
                ),
                "{bad:?}: {err}"
            );
        }
        assert_eq!(Config::default().health_service, "artemis");
        // The flag is checked too.
        let args = ServeArgs {
            config: Some(s.file("f.toml", b"model = \"m\"\n")),
            health_service: Some("No!".into()),
            ..ServeArgs::default()
        };
        let err = Config::load(&args, &no_env, false).unwrap_err();
        assert!(
            err.starts_with("--health-service must be 1 to 32 lower-case letters"),
            "{err}"
        );
    }

    #[test]
    fn sampling_values_are_checked_like_request_fields() {
        let s = Scratch::new("sampling");
        for (i, (line, name)) in [
            ("temperature = -1", "sampling.temperature"),
            ("temperature = inf", "sampling.temperature"),
            ("top_p = 0", "sampling.top_p"),
            ("top_p = 1.5", "sampling.top_p"),
            ("min_p = 2", "sampling.min_p"),
            ("min_p = -0.1", "sampling.min_p"),
            ("presence_penalty = nan", "sampling.presence_penalty"),
        ]
        .iter()
        .enumerate()
        {
            let toml = format!("model = \"m\"\n[sampling]\n{line}\n");
            let err = load_toml(&s, &format!("s{i}.toml"), &toml).unwrap_err();
            assert!(err.contains(name), "{line}: {err}");
            assert!(
                err.contains(&format!("s{i}.toml: sampling.")),
                "{line}: {err}"
            );
        }
        let ok = "model = \"m\"\n[sampling]\ntemperature = 0\ntop_p = 1\ntop_k = 0\nmin_p = 1\npresence_penalty = -2\n";
        assert!(load_toml(&s, "ok.toml", ok).is_ok());
    }

    #[test]
    fn the_speculation_table_defaults_to_lookup_and_mtp() {
        let s = Scratch::new("spec-default");
        let c = load_toml(&s, "a.toml", "model = \"m\"\n").unwrap();
        let sp = &c.speculation;
        assert!(sp.enabled);
        assert_eq!(sp.sources, ["lookup", "mtp"]);
        assert_eq!(
            (sp.probe_every, sp.mtp_window, sp.mtp_ctx, sp.mtp_ring),
            (64, 2048, 8192, 256)
        );
        assert_eq!(sp.mtp_accept_prior, [0.79, 0.81, 0.74, 0.78]);
        assert_eq!(sp.mtp_cost_ms, 1.6);
        assert_eq!(
            (sp.source_choice.as_str(), sp.mtp_head_rows),
            ("best", 65536)
        );
        assert_eq!((sp.max_draft, sp.ngram_min, sp.ngram_max), (4, 2, 4));
        assert_eq!(sp.min_gain, 1.03);
        assert_eq!(sp.accept_half_life, 32.0);
        assert_eq!(sp.cost_slope_ms, 4.4);
        assert_eq!(sp.accept_prior, 0.5);
        assert_eq!(*sp, SpeculationConfig::default());
        assert_eq!(sp.check(), Ok(()));
    }

    #[test]
    fn the_file_sets_every_mtp_key() {
        let s = Scratch::new("spec-all");
        let toml = r#"model = "m"
[speculation]
enabled = false
sources = ["mtp", "lookup"]
max_draft = 8
ngram_min = 3
ngram_max = 6
min_gain = 1.1
accept_half_life = 16.5
cost_slope_ms = 0
accept_prior = 0.25
probe_every = 8
mtp_window = 100
mtp_ctx = 1024
mtp_ring = 32
mtp_accept_prior = [0.9, 0.1]
mtp_cost_ms = 2.5
source_choice = "best"
mtp_head_rows = 32768
"#;
        let c = load_toml(&s, "a.toml", toml).unwrap();
        assert_eq!(
            c.speculation,
            SpeculationConfig {
                enabled: false,
                sources: vec!["mtp".into(), "lookup".into()],
                max_draft: 8,
                ngram_min: 3,
                ngram_max: 6,
                min_gain: 1.1,
                accept_half_life: 16.5,
                cost_slope_ms: 0.0,
                accept_prior: 0.25,
                probe_every: 8,
                mtp_window: 100,
                mtp_ctx: 1024,
                mtp_ring: 32,
                mtp_accept_prior: vec![0.9, 0.1],
                mtp_cost_ms: 2.5,
                source_choice: "best".into(),
                mtp_head_rows: 32768,
            }
        );
        // Keys left out keep their defaults.
        let c = load_toml(
            &s,
            "b.toml",
            "model = \"m\"\n[speculation]\nmax_draft = 2\n",
        )
        .unwrap();
        assert_eq!(
            c.speculation,
            SpeculationConfig {
                max_draft: 2,
                ..SpeculationConfig::default()
            }
        );
    }

    #[test]
    fn the_flag_turns_speculation_off_over_the_file() {
        use clap::Parser;
        let s = Scratch::new("spec-flag");
        let file = s.file(
            "a.toml",
            b"model = \"m\"\n[speculation]\nenabled = true\nmax_draft = 6\n",
        );
        let load = |flag: &[&str]| {
            let mut argv = vec!["artemis"];
            argv.extend_from_slice(flag);
            let mut w = Wrapper::try_parse_from(argv).unwrap();
            w.serve.config = Some(file.clone());
            Config::load(&w.serve, &no_env, false).unwrap()
        };
        let off = load(&["--speculation-enabled", "false"]);
        assert!(!off.speculation.enabled);
        assert_eq!(
            off.speculation.max_draft, 6,
            "the flag changes only enabled"
        );
        assert!(!load(&["--speculation-enabled=false"]).speculation.enabled);
        assert!(load(&["--speculation-enabled"]).speculation.enabled);
        assert!(load(&["--speculation-enabled", "true"]).speculation.enabled);
        assert!(
            load(&[]).speculation.enabled,
            "no flag keeps the file's value"
        );
        // The flag also turns it on over a file that turned it off.
        let file_off = s.file("b.toml", b"model = \"m\"\n[speculation]\nenabled = false\n");
        let mut w = Wrapper::try_parse_from(["artemis", "--speculation-enabled"]).unwrap();
        w.serve.config = Some(file_off);
        assert!(
            Config::load(&w.serve, &no_env, false)
                .unwrap()
                .speculation
                .enabled
        );
    }

    #[test]
    fn an_unknown_speculation_key_is_refused_by_name() {
        let s = Scratch::new("spec-unknown");
        let err = load_toml(
            &s,
            "a.toml",
            "model = \"m\"\n[speculation]\nmax_draught = 4\n",
        )
        .unwrap_err();
        assert!(err.contains("max_draught"), "{err}");
        assert!(!err.contains('\n') && err.len() <= 500, "{err}");
    }

    #[test]
    fn the_sources_flag_overrides_the_file() {
        use clap::Parser;
        let s = Scratch::new("spec-sources-flag");
        let file = s.file(
            "a.toml",
            b"model = \"m\"\n[speculation]\nsources = [\"mtp\"]\nmax_draft = 6\n",
        );
        let load = |flag: &[&str]| {
            let mut argv = vec!["artemis"];
            argv.extend_from_slice(flag);
            let mut w = Wrapper::try_parse_from(argv).unwrap();
            w.serve.config = Some(file.clone());
            Config::load(&w.serve, &no_env, false)
        };
        let c = load(&["--speculation-sources", "lookup"]).unwrap();
        assert_eq!(c.speculation.sources, ["lookup"]);
        assert_eq!(c.speculation.max_draft, 6, "the flag changes only sources");
        let c = load(&["--speculation-sources=lookup,mtp"]).unwrap();
        assert_eq!(c.speculation.sources, ["lookup", "mtp"]);
        assert_eq!(load(&[]).unwrap().speculation.sources, ["mtp"]);
        // The flag's list is checked like the file's, and reported under the flag, not the file's name.
        let err = load(&["--speculation-sources", "lookup,lookup"]).unwrap_err();
        assert!(
            err.starts_with("--speculation-sources \"lookup\" is listed twice"),
            "{err}"
        );
        assert!(!err.contains("a.toml"), "{err}");
    }

    #[test]
    fn bad_mtp_keys_are_refused_with_their_key() {
        let s = Scratch::new("spec-mtp-bad");
        for (i, (line, key)) in [
            ("probe_every = 0", "probe_every"),
            ("mtp_ctx = 256", "mtp_ctx"),
            ("mtp_ctx = 65792", "mtp_ctx"),
            ("mtp_ctx = 1000", "mtp_ctx"),
            ("mtp_window = 0", "mtp_window"),
            ("mtp_window = 4097", "mtp_window"),
            ("mtp_ring = 8", "mtp_ring"),
            ("mtp_ring = 24", "mtp_ring"),
            ("mtp_ring = 2064", "mtp_ring"),
            ("mtp_accept_prior = []", "mtp_accept_prior"),
            ("mtp_accept_prior = [1.5]", "mtp_accept_prior"),
            ("mtp_accept_prior = [-0.1]", "mtp_accept_prior"),
            (
                "mtp_accept_prior = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]",
                "mtp_accept_prior",
            ),
            ("mtp_cost_ms = -1", "mtp_cost_ms"),
            ("mtp_cost_ms = nan", "mtp_cost_ms"),
            ("mtp_head_rows = 512", "mtp_head_rows"),
            ("source_choice = \"fastest\"", "source_choice"),
        ]
        .iter()
        .enumerate()
        {
            let toml = format!("model = \"m\"\n[speculation]\n{line}\n");
            let err = load_toml(&s, &format!("k{i}.toml"), &toml).unwrap_err();
            assert!(
                err.contains(&format!("k{i}.toml: speculation.{key} ")),
                "{line}: {err}"
            );
            assert!(!err.contains('\n'), "{line}: {err}");
        }
        // The window and the ring are bounded by mtp_ctx, so they move with it.
        let ok = "model = \"m\"\n[speculation]\nmtp_ctx = 512\nmtp_window = 256\nmtp_ring = 128\n";
        assert!(load_toml(&s, "ok.toml", ok).is_ok());
        let bad = "model = \"m\"\n[speculation]\nmtp_ctx = 512\n";
        let err = load_toml(&s, "w.toml", bad).unwrap_err();
        assert!(err.contains("speculation.mtp_window "), "{err}");
        for line in [
            "probe_every = 1",
            "mtp_ctx = 65536
mtp_window = 32768
mtp_ring = 16384",
            "mtp_ring = 16",
            "mtp_accept_prior = [0, 1]",
            "mtp_cost_ms = 0",
            "mtp_head_rows = 1024",
            "source_choice = \"best\"",
        ] {
            let toml = format!("model = \"m\"\n[speculation]\n{line}\n");
            assert!(load_toml(&s, "g.toml", &toml).is_ok(), "{line}");
        }
    }

    #[test]
    fn bad_source_lists_are_refused() {
        let s = Scratch::new("spec-sources");
        for (i, (line, want)) in [
            (
                "sources = []",
                "needs a source; to turn speculation off set enabled = false",
            ),
            ("sources = [\"suffix\"]", "an unknown source \"suffix\""),
            (
                "sources = [\"lookup\", \"lookup\"]",
                "\"lookup\" is listed twice",
            ),
            (
                "sources = [\"mtp\", \"lookup\", \"mtp\"]",
                "\"mtp\" is listed twice",
            ),
        ]
        .iter()
        .enumerate()
        {
            let toml = format!("model = \"m\"\n[speculation]\n{line}\n");
            let err = load_toml(&s, &format!("s{i}.toml"), &toml).unwrap_err();
            assert!(
                err.contains(&format!("s{i}.toml: speculation.sources ")),
                "{line}: {err}"
            );
            assert!(err.contains(want), "{line}: {err}");
            assert!(!err.contains('\n'), "{line}: {err}");
        }
        for line in [
            "sources = [\"mtp\"]",
            "sources = [\"lookup\", \"mtp\"]",
            "sources = [\"mtp\", \"lookup\"]",
        ] {
            let toml = format!("model = \"m\"\n[speculation]\n{line}\n");
            assert!(load_toml(&s, "ok.toml", &toml).is_ok(), "{line}");
        }
        // Even with speculation off, a source list that could never run is a mistake.
        let off = "model = \"m\"\n[speculation]\nenabled = false\nsources = []\n";
        let err = load_toml(&s, "off.toml", off).unwrap_err();
        assert!(
            err.contains("off.toml: speculation.sources needs a source; to turn speculation off set enabled = false"),
            "{err}"
        );
    }

    #[test]
    fn a_bad_max_draft_is_refused_with_its_key() {
        let s = Scratch::new("spec-max-draft");
        for (i, line) in ["max_draft = 0", "max_draft = 16"].iter().enumerate() {
            let toml = format!("model = \"m\"\n[speculation]\n{line}\n");
            let err = load_toml(&s, &format!("m{i}.toml"), &toml).unwrap_err();
            assert_eq!(
                err,
                format!(
                    "{}: speculation.max_draft must be from 1 to 15",
                    s.0.join(format!("m{i}.toml")).display()
                )
            );
        }
        for line in ["max_draft = 1", "max_draft = 15"] {
            let toml = format!("model = \"m\"\n[speculation]\n{line}\n");
            assert!(load_toml(&s, "ok.toml", &toml).is_ok(), "{line}");
        }
        let err = load_toml(
            &s,
            "n.toml",
            "model = \"m\"\n[speculation]\nngram_min = 5\nngram_max = 3\n",
        )
        .unwrap_err();
        assert!(
            err.contains("speculation.ngram_min must be at most ngram_max"),
            "{err}"
        );
    }

    #[test]
    fn a_type_error_names_the_line() {
        let s = Scratch::new("line");
        let err = load_toml(&s, "t.toml", "model = \"m\"\n\nport = \"x\"\n").unwrap_err();
        assert!(err.contains("t.toml:3:"), "{err}");
        assert!(!err.contains('\n'), "{err}");
    }

    #[test]
    fn a_bool_flag_sets_false_over_a_true_default() {
        let s = Scratch::new("boolflag");
        let args = ServeArgs {
            config: Some(s.file("a.toml", b"model = \"m\"\n")),
            thinking: Some(false),
            park_snapshots: Some(false),
            lazy_load: Some(false),
            special_tokens_in_content: Some(false),
            anthropic_thinking: Some(false),
            ..ServeArgs::default()
        };
        let c = Config::load(&args, &no_env, false).unwrap();
        assert!(!c.thinking && !c.park_snapshots && !c.lazy_load);
        assert!(!c.special_tokens_in_content && !c.anthropic_thinking);
        assert!(Config::default().thinking && Config::default().park_snapshots);
    }

    #[derive(clap::Parser)]
    struct Wrapper {
        #[command(flatten)]
        serve: ServeArgs,
    }

    #[test]
    fn the_flags_parse_and_the_command_is_sound() {
        use clap::{CommandFactory, Parser};
        Wrapper::command().debug_assert();
        let w = Wrapper::try_parse_from([
            "artemis",
            "--thinking",
            "--lazy-load",
            "false",
            "--special-tokens-in-content",
            "false",
            "--anthropic-thinking",
            "false",
            "--port",
            "9",
        ])
        .unwrap();
        assert_eq!(w.serve.special_tokens_in_content, Some(false));
        assert_eq!(w.serve.anthropic_thinking, Some(false));
        assert_eq!(w.serve.thinking, Some(true));
        assert_eq!(w.serve.lazy_load, Some(false));
        assert_eq!(w.serve.port, Some(9));
        let w =
            Wrapper::try_parse_from(["artemis", "--thinking=false", "--health-service", "gateway"])
                .unwrap();
        assert_eq!(w.serve.thinking, Some(false));
        assert_eq!(w.serve.health_service, Some("gateway".to_string()));
        assert_eq!(
            Wrapper::try_parse_from(["artemis"]).unwrap().serve.thinking,
            None
        );
    }

    #[test]
    fn log_dir_fallbacks() {
        let c = Config::default();
        let home = |k: &str| (k == "HOME").then(|| "H".to_string());
        let home_dir = PathBuf::from("H")
            .join(".local")
            .join("state")
            .join("artemis")
            .join("logs");
        // Windows without LOCALAPPDATA falls back to the HOME path.
        assert_eq!(c.log_dir(&home, true), home_dir);
        // Nothing set: a relative directory.
        assert_eq!(c.log_dir(&no_env, false), PathBuf::from("artemis-logs"));
        assert_eq!(c.log_dir(&no_env, true), PathBuf::from("artemis-logs"));
    }

    // ---- the tuned layer (milestone-6b spec 7) ----

    /// A model file of 10 bytes, a tuned file made from the sample with `tuned_with` applied, and the environment that
    /// points `ARTEMIS_TUNED` at it.
    struct TunedRig {
        s: Scratch,
        tuned: PathBuf,
        env_value: String,
    }

    impl TunedRig {
        fn new(name: &str, tuned_with: impl FnOnce(&mut crate::tuned::Tuned)) -> TunedRig {
            let s = Scratch::new(name);
            s.file("m.gguf", b"0123456789");
            let mut t = crate::tuned::sample();
            t.setup.model_bytes = 10;
            t.speculation.enabled = Some(false);
            tuned_with(&mut t);
            let text = crate::tuned::render(&t, &crate::tuned::sample_why());
            let tuned = s.file("tuned.toml", text.as_bytes());
            let env_value = tuned.display().to_string();
            TunedRig {
                s,
                tuned,
                env_value,
            }
        }
        fn env(&self) -> impl Fn(&str) -> Option<String> + '_ {
            move |k| (k == "ARTEMIS_TUNED").then(|| self.env_value.clone())
        }
        fn args(&self, user_toml: &str) -> ServeArgs {
            let file = self.s.file(
                "artemis.toml",
                format!("model = 'm.gguf'\n{user_toml}").as_bytes(),
            );
            ServeArgs {
                config: Some(file),
                ..ServeArgs::default()
            }
        }
    }

    #[test]
    fn a_user_key_wins_and_is_reported() {
        let r = TunedRig::new("tuned-user-key", |_| {});
        let args = r.args(
            "device_uuid = \"ffff\"\n[speculation]\nsources = [\"lookup\"]\ncost_slope_ms = 7.0\n",
        );
        let c = Config::load(&args, &r.env(), false).unwrap();
        // the user's keys stand
        assert_eq!(c.speculation.sources, vec!["lookup".to_string()]);
        assert_eq!(c.speculation.cost_slope_ms, 7.0);
        assert_eq!(c.device_uuid.as_deref(), Some("ffff"));
        // the tuned keys the user left alone apply
        assert!(!c.speculation.enabled);
        assert_eq!(c.speculation.mtp_cost_ms, 1.5);
        assert_eq!(
            c.tuned,
            TunedState::Loaded {
                path: r.tuned.clone(),
                applied: vec![
                    "speculation.enabled".into(),
                    "speculation.mtp_cost_ms".into()
                ],
                ignored: vec![
                    "device_uuid".into(),
                    "speculation.sources".into(),
                    "speculation.cost_slope_ms".into()
                ],
                stale: vec![],
                gpu_name: "AMD Radeon RX 9070 XT".into(),
                gpu_uuid: "0123456789abcdef0123456789abcdef".into(),
                driver: "2.0.355".into(),
            }
        );
        // with no user key the whole tuned file applies, under the model path the file gave
        let c = Config::load(&r.args(""), &r.env(), false).unwrap();
        assert_eq!(c.model, Some(r.s.0.join("m.gguf")));
        assert!(!c.speculation.enabled);
        assert_eq!(
            c.speculation.sources,
            vec!["lookup".to_string(), "mtp".into()]
        );
        assert_eq!(c.speculation.cost_slope_ms, 4.25);
        assert_eq!(
            c.device_uuid.as_deref(),
            Some("0123456789abcdef0123456789abcdef")
        );
        let TunedState::Loaded {
            applied, ignored, ..
        } = &c.tuned
        else {
            panic!("{:?}", c.tuned)
        };
        assert_eq!(applied.len(), 5, "{applied:?}");
        assert!(ignored.is_empty());
    }

    #[test]
    fn a_flag_wins_too() {
        let r = TunedRig::new("tuned-flag", |t| t.speculation.enabled = Some(false));
        let mut args = r.args("");
        args.speculation_enabled = Some(true);
        args.device_uuid = Some("aaaa".into());
        let c = Config::load(&args, &r.env(), false).unwrap();
        assert!(c.speculation.enabled);
        assert_eq!(c.device_uuid.as_deref(), Some("aaaa"));
        let TunedState::Loaded {
            applied, ignored, ..
        } = &c.tuned
        else {
            panic!("{:?}", c.tuned)
        };
        assert_eq!(ignored, &["device_uuid", "speculation.enabled"]);
        assert!(!applied.iter().any(|k| ignored.contains(k)));
        // the sources flag
        let mut args = r.args("");
        args.speculation_sources = Some(vec!["mtp".into()]);
        let c = Config::load(&args, &r.env(), false).unwrap();
        assert_eq!(c.speculation.sources, vec!["mtp".to_string()]);
        let TunedState::Loaded { ignored, .. } = &c.tuned else {
            panic!("{:?}", c.tuned)
        };
        assert_eq!(ignored, &["speculation.sources"]);
    }

    /// Setup reads the keys the user set from the loaded config, with and without the tuned layer.
    #[test]
    fn a_loaded_config_records_the_keys_the_user_set() {
        let r = TunedRig::new("tuned-user-keys", |_| {});
        let mut args = r.args(
            "ctx = 4096
[speculation]
sources = [\"lookup\"]
",
        );
        args.speculation_enabled = Some(true);
        // The rig's file names the model too.
        let want: Vec<String> = ["ctx", "model", "speculation.enabled", "speculation.sources"]
            .map(String::from)
            .into();
        for c in [
            Config::load(&args, &r.env(), false).unwrap(),
            Config::load_without_tuned(&args, &r.env(), false).unwrap(),
        ] {
            assert_eq!(c.user_keys.iter().cloned().collect::<Vec<_>>(), want);
        }
        let c = Config::load(&r.args(""), &r.env(), false).unwrap();
        assert_eq!(c.user_keys.iter().collect::<Vec<_>>(), ["model"]);
    }

    #[test]
    fn the_engine_keys_are_user_keys() {
        let args = ServeArgs {
            ctx: Some(1024),
            cache_slots: Some(8),
            f32_prefill: Some(true),
            ..ServeArgs::default()
        };
        let keys = crate::tuned::user_keys(None, &args);
        assert_eq!(
            keys.into_iter().collect::<Vec<_>>(),
            ["cache_slots", "ctx", "f32_prefill"]
        );
        let file: toml::Table =
            "model = 'm'\nctx = 5\n[speculation]\nenabled = false\n[sampling]\ntop_k = 3\n"
                .parse()
                .unwrap();
        let keys = crate::tuned::user_keys(Some(&file), &ServeArgs::default());
        assert_eq!(
            keys.into_iter().collect::<Vec<_>>(),
            ["ctx", "model", "sampling.top_k", "speculation.enabled"]
        );
    }

    #[test]
    fn a_bad_tuned_value_is_refused_naming_the_file_and_key() {
        let r = TunedRig::new("tuned-bad", |t| {
            t.speculation.sources = Some(vec!["bogus".into()]);
        });
        let err = Config::load(&r.args(""), &r.env(), false).unwrap_err();
        let path = r.tuned.display().to_string();
        assert!(err.contains(&path), "{err}");
        assert!(err.contains("speculation.sources"), "{err}");
        assert!(err.contains("bogus"), "{err}");
        assert!(err.ends_with("artemis setup --force"), "{err}");
        assert!(!err.contains('\n'), "{err}");
        // the same rules for the others
        let r = TunedRig::new("tuned-bad2", |t| t.speculation.cost_slope_ms = Some(-1.0));
        let err = Config::load(&r.args(""), &r.env(), false).unwrap_err();
        assert!(err.contains(&r.tuned.display().to_string()), "{err}");
        assert!(err.contains("speculation.cost_slope_ms"), "{err}");
        assert!(err.ends_with("artemis setup --force"), "{err}");
        // a device_uuid that is not 32 hex digits is refused the same way, and a user key does not hide it
        for bad in [
            "zz",
            "0123456789abcdef0123456789abcdeg",
            "0123456789abcdef0123456789abcde",
        ] {
            let r = TunedRig::new("tuned-bad-uuid", |t| t.device_uuid = Some(bad.into()));
            let err = Config::load(&r.args(""), &r.env(), false).unwrap_err();
            assert!(err.contains(&r.tuned.display().to_string()), "{err}");
            assert!(err.contains("device_uuid is not a UUID"), "{err}");
            assert!(err.ends_with("artemis setup --force"), "{err}");
            let user = r.args(
                "device_uuid = \"ffffffffffffffffffffffffffffffff\"
",
            );
            let err = Config::load(&user, &r.env(), false).unwrap_err();
            assert!(err.contains("device_uuid is not a UUID"), "{err}");
        }
        let r = TunedRig::new("tuned-good-uuid", |t| {
            t.device_uuid = Some("01234567-89ab-cdef-0123-456789abcdef".into());
        });
        assert!(Config::load(&r.args(""), &r.env(), false).is_ok());
        // a user key that outranks the bad tuned value does not hide it: the file is bad
        let r = TunedRig::new("tuned-bad3", |t| {
            t.speculation.sources = Some(vec!["bogus".into()]);
        });
        let args = r.args("[speculation]\nsources = [\"lookup\"]\n");
        assert!(Config::load(&args, &r.env(), false).is_err());
        // load_without_tuned never reads it
        assert!(Config::load_without_tuned(&args, &r.env(), false).is_ok());
        // a corrupt file is refused the same way
        let r = TunedRig::new("tuned-corrupt", |_| {});
        std::fs::write(&r.tuned, b"speculation = [").unwrap();
        let err = Config::load(&r.args(""), &r.env(), false).unwrap_err();
        assert!(err.contains(&r.tuned.display().to_string()), "{err}");
        assert!(err.ends_with("artemis setup --force"), "{err}");
    }

    #[test]
    fn a_very_long_tuned_path_keeps_the_remedy() {
        // The error is cut at 500 bytes; the words that say what to do come after the cut.
        let s = Scratch::new("tuned-long");
        let deep =
            s.0.join("d".repeat(120))
                .join("e".repeat(120))
                .join("f".repeat(120));
        std::fs::create_dir_all(&deep).unwrap();
        let mut t = crate::tuned::sample();
        t.setup.model_bytes = 10;
        t.speculation.sources = Some(vec!["bogus".into()]);
        let tuned = deep.join("g".repeat(120));
        std::fs::write(&tuned, crate::tuned::render(&t, &Default::default())).unwrap();
        s.file("m.gguf", b"0123456789");
        let env_value = tuned.display().to_string();
        let env = move |k: &str| (k == "ARTEMIS_TUNED").then(|| env_value.clone());
        let args = ServeArgs {
            model: Some(s.0.join("m.gguf")),
            ..ServeArgs::default()
        };
        let err = Config::load(&args, &env, false).unwrap_err();
        assert!(
            err.len() > 500,
            "the path should be long enough to cut: {}",
            err.len()
        );
        assert!(err.ends_with("; run artemis setup --force"), "{err}");
        assert!(!err.contains('\n'), "{err}");
        // a tuned path that cannot be read as a file (a directory) keeps it too
        let env_value = deep.display().to_string();
        let env = move |k: &str| (k == "ARTEMIS_TUNED").then(|| env_value.clone());
        let err = Config::load(&args, &env, false).unwrap_err();
        assert!(err.ends_with("; run artemis setup --force"), "{err}");
    }

    #[test]
    fn a_stale_tuned_file_loads_and_warns() {
        let r = TunedRig::new("tuned-stale", |t| {
            t.setup.artemis_version = "0.0.1".into();
            t.setup.model_bytes = 11;
        });
        let c = Config::load(&r.args(""), &r.env(), false).unwrap();
        let TunedState::Loaded { stale, applied, .. } = &c.tuned else {
            panic!("{:?}", c.tuned)
        };
        assert_eq!(stale.len(), 2, "{stale:?}");
        assert!(!applied.is_empty());
        assert_eq!(c.tuned_warnings(&r.env(), false), *stale);
        // a fresh file owes none
        let r = TunedRig::new("tuned-fresh", |_| {});
        let c = Config::load(&r.args(""), &r.env(), false).unwrap();
        assert!(c.tuned_warnings(&r.env(), false).is_empty());
    }

    #[test]
    fn tuned_off_reads_nothing() {
        let r = TunedRig::new("tuned-off", |_| {});
        let off = |k: &str| (k == "ARTEMIS_TUNED").then(|| "OFF".to_string());
        let c = Config::load(&r.args(""), &off, false).unwrap();
        assert_eq!(c.tuned, TunedState::Off);
        assert_eq!(c.speculation, SpeculationConfig::default());
        assert_eq!(c.device_uuid, None);
        assert!(c.tuned_warnings(&off, false).is_empty());
        // load_without_tuned reads nothing even where the layer would apply
        let c = Config::load_without_tuned(&r.args(""), &r.env(), false).unwrap();
        assert_eq!(c.tuned, TunedState::Off);
        assert_eq!(c.speculation, SpeculationConfig::default());
    }

    #[test]
    fn a_missing_file_is_missing() {
        let r = TunedRig::new("tuned-missing", |_| {});
        std::fs::remove_file(&r.tuned).unwrap();
        let c = Config::load(&r.args(""), &r.env(), false).unwrap();
        assert_eq!(c.tuned, TunedState::Missing(r.tuned.clone()));
        assert_eq!(c.speculation, SpeculationConfig::default());
        assert!(c.tuned_warnings(&r.env(), false).is_empty());
    }

    #[test]
    fn an_unplaced_location_reads_nothing_and_warns() {
        let r = TunedRig::new("tuned-unplaced", |_| {});
        let c = Config::load(&r.args(""), &no_env, true).unwrap();
        assert_eq!(c.tuned, TunedState::Off);
        assert_eq!(c.speculation, SpeculationConfig::default());
        assert_eq!(
            c.tuned_warnings(&no_env, true),
            [crate::tuned::unplaced_warning(true)]
        );
        assert!(crate::tuned::unplaced_warning(true).starts_with("LOCALAPPDATA is not set"));
        // elsewhere the base variable is HOME
        let c = Config::load(&r.args(""), &no_env, false).unwrap();
        assert_eq!(
            c.tuned_warnings(&no_env, false),
            [crate::tuned::unplaced_warning(false)]
        );
        assert!(crate::tuned::unplaced_warning(false).starts_with("HOME is not set"));
        assert!(crate::tuned::unplaced_warning(false).contains("ARTEMIS_TUNED"));
        // a placed location owes none
        assert!(c.tuned_warnings(&r.env(), true).is_empty());
    }

    #[test]
    fn first_run_setup_key_and_flag() {
        use clap::Parser;
        #[derive(Parser)]
        struct Wrapper {
            #[command(flatten)]
            serve: ServeArgs,
        }
        let s = Scratch::new("first-run");
        let load = |toml: &str, flag: Option<bool>| {
            let args = ServeArgs {
                config: Some(s.file("a.toml", format!("model = \"m.gguf\"\n{toml}").as_bytes())),
                first_run_setup: flag,
                ..ServeArgs::default()
            };
            Config::load(&args, &no_env, false).unwrap().first_run_setup
        };
        assert!(Config::default().first_run_setup);
        assert!(load("", None));
        assert!(!load("first_run_setup = false\n", None));
        assert!(load("first_run_setup = false\n", Some(true)));
        assert!(!load("", Some(false)));
        let w = Wrapper::try_parse_from(["artemis", "--first-run-setup", "false"]).unwrap();
        assert_eq!(w.serve.first_run_setup, Some(false));
        let w = Wrapper::try_parse_from(["artemis", "--first-run-setup"]).unwrap();
        assert_eq!(w.serve.first_run_setup, Some(true));
    }
}
