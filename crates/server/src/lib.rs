//! The Artemis server: the OpenAI- and Anthropic-compatible HTTP API over an engine behind `Backend`.
//! A request is read into a conversation, rendered through the chat template into a prompt, queued and handed to the
//! backend, and its tokens come back through the output pipeline (reasoning, tool calls, stop strings) as a whole or a
//! streamed reply. It also holds the config, the logs and auth, `/health`, the lifecycle routes and a fake engine for
//! tests. It depends on `qwen36` and never on `engine`: only the fake implements `Backend` here, and the real
//! engine's adapter lives in the `tools` crate.
pub mod anthropic;
pub mod app;
pub mod backend;
pub mod config;
pub mod convo;
pub mod error;
pub mod exchange;
pub mod fake;
pub mod handlers;
pub mod ids;
pub mod jsonlimit;
pub mod keyfile;
mod listen;
pub mod logs;
pub mod openai;
pub mod params;
pub mod pipeline;
pub mod prompt;
pub mod pyfmt;
pub mod queue;
pub mod schema;
pub mod template;
pub mod tokens;
pub mod tuned;
pub mod watchdog;

pub use app::{runtime, serve, App, AppParts, Exit, Tuning};
pub use backend::*;
pub use config::{key_strength, Config, ServeArgs, TunedState, MIN_KEY_CHARS};
