//! The seam: what the server asks of the engine (`Request`) and what the engine
//! sends back (`Event`), through an `EventSink`. The server crate defines its own mirror of these types; `tools`'
//! adapter maps between them.

use crate::sample::Sampling;
use crate::spec::SpecDelta;
use std::time::Duration;

/// The tokens that end a think block when sampled: `</think>` and `<tool_call>`. The server fills `ThinkLimit::ends`
/// from it.
pub const THINK_ENDS: [u32; 2] = [248069, 248058];

/// The most top-N logprobs a request can ask for.
const MAX_LOGPROBS: u8 = 20;

/// One generation: a prompt, how to sample after it, and when to stop.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    /// The rendered prompt's tokens, generation prompt included.
    pub prompt: Vec<u32>,
    /// Where the prompt-end snapshot is taken, if the server asked for one.
    pub cut: Option<Cut>,
    /// How to sample each token.
    pub sampling: Sampling,
    /// Seeds the sampler's generator: from the request, or from the clock and a request counter.
    pub seed: u64,
    /// The most tokens to generate (forced tokens count), already clamped by the server to `ctx - prompt.len()`.
    pub max_tokens: u32,
    /// Tokens that end the generation when sampled. The stop token is counted but never sent as a `Token` event.
    pub stop_tokens: Vec<u32>,
    /// The think-block limit, when thinking is on.
    pub think: Option<ThinkLimit>,
    /// Top-N logprobs per sampled token, 0 to 20, when asked.
    pub logprobs: Option<u8>,
}

/// The snapshot point of a prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cut {
    /// The token index where the generation-prompt suffix begins: where the prompt-end snapshot is taken.
    pub at: u32,
    /// Whether the prompt's last message is the template's last real user query.
    pub user_query: bool,
}

/// How the generation loop limits the think block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThinkLimit {
    /// Whether the prompt ends inside an open think block.
    pub open: bool,
    /// The most sampled tokens the block may hold before `close` is forced; `None` is no limit.
    pub budget: Option<u32>,
    /// Sampling one of these closes the block (`THINK_ENDS`).
    pub ends: Vec<u32>,
    /// The tokens forced when the budget runs out: `\n</think>\n\n`.
    pub close: Vec<u32>,
}

impl Request {
    /// Refuses a request the engine can't run, with a one-line message naming the field. `ctx` is the session's
    /// positions and `n_vocab` the vocabulary's size. The server validates first, so a failure here is its bug.
    pub fn check(&self, ctx: usize, n_vocab: usize) -> Result<(), String> {
        if self.prompt.is_empty() {
            return Err("prompt is empty".into());
        }
        if self.prompt.len() > ctx {
            return Err(format!(
                "prompt has {} tokens; the context holds {ctx}",
                self.prompt.len()
            ));
        }
        let mut lists: Vec<(&str, &[u32])> =
            vec![("prompt", &self.prompt), ("stop_tokens", &self.stop_tokens)];
        if let Some(t) = &self.think {
            lists.push(("think.ends", &t.ends));
            lists.push(("think.close", &t.close));
        }
        for (field, tokens) in lists {
            if let Some(t) = tokens.iter().find(|&&t| t as usize >= n_vocab) {
                return Err(format!(
                    "{field} holds {t}, past the vocabulary's {n_vocab}"
                ));
            }
        }
        if let Some(cut) = self.cut {
            if cut.at as usize > self.prompt.len() {
                return Err(format!(
                    "cut.at {} is past the prompt's {} tokens",
                    cut.at,
                    self.prompt.len()
                ));
            }
        }
        if let Some(n) = self.logprobs {
            if n > MAX_LOGPROBS {
                return Err(format!("logprobs asks for {n}; at most {MAX_LOGPROBS}"));
            }
        }
        self.sampling.check()
    }
}

/// What the worker sends for a request. A request's events end with `Done` or `Error`: `Done` follows every ending
/// that isn't an `Error`. A `Done { finish: Unloaded }` can come before `Started`.
#[derive(Clone, Debug, PartialEq)]
// `Done` is sent once a request and `Token` once a token; boxing its fields would change every pattern that names them.
#[allow(clippy::large_enum_variant)]
pub enum Event {
    /// The worker took the request, with the model loaded and the restore chosen.
    Started {
        /// The prompt's length in tokens.
        prompt_tokens: u32,
        /// The tokens of the prompt the restore covers: not prefilled.
        reused_tokens: u32,
        /// Where the restored state came from.
        restored: Source,
        /// The longest common prefix of the prompt with any snapshot or the live history, before the restore point
        /// rounds it down.
        diverged_at: u32,
    },
    /// One per generated or forced token. A stop token is never sent.
    Token {
        id: u32,
        /// Forced by the think budget rather than sampled.
        forced: bool,
        /// `None` for forced tokens, and when logprobs weren't asked for.
        logprobs: Option<TokenLogprobs>,
    },
    /// The request is over, however it ended, unless by an error.
    Done {
        finish: Finish,
        counts: Counts,
        timings: Timings,
        cache: CacheDelta,
        /// Bytes copied back into the live state by the restore.
        restore_bytes: u64,
        /// The snapshot pool's bytes in use as of the end of the request's prefill and generation; the reply-end
        /// snapshot follows `Done`.
        pool_bytes_used: u64,
        /// What speculation did for this request: `None` with speculation off, and when the request ran no
        /// decode or verify step (not zeros).
        spec: Option<SpecDelta>,
    },
    /// The request failed. No `Done` follows.
    Error { kind: ErrorKind, message: String },
}

/// Where a restore's state came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Nothing was restored: the prompt was prefilled from 0.
    None,
    /// The live state, still holding the previous request's history.
    Live,
    /// An interval snapshot.
    Interval,
    /// A prompt-end snapshot.
    PromptEnd,
    /// A user-query snapshot.
    UserQuery,
    /// A reply-end snapshot.
    ReplyEnd,
}

/// How a generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// A stop token was sampled; its id.
    Stop(u32),
    /// `max_tokens` tokens were generated.
    Length,
    /// The session ran out of positions.
    ContextFull,
    /// The sink closed, or the request was cancelled.
    Cancelled,
    /// The model was unloaded under the request.
    Unloaded,
}

/// A request's token counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// The prompt's tokens.
    pub prompt: u32,
    /// The prompt's tokens the restore covered.
    pub reused: u32,
    /// The prompt's tokens prefilled.
    pub prefilled: u32,
    /// Every generated or forced token, the sampled stop token included.
    pub completion: u32,
    /// Advisory: tokens sampled while the think block was open, forced tokens excluded. The server's own count is
    /// what the API reports.
    pub reasoning: u32,
}

/// A request's wall times.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timings {
    /// Queued to dequeued, excluding the load and the restore.
    pub queue: Duration,
    /// Loading the model, when the request had to.
    pub load: Duration,
    /// Copying the restored state back into the live state.
    pub restore: Duration,
    /// Prefilling the rest of the prompt.
    pub prefill: Duration,
    /// Generating, from the end of the prefill to the end of the generation.
    pub decode: Duration,
    /// Queued to the first `Token`.
    pub first_token: Duration,
}

/// The expert cache's counters over one request.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CacheDelta {
    pub hits: u64,
    pub staged: u64,
    pub misses: u64,
    pub decode_misses: u64,
    pub decoded: u64,
    /// `None` when nothing was routed.
    pub hit_rate: Option<f64>,
    /// `None` when nothing was decoded; logged as `null`.
    pub miss_mib_per_token: Option<f64>,
}

/// A sampled token's log-probability and the top-N alternatives, from the model's own distribution (before penalty,
/// temperature and filters).
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprobs {
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
}

/// What an `Event::Error` reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    DeviceLost,
    NonFinite,
    OutOfMemory,
    LoadFailed,
    Internal,
}

/// The worker's lifecycle state, kept in an atomic as its `u8`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Lifecycle {
    Idle = 0,
    Loading = 1,
    Ready = 2,
    Recovering = 3,
}

impl Lifecycle {
    /// The state `v` stands for; out of range is `Idle`.
    pub fn from_u8(v: u8) -> Lifecycle {
        match v {
            1 => Lifecycle::Loading,
            2 => Lifecycle::Ready,
            3 => Lifecycle::Recovering,
            _ => Lifecycle::Idle,
        }
    }
}

/// Where a request's events go.
pub trait EventSink: Send {
    /// Delivers `event`; false when the receiver is gone.
    fn send(&mut self, event: Event) -> bool;
    /// Whether the receiver is known to be gone, so the worker can stop a request without sending.
    fn closed(&self) -> bool;
}

/// A sink over a std channel, for `tools` and the GPU tests. std's `Sender` can't tell that its receiver is gone
/// except by a failed send, so `closed` is always false.
#[derive(Debug)]
pub struct ChannelSink(pub std::sync::mpsc::Sender<Event>);

impl EventSink for ChannelSink {
    fn send(&mut self, event: Event) -> bool {
        self.0.send(event).is_ok()
    }

    fn closed(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_is_checked_field_by_field() {
        let ok = Request {
            prompt: vec![1, 2, 3],
            cut: Some(Cut {
                at: 2,
                user_query: true,
            }),
            sampling: Sampling::DEFAULT,
            seed: 7,
            max_tokens: 10,
            stop_tokens: vec![248_046],
            think: None,
            logprobs: Some(20),
        };
        assert_eq!(ok.check(65_536, 248_320), Ok(()));
        let bad = |f: fn(&mut Request)| {
            let mut r = ok.clone();
            f(&mut r);
            r.check(65_536, 248_320).unwrap_err()
        };
        assert!(bad(|r| r.prompt.clear()).contains("prompt is empty"));
        assert!(bad(|r| r.prompt = vec![1; 70_000]).contains("70000"));
        assert!(bad(|r| r.prompt = vec![999_999]).contains("prompt holds 999999"));
        assert!(bad(|r| r.cut = Some(Cut {
            at: 4,
            user_query: false
        }))
        .contains("cut.at 4"));
        assert!(bad(|r| r.logprobs = Some(21)).contains("21"));
        assert!(bad(|r| r.stop_tokens = vec![999_999]).contains("stop_tokens"));
        assert!(bad(|r| r.sampling.temp = -1.0).contains("--temp"));
        assert!(bad(|r| r.think = Some(ThinkLimit {
            open: true,
            budget: None,
            ends: THINK_ENDS.to_vec(),
            close: vec![248_320],
        }))
        .contains("think.close"));
        // A cut at the prompt's end is allowed.
        let mut r = ok.clone();
        r.cut = Some(Cut {
            at: 3,
            user_query: false,
        });
        assert_eq!(r.check(65_536, 248_320), Ok(()));
    }

    #[test]
    fn lifecycle_round_trips_through_u8() {
        for l in [
            Lifecycle::Idle,
            Lifecycle::Loading,
            Lifecycle::Ready,
            Lifecycle::Recovering,
        ] {
            assert_eq!(Lifecycle::from_u8(l as u8), l);
        }
        assert_eq!(Lifecycle::from_u8(9), Lifecycle::Idle);
    }

    #[test]
    fn a_channel_sink_delivers_until_its_receiver_is_dropped() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut sink = ChannelSink(tx);
        assert!(!sink.closed());
        let e = Event::Token {
            id: 5,
            forced: false,
            logprobs: None,
        };
        assert!(sink.send(e.clone()));
        assert_eq!(rx.recv().unwrap(), e);
        drop(rx);
        assert!(!sink.send(e));
    }
}
