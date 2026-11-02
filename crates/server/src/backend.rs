//! The seam between the server and the engine. `GenRequest` and `GenEvent` mirror the engine's `Request` and
//! `Event` field for field; the `tools` crate adapts the real worker to `Backend`, and `fake::FakeBackend` plays it for
//! the tests.
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};

/// Where the engine may snapshot the prompt's KV state: the token index of the final `<|im_start|>`, and whether the
/// message at the template's last query ends the prompt, with only tool results after it.
#[derive(Clone, Debug, PartialEq)]
pub struct Cut {
    pub at: u32,
    pub user_query: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sampling {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub presence_penalty: f32,
}

impl Sampling {
    /// The settings' ranges, as `engine::Sampling::check` has them. The error is the request field's name and what is
    /// wrong with it: the request rules (`params.rs`) and the config's `[sampling]` table (`config.rs`) share this one
    /// copy. `top_k` is unsigned, so it has no range to break.
    pub fn check(&self) -> Result<(), (&'static str, &'static str)> {
        if !(self.temp.is_finite() && self.temp >= 0.0) {
            return Err(("temperature", "must be a finite number, 0 or more"));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(("top_p", "must be above 0 and at most 1"));
        }
        if !(self.min_p >= 0.0 && self.min_p <= 1.0) {
            return Err(("min_p", "must be from 0 to 1"));
        }
        if !self.presence_penalty.is_finite() {
            return Err(("presence_penalty", "must be a finite number"));
        }
        Ok(())
    }
}

/// The reasoning budget. While `open` holds the engine counts generated tokens; sampling one of
/// `ends` clears it, and reaching `budget` with it still set forces `close`.
#[derive(Clone, Debug, PartialEq)]
pub struct ThinkLimit {
    pub open: bool,
    pub budget: Option<u32>,
    pub ends: [u32; 2],
    pub close: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenRequest {
    pub prompt: Vec<u32>,
    pub cut: Option<Cut>,
    pub sampling: Sampling,
    pub seed: u64,
    pub max_tokens: u32,
    pub stop_tokens: Vec<u32>,
    pub think: Option<ThinkLimit>,
    pub logprobs: Option<u8>,
}

/// Which snapshot a request restored from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Source {
    #[default]
    None,
    Live,
    Interval,
    PromptEnd,
    UserQuery,
    ReplyEnd,
}

impl Source {
    /// The name the request log gives it.
    pub fn name(self) -> &'static str {
        match self {
            Source::None => "none",
            Source::Live => "live",
            Source::Interval => "interval",
            Source::PromptEnd => "prompt_end",
            Source::UserQuery => "user_query",
            Source::ReplyEnd => "reply_end",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprobs {
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    Stop(u32),
    Length,
    ContextFull,
    Cancelled,
    Unloaded,
}

/// `completion` counts every sampled or forced token, the sampled stop token included. `reasoning` is the engine's
/// advisory count: the API reports the server's own (`Summary::reasoning_tokens`).
#[derive(Clone, Debug, PartialEq)]
pub struct Counts {
    pub prompt: u32,
    pub reused: u32,
    pub prefilled: u32,
    pub completion: u32,
    pub reasoning: u32,
}

/// `queue_ms` is queued -> dequeued and excludes the load; `load_ms` is the lazy load, if any.
#[derive(Clone, Debug, PartialEq)]
pub struct Timings {
    pub queue_ms: f64,
    pub load_ms: f64,
    pub restore_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub ttft_ms: f64,
}

/// `None` when the engine has no figure (no decode steps); logged as `null`.
#[derive(Clone, Debug, PartialEq)]
pub struct CacheReport {
    pub hit_rate: Option<f64>,
    pub miss_mib_per_token: Option<f64>,
    pub restore_bytes: u64,
    pub pool_bytes_used: u64,
    /// The expert cache's misses during the decode steps. The engine's adapter always sends a figure (zero decode steps
    /// give 0); only the fake sends `None`.
    pub decode_misses: Option<u64>,
}

/// What speculation did for one request: counts only. `None` on the `Done` when speculation is off or no
/// decode step ran.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecReport {
    /// Verify steps (a draft of one token or more).
    pub steps: u64,
    /// Steps run without a draft.
    pub plain_steps: u64,
    /// Draft tokens checked, and the ones kept.
    pub drafted: u64,
    pub accepted: u64,
    /// Verify steps that kept fewer than all their drafts.
    pub replays: u64,
    /// `drafted` and `accepted` by source (they sum to those two): prompt lookup's, then the MTP block's.
    pub drafted_lookup: u64,
    pub accepted_lookup: u64,
    pub drafted_mtp: u64,
    pub accepted_mtp: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    DeviceLost,
    NonFinite,
    OutOfMemory,
    LoadFailed,
    Internal,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GenEvent {
    Started {
        prompt_tokens: u32,
        reused_tokens: u32,
        restored: Source,
        diverged_at: u32,
    },
    /// A forced token (the think budget's close tokens) always has `logprobs: None`.
    Token {
        id: u32,
        forced: bool,
        logprobs: Option<TokenLogprobs>,
    },
    /// A `Done` with no `Started` before it means the model was unloaded under the request (the server answers 503
    /// "the model was unloaded").
    Done {
        finish: Finish,
        counts: Counts,
        timings: Timings,
        cache: CacheReport,
        /// `None` with speculation off, or when the request ran no decode step.
        spec: Option<SpecReport>,
    },
    Error {
        kind: ErrorKind,
        message: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifecycle {
    Idle,
    Loading,
    Ready,
    Recovering,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackendError {
    pub kind: ErrorKind,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct BackendStats {
    pub snapshots_held: u64,
    pub pool_bytes_used: u64,
    pub pool_bytes_limit: u64,
    pub restores: u64,
    pub restore_bytes: u64,
    pub reused_tokens: u64,
    pub prefilled_tokens: u64,
    pub evicted_protected: u64,
    pub evicted_unprotected: u64,
    /// Unloads whose park of the snapshots failed (the model was unloaded anyway).
    pub park_errors: u64,
    /// Snapshot pool chunks the driver refused.
    pub chunks_refused: u64,
    /// Models loaded so far, reloads after a device loss included.
    pub loads: u64,
    pub device_losses: u64,
    /// Failures of the snapshot and budget check that follows a reply, which reach no request, and the commit faults
    /// that came after a burst had already ended (the reply was complete; the cached session was dropped).
    pub after_done_errors: u64,
    /// Snapshots parked in host memory while unloaded.
    pub parked_snapshots: u64,
    /// The expert cache's counters; `None` while no model is loaded.
    pub cache: Option<CacheCounters>,
    /// Speculation's counters since the last load; `None` while no model is loaded, and with speculation off.
    pub spec: Option<SpecCounters>,
}

/// The expert cache's counters the log and `/v1/stats` carry. They are per load: a reload starts them again.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct CacheCounters {
    pub slots_per_layer: u64,
    pub hits: u64,
    pub misses: u64,
    pub staged: u64,
    pub cpu_misses: u64,
    pub cpu_timeouts: u64,
    pub longest_submit_us: u64,
}

/// One size of step: `t` tokens through the model (`x` and `t - 1` drafts; 1 is a plain step). The means are per step
/// of that size: `mean_ms` is the wall time of the step, `mean_miss_mib` the MiB the expert cache read for it.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct StepSize {
    pub t: u32,
    pub steps: u64,
    pub mean_ms: f64,
    pub mean_miss_mib: f64,
    /// Positions kept by these steps.
    pub tokens: u64,
}

/// One `k` of the MTP draft pass: how many passes drafted `k` tokens and their mean time.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MtpDraftRow {
    pub k: u32,
    pub passes: u64,
    pub mean_ms: f64,
}

/// Speculation's counters: per load, as the cache's are. `accept_rate` is the controller's current
/// estimate of the chance a draft token is kept, by source ("lookup", and "mtp" at depth 1); it is not `accepted / drafted`,
/// which is the request line's `spec_accept_rate` and these counters' ratio. `by_size` lists the sizes with steps only.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SpecCounters {
    pub steps: u64,
    pub plain_steps: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub replays: u64,
    pub accept_rate: std::collections::BTreeMap<String, f64>,
    pub by_size: Vec<StepSize>,
    /// MTP's accept rate at each draft depth (depth 1 first); empty until MTP has run.
    pub mtp_depth_rates: Vec<f64>,
    /// The MTP draft pass by `k`, the values that ran only.
    pub mtp_draft: Vec<MtpDraftRow>,
}

/// What the watchdog reads: `ticks` moves whenever the engine does something, and `pending` counts the work
/// it was given and has not finished (requests until their sink is dropped, commands until their reply).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Progress {
    pub ticks: u64,
    pub pending: u32,
}

/// The last answer the engine gave to `stats`, and when it gave it.
#[derive(Clone, Debug, PartialEq)]
pub struct StatsCopy {
    pub stats: BackendStats,
    pub at: Instant,
}

pub type Reply<T> = oneshot::Receiver<Result<T, BackendError>>;

/// `submit` found the engine gone (its worker has ended). The server answers 500 "the engine stopped" and asks `serve`
/// to exit with `Exit::EngineGone`, as for a channel that closes before its `Done`. The HTTP queue's own fullness is
/// not this: it is `Queue::enter`'s, and a 503.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Full;

/// The engine's end of one request's event channel. `send` is synchronous and safe from any thread; it returns false
/// once the receiver is gone. `closed` is true once the receiver is gone or the server cancelled (a stop string, a
/// queue timeout, an unload of queued work): the engine then ends the request (`Done { Cancelled }`) or skips it.
#[derive(Debug)]
pub struct Sink {
    tx: mpsc::UnboundedSender<GenEvent>,
    cancelled: Arc<AtomicBool>,
}

impl Sink {
    pub fn channel() -> (Sink, mpsc::UnboundedReceiver<GenEvent>, CancelHandle) {
        let (tx, rx) = mpsc::unbounded_channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let handle = CancelHandle {
            cancelled: cancelled.clone(),
        };
        (Sink { tx, cancelled }, rx, handle)
    }

    pub fn send(&self, ev: GenEvent) -> bool {
        self.tx.send(ev).is_ok()
    }

    pub fn closed(&self) -> bool {
        self.cancelled.load(Ordering::Acquire) || self.tx.is_closed()
    }
}

/// The server's half: cancels the request without dropping its receiver, so the events up to `Done` still arrive.
#[derive(Clone, Debug)]
pub struct CancelHandle {
    cancelled: Arc<AtomicBool>,
}

impl CancelHandle {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// The engine as the server sees it. No method may block: they are called on tokio workers, `submit` and `load` under
/// the std `submit_order` mutex. One loss of the device is one `Error { DeviceLost }` to one sink or reply; the server
/// counts each it reads, and two within `loss_window` end the process, so an adapter must never fan one loss out.
pub trait Backend: Send + Sync + 'static {
    fn submit(&self, req: GenRequest, sink: Sink) -> Result<(), Full>;
    fn state(&self) -> Lifecycle;
    /// Loads now if idle; Ok at once if loaded.
    fn load(&self) -> Reply<()>;
    /// Ends the request in flight as `Finish::Unloaded`, then unloads; Ok at once if idle. The reply is whether a
    /// model was loaded and is now unloaded: `false` for an unload that found nothing loaded (one that lands just
    /// after an idle unload, say), which the server then does not log. Queued requests are the server's to answer (it
    /// cancels their sinks first, `Queue::abort_all`).
    fn unload(&self) -> Reply<bool>;
    /// The unload at the end of the process: same reply as `unload`. An engine that parks snapshots
    /// across an unload parks nothing here.
    fn shutdown(&self) -> Reply<bool> {
        self.unload()
    }
    fn stats(&self) -> Reply<BackendStats>;
    /// The engine's progress counter and its pending work, or `None` for a backend that keeps none. Never
    /// waits on the engine.
    fn progress(&self) -> Option<Progress> {
        None
    }
    /// The last answer to `stats`, kept whether or not its asker still waited, or `None` before the first.
    /// Never waits on the engine.
    fn latest_stats(&self) -> Option<StatsCopy> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev() -> GenEvent {
        GenEvent::Token {
            id: 1,
            forced: false,
            logprobs: None,
        }
    }

    #[test]
    fn closed_follows_cancel_and_drop() {
        let (sink, mut rx, cancel) = Sink::channel();
        assert!(!sink.closed() && !cancel.is_cancelled());
        assert!(sink.send(ev()));
        assert_eq!(rx.try_recv(), Ok(ev()));

        // A cancel closes the sink and is visible through every clone of the handle; the receiver still works, so
        // the events up to `Done` still arrive.
        let clone = cancel.clone();
        clone.cancel();
        assert!(sink.closed() && cancel.is_cancelled());
        assert!(sink.send(ev()), "a cancelled sink still delivers");
        assert_eq!(rx.try_recv(), Ok(ev()));

        // Dropping the receiver closes a sink that was never cancelled, and `send` then says so.
        let (sink, rx, cancel) = Sink::channel();
        drop(rx);
        assert!(sink.closed() && !cancel.is_cancelled());
        assert!(!sink.send(ev()));
    }

    #[test]
    fn the_stats_serialize_under_their_field_names() {
        // `/v1/stats` and the `stats` line carry these names.
        let v = serde_json::to_value(BackendStats {
            loads: 2,
            cache: Some(CacheCounters {
                cpu_timeouts: 3,
                ..CacheCounters::default()
            }),
            spec: Some(SpecCounters {
                steps: 5,
                plain_steps: 2,
                drafted: 12,
                accepted: 9,
                replays: 3,
                accept_rate: [("lookup".to_string(), 0.75), ("mtp".to_string(), 0.7)].into(),
                by_size: vec![StepSize {
                    t: 3,
                    steps: 5,
                    mean_ms: 21.5,
                    mean_miss_mib: 40.0,
                    tokens: 11,
                }],
                mtp_depth_rates: vec![0.7, 0.6],
                mtp_draft: vec![MtpDraftRow {
                    k: 2,
                    passes: 4,
                    mean_ms: 3.5,
                }],
            }),
            ..BackendStats::default()
        })
        .unwrap();
        for key in [
            "snapshots_held",
            "pool_bytes_used",
            "park_errors",
            "chunks_refused",
            "loads",
            "device_losses",
            "after_done_errors",
            "parked_snapshots",
            "cache",
        ] {
            assert!(v.get(key).is_some(), "{key}: {v}");
        }
        assert_eq!(v["loads"], 2);
        assert_eq!(v["cache"]["cpu_timeouts"], 3);
        for key in [
            "slots_per_layer",
            "hits",
            "misses",
            "staged",
            "cpu_misses",
            "longest_submit_us",
        ] {
            assert!(v["cache"].get(key).is_some(), "{key}: {v}");
        }
        assert!(v.get("spec").is_some(), "spec: {v}");
        for key in [
            "steps",
            "plain_steps",
            "drafted",
            "accepted",
            "replays",
            "accept_rate",
            "by_size",
            "mtp_depth_rates",
            "mtp_draft",
        ] {
            assert!(v["spec"].get(key).is_some(), "{key}: {v}");
        }
        assert_eq!(
            v["spec"]["accept_rate"],
            serde_json::json!({"lookup": 0.75, "mtp": 0.7})
        );
        assert_eq!(v["spec"]["mtp_depth_rates"], serde_json::json!([0.7, 0.6]));
        assert_eq!(
            v["spec"]["mtp_draft"],
            serde_json::json!([{"k": 2, "passes": 4, "mean_ms": 3.5}])
        );
        let row = &v["spec"]["by_size"][0];
        for key in ["t", "steps", "mean_ms", "mean_miss_mib", "tokens"] {
            assert!(row.get(key).is_some(), "{key}: {v}");
        }
        assert_eq!(row["t"], 3);
        let unloaded = serde_json::to_value(BackendStats::default()).unwrap();
        assert!(unloaded["cache"].is_null() && unloaded["spec"].is_null());
        let p = serde_json::to_value(Progress {
            ticks: 5,
            pending: 1,
        })
        .unwrap();
        assert_eq!(p, serde_json::json!({"ticks": 5, "pending": 1}));
    }

    /// A backend that records what `unload` was asked, to see the default `shutdown` reach it.
    struct Counting(std::sync::atomic::AtomicUsize);

    impl Backend for Counting {
        fn submit(&self, _req: GenRequest, _sink: Sink) -> Result<(), Full> {
            Err(Full)
        }
        fn state(&self) -> Lifecycle {
            Lifecycle::Ready
        }
        fn load(&self) -> Reply<()> {
            unreachable!("not asked")
        }
        fn unload(&self) -> Reply<bool> {
            self.0.fetch_add(1, Ordering::AcqRel);
            let (tx, rx) = oneshot::channel();
            let _ = tx.send(Ok(true));
            rx
        }
        fn stats(&self) -> Reply<BackendStats> {
            unreachable!("not asked")
        }
    }

    #[tokio::test]
    async fn the_default_shutdown_is_an_unload() {
        let b = Counting(std::sync::atomic::AtomicUsize::new(0));
        assert_eq!(b.shutdown().await.unwrap(), Ok(true));
        assert_eq!(b.0.load(Ordering::Acquire), 1);
    }
}
