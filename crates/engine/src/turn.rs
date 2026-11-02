//! The generation loop: one request's restore, segmented prefill with its snapshots, and the
//! sampling loop with its think budget, stop tokens and logprobs, over the `Live` trait the worker drives. The real
//! engine implements `Live`; `fake::FakeLive` scripts it for the tests.
use crate::cache::Stats;
use crate::logprobs::logprobs;
use crate::request::{CacheDelta, Counts, Cut, Event, EventSink, Finish, Request, Source, Timings};
use crate::reuse::{split_points, Marks, ReuseStats};
use crate::sample::Sampler;
use crate::spec::{Next, SpecDelta, SpecState, Verifier};
use crate::Error;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// The engine as the generation loop sees it: a live state (the KV and DeltaNet state of a token history) over a
/// snapshot pool.
pub trait Live {
    fn ctx(&self) -> usize;
    fn pos(&self) -> usize;
    /// Chooses and performs the restore for `prompt`: the live state, a snapshot, or a reset to 0.
    fn restore_for(&mut self, prompt: &[u32]) -> Result<Restored, Error>;
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, Error>;
    fn decode(&mut self, token: u32) -> Result<Vec<f32>, Error>;
    /// Runs `tokens = [x, d1 .. dk]` at the next positions as if all were kept, and leaves the step pending until
    /// `commit`. The rows are read with `row`.
    fn verify(&mut self, tokens: &[u32]) -> Result<(), Error>;
    /// Row `i` of the pending verify: the logits after `tokens[i]`, as `decode` would return them at that position.
    fn row(&mut self, i: usize) -> Result<Vec<f32>, Error>;
    /// Keeps `tokens[..kept]` of the pending verify. After an error from `verify`, `row` or `commit` the live state
    /// has failed the step itself (cut back to its start, unusable until a reset or a restore).
    fn commit(&mut self, kept: usize) -> Result<(), Error>;
    /// The vocabulary's size: a draft token at or past it is never verified.
    fn n_vocab(&self) -> usize;
    /// MTP's drafts after `x` (`Verifier::mtp_draft`): up to `k` of them, none when the state can't draft. An error
    /// has failed the step the way a failed write does (the live state is unusable unless the error left it usable).
    /// Default: no drafts.
    fn mtp_draft(&mut self, x: u32, k: usize) -> Result<Vec<u32>, Error> {
        let _ = (x, k);
        Ok(Vec::new())
    }
    /// Whether the live state can draft with MTP (the load allocated it). Default: no.
    fn has_mtp(&self) -> bool {
        false
    }
    /// The most tokens one `verify` takes (`Verifier::most`): the step loop drafts at most one fewer.
    fn most_verify(&self) -> usize {
        crate::spec::MAX_VERIFY
    }
    /// A snapshot of the live state at its position. A skip is `Ok`.
    fn snapshot(&mut self, marks: Marks) -> Result<(), Error>;
    fn check_budget(&mut self) -> Result<(), Error>;
    fn cache_stats(&self) -> Stats;
    fn expert_bytes(&self) -> u64;
    fn reuse_stats(&self) -> ReuseStats;
    /// After an error the client never sees (the reply-end snapshot or the budget check): the live state can't be
    /// trusted until a reset or a restore.
    fn mark_unusable(&mut self);
}

/// What a restore did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Restored {
    pub source: Source,
    /// Where the prefill resumes: the tokens of the prompt the restore covers.
    pub pos: usize,
    /// The longest common prefix of the prompt with any snapshot or the live history.
    pub diverged_at: usize,
    /// Bytes copied into the live state.
    pub bytes: u64,
}

/// How a turn ended, after its last event was sent.
#[derive(Debug)]
pub enum Ending {
    Done(Finish),
    /// `Done` was sent, but settling the last burst failed: the commit's replay faulted after the burst had already ended
    /// (a stop, `max_tokens`, a cancel or an unload). Every token the client streamed is right; only the live state is
    /// lost, and it is already failed. The worker marks it unusable and counts the fault.
    Unsettled(Finish, Error),
    Failed(Error),
}

/// Prefills `prompt[from..]` in segments ending at every split point, with a snapshot after each point
/// that has marks. `stop` runs before each segment: `Some(f)` ends the prefill early with `Err(f)`. `Ok(Ok(logits))` is
/// the last segment's. The live state must be at `from`.
pub fn prefill_points<L: Live>(
    live: &mut L,
    prompt: &[u32],
    from: usize,
    cut: Option<Cut>,
    mut stop: impl FnMut() -> Option<Finish>,
) -> Result<Result<Vec<f32>, Finish>, Error> {
    let mut at = from;
    let mut last = None;
    for (end, marks) in split_points(from, prompt.len(), cut) {
        if let Some(f) = stop() {
            return Ok(Err(f));
        }
        last = Some(live.prefill(&prompt[at..end])?);
        at = end;
        if !marks.is_empty() {
            live.snapshot(marks)?;
        }
    }
    // No segment means `from` is already the prompt's end: the restore left nothing to prefill, which the choice
    // (a proper prefix only) never does.
    last.map(Ok).ok_or(Error::EmptyPrefill)
}

/// The expert cache's counters over one request: `hits`, `staged` and `misses` from the whole request, `decode_misses`
/// from the decode steps after the prefill.
pub fn cache_delta(
    start: &Stats,
    after_prefill: &Stats,
    end: &Stats,
    expert_bytes: u64,
    decoded: u64,
) -> CacheDelta {
    let whole = Stats {
        hits: end.hits.saturating_sub(start.hits),
        staged: end.staged.saturating_sub(start.staged),
        misses: end.misses.saturating_sub(start.misses),
        ..Stats::default()
    };
    let decode_misses = end.misses.saturating_sub(after_prefill.misses);
    CacheDelta {
        hits: whole.hits,
        staged: whole.staged,
        misses: whole.misses,
        decode_misses,
        decoded,
        hit_rate: whole.from_vram(),
        miss_mib_per_token: (decoded > 0)
            .then(|| decode_misses as f64 * expert_bytes as f64 / decoded as f64 / MIB),
    }
}

const MIB: f64 = (1u64 << 20) as f64;

/// A count as `u32`, saturating.
fn count(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// What cuts a turn short: a closed sink is a cancel, a pending unload an unload.
fn interrupted(sink: &dyn EventSink, unload_now: &AtomicUsize) -> Option<Finish> {
    if sink.closed() {
        Some(Finish::Cancelled)
    } else if unload_now.load(Ordering::SeqCst) > 0 {
        Some(Finish::Unloaded)
    } else {
        None
    }
}

/// The step loop's view of a `Live` (`spec::Verifier`): its decode, its verify with lazy rows and its commit, and the
/// expert cache's miss counter and the bytes a miss reads for the step table.
pub struct LiveVerifier<'a, L: Live>(pub &'a mut L);

impl<L: Live> Verifier for LiveVerifier<'_, L> {
    type Error = Error;

    fn decode(&mut self, token: u32) -> Result<Vec<f32>, Error> {
        self.0.decode(token)
    }

    fn verify(&mut self, tokens: &[u32]) -> Result<(), Error> {
        self.0.verify(tokens)
    }

    fn row(&mut self, i: usize) -> Result<Vec<f32>, Error> {
        self.0.row(i)
    }

    fn commit(&mut self, kept: usize) -> Result<(), Error> {
        self.0.commit(kept)
    }

    fn n_vocab(&self) -> usize {
        self.0.n_vocab()
    }

    fn most(&self) -> usize {
        self.0.most_verify()
    }

    fn mtp_draft(&mut self, x: u32, k: usize) -> Result<Vec<u32>, Error> {
        self.0.mtp_draft(x, k)
    }

    fn has_mtp(&self) -> bool {
        self.0.has_mtp()
    }

    fn misses(&self) -> u64 {
        self.0.cache_stats().misses
    }

    fn miss_bytes(&self) -> u64 {
        self.0.expert_bytes()
    }
}

/// What a turn has measured so far, for the `Done` it ends with.
#[derive(Default)]
struct Tally {
    restored: Option<Restored>,
    restore: Duration,
    prefill: Duration,
    /// When the prefill ended: the decode's clock starts here.
    decode_from: Option<Instant>,
    after_prefill: Option<Stats>,
    first_token: Option<Duration>,
    completion: u32,
    reasoning: u32,
    /// Positions committed by decode and verify steps: one a plain step, `kept` a verify step.
    decoded: u64,
    /// What speculation did, once a step has run; `None` with speculation off or before the first step.
    spec: Option<SpecDelta>,
    /// A commit fault after the burst had ended: the turn finishes as if the commit had worked (`Ending::Unsettled`).
    late: Option<Error>,
}

/// Runs one request on a live state whose restore, snapshots and lifecycle the caller owns. Sends `Started`, every
/// `Token` and `Done` or `Error`. `unload_now` is the pending-unload count: non-zero ends the turn `Unloaded`. With
/// `spec` the decode loop is `SpecState::step`; without it, today's decode calls and nothing else.
#[allow(clippy::too_many_arguments)] // a turn's inputs
pub fn run_turn<L: Live>(
    live: &mut L,
    req: &Request,
    sink: &mut dyn EventSink,
    unload_now: &AtomicUsize,
    queued: Instant,
    dequeued: Instant,
    load: Duration,
    spec: Option<&mut SpecState>,
) -> Ending {
    // The windows start where the caller's time ends (`load` runs from `dequeued` to this call), so
    // `queue + load + restore + prefill + decode` is exactly `done - queued`, which covers the first token's time from
    // `queued` whatever the scheduler does.
    let began = dequeued + load;
    let start = live.cache_stats();
    let mut tally = Tally::default();
    match generate(live, req, sink, unload_now, queued, began, spec, &mut tally) {
        Ok(finish) => {
            let end = live.cache_stats();
            let restored = tally
                .restored
                .expect("a turn that ends in Done restored first");
            let n = req.prompt.len();
            let counts = Counts {
                prompt: count(n),
                reused: count(restored.pos),
                prefilled: count(live.pos().min(n).saturating_sub(restored.pos)),
                completion: tally.completion,
                reasoning: tally.reasoning,
            };
            let timings = Timings {
                queue: dequeued.saturating_duration_since(queued),
                load,
                restore: tally.restore,
                prefill: tally.prefill,
                decode: tally.decode_from.map_or(Duration::ZERO, |t| t.elapsed()),
                first_token: tally.first_token.unwrap_or(Duration::ZERO),
            };
            let cache = cache_delta(
                &start,
                tally.after_prefill.as_ref().unwrap_or(&end),
                &end,
                live.expert_bytes(),
                tally.decoded,
            );
            sink.send(Event::Done {
                finish,
                counts,
                timings,
                cache,
                restore_bytes: restored.bytes,
                pool_bytes_used: live.reuse_stats().pool_bytes_used,
                spec: tally.spec,
            });
            match tally.late.take() {
                Some(e) => Ending::Unsettled(finish, e),
                None => Ending::Done(finish),
            }
        }
        Err(e) => {
            sink.send(Event::Error {
                kind: e.kind().event_kind(),
                message: e.to_string(),
            });
            Ending::Failed(e)
        }
    }
}

/// The per-position rules of the turn: what happens to the logits after one token, wherever they come
/// from (the prefill, a decode, a row of a verify). It samples (or takes the think budget's forced token), ends on a
/// stop token, sends, and ends on a failed send, a closed sink, an unload, `max_tokens` or a full context, so a burst
/// of rows is cut exactly where the plain loop would have stopped.
struct Picker<'a> {
    req: &'a Request,
    unload_now: &'a AtomicUsize,
    queued: Instant,
    ctx: usize,
    /// A fresh sampler per request; forced tokens never reach it.
    sampler: Sampler,
    open: bool,
    pending: VecDeque<u32>,
    /// The state position the next logits are the row of: the plain loop's `live.pos()` at that point. A burst's rows
    /// are at its own next positions, not at `live.pos()`, which already counts the whole burst after a verify.
    at: usize,
    /// The token the last `Go` picked: the next step's `x`.
    last: u32,
    /// How the turn ended, once a pick returned `End`.
    finish: Option<Finish>,
}

impl Picker<'_> {
    fn end(&mut self, finish: Finish) -> Next {
        self.finish = Some(finish);
        Next::End
    }

    /// One position's pick, in the plain loop's order. `logits` are the row after the state position `self.at`.
    fn pick(&mut self, logits: &[f32], sink: &mut dyn EventSink, t: &mut Tally) -> Next {
        let req = self.req;
        let limit = req.think.as_ref();
        let at = self.at;
        self.at += 1;
        if t.completion >= req.max_tokens {
            return self.end(Finish::Length);
        }
        if let Some(l) = limit.filter(|_| self.open && self.pending.is_empty()) {
            if l.budget.is_some_and(|b| t.reasoning >= b) {
                self.pending.extend(&l.close);
                self.open = false;
            }
        }
        let (token, forced) = match self.pending.pop_front() {
            Some(token) => (token, true),
            None => (self.sampler.sample(logits), false),
        };
        t.completion = t.completion.saturating_add(1);
        if !forced {
            if req.stop_tokens.contains(&token) {
                return self.end(Finish::Stop(token));
            }
            if let Some(l) = limit.filter(|_| self.open) {
                if l.ends.contains(&token) {
                    self.open = false;
                } else {
                    t.reasoning = t.reasoning.saturating_add(1);
                }
            }
        }
        let lp = req
            .logprobs
            .filter(|_| !forced)
            .map(|top| logprobs(logits, token, top));
        let sent = sink.send(Event::Token {
            id: token,
            forced,
            logprobs: lp,
        });
        if sent && t.first_token.is_none() {
            t.first_token = Some(self.queued.elapsed());
        }
        if !sent || sink.closed() {
            return self.end(Finish::Cancelled);
        }
        if self.unload_now.load(Ordering::SeqCst) > 0 {
            return self.end(Finish::Unloaded);
        }
        if t.completion >= req.max_tokens {
            return self.end(Finish::Length);
        }
        if at >= self.ctx {
            return self.end(Finish::ContextFull);
        }
        self.last = token;
        Next::Go(token)
    }

    /// The finish after an `End`.
    fn finished(&self) -> Finish {
        self.finish.expect("a pick returned End")
    }
}

/// The turn's steps, mirroring `tools::generate::run`: restore, prefill, then per position `Picker::pick`
/// (stop at `max_tokens` before picking, pick, end on a sampled stop token, send, end on a failed send, a closed sink
/// or an unload, stop at `max_tokens` without decoding the last token, stop at a full context) and a decode of the
/// token it picked, or with `spec` a speculation step that verifies drafts after it and picks each row. So the
/// state is always one sampled token short of what the client saw.
#[allow(clippy::too_many_arguments)] // the turn's inputs, and `spec`
fn generate<L: Live>(
    live: &mut L,
    req: &Request,
    sink: &mut dyn EventSink,
    unload_now: &AtomicUsize,
    queued: Instant,
    began: Instant,
    spec: Option<&mut SpecState>,
    t: &mut Tally,
) -> Result<Finish, Error> {
    let restored = live.restore_for(&req.prompt)?;
    let restore_end = Instant::now();
    t.restore = restore_end.duration_since(began);
    t.restored = Some(restored);
    let started = Event::Started {
        prompt_tokens: count(req.prompt.len()),
        reused_tokens: count(restored.pos),
        restored: restored.source,
        diverged_at: count(restored.diverged_at),
    };
    if !sink.send(started) {
        return Ok(Finish::Cancelled);
    }

    let prefilled = prefill_points(live, &req.prompt, restored.pos, req.cut, || {
        interrupted(&*sink, unload_now)
    });
    let prefill_end = Instant::now();
    t.prefill = prefill_end.duration_since(restore_end);
    let logits = match prefilled? {
        Ok(logits) => logits,
        Err(finish) => return Ok(finish),
    };
    t.decode_from = Some(prefill_end);
    t.after_prefill = Some(live.cache_stats());

    let mut picker = Picker {
        req,
        unload_now,
        queued,
        ctx: live.ctx(),
        sampler: Sampler::new(req.sampling, req.seed),
        open: req.think.as_ref().is_some_and(|l| l.open),
        pending: VecDeque::new(),
        at: live.pos(),
        last: 0,
        finish: None,
    };
    let mut x = match picker.pick(&logits, sink, t) {
        Next::Go(x) => x,
        Next::End => return Ok(picker.finished()),
    };
    let Some(spec) = spec else {
        loop {
            let logits = live.decode(x)?;
            t.decoded += 1;
            match picker.pick(&logits, sink, t) {
                Next::Go(next) => x = next,
                Next::End => return Ok(picker.finished()),
            }
        }
    };

    // The drafter is made at the first decode step and sees the prompt (a restored prefix is part of it) and `x`.
    let mut drafter = spec.drafter();
    drafter.observe(&req.prompt);
    drafter.observe(&[x]);
    let mut delta = SpecDelta::default();
    loop {
        // The step keeps at most the tokens `max_tokens` still allows, and its last position must fit the context.
        let room = (req.max_tokens.saturating_sub(t.completion) as usize).saturating_sub(1);
        let k_max = room.min(picker.ctx.saturating_sub(live.pos()).saturating_sub(1));
        let result = spec.step(
            &mut LiveVerifier(live),
            drafter.as_mut(),
            x,
            k_max,
            &mut |row| Ok(picker.pick(row, sink, t)),
            &mut delta,
        );
        let stepped = match result {
            Ok(stepped) => stepped,
            // The burst had ended already (verify and row faults come before any `End`, so this is the commit's replay):
            // the client has its whole answer, and only the cached state is lost. Finish normally and say so.
            Err(e) if picker.finish.is_some() => {
                t.late = Some(e);
                return Ok(picker.finished());
            }
            Err(e) => return Err(e),
        };
        t.decoded += stepped.kept as u64;
        t.spec = Some(delta);
        if stepped.ended {
            return Ok(picker.finished());
        }
        x = picker.last;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::{Call, FakeLive, TestSink};
    use crate::request::{ErrorKind, ThinkLimit};
    use crate::reuse::INTERVAL;
    use crate::sample::Sampling;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    /// A prompt of `n` tokens below the fake's vocabulary.
    fn prompt(n: usize) -> Vec<u32> {
        (0..n).map(|i| (i * 7 % 50) as u32).collect()
    }

    /// A greedy request for `prompt(n)` with stop token 63, 8 tokens and no cut.
    fn req(n: usize) -> Request {
        Request {
            prompt: prompt(n),
            cut: None,
            sampling: Sampling::GREEDY,
            seed: 1,
            max_tokens: 8,
            stop_tokens: vec![63],
            think: None,
            logprobs: None,
        }
    }

    /// The model picks `toks[i]` after `n + i` tokens of history, and 1 after that (and for shorter histories).
    fn seq(n: usize, toks: &[u32]) -> impl Fn(&[u32]) -> u32 + Send + 'static {
        let toks = toks.to_vec();
        move |h| {
            h.len()
                .checked_sub(n)
                .and_then(|i| toks.get(i))
                .copied()
                .unwrap_or(1)
        }
    }

    fn run(live: &mut FakeLive, req: &Request, sink: &mut TestSink) -> Ending {
        let now = Instant::now();
        run_turn(
            live,
            req,
            sink,
            &AtomicUsize::new(0),
            now,
            now,
            Duration::ZERO,
            None,
        )
    }

    fn done(sink: &TestSink) -> (Finish, Counts, Timings, CacheDelta, u64, u64) {
        match sink.events.last() {
            Some(Event::Done {
                finish,
                counts,
                timings,
                cache,
                restore_bytes,
                pool_bytes_used,
                spec: _,
            }) => (
                *finish,
                *counts,
                *timings,
                *cache,
                *restore_bytes,
                *pool_bytes_used,
            ),
            other => panic!("the events don't end with Done: {other:?}"),
        }
    }

    fn started(sink: &TestSink) -> (u32, u32, Source, u32) {
        match sink.events.first() {
            Some(Event::Started {
                prompt_tokens,
                reused_tokens,
                restored,
                diverged_at,
            }) => (*prompt_tokens, *reused_tokens, *restored, *diverged_at),
            other => panic!("the events don't start with Started: {other:?}"),
        }
    }

    fn forced_flags(sink: &TestSink) -> Vec<bool> {
        sink.events
            .iter()
            .filter_map(|e| match e {
                Event::Token { forced, .. } => Some(*forced),
                _ => None,
            })
            .collect()
    }

    fn decodes(live: &FakeLive) -> Vec<u32> {
        live.calls
            .iter()
            .filter_map(|c| match c {
                Call::Decode(t) => Some(*t),
                _ => None,
            })
            .collect()
    }

    fn think(budget: Option<u32>, close: &[u32]) -> Option<ThinkLimit> {
        Some(ThinkLimit {
            open: true,
            budget,
            ends: vec![62, 61],
            close: close.to_vec(),
        })
    }

    fn non_finite() -> Error {
        Error::NonFinite {
            pos: 0,
            index: 0,
            value: f32::NAN,
        }
    }

    #[test]
    fn prefill_splits_at_every_4096_and_at_the_cut_and_snapshots_each() {
        let mut live = FakeLive::new(10_000, |_| 1);
        let mut r = req(9000);
        r.cut = Some(Cut {
            at: 8995,
            user_query: true,
        });
        let mut sink = TestSink::new();
        assert!(matches!(run(&mut live, &r, &mut sink), Ending::Done(_)));
        assert_eq!(
            live.calls[..7],
            [
                Call::Prefill(0, 4096),
                Call::Snapshot(4096, Marks::INTERVAL),
                Call::Prefill(4096, 4096),
                Call::Snapshot(8192, Marks::INTERVAL),
                Call::Prefill(8192, 803),
                Call::Snapshot(8995, Marks::USER_QUERY),
                Call::Prefill(8995, 5),
            ]
        );
        assert!(live.calls[7..].iter().all(|c| matches!(c, Call::Decode(_))));
        assert_eq!(live.calls.len(), 7 + 7);
    }

    #[test]
    fn a_restore_prefills_only_the_rest() {
        let state_len = 50 + 2;
        for reset in [false, true] {
            let mut live = FakeLive::new(1000, |_| 1);
            let mut first = req(50);
            first.max_tokens = 3;
            run(&mut live, &first, &mut TestSink::new());
            let state = live.tokens().to_vec();
            assert_eq!(state.len(), state_len);
            // The worker's reply-end snapshot.
            live.snapshot(Marks::REPLY_END).unwrap();
            if reset {
                live.reset();
                // The tokens go; the KV record stays.
                assert!(live.tokens().is_empty());
                assert_eq!(live.kv_tokens(), state);
            }
            live.calls.clear();
            let mut second = req(0);
            second.prompt = [state.clone(), vec![9, 9, 9]].concat();
            let mut sink = TestSink::new();
            run(&mut live, &second, &mut sink);
            let want = if reset {
                Source::ReplyEnd
            } else {
                Source::Live
            };
            assert_eq!(
                started(&sink),
                (
                    state_len as u32 + 3,
                    state_len as u32,
                    want,
                    state_len as u32
                ),
                "reset: {reset}"
            );
            let prefills: Vec<_> = live
                .calls
                .iter()
                .filter(|c| matches!(c, Call::Prefill(..)))
                .collect();
            assert_eq!(prefills, [&Call::Prefill(state_len, 3)]);
            let restored = live
                .calls
                .contains(&Call::Restore(Source::ReplyEnd, state_len));
            assert_eq!(restored, reset);
            let (_, counts, _, _, restore_bytes, _) = done(&sink);
            assert_eq!((counts.reused, counts.prefilled), (state_len as u32, 3));
            assert_eq!(restore_bytes > 0, reset);
        }
    }

    #[test]
    fn an_exact_repeat_prefills_the_suffix_again() {
        let mut live = FakeLive::new(1000, |_| 1);
        let mut r = req(100);
        r.cut = Some(Cut {
            at: 90,
            user_query: true,
        });
        run(&mut live, &r, &mut TestSink::new());
        live.calls.clear();
        let mut sink = TestSink::new();
        // The fake refuses a zero-length prefill with `EmptyPrefill` before recording it, so `Done` (not `Failed`) is
        // what shows none was asked for.
        assert!(matches!(run(&mut live, &r, &mut sink), Ending::Done(_)));
        assert_eq!(live.calls[0], Call::Restore(Source::UserQuery, 90));
        assert_eq!(live.calls[1], Call::Prefill(90, 10));
        assert_eq!(started(&sink), (100, 90, Source::UserQuery, 100));
    }

    #[test]
    fn the_stop_token_is_not_sent_and_ends_the_request() {
        let mut live = FakeLive::new(1000, seq(20, &[5, 6, 63]));
        let mut sink = TestSink::new();
        let ending = run(&mut live, &req(20), &mut sink);
        assert!(matches!(ending, Ending::Done(Finish::Stop(63))));
        assert_eq!(sink.tokens(), [5, 6]);
        let (finish, counts, ..) = done(&sink);
        assert_eq!(finish, Finish::Stop(63));
        assert_eq!(counts.completion, 3);
        assert_eq!(decodes(&live), [5, 6]);
    }

    #[test]
    fn max_tokens_ends_with_length_without_decoding_the_last_token() {
        let mut live = FakeLive::new(1000, |_| 1);
        let mut r = req(20);
        r.max_tokens = 3;
        let mut sink = TestSink::new();
        assert!(matches!(
            run(&mut live, &r, &mut sink),
            Ending::Done(Finish::Length)
        ));
        assert_eq!(sink.tokens().len(), 3);
        assert_eq!(decodes(&live).len(), 2);
        assert_eq!(live.pos(), 22);
        assert_eq!(done(&sink).1.completion, 3);

        let mut live = FakeLive::new(1000, |_| 1);
        r.max_tokens = 0;
        let mut sink = TestSink::new();
        run(&mut live, &r, &mut sink);
        assert!(sink.tokens().is_empty());
        let (finish, counts, timings, ..) = done(&sink);
        assert_eq!(finish, Finish::Length);
        assert_eq!(counts.completion, 0);
        assert_eq!(timings.first_token, Duration::ZERO);
        assert!(decodes(&live).is_empty());
    }

    #[test]
    fn a_full_context_ends_with_context_full() {
        let mut live = FakeLive::new(22, |_| 1);
        let mut sink = TestSink::new();
        let ending = run(&mut live, &req(20), &mut sink);
        assert!(matches!(ending, Ending::Done(Finish::ContextFull)));
        assert_eq!(sink.tokens().len(), 3);
        assert_eq!(live.pos(), 22);
        assert_eq!(decodes(&live).len(), 2);
    }

    #[test]
    fn a_failed_send_cancels() {
        let mut live = FakeLive::new(1000, |_| 1);
        let mut sink = TestSink::failing_after(1);
        let ending = run(&mut live, &req(20), &mut sink);
        assert!(matches!(ending, Ending::Done(Finish::Cancelled)));
        assert_eq!(sink.events.len(), 1);
        assert!(decodes(&live).is_empty());
        assert_eq!(live.calls, [Call::Prefill(0, 20)]);
    }

    #[test]
    fn a_closed_sink_between_prefill_segments_cancels() {
        let mut live = FakeLive::new(10_000, |_| 1);
        let mut sink = TestSink::new();
        let flag = sink.closed_flag.clone();
        live.on_call(move |c| {
            if *c == Call::Prefill(0, 4096) {
                flag.store(true, Ordering::SeqCst);
            }
        });
        let ending = run(&mut live, &req(9000), &mut sink);
        assert!(matches!(ending, Ending::Done(Finish::Cancelled)));
        // The snapshot after the segment is taken; nothing follows the failure.
        assert_eq!(
            live.calls,
            [
                Call::Prefill(0, 4096),
                Call::Snapshot(4096, Marks::INTERVAL)
            ]
        );
        assert_eq!(sink.events.len(), 1);

        // A sink that closes after Started cancels before any prefill.
        let mut live = FakeLive::new(10_000, |_| 1);
        let mut sink = TestSink::closing_after(1);
        let ending = run(&mut live, &req(20), &mut sink);
        assert!(matches!(ending, Ending::Done(Finish::Cancelled)));
        assert!(live.calls.is_empty());
    }

    #[test]
    fn unload_now_ends_with_unloaded_in_prefill_and_in_decode() {
        for (events, calls, completion) in [(1, 0, 0), (3, 2, 2)] {
            let mut live = FakeLive::new(1000, |_| 1);
            let mut sink = TestSink::new();
            let unload = Arc::new(AtomicUsize::new(0));
            let flag = unload.clone();
            sink.on_event(move |e| {
                if e.len() == events {
                    flag.store(1, Ordering::SeqCst);
                }
            });
            let now = Instant::now();
            let ending = run_turn(
                &mut live,
                &req(20),
                &mut sink,
                &unload,
                now,
                now,
                Duration::ZERO,
                None,
            );
            assert!(matches!(ending, Ending::Done(Finish::Unloaded)));
            let (finish, counts, ..) = done(&sink);
            assert_eq!(finish, Finish::Unloaded);
            assert_eq!(counts.completion, completion);
            // Prefill variant: no call at all; decode variant: the prefill and one decode.
            assert_eq!(live.calls.len(), calls);
        }
    }

    #[test]
    fn the_think_budget_forces_the_close_tokens_then_samples_again() {
        let mut live = FakeLive::new(1000, |h| (h.len() - 20) as u32 + 1);
        let mut r = req(20);
        r.think = think(Some(3), &[40, 41, 42]);
        let mut sink = TestSink::new();
        run(&mut live, &r, &mut sink);
        assert_eq!(sink.tokens(), [1, 2, 3, 40, 41, 42, 7, 8]);
        assert_eq!(
            forced_flags(&sink),
            [false, false, false, true, true, true, false, false]
        );
        assert_eq!(&decodes(&live)[..6], [1, 2, 3, 40, 41, 42]);
        let (finish, counts, ..) = done(&sink);
        assert_eq!(finish, Finish::Length);
        assert_eq!((counts.reasoning, counts.completion), (3, 8));
    }

    #[test]
    fn an_end_token_clears_the_budget() {
        let mut live = FakeLive::new(1000, seq(20, &[5, 62, 6, 7, 8, 9]));
        let mut r = req(20);
        r.think = think(Some(2), &[40, 41, 42]);
        r.max_tokens = 6;
        let mut sink = TestSink::new();
        run(&mut live, &r, &mut sink);
        assert_eq!(sink.tokens(), [5, 62, 6, 7, 8, 9]);
        assert!(forced_flags(&sink).iter().all(|f| !f));
        assert_eq!(done(&sink).1.reasoning, 1);
    }

    #[test]
    fn a_zero_budget_closes_at_once() {
        let mut live = FakeLive::new(1000, |_| 1);
        let mut r = req(20);
        r.think = think(Some(0), &[40, 41, 42]);
        let mut sink = TestSink::new();
        run(&mut live, &r, &mut sink);
        assert_eq!(&sink.tokens()[..4], [40, 41, 42, 1]);
        assert_eq!(&forced_flags(&sink)[..4], [true, true, true, false]);
        assert_eq!(done(&sink).1.reasoning, 0);
    }

    #[test]
    fn forced_tokens_are_not_stop_tokens() {
        let mut live = FakeLive::new(1000, |_| 1);
        let mut r = req(20);
        r.think = think(Some(0), &[63]);
        let mut sink = TestSink::new();
        let ending = run(&mut live, &r, &mut sink);
        assert!(matches!(ending, Ending::Done(Finish::Length)));
        assert_eq!(sink.tokens()[0], 63);
        assert!(forced_flags(&sink)[0]);
        assert_eq!(decodes(&live)[0], 63);
    }

    #[test]
    fn logprobs_come_with_every_sampled_token_when_asked() {
        let mut live = FakeLive::new(1000, |_| 1);
        let mut r = req(20);
        r.logprobs = Some(3);
        r.think = think(Some(1), &[40]);
        let mut sink = TestSink::new();
        run(&mut live, &r, &mut sink);
        let mut peaked = vec![0.0f32; 64];
        peaked[1] = 10.0;
        let want = logprobs(&peaked, 1, 3);
        assert_eq!(want.top.len(), 3);
        let mut seen_forced = false;
        for e in &sink.events {
            if let Event::Token {
                id,
                forced,
                logprobs,
            } = e
            {
                if *forced {
                    seen_forced = true;
                    assert_eq!((*id, logprobs), (40, &None));
                } else {
                    assert_eq!(logprobs.as_ref(), Some(&want));
                }
            }
        }
        assert!(seen_forced);

        r.logprobs = None;
        let mut live = FakeLive::new(1000, |_| 1);
        let mut sink = TestSink::new();
        run(&mut live, &r, &mut sink);
        assert!(sink.events.iter().all(|e| !matches!(
            e,
            Event::Token {
                logprobs: Some(_),
                ..
            }
        )));
    }

    #[test]
    fn presence_penalty_counts_sampled_tokens_only() {
        let sampling = Sampling {
            presence_penalty: 100.0,
            ..Sampling::GREEDY
        };
        // A forced 40 is not in the sampler's seen set: the first sampled 40 stands, the next is penalised.
        let mut live = FakeLive::new(1000, |_| 40);
        let mut r = req(20);
        r.sampling = sampling;
        r.max_tokens = 4;
        r.think = think(Some(0), &[40]);
        let mut sink = TestSink::new();
        run(&mut live, &r, &mut sink);
        assert_eq!(sink.tokens(), [40, 40, 0, 1]);
        assert_eq!(forced_flags(&sink), [true, false, false, false]);

        // The same script with 40 sampled first: the second pick is something else.
        let mut live = FakeLive::new(1000, |_| 40);
        r.think = None;
        let mut sink = TestSink::new();
        run(&mut live, &r, &mut sink);
        assert_eq!(sink.tokens(), [40, 0, 1, 2]);
    }

    #[test]
    fn counts_and_timings_add_up() {
        let mut live = FakeLive::new(1000, |_| 1);
        let mut r = req(50);
        r.max_tokens = 4;
        let mut sink = TestSink::new();
        // Time passes in the sink after `Started` (inside the prefill window) and after the first token (inside the
        // decode window, and before `first_token` is measured), so a gap between the windows would show.
        sink.on_event(|e| match e.len() {
            1 => std::thread::sleep(Duration::from_millis(5)),
            2 => std::thread::sleep(Duration::from_millis(2)),
            _ => {}
        });
        // A deliberate gap between `queued` and the call: the windows must still cover the first token's time.
        let queued = Instant::now();
        std::thread::sleep(Duration::from_millis(5));
        run_turn(
            &mut live,
            &r,
            &mut sink,
            &AtomicUsize::new(0),
            queued,
            queued,
            Duration::ZERO,
            None,
        );
        let (_, counts, timings, ..) = done(&sink);
        assert!(timings.prefill >= Duration::from_millis(5));
        assert!(timings.decode >= Duration::from_millis(2));
        assert!(timings.first_token >= Duration::from_millis(5));
        assert_eq!(
            counts,
            Counts {
                prompt: 50,
                reused: 0,
                prefilled: 50,
                completion: 4,
                reasoning: 0
            }
        );
        assert_eq!(timings.queue, Duration::ZERO);
        assert_eq!(timings.load, Duration::ZERO);
        assert!(
            timings.first_token
                <= timings.queue
                    + timings.load
                    + timings.restore
                    + timings.prefill
                    + timings.decode
        );

        // A request extending the live state reuses it.
        let mut r2 = req(0);
        r2.prompt = [live.tokens().to_vec(), vec![2, 3, 4]].concat();
        let mut sink = TestSink::new();
        run(&mut live, &r2, &mut sink);
        let counts = done(&sink).1;
        assert_eq!(
            (counts.prompt, counts.reused, counts.prefilled),
            (56, 53, 3)
        );
        assert_eq!(counts.prefilled, counts.prompt - counts.reused);
    }

    #[test]
    fn done_carries_the_pools_bytes() {
        let mut live = FakeLive::new(10_000, |_| 1);
        let mut r = req(9000);
        r.cut = Some(Cut {
            at: 8995,
            user_query: true,
        });
        let mut sink = TestSink::new();
        run(&mut live, &r, &mut sink);
        let pool = done(&sink).5;
        assert!(pool > 0);
        assert_eq!(pool, live.reuse_stats().pool_bytes_used);
        // The reply-end snapshot comes after `Done` and is not in it.
        live.snapshot(Marks::REPLY_END).unwrap();
        assert!(live.reuse_stats().pool_bytes_used > pool);
    }

    #[test]
    fn done_carries_the_cache_delta() {
        let mut live = FakeLive::new(1000, |_| 1);
        live.prefill_routes = (5, 1, 2);
        live.decode_routes = (0, 0, 1);
        live.expert_bytes = 2 << 20;
        let mut r = req(20);
        r.max_tokens = 4;
        let mut sink = TestSink::new();
        run(&mut live, &r, &mut sink);
        let cache = done(&sink).3;
        assert_eq!(
            (
                cache.hits,
                cache.staged,
                cache.misses,
                cache.decode_misses,
                cache.decoded
            ),
            (5, 1, 5, 3, 3)
        );
        assert_eq!(cache.hit_rate, Some(6.0 / 11.0));
        assert_eq!(cache.miss_mib_per_token, Some(2.0));
    }

    #[test]
    fn a_failed_restore_sends_an_error_and_the_next_request_does_not_use_live() {
        let mut live = FakeLive::new(1000, |_| 1);
        let mut r = req(100);
        r.cut = Some(Cut {
            at: 90,
            user_query: true,
        });
        run(&mut live, &r, &mut TestSink::new());
        let after_first = live.tokens().to_vec();
        live.fail_restore = Some(Error::NoSuchSnapshot { id: 0 });
        let mut sink = TestSink::new();
        let ending = run(&mut live, &r, &mut sink);
        assert!(matches!(
            ending,
            Ending::Failed(Error::NoSuchSnapshot { .. })
        ));
        assert!(matches!(
            sink.events.as_slice(),
            [Event::Error {
                kind: ErrorKind::Internal,
                ..
            }]
        ));
        // As `Snapshots::restore` leaves it: unusable, the KV record cut back to the common prefix with the snapshot.
        assert!(!live.usable());
        assert_eq!(live.kv_tokens(), &r.prompt[..90]);

        // The live tokens are a proper prefix of this prompt, but the state can't be trusted: the snapshot serves.
        let mut next = req(0);
        next.prompt = [after_first, vec![9, 9]].concat();
        let mut sink = TestSink::new();
        assert!(matches!(run(&mut live, &next, &mut sink), Ending::Done(_)));
        assert_eq!(started(&sink).2, Source::UserQuery);
        assert_eq!(started(&sink).1, 90);
        assert!(live.usable());
    }

    #[test]
    fn an_error_in_prefill_sends_an_error_event() {
        let mut live = FakeLive::new(1000, |_| 1);
        live.fail_at = Some((0, non_finite()));
        let mut sink = TestSink::new();
        let ending = run(&mut live, &req(20), &mut sink);
        assert!(matches!(ending, Ending::Failed(Error::NonFinite { .. })));
        assert_eq!(sink.events.len(), 2);
        assert!(matches!(
            &sink.events[1],
            Event::Error {
                kind: ErrorKind::NonFinite,
                ..
            }
        ));
        assert!(!sink.events.iter().any(|e| matches!(e, Event::Done { .. })));
        assert!(!live.usable());

        // The same in decode: after two tokens.
        let mut live = FakeLive::new(1000, |_| 1);
        live.fail_at = Some((2, non_finite()));
        let mut sink = TestSink::new();
        let ending = run(&mut live, &req(20), &mut sink);
        assert!(matches!(ending, Ending::Failed(_)));
        assert_eq!(sink.tokens().len(), 2);
        assert!(matches!(
            sink.events.last(),
            Some(Event::Error {
                kind: ErrorKind::NonFinite,
                ..
            })
        ));
    }

    #[test]
    fn a_cancel_between_segments_ends_cancelled_and_its_reply_end_reuses_the_split() {
        for reset in [false, true] {
            let mut live = FakeLive::new(10_000, |_| 1);
            let mut sink = TestSink::new();
            let flag = sink.closed_flag.clone();
            live.on_call(move |c| {
                if *c == Call::Prefill(0, 4096) {
                    flag.store(true, Ordering::SeqCst);
                }
            });
            let ending = run(&mut live, &req(9000), &mut sink);
            assert!(matches!(ending, Ending::Done(Finish::Cancelled)));
            // The worker's reply-end snapshot after the Done.
            live.snapshot(Marks::REPLY_END).unwrap();
            let ids = live.tree().ids();
            assert_eq!(ids.len(), 1);
            let node = live.tree().node(ids[0]).unwrap();
            assert_eq!(node.pos(), INTERVAL);
            assert_eq!(node.marks, Marks::INTERVAL.union(Marks::REPLY_END));

            live.clear_hooks();
            if reset {
                live.reset();
            }
            live.calls.clear();
            let mut sink = TestSink::new();
            run(&mut live, &req(9000), &mut sink);
            let want = if reset {
                Source::ReplyEnd
            } else {
                Source::Live
            };
            assert_eq!(started(&sink).1, 4096);
            assert_eq!(started(&sink).2, want, "reset: {reset}");
            assert_eq!(
                live.calls.contains(&Call::Restore(Source::ReplyEnd, 4096)),
                reset
            );
        }
    }

    #[test]
    fn the_sampler_is_fresh_per_request() {
        let sampled = |seed: u64, live: &mut FakeLive| {
            let mut r = req(20);
            r.sampling = Sampling::DEFAULT;
            r.seed = seed;
            let mut sink = TestSink::new();
            run(live, &r, &mut sink);
            sink.tokens()
        };
        let mut live = FakeLive::new(1000, |_| 1);
        live.peak = 0.5;
        let a = sampled(9, &mut live);
        live.reset();
        let b = sampled(9, &mut live);
        assert_eq!(a, b);
        live.reset();
        assert_ne!(a, sampled(10, &mut live));
    }

    #[test]
    fn cache_delta_takes_hits_from_the_whole_request_and_misses_per_token_from_decode() {
        let s = |hits, staged, misses| Stats {
            hits,
            staged,
            misses,
            ..Stats::default()
        };
        let d = cache_delta(&s(10, 2, 3), &s(30, 6, 5), &s(40, 6, 9), 2 << 20, 4);
        assert_eq!(
            (d.hits, d.staged, d.misses, d.decode_misses, d.decoded),
            (30, 4, 6, 4, 4)
        );
        assert_eq!(d.hit_rate, Some(34.0 / 40.0));
        assert_eq!(d.miss_mib_per_token, Some(2.0));
        // Nothing routed, nothing decoded.
        let z = cache_delta(&s(1, 1, 1), &s(1, 1, 1), &s(1, 1, 1), 2 << 20, 0);
        assert_eq!((z.hits, z.decode_misses, z.decoded), (0, 0, 0));
        assert_eq!((z.hit_rate, z.miss_mib_per_token), (None, None));
    }

    #[test]
    fn prefill_points_stops_early_and_resumes_from_a_position() {
        let mut live = FakeLive::new(10_000, |_| 1);
        let p = prompt(9000);
        let mut segments = 0;
        let out = prefill_points(&mut live, &p, 0, None, || {
            segments += 1;
            (segments == 2).then_some(Finish::Cancelled)
        });
        assert!(matches!(out, Ok(Err(Finish::Cancelled))));
        assert_eq!(live.pos(), 4096);

        let out = prefill_points(&mut live, &p, 4096, None, || None)
            .unwrap()
            .unwrap();
        assert_eq!(out.len(), 64);
        assert_eq!(live.pos(), 9000);
    }

    // ---- speculation ----

    use crate::spec::{Drafter, SpecConfig, SpecDelta, SpecState};

    /// The settings every burst test runs under: a fixed draft length, so the wall-clock controller never
    /// turns a fake's microsecond steps into plain decodes.
    fn cfg4() -> SpecConfig {
        SpecConfig {
            max_draft: 4,
            fixed_draft: Some(4),
            ..SpecConfig::default()
        }
    }

    /// A drafter that knows the stream the plain loop produces after a prompt of `n` tokens, and corrupts the draft
    /// tokens at the stream positions `wrong` names.
    struct Truth {
        truth: Vec<u32>,
        n: usize,
        seen: usize,
        wrong: Box<dyn Fn(usize) -> bool + Send>,
    }

    impl Drafter for Truth {
        fn name(&self) -> &'static str {
            "truth"
        }

        fn draft(&mut self, max: usize) -> Vec<u32> {
            // The tokens emitted so far; the draft continues after the last of them.
            let e = (self.seen - self.n).min(self.truth.len());
            let to = (e + max).min(self.truth.len());
            (e..to)
                .map(|i| {
                    let t = self.truth[i];
                    if (self.wrong)(i) {
                        (t + 1) % 60
                    } else {
                        t
                    }
                })
                .collect()
        }

        fn observe(&mut self, tokens: &[u32]) {
            self.seen += tokens.len();
        }
    }

    /// A `SpecState` whose request drafts `truth` (the stream after a prompt of `n` tokens) right, but wrong where
    /// `wrong` says.
    fn truth_state(
        cfg: SpecConfig,
        n: usize,
        truth: &[u32],
        wrong: impl Fn(usize) -> bool + Clone + Send + 'static,
    ) -> SpecState {
        let truth = truth.to_vec();
        SpecState::with_drafters(
            cfg,
            Box::new(move || {
                Box::new(Truth {
                    truth: truth.clone(),
                    n,
                    seen: 0,
                    wrong: Box::new(wrong.clone()),
                })
            }),
        )
    }

    fn right(cfg: SpecConfig, n: usize, truth: &[u32]) -> SpecState {
        truth_state(cfg, n, truth, |_| false)
    }

    fn run_spec(
        live: &mut FakeLive,
        req: &Request,
        sink: &mut TestSink,
        spec: &mut SpecState,
    ) -> Ending {
        let now = Instant::now();
        run_turn(
            live,
            req,
            sink,
            &AtomicUsize::new(0),
            now,
            now,
            Duration::ZERO,
            Some(spec),
        )
    }

    fn done_spec(sink: &TestSink) -> Option<SpecDelta> {
        match sink.events.last() {
            Some(Event::Done { spec, .. }) => *spec,
            other => panic!("the events don't end with Done: {other:?}"),
        }
    }

    fn verifies(live: &FakeLive) -> Vec<Vec<u32>> {
        live.calls
            .iter()
            .filter_map(|c| match c {
                Call::Verify(t) => Some(t.clone()),
                _ => None,
            })
            .collect()
    }

    fn token_events(sink: &TestSink) -> Vec<Event> {
        sink.events
            .iter()
            .filter(|e| matches!(e, Event::Token { .. }))
            .cloned()
            .collect()
    }

    /// The model says `1, 2, 3 ...` after a prompt of `n` tokens: every position's row is different.
    fn counting(n: usize) -> impl Fn(&[u32]) -> u32 + Send + 'static {
        move |h| (h.len() - n) as u32 + 1
    }

    #[test]
    fn speculation_off_calls_live_exactly_as_today() {
        let mut live = FakeLive::new(1000, |_| 1);
        let mut r = req(20);
        r.max_tokens = 20;
        let mut sink = TestSink::new();
        assert!(matches!(
            run(&mut live, &r, &mut sink),
            Ending::Done(Finish::Length)
        ));
        let mut want = vec![Call::Prefill(0, 20)];
        want.extend((0..19).map(|_| Call::Decode(1)));
        assert_eq!(live.calls, want);
        assert_eq!(done_spec(&sink), None);
        assert_eq!(done(&sink).3.decoded, 19);
    }

    #[test]
    fn speculation_gives_the_same_events_as_off() {
        for (sampling, seed) in [(Sampling::GREEDY, 1), (Sampling::DEFAULT, 1234)] {
            for wrong in [vec![], vec![3usize, 11]] {
                let mut r = req(20);
                r.sampling = sampling;
                r.seed = seed;
                r.max_tokens = 24;
                r.stop_tokens = vec![];
                r.logprobs = Some(3);
                let model = |h: &[u32]| ((h.iter().sum::<u32>() * 7 + h.len() as u32) % 60) + 1;

                let mut off = FakeLive::new(1000, model);
                off.peak = 0.5;
                let mut off_sink = TestSink::new();
                run(&mut off, &r, &mut off_sink);
                let truth = off_sink.tokens();
                assert_eq!(truth.len(), 24);

                let mut on = FakeLive::new(1000, model);
                on.peak = 0.5;
                let mut on_sink = TestSink::new();
                let bad = wrong.clone();
                let mut spec = truth_state(cfg4(), 20, &truth, move |i| bad.contains(&i));
                run_spec(&mut on, &r, &mut on_sink, &mut spec);

                let case = format!("{sampling:?} wrong at {wrong:?}");
                assert_eq!(token_events(&on_sink), token_events(&off_sink), "{case}");
                assert_eq!(done(&on_sink).1, done(&off_sink).1, "{case}");
                assert_eq!(done(&on_sink).0, done(&off_sink).0, "{case}");
                // The committed history is the plain loop's too.
                assert_eq!(on.tokens(), off.tokens(), "{case}");
                assert_eq!(on.kv_tokens(), off.kv_tokens(), "{case}");
                // Bursts ran, and the wrong drafts cost replays.
                let spec = done_spec(&on_sink).expect("a speculating request");
                assert!(spec.steps > 0, "{case}: {spec:?}");
                assert_eq!(spec.replays > 0, !wrong.is_empty(), "{case}: {spec:?}");
                assert_eq!(done_spec(&off_sink), None);
            }
        }
    }

    #[test]
    fn a_stop_token_inside_a_burst_ends_the_turn_there() {
        let mut live = FakeLive::new(1000, seq(20, &[5, 6, 63, 7, 8, 9]));
        let mut spec = right(cfg4(), 20, &[5, 6, 63, 7, 8, 9]);
        let mut sink = TestSink::new();
        let ending = run_spec(&mut live, &req(20), &mut sink, &mut spec);
        assert!(matches!(ending, Ending::Done(Finish::Stop(63))));
        assert_eq!(sink.tokens(), [5, 6]);
        let (finish, counts, ..) = done(&sink);
        assert_eq!((finish, counts.completion), (Finish::Stop(63), 3));
        // The stop token's row is the last one read; the rows past it never were.
        assert_eq!(verifies(&live), [vec![5, 6, 63, 7, 8]]);
        assert_eq!(live.calls.last(), Some(&Call::Commit(2)));
        assert!(decodes(&live).is_empty());
        assert_eq!(live.tokens(), [prompt(20), vec![5, 6]].concat());
        assert_eq!(live.kv_tokens(), live.tokens());
        let delta = done_spec(&sink).expect("one verify step ran");
        assert_eq!(
            (delta.steps, delta.drafted, delta.accepted, delta.replays),
            (1, 4, 1, 1)
        );
    }

    #[test]
    fn max_tokens_caps_the_draft() {
        let mut live = FakeLive::new(1000, |_| 1);
        let mut spec = right(cfg4(), 20, &[1; 10]);
        let mut r = req(20);
        r.max_tokens = 3;
        let mut sink = TestSink::new();
        let ending = run_spec(&mut live, &r, &mut sink, &mut spec);
        assert!(matches!(ending, Ending::Done(Finish::Length)));
        // One token emitted before the step, so the step may keep two: `x` and one draft.
        assert_eq!(verifies(&live), [vec![1, 1]]);
        assert_eq!(sink.tokens().len(), 3);
        assert_eq!(done(&sink).1.completion, 3);
        // Like the plain loop, the last token is not in the state.
        assert_eq!(live.pos(), 22);
        assert_eq!(live.calls.last(), Some(&Call::Commit(2)));
    }

    #[test]
    fn the_context_end_caps_the_draft() {
        // ctx 24, a prompt of 20: the state can take four more tokens, so a step verifies four at most.
        let truth = [1u32; 12];
        let mut live = FakeLive::new(24, |_| 1);
        let mut spec = right(cfg4(), 20, &truth);
        let mut sink = TestSink::new();
        let ending = run_spec(&mut live, &req(20), &mut sink, &mut spec);
        assert!(matches!(ending, Ending::Done(Finish::ContextFull)));
        assert_eq!(verifies(&live), [vec![1; 4]]);
        assert_eq!(sink.tokens().len(), 5);
        assert_eq!(live.pos(), 24);

        // The plain loop stops in the same place.
        let mut plain = FakeLive::new(24, |_| 1);
        let mut plain_sink = TestSink::new();
        run(&mut plain, &req(20), &mut plain_sink);
        assert_eq!(done(&plain_sink).0, Finish::ContextFull);
        assert_eq!(plain_sink.tokens(), sink.tokens());
        assert_eq!(plain.tokens(), live.tokens());
    }

    #[test]
    fn the_think_budget_forces_its_close_inside_a_step() {
        let mut r = req(20);
        r.think = think(Some(3), &[40, 41, 42]);
        let mut off = FakeLive::new(1000, counting(20));
        let mut off_sink = TestSink::new();
        run(&mut off, &r, &mut off_sink);
        assert_eq!(off_sink.tokens(), [1, 2, 3, 40, 41, 42, 7, 8]);

        let mut live = FakeLive::new(1000, counting(20));
        let mut spec = right(cfg4(), 20, &off_sink.tokens());
        let mut sink = TestSink::new();
        run_spec(&mut live, &r, &mut sink, &mut spec);
        // The forced tokens come where plain decode puts them, drafts equal to them were accepted, in one step.
        assert_eq!(token_events(&sink), token_events(&off_sink));
        assert_eq!(
            forced_flags(&sink),
            [false, false, false, true, true, true, false, false]
        );
        assert_eq!(verifies(&live), [vec![1, 2, 3, 40, 41], vec![42, 7]]);
        assert_eq!(done(&sink).1, done(&off_sink).1);
        assert_eq!(done(&sink).1.reasoning, 3);

        // The sampler isn't called for a forced token: the penalty must not see it. The model says 40, the block closes
        // with the forced 0 after one sampled token, and a sampler that had seen the 0 would answer 1 after it.
        let penalised = Sampling {
            presence_penalty: 100.0,
            ..Sampling::GREEDY
        };
        let mut r = req(20);
        r.sampling = penalised;
        r.max_tokens = 4;
        r.think = think(Some(1), &[0]);
        let mut off = FakeLive::new(1000, |_| 40);
        let mut off_sink = TestSink::new();
        run(&mut off, &r, &mut off_sink);
        assert_eq!(off_sink.tokens(), [40, 0, 0, 1]);
        let mut live = FakeLive::new(1000, |_| 40);
        let mut spec = right(cfg4(), 20, &[40, 0, 0, 1]);
        let mut sink = TestSink::new();
        run_spec(&mut live, &r, &mut sink, &mut spec);
        assert_eq!(token_events(&sink), token_events(&off_sink));
        assert_eq!(forced_flags(&sink), [false, true, false, false]);
        assert_eq!(verifies(&live), [vec![40, 0, 0]]);
    }

    /// A 12-token counting stream, a burst of five tokens, and a sink that closes after the second token the burst sent.
    fn cut_in_a_burst(unload: bool, fail_commit: bool) -> (FakeLive, TestSink, Ending) {
        let truth: Vec<u32> = (1..=12).collect();
        let mut live = FakeLive::new(1000, counting(20));
        if fail_commit {
            live.fail_commit = Some(non_finite());
        }
        let mut spec = right(cfg4(), 20, &truth);
        let mut r = req(20);
        r.max_tokens = 12;
        let flag = Arc::new(AtomicUsize::new(0));
        let mut sink = if unload {
            TestSink::new()
        } else {
            // Started, the first token, then two from the burst.
            TestSink::closing_after(4)
        };
        let f = flag.clone();
        sink.on_event(move |e| {
            if unload && e.len() == 4 {
                f.store(1, Ordering::SeqCst);
            }
        });
        let now = Instant::now();
        let ending = run_turn(
            &mut live,
            &r,
            &mut sink,
            &flag,
            now,
            now,
            Duration::ZERO,
            Some(&mut spec),
        );
        (live, sink, ending)
    }

    #[test]
    fn a_cancel_inside_a_burst_commits_the_rows_reached() {
        let (mut live, sink, ending) = cut_in_a_burst(false, false);
        assert!(matches!(ending, Ending::Done(Finish::Cancelled)));
        assert_eq!(sink.tokens(), [1, 2, 3]);
        assert_eq!(verifies(&live), [vec![1, 2, 3, 4, 5]]);
        assert_eq!(live.calls.last(), Some(&Call::Commit(2)));
        // `x` and the first draft are in the state; the token the cancel saw is not.
        assert_eq!(live.tokens(), [prompt(20), vec![1, 2]].concat());
        assert_eq!(live.kv_tokens(), live.tokens());
        assert!(live.usable());

        // The next request restores from the live state.
        let mut next = req(0);
        next.prompt = [live.tokens().to_vec(), vec![9, 9]].concat();
        let mut sink = TestSink::new();
        assert!(matches!(run(&mut live, &next, &mut sink), Ending::Done(_)));
        assert_eq!(started(&sink).2, Source::Live);
        assert_eq!(started(&sink).1, 22);
        assert_eq!(done(&sink).1.prefilled, 2);
    }

    #[test]
    fn an_unload_inside_a_burst_ends_after_that_token() {
        let (mut live, sink, ending) = cut_in_a_burst(true, false);
        assert!(matches!(ending, Ending::Done(Finish::Unloaded)));
        // The token whose send raised the flag was sent; nothing after it was.
        assert_eq!(sink.tokens(), [1, 2, 3]);
        assert_eq!(done(&sink).0, Finish::Unloaded);
        assert_eq!(live.calls.last(), Some(&Call::Commit(2)));
        assert_eq!(live.tokens(), [prompt(20), vec![1, 2]].concat());
        assert_eq!(live.kv_tokens(), live.tokens());
        assert!(live.usable());

        // The next request restores from the live state, as after a cancel.
        let mut next = req(0);
        next.prompt = [live.tokens().to_vec(), vec![9, 9]].concat();
        let mut sink = TestSink::new();
        assert!(matches!(run(&mut live, &next, &mut sink), Ending::Done(_)));
        assert_eq!(started(&sink).2, Source::Live);
        assert_eq!(started(&sink).1, 22);
        assert_eq!(done(&sink).1.prefilled, 2);
    }

    #[test]
    fn a_commit_failing_after_the_burst_ended_still_gives_done() {
        // The client has every token it will get; only the live state is lost. The turn says so with `Unsettled`,
        // sends `Done` (no `Error`), and the next request starts fresh.
        let counting_truth: Vec<u32> = (1..=12).collect();
        let cases: Vec<(&str, FakeLive, Request, Vec<u32>, Finish)> = vec![
            (
                "stop",
                FakeLive::new(1000, seq(20, &[5, 6, 63, 7])),
                req(20),
                vec![5, 6, 63, 7, 1, 2],
                Finish::Stop(63),
            ),
            (
                "length",
                FakeLive::new(1000, counting(20)),
                Request {
                    max_tokens: 4,
                    ..req(20)
                },
                counting_truth,
                Finish::Length,
            ),
        ];
        for (name, mut live, r, truth, finish) in cases {
            live.fail_commit = Some(non_finite());
            let mut spec = right(cfg4(), 20, &truth);
            let mut sink = TestSink::new();
            let ending = run_spec(&mut live, &r, &mut sink, &mut spec);
            let Ending::Unsettled(got, Error::NonFinite { .. }) = ending else {
                panic!("{name}: {ending:?}");
            };
            assert_eq!(got, finish, "{name}");
            assert_eq!(done(&sink).0, got, "{name}");
            assert!(!sink.events.iter().any(|e| matches!(e, Event::Error { .. })));
            assert!(!live.usable(), "{name}");
            assert_eq!(live.tokens(), prompt(20), "{name}");
            // The next request starts fresh.
            let mut sink = TestSink::new();
            run(&mut live, &r, &mut sink);
            assert_eq!(started(&sink).2, Source::None, "{name}");
        }
        // A cancel and an unload inside the burst end the same way.
        for unload in [false, true] {
            let (mut live, sink, ending) = cut_in_a_burst(unload, true);
            let want = if unload {
                Finish::Unloaded
            } else {
                Finish::Cancelled
            };
            assert!(
                matches!(&ending, Ending::Unsettled(f, _) if *f == want),
                "unload {unload}: {ending:?}"
            );
            // A closed sink takes no `Done`; an unloading one does.
            if unload {
                assert_eq!(done(&sink).0, want);
            }
            assert!(!sink.events.iter().any(|e| matches!(e, Event::Error { .. })));
            assert_eq!(sink.tokens(), [1, 2, 3]);
            assert!(!live.usable());
            let mut next = req(20);
            next.max_tokens = 3;
            let mut sink = TestSink::new();
            run(&mut live, &next, &mut sink);
            assert_eq!(started(&sink).2, Source::None);
        }
    }

    #[test]
    fn the_reply_end_snapshot_after_a_partial_accept_holds_the_committed_tokens() {
        // A stop token cut the burst after two of its five tokens.
        let mut live = FakeLive::new(1000, seq(20, &[5, 6, 63, 7, 8, 9]));
        let mut spec = right(cfg4(), 20, &[5, 6, 63, 7, 8, 9]);
        let mut sink = TestSink::new();
        run_spec(&mut live, &req(20), &mut sink, &mut spec);
        assert_eq!(live.calls.last(), Some(&Call::Commit(2)));
        // As the worker does after `Done`: the fake takes a snapshot only when the KV record is the token list.
        live.snapshot(Marks::REPLY_END).unwrap();
        let committed = [prompt(20), vec![5, 6]].concat();
        assert_eq!(live.kv_tokens(), committed);
        let nodes = live.parkable();
        assert_eq!(nodes.len(), 1, "the snapshot was taken, not skipped");
        assert_eq!(nodes[0].0, committed);

        // And a request that reuses the conversation restores it.
        live.reset();
        let mut next = req(0);
        next.prompt = [committed.clone(), vec![9]].concat();
        let mut sink = TestSink::new();
        run(&mut live, &next, &mut sink);
        assert_eq!(started(&sink).1, 22);
        assert_eq!(started(&sink).2, Source::ReplyEnd);
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Fault {
        Verify,
        Row,
        Commit,
    }

    /// A speculating request whose `fail` hook breaks the first verify (or its commit), and the request after it.
    fn failed_step_then_next(fault: Fault) {
        let truth: Vec<u32> = (1..=12).collect();
        let mut live = FakeLive::new(1000, counting(20));
        let error = non_finite();
        let fail_commit = fault == Fault::Commit;
        match fault {
            Fault::Verify => live.fail_verify = Some(error),
            // Row 1: row 0's token is sent first.
            Fault::Row => live.fail_row = Some((1, error)),
            Fault::Commit => live.fail_commit = Some(error),
        }
        let mut spec = right(cfg4(), 20, &truth);
        let mut r = req(20);
        r.max_tokens = 12;
        let mut sink = TestSink::new();
        let ending = run_spec(&mut live, &r, &mut sink, &mut spec);
        assert!(matches!(ending, Ending::Failed(Error::NonFinite { .. })));
        // The tokens the step's rows had already given the client stay sent; then the error, and no `Done`.
        assert_eq!(
            sink.tokens(),
            match fault {
                Fault::Verify => vec![1],
                Fault::Row => vec![1, 2],
                Fault::Commit => vec![1, 2, 3, 4, 5, 6],
            }
        );
        assert!(matches!(
            sink.events.last(),
            Some(Event::Error {
                kind: ErrorKind::NonFinite,
                ..
            })
        ));
        assert!(!sink.events.iter().any(|e| matches!(e, Event::Done { .. })));
        assert_eq!(verifies(&live).len(), 1);
        assert_eq!(
            live.calls.iter().any(|c| matches!(c, Call::Commit(_))),
            fail_commit
        );
        // The state is unusable, back at the step's start, and the KV record is cut there too.
        assert!(!live.usable());
        assert_eq!(live.tokens(), prompt(20));
        assert_eq!(live.kv_tokens(), prompt(20));

        // The next request starts fresh, and speculates again (the engine-wide state survived).
        let mut sink = TestSink::new();
        let ending = run_spec(&mut live, &r, &mut sink, &mut spec);
        assert!(matches!(ending, Ending::Done(Finish::Length)));
        assert_eq!(started(&sink).2, Source::None);
        assert_eq!(started(&sink).1, 0);
        assert_eq!(sink.tokens(), truth);
        assert!(live.usable());
        assert_eq!(live.kv_tokens(), live.tokens());
    }

    #[test]
    fn a_failed_verify_ends_the_request_and_the_next_one_starts_fresh() {
        failed_step_then_next(Fault::Verify);
    }

    #[test]
    fn a_failed_row_ends_the_request_and_the_next_one_starts_fresh() {
        failed_step_then_next(Fault::Row);
    }

    #[test]
    fn a_failed_commit_ends_the_request_and_the_next_one_starts_fresh() {
        failed_step_then_next(Fault::Commit);
    }

    #[test]
    fn each_tokens_logprobs_come_from_its_own_row() {
        let truth: Vec<u32> = (1..=8).collect();
        let mut r = req(20);
        r.logprobs = Some(3);
        let mut live = FakeLive::new(1000, counting(20));
        let mut spec = right(cfg4(), 20, &truth);
        let mut sink = TestSink::new();
        run_spec(&mut live, &r, &mut sink, &mut spec);
        assert!(!verifies(&live).is_empty());
        let mut off = FakeLive::new(1000, counting(20));
        let mut off_sink = TestSink::new();
        run(&mut off, &r, &mut off_sink);
        assert_eq!(token_events(&sink), token_events(&off_sink));

        // Every position's row peaks at its own token, so a row read for the wrong position would show.
        let mut seen = Vec::new();
        for e in &sink.events {
            if let Event::Token {
                id, logprobs: lp, ..
            } = e
            {
                let mut row = vec![0.0f32; 64];
                row[*id as usize] = 10.0;
                assert_eq!(lp.as_ref(), Some(&logprobs(&row, *id, 3)), "token {id}");
                seen.push(*id);
            }
        }
        assert_eq!(seen, truth);
    }

    #[test]
    fn the_restored_prefix_is_in_the_drafters_history() {
        // The model continues a chain: 30 31 32 33 34 35, and then stops.
        let chain = |h: &[u32]| match h.last() {
            Some(30) => 31,
            Some(31) => 32,
            Some(32) => 33,
            Some(33) => 34,
            Some(34) => 35,
            Some(35) => 63,
            _ => 1,
        };
        let mut live = FakeLive::new(1000, chain);
        let mut spec = SpecState::new(cfg4());
        let mut first = req(0);
        first.prompt = vec![2, 3, 30, 31, 32, 33, 34, 35, 3, 4, 5, 6, 7, 8];
        first.max_tokens = 3;
        let mut sink = TestSink::new();
        run_spec(&mut live, &first, &mut sink, &mut spec);
        let state = live.tokens().to_vec();
        assert!(state.len() > 14);

        // Request 2 restores the live state and prefills only [40, 30]. Its first token is 31, and the only earlier
        // `30 31` is in the restored prefix.
        let mut second = req(0);
        second.prompt = [state.clone(), vec![40, 30]].concat();
        second.max_tokens = 8;
        live.calls.clear();
        let mut sink = TestSink::new();
        run_spec(&mut live, &second, &mut sink, &mut spec);
        assert_eq!(started(&sink).2, Source::Live);
        assert_eq!(started(&sink).1 as usize, state.len());
        assert_eq!(done(&sink).1.prefilled, 2);
        assert_eq!(verifies(&live)[0], [31, 32, 33, 34, 35]);
        assert_eq!(sink.tokens(), [31, 32, 33, 34, 35]);
        assert_eq!(done(&sink).0, Finish::Stop(63));
    }

    #[test]
    fn the_done_carries_the_spec_counts() {
        let truth: Vec<u32> = (1..=10).collect();
        let mut r = req(20);
        r.max_tokens = 10;

        // Every draft right: a verify of five, then one of four (max_tokens caps the draft at three).
        let mut live = FakeLive::new(1000, counting(20));
        let mut spec = right(cfg4(), 20, &truth);
        let mut sink = TestSink::new();
        run_spec(&mut live, &r, &mut sink, &mut spec);
        assert_eq!(verifies(&live), [vec![1, 2, 3, 4, 5], vec![6, 7, 8, 9]]);
        assert_eq!(
            done_spec(&sink),
            Some(SpecDelta {
                steps: 2,
                plain_steps: 0,
                drafted: 7,
                accepted: 7,
                drafted_lookup: 7,
                accepted_lookup: 7,
                replays: 0,
                ..SpecDelta::default()
            })
        );
        // The request's counts are also the engine-wide state's.
        assert_eq!(spec.totals().delta, done_spec(&sink).unwrap());
        // `decoded` counts committed positions: nine (the last token is not decoded).
        assert_eq!(done(&sink).3.decoded, 9);

        // One wrong draft in the first step: it keeps one position.
        let mut live = FakeLive::new(1000, counting(20));
        let mut spec = truth_state(cfg4(), 20, &truth, |i| i == 1);
        let mut sink = TestSink::new();
        run_spec(&mut live, &r, &mut sink, &mut spec);
        assert_eq!(sink.tokens(), truth);
        assert_eq!(
            done_spec(&sink),
            Some(SpecDelta {
                steps: 3,
                plain_steps: 0,
                drafted: 10,
                accepted: 6,
                drafted_lookup: 10,
                accepted_lookup: 6,
                replays: 1,
                ..SpecDelta::default()
            })
        );

        // A fixed draft of 0 is plain steps; the counts say so.
        let mut live = FakeLive::new(1000, counting(20));
        let cfg = SpecConfig {
            fixed_draft: Some(0),
            ..SpecConfig::default()
        };
        let mut spec = right(cfg, 20, &truth);
        let mut sink = TestSink::new();
        run_spec(&mut live, &r, &mut sink, &mut spec);
        assert!(verifies(&live).is_empty());
        assert_eq!(
            done_spec(&sink),
            Some(SpecDelta {
                plain_steps: 9,
                ..SpecDelta::default()
            })
        );

        // No step ran: the first token ended the request. `None`, not zeros.
        for (limit, script) in [(1u32, vec![5u32]), (8, vec![63])] {
            let mut r = req(20);
            r.max_tokens = limit;
            let mut live = FakeLive::new(1000, seq(20, &script));
            let mut spec = right(cfg4(), 20, &script);
            let mut sink = TestSink::new();
            assert!(matches!(
                run_spec(&mut live, &r, &mut sink, &mut spec),
                Ending::Done(_)
            ));
            assert_eq!(done_spec(&sink), None);
            assert_eq!(live.calls, [Call::Prefill(0, 20)]);
        }
    }

    #[test]
    fn without_a_fixed_draft_the_controller_decides() {
        // Every draft is wrong and the prior is 0, so whatever the clock says the controller never speculates.
        let truth: Vec<u32> = (1..=10).collect();
        let mut r = req(20);
        r.max_tokens = 10;
        let mut live = FakeLive::new(1000, counting(20));
        let cfg = SpecConfig {
            accept_prior: 0.0,
            ..SpecConfig::default()
        };
        let mut spec = truth_state(cfg, 20, &truth, |_| true);
        let mut sink = TestSink::new();
        run_spec(&mut live, &r, &mut sink, &mut spec);
        assert_eq!(sink.tokens(), truth);
        assert!(verifies(&live).is_empty());
        let delta = done_spec(&sink).expect("steps ran");
        assert_eq!((delta.steps, delta.plain_steps), (0, 9));
    }

    // ---- the MTP source ----

    use crate::fake::{FakeMtp, MtpFault};
    use crate::SpecSource;

    /// A speculating state whose lookup never has a draft, so MTP is the only source that drafts.
    fn mtp_only(cfg: SpecConfig) -> SpecState {
        right(cfg, 20, &[])
    }

    #[test]
    fn mtp_drafts_reach_the_turn_and_the_tokens_are_the_plain_loops() {
        let truth: Vec<u32> = (1..=10).collect();
        let mut r = req(20);
        r.max_tokens = 10;
        let mut plain = FakeLive::new(1000, counting(20));
        let mut plain_sink = TestSink::new();
        run(&mut plain, &r, &mut plain_sink);
        assert_eq!(plain_sink.tokens(), truth);

        for (case, mtp, accepted) in [
            ("right", FakeMtp::Right, 7),
            ("wrong", FakeMtp::Wrong, 0),
            ("short", FakeMtp::Fixed(vec![2, 3]), 2),
            ("none", FakeMtp::Fixed(vec![]), 0),
        ] {
            let mut live = FakeLive::new(1000, counting(20)).with_mtp(mtp);
            let mut spec = mtp_only(cfg4());
            let mut sink = TestSink::new();
            let ending = run_spec(&mut live, &r, &mut sink, &mut spec);
            assert!(matches!(ending, Ending::Done(Finish::Length)), "{case}");
            // MTP changes which drafts are offered, never the tokens.
            assert_eq!(sink.tokens(), plain_sink.tokens(), "{case}");
            assert_eq!(token_events(&sink), token_events(&plain_sink), "{case}");
            assert!(
                live.calls.iter().any(|c| matches!(c, Call::MtpDraft(1, 4))),
                "{case}: {:?}",
                live.calls
            );
            let delta = done_spec(&sink).expect("steps ran");
            assert_eq!(delta.drafted_lookup, 0, "{case}");
            assert_eq!(delta.accepted_mtp, accepted, "{case}: {delta:?}");
            assert_eq!(live.kv_tokens(), live.tokens(), "{case}");
        }
        // The right MTP's verifies are the lookup test's: a draft of four, then three (max_tokens caps it).
        let mut live = FakeLive::new(1000, counting(20)).with_mtp(FakeMtp::Right);
        let mut spec = mtp_only(cfg4());
        run_spec(&mut live, &r, &mut TestSink::new(), &mut spec);
        assert_eq!(verifies(&live), [vec![1, 2, 3, 4, 5], vec![6, 7, 8, 9]]);
        assert_eq!(
            live.calls
                .iter()
                .filter(|c| matches!(c, Call::MtpDraft(..)))
                .count(),
            2
        );
    }

    #[test]
    fn without_mtp_the_turn_is_5as() {
        // No MTP in the state: the same request makes the lookup-only calls, and never asks for a draft pass.
        let truth: Vec<u32> = (1..=10).collect();
        let mut r = req(20);
        r.max_tokens = 10;
        let mut live = FakeLive::new(1000, counting(20));
        let mut spec = right(cfg4(), 20, &truth);
        let mut sink = TestSink::new();
        run_spec(&mut live, &r, &mut sink, &mut spec);
        assert_eq!(
            live.calls,
            [
                Call::Prefill(0, 20),
                Call::Verify(vec![1, 2, 3, 4, 5]),
                Call::Commit(5),
                Call::Verify(vec![6, 7, 8, 9]),
                Call::Commit(4),
            ]
        );
        // Lookup finds nothing and the live state has no MTP, with the default sources (lookup and mtp): the step is
        // a plain decode, and nothing asks it for a draft pass.
        let mut live = FakeLive::new(1000, counting(20));
        let mut spec = mtp_only(cfg4());
        let mut sink = TestSink::new();
        run_spec(&mut live, &r, &mut sink, &mut spec);
        let mut want = vec![Call::Prefill(0, 20)];
        want.extend((1..=9).map(Call::Decode));
        assert_eq!(live.calls, want);
        assert_eq!(sink.tokens(), truth);
        // A state that can draft with MTP but is configured without it asks for nothing either.
        let cfg = SpecConfig {
            sources: vec![SpecSource::Lookup],
            ..cfg4()
        };
        let mut live = FakeLive::new(1000, counting(20)).with_mtp(FakeMtp::Right);
        let mut spec = mtp_only(cfg);
        run_spec(&mut live, &r, &mut TestSink::new(), &mut spec);
        assert!(!live.calls.iter().any(|c| matches!(c, Call::MtpDraft(..))));
    }

    #[test]
    fn the_controller_probes_and_the_probes_reach_the_done() {
        // No fixed draft: the controller decides. A min_gain nothing reaches keeps MTP from drafting except as a
        // probe, which comes after `probe_every` plain steps since it last drafted.
        let truth: Vec<u32> = (1..=14).collect();
        let mut r = req(20);
        r.max_tokens = 14;
        let cfg = SpecConfig {
            min_gain: 1000.0,
            probe_every: 2,
            ..SpecConfig::default()
        };
        let mut plain = FakeLive::new(1000, counting(20));
        let mut plain_sink = TestSink::new();
        run(&mut plain, &r, &mut plain_sink);
        let mut live = FakeLive::new(1000, counting(20)).with_mtp(FakeMtp::Right);
        let mut spec = mtp_only(cfg);
        let mut sink = TestSink::new();
        run_spec(&mut live, &r, &mut sink, &mut spec);
        assert_eq!(sink.tokens(), truth);
        assert_eq!(sink.tokens(), plain_sink.tokens());
        let d = done_spec(&sink).expect("steps ran");
        assert!(d.probes > 0, "{d:?}");
        // Every MTP step here is a probe: one draft token each.
        assert_eq!(d.drafted_mtp, d.probes, "{d:?}");
        assert_eq!(d.drafted_lookup, 0);
        assert!(live
            .calls
            .iter()
            .all(|c| !matches!(c, Call::MtpDraft(_, k) if *k != 1)));
        assert_eq!(spec.totals().delta.probes, d.probes);
    }

    #[test]
    fn the_live_adapter_forwards_the_mtp() {
        // `LiveVerifier` is how the worker's turn reaches `Live`: without these two forwards every real run is
        // lookup-only.
        let mut live = FakeLive::new(1000, counting(20)).with_mtp(FakeMtp::Right);
        live.prefill(&prompt(20)).expect("prefill");
        let mut v = LiveVerifier(&mut live);
        assert!(v.has_mtp());
        assert_eq!(v.mtp_draft(1, 3).expect("a draft"), [2, 3, 4]);
        let mut bare = FakeLive::new(1000, counting(20));
        assert!(!LiveVerifier(&mut bare).has_mtp());
    }

    /// The engine's own adapters have no CPU double (a `Session` needs the GPU), so the forwarding is pinned in their
    /// source: an impl that dropped `mtp_draft` or `has_mtp` would run the lookup-only loop on the real model.
    #[test]
    fn the_engines_adapters_forward_the_mtp() {
        for (name, source, marker) in [
            (
                "SessionVerifier",
                include_str!("verify.rs"),
                "impl spec::Verifier for SessionVerifier",
            ),
            ("Loaded", include_str!("loaded.rs"), "impl Live for Loaded"),
        ] {
            let at = source.find(marker).unwrap_or_else(|| panic!("{name}"));
            let body = &source[at..];
            let end = body.find("\n}\n").expect("the impl ends");
            let body = &body[..end];
            assert!(
                body.contains("fn mtp_draft(") && body.contains("fn has_mtp("),
                "{name} must forward mtp_draft and has_mtp"
            );
            assert!(
                body.contains(".mtp_draft(") && body.contains(".has_mtp()"),
                "{name} must call the session's"
            );
        }
    }

    #[test]
    fn a_failed_mtp_pass_ends_the_request_and_the_next_one_starts_fresh() {
        let truth: Vec<u32> = (1..=12).collect();
        let mut r = req(20);
        r.max_tokens = 12;
        for submit in [true, false] {
            let mut live = FakeLive::new(1000, counting(20)).with_mtp(FakeMtp::Right);
            live.fail_mtp = Some(MtpFault {
                error: if submit {
                    non_finite()
                } else {
                    Error::BadSlotEntry {
                        pos: 20,
                        layer: 40,
                        expert: 3,
                    }
                },
                submit,
            });
            let mut spec = mtp_only(cfg4());
            let mut sink = TestSink::new();
            let ending = run_spec(&mut live, &r, &mut sink, &mut spec);
            assert!(matches!(ending, Ending::Failed(_)), "submit {submit}");
            // The first token came from the prefill; then the error, and no `Done`.
            assert_eq!(sink.tokens(), [1], "submit {submit}");
            assert!(
                matches!(sink.events.last(), Some(Event::Error { .. })),
                "submit {submit}"
            );
            assert!(!sink.events.iter().any(|e| matches!(e, Event::Done { .. })));
            // Nothing of the main state moved, and nothing is left pending.
            assert!(verifies(&live).is_empty());
            assert_eq!(live.tokens(), prompt(20));
            // A failed submit leaves the state unusable (the GPU may still be writing); a refused slot doesn't.
            assert_eq!(live.usable(), !submit, "submit {submit}");

            // The next request still speculates with MTP and gives the plain loop's tokens.
            let mut sink = TestSink::new();
            let ending = run_spec(&mut live, &r, &mut sink, &mut spec);
            assert!(matches!(ending, Ending::Done(Finish::Length)));
            if submit {
                assert_eq!(started(&sink).2, Source::None);
                assert_eq!(started(&sink).1, 0);
            }
            assert_eq!(sink.tokens(), truth, "submit {submit}");
            assert!(live.usable());
            assert!(done_spec(&sink).expect("steps").drafted_mtp > 0);
        }
    }

    #[test]
    fn the_done_carries_the_per_source_counts() {
        let truth: Vec<u32> = (1..=10).collect();
        let mut r = req(20);
        r.max_tokens = 10;

        // MTP alone, every draft right: a pass of four, then one of three.
        let mut live = FakeLive::new(1000, counting(20)).with_mtp(FakeMtp::Right);
        let mut spec = mtp_only(cfg4());
        let mut sink = TestSink::new();
        run_spec(&mut live, &r, &mut sink, &mut spec);
        let want = SpecDelta {
            steps: 2,
            drafted: 7,
            accepted: 7,
            drafted_mtp: 7,
            accepted_mtp: 7,
            ..SpecDelta::default()
        };
        assert_eq!(done_spec(&sink), Some(want));
        assert_eq!(spec.totals().delta, want);
        let t = spec.totals();
        assert_eq!((t.mtp_by_k[3].steps, t.mtp_by_k[2].steps), (1, 1));

        // Both sources in one request: lookup drafts while its truth lasts (four tokens), MTP the rest.
        let mut live = FakeLive::new(1000, counting(20)).with_mtp(FakeMtp::Right);
        let mut spec = right(cfg4(), 20, &truth[..4]);
        let mut sink = TestSink::new();
        run_spec(&mut live, &r, &mut sink, &mut spec);
        assert_eq!(sink.tokens(), truth);
        let d = done_spec(&sink).expect("steps ran");
        assert!(d.drafted_lookup > 0 && d.drafted_mtp > 0, "{d:?}");
        assert_eq!(d.drafted, d.drafted_lookup + d.drafted_mtp);
        assert_eq!(d.accepted, d.accepted_lookup + d.accepted_mtp);
    }
}
