//! A scripted engine for the generation loop's tests and the worker's: a model of vocabulary 64
//! whose logits peak at whatever `script` names for the history, over the real snapshot `Tree` (byte accounting only),
//! recording every call and failing where a test says. It follows `Session`'s and `Snapshots`' rules: a write that
//! fails past validation marks the state unusable and cuts the KV record; `reset` clears the tokens, keeps the record
//! and makes the state usable again; an unusable or empty state is never a candidate. `FakeLoader` is the worker's
//! scripted `Loader` over it.
use crate::request::{Event, EventSink};
use crate::reuse::{lcp, KvRecord, Marks, Plan, ReuseStats, Sizes, Tree};
use crate::session::is_validation;
use crate::spec::MAX_VERIFY;
use crate::turn::{Live, Restored};
use crate::worker::{LoadFailure, LoadInfo, Loader, ParkedStats};
use crate::{Error, PoolLimit, Source, Stats, Unparked};
use qwen36::dims::Dims;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// The fake model's vocabulary.
pub const VOCAB: usize = 64;

/// One call into a `FakeLive`, in order.
#[derive(Clone, Debug, PartialEq)]
pub enum Call {
    /// `Prefill(from position, tokens)`.
    Prefill(usize, usize),
    Decode(u32),
    /// `Verify(tokens)`: recorded when asked, whether or not it fails.
    Verify(Vec<u32>),
    /// `Commit(kept)`: recorded when asked, whether or not it fails.
    Commit(usize),
    /// `Snapshot(position, marks)`: recorded when asked, whether or not it was taken.
    Snapshot(usize, Marks),
    /// A restore from a snapshot: `Restore(source, position)`. The live state and a reset are not restores.
    Restore(Source, usize),
    /// `MtpDraft(x, k)`: recorded when asked, whether or not it fails.
    MtpDraft(u32, usize),
}

/// What the fake's MTP drafts (`FakeLive::with_mtp`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FakeMtp {
    /// The model's own continuation after the history and `x`: every draft right.
    Right,
    /// The model's continuation with every draft corrupted: every draft wrong.
    Wrong,
    /// These tokens (the first `k`), whatever the history.
    Fixed(Vec<u32>),
}

/// How the fake's next MTP pass fails (`FakeLive::fail_mtp`).
#[derive(Debug)]
pub struct MtpFault {
    pub error: Error,
    /// A failed submit (the GPU may still be writing): the state is unusable and its KV record is cut back to the
    /// position, as `Session::mtp_draft` does. Otherwise a refused slot: the error alone, the state still usable.
    pub submit: bool,
}

/// What a scripted model picks after a history.
type Script = Box<dyn Fn(&[u32]) -> u32 + Send>;

/// Called after each recorded call.
type CallHook = Box<dyn FnMut(&Call) + Send>;

/// The scripted engine behind `Live`.
pub struct FakeLive {
    ctx: usize,
    script: Script,
    /// The logit at the scripted token; 0.0 elsewhere. Flatten it to give the sampler something to draw from.
    pub peak: f32,
    tokens: Vec<u32>,
    kv: KvRecord,
    usable: bool,
    tree: Tree,
    pub calls: Vec<Call>,
    /// The error the prefill or decode at this step returns (steps count the prefill and decode calls that pass
    /// validation, from 0, over the fake's life: a refused call, `EmptyPrefill`, `PrefillTooLong` or `ContextFull`,
    /// takes no step); it marks the state unusable and cuts the KV record, as `Session` does for a non-validation error.
    pub fail_at: Option<(usize, Error)>,
    /// Returned by the next restore from a snapshot, once, as `Snapshots::restore` fails after its copy submit: the
    /// KV record is cut back to the common prefix with the snapshot and the state is unusable.
    pub fail_restore: Option<Error>,
    /// Returned by the next `snapshot`, once.
    pub fail_snapshot: Option<Error>,
    /// Returned by the next `check_budget`, once.
    pub fail_budget: Option<Error>,
    /// Returned by the next `verify`, once, as a failed submit would: the KV record is cut to the step's start, the state
    /// is unusable and nothing is pending. It does not count as a `fail_at` step.
    pub fail_verify: Option<Error>,
    /// Returned by the next `commit`, once, as a failed replay would: the same write rule as `fail_verify`, with the
    /// tokens cut back to the step's start.
    pub fail_commit: Option<Error>,
    /// `(i, error)`: reading row `i` fails with `error`, once, as a non-finite row does: the tokens and the KV record
    /// are cut back to the step's start, the state is unusable and nothing is pending.
    pub fail_row: Option<(usize, Error)>,
    /// Returned by the next `mtp_draft`, once, as a failed MTP pass would (`MtpFault`). Nothing pending moves either way.
    pub fail_mtp: Option<MtpFault>,
    mtp: Option<FakeMtp>,
    /// Where the pending verify started, between `verify` and `commit`.
    pending: Option<usize>,
    steps: usize,
    /// Whether `mark_unusable` was called, and how often.
    pub unusable_marks: usize,
    /// Counts the same calls into a counter a test can read from outside (`FakeLoader` shares one through its log).
    pub mark_counter: Option<Arc<AtomicUsize>>,
    pub stats: Stats,
    /// Added to `stats` by each prefill call and each decode call: (hits, staged, misses).
    pub prefill_routes: (u64, u64, u64),
    pub decode_routes: (u64, u64, u64),
    pub expert_bytes: u64,
    hooks: Vec<CallHook>,
}

impl FakeLive {
    /// A fake of `ctx` positions with a pool of 1 GiB and the tiny model's sizes.
    pub fn new(ctx: usize, script: impl Fn(&[u32]) -> u32 + Send + 'static) -> FakeLive {
        FakeLive::with_limit(ctx, 1 << 30, script)
    }

    pub fn with_limit(
        ctx: usize,
        limit: u64,
        script: impl Fn(&[u32]) -> u32 + Send + 'static,
    ) -> FakeLive {
        FakeLive {
            ctx,
            script: Box::new(script),
            peak: 10.0,
            tokens: Vec::new(),
            kv: KvRecord::default(),
            usable: true,
            tree: Tree::new(limit, Sizes::of(&Dims::tiny())),
            calls: Vec::new(),
            fail_at: None,
            fail_restore: None,
            fail_snapshot: None,
            fail_budget: None,
            fail_verify: None,
            fail_commit: None,
            fail_row: None,
            fail_mtp: None,
            mtp: None,
            pending: None,
            steps: 0,
            unusable_marks: 0,
            mark_counter: None,
            stats: Stats::default(),
            prefill_routes: (0, 0, 0),
            decode_routes: (0, 0, 0),
            expert_bytes: 1 << 20,
            hooks: Vec::new(),
        }
    }

    /// Gives the fake an MTP that drafts as `mtp` says (`has_mtp` is then true).
    pub fn with_mtp(mut self, mtp: FakeMtp) -> FakeLive {
        self.set_mtp(mtp);
        self
    }

    /// `with_mtp` for a fake a `FakeLoader`'s `tweak` holds.
    pub fn set_mtp(&mut self, mtp: FakeMtp) {
        self.mtp = Some(mtp);
    }

    /// Runs `f` after each call is recorded (a test cancels from here, between prefill segments). Hooks run in the
    /// order they were added.
    pub fn on_call(&mut self, f: impl FnMut(&Call) + Send + 'static) {
        self.hooks.push(Box::new(f));
    }

    /// Drops every hook added with `on_call`.
    pub fn clear_hooks(&mut self) {
        self.hooks.clear();
    }

    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    /// The session's tokens.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn kv_tokens(&self) -> &[u32] {
        self.kv.tokens()
    }

    pub fn usable(&self) -> bool {
        self.usable
    }

    /// As `Session::reset`: the tokens go, the KV record stays, the state is usable again.
    pub fn reset(&mut self) {
        self.tokens.clear();
        self.usable = true;
    }

    fn record(&mut self, call: Call) {
        self.calls.push(call.clone());
        for h in &mut self.hooks {
            h(&call);
        }
    }

    /// The logits the model gives after the current history.
    fn logits(&self) -> Vec<f32> {
        self.logits_after(self.tokens.len())
    }

    /// The logits the model gives after the first `n` tokens of the history: what a decode that left the state at `n`
    /// returns.
    fn logits_after(&self, n: usize) -> Vec<f32> {
        let mut l = vec![0.0; VOCAB];
        l[(self.script)(&self.tokens[..n]) as usize % VOCAB] = self.peak;
        l
    }

    /// The injected failure for this step, applied as `Session` would: a write that fails past validation cuts the
    /// record back to `at` and marks the state unusable.
    fn step(&mut self, at: usize) -> Result<(), Error> {
        let step = self.steps;
        self.steps += 1;
        if self.fail_at.as_ref().is_some_and(|f| f.0 == step) {
            let (_, e) = self.fail_at.take().expect("checked");
            if !is_validation(&e) {
                self.kv.failed(at);
                self.usable = false;
            }
            return Err(e);
        }
        Ok(())
    }

    /// Takes a snapshot at `tokens`, as `Snapshots::take` plans it: a touch, a state for a stateless node or a new
    /// node, after making room; a refusal is counted as a skip.
    fn put(&mut self, tokens: &[u32], marks: Marks) {
        let plan = self.tree.plan(tokens);
        if let Plan::Touch(id) = plan {
            self.tree.touch(id, marks);
            return;
        }
        let need = self.tree.need(plan, tokens.len());
        if self.tree.make_room(need, tokens).is_err() {
            self.tree.count_skip();
            return;
        }
        match plan {
            Plan::New { parent, .. } => {
                self.tree.insert(tokens, parent, marks);
            }
            Plan::AddState(id) => self.tree.add_state(id, marks),
            Plan::Touch(_) => unreachable!("handled above"),
        }
    }

    /// Brings parked snapshots back into the tree, parents first.
    pub fn adopt(&mut self, nodes: &[(Vec<u32>, Marks)]) {
        for (tokens, marks) in nodes {
            self.put(tokens, *marks);
        }
    }

    /// The nodes that hold a state, parents first: what `FakeLoader::park` carries.
    pub fn parkable(&self) -> Vec<(Vec<u32>, Marks)> {
        let mut nodes: Vec<(Vec<u32>, Marks)> = self
            .tree
            .ids()
            .into_iter()
            .filter_map(|id| self.tree.node(id))
            .filter(|n| n.has_state)
            .map(|n| (n.tokens.clone(), n.marks))
            .collect();
        nodes.sort_by_key(|(t, _)| t.len());
        nodes
    }

    fn route(&mut self, r: (u64, u64, u64)) {
        self.stats.hits += r.0;
        self.stats.staged += r.1;
        self.stats.misses += r.2;
        self.stats.steps += 1;
    }
}

impl Live for FakeLive {
    fn ctx(&self) -> usize {
        self.ctx
    }

    fn pos(&self) -> usize {
        self.tokens.len()
    }

    fn restore_for(&mut self, prompt: &[u32]) -> Result<Restored, Error> {
        assert!(self.pending.is_none(), "a restore with a verify pending");
        let live = self.usable.then_some(self.tokens.as_slice());
        let c = self.tree.choose(prompt, live);
        match (c.source, c.snap) {
            (Source::None, _) => {
                self.reset();
                Ok(Restored {
                    source: Source::None,
                    pos: 0,
                    diverged_at: c.diverged_at,
                    bytes: 0,
                })
            }
            (Source::Live, _) => Ok(Restored {
                source: Source::Live,
                pos: c.pos,
                diverged_at: c.diverged_at,
                bytes: 0,
            }),
            (source, Some(id)) => {
                let node = self.tree.node(id).expect("the chosen node");
                let tokens = node.tokens.clone();
                let sizes = self.tree.sizes();
                let from = lcp(self.kv.tokens(), &tokens);
                let bytes = sizes.state + sizes.segment(from, tokens.len());
                self.record(Call::Restore(source, c.pos));
                if let Some(e) = self.fail_restore.take() {
                    self.kv.failed(from);
                    self.usable = false;
                    return Err(e);
                }
                self.kv.set(&tokens);
                self.tokens = tokens;
                self.usable = true;
                self.tree.touch(id, Marks::default());
                self.tree.count_restore(bytes);
                Ok(Restored {
                    source,
                    pos: c.pos,
                    diverged_at: c.diverged_at,
                    bytes,
                })
            }
            (source, None) => unreachable!("a {source:?} choice names a snapshot"),
        }
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, Error> {
        assert!(self.pending.is_none(), "a prefill with a verify pending");
        let at = self.tokens.len();
        if tokens.is_empty() {
            return Err(Error::EmptyPrefill);
        }
        if at + tokens.len() > self.ctx {
            return Err(Error::PrefillTooLong {
                pos: at,
                tokens: tokens.len(),
                ctx: self.ctx,
            });
        }
        self.record(Call::Prefill(at, tokens.len()));
        self.step(at)?;
        self.kv.wrote(at, tokens);
        self.tokens.extend_from_slice(tokens);
        self.route(self.prefill_routes);
        Ok(self.logits())
    }

    fn decode(&mut self, token: u32) -> Result<Vec<f32>, Error> {
        assert!(self.pending.is_none(), "a decode with a verify pending");
        let at = self.tokens.len();
        if at >= self.ctx {
            return Err(Error::ContextFull { ctx: self.ctx });
        }
        self.record(Call::Decode(token));
        self.step(at)?;
        self.kv.wrote(at, &[token]);
        self.tokens.push(token);
        self.route(self.decode_routes);
        Ok(self.logits())
    }

    fn verify(&mut self, tokens: &[u32]) -> Result<(), Error> {
        assert!(self.pending.is_none(), "a verify with another pending");
        let at = self.tokens.len();
        if tokens.is_empty() || tokens.len() > MAX_VERIFY {
            return Err(Error::VerifyShape {
                tokens: tokens.len(),
                most: MAX_VERIFY,
            });
        }
        if at + tokens.len() > self.ctx {
            return Err(Error::PrefillTooLong {
                pos: at,
                tokens: tokens.len(),
                ctx: self.ctx,
            });
        }
        self.record(Call::Verify(tokens.to_vec()));
        if let Some(e) = self.fail_verify.take() {
            self.kv.failed(at);
            self.usable = false;
            return Err(e);
        }
        self.kv.wrote(at, tokens);
        self.tokens.extend_from_slice(tokens);
        self.pending = Some(at);
        self.route(self.decode_routes);
        Ok(())
    }

    fn row(&mut self, i: usize) -> Result<Vec<f32>, Error> {
        let start = self.pending.expect("a row with no verify pending");
        assert!(start + i < self.tokens.len(), "row {i} past the verify");
        if self.fail_row.as_ref().is_some_and(|f| f.0 == i) {
            let (_, e) = self.fail_row.take().expect("checked");
            self.pending = None;
            self.tokens.truncate(start);
            self.kv.failed(start);
            self.usable = false;
            return Err(e);
        }
        Ok(self.logits_after(start + i + 1))
    }

    fn commit(&mut self, kept: usize) -> Result<(), Error> {
        let start = self.pending.expect("a commit with no verify pending");
        assert!(
            (1..=self.tokens.len() - start).contains(&kept),
            "kept {kept} of a verify of {}",
            self.tokens.len() - start
        );
        self.record(Call::Commit(kept));
        self.pending = None;
        if let Some(e) = self.fail_commit.take() {
            self.tokens.truncate(start);
            self.kv.failed(start);
            self.usable = false;
            return Err(e);
        }
        self.tokens.truncate(start + kept);
        self.kv.wrote(start + kept, &[]);
        Ok(())
    }

    fn n_vocab(&self) -> usize {
        VOCAB
    }

    fn mtp_draft(&mut self, x: u32, k: usize) -> Result<Vec<u32>, Error> {
        assert!(self.pending.is_none(), "an MTP pass with a verify pending");
        self.record(Call::MtpDraft(x, k));
        if let Some(f) = self.fail_mtp.take() {
            if f.submit {
                self.kv.failed(self.tokens.len());
                self.usable = false;
            }
            return Err(f.error);
        }
        let Some(mtp) = self.mtp.clone() else {
            return Ok(Vec::new());
        };
        let wrong = |t: u32| (t + 1) % VOCAB as u32;
        Ok(match mtp {
            FakeMtp::Fixed(t) => t.into_iter().take(k).collect(),
            FakeMtp::Right | FakeMtp::Wrong => {
                let mut h = self.tokens.clone();
                h.push(x);
                let mut out = Vec::new();
                for _ in 0..k {
                    let t = (self.script)(&h) % VOCAB as u32;
                    h.push(t);
                    out.push(if mtp == FakeMtp::Wrong { wrong(t) } else { t });
                }
                out
            }
        })
    }

    fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    fn snapshot(&mut self, marks: Marks) -> Result<(), Error> {
        assert!(self.pending.is_none(), "a snapshot with a verify pending");
        self.record(Call::Snapshot(self.tokens.len(), marks));
        if let Some(e) = self.fail_snapshot.take() {
            return Err(e);
        }
        if !self.usable || self.tokens.is_empty() || self.kv.tokens() != self.tokens {
            return Ok(());
        }
        let tokens = self.tokens.clone();
        self.put(&tokens, marks);
        Ok(())
    }

    fn check_budget(&mut self) -> Result<(), Error> {
        self.fail_budget.take().map_or(Ok(()), Err)
    }

    fn cache_stats(&self) -> Stats {
        self.stats
    }

    fn expert_bytes(&self) -> u64 {
        self.expert_bytes
    }

    fn reuse_stats(&self) -> ReuseStats {
        self.tree.stats()
    }

    fn mark_unusable(&mut self) {
        self.usable = false;
        self.unusable_marks += 1;
        if let Some(c) = &self.mark_counter {
            c.fetch_add(1, Ordering::SeqCst);
        }
    }
}

/// Called with the events so far after each one is recorded.
type EventHook = Box<dyn FnMut(&[Event]) + Send>;

/// A sink that records what it is sent, and can close or fail where a test says.
pub struct TestSink {
    pub events: Vec<Event>,
    close_after: Option<usize>,
    fail_after: Option<usize>,
    /// Set it from anywhere to close the sink (`closed` then reads true and `send` fails).
    pub closed_flag: Arc<AtomicBool>,
    hook: Option<EventHook>,
}

impl TestSink {
    pub fn new() -> TestSink {
        TestSink {
            events: Vec::new(),
            close_after: None,
            fail_after: None,
            closed_flag: Arc::new(AtomicBool::new(false)),
            hook: None,
        }
    }

    /// Closed once `n` events were received.
    pub fn closing_after(n: usize) -> TestSink {
        TestSink {
            close_after: Some(n),
            ..TestSink::new()
        }
    }

    /// `send` fails once `n` events were received (the failed event is not recorded).
    pub fn failing_after(n: usize) -> TestSink {
        TestSink {
            fail_after: Some(n),
            ..TestSink::new()
        }
    }

    /// Runs `f` with the events so far after each one is recorded.
    pub fn on_event(&mut self, f: impl FnMut(&[Event]) + Send + 'static) {
        self.hook = Some(Box::new(f));
    }

    /// The ids of the `Token` events.
    pub fn tokens(&self) -> Vec<u32> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Token { id, .. } => Some(*id),
                _ => None,
            })
            .collect()
    }
}

impl Default for TestSink {
    fn default() -> TestSink {
        TestSink::new()
    }
}

impl EventSink for TestSink {
    fn send(&mut self, event: Event) -> bool {
        if self.closed() || self.fail_after.is_some_and(|n| self.events.len() >= n) {
            return false;
        }
        self.events.push(event);
        if let Some(h) = &mut self.hook {
            h(&self.events);
        }
        true
    }

    fn closed(&self) -> bool {
        self.closed_flag.load(Ordering::SeqCst)
            || self.close_after.is_some_and(|n| self.events.len() >= n)
    }
}

/// One call into a `FakeLoader`, in order.
#[derive(Clone, Debug, PartialEq)]
pub enum LoaderCall {
    /// `Load { cache_slots, had_parked }`: recorded when the call starts, whether or not it fails.
    Load {
        cache_slots: Option<usize>,
        had_parked: bool,
    },
    /// An unload that parked: the nodes that held a state, parents first.
    Park(Vec<(Vec<u32>, Marks)>),
    /// An unload that did not park.
    Unload,
    /// An unload whose park failed; the live state is gone.
    ParkFailed,
}

#[derive(Default)]
struct LogInner {
    calls: Vec<LoaderCall>,
    live: Vec<Call>,
}

/// What a `FakeLoader` and the `FakeLive`s it made were asked, shared with the test that holds the worker.
#[derive(Default)]
pub struct LoaderLog {
    inner: Mutex<LogInner>,
    changed: Condvar,
    marks: Arc<AtomicUsize>,
}

impl LoaderLog {
    fn with<R>(&self, f: impl FnOnce(&mut LogInner) -> R) -> R {
        // A panic in a fake's prefill is a test (the worker dies); the log outlives it.
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let r = f(&mut g);
        self.changed.notify_all();
        r
    }

    /// Every loader call, in order.
    pub fn calls(&self) -> Vec<LoaderCall> {
        self.with(|l| l.calls.clone())
    }

    /// Every call into any `FakeLive` the loader made, in order.
    pub fn live_calls(&self) -> Vec<Call> {
        self.with(|l| l.live.clone())
    }

    /// `(cache_slots, had_parked)` of each load call.
    pub fn loads(&self) -> Vec<(Option<usize>, bool)> {
        self.calls()
            .into_iter()
            .filter_map(|c| match c {
                LoaderCall::Load {
                    cache_slots,
                    had_parked,
                } => Some((cache_slots, had_parked)),
                _ => None,
            })
            .collect()
    }

    /// The nodes each park carried.
    pub fn parks(&self) -> Vec<Vec<(Vec<u32>, Marks)>> {
        self.calls()
            .into_iter()
            .filter_map(|c| match c {
                LoaderCall::Park(n) => Some(n),
                _ => None,
            })
            .collect()
    }

    /// How many times the worker called `mark_unusable` on any `FakeLive` the loader made.
    pub fn unusable_marks(&self) -> usize {
        self.marks.load(Ordering::SeqCst)
    }

    /// How many unloads did not park.
    pub fn unloads(&self) -> usize {
        self.calls()
            .iter()
            .filter(|c| **c == LoaderCall::Unload)
            .count()
    }

    /// Waits up to 5 s for `f` to hold over the loader calls; false on timeout.
    pub fn wait_for(&self, f: impl Fn(&[LoaderCall]) -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        while !f(&g.calls) {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            g = self
                .changed
                .wait_timeout(g, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        true
    }
}

/// What a `FakeLoader` parks: the fake tree's nodes that held a state.
pub struct FakeParked {
    nodes: Vec<(Vec<u32>, Marks)>,
}

impl ParkedStats for FakeParked {
    fn count(&self) -> usize {
        self.nodes.len()
    }

    fn bytes(&self) -> u64 {
        let sizes = Sizes::of(&Dims::tiny());
        self.nodes
            .iter()
            .map(|(t, _)| sizes.state + sizes.segment(0, t.len()))
            .sum()
    }
}

type Tweak = Box<dyn FnMut(usize, &mut FakeLive) + Send>;

/// A script several `FakeLive`s share.
type SharedScript = Arc<dyn Fn(&[u32]) -> u32 + Send + Sync>;

/// A scripted `Loader`: each load call takes the next entry of `script_loads` (an `Ok` when they run out), reports
/// `load_time`, and returns a `FakeLive` of the given script. `on_load` runs inside each call after it is logged (a
/// test blocks the load there), `tweak` runs on each new `FakeLive` (a test sets its failures there). Parking carries
/// the tree's stateful nodes and loading brings them back.
pub struct FakeLoader {
    ctx: usize,
    script: SharedScript,
    limit: u64,
    loads: VecDeque<Result<(), LoadFailure>>,
    load_time: Duration,
    slots_per_layer: usize,
    vram_used: Box<dyn FnMut(usize) -> Option<u64> + Send>,
    park_time: Duration,
    on_load: Option<Box<dyn FnMut(usize) + Send>>,
    tweak: Option<Tweak>,
    park_errors: VecDeque<Error>,
    calls: usize,
    log: Arc<LoaderLog>,
}

impl FakeLoader {
    pub fn new(ctx: usize, script: impl Fn(&[u32]) -> u32 + Send + Sync + 'static) -> FakeLoader {
        FakeLoader {
            ctx,
            script: Arc::new(script),
            limit: 1 << 30,
            loads: VecDeque::new(),
            load_time: Duration::from_millis(7),
            slots_per_layer: 128,
            vram_used: Box::new(|_| Some(1 << 30)),
            park_time: Duration::ZERO,
            on_load: None,
            tweak: None,
            park_errors: VecDeque::new(),
            calls: 0,
            log: Arc::new(LoaderLog::default()),
        }
    }

    /// The log, for the test to read while the worker owns the loader.
    pub fn log(&self) -> Arc<LoaderLog> {
        Arc::clone(&self.log)
    }

    /// The outcome of the next load calls, one each.
    pub fn script_loads(mut self, loads: Vec<Result<(), LoadFailure>>) -> FakeLoader {
        self.loads = loads.into();
        self
    }

    /// The `LoadInfo.time` every load reports (it is not slept).
    pub fn load_time(mut self, time: Duration) -> FakeLoader {
        self.load_time = time;
        self
    }

    /// The `LoadInfo.vram_used` of each load: `f(load ordinal from 0)`. The default is `Some(1 << 30)`.
    pub fn vram_used(mut self, f: impl FnMut(usize) -> Option<u64> + Send + 'static) -> FakeLoader {
        self.vram_used = Box::new(f);
        self
    }

    /// How long each park sleeps, on the worker thread, before it parks (a test reaches the worker mid-park).
    pub fn park_time(mut self, time: Duration) -> FakeLoader {
        self.park_time = time;
        self
    }

    /// The errors the next park calls fail with, one each (the live state is gone, as a real failed park leaves it).
    pub fn fail_parks(mut self, errors: Vec<Error>) -> FakeLoader {
        self.park_errors = errors.into();
        self
    }

    /// Runs `f(call index)` inside each load call, after the call is logged and before its outcome.
    pub fn on_load(mut self, f: impl FnMut(usize) + Send + 'static) -> FakeLoader {
        self.on_load = Some(Box::new(f));
        self
    }

    /// Runs `f(call index, live)` on each `FakeLive` the loader makes, before it is returned.
    pub fn tweak(mut self, f: impl FnMut(usize, &mut FakeLive) + Send + 'static) -> FakeLoader {
        self.tweak = Some(Box::new(f));
        self
    }
}

impl Loader for FakeLoader {
    type Live = FakeLive;
    type Parked = FakeParked;

    fn load(
        &mut self,
        parked: &mut Option<FakeParked>,
        cache_slots: Option<usize>,
    ) -> Result<(FakeLive, LoadInfo), LoadFailure> {
        let index = self.calls;
        self.calls += 1;
        self.log.with(|l| {
            l.calls.push(LoaderCall::Load {
                cache_slots,
                had_parked: parked.is_some(),
            })
        });
        if let Some(f) = &mut self.on_load {
            f(index);
        }
        if let Some(Err(failure)) = self.loads.pop_front() {
            return Err(failure);
        }
        let script = Arc::clone(&self.script);
        let mut live = FakeLive::with_limit(self.ctx, self.limit, move |h| script(h));
        let log = Arc::clone(&self.log);
        live.on_call(move |c| log.with(|l| l.live.push(c.clone())));
        live.mark_counter = Some(Arc::clone(&self.log.marks));
        let held = parked.take();
        let restored = held.as_ref().map_or(0, |p| p.nodes.len());
        if let Some(p) = &held {
            live.adopt(&p.nodes);
        }
        if let Some(t) = &mut self.tweak {
            t(index, &mut live);
        }
        let info = LoadInfo {
            time: self.load_time,
            device: "fake".into(),
            device_uuid: [0; 16],
            driver: "fake".into(),
            slots_per_layer: cache_slots.unwrap_or(self.slots_per_layer),
            vram_weights: 0,
            vram_session: 0,
            vram_used: (self.vram_used)(index),
            pool: PoolLimit {
                bytes: self.limit,
                requested: self.limit,
                cut: false,
            },
            unparked: Unparked {
                restored,
                dropped: 0,
            },
            retried_with_slots: None,
        };
        Ok((live, info))
    }

    fn park(&mut self, live: FakeLive) -> Result<FakeParked, Error> {
        std::thread::sleep(self.park_time);
        if let Some(e) = self.park_errors.pop_front() {
            self.log.with(|l| l.calls.push(LoaderCall::ParkFailed));
            return Err(e);
        }
        let nodes = live.parkable();
        self.log
            .with(|l| l.calls.push(LoaderCall::Park(nodes.clone())));
        Ok(FakeParked { nodes })
    }

    fn unload(&mut self, _live: FakeLive) {
        self.log.with(|l| l.calls.push(LoaderCall::Unload));
    }
}
