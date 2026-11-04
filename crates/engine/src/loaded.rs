//! The real engine behind the worker: `Loaded` is a loaded engine with its session and
//! snapshots, the `Live` the generation loop drives; `EngineLoader` loads one, parks it and unloads it for the worker
//! thread. `fake::FakeLive` and `FakeLoader` stand in for both in the CI's tests.
use crate::pool::limit_at_load;
use crate::request::Source;
use crate::reuse::{Marks, ReuseStats};
use crate::turn::{Live, Restored};
use crate::worker::{LoadFailure, LoadInfo, Loader};
use crate::{Config, Engine, Error, Parked, Session, Snapshots, Stats, Unparked};
use qwen36::dims::Dims;
use std::time::Instant;

/// A loaded engine, its one session and the snapshots taken of it. The fields are in drop order: every `vk::Buffer`
/// holds the device's `Arc`, so the snapshots (the pool's buffers) and the session go before the engine.
/// Parking is the exception, on purpose: `unload` drops the session and the engine first, and the pool, which keeps the
/// device alive through its buffers, is copied into host RAM afterwards.
pub struct Loaded {
    snaps: Snapshots,
    session: Session,
    engine: Engine,
}

impl Loaded {
    /// `session` and `snaps` must be `engine`'s own (every call checks, and answers `WrongEngine` otherwise).
    pub fn new(engine: Engine, session: Session, snaps: Snapshots) -> Loaded {
        Loaded {
            snaps,
            session,
            engine,
        }
    }

    /// The three parts, for tests and diagnostics that reach past `Live`.
    pub fn parts(&mut self) -> (&Engine, &mut Session, &mut Snapshots) {
        (&self.engine, &mut self.session, &mut self.snaps)
    }

    /// Fences the snapshots with the engine, drops the session and the engine, then copies the pool into host memory:
    /// the engine's host expert copies are freed before the pool is read, so parking needs no RAM beyond
    /// the load-time limit's margin. The pool's buffers hold the device's `Arc` and stay mapped after the `Engine`
    /// drops, which waits for the queues. On `Err` the snapshots are gone and the rest is dropped all the same.
    pub fn unload(self) -> Result<Parked, Error> {
        let Loaded {
            mut snaps,
            session,
            engine,
        } = self;
        let fenced = snaps.fence_for_park(&engine);
        drop(session);
        engine.unload();
        fenced?;
        Ok(snaps.into_parked())
    }

    /// Drops everything without parking. An error settling the snapshots' copies is ignored: the device may be lost,
    /// and then nothing is running (`Snapshots`' `Drop` waits for the device too, whatever happens here).
    pub fn drop_all(mut self) {
        let _ = self.snaps.settle(&self.engine);
        drop(self);
    }
}

impl Live for Loaded {
    fn ctx(&self) -> usize {
        self.session.ctx()
    }

    fn pos(&self) -> usize {
        self.session.pos()
    }

    fn restore_for(&mut self, prompt: &[u32]) -> Result<Restored, Error> {
        self.snaps.settle(&self.engine)?;
        let c = self.snaps.choose(prompt, &self.session);
        match (c.source, c.snap) {
            (Source::None, _) => {
                self.session.reset(&self.engine)?;
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
                let r = self.snaps.restore(&self.engine, &mut self.session, id)?;
                Ok(Restored {
                    source,
                    pos: c.pos,
                    diverged_at: c.diverged_at,
                    bytes: r.bytes,
                })
            }
            (source, None) => unreachable!("a {source:?} choice names a snapshot"),
        }
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, Error> {
        self.session.prefill(&self.engine, tokens)
    }

    fn decode(&mut self, token: u32) -> Result<Vec<f32>, Error> {
        self.session.decode(&self.engine, token)
    }

    fn verify(&mut self, tokens: &[u32]) -> Result<(), Error> {
        self.session.verify(&self.engine, tokens)
    }

    fn row(&mut self, i: usize) -> Result<Vec<f32>, Error> {
        self.session.row(&self.engine, i)
    }

    fn commit(&mut self, kept: usize) -> Result<(), Error> {
        self.session.commit(&self.engine, kept)
    }

    fn n_vocab(&self) -> usize {
        self.engine.dims.n_vocab
    }

    fn mtp_draft(&mut self, x: u32, k: usize) -> Result<Vec<u32>, Error> {
        self.session.mtp_draft(&self.engine, x, k)
    }

    fn has_mtp(&self) -> bool {
        self.session.has_mtp()
    }

    fn most_verify(&self) -> usize {
        crate::spec::MAX_VERIFY.min(self.engine.prefill_chunk())
    }

    fn snapshot(&mut self, marks: Marks) -> Result<(), Error> {
        self.snaps
            .take(&self.engine, &mut self.session, marks)
            .map(|_| ())
    }

    fn check_budget(&mut self) -> Result<(), Error> {
        self.engine.check_budget()
    }

    fn cache_stats(&self) -> Stats {
        self.engine.stats()
    }

    fn expert_bytes(&self) -> u64 {
        self.engine.expert_bytes()
    }

    fn reuse_stats(&self) -> ReuseStats {
        self.snaps.stats()
    }

    fn mark_unusable(&mut self) {
        self.session.usable = false;
    }
}

/// Loads, parks and unloads the real engine for the worker thread.
pub struct EngineLoader {
    engine: Config,
    dims: Dims,
}

impl EngineLoader {
    pub fn new(engine: Config, dims: Dims) -> EngineLoader {
        EngineLoader { engine, dims }
    }
}

impl Loader for EngineLoader {
    type Live = Loaded;
    type Parked = Parked;

    fn load(
        &mut self,
        parked: &mut Option<Parked>,
        cache_slots: Option<usize>,
    ) -> Result<(Loaded, LoadInfo), LoadFailure> {
        let started = Instant::now();
        let configured = self.engine.cache_slots;
        self.engine.cache_slots = cache_slots.or(configured);
        let opened = Engine::load_dims(&self.engine, &self.dims);
        self.engine.cache_slots = configured;
        // No count to halve if the engine did not open: `fit_cache` had already sized the cache to the budget.
        let engine = opened.map_err(|error| LoadFailure { error, slots: None })?;
        // From here every failure knows the slot count the cache was fitted to. The locals drop in reverse order:
        // snapshots, session, engine.
        let slots = engine.stats().slots_per_layer;
        let fail = |error| LoadFailure {
            error,
            slots: Some(slots),
        };
        let session = engine.session(self.engine.ctx).map_err(fail)?;
        let reclaimable = parked.as_ref().map_or(0, |p| p.bytes());
        let pool =
            limit_at_load(&engine, self.engine.snapshot_pool_gib, reclaimable).map_err(fail)?;
        // The parked snapshots are taken only here: a failure above leaves them for the retry.
        let (snaps, unparked) = match parked.take() {
            Some(p) => Snapshots::unpark(&engine, pool, p).map_err(fail)?,
            None => (Snapshots::new(&engine, pool), Unparked::default()),
        };
        let info = LoadInfo {
            time: started.elapsed(),
            device: engine.device_name().to_string(),
            device_uuid: engine.device_uuid(),
            driver: engine.driver().to_string(),
            slots_per_layer: slots,
            mtp_slots: engine.stats().mtp_slots,
            mtp_slot_bytes: engine.stats().mtp_slot_bytes,
            vram_weights: engine.vram_bytes(),
            vram_session: session.vram_bytes(),
            // The extension is required at device creation; a usage of 0 would mean no figure, not a failed load.
            vram_used: engine
                .vram_heap()
                .ok()
                .map(|h| h.usage)
                .filter(|&usage| usage > 0),
            pool,
            unparked,
            retried_with_slots: None,
        };
        Ok((Loaded::new(engine, session, snaps), info))
    }

    fn park(&mut self, live: Loaded) -> Result<Parked, Error> {
        live.unload()
    }

    fn unload(&mut self, live: Loaded) {
        live.drop_all();
    }
}
