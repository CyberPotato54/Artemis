//! Snapshots in host RAM: a session's DeltaNet state and KV copied into the
//! host pool, restored from it, demoted, and parked across an engine's unload. `reuse::Tree` decides what the pool
//! holds; this module does the copying.
//!
//! The snapshot copies' recorders come from `Device::record`, not `Engine::record`: they are not model steps, and a
//! profile does not time them.
use crate::pool::{Pool, PoolLimit, PoolSlice};
use crate::reuse::{kv_copies, lcp, unpark_order};
use crate::{Choice, Engine, Error, Freed, Marks, Plan, ReuseStats, Session, SnapId, Tree};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// `take_log` keeps this many takes; older ones go first.
const LOG_CAP: usize = 4096;

/// A take whose pool allocation fails after the tree made room demotes up to this many more states, one a retry.
const ALLOC_RETRIES: u64 = 4;

/// What a take did.
#[derive(Clone, Copy, Debug)]
pub struct Taken {
    pub id: SnapId,
    /// The bytes the take copied: the state and the KV segment (0 when the snapshot already existed).
    pub bytes: u64,
    /// The blocking submit of the state's copies.
    pub state_time: Duration,
    /// The KV segment's copy, from its submit to its landing; only when `wait_each` is on (the copy is otherwise left
    /// running), and `None` when there was no segment to copy.
    pub kv_time: Option<Duration>,
}

/// What a restore did.
#[derive(Clone, Copy, Debug)]
pub struct Restore {
    /// The bytes copied: the state and the KV past the common prefix.
    pub bytes: u64,
    /// The first KV position the restore copied: the common prefix of the session's KV and the snapshot's tokens.
    pub kv_from: usize,
    /// The blocking submit of all the copies.
    pub time: Duration,
}

/// What `unpark` brought back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unparked {
    pub restored: usize,
    /// The parked nodes not brought back: they did not fit the new limit, or a pool allocation was refused (a node
    /// whose ancestor was dropped goes with it).
    pub dropped: usize,
}

/// One node's pool memory: a slice per KV buffer for its segment, and the state if it holds one.
struct NodeMem {
    kv: Vec<PoolSlice>,
    state: Option<PoolSlice>,
}

/// The snapshots in host RAM: the tree, the pool and the node-to-pool map. Every pool read, free and drop waits for the
/// transfer queue's pending copies first (`settle`).
pub struct Snapshots {
    /// The engine that made it (`Engine::id`).
    engine: u64,
    /// Waits for the device without an engine: `Drop` has none.
    idle: vk::IdleHandle,
    tree: Tree,
    pool: Pool,
    mem: BTreeMap<SnapId, NodeMem>,
    /// The highest transfer value any take returned.
    newest: u64,
    /// The highest transfer value `settle` has waited for.
    settled: u64,
    pool_cut: bool,
    wait_each: bool,
    log: Vec<Taken>,
    /// Tests: a VRAM scratch every KV copy fills first, so the copies land late.
    delay: Option<(vk::Arena, vk::Slice)>,
}

impl Snapshots {
    /// An empty set of snapshots on `limit` bytes of pool.
    pub fn new(e: &Engine, limit: PoolLimit) -> Snapshots {
        Snapshots {
            engine: e.id,
            idle: e.dev.idle_handle(),
            tree: Tree::new(limit.bytes, crate::Sizes::of(&e.dims)),
            pool: Pool::new(limit.bytes),
            mem: BTreeMap::new(),
            newest: 0,
            settled: 0,
            pool_cut: limit.cut,
            wait_each: false,
            log: Vec::new(),
            delay: None,
        }
    }

    /// Refuses an engine other than the one that made this.
    fn check(&self, e: &Engine) -> Result<(), Error> {
        if self.engine == e.id {
            Ok(())
        } else {
            Err(Error::WrongEngine)
        }
    }

    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    /// The reuse counters, with `pool_cut` from the limit this was made with and `chunks_refused` from the pool.
    pub fn stats(&self) -> ReuseStats {
        ReuseStats {
            pool_cut: self.pool_cut,
            chunks_refused: self.pool.refused(),
            ..self.tree.stats()
        }
    }

    /// The state a request for `prompt` restores from; the session counts only when it is usable.
    pub fn choose(&self, prompt: &[u32], s: &Session) -> Choice {
        self.tree.choose(prompt, s.usable().then(|| s.tokens()))
    }

    /// Waits for every KV copy a take left running.
    pub fn settle(&mut self, e: &Engine) -> Result<(), Error> {
        self.check(e)?;
        if self.settled < self.newest {
            let waited = e.dev.wait(vk::QueueKind::Transfer, self.newest);
            // Only a lost device ends the copies for certain (nothing will land); after a timeout they may still be
            // running, so they stay pending for the next restore, free or drop to wait for.
            if matches!(waited, Ok(()) | Err(vk::Error::DeviceLost { .. })) {
                self.settled = self.newest;
            }
            waited?;
        }
        Ok(())
    }

    /// Snapshots the session's tokens, marked `marks`: a node under the deepest node whose tokens are a prefix. The
    /// state's copies go in a blocking compute submit (the next chunk changes the state) that ends with a host barrier,
    /// so parking reads them; the KV segment's go out on the transfer queue, after that submit, and run beside the next
    /// chunk (`Session` waits for them before it overwrites what they read). `Ok(None)` when the snapshot is skipped
    /// (counted: the pool is off, too small, out of room or refused a block; room freed for it stays freed) or the
    /// session is not worth one (unusable, empty, or its KV does not match its tokens: not counted). A snapshot that
    /// already exists is touched and reported with 0 bytes. When the tree has room but the pool finds no run for the
    /// slices, up to `ALLOC_RETRIES` more states are demoted, least recently used first, before the skip
    /// (`Tree::make_room_and_alloc`).
    pub fn take(
        &mut self,
        e: &Engine,
        s: &mut Session,
        marks: Marks,
    ) -> Result<Option<Taken>, Error> {
        self.check(e)?;
        if s.engine != e.id {
            return Err(Error::WrongEngine);
        }
        assert!(!s.pending(), "a snapshot with a verify step pending");
        if !s.usable() || s.tokens().is_empty() || s.kv_tokens() != s.tokens() {
            return Ok(None);
        }
        let tokens = s.tokens().to_vec();
        let pos = tokens.len();
        let plan = self.tree.plan(&tokens);
        if let Plan::Touch(id) = plan {
            self.tree.touch(id, marks);
            return Ok(Some(Taken {
                id,
                bytes: 0,
                state_time: Duration::ZERO,
                kv_time: None,
            }));
        }
        let need = self.tree.need(plan, pos);
        let sizes = self.tree.sizes();
        let b = sizes.kv_buffer_per_pos();
        let kv_from = match plan {
            Plan::New { from, .. } => from,
            _ => pos,
        };
        let seg = (pos - kv_from) as u64 * b;
        let mut want = vec![sizes.state];
        if seg > 0 {
            want.extend(std::iter::repeat_n(seg, sizes.kv_buffers));
        }
        // What the tree frees below is freed in the pool at once: a copy still running may be writing to it, so wait
        // for them first, while a failed wait can still leave the tree alone.
        self.settle(e)?;
        // The tree counts exact bytes and the pool hands out aligned runs of its chunks, so a fit in the tree can still
        // find no run (fragmented, or every chunk open). Each retry demotes one more state, with its slices.
        let (pool, mem) = (&mut self.pool, &mut self.mem);
        let got = self
            .tree
            .make_room_and_alloc(need, &tokens, ALLOC_RETRIES, |freed| {
                free_nodes(pool, mem, freed);
                alloc_slices(pool, &e.dev, &want)
            })?;
        let Some(got) = got else {
            self.tree.count_skip();
            return Ok(None);
        };
        let state_slices = s.state_slices();
        assert_eq!(
            state_slices.iter().map(|x| x.size).sum::<u64>(),
            sizes.state,
            "the session's state is not the size the snapshot tree counts"
        );
        // The state: 60 copies in one blocking submit, ending with the host barrier.
        let mut r = match e.dev.record() {
            Ok(r) => r,
            Err(err) => {
                self.free_all(&got);
                return Err(err.into());
            }
        };
        let to = self.pool.slice(got[0]);
        let mut at = 0;
        for x in &state_slices {
            r.copy(*x, to.sub(at, x.size));
            at += x.size;
        }
        r.host_barrier();
        let started = Instant::now();
        if let Err(err) = e.dev.submit(r) {
            self.free_all(&got);
            return Err(err.into());
        }
        let state_time = started.elapsed();
        // The KV segment: the transfer queue, after the state's submit.
        let mut copy = None;
        if seg > 0 {
            let regions: Vec<(vk::Slice, vk::Slice)> = s
                .kv_slices()
                .iter()
                .zip(&got[1..])
                .map(|(src, &ps)| (src.sub(kv_from as u64 * b, seg), self.pool.slice(ps)))
                .collect();
            let opts = vk::StageOpts {
                delay: self.delay.as_ref().map(|d| d.1),
                ..vk::StageOpts::default()
            };
            let started = Instant::now();
            match e
                .dev
                .stage_async(&regions, e.dev.submitted(vk::QueueKind::Compute), opts)
            {
                Ok(v) => copy = Some((v, started)),
                Err(err) => {
                    self.free_all(&got);
                    return Err(err.into());
                }
            }
        }
        let id = match plan {
            Plan::New { parent, .. } => {
                let id = self.tree.insert(&tokens, parent, marks);
                self.mem.insert(
                    id,
                    NodeMem {
                        kv: got[1..].to_vec(),
                        state: Some(got[0]),
                    },
                );
                id
            }
            Plan::AddState(id) => {
                self.tree.add_state(id, marks);
                self.mem.get_mut(&id).expect("a node's memory").state = Some(got[0]);
                id
            }
            Plan::Touch(_) => unreachable!("a touch returned above"),
        };
        let mut kv_time = None;
        if let Some((v, started)) = copy {
            self.newest = self.newest.max(v);
            // Writes below the segment's end wait for it, and so do those below an older copy's end: the queue is
            // FIFO, so the newest value covers both.
            let end = s.kv_copy.map_or(pos, |(end, _)| end.max(pos));
            s.kv_copy = Some((end, v));
            if self.wait_each {
                e.dev.wait(vk::QueueKind::Transfer, v)?;
                kv_time = Some(started.elapsed());
            }
        }
        let taken = Taken {
            id,
            bytes: sizes.state + seg * sizes.kv_buffers as u64,
            state_time,
            kv_time,
        };
        if self.log.len() == LOG_CAP {
            self.log.remove(0);
        }
        self.log.push(taken);
        Ok(Some(taken))
    }

    /// Puts snapshot `id` back into the session: its state, and the KV of its chain past the common prefix of the
    /// session's KV and its tokens, in one blocking compute submit. The session then holds the snapshot's tokens.
    /// `Error::NoSuchSnapshot` when `id` is gone or holds no state.
    pub fn restore(&mut self, e: &Engine, s: &mut Session, id: SnapId) -> Result<Restore, Error> {
        self.check(e)?;
        if s.engine != e.id {
            return Err(Error::WrongEngine);
        }
        assert!(!s.pending(), "a restore with a verify step pending");
        let Some(node) = self.tree.node(id).filter(|n| n.has_state) else {
            return Err(Error::NoSuchSnapshot { id: id.0 });
        };
        let tokens = node.tokens.clone();
        // The copies below overwrite KV a take's copy may still be reading.
        self.settle(e)?;
        let sizes = self.tree.sizes();
        let b = sizes.kv_buffer_per_pos();
        let kv_from = lcp(s.kv_tokens(), &tokens);
        let chain = self.tree.chain(id);
        let kv = s.kv_slices();
        let mut r = e.dev.record()?;
        let state = self.pool.slice(self.mem[&id].state.expect("a state"));
        let mut at = 0;
        for x in s.state_slices() {
            r.copy(state.sub(at, x.size), x);
            at += x.size;
        }
        assert_eq!(at, sizes.state, "the snapshot's state is not the session's");
        let mut bytes = sizes.state;
        for (node, lo, hi) in kv_copies(&chain, kv_from) {
            let start = chain.iter().find(|c| c.0 == node).expect("a chain node").1;
            let n = (hi - lo) as u64 * b;
            for (src, dst) in self.mem[&node].kv.iter().zip(&kv) {
                let src = PoolSlice {
                    offset: src.offset + (lo - start) as u64 * b,
                    size: n,
                    ..*src
                };
                r.copy(self.pool.slice(src), dst.sub(lo as u64 * b, n));
            }
            bytes += sizes.segment(lo, hi);
        }
        let started = Instant::now();
        let done = e.dev.submit(r).map_err(Error::from);
        let time = started.elapsed();
        // The copies may have overwritten the state and the KV from `kv_from` on: a failure from here is a write that
        // failed past validation (the session is unusable until `reset`, its KV record cut back).
        if let Err(err) = done.and_then(|()| s.set_restored(e, &tokens)) {
            s.kv.failed(kv_from);
            if let Some(m) = &mut s.mtp {
                m.on_reset();
            }
            s.usable = false;
            return Err(err);
        }
        self.tree.touch(id, Marks::default());
        self.tree.count_restore(bytes);
        Ok(Restore {
            bytes,
            kv_from,
            time,
        })
    }

    /// Demotes snapshot `id`: it loses its state, and stateless childless nodes are freed up its chain.
    pub fn drop_state(&mut self, e: &Engine, id: SnapId) -> Result<(), Error> {
        self.settle(e)?;
        let freed = self.tree.drop_state(id, &[]);
        self.release(e, &freed)
    }

    /// Measurement: each take waits for its KV copy and reports `kv_time`.
    pub fn wait_each(&mut self, on: bool) {
        self.wait_each = on;
    }

    /// Measurement: every take since the last call (at most the newest 4,096).
    pub fn take_log(&mut self) -> Vec<Taken> {
        std::mem::take(&mut self.log)
    }

    /// Tests: allocates the most one allocation may be, 1 GiB of VRAM, and has every KV copy fill it first, so the
    /// copies land late and a missing wait reads stale bytes. Out of the VRAM the session leaves.
    pub fn delay_copies(&mut self, e: &Engine) -> Result<(), Error> {
        self.check(e)?;
        if self.delay.is_none() {
            let mut a = vk::Arena::new(
                vk::MemKind::Vram,
                vk::MAX_ALLOCATION,
                "the snapshot copy delay (tests)",
            );
            let s = a.alloc(&e.dev, vk::MAX_ALLOCATION)?;
            self.delay = Some((a, s));
        }
        Ok(())
    }

    /// The engine's half of parking: settles, then makes the transfer queue's KV writes visible to the CPU with one
    /// blocking compute submit that waits for it and holds only a host barrier (the pool is `HostCached`). After it
    /// `into_parked` needs no engine, so the engine can go first and free its host memory before the pool is copied.
    /// With another engine it returns `WrongEngine`.
    pub fn fence_for_park(&mut self, e: &Engine) -> Result<(), Error> {
        self.settle(e)?;
        let mut r = e.dev.record()?;
        r.wait(vk::QueueKind::Transfer, self.newest);
        r.host_barrier();
        e.dev.submit(r)?;
        Ok(())
    }

    /// The CPU's half of parking, after `fence_for_park`: reads every node into host `Vec`s, parents first, and drops
    /// the pool's chunks. The pool's buffers hold the device, so this works after the engine is gone.
    pub fn into_parked(self) -> Parked {
        assert!(
            self.settled >= self.newest,
            "into_parked without fence_for_park: a KV copy may still be running"
        );
        // Ids rise with insertion and parents come first, so id order is parents first.
        let ids = self.tree.ids();
        let index: BTreeMap<SnapId, usize> =
            ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let read = |ps: PoolSlice| {
            let mut v = vec![0u8; ps.size as usize];
            self.pool.read(ps, &mut v);
            v
        };
        let nodes = ids
            .iter()
            .map(|id| {
                let n = self.tree.node(*id).expect("a listed node");
                let m = &self.mem[id];
                ParkedNode {
                    tokens: n.tokens.clone(),
                    parent: n.parent.map(|p| index[&p]),
                    marks: n.marks,
                    has_state: n.has_state,
                    taken: n.taken,
                    last_used: n.last_used,
                    kv: m.kv.iter().map(|&ps| read(ps)).collect(),
                    state: m.state.map(read),
                }
            })
            .collect();
        Parked {
            sizes: self.tree.sizes(),
            nodes,
        }
    }

    /// `fence_for_park`, then `into_parked`, with the engine still loaded. With another engine it returns
    /// `WrongEngine` and still consumes `self` (`Drop` frees the snapshots cleanly).
    pub fn park(mut self, e: &Engine) -> Result<Parked, Error> {
        self.fence_for_park(e)?;
        Ok(self.into_parked())
    }

    /// Brings parked snapshots back into a fresh pool of `limit` bytes: the newest first, each with the ancestors it
    /// lacks, while they fit. Each node's `Vec`s are freed as soon as they are copied, so the parked bytes fall as the
    /// pool fills. A refused pool allocation drops that node and its descendants.
    pub fn unpark(
        e: &Engine,
        limit: PoolLimit,
        parked: Parked,
    ) -> Result<(Snapshots, Unparked), Error> {
        let mut snaps = Snapshots::new(e, limit);
        let total = parked.nodes.len();
        let mut nodes = parked.nodes;
        // Parked by another model: nothing in it fits this engine's buffers.
        let order = if parked.sizes == snaps.tree.sizes() {
            let sizes: Vec<(Option<usize>, u64, u64)> = nodes
                .iter()
                .map(|n| (n.parent, n.last_used, n.bytes()))
                .collect();
            unpark_order(&sizes, limit.bytes)
        } else {
            Vec::new()
        };
        let mut picked = vec![false; total];
        for &i in &order {
            picked[i] = true;
        }
        for (n, keep) in nodes.iter_mut().zip(&picked) {
            if !*keep {
                n.kv = Vec::new();
                n.state = None;
            }
        }
        let mut back: BTreeMap<usize, SnapId> = BTreeMap::new();
        for i in order {
            let n = &mut nodes[i];
            let parent = match n.parent {
                None => None,
                Some(p) => match back.get(&p) {
                    Some(&id) => Some(id),
                    // The ancestor was refused: this node goes with it.
                    None => {
                        n.kv = Vec::new();
                        n.state = None;
                        continue;
                    }
                },
            };
            let mut want: Vec<u64> = n.state.iter().map(|v| v.len() as u64).collect();
            want.extend(n.kv.iter().map(|v| v.len() as u64));
            let Some(got) = snaps.alloc_all(e, &want)? else {
                n.kv = Vec::new();
                n.state = None;
                continue;
            };
            let mut slices = got.into_iter();
            let state = n.state.take().map(|v| {
                let ps = slices.next().expect("a slice for the state");
                snaps.pool.write(ps, &v);
                ps
            });
            let kv = std::mem::take(&mut n.kv)
                .into_iter()
                .map(|v| {
                    let ps = slices.next().expect("a slice for a KV buffer");
                    snaps.pool.write(ps, &v);
                    ps
                })
                .collect();
            let id = snaps.tree.insert_parked(
                &n.tokens,
                parent,
                n.marks,
                n.has_state,
                n.taken,
                n.last_used,
            );
            snaps.mem.insert(id, NodeMem { kv, state });
            back.insert(i, id);
        }
        let restored = back.len();
        Ok((
            snaps,
            Unparked {
                restored,
                dropped: total - restored,
            },
        ))
    }

    /// See `alloc_slices`.
    fn alloc_all(&mut self, e: &Engine, sizes: &[u64]) -> Result<Option<Vec<PoolSlice>>, Error> {
        alloc_slices(&mut self.pool, &e.dev, sizes)
    }

    fn free_all(&mut self, slices: &[PoolSlice]) {
        free_slices(&mut self.pool, slices);
    }

    /// Frees the pool memory the tree let go of. Waits for the pending copies first: one may be writing to it.
    fn release(&mut self, e: &Engine, freed: &[Freed]) -> Result<(), Error> {
        if freed.is_empty() {
            return Ok(());
        }
        self.settle(e)?;
        free_nodes(&mut self.pool, &mut self.mem, freed);
        Ok(())
    }
}

/// Allocates one pool slice for each size, or none: `Ok(None)` frees what it got when one is refused.
fn alloc_slices(
    pool: &mut Pool,
    dev: &vk::Device,
    sizes: &[u64],
) -> Result<Option<Vec<PoolSlice>>, Error> {
    let mut got = Vec::with_capacity(sizes.len());
    for &n in sizes {
        match pool.alloc(dev, n) {
            Ok(Some(ps)) => got.push(ps),
            Ok(None) => {
                free_slices(pool, &got);
                return Ok(None);
            }
            Err(err) => {
                free_slices(pool, &got);
                return Err(err);
            }
        }
    }
    Ok(Some(got))
}

fn free_slices(pool: &mut Pool, slices: &[PoolSlice]) {
    for &ps in slices {
        pool.free(ps);
    }
}

/// Frees the pool memory the tree let go of. The caller has settled the pending copies: one may be writing to it.
fn free_nodes(pool: &mut Pool, mem: &mut BTreeMap<SnapId, NodeMem>, freed: &[Freed]) {
    for f in freed {
        match *f {
            Freed::State(id) => {
                if let Some(ps) = mem.get_mut(&id).and_then(|m| m.state.take()) {
                    pool.free(ps);
                }
            }
            Freed::Node(id) => {
                if let Some(m) = mem.remove(&id) {
                    free_slices(pool, &m.kv);
                    free_slices(pool, m.state.as_slice());
                }
            }
        }
    }
}

impl Drop for Snapshots {
    /// A copy a take left running may still be writing the pool, so a set dropped with one pending waits for the whole
    /// device first (the engine is out of reach here, hence the idle handle). The error is ignored: the device may be
    /// lost, and then nothing is running.
    fn drop(&mut self) {
        if self.settled < self.newest {
            let _ = self.idle.wait_idle();
        }
    }
}

/// The link's measured rates in GB/s (10⁹ bytes a second), for the pool's two candidate memory kinds: a copy from VRAM
/// to the host buffer and back on the transfer queue, and the CPU reading the host buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LinkRates {
    pub cached_d2h: f64,
    pub cached_h2d: f64,
    pub pinned_d2h: f64,
    pub pinned_h2d: f64,
    pub cached_cpu_read: f64,
    pub pinned_cpu_read: f64,
}

/// Measures the link on its own `vk::Device` (call it before `Engine::open`): for `HostCached` and `HostPinned`, the
/// best of 5 `bytes` copies VRAM to the host buffer and back on the transfer queue (`stage_async`, then `wait`), and
/// the best of 3 CPU reads of the host buffer into a `Vec<u8>` (allocated and touched before the clock starts). The
/// CPU reads what the GPU wrote, so a compute submit that waits for the transfer queue and holds `host_barrier()` makes
/// it visible first, as `park` does. `bytes` is at most `vk::MAX_ALLOCATION`.
pub fn link_rates(device: Option<[u8; 16]>, bytes: u64) -> Result<LinkRates, Error> {
    const COPIES: usize = 5;
    const READS: usize = 3;
    assert!(
        (1..=vk::MAX_ALLOCATION).contains(&bytes),
        "link_rates measures 1 byte to one allocation"
    );
    let dev = vk::Device::open(device)?;
    let vram = dev.alloc(bytes, vk::MemKind::Vram, "the link measurement's VRAM")?;
    let gbps = |t: Duration| bytes as f64 / t.as_secs_f64() / 1e9;
    let mut out = [(0.0, 0.0, 0.0); 2];
    for (kind, rates) in [vk::MemKind::HostCached, vk::MemKind::HostPinned]
        .into_iter()
        .zip(&mut out)
    {
        let host = dev.alloc(bytes, kind, "the link measurement's host buffer")?;
        // Touch the host pages first, so the copies don't time the OS committing them.
        host.write(0, &vec![0xA5u8; bytes as usize]);
        let copy = |src: vk::Slice, dst: vk::Slice| -> Result<Duration, Error> {
            let started = Instant::now();
            let v = dev.stage_async(
                &[(src, dst)],
                dev.submitted(vk::QueueKind::Compute),
                vk::StageOpts::default(),
            )?;
            dev.wait(vk::QueueKind::Transfer, v)?;
            Ok(started.elapsed())
        };
        let (mut h2d, mut d2h, mut read) = (Duration::MAX, Duration::MAX, Duration::MAX);
        for _ in 0..COPIES {
            h2d = h2d.min(copy(host.whole(), vram.whole())?);
        }
        // The last write to the host buffer is the GPU's.
        for _ in 0..COPIES {
            d2h = d2h.min(copy(vram.whole(), host.whole())?);
        }
        let mut r = dev.record()?;
        r.wait(
            vk::QueueKind::Transfer,
            dev.submitted(vk::QueueKind::Transfer),
        );
        r.host_barrier();
        dev.submit(r)?;
        let mut sink = vec![0u8; bytes as usize];
        sink.fill(1);
        for _ in 0..READS {
            let started = Instant::now();
            host.read(0, &mut sink);
            read = read.min(started.elapsed());
            // Keeps the read from being optimised away.
            std::hint::black_box(&sink);
        }
        *rates = (gbps(d2h), gbps(h2d), gbps(read));
    }
    let [(cached_d2h, cached_h2d, cached_cpu_read), (pinned_d2h, pinned_h2d, pinned_cpu_read)] =
        out;
    Ok(LinkRates {
        cached_d2h,
        cached_h2d,
        pinned_d2h,
        pinned_h2d,
        cached_cpu_read,
        pinned_cpu_read,
    })
}

/// A parked node: everything but its place in the pool.
struct ParkedNode {
    tokens: Vec<u32>,
    parent: Option<usize>,
    marks: Marks,
    has_state: bool,
    taken: u64,
    last_used: u64,
    kv: Vec<Vec<u8>>,
    state: Option<Vec<u8>>,
}

impl ParkedNode {
    fn bytes(&self) -> u64 {
        self.kv.iter().map(|v| v.len() as u64).sum::<u64>()
            + self.state.as_ref().map_or(0, |v| v.len() as u64)
    }
}

/// The snapshots of an unloaded engine, in ordinary memory: per node, parents first, its tokens, place, marks, ticks,
/// KV (a `Vec<u8>` per buffer) and state.
pub struct Parked {
    sizes: crate::Sizes,
    nodes: Vec<ParkedNode>,
}

impl Parked {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The bytes of KV and state held.
    pub fn bytes(&self) -> u64 {
        self.nodes.iter().map(ParkedNode::bytes).sum()
    }
}
