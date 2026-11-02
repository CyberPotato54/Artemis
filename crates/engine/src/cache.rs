//! The expert cache: how many slots each layer gets and what the cache serves.
//! It is filled from the routing profile at load and then adapted as the session runs. This
//! module holds its memory, in chunks of whole rows and the rows the staging area lends; `prepare_step`,
//! which moves it on between steps with the policy; and the budget watch, which retires and grows chunks.
//! Swaps and fills change what it holds; `SwapPlanner` tracks that.
use crate::{
    policy::{allowance, bank_after_verify, CachePolicy, PolicyStep},
    swap::{Entry, RowState, SwapPlanner},
    timing::CpuPhase,
    watch::{BudgetWatch, Verdict},
    weights::{slot_layout, HostCopies, DRIVER_WORK, VRAM_MARGIN},
    Engine, Error,
};
use kernels::dispatch::ENTRY_BYTES;
use qwen36::model::Model;
use std::ops::Range;
use std::time::Instant;

/// What a poisoned swap writes over its slot before the expert (`Options::poison_swaps`): f16 NaN in both halves of
/// every word, so every quantized block read mid-copy has a NaN scale and the step's logits are NaN.
pub const SWAP_POISON: u32 = 0x7E00_7E00;

/// Each main layer's slot bytes (its `SlotLayout`), for layers 0..n_layer.
pub fn slot_sizes(m: &Model) -> Vec<u64> {
    m.layers
        .iter()
        .map(|l| slot_layout(l, &m.dims).bytes() as u64)
        .collect()
}

/// The bytes of one row of the cache: one slot in every layer, 76,439,552 on the real model.
pub fn row_bytes(slot_sizes: &[u64]) -> u64 {
    slot_sizes.iter().sum()
}

/// Rows per chunk of cache memory: as many whole rows as one allocation of at most 1 GiB holds, at least one. On the
/// real model 14 (1,070,153,728 bytes). A model whose row passes 1 GiB still gets one row a
/// chunk, and the allocation of that chunk is refused (`TooLarge`), so its load fails.
pub fn chunk_rows(slot_sizes: &[u64]) -> usize {
    (vk::MAX_ALLOCATION / row_bytes(slot_sizes).max(1)).max(1) as usize
}

/// The row ranges of `n` rows in chunks of `per_chunk` (at least 1): whole chunks, then a last one holding what is left.
fn chunk_ranges(n: usize, per_chunk: usize) -> impl Iterator<Item = Range<usize>> {
    let per = per_chunk.max(1);
    (0..n)
        .step_by(per)
        .map(move |start| start..(start + per).min(n))
}

/// The cache's allocations for `n` slots per layer: chunks of `per_chunk` whole rows, the last one partial when
/// `per_chunk` doesn't divide `n`. Empty when `n` is 0.
pub fn cache_plan(slot_sizes: &[u64], n: usize, per_chunk: usize) -> Vec<u64> {
    let row = row_bytes(slot_sizes);
    chunk_ranges(n, per_chunk)
        .map(|r| r.len() as u64 * row)
        .collect()
}

/// The VRAM a cache of `n` slots per layer takes: `n` rows, whatever the chunks (each chunk is exactly its rows).
pub fn cache_bytes(slot_sizes: &[u64], n: usize) -> u64 {
    n as u64 * row_bytes(slot_sizes)
}

/// The staging area's allocations for a cache of `n` slots per layer: two buffers of
/// `n_expert − n` experts at the largest slot size, then the prefill table, (layers + 1) × n_expert entries. Empty
/// when every expert is cached.
pub fn staging_plan(slot_sizes: &[u64], n_expert: usize, n: usize) -> Vec<u64> {
    if n >= n_expert || slot_sizes.is_empty() {
        return Vec::new();
    }
    let one = (n_expert - n) as u64 * slot_sizes.iter().max().expect("a layer");
    let table = ((slot_sizes.len() + 1) * n_expert * ENTRY_BYTES) as u64;
    vec![one, one, table]
}

/// The VRAM the staging area takes for a cache of `n` slots per layer, as its planned arena packs it: 1,044,549,632
/// bytes with no cache on the real model, 485,642,240 at 137 slots.
pub fn staging_bytes(slot_sizes: &[u64], n_expert: usize, n: usize) -> u64 {
    vk::planned_bytes(&staging_plan(slot_sizes, n_expert, n))
}

/// The most slots per layer, at most `n_expert`, whose cache takes at most `room` bytes, with the staging area beside
/// it when `staging`. The cost grows with `n`: each slot adds a row to every layer (76 MB on the real model) and takes
/// two experts' room (4 MB) out of the staging area, so the search holds.
pub fn fit_slots(slot_sizes: &[u64], n_expert: usize, room: u64, staging: bool) -> usize {
    let cost = |n| {
        cache_bytes(slot_sizes, n)
            + if staging {
                staging_bytes(slot_sizes, n_expert, n)
            } else {
                0
            }
    };
    let (mut lo, mut hi) = (0, n_expert);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if cost(mid) <= room {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// The cache gets what is `free` in the VRAM budget once everything else Artemis holds is allocated, less
/// the declared session's `session` bytes, the 0.5 GiB margin, `DRIVER_WORK` and, when `staging`, the staging area.
/// `None` takes the most that fits; `Some(n)` (`--cache-slots`) is clamped to it.
pub fn choose_slots(
    free: u64,
    session: u64,
    slot_sizes: &[u64],
    n_expert: usize,
    requested: Option<usize>,
    staging: bool,
) -> usize {
    let room = free.saturating_sub(session + VRAM_MARGIN + DRIVER_WORK);
    let fits = fit_slots(slot_sizes, n_expert, room, staging);
    requested.map_or(fits, |n| n.min(fits))
}

/// Sizes the cache, and with `staging` the staging area beside it, against a VRAM budget that can shrink as this
/// process's usage grows. The RX 9070 XT's driver gives 15,416 MiB until the process uses about 11.5 GB,
/// then about 14,340, so a cache sized from the budget before it is allocated can overrun the budget the driver gives
/// once it is.
///
/// `start` is the VRAM heap just before the cache, when everything else Artemis holds in VRAM is allocated except the
/// session of `session` bytes. Each round sizes the cache with `choose_slots` against the lowest budget seen, then asks
/// `alloc(slots, staging)` for the cache, the staging area if asked, and the session's bytes together, and `heap` for
/// the budget at that usage, which must keep the margin and `DRIVER_WORK` free. A round refused by an allocation
/// (`vk::Error::OutOfBudget`), or that leaves less than the margin and `DRIVER_WORK`, retries with fewer slots against
/// the budget the driver then gave, so the loop ends: 0 slots ends it. If even 0 slots leave no room for the staging
/// area, the whole fit runs again without it (prefill then reads non-resident experts in place). Returns the slots,
/// whether the staging pass was kept (also true when every expert is cached and there is no staging area: the engine
/// uses `staging.is_some()`), and what `alloc` holds; any other error is returned as it is.
#[allow(clippy::too_many_arguments)]
pub fn fit_cache<T>(
    start: vk::HeapBudget,
    session: u64,
    slot_sizes: &[u64],
    n_expert: usize,
    requested: Option<usize>,
    staging: bool,
    mut alloc: impl FnMut(usize, bool) -> Result<T, vk::Error>,
    mut heap: impl FnMut() -> Result<vk::HeapBudget, vk::Error>,
) -> Result<(usize, bool, T), vk::Error> {
    let mut fit = |staging: bool| -> Result<(usize, T), vk::Error> {
        let size = |budget: u64| {
            choose_slots(
                budget.saturating_sub(start.usage),
                session,
                slot_sizes,
                n_expert,
                requested,
                staging,
            )
        };
        let mut budget = start.budget;
        let mut n = size(budget);
        loop {
            let lower = match alloc(n, staging) {
                Ok(held) => {
                    let h = heap()?;
                    if h.usage + VRAM_MARGIN + DRIVER_WORK <= h.budget || (n == 0 && !staging) {
                        return Ok((n, held));
                    }
                    if n == 0 {
                        // No cache at all, and the staging area alone leaves less than the margin.
                        return Err(vk::Error::OutOfBudget {
                            what: "the staging buffers".into(),
                            want: staging_bytes(slot_sizes, n_expert, 0),
                            budget: h.budget,
                            usage: h.usage,
                            host: false,
                        });
                    }
                    // `held` drops here: the round's VRAM is freed before the next one.
                    h.budget
                }
                Err(vk::Error::OutOfBudget { budget: b, .. }) if n > 0 => b,
                Err(e) => return Err(e),
            };
            budget = budget.min(lower);
            n = size(budget).min(n - 1);
        }
    };
    if staging {
        match fit(true) {
            // Even no cache leaves no room for the staging area: load without it.
            Err(vk::Error::OutOfBudget { .. }) => {}
            other => return other.map(|(n, held)| (n, true, held)),
        }
    }
    fit(false).map(|(n, held)| (n, false, held))
}

/// What the cache holds and serves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// The slots per layer the cache loaded with in chunks of its own. The lent rows are filled at load too, and
    /// `lent_slots` reports them.
    pub slots_per_layer: usize,
    /// The bytes the cache's own chunks hold when read (`slots_per_layer` is the size at load).
    pub cache_bytes: u64,
    /// The VRAM the staging area takes: its two buffers and the prefill table; 0 without one.
    pub staging_bytes: u64,
    /// Routed-expert reads over every step so far: from the cache (VRAM slots), from host memory in place, and
    /// (`staged`) from a staging buffer in a staged prefill chunk. `hits` counts the cache only,
    /// so test 5's guards keep their meaning.
    pub hits: u64,
    pub misses: u64,
    pub staged: u64,
    /// Prefill chunks that were staged.
    pub staged_steps: u64,
    pub swaps_started: u64,
    pub swaps_done: u64,
    /// Steps run: each decoded token, and each prefill chunk.
    pub steps: u64,
    /// Steps prepared while a swap's copy was in flight. It is counted at the end of
    /// `prepare_step`, before the step is recorded and submitted, so it shows a copy still outstanding when the step
    /// was prepared, not GPU overlap measured.
    pub steps_overlapping_copies: u64,
    /// The longest a step's submit has taken, from submit to finish (the command buffer is already recorded), in
    /// microseconds: the watchdog's measure. Every step's submits count: decode's one, each prefill layer's,
    /// and each output-head sub-batch's. While step timings are on, it includes reading the timestamps back.
    pub longest_submit_us: u64,
    /// Experts copied into empty slots and published (`SwapPlanner::filled`): the slots of rows the cache grew or got
    /// back, now serving.
    pub fills: u64,
    /// Budget checks that retired a chunk.
    pub shrinks: u64,
    /// Budget checks that grew a chunk: allocated it and added its rows, empty, for the policy to fill.
    pub grows: u64,
    /// Budget checks whose grow the budget refused: the allocation itself was refused (nothing allocated), or the chunk
    /// was allocated and the heap read with it left less than the margin and `GROW_SLACK`, so it was given back at once.
    /// Not an error; growing then waits (`BudgetWatch::grown`).
    pub grows_refused: u64,
    /// Budget checks that found the budget short with no chunk of the cache left to retire.
    pub budget_short: u64,
    /// Rows the staging area lends the cache between staged prefill chunks; 0 without.
    pub lent_slots: usize,
    /// The CPU miss path: decode tokens whose wait hit its 20 ms bound (each costs one bound and falls
    /// back to the HOST path for the rest of the token), and routed experts the CPU computed. A CPU miss is still a
    /// miss: neither enters `from_vram` or the hit rate.
    pub cpu_timeouts: u64,
    pub cpu_misses: u64,
}

impl Stats {
    /// The share of routed-expert reads served from VRAM: the cache's and the staging buffers', over all reads. None
    /// with no reads.
    pub fn from_vram(&self) -> Option<f64> {
        let total = self.hits + self.staged + self.misses;
        (total > 0).then(|| (self.hits + self.staged) as f64 / total as f64)
    }
}

/// Where a group of the cache's rows lives.
pub(crate) enum Mem {
    /// A chunk of its own: one allocation of whole rows, row-major (row r's slot of layer l at r × row + the layers
    /// before l). None once its rows have retired and it has been freed, or a grow the budget refused gave it back.
    Own(Option<vk::Buffer>),
    /// Slots in the staging buffers, lent to the cache between staged prefill chunks: `[row][layer]`.
    Lent(Vec<Vec<vk::Slice>>),
}

/// A run of the cache's rows and their memory.
pub(crate) struct Group {
    pub rows: Range<usize>,
    pub mem: Mem,
}

/// The cache's VRAM: rows of one slot in every main layer, each slot sized for its layer's down
/// type, in chunks of whole rows of at most 1 GiB each, and the rows the staging area lends. A chunk is allocated
/// and freed whole, so the cache can shrink and grow a chunk at a time.
pub(crate) struct Cache {
    sizes: Vec<u64>,
    /// Each layer's offset in a row.
    offsets: Vec<u64>,
    row: u64,
    per_chunk: usize,
    /// The rows the cache loaded with in chunks of its own: what `fit_cache` sized.
    n: usize,
    groups: Vec<Group>,
}

impl Cache {
    /// A cache of `n` rows in chunks of `per_chunk` rows, each its own allocation. A `per_chunk` of 0 is taken as 1, and
    /// one past `chunk_rows(&sizes)` makes a chunk the allocation refuses (`TooLarge`), so the load fails; so does a row
    /// past 1 GiB, which `chunk_rows` still gives one a chunk.
    pub fn new(
        dev: &vk::Device,
        sizes: Vec<u64>,
        n: usize,
        per_chunk: usize,
    ) -> Result<Cache, vk::Error> {
        assert!(
            sizes.iter().all(|b| b.is_multiple_of(256)),
            "every slot size must be a multiple of 256 bytes, so every slot of a chunk starts 256-aligned: {sizes:?}"
        );
        let offsets = sizes
            .iter()
            .scan(0, |at, &b| {
                let o = *at;
                *at += b;
                Some(o)
            })
            .collect();
        let mut c = Cache {
            row: row_bytes(&sizes),
            sizes,
            offsets,
            per_chunk: per_chunk.max(1),
            n,
            groups: Vec::new(),
        };
        for rows in chunk_ranges(n, c.per_chunk) {
            let bytes = rows.len() as u64 * c.row;
            let buf = dev.alloc(bytes, vk::MemKind::Vram, "the expert cache")?;
            c.groups.push(Group {
                rows,
                mem: Mem::Own(Some(buf)),
            });
        }
        Ok(c)
    }

    /// The rows the cache loaded with in chunks of its own.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The VRAM the cache's own chunks hold now.
    pub fn bytes(&self) -> u64 {
        self.groups
            .iter()
            .map(|g| match &g.mem {
                Mem::Own(Some(b)) => b.size(),
                Mem::Own(None) | Mem::Lent(_) => 0,
            })
            .sum()
    }

    /// Slot indices so far: the end of the last group.
    pub fn rows(&self) -> usize {
        self.groups.last().map_or(0, |g| g.rows.end)
    }

    /// The chunks of its own the cache holds memory for, in row order.
    pub fn own(&self) -> Vec<Range<usize>> {
        self.groups
            .iter()
            .filter(|g| matches!(g.mem, Mem::Own(Some(_))))
            .map(|g| g.rows.clone())
            .collect()
    }

    /// Slots per layer with memory behind them: chunks not freed, and the lent rows.
    pub fn held_rows(&self) -> usize {
        self.groups
            .iter()
            .filter(|g| !matches!(g.mem, Mem::Own(None)))
            .map(|g| g.rows.len())
            .sum()
    }

    /// Adds the rows the staging area lends (`places[row][layer]`) after the last group.
    pub fn lend(&mut self, places: Vec<Vec<vk::Slice>>) {
        if !places.is_empty() {
            let rows = self.rows()..self.rows() + places.len();
            self.groups.push(Group {
                rows,
                mem: Mem::Lent(places),
            });
        }
    }

    /// The lent rows, if any.
    pub fn lent(&self) -> Option<Range<usize>> {
        self.groups
            .iter()
            .find(|g| matches!(g.mem, Mem::Lent(_)))
            .map(|g| g.rows.clone())
    }

    /// The rows a grow would add, keeping at most `ceiling` held rows: the first freed chunk, else a new one after the
    /// last group. None at the ceiling.
    pub fn next_chunk(&self, ceiling: usize) -> Option<Range<usize>> {
        let room = ceiling.saturating_sub(self.held_rows());
        if let Some(g) = self.groups.iter().find(|g| matches!(g.mem, Mem::Own(None))) {
            return (g.rows.len() <= room).then(|| g.rows.clone());
        }
        let n = self.per_chunk.min(room);
        (n > 0).then(|| self.rows()..self.rows() + n)
    }

    /// Bytes of a chunk of `rows`.
    pub fn chunk_bytes(&self, rows: &Range<usize>) -> u64 {
        rows.len() as u64 * self.row
    }

    /// Allocates the chunk of `rows` (`next_chunk`'s).
    pub fn grow(&mut self, dev: &vk::Device, rows: Range<usize>) -> Result<(), vk::Error> {
        let buf = dev.alloc(
            self.chunk_bytes(&rows),
            vk::MemKind::Vram,
            "the expert cache",
        )?;
        match self.groups.iter_mut().find(|g| g.rows == rows) {
            Some(g) => {
                // Only a freed chunk is grown again: memory the group still held would be dropped under the steps
                // and copies using it.
                assert!(
                    matches!(g.mem, Mem::Own(None)),
                    "rows {rows:?} grown again while their chunk still holds memory"
                );
                g.mem = Mem::Own(Some(buf));
            }
            None => self.groups.push(Group {
                rows,
                mem: Mem::Own(Some(buf)),
            }),
        }
        Ok(())
    }

    /// Frees the chunk of `rows`: a grow the budget refused, or rows that have retired, which no step or copy uses.
    pub fn free(&mut self, rows: &Range<usize>) {
        if let Some(g) = self.groups.iter_mut().find(|g| &g.rows == rows) {
            g.mem = Mem::Own(None);
        }
    }

    /// Slot `slot` of `layer`. Panics for a slot with no memory: past the groups, or in a freed chunk.
    pub fn slot(&self, layer: usize, slot: usize) -> vk::Slice {
        let g = self
            .groups
            .iter()
            .find(|g| g.rows.contains(&slot))
            .unwrap_or_else(|| {
                panic!(
                    "slot {slot} of layer {layer}: the cache has {} rows",
                    self.rows()
                )
            });
        let r = slot - g.rows.start;
        match &g.mem {
            Mem::Own(Some(b)) => {
                b.slice(r as u64 * self.row + self.offsets[layer], self.sizes[layer])
            }
            Mem::Own(None) => panic!("slot {slot} of layer {layer}: its chunk was freed"),
            Mem::Lent(places) => places[r][layer],
        }
    }

    /// Copies each layer's `resident` experts, slot by slot, from their host copies into VRAM: one submit per layer,
    /// about 200 MB at 102 slots, well inside the watchdog's limit.
    pub fn fill(
        &self,
        dev: &vk::Device,
        host: &HostCopies,
        resident: &[Vec<usize>],
    ) -> Result<(), Error> {
        for (l, experts) in resident.iter().enumerate() {
            if experts.is_empty() {
                continue;
            }
            let mut r = dev.record()?;
            for (s, &e) in experts.iter().enumerate() {
                r.copy(host.slot_slice(l, e), self.slot(l, s));
            }
            dev.submit(r)?;
        }
        Ok(())
    }
}

/// VRAM in the shape a session of `sizes` allocates it (`session_sizes`), held while the cache is sized and freed
/// after, so the driver's budget is read at the usage the engine will have once the caller's session exists. None for
/// no session.
pub(crate) fn reserve(dev: &vk::Device, sizes: &[u64]) -> Result<Option<vk::Arena>, vk::Error> {
    if sizes.is_empty() {
        return Ok(None);
    }
    let mut a = vk::Arena::planned(
        vk::MemKind::Vram,
        sizes,
        "the sequence state (held while the cache is sized)",
    );
    for &s in sizes {
        a.alloc(dev, s)?;
    }
    Ok(Some(a))
}

/// The cache's changing state: the planner's view of the table, and the statistics.
pub(crate) struct Swaps {
    pub planner: SwapPlanner,
    pub stats: Stats,
    /// `--force-swap-every-token`'s RNG: one random swap starts before every step.
    pub forced: Option<crate::rng::Rng>,
    /// The cache policy.
    pub policy: CachePolicy,
    /// Bytes of copies prefill chunks have saved for the steps after them (`policy::allowance`).
    pub bank: u64,
    /// Budget watching between checks.
    pub watch: BudgetWatch,
    /// Tokens since the last budget check.
    pub since_check: usize,
    /// Slot bytes of the copies started since the last `next_step`: the policy's swaps and fills, the forced swap and
    /// `request_swap`'s swaps.
    pub copies_since: u64,
    /// What `copies_since` held when the last `next_step` ran: the copies started for the last step prepared, which
    /// the next poll sends, since the engine waits for each step. A poll's copy submit carries at most this and
    /// `copies_since` (`prepare_step` asserts it).
    pub copies_last: u64,
}

impl Engine {
    /// Expert slots per layer in the cache's own chunks at load: what the budget fitted, or `--cache-slots` clamped to
    /// it. The rows the staging area lends come on top (`Stats::lent_slots`), and chunks may retire and grow
    /// (`live_slots`).
    pub fn cache_slots(&self) -> usize {
        self.cache.borrow().n()
    }

    /// Slots per layer the cache can hold experts in now: its live rows.
    pub fn live_slots(&self) -> usize {
        self.swaps.borrow().planner.live_rows()
    }

    /// Tests only: takes `bytes` off every budget reading from now on, as if another process had taken them, so a test
    /// can make the cache shrink and grow.
    pub fn set_budget_cut(&self, bytes: u64) {
        self.budget_cut.set(bytes);
    }

    /// Tests only: takes `bytes` off the budget reading a grow takes with its chunk allocated (on top of
    /// `set_budget_cut`'s), from now on, as the RX 9070 XT's driver lowers the budget once usage passes 11.5 GB, so a
    /// test can make the budget refuse a grow it had room for. No other reading changes.
    pub fn set_grow_cut(&self, bytes: u64) {
        self.grow_cut.set(bytes);
    }

    /// Whether the cache policy adapts: off with `--cache-policy static` or a forced swap a step.
    pub fn policy_adapts(&self) -> bool {
        self.swaps.borrow().policy.adapts()
    }

    /// Submits a step's recorded command buffer and waits for it to finish, timing that for
    /// `Stats::longest_submit_us`, and, while timings are on, keeping each command's GPU time (reading the timestamps
    /// back counts in that time). Every step's submit goes through here, decode's and prefill's, so none is missed; a
    /// failed submit is timed too.
    pub(crate) fn submit_step(&self, r: vk::Recorder<'_>) -> Result<(), vk::Error> {
        self.dev.set_trace_label(self.trace_label.get());
        let t0 = std::time::Instant::now();
        let out = if self.timing_on() {
            self.dev.submit_profiled(r).map(|t| {
                if let Some((start_ns, end_ns)) = self.dev.last_span().filter(|_| !t.is_empty()) {
                    self.add_span(crate::timing::Span {
                        kind: crate::timing::SpanKind::Compute,
                        tag: t.last().map_or(crate::timing::UNTAGGED, |x| x.tag),
                        start_ns,
                        end_ns,
                    });
                }
                self.add_gpu(t)
            })
        } else {
            self.dev.submit(r)
        };
        let us = t0.elapsed().as_micros() as u64;
        self.add_time(crate::timing::CpuPhase::Submit, us);
        let mut s = self.swaps.borrow_mut();
        s.stats.longest_submit_us = s.stats.longest_submit_us.max(us);
        out
    }

    /// What the cache holds and has served so far.
    pub fn stats(&self) -> Stats {
        let s = self.swaps.borrow();
        Stats {
            swaps_started: s.planner.started(),
            swaps_done: s.planner.done(),
            fills: s.planner.filled(),
            cache_bytes: self.cache.borrow().bytes(),
            ..s.stats
        }
    }

    /// The VRAM heap's budget and this process's usage, as the driver reports them now. For tests and diagnostics.
    pub fn vram_heap(&self) -> Result<vk::HeapBudget, Error> {
        Ok(self.dev.heap_budget(vk::MemKind::Vram)?)
    }

    /// The entry the planner holds for `expert` in `layer`: what the table should hold. For tests and diagnostics.
    pub fn planner_entry(&self, layer: usize, expert: usize) -> Option<Entry> {
        self.swaps.borrow().planner.get_entry(layer, expert)
    }

    /// The device address of `expert`'s host slot in `layer`: what a HOST entry holds. For tests and diagnostics.
    pub fn host_slot_addr(&self, layer: usize, expert: usize) -> u64 {
        self.host.slot_addr(layer, expert)
    }

    /// The device address of slot `slot` in `layer`'s cache: what a VRAM entry holds. For tests and diagnostics. Refuses
    /// a slot past the cache's rows (`NoSuchSlot`) and one in a dead row (`DeadSlot`), whose chunk may be freed.
    pub fn cache_slot_addr(&self, layer: usize, slot: usize) -> Result<u64, Error> {
        Ok(self.held_slot(layer, slot)?.addr)
    }

    /// The last decoded token's routed expert ids (a prefill step doesn't write them), `[n_layer][n_expert_used]`
    /// flattened. For tests and diagnostics.
    pub fn last_routes(&self) -> Vec<usize> {
        let d = &self.dims;
        let ids = self.routes.read_f32(0, d.n_layer * d.n_expert_used);
        ids.iter().map(|&v| v as usize).collect()
    }

    /// The last prefill chunk's or verify step's routed expert ids, `[n_layer][tokens][n_expert_used]` flattened (a
    /// verify step's rows are its tokens). Empty before any chunk and after a failed one, whose layers overwrote only
    /// part of the log. For tests and diagnostics.
    pub fn last_prefill_routes(&self) -> Vec<usize> {
        let d = &self.dims;
        let n = d.n_layer * self.prefill.last.get() * d.n_expert_used;
        let ids = self.prefill.routes.read_f32(0, n);
        ids.iter().map(|&v| v as usize).collect()
    }

    /// Their weights, laid out the same. For tests and diagnostics.
    pub fn last_prefill_route_weights(&self) -> Vec<f32> {
        let d = &self.dims;
        let n = d.n_layer * self.prefill.last.get() * d.n_expert_used;
        self.prefill.route_weights.read_f32(0, n)
    }

    /// The bytes in slot `slot` of `layer`'s cache. For tests and diagnostics. It first waits for every copy submitted
    /// on the transfer queue, so a slot whose swap copy is still in flight reads as the copy leaves it, never mid-copy.
    /// Refuses a slot past the cache's rows (`NoSuchSlot`) and one in a dead row (`DeadSlot`).
    pub fn cache_slot_bytes(&self, layer: usize, slot: usize) -> Result<Vec<u8>, Error> {
        let at = self.held_slot(layer, slot)?;
        // The host wait raises what the download's cross wait covers, which makes the copy's writes visible to it.
        self.dev.wait(
            vk::QueueKind::Transfer,
            self.dev.submitted(vk::QueueKind::Transfer),
        )?;
        Ok(self.dev.download(at)?)
    }

    /// Slot `slot` of `layer`'s cache, refused where `Cache::slot` would panic or a dead row's chunk may be freed: a
    /// slot past the cache's rows (`NoSuchSlot`, with the slots the cache holds now) or in a dead row (`DeadSlot`). A
    /// row that isn't dead always has memory: a chunk is freed only once its rows are all dead, and rows are added only
    /// once their memory is allocated.
    fn held_slot(&self, layer: usize, slot: usize) -> Result<vk::Slice, Error> {
        let cache = self.cache.borrow();
        if layer >= self.dims.n_layer || slot >= cache.rows() {
            return Err(Error::NoSuchSlot {
                layer,
                slot,
                slots: cache.held_rows(),
            });
        }
        if self.swaps.borrow().planner.row_state(slot) == RowState::Dead {
            return Err(Error::DeadSlot { layer, slot });
        }
        Ok(cache.slot(layer, slot))
    }

    /// Adds one decode step's CPU-path counts (`CpuPath::after_step`) to the run's.
    pub(crate) fn count_cpu(&self, timeouts: u64, computed: u64) {
        let mut s = self.swaps.borrow_mut();
        s.stats.cpu_timeouts += timeouts;
        s.stats.cpu_misses += computed;
    }

    /// Counts a finished step and its routed experts (the routing log, `ids`: each layer's `tokens` × n_expert_used ids
    /// in turn), by the table that step ran with, the one the planner last published: hits, served from the cache;
    /// staged, served from a staging buffer (`staged`, per `layer × n_expert + expert`, in a staged chunk); misses,
    /// served from host memory. Entries changed behind the planner's back (`set_slot_entry`, `stage_slot_entry`,
    /// for tests) aren't seen.
    pub(crate) fn count_routes(&self, ids: &[f32], tokens: usize, staged: Option<&[bool]>) {
        let (per_layer, n) = (tokens * self.dims.n_expert_used, self.dims.n_expert);
        let mut s = self.swaps.borrow_mut();
        let (mut hits, mut from_staging, mut misses) = (0, 0, 0);
        for (i, &id) in ids.iter().enumerate() {
            let (l, x) = (i / per_layer, id as usize);
            // An id outside the table is the status check's to report (`BadSlotEntry`); it isn't counted.
            match s.planner.get_entry(l, x) {
                Some(Entry::Vram(_)) => hits += 1,
                Some(Entry::Host) if staged.is_some_and(|st| st.get(l * n + x) == Some(&true)) => {
                    from_staging += 1
                }
                Some(Entry::Host) => misses += 1,
                None => {}
            }
        }
        s.stats.hits += hits;
        s.stats.staged += from_staging;
        s.stats.misses += misses;
        s.stats.steps += 1;
        if staged.is_some() {
            s.stats.staged_steps += 1;
        }
    }
}

impl crate::Engine {
    /// Starts swapping expert `inn` into the cache slot of expert `out` in `layer`. Nothing changes until
    /// the next step: from then `out` is read from host memory, and `inn` from VRAM once its copy has landed,
    /// two steps later. Refuses ids out of range, an `out` that isn't cached, an `inn` that is, and an
    /// expert already in a swap.
    pub fn request_swap(&self, layer: usize, out: usize, inn: usize) -> Result<(), crate::Error> {
        let mut sw = self.swaps.borrow_mut();
        sw.planner.request(layer, out, inn)?;
        sw.copies_since += self.slot_bytes[layer];
        Ok(())
    }

    /// Moves the cache on before a step of `kind` and `tokens` records: a staged chunk
    /// takes the lent rows back for staging; copies into slots no running step can read go out on the transfer queue;
    /// chunks whose rows have retired are freed; any other step returns the lent rows to the cache once they are dead;
    /// every `budget_every` tokens the budget is checked; a forced swap starts; the policy picks its swaps and fills; and
    /// the step's table changes are staged for `apply_slot_updates`. The lent rows return after the poll, which ends
    /// their last moves (before it they come back a step late), and dead chunks are freed before the budget check, which
    /// would otherwise read a heap still holding them and retire another chunk.
    pub(crate) fn prepare_step(&self, kind: PolicyStep, tokens: usize) -> Result<(), crate::Error> {
        let compute_done = self.dev.completed(vk::QueueKind::Compute)?;
        let transfer_done = self.dev.completed(vk::QueueKind::Transfer)?;
        let changes = {
            let mut guard = self.swaps.borrow_mut();
            let sw = &mut *guard;
            let staged = kind == PolicyStep::Staged;
            if staged {
                self.take_lent_rows(sw)?;
            }
            {
                // Every copy the drains allow goes in one transfer submit: a burst of them costs one submit, not one
                // each. Its leading barrier lands it after every earlier copy on the transfer queue. The cache's borrow
                // ends with the block, before `free_retired` changes it.
                //
                // The submit carries one step's copies: the policy's picks (at most `RESHAPE_BYTES`, `allowance`), the
                // forced swap and `request_swap`'s, all started for the last step prepared. This rests on the engine
                // waiting for each step (`submit_step`): by this poll the compute queue has passed every value those
                // copies drain against. A pipelined engine must bound the poll's batch itself; the assert checks it.
                // A refused submit returns before the picks below, so its retry carries the same copies or fewer.
                let (dev, host, cache) = (&self.dev, &self.host, self.cache.borrow());
                let (last, since) = (sw.copies_last, sw.copies_since);
                sw.planner.poll(compute_done, transfer_done, |orders| {
                    let regions: Vec<(vk::Slice, vk::Slice)> = orders
                        .iter()
                        .map(|o| {
                            (
                                host.slot_slice(o.layer, o.expert),
                                cache.slot(o.layer, o.slot),
                            )
                        })
                        .collect();
                    let bytes: u64 = regions.iter().map(|(_, slot)| slot.size).sum();
                    assert!(
                        bytes <= last + since,
                        "a poll's copy submit of {bytes} bytes passes the one-step bound of {} bytes: the copies \
                         started for the last step prepared ({last}: the policy's picks, at most RESHAPE_BYTES, its \
                         forced swap and request_swap's) and since ({since}). The bound rests on the engine waiting \
                         for each step (submit_step)",
                        last + since
                    );
                    dev.stage_async(
                        &regions,
                        compute_done,
                        vk::StageOpts {
                            poison: self.poison,
                            ..vk::StageOpts::default()
                        },
                    )
                })?;
            }
            // Before the budget check: after it, the check reads a heap still holding a dead chunk and retires another.
            self.free_retired(sw);
            // After the poll, which ends the lent rows' last moves: before it they are still retiring and come back a
            // step late.
            if !staged {
                self.return_lent_rows(sw)?;
            }
            sw.since_check += tokens;
            if sw.since_check >= self.budget_every {
                sw.since_check = 0;
                self.watch_budget(sw)?;
            }
            if let Some(rng) = sw.forced.as_mut() {
                if let Some((layer, out, inn)) = sw.planner.random_swap(rng) {
                    sw.planner.request(layer, out, inn)?;
                    sw.copies_since += self.slot_bytes[layer];
                }
            }
            self.pick(sw, kind, tokens)?;
            // The next poll sends the copies started for this step (its assert's bound).
            sw.copies_last = std::mem::take(&mut sw.copies_since);
            sw.planner
                .next_step(self.dev.submitted(vk::QueueKind::Compute))
        };
        let cache = self.cache.borrow();
        for c in changes {
            let (addr, loc) = match c.to {
                crate::swap::Entry::Host => (
                    self.host.slot_addr(c.layer, c.expert),
                    kernels::dispatch::WHERE_HOST,
                ),
                crate::swap::Entry::Vram(slot) => (
                    cache.slot(c.layer, slot).addr,
                    kernels::dispatch::WHERE_VRAM,
                ),
            };
            // The planner's ids are in range, so `stage_slot_entry` can't fail here; if it could, `next_step` would
            // already have counted the rest of these changes as published.
            debug_assert!(
                c.layer < self.dims.n_layer && c.expert < self.dims.n_expert,
                "the planner made a change outside the table: {c:?}"
            );
            self.stage_slot_entry(c.layer, c.expert, addr, loc)?;
        }
        // A step may now be prepared while a copy is still landing; count it (`Stats::steps_overlapping_copies`).
        if self.dev.submitted(vk::QueueKind::Transfer)
            > self.dev.completed(vk::QueueKind::Transfer)?
        {
            self.swaps.borrow_mut().stats.steps_overlapping_copies += 1;
        }
        Ok(())
    }

    /// Frees each chunk of the cache's own whose rows are all dead (retired and drained): no step reads them and no copy
    /// writes them (`SwapPlanner::retire`, `RowState::Dead`).
    fn free_retired(&self, sw: &Swaps) {
        let mut cache = self.cache.borrow_mut();
        for rows in cache.own() {
            if rows
                .clone()
                .all(|r| sw.planner.row_state(r) == RowState::Dead)
            {
                cache.free(&rows);
            }
        }
    }

    /// The lent rows go back to the staging area for a staged chunk, before the step's poll, so
    /// no copy goes into them that the retire would only cancel. Retired, their experts are HOST from this step on, and
    /// the chunk's staging copies wait for the steps before it and land after every copy already sent (`stage_async`'s
    /// `after` and leading barrier), so they may overwrite the rows at once.
    fn take_lent_rows(&self, sw: &mut Swaps) -> Result<(), Error> {
        let Some(lent) = self.cache.borrow().lent() else {
            return Ok(());
        };
        if lent
            .clone()
            .all(|r| sw.planner.row_state(r) == RowState::Live)
        {
            sw.planner.retire(lent)?;
        }
        Ok(())
    }

    /// For any other step the lent rows return to the cache empty once they are dead, after the poll that may have
    /// ended their last moves, for the policy to fill. The fills wait for the steps that staged into them (`add`'s
    /// `ready`), and their copies land after any staging copy a failed chunk left in flight, by the same barrier.
    fn return_lent_rows(&self, sw: &mut Swaps) -> Result<(), Error> {
        let Some(lent) = self.cache.borrow().lent() else {
            return Ok(());
        };
        if lent
            .clone()
            .all(|r| sw.planner.row_state(r) == RowState::Dead)
        {
            sw.planner
                .add(lent, self.dev.submitted(vk::QueueKind::Compute))?;
        }
        Ok(())
    }

    /// The VRAM heap now, less `set_budget_cut`'s bytes.
    fn heap_now(&self) -> Result<vk::HeapBudget, Error> {
        let h = self.dev.heap_budget(vk::MemKind::Vram)?;
        Ok(vk::HeapBudget {
            budget: h.budget.saturating_sub(self.budget_cut.get()),
            ..h
        })
    }

    /// Checks the budget now: `Engine::session` and `Session::reset` call it.
    /// The server's worker calls it between requests, as `Engine::session` does.
    pub fn check_budget(&self) -> Result<(), Error> {
        let mut guard = self.swaps.borrow_mut();
        let sw = &mut *guard;
        sw.since_check = 0;
        self.free_retired(sw);
        self.watch_budget(sw)
    }

    /// One budget check: retire the cache's last chunk when the margin is gone, grow one after two
    /// checks in a row with room for it, or nothing. A grow the budget refuses, at the allocation or once allocated, is
    /// given back at once and isn't an error.
    fn watch_budget(&self, sw: &mut Swaps) -> Result<(), Error> {
        let h = self.heap_now()?;
        let (own, next) = {
            let c = self.cache.borrow();
            let next = c.next_chunk(self.ceiling).map(|r| (c.chunk_bytes(&r), r));
            (c.own(), next)
        };
        let state = |r: usize| sw.planner.row_state(r);
        let last_live = own
            .iter()
            .rev()
            .find(|g| (*g).clone().all(|r| state(r) == RowState::Live))
            .cloned();
        let retiring = own
            .iter()
            .any(|g| g.clone().any(|r| state(r) == RowState::Retiring));
        match sw
            .watch
            .check(h, next.as_ref().map(|n| n.0), last_live.is_some(), retiring)
        {
            Verdict::Hold => {}
            Verdict::Shrink => {
                sw.planner
                    .retire(last_live.expect("a live chunk to retire"))?;
                sw.stats.shrinks += 1;
            }
            Verdict::Short => sw.stats.budget_short += 1,
            Verdict::Grow => {
                let (_, rows) = next.expect("a chunk to grow");
                let allocated = match self.cache.borrow_mut().grow(&self.dev, rows.clone()) {
                    Ok(()) => true,
                    Err(vk::Error::OutOfBudget { .. }) => false,
                    Err(e) => return Err(e.into()),
                };
                let after = if allocated {
                    let h = self.heap_now()?;
                    Some(vk::HeapBudget {
                        budget: h.budget.saturating_sub(self.grow_cut.get()),
                        ..h
                    })
                } else {
                    None
                };
                if sw.watch.grown(after) {
                    sw.planner
                        .add(rows, self.dev.submitted(vk::QueueKind::Compute))?;
                    sw.stats.grows += 1;
                } else {
                    if allocated {
                        self.cache.borrow_mut().free(&rows);
                    }
                    sw.stats.grows_refused += 1;
                }
            }
        }
        Ok(())
    }

    /// A verify step has kept `kept` tokens: its first token's share was spent when it was prepared
    /// (`PolicyStep::Verify` prepares like a one-token decode step), and each further token adds a cap's worth to the
    /// bank (at most what a bank holds) and one to the budget watch's count. A rejected draft earns nothing. The next
    /// `prepare_step` spends the bank and runs the budget check, as it does after a decode step.
    pub(crate) fn settle_verify(&self, kept: usize) {
        let mut sw = self.swaps.borrow_mut();
        let cap = sw.policy.params().cap;
        sw.bank = bank_after_verify(sw.bank, kept, cap);
        sw.since_check += kept.saturating_sub(1);
    }

    /// The policy's swaps and fills for the step about to run, within what the policy's upload
    /// cap allows it (`policy::allowance`): a prefill chunk's share goes to the step after it, which picks with the
    /// chunk's routes counted, and no step spends more than `RESHAPE_BYTES` (`Engine::open` refuses a larger cap). Each
    /// copy started counts in `Swaps::copies_since`.
    fn pick(&self, sw: &mut Swaps, kind: PolicyStep, tokens: usize) -> Result<(), Error> {
        let (spend, bank) = allowance(kind, tokens, sw.bank, sw.policy.params().cap);
        sw.bank = bank;
        if spend == 0 {
            return Ok(());
        }
        for p in sw.policy.choose(&sw.planner, spend, &self.slot_bytes) {
            match p.out {
                Some(out) => sw.planner.request(p.layer, out, p.inn)?,
                None => {
                    sw.planner.fill(p.layer, p.inn)?;
                }
            }
            sw.copies_since += self.slot_bytes[p.layer];
        }
        Ok(())
    }

    /// The policy counts a finished step's routes, timed as preparation: the policy's share of
    /// the step.
    pub(crate) fn observe_routes(&self, ids: &[f32], tokens: usize) {
        let t0 = Instant::now();
        self.swaps.borrow_mut().policy.observe(ids, tokens);
        self.time(CpuPhase::Prepare, t0);
    }
}

/// The RNG for `--force-swap-every-token`, or None without a seed. A forced swap needs an expert in the
/// cache to send out and one outside it to bring in, so a cache of none or of every expert is refused.
pub(crate) fn forced_swaps(
    seed: Option<u64>,
    slots: usize,
    n_expert: usize,
) -> Result<Option<crate::rng::Rng>, crate::Error> {
    let Some(seed) = seed else {
        return Ok(None);
    };
    if slots == 0 || slots >= n_expert {
        return Err(crate::Error::NoRoomToSwap { slots, n_expert });
    }
    Ok(Some(crate::rng::Rng::new(seed)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::session_sizes;
    use kernels::dispatch::SlotLayout;
    use qwen36::{dims::Dims, quant::WType};
    use std::cell::Cell;

    #[test]
    fn the_share_from_vram_counts_staged_reads_with_the_cache_hits() {
        let st = Stats {
            hits: 600,
            staged: 300,
            misses: 100,
            ..Stats::default()
        };
        assert_eq!(st.from_vram(), Some(0.9));
        let host_only = Stats {
            misses: 5,
            ..Stats::default()
        };
        assert_eq!(host_only.from_vram(), Some(0.0));
        assert_eq!(Stats::default().from_vram(), None, "no reads, no share");
    }

    /// The real model's slot sizes: Q4_K gate and up, Q5_K down, except Q6_K down in layers 34, 38 and 39.
    fn real_sizes() -> Vec<u64> {
        let slot = |down| {
            SlotLayout {
                gate: WType::Q4K,
                up: WType::Q4K,
                down,
                n_embd: 2048,
                n_ff: 512,
            }
            .bytes() as u64
        };
        (0..40)
            .map(|l| {
                slot(if [34, 38, 39].contains(&l) {
                    WType::Q6K
                } else {
                    WType::Q5K
                })
            })
            .collect()
    }

    /// One slot in every layer of the real model: 37 × 1,900,544 + 3 × 2,039,808 bytes.
    const ROW: u64 = 76_439_552;
    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    #[test]
    fn a_forced_swap_needs_experts_in_and_out_of_the_cache() {
        assert!(
            forced_swaps(None, 0, 256).unwrap().is_none(),
            "no seed, no forced swaps"
        );
        assert!(forced_swaps(Some(1), 102, 256).unwrap().is_some());
        for slots in [0, 256] {
            match forced_swaps(Some(1), slots, 256) {
                Err(crate::Error::NoRoomToSwap {
                    slots: s,
                    n_expert: 256,
                }) => assert_eq!(s, slots),
                other => panic!(
                    "{slots} slots: expected NoRoomToSwap, got {:?}",
                    other.err()
                ),
            }
        }
    }

    #[test]
    fn a_slot_row_of_the_real_model_is_76_439_552_bytes() {
        let s = real_sizes();
        assert_eq!((s[0], s[34]), (1_900_544, 2_039_808));
        assert_eq!(s.iter().sum::<u64>(), ROW);
        // Slots are whole multiples of 256 bytes, so every slot of a chunk starts 256-aligned.
        assert!(s.iter().all(|b| b.is_multiple_of(256)));
        for n in [1, 26, 102, 112, 256] {
            assert_eq!(cache_bytes(&s, n), n as u64 * ROW, "{n} slots");
        }
        assert_eq!(cache_bytes(&s, 0), 0);
    }

    #[test]
    fn the_cache_comes_in_chunks_of_14_rows() {
        // 14 rows are 1,070,153,728 bytes, under the 1 GiB allocation limit; 15 would pass it.
        let s = real_sizes();
        assert_eq!(chunk_rows(&s), 14);
        const { assert!(14 * ROW < GIB && 15 * ROW > GIB) };
        // 124 slots: eight chunks of 14 rows and one of 12.
        let plan = cache_plan(&s, 124, 14);
        assert_eq!(plan.len(), 9);
        assert!(plan[..8].iter().all(|&b| b == 14 * ROW));
        assert_eq!(plan[8], 12 * ROW);
        assert_eq!(plan.iter().sum::<u64>(), cache_bytes(&s, 124));
        assert_eq!(cache_plan(&s, 28, 14), [14 * ROW, 14 * ROW]);
        assert!(cache_plan(&s, 0, 14).is_empty());
        assert!(cache_plan(&s, 256, 14).iter().all(|&b| b < GIB));
        // Smaller chunks, as the tiny model's tests ask for.
        assert_eq!(cache_plan(&s, 5, 2), [2 * ROW, 2 * ROW, ROW]);
        // A model whose row is larger than 1 GiB still gets one row a chunk (its allocation is then refused, so such a model doesn't load).
        assert_eq!(chunk_rows(&[GIB + 1]), 1);
    }

    #[test]
    fn fit_slots_takes_the_most_that_fits() {
        let s = real_sizes();
        // 8 GiB holds 112 rows (8,561,229,824 bytes); a 113th would need 8,637,669,376.
        assert_eq!(fit_slots(&s, 256, 8 * GIB, false), 112);
        assert_eq!(fit_slots(&s, 256, 112 * ROW, false), 112);
        assert_eq!(fit_slots(&s, 256, 112 * ROW - 1, false), 111);
        assert_eq!(fit_slots(&s, 256, ROW - 1, false), 0);
        assert_eq!(fit_slots(&s, 256, 0, false), 0);
        assert_eq!(fit_slots(&s, 256, u64::MAX, false), 256);
    }

    #[test]
    fn choose_slots_clamps_and_saturates() {
        // 14 GiB of budget with 2,000 MiB of weights allocated; a 65,536-position session (1,412,268,032 bytes); the
        // 0.5 GiB margin and the driver's 64 MiB. That leaves 10,918,985,728 bytes: 142 rows (10,854,416,384), not 143
        // (10,930,855,936).
        let s = real_sizes();
        let session = 1_412_268_032;
        let free = 14 * GIB - 2_000 * MIB;
        assert_eq!(choose_slots(free, session, &s, 256, None, false), 142);
        assert_eq!(choose_slots(free, session, &s, 256, Some(102), false), 102);
        assert_eq!(choose_slots(free, session, &s, 256, Some(500), false), 142);
        assert_eq!(choose_slots(free, session, &s, 256, Some(0), false), 0);
        // Less free than the session and the margin leaves no room, never an underflow.
        assert_eq!(choose_slots(GIB, session, &s, 256, None, false), 0);
        assert_eq!(choose_slots(GIB, session, &s, 256, Some(26), false), 0);
    }

    /// A VRAM heap like the RX 9070 XT's, measured on 2026-10-09: its budget is 15,416 MiB until this process uses
    /// 11,500 MiB, then 14,339 MiB, and it refuses any allocation that would take the usage past the budget it gives
    /// at that moment. `start` is the usage before the cache: the weights' arena (1,922 MiB) and the 11 MiB the
    /// driver takes with the first allocation.
    struct FakeHeap {
        start: u64,
        usage: Cell<u64>,
        rounds: Cell<usize>,
    }

    /// One allocation of the fake heap, given back when dropped.
    struct Held<'a>(&'a FakeHeap, u64);

    impl Drop for Held<'_> {
        fn drop(&mut self) {
            self.0.usage.set(self.0.usage.get() - self.1);
        }
    }

    impl FakeHeap {
        const START: u64 = 1_933 * MIB;

        fn new() -> FakeHeap {
            FakeHeap::at(FakeHeap::START)
        }

        /// A heap already holding `start` bytes.
        fn at(start: u64) -> FakeHeap {
            FakeHeap {
                start,
                usage: Cell::new(start),
                rounds: Cell::new(0),
            }
        }

        fn heap(&self) -> vk::HeapBudget {
            let usage = self.usage.get();
            vk::HeapBudget {
                size: 16_304 * MIB,
                budget: if usage > 11_500 * MIB {
                    14_339 * MIB
                } else {
                    15_416 * MIB
                },
                usage,
            }
        }

        /// The cache of `n` slots per layer and a session of `ctx` positions (none at 0), allocation by allocation, as
        /// `Engine::open` makes them.
        fn alloc(&self, n: usize, staged: bool, ctx: usize) -> Result<Vec<Held<'_>>, vk::Error> {
            self.rounds.set(self.rounds.get() + 1);
            let session = if ctx > 0 {
                session_sizes(&Dims::QWEN36, ctx)
            } else {
                Vec::new()
            };
            let mut held = Vec::new();
            for size in cache_plan(&real_sizes(), n, chunk_rows(&real_sizes()))
                .into_iter()
                .chain(if staged {
                    staging_plan(&real_sizes(), 256, n)
                } else {
                    Vec::new()
                })
                .chain(session)
            {
                let h = self.heap();
                if h.usage + size > h.budget {
                    return Err(vk::Error::OutOfBudget {
                        what: "a fake allocation".into(),
                        want: size,
                        budget: h.budget,
                        usage: h.usage,
                        host: false,
                    });
                }
                self.usage.set(h.usage + size);
                held.push(Held(self, size));
            }
            Ok(held)
        }

        /// `fit_cache` on this heap: (slots, staging kept, rounds), checking that the usage it settles at keeps the
        /// margin and that every round's VRAM is freed.
        fn fit(&self, ctx: usize, requested: Option<usize>, staging: bool) -> (usize, bool, usize) {
            let session = if ctx > 0 {
                crate::session_bytes(&Dims::QWEN36, ctx)
            } else {
                0
            };
            let (n, staged, held) = fit_cache(
                self.heap(),
                session,
                &real_sizes(),
                256,
                requested,
                staging,
                |n, staged| self.alloc(n, staged, ctx),
                || Ok(self.heap()),
            )
            .unwrap();
            let h = self.heap();
            assert!(
                h.usage + VRAM_MARGIN + DRIVER_WORK <= h.budget,
                "{n} slots: {h:?}"
            );
            drop(held);
            assert_eq!(self.usage.get(), self.start, "every round's VRAM is freed");
            (n, staged, self.rounds.get())
        }
    }

    #[test]
    fn the_cache_fits_the_budget_the_driver_gives_once_it_is_full() {
        // ctx 65,536: sized against 15,416 MiB, 158 slots fit, but with the session's 1,347 MiB the usage passes
        // 11,500 MiB and the budget falls to 14,339 MiB, which refuses the session. Against 14,339, with the margin and
        // the driver's 64 MiB reserved: 143 slots (10,424 MiB; 144 would need 10,497 MiB of the 10,483 left).
        assert_eq!(FakeHeap::new().fit(65_536, None, false), (143, false, 2));
        // ctx 4,096: 175 slots are refused partway, once the budget has fallen; 160 fit beside the 143 MiB session
        // (161 would need 11,737 MiB of the 11,687 left).
        assert_eq!(FakeHeap::new().fit(4_096, None, false), (160, false, 2));
        // Asking for more is clamped to the same.
        assert_eq!(
            FakeHeap::new().fit(4_096, Some(256), false),
            (160, false, 2)
        );
    }

    #[test]
    fn a_cache_that_stays_under_the_drop_takes_one_round() {
        // 102 slots (7,436 MiB) and a 65,536-position session stay under 11,500 MiB, where the budget doesn't move.
        assert_eq!(
            FakeHeap::new().fit(65_536, Some(102), false),
            (102, false, 1)
        );
        assert_eq!(FakeHeap::new().fit(4_096, Some(0), false), (0, false, 1));
        // No session at all (ctx 0): the cache alone is sized, against the lower budget once it passes the drop.
        assert_eq!(FakeHeap::new().fit(0, None, false), (162, false, 2));
    }

    #[test]
    fn staging_comes_out_of_the_cache() {
        // Two buffers of (256 − n) experts at 2,039,808 bytes and the 167,936-byte prefill table
        // come out of the cache. At ctx 65,536: 137 slots, not 143 (137 rows and 2 × 119 staged experts take 10,450 MiB
        // of the 10,483 left at the lower budget; 138 would take 10,519). At ctx 4,096: 154, not 160 (11,623 MiB of the
        // 11,687 left; 155 would take 11,692). A replay of the fit on 2026-10-10 gave the same.
        assert_eq!(FakeHeap::new().fit(65_536, None, true), (137, true, 2));
        assert_eq!(FakeHeap::new().fit(4_096, None, true), (154, true, 2));
        assert_eq!(FakeHeap::new().fit(4_096, Some(256), true), (154, true, 2));
        // A request that fits keeps its slots; the staging buffers take what is left over.
        assert_eq!(FakeHeap::new().fit(65_536, Some(102), true), (102, true, 1));
        assert_eq!(FakeHeap::new().fit(4_096, Some(0), true), (0, true, 1));
        let s = real_sizes();
        assert_eq!(staging_bytes(&s, 256, 137), 2 * 119 * 2_039_808 + 167_936);
        assert_eq!(staging_bytes(&s, 256, 0), 1_044_549_632);
        assert_eq!(staging_bytes(&s, 256, 256), 0);
        assert!(staging_plan(&s, 256, 0).iter().all(|&b| b < GIB));
        // The cost of a slot with staging still grows with n, so the binary search holds.
        assert!(
            cache_bytes(&s, 101) + staging_bytes(&s, 256, 101)
                < cache_bytes(&s, 102) + staging_bytes(&s, 256, 102)
        );
        assert_eq!(
            fit_slots(
                &s,
                256,
                cache_bytes(&s, 137) + staging_bytes(&s, 256, 137),
                true
            ),
            137
        );
    }

    #[test]
    fn a_budget_too_small_for_staging_loads_without_it() {
        // With 13,000 MiB already in use, even no cache leaves less than the staging area's 996 MiB
        // beside a 4,096-position session and the margin, so the engine loads without staging, and the cache gets the 8
        // slots that fit (a staging round whose allocations succeed and whose margin check rejects it, then one without).
        assert_eq!(
            FakeHeap::at(13_000 * MIB).fit(4_096, None, true),
            (8, false, 2)
        );
        assert_eq!(
            FakeHeap::at(13_000 * MIB).fit(4_096, None, false),
            (8, false, 1)
        );
        // The other way out: at 13,500 MiB the budget is 14,339 MiB, and with 0 slots the first staging buffer (498 MiB)
        // fits (13,998) but the second (14,496) is refused by the allocation itself, which with n = 0 ends the staging
        // pass with `OutOfBudget` (the retry arm only takes n > 0). The result is what the same heap gives unstaged.
        let (n, staged, _) = FakeHeap::at(13_500 * MIB).fit(4_096, None, true);
        let (m, _, _) = FakeHeap::at(13_500 * MIB).fit(4_096, None, false);
        assert!(!staged);
        assert_eq!(n, m);
    }

    #[test]
    fn a_budget_that_keeps_shrinking_still_ends() {
        // A driver that answers every round with a budget at its usage: no cache ever keeps the margin, so each round
        // gives up slots until none are left, and 0 slots ends the loop.
        let rounds = Cell::new(0);
        let usage = Cell::new(FakeHeap::START);
        let (n, _, ()) = fit_cache(
            vk::HeapBudget {
                size: 16_304 * MIB,
                budget: 15_416 * MIB,
                usage: FakeHeap::START,
            },
            0,
            &real_sizes(),
            256,
            None,
            false,
            |n, _| {
                rounds.set(rounds.get() + 1);
                usage.set(FakeHeap::START + cache_bytes(&real_sizes(), n));
                Ok(())
            },
            || {
                Ok(vk::HeapBudget {
                    size: 16_304 * MIB,
                    budget: usage.get(),
                    usage: usage.get(),
                })
            },
        )
        .unwrap();
        assert_eq!(n, 0);
        assert!(rounds.get() < 40, "{} rounds", rounds.get());
    }

    #[test]
    fn errors_other_than_the_budget_are_returned() {
        let heap = FakeHeap::new();
        let r = fit_cache(
            heap.heap(),
            0,
            &real_sizes(),
            256,
            None,
            false,
            |_, _| Err::<(), _>(vk::Error::Timeout),
            || Ok(heap.heap()),
        );
        assert!(matches!(r, Err(vk::Error::Timeout)));
        // The staging pass returns a non-budget error as it is, without retrying unstaged.
        let calls = Cell::new(0);
        let r = fit_cache(
            heap.heap(),
            0,
            &real_sizes(),
            256,
            None,
            true,
            |_, _| {
                calls.set(calls.get() + 1);
                Err::<(), _>(vk::Error::Timeout)
            },
            || Ok(heap.heap()),
        );
        assert!(matches!(r, Err(vk::Error::Timeout)));
        assert_eq!(calls.get(), 1);
    }
}
