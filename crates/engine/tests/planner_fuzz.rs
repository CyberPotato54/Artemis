//! The swap planner driven at random for thousands of steps, with its invariants
//! checked after every call: the CI half of what has to hold before steps and copies may overlap,
//! before a retired chunk's memory is freed, and before the staging area lent to the cache is used for staging again.
//! Copy submits are refused at random, the queues finish late by random amounts, polls come zero to two times a step,
//! and chunks retire and come back, empty slots are filled, and the lent rows go back to staging and return.
use engine::{Entry, Kind, Move, Phase, Rng, RowState, SwapError, SwapPlanner};
use std::collections::{BTreeSet, HashMap};

const N_LAYER: usize = 3;
const N_EXPERT: usize = 12;
/// Rows the cache loads with: two chunks of two rows, each its own memory, then two rows lent by the staging area.
const SLOTS: usize = 4;
const LENT: std::ops::Range<usize> = 4..6;
/// Rows 6..8 are a chunk the cache can grow into.
const ROWS: usize = 8;
const CHUNKS: [std::ops::Range<usize>; 3] = [0..2, 2..4, 6..8];

/// The planner's bookkeeping agrees with itself.
fn check_books(p: &SwapPlanner, at: &str) {
    let moves = p.moves();
    let mut in_flight = BTreeSet::new();
    for m in &moves {
        for e in [m.out, m.inn].into_iter().flatten() {
            assert!(
                in_flight.insert((m.layer, e)),
                "{at}: an expert in two moves: {moves:?}"
            );
        }
        match p.row_state(m.slot) {
            RowState::Live => {}
            RowState::Retiring => assert_eq!(m.kind, Kind::Evict, "{at}: {m:?} in a retiring row"),
            RowState::Dead => panic!("{at}: a move in a dead row: {m:?}"),
        }
        let shape = match m.kind {
            Kind::Swap => m.out.is_some() && m.inn.is_some(),
            Kind::Fill => m.out.is_none() && m.inn.is_some(),
            Kind::Evict => m.inn.is_none(),
        };
        assert!(shape, "{at}: {m:?}");
    }
    for l in 0..N_LAYER {
        for e in 0..N_EXPERT {
            assert_eq!(
                p.busy(l, e),
                in_flight.contains(&(l, e)),
                "{at}: layer {l} expert {e}'s busy flag against the moves in flight {moves:?}"
            );
        }
    }
    for kind in [Kind::Swap, Kind::Fill, Kind::Evict] {
        let (started, finished) = p.counts(kind);
        let open = moves.iter().filter(|m| m.kind == kind).count() as u64;
        assert_eq!(
            started,
            finished + open,
            "{at}: {kind:?}s started, finished and in flight"
        );
    }
    // A retiring row waits on a move in it; with none left it must be Dead, or its memory is never freed.
    for r in 0..p.rows() {
        if p.row_state(r) == RowState::Retiring {
            assert!(
                moves.iter().any(|m| m.slot == r),
                "{at}: row {r} is retiring with no move left in it: {moves:?}"
            );
        }
    }
    for l in 0..N_LAYER {
        let mut held = BTreeSet::new();
        for s in 0..p.rows() {
            let on = moves.iter().filter(|m| m.layer == l && m.slot == s).count();
            assert!(on <= 1, "{at}: two moves in layer {l} slot {s}: {moves:?}");
            if let Some(h) = p.holder(l, s) {
                assert_eq!(
                    p.row_state(s),
                    RowState::Live,
                    "{at}: layer {l} slot {s} holds {h} in a row that isn't live"
                );
                assert!(
                    held.insert(h),
                    "{at}: layer {l}: expert {h} holds two slots"
                );
                // A holder the table doesn't point at is a move's incoming expert, not yet published.
                if p.entry(l, h) != Entry::Vram(s) {
                    assert!(
                        moves
                            .iter()
                            .any(|m| m.layer == l && m.slot == s && m.inn == Some(h)),
                        "{at}: layer {l} slot {s}'s holder {h} is neither published nor coming in"
                    );
                }
            }
        }
        let mut pointed = BTreeSet::new();
        for e in 0..N_EXPERT {
            if let Entry::Vram(s) = p.entry(l, e) {
                assert_ne!(
                    p.row_state(s),
                    RowState::Dead,
                    "{at}: layer {l} expert {e} points at dead row {s}"
                );
                assert!(
                    pointed.insert(s),
                    "{at}: layer {l}: two experts point at slot {s}"
                );
                // An entry names its slot's holder, unless the expert is on its way out of it (a swap's or an
                // eviction's `out`), requested and not yet published as HOST.
                if p.holder(l, s) != Some(e) {
                    assert!(
                        moves.iter().any(|m| m.layer == l
                            && m.slot == s
                            && m.out == Some(e)
                            && m.phase == Phase::Staged),
                        "{at}: layer {l} expert {e} points at slot {s}, which {:?} holds",
                        p.holder(l, s)
                    );
                }
            }
        }
    }
}

/// Everything a refused request must leave unchanged.
type View = (
    Vec<Move>,
    [(u64, u64); 3],
    Vec<RowState>,
    Vec<Vec<Option<usize>>>,
    Vec<Vec<Entry>>,
    (u64, u64),
);

fn view(p: &SwapPlanner) -> View {
    (
        p.moves(),
        [Kind::Swap, Kind::Fill, Kind::Evict].map(|k| p.counts(k)),
        (0..p.rows()).map(|r| p.row_state(r)).collect(),
        (0..N_LAYER)
            .map(|l| (0..p.rows()).map(|s| p.holder(l, s)).collect())
            .collect(),
        (0..N_LAYER)
            .map(|l| (0..N_EXPERT).map(|e| p.entry(l, e)).collect())
            .collect(),
        (p.done(), p.filled()),
    )
}

/// Outcomes of the arbitrary requests: accepted, out of range, not resident, already resident, busy, no empty slot,
/// rows refused.
type Outcomes = [usize; 7];

/// What a run counted.
#[derive(Default)]
struct Ran {
    swaps: u64,
    fills: u64,
    evictions: u64,
    refused_copies: usize,
    retires: usize,
    adds: usize,
    /// The lent rows added back after a staged step took them (their add at load isn't counted).
    lent_adds: usize,
    staged_steps: usize,
}

/// The GPU, faked: what each slot's memory holds, the copies landing in it, and the steps still running.
struct Gpu {
    /// Per layer and row, the expert the memory holds; None for memory freed, new, or used for staging since.
    bytes: Vec<Vec<Option<usize>>>,
    /// Copies not yet landed: (layer, slot) → (transfer value, what the memory holds once it lands). That is the
    /// copy's expert, or None once a staged step has staged into the slot: staging's copies land after it.
    writing: HashMap<(usize, usize), (u64, Option<usize>)>,
    /// Whether a chunk's memory is allocated (the lent rows' always is: it is the staging area's).
    allocated: Vec<bool>,
    /// The steps still running: (compute value, the slots whose memory they use).
    running: Vec<(u64, Vec<(usize, usize)>)>,
}

impl Gpu {
    /// Frees the memory of every own chunk whose rows are all dead, as the engine does after a poll. (d) Nothing a
    /// running step uses, or a copy writes, is freed.
    fn free_dead(&mut self, p: &SwapPlanner, at: &str) {
        for rows in CHUNKS {
            if self.allocated[rows.start] && rows.clone().all(|r| p.row_state(r) == RowState::Dead)
            {
                for r in rows.clone() {
                    for l in 0..N_LAYER {
                        for (v, used) in &self.running {
                            assert!(
                                !used.contains(&(l, r)),
                                "{at}: row {r} freed while step {v} uses it"
                            );
                        }
                        assert!(
                            !self.writing.contains_key(&(l, r)),
                            "{at}: row {r} freed while a copy writes layer {l}'s slot"
                        );
                        self.bytes[l][r] = None;
                    }
                    self.allocated[r] = false;
                }
            }
        }
    }
}

/// One seed's run of `steps` steps with queues advancing up to `advance - 1` a poll, and a staged step and a chunk
/// retired or added back one step in `churn` each. The last fifth settles: no new moves, no refusals, queues caught up,
/// so every move finishes.
fn fuzz(seed: u64, steps: usize, advance: usize, churn: usize, outcomes: &mut Outcomes) -> Ran {
    let mut rng = Rng::new(seed);
    let resident: Vec<Vec<usize>> = (0..N_LAYER).map(|_| (0..SLOTS).collect()).collect();
    let mut p = SwapPlanner::new(N_EXPERT, resident.clone());
    // The lent rows come in empty at load, as the engine adds them; the growable chunk is absent.
    p.add(LENT, 0).unwrap();
    check_books(&p, "start");
    let mut gpu = Gpu {
        bytes: resident
            .iter()
            .map(|r| {
                let mut v: Vec<Option<usize>> = r.iter().map(|&e| Some(e)).collect();
                v.resize(ROWS, None);
                v
            })
            .collect(),
        writing: HashMap::new(),
        allocated: (0..ROWS).map(|r| r < SLOTS || LENT.contains(&r)).collect(),
        running: Vec::new(),
    };
    // The table the GPU applies at each step.
    let mut table: Vec<Vec<Entry>> = (0..N_LAYER)
        .map(|l| (0..N_EXPERT).map(|e| p.entry(l, e)).collect())
        .collect();
    let (mut c_sub, mut c_done, mut t_sub, mut t_done) = (0u64, 0u64, 0u64, 0u64);
    let mut ran = Ran::default();
    let settle = steps * 4 / 5;
    for step in 0..steps {
        let settling = step >= settle;
        let at = |what: &str| format!("seed {seed} step {step} {what}");
        // A staged prefill step (one in `churn`) takes the lent rows back for staging before its polls, as the engine
        // does, so no copy goes into them for nothing; another step returns them after its polls, once they are dead:
        // no step reads them and no copy of the planner's writes them. (A staging copy still landing in them after a
        // failed staged step isn't modelled: the engine's fills into them land after it, behind the transfer queue's
        // leading barrier.)
        let staged = !settling && rng.below(churn) == 0;
        if staged {
            ran.staged_steps += 1;
            if LENT.clone().all(|r| p.row_state(r) == RowState::Live) {
                p.retire(LENT).unwrap();
                check_books(&p, &at("after retiring the lent rows"));
            }
        }
        let polls = if settling { 1 } else { rng.below(3) };
        for _ in 0..polls {
            c_done = if settling {
                c_sub
            } else {
                (c_done + rng.below(advance) as u64).min(c_sub)
            };
            t_done = if settling {
                t_sub
            } else {
                (t_done + rng.below(advance) as u64).min(t_sub)
            };
            gpu.running.retain(|(v, _)| *v > c_done);
            for (&(l, s), &(v, e)) in &gpu.writing {
                if t_done >= v {
                    gpu.bytes[l][s] = e;
                }
            }
            gpu.writing.retain(|_, (v, _)| *v > t_done);
            let g = &mut gpu;
            let r = p.poll(c_done, t_done, |orders| {
                if !settling && rng.below(8) == 0 {
                    ran.refused_copies += 1;
                    return Err("refused");
                }
                t_sub += 1;
                for o in orders {
                    // (a) No running step uses the memory a copy writes, no slot gets two copies at once, and the memory
                    // is allocated.
                    for (v, used) in &g.running {
                        assert!(
                            !used.contains(&(o.layer, o.slot)),
                            "seed {seed} step {step}: a copy into layer {} slot {} while step {v} uses it",
                            o.layer,
                            o.slot
                        );
                    }
                    assert!(
                        !g.writing.contains_key(&(o.layer, o.slot)),
                        "seed {seed} step {step}: two copies into layer {} slot {}",
                        o.layer,
                        o.slot
                    );
                    assert!(g.allocated[o.slot], "seed {seed} step {step}: a copy into freed row {}", o.slot);
                    g.writing.insert((o.layer, o.slot), (t_sub, Some(o.expert)));
                }
                Ok(t_sub)
            });
            assert!(r.is_ok() || !settling, "{}", at(""));
            check_books(&p, &at("after a poll"));
            gpu.free_dead(&p, &at("after a poll"));
        }
        if !staged && LENT.clone().all(|r| p.row_state(r) == RowState::Dead) {
            p.add(LENT, c_sub).unwrap();
            ran.lent_adds += 1;
            check_books(&p, &at("after adding the lent rows back"));
        }
        if !settling {
            for _ in 0..rng.below(3) {
                if let Some((l, out, inn)) = p.random_swap(&mut rng) {
                    p.request(l, out, inn).unwrap();
                    check_books(&p, &at("after a random swap's request"));
                }
            }
            // An empty slot takes a random free expert.
            let l = rng.below(N_LAYER);
            if p.empty_slots(l) > 0 {
                let free: Vec<usize> = (0..N_EXPERT)
                    .filter(|&e| !p.busy(l, e) && p.entry(l, e) == Entry::Host)
                    .collect();
                if !free.is_empty() {
                    p.fill(l, free[rng.below(free.len())]).unwrap();
                    check_books(&p, &at("after a fill"));
                }
            }
            // A chunk retires, or comes back on newly allocated memory, one step in `churn`.
            if rng.below(churn) == 0 {
                let rows = CHUNKS[rng.below(CHUNKS.len())].clone();
                if rows.clone().all(|r| p.row_state(r) == RowState::Live) {
                    p.retire(rows).unwrap();
                    ran.retires += 1;
                    check_books(&p, &at("after a retire"));
                    gpu.free_dead(&p, &at("after a retire"));
                } else if rows
                    .clone()
                    .all(|r| p.row_state(r) == RowState::Dead && !gpu.allocated[r])
                {
                    p.add(rows.clone(), c_sub).unwrap();
                    for r in rows {
                        gpu.allocated[r] = true;
                    }
                    ran.adds += 1;
                    check_books(&p, &at("after an add"));
                }
            }
            // Any request, legal or not: a refusal must change nothing. Ids and rows range one past the table.
            let before = view(&p);
            let (l, a, b) = (
                rng.below(N_LAYER + 1),
                rng.below(N_EXPERT + 1),
                rng.below(N_EXPERT + 1),
            );
            // Only refused retires and adds are tried: the engine retires and adds whole chunks (above), and a row
            // added here would have no memory behind it.
            let rows = {
                let s = rng.below(ROWS + 1);
                s..s + 1 + rng.below(2)
            };
            let r = match rng.below(4) {
                0 => p.fill(l, a).map(|_| ()),
                1 if !rows.clone().all(|r| p.row_state(r) == RowState::Live) => p.retire(rows),
                2 if !rows.clone().all(|r| p.row_state(r) == RowState::Dead) => p.add(rows, c_sub),
                _ => p.request(l, a, b),
            };
            match r {
                Ok(()) => {
                    outcomes[0] += 1;
                    check_books(&p, &at("after the arbitrary request"));
                    gpu.free_dead(&p, &at("after the arbitrary request"));
                }
                Err(e) => {
                    outcomes[match e {
                        SwapError::OutOfRange { .. } => 1,
                        SwapError::NotResident { .. } => 2,
                        SwapError::AlreadyResident { .. } => 3,
                        SwapError::Busy { .. } => 4,
                        SwapError::NoEmptySlot { .. } => 5,
                        SwapError::Rows { .. } => 6,
                    }] += 1;
                    assert_eq!(
                        view(&p),
                        before,
                        "seed {seed} step {step}: a refusal changed the planner"
                    );
                }
            }
            // Work that isn't a step also moves the compute timeline.
            if rng.below(4) == 0 {
                c_sub += 1;
            }
        }
        // The publishing moves finish with this step: each swap counts as done, each fill as filled.
        let publishing = |k: Kind| {
            p.moves()
                .iter()
                .filter(|m| m.kind == k && matches!(m.phase, Phase::Publishing))
                .count() as u64
        };
        let counted = (
            p.done() + publishing(Kind::Swap),
            p.filled() + publishing(Kind::Fill),
        );
        for c in p.next_step(c_sub) {
            table[c.layer][c.expert] = c.to;
        }
        check_books(&p, &at("after next_step"));
        assert_eq!(
            (p.done(), p.filled()),
            counted,
            "{}",
            at("next_step's counts")
        );
        let mut used = Vec::new();
        for (l, row) in table.iter().enumerate() {
            for (e, &entry) in row.iter().enumerate() {
                assert_eq!(
                    entry,
                    p.entry(l, e),
                    "seed {seed} step {step}: the planner's view is the table"
                );
                if let Entry::Vram(s) = entry {
                    // (b) No entry points at a slot being written; (c) it names the expert the slot's memory holds.
                    assert!(
                        !gpu.writing.contains_key(&(l, s)),
                        "seed {seed} step {step}: layer {l} expert {e} points at slot {s} while it is written"
                    );
                    assert_eq!(
                        gpu.bytes[l][s],
                        Some(e),
                        "seed {seed} step {step}: layer {l} slot {s} holds {:?}, but expert {e} points at it",
                        gpu.bytes[l][s]
                    );
                    used.push((l, s));
                }
            }
        }
        if staged {
            // The staged step stages into the lent rows' memory: no entry points there, and what they held is gone.
            // Its staging copies wait for the steps before it and land after the copies already sent (the engine's
            // `stage_async`): a copy still landing in a lent row isn't a hazard here, and what the memory holds after
            // is staging's bytes, not the copy's expert.
            for l in 0..N_LAYER {
                for s in LENT {
                    assert!(
                        !used.contains(&(l, s)),
                        "seed {seed} step {step}: a staged step's table points into the lent rows"
                    );
                    used.push((l, s));
                    gpu.bytes[l][s] = None;
                    if let Some(w) = gpu.writing.get_mut(&(l, s)) {
                        w.1 = None;
                    }
                }
            }
        }
        c_sub += 1;
        gpu.running.push((c_sub, used));
    }
    assert_eq!(
        p.in_flight(),
        0,
        "seed {seed}: moves left after settling: {:?}",
        p.moves()
    );
    ran.swaps = p.done();
    ran.fills = p.filled();
    ran.evictions = p.counts(Kind::Evict).1;
    ran
}

#[test]
fn forty_seeds_of_random_moves_keep_every_invariant() {
    let (mut total, mut outcomes) = (Ran::default(), Outcomes::default());
    // The least a seed made: swaps, fills, evictions, chunk retires, chunk adds, lent rows added back.
    let mut least = (
        u64::MAX,
        u64::MAX,
        u64::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
    );
    for seed in 0..40 {
        let r = fuzz(seed, 3_000, 4, 8, &mut outcomes);
        least = (
            least.0.min(r.swaps),
            least.1.min(r.fills),
            least.2.min(r.evictions),
            least.3.min(r.retires),
            least.4.min(r.adds),
            least.5.min(r.lent_adds),
        );
        // The thresholds are about half the least a seed made (printed below).
        assert!(
            r.swaps > 550,
            "seed {seed}: only {} swaps finished",
            r.swaps
        );
        assert!(
            r.fills > 495 && r.evictions > 520,
            "seed {seed}: {} fills, {} evictions",
            r.fills,
            r.evictions
        );
        assert!(
            r.retires > 64 && r.adds > 64,
            "seed {seed}: {} retires, {} adds",
            r.retires,
            r.adds
        );
        assert!(
            r.lent_adds > 104,
            "seed {seed}: the lent rows came back {} times",
            r.lent_adds
        );
        total.swaps += r.swaps;
        total.fills += r.fills;
        total.evictions += r.evictions;
        total.refused_copies += r.refused_copies;
        total.retires += r.retires;
        total.adds += r.adds;
        total.lent_adds += r.lent_adds;
        total.staged_steps += r.staged_steps;
    }
    eprintln!(
        "{} swaps, {} fills, {} evictions, {} chunk retires, {} chunk adds, {} lent rows' re-adds, {} staged steps over \
         40 seeds; {} copy submits refused and retried",
        total.swaps,
        total.fills,
        total.evictions,
        total.retires,
        total.adds,
        total.lent_adds,
        total.staged_steps,
        total.refused_copies
    );
    eprintln!(
        "the least a seed made: {} swaps, {} fills, {} evictions, {} chunk retires, {} chunk adds, {} lent rows' re-adds",
        least.0, least.1, least.2, least.3, least.4, least.5
    );
    let [ok, range, not_res, already, busy, empty, rows] = outcomes;
    eprintln!(
        "arbitrary requests: {ok} accepted, {range} out of range, {not_res} not resident, {already} already resident, \
         {busy} busy, {empty} no empty slot, {rows} rows refused"
    );
}

#[test]
fn slow_queues_keep_every_invariant_too() {
    // Queues that advance 0 or 1 a poll: the compute queue falls far behind, so swaps drain and rows retire slowly.
    // Chunks retire and staged steps come one step in 64, so swaps live long enough to finish.
    // The least a seed made: swaps, fills, evictions, chunk retires, lent rows added back.
    let mut least = (u64::MAX, u64::MAX, u64::MAX, usize::MAX, usize::MAX);
    for seed in 100..110 {
        let r = fuzz(seed, 3_000, 2, 64, &mut Outcomes::default());
        least = (
            least.0.min(r.swaps),
            least.1.min(r.fills),
            least.2.min(r.evictions),
            least.3.min(r.retires),
            least.4.min(r.lent_adds),
        );
        // The thresholds are about half the least a seed made (printed below).
        assert!(
            r.swaps > 10 && r.retires > 4,
            "seed {seed}: {} swaps, {} retires",
            r.swaps,
            r.retires
        );
        assert!(
            r.fills > 8 && r.evictions > 9 && r.lent_adds > 14,
            "seed {seed}: {} fills, {} evictions, the lent rows came back {} times",
            r.fills,
            r.evictions,
            r.lent_adds
        );
    }
    eprintln!(
        "the least a seed made: {} swaps, {} fills, {} evictions, {} chunk retires, {} lent rows' re-adds",
        least.0, least.1, least.2, least.3, least.4
    );
}
