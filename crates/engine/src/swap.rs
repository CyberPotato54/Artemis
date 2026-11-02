//! The swap protocol as a pure state machine, apart from the GPU so CI tests it. The engine tells it
//! timeline values; it answers with slot-table changes for the next step and copies to submit. A swap moves expert
//! `in` into the cache slot of expert `out` in one layer:
//!
//! 1. **Staged.** `out → HOST` waits for the next step.
//! 2. **Draining.** That step applied it. The planner waits until the compute timeline reaches the value submitted
//!    just before that step: then no step that could read the slot as VRAM is still running.
//! 3. **Copying.** `in`'s host copy goes into the slot on the transfer queue.
//! 4. **Publishing.** The copy has landed. `in → VRAM` waits for the next step, and the swap is done.
//!
//! So no step ever reads a slot while it is being written, and no table entry points at it then.
//!
//! The cache's slots come in rows, one slot in every layer, and rows can be retired and added.
//! A **fill** brings an expert into an empty slot of a live row: it starts at Draining, waiting for the compute
//! value the row was added at, so it can't write memory a step still reads. **Retiring** rows evicts each slot's expert
//! with a swap's first two steps and no copy: `out → HOST` waits for the next step, then the slot drains. Swaps and
//! fills into a retiring slot are cancelled: the incoming expert is never published, and a copy already sent is waited
//! for. A retired row is **Dead** once nothing reads or writes its slots; only then may its memory be freed.

use crate::rng::Rng;
use std::{fmt, ops::Range};

/// Where a slot-table entry points, as the planner tracks it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entry {
    /// The expert's pinned host copy.
    Host,
    /// Slot `n` of the layer's cache.
    Vram(usize),
}

/// One entry change, applied by `apply_slot_updates` at the start of the next step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Change {
    pub layer: usize,
    pub expert: usize,
    pub to: Entry,
}

/// A copy the engine submits on the transfer queue: `expert`'s host copy into `slot` of `layer`'s cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyOrder {
    pub layer: usize,
    pub expert: usize,
    pub slot: usize,
}

/// Where a move in flight is in the protocol (the numbered steps above).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Requested: `out → HOST` waits for the next step.
    Staged,
    /// Waiting for the compute timeline to reach `until`: for a swap or an eviction, the last value submitted before
    /// the step that published `out → HOST`; for a fill, the value its row was added at. Then no step that could read
    /// the slot is still running.
    Draining { until: u64 },
    /// The copy into the slot was submitted and signals transfer value `until`.
    Copying { until: u64 },
    /// The copy has landed: `in → VRAM(slot)` waits for the next step.
    Publishing,
}

/// What a move in flight does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `in` replaces `out` in its slot (`request`).
    Swap,
    /// `in` goes into an empty slot (`fill`).
    Fill,
    /// `out` leaves a retiring slot and nothing comes in (`retire`); also what a cancelled swap or fill leaves behind
    /// while its expert drains or its copy, already sent, lands.
    Evict,
}

/// A move in flight, as `SwapPlanner::moves` reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    pub kind: Kind,
    pub layer: usize,
    /// The expert leaving the slot, if any.
    pub out: Option<usize>,
    /// The expert coming in, if any.
    pub inn: Option<usize>,
    pub slot: usize,
    pub phase: Phase,
}

/// What a row of slots is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowState {
    /// In use: each of its slots holds an expert, receives one, or is empty and may be filled.
    Live,
    /// Retiring: no expert will be published in it again, and a step or a copy may still use it.
    Retiring,
    /// Nothing the planner tracks uses it: no step still running reads it through the table (each expert that was
    /// published in it has gone HOST and drained), no copy of the planner's writes it (each cancelled copy has landed),
    /// and no entry points at it. Its memory may then be freed, or the row added again. The planner knows nothing of
    /// what used the memory before `add`: a row retired before compute reaches the `ready` it was added at is Dead at
    /// once, so whatever used the memory before the `add` is the caller's to wait for.
    Dead,
}

/// Where an expert stands, for the cache policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    /// Published in a slot and in no move: it may be swapped out.
    Held,
    /// Read from pinned host memory and in no move: it may be swapped or filled in.
    Free,
    /// In a move in flight.
    Busy,
}

/// Why `SwapPlanner` refused a request. A refusal changes nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SwapError {
    /// No such layer or expert.
    OutOfRange { layer: usize, expert: usize },
    /// The expert to swap out isn't in the cache.
    NotResident { layer: usize, expert: usize },
    /// The expert to swap in already is.
    AlreadyResident { layer: usize, expert: usize },
    /// The expert is part of a move in flight.
    Busy { layer: usize, expert: usize },
    /// A fill with no empty slot in a live row of the layer.
    NoEmptySlot { layer: usize },
    /// Rows to retire that aren't all live, or rows to add that aren't all dead; or an empty range, which `retire` and
    /// `add` both refuse.
    Rows { rows: Range<usize>, want: RowState },
}

impl fmt::Display for SwapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SwapError::OutOfRange { layer, expert } => {
                write!(f, "layer {layer}: expert {expert} is not in the slot table")
            }
            SwapError::NotResident { layer, expert } => write!(
                f,
                "layer {layer}: expert {expert} is not in the cache, so it can't be swapped out"
            ),
            SwapError::AlreadyResident { layer, expert } => write!(
                f,
                "layer {layer}: expert {expert} is already in the cache, so it can't be swapped in"
            ),
            SwapError::Busy { layer, expert } => {
                write!(f, "layer {layer}: expert {expert} is already in a swap")
            }
            SwapError::NoEmptySlot { layer } => {
                write!(f, "layer {layer} has no empty slot in a live row to fill")
            }
            SwapError::Rows { rows, .. } if rows.is_empty() => write!(
                f,
                "slot rows {}..{} are an empty range: no row to retire or add",
                rows.start, rows.end
            ),
            SwapError::Rows { rows, want } => write!(
                f,
                "slot rows {}..{} aren't all {}",
                rows.start,
                rows.end,
                match want {
                    RowState::Live => "live",
                    RowState::Retiring => "retiring",
                    RowState::Dead => "dead",
                }
            ),
        }
    }
}

impl std::error::Error for SwapError {}

/// The cache's residency, its rows and the moves in flight.
pub struct SwapPlanner {
    n_expert: usize,
    /// Per layer, per slot, the expert the slot holds or is receiving; None when it is empty, retiring or dead.
    holders: Vec<Vec<Option<usize>>>,
    /// Per row (slot index), its state.
    rows: Vec<RowState>,
    /// Per row, the compute value a fill into it waits for (`add`'s `ready`); 0 for the rows `new` makes.
    ready: Vec<u64>,
    /// Per layer and expert, the entry the last `next_step` published.
    entries: Vec<Vec<Entry>>,
    /// Per layer and expert, whether a move in flight involves it.
    busy: Vec<Vec<bool>>,
    moves: Vec<Move>,
    /// Per kind (`Kind as usize`): moves started, and moves finished (published, evicted, or cancelled).
    started: [u64; 3],
    finished: [u64; 3],
    /// Swaps whose `in → VRAM` went out with a step.
    done: u64,
    /// Fills whose `in → VRAM` went out with a step.
    filled: u64,
}

impl SwapPlanner {
    /// `resident[l][s]` is the expert the initial fill put in slot `s` of layer `l` (every layer has the same number
    /// of slots, every row live); every other expert is HOST.
    pub fn new(n_expert: usize, resident: Vec<Vec<usize>>) -> SwapPlanner {
        let slots = resident.first().map_or(0, Vec::len);
        let mut entries = vec![vec![Entry::Host; n_expert]; resident.len()];
        for (l, layer) in resident.iter().enumerate() {
            assert_eq!(
                layer.len(),
                slots,
                "layer {l} has {} slots, not {slots}",
                layer.len()
            );
            for (s, &e) in layer.iter().enumerate() {
                assert!(e < n_expert, "layer {l}: expert {e} of {n_expert}");
                assert_eq!(
                    entries[l][e],
                    Entry::Host,
                    "layer {l}: expert {e} fills two slots"
                );
                entries[l][e] = Entry::Vram(s);
            }
        }
        SwapPlanner {
            n_expert,
            busy: vec![vec![false; n_expert]; resident.len()],
            holders: resident
                .into_iter()
                .map(|l| l.into_iter().map(Some).collect())
                .collect(),
            rows: vec![RowState::Live; slots],
            ready: vec![0; slots],
            entries,
            moves: Vec::new(),
            started: [0; 3],
            finished: [0; 3],
            done: 0,
            filled: 0,
        }
    }

    pub fn n_layer(&self) -> usize {
        self.holders.len()
    }

    /// Slot indices per layer, in rows of every state.
    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    /// Live rows: the slots per layer the cache can hold experts in now.
    pub fn live_rows(&self) -> usize {
        self.rows.iter().filter(|&&r| r == RowState::Live).count()
    }

    /// Row `row`'s state; Dead past the last row.
    pub fn row_state(&self, row: usize) -> RowState {
        self.rows.get(row).copied().unwrap_or(RowState::Dead)
    }

    /// The entry the last `next_step` published: what the table holds for the step being prepared. Panics if `layer`
    /// or `expert` is out of range.
    pub fn entry(&self, layer: usize, expert: usize) -> Entry {
        assert!(
            layer < self.holders.len(),
            "layer {layer} of {}: the MTP block (layer n_layer) is HOST and never in the planner (C13)",
            self.holders.len()
        );
        self.entries[layer][expert]
    }

    /// `entry`, or None for a layer or expert out of range.
    pub fn get_entry(&self, layer: usize, expert: usize) -> Option<Entry> {
        self.entries.get(layer)?.get(expert).copied()
    }

    /// The expert that holds `slot`, or is being copied into it; None for an empty, retiring or dead slot. Panics if
    /// `layer` or `slot` is out of range.
    pub fn holder(&self, layer: usize, slot: usize) -> Option<usize> {
        self.holders[layer][slot]
    }

    /// Whether a move in flight involves `expert`. Panics if `layer` or `expert` is out of range.
    pub fn busy(&self, layer: usize, expert: usize) -> bool {
        self.busy[layer][expert]
    }

    /// Where `expert` stands, for the cache policy. Panics if `layer` or `expert` is out of range.
    pub fn residency(&self, layer: usize, expert: usize) -> Residency {
        match (self.busy[layer][expert], self.entries[layer][expert]) {
            (true, _) => Residency::Busy,
            (false, Entry::Vram(_)) => Residency::Held,
            (false, Entry::Host) => Residency::Free,
        }
    }

    /// Empty slots of live rows in `layer`, which `fill` may use. Panics if `layer` is out of range.
    pub fn empty_slots(&self, layer: usize) -> usize {
        (0..self.rows.len())
            .filter(|&s| self.rows[s] == RowState::Live && self.holders[layer][s].is_none())
            .count()
    }

    /// Moves in flight, from Staged to Publishing.
    pub fn in_flight(&self) -> usize {
        self.moves.len()
    }

    /// Each move in flight, in the order they started. For tests and diagnostics.
    pub fn moves(&self) -> Vec<Move> {
        self.moves.clone()
    }

    /// Swaps requested so far (`request`: forced ones and the policy's). Fills and evictions aren't swaps.
    pub fn started(&self) -> u64 {
        self.started[Kind::Swap as usize]
    }

    /// Swaps whose `in → VRAM` change has gone out with a step.
    pub fn done(&self) -> u64 {
        self.done
    }

    /// Fills whose `in → VRAM` change has gone out with a step: experts the cache's empty slots now serve.
    pub fn filled(&self) -> u64 {
        self.filled
    }

    /// Moves of `kind` started, and finished (published, evicted, or cancelled by a retire).
    pub fn counts(&self, kind: Kind) -> (u64, u64) {
        (self.started[kind as usize], self.finished[kind as usize])
    }

    fn check(&self, layer: usize, expert: usize) -> Result<(), SwapError> {
        if layer >= self.n_layer() || expert >= self.n_expert {
            return Err(SwapError::OutOfRange { layer, expert });
        }
        Ok(())
    }

    /// Starts swapping `inn` into `out`'s slot. Refuses ids out of range, an `out` that isn't resident, an `inn` that
    /// is, and either one already in a move.
    pub fn request(&mut self, layer: usize, out: usize, inn: usize) -> Result<(), SwapError> {
        self.check(layer, out)?;
        self.check(layer, inn)?;
        for expert in [out, inn] {
            if self.busy[layer][expert] {
                return Err(SwapError::Busy { layer, expert });
            }
        }
        let Entry::Vram(slot) = self.entries[layer][out] else {
            return Err(SwapError::NotResident { layer, expert: out });
        };
        if self.entries[layer][inn] != Entry::Host {
            return Err(SwapError::AlreadyResident { layer, expert: inn });
        }
        self.holders[layer][slot] = Some(inn);
        self.busy[layer][out] = true;
        self.busy[layer][inn] = true;
        self.start(Move {
            kind: Kind::Swap,
            layer,
            out: Some(out),
            inn: Some(inn),
            slot,
            phase: Phase::Staged,
        });
        Ok(())
    }

    /// Starts copying `inn` into the lowest empty slot of a live row in `layer`, and returns the slot. The copy goes out
    /// once compute has reached the value the row was added at. Refuses ids out of range, an `inn` resident or busy,
    /// and a layer with no empty slot.
    pub fn fill(&mut self, layer: usize, inn: usize) -> Result<usize, SwapError> {
        self.check(layer, inn)?;
        if self.busy[layer][inn] {
            return Err(SwapError::Busy { layer, expert: inn });
        }
        if self.entries[layer][inn] != Entry::Host {
            return Err(SwapError::AlreadyResident { layer, expert: inn });
        }
        let slot = (0..self.rows.len())
            .find(|&s| self.rows[s] == RowState::Live && self.holders[layer][s].is_none())
            .ok_or(SwapError::NoEmptySlot { layer })?;
        self.holders[layer][slot] = Some(inn);
        self.busy[layer][inn] = true;
        self.start(Move {
            kind: Kind::Fill,
            layer,
            out: None,
            inn: Some(inn),
            slot,
            phase: Phase::Draining {
                until: self.ready[slot],
            },
        });
        Ok(slot)
    }

    /// Retires `rows` in every layer: each slot's expert goes HOST from the next step and the
    /// slot drains; swaps and fills into these slots are cancelled, their incoming experts never published, and a copy
    /// already sent is waited for. Once nothing reads or writes them the rows are Dead (`row_state`). Refuses an empty
    /// range, and rows that aren't all live.
    pub fn retire(&mut self, rows: Range<usize>) -> Result<(), SwapError> {
        if rows.is_empty() || rows.clone().any(|r| self.row_state(r) != RowState::Live) {
            return Err(SwapError::Rows {
                rows,
                want: RowState::Live,
            });
        }
        for r in rows.clone() {
            self.rows[r] = RowState::Retiring;
        }
        for l in 0..self.n_layer() {
            for s in rows.clone() {
                match self.moves.iter().position(|m| m.layer == l && m.slot == s) {
                    Some(i) => self.cancel(i),
                    None => {
                        if let Some(h) = self.holders[l][s] {
                            // A published expert in no move leaves as a swap's `out` does.
                            self.busy[l][h] = true;
                            self.start(Move {
                                kind: Kind::Evict,
                                layer: l,
                                out: Some(h),
                                inn: None,
                                slot: s,
                                phase: Phase::Staged,
                            });
                        }
                    }
                }
                self.holders[l][s] = None;
            }
        }
        self.settle();
        Ok(())
    }

    /// The move at `i` targets a retiring slot. Its incoming expert is dropped; what is left is an eviction of its
    /// `out` (still to be published HOST, or draining), a sent copy to wait for, or nothing.
    fn cancel(&mut self, i: usize) {
        let m = self.moves[i];
        if m.kind == Kind::Evict {
            return;
        }
        if let Some(inn) = m.inn {
            self.busy[m.layer][inn] = false;
        }
        self.finished[m.kind as usize] += 1;
        let keep = match m.phase {
            // `out` is still published in the slot (Staged), or has just gone HOST and the slot drains for it.
            Phase::Staged | Phase::Draining { .. } => m.out.is_some(),
            // The copy into the slot is in flight: wait for it.
            Phase::Copying { .. } => true,
            // The copy landed and nothing was published: nothing is left to do.
            Phase::Publishing => false,
        };
        if keep {
            self.started[Kind::Evict as usize] += 1;
            self.moves[i] = Move {
                kind: Kind::Evict,
                inn: None,
                ..m
            };
        } else {
            if let Some(out) = m.out {
                self.busy[m.layer][out] = false;
            }
            self.moves.remove(i);
        }
    }

    /// Adds `rows` to every layer, empty, for `fill` to fill once compute has reached `ready`: the last value submitted
    /// before the rows' memory was handed over, so no step that used it before is still running. Rows past `rows()` are
    /// new, and any between are created Dead. Refuses an empty range, and rows that aren't all Dead.
    pub fn add(&mut self, rows: Range<usize>, ready: u64) -> Result<(), SwapError> {
        if rows.is_empty() || rows.clone().any(|r| self.row_state(r) != RowState::Dead) {
            return Err(SwapError::Rows {
                rows,
                want: RowState::Dead,
            });
        }
        if rows.end > self.rows.len() {
            self.rows.resize(rows.end, RowState::Dead);
            self.ready.resize(rows.end, 0);
            for h in &mut self.holders {
                h.resize(rows.end, None);
            }
        }
        for r in rows {
            self.rows[r] = RowState::Live;
            self.ready[r] = ready;
        }
        Ok(())
    }

    fn start(&mut self, m: Move) {
        self.started[m.kind as usize] += 1;
        self.moves.push(m);
    }

    /// Ends the move at `i`, which publishes nothing more, and frees its experts.
    fn finish(&mut self, i: usize) {
        let m = self.moves.remove(i);
        for e in [m.out, m.inn].into_iter().flatten() {
            self.busy[m.layer][e] = false;
        }
        self.finished[m.kind as usize] += 1;
    }

    /// Retiring rows with no move left on any of their slots are Dead.
    fn settle(&mut self) {
        for r in 0..self.rows.len() {
            if self.rows[r] == RowState::Retiring && !self.moves.iter().any(|m| m.slot == r) {
                self.rows[r] = RowState::Dead;
            }
        }
    }

    /// The changes the next step applies. `submitted` is the compute timeline's last submitted value. Staged moves
    /// publish `out → HOST` and start draining until `submitted`; Publishing moves publish `in → VRAM(slot)` and finish.
    pub fn next_step(&mut self, submitted: u64) -> Vec<Change> {
        let mut changes = Vec::new();
        let mut i = 0;
        while i < self.moves.len() {
            let m = self.moves[i];
            match m.phase {
                Phase::Staged => {
                    let out = m.out.expect("a staged move sends an expert out");
                    self.entries[m.layer][out] = Entry::Host;
                    changes.push(Change {
                        layer: m.layer,
                        expert: out,
                        to: Entry::Host,
                    });
                    self.moves[i].phase = Phase::Draining { until: submitted };
                    i += 1;
                }
                Phase::Publishing => {
                    let inn = m.inn.expect("a publishing move brings an expert in");
                    self.entries[m.layer][inn] = Entry::Vram(m.slot);
                    changes.push(Change {
                        layer: m.layer,
                        expert: inn,
                        to: Entry::Vram(m.slot),
                    });
                    match m.kind {
                        Kind::Swap => self.done += 1,
                        Kind::Fill => self.filled += 1,
                        Kind::Evict => unreachable!("an eviction brings nothing in"),
                    }
                    self.finish(i);
                }
                Phase::Draining { .. } | Phase::Copying { .. } => i += 1,
            }
        }
        changes
    }

    /// Moves things on by the timelines' completed values: copies that have landed start publishing (cancelled ones
    /// end); drained evictions end, with no copy; and the copies of every other drained move go out together through one
    /// call of `submit` (none when there are none), which returns the transfer timeline value they signal. Retiring rows
    /// whose moves have all ended are then Dead.
    ///
    /// If `submit` fails, the error is returned and the progress made before it stays: copies that landed publish (or
    /// end), drained evictions end, and retiring rows left with no move are Dead; the moves whose copies were refused
    /// stay Draining, and the next `poll` submits them again.
    pub fn poll<E>(
        &mut self,
        compute_done: u64,
        transfer_done: u64,
        submit: impl FnOnce(&[CopyOrder]) -> Result<u64, E>,
    ) -> Result<(), E> {
        let mut i = 0;
        while i < self.moves.len() {
            let m = self.moves[i];
            match m.phase {
                Phase::Copying { until } if transfer_done >= until => {
                    if m.inn.is_some() {
                        self.moves[i].phase = Phase::Publishing;
                        i += 1;
                    } else {
                        self.finish(i);
                    }
                }
                Phase::Draining { until } if compute_done >= until && m.inn.is_none() => {
                    self.finish(i)
                }
                _ => i += 1,
            }
        }
        let drained: Vec<usize> = (0..self.moves.len())
            .filter(|&i| matches!(self.moves[i].phase, Phase::Draining { until } if compute_done >= until))
            .collect();
        let out = if drained.is_empty() {
            Ok(())
        } else {
            let orders: Vec<CopyOrder> = drained
                .iter()
                .map(|&i| {
                    let m = self.moves[i];
                    CopyOrder {
                        layer: m.layer,
                        expert: m.inn.expect("a drained move with no copy has ended"),
                        slot: m.slot,
                    }
                })
                .collect();
            submit(&orders).map(|v| {
                for &i in &drained {
                    self.moves[i].phase = Phase::Copying { until: v };
                }
            })
        };
        self.settle();
        out
    }

    /// A random legal swap (layer, out, in): a layer with both a free resident and a free non-resident expert, each
    /// such layer equally likely however many experts it has free, then a uniformly random free resident expert and
    /// free non-resident one in it. "Free" means not busy. None when no layer has both.
    pub fn random_swap(&self, rng: &mut Rng) -> Option<(usize, usize, usize)> {
        let free = |l: usize, resident: bool| -> Vec<usize> {
            (0..self.n_expert)
                .filter(|&e| {
                    !self.busy[l][e] && matches!(self.entries[l][e], Entry::Vram(_)) == resident
                })
                .collect()
        };
        let mut layers: Vec<(usize, Vec<usize>, Vec<usize>)> = (0..self.n_layer())
            .filter_map(|l| {
                let (outs, ins) = (free(l, true), free(l, false));
                (!outs.is_empty() && !ins.is_empty()).then_some((l, outs, ins))
            })
            .collect();
        if layers.is_empty() {
            return None;
        }
        let (l, outs, ins) = layers.swap_remove(rng.below(layers.len()));
        let out = outs[rng.below(outs.len())];
        let inn = ins[rng.below(ins.len())];
        Some((l, out, inn))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(slots: &[usize]) -> SwapPlanner {
        SwapPlanner::new(16, vec![slots.to_vec()])
    }

    /// Each move in flight as (layer, out, in, phase).
    type Brief = (usize, Option<usize>, Option<usize>, Phase);

    fn phases(p: &SwapPlanner) -> Vec<Brief> {
        p.moves()
            .iter()
            .map(|m| (m.layer, m.out, m.inn, m.phase))
            .collect()
    }

    /// Polls with the timelines at `compute` and `transfer`; the copies submitted signal transfer value 1.
    fn copies(p: &mut SwapPlanner, compute: u64, transfer: u64) -> Vec<CopyOrder> {
        let mut orders = Vec::new();
        p.poll(compute, transfer, |o| {
            orders.extend_from_slice(o);
            Ok::<u64, ()>(1)
        })
        .unwrap();
        orders
    }

    #[test]
    fn a_swap_moves_through_every_phase() {
        let mut p = fill(&[0, 1, 2, 3]);
        p.request(0, 2, 9).unwrap();
        assert_eq!(phases(&p), [(0, Some(2), Some(9), Phase::Staged)]);
        assert_eq!(
            p.holder(0, 2),
            Some(9),
            "the slot is reserved for the new expert"
        );
        assert_eq!(
            p.entry(0, 2),
            Entry::Vram(2),
            "nothing is published before a step"
        );
        let host = Change {
            layer: 0,
            expert: 2,
            to: Entry::Host,
        };
        assert_eq!(p.next_step(5), [host]);
        assert_eq!(p.entry(0, 2), Entry::Host);
        assert_eq!(
            phases(&p),
            [(0, Some(2), Some(9), Phase::Draining { until: 5 })]
        );
        // The step before the change (value 5) may still read slot 2: no copy yet.
        assert_eq!(copies(&mut p, 4, 0), []);
        assert_eq!(p.next_step(6), [], "a draining swap changes nothing");
        assert_eq!(
            copies(&mut p, 5, 0),
            [CopyOrder {
                layer: 0,
                expert: 9,
                slot: 2
            }]
        );
        assert_eq!(
            phases(&p),
            [(0, Some(2), Some(9), Phase::Copying { until: 1 })]
        );
        assert_eq!(copies(&mut p, 6, 0), [], "one copy per swap");
        assert_eq!(copies(&mut p, 6, 1), []);
        assert_eq!(phases(&p), [(0, Some(2), Some(9), Phase::Publishing)]);
        assert_eq!(
            p.next_step(7),
            [Change {
                layer: 0,
                expert: 9,
                to: Entry::Vram(2)
            }]
        );
        assert_eq!(p.entry(0, 9), Entry::Vram(2));
        assert_eq!((p.in_flight(), p.started(), p.done()), (0, 1, 1));
        assert!(!p.busy(0, 2) && !p.busy(0, 9));
        // Expert 2 can come back, into the slot expert 9 now holds.
        p.request(0, 9, 2).unwrap();
    }

    #[test]
    fn a_failed_copy_submit_is_reported_and_retried() {
        let mut p = fill(&[0, 1]);
        p.request(0, 0, 5).unwrap();
        p.next_step(3);
        assert_eq!(p.poll(3, 0, |_| Err("device lost")), Err("device lost"));
        assert_eq!(
            phases(&p),
            [(0, Some(0), Some(5), Phase::Draining { until: 3 })]
        );
        p.poll(3, 0, |_| Ok::<u64, ()>(8)).unwrap();
        assert_eq!(
            phases(&p),
            [(0, Some(0), Some(5), Phase::Copying { until: 8 })]
        );
    }

    #[test]
    fn illegal_requests_are_refused() {
        let mut p = fill(&[0, 1, 2, 3]);
        assert_eq!(
            p.request(1, 0, 9),
            Err(SwapError::OutOfRange {
                layer: 1,
                expert: 0
            })
        );
        assert_eq!(
            p.request(0, 0, 16),
            Err(SwapError::OutOfRange {
                layer: 0,
                expert: 16
            })
        );
        assert_eq!(
            p.request(0, 9, 10),
            Err(SwapError::NotResident {
                layer: 0,
                expert: 9
            })
        );
        assert_eq!(
            p.request(0, 0, 1),
            Err(SwapError::AlreadyResident {
                layer: 0,
                expert: 1
            })
        );
        p.request(0, 0, 9).unwrap();
        for (out, inn, busy) in [(0, 10, 0), (1, 9, 9)] {
            assert_eq!(
                p.request(0, out, inn),
                Err(SwapError::Busy {
                    layer: 0,
                    expert: busy
                })
            );
        }
        // An empty range names no row: retiring or adding it is refused.
        assert_eq!(
            p.retire(3..3),
            Err(SwapError::Rows {
                rows: 3..3,
                want: RowState::Live
            })
        );
        assert_eq!(
            p.add(3..3, 0),
            Err(SwapError::Rows {
                rows: 3..3,
                want: RowState::Dead
            })
        );
        assert_eq!(p.in_flight(), 1, "a refused request changes nothing");
        assert_eq!(p.started(), 1);
        let msg = SwapError::NotResident {
            layer: 3,
            expert: 17,
        }
        .to_string();
        assert_eq!(
            msg,
            "layer 3: expert 17 is not in the cache, so it can't be swapped out"
        );
        assert_eq!(
            SwapError::Rows {
                rows: 2..5,
                want: RowState::Dead
            }
            .to_string(),
            "slot rows 2..5 aren't all dead"
        );
        assert_eq!(
            SwapError::Rows {
                rows: 3..3,
                want: RowState::Live
            }
            .to_string(),
            "slot rows 3..3 are an empty range: no row to retire or add"
        );
    }

    #[test]
    fn random_swaps_are_legal_and_none_without_room() {
        let mut rng = Rng::new(5);
        assert_eq!(
            SwapPlanner::new(4, vec![vec![]; 3]).random_swap(&mut rng),
            None
        );
        let full = SwapPlanner::new(4, vec![vec![0, 1, 2, 3]; 3]);
        assert_eq!(full.random_swap(&mut rng), None);
        let mut p = SwapPlanner::new(8, vec![vec![0, 1, 2]; 3]);
        for _ in 0..100 {
            let (l, out, inn) = p.random_swap(&mut rng).unwrap();
            assert!(matches!(p.entry(l, out), Entry::Vram(_)) && p.entry(l, inn) == Entry::Host);
            p.request(l, out, inn).unwrap();
            let v = p.started();
            p.next_step(v);
            p.poll(v, v, |_| Ok::<u64, ()>(v)).unwrap();
            p.poll(v, v, |_| Ok::<u64, ()>(v)).unwrap();
            p.next_step(v);
            assert_eq!(p.in_flight(), 0);
        }
        // With every free expert of every layer busy, there is no legal swap.
        let mut p = SwapPlanner::new(2, vec![vec![0]; 2]);
        p.request(0, 0, 1).unwrap();
        p.request(1, 0, 1).unwrap();
        assert_eq!(p.random_swap(&mut rng), None);
    }

    #[test]
    fn the_copies_of_a_poll_go_out_together_or_not_at_all() {
        // Two drained swaps: their copies go in one submit and signal one value. A refused submit leaves both draining,
        // and the next poll sends both again. The refused poll keeps what it did before the submit: a third swap whose
        // copy has landed starts publishing, and a drained eviction ends, its row Dead. With nothing drained, a poll
        // submits nothing.
        let mut p = fill(&[0, 1, 2, 3]);
        // The third swap, 2 → 10, copying until transfer value 1.
        p.request(0, 2, 10).unwrap();
        p.next_step(1);
        copies(&mut p, 1, 0);
        // Row 3's expert is evicted, and drains with the two swaps.
        p.retire(3..4).unwrap();
        p.request(0, 0, 8).unwrap();
        p.request(0, 1, 9).unwrap();
        p.next_step(3);
        assert_eq!(
            phases(&p),
            [
                (0, Some(2), Some(10), Phase::Copying { until: 1 }),
                (0, Some(3), None, Phase::Draining { until: 3 }),
                (0, Some(0), Some(8), Phase::Draining { until: 3 }),
                (0, Some(1), Some(9), Phase::Draining { until: 3 }),
            ]
        );
        let mut batches = Vec::new();
        assert_eq!(
            p.poll(3, 1, |o| {
                batches.push(o.to_vec());
                Err("device lost")
            }),
            Err("device lost")
        );
        assert_eq!(
            phases(&p),
            [
                (0, Some(2), Some(10), Phase::Publishing),
                (0, Some(0), Some(8), Phase::Draining { until: 3 }),
                (0, Some(1), Some(9), Phase::Draining { until: 3 }),
            ]
        );
        assert_eq!(
            (p.counts(Kind::Evict), p.row_state(3)),
            ((1, 1), RowState::Dead)
        );
        p.poll(3, 1, |o| {
            batches.push(o.to_vec());
            Ok::<u64, ()>(5)
        })
        .unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], batches[1]);
        assert_eq!(
            batches[1].iter().map(|o| o.expert).collect::<Vec<_>>(),
            [8, 9]
        );
        assert_eq!(
            phases(&p),
            [
                (0, Some(2), Some(10), Phase::Publishing),
                (0, Some(0), Some(8), Phase::Copying { until: 5 }),
                (0, Some(1), Some(9), Phase::Copying { until: 5 }),
            ]
        );
        // Nothing drained: no submit at all.
        p.poll(3, 1, |_| -> Result<u64, ()> { panic!("an empty submit") })
            .unwrap();
    }

    #[test]
    fn a_fill_copies_into_an_empty_slot_once_its_row_is_ready() {
        // Rows 4 and 5 are added at compute value 7, when their memory was handed over: a fill into row 4 waits for
        // it, then copies, then publishes. A fill isn't a swap: test 5's forced run counts swaps.
        let mut p = fill(&[0, 1, 2, 3]);
        assert_eq!(p.fill(0, 9), Err(SwapError::NoEmptySlot { layer: 0 }));
        p.add(4..6, 7).unwrap();
        assert_eq!((p.rows(), p.live_rows(), p.empty_slots(0)), (6, 6, 2));
        assert_eq!(p.fill(0, 9), Ok(4));
        assert_eq!(
            p.fill(0, 2),
            Err(SwapError::AlreadyResident {
                layer: 0,
                expert: 2
            })
        );
        assert_eq!(
            p.fill(0, 9),
            Err(SwapError::Busy {
                layer: 0,
                expert: 9
            })
        );
        assert_eq!(
            phases(&p),
            [(0, None, Some(9), Phase::Draining { until: 7 })]
        );
        assert_eq!(
            p.next_step(8),
            [],
            "nothing is published before the copy lands"
        );
        assert_eq!(
            copies(&mut p, 6, 0),
            [],
            "a step from before the rows were added may still use their memory"
        );
        assert_eq!(
            copies(&mut p, 7, 0),
            [CopyOrder {
                layer: 0,
                expert: 9,
                slot: 4
            }]
        );
        assert_eq!(
            phases(&p),
            [(0, None, Some(9), Phase::Copying { until: 1 })]
        );
        assert_eq!(
            p.next_step(8),
            [],
            "nothing is published while the copy is in flight"
        );
        copies(&mut p, 7, 1);
        assert_eq!(p.filled(), 0, "a fill counts once it is published");
        assert_eq!(
            p.next_step(9),
            [Change {
                layer: 0,
                expert: 9,
                to: Entry::Vram(4)
            }]
        );
        assert_eq!(p.in_flight(), 0);
        assert_eq!((p.counts(Kind::Fill), p.filled()), ((1, 1), 1));
        assert_eq!((p.started(), p.done()), (0, 0), "a fill isn't a swap");
        assert_eq!((p.residency(0, 9), p.empty_slots(0)), (Residency::Held, 1));
    }

    #[test]
    fn retired_rows_evict_their_experts_and_die_once_drained() {
        // Two layers of four slots; rows 2 and 3 retire. Their experts go HOST with the next step, the slots drain until
        // the value submitted before it, and then the rows are Dead. Added back, they are empty.
        let mut p = SwapPlanner::new(16, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
        p.retire(2..4).unwrap();
        assert_eq!(
            (p.row_state(2), p.row_state(3), p.live_rows()),
            (RowState::Retiring, RowState::Retiring, 2)
        );
        assert_eq!(p.residency(0, 2), Residency::Busy);
        assert_eq!(
            p.entry(0, 2),
            Entry::Vram(2),
            "nothing is published before a step"
        );
        assert_eq!(
            p.request(1, 6, 12),
            Err(SwapError::Busy {
                layer: 1,
                expert: 6
            })
        );
        let changes = p.next_step(5);
        assert_eq!(
            changes
                .iter()
                .map(|c| (c.layer, c.expert))
                .collect::<Vec<_>>(),
            [(0, 2), (0, 3), (1, 6), (1, 7)]
        );
        assert!(changes.iter().all(|c| c.to == Entry::Host));
        assert_eq!(
            copies(&mut p, 4, 0),
            [],
            "the step before may still read the slots"
        );
        assert_eq!(p.row_state(2), RowState::Retiring);
        assert_eq!(copies(&mut p, 5, 0), [], "an eviction copies nothing");
        assert_eq!(
            (p.row_state(2), p.row_state(3), p.row_state(4)),
            (RowState::Dead, RowState::Dead, RowState::Dead)
        );
        assert_eq!((p.in_flight(), p.counts(Kind::Evict)), (0, (4, 4)));
        assert_eq!(p.residency(1, 7), Residency::Free);
        assert_eq!(
            p.retire(1..3),
            Err(SwapError::Rows {
                rows: 1..3,
                want: RowState::Live
            })
        );
        p.add(2..4, 6).unwrap();
        assert_eq!((p.empty_slots(0), p.empty_slots(1)), (2, 2));
        assert_eq!(
            p.add(3..5, 6),
            Err(SwapError::Rows {
                rows: 3..5,
                want: RowState::Dead
            })
        );
        // Rows past the end are new, with a dead gap before them.
        p.add(6..7, 6).unwrap();
        assert_eq!(
            (p.rows(), p.row_state(5), p.live_rows()),
            (7, RowState::Dead, 5)
        );
    }

    #[test]
    fn retiring_cancels_the_moves_into_its_slots_in_every_phase() {
        // One layer of seven slots: in slot 1 a swap still Staged, in 2 one Draining, in 3 one Copying, in 4 one
        // Publishing, in 5 an expert in no move, in 6 a fill Draining; slot 0 stays. Rows 1..7 retire.
        let mut p = fill(&[0, 1, 2, 3, 4, 5]);
        let mut t = 0u64;
        let mut poll = |p: &mut SwapPlanner, compute: u64, transfer: u64| {
            p.poll(compute, transfer, |_| {
                t += 1;
                Ok::<u64, ()>(t)
            })
            .unwrap()
        };
        p.request(0, 4, 12).unwrap();
        p.next_step(1);
        poll(&mut p, 1, 0); // 4 → 12 copies (transfer 1)
        p.request(0, 3, 11).unwrap();
        p.next_step(1);
        p.request(0, 2, 10).unwrap();
        p.next_step(2);
        poll(&mut p, 1, 0); // 3 → 11 copies (transfer 2); 2 → 10 still drains until 2
        p.request(0, 1, 9).unwrap();
        poll(&mut p, 1, 1); // 4 → 12 landed
        p.add(6..7, 9).unwrap();
        assert_eq!(p.fill(0, 13), Ok(6));
        assert_eq!(
            phases(&p),
            [
                (0, Some(4), Some(12), Phase::Publishing),
                (0, Some(3), Some(11), Phase::Copying { until: 2 }),
                (0, Some(2), Some(10), Phase::Draining { until: 2 }),
                (0, Some(1), Some(9), Phase::Staged),
                (0, None, Some(13), Phase::Draining { until: 9 }),
            ]
        );
        p.retire(1..7).unwrap();
        // What is left: 3's slot waits for its copy, 2's drains, 1 and 5 are evicted; 4 → 12 and the fill are gone.
        assert_eq!(
            phases(&p),
            [
                (0, Some(3), None, Phase::Copying { until: 2 }),
                (0, Some(2), None, Phase::Draining { until: 2 }),
                (0, Some(1), None, Phase::Staged),
                (0, Some(5), None, Phase::Staged),
            ]
        );
        assert!(p.moves().iter().all(|m| m.kind == Kind::Evict));
        assert_eq!(
            (1..7).map(|r| p.row_state(r)).collect::<Vec<_>>(),
            [
                RowState::Retiring,
                RowState::Retiring,
                RowState::Retiring,
                RowState::Dead,
                RowState::Retiring,
                RowState::Dead
            ]
        );
        for e in [9, 10, 11, 12, 13, 4] {
            assert_eq!(p.residency(0, e), Residency::Free, "expert {e}");
        }
        assert_eq!(p.counts(Kind::Swap), (4, 4));
        assert_eq!(p.counts(Kind::Fill), (1, 1));
        assert_eq!(p.done(), 0, "no swap was published");
        // The next step sends 1 and 5 HOST and publishes nothing in VRAM.
        assert_eq!(
            p.next_step(3),
            [
                Change {
                    layer: 0,
                    expert: 1,
                    to: Entry::Host
                },
                Change {
                    layer: 0,
                    expert: 5,
                    to: Entry::Host
                }
            ]
        );
        poll(&mut p, 2, 1);
        assert_eq!(
            (p.row_state(2), p.row_state(3)),
            (RowState::Dead, RowState::Retiring),
            "2's slot has drained; 3's copy is still landing"
        );
        poll(&mut p, 3, 2);
        assert_eq!(
            (p.in_flight(), p.live_rows(), p.counts(Kind::Evict)),
            (0, 1, (4, 4))
        );
        assert_eq!(t, 2, "no copy went out after the retire");
        for e in 1..16 {
            assert_eq!(p.entry(0, e), Entry::Host, "expert {e}");
        }
        assert_eq!(p.entry(0, 0), Entry::Vram(0));
    }

    #[test]
    fn the_planner_never_touches_the_mtp_block() {
        // The planner is built over the main layers only: the MTP block (layer n_layer = 4 on the tiny model) isn't in
        // its view, and asking for it panics with the reason.
        let p = SwapPlanner::new(16, vec![vec![0, 1]; 4]);
        assert_eq!(p.n_layer(), 4);
        // Every per-layer table stops at the main layers.
        assert_eq!((p.entries.len(), p.busy.len(), p.holders.len()), (4, 4, 4));
        assert_eq!(p.get_entry(3, 0), Some(Entry::Vram(0)));
        assert_eq!(p.get_entry(4, 0), None, "no entry for the MTP block");
        let err =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.entry(4, 0))).unwrap_err();
        let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(msg.contains("the MTP block"), "{msg}");
    }
}
