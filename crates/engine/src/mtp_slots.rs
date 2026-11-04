//! K-64: the MTP block's (`blk.40`'s) most-routed experts in VRAM slots of their own. The draft pass reads them
//! through the slot table, with the same kernels as every other expert, so no bit depends on where an expert lives.
//!
//! The slots start empty at load (every `blk.40` entry HOST) and fill from the block's own routes: every draft row's
//! eight experts are counted, and at the start of each draft pass up to `MTP_SWAPS_PER_PASS` experts move toward the
//! most-counted ones (ties to the lower id), in the pass's own command buffer: the copies, a barrier, the table's
//! entries (`apply_slot_updates`), a barrier, then the pass. The engine waits for every submit, so no other command
//! reads a slot while it changes. The trunk's cache, its planner and its policy never see these slots.
//!
//! Their VRAM comes out of the cache's budget (`cache::fit_cache_mtp`): at load the trunk's cache is sized with the
//! slots held, while it keeps at least `MTP_TRUNK_FLOOR` slots a layer or all it asked for; with less room the trunk
//! keeps its slots and the MTP slots take only what is left beside it, or none.
use crate::Engine;
use kernels::dispatch::{slot_update, UPDATE_BYTES, WHERE_HOST, WHERE_VRAM};
use std::cell::RefCell;

/// The MTP slots asked for by default: 64 experts of `blk.40` (121,634,816 bytes on the real model). The route trace
/// of plan v011 Task 8 had the top 64 serve 74 to 80% of the block's reads on the write, edit and explain prompts.
pub const MTP_SLOTS: usize = 64;

/// The trunk keeps at least this many slots a layer before it gives VRAM to the MTP slots. On the routing profile a
/// trunk row at rank 48 serves about 1.7 routed reads a decoded token, as much as the 1.6 rows' worth of VRAM the
/// MTP slots take are expected to save (about 75% of 8 reads a draft row, at 0.25 to 0.75 draft rows a token). Below
/// it a trunk row is worth more than its share of the MTP slots.
pub const MTP_TRUNK_FLOOR: usize = 48;

/// Experts copied into the MTP slots at the start of one draft pass, at most: one draft row's worth of reads (about
/// 15 MB on the real model).
pub const MTP_SWAPS_PER_PASS: usize = 8;

/// A resident expert is replaced only by one counted at least this many more times, so two experts near the cut don't
/// trade places on every route.
pub const MTP_SWAP_MARGIN: u32 = 2;

/// Every this many counted draft rows, every count is halved, so the slots follow a change of work.
pub const MTP_HALVE_ROWS: u64 = 4096;

/// Whether the trunk's cache, sized with the MTP slots held, keeps them: it got all it asked for (`limit`, the
/// requested slots or every expert), or at least `MTP_TRUNK_FLOOR`.
pub fn keep_reserved(trunk: usize, limit: usize) -> bool {
    trunk >= limit || trunk >= MTP_TRUNK_FLOOR
}

/// The MTP slots of `slot` bytes that fit in `room` bytes, at most `most`.
pub fn leftover_slots(room: u64, slot: u64, most: usize) -> usize {
    match room.checked_div(slot) {
        Some(n) => usize::try_from(n).unwrap_or(usize::MAX).min(most),
        None => 0,
    }
}

/// One expert copied into one slot: `inn` into `slot`, which held `out` (None for an empty slot).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotMove {
    pub slot: usize,
    pub out: Option<usize>,
    pub inn: usize,
}

/// The MTP slots' accounting: the block's route counts and which expert each slot holds.
#[derive(Clone, Debug)]
pub(crate) struct MtpSlots {
    counts: Vec<u32>,
    /// Draft rows counted since the last halving.
    rows: u64,
    /// Per slot, the expert it holds.
    held: Vec<Option<usize>>,
    /// Per expert, its slot.
    slot_of: Vec<Option<usize>>,
    /// The table's `blk.40` entries may differ from `held` (a pass's submit failed after its moves were recorded):
    /// the next pass rewrites every entry first.
    reset: bool,
}

impl MtpSlots {
    pub fn new(n_expert: usize, slots: usize) -> MtpSlots {
        MtpSlots {
            counts: vec![0; n_expert],
            rows: 0,
            held: vec![None; slots],
            slot_of: vec![None; n_expert],
            reset: false,
        }
    }

    pub fn slots(&self) -> usize {
        self.held.len()
    }

    #[cfg(test)]
    pub fn slot_of(&self, expert: usize) -> Option<usize> {
        self.slot_of[expert]
    }

    #[cfg(test)]
    pub fn count(&self, expert: usize) -> u32 {
        self.counts[expert]
    }

    /// Counts one draft row's routed experts and returns how many of them the slots served (the residency the pass ran
    /// with). Ids past the block's experts are skipped.
    pub fn count_row(&mut self, ids: &[u32]) -> usize {
        let mut served = 0;
        for &id in ids {
            let Some(c) = self.counts.get_mut(id as usize) else {
                continue;
            };
            *c = c.saturating_add(1);
            served += usize::from(self.slot_of[id as usize].is_some());
        }
        self.rows += 1;
        if self.rows >= MTP_HALVE_ROWS {
            self.rows = 0;
            for c in &mut self.counts {
                *c /= 2;
            }
        }
        served
    }

    /// Up to `cap` moves toward the most-counted experts (ties to the lower id): empty slots first, each taking the
    /// best uncached expert counted at least once; then the least-counted resident (ties: the higher id first) gives
    /// way to the best uncached expert while that one is counted at least `MTP_SWAP_MARGIN` more.
    pub fn plan(&self, cap: usize) -> Vec<SlotMove> {
        let c = &self.counts;
        let mut uncached: Vec<usize> = (0..c.len())
            .filter(|&e| c[e] > 0 && self.slot_of[e].is_none())
            .collect();
        uncached.sort_by_key(|&e| (std::cmp::Reverse(c[e]), e));
        let mut free = (0..self.held.len()).filter(|&s| self.held[s].is_none());
        let mut residents: Vec<usize> = self.held.iter().flatten().copied().collect();
        residents.sort_by_key(|&e| (c[e], std::cmp::Reverse(e)));
        let mut residents = residents.into_iter();
        let mut moves = Vec::new();
        for inn in uncached.into_iter().take(cap) {
            if let Some(slot) = free.next() {
                moves.push(SlotMove {
                    slot,
                    out: None,
                    inn,
                });
                continue;
            }
            match residents.next() {
                Some(out) if c[inn] >= c[out].saturating_add(MTP_SWAP_MARGIN) => {
                    moves.push(SlotMove {
                        slot: self.slot_of[out].expect("a resident has a slot"),
                        out: Some(out),
                        inn,
                    })
                }
                // The best uncached expert can't displace the least-counted resident: no later one can.
                _ => break,
            }
        }
        moves
    }

    /// The moves landed (their submit finished).
    pub fn apply(&mut self, moves: &[SlotMove]) {
        for m in moves {
            assert_eq!(
                self.held[m.slot], m.out,
                "slot {} moved from what it holds",
                m.slot
            );
            if let Some(out) = m.out {
                self.slot_of[out] = None;
            }
            assert!(
                self.slot_of[m.inn].is_none(),
                "expert {} moved in twice",
                m.inn
            );
            self.held[m.slot] = Some(m.inn);
            self.slot_of[m.inn] = Some(m.slot);
        }
        self.reset = false;
    }

    /// A pass's submit failed after its moves were recorded: nothing is taken as resident, and the next pass rewrites
    /// every `blk.40` entry.
    pub fn invalidate(&mut self) {
        self.held.fill(None);
        self.slot_of.fill(None);
        self.reset = true;
    }

    /// Whether the next pass must rewrite every `blk.40` entry.
    pub fn reset_pending(&self) -> bool {
        self.reset
    }

    /// The table changes for `moves`, as (entry index, where, address): an expert that leaves goes back to its host
    /// copy (`host(e)`), one that comes in points at its slot (`slot(s)`). With `reset`, every `blk.40` entry is
    /// written: HOST except those coming in. `layer` is the block's index (`n_layer`).
    pub fn updates(
        &self,
        moves: &[SlotMove],
        reset: bool,
        layer: usize,
        host: impl Fn(usize) -> u64,
        slot: impl Fn(usize) -> u64,
    ) -> Vec<(u32, u32, u64)> {
        let n = self.counts.len();
        let index = |e: usize| (layer * n + e) as u32;
        if reset {
            let mut to: Vec<Option<usize>> = vec![None; n];
            for m in moves {
                to[m.inn] = Some(m.slot);
            }
            return (0..n)
                .map(|e| match to[e] {
                    Some(s) => (index(e), WHERE_VRAM, slot(s)),
                    None => (index(e), WHERE_HOST, host(e)),
                })
                .collect();
        }
        let mut out = Vec::with_capacity(2 * moves.len());
        for m in moves {
            if let Some(e) = m.out {
                out.push((index(e), WHERE_HOST, host(e)));
            }
            out.push((index(m.inn), WHERE_VRAM, slot(m.slot)));
        }
        out
    }
}

/// The MTP slots on the GPU: one VRAM buffer of whole `blk.40` slots, the update list a pass applies (pinned host
/// memory, an entry for every expert of the block, as a reset writes), and the accounting.
pub(crate) struct MtpSlotsGpu {
    mem: vk::Buffer,
    /// Bytes of one slot: `blk.40`'s slot layout, a multiple of 256, so every slot starts 256-aligned.
    slot: u64,
    updates: vk::Buffer,
    state: RefCell<MtpSlots>,
}

/// A pass's recorded moves, applied to the accounting once its submit has finished.
#[derive(Default)]
pub(crate) struct Pending {
    moves: Vec<SlotMove>,
    updates: Vec<(u32, u32, u64)>,
}

impl MtpSlotsGpu {
    /// `n` slots of `slot` bytes in `mem` (allocated by the cache's fit), for a block of `n_expert` experts.
    pub fn new(
        dev: &vk::Device,
        mem: vk::Buffer,
        slot: u64,
        n: usize,
        n_expert: usize,
    ) -> Result<MtpSlotsGpu, vk::Error> {
        assert!(
            slot.is_multiple_of(256) && mem.size() >= n as u64 * slot,
            "{n} MTP slots of {slot} bytes in {} bytes",
            mem.size()
        );
        let updates = dev.alloc(
            (n_expert * UPDATE_BYTES) as u64,
            vk::MemKind::HostPinned,
            "the MTP slots' update list",
        )?;
        Ok(MtpSlotsGpu {
            mem,
            slot,
            updates,
            state: RefCell::new(MtpSlots::new(n_expert, n)),
        })
    }

    pub fn slots(&self) -> usize {
        self.state.borrow().slots()
    }

    /// The VRAM the slots hold.
    pub fn bytes(&self) -> u64 {
        self.slots() as u64 * self.slot
    }

    fn slot_slice(&self, s: usize) -> vk::Slice {
        self.mem.slice(s as u64 * self.slot, self.slot)
    }

    /// Records the start of a draft pass: up to `MTP_SWAPS_PER_PASS` experts copied from their host copies into their
    /// slots, then the table's `blk.40` entries (every one after a failed pass), each step behind a barrier, so the
    /// pass reads the new table and whole slots. Nothing is recorded when nothing moves.
    pub fn record(&self, e: &Engine, r: &mut vk::Recorder) -> Pending {
        let st = self.state.borrow();
        let reset = st.reset_pending();
        let moves = st.plan(MTP_SWAPS_PER_PASS);
        if moves.is_empty() && !reset {
            return Pending::default();
        }
        let d = &e.dims;
        let updates = st.updates(
            &moves,
            reset,
            d.n_layer,
            |x| e.host.slot_addr(d.n_layer, x),
            |s| self.slot_slice(s).addr,
        );
        let list: Vec<u8> = updates
            .iter()
            .flat_map(|&(index, loc, addr)| slot_update(index, loc, addr))
            .collect();
        self.updates.write(0, &list);
        if reset {
            // A failed pass may still have been running: its reads of the slots end before any copy.
            r.barrier();
        }
        for m in &moves {
            r.copy(e.host.slot_slice(d.n_layer, m.inn), self.slot_slice(m.slot));
        }
        r.barrier();
        e.k.apply_slot_updates(
            r,
            e.table.addr,
            self.updates.addr(),
            updates.len(),
            (d.n_layer + 1) * d.n_expert,
        );
        r.barrier();
        Pending { moves, updates }
    }

    /// The pass's submit finished: its moves are resident, and the mirror holds its table entries.
    pub fn landed(&self, e: &Engine, p: Pending) {
        if p.updates.is_empty() {
            return;
        }
        self.state.borrow_mut().apply(&p.moves);
        let mut mirror = e.mirror.borrow_mut();
        for &(index, loc, addr) in &p.updates {
            mirror[index as usize] = (addr, loc);
        }
        e.swaps.borrow_mut().stats.mtp_fills += p.moves.len() as u64;
    }

    /// The pass failed at or after its submit: the copies and the table may be anywhere, so nothing is taken as
    /// resident and the next pass rewrites every `blk.40` entry. The mirror holds what that will write.
    pub fn failed(&self, e: &Engine) {
        self.state.borrow_mut().invalidate();
        let d = &e.dims;
        let mut mirror = e.mirror.borrow_mut();
        for x in 0..d.n_expert {
            mirror[d.n_layer * d.n_expert + x] = (e.host.slot_addr(d.n_layer, x), WHERE_HOST);
        }
    }

    /// Counts the routed experts of the pass's draft rows (`n_expert_used` ids a row, as f32), into the scores and the
    /// statistics' `mtp_hits` and `mtp_misses`.
    pub fn count(&self, e: &Engine, ids: &[f32]) {
        let n = e.dims.n_expert_used;
        let (mut hits, mut reads) = (0u64, 0u64);
        let mut st = self.state.borrow_mut();
        for row in ids.chunks(n) {
            let row: Vec<u32> = row.iter().map(|&x| x as u32).collect();
            hits += st.count_row(&row) as u64;
            reads += row.len() as u64;
        }
        let mut sw = e.swaps.borrow_mut();
        sw.stats.mtp_hits += hits;
        sw.stats.mtp_misses += reads - hits;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counted(n_expert: usize, slots: usize, rows: &[&[u32]]) -> MtpSlots {
        let mut s = MtpSlots::new(n_expert, slots);
        for r in rows {
            s.count_row(r);
        }
        s
    }

    #[test]
    fn the_slots_fill_from_the_most_counted_experts_ties_to_the_lower_id() {
        // 16 experts, 4 slots. Counts: 9 ×3, 2 ×2, 12 ×2, 5 ×1, 7 ×1; the rest never routed.
        let s = counted(16, 4, &[&[9, 2, 12], &[9, 2, 12], &[9, 5, 7]]);
        let moves = s.plan(8);
        assert_eq!(
            moves,
            [
                SlotMove {
                    slot: 0,
                    out: None,
                    inn: 9
                },
                SlotMove {
                    slot: 1,
                    out: None,
                    inn: 2
                },
                SlotMove {
                    slot: 2,
                    out: None,
                    inn: 12
                },
                SlotMove {
                    slot: 3,
                    out: None,
                    inn: 5
                },
            ]
        );
        // An expert never routed never fills a slot, even an empty one.
        let s = counted(16, 4, &[&[3]]);
        assert_eq!(
            s.plan(8),
            [SlotMove {
                slot: 0,
                out: None,
                inn: 3
            }]
        );
        assert!(MtpSlots::new(16, 4).plan(8).is_empty());
    }

    #[test]
    fn a_pass_moves_at_most_the_cap_and_the_next_pass_goes_on() {
        let mut s = counted(16, 4, &[&[1, 2, 3, 4, 5]]);
        let first = s.plan(2);
        assert_eq!(first.iter().map(|m| m.inn).collect::<Vec<_>>(), [1, 2]);
        s.apply(&first);
        assert_eq!(
            (s.slot_of(1), s.slot_of(2), s.slot_of(3)),
            (Some(0), Some(1), None)
        );
        let second = s.plan(2);
        assert_eq!(
            second,
            [
                SlotMove {
                    slot: 2,
                    out: None,
                    inn: 3
                },
                SlotMove {
                    slot: 3,
                    out: None,
                    inn: 4
                },
            ]
        );
        s.apply(&second);
        // Full, and 5 is counted no more than any resident: nothing moves.
        assert!(s.plan(8).is_empty());
        assert!(s.plan(0).is_empty());
    }

    #[test]
    fn a_resident_gives_way_only_to_an_expert_counted_at_least_the_margin_more() {
        // 2 slots holding 0 (count 3) and 1 (count 1).
        let mut s = counted(8, 2, &[&[0, 1], &[0], &[0]]);
        let fill = s.plan(8);
        s.apply(&fill);
        assert_eq!((s.slot_of(0), s.slot_of(1)), (Some(0), Some(1)));
        // 5 counted twice: 2 >= 1 + MTP_SWAP_MARGIN fails (2 < 3), so 1 stays.
        s.count_row(&[5]);
        s.count_row(&[5]);
        assert!(s.plan(8).is_empty());
        // A third count: 3 >= 1 + 2, so 5 takes 1's slot.
        s.count_row(&[5]);
        assert_eq!(
            s.plan(8),
            [SlotMove {
                slot: 1,
                out: Some(1),
                inn: 5
            }]
        );
    }

    #[test]
    fn the_least_counted_resident_goes_first_ties_the_higher_id_first() {
        // 3 slots: 4 (count 1), 6 (count 1), 2 (count 5). Newcomers 7 (count 9) and 3 (count 4).
        let mut s = counted(8, 3, &[&[2, 4, 6], &[2], &[2], &[2], &[2]]);
        let fill = s.plan(8);
        s.apply(&fill);
        for _ in 0..9 {
            s.count_row(&[7]);
        }
        for _ in 0..4 {
            s.count_row(&[3]);
        }
        let moves = s.plan(8);
        // 7 replaces 6 (count 1, the higher id of the two ones), then 3 replaces 4. 2 stays.
        assert_eq!(
            moves,
            [
                SlotMove {
                    slot: s.slot_of(6).unwrap(),
                    out: Some(6),
                    inn: 7
                },
                SlotMove {
                    slot: s.slot_of(4).unwrap(),
                    out: Some(4),
                    inn: 3
                },
            ]
        );
        s.apply(&moves);
        assert_eq!(s.slot_of(6), None);
        assert_eq!(s.slot_of(4), None);
        assert!(s.slot_of(7).is_some() && s.slot_of(3).is_some() && s.slot_of(2).is_some());
        // Every slot holds a distinct expert and the two maps agree.
        for e in 0..8 {
            if let Some(slot) = s.slot_of(e) {
                assert_eq!(s.held[slot], Some(e));
            }
        }
    }

    #[test]
    fn a_row_reports_the_reads_the_slots_served_and_skips_ids_past_the_block() {
        let mut s = counted(16, 2, &[&[3, 8]]);
        let fill = s.plan(8);
        s.apply(&fill);
        assert_eq!(s.count_row(&[3, 8, 1, 2]), 2);
        assert_eq!(s.count_row(&[1, 2, 3]), 1);
        assert_eq!(s.count_row(&[16, 400, 1]), 0);
        assert_eq!((s.count(3), s.count(8), s.count(1)), (3, 2, 3));
    }

    #[test]
    fn the_counts_halve_every_4096_rows() {
        let mut s = MtpSlots::new(8, 2);
        for _ in 0..MTP_HALVE_ROWS - 1 {
            s.count_row(&[1]);
        }
        s.count_row(&[2]);
        // Row 4,096 is counted, then every count halves: 4,095 → 2,047, 1 → 0.
        assert_eq!((s.count(1), s.count(2)), (2047, 0));
        s.count_row(&[2]);
        assert_eq!(s.count(2), 1);
    }

    #[test]
    fn a_failed_pass_empties_the_slots_and_the_next_rewrites_every_entry() {
        let mut s = counted(8, 2, &[&[1, 2]]);
        let fill = s.plan(8);
        s.apply(&fill);
        assert!(!s.reset_pending());
        s.invalidate();
        assert!(s.reset_pending());
        assert_eq!((s.slot_of(1), s.slot_of(2)), (None, None));
        // The counts stay: the next pass refills from them.
        let refill = s.plan(8);
        assert_eq!(refill.len(), 2);
        s.apply(&refill);
        assert!(!s.reset_pending(), "a landed pass clears the reset");
    }

    #[test]
    fn the_table_changes_point_leavers_home_and_newcomers_at_their_slots() {
        let s = MtpSlots::new(4, 2);
        let host = |e: usize| 0x1000 + 0x100 * e as u64;
        let slot = |i: usize| 0x9000 + 0x100 * i as u64;
        let moves = [
            SlotMove {
                slot: 0,
                out: None,
                inn: 2,
            },
            SlotMove {
                slot: 1,
                out: Some(3),
                inn: 1,
            },
        ];
        // Layer 40 of 4 experts: entry index 160 + e.
        assert_eq!(
            s.updates(&moves, false, 40, host, slot),
            [
                (162, WHERE_VRAM, 0x9000),
                (163, WHERE_HOST, 0x1300),
                (161, WHERE_VRAM, 0x9100),
            ]
        );
        // A reset writes all four entries once each: HOST but the two coming in.
        assert_eq!(
            s.updates(&moves, true, 40, host, slot),
            [
                (160, WHERE_HOST, 0x1000),
                (161, WHERE_VRAM, 0x9100),
                (162, WHERE_VRAM, 0x9000),
                (163, WHERE_HOST, 0x1300),
            ]
        );
        assert!(s.updates(&[], false, 40, host, slot).is_empty());
    }

    #[test]
    fn the_trunk_keeps_the_mtp_slots_while_it_has_its_ask_or_the_floor() {
        // The 16 GB card's default: about 122 slots with the MTP slots held, past the floor.
        assert!(keep_reserved(122, 256));
        assert!(keep_reserved(MTP_TRUNK_FLOOR, 256));
        // A smaller card: under the floor, the trunk keeps its room.
        assert!(!keep_reserved(MTP_TRUNK_FLOOR - 1, 256));
        assert!(!keep_reserved(0, 256));
        // `--cache-slots 30` (serve-shrunk) got all it asked for: the MTP slots cost it nothing.
        assert!(keep_reserved(30, 30));
        assert!(keep_reserved(0, 0));
        assert!(!keep_reserved(29, 30));
    }

    #[test]
    fn the_leftover_takes_whole_slots_up_to_the_ask() {
        let slot = 1_900_544;
        assert_eq!(leftover_slots(0, slot, 64), 0);
        assert_eq!(leftover_slots(slot - 1, slot, 64), 0);
        assert_eq!(leftover_slots(10 * slot + 5, slot, 64), 10);
        assert_eq!(leftover_slots(u64::MAX, slot, 64), 64);
        assert_eq!(leftover_slots(u64::MAX, slot, 0), 0);
        assert_eq!(
            leftover_slots(u64::MAX, 0, 64),
            0,
            "a zero-byte slot fits none"
        );
    }
}
