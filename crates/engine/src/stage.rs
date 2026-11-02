//! Prefill's staging: two VRAM buffers, each with room for a layer's non-resident experts at the
//! largest slot size, filled on the transfer queue a layer ahead of the compute that reads them, and the prefill table:
//! the slot table with every staged expert pointed at its place in its layer's buffer. The grouped expert kernels read
//! the prefill table in a staged chunk and the slot table otherwise; a staged expert runs the same code on the same
//! bytes as a cached or HOST one, at another address, so results don't depend on staging.
use crate::cache::staging_plan;
use kernels::dispatch::{slot_entry, ENTRY_BYTES, WHERE_HOST, WHERE_VRAM};

/// The staging area's VRAM, as `fit_cache`'s rounds allocate it (and free it again when a round is refused): the two
/// buffers and the prefill table. `StagingVram::new` is its only constructor; the pinned host half comes after the fit.
pub(crate) struct StagingVram {
    arena: vk::Arena,
    bufs: [vk::Slice; 2],
    stride: u64,
    cap: usize,
    table: vk::Slice,
}

/// The staging buffers and the prefill table.
pub(crate) struct Staging {
    /// Keeps the buffers' and the table's VRAM alive as long as the engine: between staged chunks the buffers also hold
    /// the cache's lent rows.
    _arena: vk::Arena,
    /// Layer l stages into `bufs[l % 2]`. Between staged chunks the same memory backs the cache's lent rows.
    pub bufs: [vk::Slice; 2],
    /// Bytes from one staged expert to the next: the largest slot.
    pub stride: u64,
    /// Experts a buffer holds: `n_expert - slots`.
    pub cap: usize,
    /// The prefill table in VRAM, which the grouped expert kernels read in a staged chunk.
    pub table: vk::Slice,
    /// Where the CPU writes the prefill table for layer 0's submit to copy into `table`.
    pub table_host: vk::Buffer,
    /// The VRAM the area took, as `staging_bytes` predicts.
    pub bytes: u64,
}

impl StagingVram {
    /// The staging area's VRAM for a cache of `slots` per layer (`staging_plan`), or None when every expert is cached.
    /// Only VRAM, so a fit round's allocations are the VRAM ones the fake heap models; `StagingVram::finish` adds the
    /// pinned host buffer once the fit has settled.
    pub fn new(
        dev: &vk::Device,
        slot_sizes: &[u64],
        n_expert: usize,
        slots: usize,
    ) -> Result<Option<StagingVram>, vk::Error> {
        let sizes = staging_plan(slot_sizes, n_expert, slots);
        let &[one, _, table_bytes] = sizes.as_slice() else {
            return Ok(None);
        };
        let mut arena = vk::Arena::planned(vk::MemKind::Vram, &sizes, "the staging buffers");
        let a = arena.alloc(dev, one)?;
        let b = arena.alloc(dev, one)?;
        let table = arena.alloc(dev, table_bytes)?;
        Ok(Some(StagingVram {
            arena,
            bufs: [a, b],
            stride: *slot_sizes.iter().max().expect("a layer"),
            cap: n_expert - slots,
            table,
        }))
    }

    /// Adds the prefill table's pinned host buffer, allocated once after the fit (its size doesn't depend on the slots,
    /// and a pinned-host refusal must not feed the host heap's budget into the VRAM sizing).
    pub fn finish(self, dev: &vk::Device) -> Result<Staging, vk::Error> {
        let table_host = dev.alloc(
            self.table.size,
            vk::MemKind::HostPinned,
            "the prefill table's upload",
        )?;
        Ok(Staging {
            bytes: self.arena.bytes(),
            _arena: self.arena,
            bufs: self.bufs,
            stride: self.stride,
            cap: self.cap,
            table: self.table,
            table_host,
        })
    }
}

/// The prefill table and the copies of one staged chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StagePlan {
    /// (n_layer + 1) × n_expert entries in the slot table's format.
    pub table: Vec<u8>,
    /// Per main layer, its staged experts in expert order: (the host copy's address their entry held, their index in
    /// the layer's staging buffer).
    pub copies: Vec<Vec<(u64, usize)>>,
    /// Per main layer's entry, `layer × n_expert + expert`: whether the chunk reads it from a staging buffer.
    pub staged: Vec<bool>,
}

/// A staged chunk's prefill table, from `mirror`, the slot table it runs with (`(addr, where)` per entry, (n_layer + 1)
/// × n_expert of them). In each main layer `l`, the HOST entries that point at one of the layer's own host expert
/// copies (`pinned(l, addr)`) take, in expert order, the next place in buffer `l % 2`, `bases[l % 2] + i × stride`, as
/// VRAM entries, until the buffer's `cap` is reached. Every other entry is kept: cached experts (VRAM), HOST entries past
/// the capacity or pointing anywhere else (read in place, as a shrunk cache's extras are), CPU and unusable
/// entries (refused by the kernels as before), and the MTP block's.
pub(crate) fn plan(
    mirror: &[(u64, u32)],
    n_layer: usize,
    n_expert: usize,
    bases: [u64; 2],
    stride: u64,
    cap: usize,
    pinned: impl Fn(usize, u64) -> bool,
) -> StagePlan {
    assert_eq!(
        mirror.len(),
        (n_layer + 1) * n_expert,
        "a slot table of {} entries for {n_layer} layers of {n_expert} experts",
        mirror.len()
    );
    let mut table = Vec::with_capacity(mirror.len() * ENTRY_BYTES);
    let mut copies = vec![Vec::new(); n_layer];
    let mut staged = vec![false; n_layer * n_expert];
    for (i, &(addr, loc)) in mirror.iter().enumerate() {
        let l = i / n_expert;
        let (a, w) = if l < n_layer && loc == WHERE_HOST && copies[l].len() < cap && pinned(l, addr)
        {
            let at = copies[l].len();
            copies[l].push((addr, at));
            staged[i] = true;
            (bases[l % 2] + at as u64 * stride, WHERE_VRAM)
        } else {
            (addr, loc)
        };
        table.extend(slot_entry(a, w));
    }
    StagePlan {
        table,
        copies,
        staged,
    }
}

/// Where the staging buffers, `buf_bytes` each, hold the cache rows they lend between staged prefill chunks:
/// `[row][layer]` as (buffer, offset). Slots go row by row, layer by layer, each whole in
/// buffer 0 while it fits there and then in buffer 1; only complete rows count, at most `max`. On the real model at 124
/// slots (two buffers of 132 experts at 2,039,808 bytes) that is 7 rows: the 7 slots per layer staging takes.
pub(crate) fn lend_plan(slot_sizes: &[u64], buf_bytes: u64, max: usize) -> Vec<Vec<(usize, u64)>> {
    let (mut rows, mut buf, mut at) = (Vec::new(), 0usize, 0u64);
    'rows: while rows.len() < max {
        let mut row = Vec::with_capacity(slot_sizes.len());
        for &size in slot_sizes {
            if at + size > buf_bytes {
                (buf, at) = (buf + 1, 0);
                if buf > 1 || size > buf_bytes {
                    break 'rows;
                }
            }
            row.push((buf, at));
            at += size;
        }
        rows.push(row);
    }
    rows
}

/// One slot-table entry as (addr, where). The one parser of the entry layout (`kernels::dispatch::slot_entry`'s), shared
/// by `entries` and `Engine::slot_entry`.
pub(crate) fn entry_at(b: &[u8; ENTRY_BYTES]) -> (u64, u32) {
    (
        u64::from_le_bytes(b[..8].try_into().expect("8 bytes")),
        u32::from_le_bytes(b[8..12].try_into().expect("4 bytes")),
    )
}

/// A slot table's entries as (addr, where).
pub(crate) fn entries(table: &[u8]) -> Vec<(u64, u32)> {
    table
        .as_chunks::<ENTRY_BYTES>()
        .0
        .iter()
        .map(entry_at)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernels::dispatch::{slot_entry, WHERE_CPU, WHERE_HOST, WHERE_VRAM};

    /// The real model's slot sizes: Q6_K down in layers 34, 38 and 39.
    fn real_sizes() -> Vec<u64> {
        (0..40)
            .map(|l| {
                if [34, 38, 39].contains(&l) {
                    2_039_808
                } else {
                    1_900_544
                }
            })
            .collect()
    }

    #[test]
    fn the_staging_buffers_lend_whole_rows_of_slots() {
        // At 124 slots each buffer holds 132 experts at the largest slot size, 269,254,656 bytes: 7 rows of 76,439,552
        // fit across the two (535,076,864 of 538,509,312), row 3 split between them; an 8th doesn't.
        let s = real_sizes();
        let buf = 132 * 2_039_808;
        let plan = lend_plan(&s, buf, usize::MAX);
        assert_eq!(plan.len(), 7);
        assert_eq!(plan[0][0], (0, 0));
        assert_eq!(plan[1][0], (0, 76_439_552));
        // Row 3 starts in buffer 0 and moves to buffer 1 at the first slot that no longer fits there.
        let split = plan[3].iter().position(|&(b, _)| b == 1).unwrap();
        assert!(split > 0, "{:?}", plan[3]);
        assert_eq!(plan[3][split], (1, 0));
        let (b, at) = plan[3][split - 1];
        assert!(b == 0 && at + s[split - 1] <= buf && at + s[split - 1] + s[split] > buf);
        // Every slot lies whole in its buffer, 256-aligned, and no two overlap.
        let mut taken: Vec<(usize, u64, u64)> = Vec::new();
        for row in &plan {
            for (l, &(b, at)) in row.iter().enumerate() {
                assert!(at + s[l] <= buf);
                assert!(at.is_multiple_of(256));
                assert!(taken
                    .iter()
                    .all(|&(tb, ta, te)| tb != b || at + s[l] <= ta || at >= te));
                taken.push((b, at, at + s[l]));
            }
        }
        // `max` caps it (a `--cache-slots` ceiling); no cache at all lends 13 rows; one expert per buffer lends none.
        assert_eq!(lend_plan(&s, buf, 3).len(), 3);
        assert_eq!(lend_plan(&s, 256 * 2_039_808, usize::MAX).len(), 13);
        assert!(lend_plan(&s, 2_039_808, usize::MAX).is_empty());
    }

    #[test]
    fn entries_read_a_slot_table_back() {
        let m = [
            (0xC000, WHERE_VRAM),
            (0x10_0100, WHERE_HOST),
            (0, WHERE_CPU),
            (u64::MAX, 7),
        ];
        let table: Vec<u8> = m.iter().flat_map(|&(a, w)| slot_entry(a, w)).collect();
        assert_eq!(entries(&table), m);
        assert!(entries(&[]).is_empty());
    }

    /// Layer l's host copy of expert e, in a model of eight experts.
    fn host(l: usize, e: usize) -> u64 {
        0x10_0000 + l as u64 * 0x1_0000 + e as u64 * 0x100
    }

    fn pinned(l: usize, a: u64) -> bool {
        (0..8).any(|e| host(l, e) == a)
    }

    fn entry(t: &[u8], i: usize) -> (u64, u32) {
        let b = &t[16 * i..16 * i + 16];
        (
            u64::from_le_bytes(b[..8].try_into().unwrap()),
            u32::from_le_bytes(b[8..12].try_into().unwrap()),
        )
    }

    /// Two layers of eight experts with 0 and 3 cached, the rest HOST at their host copies, then the MTP block's
    /// eight CPU entries.
    fn mirror() -> Vec<(u64, u32)> {
        let mut m = Vec::new();
        for l in 0..2 {
            for e in 0..8 {
                m.push(if e == 0 || e == 3 {
                    (0xC000 + e as u64, WHERE_VRAM)
                } else {
                    (host(l, e), WHERE_HOST)
                });
            }
        }
        m.extend([(0, WHERE_CPU); 8]);
        m
    }

    #[test]
    fn host_experts_point_at_their_places_in_their_layers_buffer() {
        // Each HOST expert takes the next place, in expert order, in buffer l % 2; cached entries and the MTP block's
        // are copied as they are.
        let p = plan(&mirror(), 2, 8, [0xA_0000, 0xB_0000], 0x1000, 6, pinned);
        assert_eq!(
            p.copies[0],
            [
                (host(0, 1), 0),
                (host(0, 2), 1),
                (host(0, 4), 2),
                (host(0, 5), 3),
                (host(0, 6), 4),
                (host(0, 7), 5)
            ]
        );
        assert_eq!(p.copies[1][0], (host(1, 1), 0));
        assert_eq!(
            entry(&p.table, 2),
            (0xA_1000, WHERE_VRAM),
            "layer 0's expert 2: place 1 of buffer 0"
        );
        assert_eq!(
            entry(&p.table, 8 + 7),
            (0xB_5000, WHERE_VRAM),
            "layer 1's expert 7: place 5 of buffer 1"
        );
        assert_eq!(
            entry(&p.table, 3),
            (0xC003, WHERE_VRAM),
            "a cached expert keeps its slot"
        );
        for e in 0..8 {
            assert_eq!(
                entry(&p.table, 16 + e),
                (0, WHERE_CPU),
                "the MTP block's entries"
            );
        }
        assert_eq!(p.staged.iter().filter(|&&s| s).count(), 12);
        assert!(!p.staged[0] && p.staged[1] && !p.staged[3] && p.staged[15]);
        assert_eq!(p.table.len(), 24 * 16);
    }

    #[test]
    fn experts_past_the_capacity_stay_host() {
        // A swap in flight makes a layer's HOST entries outnumber its buffer's places (5 here, for six).
        // The first five in expert order are staged; the last keeps its HOST entry and is read in place.
        let p = plan(&mirror(), 2, 8, [0xA_0000, 0xB_0000], 0x1000, 5, pinned);
        assert_eq!(p.copies[0].len(), 5);
        assert_eq!(p.copies[0][4], (host(0, 6), 4));
        assert_eq!(entry(&p.table, 7), (host(0, 7), WHERE_HOST));
        assert!(p.staged[6] && !p.staged[7]);
    }

    #[test]
    fn entries_staging_cant_copy_are_left_alone() {
        // A CPU entry, a HOST entry with no address, and a HOST entry pointing at another layer's copy (as a test can
        // write behind the planner) keep their entries, so the kernels refuse or read them exactly as without staging,
        // and the rest still take places in order. With no capacity the prefill table is the slot table.
        let mut m = mirror();
        m[1] = (0, WHERE_CPU);
        m[2] = (0, WHERE_HOST);
        m[4] = (host(1, 4), WHERE_HOST);
        let p = plan(&m, 2, 8, [0xA_0000, 0xB_0000], 0x1000, 6, pinned);
        for i in [1, 2, 4] {
            assert_eq!(entry(&p.table, i), m[i], "entry {i}");
            assert!(!p.staged[i]);
        }
        assert_eq!(
            p.copies[0],
            [(host(0, 5), 0), (host(0, 6), 1), (host(0, 7), 2)]
        );
        let none = plan(&m, 2, 8, [0xA_0000, 0xB_0000], 0x1000, 0, pinned);
        let table: Vec<u8> = m.iter().flat_map(|&(a, w)| slot_entry(a, w)).collect();
        assert_eq!(none.table, table);
        assert!(none.copies.iter().all(Vec::is_empty) && none.staged.iter().all(|&s| !s));
    }
}
