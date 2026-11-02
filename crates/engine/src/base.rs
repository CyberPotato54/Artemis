//! The decode base time: a test-only slot table in which every HOST entry of a layer points at
//! one of that layer's resident experts, so a decode step reads every routed expert from VRAM. Its output is
//! meaningless; its per-token time is decode's with every routed expert read from VRAM, plus the table's per-step
//! rebuild and copy. It should stay at 8 ms or less. Reported, not gating (`Options::base_time`,
//! `artemis generate --base-time`).
use crate::Error;
use kernels::dispatch::{slot_entry, ENTRY_BYTES, WHERE_HOST, WHERE_VRAM};

/// The base time's table: written from the engine's mirror each decode step (pinned host memory) and copied into VRAM
/// by the step itself, before the expert kernels read it. Allocated after the cache is fitted, so its VRAM (164 KiB,
/// 167,936 bytes, on the real model) comes out of the margin.
pub(crate) struct BaseTable {
    pub host: vk::Buffer,
    pub vram: vk::Slice,
    _arena: vk::Arena,
}

impl BaseTable {
    pub fn new(dev: &vk::Device, n_layer: usize, n_expert: usize) -> Result<BaseTable, Error> {
        let bytes = ((n_layer + 1) * n_expert * ENTRY_BYTES) as u64;
        let mut arena = vk::Arena::new(vk::MemKind::Vram, bytes, "the base-time slot table");
        let vram = arena.alloc(dev, bytes)?;
        let host = dev.alloc(bytes, vk::MemKind::HostPinned, "the base-time slot table")?;
        Ok(BaseTable {
            host,
            vram,
            _arena: arena,
        })
    }

    /// Writes `base_table(mirror, ..)` into the host buffer, for the step about to be recorded to copy.
    pub fn write(&self, mirror: &[(u64, u32)], n_layer: usize, n_expert: usize) {
        self.host.write(0, &base_table(mirror, n_layer, n_expert));
    }
}

/// The slot table `mirror` holds, (addr, where) per entry, `n_layer + 1` layers of `n_expert`, with each layer's HOST
/// entries remapped: expert e takes the entry of the layer's (e mod r)-th resident expert, residents in expert order, r
/// of them. VRAM entries, every other kind and a layer with no resident expert are left as they are, so a cache of none
/// still reads HOST. Every resident entry is a published slot that no copy writes (the planner points an entry at a
/// slot only once its copy is in, and moves it off before a copy starts), so a remapped read is as safe as the
/// resident expert's own.
pub(crate) fn base_table(mirror: &[(u64, u32)], n_layer: usize, n_expert: usize) -> Vec<u8> {
    assert_eq!(mirror.len(), (n_layer + 1) * n_expert, "a whole slot table");
    let mut t = Vec::with_capacity(mirror.len() * ENTRY_BYTES);
    for layer in mirror.chunks(n_expert) {
        let resident: Vec<(u64, u32)> = layer
            .iter()
            .copied()
            .filter(|e| e.1 == WHERE_VRAM)
            .collect();
        for (e, &(addr, loc)) in layer.iter().enumerate() {
            let (addr, loc) = if loc == WHERE_HOST && !resident.is_empty() {
                resident[e % resident.len()]
            } else {
                (addr, loc)
            };
            t.extend(slot_entry(addr, loc));
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage::entries;
    use kernels::dispatch::WHERE_CPU;

    #[test]
    fn host_entries_take_a_resident_experts_slot_in_their_own_layer() {
        // Layer 0: experts 1 and 4 resident, expert 5 a CPU entry; layer 1: none; the MTP block's CPU entries.
        let mut m = Vec::new();
        for l in 0..2u64 {
            for e in 0..6u64 {
                m.push(if l == 0 && (e == 1 || e == 4) {
                    (0xA000 + 0x100 * e, WHERE_VRAM)
                } else if l == 0 && e == 5 {
                    (0, WHERE_CPU)
                } else {
                    (0x10_0000 + 0x1000 * l + 0x100 * e, WHERE_HOST)
                });
            }
        }
        m.extend([(0, WHERE_CPU); 6]);
        let t = entries(&base_table(&m, 2, 6));
        // Expert e of layer 0 takes resident e mod 2: 1's slot for even e, 4's for odd; the residents keep their own.
        let want0 = [0xA100, 0xA400, 0xA100, 0xA400, 0xA100];
        for (e, &a) in want0.iter().enumerate() {
            let want = if e == 1 || e == 4 { m[e].0 } else { a };
            assert_eq!(t[e], (want, WHERE_VRAM), "layer 0 expert {e}");
        }
        // A CPU entry in a layer that has residents stays as it is: only HOST entries are remapped.
        assert_eq!(t[5], m[5], "layer 0 expert 5 (CPU)");
        // A layer with nothing resident reads HOST as before, and the CPU entries stay CPU.
        assert_eq!(&t[6..12], &m[6..12]);
        assert_eq!(&t[12..], &m[12..]);
        assert!(t[..5].iter().all(|e| e.1 == WHERE_VRAM));
    }
}
