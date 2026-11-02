//! The host snapshot pool: 1 GiB chunks of `HostCached` memory that snapshots' KV and state
//! slices are carved from, first-fit with coalescing, and the limit it is given at load. `Snapshots` allocates
//! from it and parks and unparks through it.
use crate::{Engine, Error};

/// A pool chunk is at most this many bytes (the driver's `vk::MAX_ALLOCATION`).
pub const POOL_CHUNK: u64 = 1 << 30;
/// Host memory the pool leaves free: its limit is what the heap and the OS have, less this.
pub const POOL_MARGIN: u64 = 4 << 30;
/// Every pool slice starts and ends on a multiple of this.
pub const POOL_ALIGN: u64 = 256;

const GIB: u64 = 1 << 30;

fn round_up(size: u64) -> u64 {
    size.max(1).next_multiple_of(POOL_ALIGN)
}

/// The free space of the pool's chunks, as sorted `(offset, len)` runs a chunk. Pure: it holds no memory, only
/// arithmetic about it.
#[derive(Clone, Debug, Default)]
pub struct FreeList {
    chunks: Vec<Vec<(u64, u64)>>,
}

impl FreeList {
    /// A new chunk of `size` bytes, all free; its index.
    pub fn add_chunk(&mut self, size: u64) -> usize {
        self.chunks.push(if size > 0 {
            vec![(0, size)]
        } else {
            Vec::new()
        });
        self.chunks.len() - 1
    }

    /// `size` bytes (rounded up to `POOL_ALIGN`) at the lowest offset of the lowest chunk that has room, as
    /// `(chunk, offset)`. A request of 0 bytes takes `POOL_ALIGN`.
    pub fn alloc(&mut self, size: u64) -> Option<(usize, u64)> {
        let size = round_up(size);
        for (c, runs) in self.chunks.iter_mut().enumerate() {
            if let Some(i) = runs.iter().position(|&(_, len)| len >= size) {
                let (offset, len) = runs[i];
                if len == size {
                    runs.remove(i);
                } else {
                    runs[i] = (offset + size, len - size);
                }
                return Some((c, offset));
            }
        }
        None
    }

    /// Gives back `size` bytes (rounded up as `alloc` did) at `offset` of `chunk`, merging with a free neighbour on
    /// either side.
    pub fn free(&mut self, chunk: usize, offset: u64, size: u64) {
        let size = round_up(size);
        let runs = &mut self.chunks[chunk];
        let at = runs.partition_point(|&(o, _)| o < offset);
        assert!(
            at == 0 || runs[at - 1].0 + runs[at - 1].1 <= offset,
            "freed twice"
        );
        assert!(
            at == runs.len() || offset + size <= runs[at].0,
            "freed twice"
        );
        runs.insert(at, (offset, size));
        if at + 1 < runs.len() && runs[at].0 + runs[at].1 == runs[at + 1].0 {
            runs[at].1 += runs[at + 1].1;
            runs.remove(at + 1);
        }
        if at > 0 && runs[at - 1].0 + runs[at - 1].1 == runs[at].0 {
            runs[at - 1].1 += runs[at].1;
            runs.remove(at);
        }
    }

    /// The bytes free across every chunk.
    pub fn free_bytes(&self) -> u64 {
        self.chunks.iter().flatten().map(|&(_, len)| len).sum()
    }
}

/// The pool's size for this load: what was asked, what it was cut to, and whether it was cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolLimit {
    pub bytes: u64,
    pub requested: u64,
    pub cut: bool,
}

/// The pool limit: `min(requested, floor_GiB(available - POOL_MARGIN))`, where `available` is what the host heap's
/// budget leaves or, when known, what the OS has free, whichever is less. Under 1 GiB it is 0 (reuse off).
pub fn pool_limit(
    requested_gib: u32,
    heap: vk::HeapBudget,
    os_available: Option<u64>,
) -> PoolLimit {
    let requested = u64::from(requested_gib) * GIB;
    let available = heap
        .budget
        .saturating_sub(heap.usage)
        .min(os_available.unwrap_or(u64::MAX));
    let room = available.saturating_sub(POOL_MARGIN) / GIB * GIB;
    let bytes = match requested.min(room) {
        b if b < GIB => 0,
        b => b,
    };
    PoolLimit {
        bytes,
        requested,
        cut: bytes < requested,
    }
}

/// The pool limit on a machine with `free_before` bytes of host memory free before the load, where `used_after_load` is
/// the host-cached heap's measured usage after the whole load (the experts, the embedding, prefill's host buffers, the
/// readbacks; with the MTP, `blk.40`'s experts too): `pool_limit` on a heap of that budget and usage, no OS reading
/// (the 32 GB check). At 26 GiB free the pool is `floor_GiB(26 GiB - used - 4 GiB)`.
pub fn pool_on_machine(requested_gib: u32, free_before: u64, used_after_load: u64) -> PoolLimit {
    pool_limit(
        requested_gib,
        vk::HeapBudget {
            size: free_before,
            budget: free_before,
            usage: used_after_load,
        },
        None,
    )
}

/// The OS's available physical memory in bytes: `GlobalMemoryStatusEx`'s `ullAvailPhys` on Windows, `/proc/meminfo`'s
/// `MemAvailable` on Linux, `None` elsewhere (or when it can't be read).
#[cfg(windows)]
pub fn os_available_bytes() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: MEMORYSTATUSEX is plain integers, so all-zero is a valid value; `dwLength` is set as the call requires,
    // and the pointer is to a live local.
    unsafe {
        let mut s: MEMORYSTATUSEX = std::mem::zeroed();
        s.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        (GlobalMemoryStatusEx(&mut s) != 0).then_some(s.ullAvailPhys)
    }
}

/// See the Windows twin.
#[cfg(target_os = "linux")]
pub fn os_available_bytes() -> Option<u64> {
    parse_meminfo(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

/// See the Windows twin.
#[cfg(not(any(windows, target_os = "linux")))]
pub fn os_available_bytes() -> Option<u64> {
    None
}

/// `/proc/meminfo`'s `MemAvailable:` line, kB to bytes.
pub fn parse_meminfo(text: &str) -> Option<u64> {
    let rest = text.lines().find_map(|l| l.strip_prefix("MemAvailable:"))?;
    let kb: u64 = rest.trim().strip_suffix("kB")?.trim().parse().ok()?;
    kb.checked_mul(1024)
}

/// `os_available` plus the bytes of parked snapshots still in RAM: unpark frees each as it copies it into the pool, so
/// they would otherwise cut the new pool for nothing. Saturating; `None` stays `None`.
pub fn add_reclaimable(os_available: Option<u64>, reclaimable: u64) -> Option<u64> {
    os_available.map(|b| b.saturating_add(reclaimable))
}

/// The pool limit for a load, read after `Engine::open` so the experts and the embedding are in the heap's usage.
/// `reclaimable` is the bytes of parked snapshots still in RAM (0 at a first load).
pub fn limit_at_load(e: &Engine, requested_gib: u32, reclaimable: u64) -> Result<PoolLimit, Error> {
    let heap = e.dev.heap_budget(vk::MemKind::HostCached)?;
    Ok(pool_limit(
        requested_gib,
        heap,
        add_reclaimable(os_available_bytes(), reclaimable),
    ))
}

/// The size of the next chunk for a request of `size` bytes: `min(POOL_CHUNK, limit - allocated)`, or `None` when that
/// is 0 or too small to hold the request.
fn next_chunk(limit: u64, allocated: u64, size: u64) -> Option<u64> {
    let chunk = POOL_CHUNK.min(limit.saturating_sub(allocated));
    (chunk > 0 && chunk >= round_up(size)).then_some(chunk)
}

/// A pool allocation: `size` bytes at `offset` of chunk `chunk`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolSlice {
    pub chunk: usize,
    pub offset: u64,
    pub size: u64,
}

/// The pool: `HostCached` chunks of `min(POOL_CHUNK, limit - allocated)` bytes, opened as snapshots need them, never
/// shrunk, freed when the pool drops (after the GPU work that used them has finished).
pub struct Pool {
    limit: u64,
    allocated: u64,
    refused: u64,
    chunks: Vec<vk::Buffer>,
    free: FreeList,
}

impl Pool {
    pub fn new(limit: u64) -> Pool {
        Pool {
            limit,
            allocated: 0,
            refused: 0,
            chunks: Vec::new(),
            free: FreeList::default(),
        }
    }

    /// `size` bytes, from free space or from one more chunk. `Ok(None)` when the limit, a last chunk too small, or a
    /// chunk the driver refused (`OutOfBudget`, `TooLarge`) stops it; any other error is the device's.
    pub fn alloc(&mut self, dev: &vk::Device, size: u64) -> Result<Option<PoolSlice>, Error> {
        let slice = |(chunk, offset)| PoolSlice {
            chunk,
            offset,
            size,
        };
        if let Some(at) = self.free.alloc(size) {
            return Ok(Some(slice(at)));
        }
        let Some(chunk) = next_chunk(self.limit, self.allocated, size) else {
            return Ok(None);
        };
        match dev.alloc(chunk, vk::MemKind::HostCached, "the snapshot pool") {
            Ok(b) => {
                self.chunks.push(b);
                self.allocated += chunk;
                let c = self.free.add_chunk(chunk);
                assert_eq!(c + 1, self.chunks.len());
                Ok(self.free.alloc(size).map(slice))
            }
            Err(vk::Error::OutOfBudget { .. } | vk::Error::TooLarge { .. }) => {
                self.refused += 1;
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn free(&mut self, s: PoolSlice) {
        self.free.free(s.chunk, s.offset, s.size);
    }

    /// The slice for GPU copies.
    pub fn slice(&self, s: PoolSlice) -> vk::Slice {
        self.chunks[s.chunk].slice(s.offset, s.size)
    }

    /// Reads `out.len()` bytes from the start of `s`.
    pub fn read(&self, s: PoolSlice, out: &mut [u8]) {
        assert!(out.len() as u64 <= s.size, "reading past a pool slice");
        self.chunks[s.chunk].read(s.offset, out);
    }

    /// Writes `data` at the start of `s`.
    pub fn write(&self, s: PoolSlice, data: &[u8]) {
        assert!(data.len() as u64 <= s.size, "writing past a pool slice");
        self.chunks[s.chunk].write(s.offset, data);
    }

    /// The bytes of chunks opened so far.
    pub fn allocated(&self) -> u64 {
        self.allocated
    }

    /// The chunk attempts the driver has refused so far (each later miss tries again, so one refusal can count many
    /// times).
    pub fn refused(&self) -> u64 {
        self.refused
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heap(budget: u64, usage: u64) -> vk::HeapBudget {
        vk::HeapBudget {
            size: 48 * GIB,
            budget,
            usage,
        }
    }

    #[test]
    fn the_limit_is_cut_to_what_the_heap_and_the_os_leave() {
        // The test PC after load: about 19.2 GiB used of a 48 GiB pinned heap, about 35 GiB free in the OS.
        assert_eq!(
            pool_limit(8, heap(48 * GIB, 19 * GIB), Some(35 * GIB)),
            PoolLimit {
                bytes: 8 * GIB,
                requested: 8 * GIB,
                cut: false
            }
        );
        assert_eq!(
            pool_limit(8, heap(48 * GIB, 19 * GIB), Some(10 * GIB + GIB / 2)),
            PoolLimit {
                bytes: 6 * GIB,
                requested: 8 * GIB,
                cut: true
            }
        ); // 10.5 - 4 = 6.5, whole GiB: 6
        assert_eq!(pool_limit(8, heap(24 * GIB, 19 * GIB), None).bytes, GIB); // 5 - 4 = 1
        assert_eq!(pool_limit(8, heap(24 * GIB, 20 * GIB + 1), None).bytes, 0); // under 1 GiB: reuse off
        assert_eq!(
            pool_limit(0, heap(48 * GIB, 0), None),
            PoolLimit {
                bytes: 0,
                requested: 0,
                cut: false
            }
        );
    }

    #[test]
    fn first_fit_reuses_freed_space_and_coalesces() {
        let mut f = FreeList::default();
        let c = f.add_chunk(1 << 20);
        let a = f.alloc(1000).unwrap();
        let b = f.alloc(4096).unwrap();
        assert_eq!((a, b), ((c, 0), (c, 1024))); // 1,000 rounds up to 1,024
        f.free(c, 0, 1000);
        assert_eq!(f.alloc(512), Some((c, 0)));
        f.free(c, 0, 512);
        f.free(c, 1024, 4096);
        assert_eq!(f.free_bytes(), 1 << 20);
        assert_eq!(f.alloc(1 << 20), Some((c, 0))); // all one block again
        assert_eq!(f.alloc(1), None);
    }

    #[test]
    fn free_list_spans_chunks_and_merges_a_gap_between_neighbours() {
        let mut f = FreeList::default();
        let (c0, c1) = (f.add_chunk(1024), f.add_chunk(1024));
        let a = f.alloc(512).unwrap();
        let b = f.alloc(256).unwrap();
        let c = f.alloc(256).unwrap();
        assert_eq!((a, b, c), ((c0, 0), (c0, 512), (c0, 768)));
        assert_eq!(f.alloc(1024), Some((c1, 0))); // chunk 0 is full, chunk 1 whole
        f.free(c0, 0, 512);
        f.free(c0, 768, 256);
        assert_eq!(f.alloc(768), None); // the middle block still holds 512..768
        f.free(c0, 512, 256);
        assert_eq!(f.alloc(1024), Some((c0, 0)));
    }

    #[test]
    fn the_next_chunk_is_whole_then_short_then_none() {
        assert_eq!(next_chunk(0, 0, 1000), None); // a pool of 0
        assert_eq!(next_chunk(3 * GIB, 0, 1000), Some(GIB));
        let limit = 2 * GIB + (GIB / 2); // not a whole number of chunks
        assert_eq!(next_chunk(limit, 2 * GIB, 1000), Some(GIB / 2)); // the short last chunk
        assert_eq!(next_chunk(limit, 2 * GIB, GIB / 2), Some(GIB / 2));
        assert_eq!(next_chunk(limit, 2 * GIB, GIB / 2 + 1), None); // too big for what is left
        assert_eq!(next_chunk(limit, limit, 1), None); // the limit is reached
        assert_eq!(next_chunk(4 * GIB, 0, GIB + 1), None); // bigger than any chunk
    }

    #[test]
    fn meminfo_gives_memavailable_in_bytes() {
        let text = "MemTotal:       65536000 kB\nMemFree:  100 kB\nMemAvailable:   36700160 kB\n";
        assert_eq!(parse_meminfo(text), Some(36_700_160 * 1024));
        assert_eq!(parse_meminfo("MemTotal: 1 kB\n"), None);
    }

    #[test]
    fn os_available_bytes_reads_something_here() {
        // Windows and Linux (CI) both have a reading; elsewhere it is None and there is nothing to read.
        if cfg!(any(windows, target_os = "linux")) {
            assert!(os_available_bytes().is_some_and(|n| n > 0));
        }
    }

    #[test]
    fn parked_bytes_count_as_available() {
        assert_eq!(add_reclaimable(Some(10 * GIB), 6 * GIB), Some(16 * GIB));
        assert_eq!(add_reclaimable(None, 6 * GIB), None);
        assert_eq!(add_reclaimable(Some(u64::MAX), 6 * GIB), Some(u64::MAX));
    }

    #[test]
    fn the_32gb_pool_is_2gib_with_and_without_mtp() {
        // A 32 GB machine has about 26 GiB free before the load. Today's load leaves 19.2 GiB and
        // prefill's 66 MiB in the heap; the MTP adds blk.40's 464 MiB of experts.
        let used = 19 * GIB + GIB / 5 + (66 << 20);
        let off = pool_on_machine(8, 26 * GIB, used);
        let on = pool_on_machine(8, 26 * GIB, used + (464 << 20));
        assert_eq!((off.bytes, on.bytes), (2 * GIB, 2 * GIB));
        assert!(off.cut && on.cut, "8 GiB asked, so both are cut");
        // A load that leaves 20.2 GiB in the heap is the 1 GiB pool; one that leaves more than 22 GiB, none.
        let more = pool_on_machine(8, 26 * GIB, 20 * GIB + GIB / 5);
        assert_eq!(more.bytes, GIB);
        assert_eq!(pool_on_machine(8, 26 * GIB, 22 * GIB + 1).bytes, 0);
        // A big machine gets what it asked for.
        assert_eq!(pool_on_machine(8, 48 * GIB, used).bytes, 8 * GIB);
    }
}
