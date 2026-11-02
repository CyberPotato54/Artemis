//! Memory types, buffers with device addresses, and arenas that place allocations first-fit in chunks of at most
//! 1 GiB.
use crate::{
    device::{Device, Inner},
    error::{vk_err, Error},
};
use ash::vk;
use std::sync::Arc;

/// The largest single allocation Artemis makes. This device allows 2 GiB.
pub const MAX_ALLOCATION: u64 = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemKind {
    /// VRAM the CPU can't see.
    Vram,
    /// Pinned system memory the GPU reads over PCIe: host-visible, host-coherent and uncached. Never the ReBAR window.
    HostPinned,
    /// Host-visible, host-coherent, host-cached memory for reading results back.
    Readback,
    /// Host-visible, host-coherent, host-cached memory for the expert copies the CPU reads at DRAM
    /// speed. The GPU reads it exactly as it reads
    /// `HostPinned` (measured 2026-10-13: the same bandwidth, identical outputs).
    HostCached,
    /// AMD device-coherent, device-uncached host memory (`VK_AMD_device_coherent_memory`): the only
    /// memory where a running shader's atomics see the CPU's writes promptly. The CPU
    /// miss path's mailbox.
    AmdUncached,
}

impl MemKind {
    /// (required, forbidden) property flags.
    pub fn flags(self) -> (vk::MemoryPropertyFlags, vk::MemoryPropertyFlags) {
        use vk::MemoryPropertyFlags as F;
        let amd = F::DEVICE_COHERENT_AMD | F::DEVICE_UNCACHED_AMD;
        match self {
            MemKind::Vram => (F::DEVICE_LOCAL, F::HOST_VISIBLE | amd),
            MemKind::HostPinned => (
                F::HOST_VISIBLE | F::HOST_COHERENT,
                F::DEVICE_LOCAL | F::HOST_CACHED | amd,
            ),
            MemKind::Readback => (
                F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED,
                F::DEVICE_LOCAL | amd,
            ),
            MemKind::HostCached => (
                F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED,
                F::DEVICE_LOCAL | amd,
            ),
            MemKind::AmdUncached => (F::HOST_VISIBLE | F::HOST_COHERENT | amd, F::DEVICE_LOCAL),
        }
    }
}

/// The first memory type with every flag `kind` requires and none it forbids.
pub fn select_type(types: &[vk::MemoryPropertyFlags], kind: MemKind) -> Option<u32> {
    let (required, forbidden) = kind.flags();
    types
        .iter()
        .position(|f| f.contains(required) && !f.intersects(forbidden))
        .map(|i| i as u32)
}

/// A buffer bound to its own allocation, with a device address. Host-visible kinds stay mapped. Freed on drop.
pub struct Buffer {
    inner: Arc<Inner>,
    pub(crate) buf: vk::Buffer,
    mem: vk::DeviceMemory,
    size: u64,
    addr: u64,
    ptr: *mut u8,
    kind: MemKind,
}

impl Buffer {
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn addr(&self) -> u64 {
        self.addr
    }

    pub fn kind(&self) -> MemKind {
        self.kind
    }

    /// The mapping's base, null for VRAM. Valid while the buffer lives: the CPU miss path's mailbox
    /// and expert copies are reached through it. `Buffer` stays `!Send`; the pointer
    /// crosses threads as a raw address, never through the buffer.
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Bytes [offset, offset + size). Panics if that runs past the end.
    pub fn slice(&self, offset: u64, size: u64) -> Slice {
        assert!(
            within(offset, size, self.size),
            "slice [{offset}, {}) of a {}-byte buffer",
            offset.saturating_add(size),
            self.size
        );
        Slice {
            buffer: self.buf,
            offset,
            size,
            addr: self.addr + offset,
        }
    }

    pub fn whole(&self) -> Slice {
        self.slice(0, self.size)
    }

    /// Copies `data` in at byte `offset`. Host-visible memory only, and only while no GPU work uses these bytes.
    pub fn write(&self, offset: u64, data: &[u8]) {
        assert!(!self.ptr.is_null(), "writing VRAM from the CPU");
        assert!(
            within(offset, data.len() as u64, self.size),
            "write past the end of a buffer"
        );
        // SAFETY: the range is inside this buffer's mapping, which lives as long as `self`; the GPU isn't using it
        // (the caller's contract).
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(offset as usize), data.len())
        };
    }

    /// `v` as little-endian f32s at byte `offset`.
    pub fn write_f32(&self, offset: u64, v: &[f32]) {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.write(offset, &bytes);
    }

    /// Copies bytes [offset, offset + out.len()) out. Host-visible memory only, after the GPU work writing them ended.
    pub fn read(&self, offset: u64, out: &mut [u8]) {
        assert!(!self.ptr.is_null(), "reading VRAM from the CPU");
        assert!(
            within(offset, out.len() as u64, self.size),
            "read past the end of a buffer"
        );
        // SAFETY: the range is inside this buffer's mapping, which lives as long as `self`; the GPU has finished with
        // it (the caller's contract).
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.add(offset as usize),
                out.as_mut_ptr(),
                out.len(),
            )
        };
    }

    /// `n` f32s from byte `offset`.
    pub fn read_f32(&self, offset: u64, n: usize) -> Vec<f32> {
        let mut b = vec![0u8; 4 * n];
        self.read(offset, &mut b);
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: owners drop a buffer only after the GPU work using it has finished; the device outlives it.
        unsafe {
            self.inner.device.destroy_buffer(self.buf, None);
            self.inner.device.free_memory(self.mem, None);
        }
    }
}

/// Part of a buffer: what a shader needs (`addr`) and what a copy needs (`buffer`, `offset`). Valid while the buffer or
/// arena it came from lives.
///
/// A slice is only an address. The CPU reads and writes host-visible memory through the `Buffer` or `Arena` that owns
/// it, whose borrow keeps the mapping alive, so a slice that outlives its buffer can't reach freed memory from safe
/// code:
///
/// ```compile_fail
/// fn write_through(s: vk::Slice) {
///     s.write(0, &[0u8; 4]);
/// }
/// ```
///
/// The safety boundary: a `Slice` is an address with no lifetime, and the recording API (`Recorder::dispatch`,
/// `copy`, ...) takes such addresses. The engine keeps each buffer alive until the GPU work that reads it has
/// finished (`Engine`'s `Drop` waits for both queues before any field is freed, and `Snapshots`' `Drop` waits for the
/// device while a copy is pending). Recording a command on a buffer that has been freed is a GPU use-after-free that
/// this crate does not prevent.
#[derive(Clone, Copy, Debug)]
pub struct Slice {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub size: u64,
    pub addr: u64,
}

impl Slice {
    /// Bytes [offset, offset + size) of this slice. Panics if that runs past its end.
    pub fn sub(&self, offset: u64, size: u64) -> Slice {
        assert!(
            within(offset, size, self.size),
            "sub-slice [{offset}, {}) of a {}-byte slice",
            offset.saturating_add(size),
            self.size
        );
        Slice {
            buffer: self.buffer,
            offset: self.offset + offset,
            size,
            addr: self.addr + offset,
        }
    }

    /// f32 elements [first, first + n).
    pub fn f32s(&self, first: usize, n: usize) -> Slice {
        self.sub(4 * first as u64, 4 * n as u64)
    }
}

/// Whether bytes [offset, offset + len) lie inside `size` bytes, without wrapping.
fn within(offset: u64, len: u64, size: u64) -> bool {
    offset.checked_add(len).is_some_and(|end| end <= size)
}

impl Device {
    /// A storage buffer of `size` bytes with a device address, in `kind` memory. Refuses more than `MAX_ALLOCATION`,
    /// and anything that would take its heap past the driver's budget. `what` names it in errors.
    pub fn alloc(&self, size: u64, kind: MemKind, what: &str) -> Result<Buffer, Error> {
        let size = size.max(4);
        if size > MAX_ALLOCATION {
            return Err(Error::TooLarge {
                what: what.into(),
                size,
            });
        }
        let (ty, heap) = self.mem_type(kind)?;
        let hb = self.budget()[heap as usize];
        if hb.usage + size > hb.budget {
            return Err(Error::OutOfBudget {
                what: what.into(),
                want: size,
                budget: hb.budget,
                usage: hb.usage,
                host: kind != MemKind::Vram,
            });
        }
        let d = &self.inner.device;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let mut info = vk::BufferCreateInfo::default().size(size).usage(usage);
        info = if self.families.len() > 1 {
            info.sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&self.families)
        } else {
            info.sharing_mode(vk::SharingMode::EXCLUSIVE)
        };
        // SAFETY: the buffer is bound to fresh memory before any use, and freed with it on drop.
        unsafe {
            let buf = d
                .create_buffer(&info, None)
                .map_err(vk_err("creating a buffer"))?;
            let req = d.get_buffer_memory_requirements(buf);
            if req.memory_type_bits & (1 << ty) == 0 {
                d.destroy_buffer(buf, None);
                return Err(Error::NoMemoryType(format!("{kind:?} storage-buffer")));
            }
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(ty)
                .push_next(&mut flags);
            let mem = match d.allocate_memory(&alloc, None) {
                Ok(m) => m,
                Err(e) => {
                    d.destroy_buffer(buf, None);
                    return Err(match e {
                        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                        | vk::Result::ERROR_OUT_OF_HOST_MEMORY => Error::OutOfBudget {
                            what: what.into(),
                            want: size,
                            budget: hb.budget,
                            usage: hb.usage,
                            host: kind != MemKind::Vram
                                || e == vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                        },
                        e => vk_err("allocating memory")(e),
                    });
                }
            };
            if let Err(e) = d.bind_buffer_memory(buf, mem, 0) {
                d.destroy_buffer(buf, None);
                d.free_memory(mem, None);
                return Err(vk_err("binding memory")(e));
            }
            let addr =
                d.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf));
            let ptr = if kind == MemKind::Vram {
                std::ptr::null_mut()
            } else {
                match d.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) {
                    Ok(p) => p as *mut u8,
                    Err(e) => {
                        d.destroy_buffer(buf, None);
                        d.free_memory(mem, None);
                        return Err(vk_err("mapping memory")(e));
                    }
                }
            };
            Ok(Buffer {
                inner: self.inner.clone(),
                buf,
                mem,
                size,
                addr,
                ptr,
                kind,
            })
        }
    }
}

/// Allocations from chunks of at most 1 GiB, each placed first-fit: in the first chunk whose tail has room, else in a
/// new chunk. `Arena::new` opens chunks of a fixed size (or a request's size, if larger). `Arena::planned` takes the
/// sizes its owner will allocate, in order, and opens each chunk at exactly what it will hold, so it takes what its
/// allocations need and no more, as `planned_bytes` predicts.
pub struct Arena {
    kind: MemKind,
    what: String,
    chunks: Vec<Buffer>,
    placer: Placer,
}

impl Arena {
    pub fn new(kind: MemKind, chunk: u64, what: &str) -> Arena {
        Arena {
            kind,
            what: what.into(),
            chunks: Vec::new(),
            placer: Placer::new(chunk.min(MAX_ALLOCATION)),
        }
    }

    /// An arena for `sizes`, allocated in that order. Allocations past the plan, or that differ from it, get chunks of
    /// their own size.
    pub fn planned(kind: MemKind, sizes: &[u64], what: &str) -> Arena {
        Arena {
            kind,
            what: what.into(),
            chunks: Vec::new(),
            placer: Placer::planned(MAX_ALLOCATION, sizes),
        }
    }

    /// `size` bytes, 256-byte aligned.
    pub fn alloc(&mut self, dev: &Device, size: u64) -> Result<Slice, Error> {
        let size = size.max(4);
        let at = self.placer.place(size);
        let s = match at {
            Place::In { chunk, offset } => self.chunks[chunk].slice(offset, size),
            Place::New { size: chunk, .. } => {
                let b = dev.alloc(chunk, self.kind, &self.what)?;
                let s = b.slice(0, size);
                self.chunks.push(b);
                s
            }
        };
        self.placer.commit(at, size);
        Ok(s)
    }

    /// Copies `data` into `s`, a slice this arena allocated from host-visible memory. Panics on a slice from elsewhere.
    pub fn write(&self, s: Slice, data: &[u8]) {
        assert!(data.len() as u64 <= s.size, "write past the end of a slice");
        let b = self
            .chunks
            .iter()
            .find(|b| b.buf == s.buffer)
            .expect("the slice comes from this arena");
        b.write(s.offset, data);
    }

    /// Bytes allocated from the device so far.
    pub fn bytes(&self) -> u64 {
        self.chunks.iter().map(Buffer::size).sum()
    }
}

/// The bytes an `Arena::planned` for `sizes` opens: what its chunks hold, alignment included.
pub fn planned_bytes(sizes: &[u64]) -> u64 {
    plan_chunks(MAX_ALLOCATION, sizes).iter().sum()
}

/// Where an allocation goes in an arena's chunks. `planned` says a new chunk takes the plan's next size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    In { chunk: usize, offset: u64 },
    New { size: u64, planned: bool },
}

/// An arena's placement, apart from the device so CI tests it: first fit across the chunks' tails, else a new chunk:
/// the plan's next size if it has one that fits; past a plan, the request's size; without one, the chunk size (or the
/// request's, if larger).
#[derive(Debug)]
struct Placer {
    chunk: u64,
    planned: bool,
    /// The sizes of the chunks still to open, from `plan_chunks`.
    plan: std::collections::VecDeque<u64>,
    /// Each chunk's size and the end of its last allocation.
    chunks: Vec<(u64, u64)>,
}

impl Placer {
    fn new(chunk: u64) -> Placer {
        Placer {
            chunk,
            planned: false,
            plan: Default::default(),
            chunks: Vec::new(),
        }
    }

    fn planned(chunk: u64, sizes: &[u64]) -> Placer {
        Placer {
            planned: true,
            plan: plan_chunks(chunk, sizes).into(),
            ..Placer::new(chunk)
        }
    }

    fn place(&self, size: u64) -> Place {
        for (chunk, &(cap, used)) in self.chunks.iter().enumerate() {
            let offset = used.next_multiple_of(256);
            if offset + size <= cap {
                return Place::In { chunk, offset };
            }
        }
        match self.plan.front() {
            Some(&next) if next >= size => Place::New {
                size: next,
                planned: true,
            },
            _ => Place::New {
                size: if self.planned {
                    size
                } else {
                    self.chunk.max(size)
                },
                planned: false,
            },
        }
    }

    fn commit(&mut self, at: Place, size: u64) {
        match at {
            Place::In { chunk, offset } => self.chunks[chunk].1 = offset + size,
            Place::New { size: cap, planned } => {
                if planned {
                    self.plan.pop_front();
                }
                self.chunks.push((cap, size));
            }
        }
    }

    #[cfg(test)]
    fn chunk_sizes(&self) -> Vec<u64> {
        self.chunks.iter().map(|c| c.0).collect()
    }
}

/// The chunks an arena needs for `sizes`, allocated in order: placed first-fit in chunks of `chunk` bytes, then each
/// chunk cut to the end of its last allocation. Replaying the same sizes into chunks of exactly these sizes makes the
/// same choices, because each chunk holds the same allocations at the same offsets.
fn plan_chunks(chunk: u64, sizes: &[u64]) -> Vec<u64> {
    let mut p = Placer::new(chunk);
    for &n in sizes {
        let n = n.max(4);
        let at = p.place(n);
        p.commit(at, n);
    }
    p.chunks.iter().map(|c| c.1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::MemoryPropertyFlags as F;

    #[test]
    fn memory_kinds_pick_this_gpus_types_and_never_the_bar_window() {
        // The RX 9070 XT on Windows: VRAM, pinned host, the ReBAR window, cached host, a VRAM
        // variant, and AMD's device-uncached host memory.
        let types = [
            F::DEVICE_LOCAL,
            F::HOST_VISIBLE | F::HOST_COHERENT,
            F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT,
            F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED,
            F::DEVICE_LOCAL | F::DEVICE_COHERENT_AMD | F::DEVICE_UNCACHED_AMD,
            F::HOST_VISIBLE | F::HOST_COHERENT | F::DEVICE_COHERENT_AMD | F::DEVICE_UNCACHED_AMD,
        ];
        assert_eq!(select_type(&types, MemKind::Vram), Some(0));
        assert_eq!(select_type(&types, MemKind::HostPinned), Some(1));
        assert_eq!(select_type(&types, MemKind::Readback), Some(3));
        assert_eq!(select_type(&types, MemKind::HostCached), Some(3));
        assert_eq!(select_type(&types, MemKind::AmdUncached), Some(5));
        assert_eq!(select_type(&types[0..5], MemKind::AmdUncached), None);
        assert_eq!(select_type(&types[2..3], MemKind::HostPinned), None);
    }

    #[test]
    fn budget_errors_give_the_sizes() {
        let e = Error::OutOfBudget {
            what: "the experts of layer 3".into(),
            want: 500 << 20,
            budget: 1000 << 20,
            usage: 800 << 20,
            host: false,
        };
        assert_eq!(
            e.to_string(),
            "the experts of layer 3 needs 500 MiB, but its heap has 200 MiB left of a 1000 MiB budget"
        );
        let e = Error::TooLarge {
            what: "output.weight".into(),
            size: 2 << 30,
        };
        assert!(
            e.to_string()
                .contains("2048 MiB; Artemis allocates at most 1024 MiB"),
            "{e}"
        );
    }

    const MIB: u64 = 1 << 20;

    /// Places `sizes` in order and returns the chunk sizes it opened.
    fn chunks(p: &mut Placer, sizes: &[u64]) -> Vec<u64> {
        for &n in sizes {
            let at = p.place(n);
            p.commit(at, n);
        }
        p.chunk_sizes()
    }

    #[test]
    fn small_allocations_fill_earlier_chunks_tails_first() {
        let mut p = Placer::new(256 * MIB);
        let first = p.place(200 * MIB);
        assert_eq!(
            first,
            Place::New {
                size: 256 * MIB,
                planned: false
            }
        );
        p.commit(first, 200 * MIB);
        // Larger than a chunk: a chunk of its own, exactly its size.
        let big = p.place(417 * MIB);
        assert_eq!(
            big,
            Place::New {
                size: 417 * MIB,
                planned: false
            }
        );
        p.commit(big, 417 * MIB);
        // The first chunk's tail still has room.
        assert_eq!(
            p.place(10 * MIB),
            Place::In {
                chunk: 0,
                offset: 200 * MIB
            }
        );
    }

    /// A session's allocations at `ctx` positions: per layer the K and V caches (attention, 1 KiB per position each) or
    /// the conv and recurrent state (DeltaNet), then the attention partials.
    fn session_like(ctx: u64) -> Vec<u64> {
        let mut sizes = Vec::new();
        for l in 0..40 {
            if l % 4 == 3 {
                sizes.extend([1024 * ctx, 1024 * ctx]);
            } else {
                sizes.extend([98_304, 2 * MIB]);
            }
        }
        sizes.push(4 * 16 * ctx.div_ceil(256) * 258);
        sizes
    }

    /// Replays `sizes` through a planned arena's placer and returns the chunks it opened.
    fn replay(sizes: &[u64]) -> Vec<u64> {
        chunks(&mut Placer::planned(MAX_ALLOCATION, sizes), sizes)
    }

    #[test]
    fn a_planned_arena_opens_its_allocations_and_no_more() {
        // Two 600 MiB allocations: two 600 MiB chunks, not a 1 GiB chunk with a 424 MiB tail and a second chunk.
        assert_eq!(replay(&[600 * MIB, 600 * MIB]), [600 * MIB, 600 * MIB]);
        // A session at 65,536 positions and at 300,000, where three K or V caches fill a chunk short of 1 GiB: the
        // chunks hold exactly the allocations. `planned_bytes` predicts it without a device.
        for ctx in [65_536, 300_000] {
            let sizes = session_like(ctx);
            let opened = replay(&sizes);
            let total: u64 = opened.iter().sum();
            assert_eq!(total, sizes.iter().sum::<u64>(), "ctx {ctx}: {opened:?}");
            assert_eq!(total, planned_bytes(&sizes), "ctx {ctx}");
            assert!(
                opened.iter().all(|&c| c <= MAX_ALLOCATION),
                "ctx {ctx}: {opened:?}"
            );
        }
    }

    #[test]
    fn a_request_larger_than_a_chunk_gets_its_own_size() {
        // The device then refuses it as TooLarge, rather than the arena slicing past a 1 GiB chunk.
        let big = 1536 * MIB;
        assert_eq!(
            Placer::planned(MAX_ALLOCATION, &[big]).place(big),
            Place::New {
                size: big,
                planned: true
            }
        );
        assert_eq!(
            Placer::new(MAX_ALLOCATION).place(big),
            Place::New {
                size: big,
                planned: false
            }
        );
    }

    #[test]
    fn allocations_past_the_plan_get_chunks_of_their_own_size() {
        let mut p = Placer::planned(MAX_ALLOCATION, &[100 * MIB]);
        let at = p.place(100 * MIB);
        p.commit(at, 100 * MIB);
        assert_eq!(
            p.place(5 * MIB),
            Place::New {
                size: 5 * MIB,
                planned: false
            }
        );
    }

    #[test]
    fn bounds_checks_never_wrap() {
        assert!(within(96, 4, 100));
        assert!(!within(97, 4, 100));
        assert!(
            !within(u64::MAX - 1, 4, 100),
            "offset + len wraps past u64::MAX"
        );
    }
}
