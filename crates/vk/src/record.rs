//! Pipelines and command recording: dispatches, barriers, copies and fills, then one submit that signals the
//! queue's timeline semaphore and waits for it, after any explicit wait (`Recorder::wait`). Copies into the expert cache
//! and prefill's staging copies go on the transfer queue without waiting
//! (`copy_async`, `stage_async`); every submit waits only for what the host has seen the other queue finish, or what it
//! asks for, so such a copy can overlap compute work. The engine watches the timelines to know when they have landed.
use crate::{
    device::{Device, Inner, Queue},
    error::{vk_err, Error},
    memory::{MemKind, Slice},
};
use ash::vk;
use std::sync::Arc;

/// How long `submit` waits. Every submit is sized to finish well under 0.5 s, so reaching
/// this means the GPU is stuck.
pub(crate) const WAIT_NS: u64 = 10_000_000_000;

/// Vulkan's guaranteed `maxComputeWorkGroupCount` per dimension: a dispatch may use up to this many workgroups.
pub const MAX_WORKGROUPS: u32 = 65_535;

/// Timestamps a profiled command buffer holds: the start, then one after each command it times.
pub const PROFILE_STAMPS: u32 = 8192;

/// The tag of commands recorded before any `Recorder::tag`.
pub const UNTAGGED: u32 = u32::MAX;

/// One command's GPU time in a profiled submit (`Device::submit_profiled`): from the end of the command before it, or
/// the command buffer's start, to its own end. Commands with no barrier between them can overlap, so one of them may
/// show the time of both and the other none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timing {
    /// The pipeline's name, or "copy" or "fill".
    pub name: &'static str,
    /// The tag set when it was recorded (`Recorder::tag`).
    pub tag: u32,
    pub ns: u64,
}

/// A profiled recorder's query pool and the label of each timestamp after the first.
struct Stamps {
    pool: vk::QueryPool,
    labels: Vec<(&'static str, u32)>,
    tag: u32,
}

/// Each command's time in ns from a profiled submit's raw timestamps (the start, then one after each command). The
/// counter wraps at `mask`; `period` is ns per tick.
pub(crate) fn durations(raw: &[u64], mask: u64, period: f64) -> Vec<u64> {
    raw.windows(2)
        .map(|w| {
            let ticks = w[1].wrapping_sub(w[0]) & mask;
            // A command that finished before the one recorded ahead of it reads as 0, not as a wrap of the counter.
            if ticks > mask / 2 {
                0
            } else {
                (ticks as f64 * period).round() as u64
            }
        })
        .collect()
}

/// Upload staging: copies go through pinned memory in pieces this size.
const STAGE: u64 = 64 << 20;

/// One of the device's two queues, each with its own timeline semaphore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueKind {
    Compute,
    Transfer,
}

/// A compute pipeline whose shader takes everything through push constants of `push_size` bytes.
pub struct Pipeline {
    inner: Arc<Inner>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    push_size: u32,
    name: &'static str,
}

impl Pipeline {
    pub fn name(&self) -> &'static str {
        self.name
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        // SAFETY: no submitted work uses the pipeline once its owner drops it.
        unsafe {
            self.inner.device.destroy_pipeline(self.pipeline, None);
            self.inner.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// A type `Recorder::dispatch` may push to a shader: its bytes are the push-constant block, every one initialized.
///
/// # Safety
/// The type is `#[repr(C)]`, has no padding (the size is the sum of the fields' sizes) and holds only integers and
/// floats, so reading all `size_of::<Self>()` bytes reads no uninitialized memory. `kernels::push_constants!` writes
/// the impl together with a compile-time padding check.
pub unsafe trait PushConstants: Copy {}

fn bytes_of<T: PushConstants>(t: &T) -> &[u8] {
    // SAFETY: `PushConstants` promises a padding-free `#[repr(C)]` plain-data type, so every byte is initialized.
    unsafe { std::slice::from_raw_parts(t as *const T as *const u8, std::mem::size_of::<T>()) }
}

/// Commands for one submit on the compute queue. Dropped without `Device::submit`, it is discarded.
pub struct Recorder<'d> {
    dev: &'d Device,
    cmd: Option<vk::CommandBuffer>,
    names: Vec<&'static str>,
    stamps: Option<Stamps>,
    /// The transfer value this submit waits for beyond what every submit waits for (`wait`); 0 for none.
    wait_transfer: u64,
    /// When a traced recorder was made (`trace.rs`); None while the trace is off and in a profiled recorder.
    traced: Option<std::time::Instant>,
}

impl Recorder<'_> {
    fn cmd(&self) -> vk::CommandBuffer {
        self.cmd.expect("the recorder was already submitted")
    }

    /// Labels the commands recorded from here on, in a profiled recorder's timings (an engine passes its layer). With
    /// `ARTEMIS_MARKERS=layer` (`VK_AMD_buffer_marker`), a marker here counts the dispatches that finished before this tag, so device loss can
    /// name the layer in flight.
    pub fn tag(&mut self, tag: u32) {
        if let Some(s) = self.stamps.as_mut() {
            s.tag = tag;
        }
        if !self.dev.marker_every_dispatch {
            self.mark();
        }
    }

    /// A marker, once every earlier command has finished, counting the dispatches recorded so far.
    fn mark(&self) {
        if let Some((m, buf)) = &self.dev.marker {
            // SAFETY: cmd is recording; the marker buffer lives as long as the device.
            unsafe {
                m.cmd_write_buffer_marker(
                    self.cmd(),
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    buf.buf,
                    0,
                    self.names.len() as u32,
                )
            };
        }
    }

    /// This submit waits until queue `q` reaches `value` before any of its commands run: for work still in flight
    /// there that the submit reads, such as a staged layer's staging copy. A compute recorder
    /// waits only for the transfer queue, and only for a value already submitted. Several calls keep the highest; 0
    /// adds nothing to the wait every submit has for what the host has seen finish.
    pub fn wait(&mut self, q: QueueKind, value: u64) {
        assert_eq!(
            q,
            QueueKind::Transfer,
            "a compute submit waits only for the transfer queue"
        );
        assert!(
            value <= self.dev.transfer.value.get(),
            "a wait for transfer value {value}, which was never submitted (the last is {})",
            self.dev.transfer.value.get()
        );
        self.wait_transfer = self.wait_transfer.max(value);
    }

    /// In a profiled recorder, a timestamp once every earlier command has finished, labelled `name`.
    fn stamp(&mut self, name: &'static str) {
        let cmd = self.cmd();
        let Some(s) = self.stamps.as_mut() else {
            return;
        };
        let i = s.labels.len() as u32 + 1;
        assert!(
            i < PROFILE_STAMPS,
            "a profiled command buffer times at most {} commands",
            PROFILE_STAMPS - 1
        );
        // SAFETY: cmd is recording; the pool's queries were reset at its start and query i is unwritten.
        unsafe {
            self.dev.inner.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                s.pool,
                i,
            )
        };
        s.labels.push((name, s.tag));
    }

    /// Binds `p`, pushes `push` and dispatches `groups` workgroups. With `VK_AMD_buffer_marker` and
    /// `ARTEMIS_MARKERS=dispatch`, a marker after each dispatch counts the dispatches that finished, so device loss can
    /// name the one in flight.
    pub fn dispatch<T: PushConstants>(&mut self, p: &Pipeline, push: &T, groups: [u32; 3]) {
        assert_eq!(
            std::mem::size_of::<T>(),
            p.push_size as usize,
            "{}: push constants of the wrong size",
            p.name
        );
        assert!(
            groups.iter().all(|&g| g <= MAX_WORKGROUPS),
            "{}: {groups:?} workgroups; Vulkan guarantees only {MAX_WORKGROUPS} per dimension",
            p.name
        );
        let (d, cmd) = (&self.dev.inner.device, self.cmd());
        // SAFETY: cmd is recording; push is plain #[repr(C)] data of the pipeline's push-constant size.
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.pipeline);
            d.cmd_push_constants(
                cmd,
                p.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(push),
            );
            d.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
        }
        self.names.push(p.name);
        if self.dev.marker_every_dispatch {
            self.mark();
        }
        self.stamp(p.name);
    }

    /// Earlier compute and transfer writes finish, and are visible, before later compute and transfer work starts.
    pub fn barrier(&mut self) {
        let both = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        let mb = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            )];
        // SAFETY: cmd is recording.
        unsafe {
            self.dev.inner.device.cmd_pipeline_barrier(
                self.cmd(),
                both,
                both,
                vk::DependencyFlags::empty(),
                &mb,
                &[],
                &[],
            )
        };
    }

    /// Earlier writes are visible to the CPU once the submit has finished.
    pub fn host_barrier(&mut self) {
        let mb = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        // SAFETY: cmd is recording.
        unsafe {
            self.dev.inner.device.cmd_pipeline_barrier(
                self.cmd(),
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &mb,
                &[],
                &[],
            )
        };
    }

    /// Copies `src` into `dst`, which must be the same size.
    pub fn copy(&mut self, src: Slice, dst: Slice) {
        assert_eq!(src.size, dst.size, "copy between slices of different sizes");
        let region = [vk::BufferCopy::default()
            .src_offset(src.offset)
            .dst_offset(dst.offset)
            .size(src.size)];
        // SAFETY: cmd is recording; both ranges lie inside their buffers (Slice checks that).
        unsafe {
            self.dev
                .inner
                .device
                .cmd_copy_buffer(self.cmd(), src.buffer, dst.buffer, &region)
        };
        self.stamp("copy");
    }

    /// Copies `n` rows of `row_bytes` each in one command: row `i` from `src` at byte `i * src_stride` to `dst` at byte
    /// `i * dst_stride`. Each slice must hold all its rows, and no destination row may overlap a source row of the same
    /// buffer (`vkCmdCopyBuffer`'s rule).
    pub fn copy_rows(
        &mut self,
        src: Slice,
        src_stride: u64,
        dst: Slice,
        dst_stride: u64,
        row_bytes: u64,
        n: usize,
    ) {
        if n == 0 {
            return;
        }
        let last = n as u64 - 1;
        assert!(
            last * src_stride + row_bytes <= src.size && last * dst_stride + row_bytes <= dst.size,
            "copy_rows: {n} rows of {row_bytes} bytes at strides {src_stride} and {dst_stride} past a {}- or {}-byte slice",
            src.size,
            dst.size
        );
        if src.buffer == dst.buffer {
            // vkCmdCopyBuffer: no source row may overlap a destination row of the same buffer.
            let spans = |s: Slice, stride: u64| -> Vec<(u64, u64)> {
                (0..n as u64)
                    .map(|i| (s.offset + i * stride, s.offset + i * stride + row_bytes))
                    .collect()
            };
            let (a, b) = (spans(src, src_stride), spans(dst, dst_stride));
            // Both lists ascend (strides are not negative), so one merge finds an overlap.
            let (mut i, mut j) = (0, 0);
            while i < a.len() && j < b.len() {
                assert!(
                    a[i].1 <= b[j].0 || b[j].1 <= a[i].0,
                    "copy_rows: source row {i} overlaps destination row {j} in one buffer"
                );
                if a[i].1 <= b[j].1 {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        }
        let regions: Vec<vk::BufferCopy> = (0..n as u64)
            .map(|i| {
                vk::BufferCopy::default()
                    .src_offset(src.offset + i * src_stride)
                    .dst_offset(dst.offset + i * dst_stride)
                    .size(row_bytes)
            })
            .collect();
        // SAFETY: cmd is recording; every range lies inside its slice, checked above.
        unsafe {
            self.dev
                .inner
                .device
                .cmd_copy_buffer(self.cmd(), src.buffer, dst.buffer, &regions)
        };
        self.stamp("copy");
    }

    /// Fills `dst` with the 32-bit word `value`. `dst.size` must be a multiple of 4.
    pub fn fill(&mut self, dst: Slice, value: u32) {
        // SAFETY: cmd is recording; the range lies inside the buffer.
        unsafe {
            self.dev.inner.device.cmd_fill_buffer(
                self.cmd(),
                dst.buffer,
                dst.offset,
                dst.size,
                value,
            )
        };
        self.stamp("fill");
    }

    pub fn dispatches(&self) -> usize {
        self.names.len()
    }
}

impl Drop for Recorder<'_> {
    fn drop(&mut self) {
        if let Some(cmd) = self.cmd.take() {
            // SAFETY: the command buffer was never submitted, so neither it nor its query pool is in use.
            unsafe {
                let d = &self.dev.inner.device;
                d.free_command_buffers(self.dev.compute.pool, &[cmd]);
                if let Some(s) = self.stamps.take() {
                    d.destroy_query_pool(s.pool, None);
                }
            }
        }
    }
}

impl Device {
    pub fn pipeline(
        &self,
        name: &'static str,
        spv: &[u8],
        push_size: u32,
    ) -> Result<Pipeline, Error> {
        self.pipeline_sg(name, spv, push_size, None)
    }

    /// `pipeline`, with `subgroup` Some(n) building it with a required subgroup size of n and full subgroups, for a
    /// kernel whose reductions run on subgroup shuffles. The device opens only where 32 can be required
    /// (`Support::subgroup_size_32`).
    pub fn pipeline_sg(
        &self,
        name: &'static str,
        spv: &[u8],
        push_size: u32,
        subgroup: Option<u32>,
    ) -> Result<Pipeline, Error> {
        self.pipeline_spec(name, spv, push_size, subgroup, &[])
    }

    /// `pipeline_sg`, with the shader's specialization constants set: `spec[i]` is the u32 value of constant_id i.
    /// Constants not listed keep the shader's defaults.
    pub fn pipeline_spec(
        &self,
        name: &'static str,
        spv: &[u8],
        push_size: u32,
        subgroup: Option<u32>,
        spec: &[u32],
    ) -> Result<Pipeline, Error> {
        let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))
            .ok()
            .filter(|c| !c.is_empty())
            .ok_or(Error::BadSpirv(name))?;
        let d = &self.inner.device;
        // SAFETY: the module is only needed until the pipeline exists; create infos outlive their calls.
        unsafe {
            let module = d
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                .map_err(vk_err("creating a shader module"))?;
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(push_size)];
            let layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges),
                    None,
                )
                .map_err(vk_err("creating a pipeline layout"))?;
            let mut req = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(subgroup.unwrap_or(0));
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            if subgroup.is_some() {
                stage = stage
                    .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
                    .push_next(&mut req);
            }
            let entries: Vec<vk::SpecializationMapEntry> = (0..spec.len() as u32)
                .map(|i| {
                    vk::SpecializationMapEntry::default()
                        .constant_id(i)
                        .offset(4 * i)
                        .size(4)
                })
                .collect();
            let data: Vec<u8> = spec.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let spec_info = vk::SpecializationInfo::default()
                .map_entries(&entries)
                .data(&data);
            if !spec.is_empty() {
                stage = stage.specialization_info(&spec_info);
            }
            let info = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)];
            let made = d.create_compute_pipelines(vk::PipelineCache::null(), &info, None);
            d.destroy_shader_module(module, None);
            let pipeline = match made {
                Ok(p) => p[0],
                Err((_, e)) => {
                    d.destroy_pipeline_layout(layout, None);
                    return Err(vk_err("creating a compute pipeline")(e));
                }
            };
            Ok(Pipeline {
                inner: self.inner.clone(),
                pipeline,
                layout,
                push_size,
                name,
            })
        }
    }

    /// A command buffer on the compute queue, recording. Under the submit trace (`trace.rs`) it starts with the trace's
    /// first timestamp.
    pub fn record(&self) -> Result<Recorder<'_>, Error> {
        let mut r = self.record_untraced()?;
        if self.trace.is_some() {
            r.traced = self.trace_begin(r.cmd());
        }
        Ok(r)
    }

    fn record_untraced(&self) -> Result<Recorder<'_>, Error> {
        if let Some((_, buf)) = &self.marker {
            buf.write(0, &0u32.to_le_bytes());
        }
        let cmd = self.begin(&self.compute)?;
        Ok(Recorder {
            dev: self,
            cmd: Some(cmd),
            names: Vec::new(),
            stamps: None,
            wait_transfer: 0,
            traced: None,
        })
    }

    /// A command buffer on the compute queue, recording, that times each command it records (`submit_profiled`).
    /// Refused when the compute queue has no timestamps.
    pub fn record_profiled(&self) -> Result<Recorder<'_>, Error> {
        if self.timestamps.is_none() {
            return Err(Error::NoTimestamps);
        }
        let d = &self.inner.device;
        // SAFETY: the create info outlives the call. The pool is destroyed with the recorder, or after its submit.
        let pool = unsafe {
            d.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(PROFILE_STAMPS),
                None,
            )
        }
        .map_err(vk_err("creating a timestamp query pool"))?;
        let mut r = match self.record_untraced() {
            Ok(r) => r,
            Err(e) => {
                // SAFETY: nothing uses the pool yet.
                unsafe { d.destroy_query_pool(pool, None) };
                return Err(e);
            }
        };
        let cmd = r.cmd();
        // SAFETY: cmd is recording; the pool holds PROFILE_STAMPS queries.
        unsafe {
            d.cmd_reset_query_pool(cmd, pool, 0, PROFILE_STAMPS);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, pool, 0);
        }
        r.stamps = Some(Stamps {
            pool,
            labels: Vec::new(),
            tag: UNTAGGED,
        });
        Ok(r)
    }

    /// Submits `r`, signals the compute timeline, and waits for it. On device loss, names the dispatch in flight. A
    /// profiled recorder's timings are discarded (`submit_profiled` returns them).
    pub fn submit(&self, mut r: Recorder<'_>) -> Result<(), Error> {
        let cmd = r.cmd.take().expect("the recorder was already submitted");
        let stamps = r.stamps.take();
        if let Some(record) = r.traced {
            self.trace_end(cmd);
            let out = self.finish_traced(cmd, r.wait_transfer, record);
            return match out {
                Err(Error::DeviceLost { .. }) => Err(Error::DeviceLost {
                    in_flight: self.in_flight(&r.names),
                }),
                Ok(h) => self.trace_log(h),
                Err(e) => Err(e),
            };
        }
        let out = match self.finish(&self.compute, cmd, r.wait_transfer) {
            Err(Error::DeviceLost { .. }) => Err(Error::DeviceLost {
                in_flight: self.in_flight(&r.names),
            }),
            other => other,
        };
        if let (Ok(()), Some(s)) = (&out, stamps) {
            // SAFETY: the submit has finished, so nothing uses the pool. After an error the command buffer may still
            // be pending, so the pool is leaked with it, as `finish` leaks the buffer.
            unsafe { self.inner.device.destroy_query_pool(s.pool, None) };
        }
        out
    }

    /// Submits `r` as `submit` does, and returns each command's GPU time in recording order; empty for a recorder that
    /// isn't profiled.
    pub fn submit_profiled(&self, mut r: Recorder<'_>) -> Result<Vec<Timing>, Error> {
        let Some(s) = r.stamps.take() else {
            self.submit(r)?;
            return Ok(Vec::new());
        };
        // With its stamps taken, `submit` leaves the pool alone; after an error the pool is leaked with the buffer.
        self.submit(r)?;
        let mut raw = vec![0u64; s.labels.len() + 1];
        let d = &self.inner.device;
        // SAFETY: the submit has finished, so every query it wrote has landed; the pool is destroyed once read.
        let read = unsafe {
            let read = d.get_query_pool_results(
                s.pool,
                0,
                &mut raw,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            );
            d.destroy_query_pool(s.pool, None);
            read
        };
        read.map_err(vk_err("reading timestamps"))?;
        let (period, mask) = self
            .timestamps
            .expect("a profiled recorder exists only with timestamps");
        self.last_span.set(Some((
            stamp_ns(raw[0], mask, period),
            stamp_ns(raw[raw.len() - 1], mask, period),
        )));
        Ok(durations(&raw, mask, period)
            .into_iter()
            .zip(s.labels)
            .map(|(ns, (name, tag))| Timing { name, tag, ns })
            .collect())
    }

    /// Copies `data` into `dst` through pinned staging on the transfer queue. Host-visible memory is written directly
    /// through its `Buffer` instead.
    pub fn upload(&self, dst: Slice, data: &[u8]) -> Result<(), Error> {
        assert!(
            data.len() as u64 <= dst.size,
            "upload larger than its slice"
        );
        let staging = self.alloc(
            STAGE.min(data.len() as u64),
            MemKind::HostPinned,
            "upload staging",
        )?;
        for (i, chunk) in data.chunks(STAGE as usize).enumerate() {
            staging.write(0, chunk);
            let cmd = self.begin(&self.transfer)?;
            let region = [vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(dst.offset + i as u64 * STAGE)
                .size(chunk.len() as u64)];
            // SAFETY: cmd is recording; both ranges lie inside their buffers.
            unsafe {
                self.inner
                    .device
                    .cmd_copy_buffer(cmd, staging.buf, dst.buffer, &region)
            };
            self.finish(&self.transfer, cmd, 0)?;
        }
        Ok(())
    }

    /// The bytes of `src`, copied by the GPU into host-cached memory and read back. For tests and one-off reads.
    pub fn download(&self, src: Slice) -> Result<Vec<u8>, Error> {
        let rb = self.alloc(src.size, MemKind::Readback, "download")?;
        let mut r = self.record()?;
        r.copy(src, rb.slice(0, src.size));
        r.host_barrier();
        self.submit(r)?;
        let mut out = vec![0u8; src.size as usize];
        rb.read(0, &mut out);
        Ok(out)
    }

    fn queue(&self, q: QueueKind) -> &Queue {
        match q {
            QueueKind::Compute => &self.compute,
            QueueKind::Transfer => &self.transfer,
        }
    }

    /// The value the queue's last submit signals; 0 before any.
    pub fn submitted(&self, q: QueueKind) -> u64 {
        self.queue(q).value.get()
    }

    /// The value the queue's timeline semaphore has reached. Frees the command buffers of async copies that finished.
    pub fn completed(&self, q: QueueKind) -> Result<u64, Error> {
        // SAFETY: the semaphore belongs to this device.
        let v = unsafe {
            self.inner
                .device
                .get_semaphore_counter_value(self.queue(q).timeline)
        }
        .map_err(vk_err("reading a timeline semaphore"))?;
        let queue = self.queue(q);
        queue.synced.set(queue.synced.get().max(v));
        if q == QueueKind::Transfer {
            self.free_copies(v);
        }
        Ok(v)
    }

    /// Copies `src` into `dst`, which must be the same size, on the transfer queue, and returns the transfer timeline
    /// value it signals, without waiting. Like every submit, it waits for what the host has seen the compute queue
    /// finish: the planner sends a swap's copy only once the steps that read its slot have finished. Its writes are
    /// visible to a compute submit once the host has seen this value reached (`completed` or `wait`). With `poison`, the
    /// copy first fills `dst` with that 32-bit word, so a read of `dst` mid-copy sees it.
    /// `dst`'s offset and size must then be multiples of 4.
    pub fn copy_async(&self, src: Slice, dst: Slice, poison: Option<u32>) -> Result<u64, Error> {
        assert_eq!(src.size, dst.size, "copy between slices of different sizes");
        assert!(
            poison.is_none() || (dst.offset.is_multiple_of(4) && dst.size.is_multiple_of(4)),
            "a poisoned copy fills whole 32-bit words: its offset and size must be multiples of 4"
        );
        let cmd = self.begin(&self.transfer)?;
        let d = &self.inner.device;
        let region = [vk::BufferCopy::default()
            .src_offset(src.offset)
            .dst_offset(dst.offset)
            .size(src.size)];
        // SAFETY: cmd is recording; both ranges lie inside their buffers (Slice checks that).
        unsafe {
            if let Some(word) = poison {
                d.cmd_fill_buffer(cmd, dst.buffer, dst.offset, dst.size, word);
                transfer_barrier(d, cmd);
            }
            d.cmd_copy_buffer(cmd, src.buffer, dst.buffer, &region);
        }
        let v = match self.signal(&self.transfer, cmd, 0) {
            Ok(v) => v,
            Err(e) => {
                // SAFETY: `signal` failed before the submit took the buffer (ending it, or the submit itself), so it
                // isn't pending and nothing else will free it: it isn't in `pending` yet.
                unsafe { d.free_command_buffers(self.transfer.pool, &[cmd]) };
                return Err(e);
            }
        };
        self.pending.borrow_mut().push((cmd, v));
        Ok(v)
    }

    /// A staging copy: every `(src, dst)` pair, each the same size, copied in one transfer
    /// submit that first waits for compute value `after` (an earlier value costs nothing; every submit also waits for
    /// what the host has seen the compute queue finish) and lands after every earlier submit on the transfer queue.
    /// Returns the transfer value it signals, without waiting; a compute submit that reads the copies waits for it
    /// (`Recorder::wait`). Consecutive pairs between the same two buffers go in one copy command. A timed copy
    /// (`StageOpts::timed`) holds its timestamps' query pool until `copy_spans` reads it, or the device drops, so a
    /// caller that times copies must call `copy_spans`.
    pub fn stage_async(
        &self,
        regions: &[(Slice, Slice)],
        after: u64,
        opts: StageOpts,
    ) -> Result<u64, Error> {
        for (s, d) in regions {
            assert_eq!(s.size, d.size, "copy between slices of different sizes");
            assert!(
                s.size > 0,
                "a staging copy region of 0 bytes (Vulkan refuses an empty copy region)"
            );
            assert!(
                opts.poison.is_none() || (d.offset.is_multiple_of(4) && d.size.is_multiple_of(4)),
                "a poisoned copy fills whole 32-bit words: its offset and size must be multiples of 4"
            );
        }
        if let Some(s) = opts.delay {
            assert!(
                s.size > 0 && s.offset.is_multiple_of(4) && s.size.is_multiple_of(4),
                "the delay fills whole 32-bit words: its offset and size must be multiples of 4, and its size not 0"
            );
        }
        assert!(
            after <= self.compute.value.get(),
            "a staging copy waits for compute value {after}, which was never submitted (the last is {})",
            self.compute.value.get()
        );
        let d = &self.inner.device;
        let pool = match (opts.timed, self.transfer_stamps) {
            (true, Some(_)) => {
                // SAFETY: the create info outlives the call. Host query reset is on whenever there are transfer
                // timestamps, and resets the pool before the command buffer writes it.
                let p = unsafe {
                    d.create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2),
                        None,
                    )
                }
                .map_err(vk_err("creating a copy's timestamp pool"))?;
                // SAFETY: nothing uses the new pool yet.
                unsafe { d.reset_query_pool(p, 0, 2) };
                Some(p)
            }
            _ => None,
        };
        let drop_pool = |pool: Option<vk::QueryPool>| {
            if let Some(p) = pool {
                // SAFETY: no submitted work uses the pool.
                unsafe { d.destroy_query_pool(p, None) };
            }
        };
        let cmd = match self.begin(&self.transfer) {
            Ok(c) => c,
            Err(e) => {
                drop_pool(pool);
                return Err(e);
            }
        };
        // SAFETY: cmd is recording; every range lies inside its buffer (Slice checks that).
        unsafe {
            // Earlier submits on this queue land first: a failed step's staging copies, still in flight, and the next
            // step's copies into the same buffer otherwise have no order between their writes.
            transfer_barrier(d, cmd);
            if let Some(p) = pool {
                d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, p, 0);
            }
            if let Some(s) = opts.delay {
                d.cmd_fill_buffer(cmd, s.buffer, s.offset, s.size, 0);
                transfer_barrier(d, cmd);
            }
            if let Some(word) = opts.poison {
                for (_, dst) in regions {
                    d.cmd_fill_buffer(cmd, dst.buffer, dst.offset, dst.size, word);
                }
                transfer_barrier(d, cmd);
            }
            for run in copy_runs(regions) {
                d.cmd_copy_buffer(cmd, run.src, run.dst, &run.regions);
            }
            if let Some(p) = pool {
                d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, p, 1);
            }
        }
        let v = match self.signal(&self.transfer, cmd, after) {
            Ok(v) => v,
            Err(e) => {
                // SAFETY: `signal` failed before the submit took the buffer, so neither it nor the pool is in use.
                unsafe { d.free_command_buffers(self.transfer.pool, &[cmd]) };
                drop_pool(pool);
                return Err(e);
            }
        };
        self.pending.borrow_mut().push((cmd, v));
        if let Some(p) = pool {
            self.stamped.borrow_mut().push((v, p));
        }
        Ok(v)
    }

    /// The spans of the timed staging copies the host has seen finish (through `completed` or `wait` on the transfer
    /// queue), oldest first. Each is reported once; copies still in flight wait for a later call. Every timed copy's
    /// query pool is held until this reads it (or the device drops), so a caller that times copies must call it. A
    /// failed read returns the error and drops this call's other spans (their pools are destroyed all the same): a
    /// failed read of a finished copy's timestamps means the device is gone.
    pub fn copy_spans(&self) -> Result<Vec<CopySpan>, Error> {
        let Some((period, mask)) = self.transfer_stamps else {
            return Ok(Vec::new());
        };
        let seen = self.transfer.synced.get();
        let d = &self.inner.device;
        let done: Vec<(u64, vk::QueryPool)> = {
            let mut stamped = self.stamped.borrow_mut();
            let (done, left) = stamped.drain(..).partition(|&(v, _)| v <= seen);
            *stamped = left;
            done
        };
        let mut out = Vec::with_capacity(done.len());
        let mut failed = None;
        for (value, pool) in done {
            let mut raw = [0u64; 2];
            // SAFETY: the copy has finished, so both timestamps have landed; the pool is destroyed once read.
            let read = unsafe {
                let r = d.get_query_pool_results(
                    pool,
                    0,
                    &mut raw,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                );
                d.destroy_query_pool(pool, None);
                r
            };
            match read {
                Ok(()) => out.push(CopySpan {
                    value,
                    start_ns: stamp_ns(raw[0], mask, period),
                    end_ns: stamp_ns(raw[1], mask, period),
                }),
                Err(e) => failed = failed.or(Some(e)),
            }
        }
        match failed {
            Some(e) => Err(vk_err("reading a copy's timestamps")(e)),
            None => Ok(out),
        }
    }

    /// The last profiled compute submit's first and last timestamps (when it started, and when its last command
    /// finished), in device ns on the clock `copy_spans` reads. None before any profiled submit. It masks with the
    /// compute family's valid timestamp bits and `copy_spans` with the transfer family's, so the two are compared only
    /// within a step: a counter wrap between them would misorder them.
    pub fn last_span(&self) -> Option<(u64, u64)> {
        self.last_span.get()
    }

    /// Waits until the queue's timeline reaches `value`, for at most the usual limit (`Error::Timeout`).
    pub fn wait(&self, q: QueueKind, value: u64) -> Result<(), Error> {
        self.wait_for(self.queue(q), value)?;
        if q == QueueKind::Transfer {
            self.free_copies(value);
        }
        Ok(())
    }

    fn wait_for(&self, q: &Queue, value: u64) -> Result<(), Error> {
        let (sems, vals) = ([q.timeline], [value]);
        // SAFETY: the semaphore belongs to this device; the arrays outlive the call.
        let waited = unsafe {
            self.inner.device.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&sems)
                    .values(&vals),
                WAIT_NS,
            )
        };
        match waited {
            Ok(()) => {
                q.synced.set(q.synced.get().max(value));
                Ok(())
            }
            Err(vk::Result::TIMEOUT) => Err(Error::Timeout),
            Err(e) => Err(vk_err("waiting for the GPU")(e)),
        }
    }

    /// Frees the command buffers of async copies whose value the transfer timeline has reached.
    fn free_copies(&self, reached: u64) {
        let mut pending = self.pending.borrow_mut();
        let done: Vec<vk::CommandBuffer> = pending
            .iter()
            .filter(|(_, v)| *v <= reached)
            .map(|(c, _)| *c)
            .collect();
        if done.is_empty() {
            return;
        }
        pending.retain(|(_, v)| *v > reached);
        // SAFETY: the transfer timeline has passed these command buffers' submits, so none is still pending.
        unsafe {
            self.inner
                .device
                .free_command_buffers(self.transfer.pool, &done)
        };
    }

    fn begin(&self, q: &Queue) -> Result<vk::CommandBuffer, Error> {
        let d = &self.inner.device;
        // SAFETY: the pool belongs to this device; the buffer is freed after its submit finishes.
        unsafe {
            let cmd = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(q.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(vk_err("allocating a command buffer"))?[0];
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(vk_err("beginning a command buffer"))?;
            Ok(cmd)
        }
    }

    /// Ends, submits and waits for `cmd` on `q`, then frees it. The submit also waits for the other queue's `extra`
    /// (`Recorder::wait`). A command buffer still pending after an error is leaked, never freed.
    fn finish(&self, q: &Queue, cmd: vk::CommandBuffer, extra: u64) -> Result<(), Error> {
        let v = self.signal(q, cmd, extra)?;
        self.wait_for(q, v)?;
        // SAFETY: the submit has finished.
        unsafe { self.inner.device.free_command_buffers(q.pool, &[cmd]) };
        if std::ptr::eq(q, &self.transfer) {
            self.free_copies(v);
        }
        Ok(())
    }

    /// `finish` on the compute queue for a traced recorder made at `record`, reading the host's clock around the
    /// submit and after the wait.
    fn finish_traced(
        &self,
        cmd: vk::CommandBuffer,
        extra: u64,
        record: std::time::Instant,
    ) -> Result<crate::trace::HostTimes, Error> {
        let q = &self.compute;
        let submit = std::time::Instant::now();
        let v = self.signal(q, cmd, extra)?;
        let submitted = std::time::Instant::now();
        self.wait_for(q, v)?;
        let woke = std::time::Instant::now();
        // SAFETY: the submit has finished.
        unsafe { self.inner.device.free_command_buffers(q.pool, &[cmd]) };
        Ok(crate::trace::HostTimes {
            record,
            submit,
            submitted,
            woke,
        })
    }

    /// Ends `cmd` and submits it on `q`, signalling the queue's next timeline value, which it returns. The submit first
    /// waits for what the host has seen the other queue finish, or for the other queue's `extra` if that is higher
    /// (`cross_wait`).
    fn signal(&self, q: &Queue, cmd: vk::CommandBuffer, extra: u64) -> Result<u64, Error> {
        let d = &self.inner.device;
        let v = q.value.get() + 1;
        let (cmds, sems, vals) = ([cmd], [q.timeline], [v]);
        let other = if std::ptr::eq(q, &self.compute) {
            &self.transfer
        } else {
            &self.compute
        };
        let (wait_sems, wait_vals): (Vec<_>, Vec<_>) = cross_wait(other, extra).into_iter().unzip();
        let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; wait_sems.len()];
        // SAFETY: cmd was recorded on this device; the arrays outlive the calls.
        unsafe {
            d.end_command_buffer(cmd)
                .map_err(vk_err("ending a command buffer"))?;
            let mut tl = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_vals)
                .signal_semaphore_values(&vals);
            let info = vk::SubmitInfo::default()
                .command_buffers(&cmds)
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&wait_stages)
                .signal_semaphores(&sems)
                .push_next(&mut tl);
            d.queue_submit(q.queue, &[info], vk::Fence::null())
                .map_err(vk_err("submitting work"))?;
        }
        q.value.set(v);
        Ok(v)
    }

    /// The first dispatch that hadn't finished, from the last marker written: exact with a marker after every
    /// dispatch, else the first of the layer in flight.
    fn in_flight(&self, names: &[&'static str]) -> Option<String> {
        let (_, buf) = self.marker.as_ref()?;
        let mut b = [0u8; 4];
        buf.read(0, &mut b);
        let done = u32::from_le_bytes(b) as usize;
        let at = if self.marker_every_dispatch {
            ""
        } else {
            " or a later one in its layer"
        };
        Some(match names.get(done) {
            Some(n) => format!("dispatch {} of {} ({n}){at}", done + 1, names.len()),
            None => format!("the copies after the last of {} dispatches", names.len()),
        })
    }
}

/// The wait a submit needs for the other queue's work: the highest value the host has seen that queue finish
/// or `extra` if that is higher. Work the host has seen finish costs nothing to wait for, but
/// only a semaphore wait makes its writes visible to this queue: an upload on the transfer queue, read by the next
/// dispatch on the compute queue; a swap's copy, published only once the planner has polled it finished. Work still in
/// flight is waited for only when asked (`extra`): a staged layer waits for its staging copy, and a staging copy
/// names the compute submit that last read its buffer. After a host wait times out, `synced` doesn't advance, so later
/// submits on the other queue no longer order after the stuck work; a timeout is already fatal (the device rules stop
/// on it).
pub(crate) fn cross_wait(other: &Queue, extra: u64) -> Option<(vk::Semaphore, u64)> {
    let v = other.synced.get().max(extra);
    (v > 0).then_some((other.timeline, v))
}

/// A timestamp's device nanoseconds: its counter's valid bits, `mask`, times `period` ns per tick.
pub(crate) fn stamp_ns(raw: u64, mask: u64, period: f64) -> u64 {
    ((raw & mask) as f64 * period).round() as u64
}

/// One copy command of a staging copy: regions between one source and one destination buffer.
pub(crate) struct CopyRun {
    pub src: vk::Buffer,
    pub dst: vk::Buffer,
    pub regions: Vec<vk::BufferCopy>,
}

/// The copy commands for `regions`: a run of consecutive pairs between the same two buffers is one command, with a
/// region each, in order.
pub(crate) fn copy_runs(regions: &[(Slice, Slice)]) -> Vec<CopyRun> {
    let mut runs: Vec<CopyRun> = Vec::new();
    for (s, d) in regions {
        let r = vk::BufferCopy::default()
            .src_offset(s.offset)
            .dst_offset(d.offset)
            .size(s.size);
        match runs.last_mut() {
            Some(run) if run.src == s.buffer && run.dst == d.buffer => run.regions.push(r),
            _ => runs.push(CopyRun {
                src: s.buffer,
                dst: d.buffer,
                regions: vec![r],
            }),
        }
    }
    runs
}

/// Transfer writes recorded before it land before the transfer work recorded after it.
///
/// # Safety
/// `cmd` is a command buffer of `d`'s, recording.
unsafe fn transfer_barrier(d: &ash::Device, cmd: vk::CommandBuffer) {
    let mb = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)];
    d.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &mb,
        &[],
        &[],
    );
}

/// What `Device::stage_async` does besides its copies.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageOpts {
    /// Tests only: first fill every destination region with this 32-bit word, so a read that comes before the copy
    /// lands sees it (the poison, for staging).
    pub poison: Option<u32>,
    /// Tests only: first fill this slice with zeros, so the copies land late and a missing wait reads stale bytes.
    pub delay: Option<Slice>,
    /// A timestamp before and after the copy, for `copy_spans`; ignored without transfer timestamps
    /// (`Device::can_time_copies`). The timestamps' query pool is held until `copy_spans` reads it (or the device
    /// drops), so a caller that times copies must call `copy_spans`.
    pub timed: bool,
}

/// One timed staging copy (`stage_async` with `timed`): the transfer value it signalled and when it ran, in device
/// nanoseconds on the clock `Device::last_span` reads (`a_timed_copy_reports_its_span_on_the_compute_queues_clock`
/// checks that the two are one clock). The span is masked with the transfer family's valid timestamp bits and
/// `last_span` with the compute family's, so the two are compared only within a step: a counter wrap between them
/// would misorder them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopySpan {
    pub value: u64,
    pub start_ns: u64,
    pub end_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Queue;
    use ash::vk::Handle;
    use std::cell::Cell;

    fn queue(timeline: u64, value: u64, synced: u64) -> Queue {
        Queue {
            queue: vk::Queue::null(),
            pool: vk::CommandPool::null(),
            timeline: vk::Semaphore::from_raw(timeline),
            value: Cell::new(value),
            synced: Cell::new(synced),
        }
    }

    #[test]
    fn a_submit_waits_for_the_higher_of_what_was_seen_and_what_it_asked() {
        // Five copies submitted, three seen finished: a compute submit waits for the third, which costs nothing and
        // makes its writes visible, and not for the two in flight, unless it asks for one, as a
        // staged layer asks for its staging copy.
        let t = vk::Semaphore::from_raw(7);
        assert_eq!(cross_wait(&queue(7, 5, 3), 0), Some((t, 3)));
        assert_eq!(
            cross_wait(&queue(7, 5, 3), 5),
            Some((t, 5)),
            "a copy still in flight"
        );
        assert_eq!(
            cross_wait(&queue(7, 5, 3), 2),
            Some((t, 3)),
            "already covered"
        );
        assert_eq!(
            cross_wait(&queue(7, 5, 0), 0),
            None,
            "nothing seen finished, nothing asked"
        );
        assert_eq!(cross_wait(&queue(7, 5, 0), 4), Some((t, 4)));
    }

    #[test]
    fn consecutive_regions_between_the_same_buffers_share_a_copy_command() {
        let s = |b: u64, offset: u64, size: u64| Slice {
            buffer: vk::Buffer::from_raw(b),
            offset,
            size,
            addr: 0,
        };
        let runs = copy_runs(&[
            (s(1, 0, 16), s(9, 64, 16)),
            (s(1, 32, 8), s(9, 0, 8)),
            (s(2, 0, 4), s(9, 128, 4)),
            (s(1, 48, 4), s(9, 200, 4)),
        ]);
        let regions = |r: &CopyRun| -> Vec<(u64, u64, u64)> {
            r.regions
                .iter()
                .map(|c| (c.src_offset, c.dst_offset, c.size))
                .collect()
        };
        assert_eq!(
            runs.len(),
            3,
            "a change of source buffer starts a new command"
        );
        assert_eq!((runs[0].src.as_raw(), runs[0].dst.as_raw()), (1, 9));
        assert_eq!(regions(&runs[0]), [(0, 64, 16), (32, 0, 8)]);
        assert_eq!(
            (runs[1].src.as_raw(), regions(&runs[1])),
            (2, vec![(0, 128, 4)])
        );
        assert_eq!(
            (runs[2].src.as_raw(), regions(&runs[2])),
            (1, vec![(48, 200, 4)])
        );
        assert!(copy_runs(&[]).is_empty());
    }

    #[test]
    fn timestamps_read_as_device_nanoseconds() {
        assert_eq!(stamp_ns(1000, u64::MAX, 10.0), 10_000);
        let mask = (1u64 << 36) - 1;
        assert_eq!(
            stamp_ns(mask + 1 + 7, mask, 1.0),
            7,
            "only the counter's valid bits count"
        );
        assert_eq!(stamp_ns(3, u64::MAX, 2.5), 8, "rounded to the nearest ns");
    }

    #[test]
    fn durations_are_the_gaps_between_timestamps_in_ns() {
        assert_eq!(
            durations(&[100, 150, 150, 400], u64::MAX, 10.0),
            [500, 0, 2500]
        );
        assert_eq!(durations(&[7], u64::MAX, 10.0), Vec::<u64>::new());
        // A counter of 36 valid bits wraps between two stamps.
        let mask = (1u64 << 36) - 1;
        assert_eq!(durations(&[mask - 5, 4], mask, 1.0), [10]);
        // A stamp earlier than the one before it (two commands overlapping, no barrier between them) reads as 0, not
        // as most of the counter's range.
        assert_eq!(durations(&[400, 390], u64::MAX, 10.0), [0]);
        assert_eq!(durations(&[400, 390], mask, 10.0), [0]);
    }
}
