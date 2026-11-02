//! Vulkan plumbing for the probes: device choice, memory, pipelines, submission and timestamps.
//! Shaders take buffer device addresses through push constants, so there are no descriptor sets.
use crate::record::Record;
use anyhow::{anyhow, bail, Context as _, Result};
use ash::vk;
use serde::Serialize;
use serde_json::{json, Value};
use std::ffi::CStr;

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub discrete: bool,
}

/// The device to use: the discrete GPU whose name contains `want`, or the only discrete GPU. Never an iGPU.
pub fn pick_device(devs: &[DeviceInfo], want: Option<&str>) -> Result<usize> {
    let listing = devs
        .iter()
        .map(|d| {
            format!(
                "{} ({})",
                d.name,
                if d.discrete {
                    "discrete"
                } else {
                    "not discrete"
                }
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let matches: Vec<usize> = (0..devs.len())
        .filter(|&i| devs[i].discrete && want.is_none_or(|w| devs[i].name.contains(w)))
        .collect();
    match matches.as_slice() {
        [i] => Ok(*i),
        [] => bail!("no matching discrete GPU; found: {listing}"),
        _ => bail!("more than one discrete GPU matches, pass --device; found: {listing}"),
    }
}

/// The first memory type with every `required` flag and no `forbidden` flag.
pub fn select_mem_type(
    types: &[vk::MemoryPropertyFlags],
    required: vk::MemoryPropertyFlags,
    forbidden: vk::MemoryPropertyFlags,
) -> Option<u32> {
    types
        .iter()
        .position(|f| f.contains(required) && !f.intersects(forbidden))
        .map(|i| i as u32)
}

/// How long to run a probe's own dispatches before timing them, so GPU clocks have ramped up. Without it, short
/// runs in a fresh process measured as little as a third of the warm speed.
pub const WARM_UP: std::time::Duration = std::time::Duration::from_millis(250);

/// Dispatches per submit so each submit is about 20 ms: far under the Windows watchdog, long enough to amortize
/// the submit. `single_us` is one dispatch's time; 0 or NaN gives the cap.
pub fn dispatches_per_submit(single_us: f64) -> u32 {
    ((20_000.0 / single_us.max(1e-3)) as u32).clamp(1, 1000)
}

/// GPU timestamps if the queue family has them; otherwise the caller times submit-and-wait on the CPU.
pub fn use_gpu_timestamps(timestamp_valid_bits: u32) -> bool {
    timestamp_valid_bits > 0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mem {
    /// VRAM the CPU can't see.
    Vram,
    /// Pinned system memory the GPU reads over PCIe (not the ReBAR window, not CPU-cached).
    Host,
    /// Pinned system memory that the CPU caches.
    HostCached,
    /// AMD's device-uncached host memory (VK_AMD_device_coherent_memory), meant for polling.
    AmdUncached,
}

impl Mem {
    pub fn flags(self) -> (vk::MemoryPropertyFlags, vk::MemoryPropertyFlags) {
        use vk::MemoryPropertyFlags as F;
        let amd = F::DEVICE_COHERENT_AMD | F::DEVICE_UNCACHED_AMD;
        match self {
            Mem::Vram => (F::DEVICE_LOCAL, F::HOST_VISIBLE | amd),
            Mem::Host => (
                F::HOST_VISIBLE | F::HOST_COHERENT,
                F::DEVICE_LOCAL | F::HOST_CACHED | amd,
            ),
            Mem::HostCached => (
                F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED,
                F::DEVICE_LOCAL | amd,
            ),
            Mem::AmdUncached => (
                F::HOST_VISIBLE | F::HOST_COHERENT | F::DEVICE_UNCACHED_AMD,
                F::DEVICE_LOCAL,
            ),
        }
    }
}

pub fn bytes_of<T: Copy>(t: &T) -> &[u8] {
    // SAFETY: push-constant structs are #[repr(C)] plain data; reading their bytes is fine.
    unsafe { std::slice::from_raw_parts(t as *const T as *const u8, std::mem::size_of::<T>()) }
}

pub fn is_device_lost(e: &anyhow::Error) -> bool {
    e.chain()
        .any(|c| c.downcast_ref::<vk::Result>() == Some(&vk::Result::ERROR_DEVICE_LOST))
}

pub fn names(props: &vk::PhysicalDeviceProperties) -> (String, String) {
    let name = props
        .device_name_as_c_str()
        .unwrap_or(c"?")
        .to_string_lossy()
        .into_owned();
    let v = props.driver_version;
    let driver = format!(
        "{}.{}.{} (0x{v:x})",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    );
    (name, driver)
}

pub fn open_instance(
    want: Option<&str>,
) -> Result<(ash::Entry, ash::Instance, vk::PhysicalDevice)> {
    // SAFETY: loading the system Vulkan loader and creating an instance with no layers or extensions.
    unsafe {
        let entry = ash::Entry::load().context("loading vulkan-1.dll")?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"artemis-m0")
            .api_version(vk::API_VERSION_1_3);
        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )?;
        let pdevs = instance.enumerate_physical_devices()?;
        let infos: Vec<DeviceInfo> = pdevs
            .iter()
            .map(|&pd| {
                let p = instance.get_physical_device_properties(pd);
                DeviceInfo {
                    name: names(&p).0,
                    discrete: p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
                }
            })
            .collect();
        let i = pick_device(&infos, want)?;
        Ok((entry, instance, pdevs[i]))
    }
}

pub struct Buf {
    pub buf: vk::Buffer,
    pub mem: vk::DeviceMemory,
    pub size: u64,
    pub addr: u64,
    pub ptr: *mut u8,
}

pub struct Pipe {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

pub struct Stamps {
    pub pool: vk::QueryPool,
    pub n: u32,
}

pub struct Ctx {
    _entry: ash::Entry,
    pub instance: ash::Instance,
    pub pdev: vk::PhysicalDevice,
    pub device: ash::Device,
    pub name: String,
    pub driver: String,
    pub compute_family: u32,
    pub transfer_family: u32,
    pub compute_queue: vk::Queue,
    pub transfer_queue: vk::Queue,
    pub compute_pool: vk::CommandPool,
    pub transfer_pool: vk::CommandPool,
    pub mem_types: Vec<vk::MemoryPropertyFlags>,
    pub timestamp_ns: f64,
    pub transfer_timestamps: bool,
    pub has_amd_coherent: bool,
}

impl Ctx {
    pub fn new(want: Option<&str>) -> Result<Ctx> {
        let (entry, instance, pdev) = open_instance(want)?;
        // SAFETY: all handles come from this instance and device; create-info structs outlive the calls.
        unsafe {
            let props = instance.get_physical_device_properties(pdev);
            let (name, driver) = names(&props);
            let families = instance.get_physical_device_queue_family_properties(pdev);
            let compute_family = families
                .iter()
                .position(|f| {
                    f.queue_flags.contains(vk::QueueFlags::COMPUTE) && f.timestamp_valid_bits > 0
                })
                .context("no compute queue family with timestamps")?
                as u32;
            let transfer_family = families
                .iter()
                .position(|f| {
                    f.queue_flags.contains(vk::QueueFlags::TRANSFER)
                        && !f
                            .queue_flags
                            .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                })
                .map(|i| i as u32)
                .unwrap_or(compute_family);
            let transfer_timestamps =
                use_gpu_timestamps(families[transfer_family as usize].timestamp_valid_bits);

            let exts: Vec<String> = instance
                .enumerate_device_extension_properties(pdev)?
                .iter()
                .map(|e| {
                    e.extension_name_as_c_str()
                        .unwrap_or(c"")
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            let has = |n: &CStr| exts.iter().any(|e| e.as_str() == n.to_str().unwrap_or(""));
            if !has(ash::ext::memory_budget::NAME) {
                bail!("VK_EXT_memory_budget is missing");
            }
            let has_amd_coherent = has(ash::amd::device_coherent_memory::NAME);
            let mut enable = vec![ash::ext::memory_budget::NAME.as_ptr()];
            if has_amd_coherent {
                enable.push(ash::amd::device_coherent_memory::NAME.as_ptr());
            }

            let prio = [1.0f32];
            let mut queues = vec![vk::DeviceQueueCreateInfo::default()
                .queue_family_index(compute_family)
                .queue_priorities(&prio)];
            if transfer_family != compute_family {
                queues.push(
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(transfer_family)
                        .queue_priorities(&prio),
                );
            }
            let mut f11 =
                vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .timeline_semaphore(true)
                .host_query_reset(true)
                .shader_int8(true)
                .shader_float16(true)
                .storage_buffer8_bit_access(true)
                .vulkan_memory_model(true)
                .vulkan_memory_model_device_scope(true);
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
                .synchronization2(true)
                .subgroup_size_control(true);
            let mut famd =
                vk::PhysicalDeviceCoherentMemoryFeaturesAMD::default().device_coherent_memory(true);
            let core = vk::PhysicalDeviceFeatures::default()
                .shader_int64(true)
                .shader_int16(true);
            let mut feats = vk::PhysicalDeviceFeatures2::default()
                .features(core)
                .push_next(&mut f11)
                .push_next(&mut f12)
                .push_next(&mut f13);
            if has_amd_coherent {
                feats = feats.push_next(&mut famd);
            }
            let info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&enable)
                .push_next(&mut feats);
            let device = instance
                .create_device(pdev, &info, None)
                .context("create_device failed; run `m0 limits` to see which feature is missing")?;

            let compute_queue = device.get_device_queue(compute_family, 0);
            let transfer_queue = device.get_device_queue(transfer_family, 0);
            let pool_info = |fam| {
                vk::CommandPoolCreateInfo::default()
                    .queue_family_index(fam)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            };
            let compute_pool = device.create_command_pool(&pool_info(compute_family), None)?;
            let transfer_pool = device.create_command_pool(&pool_info(transfer_family), None)?;
            let mp = instance.get_physical_device_memory_properties(pdev);
            let mem_types = mp.memory_types[..mp.memory_type_count as usize]
                .iter()
                .map(|t| t.property_flags)
                .collect();
            Ok(Ctx {
                _entry: entry,
                instance,
                pdev,
                device,
                name,
                driver,
                compute_family,
                transfer_family,
                compute_queue,
                transfer_queue,
                compute_pool,
                transfer_pool,
                mem_types,
                timestamp_ns: props.limits.timestamp_period as f64,
                transfer_timestamps,
                has_amd_coherent,
            })
        }
    }

    /// A storage buffer with a device address, in the given memory; host-visible memory is mapped.
    pub fn buf(&self, size: u64, mem: Mem) -> Result<Buf> {
        let (required, forbidden) = mem.flags();
        let ty = select_mem_type(&self.mem_types, required, forbidden)
            .with_context(|| format!("no memory type for {mem:?} on this device"))?;
        let families = [self.compute_family, self.transfer_family];
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let mut info = vk::BufferCreateInfo::default().size(size).usage(usage);
        info = if self.compute_family != self.transfer_family {
            info.sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families)
        } else {
            info.sharing_mode(vk::SharingMode::EXCLUSIVE)
        };
        // SAFETY: the buffer is bound to fresh memory before any use and freed together with it in `free`.
        unsafe {
            let buf = self.device.create_buffer(&info, None)?;
            let req = self.device.get_buffer_memory_requirements(buf);
            if req.memory_type_bits & (1 << ty) == 0 {
                self.device.destroy_buffer(buf, None);
                bail!("memory type {ty} ({mem:?}) is not allowed for storage buffers");
            }
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(ty)
                .push_next(&mut flags);
            let memory = match self.device.allocate_memory(&alloc, None) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buf, None);
                    return Err(anyhow!("allocating {} MiB of {mem:?}: {e}", size >> 20));
                }
            };
            self.device.bind_buffer_memory(buf, memory, 0)?;
            let addr = self
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf));
            let ptr = if mem == Mem::Vram {
                std::ptr::null_mut()
            } else {
                self.device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
                    as *mut u8
            };
            Ok(Buf {
                buf,
                mem: memory,
                size,
                addr,
                ptr,
            })
        }
    }

    pub fn free(&self, b: Buf) {
        // SAFETY: callers free a buffer only after the GPU work using it has completed.
        unsafe {
            self.device.destroy_buffer(b.buf, None);
            self.device.free_memory(b.mem, None);
        }
    }

    /// Copy `data` into `dst`: directly if it's mapped, otherwise through a temporary host buffer.
    pub fn upload(&self, dst: &Buf, data: &[u8]) -> Result<()> {
        if !dst.ptr.is_null() {
            // SAFETY: dst is mapped and at least data.len() bytes long.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst.ptr, data.len()) };
            return Ok(());
        }
        let staging = self.buf(data.len() as u64, Mem::Host)?;
        // SAFETY: staging is mapped and exactly data.len() bytes long.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), staging.ptr, data.len()) };
        let r = self.run(self.compute_queue, self.compute_pool, |cmd| unsafe {
            let region = [vk::BufferCopy::default().size(data.len() as u64)];
            self.device
                .cmd_copy_buffer(cmd, staging.buf, dst.buf, &region);
        });
        self.free(staging);
        r
    }

    /// Record a one-time command buffer, submit it, and wait up to 10 s. Device loss comes back as an error.
    pub fn run(
        &self,
        queue: vk::Queue,
        pool: vk::CommandPool,
        record: impl FnOnce(vk::CommandBuffer),
    ) -> Result<()> {
        // SAFETY: the command buffer and fence are created, used and destroyed here.
        unsafe {
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = self.device.allocate_command_buffers(&alloc)?[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cmd, &begin)?;
            record(cmd);
            self.device.end_command_buffer(cmd)?;
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [cmd];
            let r = self
                .device
                .queue_submit(
                    queue,
                    &[vk::SubmitInfo::default().command_buffers(&cmds)],
                    fence,
                )
                .and_then(|_| self.device.wait_for_fences(&[fence], true, 10_000_000_000));
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(pool, &cmds);
            r.map_err(|e| anyhow::Error::new(e).context("GPU submission"))
        }
    }

    pub fn pipe(&self, spv: &[u8], push_size: u32) -> Result<Pipe> {
        let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))?;
        // SAFETY: the module is only needed until the pipeline exists.
        unsafe {
            let module = self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?;
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(push_size)];
            let layout = self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges),
                None,
            )?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let info = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)];
            let pipeline = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &info, None)
                .map_err(|(_, e)| e)?[0];
            self.device.destroy_shader_module(module, None);
            Ok(Pipe { pipeline, layout })
        }
    }

    pub fn dispatch<T: Copy>(&self, cmd: vk::CommandBuffer, p: &Pipe, push: &T, groups: u32) {
        // SAFETY: cmd is recording; push matches the pipeline's push-constant range.
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.pipeline);
            self.device.cmd_push_constants(
                cmd,
                p.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(push),
            );
            self.device.cmd_dispatch(cmd, groups, 1, 1);
        }
    }

    /// Earlier compute and transfer writes complete before later compute work starts.
    pub fn barrier(&self, cmd: vk::CommandBuffer) {
        let mb = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        // SAFETY: cmd is recording.
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &mb,
                &[],
                &[],
            )
        };
    }

    /// Shader writes are visible to the CPU once the submission's fence signals.
    pub fn host_barrier(&self, cmd: vk::CommandBuffer) {
        let mb = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        // SAFETY: cmd is recording.
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &mb,
                &[],
                &[],
            )
        };
    }

    pub fn stamps(&self, n: u32) -> Result<Stamps> {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(n);
        // SAFETY: plain query-pool creation.
        Ok(Stamps {
            pool: unsafe { self.device.create_query_pool(&info, None)? },
            n,
        })
    }

    /// Reset from the host (hostQueryReset), which works on every queue family.
    pub fn reset_stamps(&self, s: &Stamps) {
        // SAFETY: no submitted work is using the pool.
        unsafe { self.device.reset_query_pool(s.pool, 0, s.n) };
    }

    pub fn stamp(&self, cmd: vk::CommandBuffer, s: &Stamps, i: u32) {
        // SAFETY: cmd is recording and i < s.n.
        unsafe {
            self.device
                .cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, s.pool, i)
        };
    }

    pub fn read_stamps_ns(&self, s: &Stamps) -> Result<Vec<f64>> {
        self.read_first_stamps_ns(s, s.n)
    }

    /// The first `count` timestamps only. Reading a query that was reset but never written waits forever.
    pub fn read_first_stamps_ns(&self, s: &Stamps, count: u32) -> Result<Vec<f64>> {
        let mut raw = vec![0u64; count.min(s.n) as usize];
        // SAFETY: raw has room for the requested 64-bit results.
        unsafe {
            self.device.get_query_pool_results(
                s.pool,
                0,
                &mut raw,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?
        };
        Ok(raw.iter().map(|&t| t as f64 * self.timestamp_ns).collect())
    }

    /// Budget and usage of every memory heap, in MiB, as the driver reports them now.
    pub fn heap_budgets(&self) -> Value {
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut m2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        // SAFETY: plain property query.
        unsafe {
            self.instance
                .get_physical_device_memory_properties2(self.pdev, &mut m2)
        };
        let n = m2.memory_properties.memory_heap_count as usize;
        let sizes: Vec<u64> = m2.memory_properties.memory_heaps[..n]
            .iter()
            .map(|h| h.size >> 20)
            .collect();
        (0..n)
            .map(|i| json!({"heap": i, "size_mib": sizes[i], "budget_mib": budget.heap_budget[i] >> 20, "usage_mib": budget.heap_usage[i] >> 20}))
            .collect()
    }

    /// A command buffer that may be submitted many times, even while a previous submission is pending.
    pub fn reusable(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<vk::CommandBuffer> {
        // SAFETY: the command buffer comes from the compute pool and lives as long as the Ctx.
        unsafe {
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.compute_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = self.device.allocate_command_buffers(&alloc)?[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
            self.device.begin_command_buffer(cmd, &begin)?;
            record(cmd);
            self.device.end_command_buffer(cmd)?;
            Ok(cmd)
        }
    }

    pub fn timeline(&self) -> Result<vk::Semaphore> {
        let mut ty = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        // SAFETY: plain semaphore creation.
        Ok(unsafe {
            self.device
                .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut ty), None)?
        })
    }

    /// Submit to the compute queue without a fence, optionally waiting on a timeline value first.
    pub fn submit(&self, cmd: vk::CommandBuffer, wait: Option<(vk::Semaphore, u64)>) -> Result<()> {
        let cmds = [cmd];
        let (sems, values) = match wait {
            Some((s, v)) => (vec![s], vec![v]),
            None => (vec![], vec![]),
        };
        let stages = vec![vk::PipelineStageFlags::COMPUTE_SHADER; sems.len()];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values);
        let info = vk::SubmitInfo::default()
            .wait_semaphores(&sems)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&cmds)
            .push_next(&mut timeline);
        // SAFETY: cmd was recorded with SIMULTANEOUS_USE; the arrays outlive the call.
        unsafe {
            self.device
                .queue_submit(self.compute_queue, &[info], vk::Fence::null())
        }
        .map_err(|e| anyhow::Error::new(e).context("GPU submission"))
    }

    pub fn signal(&self, sem: vk::Semaphore, value: u64) -> Result<()> {
        // SAFETY: value is larger than the semaphore's current value.
        unsafe {
            self.device.signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(sem)
                    .value(value),
            )
        }
        .map_err(|e| anyhow::Error::new(e).context("host signal"))
    }

    /// Submit `record` on the compute queue repeatedly for `WARM_UP`, so the timed runs that follow see warm clocks.
    pub fn warm_up(&self, record: impl Fn(vk::CommandBuffer)) -> Result<()> {
        let t0 = std::time::Instant::now();
        while t0.elapsed() < WARM_UP {
            self.run(self.compute_queue, self.compute_pool, &record)?;
        }
        Ok(())
    }

    pub fn record(&self, probe: &str, params: impl Serialize, results: Value) -> Result<Record> {
        Ok(Record::new(
            probe,
            &self.name,
            &self.driver,
            serde_json::to_value(params)?,
            results,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::MemoryPropertyFlags as F;

    fn dev(name: &str, discrete: bool) -> DeviceInfo {
        DeviceInfo {
            name: name.into(),
            discrete,
        }
    }

    #[test]
    fn picks_the_discrete_gpu_not_the_igpu() {
        let devs = [
            dev("AMD Radeon(TM) Graphics", false),
            dev("AMD Radeon RX 9070 XT", true),
        ];
        assert_eq!(pick_device(&devs, None).unwrap(), 1);
        assert_eq!(pick_device(&devs, Some("9070")).unwrap(), 1);
    }

    #[test]
    fn refuses_when_only_an_igpu_exists() {
        let err = pick_device(&[dev("AMD Radeon(TM) Graphics", false)], None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Radeon(TM) Graphics (not discrete)"), "{err}");
    }

    #[test]
    fn host_memory_never_selects_the_bar_heap() {
        // AMD on Windows: plain VRAM, the ReBAR window, plain pinned host memory, cached host memory.
        let types = [
            F::DEVICE_LOCAL,
            F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT,
            F::HOST_VISIBLE | F::HOST_COHERENT,
            F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED,
        ];
        let pick = |m: Mem| {
            let (req, forbid) = m.flags();
            select_mem_type(&types, req, forbid)
        };
        assert_eq!(pick(Mem::Vram), Some(0));
        assert_eq!(pick(Mem::Host), Some(2));
        assert_eq!(pick(Mem::HostCached), Some(3));
        assert_eq!(pick(Mem::AmdUncached), None);
    }

    #[test]
    fn submits_are_sized_to_about_20_ms() {
        assert_eq!(dispatches_per_submit(25.0), 800);
        assert_eq!(dispatches_per_submit(10_000.0), 2);
        assert_eq!(dispatches_per_submit(20_000.0), 1);
        assert_eq!(dispatches_per_submit(60_000.0), 1);
        assert_eq!(dispatches_per_submit(0.0), 1000);
    }

    #[test]
    fn falls_back_to_cpu_timing_without_timestamp_bits() {
        assert!(!use_gpu_timestamps(0));
        assert!(use_gpu_timestamps(64));
    }
}
