//! Choosing the GPU, checking its features, and the device itself: queues, command pools, timeline semaphores and
//! memory types.
use crate::{
    error::{vk_err, Error},
    memory::{select_type, Buffer, MemKind},
};
use ash::vk;
use std::{
    cell::{Cell, RefCell},
    sync::Arc,
};

/// A physical device as `pick_device` sees it.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub name: String,
    pub uuid: [u8; 16],
    pub discrete: bool,
}

/// The discrete GPU with UUID `want`, or the only discrete GPU when `want` is None. Never an integrated GPU, such as a
/// CPU's.
pub fn pick_device(cands: &[Candidate], want: Option<[u8; 16]>) -> Result<usize, Error> {
    let listing = cands
        .iter()
        .map(|c| {
            let kind = if c.discrete {
                "discrete"
            } else {
                "not discrete"
            };
            format!("{} ({kind}, uuid {})", c.name, format_uuid(&c.uuid))
        })
        .collect::<Vec<_>>()
        .join("; ");
    let matches: Vec<usize> = (0..cands.len())
        .filter(|&i| cands[i].discrete && want.is_none_or(|u| cands[i].uuid == u))
        .collect();
    match matches.as_slice() {
        [i] => Ok(*i),
        [] => Err(Error::NoDevice(listing)),
        _ => Err(Error::NoDevice(format!(
            "more than one discrete GPU, so name one by UUID: {listing}"
        ))),
    }
}

/// A UUID written as 32 hex digits, with or without dashes.
pub fn parse_uuid(s: &str) -> Option<[u8; 16]> {
    let hex: Vec<u8> = s.bytes().filter(|&c| c != b'-').collect();
    if hex.len() != 32 || !hex.iter().all(u8::is_ascii_hexdigit) {
        return None;
    }
    let mut out = [0u8; 16];
    for (o, pair) in out.iter_mut().zip(hex.as_chunks::<2>().0) {
        *o = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
    }
    Some(out)
}

/// 8-4-4-4-12 hex, as vulkaninfo prints it.
pub fn format_uuid(u: &[u8; 16]) -> String {
    let h: Vec<String> = u.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        h[0..4].concat(),
        h[4..6].concat(),
        h[6..8].concat(),
        h[8..10].concat(),
        h[10..16].concat()
    )
}

/// What the device supports of the required features, plus the 16- and 64-bit integers the kernels use.
#[derive(Clone, Debug, Default)]
pub struct Support {
    pub vulkan_1_3: bool,
    pub buffer_device_address: bool,
    pub timeline_semaphore: bool,
    pub storage_buffer_8bit: bool,
    pub storage_buffer_16bit: bool,
    pub shader_float16: bool,
    pub shader_int8: bool,
    pub shader_int16: bool,
    pub shader_int64: bool,
    pub subgroup_arithmetic: bool,
    pub subgroup_shuffle: bool,
    pub memory_budget: bool,
}

/// The required features `s` lacks, by name; empty when the device will do.
pub fn missing_features(s: &Support) -> Vec<&'static str> {
    [
        (s.vulkan_1_3, "Vulkan 1.3"),
        (s.buffer_device_address, "bufferDeviceAddress"),
        (s.timeline_semaphore, "timelineSemaphore"),
        (s.storage_buffer_8bit, "storageBuffer8BitAccess"),
        (s.storage_buffer_16bit, "storageBuffer16BitAccess"),
        (s.shader_float16, "shaderFloat16"),
        (s.shader_int8, "shaderInt8"),
        (s.shader_int16, "shaderInt16"),
        (s.shader_int64, "shaderInt64"),
        (s.subgroup_arithmetic, "subgroup arithmetic in compute"),
        (s.subgroup_shuffle, "subgroup shuffle"),
        (s.memory_budget, "VK_EXT_memory_budget"),
    ]
    .into_iter()
    .filter(|(ok, _)| !ok)
    .map(|(_, name)| name)
    .collect()
}

/// Whether `shapes` (from `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`) lists the shape prefill's
/// cooperative-matrix kernels use: 16 × 16 × 16, f16 A and B, f32 C and result, subgroup scope.
pub fn has_f16_shape(shapes: &[vk::CooperativeMatrixPropertiesKHR<'_>]) -> bool {
    shapes.iter().any(|s| {
        (s.m_size, s.n_size, s.k_size) == (16, 16, 16)
            && s.a_type == vk::ComponentTypeKHR::FLOAT16
            && s.b_type == vk::ComponentTypeKHR::FLOAT16
            && s.c_type == vk::ComponentTypeKHR::FLOAT32
            && s.result_type == vk::ComponentTypeKHR::FLOAT32
            && s.scope == vk::ScopeKHR::SUBGROUP
    })
}

/// The mask of a timestamp counter with `bits` valid bits.
fn stamp_mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// A heap's size, the budget the driver gives this process, and this process's usage, in bytes.
#[derive(Clone, Copy, Debug)]
pub struct HeapBudget {
    pub size: u64,
    pub budget: u64,
    pub usage: u64,
}

/// The instance and logical device. Buffers and pipelines hold it, so it outlives them.
pub(crate) struct Inner {
    _entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) pdev: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: the last buffer, pipeline and Device using this device are gone.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub(crate) struct Queue {
    pub(crate) queue: vk::Queue,
    pub(crate) pool: vk::CommandPool,
    pub(crate) timeline: vk::Semaphore,
    /// The value the last submit signals.
    pub(crate) value: Cell<u64>,
    /// The highest value the host has seen the timeline reach: through `completed`, `wait`, or a submit it waited for.
    /// The other queue's submits wait for it (`cross_wait`).
    pub(crate) synced: Cell<u64>,
}

pub struct Device {
    pub(crate) inner: Arc<Inner>,
    pub name: String,
    pub driver: String,
    pub uuid: [u8; 16],
    /// The default subgroup size. Kernels never depend on it; it is reported for the evidence.
    pub subgroup_size: u32,
    /// The compute queue's timestamps, for profiling: nanoseconds per tick, and the mask of
    /// the counter's valid bits. None when the queue family has none.
    pub(crate) timestamps: Option<(f64, u64)>,
    pub(crate) compute: Queue,
    pub(crate) transfer: Queue,
    pub(crate) families: Vec<u32>,
    /// Each memory type's flags and heap.
    pub(crate) mem_types: Vec<(vk::MemoryPropertyFlags, u32)>,
    /// `VK_AMD_buffer_marker`, and the host buffer the markers land in: only with `ARTEMIS_MARKERS` set (`layer` or
    /// `dispatch`), since markers cost decode speed.
    pub(crate) marker: Option<(ash::amd::buffer_marker::Device, Buffer)>,
    /// A marker after every dispatch (`ARTEMIS_MARKERS=dispatch`), not only at each tag (`Recorder::tag`, a layer):
    /// device loss then names the exact dispatch in flight.
    pub(crate) marker_every_dispatch: bool,
    /// Async copies on the transfer queue (`copy_async`, `stage_async`): each command buffer and the timeline value it
    /// signals, freed once the timeline has passed it.
    pub(crate) pending: RefCell<Vec<(vk::CommandBuffer, u64)>>,
    /// The transfer queue's timestamps (ns per tick, valid-bit mask), for timing staging copies;
    /// None when its family has none, or without host query reset (a transfer-only queue can't reset a query pool).
    pub(crate) transfer_stamps: Option<(f64, u64)>,
    /// Whether the device runs f16 × f16 → f32 16 × 16 × 16 cooperative matrices at subgroup scope, with the extension
    /// and the Vulkan memory model their shaders declare enabled.
    pub(crate) coop_f16: bool,
    /// Whether `VK_KHR_shader_clock`'s device clock is enabled, so compute shaders can read
    /// `clockRealtimeEXT()`: the miss path's bounded spin times itself on it.
    pub shader_clock: bool,
    /// Timed staging copies (`stage_async`): the transfer value each signals and its two timestamps' pool, read and
    /// destroyed by `copy_spans` once the host has seen the copy finish, or by `Drop`.
    pub(crate) stamped: RefCell<Vec<(u64, vk::QueryPool)>>,
    /// The last profiled compute submit's first and last timestamps, in device ns (`submit_profiled`).
    pub(crate) last_span: Cell<Option<(u64, u64)>>,
    /// The submit trace (`trace.rs`): only with `ARTEMIS_SUBMIT_TRACE` set when the device opened.
    pub(crate) trace: Option<crate::trace::Trace>,
    /// `VK_KHR_calibrated_timestamps` was enabled when the device opened (only for the submit trace).
    pub(crate) calibration_enabled: bool,
}

struct Opened {
    pdev: vk::PhysicalDevice,
    device: ash::Device,
    name: String,
    driver: String,
    uuid: [u8; 16],
    subgroup_size: u32,
    compute_family: u32,
    transfer_family: u32,
    has_marker: bool,
    marker_every_dispatch: bool,
    timestamps: Option<(f64, u64)>,
    transfer_stamps: Option<(f64, u64)>,
    coop_f16: bool,
    shader_clock: bool,
    /// `VK_KHR_calibrated_timestamps` is enabled, for the submit trace.
    calibrated: bool,
}

/// A driver version as the logs and the tuned file spell it.
fn driver_text(v: u32) -> String {
    format!(
        "{}.{}.{} (0x{v:x})",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    )
}

/// One GPU as `list_gpus` sees it, before any device is opened.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub name: String,
    pub uuid: [u8; 16],
    pub discrete: bool,
    pub driver: String,
    /// The largest device-local heap's size, the budget the driver gives and the usage (this process's, and so zero
    /// before it allocates), from `VK_EXT_memory_budget`. Without the extension the budget is the size and usage 0.
    pub heap: HeapBudget,
}

/// Every GPU the Vulkan loader lists. It makes an instance and nothing else (no device, no allocation), so the heap
/// figures are read before this process uses any memory.
pub fn list_gpus() -> Result<Vec<GpuInfo>, Error> {
    let entry = load_entry()?;
    let app = vk::ApplicationInfo::default()
        .application_name(c"artemis")
        .api_version(vk::API_VERSION_1_3);
    // SAFETY: the create info outlives the call.
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )
    }
    .map_err(vk_err("creating the Vulkan instance"))?;
    let out = read_gpus(&instance);
    // SAFETY: nothing else holds the instance.
    unsafe { instance.destroy_instance(None) };
    out
}

fn read_gpus(instance: &ash::Instance) -> Result<Vec<GpuInfo>, Error> {
    // SAFETY: queries on handles from this instance; every struct outlives its call.
    unsafe {
        let pdevs = instance
            .enumerate_physical_devices()
            .map_err(vk_err("listing GPUs"))?;
        let mut out = Vec::new();
        for pd in pdevs {
            let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut p11);
            instance.get_physical_device_properties2(pd, &mut p2);
            let props = p2.properties;
            let has_budget = instance
                .enumerate_device_extension_properties(pd)
                .map_err(vk_err("listing device extensions"))?
                .iter()
                .any(|e| {
                    e.extension_name_as_c_str()
                        .is_ok_and(|n| n == ash::ext::memory_budget::NAME)
                });
            let mut b = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let mut m2 = vk::PhysicalDeviceMemoryProperties2::default();
            if has_budget {
                m2 = m2.push_next(&mut b);
            }
            instance.get_physical_device_memory_properties2(pd, &mut m2);
            let mp = m2.memory_properties;
            let heap = (0..mp.memory_heap_count as usize)
                .filter(|&i| {
                    mp.memory_heaps[i]
                        .flags
                        .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                })
                .max_by_key(|&i| mp.memory_heaps[i].size)
                .map(|i| {
                    let size = mp.memory_heaps[i].size;
                    if has_budget {
                        HeapBudget {
                            size,
                            budget: b.heap_budget[i],
                            usage: b.heap_usage[i],
                        }
                    } else {
                        HeapBudget {
                            size,
                            budget: size,
                            usage: 0,
                        }
                    }
                })
                .unwrap_or(HeapBudget {
                    size: 0,
                    budget: 0,
                    usage: 0,
                });
            out.push(GpuInfo {
                name: device_name(&props),
                uuid: p11.device_uuid,
                discrete: props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
                driver: driver_text(props.driver_version),
                heap,
            });
        }
        Ok(out)
    }
}

fn device_name(p: &vk::PhysicalDeviceProperties) -> String {
    p.device_name_as_c_str()
        .unwrap_or(c"?")
        .to_string_lossy()
        .into_owned()
}

/// The Vulkan loader's path on Windows: `vulkan-1.dll` in the system directory. A bare `vulkan-1.dll` is searched for in
/// the executable's folder first, then the current directory and `PATH`, so a DLL planted next to `artemis.exe` would
/// run with the user's rights. The GPU driver installs the real loader in System32.
#[cfg(windows)]
pub(crate) fn system_loader() -> Result<std::path::PathBuf, Error> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;
    let mut buf = [0u16; 260];
    // SAFETY: the buffer is writable for the length passed.
    let n = unsafe { GetSystemDirectoryW(buf.as_mut_ptr(), buf.len() as u32) } as usize;
    if n == 0 || n >= buf.len() {
        return Err(Error::Loader(
            "cannot find the Windows system directory".into(),
        ));
    }
    Ok(std::path::PathBuf::from(std::ffi::OsString::from_wide(&buf[..n])).join("vulkan-1.dll"))
}

/// Limits the process's default DLL search to System32 (`SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32)`), once.
///
/// Loading `vulkan-1.dll` with that flag covers its own imports, but the loader then opens the GPU driver (the ICD) and
/// any implicit layers with a plain `LoadLibraryW`, and a driver's dependents that are not KnownDLLs (`winmm.dll` for
/// AMD's) are searched for in the executable's folder first. With the process default set, those dependents come from
/// System32 as well, so a DLL planted next to `artemis.exe` is not loaded. A failure is ignored: the flagged load below
/// still applies.
///
/// The `artemis` binary calls this first thing in `main`, before argument parsing, logging or any error path (a backtrace
/// loads `dbghelp.dll`); `Device::open` calls it again, so library users and tests are covered. Later calls do nothing.
/// It does nothing off Windows.
#[cfg(windows)]
pub fn limit_dll_search_to_system32() {
    use std::sync::Once;
    use windows_sys::Win32::System::LibraryLoader::{
        SetDefaultDllDirectories, LOAD_LIBRARY_SEARCH_SYSTEM32,
    };
    static ONCE: Once = Once::new();
    // SAFETY: a plain Win32 call with a documented flag value.
    ONCE.call_once(|| unsafe {
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32);
    });
}

#[cfg(not(windows))]
pub fn limit_dll_search_to_system32() {}

/// Loads the Vulkan loader: System32's on Windows, the platform's default elsewhere.
///
/// On Windows the DLL is loaded with `LOAD_LIBRARY_SEARCH_SYSTEM32`, so the loader's own imports (`cfgmgr32.dll`, which
/// is not a KnownDLL, among them) also come from System32. `Entry::load_from` passes no flags, and `vulkan-1.dll` has no
/// dependent load flag of its own, so the DLLs it imports would be searched for in the executable's folder first.
fn load_entry() -> Result<ash::Entry, Error> {
    #[cfg(windows)]
    {
        use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_SYSTEM32};
        limit_dll_search_to_system32();
        let path = system_loader()?;
        // The error with its OS error appended (libloading's text alone is just "LoadLibraryExW failed").
        let chain = |e: libloading::Error| {
            let mut why = e.to_string();
            let mut src = std::error::Error::source(&e);
            while let Some(s) = src {
                why.push_str(": ");
                why.push_str(&s.to_string());
                src = s.source();
            }
            why
        };
        // SAFETY: loads the system Vulkan loader from an absolute path in System32; its initialisers are the driver
        // vendor's and Microsoft's code.
        let lib = unsafe { Library::load_with_flags(&path, LOAD_LIBRARY_SEARCH_SYSTEM32) }
            .map_err(|e| {
                Error::Loader(format!(
                    "no loader at {}; install or update the GPU driver ({})",
                    path.display(),
                    chain(e)
                ))
            })?;
        // SAFETY: the symbol has this signature in every Vulkan loader.
        let get_instance_proc_addr =
            unsafe { lib.get::<vk::PFN_vkGetInstanceProcAddr>(b"vkGetInstanceProcAddr\0") }
                .map_err(|e| {
                    Error::Loader(format!(
                        "{} loaded but has no vkGetInstanceProcAddr; update the GPU driver ({})",
                        path.display(),
                        chain(e)
                    ))
                })?;
        let get_instance_proc_addr = *get_instance_proc_addr;
        // The entry's function pointers point into the loader, which must stay loaded for the life of the process (the
        // instance and device are built from it). Dropping the handle would unload it, so it is leaked on purpose: one
        // reference count per call.
        std::mem::forget(lib);
        // SAFETY: the pointer is vkGetInstanceProcAddr of a loader that is never unloaded.
        Ok(unsafe {
            ash::Entry::from_static_fn(ash::StaticFn {
                get_instance_proc_addr,
            })
        })
    }
    #[cfg(not(windows))]
    {
        // SAFETY: loads the system Vulkan loader.
        unsafe { ash::Entry::load() }.map_err(|e| Error::Loader(e.to_string()))
    }
}

impl Device {
    /// Whether the compute queue has timestamps, so `record_profiled` works.
    pub fn can_profile(&self) -> bool {
        self.timestamps.is_some()
    }

    /// Whether staging copies can be timed (`StageOpts::timed`, `copy_spans`).
    pub fn can_time_copies(&self) -> bool {
        self.transfer_stamps.is_some()
    }

    /// Whether prefill's cooperative-matrix kernels can run here: the f16 shape is listed and enabled. Without it
    /// prefill runs the f32 kernels.
    pub fn coop_f16(&self) -> bool {
        self.coop_f16
    }

    /// Whether `kind` resolved to one of this device's memory types at open (the CPU miss
    /// path's mailbox needs `AmdUncached`, and only this GPU family has it).
    pub fn has_kind(&self, kind: MemKind) -> bool {
        self.mem_type(kind).is_ok()
    }

    /// Opens the discrete GPU, the one with UUID `want` if given, and refuses with a list of any required
    /// feature it lacks.
    pub fn open(want: Option<[u8; 16]>) -> Result<Device, Error> {
        let entry = load_entry()?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"artemis")
            .api_version(vk::API_VERSION_1_3);
        // SAFETY: the create info outlives the call.
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
        }
        .map_err(vk_err("creating the Vulkan instance"))?;
        let o = match open_device(&entry, &instance, want) {
            Ok(o) => o,
            Err(e) => {
                // SAFETY: nothing else holds the instance yet.
                unsafe { instance.destroy_instance(None) };
                return Err(e);
            }
        };
        let queue = |family: u32| -> Result<Queue, Error> {
            // SAFETY: the family was requested at device creation; the create infos outlive the calls.
            unsafe {
                let pool = o
                    .device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(family)
                            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                        None,
                    )
                    .map_err(vk_err("creating a command pool"))?;
                let mut ty = vk::SemaphoreTypeCreateInfo::default()
                    .semaphore_type(vk::SemaphoreType::TIMELINE)
                    .initial_value(0);
                let timeline = o
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut ty), None)
                    .map_err(vk_err("creating a timeline semaphore"))?;
                Ok(Queue {
                    queue: o.device.get_device_queue(family, 0),
                    pool,
                    timeline,
                    value: Cell::new(0),
                    synced: Cell::new(0),
                })
            }
        };
        let compute = queue(o.compute_family)?;
        let transfer = queue(o.transfer_family)?;
        // SAFETY: a plain property query.
        let mp = unsafe { instance.get_physical_device_memory_properties(o.pdev) };
        let mem_types = mp.memory_types[..mp.memory_type_count as usize]
            .iter()
            .map(|t| (t.property_flags, t.heap_index))
            .collect();
        let mut families = vec![o.compute_family];
        if o.transfer_family != o.compute_family {
            families.push(o.transfer_family);
        }
        let marker_fns = o
            .has_marker
            .then(|| ash::amd::buffer_marker::Device::new(&instance, &o.device));
        let trace = if o.calibrated {
            crate::trace::Trace::new(&entry, &instance, o.pdev, &o.device)
        } else {
            None
        };
        let mut dev = Device {
            inner: Arc::new(Inner {
                _entry: entry,
                instance,
                pdev: o.pdev,
                device: o.device,
            }),
            name: o.name,
            driver: o.driver,
            uuid: o.uuid,
            subgroup_size: o.subgroup_size,
            timestamps: o.timestamps,
            compute,
            transfer,
            families,
            mem_types,
            marker: None,
            marker_every_dispatch: o.marker_every_dispatch,
            pending: RefCell::new(Vec::new()),
            transfer_stamps: o.transfer_stamps,
            coop_f16: o.coop_f16,
            shader_clock: o.shader_clock,
            stamped: RefCell::new(Vec::new()),
            last_span: Cell::new(None),
            trace,
            calibration_enabled: o.calibrated,
        };
        if let Some(fns) = marker_fns {
            let buf = dev.alloc(256, MemKind::HostPinned, "dispatch markers")?;
            dev.marker = Some((fns, buf));
        }
        Ok(dev)
    }

    /// The device's `maxComputeSharedMemorySize`, in bytes: the shared memory one compute workgroup may declare.
    pub fn max_compute_shared_memory(&self) -> u32 {
        // SAFETY: a plain property query on this device's physical device.
        let p = unsafe {
            self.inner
                .instance
                .get_physical_device_properties(self.inner.pdev)
        };
        p.limits.max_compute_shared_memory_size
    }

    /// Every heap's size, budget and usage, as the driver reports them now (`VK_EXT_memory_budget`).
    pub fn budget(&self) -> Vec<HeapBudget> {
        let mut b = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut m2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut b);
        // SAFETY: a plain property query.
        unsafe {
            self.inner
                .instance
                .get_physical_device_memory_properties2(self.inner.pdev, &mut m2)
        };
        let n = m2.memory_properties.memory_heap_count as usize;
        let sizes: Vec<u64> = m2.memory_properties.memory_heaps[..n]
            .iter()
            .map(|h| h.size)
            .collect();
        (0..n)
            .map(|i| HeapBudget {
                size: sizes[i],
                budget: b.heap_budget[i],
                usage: b.heap_usage[i],
            })
            .collect()
    }

    /// The memory type for `kind`, and its heap.
    pub(crate) fn mem_type(&self, kind: MemKind) -> Result<(u32, u32), Error> {
        let flags: Vec<vk::MemoryPropertyFlags> = self.mem_types.iter().map(|t| t.0).collect();
        let ty =
            select_type(&flags, kind).ok_or_else(|| Error::NoMemoryType(format!("{kind:?}")))?;
        Ok((ty, self.mem_types[ty as usize].1))
    }

    /// The size, budget and this process's usage of the heap `kind` is allocated from, as the driver reports them now.
    /// The budget can move with the usage: on the RX 9070 XT the VRAM heap's falls from 15,416 MiB to about 14,340 once
    /// this process uses about 11.5 GB, and rises again when it frees memory.
    pub fn heap_budget(&self, kind: MemKind) -> Result<HeapBudget, Error> {
        let (_, heap) = self.mem_type(kind)?;
        Ok(self.budget()[heap as usize])
    }

    /// Bytes left in the budget of `kind`'s heap.
    pub fn free_bytes(&self, kind: MemKind) -> Result<u64, Error> {
        let h = self.heap_budget(kind)?;
        Ok(h.budget.saturating_sub(h.usage))
    }
}

/// A handle that can wait for the whole device to go idle after the `Device` itself is out of reach. It holds the
/// device alive like a buffer does (so it is dropped with the buffers it protects), and it is for an owner of buffers
/// that has no `&Device` in its `Drop` (the engine's snapshot pool must not free memory a transfer-queue copy may still
/// be writing). Like `Buffer` it is `!Send` and `!Sync`: it stays on the thread that owns the `Device`.
#[derive(Clone)]
pub struct IdleHandle {
    inner: Arc<Inner>,
    /// Keeps the handle on the `Device`'s thread (`*const ()` is neither `Send` nor `Sync`).
    _thread: std::marker::PhantomData<*const ()>,
}

impl IdleHandle {
    /// Waits until every queue of the device has finished its work (`vkDeviceWaitIdle`). The caller must not submit to
    /// the device from another thread meanwhile (Vulkan requires the queues' host access to be externally
    /// synchronised); the handle never leaves the `Device`'s thread, so no submit can overlap the wait.
    pub fn wait_idle(&self) -> Result<(), Error> {
        // SAFETY: the device is alive (this handle holds it). The handle is `!Send`, so it is on the thread that owns
        // the `Device`, the only one that submits to its queues: no submit overlaps this wait.
        unsafe { self.inner.device.device_wait_idle() }
            .map_err(vk_err("waiting for the device to go idle"))
    }
}

impl Device {
    /// Waits until every queue of the device has finished its work (`vkDeviceWaitIdle`), with no time limit: unlike
    /// `wait`, it does not give up after the usual limit, so a hung GPU holds the caller until the driver ends the work
    /// or the process is killed. `Engine`'s `Drop` uses it when a queue wait failed, so no buffer is freed under work
    /// that may still run.
    pub fn wait_idle(&self) -> Result<(), Error> {
        self.idle_handle().wait_idle()
    }

    /// A handle that waits for the device to go idle without the `Device` (see `IdleHandle`).
    pub fn idle_handle(&self) -> IdleHandle {
        IdleHandle {
            inner: self.inner.clone(),
            _thread: std::marker::PhantomData,
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: waits for all work, then destroys what this Device created. Buffers and pipelines keep `inner` alive.
        unsafe {
            let d = &self.inner.device;
            let _ = d.device_wait_idle();
            for (_, pool) in self.stamped.borrow_mut().drain(..) {
                d.destroy_query_pool(pool, None);
            }
            if let Some(t) = &self.trace {
                d.destroy_query_pool(t.pool, None);
            }
            for q in [&self.compute, &self.transfer] {
                d.destroy_command_pool(q.pool, None);
                d.destroy_semaphore(q.timeline, None);
            }
        }
    }
}

fn open_device(
    entry: &ash::Entry,
    instance: &ash::Instance,
    want: Option<[u8; 16]>,
) -> Result<Opened, Error> {
    // SAFETY: queries and creation on handles from this instance; every create-info struct outlives its call.
    unsafe {
        let pdevs = instance
            .enumerate_physical_devices()
            .map_err(vk_err("listing GPUs"))?;
        let mut cands = Vec::new();
        for &pd in &pdevs {
            let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut p11);
            instance.get_physical_device_properties2(pd, &mut p2);
            let props = p2.properties;
            cands.push(Candidate {
                name: device_name(&props),
                uuid: p11.device_uuid,
                discrete: props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
            });
        }
        let i = pick_device(&cands, want)?;
        let pd = pdevs[i];

        let exts: Vec<Vec<u8>> = instance
            .enumerate_device_extension_properties(pd)
            .map_err(vk_err("listing device extensions"))?
            .iter()
            .map(|e| {
                e.extension_name_as_c_str()
                    .map(|c| c.to_bytes().to_vec())
                    .unwrap_or_default()
            })
            .collect();
        let has = |n: &std::ffi::CStr| exts.iter().any(|e| e.as_slice() == n.to_bytes());
        let has_cm = has(ash::khr::cooperative_matrix::NAME);
        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut fcm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f11)
            .push_next(&mut f12);
        if has_cm {
            f2 = f2.push_next(&mut fcm);
        }
        instance.get_physical_device_features2(pd, &mut f2);
        let core = f2.features;
        let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut pcm = vk::PhysicalDeviceCooperativeMatrixPropertiesKHR::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut p11);
        if has_cm {
            p2 = p2.push_next(&mut pcm);
        }
        instance.get_physical_device_properties2(pd, &mut p2);
        let props = p2.properties;
        let on = |b: vk::Bool32| b == vk::TRUE;
        let support = Support {
            vulkan_1_3: props.api_version >= vk::API_VERSION_1_3,
            buffer_device_address: on(f12.buffer_device_address),
            timeline_semaphore: on(f12.timeline_semaphore),
            storage_buffer_8bit: on(f12.storage_buffer8_bit_access),
            storage_buffer_16bit: on(f11.storage_buffer16_bit_access),
            shader_float16: on(f12.shader_float16),
            shader_int8: on(f12.shader_int8),
            shader_int16: on(core.shader_int16),
            shader_int64: on(core.shader_int64),
            subgroup_arithmetic: p11
                .subgroup_supported_operations
                .contains(vk::SubgroupFeatureFlags::ARITHMETIC)
                && p11
                    .subgroup_supported_stages
                    .contains(vk::ShaderStageFlags::COMPUTE),
            subgroup_shuffle: p11
                .subgroup_supported_operations
                .contains(vk::SubgroupFeatureFlags::SHUFFLE),
            memory_budget: has(ash::ext::memory_budget::NAME),
        };
        let missing = missing_features(&support);
        if !missing.is_empty() {
            return Err(Error::MissingFeatures(missing));
        }
        // Prefill's cooperative-matrix kernels: the f16 shape, in compute shaders, with the
        // Vulkan memory model their SPIR-V declares. Without any of it, prefill runs the f32 kernels.
        let coop_f16 = has_cm
            && on(fcm.cooperative_matrix)
            && pcm
                .cooperative_matrix_supported_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
            && on(f12.vulkan_memory_model)
            && ash::khr::cooperative_matrix::Instance::new(entry, instance)
                .get_physical_device_cooperative_matrix_properties(pd)
                .is_ok_and(|s| has_f16_shape(&s));
        // A transfer-only queue can't reset a query pool, so timing staging copies needs host query reset.
        let host_reset = on(f12.host_query_reset);
        let dev_scope = on(f12.vulkan_memory_model_device_scope);

        let families = instance.get_physical_device_queue_family_properties(pd);
        let compute_family = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| Error::MissingFeatures(vec!["a compute queue"]))?
            as u32;
        let transfer_family = families
            .iter()
            .position(|f| {
                f.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !f
                        .queue_flags
                        .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
            .map_or(compute_family, |i| i as u32);
        // Profiling needs timestamps on the compute queue.
        let bits = families[compute_family as usize].timestamp_valid_bits;
        let period = props.limits.timestamp_period as f64;
        let timestamps = (bits > 0).then(|| (period, stamp_mask(bits)));
        let tbits = families[transfer_family as usize].timestamp_valid_bits;
        let transfer_stamps = (host_reset && tbits > 0).then(|| (period, stamp_mask(tbits)));

        // Device-loss markers, for debugging only: they cost about 3% of decode speed even at one a layer.
        // ARTEMIS_MARKERS=layer writes one at each layer, =dispatch one after every dispatch; otherwise none.
        let markers = std::env::var("ARTEMIS_MARKERS").unwrap_or_default();
        let has_marker =
            has(ash::amd::buffer_marker::NAME) && (markers == "layer" || markers == "dispatch");
        let marker_every_dispatch = markers == "dispatch";
        let has_coherent = has(ash::amd::device_coherent_memory::NAME);
        // The miss path's bounded spin times itself on the device clock:
        // `clockRealtimeEXT()` needs VK_KHR_shader_clock's device clock.
        let has_clock = has(ash::khr::shader_clock::NAME);
        let mut fclk = vk::PhysicalDeviceShaderClockFeaturesKHR::default();
        if has_clock {
            let mut f2c = vk::PhysicalDeviceFeatures2::default().push_next(&mut fclk);
            instance.get_physical_device_features2(pd, &mut f2c);
        }
        let shader_clock = has_clock && on(fclk.shader_device_clock);
        let mut enable = vec![ash::ext::memory_budget::NAME.as_ptr()];
        if has_marker {
            enable.push(ash::amd::buffer_marker::NAME.as_ptr());
        }
        if has_coherent {
            // For the CPU miss path.
            enable.push(ash::amd::device_coherent_memory::NAME.as_ptr());
        }
        if shader_clock {
            enable.push(ash::khr::shader_clock::NAME.as_ptr());
        }
        if coop_f16 {
            enable.push(ash::khr::cooperative_matrix::NAME.as_ptr());
        }
        // The submit trace's calibration, only when the trace is asked for: otherwise the device is opened as before.
        let calibrated = crate::trace::calibration_wanted(
            std::env::var_os(crate::trace::SUBMIT_TRACE_ENV).as_deref(),
            timestamps.is_some(),
            has(ash::khr::calibrated_timestamps::NAME),
        );
        if calibrated {
            enable.push(ash::khr::calibrated_timestamps::NAME.as_ptr());
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
        let mut e11 =
            vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut e12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .shader_int8(true)
            .shader_float16(true)
            .storage_buffer8_bit_access(true)
            .host_query_reset(host_reset)
            .vulkan_memory_model(coop_f16)
            .vulkan_memory_model_device_scope(coop_f16 && dev_scope);
        let mut eamd =
            vk::PhysicalDeviceCoherentMemoryFeaturesAMD::default().device_coherent_memory(true);
        let mut ecm =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let core_on = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .shader_int16(true);
        let mut feats = vk::PhysicalDeviceFeatures2::default()
            .features(core_on)
            .push_next(&mut e11)
            .push_next(&mut e12);
        if has_coherent {
            feats = feats.push_next(&mut eamd);
        }
        let mut eclk = vk::PhysicalDeviceShaderClockFeaturesKHR::default()
            .shader_device_clock(true)
            .shader_subgroup_clock(on(fclk.shader_subgroup_clock));
        if shader_clock {
            feats = feats.push_next(&mut eclk);
        }
        if coop_f16 {
            feats = feats.push_next(&mut ecm);
        }
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&enable)
            .push_next(&mut feats);
        let device = instance
            .create_device(pd, &info, None)
            .map_err(vk_err("creating the device"))?;
        let v = props.driver_version;
        Ok(Opened {
            pdev: pd,
            device,
            name: device_name(&props),
            driver: driver_text(v),
            uuid: p11.device_uuid,
            subgroup_size: p11.subgroup_size,
            compute_family,
            transfer_family,
            has_marker,
            marker_every_dispatch,
            timestamps,
            transfer_stamps,
            coop_f16,
            shader_clock,
            calibrated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn the_loader_is_system32s() {
        let p = system_loader().unwrap();
        assert!(p.is_absolute(), "{}", p.display());
        let t = p.to_string_lossy().to_lowercase();
        assert!(t.ends_with(r"\system32\vulkan-1.dll"), "{t}");
    }

    fn cand(name: &str, uuid: u8, discrete: bool) -> Candidate {
        Candidate {
            name: name.into(),
            uuid: [uuid; 16],
            discrete,
        }
    }

    #[test]
    fn picks_the_discrete_gpu_never_the_igpu() {
        let c = [
            cand("AMD Radeon(TM) Graphics", 1, false),
            cand("AMD Radeon RX 9070 XT", 2, true),
        ];
        assert_eq!(pick_device(&c, None).unwrap(), 1);
        assert_eq!(pick_device(&c, Some([2; 16])).unwrap(), 1);
        let e = pick_device(&c, Some([1; 16])).unwrap_err().to_string();
        assert!(e.contains("Radeon(TM) Graphics (not discrete"), "{e}");
    }

    #[test]
    fn two_discrete_gpus_need_a_uuid() {
        let c = [cand("A", 1, true), cand("B", 2, true)];
        let e = pick_device(&c, None).unwrap_err().to_string();
        assert!(e.contains("more than one discrete GPU"), "{e}");
        assert_eq!(pick_device(&c, Some([2; 16])).unwrap(), 1);
    }

    #[test]
    fn uuids_round_trip_with_or_without_dashes() {
        let u = parse_uuid("00000000-4800-0000-0000-000000000000").unwrap();
        assert_eq!(u[4], 0x48);
        assert_eq!(format_uuid(&u), "00000000-4800-0000-0000-000000000000");
        assert_eq!(parse_uuid("00000000480000000000000000000000"), Some(u));
        assert_eq!(parse_uuid("0000"), None);
        assert_eq!(parse_uuid("zz000000480000000000000000000000"), None);
        // from_str_radix takes a leading sign; a UUID digit pair never has one.
        assert_eq!(parse_uuid("+1000000480000000000000000000000"), None);
    }

    #[test]
    fn missing_features_are_listed_by_name() {
        let all = Support {
            vulkan_1_3: true,
            buffer_device_address: true,
            timeline_semaphore: true,
            storage_buffer_8bit: true,
            storage_buffer_16bit: true,
            shader_float16: true,
            shader_int8: true,
            shader_int16: true,
            shader_int64: true,
            subgroup_arithmetic: true,
            subgroup_shuffle: true,
            memory_budget: true,
        };
        assert!(missing_features(&all).is_empty());
        let some = Support {
            shader_int64: false,
            memory_budget: false,
            ..all
        };
        assert_eq!(
            missing_features(&some),
            ["shaderInt64", "VK_EXT_memory_budget"]
        );
        let e = Error::MissingFeatures(missing_features(&some)).to_string();
        assert!(e.contains("shaderInt64, VK_EXT_memory_budget"), "{e}");
    }

    #[test]
    fn the_f16_shape_is_16_by_16_by_16_with_f32_results_at_subgroup_scope() {
        let (half, full) = (vk::ComponentTypeKHR::FLOAT16, vk::ComponentTypeKHR::FLOAT32);
        let good = vk::CooperativeMatrixPropertiesKHR::default()
            .m_size(16)
            .n_size(16)
            .k_size(16)
            .a_type(half)
            .b_type(half)
            .c_type(full)
            .result_type(full)
            .scope(vk::ScopeKHR::SUBGROUP);
        assert!(has_f16_shape(&[good]));
        let f16_results = good.c_type(half).result_type(half);
        assert!(
            has_f16_shape(&[f16_results, good]),
            "listed after another shape"
        );
        assert!(!has_f16_shape(&[]));
        // Each requirement broken alone, so dropping any one check fails a case.
        let wrong = [
            (f16_results, "f16 C and results"),
            (good.m_size(8), "8 rows"),
            (good.n_size(8), "8 columns"),
            (good.k_size(8), "a k of 8"),
            (good.a_type(full), "f32 A"),
            (good.b_type(full), "f32 B"),
            (good.c_type(half), "f16 C alone"),
            (good.result_type(half), "f16 results alone"),
            (good.scope(vk::ScopeKHR::WORKGROUP), "workgroup scope"),
        ];
        for (shape, what) in wrong {
            assert!(!has_f16_shape(&[shape]), "{what}");
        }
    }
}
