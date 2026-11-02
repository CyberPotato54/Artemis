use ash::vk;
use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// The Vulkan loader (`vulkan-1.dll`) couldn't be loaded.
    Loader(String),
    /// No device fits; the message lists every device found.
    NoDevice(String),
    /// The device has no memory type of this kind.
    NoMemoryType(String),
    /// A kernel's SPIR-V is empty or malformed, as when it was built with `ARTEMIS_SKIP_SHADERS=1`.
    BadSpirv(&'static str),
    /// The device lacks features Artemis needs, by name.
    MissingFeatures(Vec<&'static str>),
    /// A kernel declares more compute shared memory per workgroup than the device allows.
    SharedMemory {
        kernel: &'static str,
        need: u32,
        limit: u32,
    },
    /// An allocation would exceed its heap's budget (`VK_EXT_memory_budget`), or the driver reported memory
    /// exhausted. `host` is true when the heap is system memory (a host-visible allocation), or the driver said the
    /// host was out of memory.
    OutOfBudget {
        what: String,
        want: u64,
        budget: u64,
        usage: u64,
        host: bool,
    },
    /// One allocation larger than Artemis allows (at most 1 GiB).
    TooLarge { what: String, size: u64 },
    /// The GPU was lost. `in_flight` names the first dispatch that hadn't finished, when the driver writes markers.
    DeviceLost { in_flight: Option<String> },
    /// A submission didn't finish within the wait limit.
    Timeout,
    /// `Device::record_profiled` on a GPU whose compute queue has no timestamps.
    NoTimestamps,
    /// Any other Vulkan failure, with what was being done.
    Vk {
        what: &'static str,
        result: vk::Result,
    },
}

const MIB: u64 = 1 << 20;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Loader(e) => write!(f, "cannot load the Vulkan loader: {e}"),
            Error::NoDevice(found) => write!(f, "no usable discrete GPU; found: {found}"),
            Error::NoMemoryType(kind) => write!(f, "the GPU has no {kind} memory type"),
            Error::BadSpirv(k) => write!(
                f,
                "kernel {k}: its SPIR-V is empty or malformed; unset ARTEMIS_SKIP_SHADERS and rebuild (the Vulkan SDK is needed only if a shader changed)"
            ),
            Error::MissingFeatures(m) => {
                write!(f, "the GPU lacks required features: {}", m.join(", "))
            }
            Error::SharedMemory {
                kernel,
                need,
                limit,
            } => write!(
                f,
                "kernel {kernel} needs {need} bytes of compute shared memory per workgroup; the GPU offers {limit}"
            ),
            Error::OutOfBudget {
                what,
                want,
                budget,
                usage,
                ..
            } => write!(
                f,
                "{what} needs {} MiB, but its heap has {} MiB left of a {} MiB budget",
                want.div_ceil(MIB),
                budget.saturating_sub(*usage) / MIB,
                budget / MIB
            ),
            Error::TooLarge { what, size } => write!(
                f,
                "{what} is {} MiB; Artemis allocates at most {} MiB at a time",
                size.div_ceil(MIB),
                crate::MAX_ALLOCATION / MIB
            ),
            Error::DeviceLost { in_flight: Some(k) } => {
                write!(f, "the GPU was lost (VK_ERROR_DEVICE_LOST) while running {k}")
            }
            Error::DeviceLost { in_flight: None } => {
                write!(
                    f,
                    "the GPU was lost (VK_ERROR_DEVICE_LOST); to name the work in flight on an AMD driver, run with \
                     ARTEMIS_MARKERS=layer or ARTEMIS_MARKERS=dispatch"
                )
            }
            Error::Timeout => write!(
                f,
                "a GPU submission didn't finish within {} s",
                crate::record::WAIT_NS / 1_000_000_000
            ),
            Error::NoTimestamps => write!(
                f,
                "the GPU's compute queue has no timestamps, so its work can't be profiled"
            ),
            Error::Vk { what, result } => write!(f, "{what}: {result}"),
        }
    }
}

impl std::error::Error for Error {}

impl Error {
    /// Whether this failure is system memory running out: a budget refusal of a host-visible allocation, or the
    /// driver's `ERROR_OUT_OF_HOST_MEMORY`. The worker does not retry such a load with a smaller expert cache, which
    /// lives in VRAM.
    pub fn is_host_memory(&self) -> bool {
        matches!(
            self,
            Error::OutOfBudget { host: true, .. }
                | Error::Vk {
                    result: vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                    ..
                }
        )
    }

    /// Whether this failure is memory running out: the budget refusing an allocation, or the driver reporting that
    /// device or host memory is exhausted.
    pub fn is_out_of_memory(&self) -> bool {
        matches!(
            self,
            Error::OutOfBudget { .. }
                | Error::Vk {
                    result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                        | vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                    ..
                }
        )
    }
}

/// Maps a Vulkan result to an error, keeping device loss typed.
pub(crate) fn vk_err(what: &'static str) -> impl Fn(vk::Result) -> Error {
    move |result| {
        if result == vk::Result::ERROR_DEVICE_LOST {
            Error::DeviceLost { in_flight: None }
        } else {
            Error::Vk { what, result }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_memory_is_told_from_device_memory() {
        let budget = |host| Error::OutOfBudget {
            what: "x".into(),
            want: 1,
            budget: 0,
            usage: 0,
            host,
        };
        assert!(budget(true).is_host_memory());
        assert!(!budget(false).is_host_memory());
        // Both kinds are still memory running out.
        assert!(budget(true).is_out_of_memory() && budget(false).is_out_of_memory());
        let code = |result| Error::Vk { what: "x", result };
        assert!(code(vk::Result::ERROR_OUT_OF_HOST_MEMORY).is_host_memory());
        assert!(!code(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY).is_host_memory());
        assert!(!Error::Timeout.is_host_memory());
    }

    #[test]
    fn out_of_memory_covers_the_budget_and_the_two_vulkan_codes() {
        let oob = Error::OutOfBudget {
            what: "x".into(),
            want: 1,
            budget: 0,
            usage: 0,
            host: false,
        };
        assert!(oob.is_out_of_memory());
        for result in [
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        ] {
            assert!(Error::Vk { what: "x", result }.is_out_of_memory());
        }
        let unknown = Error::Vk {
            what: "x",
            result: vk::Result::ERROR_UNKNOWN,
        };
        let too_large = Error::TooLarge {
            what: "x".into(),
            size: 1,
        };
        let lost = Error::DeviceLost { in_flight: None };
        for e in [unknown, too_large, Error::Timeout, lost] {
            assert!(!e.is_out_of_memory(), "{e}");
        }
    }
}
