//! A thin layer over `ash` for Artemis. It picks the discrete GPU and checks its features,
//! allocates within the driver's memory budget, gives every buffer a device address, builds compute pipelines that take
//! those addresses through push constants (there are no descriptor sets), records command buffers, and submits them on
//! a compute and a transfer queue, each with a timeline semaphore. Device loss is a typed error.
mod device;
mod error;
mod memory;
mod record;
mod trace;

pub use device::{
    format_uuid, has_f16_shape, limit_dll_search_to_system32, list_gpus, missing_features,
    parse_uuid, pick_device, Candidate, Device, GpuInfo, HeapBudget, IdleHandle, Support,
};
pub use error::Error;
pub use memory::{planned_bytes, select_type, Arena, Buffer, MemKind, Slice, MAX_ALLOCATION};
pub use record::{
    CopySpan, Pipeline, PushConstants, QueueKind, Recorder, StageOpts, Timing, MAX_WORKGROUPS,
    PROFILE_STAMPS, UNTAGGED,
};
pub use trace::{submit_trace_wanted, SubmitTrace, MAX_TRACE_ENTRIES, SUBMIT_TRACE_ENV};
