//! The submit trace, a measurement aid: with `ARTEMIS_SUBMIT_TRACE` set when the device opens (any value but empty;
//! `artemis generate` takes it as the file to write), every compute submit through `Device::submit` from a plain
//! recorder (`Device::record`, never a profiled one) carries two timestamps, one before its first command and one after
//! its last, and logs when the host recorded, submitted and woke. Each submit's two timestamps are mapped onto the host
//! clock by a calibration (`VK_KHR_calibrated_timestamps`) taken right after the host wakes, so the time from the call
//! to vkQueueSubmit to the GPU's start and from the GPU's end to the host's wake can be read off. Unset, the device is
//! opened exactly as without this module (the extension isn't enabled), no command is added and no clock is read.
//!
//! The log is bounded: it holds at most `MAX_TRACE_ENTRIES` submits (about 16 MiB), and a submit past that is counted
//! (`Device::take_submit_trace_dropped`) and not kept, so a long run can't grow it without limit. A traced submit whose
//! timestamps can't be read or whose clock can't be calibrated is counted the same way and reported once on stderr;
//! the submit itself has finished and is not failed.
use crate::{device::Device, error::vk_err, error::Error};
use ash::vk;
use std::{
    cell::{Cell, RefCell},
    ffi::OsStr,
    time::Instant,
};

/// The environment variable that turns the trace on.
pub const SUBMIT_TRACE_ENV: &str = "ARTEMIS_SUBMIT_TRACE";

/// The most submits the log holds (64 bytes each, about 16 MiB); a longer run counts the rest as dropped. A 700-token
/// speculative run is a few tens of thousands of submits.
pub const MAX_TRACE_ENTRIES: usize = 1 << 18;

/// Whether a value of `ARTEMIS_SUBMIT_TRACE` turns the trace on: any value but an empty one.
pub fn submit_trace_wanted(v: Option<&OsStr>) -> bool {
    v.is_some_and(|v| !v.is_empty())
}

/// Whether the device opens with `VK_KHR_calibrated_timestamps` enabled: the trace is asked for (`env`), the device
/// has timestamps, and the driver offers the extension. With the trace off the answer is no whatever the driver has,
/// so an untraced device is opened exactly as it was before the trace existed.
pub(crate) fn calibration_wanted(
    env: Option<&OsStr>,
    has_timestamps: bool,
    has_extension: bool,
) -> bool {
    submit_trace_wanted(env) && has_timestamps && has_extension
}

/// One traced compute submit. Host times are ns since the device opened; the GPU's two are on the same clock, through
/// the calibration, to within `cal_err_ns`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubmitTrace {
    /// What the caller named the submit (`Device::set_trace_label`); 0 when it named nothing.
    pub label: u32,
    /// When the recorder was made (`Device::record`).
    pub record_ns: i64,
    /// Just before vkQueueSubmit.
    pub submit_ns: i64,
    /// Just after vkQueueSubmit returned.
    pub submitted_ns: i64,
    /// When the host's wait for the submit returned.
    pub woke_ns: i64,
    /// The timestamp before the first command (top of pipe, once the submit's semaphore waits are met).
    pub gpu_start_ns: i64,
    /// The timestamp after the last command (bottom of pipe).
    pub gpu_end_ns: i64,
    /// The mapping's uncertainty: half the calibration call's host span plus the driver's maximum deviation.
    pub cal_err_ns: i64,
}

/// The device's side of the trace: the timestamps' pool (two queries, reset at the start of each traced command buffer;
/// every submit waits, so one pool serves them all), the calibration, and the log.
pub(crate) struct Trace {
    pub(crate) pool: vk::QueryPool,
    cal: ash::khr::calibrated_timestamps::Device,
    /// The device domain first, then the host's when the driver offers one (it only makes the deviation meaningful).
    domains: Vec<vk::TimeDomainKHR>,
    origin: Instant,
    label: Cell<u32>,
    log: RefCell<Vec<SubmitTrace>>,
    /// Traced submits not kept: the log was full, or reading or calibrating them failed.
    dropped: Cell<u64>,
}

impl Trace {
    /// The trace for a device opened with the calibration extension enabled, or None when the driver can't calibrate
    /// the device clock or the pool can't be made.
    pub(crate) fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        pdev: vk::PhysicalDevice,
        device: &ash::Device,
    ) -> Option<Trace> {
        let ci = ash::khr::calibrated_timestamps::Instance::new(entry, instance);
        // SAFETY: a plain query on this instance's physical device.
        let have = unsafe { ci.get_physical_device_calibrateable_time_domains(pdev) }.ok()?;
        if !have.contains(&vk::TimeDomainKHR::DEVICE) {
            return None;
        }
        let mut domains = vec![vk::TimeDomainKHR::DEVICE];
        let host = if cfg!(windows) {
            vk::TimeDomainKHR::QUERY_PERFORMANCE_COUNTER
        } else {
            vk::TimeDomainKHR::CLOCK_MONOTONIC
        };
        if have.contains(&host) {
            domains.push(host);
        }
        // SAFETY: the create info outlives the call; the pool is destroyed with the device.
        let pool = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
        }
        .ok()?;
        Some(Trace {
            pool,
            cal: ash::khr::calibrated_timestamps::Device::new(instance, device),
            domains,
            origin: Instant::now(),
            label: Cell::new(0),
            log: RefCell::new(Vec::new()),
            dropped: Cell::new(0),
        })
    }

    fn ns(&self, t: Instant) -> i64 {
        t.duration_since(self.origin).as_nanos() as i64
    }
}

/// A traced submit's host instants: the recorder's creation, before and after vkQueueSubmit, and the wake.
pub(crate) struct HostTimes {
    pub(crate) record: Instant,
    pub(crate) submit: Instant,
    pub(crate) submitted: Instant,
    pub(crate) woke: Instant,
}

impl Device {
    /// Whether compute submits are traced (`ARTEMIS_SUBMIT_TRACE` was set when the device opened, and the driver can
    /// calibrate the device clock).
    pub fn submit_trace_on(&self) -> bool {
        self.trace.is_some()
    }

    /// Whether `VK_KHR_calibrated_timestamps` was enabled when the device opened: only when the trace was asked for.
    pub fn clock_calibration_enabled(&self) -> bool {
        self.calibration_enabled
    }

    /// Names the next traced submit (`SubmitTrace::label`); each submit takes the label and leaves 0. Nothing while the
    /// trace is off.
    pub fn set_trace_label(&self, label: u32) {
        if let Some(t) = &self.trace {
            t.label.set(label);
        }
    }

    /// The traced submits since the device opened or this was last called; empty while the trace is off.
    pub fn take_submit_trace(&self) -> Vec<SubmitTrace> {
        self.trace
            .as_ref()
            .map(|t| std::mem::take(&mut *t.log.borrow_mut()))
            .unwrap_or_default()
    }

    /// The number of traced submits that were not kept since the device opened or this was last called: the log was
    /// full (`MAX_TRACE_ENTRIES`), or reading their timestamps or calibrating the clock failed. 0 while the trace is
    /// off.
    pub fn take_submit_trace_dropped(&self) -> u64 {
        self.trace.as_ref().map_or(0, |t| t.dropped.replace(0))
    }

    /// In a traced recorder, resets the pool and writes the first timestamp; returns when the recorder was made.
    pub(crate) fn trace_begin(&self, cmd: vk::CommandBuffer) -> Option<Instant> {
        let t = self.trace.as_ref()?;
        let d = &self.inner.device;
        // SAFETY: cmd is recording; the pool holds two queries, and no submit that wrote them is still running.
        unsafe {
            d.cmd_reset_query_pool(cmd, t.pool, 0, 2);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, t.pool, 0);
        }
        Some(Instant::now())
    }

    /// Writes a traced recorder's last timestamp, after every command it holds.
    pub(crate) fn trace_end(&self, cmd: vk::CommandBuffer) {
        if let Some(t) = &self.trace {
            // SAFETY: cmd is recording; query 1 was reset at its start.
            unsafe {
                self.inner.device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    t.pool,
                    1,
                )
            };
        }
    }

    /// Logs a traced submit that has finished. The submit's work is done, so a failure here (reading the timestamps,
    /// calibrating the clock) is reported once on stderr and counted as dropped, never returned: the step succeeded.
    pub(crate) fn trace_log(&self, h: HostTimes) -> Result<(), Error> {
        let Some(t) = &self.trace else {
            return Ok(());
        };
        if t.log.borrow().len() >= MAX_TRACE_ENTRIES {
            t.dropped.set(t.dropped.get() + 1);
            return Ok(());
        }
        match self.trace_entry(t, &h) {
            Ok(entry) => t.log.borrow_mut().push(entry),
            Err(e) => {
                if t.dropped.get() == 0 {
                    eprintln!(
                        "submit trace: an entry was dropped ({e}); later ones are only counted"
                    );
                }
                t.dropped.set(t.dropped.get() + 1);
            }
        }
        Ok(())
    }

    /// Reads a finished traced submit's two timestamps and maps them onto the host clock.
    fn trace_entry(&self, t: &Trace, h: &HostTimes) -> Result<SubmitTrace, Error> {
        let (period, mask) = self
            .timestamps
            .expect("the trace is on only with timestamps");
        let d = &self.inner.device;
        let mut raw = [0u64; 2];
        // SAFETY: the submit has finished, so both queries have landed.
        unsafe {
            d.get_query_pool_results(
                t.pool,
                0,
                &mut raw,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .map_err(vk_err("reading the trace's timestamps"))?;
        let infos: Vec<_> = t
            .domains
            .iter()
            .map(|&dom| vk::CalibratedTimestampInfoKHR::default().time_domain(dom))
            .collect();
        let before = Instant::now();
        // SAFETY: the infos outlive the call; the domains are ones the driver listed.
        let (stamps, dev_err) = unsafe { t.cal.get_calibrated_timestamps(&infos) }
            .map_err(vk_err("calibrating the trace's clock"))?;
        let after = Instant::now();
        let span = after.duration_since(before).as_nanos() as i64;
        let cal_host = t.ns(before) + span / 2;
        // A stamp's host time: the calibration's, less the ticks between them (masked, so a wrap reads right).
        let host_of = |raw: u64| {
            let ticks = stamps[0].wrapping_sub(raw) & mask;
            cal_host - (ticks as f64 * period).round() as i64
        };
        Ok(SubmitTrace {
            label: t.label.replace(0),
            record_ns: t.ns(h.record),
            submit_ns: t.ns(h.submit),
            submitted_ns: t.ns(h.submitted),
            woke_ns: t.ns(h.woke),
            gpu_start_ns: host_of(raw[0]),
            gpu_end_ns: host_of(raw[1]),
            cal_err_ns: span / 2 + dev_err as i64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_trace_is_off_unless_the_variable_has_a_value() {
        assert!(!submit_trace_wanted(None));
        assert!(!submit_trace_wanted(Some(OsStr::new(""))));
        assert!(submit_trace_wanted(Some(OsStr::new("trace.json"))));
    }

    #[test]
    fn the_clock_calibration_is_enabled_only_when_the_trace_is_asked_for() {
        let on = Some(OsStr::new("trace.json"));
        // Off: never, whatever the driver offers, so the device opens as before.
        assert!(!calibration_wanted(None, true, true));
        assert!(!calibration_wanted(Some(OsStr::new("")), true, true));
        // On: only with timestamps and the extension.
        assert!(calibration_wanted(on, true, true));
        assert!(!calibration_wanted(on, false, true));
        assert!(!calibration_wanted(on, true, false));
    }
}
