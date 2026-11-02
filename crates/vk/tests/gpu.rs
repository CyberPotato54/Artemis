//! The vk layer on the test PC's GPU: device choice, the budget, uploads through staging, copies, fills and readback.
use vk::{Arena, Device, MemKind, QueueKind, StageOpts};

#[test]
#[ignore = "needs the GPU"]
fn opens_the_discrete_gpu_and_reports_a_budget() {
    let dev = Device::open(None).unwrap();
    eprintln!(
        "{} ({}), uuid {}, subgroup {}",
        dev.name,
        dev.driver,
        vk::format_uuid(&dev.uuid),
        dev.subgroup_size
    );
    assert!(dev.subgroup_size == 32 || dev.subgroup_size == 64);
    assert!(dev.free_bytes(MemKind::Vram).unwrap() > 1 << 30);
    assert!(dev.free_bytes(MemKind::HostPinned).unwrap() > 1 << 30);
    // The same device by its UUID.
    assert_eq!(Device::open(Some(dev.uuid)).unwrap().uuid, dev.uuid);
}

#[test]
#[ignore = "needs the GPU"]
fn upload_copy_fill_and_readback_round_trip() {
    let dev = Device::open(None).unwrap();
    let mut vram = Arena::new(MemKind::Vram, 256 << 20, "test");
    // 150 MiB crosses the 64 MiB upload staging twice.
    let n = 150 << 20;
    let data: Vec<u8> = (0..n).map(|i| (i * 7 + i / 4093) as u8).collect();
    let a = vram.alloc(&dev, n as u64).unwrap();
    dev.upload(a, &data).unwrap();
    let b = vram.alloc(&dev, n as u64).unwrap();
    assert_eq!(b.addr % 256, 0);
    assert!(b.addr >= a.addr + n as u64 || b.addr + n as u64 <= a.addr);
    let mut r = dev.record().unwrap();
    r.copy(a, b);
    r.barrier();
    r.fill(b.sub(0, 4096), 0xDEAD_BEEF);
    dev.submit(r).unwrap();
    let back = dev.download(b).unwrap();
    assert!(back[..4096]
        .as_chunks::<4>()
        .0
        .iter()
        .all(|c| u32::from_le_bytes(*c) == 0xDEAD_BEEF));
    assert!(
        back[4096..] == data[4096..],
        "the copy differs from the upload"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn allocations_over_a_gib_are_refused() {
    let dev = Device::open(None).unwrap();
    let e = dev
        .alloc((1 << 30) + 1, MemKind::Vram, "a huge tensor")
        .err()
        .unwrap();
    assert!(matches!(e, vk::Error::TooLarge { .. }), "{e}");
}

#[test]
#[ignore = "needs the GPU"]
fn an_async_copy_lands_and_the_next_compute_submit_sees_it() {
    let dev = Device::open(None).unwrap();
    let n = 4 << 20;
    let data: Vec<u8> = (0..n).map(|i| (i * 13 + i / 977) as u8).collect();
    let src = dev
        .alloc(n as u64, MemKind::HostPinned, "test source")
        .unwrap();
    src.write(0, &data);
    let mut vram = Arena::new(MemKind::Vram, 64 << 20, "test");
    let dsts: Vec<vk::Slice> = (0..3)
        .map(|_| vram.alloc(&dev, n as u64).unwrap())
        .collect();
    let before = dev.submitted(QueueKind::Transfer);
    // Three copies in flight at once, none waited for: each signals the next transfer value.
    let values: Vec<u64> = dsts
        .iter()
        .map(|d| dev.copy_async(src.whole(), *d, None).unwrap())
        .collect();
    assert_eq!(values, [before + 1, before + 2, before + 3]);
    assert_eq!(dev.submitted(QueueKind::Transfer), before + 3);
    // A compute submit waits for what the host has seen the transfer queue finish, so once the host has waited for
    // the copies it reads what they wrote.
    let rb = dev
        .alloc(n as u64, MemKind::Readback, "test readback")
        .unwrap();
    dev.wait(QueueKind::Transfer, values[2]).unwrap();
    let mut r = dev.record().unwrap();
    r.copy(dsts[2], rb.whole());
    r.host_barrier();
    dev.submit(r).unwrap();
    assert!(dev.completed(QueueKind::Transfer).unwrap() >= values[2]);
    let mut back = vec![0u8; n];
    rb.read(0, &mut back);
    assert!(back == data, "the compute submit didn't see the async copy");
    for d in &dsts {
        assert!(dev.download(*d).unwrap() == data);
    }
    assert_eq!(
        dev.completed(QueueKind::Compute).unwrap(),
        dev.submitted(QueueKind::Compute)
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_compute_submit_doesnt_wait_for_a_copy_in_flight() {
    // A step no longer waits for a swap's copy it doesn't read. 256 MiB from pinned memory on
    // the transfer queue takes about 19 ms (measured, about 14 GB/s); a 4 KiB fill on the compute queue, about 0.15 ms.
    let dev = Device::open(None).unwrap();
    let n = 256u64 << 20;
    let src = dev.alloc(n, MemKind::HostPinned, "test source").unwrap();
    let mut vram = Arena::new(MemKind::Vram, 300 << 20, "test");
    let dst = vram.alloc(&dev, n).unwrap();
    let small = vram.alloc(&dev, 4096).unwrap();
    let v = dev.copy_async(src.whole(), dst, None).unwrap();
    let mut r = dev.record().unwrap();
    r.fill(small, 1);
    dev.submit(r).unwrap();
    let seen = dev.completed(QueueKind::Transfer).unwrap();
    // The copy finishes before the assert, so a failure doesn't free the arena under it.
    dev.wait(QueueKind::Transfer, v).unwrap();
    assert!(
        seen < v,
        "the compute submit waited for the copy in flight: transfer value {seen} seen after it, the copy's {v}"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_poisoned_copy_lands_the_source_not_the_poison() {
    let dev = Device::open(None).unwrap();
    let n = 1 << 20;
    let data: Vec<u8> = (0..n).map(|i| (i * 31 + i / 1021) as u8).collect();
    let src = dev
        .alloc(n as u64, MemKind::HostPinned, "test source")
        .unwrap();
    src.write(0, &data);
    let mut vram = Arena::new(MemKind::Vram, 4 << 20, "test");
    let dst = vram.alloc(&dev, n as u64).unwrap();
    let v = dev.copy_async(src.whole(), dst, Some(0x7E00_7E00)).unwrap();
    dev.wait(QueueKind::Transfer, v).unwrap();
    assert!(
        dev.download(dst).unwrap() == data,
        "the poison outlived the copy"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_profiled_submit_times_each_command_under_its_tag() {
    let dev = Device::open(None).unwrap();
    assert!(
        dev.can_profile(),
        "the RX 9070 XT's compute queue has timestamps"
    );
    let mut vram = Arena::new(MemKind::Vram, 256 << 20, "test");
    let n = 256u64 << 20;
    let (a, b) = (vram.alloc(&dev, n).unwrap(), vram.alloc(&dev, n).unwrap());
    let mut r = dev.record_profiled().unwrap();
    r.fill(a, 7);
    r.barrier();
    r.tag(3);
    r.copy(a, b);
    r.barrier();
    r.fill(b.sub(0, 4096), 9);
    let t = dev.submit_profiled(r).unwrap();
    let labels: Vec<(&str, u32)> = t.iter().map(|t| (t.name, t.tag)).collect();
    assert_eq!(labels, [("fill", vk::UNTAGGED), ("copy", 3), ("fill", 3)]);
    // A 256 MiB copy within VRAM moves 512 MiB: at least 0.8 ms at 640 GB/s, and far under a second.
    assert!(t[1].ns > 800_000 && t[1].ns < 1_000_000_000, "{t:?}");
    // Profiling changed nothing the commands did.
    assert_eq!(dev.download(b.sub(4096, 4)).unwrap(), 7u32.to_le_bytes());
    assert_eq!(dev.download(b.sub(0, 4)).unwrap(), 9u32.to_le_bytes());
}

#[test]
#[ignore = "needs the GPU"]
fn an_unprofiled_recorder_gives_no_timings_and_a_profiled_one_submits_plainly() {
    let dev = Device::open(None).unwrap();
    let buf = dev.alloc(4096, MemKind::Vram, "test").unwrap();
    let mut r = dev.record().unwrap();
    r.tag(5);
    r.fill(buf.whole(), 1);
    assert!(dev.submit_profiled(r).unwrap().is_empty());
    let mut r = dev.record_profiled().unwrap();
    r.fill(buf.whole(), 2);
    dev.submit(r).unwrap();
    // A profiled recorder dropped unsubmitted frees its query pool and command buffer, and runs nothing.
    let mut r = dev.record_profiled().unwrap();
    r.fill(buf.whole(), 3);
    drop(r);
    assert_eq!(dev.download(buf.whole()).unwrap()[..4], 2u32.to_le_bytes());
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "a profiled command buffer times at most 8191 commands")]
fn a_profiled_recorder_refuses_more_commands_than_it_can_time() {
    let dev = Device::open(None).unwrap();
    let buf = dev.alloc(4096, MemKind::Vram, "test").unwrap();
    let mut r = dev.record_profiled().unwrap();
    for _ in 0..vk::PROFILE_STAMPS {
        r.fill(buf.whole(), 0);
    }
}

#[test]
#[ignore = "needs the GPU"]
fn a_profiled_recorder_times_its_full_8191_commands() {
    let dev = Device::open(None).unwrap();
    let buf = dev.alloc(4096, MemKind::Vram, "test").unwrap();
    let mut r = dev.record_profiled().unwrap();
    for _ in 0..vk::PROFILE_STAMPS - 1 {
        r.fill(buf.whole(), 0);
    }
    let t = dev.submit_profiled(r).unwrap();
    assert_eq!(t.len(), 8191);
    assert!(t.iter().all(|t| t.name == "fill"));
}

#[test]
#[ignore = "needs the GPU"]
fn a_compute_submit_that_waits_for_a_staging_copy_reads_what_it_wrote() {
    // Layer l's compute waits for layer l's staging copy, still in flight when the compute is
    // submitted. 256 MiB from pinned memory takes about 18 ms at 14.5 GB/s; the compute submit's copy of the last
    // 4 KiB would run at once without the wait. The staging copy poisons its destination first, so an early read sees
    // the poison (or whatever VRAM held), never the tail.
    let dev = Device::open(None).unwrap();
    let n = 256u64 << 20;
    let src = dev.alloc(n, MemKind::HostPinned, "test source").unwrap();
    let tail: Vec<u8> = (0..4096u32).map(|i| (i * 13 + 7) as u8).collect();
    src.write(n - 4096, &tail);
    let mut vram = Arena::new(MemKind::Vram, 300 << 20, "test");
    let dst = vram.alloc(&dev, n).unwrap();
    let out = vram.alloc(&dev, 4096).unwrap();
    let opts = StageOpts {
        poison: Some(0x7E00_7E00),
        ..StageOpts::default()
    };
    let v = dev.stage_async(&[(src.whole(), dst)], 0, opts).unwrap();
    // `out` is read back right after the compute submit, before the host waits for the copy; the results are kept, not
    // unwrapped, until the copy has finished, so a failure doesn't free the arena under it.
    let got = dev
        .record()
        .and_then(|mut r| {
            r.wait(QueueKind::Transfer, v);
            r.copy(dst.sub(n - 4096, 4096), out);
            dev.submit(r)
        })
        .and_then(|()| dev.download(out));
    dev.wait(QueueKind::Transfer, v).unwrap();
    assert!(
        got.unwrap() == tail,
        "the compute submit read the staging buffer before the copy landed"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn staged_regions_land_where_they_were_sent() {
    // Five regions from two pinned buffers into one VRAM buffer, out of order, poisoned first and delayed behind a
    // 64 MiB fill: every region lands with its source's bytes, and a gap between regions keeps what was there.
    let dev = Device::open(None).unwrap();
    let pinned = |mul: u32| {
        let data: Vec<u8> = (0..1u32 << 20)
            .map(|i| (i.wrapping_mul(mul) >> 7) as u8)
            .collect();
        let b = dev
            .alloc(1 << 20, MemKind::HostPinned, "test source")
            .unwrap();
        b.write(0, &data);
        (b, data)
    };
    let ((a, da), (b, db)) = (pinned(2_654_435_761), pinned(40_503));
    let mut vram = Arena::new(MemKind::Vram, 128 << 20, "test");
    let dst = vram.alloc(&dev, 1 << 20).unwrap();
    let delay = vram.alloc(&dev, 64 << 20).unwrap();
    let mut r = dev.record().unwrap();
    r.fill(dst, 0x1111_1111);
    dev.submit(r).unwrap();
    let regions = [
        (a.slice(0, 4096), dst.sub(65_536, 4096)),
        (a.slice(8192, 4096), dst.sub(0, 4096)),
        (b.slice(4096, 8192), dst.sub(16_384, 8192)),
        (a.slice(100_000, 512), dst.sub(200_000, 512)),
        (b.slice(0, 256), dst.sub(300_000, 256)),
    ];
    let opts = StageOpts {
        poison: Some(0x7E00_7E00),
        delay: Some(delay),
        timed: false,
    };
    let v = dev
        .stage_async(&regions, dev.submitted(QueueKind::Compute), opts)
        .unwrap();
    dev.wait(QueueKind::Transfer, v).unwrap();
    let got = dev.download(dst).unwrap();
    for (i, (s, d)) in regions.iter().enumerate() {
        let data = if s.buffer == a.whole().buffer {
            &da
        } else {
            &db
        };
        let at = (d.offset - dst.offset) as usize;
        assert!(
            got[at..at + d.size as usize] == data[s.offset as usize..(s.offset + s.size) as usize],
            "region {i}"
        );
    }
    assert!(
        got[4096..8192]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|c| u32::from_le_bytes(*c) == 0x1111_1111),
        "a gap between regions was written"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn a_timed_copy_reports_its_span_on_the_compute_queues_clock() {
    // The profiler shows overlap by comparing staging copies' spans with compute submits'. A
    // 256 MiB copy (about 18 ms) and a short profiled compute submit sent right after it: the compute submit's span lies
    // inside the copy's, so the two queues' timestamps share a clock, and the copy overlapped the compute.
    let dev = Device::open(None).unwrap();
    assert!(dev.can_time_copies() && dev.can_profile());
    let n = 256u64 << 20;
    let src = dev.alloc(n, MemKind::HostPinned, "test source").unwrap();
    let mut vram = Arena::new(MemKind::Vram, 300 << 20, "test");
    let dst = vram.alloc(&dev, n).unwrap();
    let small = vram.alloc(&dev, 1 << 20).unwrap();
    let opts = StageOpts {
        timed: true,
        ..StageOpts::default()
    };
    let v = dev.stage_async(&[(src.whole(), dst)], 0, opts).unwrap();
    // The results are kept, not unwrapped, until the copy has finished, so a failure doesn't free the arena under it.
    let profiled = dev.record_profiled().and_then(|mut r| {
        r.fill(small, 1);
        dev.submit_profiled(r)
    });
    let span = dev.last_span();
    dev.wait(QueueKind::Transfer, v).unwrap();
    profiled.unwrap();
    let (c0, c1) = span.expect("a profiled submit's span");
    let spans = dev.copy_spans().unwrap();
    assert_eq!(spans.len(), 1, "{spans:?}");
    let s = spans[0];
    assert_eq!(s.value, v);
    assert!(
        s.end_ns.saturating_sub(s.start_ns) > 5_000_000,
        "a 256 MiB copy took under 5 ms: {s:?}"
    );
    assert!(
        s.start_ns <= c0 && c1 <= s.end_ns,
        "the compute submit [{c0}, {c1}] ran outside the copy {s:?}"
    );
    assert!(
        dev.copy_spans().unwrap().is_empty(),
        "each span is reported once"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn the_rx_9070_xt_runs_f16_cooperative_matrices() {
    // The device lists the shape; prefill's split-f16 kernels need it enabled.
    assert!(Device::open(None).unwrap().coop_f16());
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "a compute submit waits only for the transfer queue")]
fn a_compute_recorder_refuses_to_wait_for_the_compute_queue() {
    let dev = Device::open(None).unwrap();
    let mut r = dev.record().unwrap();
    r.wait(QueueKind::Compute, 0);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "a wait for transfer value 2, which was never submitted (the last is 1)")]
fn a_recorder_refuses_to_wait_for_a_transfer_value_never_submitted() {
    // The last value submitted is a valid wait; the one after it would never be signalled. The copy has finished
    // before the panic, so nothing is in flight.
    let dev = Device::open(None).unwrap();
    let src = dev.alloc(4096, MemKind::HostPinned, "test source").unwrap();
    let dst = dev.alloc(4096, MemKind::Vram, "test").unwrap();
    let v = dev.copy_async(src.whole(), dst.whole(), None).unwrap();
    dev.wait(QueueKind::Transfer, v).unwrap();
    let mut r = dev.record().unwrap();
    r.wait(QueueKind::Transfer, v);
    r.wait(QueueKind::Transfer, v + 1);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(
    expected = "a staging copy waits for compute value 2, which was never submitted (the last is 1)"
)]
fn a_staging_copy_refuses_to_wait_for_a_compute_value_never_submitted() {
    // The last compute value submitted is a valid `after`; the one after it would never be signalled. The first copy
    // has finished before the panic, and the refused one is never recorded, so nothing is in flight.
    let dev = Device::open(None).unwrap();
    let src = dev.alloc(4096, MemKind::HostPinned, "test source").unwrap();
    let dst = dev.alloc(4096, MemKind::Vram, "test").unwrap();
    let mut r = dev.record().unwrap();
    r.fill(dst.whole(), 0);
    dev.submit(r).unwrap();
    let last = dev.submitted(QueueKind::Compute);
    let regions = [(src.whole(), dst.whole())];
    let v = dev
        .stage_async(&regions, last, StageOpts::default())
        .unwrap();
    dev.wait(QueueKind::Transfer, v).unwrap();
    let _ = dev.stage_async(&regions, last + 1, StageOpts::default());
}

#[test]
#[ignore = "needs the GPU"]
fn a_vram_to_host_copy_on_the_transfer_queue_lands() {
    // Snapshots copy KV from VRAM into host-cached memory on the transfer queue; nothing else
    // sends VRAM to host memory that way (`download` uses the compute queue). 1 MiB each way. The VRAM pattern is
    // written by a compute submit, so the copy's `after` is a real compute value and it reads compute-written VRAM;
    // the host reads the result as `park` does, after a compute submit that waits for the transfer value and holds only
    // a host barrier. The way back is checked with `download`.
    let dev = Device::open(None).unwrap();
    let n = 1u64 << 20;
    let pattern = |k: u32| -> Vec<u8> {
        (0..n as u32)
            .map(|i| (i.wrapping_mul(k) >> 3) as u8)
            .collect()
    };
    let (down, up) = (pattern(2_654_435_761), pattern(40_503));
    assert!(down != up);
    let vram = dev.alloc(n, MemKind::Vram, "test source").unwrap();
    let host = dev.alloc(n, MemKind::HostCached, "test host").unwrap();
    host.write(0, &down);
    let mut r = dev.record().unwrap();
    r.copy(host.whole(), vram.whole());
    dev.submit(r).unwrap();
    let after = dev.submitted(QueueKind::Compute);
    assert!(after > 0);
    host.write(0, &vec![0u8; n as usize]);
    let v = dev
        .stage_async(&[(vram.whole(), host.whole())], after, StageOpts::default())
        .unwrap();
    dev.wait(QueueKind::Transfer, v).unwrap();
    let mut r = dev.record().unwrap();
    r.wait(QueueKind::Transfer, v);
    r.host_barrier();
    dev.submit(r).unwrap();
    let mut got = vec![0u8; n as usize];
    host.read(0, &mut got);
    assert!(got == down, "the VRAM bytes did not reach the host buffer");
    // And back: the host buffer to VRAM on the same queue.
    host.write(0, &up);
    let v = dev
        .stage_async(
            &[(host.whole(), vram.whole())],
            dev.submitted(QueueKind::Compute),
            StageOpts::default(),
        )
        .unwrap();
    dev.wait(QueueKind::Transfer, v).unwrap();
    assert!(
        dev.download(vram.whole()).unwrap() == up,
        "the host bytes did not reach the VRAM buffer"
    );
}

/// The submit trace (`ARTEMIS_SUBMIT_TRACE`): off, a submit logs nothing and a label is ignored; on, each plain submit
/// logs once, in order, with its host times ordered and its GPU span inside its submit-to-wake window (to within the
/// calibration's error), and a profiled submit logs nothing.
#[test]
#[ignore = "needs the GPU"]
fn the_submit_trace_is_off_by_default_and_logs_each_submit_when_on() {
    let fill = |dev: &Device, slice: vk::Slice, n: u32| {
        let mut r = dev.record().unwrap();
        for i in 0..n {
            r.fill(slice, i);
        }
        dev.submit(r).unwrap();
    };
    std::env::remove_var(vk::SUBMIT_TRACE_ENV);
    let off = Device::open(None).unwrap();
    let mut vram = Arena::new(MemKind::Vram, 64 << 20, "test");
    let a = vram.alloc(&off, 64 << 20).unwrap();
    assert!(!off.submit_trace_on());
    assert!(
        !off.clock_calibration_enabled(),
        "off, the device opens without the calibration extension"
    );
    assert_eq!(off.take_submit_trace_dropped(), 0);
    off.set_trace_label(3);
    fill(&off, a, 4);
    assert!(off.take_submit_trace().is_empty(), "the trace is off");
    drop(vram);
    drop(off);

    std::env::set_var(vk::SUBMIT_TRACE_ENV, "on");
    let opened = Device::open(None);
    std::env::remove_var(vk::SUBMIT_TRACE_ENV);
    let on = opened.unwrap();
    assert!(
        on.submit_trace_on() && on.clock_calibration_enabled(),
        "this GPU's driver calibrates its clock"
    );
    let mut vram = Arena::new(MemKind::Vram, 64 << 20, "test");
    let a = vram.alloc(&on, 64 << 20).unwrap();
    on.set_trace_label(7);
    fill(&on, a, 64);
    fill(&on, a, 1);
    let mut r = on.record_profiled().unwrap();
    r.fill(a, 1);
    on.submit_profiled(r).unwrap();
    let t = on.take_submit_trace();
    assert_eq!(t.len(), 2, "one entry a plain submit: {t:?}");
    assert_eq!((t[0].label, t[1].label), (7, 0), "a label names one submit");
    for s in &t {
        assert!(
            s.record_ns <= s.submit_ns
                && s.submit_ns <= s.submitted_ns
                && s.submitted_ns <= s.woke_ns,
            "{s:?}"
        );
        assert!(s.gpu_start_ns <= s.gpu_end_ns, "{s:?}");
        let e = s.cal_err_ns + 1_000;
        assert!(
            s.gpu_start_ns + e >= s.submit_ns && s.gpu_end_ns <= s.woke_ns + e,
            "the GPU ran outside its submit's window: {s:?}"
        );
    }
    assert!(t[0].woke_ns <= t[1].record_ns, "in order: {t:?}");
    // 64 fills of 64 MiB take far longer than one.
    assert!(
        t[0].gpu_end_ns - t[0].gpu_start_ns > t[1].gpu_end_ns - t[1].gpu_start_ns,
        "{t:?}"
    );
    assert!(
        on.take_submit_trace().is_empty(),
        "each entry is taken once"
    );
    assert_eq!(on.take_submit_trace_dropped(), 0);
}
