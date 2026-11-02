//! The pin set: pure, so these run on CI's runner, which has no AVX-512. They never skip.
use cpu_expert::pin_set;

fn bits(range: std::ops::Range<u32>) -> u64 {
    range.fold(0u64, |m, b| m | (1 << b))
}

#[test]
fn workers_stay_inside_the_process_mask() {
    // The CCD1 mask: eight even CPUs, one per physical core, 16 last.
    let ccd1 = bits(16..32);
    assert_eq!(pin_set(ccd1), Some(vec![18, 20, 22, 24, 26, 28, 30, 16]));
    assert!(pin_set(0).is_none());
    let one = 1u64 << 17; // an odd CPU only
    assert!(pin_set(one).is_none());
    // Every CPU in a set lies inside the mask, and is even.
    for mask in [
        ccd1,
        bits(0..4),
        bits(0..64),
        1 << 5 | 1 << 40 | 1 << 62,
        bits(8..24),
    ] {
        for c in pin_set(mask).unwrap_or_default() {
            assert!(mask >> c & 1 == 1 && c % 2 == 0, "{c} outside {mask:#x}");
        }
    }
}

#[test]
fn a_mask_the_workers_fill_leaves_the_engine_no_cpu() {
    // Each worker spins at TIME_CRITICAL, so the engine thread needs a CPU of the mask besides the workers'. CCD1's
    // mask leaves the odd CPUs; the physical-cores-only mask (the even CPUs of CCD1) and one even CPU
    // leave none.
    for (mask, spare) in [
        (0xFFFF_0000u64, true),
        (0x5555_0000, false),
        (1 << 18, false),
    ] {
        let pins = pin_set(mask).expect("an even CPU");
        assert_eq!(cpu_expert::spare_cpu(mask, &pins), spare, "{mask:#x}");
    }
}

#[test]
fn a_small_mask_gives_its_even_cpus_low_to_high() {
    assert_eq!(pin_set(bits(0..4)), Some(vec![0, 2]));
    // At most eight, low to high, for a mask that is not CCD1's.
    assert_eq!(pin_set(bits(0..32)), Some(vec![0, 2, 4, 6, 8, 10, 12, 14]));
    assert_eq!(pin_set(1 << 5 | 1 << 40), Some(vec![40]));
}

#[test]
fn the_roots_re_export_what_task_six_calls() {
    // Compile-time: the names the engine uses sit at the crate root.
    type ExpertRows = fn(
        &cpu_expert::Pool,
        &[&[u8]],
        &kernels::dispatch::SlotLayout,
        &cpu_expert::XPrep,
        &mut [f32],
        &mut [f32],
    );
    let _: ExpertRows = cpu_expert::expert_rows;
    let _ = cpu_expert::available();
    // Off Windows the process mask is unknown (0); on Windows it is non-zero.
    let m = cpu_expert::process_affinity_mask();
    assert_eq!(m == 0, !cfg!(windows));
}
