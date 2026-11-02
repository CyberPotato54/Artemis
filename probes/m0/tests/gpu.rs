//! GPU tests: need the RX 9070 XT. Run on the test PC with
//! `cargo test -p m0 --release --test gpu -- --ignored --test-threads=1`.
use m0::ctx::{Ctx, Mem};

#[test]
#[ignore = "needs the RX 9070 XT"]
fn opens_the_discrete_gpu_and_maps_host_memory() {
    let ctx = Ctx::new(None).unwrap();
    assert!(ctx.name.contains("9070"), "{}", ctx.name);
    let b = ctx.buf(1 << 20, Mem::Host).unwrap();
    assert!(!b.ptr.is_null() && b.addr != 0);
    ctx.free(b);
    let v = ctx.buf(1 << 20, Mem::Vram).unwrap();
    assert!(v.ptr.is_null() && v.addr != 0);
    ctx.free(v);
}

use m0::probes::read_bw;

#[test]
#[ignore = "needs the RX 9070 XT"]
fn read_bw_checksum_matches_on_host_memory() {
    let ctx = Ctx::new(None).unwrap();
    for mem in [Mem::Host, Mem::Vram] {
        let r = read_bw::run(
            &ctx,
            &read_bw::Args {
                mem,
                kib: 4352,
                wgs: 64,
                chain: 2,
                reps: 2,
            },
        )
        .unwrap();
        assert_eq!(r.results["checksum_ok"], true);
        let gbps = r.results["gbps"].as_f64().unwrap();
        // Faster than PCIe 5.0 x16 from host memory, or than GDDR6 from VRAM, means the measurement is wrong.
        let ceiling = if mem == Mem::Host { 64.0 } else { 700.0 };
        assert!(gbps > 0.1 && gbps < ceiling, "{mem:?}: {gbps} GB/s");
    }
}

use m0::probes::roundtrip;

#[test]
#[ignore = "needs the RX 9070 XT"]
fn spin_without_responder_times_out_instead_of_hanging() {
    let ctx = Ctx::new(None).unwrap();
    let args = roundtrip::Args {
        mode: roundtrip::Mode::Spin,
        flag_mem: Mem::Host,
        n: 1,
        batch: 1,
        max_spins: 200_000,
        no_responder: true,
        decode_shaped: false,
        layers: 40,
        layer_kib: 36_864,
    };
    let r = roundtrip::run(&ctx, &args).unwrap();
    assert_eq!(r.results["gpu_timeouts"], 1);
    assert_eq!(r.results["device_lost"], false);
}

#[test]
#[ignore = "needs the RX 9070 XT"]
fn spin_round_trips_complete_with_a_responder() {
    let ctx = Ctx::new(None).unwrap();
    // Only AMD device-uncached memory works for polling on this GPU; plain host memory stalls (measured by the
    // roundtrip probe).
    let args = roundtrip::Args {
        mode: roundtrip::Mode::Spin,
        flag_mem: Mem::AmdUncached,
        n: 20_000,
        batch: 1000,
        max_spins: 200_000,
        no_responder: false,
        decode_shaped: false,
        layers: 40,
        layer_kib: 36_864,
    };
    let r = roundtrip::run(&ctx, &args).unwrap();
    assert_eq!(r.results["gpu_timeouts"], 0);
    assert_eq!(r.results["cpu_timeouts"], 0);
    // 20,000 round trips minus the first of each of the 20 dispatches, which includes a submit.
    assert_eq!(r.results["samples"], 19_980);
}

/// The decode-shaped probe: every handoff answered and timed on the GPU, one per layer; a
/// partial last command buffer (2,010 handoffs is 50 buffers of 40 and one of 10). Layers of 36 MiB read for about
/// 60 us each, so a median under 20 us means the stamps bracket the wait alone, not the read before it.
#[test]
#[ignore = "needs the RX 9070 XT"]
fn decode_shaped_handoffs_are_answered_and_timed_one_a_layer() {
    let ctx = Ctx::new(None).unwrap();
    let args = roundtrip::Args {
        mode: roundtrip::Mode::Spin,
        flag_mem: Mem::AmdUncached,
        n: 2_010,
        batch: 1,
        max_spins: 200_000,
        no_responder: false,
        decode_shaped: true,
        layers: 40,
        layer_kib: 36_864,
    };
    let r = roundtrip::run(&ctx, &args).unwrap();
    assert_eq!(r.results["gpu_timeouts"], 0);
    assert_eq!(r.results["cpu_timeouts"], 0);
    assert_eq!(r.results["samples"], 2_010);
    let median = r.results["median_us"].as_f64().unwrap();
    assert!(
        median > 0.0 && median < 20.0,
        "{median} us: the stamps must bracket the wait, not the read"
    );
}

/// With no CPU to answer, the decode-shaped probe's first wait times out after its spins (the one counted timeout),
/// every later wait of its buffer gives up, uncounted, within 1,024 polls (`spin.comp`), and the run stops: one timeout,
/// no hang. Without the early give-up each of the buffer's 4 waits would time out in turn (a few tenths of a second
/// each).
#[test]
#[ignore = "needs the RX 9070 XT"]
fn decode_shaped_without_responder_times_out_instead_of_hanging() {
    let ctx = Ctx::new(None).unwrap();
    let args = roundtrip::Args {
        mode: roundtrip::Mode::Spin,
        flag_mem: Mem::AmdUncached,
        n: 80,
        batch: 1,
        max_spins: 200_000,
        no_responder: true,
        decode_shaped: true,
        layers: 4,
        layer_kib: 4_096,
    };
    let r = roundtrip::run(&ctx, &args).unwrap();
    assert_eq!(r.results["gpu_timeouts"], 1);
    assert_eq!(r.results["device_lost"], false);
    assert_eq!(
        r.results["samples"], 0,
        "the run stops at the buffer that timed out"
    );
}

use m0::{probes::matvec, q::Quant};

#[test]
#[ignore = "needs the RX 9070 XT"]
fn matvec_matches_the_cpu_reference() {
    let ctx = Ctx::new(None).unwrap();
    for quant in [Quant::Q4k, Quant::Q8_0] {
        for mem in [Mem::Vram, Mem::Host] {
            // run() fails if the GPU result differs from the CPU reference by more than 1e-4 relative.
            let r = matvec::run(
                &ctx,
                &matvec::Args {
                    quant,
                    mem,
                    rows: 64,
                    k: 2048,
                    reps: 2,
                },
            )
            .unwrap();
            assert!(r.results["max_rel_err"].as_f64().unwrap() <= 1e-4);
        }
    }
}
