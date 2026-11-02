//! The CPU miss path's expert worker: an expert (gate, up, silu_det, down) whose every
//! dot product equals `kernels::twin::matvec::row_dot` bit for bit and whose activation is
//! `kernels::twin::silu_det::silu_det`. It comes as AVX-512 (16 rows in lanes, on-the-fly transpose,
//! Q4/Q5 products as table lookups) or as the scalar twin, plus the worker pool that splits several
//! experts' rows across cores. Proven bit-exact on 768 random experts (64 type combinations,
//! extreme x including denormals and inf/NaN scales) and 24 real experts, pooled on 1 to 7 threads.
pub mod avx;
pub mod pool;
pub mod scalar;
pub mod slots;
pub use avx::XPrep;
pub use pool::{expert_rows, with_ftz, Pool};

use kernels::dispatch::SlotLayout;

/// One expert on this thread: AVX-512 when the CPU has it, else the scalar twin. `act` holds
/// `n_ff` f32, `y` holds `n_embd`; `pool::expert_rows` is the pooled form. Runs under the
/// caller's MXCSR: wrap the call in `with_ftz` to flush denormal inputs and intermediates as the
/// GPU's plain shaders do, which changes `y` wherever one occurs (`Pool::run` already
/// wraps every pooled job).
pub fn expert(slot: &[u8], lay: &SlotLayout, x: &[f32], act: &mut [f32], y: &mut [f32]) {
    if avx::available() {
        thread_local! {
            static SCRATCH: std::cell::RefCell<(avx::XPrep, avx::XPrep)> = Default::default();
        }
        SCRATCH.with(|s| {
            let (xs, as_) = &mut *s.borrow_mut();
            unsafe { avx::expert(slot, lay, x, act, y, xs, as_) }
        })
    } else {
        scalar::expert(slot, lay, x, act, y)
    }
}

/// Whether the AVX-512 expert can run on this CPU (all four features `avx::available` names); false means `expert`
/// runs the scalar twin, and the CPU miss path refuses to start.
pub fn available() -> bool {
    avx::available()
}

/// The CPUs the pool's workers pin to, inside `mask_bits` (the process's affinity mask): the eight (or fewer) even
/// logical CPUs, one per physical core when SMT pairs are (2k, 2k+1). For the mask of bits 16 to 31
/// (CCD1), the order is `18, 20, 22, 24, 26, 28, 30, 16`: worker 0 is 18, and its SMT sibling 19 is the submit
/// thread's. For any other mask, the even set bits from low to high, at most 8. `None` when the mask has no even bit,
/// so a non-zero share is refused rather than pinned outside the mask. Pure: it never calls `SetProcessAffinityMask`.
pub fn pin_set(mask_bits: u64) -> Option<Vec<usize>> {
    if mask_bits == 0xFFFF_0000 {
        return Some(vec![18, 20, 22, 24, 26, 28, 30, 16]);
    }
    let cpus: Vec<usize> = (0..64usize)
        .step_by(2)
        .filter(|&c| mask_bits >> c & 1 == 1)
        .take(8)
        .collect();
    (!cpus.is_empty()).then_some(cpus)
}

/// Whether `mask_bits` holds a CPU besides `pins` (the workers' CPUs, `pin_set(mask_bits)`): the engine thread's
/// room. Each worker spins at `TIME_CRITICAL`, so a mask the workers fill (the even CPUs of CCD1,
/// `0x5555_0000`, or one CPU) would leave the engine thread no CPU to submit from, and the CPU miss path refuses it.
/// Pure.
pub fn spare_cpu(mask_bits: u64, pins: &[usize]) -> bool {
    let taken = pins.iter().fold(0u64, |m, &c| m | 1u64 << c);
    mask_bits & !taken != 0
}

/// The Windows process affinity mask (`pool::affinity().0`); 0 off Windows, where it is unknown.
pub fn process_affinity_mask() -> u64 {
    pool::affinity().0 as u64
}
