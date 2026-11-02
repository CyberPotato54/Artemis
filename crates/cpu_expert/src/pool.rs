//! A persistent pool of spin-waiting workers (the caller is worker 0), and `expert_rows`, which splits the rows of
//! several experts' matrices across it: gate/up row groups, a join, then down row groups. Every job runs inside
//! `with_ftz`, so workers 1..n-1 carry the FTZ+DAZ bits.
use crate::avx::{self, XPrep};
use kernels::dispatch::SlotLayout;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::*};
use std::sync::Arc;

type Job = &'static (dyn Fn(usize) + Sync);

/// Runs `f` with the MXCSR's FTZ and DAZ bits set (`MXCSR |= 0x8040`), then restores them. The GPU's plain f32
/// shaders flush denormals on this driver (measured 2026-10-13), and the worker must match them (the CPU's
/// default MXCSR differs from the GPU exactly there). It flushes denormal inputs and intermediates (the act
/// product among them) to 0, so it changes `y` wherever one occurs; the GPU's plain shaders flush the same.
///
/// A drop guard restores the MXCSR, so a panicking `f` leaves it as it found it (the thread, a pool worker's or the
/// caller's, goes on to run code that expects the default rounding and denormal behaviour).
pub fn with_ftz<R>(f: impl FnOnce() -> R) -> R {
    /// Puts the MXCSR back to the value held on drop.
    struct Restore(u32);
    impl Drop for Restore {
        #[allow(deprecated)]
        fn drop(&mut self) {
            // SAFETY: writes back the value `_mm_getcsr` returned on this thread.
            unsafe { std::arch::x86_64::_mm_setcsr(self.0) };
        }
    }
    // SAFETY: reads and writes this thread's MXCSR, which exists on every x86_64 CPU.
    #[allow(deprecated)]
    let _restore = unsafe {
        use std::arch::x86_64::*;
        let old = _mm_getcsr();
        _mm_setcsr(old | 0x8040);
        Restore(old)
    };
    f()
}

struct Shared {
    gen: AtomicUsize,
    done: AtomicUsize,
    stop: AtomicBool,
    /// A worker's job panicked (sticky): `run` panics once no worker is still inside the job.
    failed: AtomicBool,
    job: UnsafeCell<Option<Job>>,
}
unsafe impl Sync for Shared {}

/// The pool: `run` has one caller at a time, so a `Pool` is `Send` (it moves to the thread that runs it) but not
/// `Sync` (two threads cannot call `run` on one pool, which would race on the job and the done count and let a worker
/// run a closure after its `run` returned).
///
/// ```compile_fail
/// fn needs_sync<T: Sync>() {}
/// needs_sync::<cpu_expert::Pool>();
/// ```
///
/// ```
/// fn needs_send<T: Send>() {}
/// needs_send::<cpu_expert::Pool>();
/// ```
pub struct Pool {
    pub n: usize,
    sh: Arc<Shared>,
    hs: Vec<std::thread::JoinHandle<()>>,
    pub pinned: bool,
    /// Makes the pool `!Sync` (it stays `Send`): `run` takes `&self` but supports one caller at a time.
    _not_sync: std::marker::PhantomData<std::cell::Cell<()>>,
}

/// Pins the calling thread to logical CPU `cpu`; false if Windows refused.
#[cfg(windows)]
pub fn pin(cpu: usize) -> bool {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
    unsafe { SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu) != 0 }
}

/// Unpins the calling thread (to the process's affinity).
#[cfg(windows)]
pub fn unpin() {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
    let (p, _) = affinity();
    unsafe { SetThreadAffinityMask(GetCurrentThread(), p) };
}

/// (process affinity, system affinity).
#[cfg(windows)]
pub fn affinity() -> (usize, usize) {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessAffinityMask};
    let (mut p, mut s) = (0usize, 0usize);
    unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut p, &mut s) };
    (p, s)
}

/// `THREAD_PRIORITY_TIME_CRITICAL`: the CPU miss path's workers.
pub const TIME_CRITICAL: i32 = 15;
/// `THREAD_PRIORITY_HIGHEST`: the engine thread for a submit while the CPU miss path is on.
pub const HIGHEST: i32 = 2;

/// Sets the calling thread's priority (`SetThreadPriority`); false if Windows refused.
#[cfg(windows)]
pub fn set_priority(p: i32) -> bool {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority};
    unsafe { SetThreadPriority(GetCurrentThread(), p) != 0 }
}

/// The calling thread's priority (`GetThreadPriority`).
#[cfg(windows)]
pub fn priority() -> i32 {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadPriority};
    unsafe { GetThreadPriority(GetCurrentThread()) }
}

/// Widens this process's affinity to every logical CPU (an affinity tool can restrict a new process to CPUs 16-31
/// only).
#[cfg(windows)]
pub fn widen_process_affinity() -> bool {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetProcessAffinityMask};
    let (_, sys) = affinity();
    unsafe { SetProcessAffinityMask(GetCurrentProcess(), sys) != 0 }
}

// Elsewhere (CI's Linux runner) threads stay unpinned: pin and widen report refusal, affinity is unknown (0).
#[cfg(not(windows))]
pub fn pin(_cpu: usize) -> bool {
    false
}
#[cfg(not(windows))]
pub fn unpin() {}
#[cfg(not(windows))]
pub fn affinity() -> (usize, usize) {
    (0, 0)
}
#[cfg(not(windows))]
pub fn widen_process_affinity() -> bool {
    false
}
#[cfg(not(windows))]
pub fn set_priority(_p: i32) -> bool {
    false
}
#[cfg(not(windows))]
pub fn priority() -> i32 {
    0
}

impl Pool {
    /// One worker per entry of `cpus` (None: unpinned); the caller is worker 0 and is pinned to cpus[0].
    pub fn new(cpus: &[Option<usize>]) -> Pool {
        let n = cpus.len();
        let sh = Arc::new(Shared {
            gen: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            job: UnsafeCell::new(None),
        });
        let mut pinned = true;
        match cpus[0] {
            Some(c) => pinned &= pin(c),
            None => unpin(),
        }
        let ok = Arc::new(AtomicBool::new(true));
        let mut hs = Vec::with_capacity(n.saturating_sub(1));
        for (i, &cpu) in cpus.iter().enumerate().skip(1) {
            let (sh2, ok) = (sh.clone(), ok.clone());
            let spawned = std::thread::Builder::new().spawn(move || {
                let sh = sh2;
                if let Some(c) = cpu {
                    if !pin(c) {
                        ok.store(false, Relaxed);
                    }
                }
                let mut seen = 0;
                loop {
                    let g = loop {
                        let g = sh.gen.load(Acquire);
                        if g != seen {
                            break g;
                        }
                        if sh.stop.load(Relaxed) {
                            return;
                        }
                        std::hint::spin_loop();
                    };
                    seen = g;
                    let f = unsafe { (*sh.job.get()).unwrap() };
                    // A job that panics still counts itself done, so `run` never spins on a dead worker; `run`
                    // panics instead, on the caller's thread, once no worker is inside the job.
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        with_ftz(|| f(i))
                    }));
                    if r.is_err() {
                        sh.failed.store(true, Relaxed);
                    }
                    sh.done.fetch_add(1, Release);
                }
            });
            match spawned {
                Ok(h) => hs.push(h),
                Err(err) => {
                    // No worker is left spinning: the ones already started see stop and are joined.
                    sh.stop.store(true, Relaxed);
                    for h in hs {
                        let _ = h.join();
                    }
                    panic!("the expert pool could not start worker {i}: {err}");
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        pinned &= ok.load(Relaxed);
        Pool {
            n,
            sh,
            hs,
            pinned,
            _not_sync: std::marker::PhantomData,
        }
    }

    /// f(i) for i in 0..n, f(0) on the caller; returns when all are done. Every job runs inside `with_ftz`.
    pub fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        if self.n == 1 {
            with_ftz(|| f(0));
            return;
        }
        unsafe {
            *self.sh.job.get() = Some(std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(f))
        };
        self.sh.done.store(0, Relaxed);
        self.sh.gen.fetch_add(1, Release);
        let mine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| with_ftz(|| f(0))));
        // Not bounded by time: the workers hold `f` (and the caller's buffers it writes) as 'static, so `run` must not
        // return, or unwind, before each has left it. Every worker counts itself done, its job panicked or not, so
        // this spin ends when the slowest worker's share of the job does.
        while self.sh.done.load(Acquire) != self.n - 1 {
            std::hint::spin_loop();
        }
        if let Err(p) = mine {
            std::panic::resume_unwind(p);
        }
        assert!(
            !self.sh.failed.load(Relaxed),
            "a pool worker's job panicked"
        );
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.sh.stop.store(true, Relaxed);
        // Workers catch their jobs' panics, so a join fails only if the worker loop itself did; dropping a pool while
        // unwinding must not panic again.
        for h in self.hs.drain(..) {
            let _ = h.join();
        }
        unpin();
    }
}

#[derive(Clone, Copy)]
struct P(*mut f32);
unsafe impl Send for P {}
unsafe impl Sync for P {}

/// Items [i·total/n, (i+1)·total/n) of `total`, as (expert, group range) runs.
fn share(
    i: usize,
    n: usize,
    total: usize,
    per: usize,
) -> impl Iterator<Item = (usize, std::ops::Range<usize>)> {
    let (a, b) = (i * total / n, (i + 1) * total / n);
    let mut at = a;
    std::iter::from_fn(move || {
        if at >= b {
            return None;
        }
        let e = at / per;
        let end = b.min((e + 1) * per);
        let r = (e, at - e * per..end - e * per);
        at = end;
        Some(r)
    })
}

/// Experts `slots` (one token, x prepared in `xs`) on the pool: act = [E][n_ff], y = [E][n_embd]. The locals that
/// rebind the pointer wrappers are load-bearing, not redundant: edition 2021's disjoint capture would otherwise
/// capture the raw-pointer field by reference (`&*mut f32`, not `Sync`), and the rebinding moves the `Copy`
/// wrapper itself into the job.
#[allow(clippy::redundant_locals)]
pub fn expert_rows(
    pool: &Pool,
    slots: &[&[u8]],
    lay: &SlotLayout,
    xs: &XPrep,
    act: &mut [f32],
    y: &mut [f32],
) {
    let e = slots.len();
    let (nf, ne) = (lay.n_ff, lay.n_embd);
    assert!(act.len() >= e * nf && y.len() >= e * ne && nf % 16 == 0 && ne % 16 == 0);
    let (pa, py) = (P(act.as_mut_ptr()), P(y.as_mut_ptr()));
    let n = pool.n;
    let (g1, g2) = (nf / 16, ne / 16);
    pool.run(&|i| {
        let pa = pa;
        for (ex, r) in share(i, n, e * g1, g1) {
            let a = unsafe { std::slice::from_raw_parts_mut(pa.0.add(ex * nf), nf) };
            unsafe { avx::gate_up_groups(slots[ex], lay, xs, r, a) };
        }
    });
    pool.run(&|i| {
        thread_local! {
            static AP: std::cell::RefCell<XPrep> = Default::default();
        }
        let (pa, py) = (pa, py);
        AP.with(|ap| {
            let ap = &mut *ap.borrow_mut();
            for (ex, r) in share(i, n, e * g2, g2) {
                let a = unsafe { std::slice::from_raw_parts(pa.0.add(ex * nf), nf) };
                let yy = unsafe { std::slice::from_raw_parts_mut(py.0.add(ex * ne), ne) };
                unsafe {
                    ap.build(a, &[lay.down]);
                    avx::down_groups(slots[ex], lay, ap, r, yy);
                }
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(deprecated)]
    fn mxcsr() -> u32 {
        unsafe { std::arch::x86_64::_mm_getcsr() }
    }

    #[test]
    fn ftz_is_restored_after_a_panic() {
        let before = mxcsr() & 0x8040;
        let r = std::panic::catch_unwind(|| with_ftz(|| panic!("a job panicked")));
        assert!(r.is_err());
        assert_eq!(mxcsr() & 0x8040, before, "MXCSR's FTZ and DAZ bits");
    }

    #[test]
    fn ftz_is_set_inside_and_restored_after() {
        let before = mxcsr() & 0x8040;
        let inside = with_ftz(mxcsr) & 0x8040;
        assert_eq!(inside, 0x8040);
        assert_eq!(mxcsr() & 0x8040, before);
    }
}
