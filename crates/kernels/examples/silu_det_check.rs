//! Exhaustive check of `silu_det` (and, for comparison, today's twin `silu`) over all 2³² f32 bit
//! patterns against an f64 reference, reporting the worst errors. Not a CI
//! test (minutes, not a `cargo test`). Usage: silu_det_check [threads].
use kernels::twin::silu;
use kernels::twin::silu_det::{silu_det_probe, silu_f64};
use std::sync::Mutex;

#[derive(Default, Clone, Copy, Debug)]
struct Worst {
    v: f64,
    x: u32,
}
impl Worst {
    fn put(&mut self, v: f64, x: u32) {
        if v > self.v {
            *self = Worst { v, x };
        }
    }
    fn merge(&mut self, o: Worst) {
        self.put(o.v, o.x)
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct Stats {
    nan_bad: u64,
    denorm_out: u64,
    denorm_mid: u64,
    ulp_core: Worst, // 2^-125 <= |x| < 30, reference normal
    ulp_all: Worst,  // outside silu_det's flush ranges (|x| < 2^-125, x < -87), reference normal
    rel_core: Worst,
    rel_all: Worst,
    abs_tiny: Worst,  // reference denormal or zero
    abs_flush: Worst, // silu_det's flush ranges: |x| < 2^-125 or x < -87
    hist: [u64; 5],   // ulp 0, 1, 2, 3, >3 over the core
}

impl Stats {
    fn merge(&mut self, o: &Stats) {
        self.nan_bad += o.nan_bad;
        self.denorm_out += o.denorm_out;
        self.denorm_mid += o.denorm_mid;
        self.ulp_core.merge(o.ulp_core);
        self.ulp_all.merge(o.ulp_all);
        self.rel_core.merge(o.rel_core);
        self.rel_all.merge(o.rel_all);
        self.abs_tiny.merge(o.abs_tiny);
        self.abs_flush.merge(o.abs_flush);
        for i in 0..5 {
            self.hist[i] += o.hist[i];
        }
    }
}

fn ord(v: f32) -> i64 {
    let b = v.to_bits();
    if b & 0x8000_0000 != 0 {
        -((b & 0x7fff_ffff) as i64)
    } else {
        b as i64
    }
}

fn score(st: &mut Stats, xb: u32, x: f32, out: f32, refv: f64) {
    let ref32 = refv as f32;
    if x.is_nan() {
        if !out.is_nan() {
            st.nan_bad += 1;
        }
        return;
    }
    if out.is_nan() {
        st.nan_bad += 1;
        return;
    }
    if out.is_subnormal() {
        st.denorm_out += 1;
    }
    if !x.is_finite() {
        return;
    }
    let a = xb & 0x7fff_ffff;
    let flush = a < 0x0100_0000 || x < -87.0;
    let core = !flush && x.abs() < 30.0;
    if flush {
        st.abs_flush.put((out as f64 - refv).abs(), xb);
    } else if ref32.is_normal() {
        let u = (ord(out) - ord(ref32)).unsigned_abs() as f64;
        let rel = ((out as f64 - refv) / refv).abs();
        st.ulp_all.put(u, xb);
        st.rel_all.put(rel, xb);
        if core {
            st.ulp_core.put(u, xb);
            st.rel_core.put(rel, xb);
            st.hist[(u as usize).min(4)] += 1;
        }
    } else {
        st.abs_tiny.put((out as f64 - refv).abs(), xb);
    }
}

fn main() {
    let threads: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(32);
    let det = Mutex::new(Stats::default());
    let twin = Mutex::new(Stats::default());
    let t0 = std::time::Instant::now();
    let chunk = (1u64 << 32) / threads as u64;
    std::thread::scope(|sc| {
        for t in 0..threads {
            let (det, twin) = (&det, &twin);
            sc.spawn(move || {
                let (mut sd, mut stw) = (Stats::default(), Stats::default());
                let lo = t as u64 * chunk;
                let hi = if t + 1 == threads {
                    1u64 << 32
                } else {
                    lo + chunk
                };
                for b in lo..hi {
                    let xb = b as u32;
                    let x = f32::from_bits(xb);
                    let mut mid = false;
                    let o = silu_det_probe(x, &mut |v: f32| {
                        if v.is_subnormal() {
                            mid = true
                        }
                    });
                    if mid {
                        sd.denorm_mid += 1;
                    }
                    let r = silu_f64(x);
                    score(&mut sd, xb, x, o, r);
                    score(&mut stw, xb, x, silu(x), r);
                }
                det.lock().unwrap().merge(&sd);
                twin.lock().unwrap().merge(&stw);
            });
        }
    });
    let el = t0.elapsed();
    for (name, s) in [
        ("silu_det", det.into_inner().unwrap()),
        ("twin silu", twin.into_inner().unwrap()),
    ] {
        let w = |w: Worst| {
            format!(
                "{:.4e} at x = {:e} ({:#010x})",
                w.v,
                f32::from_bits(w.x),
                w.x
            )
        };
        println!("== {name}");
        println!("  NaN contract violations: {}", s.nan_bad);
        println!("  denormal outputs: {}", s.denorm_out);
        if name == "silu_det" {
            println!("  inputs with a denormal intermediate: {}", s.denorm_mid);
        }
        println!("  max ulp, 2^-125<=|x|<30:      {}", w(s.ulp_core));
        println!("  max ulp, -87<=x, |x|>=2^-125: {}", w(s.ulp_all));
        println!("  max rel, 2^-125<=|x|<30:      {}", w(s.rel_core));
        println!("  max rel, -87<=x, |x|>=2^-125: {}", w(s.rel_all));
        println!(
            "  max abs where ref is denormal/zero (outside flush ranges): {}",
            w(s.abs_tiny)
        );
        println!(
            "  max abs in the flush ranges (|x|<2^-125, x<-87): {}",
            w(s.abs_flush)
        );
        let tot: u64 = s.hist.iter().sum();
        println!(
            "  ulp histogram, core (0/1/2/3/>3): {:?}  ({:.4}% exact)",
            s.hist,
            100.0 * s.hist[0] as f64 / tot as f64
        );
    }
    println!("{threads} threads, {:.1} s", el.as_secs_f64());
}
