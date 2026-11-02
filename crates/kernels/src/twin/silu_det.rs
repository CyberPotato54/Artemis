//! `silu_det`: silu(x) = x / (1 + exp(-x)) from operations every IEEE-754 machine rounds the same way: f32 add, sub
//! and mul (correctly rounded on the CPU and under Vulkan), integer/bit operations, comparisons and selects. No fma,
//! no division, no exp/log intrinsics, and no denormal value anywhere (inputs are classified on their bits, so a
//! GPU that flushes denormals takes the same branch). `silu_det.glsl` is its operation-for-operation GLSL twin.
//!
//! Ranges (on |x|'s bits):
//! - NaN: x itself.
//! - |x| < 2⁻¹²⁵ (zeros and denormals included): +0 (x/2 would be denormal).
//! - |x| < 2⁻²⁴: x·0.5 (silu(x) = x/2 + x²/4 − …, and x/4 < 2⁻²⁶ relative).
//! - x ≥ 32 (+inf included): x (x·e⁻ˣ < 2⁻⁴⁰ relative).
//! - x < −87 (−inf included): +0 (|silu| < 1.4·10⁻³⁶; below −91.9 it would be denormal).
//! - otherwise: e = exp(−x) by Cody-Waite (n = round(−x·log₂e) by the 1.5·2²³ magic add; r = (−x − n·C1) − n·C2 with
//!   C1 = 0.693359375, 9 bits, so n·C1 and −x − n·C1 are exact), e^r − 1 by its Taylor series to r⁸ in Horner form,
//!   kept as 1 + q in two parts (Fast2Sum); d = 1 + e in two parts (TwoSum); d = m·2^E with m in [1, 2) read off the
//!   bits; 1/m by a linear seed (24 − 8m)/17, two Newton steps y·(2 − m·y), and one correction y + y·ρ with
//!   ρ = 1 − (m + m_lo)·y from Dekker's exact product (Veltkamp splits); silu = (x·y)·2^−E, the last scaling exact.

pub const LOG2E: u32 = 0x3fb8aa3b;
pub const C1: u32 = 0x3f318000;
pub const C2: u32 = 0xb95e8083;
pub const MAGIC: u32 = 0x4b400000;
pub const POLY: [u32; 7] = [
    0x37d00d01, // 1/40320
    0x39500d01, // 1/5040
    0x3ab60b61, // 1/720
    0x3c088889, // 1/120
    0x3d2aaaab, // 1/24
    0x3e2aaaab, // 1/6
    0x3f000000, // 1/2
];
pub const SEED_A: u32 = 0x3fb4b4b5; // 24/17
pub const SEED_B: u32 = 0x3ef0f0f1; // 8/17
pub const SPLIT: u32 = 0x45800800; // 4097

#[inline(always)]
fn f(b: u32) -> f32 {
    f32::from_bits(b)
}

/// The production entry point.
#[inline]
pub fn silu_det(x: f32) -> f32 {
    silu_det_probe(x, &mut |_| {})
}

/// `silu_det`, calling `c` on every intermediate f32 value (the exhaustive check looks for denormals).
#[inline(always)]
pub fn silu_det_probe(x: f32, c: &mut impl FnMut(f32)) -> f32 {
    let b = x.to_bits();
    let a = b & 0x7fff_ffff;
    if a > 0x7f80_0000 {
        return x;
    }
    if a < 0x3380_0000 {
        if a < 0x0100_0000 {
            return 0.0;
        }
        let h = x * 0.5;
        c(h);
        return h;
    }
    if b < 0x8000_0000 {
        if a >= 0x4200_0000 {
            return x;
        }
    } else if a > 0x42ae_0000 {
        return 0.0;
    }
    let v = -x;
    // n = round(v·log₂e).
    let t = v * f(LOG2E);
    c(t);
    let kf = t + f(MAGIC);
    c(kf);
    let nf = kf - f(MAGIC);
    c(nf);
    let n = (kf.to_bits() as i32) - (MAGIC as i32);
    // r = (v − n·C1) − n·C2.
    let a1 = nf * f(C1);
    c(a1);
    let r1 = v - a1;
    c(r1);
    let a2 = nf * f(C2);
    c(a2);
    let r = r1 - a2;
    c(r);
    // q = e^r − 1 = r + r²·(1/2 + r·(1/6 + …)).
    let z = r * r;
    c(z);
    let mut p = f(POLY[0]);
    for &k in &POLY[1..] {
        p *= r;
        c(p);
        p += f(k);
        c(p);
    }
    let zp = z * p;
    c(zp);
    let q = r + zp;
    c(q);
    // 1 + q = p_hi + p_lo exactly (Fast2Sum, |q| < 1).
    let p_hi = 1.0 + q;
    c(p_hi);
    let w = 1.0 - p_hi;
    c(w);
    let p_lo = w + q;
    c(p_lo);
    // e = 2ⁿ·(p_hi + p_lo), n in [−46, 126].
    let s = f(((n + 127) as u32) << 23);
    c(s);
    let e_hi = s * p_hi;
    c(e_hi);
    let e_lo = s * p_lo;
    c(e_lo);
    // d = 1 + e = d_hi + d_lo (TwoSum).
    let d_hi = 1.0 + e_hi;
    c(d_hi);
    let bb = d_hi - 1.0;
    c(bb);
    let t1 = d_hi - bb;
    c(t1);
    let t2 = 1.0 - t1;
    c(t2);
    let t3 = e_hi - bb;
    c(t3);
    let err = t2 + t3;
    c(err);
    let d_lo = err + e_lo;
    c(d_lo);
    // d = (m + m_lo)·2^E, m in [1, 2), E in [0, 126].
    let eb = ((d_hi.to_bits() >> 23) as i32) - 127;
    let inv = f(((127 - eb) as u32) << 23);
    c(inv);
    let m = d_hi * inv;
    c(m);
    let m_lo = d_lo * inv;
    c(m_lo);
    // y ≈ 1/m.
    let sb = f(SEED_B) * m;
    c(sb);
    let mut y = f(SEED_A) - sb;
    c(y);
    for _ in 0..2 {
        let my = m * y;
        c(my);
        let tw = 2.0 - my;
        c(tw);
        y *= tw;
        c(y);
    }
    // ρ = 1 − (m + m_lo)·y, with m·y exact as ph + pl.
    let (mh, ml) = split(m, c);
    let (yh, yl) = split(y, c);
    let ph = m * y;
    c(ph);
    let u1 = mh * yh;
    c(u1);
    let u2 = u1 - ph;
    c(u2);
    let u3 = mh * yl;
    c(u3);
    let u4 = u2 + u3;
    c(u4);
    let u5 = ml * yh;
    c(u5);
    let u6 = u4 + u5;
    c(u6);
    let u7 = ml * yl;
    c(u7);
    let pl = u6 + u7;
    c(pl);
    let v1 = 1.0 - ph;
    c(v1);
    let v2 = v1 - pl;
    c(v2);
    let v3 = m_lo * y;
    c(v3);
    let rho = v2 - v3;
    c(rho);
    let yr = y * rho;
    c(yr);
    let y = y + yr;
    c(y);
    let xy = x * y;
    c(xy);
    let out = xy * inv;
    c(out);
    out
}

#[inline(always)]
fn split(a: f32, c: &mut impl FnMut(f32)) -> (f32, f32) {
    let t = f(SPLIT) * a;
    c(t);
    let u = t - a;
    c(u);
    let hi = t - u;
    c(hi);
    let lo = a - hi;
    c(lo);
    (hi, lo)
}

/// The f64 reference, x / (1 + exp(−x)).
pub fn silu_f64(x: f32) -> f64 {
    let x = x as f64;
    x / (1.0 + (-x).exp())
}
