// silu_det: the GLSL twin of twin::silu_det (crates/kernels/src/twin/silu_det.rs), operation for operation.
// silu(x) = x / (1 + exp(-x)) from f32 add, sub and mul (each correctly rounded under Vulkan), integer/bit operations,
// comparisons and selects only: no fma (every intermediate is `precise`, so nothing fuses or reassociates), no
// division, no exp, and no denormal value anywhere (ranges are chosen on the input's bits, so a device that flushes
// denormal inputs takes the same branch). Proven equal to the CPU on all 2^32 inputs on this driver (2026-10-13).
//
// Ranges (on |x|'s bits): NaN -> x; |x| < 2^-125 -> 0; |x| < 2^-24 -> x * 0.5; x >= 32 -> x; x < -87 -> 0;
// otherwise exp(-x) by Cody-Waite (n by the 1.5 * 2^23 magic add, r = (v - n * C1) - n * C2 with C1 nine bits wide),
// e^r - 1 by Taylor to r^8 in Horner form, 1 + q and 1 + e kept in two parts (Fast2Sum, TwoSum), d = m * 2^E,
// 1/m by the seed (24 - 8m)/17, two Newton steps and one correction from Dekker's exact product; silu = (x * y) * 2^-E.

float silu_det(float x) {
    uint b = floatBitsToUint(x);
    uint a = b & 0x7fffffffu;
    if (a > 0x7f800000u) {
        return x;
    }
    if (a < 0x33800000u) {
        if (a < 0x01000000u) {
            return 0.0;
        }
        precise float h = x * 0.5;
        return h;
    }
    if (b < 0x80000000u) {
        if (a >= 0x42000000u) {
            return x;
        }
    } else if (a > 0x42ae0000u) {
        return 0.0;
    }
    const float LOG2E = uintBitsToFloat(0x3fb8aa3bu);
    const float MAGIC = uintBitsToFloat(0x4b400000u);
    const float C1 = uintBitsToFloat(0x3f318000u);
    const float C2 = uintBitsToFloat(0xb95e8083u);
    const float SPLIT = uintBitsToFloat(0x45800800u);
    precise float v = -x;
    // n = round(v * log2(e)).
    precise float t = v * LOG2E;
    precise float kf = t + MAGIC;
    precise float nf = kf - MAGIC;
    int n = int(floatBitsToUint(kf)) - 0x4b400000;
    // r = (v - n * C1) - n * C2.
    precise float a1 = nf * C1;
    precise float r1 = v - a1;
    precise float a2 = nf * C2;
    precise float r = r1 - a2;
    // q = e^r - 1 = r + r^2 * (1/2 + r * (1/6 + ...)).
    precise float z = r * r;
    precise float p = uintBitsToFloat(0x37d00d01u);
    p = p * r;
    p = p + uintBitsToFloat(0x39500d01u);
    p = p * r;
    p = p + uintBitsToFloat(0x3ab60b61u);
    p = p * r;
    p = p + uintBitsToFloat(0x3c088889u);
    p = p * r;
    p = p + uintBitsToFloat(0x3d2aaaabu);
    p = p * r;
    p = p + uintBitsToFloat(0x3e2aaaabu);
    p = p * r;
    p = p + uintBitsToFloat(0x3f000000u);
    precise float zp = z * p;
    precise float q = r + zp;
    // 1 + q = p_hi + p_lo exactly.
    precise float p_hi = 1.0 + q;
    precise float w = 1.0 - p_hi;
    precise float p_lo = w + q;
    // e = 2^n * (p_hi + p_lo).
    float s = uintBitsToFloat(uint(n + 127) << 23);
    precise float e_hi = s * p_hi;
    precise float e_lo = s * p_lo;
    // d = 1 + e = d_hi + d_lo.
    precise float d_hi = 1.0 + e_hi;
    precise float bb = d_hi - 1.0;
    precise float t1 = d_hi - bb;
    precise float t2 = 1.0 - t1;
    precise float t3 = e_hi - bb;
    precise float err = t2 + t3;
    precise float d_lo = err + e_lo;
    // d = (m + m_lo) * 2^E.
    int eb = int(floatBitsToUint(d_hi) >> 23) - 127;
    float inv = uintBitsToFloat(uint(127 - eb) << 23);
    precise float m = d_hi * inv;
    precise float m_lo = d_lo * inv;
    // y = 1/m: seed, two Newton steps.
    precise float sb = uintBitsToFloat(0x3ef0f0f1u) * m;
    precise float y = uintBitsToFloat(0x3fb4b4b5u) - sb;
    precise float my0 = m * y;
    precise float tw0 = 2.0 - my0;
    y = y * tw0;
    precise float my1 = m * y;
    precise float tw1 = 2.0 - my1;
    y = y * tw1;
    // rho = 1 - (m + m_lo) * y, with m * y = ph + pl exactly (Veltkamp splits).
    precise float mt = SPLIT * m;
    precise float mu = mt - m;
    precise float mh = mt - mu;
    precise float ml = m - mh;
    precise float yt = SPLIT * y;
    precise float yu = yt - y;
    precise float yh = yt - yu;
    precise float yl = y - yh;
    precise float ph = m * y;
    precise float u1 = mh * yh;
    precise float u2 = u1 - ph;
    precise float u3 = mh * yl;
    precise float u4 = u2 + u3;
    precise float u5 = ml * yh;
    precise float u6 = u4 + u5;
    precise float u7 = ml * yl;
    precise float pl = u6 + u7;
    precise float v1 = 1.0 - ph;
    precise float v2 = v1 - pl;
    precise float v3 = m_lo * y;
    precise float rho = v2 - v3;
    precise float yr = y * rho;
    precise float yc = y + yr;
    precise float xy = x * yc;
    precise float o = xy * inv;
    return o;
}
