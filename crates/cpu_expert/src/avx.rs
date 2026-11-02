//! The AVX-512 expert worker, bit-exact with `twin::matvec::row_dot`.
//!
//! Layout: 16 rows of a matrix in the 16 lanes of a zmm. Each 32-value unit's sum is a sequential f32 chain per row,
//! so the 16 chains of one unit advance together, one vertical add per value. The rows' bytes are transposed on the
//! fly: 16 rows × 32 bytes become 8 zmm whose lane i holds row i's dword (4 consecutive bytes = 4 steps); three
//! stages of `vpermt2d` (each swaps one register-index bit with one lane-index bit). Products:
//! - Q4_K/Q5_K: q·x is looked up, not computed: per call, a table T[pos][q] = (q as f32)·x[pos] (16 or 32 entries,
//!   one or two zmm per position), indexed by `vpermps`/`vpermt2ps` with the quant itself (they read only the low
//!   4/5 bits of each index, so a shifted dword is an index). Σx per unit is computed once per call.
//! - Q6_K/Q8_0: the signed quant is moved to the top of the dword, arithmetic-shifted down, converted, multiplied by
//!   the broadcast x.
//!
//! Unit results go into 32 lane partials per row (lane L: units L, L + 32, … in order from 0), then the 32-lane tree.
use kernels::dispatch::{SlotLayout, WType};
use kernels::twin::silu_det::{C1, C2, LOG2E, MAGIC, POLY, SEED_A, SEED_B, SPLIT};
use std::arch::x86_64::*;

pub fn available() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512dq")
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct A64(pub [f32; 16]);

const fn swap_idx(s: u32, y: bool) -> [i32; 16] {
    let m = 1usize << s;
    let mut a = [0i32; 16];
    let mut l = 0;
    while l < 16 {
        a[l] = if !y {
            if l & m == 0 {
                l as i32
            } else {
                16 + (l ^ m) as i32
            }
        } else if l & m == 0 {
            (l ^ m) as i32
        } else {
            16 + l as i32
        };
        l += 1;
    }
    a
}

#[repr(C, align(64))]
struct Idx([[i32; 16]; 6]);
static IDX: Idx = Idx([
    swap_idx(0, false),
    swap_idx(0, true),
    swap_idx(1, false),
    swap_idx(1, true),
    swap_idx(2, false),
    swap_idx(2, true),
]);

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
#[inline]
unsafe fn ld_idx(i: usize) -> __m512i {
    _mm512_load_si512(IDX.0[i].as_ptr().cast())
}

/// Swaps register-index bit `s` with lane-index bit `s` across `v`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
#[inline]
unsafe fn swap_stage<const N: usize>(v: &mut [__m512i; N], s: usize) {
    let (ix, iy) = (ld_idx(2 * s), ld_idx(2 * s + 1));
    let m = 1usize << s;
    for i in 0..N {
        if i & m == 0 {
            let (x, y) = (v[i], v[i | m]);
            v[i] = _mm512_permutex2var_epi32(x, ix, y);
            v[i | m] = _mm512_permutex2var_epi32(x, iy, y);
        }
    }
}

/// 16 rows (at p + r·rb) × 32 bytes → 8 zmm: out[d] lane r = row r's dword d.
// The fixed array is built by index, in the hot transpose; an iterator rewrite obscures the stride arithmetic.
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
#[inline]
unsafe fn tr16x8(p: *const u8, rb: usize) -> [__m512i; 8] {
    let mut v = [_mm512_setzero_si512(); 8];
    for k in 0..8 {
        let a = _mm256_loadu_si256(p.add(k * rb).cast());
        let b = _mm256_loadu_si256(p.add((k + 8) * rb).cast());
        v[k] = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(a), b);
    }
    swap_stage(&mut v, 0);
    swap_stage(&mut v, 1);
    swap_stage(&mut v, 2);
    v
}

/// 16 rows × 16 bytes → 4 zmm: out[d] lane r = row r's dword d.
// The fixed array is built by index, in the hot transpose; an iterator rewrite obscures the stride arithmetic.
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
#[inline]
unsafe fn tr16x4(p: *const u8, rb: usize) -> [__m512i; 4] {
    let mut v = [_mm512_setzero_si512(); 4];
    for k in 0..4 {
        let mut z = _mm512_castsi128_si512(_mm_loadu_si128(p.add(k * rb).cast()));
        z = _mm512_inserti32x4::<1>(z, _mm_loadu_si128(p.add((k + 4) * rb).cast()));
        z = _mm512_inserti32x4::<2>(z, _mm_loadu_si128(p.add((k + 8) * rb).cast()));
        z = _mm512_inserti32x4::<3>(z, _mm_loadu_si128(p.add((k + 12) * rb).cast()));
        v[k] = z;
    }
    swap_stage(&mut v, 0);
    swap_stage(&mut v, 1);
    v
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
#[inline]
unsafe fn row_offsets(rb: usize) -> __m512i {
    let r = rb as i32;
    _mm512_setr_epi32(
        0,
        r,
        2 * r,
        3 * r,
        4 * r,
        5 * r,
        6 * r,
        7 * r,
        8 * r,
        9 * r,
        10 * r,
        11 * r,
        12 * r,
        13 * r,
        14 * r,
        15 * r,
    )
}

/// f16 in the low 16 bits of each dword → f32.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
#[inline]
unsafe fn h2f(v: __m512i) -> __m512 {
    _mm512_cvtph_ps(_mm512_cvtepi32_epi16(v))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
#[inline]
unsafe fn tree32(p: &mut [__m512; 32]) -> __m512 {
    let mut s = 16;
    while s > 0 {
        for i in 0..s {
            p[i] = _mm512_add_ps(p[i], p[i + s]);
        }
        s /= 2;
    }
    p[0]
}

/// Per-call data for one input vector: x, Σx per unit, and the product tables.
#[derive(Default)]
pub struct XPrep {
    pub k: usize,
    pub x: Vec<f32>,
    pub sx: Vec<f32>,
    pub t16: Vec<A64>,
    pub t32: Vec<A64>,
}

impl XPrep {
    /// Builds what matrices of types `ts` need.
    ///
    /// # Safety
    /// AVX-512 (`available()`) must hold: the body reads AVX-512 instructions.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
    pub unsafe fn build(&mut self, x: &[f32], ts: &[WType]) {
        let k = x.len();
        assert!(
            k.is_multiple_of(256)
                || !ts
                    .iter()
                    .any(|t| matches!(t, WType::Q4K | WType::Q5K | WType::Q6K))
        );
        self.k = k;
        self.x.clear();
        self.x.extend_from_slice(x);
        let units = k / 32;
        self.sx.resize(units, 0.0);
        // Σx per unit, 16 units at a time: lane i adds x[32(u0 + i) + l] for l in order from 0.
        let stride = _mm512_mullo_epi32(
            _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm512_set1_epi32(32),
        );
        let mut u0 = 0;
        while u0 + 16 <= units {
            let mut s = _mm512_setzero_ps();
            let base = x.as_ptr().add(32 * u0);
            for l in 0..32 {
                s = _mm512_add_ps(s, _mm512_i32gather_ps::<4>(stride, base.add(l)));
            }
            _mm512_storeu_ps(self.sx.as_mut_ptr().add(u0), s);
            u0 += 16;
        }
        for u in u0..units {
            self.sx[u] = x[32 * u..32 * u + 32].iter().fold(0f32, |s, v| s + v);
        }
        let iota = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let iota16 = _mm512_add_ps(iota, _mm512_set1_ps(16.0));
        if ts.contains(&WType::Q4K) {
            self.t16.resize(k, A64([0.0; 16]));
            for (p, &v) in x.iter().enumerate() {
                _mm512_store_ps(
                    self.t16[p].0.as_mut_ptr(),
                    _mm512_mul_ps(iota, _mm512_set1_ps(v)),
                );
            }
        }
        if ts.contains(&WType::Q5K) {
            self.t32.resize(2 * k, A64([0.0; 16]));
            for (p, &v) in x.iter().enumerate() {
                let xv = _mm512_set1_ps(v);
                _mm512_store_ps(self.t32[2 * p].0.as_mut_ptr(), _mm512_mul_ps(iota, xv));
                _mm512_store_ps(
                    self.t32[2 * p + 1].0.as_mut_ptr(),
                    _mm512_mul_ps(iota16, xv),
                );
            }
        }
    }
}

/// (d·sc_j, dmin·mn_j) for the 8 units of a Q4_K/Q5_K block, from its 16-byte head (h[0] = d | dmin, h[1..4] the
/// 12 scale bytes) for 16 rows.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
#[inline]
unsafe fn k4_scales(h: &[__m512i; 4]) -> ([__m512; 8], [__m512; 8]) {
    let d = h2f(h[0]);
    let dmin = h2f(_mm512_srli_epi32::<16>(h[0]));
    let (m63, m15, m3) = (
        _mm512_set1_epi32(63),
        _mm512_set1_epi32(15),
        _mm512_set1_epi32(3),
    );
    macro_rules! sr {
        ($v:expr, $n:expr) => {
            _mm512_srlv_epi32($v, _mm512_set1_epi32($n))
        };
    }
    let mut dsc = [_mm512_setzero_ps(); 8];
    let mut dmn = [_mm512_setzero_ps(); 8];
    for j in 0..8i32 {
        let (sc, mn) = if j < 4 {
            (
                _mm512_and_si512(sr!(h[1], 8 * j), m63),
                _mm512_and_si512(sr!(h[2], 8 * j), m63),
            )
        } else {
            let jj = j - 4;
            (
                _mm512_or_si512(
                    _mm512_and_si512(sr!(h[3], 8 * jj), m15),
                    _mm512_slli_epi32::<4>(_mm512_and_si512(sr!(h[1], 8 * jj + 6), m3)),
                ),
                _mm512_or_si512(
                    _mm512_and_si512(sr!(h[3], 8 * jj + 4), m15),
                    _mm512_slli_epi32::<4>(_mm512_and_si512(sr!(h[2], 8 * jj + 6), m3)),
                ),
            )
        };
        dsc[j as usize] = _mm512_mul_ps(d, _mm512_cvtepi32_ps(sc));
        dmn[j as usize] = _mm512_mul_ps(dmin, _mm512_cvtepi32_ps(mn));
    }
    (dsc, dmn)
}

/// 16 rows of a Q4_K (Q5 = false) or Q5_K matrix at w (row stride rb) · x → out[0..16].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
unsafe fn group_q45<const Q5: bool>(w: *const u8, rb: usize, xp: &XPrep, out: *mut f32) {
    let k = xp.k;
    let bsz = if Q5 { 176 } else { 144 };
    let mut part = [_mm512_setzero_ps(); 32];
    let m0f = _mm512_set1_epi32(0x0F0F_0F0F);
    let t16 = xp.t16.as_ptr() as *const f32;
    let t32 = xp.t32.as_ptr() as *const f32;
    for b in 0..k / 256 {
        let blk = w.add(b * bsz);
        let h = tr16x4(blk, rb);
        let (dsc, dmn) = k4_scales(&h);
        let hq = if Q5 {
            tr16x8(blk.add(16), rb)
        } else {
            [_mm512_setzero_si512(); 8]
        };
        for c in 0..4usize {
            let wq = tr16x8(blk.add(if Q5 { 48 } else { 16 } + 32 * c), rb);
            let u0 = 8 * b + 2 * c;
            let (mut a0, mut a1) = (_mm512_setzero_ps(), _mm512_setzero_ps());
            let r0 = _mm512_set1_epi32(((2 * c as i32) - 4) & 31);
            let r1 = _mm512_set1_epi32(((2 * c as i32) - 3) & 31);
            for l4 in 0..8usize {
                let (p0, p1) = if Q5 {
                    let h0 = _mm512_rorv_epi32(hq[l4], r0);
                    let h1 = _mm512_rorv_epi32(hq[l4], r1);
                    (
                        _mm512_ternarylogic_epi32::<0xE4>(wq[l4], h0, m0f),
                        _mm512_ternarylogic_epi32::<0xE4>(_mm512_srli_epi32::<4>(wq[l4]), h1, m0f),
                    )
                } else {
                    (wq[l4], _mm512_srli_epi32::<4>(wq[l4]))
                };
                let pos0 = 32 * u0 + 4 * l4;
                macro_rules! step {
                    ($e:literal) => {{
                        let i0 = _mm512_srli_epi32::<{ 8 * $e }>(p0);
                        let i1 = _mm512_srli_epi32::<{ 8 * $e }>(p1);
                        let (q0, q1) = (pos0 + $e, pos0 + 32 + $e);
                        let (v0, v1) = if Q5 {
                            (
                                _mm512_permutex2var_ps(
                                    _mm512_load_ps(t32.add(32 * q0)),
                                    i0,
                                    _mm512_load_ps(t32.add(32 * q0 + 16)),
                                ),
                                _mm512_permutex2var_ps(
                                    _mm512_load_ps(t32.add(32 * q1)),
                                    i1,
                                    _mm512_load_ps(t32.add(32 * q1 + 16)),
                                ),
                            )
                        } else {
                            (
                                _mm512_permutexvar_ps(i0, _mm512_load_ps(t16.add(16 * q0))),
                                _mm512_permutexvar_ps(i1, _mm512_load_ps(t16.add(16 * q1))),
                            )
                        };
                        a0 = _mm512_add_ps(a0, v0);
                        a1 = _mm512_add_ps(a1, v1);
                    }};
                }
                step!(0);
                step!(1);
                step!(2);
                step!(3);
            }
            let s0 = _mm512_set1_ps(xp.sx[u0]);
            let s1 = _mm512_set1_ps(xp.sx[u0 + 1]);
            let v0 = _mm512_sub_ps(_mm512_mul_ps(dsc[2 * c], a0), _mm512_mul_ps(dmn[2 * c], s0));
            let v1 = _mm512_sub_ps(
                _mm512_mul_ps(dsc[2 * c + 1], a1),
                _mm512_mul_ps(dmn[2 * c + 1], s1),
            );
            part[u0 % 32] = _mm512_add_ps(part[u0 % 32], v0);
            part[(u0 + 1) % 32] = _mm512_add_ps(part[(u0 + 1) % 32], v1);
        }
    }
    _mm512_storeu_ps(out, tree32(&mut part));
}

/// 16 rows of a Q6_K matrix · x → out[0..16].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
unsafe fn group_q6(w: *const u8, rb: usize, xp: &XPrep, out: *mut f32) {
    let k = xp.k;
    let x = xp.x.as_ptr();
    let mut part = [_mm512_setzero_ps(); 32];
    let offs = row_offsets(rb);
    let m3c = _mm512_set1_epi32(0x3C3C_3C3C);
    let flip = _mm512_set1_epi32(0xAAAA_AAAAu32 as i32);
    for b in 0..k / 256 {
        let blk = w.add(b * 210);
        let sc = tr16x4(blk.add(192), rb);
        let dd = h2f(_mm512_srli_epi32::<16>(_mm512_i32gather_epi32::<1>(
            offs,
            blk.add(206).cast(),
        )));
        for n in 0..2usize {
            let qa = tr16x8(blk.add(64 * n), rb);
            let qb = tr16x8(blk.add(64 * n + 32), rb);
            let mut qh = tr16x8(blk.add(128 + 32 * n), rb);
            for v in qh.iter_mut() {
                *v = _mm512_xor_si512(*v, flip);
            }
            for m in 0..4usize {
                let u = 8 * b + 4 * n + m;
                let q = if m & 1 == 1 { &qb } else { &qa };
                // Each byte becomes (v6 ^ 32) << 2: the low nibble at bits 2..5, the (flipped) high pair at 6..7.
                let sa_cnt = _mm512_set1_epi32(2);
                let sh_cnt = _mm512_set1_epi32(6 - 2 * m as i32);
                let (mut sa, mut sb) = (_mm512_setzero_ps(), _mm512_setzero_ps());
                for l4 in 0..8usize {
                    let a = if m >> 1 == 0 {
                        _mm512_sllv_epi32(q[l4], sa_cnt)
                    } else {
                        _mm512_srlv_epi32(q[l4], sa_cnt)
                    };
                    let bh = _mm512_sllv_epi32(qh[l4], sh_cnt);
                    let pk = _mm512_ternarylogic_epi32::<0xE4>(a, bh, m3c);
                    let xb = x.add(32 * u + 4 * l4);
                    macro_rules! step {
                        ($e:literal, $acc:ident) => {{
                            let v =
                                _mm512_srai_epi32::<26>(_mm512_slli_epi32::<{ 8 * (3 - $e) }>(pk));
                            let p =
                                _mm512_mul_ps(_mm512_cvtepi32_ps(v), _mm512_set1_ps(*xb.add($e)));
                            $acc = _mm512_add_ps($acc, p);
                        }};
                    }
                    if l4 < 4 {
                        step!(0, sa);
                        step!(1, sa);
                        step!(2, sa);
                        step!(3, sa);
                    } else {
                        step!(0, sb);
                        step!(1, sb);
                        step!(2, sb);
                        step!(3, sb);
                    }
                }
                let i = 8 * n + 2 * m; // scale index of the first half; i and i + 1 share dword i / 4
                let sw = sc[i / 4];
                macro_rules! byte {
                    ($bi:expr) => {
                        _mm512_cvtepi32_ps(_mm512_srai_epi32::<24>(_mm512_sllv_epi32(
                            sw,
                            _mm512_set1_epi32(24 - 8 * (($bi) % 4) as i32),
                        )))
                    };
                }
                let (sca, scb) = (byte!(i), byte!(i + 1));
                let r = _mm512_mul_ps(
                    dd,
                    _mm512_add_ps(_mm512_mul_ps(sca, sa), _mm512_mul_ps(scb, sb)),
                );
                part[u % 32] = _mm512_add_ps(part[u % 32], r);
            }
        }
    }
    _mm512_storeu_ps(out, tree32(&mut part));
}

/// One 16-row group of a Q8_0 matrix · x → out[0..16]. (The transpose and step loops index the fixed registers
/// by their lane arithmetic; an iterator rewrite obscures it.)
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
unsafe fn group_q8(w: *const u8, rb: usize, xp: &XPrep, out: *mut f32) {
    let x = xp.x.as_ptr();
    let mut part = [_mm512_setzero_ps(); 32];
    let offs = row_offsets(rb);
    for u in 0..xp.k / 32 {
        let blk = w.add(34 * u);
        let q = tr16x8(blk.add(2), rb);
        let dd = h2f(_mm512_i32gather_epi32::<1>(offs, blk.cast()));
        let mut s = _mm512_setzero_ps();
        for l4 in 0..8usize {
            let pk = q[l4];
            let xb = x.add(32 * u + 4 * l4);
            macro_rules! step {
                ($e:literal) => {{
                    let v = _mm512_srai_epi32::<24>(_mm512_slli_epi32::<{ 8 * (3 - $e) }>(pk));
                    s = _mm512_add_ps(
                        s,
                        _mm512_mul_ps(_mm512_cvtepi32_ps(v), _mm512_set1_ps(*xb.add($e))),
                    );
                }};
            }
            step!(0);
            step!(1);
            step!(2);
            step!(3);
        }
        part[u % 32] = _mm512_add_ps(part[u % 32], _mm512_mul_ps(dd, s));
    }
    _mm512_storeu_ps(out, tree32(&mut part));
}

/// Kernel choice for timing comparisons: v1 = one 16-row group at a time (2 chains in flight for Q4_K/Q5_K, 1 for
/// Q6_K: add-latency bound); v2 (default) = up to 4 groups (Q4_K/Q5_K, 8 chains) or 8 chains of one group (Q6_K),
/// transposes staged on the stack, and a software prefetch of the next super-group. Both are bit-exact.
pub static V1: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Software prefetch on (v2 only).
pub static PREFETCH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// Prefetch into L2 (T1) instead of L1 (T0).
pub static PF_L2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// Most 16-row groups per Q4_K/Q5_K super-group (1, 2 or 4).
pub static MAXG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(2);

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
#[inline]
unsafe fn prefetch_part(pf: *const u8, pf_len: usize, b: usize, nb: usize) {
    if pf_len == 0 {
        return;
    }
    let step = pf_len.div_ceil(nb);
    let (mut o, end) = (b * step, ((b + 1) * step).min(pf_len));
    while o < end {
        if PF_L2.load(std::sync::atomic::Ordering::Relaxed) {
            _mm_prefetch::<_MM_HINT_T1>(pf.add(o).cast())
        } else {
            _mm_prefetch::<_MM_HINT_T0>(pf.add(o).cast())
        }
        o += 64;
    }
}

/// G consecutive 16-row groups of a Q4_K (Q5 = false) or Q5_K matrix · x → out[0..16G]. All G·2 unit chains of a
/// unit pair advance together (8 chains for G = 4), sharing each table load. (The chunk and register loops index
/// the fixed registers by their lane arithmetic; an iterator rewrite obscures it.)
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
unsafe fn sg_q45<const Q5: bool, const G: usize>(
    w: *const u8,
    rb: usize,
    xp: &XPrep,
    out: *mut f32,
    pf: *const u8,
    pf_len: usize,
) {
    let nb = xp.k / 256;
    let bsz = if Q5 { 176 } else { 144 };
    let m0f = _mm512_set1_epi32(0x0F0F_0F0F);
    let t16 = xp.t16.as_ptr() as *const f32;
    let t32 = xp.t32.as_ptr() as *const f32;
    let mut part = [[_mm512_setzero_ps(); 32]; G];
    let mut pk = [[[_mm512_setzero_si512(); 16]; 4]; G];
    let mut dsc = [[_mm512_setzero_ps(); 8]; G];
    let mut dmn = [[_mm512_setzero_ps(); 8]; G];
    for b in 0..nb {
        prefetch_part(pf, pf_len, b, nb);
        for g in 0..G {
            let blk = w.add(16 * g * rb + b * bsz);
            let h = tr16x4(blk, rb);
            let (s, m) = k4_scales(&h);
            dsc[g] = s;
            dmn[g] = m;
            let hq = if Q5 {
                tr16x8(blk.add(16), rb)
            } else {
                [_mm512_setzero_si512(); 8]
            };
            for c in 0..4usize {
                let wq = tr16x8(blk.add(if Q5 { 48 } else { 16 } + 32 * c), rb);
                let r0 = _mm512_set1_epi32(((2 * c as i32) - 4) & 31);
                let r1 = _mm512_set1_epi32(((2 * c as i32) - 3) & 31);
                for l4 in 0..8usize {
                    let (p0, p1) = if Q5 {
                        (
                            _mm512_ternarylogic_epi32::<0xE4>(
                                wq[l4],
                                _mm512_rorv_epi32(hq[l4], r0),
                                m0f,
                            ),
                            _mm512_ternarylogic_epi32::<0xE4>(
                                _mm512_srli_epi32::<4>(wq[l4]),
                                _mm512_rorv_epi32(hq[l4], r1),
                                m0f,
                            ),
                        )
                    } else {
                        (wq[l4], _mm512_srli_epi32::<4>(wq[l4]))
                    };
                    pk[g][c][2 * l4] = p0;
                    pk[g][c][2 * l4 + 1] = p1;
                }
            }
        }
        for c in 0..4usize {
            let u0 = 8 * b + 2 * c;
            let mut a0 = [_mm512_setzero_ps(); G];
            let mut a1 = [_mm512_setzero_ps(); G];
            for l4 in 0..8usize {
                let pos0 = 32 * u0 + 4 * l4;
                macro_rules! step {
                    ($e:literal) => {{
                        let (q0, q1) = (pos0 + $e, pos0 + 32 + $e);
                        if Q5 {
                            let (ta, tb) = (
                                _mm512_load_ps(t32.add(32 * q0)),
                                _mm512_load_ps(t32.add(32 * q0 + 16)),
                            );
                            let (tc, td) = (
                                _mm512_load_ps(t32.add(32 * q1)),
                                _mm512_load_ps(t32.add(32 * q1 + 16)),
                            );
                            for g in 0..G {
                                let i0 = _mm512_srli_epi32::<{ 8 * $e }>(pk[g][c][2 * l4]);
                                let i1 = _mm512_srli_epi32::<{ 8 * $e }>(pk[g][c][2 * l4 + 1]);
                                a0[g] = _mm512_add_ps(a0[g], _mm512_permutex2var_ps(ta, i0, tb));
                                a1[g] = _mm512_add_ps(a1[g], _mm512_permutex2var_ps(tc, i1, td));
                            }
                        } else {
                            let (ta, tc) = (
                                _mm512_load_ps(t16.add(16 * q0)),
                                _mm512_load_ps(t16.add(16 * q1)),
                            );
                            for g in 0..G {
                                let i0 = _mm512_srli_epi32::<{ 8 * $e }>(pk[g][c][2 * l4]);
                                let i1 = _mm512_srli_epi32::<{ 8 * $e }>(pk[g][c][2 * l4 + 1]);
                                a0[g] = _mm512_add_ps(a0[g], _mm512_permutexvar_ps(i0, ta));
                                a1[g] = _mm512_add_ps(a1[g], _mm512_permutexvar_ps(i1, tc));
                            }
                        }
                    }};
                }
                step!(0);
                step!(1);
                step!(2);
                step!(3);
            }
            let s0 = _mm512_set1_ps(xp.sx[u0]);
            let s1 = _mm512_set1_ps(xp.sx[u0 + 1]);
            for g in 0..G {
                let v0 = _mm512_sub_ps(
                    _mm512_mul_ps(dsc[g][2 * c], a0[g]),
                    _mm512_mul_ps(dmn[g][2 * c], s0),
                );
                let v1 = _mm512_sub_ps(
                    _mm512_mul_ps(dsc[g][2 * c + 1], a1[g]),
                    _mm512_mul_ps(dmn[g][2 * c + 1], s1),
                );
                part[g][u0 % 32] = _mm512_add_ps(part[g][u0 % 32], v0);
                part[g][(u0 + 1) % 32] = _mm512_add_ps(part[g][(u0 + 1) % 32], v1);
            }
        }
    }
    for g in 0..G {
        _mm512_storeu_ps(out.add(16 * g), tree32(&mut part[g]));
    }
}

/// One 16-row group of a Q6_K matrix · x → out[0..16], with the 8 chains of a half-block (4 units × 2 halves) in
/// flight together.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
unsafe fn sg_q6(w: *const u8, rb: usize, xp: &XPrep, out: *mut f32, pf: *const u8, pf_len: usize) {
    let nb = xp.k / 256;
    let x = xp.x.as_ptr();
    let mut part = [_mm512_setzero_ps(); 32];
    let offs = row_offsets(rb);
    let m3c = _mm512_set1_epi32(0x3C3C_3C3C);
    let flip = _mm512_set1_epi32(0xAAAA_AAAAu32 as i32);
    let two = _mm512_set1_epi32(2);
    let mut pk = [[_mm512_setzero_si512(); 8]; 4];
    for b in 0..nb {
        prefetch_part(pf, pf_len, b, nb);
        let blk = w.add(b * 210);
        let sc = tr16x4(blk.add(192), rb);
        let dd = h2f(_mm512_srli_epi32::<16>(_mm512_i32gather_epi32::<1>(
            offs,
            blk.add(206).cast(),
        )));
        for n in 0..2usize {
            let qa = tr16x8(blk.add(64 * n), rb);
            let qb = tr16x8(blk.add(64 * n + 32), rb);
            let qh = tr16x8(blk.add(128 + 32 * n), rb);
            for l4 in 0..8 {
                let h = _mm512_xor_si512(qh[l4], flip);
                // Each byte becomes (v6 ^ 32) << 2: the low nibble at bits 2..5, the flipped high pair at 6..7.
                pk[0][l4] = _mm512_ternarylogic_epi32::<0xE4>(
                    _mm512_sllv_epi32(qa[l4], two),
                    _mm512_slli_epi32::<6>(h),
                    m3c,
                );
                pk[1][l4] = _mm512_ternarylogic_epi32::<0xE4>(
                    _mm512_sllv_epi32(qb[l4], two),
                    _mm512_slli_epi32::<4>(h),
                    m3c,
                );
                pk[2][l4] = _mm512_ternarylogic_epi32::<0xE4>(
                    _mm512_srlv_epi32(qa[l4], two),
                    _mm512_slli_epi32::<2>(h),
                    m3c,
                );
                pk[3][l4] =
                    _mm512_ternarylogic_epi32::<0xE4>(_mm512_srlv_epi32(qb[l4], two), h, m3c);
            }
            let u0 = 8 * b + 4 * n;
            let mut sa = [_mm512_setzero_ps(); 4];
            let mut sb = [_mm512_setzero_ps(); 4];
            for l4 in 0..4usize {
                macro_rules! step {
                    ($e:literal) => {{
                        for m in 0..4usize {
                            let xa = _mm512_set1_ps(*x.add(32 * (u0 + m) + 4 * l4 + $e));
                            let xb = _mm512_set1_ps(*x.add(32 * (u0 + m) + 16 + 4 * l4 + $e));
                            let va = _mm512_srai_epi32::<26>(_mm512_slli_epi32::<{ 8 * (3 - $e) }>(pk[m][l4]));
                            let vb = _mm512_srai_epi32::<26>(_mm512_slli_epi32::<{ 8 * (3 - $e) }>(pk[m][l4 + 4]));
                            sa[m] = _mm512_add_ps(sa[m], _mm512_mul_ps(_mm512_cvtepi32_ps(va), xa));
                            sb[m] = _mm512_add_ps(sb[m], _mm512_mul_ps(_mm512_cvtepi32_ps(vb), xb));
                        }
                    }};
                }
                step!(0);
                step!(1);
                step!(2);
                step!(3);
            }
            for m in 0..4usize {
                let i = 8 * n + 2 * m;
                let sw = sc[i / 4];
                macro_rules! byte {
                    ($bi:expr) => {
                        _mm512_cvtepi32_ps(_mm512_srai_epi32::<24>(_mm512_sllv_epi32(
                            sw,
                            _mm512_set1_epi32(24 - 8 * (($bi) % 4) as i32),
                        )))
                    };
                }
                let (sca, scb) = (byte!(i), byte!(i + 1));
                let r = _mm512_mul_ps(
                    dd,
                    _mm512_add_ps(_mm512_mul_ps(sca, sa[m]), _mm512_mul_ps(scb, sb[m])),
                );
                let u = u0 + m;
                part[u % 32] = _mm512_add_ps(part[u % 32], r);
            }
        }
    }
    _mm512_storeu_ps(out, tree32(&mut part));
}

/// `n` (1..=4) consecutive 16-row groups at w → out[0..16n], prefetching `pf_len` bytes at pf meanwhile.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
#[inline]
unsafe fn super_group(
    t: WType,
    w: *const u8,
    rb: usize,
    xp: &XPrep,
    out: *mut f32,
    n: usize,
    pf: *const u8,
    pf_len: usize,
) {
    use std::sync::atomic::Ordering::Relaxed;
    if V1.load(Relaxed) {
        for g in 0..n {
            let (wg, o) = (w.add(16 * g * rb), out.add(16 * g));
            match t {
                WType::Q4K => group_q45::<false>(wg, rb, xp, o),
                WType::Q5K => group_q45::<true>(wg, rb, xp, o),
                WType::Q6K => group_q6(wg, rb, xp, o),
                WType::Q8_0 => group_q8(wg, rb, xp, o),
                _ => unimplemented!("{t:?} expert matrices"),
            }
        }
        return;
    }
    let pf_len = if PREFETCH.load(Relaxed) { pf_len } else { 0 };
    match (t, n) {
        (WType::Q4K, 4) => sg_q45::<false, 4>(w, rb, xp, out, pf, pf_len),
        (WType::Q4K, 2) => sg_q45::<false, 2>(w, rb, xp, out, pf, pf_len),
        (WType::Q5K, 4) => sg_q45::<true, 4>(w, rb, xp, out, pf, pf_len),
        (WType::Q5K, 2) => sg_q45::<true, 2>(w, rb, xp, out, pf, pf_len),
        (WType::Q4K, 3) => {
            sg_q45::<false, 2>(w, rb, xp, out, pf, 0);
            sg_q45::<false, 1>(w.add(32 * rb), rb, xp, out.add(32), pf, pf_len);
        }
        (WType::Q5K, 3) => {
            sg_q45::<true, 2>(w, rb, xp, out, pf, 0);
            sg_q45::<true, 1>(w.add(32 * rb), rb, xp, out.add(32), pf, pf_len);
        }
        (WType::Q4K, _) => sg_q45::<false, 1>(w, rb, xp, out, pf, pf_len),
        (WType::Q5K, _) => sg_q45::<true, 1>(w, rb, xp, out, pf, pf_len),
        (WType::Q6K, _) => {
            for g in 0..n {
                let (p, l) = if g + 1 == n {
                    (pf, pf_len)
                } else {
                    (w.add(16 * (g + 1) * rb), 16 * rb)
                };
                sg_q6(w.add(16 * g * rb), rb, xp, out.add(16 * g), p, l);
            }
        }
        (WType::Q8_0, _) => {
            for g in 0..n {
                group_q8(w.add(16 * g * rb), rb, xp, out.add(16 * g));
            }
        }
        _ => unimplemented!("{t:?} expert matrices"),
    }
}

/// How many 16-row groups the super-group starting at g (of [g, end)) takes.
fn sg_len(g: usize, end: usize) -> usize {
    (end - g).min(MAXG.load(std::sync::atomic::Ordering::Relaxed))
}

/// Rows [16·g, 16·g + 16) for g in `groups` of matrix `mat` (type t, k = xp.k columns) into out[row].
///
/// # Safety
/// AVX-512 (`available()`) must hold: the body reads AVX-512 instructions. The asserts bound every read and write.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
pub unsafe fn mat_groups(
    t: WType,
    mat: &[u8],
    xp: &XPrep,
    groups: std::ops::Range<usize>,
    out: &mut [f32],
) {
    let rb = t.row_bytes(xp.k);
    assert!(groups.end * 16 * rb <= mat.len() && groups.end * 16 <= out.len());
    assert_tables(t, xp);
    let mut g = groups.start;
    while g < groups.end {
        let n = sg_len(g, groups.end);
        let next = g + n;
        let pf_len = if next < groups.end {
            16 * sg_len(next, groups.end) * rb
        } else {
            0
        };
        let pf = mat.as_ptr().add(16 * next.min(groups.end) * rb);
        super_group(
            t,
            mat.as_ptr().add(16 * g * rb),
            rb,
            xp,
            out.as_mut_ptr().add(16 * g),
            n,
            pf,
            pf_len,
        );
        g = next;
    }
}

/// Asserts that `xp` holds the product tables a matrix of type `t` reads (`XPrep::build` makes `t16` for Q4_K and
/// `t32` for Q5_K, each of the length the kernels index): an empty table would be read through a dangling pointer.
fn assert_tables(t: WType, xp: &XPrep) {
    match t {
        WType::Q4K => assert_eq!(xp.t16.len(), xp.k, "x not prepared for Q4_K (no t16 table)"),
        WType::Q5K => assert_eq!(
            xp.t32.len(),
            2 * xp.k,
            "x not prepared for Q5_K (no t32 table)"
        ),
        _ => {}
    }
}

/// Gate/up row groups `groups` of one expert: act[r] = silu_det(gate_r·x)·(up_r·x).
///
/// # Safety
/// AVX-512 (`available()`) must hold: the body reads AVX-512 instructions. The asserts bound every read and write.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
pub unsafe fn gate_up_groups(
    slot: &[u8],
    lay: &SlotLayout,
    xp: &XPrep,
    groups: std::ops::Range<usize>,
    act: &mut [f32],
) {
    let (rg, ru) = (lay.gate.row_bytes(lay.n_embd), lay.up.row_bytes(lay.n_embd));
    assert!(
        lay.n_ff.is_multiple_of(16) && groups.end * 16 <= lay.n_ff && act.len() >= groups.end * 16
    );
    assert!(slot.len() >= lay.bytes());
    // The block loops run to `xp.k / 256` while the row stride comes from `lay.n_embd`: a larger k reads past the row.
    assert_eq!(xp.k, lay.n_embd, "x prepared for another row length");
    assert_tables(lay.gate, xp);
    assert_tables(lay.up, xp);
    let gate = slot.as_ptr();
    let up = slot.as_ptr().add(lay.up_offset());
    let mut gv = [0f32; 64];
    let mut uv = [0f32; 64];
    let mut gi = groups.start;
    while gi < groups.end {
        let n = sg_len(gi, groups.end);
        let next = gi + n;
        let (wg, wu) = (gate.add(16 * gi * rg), up.add(16 * gi * ru));
        // Gate prefetches this super-group's up rows; up prefetches the next super-group's gate rows.
        super_group(lay.gate, wg, rg, xp, gv.as_mut_ptr(), n, wu, 16 * n * ru);
        let pf_len = if next < groups.end {
            16 * sg_len(next, groups.end) * rg
        } else {
            0
        };
        super_group(
            lay.up,
            wu,
            ru,
            xp,
            uv.as_mut_ptr(),
            n,
            gate.add(16 * next.min(groups.end) * rg),
            pf_len,
        );
        for j in 0..n {
            let a = _mm512_mul_ps(
                silu_det16(_mm512_loadu_ps(gv.as_ptr().add(16 * j))),
                _mm512_loadu_ps(uv.as_ptr().add(16 * j)),
            );
            _mm512_storeu_ps(act.as_mut_ptr().add(16 * (gi + j)), a);
        }
        gi = next;
    }
}

/// Down row groups `groups` of one expert: y[r] = down_r·act (ap prepared from act).
///
/// # Safety
/// AVX-512 (`available()`) must hold: the body reads AVX-512 instructions. The assert bounds every read.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
pub unsafe fn down_groups(
    slot: &[u8],
    lay: &SlotLayout,
    ap: &XPrep,
    groups: std::ops::Range<usize>,
    y: &mut [f32],
) {
    assert!(lay.n_embd.is_multiple_of(16) && slot.len() >= lay.bytes());
    mat_groups(lay.down, &slot[lay.down_offset()..], ap, groups, y);
}

/// One expert on this thread. `xs`/`as_` are reusable scratch.
///
/// # Safety
/// AVX-512 (`available()`) must hold: the body reads AVX-512 instructions. `act` must hold `lay.n_ff` and `y`
/// `lay.n_embd` f32.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
pub unsafe fn expert(
    slot: &[u8],
    lay: &SlotLayout,
    x: &[f32],
    act: &mut [f32],
    y: &mut [f32],
    xs: &mut XPrep,
    as_: &mut XPrep,
) {
    xs.build(x, &[lay.gate, lay.up]);
    expert_prepped(slot, lay, xs, act, y, as_);
}

/// `expert` with x already prepared (several experts of one token share it).
///
/// # Safety
/// AVX-512 (`available()`) must hold: the body reads AVX-512 instructions. `act` must hold `lay.n_ff` and `y`
/// `lay.n_embd` f32.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
pub unsafe fn expert_prepped(
    slot: &[u8],
    lay: &SlotLayout,
    xs: &XPrep,
    act: &mut [f32],
    y: &mut [f32],
    as_: &mut XPrep,
) {
    gate_up_groups(slot, lay, xs, 0..lay.n_ff / 16, act);
    as_.build(&act[..lay.n_ff], &[lay.down]);
    down_groups(slot, lay, as_, 0..lay.n_embd / 16, y);
}

// ---- silu_det on 16 lanes: the scalar's operations, lane for lane ----

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
#[inline]
unsafe fn cf(b: u32) -> __m512 {
    _mm512_castsi512_ps(_mm512_set1_epi32(b as i32))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
#[inline]
unsafe fn split16(a: __m512) -> (__m512, __m512) {
    let t = _mm512_mul_ps(cf(SPLIT), a);
    let u = _mm512_sub_ps(t, a);
    let hi = _mm512_sub_ps(t, u);
    (hi, _mm512_sub_ps(a, hi))
}

/// `silu_det` over the 16 lanes, the scalar's operations lane for lane (see `twin::silu_det`).
///
/// # Safety
/// AVX-512 (`available()`) must hold: the body reads AVX-512 instructions.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
pub unsafe fn silu_det16(x: __m512) -> __m512 {
    let one = _mm512_set1_ps(1.0);
    let b = _mm512_castps_si512(x);
    let a = _mm512_and_si512(b, _mm512_set1_epi32(0x7fff_ffff));
    macro_rules! c {
        ($v:expr) => {
            _mm512_set1_epi32(($v) as u32 as i32)
        };
    }
    let nan = _mm512_cmpgt_epu32_mask(a, c!(0x7f80_0000));
    let tiny0 = _mm512_cmplt_epu32_mask(a, c!(0x0100_0000));
    let tiny = _mm512_cmplt_epu32_mask(a, c!(0x3380_0000));
    let pos = _mm512_cmplt_epu32_mask(b, c!(0x8000_0000));
    let big = pos & _mm512_cmpge_epu32_mask(a, c!(0x4200_0000));
    let negbig = !pos & _mm512_cmpgt_epu32_mask(a, c!(0x42ae_0000));
    let special = nan | tiny | big | negbig;
    let xg = _mm512_mask_blend_ps(special, x, one);

    let v = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(xg), c!(0x8000_0000)));
    let t = _mm512_mul_ps(v, cf(LOG2E));
    let kf = _mm512_add_ps(t, cf(MAGIC));
    let nf = _mm512_sub_ps(kf, cf(MAGIC));
    let n = _mm512_sub_epi32(_mm512_castps_si512(kf), c!(MAGIC));
    let a1 = _mm512_mul_ps(nf, cf(C1));
    let r1 = _mm512_sub_ps(v, a1);
    let a2 = _mm512_mul_ps(nf, cf(C2));
    let r = _mm512_sub_ps(r1, a2);
    let z = _mm512_mul_ps(r, r);
    let mut p = cf(POLY[0]);
    for &k in &POLY[1..] {
        p = _mm512_mul_ps(p, r);
        p = _mm512_add_ps(p, cf(k));
    }
    let zp = _mm512_mul_ps(z, p);
    let q = _mm512_add_ps(r, zp);
    let p_hi = _mm512_add_ps(one, q);
    let w = _mm512_sub_ps(one, p_hi);
    let p_lo = _mm512_add_ps(w, q);
    let s = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(n, c!(127))));
    let e_hi = _mm512_mul_ps(s, p_hi);
    let e_lo = _mm512_mul_ps(s, p_lo);
    let d_hi = _mm512_add_ps(one, e_hi);
    let bb = _mm512_sub_ps(d_hi, one);
    let t1 = _mm512_sub_ps(d_hi, bb);
    let t2 = _mm512_sub_ps(one, t1);
    let t3 = _mm512_sub_ps(e_hi, bb);
    let err = _mm512_add_ps(t2, t3);
    let d_lo = _mm512_add_ps(err, e_lo);
    let eb = _mm512_sub_epi32(_mm512_srli_epi32::<23>(_mm512_castps_si512(d_hi)), c!(127));
    let inv = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_sub_epi32(c!(127), eb)));
    let m = _mm512_mul_ps(d_hi, inv);
    let m_lo = _mm512_mul_ps(d_lo, inv);
    let sb = _mm512_mul_ps(cf(SEED_B), m);
    let mut y = _mm512_sub_ps(cf(SEED_A), sb);
    let two = _mm512_set1_ps(2.0);
    for _ in 0..2 {
        let my = _mm512_mul_ps(m, y);
        let tw = _mm512_sub_ps(two, my);
        y = _mm512_mul_ps(y, tw);
    }
    let (mh, ml) = split16(m);
    let (yh, yl) = split16(y);
    let ph = _mm512_mul_ps(m, y);
    let u1 = _mm512_mul_ps(mh, yh);
    let u2 = _mm512_sub_ps(u1, ph);
    let u3 = _mm512_mul_ps(mh, yl);
    let u4 = _mm512_add_ps(u2, u3);
    let u5 = _mm512_mul_ps(ml, yh);
    let u6 = _mm512_add_ps(u4, u5);
    let u7 = _mm512_mul_ps(ml, yl);
    let pl = _mm512_add_ps(u6, u7);
    let v1 = _mm512_sub_ps(one, ph);
    let v2 = _mm512_sub_ps(v1, pl);
    let v3 = _mm512_mul_ps(m_lo, y);
    let rho = _mm512_sub_ps(v2, v3);
    let yr = _mm512_mul_ps(y, rho);
    let y = _mm512_add_ps(y, yr);
    let xy = _mm512_mul_ps(xg, y);
    let g = _mm512_mul_ps(xy, inv);

    let zero = _mm512_setzero_ps();
    let mut out = g;
    out = _mm512_mask_blend_ps(negbig, out, zero);
    out = _mm512_mask_blend_ps(big, out, x);
    out = _mm512_mask_blend_ps(tiny, out, _mm512_mul_ps(x, _mm512_set1_ps(0.5)));
    out = _mm512_mask_blend_ps(tiny0, out, zero);
    out = _mm512_mask_blend_ps(nan, out, x);
    out
}

/// silu_det over a slice of up to 16 values.
///
/// # Safety
/// AVX-512 (`available()`) must hold: the body reads AVX-512 instructions.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
pub unsafe fn silu_det_slice(x: &[f32; 16], y: &mut [f32; 16]) {
    _mm512_storeu_ps(y.as_mut_ptr(), silu_det16(_mm512_loadu_ps(x.as_ptr())));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn layout(ty: WType) -> SlotLayout {
        SlotLayout {
            gate: ty,
            up: ty,
            down: ty,
            n_embd: 256,
            n_ff: 256,
        }
    }

    fn call(lay: &SlotLayout, xp: &XPrep) -> std::thread::Result<()> {
        let slot = vec![0u8; lay.bytes()];
        let mut act = vec![0f32; lay.n_ff];
        catch_unwind(AssertUnwindSafe(|| unsafe {
            gate_up_groups(&slot, lay, xp, 0..1, &mut act)
        }))
    }

    #[test]
    fn gate_up_groups_refuses_a_mismatched_k() {
        if !available() {
            return;
        }
        let lay = layout(WType::Q4K);
        let mut xp = XPrep::default();
        unsafe { xp.build(&vec![0f32; lay.n_embd + 256], &[WType::Q4K]) };
        assert_eq!(xp.k, lay.n_embd + 256);
        assert!(call(&lay, &xp).is_err(), "a k the slot's rows do not have");
    }

    #[test]
    fn gate_up_groups_refuses_a_missing_table() {
        if !available() {
            return;
        }
        // Q4_K reads t16 and Q5_K reads t32; x prepared for the other type leaves the table empty.
        for (ty, built) in [(WType::Q4K, WType::Q5K), (WType::Q5K, WType::Q4K)] {
            let lay = layout(ty);
            let mut xp = XPrep::default();
            unsafe { xp.build(&vec![0f32; lay.n_embd], &[built]) };
            assert!(
                call(&lay, &xp).is_err(),
                "{ty:?} with x prepared for {built:?}"
            );
        }
    }

    #[test]
    fn gate_up_groups_runs_with_the_right_k_and_tables() {
        if !available() {
            return;
        }
        for ty in [WType::Q4K, WType::Q5K, WType::Q6K, WType::Q8_0] {
            let lay = layout(ty);
            let mut xp = XPrep::default();
            unsafe { xp.build(&vec![0f32; lay.n_embd], &[ty]) };
            assert!(call(&lay, &xp).is_ok(), "{ty:?}");
        }
    }
}
