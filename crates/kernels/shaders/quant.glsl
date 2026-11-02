// The six weight types Artemis accepts, in ggml's block layouts. Codes match kernels::wtype_code.
// Rows are addressed as bytes, because Q8_0 and Q6_K blocks (34 and 210 bytes) are only 2-byte aligned.
const uint T_F32 = 0u;
const uint T_BF16 = 1u;
const uint T_Q8_0 = 2u;
const uint T_Q4_K = 3u;
const uint T_Q5_K = 4u;
const uint T_Q6_K = 5u;

// Bytes in one row of k values.
uint row_bytes(uint t, uint k) {
    if (t == T_F32) return 4u * k;
    if (t == T_BF16) return 2u * k;
    if (t == T_Q8_0) return k / 32u * 34u;
    if (t == T_Q4_K) return k / 256u * 144u;
    if (t == T_Q5_K) return k / 256u * 176u;
    return k / 256u * 210u;
}

float f16_at(uint64_t a) {
    return unpackHalf2x16(uint(U16s(a).v[0])).x;
}

float bf16_at(uint64_t a) {
    return uintBitsToFloat(uint(U16s(a).v[0]) << 16);
}

uint u8_at(uint64_t a) {
    return uint(U8s(a).v[0]);
}

// ggml's get_scale_min_k4: (scale, min) of K-quant sub-block j (0..8), from the 12 packed bytes at s.
uvec2 scale_min_k4(uint64_t s, uint j) {
    if (j < 4u) {
        return uvec2(u8_at(s + j) & 63u, u8_at(s + j + 4u) & 63u);
    }
    return uvec2((u8_at(s + j + 4u) & 0xFu) | ((u8_at(s + j - 4u) >> 6) << 4),
                 (u8_at(s + j + 4u) >> 4) | ((u8_at(s + j) >> 6) << 4));
}

// Element i of the row at address `row`, dequantized in the same f32 operations as qwen36::quant::dequant_row.
float dq(uint t, uint64_t row, uint i) {
    if (t == T_F32) {
        return F32s(row).v[i];
    }
    if (t == T_BF16) {
        return bf16_at(row + uint64_t(2u * i));
    }
    if (t == T_Q8_0) {
        uint64_t b = row + uint64_t(34u * (i / 32u));
        return f16_at(b) * float(int(I8s(b + 2ul).v[i % 32u]));
    }
    if (t == T_Q4_K || t == T_Q5_K) {
        uint64_t b = row + uint64_t((t == T_Q4_K ? 144u : 176u) * (i / 256u));
        uint e = i % 256u;
        uint c = e / 64u, hi = (e % 64u) / 32u, l = e % 32u, j = 2u * c + hi;
        uvec2 sm = scale_min_k4(b + 4ul, j);
        uint q = (u8_at(b + (t == T_Q4_K ? 16ul : 48ul) + uint64_t(32u * c + l)) >> (4u * hi)) & 0xFu;
        if (t == T_Q5_K && ((u8_at(b + 16ul + uint64_t(l)) >> j) & 1u) != 0u) {
            q += 16u;
        }
        return (f16_at(b) * float(sm.x)) * float(q) - f16_at(b + 2ul) * float(sm.y);
    }
    // Q6_K: 128 bytes of low nibbles, 64 of high 2-bit pairs, 16 int8 scales, f16 d.
    uint64_t b = row + uint64_t(210u * (i / 256u));
    uint e = i % 256u;
    uint n = e / 128u, m = (e % 128u) / 32u, l = e % 32u;
    uint lo = (u8_at(b + uint64_t(64u * n + 32u * (m & 1u) + l)) >> (4u * (m >> 1))) & 0xFu;
    uint hi = (u8_at(b + uint64_t(128u + 32u * n + l)) >> (2u * m)) & 3u;
    float sc = float(int(I8s(b + uint64_t(192u + 8u * n + 2u * m + l / 16u)).v[0]));
    return (f16_at(b + 208ul) * sc) * float(int(lo | (hi << 4)) - 32);
}
// Σ w·x over 32-value unit u of the row at `row` (elements 32u .. 32u + 32), with x at address xa. A unit is a Q8_0
// block, a K-quant sub-block, or a quarter of a Q6_K half-block with two 16-value scales.
// kernels::twin::matvec::unit_dot mirrors every operation, and `precise` keeps each product apart from the sum it joins
// (no fused multiply-add), so the two agree bit for bit.
//
// Loads: x four floats at a time; F32, BF16, Q4_K and Q5_K weights 16 bytes at a time (their units
// start 16-byte aligned in a 16-byte-aligned row); Q8_0 and Q6_K two bytes at a time (their blocks, 34 and 210 bytes,
// are only 2-byte aligned). dispatch.rs checks the addresses: `matvec` asserts x 16-byte aligned and the weights
// 16-byte aligned (F32, BF16, Q4_K, Q5_K rows) or 2-byte aligned (Q8_0, Q6_K rows); the expert dispatches assert their
// activations and matrices 16-byte aligned, and `check_slot_addr` the slot entries (`slot_entry` only encodes). The
// sums still run one value at a time, in order.
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer QuantV4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer QuantU4 { uvec4 v[]; };

// Byte l (0..32) of the 32 bytes a, b.
uint byte32(uvec4 a, uvec4 b, uint l) {
    uvec4 w = l < 16u ? a : b;
    uint k = l % 16u;
    return (w[k / 4u] >> (8u * (k % 4u))) & 0xFFu;
}

// Byte i (0..12) of the 12 bytes s.
uint byte12(uvec3 s, uint i) {
    return (s[i / 4u] >> (8u * (i % 4u))) & 0xFFu;
}

float unit_dot(uint t, uint64_t row, uint u, uint64_t xa) {
    QuantV4 x = QuantV4(xa + uint64_t(128u * u));
    if (t == T_F32) {
        QuantV4 w = QuantV4(row + uint64_t(128u * u));
        precise float s = 0.0;
        [[unroll]] for (uint i4 = 0u; i4 < 8u; i4++) {
            vec4 wv = w.v[i4], xv = x.v[i4];
            s += wv.x * xv.x;
            s += wv.y * xv.y;
            s += wv.z * xv.z;
            s += wv.w * xv.w;
        }
        return s;
    }
    if (t == T_BF16) {
        // Eight values a load: value 2k in word k's low half, 2k + 1 in its high half.
        QuantU4 w = QuantU4(row + uint64_t(64u * u));
        precise float s = 0.0;
        [[unroll]] for (uint i8 = 0u; i8 < 4u; i8++) {
            uvec4 wv = w.v[i8];
            vec4 xa4 = x.v[2u * i8], xb4 = x.v[2u * i8 + 1u];
            s += uintBitsToFloat(wv.x << 16) * xa4.x;
            s += uintBitsToFloat(wv.x & 0xFFFF0000u) * xa4.y;
            s += uintBitsToFloat(wv.y << 16) * xa4.z;
            s += uintBitsToFloat(wv.y & 0xFFFF0000u) * xa4.w;
            s += uintBitsToFloat(wv.z << 16) * xb4.x;
            s += uintBitsToFloat(wv.z & 0xFFFF0000u) * xb4.y;
            s += uintBitsToFloat(wv.w << 16) * xb4.z;
            s += uintBitsToFloat(wv.w & 0xFFFF0000u) * xb4.w;
        }
        return s;
    }
    if (t == T_Q8_0) {
        uint64_t b = row + uint64_t(34u * u);
        U16s q = U16s(b + 2ul);
        precise float s = 0.0;
        [[unroll]] for (uint i4 = 0u; i4 < 8u; i4++) {
            vec4 xv = x.v[i4];
            int p0 = int(uint(q.v[2u * i4])), p1 = int(uint(q.v[2u * i4 + 1u]));
            s += float(bitfieldExtract(p0, 0, 8)) * xv.x;
            s += float(bitfieldExtract(p0, 8, 8)) * xv.y;
            s += float(bitfieldExtract(p1, 0, 8)) * xv.z;
            s += float(bitfieldExtract(p1, 8, 8)) * xv.w;
        }
        precise float r = f16_at(b) * s;
        return r;
    }
    if (t == T_Q4_K || t == T_Q5_K) {
        uint64_t b = row + uint64_t((t == T_Q4_K ? 144u : 176u) * (u / 8u));
        uint j = u % 8u, c = j / 2u, sh = 4u * (j % 2u);
        // d and dmin, then the 12 bytes of packed scales and mins (ggml's get_scale_min_k4).
        uvec4 head = QuantU4(b).v[0];
        uvec3 sc = head.yzw;
        uvec2 sm;
        if (j < 4u) {
            sm = uvec2(byte12(sc, j) & 63u, byte12(sc, j + 4u) & 63u);
        } else {
            sm = uvec2((byte12(sc, j + 4u) & 0xFu) | ((byte12(sc, j - 4u) >> 6) << 4),
                       (byte12(sc, j + 4u) >> 4) | ((byte12(sc, j) >> 6) << 4));
        }
        QuantU4 qw = QuantU4(b + (t == T_Q4_K ? 16ul : 48ul) + uint64_t(32u * c));
        uvec4 qa = qw.v[0], qb = qw.v[1];
        uvec4 ha = uvec4(0u), hb = uvec4(0u);
        if (t == T_Q5_K) {
            QuantU4 hw = QuantU4(b + 16ul);
            ha = hw.v[0];
            hb = hw.v[1];
        }
        precise float sq = 0.0, sx = 0.0;
        [[unroll]] for (uint l4 = 0u; l4 < 8u; l4++) {
            vec4 xv4 = x.v[l4];
            [[unroll]] for (uint i = 0u; i < 4u; i++) {
                uint l = 4u * l4 + i;
                uint q = (byte32(qa, qb, l) >> sh) & 0xFu;
                if (t == T_Q5_K && ((byte32(ha, hb, l) >> j) & 1u) != 0u) {
                    q += 16u;
                }
                float xv = xv4[i];
                sq += float(q) * xv;
                sx += xv;
            }
        }
        vec2 dm = unpackHalf2x16(head.x);
        precise float r = (dm.x * float(sm.x)) * sq - (dm.y * float(sm.y)) * sx;
        return r;
    }
    // Q6_K: unit j of a block is half n = j / 4, quarter m = j % 4: 128 bytes of low nibbles, 64 of high 2-bit pairs,
    // 16 int8 scales, f16 d.
    uint64_t b = row + uint64_t(210u * (u / 8u));
    uint j = u % 8u, n = j / 4u, m = j % 4u;
    U16s ql = U16s(b + uint64_t(64u * n + 32u * (m & 1u)));
    U16s qh = U16s(b + uint64_t(128u + 32u * n));
    int scs = int(uint(U16s(b + uint64_t(192u + 8u * n + 2u * m)).v[0]));
    uint shl = 4u * (m >> 1), shh = 2u * m;
    precise float sa = 0.0, sb = 0.0;
    [[unroll]] for (uint i4 = 0u; i4 < 8u; i4++) {
        vec4 xv = x.v[i4];
        uint l0 = uint(ql.v[2u * i4]), l1 = uint(ql.v[2u * i4 + 1u]);
        uint h0 = uint(qh.v[2u * i4]), h1 = uint(qh.v[2u * i4 + 1u]);
        uint lo[4] = uint[4](l0 & 0xFFu, l0 >> 8, l1 & 0xFFu, l1 >> 8);
        uint hi[4] = uint[4](h0 & 0xFFu, h0 >> 8, h1 & 0xFFu, h1 >> 8);
        [[unroll]] for (uint e = 0u; e < 4u; e++) {
            uint v6 = ((lo[e] >> shl) & 0xFu) | (((hi[e] >> shh) & 3u) << 4);
            float prod = float(int(v6) - 32) * xv[e];
            if (4u * i4 + e < 16u) {
                sa += prod;
            } else {
                sb += prod;
            }
        }
    }
    float sca = float(bitfieldExtract(scs, 0, 8));
    float scb = float(bitfieldExtract(scs, 8, 8));
    precise float r = f16_at(b + 208ul) * (sca * sa + scb * sb);
    return r;
}

// The most tokens a token kernel carries: kernels::tokens::MAX_TOKENS (a test pins the two equal).
const uint MAX_TOKENS = 16u;

// Decodes unit u of the row at `row` into wv (exact floats: integers, or bf16/f32 values) and the block's scales c0..c2.
// Type-specific scalars: Q8_0 c0 = d; Q4_K/Q5_K c0 = d·scale, c1 = dmin·min; Q6_K c0 = d, c1 and c2 = the two scales.
// unit_dot_n and unit_dot_n_at both decode through this and dot through unit_dot_loaded, so the decode and the
// per-token sums exist once.
void unit_load(uint t, uint64_t row, uint u, out float wv[32], out float c0, out float c1, out float c2) {
    c0 = 0.0;
    c1 = 0.0;
    c2 = 0.0;
    if (t == T_F32) {
        QuantV4 w = QuantV4(row + uint64_t(128u * u));
        [[unroll]] for (uint i4 = 0u; i4 < 8u; i4++) {
            vec4 wq = w.v[i4];
            [[unroll]] for (uint e = 0u; e < 4u; e++) {
                wv[4u * i4 + e] = wq[e];
            }
        }
    } else if (t == T_BF16) {
        QuantU4 w = QuantU4(row + uint64_t(64u * u));
        [[unroll]] for (uint i8 = 0u; i8 < 4u; i8++) {
            uvec4 wq = w.v[i8];
            [[unroll]] for (uint k = 0u; k < 4u; k++) {
                wv[8u * i8 + 2u * k] = uintBitsToFloat(wq[k] << 16);
                wv[8u * i8 + 2u * k + 1u] = uintBitsToFloat(wq[k] & 0xFFFF0000u);
            }
        }
    } else if (t == T_Q8_0) {
        uint64_t b = row + uint64_t(34u * u);
        U16s q = U16s(b + 2ul);
        [[unroll]] for (uint i4 = 0u; i4 < 8u; i4++) {
            int p0 = int(uint(q.v[2u * i4])), p1 = int(uint(q.v[2u * i4 + 1u]));
            wv[4u * i4] = float(bitfieldExtract(p0, 0, 8));
            wv[4u * i4 + 1u] = float(bitfieldExtract(p0, 8, 8));
            wv[4u * i4 + 2u] = float(bitfieldExtract(p1, 0, 8));
            wv[4u * i4 + 3u] = float(bitfieldExtract(p1, 8, 8));
        }
        c0 = f16_at(b);
    } else if (t == T_Q4_K || t == T_Q5_K) {
        uint64_t b = row + uint64_t((t == T_Q4_K ? 144u : 176u) * (u / 8u));
        uint j = u % 8u, c = j / 2u, sh = 4u * (j % 2u);
        uvec4 head = QuantU4(b).v[0];
        uvec3 sc = head.yzw;
        uvec2 sm;
        if (j < 4u) {
            sm = uvec2(byte12(sc, j) & 63u, byte12(sc, j + 4u) & 63u);
        } else {
            sm = uvec2((byte12(sc, j + 4u) & 0xFu) | ((byte12(sc, j - 4u) >> 6) << 4),
                       (byte12(sc, j + 4u) >> 4) | ((byte12(sc, j) >> 6) << 4));
        }
        QuantU4 qw = QuantU4(b + (t == T_Q4_K ? 16ul : 48ul) + uint64_t(32u * c));
        uvec4 qa = qw.v[0], qb = qw.v[1];
        uvec4 ha = uvec4(0u), hb = uvec4(0u);
        if (t == T_Q5_K) {
            QuantU4 hw = QuantU4(b + 16ul);
            ha = hw.v[0];
            hb = hw.v[1];
        }
        [[unroll]] for (uint l = 0u; l < 32u; l++) {
            uint q = (byte32(qa, qb, l) >> sh) & 0xFu;
            if (t == T_Q5_K && ((byte32(ha, hb, l) >> j) & 1u) != 0u) {
                q += 16u;
            }
            wv[l] = float(q);
        }
        vec2 dm = unpackHalf2x16(head.x);
        precise float s0 = dm.x * float(sm.x);
        precise float s1 = dm.y * float(sm.y);
        c0 = s0;
        c1 = s1;
    } else {
        uint64_t b = row + uint64_t(210u * (u / 8u));
        uint j = u % 8u, nn = j / 4u, m = j % 4u;
        U16s ql = U16s(b + uint64_t(64u * nn + 32u * (m & 1u)));
        U16s qh = U16s(b + uint64_t(128u + 32u * nn));
        int scs = int(uint(U16s(b + uint64_t(192u + 8u * nn + 2u * m)).v[0]));
        uint shl = 4u * (m >> 1), shh = 2u * m;
        [[unroll]] for (uint i4 = 0u; i4 < 8u; i4++) {
            uint l0 = uint(ql.v[2u * i4]), l1 = uint(ql.v[2u * i4 + 1u]);
            uint h0 = uint(qh.v[2u * i4]), h1 = uint(qh.v[2u * i4 + 1u]);
            uint lo[4] = uint[4](l0 & 0xFFu, l0 >> 8, l1 & 0xFFu, l1 >> 8);
            uint hi[4] = uint[4](h0 & 0xFFu, h0 >> 8, h1 & 0xFFu, h1 >> 8);
            [[unroll]] for (uint e = 0u; e < 4u; e++) {
                uint v6 = ((lo[e] >> shl) & 0xFu) | (((hi[e] >> shh) & 3u) << 4);
                wv[4u * i4 + e] = float(int(v6) - 32);
            }
        }
        c0 = f16_at(b + 208ul);
        c1 = float(bitfieldExtract(scs, 0, 8));
        c2 = float(bitfieldExtract(scs, 8, 8));
    }
}

// One input's dot with a unit unit_load decoded: unit_dot's products and sums, in unit_dot's order and `precise`, so
// unit_dot(t, row, u, xa) == unit_dot_loaded(t, wv, c0, c1, c2, xa) bit for bit. xa is the 16-byte aligned address of
// the unit's 32 inputs. Every result is `precise` (its last operation is a NoContraction add, subtract or multiply),
// so the caller's `acc += ...` cannot fuse into it.
float unit_dot_loaded(uint t, float wv[32], float c0, float c1, float c2, uint64_t xa) {
    QuantV4 x = QuantV4(xa);
    if (t == T_F32 || t == T_BF16) {
        precise float s = 0.0;
        [[unroll]] for (uint i4 = 0u; i4 < 8u; i4++) {
            vec4 xv = x.v[i4];
            [[unroll]] for (uint e = 0u; e < 4u; e++) {
                s += wv[4u * i4 + e] * xv[e];
            }
        }
        return s;
    } else if (t == T_Q8_0) {
        precise float s = 0.0;
        [[unroll]] for (uint i4 = 0u; i4 < 8u; i4++) {
            vec4 xv = x.v[i4];
            [[unroll]] for (uint e = 0u; e < 4u; e++) {
                s += wv[4u * i4 + e] * xv[e];
            }
        }
        precise float r = c0 * s;
        return r;
    } else if (t == T_Q4_K || t == T_Q5_K) {
        precise float sq = 0.0, sx = 0.0;
        [[unroll]] for (uint l4 = 0u; l4 < 8u; l4++) {
            vec4 xv4 = x.v[l4];
            [[unroll]] for (uint e = 0u; e < 4u; e++) {
                sq += wv[4u * l4 + e] * xv4[e];
                sx += xv4[e];
            }
        }
        precise float r = c0 * sq - c1 * sx;
        return r;
    } else {
        precise float sa = 0.0, sb = 0.0;
        [[unroll]] for (uint i4 = 0u; i4 < 8u; i4++) {
            vec4 xv = x.v[i4];
            [[unroll]] for (uint e = 0u; e < 4u; e++) {
                float prod = wv[4u * i4 + e] * xv[e];
                if (4u * i4 + e < 16u) {
                    sa += prod;
                } else {
                    sb += prod;
                }
            }
        }
        precise float r = c0 * (c1 * sa + c2 * sb);
        return r;
    }
}

// unit_dot for n inputs (the token kernels): decodes unit u of the row at `row` once, then adds its dot with
// token i's input (at x + 4·i·x_stride bytes, x_stride in values, 16-byte aligned like unit_dot's xa) to acc[i], for
// i < n <= MAX_TOKENS. Each product and sum is unit_dot's, in unit_dot's order and `precise`, so
// acc[i] += unit_dot(t, row, u, x + 4·i·x_stride) gives the same bits. The weights are decoded to exact floats first
// (integers, or bf16/f32 values) and the block's scales kept apart: that is all unit_dot does before multiplying.
void unit_dot_n(uint t, uint64_t row, uint u, uint64_t xa, uint x_stride, uint n, inout precise float acc[MAX_TOKENS]) {
    float wv[32];
    float c0, c1, c2;
    unit_load(t, row, u, wv, c0, c1, c2);
    // Unrolled to MAX_TOKENS and cut at n, so every acc[i] is a constant index and acc can live in registers.
    [[unroll]] for (uint i = 0u; i < MAX_TOKENS; i++) {
        if (i >= n) {
            break;
        }
        acc[i] += unit_dot_loaded(t, wv, c0, c1, c2, xa + uint64_t(4u * i * x_stride) + uint64_t(128u * u));
    }
}

// unit_dot_n for inputs at per-token offsets (the experts' token kernels): for each set bit i of `mask`
// (ascending, at most MAX_TOKENS), token i's input is at xa + 4·xoff[i] bytes (xoff in values, 16-byte aligned like
// unit_dot's xa), and acc[i] += unit_dot(t, row, u, xa + 4·xoff[i]) with the unit decoded once. experts_down_tokens
// needs it because token i's input is activation (i·k + j_i) of its routed index j_i, which no base plus i·stride
// gives; the gate/up kernel passes xoff[i] = i·n_embd and skips the tokens that did not choose the expert. The decode
// and the per-token sums are unit_dot_n's: both go through unit_load and unit_dot_loaded.
void unit_dot_n_at(uint t, uint64_t row, uint u, uint64_t xa, uint xoff[MAX_TOKENS], uint mask,
                   inout precise float acc[MAX_TOKENS]) {
    float wv[32];
    float c0, c1, c2;
    unit_load(t, row, u, wv, c0, c1, c2);
    for (uint m = mask; m != 0u; m &= m - 1u) {
        uint i = uint(findLSB(m));
        acc[i] += unit_dot_loaded(t, wv, c0, c1, c2, xa + uint64_t(4u * xoff[i]) + uint64_t(128u * u));
    }
}
