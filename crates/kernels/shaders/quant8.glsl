// Loads for prefill's split-f16 cooperative-matrix kernels (inputs split by the decision of
// 2026-10-10). Included after common.glsl and quant.glsl by shaders that enable GL_EXT_control_flow_attributes and
// GL_EXT_shader_explicit_arithmetic_types_float16.

uint u16_at(uint64_t a) {
    return uint(U16s(a).v[0]);
}

// Eight values i0 .. i0 + 8 of the row at `row` (i0 a multiple of 8), each in exactly dq's f32 operations, and so
// qwen36::quant::dequant_row's: the eight share a block, so its scales are read once, and the quants are read two bytes
// at a time (every offset read here is even). `precise` keeps a scale's product from being fused into the subtraction
// after it, which dequant_row never fuses.
void dq_eight(uint t, uint64_t row, uint i0, out float v[8]) {
    if (t == T_F32) {
        [[unroll]] for (uint e = 0u; e < 8u; e++) {
            v[e] = F32s(row).v[i0 + e];
        }
        return;
    }
    if (t == T_BF16) {
        [[unroll]] for (uint e = 0u; e < 8u; e++) {
            v[e] = uintBitsToFloat(u16_at(row + uint64_t(2u * (i0 + e))) << 16);
        }
        return;
    }
    if (t == T_Q8_0) {
        uint64_t b = row + uint64_t(34u * (i0 / 32u));
        float d = f16_at(b);
        uint64_t q = b + 2ul + uint64_t(i0 % 32u);
        [[unroll]] for (uint e = 0u; e < 8u; e += 2u) {
            int pair = int(u16_at(q + uint64_t(e)));
            precise float a = d * float(bitfieldExtract(pair, 0, 8));
            precise float c = d * float(bitfieldExtract(pair, 8, 8));
            v[e] = a;
            v[e + 1u] = c;
        }
        return;
    }
    if (t == T_Q4_K || t == T_Q5_K) {
        uint64_t b = row + uint64_t((t == T_Q4_K ? 144u : 176u) * (i0 / 256u));
        uint e0 = i0 % 256u;
        uint c = e0 / 64u, hi = (e0 % 64u) / 32u, l0 = e0 % 32u, j = 2u * c + hi;
        uvec2 sm = scale_min_k4(b + 4ul, j);
        precise float dl = f16_at(b) * float(sm.x);
        precise float ml = f16_at(b + 2ul) * float(sm.y);
        uint64_t qs = b + (t == T_Q4_K ? 16ul : 48ul) + uint64_t(32u * c + l0);
        [[unroll]] for (uint e = 0u; e < 8u; e += 2u) {
            uint pair = u16_at(qs + uint64_t(e));
            uint qa = ((pair & 0xFFu) >> (4u * hi)) & 0xFu;
            uint qb = ((pair >> 8) >> (4u * hi)) & 0xFu;
            if (t == T_Q5_K) {
                uint hp = u16_at(b + 16ul + uint64_t(l0 + e));
                qa += (((hp & 0xFFu) >> j) & 1u) << 4;
                qb += (((hp >> 8) >> j) & 1u) << 4;
            }
            precise float a = dl * float(qa) - ml;
            precise float bb = dl * float(qb) - ml;
            v[e] = a;
            v[e + 1u] = bb;
        }
        return;
    }
    // Q6_K: 128 bytes of low nibbles, 64 of high 2-bit pairs, 16 int8 scales (one per 16 values), f16 d.
    uint64_t b = row + uint64_t(210u * (i0 / 256u));
    uint e0 = i0 % 256u;
    uint n = e0 / 128u, m = (e0 % 128u) / 32u, l0 = e0 % 32u;
    precise float d = f16_at(b + 208ul) * float(int(I8s(b + uint64_t(192u + 8u * n + 2u * m + l0 / 16u)).v[0]));
    uint64_t ql = b + uint64_t(64u * n + 32u * (m & 1u) + l0);
    uint64_t qh = b + uint64_t(128u + 32u * n + l0);
    [[unroll]] for (uint e = 0u; e < 8u; e += 2u) {
        uint lp = u16_at(ql + uint64_t(e)), hp = u16_at(qh + uint64_t(e));
        uint la = ((lp & 0xFFu) >> (4u * (m >> 1))) & 0xFu, lb = ((lp >> 8) >> (4u * (m >> 1))) & 0xFu;
        uint ha = ((hp & 0xFFu) >> (2u * m)) & 3u, hb = ((hp >> 8) >> (2u * m)) & 3u;
        precise float a = d * float(int(la | (ha << 4)) - 32);
        precise float bb = d * float(int(lb | (hb << 4)) - 32);
        v[e] = a;
        v[e + 1u] = bb;
    }
}

// `a` and `b` each as two f16s, packed in pairs: hi = f16(v) and lo = f16(v − hi), where v − hi is exact in f32. Returns
// hi's pair and sets lo's. hi + lo is within about 2⁻²² of v (kernels::twin::matvec::split16 mirrors it; a device that
// rounded hi another way would still get hi + lo near v, since lo takes up the difference). Limits: a value above
// f16's 65,504 makes hi infinite and lo NaN, so the outputs it reaches are NaN and the step fails on its logits; and
// the 2⁻²² holds only where f16 subnormals survive (they do on the RX 9070 XT, which preserves f16 denormals; a device
// that flushed them would lose lo for values under about 6e-5 · 2¹¹, still within f16's own error).
uint split2(float a, float b, out uint lo) {
    f16vec2 h = f16vec2(float16_t(a), float16_t(b));
    lo = packFloat2x16(f16vec2(float16_t(a - float(h.x)), float16_t(b - float(h.y))));
    return packFloat2x16(h);
}
