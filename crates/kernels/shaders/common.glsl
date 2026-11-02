// Included first by every kernel: the extensions, buffer types reached through device addresses (there are no
// descriptor sets), and the elementwise math. kernels::twin mirrors each function.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_control_flow_attributes : enable

layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32s { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32s { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer U16s { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) buffer U8s { uint8_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) buffer I8s { int8_t v[]; };

float neg_inf() {
    return uintBitsToFloat(0xFF800000u);
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float silu(float x) {
    return x / (1.0 + exp(-x));
}

// log(1 + e) for e in [0, 1], to f32 relative precision: 2·atanh(z) with z = e / (2 + e) <= 1/3, as the series
// 2(z + z^3/3 + ... + z^15/15). log(1 + e) itself loses e's low bits once 1 + e rounds.
float log1p_small(float e) {
    float z = e / (2.0 + e);
    float z2 = z * z;
    float p = 1.0 / 15.0;
    p = p * z2 + 1.0 / 13.0;
    p = p * z2 + 1.0 / 11.0;
    p = p * z2 + 1.0 / 9.0;
    p = p * z2 + 1.0 / 7.0;
    p = p * z2 + 1.0 / 5.0;
    p = p * z2 + 1.0 / 3.0;
    p = p * z2 + 1.0;
    return 2.0 * z * p;
}

// llama.cpp's softplus: u above 20, log(1 + exp(u)) otherwise, computed as max(u, 0) + log1p(exp(-|u|)).
float softplus(float u) {
    return u > 20.0 ? u : max(u, 0.0) + log1p_small(exp(-abs(u)));
}

// f32 to f16 bits, rounding to nearest even (as half::f16::from_f32 does), independent of the driver's
// conversion rounding mode. NaN keeps only its quiet bit.
uint f16_bits(float f) {
    uint x = floatBitsToUint(f);
    uint sign = (x >> 16) & 0x8000u;
    uint e = (x >> 23) & 0xFFu;
    uint m = x & 0x7FFFFFu;
    if (e == 0xFFu) {
        return sign | 0x7C00u | (m != 0u ? 0x200u : 0u);
    }
    int ee = int(e) - 112;
    if (ee >= 31) {
        return sign | 0x7C00u;
    }
    if (ee <= 0) {
        if (ee < -10) {
            return sign;
        }
        m |= 0x800000u;
        uint shift = uint(14 - ee);
        uint h = m >> shift;
        uint rem = m & ((1u << shift) - 1u);
        uint halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (h & 1u) != 0u)) {
            h += 1u;
        }
        return sign | h;
    }
    uint h = sign | (uint(ee) << 10) | (m >> 13);
    uint rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) {
        h += 1u;
    }
    return h;
}
