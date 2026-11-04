// y[i] = W·x[i] for `tokens` input vectors (the matvec_tokens_{type} kernels): matvec.glsl's workgroup (8 subgroups
// of 32 lanes, one row each: lane t sums the 32-value units t, t + 32, ... of its row, and the 32 partials meet in
// reduce.glsl's tree, done by subgroup shuffles in tree32.glsl on a pipeline pinned to subgroups of 32), with a token
// dimension. A lane decodes each weight unit once (quant.glsl's unit_load) and adds its dot with every token's input
// (unit_dot_loaded) to that token's accumulator, in unit_dot's order, so each token's result is matvec's. Token i's
// input is at x + 4·i·x_stride, its output at y + 4·i·y_stride (strides in values; x_stride a multiple of 4).
// kernels::twin::matvec's matvec_tokens mirrors it. The including file defines WTYPE and enables
// GL_KHR_shader_subgroup_basic and GL_KHR_shader_subgroup_shuffle.
//
// TOKENS (specialization constant 0) is the dispatch's token count for 2 to 5 tokens and MAX_TOKENS otherwise
// (kernels::tokens::SPECIALIZED_TOKENS), as in the experts' token kernels: the accumulators are an array of TOKENS
// indexed only by unrolled loop counters, so they stay in registers, and a pipeline for a small count holds only its
// own. The general pipeline cuts its loops at p.tokens.
#include "tree32.glsl"

layout(constant_id = 0) const uint TOKENS = 16u;
layout(local_size_x = 256) in;
layout(push_constant) uniform Push {
    uint64_t w;
    uint64_t x;
    uint64_t y;
    uint rows;
    uint k;
    uint tokens;
    uint x_stride;
    uint y_stride;
    uint pad;
} p;

void main() {
    uint lane = gl_SubgroupInvocationID;
    uint row = gl_WorkGroupID.x * 8u + gl_SubgroupID;
    precise float acc[TOKENS];
    [[unroll]] for (uint i = 0u; i < TOKENS; i++) {
        acc[i] = 0.0;
    }
    if (row < p.rows) {
        uint64_t wr = p.w + uint64_t(row) * uint64_t(row_bytes(WTYPE, p.k));
        for (uint u = lane; u < p.k / 32u; u += 32u) {
            float wv[32];
            float c0, c1, c2;
            unit_load(WTYPE, wr, u, wv, c0, c1, c2);
            [[unroll]] for (uint i = 0u; i < TOKENS; i++) {
                if (i >= p.tokens) {
                    break;
                }
                uint64_t xa = p.x + uint64_t(4u * i * p.x_stride) + uint64_t(128u * u);
                acc[i] += unit_dot_loaded(WTYPE, wv, c0, c1, c2, xa);
            }
        }
    }
    // Every lane of the subgroup reaches each tree32 (shuffles read the other lanes); p.tokens is uniform. Lanes past a
    // short row's units hold +0.0, as red_sum's do, so lpr is always 32 and `zero` is unused (as in matvec.glsl).
    [[unroll]] for (uint i = 0u; i < TOKENS; i++) {
        if (i >= p.tokens) {
            break;
        }
        float s = tree32(acc[i], 32u, 0.0);
        if (lane == 0u && row < p.rows) {
            F32s(p.y + 4ul * uint64_t(i) * uint64_t(p.y_stride)).v[row] = s;
        }
    }
}
