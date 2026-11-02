// y[i] = W·x[i] for `tokens` input vectors (the matvec_tokens_{type} kernels): matvec.glsl's workgroup (8 rows × 32
// lanes, lane t sums the 32-value units t, t + 32, ... of its row, 32 partials meet in reduce.glsl's tree), with a
// token dimension. A lane decodes each weight unit once (quant.glsl's unit_dot_n) and adds its dot with every token's
// input to that token's accumulator, in unit_dot's order, so each token's result is matvec's. Token i's input is at
// x + 4·i·x_stride, its output at y + 4·i·y_stride (strides in values; x_stride a multiple of 4). kernels::twin::matvec's
// matvec_tokens mirrors it. The including file defines WTYPE.
#define RED_ROWS 8u
#define RED_N 32u
#include "reduce.glsl"

layout(local_size_x = 32, local_size_y = 8) in;
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
    uint lane = gl_LocalInvocationID.x, r = gl_LocalInvocationID.y;
    uint row = gl_WorkGroupID.x * 8u + r;
    // Constant indices only (every loop over acc is unrolled to MAX_TOKENS and cut at p.tokens), so the compiler can
    // keep the accumulators in registers.
    precise float acc[MAX_TOKENS];
    [[unroll]] for (uint i = 0u; i < MAX_TOKENS; i++) {
        acc[i] = 0.0;
    }
    if (row < p.rows) {
        uint64_t wr = p.w + uint64_t(row) * uint64_t(row_bytes(WTYPE, p.k));
        for (uint u = lane; u < p.k / 32u; u += 32u) {
            unit_dot_n(WTYPE, wr, u, p.x, p.x_stride, p.tokens, acc);
        }
    }
    // Every invocation reaches each red_sum (it holds barriers); p.tokens is uniform.
    [[unroll]] for (uint i = 0u; i < MAX_TOKENS; i++) {
        if (i >= p.tokens) {
            break;
        }
        float s = red_sum(r, lane, acc[i]);
        if (lane == 0u && row < p.rows) {
            F32s(p.y + 4ul * uint64_t(i) * uint64_t(p.y_stride)).v[row] = s;
        }
    }
}
