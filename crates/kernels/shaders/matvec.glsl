// y = W·x for `rows` rows of k values of type WTYPE (the matvec_{type} kernels). A workgroup is 8 rows × 32 lanes:
// lane t of a row sums the 32-value units t, t + 32, ... of that row in order (quant.glsl's unit_dot), and the 32
// partials meet in reduce.glsl's tree. kernels::twin::matvec mirrors it. The including file defines WTYPE.
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
} p;

void main() {
    uint lane = gl_LocalInvocationID.x, r = gl_LocalInvocationID.y;
    uint row = gl_WorkGroupID.x * 8u + r;
    float acc = 0.0;
    if (row < p.rows) {
        uint64_t wr = p.w + uint64_t(row) * uint64_t(row_bytes(WTYPE, p.k));
        for (uint u = lane; u < p.k / 32u; u += 32u) {
            acc += unit_dot(WTYPE, wr, u, p.x);
        }
    }
    float s = red_sum(r, lane, acc);
    if (lane == 0u && row < p.rows) {
        F32s(p.y).v[row] = s;
    }
}
