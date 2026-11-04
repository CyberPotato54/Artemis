// y = W·x for `rows` rows of k values of type WTYPE (the matvec_{type} kernels). A workgroup is 8 subgroups of 32
// lanes, one row each: lane t of a row sums the 32-value units t, t + 32, ... of that row in order (quant.glsl's
// lane_sum), and the 32 partials meet in reduce.glsl's tree, done by subgroup shuffles (tree32.glsl) on a pipeline
// pinned to subgroups of 32: no shared memory and no barrier. kernels::twin::matvec mirrors it. The including file
// defines WTYPE and enables GL_KHR_shader_subgroup_basic and GL_KHR_shader_subgroup_shuffle.
#include "tree32.glsl"

layout(local_size_x = 256) in;
layout(push_constant) uniform Push {
    uint64_t w;
    uint64_t x;
    uint64_t y;
    uint rows;
    uint k;
} p;

void main() {
    uint lane = gl_SubgroupInvocationID;
    uint row = gl_WorkGroupID.x * 8u + gl_SubgroupID;
    float acc = 0.0;
    if (row < p.rows) {
        uint64_t wr = p.w + uint64_t(row) * uint64_t(row_bytes(WTYPE, p.k));
        acc = lane_sum(WTYPE, wr, lane, p.k / 32u, p.x);
    }
    // Lanes past a short row's units hold +0.0, as red_sum's do, so lpr is always 32 and `zero` is unused.
    float s = tree32(acc, 32u, 0.0);
    if (lane == 0u && row < p.rows) {
        F32s(p.y).v[row] = s;
    }
}
