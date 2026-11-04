// reduce.glsl's red_sum over 32 lanes (RED_N = 32), by subgroup shuffles instead of shared memory and barriers: the
// same pairs in the same order, so the same bits (kernels::twin::tree). Needs the subgroup size pinned at 32 (the
// kernel's pipeline is built with a required subgroup size of 32 and full subgroups) and the 32 lanes of a row in one
// subgroup. Every lane of the subgroup must reach the call: shuffles read the other lanes' values.
//
// Step s adds lane l + s's value to lane l's for l < s, as red_sum does: lane l (bit s clear) computes own + partner,
// and every value that reaches lane 0 was made that way at an earlier step (lane l + s has its higher bits clear), so
// lane 0's sum is red_sum's operation for operation. Lanes with bit s set compute partner + own, which is the same
// float sum; their results only feed lanes that are discarded.
//
// lpr (lanes a row uses) is 32 or 16. With 16, the row's lanes 16 to 31 of red_sum's tree hold +0.0 (no units), so
// its first step adds +0.0 and the shuffles stay inside each half of the subgroup, which holds a second row. That
// +0.0 is `zero`, a push constant the host sets to +0.0: this driver folds x + 0.0 to x, which would keep a -0.0 sum
// where red_sum's add (a zero read from shared memory) gives +0.0.
// The including kernel enables GL_KHR_shader_subgroup_basic and GL_KHR_shader_subgroup_shuffle before its includes.

float tree32(float v, uint lpr, float zero) {
    precise float a = v;
    if (lpr == 16u) {
        a = a + zero;
    } else {
        a = a + subgroupShuffleXor(a, 16u);
    }
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}
