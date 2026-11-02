// Fixed-order reductions in shared memory: within each of RED_ROWS rows, a tree over RED_N lanes that
// halves its stride each step, so the order depends only on RED_N, never on the subgroup size.
// kernels::twin::tree mirrors it. Define RED_N (and RED_ROWS, default 1) before including. Every invocation of the
// workgroup must reach each call: the calls contain barriers.
#ifndef RED_ROWS
#define RED_ROWS 1u
#endif
shared float red_[RED_ROWS * RED_N];

float red_sum(uint row, uint lane, float v) {
    uint b = row * RED_N;
    red_[b + lane] = v;
    barrier();
    for (uint s = RED_N / 2u; s > 0u; s >>= 1u) {
        if (lane < s) {
            red_[b + lane] += red_[b + lane + s];
        }
        barrier();
    }
    float r = red_[b];
    barrier();
    return r;
}

float red_max(uint row, uint lane, float v) {
    uint b = row * RED_N;
    red_[b + lane] = v;
    barrier();
    for (uint s = RED_N / 2u; s > 0u; s >>= 1u) {
        if (lane < s) {
            red_[b + lane] = max(red_[b + lane], red_[b + lane + s]);
        }
        barrier();
    }
    float r = red_[b];
    barrier();
    return r;
}
