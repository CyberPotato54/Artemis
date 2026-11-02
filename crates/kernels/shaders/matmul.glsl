// Y = W·X for a batch of input vectors (the matmul_{type} kernels): y[t][r] = Σ_i w[r][i]·x[t][i] for `rows` rows
// of k values of type WTYPE and `tokens` vectors, x as [tokens][k] and y as [tokens][rows]. A workgroup of 16 × 16
// invocations computes 64 rows × 64 tokens, each invocation 4 rows × 4 tokens. k is taken 32 values at a time: the
// tile's weights are dequantized into shared memory as `dq` gives them (qwen36::quant::dequant_row's values), its
// inputs are copied beside them, and each output adds its 32 products in order
// (`precise` accumulators, each product rounded then added, in order, so no lane is contracted differently). So every
// output sums w·x over i = 0, 1, … in order, in f32, whatever tile or batch it falls in. kernels::twin::matvec::matmul
// adds in the same order with s + a·b, so given the same dequantized values the two sums agree; the tests compare them
// within the kernel's tolerance, not bit for bit. The including file defines WTYPE.
layout(local_size_x = 16, local_size_y = 16) in;
layout(push_constant) uniform Push {
    uint64_t w;
    uint64_t x;
    uint64_t y;
    uint rows;
    uint k;
    uint tokens;
} p;

// [32][64]: the slice's 32 values of each of the tile's 64 rows, and of its 64 inputs.
shared float ws[2048];
shared float xs[2048];

void main() {
    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y;
    uint id = 16u * ty + tx;
    uint r0 = 64u * gl_WorkGroupID.x, t0 = 64u * gl_WorkGroupID.y;
    // Invocation `id` loads row (and input) id / 4, values 8·(id % 4) .. + 8 of each slice.
    uint lr = id / 4u, lk = 8u * (id % 4u);
    bool rin = r0 + lr < p.rows, tin = t0 + lr < p.tokens;
    uint64_t wrow = p.w + uint64_t(r0 + lr) * uint64_t(row_bytes(WTYPE, p.k));
    uint64_t xrow = p.x + 4ul * uint64_t(t0 + lr) * uint64_t(p.k);
    precise float acc[16];
    for (uint i = 0u; i < 16u; i++) {
        acc[i] = 0.0;
    }
    for (uint k0 = 0u; k0 < p.k; k0 += 32u) {
        for (uint e = 0u; e < 8u; e++) {
            uint kk = lk + e;
            ws[64u * kk + lr] = rin ? dq(WTYPE, wrow, k0 + kk) : 0.0;
            xs[64u * kk + lr] = tin ? F32s(xrow).v[k0 + kk] : 0.0;
        }
        barrier();
        for (uint kk = 0u; kk < 32u; kk++) {
            vec4 a = vec4(ws[64u * kk + 4u * ty], ws[64u * kk + 4u * ty + 1u], ws[64u * kk + 4u * ty + 2u],
                          ws[64u * kk + 4u * ty + 3u]);
            vec4 b = vec4(xs[64u * kk + 4u * tx], xs[64u * kk + 4u * tx + 1u], xs[64u * kk + 4u * tx + 2u],
                          xs[64u * kk + 4u * tx + 3u]);
            for (uint i = 0u; i < 4u; i++) {
                for (uint j = 0u; j < 4u; j++) {
                    acc[4u * i + j] = acc[4u * i + j] + a[i] * b[j];
                }
            }
        }
        barrier();
    }
    for (uint i = 0u; i < 4u; i++) {
        uint r = r0 + 4u * ty + i;
        for (uint j = 0u; j < 4u; j++) {
            uint t = t0 + 4u * tx + j;
            if (r < p.rows && t < p.tokens) {
                F32s(p.y).v[t * p.rows + r] = acc[4u * i + j];
            }
        }
    }
}
