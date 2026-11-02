// Y = W·X for a batch on cooperative matrices: y[t][r] = Σ_i w[r][i]·x[t][i] for `rows` rows of
// k values of type WTYPE and `tokens` vectors, x as [tokens][k] and y as [tokens][rows], as matmul.glsl computes it.
// Every weight and input is split into two f16s (split2; the decision of 2026-10-10, since plain f16 inputs failed
// the prefill-against-decode bounds). k is taken 32 values at a time, and each slice's sum starts
// from zero: each of its two 16-value halves adds lo_x·hi_w, then hi_x·lo_w, then hi_x·hi_w through 16 × 16 × 16
// f16 × f16 → f32 cooperative matrices, and the slice's sum is then added to the output's running total with an
// ordinary f32 add (`precise`). The hardware's accumulate inside coopMatMulAdd measured biased, its error growing
// linearly with k (1.7e-5 at k = 4,096 when every product went into one accumulator); summed per slice it grows as
// √k. The split bounds each input and product, not the outputs: each product is within about 2⁻²² of the f32 product
// (only lo·lo is dropped) instead of f16's 2⁻¹² (kernels::Kernels::matmul_cm's doc gives the outputs' measured error).
//
// A workgroup computes 64 rows × 64 tokens as sixteen 16 × 16 tiles; tile i (token block i / 4, row block i % 4) belongs
// to subgroup i % gl_NumSubgroups, whatever the subgroup size, and a subgroup loads a row block's weights once for all
// its tiles in that block. Invocation id dequantizes eight consecutive weights of row id / 4 (dq_eight) and loads eight
// inputs of token id / 4, both split into shared memory. Each output adds its slices in order, the same for every token
// whatever its row in the tile, so a token's outputs don't depend on the batch (the batch tests and the 2,048-vs-500
// check pin it). kernels::twin::matvec::matmul_cm sums each slice the same way, adding its products in k order; the
// hardware adds them in its own, so the two agree within the kernel's tolerance.
//
// The including file enables the cooperative-matrix, memory-scope, subgroup, float16 and control-flow extensions,
// includes common.glsl, quant.glsl and quant8.glsl, and defines WTYPE.
layout(local_size_x = 256) in;
layout(push_constant) uniform Push {
    uint64_t w;
    uint64_t x;
    uint64_t y;
    uint rows;
    uint k;
    uint tokens;
    uint pad;
} p;

// dispatch.rs's MATMUL_CM_SHARED_BYTES (20,480) is this array's size, and Kernels::new checks it against the device:
// change them together. During the k loop it holds the tiles as packed f16 pairs, each [64][40] (32 values and 8 of
// padding): W hi, W lo, X hi, X lo. After the loop, its first [64][68] floats are the output stage.
#define SK 40u
#define T1 (64u * SK / 2u)
#define WH 0u
#define WL T1
#define XH (2u * T1)
#define XL (3u * T1)
#define ST 68u
shared uint sm[4u * T1];

void main() {
    uint id = gl_LocalInvocationIndex;
    uint r0 = 64u * gl_WorkGroupID.x, t0 = 64u * gl_WorkGroupID.y;
    uint s = gl_SubgroupID, ns = gl_NumSubgroups;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[16];
    [[unroll]] for (uint i = 0u; i < 16u; i++) {
        acc[i] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    }
    // Invocation id loads row (and token) id / 4, values 8·(id % 4) .. + 8 of each 32-value slice.
    uint lr = id / 4u, lq = 8u * (id % 4u);
    bool rin = r0 + lr < p.rows, tin = t0 + lr < p.tokens;
    uint64_t wrow = p.w + uint64_t(r0 + lr) * uint64_t(row_bytes(WTYPE, p.k));
    uint64_t xrow = p.x + 4ul * uint64_t(t0 + lr) * uint64_t(p.k);
    uint at = (SK * lr + lq) / 2u;
    for (uint k0 = 0u; k0 < p.k; k0 += 32u) {
        float wv[8], xv[8];
        if (rin) {
            dq_eight(WTYPE, wrow, k0 + lq, wv);
        } else {
            [[unroll]] for (uint e = 0u; e < 8u; e++) {
                wv[e] = 0.0;
            }
        }
        if (tin) {
            [[unroll]] for (uint e = 0u; e < 8u; e++) {
                xv[e] = F32s(xrow).v[k0 + lq + e];
            }
        } else {
            [[unroll]] for (uint e = 0u; e < 8u; e++) {
                xv[e] = 0.0;
            }
        }
        [[unroll]] for (uint e = 0u; e < 8u; e += 2u) {
            uint lo;
            sm[WH + at + e / 2u] = split2(wv[e], wv[e + 1u], lo);
            sm[WL + at + e / 2u] = lo;
            sm[XH + at + e / 2u] = split2(xv[e], xv[e + 1u], lo);
            sm[XL + at + e / 2u] = lo;
        }
        barrier();
        [[unroll]] for (uint rb = 0u; rb < 4u; rb++) {
            bool mine = false;
            [[unroll]] for (uint tb = 0u; tb < 4u; tb++) {
                mine = mine || ((4u * tb + rb) % ns == s);
            }
            if (mine) {
                // The row block's weights, both 16-value halves of the slice (h0: values 0..16, h1: 16..32).
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> bh0, bl0, bh1, bl1;
                uint wb = (SK * 16u * rb) / 2u;
                coopMatLoad(bh0, sm, WH + wb, SK / 2u, gl_CooperativeMatrixLayoutColumnMajor);
                coopMatLoad(bl0, sm, WL + wb, SK / 2u, gl_CooperativeMatrixLayoutColumnMajor);
                coopMatLoad(bh1, sm, WH + wb + 8u, SK / 2u, gl_CooperativeMatrixLayoutColumnMajor);
                coopMatLoad(bl1, sm, WL + wb + 8u, SK / 2u, gl_CooperativeMatrixLayoutColumnMajor);
                [[unroll]] for (uint tb = 0u; tb < 4u; tb++) {
                    uint i = 4u * tb + rb;
                    if (i % ns == s) {
                        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> ah0, al0, ah1, al1;
                        uint xb = (SK * 16u * tb) / 2u;
                        coopMatLoad(ah0, sm, XH + xb, SK / 2u, gl_CooperativeMatrixLayoutRowMajor);
                        coopMatLoad(al0, sm, XL + xb, SK / 2u, gl_CooperativeMatrixLayoutRowMajor);
                        coopMatLoad(ah1, sm, XH + xb + 8u, SK / 2u, gl_CooperativeMatrixLayoutRowMajor);
                        coopMatLoad(al1, sm, XL + xb + 8u, SK / 2u, gl_CooperativeMatrixLayoutRowMajor);
                        // The slice's sum from zero: each half adds lo_x·hi_w, hi_x·lo_w, then hi_x·hi_w.
                        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> part =
                            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
                        part = coopMatMulAdd(al0, bh0, part);
                        part = coopMatMulAdd(ah0, bl0, part);
                        part = coopMatMulAdd(ah0, bh0, part);
                        part = coopMatMulAdd(al1, bh1, part);
                        part = coopMatMulAdd(ah1, bl1, part);
                        part = coopMatMulAdd(ah1, bh1, part);
                        // Then one ordinary f32 add per output, kept apart from the products (`precise`).
                        precise coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> sum = acc[i] + part;
                        acc[i] = sum;
                    }
                }
            }
        }
        barrier();
    }
    // Through the stage, so only tokens and rows in range are written.
    [[unroll]] for (uint i = 0u; i < 16u; i++) {
        if (i % ns == s) {
            coopMatStore(acc[i], sm, ST * 16u * (i / 4u) + 16u * (i % 4u), ST, gl_CooperativeMatrixLayoutRowMajor);
        }
    }
    barrier();
    for (uint e = id; e < 64u * 64u; e += 256u) {
        uint t = e / 64u, r = e % 64u;
        if (t0 + t < p.tokens && r0 + r < p.rows) {
            F32s(p.y).v[(t0 + t) * p.rows + r0 + r] = uintBitsToFloat(sm[ST * t + r]);
        }
    }
}
