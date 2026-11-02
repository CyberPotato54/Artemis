//! Test 1 on the GPU for decode attention over up to 16 queries: the token kernels
//! against their twin, against decode's own kernels query by query, and their refusals. They run on the same head
//! shapes as decode's tests (`common::attention_shapes`), so heads of 96 and 224 values reach the token kernels too.
mod common;
use common::attention_shapes;
use kernels::{
    dispatch::{attn_decode_kernel, attn_splits},
    testing::{check, identity, rand_f32, score_range, Gpu},
    tokens::{attn_tokens_part_bytes, MAX_TOKENS},
    twin::{self, attention as at},
    Kernel,
};
use qwen36::testing::Rng;

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `n_ctx0 + MAX_TOKENS - 1` positions of random f16 K and V, `width` values a position, and K as f32.
fn caches(rng: &mut Rng, n_ctx0: usize, width: usize) -> (Vec<u16>, Vec<u16>, Vec<f32>) {
    let n = (n_ctx0 + MAX_TOKENS - 1) * width;
    let kc: Vec<u16> = (0..n)
        .map(|_| twin::f16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let vc: Vec<u16> = (0..n)
        .map(|_| twin::f16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let kf = kc.iter().map(|&b| twin::f16_value(b)).collect();
    (kc, vc, kf)
}

/// The token kernel that serves a query decode would run on `kernel`.
fn token_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::AttnDecode => Kernel::AttnDecodeTokens,
        Kernel::AttnDecodeGqa => Kernel::AttnDecodeGqaTokens,
        other => panic!("{} isn't a decode attention kernel", other.name()),
    }
}

#[test]
#[ignore = "needs the GPU"]
fn attention_tokens_against_its_twin() {
    let g = Gpu::new();
    let mut rng = Rng::new(62);
    for (nh, nkv, hd) in attention_shapes() {
        // 255 + 2 and 255 + 5 cross a chunk boundary (n_ctx 257 needs a second chunk) and 3,070 + 5 the kernel switch
        // (n_ctx 3,073 is past 3,072).
        for n_ctx0 in [1, 255, 256, 1000, 3070, 3072, 5000, 9000] {
            let (kc, vc, kf) = caches(&mut rng, n_ctx0, nkv * hd);
            let (gkc, gvc) = (g.bytes(&u16_bytes(&kc)), g.bytes(&u16_bytes(&vc)));
            // Queries scaled so scores spread over several units, as trained heads' do (score_range checks it).
            let q = rand_f32(&mut rng, MAX_TOKENS * nh * hd, -12.0, 12.0);
            let want = at::attention_tokens(&q, &kc, &vc, n_ctx0, MAX_TOKENS, nh, nkv, hd);
            for tokens in [1usize, 2, 5, 16] {
                let last = n_ctx0 + tokens - 1;
                if last >= 256 {
                    for i in [0, tokens - 1] {
                        let spread = score_range(
                            &q[i * nh * hd..(i + 1) * nh * hd],
                            &kf,
                            n_ctx0 + i,
                            nh,
                            nkv,
                            hd,
                        );
                        assert!(
                            spread >= 8.0,
                            "hd={hd} n_ctx0={n_ctx0} query {i}: scores span only {spread:.2}"
                        );
                    }
                }
                let max_split = attn_splits(last);
                let gq = g.f32s(&q[..tokens * nh * hd]);
                let part =
                    g.zeros((attn_tokens_part_bytes(nh, hd, max_split, tokens) / 4) as usize);
                let y = g.zeros(tokens * nh * hd);
                g.run(|k, r| {
                    k.attn_decode_tokens(
                        r, gq.addr, gkc.addr, gvc.addr, part.addr, n_ctx0, tokens, nh, nkv, hd,
                        max_split,
                    );
                    r.barrier();
                    k.attn_combine_tokens(r, part.addr, y.addr, n_ctx0, tokens, nh, hd, max_split);
                });
                let got = g.read(y);
                // One check for the queries each decode kernel serves, so the evidence names the kernel that ran.
                let row = nh * hd;
                let mut i = 0;
                while i < tokens {
                    let kernel = attn_decode_kernel(n_ctx0 + i, nh, nkv, hd);
                    let end = (i..tokens)
                        .find(|&j| attn_decode_kernel(n_ctx0 + j, nh, nkv, hd) != kernel)
                        .unwrap_or(tokens);
                    check(
                        token_kernel(kernel),
                        &format!("hd={hd} n_ctx0={n_ctx0} tokens={tokens} queries {i}..{end}"),
                        &got[i * row..end * row],
                        &want[i * row..end * row],
                    );
                    i = end;
                }
            }
        }
    }
}

/// Each query of the token kernels writes the partials and the output one decode step writes at its own length: the
/// kernel `attn_decode_kernel` chooses, its split count, then `attn_combine` for one query. Reported through `identity`
/// (a miss is still usable, and the evidence names it); the partials past a query's last chunk must stay untouched.
#[test]
#[ignore = "needs the GPU"]
fn attention_tokens_writes_decodes_bits_per_query() {
    let g = Gpu::new();
    let mut rng = Rng::new(63);
    for (nh, nkv, hd) in attention_shapes() {
        let stride = hd + 2;
        for n_ctx0 in [1, 255, 256, 1000, 3070, 3072, 5000, 9000] {
            let (kc, vc, _) = caches(&mut rng, n_ctx0, nkv * hd);
            let (gkc, gvc) = (g.bytes(&u16_bytes(&kc)), g.bytes(&u16_bytes(&vc)));
            for tokens in [2usize, 5, 16] {
                let q = rand_f32(&mut rng, tokens * nh * hd, -12.0, 12.0);
                let gq = g.f32s(&q);
                // A max_split past the need, so the partials' stride isn't the split count.
                let max_split = attn_splits(n_ctx0 + tokens - 1) + 2;
                let part =
                    g.zeros((attn_tokens_part_bytes(nh, hd, max_split, tokens) / 4) as usize);
                let y = g.zeros(tokens * nh * hd);
                // Decode's own dispatches, one set of buffers per query.
                let singles: Vec<_> = (0..tokens)
                    .map(|i| {
                        let n_ctx = n_ctx0 + i;
                        (
                            g.f32s(&q[i * nh * hd..(i + 1) * nh * hd]),
                            g.zeros(nh * attn_splits(n_ctx) * stride),
                            g.zeros(nh * hd),
                        )
                    })
                    .collect();
                g.run(|k, r| {
                    k.attn_decode_tokens(
                        r, gq.addr, gkc.addr, gvc.addr, part.addr, n_ctx0, tokens, nh, nkv, hd,
                        max_split,
                    );
                    r.barrier();
                    k.attn_combine_tokens(r, part.addr, y.addr, n_ctx0, tokens, nh, hd, max_split);
                    for (i, (sq, sp, _)) in singles.iter().enumerate() {
                        k.attn_decode(
                            r,
                            sq.addr,
                            gkc.addr,
                            gvc.addr,
                            sp.addr,
                            n_ctx0 + i,
                            nh,
                            nkv,
                            hd,
                        );
                    }
                    r.barrier();
                    for (i, (_, sp, sy)) in singles.iter().enumerate() {
                        k.attn_combine(r, sp.addr, sy.addr, n_ctx0 + i, nh, hd);
                    }
                });
                let (tp, ty) = (g.read(part), g.read(y));
                // Per decode kernel: whether every query it serves matched.
                let (mut plain_ok, mut gqa_ok, mut out_ok) = (true, true, true);
                let (mut any_plain, mut any_gqa) = (false, false);
                for (i, (_, sp, sy)) in singles.iter().enumerate() {
                    let n_ctx = n_ctx0 + i;
                    let n_split = attn_splits(n_ctx);
                    let (dp, dy) = (g.read(*sp), g.read(*sy));
                    let mut same = true;
                    for h in 0..nh {
                        for s in 0..max_split {
                            let at = ((i * nh + h) * max_split + s) * stride;
                            let slot = &tp[at..at + stride];
                            if s < n_split {
                                let want = &dp[(h * n_split + s) * stride..][..stride];
                                same &= slot
                                    .iter()
                                    .zip(want)
                                    .all(|(a, b)| a.to_bits() == b.to_bits());
                                assert!(
                                    slot.iter().all(|v| v.is_finite()) && slot[1] > 0.0,
                                    "hd={hd} n_ctx0={n_ctx0} tokens={tokens} query {i} head {h} chunk {s}: a partial isn't finite"
                                );
                            } else {
                                assert!(
                                    slot.iter().all(|v| v.to_bits() == 0),
                                    "hd={hd} n_ctx0={n_ctx0} tokens={tokens} query {i} head {h} chunk {s}: written past the query's last chunk"
                                );
                            }
                        }
                    }
                    let row = &ty[i * nh * hd..(i + 1) * nh * hd];
                    assert!(
                        row.iter().all(|v| v.is_finite()),
                        "hd={hd} n_ctx0={n_ctx0} tokens={tokens} query {i}: the output isn't finite"
                    );
                    out_ok &= row.iter().zip(&dy).all(|(a, b)| a.to_bits() == b.to_bits());
                    match attn_decode_kernel(n_ctx, nh, nkv, hd) {
                        Kernel::AttnDecode => {
                            any_plain = true;
                            plain_ok &= same;
                        }
                        _ => {
                            any_gqa = true;
                            gqa_ok &= same;
                        }
                    }
                }
                let case = format!(
                    "hd={hd} n_ctx0={n_ctx0} tokens={tokens} against decode's per-query dispatch"
                );
                if any_plain {
                    identity(Kernel::AttnDecodeTokens.name(), &case, plain_ok);
                }
                if any_gqa {
                    identity(Kernel::AttnDecodeGqaTokens.name(), &case, gqa_ok);
                }
                identity(Kernel::AttnCombineTokens.name(), &case, out_ok);
            }
        }
    }
}

/// Buffers for `attention_tokens_refuses_*`: sized for one query, so a dispatch that got past the assert would run in
/// bounds.
fn refuse(tokens: usize) {
    let g = Gpu::new();
    let (nh, nkv, hd, n_ctx0) = (4, 1, 64, 4);
    let (q, kc, vc) = (
        g.zeros(MAX_TOKENS * nh * hd),
        g.zeros((n_ctx0 + MAX_TOKENS) * nkv * hd / 2),
        g.zeros((n_ctx0 + MAX_TOKENS) * nkv * hd / 2),
    );
    let part = g.zeros((attn_tokens_part_bytes(nh, hd, 1, MAX_TOKENS) / 4) as usize);
    g.run(|k, r| {
        k.attn_decode_tokens(
            r, q.addr, kc.addr, vc.addr, part.addr, n_ctx0, tokens, nh, nkv, hd, 1,
        );
    });
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "attn_decode_tokens: 1 to 16 queries, not 0")]
fn attention_tokens_refuses_0_tokens() {
    refuse(0);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "attn_decode_tokens: 1 to 16 queries, not 17")]
fn attention_tokens_refuses_17_tokens() {
    refuse(17);
}
