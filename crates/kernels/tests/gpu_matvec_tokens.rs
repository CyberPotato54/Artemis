//! Test 1 on the GPU for `matvec_tokens`: every type against its twin within the kernel's tolerance,
//! and the identity report against decode's `matvec` once a token. `check` and `identity` record each
//! measurement when ARTEMIS_EVIDENCE is set.
use kernels::{
    dispatch::MatW,
    testing::{check, identity, rand_f32, random_weights, Gpu},
    twin::matvec as mv,
    Kernel,
};
use qwen36::{quant::WType, testing::Rng};

/// Decode's shapes: DeltaNet's projections, the expert shapes and Q6_K's head; and 37 rows of 2,304 values, whose
/// last workgroup holds 5 rows and whose 72 units leave lanes 8 to 31 of each row one unit short (a row's last
/// partial block of 32 units).
fn cases(ty: WType) -> Vec<(usize, usize)> {
    let mut v = vec![
        (8, 2048),
        (256, 2048),
        (4096, 2048),
        (8192, 2048),
        (2048, 4096),
        (37, 2304),
    ];
    if ty == WType::Q6K {
        v.push((16384, 2048));
    }
    v
}

/// Every count with a specialized pipeline (`tokens::SPECIALIZED_TOKENS`, 2 to 5) and the general pipeline at 1, 7 and
/// 16.
const TOKENS: [usize; 7] = [1, 2, 3, 4, 5, 7, 16];

#[test]
#[ignore = "needs the GPU"]
fn matvec_tokens_every_type_against_its_twin() {
    let g = Gpu::new();
    let mut rng = Rng::new(43);
    for ty in WType::ALL {
        for (rows, cols) in cases(ty) {
            let w = random_weights(&mut rng, ty, rows, cols);
            let gw = g.bytes(&w);
            let m = MatW {
                addr: gw.addr,
                ty,
                rows,
                cols,
            };
            for tokens in TOKENS {
                let x = rand_f32(&mut rng, tokens * cols, -1.0, 1.0);
                let want = mv::matvec_tokens(ty, &w, rows, &x, tokens);
                // Packed: the strides are the shape's own.
                let (gx, y) = (g.f32s(&x), g.zeros(tokens * rows));
                g.run(|k, r| k.matvec_tokens(r, m, gx.addr, y.addr, tokens, cols, rows));
                check(
                    Kernel::matvec_tokens(ty),
                    &format!("{} {rows}x{cols} t={tokens}", ty.name()),
                    &g.read(y),
                    &want,
                );
                // Padded: each token's input and output sit in a wider row, and the padding is not touched.
                let (xs, ys) = (cols + 4, rows + 3);
                let mut padded = vec![7.0f32; tokens * xs];
                for (i, row) in x.chunks_exact(cols).enumerate() {
                    padded[i * xs..i * xs + cols].copy_from_slice(row);
                }
                let (gx, y) = (g.f32s(&padded), g.f32s(&vec![-3.0f32; tokens * ys]));
                g.run(|k, r| k.matvec_tokens(r, m, gx.addr, y.addr, tokens, xs, ys));
                let got = g.read(y);
                let mut packed = Vec::with_capacity(tokens * rows);
                for i in 0..tokens {
                    packed.extend_from_slice(&got[i * ys..i * ys + rows]);
                    assert!(
                        got[i * ys + rows..(i + 1) * ys].iter().all(|&v| v == -3.0),
                        "{} {rows}x{cols} t={tokens}: token {i}'s output padding was written",
                        ty.name()
                    );
                }
                check(
                    Kernel::matvec_tokens(ty),
                    &format!("{} {rows}x{cols} t={tokens} padded", ty.name()),
                    &packed,
                    &want,
                );
            }
        }
    }
}

/// Whether `matvec_tokens` writes decode's matvec's bits for every token, reported per case and never asserted.
/// It fails only if it compared nothing.
#[test]
#[ignore = "needs the GPU"]
fn matvec_tokens_writes_matvecs_bits_per_token() {
    let g = Gpu::new();
    let mut rng = Rng::new(44);
    let mut reported = 0;
    // Counted from the case tables, not by the loops below, so a loop that skips a case leaves `reported` short.
    let expected: usize =
        WType::ALL.iter().map(|&ty| cases(ty).len()).sum::<usize>() * TOKENS.len();
    for ty in WType::ALL {
        for (rows, cols) in cases(ty) {
            let w = random_weights(&mut rng, ty, rows, cols);
            let gw = g.bytes(&w);
            let m = MatW {
                addr: gw.addr,
                ty,
                rows,
                cols,
            };
            for tokens in TOKENS {
                let x = rand_f32(&mut rng, tokens * cols, -1.0, 1.0);
                let (gx, y) = (g.f32s(&x), g.zeros(tokens * rows));
                g.run(|k, r| k.matvec_tokens(r, m, gx.addr, y.addr, tokens, cols, rows));
                let got = g.read(y);
                // Decode's matvec, once a token, into its own output.
                let mut want = Vec::with_capacity(tokens * rows);
                for xs in x.chunks_exact(cols) {
                    let (gx, y) = (g.f32s(xs), g.zeros(rows));
                    g.run(|k, r| k.matvec(r, m, gx.addr, y.addr));
                    want.extend(g.read(y));
                }
                assert_eq!(
                    got.len(),
                    tokens * rows,
                    "{} {rows}x{cols} t={tokens}",
                    ty.name()
                );
                assert_eq!(
                    want.len(),
                    got.len(),
                    "{} {rows}x{cols} t={tokens}",
                    ty.name()
                );
                let equal = got.len() == want.len()
                    && got
                        .iter()
                        .zip(&want)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                identity(
                    Kernel::matvec_tokens(ty).name(),
                    &format!("{} {rows}x{cols} t={tokens} against matvec", ty.name()),
                    equal,
                );
                reported += 1;
            }
        }
    }
    assert!(expected > 0, "no case is listed");
    assert_eq!(
        reported, expected,
        "every listed case reports one identity line"
    );
}

/// Runs `matvec_tokens` over a small F32 matrix (256 x 256) with buffers sized for 17 tokens of padded rows, so a
/// dispatch that slipped past its asserts would stay in bounds. The asserts run before anything is recorded.
fn matvec_tokens_with(tokens: usize, x_stride: usize, y_stride: usize) {
    let g = Gpu::new();
    let mut rng = Rng::new(45);
    let (rows, cols) = (256, 256);
    let w = random_weights(&mut rng, WType::F32, rows, cols);
    let gw = g.bytes(&w);
    let m = MatW {
        addr: gw.addr,
        ty: WType::F32,
        rows,
        cols,
    };
    let gx = g.zeros(17 * 260);
    let y = g.zeros(17 * 260);
    g.run(|k, r| k.matvec_tokens(r, m, gx.addr, y.addr, tokens, x_stride, y_stride));
}

// The refusals of matvec_tokens (the precedent is gpu_kernels.rs's `the_gqa_attention_refuses_a_shape_it_doesnt_serve`):
// they are asserts inside `Kernels::matvec_tokens`, which needs a device to exist, so they run with the GPU tests.

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "a matvec over 0 tokens: 1 to 16")]
fn matvec_tokens_refuses_no_tokens() {
    matvec_tokens_with(0, 256, 256);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "a matvec over 17 tokens: 1 to 16")]
fn matvec_tokens_refuses_more_than_max_tokens() {
    matvec_tokens_with(17, 256, 256);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "x_stride 258: a multiple of 4 values")]
fn matvec_tokens_refuses_an_input_stride_off_16_bytes() {
    matvec_tokens_with(2, 258, 256);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "y_stride 255: y 4-byte aligned, the stride at least the outputs")]
fn matvec_tokens_refuses_an_output_stride_below_the_rows() {
    matvec_tokens_with(2, 256, 255);
}
