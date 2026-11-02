//! Per-tap differences between two tap sets.
use crate::tapdir::TapSet;
use anyhow::{bail, Result};
use qwen36::{
    dims::Dims,
    taps::{expected_taps, TapName},
};

#[derive(Clone, Debug)]
pub struct TapDiff {
    pub tap: TapName,
    /// The worst position's relative L2 error. NaN if either side has a NaN there.
    pub max_rel: f64,
    pub max_abs: f64,
    pub worst_pos: usize,
}

#[derive(Debug, Default)]
pub struct Comparison {
    /// In graph order.
    pub diffs: Vec<TapDiff>,
    pub missing: Vec<String>,
}

/// ‖a − b‖ / ‖b‖ over one position, or ‖a − b‖ when b is all zeros.
pub use qwen36::ops::rel_l2;

/// Compares `a` against `b` (the baseline) for every tap the model has.
pub fn compare(a: &TapSet, b: &TapSet, d: &Dims) -> Result<Comparison> {
    compare_over(a, b, d, &expected_taps(d))
}

/// `compare` over the given taps: the MTP block's, for test 2's MTP half.
pub fn compare_over(a: &TapSet, b: &TapSet, d: &Dims, taps: &[TapName]) -> Result<Comparison> {
    if a.n_tokens != b.n_tokens {
        bail!(
            "the tap sets cover {} and {} positions",
            a.n_tokens,
            b.n_tokens
        );
    }
    let mut out = Comparison::default();
    for &tap in taps {
        let (Some(x), Some(y)) = (a.taps.get(&tap), b.taps.get(&tap)) else {
            let side = if a.taps.contains_key(&tap) {
                "second"
            } else {
                "first"
            };
            out.missing
                .push(format!("{tap} is missing from the {side} set"));
            continue;
        };
        let n = tap.per_pos(d);
        let mut diff = TapDiff {
            tap,
            max_rel: 0.0,
            max_abs: 0.0,
            worst_pos: 0,
        };
        for p in 0..a.n_tokens {
            let (xs, ys) = (&x[p * n..(p + 1) * n], &y[p * n..(p + 1) * n]);
            let r = rel_l2(xs, ys);
            if r.is_nan() || r > diff.max_rel {
                diff.max_rel = r;
                diff.worst_pos = p;
            }
            for (u, v) in xs.iter().zip(ys) {
                diff.max_abs = diff.max_abs.max((*u as f64 - *v as f64).abs());
            }
            if r.is_nan() {
                break;
            }
        }
        out.diffs.push(diff);
    }
    Ok(out)
}
