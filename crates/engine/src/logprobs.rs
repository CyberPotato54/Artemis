//! Log-probabilities of the model's own distribution: the log-softmax of the raw logits, before the
//! presence penalty, temperature and filters. The generation loop calls it for each sampled token when a request asks.
use crate::request::TokenLogprobs;
use std::cmp::Ordering;

/// The most alternatives a request may ask for.
const MAX_TOP: usize = 20;

/// `token`'s log-probability and the `top` highest-logit tokens' (clamped to 20 and to `logits.len()`; 0 gives an empty
/// list), sorted by logit descending, ties to the lower id. `token` must be below `logits.len()`, and the logits finite
/// (`run_step` guarantees it). The sum runs in f64 over `exp(x - max)`, so huge logits don't overflow.
pub fn logprobs(logits: &[f32], token: u32, top: u8) -> TokenLogprobs {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&x| (x as f64 - m).exp()).sum();
    let lse = m + sum.ln();
    let n = (top as usize).min(MAX_TOP).min(logits.len());
    let logprob = (logits[token as usize] as f64 - lse) as f32;
    if n == 0 {
        return TokenLogprobs {
            logprob,
            top: Vec::new(),
        };
    }
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    let by_logit = |a: &u32, b: &u32| order(logits, *a, *b);
    if n < ids.len() {
        ids.select_nth_unstable_by(n - 1, by_logit);
    }
    ids.truncate(n);
    ids.sort_unstable_by(by_logit);
    TokenLogprobs {
        logprob,
        top: ids
            .into_iter()
            .map(|i| (i, (logits[i as usize] as f64 - lse) as f32))
            .collect(),
    }
}

/// Logit descending, then id ascending: a total order over finite logits (so the selection is deterministic), in which
/// -0.0 and +0.0 tie. A NaN, which `run_step` rules out, breaks the order (it maps to `Equal`), so a sort over it may
/// panic.
fn order(logits: &[f32], a: u32, b: u32) -> Ordering {
    logits[b as usize]
        .partial_cmp(&logits[a as usize])
        .unwrap_or(Ordering::Equal)
        .then(a.cmp(&b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logprobs_are_a_log_softmax_of_the_raw_logits() {
        let l = [1.0f32, 3.0, 2.0, 3.0];
        let lp = logprobs(&l, 2, 3);
        let lse = (1f64.exp() + 2.0 * 3f64.exp() + 2f64.exp()).ln();
        assert!((lp.logprob as f64 - (2.0 - lse)).abs() < 1e-6);
        assert_eq!(
            lp.top.iter().map(|t| t.0).collect::<Vec<_>>(),
            vec![1, 3, 2]
        ); // a tie goes to the lower id
        assert!((lp.top[0].1 as f64 - (3.0 - lse)).abs() < 1e-6);
    }

    #[test]
    fn logprobs_take_at_most_twenty_and_none_when_asked_for_none() {
        let l: Vec<f32> = (0..100).map(|i| (i % 37) as f32).collect();
        let top = logprobs(&l, 5, 25).top;
        assert_eq!(top.len(), 20);
        // Many ties: the 20 are the first 20 of a full sort under the same order.
        let mut all: Vec<u32> = (0..100).collect();
        all.sort_by(|&a, &b| order(&l, a, b));
        assert_eq!(top.iter().map(|t| t.0).collect::<Vec<_>>(), all[..20]);
        assert!(logprobs(&l, 5, 0).top.is_empty());
        assert_eq!(logprobs(&[0.5, 0.25], 0, 20).top.len(), 2);
    }

    #[test]
    fn tied_logits_across_the_cut_off_keep_the_lower_ids() {
        // Five ids tie at 5.0; the top 4 take both 9.0s and then the two lowest of the tied fives.
        let l = [2.0f32, 5.0, 9.0, 5.0, 5.0, 9.0, 5.0, 1.0, 5.0];
        let ids = |top| {
            logprobs(&l, 0, top)
                .top
                .iter()
                .map(|t| t.0)
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(4), vec![2, 5, 1, 3]);
        assert_eq!(ids(3), vec![2, 5, 1]);
    }

    #[test]
    fn negative_and_positive_zero_tie_to_the_lower_id() {
        let lp = logprobs(&[-1.0, -0.0, 0.0, -0.0, 0.0], 0, 4);
        assert_eq!(
            lp.top.iter().map(|t| t.0).collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn huge_logits_dont_overflow() {
        let lp = logprobs(&[1e30, 1e30 - 1.0, -1e30], 0, 2);
        assert!(lp.logprob.is_finite() && lp.top.iter().all(|t| t.1.is_finite()));
    }
}
