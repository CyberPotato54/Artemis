//! Choosing the next token: greedy, or the sampler. The sampler applies the presence penalty, then top-k,
//! top-p, min-p and temperature, then draws from the engine's own seeded RNG. Its defaults are production's.
//! It runs on the CPU over all the logits the step read back.
use crate::rng::Rng;
use std::{cmp::Ordering, collections::BTreeSet};

/// The highest logit's id, ties going to the lower id. The logits must be finite, as `run_step` guarantees.
/// A NaN at index 0 wins; a NaN anywhere else never does.
pub fn greedy(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// How the sampler chooses. Each filter has a value that turns it off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    /// Temperature, applied after the filters. 0 is greedy.
    pub temp: f32,
    /// Keep the k highest logits, ties to the lower id. 0 keeps every token.
    pub top_k: usize,
    /// Keep the shortest run of most likely tokens whose probabilities reach top_p. 1 keeps every token.
    pub top_p: f32,
    /// Drop tokens less likely than min_p times the most likely one. 0 keeps every token.
    pub min_p: f32,
    /// Subtracted once from the logit of every token this sampler has returned before. 0 is off.
    pub presence_penalty: f32,
}

impl Sampling {
    /// Production's defaults: temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 0.
    pub const DEFAULT: Sampling = Sampling {
        temp: 0.6,
        top_k: 20,
        top_p: 0.95,
        min_p: 0.0,
        presence_penalty: 0.0,
    };

    /// Argmax, ties to the lower id: temperature 0, every filter off.
    pub const GREEDY: Sampling = Sampling {
        temp: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
    };

    /// Refuses settings the sampler can't honour, with a one-line message naming the flag.
    pub fn check(&self) -> Result<(), String> {
        if !(self.temp.is_finite() && self.temp >= 0.0) {
            return Err(format!(
                "--temp {} must be a finite number, 0 or more",
                self.temp
            ));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(format!(
                "--top-p {} must be above 0 and at most 1",
                self.top_p
            ));
        }
        if !(self.min_p >= 0.0 && self.min_p <= 1.0) {
            return Err(format!("--min-p {} must be from 0 to 1", self.min_p));
        }
        if !self.presence_penalty.is_finite() {
            return Err(format!(
                "--presence-penalty {} must be a finite number",
                self.presence_penalty
            ));
        }
        Ok(())
    }
}

/// Draws tokens from logits under `Sampling`, with a seeded RNG: the same seed and logits give the same tokens.
pub struct Sampler {
    s: Sampling,
    rng: Rng,
    /// Every token this sampler has returned, for the presence penalty.
    seen: BTreeSet<u32>,
}

/// Logit descending, then id ascending: the order every filter keeps.
fn rank(a: &(u32, f32), b: &(u32, f32)) -> Ordering {
    b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
}

impl Sampler {
    /// `s` must pass `Sampling::check`.
    pub fn new(s: Sampling, seed: u64) -> Sampler {
        Sampler {
            s,
            rng: Rng::new(seed),
            seen: BTreeSet::new(),
        }
    }

    /// The next token: the presence penalty, then greedy if the temperature is 0, else top-k, top-p, min-p and the
    /// temperature, and a draw. Probabilities are computed in f64. The logits must be finite, as `run_step` guarantees.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        assert!(!logits.is_empty(), "sampling from no logits");
        debug_assert!(
            logits.iter().all(|x| x.is_finite()),
            "the sampler needs finite logits; run_step refuses others with Error::NonFinite"
        );
        let pen = self.s.presence_penalty;
        let t = if self.s.temp <= 0.0 {
            if pen != 0.0 && !self.seen.is_empty() {
                greedy_penalized(logits, &self.seen, pen)
            } else {
                greedy(logits)
            }
        } else {
            let k = self.s.top_k;
            if k > 0 && k < logits.len() {
                // Up to `TOP_K_ONE_PASS_MAX`, top-k in one pass over the logits, the penalty taken
                // as each is read: the candidates selecting from every (id, logit) pair keeps, in its order, without
                // copying the vocabulary. Above it, selecting from every candidate, which gives the same tokens.
                let cand = if takes_one_pass(k, logits.len()) {
                    top_k(logits, k, &self.seen, pen)
                } else {
                    penalized_candidates(logits, &self.seen, pen)
                };
                self.draw_full_sort(cand)
            } else {
                self.draw_partial(penalized_candidates(logits, &self.seen, pen))
            }
        };
        self.seen.insert(t);
        t
    }

    /// The reference draw: top-k by selection, then a full sort of what is kept, then top-p, min-p and the draw.
    /// With a top-k this is cheap; without one it sorts every candidate.
    fn draw_full_sort(&mut self, mut cand: Vec<(u32, f32)>) -> u32 {
        let k = self.s.top_k;
        if k > 0 && k < cand.len() {
            cand.select_nth_unstable_by(k - 1, rank);
            cand.truncate(k);
        }
        cand.sort_unstable_by(rank);
        let top = cand[0].1 as f64;
        if self.s.top_p < 1.0 {
            // Probabilities at temperature 1 over what top-k kept.
            let w: Vec<f64> = cand.iter().map(|c| (c.1 as f64 - top).exp()).collect();
            let total: f64 = w.iter().sum();
            let (mut cum, mut keep) = (0f64, cand.len());
            for (i, wi) in w.iter().enumerate() {
                cum += wi / total;
                if cum >= self.s.top_p as f64 {
                    keep = i + 1;
                    break;
                }
            }
            cand.truncate(keep);
        }
        if self.s.min_p > 0.0 {
            // p_i ≥ min_p · p_top, as logits: the top token always stays.
            let floor = top + (self.s.min_p as f64).ln();
            cand.retain(|c| c.1 as f64 >= floor);
        }
        let temp = self.s.temp as f64;
        let w: Vec<f64> = cand
            .iter()
            .map(|c| ((c.1 as f64 - top) / temp).exp())
            .collect();
        let u = self.rng.unit() * w.iter().sum::<f64>();
        let mut cum = 0f64;
        for (c, wi) in cand.iter().zip(&w) {
            cum += wi;
            if u < cum {
                return c.0;
            }
        }
        cand.last().expect("the top token always stays").0
    }

    /// The same draw as `draw_full_sort` for every candidate (no top-k), without sorting them all. Only a sorted prefix
    /// of the candidates is needed: the kept tokens are a prefix in rank order, and the draw walks it from the top. The
    /// prefix is found by selection, starting at 64 and growing fourfold until top-p's cumulative is reached, min-p's floor is
    /// passed, or (with neither filter) the draw's `u` falls inside it.
    ///
    /// The one difference from the full sort is the order of one f64 sum: top-p's total over every candidate, and,
    /// with no filter at all, the draw's total, are added in index order, not rank order. They differ by a few ulps,
    /// which could change a token only if a cumulative lands within that of top-p, or `u` of a boundary.
    fn draw_partial(&mut self, mut cand: Vec<(u32, f32)>) -> u32 {
        let n = cand.len();
        let (top_p, min_p, temp) = (self.s.top_p as f64, self.s.min_p as f64, self.s.temp as f64);
        let cuts = self.s.top_p < 1.0 || self.s.min_p > 0.0;
        let top = cand
            .iter()
            .map(|c| c.1)
            .max_by(|a, b| a.total_cmp(b))
            .expect("candidates") as f64;
        let floor = (min_p > 0.0).then(|| top + min_p.ln());
        let total1: f64 = if self.s.top_p < 1.0 {
            cand.iter().map(|c| (c.1 as f64 - top).exp()).sum()
        } else {
            0.0
        };
        let mut kk = 64.min(n);
        let sort_prefix = |cand: &mut Vec<(u32, f32)>, kk: usize| {
            if kk < n {
                cand.select_nth_unstable_by(kk - 1, rank);
            }
            cand[..kk].sort_unstable_by(rank);
        };
        if cuts {
            // Grow the sorted prefix until it holds the cut: the first token min-p drops, or the one that takes the
            // cumulative to top-p.
            let keep = loop {
                sort_prefix(&mut cand, kk);
                let mut cum = 0f64;
                let mut found = None;
                for (i, c) in cand[..kk].iter().enumerate() {
                    if let Some(f) = floor {
                        // The logits are finite, so this is the negation of the full sort's `>= floor`.
                        if (c.1 as f64) < f {
                            found = Some(i);
                            break;
                        }
                    }
                    if self.s.top_p < 1.0 {
                        cum += (c.1 as f64 - top).exp() / total1;
                        if cum >= top_p {
                            found = Some(i + 1);
                            break;
                        }
                    }
                }
                match found {
                    Some(keep) => break keep,
                    None if kk == n => break n,
                    None => kk = (4 * kk).min(n),
                }
            };
            cand.truncate(keep);
            let w: Vec<f64> = cand
                .iter()
                .map(|c| ((c.1 as f64 - top) / temp).exp())
                .collect();
            let u = self.rng.unit() * w.iter().sum::<f64>();
            let mut cum = 0f64;
            for (c, wi) in cand.iter().zip(&w) {
                cum += wi;
                if u < cum {
                    return c.0;
                }
            }
            return cand.last().expect("the top token always stays").0;
        }
        // No filter: every token is kept, and the draw decides how much of the order is needed.
        let total: f64 = cand.iter().map(|c| ((c.1 as f64 - top) / temp).exp()).sum();
        let u = self.rng.unit() * total;
        let mut cum = 0f64;
        let mut from = 0;
        loop {
            sort_prefix(&mut cand, kk);
            for c in &cand[from..kk] {
                cum += ((c.1 as f64 - top) / temp).exp();
                if u < cum {
                    return c.0;
                }
            }
            if kk == n {
                return cand[n - 1].0;
            }
            from = kk;
            kk = (4 * kk).min(n);
        }
    }
}

/// The largest top-k taken in one pass. `top_k` inserts into a sorted list of `k`, O(n·k) for n logits: past a few
/// hundred the insertion loses to selecting from every candidate (248,320 logits: 0.65 ms either way at k = 1,000,
/// 19 ms against 0.8 at 10,000, 1.4 s against 4.9 ms at 248,319).
const TOP_K_ONE_PASS_MAX: usize = 256;

/// Whether `Sampler::sample` takes `top_k` for this `k` over `n` logits.
fn takes_one_pass(k: usize, n: usize) -> bool {
    0 < k && k <= TOP_K_ONE_PASS_MAX && k < n
}

/// Every (id, logit) pair with the presence penalty taken off each id in `seen`: the candidates the full-sort path
/// starts from.
fn penalized_candidates(logits: &[f32], seen: &BTreeSet<u32>, pen: f32) -> Vec<(u32, f32)> {
    // The candidates carry the penalty, so the logits are never copied whole.
    let mut cand: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as u32, x))
        .collect();
    if pen != 0.0 {
        for &t in seen {
            if let Some(c) = cand.get_mut(t as usize) {
                c.1 -= pen;
            }
        }
    }
    cand
}

/// The `k` best (id, logit) pairs by `rank`, best first, with `pen` taken off the logit of every id in `seen`: one pass
/// over `logits`, keeping the best so far in order. `rank` is a total order (the logit by
/// `total_cmp`, then the lower id), so these are exactly the `k` pairs `select_nth_unstable_by` keeps, in the order
/// `sort_unstable_by` gives them. `k` is at least 1. The cost is O(n·k): each kept pair shifts up to `k` others, so
/// `Sampler::sample` takes it only up to `TOP_K_ONE_PASS_MAX`.
fn top_k(logits: &[f32], k: usize, seen: &BTreeSet<u32>, pen: f32) -> Vec<(u32, f32)> {
    let mut seen = seen.iter().map(|&t| t as usize).peekable();
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    for (i, &x) in logits.iter().enumerate() {
        let v = if pen != 0.0 && seen.peek() == Some(&i) {
            seen.next();
            x - pen
        } else {
            x
        };
        let c = (i as u32, v);
        if best.len() == k {
            // Most logits fall below the k-th outright; a near one (equal, or a signed zero) takes the full comparison.
            let last = best[k - 1];
            if v < last.1 || rank(&last, &c) == Ordering::Less {
                continue;
            }
        }
        let at = best.partition_point(|b| rank(b, &c) == Ordering::Less);
        best.insert(at, c);
        best.truncate(k);
    }
    best
}

/// `greedy` over the logits with the presence penalty taken off every seen token, without copying them.
fn greedy_penalized(logits: &[f32], seen: &BTreeSet<u32>, pen: f32) -> u32 {
    let mut seen = seen.iter().map(|&t| t as usize).peekable();
    let mut value = |i: usize| {
        if seen.peek() == Some(&i) {
            seen.next();
            logits[i] - pen
        } else {
            logits[i]
        }
    };
    let (mut best, mut best_v) = (0, value(0));
    for i in 1..logits.len() {
        let v = value(i);
        if v > best_v {
            (best, best_v) = (i, v);
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MTP drafter's pick (`argmax_rows`' twin) is `greedy` on rows with no NaN and no infinity; it differs on NaN
    /// and infinity (greedy lets a NaN at index 0 win and picks +inf).
    #[test]
    fn the_argmax_twin_equals_greedy_on_finite_rows() {
        let mut rng = crate::rng::Rng::new(40);
        for r in 0..1000 {
            let n = 1 + rng.below(600);
            let mut row: Vec<f32> = (0..n).map(|_| (rng.unit() * 8.0 - 4.0) as f32).collect();
            if r % 3 == 0 && n > 2 {
                let m = row.iter().cloned().fold(f32::MIN, f32::max);
                row[n - 1] = m;
                row[n / 2] = m;
            }
            assert_eq!(
                kernels::twin::argmax_row(&row),
                Some(greedy(&row)),
                "row {r}"
            );
        }
    }

    /// Logits whose softmax is `p`.
    fn logits_of(p: &[f64]) -> Vec<f32> {
        p.iter().map(|x| x.ln() as f32).collect()
    }

    /// How often each id comes up in `n` draws.
    fn frequencies(s: Sampling, seed: u64, logits: &[f32], n: usize) -> Vec<f64> {
        let mut smp = Sampler::new(s, seed);
        let mut count = vec![0usize; logits.len()];
        for _ in 0..n {
            count[smp.sample(logits) as usize] += 1;
        }
        count.iter().map(|&c| c as f64 / n as f64).collect()
    }

    /// The ids that come up at least once in `n` draws.
    fn drawn(s: Sampling, logits: &[f32], n: usize) -> Vec<u32> {
        let f = frequencies(s, 7, logits, n);
        (0..f.len() as u32)
            .filter(|&i| f[i as usize] > 0.0)
            .collect()
    }

    /// Temperature 1, every filter off.
    const PLAIN: Sampling = Sampling {
        temp: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
    };

    fn near(got: &[f64], want: &[f64]) {
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g - w).abs() < 0.02,
                "frequencies {got:?}, expected {want:?}"
            );
        }
    }

    #[test]
    fn greedy_takes_the_lower_id_on_a_tie() {
        assert_eq!(greedy(&[0.5, 2.0, -1.0, 2.0]), 1);
        assert_eq!(greedy(&[3.0]), 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "finite logits")]
    fn non_finite_logits_are_a_precondition_violation() {
        Sampler::new(Sampling::DEFAULT, 0).sample(&[0.0, f32::NAN, 1.0]);
    }

    #[test]
    fn default_settings_are_productions() {
        assert_eq!(
            Sampling::DEFAULT,
            Sampling {
                temp: 0.6,
                top_k: 20,
                top_p: 0.95,
                min_p: 0.0,
                presence_penalty: 0.0
            }
        );
        assert!(Sampling::DEFAULT.check().is_ok() && Sampling::GREEDY.check().is_ok());
    }

    #[test]
    fn zero_temperature_is_greedy_whatever_the_filters() {
        let s = Sampling {
            temp: 0.0,
            top_k: 2,
            top_p: 0.1,
            min_p: 0.9,
            presence_penalty: 0.0,
        };
        for seed in 0..20 {
            assert_eq!(Sampler::new(s, seed).sample(&[0.1, 3.0, 3.0, -1.0]), 1);
        }
    }

    #[test]
    fn top_k_one_always_picks_the_argmax() {
        let s = Sampling {
            temp: 1.5,
            top_k: 1,
            ..PLAIN
        };
        assert_eq!(drawn(s, &[1.0, 2.0, 1.9, 0.0], 1000), [1]);
    }

    #[test]
    fn top_k_keeps_the_k_highest_with_ties_to_the_lower_id() {
        let s = Sampling { top_k: 2, ..PLAIN };
        assert_eq!(drawn(s, &[1.0, 2.0, 2.0, 2.0, 0.5], 2000), [1, 2]);
    }

    #[test]
    fn top_k_beyond_the_vocabulary_keeps_everything() {
        let l = logits_of(&[0.5, 0.25, 0.25]);
        let s = Sampling {
            top_k: 1000,
            ..PLAIN
        };
        near(&frequencies(s, 3, &l, 20_000), &[0.5, 0.25, 0.25]);
    }

    #[test]
    fn top_p_keeps_the_shortest_prefix_reaching_p() {
        let l = logits_of(&[0.5, 0.3, 0.15, 0.05]);
        for (p, want) in [
            (0.49, &[0u32][..]),
            (0.79, &[0, 1]),
            (0.81, &[0, 1, 2]),
            (1.0, &[0, 1, 2, 3]),
        ] {
            let s = Sampling { top_p: p, ..PLAIN };
            assert_eq!(drawn(s, &l, 4000), want, "top-p {p}");
        }
    }

    #[test]
    fn min_p_drops_tokens_below_its_share_of_the_top() {
        // p = .5 .3 .15 .05: min-p 0.2 keeps p ≥ 0.1; min-p 0.61 keeps p ≥ 0.305, the top alone.
        let l = logits_of(&[0.5, 0.3, 0.15, 0.05]);
        let s = Sampling {
            min_p: 0.2,
            ..PLAIN
        };
        assert_eq!(drawn(s, &l, 4000), [0, 1, 2]);
        let s = Sampling {
            min_p: 0.61,
            ..PLAIN
        };
        assert_eq!(drawn(s, &l, 4000), [0]);
    }

    #[test]
    fn presence_penalty_moves_a_seen_token_down() {
        let l = [1.0, 0.9, 0.0];
        let greedy_with = |penalty: f32| {
            let mut s = Sampler::new(
                Sampling {
                    presence_penalty: penalty,
                    ..Sampling::GREEDY
                },
                0,
            );
            [s.sample(&l), s.sample(&l), s.sample(&l)]
        };
        // With 0.5: 0 first; then 0 is at 0.5, below 1's 0.9; then both are penalized once and 0 leads again.
        assert_eq!(greedy_with(0.5), [0, 1, 0]);
        assert_eq!(greedy_with(0.0), [0, 0, 0]);
    }

    #[test]
    fn the_same_seed_gives_the_same_tokens() {
        let l: Vec<f32> = (0..1000)
            .map(|i| ((i * 7919) % 1000) as f32 / 100.0)
            .collect();
        let run = |seed| {
            let mut s = Sampler::new(Sampling::DEFAULT, seed);
            (0..200).map(|_| s.sample(&l)).collect::<Vec<u32>>()
        };
        assert_eq!(run(42), run(42));
        assert_ne!(run(42), run(43));
    }

    #[test]
    fn draws_follow_the_distribution() {
        let l = logits_of(&[0.5, 0.25, 0.25]);
        near(&frequencies(PLAIN, 1, &l, 20_000), &[0.5, 0.25, 0.25]);
    }

    #[test]
    fn temperature_reshapes_the_distribution() {
        // p ∝ p^(1/T): at T = 0.5, [2/3, 1/3] becomes [4/5, 1/5].
        let l = logits_of(&[2.0 / 3.0, 1.0 / 3.0]);
        let s = Sampling { temp: 0.5, ..PLAIN };
        near(&frequencies(s, 2, &l, 20_000), &[0.8, 0.2]);
    }

    #[test]
    fn check_refuses_settings_it_cant_honour() {
        let temp = |v| Sampling { temp: v, ..PLAIN };
        let top_p = |v| Sampling { top_p: v, ..PLAIN };
        let min_p = |v| Sampling { min_p: v, ..PLAIN };
        let presence = |v| Sampling {
            presence_penalty: v,
            ..PLAIN
        };
        let bad = [
            (temp(-0.1), "--temp"),
            (temp(f32::NAN), "--temp"),
            (temp(f32::INFINITY), "--temp"),
            (top_p(0.0), "--top-p"),
            (top_p(1.1), "--top-p"),
            (top_p(f32::NAN), "--top-p"),
            (min_p(-0.1), "--min-p"),
            (min_p(1.5), "--min-p"),
            (presence(f32::NAN), "--presence-penalty"),
            (presence(f32::NEG_INFINITY), "--presence-penalty"),
        ];
        for (s, flag) in bad {
            let m = s.check().unwrap_err();
            assert!(m.starts_with(flag) && !m.contains('\n'), "{s:?}: {m}");
        }
    }

    /// A logit vector: heavy ties when `coarse`, else spread; `n` tokens.
    fn random_logits(rng: &mut Rng, n: usize, coarse: bool) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let x = rng.unit() as f32 * 12.0 - 6.0;
                if coarse {
                    (x * 2.0).round() / 2.0
                } else {
                    x
                }
            })
            .collect()
    }

    fn candidates(l: &[f32]) -> Vec<(u32, f32)> {
        l.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect()
    }

    #[test]
    fn the_partial_draw_matches_the_full_sort() {
        let mut gen = Rng::new(99);
        let mut cases = 0;
        for n in [1usize, 2, 5, 63, 64, 65, 300, 5_000] {
            for coarse in [false, true] {
                let l = random_logits(&mut gen, n, coarse);
                for temp in [0.6f32, 1.0, 5.0, 1e-6] {
                    for top_p in [1.0f32, 0.95, 0.5, 0.99999] {
                        for min_p in [0.0f32, 0.05, 1.0] {
                            let s = Sampling {
                                temp,
                                top_k: 0,
                                top_p,
                                min_p,
                                presence_penalty: 0.0,
                            };
                            for seed in 0..6 {
                                let mut a = Sampler::new(s, seed);
                                let mut b = Sampler::new(s, seed);
                                for step in 0..4 {
                                    let want = a.draw_full_sort(candidates(&l));
                                    let got = b.draw_partial(candidates(&l));
                                    assert_eq!(
                                        got, want,
                                        "n {n} coarse {coarse} {s:?} seed {seed} step {step}"
                                    );
                                    cases += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(cases > 10_000);
    }

    /// The one-pass top-k keeps what selecting from every candidate keeps, and `sample` draws the same tokens
    /// either way: spread, heavily tied and signed-zero logits (the zeros the top values, the cut on the +0/-0 edge), k
    /// from 1 to one short of the vocabulary, with and without the presence penalty.
    #[test]
    fn the_one_pass_top_k_draws_the_selections_tokens() {
        // The sampler before the one-pass top-k: every candidate, penalized, then selected and sorted by
        // `draw_full_sort`.
        fn reference(s: &mut Sampler, l: &[f32]) -> u32 {
            let mut cand = candidates(l);
            if s.s.presence_penalty != 0.0 {
                for &t in &s.seen {
                    cand[t as usize].1 -= s.s.presence_penalty;
                }
            }
            let t = s.draw_full_sort(cand);
            s.seen.insert(t);
            t
        }
        // The k best by selection over every penalized candidate, as (id, logit bits), best first.
        fn selected(l: &[f32], k: usize, seen: &BTreeSet<u32>, pen: f32) -> Vec<(u32, u32)> {
            let mut cand = penalized_candidates(l, seen, pen);
            cand.select_nth_unstable_by(k - 1, rank);
            cand.truncate(k);
            cand.sort_unstable_by(rank);
            cand.iter().map(|c| (c.0, c.1.to_bits())).collect()
        }
        let mut gen = Rng::new(31);
        let mut cases = 0;
        for n in [2usize, 3, 50, 5_000] {
            for shape in 0..3 {
                let mut l = random_logits(&mut gen, n, shape == 1);
                if shape == 2 {
                    // Zeros of both signs above everything else, so a cut falls between or among them.
                    for (i, x) in l.iter_mut().enumerate() {
                        *x = match (i % 3, i % 2) {
                            (0, 0) => 0.0,
                            (0, _) => -0.0,
                            _ => -x.abs() - 0.5,
                        };
                    }
                }
                // The k this loop covers: 1, 2, 9, 10, 20, 300 and n - 1, those below n, each once. k = 0 and k >= n
                // take `draw_partial` over every candidate, which `the_partial_draw_matches_the_full_sort` runs. 9 and
                // 10 are on the +0/-0 edge for n = 50; 300 is past `TOP_K_ONE_PASS_MAX`.
                let mut ks = vec![1, 2, 9, 10, 20, 300, n - 1];
                ks.retain(|&k| k < n);
                ks.dedup();
                for k in ks {
                    for pen in [0.0f32, 1.5] {
                        let s = Sampling {
                            temp: 0.6,
                            top_k: k,
                            top_p: 0.95,
                            min_p: 0.0,
                            presence_penalty: pen,
                        };
                        for seed in 0..3 {
                            let (mut a, mut b) = (Sampler::new(s, seed), Sampler::new(s, seed));
                            for step in 0..6 {
                                if takes_one_pass(k, n) {
                                    assert_eq!(
                                        top_k(&l, k, &b.seen, pen)
                                            .iter()
                                            .map(|c| (c.0, c.1.to_bits()))
                                            .collect::<Vec<_>>(),
                                        selected(&l, k, &b.seen, pen),
                                        "kept: n {n} shape {shape} k {k} pen {pen} seed {seed} step {step}"
                                    );
                                }
                                let want = reference(&mut a, &l);
                                assert_eq!(
                                    b.sample(&l),
                                    want,
                                    "n {n} shape {shape} k {k} pen {pen} seed {seed} step {step}"
                                );
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(cases > 1_000);
        // Every logit equal: the k lowest ids.
        let s = Sampling { top_k: 5, ..PLAIN };
        assert_eq!(drawn(s, &[1.0; 300], 400), [0, 1, 2, 3, 4]);
        // Signed zeros are apart: +0.0 ranks above −0.0, whatever their ids.
        let none = BTreeSet::new();
        assert_eq!(
            top_k(&[0.0, -0.0, 0.0, 1.0], 3, &none, 0.0),
            [(3, 1.0), (0, 0.0), (2, 0.0)]
        );
        assert_eq!(top_k(&[-0.0, 0.0], 1, &none, 0.0)[0].0, 1);
        // The guard: up to `TOP_K_ONE_PASS_MAX` and below the vocabulary. (Both paths draw the same tokens, so only this
        // pins which one `sample` takes; the timing shows why.)
        assert!(
            takes_one_pass(1, 2) && takes_one_pass(20, 248_320) && takes_one_pass(256, 248_320)
        );
        assert!(!takes_one_pass(0, 100) && !takes_one_pass(257, 248_320) && !takes_one_pass(5, 5));
        assert!(!takes_one_pass(6, 5) && !takes_one_pass(1_000, 248_320));
    }

    #[test]
    fn an_unfiltered_draw_reaches_deep_into_the_order() {
        // top_p 1, min_p 0 and a flat distribution: the draw's u decides how many tokens the prefix must hold.
        let l = vec![0.0f32; 1000];
        let s = Sampling { temp: 1.0, ..PLAIN };
        let mut smp = Sampler::new(s, 5);
        let mut deepest = 0;
        for _ in 0..200 {
            deepest = deepest.max(smp.sample(&l));
        }
        assert!(deepest > 500, "{deepest}");
    }

    #[test]
    fn min_p_of_one_keeps_only_the_top_and_its_ties() {
        let s = Sampling {
            min_p: 1.0,
            ..PLAIN
        };
        assert_eq!(drawn(s, &[1.0, 3.0, 2.0, 3.0], 500), [1, 3]);
        let big: Vec<f32> = (0..1000).map(|i| (i % 50) as f32 * 0.1).collect();
        // The top value 4.9 is at ids 49, 99, ...
        let d = drawn(s, &big, 3000);
        assert!(d.iter().all(|&i| i % 50 == 49), "{d:?}");
    }

    #[test]
    fn a_tiny_temperature_is_greedy_in_effect() {
        let l = [0.2, 1.5, 1.4, -3.0];
        let s = Sampling {
            temp: 1e-6,
            ..PLAIN
        };
        assert_eq!(drawn(s, &l, 500), [1]);
        let big: Vec<f32> = (0..2000)
            .map(|i| ((i * 7919) % 2000) as f32 * 0.01)
            .collect();
        assert_eq!(drawn(s, &big, 200), [greedy(&big)]);
    }

    #[test]
    fn the_presence_penalty_acts_on_the_sampling_path_too() {
        // A sampler with a penalty, against one fed logits with the seen tokens' penalized by hand.
        let mut gen = Rng::new(4);
        let base = random_logits(&mut gen, 3_000, false);
        for top_k in [0usize, 20] {
            let pen = 0.7f32;
            let s = Sampling {
                temp: 0.8,
                top_k,
                top_p: 0.9,
                min_p: 0.0,
                presence_penalty: pen,
            };
            let mut a = Sampler::new(s, 3);
            let mut b = Sampler::new(
                Sampling {
                    presence_penalty: 0.0,
                    ..s
                },
                3,
            );
            let mut hand = base.clone();
            for step in 0..60 {
                let got = a.sample(&base);
                assert_eq!(got, b.sample(&hand), "top_k {top_k} step {step}");
                hand[got as usize] = base[got as usize] - pen;
            }
        }
        // And greedy over many seen tokens equals greedy over a hand-penalized copy.
        let mut g = Sampler::new(
            Sampling {
                presence_penalty: 1.5,
                ..Sampling::GREEDY
            },
            0,
        );
        let mut hand = base.clone();
        for _ in 0..40 {
            let got = g.sample(&base);
            assert_eq!(got, greedy(&hand));
            hand[got as usize] = base[got as usize] - 1.5;
        }
    }

    /// ms per token of the full sort and the partial draw over 248,320 random logits, and of the top-k selection over every
    /// candidate against the one pass (production's k of 20, and larger ks), for the record. Run with
    /// `cargo test --release -p engine sampler_speed -- --ignored --nocapture`.
    #[test]
    #[ignore = "a timing, not a check"]
    fn sampler_speed() {
        let mut gen = Rng::new(1);
        let flat = random_logits(&mut gen, 248_320, false);
        // Peaked like a language model's: a few tokens far above a long tail.
        let peaked: Vec<f32> = (0..248_320)
            .map(|_| 14.0 * (gen.unit() as f32).powi(12) - 8.0 + gen.unit() as f32)
            .collect();
        let time = |f: &mut dyn FnMut()| {
            let t = std::time::Instant::now();
            for _ in 0..20 {
                f();
            }
            t.elapsed().as_secs_f64() * 1000.0 / 20.0
        };
        for (shape, l) in [("flat", &flat), ("peaked", &peaked)] {
            for (name, top_p) in [("top_p 1, min_p 0", 1.0f32), ("top_p 0.95", 0.95)] {
                let s = Sampling {
                    temp: 0.6,
                    top_k: 0,
                    top_p,
                    min_p: 0.0,
                    presence_penalty: 0.0,
                };
                let mut a = Sampler::new(s, 1);
                let mut b = Sampler::new(s, 1);
                let full = time(&mut || {
                    a.draw_full_sort(candidates(l));
                });
                let part = time(&mut || {
                    b.draw_partial(candidates(l));
                });
                println!(
                    "{shape}, {name}: full sort {full:.2} ms/token, partial {part:.2} ms/token"
                );
            }
            // Production's top-k of 20: selecting from every candidate against the one pass.
            let (mut a, mut b) = (
                Sampler::new(Sampling::DEFAULT, 1),
                Sampler::new(Sampling::DEFAULT, 1),
            );
            let select = time(&mut || {
                a.draw_full_sort(candidates(l));
            });
            let pass = time(&mut || {
                b.sample(l);
            });
            println!("{shape}, top_k 20: selection over every candidate {select:.3} ms/token, one pass {pass:.3} ms/token");
            // Why the one pass stops at `TOP_K_ONE_PASS_MAX`: its insertion is O(n·k).
            for k in [256usize, 1_000, 10_000] {
                let s = Sampling {
                    top_k: k,
                    ..Sampling::DEFAULT
                };
                let mut a = Sampler::new(s, 1);
                let select = time(&mut || {
                    a.draw_full_sort(candidates(l));
                });
                let none = BTreeSet::new();
                let pass = time(&mut || {
                    a.draw_full_sort(top_k(l, k, &none, 0.0));
                });
                println!("{shape}, top_k {k}: selection over every candidate {select:.3} ms/token, one pass {pass:.3} ms/token");
            }
        }
    }
}
