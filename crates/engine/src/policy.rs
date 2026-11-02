//! The adaptive cache policy: a decayed count of each expert's routes per layer, seeded from the
//! static profile, and a swap rule with hysteresis under an upload cap. It is pure (no GPU), so CI tests it;
//! the engine runs it inline between steps: `observe` after each finished step, `choose` in `prepare_step`.
//!
//! Scores decay with a half-life in tokens. Rather than multiplying every score each step, the policy keeps one global
//! scale that grows by 2^(t / half-life) over a step of t tokens and adds each route at the scale of its token; a score
//! is its stored value over the scale. The scale is renormalized (every stored value divided by it, the scale set to 1)
//! before it passes `RENORM`, far inside f64's range.
//!
//! What a step may spend on copies is `allowance`'s: a token's worth of the cap for a decode step, and a prefill chunk's
//! share saved for the step after it, which picks with the chunk's routes counted, so the cache is reshaped to the
//! prompt before generation goes on.
use crate::{
    swap::{Residency, SwapPlanner},
    Profile, PREFILL_CHUNK,
};
use qwen36::dims::Dims;
use std::cmp::Ordering;

/// The policy's values: a half-life of 128 tokens, tuned from a starting 512
/// (the simulated golden set gained in every group, and the gate prompts' decode 3.9% on the GPU; 64 gained
/// no more than the runs' spread); the starting margin of 25% and upload cap of 8 MiB of copies per token, which the tuning kept. The prior is
/// about 512 tokens' worth of routes.
pub const HALF_LIFE: f64 = 128.0;
pub const MARGIN: f64 = 1.25;
pub const UPLOAD_CAP: u64 = 8 << 20;
pub const PRIOR_TOKENS: f64 = 512.0;

/// The most bytes of copies one step may spend, or save for the next, whatever the step: 2 GiB, 256 tokens' worth at 8 MiB a token, about 1,100 copies of the real model's
/// experts, about 148 ms on the transfer queue at a measured 14.5 GB/s. `allowance` bounds a step's picks by
/// it; one submit carries one step's picks, while each poll sends what the step before it picked. That rests on the
/// engine waiting for each step (`submit_step`), so every copy the step before picked has drained by the next poll; a
/// pipelined engine must bound the poll's batch itself (`prepare_step` asserts the bound). `Engine::open` refuses an
/// upload cap above it (`Error::BadUploadCap`), so one token's share never passes it either.
pub const RESHAPE_BYTES: u64 = 2 << 30;

/// The upload cap an engine runs with: `asked` (`Options::upload_cap`), else `UPLOAD_CAP`; refused outside
/// 1 ..= `RESHAPE_BYTES` (a cap of 0 would make no copy at all, which is `--cache-policy static`'s job).
pub(crate) fn cap_for(asked: Option<u64>) -> Result<u64, crate::Error> {
    let cap = asked.unwrap_or(UPLOAD_CAP);
    if !(1..=RESHAPE_BYTES).contains(&cap) {
        return Err(crate::Error::BadUploadCap {
            cap,
            max: RESHAPE_BYTES,
        });
    }
    Ok(cap)
}

/// A step, as the upload cap sees it (`allowance`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyStep {
    /// A decode step: one token.
    Decode,
    /// A prefill chunk that reads its non-resident experts in place.
    Prefill,
    /// A staged prefill chunk.
    Staged,
    /// A speculative verify step: prepared like a one-token decode step (callers pass `tokens = 1`), so
    /// it spends one token's share and the bank. `Engine::settle_verify` banks the rest once the step knows how many
    /// tokens it kept (`earned`).
    Verify,
}

/// What a step of `kind` and `tokens` may spend on the policy's copies, and what it leaves saved for the next step,
/// given `bank` saved before it and `cap` bytes a token: `(spend, bank)`.
///
/// A prefill chunk's own share is saved for the step after it, which picks with the chunk's routes counted (a chunk
/// picks before it runs, from the routes before it). An unstaged chunk spends what the steps before it saved; a staged
/// one spends nothing, since a swap would leave both its experts HOST for the chunk, one more than the staging buffer
/// holds, and adds its share to the bank. A decode step spends its token's share and the bank. No step spends or saves
/// more than `RESHAPE_BYTES` (`cap` is at most that: `Engine::open` refuses a larger one).
pub fn allowance(kind: PolicyStep, tokens: usize, bank: u64, cap: u64) -> (u64, u64) {
    let most = RESHAPE_BYTES.max(cap);
    let own = cap.saturating_mul(tokens as u64).min(most);
    match kind {
        PolicyStep::Decode | PolicyStep::Verify => (bank.saturating_add(own).min(most), 0),
        PolicyStep::Prefill => (bank.min(most), own),
        PolicyStep::Staged => (0, bank.saturating_add(own).min(most)),
    }
}

/// What a verify step that kept `kept` tokens (the committed row and its accepted drafts, at least 1) earns for the
/// steps after it: `cap` for each token beyond the first, at most `RESHAPE_BYTES.max(cap)`. The first token's share
/// was spent when the step was prepared (`allowance`'s `Verify`), so a step that kept one earns nothing and a rejected
/// draft earns nothing.
pub fn earned(kept: usize, cap: u64) -> u64 {
    cap.saturating_mul(kept.saturating_sub(1) as u64)
        .min(RESHAPE_BYTES.max(cap))
}

/// The bank after a verify step that kept `kept` tokens: what was left plus `earned`, at most `RESHAPE_BYTES.max(cap)`
/// (what `allowance` lets any bank hold).
pub fn bank_after_verify(bank: u64, kept: usize, cap: u64) -> u64 {
    bank.saturating_add(earned(kept, cap))
        .min(RESHAPE_BYTES.max(cap))
}

/// The scale past which stored scores are renormalized. A stored score is at most the prior plus about 185 tokens'
/// worth of routes (the sum of the decay over every past token at a half-life of 128) times the scale, far below f64's
/// 1.8e308.
const RENORM: f64 = 1e100;

/// The policy's tunable values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyParams {
    /// Tokens after which a route counts half. `CachePolicy::new` requires it above 0 and the growth of the scale over
    /// the largest step (a `PREFILL_CHUNK` of tokens), 2^(PREFILL_CHUNK / half_life), under `RENORM`: a half-life of
    /// at least about 6.2 tokens.
    pub half_life: f64,
    /// A non-resident expert replaces a resident one only when its score is more than `margin` times the other's.
    pub margin: f64,
    /// Bytes of copies per token of the steps the copies are made for. The engine reads the policy's cap (`params()`)
    /// for `allowance`: it keeps no copy of its own.
    pub cap: u64,
    /// The static profile's weight at the start, in tokens' worth of routes.
    pub prior_tokens: f64,
}

impl Default for PolicyParams {
    fn default() -> PolicyParams {
        PolicyParams {
            half_life: HALF_LIFE,
            margin: MARGIN,
            cap: UPLOAD_CAP,
            prior_tokens: PRIOR_TOKENS,
        }
    }
}

/// One change the policy asks for in `layer`: `inn` replaces `out` (a swap, `SwapPlanner::request`), or with no `out`
/// fills an empty slot (`SwapPlanner::fill`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pick {
    pub layer: usize,
    pub out: Option<usize>,
    pub inn: usize,
}

/// Decayed route counts per (layer, expert), and the rule that turns them into swaps.
pub struct CachePolicy {
    n_layer: usize,
    n_expert: usize,
    n_used: usize,
    params: PolicyParams,
    /// Whether scores follow the routes; off (`--cache-policy static`), they stay the prior, and only empty slots are
    /// filled, in the profile's order.
    adapt: bool,
    /// Each score times `scale`, [layer][expert].
    stored: Vec<f64>,
    scale: f64,
}

/// `a` ranks above `b`: a higher score, or an equal one and a lower id (as the profile ranks experts).
fn above(a: (f64, usize), b: (f64, usize)) -> bool {
    match a.0.total_cmp(&b.0) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => a.1 < b.1,
    }
}

impl CachePolicy {
    /// Scores seeded from `profile`: in each layer, the counts scaled to `prior_tokens` tokens' worth of routes
    /// (`n_expert_used` a token). A layer whose counts are all zero (the tiny model's profile) starts at zero.
    pub fn new(profile: &Profile, d: &Dims, adapt: bool, params: PolicyParams) -> CachePolicy {
        let mut stored = Vec::with_capacity(d.n_layer * d.n_expert);
        for l in 0..d.n_layer {
            let total: u64 = (0..d.n_expert).map(|e| profile.count(l, e)).sum();
            let per = if total > 0 {
                params.prior_tokens * d.n_expert_used as f64 / total as f64
            } else {
                0.0
            };
            stored.extend((0..d.n_expert).map(|e| profile.count(l, e) as f64 * per));
        }
        assert!(
            params.half_life > 0.0 && (PREFILL_CHUNK as f64 / params.half_life).exp2() < RENORM,
            "half_life {} is out of range: 2^({PREFILL_CHUNK} / half_life) must be under {RENORM}",
            params.half_life
        );
        CachePolicy {
            n_layer: d.n_layer,
            n_expert: d.n_expert,
            n_used: d.n_expert_used,
            params,
            adapt,
            stored,
            scale: 1.0,
        }
    }

    pub fn adapts(&self) -> bool {
        self.adapt
    }

    pub fn params(&self) -> PolicyParams {
        self.params
    }

    /// `expert`'s score in `layer`: its decayed route count, in routes. Panics out of range.
    pub fn score(&self, layer: usize, expert: usize) -> f64 {
        assert!(layer < self.n_layer && expert < self.n_expert);
        self.stored[layer * self.n_expert + expert] / self.scale
    }

    /// Counts a finished step's routes: `ids`, each layer's `tokens` × n_expert_used expert ids in turn (the routing
    /// log's layout). Token i of the step counts 2^(−(tokens − 1 − i) / half-life) of the last one. Ids that aren't
    /// an expert are skipped (the step's status check reports them). Nothing changes when the policy doesn't adapt.
    pub fn observe(&mut self, ids: &[f32], tokens: usize) {
        debug_assert_eq!(ids.len(), self.n_layer * tokens * self.n_used);
        if !self.adapt || tokens == 0 {
            return;
        }
        let h = self.params.half_life;
        let grow = (tokens as f64 / h).exp2();
        if self.scale * grow > RENORM {
            for s in &mut self.stored {
                *s /= self.scale;
            }
            self.scale = 1.0;
        }
        let end = self.scale * grow;
        let w: Vec<f64> = (0..tokens)
            .map(|i| end * (-((tokens - 1 - i) as f64) / h).exp2())
            .collect();
        let per_layer = tokens * self.n_used;
        for (j, &id) in ids.iter().enumerate().take(self.n_layer * per_layer) {
            let (l, i) = (j / per_layer, (j % per_layer) / self.n_used);
            if id >= 0.0 && id.fract() == 0.0 && (id as usize) < self.n_expert {
                self.stored[l * self.n_expert + id as usize] += w[i];
            }
        }
        self.scale = end;
    }

    /// In `layer`, ignoring `skip`: the best expert the planner says is `want`, by `above`, or the worst.
    fn find(
        &self,
        p: &SwapPlanner,
        layer: usize,
        want: Residency,
        best: bool,
        skip: &[usize],
    ) -> Option<usize> {
        let mut found: Option<(f64, usize)> = None;
        for e in 0..self.n_expert {
            if skip.contains(&e) || p.residency(layer, e) != want {
                continue;
            }
            let c = (self.score(layer, e), e);
            if found.is_none_or(|f| if best { above(c, f) } else { above(f, c) }) {
                found = Some(c);
            }
        }
        found.map(|f| f.1)
    }

    /// The changes to make before the next step, at most `budget` bytes of copies in all (`slot_bytes`, per layer).
    /// In each layer: an empty slot takes the best-scoring free expert; otherwise, when the policy adapts, the
    /// best-scoring free expert replaces the coldest held one if its score is more than `margin` times the other's.
    /// Busy experts are left alone. Across layers the change that gains most goes first (a fill gains its expert's
    /// score, a swap the difference), ties to the lower layer; it stops at the first that doesn't fit the budget.
    pub fn choose(&self, p: &SwapPlanner, budget: u64, slot_bytes: &[u64]) -> Vec<Pick> {
        assert_eq!(p.n_layer(), self.n_layer);
        assert_eq!(slot_bytes.len(), self.n_layer);
        // Per layer: the experts already picked, in and out, the empty slots left, and the next change and its gain.
        struct Layer {
            ins: Vec<usize>,
            outs: Vec<usize>,
            empty: usize,
            next: Option<(f64, Pick)>,
        }
        let next = |l: usize, s: &Layer| -> Option<(f64, Pick)> {
            let inn = self.find(p, l, Residency::Free, true, &s.ins)?;
            let gain = self.score(l, inn);
            if s.empty > 0 {
                return Some((
                    gain,
                    Pick {
                        layer: l,
                        out: None,
                        inn,
                    },
                ));
            }
            if !self.adapt {
                return None;
            }
            let out = self.find(p, l, Residency::Held, false, &s.outs)?;
            let cold = self.score(l, out);
            (gain > self.params.margin * cold).then_some((
                gain - cold,
                Pick {
                    layer: l,
                    out: Some(out),
                    inn,
                },
            ))
        };
        let mut layers: Vec<Layer> = (0..self.n_layer)
            .map(|l| {
                let mut s = Layer {
                    ins: Vec::new(),
                    outs: Vec::new(),
                    empty: p.empty_slots(l),
                    next: None,
                };
                s.next = next(l, &s);
                s
            })
            .collect();
        let (mut picks, mut left) = (Vec::new(), budget);
        loop {
            let mut top: Option<(f64, usize)> = None;
            for (l, s) in layers.iter().enumerate() {
                if let Some((g, _)) = s.next {
                    if top.is_none_or(|t| g.total_cmp(&t.0) == Ordering::Greater) {
                        top = Some((g, l));
                    }
                }
            }
            let Some((_, l)) = top else { break };
            let bytes = slot_bytes[l];
            if bytes > left {
                break;
            }
            left -= bytes;
            let pick = layers[l].next.expect("the top layer has a change").1;
            picks.push(pick);
            let s = &mut layers[l];
            s.ins.push(pick.inn);
            match pick.out {
                Some(o) => s.outs.push(o),
                None => s.empty -= 1,
            }
            layers[l].next = next(l, &layers[l]);
        }
        picks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swap::SwapPlanner;

    /// A policy on the tiny model's dims (4 layers of 16 experts, 4 routed a token) with `counts` as every layer's
    /// profile, zeros after them.
    fn policy(counts: &[u64], adapt: bool, params: PolicyParams) -> CachePolicy {
        let d = Dims::tiny();
        let mut c = vec![0u64; d.n_layer * d.n_expert];
        for l in 0..d.n_layer {
            c[l * d.n_expert..l * d.n_expert + counts.len()].copy_from_slice(counts);
        }
        let b: Vec<u8> = c.iter().flat_map(|x| x.to_le_bytes()).collect();
        CachePolicy::new(&Profile::from_bytes(&b, &d).unwrap(), &d, adapt, params)
    }

    /// One decoded token's routes, `ids[l]` in layer l, in the routing log's layout.
    fn routes(ids: [[usize; 4]; 4]) -> Vec<f32> {
        ids.iter().flat_map(|t| t.map(|e| e as f32)).collect()
    }

    /// The same four ids in every layer.
    fn token(ids: [usize; 4]) -> Vec<f32> {
        routes([ids; 4])
    }

    /// Scores barely decay: the counts, nearly.
    fn flat() -> PolicyParams {
        PolicyParams {
            half_life: 1e9,
            ..PolicyParams::default()
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * b.abs().max(1.0)
    }

    #[test]
    fn a_route_counts_half_after_one_half_life() {
        let mut p = policy(&[], true, PolicyParams::default());
        p.observe(&token([3, 4, 5, 6]), 1);
        assert!(close(p.score(0, 3), 1.0), "{}", p.score(0, 3));
        let h = HALF_LIFE as usize;
        for _ in 0..h {
            p.observe(&token([7, 8, 9, 10]), 1);
        }
        assert!(close(p.score(2, 3), 0.5), "{}", p.score(2, 3));
        // Expert 7, routed in each of those tokens: the sum of 2^(−i / half-life) for i in 0..half-life.
        let want: f64 = (0..h).map(|i| (-(i as f64) / HALF_LIFE).exp2()).sum();
        assert!(
            close(p.score(1, 7), want),
            "{} against {want}",
            p.score(1, 7)
        );
        assert_eq!(p.score(0, 0), 0.0, "never routed");
    }

    #[test]
    fn a_chunks_tokens_count_by_their_place_in_it() {
        // Three tokens in one step: the last counts 1, the one before 2^(−1/128), the first 2^(−2/128): the scores of
        // the same routes decoded a token at a time.
        let (mut chunk, mut one) = (
            policy(&[], true, PolicyParams::default()),
            policy(&[], true, PolicyParams::default()),
        );
        let toks = [[1, 2, 3, 4], [1, 5, 6, 7], [1, 2, 8, 9]];
        // The routing log's layout: each layer's tokens × 4 ids in turn.
        let ids: Vec<f32> = (0..4)
            .flat_map(|_| toks.iter().flat_map(|t| t.map(|e| e as f32)))
            .collect();
        chunk.observe(&ids, 3);
        for t in toks {
            one.observe(&token(t), 1);
        }
        let w = |i: i32| (-(i as f64) / HALF_LIFE).exp2();
        assert!(close(chunk.score(0, 1), w(0) + w(1) + w(2)));
        assert!(close(chunk.score(3, 2), w(0) + w(2)));
        assert!(close(chunk.score(1, 5), w(1)));
        for l in 0..4 {
            for e in 0..16 {
                assert!(
                    close(chunk.score(l, e), one.score(l, e)),
                    "layer {l} expert {e}"
                );
            }
        }
    }

    #[test]
    fn the_scale_is_renormalized_before_it_overflows() {
        // A half-life of 8 tokens grows the scale by 2^(1/8) a token: 4,000 tokens would take it to 2^500, past RENORM
        // (1e100). The scores stay what direct decay gives.
        let params = PolicyParams {
            half_life: 8.0,
            ..PolicyParams::default()
        };
        let r = (-1.0f64 / 8.0).exp2();
        let mut p = policy(&[], true, params);
        p.observe(&token([0, 1, 2, 3]), 1);
        for _ in 0..3_999 {
            p.observe(&token([4, 5, 6, 7]), 1);
        }
        assert!(p.scale.is_finite() && p.scale <= RENORM, "{}", p.scale);
        // Routed every token: 1 + r + r^2 + ... = 1 / (1 - r).
        assert!(close(p.score(0, 4), 1.0 / (1.0 - r)), "{}", p.score(0, 4));
        assert!(
            p.score(0, 0) < 1e-100,
            "3,999 tokens on, it counts for nothing"
        );
        // Step until the next token renormalizes, then take every score and observe that token: each old score is
        // worth r of itself, the new routes 1, and the scale is 1 grown by one token.
        while p.scale * (1.0f64 / 8.0).exp2() <= RENORM {
            p.observe(&token([4, 5, 6, 7]), 1);
        }
        assert!(p.scale > 1e99, "{}", p.scale);
        let snap: Vec<f64> = (0..4 * 16).map(|k| p.score(k / 16, k % 16)).collect();
        assert!(snap[4] > 1.0, "expert 4 holds a real score to lose");
        p.observe(&token([8, 9, 10, 11]), 1);
        assert!(close(p.scale, (1.0f64 / 8.0).exp2()), "{}", p.scale);
        for (k, &old) in snap.iter().enumerate() {
            let new = if (8..12).contains(&(k % 16)) {
                1.0
            } else {
                0.0
            };
            assert!(
                close(p.score(k / 16, k % 16), old * r + new),
                "layer {} expert {}: {} against {}",
                k / 16,
                k % 16,
                p.score(k / 16, k % 16),
                old * r + new
            );
        }
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn a_half_life_of_nothing_is_refused() {
        policy(
            &[],
            true,
            PolicyParams {
                half_life: 0.0,
                ..PolicyParams::default()
            },
        );
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn a_half_life_whose_chunk_would_overflow_the_scale_is_refused() {
        policy(
            &[],
            true,
            PolicyParams {
                half_life: 1.0,
                ..PolicyParams::default()
            },
        );
    }

    #[test]
    fn the_prior_is_512_tokens_of_routes_in_the_profiles_proportions() {
        // Each layer's prior sums to 512 tokens × n_expert_used routes, ranked as the profile ranks.
        let d = Dims::QWEN36;
        let prof = Profile::for_dims(&d);
        let p = CachePolicy::new(&prof, &d, true, PolicyParams::default());
        for l in [0, 17, 39] {
            let sum: f64 = (0..d.n_expert).map(|e| p.score(l, e)).sum();
            assert!(close(sum, 512.0 * 8.0), "layer {l}: {sum}");
            let mut by_score: Vec<usize> = (0..d.n_expert).collect();
            by_score.sort_by(|&a, &b| p.score(l, b).total_cmp(&p.score(l, a)).then(a.cmp(&b)));
            assert_eq!(by_score, prof.ranked(l), "layer {l}");
        }
        // A profile of zeros (the tiny model's) starts every score at zero.
        let z = policy(&[], true, PolicyParams::default());
        assert!((0..16).all(|e| z.score(1, e) == 0.0));
    }

    #[test]
    fn ids_that_are_no_expert_are_skipped() {
        let mut p = policy(&[], true, PolicyParams::default());
        let mut ids = token([1, 2, 3, 4]);
        // Layer 0: out of range (16 would land on layer 1's expert 0), negative (-1 casts to 0), fractional, NaN.
        ids[..4].copy_from_slice(&[16.0, -1.0, 2.5, f32::NAN]);
        // The last layer: out of range, past the table's end (an index panic without the check), and infinite.
        ids[12..].copy_from_slice(&[100.0, 16.0, 1e9, f32::INFINITY]);
        p.observe(&ids, 1);
        for e in 0..16 {
            assert_eq!(p.score(0, e), 0.0, "layer 0 expert {e}");
            assert_eq!(p.score(3, e), 0.0, "layer 3 expert {e}");
        }
        assert_eq!(
            p.score(1, 0),
            0.0,
            "16 is no expert of layer 0, and not layer 1's 0"
        );
        assert!(close(p.score(1, 1), 1.0));
    }

    #[test]
    fn a_static_policy_never_moves() {
        let mut p = policy(&[5, 4, 3, 2, 1], false, PolicyParams::default());
        let before: Vec<f64> = (0..16).map(|e| p.score(0, e)).collect();
        p.observe(&token([12, 13, 14, 15]), 1);
        assert_eq!((0..16).map(|e| p.score(0, e)).collect::<Vec<_>>(), before);
        // A full cache of experts the profile never counted: no swap, however cold.
        let pl = SwapPlanner::new(16, vec![vec![10, 11]; 4]);
        assert_eq!(p.choose(&pl, u64::MAX, &[100; 4]), []);
    }

    /// A planner of 4 layers whose slots hold `resident` in every layer.
    fn planner(resident: &[usize]) -> SwapPlanner {
        SwapPlanner::new(16, vec![resident.to_vec(); 4])
    }

    /// Scores from `(expert, tokens)`, in order: each token routes all four of layer 0's ids to the expert, and the
    /// other layers' to `other`.
    fn scored(counts: &[(usize, usize)], other: usize, params: PolicyParams) -> CachePolicy {
        let mut p = policy(&[], true, params);
        for &(e, n) in counts {
            for _ in 0..n {
                p.observe(&routes([[e; 4], [other; 4], [other; 4], [other; 4]]), 1);
            }
        }
        p
    }

    #[test]
    fn a_swap_needs_more_than_the_margin() {
        // On the prior alone: expert 1's score is exactly 1.25 × expert 0's (profile counts 5 and 4), so 1 doesn't
        // replace resident 0; at counts 6 and 4 it does, in each of the four layers (every layer has this profile).
        let pl = planner(&[0]);
        let p = policy(&[4, 5], true, PolicyParams::default());
        assert_eq!(p.score(0, 1), 1.25 * p.score(0, 0));
        assert_eq!(p.choose(&pl, u64::MAX, &[100; 4]), []);
        let p = policy(&[4, 6], true, PolicyParams::default());
        assert_eq!(
            p.choose(&pl, u64::MAX, &[100; 4]),
            (0..4)
                .map(|layer| Pick {
                    layer,
                    out: Some(0),
                    inn: 1
                })
                .collect::<Vec<_>>()
        );
        // Routes move it the same way: 0 routed for 100 tokens after 1's 125 is no swap (1's routes are older), 126
        // is. The other layers route their resident, so they have nothing to swap.
        assert_eq!(
            scored(&[(1, 125), (0, 100)], 0, flat()).choose(&pl, u64::MAX, &[100; 4]),
            []
        );
        assert_eq!(
            scored(&[(1, 126), (0, 100)], 0, flat()).choose(&pl, u64::MAX, &[100; 4]),
            [Pick {
                layer: 0,
                out: Some(0),
                inn: 1
            }]
        );
    }

    #[test]
    fn ties_go_to_the_lower_id() {
        // Experts 4 and 6 tie as the best candidates (10 routes each, added in one step so their scores are equal):
        // 4 goes first. Residents 1 and 2 tie as the coldest (no routes): 2, ranked below 1 as the profile ranks ties,
        // goes first. The other layers route their residents, so they have nothing to swap.
        let mut p = policy(&[], true, flat());
        for _ in 0..10 {
            p.observe(
                &routes([[4, 6, 0, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
                1,
            );
        }
        let pl = planner(&[0, 1, 2, 3]);
        assert_eq!(
            p.choose(&pl, 100, &[100; 4]),
            [Pick {
                layer: 0,
                out: Some(2),
                inn: 4
            }]
        );
        // With room for two, 6 then replaces 1.
        assert_eq!(
            p.choose(&pl, 200, &[100; 4]),
            [
                Pick {
                    layer: 0,
                    out: Some(2),
                    inn: 4
                },
                Pick {
                    layer: 0,
                    out: Some(1),
                    inn: 6
                }
            ]
        );
    }

    #[test]
    fn the_upload_cap_takes_the_largest_gains_first() {
        // Every layer holds experts 4 to 7. In layers 0 and 1 they are never routed: layer 0 routes 8 to 11 for ten
        // tokens and 0 to 3 for ten, layer 1 routes 12 to 15 for twenty. Layers 2 and 3 route their residents. A swap
        // gains its score difference, so layer 1's four swaps (20 each) come before layer 0's (10 each).
        let mut p = policy(&[], true, flat());
        for _ in 0..10 {
            p.observe(
                &routes([[8, 9, 10, 11], [12, 13, 14, 15], [4, 5, 6, 7], [4, 5, 6, 7]]),
                1,
            );
        }
        for _ in 0..10 {
            p.observe(
                &routes([[0, 1, 2, 3], [12, 13, 14, 15], [4, 5, 6, 7], [4, 5, 6, 7]]),
                1,
            );
        }
        let pl = planner(&[4, 5, 6, 7]);
        let layers = |picks: Vec<Pick>| picks.iter().map(|k| k.layer).collect::<Vec<_>>();
        assert_eq!(layers(p.choose(&pl, 500, &[100; 4])), [1, 1, 1, 1, 0]);
        // A budget a byte short of a copy takes none; a copy is never split.
        assert_eq!(p.choose(&pl, 99, &[100; 4]), []);
        // Each layer's own slot size counts: at 150 a copy, layer 1's three take 450 of 550. Its fourth (150) doesn't
        // fit the 100 left, and layer 0's (100) would: the choice stops at the first change that doesn't fit rather
        // than skipping it, so the order holds.
        assert_eq!(layers(p.choose(&pl, 550, &[100, 150, 100, 100])), [1, 1, 1]);
    }

    #[test]
    fn empty_slots_are_filled_first_and_busy_experts_left_alone() {
        // Rows 2 and 3 are added empty in every layer: each layer's two best free experts fill them. Expert 9, the best
        // in layer 0 but in a swap, is never picked.
        let p = scored(&[(8, 5), (9, 50), (10, 4)], 15, flat());
        let mut pl = planner(&[0, 1]);
        pl.add(2..4, 0).unwrap();
        pl.request(0, 0, 9).unwrap();
        let picks = p.choose(&pl, u64::MAX, &[100; 4]);
        let fill = |layer, inn| Pick {
            layer,
            out: None,
            inn,
        };
        assert_eq!(
            picks
                .iter()
                .copied()
                .filter(|k| k.layer == 0)
                .collect::<Vec<_>>(),
            [fill(0, 8), fill(0, 10)]
        );
        // In layers 1 to 3 every expert but 15 scores 0: their fills take 15, then the lowest free id, 2, and no swap
        // follows (a score of 0 is not more than 1.25 × 0).
        assert_eq!(picks.len(), 8, "{picks:?}");
        assert!(picks.iter().all(|k| k.out.is_none() && k.inn != 9));
        assert_eq!(
            picks
                .iter()
                .filter(|k| k.layer == 2)
                .map(|k| k.inn)
                .collect::<Vec<_>>(),
            [15, 2]
        );
        // A static policy fills too, by the profile's ranking: 2 and 3 (9 counts each) are resident, so 4 (7) is next.
        let s = policy(&[1, 1, 9, 9, 7], false, PolicyParams::default());
        let mut pl = planner(&[2, 3]);
        pl.add(2..3, 0).unwrap();
        assert_eq!(s.choose(&pl, u64::MAX, &[100; 4])[0], fill(0, 4));
    }

    #[test]
    fn a_prefill_chunk_saves_its_share_for_the_next_step_and_no_step_spends_past_the_reshape() {
        const CAP: u64 = 8 << 20;
        let most = RESHAPE_BYTES;
        assert_eq!(most, CAP * 256, "256 tokens' worth at the default cap");
        // A decode step spends its token's share and what was saved.
        assert_eq!(allowance(PolicyStep::Decode, 1, 0, CAP), (CAP, 0));
        assert_eq!(allowance(PolicyStep::Decode, 1, 5 * CAP, CAP), (6 * CAP, 0));
        assert_eq!(allowance(PolicyStep::Decode, 1, most, CAP), (most, 0));
        // A short prompt, one unstaged chunk of 20 tokens: it spends what was saved before it (nothing here) and saves
        // its 20 tokens' worth for the first decode step, which picks with the prompt's routes counted.
        assert_eq!(allowance(PolicyStep::Prefill, 20, 0, CAP), (0, 20 * CAP));
        assert_eq!(
            allowance(PolicyStep::Decode, 1, 20 * CAP, CAP),
            (21 * CAP, 0)
        );
        // Staged chunks spend nothing and save up to the reshape.
        assert_eq!(
            allowance(PolicyStep::Staged, 100, 10 * CAP, CAP),
            (0, 110 * CAP)
        );
        assert_eq!(allowance(PolicyStep::Staged, 2_048, 0, CAP), (0, most));
        assert_eq!(allowance(PolicyStep::Staged, 2_048, most, CAP), (0, most));
        // An unstaged 2,048-token chunk (no staging buffers) spends and saves at most the reshape.
        assert_eq!(
            allowance(PolicyStep::Prefill, 2_048, most, CAP),
            (most, most)
        );
        // A larger cap (`--upload-cap`) doesn't raise the bound: a step's copies stay one submit of at most 2 GiB.
        let big = 64 << 20;
        assert_eq!(allowance(PolicyStep::Staged, 2_048, 0, big), (0, most));
        assert_eq!(allowance(PolicyStep::Decode, 1, most, big), (most, 0));
        // Huge banks and caps are clamped and saturate rather than wrapping or overflowing, for every kind.
        assert_eq!(
            allowance(PolicyStep::Decode, 1, u64::MAX, u64::MAX),
            (u64::MAX, 0)
        );
        assert_eq!(
            allowance(PolicyStep::Prefill, 1, u64::MAX, CAP),
            (most, CAP)
        );
        assert_eq!(
            allowance(PolicyStep::Staged, 2, u64::MAX, u64::MAX),
            (0, u64::MAX)
        );
        // Over random runs of steps: no step spends more than the reshape, and nothing is spent that the steps' tokens
        // didn't earn.
        let mut rng = crate::rng::Rng::new(7);
        let (mut bank, mut earned, mut spent) = (0, 0, 0);
        for _ in 0..10_000 {
            // A verify step prepares as one token and settles with the tokens it kept (`bank_after_verify`).
            let (kind, tokens, kept) = match rng.below(4) {
                0 => (PolicyStep::Decode, 1, 1),
                1 => (PolicyStep::Prefill, 1 + rng.below(2_048), 0),
                2 => (PolicyStep::Staged, 128 + rng.below(1_921), 0),
                _ => (PolicyStep::Verify, 1, 1 + rng.below(16)),
            };
            let (spend, left) = allowance(kind, tokens, bank, CAP);
            let settled = if kind == PolicyStep::Verify {
                bank_after_verify(left, kept, CAP)
            } else {
                left
            };
            assert!(
                spend <= most && settled <= most,
                "{kind:?} {tokens}: {spend} {settled}"
            );
            earned += CAP * tokens.max(kept) as u64;
            spent += spend;
            assert!(spent + settled <= earned, "{kind:?} {tokens} {kept}");
            bank = settled;
        }
    }

    #[test]
    fn a_verify_step_spends_like_a_decode_step_and_banks_the_rest() {
        const CAP: u64 = 8 << 20;
        for (bank, cap) in [
            (0, CAP),
            (5 * CAP, CAP),
            (RESHAPE_BYTES, CAP),
            (u64::MAX, u64::MAX),
        ] {
            assert_eq!(
                allowance(PolicyStep::Verify, 1, bank, cap),
                allowance(PolicyStep::Decode, 1, bank, cap),
                "bank {bank} cap {cap}"
            );
        }
        // A step that kept one token (every draft rejected, or none drafted) earns nothing; each further token earns a
        // cap's worth.
        assert_eq!(earned(0, CAP), 0);
        assert_eq!(earned(1, CAP), 0);
        assert_eq!(earned(4, CAP), 3 * CAP);
        // At most the reshape, or one cap when the cap itself is larger; saturating, never wrapping.
        assert_eq!(earned(1_000, CAP), RESHAPE_BYTES);
        assert_eq!(earned(1_000, u64::MAX), u64::MAX);
        assert_eq!(earned(3, 3 << 30), 3 << 30);
        // The bank takes what the step earned, up to the most a bank holds.
        assert_eq!(bank_after_verify(0, 4, CAP), 3 * CAP);
        assert_eq!(bank_after_verify(2 * CAP, 1, CAP), 2 * CAP);
        assert_eq!(
            bank_after_verify(RESHAPE_BYTES - CAP, 4, CAP),
            RESHAPE_BYTES
        );
        assert_eq!(bank_after_verify(u64::MAX, 4, CAP), RESHAPE_BYTES);
        // Over a run of mixed steps nothing is spent that the kept tokens didn't earn: a decode step earns its token's
        // share (spent in `allowance`), a verify step its first token's share there and the rest at settle.
        let (mut bank, mut earned_total, mut spent) = (0u64, 0u64, 0u64);
        let mut rng = crate::rng::Rng::new(11);
        for _ in 0..10_000 {
            let kept = 1 + rng.below(16);
            let (spend, left) = allowance(PolicyStep::Verify, 1, bank, CAP);
            spent += spend;
            earned_total += CAP * kept as u64;
            bank = bank_after_verify(left, kept, CAP);
            assert!(spent + bank <= earned_total, "kept {kept}");
            assert!(bank <= RESHAPE_BYTES);
        }
    }

    #[test]
    fn an_upload_cap_outside_one_to_the_reshape_is_refused_before_the_model_loads() {
        assert_eq!(cap_for(None).unwrap(), UPLOAD_CAP);
        assert_eq!(cap_for(Some(1)).unwrap(), 1);
        assert_eq!(cap_for(Some(RESHAPE_BYTES)).unwrap(), RESHAPE_BYTES);
        for bad in [0, RESHAPE_BYTES + 1, u64::MAX] {
            match cap_for(Some(bad)) {
                Err(crate::Error::BadUploadCap { cap, max }) => {
                    assert_eq!((cap, max), (bad, RESHAPE_BYTES))
                }
                other => panic!("{bad}: expected BadUploadCap, got {other:?}"),
            }
        }
    }

    #[test]
    fn the_policy_never_sees_the_mtp_block() {
        // Scores exist for layers 0..n_layer only: the MTP block's experts are HOST for good.
        let p = policy(&[5, 3], true, flat());
        let d = Dims::tiny();
        assert!(p.score(d.n_layer - 1, 0) > 0.0);
        // The loops stop at n_layer: no room is kept for the MTP block.
        assert_eq!(p.stored.len(), d.n_layer * d.n_expert);
        let beyond =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.score(d.n_layer, 0)));
        assert!(beyond.is_err());
    }
}
