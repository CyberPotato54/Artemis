//! The prefetch predictor's study (item 4): layer l + 1's router applied to the residual leaving layer l, through layer
//! l + 1's `ffn_norm`, scored against the routes layer l + 1 really took. `artemis route-predict` reads a tap folder
//! (`Layer(l, Out)` and `Layer(l, TopK)`) and the model's routers and norms, and prints how many of the HOST reads the
//! predicted top 8 or top 12 would have caught, and how many copies it would have wasted.
use crate::tapdir::TapSet;
use anyhow::{bail, Result};
use qwen36::{
    dims::Dims,
    model::Model,
    ops,
    taps::{Key, TapName},
};
use serde_json::{json, Value};

/// The experts layer l + 1's router ranks highest for layer l's residual `resid`, best first (ties go to the lower id).
/// `router` is `[n_expert][n_embd]` row-major, `ffn_norm` and `resid` are `n_embd` wide.
pub fn predict(router: &[f32], ffn_norm: &[f32], resid: &[f32], eps: f32, top: usize) -> Vec<u16> {
    let n_embd = resid.len();
    assert_eq!(ffn_norm.len(), n_embd, "ffn_norm is n_embd wide");
    assert!(
        n_embd > 0 && router.len().is_multiple_of(n_embd),
        "the router is whole rows of n_embd"
    );
    let x = ops::rmsnorm(resid, ffn_norm, eps);
    let mut scored: Vec<(f32, u16)> = router
        .chunks_exact(n_embd)
        .enumerate()
        .map(|(e, row)| {
            (
                row.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>(),
                e as u16,
            )
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(top).map(|(_, e)| e).collect()
}

/// One score. A "HOST read" is a route to an expert whose `host` flag is set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Score {
    /// HOST routes that the prediction lists.
    pub caught_host: u32,
    /// HOST routes in all.
    pub host: u32,
    /// Predicted HOST experts that were not routed: copies that take link time from real misses.
    pub wasted: u32,
}

impl std::ops::AddAssign for Score {
    fn add_assign(&mut self, o: Score) {
        self.caught_host += o.caught_host;
        self.host += o.host;
        self.wasted += o.wasted;
    }
}

/// Scores `pred` against the `truth` routes; `host[e]` says whether expert `e` is read from host memory.
pub fn score(pred: &[u16], truth: &[u16], host: &[bool]) -> Score {
    let is_host = |e: u16| host.get(e as usize).copied().unwrap_or(false);
    let mut s = Score::default();
    for &t in truth.iter().filter(|&&t| is_host(t)) {
        s.host += 1;
        if pred.contains(&t) {
            s.caught_host += 1;
        }
    }
    s.wasted = pred
        .iter()
        .filter(|&&p| is_host(p) && !truth.contains(&p))
        .count() as u32;
    s
}

/// The `n` most-routed experts of `routes` (all positions' ids for one layer), ties to the lower id: the stand-in for a
/// layer's resident set. Every other expert is HOST.
pub fn host_flags(routes: &[u16], n_expert: usize, n: usize) -> Vec<bool> {
    let mut count = vec![0u32; n_expert];
    for &r in routes {
        count[r as usize] += 1;
    }
    let mut order: Vec<usize> = (0..n_expert).collect();
    order.sort_by(|&a, &b| count[b].cmp(&count[a]).then(a.cmp(&b)));
    let mut host = vec![true; n_expert];
    for &e in order.iter().take(n) {
        host[e] = false;
    }
    host
}

/// The resident-row counts the study uses, and the predictor's `top` values.
pub const ROWS: [usize; 3] = [132, 85, 30];
pub const TOPS: [usize; 2] = [8, 12];

fn pct(a: u64, b: u64) -> f64 {
    if b == 0 {
        0.0
    } else {
        100.0 * a as f64 / b as f64
    }
}

/// Runs the study on a tap set and the model, and returns the JSON verdict: per (rows, top) the recall of all routes
/// and of HOST routes, and the wasted copies against the caught ones. The bar (spec, item 4): at least 70% of HOST
/// reads caught with wasted at most caught.
pub fn study(set: &TapSet, m: &Model, d: &Dims) -> Result<Value> {
    let get = |l: usize, k: Key| set.taps.get(&TapName::Layer(l as u16, k));
    let (k, e) = (d.n_expert_used, d.n_expert);
    let pairs: Vec<usize> = (0..d.n_layer.saturating_sub(1))
        .filter(|&l| get(l, Key::Out).is_some() && get(l + 1, Key::TopK).is_some())
        .collect();
    if pairs.is_empty() || set.n_tokens == 0 {
        bail!("the tap folder has no layer pair with Layer(l, Out) and Layer(l + 1, TopK)");
    }
    // The routes of layer l + 1 for each pair, and the model's router and norm for it.
    let routes: Vec<Vec<u16>> = pairs
        .iter()
        .map(|&l| {
            get(l + 1, Key::TopK)
                .expect("filtered above")
                .iter()
                .map(|&v| v as u16)
                .collect()
        })
        .collect();
    let weights: Vec<(Vec<f32>, Vec<f32>)> = pairs
        .iter()
        .map(|&l| {
            let next = &m.layers[l + 1];
            (next.moe.router.to_f32(), next.post_attention_norm.to_f32())
        })
        .collect();
    let everything = vec![true; e];
    let mut rows = Vec::new();
    for top in TOPS {
        let preds: Vec<Vec<Vec<u16>>> = pairs
            .iter()
            .zip(&weights)
            .map(|(&l, (router, norm))| {
                let out = get(l, Key::Out).expect("filtered above");
                (0..set.n_tokens)
                    .map(|t| {
                        let resid = &out[t * d.n_embd..(t + 1) * d.n_embd];
                        predict(router, norm, resid, d.rms_eps, top)
                    })
                    .collect()
            })
            .collect();
        for n in ROWS {
            let (mut all, mut host) = (Score::default(), Score::default());
            for (truth, p) in routes.iter().zip(&preds) {
                let flags = host_flags(truth, e, n);
                for (t, pred) in p.iter().enumerate() {
                    let tr = &truth[t * k..(t + 1) * k];
                    all += score(pred, tr, &everything);
                    host += score(pred, tr, &flags);
                }
            }
            let recall_host = pct(host.caught_host as u64, host.host as u64);
            rows.push(json!({
                "rows": n,
                "top": top,
                "layers": pairs.len(),
                "positions": set.n_tokens,
                "recall_all_pct": pct(all.caught_host as u64, all.host as u64),
                "recall_host_pct": recall_host,
                "host_routes": host.host,
                "caught_host": host.caught_host,
                "wasted": host.wasted,
                "meets_bar": recall_host >= 70.0 && host.wasted <= host.caught_host,
            }));
        }
    }
    Ok(json!({ "bar": "recall_host_pct >= 70 and wasted <= caught_host", "results": rows }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_matches_a_hand_computed_router() {
        // n_embd 2, 4 experts. resid (3, 4) has rms sqrt(12.5); ffn_norm (1, 2) doubles the second input, so the normed
        // input is (3, 8) / sqrt(12.5). The router rows score, before that common factor, e0 = 3, e1 = 8, e2 = -3 and
        // e3 = 3 + 8 = 11, so the top 2 is e3 then e1.
        let router = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0];
        let top = predict(&router, &[1.0, 2.0], &[3.0, 4.0], 1e-6, 2);
        assert_eq!(top, vec![3, 1]);
        // A tie goes to the lower id: rows 1 and 2 score the same, rows 0 and 3 are lower.
        let router = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0];
        assert_eq!(
            predict(&router, &[1.0, 1.0], &[1.0, 1.0], 1e-6, 2),
            vec![1, 2]
        );
    }

    #[test]
    fn score_counts_host_reads_caught_and_copies_wasted() {
        // Experts 0..6; 4 and 5 are resident (not HOST).
        let host = [true, true, true, true, false, false];
        // Truth routes 0, 1, 4: two HOST reads. The prediction lists 1 (caught), 4 (resident, so no copy), 2 and 3
        // (HOST, not routed: two wasted copies).
        let s = score(&[1, 4, 2, 3], &[0, 1, 4], &host);
        assert_eq!(
            s,
            Score {
                caught_host: 1,
                host: 2,
                wasted: 2
            }
        );
    }

    #[test]
    fn the_resident_set_is_the_most_routed_experts() {
        // Expert 2 routed 3 times, 0 twice, 1 once; with 2 rows resident, expert 1 and every unrouted one is HOST.
        let flags = host_flags(&[2, 2, 2, 0, 0, 1], 4, 2);
        assert_eq!(flags, vec![false, true, false, true]);
    }
}
