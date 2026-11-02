//! Test 2's pass rule. For every tap, the candidate's worst relative error against llama.cpp
//! CPU must be at most FLOOR_FACTOR × llama.cpp's own Vulkan-vs-CPU error + FLOOR_SLACK. Routed experts must match
//! exactly. An input whose CPU router has a near-tie is reported for replacement, never silently excused.
//! The GPU engine is checked by gpu_forced, teacher-forced against the reference with an f16 KV cache.
use crate::{
    compare::{compare, compare_over},
    tapdir::{Forcer, Recorder, Shifted, TapSet},
};
use anyhow::Result;
use qwen36::{
    dims::Dims,
    model::Model,
    reference::{MtpReference, Rounding},
    taps::{expected_mtp_taps, expected_taps, Key, MtpKey, Tap, TapName},
};
use serde::Serialize;

pub const FLOOR_FACTOR: f64 = 2.0;
pub const FLOOR_SLACK: f64 = 1e-5;
/// Router logits closer than this between the 8th and 9th expert make an input a near-tie.
pub const NEAR_TIE: f32 = 0.01;
/// The floor for every tap when llama.cpp's Vulkan backend has no dump (it refuses the tiny model).
pub const FIXED_FLOOR: f64 = 1e-2;

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    Pass,
    Fail,
    NearTie,
}

#[derive(Debug, Serialize)]
pub struct Row {
    pub tap: String,
    pub floor: f64,
    pub err: f64,
    pub bound: f64,
    pub worst_pos: usize,
    pub pass: bool,
}

#[derive(Debug, Serialize)]
pub struct Verdict {
    pub status: Status,
    pub rows: Vec<Row>,
    /// (layer, position) where the routed expert sets differ.
    pub topk_mismatches: Vec<(u16, usize)>,
    /// (layer, position, gap) where llama.cpp CPU's 8th and 9th router logits are within NEAR_TIE.
    pub near_ties: Vec<(u16, usize, f32)>,
    pub missing: Vec<String>,
}

pub fn test2(cpu: &TapSet, vulkan: Option<&TapSet>, cand: &TapSet, d: &Dims) -> Result<Verdict> {
    let (cpu, cand) = (by_expert_id(cpu, d), by_expert_id(cand, d));
    let vulkan = vulkan.map(|v| by_expert_id(v, d));
    let main = compare(&cand, &cpu, d)?;
    let floor = vulkan.as_ref().map(|v| compare(v, &cpu, d)).transpose()?;
    let rows: Vec<Row> = main
        .diffs
        .iter()
        .filter(|x| !matches!(x.tap, TapName::Layer(_, Key::TopK)))
        .map(|x| {
            let f = floor
                .as_ref()
                .and_then(|fl| fl.diffs.iter().find(|y| y.tap == x.tap))
                .map_or(FIXED_FLOOR, |y| y.max_rel);
            let bound = FLOOR_FACTOR * f + FLOOR_SLACK;
            Row {
                tap: x.tap.to_string(),
                floor: f,
                err: x.max_rel,
                bound,
                worst_pos: x.worst_pos,
                pass: x.max_rel <= bound,
            }
        })
        .collect();
    let topk_mismatches = topk_mismatches_of(&cpu, &cand, d);
    let near_ties: Vec<_> = router_gaps(&cpu, d)
        .into_iter()
        .filter(|g| g.2 < NEAR_TIE)
        .collect();
    let status = if !near_ties.is_empty() {
        Status::NearTie
    } else if rows.iter().all(|r| r.pass) && topk_mismatches.is_empty() && main.missing.is_empty() {
        Status::Pass
    } else {
        Status::Fail
    };
    Ok(Verdict {
        status,
        rows,
        topk_mismatches,
        near_ties,
        missing: main.missing,
    })
}

/// llama.cpp's CPU switches flash attention to its tiled kernel, which keeps queries and accumulators in f32, once a
/// batch has 64 query tokens or more (GGML_FA_TILE_Q at b11509). `Rounding::LlamaCpu` emulates the smaller-batch
/// kernel, so it only matches llama.cpp for batches below that.
pub const LLAMA_FA_TILE_Q: usize = 64;

/// Refuses an emulated run whose batch llama.cpp would send to the tiled flash-attention kernel.
pub fn check_llama_rounding_tokens(n: usize) -> Result<()> {
    if n >= LLAMA_FA_TILE_Q {
        anyhow::bail!(
            "{n} tokens: llama.cpp's CPU uses its tiled flash attention from {LLAMA_FA_TILE_Q} tokens, which \
             --llama-rounding doesn't emulate; use fewer than {LLAMA_FA_TILE_Q}"
        );
    }
    Ok(())
}

/// The per-operation bound for teacher-forced test 2 (decision of 2026-10-08). Forced, almost every
/// operation agrees with llama.cpp to about 1e-7, and the worst outside attention measured 2.3e-6 (`gdn.g`).
pub const FORCED_BOUND: f64 = 1e-5;
/// `attn.pregate`'s bound: llama.cpp's CPU flash attention accumulates V in f16, and its rounding flips measured up to
/// 6.9e-5 on the real model.
pub const FORCED_BOUND_ATTN_PREGATE: f64 = 1e-3;

/// Each tap's bound in forced test 2: the one place the rule lives.
pub fn forced_bound(tap: TapName) -> f64 {
    match tap {
        TapName::Layer(_, Key::AttnPregate) | TapName::Mtp(MtpKey::AttnPregate) => {
            FORCED_BOUND_ATTN_PREGATE
        }
        _ => FORCED_BOUND,
    }
}
/// In the forced rules, routed experts may differ only by experts whose baseline router logit is within this of the
/// last chosen one's: a true tie (`swapped_within_tie`).
pub const FORCED_TIE: f32 = 1e-4;

/// Test 2's gating rule. `cand` is the reference forced with `cpu`'s values at every tap, so each tap's difference is
/// one operation's error on llama.cpp's own inputs: every tap must be within its `forced_bound`, and the routed experts
/// must match except at true ties, which are reported in `near_ties`.
pub fn test2_forced(cpu: &TapSet, cand: &TapSet, d: &Dims) -> Result<Verdict> {
    forced_rule(cpu, cand, d, &expected_taps(d))
}

/// Test 2's MTP half: the same rule over the MTP block's taps. `oracle` is `artemis-oracle mtp-taps`'s
/// rows read by `tapdir::read_mtp`; `reference` is `artemis dump --cpu --mtp` forced from them, row by row. Each tap has
/// its main-layer operation's `forced_bound`; the routing taps take the same exact-routing rule, ties and TopK forcing
/// as the main layers (reported as layer `n_layer`, position = row).
pub fn test2_mtp(oracle: &TapSet, reference: &TapSet, d: &Dims) -> Result<Verdict> {
    forced_rule(oracle, reference, d, &expected_mtp_taps())
}

/// Refuses a reference directory whose recorded `source` isn't `dump --cpu --mtp --force-from … --llama-rounding`: such a
/// run fails `test2-mtp` as an operator slip, not as a semantics bug, so it is an error, not a verdict.
pub fn check_mtp_reference_source(source: &str) -> Result<()> {
    if !source.contains("MTP") {
        anyhow::bail!(
            "the reference was not dumped with `dump --cpu --mtp` (its source: {source:?})"
        );
    }
    if !source.contains("teacher-forced") {
        anyhow::bail!("the reference was dumped without --force-from, so its taps are free-running (source: {source:?})");
    }
    if !source.contains("LlamaCpu") {
        anyhow::bail!("the reference was dumped without --llama-rounding (source: {source:?})");
    }
    Ok(())
}

/// `artemis dump --cpu --mtp`: the reference's MTP over rows `q = 1 .. n − 1`, each pairing `tokens[q]` with
/// `hidden[q − 1]`, the main model's final hidden row at the position before (llama.cpp's, from `mtp-taps`'s
/// `hidden.bin`, so each row is teacher-forced at its input). With `force_from`, the oracle's MTP rows
/// (`tapdir::read_mtp`), every tap continues from llama.cpp's value too. The set has `n − 1` rows; row `i` is position
/// `i + 1`.
pub fn dump_mtp(
    m: &Model,
    tokens: &[u32],
    hidden: &[f32],
    force_from: Option<&TapSet>,
    threads: usize,
    rounding: Rounding,
) -> Result<TapSet> {
    let (n, e) = (tokens.len(), m.dims.n_embd);
    if n < 2 {
        anyhow::bail!(
            "the MTP needs at least 2 tokens: row q pairs token q with the hidden row at q − 1"
        );
    }
    if hidden.len() != n * e {
        anyhow::bail!(
            "{} hidden values for {n} tokens of {e} wide rows: hidden.bin and the token file disagree",
            hidden.len()
        );
    }
    if let Some(o) = force_from {
        if o.n_tokens != n - 1 {
            anyhow::bail!(
                "the forcing taps hold {} MTP rows, but {n} tokens make {}",
                o.n_tokens,
                n - 1
            );
        }
    }
    if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= m.dims.n_vocab) {
        anyhow::bail!(
            "token {bad} is outside the vocabulary ({} tokens)",
            m.dims.n_vocab
        );
    }
    let mut forcer = force_from.map(|o| Forcer::new(o, &m.dims));
    let mut rec = Recorder::default();
    let mut mtp = MtpReference::new(m, threads, rounding);
    for q in 1..n {
        let sink: &mut dyn Tap = match forcer.as_mut() {
            Some(f) => f,
            None => &mut rec,
        };
        let mut shifted = Shifted { inner: sink, by: 1 };
        mtp.row(
            q as u32,
            tokens[q],
            &hidden[(q - 1) * e..q * e],
            &mut shifted,
        );
    }
    Ok(forcer.map_or(rec.set, |f| f.rec.set))
}

fn forced_rule(cpu: &TapSet, cand: &TapSet, d: &Dims, taps: &[TapName]) -> Result<Verdict> {
    // Forced, the candidate's routing weights were computed for llama.cpp's experts (its TopK was forced), so they are
    // paired with llama.cpp's ids; the candidate's own ids are only compared as sets.
    let mut weights = cand.clone();
    for l in 0..=d.n_layer as u16 {
        if let Some(ids) = cpu.taps.get(&route(l, Key::TopK, d)) {
            weights.taps.insert(route(l, Key::TopK, d), ids.clone());
        }
    }
    let (cpu, own_ids, cand) = (
        by_expert_id(cpu, d),
        by_expert_id(cand, d),
        by_expert_id(&weights, d),
    );
    let main = compare_over(&cand, &cpu, d, taps)?;
    let rows: Vec<Row> = main
        .diffs
        .iter()
        .filter(|x| {
            !matches!(
                x.tap,
                TapName::Layer(_, Key::TopK) | TapName::Mtp(MtpKey::TopK)
            )
        })
        .map(|x| {
            let bound = forced_bound(x.tap);
            Row {
                tap: x.tap.to_string(),
                floor: 0.0,
                err: x.max_rel,
                bound,
                worst_pos: x.worst_pos,
                pass: x.max_rel <= bound,
            }
        })
        .collect();
    let gaps = router_gaps(&cpu, d);
    let (mut topk_mismatches, mut near_ties) = (Vec::new(), Vec::new());
    for (l, p) in topk_mismatches_of(&cpu, &own_ids, d) {
        if swapped_within_tie(&cpu, &own_ids, l, p, d) {
            near_ties.push((l, p, gap_at(&gaps, l, p)));
        } else {
            topk_mismatches.push((l, p));
        }
    }
    let pass = rows.iter().all(|r| r.pass) && topk_mismatches.is_empty() && main.missing.is_empty();
    let status = if pass { Status::Pass } else { Status::Fail };
    Ok(Verdict {
        status,
        rows,
        topk_mismatches,
        near_ties,
        missing: main.missing,
    })
}

/// Every matvec kernel states the same tolerance (pinned by a test); this one stands for them in `gpu_bound`.
const MATVEC: kernels::Kernel = kernels::Kernel::MatvecQ8_0;

/// The bound for one tap of the GPU engine, teacher-forced from the reference in `Rounding::F16Kv` mode, which stores
/// KV in f16 as the engine does. It is twice the summed stated tolerances of the
/// kernels that compute the tap from forced inputs: one factor bounds each kernel against its twin, the other
/// each twin against `qwen36::ops` (CI).
pub fn gpu_bound(tap: TapName) -> f64 {
    use kernels::Kernel::*;
    let chain: &[kernels::Kernel] = match tap {
        TapName::Embd => &[EmbedRow],
        TapName::FinalNorm => &[Rmsnorm],
        TapName::Logits => &[MATVEC],
        // The MTP block's taps: each has its main-layer operation's chain. `AttnGated` is oracle-assigned
        // (llama.cpp leaves the node unnamed), forced like the main layers', so `AttnOut` is `wo` alone.
        TapName::Mtp(key) => match key {
            MtpKey::Enorm
            | MtpKey::Hnorm
            | MtpKey::AttnNorm
            | MtpKey::PostNorm
            | MtpKey::HeadNorm => &[Rmsnorm],
            // Two halves joined: copies, exact.
            MtpKey::Concat => &[],
            MtpKey::EhProj | MtpKey::V | MtpKey::RouterLogits | MtpKey::Logits => &[MATVEC],
            MtpKey::Q | MtpKey::K => &[MATVEC, QkNormRope],
            MtpKey::AttnPregate => &[KvWrite, AttnDecode, AttnCombine],
            MtpKey::AttnGated => &[MATVEC, SigmoidMul],
            MtpKey::AttnOut => &[MATVEC],
            MtpKey::AttnResidual | MtpKey::FfnOut | MtpKey::PostFfn => &[Add],
            MtpKey::RouterProbs | MtpKey::TopK | MtpKey::TopKW => &[RouterTopk],
            MtpKey::MoeSwiglu => &[ExpertsGateUp],
            MtpKey::MoeOut => &[ExpertsDown, MoeCombine],
            MtpKey::ShexpSwiglu => &[MATVEC, MATVEC, Swiglu],
            MtpKey::ShexpGated => &[MATVEC, MATVEC, MoeCombine],
        },
        TapName::Layer(_, key) => match key {
            Key::AttnNorm | Key::FfnNorm => &[Rmsnorm],
            Key::GdnQkv | Key::GdnZ | Key::AttnV | Key::MixerOut | Key::RouterLogits => &[MATVEC],
            Key::GdnBeta | Key::GdnG => &[MATVEC, GdnGates],
            Key::GdnConv => &[ConvStep],
            Key::GdnQ | Key::GdnK => &[L2norm],
            // A slice of the forced conv output: exact.
            Key::GdnV => &[],
            Key::GdnO => &[GdnStep],
            Key::GdnNormed => &[GatedRmsnorm],
            Key::AttnQ | Key::AttnK => &[MATVEC, QkNormRope],
            Key::AttnPregate => &[KvWrite, AttnDecode, AttnCombine],
            // The gate comes from the Q projection, which no tap forces.
            Key::AttnGated => &[MATVEC, SigmoidMul],
            Key::Resid | Key::FfnOut | Key::Out => &[Add],
            Key::RouterProbs | Key::TopK | Key::TopKW => &[RouterTopk],
            Key::MoeSwiglu => &[ExpertsGateUp],
            Key::MoeOut => &[ExpertsDown, MoeCombine],
            Key::ShexpSwiglu => &[MATVEC, MATVEC, Swiglu],
            // The shared expert's down projection, its gate's projection, then the combine.
            Key::ShexpOut => &[MATVEC, MATVEC, MoeCombine],
        },
    };
    2.0 * chain.iter().map(|k| k.tol()).sum::<f64>()
}

/// The GPU engine's gating rule. `gpu` is the engine teacher-forced from `reference`, the taps of the
/// reference in `Rounding::F16Kv` mode, so each tap's difference is one GPU operation's error on the reference's
/// inputs: every tap must be within its `gpu_bound`. The engine's routed experts must equal the reference's, except
/// for a swap among experts tied with the reference's last chosen one (`swapped_within_tie`). Those ties are
/// reported, and there the engine's weights, which belong to the experts it chose, are compared as the reference's.
pub fn gpu_forced(reference: &TapSet, gpu: &TapSet, d: &Dims) -> Result<Verdict> {
    let (refs, mut cand) = (by_expert_id(reference, d), by_expert_id(gpu, d));
    let gaps = router_gaps(&refs, d);
    let k = d.n_expert_used;
    let (mut topk_mismatches, mut near_ties) = (Vec::new(), Vec::new());
    for (l, p) in topk_mismatches_of(&refs, &cand, d) {
        if swapped_within_tie(&refs, &cand, l, p, d) {
            near_ties.push((l, p, gap_at(&gaps, l, p)));
            for key in [Key::TopK, Key::TopKW] {
                let name = route(l, key, d);
                let want = refs.taps[&name][p * k..(p + 1) * k].to_vec();
                if let Some(c) = cand.taps.get_mut(&name) {
                    c[p * k..(p + 1) * k].copy_from_slice(&want);
                }
            }
        } else {
            topk_mismatches.push((l, p));
        }
    }
    let main = compare(&cand, &refs, d)?;
    let rows: Vec<Row> = main
        .diffs
        .iter()
        .filter(|x| !matches!(x.tap, TapName::Layer(_, Key::TopK)))
        .map(|x| {
            let bound = gpu_bound(x.tap);
            Row {
                tap: x.tap.to_string(),
                floor: 0.0,
                err: x.max_rel,
                bound,
                worst_pos: x.worst_pos,
                pass: x.max_rel <= bound,
            }
        })
        .collect();
    let pass = rows.iter().all(|r| r.pass) && topk_mismatches.is_empty() && main.missing.is_empty();
    Ok(Verdict {
        status: if pass { Status::Pass } else { Status::Fail },
        rows,
        topk_mismatches,
        near_ties,
        missing: main.missing,
    })
}

/// The tap of a routing key at "layer" `l`: the MTP block is layer `n_layer` (`Dims::n_layer`).
fn route(l: u16, key: Key, d: &Dims) -> TapName {
    if (l as usize) < d.n_layer {
        return TapName::Layer(l, key);
    }
    TapName::Mtp(match key {
        Key::TopK => MtpKey::TopK,
        Key::TopKW => MtpKey::TopKW,
        Key::RouterLogits => MtpKey::RouterLogits,
        _ => unreachable!("only routing keys have a site"),
    })
}

/// Reorders each position's routed experts, and their weights, by expert id, so weights compare expert by expert
/// whatever order each side listed them in.
fn by_expert_id(s: &TapSet, d: &Dims) -> TapSet {
    let mut out = s.clone();
    let k = d.n_expert_used;
    for l in 0..=d.n_layer as u16 {
        let (Some(ids), Some(ws)) = (
            s.taps.get(&route(l, Key::TopK, d)),
            s.taps.get(&route(l, Key::TopKW, d)),
        ) else {
            continue;
        };
        let (mut ni, mut nw) = (ids.clone(), ws.clone());
        for p in 0..s.n_tokens {
            let mut pairs: Vec<(f32, f32)> =
                (0..k).map(|j| (ids[p * k + j], ws[p * k + j])).collect();
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
            for (j, (i, w)) in pairs.into_iter().enumerate() {
                ni[p * k + j] = i;
                nw[p * k + j] = w;
            }
        }
        out.taps.insert(route(l, Key::TopK, d), ni);
        out.taps.insert(route(l, Key::TopKW, d), nw);
    }
    out
}

fn topk_mismatches_of(a: &TapSet, b: &TapSet, d: &Dims) -> Vec<(u16, usize)> {
    let k = d.n_expert_used;
    let mut out = Vec::new();
    for l in 0..=d.n_layer as u16 {
        let (Some(x), Some(y)) = (
            a.taps.get(&route(l, Key::TopK, d)),
            b.taps.get(&route(l, Key::TopK, d)),
        ) else {
            continue;
        };
        for p in 0..a.n_tokens {
            if x[p * k..(p + 1) * k] != y[p * k..(p + 1) * k] {
                out.push((l, p));
            }
        }
    }
    out
}

/// Whether `cand`'s routed experts at layer `l`, position `p` differ from `base`'s only by experts tied at the
/// boundary: every expert one of them chose and the other didn't has a `base` router logit within FORCED_TIE of the
/// `k`-th largest. A tie excuses a swap among the tied experts, never another routing error at the same place.
fn swapped_within_tie(base: &TapSet, cand: &TapSet, l: u16, p: usize, d: &Dims) -> bool {
    let (n, k) = (d.n_expert, d.n_expert_used);
    let Some(logits) = base.taps.get(&route(l, Key::RouterLogits, d)) else {
        return false;
    };
    let logits = &logits[p * n..(p + 1) * n];
    let mut sorted = logits.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let kth = sorted[k - 1];
    // `k` distinct whole ids below `n`, or nothing: a repeated or fractional id only drops an expert, never swaps one.
    let ids = |s: &TapSet| -> Option<Vec<usize>> {
        let v = s.taps.get(&route(l, Key::TopK, d))?;
        let ids: Vec<usize> = v[p * k..(p + 1) * k]
            .iter()
            .map(|&e| (e >= 0.0 && e.fract() == 0.0 && e < n as f32).then_some(e as usize))
            .collect::<Option<_>>()?;
        let mut distinct = ids.clone();
        distinct.sort_unstable();
        distinct.dedup();
        (distinct.len() == k).then_some(ids)
    };
    let (Some(a), Some(b)) = (ids(base), ids(cand)) else {
        return false;
    };
    a.iter()
        .filter(|e| !b.contains(e))
        .chain(b.iter().filter(|e| !a.contains(e)))
        .all(|&e| (logits[e] - kth).abs() < FORCED_TIE)
}

/// The gap between the k-th and (k+1)-th router logit at (l, p), for reporting.
fn gap_at(gaps: &[(u16, usize, f32)], l: u16, p: usize) -> f32 {
    gaps.iter()
        .find(|g| g.0 == l && g.1 == p)
        .map_or(f32::NAN, |g| g.2)
}

/// (layer, position, gap) for every position: the gap between llama.cpp's last chosen and first unchosen router logit.
fn router_gaps(cpu: &TapSet, d: &Dims) -> Vec<(u16, usize, f32)> {
    let (n, k) = (d.n_expert, d.n_expert_used);
    let mut out = Vec::new();
    for l in 0..=d.n_layer as u16 {
        let Some(logits) = cpu.taps.get(&route(l, Key::RouterLogits, d)) else {
            continue;
        };
        for p in 0..cpu.n_tokens {
            let mut v = logits[p * n..(p + 1) * n].to_vec();
            v.sort_by(|a, b| b.total_cmp(a));
            out.push((l, p, v[k - 1] - v[k]));
        }
    }
    out
}

/// The row closest to, or furthest past, its bound: the CLI's summary line.
pub fn worst(v: &Verdict) -> Option<&Row> {
    let ratio = |r: &Row| {
        if r.bound > 0.0 {
            r.err / r.bound
        } else if r.err == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    };
    v.rows.iter().max_by(|a, b| ratio(a).total_cmp(&ratio(b)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen36::{taps::expected_taps, testing::Rng};

    /// A full tap set for the tiny model with distinct routed experts and clear router gaps.
    fn base(seed: u64) -> TapSet {
        let d = Dims::tiny();
        let mut rng = Rng::new(seed);
        let n_tokens = 3;
        let mut set = TapSet {
            n_tokens,
            ..Default::default()
        };
        for t in expected_taps(&d) {
            let n = t.per_pos(&d) * n_tokens;
            let data: Vec<f32> = match t {
                TapName::Layer(_, Key::TopK) => (0..n_tokens)
                    .flat_map(|p| (0..4).map(move |j| ((p + 3 * j) % 16) as f32))
                    .collect(),
                TapName::Layer(_, Key::TopKW) => vec![0.25; n],
                // Logits 0, 1, …, 15 per position: every gap is 1.
                TapName::Layer(_, Key::RouterLogits) => (0..n).map(|i| (i % 16) as f32).collect(),
                _ => (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect(),
            };
            set.taps.insert(t, data);
        }
        set
    }

    /// `s` with every value except routing ids nudged by a relative `eps`.
    fn nudged(s: &TapSet, eps: f32) -> TapSet {
        let mut out = s.clone();
        for (t, v) in out.taps.iter_mut() {
            if !matches!(t, TapName::Layer(_, Key::TopK)) {
                for (i, x) in v.iter_mut().enumerate() {
                    *x *= 1.0 + if i % 2 == 0 { eps } else { -eps };
                }
            }
        }
        out
    }

    #[test]
    fn llama_rounding_is_refused_from_64_tokens() {
        assert!(check_llama_rounding_tokens(63).is_ok());
        let e = check_llama_rounding_tokens(64).unwrap_err().to_string();
        assert!(e.contains("64") && e.contains("tiled"), "{e}");
    }

    /// `s` with one tap scaled by `1 + eps`: a small constant error in one operation.
    fn scaled(s: &TapSet, tap: TapName, eps: f32) -> TapSet {
        let mut out = s.clone();
        for x in out.taps.get_mut(&tap).unwrap() {
            *x *= 1.0 + eps;
        }
        out
    }

    #[test]
    fn forced_bound_is_loose_only_for_attention_pregate() {
        assert_eq!(forced_bound(TapName::Layer(39, Key::AttnPregate)), 1e-3);
        for t in [
            TapName::Layer(4, Key::GdnG),
            TapName::Layer(3, Key::AttnGated),
            TapName::Layer(3, Key::AttnQ),
            TapName::Embd,
            TapName::Logits,
        ] {
            assert_eq!(forced_bound(t), 1e-5, "{t}");
        }
    }

    #[test]
    fn forced_within_the_per_op_bound_passes() {
        let cpu = base(1);
        let v = test2_forced(&cpu, &nudged(&cpu, 5e-6), &Dims::tiny()).unwrap();
        assert_eq!(
            v.status,
            Status::Pass,
            "{:?}",
            v.rows.iter().find(|r| !r.pass)
        );
        for r in &v.rows {
            assert_eq!(r.bound, forced_bound(r.tap.parse().unwrap()), "{}", r.tap);
        }
    }

    #[test]
    fn forced_beyond_the_per_op_bound_fails() {
        let cpu = base(1);
        let v = test2_forced(&cpu, &nudged(&cpu, 2e-3), &Dims::tiny()).unwrap();
        assert_eq!(v.status, Status::Fail);
    }

    #[test]
    fn a_small_constant_error_in_one_operation_fails() {
        // The case the tight bound exists for, such as a misplaced eps: 2e-5 in one DeltaNet gate.
        let cpu = base(1);
        let cand = scaled(&cpu, TapName::Layer(0, Key::GdnG), 2e-5);
        let v = test2_forced(&cpu, &cand, &Dims::tiny()).unwrap();
        assert_eq!(v.status, Status::Fail);
        let failing: Vec<&str> = v
            .rows
            .iter()
            .filter(|r| !r.pass)
            .map(|r| r.tap.as_str())
            .collect();
        assert_eq!(failing, ["l0.gdn.g"]);
    }

    #[test]
    fn attention_pregate_keeps_the_loose_bound() {
        // f16 rounding flips in llama.cpp's flash-attention accumulator measured up to 6.9e-5 on the real model.
        let cpu = base(1);
        let cand = scaled(&cpu, TapName::Layer(3, Key::AttnPregate), 5e-4);
        let v = test2_forced(&cpu, &cand, &Dims::tiny()).unwrap();
        assert_eq!(
            v.status,
            Status::Pass,
            "{:?}",
            v.rows.iter().find(|r| !r.pass)
        );
    }

    #[test]
    fn forced_routing_differences_are_excused_only_for_tied_experts() {
        let mut cpu = base(1);
        // Distinct weights, so pairing a weight with the wrong expert shows. Forced, the candidate's weights belong to
        // llama.cpp's experts, whatever experts it picked itself.
        for l in 0..4 {
            let w = cpu.taps.get_mut(&TapName::Layer(l, Key::TopKW)).unwrap();
            for (i, x) in w.iter_mut().enumerate() {
                *x = [0.4, 0.3, 0.2, 0.1][i % 4];
            }
        }
        // Position 1 of layer 2 routes its four largest logits (0, 1, …, 15): experts 15, 14, 13 and 12.
        cpu.taps.get_mut(&TapName::Layer(2, Key::TopK)).unwrap()[4..8]
            .copy_from_slice(&[15.0, 14.0, 13.0, 12.0]);
        let mut cand = cpu.clone();
        // The candidate routes expert 11 instead of 12.
        cand.taps.get_mut(&TapName::Layer(2, Key::TopK)).unwrap()[7] = 11.0;
        let v = test2_forced(&cpu, &cand, &Dims::tiny()).unwrap();
        assert_eq!(
            (v.status, v.topk_mismatches.clone()),
            (Status::Fail, vec![(2, 1)])
        );
        // Make llama.cpp's own 4th and 5th logits there, experts 12 and 11, a true tie (gap 5e-5): now it is excused
        // and reported. Forced, the candidate saw the same logits.
        for set in [&mut cpu, &mut cand] {
            set.taps
                .get_mut(&TapName::Layer(2, Key::RouterLogits))
                .unwrap()[16 + 11] = 11.99995;
        }
        let v = test2_forced(&cpu, &cand, &Dims::tiny()).unwrap();
        assert_eq!(v.status, Status::Pass);
        assert!(v.topk_mismatches.is_empty());
        assert_eq!((v.near_ties[0].0, v.near_ties[0].1), (2, 1));
        // A tie excuses only a swap among the tied experts: expert 3 instead of 12 is a real routing error.
        cand.taps.get_mut(&TapName::Layer(2, Key::TopK)).unwrap()[7] = 3.0;
        let v = test2_forced(&cpu, &cand, &Dims::tiny()).unwrap();
        assert_eq!(
            (v.status, v.topk_mismatches.clone()),
            (Status::Fail, vec![(2, 1)])
        );
        // Nor a repeated or fractional id, which only drops an expert: at a position with no tie (layer 1, gap 1)
        // they are routing errors.
        cpu.taps.get_mut(&TapName::Layer(1, Key::TopK)).unwrap()[4..8]
            .copy_from_slice(&[15.0, 14.0, 13.0, 12.0]);
        for bad in [13.0, 12.5] {
            let mut cand = cpu.clone();
            cand.taps.get_mut(&TapName::Layer(1, Key::TopK)).unwrap()[7] = bad;
            let v = test2_forced(&cpu, &cand, &Dims::tiny()).unwrap();
            assert_eq!(
                (v.status, v.topk_mismatches.clone()),
                (Status::Fail, vec![(1, 1)]),
                "{bad}"
            );
        }
    }

    #[test]
    fn forced_missing_tap_fails() {
        let cpu = base(1);
        let mut cand = cpu.clone();
        cand.taps.remove(&TapName::Layer(3, Key::AttnQ));
        let v = test2_forced(&cpu, &cand, &Dims::tiny()).unwrap();
        assert_eq!(v.status, Status::Fail);
        assert_eq!(v.missing.len(), 1);
    }

    #[test]
    fn close_enough_passes() {
        let cpu = base(1);
        let v = test2(
            &cpu,
            Some(&nudged(&cpu, 1e-3)),
            &nudged(&cpu, 1.5e-3),
            &Dims::tiny(),
        )
        .unwrap();
        assert_eq!(
            v.status,
            Status::Pass,
            "{:?}",
            v.rows.iter().find(|r| !r.pass)
        );
    }

    #[test]
    fn beyond_twice_the_floor_fails() {
        let cpu = base(1);
        let v = test2(
            &cpu,
            Some(&nudged(&cpu, 1e-3)),
            &nudged(&cpu, 3e-3),
            &Dims::tiny(),
        )
        .unwrap();
        assert_eq!(v.status, Status::Fail);
        assert!(v.rows.iter().any(|r| !r.pass));
    }

    #[test]
    fn routed_experts_must_match() {
        let cpu = base(1);
        let mut cand = cpu.clone();
        cand.taps.get_mut(&TapName::Layer(2, Key::TopK)).unwrap()[5] = 15.0;
        let v = test2(&cpu, Some(&cpu), &cand, &Dims::tiny()).unwrap();
        assert_eq!(v.status, Status::Fail);
        assert_eq!(v.topk_mismatches, vec![(2, 1)]);
    }

    #[test]
    fn the_same_experts_in_another_order_pass() {
        let cpu = base(1);
        let mut cand = cpu.clone();
        for key in [Key::TopK, Key::TopKW] {
            let v = cand.taps.get_mut(&TapName::Layer(1, key)).unwrap();
            v.swap(0, 3);
        }
        let v = test2(&cpu, Some(&cpu), &cand, &Dims::tiny()).unwrap();
        assert_eq!(v.status, Status::Pass);
    }

    #[test]
    fn a_router_near_tie_is_reported() {
        let mut cpu = base(1);
        // Make the 4th and 5th largest logits of layer 0, position 2 (12 and 11), nearly equal.
        let l = cpu
            .taps
            .get_mut(&TapName::Layer(0, Key::RouterLogits))
            .unwrap();
        l[2 * 16 + 11] = 11.995;
        let v = test2(&cpu, Some(&cpu), &cpu, &Dims::tiny()).unwrap();
        assert_eq!(v.status, Status::NearTie);
        assert_eq!(v.near_ties.len(), 1);
        assert_eq!((v.near_ties[0].0, v.near_ties[0].1), (0, 2));
    }

    #[test]
    fn nan_fails() {
        let cpu = base(1);
        let mut cand = cpu.clone();
        cand.taps.get_mut(&TapName::Layer(3, Key::Out)).unwrap()[100] = f32::NAN;
        let v = test2(&cpu, Some(&cpu), &cand, &Dims::tiny()).unwrap();
        assert_eq!(v.status, Status::Fail);
        assert!(v.rows.iter().any(|r| r.tap == "l3.out" && !r.pass));
    }

    #[test]
    fn missing_tap_fails() {
        let cpu = base(1);
        let mut cand = cpu.clone();
        cand.taps.remove(&TapName::Layer(0, Key::GdnO));
        let v = test2(&cpu, Some(&cpu), &cand, &Dims::tiny()).unwrap();
        assert_eq!(v.status, Status::Fail);
        assert_eq!(
            v.missing,
            vec!["l0.gdn.o is missing from the first set".to_string()]
        );
    }

    #[test]
    fn different_lengths_are_an_error() {
        let cpu = base(1);
        let mut cand = cpu.clone();
        cand.n_tokens = 2;
        let e = test2(&cpu, None, &cand, &Dims::tiny())
            .unwrap_err()
            .to_string();
        assert!(e.contains("2 and 3 positions"), "{e}");
    }

    #[test]
    fn without_vulkan_the_fixed_floor_applies() {
        let cpu = base(1);
        let v = test2(&cpu, None, &nudged(&cpu, 5e-3), &Dims::tiny()).unwrap();
        assert_eq!(v.status, Status::Pass);
        assert!(v.rows.iter().all(|r| r.floor == FIXED_FLOOR));
    }

    use qwen36::quant::WType;

    #[test]
    fn gpu_bounds_follow_the_kernels_stated_tolerances() {
        use kernels::{tol, Kernel};
        for t in WType::ALL {
            assert_eq!(
                Kernel::matvec(t).tol(),
                tol::REDUCTION,
                "every matvec states one tolerance"
            );
        }
        let (e, r) = (tol::ELEMENTWISE, tol::REDUCTION);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-15;
        assert!(close(gpu_bound(TapName::Layer(0, Key::GdnQkv)), 2.0 * r));
        assert!(close(
            gpu_bound(TapName::Layer(0, Key::GdnG)),
            2.0 * (r + e)
        ));
        assert!(close(gpu_bound(TapName::Layer(0, Key::MoeOut)), 4.0 * r));
        assert!(close(gpu_bound(TapName::Layer(0, Key::Out)), 2.0 * e));
        // The reference stores KV in f16 as the engine does (`Rounding::F16Kv`), so attention gets no extra allowance.
        assert!(close(
            gpu_bound(TapName::Layer(3, Key::AttnPregate)),
            2.0 * (e + 2.0 * r)
        ));
        assert_eq!(
            gpu_bound(TapName::Layer(0, Key::GdnV)),
            0.0,
            "a slice of the forced conv output"
        );
        assert!(close(gpu_bound(TapName::Logits), 2.0 * r));
    }

    /// `s` with one tap scaled by `1 + eps`.
    fn scaled_tap(s: &TapSet, tap: TapName, eps: f32) -> TapSet {
        let mut out = s.clone();
        for x in out.taps.get_mut(&tap).unwrap() {
            *x *= 1.0 + eps;
        }
        out
    }

    /// `s` nudged by `eps` everywhere except routing ids and `gdn.v`, whose GPU bound is 0.
    fn nudged_gpu(s: &TapSet, eps: f32) -> TapSet {
        let mut out = nudged(s, eps);
        for (t, v) in s.taps.iter() {
            if matches!(t, TapName::Layer(_, Key::GdnV)) {
                out.taps.insert(*t, v.clone());
            }
        }
        out
    }

    #[test]
    fn gpu_forced_within_its_bounds_passes() {
        let refs = base(1);
        let v = gpu_forced(&refs, &nudged_gpu(&refs, 1e-6), &Dims::tiny()).unwrap();
        assert_eq!(
            v.status,
            Status::Pass,
            "{:?}",
            v.rows.iter().find(|r| !r.pass)
        );
        for r in &v.rows {
            assert_eq!(r.bound, gpu_bound(r.tap.parse().unwrap()), "{}", r.tap);
        }
    }

    #[test]
    fn gpu_forced_one_wrong_operation_fails_alone() {
        let refs = base(1);
        let v = gpu_forced(
            &refs,
            &scaled_tap(&refs, TapName::Layer(0, Key::GdnO), 1e-4),
            &Dims::tiny(),
        )
        .unwrap();
        assert_eq!(v.status, Status::Fail);
        let failing: Vec<&str> = v
            .rows
            .iter()
            .filter(|r| !r.pass)
            .map(|r| r.tap.as_str())
            .collect();
        assert_eq!(failing, ["l0.gdn.o"]);
        // gdn.v must match exactly.
        let v = gpu_forced(
            &refs,
            &scaled_tap(&refs, TapName::Layer(1, Key::GdnV), 1e-7),
            &Dims::tiny(),
        )
        .unwrap();
        assert_eq!(v.status, Status::Fail);
    }

    #[test]
    fn gpu_forced_excuses_tied_experts_and_compares_the_references_weights_there() {
        let mut refs = base(1);
        // Position 1 of layer 2 routes its four largest logits (0, 1, …, 15): experts 15, 14, 13 and 12.
        refs.taps.get_mut(&TapName::Layer(2, Key::TopK)).unwrap()[4..8]
            .copy_from_slice(&[15.0, 14.0, 13.0, 12.0]);
        let mut gpu = refs.clone();
        // The GPU routes expert 11 instead of 12, with weights of its own.
        gpu.taps.get_mut(&TapName::Layer(2, Key::TopK)).unwrap()[7] = 11.0;
        gpu.taps.get_mut(&TapName::Layer(2, Key::TopKW)).unwrap()[4..8]
            .copy_from_slice(&[0.4, 0.3, 0.2, 0.1]);
        let v = gpu_forced(&refs, &gpu, &Dims::tiny()).unwrap();
        assert_eq!(
            (v.status, v.topk_mismatches.clone()),
            (Status::Fail, vec![(2, 1)])
        );
        // The reference's own 4th and 5th logits there, experts 12 and 11, are a true tie (gap 5e-5): excused,
        // reported, and the GPU's weights for other experts aren't held against it. Forced, the GPU computed the same
        // logits.
        for set in [&mut refs, &mut gpu] {
            set.taps
                .get_mut(&TapName::Layer(2, Key::RouterLogits))
                .unwrap()[16 + 11] = 11.99995;
        }
        let v = gpu_forced(&refs, &gpu, &Dims::tiny()).unwrap();
        assert_eq!(
            v.status,
            Status::Pass,
            "{:?}",
            v.rows.iter().find(|r| !r.pass)
        );
        assert_eq!((v.near_ties[0].0, v.near_ties[0].1), (2, 1));
        // A tie excuses only a swap among the tied experts: expert 3 instead of 12 is a real routing error.
        gpu.taps.get_mut(&TapName::Layer(2, Key::TopK)).unwrap()[7] = 3.0;
        let v = gpu_forced(&refs, &gpu, &Dims::tiny()).unwrap();
        assert_eq!(
            (v.status, v.topk_mismatches.clone()),
            (Status::Fail, vec![(2, 1)])
        );
        // Nor a repeated or fractional id, which only drops an expert: at a position with no tie (layer 1, gap 1)
        // they are routing errors.
        refs.taps.get_mut(&TapName::Layer(1, Key::TopK)).unwrap()[4..8]
            .copy_from_slice(&[15.0, 14.0, 13.0, 12.0]);
        for bad in [13.0, 12.5] {
            let mut gpu = refs.clone();
            gpu.taps.get_mut(&TapName::Layer(1, Key::TopK)).unwrap()[7] = bad;
            let v = gpu_forced(&refs, &gpu, &Dims::tiny()).unwrap();
            assert_eq!(
                (v.status, v.topk_mismatches.clone()),
                (Status::Fail, vec![(1, 1)]),
                "{bad}"
            );
        }
    }

    #[test]
    fn gpu_forced_missing_tap_fails() {
        let refs = base(1);
        let mut gpu = refs.clone();
        gpu.taps.remove(&TapName::Layer(3, Key::AttnPregate));
        let v = gpu_forced(&refs, &gpu, &Dims::tiny()).unwrap();
        assert_eq!((v.status, v.missing.len()), (Status::Fail, 1));
    }
}
