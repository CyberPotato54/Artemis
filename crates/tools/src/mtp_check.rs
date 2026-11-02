//! `artemis mtp-check`: the GPU's MTP draft against llama.cpp's, on the real model.
//!
//! The oracle (`artemis-oracle mtp-taps`) ran the MTP over a token sequence, row `q` pairing token `q` with the main
//! model's final hidden row at `q - 1`, and wrote each row's `result_output` (the MTP head's logits). This command
//! decodes the same tokens one at a time on the engine and, at each position `q >= 1`, asks the session for the MTP's
//! logits for token `q` (`Session::mtp_draft_logits`), which it takes from the hidden row its own decode left in the
//! ring. Pass: the top-1 ids are equal wherever the oracle's top two logits are more than `TIE_MARGIN` apart (test 3's
//! rule). Each position's relative L2 error, and the worst, are reported and never gate: the GPU's kernels differ from
//! llama.cpp's CPU ones by rounding, and the top-1 rule is what a draft needs.
//!
//! The record names counts, margins and errors; no token or text goes into it.
use crate::{
    generate::{read_tokens, top_two},
    tapdir,
    test3::{Status, TIE_MARGIN},
};
use anyhow::{bail, ensure, Context, Result};
use engine::{Engine, Options, Session};
use gguf::Gguf;
use qwen36::{dims::Dims, taps::MtpKey, taps::TapName};
use serde::Serialize;
use std::{fs, path::Path};

/// The context the check opens its session with: it only has to hold the tokens and the MTP (at least 512).
pub const CTX: usize = 4096;

/// One position's comparison.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Row {
    /// The position `q` of the row.
    pub pos: usize,
    pub top1_equal: bool,
    /// The oracle's top logit minus its second.
    pub margin: f32,
    /// Whether the oracle's gap is at most `TIE_MARGIN`: a difference here is excused.
    pub near_tie: bool,
    /// `‖ours − oracle‖ / ‖oracle‖`; null when a value is not finite.
    pub rel_l2: Option<f64>,
    /// Whether the id the GPU's own argmax (`argmax_rows`) drafted is the argmax of the logits it returned. Report
    /// only: it checks that kernel on the real model.
    pub draft_id_is_argmax: bool,
}

/// The verdict, as the runner reads it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Verdict {
    pub status: Status,
    pub tie_margin: f32,
    pub positions: usize,
    /// Positions whose top-1 ids are equal.
    pub top1_equal: usize,
    /// Positions whose top-1 ids differ at a near tie (excused).
    pub near_tie_differences: usize,
    /// Positions where the GPU's drafted id is not the argmax of its returned logits; report only, expected 0.
    pub draft_id_mismatches: usize,
    /// Positions whose top-1 ids differ with the oracle's top two more than `TIE_MARGIN` apart: the failures.
    pub outside_near_tie: usize,
    /// The largest relative L2 over the positions; report only.
    pub worst_rel_l2: Option<f64>,
    pub rows: Vec<Row>,
    pub problems: Vec<String>,
}

/// `‖a − b‖ / ‖b‖` in f64, `None` when either holds a non-finite value or `b` is all zeros.
pub fn rel_l2(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.iter().chain(b).any(|v| !v.is_finite()) {
        return None;
    }
    let (mut num, mut den) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = f64::from(x) - f64::from(y);
        num += d * d;
        den += f64::from(y) * f64::from(y);
    }
    (den > 0.0).then(|| (num / den).sqrt())
}

/// Compares the engine's logits with the oracle's at position `pos`.
pub fn compare_row(pos: usize, ours: &[f32], oracle: &[f32]) -> Row {
    let (o_top, o_logit, _, o_second) = top_two(oracle);
    let (top, _, _, _) = top_two(ours);
    let margin = o_logit - o_second;
    Row {
        pos,
        top1_equal: top == o_top,
        margin,
        // A margin that is not a number is no near tie (the failure fails safe).
        near_tie: margin <= TIE_MARGIN,
        rel_l2: rel_l2(ours, oracle),
        draft_id_is_argmax: true,
    }
}

/// Pass iff no position's top-1 ids differ with the oracle's top two more than `TIE_MARGIN` apart, there is at least
/// one position, and every logit the engine gave is finite.
pub fn judge(rows: Vec<Row>) -> Verdict {
    let positions = rows.len();
    let top1_equal = rows.iter().filter(|r| r.top1_equal).count();
    let near_tie_differences = rows.iter().filter(|r| !r.top1_equal && r.near_tie).count();
    let draft_id_mismatches = rows.iter().filter(|r| !r.draft_id_is_argmax).count();
    let outside_near_tie = rows.iter().filter(|r| !r.top1_equal && !r.near_tie).count();
    let worst_rel_l2 = rows
        .iter()
        .filter_map(|r| r.rel_l2)
        .fold(None, |w: Option<f64>, v| Some(w.map_or(v, |w| w.max(v))));
    let mut problems = Vec::new();
    if rows.is_empty() {
        problems.push("no position was compared".to_string());
    }
    let unreadable = rows.iter().filter(|r| r.rel_l2.is_none()).count();
    if unreadable > 0 {
        problems.push(format!(
            "{unreadable} positions have a non-finite logit or an all-zero oracle row, so they have no error figure"
        ));
    }
    let pass = outside_near_tie == 0 && !rows.is_empty() && unreadable == 0;
    Verdict {
        status: if pass { Status::Pass } else { Status::Fail },
        tie_margin: TIE_MARGIN,
        positions,
        top1_equal,
        near_tie_differences,
        draft_id_mismatches,
        outside_near_tie,
        worst_rel_l2,
        rows,
        problems,
    }
}

/// The engine's side: what the MTP gives for token `q`, for each `q >= 1`.
trait Drafter {
    /// Feeds token 0 (a one-token prefill), the first position the MTP's hidden row comes from.
    fn start(&mut self, token: u32) -> Result<()>;
    /// The id the MTP drafts for `x`, the token at the session's next position, and the logits it drafted from.
    fn logits(&mut self, x: u32) -> Result<(u32, Vec<f32>)>;
    /// Decodes `x`, leaving the hidden row before the next token in the ring.
    fn decode(&mut self, x: u32) -> Result<()>;
}

struct Gpu<'a> {
    e: &'a Engine,
    s: Session,
}

impl Drafter for Gpu<'_> {
    fn start(&mut self, token: u32) -> Result<()> {
        self.s.prefill(self.e, &[token])?;
        Ok(())
    }

    fn logits(&mut self, x: u32) -> Result<(u32, Vec<f32>)> {
        Ok(self.s.mtp_draft_logits(self.e, x)?)
    }

    fn decode(&mut self, x: u32) -> Result<()> {
        self.s.decode(self.e, x)?;
        Ok(())
    }
}

/// Decodes `tokens` one at a time and compares the MTP's logits at each position `q >= 1` with `oracle`'s row `q - 1`
/// (`n_vocab` logits each, `tokens.len() - 1` rows).
fn check(d: &mut impl Drafter, tokens: &[u32], oracle: &[f32], n_vocab: usize) -> Result<Vec<Row>> {
    let n = tokens.len();
    ensure!(n >= 2, "the check needs at least 2 tokens");
    ensure!(
        oracle.len() == (n - 1) * n_vocab,
        "the oracle holds {} logits, but {n} tokens make {} rows of {n_vocab}",
        oracle.len(),
        n - 1
    );
    d.start(tokens[0])?;
    let mut rows = Vec::with_capacity(n - 1);
    for q in 1..n {
        let (id, ours) = d.logits(tokens[q])?;
        ensure!(
            ours.len() == n_vocab,
            "position {q}: the engine gave {} logits, not {n_vocab}",
            ours.len()
        );
        let mut row = compare_row(q, &ours, &oracle[(q - 1) * n_vocab..q * n_vocab]);
        row.draft_id_is_argmax = id == top_two(&ours).0;
        rows.push(row);
        // The last token is only compared, never decoded.
        if q + 1 < n {
            d.decode(tokens[q])?;
        }
    }
    Ok(rows)
}

/// `artemis mtp-check`. `tokens` is the file given to `artemis-oracle mtp-taps --tokens`; `oracle` its `--out`.
/// Writes the verdict to `json` and returns the process's exit code: 1 unless it passes.
pub fn run(
    model: &Path,
    oracle: &Path,
    tokens: &Path,
    json: &Path,
    ctx: usize,
    device_uuid: Option<[u8; 16]>,
) -> Result<i32> {
    // Everything that needs no GPU is read and checked before the model loads.
    let d = Dims::QWEN36;
    let toks = read_tokens(tokens)?;
    ensure!(
        toks.len() >= 2,
        "{} holds fewer than 2 tokens",
        tokens.display()
    );
    if let Some(&bad) = toks.iter().find(|&&t| t as usize >= d.n_vocab) {
        bail!(
            "token {bad} is outside the vocabulary ({} tokens)",
            d.n_vocab
        );
    }
    ensure!(
        toks.len() <= ctx,
        "{} tokens are longer than --ctx {ctx}",
        toks.len()
    );
    let set = tapdir::read_mtp(oracle, &d)?;
    ensure!(
        set.n_tokens == toks.len() - 1,
        "{} holds {} MTP rows, but {} tokens make {}: --tokens is not the file the oracle ran",
        oracle.display(),
        set.n_tokens,
        toks.len(),
        toks.len() - 1
    );
    let logits = set
        .taps
        .get(&TapName::Mtp(MtpKey::Logits))
        .with_context(|| format!("{} has no MTP result_output", oracle.display()))?;
    let out = fs::File::create(json)
        .with_context(|| format!("--json: can't write {}", json.display()))?;

    let g = Gguf::open(model).with_context(|| format!("opening {}", model.display()))?;
    let e = Engine::open(&g, &d, &options(device_uuid, ctx)?)?;
    let s = e.session(ctx)?;
    eprintln!(
        "mtp-check: loaded on {}; {} tokens, {} MTP rows",
        e.device_name(),
        toks.len(),
        toks.len() - 1
    );
    let rows = check(&mut Gpu { e: &e, s }, &toks, logits, d.n_vocab)?;
    let v = judge(rows);
    serde_json::to_writer_pretty(out, &v).with_context(|| format!("writing {}", json.display()))?;
    println!(
        "mtp-check: {:?}: {} positions, {} top-1 equal, {} near-tie differences, {} outside a near tie; worst relative L2 {}",
        v.status,
        v.positions,
        v.top1_equal,
        v.near_tie_differences,
        v.outside_near_tie,
        v.worst_rel_l2.map_or("n/a".to_string(), |w| format!("{w:.3e}"))
    );
    for p in &v.problems {
        println!("mtp-check: problem: {p}");
    }
    Ok(i32::from(v.status != Status::Pass))
}

/// The engine's options for the check. The oracle's logits cover the whole vocabulary, so the MTP drafts with the full
/// head (`mtp_head_rows` 0), whatever the default is.
fn options(device_uuid: Option<[u8; 16]>, ctx: usize) -> Result<Options> {
    let cfg = engine::SpecConfig {
        mtp_head_rows: 0,
        ..engine::SpecConfig::default()
    };
    Ok(Options {
        device_uuid,
        ctx,
        speculation: true,
        mtp: crate::generate::mtp_options(Some(&cfg), ctx)?,
        ..Options::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_check_loads_the_full_mtp_head() {
        let o = options(None, 4096).unwrap();
        assert_eq!(o.mtp.map(|m| m.head_rows), Some(0));
        assert!(o.speculation);
        // The default drafts from a prefix of the head, which would leave the oracle's logits unmatched.
        assert_ne!(engine::SpecConfig::default().mtp_head_rows, 0);
    }

    /// Logits over a vocabulary of 4 with `top` the largest and `second` the next by `gap`.
    fn logits(top: usize, second: usize, gap: f32) -> Vec<f32> {
        let mut l = vec![0.0; 4];
        l[top] = 5.0;
        l[second] = 5.0 - gap;
        l
    }

    #[test]
    fn a_top1_difference_outside_a_near_tie_fails() {
        // The oracle's top two are 1.0 apart; the engine's top is the oracle's second.
        let oracle = logits(0, 1, 1.0);
        let ours = logits(1, 0, 0.5);
        let row = compare_row(7, &ours, &oracle);
        assert!(!row.top1_equal && !row.near_tie, "{row:?}");
        assert_eq!(row.pos, 7);
        let ok = compare_row(8, &oracle, &oracle);
        let v = judge(vec![ok.clone(), row]);
        assert_eq!(v.status, Status::Fail, "{v:?}");
        assert_eq!(
            (
                v.positions,
                v.top1_equal,
                v.outside_near_tie,
                v.near_tie_differences
            ),
            (2, 1, 1, 0)
        );
        // The error figures are reported either way.
        assert!(v.worst_rel_l2.unwrap() > 0.0);
        // Nothing compared is no pass, and neither is a non-finite logit.
        assert_eq!(judge(Vec::new()).status, Status::Fail);
        let mut nan = oracle.clone();
        nan[2] = f32::NAN;
        let v = judge(vec![compare_row(1, &nan, &oracle)]);
        assert_eq!(v.status, Status::Fail, "{v:?}");
        assert!(v.problems.iter().any(|p| p.contains("non-finite")));
    }

    #[test]
    fn a_near_tie_difference_passes() {
        // The oracle's top two are 0.01 apart, within TIE_MARGIN: the engine may take either.
        let oracle = logits(0, 1, 0.01);
        let ours = logits(1, 0, 0.3);
        let row = compare_row(3, &ours, &oracle);
        assert!(!row.top1_equal && row.near_tie, "{row:?}");
        let v = judge(vec![compare_row(1, &oracle, &oracle), row]);
        assert_eq!(v.status, Status::Pass, "{v:?}");
        assert_eq!(
            (
                v.positions,
                v.top1_equal,
                v.near_tie_differences,
                v.outside_near_tie
            ),
            (2, 1, 1, 0)
        );
        // A gap under TIE_MARGIN is excused, a gap over it is not.
        let at = compare_row(1, &logits(1, 0, 0.3), &logits(0, 1, TIE_MARGIN / 2.0));
        assert!(at.near_tie);
        let above = compare_row(1, &logits(1, 0, 0.3), &logits(0, 1, TIE_MARGIN * 2.0));
        assert!(!above.near_tie);
    }

    #[test]
    fn the_relative_error_is_the_l2_distance_over_the_oracles_norm() {
        assert_eq!(rel_l2(&[3.0, 0.0], &[3.0, 0.0]), Some(0.0));
        let e = rel_l2(&[4.0, 0.0], &[3.0, 0.0]).unwrap();
        assert!((e - 1.0 / 3.0).abs() < 1e-12, "{e}");
        assert_eq!(rel_l2(&[1.0, f32::INFINITY], &[1.0, 1.0]), None);
        assert_eq!(rel_l2(&[1.0], &[0.0]), None);
        assert_eq!(rel_l2(&[1.0], &[1.0, 2.0]), None);
    }

    /// Feeds the check scripted logits and records the calls it made.
    struct Fake {
        calls: Vec<String>,
        n_vocab: usize,
        /// The logits for the token `x`: the one-hot of `x + 1`, except at this `x` (the one-hot of `x + 2`).
        wrong_at: Option<u32>,
        /// The `x` whose drafted id the fake gets wrong (not the argmax of its logits).
        bad_id_at: Option<u32>,
    }

    impl Drafter for Fake {
        fn start(&mut self, token: u32) -> Result<()> {
            self.calls.push(format!("start {token}"));
            Ok(())
        }
        fn logits(&mut self, x: u32) -> Result<(u32, Vec<f32>)> {
            self.calls.push(format!("logits {x}"));
            let mut l = vec![0.0; self.n_vocab];
            let hot = if self.wrong_at == Some(x) {
                x + 2
            } else {
                x + 1
            };
            l[hot as usize] = 4.0;
            let id = if self.bad_id_at == Some(x) { 0 } else { hot };
            Ok((id, l))
        }
        fn decode(&mut self, x: u32) -> Result<()> {
            self.calls.push(format!("decode {x}"));
            Ok(())
        }
    }

    #[test]
    fn the_check_decodes_one_token_at_a_time_and_compares_row_q_minus_1_with_position_q() {
        let n_vocab = 8;
        let tokens = [0u32, 1, 2, 3];
        // The oracle's row for token q is the one-hot of q + 1.
        let mut oracle = Vec::new();
        for q in 1..tokens.len() {
            let mut l = vec![0.0; n_vocab];
            l[tokens[q] as usize + 1] = 4.0;
            oracle.extend(l);
        }
        let mut f = Fake {
            calls: Vec::new(),
            n_vocab,
            wrong_at: None,
            bad_id_at: None,
        };
        let rows = check(&mut f, &tokens, &oracle, n_vocab).unwrap();
        assert_eq!(
            f.calls,
            ["start 0", "logits 1", "decode 1", "logits 2", "decode 2", "logits 3"]
        );
        assert_eq!(rows.iter().map(|r| r.pos).collect::<Vec<_>>(), [1, 2, 3]);
        assert_eq!(judge(rows).status, Status::Pass);
        // A wrong draft at token 2 is that row's failure, and only its.
        let mut f = Fake {
            calls: Vec::new(),
            n_vocab,
            wrong_at: Some(2),
            bad_id_at: None,
        };
        let v = judge(check(&mut f, &tokens, &oracle, n_vocab).unwrap());
        assert_eq!(v.status, Status::Fail);
        assert_eq!(
            v.rows.iter().map(|r| r.top1_equal).collect::<Vec<_>>(),
            [true, false, true]
        );
        // A drafted id that isn't the argmax of its logits is counted and reported, and never fails the check.
        let mut f = Fake {
            calls: Vec::new(),
            n_vocab,
            wrong_at: None,
            bad_id_at: Some(2),
        };
        let v = judge(check(&mut f, &tokens, &oracle, n_vocab).unwrap());
        assert_eq!((v.status, v.draft_id_mismatches), (Status::Pass, 1));
        assert_eq!(
            v.rows
                .iter()
                .map(|r| r.draft_id_is_argmax)
                .collect::<Vec<_>>(),
            [true, false, true]
        );
        // Rows and tokens must agree.
        let e = check(&mut f, &tokens[..3], &oracle, n_vocab).unwrap_err();
        assert!(e.to_string().contains("rows of 8"), "{e}");
    }
}
