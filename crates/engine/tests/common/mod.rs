//! What the MTP's GPU tests share: the CPU reference of `blk.4`/`blk.40` kept in step with a session's
//! rows, fed the hidden rows the session computed, and the picks made outside near ties.
#![allow(dead_code)]
use qwen36::{
    model::Model,
    ops::rel_l2,
    reference::{MtpReference, Rounding},
    taps::NullTap,
};

/// The oracle's top-two logit gap under which two ids are a near tie (test 3's `TIE_MARGIN`).
pub const TIE_MARGIN: f32 = 0.05;

/// `NO_DRAFT`: a row with no finite logit.
pub const NO_DRAFT: u32 = u32::MAX;

/// The largest logit's id and its lead over the second.
pub fn top2(logits: &[f32]) -> (u32, f32) {
    let (mut best, mut second) = ((0usize, f32::NEG_INFINITY), f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            second = best.1;
            best = (i, v);
        } else if v > second {
            second = v;
        }
    }
    (best.0 as u32, best.1 - second)
}

/// The relative L2 distance, the engine tests' error measure.
pub fn rel(a: &[f32], b: &[f32]) -> f64 {
    rel_l2(a, b)
}

pub struct MtpOracle<'a> {
    mtp: MtpReference<'a>,
}

impl<'a> MtpOracle<'a> {
    pub fn new(m: &'a Model<'a>) -> MtpOracle<'a> {
        MtpOracle::with_threads(m, 4)
    }

    pub fn with_threads(m: &'a Model<'a>, threads: usize) -> MtpOracle<'a> {
        MtpOracle {
            mtp: MtpReference::new(m, threads, Rounding::Exact),
        }
    }

    /// Brings the reference's rows to the session's (`rows`, below `x_pos`), each a KV-only row at its position from
    /// `tokens[q]` and `h(q - 1)`, keeping the rows it already has when they are a prefix of those.
    fn sync(
        &mut self,
        rows: &[u32],
        tokens: &[u32],
        x_pos: usize,
        h: &mut dyn FnMut(usize) -> Vec<f32>,
    ) {
        let want: Vec<u32> = rows
            .iter()
            .copied()
            .filter(|&q| (q as usize) < x_pos)
            .collect();
        let have = self.mtp.positions().to_vec();
        let same = have.iter().zip(&want).take_while(|(a, b)| a == b).count();
        if same < have.len() {
            self.mtp.truncate(have[same]);
        }
        for &q in &want[same..] {
            let q = q as usize;
            self.mtp.kv_row(q as u32, tokens[q], &h(q - 1));
        }
    }

    /// The draft row for `x` at `x_pos`: its logits and `m`. `rows` are the session's MTP rows, `tokens` its committed
    /// tokens, and `h(q)` the main model's final hidden row at `q`.
    pub fn expect(
        &mut self,
        rows: &[u32],
        tokens: &[u32],
        x_pos: usize,
        x: u32,
        h: &mut dyn FnMut(usize) -> Vec<f32>,
    ) -> (Vec<f32>, Vec<f32>) {
        self.sync(rows, tokens, x_pos, h);
        let hp = h(x_pos - 1);
        self.mtp.row(x_pos as u32, x, &hp, &mut NullTap)
    }

    /// `k` chained drafts after `x`: (the reference's argmax, its lead, its logits) for each depth. The chained rows are
    /// dropped again, as the engine's are.
    pub fn chain(
        &mut self,
        rows: &[u32],
        tokens: &[u32],
        x_pos: usize,
        x: u32,
        k: usize,
        h: &mut dyn FnMut(usize) -> Vec<f32>,
    ) -> Vec<(u32, f32, Vec<f32>)> {
        let (mut logits, mut m) = self.expect(rows, tokens, x_pos, x, h);
        let mut out = Vec::new();
        for j in 0..k {
            let (id, gap) = top2(&logits);
            out.push((id, gap, logits));
            if j + 1 == k {
                break;
            }
            (logits, m) = self.mtp.row((x_pos + 1 + j) as u32, id, &m, &mut NullTap);
        }
        self.mtp.truncate(x_pos as u32 + 1);
        out
    }
}
