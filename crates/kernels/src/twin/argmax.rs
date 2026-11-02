//! `argmax_rows`' twin: the index of the largest finite value, ties to the lower index, `None` when
//! the row has no finite value (the kernel writes `NO_DRAFT`). Max is exact, so the kernel's strided scan and tree give
//! the same answer as this one pass. `engine::sample::greedy` agrees with it only on rows with no NaN and no infinity
//! (greedy lets a NaN at index 0 win and picks +inf).

/// The index of `logits`' largest finite value, ties to the lower index; `None` when none is finite (NaN and the
/// infinities never win).
pub fn argmax_row(logits: &[f32]) -> Option<u32> {
    let mut best: Option<usize> = None;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() && best.is_none_or(|b| v > logits[b]) {
            best = Some(i);
        }
    }
    best.map(|b| b as u32)
}
