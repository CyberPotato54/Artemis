//! The MTP's rows and the hidden-row bookkeeping around them, CPU side; the module's second half records
//! the GPU work beside each call of `MtpState`.
//!
//! Vocabulary. A *row* is the K and V the MTP block computed for one position `q` from `u_q` (the token at `q`) and
//! `h_{q-1}` (the main model's final hidden row of the position before). `MtpRows` knows which positions have a row, in
//! buffer order. `MtpRing` knows which positions' `h` the VRAM ring still holds. A catch-up row at `q` needs
//! `h_{q-1}` in the ring.
//!
//! `MtpState` and the plans touch no GPU, so nothing there can fail; the session's write rules decide which `on_*` runs.
//! The second half of the module (`MtpGpu`, `Session::mtp_draft`, the prefill window) records the GPU work beside each
//! call of `MtpState`.

use crate::{
    prefill::carve,
    session::MtpOptions,
    spec::MAX_VERIFY,
    timing::{CpuPhase, StepKind, TAG_EMBED, TAG_MTP_WINDOW},
    weights::{GpuMixer, GpuMtp},
    Engine, Error, Session,
};
use kernels::{
    dispatch::{read_status, MatW, NO_DRAFT, STATUS_BYTES, STATUS_CLEAR},
    tokens::MAX_TOKENS,
    twin::attention::rope_table,
};
use qwen36::dims::Dims;
use std::{
    io::Write,
    ops::Range,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        OnceLock,
    },
    time::Instant,
};

/// The MTP route log, a measurement aid: with `ARTEMIS_MTP_ROUTES=FILE` set (read once, any non-empty value), each
/// draft pass appends one JSON line per draft row with the eight experts block 40's router picked for it. Unset, and
/// with no MTP slots (`Options::mtp_slots`, on by default, which copy the same ids for their counts), no buffer is
/// allocated and no command is recorded. Logging only copies the ids the pass computed, so no output changes.
pub const MTP_ROUTES_ENV: &str = "ARTEMIS_MTP_ROUTES";

/// The route log's file, from `ARTEMIS_MTP_ROUTES` (`None` when unset or empty); the variable is read once.
fn mtp_routes_path() -> Option<&'static PathBuf> {
    static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| route_log_path(std::env::var_os(MTP_ROUTES_ENV)))
        .as_ref()
}

/// `mtp_routes_path`'s rule: any non-empty value is the file's path.
fn route_log_path(v: Option<std::ffi::OsString>) -> Option<PathBuf> {
    v.filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// One route-log line: pass number, the position of the draft row's input token, the draft index in the pass, and the
/// routed expert ids of block 40 for that row (the router's order, highest weight first).
fn route_line(pass: u64, pos: usize, j: usize, ids: &[f32]) -> String {
    let ids: Vec<String> = ids.iter().map(|&i| (i as u32).to_string()).collect();
    format!(
        "{{\"pass\":{pass},\"pos\":{pos},\"j\":{j},\"ids\":[{}]}}
",
        ids.join(",")
    )
}

/// The passes logged so far in this process.
static ROUTE_PASSES: AtomicU64 = AtomicU64::new(0);

/// The MTP top-3 log, a measurement aid: with `ARTEMIS_MTP_TOPK=FILE` set (read once, any non-empty value), each draft
/// row appends `{"pos":P,"depth":D,"ids":[a,b,c],"logits":[x,y,z],"lse":L,"rows":N}` (the head rows' top 3 by logit,
/// ties to the lower id, and their log-sum-exp, both computed on the CPU from the rows read back), and the step loop
/// appends `{"pos":P,"pick":T}` for every token it picks. `P` is the position of the row's input token, so a draft row
/// and the pick from the same row share it. Unset, no buffer is allocated, nothing is read back and no line is written.
pub const MTP_TOPK_ENV: &str = "ARTEMIS_MTP_TOPK";

/// The top-3 log's file, from `ARTEMIS_MTP_TOPK` (`None` when unset or empty); the variable is read once.
pub(crate) fn mtp_topk_path() -> Option<&'static PathBuf> {
    static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| route_log_path(std::env::var_os(MTP_TOPK_ENV)))
        .as_ref()
}

/// The `n` highest of `logits` as `(id, logit)`, highest first, equal logits in id order.
fn top_n(logits: &[f32], n: usize) -> Vec<(u32, f32)> {
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(n + 1);
    for (id, &x) in logits.iter().enumerate() {
        // Ids come in ascending order, so a new entry goes after every entry it equals.
        let at = top.iter().position(|&(_, y)| x > y).unwrap_or(top.len());
        if at < n {
            top.insert(at, (id as u32, x));
            top.truncate(n);
        }
    }
    top
}

/// One top-3 log line for the draft row whose input token is at `pos`, `depth` rows into its pass.
fn topk_line(pos: usize, depth: usize, logits: &[f32]) -> String {
    let top = top_n(logits, 3);
    let max = top.first().map_or(0.0, |&(_, x)| f64::from(x));
    let lse = max
        + logits
            .iter()
            .map(|&x| (f64::from(x) - max).exp())
            .sum::<f64>()
            .ln();
    let ids: Vec<String> = top.iter().map(|&(i, _)| i.to_string()).collect();
    let xs: Vec<String> = top.iter().map(|&(_, x)| x.to_string()).collect();
    format!(
        "{{\"pos\":{pos},\"depth\":{depth},\"ids\":[{}],\"logits\":[{}],\"lse\":{lse},\"rows\":{}}}\n",
        ids.join(","),
        xs.join(","),
        logits.len()
    )
}

/// Appends `text` to the log file `path` (named by `env`). A measurement aid: a file that can't be written is
/// reported once and the run goes on.
fn append_log(path: &std::path::Path, env: &str, text: &str) {
    let wrote = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| f.write_all(text.as_bytes()));
    if let Err(err) = wrote {
        static REPORTED: AtomicBool = AtomicBool::new(false);
        if !REPORTED.swap(true, Ordering::Relaxed) {
            eprintln!("{env}: {}: {err}", path.display());
        }
    }
}

/// With the top-3 log on, appends the pick line for the row whose input token is at `pos`.
pub(crate) fn log_pick(pos: usize, token: u32) {
    if let Some(path) = mtp_topk_path() {
        append_log(
            path,
            MTP_TOPK_ENV,
            &format!("{{\"pos\":{pos},\"pick\":{token}}}\n"),
        );
    }
}

/// Copy rows [from_row, from_row + rows) to [0, rows) through a temp region in `Prefill::work` (two copies), so the
/// source and destination never overlap in one vkCmdCopyBuffer. `MtpGpu::record_slide` applies it to the K buffer and the V buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Slide {
    pub from_row: usize,
    pub rows: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowBuf {
    Kv,
    Temp,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RowCopy {
    pub src: (RowBuf, Range<usize>),
    pub dst: (RowBuf, Range<usize>),
}

impl Slide {
    /// `Kv[from_row ..] -> Temp[0 ..]`, then (after a barrier) `Temp[0 ..] -> Kv[0 ..]`. Each copy's source and
    /// destination are different buffers, so neither can overlap itself.
    pub fn copies(&self) -> [RowCopy; 2] {
        [
            RowCopy {
                src: (RowBuf::Kv, self.from_row..self.from_row + self.rows),
                dst: (RowBuf::Temp, 0..self.rows),
            },
            RowCopy {
                src: (RowBuf::Temp, 0..self.rows),
                dst: (RowBuf::Kv, 0..self.rows),
            },
        ]
    }
}

/// The positions the MTP's KV buffer holds a row for, in buffer order (row `r` is `positions[r]`).
#[derive(Clone, Debug)]
pub(crate) struct MtpRows {
    cap: usize,
    positions: Vec<u32>,
}

impl MtpRows {
    pub fn new(cap: usize) -> MtpRows {
        assert!(cap >= 2, "an MTP buffer of {cap} rows");
        MtpRows {
            cap,
            positions: Vec::new(),
        }
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn positions(&self) -> &[u32] {
        &self.positions
    }

    pub fn last_position(&self) -> Option<u32> {
        self.positions.last().copied()
    }

    /// Before a pass that writes `n` rows (catch-up + 1 + (k - 1): the chained rows are written, then dropped): if
    /// `len + n > cap`, keep the newest `cap / 2` and return the slide. Asserts `len + n <= cap` afterwards (`n` is at
    /// most `mtp_ring + MAX_VERIFY`, under `cap / 2` by the session's sizing rules and the ring clamp). Applies it to
    /// the positions; the caller records it, before the pass's first row.
    pub fn make_room(&mut self, n: usize) -> Option<Slide> {
        let len = self.len();
        let mut slide = None;
        if len + n > self.cap {
            let keep = self.cap / 2;
            if len > keep {
                slide = Some(Slide {
                    from_row: len - keep,
                    rows: keep,
                });
                self.positions.drain(..len - keep);
            }
        }
        assert!(
            self.len() + n <= self.cap,
            "a pass of {n} rows does not fit {} rows in a buffer of {}",
            self.len(),
            self.cap
        );
        slide
    }

    /// Appends a row at `pos` (above every row's position) and returns its index.
    pub fn push(&mut self, pos: u32) -> usize {
        assert!(
            self.last_position().is_none_or(|l| pos > l),
            "a row at {pos} after the row at {:?}",
            self.last_position()
        );
        assert!(self.len() < self.cap, "a row pushed into a full buffer");
        self.positions.push(pos);
        self.positions.len() - 1
    }

    /// Drops the rows at positions >= `pos`.
    pub fn cut(&mut self, pos: u32) {
        let keep = self.positions.partition_point(|&p| p < pos);
        self.positions.truncate(keep);
    }

    pub fn clear(&mut self) {
        self.positions.clear();
    }
}

/// Which positions' final hidden rows the VRAM ring holds (slot = position mod cap): one contiguous run of positions.
#[derive(Clone, Debug)]
pub(crate) struct MtpRing {
    cap: usize,
    first: usize,
    count: usize,
}

impl MtpRing {
    pub fn new(cap: usize) -> MtpRing {
        assert!(cap >= 1, "a hidden-row ring of {cap} rows");
        MtpRing {
            cap,
            first: 0,
            count: 0,
        }
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Positions `first .. first + count` were just copied in, after a cut at `first`: whatever the ring held at or
    /// past `first` is gone. A run that starts where the kept one ends extends it; any other (a gap) replaces it. A run
    /// longer than the ring keeps its newest `cap`.
    pub fn record(&mut self, first: usize, count: usize) {
        let old_end = self.first + self.count;
        let kept = if self.count > 0 && first > self.first {
            first.min(old_end) - self.first
        } else {
            0
        };
        let (start, end) = if kept > 0 && self.first + kept == first {
            (self.first, first + count)
        } else if count == 0 {
            // Only the cut: the kept prefix stays.
            (self.first, self.first + kept)
        } else {
            (first, first + count)
        };
        let start = start.max(end.saturating_sub(self.cap));
        self.first = start;
        self.count = end - start;
    }

    pub fn holds(&self, pos: usize) -> bool {
        pos >= self.first && pos < self.first + self.count
    }

    pub fn slot(&self, pos: usize) -> usize {
        pos % self.cap
    }

    pub fn clear(&mut self) {
        self.first = 0;
        self.count = 0;
    }
}

/// What a draft pass will do: the slide, the catch-up rows, x's row, and the chained rows' positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DraftPlan {
    pub slide: Option<Slide>,
    /// (position, token), in position order.
    pub catch_up: Vec<(u32, u32)>,
    /// The position of `x`: the session's next position. Its row is a draft row (the whole block).
    pub x_pos: u32,
    /// The positions of the chained draft rows (`k - 1` of them), written and then dropped.
    pub chained: Vec<u32>,
}

impl DraftPlan {
    /// The rows the pass writes into the buffer: catch-up, x's and the chained ones.
    pub fn rows_written(&self) -> usize {
        self.catch_up.len() + 1 + self.chained.len()
    }
}

/// Catch-up: every position `q` with no row (`q` > the last row's position) whose `h_{q-1}` the ring holds, up to
/// `x`'s. `tokens` are the session's tokens (positions `0 ..`), so `x` goes at position `tokens.len()`. Makes room for
/// the whole pass first (`MtpRows::make_room`), so the plan's slide is already applied to `rows`; the caller records it
/// first in the pass, then calls `MtpState::on_draft`. `x`'s row needs `h` of the position before it: the caller checks
/// `MtpState::ready` (a plan without it would draft from a stale slot, so this asserts).
pub(crate) fn plan_draft(
    rows: &mut MtpRows,
    ring: &MtpRing,
    tokens: &[u32],
    k: usize,
) -> DraftPlan {
    assert!(k >= 1, "a draft pass of 0 drafts");
    let x_pos = tokens.len();
    assert!(
        x_pos >= 1 && ring.holds(x_pos - 1),
        "a draft at {x_pos} without the hidden row before it"
    );
    let from = rows.last_position().map_or(1, |l| l as usize + 1).max(1);
    let catch_up: Vec<(u32, u32)> = (from..x_pos)
        .filter(|&q| ring.holds(q - 1))
        .map(|q| (q as u32, tokens[q]))
        .collect();
    let chained = (1..k).map(|i| (x_pos + i) as u32).collect();
    let plan = DraftPlan {
        slide: None,
        catch_up,
        x_pos: x_pos as u32,
        chained,
    };
    let slide = rows.make_room(plan.rows_written());
    DraftPlan { slide, ..plan }
}

/// The session's MTP bookkeeping: what each session operation does to the rows and the ring. CPU only; the GPU work is recorded
/// beside each call.
#[derive(Clone, Debug)]
pub(crate) struct MtpState {
    pub rows: MtpRows,
    pub ring: MtpRing,
}

impl MtpState {
    pub fn new(rows: usize, ring: usize) -> MtpState {
        assert!(
            ring + MAX_TOKENS <= rows / 2,
            "a ring of {ring} rows in an MTP buffer of {rows}"
        );
        MtpState {
            rows: MtpRows::new(rows),
            ring: MtpRing::new(ring),
        }
    }

    /// Whether a draft at `x_pos` (the session's next position) can run: the ring holds `h` of the position before.
    /// False after a restore with nothing prefilled, for one.
    pub fn ready(&self, x_pos: usize) -> bool {
        x_pos >= 1 && self.ring.holds(x_pos - 1)
    }

    /// A plain decode at `pos` succeeded: its final hidden row goes into the ring. A row of the MTP at a later
    /// position can't be valid (the position is being written again), so it is dropped.
    pub fn on_decode(&mut self, pos: usize) {
        self.rows.cut(pos as u32 + 1);
        self.ring.record(pos, 1);
    }

    /// A verify at `start` was committed keeping `kept` tokens: rows at or past the kept end are rejected drafts'
    /// (x's row, at `start`, stays), and the kept rows' final hidden rows are in the ring.
    pub fn on_commit(&mut self, start: usize, kept: usize) {
        self.rows.cut((start + kept) as u32);
        self.ring.record(start, kept);
    }

    /// A prefill call wrote positions starting at its first; `window` is the part the MTP keeps rows for (the last
    /// `MtpOptions::window` positions of the call). The window's first row needs `h` of the position before it: the
    /// chunk's carry when the window starts inside the call, the ring when the call continues a live session whose ring
    /// holds it, else that position stays a hole. Returns the slide the window's rows need (a live
    /// continuation prefills onto rows that aren't empty); the prefill records it before the first window row. The ring
    /// ends holding the call's last position. This form treats the window's start as the call's first position; a
    /// window that starts later in the call uses `on_prefill_call`.
    #[cfg(test)]
    pub fn on_prefill(&mut self, window: Range<usize>) -> Option<Slide> {
        self.on_prefill_call(window.start, window)
    }

    /// `on_prefill` for a window that starts at or after the call's first position `call_start`.
    #[cfg(test)]
    pub fn on_prefill_call(&mut self, call_start: usize, window: Range<usize>) -> Option<Slide> {
        self.plan_prefill_call(call_start, window)
            .and_then(|p| p.slide)
    }

    /// `on_prefill_call`, returning the whole plan the GPU work follows: where the window's first row is, whether its
    /// `h` comes from the ring (a live continuation) or the chunk's carry, the slide and the buffer row of the
    /// first row. None for an empty window (nothing changes).
    pub fn plan_prefill_call(
        &mut self,
        call_start: usize,
        window: Range<usize>,
    ) -> Option<WindowPlan> {
        assert!(call_start <= window.start, "a window before its call");
        if window.is_empty() {
            return None;
        }
        // Rows at or past the call's first position are stale (a live continuation rewrites them, hole or not).
        self.rows.cut(call_start as u32);
        let from_ring = window.start == call_start && self.ring.holds(window.start.wrapping_sub(1));
        let first_row = if window.start > call_start || from_ring {
            window.start.max(1)
        } else {
            window.start + 1
        };
        let n = window.end.saturating_sub(first_row);
        let slide = self.rows.make_room(n);
        let base = self.rows.len();
        for q in first_row..window.end {
            self.rows.push(q as u32);
        }
        self.ring.record(window.end - 1, 1);
        Some(WindowPlan {
            call_start,
            first_row,
            end: window.end,
            base,
            slide,
            first_from_ring: from_ring,
        })
    }

    /// A draft pass will write `plan`'s rows: pushes the catch-up rows and x's only. The chained rows are written and
    /// dropped (their K and V came from `m`, not the main model's `h`). The plan's slide was applied by `plan_draft`.
    pub fn on_draft(&mut self, plan: &DraftPlan) {
        for &(q, _) in &plan.catch_up {
            self.rows.push(q);
        }
        self.rows.push(plan.x_pos);
    }

    /// `plan_draft` over this state.
    pub fn plan(&mut self, tokens: &[u32], k: usize) -> DraftPlan {
        plan_draft(&mut self.rows, &self.ring, tokens, k)
    }

    /// A draft pass's submit failed: the GPU may have written anything into the MTP's KV and the ring's slots.
    pub fn on_draft_failed(&mut self) {
        self.rows.clear();
        self.ring.clear();
    }

    /// A refused slot, or a row with no finite logit (`NO_DRAFT`): the pass's rows can't be trusted, the ring (the
    /// main model's) can; the next pass catches up from it.
    pub fn on_poisoned(&mut self) {
        self.rows.clear();
    }

    /// A step starting at `start` failed past validation: its rows are cut, and the ring (its slots may be half
    /// written) is cleared.
    pub fn on_failed(&mut self, start: usize) {
        self.rows.cut(start as u32);
        self.ring.clear();
    }

    /// A reset or a restore: nothing the MTP held describes the session any more.
    pub fn on_reset(&mut self) {
        self.rows.clear();
        self.ring.clear();
    }
}

/// The offsets of one batch of `t` rows' temporaries in `Prefill::work`: the embedding rows after `enorm`, the
/// hidden rows after `hnorm`, their concatenation, the `eh_proj` output, its `attn_norm`, and the key and value rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RowRegions {
    pub t: usize,
    pub e: u64,
    pub g: u64,
    pub cat: u64,
    pub proj: u64,
    pub normed: u64,
    pub k: u64,
    pub v: u64,
    pub bytes: u64,
}

impl RowRegions {
    fn new(d: &Dims, t: usize) -> RowRegions {
        let (e, kv) = (d.n_embd, d.n_head_kv * d.head_dim);
        let (o, bytes) = carve(&[t * e, t * e, t * 2 * e, t * e, t * e, t * kv, t * kv]);
        RowRegions {
            t,
            e: o[0],
            g: o[1],
            cat: o[2],
            proj: o[3],
            normed: o[4],
            k: o[5],
            v: o[6],
            bytes,
        }
    }
}

/// The slide's temp: `cap / 2` rows of K and of V (f16), a region each so the two buffers' copies need no barrier
/// between them. 8 MiB at 8,192 rows.
fn slide_temp_bytes(d: &Dims, cap: usize) -> u64 {
    2 * ((cap / 2 * d.n_head_kv * d.head_dim * 2) as u64).next_multiple_of(256)
}

/// A draft pass's carve of `Prefill::work`: one 16-row catch-up region, reused chunk by chunk behind a barrier, and
/// the slide's temp (K's half at `temp_k`, V's at `temp_v`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MtpLayout {
    pub catch_up: RowRegions,
    pub temp_k: u64,
    pub temp_v: u64,
    pub bytes: u64,
}

impl MtpLayout {
    pub fn new(d: &Dims, mtp_ctx: usize) -> MtpLayout {
        let catch_up = RowRegions::new(d, MAX_TOKENS);
        let half = slide_temp_bytes(d, mtp_ctx) / 2;
        let temp_k = catch_up.bytes;
        let temp_v = temp_k + half;
        MtpLayout {
            bytes: temp_v + half,
            catch_up,
            temp_k,
            temp_v,
        }
    }
}

/// A prefill chunk's window rows in `Prefill::work` (after the chunk's layers, when the work area is idle): up to
/// `chunk` rows at once, then the slide's temp (the first window chunk may slide first).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WindowLayout {
    pub rows: RowRegions,
    pub temp_k: u64,
    pub temp_v: u64,
    pub bytes: u64,
}

impl WindowLayout {
    pub fn new(d: &Dims, chunk: usize, mtp_ctx: usize) -> WindowLayout {
        let rows = RowRegions::new(d, chunk);
        let half = slide_temp_bytes(d, mtp_ctx) / 2;
        let temp_k = rows.bytes;
        let temp_v = temp_k + half;
        WindowLayout {
            bytes: temp_v + half,
            rows,
            temp_k,
            temp_v,
        }
    }
}

/// The head rows a draft pass reads: 0 is the full head, and a vocabulary under `asked` (the tiny test models, under the
/// default 65,536) is the full head too.
pub(crate) fn head_rows(asked: usize, n_vocab: usize) -> usize {
    if asked == 0 {
        n_vocab
    } else {
        asked.min(n_vocab)
    }
}

/// The most a ring may hold in an engine whose prefill chunk is `chunk`: the chunk minus 16 (a pass's rows are the ring's
/// catch-up plus up to 16 more), rounded down to 16, and at least 16.
pub(crate) fn clamp_ring(ring: usize, chunk: usize) -> usize {
    ring.min((chunk.saturating_sub(MAX_TOKENS) / 16 * 16).max(16))
}

/// The load's check that the engine's prefill buffers carry the MTP's passes (refused before the model loads): a draft
/// pass keeps one RoPE table and one embedding address per row in `Prefill::rope` and `Prefill::rows` (the catch-up
/// rows, `x` and the chained rows: up to `ring + MAX_TOKENS`), and its rows and the window's work in `Prefill::work`
/// (`MtpLayout`, `WindowLayout`), all sized by the prefill chunk.
pub(crate) fn check_prefill(
    d: &Dims,
    chunk: usize,
    verify_ctx: Option<usize>,
    o: MtpOptions,
) -> Result<(), Error> {
    if chunk < o.ring + MAX_TOKENS {
        return Err(Error::SpeculationRefused {
            why: "the MTP needs a prefill chunk of at least 32 positions",
        });
    }
    let work = crate::prefill::work_bytes(d, chunk, verify_ctx);
    if MtpLayout::new(d, o.rows).bytes > work || WindowLayout::new(d, chunk, o.rows).bytes > work {
        return Err(Error::SpeculationRefused {
            why: "the prefill work area is too small for the MTP's rows (a larger prefill chunk or a lower mtp_ctx)",
        });
    }
    Ok(())
}

/// What a prefill call does for the MTP's window (`MtpState::plan_prefill_call`): the rows `first_row .. end` get KV-only
/// rows, written chunk by chunk; `base` is the buffer row of `first_row` (after the slide, before the pushes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WindowPlan {
    pub call_start: usize,
    pub first_row: usize,
    pub end: usize,
    pub base: usize,
    pub slide: Option<Slide>,
    /// The window's first row is the call's first position and its `h` is in the ring (a live continuation).
    pub first_from_ring: bool,
}

/// Where a chunk's first in-window row takes its `h` from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrevH {
    /// From the chunk's own final-norm rows (the row's `q − 1` is in the chunk), or the chunk has no window rows.
    Chunk,
    /// The previous chunk's last row, held in the carry.
    Carry,
    /// The ring's slot (a live continuation's first position).
    Ring(usize),
}

/// What one prefill chunk of positions `c0 .. c0 + t` does for the MTP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ChunkWindow {
    /// The chunk's rows that get an MTP row: positions `c0 + lo .. c0 + t`.
    pub n_rows: usize,
    pub lo: usize,
    /// The chunk-relative index of the first row whose final norm is computed: the one before the first window row, or
    /// the chunk's last row when it has none. The chunk's last row is the carry's.
    pub a: usize,
    /// The buffer row of the first window row of this chunk.
    pub row0: usize,
    pub prev: PrevH,
    /// The slide to record first: on the chunk that holds the window's first row.
    pub slide: Option<Slide>,
    /// The call's last chunk: its last row also goes into the ring.
    pub last: bool,
}

impl WindowPlan {
    pub fn chunk(&self, c0: usize, t: usize, ring: usize) -> ChunkWindow {
        let end_c = c0 + t;
        assert!(
            t >= 1 && c0 >= self.call_start && end_c <= self.end,
            "a chunk {c0} .. {end_c} outside its call {} .. {}",
            self.call_start,
            self.end
        );
        let lo_abs = self.first_row.max(c0).min(end_c);
        let n_rows = end_c - lo_abs;
        let lo = lo_abs - c0;
        let (a, prev) = if n_rows == 0 {
            (t - 1, PrevH::Chunk)
        } else if lo > 0 {
            (lo - 1, PrevH::Chunk)
        } else if c0 == self.call_start {
            assert!(
                self.first_from_ring && c0 >= 1,
                "a window row at the call's first position without a ring row before it"
            );
            (0, PrevH::Ring((c0 - 1) % ring))
        } else {
            (0, PrevH::Carry)
        };
        ChunkWindow {
            n_rows,
            lo,
            a,
            row0: self.base + lo_abs.saturating_sub(self.first_row),
            prev,
            slide: if n_rows > 0 && lo_abs == self.first_row {
                self.slide
            } else {
                None
            },
            last: end_c == self.end,
        }
    }
}

/// Positions `first .. first + n` in a ring of `cap` slots (slot = position mod `cap`), as (slot, run length, source row
/// offset) runs that are each contiguous in the ring. A run longer than the ring keeps its newest `cap` rows.
pub(crate) fn ring_runs(cap: usize, first: usize, n: usize) -> Vec<(usize, usize, usize)> {
    let skip = n.saturating_sub(cap);
    let (mut pos, mut off, mut left) = (first + skip, skip, n - skip);
    let mut out = Vec::new();
    while left > 0 {
        let slot = pos % cap;
        let len = left.min(cap - slot);
        out.push((slot, len, off));
        pos += len;
        off += len;
        left -= len;
    }
    out
}

/// The session's MTP buffers: allocated in `mtp_session_sizes`' order from the session's arena.
pub(crate) struct MtpGpu {
    /// The MTP's key and value caches: `rows` rows of `n_head_kv * head_dim` f16 each, in position order.
    pub k: vk::Slice,
    pub v: vk::Slice,
    /// The hidden-row ring: `ring_rows` rows of `n_embd` f32, at slot `position mod ring_rows`.
    pub ring: vk::Slice,
    /// The last hidden row of the prefill chunk before, across a chunk boundary.
    pub carry: vk::Slice,
    /// A pass's draft ids (u32) and the embedding-row addresses of the rows they chain into (u64), by draft index.
    pub ids: vk::Slice,
    pub addrs: vk::Slice,
    /// Block 40's routed ids per draft row (f32, `n_expert_used` a row), read back: for the MTP slots' counts
    /// (`Engine::mtp_slots`) and `ARTEMIS_MTP_ROUTES`' log; None when neither is on.
    pub routes: Option<vk::Buffer>,
    /// Each draft row's head logits (`head_rows` f32 a row), read back for `ARTEMIS_MTP_TOPK`'s log; None when it is
    /// off.
    pub topk: Option<vk::Buffer>,
    pub ring_rows: usize,
    pub catch: MtpLayout,
    pub window: WindowLayout,
    /// Bytes of one K or V row.
    pub row_bytes: u64,
    /// The rows of the main head a draft pass reads: `MtpOptions::head_rows`, or `n_vocab` for the full head.
    pub head_rows: usize,
}

impl MtpGpu {
    pub fn new(e: &Engine, arena: &mut vk::Arena, o: MtpOptions) -> Result<MtpGpu, Error> {
        let d = &e.dims;
        let sizes = crate::session::mtp_session_sizes(d, o.rows, o.ring);
        let mut s = Vec::with_capacity(sizes.len());
        for b in sizes {
            s.push(arena.alloc(&e.dev, b)?);
        }
        let [k, v, ring, carry, ids, addrs]: [vk::Slice; 6] =
            s.try_into().expect("one slice per MTP buffer");
        Ok(MtpGpu {
            k,
            v,
            ring,
            carry,
            ids,
            addrs,
            routes: if e.mtp_slots.is_some() || mtp_routes_path().is_some() {
                Some(e.dev.alloc(
                    (MAX_VERIFY * d.n_expert_used * 4) as u64,
                    vk::MemKind::Readback,
                    "the MTP's routes",
                )?)
            } else {
                None
            },
            topk: if mtp_topk_path().is_some() {
                Some(e.dev.alloc(
                    (MAX_VERIFY * head_rows(o.head_rows, d.n_vocab) * 4) as u64,
                    vk::MemKind::Readback,
                    "the MTP's top-3 log rows",
                )?)
            } else {
                None
            },
            ring_rows: o.ring,
            catch: MtpLayout::new(d, o.rows),
            window: WindowLayout::new(d, e.prefill.chunk, o.rows),
            row_bytes: (d.n_head_kv * d.head_dim * 2) as u64,
            head_rows: head_rows(o.head_rows, d.n_vocab),
        })
    }

    /// Copies `n` final hidden rows (positions `first ..`, in `src`) into the ring.
    pub fn copy_to_ring(
        &self,
        r: &mut vk::Recorder,
        src: vk::Slice,
        first: usize,
        n: usize,
        n_embd: usize,
    ) {
        let rb = 4 * n_embd as u64;
        for (slot, len, off) in ring_runs(self.ring_rows, first, n) {
            r.copy(
                src.sub(off as u64 * rb, len as u64 * rb),
                self.ring.sub(slot as u64 * rb, len as u64 * rb),
            );
        }
    }

    /// The slide's copies (`Slide::copies`: K/V rows `from_row ..` to a temp region, then back to row 0, with a
    /// barrier between), for K and V through `temp_k` and `temp_v` (offsets into `work`): no copy's source and
    /// destination overlap, so no `vkCmdCopyBuffer` rule is broken whatever `from_row` is.
    pub fn record_slide(
        &self,
        r: &mut vk::Recorder,
        s: &Slide,
        work: vk::Slice,
        temp_k: u64,
        temp_v: u64,
    ) {
        let rb = self.row_bytes;
        for copy in s.copies() {
            for (kv, temp) in [(self.k, temp_k), (self.v, temp_v)] {
                let side = |(buf, rows): &(RowBuf, Range<usize>)| {
                    let (start, len) = (rows.start as u64 * rb, rows.len() as u64 * rb);
                    match buf {
                        RowBuf::Kv => kv.sub(start, len),
                        RowBuf::Temp => work.sub(temp + start, len),
                    }
                };
                r.copy(side(&copy.src), side(&copy.dst));
            }
            r.barrier();
        }
    }
}

/// The recording context of one MTP command buffer: the engine, the session's MTP buffers, `blk.40`, and the work
/// area's base address.
struct Rec<'a> {
    e: &'a Engine,
    g: &'a MtpGpu,
    w: &'a GpuMtp,
    work: vk::Slice,
    /// The session's attention partials.
    part: u64,
}

impl Rec<'_> {
    fn at(&self, off: u64) -> u64 {
        self.work.addr + off
    }

    /// A KV-only row's tail, from the concatenated `[e ; g]` rows in `reg.cat`: `eh_proj`, `attn_norm`, the key and
    /// value projections, the key's norm and RoPE (`n_head = 0` runs the key heads alone) and the KV write at buffer
    /// rows `row0 ..`. `rope` is the first row's RoPE table; `dense` runs a projection over the `t` rows.
    fn kv_tail(
        &self,
        r: &mut vk::Recorder,
        reg: &RowRegions,
        rope: u64,
        row0: usize,
        dense: &dyn Fn(&mut vk::Recorder, MatW, u64, u64),
    ) {
        let (d, k) = (&self.e.dims, &self.e.k);
        let t = reg.t;
        let eps = d.rms_eps;
        let GpuMixer::Attention {
            k: kw,
            v,
            q_norm,
            k_norm,
            ..
        } = &self.w.layer.mixer
        else {
            unreachable!("blk.{} is a full-attention block", d.n_layer)
        };
        dense(r, self.w.eh_proj, self.at(reg.cat), self.at(reg.proj));
        r.barrier();
        k.rmsnorm_tokens(
            r,
            self.at(reg.proj),
            self.w.layer.attn_norm,
            self.at(reg.normed),
            d.n_embd,
            eps,
            t,
        );
        r.barrier();
        dense(r, *kw, self.at(reg.normed), self.at(reg.k));
        dense(r, *v, self.at(reg.normed), self.at(reg.v));
        r.barrier();
        // `n_head = 0`: no query head runs, so `qfull`, `q` and `gate` are never touched.
        let kk = self.at(reg.k);
        k.qk_norm_rope_tokens(
            r,
            kk,
            kk,
            kk,
            kk,
            *q_norm,
            *k_norm,
            rope,
            0,
            d.n_head_kv,
            d.head_dim,
            d.n_rot,
            eps,
            t,
        );
        r.barrier();
        k.kv_write_tokens(
            r,
            self.at(reg.k),
            self.at(reg.v),
            self.g.k.addr,
            self.g.v.addr,
            row0,
            d.n_head_kv * d.head_dim,
            t,
        );
    }

    /// `[e ; g]` rows from the `e` and `g` regions into `cat`, the embedding half first.
    fn concat(&self, r: &mut vk::Recorder, reg: &RowRegions, t: usize) {
        let rb = 4 * self.e.dims.n_embd as u64;
        let w = self.work;
        r.copy_rows(
            w.sub(reg.e, t as u64 * rb),
            rb,
            w.sub(reg.cat, 2 * t as u64 * rb),
            2 * rb,
            rb,
            t,
        );
        r.copy_rows(
            w.sub(reg.g, t as u64 * rb),
            rb,
            w.sub(reg.cat + rb, (2 * t as u64 - 1) * rb),
            2 * rb,
            rb,
            t,
        );
    }

    /// The catch-up rows: KV-only rows for consecutive positions, in chunks of `MAX_TOKENS` through one region,
    /// reading each row's `h` from the ring. Row `i` of the pass has the embedding address `Prefill::rows[i]`, the RoPE
    /// table `Prefill::rope[i]` and the buffer row `base_row + i`.
    fn catch_up(&self, r: &mut vk::Recorder, plan: &DraftPlan, ring_rows: usize, base_row: usize) {
        let (d, k, p) = (&self.e.dims, &self.e.k, &self.e.prefill);
        let (eps, n_embd) = (d.rms_eps, d.n_embd);
        let reg = &self.g.catch.catch_up;
        let rb = 4 * n_embd as u64;
        let ring = self.g.ring;
        for (ci, rows) in plan.catch_up.chunks(MAX_TOKENS).enumerate() {
            let t = rows.len();
            let i0 = ci * MAX_TOKENS;
            let q0 = rows[0].0 as usize;
            assert!(
                rows.iter()
                    .enumerate()
                    .all(|(j, &(q, _))| q as usize == q0 + j),
                "catch-up rows at consecutive positions"
            );
            // The regions are sized for MAX_TOKENS rows; this chunk may be shorter, so the views use `t` rows of them.
            let view = RowRegions { t, ..reg.clone() };
            k.embed_rows(
                r,
                p.rows.addr() + 8 * i0 as u64,
                self.e.host.embd_ty,
                n_embd,
                self.at(view.normed),
                t,
            );
            r.barrier();
            k.rmsnorm_tokens(
                r,
                self.at(view.normed),
                self.w.enorm,
                self.at(view.e),
                n_embd,
                eps,
                t,
            );
            // g: h of the positions before, from the ring (slots wrap).
            for (slot, len, off) in ring_runs(ring_rows, q0 - 1, t) {
                k.rmsnorm_tokens(
                    r,
                    ring.addr + slot as u64 * rb,
                    self.w.hnorm,
                    self.at(view.g) + off as u64 * rb,
                    n_embd,
                    eps,
                    len,
                );
            }
            r.barrier();
            self.concat(r, &view, t);
            r.barrier();
            let dense = |r: &mut vk::Recorder, w: MatW, x: u64, y: u64| {
                k.matvec_tokens(r, w, x, y, t, w.cols, w.rows)
            };
            self.kv_tail(
                r,
                &view,
                p.rope.addr() + 4 * (d.n_rot * i0) as u64,
                base_row + i0,
                &dense,
            );
            // The next chunk reuses the region.
            r.barrier();
        }
    }

    /// The draft rows: row `j` embeds its token, runs the whole block in decode's `Scratch`, and picks the next
    /// draft with `argmax_rows`. Row 0 is `x`'s, at buffer row `x_row`, reading `h` from ring slot `h_slot`; the
    /// chained rows read the previous row's `m` (in `fnorm`). `first_table` is the pass's table index of row 0. With
    /// `logits`, row 0's logits are also copied to the engine's readback.
    #[allow(clippy::too_many_arguments)]
    fn draft_rows(
        &self,
        r: &mut vk::Recorder,
        k_rows: usize,
        x_row: usize,
        first_table: usize,
        h_slot: usize,
        embd: (u64, usize),
        logits: bool,
    ) {
        let (e, kn) = (self.e, &self.e.k);
        let (d, sc, p) = (&e.dims, &e.scratch, &e.prefill);
        let eps = d.rms_eps;
        let (nh, nkv, hd) = (d.n_head, d.n_head_kv, d.head_dim);
        let layer = &self.w.layer;
        let GpuMixer::Attention {
            q,
            k: kw,
            v,
            out,
            q_norm,
            k_norm,
        } = &layer.mixer
        else {
            unreachable!("blk.{} is a full-attention block", d.n_layer)
        };
        let table = e.expert_table();
        let status = e.status.addr();
        let part = self.part;
        assert!(
            sc.qkv.size >= 8 * d.n_embd as u64,
            "Scratch::qkv holds the concatenated [e ; g] row"
        );
        let cat = sc.qkv;
        for j in 0..k_rows {
            let row = x_row + j;
            let n_ctx = row + 1;
            let embd_rows = if j == 0 {
                p.rows.addr() + 8 * first_table as u64
            } else {
                self.g.addrs.addr + 8 * (j as u64 - 1)
            };
            kn.embed_rows(r, embd_rows, e.host.embd_ty, d.n_embd, sc.x.addr, 1);
            r.barrier();
            // e = RMSNorm(embd) ⊙ enorm, g = RMSNorm(h) ⊙ hnorm, side by side in `qkv` (4,096 of its 8,192 values).
            kn.rmsnorm(r, sc.x.addr, self.w.enorm, cat.addr, d.n_embd, eps);
            let h = if j == 0 {
                self.g.ring.addr + (h_slot * d.n_embd * 4) as u64
            } else {
                sc.fnorm.addr
            };
            kn.rmsnorm(
                r,
                h,
                self.w.hnorm,
                cat.addr + 4 * d.n_embd as u64,
                d.n_embd,
                eps,
            );
            r.barrier();
            // u = eh_proj · [e ; g], into `x` (the embedding it held is consumed).
            kn.matvec(r, self.w.eh_proj, cat.addr, sc.x.addr);
            r.barrier();
            kn.rmsnorm(r, sc.x.addr, layer.attn_norm, sc.h.addr, d.n_embd, eps);
            r.barrier();
            kn.matvec(r, *q, sc.h.addr, sc.qfull.addr);
            kn.matvec(r, *kw, sc.h.addr, sc.k.addr);
            kn.matvec(r, *v, sc.h.addr, sc.v.addr);
            r.barrier();
            kn.qk_norm_rope(
                r,
                sc.qfull.addr,
                sc.k.addr,
                sc.q.addr,
                sc.gate.addr,
                *q_norm,
                *k_norm,
                p.rope.addr() + 4 * (d.n_rot * (first_table + j)) as u64,
                nh,
                nkv,
                hd,
                d.n_rot,
                eps,
            );
            r.barrier();
            kn.kv_write(
                r,
                sc.k.addr,
                sc.v.addr,
                self.g.k.addr,
                self.g.v.addr,
                row,
                nkv * hd,
            );
            r.barrier();
            kn.attn_decode(
                r,
                sc.q.addr,
                self.g.k.addr,
                self.g.v.addr,
                part,
                n_ctx,
                nh,
                nkv,
                hd,
            );
            r.barrier();
            kn.attn_combine(r, part, sc.pre.addr, n_ctx, nh, hd);
            r.barrier();
            kn.sigmoid_mul(r, sc.pre.addr, sc.gate.addr, sc.gated.addr, nh * hd);
            r.barrier();
            kn.matvec(r, *out, sc.gated.addr, sc.mix.addr);
            r.barrier();
            kn.add(r, sc.x.addr, sc.mix.addr, sc.resid.addr, d.n_embd);
            r.barrier();
            kn.rmsnorm(r, sc.resid.addr, layer.ffn_norm, sc.f.addr, d.n_embd, eps);
            r.barrier();
            // The MoE reads the slot table's layer-`n_layer` entries: HOST, or an MTP slot (`mtp_slots.rs`) for the
            // experts the slots hold, through the same kernels. The router's top-k lands in
            // `Scratch::ids` and `w`; the routing log (`Engine::routes`) is never written.
            kn.matvec(r, layer.router, sc.f.addr, sc.rlog.addr);
            r.barrier();
            kn.router_topk(
                r,
                sc.rlog.addr,
                sc.probs.addr,
                sc.ids.addr,
                sc.w.addr,
                d.n_expert,
                d.n_expert_used,
            );
            r.barrier();
            if let Some(log) = &self.g.routes {
                // The routes: this row's ids, before the next row overwrites `ids`.
                let b = 4 * d.n_expert_used as u64;
                r.copy(sc.ids, log.slice(j as u64 * b, b));
            }
            kn.experts_gate_up(
                r,
                table,
                d.n_layer,
                d.n_expert,
                sc.ids.addr,
                sc.f.addr,
                sc.act.addr,
                status,
                &layer.slot,
                d.n_expert_used,
                0,
                0,
                0,
            );
            r.barrier();
            kn.experts_down(
                r,
                table,
                d.n_layer,
                d.n_expert,
                sc.ids.addr,
                sc.act.addr,
                sc.ye.addr,
                status,
                &layer.slot,
                d.n_expert_used,
                0,
                0,
                0,
            );
            kn.matvec(r, layer.sh_gate, sc.f.addr, sc.sg_g.addr);
            kn.matvec(r, layer.sh_up, sc.f.addr, sc.sg_u.addr);
            r.barrier();
            kn.swiglu(r, sc.sg_g.addr, sc.sg_u.addr, sc.sact.addr, d.n_ff_shexp);
            r.barrier();
            kn.matvec(r, layer.sh_down, sc.sact.addr, sc.sh.addr);
            kn.matvec(r, layer.sh_gate_inp, sc.f.addr, sc.sg.addr);
            r.barrier();
            kn.moe_combine(
                r,
                sc.ye.addr,
                sc.w.addr,
                sc.sh.addr,
                sc.sg.addr,
                sc.moe_out.addr,
                sc.shexp_out.addr,
                d.n_embd,
                d.n_expert_used,
            );
            r.barrier();
            kn.add(
                r,
                sc.moe_out.addr,
                sc.shexp_out.addr,
                sc.ffn_out.addr,
                d.n_embd,
            );
            r.barrier();
            // y = r + FFN(..), then m = RMSNorm(y) ⊙ shared_head_norm and the main model's head.
            kn.add(r, sc.resid.addr, sc.ffn_out.addr, sc.x.addr, d.n_embd);
            r.barrier();
            kn.rmsnorm(r, sc.x.addr, self.w.head_norm, sc.fnorm.addr, d.n_embd, eps);
            r.barrier();
            // The head's first `head_rows` rows: a prefix view of the main head (rows are contiguous per token id).
            let head = MatW {
                rows: self.g.head_rows,
                ..e.w.output
            };
            kn.matvec(r, head, sc.fnorm.addr, sc.logits.addr);
            r.barrier();
            if let Some(log) = &self.g.topk {
                // This row's logits, before the next row overwrites them.
                let bytes = 4 * self.g.head_rows as u64;
                r.copy(sc.logits.sub(0, bytes), log.slice(j as u64 * bytes, bytes));
            }
            if logits && j == 0 {
                let bytes = 4 * self.g.head_rows as u64;
                r.copy(sc.logits.sub(0, bytes), e.logits.whole().sub(0, bytes));
            }
            kn.argmax_rows(
                r,
                sc.logits.addr,
                self.g.head_rows,
                self.g.ids.addr,
                self.g.addrs.addr,
                embd.0,
                embd.1,
                j,
            );
            r.barrier();
        }
    }
}

/// A key row and a value row of the MTP's KV, as f32.
pub type KvRow = (Vec<f32>, Vec<f32>);

impl Session {
    /// MTP drafts after `x`: up to `k` tokens, the argmax of each draft row's logits, chained. `x` is the
    /// token about to be decoded at the session's next position. One command buffer and one submit: the slide if the
    /// MTP's buffer is full, the catch-up rows for every position committed since the last pass (from the ring of hidden
    /// rows), `x`'s row and the `k - 1` chained rows. Nothing of the main state moves: no position, no pending verify,
    /// not the KV, the DeltaNet state or the routing log.
    ///
    /// Empty when the session has no MTP, or the ring doesn't hold the hidden row before `x` (right after a restore), or
    /// the first draft row has no finite logit; fewer than `k` when a later row has none. Refuses a `k` of 0 or past
    /// `min(MAX_VERIFY - 1, ctx - pos - 1)`, a token outside the vocabulary and a full context, before it changes
    /// anything (`MtpShape`, `TokenOutOfRange`, `ContextFull`: the session stays as it was).
    ///
    /// Failures: a refused slot (`BadSlotEntry`, read after the submit) returns with the session usable and the
    /// MTP's rows cleared; a failed submit makes the session unusable until `reset`, as decode's write rule says (the
    /// GPU may still be writing `Scratch`, the work area and the MTP's KV), and clears the rows and the ring. A NaN in
    /// the logits is no error: see `argmax_rows`.
    ///
    /// `assert!`s that no verify is pending and that the CPU miss path and the base time are off.
    pub fn mtp_draft(&mut self, e: &Engine, x: u32, k: usize) -> Result<Vec<u32>, Error> {
        Ok(self.draft_pass(e, x, k, false)?.0)
    }

    /// `mtp_draft` at `k = 1`, also returning the draft row's logits: `n_vocab` of them, or `mtp_head_rows` when that is
    /// set. Tests only. A row with no finite logit gives `NO_DRAFT` as the id; a session that can't draft (no MTP, no
    /// hidden row before `x`) is an error.
    #[doc(hidden)]
    pub fn mtp_draft_logits(&mut self, e: &Engine, x: u32) -> Result<(u32, Vec<f32>), Error> {
        match self.draft_pass(e, x, 1, true)? {
            (ids, Some(logits)) => Ok((ids.first().copied().unwrap_or(NO_DRAFT), logits)),
            _ => Err(Error::SpeculationRefused {
                why: "the MTP has no hidden row before x",
            }),
        }
    }

    /// Whether the session has the MTP's buffers (the engine opened with `Options::mtp`).
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// The positions the MTP has a key/value row for, in buffer order. Tests only.
    #[doc(hidden)]
    pub fn mtp_rows(&self) -> Vec<u32> {
        self.mtp
            .as_ref()
            .map_or_else(Vec::new, |m| m.rows.positions().to_vec())
    }

    /// The final hidden row of `pos` (the head's input) if the ring still holds it. Tests only: the reference is fed
    /// the same rows the session computed.
    #[doc(hidden)]
    pub fn mtp_hidden(&self, e: &Engine, pos: usize) -> Result<Option<Vec<f32>>, Error> {
        let (Some(m), Some(g)) = (&self.mtp, &self.mtp_gpu) else {
            return Ok(None);
        };
        if !m.ring.holds(pos) {
            return Ok(None);
        }
        let rb = 4 * e.dims.n_embd as u64;
        let bytes = e
            .dev
            .download(g.ring.sub(m.ring.slot(pos) as u64 * rb, rb))?;
        Ok(Some(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
        ))
    }

    /// The MTP's key and value of buffer row `row` (position `mtp_rows()[row]`), as f32. Tests only.
    #[doc(hidden)]
    pub fn mtp_kv_row(&self, e: &Engine, row: usize) -> Result<Option<KvRow>, Error> {
        let (Some(m), Some(g)) = (&self.mtp, &self.mtp_gpu) else {
            return Ok(None);
        };
        if row >= m.rows.len() {
            return Ok(None);
        }
        let f16s = |s: vk::Slice| -> Result<Vec<f32>, Error> {
            Ok(e.dev
                .download(s)?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| kernels::twin::f16_value(u16::from_le_bytes(*c)))
                .collect())
        };
        let at = row as u64 * g.row_bytes;
        Ok(Some((
            f16s(g.k.sub(at, g.row_bytes))?,
            f16s(g.v.sub(at, g.row_bytes))?,
        )))
    }

    /// Tests only: from now on the prefill window also reads back the final-norm rows it computes, so `window_hidden`
    /// can hand the reference the session's own hidden rows for the positions the ring no longer holds. Allocates a
    /// readback of `ctx` rows.
    #[doc(hidden)]
    pub fn capture_window_hidden(&mut self, e: &Engine) -> Result<(), Error> {
        if self.mtp.is_some() {
            let rb = 4 * e.dims.n_embd as u64;
            let buf = e.dev.alloc(
                self.ctx as u64 * rb,
                vk::MemKind::Readback,
                "the captured window rows",
            )?;
            self.capture = Some((buf, vec![false; self.ctx]));
        }
        Ok(())
    }

    /// The final hidden row of `pos` as the prefill window computed it (`capture_window_hidden`), if it did. Tests only.
    #[doc(hidden)]
    pub fn window_hidden(&self, e: &Engine, pos: usize) -> Option<Vec<f32>> {
        let (buf, have) = self.capture.as_ref()?;
        have.get(pos)
            .copied()
            .unwrap_or(false)
            .then(|| buf.read_f32(pos as u64 * 4 * e.dims.n_embd as u64, e.dims.n_embd))
    }

    /// Tests only: the next draft pass's submit fails, as a device error would (nothing is submitted).
    #[doc(hidden)]
    pub fn fail_next_mtp_submit(&mut self) {
        self.fail_mtp_submit = true;
    }

    fn draft_pass(
        &mut self,
        e: &Engine,
        x: u32,
        k: usize,
        want_logits: bool,
    ) -> Result<(Vec<u32>, Option<Vec<f32>>), Error> {
        assert!(!self.pending(), "an MTP draft with a verify step pending");
        if self.engine != e.id {
            return Err(Error::WrongEngine);
        }
        let Some(m) = &self.mtp else {
            return Ok((Vec::new(), None));
        };
        let (d, x_pos) = (&e.dims, self.pos);
        // Validation first: the slide `plan` applies to the rows can't be taken back by a refusal.
        if x as usize >= d.n_vocab {
            return Err(Error::TokenOutOfRange {
                token: x,
                n_vocab: d.n_vocab,
            });
        }
        if x_pos >= self.ctx {
            return Err(Error::ContextFull { ctx: self.ctx });
        }
        let most = (MAX_VERIFY - 1).min(self.ctx - x_pos - 1);
        if k == 0 || k > most {
            return Err(Error::MtpShape { k, most });
        }
        assert!(
            e.cpu.borrow().is_none() && e.base.is_none(),
            "Session::mtp_draft beside the CPU miss path or the base time (C12)"
        );
        if !m.ready(x_pos) {
            return Ok((Vec::new(), None));
        }
        let m = self.mtp.as_mut().expect("checked above");
        // A second draft at the same position redoes `x`'s row.
        m.rows.cut(x_pos as u32);
        let plan = m.plan(&self.tokens, k);
        let base_row = m.rows.len();
        m.on_draft(&plan);
        // From here the rows lead the GPU's: any refusal clears them (`on_poisoned`).
        e.begin_timing(StepKind::Mtp, x_pos, k);
        self.run_draft(e, x, &plan, base_row, want_logits)
    }

    fn run_draft(
        &mut self,
        e: &Engine,
        x: u32,
        plan: &DraftPlan,
        base_row: usize,
        want_logits: bool,
    ) -> Result<(Vec<u32>, Option<Vec<f32>>), Error> {
        let (d, p) = (&e.dims, &e.prefill);
        let x_pos = plan.x_pos as usize;
        let (c, k) = (plan.catch_up.len(), plan.chained.len() + 1);
        let step = |source| Error::Step {
            pos: x_pos,
            layer: None,
            source,
        };
        assert!(
            c + k <= p.chunk,
            "a pass of {} rows past the prefill chunk's RoPE tables",
            c + k
        );
        let t0 = Instant::now();
        let embd = e.host.embd_base();
        // The pass's embedding addresses and RoPE tables, one per row: the catch-up rows, `x`, then the chained rows
        // (whose addresses come from `argmax_rows`).
        let rows: Vec<u8> = plan
            .catch_up
            .iter()
            .map(|&(_, t)| t)
            .chain([x])
            .flat_map(|t| e.host.embd_row(t).to_le_bytes())
            .collect();
        p.rows.write(0, &rows);
        let tables: Vec<f32> = plan
            .catch_up
            .iter()
            .map(|&(q, _)| q)
            .chain([plan.x_pos])
            .chain(plan.chained.iter().copied())
            .flat_map(|q| rope_table(q, d.n_rot, d.rope_base))
            .collect();
        p.rope.write_f32(0, &tables);
        e.status.write(0, &STATUS_CLEAR);
        let mut r = match e.record() {
            Ok(r) => r,
            Err(source) => {
                self.mtp_poisoned();
                return Err(step(source));
            }
        };
        let head_rows;
        let moves;
        {
            let (Some(g), Some(w)) = (&self.mtp_gpu, &e.w.mtp) else {
                unreachable!(
                    "a session with MTP rows has MTP buffers on an engine that uploaded blk.40"
                )
            };
            head_rows = g.head_rows;
            let rec = Rec {
                e,
                g,
                w,
                work: p.work,
                part: self.part.addr,
            };
            r.tag(TAG_EMBED);
            // The MTP slots' moves first: the pass below reads the table they leave.
            moves = e
                .mtp_slots
                .as_ref()
                .map(|s| s.record(e, &mut r))
                .unwrap_or_default();
            if let Some(s) = &plan.slide {
                g.record_slide(&mut r, s, p.work, g.catch.temp_k, g.catch.temp_v);
            }
            rec.catch_up(&mut r, plan, g.ring_rows, base_row);
            r.tag(d.n_layer as u32);
            rec.draft_rows(
                &mut r,
                k,
                base_row + c,
                c,
                (x_pos - 1) % g.ring_rows,
                embd,
                want_logits,
            );
            r.copy(g.ids.sub(0, 4 * k as u64), p.out.slice(0, 4 * k as u64));
            r.host_barrier();
        }
        e.time(CpuPhase::Record, t0);
        let failed = if std::mem::take(&mut self.fail_mtp_submit) {
            // As a failed submit: the recorded buffer is dropped unsubmitted.
            drop(r);
            Some(vk::Error::Timeout)
        } else {
            e.submit_step(r).err()
        };
        if let Some(source) = failed {
            // The GPU may still be writing Scratch, the work area, `part` and the MTP's KV (`decode.rs`): unusable
            // until reset, the rows and the ring cleared. The MTP slots and their table entries may be anywhere too.
            if let Some(m) = &mut self.mtp {
                m.on_draft_failed();
            }
            if let Some(s) = &e.mtp_slots {
                s.failed(e);
            }
            self.usable = false;
            return Err(step(source));
        }
        if let Some(s) = &e.mtp_slots {
            s.landed(e, moves);
        }
        let t0 = Instant::now();
        let mut st = [0u8; STATUS_BYTES];
        e.status.read(0, &mut st);
        let st = read_status(&st);
        if let Some((layer, expert)) = st.bad {
            self.mtp_poisoned();
            return Err(Error::BadSlotEntry {
                pos: x_pos,
                layer,
                expert,
            });
        }
        if let Some((layer, expert)) = st.cpu {
            self.mtp_poisoned();
            return Err(Error::ExpertOnCpu {
                pos: x_pos,
                layer,
                expert,
            });
        }
        let mut raw = vec![0u8; 4 * k];
        p.out.read(0, &mut raw);
        let mut drafts = Vec::with_capacity(k);
        for c4 in raw.as_chunks::<4>().0 {
            let id = u32::from_le_bytes(*c4);
            if id == NO_DRAFT {
                // That row's input was poisoned: the rows go, the ring (the main model's) stays.
                self.mtp_poisoned();
                break;
            }
            assert!(
                (id as usize) < head_rows,
                "argmax_rows picked {id} of {head_rows} head rows"
            );
            drafts.push(id);
        }
        // The routes of the rows that drafted (the rest, past a poisoned row, read garbage).
        let ids = self
            .mtp_gpu
            .as_ref()
            .and_then(|g| g.routes.as_ref())
            .map(|log| log.read_f32(0, drafts.len() * d.n_expert_used));
        if let (Some(s), Some(ids)) = (&e.mtp_slots, &ids) {
            s.count(e, ids);
        }
        if let (Some(path), Some(ids)) = (mtp_routes_path(), &ids) {
            let n = d.n_expert_used;
            let pass = ROUTE_PASSES.fetch_add(1, Ordering::Relaxed);
            let text: String = ids
                .chunks(n)
                .enumerate()
                .map(|(j, row)| route_line(pass, x_pos + j, j, row))
                .collect();
            append_log(path, MTP_ROUTES_ENV, &text);
        }
        if let (Some(path), Some(log)) = (
            mtp_topk_path(),
            self.mtp_gpu.as_ref().and_then(|g| g.topk.as_ref()),
        ) {
            let text: String = (0..drafts.len())
                .map(|j| {
                    let row = log.read_f32((j * head_rows * 4) as u64, head_rows);
                    topk_line(x_pos + j, j, &row)
                })
                .collect();
            append_log(path, MTP_TOPK_ENV, &text);
        }
        let logits = want_logits.then(|| e.logits.read_f32(0, head_rows));
        e.time(CpuPhase::Read, t0);
        Ok((drafts, logits))
    }

    fn mtp_poisoned(&mut self) {
        if let Some(m) = &mut self.mtp {
            m.on_poisoned();
        }
    }

    /// The prefill call's window: plans the MTP's rows for the call's last `MtpOptions::window` positions and
    /// updates the rows and the ring beside it. None without the MTP. Called after the call's validation; a failure
    /// later in the call resets the state (`run_prefill`).
    pub(crate) fn begin_window(&mut self, e: &Engine, n: usize) -> Option<WindowPlan> {
        let o = e.mtp?;
        let m = self.mtp.as_mut()?;
        let end = self.pos + n;
        let start = self.pos.max(end.saturating_sub(o.window));
        let plan = m.plan_prefill_call(self.pos, start..end);
        if let (Some(p), Some((_, have))) = (&plan, &mut self.capture) {
            have[self.pos..].fill(false);
            // The window records its rows and the one before them (the chunk's own, or the previous chunk's last), unless
            // that one is the ring's.
            let from = if p.first_from_ring {
                p.first_row
            } else {
                p.first_row.saturating_sub(1)
            };
            have[from..p.end].fill(true);
        }
        plan
    }

    /// The window rows of one prefill chunk (positions `c0 .. c0 + t`), recorded in its last layer's command buffer
    /// after the layer's output is in `Prefill::x`: the slide if this chunk holds the window's first row, the
    /// final norm of the rows `h` comes from, the carry (and the ring, on the call's last chunk), and the KV-only rows
    /// through prefill's kernels (`dense`, `rmsnorm_tokens`, `qk_norm_rope_tokens`, `kv_write_tokens`).
    pub(crate) fn record_window(
        &self,
        e: &Engine,
        r: &mut vk::Recorder,
        plan: &WindowPlan,
        c0: usize,
        t: usize,
    ) {
        let (Some(g), Some(w)) = (&self.mtp_gpu, &e.w.mtp) else {
            unreachable!("the window was planned for a session with MTP buffers on an engine that uploaded blk.40")
        };
        r.tag(TAG_MTP_WINDOW);
        let (d, k, p) = (&e.dims, &e.k, &e.prefill);
        let (eps, n_embd) = (d.rms_eps, d.n_embd);
        let rb = 4 * n_embd as u64;
        let cw = plan.chunk(c0, t, g.ring_rows);
        let rec = Rec {
            e,
            g,
            w,
            work: p.work,
            part: self.part.addr,
        };
        let reg = RowRegions {
            t: cw.n_rows,
            ..g.window.rows.clone()
        };
        // The chunk's last layer has finished.
        r.barrier();
        if let Some(s) = &cw.slide {
            g.record_slide(r, s, p.work, g.window.temp_k, g.window.temp_v);
        }
        let nf = t - cw.a;
        k.rmsnorm_tokens(
            r,
            p.x.addr + cw.a as u64 * rb,
            e.w.output_norm,
            rec.at(reg.proj),
            n_embd,
            eps,
            nf,
        );
        if cw.n_rows > 0 {
            k.embed_rows(
                r,
                p.rows.addr() + 8 * cw.lo as u64,
                e.host.embd_ty,
                n_embd,
                rec.at(reg.normed),
                cw.n_rows,
            );
        }
        r.barrier();
        if cw.n_rows > 0 {
            k.rmsnorm_tokens(
                r,
                rec.at(reg.normed),
                w.enorm,
                rec.at(reg.e),
                n_embd,
                eps,
                cw.n_rows,
            );
            // g row i takes h of the position before its own: the chunk's final-norm row i (or i - 1 when the first
            // row's predecessor is the carry's or the ring's).
            match cw.prev {
                PrevH::Chunk => k.rmsnorm_tokens(
                    r,
                    rec.at(reg.proj),
                    w.hnorm,
                    rec.at(reg.g),
                    n_embd,
                    eps,
                    cw.n_rows,
                ),
                prev => {
                    let src = match prev {
                        PrevH::Carry => g.carry.addr,
                        PrevH::Ring(slot) => g.ring.addr + slot as u64 * rb,
                        PrevH::Chunk => unreachable!("handled above"),
                    };
                    k.rmsnorm(r, src, w.hnorm, rec.at(reg.g), n_embd, eps);
                    if cw.n_rows > 1 {
                        k.rmsnorm_tokens(
                            r,
                            rec.at(reg.proj),
                            w.hnorm,
                            rec.at(reg.g) + rb,
                            n_embd,
                            eps,
                            cw.n_rows - 1,
                        );
                    }
                }
            }
        }
        r.barrier();
        // The chunk's last final-norm row is the next chunk's `h`, and on the call's last chunk the ring's.
        let last = p.work.sub(reg.proj + (nf as u64 - 1) * rb, rb);
        r.copy(last, g.carry);
        if let Some((buf, _)) = &self.capture {
            // Tests only: the rows the window computed, for the reference to be fed (`window_hidden`).
            r.copy(
                p.work.sub(reg.proj, nf as u64 * rb),
                buf.slice((c0 + cw.a) as u64 * rb, nf as u64 * rb),
            );
        }
        if cw.last {
            g.copy_to_ring(r, last, plan.end - 1, 1, n_embd);
        }
        if cw.n_rows > 0 {
            rec.concat(r, &reg, cw.n_rows);
        }
        r.barrier();
        if cw.n_rows > 0 {
            let dense = |r: &mut vk::Recorder, w: MatW, x: u64, y: u64| {
                crate::prefill::dense(e, r, w, x, y, cw.n_rows)
            };
            rec.kv_tail(
                r,
                &reg,
                p.rope.addr() + 4 * (d.n_rot * cw.lo) as u64,
                cw.row0,
                &dense,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefill::{prefill_bytes, prefill_sizes};

    #[test]
    fn the_route_log_line_and_switch() {
        assert_eq!(
            route_line(3, 1200, 1, &[5.0, 255.0, 0.0, 17.0, 9.0, 8.0, 7.0, 6.0]),
            "{\"pass\":3,\"pos\":1200,\"j\":1,\"ids\":[5,255,0,17,9,8,7,6]}
"
        );
        assert_eq!(route_log_path(None), None);
        assert_eq!(route_log_path(Some("".into())), None);
        assert_eq!(
            route_log_path(Some("r.jsonl".into())),
            Some(PathBuf::from("r.jsonl"))
        );
    }

    #[test]
    fn topk_line_orders_by_logit_then_id() {
        let line = topk_line(40, 2, &[1.0, 3.0, 3.0, 2.0]);
        let head = "{\"pos\":40,\"depth\":2,\"ids\":[1,2,3],\"logits\":[3,3,2],\"lse\":";
        assert!(line.starts_with(head), "{line}");
        let lse: f64 = line[head.len()..line.find(",\"rows\"").unwrap()]
            .parse()
            .unwrap();
        let want = (1f64.exp() + 2.0 * 3f64.exp() + 2f64.exp()).ln();
        assert!((lse - want).abs() < 1e-9, "lse {lse}, want {want}");
        assert!(
            line.ends_with(
                ",\"rows\":4}
"
            ),
            "{line}"
        );
        // Every id of a flat row ties: the lowest three, in order.
        let flat = top_n(&[0.5; 10], 3);
        assert_eq!(flat.iter().map(|t| t.0).collect::<Vec<_>>(), [0, 1, 2]);
    }

    #[test]
    fn the_topk_log_is_off_when_unset_or_empty() {
        assert_eq!(route_log_path(None), None);
        assert_eq!(route_log_path(Some("".into())), None);
        assert_eq!(MTP_TOPK_ENV, "ARTEMIS_MTP_TOPK");
        // This process never sets it: the log's switch is off and a pick line is a no-op.
        assert!(mtp_topk_path().is_none());
        log_pick(7, 9);
    }

    fn rows_of(s: &MtpState) -> Vec<u32> {
        s.rows.positions().to_vec()
    }

    #[test]
    fn the_slide_keeps_the_newest_half_and_rollback_follows_it() {
        let mut r = MtpRows::new(512);
        let mut slides = Vec::new();
        for pos in 0..600u32 {
            if let Some(s) = r.make_room(1) {
                slides.push((pos, s));
            }
            r.push(pos);
        }
        // The 513th pass (position 512) is the first that doesn't fit: the newest 256 rows move to the front.
        assert_eq!(
            slides,
            [(
                512,
                Slide {
                    from_row: 256,
                    rows: 256
                }
            )]
        );
        // The buffer holds positions 256 ..= 599.
        assert_eq!(r.len(), 256 + 88);
        assert_eq!(r.positions()[0], 256);
        assert_eq!(r.last_position(), Some(599));
        // A rollback by position drops the right rows.
        r.cut(550);
        assert_eq!(r.last_position(), Some(549));
        assert_eq!(r.len(), 550 - 256);
        r.cut(100);
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn a_multi_row_pass_near_the_cap_slides_without_overlap() {
        let mut r = MtpRows::new(512);
        for p in 0..500u32 {
            r.push(p);
        }
        let mut next = 500u32;
        let mut slide = None;
        // Passes of 5 rows: 3 catch-up, x and one chained (dropped after, so a pass adds 4).
        while slide.is_none() {
            let len = r.len();
            slide = r.make_room(5);
            if slide.is_some() {
                assert_eq!(len, 508);
            }
            for _ in 0..4 {
                r.push(next);
                next += 1;
            }
        }
        let s = slide.unwrap();
        assert_eq!(
            s,
            Slide {
                from_row: 252,
                rows: 256
            }
        );
        // 508 + 5 > 512; after the slide the whole pass fits, chained row included.
        assert_eq!(r.len(), 256 + 4);
        assert!(r.len() - 4 + 5 <= r.cap());
        // One direct copy would have overlapped (the source 252.. against the destination 0..256); two do not.
        assert!(s.from_row < s.rows);
        let c = s.copies();
        for copy in &c {
            let disjoint = copy.src.1.end <= copy.dst.1.start || copy.dst.1.end <= copy.src.1.start;
            assert!(copy.src.0 != copy.dst.0 || disjoint, "{copy:?}");
            assert_eq!(copy.src.1.len(), 256);
            assert_eq!(copy.dst.1.len(), 256);
        }
        assert_eq!(c[0].src, (RowBuf::Kv, 252..508));
        assert_eq!(c[0].dst, (RowBuf::Temp, 0..256));
        assert_eq!(c[1].src, (RowBuf::Temp, 0..256));
        assert_eq!(c[1].dst, (RowBuf::Kv, 0..256));
        // The positions stay in order across the slide; the chained row is never pushed, so its drop changes nothing.
        assert!(r.positions().windows(2).all(|w| w[0] < w[1]));
        assert_eq!(r.positions()[0], 252);
    }

    #[test]
    fn a_pass_that_cannot_fit_after_the_slide_asserts() {
        let mut r = MtpRows::new(512);
        for p in 0..300u32 {
            r.push(p);
        }
        // 300 + 300 > 512 and the newest 256 + 300 still doesn't fit: the sizing rules keep n under cap / 2.
        let out = std::panic::catch_unwind(move || r.make_room(300));
        let msg = out.unwrap_err();
        let msg = msg.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(msg.contains("does not fit"), "{msg}");
    }

    #[test]
    fn a_restore_clears_the_mtp_rows_and_the_ring() {
        let mut s = MtpState::new(512, 16);
        s.on_prefill(0..10);
        for p in 10..14 {
            s.on_decode(p);
        }
        let toks: Vec<u32> = (0..14).collect();
        let plan = s.plan(&toks, 2);
        s.on_draft(&plan);
        assert!(!rows_of(&s).is_empty());
        s.on_reset();
        assert_eq!(s.rows.len(), 0);
        assert!(!s.ring.holds(13) && !s.ring.holds(9));
        // The suffix's window starts at the call's first position: its first row needs the carry it doesn't have.
        assert_eq!(s.on_prefill(40..45), None);
        assert_eq!(rows_of(&s), [41, 42, 43, 44]);
        assert!(s.ring.holds(44));
        for p in 0..44 {
            assert!(!s.ring.holds(p), "{p}");
        }
        assert!(!s.ready(40) && s.ready(45));
    }

    #[test]
    fn a_window_inside_the_call_has_its_first_row() {
        // The call started at 0; the window is its last 5 positions: h of 39 came from the chunk's carry.
        let mut s = MtpState::new(512, 16);
        s.on_prefill_call(0, 40..45);
        assert_eq!(rows_of(&s), [40, 41, 42, 43, 44]);
        // Position 0 never has a row, and a call of one token has none.
        let mut s = MtpState::new(512, 16);
        s.on_prefill(0..3);
        assert_eq!(rows_of(&s), [1, 2]);
        let mut s = MtpState::new(512, 16);
        s.on_prefill(7..8);
        assert_eq!(s.rows.len(), 0);
        assert!(s.ring.holds(7));
    }

    #[test]
    fn a_live_continuation_prefill_near_the_cap_slides_first() {
        // A live session: 500 rows (positions 1 ..= 500) from an earlier request, the ring holding h of position 500.
        let mut s = MtpState::new(512, 16);
        s.on_prefill(0..501);
        assert_eq!(s.rows.len(), 500);
        // The follow-up request prefills a suffix at 501 with a window of 200 positions.
        let slide = s.on_prefill_call(501, 501..701);
        // The ring held h_500, so position 501 gets its row (200 rows); 500 + 200 > 512 slid first, keeping 256.
        assert_eq!(
            slide,
            Some(Slide {
                from_row: 244,
                rows: 256
            })
        );
        assert_eq!(s.rows.len(), 256 + 200);
        assert_eq!(s.rows.positions()[0], 245);
        assert_eq!(s.rows.last_position(), Some(700));
        assert!(s.rows.len() <= s.rows.cap());
        assert!(s.rows.positions().windows(2).all(|w| w[0] < w[1]));
        assert!(s.ring.holds(700));
        // Without the ring's h for the window's first position, that row stays a hole.
        let mut t = MtpState::new(512, 16);
        t.on_prefill(0..501);
        t.ring.clear();
        t.on_prefill_call(501, 501..511);
        assert_eq!(t.rows.last_position(), Some(510));
        assert!(!t.rows.positions().contains(&501));
        assert!(t.rows.positions().contains(&502));
        // Rows in [call_start, window.start) are stale too: a call from 400 with a window from 450 drops 400 .. 450.
        let mut u = MtpState::new(512, 16);
        u.on_prefill(0..501);
        u.on_prefill_call(400, 450..460);
        let mut want: Vec<u32> = (1..400).collect();
        want.extend(450..460);
        assert_eq!(rows_of(&u), want);
    }

    #[test]
    fn the_mtp_rows_follow_decode_verify_commit_and_draft() {
        let mut s = MtpState::new(512, 16);
        s.on_prefill(0..10);
        assert_eq!(rows_of(&s), (1..=9).collect::<Vec<u32>>());
        // Plain decodes of positions 10 and 11.
        s.on_decode(10);
        s.on_decode(11);
        // A lookup verify of 4 at 12 keeping 2.
        s.on_commit(12, 2);
        assert!(s.ring.holds(9) && s.ring.holds(13) && !s.ring.holds(14));
        // A draft pass at 14 with k = 3: catch-up 10 ..= 13, x's row 14; the chained rows 15 and 16 are never rows.
        let toks: Vec<u32> = (0..14).map(|p| p + 100).collect();
        let plan = s.plan(&toks, 3);
        assert_eq!(plan.chained, [15, 16]);
        s.on_draft(&plan);
        assert_eq!(rows_of(&s), (1..=14).collect::<Vec<u32>>());
        // The verify of [x, d1, d2] at 14 keeps 1: d1 and d2 were rejected, x's row stays.
        s.on_commit(14, 1);
        assert_eq!(rows_of(&s), (1..=14).collect::<Vec<u32>>());
        // A plain decode at 15, then a draft pass at 16 with k = 2: catch-up 15, x 16.
        s.on_decode(15);
        let toks: Vec<u32> = (0..16).map(|p| p + 100).collect();
        let plan = s.plan(&toks, 2);
        assert_eq!(plan.catch_up, [(15, 115)]);
        s.on_draft(&plan);
        assert_eq!(rows_of(&s), (1..=16).collect::<Vec<u32>>());
        // A verify of 4 at 16 keeping 3 adds no rows; the next pass catches up on 17 and 18 from the ring.
        s.on_commit(16, 3);
        assert_eq!(rows_of(&s), (1..=16).collect::<Vec<u32>>());
        let toks: Vec<u32> = (0..19).map(|p| p + 100).collect();
        let plan = s.plan(&toks, 1);
        assert_eq!(plan.catch_up, [(17, 117), (18, 118)]);
        assert_eq!(plan.x_pos, 19);
        assert!(plan.chained.is_empty());
        s.on_draft(&plan);
        assert_eq!(rows_of(&s), (1..=19).collect::<Vec<u32>>());
        // Keeping only x cuts everything above it: nothing past 19 ever was a row.
        s.on_commit(19, 1);
        assert_eq!(s.rows.last_position(), Some(19));
    }

    #[test]
    fn plan_draft_catches_up_on_every_position_the_ring_holds() {
        let mut rows = MtpRows::new(512);
        for p in 1..=9u32 {
            rows.push(p);
        }
        let mut ring = MtpRing::new(16);
        ring.record(9, 5); // h_9 ..= h_13
        let toks: Vec<u32> = (0..14).map(|p| p * 3).collect();
        let plan = plan_draft(&mut rows, &ring, &toks, 4);
        assert_eq!(plan.slide, None);
        assert_eq!(plan.catch_up, [(10, 30), (11, 33), (12, 36), (13, 39)]);
        assert_eq!(plan.x_pos, 14);
        assert_eq!(plan.chained, [15, 16, 17]);
        assert_eq!(plan.rows_written(), 4 + 1 + 3);
    }

    #[test]
    fn positions_older_than_the_ring_stay_holes() {
        let mut rows = MtpRows::new(512);
        for p in 1..=9u32 {
            rows.push(p);
        }
        // 40 plain steps since the last row: positions 10 ..= 49 were decoded, x is at 50.
        let mut ring = MtpRing::new(16);
        for p in 10..50 {
            ring.record(p, 1);
        }
        let toks: Vec<u32> = (0..50).collect();
        let plan = plan_draft(&mut rows, &ring, &toks, 1);
        // The ring holds h_34 ..= h_49: catch-up rows q = 35 ..= 49 (15 rows), then x's row at 50.
        let qs: Vec<u32> = plan.catch_up.iter().map(|c| c.0).collect();
        assert_eq!(qs, (35..50).collect::<Vec<u32>>());
        assert_eq!(plan.x_pos, 50);
        let mut s = MtpState::new(512, 16);
        s.rows = rows;
        s.ring = ring;
        s.on_draft(&plan);
        // Positions 10 ..= 34 (25 of them) stay holes.
        assert_eq!(
            (10..35).filter(|q| s.rows.positions().contains(q)).count(),
            0
        );
        assert_eq!(s.rows.len(), 9 + 15 + 1);
    }

    #[test]
    fn the_ring_extends_cuts_and_wraps() {
        let mut r = MtpRing::new(16);
        r.record(5, 3);
        r.record(8, 2);
        assert!((5..10).all(|p| r.holds(p)) && !r.holds(4) && !r.holds(10));
        // A cut at 7 drops 7 and up before the new run.
        r.record(7, 1);
        assert!(r.holds(6) && r.holds(7) && !r.holds(8));
        // A gap replaces.
        r.record(20, 2);
        assert!(!r.holds(7) && r.holds(20) && r.holds(21));
        // A run longer than the ring keeps its newest cap.
        r.record(100, 40);
        assert!(!r.holds(123) && r.holds(124) && r.holds(139) && !r.holds(140));
        assert_eq!(r.slot(139), 11);
        r.clear();
        assert!(!r.holds(139));
    }

    #[test]
    fn a_failed_step_cuts_the_rows_to_its_start() {
        let mut s = MtpState::new(512, 16);
        s.on_prefill(0..20);
        s.on_decode(20);
        s.on_failed(12);
        assert_eq!(rows_of(&s), (1..=11).collect::<Vec<u32>>());
        assert!(!s.ring.holds(20) && !s.ring.holds(19));
        assert!(!s.ready(21));
    }

    #[test]
    fn a_failed_draft_clears_rows_and_ring() {
        let mut s = MtpState::new(512, 16);
        s.on_prefill(0..20);
        s.on_draft_failed();
        assert_eq!(s.rows.len(), 0);
        assert!(!s.ring.holds(19));
    }

    #[test]
    fn a_poisoned_draft_clears_the_rows_and_the_next_pass_catches_up() {
        let mut s = MtpState::new(512, 16);
        s.on_prefill(0..10);
        s.on_decode(10);
        let toks: Vec<u32> = (0..11).collect();
        let plan = s.plan(&toks, 2);
        s.on_draft(&plan);
        assert!(s.rows.len() > 0);
        // NO_DRAFT at depth 1: no rows, the ring kept.
        s.on_poisoned();
        assert_eq!(s.rows.len(), 0);
        assert!(s.ring.holds(10) && s.ring.holds(9));
        // The next step decodes x (position 11); the next pass catches up from the ring: positions 10 and 11, then x.
        s.on_decode(11);
        let toks: Vec<u32> = (0..12).collect();
        let plan = s.plan(&toks, 1);
        let qs: Vec<u32> = plan.catch_up.iter().map(|c| c.0).collect();
        assert_eq!(qs, [10, 11]);
        assert_eq!(plan.x_pos, 12);
    }

    #[test]
    fn a_plan_that_slides_applies_it_before_the_pushes() {
        let mut s = MtpState::new(512, 16);
        s.on_prefill(0..510);
        assert_eq!(s.rows.len(), 509);
        let toks: Vec<u32> = (0..510).collect();
        let plan = s.plan(&toks, 4);
        assert_eq!(plan.slide.map(|x| (x.from_row, x.rows)), Some((253, 256)));
        s.on_draft(&plan);
        assert_eq!(s.rows.len(), 257);
        assert!(s.rows.len() + plan.chained.len() <= s.rows.cap());
    }

    #[test]
    fn the_layouts_fit_in_prefills_work_at_the_production_chunk() {
        let d = Dims::QWEN36;
        let work = prefill_sizes(&d, 2048, Some(65_536))[8];
        assert_eq!(work >> 20, 225);
        let m = MtpLayout::new(&d, 8192);
        // One 16-row catch-up region and the slide's 8 MiB temp.
        assert_eq!(m.catch_up.t, 16);
        assert_eq!(m.temp_v - m.temp_k, 4 << 20);
        assert_eq!(m.bytes - m.temp_k, 8 << 20);
        assert!(m.bytes < work, "{}", m.bytes);
        let w = WindowLayout::new(&d, 2048, 8192);
        assert!(w.bytes < work, "{} MiB of {}", w.bytes >> 20, work >> 20);
        assert!(w.bytes < 225 << 20);
        // The regions are 256-aligned and disjoint, in order.
        let r = &w.rows;
        let offs = [r.e, r.g, r.cat, r.proj, r.normed, r.k, r.v];
        assert!(offs.windows(2).all(|p| p[0] < p[1]));
        assert!(offs.iter().all(|o| o % 256 == 0));
        assert!(r.v < r.bytes && r.bytes <= w.temp_k);
        assert_eq!(prefill_bytes(&d, 2048, Some(65_536)) >> 20, 414);
    }

    fn tiny_opts(rows: usize, ring: usize) -> MtpOptions {
        MtpOptions {
            rows,
            ring,
            window: rows / 2,
            head_rows: 0,
        }
    }

    #[test]
    fn ring_runs_split_at_the_wrap_and_keep_the_newest_cap() {
        assert_eq!(ring_runs(16, 3, 4), [(3, 4, 0)]);
        assert_eq!(ring_runs(16, 14, 4), [(14, 2, 0), (0, 2, 2)]);
        assert_eq!(ring_runs(16, 32, 16), [(0, 16, 0)]);
        assert_eq!(ring_runs(16, 0, 0), []);
        // 40 rows from position 0: only the newest 16 (positions 24 ..= 39) are kept, rows 24 ..= 39 of the source.
        assert_eq!(ring_runs(16, 0, 40), [(8, 8, 24), (0, 8, 32)]);
        // Every kept row lands in the slot of its position, once.
        let mut seen = [false; 16];
        for (slot, len, off) in ring_runs(16, 5, 40) {
            for i in 0..len {
                let pos = 5 + off + i;
                assert_eq!(slot + i, pos % 16);
                assert!(!seen[slot + i]);
                seen[slot + i] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn a_call_in_chunks_gives_every_window_row_its_h() {
        // A call at position 0 of 40 tokens in chunks of 16, 16 and 8; the window is the whole call.
        let mut s = MtpState::new(512, 16);
        let plan = s.plan_prefill_call(0, 0..40).unwrap();
        assert_eq!(
            (plan.first_row, plan.end, plan.base, plan.slide),
            (1, 40, 0, None)
        );
        assert!(!plan.first_from_ring);
        let c = plan.chunk(0, 16, 16);
        // Positions 1 ..= 15 get rows; row 1's predecessor (position 0) is in the chunk.
        assert_eq!((c.n_rows, c.lo, c.a, c.row0), (15, 1, 0, 0));
        assert_eq!((c.prev, c.slide, c.last), (PrevH::Chunk, None, false));
        // The second chunk's first row takes the carry, and its row follows the first chunk's 15.
        let c = plan.chunk(16, 16, 16);
        assert_eq!((c.n_rows, c.lo, c.a, c.row0), (16, 0, 0, 15));
        assert_eq!((c.prev, c.last), (PrevH::Carry, false));
        let c = plan.chunk(32, 8, 16);
        assert_eq!((c.n_rows, c.row0, c.last), (8, 31, true));
        assert_eq!(s.rows.len(), 39);
        assert_eq!(s.rows.positions()[31], 32);
    }

    #[test]
    fn a_window_inside_the_call_skips_the_chunks_before_it() {
        let mut s = MtpState::new(512, 16);
        // 40 tokens, the window the last 10: rows 30 ..= 39.
        let plan = s.plan_prefill_call(0, 30..40).unwrap();
        assert_eq!((plan.first_row, plan.base), (30, 0));
        // The first chunk has none: only its last row's final norm is computed, for the carry.
        let c = plan.chunk(0, 16, 16);
        assert_eq!(
            (c.n_rows, c.a, c.prev, c.slide),
            (0, 15, PrevH::Chunk, None)
        );
        // The second holds rows 30 and 31 at its end; their h comes from its own rows 13 and 14 (a = lo - 1).
        let c = plan.chunk(16, 16, 16);
        assert_eq!(
            (c.n_rows, c.lo, c.a, c.row0, c.prev),
            (2, 14, 13, 0, PrevH::Chunk)
        );
        // The third starts in the window: the carry, and row 2 of the window.
        let c = plan.chunk(32, 8, 16);
        assert_eq!(
            (c.n_rows, c.lo, c.a, c.row0, c.prev),
            (8, 0, 0, 2, PrevH::Carry)
        );
        assert!(c.last);
    }

    #[test]
    fn a_live_continuation_slides_in_its_first_chunk_and_reads_the_ring() {
        let mut s = MtpState::new(512, 16);
        s.on_prefill(0..501);
        let plan = s.plan_prefill_call(501, 501..701).unwrap();
        assert!(plan.first_from_ring);
        assert_eq!(
            plan.slide,
            Some(Slide {
                from_row: 244,
                rows: 256
            })
        );
        assert_eq!(plan.base, 256);
        // Chunks of 100: the first reads h of 500 from its ring slot and slides; the second takes the carry.
        let c = plan.chunk(501, 100, 16);
        assert_eq!((c.n_rows, c.lo, c.a, c.row0), (100, 0, 0, 256));
        assert_eq!(c.prev, PrevH::Ring(500 % 16));
        assert_eq!(c.slide, plan.slide);
        let c = plan.chunk(601, 100, 16);
        assert_eq!(
            (c.row0, c.prev, c.slide, c.last),
            (356, PrevH::Carry, None, true)
        );
        // Without the ring's h for position 500, the call's first position stays a hole and its second row starts it.
        let mut s = MtpState::new(512, 16);
        s.on_prefill(0..501);
        s.ring.clear();
        let plan = s.plan_prefill_call(501, 501..511).unwrap();
        assert!(!plan.first_from_ring);
        let c = plan.chunk(501, 10, 16);
        assert_eq!(
            (c.n_rows, c.lo, c.a, c.prev, c.row0),
            (9, 1, 0, PrevH::Chunk, 500)
        );
    }

    #[test]
    fn a_one_token_call_has_no_rows_and_still_feeds_the_ring() {
        let mut s = MtpState::new(512, 16);
        let plan = s.plan_prefill_call(7, 7..8).unwrap();
        let c = plan.chunk(7, 1, 16);
        assert_eq!((c.n_rows, c.a, c.last), (0, 0, true));
        assert!(s.ring.holds(7));
        assert_eq!(s.rows.len(), 0);
        // An empty window plans nothing.
        assert_eq!(s.plan_prefill_call(8, 8..8), None);
    }

    #[test]
    fn the_load_refuses_a_chunk_or_work_area_too_small_for_the_mtp() {
        let d = Dims::tiny();
        // The ring and a verify's worth of rows must fit one chunk's RoPE tables.
        assert!(check_prefill(&d, 144, Some(512), tiny_opts(512, 128)).is_ok());
        assert!(matches!(
            check_prefill(&d, 143, Some(512), tiny_opts(512, 128)),
            Err(Error::SpeculationRefused { .. })
        ));
        // The production load: a chunk of 2,048, 8,192 rows and a ring of 256.
        let real = Dims::QWEN36;
        assert!(check_prefill(&real, 2048, Some(65_536), tiny_opts(8192, 256)).is_ok());
        // A work area sized for a short chunk can't hold a 65,536-row slide's temp.
        assert!(matches!(
            check_prefill(&d, 144, Some(512), tiny_opts(65_536, 128)),
            Err(Error::SpeculationRefused { .. })
        ));
    }

    #[test]
    fn the_head_rows_are_the_full_head_for_zero_or_more_than_the_vocabulary() {
        let v = 248_320;
        assert_eq!(head_rows(0, v), v);
        assert_eq!(head_rows(256, v), 256);
        assert_eq!(head_rows(v, v), v);
        assert_eq!(head_rows(v + 1, v), v);
        assert_eq!(head_rows(65_536, 1000), 1000);
    }

    #[test]
    fn the_ring_is_cut_to_the_prefill_chunk_minus_16_in_steps_of_16() {
        // A ring of 2,048 in the production chunk of 2,048 loads with 2,032; a small ring is left alone.
        assert_eq!(clamp_ring(2048, 2048), 2032);
        assert_eq!(clamp_ring(256, 2048), 256);
        assert_eq!(clamp_ring(128, 100), 80);
        assert_eq!(clamp_ring(128, 144), 128);
        // Under 32 positions of chunk it stays 16, and `check_prefill` then refuses the load.
        assert_eq!(clamp_ring(128, 20), 16);
        let d = Dims::tiny();
        assert!(matches!(
            check_prefill(&d, 20, Some(512), tiny_opts(512, 16)),
            Err(Error::SpeculationRefused { .. })
        ));
        assert!(check_prefill(&d, 32, Some(512), tiny_opts(512, 16)).is_ok());
    }

    #[test]
    fn the_slide_records_the_two_copies_of_task_7_through_the_temp() {
        // `record_slide` maps `Slide::copies` onto K, V and the work area's temps; the planner's copies are the
        // contract: K and V both go through, each copy's source and destination in different buffers.
        let s = Slide {
            from_row: 252,
            rows: 256,
        };
        let [a, b] = s.copies();
        assert_eq!((a.src.0, a.dst.0), (RowBuf::Kv, RowBuf::Temp));
        assert_eq!((b.src.0, b.dst.0), (RowBuf::Temp, RowBuf::Kv));
    }
}
