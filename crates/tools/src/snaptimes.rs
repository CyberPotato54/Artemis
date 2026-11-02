//! The snapshot copy measurement's pure side: where prompt 20 is cut, the
//! estimates the copies are set against, the snapshots' cost against a plain prefill, and the record
//! `artemis snapshot-times` writes. The command in `main.rs` drives the engine and fills it in.
use serde::{Deserialize, Serialize};

/// The record's format name.
pub const FORMAT: &str = "artemis-snapshot-times-1";
/// `<|im_start|>`, whose last occurrence in a rendered chat prompt starts the generation-prompt suffix.
pub const IM_START: u32 = 248_045;
/// The measured PCIe rate on 2026-10-18, GB/s.
pub const LINK_NOW: f64 = 28.43;
/// And its rate on 2026-10-13.
pub const LINK_BEFORE: f64 = 14.22;

/// The index of the last `<|im_start|>`: the prompt-end cut of a chat prompt rendered with the generation prompt.
/// `None` for a prompt with no chat turn.
pub fn last_im_start(tokens: &[u32]) -> Option<usize> {
    tokens.iter().rposition(|&t| t == IM_START)
}

/// Milliseconds to copy `bytes` at `gb_per_s` (10⁹ bytes a second).
pub fn estimate_ms(bytes: u64, gb_per_s: f64) -> f64 {
    bytes as f64 / (gb_per_s * 1e9) * 1e3
}

/// What the overlapped run costs over the plain one, in percent: `100 × (overlapped − plain) / plain`. Negative when
/// the overlapped run was faster.
pub fn cost_pct(plain_s: f64, overlapped_s: f64) -> f64 {
    100.0 * (overlapped_s - plain_s) / plain_s
}

/// The link's rates in GB/s, as `engine::LinkRates` measures them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LinkRatesJson {
    pub cached_d2h: f64,
    pub cached_h2d: f64,
    pub pinned_d2h: f64,
    pub pinned_h2d: f64,
    pub cached_cpu_read: f64,
    pub pinned_cpu_read: f64,
}

impl From<engine::LinkRates> for LinkRatesJson {
    fn from(r: engine::LinkRates) -> Self {
        LinkRatesJson {
            cached_d2h: r.cached_d2h,
            cached_h2d: r.cached_h2d,
            pinned_d2h: r.pinned_d2h,
            pinned_h2d: r.pinned_h2d,
            cached_cpu_read: r.cached_cpu_read,
            pinned_cpu_read: r.pinned_cpu_read,
        }
    }
}

/// The snapshots one prefill of prompt 20 takes with the server's points: six intervals (4,096 … 24,576) and the
/// prompt end.
pub const POINTS: usize = 7;

/// What is wrong with a run that didn't take every snapshot point, for the command to fail on after it has written the
/// record: the overlapped run's takes, the waited run's takes and the restored node's chain, each `POINTS` long.
pub fn problems(snapshots: usize, waited: usize, chain_len: usize) -> Vec<String> {
    [
        ("the overlapped run took", snapshots),
        ("the waited run took", waited),
        ("the prompt-end chain has", chain_len),
    ]
    .into_iter()
    .filter(|&(_, n)| n != POINTS)
    .map(|(what, n)| format!("{what} {n} snapshots, not {POINTS}"))
    .collect()
}

/// Whether `pinned` is at least 10% faster than `cached`, the pool's kind being `HostCached` (the case for switching to
/// pinned memory).
pub fn pinned_faster(cached: f64, pinned: f64) -> bool {
    pinned >= cached * 1.1
}

/// The whole-prompt copies: one snapshot of the state and the KV at the prompt's end, and its restores.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Whole {
    pub kv_bytes: u64,
    pub state_bytes: u64,
    pub take_state_ms: f64,
    pub take_kv_ms: f64,
    /// A restore at once after the take: the state alone, the KV already live.
    pub restore_state_only_ms: f64,
    /// A restore after another prompt dirtied the engine from position 0.
    pub restore_after_dirty_ms: f64,
    pub restore_after_dirty_bytes: u64,
    /// The first KV position the dirty restore copied (0 for a full copy).
    pub kv_from: usize,
}

/// One take of the segmented run: the position, the KV bytes of its segment, the two times and the segment's KV copy
/// beside the estimates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Take {
    pub pos: usize,
    pub kv_bytes: u64,
    pub state_ms: f64,
    pub kv_ms: f64,
    pub kv_estimate: Estimate,
}

/// The server's snapshot points: the interval snapshots and the prompt end.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segmented {
    pub snapshots: usize,
    /// The prefill with the snapshots taken, each copy left to overlap the next chunk.
    pub overlapped_prefill_s: f64,
    pub overlapped_tok_s: f64,
    pub cost_vs_plain_pct: f64,
    /// Each take with its KV copy waited for (the copies do not overlap).
    pub waited: Vec<Take>,
    pub restore_after_dirty_ms: f64,
    pub restore_after_dirty_bytes: u64,
    pub chain_len: usize,
}

/// One copy's estimate: its bytes and its time at the two estimated rates and at the measured one.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Estimate {
    pub bytes: u64,
    pub ms_at_28_43: f64,
    pub ms_at_14_22: f64,
    pub measured_gb_per_s: f64,
    pub ms_at_measured: f64,
}

impl Estimate {
    pub fn of(bytes: u64, measured_gb_per_s: f64) -> Estimate {
        Estimate {
            bytes,
            ms_at_28_43: estimate_ms(bytes, LINK_NOW),
            ms_at_14_22: estimate_ms(bytes, LINK_BEFORE),
            measured_gb_per_s,
            ms_at_measured: estimate_ms(bytes, measured_gb_per_s),
        }
    }
}

/// What each copy should cost by the estimated rates, and at this day's measured rate: takes at the device-to-host
/// rate, restores at the host-to-device rate, both of the `HostCached` pool.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Estimates {
    pub take_state: Estimate,
    pub take_kv: Estimate,
    pub restore_state_only: Estimate,
    pub restore_after_dirty: Estimate,
    pub chain_restore_after_dirty: Estimate,
}

impl Estimates {
    pub fn new(link: &LinkRatesJson, whole: &Whole, segmented: &Segmented) -> Estimates {
        Estimates {
            take_state: Estimate::of(whole.state_bytes, link.cached_d2h),
            take_kv: Estimate::of(whole.kv_bytes, link.cached_d2h),
            restore_state_only: Estimate::of(whole.state_bytes, link.cached_h2d),
            restore_after_dirty: Estimate::of(whole.restore_after_dirty_bytes, link.cached_h2d),
            chain_restore_after_dirty: Estimate::of(
                segmented.restore_after_dirty_bytes,
                link.cached_h2d,
            ),
        }
    }
}

/// `artemis snapshot-times --json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Times {
    pub format: String,
    pub link: LinkRatesJson,
    pub pool_limit_bytes: u64,
    pub pool_cut: bool,
    pub prompt_tokens: usize,
    pub cut_at: usize,
    /// The process's first prefill, with its warm-up (kernels, the expert cache, the page cache): never compared.
    pub first_prefill_s: f64,
    pub first_tok_s: f64,
    /// A plain prefill of the same prompt in the same process, no snapshots: what the overlapped run is set against.
    pub plain_prefill_s: f64,
    pub plain_tok_s: f64,
    pub whole: Whole,
    pub segmented: Segmented,
    pub estimates: Estimates,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn tokens(name: &str) -> Vec<u32> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/golden/prompts")
            .join(name);
        let b = std::fs::read(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()));
        assert!(b.len().is_multiple_of(4));
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()
    }

    #[test]
    fn the_cut_is_the_last_im_start() {
        let long = tokens("20-long-option-rs.tokens");
        assert_eq!(long.len(), 27_416);
        assert_eq!(last_im_start(&long), Some(27_409));
        // The 7-token suffix of the thinking-off render: <|im_start|>assistant\n<think>\n\n</think>\n\n
        assert_eq!(long.len() - 27_409, 7);
        // A raw completion has no chat turn.
        assert_eq!(last_im_start(&tokens("01-code.tokens")), None);
        assert_eq!(last_im_start(&[]), None);
        assert_eq!(last_im_start(&[1, IM_START, 2, IM_START, 3]), Some(3));
    }

    #[test]
    fn estimates_match_the_specs_figures() {
        let near = |got: f64, want: f64| {
            assert!((got - want).abs() < 0.1, "{got} ms, want about {want}");
        };
        near(estimate_ms(65_863_680, LINK_NOW), 2.3);
        near(estimate_ms(614_400_000, LINK_NOW), 21.6);
        near(estimate_ms(27_416 * 20_480, LINK_NOW), 19.7);
        near(estimate_ms(65_863_680, LINK_BEFORE), 4.6);
    }

    #[test]
    fn the_snapshot_cost_is_taken_against_the_plain_prefill() {
        assert!((cost_pct(12.8, 13.0) - 1.5625).abs() < 0.01);
        assert!(cost_pct(13.0, 12.8) < 0.0);
        assert_eq!(cost_pct(10.0, 10.0), 0.0);
    }

    #[test]
    fn pinned_must_be_ten_percent_faster_to_count() {
        assert!(!pinned_faster(28.0, 30.0));
        assert!(pinned_faster(28.0, 31.0));
        assert!(!pinned_faster(28.0, 20.0));
    }

    #[test]
    fn a_run_missing_a_snapshot_point_is_a_problem() {
        assert!(problems(7, 7, 7).is_empty());
        let p = problems(6, 7, 5);
        assert_eq!(p.len(), 2, "{p:?}");
        assert!(p[0].contains("overlapped") && p[0].contains("6"));
        assert!(p[1].contains("chain") && p[1].contains("5"));
        assert_eq!(problems(7, 0, 7).len(), 1);
    }

    fn sample() -> Times {
        let link = LinkRatesJson {
            cached_d2h: 26.5,
            cached_h2d: 27.1,
            pinned_d2h: 26.4,
            pinned_h2d: 27.0,
            cached_cpu_read: 9.5,
            pinned_cpu_read: 0.4,
        };
        let whole = Whole {
            kv_bytes: 561_479_680,
            state_bytes: 65_863_680,
            take_state_ms: 3.1,
            take_kv_ms: 22.0,
            restore_state_only_ms: 3.0,
            restore_after_dirty_ms: 25.5,
            restore_after_dirty_bytes: 627_343_360,
            kv_from: 0,
        };
        let segmented = Segmented {
            snapshots: 7,
            overlapped_prefill_s: 13.0,
            overlapped_tok_s: 2108.9,
            cost_vs_plain_pct: cost_pct(12.8, 13.0),
            waited: vec![Take {
                pos: 4096,
                kv_bytes: 4096 * 20_480,
                state_ms: 3.1,
                kv_ms: 3.4,
                kv_estimate: Estimate::of(4096 * 20_480, 26.5),
            }],
            restore_after_dirty_ms: 25.0,
            restore_after_dirty_bytes: 627_000_000,
            chain_len: 7,
        };
        Times {
            format: FORMAT.into(),
            estimates: Estimates::new(&link, &whole, &segmented),
            link,
            pool_limit_bytes: 8 << 30,
            pool_cut: false,
            prompt_tokens: 27_416,
            cut_at: 27_409,
            first_prefill_s: 14.5,
            first_tok_s: 1890.8,
            plain_prefill_s: 12.8,
            plain_tok_s: 2141.9,
            whole,
            segmented,
        }
    }

    #[test]
    fn times_round_trip_through_json() {
        let t = sample();
        let text = serde_json::to_string_pretty(&t).unwrap();
        let back: Times = serde_json::from_str(&text).unwrap();
        assert_eq!(back.format, FORMAT);
        assert_eq!(back.link, t.link);
        assert_eq!(back.whole.kv_bytes, t.whole.kv_bytes);
        assert_eq!(back.segmented.waited.len(), 1);
        assert_eq!(back.segmented.waited[0].kv_estimate.bytes, 4096 * 20_480);
        assert_eq!(back.segmented.chain_len, 7);
        assert_eq!(serde_json::to_string_pretty(&back).unwrap(), text);
        // The estimates follow the bytes: the state alone at the measured host-to-device rate.
        let e = &t.estimates.restore_state_only;
        assert_eq!(e.bytes, 65_863_680);
        assert!((e.ms_at_measured - estimate_ms(65_863_680, 27.1)).abs() < 1e-9);
    }
}
