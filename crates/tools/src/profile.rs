//! `generate --profile`'s report: the engine's step timings summed per kernel and per layer,
//! for the prompt's prefill and for decode, with each part's CPU phases and the sampler's time.
use engine::{
    Span, SpanKind, StepKind, StepTiming, TAG_EMBED, TAG_MTP_WINDOW, TAG_OUTPUT, UNTAGGED,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const FORMAT: &str = "artemis-profile-1";

/// CPU time over a part's steps, in microseconds.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cpu {
    pub prepare_us: u64,
    pub record_us: u64,
    pub submit_us: u64,
    pub read_us: u64,
    /// The sampler's time choosing tokens (decode only).
    pub sample_us: u64,
}

/// One kernel or one layer: its commands and their GPU time in microseconds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Row {
    pub name: String,
    pub calls: u64,
    pub us: f64,
}

/// Staging copies over a part's steps: their GPU time, and how much of it ran
/// while one of the same step's compute submits was running.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Copies {
    pub count: u64,
    pub gpu_us: f64,
    /// The part of `gpu_us` that overlapped the step's compute; the rest held the step up.
    pub hidden_us: f64,
}

/// (copy ns, hidden ns) over one step's spans: each staging copy's time, and its overlap with the step's compute
/// submits, which run one after another and so never overlap each other.
pub fn copy_overlap(spans: &[Span]) -> (u64, u64) {
    let (mut total, mut hidden) = (0, 0);
    for c in spans.iter().filter(|s| s.kind == SpanKind::Copy) {
        total += c.end_ns.saturating_sub(c.start_ns);
        for k in spans.iter().filter(|s| s.kind == SpanKind::Compute) {
            hidden += c
                .end_ns
                .min(k.end_ns)
                .saturating_sub(c.start_ns.max(k.start_ns));
        }
    }
    (total, hidden)
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Part {
    pub steps: usize,
    pub tokens: usize,
    /// GPU time over the part's steps, in microseconds.
    pub gpu_us: f64,
    pub cpu: Cpu,
    /// Per kernel name, most time first.
    pub kernels: Vec<Row>,
    /// Per layer, in graph order: "embed", "l0" … "l39", "output".
    pub layers: Vec<Row>,
    /// Each step's GPU time, in microseconds.
    pub step_gpu_us: Vec<f64>,
    /// The steps' staging copies. Absent from older reports.
    #[serde(default)]
    pub copies: Copies,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub format: String,
    pub prefill: Part,
    pub decode: Part,
    /// The speculative verify steps, each `tokens` positions; empty without speculation. Absent from older
    /// reports.
    #[serde(default)]
    pub verify: Part,
    /// The MTP draft passes, each `tokens` draft rows; empty without the MTP. Absent from older reports.
    #[serde(default)]
    pub mtp: Part,
}

fn tag_name(tag: u32) -> String {
    match tag {
        TAG_EMBED => "embed".into(),
        TAG_OUTPUT => "output".into(),
        TAG_MTP_WINDOW => "mtp window".into(),
        UNTAGGED => "untagged".into(),
        l => format!("l{l}"),
    }
}

/// Graph order: the embedding, the layers, the output head, then anything untagged.
fn tag_order(tag: u32) -> (u8, u32) {
    match tag {
        TAG_EMBED => (0, 0),
        TAG_OUTPUT => (2, 0),
        TAG_MTP_WINDOW => (2, 1),
        UNTAGGED => (3, 0),
        l => (1, l),
    }
}

fn us(ns: u64) -> f64 {
    ns as f64 / 1e3
}

/// Sums one kind's steps into a part; `sample_us` is the sampler's time over them.
pub fn part(steps: &[&StepTiming], sample_us: u64) -> Part {
    let mut kernels: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    let mut layers: BTreeMap<(u8, u32), (u32, u64, u64)> = BTreeMap::new();
    let mut p = Part {
        steps: steps.len(),
        ..Part::default()
    };
    p.cpu.sample_us = sample_us;
    let mut total = 0u64;
    for s in steps {
        p.tokens += s.tokens;
        p.cpu.prepare_us += s.cpu.prepare_us;
        p.cpu.record_us += s.cpu.record_us;
        p.cpu.submit_us += s.cpu.submit_us;
        p.cpu.read_us += s.cpu.read_us;
        let mut step = 0u64;
        for t in &s.gpu {
            step += t.ns;
            let k = kernels.entry(t.name).or_default();
            k.0 += 1;
            k.1 += t.ns;
            let l = layers.entry(tag_order(t.tag)).or_insert((t.tag, 0, 0));
            l.1 += 1;
            l.2 += t.ns;
        }
        p.step_gpu_us.push(us(step));
        let (copy, hidden) = copy_overlap(&s.spans);
        p.copies.count += s.spans.iter().filter(|x| x.kind == SpanKind::Copy).count() as u64;
        p.copies.gpu_us += us(copy);
        p.copies.hidden_us += us(hidden);
        total += step;
    }
    // Summed in ns and converted once, so the totals are exact.
    p.gpu_us = us(total);
    p.kernels = kernels
        .into_iter()
        .map(|(name, (calls, ns))| Row {
            name: name.into(),
            calls,
            us: us(ns),
        })
        .collect();
    p.kernels
        .sort_by(|a, b| b.us.total_cmp(&a.us).then_with(|| a.name.cmp(&b.name)));
    p.layers = layers
        .into_values()
        .map(|(tag, calls, ns)| Row {
            name: tag_name(tag),
            calls,
            us: us(ns),
        })
        .collect();
    p
}

/// The report over a run's steps. The sampler's time goes to decode.
pub fn report(steps: &[StepTiming], decode_sample_us: u64) -> Report {
    let of = |k: StepKind| steps.iter().filter(|s| s.kind == k).collect::<Vec<_>>();
    Report {
        format: FORMAT.into(),
        prefill: part(&of(StepKind::Prefill), 0),
        decode: part(&of(StepKind::Decode), decode_sample_us),
        verify: part(&of(StepKind::Verify), 0),
        mtp: part(&of(StepKind::Mtp), 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{CpuTimes, StepKind, StepTiming, Timing, TAG_EMBED, TAG_OUTPUT};

    fn t(name: &'static str, tag: u32, ns: u64) -> Timing {
        Timing { name, tag, ns }
    }

    fn step(kind: StepKind, pos: usize, tokens: usize, gpu: Vec<Timing>) -> StepTiming {
        StepTiming {
            kind,
            pos,
            tokens,
            cpu: CpuTimes {
                prepare_us: 1,
                record_us: 2,
                submit_us: 3,
                read_us: 4,
            },
            gpu,
            spans: Vec::new(),
        }
    }

    #[test]
    fn steps_sum_per_kernel_and_per_layer_in_graph_order() {
        let steps = [
            step(
                StepKind::Prefill,
                0,
                5,
                vec![
                    t("embed_rows", TAG_EMBED, 1_000),
                    t("matmul_q8_0", 0, 5_000),
                    t("matmul_q8_0", 1, 3_000),
                    t("matmul_q6_k", TAG_OUTPUT, 2_000),
                ],
            ),
            step(
                StepKind::Decode,
                5,
                1,
                vec![
                    t("embed_row", TAG_EMBED, 100),
                    t("matvec_q8_0", 1, 700),
                    t("matvec_q8_0", 0, 300),
                    t("copy", TAG_OUTPUT, 50),
                ],
            ),
            step(StepKind::Decode, 6, 1, vec![t("matvec_q8_0", 0, 400)]),
        ];
        let r = report(&steps, 90);
        assert_eq!(r.format, FORMAT);
        assert_eq!(
            (
                r.prefill.steps,
                r.prefill.tokens,
                r.decode.steps,
                r.decode.tokens
            ),
            (1, 5, 2, 2)
        );
        assert_eq!(r.prefill.gpu_us, 11.0);
        assert_eq!(r.decode.gpu_us, 1.55);
        assert_eq!(r.decode.step_gpu_us, [1.15, 0.4]);
        assert_eq!(
            r.decode.kernels[0],
            Row {
                name: "matvec_q8_0".into(),
                calls: 3,
                us: 1.4
            }
        );
        let layers: Vec<&str> = r.decode.layers.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(layers, ["embed", "l0", "l1", "output"]);
        assert_eq!(r.decode.layers[1].us, 0.7, "layer 0: 300 + 400 ns");
        assert_eq!(
            r.decode.cpu,
            Cpu {
                prepare_us: 2,
                record_us: 4,
                submit_us: 6,
                read_us: 8,
                sample_us: 90
            }
        );
        assert_eq!(r.prefill.cpu.sample_us, 0, "sampling is decode's");
    }

    #[test]
    fn a_copy_is_hidden_where_compute_runs_beside_it() {
        // A staging copy's time counts as hidden where one of its step's compute
        // submits ran at the same time. Inside a submit: all hidden. Across the gap between two submits: the gap is
        // exposed. Before any compute: all exposed.
        use engine::{
            Span,
            SpanKind::{Compute, Copy},
        };
        let sp = |kind, start_ns, end_ns| Span {
            kind,
            tag: 0,
            start_ns,
            end_ns,
        };
        assert_eq!(
            copy_overlap(&[sp(Compute, 100, 200), sp(Copy, 120, 180)]),
            (60, 60)
        );
        assert_eq!(
            copy_overlap(&[
                sp(Compute, 100, 200),
                sp(Compute, 250, 400),
                sp(Copy, 150, 300)
            ]),
            (150, 100)
        );
        assert_eq!(
            copy_overlap(&[sp(Copy, 0, 50), sp(Compute, 60, 100)]),
            (50, 0)
        );
        assert_eq!(copy_overlap(&[sp(Compute, 0, 10)]), (0, 0));
    }

    #[test]
    fn a_parts_copies_sum_over_its_steps() {
        use engine::{
            Span,
            SpanKind::{Compute, Copy},
        };
        let mut a = step(StepKind::Prefill, 0, 8, vec![t("matmul_q8_0", 0, 1_000)]);
        a.spans = vec![
            Span {
                kind: Compute,
                tag: 0,
                start_ns: 0,
                end_ns: 1_000_000,
            },
            Span {
                kind: Copy,
                tag: 1,
                start_ns: 500_000,
                end_ns: 1_500_000,
            },
        ];
        let mut b = step(StepKind::Prefill, 8, 8, vec![t("matmul_q8_0", 0, 1_000)]);
        b.spans = vec![Span {
            kind: Copy,
            tag: 0,
            start_ns: 0,
            end_ns: 2_000_000,
        }];
        let p = part(&[&a, &b], 0);
        assert_eq!(p.copies.count, 2);
        assert_eq!((p.copies.gpu_us, p.copies.hidden_us), (3_000.0, 500.0));
    }

    #[test]
    fn a_run_without_steps_of_a_kind_has_an_empty_part() {
        let r = report(&[], 0);
        assert_eq!(r.prefill, Part::default());
        assert_eq!(r.decode, Part::default());
        assert_eq!(r.verify, Part::default());
    }

    #[test]
    fn verify_steps_have_their_own_line_and_are_not_decodes() {
        let steps = [
            step(StepKind::Decode, 6, 1, vec![t("matvec_q8_0", 0, 400)]),
            step(StepKind::Verify, 7, 4, vec![t("matvec_tokens", 0, 900)]),
        ];
        let r = report(&steps, 90);
        assert_eq!((r.decode.steps, r.decode.tokens), (1, 1));
        assert_eq!((r.verify.steps, r.verify.tokens), (1, 4));
        assert_eq!(r.verify.gpu_us, 0.9);
        assert_eq!(r.verify.cpu.sample_us, 0, "sampling is decode's");
        // A report written before the verify line existed reads with an empty one.
        let mut v = serde_json::to_value(&r).unwrap();
        v.as_object_mut().unwrap().remove("verify");
        let old: Report = serde_json::from_value(v).unwrap();
        assert_eq!(old.verify, Part::default());
    }

    #[test]
    fn the_mtp_windows_time_has_its_own_tag() {
        assert_eq!(tag_name(TAG_MTP_WINDOW), "mtp window");
        assert!(tag_order(TAG_OUTPUT) < tag_order(TAG_MTP_WINDOW));
        assert!(tag_order(TAG_MTP_WINDOW) < tag_order(UNTAGGED));
        assert_ne!(tag_name(TAG_MTP_WINDOW), tag_name(39));
    }

    #[test]
    fn mtp_passes_have_their_own_line_and_are_not_decodes() {
        let steps = [
            step(StepKind::Decode, 6, 1, vec![t("matvec_q8_0", 0, 400)]),
            step(StepKind::Mtp, 7, 3, vec![t("matvec_q8_0", 0, 1_200)]),
        ];
        let r = report(&steps, 90);
        assert_eq!((r.decode.steps, r.decode.tokens), (1, 1));
        assert_eq!((r.mtp.steps, r.mtp.tokens), (1, 3));
        assert_eq!(r.mtp.gpu_us, 1.2);
        assert_eq!(r.verify, Part::default());
        // A report written before the MTP line existed reads with an empty one.
        let mut v = serde_json::to_value(&r).unwrap();
        v.as_object_mut().unwrap().remove("mtp");
        let old: Report = serde_json::from_value(v).unwrap();
        assert_eq!(old.mtp, Part::default());
    }
}
