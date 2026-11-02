//! The two verdicts the probes exist to produce (host-read and cpu-path), and a plain summary of a results file.
use crate::record::Record;
use std::fmt::Write;

#[derive(Debug, PartialEq)]
pub struct Gate {
    pub name: &'static str,
    pub pass: bool,
    pub detail: String,
}

/// Approach A needs ≥ 15 GB/s of shader reads from pinned host memory in the decode-shaped case:
/// 40 chained reads with barriers, best workgroup count.
pub fn host_read(recs: &[Record]) -> Gate {
    let best = recs
        .iter()
        .filter(|r| r.probe == "read-bw" && r.params["mem"] == "host" && r.params["chain"] == 40)
        .filter_map(|r| r.results["gbps"].as_f64())
        .fold(f64::NAN, f64::max);
    Gate {
        name: "host-read",
        pass: best >= 15.0,
        detail: if best.is_nan() {
            "no chained host read-bw run".into()
        } else {
            format!("best chained host read {best:.1} GB/s (needs >= 15)")
        },
    }
}

/// The CPU miss path needs a spin-wait round trip with median ≤ 20 µs and no GPU timeouts, CPU timeouts
/// or device loss over ≥ 1M round trips, all of them timed (no run that stopped early).
pub fn cpu_path(recs: &[Record]) -> Gate {
    let best = recs
        .iter()
        .filter(|r| r.probe == "roundtrip" && r.params["mode"] == "spin")
        .filter(|r| r.params["n"].as_u64().unwrap_or(0) >= 1_000_000)
        .filter(|r| {
            // Every round trip timed except the first of each dispatch: a shorter run stopped early.
            let n = r.params["n"].as_u64().unwrap_or(0);
            let batch = r.params["batch"].as_u64().unwrap_or(1).max(1);
            r.results["samples"].as_u64().unwrap_or(0) >= n - n.div_ceil(batch)
        })
        .filter(|r| {
            r.results["gpu_timeouts"] == 0
                && r.results["cpu_timeouts"] == 0
                && r.results["device_lost"] == false
        })
        .filter_map(|r| r.results["median_us"].as_f64())
        .fold(f64::NAN, f64::min);
    Gate {
        name: "cpu-path",
        pass: best <= 20.0,
        detail: if best.is_nan() {
            "no clean spin run of >= 1M round trips".into()
        } else {
            format!("best clean spin median {best:.1} us (needs <= 20)")
        },
    }
}

/// Markdown: one table per probe (params and results as JSON), then the two gate verdicts.
pub fn summarize(recs: &[Record]) -> String {
    let mut s = String::new();
    for probe in [
        "limits",
        "host-alloc",
        "read-bw",
        "copy-bw",
        "matvec",
        "roundtrip",
    ] {
        let rows: Vec<&Record> = recs.iter().filter(|r| r.probe == probe).collect();
        if rows.is_empty() {
            continue;
        }
        writeln!(s, "### {probe}\n\n| params | results |\n|---|---|").unwrap();
        for r in rows {
            writeln!(s, "| `{}` | `{}` |", r.params, r.results).unwrap();
        }
        s.push('\n');
    }
    for g in [host_read(recs), cpu_path(recs)] {
        writeln!(
            s,
            "- **gate {}**: {}: {}",
            g.name,
            if g.pass { "PASS" } else { "FAIL" },
            g.detail
        )
        .unwrap();
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rec(probe: &str, params: serde_json::Value, results: serde_json::Value) -> Record {
        Record::new(probe, "test", "test", params, results)
    }

    /// A spin run that completed: every round trip timed except the first of each 1,000-ping dispatch.
    fn clean_spin(median: f64, n: u64) -> Record {
        rec(
            "roundtrip",
            json!({"mode": "spin", "n": n, "batch": 1000}),
            json!({"median_us": median, "samples": n - n.div_ceil(1000), "gpu_timeouts": 0,
                   "cpu_timeouts": 0, "device_lost": false}),
        )
    }

    #[test]
    fn cpu_path_rejects_runs_that_stopped_early() {
        // A run that ended on a non-device-loss error has no timeouts but too few samples.
        let mut short = clean_spin(5.0, 1_000_000);
        short.results["samples"] = json!(120_000);
        assert!(!cpu_path(&[short]).pass);
    }

    #[test]
    fn host_read_uses_the_best_chained_host_run_only() {
        let recs = vec![
            rec(
                "read-bw",
                json!({"mem": "host", "chain": 40}),
                json!({"gbps": 12.0}),
            ),
            rec(
                "read-bw",
                json!({"mem": "host", "chain": 40}),
                json!({"gbps": 18.5}),
            ),
            rec(
                "read-bw",
                json!({"mem": "vram", "chain": 40}),
                json!({"gbps": 500.0}),
            ),
            rec(
                "read-bw",
                json!({"mem": "host", "chain": 1}),
                json!({"gbps": 30.0}),
            ),
        ];
        let g = host_read(&recs);
        assert!(g.pass);
        assert!(g.detail.contains("18.5"), "{}", g.detail);
    }

    #[test]
    fn host_read_fails_below_threshold_and_without_data() {
        assert!(
            !host_read(&[rec(
                "read-bw",
                json!({"mem": "host", "chain": 40}),
                json!({"gbps": 14.9})
            )])
            .pass
        );
        assert!(!host_read(&[]).pass);
    }

    #[test]
    fn cpu_path_ignores_runs_with_timeouts_or_device_loss() {
        let mut timed_out = clean_spin(5.0, 1_000_000);
        timed_out.results["gpu_timeouts"] = json!(3);
        let mut lost = clean_spin(4.0, 1_000_000);
        lost.results["device_lost"] = json!(true);
        let g = cpu_path(&[timed_out, lost, clean_spin(25.0, 1_000_000)]);
        assert!(!g.pass);
        assert!(g.detail.contains("25.0"), "{}", g.detail);
    }

    #[test]
    fn cpu_path_needs_a_million_round_trips() {
        assert!(!cpu_path(&[clean_spin(5.0, 1000)]).pass);
        assert!(cpu_path(&[clean_spin(5.0, 1_000_000)]).pass);
    }

    #[test]
    fn summary_lists_both_gates() {
        let s = summarize(&[clean_spin(5.0, 1_000_000)]);
        assert!(s.contains("gate host-read**: FAIL"), "{s}");
        assert!(s.contains("gate cpu-path**: PASS"), "{s}");
        assert!(s.contains("### roundtrip"), "{s}");
    }
}
