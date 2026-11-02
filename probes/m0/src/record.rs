//! One JSON line per measurement, appended to a results file.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fs::OpenOptions, io::Write, path::Path};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Record {
    pub probe: String,
    pub device: String,
    pub driver: String,
    pub unix_time: u64,
    pub params: Value,
    pub results: Value,
}

impl Record {
    pub fn new(probe: &str, device: &str, driver: &str, params: Value, results: Value) -> Record {
        let unix_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Record {
            probe: probe.into(),
            device: device.into(),
            driver: driver.into(),
            unix_time,
            params,
            results,
        }
    }
}

pub fn append(path: &Path, rec: &Record) -> Result<()> {
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{}", serde_json::to_string(rec)?)?;
    println!(
        "{} {}",
        rec.probe,
        serde_json::to_string_pretty(&rec.results)?
    );
    Ok(())
}

pub fn load(path: &Path) -> Result<Vec<Record>> {
    std::fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}

/// Nearest-rank percentile of an ascending slice; NaN if empty.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_picks_nearest_rank() {
        assert_eq!(percentile(&[1.0, 2.0, 3.0], 0.5), 2.0);
        let v: Vec<f64> = (0..1000).map(f64::from).collect();
        assert_eq!(percentile(&v, 0.999), 998.0);
        assert!(percentile(&[], 0.5).is_nan());
    }

    #[test]
    fn records_round_trip_through_a_file() {
        let path =
            std::env::temp_dir().join(format!("m0-record-test-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let r = Record::new(
            "read-bw",
            "gpu",
            "1.0",
            serde_json::json!({"mem": "host"}),
            serde_json::json!({"gbps": 20.5}),
        );
        append(&path, &r).unwrap();
        append(&path, &r).unwrap();
        let back = load(&path).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[1].results["gbps"], 20.5);
        std::fs::remove_file(&path).unwrap();
    }
}
