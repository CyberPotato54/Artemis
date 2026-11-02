//! `artemis reuse-report`: how much of the prompts the server's cache gave back, read from request logs.
//! Over every generation line it sums `reused_tokens` and `prompt_tokens` overall, by where
//! the restore came from (`restored`) and by why it stopped short (`cause`), and counts the think-budget insertions by
//! `think_budget`. Reported only: nothing here gates. A line that is not a generation (a lifecycle line, `/v1/load`,
//! a 401, `count_tokens`, a request the engine never started) is left out.
use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(clap::Args, Debug)]
pub struct ReuseReportArgs {
    /// A server's log directory (`--log-dir` of `artemis serve`); repeat it or give several for several servers.
    #[arg(long, num_args = 1.., required = true)]
    pub log_dir: Vec<PathBuf>,
    /// Where the report goes, as JSON.
    #[arg(long)]
    pub out: PathBuf,
}

/// Reuse over a set of requests.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Rate {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub reused_tokens: u64,
    /// `reused_tokens / prompt_tokens`; `null` when there were no prompt tokens.
    pub rate: Option<f64>,
}

impl Rate {
    fn add(&mut self, prompt: u64, reused: u64) {
        self.requests += 1;
        self.prompt_tokens += prompt;
        self.reused_tokens += reused;
        self.rate =
            (self.prompt_tokens > 0).then(|| self.reused_tokens as f64 / self.prompt_tokens as f64);
    }
}

/// The requests that carried one think budget, and how many of them the engine closed at it.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Budget {
    pub requests: u64,
    pub closed: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Report {
    pub overall: Rate,
    /// By the line's `restored` (`none`, `live`, `interval`, `prompt_end`, `user_query`, `reply_end`).
    pub by_source: BTreeMap<String, Rate>,
    /// By the line's `cause` (`none`, `new_conversation`, `reasoning_dropped`, `reply_retokenized`, `new_input`,
    /// `no_snapshot`).
    pub by_cause: BTreeMap<String, Rate>,
    /// By `think_budget`, `"none"` for `null`: the budget's requests and how many `budget_closed`.
    pub budgets: BTreeMap<String, Budget>,
}

/// Whether `line` is a generation the engine started: a request line of a chat endpoint with its prompt and reuse
/// counts. Lifecycle lines have an `event`; `/v1/load`, `/v1/unload` and `count_tokens` are other endpoints; a 401 or
/// a request that timed out in the queue never got a restore, so has no `reused_tokens`.
fn is_generation(line: &Value) -> bool {
    line.get("event").is_none()
        && matches!(
            line["endpoint"].as_str(),
            Some("/v1/chat/completions" | "/v1/messages")
        )
        && line["prompt_tokens"].is_u64()
        && line["reused_tokens"].is_u64()
}

/// The report over `lines` (any mix of lifecycle and request lines).
pub fn summarize(lines: &[Value]) -> Report {
    let mut r = Report::default();
    for l in lines.iter().filter(|l| is_generation(l)) {
        let (prompt, reused) = (
            l["prompt_tokens"].as_u64().unwrap_or(0),
            l["reused_tokens"].as_u64().unwrap_or(0),
        );
        r.overall.add(prompt, reused);
        let key = |k: &str| l[k].as_str().unwrap_or("unknown").to_string();
        r.by_source
            .entry(key("restored"))
            .or_default()
            .add(prompt, reused);
        r.by_cause
            .entry(key("cause"))
            .or_default()
            .add(prompt, reused);
        let budget = l["think_budget"]
            .as_u64()
            .map_or("none".to_string(), |b| b.to_string());
        let b = r.budgets.entry(budget).or_default();
        b.requests += 1;
        b.closed += u64::from(l["budget_closed"] == true);
    }
    r
}

/// `artemis reuse-report`: reads every log directory, writes the report to `--out` and prints its headline. An
/// unreadable directory or log is an error (exit 2).
#[inline(never)]
pub fn run(args: ReuseReportArgs) -> Result<()> {
    let mut lines = Vec::new();
    for dir in &args.log_dir {
        lines.extend(super::log::read_lines(dir)?);
    }
    let report = summarize(&lines);
    let mut text = serde_json::to_string_pretty(&report)?;
    text.push('\n');
    std::fs::write(&args.out, text)
        .with_context(|| format!("cannot write {}", args.out.display()))?;
    let o = &report.overall;
    println!(
        "reuse-report: {} generations, {} of {} prompt tokens reused ({}) ({})",
        o.requests,
        o.reused_tokens,
        o.prompt_tokens,
        o.rate
            .map_or("no rate".to_string(), |r| format!("{:.1}%", 100.0 * r)),
        args.out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn gen(prompt: u64, reused: u64, restored: &str, cause: &str) -> Value {
        json!({"id": "chatcmpl-1", "endpoint": "/v1/chat/completions", "status": 200,
            "prompt_tokens": prompt, "reused_tokens": reused, "restored": restored, "cause": cause,
            "think_budget": null, "budget_closed": false})
    }

    #[test]
    fn summarize_sums_by_source_and_cause() {
        let lines = vec![
            json!({"event": "load", "seconds": 9.5}),
            gen(1000, 900, "prompt_end", "new_input"),
            json!({"id": "req_1", "endpoint": "/v1/load", "status": 200, "prompt_tokens": null}),
            gen(500, 400, "prompt_end", "new_input"),
            json!({"id": "req_2", "endpoint": "/v1/chat/completions", "status": 401,
                "prompt_tokens": null, "reused_tokens": null}),
            json!({"id": "msg_3", "endpoint": "/v1/messages/count_tokens", "status": 200,
                "prompt_tokens": 77, "reused_tokens": null}),
            gen(300, 0, "none", "new_conversation"),
            json!({"id": "chatcmpl-9", "endpoint": "/v1/chat/completions", "status": 503,
                "prompt_tokens": 40, "reused_tokens": null}),
            gen(200, 200, "reply_end", "none"),
        ];
        let r = summarize(&lines);
        let rate = |requests, prompt_tokens, reused_tokens| Rate {
            requests,
            prompt_tokens,
            reused_tokens,
            rate: Some(reused_tokens as f64 / prompt_tokens as f64),
        };
        assert_eq!(r.overall, rate(4, 2000, 1500));
        assert_eq!(r.overall.rate, Some(0.75));
        assert_eq!(
            r.by_source,
            BTreeMap::from([
                ("prompt_end".to_string(), rate(2, 1500, 1300)),
                ("none".to_string(), rate(1, 300, 0)),
                ("reply_end".to_string(), rate(1, 200, 200)),
            ])
        );
        assert_eq!(
            r.by_cause,
            BTreeMap::from([
                ("new_input".to_string(), rate(2, 1500, 1300)),
                ("new_conversation".to_string(), rate(1, 300, 0)),
                ("none".to_string(), rate(1, 200, 200)),
            ])
        );
    }

    #[test]
    fn an_empty_log_reports_no_rate() {
        let r = summarize(&[]);
        assert_eq!(r.overall, Rate::default());
        assert_eq!(r.overall.rate, None);
        assert!(r.by_source.is_empty() && r.by_cause.is_empty() && r.budgets.is_empty());
        // Lines with nothing to count give the same.
        let r = summarize(&[json!({"event": "unload", "cause": "idle"})]);
        assert_eq!(r, Report::default());
        assert!(serde_json::to_value(&r).unwrap()["overall"]["rate"].is_null());
    }

    #[test]
    fn a_generation_of_no_prompt_tokens_has_no_rate() {
        let r = summarize(&[gen(0, 0, "none", "none")]);
        assert_eq!((r.overall.requests, r.overall.rate), (1, None));
    }

    #[test]
    fn budgets_count_forced_closes() {
        let mut a = gen(10, 0, "none", "new_conversation");
        a["think_budget"] = json!(1024);
        let mut b = a.clone();
        b["budget_closed"] = json!(true);
        let r = summarize(&[a, b, gen(10, 0, "none", "new_conversation")]);
        assert_eq!(
            r.budgets,
            BTreeMap::from([
                (
                    "1024".to_string(),
                    Budget {
                        requests: 2,
                        closed: 1
                    }
                ),
                (
                    "none".to_string(),
                    Budget {
                        requests: 1,
                        closed: 0
                    }
                ),
            ])
        );
    }

    #[test]
    fn run_reads_every_directory_and_writes_the_report() {
        let base =
            std::env::temp_dir().join(format!("artemis-reuse-report-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (a, b) = (base.join("a"), base.join("b"));
        for (d, l) in [
            (&a, gen(100, 50, "user_query", "new_input")),
            (&b, gen(100, 0, "none", "none")),
        ] {
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join("artemis-2026-10-26.jsonl"), format!("{l}\n")).unwrap();
        }
        let out = base.join("report.json");
        run(ReuseReportArgs {
            log_dir: vec![a, b],
            out: out.clone(),
        })
        .unwrap();
        let v: Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
        assert_eq!(v["overall"]["requests"], 2);
        assert_eq!(v["overall"]["rate"], 0.25);
        assert_eq!(v["by_source"]["user_query"]["reused_tokens"], 50);
        // An unreadable directory is an error naming it.
        let err = run(ReuseReportArgs {
            log_dir: vec![base.join("missing")],
            out,
        })
        .unwrap_err();
        assert!(format!("{err:#}").contains("missing"), "{err:#}");
        let _ = std::fs::remove_dir_all(&base);
    }
}
