//! `artemis test7`'s cases, driven over HTTP against the server's fake engine
//! through `tests/support`, with the scripts pushed per case. The server runs in-process on `127.0.0.1:0`; the driver
//! is the library's `run_case` with a real `Client`, so these tests cover the client, the log reader, the judges and
//! the verdict files together. The fake has no model: what each reply says is the script's, and a failing case is a
//! script that does not do what the model should.
mod support;

use artemis_tools::test7::cases::{LOG_GATED, REQUESTS_STEPS};
use artemis_tools::test7::{run_case, write_verdict, Ctx, SpeedRun, Status, Verdict};
use serde_json::Value;
use server::app::Tuning;
use server::backend::{
    Backend, BackendStats, CacheReport, Finish, Full, GenRequest, Lifecycle, Progress, Reply, Sink,
    Source, SpecReport, StatsCopy,
};
use server::config::Config;
use server::fake::{FakeBackend, FakeConfig, Script};
use server::logs::Logger;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use support::{fake, parts, scratch_dir, start, tokenizer, Running};

/// An invented key-shaped literal, not a secret.
const KEY: &str = "test7-key-not-secret-7777"; // gitleaks:allow

struct Rig {
    running: Running,
    fake: Arc<FakeBackend>,
    ctx: Ctx,
}

/// A server over a fake engine that starts idle and takes `load_time` to load, writing its loads to the log as the
/// real adapter does, and a driver context for it.
async fn rig(load_time: Duration) -> Rig {
    rig_with(
        FakeConfig {
            load_time,
            ..FakeConfig::default()
        },
        None,
    )
    .await
}

/// The same with the fake's own config and the server's `queue_timeout_s`. The context has the fake-only hook that
/// lets a load line's missing figures (the fake gives none) pass; the command never sets it.
async fn rig_with(fake_cfg: FakeConfig, queue_timeout_s: Option<u64>) -> Rig {
    rig_around(fake_cfg, queue_timeout_s, |fake| fake).await
}

/// `rig_with`, with the server reaching the fake through the backend `wrap` makes of it.
async fn rig_around(
    fake_cfg: FakeConfig,
    queue_timeout_s: Option<u64>,
    wrap: impl FnOnce(Arc<FakeBackend>) -> Arc<dyn Backend>,
) -> Rig {
    let fake = fake(fake_cfg);
    let log_dir = scratch_dir("test7-logs");
    fake.set_logger(Logger::new(log_dir.clone(), 0).unwrap());
    let mut cfg = Config {
        model: Some("m.gguf".into()),
        ctx: 65_536,
        ..Config::default()
    };
    if let Some(s) = queue_timeout_s {
        cfg.queue_timeout_s = s;
    }
    let running = start(
        parts(wrap(fake.clone()), cfg, Some(KEY), &log_dir),
        Tuning::default(),
    )
    .await;
    let mut ctx = Ctx::new(&running.base_url, KEY, log_dir, repo_root()).unwrap();
    ctx.allow_missing_vram = true;
    Rig { running, fake, ctx }
}

/// The fake as the server sees it, except that an unload is passed on only once every request the fake has started has
/// ended: an unload that always comes too late to cut a stream. Without this the unload races the engine, which under
/// load can lag the client by more than the tokens left in the reply.
struct UnloadAfterRuns(Arc<FakeBackend>);

impl Backend for UnloadAfterRuns {
    fn submit(&self, req: GenRequest, sink: Sink) -> Result<(), Full> {
        self.0.submit(req, sink)
    }
    fn state(&self) -> Lifecycle {
        self.0.state()
    }
    fn load(&self) -> Reply<()> {
        self.0.load()
    }
    fn unload(&self) -> Reply<bool> {
        // `Backend` methods must not block, so a thread waits and forwards the fake's answer.
        let (tx, rx) = tokio::sync::oneshot::channel();
        let fake = self.0.clone();
        std::thread::spawn(move || {
            let end = Instant::now() + Duration::from_secs(60);
            while fake.runs().iter().any(|r| r.finish.is_none()) && Instant::now() < end {
                std::thread::sleep(Duration::from_millis(5));
            }
            if let Ok(answer) = fake.unload().blocking_recv() {
                let _ = tx.send(answer);
            }
        });
        rx
    }
    fn stats(&self) -> Reply<BackendStats> {
        self.0.stats()
    }
    fn progress(&self) -> Option<Progress> {
        self.0.progress()
    }
    fn latest_stats(&self) -> Option<StatsCopy> {
        self.0.latest_stats()
    }
}

/// The repository root: the committed vocabulary the `budget` case reads is under it.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn say(fake: &FakeBackend, text: &str) {
    fake.push(Script::text(tokenizer(), text));
}

fn call_text() -> String {
    "<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n</tool_call>"
        .to_string()
}

/// What the model says at `step` of `requests`, as the case's own order has it: a `read_file` call for each round
/// trip's first turn, text otherwise; with the reasoning and its close where the request has thinking on.
fn reply_for(step: &str) -> String {
    match step {
        s if s.ends_with("_round_trip_call") => call_text(),
        s if s.ends_with("_round_trip_echo") => "The file is about foxes.".into(),
        "openai_effort_none" | "anthropic_thinking_disabled" => "Paris.".into(),
        _ => "Let me think.\n</think>\n\nParis.".into(),
    }
}

/// Scripts every step of `requests` by `reply_for`, except that `replace` may swap one step's text.
fn script_requests(fake: &FakeBackend, replace: Option<(&str, &str)>) {
    for step in REQUESTS_STEPS {
        let text = match replace {
            Some((s, t)) if s == step => t.to_string(),
            _ => reply_for(step),
        };
        say(fake, &text);
    }
}

fn check<'a>(v: &'a Verdict, name: &str) -> &'a artemis_tools::test7::Check {
    v.checks
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no check {name}: {:?}", v.checks))
}

#[tokio::test]
async fn load_case_passes_when_health_shows_loading() {
    let rig = rig(Duration::from_secs(2)).await;
    let v = run_case(&rig.ctx, "load").await.unwrap();
    assert_eq!(v.status, Status::Pass, "{:?}", v.checks);
    assert!(
        v.figures["loading_polls"].as_u64().unwrap() >= 1,
        "{:?}",
        v.figures
    );
    // Polls come every 250 ms, so a 2 s load shows several.
    assert!(v.figures["polls"].as_u64().unwrap() > v.figures["loading_polls"].as_u64().unwrap());
    // The load's own line is read back: the fake wrote one with its time.
    assert!(
        v.figures["load_seconds"].as_f64().unwrap() >= 1.9,
        "{:?}",
        v.figures
    );
    assert_eq!(rig.fake.loads().len(), 1);
}

#[tokio::test]
async fn load_case_fails_when_already_loaded() {
    let rig = rig(Duration::ZERO).await;
    let r = rig
        .running
        .send("POST", "/v1/load", &[("x-api-key", KEY)], b"")
        .await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let v = run_case(&rig.ctx, "load").await.unwrap();
    assert_eq!(v.status, Status::Fail);
    assert!(!check(&v, "loading_seen").pass);
    assert!(check(&v, "chat_answers").pass && check(&v, "ready_last").pass);
    assert_eq!(v.figures["loading_polls"], 0);
    // The preload's line is not this case's: it has none.
    assert!(v.figures["load_seconds"].is_null(), "{:?}", v.figures);
}

#[tokio::test]
async fn requests_case_passes_on_scripted_replies() {
    let rig = rig(Duration::ZERO).await;
    script_requests(&rig.fake, None);
    let v = run_case(&rig.ctx, "requests").await.unwrap();
    assert_eq!(v.status, Status::Pass, "{:#?}", v.checks);
    // One check per step, then the log-side reasoning check of each no-reasoning step.
    let mut names: Vec<String> = REQUESTS_STEPS.iter().map(|s| s.to_string()).collect();
    names.extend(
        LOG_GATED
            .iter()
            .map(|s| format!("{s}_log_reasoning_tokens")),
    );
    assert_eq!(
        v.checks.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        names
    );
    assert_eq!(v.figures["scripted_calls"], serde_json::json!([]));

    // Every gating request was greedy with seed 7.
    let seen = rig.fake.requests();
    assert_eq!(seen.len(), REQUESTS_STEPS.len());
    for (i, (_, req)) in seen.iter().enumerate() {
        assert_eq!((req.sampling.temp, req.seed), (0.0, 7), "request {i}");
    }

    // What the server's log said of each request is reported: the round trips' calls, the streams, the budgets.
    let steps = &v.figures["steps"];
    assert_eq!(steps["openai_round_trip_call"]["tool_calls"], 1);
    assert_eq!(steps["anthropic_round_trip_call"]["tool_calls"], 1);
    assert_eq!(steps["openai_round_trip_echo"]["tool_calls"], 0);
    assert_eq!(steps["openai_stream"]["stream"], true);
    assert_eq!(steps["anthropic_stream"]["stream"], true);
    assert_eq!(steps["openai_nonstream"]["stream"], false);
    // The plain steps carry a think budget of 1,024, as `low` does; `high` has none.
    for plain in [
        "openai_nonstream",
        "openai_stream",
        "anthropic_nonstream",
        "anthropic_stream",
    ] {
        assert_eq!(steps[plain]["think_budget"], 1024, "{plain}");
    }
    assert_eq!(steps["openai_effort_low"]["think_budget"], 1024);
    assert_eq!(steps["openai_effort_medium"]["think_budget"], 4096);
    assert!(steps["openai_effort_high"]["think_budget"].is_null());
    assert!(steps["openai_effort_none"]["think_budget"].is_null());
    for (name, s) in steps.as_object().unwrap() {
        assert_eq!(s["status"], 200, "{name}: {s}");
        assert_eq!(s["malformed_calls"], 0, "{name}: {s}");
    }
}

#[tokio::test]
async fn requests_case_fails_without_a_tool_call() {
    let rig = rig(Duration::ZERO).await;
    // Text where the OpenAI round trip's call should be.
    script_requests(
        &rig.fake,
        Some(("openai_round_trip_call", "I do not need a tool for that.")),
    );
    let v = run_case(&rig.ctx, "requests").await.unwrap();
    assert_eq!(v.status, Status::Fail);
    assert_eq!(v.failed(), ["openai_round_trip_call"], "{:#?}", v.checks);
    assert!(
        check(&v, "openai_round_trip_call")
            .detail
            .contains("0 tool calls"),
        "{:?}",
        check(&v, "openai_round_trip_call")
    );
    // The echo went out with a scripted call, so every later step was judged on its own reply.
    assert_eq!(
        v.figures["scripted_calls"],
        serde_json::json!(["openai_round_trip_call"])
    );
    assert_eq!(rig.fake.requests().len(), REQUESTS_STEPS.len());
}

#[tokio::test]
async fn requests_case_fails_on_a_reply_with_no_text() {
    let rig = rig(Duration::ZERO).await;
    // `reasoning_effort: none` answered with nothing: a reply of the right shape but with no text fails its step only.
    script_requests(&rig.fake, Some(("openai_effort_none", "")));
    let v = run_case(&rig.ctx, "requests").await.unwrap();
    assert_eq!(v.failed(), ["openai_effort_none"], "{:#?}", v.checks);
}

/// Every file under `dir`, recursively.
fn files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            out.extend(files(&p));
        } else {
            out.push(p);
        }
    }
    out
}

#[tokio::test]
async fn verdicts_never_hold_the_key() {
    let rig = rig(Duration::from_secs(1)).await;
    let out = scratch_dir("test7-out").join("verdicts");
    let load = run_case(&rig.ctx, "load").await.unwrap();
    write_verdict(&out, &load).unwrap();
    script_requests(&rig.fake, None);
    let requests = run_case(&rig.ctx, "requests").await.unwrap();
    write_verdict(&out, &requests).unwrap();

    let written = files(&out);
    assert_eq!(written.len(), 2, "{written:?}");
    for p in &written {
        let text = std::fs::read_to_string(p).unwrap();
        assert!(!text.contains(KEY), "{} holds the key", p.display());
        let v: Value = serde_json::from_str(&text).unwrap();
        assert!(["load", "requests"].contains(&v["case"].as_str().unwrap()));
        assert!(["pass", "fail", "ran"].contains(&v["status"].as_str().unwrap()));
        assert!(v["checks"].is_array() && v["figures"].is_object());
    }
    // The verdict is named by its case.
    let mut names: Vec<_> = written
        .iter()
        .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
        .collect();
    names.sort();
    assert_eq!(names, ["load.json", "requests.json"]);
    let _ = std::fs::remove_dir_all(out.parent().unwrap());
}

// ---- budget, reuse, disconnect, idle, unload-stream, queue ----

/// `1\n2\n...n\n`.
fn count_text(n: u32) -> String {
    (1..=n).map(|i| format!("{i}\n")).collect()
}

fn failed(v: &Verdict) -> Vec<&str> {
    v.failed()
}

/// The first `n` tokens of the count to 400, each sent after `per_token`: a reply that goes on long enough for a
/// client to leave or a queue to time out.
fn long_count(n: usize, per_token: Duration) -> Script {
    let mut s = Script::text(tokenizer(), &count_text(400));
    assert!(s.tokens.len() > n, "the count is too short");
    s.tokens.truncate(n);
    s.per_token = per_token;
    s
}

/// A script that ends with `reused` tokens reused from a restore.
fn reusing(text: &str, reused: u32, source: Source, restore_bytes: u64) -> Script {
    let mut s = Script::text(tokenizer(), text);
    s.reused = reused;
    s.source = source;
    if restore_bytes > 0 {
        s.restore_ms = 12.5;
        s.cache = Some(CacheReport {
            hit_rate: None,
            miss_mib_per_token: None,
            restore_bytes,
            pool_bytes_used: 0,
            decode_misses: None,
        });
    }
    s
}

/// Writes a `load` line to a log file that sorts after the server's own (a later load).
fn inject_load(dir: &Path, pool_limit_gib: f64) {
    inject_load_in(dir, "artemis-9999-12-31.jsonl", pool_limit_gib);
}

/// Writes a `load` line to a log file that sorts before the server's own (an earlier load).
fn inject_first_load(dir: &Path, pool_limit_gib: f64) {
    inject_load_in(dir, "artemis-0000-01-01.jsonl", pool_limit_gib);
}

fn inject_load_in(dir: &Path, file: &str, pool_limit_gib: f64) {
    let line = format!("{{\"event\":\"load\",\"pool_limit_gib\":{pool_limit_gib}}}\n");
    std::fs::write(dir.join(file), line).unwrap();
}

// -- budget --

/// The reasoning is cut at the budget: 1,024 sampled tokens of thinking, then the forced close, then the answer.
#[tokio::test]
async fn budget_case_passes_when_the_budget_closes_the_reasoning() {
    let rig = rig(Duration::ZERO).await;
    let think = tokenizer().encode(" think", true);
    assert_eq!(think.len(), 1);
    let mut tokens = think.repeat(1024);
    tokens.extend(tokenizer().encode("The answer is 42.", true));
    rig.fake.push(Script {
        tokens,
        ..Script::default()
    });
    let v = run_case(&rig.ctx, "budget").await.unwrap();
    assert_eq!(v.status, Status::Pass, "{:#?}", v.checks);
    let close = support::specials().think_close.len() as u64;
    let r = v.figures["reasoning_tokens"].as_u64().unwrap();
    assert!((1024..=1024 + close).contains(&r), "{r} {close}");
    assert_eq!(v.figures["close_len"], close);
    // The request was the case's: greedy, seed 7, `low`'s budget, 4,096 tokens.
    let (_, req) = &rig.fake.requests()[0];
    assert_eq!(
        (req.max_tokens, req.sampling.temp, req.seed),
        (4096, 0.0, 7)
    );
    assert_eq!(req.think.as_ref().and_then(|t| t.budget), Some(1024));
}

/// A greedy run that closes its reasoning before the budget fails with its count, not a weaker rule.
#[tokio::test]
async fn budget_case_fails_when_the_reasoning_closes_early() {
    let rig = rig(Duration::ZERO).await;
    say(&rig.fake, "Let me think.\n</think>\n\nThe answer is 42.");
    let v = run_case(&rig.ctx, "budget").await.unwrap();
    assert_eq!(failed(&v), ["reasoning_budget"], "{:#?}", v.checks);
    let c = check(&v, "reasoning_budget");
    assert!(c.detail.contains("outside 1024"), "{c:?}");
    assert!(v.figures["reasoning_tokens"].as_u64().unwrap() < 1024);
}

// -- reuse --

/// The eight requests of `reuse`, in the order it sends them: the tool loop (t1, t2, t3), the new user turn (t4),
/// conversation B, the side call, A's next turn (a5), and A's turn after the unload and the load (a6). `reused` is
/// the six A turns' own.
fn script_reuse(fake: &FakeBackend, reused: [u32; 6], a5: (Source, u64)) {
    let [t1, t2, t3, t4, a5_reused, a6] = reused;
    fake.push(reusing(&call_text(), t1, Source::None, 0));
    fake.push(reusing(&call_text(), t2, Source::PromptEnd, 0));
    // Text where the loop's call should be: the driver scripts the call and says so.
    fake.push(reusing("Done reading.", t3, Source::PromptEnd, 0));
    fake.push(reusing(&call_text(), t4, Source::UserQuery, 0));
    fake.push(reusing("The notes are terse.", 0, Source::None, 0));
    fake.push(reusing("Ready.", 0, Source::None, 0));
    fake.push(reusing(&call_text(), a5_reused, a5.0, a5.1));
    fake.push(reusing(&call_text(), a6, Source::PromptEnd, 63_000_000));
}

/// a5 restored its user-query snapshot, copying 63 MB of KV back.
const KV_RESTORE: (Source, u64) = (Source::UserQuery, 63_000_000);

/// Each A turn's `cut_at`, from the verdict's figures.
fn cuts(v: &Verdict) -> Vec<u32> {
    v.figures["turns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["cut_at"].as_u64().unwrap() as u32)
        .collect()
}

/// The `reused` that meets each floor exactly, given every turn's cut: t2 and t3 reuse the turn before's cut, t4 the
/// first turn's, a5 t4's and a6 a5's.
fn floors_for(cut: &[u32]) -> [u32; 6] {
    [0, cut[0], cut[1], cut[0], cut[3], cut[4]]
}

/// The two-pass recipe: the prompts are deterministic, so a run with every script at `reused: 0` shows each turn's
/// cut, and the scripts that meet the floors follow from it.
async fn reuse_cuts(rig: &Rig) -> Vec<u32> {
    script_reuse(&rig.fake, [0; 6], KV_RESTORE);
    let first = run_case(&rig.ctx, "reuse").await.unwrap();
    assert_eq!(
        failed(&first),
        [
            "a_t2_reused",
            "a_t3_reused",
            "b_t4_reused",
            "c_a5_reused",
            "resume_a6_reused"
        ],
        "{:#?}",
        first.checks
    );
    let cut = cuts(&first);
    assert_eq!(cut.len(), 6);
    assert!(cut.iter().all(|&c| c > 0), "{cut:?}");
    cut
}

#[tokio::test]
async fn reuse_case_passes_when_the_script_meets_the_floors() {
    let rig = rig(Duration::ZERO).await;
    let cut = reuse_cuts(&rig).await;
    script_reuse(&rig.fake, floors_for(&cut), KV_RESTORE);
    let v = run_case(&rig.ctx, "reuse").await.unwrap();
    assert_eq!(v.status, Status::Pass, "{:#?}", v.checks);
    assert_eq!(cuts(&v), cut, "the prompts are deterministic");
    assert_eq!(v.figures["scripted_calls"], serde_json::json!(["t3"]));

    // What each turn left is reported: the restore, its bytes and time, and the reply-end hits.
    let turns = v.figures["turns"].as_array().unwrap();
    assert_eq!(turns.len(), 6);
    assert_eq!(turns[0]["user_query"], true);
    assert_eq!(turns[4]["restored"], "user_query");
    assert_eq!(turns[4]["restore_bytes"], 63_000_000);
    assert_eq!(turns[4]["restore_ms"], 12.5);
    assert_eq!(turns[2]["scripted"], true);
    assert_eq!(turns[1]["scripted"], false);
    assert_eq!(v.figures["reply_end_hits"], serde_json::json!([]));

    // The conversations are the case's: A's system prompt of about 3,000 tokens, B's of about 2,000 from position 0,
    // and a side call of at most 30 tokens; every request greedy with seed 7. The second run's are requests 8 to 15.
    let seen = rig.fake.requests();
    assert_eq!(seen.len(), 16);
    let a = seen[8].1.prompt.len();
    let b = seen[12].1.prompt.len();
    assert!((2800..3300).contains(&a), "A's first prompt is {a} tokens");
    assert!((1800..2200).contains(&b), "B's prompt is {b} tokens");
    assert_eq!(v.figures["a_prompt_tokens"], a);
    assert_eq!(v.figures["b_prompt_tokens"], b);
    assert_eq!(seen[13].1.max_tokens, 30);
    for (i, (_, req)) in seen.iter().enumerate() {
        assert_eq!((req.sampling.temp, req.seed), (0.0, 7), "request {i}");
    }
    // Each run unloaded and loaded once; the first run's first request loaded lazily.
    assert_eq!(rig.fake.unloads().len(), 2);
    assert_eq!(rig.fake.loads().len(), 3);
}

#[tokio::test]
async fn reuse_case_fails_one_token_short() {
    let rig = rig(Duration::ZERO).await;
    let cut = reuse_cuts(&rig).await;
    // One below each floor in turn fails that floor and no other: (a) twice, (b), (c) and the resume.
    for (turn, name) in [
        (1, "a_t2_reused"),
        (2, "a_t3_reused"),
        (3, "b_t4_reused"),
        (4, "c_a5_reused"),
        (5, "resume_a6_reused"),
    ] {
        let mut reused = floors_for(&cut);
        reused[turn] -= 1;
        script_reuse(&rig.fake, reused, KV_RESTORE);
        let v = run_case(&rig.ctx, "reuse").await.unwrap();
        assert_eq!(v.status, Status::Fail, "{name}");
        assert_eq!(failed(&v), [name], "{:#?}", v.checks);
    }
}

/// (c) is a restore that copies KV back: a reuse of the live state, or of nothing copied, fails whatever the count.
#[tokio::test]
async fn reuse_case_c_needs_the_restore_to_copy_kv() {
    let rig = rig(Duration::ZERO).await;
    let cut = reuse_cuts(&rig).await;
    for restore in [(Source::Live, 63_000_000), (Source::UserQuery, 0)] {
        script_reuse(&rig.fake, floors_for(&cut), restore);
        let v = run_case(&rig.ctx, "reuse").await.unwrap();
        assert_eq!(failed(&v), ["c_a5_restores_kv"], "{:#?}", v.checks);
    }
}

/// A pool below 2 GiB makes the verdict inconclusive, whatever the floors say. The pool is read from the
/// `load` lines; with a real figure the fake-only hook isn't needed.
#[tokio::test]
async fn reuse_case_reads_the_pool_from_the_log() {
    let mut rig = rig(Duration::ZERO).await;
    let cut = reuse_cuts(&rig).await;
    rig.ctx.allow_missing_vram = false;
    inject_load(&rig.running.log_dir, 8.0);
    script_reuse(&rig.fake, floors_for(&cut), KV_RESTORE);
    let v = run_case(&rig.ctx, "reuse").await.unwrap();
    assert_eq!(v.status, Status::Pass, "{:#?}", v.checks);
    assert_eq!(v.checks[0].name, "pool_at_least_2_gib");
    assert_eq!(v.figures["pool_limit_gib"], 8.0);
    assert_eq!(v.figures["inconclusive"], false);

    inject_load(&rig.running.log_dir, 1.5);
    script_reuse(&rig.fake, floors_for(&cut), KV_RESTORE);
    let v = run_case(&rig.ctx, "reuse").await.unwrap();
    assert_eq!(v.status, Status::Fail);
    assert_eq!(failed(&v), ["pool_at_least_2_gib"], "{:#?}", v.checks);
    assert_eq!(v.checks[0].detail, "inconclusive: pool below 2 GiB");
    assert_eq!(v.figures["inconclusive"], true);
}

/// `reuse` spans two loads, and each can be cut differently: a first load below 2 GiB makes the verdict inconclusive
/// although the load after it (the newest) had 8, and every load's figures are reported.
#[tokio::test]
async fn reuse_case_is_inconclusive_when_any_load_had_a_small_pool() {
    let rig = rig(Duration::ZERO).await;
    let cut = reuse_cuts(&rig).await;
    inject_first_load(&rig.running.log_dir, 1.5);
    inject_load(&rig.running.log_dir, 8.0);
    script_reuse(&rig.fake, floors_for(&cut), KV_RESTORE);
    let v = run_case(&rig.ctx, "reuse").await.unwrap();
    assert_eq!(v.status, Status::Fail);
    assert_eq!(failed(&v), ["pool_at_least_2_gib"], "{:#?}", v.checks);
    assert_eq!(v.checks[0].detail, "inconclusive: pool below 2 GiB");
    assert_eq!(v.figures["inconclusive"], true);
    assert_eq!(v.figures["pool_limit_gib"], 1.5);
    // The first injected line, the fake's own loads (no figure), then the injected 8.0: each with its `pool_cut`.
    let loads = v.figures["pool_loads"].as_array().unwrap();
    assert_eq!(loads.len(), 5, "{loads:?}");
    assert_eq!(loads[0]["pool_limit_gib"], 1.5);
    assert_eq!(loads[4]["pool_limit_gib"], 8.0);
    assert!(loads[0].as_object().unwrap().contains_key("pool_cut"));
}

// -- disconnect --

/// The first request goes on until the client leaves (the fake sees the close within a token), and the second
/// carries the partial reply.
fn script_disconnect(fake: &FakeBackend, reused: u32) {
    fake.push(long_count(800, Duration::from_millis(5)));
    fake.push(reusing(
        "Sure: 41, 42, 43.",
        reused,
        Source::UserQuery,
        63_000_000,
    ));
}

#[tokio::test]
async fn disconnect_case_passes_when_the_second_reuses_the_cut() {
    let rig = rig(Duration::ZERO).await;
    // Pass 1: nothing reused; the first request's cut is in the verdict.
    script_disconnect(&rig.fake, 0);
    let first = run_case(&rig.ctx, "disconnect").await.unwrap();
    assert_eq!(
        failed(&first),
        ["second_reused_floor"],
        "{:#?}",
        first.checks
    );
    let cut = first.figures["first_cut_at"].as_u64().unwrap() as u32;
    assert!(cut > 0);

    script_disconnect(&rig.fake, cut);
    let v = run_case(&rig.ctx, "disconnect").await.unwrap();
    assert_eq!(v.status, Status::Pass, "{:#?}", v.checks);
    assert_eq!(v.figures["second_restored"], "user_query");
    assert_eq!(v.figures["reused_minus_cut"], 0);
    assert!(v.figures["t1_minus_t0_ms"].as_f64().unwrap() < 250.0);

    // Each first request stopped within tokens of the client's leaving: the fake recorded its cancellation, far short
    // of the 800 scripted tokens (the client had read 40 deltas).
    let cancelled = rig.fake.cancellations();
    assert_eq!(cancelled.len(), 2);
    for (_, sent) in cancelled {
        assert!(sent < 200, "{sent} tokens went out before the cancel");
    }
    let runs = rig.fake.runs();
    assert_eq!(runs[2].finish, Some(Finish::Cancelled));
    // The second carried the partial reply as the assistant turn, and "Continue.".
    assert!(runs[3].request.prompt.len() > runs[2].request.prompt.len());
}

/// `disconnect` gates on every load of the log too, as `reuse` does.
#[tokio::test]
async fn disconnect_case_is_inconclusive_when_any_load_had_a_small_pool() {
    let rig = rig(Duration::ZERO).await;
    inject_first_load(&rig.running.log_dir, 1.5);
    inject_load(&rig.running.log_dir, 8.0);
    // Far above any cut: the floor holds, so only the pool fails.
    script_disconnect(&rig.fake, 1_000_000);
    let v = run_case(&rig.ctx, "disconnect").await.unwrap();
    assert_eq!(failed(&v), ["pool_at_least_2_gib"], "{:#?}", v.checks);
    assert_eq!(v.checks[0].detail, "inconclusive: pool below 2 GiB");
    assert_eq!(v.figures["inconclusive"], true);
}

#[tokio::test]
async fn disconnect_case_fails_one_token_short() {
    let rig = rig(Duration::ZERO).await;
    script_disconnect(&rig.fake, 0);
    let first = run_case(&rig.ctx, "disconnect").await.unwrap();
    let cut = first.figures["first_cut_at"].as_u64().unwrap() as u32;
    script_disconnect(&rig.fake, cut - 1);
    let v = run_case(&rig.ctx, "disconnect").await.unwrap();
    assert_eq!(failed(&v), ["second_reused_floor"], "{:#?}", v.checks);
    assert_eq!(v.figures["reused_minus_cut"], -1);
}

// -- idle --

/// A fake that unloads itself after 2 s idle.
fn idle_cfg() -> FakeConfig {
    FakeConfig {
        idle_unload: Some(Duration::from_secs(2)),
        ..FakeConfig::default()
    }
}

#[tokio::test]
async fn idle_case_passes_when_the_unload_comes_in_the_window() {
    let mut rig = rig_with(idle_cfg(), None).await;
    // The case's 30 to 45 s, scaled to the fake's 2 s. The window's low edge is below the 2 s, for the time the reply
    // takes to reach the client after the fake's timer starts.
    rig.ctx.idle_window = (Duration::from_millis(1500), Duration::from_secs(4));
    let v = run_case(&rig.ctx, "idle").await.unwrap();
    assert_eq!(v.status, Status::Pass, "{:#?}", v.checks);
    let after = v.figures["unload_after_s"].as_f64().unwrap();
    assert!((1.5..=4.0).contains(&after), "{after}");
    assert_eq!(rig.fake.idle_unloads_begun(), 1);
    // Loaded by the case, parked by the timer, loaded again by C's next turn.
    assert_eq!(rig.fake.loads().len(), 2);
    assert_eq!(v.figures["c_next_restored"], "none");
    assert_eq!(v.figures["c_next_reused"], 0);
    // The VRAM figures are the fake's `null`: reported as such, and passed only under the hook.
    assert!(v.figures["l1_vram_used_mib"].is_null());
    assert!(check(&v, "vram_within_64_mib")
        .detail
        .starts_with("no figure"));
}

/// The unload comes at the fake's 2 s, which a window of 3 to 4 s calls too early.
#[tokio::test]
async fn idle_case_fails_when_the_unload_is_outside_the_window() {
    let mut rig = rig_with(idle_cfg(), None).await;
    rig.ctx.idle_window = (Duration::from_secs(3), Duration::from_secs(4));
    let v = run_case(&rig.ctx, "idle").await.unwrap();
    assert_eq!(failed(&v), ["idle_unload_in_window"], "{:#?}", v.checks);
    // The case read the rest all the same: the model was idle, and C's next turn answered.
    assert!(check(&v, "health_idle").pass && check(&v, "c_next_turn_answers").pass);
}

/// `idle` needs a fresh server: on a loaded one `/v1/load` writes no line, and the case says so.
#[tokio::test]
async fn idle_case_fails_on_a_server_that_was_already_loaded() {
    // The fake's idle clock starts when it is built, not at the case's `/v1/load`. At 2 s it can run out before the
    // case starts, on a loaded machine, and the `/v1/load` then finds the model unloaded and writes a line: the case
    // would see the two lines of a fresh server. So the idle time is long against any such delay; the window moves
    // with it. The window is not what this test checks.
    let mut rig = rig_with(
        FakeConfig {
            start_loaded: true,
            idle_unload: Some(Duration::from_secs(8)),
            ..FakeConfig::default()
        },
        None,
    )
    .await;
    rig.ctx.idle_window = (Duration::from_millis(7500), Duration::from_secs(12));
    let v = run_case(&rig.ctx, "idle").await.unwrap();
    let c = check(&v, "load_lines");
    assert!(!c.pass, "{:#?}", v.checks);
    assert_eq!(
        c.detail,
        "the server was already loaded: run idle first on a fresh server"
    );
}

/// Without the fake-only hook the missing VRAM figure fails the case: the command never sets it.
#[tokio::test]
async fn idle_case_fails_on_a_missing_vram_figure_without_the_hook() {
    let mut rig = rig_with(idle_cfg(), None).await;
    rig.ctx.idle_window = (Duration::from_millis(1500), Duration::from_secs(4));
    rig.ctx.allow_missing_vram = false;
    let v = run_case(&rig.ctx, "idle").await.unwrap();
    assert_eq!(
        failed(&v),
        ["vram_within_64_mib", "slots_equal"],
        "{:#?}",
        v.checks
    );
}

// -- unload-stream --

#[tokio::test]
async fn unload_stream_case_passes() {
    let rig = rig(Duration::ZERO).await;
    rig.fake.push(long_count(600, Duration::from_millis(5)));
    let v = run_case(&rig.ctx, "unload-stream").await.unwrap();
    assert_eq!(v.status, Status::Pass, "{:#?}", v.checks);
    // The stream was cut by the unload, not run to its end.
    let runs = rig.fake.runs();
    assert_eq!(runs[0].finish, Some(Finish::Unloaded));
    assert!(runs[0].token_times.len() < 600);
    assert_eq!(rig.fake.unloads().len(), 1);
    assert_eq!(rig.fake.loads().len(), 2);
    assert!(v.figures["l1_vram_used_mib"].is_null());
}

/// A stream that ends cleanly (the unload came too late to cut it) is not "an in-band error event".
#[tokio::test]
async fn unload_stream_case_fails_when_the_stream_is_not_cut() {
    // The unload reaches the fake only after the whole reply has been sent (`UnloadAfterRuns`), whatever the client
    // and the engine threads are doing; the 40 tokens are more than the 20 deltas the case reads before it unloads.
    let rig = rig_around(
        FakeConfig {
            load_time: Duration::ZERO,
            ..FakeConfig::default()
        },
        None,
        |fake| Arc::new(UnloadAfterRuns(fake)),
    )
    .await;
    let mut s = Script::text(tokenizer(), &count_text(30));
    s.tokens.truncate(40);
    rig.fake.push(s);
    let v = run_case(&rig.ctx, "unload-stream").await.unwrap();
    assert!(
        failed(&v).contains(&"stream_ends_in_error"),
        "{:#?}",
        v.checks
    );
    // It failed because the stream ended cleanly, not for another reason: the run finished uncut, and the check says
    // the last event was not an error.
    let c = check(&v, "stream_ends_in_error");
    assert!(c.detail.contains("the last is not an error event"), "{c:?}");
    let runs = rig.fake.runs();
    assert!(
        runs[0].finish.is_some() && runs[0].finish != Some(Finish::Unloaded),
        "{:?}",
        runs[0].finish
    );
}

// -- queue --

/// A long request L holds the engine for about 3 s; Q waits behind it, and the server's queue timeout (1 s here, 5 s on
/// server C) answers it 503.
fn script_queue(fake: &FakeBackend) {
    fake.push(long_count(32, Duration::from_millis(100)));
}

#[tokio::test]
async fn queue_case_passes() {
    let mut rig = rig_with(FakeConfig::default(), Some(1)).await;
    // 5.0 to 8.0 s at a queue timeout of 5 s is 1.0 to 3.0 s at 1 s.
    rig.ctx.queue_window = (Duration::from_secs(1), Duration::from_secs(3));
    script_queue(&rig.fake);
    let v = run_case(&rig.ctx, "queue").await.unwrap();
    assert_eq!(v.status, Status::Pass, "{:#?}", v.checks);
    let q = v.figures["q_seconds"].as_f64().unwrap();
    assert!((1.0..3.0).contains(&q), "{q}");
    assert_eq!(v.figures["q_status"], 503);
    assert_eq!(v.figures["l_status"], 200);
    // Q never started: the engine skipped it, so only L ran.
    assert_eq!(rig.fake.requests().len(), 1);
}

/// The case's own window is 5.0 to 8.0 s: a queue timeout of 1 s is outside it.
#[tokio::test]
async fn queue_case_fails_outside_the_default_window() {
    let rig = rig_with(FakeConfig::default(), Some(1)).await;
    script_queue(&rig.fake);
    let v = run_case(&rig.ctx, "queue").await.unwrap();
    assert_eq!(failed(&v), ["q_in_window"], "{:#?}", v.checks);
    assert!(check(&v, "q_times_out_in_the_queue").pass && check(&v, "l_answers").pass);
}

// ---- speed ----

/// A repository root of its own for `speed`: the real `prompts.json`, and a stand-in for the committed source
/// (`sources/encoder.py`) only when `with_source` is true. The stand-in is 9,000 characters and a little more, which the
/// edit prompt quotes the first 9,000 of.
fn speed_root(tag: &str, with_source: bool) -> PathBuf {
    let root = scratch_dir(tag);
    let speed = root.join("testdata/speed");
    std::fs::create_dir_all(speed.join("sources")).unwrap();
    std::fs::copy(
        repo_root().join("testdata/speed/prompts.json"),
        speed.join("prompts.json"),
    )
    .unwrap();
    if with_source {
        let text = "class JSONEncoder(object):\n    pass\n".repeat(300);
        assert!(text.chars().count() > 9_000);
        std::fs::write(speed.join("sources/encoder.py"), text).unwrap();
    }
    root
}

/// A reply of a few tokens that takes measurable prefill and decode time (the log's rates need both), restoring from
/// `source` with `reused` tokens.
fn timed(source: Source, reused: u32) -> Script {
    let mut s = Script::text(tokenizer(), "def f():\n    return 1\n");
    s.prefill = Duration::from_millis(5);
    s.per_token = Duration::from_millis(2);
    s.source = source;
    s.reused = reused;
    s
}

#[tokio::test]
async fn speed_case_reports_ran() {
    let mut rig = rig(Duration::ZERO).await;
    rig.ctx.prompts_root = speed_root("test7-speed", true);
    // Seed 1234's three prompts restore nothing; seed 1235's restore the first's prompt end.
    for _ in 0..3 {
        rig.fake.push(timed(Source::None, 0));
    }
    for _ in 0..3 {
        rig.fake.push(timed(Source::PromptEnd, 10));
    }
    let v = run_case(&rig.ctx, "speed").await.unwrap();
    assert_eq!(v.status, Status::Ran, "{:#?}", v.checks);

    // The six requests, as `speed_llama.py` sends them: the seed outermost, the sampling of prompts.json, thinking off.
    let seen = rig.fake.requests();
    assert_eq!(seen.len(), 6);
    let seeds: Vec<u64> = seen.iter().map(|(_, r)| r.seed).collect();
    assert_eq!(seeds, [1234, 1234, 1234, 1235, 1235, 1235]);
    for (_, r) in &seen {
        assert_eq!(
            (
                r.sampling.temp,
                r.sampling.top_p,
                r.sampling.top_k,
                r.sampling.min_p
            ),
            (0.6, 0.95, 20, 0.0)
        );
        assert_eq!(r.max_tokens, 700);
        assert!(r.think.is_none(), "thinking was on");
    }
    // The prompts are the write prompt, the edit prompt with its 9,000 characters, and the explain prompt.
    let lens: Vec<usize> = seen.iter().take(3).map(|(_, r)| r.prompt.len()).collect();
    assert!(lens[1] > lens[0] && lens[1] > lens[2], "{lens:?}");

    let runs = v.figures["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 6);
    let prompts: Vec<&str> = runs.iter().map(|r| r["prompt"].as_str().unwrap()).collect();
    assert_eq!(
        prompts,
        ["write", "edit", "explain", "write", "edit", "explain"]
    );
    for r in runs {
        assert!(r["decode_tok_s"].as_f64().unwrap() > 0.0, "{r}");
        // The steps' rate beside it: one token fewer over the same time, so a little lower.
        assert!(
            r["decode_steps_tok_s"].as_f64().unwrap() < r["decode_tok_s"].as_f64().unwrap(),
            "{r}"
        );
    }
    assert_eq!(v.figures["decode_steps_tok_s"].as_array().unwrap().len(), 6);
    // Prefill and restore come from each prompt's first seed only.
    for r in &runs[..3] {
        assert!(r["prefill_tok_s"].as_f64().unwrap() > 0.0, "{r}");
        assert_eq!(r["restored"], "none");
    }
    for r in &runs[3..] {
        assert!(
            r.get("prefill_tok_s").is_none() && r.get("restored").is_none(),
            "{r}"
        );
    }
    // Every run carries the completion's hash (the same reply each time here) and the spec keys (`null`: no report).
    let want = format!(
        "{:016x}",
        qwen36::testing::fnv1a64(
            "def f():
    return 1
"
            .as_bytes()
        )
    );
    for r in runs {
        assert_eq!(r["completion_fnv"], want.as_str(), "{r}");
        assert!(r.get("spec_steps").is_some_and(Value::is_null), "{r}");
        assert!(r.get("spec_replays").is_some_and(Value::is_null), "{r}");
    }
    assert!(v.figures["decode_median_tok_s"].as_f64().unwrap() > 0.0);
    assert_eq!(v.figures["artemis_3a_decode_tok_s"], 95.69);
    assert!(v.figures["decode_source"]
        .as_str()
        .unwrap()
        .starts_with("server log"));
    assert_eq!(v.figures["milestone2_decode_tok_s"], 95.57);
    assert_eq!(v.figures["skipped"], serde_json::json!([]));
}

/// The mixed `speed` case carries each run's own spec counts and rate from its request line.
#[tokio::test]
async fn speed_case_carries_each_runs_spec_counts() {
    let mut rig = rig(Duration::ZERO).await;
    rig.ctx.prompts_root = speed_root("test7-speed-spec", true);
    for i in 0..6u64 {
        let mut s = timed(Source::None, 0);
        s.spec = Some(SpecReport {
            steps: 4 + i,
            plain_steps: 1,
            drafted: 16 + 4 * i,
            accepted: 8 + 2 * i,
            replays: i,
            ..SpecReport::default()
        });
        rig.fake.push(s);
    }
    let v = run_case(&rig.ctx, "speed").await.unwrap();
    assert_eq!(v.status, Status::Ran, "{:#?}", v.checks);
    let runs = v.figures["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 6);
    for (i, r) in runs.iter().enumerate() {
        let i = i as u64;
        assert_eq!(r["spec_steps"], 4 + i, "{r}");
        assert_eq!(r["spec_plain_steps"], 1, "{r}");
        assert_eq!(r["spec_drafted"], 16 + 4 * i, "{r}");
        assert_eq!(r["spec_accepted"], 8 + 2 * i, "{r}");
        assert_eq!(r["spec_replays"], i, "{r}");
        assert_eq!(r["spec_accept_rate"], 0.5, "{r}");
    }
}

#[tokio::test]
async fn speed_case_fails_naming_the_missing_edit_file() {
    let mut rig = rig(Duration::ZERO).await;
    rig.ctx.prompts_root = speed_root("test7-speed-missing", false);
    let v = run_case(&rig.ctx, "speed").await.unwrap();
    assert_eq!(v.status, Status::Fail, "{:#?}", v.checks);
    let c = check(&v, "edit_source");
    assert!(!c.pass && c.detail.contains("encoder.py"), "{c:?}");
    // Nothing was sent: the figures would have been the other two prompts'.
    assert!(rig.fake.requests().is_empty());
}

#[tokio::test]
async fn speed_case_skips_a_missing_edit_file_with_the_hook() {
    let mut rig = rig(Duration::ZERO).await;
    rig.ctx.prompts_root = speed_root("test7-speed-skip", false);
    rig.ctx.allow_missing_edit = true;
    for _ in 0..4 {
        rig.fake.push(timed(Source::None, 0));
    }
    let v = run_case(&rig.ctx, "speed").await.unwrap();
    assert_eq!(v.status, Status::Ran, "{:#?}", v.checks);
    assert_eq!(v.figures["runs"].as_array().unwrap().len(), 4);
    let c = check(&v, "edit_source");
    assert!(
        c.pass && c.detail.contains("skipped") && c.detail.contains("encoder.py"),
        "{c:?}"
    );
    assert_eq!(v.figures["skipped"][0]["prompt"], "edit");
}

#[tokio::test]
async fn speed_case_fails_when_a_run_does_not_answer() {
    let mut rig = rig(Duration::ZERO).await;
    rig.ctx.prompts_root = speed_root("test7-speed-500", true);
    let mut broken = timed(Source::None, 0);
    broken.fail_at = Some((0, server::fake::Fail::Internal));
    rig.fake.push(broken);
    for _ in 0..5 {
        rig.fake.push(timed(Source::None, 0));
    }
    let v = run_case(&rig.ctx, "speed").await.unwrap();
    assert_eq!(v.status, Status::Fail, "{:#?}", v.checks);
    assert!(!check(&v, "runs_answer").pass);
    assert!(check(&v, "runs_answer").detail.contains("write seed 1234"));
    // The other five were measured all the same.
    assert_eq!(v.figures["decode_tok_s"].as_array().unwrap().len(), 5);
}

/// An unanswered run is `runs_answer`'s finding alone: `decode_rates_logged` speaks of the logs it did read.
#[tokio::test]
async fn speed_case_blames_only_runs_answer_for_an_unanswered_run() {
    let mut rig = rig(Duration::ZERO).await;
    rig.ctx.prompts_root = speed_root("test7-speed-only500", true);
    let mut broken = timed(Source::None, 0);
    broken.fail_at = Some((0, server::fake::Fail::Internal));
    rig.fake.push(broken);
    for _ in 0..5 {
        rig.fake.push(timed(Source::None, 0));
    }
    let v = run_case(&rig.ctx, "speed").await.unwrap();
    assert_eq!(v.failed(), ["runs_answer"], "{:#?}", v.checks);
    assert!(check(&v, "decode_rates_logged")
        .detail
        .starts_with("5 runs"));
}

/// The hook is the `edit` prompt's only: another prompt's missing source still fails the case.
#[tokio::test]
async fn speed_case_hook_does_not_excuse_another_prompts_source() {
    let mut rig = rig(Duration::ZERO).await;
    let root = speed_root("test7-speed-hook", false);
    let path = root.join("testdata/speed/prompts.json");
    let mut defs: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    defs["prompts"][0]["content"] =
        serde_json::json!([{"source": "cpython-json-encoder", "chars": 10}]);
    std::fs::write(&path, serde_json::to_vec(&defs).unwrap()).unwrap();
    rig.ctx.prompts_root = root;
    rig.ctx.allow_missing_edit = true;
    let v = run_case(&rig.ctx, "speed").await.unwrap();
    assert_eq!(v.status, Status::Fail, "{:#?}", v.checks);
    assert!(!check(&v, "write_source").pass);
    assert!(check(&v, "edit_source").detail.contains("skipped"));
    assert!(rig.fake.requests().is_empty());
}

// ---- speed-fresh ----

/// `speed-fresh` for the decode run `prompt`:`seed`.
fn decode_run(prompt: &str, seed: u64) -> Option<SpeedRun> {
    Some(SpeedRun::Decode {
        prompt: prompt.into(),
        seed,
    })
}

/// One request, with the sampling of `prompts.json`, and the figures read from its log line: the decode steps' rate
/// is `decode_tokens / decode_ms`, one token fewer than `decode_tok_s`'s numerator.
#[tokio::test]
async fn speed_fresh_runs_one_request_and_reads_its_figures() {
    let mut rig = rig(Duration::ZERO).await;
    // The write prompt has no source: the missing `encoder.py` is not its concern.
    rig.ctx.prompts_root = speed_root("test7-speed-fresh", false);
    rig.ctx.speed_run = decode_run("write", 1234);
    rig.fake.push(timed(Source::None, 0));
    let v = run_case(&rig.ctx, "speed-fresh").await.unwrap();
    assert_eq!(v.status, Status::Ran, "{:#?}", v.checks);
    assert_eq!(
        v.checks.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        ["answers", "logged", "fresh_engine"]
    );

    let seen = rig.fake.requests();
    assert_eq!(seen.len(), 1);
    let r = &seen[0].1;
    assert_eq!(r.seed, 1234);
    assert_eq!(
        (
            r.sampling.temp,
            r.sampling.top_p,
            r.sampling.top_k,
            r.sampling.min_p
        ),
        (0.6, 0.95, 20, 0.0)
    );
    assert_eq!(r.max_tokens, 700);
    assert!(r.think.is_none(), "thinking was on");

    let f = &v.figures;
    assert_eq!(
        (f["prompt"].as_str(), f["seed"].as_u64()),
        (Some("write"), Some(1234))
    );
    let completion = f["completion_tokens"].as_u64().unwrap();
    let decode_tokens = f["decode_tokens"].as_u64().unwrap();
    assert!(completion > 1);
    assert_eq!(decode_tokens, completion - 1);
    let ms = f["decode_ms"].as_f64().unwrap();
    assert!(ms > 0.0, "{f:?}");
    let steps = f["decode_steps_tok_s"].as_f64().unwrap();
    assert!(
        (steps - decode_tokens as f64 / ms * 1000.0).abs() < 1e-6 * steps,
        "{f:?}"
    );
    // `decode_tok_s` keeps its earlier meaning: all the completion's tokens over the decode time.
    let all = f["decode_tok_s"].as_f64().unwrap();
    assert!(
        (all - completion as f64 / ms * 1000.0).abs() < 1e-6 * all,
        "{f:?}"
    );
    assert!(f["prefill_tok_s"].as_f64().unwrap() > 0.0);
    assert!(f["prefilled_tokens"].as_u64().unwrap() > 0);
    assert!(f["prompt_tokens"].as_u64().unwrap() > 0);
    // The fake reports no misses: `null`, not 0.
    assert!(f["decode_misses"].is_null(), "{f:?}");
    assert!(!f.contains_key("long_prompt_tokens_reference"));
}

/// The spec counts of the request line and a hash of the completion are in the figures: the record the gate's
/// `serve-speed-spec` runs read. The hash is the FNV-1a 64 of the reply's text as 16 hex digits; the text itself is in
/// no part of the verdict.
#[tokio::test]
async fn speed_fresh_figures_carry_the_spec_counts_and_a_completion_hash() {
    let mut rig = rig(Duration::ZERO).await;
    rig.ctx.prompts_root = speed_root("test7-speed-fresh-spec", false);
    rig.ctx.speed_run = decode_run("write", 1234);
    let text = "def f():
    return 1
";
    let mut s = timed(Source::None, 0);
    s.spec = Some(SpecReport {
        steps: 5,
        plain_steps: 2,
        drafted: 12,
        accepted: 9,
        replays: 3,
        ..SpecReport::default()
    });
    rig.fake.push(s);
    let v = run_case(&rig.ctx, "speed-fresh").await.unwrap();
    assert_eq!(v.status, Status::Ran, "{:#?}", v.checks);

    let f = &v.figures;
    assert_eq!(
        (
            &f["spec_steps"],
            &f["spec_plain_steps"],
            &f["spec_drafted"],
            &f["spec_accepted"],
            &f["spec_accept_rate"],
            &f["spec_replays"]
        ),
        (
            &serde_json::json!(5),
            &serde_json::json!(2),
            &serde_json::json!(12),
            &serde_json::json!(9),
            &serde_json::json!(0.75),
            &serde_json::json!(3)
        ),
        "{f:?}"
    );
    let want = format!("{:016x}", qwen36::testing::fnv1a64(text.as_bytes()));
    assert_eq!(want.len(), 16);
    assert_eq!(f["completion_fnv"], want.as_str(), "{f:?}");
    // The text is in no part of the verdict, in any form of it.
    let all = serde_json::to_string(&v).unwrap();
    assert!(!all.contains("return 1"), "{all}");
}

/// Speculation off: the counts are `null` (not 0), and the hash is there all the same, so an off run and an on run
/// compare.
#[tokio::test]
async fn speed_fresh_spec_counts_are_null_with_speculation_off() {
    let mut rig = rig(Duration::ZERO).await;
    rig.ctx.prompts_root = speed_root("test7-speed-fresh-nospec", false);
    rig.ctx.speed_run = decode_run("write", 1234);
    rig.fake.push(timed(Source::None, 0));
    let v = run_case(&rig.ctx, "speed-fresh").await.unwrap();
    assert_eq!(v.status, Status::Ran, "{:#?}", v.checks);
    for k in [
        "spec_steps",
        "spec_plain_steps",
        "spec_drafted",
        "spec_accepted",
        "spec_accept_rate",
        "spec_replays",
    ] {
        assert!(v.figures.get(k).is_some_and(Value::is_null), "{k}");
    }
    assert_eq!(v.figures["completion_fnv"].as_str().map(str::len), Some(16));
}

/// A run that restores from the cache is not a fresh engine's: the rate would be a warm one's.
#[tokio::test]
async fn speed_fresh_fails_fresh_engine_on_a_restore() {
    let mut rig = rig(Duration::ZERO).await;
    rig.ctx.prompts_root = speed_root("test7-speed-fresh-restore", false);
    rig.ctx.speed_run = decode_run("write", 1235);
    rig.fake.push(timed(Source::PromptEnd, 10));
    let v = run_case(&rig.ctx, "speed-fresh").await.unwrap();
    assert_eq!(v.status, Status::Fail, "{:#?}", v.checks);
    assert_eq!(failed(&v), ["fresh_engine"], "{:#?}", v.checks);
    let c = check(&v, "fresh_engine");
    assert!(
        c.detail.contains("prompt_end") && c.detail.contains("10"),
        "{c:?}"
    );
    assert!(check(&v, "answers").pass && check(&v, "logged").pass);
}

/// `long`: the golden prompt 20 without its ChatML wrapper, one token asked for, greedy. The prefill is logged and
/// there is no decode rate: no decode steps ran.
#[tokio::test]
async fn speed_fresh_long_logs_prefill_and_no_decode_rate() {
    let mut rig = rig(Duration::ZERO).await;
    rig.ctx.speed_run = Some(SpeedRun::Long);
    let mut one = timed(Source::None, 0);
    one.tokens = tokenizer().encode("x", true);
    rig.fake.push(one);
    let v = run_case(&rig.ctx, "speed-fresh").await.unwrap();
    assert_eq!(v.status, Status::Ran, "{:#?}", v.checks);
    assert!(check(&v, "logged").pass && check(&v, "prompt_unwrapped").pass);

    let seen = rig.fake.requests();
    assert_eq!(seen.len(), 1);
    let r = &seen[0].1;
    assert_eq!(r.max_tokens, 1);
    assert_eq!(r.sampling.temp, 0.0);
    assert!(r.think.is_none(), "thinking was on");
    // The unwrapped content, rendered by the committed template and tokenised, is the golden prompt's tokens exactly:
    // the wrapper is the template's to add, not the case's (it renders back to the same length).
    let golden =
        std::fs::read(repo_root().join("testdata/golden/prompts/20-long-option-rs.tokens"))
            .unwrap();
    let golden: Vec<u32> = golden
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    assert_eq!(golden.len(), 27_416);
    assert_eq!(r.prompt.len(), 27_416, "{} tokens", r.prompt.len());
    assert!(
        r.prompt == golden,
        "the rendered prompt differs from the golden tokens"
    );

    let f = &v.figures;
    assert_eq!(f["prompt"], "long");
    assert_eq!(f["long_prompt_tokens_reference"], 27416);
    assert_eq!(f["completion_tokens"], 1);
    assert_eq!(f["decode_tokens"], 0);
    assert_eq!(
        f["completion_fnv"].as_str().map(str::len),
        Some(16),
        "{f:?}"
    );
    assert!(f["decode_steps_tok_s"].is_null(), "{f:?}");
    assert!(f["prefilled_tokens"].as_u64().unwrap() > 5_000);
    assert!(f["prefill_tok_s"].as_f64().unwrap() > 0.0);
}

/// A request that errors is `answers`' finding, with no figures to report.
#[tokio::test]
async fn speed_fresh_fails_answers_when_the_request_errors() {
    let mut rig = rig(Duration::ZERO).await;
    rig.ctx.prompts_root = speed_root("test7-speed-fresh-nolog", false);
    rig.ctx.speed_run = decode_run("write", 1234);
    let mut broken = timed(Source::None, 0);
    broken.fail_at = Some((0, server::fake::Fail::Internal));
    rig.fake.push(broken);
    let v = run_case(&rig.ctx, "speed-fresh").await.unwrap();
    assert_eq!(v.status, Status::Fail, "{:#?}", v.checks);
    let answers = check(&v, "answers");
    assert!(!answers.pass);
    assert_eq!(
        answers.detail.matches("status 500").count(),
        1,
        "{answers:?}"
    );
}

// ---- chat-tools ----

/// A `<tool_call>` the model would write: `name` with `params` (key, value) in order.
fn tool_call(name: &str, params: &[(&str, &str)]) -> String {
    let mut s = format!("<tool_call>\n<function={name}>\n");
    for (k, v) in params {
        s.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
    }
    s.push_str("</function>\n</tool_call>");
    s
}

fn lookup_call() -> String {
    tool_call("lookup", &[("query", "scheduler retry policy")])
}

fn read_entry_call() -> String {
    tool_call("read_entry", &[("entry", "scheduler-retry")])
}

/// Scripts `texts` as the replies to the loop's requests, in order. The loop asks for low effort, so thinking is on:
/// each reply is a short reasoning, its close, then the text.
fn script_texts(fake: &FakeBackend, texts: &[String]) {
    for t in texts {
        say(fake, &format!("Let me think.\n</think>\n\n{t}"));
    }
}

/// A search, a read, then the answer.
#[tokio::test]
async fn chat_tools_passes_a_two_call_loop() {
    let rig = rig(Duration::ZERO).await;
    script_texts(
        &rig.fake,
        &[
            lookup_call(),
            read_entry_call(),
            "Failed jobs are retried three times.".into(),
        ],
    );
    let v = run_case(&rig.ctx, "chat-tools").await.unwrap();
    assert_eq!(v.status, Status::Pass, "{:#?}", v.checks);
    assert_eq!(
        v.checks.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        [
            "one_choice_each",
            "calls_parse",
            "no_function_call",
            "ends_in_text_within_4",
            "finish_reasons_match",
            "at_least_one_call",
            "arguments_match_schema",
            "final_text_non_empty",
            "no_malformed_calls",
            "loop_within_150_s",
        ]
    );

    let f = &v.figures;
    assert_eq!(
        (f["requests"].as_u64(), f["tool_calls"].as_u64()),
        (Some(3), Some(2))
    );
    assert!(f["loop_ms"].as_f64().unwrap() > 0.0, "{f:?}");
    let ms = f["request_ms"].as_array().unwrap();
    assert_eq!(ms.len(), 3);
    assert!(ms.iter().all(|m| m.as_f64().unwrap() > 0.0), "{f:?}");

    // The client's request: low effort, 4,096 tokens, and the conversation grows by the echo (the assistant message
    // with its call, then one tool message per call).
    let seen = rig.fake.requests();
    assert_eq!(seen.len(), 3);
    for (i, (_, r)) in seen.iter().enumerate() {
        assert_eq!(r.max_tokens, 4096, "request {i}");
        assert_eq!(
            r.think.as_ref().and_then(|t| t.budget),
            Some(1024),
            "request {i}"
        );
        assert_eq!((r.sampling.temp, r.seed), (0.0, 7), "request {i}");
    }
    assert!(
        seen[1].1.prompt.len() > seen[0].1.prompt.len()
            && seen[2].1.prompt.len() > seen[1].1.prompt.len()
    );
}

/// Calls on the first three requests and a fourth call on the fourth: the client errors there.
#[tokio::test]
async fn chat_tools_fails_a_call_on_the_fourth_request() {
    let rig = rig(Duration::ZERO).await;
    script_texts(
        &rig.fake,
        &[
            lookup_call(),
            read_entry_call(),
            lookup_call(),
            read_entry_call(),
        ],
    );
    let v = run_case(&rig.ctx, "chat-tools").await.unwrap();
    assert_eq!(v.status, Status::Fail);
    assert_eq!(
        failed(&v),
        [
            "ends_in_text_within_4",
            "finish_reasons_match",
            "final_text_non_empty"
        ],
        "{:#?}",
        v.checks
    );
    let c = check(&v, "ends_in_text_within_4");
    assert!(c.detail.contains("request 4"), "{c:?}");
    assert_eq!(rig.fake.requests().len(), 4);
    assert_eq!(v.figures["requests"], 4);
}

/// Two calls in the first reply and two in the second make four: the client errors at the fourth, on request 2.
#[tokio::test]
async fn chat_tools_fails_a_fourth_tool_call() {
    let rig = rig(Duration::ZERO).await;
    let two = format!("{}\n{}", lookup_call(), read_entry_call());
    script_texts(&rig.fake, &[two.clone(), two, "Done.".into()]);
    let v = run_case(&rig.ctx, "chat-tools").await.unwrap();
    assert_eq!(
        failed(&v),
        [
            "ends_in_text_within_4",
            "finish_reasons_match",
            "final_text_non_empty"
        ],
        "{:#?}",
        v.checks
    );
    let c = check(&v, "ends_in_text_within_4");
    assert!(
        c.detail.contains("4 tool calls") && c.detail.contains("request 2"),
        "{c:?}"
    );
    // The loop stopped where the client would have: no request 3.
    assert_eq!(rig.fake.requests().len(), 2);
    assert_eq!(v.figures["tool_calls"], 4);
}

/// A call with a key the schema does not declare (`additionalProperties: false`) fails the schema check alone.
#[tokio::test]
async fn chat_tools_fails_extra_argument_keys() {
    let rig = rig(Duration::ZERO).await;
    script_texts(
        &rig.fake,
        &[
            tool_call("lookup", &[("query", "retry policy"), ("limit", "5")]),
            read_entry_call(),
            "Three times.".into(),
        ],
    );
    let v = run_case(&rig.ctx, "chat-tools").await.unwrap();
    assert_eq!(failed(&v), ["arguments_match_schema"], "{:#?}", v.checks);
    assert!(check(&v, "arguments_match_schema")
        .detail
        .contains("1 extra key"));
    assert!(check(&v, "calls_parse").pass && check(&v, "ends_in_text_within_4").pass);
}

/// A model that answers without searching proves nothing about the loop.
#[tokio::test]
async fn chat_tools_fails_when_no_tool_is_called() {
    let rig = rig(Duration::ZERO).await;
    script_texts(&rig.fake, &["I know this already.".into()]);
    let v = run_case(&rig.ctx, "chat-tools").await.unwrap();
    assert_eq!(failed(&v), ["at_least_one_call"], "{:#?}", v.checks);
    assert_eq!(v.figures["tool_calls"], 0);
    assert_eq!(v.figures["requests"], 1);
}

/// A loop that ends in an empty answer fails `final_text_non_empty` alone.
#[tokio::test]
async fn chat_tools_fails_an_empty_final_text() {
    let rig = rig(Duration::ZERO).await;
    script_texts(&rig.fake, &[lookup_call(), read_entry_call(), "  ".into()]);
    let v = run_case(&rig.ctx, "chat-tools").await.unwrap();
    assert_eq!(failed(&v), ["final_text_non_empty"], "{:#?}", v.checks);
}

/// A call the parser could not read goes out as content and counts in the log's `malformed_calls`.
#[tokio::test]
async fn chat_tools_fails_a_malformed_call_in_the_log() {
    let rig = rig(Duration::ZERO).await;
    script_texts(
        &rig.fake,
        &[
            "<tool_call>\nnot a call\n</tool_call>".into(),
            "Nothing.".into(),
        ],
    );
    let v = run_case(&rig.ctx, "chat-tools").await.unwrap();
    let names = failed(&v);
    assert!(names.contains(&"no_malformed_calls"), "{:#?}", v.checks);
    assert!(names.contains(&"at_least_one_call"), "{:#?}", v.checks);
}

/// The finish reasons, usage and shape problems are reported per request; of them only the finish reasons are gated
/// (`finish_reasons_match`): a final reply cut at the token limit fails that check alone.
#[tokio::test]
async fn chat_tools_reports_finish_reasons_and_gates_them() {
    let rig = rig(Duration::ZERO).await;
    // More tokens than the request's 4,096 (the fake ends a request at `max_tokens` with `length`).
    script_texts(
        &rig.fake,
        &[lookup_call(), read_entry_call(), "word ".repeat(5000)],
    );
    let v = run_case(&rig.ctx, "chat-tools").await.unwrap();
    assert_eq!(failed(&v), ["finish_reasons_match"], "{:#?}", v.checks);
    assert_eq!(
        check(&v, "finish_reasons_match").detail,
        "request 3 finished length, want stop"
    );
    let f = &v.figures;
    assert_eq!(
        f["finish_reasons"],
        serde_json::json!(["tool_calls", "tool_calls", "length"])
    );
    assert_eq!(f["shape_problems"], serde_json::json!([[], [], []]));
    for k in ["completion_tokens", "reasoning_tokens"] {
        let a = f[k].as_array().unwrap();
        assert!(
            a.len() == 3 && a.iter().all(Value::is_u64),
            "{k}: {:?}",
            f[k]
        );
    }
}

/// A reply that is not 200 keeps only its status in the figures: the server's message is not copied.
#[tokio::test]
async fn chat_tools_reports_only_the_status_of_a_failed_reply() {
    let rig = rig(Duration::ZERO).await;
    let mut broken = Script::text(tokenizer(), "x");
    broken.fail_at = Some((0, server::fake::Fail::Internal));
    rig.fake.push(broken);
    let v = run_case(&rig.ctx, "chat-tools").await.unwrap();
    let f = &v.figures;
    assert_eq!(f["finish_reasons"], serde_json::json!([null]));
    assert_eq!(f["shape_problems"], serde_json::json!([["status 500"]]));
    assert_eq!(f["completion_tokens"], serde_json::json!([null]));
    assert_eq!(
        check(&v, "no_function_call").detail,
        "0 replies (no replies)"
    );
}

/// No verdict field holds a call's argument, a key the model chose, or the model's text.
#[tokio::test]
async fn chat_tools_details_hold_no_argument_text() {
    const MARKER: &str = "zq7marker-argument-value";
    const KEY_MARKER: &str = "zq7marker_key";
    const TEXT_MARKER: &str = "zq7marker-final-text";
    let rig = rig(Duration::ZERO).await;
    // A query over its 200 characters and an extra key: both fail the schema check, and both are the details' risk.
    let long = MARKER.repeat(10);
    assert!(long.len() > 200);
    script_texts(
        &rig.fake,
        &[
            tool_call("lookup", &[("query", &long), (KEY_MARKER, MARKER)]),
            tool_call("read_entry", &[("entry", MARKER)]),
            format!("The answer is {TEXT_MARKER}."),
        ],
    );
    let v = run_case(&rig.ctx, "chat-tools").await.unwrap();
    // The loop reached its final text, so the text marker was in a reply the case read.
    assert_eq!(v.figures["requests"], 3, "{:#?}", v.checks);
    assert!(check(&v, "final_text_non_empty").pass, "{:#?}", v.checks);
    assert!(!check(&v, "arguments_match_schema").pass, "{:#?}", v.checks);
    let json = serde_json::to_string(&v).unwrap();
    for m in [MARKER, KEY_MARKER, TEXT_MARKER, "zq7marker"] {
        assert!(!json.contains(m), "the verdict holds {m}: {json}");
    }
}

/// The four per-source counts reach `speed-fresh`'s figures and each `speed` run, and are `null` with speculation off.
#[tokio::test]
async fn speed_fresh_figures_carry_the_per_source_counts() {
    let mut rig = rig(Duration::ZERO).await;
    rig.ctx.prompts_root = speed_root("test7-speed-fresh-sources", false);
    rig.ctx.speed_run = decode_run("write", 1234);
    let mut s = timed(Source::None, 0);
    s.spec = Some(SpecReport {
        steps: 5,
        plain_steps: 2,
        drafted: 12,
        accepted: 9,
        replays: 3,
        drafted_lookup: 7,
        accepted_lookup: 6,
        drafted_mtp: 5,
        accepted_mtp: 3,
    });
    rig.fake.push(s);
    let v = run_case(&rig.ctx, "speed-fresh").await.unwrap();
    assert_eq!(v.status, Status::Ran, "{:#?}", v.checks);
    let f = &v.figures;
    assert_eq!(f["spec_drafted_lookup"], 7, "{f:?}");
    assert_eq!(f["spec_accepted_lookup"], 6, "{f:?}");
    assert_eq!(f["spec_drafted_mtp"], 5, "{f:?}");
    assert_eq!(f["spec_accepted_mtp"], 3, "{f:?}");

    // Speculation off: all four are present and null, not 0.
    rig.fake.push(timed(Source::None, 0));
    let v = run_case(&rig.ctx, "speed-fresh").await.unwrap();
    for k in [
        "spec_drafted_lookup",
        "spec_accepted_lookup",
        "spec_drafted_mtp",
        "spec_accepted_mtp",
    ] {
        assert!(v.figures.get(k).is_some_and(Value::is_null), "{k}");
    }

    // The mixed `speed` case carries them on each run.
    rig.ctx.prompts_root = speed_root("test7-speed-sources", true);
    for i in 0..6u64 {
        let mut s = timed(Source::None, 0);
        s.spec = Some(SpecReport {
            steps: 4,
            drafted: 10 + i,
            accepted: 5 + i,
            drafted_lookup: 4,
            accepted_lookup: 2,
            drafted_mtp: 6 + i,
            accepted_mtp: 3 + i,
            ..SpecReport::default()
        });
        rig.fake.push(s);
    }
    let v = run_case(&rig.ctx, "speed").await.unwrap();
    let runs = v.figures["runs"].as_array().unwrap();
    for (i, r) in runs.iter().enumerate() {
        let i = i as u64;
        assert_eq!(r["spec_drafted_lookup"], 4, "{r}");
        assert_eq!(r["spec_accepted_lookup"], 2, "{r}");
        assert_eq!(r["spec_drafted_mtp"], 6 + i, "{r}");
        assert_eq!(r["spec_accepted_mtp"], 3 + i, "{r}");
    }
}
