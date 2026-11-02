//! `speed`: the three speed prompts through the server, reported not gated; and
//! `speed-fresh`, one of those requests (or the long prefill) on a server process of its own.
use super::{fig, line_of, CHAT};
use crate::test7::judge;
use crate::test7::log::RequestLine;
use crate::test7::{Check, Ctx, SpeedRun, Verdict};
use anyhow::{Context, Result};
use qwen36::testing::fnv1a64;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

/// The decode median `artemis speed` measured when the server was first released (see `docs/evidence.md`), tok/s: the
/// figure the case's own median is put beside.
pub const DECODE_3A_TOK_S: f64 = 95.69;
/// The fast-engine gate's own decode median (`artemis speed`; see `docs/evidence.md`), tok/s. It is
/// Artemis's figure, not llama.cpp's.
pub const DECODE_M2_TOK_S: f64 = 95.57;

/// Golden prompt 20 (the long prefill) is stored as the chat template rendered it: the user turn's start, and the
/// end of the user turn with the assistant's start and an empty thinking block. `speed-fresh` sends the content
/// between them as one user message, and the server's template adds the wrapper again.
const USER_PREFIX: &str = "<|im_start|>user\n";
const USER_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
/// The golden prompt `speed-fresh`'s `long` run sends (under the repository root), and the token count of its
/// rendering in the committed tokens file (`testdata/golden/prompts/20-long-option-rs.tokens`), which the server's
/// `prompt_tokens` is printed beside.
const LONG_PROMPT: &str = "testdata/golden/prompts/20-long-option-rs.txt";
const LONG_PROMPT_TOKENS: u64 = 27_416;

/// `text` without the user turn's wrapper (`USER_PREFIX` at the start and `USER_SUFFIX` at the end), or `None` when
/// either is missing.
pub fn unwrap_chatml_user(text: &str) -> Option<&str> {
    text.strip_prefix(USER_PREFIX)?.strip_suffix(USER_SUFFIX)
}

/// The rate of the decode steps in a request line: `decode_tokens` over `decode_ms`, tok/s. `None` (not 0) when no
/// step ran, when either figure is missing or when the time is not positive. (`decode_tok_s` counts the completion's
/// first token too, which came out of the prefill.)
fn decode_steps_tok_s(line: Option<&Value>) -> Option<f64> {
    let line = line?;
    let tokens = line["decode_tokens"].as_u64().filter(|&t| t > 0)?;
    let ms = line["decode_ms"].as_f64().filter(|&m| m > 0.0)?;
    Some(tokens as f64 / ms * 1000.0)
}

/// The sampling of one request: the body's fields beyond `seed`, `chat_template_kwargs` and the model's own, and
/// `max_tokens`.
struct Sampling {
    fields: Value,
    max_tokens: u32,
}

/// `prompts.json`'s `sampling`, as a request's fields.
fn sampling_of(s: &Value) -> Result<Sampling> {
    let max_tokens = s["max_tokens"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .context("prompts.json's sampling has no max_tokens")?;
    Ok(Sampling {
        fields: json!({"temperature": s["temp"], "top_p": s["top_p"], "top_k": s["top_k"], "min_p": s["min_p"]}),
        max_tokens,
    })
}

/// The request line's ten speculation counts (the sums, the rate, the replays, each source's drafted and accepted), as
/// `speed` and `speed-fresh` carry them (`null` with speculation off).
const SPEC_KEYS: [&str; 10] = [
    "spec_steps",
    "spec_plain_steps",
    "spec_drafted",
    "spec_accepted",
    "spec_accept_rate",
    "spec_replays",
    "spec_drafted_lookup",
    "spec_accepted_lookup",
    "spec_drafted_mtp",
    "spec_accepted_mtp",
];

/// What one request left: its reply's status and shape problems, and its log line.
struct RunFigures {
    status: u16,
    problems: Vec<String>,
    line: Option<RequestLine>,
    /// The FNV-1a 64 of the reply's `choices[0].message.content` as 16 hex digits (never the text); `None` when the
    /// request was not answered 200 or the reply has shape problems, so a missing completion never compares equal to
    /// an empty one.
    completion_fnv: Option<String>,
}

impl RunFigures {
    fn answered(&self) -> bool {
        self.status == 200
    }

    /// The figure `k` of the log line, `null` where there is none.
    fn fig(&self, k: &str) -> Value {
        fig(self.line.as_deref(), k)
    }

    /// The completion's hash, `null` where the request was not answered or its reply was malformed.
    fn completion_fnv(&self) -> Value {
        json!(self.completion_fnv)
    }

    fn steps_tok_s(&self) -> Option<f64> {
        decode_steps_tok_s(self.line.as_deref())
    }
}

/// One non-streaming request, thinking off: the user message `prompt` with `seed`, sampled as `sampling` says. Its
/// reply is read and its log line fetched (a missing line is `None`, the caller's finding).
async fn one_run(ctx: &Ctx, prompt: &str, seed: u64, sampling: &Sampling) -> Result<RunFigures> {
    let mut extra = sampling.fields.clone();
    extra["seed"] = json!(seed);
    extra["chat_template_kwargs"] = json!({"enable_thinking": false});
    let body = super::body(
        json!([{"role": "user", "content": prompt}]),
        extra,
        sampling.max_tokens,
        false,
    );
    let r = ctx.http.send("POST", CHAT, Some(&body)).await?;
    let reply = judge::openai_reply(r.status, &r.json().unwrap_or(Value::Null));
    let line = if reply.status == 200 {
        line_of(ctx, &reply.id).await?
    } else {
        None
    };
    let completion_fnv = completion_hash(&reply);
    Ok(RunFigures {
        status: reply.status,
        problems: reply.problems,
        line,
        completion_fnv,
    })
}

/// The FNV-1a 64 of `reply`'s content as 16 hex digits; `None` for a reply that wasn't 200 or has shape problems.
fn completion_hash(reply: &judge::Reply) -> Option<String> {
    (reply.status == 200 && reply.problems.is_empty())
        .then(|| format!("{:016x}", fnv1a64(reply.text.as_bytes())))
}

/// What stops a prompt's message from being built.
enum Unusable {
    /// A pinned source that is not on disk (the committed `edit` source, `speed/sources/encoder.py`, in a root without it).
    Missing(PathBuf),
    /// A source with fewer characters than the prompt quotes.
    Short(PathBuf, usize, usize),
}

/// The prompt `p`'s one user message, as `tools/py/build_speed_prompts.py`'s `message()` builds it: its text parts, and
/// each source part's first `chars` characters (characters, as Python slices a `str`; the file's bytes are read as they
/// are, UTF-8, no line-ending translation). The sources' sha256 is the runner's `--stage check`'s to verify.
fn message(root: &Path, defs: &Value, p: &Value) -> Result<std::result::Result<String, Unusable>> {
    let mut out = String::new();
    for part in p["content"].as_array().context("a prompt has no content")? {
        if let Some(t) = part["text"].as_str() {
            out.push_str(t);
            continue;
        }
        let id = part["source"]
            .as_str()
            .context("a content part is neither text nor a source")?;
        let chars = part["chars"]
            .as_u64()
            .context("a source part has no `chars`")? as usize;
        let rel = defs["sources"]
            .as_array()
            .and_then(|s| s.iter().find(|s| s["id"] == id))
            .and_then(|s| s["path"].as_str())
            .with_context(|| format!("prompts.json has no source {id}"))?;
        let path = root.join("testdata").join(rel);
        let text = match std::fs::read(&path) {
            Ok(b) => {
                String::from_utf8(b).with_context(|| format!("{} is not UTF-8", path.display()))?
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Err(Unusable::Missing(path)))
            }
            Err(e) => return Err(e).with_context(|| format!("cannot read {}", path.display())),
        };
        let have = text.chars().count();
        if have < chars {
            return Ok(Err(Unusable::Short(path, have, chars)));
        }
        out.extend(text.chars().take(chars));
    }
    Ok(Ok(out))
}

/// The failed `<id>_source` check for a prompt that can't be built.
fn unusable_check(id: &str, u: &Unusable) -> Check {
    let detail = match u {
        Unusable::Missing(f) => format!(
            "{} is missing: it is committed under testdata/speed/sources, restore it from git",
            f.display()
        ),
        Unusable::Short(f, have, want) => format!(
            "{} has {have} characters; the prompt quotes {want}",
            f.display()
        ),
    };
    Check::new(format!("{id}_source"), false, detail)
}

/// `decode_rates_logged`: passes when no answered run lacks a `decode_tok_s` in its log line (an unanswered run is
/// `runs_answer`'s finding, not this one's); when it fails, the detail names the runs.
fn decode_rates_check(unlogged: &[String], logged: usize) -> Check {
    let detail = if unlogged.is_empty() {
        format!("{logged} runs have a decode_tok_s in their log line")
    } else {
        format!(
            "no decode_tok_s in the log line of: {}",
            unlogged.join(", ")
        )
    };
    Check::new("decode_rates_logged", unlogged.is_empty(), detail)
}

/// The median as Python's `statistics.median` takes it (the mean of the middle two of an even count).
fn median(mut v: Vec<f64>) -> Option<f64> {
    v.sort_by(f64::total_cmp);
    match v.len() {
        0 => None,
        n if n % 2 == 1 => Some(v[n / 2]),
        n => Some((v[n / 2 - 1] + v[n / 2]) / 2.0),
    }
}

/// `speed` (on server A): each of `testdata/speed/prompts.json`'s prompts (`write`, `edit`, `explain`), for each of its
/// seeds in turn (the seed outermost, as `tools/py/speed_llama.py` runs them), as one non-streaming request with
/// thinking off (`chat_template_kwargs.enable_thinking: false`, as the Python sends it), `max_tokens` and the sampling
/// of the file's `sampling` (`temp` as `temperature`, `top_p`, `top_k`, `min_p`, `seed`).
///
/// Reported, not gated (`ran`): the engine's `decode_tok_s` of every run, from its request line (completion tokens
/// over the decode time), and their median beside `DECODE_3A_TOK_S` and `DECODE_M2_TOK_S`; `prefill_tok_s`
/// (prefilled tokens over the prefill time) and `restored` from each prompt's first seed only (the second seed restores
/// the first's prompt end). Beside each `decode_tok_s`, a `decode_steps_tok_s`: the steps' own rate (`decode_tokens`
/// over `decode_ms`), the one `speed-fresh` and the engine tool's `speed` take. A run that doesn't answer 200, has no
/// log line or no decode rate fails the case: the figures would be missing. So does a source that isn't on disk (the
/// `edit` prompt's, committed under `testdata/speed/sources`), unless `ctx.allow_missing_edit`: then the `edit`
/// prompt (and only it) is skipped and noted. Each answered run also carries the line's six `spec_*` counts and its
/// `completion_fnv` (see `speed_fresh`).
pub async fn speed(ctx: &Ctx) -> Result<Verdict> {
    let path = ctx.prompts_root.join("testdata/speed/prompts.json");
    let defs: Value = serde_json::from_slice(
        &std::fs::read(&path).with_context(|| format!("cannot read {}", path.display()))?,
    )
    .with_context(|| format!("{} is not JSON", path.display()))?;
    let s = &defs["sampling"];
    let sampling = sampling_of(s)?;
    let seeds = s["seeds"]
        .as_array()
        .context("prompts.json's sampling has no seeds")?;
    let prompts = defs["prompts"]
        .as_array()
        .context("prompts.json has no prompts")?;

    let mut checks = Vec::new();
    let mut figures = Map::new();
    let mut messages = Vec::new();
    let mut skipped = Vec::new();
    for p in prompts {
        let id = p["id"].as_str().context("a prompt has no id")?;
        match message(&ctx.prompts_root, &defs, p)? {
            Ok(m) => messages.push((id.to_string(), m)),
            Err(Unusable::Missing(f)) if ctx.allow_missing_edit && id == "edit" => {
                checks.push(Check::new(
                    format!("{id}_source"),
                    true,
                    format!("skipped: {} is missing (allow_missing_edit)", f.display()),
                ));
                skipped.push(json!({"prompt": id, "missing": f.display().to_string()}));
            }
            Err(u) => checks.push(unusable_check(id, &u)),
        }
    }
    figures.insert("skipped".into(), json!(skipped));
    if checks.iter().any(|c| !c.pass) {
        return Ok(Verdict::judged("speed", checks, figures));
    }

    let mut runs = Vec::new();
    let mut decode = Vec::new();
    let mut steps = Vec::new();
    let mut unanswered = Vec::new();
    let mut unlogged = Vec::new();
    for (i, seed) in seeds.iter().enumerate() {
        for (id, text) in &messages {
            let tag = format!("{id} seed {seed}");
            let seed_n = seed.as_u64().context("a seed is not a number")?;
            let r = one_run(ctx, text, seed_n, &sampling).await?;
            let mut run = Map::new();
            run.insert("prompt".into(), json!(id));
            run.insert("seed".into(), seed.clone());
            run.insert("status".into(), json!(r.status));
            if !r.answered() {
                unanswered.push(format!("{tag}: {}", r.status));
                runs.push(Value::Object(run));
                continue;
            }
            for k in ["prompt_tokens", "completion_tokens", "decode_tok_s"] {
                run.insert(k.into(), r.fig(k));
            }
            run.insert("decode_steps_tok_s".into(), json!(r.steps_tok_s()));
            for k in SPEC_KEYS {
                run.insert(k.into(), r.fig(k));
            }
            run.insert("completion_fnv".into(), r.completion_fnv());
            if i == 0 {
                for k in [
                    "prefill_tok_s",
                    "prefilled_tokens",
                    "reused_tokens",
                    "restored",
                ] {
                    run.insert(k.into(), r.fig(k));
                }
            }
            steps.extend(r.steps_tok_s());
            match r.line.as_ref().and_then(|l| l["decode_tok_s"].as_f64()) {
                Some(d) => decode.push(d),
                None => unlogged.push(tag),
            }
            runs.push(Value::Object(run));
        }
    }
    let n = seeds.len() * messages.len();
    checks.push(Check::new(
        "runs_answer",
        unanswered.is_empty(),
        if unanswered.is_empty() {
            format!("{n} runs answered 200")
        } else {
            format!("not 200: {}", unanswered.join(", "))
        },
    ));
    checks.push(decode_rates_check(&unlogged, decode.len()));

    let med = median(decode.clone());
    figures.insert("runs".into(), Value::Array(runs));
    figures.insert("decode_tok_s".into(), json!(decode));
    figures.insert("decode_steps_tok_s".into(), json!(steps));
    figures.insert("decode_median_tok_s".into(), json!(med));
    figures.insert("artemis_3a_decode_tok_s".into(), json!(DECODE_3A_TOK_S));
    figures.insert("decode_source".into(), json!("server log: completion_tokens / decode_ms of each request line; 3a and milestone 2 were timed by their own tools"));
    figures.insert("milestone2_decode_tok_s".into(), json!(DECODE_M2_TOK_S));
    figures.insert(
        "decode_median_vs_3a_pct".into(),
        json!(med.map(|m| 100.0 * (m / DECODE_3A_TOK_S - 1.0))),
    );
    Ok(if checks.iter().all(|c| c.pass) {
        Verdict::ran("speed", checks, figures)
    } else {
        Verdict::judged("speed", checks, figures)
    })
}

/// `speed-fresh`: the one request `ctx.speed_run` names, on a server process that has done nothing else (the runner
/// starts one per run), so its figures are a fresh engine's. The decode runs are `speed`'s requests, one at a time: a
/// prompt of `testdata/speed/prompts.json` with one of its seeds. `long` is golden prompt 20 without its ChatML wrapper
/// (`unwrap_chatml_user`), temperature 0 and one token asked for: a prefill, with no decode steps to time.
///
/// Reported, with nothing gated against a bound (`ran`) when every check passes. The checks: `answers` (200, and the
/// reply has the API's response shape); `logged` (the line has `decode_ms` and `decode_tokens` for a decode run, and
/// `prefill_ms` and `prefilled_tokens` for `long`); `fresh_engine` (the request restored nothing and reused no tokens:
/// a warm cache would make the figures another run's); and, for `long`, `prompt_unwrapped`. The figures are the
/// line's, with `decode_steps_tok_s` (`decode_tokens` over `decode_ms`, `null` when no step ran) and, for `long`,
/// `long_prompt_tokens_reference` for `prompt_tokens` to be read beside. They also hold the line's six `spec_*` counts
/// (`null` with speculation off) and `completion_fnv`, the FNV-1a 64 of the reply's text as 16 hex digits, so an off run
/// and an on run can be compared without the text being kept.
pub async fn speed_fresh(ctx: &Ctx) -> Result<Verdict> {
    let run = ctx
        .speed_run
        .as_ref()
        .context("speed-fresh needs ctx.speed_run (--speed-run)")?;
    let mut checks = Vec::new();
    let mut figures = Map::new();
    let (name, seed, text, sampling) = match run {
        SpeedRun::Decode { prompt, seed } => {
            let path = ctx.prompts_root.join("testdata/speed/prompts.json");
            let defs: Value = serde_json::from_slice(
                &std::fs::read(&path).with_context(|| format!("cannot read {}", path.display()))?,
            )
            .with_context(|| format!("{} is not JSON", path.display()))?;
            let p = defs["prompts"]
                .as_array()
                .and_then(|ps| ps.iter().find(|p| p["id"] == prompt.as_str()))
                .with_context(|| format!("prompts.json has no prompt {prompt}"))?;
            let text = match message(&ctx.prompts_root, &defs, p)? {
                Ok(m) => m,
                Err(u) => {
                    checks.push(unusable_check(prompt, &u));
                    return Ok(Verdict::judged("speed-fresh", checks, figures));
                }
            };
            (
                prompt.as_str(),
                *seed,
                text,
                sampling_of(&defs["sampling"])?,
            )
        }
        SpeedRun::Long => {
            let path = ctx.prompts_root.join(LONG_PROMPT);
            let raw = std::fs::read_to_string(&path)
                .with_context(|| format!("cannot read {}", path.display()))?;
            let Some(content) = unwrap_chatml_user(&raw) else {
                checks.push(Check::new(
                    "prompt_unwrapped",
                    false,
                    format!(
                        "{} does not start and end with the user turn's ChatML wrapper",
                        path.display()
                    ),
                ));
                return Ok(Verdict::judged("speed-fresh", checks, figures));
            };
            checks.push(Check::new(
                "prompt_unwrapped",
                true,
                format!("{} characters of user content", content.chars().count()),
            ));
            // Greedy: the body's own temperature 0, and its seed.
            let sampling = Sampling {
                fields: json!({}),
                max_tokens: 1,
            };
            ("long", 7, content.to_string(), sampling)
        }
    };

    let r = one_run(ctx, &text, seed, &sampling).await?;
    let long = matches!(run, SpeedRun::Long);
    figures.insert("prompt".into(), json!(name));
    figures.insert("seed".into(), json!(seed));
    for k in [
        "prompt_tokens",
        "completion_tokens",
        "decode_tokens",
        "decode_ms",
    ] {
        figures.insert(k.into(), r.fig(k));
    }
    figures.insert("decode_steps_tok_s".into(), json!(r.steps_tok_s()));
    for k in [
        "decode_tok_s",
        "prefill_tok_s",
        "prefilled_tokens",
        "decode_misses",
        "miss_mib_per_token",
    ] {
        figures.insert(k.into(), r.fig(k));
    }
    for k in SPEC_KEYS {
        figures.insert(k.into(), r.fig(k));
    }
    figures.insert("completion_fnv".into(), r.completion_fnv());
    if long {
        figures.insert(
            "long_prompt_tokens_reference".into(),
            json!(LONG_PROMPT_TOKENS),
        );
    }

    checks.push(Check::new(
        "answers",
        r.answered() && r.problems.is_empty(),
        if r.answered() && r.problems.is_empty() {
            "200".to_string()
        } else {
            // A non-200 reply's first problem already names the status.
            r.problems.join("; ")
        },
    ));
    let needed: [&str; 2] = if long {
        ["prefill_ms", "prefilled_tokens"]
    } else {
        ["decode_ms", "decode_tokens"]
    };
    let missing: Vec<&str> = needed.into_iter().filter(|k| r.fig(k).is_null()).collect();
    checks.push(Check::new(
        "logged",
        missing.is_empty(),
        match (&r.line, missing.is_empty()) {
            (_, true) => format!("the log line has {}", needed.join(" and ")),
            (None, false) => "the request has no log line".to_string(),
            (Some(_), false) => format!("the log line has no {}", missing.join(" or ")),
        },
    ));
    let restored = r.fig("restored");
    let reused = r.fig("reused_tokens");
    checks.push(Check::new(
        "fresh_engine",
        restored == "none" && reused == 0,
        format!("restored {restored}, reused_tokens {reused}"),
    ));
    Ok(if checks.iter().all(|c| c.pass) {
        Verdict::ran("speed-fresh", checks, figures)
    } else {
        Verdict::judged("speed-fresh", checks, figures)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_decode_rate_check_names_the_unlogged_runs() {
        let ok = decode_rates_check(&[], 6);
        assert!(ok.pass && ok.detail.starts_with("6 runs"));
        let bad = decode_rates_check(&["edit seed 1234".into(), "write seed 1235".into()], 4);
        assert!(!bad.pass, "{bad:?}");
        assert!(
            bad.detail.contains("edit seed 1234") && bad.detail.contains("write seed 1235"),
            "{}",
            bad.detail
        );
    }

    #[test]
    fn the_completion_hash_is_for_a_clean_200_only() {
        let body = |content: Value| {
            json!({"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "completion_tokens_details": {"reasoning_tokens": 0}}})
        };
        // A clean reply: the hash of its text (empty included).
        let ok = judge::openai_reply(200, &body(json!("hi")));
        assert!(ok.problems.is_empty(), "{:?}", ok.problems);
        assert_eq!(
            completion_hash(&ok),
            Some(format!("{:016x}", fnv1a64(b"hi")))
        );
        let empty = judge::openai_reply(200, &body(json!("")));
        assert_eq!(completion_hash(&empty), Some("cbf29ce484222325".into()));
        // A 200 whose content is not a string has a shape problem and no hash: it never equals an empty completion.
        let null = judge::openai_reply(200, &body(Value::Null));
        assert!(!null.problems.is_empty());
        assert_eq!(completion_hash(&null), None);
        // Not 200: no hash.
        assert_eq!(completion_hash(&judge::openai_reply(503, &json!({}))), None);
    }

    #[test]
    fn median_is_pythons() {
        assert_eq!(median(vec![]), None);
        assert_eq!(median(vec![3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(vec![4.0, 1.0, 2.0, 3.0]), Some(2.5));
    }

    /// The edit prompt's message is its text, the first `chars` characters of the source, and its closing fence; a
    /// source with multibyte characters is cut by character, as Python slices.
    #[test]
    fn a_source_part_is_cut_by_characters() {
        let root = std::env::temp_dir().join(format!("artemis-speed-msg-{}", std::process::id()));
        std::fs::create_dir_all(root.join("testdata/s")).unwrap();
        std::fs::write(root.join("testdata/s/src.txt"), "aé\r\nbcdef").unwrap();
        let defs = json!({"sources": [{"id": "x", "path": "s/src.txt"}]});
        let p =
            json!({"content": [{"text": "T\n"}, {"source": "x", "chars": 5}, {"text": "\n```"}]});
        let Ok(m) = message(&root, &defs, &p).unwrap() else {
            panic!("unusable")
        };
        assert_eq!(m, "T\naé\r\nb\n```");
        let long = json!({"content": [{"source": "x", "chars": 50}]});
        assert!(matches!(
            message(&root, &defs, &long).unwrap(),
            Err(Unusable::Short(_, 9, 50))
        ));
        let gone = json!({"sources": [{"id": "x", "path": "s/none.txt"}]});
        assert!(matches!(
            message(&root, &gone, &p).unwrap(),
            Err(Unusable::Missing(f)) if f.ends_with("none.txt")
        ));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn unwrap_chatml_takes_the_user_content() {
        let wrapped = format!(
            "{USER_PREFIX}Here is a file.
<|im_start|>x
{USER_SUFFIX}"
        );
        assert_eq!(
            unwrap_chatml_user(&wrapped),
            Some(
                "Here is a file.
<|im_start|>x
"
            )
        );
        assert_eq!(
            unwrap_chatml_user(&format!("{USER_PREFIX}{USER_SUFFIX}")),
            Some("")
        );
        // `None` when either end is missing, or when the ends overlap.
        assert_eq!(unwrap_chatml_user(&format!("{USER_PREFIX}body")), None);
        assert_eq!(unwrap_chatml_user(&format!("body{USER_SUFFIX}")), None);
        assert_eq!(unwrap_chatml_user("body"), None);
        assert_eq!(unwrap_chatml_user(""), None);
        assert_eq!(
            unwrap_chatml_user(&format!(
                "{USER_PREFIX}<|im_end|>
"
            )),
            None
        );
    }

    #[test]
    fn the_decode_steps_rate_is_the_steps_over_their_time() {
        let line = json!({"decode_tokens": 699, "decode_ms": 7000.0});
        let rate = decode_steps_tok_s(Some(&line)).unwrap();
        assert!((rate - 99.857142857).abs() < 1e-6, "{rate}");
        // No steps ran (a one-token completion): `None`, not 0.
        let one = json!({"decode_tokens": 0, "decode_ms": 3.0});
        assert_eq!(decode_steps_tok_s(Some(&one)), None);
        for bad in [
            json!({"decode_tokens": 5, "decode_ms": 0.0}),
            json!({"decode_tokens": null, "decode_ms": 3.0}),
            json!({"decode_tokens": 5}),
        ] {
            assert_eq!(decode_steps_tok_s(Some(&bad)), None, "{bad}");
        }
        assert_eq!(decode_steps_tok_s(None), None);
    }
}
