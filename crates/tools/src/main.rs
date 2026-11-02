//! `artemis`: the Qwen3.6-35B-A3B engine's command line: the CPU reference and test tools, the GPU
//! engine's dump and its tap rule, and `generate`, which runs prompts through prefill.
use anyhow::{bail, Context, Result};
use artemis_tools::{
    compare, generate, hits, kl, mtp_check, profile, snaptimes, spec_check, speed, tapdir, test2,
    test3, test5, test6, verify_cost,
};
use clap::{Parser, Subcommand};
use engine::{Engine, Live, Loaded, Marks, Options, Sampler, SnapId, Snapshots};
use gguf::Gguf;
use qwen36::{
    dims::Dims,
    model::Model,
    reference::{Reference, Rounding},
    stream::TextStream,
    taps::{self, Tap},
    testing,
    tokenizer::{Tokenizer, Vocab},
};
use std::{collections::BTreeMap, fs, io::Write, path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(
    name = "artemis",
    about = "Artemis: a Qwen3.6-35B-A3B engine on Vulkan"
)]
struct Cli {
    /// Without a command, `artemis` says how to start the server.
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

/// `--cache-policy`'s values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum PolicyArg {
    Adaptive,
    Static,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a token file through the CPU reference (--cpu) or the GPU engine (--gpu) and write every tap.
    Dump {
        #[arg(long)]
        model: PathBuf,
        /// Little-endian u32 token ids.
        #[arg(long)]
        tokens: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Use the CPU reference.
        #[arg(long)]
        cpu: bool,
        /// Use the GPU engine.
        #[arg(long)]
        gpu: bool,
        /// The reference's threads.
        #[arg(long, default_value_t = 16)]
        threads: usize,
        /// Round as llama.cpp's CPU backend does (test 2's candidate) instead of exactly. Reference only.
        #[arg(long)]
        llama_rounding: bool,
        /// Keep the reference's KV cache in f16, as the GPU engine does: the baseline `gpu-taps` compares against.
        /// Reference only.
        #[arg(long)]
        f16_kv: bool,
        /// Teacher-force the run with this tap directory's values.
        #[arg(long)]
        force_from: Option<PathBuf>,
        /// The GPU's UUID, when there is more than one discrete GPU.
        #[arg(long)]
        device: Option<String>,
        /// Run the MTP block (`blk.40`) instead of the main model, over rows `q = 1 .. n − 1`, each pairing token `q`
        /// with llama.cpp's hidden row at `q − 1` (--hidden). --force-from then names an `artemis-oracle mtp-taps`
        /// directory. CPU reference only.
        #[arg(long, requires = "hidden")]
        mtp: bool,
        /// `artemis-oracle mtp-taps`'s `hidden.bin`: the main model's final hidden row at each token's position.
        #[arg(long, requires = "mtp")]
        hidden: Option<PathBuf>,
    },
    /// Print per-tap differences of A against B (either directory format).
    Compare {
        a: PathBuf,
        b: PathBuf,
        /// Report the first tap whose worst relative error exceeds this.
        #[arg(long, default_value_t = 1e-3)]
        threshold: f64,
        /// The dumps are of the tiny test model.
        #[arg(long)]
        tiny: bool,
    },
    /// Apply test 2's pass rule and write the verdict as JSON.
    Test2 {
        #[arg(long)]
        llama_cpu: PathBuf,
        #[arg(long)]
        llama_vulkan: Option<PathBuf>,
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        tiny: bool,
        /// The candidate was teacher-forced from --llama-cpu: apply the gating per-operation rule. Without it, the
        /// free-running floor rule is applied, which is reported but doesn't gate.
        #[arg(long)]
        forced: bool,
    },
    /// Apply the GPU engine's rule: the engine teacher-forced from the reference's taps, tap by tap. The
    /// reference must come from `dump --cpu --f16-kv`. Writes the verdict as JSON and exits 1 if it fails.
    GpuTaps {
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        tiny: bool,
    },
    /// Generate text, greedy or sampled. The prompt runs through prefill, then each token
    /// through decode, or, with speculation (on by default), through verify steps over drafts from the run's own
    /// history. --json records every step and the logits fingerprints, for tests 3 and 5.
    Generate {
        #[arg(long)]
        model: PathBuf,
        /// Raw prompt text; special tokens are parsed.
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        /// Little-endian u32 token ids.
        #[arg(long)]
        tokens_file: Option<PathBuf>,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        /// Take the top token every step (ties to the lower id) instead of sampling.
        #[arg(long)]
        greedy: bool,
        /// Sampling temperature (default 0.6).
        #[arg(long)]
        temp: Option<f32>,
        /// Keep the k most likely tokens; 0 keeps them all (default 20).
        #[arg(long)]
        top_k: Option<usize>,
        /// Keep the most likely tokens until their probabilities reach this (default 0.95).
        #[arg(long)]
        top_p: Option<f32>,
        /// Drop tokens less likely than this share of the top token's probability (default 0).
        #[arg(long)]
        min_p: Option<f32>,
        /// Subtracted from the logit of every token this run has already generated (default 0).
        #[arg(long, allow_negative_numbers = true)]
        presence_penalty: Option<f32>,
        /// The sampler's seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 65_536)]
        ctx: usize,
        /// Expert slots per layer in VRAM, clamped to what fits. By default, as many as the budget allows. N is also
        /// the ceiling for lent and grown rows; without it, the cache may grow toward every expert as the budget allows.
        #[arg(long)]
        cache_slots: Option<usize>,
        /// Start one seeded pseudo-random expert swap every step.
        #[arg(long)]
        force_swap_every_token: bool,
        /// Feed these tokens (little-endian u32), one per step, instead of choosing, and record the engine's own top
        /// two beside each. The run takes exactly these tokens.
        #[arg(long, requires = "greedy")]
        force_tokens: Option<PathBuf>,
        /// Write the prompt tokens, every step with its top two logits, the logits fingerprints, the timings and the
        /// cache statistics here, as JSON.
        #[arg(long)]
        json: Option<PathBuf>,
        /// Compute every prompt position's logits in prefill, and fingerprint each in --json. By default only
        /// the last prompt position's are computed.
        #[arg(long, requires = "json")]
        all_logits: bool,
        /// Leave the logits' fingerprints out of --json (`positions` empty): hashing them costs about a millisecond a
        /// token inside the timed loop, so the speed gate's runs skip it.
        #[arg(long, requires = "json", conflicts_with = "all_logits")]
        no_fingerprints: bool,
        /// Time every step on the GPU and CPU and write the report here as JSON. Timestamps
        /// between commands can slow the run, so a profiled run is not a timed one.
        #[arg(long)]
        profile: Option<PathBuf>,
        /// Prefill on today's f32 kernels even where the split-f16 cooperative-matrix kernels exist.
        /// For comparisons; the speed gate refuses such runs.
        #[arg(long)]
        f32_prefill: bool,
        /// Prefill without staging buffers, as before staging existed: the cache keeps their slots. For
        /// comparisons; the speed gate refuses such runs.
        #[arg(long)]
        no_staging: bool,
        /// Stage prefill chunks of at least N tokens instead of the measured threshold (1 stages every chunk).
        #[arg(long, conflicts_with = "no_staging")]
        stage_from: Option<usize>,
        /// The expert cache's policy: adaptive swaps experts as the routes move; static keeps
        /// the profile's cache, as it did before the adaptive policy, and also refills rows that come back empty (lent
        /// or grown) in the profile's order, never swapping. --force-swap-every-token always runs static.
        #[arg(long, value_enum, default_value_t = PolicyArg::Adaptive)]
        cache_policy: PolicyArg,
        /// MiB of expert copies per token the adaptive policy may make (default the engine's UPLOAD_CAP, 8); 1 to 2048,
        /// as the engine refuses a cap outside 1 MiB to 2 GiB. For tuning; the speed gate refuses such runs.
        #[arg(long, value_parser = clap::value_parser!(u64).range(1..=2048))]
        upload_cap: Option<u64>,
        /// Don't lend the staging buffers' VRAM to the cache between staged prefill chunks. For
        /// comparisons; the speed gate refuses such runs.
        #[arg(long)]
        no_lend: bool,
        /// Decode reads a resident expert in place of every HOST one: the per-token time with
        /// every routed expert read from VRAM, plus the table's per-step rebuild and copy (decode's base time). The
        /// output is meaningless; the speed and hit-rate gates refuse such runs.
        #[arg(long)]
        base_time: bool,
        /// The share of HOST misses the CPU serves, 0.0 to 1.0: that fraction of the 256 experts, the same
        /// ones in every layer, is computed on the CPU and the rest read over PCIe. 0 (the default) leaves the HOST
        /// path as it was. The engine refuses a value outside 0.0 to 1.0, a share with --base-time, and a machine
        /// without what the CPU path needs, before it loads the model; the speed and hit-rate gates refuse such runs.
        #[arg(long, default_value_t = 0.0)]
        cpu_share: f32,
        #[command(flatten)]
        spec: generate::SpecFlags,
        /// The GPU's UUID, when there is more than one discrete GPU.
        #[arg(long)]
        device: Option<String>,
    },
    /// Apply test 3's rule to golden prompts: our runs, fed llama.cpp's greedy tokens, may flip at most 1.5
    /// times as often as llama.cpp CPU's along the same tokens, pooled over the prompts. Writes the verdict as JSON and
    /// exits 1 if it fails.
    Test3 {
        /// The golden set: prompts/<id>.tokens, expected/<id>.json and forced-cpu/<id>.json.
        #[arg(long)]
        golden: PathBuf,
        /// ID=FILE: a golden prompt's id and our run on it, `artemis generate --greedy --max-tokens 256
        /// --force-tokens <its oracle's tokens> --json FILE`.
        #[arg(long = "run", required = true)]
        runs: Vec<String>,
        #[arg(long)]
        out: PathBuf,
    },
    /// One warm path of test 6, in a fresh process: build a snapshot along path 1 (prefill the
    /// first half of the prompt) or path 2 (prefill it, decode half the oracle's tokens), dirty the engine with
    /// another golden prompt, restore, and decode the oracle's tokens forced. --json is a `generate` record that
    /// `test3` accepts, plus the `warm` facts. The engine opens as `generate`'s default does.
    Warm {
        #[arg(long)]
        model: PathBuf,
        /// The golden set: prompts/<id>.tokens (and the dirtying prompt's) and expected/<id>.json.
        #[arg(long)]
        golden: PathBuf,
        /// The golden prompt to run.
        #[arg(long)]
        id: String,
        /// 1: the snapshot is built by prefill at the cut; 2: by decode, after half the oracle's tokens.
        #[arg(long, value_parser = clap::value_parser!(u8).range(1..=2))]
        path: u8,
        #[arg(long)]
        json: PathBuf,
        /// Fingerprint every prompt position (the bit-for-bit gate needs them), as `generate --all-logits` does.
        #[arg(long)]
        all_logits: bool,
        #[arg(long, default_value_t = 65_536)]
        ctx: usize,
        /// Expert slots per layer in VRAM, as `generate --cache-slots`.
        #[arg(long)]
        cache_slots: Option<usize>,
        /// The snapshot pool's size in GiB, cut to what the host has free.
        #[arg(long, default_value_t = 8)]
        snapshot_pool_gib: u32,
        /// Speculation, as `generate`'s flags (on by default), so the warm run decodes as its cold run does.
        #[command(flatten)]
        spec: generate::SpecFlags,
        /// The GPU's UUID, when there is more than one discrete GPU.
        #[arg(long)]
        device: Option<String>,
    },
    /// Apply test 6's rule to the golden prompts' cold runs and warm paths: each path's flips,
    /// pooled, at most 48; the long prompt's prefilled positions after a path-1 restore bit for bit the cold run's.
    /// Writes the verdict as JSON and exits 1 if it fails.
    Test6 {
        /// The golden set: prompts/<id>.tokens, expected/<id>.json and forced-cpu/<id>.json.
        #[arg(long)]
        golden: PathBuf,
        /// ID=FILE: a cold run, `artemis generate --tokens-file <its tokens> --greedy --max-tokens 256 --force-tokens
        /// <its oracle's tokens> --all-logits --json FILE`.
        #[arg(long = "cold", required = true)]
        cold: Vec<String>,
        /// ID=FILE: `artemis warm --path 1 --all-logits --json FILE`.
        #[arg(long = "path1")]
        path1: Vec<String>,
        /// ID=FILE: `artemis warm --path 2 --all-logits --json FILE`.
        #[arg(long = "path2")]
        path2: Vec<String>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Measure the snapshot copies on the long golden prompt: the PCIe link's rates for both
    /// pool memory kinds, then the DeltaNet state and the KV to the host and back (whole, and as the server's snapshot
    /// points), a restore after a dirtying, and the prefill's tok/s with and without the snapshots. Report only: it
    /// fails only when it can't run.
    SnapshotTimes {
        #[arg(long)]
        model: PathBuf,
        /// The golden set: prompts/20-long-option-rs.tokens and prompts/01-code.tokens.
        #[arg(long)]
        golden: PathBuf,
        #[arg(long)]
        json: PathBuf,
        /// The snapshot pool's size in GiB, cut to what the host has free.
        #[arg(long, default_value_t = 8)]
        pool_gib: u32,
        /// The GPU's UUID, when there is more than one discrete GPU.
        #[arg(long)]
        device: Option<String>,
    },
    /// Apply the hit-rate gate: the golden set with 128 greedy tokens, the adaptive policy
    /// against the static one, by group. Writes the verdict as JSON and exits 1 if it fails.
    Hits {
        /// The golden set: the runs' prompts must be its prompts/<id>.tokens.
        #[arg(long)]
        golden: PathBuf,
        /// ID=POLICY,STATIC: a golden prompt's id, then its two `generate --json` records.
        #[arg(long = "run", required = true)]
        runs: Vec<String>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Apply test 5's rule: runs of one prompt at several cache sizes, and with forced swaps, must give the
    /// same tokens and bit-identical logits. Writes the verdict as JSON and exits 1 if it fails.
    Test5 {
        /// LABEL=FILE: a `generate --json` record and its name in the verdict. The first run is the baseline.
        #[arg(long = "run", required = true)]
        runs: Vec<String>,
        /// LABEL=FILE: a record compared with the first run and listed in the verdict's `reports`, never in its status
        /// (the run without speculation, which is expected to differ in its DeltaNet layers).
        #[arg(long = "report-run")]
        report_runs: Vec<String>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Test 4's measurement: score the engine's logits against a llama-perplexity KL base, chunk by chunk
    /// from a fresh state, and write the mean KL divergence as JSON. Each chunk runs through prefill, with the scored
    /// positions' logits.
    Kl {
        #[arg(long)]
        model: PathBuf,
        /// A `llama-perplexity --kl-divergence-base` file.
        #[arg(long)]
        base: PathBuf,
        /// The text the base was made from: checks that our tokenizer gives the base's tokens.
        #[arg(long)]
        corpus: Option<PathBuf>,
        /// The base's context; refused if it differs.
        #[arg(short = 'c', long = "ctx", default_value_t = 4096)]
        ctx: usize,
        /// Score only the first N chunks (default: all of the base's).
        #[arg(long)]
        chunks: Option<usize>,
        /// Expert slots per layer; default: the most the budget allows. N is also the ceiling for lent and grown rows;
        /// without it, the cache may grow toward every expert as the budget allows.
        #[arg(long)]
        cache_slots: Option<usize>,
        /// The GPU's UUID, when there is more than one discrete GPU.
        #[arg(long)]
        device: Option<String>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Apply test 4's rule to `artemis kl`'s report and llama.cpp's Vulkan-vs-CPU KL, read from
    /// llama-perplexity's output. Writes the verdict as JSON and exits 1 if it fails.
    Test4 {
        /// llama-perplexity's output from the --kl-divergence run on Vulkan.
        #[arg(long)]
        floor_log: PathBuf,
        /// `artemis kl --out`'s report.
        #[arg(long)]
        kl: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// The speed gate over `generate --json` records: the median decode speed of the
    /// write, edit and explain prompts at seeds 1234 and 1235, and the long golden prompt's prefill in the slower of
    /// two runs. Writes the verdict as JSON. Exits 1 when a record isn't one of the gate's runs, and, with --gate,
    /// when a speed is below its bound.
    Speed {
        /// PROMPT=FILE: a decode run of speed prompt PROMPT (write, edit or explain).
        #[arg(long = "decode", required = true)]
        decode: Vec<String>,
        /// A prefill run of the long golden prompt (`--greedy --max-tokens 1`).
        #[arg(long = "prefill", required = true)]
        prefill: Vec<PathBuf>,
        /// testdata/speed/made-from.json.
        #[arg(long)]
        made_from: PathBuf,
        /// The long golden prompt's ids: testdata/golden/prompts/20-long-option-rs.tokens.
        #[arg(long)]
        long_tokens: PathBuf,
        /// Fail when a speed is below its bound, not only when a record is wrong.
        #[arg(long)]
        gate: bool,
        /// A base-time run (`generate --base-time`) of the write prompt, reported beside the verdict, not gated.
        #[arg(long)]
        base: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Print the --filter regex for `artemis-oracle taps` (or, with --mtp, `mtp-taps`).
    TapsFilter {
        /// The MTP block's names instead of the main model's.
        #[arg(long)]
        mtp: bool,
    },
    /// Test 2's MTP half: the reference's MTP taps, teacher-forced from `artemis-oracle mtp-taps`,
    /// against it per operation. Writes the verdict as JSON and exits 1 unless it passes.
    Test2Mtp {
        /// `artemis-oracle mtp-taps --out`.
        #[arg(long)]
        oracle: PathBuf,
        /// `artemis dump --cpu --mtp --force-from <oracle>`.
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// The dumps are of the tiny test model.
        #[arg(long)]
        tiny: bool,
    },
    /// The GPU's MTP draft against llama.cpp's on the real model: decodes the oracle's tokens one at a time
    /// and compares the MTP's logits at each position with `artemis-oracle mtp-taps`'s `result_output`. Pass: the top-1
    /// ids are equal wherever the oracle's top two are more than the test 3 tie margin apart; the relative L2 is
    /// reported. Writes the verdict as JSON and exits 1 unless it passes. GPU.
    MtpCheck {
        #[arg(long)]
        model: PathBuf,
        /// `artemis-oracle mtp-taps --out`.
        #[arg(long)]
        oracle: PathBuf,
        /// The token file given to `artemis-oracle mtp-taps --tokens` (the oracle's directory doesn't keep it).
        #[arg(long)]
        tokens: PathBuf,
        #[arg(long)]
        json: PathBuf,
        /// The session's positions: at least the tokens' count and 512.
        #[arg(long, default_value_t = mtp_check::CTX)]
        ctx: usize,
        /// The GPU's UUID, when there is more than one discrete GPU.
        #[arg(long)]
        device: Option<String>,
    },
    /// Write the tiny synthetic test model.
    TinyModel {
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Write the compact vocabulary extract committed for CI.
    ExtractVocab {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// What a t-token pass costs on the prefill path: at each --at context, --reps timed decode
    /// steps, then --reps timed prefill passes over the prompt's next t tokens for every t of 2 or more in --t. Writes
    /// the per-(path, context, t) means and minimums as JSON.
    VerifyCost {
        #[arg(long)]
        model: PathBuf,
        /// Little-endian u32 token ids; at least the last --at plus the passes' positions.
        #[arg(long)]
        tokens_file: PathBuf,
        /// Contexts to measure at, ascending; the prompt is prefilled up to each.
        #[arg(long, value_delimiter = ',', default_values_t = [64, 2048, 8192, 20_000])]
        at: Vec<usize>,
        /// Pass lengths. 1 is the decode step, always timed; each other value is a prefill pass.
        #[arg(long, value_delimiter = ',', default_values_t = [1, 2, 3, 4, 5])]
        t: Vec<usize>,
        /// Timed rounds per context: one decode step and one pass of each length in --t.
        #[arg(long, default_value_t = 10)]
        reps: usize,
        /// Untimed decode steps after each fill, so the expert cache has refilled the rows staging lent before the
        /// timed rounds (default 96: about 70 tokens of the policy's copy cap, with margin).
        #[arg(long, default_value_t = verify_cost::WARMUP)]
        warmup: usize,
        /// The session's positions, which size the expert cache: production's 65,536 by default, so the cache holds
        /// what the server's does. At least the plan's last --at plus the positions its passes use.
        #[arg(long, default_value_t = 65_536)]
        ctx: usize,
        /// Also time the engine's verify step for each value of --t (1 to 16): `Session::verify` over the next t tokens,
        /// every row read, then `commit(1)` (rows with path "verify") and, for t of 2 or more, `commit(t)` (path
        /// "verify_all"), so the replay's growth with the tokens kept shows. The engine opens with speculation on.
        #[arg(long)]
        verify: bool,
        /// Also time the MTP's draft pass (`Session::mtp_draft`) for each value of --t from 1 to 15 (rows with path
        /// "mtp_draft", t = k), each after one untimed decode step. The engine opens with the MTP and speculation on.
        #[arg(long)]
        mtp: bool,
        /// Written whole at the end, its directory made first; a failed run leaves no file.
        #[arg(long)]
        json: PathBuf,
        /// The GPU's UUID, when there is more than one discrete GPU.
        #[arg(long)]
        device: Option<String>,
    },
    /// Run the HTTP server over the engine: the OpenAI and Anthropic APIs on one port.
    Serve(Box<artemis_tools::serve::run::ServeCmd>),
    /// Measure this PC and write the settings the server then uses (a per-user tuned file). `artemis serve` does this
    /// on its first start; run it again after a driver or hardware change with --force.
    Setup(Box<artemis_tools::setup::SetupCmd>),
    /// Run test 7's real-engine cases against a running server on loopback (never port 8080) and write one verdict
    /// file per case.
    Test7(Box<artemis_tools::test7::Test7Args>),
    /// Report how much of the prompts the cache gave back, from request logs: overall, by where the restore came from
    /// and by why it stopped short, and the think-budget insertions by budget.
    ReuseReport(Box<artemis_tools::test7::report::ReuseReportArgs>),
    /// Replay recorded token streams through the prompt-lookup drafter and the acceptance rule, and predict the accept
    /// rate and the decode speed with the cost model. CPU only.
    LookupSim(artemis_tools::lookup_sim::LookupSimArgs),
    /// Speculation on against off in one process, one model load: `golden` judges the
    /// golden prompts' greedy runs by test 3's near-tie rule, `speed` times the speed prompts. GPU.
    SpecCheck {
        #[command(subcommand)]
        cmd: spec_check::Cmd,
    },
}

fn dims(tiny: bool) -> Dims {
    if tiny {
        Dims::tiny()
    } else {
        Dims::QWEN36
    }
}

use generate::read_tokens;

fn read_json<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<T> {
    let b = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&b).with_context(|| format!("parsing {}", path.display()))
}

fn device_uuid(s: Option<&str>) -> Result<Option<[u8; 16]>> {
    s.map(|s| engine::parse_uuid(s).with_context(|| format!("--device {s:?} is not a UUID")))
        .transpose()
}

/// Prints a verdict's one-line summary.
fn summary(what: &str, v: &test2::Verdict) {
    let w = test2::worst(v).map_or(String::new(), |r| {
        format!(
            "; worst {} at {:.3e} of a {:.3e} bound",
            r.tap, r.err, r.bound
        )
    });
    println!("{what}: {:?}{w}", v.status);
}

/// The golden prompts' ids, sorted: the names of `prompts/*.tokens`.
fn golden_ids(golden: &std::path::Path) -> Result<Vec<String>> {
    let dir = golden.join("prompts");
    let mut ids = Vec::new();
    for entry in fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|x| x == "tokens") {
            if let Some(id) = path.file_stem().and_then(|s| s.to_str()) {
                ids.push(id.to_string());
            }
        }
    }
    ids.sort();
    Ok(ids)
}

/// Test 6's `WarmOps` on the real engine: a `Loaded` (engine, session and snapshots, in its drop order), the
/// snapshot `drive` takes and the position it covers.
struct RealWarm {
    live: Loaded,
    /// What the snapshot is marked: the prompt's end for path 1, a reply's end for path 2.
    marks: Marks,
    snap: Option<(SnapId, usize)>,
}

impl test6::WarmOps for RealWarm {
    fn room(&self) -> usize {
        self.live.ctx() - self.live.pos()
    }

    fn prefill(
        &mut self,
        tokens: &[u32],
        all: bool,
        each: &mut dyn FnMut(&[f32]),
    ) -> Result<Vec<f32>> {
        let (e, s, _) = self.live.parts();
        if all {
            let mut last = Vec::new();
            s.prefill_each(e, tokens, 0, |i, l| {
                each(l);
                if i + 1 == tokens.len() {
                    last = l.to_vec();
                }
                Ok::<(), engine::Error>(())
            })?;
            Ok(last)
        } else {
            let l = s.prefill(e, tokens)?;
            each(&l);
            Ok(l)
        }
    }

    fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        let (e, s, _) = self.live.parts();
        Ok(s.decode(e, token)?)
    }

    fn verify(&mut self, tokens: &[u32]) -> Result<()> {
        Ok(Live::verify(&mut self.live, tokens)?)
    }

    fn row(&mut self, i: usize) -> Result<Vec<f32>> {
        Ok(Live::row(&mut self.live, i)?)
    }

    fn commit(&mut self, kept: usize) -> Result<()> {
        Ok(Live::commit(&mut self.live, kept)?)
    }

    fn n_vocab(&self) -> usize {
        Live::n_vocab(&self.live)
    }

    fn mtp_draft(&mut self, x: u32, k: usize) -> Result<Vec<u32>> {
        Ok(Live::mtp_draft(&mut self.live, x, k)?)
    }

    fn has_mtp(&self) -> bool {
        Live::has_mtp(&self.live)
    }

    fn misses(&self) -> u64 {
        Live::cache_stats(&self.live).misses
    }

    fn miss_bytes(&self) -> u64 {
        Live::expert_bytes(&self.live)
    }

    fn snapshot(&mut self) -> Result<u64> {
        let (e, s, snaps) = self.live.parts();
        let at = s.tokens().len();
        let Some(taken) = snaps.take(e, s, self.marks)? else {
            bail!(
                "the snapshot at position {at} was skipped (the pool is too small or out of room); test 6 needs it"
            );
        };
        self.snap = Some((taken.id, at));
        Ok(taken.bytes)
    }

    fn dirty(&mut self, tokens: &[u32]) -> Result<()> {
        let at = self.snap.context("dirty before a snapshot")?.1;
        let (e, s, _) = self.live.parts();
        s.reset(e)?;
        s.prefill(e, tokens)?;
        // The dirtier is short: the KV from its end up to the snapshot's position still holds the first run's bytes, so
        // a restore that skipped its copy would pass by luck. NaN words there leave nothing to pass on.
        let from = s.kv_tokens().len();
        if from < at {
            s.poison_kv(e, from, at)?;
        }
        Ok(())
    }

    fn restore(&mut self) -> Result<test6::RestoreFacts> {
        let id = self.snap.context("restore before a snapshot")?.0;
        let (e, s, snaps) = self.live.parts();
        let r = snaps.restore(e, s, id)?;
        Ok(test6::RestoreFacts {
            bytes: r.bytes,
            kv_from: r.kv_from,
            ms: r.time.as_secs_f64() * 1e3,
        })
    }
}

/// How a bit comparison reads in a prompt's line.
fn bits_word(b: Option<bool>) -> &'static str {
    match b {
        Some(true) => "equal",
        Some(false) => "DIFFER",
        None => "not compared",
    }
}

/// `artemis warm`'s arguments.
struct WarmArgs {
    model: PathBuf,
    golden: PathBuf,
    id: String,
    path: u8,
    json: PathBuf,
    all_logits: bool,
    ctx: usize,
    cache_slots: Option<usize>,
    snapshot_pool_gib: u32,
    /// Speculation's flags, as `generate` takes them.
    spec: generate::SpecFlags,
    device: Option<String>,
}

/// `artemis warm`: one warm path of test 6. Its own function, not an arm of `main`: `main`'s frame is already large.
fn run_warm(args: WarmArgs) -> Result<()> {
    let WarmArgs {
        model,
        golden,
        id,
        path,
        json,
        all_logits,
        ctx,
        cache_slots,
        snapshot_pool_gib,
        spec,
        device,
    } = args;
    // Checked before anything is read or loaded; warm has no CPU share and no base time.
    let spec = spec.config(0.0, false)?;
    let file = |dir: &str, name: &str, ext: &str| golden.join(dir).join(format!("{name}.{ext}"));
    let ids = golden_ids(&golden)?;
    if !ids.contains(&id) {
        bail!("--id {id} is not a prompt of {}", golden.display());
    }
    let dirty_id = test6::dirtier(&ids, &id)
        .context("test 6 dirties the engine with another golden prompt; there is none")?
        .to_string();
    let prompt = read_tokens(&file("prompts", &id, "tokens"))?;
    let dirty = read_tokens(&file("prompts", &dirty_id, "tokens"))?;
    let oracle: test3::Oracle = read_json(&file("expected", &id, "json"))?;
    let forced: Vec<u32> = oracle.steps.iter().map(|s| s.token).collect();
    // The record is written at the end; an unwritable path fails now.
    let mut out = fs::File::create(&json)
        .with_context(|| format!("--json: can't write {}", json.display()))?;
    let g = Gguf::open(&model).with_context(|| format!("opening {}", model.display()))?;
    let vocab = Vocab::from_gguf(&g).map_err(anyhow::Error::msg)?;
    let eog = vocab.eog();
    // Checked before the model loads.
    if prompt.len().max(dirty.len()) > ctx {
        bail!(
            "the prompts are {} and {} tokens, longer than --ctx {ctx}",
            prompt.len(),
            dirty.len()
        );
    }
    generate::check_forced(&forced, test3::STEPS, &eog)?;
    generate::check_forced_vocab(&forced, Dims::QWEN36.n_vocab)?;
    let (how, marks, cut, split) = if path == 1 {
        let cut = test6::cut_for(&id, prompt.len());
        (
            test6::Path::Prefill { cut },
            Marks::PROMPT_END,
            Some(cut),
            None,
        )
    } else {
        let split = test6::split_for(forced.len());
        (
            test6::Path::Decode { split },
            Marks::REPLY_END,
            None,
            Some(split),
        )
    };
    let t_load = Instant::now();
    // generate's defaults: the same staging, policy, kernels and share as the cold run.
    let e = Engine::open(
        &g,
        &Dims::QWEN36,
        &Options {
            device_uuid: device_uuid(device.as_deref())?,
            ctx,
            cache_slots,
            speculation: spec.is_some(),
            mtp: generate::mtp_options_for_flags(spec.as_ref(), ctx)?,
            ..Options::default()
        },
    )?;
    let s = e.session(ctx)?;
    let limit = engine::limit_at_load(&e, snapshot_pool_gib, 0)?;
    if limit.bytes == 0 {
        bail!(
            "the snapshot pool is 0 bytes (asked {} MiB{}): test 6 needs snapshots",
            limit.requested >> 20,
            if limit.cut {
                ", cut to what the host has free"
            } else {
                ""
            }
        );
    }
    let snaps = Snapshots::new(&e, limit);
    let load_s = t_load.elapsed().as_secs_f64();
    eprintln!(
        "loaded on {} in {load_s:.1} s; {id} on path {path} against {dirty_id}; prompt {} tokens, {} forced; \
         {} expert slots per layer; snapshot pool {} MiB{}",
        e.device_name(),
        prompt.len(),
        forced.len(),
        e.cache_slots(),
        limit.bytes >> 20,
        if limit.cut { " (cut to what the host has free)" } else { "" },
    );
    let (at_open, loaded) = (e.vram_at_open(), e.vram_heap()?);
    let memory = generate::Memory::of(&e);
    let mut ops = RealWarm {
        live: Loaded::new(e, s, snaps),
        marks,
        snap: None,
    };
    let mut state = spec.clone().map(engine::spec::SpecState::new);
    let t0 = Instant::now();
    let r = test6::drive(
        &mut ops,
        &prompt,
        &dirty,
        how,
        &forced,
        &eog,
        all_logits,
        state.as_mut(),
    )?;
    // One time for the whole run: the prompt, the snapshot and restore, and the decodes are not split.
    let generate_s = t0.elapsed().as_secs_f64();
    let decoded = r.positions.len() - if all_logits { prompt.len() } else { 1 };
    eprintln!(
        "snapshot {} MiB; restore copied {} MiB from KV position {} in {:.1} ms; {} steps, stopped at {}; {generate_s:.1} s",
        r.snapshot_bytes >> 20,
        r.restore.bytes >> 20,
        r.restore.kv_from,
        r.restore.ms,
        r.steps.len(),
        r.stop.describe(),
    );
    let (e, _, _) = ops.live.parts();
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    let record = generate::record(
        e,
        &generate::Asked {
            greedy: true,
            sampling: generate::sampling(true, None, None, None, None, None)?,
            seed: 0,
            max_tokens: test3::STEPS,
            ctx,
            cache_slots,
            force_swap_every_token: false,
            all_logits,
            fingerprints: true,
            profiled: false,
            f32_prefill: false,
            no_staging: false,
            stage_from: None,
            upload_cap_mib: None,
            no_lend: false,
            base_time: false,
            cpu_share: 0.0,
            speculation: spec,
        },
        generate::Ran {
            prompt,
            steps: r.steps,
            stop: r.stop,
            positions: r.positions,
            load_s,
            prompt_s: 0.0,
            generate_s,
            decoded,
            stats: e.stats(),
            // The cache's reads are not split into prefill's and decode's: decode_hits and decode_misses are 0.
            after_prefill: e.stats(),
            memory,
            vram: generate::Vram {
                heap_mib: mib(at_open.size),
                budget_at_open_mib: mib(at_open.budget),
                usage_at_open_mib: mib(at_open.usage),
                budget_mib: mib(loaded.budget),
                usage_mib: mib(loaded.usage),
            },
            spec: r.spec,
        },
        Some(generate::WarmInfo {
            path,
            id,
            dirty_id,
            cut,
            split,
            snapshot_bytes: r.snapshot_bytes,
            restore_bytes: r.restore.bytes,
            restore_kv_from: r.restore.kv_from,
            restore_ms: r.restore.ms,
            pool_limit_bytes: limit.bytes,
        }),
    );
    out.write_all(&serde_json::to_vec_pretty(&record)?)
        .with_context(|| format!("writing {}", json.display()))?;
    ops.live.drop_all();
    Ok(())
}

/// `artemis test6`: the verdict over the cold runs and both warm paths.
fn run_test6(
    golden: PathBuf,
    cold: Vec<String>,
    path1: Vec<String>,
    path2: Vec<String>,
    out: PathBuf,
) -> Result<()> {
    let file = |dir: &str, id: &str, ext: &str| golden.join(dir).join(format!("{id}.{ext}"));
    let read = |flag: &str, runs: &[String]| -> Result<Vec<(String, generate::Record)>> {
        test3::split_runs(flag, runs)?
            .into_iter()
            .map(|(id, f)| {
                Ok((
                    id.to_string(),
                    generate::read_record(std::path::Path::new(f))?,
                ))
            })
            .collect()
    };
    let compared = |runs: &[(String, generate::Record)]| -> Result<Vec<test3::PromptResult>> {
        runs.iter()
            .map(|(id, r)| {
                Ok(test3::compare(
                    id,
                    &read_tokens(&file("prompts", id, "tokens"))?,
                    &read_json(&file("expected", id, "json"))?,
                    &read_json(&file("forced-cpu", id, "json"))?,
                    r,
                ))
            })
            .collect()
    };
    let (cold_runs, path1_runs, path2_runs) = (
        read("--cold", &cold)?,
        read("--path1", &path1)?,
        read("--path2", &path2)?,
    );
    let (cold_res, path1_res, path2_res) = (
        compared(&cold_runs)?,
        compared(&path1_runs)?,
        compared(&path2_runs)?,
    );
    fn find<'a>(runs: &'a [(String, generate::Record)], id: &str) -> Option<&'a generate::Record> {
        runs.iter().find(|(i, _)| i == id).map(|(_, r)| r)
    }
    let flips = |res: &[test3::PromptResult], id: &str| {
        res.iter().find(|r| r.id == id).map_or(0, |r| r.flips.len())
    };
    // The long prompt's prefilled positions after the restore, against the cold run's: the gate.
    let bits = match (
        find(&cold_runs, test6::LONG_ID),
        find(&path1_runs, test6::LONG_ID),
    ) {
        (Some(c), Some(p)) => Some(test6::bit_check(
            test6::LONG_ID,
            c,
            p,
            test6::LONG_CUT,
            27_416,
        )),
        _ => None,
    };
    // `--all-logits` records a fingerprint for every prompt position.
    let has_prompt_positions =
        |r: &generate::Record| r.settings.all_logits && r.positions.len() >= r.prompt_tokens.len();
    let per_prompt: Vec<test6::PerPrompt> = cold_runs
        .iter()
        .map(|(id, c)| {
            let (p1, p2) = (find(&path1_runs, id), find(&path2_runs, id));
            let n = c.prompt_tokens.len();
            test6::PerPrompt {
                id: id.clone(),
                cold_flips: flips(&cold_res, id),
                path1_flips: flips(&path1_res, id),
                path2_flips: flips(&path2_res, id),
                path1_prefill_bits: p1
                    .filter(|p| has_prompt_positions(c) && has_prompt_positions(p))
                    .map(|p| test6::bit_check(id, c, p, test6::cut_for(id, n), n).equal),
                path1_decode_bits: p1.map(|p| test6::decode_bits_equal(c, p)),
                path2_decode_bits: p2.map(|p| test6::decode_bits_equal(c, p)),
            }
        })
        .collect();
    let v = test6::judge(cold_res, path1_res, path2_res, bits, per_prompt);
    fs::write(&out, serde_json::to_vec_pretty(&v)?)?;
    for p in &v.per_prompt {
        println!(
            "test 6: {}: flips cold {}, path 1 {}, path 2 {}; path 1 prefill bits {}, decode bits {}; path 2 decode bits {}",
            p.id,
            p.cold_flips,
            p.path1_flips,
            p.path2_flips,
            bits_word(p.path1_prefill_bits),
            bits_word(p.path1_decode_bits),
            bits_word(p.path2_decode_bits),
        );
    }
    println!("test 6: {}", v.line());
    if v.status != test3::Status::Pass {
        std::process::exit(1);
    }
    Ok(())
}

/// `artemis snapshot-times`' context: `generate`'s default.
const SNAPSHOT_TIMES_CTX: usize = 65_536;

/// `artemis snapshot-times`' arguments.
struct SnapshotTimesArgs {
    model: PathBuf,
    golden: PathBuf,
    json: PathBuf,
    pool_gib: u32,
    device: Option<String>,
}

fn millis(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Puts a fresh set of snapshots on `limit` in place of `live`'s, for the next measurement.
fn fresh_snapshots(live: &mut Loaded, limit: engine::PoolLimit, wait_each: bool) {
    let (e, _, snaps) = live.parts();
    *snaps = Snapshots::new(e, limit);
    snaps.wait_each(wait_each);
}

/// Resets the session and prefills `prompt` whole, timed.
fn prefill_plain(live: &mut Loaded, prompt: &[u32]) -> Result<std::time::Duration> {
    {
        let (e, s, _) = live.parts();
        s.reset(e)?;
    }
    let t = Instant::now();
    live.prefill(prompt)?;
    Ok(t.elapsed())
}

/// Resets the session and prefills `prompt` with the server's snapshot points (`cut` is the prompt end), timed.
fn prefill_with_points(
    live: &mut Loaded,
    prompt: &[u32],
    cut: usize,
) -> Result<std::time::Duration> {
    {
        let (e, s, _) = live.parts();
        s.reset(e)?;
    }
    let cut = engine::Cut {
        at: u32::try_from(cut)?,
        user_query: true,
    };
    let t = Instant::now();
    if engine::prefill_points(live, prompt, 0, Some(cut), || None)?.is_err() {
        bail!("the prefill with snapshot points was interrupted");
    }
    Ok(t.elapsed())
}

/// `artemis snapshot-times`. Its own function, not an arm of `main`: `main`'s frame is already large.
fn run_snapshot_times(args: SnapshotTimesArgs) -> Result<()> {
    let SnapshotTimesArgs {
        model,
        golden,
        json,
        pool_gib,
        device,
    } = args;
    let prompt_file = |id: &str| golden.join("prompts").join(format!("{id}.tokens"));
    let long = read_tokens(&prompt_file(test6::LONG_ID))?;
    let other = read_tokens(&prompt_file("01-code"))?;
    let n = long.len();
    let cut = snaptimes::last_im_start(&long)
        .with_context(|| format!("{} has no <|im_start|>", test6::LONG_ID))?;
    if n > SNAPSHOT_TIMES_CTX {
        bail!("the prompt is {n} tokens, longer than the context");
    }
    // The record is written at the end; an unwritable path fails now, and leaves no empty file if the run fails.
    fs::File::create(&json)
        .and_then(|_| fs::remove_file(&json))
        .with_context(|| format!("--json: can't write {}", json.display()))?;
    let uuid = device_uuid(device.as_deref())?;

    // 1. The link, on its own device, before the engine takes the GPU's memory.
    let link = snaptimes::LinkRatesJson::from(engine::link_rates(uuid, 256 << 20)?);
    println!(
        "link (256 MiB; best of 5, CPU reads best of 3), GB/s: HostCached {:.2} to host, {:.2} to VRAM, CPU reads {:.2}; \
         HostPinned {:.2} to host, {:.2} to VRAM, CPU reads {:.2}",
        link.cached_d2h,
        link.cached_h2d,
        link.cached_cpu_read,
        link.pinned_d2h,
        link.pinned_h2d,
        link.pinned_cpu_read,
    );
    if snaptimes::pinned_faster(link.cached_d2h, link.pinned_d2h)
        || snaptimes::pinned_faster(link.cached_h2d, link.pinned_h2d)
    {
        println!("link: HostPinned copies at least 10% faster than HostCached in one direction");
    }

    // 2. The engine with generate's defaults.
    let g = Gguf::open(&model).with_context(|| format!("opening {}", model.display()))?;
    let e = Engine::open(
        &g,
        &Dims::QWEN36,
        &Options {
            device_uuid: uuid,
            ctx: SNAPSHOT_TIMES_CTX,
            ..Options::default()
        },
    )?;
    let s = e.session(SNAPSHOT_TIMES_CTX)?;
    let limit = engine::limit_at_load(&e, pool_gib, 0)?;
    if limit.bytes == 0 {
        bail!(
            "the snapshot pool is 0 bytes (asked {} MiB{}): the measurement needs snapshots",
            limit.requested >> 20,
            if limit.cut {
                ", cut to what the host has free"
            } else {
                ""
            }
        );
    }
    eprintln!(
        "loaded on {}; prompt {n} tokens, cut {cut}; {} expert slots per layer; snapshot pool {} MiB{}",
        e.device_name(),
        e.cache_slots(),
        limit.bytes >> 20,
        if limit.cut { " (cut to what the host has free)" } else { "" },
    );
    let sizes = engine::Sizes::of(&Dims::QWEN36);
    let snaps = Snapshots::new(&e, limit);
    let mut live = Loaded::new(e, s, snaps);
    let tok_s = |t: std::time::Duration| n as f64 / t.as_secs_f64();

    // 3. The process's first prefill: it carries the warm-up, and is never compared against.
    let first = prefill_plain(&mut live, &long)?;
    println!(
        "first prefill (warm-up included, not compared): {:.2} s, {:.1} tok/s",
        first.as_secs_f64(),
        tok_s(first)
    );

    // 4. A take at the end of the prompt, a restore at once (the state alone), and one after a dirtying.
    live.parts().2.wait_each(true);
    let (taken, state_only) = {
        let (e, s, snaps) = live.parts();
        let Some(taken) = snaps.take(e, s, Marks::PROMPT_END)? else {
            bail!("the snapshot at {n} was skipped (the pool is too small)");
        };
        (taken, snaps.restore(e, s, taken.id)?)
    };
    if state_only.kv_from != n {
        bail!(
            "a restore at once copied KV from position {}, not {n}",
            state_only.kv_from
        );
    }
    let take_kv = taken.kv_time.context("the whole take copied no KV")?;
    prefill_plain(&mut live, &other)?;
    let after_dirty = {
        let (e, s, snaps) = live.parts();
        snaps.restore(e, s, taken.id)?
    };
    let whole = snaptimes::Whole {
        kv_bytes: sizes.segment(0, n),
        state_bytes: sizes.state,
        take_state_ms: millis(taken.state_time),
        take_kv_ms: millis(take_kv),
        restore_state_only_ms: millis(state_only.time),
        restore_after_dirty_ms: millis(after_dirty.time),
        restore_after_dirty_bytes: after_dirty.bytes,
        kv_from: after_dirty.kv_from,
    };
    println!(
        "whole prompt ({n} positions): take state {:.2} ms, KV {:.2} ms ({} MiB); restore of the state alone {:.2} ms; \
         restore after a {}-token dirtying from position 0: {:.2} ms ({} MiB from KV position {})",
        whole.take_state_ms,
        whole.take_kv_ms,
        whole.kv_bytes >> 20,
        whole.restore_state_only_ms,
        other.len(),
        whole.restore_after_dirty_ms,
        whole.restore_after_dirty_bytes >> 20,
        whole.kv_from,
    );

    // 5. The same prompt plain, then with the server's snapshot points (copies left to overlap): the snapshots' cost.
    fresh_snapshots(&mut live, limit, false);
    let plain = prefill_plain(&mut live, &long)?;
    fresh_snapshots(&mut live, limit, false);
    let overlapped = prefill_with_points(&mut live, &long, cut)?;
    let snapshots = {
        let (e, _, snaps) = live.parts();
        snaps.settle(e)?;
        snaps.take_log().len()
    };
    let cost = snaptimes::cost_pct(plain.as_secs_f64(), overlapped.as_secs_f64());
    println!(
        "plain prefill: {:.2} s, {:.1} tok/s; with {snapshots} snapshots overlapped: {:.2} s, {:.1} tok/s; \
         the snapshots cost {cost:.2}%",
        plain.as_secs_f64(),
        tok_s(plain),
        overlapped.as_secs_f64(),
        tok_s(overlapped),
    );

    // 6. The same prefill with each take's KV copy waited for: each take's times.
    fresh_snapshots(&mut live, limit, true);
    prefill_with_points(&mut live, &long, cut)?;
    let waited: Vec<snaptimes::Take> = {
        let (_, _, snaps) = live.parts();
        let log = snaps.take_log();
        log.iter()
            .map(|t| snaptimes::Take {
                pos: snaps.tree().node(t.id).map_or(0, |n| n.pos()),
                kv_bytes: t.bytes.saturating_sub(sizes.state),
                state_ms: millis(t.state_time),
                kv_ms: t.kv_time.map_or(0.0, millis),
                kv_estimate: snaptimes::Estimate::of(
                    t.bytes.saturating_sub(sizes.state),
                    link.cached_d2h,
                ),
            })
            .collect()
    };
    for t in &waited {
        println!(
            "take at {}: state {:.2} ms, KV {:.2} ms ({} MiB; estimated {:.2} ms at 28.43 GB/s, {:.2} at 14.22, {:.2} at the measured {:.2})",
            t.pos,
            t.state_ms,
            t.kv_ms,
            t.kv_bytes >> 20,
            t.kv_estimate.ms_at_28_43,
            t.kv_estimate.ms_at_14_22,
            t.kv_estimate.ms_at_measured,
            t.kv_estimate.measured_gb_per_s,
        );
    }

    // 7. A dirtying, then the restore of the prompt-end node through its chain.
    prefill_plain(&mut live, &other)?;
    // A cut node that a skipped take left out is a problem reported after the record is written, not an early exit.
    let (chain_len, chain_restore) = {
        let (e, s, snaps) = live.parts();
        let id = snaps
            .tree()
            .ids()
            .into_iter()
            .find(|&id| snaps.tree().node(id).is_some_and(|n| n.pos() == cut));
        match id {
            Some(id) => (snaps.tree().chain(id).len(), snaps.restore(e, s, id)?),
            None => (
                0,
                engine::Restore {
                    bytes: 0,
                    kv_from: 0,
                    time: std::time::Duration::ZERO,
                },
            ),
        }
    };
    println!(
        "restore of the prompt-end chain ({chain_len} nodes) after a dirtying: {:.2} ms, {} MiB from KV position {}",
        millis(chain_restore.time),
        chain_restore.bytes >> 20,
        chain_restore.kv_from,
    );
    let segmented = snaptimes::Segmented {
        snapshots,
        overlapped_prefill_s: overlapped.as_secs_f64(),
        overlapped_tok_s: tok_s(overlapped),
        cost_vs_plain_pct: cost,
        waited,
        restore_after_dirty_ms: millis(chain_restore.time),
        restore_after_dirty_bytes: chain_restore.bytes,
        chain_len,
    };

    // 8. The record, and each copy beside the estimates.
    let estimates = snaptimes::Estimates::new(&link, &whole, &segmented);
    for (what, est, got) in [
        ("take state", &estimates.take_state, whole.take_state_ms),
        ("take KV", &estimates.take_kv, whole.take_kv_ms),
        (
            "restore of the state alone",
            &estimates.restore_state_only,
            whole.restore_state_only_ms,
        ),
        (
            "restore after a dirtying",
            &estimates.restore_after_dirty,
            whole.restore_after_dirty_ms,
        ),
        (
            "chain restore after a dirtying",
            &estimates.chain_restore_after_dirty,
            segmented.restore_after_dirty_ms,
        ),
    ] {
        println!(
            "{what}: {got:.2} ms; estimated {:.2} ms at 28.43 GB/s, {:.2} at 14.22, {:.2} at the measured {:.2}",
            est.ms_at_28_43, est.ms_at_14_22, est.ms_at_measured, est.measured_gb_per_s
        );
    }
    let times = snaptimes::Times {
        format: snaptimes::FORMAT.into(),
        link,
        pool_limit_bytes: limit.bytes,
        pool_cut: limit.cut,
        prompt_tokens: n,
        cut_at: cut,
        first_prefill_s: first.as_secs_f64(),
        first_tok_s: tok_s(first),
        plain_prefill_s: plain.as_secs_f64(),
        plain_tok_s: tok_s(plain),
        whole,
        segmented,
        estimates,
    };
    fs::write(&json, serde_json::to_vec_pretty(&times)?)
        .with_context(|| format!("writing {}", json.display()))?;
    live.drop_all();
    // The figures are on disk; a run that missed a snapshot point is not a measurement of the server's points.
    let problems = snaptimes::problems(
        times.segmented.snapshots,
        times.segmented.waited.len(),
        times.segmented.chain_len,
    );
    if !problems.is_empty() {
        bail!(
            "{} (the record is in {})",
            problems.join("; "),
            json.display()
        );
    }
    Ok(())
}

/// `artemis test7`: the verdicts are written as each case ends, and the process exits 0 once every one is, 2 on a
/// harness error (a refused URL, an unreachable server, an unreadable log). Its own function, not an arm of `cli`:
/// `cli`'s frame is already large. The error chain names paths and counts, never the key.
#[inline(never)]
fn run_test7(args: artemis_tools::test7::Test7Args) -> ! {
    match artemis_tools::test7::run(args) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("test7: {e:#}");
            std::process::exit(2)
        }
    }
}

/// `artemis reuse-report`: writes the report and exits 0, or 2 on a harness error (an unreadable log directory). Its own
/// function, not an arm of `cli`, as `run_test7` is.
#[inline(never)]
fn run_reuse_report(args: artemis_tools::test7::report::ReuseReportArgs) -> ! {
    match artemis_tools::test7::report::run(args) {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("reuse-report: {e:#}");
            std::process::exit(2)
        }
    }
}

/// The stack `main`'s work runs on (the main thread's 1 MB on Windows is not enough, see `main`).
const STACK: usize = 32 << 20;

/// Runs the command line on a thread with a big stack. clap's derive builds every subcommand in one function whose
/// debug frame is 830 KB, on top of the 150 KB of `cli`'s own: adding `serve` to the enum took the main thread's 1 MB
/// Windows stack past its end, even for `--help`. A `run_warm` split could not help, since the frame is clap's.
fn main() -> Result<()> {
    // Before anything else can load a DLL (argument errors, logging, a backtrace's dbghelp.dll, the GPU driver): the
    // default search is System32 only, so a DLL planted beside artemis.exe is not picked up. Windows only.
    vk::limit_dll_search_to_system32();
    let thread = std::thread::Builder::new()
        .name("main".into())
        .stack_size(STACK)
        .spawn(cli)?;
    match thread.join() {
        Ok(result) => result,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

fn cli() -> Result<()> {
    let Some(cmd) = Cli::parse().cmd else {
        println!("{}", artemis_tools::setup::BARE_HELP);
        return Ok(());
    };
    match cmd {
        Cmd::Dump {
            model,
            tokens,
            out,
            cpu,
            gpu,
            threads,
            llama_rounding,
            f16_kv,
            force_from,
            device,
            mtp,
            hidden,
        } => {
            if cpu == gpu {
                bail!("pass exactly one of --cpu (the reference) and --gpu (the engine)");
            }
            let d = Dims::QWEN36;
            let g = Gguf::open(&model).with_context(|| format!("opening {}", model.display()))?;
            let toks = read_tokens(&tokens)?;
            if toks.is_empty() {
                bail!("{} holds no tokens", tokens.display());
            }
            if let (true, Some(hidden)) = (mtp, hidden) {
                if gpu || f16_kv {
                    bail!("--mtp is the CPU reference's, exact or with --llama-rounding");
                }
                let m = Model::open(&g, &d)?;
                if llama_rounding {
                    test2::check_llama_rounding_tokens(toks.len())?;
                }
                let rounding = if llama_rounding {
                    Rounding::LlamaCpu
                } else {
                    Rounding::Exact
                };
                let oracle = force_from
                    .map(|dir| tapdir::read_mtp(&dir, &d))
                    .transpose()?;
                let set = test2::dump_mtp(
                    &m,
                    &toks,
                    &tapdir::read_hidden(&hidden, d.n_embd)?,
                    oracle.as_ref(),
                    threads,
                    rounding,
                )?;
                let forced = if oracle.is_some() {
                    ", teacher-forced"
                } else {
                    ""
                };
                tapdir::write(
                    &set,
                    &out,
                    &format!("artemis reference MTP ({rounding:?} rounding){forced}"),
                )?;
                return Ok(());
            }
            let oracle = force_from.map(|dir| tapdir::read(&dir, &d)).transpose()?;
            if let Some(o) = &oracle {
                if o.n_tokens != toks.len() {
                    bail!(
                        "the forcing taps cover {} positions, the token file {}",
                        o.n_tokens,
                        toks.len()
                    );
                }
            }
            let mut forcer = oracle.as_ref().map(|o| tapdir::Forcer::new(o, &d));
            let mut rec = tapdir::Recorder::default();
            let source = if gpu {
                if llama_rounding || f16_kv {
                    bail!("--llama-rounding and --f16-kv choose the reference's rounding; the GPU engine has its own");
                }
                let e = Engine::from_gguf_with_ctx(
                    &g,
                    &d,
                    device_uuid(device.as_deref())?,
                    toks.len(),
                )?;
                eprintln!("loaded on {}", e.device_name());
                let mut s = e.session(toks.len())?;
                for (i, &t) in toks.iter().enumerate() {
                    let sink: &mut dyn Tap = match forcer.as_mut() {
                        Some(f) => f,
                        None => &mut rec,
                    };
                    s.decode_tapped(&e, t, sink)
                        .with_context(|| format!("position {i}"))?;
                    eprintln!("position {}/{}", i + 1, toks.len());
                }
                "artemis gpu engine".to_string()
            } else {
                let m = Model::open(&g, &d)?;
                if llama_rounding {
                    test2::check_llama_rounding_tokens(toks.len())?;
                }
                let rounding = match (llama_rounding, f16_kv) {
                    (true, true) => bail!("pass at most one of --llama-rounding and --f16-kv"),
                    (true, false) => Rounding::LlamaCpu,
                    (false, true) => Rounding::F16Kv,
                    (false, false) => Rounding::Exact,
                };
                let mut r = Reference::with_options(&m, threads, rounding);
                for (i, &t) in toks.iter().enumerate() {
                    let sink: &mut dyn Tap = match forcer.as_mut() {
                        Some(f) => f,
                        None => &mut rec,
                    };
                    r.step(t, sink).with_context(|| format!("position {i}"))?;
                    eprintln!("position {}/{}", i + 1, toks.len());
                }
                format!("artemis reference ({rounding:?} rounding)")
            };
            let forced = if oracle.is_some() {
                ", teacher-forced"
            } else {
                ""
            };
            let set = forcer.map_or(rec.set, |f| f.rec.set);
            tapdir::write(&set, &out, &format!("{source}{forced}"))?;
        }
        Cmd::Compare {
            a,
            b,
            threshold,
            tiny,
        } => {
            let d = dims(tiny);
            let c = compare::compare(&tapdir::read(&a, &d)?, &tapdir::read(&b, &d)?, &d)?;
            println!(
                "{:<28} {:>12} {:>12} {:>6}",
                "tap", "max rel", "max abs", "pos"
            );
            for x in &c.diffs {
                println!(
                    "{:<28} {:>12.3e} {:>12.3e} {:>6}",
                    x.tap.to_string(),
                    x.max_rel,
                    x.max_abs,
                    x.worst_pos
                );
            }
            for m in &c.missing {
                println!("missing: {m}");
            }
            match c
                .diffs
                .iter()
                .find(|x| x.max_rel.is_nan() || x.max_rel > threshold)
            {
                Some(x) => println!(
                    "first tap above {threshold:e}: {} ({:.3e} at position {})",
                    x.tap, x.max_rel, x.worst_pos
                ),
                None => println!("no tap above {threshold:e}"),
            }
        }
        Cmd::Test2 {
            llama_cpu,
            llama_vulkan,
            candidate,
            out,
            tiny,
            forced,
        } => {
            let d = dims(tiny);
            let cpu = tapdir::read(&llama_cpu, &d)?;
            let vk = llama_vulkan.map(|p| tapdir::read(&p, &d)).transpose()?;
            let cand = tapdir::read(&candidate, &d)?;
            let v = if forced {
                test2::test2_forced(&cpu, &cand, &d)?
            } else {
                test2::test2(&cpu, vk.as_ref(), &cand, &d)?
            };
            fs::write(&out, serde_json::to_vec_pretty(&v)?)?;
            summary("test 2", &v);
            // The forced rule gates; the free-running comparison is report-only.
            if forced && v.status != test2::Status::Pass {
                std::process::exit(1);
            }
        }
        Cmd::GpuTaps {
            reference,
            candidate,
            out,
            tiny,
        } => {
            let d = dims(tiny);
            let index = reference.join("index.json");
            let source = serde_json::from_slice::<serde_json::Value>(
                &fs::read(&index).with_context(|| format!("reading {}", index.display()))?,
            )?["source"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            if !source.contains("F16Kv") {
                bail!(
                    "--reference is {source:?}; the GPU rule compares against the reference with an f16 KV \
                     cache: make it with `artemis dump --cpu --f16-kv`"
                );
            }
            let v = test2::gpu_forced(
                &tapdir::read(&reference, &d)?,
                &tapdir::read(&candidate, &d)?,
                &d,
            )?;
            fs::write(&out, serde_json::to_vec_pretty(&v)?)?;
            summary("gpu taps", &v);
            if v.status != test2::Status::Pass {
                std::process::exit(1);
            }
        }
        Cmd::Generate {
            model,
            prompt,
            prompt_file,
            tokens_file,
            max_tokens,
            greedy,
            temp,
            top_k,
            top_p,
            min_p,
            presence_penalty,
            seed,
            ctx,
            cache_slots,
            force_swap_every_token,
            force_tokens,
            json,
            all_logits,
            no_fingerprints,
            profile,
            f32_prefill,
            no_staging,
            stage_from,
            cache_policy,
            upload_cap,
            no_lend,
            base_time,
            cpu_share,
            spec,
            device,
        } => {
            let sampling = generate::sampling(greedy, temp, top_k, top_p, min_p, presence_penalty)?;
            // Every speculation flag is checked before the prompt is read or the model opens: a flag beside
            // `--speculation off`, the CPU share or the base time beside speculation on, and each setting's range.
            let spec = spec.config(cpu_share, base_time)?;
            // An unwritable --json path fails now, not after the run. The record is written at the end.
            let json = match json {
                Some(path) => Some((
                    fs::File::create(&path)
                        .with_context(|| format!("--json: can't write {}", path.display()))?,
                    path,
                )),
                None => None,
            };
            let mut profile = match profile {
                Some(path) => Some((
                    fs::File::create(&path)
                        .with_context(|| format!("--profile: can't write {}", path.display()))?,
                    path,
                )),
                None => None,
            };
            let g = Gguf::open(&model).with_context(|| format!("opening {}", model.display()))?;
            let vocab = Vocab::from_gguf(&g).map_err(anyhow::Error::msg)?;
            let tk = Tokenizer::new(&vocab).map_err(anyhow::Error::msg)?;
            let eog = vocab.eog();
            let prompt: Vec<u32> = match (prompt, prompt_file, tokens_file) {
                (Some(p), None, None) => tk.try_encode(&p, true)?,
                (None, Some(f), None) => tk.try_encode(
                    &fs::read_to_string(&f).with_context(|| format!("reading {}", f.display()))?,
                    true,
                )?,
                (None, None, Some(f)) => read_tokens(&f)?,
                _ => bail!(
                    "give the prompt with exactly one of --prompt, --prompt-file and --tokens-file"
                ),
            };
            if prompt.is_empty() {
                bail!("the prompt is empty");
            }
            if prompt.len() > ctx {
                bail!(
                    "the prompt is {} tokens, longer than --ctx {ctx}",
                    prompt.len()
                );
            }
            // Checked before the model loads.
            let forced = match &force_tokens {
                Some(f) => {
                    let t = read_tokens(f)?;
                    generate::check_forced(&t, max_tokens, &eog)?;
                    generate::check_forced_vocab(&t, Dims::QWEN36.n_vocab)?;
                    Some(t)
                }
                None => None,
            };
            let t_load = Instant::now();
            let e = Engine::open(
                &g,
                &Dims::QWEN36,
                &Options {
                    device_uuid: device_uuid(device.as_deref())?,
                    ctx,
                    cache_slots,
                    force_swap_seed: force_swap_every_token.then_some(generate::FORCE_SWAP_SEED),
                    prefill_chunk: None,
                    poison_swaps: false,
                    f32_prefill,
                    no_staging,
                    stage_from,
                    poison_staging: false,
                    fail_staging_once: None,
                    chunk_rows: None,
                    cache_policy: match cache_policy {
                        PolicyArg::Adaptive => engine::CachePolicyKind::Adaptive,
                        PolicyArg::Static => engine::CachePolicyKind::Static,
                    },
                    upload_cap: upload_cap.map(|m| m << 20),
                    budget_every: None,
                    own_slots: None,
                    no_lend,
                    base_time,
                    cpu_share,
                    // `Session::verify` asserts it: on exactly when this run speculates.
                    speculation: spec.is_some(),
                    // The MTP when this run speculates with it.
                    mtp: generate::mtp_options_for_flags(spec.as_ref(), ctx)?,
                    load_ticks: None,
                },
            )?;
            let mut s = e.session(ctx)?;
            let load_s = t_load.elapsed().as_secs_f64();
            eprintln!(
                "loaded on {} in {load_s:.1} s; prompt {} tokens; VRAM: {} MiB of weights, {} MiB for the \
                 {ctx}-position session, {} MiB for {} expert slots per layer and {} more lent by staging, the {} cache \
                 policy, prefill on {} kernels, {} MiB of staging buffers{}{}{}",
                e.device_name(),
                prompt.len(),
                e.vram_bytes() >> 20,
                s.vram_bytes() >> 20,
                e.stats().cache_bytes >> 20,
                generate::slots_note(cache_slots, e.cache_slots()),
                e.stats().lent_slots,
                if e.policy_adapts() { "adaptive" } else { "static" },
                e.prefill_kernels(),
                e.stats().staging_bytes >> 20,
                generate::coop_note(e.coop_error()),
                generate::mtp_ring_note(e.mtp_ring_clamp()),
                if e.base_time() {
                    "; the base time: decode reads a resident expert for every HOST one (meaningless output)"
                } else {
                    ""
                },
            );
            let (at_open, loaded) = (e.vram_at_open(), e.vram_heap()?);
            let memory = generate::Memory::of(&e);
            let timing = profile.is_some();
            if timing {
                if !e.can_profile() {
                    // Don't leave the empty file the early create made.
                    if let Some((file, path)) = profile.take() {
                        drop(file);
                        let _ = fs::remove_file(path);
                    }
                    bail!("--profile: this GPU's compute queue has no timestamps");
                }
                e.set_timing(true);
            }
            let mut positions = generate::Fingerprints::new(
                json.is_some() && !no_fingerprints,
                prompt.len() + max_tokens,
            );
            let t0 = Instant::now();
            let logits = if all_logits {
                let mut last = Vec::new();
                s.prefill_each(&e, &prompt, 0, |i, l| {
                    positions.add(l);
                    if i + 1 == prompt.len() {
                        last = l.to_vec();
                    }
                    Ok::<(), engine::Error>(())
                })?;
                last
            } else {
                let l = s.prefill(&e, &prompt)?;
                positions.add(&l);
                l
            };
            let prompt_s = t0.elapsed().as_secs_f64();
            // The cache's reads so far are prefill's: what follows is decode's.
            let after_prefill = e.stats();
            let mut sampler = Sampler::new(sampling, seed);
            let mut force = forced.as_deref().map(generate::forced);
            let plain_greedy = sampling.temp <= 0.0 && sampling.presence_penalty == 0.0;
            let mut text = TextStream::new();
            let mut so = std::io::stdout().lock();
            // The submit trace (`ARTEMIS_SUBMIT_TRACE=FILE`) covers decode only: prefill's submits are dropped here.
            let trace_path = std::env::var_os(vk::SUBMIT_TRACE_ENV)
                .filter(|v| vk::submit_trace_wanted(Some(v)))
                .map(PathBuf::from);
            if trace_path.is_some() {
                if !e.submit_trace_on() {
                    bail!(
                        "{}: this GPU's driver can't calibrate its clock (VK_KHR_calibrated_timestamps)",
                        vk::SUBMIT_TRACE_ENV
                    );
                }
                e.take_submit_trace();
                e.take_submit_trace_dropped();
            }
            let t1 = Instant::now();
            let room = ctx - s.pos();
            let mut sampled = std::time::Duration::ZERO;
            // The prompt's last position was fingerprinted above; every later row the loop picks from was decoded or
            // verified, so each one is fingerprinted as it is picked.
            let mut first_row = true;
            let mut pick = |l: &[f32], top: u32| {
                if !std::mem::take(&mut first_row) {
                    positions.add(l);
                }
                // Only a profiled run pays for the clock.
                let ts = timing.then(Instant::now);
                let token = match &mut force {
                    Some(next) => next(l, top),
                    // The scan that found the top two is the greedy pick: no second pass over the logits.
                    None if plain_greedy => top,
                    None => sampler.sample(l),
                };
                if let Some(ts) = ts {
                    sampled += ts.elapsed();
                }
                token
            };
            let mut emit = |so_far: &[u32]| {
                write!(so, "{}", text.next_tokens(so_far, |t| tk.decode(t)))?;
                so.flush()?;
                Ok(())
            };
            let mut state = spec.clone().map(engine::spec::SpecState::new);
            let (steps, stop, spec_delta) = match state.as_mut() {
                Some(state) => {
                    let mut stepper = generate::Speculating::new(
                        state,
                        generate::Anyhow(engine::SessionVerifier { e: &e, s: &mut s }),
                        &prompt,
                    );
                    let (steps, stop) = generate::run(
                        logits,
                        max_tokens,
                        room,
                        &eog,
                        &mut pick,
                        &mut stepper,
                        &mut emit,
                    )?;
                    (steps, stop, Some(stepper.delta))
                }
                None => {
                    let mut stepper = generate::Plain(|t| Ok(s.decode(&e, t)?));
                    let (steps, stop) = generate::run(
                        logits,
                        max_tokens,
                        room,
                        &eog,
                        &mut pick,
                        &mut stepper,
                        &mut emit,
                    )?;
                    (steps, stop, None)
                }
            };
            // Positions run: every row picked after the prompt's, a stop token's row included.
            let decoded = steps.len().saturating_sub(1);
            let generate_s = t1.elapsed().as_secs_f64();
            // This run's state is fresh, so its totals are the run's.
            let depth_counts = state
                .as_ref()
                .map_or_else(Vec::new, |s| s.totals().mtp_depth_counts.to_vec());
            // The end-of-generation token is a step, but not text.
            let n_out = steps.len() - usize::from(stop == generate::Stop::EndOfGeneration);
            let out: Vec<u32> = steps[..n_out].iter().map(|x| x.token).collect();
            writeln!(so, "{}", text.finish_tokens(&out, |t| tk.decode(t)))?;
            let st = e.stats();
            eprintln!(
                "prompt: {} tokens in {prompt_s:.2} s ({:.1} tok/s); generated {} tokens, decoding {decoded} in \
                 {generate_s:.2} s ({:.1} tok/s); stopped at {}; cache: {} slots per layer at load, {} live at the \
                 end, lent rows included ({} MiB in its own chunks at the end), {:.1}% of routed experts from VRAM, {} swaps",
                prompt.len(),
                prompt.len() as f64 / prompt_s,
                out.len(),
                decoded as f64 / generate_s.max(1e-9),
                stop.describe(),
                st.slots_per_layer,
                e.live_slots(),
                st.cache_bytes >> 20,
                100.0 * st.from_vram().unwrap_or(0.0),
                st.swaps_done,
            );
            if let Some(d) = spec_delta {
                eprintln!(
                    "speculation: {} verify steps and {} plain ones; {} drafts verified, {} kept, {} steps kept fewer \
                     than all; lookup {} drafted, {} kept; MTP {} drafted, {} kept, {} probes",
                    d.steps,
                    d.plain_steps,
                    d.drafted,
                    d.accepted,
                    d.replays,
                    d.drafted_lookup,
                    d.accepted_lookup,
                    d.drafted_mtp,
                    d.accepted_mtp,
                    d.probes
                );
            }
            if let Some(path) = trace_path {
                let t = generate::submit_trace_json(&e.take_submit_trace());
                fs::write(&path, serde_json::to_vec(&t)?)
                    .with_context(|| format!("writing {}", path.display()))?;
                eprintln!("submit trace: written to {}", path.display());
                let dropped = e.take_submit_trace_dropped();
                if dropped > 0 {
                    eprintln!("submit trace: {dropped} submits were not kept (the log is full or a read failed)");
                }
            }
            if let Some((mut file, path)) = profile {
                let r = profile::report(&e.take_timings(), sampled.as_micros() as u64);
                file.write_all(&serde_json::to_vec_pretty(&r)?)
                    .with_context(|| format!("writing {}", path.display()))?;
                eprintln!(
                    "profile: prefill {:.1} ms of GPU time over {} chunks; decode {:.2} ms of GPU time a token over {} \
                     tokens{}{}; written to {}",
                    r.prefill.gpu_us / 1e3,
                    r.prefill.steps,
                    r.decode.gpu_us / 1e3 / r.decode.steps.max(1) as f64,
                    r.decode.steps,
                    generate::verify_note(r.verify.steps, r.verify.tokens, r.verify.gpu_us),
                    generate::mtp_note(r.mtp.steps, r.mtp.tokens, r.mtp.gpu_us),
                    path.display()
                );
            }
            if let Some((mut file, path)) = json {
                let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
                let record = generate::record(
                    &e,
                    &generate::Asked {
                        greedy,
                        sampling,
                        seed,
                        max_tokens,
                        ctx,
                        cache_slots,
                        force_swap_every_token,
                        all_logits,
                        fingerprints: !no_fingerprints,
                        profiled: timing,
                        f32_prefill,
                        no_staging,
                        stage_from,
                        upload_cap_mib: upload_cap,
                        no_lend,
                        base_time,
                        cpu_share,
                        speculation: spec,
                    },
                    generate::Ran {
                        prompt,
                        steps,
                        stop,
                        positions: positions.into_list(),
                        load_s,
                        prompt_s,
                        generate_s,
                        decoded,
                        stats: st,
                        after_prefill,
                        memory,
                        vram: generate::Vram {
                            heap_mib: mib(at_open.size),
                            budget_at_open_mib: mib(at_open.budget),
                            usage_at_open_mib: mib(at_open.usage),
                            budget_mib: mib(loaded.budget),
                            usage_mib: mib(loaded.usage),
                        },
                        spec: spec_delta,
                    },
                    None,
                );
                let record = generate::Record {
                    spec: record.spec.map(|r| r.with_depths(&[], &depth_counts)),
                    ..record
                };
                file.write_all(&serde_json::to_vec_pretty(&record)?)
                    .with_context(|| format!("writing {}", path.display()))?;
            }
        }
        Cmd::Test3 { golden, runs, out } => {
            let mut results = Vec::with_capacity(runs.len());
            for (id, file) in test3::split_runs("--run", &runs)? {
                let at = |dir: &str, ext: &str| golden.join(dir).join(format!("{id}.{ext}"));
                results.push(test3::compare(
                    id,
                    &read_tokens(&at("prompts", "tokens"))?,
                    &read_json(&at("expected", "json"))?,
                    &read_json(&at("forced-cpu", "json"))?,
                    &generate::read_record(std::path::Path::new(file))?,
                ));
            }
            let v = test3::judge(results);
            fs::write(&out, serde_json::to_vec_pretty(&v)?)?;
            for p in &v.prompts {
                println!("test 3: {}", p.line());
            }
            println!("test 3: {}", v.line());
            if v.status != test3::Status::Pass {
                std::process::exit(1);
            }
        }
        Cmd::Warm {
            model,
            golden,
            id,
            path,
            json,
            all_logits,
            ctx,
            cache_slots,
            snapshot_pool_gib,
            spec,
            device,
        } => run_warm(WarmArgs {
            model,
            golden,
            id,
            path,
            json,
            all_logits,
            ctx,
            cache_slots,
            snapshot_pool_gib,
            spec,
            device,
        })?,
        Cmd::Test6 {
            golden,
            cold,
            path1,
            path2,
            out,
        } => run_test6(golden, cold, path1, path2, out)?,
        Cmd::SnapshotTimes {
            model,
            golden,
            json,
            pool_gib,
            device,
        } => run_snapshot_times(SnapshotTimesArgs {
            model,
            golden,
            json,
            pool_gib,
            device,
        })?,
        Cmd::Hits { golden, runs, out } => {
            let mut pairs = Vec::with_capacity(runs.len());
            for r in &runs {
                let parts = r
                    .split_once('=')
                    .and_then(|(id, files)| files.split_once(',').map(|(p, s)| (id, p, s)));
                let Some((id, p, s)) = parts else {
                    bail!("--run {r:?} is not ID=POLICY,STATIC");
                };
                pairs.push(hits::Pair {
                    id: id.to_string(),
                    policy: generate::read_record(std::path::Path::new(p))?,
                    fixed: generate::read_record(std::path::Path::new(s))?,
                    prompt: read_tokens(&golden.join("prompts").join(format!("{id}.tokens")))?,
                });
            }
            let v = hits::judge(&pairs);
            fs::write(&out, serde_json::to_vec_pretty(&v)?)?;
            for g in &v.groups {
                println!(
                    "{}: {:.2}% of routed reads from the cache with the policy against {:.2}% static (decode {:.2}% \
                     against {:.2}%); {}: {}",
                    g.name,
                    g.policy_pct,
                    g.static_pct,
                    g.policy_decode_pct,
                    g.static_decode_pct,
                    g.rule,
                    if g.pass { "pass" } else { "FAIL" }
                );
            }
            for p in &v.problems {
                println!("problem: {p}");
            }
            if v.status != test3::Status::Pass {
                std::process::exit(1);
            }
        }
        Cmd::Test5 {
            runs,
            report_runs,
            out,
        } => {
            let read = |flag: &str, runs: &[String]| -> Result<Vec<(String, generate::Record)>> {
                runs.iter()
                    .map(|r| {
                        let Some((label, file)) = r.split_once('=') else {
                            bail!("{flag} {r:?} is not LABEL=FILE");
                        };
                        Ok((
                            label.to_string(),
                            generate::read_record(std::path::Path::new(file))?,
                        ))
                    })
                    .collect()
            };
            let records = read("--run", &runs)?;
            let reports = read("--report-run", &report_runs)?;
            let v = test5::judge_with_reports(&records, &reports);
            fs::write(&out, serde_json::to_vec_pretty(&v)?)?;
            println!(
                "test 5: {:?}; {} runs of {} positions{}",
                v.status,
                v.runs.len(),
                v.runs.first().map_or(0, |r| r.positions),
                if v.problems.is_empty() {
                    ", bit-identical".to_string()
                } else {
                    format!(": {}", v.problems.join("; "))
                }
            );
            for r in &v.reports {
                println!(
                    "test 5 report {}: tokens {}, logits {} (reported, not gated)",
                    r.label,
                    if r.identical_tokens {
                        "identical"
                    } else {
                        "DIFFER"
                    },
                    if r.identical_logits {
                        "identical"
                    } else {
                        "DIFFER"
                    }
                );
            }
            if v.status != test3::Status::Pass {
                std::process::exit(1);
            }
        }
        Cmd::Kl {
            model,
            base,
            corpus,
            ctx,
            chunks,
            cache_slots,
            device,
            out,
        } => {
            let mut b = kl::Base::open(&base)?;
            let h = b.header;
            kl::check_ctx(&h, ctx)?;
            let d = Dims::QWEN36;
            if h.n_vocab != d.n_vocab {
                bail!(
                    "the base has a vocabulary of {}, the model {}",
                    h.n_vocab,
                    d.n_vocab
                );
            }
            let chunks = chunks.unwrap_or(h.n_chunk);
            if chunks == 0 || chunks > h.n_chunk {
                bail!("--chunks {chunks}: the base has {} chunks", h.n_chunk);
            }
            let g = Gguf::open(&model).with_context(|| format!("opening {}", model.display()))?;
            if let Some(c) = corpus {
                let vocab = Vocab::from_gguf(&g).map_err(anyhow::Error::msg)?;
                let tk = Tokenizer::new(&vocab).map_err(anyhow::Error::msg)?;
                let text =
                    fs::read_to_string(&c).with_context(|| format!("reading {}", c.display()))?;
                // llama-perplexity drops one trailing newline and parses no special tokens.
                let text = text.strip_suffix('\n').unwrap_or(&text);
                kl::check_corpus(&b.tokens, &tk.try_encode(text, false)?)?;
            }
            let t0 = Instant::now();
            let e = Engine::open(
                &g,
                &d,
                &Options {
                    device_uuid: device_uuid(device.as_deref())?,
                    ctx: h.n_ctx,
                    cache_slots,
                    force_swap_seed: None,
                    prefill_chunk: None,
                    poison_swaps: false,
                    f32_prefill: false,
                    no_staging: false,
                    stage_from: None,
                    poison_staging: false,
                    fail_staging_once: None,
                    chunk_rows: None,
                    cache_policy: engine::CachePolicyKind::Adaptive,
                    upload_cap: None,
                    budget_every: None,
                    own_slots: None,
                    no_lend: false,
                    base_time: false,
                    cpu_share: 0.0,
                    speculation: false,
                    mtp: None,
                    load_ticks: None,
                },
            )?;
            eprintln!(
                "loaded on {} with {} expert slots per layer; scoring {chunks} chunks of {} ({} positions each)",
                e.device_name(),
                generate::slots_note(cache_slots, e.cache_slots()),
                h.n_ctx,
                h.scored()
            );
            let mut sum = kl::KlSum::default();
            let mut rec = Vec::new();
            let mut s = e.session(h.n_ctx)?;
            for c in 0..chunks {
                let toks = b.chunk_tokens(c).to_vec();
                // Each chunk starts from a fresh state, in the same allocations.
                if c > 0 {
                    s.reset(&e)?;
                }
                // Position p's logits predict token p + 1, so the last token is never fed; prefill computes the logits
                // of the scored positions only.
                s.prefill_each(&e, &toks[..h.n_ctx - 1], h.first(), |p, logits| {
                    b.next_record(&mut rec)
                        .with_context(|| format!("position {p}"))?;
                    sum.add(logits, &rec, toks[p + 1]);
                    anyhow::Ok(())
                })
                .with_context(|| format!("chunk {c}"))?;
                let (kld, unc) = sum.kld();
                eprintln!(
                    "chunk {}/{chunks}: mean KLD {kld:.6} ± {unc:.6} over {} positions ({:.0} s)",
                    c + 1,
                    sum.count,
                    t0.elapsed().as_secs_f64()
                );
            }
            let ((kld, kld_unc), (ppl, ppl_unc)) = (sum.kld(), sum.ppl());
            let report = kl::KlReport {
                n_ctx: h.n_ctx,
                chunks,
                base_chunks: Some(h.n_chunk),
                count: sum.count,
                kld,
                kld_unc,
                ppl,
                same_top: sum.same_top(),
                seconds: t0.elapsed().as_secs_f64(),
                cache_slots: e.cache_slots(),
            };
            fs::write(&out, serde_json::to_vec_pretty(&report)?)?;
            println!(
                "kl: {} positions in {chunks} chunks; PPL {ppl:.4} ± {ppl_unc:.4}; mean KLD {kld:.6} ± {kld_unc:.6}; same top {:.2}%",
                sum.count,
                100.0 * report.same_top
            );
        }
        Cmd::Test4 {
            floor_log,
            kl: report_file,
            out,
        } => {
            let text = String::from_utf8_lossy(
                &fs::read(&floor_log)
                    .with_context(|| format!("reading {}", floor_log.display()))?,
            )
            .into_owned();
            let (floor, floor_unc) = kl::parse_llama_kld(&text)?;
            let report: kl::KlReport = serde_json::from_slice(
                &fs::read(&report_file)
                    .with_context(|| format!("reading {}", report_file.display()))?,
            )?;
            let v = kl::test4(&report, floor);
            fs::write(&out, serde_json::to_vec_pretty(&v)?)?;
            println!(
                "test 4: {:?}; Artemis {:.6} ± {:.6} against llama.cpp's Vulkan-vs-CPU {floor:.6} ± {floor_unc:.6} (bound {:.6})",
                v.status, report.kld, report.kld_unc, v.bound
            );
            if let Some(p) = &v.problem {
                println!("test 4 problem: {p}");
            }
            if v.status != artemis_tools::test3::Status::Pass {
                std::process::exit(1);
            }
        }
        Cmd::Speed {
            decode,
            prefill,
            made_from,
            long_tokens,
            gate,
            base,
            out,
        } => {
            let made: BTreeMap<String, speed::MadeFrom> = serde_json::from_slice(
                &fs::read(&made_from)
                    .with_context(|| format!("reading {}", made_from.display()))?,
            )
            .with_context(|| format!("parsing {}", made_from.display()))?;
            let mut runs = Vec::with_capacity(decode.len());
            for d in &decode {
                let Some((prompt, file)) = d.split_once('=') else {
                    bail!("--decode {d:?} is not PROMPT=FILE");
                };
                runs.push((
                    prompt.to_string(),
                    generate::read_record(std::path::Path::new(file))?,
                ));
            }
            let pre = prefill
                .iter()
                .map(|p| generate::read_record(p))
                .collect::<Result<Vec<_>>>()?;
            let mut v = speed::judge(&runs, &pre, &made, &read_tokens(&long_tokens)?, gate);
            if let Some(b) = &base {
                match speed::base_row(&generate::read_record(b)?, &made) {
                    Ok(row) => v.base = Some(row),
                    Err(why) => v
                        .warnings
                        .extend(why.into_iter().map(|w| format!("base time: {w}"))),
                }
            }
            fs::write(&out, serde_json::to_vec_pretty(&v)?)
                .with_context(|| format!("writing {}", out.display()))?;
            for r in &v.decode {
                println!(
                    "speed: {} seed {}: {:.1} tok/s ({} tokens in {:.1} s, {:.1}% of the run's routed experts from VRAM, longest submit {:.0} ms, {} slots)",
                    r.prompt,
                    r.seed,
                    r.tok_s,
                    r.decoded,
                    r.seconds,
                    r.from_vram_pct,
                    r.longest_submit_ms,
                    r.slots
                );
            }
            for r in &v.prefill {
                println!(
                    "speed: prefill {} tokens in {:.1} s: {:.0} tok/s ({:.1}% of the run's routed experts from VRAM, longest submit {:.0} ms)",
                    r.tokens, r.seconds, r.tok_s, r.from_vram_pct, r.longest_submit_ms
                );
            }
            if let Some(b) = &v.base {
                println!(
                    "speed: base time {:.2} ms a token ({:.1} tok/s, {} tokens in {:.1} s, {} slots): every routed expert from VRAM; reported, not gated",
                    b.ms_per_token, b.tok_s, b.decoded, b.seconds, b.slots
                );
            }
            println!(
                "speed: {:?}: decode median {:.1} tok/s (bound {}), slower prefill {:.0} tok/s (bound {}){}",
                v.status,
                v.decode_median,
                v.decode_bound,
                v.prefill_slowest,
                v.prefill_bound,
                if v.gated { "" } else { "; reported, not gated" }
            );
            for p in &v.problems {
                println!("speed: problem: {p}");
            }
            for w in &v.warnings {
                println!("speed: warning: {w}");
            }
            if v.exits_nonzero() {
                std::process::exit(1);
            }
        }
        Cmd::TapsFilter { mtp } => println!(
            "{}",
            if mtp {
                taps::mtp_llama_filter()
            } else {
                taps::llama_filter()
            }
        ),
        Cmd::Test2Mtp {
            oracle,
            reference,
            out,
            tiny,
        } => {
            let d = dims(tiny);
            test2::check_mtp_reference_source(&tapdir::source(&reference)?)?;
            let v = test2::test2_mtp(
                &tapdir::read_mtp(&oracle, &d)?,
                &tapdir::read_mtp(&reference, &d)?,
                &d,
            )?;
            fs::write(&out, serde_json::to_vec_pretty(&v)?)?;
            summary("test 2 (MTP)", &v);
            if v.status != test2::Status::Pass {
                std::process::exit(1);
            }
        }
        Cmd::MtpCheck {
            model,
            oracle,
            tokens,
            json,
            ctx,
            device,
        } => {
            let code = mtp_check::run(
                &model,
                &oracle,
                &tokens,
                &json,
                ctx,
                device_uuid(device.as_deref())?,
            )?;
            if code != 0 {
                std::process::exit(code);
            }
        }
        Cmd::TinyModel { seed, out } => fs::write(&out, testing::tiny_model(seed))?,
        Cmd::VerifyCost {
            model,
            tokens_file,
            at,
            t,
            reps,
            warmup,
            ctx,
            verify,
            mtp,
            json,
            device,
        } => {
            let prompt = read_tokens(&tokens_file)?;
            let plan =
                verify_cost::Plan::with_mtp(prompt.len(), &at, &t, reps, warmup, verify, mtp)?;
            plan.check_ctx(ctx)?;
            // An unwritable --json path fails now, not after the run.
            let mut out = verify_cost::Output::create(&json)?;
            verify_cost::run(
                &model,
                &prompt,
                &plan,
                ctx,
                &mut out,
                tokens_file
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new(".")),
                device_uuid(device.as_deref())?,
            )?;
            out.finish()?;
        }
        Cmd::Serve(c) => std::process::exit(artemis_tools::serve::run::run(*c)?),
        Cmd::Setup(c) => std::process::exit(artemis_tools::setup::run(*c)?),
        Cmd::Test7(a) => run_test7(*a),
        Cmd::ReuseReport(a) => run_reuse_report(*a),
        Cmd::LookupSim(a) => artemis_tools::lookup_sim::run(a)?,
        Cmd::SpecCheck { cmd } => {
            // A golden verdict that isn't a pass exits 1, after its files are written.
            let code = spec_check::run(cmd)?;
            if code != 0 {
                std::process::exit(code);
            }
        }
        Cmd::ExtractVocab { model, out } => {
            let g = Gguf::open(&model).with_context(|| format!("opening {}", model.display()))?;
            let v = Vocab::from_gguf(&g).map_err(anyhow::Error::msg)?;
            fs::write(&out, v.to_extract())?;
            println!("{} tokens, {} merges", v.tokens.len(), v.merges.len());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `spec-check` nests `golden` and `speed`; the speed subcommand takes `generate`'s speculation flags.
    #[test]
    fn spec_check_has_golden_and_speed_subcommands() {
        let parse = |args: &[&str]| {
            let mut v = vec!["artemis", "spec-check"];
            v.extend(args);
            Cli::try_parse_from(v)
        };
        match parse(&["golden", "--model", "m.gguf", "--out", "o"])
            .unwrap()
            .cmd
            .unwrap()
        {
            Cmd::SpecCheck {
                cmd: spec_check::Cmd::Golden { .. },
            } => {}
            _ => panic!("not spec-check golden"),
        }
        let speed = [
            "speed",
            "--model",
            "m.gguf",
            "--edit-tokens",
            "e.tokens",
            "--out",
            "o",
            "--max-draft",
            "6",
        ];
        match parse(&speed).unwrap().cmd.unwrap() {
            Cmd::SpecCheck {
                cmd: spec_check::Cmd::Speed { spec, .. },
            } => assert_eq!(spec.max_draft, Some(6)),
            _ => panic!("not spec-check speed"),
        }
        assert!(parse(&[]).is_err());
        assert!(parse(&["golden"]).is_err());
    }

    /// `--upload-cap` takes 1 to 2048 MiB: `m << 20` can't wrap and the engine takes the same range.
    #[test]
    fn upload_cap_takes_one_to_2048_mib() {
        let parse = |cap: &str| {
            Cli::try_parse_from([
                "artemis",
                "generate",
                "--model",
                "m.gguf",
                "--upload-cap",
                cap,
            ])
        };
        for bad in ["0", "2049", "-1", "17592186044416"] {
            assert!(parse(bad).is_err(), "{bad} was accepted");
        }
        for good in ["1", "8", "2048"] {
            let cli = parse(good).unwrap_or_else(|e| panic!("{good}: {e}"));
            match cli.cmd.unwrap() {
                Cmd::Generate { upload_cap, .. } => {
                    assert_eq!(upload_cap, Some(good.parse().unwrap()))
                }
                _ => panic!("not generate"),
            }
        }
    }

    /// `--cpu-share` takes a float and defaults to 0 (the HOST path as it was); the engine refuses a value outside
    /// 0.0 ..= 1.0 before the model loads.
    #[test]
    fn cpu_share_defaults_to_zero_and_takes_a_float() {
        let parse = |extra: &[&str]| {
            let mut a = vec!["artemis", "generate", "--model", "m.gguf"];
            a.extend(extra);
            match Cli::try_parse_from(a).map(|c| c.cmd.unwrap()) {
                Ok(Cmd::Generate { cpu_share, .. }) => Ok(cpu_share),
                Ok(_) => panic!("not generate"),
                Err(e) => Err(e),
            }
        };
        assert_eq!(parse(&[]).unwrap(), 0.0);
        assert_eq!(parse(&["--cpu-share", "0.5"]).unwrap(), 0.5);
        assert_eq!(parse(&["--cpu-share", "1"]).unwrap(), 1.0);
        assert!(parse(&["--cpu-share", "lots"]).is_err());
    }

    /// `warm` takes path 1 or 2 only, and defaults to `generate`'s context with an 8 GiB pool.
    #[test]
    fn warm_takes_path_one_or_two_and_defaults_the_context_and_pool() {
        let parse = |path: &str| {
            Cli::try_parse_from([
                "artemis", "warm", "--model", "m.gguf", "--golden", "g", "--id", "01-code",
                "--path", path, "--json", "o.json",
            ])
            .map(|c| c.cmd.unwrap())
        };
        for bad in ["0", "3", "x"] {
            assert!(parse(bad).is_err(), "--path {bad} was accepted");
        }
        match parse("2").unwrap() {
            Cmd::Warm {
                path,
                ctx,
                snapshot_pool_gib,
                all_logits,
                cache_slots,
                ..
            } => {
                assert_eq!((path, ctx, snapshot_pool_gib), (2, 65_536, 8));
                assert!(!all_logits && cache_slots.is_none());
            }
            _ => panic!("not warm"),
        }
    }

    /// `test6` needs a cold run and `--out`; the paths are optional (a missing one is the verdict's problem).
    #[test]
    fn test6_takes_repeated_runs_and_needs_a_cold_one() {
        let parse = |extra: &[&str]| {
            let mut a = vec!["artemis", "test6", "--golden", "g", "--out", "v.json"];
            a.extend(extra);
            Cli::try_parse_from(a).map(|c| c.cmd.unwrap())
        };
        assert!(parse(&["--path1", "01-code=a.json"]).is_err());
        match parse(&[
            "--cold",
            "01-code=c.json",
            "--cold",
            "02-prose=d.json",
            "--path2",
            "01-code=p.json",
        ])
        .unwrap()
        {
            Cmd::Test6 {
                cold, path1, path2, ..
            } => {
                assert_eq!((cold.len(), path1.len(), path2.len()), (2, 0, 1));
            }
            _ => panic!("not test6"),
        }
    }

    /// The golden set's ids, sorted, and the prompt that dirties each: the next, the first after the last.
    #[test]
    fn golden_ids_are_the_sorted_prompts_and_the_last_is_dirtied_by_the_first() {
        let golden = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden");
        let ids = golden_ids(&golden).unwrap();
        assert_eq!(ids.len(), 20);
        assert_eq!(ids[0], "01-code");
        assert_eq!(test6::dirtier(&ids, "01-code"), Some("02-prose"));
        assert_eq!(test6::dirtier(&ids, test6::LONG_ID), Some("01-code"));
    }

    /// `snapshot-times` needs a model, the golden set and `--json`; the pool defaults to 8 GiB.
    #[test]
    fn snapshot_times_needs_its_paths_and_defaults_the_pool() {
        let parse = |extra: &[&str]| {
            let mut a = vec!["artemis", "snapshot-times"];
            a.extend(extra);
            Cli::try_parse_from(a).map(|c| c.cmd.unwrap())
        };
        assert!(parse(&["--model", "m.gguf", "--golden", "g"]).is_err());
        match parse(&["--model", "m.gguf", "--golden", "g", "--json", "t.json"]).unwrap() {
            Cmd::SnapshotTimes {
                pool_gib, device, ..
            } => assert_eq!((pool_gib, device), (8, None)),
            _ => panic!("not snapshot-times"),
        }
    }

    /// `verify-cost` needs a model, tokens and `--json`; the contexts, lengths and reps default to the plan's.
    #[test]
    fn verify_cost_defaults_its_contexts_lengths_and_reps() {
        let parse = |extra: &[&str]| {
            let mut a = vec!["artemis", "verify-cost"];
            a.extend(extra);
            Cli::try_parse_from(a).map(|c| c.cmd.unwrap())
        };
        assert!(parse(&["--model", "m.gguf", "--tokens-file", "p.tokens"]).is_err());
        let base = [
            "--model",
            "m.gguf",
            "--tokens-file",
            "p.tokens",
            "--json",
            "o.json",
        ];
        match parse(&base).unwrap() {
            Cmd::VerifyCost {
                at,
                t,
                reps,
                warmup,
                ctx,
                ..
            } => {
                assert_eq!(at, vec![64, 2048, 8192, 20_000]);
                assert_eq!(t, vec![1, 2, 3, 4, 5]);
                assert_eq!((reps, warmup, ctx), (10, 96, 65_536));
            }
            _ => panic!("not verify-cost"),
        }
        // --verify is off by default and a plain flag.
        match parse(&base).unwrap() {
            Cmd::VerifyCost { verify, .. } => assert!(!verify),
            _ => panic!("not verify-cost"),
        }
        let mut v = base.to_vec();
        v.push("--verify");
        match parse(&v).unwrap() {
            Cmd::VerifyCost { verify, .. } => assert!(verify),
            _ => panic!("not verify-cost"),
        }
        let mut a = base.to_vec();
        a.extend(["--at", "64,128", "--t", "2,8", "--reps", "3"]);
        match parse(&a).unwrap() {
            Cmd::VerifyCost { at, t, reps, .. } => {
                assert_eq!((at, t, reps), (vec![64, 128], vec![2, 8], 3));
            }
            _ => panic!("not verify-cost"),
        }
    }

    /// `serve` takes the server's flags and the scripts' stdin switch.
    #[test]
    fn serve_parses_its_flags() {
        let cli = Cli::try_parse_from([
            "artemis",
            "serve",
            "--model",
            "m.gguf",
            "--port",
            "8090",
            "--lazy-load",
            "false",
            "--idle-unload-s",
            "30",
            "--stop-on-stdin-close",
        ])
        .unwrap();
        match cli.cmd.unwrap() {
            Cmd::Serve(c) => {
                assert_eq!(c.args.model, Some(PathBuf::from("m.gguf")));
                assert_eq!(c.args.port, Some(8090));
                assert_eq!(c.args.lazy_load, Some(false));
                assert_eq!(c.args.idle_unload_s, Some(30));
                assert!(c.stop_on_stdin_close);
            }
            _ => panic!("not serve"),
        }
        // Without the switch it is off, and every flag is optional (the file or the defaults give the rest).
        match Cli::try_parse_from(["artemis", "serve"])
            .unwrap()
            .cmd
            .unwrap()
        {
            Cmd::Serve(c) => assert!(!c.stop_on_stdin_close && c.args.port.is_none()),
            _ => panic!("not serve"),
        }
    }

    /// The stdin switch is for scripts: `serve --help` doesn't list it.
    #[test]
    fn serve_help_hides_the_stdin_switch() {
        let err = Cli::try_parse_from(["artemis", "serve", "--help"])
            .err()
            .unwrap();
        let help = err.to_string();
        assert!(help.contains("--idle-unload-s"), "{help}");
        assert!(!help.contains("stop-on-stdin-close"), "{help}");
    }

    /// `test7` takes the server's URL, its key file and log directory, a comma-separated case list and the verdict
    /// directory; the cases are required.
    #[test]
    fn test7_parses_its_flags() {
        let cli = Cli::try_parse_from([
            "artemis",
            "test7",
            "--base-url",
            "http://127.0.0.1:8090",
            "--key-file",
            "key.txt",
            "--log-dir",
            "logs",
            "--case",
            "load,requests",
            "--out",
            "verdicts",
        ])
        .unwrap();
        match cli.cmd.unwrap() {
            Cmd::Test7(a) => {
                assert_eq!(a.base_url, "http://127.0.0.1:8090");
                assert_eq!(a.case, ["load", "requests"]);
                assert_eq!(a.prompts_root, PathBuf::from("."));
            }
            _ => panic!("not test7"),
        }
        assert!(Cli::try_parse_from([
            "artemis",
            "test7",
            "--base-url",
            "u",
            "--key-file",
            "k",
            "--log-dir",
            "l",
            "--out",
            "o"
        ])
        .is_err());
    }

    /// `reuse-report` takes one or more log directories (repeated or listed) and the report's path.
    #[test]
    fn reuse_report_parses_its_flags() {
        for dirs in [
            &["--log-dir", "a", "--log-dir", "b"][..],
            &["--log-dir", "a", "b"][..],
        ] {
            let mut argv = vec!["artemis", "reuse-report"];
            argv.extend_from_slice(dirs);
            argv.extend(["--out", "r.json"]);
            match Cli::try_parse_from(argv).unwrap().cmd.unwrap() {
                Cmd::ReuseReport(a) => {
                    assert_eq!(a.log_dir, [PathBuf::from("a"), PathBuf::from("b")]);
                    assert_eq!(a.out, PathBuf::from("r.json"));
                }
                _ => panic!("not reuse-report"),
            }
        }
        assert!(Cli::try_parse_from(["artemis", "reuse-report", "--out", "r.json"]).is_err());
    }
}
