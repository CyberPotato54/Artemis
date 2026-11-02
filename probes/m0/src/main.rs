use anyhow::Result;
use clap::{Parser, Subcommand};
use m0::{
    ctx::Ctx,
    gates,
    probes::{copy_bw, host_alloc, limits, matvec, read_bw, roundtrip},
    record,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    about = "Artemis milestone-0 measurements (throwaway probes; the numbers are what we keep)"
)]
struct Cli {
    /// JSON-lines file each measurement is appended to (and that `summarize` reads).
    #[arg(long, global = true, default_value = "results.jsonl")]
    out: PathBuf,
    /// Part of the GPU's name; the default is the only discrete GPU.
    #[arg(long, global = true)]
    device: Option<String>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Heaps, memory types, budgets, limits, features and cooperative-matrix shapes.
    Limits,
    /// How much pinned host memory can be allocated.
    HostAlloc(host_alloc::Args),
    /// Shader read bandwidth from VRAM or host memory.
    ReadBw(read_bw::Args),
    /// Host→VRAM copy bandwidth.
    CopyBw(copy_bw::Args),
    /// CPU↔GPU handoff latency: spin-wait, timeline semaphore, or plain submit.
    Roundtrip(roundtrip::Args),
    /// Q4_K or Q8_0 matvec speed and correctness, weights in VRAM or host memory.
    Matvec(matvec::Args),
    /// Markdown tables and the two gate verdicts from the results file.
    Summarize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let want = cli.device.as_deref();
    let rec = match &cli.cmd {
        Cmd::Summarize => {
            print!("{}", gates::summarize(&record::load(&cli.out)?));
            return Ok(());
        }
        Cmd::Limits => limits::run(want)?,
        gpu => {
            let ctx = Ctx::new(want)?;
            match gpu {
                Cmd::HostAlloc(a) => host_alloc::run(&ctx, a)?,
                Cmd::ReadBw(a) => read_bw::run(&ctx, a)?,
                Cmd::CopyBw(a) => copy_bw::run(&ctx, a)?,
                Cmd::Roundtrip(a) => roundtrip::run(&ctx, a)?,
                Cmd::Matvec(a) => matvec::run(&ctx, a)?,
                Cmd::Limits | Cmd::Summarize => unreachable!("handled above"),
            }
        }
    };
    record::append(&cli.out, &rec)
}
