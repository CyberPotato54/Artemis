//! First-run setup: the rules that choose Artemis's speculation settings from a short calibration on the real model.
//! `choose` holds the constants and the decision rules, `calibrate` the order, the time limit and the `Bench` the
//! command drives, `prompts` the calibration texts, `facts` the readings of the GPU and the host that setup prints
//! and checks, and `rig_bench` the bench over a real engine. This file is the command: `artemis setup`, and the
//! measuring `artemis serve` runs on its first start (`run_first`). Nothing here reads the network: setup reads the
//! model, the GPU and the OS.
pub mod calibrate;
pub mod choose;
pub mod facts;
pub mod prompts;
pub mod rig_bench;

use crate::serve::backend::spec_config;
use crate::serve::run::check_model;
use anyhow::{anyhow, bail, Context};
use calibrate::{calibrate, Clock, Load};
use choose::{
    candidates, choose, Calibration, Choice, SpecCandidate, KEY_MTP_COST, KEY_SLOPE, SETUP_LIMIT_S,
    WHY_ENABLED, WHY_SOURCES,
};
use engine::{SpecConfig, SpecSource};
use rig_bench::{LoadRecord, RigBench};
use server::logs::{LifecycleLine, Logger, SetupPriors};
use server::tuned::{
    self, SetupFacts, Tuned, TunedLocation, TunedSpeculation, ARTEMIS_VERSION, TUNED_SCHEMA,
};
use server::{Config, ServeArgs};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const GIB: f64 = (1u64 << 30) as f64;

/// The bytes `link_rates` copies each way.
const LINK_BYTES: u64 = 256 << 20;

/// What a bare `artemis` prints.
pub const BARE_HELP: &str =
    "Start the server with: artemis serve --model <path to the GGUF>. Its first start \
measures this PC; artemis setup measures again. artemis --help lists every command.";

/// A lock older than this is a dead setup's: twice the setup time limit.
const LOCK_STALE_S: u64 = 2 * SETUP_LIMIT_S as u64;

/// `artemis setup`: measures this PC and writes the settings `artemis serve` then uses.
#[derive(clap::Args, Clone, Debug, Default)]
pub struct SetupCmd {
    /// The config file, as for `serve`. Setup measures with the values you set there.
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// The model (a Qwen3.6-35B-A3B GGUF); the config's `model` otherwise.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// The GPU to measure, when there is more than one discrete GPU. Setup picks the one with the most memory
    /// otherwise.
    #[arg(long, value_name = "UUID")]
    pub device_uuid: Option<String>,
    /// Measure again and replace the tuned file. Without it, a tuned file that reads is reported and left alone.
    #[arg(long)]
    pub force: bool,
    /// Also write what was measured and chosen to this file, as JSON.
    #[arg(long, value_name = "FILE")]
    pub json: Option<PathBuf>,
}

/// Why setup runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// `serve` found no tuned file.
    FirstRun,
    /// `artemis setup`.
    Asked,
}

impl Reason {
    pub fn as_str(self) -> &'static str {
        match self {
            Reason::FirstRun => "first run",
            Reason::Asked => "asked",
        }
    }

    fn start_text(self) -> &'static str {
        match self {
            Reason::FirstRun => {
                "First run: Artemis is measuring this PC to choose its settings. This takes a few minutes and \
                 happens once. To measure again later, run: artemis setup --force"
            }
            Reason::Asked => {
                "Measuring this PC to choose Artemis's settings. This takes a few minutes."
            }
        }
    }
}

// ----- the lock ------------------------------------------------------------------------------------------------

/// The lock file beside the tuned file, held while a setup measures. Created with `create_new`, so two setups cannot
/// both hold it; removed when dropped, on every way out of the command.
#[derive(Debug)]
pub struct Lock {
    path: PathBuf,
}

/// `<tuned file>.lock`.
pub fn lock_path(tuned: &Path) -> PathBuf {
    let mut name = tuned.as_os_str().to_owned();
    name.push(".lock");
    PathBuf::from(name)
}

fn unix_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The pid and the time (Unix seconds) a lock file holds; `None` for a part that is missing or garbled.
fn read_lock(text: &str) -> (Option<u32>, Option<u64>) {
    let field = |name: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(name)?.strip_prefix('='))
            .and_then(|v| v.trim().parse::<u64>().ok())
    };
    (
        field("pid").and_then(|p| u32::try_from(p).ok()),
        field("time"),
    )
}

impl Lock {
    /// Takes the lock for the tuned file at `tuned`, at time `now`. A lock whose pid is not a running process, or that
    /// is older than `LOCK_STALE_S` (the fallback for a pid another process has since taken), is a dead setup's and is
    /// replaced; a live one is the refusal `Another artemis setup is running (pid <p>). If it is not, delete <lock>.`.
    pub fn take(tuned: &Path, now: u64) -> anyhow::Result<Lock> {
        let path = lock_path(tuned);
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("cannot create the folder {}", dir.display()))?;
        }
        for attempt in 0..3 {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut file) => {
                    use std::io::Write;
                    let held = write!(file, "pid={}\ntime={now}\n", std::process::id())
                        .and_then(|()| file.sync_all());
                    drop(file);
                    if let Err(e) = held {
                        let _ = std::fs::remove_file(&path);
                        bail!("cannot write the lock {}: {e}", path.display());
                    }
                    return Ok(Lock { path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let text = std::fs::read_to_string(&path).unwrap_or_default();
                    let (pid, time) = read_lock(&text);
                    // A lock with no readable time is judged by its file's age.
                    let time = time.or_else(|| {
                        let modified = std::fs::metadata(&path).ok()?.modified().ok()?;
                        Some(modified.duration_since(UNIX_EPOCH).ok()?.as_secs())
                    });
                    // A time in the future (a clock set back) is not old.
                    let old = time.is_some_and(|t| t <= now && now - t > LOCK_STALE_S);
                    // A holder that is gone (a power cut, a closed window, a kill) leaves the lock stale at once.
                    let gone = pid.is_some_and(|p| !process_alive(p));
                    if !(old || gone) || attempt == 2 {
                        bail!("{}", running_text(pid, &path));
                    }
                    break_stale(&path, &text)?;
                }
                Err(e) => bail!("cannot create the lock {}: {e}", path.display()),
            }
        }
        unreachable!("the third attempt returns or fails")
    }
}

fn running_text(pid: Option<u32>, lock: &Path) -> String {
    let who = match pid {
        Some(p) => format!("pid {p}"),
        None => "pid unknown".to_string(),
    };
    format!(
        "Another artemis setup is running ({who}). If it is not, delete {}.",
        lock.display()
    )
}

/// Whether a process with this id is running. A process that cannot be opened for lack of rights still exists.
#[cfg(windows)]
fn process_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_ACCESS_DENIED};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    const STILL_ACTIVE: u32 = 259;
    // SAFETY: plain Win32 calls; the handle is closed before returning and the exit code is a local out value.
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return GetLastError() == ERROR_ACCESS_DENIED;
        }
        let mut code = 0u32;
        let ok = GetExitCodeProcess(handle, &mut code) != 0;
        CloseHandle(handle);
        !ok || code == STILL_ACTIVE
    }
}

/// Whether a process with this id is running: its `/proc` entry exists. Without `/proc` the lock is judged by age.
#[cfg(not(windows))]
fn process_alive(pid: u32) -> bool {
    let proc = Path::new("/proc");
    !proc.is_dir() || proc.join(pid.to_string()).exists()
}

/// Removes the stale lock at `path`, which this run read as `seen`. The file is first renamed to a name of its own, so
/// two runs that read the same stale lock cannot both remove it: the one whose rename finds nothing goes on to the new
/// lock, and one that finds a different lock (another run's fresh one) puts it back and refuses.
fn break_stale(path: &Path, seen: &str) -> anyhow::Result<()> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    let mut name = path.as_os_str().to_owned();
    name.push(format!(".stale-{}-{nanos}", std::process::id()));
    let aside = PathBuf::from(name);
    match std::fs::rename(path, &aside) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => bail!("cannot replace the old lock {}: {e}", path.display()),
    }
    let taken = std::fs::read_to_string(&aside).unwrap_or_default();
    if taken == seen {
        let _ = std::fs::remove_file(&aside);
        return Ok(());
    }
    // The file was another run's new lock. Put it back, unless a third run has taken the name since.
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        use std::io::Write;
        let _ = file
            .write_all(taken.as_bytes())
            .and_then(|()| file.sync_all());
    }
    let _ = std::fs::remove_file(&aside);
    bail!("{}", running_text(read_lock(&taken).0, path))
}

impl Drop for Lock {
    fn drop(&mut self) {
        release_lock(&self.path);
    }
}

/// Removes the lock at `path` if it still holds this run's pid: a lock another run has replaced it with stays.
fn release_lock(path: &Path) {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    if read_lock(&text).0 == Some(std::process::id()) {
        let _ = std::fs::remove_file(path);
    }
}

/// What the Ctrl-C thread and the guard share.
struct Interrupt {
    /// Set when the guard is dropped or disarmed: a Ctrl-C after this one is not setup's to handle.
    disarmed: AtomicBool,
    /// Wakes the thread when the guard is disarmed.
    wake: tokio::sync::Notify,
    /// Set when the thread has ended.
    ended: AtomicBool,
    /// The log, once `setup_start` has been written to it: a Ctrl-C then writes `setup_failed` there.
    logger: Mutex<Option<Logger>>,
}

impl Interrupt {
    #[cfg(test)]
    fn ended(&self) -> bool {
        self.ended.load(Ordering::SeqCst)
    }
}

/// While armed, a Ctrl-C removes the lock, writes `setup_failed` if `setup_start` was written, and ends the process, so
/// an interrupted setup does not leave a lock that blocks the next one for twenty minutes. Setup disarms it before it
/// writes anything, so the text it prints on a Ctrl-C is true. Disarmed or dropped, its thread ends, and a Ctrl-C after
/// that is the process's own. (On Unix tokio leaves its SIGINT handler installed, so one Ctrl-C between the end of
/// setup and the server's own listener can still be absorbed.)
struct InterruptGuard(Arc<Interrupt>);

impl InterruptGuard {
    fn disarm(&self) {
        self.0.disarmed.store(true, Ordering::SeqCst);
        self.0.wake.notify_one();
    }

    /// Tells the guard where `setup_start` went.
    fn started(&self, logger: &Logger) {
        *self
            .0
            .logger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(logger.clone());
    }

    #[cfg(test)]
    fn state(&self) -> Arc<Interrupt> {
        self.0.clone()
    }
}

impl Drop for InterruptGuard {
    fn drop(&mut self) {
        self.disarm();
    }
}

fn guard_interrupt(lock: PathBuf) -> InterruptGuard {
    let state = Arc::new(Interrupt {
        disarmed: AtomicBool::new(false),
        wake: tokio::sync::Notify::new(),
        ended: AtomicBool::new(false),
        logger: Mutex::new(None),
    });
    let shared = state.clone();
    // A thread that cannot start leaves the lock to age out.
    let spawned = std::thread::Builder::new()
        .name("artemis-setup-int".into())
        .spawn(move || {
            struct Ended(Arc<Interrupt>);
            impl Drop for Ended {
                fn drop(&mut self) {
                    self.0.ended.store(true, Ordering::SeqCst);
                }
            }
            let _ended = Ended(shared.clone());
            let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            else {
                return;
            };
            let pressed = rt.block_on(async {
                tokio::select! {
                    r = tokio::signal::ctrl_c() => r.is_ok(),
                    () = shared.wake.notified() => false,
                }
            });
            if pressed && !shared.disarmed.load(Ordering::SeqCst) {
                release_lock(&lock);
                let logger = shared
                    .logger
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .clone();
                if let Some(logger) = logger {
                    logger.lifecycle_within(
                        &LifecycleLine::SetupFailed {
                            message: "Setup was interrupted. No settings were written.".into(),
                        },
                        std::time::Duration::from_secs(2),
                    );
                }
                eprintln!("Setup was interrupted. No settings were written.");
                std::process::exit(130);
            }
        });
    if spawned.is_err() {
        state.ended.store(true, Ordering::SeqCst);
    }
    InterruptGuard(state)
}

// ----- the GPU -------------------------------------------------------------------------------------------------

/// Which GPU setup measures on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuPick {
    pub index: usize,
    /// Setup chose among several discrete GPUs, so the tuned file records the choice as `device_uuid`.
    pub write_uuid: bool,
}

/// The GPU to measure: the discrete one named by `user_uuid`, else the only discrete GPU, else (several) the one with
/// the largest device-local heap, the first of equals. With several discrete GPUs the choice is recorded, named or not
/// (`build_tuned` still leaves it out when `device_uuid` is a key in the config file). An error says why there is none.
pub fn pick_gpu(gpus: &[vk::GpuInfo], user_uuid: Option<&str>) -> Result<GpuPick, String> {
    if let Some(text) = user_uuid {
        let uuid = vk::parse_uuid(text).ok_or_else(|| {
            "device_uuid is not a UUID: it is 32 hex digits, with or without dashes".to_string()
        })?;
        let several = gpus.iter().filter(|g| g.discrete).count() > 1;
        return gpus
            .iter()
            .position(|g| g.discrete && g.uuid == uuid)
            .map(|index| GpuPick {
                index,
                write_uuid: several,
            })
            .ok_or_else(|| {
                "device_uuid names none of the discrete GPUs listed above.".to_string()
            });
    }
    let discrete: Vec<usize> = (0..gpus.len()).filter(|&i| gpus[i].discrete).collect();
    match discrete.as_slice() {
        [] => Err(
            "No discrete GPU was found. Artemis needs one; the GPUs listed above are not usable."
                .into(),
        ),
        [only] => Ok(GpuPick {
            index: *only,
            write_uuid: false,
        }),
        many => {
            let mut best = many[0];
            for &i in &many[1..] {
                if gpus[i].heap.size > gpus[best].heap.size {
                    best = i;
                }
            }
            Ok(GpuPick {
                index: best,
                write_uuid: true,
            })
        }
    }
}

/// Every load ran on the GPU the tuned file will record, so the file's `gpu_uuid` is a measured fact.
fn check_loads_on(loads: &[LoadRecord], gpu: &vk::GpuInfo) -> anyhow::Result<()> {
    match loads.iter().find(|l| l.gpu_uuid != gpu.uuid) {
        None => Ok(()),
        Some(l) => bail!(
            "A model load ran on the GPU {}, and setup chose {} ({}). The measurements would describe two GPUs, so none \
             are written.",
            vk::format_uuid(&l.gpu_uuid),
            gpu.name,
            vk::format_uuid(&gpu.uuid)
        ),
    }
}

fn print_gpus(gpus: &[vk::GpuInfo], pick: Option<GpuPick>) {
    println!("GPUs found:");
    for (i, g) in gpus.iter().enumerate() {
        println!(
            "  {}: {:.0} GiB of memory, {}, driver {}{}",
            g.name,
            g.heap.size as f64 / GIB,
            if g.discrete {
                "discrete"
            } else {
                "not discrete"
            },
            g.driver,
            if pick.is_some_and(|p| p.index == i) {
                " (measuring on this one)"
            } else {
                ""
            }
        );
    }
}

/// The note under the GPU list for a card below the 16 GB target.
pub fn under_target_text(vram_bytes: u64) -> String {
    format!(
        "This GPU has {:.0} GiB of memory. Artemis targets 16 GB cards. It runs here with a smaller expert cache, \
         and its speed on cards under 16 GB is not claimed.",
        vram_bytes as f64 / GIB
    )
}

// ----- what the run measured and writes ------------------------------------------------------------------------

/// The loads the calibration will make for `cands`: one with MTP when `Base` carries it, and one without for the
/// rest.
pub fn planned_loads(cands: &[SpecCandidate], base: &SpecConfig) -> usize {
    let base_in = cands.contains(&SpecCandidate::Base);
    let with_mtp = base_in && base.has(SpecSource::Mtp);
    let without = cands.iter().any(|&c| c != SpecCandidate::Base) || (base_in && !with_mtp);
    usize::from(with_mtp) + usize::from(without)
}

/// The key a candidate's score has in the tuned file.
pub fn score_name(c: SpecCandidate, base: &SpecConfig) -> String {
    match c {
        SpecCandidate::Base => base
            .sources
            .iter()
            .map(|s| s.name())
            .collect::<Vec<_>>()
            .join("+"),
        SpecCandidate::LookupOnly => "lookup".into(),
        SpecCandidate::Off => "off".into(),
        SpecCandidate::MeasuredPriors => "measured_priors".into(),
    }
}

/// The load whose figures describe the server `choice` leaves: with MTP when the choice keeps it, else without.
fn chosen_load<'a>(choice: &Choice, loads: &'a [LoadRecord]) -> Option<&'a LoadRecord> {
    let mtp = choice.enabled && choice.sources.iter().any(|s| s == SpecSource::Mtp.name());
    let want = if mtp { Load::WithMtp } else { Load::WithoutMtp };
    loads.iter().find(|l| l.load == want).or(loads.first())
}

/// Everything a run measured, for the file it writes.
pub struct Measured<'a> {
    pub cfg: &'a Config,
    pub gpu: &'a vk::GpuInfo,
    pub pick: GpuPick,
    pub discrete: usize,
    pub link_h2d_gbps: f64,
    /// The OS's available memory before the load; `None` where it cannot be read.
    pub host_free: Option<u64>,
    pub cal: &'a Calibration,
    pub choice: &'a Choice,
    pub loads: &'a [LoadRecord],
    pub seconds: f64,
    /// `server::ids::iso_utc` of the finish.
    pub finished: String,
}

/// The tuned file's content and the reasons for its values. A key the user set is never written: the user's value
/// wins, so a tuned one would be ignored. The facts carry the server's version and the model file's size, which `stale`
/// compares.
pub fn build_tuned(m: &Measured) -> anyhow::Result<(Tuned, BTreeMap<String, String>)> {
    let user = &m.cfg.user_keys;
    let choice = m.choice;
    let model = m
        .cfg
        .model
        .as_ref()
        .ok_or_else(|| anyhow!("model is not set"))?;
    let model_bytes = std::fs::metadata(model)
        .with_context(|| format!("cannot read the size of the model {}", model.display()))?
        .len();
    let mut sp = TunedSpeculation::default();
    let mut why = BTreeMap::new();
    let mut keep = |key: &str| {
        if let Some(text) = choice.why.get(key) {
            why.insert(key.to_string(), text.clone());
        }
    };
    if !user.contains(WHY_ENABLED) {
        sp.enabled = Some(choice.enabled);
        keep(WHY_ENABLED);
    }
    // With speculation off the sources stay as they were.
    if choice.enabled && !user.contains(WHY_SOURCES) {
        sp.sources = Some(choice.sources.clone());
        keep(WHY_SOURCES);
    }
    if let Some((slope, d1)) = choice.priors {
        sp.cost_slope_ms = Some(slope);
        sp.mtp_cost_ms = Some(d1);
        keep("speculation.cost_slope_ms");
        keep("speculation.mtp_cost_ms");
    }
    let uuid = vk::format_uuid(&m.gpu.uuid);
    let device_uuid = (m.pick.write_uuid && !user.contains("device_uuid")).then(|| uuid.clone());
    if device_uuid.is_some() {
        why.insert(
            "device_uuid".into(),
            format!(
                "{} discrete GPUs were found; {} is the one setup measured on.",
                m.discrete, m.gpu.name
            ),
        );
    }
    let base = spec_config(&m.cfg.speculation);
    let load = chosen_load(choice, m.loads).ok_or_else(|| anyhow!("no model load was measured"))?;
    let pool_gib = match (m.host_free, load.host_heap_usage) {
        (Some(free), Some(used)) => {
            engine::pool_on_machine(m.cfg.snapshot_pool_gib, free, used).bytes >> 30
        }
        _ => 0,
    };
    let facts = SetupFacts {
        schema: TUNED_SCHEMA,
        artemis_version: ARTEMIS_VERSION.to_string(),
        finished: m.finished.clone(),
        seconds: m.seconds,
        limit_hit: m.cal.limit_hit,
        target: facts::is_target(m.gpu.heap.size),
        gpu_name: m.gpu.name.clone(),
        gpu_uuid: uuid,
        driver: m.gpu.driver.clone(),
        vram_bytes: m.gpu.heap.size,
        budget_bytes: m.gpu.heap.budget,
        cache_slots: u32::try_from(load.info.cache_slots).unwrap_or(u32::MAX),
        link_h2d_gbps: m.link_h2d_gbps,
        host_free_bytes: m.host_free.unwrap_or(0),
        pool_gib: u32::try_from(pool_gib).unwrap_or(u32::MAX),
        model_bytes,
        expected_tok_s: choice.expected_tok_s,
        measured_slope_ms: m.cal.measured.slope_ms,
        measured_mtp_d1_ms: m.cal.measured.mtp_d1_ms,
        scores: m
            .cal
            .scores
            .iter()
            .map(|(&c, s)| (score_name(c, &base), s.tok_s()))
            .collect(),
    };
    Ok((
        Tuned {
            device_uuid,
            speculation: sp,
            setup: facts,
        },
        why,
    ))
}

/// The lines setup prints at the end: how long it took, one line for each value it wrote with the reason, the host
/// memory and the pool, and the expected speed (left out when nothing was measured to give one).
pub fn end_text(t: &Tuned, why: &BTreeMap<String, String>, path: &Path) -> String {
    let f = &t.setup;
    let secs = f.seconds.max(0.0) as u64;
    let mut out = format!(
        "Done in {} min {} s. Settings written to {}.\n",
        secs / 60,
        secs % 60,
        path.display()
    );
    let mut line = |names: &[&str], values: String| {
        let text = names
            .iter()
            .find_map(|n| why.get(*n))
            .map(String::as_str)
            .unwrap_or("");
        out.push_str(&format!("  {values}: {text}\n"));
    };
    if let Some(u) = &t.device_uuid {
        line(&["device_uuid"], format!("device_uuid = \"{u}\""));
    }
    let sp = &t.speculation;
    if let Some(v) = sp.enabled {
        line(&[WHY_ENABLED], format!("speculation.enabled = {v}"));
    }
    if let Some(v) = &sp.sources {
        let list: Vec<String> = v.iter().map(|s| format!("\"{s}\"")).collect();
        line(
            &[WHY_SOURCES],
            format!("speculation.sources = [{}]", list.join(", ")),
        );
    }
    if let (Some(slope), Some(d1)) = (sp.cost_slope_ms, sp.mtp_cost_ms) {
        line(
            &["speculation.cost_slope_ms", "speculation.mtp_cost_ms"],
            format!("speculation.cost_slope_ms = {slope:.2} and speculation.mtp_cost_ms = {d1:.2}"),
        );
    }
    if f.limit_hit {
        out.push_str(
            "The time limit stopped the measuring early; the settings use what was measured by then.\n",
        );
    }
    if f.host_free_bytes > 0 {
        out.push_str(&format!(
            "Host memory: {:.1} GiB free before the load; the snapshot pool will be {} GiB.\n",
            f.host_free_bytes as f64 / GIB,
            f.pool_gib
        ));
    }
    if f.expected_tok_s > 0.0 {
        out.push_str(&format!(
            "Expected decode speed on short prompts: about {:.0} tok/s.\n",
            f.expected_tok_s
        ));
    }
    out
}

/// What `artemis setup` prints for a tuned file that already reads.
pub fn existing_text(t: &Tuned, path: &Path) -> String {
    let f = &t.setup;
    let mut out = format!(
        "This PC was already measured on {}, with {} (driver {}).\nThe settings are in {}.\n",
        f.finished,
        f.gpu_name,
        f.driver,
        path.display()
    );
    if f.expected_tok_s > 0.0 {
        out.push_str(&format!(
            "Expected decode speed on short prompts: about {:.0} tok/s.\n",
            f.expected_tok_s
        ));
    }
    out.push_str("To measure again, run: artemis setup --force\n");
    out
}

/// The speculation values the server will run with after this setup: the choice for the keys setup chose, and the
/// user's own value for the keys the user set (the tuned file leaves those out).
#[derive(Clone, Debug, PartialEq)]
pub struct Effective {
    pub enabled: bool,
    pub sources: Vec<String>,
    pub priors: Option<(f64, f64)>,
}

/// The effective values for `choice` under the user's `cfg`.
pub fn effective(cfg: &Config, choice: &Choice) -> Effective {
    let user = &cfg.user_keys;
    let sp = &cfg.speculation;
    let enabled = if user.contains(WHY_ENABLED) {
        sp.enabled
    } else {
        choice.enabled
    };
    // With speculation off the sources stay as they were.
    let sources = if choice.enabled && !user.contains(WHY_SOURCES) {
        choice.sources.clone()
    } else {
        sp.sources.clone()
    };
    let priors = if user.contains(KEY_SLOPE) || user.contains(KEY_MTP_COST) {
        Some((sp.cost_slope_ms, sp.mtp_cost_ms))
    } else {
        choice.priors
    };
    Effective {
        enabled,
        sources,
        priors,
    }
}

/// The line `setup_done` logs: the facts of the tuned file and the values the server will run with.
pub fn done_line(t: &Tuned, path: &Path, eff: &Effective) -> LifecycleLine {
    let f = &t.setup;
    LifecycleLine::SetupDone {
        seconds: f.seconds,
        limit_hit: f.limit_hit,
        target: f.target,
        gpu_name: f.gpu_name.clone(),
        vram_bytes: f.vram_bytes,
        cache_slots: f.cache_slots,
        scores: f.scores.clone(),
        enabled: eff.enabled,
        sources: eff.sources.clone(),
        priors: eff.priors.map(|(cost_slope_ms, mtp_cost_ms)| SetupPriors {
            cost_slope_ms,
            mtp_cost_ms,
        }),
        measured_slope_ms: f.measured_slope_ms,
        measured_mtp_d1_ms: f.measured_mtp_d1_ms,
        expected_tok_s: f.expected_tok_s,
        path: path.display().to_string(),
    }
}

/// The `--json` report: the facts, the chosen values and the reasons.
pub fn report_json(
    t: &Tuned,
    why: &BTreeMap<String, String>,
    loads: &[LoadRecord],
    path: &Path,
) -> serde_json::Value {
    let f = &t.setup;
    let sp = &t.speculation;
    serde_json::json!({
        "tuned_file": path.display().to_string(),
        "artemis_version": f.artemis_version,
        "finished": f.finished,
        "seconds": f.seconds,
        "limit_hit": f.limit_hit,
        "target": f.target,
        "gpu": {"name": f.gpu_name, "uuid": f.gpu_uuid, "driver": f.driver,
                "vram_bytes": f.vram_bytes, "budget_bytes": f.budget_bytes},
        "link_h2d_gbps": f.link_h2d_gbps,
        "host_free_bytes": f.host_free_bytes,
        "pool_gib": f.pool_gib,
        "model_bytes": f.model_bytes,
        "cache_slots": f.cache_slots,
        "scores_tok_s": f.scores,
        "measured_slope_ms": f.measured_slope_ms,
        "measured_mtp_d1_ms": f.measured_mtp_d1_ms,
        "expected_tok_s": f.expected_tok_s,
        "chosen": {
            "device_uuid": t.device_uuid,
            "speculation_enabled": sp.enabled,
            "speculation_sources": sp.sources,
            "cost_slope_ms": sp.cost_slope_ms,
            "mtp_cost_ms": sp.mtp_cost_ms,
        },
        "why": why,
        "loads": loads.iter().map(|l| serde_json::json!({
            "with_mtp": l.load == Load::WithMtp,
            "seconds": l.info.seconds,
            "cache_slots": l.info.cache_slots,
            "mtp_expert_slots": l.info.mtp_slots,
            "host_heap_usage_bytes": l.host_heap_usage,
        })).collect::<Vec<_>>(),
    })
}

// ----- the command ---------------------------------------------------------------------------------------------

/// The tuned file's path, or the refusal that says why setup has nowhere to write.
fn tuned_path(env: server::config::Env<'_>, windows: bool) -> anyhow::Result<PathBuf> {
    match tuned::tuned_location(env, windows) {
        TunedLocation::At(path) => Ok(path),
        TunedLocation::Off => bail!(
            "ARTEMIS_TUNED is off, so setup has nowhere to write. Unset it, or set it to a file path."
        ),
        TunedLocation::Unplaced => bail!("{}", tuned::unplaced_warning(windows)),
    }
}

/// The tuned file at `path` when it reads; `None` when it is missing or does not parse.
fn read_existing(path: &Path) -> Option<Tuned> {
    let text = std::fs::read_to_string(path).ok()?;
    tuned::parse(&text, path).ok()
}

/// The config setup measures with. The `--device-uuid` flag names the GPU for this run and is not a user key: a flag
/// given to `serve` lasts one run, but one given to `setup` is a choice for the file it writes, which has to name the
/// GPU for the starts after it. Only a `device_uuid` in the config file is a user key.
fn setup_config(
    cmd: &SetupCmd,
    env: server::config::Env<'_>,
    windows: bool,
) -> Result<Config, String> {
    let args = ServeArgs {
        config: cmd.config.clone(),
        model: cmd.model.clone(),
        ..ServeArgs::default()
    };
    let mut cfg = Config::load_without_tuned(&args, env, windows)?;
    if let Some(uuid) = &cmd.device_uuid {
        cfg.device_uuid = Some(uuid.clone());
    }
    Ok(cfg)
}

/// `artemis setup`.
pub fn run(cmd: SetupCmd) -> anyhow::Result<i32> {
    let env = |k: &str| std::env::var(k).ok();
    let windows = cfg!(windows);
    let cfg = setup_config(&cmd, &env, windows).map_err(anyhow::Error::msg)?;
    let path = tuned_path(&env, windows)?;
    if !cmd.force {
        if let Some(t) = read_existing(&path) {
            print!("{}", existing_text(&t, &path));
            return Ok(0);
        }
    }
    if let Some(json) = &cmd.json {
        // Found before the measuring: a bad path would otherwise cost a whole run.
        check_json_writable(json)?;
    }
    let prepared = prepare(&cfg, &path)?;
    // The log opens once the model is accepted: a refused run prunes and creates nothing.
    let log_dir = cfg.log_dir(&env, windows);
    let logger = Logger::new(log_dir.clone(), cfg.log_keep_days)
        .with_context(|| format!("cannot create the log directory {}", log_dir.display()))?;
    measure(
        prepared,
        &cfg,
        &path,
        Reason::Asked,
        &logger,
        cmd.json.as_deref(),
    )?;
    Ok(0)
}

/// What `serve` calls on a start that finds no tuned file: measures this PC and writes the file at `path`, writing the
/// `setup_start` line and then `setup_done` or `setup_failed`. Every engine it opens is dropped before it returns.
/// A file another run wrote between the config load and the lock is left alone.
pub fn run_first(cfg: &Config, path: &Path, logger: &Logger) -> anyhow::Result<()> {
    let prepared = match prepare(cfg, path) {
        Ok(p) => p,
        Err(e) => {
            // A refusal before the measuring (a lock another setup holds, a model that fails the checks) leaves the
            // same two lines as one during it, so the log of a start that exits 1 says why.
            logger.lifecycle(&LifecycleLine::SetupStart {
                reason: Reason::FirstRun.as_str().into(),
                path: path.display().to_string(),
            });
            logger.lifecycle(&LifecycleLine::SetupFailed {
                message: format!("{e:#}"),
            });
            return Err(e);
        }
    };
    if read_existing(path).is_some() {
        return Ok(());
    }
    measure(prepared, cfg, path, Reason::FirstRun, logger, None)
}

/// The lock and the model, checked as `serve` checks it: the steps that can refuse before anything is measured.
struct Prepared {
    checked: crate::serve::run::Checked,
    interrupt: InterruptGuard,
    /// Held until the struct drops, after the model above is closed.
    _lock: Lock,
}

fn prepare(cfg: &Config, path: &Path) -> anyhow::Result<Prepared> {
    let lock = Lock::take(path, unix_s())?;
    let interrupt = guard_interrupt(lock.path.clone());
    let model = cfg
        .model
        .as_deref()
        .ok_or_else(|| anyhow!("model is required: set it in the config file or pass --model"))?;
    let checked = check_model(model, cfg.allow_custom_template)?;
    Ok(Prepared {
        checked,
        interrupt,
        _lock: lock,
    })
}

fn measure(
    prepared: Prepared,
    cfg: &Config,
    path: &Path,
    reason: Reason,
    logger: &Logger,
    json: Option<&Path>,
) -> anyhow::Result<()> {
    let started = Instant::now();
    logger.lifecycle(&LifecycleLine::SetupStart {
        reason: reason.as_str().into(),
        path: path.display().to_string(),
    });
    prepared.interrupt.started(logger);
    println!("{}", reason.start_text());
    let outcome = measure_inner(
        &prepared.checked.g,
        &prepared.interrupt,
        cfg,
        path,
        started,
        json,
    );
    // The model is closed and the lock released before the result is reported.
    drop(prepared);
    match outcome {
        Ok((tuned, eff, text)) => {
            logger.lifecycle(&done_line(&tuned, path, &eff));
            print!("{text}");
            Ok(())
        }
        Err(e) => {
            logger.lifecycle(&LifecycleLine::SetupFailed {
                message: format!("{e:#}"),
            });
            Err(e)
        }
    }
}

struct WallClock(Instant);

impl Clock for WallClock {
    fn elapsed_s(&self) -> f64 {
        self.0.elapsed().as_secs_f64()
    }
}

/// The measuring and the writing. Returns the file's content, the values the server will run with and the end text;
/// writes the tuned file (and the JSON report, first) only after everything has been measured and chosen.
fn measure_inner(
    g: &gguf::Gguf,
    interrupt: &InterruptGuard,
    cfg: &Config,
    path: &Path,
    started: Instant,
    json: Option<&Path>,
) -> anyhow::Result<(Tuned, Effective, String)> {
    // The GPUs, before this process allocates anything: the heap figures are read untouched.
    let gpus = vk::list_gpus().map_err(|e| anyhow!("cannot list the GPUs: {e}"))?;
    let pick = pick_gpu(&gpus, cfg.device_uuid.as_deref());
    print_gpus(&gpus, pick.as_ref().ok().copied());
    let pick = pick.map_err(anyhow::Error::msg)?;
    let gpu = &gpus[pick.index];
    if !facts::is_target(gpu.heap.size) {
        println!("{}", under_target_text(gpu.heap.size));
    }
    if let Some(w) = facts::vram_warning(gpu.heap) {
        println!("{w}");
    }
    let base = spec_config(&cfg.speculation);
    let mtp = cfg.speculation.enabled && base.has(SpecSource::Mtp);
    let host_free = engine::os_available_bytes();
    if let Some(free) = host_free {
        if let Some(short) = facts::host_shortfall(free, facts::host_need(g, mtp)) {
            bail!("{short}");
        }
    }
    // Measured before the engine opens, on its own device.
    let rates = engine::link_rates(Some(gpu.uuid), LINK_BYTES)
        .map_err(|e| anyhow!("cannot measure the link to the GPU: {e}"))?;

    // The engine opens on the chosen GPU, with the user's own keys.
    let mut bench_cfg = cfg.clone();
    bench_cfg.device_uuid = Some(vk::format_uuid(&gpu.uuid));
    let cands = candidates(&cfg.user_keys, &base, cfg.speculation.enabled);
    let only_off = cands == [SpecCandidate::Off];
    let mut bench = RigBench::new(bench_cfg, g, planned_loads(&cands, &base), only_off)?;
    let (cal, _) = calibrate(
        &mut bench,
        &cands,
        &base,
        &cfg.user_keys,
        &WallClock(started),
    )?;
    let loads = bench.loads.clone();
    // Every engine is gone before the result is written.
    drop(bench);
    check_loads_on(&loads, gpu)?;
    let choice = choose(&cal, &base, cfg.speculation.enabled);
    let discrete = gpus.iter().filter(|g| g.discrete).count();
    let (tuned, why) = build_tuned(&Measured {
        cfg,
        gpu,
        pick,
        discrete,
        link_h2d_gbps: rates.cached_h2d,
        host_free,
        cal: &cal,
        choice: &choice,
        loads: &loads,
        seconds: started.elapsed().as_secs_f64(),
        finished: server::ids::iso_utc(unix_s()),
    })?;
    let report = match json {
        Some(json) => {
            let text = serde_json::to_string_pretty(&report_json(&tuned, &why, &loads, path))?
                + "
";
            Some((json, text))
        }
        None => None,
    };
    // Nothing is written before this point, so a Ctrl-C until here is a clean stop. From here the writes are the
    // run's and finish; the guard is disarmed so that its message stays true.
    interrupt.disarm();
    write_outputs(
        path,
        &tuned::render(&tuned, &why),
        report.as_ref().map(|(p, t)| (*p, t.as_str())),
    )?;
    let text = end_text(&tuned, &why, path);
    Ok((tuned, effective(cfg, &choice), text))
}

/// Writes the `--json` report and then the tuned file. The report goes first, so a bad report path leaves no tuned file
/// behind a nonzero exit; if the tuned file then cannot be written, the report is removed, so a failed run leaves
/// nothing.
fn write_outputs(path: &Path, tuned_text: &str, json: Option<(&Path, &str)>) -> anyhow::Result<()> {
    if let Some((json, text)) = json {
        tuned::write_atomic(json, text).map_err(anyhow::Error::msg)?;
    }
    if let Err(e) = tuned::write_atomic(path, tuned_text) {
        if let Some((json, _)) = json {
            let _ = std::fs::remove_file(json);
        }
        return Err(anyhow::Error::msg(e));
    }
    Ok(())
}

/// Whether a `--json` report can be written at `json`, found without leaving anything behind: the folder is made if it
/// is missing, an existing report is opened for appending and not changed, and a new file is created and removed.
fn check_json_writable(json: &Path) -> anyhow::Result<()> {
    let fail = |e: &dyn std::fmt::Display| {
        anyhow!("cannot write the --json report {}: {e}", json.display())
    };
    if let Some(dir) = json.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| fail(&e))?;
    }
    let existed = json.exists();
    std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(json)
        .map_err(|e| fail(&e))?;
    if !existed {
        let _ = std::fs::remove_file(json);
    }
    Ok(())
}

/// The tuned file's path for a start that should run setup: the config asks for it (`first_run_setup`) and there is a
/// place for the file with nothing in it.
pub fn first_run_path(cfg: &Config) -> Option<PathBuf> {
    match &cfg.tuned {
        server::TunedState::Missing(path) if cfg.first_run_setup => Some(path.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use choose::Score;
    use engine::spec::MeasuredCosts;
    use server::TunedState;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Scratch {
            let dir =
                std::env::temp_dir().join(format!("artemis-setup-{}-{tag}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const MIB: u64 = 1 << 20;
    const GIB_U: u64 = 1 << 30;

    fn gpu(name: &str, tag: u8, discrete: bool, heap_mib: u64) -> vk::GpuInfo {
        vk::GpuInfo {
            name: name.into(),
            uuid: [tag; 16],
            discrete,
            driver: "2.0.355 (0x800163)".into(),
            heap: vk::HeapBudget {
                size: heap_mib * MIB,
                budget: heap_mib * MIB * 95 / 100,
                usage: 0,
            },
        }
    }

    #[test]
    fn the_gpu_is_the_only_discrete_one_the_one_named_or_the_largest() {
        let one = [
            gpu("integrated", 1, false, 512),
            gpu("card", 2, true, 16_304),
        ];
        assert_eq!(
            pick_gpu(&one, None),
            Ok(GpuPick {
                index: 1,
                write_uuid: false
            })
        );
        // Two discrete cards: the one with more memory, and the tuned file records the choice.
        let two = [
            gpu("small", 3, true, 8_000),
            gpu("big", 4, true, 16_304),
            gpu("other big", 5, true, 16_304),
        ];
        assert_eq!(
            pick_gpu(&two, None),
            Ok(GpuPick {
                index: 1,
                write_uuid: true
            })
        );
        // A named UUID wins. With several discrete cards the choice is recorded (a config-file key keeps it out of the
        // file in `build_tuned`); with one card there is nothing to record.
        let named = vk::format_uuid(&[3; 16]);
        assert_eq!(
            pick_gpu(&two, Some(&named)),
            Ok(GpuPick {
                index: 0,
                write_uuid: true
            })
        );
        assert_eq!(
            pick_gpu(&one, Some(&vk::format_uuid(&[2; 16]))),
            Ok(GpuPick {
                index: 1,
                write_uuid: false
            })
        );
        // A UUID no discrete GPU has, one that is not a UUID, and no discrete GPU at all.
        let e = pick_gpu(&one, Some(&vk::format_uuid(&[1; 16]))).unwrap_err();
        assert!(e.contains("device_uuid"), "{e}");
        assert!(pick_gpu(&one, Some("zz"))
            .unwrap_err()
            .contains("not a UUID"));
        let none = [gpu("integrated", 1, false, 512)];
        assert!(pick_gpu(&none, None)
            .unwrap_err()
            .contains("No discrete GPU"));
        assert!(pick_gpu(&[], None).is_err());
    }

    #[test]
    fn planned_loads_follow_the_candidates() {
        use SpecCandidate::*;
        let with = SpecConfig::default();
        let without = SpecConfig {
            sources: vec![SpecSource::Lookup],
            ..SpecConfig::default()
        };
        assert_eq!(planned_loads(&[Base, LookupOnly, Off], &with), 2);
        assert_eq!(planned_loads(&[Base, Off], &with), 2);
        assert_eq!(planned_loads(&[Base, Off], &without), 1);
        assert_eq!(planned_loads(&[Base], &with), 1);
        assert_eq!(planned_loads(&[Off], &with), 1);
    }

    /// The id of a process that has already exited: a short-lived child of the test binary, waited for.
    fn exited_pid() -> u32 {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--list")
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let pid = child.id();
        child.wait().unwrap();
        pid
    }

    #[test]
    fn the_lock_is_exclusive_stale_after_twice_the_limit_and_released() {
        let s = Scratch::new("lock");
        let tuned = s.0.join("state").join("tuned.toml");
        let now = 1_800_000_000;
        let me = std::process::id();
        let lock = Lock::take(&tuned, now).unwrap();
        let held = std::fs::read_to_string(lock_path(&tuned)).unwrap();
        assert_eq!(
            held,
            format!(
                "pid={me}
time={now}
"
            )
        );
        let e = Lock::take(&tuned, now + 10).unwrap_err().to_string();
        assert_eq!(
            e,
            format!(
                "Another artemis setup is running (pid {me}). If it is not, delete {}.",
                lock_path(&tuned).display()
            )
        );
        drop(lock);
        assert!(!lock_path(&tuned).exists());

        // Exactly twice the limit is still live; a second past it is stale (the pid is a running process here, so
        // only the age decides).
        std::fs::write(
            lock_path(&tuned),
            format!(
                "pid={me}
time=1000
"
            ),
        )
        .unwrap();
        let e = Lock::take(&tuned, 1000 + LOCK_STALE_S)
            .unwrap_err()
            .to_string();
        assert!(
            e.starts_with(&format!("Another artemis setup is running (pid {me}).")),
            "{e}"
        );
        let lock = Lock::take(&tuned, 1000 + LOCK_STALE_S + 1).unwrap();
        assert!(std::fs::read_to_string(lock_path(&tuned))
            .unwrap()
            .contains(&format!("pid={me}")));
        drop(lock);

        // A time in the future is not old, and a lock with no readable time is judged by its file's age.
        std::fs::write(
            lock_path(&tuned),
            format!(
                "pid={me}
time=9999999999
"
            ),
        )
        .unwrap();
        assert!(Lock::take(&tuned, 1000).is_err());
        std::fs::write(lock_path(&tuned), "garbage").unwrap();
        let e = Lock::take(&tuned, unix_s()).unwrap_err().to_string();
        assert!(
            e.starts_with("Another artemis setup is running (pid unknown). If it is not, delete "),
            "{e}"
        );
        assert!(e.contains("tuned.toml.lock"), "{e}");
        assert!(Lock::take(&tuned, unix_s() + LOCK_STALE_S + 60).is_ok());
    }

    /// A lock whose pid is not a running process is a dead run's, however new: a power cut or a killed window leaves
    /// one, and the next start must not wait for it to age out.
    #[test]
    fn a_lock_whose_process_is_gone_is_replaced_at_once() {
        let s = Scratch::new("lock-dead");
        let tuned = s.0.join("tuned.toml");
        let dead = exited_pid();
        assert!(!process_alive(dead));
        assert!(process_alive(std::process::id()));
        let now = unix_s();
        std::fs::write(
            lock_path(&tuned),
            format!(
                "pid={dead}
time={now}
"
            ),
        )
        .unwrap();
        let lock = Lock::take(&tuned, now).unwrap();
        assert!(std::fs::read_to_string(lock_path(&tuned))
            .unwrap()
            .contains(&format!("pid={}", std::process::id())));
        drop(lock);
        assert!(!lock_path(&tuned).exists());
    }

    fn base_cfg(model: &Path) -> Config {
        Config {
            model: Some(model.to_path_buf()),
            ..Config::default()
        }
    }

    fn calibration(speeds: &[(SpecCandidate, f64)]) -> Calibration {
        Calibration {
            scores: speeds
                .iter()
                .map(|&(c, v)| {
                    (
                        c,
                        Score {
                            decoded: (v * 10.0).round() as usize,
                            seconds: 10.0,
                            passes: 2,
                        },
                    )
                })
                .collect(),
            priors: None,
            measured: MeasuredCosts {
                slope_ms: Some(4.5),
                mtp_d1_ms: None,
            },
            limit_hit: false,
        }
    }

    fn load_record(load: Load, usage: u64) -> LoadRecord {
        LoadRecord {
            load,
            info: calibrate::LoadInfo {
                seconds: 20.0,
                cache_slots: 112,
                mtp_slots: 64,
                mtp_slot_bytes: 121_634_816,
            },
            gpu_uuid: [2; 16],
            host_heap_usage: Some(usage),
        }
    }

    /// A tuned file built from a setup run's facts is not stale for the same binary and the same model (A2), and
    /// carries the server crate's version and the model file's size.
    #[test]
    fn a_tuned_file_from_a_run_is_not_stale_for_the_same_binary_and_model() {
        let s = Scratch::new("facts");
        let model = s.0.join("model.gguf");
        std::fs::write(&model, vec![7u8; 4096]).unwrap();
        let cfg = base_cfg(&model);
        let card = gpu("AMD Radeon RX 9070 XT", 2, true, 16_304);
        let base = spec_config(&cfg.speculation);
        let cal = calibration(&[
            (SpecCandidate::Base, 128.0),
            (SpecCandidate::LookupOnly, 103.0),
            (SpecCandidate::Off, 101.0),
        ]);
        let choice = choose(&cal, &base, true);
        let loads = [
            load_record(Load::WithMtp, 20 * GIB_U),
            load_record(Load::WithoutMtp, 19 * GIB_U),
        ];
        let (t, why) = build_tuned(&Measured {
            cfg: &cfg,
            gpu: &card,
            pick: GpuPick {
                index: 0,
                write_uuid: false,
            },
            discrete: 1,
            link_h2d_gbps: 25.5,
            host_free: Some(30 * GIB_U),
            cal: &cal,
            choice: &choice,
            loads: &loads,
            seconds: 187.5,
            finished: "2026-11-02T10:11:12Z".into(),
        })
        .unwrap();
        let text = tuned::render(&t, &why);
        let back = tuned::parse(&text, Path::new("tuned.toml")).unwrap();
        assert_eq!(back, t);
        assert_eq!(back.setup.artemis_version, server::tuned::ARTEMIS_VERSION);
        assert_eq!(back.setup.model_bytes, 4096);
        assert!(tuned::stale(&back.setup, tuned::ARTEMIS_VERSION, Some(4096)).is_empty());
        // Another model size is stale.
        assert_eq!(
            tuned::stale(&back.setup, tuned::ARTEMIS_VERSION, Some(4097)).len(),
            1
        );

        // What was measured is what is written.
        assert_eq!(back.setup.gpu_name, "AMD Radeon RX 9070 XT");
        assert_eq!(back.setup.gpu_uuid, vk::format_uuid(&[2; 16]));
        assert_eq!(back.setup.vram_bytes, 16_304 * MIB);
        assert_eq!(back.setup.cache_slots, 112);
        assert_eq!(back.setup.measured_slope_ms, Some(4.5));
        assert_eq!(back.setup.measured_mtp_d1_ms, None);
        assert_eq!(back.setup.host_free_bytes, 30 * GIB_U);
        // The load with MTP is the one the choice keeps: 30 GiB free, 20 used, 4 kept back, so 6 GiB.
        assert_eq!(back.setup.pool_gib, 6);
        assert!(back.setup.target);
        assert_eq!(back.setup.scores["lookup+mtp"], 128.0);
        assert_eq!(back.setup.scores["lookup"], 103.0);
        assert_eq!(back.setup.scores["off"], 101.0);
        // One GPU: no device_uuid. The speculation values and their reasons are there.
        assert_eq!(back.device_uuid, None);
        assert_eq!(back.speculation.enabled, Some(true));
        assert_eq!(
            back.speculation.sources,
            Some(vec!["lookup".to_string(), "mtp".to_string()])
        );
        assert!(text.contains("# MTP pays here"), "{text}");
    }

    #[test]
    fn a_users_keys_are_not_written_and_several_gpus_are() {
        let s = Scratch::new("facts-user");
        let model = s.0.join("model.gguf");
        std::fs::write(&model, b"0123456789").unwrap();
        let mut cfg = base_cfg(&model);
        cfg.user_keys = ["speculation.sources".to_string()].into();
        let card = gpu("Big", 4, true, 16_304);
        let base = spec_config(&cfg.speculation);
        let cal = calibration(&[(SpecCandidate::Base, 128.0), (SpecCandidate::Off, 101.0)]);
        let choice = choose(&cal, &base, true);
        let loads = [load_record(Load::WithMtp, 20 * GIB_U)];
        let m = |cfg: &Config, pick: GpuPick| {
            build_tuned(&Measured {
                cfg,
                gpu: &card,
                pick,
                discrete: 2,
                link_h2d_gbps: 20.0,
                host_free: None,
                cal: &cal,
                choice: &choice,
                loads: &loads,
                seconds: 60.0,
                finished: "2026-11-02T10:11:12Z".into(),
            })
            .unwrap()
        };
        let chose = GpuPick {
            index: 1,
            write_uuid: true,
        };
        let (t, why) = m(&cfg, chose);
        assert_eq!(t.speculation.sources, None, "the user set the sources");
        assert_eq!(t.speculation.enabled, Some(true));
        assert_eq!(
            t.device_uuid.as_deref(),
            Some(vk::format_uuid(&[4; 16]).as_str())
        );
        assert!(
            why["device_uuid"].starts_with("2 discrete GPUs were found; Big"),
            "{why:?}"
        );
        // Host memory unreadable: no figure, and the end text says nothing of it.
        assert_eq!((t.setup.host_free_bytes, t.setup.pool_gib), (0, 0));
        assert!(!end_text(&t, &why, Path::new("t.toml")).contains("Host memory"));
        // The user named the GPU: it is not written.
        cfg.user_keys.insert("device_uuid".into());
        let (t, _) = m(&cfg, chose);
        assert_eq!(t.device_uuid, None);
    }

    #[test]
    fn a_load_on_another_gpu_than_the_chosen_one_is_refused() {
        let card = gpu("card", 2, true, 16_304);
        let ok = [
            load_record(Load::WithMtp, 1),
            load_record(Load::WithoutMtp, 1),
        ];
        assert!(check_loads_on(&ok, &card).is_ok());
        let other = LoadRecord {
            gpu_uuid: [9; 16],
            ..load_record(Load::WithoutMtp, 1)
        };
        let e = check_loads_on(&[ok[0].clone(), other], &card)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains(&vk::format_uuid(&[9; 16])) && e.contains("card"),
            "{e}"
        );
        assert!(check_loads_on(&[], &card).is_ok());
    }

    /// `--device-uuid` on `setup` is a choice for the file it writes, not a user key: with two cards the tuned file
    /// records it. A `device_uuid` in the config file is a user key and stays out of the file.
    #[test]
    fn the_device_uuid_flag_is_written_and_a_config_file_key_is_not() {
        let s = Scratch::new("flag-uuid");
        let model = s.0.join("model.gguf");
        std::fs::write(&model, b"0123456789").unwrap();
        let flag = vk::format_uuid(&[3; 16]);
        let no_env = |_: &str| None;
        let cmd = SetupCmd {
            model: Some(model.clone()),
            device_uuid: Some(flag.clone()),
            ..SetupCmd::default()
        };
        let cfg = setup_config(&cmd, &no_env, true).unwrap();
        assert_eq!(cfg.device_uuid.as_deref(), Some(flag.as_str()));
        assert!(
            !cfg.user_keys.contains("device_uuid"),
            "{:?}",
            cfg.user_keys
        );

        let two = [gpu("small", 3, true, 8_000), gpu("big", 4, true, 16_304)];
        let pick = pick_gpu(&two, cfg.device_uuid.as_deref()).unwrap();
        assert_eq!(pick.index, 0, "the named card, not the larger one");
        let card = &two[pick.index];
        let base = spec_config(&cfg.speculation);
        let cal = calibration(&[(SpecCandidate::Base, 128.0), (SpecCandidate::Off, 101.0)]);
        let choice = choose(&cal, &base, true);
        let loads = [LoadRecord {
            gpu_uuid: card.uuid,
            ..load_record(Load::WithMtp, 20 * GIB_U)
        }];
        let build = |cfg: &Config| {
            build_tuned(&Measured {
                cfg,
                gpu: card,
                pick,
                discrete: 2,
                link_h2d_gbps: 20.0,
                host_free: None,
                cal: &cal,
                choice: &choice,
                loads: &loads,
                seconds: 60.0,
                finished: "2026-11-02T10:11:12Z".into(),
            })
            .unwrap()
            .0
        };
        assert_eq!(build(&cfg).device_uuid.as_deref(), Some(flag.as_str()));

        // The same key in the config file belongs to the user, flag or not.
        let file = s.0.join("artemis.toml");
        std::fs::write(
            &file,
            format!(
                "device_uuid = \"{flag}\"
"
            ),
        )
        .unwrap();
        let cmd = SetupCmd {
            config: Some(file),
            ..cmd
        };
        let cfg = setup_config(&cmd, &no_env, true).unwrap();
        assert!(cfg.user_keys.contains("device_uuid"), "{:?}", cfg.user_keys);
        assert_eq!(build(&cfg).device_uuid, None);
    }

    #[test]
    fn the_end_text_has_the_brief_lines_and_no_speed_when_none_was_measured() {
        let s = Scratch::new("end");
        let model = s.0.join("model.gguf");
        std::fs::write(&model, b"0123456789").unwrap();
        let cfg = base_cfg(&model);
        let card = gpu("AMD Radeon RX 9070 XT", 2, true, 16_304);
        let base = spec_config(&cfg.speculation);
        let cal = calibration(&[(SpecCandidate::Base, 128.0), (SpecCandidate::Off, 101.0)]);
        let choice = choose(&cal, &base, true);
        let loads = [load_record(Load::WithMtp, 20 * GIB_U)];
        let measured = |choice: &Choice, cal: &Calibration| {
            let (t, why) = build_tuned(&Measured {
                cfg: &cfg,
                gpu: &card,
                pick: GpuPick {
                    index: 0,
                    write_uuid: false,
                },
                discrete: 1,
                link_h2d_gbps: 25.0,
                host_free: Some(26 * GIB_U),
                cal,
                choice,
                loads: &loads,
                seconds: 187.4,
                finished: "2026-11-02T10:11:12Z".into(),
            })
            .unwrap();
            end_text(&t, &why, Path::new("T/tuned.toml"))
        };
        let text = measured(&choice, &cal);
        let mut lines = text.lines();
        assert_eq!(
            lines.next(),
            Some("Done in 3 min 7 s. Settings written to T/tuned.toml.")
        );
        assert!(
            text.contains("\n  speculation.enabled = true: Speculation stays on: 128.0 tok/s"),
            "{text}"
        );
        assert!(
            text.contains("\n  speculation.sources = [\"lookup\", \"mtp\"]: "),
            "{text}"
        );
        // 26 GiB free, 20 used by the load, 4 kept back: the pool is 2 GiB (milestone 3's floor at 26 GiB free).
        assert!(
            text.contains(
                "Host memory: 26.0 GiB free before the load; the snapshot pool will be 2 GiB.\n"
            ),
            "{text}"
        );
        assert!(
            text.ends_with("Expected decode speed on short prompts: about 128 tok/s.\n"),
            "{text}"
        );

        // Nothing measured: the speed line is left out, never "about 0 tok/s".
        let empty = calibration(&[]);
        let none = choose(&empty, &base, true);
        assert_eq!(none.expected_tok_s, 0.0);
        let text = measured(&none, &empty);
        assert!(!text.contains("Expected decode speed"), "{text}");
        assert!(!text.contains("0 tok/s"), "{text}");
        assert!(text.contains("Host memory:"), "{text}");
    }

    #[test]
    fn the_printed_texts_follow_the_public_text_rules() {
        let texts = [
            Reason::FirstRun.start_text().to_string(),
            Reason::Asked.start_text().to_string(),
            under_target_text(12 * GIB_U),
            BARE_HELP.to_string(),
            tuned::unplaced_warning(true),
        ];
        for t in &texts {
            assert!(
                !t.contains('\u{2014}') && !t.contains('\u{2013}') && t.is_ascii(),
                "{t}"
            );
        }
        assert_eq!(
            Reason::FirstRun.start_text(),
            "First run: Artemis is measuring this PC to choose its settings. This takes a few minutes and happens \
             once. To measure again later, run: artemis setup --force"
        );
        assert_eq!(
            Reason::Asked.start_text(),
            "Measuring this PC to choose Artemis's settings. This takes a few minutes."
        );
        assert_eq!(
            under_target_text(12 * GIB_U),
            "This GPU has 12 GiB of memory. Artemis targets 16 GB cards. It runs here with a smaller expert cache, \
             and its speed on cards under 16 GB is not claimed."
        );
    }

    #[test]
    fn the_hook_runs_only_for_a_missing_file_and_a_config_that_asks() {
        let at = PathBuf::from("T/tuned.toml");
        let mut cfg = Config {
            tuned: TunedState::Missing(at.clone()),
            ..Config::default()
        };
        assert_eq!(first_run_path(&cfg), Some(at));
        cfg.first_run_setup = false;
        assert_eq!(first_run_path(&cfg), None);
        for tuned in [
            TunedState::Off,
            TunedState::Loaded {
                path: "T/tuned.toml".into(),
                applied: vec![],
                ignored: vec![],
                stale: vec![],
                gpu_name: "g".into(),
                gpu_uuid: "u".into(),
                driver: "d".into(),
            },
        ] {
            let cfg = Config {
                tuned,
                ..Config::default()
            };
            assert_eq!(first_run_path(&cfg), None);
        }
    }

    /// The lifecycle lines in the day files of `dir`, oldest first.
    fn log_lines(dir: &Path) -> Vec<serde_json::Value> {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .collect();
        files.sort();
        files
            .iter()
            .flat_map(|f| {
                std::fs::read_to_string(f)
                    .unwrap()
                    .lines()
                    .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// `setup_done` reports what the server will run with: the choice for the keys setup chose, and the user's own
    /// value for the keys the user set.
    #[test]
    fn setup_done_reports_the_effective_values_when_the_user_set_keys() {
        let s = Scratch::new("effective");
        let model = s.0.join("model.gguf");
        std::fs::write(&model, b"0123456789").unwrap();
        let card = gpu("AMD Radeon RX 9070 XT", 2, true, 16_304);
        let loads = [load_record(Load::WithMtp, 20 * GIB_U)];
        let done = |cfg: &Config, cal: &Calibration| {
            let base = spec_config(&cfg.speculation);
            let choice = choose(cal, &base, cfg.speculation.enabled);
            let (t, _) = build_tuned(&Measured {
                cfg,
                gpu: &card,
                pick: GpuPick {
                    index: 0,
                    write_uuid: false,
                },
                discrete: 1,
                link_h2d_gbps: 25.0,
                host_free: Some(26 * GIB_U),
                cal,
                choice: &choice,
                loads: &loads,
                seconds: 60.0,
                finished: "2026-11-02T10:11:12Z".into(),
            })
            .unwrap();
            done_line(&t, Path::new("T/tuned.toml"), &effective(cfg, &choice))
        };

        // The user turned speculation off: the line says off, with the user's sources, never "on with nothing".
        let mut cfg = base_cfg(&model);
        cfg.speculation.enabled = false;
        cfg.user_keys = ["speculation.enabled".to_string()].into();
        let line = done(&cfg, &calibration(&[(SpecCandidate::Off, 101.0)]));
        assert!(line.human().contains("speculation off"), "{}", line.human());
        let LifecycleLine::SetupDone {
            enabled, sources, ..
        } = line
        else {
            panic!("not a setup_done line");
        };
        assert!(!enabled);
        assert_eq!(sources, cfg.speculation.sources);

        // The user named the sources: they are the line's sources, though the tuned file holds none.
        let mut cfg = base_cfg(&model);
        cfg.speculation.sources = vec!["lookup".to_string()];
        cfg.user_keys = ["speculation.sources".to_string()].into();
        let line = done(
            &cfg,
            &calibration(&[(SpecCandidate::Base, 128.0), (SpecCandidate::Off, 101.0)]),
        );
        assert!(
            line.human().contains("speculation on with lookup,"),
            "{}",
            line.human()
        );
        let LifecycleLine::SetupDone {
            enabled, sources, ..
        } = line
        else {
            panic!("not a setup_done line");
        };
        assert!(enabled);
        assert_eq!(sources, ["lookup"]);

        // The user set the priors: the line carries theirs, not a measured pair.
        let mut cfg = base_cfg(&model);
        cfg.speculation.cost_slope_ms = 9.0;
        cfg.speculation.mtp_cost_ms = 3.0;
        cfg.user_keys = ["speculation.cost_slope_ms".to_string()].into();
        let line = done(&cfg, &calibration(&[(SpecCandidate::Base, 128.0)]));
        let LifecycleLine::SetupDone { priors, .. } = line else {
            panic!("not a setup_done line");
        };
        let priors = priors.expect("the user's priors are the effective ones");
        assert_eq!((priors.cost_slope_ms, priors.mtp_cost_ms), (9.0, 3.0));
    }

    /// A first `serve` whose setup is refused before the measuring starts (here a live lock) leaves a `setup_start`
    /// and a `setup_failed` line that names the reason, and the lock it found stays.
    #[test]
    fn run_first_writes_the_failure_when_prepare_refuses() {
        let s = Scratch::new("first-lock");
        let tuned = s.0.join("state").join("tuned.toml");
        std::fs::create_dir_all(tuned.parent().unwrap()).unwrap();
        let live = format!("pid={}\ntime={}\n", std::process::id(), unix_s());
        std::fs::write(lock_path(&tuned), &live).unwrap();
        let logs = s.0.join("logs");
        let logger = Logger::new(logs.clone(), 0).unwrap();
        let cfg = Config {
            tuned: TunedState::Missing(tuned.clone()),
            ..Config::default()
        };
        let e = run_first(&cfg, &tuned, &logger).unwrap_err();
        let me = std::process::id();
        assert_eq!(
            e.to_string(),
            format!(
                "Another artemis setup is running (pid {me}). If it is not, delete {}.",
                lock_path(&tuned).display()
            )
        );
        let lines = log_lines(&logs);
        let events: Vec<&str> = lines.iter().map(|l| l["event"].as_str().unwrap()).collect();
        assert_eq!(events, ["setup_start", "setup_failed"], "{lines:?}");
        assert_eq!(lines[0]["reason"], "first run");
        assert!(
            lines[1]["message"]
                .as_str()
                .unwrap()
                .contains(&format!("pid {me}")),
            "{lines:?}"
        );
        assert_eq!(std::fs::read_to_string(lock_path(&tuned)).unwrap(), live);
        assert!(!tuned.exists());
    }

    /// Two runs that read the same stale lock: the one that finds a live lock where it meant to remove a stale one
    /// puts it back and refuses, and a lock is released only by the run that holds it.
    #[test]
    fn a_stale_lock_is_replaced_by_one_run_only_and_released_by_its_holder() {
        let s = Scratch::new("lock-race");
        let tuned = s.0.join("tuned.toml");
        let path = lock_path(&tuned);
        let stale = "pid=7\ntime=1000\n";
        // The stale lock is what this run read: it goes.
        std::fs::write(&path, stale).unwrap();
        break_stale(&path, stale).unwrap();
        assert!(!path.exists());
        // Another run replaced it first: this one finds that run's live lock, restores it and refuses.
        let fresh = "pid=9\ntime=2000\n";
        std::fs::write(&path, fresh).unwrap();
        let e = break_stale(&path, stale).unwrap_err().to_string();
        assert_eq!(
            e,
            format!(
                "Another artemis setup is running (pid 9). If it is not, delete {}.",
                path.display()
            )
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), fresh);
        let beside = std::fs::read_dir(&s.0).unwrap().count();
        assert_eq!(beside, 1, "a side file was left behind");
        // Nothing to remove: another run already did.
        std::fs::remove_file(&path).unwrap();
        break_stale(&path, stale).unwrap();

        // Drop removes the lock only while it still holds this run's pid.
        let lock = Lock::take(&tuned, unix_s()).unwrap();
        std::fs::write(&path, "pid=1\ntime=1\n").unwrap();
        drop(lock);
        assert!(path.exists(), "the lock of another run was removed");
        std::fs::remove_file(&path).unwrap();
        let lock = Lock::take(&tuned, unix_s()).unwrap();
        drop(lock);
        assert!(!path.exists());
    }

    /// The guard's thread ends when the guard is dropped, so it does not stay behind to swallow a later Ctrl-C.
    #[test]
    fn the_interrupt_guards_thread_ends_when_it_is_disarmed() {
        let s = Scratch::new("guard");
        let guard = guard_interrupt(s.0.join("tuned.toml.lock"));
        let state = guard.state();
        drop(guard);
        let start = Instant::now();
        while !state.ended() {
            assert!(
                start.elapsed().as_secs() < 20,
                "the guard's thread is still waiting for a Ctrl-C"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// A `--json` report written before a failed tuned write is removed, and a folder that cannot be written to is
    /// found before the measuring.
    #[test]
    fn the_json_report_is_removed_when_the_tuned_write_fails() {
        let s = Scratch::new("json-cleanup");
        let json = s.0.join("report.json");
        let blocker = s.0.join("blocker");
        std::fs::write(&blocker, "a file where a folder should be").unwrap();
        let tuned = blocker.join("tuned.toml");
        let e = write_outputs(&tuned, "tuned = 1\n", Some((&json, "{}\n"))).unwrap_err();
        assert!(e.to_string().contains("cannot write"), "{e:#}");
        assert!(!json.exists(), "the report outlived the failed run");
        // A good pair of paths writes both.
        let tuned = s.0.join("state").join("tuned.toml");
        write_outputs(&tuned, "tuned = 1\n", Some((&json, "{}\n"))).unwrap();
        assert_eq!(std::fs::read_to_string(&tuned).unwrap(), "tuned = 1\n");
        assert_eq!(std::fs::read_to_string(&json).unwrap(), "{}\n");

        // The check before the run: a writable path passes and leaves nothing, an existing report is untouched, and a
        // path under a file fails.
        let fresh = s.0.join("deep").join("new.json");
        check_json_writable(&fresh).unwrap();
        assert!(!fresh.exists());
        check_json_writable(&json).unwrap();
        assert_eq!(std::fs::read_to_string(&json).unwrap(), "{}\n");
        assert!(check_json_writable(&blocker.join("x.json")).is_err());
    }

    #[test]
    fn the_json_report_carries_the_facts_and_the_reasons() {
        let s = Scratch::new("json");
        let model = s.0.join("model.gguf");
        std::fs::write(&model, b"0123456789").unwrap();
        let cfg = base_cfg(&model);
        let card = gpu("AMD Radeon RX 9070 XT", 2, true, 16_304);
        let base = spec_config(&cfg.speculation);
        let cal = calibration(&[(SpecCandidate::Base, 128.0), (SpecCandidate::Off, 101.0)]);
        let choice = choose(&cal, &base, true);
        let loads = [load_record(Load::WithMtp, 20 * GIB_U)];
        let (t, why) = build_tuned(&Measured {
            cfg: &cfg,
            gpu: &card,
            pick: GpuPick {
                index: 0,
                write_uuid: false,
            },
            discrete: 1,
            link_h2d_gbps: 25.0,
            host_free: Some(26 * GIB_U),
            cal: &cal,
            choice: &choice,
            loads: &loads,
            seconds: 60.0,
            finished: "2026-11-02T10:11:12Z".into(),
        })
        .unwrap();
        let j = report_json(&t, &why, &loads, Path::new("T/tuned.toml"));
        assert_eq!(j["gpu"]["name"], "AMD Radeon RX 9070 XT");
        assert_eq!(j["scores_tok_s"]["lookup+mtp"], 128.0);
        assert_eq!(j["chosen"]["speculation_enabled"], true);
        assert_eq!(j["loads"][0]["with_mtp"], true);
        assert_eq!(j["loads"][0]["mtp_expert_slots"], 64);
        assert!(j["why"]["speculation.enabled"].is_string());
        let line = done_line(&t, Path::new("T/tuned.toml"), &effective(&cfg, &choice));
        let LifecycleLine::SetupDone {
            enabled,
            sources,
            scores,
            ..
        } = line
        else {
            panic!("not a setup_done line");
        };
        assert!(enabled);
        assert_eq!(sources, ["lookup", "mtp"]);
        assert_eq!(scores["off"], 101.0);
    }
}
