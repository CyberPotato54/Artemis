//! The GPU side of test 1: put inputs on the GPU, run one kernel, read the output back, and `check` it
//! against the kernel's twin. Used by the `#[ignore]`d tests in `tests/gpu_*.rs`, and by the engine's tests.
use crate::{Kernel, Kernels};
/// The test RNG (`random_weights` takes it): re-exported so `cpu_expert` and the tests name it through `kernels`.
pub use qwen36::testing::Rng;
use qwen36::{ops, quant::WType, testing::random_quant};
use std::{cell::RefCell, io::Write};

pub struct Gpu {
    pub dev: vk::Device,
    pub k: Kernels,
    vram: RefCell<vk::Arena>,
    host: RefCell<vk::Arena>,
}

impl Gpu {
    /// Opens the GPU and builds every kernel. Panics with the reason if either fails.
    pub fn new() -> Gpu {
        let dev = vk::Device::open(None).unwrap_or_else(|e| panic!("opening the GPU: {e}"));
        let k = Kernels::new(&dev).unwrap_or_else(|e| panic!("building the kernels: {e}"));
        Gpu {
            dev,
            k,
            vram: RefCell::new(vk::Arena::new(vk::MemKind::Vram, 256 << 20, "test buffers")),
            host: RefCell::new(vk::Arena::new(
                vk::MemKind::HostPinned,
                256 << 20,
                "test host buffers",
            )),
        }
    }

    /// `b` in VRAM.
    pub fn bytes(&self, b: &[u8]) -> vk::Slice {
        let s = self
            .vram
            .borrow_mut()
            .alloc(&self.dev, b.len() as u64)
            .unwrap();
        self.dev.upload(s, b).unwrap();
        s
    }

    /// `b` in pinned host memory, which kernels read over PCIe.
    pub fn host_bytes(&self, b: &[u8]) -> vk::Slice {
        let mut host = self.host.borrow_mut();
        let s = host.alloc(&self.dev, b.len() as u64).unwrap();
        host.write(s, b);
        s
    }

    pub fn f32s(&self, v: &[f32]) -> vk::Slice {
        self.bytes(&f32_bytes(v))
    }

    /// `n` zeroed f32s in VRAM.
    pub fn zeros(&self, n: usize) -> vk::Slice {
        let s = self
            .vram
            .borrow_mut()
            .alloc(&self.dev, 4 * n as u64)
            .unwrap();
        let mut r = self.dev.record().unwrap();
        r.fill(s, 0);
        self.dev.submit(r).unwrap();
        s
    }

    pub fn read(&self, s: vk::Slice) -> Vec<f32> {
        self.dev
            .download(s)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    /// Records `f`, submits it and waits.
    pub fn run(&self, f: impl FnOnce(&Kernels, &mut vk::Recorder)) {
        let mut r = self.dev.record().unwrap();
        f(&self.k, &mut r);
        self.dev.submit(r).unwrap();
    }
}

impl Gpu {
    /// Records `f` on a profiled command buffer, submits it, waits, and returns the GPU time of each dispatch `f`
    /// records, in µs, in recording order (copies and fills are left out). Panics if the compute queue has no
    /// timestamps.
    pub fn run_timed(&self, f: impl FnOnce(&Kernels, &mut vk::Recorder)) -> Vec<f64> {
        let mut r = self
            .dev
            .record_profiled()
            .unwrap_or_else(|e| panic!("recording with timestamps: {e}"));
        f(&self.k, &mut r);
        self.dev
            .submit_profiled(r)
            .unwrap_or_else(|e| panic!("submitting with timestamps: {e}"))
            .into_iter()
            .filter(|t| t.name != "copy" && t.name != "fill")
            .map(|t| t.ns as f64 / 1e3)
            .collect()
    }
}

impl Default for Gpu {
    fn default() -> Gpu {
        Gpu::new()
    }
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn rand_f32(rng: &mut Rng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.uniform(lo, hi)).collect()
}

/// The narrowest spread of attention scores over the cached positions, across query heads: min over heads of
/// (max − min) of q·k / sqrt(hd). Attention tests assert it is several units, so the split's per-chunk max rescaling
/// (`attn_decode`, `attn_combine`) is exercised as trained heads exercise it. `kc` holds `[n_ctx][n_kv × hd]` as f32.
pub fn score_range(
    q: &[f32],
    kc: &[f32],
    n_ctx: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
) -> f32 {
    let group = n_head / n_kv;
    (0..n_head)
        .map(|h| {
            let qh = &q[h * hd..(h + 1) * hd];
            let s: Vec<f64> = (0..n_ctx)
                .map(|p| {
                    qwen36::ops::dot(qh, &kc[(p * n_kv + h / group) * hd..][..hd])
                        / (hd as f64).sqrt()
                })
                .collect();
            let (lo, hi) = s
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            (hi - lo) as f32
        })
        .fold(f32::INFINITY, f32::min)
}

/// `rows` random rows of `cols` values of type `ty`, within about ±sqrt(3 / cols), like the tiny model's weights.
pub fn random_weights(rng: &mut Rng, ty: WType, rows: usize, cols: usize) -> Vec<u8> {
    let amp = (3.0 / cols as f32).sqrt();
    let n = rows * cols;
    match ty {
        WType::F32 => (0..n)
            .flat_map(|_| rng.uniform(-amp, amp).to_le_bytes())
            .collect(),
        WType::Bf16 => (0..n)
            .flat_map(|_| half::bf16::from_f32(rng.uniform(-amp, amp)).to_le_bytes())
            .collect(),
        _ => random_quant(rng, ty, n, amp),
    }
}

/// Appends `line` (one JSON object) to the file `ARTEMIS_EVIDENCE` names, when that is set.
fn append_evidence(line: &str) {
    if let Ok(path) = std::env::var("ARTEMIS_EVIDENCE") {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| f.write_all(line.as_bytes()))
            .unwrap_or_else(|e| panic!("appending to {path}: {e}"));
    }
}

/// Test 1's check: `gpu` within `kernel`'s stated tolerance of `twin`. With `ARTEMIS_EVIDENCE` set to a
/// file, appends the measurement to it as one JSON line, for `tools/run-gpu-tests.ps1`.
pub fn check(kernel: Kernel, case: &str, gpu: &[f32], twin: &[f32]) {
    assert_eq!(gpu.len(), twin.len(), "{} {case}: lengths", kernel.name());
    let err = ops::rel_l2(gpu, twin);
    let tol = kernel.tol();
    let pass = err <= tol;
    let e = if err.is_finite() {
        format!("{err:e}")
    } else {
        "null".into()
    };
    append_evidence(&format!(
        "{{\"kernel\":\"{}\",\"case\":\"{case}\",\"err\":{e},\"tol\":{tol:e},\"pass\":{pass}}}\n",
        kernel.name()
    ));
    assert!(
        pass,
        "{} {case}: relative error {err:.3e} against the twin exceeds the stated {tol:.0e}",
        kernel.name()
    );
}

/// An identity report: whether `name`'s output equals its single-token counterpart's bit for
/// bit in `case`. Reported, never asserted: a kernel that misses is still usable, and the evidence names it. `name` is
/// a kernel's (`Kernel::X.name()`) or an engine-level report's ("verify_rows", "verify_state"). Prints a line, and
/// with `ARTEMIS_EVIDENCE` set appends `{"kind":"identity","kernel":<name>,"case":..,"equal":..}` to it. The line has
/// no `pass`: the runner's test 1 keeps `kind = identity` lines out of its per-kernel pass count.
pub fn identity(name: &str, case: &str, equal: bool) {
    println!(
        "identity {name} {case}: {}",
        if equal { "equal" } else { "differs" }
    );
    append_evidence(&identity_line(name, case, equal));
}

/// `s` as the inside of a JSON string: quotes, backslashes and control characters escaped.
pub fn json_escaped(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// The evidence line `identity` appends.
fn identity_line(name: &str, case: &str, equal: bool) -> String {
    let (name, case) = (json_escaped(name), json_escaped(case));
    format!(
        "{{\"kind\":\"identity\",\"kernel\":\"{name}\",\"case\":\"{case}\",\"equal\":{equal}}}\n"
    )
}

#[cfg(test)]
mod tests {
    use super::identity_line;

    #[test]
    fn an_identity_line_is_marked_and_has_no_pass() {
        // The runner's test 1 groups lines by `kernel` and counts those without `pass` as bad, unless they carry
        // `kind = identity`: this line must say so and must not pretend to be a check.
        let line = identity_line("gdn_seq", "against gdn_step, 24 tokens", false);
        assert_eq!(
            line,
            "{\"kind\":\"identity\",\"kernel\":\"gdn_seq\",\"case\":\"against gdn_step, 24 tokens\",\"equal\":false}\n"
        );
        assert!(!line.contains("\"pass\""));
    }

    #[test]
    fn an_identity_line_escapes_a_quote_a_backslash_and_a_newline() {
        let line = identity_line("verify_rows", "the \"a\\b\" case\n", true);
        assert_eq!(
            line,
            "{\"kind\":\"identity\",\"kernel\":\"verify_rows\",\"case\":\"the \\\"a\\\\b\\\" case\\u000a\",\"equal\":true}\n"
        );
    }
}
