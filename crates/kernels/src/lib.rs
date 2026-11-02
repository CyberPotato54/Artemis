//! Artemis's GPU kernels: GLSL compute shaders compiled by `build.rs`, one recording helper per
//! kernel (`dispatch`), and a CPU twin of each (`twin`) that mirrors its arithmetic, reduction order included. Test 1
//! checks every kernel against its twin within the tolerance stated here; CI checks every twin against the
//! reference's operations (`qwen36::ops`).
//!
//! Every kernel takes buffer device addresses through push constants. No kernel assumes a subgroup size: reductions
//! are shared-memory trees over fixed workgroup sizes, so their order depends only on shapes. There are no float
//! atomics.
/// Declares that a `#[repr(C)]` push-constant struct is a `vk::PushConstants`: `push_constants!(Type: FieldType, ...)`
/// names the fields' types in order. The assertion fails the build if the struct has padding (its size would exceed
/// the fields' sizes), the one thing that would let `Recorder::dispatch` read uninitialized bytes.
macro_rules! push_constants {
    ($ty:ident: $($field:ty),+ $(,)?) => {
        // SAFETY: the struct is `#[repr(C)]`, holds only integers and floats, and the assertion below proves it has
        // no padding.
        unsafe impl vk::PushConstants for $ty {}
        const _: () = assert!(
            std::mem::size_of::<$ty>() == 0 $(+ std::mem::size_of::<$field>())+,
            "a push-constant struct has padding"
        );
    };
}

pub mod dispatch;
pub mod testing;
pub mod tokens;
pub mod twin;

use dispatch::*;
use qwen36::quant::WType;
use tokens::attention::{AttnCombineTokensPush, AttnTokensPush};
use tokens::experts::{ExpertsDownTokensPush, ExpertsGateUpTokensPush, MoeUniquePush};
use tokens::matvec::MatvecTokensPush;

/// Tolerances, as relative L2 over each output vector: a kernel against its twin (test 1), and the twin
/// against `qwen36::ops` (CI).
pub mod tol {
    /// Kernels that compute each output from a few inputs.
    pub const ELEMENTWISE: f64 = 1e-6;
    /// Kernels that sum over a row, a head or the context.
    pub const REDUCTION: f64 = 1e-5;
}

macro_rules! kernels {
    ($($v:ident => $name:literal, $push:ty, $tol:expr;)*) => {
        /// Every kernel, named after its shader file.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Kernel {
            $($v),*
        }

        impl Kernel {
            pub const ALL: &'static [Kernel] = &[$(Kernel::$v),*];

            pub fn name(self) -> &'static str {
                match self {
                    $(Kernel::$v => $name),*
                }
            }

            /// The kernel's stated tolerance (see `tol`).
            pub fn tol(self) -> f64 {
                match self {
                    $(Kernel::$v => $tol),*
                }
            }

            fn push_size(self) -> u32 {
                match self {
                    $(Kernel::$v => std::mem::size_of::<$push>() as u32),*
                }
            }

            fn spirv(self) -> &'static [u8] {
                match self {
                    $(Kernel::$v => include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))),*
                }
            }
        }
    };
}

kernels! {
    EmbedRow => "embed_row", EmbedRowPush, tol::ELEMENTWISE;
    Add => "add", AddPush, tol::ELEMENTWISE;
    Rmsnorm => "rmsnorm", RmsnormPush, tol::REDUCTION;
    MatvecF32 => "matvec_f32", MatvecPush, tol::REDUCTION;
    MatvecBf16 => "matvec_bf16", MatvecPush, tol::REDUCTION;
    MatvecQ8_0 => "matvec_q8_0", MatvecPush, tol::REDUCTION;
    MatvecQ4K => "matvec_q4_k", MatvecPush, tol::REDUCTION;
    MatvecQ5K => "matvec_q5_k", MatvecPush, tol::REDUCTION;
    MatvecQ6K => "matvec_q6_k", MatvecPush, tol::REDUCTION;
    ConvStep => "conv_step", ConvStepPush, tol::REDUCTION;
    GdnGates => "gdn_gates", GdnGatesPush, tol::ELEMENTWISE;
    L2norm => "l2norm", L2normPush, tol::REDUCTION;
    GdnStep => "gdn_step", GdnStepPush, tol::REDUCTION;
    GatedRmsnorm => "gated_rmsnorm", GatedRmsnormPush, tol::REDUCTION;
    QkNormRope => "qk_norm_rope", QkNormRopePush, tol::REDUCTION;
    KvWrite => "kv_write", KvWritePush, tol::ELEMENTWISE;
    AttnDecode => "attn_decode", AttnDecodePush, tol::REDUCTION;
    AttnCombine => "attn_combine", AttnCombinePush, tol::REDUCTION;
    SigmoidMul => "sigmoid_mul", SigmoidMulPush, tol::ELEMENTWISE;
    RouterTopk => "router_topk", RouterTopkPush, tol::REDUCTION;
    Swiglu => "swiglu", SwigluPush, tol::ELEMENTWISE;
    ExpertsGateUp => "experts_gate_up", ExpertsGateUpPush, tol::REDUCTION;
    ExpertsDown => "experts_down", ExpertsDownPush, tol::REDUCTION;
    MoeCombine => "moe_combine", MoeCombinePush, tol::REDUCTION;
    ApplySlotUpdates => "apply_slot_updates", ApplySlotUpdatesPush, tol::ELEMENTWISE;
    Stamp => "stamp", StampPush, tol::ELEMENTWISE;
    MissPublish => "miss_publish", MissPublishPush, tol::ELEMENTWISE;
    MissWait => "miss_wait", MissWaitPush, tol::ELEMENTWISE;
    MatmulF32 => "matmul_f32", MatmulPush, tol::REDUCTION;
    MatmulBf16 => "matmul_bf16", MatmulPush, tol::REDUCTION;
    MatmulQ8_0 => "matmul_q8_0", MatmulPush, tol::REDUCTION;
    MatmulQ4K => "matmul_q4_k", MatmulPush, tol::REDUCTION;
    MatmulQ5K => "matmul_q5_k", MatmulPush, tol::REDUCTION;
    MatmulQ6K => "matmul_q6_k", MatmulPush, tol::REDUCTION;
    EmbedRows => "embed_rows", EmbedRowsPush, tol::ELEMENTWISE;
    ConvSeq => "conv_seq", ConvSeqPush, tol::REDUCTION;
    GdnSeq => "gdn_seq", GdnSeqPush, tol::REDUCTION;
    AttnPrefill => "attn_prefill", AttnPrefillPush, tol::REDUCTION;
    MoeGroup => "moe_group", MoeGroupPush, tol::ELEMENTWISE;
    ExpertsMatmulGateUp => "experts_matmul_gate_up", ExpertsMatmulGateUpPush, tol::REDUCTION;
    ExpertsMatmulDown => "experts_matmul_down", ExpertsMatmulDownPush, tol::REDUCTION;
    MatmulCmF32 => "matmul_cm_f32", MatmulPush, tol::REDUCTION;
    MatmulCmBf16 => "matmul_cm_bf16", MatmulPush, tol::REDUCTION;
    MatmulCmQ8_0 => "matmul_cm_q8_0", MatmulPush, tol::REDUCTION;
    MatmulCmQ4K => "matmul_cm_q4_k", MatmulPush, tol::REDUCTION;
    MatmulCmQ5K => "matmul_cm_q5_k", MatmulPush, tol::REDUCTION;
    MatmulCmQ6K => "matmul_cm_q6_k", MatmulPush, tol::REDUCTION;
    ExpertsMatmulCmGateUp => "experts_matmul_cm_gate_up", ExpertsMatmulGateUpPush, tol::REDUCTION;
    ExpertsMatmulCmDown => "experts_matmul_cm_down", ExpertsMatmulDownPush, tol::REDUCTION;
    AttnDecodeGqa => "attn_decode_gqa", AttnDecodePush, tol::REDUCTION;
    AttnPrefillCm => "attn_prefill_cm", AttnPrefillCmPush, tol::REDUCTION;
    MatvecTokensF32 => "matvec_tokens_f32", MatvecTokensPush, tol::REDUCTION;
    MatvecTokensBf16 => "matvec_tokens_bf16", MatvecTokensPush, tol::REDUCTION;
    MatvecTokensQ8_0 => "matvec_tokens_q8_0", MatvecTokensPush, tol::REDUCTION;
    MatvecTokensQ4K => "matvec_tokens_q4_k", MatvecTokensPush, tol::REDUCTION;
    MatvecTokensQ5K => "matvec_tokens_q5_k", MatvecTokensPush, tol::REDUCTION;
    MatvecTokensQ6K => "matvec_tokens_q6_k", MatvecTokensPush, tol::REDUCTION;
    AttnDecodeTokens => "attn_decode_tokens", AttnTokensPush, tol::REDUCTION;
    AttnDecodeGqaTokens => "attn_decode_gqa_tokens", AttnTokensPush, tol::REDUCTION;
    AttnCombineTokens => "attn_combine_tokens", AttnCombineTokensPush, tol::REDUCTION;
    MoeUnique => "moe_unique", MoeUniquePush, tol::ELEMENTWISE;
    ExpertsGateUpTokens => "experts_gate_up_tokens", ExpertsGateUpTokensPush, tol::REDUCTION;
    ExpertsDownTokens => "experts_down_tokens", ExpertsDownTokensPush, tol::REDUCTION;
    ArgmaxRows => "argmax_rows", ArgmaxRowsPush, tol::ELEMENTWISE;
}

impl Kernel {
    /// The matvec kernel for weights of type `t`.
    pub fn matvec(t: WType) -> Kernel {
        match t {
            WType::F32 => Kernel::MatvecF32,
            WType::Bf16 => Kernel::MatvecBf16,
            WType::Q8_0 => Kernel::MatvecQ8_0,
            WType::Q4K => Kernel::MatvecQ4K,
            WType::Q5K => Kernel::MatvecQ5K,
            WType::Q6K => Kernel::MatvecQ6K,
        }
    }

    /// The `matvec_tokens` kernel for weights of type `t`.
    pub fn matvec_tokens(t: WType) -> Kernel {
        match t {
            WType::F32 => Kernel::MatvecTokensF32,
            WType::Bf16 => Kernel::MatvecTokensBf16,
            WType::Q8_0 => Kernel::MatvecTokensQ8_0,
            WType::Q4K => Kernel::MatvecTokensQ4K,
            WType::Q5K => Kernel::MatvecTokensQ5K,
            WType::Q6K => Kernel::MatvecTokensQ6K,
        }
    }

    /// The matmul kernel for weights of type `t`.
    pub fn matmul(t: WType) -> Kernel {
        match t {
            WType::F32 => Kernel::MatmulF32,
            WType::Bf16 => Kernel::MatmulBf16,
            WType::Q8_0 => Kernel::MatmulQ8_0,
            WType::Q4K => Kernel::MatmulQ4K,
            WType::Q5K => Kernel::MatmulQ5K,
            WType::Q6K => Kernel::MatmulQ6K,
        }
    }

    /// The split-f16 cooperative-matrix matmul kernel for weights of type `t`.
    pub fn matmul_cm(t: WType) -> Kernel {
        match t {
            WType::F32 => Kernel::MatmulCmF32,
            WType::Bf16 => Kernel::MatmulCmBf16,
            WType::Q8_0 => Kernel::MatmulCmQ8_0,
            WType::Q4K => Kernel::MatmulCmQ4K,
            WType::Q5K => Kernel::MatmulCmQ5K,
            WType::Q6K => Kernel::MatmulCmQ6K,
        }
    }

    /// Whether the kernel runs on cooperative matrices, built only where the device has them (`Kernels::coop`).
    pub fn is_coop(self) -> bool {
        matches!(
            self,
            Kernel::MatmulCmF32
                | Kernel::MatmulCmBf16
                | Kernel::MatmulCmQ8_0
                | Kernel::MatmulCmQ4K
                | Kernel::MatmulCmQ5K
                | Kernel::MatmulCmQ6K
                | Kernel::ExpertsMatmulCmGateUp
                | Kernel::ExpertsMatmulCmDown
                | Kernel::AttnPrefillCm
        )
    }

    /// Whether the kernel reads the device clock (`clockRealtimeEXT`), built only where the device enabled
    /// `VK_KHR_shader_clock`'s device clock (`vk::Device::shader_clock`): the CPU miss path's calibration and wait.
    pub fn needs_clock(self) -> bool {
        matches!(self, Kernel::Stamp | Kernel::MissWait)
    }
}

/// Every kernel's pipeline, built once. The cooperative-matrix kernels are built only where the device lists the f16
/// shape (`vk::Device::coop_f16`) and allows the shared memory they declare (`coop_fits`); without them prefill runs
/// the f32 kernels.
pub struct Kernels {
    pipes: Vec<Option<vk::Pipeline>>,
    coop: bool,
    coop_error: Option<String>,
}

impl Kernels {
    /// Builds every kernel's pipeline. Refuses a device that allows a workgroup less shared memory than one of the f32
    /// kernels declares (`vk::Error::SharedMemory`, naming the kernel): `experts_matmul_gate_up` declares the most of
    /// those, 24,576 bytes, where Vulkan guarantees only 16,384.
    pub fn new(dev: &vk::Device) -> Result<Kernels, vk::Error> {
        let limit = dev.max_compute_shared_memory();
        check_shared_memory(limit)?;
        let b = build_pipes(
            Kernel::ALL,
            dev.coop_f16() && coop_fits(limit),
            dev.shader_clock,
            |k| dev.pipeline(k.name(), k.spirv(), k.push_size()),
        )?;
        Ok(Kernels {
            pipes: b.pipes,
            coop: b.coop,
            coop_error: b.coop_error,
        })
    }

    /// Whether the cooperative-matrix kernels were built, so prefill can run them.
    pub fn coop(&self) -> bool {
        self.coop
    }

    /// Why the cooperative-matrix kernels aren't running on a device that lists their shape: the error building one of
    /// them. None when they run, or when the device lacks the shape or the shared memory.
    pub fn coop_error(&self) -> Option<&str> {
        self.coop_error.as_deref()
    }

    pub(crate) fn pipe(&self, k: Kernel) -> &vk::Pipeline {
        self.pipes[k as usize].as_ref().unwrap_or_else(|| {
            if k.needs_clock() {
                panic!(
                    "{}: this device has no shader device clock (vk::Device::shader_clock)",
                    k.name()
                )
            }
            panic!(
                "{}: this device runs no cooperative-matrix kernels (Kernels::coop)",
                k.name()
            )
        })
    }
}

/// What `build_pipes` built: `pipes[i]` is `kernels[i]`'s pipeline, or None for a cooperative-matrix kernel that wasn't
/// built; `coop` whether the cooperative-matrix kernels all were; `coop_error` why not, when one failed.
struct Built<T> {
    pipes: Vec<Option<T>>,
    coop: bool,
    coop_error: Option<String>,
}

/// `Kernels::new`'s choices, over any pipeline type so CI can test them without a device. Every f32 kernel is built,
/// and one that fails refuses the device (the error is returned). The cooperative-matrix kernels are built only when
/// `coop` (the device has their shape and shared memory). One that fails to build leaves prefill on the f32 kernels
/// rather than refusing the device: every cooperative-matrix entry stays None and the error is
/// kept.
fn build_pipes<T>(
    kernels: &[Kernel],
    coop: bool,
    clock: bool,
    mut build: impl FnMut(Kernel) -> Result<T, vk::Error>,
) -> Result<Built<T>, vk::Error> {
    let mut pipes: Vec<Option<T>> = kernels
        .iter()
        .map(|&k| {
            if k.is_coop() || (k.needs_clock() && !clock) {
                Ok(None)
            } else {
                build(k).map(Some)
            }
        })
        .collect::<Result<_, _>>()?;
    if !coop {
        return Ok(Built {
            pipes,
            coop: false,
            coop_error: None,
        });
    }
    let built: Result<Vec<(usize, T)>, vk::Error> = kernels
        .iter()
        .enumerate()
        .filter(|(_, k)| k.is_coop())
        .map(|(i, &k)| build(k).map(|p| (i, p)))
        .collect();
    Ok(match built {
        Ok(b) => {
            for (i, p) in b {
                pipes[i] = Some(p);
            }
            Built {
                pipes,
                coop: true,
                coop_error: None,
            }
        }
        Err(e) => Built {
            pipes,
            coop: false,
            coop_error: Some(e.to_string()),
        },
    })
}

/// Refuses a device whose per-workgroup shared memory `limit` is below what any kernel declares, naming the kernel
/// that needs the most.
pub fn check_shared_memory(limit: u32) -> Result<(), vk::Error> {
    match BIG_SHARED_KERNELS.iter().max_by_key(|k| k.1) {
        Some(&(kernel, need)) if limit < need => Err(vk::Error::SharedMemory {
            kernel,
            need,
            limit,
        }),
        _ => Ok(()),
    }
}

/// A weight type's code in the shaders (`quant.glsl`).
pub fn wtype_code(t: WType) -> u32 {
    match t {
        WType::F32 => 0,
        WType::Bf16 => 1,
        WType::Q8_0 => 2,
        WType::Q4K => 3,
        WType::Q5K => 4,
        WType::Q6K => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_memory_constants_are_pinned_and_the_maximum_is_gate_up() {
        assert_eq!(EXPERTS_GATE_UP_SHARED_BYTES, 24_576);
        assert_eq!(ATTN_SHARED_BYTES, 19_136);
        assert_eq!(MOE_GROUP_SHARED_BYTES, 18_432);
        // attn_decode_gqa's and attn_decode_gqa_tokens's qw (8 heads × 256 floats) and K tile (32 × 128 f16) stay
        // within Vulkan's 16,384, so they need no entry here.
        assert_eq!(ATTN_GQA_SHARED_BYTES, 16_384);
        const { assert!(ATTN_GQA_SHARED_BYTES <= 16_384) };
        let max = BIG_SHARED_KERNELS.iter().max_by_key(|k| k.1).unwrap();
        assert_eq!(*max, ("experts_matmul_gate_up", 24_576));
    }

    #[test]
    fn a_device_short_of_shared_memory_is_refused_naming_the_kernel() {
        assert!(check_shared_memory(32_768).is_ok());
        assert!(check_shared_memory(24_576).is_ok());
        match check_shared_memory(20_000) {
            Err(vk::Error::SharedMemory {
                kernel,
                need,
                limit,
            }) => {
                assert_eq!(
                    (kernel, need, limit),
                    ("experts_matmul_gate_up", 24_576, 20_000)
                );
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        // A 16 KiB device is told the largest need, not the first kernel's.
        match check_shared_memory(16_384) {
            Err(vk::Error::SharedMemory { kernel, need, .. }) => {
                assert_eq!((kernel, need), ("experts_matmul_gate_up", 24_576));
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn the_cooperative_matrix_kernels_need_their_shared_memory() {
        assert_eq!(MATMUL_CM_SHARED_BYTES, 20_480);
        assert_eq!(EXPERTS_CM_GATE_UP_SHARED_BYTES, 30_720);
        assert_eq!(EXPERTS_CM_DOWN_SHARED_BYTES, 20_480);
        assert_eq!(ATTN_CM_SHARED_BYTES, 29_952);
        let max = COOP_SHARED_KERNELS.iter().max_by_key(|k| k.1).unwrap();
        assert_eq!(*max, ("experts_matmul_cm_gate_up", 30_720));
        // The cap: no cooperative-matrix kernel declares more than 30,720 bytes, below the RX 9070 XT's 32,768.
        assert!(
            max.1 <= 30_720,
            "a cooperative-matrix kernel declares {} bytes",
            max.1
        );
        assert!(coop_fits(32_768) && coop_fits(30_720));
        assert!(
            !coop_fits(30_719),
            "a device short of the gate/up kernel's shared memory runs the f32 kernels"
        );
        assert_eq!(Kernel::ALL.iter().filter(|k| k.is_coop()).count(), 9);
        for t in WType::ALL {
            assert!(Kernel::matmul_cm(t).is_coop() && !Kernel::matmul(t).is_coop());
            assert_eq!(Kernel::matmul_cm(t).tol(), tol::REDUCTION);
        }
        assert!(Kernel::ExpertsMatmulCmGateUp.is_coop() && Kernel::ExpertsMatmulCmDown.is_coop());
        assert!(!Kernel::ExpertsMatmulGateUp.is_coop() && !Kernel::ExpertsMatmulDown.is_coop());
        assert!(Kernel::AttnPrefillCm.is_coop() && !Kernel::AttnPrefill.is_coop());
        assert_eq!(Kernel::AttnPrefillCm.tol(), tol::REDUCTION);
        assert_eq!(Kernel::AttnPrefillCm.push_size(), 64);
    }

    /// `build_pipes` over a fake pipeline (the kernel itself), with `fail` naming the kernels whose build fails. Also
    /// returns the kernels it tried to build, in order.
    fn fake_build(coop: bool, fail: &[Kernel]) -> (Result<Built<Kernel>, vk::Error>, Vec<Kernel>) {
        fake_build_clock(coop, true, fail)
    }

    fn fake_build_clock(
        coop: bool,
        clock: bool,
        fail: &[Kernel],
    ) -> (Result<Built<Kernel>, vk::Error>, Vec<Kernel>) {
        let mut tried = Vec::new();
        let b = build_pipes(Kernel::ALL, coop, clock, |k| {
            tried.push(k);
            if fail.contains(&k) {
                Err(vk::Error::BadSpirv(k.name()))
            } else {
                Ok(k)
            }
        });
        (b, tried)
    }

    #[test]
    fn the_clock_kernels_are_built_only_where_the_device_has_the_clock() {
        // Creating a pipeline whose SPIR-V declares ShaderClockKHR on a device without shaderDeviceClock is invalid
        // usage: without the clock neither stamp nor miss_wait is tried, and every other kernel still is.
        assert_eq!(Kernel::ALL.iter().filter(|k| k.needs_clock()).count(), 2);
        let (b, tried) = fake_build_clock(false, false, &[]);
        let b = b.unwrap();
        assert!(!tried.contains(&Kernel::Stamp) && !tried.contains(&Kernel::MissWait));
        assert!(tried.contains(&Kernel::MissPublish) && tried.contains(&Kernel::ExpertsGateUp));
        for (i, &k) in Kernel::ALL.iter().enumerate() {
            assert_eq!(
                b.pipes[i].is_some(),
                !k.is_coop() && !k.needs_clock(),
                "{}",
                k.name()
            );
        }
        let (b, tried) = fake_build_clock(false, true, &[]);
        assert!(tried.contains(&Kernel::Stamp) && tried.contains(&Kernel::MissWait));
        assert!(b.unwrap().pipes[Kernel::MissWait as usize].is_some());
    }

    #[test]
    fn kernels_new_builds_the_cooperative_matrix_kernels_only_where_it_can() {
        // Kernels::pipe indexes by `k as usize`, so Kernel::ALL must list the kernels in that order.
        assert!(Kernel::ALL
            .iter()
            .enumerate()
            .all(|(i, &k)| k as usize == i));

        // Without the shape or the shared memory: the f32 kernels, and no cooperative-matrix kernel is even tried.
        let (b, tried) = fake_build(false, &[]);
        let b = b.unwrap();
        assert!(!b.coop && b.coop_error.is_none());
        assert!(tried.iter().all(|k| !k.is_coop()));
        for (i, &k) in Kernel::ALL.iter().enumerate() {
            assert_eq!(b.pipes[i], (!k.is_coop()).then_some(k), "{}", k.name());
        }

        // With them, and every build succeeding: every kernel.
        let (b, _) = fake_build(true, &[]);
        let b = b.unwrap();
        assert!(b.coop && b.coop_error.is_none());
        for (i, &k) in Kernel::ALL.iter().enumerate() {
            assert_eq!(b.pipes[i], Some(k), "{}", k.name());
        }

        // One cooperative-matrix kernel failing: none of them runs, the device isn't refused, the f32 kernels stay,
        // and the error says which kernel failed.
        let (b, _) = fake_build(true, &[Kernel::MatmulCmQ4K]);
        let b = b.unwrap();
        assert!(!b.coop);
        let e = b.coop_error.expect("the failure is kept");
        assert!(e.contains("matmul_cm_q4_k"), "{e}");
        for (i, &k) in Kernel::ALL.iter().enumerate() {
            assert_eq!(b.pipes[i], (!k.is_coop()).then_some(k), "{}", k.name());
        }

        // The attention kernel is a member of the family: its failing turns the family off, and says which kernel.
        let (b, _) = fake_build(true, &[Kernel::AttnPrefillCm]);
        let b = b.unwrap();
        assert!(!b.coop);
        let e = b.coop_error.expect("the failure is kept");
        assert!(e.contains("attn_prefill_cm"), "{e}");
        for (i, &k) in Kernel::ALL.iter().enumerate() {
            assert_eq!(b.pipes[i], (!k.is_coop()).then_some(k), "{}", k.name());
        }

        // An f32 kernel failing refuses the device, coop or not.
        for coop in [false, true] {
            match fake_build(coop, &[Kernel::Rmsnorm]).0 {
                Err(vk::Error::BadSpirv(name)) => assert_eq!(name, "rmsnorm"),
                Err(e) => panic!("the wrong error: {e}"),
                Ok(_) => panic!("an f32 kernel that fails to build must refuse the device"),
            }
        }
    }
}
