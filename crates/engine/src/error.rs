use crate::request::ErrorKind;
use crate::swap::SwapError;
use std::fmt;

#[derive(Debug)]
pub enum Error {
    Gguf(gguf::Error),
    Contract(qwen36::contract::ContractError),
    Vk(vk::Error),
    /// The model doesn't fit in what the driver's budget leaves free.
    DoesNotFit {
        what: &'static str,
        need: u64,
        free: u64,
    },
    TokenOutOfRange {
        token: u32,
        n_vocab: usize,
    },
    /// A session was given an engine other than the one that made it.
    WrongEngine,
    /// The session has used all `ctx` positions.
    ContextFull {
        ctx: usize,
    },
    /// A session must have at least one position.
    BadCtx {
        ctx: usize,
    },
    /// A prefill of no tokens.
    EmptyPrefill,
    /// A prefill that would pass the session's context; nothing ran.
    PrefillTooLong {
        pos: usize,
        tokens: usize,
        ctx: usize,
    },
    /// `Options::prefill_chunk` outside 1 ..= `max` (`PREFILL_CHUNK`); refused before the model loads.
    BadPrefillChunk {
        chunk: usize,
        max: usize,
    },
    /// `Options::upload_cap` outside 1 ..= `max` (`RESHAPE_BYTES`) bytes a token; refused before the model loads.
    BadUploadCap {
        cap: u64,
        max: u64,
    },
    /// `Options::cpu_share` outside `0.0..=1.0`, or NaN; refused before the model loads.
    BadCpuShare {
        share: f32,
    },
    /// `Options::base_time` with a non-zero `cpu_share`: both are measurement modes; refused before the model loads.
    BaseTimeAndCpuShare,
    /// A non-zero `cpu_share` on a machine that lacks `what`, which the CPU miss path needs; refused before the model
    /// loads.
    CpuPathUnavailable {
        what: &'static str,
    },
    /// A non-zero `cpu_share` on a model whose experts include a matrix type the CPU worker has no kernel for (only
    /// Q4_K, Q5_K, Q6_K and Q8_0 have one); `ty` names the first such type. Refused before the model loads.
    CpuShareExpertType {
        ty: &'static str,
    },
    /// `Options::speculation` with a setting speculation can't run beside (the CPU miss path, the base time); refused
    /// before the model loads.
    SpeculationRefused {
        why: &'static str,
    },
    /// A verify of no tokens, or of more than `most` (the lesser of `MAX_VERIFY` and the prefill chunk the verify's
    /// buffers are sized by); nothing ran.
    VerifyShape {
        tokens: usize,
        most: usize,
    },
    /// A verify in a session of `ctx` positions, whose step needs `need` bytes of the prefill work area, which holds
    /// `have`: the engine was opened (`Options::ctx`) for a smaller context than this session's. Nothing ran.
    VerifyBuffers {
        ctx: usize,
        need: u64,
        have: u64,
    },
    /// An MTP draft of `k` tokens, where a draft holds 1 to `most` (the lesser of `MAX_VERIFY - 1` and the positions
    /// the context has left after `x`); nothing ran.
    MtpShape {
        k: usize,
        most: usize,
    },
    /// Routing reached an expert whose slot-table entry is `CPU`, which a step without the CPU miss path refuses.
    ExpertOnCpu {
        pos: usize,
        layer: u32,
        expert: u32,
    },
    /// Routing reached an expert whose slot-table entry has an unknown `where` or no address: a corrupt table.
    BadSlotEntry {
        pos: usize,
        layer: u32,
        expert: u32,
    },
    /// `ExpertOnCpu` found during a prefill chunk of `tokens` positions starting at `pos`; which position routed
    /// there isn't known.
    PrefillExpertOnCpu {
        pos: usize,
        tokens: usize,
        layer: u32,
        expert: u32,
    },
    /// `BadSlotEntry` found during a prefill chunk of `tokens` positions starting at `pos`.
    PrefillBadSlotEntry {
        pos: usize,
        tokens: usize,
        layer: u32,
        expert: u32,
    },
    /// NaN or Inf in the logits.
    NonFinite {
        pos: usize,
        index: usize,
        value: f32,
    },
    /// A tap's forced value has the wrong length.
    BadForce {
        tap: String,
        want: usize,
        got: usize,
    },
    /// A forced routed-expert id that isn't a whole number below `n_expert`: the expert kernels would index past the
    /// slot table.
    BadForcedExpert {
        tap: String,
        value: f32,
        n_expert: usize,
    },
    /// A swap the cache can't make.
    Swap(SwapError),
    /// `--force-swap-every-token` with no expert to send out of the cache, or none to bring in.
    NoRoomToSwap {
        slots: usize,
        n_expert: usize,
    },
    /// A GPU failure during the step at `pos`, such as device loss; `layer` names the prefill layer whose submit or
    /// staging copy failed.
    Step {
        pos: usize,
        layer: Option<usize>,
        source: vk::Error,
    },
    /// A slot-table entry that doesn't exist.
    NoSuchEntry {
        layer: usize,
        expert: usize,
    },
    /// A routing profile of the wrong size: it holds a u64 count for every layer and expert.
    BadProfile {
        want: usize,
        got: usize,
    },
    /// A cache slot that doesn't exist: the cache holds `slots` per layer now, its rows with memory behind them.
    NoSuchSlot {
        layer: usize,
        slot: usize,
        slots: usize,
    },
    /// A cache slot in a dead row: its chunk retired and drained, and is freed or about to be,
    /// or a grow the budget refused gave it back. It holds no expert.
    DeadSlot {
        layer: usize,
        slot: usize,
    },
    /// A restore named a snapshot the pool no longer holds, or one that holds no state.
    NoSuchSnapshot {
        id: u32,
    },
}

/// How `Error::kind` sorts an error: what the worker retries, what it reports, and as what.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorClass {
    /// The GPU was lost; the worker recovers it (reloads).
    DeviceLost,
    /// NaN or Inf in the logits.
    NonFinite,
    /// Memory ran out, in VRAM or on the host.
    OutOfMemory,
    /// The request was invalid. The server validates first, so one that reaches the worker is a server bug.
    BadRequest,
    /// Anything else.
    Internal,
}

impl ErrorClass {
    /// The kind an `Event::Error` carries. `BadRequest` becomes `Internal`: the server validates first, so a bad
    /// request reaching the worker is the server's bug, not the client's.
    pub fn event_kind(self) -> ErrorKind {
        match self {
            ErrorClass::DeviceLost => ErrorKind::DeviceLost,
            ErrorClass::NonFinite => ErrorKind::NonFinite,
            ErrorClass::OutOfMemory => ErrorKind::OutOfMemory,
            ErrorClass::BadRequest | ErrorClass::Internal => ErrorKind::Internal,
        }
    }
}

fn class_of_vk(e: &vk::Error) -> ErrorClass {
    if matches!(e, vk::Error::DeviceLost { .. }) {
        ErrorClass::DeviceLost
    } else if e.is_out_of_memory() {
        ErrorClass::OutOfMemory
    } else {
        ErrorClass::Internal
    }
}

impl Error {
    /// Sorts this error for the worker.
    pub fn kind(&self) -> ErrorClass {
        match self {
            Error::Vk(e) | Error::Step { source: e, .. } => class_of_vk(e),
            Error::NonFinite { .. } => ErrorClass::NonFinite,
            Error::DoesNotFit { .. } => ErrorClass::OutOfMemory,
            Error::PrefillTooLong { .. }
            | Error::ContextFull { .. }
            | Error::TokenOutOfRange { .. }
            | Error::EmptyPrefill => ErrorClass::BadRequest,
            // The step loop's verify is never a client's: a shape it can't take is the engine's limit, not a bad
            // request (the loop drafts within `Verifier::most`, so it is not reached short of a bug).
            Error::VerifyShape { .. } => ErrorClass::Internal,
            _ => ErrorClass::Internal,
        }
    }

    /// Whether this is host memory running out: `DoesNotFit` for host memory (the two messages
    /// `weights::check_fits` builds for it end in "in host memory"), or, during the load, a host heap's budget
    /// refusing an allocation or the driver reporting `ERROR_OUT_OF_HOST_MEMORY` (`vk::Error::is_host_memory`). The
    /// worker gives it its own message and does not retry with a smaller cache: the cache lives in VRAM.
    pub fn is_host_memory(&self) -> bool {
        match self {
            Error::DoesNotFit { what, .. } => what.ends_with("in host memory"),
            Error::Vk(e) => e.is_host_memory(),
            _ => false,
        }
    }
}

const MIB: u64 = 1 << 20;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Gguf(e) => write!(f, "{e}"),
            Error::Contract(e) => write!(f, "the model file doesn't match Qwen3.6-35B-A3B: {e}"),
            Error::Vk(e) => write!(f, "{e}"),
            Error::DoesNotFit { what, need, free } => write!(
                f,
                "{what} need {} MiB, but only {} MiB is free",
                need.div_ceil(MIB),
                free / MIB
            ),
            Error::TokenOutOfRange { token, n_vocab } => {
                write!(f, "token {token} is outside the vocabulary ({n_vocab} tokens)")
            }
            Error::ContextFull { ctx } => {
                write!(f, "the context is full: all {ctx} positions are used (--ctx)")
            }
            Error::WrongEngine => write!(
                f,
                "this session belongs to another engine: its state lives in that engine's device memory"
            ),
            Error::BadCtx { ctx } => {
                write!(f, "a session needs at least one position, not --ctx {ctx}")
            }
            Error::EmptyPrefill => write!(f, "a prefill needs at least one token"),
            Error::BadPrefillChunk { chunk, max } => write!(
                f,
                "a prefill chunk of {chunk} tokens: a chunk holds 1 to {max} tokens, which its buffers and kernels \
                 are sized for"
            ),
            Error::BadUploadCap { cap, max } => write!(
                f,
                "an upload cap of {cap} bytes a token: a cap holds 1 to {max} bytes (a step's copies are one \
                 submit, bounded at {max} bytes)"
            ),
            Error::BadCpuShare { share } => write!(f, "cpu_share {share} is outside 0.0..=1.0"),
            Error::BaseTimeAndCpuShare => write!(
                f,
                "the base time and the CPU path are both measurement modes; pick one"
            ),
            Error::CpuPathUnavailable { what } => write!(f, "the CPU miss path needs {what}"),
            Error::CpuShareExpertType { ty } => write!(
                f,
                "--cpu-share needs quantized expert matrices (Q4_K, Q5_K, Q6_K or Q8_0); this model's experts include {ty}"
            ),
            Error::SpeculationRefused { why } => write!(f, "{why}"),
            Error::VerifyShape { tokens, most } => write!(
                f,
                "a verify of {tokens} tokens: a verify holds 1 to {most} tokens"
            ),
            Error::VerifyBuffers { ctx, need, have } => write!(
                f,
                "a verify step in a context of {ctx} positions needs {need} bytes of the prefill work area, which holds {have}: open the engine for the session's context (--ctx)"
            ),
            Error::MtpShape { k, most } => write!(
                f,
                "an MTP draft of {k} tokens: a draft holds 1 to {most} tokens here"
            ),
            Error::PrefillTooLong { pos, tokens, ctx } => write!(
                f,
                "{tokens} tokens from position {pos} pass the context of {ctx} positions (--ctx)"
            ),
            Error::ExpertOnCpu { pos, layer, expert } => write!(
                f,
                "position {pos}: layer {layer} routed to expert {expert}, whose slot-table entry is CPU, \
                 and the CPU miss path is off (only decode runs it, and only with a non-zero --cpu-share)"
            ),
            Error::BadSlotEntry { pos, layer, expert } => write!(
                f,
                "position {pos}: layer {layer} routed to expert {expert}, whose slot-table entry is corrupt (an unknown location or no address)"
            ),
            Error::PrefillExpertOnCpu {
                pos,
                tokens,
                layer,
                expert,
            } => write!(
                f,
                "the prefill chunk at positions {pos}..{}: layer {layer} routed to expert {expert}, whose slot-table entry is CPU; prefill has no CPU miss path",
                pos + tokens
            ),
            Error::PrefillBadSlotEntry {
                pos,
                tokens,
                layer,
                expert,
            } => write!(
                f,
                "the prefill chunk at positions {pos}..{}: layer {layer} routed to expert {expert}, whose slot-table entry is corrupt (an unknown location or no address)",
                pos + tokens
            ),
            Error::BadForcedExpert {
                tap,
                value,
                n_expert,
            } => write!(
                f,
                "the forced value of {tap} holds {value}, not an expert id below {n_expert}"
            ),
            Error::Swap(e) => write!(f, "{e}"),
            Error::NoRoomToSwap { slots, n_expert } => write!(
                f,
                "--force-swap-every-token needs some experts in the cache and some out of it; the cache holds \
                 {slots} of each layer's {n_expert}; give --cache-slots from 1 to {}, or free VRAM",
                n_expert - 1
            ),
            Error::NonFinite { pos, index, value } => write!(
                f,
                "position {pos}: logit {index} is {value}; the run stops here"
            ),
            Error::BadForce { tap, want, got } => {
                write!(f, "the forced value for {tap} has {got} values; the tap has {want}")
            }
            Error::Step {
                pos,
                layer: None,
                source,
            } => write!(f, "position {pos}: {source}"),
            Error::Step {
                pos,
                layer: Some(l),
                source,
            } => write!(f, "position {pos}, layer {l}: {source}"),
            Error::NoSuchEntry { layer, expert } => write!(
                f,
                "the slot table has no entry for expert {expert} of layer {layer}"
            ),
            Error::BadProfile { want, got } => write!(
                f,
                "the expert profile holds {got} bytes, not the {want} of a u64 count for every layer and expert"
            ),
            Error::NoSuchSlot { layer, slot, slots } => write!(
                f,
                "layer {layer} has no cache slot {slot}: the cache has {slots} slots per main layer"
            ),
            Error::DeadSlot { layer, slot } => write!(
                f,
                "slot {slot} of layer {layer} is in a dead row of the cache: its chunk was retired, or a grow of it \
                 refused"
            ),
            Error::NoSuchSnapshot { id } => {
                write!(f, "snapshot {id} is gone or holds no state")
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<gguf::Error> for Error {
    fn from(e: gguf::Error) -> Error {
        Error::Gguf(e)
    }
}

impl From<qwen36::contract::ContractError> for Error {
    fn from(e: qwen36::contract::ContractError) -> Error {
        Error::Contract(e)
    }
}

impl From<vk::Error> for Error {
    fn from(e: vk::Error) -> Error {
        Error::Vk(e)
    }
}

impl From<SwapError> for Error {
    fn from(e: SwapError) -> Error {
        Error::Swap(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_read_as_one_line() {
        let all = [
            Error::BadSlotEntry {
                pos: 3,
                layer: 1,
                expert: 7,
            },
            Error::ExpertOnCpu {
                pos: 3,
                layer: 1,
                expert: 7,
            },
            Error::PrefillBadSlotEntry {
                pos: 3,
                tokens: 5,
                layer: 1,
                expert: 7,
            },
            Error::PrefillExpertOnCpu {
                pos: 3,
                tokens: 5,
                layer: 1,
                expert: 7,
            },
            Error::BadForcedExpert {
                tap: "l0.topk".into(),
                value: 300.0,
                n_expert: 256,
            },
            Error::Swap(crate::swap::SwapError::Busy {
                layer: 3,
                expert: 17,
            }),
            Error::NoRoomToSwap {
                slots: 0,
                n_expert: 256,
            },
            Error::WrongEngine,
            Error::BadCtx { ctx: 0 },
            Error::ContextFull { ctx: 8 },
            Error::EmptyPrefill,
            Error::PrefillTooLong {
                pos: 3,
                tokens: 10,
                ctx: 8,
            },
            Error::BadPrefillChunk {
                chunk: 4096,
                max: 2048,
            },
            Error::BadUploadCap {
                cap: 0,
                max: 2 << 30,
            },
            Error::BadCpuShare { share: 1.5 },
            Error::BaseTimeAndCpuShare,
            Error::CpuPathUnavailable { what: "AVX-512" },
            Error::CpuShareExpertType { ty: "BF16" },
            Error::SpeculationRefused {
                why: "speculation can't run with --base-time",
            },
            Error::VerifyShape {
                tokens: 17,
                most: 16,
            },
            Error::VerifyBuffers {
                ctx: 262_144,
                need: 400 << 20,
                have: 225 << 20,
            },
            Error::MtpShape { k: 16, most: 15 },
            Error::BadProfile {
                want: 81_920,
                got: 100,
            },
            Error::NoSuchSlot {
                layer: 3,
                slot: 9,
                slots: 4,
            },
            Error::DeadSlot { layer: 0, slot: 4 },
            Error::NoSuchEntry {
                layer: 41,
                expert: 0,
            },
            Error::Step {
                pos: 2048,
                layer: Some(17),
                source: vk::Error::Timeout,
            },
            Error::Step {
                pos: 9,
                layer: None,
                source: vk::Error::Timeout,
            },
            Error::NoSuchSnapshot { id: 4 },
        ];
        for e in all {
            let m = e.to_string();
            assert!(!m.contains("  ") && !m.contains('\n'), "{m:?}");
        }
        // A message a user can read names no milestone and no spec section.
        for e in [
            Error::ExpertOnCpu {
                pos: 3,
                layer: 1,
                expert: 7,
            },
            Error::PrefillExpertOnCpu {
                pos: 3,
                tokens: 5,
                layer: 1,
                expert: 7,
            },
            Error::NonFinite {
                pos: 3,
                index: 2,
                value: f32::NAN,
            },
        ] {
            let m = e.to_string();
            assert!(!m.contains("milestone") && !m.contains("spec"), "{m}");
        }
        let m = Error::ExpertOnCpu {
            pos: 3,
            layer: 1,
            expert: 7,
        }
        .to_string();
        assert!(
            m.ends_with(
                "whose slot-table entry is CPU, and the CPU miss path is off (only decode runs it, and only with a non-zero --cpu-share)"
            ),
            "{m}"
        );
        let m = Error::PrefillExpertOnCpu {
            pos: 3,
            tokens: 5,
            layer: 1,
            expert: 7,
        }
        .to_string();
        assert!(
            m.ends_with("whose slot-table entry is CPU; prefill has no CPU miss path"),
            "{m}"
        );
        assert!(
            m.starts_with("the prefill chunk at positions 3..8: "),
            "{m}"
        );
        let m = Error::NonFinite {
            pos: 3,
            index: 2,
            value: f32::NAN,
        }
        .to_string();
        assert!(m.ends_with("the run stops here"), "{m}");
        let m = Error::Step {
            pos: 2048,
            layer: Some(17),
            source: vk::Error::Timeout,
        }
        .to_string();
        assert!(m.starts_with("position 2048, layer 17: "), "{m}");
    }

    #[test]
    fn errors_are_classified_as_the_spec_sorts_them() {
        use vk::Error as V;
        let lost = || V::DeviceLost { in_flight: None };
        let oob = || V::OutOfBudget {
            what: "x".into(),
            want: 1,
            budget: 0,
            usage: 0,
            host: false,
        };
        assert_eq!(Error::Vk(lost()).kind(), ErrorClass::DeviceLost);
        assert_eq!(
            Error::Step {
                pos: 3,
                layer: Some(1),
                source: lost()
            }
            .kind(),
            ErrorClass::DeviceLost
        );
        assert_eq!(
            Error::NonFinite {
                pos: 1,
                index: 2,
                value: f32::NAN
            }
            .kind(),
            ErrorClass::NonFinite
        );
        assert_eq!(Error::Vk(oob()).kind(), ErrorClass::OutOfMemory);
        assert_eq!(
            Error::Step {
                pos: 0,
                layer: None,
                source: oob()
            }
            .kind(),
            ErrorClass::OutOfMemory
        );
        assert_eq!(
            Error::DoesNotFit {
                what: "w",
                need: 2,
                free: 1
            }
            .kind(),
            ErrorClass::OutOfMemory
        );
        for e in [
            Error::PrefillTooLong {
                pos: 0,
                tokens: 9,
                ctx: 8,
            },
            Error::ContextFull { ctx: 8 },
            Error::TokenOutOfRange {
                token: 9,
                n_vocab: 8,
            },
            Error::EmptyPrefill,
        ] {
            assert_eq!(e.kind(), ErrorClass::BadRequest);
        }
        // A verify shape the step loop could not take is the engine's, not the client's.
        for tokens in [0, 17] {
            assert_eq!(
                Error::VerifyShape { tokens, most: 16 }.kind(),
                ErrorClass::Internal
            );
        }
        assert_eq!(Error::Vk(V::Timeout).kind(), ErrorClass::Internal);
        assert_eq!(Error::WrongEngine.kind(), ErrorClass::Internal);
        assert_eq!(Error::NoSuchSnapshot { id: 1 }.kind(), ErrorClass::Internal);
        assert_eq!(ErrorClass::BadRequest.event_kind(), ErrorKind::Internal);
        assert_eq!(ErrorClass::OutOfMemory.event_kind(), ErrorKind::OutOfMemory);
        assert_eq!(ErrorClass::DeviceLost.event_kind(), ErrorKind::DeviceLost);
        assert_eq!(ErrorClass::NonFinite.event_kind(), ErrorKind::NonFinite);
    }

    #[test]
    fn host_memory_is_told_from_vram() {
        use crate::weights::{check_fits, Need};
        let need = Need {
            vram: 2000 << 20,
            host: 19_000 << 20,
            session: 0,
            prefill: 0,
            prefill_host: 0,
        };
        let host = check_fits(&need, 3000 << 20, 10_000 << 20).unwrap_err();
        assert!(host.is_host_memory(), "{host}");
        let with_prefill = Need {
            prefill_host: 66 << 20,
            ..need
        };
        let host = check_fits(&with_prefill, 3000 << 20, 10_000 << 20).unwrap_err();
        assert!(host.is_host_memory(), "{host}");
        let vram = check_fits(&need, 2200 << 20, 40_000 << 20).unwrap_err();
        assert!(!vram.is_host_memory(), "{vram}");
        assert!(!Error::WrongEngine.is_host_memory());
    }

    #[test]
    fn a_host_budget_or_host_driver_refusal_at_load_is_host_memory() {
        use vk::Error as V;
        let budget = |host| {
            Error::Vk(V::OutOfBudget {
                what: "the experts of layer 3".into(),
                want: 500 << 20,
                budget: 1000 << 20,
                usage: 800 << 20,
                host,
            })
        };
        // Host kind: still out of memory for the server, and host memory for the worker.
        assert_eq!(budget(true).kind(), ErrorClass::OutOfMemory);
        assert!(budget(true).is_host_memory());
        // VRAM kind: GPU advice and the half-cache retry stay.
        assert_eq!(budget(false).kind(), ErrorClass::OutOfMemory);
        assert!(!budget(false).is_host_memory());
        // The driver's own codes are told apart in vk's tests; here only that a non-memory error is not host memory.
        assert!(!Error::Vk(V::Timeout).is_host_memory());
    }
}
