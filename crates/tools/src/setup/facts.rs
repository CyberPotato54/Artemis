//! Readings of the GPU and the host that setup checks and prints: whether the card is the target, whether another
//! program holds its memory, and what loading the model needs of the host.
use super::choose::{TARGET_VRAM_BYTES, VRAM_HELD_WARN};
use gguf::Gguf;
use qwen36::dims::Dims;

const GIB: f64 = (1u64 << 30) as f64;

/// Whether a GPU with `vram_bytes` of device memory is the 16 GB target (15 GiB or more).
pub fn is_target(vram_bytes: u64) -> bool {
    vram_bytes >= TARGET_VRAM_BYTES
}

/// A warning when the driver's budget for the largest device-local heap is more than `VRAM_HELD_WARN` under its size,
/// read from `vk::list_gpus` before this process allocates anything: another program holds the difference.
pub fn vram_warning(heap: vk::HeapBudget) -> Option<String> {
    let held = heap.size.saturating_sub(heap.budget);
    (held > VRAM_HELD_WARN).then(|| {
        format!(
            "Another program holds about {:.0} GiB of this GPU's memory. Setup measures with what is left; run \
             artemis setup --force once it is closed.",
            held as f64 / GIB
        )
    })
}

/// The bytes a load pins in host memory: the expert tensors of the 40 blocks and `token_embd`, and `blk.40`'s experts
/// when the MTP block loads, plus prefill's host buffers at the largest chunk. The engine computes the same sum from
/// its loaded model (`engine::weights::need` and `Need::with_prefill`, counted by `check_fits`), which needs a model
/// rather than the file, so this one reads the file's tensor table.
pub fn host_need(g: &Gguf, mtp: bool) -> u64 {
    let d = &Dims::QWEN36;
    let last = d.n_layer + usize::from(mtp);
    let tensors: u64 = g
        .tensors()
        .filter(|t| t.name == "token_embd.weight" || expert_block(t.name).is_some_and(|l| l < last))
        .map(|t| t.bytes.len() as u64)
        .sum();
    tensors + engine::prefill_host_bytes(d, engine::PREFILL_CHUNK)
}

/// The block `name` is the experts of (`blk.7.ffn_up_exps.weight` is 7).
fn expert_block(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let (l, tensor) = rest.split_once('.')?;
    matches!(
        tensor,
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight"
    )
    .then(|| l.parse().ok())
    .flatten()
}

/// The refusal when `available` host bytes are fewer than `need`.
pub fn host_shortfall(available: u64, need: u64) -> Option<String> {
    (available < need).then(|| {
        format!(
            "This PC has {:.1} GiB of memory available and loading the model needs about {:.1} GiB. Close other \
             programs, or stop the Artemis server that is running, and try again.",
            available as f64 / GIB,
            need as f64 / GIB
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gguf::{types::F32, write::Writer};

    const MIB: u64 = 1 << 20;
    const GIB_U: u64 = 1 << 30;

    fn heap(budget: u64) -> vk::HeapBudget {
        vk::HeapBudget {
            size: 16 * GIB_U,
            budget,
            usage: 0,
        }
    }

    #[test]
    fn vram_warning_on_the_9070_xt_figures() {
        // The driver's idle budget on the RX 9070 XT (engine's cache.rs): 15,416 MiB of 16 GiB. No warning.
        assert_eq!(vram_warning(heap(15_416 * MIB)), None);
        // The heap's size as the driver reports it (engine's cache.rs): 16,304 MiB. The same outcomes.
        let real = |budget| vk::HeapBudget {
            size: 16_304 * MIB,
            budget,
            usage: 0,
        };
        assert_eq!(vram_warning(real(15_416 * MIB)), None);
        let w = vram_warning(real(13_000 * MIB)).expect("3.2 GiB held");
        assert!(
            w.starts_with("Another program holds about 3 GiB of this GPU's memory."),
            "{w}"
        );
        let w = vram_warning(heap(13_000 * MIB)).expect("3.3 GiB held");
        assert_eq!(
            w,
            "Another program holds about 3 GiB of this GPU's memory. Setup measures with what is left; run \
             artemis setup --force once it is closed."
        );
        // Exactly the warning line is not over it.
        assert_eq!(vram_warning(heap(14 * GIB_U)), None);
        assert!(vram_warning(heap(14 * GIB_U - MIB)).is_some());
    }

    #[test]
    fn is_target_from_15_gib() {
        assert!(is_target(16 * GIB_U));
        assert!(is_target(15 * GIB_U));
        assert!(!is_target(15 * GIB_U - 1));
        assert!(!is_target(12 * GIB_U));
    }

    #[test]
    fn host_shortfall_names_both_figures() {
        assert_eq!(host_shortfall(20 * GIB_U, 20 * GIB_U), None);
        assert_eq!(host_shortfall(30 * GIB_U, 20 * GIB_U), None);
        let s = host_shortfall(10 * GIB_U, 21 * GIB_U + GIB_U / 2).expect("short");
        assert_eq!(
            s,
            "This PC has 10.0 GiB of memory available and loading the model needs about 21.5 GiB. Close other \
             programs, or stop the Artemis server that is running, and try again."
        );
    }

    #[test]
    fn host_need_counts_the_experts_and_the_embedding() {
        // Each tensor is a vector of f32: 4 bytes an element.
        let mut w = Writer::new();
        for (name, n) in [
            ("token_embd.weight", 8u64),
            ("blk.0.ffn_gate_exps.weight", 16),
            ("blk.0.ffn_up_exps.weight", 16),
            ("blk.39.ffn_down_exps.weight", 32),
            ("blk.40.ffn_up_exps.weight", 64),
            // Not counted: the shared expert, a mixer, the router, the output head.
            ("blk.0.ffn_gate_shexp.weight", 100),
            ("blk.0.attn_qkv.weight", 100),
            ("blk.40.attn_q.weight", 100),
            ("output.weight", 100),
        ] {
            w.tensor(name, &[n], F32, vec![0u8; n as usize * 4]);
        }
        let g = Gguf::from_bytes(w.finish()).unwrap();
        // Plus prefill's host buffers, which the engine counts beside the weights.
        let prefill = engine::prefill_host_bytes(&Dims::QWEN36, engine::PREFILL_CHUNK);
        assert!(prefill > 0);
        assert_eq!(host_need(&g, false), (8 + 16 + 16 + 32) * 4 + prefill);
        assert_eq!(host_need(&g, true), (8 + 16 + 16 + 32 + 64) * 4 + prefill);
    }
}
