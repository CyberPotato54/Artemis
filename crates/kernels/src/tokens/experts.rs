//! The routed experts over up to `MAX_TOKENS` tokens, each distinct expert read once: `moe_unique` lists a
//! layer's distinct experts with the tokens that chose each, and the two projections then run one workgroup per row
//! block and listed expert, dotting each weight unit they load with every token of the expert's mask in decode's
//! order. Token `i`'s values are `experts_gate_up`'s and `experts_down`'s bit for bit, and land at the routed index
//! decode would use: the outputs are decode's `[k][...]` with a token index in front.
use super::MAX_TOKENS;
use crate::{
    dispatch::{expert_rows, groups, SlotLayout},
    wtype_code, Kernel, Kernels,
};
use vk::Recorder;

/// The most (token, choice) pairs `moe_unique` lists: one workgroup of 256 invocations, one per pair.
use super::MAX_PAIRS;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MoeUniquePush {
    pub ids: u64,
    pub list: u64,
    pub tokens: u32,
    pub k: u32,
}
push_constants!(MoeUniquePush: u64, u64, u32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertsGateUpTokensPush {
    pub table: u64,
    pub ids: u64,
    pub list: u64,
    pub x: u64,
    pub y: u64,
    pub status: u64,
    pub layer_base: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub gate_type: u32,
    pub up_type: u32,
    pub up_off: u32,
    pub layer: u32,
    pub k: u32,
    /// +0.0, the tree's added zero on rows of at most 16 units (`tree32.glsl`), as in `ExpertsGateUpPush`.
    pub zero: f32,
    pub pad: u32,
}
push_constants!(ExpertsGateUpTokensPush: u64, u64, u64, u64, u64, u64, u32, u32, u32, u32, u32, u32, u32, u32, f32, u32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertsDownTokensPush {
    pub table: u64,
    pub ids: u64,
    pub list: u64,
    pub act: u64,
    pub y: u64,
    pub status: u64,
    pub layer_base: u32,
    pub n_ff: u32,
    pub n_embd: u32,
    pub down_type: u32,
    pub down_off: u32,
    pub layer: u32,
    pub k: u32,
    /// +0.0, the tree's added zero on rows of at most 16 units (`tree32.glsl`), as in `ExpertsDownPush`.
    pub zero: f32,
}
push_constants!(ExpertsDownTokensPush: u64, u64, u64, u64, u64, u64, u32, u32, u32, u32, u32, u32, u32, f32);

// Pushes carry no implicit padding: `Recorder` hands the struct's bytes to Vulkan.
const _: () = {
    use std::mem::size_of;
    assert!(size_of::<MoeUniquePush>() == 2 * 8 + 2 * 4);
    assert!(size_of::<ExpertsGateUpTokensPush>() == 6 * 8 + 10 * 4);
    assert!(size_of::<ExpertsDownTokensPush>() == 6 * 8 + 8 * 4);
};

/// The checks every token expert dispatch makes on its shape.
fn check_shape(what: &str, tokens: usize, k: usize) {
    assert!(
        (1..=MAX_TOKENS).contains(&tokens),
        "{what}: {tokens} tokens; 1 to {MAX_TOKENS}"
    );
    // The bounds above already imply `tokens * k <= MAX_PAIRS` (a const assert below pins `MAX_TOKENS * 16`), so the
    // second condition cannot fire today; it stays as the statement of the kernels' limit, for a later change to
    // either.
    assert!(
        (1..=16).contains(&k) && tokens * k <= MAX_PAIRS,
        "{what}: {tokens} tokens of {k} choices; 1 to 16 choices, at most {MAX_PAIRS} in all"
    );
}

impl Kernels {
    /// `list` ← the distinct experts of `ids` (`[tokens][k]` f32, as `router_topk` writes them), as
    /// `moe_unique_bytes(tokens, k)` bytes: `[count: u32]`, then per unique expert in order of first appearance
    /// `[expert: u32, mask: u32]`, bit `i` of the mask set when token `i` chose it.
    pub fn moe_unique(&self, r: &mut Recorder, ids: u64, list: u64, tokens: usize, k: usize) {
        check_shape("moe_unique", tokens, k);
        assert!(
            ids.is_multiple_of(4) && list.is_multiple_of(4),
            "moe_unique: ids and list must be 4-byte aligned"
        );
        let push = MoeUniquePush {
            ids,
            list,
            tokens: tokens as u32,
            k: k as u32,
        };
        r.dispatch(self.pipe(Kernel::MoeUnique), &push, [1, 1, 1]);
    }

    /// `y` (`[tokens][k][n_ff]`) ← each routed expert's `silu_det(gate·x[i]) ⊙ (up·x[i])` for every token `i` that
    /// chose it, reading layer `layer`'s entries of `table` and each distinct expert once. `ids` is `[tokens][k]`,
    /// `list` what `moe_unique` wrote for them, `x` `[tokens][n_embd]`. Grid `[ceil(n_ff / expert_rows(n_embd)),
    /// tokens·k]`, decode's row blocks: a workgroup whose unique index is past the list's count exits. 2 to 5 tokens
    /// run a pipeline specialized to the count (`SPECIALIZED_TOKENS`), others the general one. An unusable slot gives
    /// NaN and `refuse(status, …)`, as `experts_gate_up` does (there is no CPU path here: speculation refuses the
    /// CPU share).
    #[allow(clippy::too_many_arguments)]
    pub fn experts_gate_up_tokens(
        &self,
        r: &mut Recorder,
        table: u64,
        layer: usize,
        n_expert: usize,
        ids: u64,
        list: u64,
        x: u64,
        y: u64,
        status: u64,
        lay: &SlotLayout,
        k: usize,
        tokens: usize,
    ) {
        check_shape("experts_gate_up_tokens", tokens, k);
        // Every token's input is read 16 bytes at a time: x, and the stride between tokens' inputs.
        assert!(
            x.is_multiple_of(16)
                && (4 * lay.n_embd).is_multiple_of(16)
                && lay.up_offset().is_multiple_of(16),
            "experts_gate_up_tokens: every token's input and the up matrix must be 16-byte aligned"
        );
        assert!(
            ids.is_multiple_of(4) && list.is_multiple_of(4),
            "experts_gate_up_tokens: ids and list must be 4-byte aligned"
        );
        let push = ExpertsGateUpTokensPush {
            table,
            ids,
            list,
            x,
            y,
            status,
            layer_base: (layer * n_expert) as u32,
            n_embd: lay.n_embd as u32,
            n_ff: lay.n_ff as u32,
            gate_type: wtype_code(lay.gate),
            up_type: wtype_code(lay.up),
            up_off: lay.up_offset() as u32,
            layer: layer as u32,
            k: k as u32,
            zero: 0.0,
            pad: 0,
        };
        r.dispatch(
            self.pipe_tokens(Kernel::ExpertsGateUpTokens, tokens),
            &push,
            [
                groups(lay.n_ff, expert_rows(lay.n_embd)),
                groups(tokens * k, 1),
                1,
            ],
        );
    }

    /// `y` (`[tokens][k][n_embd]`) ← each routed expert's `down · act[i][j]` for every token `i` that chose it as its
    /// choice `j`, each distinct expert read once. `act` is `[tokens][k][n_ff]`; the rest is
    /// `experts_gate_up_tokens`'s.
    #[allow(clippy::too_many_arguments)]
    pub fn experts_down_tokens(
        &self,
        r: &mut Recorder,
        table: u64,
        layer: usize,
        n_expert: usize,
        ids: u64,
        list: u64,
        act: u64,
        y: u64,
        status: u64,
        lay: &SlotLayout,
        k: usize,
        tokens: usize,
    ) {
        check_shape("experts_down_tokens", tokens, k);
        // Token i's input is act + 4·(i·k + j)·n_ff bytes: aligned when act and one activation row are.
        assert!(
            act.is_multiple_of(16)
                && (4 * lay.n_ff).is_multiple_of(16)
                && lay.down_offset().is_multiple_of(16),
            "experts_down_tokens: every token's activations and the down matrix must be 16-byte aligned"
        );
        assert!(
            ids.is_multiple_of(4) && list.is_multiple_of(4),
            "experts_down_tokens: ids and list must be 4-byte aligned"
        );
        let push = ExpertsDownTokensPush {
            table,
            ids,
            list,
            act,
            y,
            status,
            layer_base: (layer * n_expert) as u32,
            n_ff: lay.n_ff as u32,
            n_embd: lay.n_embd as u32,
            down_type: wtype_code(lay.down),
            down_off: lay.down_offset() as u32,
            layer: layer as u32,
            k: k as u32,
            zero: 0.0,
        };
        r.dispatch(
            self.pipe_tokens(Kernel::ExpertsDownTokens, tokens),
            &push,
            [
                groups(lay.n_embd, expert_rows(lay.n_ff)),
                groups(tokens * k, 1),
                1,
            ],
        );
    }
}
