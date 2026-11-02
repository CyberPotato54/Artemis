/// Every hyperparameter the contract checks (semantics reference §1). `Dims::QWEN36` is the real model and the only
/// value `artemis` accepts; `Dims::tiny()` describes the synthetic test model.
#[derive(Clone, Debug, PartialEq)]
pub struct Dims {
    /// Layers in the main pass: 40. The MTP block is layer `n_layer`.
    pub n_layer: usize,
    pub n_nextn: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_ctx_train: usize,
    pub rms_eps: f32,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_sections: [u32; 4],
    pub rope_base: f32,
    pub full_attn_interval: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub ssm_conv: usize,
    /// The DeltaNet key head size, which is also the value head size.
    pub ssm_d_state: usize,
    /// Key groups: 16.
    pub ssm_n_group: usize,
    /// Value heads: 32.
    pub ssm_dt_rank: usize,
    pub ssm_inner: usize,
}

impl Dims {
    pub const QWEN36: Dims = Dims {
        n_layer: 40,
        n_nextn: 1,
        n_embd: 2048,
        n_vocab: 248_320,
        n_ctx_train: 262_144,
        rms_eps: 1e-6,
        n_head: 16,
        n_head_kv: 2,
        head_dim: 256,
        n_rot: 64,
        rope_sections: [11, 11, 10, 0],
        rope_base: 1e7,
        full_attn_interval: 4,
        n_expert: 256,
        n_expert_used: 8,
        n_ff_exp: 512,
        n_ff_shexp: 512,
        ssm_conv: 4,
        ssm_d_state: 128,
        ssm_n_group: 16,
        ssm_dt_rank: 32,
        ssm_inner: 4096,
    };

    /// The tiny synthetic model. Rows are at least 256 wide because K-quants need it. Test-only:
    /// `artemis` never opens a model with these dims.
    pub fn tiny() -> Dims {
        Dims {
            n_layer: 4,
            n_nextn: 1,
            n_embd: 256,
            n_vocab: 512,
            n_ctx_train: 4096,
            rms_eps: 1e-6,
            n_head: 4,
            n_head_kv: 1,
            head_dim: 64,
            n_rot: 16,
            rope_sections: [3, 3, 2, 0],
            rope_base: 1e7,
            full_attn_interval: 4,
            n_expert: 16,
            n_expert_used: 4,
            n_ff_exp: 256,
            n_ff_shexp: 256,
            ssm_conv: 4,
            ssm_d_state: 64,
            ssm_n_group: 4,
            ssm_dt_rank: 8,
            ssm_inner: 512,
        }
    }

    /// Full attention in layers 3, 7, …, and in the MTP block; Gated DeltaNet elsewhere.
    pub fn is_attention(&self, layer: usize) -> bool {
        layer >= self.n_layer || (layer + 1).is_multiple_of(self.full_attn_interval)
    }

    /// Width of q (and of k) in a DeltaNet layer: key groups × key head size.
    pub fn key_dim(&self) -> usize {
        self.ssm_d_state * self.ssm_n_group
    }

    /// Channels of the fused QKV projection and the conv: q, k, then v.
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.ssm_inner
    }

    pub fn v_head_dim(&self) -> usize {
        self.ssm_inner / self.ssm_dt_rank
    }
}
