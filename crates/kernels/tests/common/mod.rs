//! Helpers the GPU test files share.
use qwen36::dims::Dims;

/// Decode attention's shapes as (query heads, KV heads, head size): both models', then heads of 96 values (the
/// combine's second 64-dimension workgroup partly used) and 224 (the GQA kernels' second 128-value K tile partly used).
pub fn attention_shapes() -> [(usize, usize, usize); 4] {
    let (a, b) = (Dims::QWEN36, Dims::tiny());
    [
        (a.n_head, a.n_head_kv, a.head_dim),
        (b.n_head, b.n_head_kv, b.head_dim),
        (8, 2, 96),
        (8, 1, 224),
    ]
}
