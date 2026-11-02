//! The Qwen3.6-35B-A3B model contract (`docs/reference/qwen35moe-semantics-b11509.md`): what a GGUF must hold, how
//! its tensors map to roles, dequantization, the reference's operations, the f32 CPU reference forward pass, the
//! tokenizer, and streamed text.
pub mod contract;
pub mod dims;
pub mod model;
pub mod ops;
pub mod pins;
pub mod quant;
pub mod reference;
pub mod sha256;
pub mod stream;
pub mod taps;
pub mod testing;
pub mod tokenizer;
