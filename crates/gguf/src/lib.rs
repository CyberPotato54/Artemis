//! GGUF v3: memory-mapped reading of metadata and the tensor table, plus a minimal writer for test models.
//! It knows nothing about any model; `qwen36` decides what a file must contain.
mod read;
pub mod write;

pub use read::{Error, Gguf, TensorView, Value};

/// ggml type ids that Artemis names. Other ids are kept as plain numbers.
pub mod types {
    pub const F32: u32 = 0;
    pub const F16: u32 = 1;
    pub const Q8_0: u32 = 8;
    pub const Q4_K: u32 = 12;
    pub const Q5_K: u32 = 13;
    pub const Q6_K: u32 = 14;
    pub const BF16: u32 = 30;
}

/// (elements per block, bytes per block) for the ggml types a GGUF file can hold; None for an unknown id.
pub fn block_layout(ggml_type: u32) -> Option<(u64, u64)> {
    Some(match ggml_type {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 36),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        30 => (1, 2),     // BF16
        _ => return None,
    })
}
