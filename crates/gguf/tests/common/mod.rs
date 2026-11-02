//! Test files: one built with the writer, and raw bytes for the cases the writer refuses to produce.
#![allow(dead_code)]
use gguf::{types, write::Writer, Value};

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Every value type, nested arrays, and three tensors of different types.
pub fn sample() -> Vec<u8> {
    let mut w = Writer::new();
    w.meta("general.architecture", Value::String("test".into()))
        .meta("u8", Value::U8(7))
        .meta("i8", Value::I8(-7))
        .meta("u16", Value::U16(700))
        .meta("i16", Value::I16(-700))
        .meta("u32", Value::U32(70_000))
        .meta("i32", Value::I32(-70_000))
        .meta("u64", Value::U64(1 << 40))
        .meta("i64", Value::I64(-(1 << 40)))
        .meta("f32", Value::F32(1.5))
        .meta("f64", Value::F64(-2.25))
        .meta("bool", Value::Bool(true))
        .meta(
            "strings",
            Value::Array(vec![Value::String("a".into()), Value::String("b c".into())]),
        )
        .meta(
            "nested",
            Value::Array(vec![
                Value::Array(vec![Value::I32(1), Value::I32(2)]),
                Value::Array(vec![Value::I32(3)]),
            ]),
        );
    w.tensor("a", &[4], types::F32, f32_bytes(&[1.0, 2.0, 3.0, 4.0]))
        .tensor("b", &[32, 2], types::Q8_0, vec![0x11; 68])
        .tensor("c", &[256], types::Q4_K, vec![0x22; 144]);
    w.finish()
}

/// A tensor-info record for `raw`: name, dims, ggml type, offset into the data section.
pub struct Info<'a>(pub &'a str, pub &'a [u64], pub u32, pub u64);

/// A file with no metadata, the given tensor infos written verbatim, and `data_len` zero bytes of data
/// after 32-byte alignment. Lets tests write offsets and types the writer would never produce.
pub fn raw(infos: &[Info], data_len: usize) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((infos.len() as u64).to_le_bytes());
    out.extend(0u64.to_le_bytes());
    for Info(name, dims, ty, offset) in infos {
        out.extend((name.len() as u64).to_le_bytes());
        out.extend(name.as_bytes());
        out.extend((dims.len() as u32).to_le_bytes());
        for d in *dims {
            out.extend(d.to_le_bytes());
        }
        out.extend(ty.to_le_bytes());
        out.extend(offset.to_le_bytes());
    }
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    out.resize(out.len() + data_len, 0);
    out
}

/// A file with one metadata entry whose value bytes are written verbatim after the type id.
pub fn raw_meta(key: &str, ty: u32, value: &[u8]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend(1u64.to_le_bytes());
    out.extend((key.len() as u64).to_le_bytes());
    out.extend(key.as_bytes());
    out.extend(ty.to_le_bytes());
    out.extend(value);
    out
}
