//! A minimal GGUF v3 writer, for building test models. Not used to write real model files.
use crate::{block_layout, Value};

pub struct Writer {
    alignment: u64,
    meta: Vec<(String, Value)>,
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
}

impl Default for Writer {
    fn default() -> Self {
        Writer {
            alignment: 32,
            meta: Vec::new(),
            tensors: Vec::new(),
        }
    }
}

impl Writer {
    pub fn new() -> Writer {
        Writer::default()
    }

    /// Writes `general.alignment` (a power of two) and aligns tensor data to it.
    pub fn alignment(&mut self, a: u64) -> &mut Self {
        assert!(a.is_power_of_two(), "alignment {a} is not a power of two");
        self.alignment = a;
        self.meta
            .push(("general.alignment".into(), Value::U32(a as u32)));
        self
    }

    pub fn meta(&mut self, key: &str, value: Value) -> &mut Self {
        self.meta.push((key.into(), value));
        self
    }

    /// Adds a tensor; `data` must be exactly the size its dims and type imply.
    pub fn tensor(&mut self, name: &str, dims: &[u64], ggml_type: u32, data: Vec<u8>) -> &mut Self {
        let (blk, bsize) = block_layout(ggml_type).expect("a known ggml type");
        let want = dims[1..].iter().product::<u64>() * (dims[0] / blk * bsize);
        assert_eq!(
            data.len() as u64,
            want,
            "tensor {name}: {} bytes, but its dims and type need {want}",
            data.len()
        );
        self.tensors
            .push((name.into(), dims.to_vec(), ggml_type, data));
        self
    }

    pub fn finish(&self) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend((self.tensors.len() as u64).to_le_bytes());
        out.extend((self.meta.len() as u64).to_le_bytes());
        for (k, v) in &self.meta {
            put_string(&mut out, k);
            out.extend(type_id(v).to_le_bytes());
            put_value(&mut out, v);
        }
        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut next = 0u64;
        for (name, dims, ty, data) in &self.tensors {
            put_string(&mut out, name);
            out.extend((dims.len() as u32).to_le_bytes());
            for d in dims {
                out.extend(d.to_le_bytes());
            }
            out.extend(ty.to_le_bytes());
            out.extend(next.to_le_bytes());
            offsets.push(next);
            next = (next + data.len() as u64).div_ceil(self.alignment) * self.alignment;
        }
        let data_start = (out.len() as u64).div_ceil(self.alignment) * self.alignment;
        for ((_, _, _, data), offset) in self.tensors.iter().zip(offsets) {
            out.resize((data_start + offset) as usize, 0);
            out.extend(data);
        }
        out
    }
}

fn type_id(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn put_string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

fn put_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => out.push(*x),
        Value::I8(x) => out.push(*x as u8),
        Value::U16(x) => out.extend(x.to_le_bytes()),
        Value::I16(x) => out.extend(x.to_le_bytes()),
        Value::U32(x) => out.extend(x.to_le_bytes()),
        Value::I32(x) => out.extend(x.to_le_bytes()),
        Value::F32(x) => out.extend(x.to_le_bytes()),
        Value::Bool(x) => out.push(u8::from(*x)),
        Value::String(s) => put_string(out, s),
        Value::U64(x) => out.extend(x.to_le_bytes()),
        Value::I64(x) => out.extend(x.to_le_bytes()),
        Value::F64(x) => out.extend(x.to_le_bytes()),
        Value::Array(a) => {
            let elem = a.first().map_or(4, type_id);
            out.extend(elem.to_le_bytes());
            out.extend((a.len() as u64).to_le_bytes());
            for x in a {
                assert_eq!(type_id(x), elem, "a GGUF array holds one type");
                put_value(out, x);
            }
        }
    }
}
