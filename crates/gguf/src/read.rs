use crate::block_layout;
use std::{collections::HashMap, fmt, fs::File, path::Path};

/// A metadata value. Arrays hold values of one type; GGUF allows arrays of arrays.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    /// Any integer type holding a non-negative value.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    /// Any integer type whose value fits in an i64.
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => i64::try_from(v).ok(),
            Value::I8(v) => Some(v.into()),
            Value::I16(v) => Some(v.into()),
            Value::I32(v) => Some(v.into()),
            Value::I64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v.into()),
            Value::F64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// Why a file was refused. `offset` is the byte where reading went wrong.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Malformed { offset: u64, what: String },
    UnknownType { tensor: String, ggml_type: u32 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "reading the GGUF file: {e}"),
            Error::Malformed { offset, what } => {
                write!(f, "malformed GGUF at byte {offset}: {what}")
            }
            Error::UnknownType { tensor, ggml_type } => {
                write!(f, "tensor {tensor:?} has unknown ggml type {ggml_type}")
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// One tensor: dims with `dims[0]` the contiguous one, the ggml type id, its absolute file offset and its bytes.
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    pub name: &'a str,
    pub dims: &'a [u64],
    pub ggml_type: u32,
    pub offset: u64,
    pub bytes: &'a [u8],
}

struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    start: u64,
    len: u64,
}

enum Bytes {
    /// The mapping, and the file handle it was made from. On Windows the handle's share mode (read only) is what
    /// keeps other processes from writing the file, so it lives as long as the map.
    Map {
        map: memmap2::Mmap,
        _file: File,
    },
    Owned(Vec<u8>),
}

impl Bytes {
    fn get(&self) -> &[u8] {
        match self {
            Bytes::Map { map, .. } => map,
            Bytes::Owned(v) => v,
        }
    }
}

pub struct Gguf {
    bytes: Bytes,
    version: u32,
    alignment: u64,
    meta: Vec<(String, Value)>,
    meta_index: HashMap<String, usize>,
    tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
}

const MAX_DEPTH: u32 = 4;
const MAX_DIMS: u32 = 4;
/// The most metadata values (array elements included) a file may hold. Each costs about 32 bytes of memory, so this
/// bounds what a hostile file can make us allocate; the real model has about 0.75 million.
const MAX_META_VALUES: u64 = 1 << 21;
/// The longest metadata key or tensor name, in bytes.
const MAX_NAME: u64 = 256;
/// The longest metadata string value, in bytes; the chat template is the largest the real model has.
const MAX_STRING: u64 = 1 << 20;
/// All metadata strings' bytes together.
const MAX_META_BYTES: u64 = 128 << 20;
/// The most tensors a file may hold (the real model has 753).
const MAX_TENSORS: u64 = 1 << 16;
/// The longest a file-provided value is shown in an error, in characters.
const SHOW_CHARS: usize = 200;

/// `s` cut to `SHOW_CHARS` characters, so a crafted value can't flood the terminal.
fn clip(s: String) -> String {
    match s.char_indices().nth(SHOW_CHARS) {
        Some((end, _)) => format!("{}...", &s[..end]),
        None => s,
    }
}

/// `n` with thousands separators.
fn group(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf, Error> {
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        // FILE_SHARE_READ: other processes may read the file but not write, truncate or delete it while it is open.
        #[cfg(windows)]
        std::os::windows::fs::OpenOptionsExt::share_mode(&mut options, 1);
        let file = options.open(path)?;
        // SAFETY: the mapping is read-only. On Windows the file is opened without write or delete sharing and the
        // handle is kept beside the map, so no other process can modify the bytes while the map lives. Elsewhere
        // nothing stops another process from rewriting the file; model files aren't modified while Artemis runs.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Gguf::parse(Bytes::Map { map, _file: file })
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Gguf, Error> {
        Gguf::parse(Bytes::Owned(bytes))
    }

    fn parse(bytes: Bytes) -> Result<Gguf, Error> {
        let (version, alignment, meta, tensors) = parse_parts(bytes.get())?;
        let mut meta_index = HashMap::new();
        for (i, (k, _)) in meta.iter().enumerate() {
            if meta_index.insert(k.clone(), i).is_some() {
                return Err(Error::Malformed {
                    offset: 0,
                    what: format!("duplicate metadata key {k:?}"),
                });
            }
        }
        let mut tensor_index = HashMap::new();
        for (i, t) in tensors.iter().enumerate() {
            if tensor_index.insert(t.name.clone(), i).is_some() {
                return Err(Error::Malformed {
                    offset: t.start,
                    what: format!("duplicate tensor name {:?}", t.name),
                });
            }
        }
        Ok(Gguf {
            bytes,
            version,
            alignment,
            meta,
            meta_index,
            tensors,
            tensor_index,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn meta(&self, key: &str) -> Option<&Value> {
        self.meta_index.get(key).map(|&i| &self.meta[i].1)
    }

    pub fn meta_keys(&self) -> impl Iterator<Item = &str> + '_ {
        self.meta.iter().map(|(k, _)| k.as_str())
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensor(&self, name: &str) -> Option<TensorView<'_>> {
        self.tensor_index
            .get(name)
            .map(|&i| self.view(&self.tensors[i]))
    }

    pub fn tensors(&self) -> impl Iterator<Item = TensorView<'_>> + '_ {
        self.tensors.iter().map(|t| self.view(t))
    }

    fn view<'a>(&'a self, t: &'a TensorInfo) -> TensorView<'a> {
        TensorView {
            name: &t.name,
            dims: &t.dims,
            ggml_type: t.ggml_type,
            offset: t.start,
            bytes: &self.bytes.get()[t.start as usize..(t.start + t.len) as usize],
        }
    }
}

type Parts = (u32, u64, Vec<(String, Value)>, Vec<TensorInfo>);

fn parse_parts(b: &[u8]) -> Result<Parts, Error> {
    let mut c = Cursor {
        b,
        pos: 0,
        values: 0,
        str_bytes: 0,
        in_meta: true,
    };
    let magic = c.take(4, "magic")?;
    if magic != b"GGUF" {
        return Err(c.err_at(0, format!("bad magic {magic:02x?}, expected \"GGUF\"")));
    }
    let version = c.u32("version")?;
    if version != 3 {
        return Err(c.err_at(4, format!("version {version}; Artemis reads version 3")));
    }
    let n_tensors = c.u64("tensor count")?;
    if n_tensors > MAX_TENSORS {
        return Err(c.err_at(
            c.pos,
            format!("tensor count {n_tensors} is over the {MAX_TENSORS} limit"),
        ));
    }
    let n_kv = c.u64("metadata count")?;
    // Each entry needs at least this many bytes, so a larger count is corrupt. Checking first avoids
    // allocating for it.
    c.check_count(n_kv, 13, "metadata count")?;
    let mut meta = Vec::new();
    for _ in 0..n_kv {
        let key = c.string("metadata key", MAX_NAME)?;
        let ty = c.u32("value type")?;
        let v = c.value(ty, 0)?;
        meta.push((key, v));
    }
    c.in_meta = false;
    let alignment = match meta.iter().find(|(k, _)| k == "general.alignment") {
        None => 32,
        Some((_, v)) => v.as_u64().filter(|a| a.is_power_of_two()).ok_or_else(|| {
            c.err_at(
                0,
                format!(
                    "general.alignment {} is not a power of two",
                    clip(format!("{v:?}"))
                ),
            )
        })?,
    };
    c.check_count(n_tensors, 32, "tensor count")?;
    let mut infos = Vec::new();
    for _ in 0..n_tensors {
        let at = c.pos;
        let name = c.string("tensor name", MAX_NAME)?;
        let n_dims = c.u32("tensor dims")?;
        if n_dims == 0 || n_dims > MAX_DIMS {
            return Err(c.err_at(at, format!("tensor {name:?} has {n_dims} dims")));
        }
        let mut dims = Vec::new();
        for _ in 0..n_dims {
            let d = c.u64("tensor dim")?;
            if d == 0 {
                return Err(c.err_at(at, format!("tensor {name:?} has a zero dim")));
            }
            dims.push(d);
        }
        let ggml_type = c.u32("tensor type")?;
        let offset = c.u64("tensor offset")?;
        infos.push((at, name, dims, ggml_type, offset));
    }
    let data_start = (c.pos as u64).div_ceil(alignment) * alignment;
    let file_len = b.len() as u64;
    let mut tensors = Vec::with_capacity(infos.len());
    for (at, name, dims, ggml_type, offset) in infos {
        let (blk, bsize) = block_layout(ggml_type).ok_or_else(|| Error::UnknownType {
            tensor: name.clone(),
            ggml_type,
        })?;
        if !dims[0].is_multiple_of(blk) {
            return Err(c.err_at(
                at,
                format!(
                    "tensor {name:?}: first dim {} is not a whole number of {blk}-element blocks",
                    dims[0]
                ),
            ));
        }
        let len = dims[1..]
            .iter()
            .try_fold(dims[0] / blk, |acc, &d| acc.checked_mul(d))
            .and_then(|blocks| blocks.checked_mul(bsize))
            .ok_or_else(|| c.err_at(at, format!("tensor {name:?}: its size overflows")))?;
        if !offset.is_multiple_of(alignment) {
            return Err(c.err_at(
                at,
                format!("tensor {name:?}: offset {offset} is not aligned to {alignment}"),
            ));
        }
        let start = data_start
            .checked_add(offset)
            .filter(|s| s.checked_add(len).is_some_and(|end| end <= file_len));
        let Some(start) = start else {
            return Err(c.err_at(at, format!("tensor {name:?}: {len} bytes at data offset {offset} run past the end of the file ({file_len} bytes)")));
        };
        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            start,
            len,
        });
    }
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by_key(|&i| tensors[i].start);
    for w in order.windows(2) {
        let (a, b) = (&tensors[w[0]], &tensors[w[1]]);
        if a.start + a.len > b.start {
            return Err(Error::Malformed {
                offset: b.start,
                what: format!("tensors {:?} and {:?} overlap", a.name, b.name),
            });
        }
    }
    Ok((version, alignment, meta, tensors))
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
    /// Metadata values read so far, against MAX_META_VALUES.
    values: u64,
    /// Bytes of metadata strings read so far, against MAX_META_BYTES.
    str_bytes: u64,
    /// True while reading metadata, false once tensor infos start.
    in_meta: bool,
}

impl<'a> Cursor<'a> {
    fn err_at(&self, offset: usize, what: impl Into<String>) -> Error {
        Error::Malformed {
            offset: offset as u64,
            what: what.into(),
        }
    }

    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.b.len());
        let Some(end) = end else {
            return Err(self.err_at(self.pos, format!("truncated while reading {what}")));
        };
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn array<const N: usize>(&mut self, what: &str) -> Result<[u8; N], Error> {
        Ok(self
            .take(N, what)?
            .try_into()
            .expect("take returned N bytes"))
    }

    fn u32(&mut self, what: &str) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.array(what)?))
    }

    fn u64(&mut self, what: &str) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.array(what)?))
    }

    fn string(&mut self, what: &str, max: u64) -> Result<String, Error> {
        let at = self.pos;
        let n = self.u64(what)?;
        if n > max {
            return Err(self.err_at(
                at,
                format!("{what}: length {n} is over the {max}-byte limit"),
            ));
        }
        if self.in_meta {
            self.str_bytes += n;
            if self.str_bytes > MAX_META_BYTES {
                return Err(self.err_at(
                    at,
                    format!(
                        "the metadata strings total more than {} bytes",
                        group(MAX_META_BYTES)
                    ),
                ));
            }
        }
        let left = (self.b.len() - self.pos) as u64;
        if n > left {
            return Err(self.err_at(
                at,
                format!("{what}: length {n} runs past the end of the file"),
            ));
        }
        let bytes = self.take(n as usize, what)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| self.err_at(at, format!("{what} is not UTF-8")))
    }

    fn count_values(&mut self, n: u64) -> Result<(), Error> {
        self.values += n;
        if self.values > MAX_META_VALUES {
            return Err(self.err_at(
                self.pos,
                format!("more than {MAX_META_VALUES} metadata values"),
            ));
        }
        Ok(())
    }

    fn check_count(&self, n: u64, min_each: u64, what: &str) -> Result<(), Error> {
        let left = (self.b.len() - self.pos) as u64;
        if n.checked_mul(min_each).is_none_or(|need| need > left) {
            return Err(self.err_at(
                self.pos,
                format!("{what} {n} is larger than the file allows"),
            ));
        }
        Ok(())
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<Value, Error> {
        let at = self.pos;
        self.count_values(1)?;
        Ok(match ty {
            0 => Value::U8(self.array::<1>("u8")?[0]),
            1 => Value::I8(self.array::<1>("i8")?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.array("u16")?)),
            3 => Value::I16(i16::from_le_bytes(self.array("i16")?)),
            4 => Value::U32(u32::from_le_bytes(self.array("u32")?)),
            5 => Value::I32(i32::from_le_bytes(self.array("i32")?)),
            6 => Value::F32(f32::from_le_bytes(self.array("f32")?)),
            7 => match self.array::<1>("bool")?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                v => return Err(self.err_at(at, format!("bool value {v}"))),
            },
            8 => Value::String(self.string("string value", MAX_STRING)?),
            9 => {
                if depth >= MAX_DEPTH {
                    return Err(self.err_at(at, "arrays nested too deeply"));
                }
                let elem = self.u32("array type")?;
                let n = self.u64("array length")?;
                self.check_count(n, min_size(elem), "array length")?;
                if self.values + n > MAX_META_VALUES {
                    return Err(self.err_at(at, format!("array length {n} takes the file past {MAX_META_VALUES} metadata values")));
                }
                let mut v = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    v.push(self.value(elem, depth + 1)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(u64::from_le_bytes(self.array("u64")?)),
            11 => Value::I64(i64::from_le_bytes(self.array("i64")?)),
            12 => Value::F64(f64::from_le_bytes(self.array("f64")?)),
            t => return Err(self.err_at(at, format!("unknown value type {t}"))),
        })
    }
}

/// The fewest bytes one array element of this type can take.
fn min_size(ty: u32) -> u64 {
    match ty {
        2 | 3 => 2,
        4..=6 => 4,
        8 | 10..=12 => 8,
        9 => 12,
        _ => 1,
    }
}
