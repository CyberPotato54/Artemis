mod common;
use common::{f32_bytes, sample};
use gguf::{types, write::Writer, Gguf, Value};

#[test]
fn round_trip_metadata_and_tensors() {
    let g = Gguf::from_bytes(sample()).unwrap();
    assert_eq!(g.version(), 3);
    assert_eq!(g.alignment(), 32);
    assert_eq!(g.meta_keys().count(), 14);
    assert_eq!(
        g.meta("general.architecture").and_then(Value::as_str),
        Some("test")
    );
    assert_eq!(g.meta("u8"), Some(&Value::U8(7)));
    assert_eq!(g.meta("i8"), Some(&Value::I8(-7)));
    assert_eq!(g.meta("u16"), Some(&Value::U16(700)));
    assert_eq!(g.meta("i16"), Some(&Value::I16(-700)));
    assert_eq!(g.meta("u32").and_then(Value::as_u64), Some(70_000));
    assert_eq!(g.meta("i32").and_then(Value::as_i64), Some(-70_000));
    assert_eq!(g.meta("i32").and_then(Value::as_u64), None);
    assert_eq!(g.meta("u64").and_then(Value::as_u64), Some(1 << 40));
    assert_eq!(g.meta("i64").and_then(Value::as_i64), Some(-(1 << 40)));
    assert_eq!(g.meta("f32").and_then(Value::as_f64), Some(1.5));
    assert_eq!(g.meta("f64").and_then(Value::as_f64), Some(-2.25));
    assert_eq!(g.meta("bool").and_then(Value::as_bool), Some(true));
    let strings: Vec<&str> = g
        .meta("strings")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(strings, ["a", "b c"]);
    let nested = g.meta("nested").and_then(Value::as_array).unwrap();
    assert_eq!(nested[0], Value::Array(vec![Value::I32(1), Value::I32(2)]));
    assert_eq!(nested[1], Value::Array(vec![Value::I32(3)]));

    assert_eq!(g.tensor_count(), 3);
    let a = g.tensor("a").unwrap();
    assert_eq!((a.dims, a.ggml_type), (&[4u64][..], types::F32));
    assert_eq!(a.bytes, f32_bytes(&[1.0, 2.0, 3.0, 4.0]).as_slice());
    let b = g.tensor("b").unwrap();
    assert_eq!(
        (b.dims, b.ggml_type, b.bytes.len()),
        (&[32u64, 2][..], types::Q8_0, 68)
    );
    assert!(b.bytes.iter().all(|&x| x == 0x11));
    let c = g.tensor("c").unwrap();
    assert_eq!((c.ggml_type, c.bytes.len()), (types::Q4_K, 144));
    assert!(g.tensor("missing").is_none());
    assert_eq!(
        g.tensors().map(|t| t.name).collect::<Vec<_>>(),
        ["a", "b", "c"]
    );
}

#[test]
fn tensor_data_is_aligned() {
    let g = Gguf::from_bytes(sample()).unwrap();
    for t in g.tensors() {
        assert!(t.offset.is_multiple_of(32), "{} at {}", t.name, t.offset);
    }
    let mut w = Writer::new();
    w.alignment(64)
        .tensor("x", &[3], types::F32, f32_bytes(&[1.0, 2.0, 3.0]))
        .tensor("y", &[1], types::F32, f32_bytes(&[4.0]));
    let g = Gguf::from_bytes(w.finish()).unwrap();
    assert_eq!(g.alignment(), 64);
    for t in g.tensors() {
        assert!(t.offset.is_multiple_of(64), "{} at {}", t.name, t.offset);
    }
    assert_eq!(g.tensor("y").unwrap().bytes, f32_bytes(&[4.0]).as_slice());
}

#[test]
fn opens_a_file_from_disk() {
    let dir = std::env::temp_dir().join(format!("gguf-open-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sample.gguf");
    std::fs::write(&path, sample()).unwrap();
    let g = Gguf::open(&path).unwrap();
    assert_eq!(g.tensor("c").unwrap().bytes.len(), 144);
    drop(g);
    std::fs::remove_dir_all(&dir).unwrap();
}
