mod common;
use common::{raw, raw_meta, sample, Info};
use gguf::{types, Error, Gguf};

fn malformed(bytes: Vec<u8>) -> (u64, String) {
    match Gguf::from_bytes(bytes) {
        Err(Error::Malformed { offset, what }) => (offset, what),
        Err(e) => panic!("expected Malformed, got {e}"),
        Ok(_) => panic!("expected an error, but the file parsed"),
    }
}

#[test]
fn every_truncation_is_an_error_not_a_panic() {
    let full = sample();
    for n in 0..full.len() {
        assert!(
            Gguf::from_bytes(full[..n].to_vec()).is_err(),
            "a {n}-byte prefix parsed"
        );
    }
    assert!(Gguf::from_bytes(full).is_ok());
}

#[test]
fn bad_magic() {
    let mut b = sample();
    b[0] = b'X';
    let (offset, what) = malformed(b);
    assert_eq!(offset, 0);
    assert!(what.contains("magic"), "{what}");
}

#[test]
fn version_2_is_refused() {
    let mut b = sample();
    b[4..8].copy_from_slice(&2u32.to_le_bytes());
    let (offset, what) = malformed(b);
    assert_eq!(offset, 4);
    assert!(what.contains("version 2"), "{what}");
}

#[test]
fn absurd_counts_are_refused_without_allocating() {
    let mut b = sample();
    b[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(malformed(b).1.contains("tensor count"));
    let mut b = sample();
    b[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(malformed(b).1.contains("metadata count"));
    // An array claiming 2^40 u32s.
    let mut v = 4u32.to_le_bytes().to_vec();
    v.extend((1u64 << 40).to_le_bytes());
    assert!(malformed(raw_meta("k", 9, &v)).1.contains("array length"));
}

#[test]
fn string_running_past_the_end() {
    let v = 1_000_000u64.to_le_bytes();
    assert!(malformed(raw_meta("k", 8, &v))
        .1
        .contains("runs past the end"));
}

#[test]
fn bad_bool_and_unknown_value_type() {
    assert!(malformed(raw_meta("k", 7, &[2])).1.contains("bool"));
    assert!(malformed(raw_meta("k", 13, &[0; 8]))
        .1
        .contains("unknown value type 13"));
}

#[test]
fn arrays_nested_too_deeply() {
    // Five levels of one-element arrays of arrays.
    let mut v = Vec::new();
    for _ in 0..5 {
        v.extend(9u32.to_le_bytes());
        v.extend(1u64.to_le_bytes());
    }
    v.extend(4u32.to_le_bytes());
    v.extend(0u64.to_le_bytes());
    assert!(malformed(raw_meta("k", 9, &v)).1.contains("nested"));
}

#[test]
fn too_many_dims_and_zero_dims() {
    let (_, what) = malformed(raw(&[Info("t", &[1, 1, 1, 1, 1], types::F32, 0)], 4));
    assert!(what.contains("5 dims"), "{what}");
    let (_, what) = malformed(raw(&[Info("t", &[4, 0], types::F32, 0)], 16));
    assert!(what.contains("zero"), "{what}");
}

#[test]
fn tensor_past_the_end() {
    let (_, what) = malformed(raw(&[Info("t", &[8], types::F32, 0)], 16));
    assert!(what.contains("past the end"), "{what}");
    let (_, what) = malformed(raw(&[Info("t", &[4], types::F32, 1 << 40)], 16));
    assert!(what.contains("past the end"), "{what}");
}

#[test]
fn overlapping_tensors() {
    let (_, what) = malformed(raw(
        &[
            Info("a", &[8], types::F32, 0),
            Info("b", &[8], types::F32, 0),
        ],
        64,
    ));
    assert!(what.contains("overlap"), "{what}");
}

#[test]
fn misaligned_offset() {
    let (_, what) = malformed(raw(&[Info("t", &[4], types::F32, 4)], 64));
    assert!(what.contains("not aligned"), "{what}");
}

#[test]
fn duplicate_tensor_names() {
    let (_, what) = malformed(raw(
        &[
            Info("t", &[4], types::F32, 0),
            Info("t", &[4], types::F32, 32),
        ],
        64,
    ));
    assert!(what.contains("duplicate"), "{what}");
}

#[test]
fn first_dim_not_a_whole_number_of_blocks() {
    let (_, what) = malformed(raw(&[Info("t", &[100], types::Q4_K, 0)], 1024));
    assert!(what.contains("block"), "{what}");
}

#[test]
fn unknown_ggml_type() {
    match Gguf::from_bytes(raw(&[Info("t", &[4], 23, 0)], 64)) {
        Err(Error::UnknownType { tensor, ggml_type }) => {
            assert_eq!((tensor.as_str(), ggml_type), ("t", 23))
        }
        other => panic!(
            "expected UnknownType, got {:?}",
            other.err().map(|e| e.to_string())
        ),
    }
}

#[test]
fn errors_read_well() {
    let e = Gguf::from_bytes(b"GGU".to_vec()).err().unwrap();
    assert_eq!(
        e.to_string(),
        "malformed GGUF at byte 0: truncated while reading magic"
    );
}

#[test]
fn huge_first_dim_is_refused_not_wrapped() {
    // (2^62 / 1) × 4 bytes overflows u64; it must be an error, not a wrapped size that passes the bounds check.
    let (_, what) = malformed(raw(&[Info("t", &[1 << 62, 1], types::F32, 0)], 16));
    assert!(what.contains("overflows"), "{what}");
}

#[test]
fn metadata_value_count_is_capped() {
    // 2^21 + 1 one-byte values: each fits in the file, but together they would cost ~32 bytes of memory each.
    let n: u64 = (1 << 21) + 1;
    let mut v = 0u32.to_le_bytes().to_vec();
    v.extend(n.to_le_bytes());
    v.resize(v.len() + n as usize, 0);
    let (_, what) = malformed(raw_meta("k", 9, &v));
    assert!(what.contains("metadata values"), "{what}");
}

/// A string-valued metadata entry of `n` bytes: the raw value (length, then bytes).
fn string_value(n: usize) -> Vec<u8> {
    let mut v = (n as u64).to_le_bytes().to_vec();
    v.resize(8 + n, b'x');
    v
}

#[test]
fn a_key_over_256_bytes_is_refused() {
    let long = "k".repeat(257);
    let (_, what) = malformed(raw_meta(&long, 4, &1u32.to_le_bytes()));
    assert!(what.contains("limit"), "{what}");
    assert!(Gguf::from_bytes(raw_meta(&"k".repeat(256), 4, &1u32.to_le_bytes())).is_ok());
}

#[test]
fn a_string_value_over_1_mib_is_refused() {
    let (_, what) = malformed(raw_meta("k", 8, &string_value((1 << 20) + 1)));
    assert!(what.contains("limit"), "{what}");
}

#[test]
fn a_string_of_1_mib_loads() {
    assert!(Gguf::from_bytes(raw_meta("k", 8, &string_value(1 << 20))).is_ok());
}

#[test]
fn more_than_2097152_values_are_refused() {
    // An array of 2^21 u8 plus the entry itself is one value over the cap.
    let mut v = 0u32.to_le_bytes().to_vec();
    v.extend((1u64 << 21).to_le_bytes());
    v.resize(v.len() + (1 << 21), 0);
    let (_, what) = malformed(raw_meta("k", 9, &v));
    assert!(what.contains("metadata values"), "{what}");
}

#[test]
fn metadata_strings_over_128_mib_in_total_are_refused() {
    let mut w = gguf::write::Writer::new();
    let s = "x".repeat(1 << 20);
    for i in 0..130 {
        w.meta(&format!("k{i}"), gguf::Value::String(s.clone()));
    }
    let (_, what) = malformed(w.finish());
    assert!(what.contains("134,217,728"), "{what}");
}

#[test]
fn a_control_character_in_a_name_is_escaped() {
    let name = "a\u{1b}[2K";
    let mut w = gguf::write::Writer::new();
    w.tensor(name, &[4], types::F32, vec![0; 16])
        .tensor(name, &[4], types::F32, vec![0; 16]);
    let (_, what) = malformed(w.finish());
    assert!(what.contains(r"\u{1b}"), "{what}");
    assert!(!what.contains('\u{1b}'), "{what:?}");
}

#[cfg(windows)]
#[test]
fn a_mapped_model_refuses_writers() {
    let path = std::env::temp_dir().join(format!("gguf-share-{}.gguf", std::process::id()));
    std::fs::write(&path, sample()).unwrap();
    let g = Gguf::open(&path).unwrap();
    let err = std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap_err();
    assert_eq!(err.raw_os_error(), Some(32), "{err}");
    // Readers are still welcome.
    assert!(std::fs::File::open(&path).is_ok());
    drop(g);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn more_than_65536_tensors_are_refused_before_allocating() {
    let mut b = sample();
    b[8..16].copy_from_slice(&65_537u64.to_le_bytes());
    let (_, what) = malformed(b);
    assert!(
        what.contains("tensor count") && what.contains("limit"),
        "{what}"
    );
}

#[test]
fn a_long_alignment_value_is_cut_in_the_error() {
    let mut w = gguf::write::Writer::new();
    w.meta("general.alignment", gguf::Value::String("x".repeat(5000)));
    let (_, what) = malformed(w.finish());
    assert!(what.len() < 400, "{} bytes", what.len());
}
