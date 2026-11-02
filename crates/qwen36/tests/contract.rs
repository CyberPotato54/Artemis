//! `validate` accepts exactly the files the model contract describes, and names the first difference otherwise.
use gguf::{write::Writer, Gguf, Value};
use qwen36::{
    contract::{expected_meta, expected_tensors, ContractError},
    dims::Dims,
    model::{Mixer, Model},
    quant::WType,
};

/// A zero-filled file that satisfies the contract for `d`, after `edit` changes its metadata or tensors.
/// Matrices default to Q8_0 (experts Q4_K) and everything else to its first allowed type.
fn file(
    d: &Dims,
    edit: impl FnOnce(&mut Vec<(String, Value)>, &mut Vec<(String, Vec<u64>, WType)>),
) -> Vec<u8> {
    let mut meta = expected_meta(d);
    meta.push((
        "tokenizer.ggml.tokens".into(),
        Value::Array(vec![Value::String("x".into()); d.n_vocab]),
    ));
    let mut tensors: Vec<(String, Vec<u64>, WType)> = expected_tensors(d)
        .into_iter()
        .map(|e| {
            let ty = if e.name.ends_with("_exps.weight") {
                WType::Q4K
            } else if e.allowed.contains(&WType::Q8_0) {
                WType::Q8_0
            } else {
                e.allowed[0]
            };
            (e.name, e.dims, ty)
        })
        .collect();
    edit(&mut meta, &mut tensors);
    let mut w = Writer::new();
    for (k, v) in meta {
        w.meta(&k, v);
    }
    for (name, dims, ty) in tensors {
        let n: u64 = dims[1..].iter().product();
        let bytes = ty.row_bytes(dims[0] as usize) * n as usize;
        w.tensor(&name, &dims, ty.ggml(), vec![0; bytes]);
    }
    w.finish()
}

fn open_err(bytes: Vec<u8>) -> ContractError {
    let g = Gguf::from_bytes(bytes).unwrap();
    match Model::open(&g, &Dims::tiny()) {
        Err(e) => e,
        Ok(_) => panic!("the contract accepted the file"),
    }
}

fn set(meta: &mut [(String, Value)], key: &str, v: Value) {
    meta.iter_mut().find(|(k, _)| k == key).unwrap().1 = v;
}

#[test]
fn a_valid_file_opens_with_its_roles() {
    let d = Dims::tiny();
    let g = Gguf::from_bytes(file(&d, |_, _| {})).unwrap();
    let m = Model::open(&g, &d).unwrap();
    assert_eq!(m.layers.len(), 4);
    let kinds: Vec<bool> = m
        .layers
        .iter()
        .map(|l| matches!(l.mixer, Mixer::Attention(_)))
        .collect();
    assert_eq!(kinds, [false, false, false, true]);
    assert!(matches!(m.mtp.layer.mixer, Mixer::Attention(_)));
    assert_eq!((m.embd.cols, m.embd.rows), (256, 512));
    let Mixer::DeltaNet(dn) = &m.layers[0].mixer else {
        unreachable!()
    };
    assert_eq!((dn.qkv.cols, dn.qkv.rows), (256, 1024));
    assert_eq!(
        (
            m.layers[0].moe.down_exps.cols,
            m.layers[0].moe.down_exps.rows
        ),
        (256, 256 * 16)
    );
    let e3 = m.layers[0].moe.gate_exps.expert(3, 256);
    assert_eq!((e3.rows, e3.bytes.len()), (256, 256 * 144));
}

#[test]
fn wrong_architecture() {
    let e = open_err(file(&Dims::tiny(), |m, _| {
        set(m, "general.architecture", Value::String("llama".into()))
    }));
    assert_eq!(
        e.to_string(),
        "metadata general.architecture: expected \"qwen35moe\", found \"llama\""
    );
}

#[test]
fn wrong_hyperparameter() {
    let e = open_err(file(&Dims::tiny(), |m, _| {
        set(m, "qwen35moe.expert_count", Value::U32(32))
    }));
    assert_eq!(
        e.to_string(),
        "metadata qwen35moe.expert_count: expected 16, found 32"
    );
}

#[test]
fn missing_metadata() {
    let e = open_err(file(&Dims::tiny(), |m, _| {
        m.retain(|(k, _)| k != "qwen35moe.ssm.inner_size")
    }));
    assert_eq!(
        e.to_string(),
        "metadata qwen35moe.ssm.inner_size: expected 512, found missing"
    );
}

#[test]
fn integer_types_are_coerced() {
    let d = Dims::tiny();
    let g = Gguf::from_bytes(file(&d, |m, _| {
        set(m, "qwen35moe.block_count", Value::I32(5))
    }))
    .unwrap();
    Model::open(&g, &d).unwrap();
}

#[test]
fn vocabulary_size() {
    let e = open_err(file(&Dims::tiny(), |m, _| {
        set(
            m,
            "tokenizer.ggml.tokens",
            Value::Array(vec![Value::String("x".into()); 7]),
        )
    }));
    assert_eq!(
        e.to_string(),
        "metadata tokenizer.ggml.tokens: expected 512 entries, found 7 entries"
    );
}

#[test]
fn missing_tensor() {
    let e = open_err(file(&Dims::tiny(), |_, t| {
        t.retain(|(n, _, _)| n != "blk.2.ssm_a")
    }));
    assert_eq!(e, ContractError::MissingTensor("blk.2.ssm_a".into()));
    assert_eq!(e.to_string(), "missing tensor blk.2.ssm_a");
}

#[test]
fn unexpected_tensor() {
    let e = open_err(file(&Dims::tiny(), |_, t| {
        t.push(("blk.0.bogus.weight".into(), vec![256], WType::F32))
    }));
    assert_eq!(
        e,
        ContractError::UnexpectedTensor("blk.0.bogus.weight".into())
    );
}

#[test]
fn wrong_shape() {
    let e = open_err(file(&Dims::tiny(), |_, t| {
        t.iter_mut()
            .find(|(n, _, _)| n == "blk.0.attn_qkv.weight")
            .unwrap()
            .1 = vec![256, 512];
    }));
    assert_eq!(
        e.to_string(),
        "tensor blk.0.attn_qkv.weight: shape expected [256, 1024], found [256, 512]"
    );
}

#[test]
fn wrong_type() {
    let e = open_err(file(&Dims::tiny(), |_, t| {
        t.iter_mut()
            .find(|(n, _, _)| n == "blk.1.attn_norm.weight")
            .unwrap()
            .2 = WType::Q8_0;
    }));
    assert_eq!(
        e.to_string(),
        "tensor blk.1.attn_norm.weight: type Q8_0 is not one of [F32]"
    );
}

#[test]
fn accepts_any_allowed_type_per_role() {
    // Experts in Q8_0, projections in F32, the output in BF16: a different quant mix of the same model.
    let d = Dims::tiny();
    let g = Gguf::from_bytes(file(&d, |_, t| {
        for (name, _, ty) in t.iter_mut() {
            if name.ends_with("_exps.weight") {
                *ty = WType::Q8_0;
            } else if name.ends_with("attn_qkv.weight") || name.ends_with("ssm_out.weight") {
                *ty = WType::F32;
            } else if name == "output.weight" {
                *ty = WType::Bf16;
            }
        }
    }))
    .unwrap();
    let m = Model::open(&g, &d).unwrap();
    assert_eq!(m.layers[0].moe.gate_exps.ty, WType::Q8_0);
    assert_eq!(m.output.ty, WType::Bf16);
}

#[test]
fn layer_kinds_follow_the_interval() {
    let d = Dims::QWEN36;
    let attention: Vec<usize> = (0..41).filter(|&l| d.is_attention(l)).collect();
    assert_eq!(attention, [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 40]);
    assert_eq!(
        (d.key_dim(), d.conv_dim(), d.v_head_dim()),
        (2048, 8192, 128)
    );
    assert_eq!(expected_tensors(&d).len(), 753);
}

#[test]
fn a_file_without_the_mtp_layer_names_the_mtp_repository() {
    // The non-MTP repository's files: 4 blocks, no nextn layer.
    let e = open_err(file(&Dims::tiny(), |m, _| {
        set(m, "qwen35moe.block_count", Value::U32(4))
    }));
    assert_eq!(
        e,
        ContractError::NoMtpLayer {
            key: "qwen35moe.block_count".into(),
            found: "4".into(),
            expected: "5".into()
        }
    );
    assert!(e.to_string().contains("-MTP-GGUF"), "{e}");
}

#[test]
fn missing_nextn_predict_layers_is_no_mtp_layer() {
    let e = open_err(file(&Dims::tiny(), |m, _| {
        m.retain(|(k, _)| k != "qwen35moe.nextn_predict_layers")
    }));
    assert!(matches!(e, ContractError::NoMtpLayer { .. }), "{e}");
    assert!(
        e.to_string().contains("nextn_predict_layers is missing"),
        "{e}"
    );
    let e = open_err(file(&Dims::tiny(), |m, _| {
        set(m, "qwen35moe.nextn_predict_layers", Value::U32(0))
    }));
    assert!(matches!(e, ContractError::NoMtpLayer { .. }), "{e}");
    assert!(e.to_string().contains("nextn_predict_layers is 0"), "{e}");
}

#[test]
fn an_unexpected_tensor_name_is_escaped() {
    let e = open_err(file(&Dims::tiny(), |_, t| {
        t.push(("x\u{1b}[2K".into(), vec![256], WType::F32))
    }));
    let text = e.to_string();
    assert!(text.contains(r"\u{1b}"), "{text}");
    assert!(!text.contains('\u{1b}'), "{text:?}");
}

#[test]
fn another_architecture_gets_the_architecture_mismatch_not_the_mtp_hint() {
    let mut w = Writer::new();
    w.meta("general.architecture", Value::String("clip".into()));
    let g = Gguf::from_bytes(w.finish()).unwrap();
    let Err(e) = Model::open(&g, &Dims::tiny()) else {
        panic!("the contract accepted a clip file")
    };
    assert!(
        matches!(&e, ContractError::Meta { key, .. } if key == "general.architecture"),
        "{e}"
    );
}

#[test]
fn a_crafted_value_is_cut_in_the_error() {
    let e = open_err(file(&Dims::tiny(), |m, _| {
        set(
            m,
            "general.architecture",
            Value::String("y".repeat(100_000)),
        )
    }));
    assert!(e.to_string().len() < 500, "{} bytes", e.to_string().len());
}
