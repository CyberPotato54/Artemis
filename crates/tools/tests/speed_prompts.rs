//! The speed prompts are whole: the write, edit and explain prompts committed with llama.cpp's
//! ids, which our tokenizer reproduces, and every prompt's ids recorded in made-from.json, which `artemis speed` checks
//! each run against. The edit prompt quotes PSF-licensed text, committed in speed/sources/ with its licence.
use qwen36::{
    testing::fnv1a64,
    tokenizer::{Tokenizer, Vocab},
};
use serde_json::Value;
use std::fs;

const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/");

fn read(path: &str) -> Vec<u8> {
    fs::read(format!("{TESTDATA}{path}")).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn ids(path: &str) -> Vec<u32> {
    read(path)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

#[test]
fn the_committed_speed_prompts_tokenize_as_llama_cpp_did() {
    let tk =
        Tokenizer::new(&Vocab::from_extract(&read("tokenizer/qwen36-vocab.bin")).unwrap()).unwrap();
    for id in ["write", "edit", "explain"] {
        let text = String::from_utf8(read(&format!("speed/prompts/{id}.txt"))).unwrap();
        assert!(
            text.contains("<|im_start|>user"),
            "{id}: not a rendered chat prompt"
        );
        assert_eq!(
            tk.encode(&text, true),
            ids(&format!("speed/prompts/{id}.tokens")),
            "{id}"
        );
    }
}

#[test]
fn made_from_records_every_speed_prompt_and_matches_the_committed_ids() {
    let m: Value = serde_json::from_slice(&read("speed/made-from.json")).unwrap();
    for id in ["write", "edit", "explain"] {
        assert!(m[id]["tokens"].as_u64().unwrap_or(0) > 0, "{id}");
        assert_eq!(m[id]["fnv1a64"].as_str().map(str::len), Some(16), "{id}");
    }
    for id in ["write", "edit", "explain"] {
        let b = read(&format!("speed/prompts/{id}.tokens"));
        assert_eq!(m[id]["tokens"].as_u64(), Some(b.len() as u64 / 4), "{id}");
        assert_eq!(
            m[id]["fnv1a64"].as_str().unwrap(),
            format!("{:016x}", fnv1a64(&b)),
            "{id}"
        );
    }
    // The edit prompt quotes 9,000 characters of Python: thousands of tokens.
    assert!(m["edit"]["tokens"].as_u64().unwrap() > 2_000);
}

#[test]
fn the_prompts_are_the_spikes() {
    let p: Value = serde_json::from_slice(&read("speed/prompts.json")).unwrap();
    let ids: Vec<&str> = p["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["write", "edit", "explain"]);
    let s = &p["sampling"];
    assert_eq!(
        (s["temp"].as_f64(), s["top_p"].as_f64(), s["top_k"].as_u64()),
        (Some(0.6), Some(0.95), Some(20))
    );
    assert_eq!(s["max_tokens"].as_u64(), Some(700));
    assert_eq!(s["seeds"], serde_json::json!([1234, 1235]));
    let src = &p["sources"][0];
    assert_eq!(src["committed"], true);
    assert_eq!(src["path"], "speed/sources/encoder.py");
    assert_eq!(
        src["sha256"],
        "c541dfbf3bd73dcd11fbeae6f1aa133626bfb1161cc4a8a7b5b22b6543992e27"
    );
}
