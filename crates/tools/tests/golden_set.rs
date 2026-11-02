//! The golden set (test 3's inputs) is whole, and our tokenizer gives llama.cpp b11509's ids for every
//! prompt: 20 prompts in the set's categories, each with its rendered text, llama.cpp's token ids, llama.cpp's
//! greedy output for those ids, and llama.cpp CPU's run fed that output's tokens (test 3's floor).
use qwen36::{
    testing::fnv1a64,
    tokenizer::{Tokenizer, Vocab},
};
use serde_json::Value;
use std::{collections::BTreeMap, fs};

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

fn prompts() -> Vec<Value> {
    let p: Value = serde_json::from_slice(&read("golden/prompts.json")).unwrap();
    p["prompts"].as_array().unwrap().clone()
}

#[test]
fn the_golden_set_holds_the_specs_twenty_prompts() {
    let ps = prompts();
    let mut by_category = BTreeMap::new();
    for p in &ps {
        *by_category
            .entry(p["category"].as_str().unwrap())
            .or_insert(0) += 1;
    }
    assert_eq!(
        by_category,
        BTreeMap::from([
            ("code", 8),
            ("long", 1),
            ("multilingual", 2),
            ("prose", 5),
            ("tool use", 4)
        ])
    );
    for p in &ps {
        let id = p["id"].as_str().unwrap();
        let tokens = ids(&format!("golden/prompts/{id}.tokens"));
        let e: Value =
            serde_json::from_slice(&read(&format!("golden/expected/{id}.json"))).unwrap();
        assert_eq!(
            e["prompt_tokens"].as_u64(),
            Some(tokens.len() as u64),
            "{id}: expected/{id}.json was made from other tokens"
        );
        assert_eq!(e["backend"], "vulkan", "{id}");
        let steps = e["steps"].as_array().unwrap().len();
        assert!((1..=256).contains(&steps), "{id}: {steps} steps");
        if p["category"] == "long" {
            assert!(
                (25_000..=32_000).contains(&tokens.len()),
                "{id}: {} tokens, not about 30K",
                tokens.len()
            );
        }
    }
    for id in ["01-code", "02-prose", "03-chat", "04-multilingual"] {
        assert_eq!(
            read(&format!("golden/prompts/{id}.txt")),
            read(&format!("test2/inputs/{id}.txt")),
            "{id} is test 2's input"
        );
        assert_eq!(
            ids(&format!("golden/prompts/{id}.tokens")),
            ids(&format!("test2/inputs/{id}.tokens")),
            "{id}"
        );
    }
}

#[test]
fn our_tokenizer_gives_llama_cpps_ids_for_every_golden_prompt() {
    let tk =
        Tokenizer::new(&Vocab::from_extract(&read("tokenizer/qwen36-vocab.bin")).unwrap()).unwrap();
    for p in prompts() {
        let id = p["id"].as_str().unwrap();
        let text = String::from_utf8(read(&format!("golden/prompts/{id}.txt"))).unwrap();
        let (got, want) = (
            tk.encode(&text, true),
            ids(&format!("golden/prompts/{id}.tokens")),
        );
        if let Some(i) = (0..got.len().max(want.len())).find(|&i| got.get(i) != want.get(i)) {
            panic!(
                "{id}: first difference at token {i} of {} (llama.cpp {}): ours {:?}, llama.cpp {:?}",
                got.len(),
                want.len(),
                got.get(i),
                want.get(i)
            );
        }
    }
}

#[test]
fn the_floor_follows_the_oracles_tokens() {
    let tokens = |v: &Value| -> Vec<u64> {
        v["steps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["token"].as_u64().unwrap())
            .collect()
    };
    for p in prompts() {
        let id = p["id"].as_str().unwrap();
        let json = |dir: &str| -> Value {
            serde_json::from_slice(&read(&format!("golden/{dir}/{id}.json"))).unwrap()
        };
        let (e, f) = (json("expected"), json("forced-cpu"));
        assert_eq!(f["backend"], "cpu", "{id}");
        assert_eq!(
            f["prompt_tokens"], e["prompt_tokens"],
            "{id}: the floor was made from another prompt"
        );
        assert_eq!(
            tokens(&f),
            tokens(&e),
            "{id}: the floor wasn't fed expected/{id}.json's tokens: rerun build_golden_set.py --stage floor"
        );
    }
}

#[test]
fn golden_outputs_were_made_from_the_committed_tokens() {
    let hash = |path: &str| format!("{:016x}", fnv1a64(&read(path)));
    let made: Value = serde_json::from_slice(&read("golden/made-from.json")).unwrap();
    for p in prompts() {
        let id = p["id"].as_str().unwrap();
        let (tokens, expected) = (
            hash(&format!("golden/prompts/{id}.tokens")),
            hash(&format!("golden/expected/{id}.json")),
        );
        let m = &made[id];
        assert_eq!(
            m["expected"]["prompt"], tokens,
            "expected/{id}.json was made from other tokens than prompts/{id}.tokens; rerun --stage expected --only {id}"
        );
        assert_eq!(
            m["floor"]["prompt"], tokens,
            "forced-cpu/{id}.json was made from other tokens than prompts/{id}.tokens; rerun --stage floor --only {id}"
        );
        assert_eq!(
            m["floor"]["expected"], expected,
            "forced-cpu/{id}.json was made from another expected/{id}.json; rerun --stage floor --only {id}"
        );
    }
}
