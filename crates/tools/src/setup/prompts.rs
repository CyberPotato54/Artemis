//! The calibration's prompts: the speed gate's write and explain prompts and an edit prompt over the drafter's own
//! source, so the calibration measures speculation on the kinds of text it meets.
use qwen36::tokenizer::Tokenizer;
use server::template::Template;

/// The speed prompt `write` (`testdata/speed/prompts.json`, MIT OR Apache-2.0).
pub const WRITE: &str =
    "Write a Python module that implements an LRU cache with TTL expiry, thread safety, and a \
decorator form. Include docstrings and type hints. Code only.";

/// The speed prompt `explain` (`testdata/speed/prompts.json`, MIT OR Apache-2.0).
pub const EXPLAIN: &str =
    "Explain, for an experienced engineer, how speculative decoding with a multi-token-prediction \
head works and when it stops paying off. About 300 words.";

/// The edit prompt: a rename over `crates/engine/src/spec/lookup.rs`, which copies most of its text back.
pub fn edit() -> String {
    // `.gitattributes` keeps `*.rs` LF, so a checkout holds LF; the replace covers a source archive unpacked with
    // CRLF conversion, so the prompt (and its token count) is the same text either way.
    let source = include_str!("../../../engine/src/spec/lookup.rs").replace("\r\n", "\n");
    format!(
        "Here is a file. Return the COMPLETE file unchanged except: rename the struct Lookup to PromptLookup \
         everywhere. No commentary.\n\n```rust\n{}\n```",
        source.trim_end_matches('\n')
    )
}

/// `text` as one user message through the chat template, with the generation prompt and thinking off.
pub fn render_text(t: &Template, text: &str) -> Result<String, String> {
    t.render_json(&serde_json::json!({
        "messages": [{"role": "user", "content": text}],
        "add_generation_prompt": true,
        "enable_thinking": false,
    }))
    .map_err(|e| format!("rendering a calibration prompt: {}", e.message))
}

/// `render_text`, encoded with the special tokens parsed.
pub fn render(t: &Template, tok: &Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    let rendered = render_text(t, text)?;
    tok.try_encode(&rendered, true)
        .map_err(|e| format!("encoding a calibration prompt: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen36::tokenizer::Vocab;

    const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/");

    fn read(path: &str) -> Vec<u8> {
        std::fs::read(format!("{TESTDATA}{path}")).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    fn template() -> Template {
        let src = String::from_utf8(read("template/qwen36-chat-template.jinja")).unwrap();
        Template::new(src).unwrap()
    }

    fn tokenizer() -> Tokenizer {
        Tokenizer::new(&Vocab::from_extract(&read("tokenizer/qwen36-vocab.bin")).unwrap()).unwrap()
    }

    fn ids(path: &str) -> Vec<u32> {
        read(path)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()
    }

    fn check_speed_prompt(id: &str, text: &str) {
        let t = template();
        let want = String::from_utf8(read(&format!("speed/prompts/{id}.txt"))).unwrap();
        assert_eq!(render_text(&t, text).unwrap(), want, "{id}: text");
        assert_eq!(
            render(&t, &tokenizer(), text).unwrap(),
            ids(&format!("speed/prompts/{id}.tokens")),
            "{id}: ids"
        );
    }

    #[test]
    fn the_write_prompt_renders_as_the_speed_prompt() {
        check_speed_prompt("write", WRITE);
    }

    #[test]
    fn the_explain_prompt_renders_as_the_speed_prompt() {
        check_speed_prompt("explain", EXPLAIN);
    }

    #[test]
    fn the_prompt_texts_are_the_speed_prompts_json() {
        let p: serde_json::Value = serde_json::from_slice(&read("speed/prompts.json")).unwrap();
        let text = |id: &str| {
            let item = p["prompts"]
                .as_array()
                .unwrap()
                .iter()
                .find(|x| x["id"] == id)
                .unwrap();
            item["content"][0]["text"].as_str().unwrap().to_string()
        };
        assert_eq!(WRITE, text("write"));
        assert_eq!(EXPLAIN, text("explain"));
    }

    #[test]
    fn the_edit_prompt_renders_to_1500_to_4000_tokens() {
        let e = edit();
        assert!(e.starts_with(
            "Here is a file. Return the COMPLETE file unchanged except: rename the struct Lookup to PromptLookup \
             everywhere. No commentary."
        ));
        assert!(e.contains("pub struct Lookup"));
        assert!(e.ends_with("\n```"));
        let n = render(&template(), &tokenizer(), &e).unwrap().len();
        assert!((1500..=4000).contains(&n), "{n} tokens");
    }
}
