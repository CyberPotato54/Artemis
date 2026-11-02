//! The token ids the server needs by name.
use qwen36::tokenizer::{Tokenizer, Vocab, CONTROL};

/// The model's special token ids, found by their text.
#[derive(Clone, Debug, PartialEq)]
pub struct Specials {
    pub im_start: u32,
    pub im_end: u32,
    pub think: u32,
    pub end_think: u32,
    pub tool_call: u32,
    pub end_tool_call: u32,
    /// Every end-of-generation token.
    pub eog: Vec<u32>,
    /// What a forced close of the reasoning is tokenized as: `\n</think>\n\n`.
    pub think_close: Vec<u32>,
    /// The text of every vocabulary token of type CONTROL: what `special_tokens_in_content = false` keeps out of
    /// client text (the six USER_DEFINED tags are not among them).
    pub control_texts: Vec<String>,
}

impl Specials {
    pub fn new(v: &Vocab, tk: &Tokenizer) -> Result<Specials, String> {
        let id = |text: &str| {
            v.tokens
                .iter()
                .position(|t| t == text)
                .map(|i| i as u32)
                .ok_or_else(|| format!("the vocabulary has no {text} token"))
        };
        Ok(Specials {
            im_start: id("<|im_start|>")?,
            im_end: id("<|im_end|>")?,
            think: id("<think>")?,
            end_think: id("</think>")?,
            tool_call: id("<tool_call>")?,
            end_tool_call: id("</tool_call>")?,
            eog: v.eog(),
            think_close: tk.encode("\n</think>\n\n", true),
            control_texts: v
                .tokens
                .iter()
                .zip(&v.types)
                .filter(|(_, &ty)| ty == CONTROL)
                .map(|(t, _)| t.clone())
                .collect(),
        })
    }
}
