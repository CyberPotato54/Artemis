//! Streaming detokenized text. A byte-level BPE token can end inside a UTF-8 character; the tokenizer then decodes the
//! partial bytes as U+FFFD until the next token completes them. `TextStream` holds a trailing U+FFFD back so it never
//! prints a character it will have to take back.
#[derive(Default)]
pub struct TextStream {
    /// Bytes of the current window's text already printed.
    printed: usize,
    /// The first token of the window: tokens before it are printed whole (the text ended on a character boundary).
    start: usize,
}

impl TextStream {
    pub fn new() -> TextStream {
        TextStream::default()
    }

    /// Given the decoded text of every generated token so far, the part not yet printed, minus a trailing run of
    /// U+FFFD that the next token may complete.
    pub fn next<'a>(&mut self, full: &'a str) -> &'a str {
        let end = full.trim_end_matches('\u{FFFD}').len();
        if end <= self.printed || !full.is_char_boundary(self.printed) {
            return "";
        }
        let out = &full[self.printed..end];
        self.printed = end;
        out
    }

    /// Like `next`, from the generated token ids, decoding only the tokens since the last complete character rather
    /// than all of them each time. The text is byte for byte what `next(&decode(ids))` would give: byte-level BPE
    /// decoding is the lossy UTF-8 of the tokens' bytes in order, so a window that starts on a character boundary
    /// decodes to the tail of the whole text. `ids` must only grow between calls.
    pub fn next_tokens(&mut self, ids: &[u32], decode: impl Fn(&[u32]) -> String) -> String {
        let window = decode(&ids[self.start..]);
        let out = self.next(&window).to_string();
        if !window.ends_with('\u{FFFD}') {
            // The window ends on a boundary and is all printed: the next one starts after it.
            self.start = ids.len();
            self.printed = 0;
        }
        out
    }

    /// Like `finish`, from the token ids.
    pub fn finish_tokens(&mut self, ids: &[u32], decode: impl Fn(&[u32]) -> String) -> String {
        let window = decode(&ids[self.start..]);
        self.finish(&window).to_string()
    }

    /// The rest at the end of generation, replacement characters included.
    pub fn finish<'a>(&mut self, full: &'a str) -> &'a str {
        let from = if full.is_char_boundary(self.printed) {
            self.printed.min(full.len())
        } else {
            full.len()
        };
        self.printed = full.len();
        &full[from..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_split_character_waits_for_its_last_byte() {
        let mut s = TextStream::new();
        assert_eq!(s.next("h"), "h");
        assert_eq!(s.next("h\u{FFFD}"), "");
        assert_eq!(s.next("hé"), "é");
        assert_eq!(s.next("héllo"), "llo");
        assert_eq!(s.finish("héllo"), "");
    }

    #[test]
    fn a_dangling_partial_character_is_printed_at_the_end() {
        let mut s = TextStream::new();
        assert_eq!(s.next("ok\u{FFFD}\u{FFFD}"), "ok");
        assert_eq!(s.finish("ok\u{FFFD}\u{FFFD}"), "\u{FFFD}\u{FFFD}");
    }
}
