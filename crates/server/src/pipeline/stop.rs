//! Stop strings: matched on the content text as it arrives, holding back the longest suffix that could
//! still become a stop string, so the pieces' boundaries never change where the text is cut.

/// The matcher of one request's stop strings.
pub struct StopMatcher {
    stops: Vec<String>,
    /// Text received and not released: always a suffix-prefix of some stop string, never a match.
    held: String,
    /// A stop string matched: nothing more is released.
    done: bool,
}

impl StopMatcher {
    /// Empty stop strings are ignored (the request handler logs them).
    pub fn new(stops: Vec<String>) -> StopMatcher {
        StopMatcher {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            held: String::new(),
            done: false,
        }
    }

    /// Text in: `(text released, matched stop)`. On a match the text before it is released and the rest is dropped;
    /// the matcher then releases nothing. Of several stop strings that match, the one that ends first wins (the
    /// longer, if they end together), so the outcome doesn't depend on how the text is cut into pieces.
    pub fn push(&mut self, s: &str) -> (String, Option<String>) {
        if self.done {
            return (String::new(), None);
        }
        if self.stops.is_empty() {
            return (s.to_string(), None);
        }
        self.held.push_str(s);
        // The earliest end, then the earliest start. Each stop's first occurrence is its earliest end.
        let hit = self
            .stops
            .iter()
            .filter_map(|st| self.held.find(st.as_str()).map(|i| (i + st.len(), i, st)))
            .min_by_key(|&(end, start, _)| (end, start));
        if let Some((_, start, stop)) = hit {
            let matched = stop.clone();
            self.held.truncate(start);
            self.done = true;
            return (std::mem::take(&mut self.held), Some(matched));
        }
        let keep = self.hold_len();
        let cut = self.held.len() - keep;
        let rest = self.held.split_off(cut);
        (std::mem::replace(&mut self.held, rest), None)
    }

    /// The end of the text: what is held back is not a stop string and is released.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    /// The length of the longest suffix of the held text that is a proper prefix of a stop string. A suffix that
    /// equals a prefix of valid UTF-8 starts on a character boundary, so the cut is always one.
    fn hold_len(&self) -> usize {
        self.stops
            .iter()
            .filter_map(|st| {
                (1..st.len())
                    .rev()
                    .filter(|&n| st.is_char_boundary(n) && n <= self.held.len())
                    .find(|&n| self.held.ends_with(&st[..n]))
            })
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops(s: &[&str]) -> StopMatcher {
        StopMatcher::new(s.iter().map(|s| s.to_string()).collect())
    }

    /// splitmix64, as the tool-call tests have it.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    fn chunks(text: &str, seed: u64) -> Vec<&str> {
        let mut rng = Rng(seed);
        let denom = 2u64 << (seed % 4);
        let mut out = Vec::new();
        let mut start = 0;
        for (i, c) in text.char_indices() {
            let end = i + c.len_utf8();
            if end < text.len() && rng.next().is_multiple_of(denom) {
                out.push(&text[start..end]);
                start = end;
            }
        }
        out.push(&text[start..]);
        out
    }

    /// What `text` gives when pushed in `pieces`, flushed at the end: the released text and the match.
    fn run(m: &mut StopMatcher, pieces: &[&str]) -> (String, Option<String>) {
        let mut text = String::new();
        let mut matched = None;
        for p in pieces {
            let (t, hit) = m.push(p);
            text.push_str(&t);
            if hit.is_some() {
                matched = hit;
            }
        }
        text.push_str(&m.flush());
        (text, matched)
    }

    #[test]
    fn a_stop_string_across_pieces_cuts_the_text_before_it() {
        let mut m = stops(&["STOP"]);
        assert_eq!(m.push("abc ST"), ("abc ".to_string(), None));
        assert_eq!(m.push("OP def"), (String::new(), Some("STOP".to_string())));
        assert_eq!(m.push("more"), (String::new(), None));
        assert_eq!(m.flush(), "");
    }

    #[test]
    fn a_false_start_is_released_with_the_next_text() {
        let mut m = stops(&["STOP"]);
        assert_eq!(m.push("a ST"), ("a ".to_string(), None));
        assert_eq!(m.push("AR"), ("STAR".to_string(), None));
        assert_eq!(m.push("S"), (String::new(), None));
        assert_eq!(m.flush(), "S");
    }

    #[test]
    fn what_is_held_at_the_end_is_released_by_flush() {
        let mut m = stops(&["STOP"]);
        assert_eq!(m.push("ST"), (String::new(), None));
        assert_eq!(m.flush(), "ST");
        assert_eq!(m.flush(), "");
    }

    #[test]
    fn no_stops_releases_everything() {
        let mut m = stops(&[]);
        assert_eq!(m.push("a ST"), ("a ST".to_string(), None));
        assert_eq!(m.flush(), "");
    }

    #[test]
    fn an_empty_stop_string_is_ignored() {
        let mut m = stops(&["", "x"]);
        assert_eq!(m.push("ab"), ("ab".to_string(), None));
        assert_eq!(m.push("x"), (String::new(), Some("x".to_string())));
    }

    #[test]
    fn the_stop_that_ends_first_wins_however_the_text_is_cut() {
        // "ab" ends before "xabc" does, so it wins in one piece as it does one character at a time.
        for text in ["xabc", "yxabcz"] {
            let mut m = stops(&["xabc", "ab"]);
            let want = (
                text[..text.find("ab").unwrap()].to_string(),
                Some("ab".to_string()),
            );
            assert_eq!(run(&mut m, &[text]), want);
            let each: Vec<String> = text.chars().map(String::from).collect();
            let each: Vec<&str> = each.iter().map(String::as_str).collect();
            assert_eq!(run(&mut stops(&["xabc", "ab"]), &each), want);
        }
        // Ending together: the longer one, which starts first.
        let mut m = stops(&["b", "ab"]);
        assert_eq!(
            run(&mut m, &["xab"]),
            ("x".to_string(), Some("ab".to_string()))
        );
    }

    #[test]
    fn every_cut_gives_the_same_text_and_match() {
        let cases: [(&[&str], &str, &str, Option<&str>); 5] = [
            (&["STOP"], "one STOP two", "one ", Some("STOP")),
            (&["。", "终止"], "你好。再见", "你好", Some("。")),
            (&["。", "终止"], "好的终止吧", "好的", Some("终止")),
            (&["ab", "bc"], "xabcx", "x", Some("ab")),
            (
                &["STOP"],
                "no S T O P here, ST",
                "no S T O P here, ST",
                None,
            ),
        ];
        for (s, text, want, hit) in cases {
            for seed in 0..100 {
                let got = run(&mut stops(s), &chunks(text, seed));
                assert_eq!(
                    got,
                    (want.to_string(), hit.map(String::from)),
                    "seed {seed}"
                );
            }
        }
    }

    #[test]
    fn the_hold_back_is_exactly_the_longest_possible_prefix() {
        let mut m = stops(&["abcd", "cx"]);
        // "ab" is a prefix of abcd, "abc" too: hold "abc", release "z".
        assert_eq!(m.push("zabc"), ("z".to_string(), None));
        // "abcc" ends in "c", which starts "cx": release "abc".
        assert_eq!(m.push("c"), ("abc".to_string(), None));
        assert_eq!(m.push("x"), (String::new(), Some("cx".to_string())));
    }
}
