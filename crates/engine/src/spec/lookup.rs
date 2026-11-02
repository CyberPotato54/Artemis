//! Prompt lookup: draft what followed the last time the history ended the way it does now.
use super::{Drafter, LookupMatch};
use std::collections::HashMap;

/// The request's tokens and an index from each n-gram (`ngram_min` to `ngram_max` tokens) to the position after its
/// most recent occurrence that has a follower. Token ids only, in pageable memory.
///
/// The key is a 64-bit hash of the n-gram and its length; a lookup checks the tokens against the history, so a hash
/// collision costs a missed draft and never a wrong one.
pub struct Lookup {
    ngram_min: usize,
    ngram_max: usize,
    history: Vec<u32>,
    follower: HashMap<u64, usize>,
    /// The last draft's source position and the history's length when it was made; `None` after an empty draft.
    last_run: Option<(usize, usize)>,
    /// What produced the last draft; `None` after an empty draft.
    last: Option<LookupMatch>,
}

fn key(gram: &[u32]) -> u64 {
    // FNV-1a over the length and the tokens.
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ gram.len() as u64;
    h = h.wrapping_mul(0x0000_0100_0000_01b3);
    for &t in gram {
        h = (h ^ u64::from(t)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl Lookup {
    /// `ngram_min` and `ngram_max` must pass `SpecConfig::check`.
    pub fn new(ngram_min: usize, ngram_max: usize) -> Lookup {
        assert!(
            1 <= ngram_min && ngram_min <= ngram_max,
            "lookup n-gram sizes {ngram_min}..{ngram_max}"
        );
        Lookup {
            ngram_min,
            ngram_max,
            history: Vec::new(),
            follower: HashMap::new(),
            last_run: None,
            last: None,
        }
    }

    /// Tokens seen so far.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Distinct n-grams indexed.
    pub fn index_entries(&self) -> usize {
        self.follower.len()
    }
}

impl Drafter for Lookup {
    fn name(&self) -> &'static str {
        "lookup"
    }

    fn draft(&mut self, max: usize) -> Vec<u32> {
        let len = self.history.len();
        let run = self.last_run.take();
        self.last = None;
        for n in (self.ngram_min..=self.ngram_max).rev() {
            if max == 0 || len < n {
                continue;
            }
            let suffix = &self.history[len - n..];
            // The index holds only n-grams that already have a follower, so this is an earlier occurrence.
            let Some(&at) = self.follower.get(&key(suffix)) else {
                continue;
            };
            // `at < n` can only come from a hash collision with a shorter n-gram's entry.
            if at < n || self.history[at - n..at] != *suffix {
                continue;
            }
            // The same run when the source has moved on by exactly the tokens committed since the last draft.
            let continues = run.is_some_and(|(was_at, was_len)| at == was_at + (len - was_len));
            self.last_run = Some((at, len));
            self.last = Some(LookupMatch { n, continues });
            // Cut at the history's end: a run of one token drafts one token.
            return self.history[at..(at + max).min(len)].to_vec();
        }
        Vec::new()
    }

    fn last_match(&self) -> Option<LookupMatch> {
        self.last
    }

    fn observe(&mut self, tokens: &[u32]) {
        for &t in tokens {
            // The new token follows the n-grams that end just before it.
            let at = self.history.len();
            self.history.push(t);
            for n in self.ngram_min..=self.ngram_max {
                if at >= n {
                    self.follower.insert(key(&self.history[at - n..at]), at);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup_over(history: &[u32]) -> Lookup {
        let mut l = Lookup::new(2, 4);
        l.observe(history);
        l
    }

    #[test]
    fn lookup_drafts_what_followed_the_longest_earlier_match() {
        // The 3-gram [1,2,3] is at 0; nothing earlier matches [9,1,2,3].
        let mut l = lookup_over(&[1, 2, 3, 4, 9, 1, 2, 3]);
        assert_eq!(l.draft(4), vec![4, 9, 1, 2]);
    }

    #[test]
    fn the_most_recent_occurrence_wins() {
        let mut l = lookup_over(&[5, 6, 7, 5, 6, 8, 5, 6]);
        assert_eq!(l.draft(2), vec![8, 5]);
    }

    #[test]
    fn no_match_drafts_nothing() {
        let mut l = lookup_over(&[1, 2, 3, 4]);
        assert!(l.draft(4).is_empty());
        let mut short = lookup_over(&[1]);
        assert!(short.draft(4).is_empty());
        let mut empty = lookup_over(&[]);
        assert!(empty.draft(4).is_empty());
    }

    #[test]
    fn a_zero_max_drafts_nothing() {
        let mut l = lookup_over(&[1, 2, 3, 4, 9, 1, 2, 3]);
        assert!(l.draft(0).is_empty());
    }

    #[test]
    fn a_run_of_one_token_drafts_one_token_and_the_index_stays_bounded() {
        assert!(lookup_over(&[7, 7]).draft(4).is_empty());
        assert_eq!(lookup_over(&[7, 7, 7]).draft(4), vec![7]);
        let mut l = lookup_over(&vec![7; 10_000]);
        assert_eq!(l.draft(4), vec![7]);
        assert_eq!(l.len(), 10_000);
        assert!(l.index_entries() <= 4 - 2 + 1, "{}", l.index_entries());
    }

    #[test]
    fn the_draft_is_cut_at_the_end_of_the_history() {
        // [1,2] occurs at 0 and is followed by 3 and then the suffix itself.
        let mut l = lookup_over(&[1, 2, 3, 1, 2]);
        assert_eq!(l.draft(4), vec![3, 1, 2]);
    }

    #[test]
    fn observe_one_at_a_time_equals_observe_at_once() {
        let stream: Vec<u32> = (0..300u32).map(|i| (i * i + i / 3) % 7).collect();
        let mut once = Lookup::new(2, 4);
        once.observe(&stream);
        let mut single = Lookup::new(2, 4);
        for t in &stream {
            single.observe(std::slice::from_ref(t));
        }
        assert_eq!(once.len(), single.len());
        assert_eq!(once.index_entries(), single.index_entries());
        assert_eq!(once.draft(4), single.draft(4));
        // The drafts agree at every prefix too.
        let mut a = Lookup::new(2, 4);
        let mut b = Lookup::new(2, 4);
        for (i, t) in stream.iter().enumerate() {
            a.observe(&[*t]);
            if i % 5 == 4 {
                b.observe(&stream[i - 4..=i]);
                assert_eq!(a.draft(4), b.draft(4), "at {i}");
            }
        }
    }

    #[test]
    fn a_draft_is_a_real_continuation_of_the_history() {
        // Whatever it drafts, the tokens are the ones that followed an earlier copy of the suffix.
        let stream: Vec<u32> = (0..500u32).map(|i| (i * 7 + i / 11) % 5).collect();
        let mut l = Lookup::new(2, 4);
        let mut drafted = 0;
        for (i, t) in stream.iter().enumerate() {
            l.observe(&[*t]);
            let d = l.draft(4);
            if d.is_empty() {
                continue;
            }
            drafted += 1;
            let hist = &stream[..=i];
            let found = (1..hist.len()).any(|p| {
                hist[p..].len() >= d.len()
                    && hist[p..p + d.len()] == d[..]
                    && (2..=4).any(|n| p >= n && hist[p - n..p] == hist[hist.len() - n..])
            });
            assert!(found, "draft {d:?} at {i}");
        }
        assert!(drafted > 100, "{drafted}");
    }

    #[test]
    fn a_colliding_entry_near_the_start_is_skipped_not_a_crash() {
        // Plant what a hash collision would: the 4-gram suffix's key pointing at a follower position under 4.
        let mut l = lookup_over(&[1, 2, 3, 4, 5, 6]);
        let suffix_key = key(&[3, 4, 5, 6]);
        l.follower.insert(suffix_key, 2);
        assert_eq!(l.draft(4), Vec::<u32>::new());
        // A planted entry whose tokens don't match is skipped too, and a shorter n-gram still drafts.
        let mut m = lookup_over(&[9, 8, 1, 2, 7, 7, 1, 2]);
        m.follower.insert(key(&[7, 7, 1, 2]), 5);
        assert_eq!(m.draft(2), vec![7, 7]);
    }

    #[test]
    fn a_draft_reports_its_match_length() {
        // A 3-gram match: [1,2,3] at 0, and nothing earlier matches the 4-gram [9,1,2,3].
        let mut l = lookup_over(&[1, 2, 3, 4, 9, 1, 2, 3]);
        assert!(l.last_match().is_none(), "nothing drafted yet");
        assert_eq!(l.draft(4), vec![4, 9, 1, 2]);
        assert_eq!(
            l.last_match(),
            Some(LookupMatch {
                n: 3,
                continues: false
            })
        );
        // A 2-gram match.
        let mut two = lookup_over(&[5, 6, 7, 8, 5, 6]);
        assert_eq!(two.draft(4), vec![7, 8, 5, 6]);
        assert_eq!(two.last_match().map(|m| m.n), Some(2));
        // No draft: no match, whatever came before.
        let mut none = lookup_over(&[1, 2, 3, 4, 9, 1, 2, 3]);
        assert!(!none.draft(4).is_empty());
        none.observe(&[42]);
        assert!(none.draft(4).is_empty());
        assert_eq!(none.last_match(), None);
    }

    #[test]
    fn a_draft_that_carries_on_the_last_run_continues() {
        // The run [10..=17] is in the history; the tail [10,11,12,13] matches its start.
        let run: Vec<u32> = (10..18).collect();
        let mut l = Lookup::new(2, 4);
        l.observe(&run);
        l.observe(&[99]);
        l.observe(&run[..4]);
        assert_eq!(l.draft(4), vec![14, 15, 16, 17]);
        assert_eq!(l.last_match().map(|m| (m.n, m.continues)), Some((4, false)));
        // The step kept two drafts and picked the run's next token: still the same run.
        l.observe(&[14, 15, 16]);
        assert_eq!(l.draft(4), vec![17, 99, 10, 11]);
        assert_eq!(l.last_match().map(|m| (m.n, m.continues)), Some((4, true)));
        // A token off the run breaks it: the next match, if any, is a new run.
        l.observe(&[17, 50]);
        assert_eq!(
            l.last_match().map(|m| m.continues),
            Some(true),
            "unchanged until the next draft"
        );
        let _ = l.draft(4);
        assert_ne!(l.last_match().map(|m| m.continues), Some(true));
    }

    #[test]
    fn a_step_without_a_draft_ends_the_run() {
        let run: Vec<u32> = (10..18).collect();
        let mut l = Lookup::new(2, 4);
        l.observe(&run);
        l.observe(&[99]);
        l.observe(&run[..4]);
        assert!(!l.draft(4).is_empty());
        // A step asked for nothing (max 0): the run is not carried over it.
        l.observe(&[14]);
        assert!(l.draft(0).is_empty());
        assert_eq!(l.last_match(), None);
        l.observe(&[15]);
        assert_eq!(l.draft(4), vec![16, 17, 99, 10]);
        assert_eq!(l.last_match().map(|m| m.continues), Some(false));
    }

    #[test]
    fn it_is_the_lookup_drafter() {
        assert_eq!(Lookup::new(2, 4).name(), "lookup");
    }
}
