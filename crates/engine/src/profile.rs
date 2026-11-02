//! The routing profile that fills the expert cache at load: how often each expert was routed, per layer,
//! in the 2026-10-08 expert-cache spike, summed over its 16 sessions and 364,544 tokens. Each layer's cache takes that
//! layer's most-counted experts.
use crate::Error;
use qwen36::dims::Dims;

/// The spike's counts for Qwen3.6-35B-A3B: u64, row-major [layer][expert], little-endian (`data/README.md`).
pub const QWEN36_PROFILE: &[u8] = include_bytes!("../data/qwen36-expert-profile.bin");

/// Routing counts for every expert of every main layer.
pub struct Profile {
    n_layer: usize,
    n_expert: usize,
    counts: Vec<u64>,
}

impl Profile {
    /// u64 counts, [layer][expert], little-endian. Refuses any length other than 8 × n_layer × n_expert.
    pub fn from_bytes(b: &[u8], d: &Dims) -> Result<Profile, Error> {
        let want = 8 * d.n_layer * d.n_expert;
        if b.len() != want {
            return Err(Error::BadProfile { want, got: b.len() });
        }
        Ok(Profile {
            n_layer: d.n_layer,
            n_expert: d.n_expert,
            counts: b
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| u64::from_le_bytes(*c))
                .collect(),
        })
    }

    /// The spike's counts for the real model; all zeros for any other dims (the tiny test model), so a cache fills
    /// the lowest ids.
    pub fn for_dims(d: &Dims) -> Profile {
        if *d == Dims::QWEN36 {
            Profile::from_bytes(QWEN36_PROFILE, d)
                .expect("the committed profile covers Dims::QWEN36")
        } else {
            Profile {
                n_layer: d.n_layer,
                n_expert: d.n_expert,
                counts: vec![0; d.n_layer * d.n_expert],
            }
        }
    }

    pub fn count(&self, layer: usize, expert: usize) -> u64 {
        assert!(
            layer < self.n_layer && expert < self.n_expert,
            "expert {expert} of layer {layer}: the profile has {} layers of {} experts",
            self.n_layer,
            self.n_expert
        );
        self.counts[layer * self.n_expert + expert]
    }

    /// `layer`'s experts, most counted first, ties to the lower id.
    pub fn ranked(&self, layer: usize) -> Vec<usize> {
        let mut ids: Vec<usize> = (0..self.n_expert).collect();
        ids.sort_by_key(|&e| (std::cmp::Reverse(self.count(layer, e)), e));
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen36::testing::fnv1a64;

    #[test]
    fn the_committed_profile_is_the_spikes_counts() {
        // data/README.md: 16 sessions, 364,544 tokens, 8 routed experts per token, so every layer sums to 2,916,352.
        let d = Dims::QWEN36;
        assert_eq!(QWEN36_PROFILE.len(), 81_920);
        assert_eq!(fnv1a64(QWEN36_PROFILE), 0xeb42_e627_90eb_94da);
        let p = Profile::for_dims(&d);
        for l in 0..d.n_layer {
            let sum: u64 = (0..d.n_expert).map(|e| p.count(l, e)).sum();
            assert_eq!(sum, 364_544 * 8, "layer {l}");
        }
        let top0 = p.ranked(0);
        assert_eq!(top0[..3], [211, 169, 75]);
        assert_eq!(
            (p.count(0, 211), p.count(0, 169), p.count(0, 75)),
            (33_672, 32_628, 30_902)
        );
        assert_eq!((top0[255], p.count(0, 199)), (199, 3_870));
        assert_eq!(p.ranked(39)[..3], [195, 184, 150]);
        assert_eq!(p.count(39, 195), 87_881);
    }

    #[test]
    fn ties_go_to_the_lower_id() {
        // The tiny model's dims: 4 layers of 16 experts. Layer 1: expert 12 leads; 9 and 3 tie behind it.
        let d = Dims::tiny();
        let mut counts = vec![0u64; d.n_layer * d.n_expert];
        counts[16 + 12] = 7;
        counts[16 + 9] = 5;
        counts[16 + 3] = 5;
        let b: Vec<u8> = counts.iter().flat_map(|c| c.to_le_bytes()).collect();
        let p = Profile::from_bytes(&b, &d).unwrap();
        assert_eq!(p.ranked(1)[..4], [12, 3, 9, 0]);
        assert_eq!(p.ranked(0), (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn a_profile_of_the_wrong_length_is_refused() {
        let e = Profile::from_bytes(&[0u8; 100], &Dims::tiny())
            .err()
            .unwrap();
        assert!(
            matches!(
                e,
                Error::BadProfile {
                    want: 512,
                    got: 100
                }
            ),
            "{e:?}"
        );
        let m = e.to_string();
        assert!(m.contains("100 bytes") && m.contains("512"), "{m}");
    }

    #[test]
    fn other_dims_get_an_all_zero_profile_that_ranks_by_id() {
        let d = Dims::tiny();
        let p = Profile::for_dims(&d);
        for l in 0..d.n_layer {
            assert_eq!(p.ranked(l), (0..d.n_expert).collect::<Vec<_>>());
        }
    }
}
