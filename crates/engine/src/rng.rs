//! The engine's own RNG: xoshiro256**, seeded through splitmix64, so a seed gives the same numbers on every
//! platform. The sampler draws from it, and `--force-swap-every-token` picks its swaps with it.

/// xoshiro256** (Blackman and Vigna), its state filled from the seed by splitmix64.
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// An RNG whose numbers depend only on `seed`; the same seed gives the same sequence everywhere.
    pub fn new(seed: u64) -> Rng {
        let mut x = seed;
        let mut splitmix = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Rng {
            s: [splitmix(), splitmix(), splitmix(), splitmix()],
        }
    }

    /// The next 64 random bits (xoshiro256**'s output).
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let out = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// Uniform in 0..n, without modulo bias: Lemire's multiply-shift, redrawing the few values that would favour some
    /// results. Panics if n is 0.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "a draw below 0");
        let n = n as u64;
        let mut m = self.next_u64() as u128 * n as u128;
        if (m as u64) < n {
            let floor = n.wrapping_neg() % n;
            while (m as u64) < floor {
                m = self.next_u64() as u128 * n as u128;
            }
        }
        (m >> 64) as usize
    }

    /// Uniform in [0, 1), from the top 53 bits of a draw.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_seed_gives_the_same_numbers_everywhere() {
        let first = |seed| {
            let mut r = Rng::new(seed);
            [r.next_u64(), r.next_u64(), r.next_u64()]
        };
        // Computed with an independent Python implementation of splitmix64 and xoshiro256**.
        assert_eq!(
            first(0),
            [
                0x99ec_5f36_cb75_f2b4,
                0xbf6e_1f78_4956_452a,
                0x1a5f_849d_4933_e6e0
            ]
        );
        assert_eq!(
            first(1),
            [
                0xb3f2_af6d_0fc7_10c5,
                0x853b_5596_4736_4cea,
                0x92f8_9756_082a_4514
            ]
        );
    }

    #[test]
    fn below_covers_every_value_and_never_reaches_n() {
        let mut r = Rng::new(7);
        let mut seen = [0usize; 7];
        for _ in 0..7_000 {
            seen[r.below(7)] += 1;
        }
        // Each value about 1,000 times; a biased or broken draw would starve one.
        assert!(seen.iter().all(|&c| (850..1150).contains(&c)), "{seen:?}");
        assert_eq!(r.below(1), 0);
    }

    #[test]
    fn unit_stays_in_the_half_open_interval() {
        let mut r = Rng::new(3);
        let mut sum = 0.0;
        for _ in 0..10_000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u), "{u}");
            sum += u;
        }
        assert!(
            (sum / 10_000.0 - 0.5).abs() < 0.01,
            "mean {}",
            sum / 10_000.0
        );
    }

    #[test]
    fn below_redraws_for_a_huge_n() {
        // n = 2^63 + 1: about half the draws land in the biased region and are redrawn.
        let n = (1usize << 63) + 1;
        let (mut a, mut b) = (Rng::new(11), Rng::new(11));
        let mut redraws = 0;
        for _ in 0..200 {
            // The same arithmetic, counting the redraws.
            let mut m = b.next_u64() as u128 * n as u128;
            if (m as u64) < n as u64 {
                let floor = (n as u64).wrapping_neg() % n as u64;
                while (m as u64) < floor {
                    redraws += 1;
                    m = b.next_u64() as u128 * n as u128;
                }
            }
            let got = a.below(n);
            assert!(got < n);
            assert_eq!(got, (m >> 64) as usize);
        }
        assert!(redraws > 0, "the rejection branch never ran");
    }

    #[test]
    #[should_panic(expected = "a draw below 0")]
    fn below_zero_panics() {
        Rng::new(1).below(0);
    }
}
