//! Budget watching as a pure state machine, apart from the GPU so CI tests it. The engine reads
//! the VRAM heap every `BUDGET_EVERY` tokens and between sessions and asks `BudgetWatch::check` what to do: retire the
//! cache's last chunk when the margin is gone, grow a chunk after two checks in a row with room for it, or hold.
//!
//! The cache is sized with `DRIVER_WORK` free beside `VRAM_MARGIN`, for what the driver allocates on first use of the
//! kernels (about 25 MiB at the first prefill chunk, 1 MiB at the first decode). Once spent, that memory is in the
//! usage, so a shrink waits only for the margin to go: usage plus `VRAM_MARGIN` past the budget. A grow keeps the margin
//! and `GROW_SLACK` free with the chunk allocated: the 32 MiB between the two lines keeps a cache from shrinking and
//! growing back over a few MiB of drift, and, being less than what the driver leaves of `DRIVER_WORK`, lets a cache
//! fitted with nothing to spare grow back to its size at load after a dip.
//!
//! The RX 9070 XT's driver lowers this process's budget by 1,741 MiB once its usage passes about 11.5 GB (measured:
//! 15,416 MiB, then 13,675 at 12,260 MiB of usage), and raises it again as soon as usage falls back.
//! Room read below that line may vanish once a chunk is allocated across it, so a grow is checked again with the chunk
//! allocated (`grown`), and given back at once if the budget then leaves less than the margin and `GROW_SLACK`. Growing
//! then waits 2, 4, … up to 64 checks with room before trying again, so a cache that sits at the line doesn't allocate
//! and free a chunk every two checks.
use crate::weights::{DRIVER_WORK, VRAM_MARGIN};

/// Tokens between budget checks.
pub const BUDGET_EVERY: usize = 256;

/// The most checks with room growing waits after a refused grow.
const MAX_WAIT: u32 = 64;

/// What a grow keeps free beyond the margin, with its chunk allocated: half of `DRIVER_WORK`, less than the about 38 MiB
/// the driver leaves of it once it has run the kernels.
pub const GROW_SLACK: u64 = DRIVER_WORK / 2;

/// What a budget check calls for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Nothing to do.
    Hold,
    /// The margin is gone: retire the cache's last chunk.
    Shrink,
    /// The margin is gone and no chunk is left to retire (`Stats::budget_short`).
    Short,
    /// Two checks in a row had room for the next chunk: allocate it, then call `grown` with the heap.
    Grow,
}

/// Budget watching's state between checks.
#[derive(Clone, Debug, Default)]
pub struct BudgetWatch {
    /// Checks in a row that had room for the next chunk.
    room: u32,
    /// Checks with room still to wait before growing again, after a refused grow: a check without room doesn't count.
    wait: u32,
    /// The wait the last refusal set.
    backoff: u32,
}

/// Whether the heap keeps the margin free.
fn keeps_margin(h: vk::HeapBudget) -> bool {
    h.usage + VRAM_MARGIN <= h.budget
}

/// Whether the heap keeps the margin and `GROW_SLACK` free with `more` bytes allocated: room to grow.
fn has_room(h: vk::HeapBudget, more: u64) -> bool {
    h.usage + more + VRAM_MARGIN + GROW_SLACK <= h.budget
}

impl BudgetWatch {
    /// One check of the heap `h`. `next` is the bytes of the chunk a grow would add (None at the cache's ceiling);
    /// `shrinkable` whether a chunk of the cache's own is live to retire; `retiring` whether one is still retiring, in
    /// which case a short budget waits for it to land (the chunk is kept, the next check retries).
    pub fn check(
        &mut self,
        h: vk::HeapBudget,
        next: Option<u64>,
        shrinkable: bool,
        retiring: bool,
    ) -> Verdict {
        if !keeps_margin(h) {
            self.room = 0;
            return if retiring {
                Verdict::Hold
            } else if shrinkable {
                Verdict::Shrink
            } else {
                Verdict::Short
            };
        }
        if !next.is_some_and(|b| has_room(h, b)) {
            self.room = 0;
            return Verdict::Hold;
        }
        self.room += 1;
        if self.wait > 0 {
            self.wait -= 1;
            return Verdict::Hold;
        }
        if self.room >= 2 {
            self.room = 0;
            return Verdict::Grow;
        }
        Verdict::Hold
    }

    /// After a grow allocated its chunk: whether the heap `h`, read with the chunk allocated, keeps the margin and
    /// `GROW_SLACK` free. If not (or `h` is None: the allocation itself was refused), the caller gives the chunk back, and
    /// growing waits.
    pub fn grown(&mut self, h: Option<vk::HeapBudget>) -> bool {
        if h.is_some_and(|h| has_room(h, 0)) {
            self.backoff = 0;
            return true;
        }
        self.backoff = (self.backoff * 2).clamp(2, MAX_WAIT);
        self.wait = self.backoff;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;
    /// A 14-row chunk of the real model.
    const CHUNK: u64 = 14 * 76_439_552;

    /// The RX 9070 XT's VRAM heap as it was measured: a budget of 15,416 MiB until this process's usage passes
    /// 11,500 MiB, then 13,675, less what other processes take (`desktop`).
    fn heap(usage: u64, desktop: u64) -> vk::HeapBudget {
        let budget = if usage > 11_500 * MIB {
            13_675 * MIB
        } else {
            15_416 * MIB
        };
        vk::HeapBudget {
            size: 16_304 * MIB,
            budget: budget - desktop,
            usage,
        }
    }

    #[test]
    fn a_short_budget_shrinks_and_a_chunk_still_retiring_is_waited_for() {
        let mut w = BudgetWatch::default();
        // 12,000 MiB used, the desktop's 1,500 off the 13,675 budget: 12,175 leaves less than the 512 MiB margin free.
        let h = heap(12_000 * MIB, 1_500 * MIB);
        assert_eq!(w.check(h, Some(CHUNK), true, false), Verdict::Shrink);
        assert_eq!(
            w.check(h, Some(CHUNK), true, true),
            Verdict::Hold,
            "one retire at a time"
        );
        assert_eq!(
            w.check(h, Some(CHUNK), false, false),
            Verdict::Short,
            "no chunk left"
        );
        // Exactly the margin left is enough; a byte less isn't.
        let edge = vk::HeapBudget {
            usage: 13_675 * MIB - VRAM_MARGIN,
            ..heap(12_000 * MIB, 0)
        };
        assert_eq!(w.check(edge, None, true, false), Verdict::Hold);
        let short = vk::HeapBudget {
            usage: edge.usage + 1,
            ..edge
        };
        assert_eq!(w.check(short, None, true, false), Verdict::Shrink);
    }

    #[test]
    fn the_driver_work_spent_after_load_is_no_reason_to_shrink() {
        // A cache fitted with 10 MiB to spare beyond the margin and DRIVER_WORK (the fit leaves 0 to 69 MiB). The driver
        // then takes 25 MiB at the first prefill chunk and 1 MiB at the first decode: out of DRIVER_WORK, not the margin,
        // so nothing shrinks, and nothing grows either. Only when the desktop takes the margin does the cache shrink.
        let mut w = BudgetWatch::default();
        let budget = 13_675 * MIB;
        let at = |usage| vk::HeapBudget {
            usage,
            ..heap(12_000 * MIB, 0)
        };
        let fitted = budget - VRAM_MARGIN - DRIVER_WORK - 10 * MIB;
        for usage in [fitted, fitted + 25 * MIB, fitted + 26 * MIB] {
            for _ in 0..3 {
                assert_eq!(w.check(at(usage), Some(CHUNK), true, false), Verdict::Hold);
            }
        }
        let desktop = vk::HeapBudget {
            budget: budget - 600 * MIB,
            ..at(fitted + 26 * MIB)
        };
        assert_eq!(w.check(desktop, Some(CHUNK), true, false), Verdict::Shrink);
    }

    #[test]
    fn a_cache_fitted_with_nothing_to_spare_grows_back_after_a_dip() {
        // Fitted with exactly the margin and DRIVER_WORK free; the driver then takes 26 MiB of DRIVER_WORK. The desktop
        // takes 600 MiB for a while, and the last chunk retires and is freed. Once the desktop gives the memory back, the
        // chunk grows back to the cache's size at load, though DRIVER_WORK is no longer all free: a grow needs the margin
        // and GROW_SLACK.
        let mut w = BudgetWatch::default();
        let budget = 13_675 * MIB;
        let at = |usage, budget| vk::HeapBudget {
            size: 16_304 * MIB,
            budget,
            usage,
        };
        let loaded = budget - VRAM_MARGIN - DRIVER_WORK + 26 * MIB;
        assert_eq!(
            w.check(at(loaded, budget), Some(CHUNK), true, false),
            Verdict::Hold
        );
        assert_eq!(
            w.check(at(loaded, budget - 600 * MIB), Some(CHUNK), true, false),
            Verdict::Shrink
        );
        let shed = loaded - CHUNK;
        assert_eq!(
            w.check(at(shed, budget), Some(CHUNK), true, false),
            Verdict::Hold
        );
        assert_eq!(
            w.check(at(shed, budget), Some(CHUNK), true, false),
            Verdict::Grow
        );
        assert!(
            w.grown(Some(at(loaded, budget))),
            "back to its size at load"
        );
    }

    #[test]
    fn growing_takes_two_checks_in_a_row_with_room() {
        let mut w = BudgetWatch::default();
        let roomy = heap(10_000 * MIB, 0);
        assert_eq!(w.check(roomy, Some(CHUNK), true, false), Verdict::Hold);
        // A check without room in between starts the count again.
        assert_eq!(
            w.check(heap(10_000 * MIB, 4_500 * MIB), Some(CHUNK), true, false),
            Verdict::Hold
        );
        assert_eq!(w.check(roomy, Some(CHUNK), true, false), Verdict::Hold);
        // So does a short check: 10,000 MiB used of a 10,416 MiB budget leaves less than the margin.
        assert_eq!(
            w.check(heap(10_000 * MIB, 5_000 * MIB), Some(CHUNK), true, false),
            Verdict::Shrink
        );
        assert_eq!(w.check(roomy, Some(CHUNK), true, false), Verdict::Hold);
        assert_eq!(w.check(roomy, Some(CHUNK), true, false), Verdict::Grow);
        assert!(w.grown(Some(heap(10_000 * MIB + CHUNK, 0))));
        // At the ceiling there is nothing to grow.
        for _ in 0..4 {
            assert_eq!(w.check(roomy, None, true, false), Verdict::Hold);
        }
        // Exactly the chunk, the margin and GROW_SLACK free is room to grow; and once the chunk is allocated, exactly the
        // margin and GROW_SLACK left keeps it.
        let edge = vk::HeapBudget {
            usage: 15_416 * MIB - CHUNK - VRAM_MARGIN - GROW_SLACK,
            ..roomy
        };
        assert_eq!(w.check(edge, Some(CHUNK), true, false), Verdict::Hold);
        assert_eq!(w.check(edge, Some(CHUNK), true, false), Verdict::Grow);
        assert!(w.grown(Some(vk::HeapBudget {
            usage: 15_416 * MIB - VRAM_MARGIN - GROW_SLACK,
            ..roomy
        })));
        // Room for the chunk and the margin, but not GROW_SLACK too, isn't room to grow; nor, once a chunk is allocated,
        // a reading that keeps the margin but not GROW_SLACK.
        let tight = vk::HeapBudget {
            usage: 15_416 * MIB - CHUNK - VRAM_MARGIN - GROW_SLACK + 1,
            ..roomy
        };
        for _ in 0..4 {
            assert_eq!(w.check(tight, Some(CHUNK), true, false), Verdict::Hold);
        }
        assert!(!w.grown(Some(vk::HeapBudget {
            usage: 15_416 * MIB - VRAM_MARGIN - GROW_SLACK + 1,
            ..roomy
        })));
    }

    #[test]
    fn a_grow_across_the_drivers_line_is_refused_and_growing_backs_off() {
        // At 11,000 MiB of usage with 1,300 MiB taken by the desktop, the budget below the line (14,116) has room for a
        // chunk; with the chunk allocated, usage passes 11,500 and the budget (12,375) leaves less than the margin and
        // GROW_SLACK. Each refusal is followed by a longer wait: over 1,000 checks the cache tries 20 times, not 500,
        // and never shrinks.
        let (usage, desktop) = (11_000 * MIB, 1_300 * MIB);
        assert!(has_room(heap(usage, desktop), CHUNK));
        assert!(!has_room(heap(usage + CHUNK, desktop), 0));
        let mut w = BudgetWatch::default();
        let mut tries = Vec::new();
        for i in 0..1_000 {
            match w.check(heap(usage, desktop), Some(CHUNK), true, false) {
                Verdict::Grow => {
                    tries.push(i);
                    assert!(!w.grown(Some(heap(usage + CHUNK, desktop))));
                }
                Verdict::Hold => {}
                v => panic!("check {i}: {v:?}"),
            }
        }
        // The second check tries; each refusal then waits 2, 4, … 64 checks (the checks with room keep counting, so
        // the try comes as the wait ends), and from the sixth refusal on a try comes every 65 checks.
        assert_eq!(&tries[..6], [1, 4, 9, 18, 35, 68]);
        assert!(
            tries.windows(2).skip(5).all(|w| w[1] - w[0] == 65),
            "{tries:?}"
        );
        assert_eq!(tries.len(), 20, "{tries:?}");
        // A refused allocation backs off the same way, and a grow that keeps the margin resets the wait.
        let mut w = BudgetWatch::default();
        assert!(!w.grown(None));
        assert_eq!(w.wait, 2);
        assert!(w.grown(Some(heap(usage, 0))));
        assert_eq!(w.backoff, 0);
    }
}
