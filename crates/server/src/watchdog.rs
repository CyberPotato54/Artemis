//! The stall watchdog. `Watch` is the pure part: given the backend's `Progress` at each tick
//! it says whether work has gone without moving for longer than its bound. It owns no clock and no task: the caller
//! passes `now` (the app's `Clock`), so a test drives it with plain `Duration`s. `spawn` is the task around it, which
//! ends the process with code 4 when `Watch` says stalled.
use crate::app::App;
use crate::backend::{Lifecycle, Progress};
use crate::logs::{LifecycleLine, Logger};
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tokio::time::{interval_at, MissedTickBehavior};

/// The exit code of a stall.
pub const EXIT_STALL: i32 = 4;

/// A stall is decided when the quiet, plus this, reaches the bound. tokio keeps an interval on its grid when a tick is
/// under 5 ms late, so a gap meant to be the whole bound can sample a few ms short, and without the slack the stall
/// would wait a further tick. It absorbs that jitter and can only make a stall fire up to this much early, never late;
/// the `stall` line still reports the quiet that was measured.
pub const STALL_SLACK: Duration = Duration::from_millis(50);

/// What the stall's lines may take in all: the log's lock or its disk can be the very thing that is stuck.
const LINE_BUDGET: Duration = Duration::from_secs(2);

/// What the stuck work is, for the `stall` line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Doing {
    Request,
    Load,
    Reload,
    Command,
}

impl Doing {
    pub fn name(self) -> &'static str {
        match self {
            Doing::Request => "request",
            Doing::Load => "load",
            Doing::Reload => "reload",
            Doing::Command => "command",
        }
    }
}

/// Work that has not moved for `quiet`, past `bound_s`; `ticks` is the counter it stopped at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stalled {
    pub doing: Doing,
    pub quiet: Duration,
    pub bound_s: u64,
    pub ticks: u64,
}

/// The ticks and the state at the last check, and when either last changed.
struct Since {
    ticks: u64,
    state: Lifecycle,
    moved: Duration,
}

pub struct Watch {
    stall_s: u64,
    stall_load_s: u64,
    /// `None` while nothing is pending.
    since: Option<Since>,
}

impl Watch {
    /// `stall_s` bounds a request or a command; `stall_load_s` bounds a load or a reload. 0 for `stall_s` turns the
    /// watch off; 0 for `stall_load_s` lifts only the load bound.
    pub fn new(stall_s: u64, stall_load_s: u64) -> Watch {
        Watch {
            stall_s,
            stall_load_s,
            since: None,
        }
    }

    /// One look at the backend. Work is `pending > 0` or a load running (the state `Loading` or `Recovering`); with
    /// neither, the watch forgets its baseline and the next work starts a fresh clock. Work arriving does not tick, so
    /// more of it, however steady, never reads as movement. `now` is on any clock that never runs backwards;
    /// `has_request` says whether a request is among the pending work (the caller's in-flight count), which only names
    /// the stall.
    pub fn check(
        &mut self,
        now: Duration,
        p: Progress,
        state: Lifecycle,
        has_request: bool,
    ) -> Option<Stalled> {
        // A load running is work even when no command is pending (a device loss starts one on its own).
        let loading = matches!(state, Lifecycle::Loading | Lifecycle::Recovering);
        if p.pending == 0 && !loading {
            self.since = None;
            return None;
        }
        let since = self.since.get_or_insert(Since {
            ticks: p.ticks,
            state,
            moved: now,
        });
        // A load ticks per weight tensor and per layer of expert copies, and at its ends; a change of state counts as
        // movement too.
        if since.ticks != p.ticks || since.state != state {
            *since = Since {
                ticks: p.ticks,
                state,
                moved: now,
            };
        }
        let bound_s = if loading {
            self.stall_load_s
        } else {
            self.stall_s
        };
        if self.stall_s == 0 || bound_s == 0 {
            return None;
        }
        let quiet = now.saturating_sub(since.moved);
        (quiet + STALL_SLACK >= Duration::from_secs(bound_s)).then_some(Stalled {
            doing: match state {
                Lifecycle::Loading => Doing::Load,
                Lifecycle::Recovering => Doing::Reload,
                _ if has_request => Doing::Request,
                _ => Doing::Command,
            },
            quiet,
            bound_s,
            ticks: p.ticks,
        })
    }
}

/// Starts the watchdog on the current runtime. `serve` aborts it when its stop begins. With a backend that
/// reports no `Progress` it does nothing.
pub fn spawn(app: &App) -> JoinHandle<()> {
    tokio::spawn(run(app.clone()))
}

async fn run(app: App) {
    let c = app.config();
    let mut watch = Watch::new(c.stall_s, c.stall_load_s);
    let tuning = app.tuning();
    let every = tuning.watchdog_tick.max(Duration::from_millis(1));
    let mut tick = interval_at(tokio::time::Instant::now() + every, every);
    tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        let Some(progress) = app.backend().progress() else {
            continue;
        };
        let request = app.oldest_in_flight();
        let stalled = watch.check(
            tuning.clock.now(),
            progress,
            app.backend().state(),
            request.is_some(),
        );
        if let Some(stalled) = stalled {
            // The only halter: `serve` and the heartbeat see the flag and write nothing more.
            if app.begin_halt() {
                halt(&app, &stalled, request).await;
            }
            return;
        }
    }
}

/// Writes the `stall`, `stats` and `exit` lines and ends the process. It waits on nothing but the log, and on that for at
/// most `LINE_BUDGET` in all: not on the backend, the queue or `serve`'s stop, all of which wait on the stuck worker.
/// The `stats` line is the backend's latest copy, which asks the engine nothing.
///
/// The lines and the halt run in one closure on a blocking thread that nothing can cancel. `serve` aborts this task when
/// a stop begins, and an abort that landed between the lines and the halt must not leave the process running.
async fn halt(app: &App, stalled: &Stalled, request: Option<String>) {
    let logger = app.logger().clone();
    let halt = app.tuning().halt.clone();
    let lines = [
        LifecycleLine::Stall {
            doing: stalled.doing.name().into(),
            request,
            quiet_s: stalled.quiet.as_secs_f64(),
            bound_s: stalled.bound_s,
            ticks: stalled.ticks,
        },
        app.stats_line(),
        LifecycleLine::Exit {
            reason: "stall".into(),
        },
    ];
    // The write blocks (a helper thread and a wait), so it runs off the runtime's workers. Awaiting the handle only
    // keeps this task alive until the halt: dropping it, as an abort does, leaves the closure running.
    let _ = tokio::task::spawn_blocking(move || {
        write_within(&logger, &lines, LINE_BUDGET);
        halt(EXIT_STALL);
    })
    .await;
}

/// Writes `lines` in order, all within `budget`. A line that misses it ends the writing: the lines after it would only
/// queue behind it on the same lock, and could be written out of order.
fn write_within(logger: &Logger, lines: &[LifecycleLine], budget: Duration) {
    let end = Instant::now() + budget;
    for line in lines {
        if !logger.lifecycle_within(line, end.saturating_duration_since(Instant::now())) {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::test_app;
    use std::sync::{Arc, Mutex};

    fn s(secs: u64) -> Duration {
        Duration::from_secs(secs)
    }

    fn busy(ticks: u64) -> Progress {
        Progress { ticks, pending: 1 }
    }

    fn watch() -> Watch {
        Watch::new(120, 300)
    }

    #[test]
    fn an_idle_backend_never_stalls() {
        let mut w = watch();
        for t in [0, 120, 10_000] {
            assert_eq!(
                w.check(s(t), Progress::default(), Lifecycle::Ready, false),
                None
            );
        }
    }

    #[test]
    fn pending_work_with_no_movement_stalls_at_stall_s() {
        let mut w = watch();
        assert_eq!(w.check(s(0), busy(7), Lifecycle::Ready, true), None);
        assert_eq!(w.check(s(119), busy(7), Lifecycle::Ready, true), None);
        assert_eq!(
            w.check(s(120), busy(7), Lifecycle::Ready, true),
            Some(Stalled {
                doing: Doing::Request,
                quiet: s(120),
                bound_s: 120,
                ticks: 7
            })
        );
    }

    #[test]
    fn a_tick_a_few_ms_short_of_the_bound_still_stalls() {
        // tokio keeps an interval on its grid when a tick is under 5 ms late, so a gap meant to be `bound` can sample
        // a few ms short; the slack absorbs it, and no more than it.
        let ms = Duration::from_millis;
        for (state, has_request, bound) in [
            (Lifecycle::Ready, true, 120),
            (Lifecycle::Loading, false, 300),
        ] {
            let mut w = watch();
            assert_eq!(w.check(s(0), busy(7), state, has_request), None);
            assert_eq!(
                w.check(s(bound) - ms(60), busy(7), state, has_request),
                None,
                "60 ms short of {bound} s"
            );
            let stalled = w
                .check(s(bound) - ms(40), busy(7), state, has_request)
                .unwrap_or_else(|| panic!("40 ms short of {bound} s"));
            // The line reports the quiet that was measured, not the bound.
            assert_eq!(stalled.quiet, s(bound) - ms(40));
            assert_eq!(stalled.bound_s, bound);
        }
    }

    #[test]
    fn work_with_no_request_is_a_command() {
        let mut w = watch();
        assert_eq!(w.check(s(0), busy(1), Lifecycle::Ready, false), None);
        let stalled = w.check(s(120), busy(1), Lifecycle::Ready, false).unwrap();
        assert_eq!(stalled.doing, Doing::Command);
        assert_eq!(stalled.doing.name(), "command");
    }

    #[test]
    fn movement_resets_the_clock() {
        let mut w = watch();
        assert_eq!(w.check(s(0), busy(1), Lifecycle::Ready, true), None);
        assert_eq!(w.check(s(100), busy(2), Lifecycle::Ready, true), None);
        assert_eq!(w.check(s(219), busy(2), Lifecycle::Ready, true), None);
        let stalled = w.check(s(220), busy(2), Lifecycle::Ready, true).unwrap();
        assert_eq!((stalled.quiet, stalled.ticks), (s(120), 2));
    }

    #[test]
    fn idle_in_between_restarts_the_clock() {
        let mut w = watch();
        assert_eq!(w.check(s(0), busy(1), Lifecycle::Ready, true), None);
        assert_eq!(w.check(s(100), busy(1), Lifecycle::Ready, true), None);
        // The work ends; the next piece of work arrives with the same count and is a new start.
        assert_eq!(
            w.check(
                s(110),
                Progress {
                    ticks: 1,
                    pending: 0
                },
                Lifecycle::Ready,
                false
            ),
            None
        );
        assert_eq!(w.check(s(200), busy(1), Lifecycle::Ready, true), None);
        assert_eq!(w.check(s(319), busy(1), Lifecycle::Ready, true), None);
        assert!(w.check(s(320), busy(1), Lifecycle::Ready, true).is_some());
    }

    #[test]
    fn a_load_gets_stall_load_s() {
        let mut w = watch();
        assert_eq!(w.check(s(0), busy(3), Lifecycle::Loading, false), None);
        assert_eq!(w.check(s(299), busy(3), Lifecycle::Loading, false), None);
        assert_eq!(
            w.check(s(300), busy(3), Lifecycle::Loading, false),
            Some(Stalled {
                doing: Doing::Load,
                quiet: s(300),
                bound_s: 300,
                ticks: 3
            })
        );
        assert_eq!(Doing::Load.name(), "load");
    }

    #[test]
    fn steady_arrivals_do_not_hide_a_stuck_worker() {
        // The worker ticks nowhere, but requests keep arriving and ending behind the stuck one: `pending` rises and
        // falls, and the ticks stand still.
        let mut w = watch();
        for (t, pending) in [(0, 1), (30, 2), (60, 1), (90, 3), (119, 2)] {
            let p = Progress { ticks: 9, pending };
            assert_eq!(w.check(s(t), p, Lifecycle::Ready, true), None);
        }
        let p = Progress {
            ticks: 9,
            pending: 4,
        };
        let stalled = w.check(s(120), p, Lifecycle::Ready, true).unwrap();
        assert_eq!(
            (stalled.doing, stalled.quiet, stalled.ticks),
            (Doing::Request, s(120), 9)
        );
    }

    #[test]
    fn a_load_running_is_work_with_nothing_pending() {
        let idle_load = Progress {
            ticks: 4,
            pending: 0,
        };
        let mut w = watch();
        assert_eq!(w.check(s(0), idle_load, Lifecycle::Loading, false), None);
        assert_eq!(w.check(s(299), idle_load, Lifecycle::Loading, false), None);
        let stalled = w
            .check(s(300), idle_load, Lifecycle::Loading, false)
            .unwrap();
        assert_eq!(
            (stalled.doing, stalled.bound_s, stalled.ticks),
            (Doing::Load, 300, 4)
        );

        let mut w = watch();
        assert_eq!(w.check(s(0), idle_load, Lifecycle::Recovering, false), None);
        assert_eq!(
            w.check(s(299), idle_load, Lifecycle::Recovering, false),
            None
        );
        let stalled = w
            .check(s(300), idle_load, Lifecycle::Recovering, false)
            .unwrap();
        assert_eq!((stalled.doing, stalled.bound_s), (Doing::Reload, 300));

        // When the load ends and nothing is pending, the watch is idle again.
        assert_eq!(w.check(s(301), idle_load, Lifecycle::Ready, false), None);
        assert_eq!(w.check(s(10_000), idle_load, Lifecycle::Ready, false), None);
    }

    #[test]
    fn a_reload_gets_stall_load_s_too() {
        let mut w = watch();
        assert_eq!(w.check(s(0), busy(3), Lifecycle::Recovering, true), None);
        assert_eq!(w.check(s(299), busy(3), Lifecycle::Recovering, true), None);
        let stalled = w
            .check(s(300), busy(3), Lifecycle::Recovering, true)
            .unwrap();
        assert_eq!((stalled.doing, stalled.bound_s), (Doing::Reload, 300));
        assert_eq!(stalled.doing.name(), "reload");
    }

    #[test]
    fn a_state_change_is_movement() {
        let mut w = watch();
        assert_eq!(w.check(s(0), busy(5), Lifecycle::Ready, true), None);
        // The ticks stand still, but a device loss moved the state: the clock restarts, with the load bound.
        assert_eq!(w.check(s(100), busy(5), Lifecycle::Recovering, true), None);
        assert_eq!(w.check(s(399), busy(5), Lifecycle::Recovering, true), None);
        let stalled = w
            .check(s(400), busy(5), Lifecycle::Recovering, true)
            .unwrap();
        assert_eq!((stalled.doing, stalled.quiet), (Doing::Reload, s(300)));
        assert_eq!(Doing::Request.name(), "request");
    }

    #[test]
    fn zero_turns_it_off() {
        let mut off = Watch::new(0, 300);
        assert_eq!(off.check(s(0), busy(1), Lifecycle::Ready, true), None);
        assert_eq!(off.check(s(100_000), busy(1), Lifecycle::Ready, true), None);
        assert_eq!(
            off.check(s(200_000), busy(1), Lifecycle::Loading, false),
            None
        );

        // `stall_load_s = 0` lifts only the load bound.
        let mut no_load_bound = Watch::new(120, 0);
        assert_eq!(
            no_load_bound.check(s(0), busy(1), Lifecycle::Loading, false),
            None
        );
        assert_eq!(
            no_load_bound.check(s(100_000), busy(1), Lifecycle::Loading, false),
            None
        );
        assert_eq!(
            no_load_bound.check(s(100_001), busy(2), Lifecycle::Ready, true),
            None
        );
        assert!(no_load_bound
            .check(s(100_121), busy(2), Lifecycle::Ready, true)
            .is_some());
    }

    #[tokio::test]
    async fn an_abort_while_the_lines_are_written_does_not_cancel_the_halt() {
        let halts = Arc::new(Mutex::new(Vec::new()));
        let (logger, app, dir) = test_app(halts.clone());
        // The log's lock is held, as a stuck disk would hold it: the stall's lines wait inside the halt's closure.
        let held = logger.hold_the_lock();
        let stalled = Stalled {
            doing: Doing::Request,
            quiet: s(120),
            bound_s: 120,
            ticks: 1,
        };
        let task = tokio::spawn({
            let app = app.clone();
            async move { halt(&app, &stalled, None).await }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        // `serve`'s stop aborts the task while the closure is writing; the lock then frees.
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(halts.lock().unwrap().is_empty(), "halted before the lines");
        drop(held);
        let end = Instant::now() + Duration::from_secs(5);
        while halts.lock().unwrap().is_empty() && Instant::now() < end {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(*halts.lock().unwrap(), [EXIT_STALL]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
