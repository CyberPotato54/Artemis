//! The HTTP-side queue: which requests wait for the engine, at most 64 of them. The engine takes its
//! commands in order itself; this holds the handlers' side of the wait, so a test can see who is waiting, a full queue
//! answers 503 at once, and an unload can cancel and wake every waiting handler.
use crate::backend::{CancelHandle, Full};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use tokio::sync::Notify;

/// Set and woken by `abort_all`; `aborted` waits on it.
struct Flag {
    set: AtomicBool,
    wake: Notify,
}

struct Slot {
    cancel: CancelHandle,
    flag: Arc<Flag>,
}

#[derive(Default)]
struct State {
    next: u64,
    /// Arrival order: the keys only grow.
    waiting: BTreeMap<u64, Slot>,
}

fn lock(s: &Mutex<State>) -> MutexGuard<'_, State> {
    s.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The requests waiting for the engine (those entered and not yet started).
pub struct Queue {
    limit: usize,
    state: Arc<Mutex<State>>,
}

/// A request's place in the queue. Dropping it leaves the queue, so a handler that is dropped, times out or fails
/// frees its place; `started` is the same on purpose, for the handler that was served.
pub struct Ticket {
    id: u64,
    flag: Arc<Flag>,
    state: Arc<Mutex<State>>,
}

impl Queue {
    pub fn new(limit: usize) -> Queue {
        Queue {
            limit,
            state: Arc::new(Mutex::new(State::default())),
        }
    }

    /// Takes a place at the back, or `Full` when `limit` requests already wait. `cancel` is the request's
    /// `CancelHandle`, which `abort_all` sets.
    pub fn enter(&self, cancel: CancelHandle) -> Result<Ticket, Full> {
        let mut s = lock(&self.state);
        if s.waiting.len() >= self.limit {
            return Err(Full);
        }
        let id = s.next;
        s.next += 1;
        let flag = Arc::new(Flag {
            set: AtomicBool::new(false),
            wake: Notify::new(),
        });
        s.waiting.insert(
            id,
            Slot {
                cancel,
                flag: flag.clone(),
            },
        );
        Ok(Ticket {
            id,
            flag,
            state: self.state.clone(),
        })
    }

    /// Takes every waiting request out of the queue: each one's `CancelHandle` is set before this returns, and its
    /// handler is woken (`Ticket::aborted`). The number taken.
    pub fn abort_all(&self) -> usize {
        let taken = std::mem::take(&mut lock(&self.state).waiting);
        for slot in taken.values() {
            // The flag first: the engine may see the cancel and drop the sink at once, and the handler that finds
            // its channel closed must already find itself aborted.
            slot.flag.set.store(true, Ordering::Release);
            slot.cancel.cancel();
            // A permit is kept for a handler that is not waiting yet.
            slot.flag.wake.notify_one();
        }
        taken.len()
    }

    pub fn waiting(&self) -> usize {
        lock(&self.state).waiting.len()
    }
}

impl Ticket {
    /// Resolves once `abort_all` took this request; never otherwise.
    pub async fn aborted(&self) {
        while !self.flag.set.load(Ordering::Acquire) {
            self.flag.wake.notified().await;
        }
    }

    /// The engine took the request: it no longer waits.
    pub fn started(self) {}
}

impl Drop for Ticket {
    fn drop(&mut self) {
        lock(&self.state).waiting.remove(&self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Sink;
    use std::time::Duration;

    fn handle() -> CancelHandle {
        Sink::channel().2
    }

    #[test]
    fn the_65th_is_refused_and_a_place_frees_on_drop_or_start() {
        let q = Queue::new(64);
        let mut tickets: Vec<Ticket> = (0..64).map(|_| q.enter(handle()).unwrap()).collect();
        assert_eq!(q.waiting(), 64);
        assert!(q.enter(handle()).is_err());
        assert_eq!(q.waiting(), 64, "a refusal takes no place");
        drop(tickets.remove(10));
        assert_eq!(q.waiting(), 63);
        tickets.remove(0).started();
        assert_eq!(q.waiting(), 62);
        let again = q.enter(handle()).unwrap();
        assert_eq!(q.waiting(), 63);
        drop(again);
        drop(tickets);
        assert_eq!(q.waiting(), 0);
    }

    #[tokio::test]
    async fn abort_all_cancels_before_it_returns_and_wakes_each_handler() {
        let q = Queue::new(4);
        let (c1, c2) = (handle(), handle());
        let t1 = q.enter(c1.clone()).unwrap();
        let t2 = q.enter(c2.clone()).unwrap();
        let untouched = handle();
        let t3 = q.enter(untouched.clone()).unwrap();
        // The third has started, so it is not waiting any more.
        t3.started();
        assert_eq!(q.abort_all(), 2);
        assert!(c1.is_cancelled() && c2.is_cancelled());
        assert!(!untouched.is_cancelled());
        assert_eq!(q.waiting(), 0);
        // The wake was kept: a handler that looks after the abort still sees it.
        for t in [&t1, &t2] {
            tokio::time::timeout(Duration::from_secs(30), t.aborted())
                .await
                .expect("an aborted ticket resolves");
        }
        // Dropping a ticket that was taken is harmless, and the place can be reused.
        drop(t1);
        drop(t2);
        assert_eq!(q.abort_all(), 0);
        assert!(q.enter(handle()).is_ok());
    }

    #[tokio::test]
    async fn a_ticket_not_aborted_does_not_resolve() {
        let q = Queue::new(1);
        let t = q.enter(handle()).unwrap();
        let waited = tokio::time::timeout(Duration::from_millis(50), t.aborted()).await;
        assert!(waited.is_err());
        assert_eq!(q.waiting(), 1);
    }
}
