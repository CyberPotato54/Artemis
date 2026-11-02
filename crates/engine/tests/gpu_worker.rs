//! The worker over the real engine on the GPU, with the tiny model: a
//! request generates and snapshots, a later one reuses the live state or a snapshot, an unload parks the snapshots, and
//! the load after it brings them back, with the same bits every time.
use engine::{
    ChannelSink, Config, Cut, Event, Finish, Gone, Lifecycle, Request, Sampling, Source,
    SpecConfig, SpecSource, TokenLogprobs, Worker, WorkerConfig, WorkerEvent,
};
use qwen36::{dims::Dims, testing::tiny_model};
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(120);

/// The tiny model on disk, removed when dropped.
struct ModelFile(PathBuf);

impl ModelFile {
    fn write() -> ModelFile {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = std::env::temp_dir().join(format!(
            "artemis-gpu-worker-{}-{nanos}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, tiny_model(1)).expect("write the tiny model");
        ModelFile(path)
    }
}

impl Drop for ModelFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The worker, shut down (best effort) when dropped, so a panicking test does not leave the engine holding the model
/// file open when `ModelFile` removes it (Windows refuses to remove an open file). Declared after the `ModelFile`, it
/// drops first.
struct Shutdown(Option<Worker>);

impl Shutdown {
    /// The clean end: `Ok` if the worker thread ended without a panic.
    fn finish(mut self) -> Result<(), Gone> {
        self.0.take().expect("not yet shut down").shutdown()
    }
}

impl Deref for Shutdown {
    type Target = Worker;

    fn deref(&self) -> &Worker {
        self.0.as_ref().expect("not yet shut down")
    }
}

impl Drop for Shutdown {
    fn drop(&mut self) {
        if let Some(w) = self.0.take() {
            let _ = w.shutdown();
        }
    }
}

/// `n` tokens of the tiny model's vocabulary, from `first` on, spaced by `step`.
fn tokens(n: usize, first: u32, step: u32) -> Vec<u32> {
    (0..n as u32).map(|i| (first + i * step) % 512).collect()
}

/// A greedy request that asks for the top 2 logprobs and five tokens.
fn request(prompt: Vec<u32>, cut: Option<Cut>) -> Request {
    Request {
        prompt,
        cut,
        sampling: Sampling::GREEDY,
        seed: 1,
        max_tokens: 5,
        stop_tokens: Vec::new(),
        think: None,
        logprobs: Some(2),
    }
}

/// What a request sent, split up.
struct Run {
    restored: Source,
    reused_tokens: u32,
    tokens: Vec<(u32, Option<TokenLogprobs>)>,
    finish: Finish,
    reused_count: u32,
}

/// Submits `req` and reads its events to the end: `Started`, the tokens, then `Done`.
fn run(w: &Worker, req: Request) -> Run {
    let (tx, rx): (_, Receiver<Event>) = channel();
    w.submit(req, Box::new(ChannelSink(tx))).expect("submit");
    let mut events = Vec::new();
    loop {
        let e = rx.recv_timeout(WAIT).expect("an event within the wait");
        let end = matches!(e, Event::Done { .. } | Event::Error { .. });
        events.push(e);
        if end {
            break;
        }
    }
    let mut it = events.into_iter();
    let Some(Event::Started {
        restored,
        reused_tokens,
        ..
    }) = it.next()
    else {
        panic!("the first event was not Started");
    };
    let mut tokens = Vec::new();
    for e in it {
        match e {
            Event::Token {
                id,
                forced: false,
                logprobs,
            } => tokens.push((id, logprobs)),
            Event::Done { finish, counts, .. } => {
                return Run {
                    restored,
                    reused_tokens,
                    tokens,
                    finish,
                    reused_count: counts.reused,
                }
            }
            other => panic!("unexpected event {other:?}"),
        }
    }
    unreachable!("the loop ended on Done or Error")
}

/// A token's logprob and its top alternatives, as bits.
type LogprobBits = (u32, Vec<(u32, u32)>);

/// The tokens and the logprobs as bits, to compare runs exactly.
fn bits(run: &Run) -> Vec<(u32, Option<LogprobBits>)> {
    run.tokens
        .iter()
        .map(|(id, lp)| {
            let lp = lp.as_ref().map(|l| {
                (
                    l.logprob.to_bits(),
                    l.top.iter().map(|&(t, p)| (t, p.to_bits())).collect(),
                )
            });
            (*id, lp)
        })
        .collect()
}

/// A load adds to `Config::load_ticks` as it uploads, so the server can tell a slow load from a stuck one.
#[test]
#[ignore = "needs the GPU"]
fn a_load_ticks_its_progress_counter() {
    let model = ModelFile::write();
    let ticks = Arc::new(AtomicU64::new(0));
    let cfg = Config {
        cache_slots: Some(4),
        ctx: 64,
        snapshot_pool_gib: 1,
        speculation: SpecConfig {
            sources: vec![SpecSource::Lookup],
            ..SpecConfig::default()
        },
        load_ticks: Some(ticks.clone()),
        ..Config::new(model.0.clone())
    };
    let w = Shutdown(Some(Worker::spawn(WorkerConfig {
        dims: Dims::tiny(),
        idle_unload: None,
        ..WorkerConfig::new(cfg)
    })));
    assert_eq!(ticks.load(Ordering::Relaxed), 0);
    let (tx, rx) = channel();
    w.load(Box::new(move |r| {
        let _ = tx.send(r);
    }))
    .expect("load");
    rx.recv_timeout(WAIT)
        .expect("a load answer within the wait")
        .expect("the tiny model loads");
    let counted = ticks.load(Ordering::Relaxed);
    assert!(counted > 0, "the load never ticked");
    w.finish().expect("the worker thread ended cleanly");
}

#[test]
#[ignore = "needs the GPU"]
fn the_worker_generates_reuses_unloads_and_resumes() {
    let model = ModelFile::write();
    let cfg = Config {
        cache_slots: Some(4),
        ctx: 64,
        snapshot_pool_gib: 1,
        // Lookup alone: ctx 64 is under MTP's 512-position floor (the session sizing rules).
        speculation: SpecConfig {
            sources: vec![SpecSource::Lookup],
            ..SpecConfig::default()
        },
        ..Config::new(model.0.clone())
    };
    // No idle timer: the unload below is the test's own.
    let (events_tx, events) = channel();
    let w = Shutdown(Some(Worker::spawn(WorkerConfig {
        dims: Dims::tiny(),
        idle_unload: None,
        notify: Some(Box::new(move |e| {
            let _ = events_tx.send(e);
        })),
        ..WorkerConfig::new(cfg)
    })));

    // A: from nothing. Its prompt-end snapshot is at 26 and its reply end at 34 (the state is one token short of what
    // the client saw).
    let a_prompt = tokens(30, 17, 211);
    let cut = Cut {
        at: 26,
        user_query: true,
    };
    let a = run(&w, request(a_prompt.clone(), Some(cut)));
    assert_eq!((a.restored, a.reused_tokens), (Source::None, 0));
    assert_eq!(a.tokens.len(), 5);
    assert_eq!(a.finish, Finish::Length);
    assert!(a.tokens.iter().all(|(_, lp)| lp.is_some()));

    // B continues A's reply: the live state is A's, tying A's reply-end snapshot, and live wins.
    let mut b_prompt = a_prompt;
    b_prompt.extend(a.tokens.iter().take(4).map(|(id, _)| *id));
    b_prompt.extend([5, 6, 7]);
    let b1 = run(&w, request(b_prompt.clone(), None));
    assert_eq!((b1.restored, b1.reused_tokens), (Source::Live, 34));
    assert_eq!(b1.reused_count, 34);
    assert_eq!(b1.tokens.len(), 5);
    // The comparisons below must compare logprobs, not two `None`s.
    assert!(
        b1.tokens
            .iter()
            .all(|(_, lp)| lp.as_ref().is_some_and(|l| l.top.len() == 2)),
        "a token lacks its top-2 logprobs"
    );

    // C shares nothing with them: a reset, and the live state becomes C's.
    let c = run(&w, request(tokens(20, 300, 97), None));
    assert_eq!((c.restored, c.reused_tokens), (Source::None, 0));

    // B again: the live state is C's, so A's reply end is restored. Same bits as the live continuation.
    let b2 = run(&w, request(b_prompt.clone(), None));
    assert_eq!((b2.restored, b2.reused_tokens), (Source::ReplyEnd, 34));
    assert_eq!(bits(&b2), bits(&b1), "a restore changed the bits");

    // Unload parks every snapshot: A's two or three and C's.
    let (tx, rx) = channel();
    w.unload(Box::new(move |u| {
        let _ = tx.send(u);
    }))
    .expect("unload");
    let unloaded = rx.recv_timeout(WAIT).expect("the unload's answer");
    assert!(unloaded.was_loaded);
    assert!(unloaded.parked >= 3, "parked {}", unloaded.parked);
    assert_eq!(w.state(), Lifecycle::Idle);

    // B once more: it loads, brings the snapshots back and restores A's reply end, with the same bits again.
    let b3 = run(&w, request(b_prompt, None));
    assert_eq!((b3.restored, b3.reused_tokens), (Source::ReplyEnd, 34));
    assert_eq!(bits(&b3), bits(&b1), "a parked snapshot changed the bits");

    // The reload brought snapshots back from the parked ones, not from nothing.
    let loads: Vec<_> = events
        .try_iter()
        .filter_map(|e| match e {
            WorkerEvent::Loaded(info) => Some(info),
            _ => None,
        })
        .collect();
    assert_eq!(loads.len(), 2, "loads: {loads:?}");
    assert_eq!(loads[0].unparked.restored, 0);
    assert!(
        loads[0].vram_used.is_some_and(|b| b > 0),
        "the load read no VRAM use: {:?}",
        loads[0].vram_used
    );
    assert!(
        loads[1].unparked.restored >= 1,
        "the reload unparked {:?}",
        loads[1].unparked
    );

    w.finish().expect("the worker thread ended cleanly");
}
