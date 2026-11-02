//! The accept loop. `axum::serve` sets no timer, so hyper's header timeout never runs, and it caps nothing. This
//! loop gives each connection a header-read timeout, takes a permit from a semaphore before it accepts (the connections
//! past `Tuning::max_connections` wait in the OS backlog), and hands the router the peer address for the log.
//!
//! The header timeout covers only the reading of a request head, including the wait for the next one on a kept-alive
//! connection. hyper starts it when it begins to read a head and does not read the next head until the response is
//! written, so a stream or a long prefill is not cut by it; `tests/limits.rs` checks both.
use crate::app::Tuning;
use axum::extract::ConnectInfo;
use axum::Router;
use hyper::server::conn::http1;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;

/// How long the loop waits after a failed `accept` (a full file table, say) before it tries again.
const ACCEPT_RETRY: Duration = Duration::from_millis(100);

/// Whether a failed `accept` is about the one connection (it was refused, aborted or reset while queued) and not the
/// listener (no file handles, no memory): the first kind is retried at once, as `axum::serve` does.
fn accept_error_is_per_connection(kind: std::io::ErrorKind) -> bool {
    use std::io::ErrorKind::{ConnectionAborted, ConnectionRefused, ConnectionReset};
    matches!(
        kind,
        ConnectionRefused | ConnectionAborted | ConnectionReset
    )
}

/// Serves `router` on `listener` until `stop` resolves, then stops accepting and waits for the connections in flight
/// to finish (idle kept-alive connections close at once; a request in progress completes). Dropping the future
/// stops accepting and signals the connections to finish gracefully; it does not abort a response in flight (as with
/// `axum::serve`), so `serve`'s grace limit works by returning, after which the runtime ends the tasks.
pub(crate) async fn run(
    listener: TcpListener,
    router: Router,
    tuning: &Tuning,
    stop: impl Future<Output = ()> + Send + 'static,
) {
    let slots = Arc::new(Semaphore::new(tuning.max_connections.max(1)));
    let graceful = GracefulShutdown::new();
    let header_timeout = tuning.header_timeout;
    let mut stop = pin!(stop);
    loop {
        // The permit comes first, so a connection past the cap is never accepted: it waits in the backlog.
        let permit = tokio::select! {
            () = &mut stop => break,
            permit = slots.clone().acquire_owned() => match permit {
                Ok(permit) => permit,
                // The semaphore is never closed.
                Err(_) => break,
            },
        };
        let (stream, peer) = tokio::select! {
            () = &mut stop => break,
            accepted = listener.accept() => match accepted {
                Ok(accepted) => accepted,
                Err(e) => {
                    drop(permit);
                    if accept_error_is_per_connection(e.kind()) {
                        // A peer that left while queued (a reset on Windows): nothing wrong with the server, so no
                        // pause and no warning for a remote host to provoke.
                        tracing::debug!("a queued connection failed: {:?}", e.kind());
                        continue;
                    }
                    tracing::warn!("accepting a connection failed: {:?}", e.kind());
                    tokio::select! {
                        () = &mut stop => break,
                        () = tokio::time::sleep(ACCEPT_RETRY) => {}
                    }
                    continue;
                }
            },
        };
        let service = TowerToHyperService::new(
            router
                .clone()
                .layer(axum::Extension(ConnectInfo::<SocketAddr>(peer))),
        );
        let conn = http1::Builder::new()
            .timer(TokioTimer::new())
            .header_read_timeout(header_timeout)
            .keep_alive(true)
            .serve_connection(TokioIo::new(stream), service);
        let conn = graceful.watch(conn);
        tokio::spawn(async move {
            // Held until the connection ends.
            let _permit = permit;
            if let Err(e) = conn.await {
                // A client that left, a header timeout: not the server's fault, and nothing in it is request text.
                tracing::debug!("a connection ended with an error: {e}");
            }
        });
    }
    graceful.shutdown().await;
}

#[cfg(test)]
mod tests {
    use super::accept_error_is_per_connection;
    use std::io::ErrorKind;

    #[test]
    fn a_connection_that_left_is_retried_at_once_and_a_listener_fault_is_not() {
        for kind in [
            ErrorKind::ConnectionRefused,
            ErrorKind::ConnectionAborted,
            ErrorKind::ConnectionReset,
        ] {
            assert!(accept_error_is_per_connection(kind), "{kind:?}");
        }
        for kind in [
            ErrorKind::OutOfMemory,
            ErrorKind::PermissionDenied,
            ErrorKind::Other,
            ErrorKind::TimedOut,
        ] {
            assert!(!accept_error_is_per_connection(kind), "{kind:?}");
        }
    }
}
