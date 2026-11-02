//! `POST /v1/load` and `POST /v1/unload`: the key is checked by the guard before either runs.
use crate::app::{read_body, App, LIFECYCLE_BODY};
use crate::backend::ErrorKind;
use crate::error::{ApiError, ErrorType, Flavour};
use crate::exchange::{gone, shutting_down};
use crate::ids::new_id;
use crate::logs::{LifecycleLine, LineGuard, RequestLine};
use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

/// The response for `result`, with the request's line: `id`, `endpoint`, `status` and `error_kind`, nothing else (these
/// two take no prompt). A client that leaves while the call waits leaves the guard's line, 499.
fn answer(
    app: &App,
    guard: LineGuard,
    flavour: Flavour,
    result: Result<Value, ApiError>,
) -> Response {
    let mut line = guard.take();
    let response = match result {
        Ok(body) => {
            line.status = 200;
            Json(body).into_response()
        }
        Err(e) => {
            line.status = e.status;
            line.set_error(&e, flavour);
            e.into_response(flavour)
        }
    };
    app.logger().request(&line);
    response
}

/// Loads now if idle; 200 once the model is loaded, at once if it already is, and 503 with the engine's message when
/// the load fails.
pub async fn load(State(app): State<App>, req: Request) -> Response {
    let flavour = Flavour::of(req.uri().path());
    let guard = LineGuard::new(app.logger(), RequestLine::new(new_id("req_"), "/v1/load"));
    let result = load_now(&app, req).await;
    answer(&app, guard, flavour, result)
}

async fn load_now(app: &App, req: Request) -> Result<Value, ApiError> {
    read_body(app, req, LIFECYCLE_BODY).await?;
    // The check and the send are one step against `serve`'s stop (as `Exchange::begin`'s are), so a load never reaches
    // the engine behind the unload.
    let reply = {
        let _order = app.submit_order();
        if app.is_stopping() {
            return Err(shutting_down());
        }
        app.backend().load()
    };
    match reply.await {
        Ok(Ok(())) => Ok(json!({"status": "ready", "loaded": true})),
        Ok(Err(e)) => {
            if e.kind == ErrorKind::DeviceLost {
                app.device_lost(&e.message);
            }
            Err(ApiError::new(503, ErrorType::Unavailable, e.message))
        }
        // The worker dropped the reply: it is gone.
        Err(_) => Err(gone(app)),
    }
}

/// Answers every queued request 503 (`Queue::abort_all`, first, so none is handed to the engine behind the unload),
/// then unloads: the engine ends the request in flight `Unloaded`. 200 once the memory is freed.
pub async fn unload(State(app): State<App>, req: Request) -> Response {
    let flavour = Flavour::of(req.uri().path());
    let guard = LineGuard::new(app.logger(), RequestLine::new(new_id("req_"), "/v1/unload"));
    let result = unload_now(&app, req).await;
    answer(&app, guard, flavour, result)
}

async fn unload_now(app: &App, req: Request) -> Result<Value, ApiError> {
    read_body(app, req, LIFECYCLE_BODY).await?;
    app.queue().abort_all();
    match app.backend().unload().await {
        Ok(Ok(unloaded)) => {
            // An unload of a model that is not loaded is not one.
            if unloaded {
                app.logger().lifecycle(&LifecycleLine::Unload {
                    cause: "request".into(),
                });
                app.logger().lifecycle(&app.stats_line());
            }
            Ok(json!({"status": "idle", "loaded": false}))
        }
        Ok(Err(e)) => Err(ApiError::new(503, ErrorType::Unavailable, e.message)),
        Err(_) => Err(gone(app)),
    }
}
