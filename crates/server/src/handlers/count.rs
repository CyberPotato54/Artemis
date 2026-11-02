//! `POST /v1/messages/count_tokens`: renders and tokenizes only, so it answers while the model is
//! unloaded and never touches the queue or the backend.
use crate::anthropic::request::parse;
use crate::app::{read_body, App, MAX_BODY};
use crate::convo::dropped_by_template;
use crate::error::{ApiError, Flavour};
use crate::ids::new_id;
use crate::logs::{cut_model, LineGuard, RequestLine};
use crate::prompt::build_prompt_blocking;
use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

const ENDPOINT: &str = "/v1/messages/count_tokens";

pub async fn count_tokens(State(app): State<App>, req: Request) -> Response {
    let mut guard = LineGuard::new(app.logger(), RequestLine::new(new_id("msg_"), ENDPOINT));
    let result = count(&app, req, guard.line()).await;
    let mut line = guard.take();
    let response = match result {
        Ok(n) => {
            line.status = 200;
            Json(json!({"input_tokens": n})).into_response()
        }
        Err(e) => {
            line.status = e.status;
            line.set_error(&e, Flavour::Anthropic);
            e.into_response(Flavour::Anthropic)
        }
    };
    app.logger().request(&line);
    response
}

/// The tokens of the prompt `/v1/messages` would send, generation prompt included. A prompt longer than the context
/// is counted all the same: the client asks how long it is, and the answer is what tells it to cut.
async fn count(app: &App, req: Request, line: &mut RequestLine) -> Result<u32, ApiError> {
    let bytes = read_body(app, req, MAX_BODY).await?;
    // Held until the prompt is built (the build owns it): this endpoint never queues, so the permits are its only
    // bound.
    let permit = app.prep_permit().await?;
    let mut request = parse(&bytes)?;
    let fields = request.fields(&app.config().defaults(), true)?;
    line.model = fields.model.as_deref().map(cut_model);
    line.ignored = fields.ignored.clone();
    line.ignored_more = fields.ignored_more;
    let convo = request.conversation(&fields)?;
    line.dropped_messages = dropped_by_template(&convo.messages);
    drop(bytes);
    drop(request);
    let (_, prompt) =
        build_prompt_blocking(app, convo, u32::MAX, Flavour::Anthropic, permit).await?;
    let n = u32::try_from(prompt.tokens.len()).unwrap_or(u32::MAX);
    line.prompt_tokens = Some(n);
    Ok(n)
}
