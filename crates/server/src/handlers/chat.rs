//! `POST /v1/chat/completions`.
use crate::app::{read_body, App, MAX_BODY};
use crate::convo::dropped_by_template;
use crate::error::{ApiError, Flavour};
use crate::exchange::{run_whole, Ending, Usage};
use crate::handlers::stream::{
    answer, log_generation, note_fields, prepare, Live, Prepared, Reply, StreamWriter,
};
use crate::ids::{new_id, unix_now};
use crate::logs::{LineGuard, RequestLine};
use crate::openai::request::parse;
use crate::openai::response::{finish_reason, openai_body, ChunkWriter};
use crate::pipeline::Out;
use axum::body::Bytes;
use axum::extract::{Request, State};
use axum::response::Response;
use serde_json::Value;

const ENDPOINT: &str = "/v1/chat/completions";

/// The handler: whatever the outcome, one log line (a stream's is written when the stream ends).
pub async fn chat(State(app): State<App>, req: Request) -> Response {
    let id = new_id("chatcmpl-");
    let mut guard = LineGuard::new(app.logger(), RequestLine::new(&id, ENDPOINT));
    let result = respond(&app, req, &id, guard.line()).await;
    answer(&app, guard, Flavour::OpenAi, result)
}

/// Read, take a preparation permit, parse, render, queue, run, answer. `line` is filled with what is known as it
/// becomes known, so an error after `Started` still logs the prompt.
async fn respond(
    app: &App,
    req: Request,
    id: &str,
    line: &mut RequestLine,
) -> Result<Reply<Chunks>, ApiError> {
    let bytes = read_body(app, req, MAX_BODY).await?;
    let permit = app.prep_permit().await?;
    let mut chat = parse(&bytes)?;
    let defaults = app.config().defaults();
    let fields = chat.fields(&defaults)?;
    note_fields(line, &fields);
    let convo = chat.conversation(&fields)?;
    line.dropped_messages = dropped_by_template(&convo.messages);
    // The body and its parsed form are not needed from here: they must not sit in memory through the queue and the
    // generation.
    drop(bytes);
    drop(chat);
    let p = prepare(
        app,
        &defaults,
        &fields,
        convo,
        Flavour::OpenAi,
        line,
        permit,
    )
    .await?;
    if fields.stream {
        let writer = ChunkWriter::new(
            id,
            &p.model,
            unix_now(),
            fields.include_usage,
            fields.logprobs.is_some(),
        );
        return Ok(Reply::Stream(Box::new(Live::new(app, p, Chunks(writer)))));
    }
    let Prepared {
        ex,
        pipeline,
        model,
    } = p;
    let g = run_whole(ex, pipeline).await?;
    let name = finish_reason(&g.ending());
    log_generation(
        line,
        (&g.counts, &g.timings, &g.cache, g.spec.as_ref()),
        &g.usage(),
        name,
        &g.summary,
    );
    Ok(Reply::Json(openai_body(&g, &model, id, unix_now())))
}

/// One `data:` event.
fn event(v: &Value) -> Bytes {
    Bytes::from(format!("data: {v}\n\n"))
}

const DONE: &[u8] = b"data: [DONE]\n\n";
const KEEP_ALIVE: &[u8] = b": keep-alive\n\n";

/// The chunks of an OpenAI stream, as `data:` events.
pub struct Chunks(ChunkWriter);

impl StreamWriter for Chunks {
    fn flavour(&self) -> Flavour {
        Flavour::OpenAi
    }

    fn first(&self) -> Vec<Bytes> {
        vec![event(&self.0.first())]
    }

    fn push_all(&mut self, outs: &[Out]) -> Vec<Bytes> {
        self.0.push_all(outs).iter().map(event).collect()
    }

    fn finish(&mut self, e: &Ending, u: &Usage) -> Vec<Bytes> {
        self.0.finish(e, u).iter().map(event).collect()
    }

    fn error(&self, e: &ApiError) -> Bytes {
        event(&self.0.error(e))
    }

    fn keepalive(&self) -> Bytes {
        Bytes::from_static(KEEP_ALIVE)
    }

    /// `[DONE]` follows the last event.
    fn terminal(&self) -> Option<Bytes> {
        Some(Bytes::from_static(DONE))
    }

    fn finish_name(&self, e: &Ending) -> &'static str {
        finish_reason(e)
    }
}
