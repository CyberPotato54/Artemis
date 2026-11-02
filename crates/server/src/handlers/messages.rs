//! `POST /v1/messages`.
use crate::anthropic::request::parse;
use crate::anthropic::response::{anthropic_body, stop_reason, Event, EventWriter};
use crate::app::{read_body, App, MAX_BODY};
use crate::convo::dropped_by_template;
use crate::error::{ApiError, Flavour};
use crate::exchange::{run_whole, Ending, Usage};
use crate::handlers::stream::{
    answer, log_generation, note_fields, prepare, Live, Prepared, Reply, StreamWriter,
};
use crate::ids::new_id;
use crate::logs::{LineGuard, RequestLine};
use crate::pipeline::Out;
use axum::body::Bytes;
use axum::extract::{Request, State};
use axum::response::Response;

const ENDPOINT: &str = "/v1/messages";

/// The handler: whatever the outcome, one log line (a stream's is written when the stream ends).
pub async fn messages(State(app): State<App>, req: Request) -> Response {
    let id = new_id("msg_");
    let mut guard = LineGuard::new(app.logger(), RequestLine::new(&id, ENDPOINT));
    let result = respond(&app, req, &id, guard.line()).await;
    answer(&app, guard, Flavour::Anthropic, result)
}

/// Read, take a preparation permit, parse, render, queue, run, answer. `line` is filled with what is known as it
/// becomes known, so an error after `Started` still logs the prompt.
async fn respond(
    app: &App,
    req: Request,
    id: &str,
    line: &mut RequestLine,
) -> Result<Reply<Events>, ApiError> {
    let bytes = read_body(app, req, MAX_BODY).await?;
    let permit = app.prep_permit().await?;
    let mut request = parse(&bytes)?;
    let defaults = app.config().defaults();
    let fields = request.fields(&defaults, false)?;
    note_fields(line, &fields);
    let convo = request.conversation(&fields)?;
    line.dropped_messages = dropped_by_template(&convo.messages);
    // The body and its parsed form are not needed from here: they must not sit in memory through the queue and the
    // generation.
    drop(bytes);
    drop(request);
    let p = prepare(
        app,
        &defaults,
        &fields,
        convo,
        Flavour::Anthropic,
        line,
        permit,
    )
    .await?;
    if fields.stream {
        let s = &p.ex.started;
        let writer = EventWriter::new(id, &p.model, (s.prompt_tokens, s.reused_tokens));
        return Ok(Reply::Stream(Box::new(Live::new(app, p, Events(writer)))));
    }
    let Prepared {
        ex,
        pipeline,
        model,
    } = p;
    let g = run_whole(ex, pipeline).await?;
    let name = stop_reason(&g.ending());
    log_generation(
        line,
        (&g.counts, &g.timings, &g.cache, g.spec.as_ref()),
        &g.usage(),
        name,
        &g.summary,
    );
    Ok(Reply::Json(anthropic_body(&g, &model, id)))
}

/// One event: `event: NAME` and its `data:` line.
fn sse(e: &Event) -> Bytes {
    Bytes::from(format!("event: {}\ndata: {}\n\n", e.0, e.1))
}

/// The events of an Anthropic stream, as SSE. A `ping` is the keep-alive, and nothing follows `message_stop`.
pub struct Events(EventWriter);

impl StreamWriter for Events {
    fn flavour(&self) -> Flavour {
        Flavour::Anthropic
    }

    fn first(&self) -> Vec<Bytes> {
        vec![sse(&self.0.start())]
    }

    fn push_all(&mut self, outs: &[Out]) -> Vec<Bytes> {
        self.0.push_all(outs).iter().map(sse).collect()
    }

    fn finish(&mut self, e: &Ending, u: &Usage) -> Vec<Bytes> {
        self.0.finish(e, u).iter().map(sse).collect()
    }

    fn error(&self, e: &ApiError) -> Bytes {
        sse(&self.0.error(e))
    }

    fn keepalive(&self) -> Bytes {
        sse(&self.0.ping())
    }

    fn terminal(&self) -> Option<Bytes> {
        None
    }

    fn finish_name(&self, e: &Ending) -> &'static str {
        stop_reason(e)
    }
}
