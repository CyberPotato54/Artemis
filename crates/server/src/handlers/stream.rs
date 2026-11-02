//! What the two generation endpoints share: the request's preparation up to the engine's `Started`, the
//! reply's two shapes, the request's log line, and the SSE driver. An endpoint brings its API's writer.
use crate::app::App;
use crate::backend::{CacheReport, Counts, GenEvent, SpecReport, Timings};
use crate::convo::Conversation;
use crate::error::{ApiError, Flavour};
use crate::exchange::{ending, step, usage_of, Ending, Exchange, Step, Usage};
use crate::ids::new_seed;
use crate::logs::{classify_divergence, cut_model, LineGuard, RequestLine};
use crate::params::{assemble, Defaults, Fields, ToolChoice, IGNORED_SHOWN};
use crate::pipeline::{Out, Pipeline, PipelineSetup, Summary};
use crate::prompt::build_prompt_blocking;
use crate::schema::ToolSet;
use axum::body::{Body, Bytes};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::stream;
use serde_json::Value;
use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::OwnedSemaphorePermit;
use tokio::time::{interval_at, Instant, Interval, MissedTickBehavior};

/// What the request's line knows once the fields are read: the model it named, whether it asked for a stream and what
/// was ignored. Called before the conversation is built, so a 400 from it still logs them.
pub fn note_fields(line: &mut RequestLine, fields: &Fields) {
    line.model = fields.model.as_deref().map(cut_model);
    line.stream = fields.stream;
    line.ignored = fields.ignored.clone();
    line.ignored_more = fields.ignored_more;
    if matches!(fields.tool_choice, ToolChoice::Unenforced(_))
        && !line.ignored.iter().any(|n| n == "tool_choice")
    {
        if line.ignored.len() < IGNORED_SHOWN {
            line.ignored.push("tool_choice".into());
        } else {
            line.ignored_more = line.ignored_more.saturating_add(1);
        }
    }
}

/// A request the engine has taken.
pub struct Prepared {
    pub ex: Exchange,
    pub pipeline: Pipeline,
    /// What the reply calls the model: the request's, or `model_id`.
    pub model: String,
}

/// Renders the prompt, queues the request and waits for the engine to take it. A streamed reply's headers go out only
/// then, so every error up to here is a real status. `line` gets what `Started` said. `permit` is the
/// request's preparation slot (`App::prep_permit`); the prompt build owns it and gives it up when the build ends, so
/// the wait for the engine does not hold one.
pub async fn prepare(
    app: &App,
    defaults: &Defaults,
    fields: &Fields,
    convo: Conversation,
    flavour: Flavour,
    line: &mut RequestLine,
    permit: OwnedSemaphorePermit,
) -> Result<Prepared, ApiError> {
    let (convo, prompt) =
        build_prompt_blocking(app, convo, app.config().ctx, flavour, permit).await?;
    // `tool_choice: none` took the tools out of the conversation, so they count as rendered only if they are left.
    let tools_rendered = !convo.tools.is_empty();
    let request = assemble(
        fields,
        &prompt,
        app.specials(),
        tools_rendered,
        defaults,
        new_seed(),
    );
    let pipeline = Pipeline::new(
        PipelineSetup {
            thinking: fields.thinking.on,
            tools: tools_rendered.then(|| Arc::new(ToolSet::new(&convo.tools))),
            stop: fields.stop.clone(),
        },
        app.tokenizer().clone(),
        app.specials(),
    );
    // The conversation is not needed past the pipeline: it must not sit in memory through the queue wait.
    drop(convo);
    // Known before the engine takes the request, so a request it never starts still logs its size.
    line.prompt_tokens = u32::try_from(prompt.tokens.len()).ok();
    line.cut_at = prompt.cut.as_ref().map(|c| c.at);
    line.user_query = prompt.cut.as_ref().map(|c| c.user_query);
    line.think_budget = request.think.as_ref().and_then(|t| t.budget);
    let ex = Exchange::begin(app, &line.id, request).await?;
    let s = &ex.started;
    line.prompt_tokens = Some(s.prompt_tokens);
    line.reused_tokens = Some(s.reused_tokens);
    line.restored = Some(s.restored.name().to_string());
    line.diverged_at = Some(s.diverged_at);
    line.cause = Some(classify_divergence(
        &prompt.tokens,
        prompt.cut.as_ref(),
        s.diverged_at,
        &s.restored,
        app.specials(),
        app.tokenizer(),
    ));
    Ok(Prepared {
        ex,
        pipeline,
        model: fields
            .model
            .clone()
            .unwrap_or_else(|| defaults.model_id.clone()),
    })
}

/// What a finished generation adds to the request's line; `finish` is the API's name for the ending.
pub fn log_generation(
    line: &mut RequestLine,
    (counts, timings, cache, spec): (&Counts, &Timings, &CacheReport, Option<&SpecReport>),
    u: &Usage,
    finish: &str,
    summary: &Summary,
) {
    line.measured(counts, timings, cache, spec);
    line.prefilled_tokens = Some(counts.prefilled);
    line.completion_tokens = Some(u.completion);
    line.reasoning_tokens = Some(u.reasoning);
    line.finish = Some(finish.to_string());
    line.note_repairs(&summary.repairs);
    note_calls(line, summary);
}

/// The calls the reply made and whether the think budget closed its reasoning: counts and a flag, never text.
fn note_calls(line: &mut RequestLine, summary: &Summary) {
    line.tool_calls = u32::try_from(summary.calls).unwrap_or(u32::MAX);
    line.malformed_calls = u32::try_from(summary.malformed_calls).unwrap_or(u32::MAX);
    line.budget_closed = summary.budget_closed;
}

/// An API's side of a stream: what its events look like on the wire.
pub trait StreamWriter {
    /// The API's error shape.
    fn flavour(&self) -> Flavour;
    /// The events the stream opens with.
    fn first(&self) -> Vec<Bytes>;
    /// The events for a run of pieces.
    fn push_all(&mut self, outs: &[Out]) -> Vec<Bytes>;
    /// The events that end the reply.
    fn finish(&mut self, e: &Ending, u: &Usage) -> Vec<Bytes>;
    /// A failure after the headers went out.
    fn error(&self, e: &ApiError) -> Bytes;
    /// What goes out every `Tuning::keepalive`.
    fn keepalive(&self) -> Bytes;
    /// What follows the last event, if anything (`[DONE]`).
    fn terminal(&self) -> Option<Bytes>;
    /// The request line's `finish`.
    fn finish_name(&self, e: &Ending) -> &'static str;
}

/// What a request comes to: a JSON body, or a stream whose headers the engine's `Started` has earned.
pub enum Reply<W> {
    Json(Value),
    Stream(Box<Live<W>>),
}

/// The response for `result`, whatever the outcome, with one log line (a stream's is written when the stream ends).
pub fn answer<W: StreamWriter + Send + 'static>(
    app: &App,
    guard: LineGuard,
    flavour: Flavour,
    result: Result<Reply<W>, ApiError>,
) -> Response {
    let mut line = guard.take();
    let response = match result {
        Ok(Reply::Json(body)) => {
            line.status = 200;
            Json(body).into_response()
        }
        Ok(Reply::Stream(mut live)) => {
            line.status = 200;
            live.line = Some(line);
            return event_stream(live);
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

/// A stream in progress. Dropping it (the client left, or the server is shutting down) drops the `Exchange`, which
/// cancels the engine's request within a token, and writes the request's line.
pub struct Live<W> {
    app: App,
    ex: Exchange,
    pipeline: Pipeline,
    writer: W,
    keepalive: Interval,
    /// Events ready to go out, in order.
    out: VecDeque<Bytes>,
    /// The terminal event has been queued: nothing more is read.
    ended: bool,
    /// The request's line, until it is written.
    line: Option<RequestLine>,
    /// The tokens the engine has sent so far, for a line that is written before `Done`.
    tokens: u32,
}

impl<W: StreamWriter> Live<W> {
    /// The opening events are queued at once, so the client's first read does not wait for the engine.
    pub fn new(app: &App, p: Prepared, writer: W) -> Live<W> {
        let every = app.tuning().keepalive;
        let mut keepalive = interval_at(Instant::now() + every, every);
        keepalive.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let out = VecDeque::from(writer.first());
        Live {
            app: app.clone(),
            ex: p.ex,
            pipeline: p.pipeline,
            writer,
            keepalive,
            out,
            ended: false,
            line: None,
            tokens: 0,
        }
    }

    /// The next bytes to send, or `None` when the stream is over.
    async fn next(&mut self) -> Option<Bytes> {
        loop {
            if let Some(b) = self.out.pop_front() {
                return Some(b);
            }
            if self.ended {
                return None;
            }
            tokio::select! {
                ev = self.ex.next() => self.on_event(ev),
                _ = self.keepalive.tick() => self.out.push_back(self.writer.keepalive()),
            }
        }
    }

    fn on_event(&mut self, ev: Option<GenEvent>) {
        if matches!(ev, Some(GenEvent::Token { .. })) {
            self.tokens += 1;
        }
        match step(&self.ex, &mut self.pipeline, ev) {
            Step::Pieces(outs) => self.events(&outs),
            Step::Done {
                finish,
                counts,
                timings,
                cache,
                spec,
                summary,
                tail,
            } => {
                self.events(&tail);
                let usage = usage_of(&counts, &summary);
                let ending = ending(&finish, &summary);
                if let Some(line) = &mut self.line {
                    let name = self.writer.finish_name(&ending);
                    log_generation(
                        line,
                        (&counts, &timings, &cache, spec.as_ref()),
                        &usage,
                        name,
                        &summary,
                    );
                }
                self.out.extend(self.writer.finish(&ending, &usage));
                self.end();
            }
            Step::Fail(e) => self.fail(&e),
        }
    }

    fn events(&mut self, outs: &[Out]) {
        self.out.extend(self.writer.push_all(outs));
    }

    /// The failure goes out as an event (the status is already 200), then the stream ends.
    fn fail(&mut self, e: &ApiError) {
        if let Some(line) = &mut self.line {
            line.set_error(e, self.writer.flavour());
        }
        self.out.push_back(self.writer.error(e));
        self.end();
    }

    /// The terminal bytes follow the last event; the line is written first, so a client that has read them finds it.
    fn end(&mut self) {
        self.log();
        self.out.extend(self.writer.terminal());
        self.ended = true;
    }
}

impl<W> Live<W> {
    /// Writes the request's line, once.
    fn log(&mut self) {
        if let Some(line) = self.line.take() {
            self.app.logger().request(&line);
        }
    }
}

impl<W> Drop for Live<W> {
    fn drop(&mut self) {
        if let Some(line) = &mut self.line {
            // The client left before the end (the status is the 200 already sent): what the line knows of the run.
            let summary = self.pipeline.summary();
            line.finish = Some("cancelled".to_string());
            line.completion_tokens = Some(self.tokens);
            line.reasoning_tokens = Some(summary.reasoning_tokens);
            note_calls(line, &summary);
            line.note_repairs(&summary.repairs);
        }
        self.log();
    }
}

/// The SSE response over `live`. The body is the stream: when hyper drops it, `live` goes with it.
fn event_stream<W: StreamWriter + Send + 'static>(live: Box<Live<W>>) -> Response {
    let body = Body::from_stream(stream::unfold(live, |mut live| async move {
        let bytes = live.next().await?;
        Some((Ok::<_, Infallible>(bytes), live))
    }));
    Response::builder()
        .status(200)
        .header(CONTENT_TYPE, "text/event-stream")
        .header(CACHE_CONTROL, "no-cache")
        .body(body)
        .expect("a valid response")
}
