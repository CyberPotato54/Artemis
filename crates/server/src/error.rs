//! The server's error type, and the one place a request body's field is read with its path in the error.
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::de::DeserializeOwned;
use serde_json::{json, Map, Value};
use std::fmt;

/// Error messages are one line, at most this many bytes.
const MAX_MESSAGE: usize = 500;

/// The kinds the two APIs name their errors by; the API layer maps each to its own `type` string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorType {
    InvalidRequest,
    Authentication,
    Permission,
    NotFound,
    TooLarge,
    Unavailable,
    Server,
}

/// The shape an error body takes, which the request path decides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavour {
    OpenAi,
    Anthropic,
}

impl Flavour {
    /// Anthropic's under `/v1/messages` (the path itself and anything below it), OpenAI's everywhere else.
    pub fn of(path: &str) -> Flavour {
        match path.strip_prefix("/v1/messages") {
            Some("") => Flavour::Anthropic,
            Some(rest) if rest.starts_with('/') => Flavour::Anthropic,
            _ => Flavour::OpenAi,
        }
    }
}

impl ErrorType {
    /// The `type` string in each API's vocabulary.
    pub fn name(self, f: Flavour) -> &'static str {
        match (self, f) {
            (ErrorType::InvalidRequest, _) => "invalid_request_error",
            (ErrorType::Authentication, _) => "authentication_error",
            (ErrorType::Permission, _) => "permission_error",
            (ErrorType::NotFound, _) => "not_found_error",
            (ErrorType::TooLarge, Flavour::Anthropic) => "request_too_large",
            (ErrorType::TooLarge, Flavour::OpenAi) => "invalid_request_error",
            (ErrorType::Unavailable, Flavour::Anthropic) => "overloaded_error",
            (ErrorType::Unavailable, Flavour::OpenAi) => "service_unavailable",
            (ErrorType::Server, Flavour::Anthropic) => "api_error",
            (ErrorType::Server, Flavour::OpenAi) => "server_error",
        }
    }
}

/// An error answered to the client. Nothing in it quotes the request's text.
#[derive(Clone, Debug, PartialEq)]
pub struct ApiError {
    pub status: u16,
    pub kind: ErrorType,
    pub message: String,
    pub param: Option<String>,
    pub code: Option<&'static str>,
}

impl ApiError {
    pub fn new(status: u16, kind: ErrorType, message: impl Into<String>) -> ApiError {
        ApiError {
            status,
            kind,
            message: one_line(message.into()),
            param: None,
            code: None,
        }
    }

    /// A 400 about the field `param`.
    pub fn invalid(param: impl Into<String>, msg: impl Into<String>) -> ApiError {
        ApiError {
            param: Some(param.into()),
            ..ApiError::new(400, ErrorType::InvalidRequest, msg)
        }
    }

    pub fn status(&self) -> u16 {
        self.status
    }

    /// The JSON body in the path's shape: Anthropic `{"type":"error","error":{"type","message"}}`,
    /// OpenAI `{"error":{"message","type","param","code"}}`. The message is one line of at most 500 bytes whoever built
    /// the error.
    pub fn body(&self, f: Flavour) -> Value {
        let message = one_line(self.message.clone());
        let kind = self.kind.name(f);
        match f {
            Flavour::Anthropic => {
                json!({"type": "error", "error": {"type": kind, "message": message}})
            }
            Flavour::OpenAi => json!({"error": {
                "message": message, "type": kind, "param": self.param, "code": self.code,
            }}),
        }
    }

    pub fn into_response(self, f: Flavour) -> Response {
        let status = StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(self.body(f))).into_response()
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.status, self.message)
    }
}

impl std::error::Error for ApiError {}

/// `s` on one line, cut to `MAX_MESSAGE` bytes at a character boundary.
fn one_line(s: String) -> String {
    let mut s = s.replace(['\r', '\n'], " ");
    if s.len() > MAX_MESSAGE {
        let mut end = MAX_MESSAGE;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        s.truncate(end);
    }
    s
}

/// Removes `key` from the body's map and deserializes it with `serde_path_to_error`; `Ok(None)` when absent or null.
/// A failure is a 400 whose `param` is `key` plus the error's path (`messages[2].content`, from `Path`'s display: a
/// path of `.` is `key`, one starting `[` is appended as is, any other gets a `.`) and whose message is built here from
/// the path and the expected type only, never from serde's text, which quotes the offending value.
pub fn take_field<T: DeserializeOwned>(
    map: &mut Map<String, Value>,
    key: &str,
) -> Result<Option<T>, ApiError> {
    let value = match map.shift_remove(key) {
        None | Some(Value::Null) => return Ok(None),
        Some(v) => v,
    };
    serde_path_to_error::deserialize::<_, T>(value)
        .map(Some)
        .map_err(|e| {
            let path = e.path().to_string();
            let param = match path.as_str() {
                "." => key.to_string(),
                p if p.starts_with('[') => format!("{key}{p}"),
                p => format!("{key}.{p}"),
            };
            let serde_text = e.inner().to_string();
            // The text is serde's and may quote the value, so only these fixed shapes are read from it.
            let message = if let Some((_, expected)) = serde_text
                .strip_prefix("invalid type: ")
                .and_then(|t| t.rsplit_once(", expected "))
            {
                format!("{param} has the wrong type, expected {expected}")
            } else if let Some(field) = serde_text
                .strip_prefix("missing field `")
                .and_then(|t| t.split_once('`'))
            {
                format!("{param}.{} is missing", field.0)
            } else {
                format!("{param} is invalid")
            };
            ApiError::invalid(param, message)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_are_one_line_and_at_most_500_bytes() {
        let e = ApiError::new(400, ErrorType::InvalidRequest, "a\nb\r\nc");
        assert_eq!(e.message, "a b  c");
        // 3-byte characters: 500 is not a multiple of 3, so the cut backs up to 498.
        let e = ApiError::new(400, ErrorType::InvalidRequest, "\u{20ac}".repeat(300));
        assert_eq!(e.message.len(), 498);
        assert!(e.message.chars().all(|c| c == '\u{20ac}'));
        let e = ApiError::new(400, ErrorType::InvalidRequest, "x".repeat(501));
        assert_eq!(e.message.len(), 500);
        // The body cuts too, for an error built field by field.
        let raw = ApiError {
            status: 500,
            kind: ErrorType::Server,
            message: format!("{}\n{}", "y".repeat(400), "z".repeat(400)),
            param: None,
            code: None,
        };
        for f in [Flavour::OpenAi, Flavour::Anthropic] {
            let b = raw.body(f);
            let m = b["error"]["message"].as_str().unwrap();
            assert!(m.len() <= 500 && !m.contains('\n'), "{m}");
        }
    }

    #[test]
    fn flavour_follows_the_path() {
        for p in [
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/v1/messages/nope",
        ] {
            assert_eq!(Flavour::of(p), Flavour::Anthropic, "{p}");
        }
        for p in [
            "/v1/messagesx",
            "/v1/chat/completions",
            "/v1/models",
            "/health",
            "/",
            "/v1/message",
        ] {
            assert_eq!(Flavour::of(p), Flavour::OpenAi, "{p}");
        }
    }

    #[test]
    fn bodies_by_flavour() {
        let e = ApiError::new(413, ErrorType::TooLarge, "too big");
        assert_eq!(
            e.body(Flavour::Anthropic),
            json!({"type": "error", "error": {"type": "request_too_large", "message": "too big"}})
        );
        assert_eq!(
            e.body(Flavour::OpenAi),
            json!({"error": {"message": "too big", "type": "invalid_request_error",
                "param": null, "code": null}})
        );
        let mut e = ApiError::invalid("messages[0]", "bad");
        e.code = Some("context_length_exceeded");
        assert_eq!(
            e.body(Flavour::OpenAi),
            json!({"error": {"message": "bad", "type": "invalid_request_error",
                "param": "messages[0]", "code": "context_length_exceeded"}})
        );
        let names = |k: ErrorType| (k.name(Flavour::Anthropic), k.name(Flavour::OpenAi));
        assert_eq!(
            names(ErrorType::Unavailable),
            ("overloaded_error", "service_unavailable")
        );
        assert_eq!(names(ErrorType::Server), ("api_error", "server_error"));
        assert_eq!(e.into_response(Flavour::OpenAi).status(), 400);
    }
}
