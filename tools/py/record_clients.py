"""A recording HTTP stub for OpenCode and Claude Code. Standard library only.

It answers both APIs (OpenAI chat completions, Anthropic messages) with canned replies, streaming and not, and writes
every request it receives to OUT/NNN-<route>.json: method, path, query, headers, the credential's class and the body.

Credentials: each request's `Authorization: Bearer` / `x-api-key` value is compared with the key file and classified
`dummy` (equal), `none` (absent) or `other`, BEFORE any header is masked. Only the class is ever stored, printed or
logged; nothing about an `other` value is kept, not its length or a prefix. A request of class `other` gets 401, writes
OUT/ABORT and stores only its method, path, class and masked headers. The stub keeps serving so the capture script can
read the report and stop the task.

The stub binds 127.0.0.1 only (there is no host option) and refuses port 8080. It prints nothing per request.

Usage: python -I tools/py/record_clients.py --port 8090 --key-file PATH --out DIR --read-path PATH [--report FILE]
       python -I tools/py/record_clients.py --selftest
"""

import argparse
import hmac
import json
import os
import re
import signal
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
REFUSED_PORT = 8080
MODEL = "qwen3.6-35b-a3b"
THINK = "Checking the file."
ANSWER = "The file lists three stages."
CANARY_PROMPT = "Say ok."
CANARY_ANSWER = "ok."
MASKED = "<masked>"
MASK_NAMES = ("authorization", "x-api-key", "cookie", "proxy-authorization")
MASK_PARTS = ("key", "token", "secret", "auth")
READ_NAMES = ("read", "read_file", "view")
MAX_BODY = 64 * 1024 * 1024


def classify(headers, key):
    """`dummy`, `none` or `other` for the credentials in these headers. Runs before any masking."""
    values = []
    for auth in headers.get_all("Authorization") or []:
        auth = auth.strip()
        if auth[:7].lower() == "bearer ":
            auth = auth[7:].strip()
        values.append(auth)
    for api_key in headers.get_all("x-api-key") or []:
        values.append(api_key.strip())
    for name in set(headers.keys()):
        low = name.lower()
        if low in ("authorization", "x-api-key") or not is_masked_name(low):
            continue
        # Any other credential-looking header (cookie, proxy-authorization, *key*, *token*, *secret*, *auth*) counts
        # too: a non-empty value that is not the dummy key is `other`.
        for v in headers.get_all(name) or []:
            v = v.strip()
            if v[:7].lower() == "bearer ":
                v = v[7:].strip()
            values.append(v)
    values = [v for v in values if v]
    if not values:
        return "none"
    ok = all(hmac.compare_digest(v.encode("utf-8", "replace"), key.encode("utf-8")) for v in values)
    return "dummy" if ok else "other"


def is_masked_name(low):
    return low in MASK_NAMES or any(part in low for part in MASK_PARTS)


def masked_headers(headers):
    out = {}
    for name, value in headers.items():
        if is_masked_name(name.lower()):
            value = MASKED
        out[name] = out[name] + ", " + value if name in out else value
    return out


def route_name(path):
    if path == "/health":
        return "health"
    if path == "/v1/models":
        return "models"
    if path.startswith("/v1/models/"):
        return "model"
    return {
        "/v1/chat/completions": "chat",
        "/v1/messages": "messages",
        "/v1/messages/count_tokens": "count_tokens",
    }.get(path, "unknown")


class State:
    """What the handler threads share: the key, the output directory, the counters and the report."""

    def __init__(self, key, out, read_path, report):
        self.key = key
        self.out = out
        self.read_path = read_path
        self.report = report
        self.lock = threading.Lock()
        self.requests = 0
        self.classes = {"dummy": 0, "none": 0, "other": 0}
        self.hosts = set()
        self.abort = False
        os.makedirs(out, exist_ok=True)
        # Never overwrite an earlier capture in a reused directory: number on from its highest file.
        taken = [int(m.group(1)) for n in os.listdir(out) if (m := re.match(r"(\d{3,})-.*\.json$", n))]
        self.seq = max(taken, default=0)

    def record(self, cls, host, route, rec):
        """Counts the request, writes its file and the report. Returns the sequence number."""
        with self.lock:
            self.seq += 1
            self.requests += 1
            self.classes[cls] += 1
            if host:
                self.hosts.add(host)
            rec = {"seq": self.seq, **rec}
            self._write(os.path.join(self.out, f"{self.seq:03d}-{route}.json"), rec)
            if cls == "other":
                self.abort = True
                self._write(os.path.join(self.out, "ABORT"), rec)
            self.write_report()
            return self.seq

    def write_report(self):
        if not self.report:
            return
        data = {
            "requests": self.requests,
            "classes": dict(self.classes),
            "hosts": sorted(self.hosts),
            "abort": self.abort,
        }
        self._write(self.report, data)

    @staticmethod
    def _write(path, obj):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)


# ---- the canned replies ----


def nid():
    return uuid.uuid4().hex[:24]


def message_texts(msg):
    """The text parts of one message's content (a string, or a list of text blocks)."""
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return []


def last_is_tool_result(api, body):
    msgs = body.get("messages")
    if isinstance(msgs, list):
        # Claude Code 2.1.302 appends messages with role "system" after the tool results; they do not count.
        msgs = [m for m in msgs if not (isinstance(m, dict) and m.get("role") == "system")] if api == "anthropic" else msgs
    if not isinstance(msgs, list) or not msgs or not isinstance(msgs[-1], dict):
        return False
    last = msgs[-1]
    if api == "openai":
        return last.get("role") in ("tool", "function")
    content = last.get("content")
    if last.get("role") != "user" or not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def is_canary(body):
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return False
    users = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    return bool(users) and any(t.strip() == CANARY_PROMPT for t in message_texts(users[-1]))


def schema_type(schema):
    """The JSON type of a property schema. An untyped one (`$ref`, no type) counts as a string."""
    if not isinstance(schema, dict):
        return "string"
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    if isinstance(t, str):
        return t
    for k in ("anyOf", "oneOf"):
        for option in schema.get(k) or []:
            inner = schema_type(option)
            if inner != "null":
                return inner
    return "string"


def tool_arguments(schema, read_path):
    args = {}
    if not isinstance(schema, dict):
        return args
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    for name in schema.get("required") or []:
        if not isinstance(name, str):
            continue
        t = schema_type(props.get(name))
        if t == "string":
            low = name.lower()
            args[name] = read_path if ("path" in low or "file" in low) else "x"
        elif t in ("integer", "number"):
            args[name] = 0
        elif t == "boolean":
            args[name] = False
        elif t == "array":
            args[name] = []
        elif t == "null":
            args[name] = None
        else:
            args[name] = {}
    return args


def pick_tool(api, body, read_path):
    """The call to make: the first `read`/`read_file`/`view` tool, else the first. None without tools."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return None
    found = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        spec = tool.get("function") if api == "openai" else tool
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
            continue
        schema = spec.get("parameters") if api == "openai" else spec.get("input_schema")
        found.append((spec["name"], schema))
    if not found:
        return None
    name, schema = next(((n, s) for n, s in found if n.lower() in READ_NAMES), found[0])
    return {"name": name, "arguments": tool_arguments(schema, read_path)}


def plan_reply(api, body, read_path):
    if is_canary(body):
        return {"text": CANARY_ANSWER}
    if not last_is_tool_result(api, body):
        tool = pick_tool(api, body, read_path)
        if tool:
            return {"tool": tool}
    return {"text": ANSWER}


def fragments(text):
    """Two argument fragments for a streamed call (one when it is too short to split)."""
    if len(text) < 4:
        return [text]
    k = len(text) // 2
    return [text[:k], text[k:]]


def sse(obj, event=None):
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(obj)}\n\n"


def tokens(text):
    return max(1, len(text) // 4)


def openai_json(plan, model, prompt_tokens):
    call = plan.get("tool")
    message = {"role": "assistant", "content": None if call else plan["text"], "reasoning_content": THINK}
    out = plan["text"] if not call else json.dumps(call["arguments"])
    if call:
        message["tool_calls"] = [
            {
                "id": "call_" + nid(),
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
            }
        ]
    completion = tokens(out) + tokens(THINK)
    return {
        "id": "chatcmpl-" + nid(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": "tool_calls" if call else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "total_tokens": prompt_tokens + completion,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": tokens(THINK)},
        },
    }


def openai_sse(plan, model, prompt_tokens, include_usage):
    call = plan.get("tool")
    cid, created = "chatcmpl-" + nid(), int(time.time())

    def chunk(delta, finish=None, choices=True, usage=None):
        obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
        obj["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}] if choices else []
        if usage is not None:
            obj["usage"] = usage
        return sse(obj)

    events = [chunk({"role": "assistant", "content": ""}), chunk({"reasoning_content": THINK})]
    if call:
        args = json.dumps(call["arguments"])
        first = {"index": 0, "id": "call_" + nid(), "type": "function", "function": {"name": call["name"], "arguments": ""}}
        events.append(chunk({"tool_calls": [first]}))
        for part in fragments(args):
            events.append(chunk({"tool_calls": [{"index": 0, "function": {"arguments": part}}]}))
        out = args
    else:
        for part in fragments(plan["text"]):
            events.append(chunk({"content": part}))
        out = plan["text"]
    events.append(chunk({}, "tool_calls" if call else "stop"))
    if include_usage:
        completion = tokens(out) + tokens(THINK)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "total_tokens": prompt_tokens + completion,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": tokens(THINK)},
        }
        events.append(chunk(None, choices=False, usage=usage))
    events.append("data: [DONE]\n\n")
    return events


def anthropic_usage(prompt_tokens, output_tokens):
    return {
        "input_tokens": prompt_tokens,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": output_tokens,
    }


def anthropic_json(plan, model, prompt_tokens):
    call = plan.get("tool")
    content = [{"type": "thinking", "thinking": THINK, "signature": ""}]
    if call:
        content.append({"type": "tool_use", "id": "toolu_" + nid(), "name": call["name"], "input": call["arguments"]})
        out = json.dumps(call["arguments"])
    else:
        content.append({"type": "text", "text": plan["text"]})
        out = plan["text"]
    return {
        "id": "msg_" + nid(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "tool_use" if call else "end_turn",
        "stop_sequence": None,
        "usage": anthropic_usage(prompt_tokens, tokens(out) + tokens(THINK)),
    }


def anthropic_sse(plan, model, prompt_tokens):
    call = plan.get("tool")
    out = json.dumps(call["arguments"]) if call else plan["text"]

    def ev(name, **fields):
        return sse({"type": name, **fields}, event=name)

    msg = {
        "id": "msg_" + nid(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": anthropic_usage(prompt_tokens, 1),
    }
    events = [ev("message_start", message=msg), ev("ping")]
    events.append(ev("content_block_start", index=0, content_block={"type": "thinking", "thinking": "", "signature": ""}))
    events.append(ev("content_block_delta", index=0, delta={"type": "thinking_delta", "thinking": THINK}))
    events.append(ev("content_block_delta", index=0, delta={"type": "signature_delta", "signature": ""}))
    events.append(ev("content_block_stop", index=0))
    if call:
        block = {"type": "tool_use", "id": "toolu_" + nid(), "name": call["name"], "input": {}}
        events.append(ev("content_block_start", index=1, content_block=block))
        for part in fragments(out):
            events.append(ev("content_block_delta", index=1, delta={"type": "input_json_delta", "partial_json": part}))
    else:
        events.append(ev("content_block_start", index=1, content_block={"type": "text", "text": ""}))
        for part in fragments(out):
            events.append(ev("content_block_delta", index=1, delta={"type": "text_delta", "text": part}))
    events.append(ev("content_block_stop", index=1))
    delta = {"stop_reason": "tool_use" if call else "end_turn", "stop_sequence": None}
    events.append(ev("message_delta", delta=delta, usage={"output_tokens": tokens(out) + tokens(THINK)}))
    events.append(ev("message_stop"))
    return events


def model_entry(model_id):
    created = int(time.time())
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created))
    return {
        "id": model_id,
        "object": "model",
        "created": created,
        "owned_by": "artemis",
        "type": "model",
        "display_name": model_id,
        "created_at": iso,
    }


# ---- the server ----


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "artemis-stub"

    def log_message(self, *args):
        pass

    def read_body(self):
        try:
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                out = b""
                while True:
                    size = int(self.rfile.readline(65537).split(b";")[0].strip() or b"0", 16)
                    if size == 0:
                        while self.rfile.readline(65537) not in (b"\r\n", b"\n", b""):
                            pass
                        break
                    out += self.rfile.read(size)
                    self.rfile.read(2)
                    if len(out) > MAX_BODY:
                        break
                raw = out
            else:
                raw = self.rfile.read(min(int(self.headers.get("Content-Length") or 0), MAX_BODY))
            if (self.headers.get("Content-Encoding") or "").lower() in ("gzip", "deflate"):
                raw = zlib.decompress(raw, 32 + zlib.MAX_WBITS)
            return raw
        except (ValueError, zlib.error, OSError):
            return b""

    def send_json(self, code, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def send_events(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        for e in events:
            self.wfile.write(e.encode("utf-8"))
            self.wfile.flush()

    def error(self, code, kind, message, path):
        if path.startswith("/v1/messages"):
            self.send_json(code, {"type": "error", "error": {"type": kind, "message": message}})
        else:
            self.send_json(code, {"error": {"message": message, "type": kind, "param": None, "code": None}})

    def handle_any(self):
        try:
            self.dispatch()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = handle_any

    def dispatch(self):
        state = self.server.state
        raw = self.read_body()
        split = urllib.parse.urlsplit(self.path)
        path = split.path
        route = route_name(path)
        cls = classify(self.headers, state.key)  # before anything is masked or stored
        host = self.headers.get("Host") or ""
        rec = {"method": self.command, "path": path}
        if cls == "other":
            rec.update({"credential": cls, "headers": masked_headers(self.headers)})
            state.record(cls, host, route, rec)
            self.error(401, "authentication_error", "Unauthorized", path)
            return
        text = raw.decode("utf-8", "replace")
        try:
            body = json.loads(text) if text.strip() else None
        except ValueError:
            body = text
        rec.update(
            {"query": split.query, "headers": masked_headers(self.headers), "credential": cls, "body": body}
        )
        state.record(cls, host, route, rec)

        if path.startswith("/v1") and cls != "dummy":
            self.error(401, "authentication_error", "Invalid or missing API key", path)
            return
        if self.command == "GET" and path == "/health":
            self.send_json(
                200,
                {
                    "service": "artemis",
                    "status": "ready",
                    "version": "0.0.0",
                    "loaded": True,
                    "model": MODEL,
                    "max_context": 65536,
                    "api_key": True,
                },
            )
        elif self.command == "GET" and path == "/v1/models":
            entry = model_entry(MODEL)
            self.send_json(
                200, {"object": "list", "data": [entry], "has_more": False, "first_id": MODEL, "last_id": MODEL}
            )
        elif self.command == "GET" and path.startswith("/v1/models/") and len(path) > len("/v1/models/"):
            self.send_json(200, model_entry(urllib.parse.unquote(path[len("/v1/models/"):])))
        elif self.command == "POST" and path in ("/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens"):
            if not isinstance(body, dict):
                self.error(400, "invalid_request_error", "The body must be a JSON object", path)
                return
            self.reply(path, body, raw)
        else:
            self.error(404, "not_found_error", "Not found", path)

    def reply(self, path, body, raw):
        state = self.server.state
        prompt_tokens = tokens(raw.decode("utf-8", "replace"))
        if path == "/v1/messages/count_tokens":
            self.send_json(200, {"input_tokens": prompt_tokens})
            return
        api = "openai" if path == "/v1/chat/completions" else "anthropic"
        model = body.get("model") if isinstance(body.get("model"), str) and body.get("model") else MODEL
        plan = plan_reply(api, body, state.read_path)
        stream = body.get("stream") is True
        if api == "openai":
            opts = body.get("stream_options")
            usage = isinstance(opts, dict) and opts.get("include_usage") is True
            if stream:
                self.send_events(openai_sse(plan, model, prompt_tokens, usage))
            else:
                self.send_json(200, openai_json(plan, model, prompt_tokens))
        elif stream:
            self.send_events(anthropic_sse(plan, model, prompt_tokens))
        else:
            self.send_json(200, anthropic_json(plan, model, prompt_tokens))


def make_server(port, key_file, out, read_path, report=None):
    if port == REFUSED_PORT:
        raise SystemExit("record_clients: port 8080 is refused (the production server listens there)")
    if not 0 <= port <= 65535:
        raise SystemExit("record_clients: bad port")
    with open(key_file, encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        raise SystemExit("record_clients: the key file is empty")
    server = ThreadingHTTPServer((HOST, port), Handler)  # 127.0.0.1 only: there is no host option
    server.daemon_threads = True
    server.state = State(key, out, read_path, report)
    server.state.write_report()
    return server


# ---- the self-test ----


class SelfTestError(Exception):
    pass


def check(cond, what):
    if not cond:
        raise SelfTestError(what)


def selftest():
    sep = chr(92)
    key = "artemis-capture-dummy-key-0000000000000000"
    bad_bearer, bad_api = "not-the-key-0123456789", "sk-ant-selftest-0000"
    bad_cookie, bad_token = "sid=zq81vxk4wm27pj", "tok-ffq93bd8hy61"  # gitleaks:allow (invented, for this selftest only)
    read_path = "Z:" + sep + "scratch" + sep + "notes.txt"
    tmp = tempfile.mkdtemp(prefix="rc-")
    key_file = os.path.join(tmp, "key.txt")
    with open(key_file, "w", encoding="utf-8") as f:
        f.write(key + "\r\n")
    out = os.path.join(tmp, "out")
    report = os.path.join(tmp, "report.json")  # beside OUT, so the leak scan reads it separately
    check_refused = False
    try:
        make_server(REFUSED_PORT, key_file, out, read_path)
    except SystemExit:
        check_refused = True
    check(check_refused, "port 8080 was not refused")

    server = make_server(0, key_file, out, read_path, report)
    port = server.server_address[1]
    check(server.server_address[0] == HOST, "the stub is not bound to 127.0.0.1")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://{HOST}:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through a proxy
    responses = []  # every reply body, for the leak scan

    def call(method, path, headers=None, body=None):
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode("utf-8"))
        req = urllib.request.Request(base + path, data=data, method=method, headers=headers or {})
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with opener.open(req, timeout=10) as r:
                code, payload = r.status, r.read()
        except urllib.error.HTTPError as e:
            code, payload = e.code, e.read()
        responses.append(payload)
        return code, payload

    def jcall(*a, **kw):
        code, payload = call(*a, **kw)
        try:
            return code, json.loads(payload)
        except ValueError:
            raise SelfTestError(f"{a[0]} {a[1]}: reply is not JSON")

    def events(payload):
        out_events = []
        for block in payload.decode("utf-8").split("\n\n"):
            if not block.strip():
                continue
            name, data = None, None
            for line in block.split("\n"):
                if line.startswith("event: "):
                    name = line[7:]
                elif line.startswith("data: "):
                    data = line[6:]
            check(data is not None, "an SSE block without data")
            out_events.append((name, data if data == "[DONE]" else json.loads(data)))
        return out_events

    bearer = {"Authorization": "Bearer " + key}
    xkey = {"x-api-key": key}
    chat, msgs = "/v1/chat/completions", "/v1/messages"
    oa_tool = {
        "type": "function",
        "function": {
            "name": "read",
            "description": "d",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "offset": {"type": "integer"},
                    "flag": {"type": "boolean"},
                    "note": {"type": "string"},
                    "extra": {"type": "object"},
                    "list": {"type": "array"},
                },
                "required": ["filePath", "offset", "flag", "note", "extra", "list"],
            },
        },
    }
    other_tool = {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}
    an_tool = {
        "name": "Read",
        "description": "d",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    }
    want_args = {"filePath": read_path, "offset": 0, "flag": False, "note": "x", "extra": {}, "list": []}

    # /health: no auth needed; the body has the keys of the server's /health.
    code, h = jcall("GET", "/health")
    check(code == 200 and h["service"] == "artemis" and h["loaded"] is True, "/health")
    check(set(h) == {"service", "status", "version", "loaded", "model", "max_context", "api_key"}, "/health keys")
    code, _ = jcall("GET", "/health", bearer)
    check(code == 200, "/health with a key")

    # /v1/models and /v1/models/{id}: 401 without a key, both credentials accepted with one.
    code, e = jcall("GET", "/v1/models")
    check(code == 401 and "error" in e, "/v1/models without a key")
    code, m = jcall("GET", "/v1/models", bearer)
    check(code == 200 and m["object"] == "list" and m["data"][0]["id"] == MODEL, "/v1/models")
    code, m = jcall("GET", "/v1/models/some-model", xkey)
    check(code == 200 and m["id"] == "some-model" and m["object"] == "model", "/v1/models/{id}")

    # OpenAI chat, non-streaming: text, a tool call, the answer after the tool result, the canary.
    ask = [{"role": "user", "content": "Read notes.txt"}]
    code, r = jcall("POST", chat, None, {"model": "m", "messages": ask})
    check(code == 401 and "error" in r, "chat without a key")
    code, r = jcall("POST", chat, bearer, {"model": "m", "messages": ask})
    msg = r["choices"][0]["message"]
    check(code == 200 and msg["content"] == ANSWER and msg["reasoning_content"] == THINK, "chat text")
    check("tool_calls" not in msg and r["choices"][0]["finish_reason"] == "stop" and r["model"] == "m", "chat shape")
    code, r = jcall("POST", chat, bearer, {"messages": ask, "tools": [other_tool, oa_tool]})
    msg, fin = r["choices"][0]["message"], r["choices"][0]["finish_reason"]
    check(fin == "tool_calls" and msg["content"] is None and msg["tool_calls"][0]["type"] == "function", "chat tool")
    check(msg["tool_calls"][0]["function"]["name"] == "read", "chat picks the read tool")
    check(json.loads(msg["tool_calls"][0]["function"]["arguments"]) == want_args, "chat tool arguments")
    code, r = jcall("POST", chat, bearer, {"messages": ask, "tools": [other_tool]})
    check(r["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "bash", "chat picks the first tool")
    after = ask + [{"role": "assistant", "content": None}, {"role": "tool", "tool_call_id": "c", "content": "t"}]
    code, r = jcall("POST", chat, bearer, {"messages": after, "tools": [oa_tool]})
    check(r["choices"][0]["message"]["content"] == ANSWER, "chat answer after a tool result")
    canary = [{"role": "user", "content": [{"type": "text", "text": CANARY_PROMPT}]}]
    code, r = jcall("POST", chat, bearer, {"messages": canary, "tools": [oa_tool]})
    check(r["choices"][0]["message"]["content"] == CANARY_ANSWER, "chat canary")

    # OpenAI chat, streaming.
    code, payload = call("POST", chat, bearer, {"messages": ask, "stream": True, "stream_options": {"include_usage": True}})
    ev = [d for _, d in events(payload)]
    check(code == 200 and ev[-1] == "[DONE]", "chat stream ends with [DONE]")
    check(ev[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}, "chat stream first chunk")
    check(ev[1]["choices"][0]["delta"] == {"reasoning_content": THINK}, "chat stream reasoning")
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in ev[2:-2])
    check(text == ANSWER and ev[-3]["choices"][0]["finish_reason"] == "stop", "chat stream text")
    check(ev[-2]["choices"] == [] and ev[-2]["usage"]["total_tokens"] > 0, "chat stream usage chunk")
    code, payload = call("POST", chat, bearer, {"messages": ask, "stream": True})
    check(b'"usage"' not in payload, "chat stream usage without include_usage")
    code, payload = call("POST", chat, bearer, {"messages": ask, "tools": [oa_tool], "stream": True})
    ev = [d for _, d in events(payload)]
    calls = [c["choices"][0]["delta"]["tool_calls"][0] for c in ev[2:-2]]
    check(calls[0]["id"].startswith("call_") and calls[0]["function"] == {"name": "read", "arguments": ""}, "stream call")
    check(all("id" not in c for c in calls[1:]) and calls[1]["index"] == 0, "stream call fragments")
    args = "".join(c["function"]["arguments"] for c in calls)
    check(json.loads(args) == want_args and ev[-2]["choices"][0]["finish_reason"] == "tool_calls", "stream call args")

    # Anthropic messages (x-api-key), with a query.
    ask_a = [{"role": "user", "content": "Read notes.txt"}]
    code, r = jcall("POST", msgs + "?beta=true", None, {"model": "m", "max_tokens": 5, "messages": ask_a})
    check(code == 401 and r["type"] == "error", "messages without a key")
    code, r = jcall("POST", msgs + "?beta=true", xkey, {"model": "m", "max_tokens": 5, "messages": ask_a})
    check(code == 200 and r["type"] == "message" and r["stop_reason"] == "end_turn", "messages text")
    check([b["type"] for b in r["content"]] == ["thinking", "text"] and r["content"][0]["signature"] == "", "blocks")
    check(r["content"][1]["text"] == ANSWER and r["usage"]["cache_creation_input_tokens"] == 0, "messages shape")
    code, r = jcall("POST", msgs, bearer, {"max_tokens": 5, "messages": ask_a, "tools": [an_tool]})
    check(r["stop_reason"] == "tool_use" and r["content"][1]["type"] == "tool_use", "messages tool")
    check(r["content"][1]["input"] == {"file_path": read_path} and r["content"][1]["id"].startswith("toolu_"), "tool input")
    result = {"type": "tool_result", "tool_use_id": "t", "content": "x"}
    after_a = ask_a + [{"role": "assistant", "content": r["content"]}, {"role": "user", "content": [result]},
                       {"role": "system", "content": [{"type": "text", "text": "reminder"}]}]  # a trailing system message
    code, r = jcall("POST", msgs, bearer, {"max_tokens": 5, "messages": after_a, "tools": [an_tool]})
    check(r["stop_reason"] == "end_turn" and r["content"][1]["text"] == ANSWER, "messages answer after a tool result")
    canary_a = [{"role": "user", "content": [{"type": "text", "text": "<system-reminder>r</system-reminder>"},
                                             {"type": "text", "text": CANARY_PROMPT}]}]
    code, r = jcall("POST", msgs, bearer, {"max_tokens": 5, "messages": canary_a, "tools": [an_tool]})
    check(r["content"][1]["text"] == CANARY_ANSWER, "messages canary")

    for tools, kind, last in (([], "text", "text_delta"), ([an_tool], "tool_use", "input_json_delta")):
        code, payload = call("POST", msgs, xkey, {"max_tokens": 5, "messages": ask_a, "tools": tools, "stream": True})
        evs = events(payload)
        names = [n for n, _ in evs]
        check(code == 200 and names[0] == "message_start" and names[-2:] == ["message_delta", "message_stop"], "sse frame")
        check(all(d["type"] == n for n, d in evs), "sse type fields")
        open_block = None  # each block closes before the next opens
        for n, d in evs:
            if n == "content_block_start":
                check(open_block is None, "a block opened before the last closed")
                open_block = d["index"]
            elif n == "content_block_delta":
                check(d["index"] == open_block, "a delta outside its block")
            elif n == "content_block_stop":
                check(d["index"] == open_block, "a stop for the wrong block")
                open_block = None
        check(open_block is None, "a block was never closed")
        starts = [d["content_block"]["type"] for n, d in evs if n == "content_block_start"]
        check(starts == ["thinking", kind], "stream block order")
        deltas = [d["delta"] for n, d in evs if n == "content_block_delta"]
        check([x["type"] for x in deltas][:2] == ["thinking_delta", "signature_delta"], "thinking deltas")
        check(deltas[1]["signature"] == "" and deltas[-1]["type"] == last, "stream delta kinds")
        if kind == "tool_use":
            joined = "".join(x["partial_json"] for x in deltas if x["type"] == "input_json_delta")
            check(json.loads(joined) == {"file_path": read_path}, "stream tool input")
            check(evs[-2][1]["delta"]["stop_reason"] == "tool_use", "stream stop_reason")
        else:
            check("".join(x["text"] for x in deltas if x["type"] == "text_delta") == ANSWER, "stream text")
            check(evs[-2][1]["delta"]["stop_reason"] == "end_turn", "stream stop_reason")

    # count_tokens (either credential), 404s, a bad body.
    body = {"model": "m", "messages": ask_a}
    code, r = jcall("POST", "/v1/messages/count_tokens", xkey, body)
    check(code == 200 and r["input_tokens"] == max(1, len(json.dumps(body)) // 4), "count_tokens")
    code, r = jcall("POST", "/v1/messages/count_tokens", None, body)
    check(code == 401 and r["type"] == "error", "count_tokens without a key")
    code, r = jcall("GET", "/nope")
    check(code == 404 and "error" in r, "404")
    code, r = jcall("GET", "/v1/nope", bearer)
    check(code == 404 and "error" in r, "404 under /v1")
    code, r = jcall("GET", "/v1/nope")
    check(code == 401, "401 before 404 under /v1")
    code, r = jcall("POST", chat, bearer, b"not json")
    check(code == 400 and "error" in r, "400 on a bad body")

    # Wrong credentials: classified `other`, 401, ABORT, and nothing of them stored anywhere.
    abort_path = os.path.join(out, "ABORT")
    check(not os.path.exists(abort_path), "ABORT before any other credential")
    code, r = jcall("POST", chat, {"Authorization": "Bearer " + bad_bearer}, {"model": "m", "messages": ask})
    check(code == 401 and "error" in r, "other credential (bearer) answered 401")
    check(os.path.exists(abort_path), "ABORT marker written")
    code, r = jcall("POST", msgs, {"x-api-key": bad_api}, {"max_tokens": 5, "messages": ask_a})
    check(code == 401 and r["type"] == "error", "other credential (x-api-key) answered 401")
    code, r = jcall("GET", "/health", {"Authorization": "Bearer " + bad_bearer})
    check(code == 401, "other credential on /health answered 401")
    # Another credential-looking header with a value that is not the dummy key is `other` too, even beside the dummy key.
    code, r = jcall("GET", "/health", {**bearer, "Cookie": bad_cookie})
    check(code == 401, "a cookie beside the dummy key answered 401")
    code, r = jcall("POST", msgs, {**xkey, "X-Session-Token": bad_token}, {"max_tokens": 5, "messages": ask_a})
    check(code == 401 and r["type"] == "error", "an other token header beside the dummy key answered 401")
    code, r = jcall("GET", "/health", {**bearer, "X-Session-Token": key})  # the dummy key in such a header is fine
    check(code == 200, "the dummy key in an extra credential header is not `other`")
    code, r = jcall("GET", "/v1/models", bearer)  # keeps serving after ABORT
    check(code == 200, "the stub keeps serving after an ABORT")

    server.shutdown()
    server.server_close()
    thread.join(timeout=10)

    with open(report, encoding="utf-8") as f:
        rep = json.load(f)
    check(rep["abort"] is True and rep["classes"]["other"] == 5, "report: abort and other count")
    check(rep["classes"]["dummy"] > 20 and rep["classes"]["none"] > 5, "report: dummy and none counts")
    check(rep["hosts"] == [f"{HOST}:{port}"] and rep["requests"] == sum(rep["classes"].values()), "report: hosts")

    files = sorted(n for n in os.listdir(out) if n != "ABORT")
    check(len(files) == rep["requests"], "one record per request")
    records = []
    for n in files:
        with open(os.path.join(out, n), encoding="utf-8") as f:
            records.append(json.load(f))
    check([r["seq"] for r in records] == list(range(1, len(records) + 1)), "records are numbered in order")
    check(any(n.endswith("-chat.json") for n in files) and any(n.endswith("-messages.json") for n in files), "names")
    check(any(n.endswith("-count_tokens.json") for n in files) and any(n.endswith("-models.json") for n in files), "names")
    for r in records:
        h = {k.lower(): v for k, v in r["headers"].items()}
        for name in ("authorization", "x-api-key"):
            check(name not in h or h[name] == MASKED, "an auth header was not masked")
        check(r["credential"] in ("dummy", "none", "other") and h["host"] == f"{HOST}:{port}", "record fields")
        if r["credential"] == "other":
            check(set(r) == {"seq", "method", "path", "credential", "headers"}, "an `other` record holds more than allowed")
    dummy = next(r for r in records if r["path"] == chat and r["credential"] == "dummy" and r["body"].get("stream"))
    check(dummy["method"] == "POST" and dummy["body"]["messages"] == ask and dummy["query"] == "", "chat record")
    check(any(r["query"] == "beta=true" for r in records), "the query is recorded")
    check(sum(r["credential"] == "other" for r in records) == 5, "other recorded")

    # Neither wrong credential, nor any part of one longer than 4 characters, appears under OUT, in the report or in
    # any reply. (The stub's own Host value, which holds the random port, is blanked first.)
    blobs = [open(os.path.join(out, n), "rb").read() for n in os.listdir(out)] + [open(report, "rb").read()]
    blobs += responses
    scrubbed = [b.decode("latin-1").replace(f"{HOST}:{port}", "HOST") for b in blobs]
    for secret in (bad_bearer, bad_api, bad_cookie, bad_token):
        parts = {secret} | {secret[i:i + 5] for i in range(len(secret) - 4)}
        for part in parts:
            for blob in scrubbed:
                check(part not in blob, "a wrong credential (or a part of one) was stored or echoed")
    for blob in scrubbed:
        check(key not in blob, "the dummy key was stored or echoed")
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int)
    ap.add_argument("--key-file")
    ap.add_argument("--out")
    ap.add_argument("--read-path")
    ap.add_argument("--report")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        try:
            selftest()
        except SelfTestError as e:
            print(f"selftest FAILED: {e}", file=sys.stderr)
            return 1
        return 0
    if args.port is None or not (args.key_file and args.out and args.read_path):
        ap.error("--port, --key-file, --out and --read-path are required")
    server = make_server(args.port, args.key_file, args.out, args.read_path, args.report)

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.state.write_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
