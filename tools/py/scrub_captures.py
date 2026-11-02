"""Turns the recording stub's raw captures into committed fixtures, and checks fixtures.
Standard library only.

    python -I tools/py/scrub_captures.py RAW_DIR OUT_DIR --client opencode|claude-code [--version V]
    python -I tools/py/scrub_captures.py --check DIR
    python -I tools/py/scrub_captures.py --selftest

Rules:
  * Both clients: absolute paths under the capture root become /scratch/...; the Windows user name, the machine name
    (COMPUTERNAME) and C:\\Users\\<name> anywhere in a string become `user`, `machine` and /home/user. Masked headers stay
    masked.
  * Both clients: the system text (messages with role system or developer, and Anthropic `system`) and every tool
    description have each non-whitespace character replaced by `x`. A system text that starts
    `x-anthropic-billing-header:` keeps that prefix and its field names, with each value replaced by `x`s of its length.
  * Claude Code: every other string value is replaced by `x`s of its length (whitespace kept), except the keep-list:
    the values of the keys role, type, name, model, stop_reason, media_type, effort, reasoning_effort and ttl; a
    string-valued tool_choice; and the entries of `required` and `enum` arrays. Header values are kept.
  * OpenCode (MIT): other strings are kept.

`--check` fails a fixture that breaks any of these, or that holds the user name, the machine name, C:\\Users,
the capture root's path, `sk-ant-`, a `Bearer` followed by anything but <masked>, a credential header that is not
exactly <masked>, or (Claude Code body strings) `Claude`, `Anthropic` or `You are`. The client comes from the directory
the fixture sits in (`claude-code/` or `opencode/`), and the `client` field must agree. For Claude Code every
keep-listed value, every dict key and every header value (bar the few named headers, each held to its own shape) must be
identifier-shaped, so a kept position cannot carry prose; the billing-header shape is accepted only at `system[0].text`.
Every `*.json` under the directory is a fixture; any other file besides LICENSE-OpenCode is a failure.
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
import tempfile

ROOT_PARTS = ("T:", "artemis-oracle", "runs", "m3b-capture")
KEEP_KEYS = ("role", "type", "name", "model", "stop_reason", "media_type", "effort", "reasoning_effort", "ttl")
SYSTEM_ROLES = ("system", "developer")
MASKED = "<masked>"
CREDENTIAL_HEADERS = ("authorization", "x-api-key", "cookie", "proxy-authorization")
BILLING = "x-anthropic-billing-header:"
BILLING_OK = re.compile(r"^x-anthropic-billing-header:\s*(?:(?:cc_version|cc_entrypoint|cch)=x*;?\s*)+$")
CLIENTS = ("claude-code", "opencode")
DOC_KEYS = {"client", "version", "seq", "method", "path", "query", "headers", "body"}
ID_RE = re.compile(r"^[A-Za-z0-9_.:/$-]{1,64}$")
QUERY_RE = re.compile(r"^[A-Za-z0-9_.=&-]{0,64}$")
VERSION_RE = re.compile(r"^[A-Za-z0-9_.-]{0,32}$")
METHOD_RE = re.compile(r"^[A-Z]{3,7}$")
# Claude Code headers whose values are not one identifier: each has its own narrow shape. Every other header value
# (x-stainless-*, session ids, content-type, host, ...) must be identifier-shaped.
HEADER_SHAPES = {
    "user-agent": re.compile(r"^[A-Za-z0-9_.-]+/[0-9][0-9.]*(?: \([a-z0-9_.-]+(?:, [a-z0-9_.-]+)*\))?$"),
    "anthropic-beta": re.compile(r"^[A-Za-z0-9_.:-]+(?:,[A-Za-z0-9_.:-]+)*$"),
    "accept-encoding": re.compile(r"^(?:gzip|deflate|br|zstd|identity|compress)(?:, ?(?:gzip|deflate|br|zstd|identity|compress))*$"),
}
OPENAI_PATH = re.compile(r"^/v1/(?:chat/completions|models(?:/[A-Za-z0-9_.:/-]*)?)$")
SECRETISH = ("key", "token", "secret", "auth")
PLACEHOLDER = re.compile(r"^[x\s]*$")
SEP = r"(?:\\+|/)"
ROOT_RE = re.compile(
    re.escape(ROOT_PARTS[0]) + SEP + SEP.join(re.escape(p) for p in ROOT_PARTS[1:]) + r"((?:" + SEP + r"[^\\/\s\"'<>|*?:]+)*)"
)
USERS_RE = re.compile(r"[A-Za-z]:" + SEP + r"Users" + SEP + r"[^\\/\s\"'<>|*?:]+")
ARTEMIS_RE = re.compile(r"T:" + SEP + r"artemis-oracle", re.I)
BEARER_RE = re.compile(r"(?i)bearer\s+(?!<masked>)\S")


def names(user, machine):
    """The user and machine names to look for: the arguments, else USERNAME and COMPUTERNAME. Both are required."""
    user = user or os.environ.get("USERNAME", "")
    machine = machine or os.environ.get("COMPUTERNAME", "")
    if not user or not machine:
        raise SystemExit("the user and machine names are required: pass --user and --machine, or set USERNAME and COMPUTERNAME")
    return user, machine


def xify(s):
    return re.sub(r"\S", "x", s)


def xify_billing(s):
    """Keeps the prefix, the field names and the separators of the billing header; each value becomes `x`s."""
    head, _, rest = s.partition(":")
    return head + ":" + re.sub(r"(=)([^;\s]*)", lambda m: m.group(1) + "x" * len(m.group(2)), rest)


def scrub_paths_names(s, user, machine):
    s = ROOT_RE.sub(lambda m: "/scratch" + re.sub(r"\\+|/", "/", m.group(1)), s)
    s = USERS_RE.sub("/home/user", s)
    for name, repl in ((user, "user"), (machine, "machine")):
        if name:
            s = re.sub(re.escape(name), repl, s, flags=re.I)
    return s


# ---- walking a body: every string gets a kind ----


def system_message(body, path):
    """True when `path` (a tuple of keys and indices) is inside a message whose role is system or developer."""
    if len(path) >= 2 and path[0] == "messages" and isinstance(path[1], int):
        msgs = body.get("messages")
        if isinstance(msgs, list) and path[1] < len(msgs) and isinstance(msgs[path[1]], dict):
            return msgs[path[1]].get("role") in SYSTEM_ROLES
    return False


def kind_of(body, path, parent_key):
    """`system`, `tooldesc`, `keep` or `other` for the string at `path`."""
    last = path[-1]
    key = last if isinstance(last, str) else parent_key
    # the system text: Anthropic `system` (a string, or blocks' `text`) and the content of system-role messages
    if path[0] == "system" and (len(path) == 1 or (len(path) == 3 and key == "text")):
        return "system"
    if system_message(body, path):
        if (len(path) == 3 and path[2] == "content") or (len(path) == 5 and path[2] == "content" and key == "text"):
            return "system"
    if path[0] == "tools" and key == "description":
        return "tooldesc"
    if isinstance(last, str) and last in KEEP_KEYS:
        return "keep"
    if path[0] == "tool_choice" and len(path) == 1:
        return "keep"
    if isinstance(last, int) and parent_key in ("required", "enum"):
        return "keep"
    return "other"


def walk(body, fn):
    """A copy of `body` with every string s replaced by fn(kind, s, path)."""

    def rec(node, path, parent_key):
        if isinstance(node, dict):
            return {k: rec(v, path + (k,), k) for k, v in node.items()}
        if isinstance(node, list):
            return [rec(v, path + (i,), parent_key if path else None) for i, v in enumerate(node)]
        if isinstance(node, str):
            return fn(kind_of(body, path, parent_key), node, path)
        return node

    if not isinstance(body, dict):
        return fn("other", body, ()) if isinstance(body, str) else body
    return {k: rec(v, (k,), k) for k, v in body.items()}


def billing_position(path):
    """Claude Code's billing header is the first system block (or the whole system string)."""
    return path == ("system",) or path == ("system", 0, "text")


def scrub_body(body, client, user, machine):
    def fn(kind, s, path):
        s = scrub_paths_names(s, user, machine)
        if kind in ("system", "tooldesc"):
            if kind == "system" and client == "claude-code" and billing_position(path) and s.startswith(BILLING):
                return xify_billing(s)
            return xify(s)
        if kind == "other" and client == "claude-code":
            return xify(s)
        return s

    return walk(body, fn)


def scrub_record(rec, client, version, user, machine):
    headers = {}
    for name, value in rec.get("headers", {}).items():
        headers[name] = value if value == MASKED else scrub_paths_names(value, user, machine)
    return {
        "client": client,
        "version": version,
        "seq": rec["seq"],
        "method": rec["method"],
        "path": scrub_paths_names(rec["path"], user, machine),
        "query": scrub_paths_names(rec.get("query", ""), user, machine),
        "headers": headers,
        "body": scrub_body(rec.get("body"), client, user, machine),
    }


# ---- the check ----


def all_strings(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from all_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from all_strings(v)
    elif isinstance(node, str):
        yield node


def all_keys(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from all_keys(v)
    elif isinstance(node, list):
        for v in node:
            yield from all_keys(v)


def check_headers(headers, client, problems):
    if not isinstance(headers, dict):
        problems.append("headers is not an object")
        return
    for name, value in headers.items():
        low = name.lower()
        if not isinstance(value, str):
            problems.append("a header value that is not a string")
            continue
        if low in CREDENTIAL_HEADERS and value != MASKED:
            problems.append(f"header {name} is not exactly <masked>")
        elif any(w in low for w in SECRETISH) and value != MASKED:
            problems.append("a key-, token-, secret- or auth-named header that is not exactly <masked>")
        if client != "claude-code" or value == MASKED:
            continue
        if not ID_RE.match(name):
            problems.append("a header name that is not identifier-shaped")
        shape = HEADER_SHAPES.get(low, ID_RE)
        if not shape.match(value):
            problems.append("a Claude Code header value that is not identifier-shaped")


def header_value(headers, name):
    """The value of header `name` (any case), or None."""
    if isinstance(headers, dict):
        for k, v in headers.items():
            if isinstance(k, str) and k.lower() == name and isinstance(v, str):
                return v
    return None


def content_matches_client(doc, client):
    """The problems in a fixture whose content does not belong to the client its directory names."""
    problems = []
    headers = doc.get("headers", {})
    names = [k.lower() for k in headers if isinstance(k, str)] if isinstance(headers, dict) else []
    ua = (header_value(headers, "user-agent") or "").lower()
    path = doc.get("path") if isinstance(doc.get("path"), str) else ""
    if client == "claude-code":
        if not ua.startswith("claude-cli/"):
            problems.append("a Claude Code fixture without a claude-cli/ User-Agent")
        return problems
    # anything judged as OpenCode (or as no client) must not look like Claude Code's
    if path.startswith("/v1/messages"):
        problems.append("an Anthropic path in a fixture not judged as claude-code")
    if any(n.startswith("anthropic-") or n in ("x-app", BILLING[:-1]) for n in names):
        problems.append("an Anthropic header in a fixture not judged as claude-code")
    if ua.startswith("claude-cli"):
        problems.append("a claude-cli User-Agent in a fixture not judged as claude-code")
    for s in all_strings(doc):
        low = s.lower()
        if BILLING[:-1] in low:
            problems.append("a billing header in a fixture not judged as claude-code")
        if "<system-reminder>" in low:
            problems.append("<system-reminder> text in a fixture not judged as claude-code")
    if client == "opencode":
        if not ua.startswith("opencode/"):
            problems.append("an OpenCode fixture without an opencode/ User-Agent")
        if not OPENAI_PATH.match(path):
            problems.append("an OpenCode fixture whose path is not an OpenAI route")
    return problems


def check_doc(doc, user, machine, client):
    """The problems found in one fixture sitting in `client`'s directory (an empty list is a pass)."""
    problems = []
    if not isinstance(doc, dict):
        return ["the fixture is not a JSON object"]
    body = doc.get("body")
    if client not in CLIENTS:
        problems.append("the fixture is not under a claude-code/ or opencode/ directory")
    if doc.get("client") != client:
        problems.append("the client field disagrees with the directory")
    if set(doc) != DOC_KEYS:
        problems.append("the fixture's fields are not exactly client, version, seq, method, path, query, headers, body")
    check_headers(doc.get("headers", {}), client, problems)
    problems += content_matches_client(doc, client)
    if client == "claude-code":
        for field, shape in (("version", VERSION_RE), ("method", METHOD_RE), ("path", ID_RE), ("query", QUERY_RE)):
            if not (isinstance(doc.get(field), str) and shape.match(doc[field])):
                problems.append(f"the Claude Code {field} is not identifier-shaped")
        for k in all_keys(body):
            if not ID_RE.match(k):
                problems.append("a Claude Code body key that is not identifier-shaped")
            for word in ("Claude", "Anthropic", "You are"):
                if word in k:
                    problems.append(f"{word!r} in a Claude Code body key")
    for s in all_strings(doc):
        low = s.lower()
        if user and user.lower() in low:
            problems.append("the user name")
        if machine and machine.lower() in low:
            problems.append("the machine name")
        if USERS_RE.search(s) or re.search(r"[A-Za-z]:" + SEP + r"Users", s):
            problems.append("C:\\Users")
        if ARTEMIS_RE.search(s):
            problems.append("T:\\artemis-oracle")
        if "sk-ant-" in low:
            problems.append("sk-ant-")
        if BEARER_RE.search(s):
            problems.append("a Bearer value other than <masked>")

    def judge(kind, s, path):
        if kind in ("system", "tooldesc"):
            billing = kind == "system" and client == "claude-code" and billing_position(path) and BILLING_OK.match(s)
            if not (PLACEHOLDER.match(s) or billing):
                problems.append(f"{kind} text that is not placeholder text")
        elif client == "claude-code":
            if kind == "keep" and not ID_RE.match(s):
                problems.append("a Claude Code keep-listed value that is not identifier-shaped")
            elif kind == "other" and not PLACEHOLDER.match(s):
                problems.append("a Claude Code body string outside the keep-list that is not placeholder text")
        if client == "claude-code":
            for word in ("Claude", "Anthropic", "You are"):
                if word in s:
                    problems.append(f"{word!r} in a Claude Code body string")
        return s

    walk(body, judge)
    return sorted(set(problems))


def client_of_dir(root, dirpath):
    """The client a directory belongs to: the nearest of it and its parents (up to `root` itself) named for one."""
    parts = [os.path.basename(os.path.abspath(root))]
    rel = os.path.relpath(dirpath, root)
    if rel != ".":
        parts += rel.split(os.sep)
    for part in reversed(parts):
        if part in CLIENTS:
            return part
    return None


def check_dir(path, user, machine, log=None, out=None):
    """Checks every file under `path`: each *.json is a fixture, judged as the client its directory names."""
    log = log or sys.stderr
    out = out or sys.stdout
    bad = 0
    count = 0
    for dirpath, dirs, files in os.walk(path):
        dirs.sort()
        client = client_of_dir(path, dirpath)
        for name in sorted(files):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, path)
            if not name.endswith(".json"):
                if not (name == "LICENSE-OpenCode" and client == "opencode"):
                    bad += 1
                    print(f"{rel}: a file that is neither a fixture nor LICENSE-OpenCode", file=log)
                continue
            count += 1
            try:
                with open(full, encoding="utf-8") as f:
                    doc = json.load(f)
            except (OSError, ValueError):
                bad += 1
                print(f"{rel}: not readable as JSON", file=log)
                continue
            for p in check_doc(doc, user, machine, client):
                bad += 1
                print(f"{rel}: {p}", file=log)
    if count == 0:
        print("check FAILED: no fixtures found", file=log)
        return 1
    if bad:
        print(f"check FAILED: {bad} problem(s) in {count} fixture(s)", file=log)
        return 1
    print(f"check ok ({count} fixtures)", file=out)
    return 0


# ---- scrubbing a directory ----


def client_mismatch(rec, client):
    """Why the raw record cannot belong to `client`, or ''. A wrong --client would scrub by the wrong rules."""
    ua = (header_value(rec.get("headers", {}), "user-agent") or "").lower()
    path = rec.get("path") if isinstance(rec.get("path"), str) else ""
    if client == "opencode" and (ua.startswith("claude-cli") or path.startswith("/v1/messages")):
        return "a Claude Code record (claude-cli User-Agent or an Anthropic path) given --client opencode"
    if client == "claude-code" and ua.startswith("opencode/"):
        return "an OpenCode record (opencode/ User-Agent) given --client claude-code"
    return ""


def scrub_dir(raw, out, client, version, user, machine):
    os.makedirs(out, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(raw)):
        if not re.match(r"\d{3,}-.*\.json$", name):
            continue
        with open(os.path.join(raw, name), encoding="utf-8") as f:
            rec = json.load(f)
        if rec.get("credential") == "other":
            raise SystemExit(f"{name}: a request with a credential other than the dummy key; not scrubbing")
        mismatch = client_mismatch(rec, client)
        if mismatch:
            raise SystemExit(f"{name}: {mismatch}; not scrubbing")
        doc = scrub_record(rec, client, version, user, machine)
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
            f.write("\n")
        n += 1
    print(f"scrubbed {n} request(s) from {raw} into {out}")
    return n


# ---- the self-test ----


def expect(cond, msg):
    """An explicit check (an `assert` vanishes under python -O)."""
    if not cond:
        raise SystemExit(f"selftest FAILED: {msg}")


def selftest():
    user, machine = "alice", "box-17"
    bs = chr(92)
    root = ROOT_PARTS[0] + bs + bs.join(ROOT_PARTS[1:])
    raw_claude = {
        "seq": 1, "method": "POST", "path": "/v1/messages", "query": "beta=true",
        "headers": {"Authorization": MASKED, "x-api-key": MASKED, "Host": "127.0.0.1:8090", "User-Agent": "claude-cli/9 (external, sdk-cli)",
                    "anthropic-beta": "a-2025-01-01,b-2025-02-02", "Accept-Encoding": "gzip, deflate, br, zstd"},
        "credential": "dummy",
        "body": {
            "model": "m", "max_tokens": 5, "stream": True,
            "system": [{"type": "text", "text": BILLING + " cc_version=1.2.3; cc_entrypoint=cli;"},
                       {"type": "text", "text": "You are a Claude agent built by Anthropic.", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "<system-reminder>As you answer, remember the rules.</system-reminder>"},
                                                                {"type": "text", "text": f"read {root}{bs}project{bs}notes.txt for alice on box-17"}]},
                         {"role": "system", "content": "reminder for alice"},
                         {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_abc", "name": "Read",
                                                             "input": {"file_path": f"C:{bs}Users{bs}alice{bs}x.txt"}}]}],
            "tools": [{"name": "Read", "description": "Reads a file for Claude.", "input_schema": {
                "type": "object", "properties": {"file_path": {"type": "string", "description": "the path"}},
                "required": ["file_path"]}}],
            "tool_choice": "auto",
        },
    }
    raw_opencode = {
        "seq": 1, "method": "POST", "path": "/v1/chat/completions", "query": "",
        "headers": {"Authorization": MASKED, "Host": "127.0.0.1:8090", "User-Agent": "opencode/latest/9/cli"}, "credential": "dummy",
        "body": {
            "model": "m", "stream": True,
            "messages": [{"role": "system", "content": "You are OpenCode."}, {"role": "user", "content": f"Read {root}{bs}project{bs}notes.txt"}],
            "tools": [{"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {
                "type": "object", "properties": {"path": {"type": "string", "description": "File to read"}}, "required": ["path"]}}}],
        },
    }
    good_c = scrub_record(raw_claude, "claude-code", "9", user, machine)
    good_o = scrub_record(raw_opencode, "opencode", "9", user, machine)
    for what, doc in (("claude-code", good_c), ("opencode", good_o)):
        problems = check_doc(doc, user, machine, what)
        expect(not problems, f"a scrubbed {what} fixture does not pass --check: {problems}")
    text = json.dumps(good_c)
    sc = json.dumps(good_o)
    expect("/scratch/project/notes.txt" in sc and root not in sc, "the root was not scrubbed to /scratch")
    expect("alice" not in text and "box-17" not in text and "Users" not in text, "names were not scrubbed")
    expect(good_c["body"]["system"][0]["text"] == BILLING + " cc_version=xxxxx; cc_entrypoint=xxx;", f"billing header: {good_c['body']['system'][0]}")
    expect(good_c["body"]["tools"][0]["name"] == "Read" and good_c["body"]["tool_choice"] == "auto", "keep-list values changed")
    expect(good_c["body"]["tools"][0]["input_schema"]["required"] == ["file_path"], "required entries changed")
    expect(good_c["body"]["messages"][2]["content"][0]["id"] == "x" * len("toolu_abc"), "tool_use id not placeholdered")
    expect(good_c["body"]["messages"][1]["content"] == "xxxxxxxx xxx xxxx", f"system-role message: {good_c['body']['messages'][1]}")
    expect(good_o["body"]["tools"][0]["function"]["description"] == "xxxx x xxxx", "OpenCode tool description")
    expect(good_o["body"]["messages"][0]["content"] == "xxx xxx xxxxxxxxx", f"OpenCode system message: {good_o['body']['messages'][0]}")
    expect(good_o["body"]["tools"][0]["function"]["parameters"]["properties"]["path"]["description"] == "xxxx xx xxxx", "nested description")
    expect(good_o["body"]["messages"][1]["content"].startswith("Read /scratch/"), "OpenCode user text must be kept")
    # the billing shape is kept only at system[0]: anywhere else (or for OpenCode) it is x'd out whole
    other = json.loads(json.dumps(raw_claude))
    other["body"]["system"][1]["text"] = BILLING + " cc_version=1.2.3;"
    other["body"]["tools"][0]["description"] = BILLING + " cc_version=1.2.3;"
    other = scrub_record(other, "claude-code", "9", user, machine)
    expect(set(other["body"]["system"][1]["text"]) <= {"x", " "}, "billing shape kept at system[1]")
    expect(set(other["body"]["tools"][0]["description"]) <= {"x", " "}, "billing shape kept in a tool description")
    as_string = json.loads(json.dumps(raw_claude))
    as_string["body"]["system"] = BILLING + " cc_version=1.2.3;"
    as_string = scrub_record(as_string, "claude-code", "9", user, machine)
    expect(as_string["body"]["system"] == BILLING + " cc_version=xxxxx;", f"billing header as a string-form system: {as_string['body']['system']}")
    expect(not check_doc(as_string, user, machine, "claude-code"), "a string-form billing system does not pass --check")

    def mutate(base, fn):
        doc = json.loads(json.dumps(base))
        fn(doc)
        return doc

    def setitem(path, value):
        def fn(d):
            node = d
            for p in path[:-1]:
                node = node[p]
            node[path[-1]] = value
        return fn

    def chain(*fns):
        def fn(d):
            for f in fns:
                f(d)
        return fn

    prose = "some prose here"
    c_msg0 = ("body", "messages", 0, "content", 0)
    cases = [
        # (what, base, client the fixture is checked as, mutation)
        ("Claude Code body string left in clear", good_c, "claude-code", setitem(c_msg0 + ("text",), "hello there")),
        ("Claude Code string, id", good_c, "claude-code", setitem(("body", "messages", 2, "content", 0, "id"), "toolu_abc")),
        ("system text in clear (Claude Code)", good_c, "claude-code", setitem(("body", "system", 1, "text"), "xx y")),
        ("system text in clear (OpenCode)", good_o, "opencode", setitem(("body", "messages", 0, "content"), "be helpful")),
        ("tool description in clear (OpenCode)", good_o, "opencode", setitem(("body", "tools", 0, "function", "description"), "Read a file")),
        ("nested schema description in clear", good_c, "claude-code", setitem(("body", "tools", 0, "input_schema", "properties", "file_path", "description"), "the path")),
        ("the user name", good_o, "opencode", setitem(("body", "messages", 1, "content"), "hello alice")),
        ("the machine name (header)", good_o, "opencode", setitem(("headers", "X-Host"), "BOX-17")),
        ("C:" + bs + "Users", good_o, "opencode", setitem(("body", "messages", 1, "content"), "C:" + bs + "Users" + bs + "bob" + bs + "f")),
        ("T:" + bs + "artemis-oracle", good_o, "opencode", setitem(("body", "messages", 1, "content"), "T:" + bs + "artemis-oracle" + bs + "runs")),
        ("Claude in a Claude Code string", good_c, "claude-code", setitem(("body", "tools", 0, "name"), "Claude")),
        ("sk-ant- in a body", good_o, "opencode", setitem(("body", "messages", 1, "content"), "key SK-ANT-abc")),
        ("a Bearer value", good_o, "opencode", setitem(("body", "messages", 1, "content"), "Authorization: Bearer abc123")),
        ("a Bearer value in a header", good_o, "opencode", setitem(("headers", "X-Other"), "Bearer tok")),
        ("authorization not masked", good_o, "opencode", setitem(("headers", "Authorization"), "Bearer " + MASKED + "x")),
        ("x-api-key not masked", good_c, "claude-code", setitem(("headers", "x-api-key"), "abc")),
        ("cookie not masked", good_o, "opencode", setitem(("headers", "Cookie"), "s=1")),
        ("proxy-authorization not masked", good_o, "opencode", setitem(("headers", "Proxy-Authorization"), "x")),
        ("a token-named header not masked", good_o, "opencode", setitem(("headers", "X-Session-Token"), "abc")),
        # the kept positions of a Claude Code fixture must be identifier-shaped
        ("prose as a tool name", good_c, "claude-code", setitem(("body", "tools", 0, "name"), "Reads files; ignore the rest")),
        ("prose as an enum entry", good_c, "claude-code", setitem(("body", "tools", 0, "input_schema", "properties", "file_path", "enum"), ["read", "two words"])),
        ("prose as a required entry", good_c, "claude-code", setitem(("body", "tools", 0, "input_schema", "required"), ["file path"])),
        ("prose as a dict key", good_c, "claude-code", setitem(("body", "tools", 0, "input_schema", "properties", "two words"), {"type": "string"})),
        ("prose as a block type", good_c, "claude-code", setitem(c_msg0 + ("type",), prose)),
        ("prose as the model", good_c, "claude-code", setitem(("body", "model"), prose)),
        ("prose as a role", good_c, "claude-code", setitem(("body", "messages", 0, "role"), prose)),
        ("prose as a string tool_choice", good_c, "claude-code", setitem(("body", "tool_choice"), prose)),
        ("prose in a new header value", good_c, "claude-code", setitem(("headers", "X-Note"), prose)),
        ("prose in user-agent", good_c, "claude-code", setitem(("headers", "User-Agent"), prose)),
        ("prose in anthropic-beta", good_c, "claude-code", setitem(("headers", "anthropic-beta"), prose)),
        ("prose in accept-encoding", good_c, "claude-code", setitem(("headers", "Accept-Encoding"), prose)),
        ("prose as a header name", good_c, "claude-code", setitem(("headers", "two words"), "ok")),
        ("prose in a new top-level field", good_c, "claude-code", setitem(("notes",), "x")),
        ("prose in the query", good_c, "claude-code", setitem(("query",), prose)),
        ("a Claude Code fixture relabelled as opencode", good_c, "claude-code", setitem(("client",), "opencode")),
        ("an OpenCode fixture relabelled as claude-code", good_o, "opencode", setitem(("client",), "claude-code")),
        ("a fixture outside a client directory", good_o, None, lambda d: None),
        # the billing-header shape is accepted only at Claude Code's system[0].text
        ("a billing-shaped tool description (Claude Code)", good_c, "claude-code", setitem(("body", "tools", 0, "description"), BILLING + " cc_version=xxx;")),
        ("a billing-shaped tool description (OpenCode)", good_o, "opencode", setitem(("body", "tools", 0, "function", "description"), BILLING + " cc_version=xxx;")),
        ("a billing-shaped system block after the first", good_c, "claude-code", setitem(("body", "system", 1, "text"), BILLING + " cc_version=xxx;")),
        ("a billing-shaped system-role message", good_c, "claude-code", setitem(("body", "messages", 1, "content"), BILLING + " cc_version=xxx;")),
        ("a billing-shaped OpenCode system message", good_o, "opencode", setitem(("body", "messages", 0, "content"), BILLING + " cc_version=xxx;")),
        # a fixture's content must belong to the client its directory names
        ("an Anthropic path under opencode", good_o, "opencode", setitem(("path",), "/v1/messages")),
        ("an anthropic-* header under opencode", good_o, "opencode", setitem(("headers", "anthropic-version"), "2023-06-01")),
        ("an x-app header under opencode", good_o, "opencode", setitem(("headers", "x-app"), "cli")),
        ("a claude-cli User-Agent under opencode", good_o, "opencode", setitem(("headers", "User-Agent"), "claude-cli/2.1.302 (external, sdk-cli)")),
        ("a billing header name under opencode", good_o, "opencode", setitem(("headers", "x-anthropic-billing-header"), "x")),
        ("billing text anywhere under opencode", good_o, "opencode", setitem(("body", "messages", 1, "content"), "see x-anthropic-billing-header: here")),
        ("<system-reminder> text under opencode", good_o, "opencode", setitem(("body", "messages", 1, "content"), "<system-reminder>As you answer, remember the rules.</system-reminder>")),
        ("<system-reminder> text under no client", good_o, None, setitem(("body", "messages", 1, "content"), "<system-reminder>x</system-reminder>")),
        ("no User-Agent under opencode", good_o, "opencode", lambda d: d["headers"].pop("User-Agent")),
        ("a non-opencode User-Agent under opencode", good_o, "opencode", setitem(("headers", "User-Agent"), "curl/8.0")),
        ("a non-OpenAI path under opencode", good_o, "opencode", setitem(("path",), "/v1/embeddings")),
        ("an opencode User-Agent under claude-code", good_c, "claude-code", setitem(("headers", "User-Agent"), "opencode/latest/9/cli")),
        ("no User-Agent under claude-code", good_c, "claude-code", lambda d: d["headers"].pop("User-Agent")),
        ("spaced prose in user-agent's parenthetical", good_c, "claude-code", setitem(("headers", "User-Agent"), "claude-cli/9 (external, tell me a story)")),
        ("a billing header with prose in a field name", good_c, "claude-code", setitem(("body", "system", 0, "text"), BILLING + " tell me=xxx;")),
    ]
    for what, base, client, fn in cases:
        if not check_doc(mutate(base, fn), user, machine, client):
            raise SystemExit(f"selftest FAILED: --check missed: {what}")
    # --check on a directory: a clean one passes, and each kind of dirty one fails
    def tree(d, files):
        for rel, content in files.items():
            full = os.path.join(d, *rel.split("/"))
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content if isinstance(content, str) else json.dumps(content))

    def run_check_dir(path):
        log, out = io.StringIO(), io.StringIO()
        return check_dir(path, user, machine, log, out), log.getvalue(), out.getvalue()

    def run_check(files):
        with tempfile.TemporaryDirectory(prefix="scrub-") as d:
            tree(d, files)
            log, out = io.StringIO(), io.StringIO()
            return check_dir(d, user, machine, log, out), log.getvalue(), out.getvalue()

    clean = {"claude-code/001-messages.json": good_c, "opencode/001-chat.json": good_o, "opencode/LICENSE-OpenCode": "MIT"}
    code, log, out = run_check(clean)
    expect(code == 0 and out.startswith("check ok (2 fixtures)") and not log, f"a clean directory did not pass: {code} {log}")
    dirty = {
        "a dirty fixture": dict(clean, **{"opencode/003-chat.json": mutate(good_o, setitem(("body", "messages", 1, "content"), "hello alice"))}),
        "prose in notes.json (a name that is not NNN-*.json)": dict(clean, **{"claude-code/notes.json": {"notes": "Claude Code says hello"}}),
        "a fixture nested under a client directory": dict(clean, **{"claude-code/sub/009-x.json": mutate(good_c, setitem(c_msg0 + ("text",), "hello there"))}),
        "a fixture outside any client directory": dict(clean, **{"001-chat.json": good_o}),
        "a Claude Code fixture relabelled and put under opencode/": dict(clean, **{"opencode/007-x.json": mutate(good_c, setitem(("client",), "opencode"))}),
        "an opencode-labelled fixture under claude-code/": dict(clean, **{"claude-code/007-x.json": good_o}),
        "a stray text file": dict(clean, **{"claude-code/README.txt": "Claude Code system prompt"}),
        "a JSON file that does not parse": dict(clean, **{"claude-code/009-x.json": "{not json"}),
        "a LICENSE-OpenCode under claude-code/": dict(clean, **{"claude-code/LICENSE-OpenCode": "MIT"}),
    }
    for what, files in dirty.items():
        code, log, _ = run_check(files)
        expect(code != 0 and "check FAILED" in log, f"--check passed a directory with {what}")
    # the mis-scrub: Claude Code's raw capture scrubbed with --client opencode is refused by the scrubber, and a directory
    # built that way (the scrubber's refusal bypassed) is refused by --check
    with tempfile.TemporaryDirectory(prefix="scrub-") as d:
        raw_c, raw_o = os.path.join(d, "raw", "claude-code"), os.path.join(d, "raw", "opencode")
        for dirpath, rec in ((raw_c, raw_claude), (raw_o, raw_opencode)):
            os.makedirs(dirpath)
            with open(os.path.join(dirpath, "001-x.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f)
        log = io.StringIO()
        for raw, client in ((raw_c, "opencode"), (raw_o, "claude-code")):
            try:
                with contextlib.redirect_stdout(log):
                    scrub_dir(raw, os.path.join(d, "refused", client), client, "9", user, machine)
            except SystemExit:
                continue
            expect(False, f"scrub_dir took {os.path.basename(raw)} records with --client {client}")
        expect(not os.path.exists(os.path.join(d, "refused", "opencode", "001-x.json")), "a refused scrub wrote a fixture")
        with contextlib.redirect_stdout(log):
            scrub_dir(raw_c, os.path.join(d, "ok", "claude-code"), "claude-code", "9", user, machine)
            scrub_dir(raw_o, os.path.join(d, "ok", "opencode"), "opencode", "9", user, machine)
        code, _, out = run_check_dir(os.path.join(d, "ok"))
        expect(code == 0, "the correctly scrubbed directory did not pass --check")
        mis = os.path.join(d, "mis", "opencode")
        os.makedirs(mis)
        with open(os.path.join(raw_c, "001-x.json"), encoding="utf-8") as f:
            wrong = scrub_record(json.load(f), "opencode", "9", user, machine)
        with open(os.path.join(mis, "001-x.json"), "w", encoding="utf-8") as f:
            json.dump(wrong, f)
        expect("<system-reminder>" in json.dumps(wrong), "the mis-scrub case does not leave the reminder text in clear")
        code, mlog, _ = run_check_dir(os.path.join(d, "mis"))
        expect(code != 0 and "claude-cli" in mlog and "<system-reminder>" in mlog, f"--check passed a Claude Code capture scrubbed as opencode: {mlog}")
    code, log, _ = run_check({"opencode/LICENSE-OpenCode": "MIT"})
    expect(code != 0, "--check passed a directory with no fixtures")
    # the names are required, not optional
    saved = {k: os.environ.pop(k, None) for k in ("USERNAME", "COMPUTERNAME")}
    try:
        for args in ((None, None), ("alice", None), (None, "box-17")):
            try:
                names(*args)
            except SystemExit:
                continue
            expect(False, f"names{args} did not fail with a name missing")
        expect(names("alice", "box-17") == ("alice", "box-17"), "explicit names not taken")
        os.environ["USERNAME"], os.environ["COMPUTERNAME"] = "envuser", "envbox"
        expect(names(None, None) == ("envuser", "envbox"), "the environment is not the default")
        expect(names("alice", None) == ("alice", "envbox"), "an explicit name does not override the environment")
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description="Scrub raw captures into fixtures, or check fixtures.")
    ap.add_argument("raw", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--client", choices=CLIENTS)
    ap.add_argument("--version", default="")
    ap.add_argument("--check")
    ap.add_argument("--user", help="the Windows user name to scrub for (default: USERNAME)")
    ap.add_argument("--machine", help="the machine name to scrub for (default: COMPUTERNAME)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return 0
    user, machine = names(args.user, args.machine)
    if args.check:
        return check_dir(args.check, user, machine)
    if not (args.raw and args.out and args.client):
        ap.error("RAW_DIR, OUT_DIR and --client are required")
    scrub_dir(args.raw, args.out, args.client, args.version, user, machine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
