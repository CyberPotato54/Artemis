"""Build testdata/template's oracle fixtures: the chat template rendered as llama-server
b11509 renders it, for the request bodies written for Artemis, and llama.cpp's token ids for those prompts.

Usage: python -I tools/py/build_template_fixtures.py <repo root> <model.gguf> <oracle bin dir>
           [--server URL]

- testdata/template/requests/<id>.json, each an OpenAI chat body, is POSTed whole to /apply-template;
  prompts/<id>.txt is the returned prompt (UTF-8 bytes).
- testdata/template/anthropic/<id>.json, each an Anthropic body, is POSTed to /v1/messages/count_tokens, which converts
  it with llama.cpp's own Anthropic path and renders it with the template; anthropic/<id>.count is its `input_tokens`.
  A non-200 is reported and that fixture skipped, not fatal.
- Then `artemis-oracle tokenize` writes prompts/<id>.tokens for every prompt rendered.
Rendering starts llama-server on the CPU (port 8097, build_golden_set.start_server) with `--reasoning-preserve` spelled
out, and always stops it afterwards, unless --server names one already running; that one must run with
`--reasoning-preserve` (llama-server's default) too, or t18 (no `preserve_thinking` of its own) and the other bodies
with earlier turns' reasoning render differently. Neither this script nor the server it starts uses port 8080 or 8090.
Each body is POSTed as the bytes of its file.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_golden_set as golden  # noqa: E402  (start_server and the CPU server's PORT)

FORBIDDEN_PORTS = (8080, 8090)  # 8080 is the production server's; 8090 is the capture stub's


def check_port(url: str) -> None:
    port = urllib.parse.urlparse(url).port
    if port in FORBIDDEN_PORTS:
        sys.exit(f"{url}: port {port} is not for the oracle fixtures")


def post(url: str, body: bytes) -> tuple[int, bytes]:
    """The status and body of a JSON POST; an HTTP error status is returned, not raised."""
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def error_text(body: bytes) -> str:
    """A short, one-line reading of an error body (the bodies here carry no secrets)."""
    return " ".join(body.decode("utf-8", "replace").split())[:300]


def render_all(server: str, template: pathlib.Path) -> tuple[list[str], list[str]]:
    """Writes prompts/<id>.txt and anthropic/<id>.count. Returns the ids of the prompts written and of the
    requests the server refused (a count_tokens failure is only reported)."""
    prompts = template / "prompts"
    prompts.mkdir(exist_ok=True)
    rendered, failed = [], []
    for path in sorted((template / "requests").glob("*.json")):
        id = path.stem
        body = path.read_bytes()
        status, data = post(f"{server}/apply-template", body)
        if status != 200:
            print(f"{id}: /apply-template answered {status}: {error_text(data)}")
            failed.append(id)
            continue
        text = json.loads(data)["prompt"].encode("utf-8")
        (prompts / f"{id}.txt").write_bytes(text)
        rendered.append(id)
        print(f"{id}: {len(text):,} bytes")
    for path in sorted((template / "anthropic").glob("*.json")):
        id = path.stem
        body = path.read_bytes()
        status, data = post(f"{server}/v1/messages/count_tokens", body)
        if status != 200:
            print(f"{id}: count_tokens answered {status}: {error_text(data)}; skipped")
            continue
        n = json.loads(data)["input_tokens"]
        (template / "anthropic" / f"{id}.count").write_bytes(f"{n}\n".encode("ascii"))
        print(f"{id}: {n} input tokens")
    return rendered, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("model")
    ap.add_argument("bin_dir")
    ap.add_argument("--server")
    a = ap.parse_args()
    template = pathlib.Path(a.root) / "testdata" / "template"
    bin_dir = pathlib.Path(a.bin_dir)
    for d in ("requests", "anthropic"):
        if not (template / d).is_dir():
            sys.exit(f"{template / d} is missing")
    assert golden.PORT not in FORBIDDEN_PORTS
    server, proc = a.server, None
    if server is not None:
        check_port(server)
    else:
        proc = golden.start_server(bin_dir, a.model, ("--reasoning-preserve",))
        server = f"http://127.0.0.1:{golden.PORT}"
    try:
        rendered, failed = render_all(server.rstrip("/"), template)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                # Not while an error is propagating: a stuck server must not hide it.
                proc.kill()
    oracle = str(bin_dir / "artemis-oracle.exe")
    for id in rendered:
        subprocess.run([oracle, "tokenize", "--model", a.model, "--file", str(template / "prompts" / f"{id}.txt"),
                        "--out", str(template / "prompts" / f"{id}.tokens")], check=True)
    if failed:
        sys.exit(f"/apply-template refused: {', '.join(failed)}")


if __name__ == "__main__":
    main()
