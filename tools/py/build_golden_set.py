"""Build testdata/golden: test 3's twenty prompts, llama.cpp's token ids for them, llama.cpp's greedy output and
llama.cpp CPU's run along it.

Usage: python -I tools/py/build_golden_set.py <repo root> <model.gguf> <oracle bin dir>
           [--stage render|tokenize|expected|floor|made-from|all] [--only ID] [--server URL]

testdata/golden/prompts.json defines the prompts. The stages:
- render: test 2's four inputs are copied byte for byte; chat prompts are rendered by llama-server b11509's
  /apply-template, which gives the prompt llama-server itself would feed the model. Unless --server names one already
  running, the script starts llama-server on the CPU (no GPU, no warm-up) and stops it afterwards.
- tokenize: `artemis-oracle tokenize`, with special tokens parsed, writes prompts/<id>.tokens.
- expected: `artemis-oracle greedy` on Vulkan with the experts in host memory writes expected/<id>.json, 256 steps or
  up to an end-of-generation token. It takes about a minute per prompt.
- floor: `artemis-oracle forced` on the CPU, fed the prompt and then expected/<id>.json's tokens, writes
  forced-cpu/<id>.json: test 3's floor. Rerun it whenever an expected output changes.
- made-from: records in golden/made-from.json, for each id, the FNV-1a 64 of the files its outputs were made from
  (prompts/<id>.tokens for expected/ and forced-cpu/, and expected/<id>.json for forced-cpu/), so golden_set.rs can
  fail on a stale output of the same length. The expected and floor stages record their own entries as they run; use
  this stage only to record the committed files as they are, when they were made together and checked.
Sources quoted in prompts are fetched once into testdata/golden/sources/ and checked against their pinned sha256.
"""
import argparse
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

PORT = 8097


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "artemis-testdata/1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def source(testdata: pathlib.Path, s: dict) -> None:
    """Fetches a pinned source into testdata if it isn't there, and checks its sha256 either way."""
    path = testdata / s["path"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(s["url"]))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != s["sha256"]:
        sys.exit(f"{s['id']}: sha256 {digest}, pinned {s['sha256']}")


def content(c, testdata: pathlib.Path) -> str:
    """A message's content: a string, or a list of {"text": ...} and {"file": path under testdata} parts, joined."""
    if isinstance(c, str):
        return c
    return "".join(p["text"] if "text" in p else (testdata / p["file"]).read_bytes().decode("utf-8") for p in c)


def render(p: dict, server: str, testdata: pathlib.Path) -> bytes:
    if "text_from" in p:
        return (testdata / p["text_from"]).read_bytes()
    messages = [dict(m, content=content(m["content"], testdata)) for m in p["messages"]]
    body = {"messages": messages, "chat_template_kwargs": {"enable_thinking": p["enable_thinking"]}}
    if "tools" in p:
        body["tools"] = p["tools"]
    return post(f"{server}/apply-template", body)["prompt"].encode("utf-8")


def start_server(bin_dir: pathlib.Path, model: str, extra: tuple[str, ...] = ()) -> subprocess.Popen:
    """llama-server on the CPU, ready to answer /apply-template; `extra` are further command-line arguments."""
    log = pathlib.Path(tempfile.gettempdir()) / "artemis-golden-llama-server.log"
    # The child keeps its own copy of the log handle; ours is closed as soon as it has started.
    with open(log, "w") as log_file:
        proc = subprocess.Popen(
            [str(bin_dir / "llama-server.exe"), "-m", model, "--device", "none", "-ngl", "0", "-c", "4096", "-np", "1",
             "--jinja", "--no-warmup", "--port", str(PORT), *extra],
            stdout=subprocess.DEVNULL, stderr=log_file)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            sys.exit(f"llama-server exited with {proc.returncode}; see {log}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return proc
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(2)
    proc.terminate()
    sys.exit(f"llama-server wasn't ready within 600 s; see {log}")


def fnv1a64(b: bytes) -> str:
    """The same fingerprint as qwen36::testing::fnv1a64, as 16 hex digits."""
    h = 0xCBF29CE484222325
    for x in b:
        h = ((h ^ x) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def record(golden: pathlib.Path, id: str, part: str, **made_from: pathlib.Path) -> None:
    """Sets made-from.json's entry for id's expected or floor outputs to the hashes of the files named."""
    path = golden / "made-from.json"
    made = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    made.setdefault(id, {})[part] = {k: fnv1a64(v.read_bytes()) for k, v in made_from.items()}
    path.write_text(json.dumps(made, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def f32(x: float) -> float:
    """x rounded to f32, as test 3 reads the logits."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def flips(floor: dict, expected: dict) -> int:
    """Test 3's count: steps where the floor's top token isn't the oracle's and the oracle's gap exceeds 0.05."""
    gap = lambda s: f32(f32(s["logit"]) - f32(s["second_logit"]))
    return sum(1 for f, e in zip(floor["steps"], expected["steps"]) if f["top"] != e["token"] and gap(e) > f32(0.05))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("model")
    ap.add_argument("bin_dir")
    ap.add_argument("--stage", choices=["render", "tokenize", "expected", "floor", "made-from", "all"], default="all")
    ap.add_argument("--only")
    ap.add_argument("--server")
    a = ap.parse_args()
    testdata = pathlib.Path(a.root) / "testdata"
    golden = testdata / "golden"
    bin_dir = pathlib.Path(a.bin_dir)
    spec = json.loads((golden / "prompts.json").read_text(encoding="utf-8"))
    for s in spec["sources"]:
        source(testdata, s)
    prompts = [p for p in spec["prompts"] if a.only in (None, p["id"])]
    if not prompts:
        sys.exit(f"no prompt {a.only!r} in prompts.json")
    for d in ("prompts", "expected", "forced-cpu"):
        (golden / d).mkdir(exist_ok=True)
    txt = lambda p: golden / "prompts" / f"{p['id']}.txt"
    tokens = lambda p: golden / "prompts" / f"{p['id']}.tokens"
    if a.stage in ("render", "all"):
        server, proc = a.server, None
        if server is None and any("messages" in p for p in prompts):
            proc = start_server(bin_dir, a.model)
            server = f"http://127.0.0.1:{PORT}"
        try:
            for p in prompts:
                text = render(p, server, testdata)
                txt(p).write_bytes(text)
                print(f"{p['id']}: {len(text):,} bytes")
        finally:
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    # Not while an error is propagating: a stuck server must not hide it.
                    proc.kill()
    oracle = str(bin_dir / "artemis-oracle.exe")
    if a.stage in ("tokenize", "all"):
        for p in prompts:
            subprocess.run([oracle, "tokenize", "--model", a.model, "--file", str(txt(p)), "--out", str(tokens(p))],
                           check=True)
    if a.stage in ("expected", "all"):
        for p in prompts:
            out = golden / "expected" / f"{p['id']}.json"
            subprocess.run([oracle, "greedy", "--model", a.model, "--backend", "vulkan", "--experts-on-cpu",
                            "--tokens", str(tokens(p)), "--n", str(spec["n"]), "--out", str(out)], check=True)
            record(golden, p["id"], "expected", prompt=tokens(p))
            e = json.loads(out.read_text(encoding="utf-8"))
            print(f"{p['id']}: {e['prompt_tokens']:,} prompt tokens, {len(e['steps'])} steps"
                  + (", stopped at an end-of-generation token" if e["stopped_at_eog"] else ""))
    if a.stage in ("floor", "all"):
        with tempfile.TemporaryDirectory() as tmp:
            for p in prompts:
                e = json.loads((golden / "expected" / f"{p['id']}.json").read_text(encoding="utf-8"))
                n_tokens = tokens(p).stat().st_size // 4
                if e["prompt_tokens"] != n_tokens:
                    sys.exit(f"{p['id']}: expected/{p['id']}.json was made from {e['prompt_tokens']:,} prompt tokens, "
                             f"but prompts/{p['id']}.tokens holds {n_tokens:,}: rerun --stage expected first")
                force = pathlib.Path(tmp) / f"{p['id']}.tokens"
                force.write_bytes(struct.pack(f"<{len(e['steps'])}I", *(s["token"] for s in e["steps"])))
                out = golden / "forced-cpu" / f"{p['id']}.json"
                start = time.monotonic()
                subprocess.run([oracle, "forced", "--model", a.model, "--backend", "cpu", "--tokens", str(tokens(p)),
                                "--force", str(force), "--n", str(spec["n"]), "--out", str(out)], check=True)
                record(golden, p["id"], "floor", prompt=tokens(p), expected=golden / "expected" / f"{p['id']}.json")
                f = json.loads(out.read_text(encoding="utf-8"))
                print(f"{p['id']}: {len(f['steps'])} steps, {flips(f, e)} flips where the oracle's gap exceeds 0.05, "
                      f"in {time.monotonic() - start:.0f} s")

    if a.stage == "made-from":
        for p in prompts:
            expected = golden / "expected" / f"{p['id']}.json"
            record(golden, p["id"], "expected", prompt=tokens(p))
            record(golden, p["id"], "floor", prompt=tokens(p), expected=expected)


if __name__ == "__main__":
    main()
