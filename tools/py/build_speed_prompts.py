"""Build testdata/speed: the speed prompts (write, edit and explain),
rendered as llama-server b11509 renders them with thinking off, and llama.cpp's token ids for them.

Usage: python -I tools/py/build_speed_prompts.py <repo root> <model.gguf> <oracle bin dir>
           [--stage render|made-from|check] [--server URL]

testdata/speed/prompts.json defines the prompts. The edit prompt quotes CPython's Lib/json/encoder.py (PSF-2.0), committed
in testdata/speed/sources/ with its licence text beside it, so nothing is fetched: a missing source is an error.
The stages:
- render: renders every prompt and tokenizes it with `artemis-oracle tokenize`, into testdata/speed/prompts/.
- made-from: records each prompt's token count and the FNV-1a 64 of its ids in testdata/speed/made-from.json.
- check (the runner's): makes any prompt whose token file is missing, then fails unless every prompt's ids match
  made-from.json.
Rendering starts llama-server on the CPU and stops it afterwards, unless --server names one already running.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_golden_set as golden  # noqa: E402  (post, start_server, fnv1a64 and the CPU server's PORT)


def speed_dir(root: pathlib.Path) -> pathlib.Path:
    return root / "testdata" / "speed"


def load(root: pathlib.Path) -> dict:
    return json.loads((speed_dir(root) / "prompts.json").read_text(encoding="utf-8"))


def source_text(root: pathlib.Path, s: dict) -> str:
    """A committed, pinned source (its path is relative to testdata/), checked against its sha256. Nothing is fetched."""
    path = root / "testdata" / s["path"]
    if not path.is_file():
        sys.exit(f"{s['id']}: {path} is missing; restore it from git (it is committed, pinned at {s['sha256']})")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != s["sha256"]:
        sys.exit(f"{s['id']}: {path} has sha256 {digest}, pinned {s['sha256']}; restore it from git")
    return data.decode("utf-8")


def message(root: pathlib.Path, defs: dict, p: dict) -> str:
    """A prompt's one user message: its text parts, and each source part's first `chars` characters."""
    sources = {s["id"]: s for s in defs["sources"]}
    return "".join(c["text"] if "text" in c else source_text(root, sources[c["source"]])[: c["chars"]]
                   for c in p["content"])


def where(root: pathlib.Path) -> pathlib.Path:
    return speed_dir(root) / "prompts"


def make(root: pathlib.Path, defs: dict, p: dict, server: str, oracle: str, model: str) -> None:
    out = where(root)
    out.mkdir(parents=True, exist_ok=True)
    body = {"messages": [{"role": "user", "content": message(root, defs, p)}],
            "chat_template_kwargs": {"enable_thinking": False}}
    txt = out / f"{p['id']}.txt"
    txt.write_bytes(golden.post(f"{server}/apply-template", body)["prompt"].encode("utf-8"))
    subprocess.run([oracle, "tokenize", "--model", model, "--file", str(txt), "--out", str(out / f"{p['id']}.tokens")],
                   check=True)


def ids_of(root: pathlib.Path, p: dict) -> dict:
    b = (where(root) / f"{p['id']}.tokens").read_bytes()
    return {"tokens": len(b) // 4, "fnv1a64": golden.fnv1a64(b)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("model")
    ap.add_argument("oracle_bin")
    ap.add_argument("--stage", choices=["render", "made-from", "check"], default="render")
    ap.add_argument("--server")
    a = ap.parse_args()
    root, bin_dir = pathlib.Path(a.repo), pathlib.Path(a.oracle_bin)
    defs = load(root)
    made_path = speed_dir(root) / "made-from.json"
    if a.stage == "made-from":
        made = {p["id"]: ids_of(root, p) for p in defs["prompts"]}
        made_path.write_text(json.dumps(made, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(f"made-from.json: {made}")
        return
    todo = defs["prompts"] if a.stage == "render" else [
        p for p in defs["prompts"] if not (where(root) / f"{p['id']}.tokens").exists()]
    if todo:
        proc = None if a.server else golden.start_server(bin_dir, a.model)
        server = a.server or f"http://127.0.0.1:{golden.PORT}"
        try:
            for p in todo:
                make(root, defs, p, server, str(bin_dir / "artemis-oracle.exe"), a.model)
        finally:
            if proc:
                proc.terminate()
                proc.wait()
    if a.stage == "check":
        made = json.loads(made_path.read_text(encoding="utf-8"))
        for p in defs["prompts"]:
            text = (where(root) / f"{p['id']}.txt").read_text(encoding="utf-8")
            if message(root, defs, p) not in text:
                sys.exit(f"{p['id']}: the rendered prompt doesn't contain the speed prompt's message; "
                         f"delete {where(root) / (p['id'] + '.txt')} and render again")
            got = ids_of(root, p)
            if got != made[p["id"]]:
                sys.exit(f"{p['id']}: {got['tokens']} tokens, fnv1a64 {got['fnv1a64']}; made-from.json has "
                         f"{made[p['id']]}. The rendered prompt isn't the speed prompt")
        print("speed prompts: " + ", ".join(f"{p['id']} {made[p['id']]['tokens']} tokens" for p in defs["prompts"]))


if __name__ == "__main__":
    main()
