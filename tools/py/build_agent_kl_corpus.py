"""Build the agent-style KL corpus from committed test data only (plan 6b, task 16). Nothing is fetched.

Usage: python -I tools/py/build_agent_kl_corpus.py --out <folder> [--root <repo root>]

The corpus is the rendered prompts the repository already holds, in sorted order of their path under testdata/:
golden/prompts/16-*, 17-* and 18-* (the tool prompts), speed/prompts/edit.txt and template/prompts/*.txt. The files are
joined by a blank line and the corpus doesn't end with a line end, because llama-perplexity reads its file in text mode
and drops one trailing newline. Each file's licence comes from testdata/manifest.toml.

Writes <out>/corpus.txt and <out>/sources.json (each file's path, sha256, licence and characters, in corpus order) and
prints the corpus's size with a token estimate of characters / 4. The estimate is rough: these are mostly system text,
tool schemas and code, and a scored half of each chunk is all a KL run counts.
"""

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path

GLOBS = ["golden/prompts/16-*.txt", "golden/prompts/17-*.txt", "golden/prompts/18-*.txt", "speed/prompts/edit.txt",
         "template/prompts/*.txt"]
REPO = Path(__file__).resolve().parents[2]


def inputs(testdata: Path) -> list[Path]:
    found = []
    for pattern in GLOBS:
        hits = sorted(testdata.glob(pattern))
        if not hits:
            sys.exit(f"no input matches {pattern} under {testdata}")
        found.extend(hits)
    return sorted(found, key=lambda p: p.relative_to(testdata).as_posix())


def licences(testdata: Path) -> dict[str, str]:
    manifest = testdata / "manifest.toml"
    if not manifest.is_file():
        sys.exit(f"{manifest} is missing")
    with open(manifest, "rb") as f:
        return {e["path"]: e.get("license", "") for e in tomllib.load(f).get("file", [])}


def build(root: Path) -> tuple[str, list[dict]]:
    testdata = root / "testdata"
    if not testdata.is_dir():
        sys.exit(f"{testdata} is not a folder: pass the repository root with --root")
    lic = licences(testdata)
    parts, record = [], []
    for p in inputs(testdata):
        rel = p.relative_to(testdata).as_posix()
        if not lic.get(rel):
            sys.exit(f"{rel} has no licence in testdata/manifest.toml")
        raw = p.read_bytes()
        if b"\r" in raw:
            sys.exit(f"{rel} has a carriage return")
        text = raw.decode("utf-8")
        parts.append(text)
        record.append({"path": rel, "sha256": hashlib.sha256(raw).hexdigest(), "license": lic[rel],
                       "characters": len(text)})
    corpus = "\n\n".join(parts).removesuffix("\n")
    return corpus, record


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path, help="folder for corpus.txt and sources.json")
    ap.add_argument("--root", type=Path, default=REPO, help="repository root (default: this checkout)")
    args = ap.parse_args(argv)
    corpus, record = build(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "corpus.txt", "w", encoding="utf-8", newline="") as f:
        f.write(corpus)
    (args.out / "sources.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8", newline="\n")
    chars = len(corpus)
    print(f"corpus.txt: {len(corpus.encode()):,} bytes, {chars:,} characters from {len(record)} files; "
          f"about {chars // 4:,} tokens (characters / 4)")


if __name__ == "__main__":
    main()
