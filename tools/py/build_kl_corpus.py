"""Build testdata/kl/corpus.txt (about 107K tokens) from pinned, permissively licensed sources.

Usage: python -I tools/py/build_kl_corpus.py <repo root>

Test 4 scores 25 chunks of 4,096 tokens. The corpus is code first, then prose, so about half the chunks are each.
Each source is downloaded, checked against the sha256 pinned in testdata/kl/sources.json when that file exists, and
cut at the last line end within its character budget. Line ends become LF and the corpus doesn't end with one:
llama-perplexity reads its file in text mode and drops one trailing newline, and this keeps its tokens and ours equal.
"""
import hashlib
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

SOURCES = [
    # id, url, license, kind, characters to keep
    ("rust-1.90.0-core-iter-iterator.rs",
     "https://raw.githubusercontent.com/rust-lang/rust/1.90.0/library/core/src/iter/traits/iterator.rs",
     "MIT OR Apache-2.0", "code", 47_500),
    ("llama.cpp-b11509-ggml.c", "https://raw.githubusercontent.com/ggml-org/llama.cpp/b11509/ggml/src/ggml.c",
     "MIT", "code", 47_500),
    ("transformers-v4.57.0-modeling-utils.py",
     "https://raw.githubusercontent.com/huggingface/transformers/v4.57.0/src/transformers/modeling_utils.py",
     "Apache-2.0", "code", 47_500),
    ("typescript-v5.9.2-core.ts", "https://raw.githubusercontent.com/microsoft/TypeScript/v5.9.2/src/compiler/core.ts",
     "Apache-2.0", "code", 47_500),
    ("pg-1661", "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
     "Public domain (Project Gutenberg #1661, The Adventures of Sherlock Holmes)", "prose", 107_000),
    ("pg-98", "https://www.gutenberg.org/cache/epub/98/pg98.txt",
     "Public domain (Project Gutenberg #98, A Tale of Two Cities)", "prose", 107_000),
]


def fetch(url: str, tries: int = 3) -> bytes:
    """The body of `url`, retried a couple of times; a failure ends with one line naming the URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "artemis-testdata/1"})
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == tries:
                sys.exit(f"can't fetch {url} after {tries} tries: {e}")
            time.sleep(2 * attempt)


def gutenberg_body(text: str) -> str:
    """The book without Project Gutenberg's header and footer (older files say THIS instead of THE)."""
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text)
    if not start or not end:
        sys.exit("no Project Gutenberg start or end marker")
    return text[start.end():end.start()].lstrip("\n")


def cut(text: str, keep: int) -> str:
    """The text up to the last line end within `keep` characters, without that line end."""
    end = text.rfind("\n", 0, keep + 1)
    if end <= 0:
        sys.exit(f"no line end in the first {keep:,} characters")
    return text[:end]


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    out_dir = root / "testdata" / "kl"
    out_dir.mkdir(parents=True, exist_ok=True)
    sources_path = out_dir / "sources.json"
    pinned = {s["id"]: s for s in json.loads(sources_path.read_text(encoding="utf-8"))} if sources_path.exists() else {}
    record, parts = [], []
    for sid, url, lic, kind, keep in SOURCES:
        raw = fetch(url)
        digest = hashlib.sha256(raw).hexdigest()
        if sid in pinned and pinned[sid]["sha256"] != digest:
            sys.exit(f"{sid} changed upstream: {digest} != {pinned[sid]['sha256']}")
        text = raw.decode("utf-8-sig").replace("\r\n", "\n")
        if "\r" in text:
            sys.exit(f"{sid} has a lone carriage return")
        if sid.startswith("pg-"):
            text = gutenberg_body(text)
        part = cut(text, keep)
        parts.append(part)
        record.append({"id": sid, "url": url, "license": lic, "kind": kind, "pinned": True, "sha256": digest,
                       "characters": len(part),
                       "used": f"the first {len(part):,} characters" + (" after the Project Gutenberg header"
                                                                         if sid.startswith("pg-") else "")
                               + ", line ends as LF"})
    corpus = "\n\n".join(parts)
    with open(out_dir / "corpus.txt", "w", encoding="utf-8", newline="") as f:
        f.write(corpus)
    sources_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    code = sum(len(p) for p, s in zip(parts, SOURCES) if s[3] == "code")
    print(f"corpus.txt: {len(corpus.encode()):,} bytes from {len(record)} sources; code {code:,} characters, "
          f"prose {len(corpus) - code - 2 * (len(parts) - 1):,}")


if __name__ == "__main__":
    main()
