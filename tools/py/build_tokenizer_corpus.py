"""Build testdata/tokenizer/corpus.txt (about 1 MB) from pinned, permissively licensed sources.

Usage: uv run --project tools/py --extra fixtures python -I tools/py/build_tokenizer_corpus.py <repo root> <model.gguf>

Each source is downloaded and sliced. Its sha256 goes into testdata/tokenizer/sources.json, and on later runs a
pinned source must match it. Tatoeba exports change daily, so they are recorded but not pinned; the committed corpus
is the canonical copy. Text is joined with "\\n\\n", and the edge cases, including every special token, come last.
"""
import bz2
import hashlib
import json
import pathlib
import re
import sys
import urllib.request

from gguf import GGUFReader

SOURCES = [
    # id, url, license, kind, characters to keep
    ("pg-1342", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
     "Public domain (Project Gutenberg #1342, Pride and Prejudice)", "prose", 250_000),
    ("pg-84", "https://www.gutenberg.org/cache/epub/84/pg84.txt",
     "Public domain (Project Gutenberg #84, Frankenstein)", "prose", 100_000),
    ("llama.cpp-b11509-llama-vocab.cpp",
     "https://raw.githubusercontent.com/ggml-org/llama.cpp/b11509/src/llama-vocab.cpp", "MIT", "code", 150_000),
    ("transformers-v4.57.0-generation-utils.py",
     "https://raw.githubusercontent.com/huggingface/transformers/v4.57.0/src/transformers/generation/utils.py",
     "Apache-2.0", "code", 150_000),
    ("typescript-v5.9.2-scanner.ts",
     "https://raw.githubusercontent.com/microsoft/TypeScript/v5.9.2/src/compiler/scanner.ts", "Apache-2.0", "code",
     100_000),
    ("rust-1.90.0-core-str-mod.rs",
     "https://raw.githubusercontent.com/rust-lang/rust/1.90.0/library/core/src/str/mod.rs", "MIT OR Apache-2.0",
     "code", 100_000),
]
TATOEBA_LANGS = ["cmn", "jpn", "kor", "rus", "ara", "hin", "deu", "fra", "spa", "vie", "tha", "heb", "ell", "tur",
                 "ukr", "pol"]
TATOEBA_URL = "https://downloads.tatoeba.org/exports/per_language/{lang}/{lang}_sentences.tsv.bz2"
TATOEBA_LICENSE = "CC-BY-2.0-FR (Tatoeba, https://tatoeba.org; sentences by Tatoeba contributors)"
TATOEBA_CHARS = 12_000

EDGE = "".join([
    "Emoji: 👍🏽 👨‍👩‍👧‍👦 🏳️‍🌈 ❤️ 🇯🇵\n",
    "Combining: e\u0301 a\u0308 n\u0303 \u0915\u093f \u0e01\u0e34 \u05e9\u05c1\n",
    "CRLF line one\r\nCRLF line two\r\n\r\n",
    "Tabs\tand\t\ttabs; spaces:" + " " * 17 + "end\n",
    "Lone CR\rmid-line\n",
    "Digits: 0 12 345 6789 1234567890 3.14159 1e-6 \u0663\u0664\u0665 \u09e7\u09e8\u09e9\n",
    "Contractions: I'll've, they'RE, Y'ALL'D, rock'n'roll, '' ' '\n",
    "Specials in context: <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\nx\n</think>\n\n"
    "<tool_call>\n<function=f>\n</function>\n</tool_call><|endoftext|>\n",
    "Near-specials: <|im_end <|im_end| |im_end|> <think <|IM_END|> <<|im_end|>>\n",
    "Unicode spaces: a\u00a0b\u2003c\u3000d\u200be\n",
    "Long runs: " + "=" * 300 + "\n" + " " * 200 + "x\n",
    "Mixed scripts: Hello世界Привет🌍مرحبا\n",
])


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "artemis-testdata/1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def gutenberg_body(text: str) -> str:
    """The book without Project Gutenberg's header and footer (older files say THIS instead of THE)."""
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text)
    if not start or not end:
        sys.exit("no Project Gutenberg start or end marker")
    return text[start.end():end.start()].lstrip("\r\n")


def main() -> None:
    root, model = pathlib.Path(sys.argv[1]), sys.argv[2]
    out_dir = root / "testdata" / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    sources_path = out_dir / "sources.json"
    pinned = {s["id"]: s for s in json.loads(sources_path.read_text())} if sources_path.exists() else {}
    record, parts = [], []
    for sid, url, lic, kind, keep in SOURCES:
        raw = fetch(url)
        digest = hashlib.sha256(raw).hexdigest()
        if sid in pinned and pinned[sid]["sha256"] != digest:
            sys.exit(f"{sid} changed upstream: {digest} != {pinned[sid]['sha256']}")
        text = raw.decode("utf-8-sig")
        if sid.startswith("pg-"):
            text = gutenberg_body(text)
        parts.append(text[:keep])
        record.append({"id": sid, "url": url, "license": lic, "kind": kind, "pinned": True, "sha256": digest,
                       "used": f"the first {keep:,} characters" + (" after the Project Gutenberg header"
                                                                   if sid.startswith("pg-") else "")})
    for lang in TATOEBA_LANGS:
        url = TATOEBA_URL.format(lang=lang)
        raw = fetch(url)
        lines, size = [], 0
        for row in bz2.decompress(raw).decode("utf-8").splitlines():
            sentence = row.split("\t", 2)[2]
            lines.append(sentence)
            size += len(sentence) + 1
            if size >= TATOEBA_CHARS:
                break
        parts.append("\n".join(lines))
        record.append({"id": f"tatoeba-{lang}", "url": url, "license": TATOEBA_LICENSE, "kind": "multilingual",
                       "pinned": False, "sha256": hashlib.sha256(raw).hexdigest(),
                       "used": f"the first {TATOEBA_CHARS:,} characters of sentences, in id order"})
    reader = GGUFReader(model)
    tokens = reader.fields["tokenizer.ggml.tokens"]
    types = reader.fields["tokenizer.ggml.token_type"]
    specials = [bytes(tokens.parts[tokens.data[i]]).decode() for i in range(len(tokens.data))
                if int(types.parts[types.data[i]][0]) in (3, 4)]
    edge = EDGE + "All specials, spaced: " + " ".join(specials) + "\nAll specials, joined: " + "".join(specials) + "\n"
    record.append({"id": "artemis-edge-cases", "url": "tools/py/build_tokenizer_corpus.py",
                   "license": "MIT OR Apache-2.0", "kind": "edge cases", "pinned": True,
                   "sha256": hashlib.sha256(edge.encode()).hexdigest(), "used": "all of it"})
    parts.append(edge)
    corpus = "\n\n".join(parts)
    with open(out_dir / "corpus.txt", "w", encoding="utf-8", newline="") as f:
        f.write(corpus)
    sources_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"corpus.txt: {len(corpus.encode()):,} bytes from {len(record)} sources; {len(specials)} special tokens")


main()
