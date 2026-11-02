"""Write testdata/manifest.toml: every committed test-data file with its source, license and sha256.

Usage: python -I tools/py/write_manifest.py <repo root>
Run it after any test-data file changes; CI doesn't check hashes, reviewers do.
"""
import hashlib
import json
import pathlib
import sys

MODEL = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf (sha256 0b21525e…c58b), Apache-2.0"
ORACLE = "artemis-oracle (llama.cpp b11509)"
OURS = "MIT OR Apache-2.0"

# (glob under testdata/, how it was made, license)
FILES = [
    ("tokenizer/corpus.txt", "tools/py/build_tokenizer_corpus.py from the sources below", "per source, see below"),
    ("tokenizer/sources.json", "tools/py/build_tokenizer_corpus.py", OURS),
    ("tokenizer/corpus.special.ids", f"{ORACLE} tokenize", OURS),
    ("tokenizer/corpus.plain.ids", f"{ORACLE} tokenize --no-parse-special", OURS),
    ("tokenizer/qwen36-vocab.bin", f"artemis extract-vocab from {MODEL}", "Apache-2.0 (the model's vocabulary)"),
    ("tokenizer/eog.json", f"{ORACLE} vocab", OURS),
    ("test2/inputs/*.txt", "written for Artemis", OURS),
    ("test2/inputs/*.tokens", f"{ORACLE} tokenize", OURS),
    ("test2/tiny/**/*", f"{ORACLE} taps on the tiny model (artemis tiny-model --seed 1)", OURS),
    ("golden/prompts.json", "written for Artemis", OURS),
    ("golden/sources/*", "tools/py/build_golden_set.py, fetched at the pinned revision below",
     "MIT OR Apache-2.0 (The Rust Project Developers)"),
    ("golden/made-from.json", "tools/py/build_golden_set.py: FNV-1a 64 of the files each output was made from", OURS),
    ("golden/prompts/*.txt",
     "tools/py/build_golden_set.py: test 2's inputs verbatim, chat prompts rendered by llama-server b11509's "
     "/apply-template", f"{OURS}; 20-long-option-rs quotes rust-lang/rust's option.rs ({OURS})"),
    ("golden/prompts/*.tokens", f"{ORACLE} tokenize", OURS),
    ("golden/expected/*.json", f"{ORACLE} greedy --backend vulkan --experts-on-cpu --n 256", OURS),
    ("golden/forced-cpu/*.json", f"{ORACLE} forced --backend cpu --n 256, fed golden/expected's tokens", OURS),
    ("speed/prompts.json", "written for Artemis from the spike's spec_client.py", OURS),
    # The edit prompt quotes CPython's encoder.py; the entries for it come first, since a path takes the first pattern
    # that matches it (see main), so the globs after them cover write and explain.
    ("speed/prompts/edit.txt", "tools/py/build_speed_prompts.py: rendered by llama-server b11509's /apply-template",
     "PSF-2.0 (the quoted CPython text, see speed/sources/LICENSE-PSF) and MIT OR Apache-2.0 (the rest)"),
    ("speed/prompts/edit.tokens", f"{ORACLE} tokenize",
     f"{OURS}; the ids of a prompt that quotes CPython's encoder.py (PSF-2.0)"),
    ("speed/prompts/*.txt", "tools/py/build_speed_prompts.py: rendered by llama-server b11509's /apply-template", OURS),
    ("speed/prompts/*.tokens", f"{ORACLE} tokenize", OURS),
    ("speed/sources/encoder.py", "CPython v3.13.3 Lib/json/encoder.py, a byte copy (sha256 pinned in speed/prompts.json)",
     "PSF-2.0; the license text is speed/sources/LICENSE-PSF"),
    ("speed/sources/LICENSE-PSF",
     "the PSF License Agreement, copied from CPython's LICENSE at tag v3.13.3", "PSF-2.0"),
    ("speed/made-from.json",
     "tools/py/build_speed_prompts.py: each speed prompt's token count and FNV-1a 64 of its ids, the edit prompt's "
     "included", OURS),
    ("kl/corpus.txt", "tools/py/build_kl_corpus.py from the sources below", "per source, see below"),
    ("kl/sources.json", "tools/py/build_kl_corpus.py", OURS),
    ("template/qwen36-chat-template.jinja", "tokenizer.chat_template of " + MODEL,
     "Apache-2.0 (the model's; GGUF general.license and the Unsloth card, checked 2026-10-18)"),
    ("template/requests/*.json", "written for Artemis", OURS),
    ("template/anthropic/*.json", "written for Artemis", OURS),
    ("template/prompts/*.txt",
     "tools/py/build_template_fixtures.py: rendered by llama-server b11509's /apply-template", OURS),
    ("template/prompts/*.tokens", f"{ORACLE} tokenize", OURS),
    ("template/anthropic/*.count",
     "tools/py/build_template_fixtures.py: llama-server b11509's /v1/messages/count_tokens", OURS),
    ("clients/opencode/*.json",
     "tools/py/capture_clients.py: OpenCode 2.0.30 against the recording stub, scrubbed by tools/py/scrub_captures.py",
     "MIT (OpenCode's text); MIT OR Apache-2.0 (the rest)"),
    ("clients/opencode/LICENSE-OpenCode",
     "OpenCode's copyright notice, copied from the upstream repository's LICENSE at tag v2.0.30 (anomalyco/opencode)", "MIT"),
    ("clients/claude-code/*.json",
     "tools/py/capture_clients.py: Claude Code 2.1.302 against the recording stub; every string replaced by "
     "tools/py/scrub_captures.py",
     "The structure of Claude Code 2.1.302's requests, with every string replaced; no licence is claimed over Claude Code itself"),
]

# (file under testdata/, where its list of sources is)
SOURCES = [
    ("tokenizer/sources.json", None),
    ("golden/prompts.json", "sources"),
    ("speed/prompts.json", "sources"),
    ("kl/sources.json", None),
]


def q(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def main() -> None:
    root = pathlib.Path(sys.argv[1]) / "testdata"
    out = ["# Every committed test-data file: how it was made, its license and its sha256. See README.md and THIRD-PARTY-NOTICES.md.", ""]
    seen = set()
    for pattern, made, lic in FILES:
        for path in sorted(p for p in root.glob(pattern) if p.is_file()):
            rel = path.relative_to(root).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            out += ["[[file]]", f"path = {q(rel)}", f"made = {q(made)}", f"license = {q(lic)}",
                    f"sha256 = {q(hashlib.sha256(path.read_bytes()).hexdigest())}", ""]
    sources = []
    for path, key in SOURCES:
        listed = json.loads((root / path).read_text(encoding="utf-8"))
        sources += listed[key] if key else listed
    for s in sources:
        out += ["[[source]]"] + [f"{k} = {q(v) if isinstance(v, str) else str(v).lower()}" for k, v in s.items()] + [""]
    (root / "manifest.toml").write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"manifest.toml: {sum(1 for l in out if l == '[[file]]')} files, {len(sources)} sources")


main()
