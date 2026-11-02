"""The client setup in docs/install.md matches what the client smokes set (tools/py/capture_clients.py), and the relative
links in the public docs resolve.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v
"""

import json
import re
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
ROOT = TOOLS.parents[1]
INSTALL = ROOT / "docs" / "install.md"

sys.path.insert(0, str(TOOLS))
import capture_clients  # noqa: E402


def section(text: str, heading: str) -> str:
    """The text under the Markdown heading whose title is `heading`, up to the next heading of the same or a higher level."""
    m = re.search(rf"^(#+) {re.escape(heading)}\s*$", text, re.M)
    assert m, f"no heading {heading!r}"
    level = len(m.group(1))
    rest = text[m.end():]
    end = re.search(rf"^#{{1,{level}}} ", rest, re.M)
    return rest[: end.start()] if end else rest


def anchors(path: Path) -> set:
    """The GitHub anchors of the headings in a Markdown file, outside code fences."""
    found, fenced = set(), False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and (m := re.match(r"#+ (.+?)\s*$", line)):
            slug = re.sub(r"[^\w\- ]", "", m.group(1).lower()).replace(" ", "-")
            found.add(slug)
    return found


class ClientDocs(unittest.TestCase):
    def setUp(self):
        self.text = INSTALL.read_text(encoding="utf-8")

    def test_claude_code_block_names_the_smokes_variables(self):
        block = section(self.text, "Claude Code")
        smoke = capture_clients.client_vars("claude-code", "unused-root", smoke=True)
        doc = dict(re.findall(r"\$env:(\w+) = '([^']*)'", block))
        for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
            self.assertIn(name, doc)
        # every variable the doc sets is one the smoke sets too
        self.assertLessEqual(set(doc), set(smoke))
        self.assertEqual(doc["ANTHROPIC_MODEL"], capture_clients.MODEL)
        self.assertEqual(doc["ANTHROPIC_DEFAULT_HAIKU_MODEL"], capture_clients.MODEL)
        self.assertEqual(doc["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8080")
        self.assertNotIn("ANTHROPIC_SMALL_FAST_MODEL", block)  # deprecated in Claude Code

    def test_opencode_block_matches_the_smokes_config(self):
        block = section(self.text, "OpenCode")
        found = re.findall(r"```json\n(.*?)```", block, re.S)
        self.assertEqual(len(found), 1, "one json block under the OpenCode heading")
        cfg = json.loads(found[0])
        provider = cfg["provider"]["artemis"]
        self.assertEqual(provider["npm"], "@ai-sdk/openai-compatible")
        self.assertEqual(provider["options"]["baseURL"], "http://127.0.0.1:8080/v1")
        model = provider["models"][capture_clients.MODEL]
        self.assertEqual(model["limit"], {"context": 65536, "output": 16384})
        self.assertIs(model["tool_call"], True)
        self.assertEqual(cfg["model"], f"artemis/{capture_clients.MODEL}")

    def test_relative_links_and_anchors_resolve(self):
        link = re.compile(r"\]\(([^)\s]+)\)")
        docs = ("README.md", "AGENTS.md", "docs/install.md", "docs/server.md", "docs/new-moe-architecture.md")
        for rel in docs:
            path = ROOT / rel
            for target in link.findall(path.read_text(encoding="utf-8")):
                if re.match(r"[a-z][a-z0-9+.-]*:", target):
                    continue
                name, _, anchor = target.partition("#")
                file = (path.parent / name).resolve() if name else path
                self.assertTrue(file.exists(), f"{rel} links to {target}, which does not exist")
                if anchor and file.suffix == ".md":
                    self.assertIn(anchor, anchors(file), f"{rel} links to {target}, which has no such heading")


if __name__ == "__main__":
    unittest.main()
