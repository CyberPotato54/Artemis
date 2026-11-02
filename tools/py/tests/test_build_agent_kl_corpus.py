"""Tests for tools/py/build_agent_kl_corpus.py: the agent-style KL corpus is built from committed files only, so nothing
is fetched.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v
"""

import hashlib
import importlib.util
import io
import json
import socket
import sys
import tempfile
import unittest
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1]
ROOT = TOOLS.parents[1]
TESTDATA = ROOT / "testdata"

spec = importlib.util.spec_from_file_location("build_agent_kl_corpus", TOOLS / "build_agent_kl_corpus.py")
m = importlib.util.module_from_spec(spec)
sys.modules["build_agent_kl_corpus"] = m
spec.loader.exec_module(m)


def no_network(*a, **k):
    raise AssertionError("the network was used")


class AgentKlCorpus(unittest.TestCase):
    def build(self, out: Path):
        buf = io.StringIO()
        with (
            mock.patch.object(urllib.request, "urlopen", no_network),
            mock.patch.object(urllib.request, "build_opener", no_network),
            mock.patch.object(socket, "create_connection", no_network),
            mock.patch.object(socket.socket, "connect", no_network),
            redirect_stdout(buf),
        ):
            m.main(["--root", str(ROOT), "--out", str(out)])
        return buf.getvalue()

    def test_corpus_is_built_from_committed_files_only(self):
        expected_files = (
            sorted(TESTDATA.glob("golden/prompts/16-*.txt"))
            + sorted(TESTDATA.glob("golden/prompts/17-*.txt"))
            + sorted(TESTDATA.glob("golden/prompts/18-*.txt"))
            + [TESTDATA / "speed" / "prompts" / "edit.txt"]
            + sorted(TESTDATA.glob("template/prompts/*.txt"))
        )
        # Sorted by path under testdata: golden, speed, template.
        expected_files.sort(key=lambda p: p.relative_to(TESTDATA).as_posix())
        self.assertEqual(len(expected_files), 22)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            printed = self.build(out)
            corpus = (out / "corpus.txt").read_bytes()
            sources = json.loads((out / "sources.json").read_text(encoding="utf-8"))
            # The files joined by a blank line, as one corpus without a final line end.
            joined = "\n\n".join(p.read_bytes().decode("utf-8") for p in expected_files).encode("utf-8")
            self.assertEqual(corpus, joined.removesuffix(b"\n"))
            self.assertFalse(corpus.endswith(b"\n"))
            self.assertNotIn(b"\r", corpus)
            # sources.json names each file in corpus order with its sha256 and licence.
            self.assertEqual([s["path"] for s in sources],
                             [p.relative_to(TESTDATA).as_posix() for p in expected_files])
            for s, p in zip(sources, expected_files):
                self.assertEqual(s["sha256"], hashlib.sha256(p.read_bytes()).hexdigest(), s["path"])
                self.assertTrue(s["license"], s["path"])
                self.assertIn("characters", s)
            by_path = {s["path"]: s for s in sources}
            self.assertIn("PSF-2.0", by_path["speed/prompts/edit.txt"]["license"])
            self.assertIn("MIT OR Apache-2.0", by_path["template/prompts/t01-multiturn-reasoning-dropped.txt"]["license"])
            # The controller reads the size estimate from the printed line (characters / 4).
            self.assertIn(f"about {len(corpus.decode('utf-8')) // 4:,} tokens", printed)

    def test_a_file_without_a_licence_in_the_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "repo"
            for p in [
                "golden/prompts/16-a.txt",
                "golden/prompts/17-b.txt",
                "golden/prompts/18-c.txt",
                "speed/prompts/edit.txt",
                "template/prompts/t01.txt",
            ]:
                f = root / "testdata" / p
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"x\n")
            (root / "testdata" / "manifest.toml").write_text(
                '[[file]]\npath = "golden/prompts/16-a.txt"\nlicense = "MIT"\nsha256 = "0"\n', encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                m.build(root)
            self.assertIn("golden/prompts/17-b.txt", str(cm.exception))

    def test_a_missing_input_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "testdata").mkdir()
            with self.assertRaises(SystemExit) as cm:
                m.build(Path(d))
            self.assertIn("testdata", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
