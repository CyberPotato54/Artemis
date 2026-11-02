"""Tests for tools/py/build_speed_prompts.py and the committed speed prompts: the edit prompt and its PSF-licensed source
are committed, so nothing fetches them.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v
"""

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1]
ROOT = TOOLS.parents[1]
SPEED = ROOT / "testdata" / "speed"

sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("build_speed_prompts", TOOLS / "build_speed_prompts.py")
m = importlib.util.module_from_spec(spec)
sys.modules["build_speed_prompts"] = m
spec.loader.exec_module(m)


def fnv1a64(b: bytes) -> str:
    h = 0xCBF29CE484222325
    for x in b:
        h = ((h ^ x) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


class SpeedPromptsAreCommitted(unittest.TestCase):
    def test_edit_source_is_committed_and_pinned(self):
        defs = json.loads((SPEED / "prompts.json").read_text(encoding="utf-8"))
        src = next(s for s in defs["sources"] if s["id"] == "cpython-json-encoder")
        self.assertTrue(src["committed"])
        self.assertEqual(src["path"], "speed/sources/encoder.py")
        path = ROOT / "testdata" / src["path"]
        self.assertTrue(path.is_file(), f"{path} is missing")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), src["sha256"])
        # The PSF licence text sits beside the source it covers.
        self.assertTrue((path.parent / "LICENSE-PSF").is_file())

    def test_every_prompt_is_committed(self):
        defs = json.loads((SPEED / "prompts.json").read_text(encoding="utf-8"))
        made = json.loads((SPEED / "made-from.json").read_text(encoding="utf-8"))
        self.assertEqual({p["id"] for p in defs["prompts"]}, set(made))
        for p in defs["prompts"]:
            self.assertTrue(p["committed"], p["id"])
            self.assertEqual(m.where(ROOT), SPEED / "prompts", p["id"])
            ids = (SPEED / "prompts" / f"{p['id']}.tokens").read_bytes()
            self.assertEqual({"tokens": len(ids) // 4, "fnv1a64": fnv1a64(ids)}, made[p["id"]], p["id"])
            self.assertTrue((SPEED / "prompts" / f"{p['id']}.txt").is_file(), p["id"])

    def test_no_fetch_left(self):
        s = {"id": "x", "url": "https://example.invalid/x", "sha256": "0" * 64, "path": "speed/sources/missing.py"}

        def boom(*a, **k):
            raise AssertionError("fetch was called")

        with tempfile.TemporaryDirectory() as d, mock.patch.object(m.golden, "fetch", boom):
            with self.assertRaises(SystemExit) as cm:
                m.source_text(Path(d), s)
            self.assertIn("missing.py", str(cm.exception))

    def test_a_source_that_differs_from_its_pin_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "testdata" / "speed" / "sources" / "changed.py"
            path.parent.mkdir(parents=True)
            content = b"x = 1"
            path.write_bytes(content)
            s = {"id": "x", "sha256": "0" * 64, "path": "speed/sources/changed.py"}
            with self.assertRaises(SystemExit) as cm:
                m.source_text(Path(d), s)
            msg = str(cm.exception)
            self.assertIn("changed.py", msg)
            self.assertIn(hashlib.sha256(content).hexdigest(), msg)


if __name__ == "__main__":
    unittest.main()
