"""capture_clients.blank_opencode_key: after a clean key scan the smoke blanks the run's key in OpenCode's two temporary
configs, so the key does not outlive the run on disk.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import capture_clients  # noqa: E402

KEY = "k" * 43


def config_paths(root):
    home = os.path.join(root, "home")
    return [os.path.join(home, "opencode.json"), os.path.join(home, ".config", "opencode", "opencode.json")]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class BlankOpenCodeKey(unittest.TestCase):
    def test_blank_opencode_key(self):
        with tempfile.TemporaryDirectory() as root:
            capture_clients.write_opencode_config(root, key=KEY, smoke=True)
            paths = config_paths(root)
            before = [load(p) for p in paths]
            for p, cfg in zip(paths, before):
                self.assertEqual(cfg["provider"]["artemis"]["options"]["apiKey"], KEY, p)

            capture_clients.blank_opencode_key(root)

            for p, cfg in zip(paths, before):
                after = load(p)
                self.assertEqual(after["provider"]["artemis"]["options"]["apiKey"], "", p)
                cfg["provider"]["artemis"]["options"]["apiKey"] = ""
                self.assertEqual(after, cfg, f"{p}: every other field is unchanged")
                self.assertNotIn(KEY, Path(p).read_text(encoding="utf-8"), p)

    def test_blank_opencode_key_skips_a_config_that_was_never_written(self):
        with tempfile.TemporaryDirectory() as root:
            capture_clients.write_opencode_config(root, key=KEY)
            first, second = config_paths(root)
            os.remove(second)
            capture_clients.blank_opencode_key(root)
            self.assertEqual(load(first)["provider"]["artemis"]["options"]["apiKey"], "")
            self.assertFalse(os.path.exists(second))

    def test_blank_opencode_key_with_no_configs_does_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            capture_clients.blank_opencode_key(root)
            self.assertEqual(os.listdir(root), [])


if __name__ == "__main__":
    unittest.main()
