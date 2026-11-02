"""The scripts the runner and the smokes start import only the standard library and each other, so they run with a plain
`python -I` and no package install. The fixture scripts that need gguf and numpy are the exception, and their packages
are an optional extra.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v
"""

import ast
import sys
import tomllib
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
# record_clients.py is started by capture_clients.py as a subprocess, so the import walk would not reach it.
RUNNER_SCRIPTS = ["build_speed_prompts.py", "gpu_link.py", "speed_llama.py", "llama_search.py", "capture_clients.py", "mtp_probe.py", "record_clients.py", "m6b_summary.py", "build_agent_kl_corpus.py", "request_quality.py", "submit_gaps.py"]


def top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


class StdlibOnly(unittest.TestCase):
    def test_runner_scripts_use_only_the_standard_library(self):
        local = {p.stem for p in TOOLS.glob("*.py")}
        pending = list(RUNNER_SCRIPTS)
        seen = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            path = TOOLS / name
            self.assertTrue(path.is_file(), f"{name} is not in tools/py")
            for mod in sorted(top_level_imports(path)):
                if mod in local:
                    pending.append(mod + ".py")
                else:
                    self.assertIn(mod, sys.stdlib_module_names, f"{name} imports {mod}, which is not in the standard library")

    def test_fixture_packages_are_optional(self):
        with open(TOOLS / "pyproject.toml", "rb") as f:
            project = tomllib.load(f)["project"]
        self.assertEqual(project["dependencies"], [])
        self.assertEqual(sorted(project["optional-dependencies"]["fixtures"]), ["gguf==0.17.1", "numpy>=2"])


if __name__ == "__main__":
    unittest.main()
