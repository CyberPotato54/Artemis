"""mtp_route_hits: the top-N set comes from the first half only, the score from the second half, and the verdict needs
32 slots at 60% on every run.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import mtp_route_hits  # noqa: E402


def synthetic():
    """First half: experts 0..7 every row. Second half: 3 of 8 reads from 0..7 and 5 from experts 100..104 (never seen)."""
    first = [list(range(8)) for _ in range(10)]
    second = [[0, 1, 2, 100, 101, 102, 103, 104] for _ in range(10)]
    return first + second


class RouteHits(unittest.TestCase):
    def test_known_answer(self):
        r = mtp_route_hits.hit_rates(synthetic())
        # top 16 and 32 and 64 all hold 0..7 (only eight experts were seen): 3 of 8 reads served.
        self.assertEqual(r, {16: 0.375, 32: 0.375, 64: 0.375})

    def test_top_n_is_cut_by_count(self):
        # Expert 9 is read every row, expert 5 only in half: with 1 slot only 9 is resident.
        rows = [[9, 5]] * 2 + [[9, 6]] * 4 + [[9, 5]] * 2 + [[5, 5]] * 2
        # first half is 5 rows: counts 9:5, 5:2, 6:3 -> top1 = {9}; second half 5 rows, 10 reads, 9 appears 3 times.
        r = mtp_route_hits.hit_rates(rows, slots=(1,))
        self.assertEqual(r, {1: 3 / 10})

    def test_too_short(self):
        self.assertIsNone(mtp_route_hits.hit_rates([[1, 2]]))

    def test_verdict(self):
        good = {"a": {16: 0.5, 32: 0.61, 64: 0.7}, "b": {16: 0.5, 32: 0.60, 64: 0.7}}
        bad = {"a": {16: 0.5, 32: 0.61, 64: 0.7}, "b": {16: 0.5, 32: 0.59, 64: 0.7}}
        self.assertTrue(mtp_route_hits.verdict(good)[0])
        self.assertFalse(mtp_route_hits.verdict(bad)[0])
        self.assertFalse(mtp_route_hits.verdict({"a": None})[0])
        self.assertFalse(mtp_route_hits.verdict({})[0])

    def test_reads_the_log_format(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            lines = [
                json.dumps({"pass": i, "pos": 100 + i, "j": 0, "ids": row}) for i, row in enumerate(synthetic())
            ]
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertEqual(mtp_route_hits.read_rows(str(p)), synthetic())


if __name__ == "__main__":
    unittest.main()
