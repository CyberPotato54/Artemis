"""submit_gaps: the per-submit gaps add up to the wall time, verifies are classed by the submit before them, and the
submits group into speculative steps.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v
"""

import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import submit_gaps  # noqa: E402


def trace(kinds):
    """Submits back to back: 10 us of host work, 20 us to launch, 100 us of GPU, 5 us to wake, recorded 4 us before."""
    out, t = [], 0
    for k in kinds:
        t += 10_000
        s = {"kind": k, "record_ns": t - 4_000, "submit_ns": t, "submitted_ns": t + 2_000}
        s["gpu_start_ns"] = t + 20_000
        s["gpu_end_ns"] = t + 120_000
        s["woke_ns"] = t + 125_000
        s["cal_err_ns"] = 9_000
        t = s["woke_ns"]
        out.append(s)
    return out


class SubmitGaps(unittest.TestCase):
    def test_the_gaps_add_up_to_the_wall_time(self):
        rows = submit_gaps.per_submit(trace(["decode", "decode"]))
        self.assertEqual(len(rows), 1, "the first submit has no previous wake")
        g = rows[0]["gaps"]
        self.assertEqual((g["host"], g["record"], g["launch"], g["gpu"], g["wake"]), (10, 4, 20, 100, 5))
        self.assertEqual(g["wall"], g["host"] + g["launch"] + g["gpu"] + g["wake"])
        self.assertEqual(g["idle"], 5 + 10 + 20)

    def test_verifies_are_classed_by_the_submit_before_them(self):
        kinds = ["decode", "mtp", "verify", "replay", "verify", "replay", "mtp", "decode", "decode"]
        rows = submit_gaps.per_submit(trace(kinds))
        classes = ["mtp_draft", "verify_mtp", "replay_mtp", "verify_lookup", "replay_lookup", "mtp_draft", "decode"]
        self.assertEqual([r["class"] for r in rows], classes + ["decode"])
        self.assertEqual([t for t, _ in submit_gaps.steps(rows)], ["mtp", "lookup", "mtp", "plain"])
        self.assertEqual([len(rs) for _, rs in submit_gaps.steps(rows)], [3, 2, 2, 1])

    def test_the_summary_shares_add_up_to_one(self):
        r = submit_gaps.analyze(trace(["decode", "mtp", "verify", "replay", "mtp", "verify", "replay"]))
        self.assertEqual(r["submits"], 6)
        s = r["by_step"]["mtp"]
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(sum(s["share_of_wall"][k] for k in submit_gaps.SHARES), 1.0, places=3)
        self.assertEqual(s["median_us"]["gpu"], 300)
        self.assertAlmostEqual(r["gpu_busy_share"], 100 / 135, places=3)


if __name__ == "__main__":
    unittest.main()
