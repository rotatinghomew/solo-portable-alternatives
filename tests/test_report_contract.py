"""Contract checks on the printed report: uncertainty, trial rules, refresh."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parent_aid.data import Dataset, RotationChange  # noqa: E402
from parent_aid.report import build_report, markdown_summary, refresh_decision, trial_plan  # noqa: E402
from parent_aid import model, schedule, support  # noqa: E402


class ReportContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset.load(ROOT / "data" / "records.json")
        cls.report = build_report(cls.ds, bootstrap_draws=6)

    def test_every_proposal_has_uncertainty(self):
        for name, data in self.report["proposals"].items():
            unc = data["uncertainty"]
            self.assertGreater(unc["draws"], 0)
            self.assertLessEqual(unc["participants_delta"]["p05"], unc["participants_delta"]["p95"])
            self.assertGreaterEqual(unc["probability_within_capacity"], 0.0)
            self.assertLessEqual(unc["probability_within_capacity"], 1.0)

    def test_markdown_covers_required_sections(self):
        text = markdown_summary(self.report)
        for heading in ("留出轮次核验", "提案", "证据缺口"):
            self.assertIn(heading, text)
        self.assertIn("不能外推", text)
        self.assertIn("试开", text)

    def test_unsupported_change_never_formal(self):
        """A change to an unobserved site/slot must be trial-only."""
        change = RotationChange("X", "B11", "站2", "周末", "迁移")
        judged = support.judge_change(self.ds, change)
        self.assertEqual(judged.tier, "unsupported")
        bundle = model.fit_bundle(self.ds, [1, 2, 3, 4, 5])
        current = schedule.evaluate_scenario(
            self.ds, schedule.build_scenario(self.ds, "current", []), bundle
        )
        plan = trial_plan(self.ds, "X", [judged], current)
        self.assertIn("试开", plan["rows"][0]["decision"])

    def test_trial_cannot_take_confirmed_seats(self):
        ds = self.ds
        bundle = model.fit_bundle(ds, [1, 2, 3, 4, 5])
        current = schedule.evaluate_scenario(
            ds, schedule.build_scenario(ds, "current", []), bundle
        )
        change = RotationChange("X", "B0", "站0", "周末", "迁移")
        judged = support.judge_change(ds, change)
        plan = trial_plan(ds, "X", [judged], current)
        headroom = plan["rows"][0]["trial_care_seats"]
        # Whole care places left after confirmed receipts and current demand.
        pool, slot = ds.activities[change.activity].care_pool, change.slot
        expected = (
            current.capacity.care_limit[(pool, slot)]
            - current.capacity.care_reserved.get((pool, slot), 0)
            - current.capacity.care_demand.get((pool, slot), 0.0)
        )
        self.assertEqual(headroom, max(0, int(expected // 1)))

    def test_refresh_after_new_signups(self):
        """A new cycle with different availability must force a redo, not silently reuse."""
        previous = self.report
        mutated = json.loads(json.dumps(self.report))
        mutated["input"]["digest"] = "new-cycle"
        mutated["proposals_summary"]["W1"]["tier_vector"] = ["partial", "supported"]
        verdict = refresh_decision(previous, mutated)
        self.assertIn(verdict["verdict"], {"需重做比较", "先试开"})

    def test_updated_capacity_vintage_is_labelled(self):
        report = build_report(self.ds, bootstrap_draws=4, capacity_vintage="updated")
        self.assertEqual(report["input"]["capacity_vintage"], "updated")


if __name__ == "__main__":
    unittest.main()
