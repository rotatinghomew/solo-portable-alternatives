"""Regression tests for the analysis contract.

These pin the business rules, not just the numbers:

* waitlisted and un-offered activities never enter a menu;
* a pending (censored) row is not a refusal;
* one family contributes at most one participant;
* migration moves supply, addition duplicates it;
* care is charged per pool/slot once, with confirmed receipts reserved;
* a proposal is compared as a bundle;
* evidence-poor cells cannot be promoted into a formal rotation.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parent_aid import model, schedule, support  # noqa: E402
from parent_aid.data import Dataset, Offering  # noqa: E402


def load():
    return Dataset.load(ROOT / "data" / "records.json")


class ChoiceSetTest(unittest.TestCase):
    def setUp(self):
        self.ds = load()

    def test_waitlisted_never_in_choice_set(self):
        for row in self.ds.choices:
            self.assertTrue(set(row.wait_not_invited) <= set(row.offered))
            self.assertFalse(set(row.wait_not_invited) & set(row.available))

    def test_pending_rows_are_censored_not_refusals(self):
        pending = [r for r in self.ds.choices if not r.decided]
        self.assertTrue(pending)
        for row in pending:
            self.assertIsNone(row.selected)
            self.assertFalse(row.declined)
            # A pending row still counts as "one non-participant", never as a
            # refusal of a listed option.
            self.assertIsNone(model.observation(self.ds, row))

    def test_unoffered_option_never_predicted(self):
        bundle = model.fit_bundle(self.ds, [1, 2, 3, 4, 5])
        for row in self.ds.rows_for([6, 7]):
            options = tuple(
                Offering(a, self.ds.activities[a].site, self.ds.activities[a].slot,
                         self.ds.activities[a].subject, self.ds.activities[a].seats,
                         self.ds.activities[a].care_pool)
                for a in row.available
            )
            if not options:
                continue
            pred = model.predict_family(bundle, row.area, row.needs_care, options)
            self.assertEqual(set(pred.initial), {o.key for o in options})
            self.assertAlmostEqual(sum(pred.initial.values()) + pred.p_outside, 1.0, places=9)
            self.assertAlmostEqual(sum(pred.final.values()) + pred.p_outside, 1.0, places=9)

    def test_switch_keeps_one_person(self):
        bundle = model.fit_bundle(self.ds, [1, 2, 3, 4, 5])
        for family in list(self.ds.families)[:10]:
            profile = self.ds.families[family]
            row = self.ds.latest_row(family)
            options = tuple(
                Offering(a, self.ds.activities[a].site, self.ds.activities[a].slot,
                         self.ds.activities[a].subject, self.ds.activities[a].seats,
                         self.ds.activities[a].care_pool)
                for a in row.available
            )
            pred = model.predict_family(bundle, profile.area, profile.needs_care, options)
            self.assertLessEqual(pred.p_participate, 1.0 + 1e-12)

    def test_large_menu_does_not_create_participants(self):
        bundle = model.fit_bundle(self.ds, [1, 2, 3, 4, 5])
        big = tuple(
            Offering(a.activity, a.site, a.slot, a.subject, a.seats, a.care_pool)
            for a in self.ds.activities.values()
        )
        pred = model.predict_family(bundle, "站0", False, big)
        self.assertAlmostEqual(sum(pred.final.values()) + pred.p_outside, 1.0, places=9)
        self.assertLessEqual(sum(pred.final.values()), 1.0 + 1e-12)


class SupportTest(unittest.TestCase):
    def setUp(self):
        self.ds = load()

    def test_empty_site_slot_cannot_be_supported(self):
        gaps = support.evidence_gaps(self.ds)
        self.assertIn("站0/周末", gaps["empty_site_slot_cells"])
        self.assertIn("站1/晚", gaps["empty_site_slot_cells"])

    def test_unobserved_target_is_unsupported_and_trial_only(self):
        from parent_aid.data import RotationChange

        change = RotationChange("X", "B0", "站0", "周末", "迁移")
        judged = support.judge_change(self.ds, change)
        self.assertEqual(judged.tier, "unsupported")
        self.assertTrue(any("从未在" in gap for gap in judged.gaps))

    def test_observed_target_is_supported(self):
        judged = support.judge_proposal(self.ds, "W1")
        self.assertTrue(all(j.tier == "supported" for j in judged))
        self.assertTrue(any("可借用" in item for j in judged for item in j.borrow))


class ScheduleTest(unittest.TestCase):
    def setUp(self):
        self.ds = load()
        self.bundle = model.fit_bundle(self.ds, [1, 2, 3, 4, 5])

    def test_migration_moves_supply(self):
        current = schedule.build_scenario(self.ds, "current", [])
        w1 = schedule.build_scenario(self.ds, "W1", self.ds.proposals()["W1"])
        self.assertIn(("B0", "站0", "晚"), current.schedules)
        self.assertNotIn(("B0", "站0", "晚"), w1.schedules)
        self.assertIn(("B0", "站1", "周末"), w1.schedules)
        self.assertEqual(
            sum(o.seats for o in current.schedules.values()),
            sum(o.seats for o in w1.schedules.values()),
        )

    def test_addition_keeps_supply(self):
        w2 = schedule.build_scenario(self.ds, "W2", self.ds.proposals()["W2"])
        self.assertIn(("B11", "站3", "周末"), w2.schedules)
        self.assertIn(("B11", "站0", "晚"), w2.schedules)
        self.assertIn(("B3", "站3", "周末"), w2.schedules)
        self.assertIn(("B3", "站1", "周末"), w2.schedules)

    def test_care_pool_travels_with_activity(self):
        w1 = schedule.build_scenario(self.ds, "W1", self.ds.proposals()["W1"])
        moved = w1.schedules[("B0", "站1", "周末")]
        self.assertEqual(moved.care_pool, "托0")
        # The site's own pool is different; the activity keeps its original one.
        self.assertNotEqual(moved.care_pool, "托1")

    def test_confirmed_receipts_reserved(self):
        current = schedule.evaluate_scenario(
            self.ds, schedule.build_scenario(self.ds, "current", []), self.bundle
        )
        self.assertEqual(current.capacity.care_reserved.get(("托0", "周末")), 1)
        self.assertEqual(current.capacity.care_reserved.get(("托0", "晚")), 1)

    def test_bundle_not_sum_of_singles(self):
        comparison = schedule.evaluate_proposal(self.ds, "W1", self.bundle)
        bundle_net = comparison.net_participants
        # Compare each change alone through an ad-hoc proposal.
        singles = []
        for change in self.ds.proposals()["W1"]:
            scenario = schedule.build_scenario(self.ds, f"single-{change.activity}", [change])
            current = schedule.evaluate_scenario(
                self.ds, schedule.build_scenario(self.ds, "current", []), self.bundle
            )
            outcome = schedule.evaluate_scenario(self.ds, scenario, self.bundle, reference=current)
            singles.append(outcome.expected_participants - current.expected_participants)
        self.assertNotAlmostEqual(bundle_net, sum(singles), places=3)

    def test_unplaced_demand_is_reported(self):
        comparison = schedule.evaluate_proposal(self.ds, "W1", self.bundle)
        self.assertGreaterEqual(comparison.proposed.expected_unplaced, 0.0)
        total = (
            comparison.proposed.expected_participants
            + comparison.proposed.expected_declines
            + comparison.proposed.expected_unplaced
        )
        self.assertAlmostEqual(total, len(self.ds.families), places=6)

    def test_weekend_care_bottleneck_blocks_migration(self):
        """B0 carries 托0; moving it to a weekend slot collides with the pool
        already saturated by the current weekend programme."""
        comparison = schedule.evaluate_proposal(self.ds, "W1", self.bundle)
        binding = {r["pool"] + "/" + r["slot"]: r for r in comparison.binding_constraints()}
        self.assertIn("托0/周末", binding)
        self.assertEqual(binding["托0/周末"]["reserved"], 1)
        self.assertLessEqual(binding["托0/周末"]["headroom_after"], 0.0)
        self.assertGreater(comparison.proposed.expected_unplaced, 0.0)

    def test_gain_does_not_override_hard_capacity(self):
        comparison = schedule.evaluate_proposal(self.ds, "W2", self.bundle)
        self.assertGreater(comparison.net_participants, 0.0)
        self.assertFalse(comparison.proposed.capacity.feasible_without_overflow)

    def test_capacity_vintage_changes_limits(self):
        proposal = schedule.evaluate_proposal(self.ds, "W2", self.bundle, capacity_vintage="proposal")
        updated = schedule.evaluate_proposal(self.ds, "W2", self.bundle, capacity_vintage="updated")
        self.assertEqual(proposal.proposed.capacity.care_limit[("托0", "周末")], 6)
        self.assertEqual(updated.proposed.capacity.care_limit[("托0", "周末")], 4)


class ReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from parent_aid.report import build_report

        cls.ds = load()
        cls.report = build_report(cls.ds, bootstrap_draws=8)

    def test_report_is_deterministic(self):
        from parent_aid.report import build_report

        again = build_report(Dataset.load(ROOT / "data" / "records.json"), bootstrap_draws=8)
        self.assertEqual(json.dumps(self.report, sort_keys=True), json.dumps(again, sort_keys=True))

    def test_held_out_cycles_not_in_fit(self):
        self.assertEqual(self.report["model"]["fit_cycles"], [1, 2, 3, 4, 5])
        self.assertEqual(self.report["held_out_validation"]["evaluation_cycles"], [6, 7])

    def test_switch_uniformity_test_reported(self):
        test = self.report["model"]["switch_model"]["destination_uniform_test"]
        self.assertGreater(test["rows"], 0)
        self.assertGreater(test["p_value"], 0.01)

    def test_refresh_holds_when_input_unchanged(self):
        from parent_aid.report import refresh_decision

        verdict = refresh_decision(self.report, self.report)
        self.assertEqual(verdict["verdict"], "判断仍成立")

    def test_refresh_detects_material_change(self):
        from parent_aid.report import refresh_decision

        mutated = json.loads(json.dumps(self.report))
        mutated["input"]["digest"] = "different"
        for name in mutated["proposals_summary"]:
            mutated["proposals_summary"][name]["feasible"] = not mutated["proposals_summary"][name]["feasible"]
        verdict = refresh_decision(self.report, mutated)
        self.assertEqual(verdict["verdict"], "需重做比较")


if __name__ == "__main__":
    unittest.main()
