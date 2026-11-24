"""Whole-proposal evaluation of a set of rotations.

A proposal is compared as a *bundle* against the current plan, never as a sum of
single-shift gains.  Moving an activity removes supply from its old place;
adding one keeps the old supply.  Both change the same families' menus, so the
bundle is re-predicted from the joint menu and the capacity pass is redone.

Resource accounting covers:

* seats per offering (a migration moves them; it does not duplicate them),
* care places per pool and slot, charged once per family that needs care and is
  predicted into that pool/slot,
* confirmed receipts, which are committed before any change,
* later capacity notices, which replace the advertised limit only when the
  report explicitly asks for the updated vintage.

A family whose predicted assignments exceed a limit is re-placed onto its
remaining feasible options in proportion to the model's probabilities.  It is
counted once: the released mass is redistributed, never re-added as fresh
demand, so a delayed offer cannot inflate the proposal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .data import Dataset, Offering, RotationChange
from .model import ModelBundle, Prediction, predict_family


# --------------------------------------------------------------------------- menus
@dataclass
class Scenario:
    name: str
    kind: str  # current | rotation
    schedules: dict[tuple[str, str, str], Offering]
    added: dict[tuple[str, str, str], Offering] = field(default_factory=dict)
    moved: dict[str, Offering] = field(default_factory=dict)
    capacity_vintage: str = "proposal"

    def open_for(self, dataset: Dataset, family: str) -> list[Offering]:
        """Build one family's menu from its last observed offer pattern.

        Home-area offerings are whatever the scenario schedules at the family's
        area.  Cross-area offerings keep exactly the cross activities the family
        was last actually shown; a rotation changes those offerings in place and
        an addition at the family's own area becomes a new menu entry.  Nothing
        unobserved is silently handed to a family.
        """
        profile = dataset.families[family]
        row = dataset.latest_row(family)
        scheduled_by_activity: dict[str, list[Offering]] = defaultdict(list)
        for offering in self.schedules.values():
            scheduled_by_activity[offering.activity].append(offering)

        menu: list[Offering] = []
        seen: set[tuple[str, str, str]] = set()

        def add(offering: Offering) -> None:
            if offering.key in seen:
                return
            seen.add(offering.key)
            menu.append(offering)

        for offering in self.schedules.values():
            if offering.site == profile.area:
                add(offering)
        for activity in row.available:
            for offering in scheduled_by_activity.get(activity, []):
                if offering.site != profile.area:
                    add(offering)
        for offering in self.added.values():
            if offering.site == profile.area:
                add(offering)
        return menu


def build_scenario(
    dataset: Dataset,
    name: str,
    changes: Sequence[RotationChange],
    kind: str = "rotation",
    capacity_vintage: str = "proposal",
) -> Scenario:
    schedules = dict(dataset.baseline_schedule())
    if not changes:
        return Scenario(
            name=name, kind="current", schedules=schedules, capacity_vintage=capacity_vintage
        )
    added: dict[tuple[str, str, str], Offering] = {}
    moved: dict[str, Offering] = {}
    for change in changes:
        activity = dataset.activities[change.activity]
        origin_key = (activity.activity, activity.site, activity.slot)
        target = Offering(
            activity=activity.activity,
            site=change.target,
            slot=change.slot,
            subject=activity.subject,
            seats=activity.seats,
            care_pool=activity.care_pool,
            origin="migrated" if change.kind == "迁移" else "added",
        )
        if change.kind == "迁移":
            if origin_key not in schedules:
                raise ValueError(
                    f"cannot migrate {change.activity}: no such offering in the current plan"
                )
            schedules.pop(origin_key)
            schedules[target.key] = target
            moved[change.activity] = target
        else:
            schedules[target.key] = target
            added[target.key] = target
    return Scenario(
        name=name,
        kind=kind,
        schedules=schedules,
        added=added,
        moved=moved,
        capacity_vintage=capacity_vintage,
    )


# --------------------------------------------------------------------------- resources
def care_reserved(
    dataset: Dataset, scenario: Scenario, vintage: str | None = None
) -> dict[tuple[str, str], int]:
    """Care places already committed for the target cycle and still offered."""
    vintage = vintage or scenario.capacity_vintage
    target = dataset.target_cycle
    reserved: dict[tuple[str, str], int] = defaultdict(int)
    for receipt in dataset.confirmed:
        if receipt.cycle != target:
            continue
        activity = dataset.activities[receipt.activity]
        if activity.care_pool != receipt.care_pool or activity.slot != receipt.slot:
            raise ValueError(f"receipt {receipt.family} disagrees with activity metadata")
        if not any(
            o.activity == receipt.activity
            and o.care_pool == receipt.care_pool
            and o.slot == receipt.slot
            for o in scenario.schedules.values()
        ):
            continue
        reserved[(receipt.care_pool, receipt.slot)] += 1
    return dict(reserved)


def _seat_key(offering: Offering) -> tuple[str, str, str, str]:
    return (offering.activity, offering.site, offering.slot, "seats")


@dataclass
class CapacityReport:
    care_limit: dict[tuple[str, str], int]
    care_reserved: dict[tuple[str, str], int]
    care_demand: dict[tuple[str, str], float]
    care_headroom: dict[tuple[str, str], float]
    care_overflow: dict[tuple[str, str], float]
    seat_capacity: dict[tuple[str, str, str, str], int]
    seat_demand: dict[tuple[str, str, str, str], float]
    seat_headroom: dict[tuple[str, str, str, str], float]
    seat_overflow: dict[tuple[str, str, str, str], float]
    displaced_families: list[str]

    @property
    def feasible_without_overflow(self) -> bool:
        return not self.care_overflow and not self.seat_overflow

    def as_dict(self) -> dict:
        def care(d):
            return {f"{p}/{s}": round(v, 3) for (p, s), v in sorted(d.items())}

        def seat(d):
            return {f"{a}@{s}/{t}": round(v, 3) for (a, s, t, _), v in sorted(d.items())}

        return {
            "care_limit": care(self.care_limit),
            "care_reserved": care(self.care_reserved),
            "care_demand": care(self.care_demand),
            "care_headroom": care(self.care_headroom),
            "care_overflow": care(self.care_overflow),
            "seat_capacity": seat(self.seat_capacity),
            "seat_demand": seat(self.seat_demand),
            "seat_headroom": seat(self.seat_headroom),
            "seat_overflow": seat(self.seat_overflow),
            "displaced_families": self.displaced_families,
        }


def account_capacity(
    dataset: Dataset,
    scenario: Scenario,
    bundle: ModelBundle,
    preds: Mapping[str, Prediction],
    menus: Mapping[str, Sequence[Offering]],
) -> tuple[CapacityReport, dict[str, dict[tuple[str, str, str], float]]]:
    """Charge predicted demand against seats and care places, then re-place.

    Returns the capacity report and each family's accepted mass per offering.
    Capacity is fractional (expected values): an expected 6.08 care places is a
    real signal of pressure, and is reported rather than rounded away.
    """
    vintage = scenario.capacity_vintage
    reserved = care_reserved(dataset, scenario, vintage)
    limits = {key: dataset.care_limit(key[0], key[1], vintage) for key in dataset.limits}
    seat_capacity = {_seat_key(o): o.seats for o in scenario.schedules.values()}

    care_used: dict[tuple[str, str], float] = defaultdict(float)
    seat_used: dict[tuple[str, str, str, str], float] = defaultdict(float)
    accepted_by_family: dict[str, dict[tuple[str, str, str], float]] = {}
    displaced: list[str] = []

    for family in sorted(preds):
        profile = dataset.families[family]
        pred = preds[family]
        menu = {o.key: o for o in menus[family]}
        mass = {k: v for k, v in pred.final.items() if v > 1e-12}
        accepted: dict[tuple[str, str, str], float] = defaultdict(float)
        accepted_by_family[family] = accepted
        if not mass:
            continue
        blocked = False
        for _ in range(10):
            allowance: dict[tuple[str, str, str], float] = {}
            for key in mass:
                offering = menu.get(key)
                if offering is None:
                    allowance[key] = 0.0
                    continue
                seat_left = seat_capacity.get(_seat_key(offering), 0) - seat_used[_seat_key(offering)]
                care_left = float("inf")
                if profile.needs_care:
                    care_key = (offering.care_pool, offering.slot)
                    care_left = (
                        limits.get(care_key, 0)
                        - reserved.get(care_key, 0)
                        - care_used[care_key]
                    )
                allowance[key] = max(0.0, min(seat_left, care_left))
            leftover = 0.0
            for key, amount in list(mass.items()):
                take = min(amount, allowance.get(key, 0.0))
                if take > 0:
                    offering = menu[key]
                    accepted[key] += take
                    seat_used[_seat_key(offering)] += take
                    if profile.needs_care:
                        care_used[(offering.care_pool, offering.slot)] += take
                    mass[key] -= take
                leftover += mass[key]
            if leftover <= 1e-9:
                break
            open_keys = [k for k in mass if allowance.get(k, 0.0) > 1e-9]
            if not open_keys:
                blocked = True
                break
            blocked = True
            weights = np.array([mass[k] for k in open_keys], dtype=float)
            if weights.sum() <= 0:
                weights = np.ones(len(open_keys))
            weights /= weights.sum()
            for k in list(mass):
                if k not in open_keys:
                    mass[k] = 0.0
            for k, w in zip(open_keys, weights):
                mass[k] += leftover * w
        if blocked:
            displaced.append(family)

    care_headroom = {
        key: limits[key] - reserved.get(key, 0) - care_used.get(key, 0.0) for key in limits
    }
    seat_headroom = {
        key: cap - seat_used.get(key, 0.0) for key, cap in seat_capacity.items()
    }
    report = CapacityReport(
        care_limit=limits,
        care_reserved=dict(reserved),
        care_demand=dict(care_used),
        care_headroom=care_headroom,
        care_overflow={k: -v for k, v in care_headroom.items() if v < -1e-9},
        seat_capacity=seat_capacity,
        seat_demand=dict(seat_used),
        seat_headroom=seat_headroom,
        seat_overflow={k: -v for k, v in seat_headroom.items() if v < -1e-9},
        displaced_families=sorted(set(displaced)),
    )
    return report, accepted_by_family


# --------------------------------------------------------------------------- comparison
@dataclass
class ScenarioOutcome:
    name: str
    kind: str
    expected_participants: float
    expected_declines: float
    expected_unplaced: float
    by_offering: dict[tuple[str, str, str], float]
    care_demand: dict[tuple[str, str], float]
    continuation: float
    displacement: float
    capacity: CapacityReport
    per_family: dict[str, Prediction] = field(default_factory=dict)
    menus: dict[str, list[Offering]] = field(default_factory=dict)
    accepted: dict[str, dict[tuple[str, str, str], float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "expected_participants": self.expected_participants,
            "expected_declines": self.expected_declines,
            "expected_unplaced": self.expected_unplaced,
            "continuation": self.continuation,
            "displacement": self.displacement,
            "by_offering": {
                f"{a}@{s}/{t}": round(v, 3) for (a, s, t), v in sorted(self.by_offering.items())
            },
            "care_demand": {
                f"{p}/{s}": round(v, 3) for (p, s), v in sorted(self.care_demand.items())
            },
            "capacity": self.capacity.as_dict(),
        }


def evaluate_scenario(
    dataset: Dataset,
    scenario: Scenario,
    bundle: ModelBundle,
    reference: ScenarioOutcome | None = None,
) -> ScenarioOutcome:
    preds: dict[str, Prediction] = {}
    menus: dict[str, list[Offering]] = {}
    for family in sorted(dataset.families):
        profile = dataset.families[family]
        menu = scenario.open_for(dataset, family)
        menus[family] = menu
        preds[family] = predict_family(bundle, profile.area, profile.needs_care, menu)

    capacity, accepted = account_capacity(dataset, scenario, bundle, preds, menus)

    by_offering: dict[tuple[str, str, str], float] = defaultdict(float)
    care_demand: dict[tuple[str, str], float] = defaultdict(float)
    participants = 0.0
    for family, rows in accepted.items():
        profile = dataset.families[family]
        menu = {o.key: o for o in menus[family]}
        participants += sum(rows.values())
        for key, mass in rows.items():
            by_offering[key] += mass
            offering = menu.get(key)
            if offering is not None and profile.needs_care:
                care_demand[(offering.care_pool, offering.slot)] += mass
    declines = sum(p.p_outside for p in preds.values())
    unplaced = max(0.0, float(len(preds)) - participants - declines)

    continuation = 0.0
    displacement = 0.0
    if reference is not None:
        for key in set(by_offering) | set(reference.by_offering):
            new = by_offering.get(key, 0.0)
            old = reference.by_offering.get(key, 0.0)
            continuation += min(new, old)
            displacement += max(old - new, 0.0)
    return ScenarioOutcome(
        name=scenario.name,
        kind=scenario.kind,
        expected_participants=participants,
        expected_declines=declines,
        expected_unplaced=unplaced,
        by_offering=dict(by_offering),
        care_demand=dict(care_demand),
        continuation=continuation,
        displacement=displacement,
        capacity=capacity,
        per_family=preds,
        menus=menus,
        accepted={f: dict(v) for f, v in accepted.items()},
    )


@dataclass
class ProposalComparison:
    proposal: str
    current: ScenarioOutcome
    proposed: ScenarioOutcome
    changes: list[RotationChange]

    @property
    def net_participants(self) -> float:
        return self.proposed.expected_participants - self.current.expected_participants

    def binding_constraints(self) -> list[dict]:
        """Care pools the proposal pushes against, with before/after headroom.

        A pool already full in the current plan is not caused by the proposal, so
        it is listed separately with ``raised_by_proposal=False``.
        """
        rows = []
        keys = set(self.current.capacity.care_demand) | set(self.proposed.capacity.care_demand)
        for key in sorted(keys):
            before = self.current.capacity.care_demand.get(key, 0.0)
            after = self.proposed.capacity.care_demand.get(key, 0.0)
            limit = self.proposed.capacity.care_limit[key]
            reserved = self.proposed.capacity.care_reserved.get(key, 0)
            headroom_before = limit - reserved - before
            headroom_after = limit - reserved - after
            raised = after > before + 1e-6
            if raised or headroom_after <= 0.05:
                rows.append(
                    {
                        "pool": key[0],
                        "slot": key[1],
                        "limit": limit,
                        "reserved": reserved,
                        "demand_before": round(before, 3),
                        "demand_after": round(after, 3),
                        "headroom_before": round(headroom_before, 3),
                        "headroom_after": round(headroom_after, 3),
                        "raised_by_proposal": bool(raised),
                    }
                )
        return rows

    def rechecked_displaced(self) -> list[dict]:
        """Families pushed out of the current plan get another look, not new demand."""
        displaced = set(self.current.capacity.displaced_families)
        rows = []
        for family in sorted(displaced):
            before = sum(self.current.accepted.get(family, {}).values())
            after = sum(self.proposed.accepted.get(family, {}).values())
            pred = self.proposed.per_family.get(family)
            menu = self.proposed.menus.get(family, [])
            best = None
            if pred is not None and pred.final:
                key = max(pred.final, key=pred.final.get)
                offering = next((o for o in menu if o.key == key), None)
                if offering is not None:
                    best = {
                        "activity": offering.activity,
                        "site": offering.site,
                        "slot": offering.slot,
                        "probability": round(pred.final[key], 3),
                    }
            rows.append(
                {
                    "family": family,
                    "expected_before": round(before, 3),
                    "expected_after": round(after, 3),
                    "delta": round(after - before, 3),
                    "top_alternative": best,
                }
            )
        return rows

    def as_dict(self) -> dict:
        return {
            "proposal": self.proposal,
            "changes": [
                {"activity": c.activity, "kind": c.kind, "target": f"{c.target}/{c.slot}"}
                for c in self.changes
            ],
            "current": self.current.as_dict(),
            "proposed": self.proposed.as_dict(),
            "net": {
                "expected_participants": self.net_participants,
                "expected_declines_delta": self.proposed.expected_declines
                - self.current.expected_declines,
                "expected_unplaced": self.proposed.expected_unplaced,
                "expected_unplaced_delta": self.proposed.expected_unplaced
                - self.current.expected_unplaced,
                "continuation": self.proposed.continuation,
                "displacement": self.proposed.displacement,
            },
            "binding_constraints": self.binding_constraints(),
            "rechecked_displaced": self.rechecked_displaced(),
            "feasible_without_overflow": self.proposed.capacity.feasible_without_overflow,
        }


def evaluate_proposal(
    dataset: Dataset,
    proposal: str,
    bundle: ModelBundle,
    capacity_vintage: str = "proposal",
) -> ProposalComparison:
    changes = dataset.proposals().get(proposal, [])
    if not changes:
        raise KeyError(f"unknown proposal {proposal!r}")
    current = evaluate_scenario(dataset, build_scenario(dataset, "current", []), bundle)
    proposed = evaluate_scenario(
        dataset,
        build_scenario(dataset, proposal, changes, capacity_vintage=capacity_vintage),
        bundle,
        reference=current,
    )
    return ProposalComparison(proposal, current, proposed, changes)


def changed_families(
    dataset: Dataset, comparison: ProposalComparison, threshold: float = 0.05
) -> list[dict]:
    """Families whose predicted outcome changed, for re-checking the displaced.

    A negative delta is a family pushed out of a first choice; it is re-checked
    against the new menu rather than added to demand.
    """
    rows = []
    for family, old in comparison.current.per_family.items():
        new = comparison.proposed.per_family.get(family)
        if new is None:
            continue
        before = sum(comparison.current.accepted.get(family, {}).values())
        after = sum(comparison.proposed.accepted.get(family, {}).values())
        if after - before < -threshold:
            rows.append(
                {
                    "family": family,
                    "area": dataset.families[family].area,
                    "needs_care": dataset.families[family].needs_care,
                    "expected_before": before,
                    "expected_after": after,
                    "delta": after - before,
                }
            )
    rows.sort(key=lambda r: r["delta"])
    return rows
