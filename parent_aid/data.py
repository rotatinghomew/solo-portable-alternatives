from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

SLOTS = ("晚", "周末")
DECLINED = "declined"
PENDING = "pending"
SELECTED = "selected"
SWITCHED = "switched"


@dataclass(frozen=True)
class Activity:
    activity: str
    site: str
    subject: str
    slot: str
    seats: int
    care_pool: str

    @property
    def key(self) -> str:
        return self.activity


@dataclass(frozen=True)
class Offering:
    """One concrete opening of an activity at a place and time.

    A rotation of type ``增设`` creates a second offering for the same activity;
    type ``迁移`` replaces the original offering with one at the target.  The
    care pool always travels with the activity, never with the site.
    """

    activity: str
    site: str
    slot: str
    subject: str
    seats: int
    care_pool: str
    origin: str = "base"  # base | migrated | added

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.activity, self.site, self.slot)

    @property
    def label(self) -> str:
        return f"{self.activity}@{self.site}/{self.slot}"


@dataclass(frozen=True)
class Family:
    family: str
    area: str
    time_preference: str
    needs_care: bool


@dataclass(frozen=True)
class ChoiceRow:
    cycle: int
    family: str
    area: str
    time_preference: str
    needs_care: bool
    offered: tuple[str, ...]
    wait_not_invited: tuple[str, ...]
    selected: str | None
    switch_to: str | None
    declined: bool

    @property
    def available(self) -> tuple[str, ...]:
        blocked = set(self.wait_not_invited)
        return tuple(a for a in self.offered if a not in blocked)

    @property
    def status(self) -> str:
        if self.selected is not None:
            return SWITCHED if self.switch_to is not None else SELECTED
        if self.declined:
            return DECLINED
        return PENDING

    @property
    def final(self) -> str | None:
        """Final observed activity.  A switch replaces the initial choice."""
        return self.switch_to or self.selected

    @property
    def decided(self) -> bool:
        """True when the outcome is observed rather than censored on a waitlist."""
        return self.status != PENDING

    @property
    def open_to_person(self) -> tuple[str, ...]:
        return self.available


@dataclass(frozen=True)
class ConfirmedSeat:
    cycle: int
    family: str
    activity: str
    care_pool: str
    slot: str


@dataclass(frozen=True)
class RotationChange:
    proposal: str
    activity: str
    target: str
    slot: str
    kind: str  # 迁移 | 增设

    @property
    def is_migration(self) -> bool:
        return self.kind == "迁移"


@dataclass(frozen=True)
class CapacityChange:
    cycle: int
    pool: str
    slot: str
    new_seat_limit: int


@dataclass
class Dataset:
    path: Path
    activities: dict[str, Activity]
    choices: list[ChoiceRow]
    limits: dict[tuple[str, str], int]
    confirmed: list[ConfirmedSeat]
    rotation: list[RotationChange]
    evaluation_cycles: tuple[int, ...]
    unshown_scenarios: list[CapacityChange]
    text_digest: str
    families: dict[str, Family] = field(init=False)

    def __post_init__(self) -> None:
        fam: dict[str, Family] = {}
        for row in self.choices:
            prof = Family(row.family, row.area, row.time_preference, row.needs_care)
            old = fam.get(row.family)
            if old is not None and old != prof:
                raise ValueError(f"family {row.family!r} changed fixed attributes across cycles")
            fam[row.family] = prof
        self.families = fam

    # ------------------------------------------------------------------ basics
    @property
    def cycles(self) -> tuple[int, ...]:
        return tuple(sorted({row.cycle for row in self.choices}))

    @property
    def latest_cycle(self) -> int:
        return max(self.cycles)

    @property
    def target_cycle(self) -> int:
        """Cycle the rotation proposals are being prepared for."""
        candidates = [self.latest_cycle + 1]
        candidates += [c.cycle for c in self.confirmed]
        candidates += [s.cycle for s in self.unshown_scenarios]
        return max(candidates)

    def rows_for(self, cycles: Iterable[int]) -> list[ChoiceRow]:
        keep = set(cycles)
        return [row for row in self.choices if row.cycle in keep]

    def latest_row(self, family: str) -> ChoiceRow:
        rows = [r for r in self.choices if r.family == family]
        return max(rows, key=lambda r: r.cycle)

    def baseline_schedule(self) -> dict[tuple[str, str, str], Offering]:
        return {
            (a.activity, a.site, a.slot): Offering(
                activity=a.activity,
                site=a.site,
                slot=a.slot,
                subject=a.subject,
                seats=a.seats,
                care_pool=a.care_pool,
                origin="base",
            )
            for a in self.activities.values()
        }

    def care_limit(self, pool: str, slot: str, vintage: str = "proposal") -> int:
        limit = self.limits[(pool, slot)]
        if vintage == "updated":
            for change in self.unshown_scenarios:
                if change.pool == pool and change.slot == slot:
                    # Later capacity notices replace the advertised limit.
                    limit = min(limit, change.new_seat_limit)
        elif vintage != "proposal":
            raise ValueError("capacity vintage must be 'proposal' or 'updated'")
        return limit

    def proposals(self) -> dict[str, list[RotationChange]]:
        out: dict[str, list[RotationChange]] = {}
        for change in self.rotation:
            out.setdefault(change.proposal, []).append(change)
        return out

    # --------------------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Validate the input contract and internal consistency.

        Returns a list of human-readable findings that are *not* fatal but should
        be shown to the operator.  Raises ValueError on contract violations.
        """
        notes: list[str] = []
        act_ids = set(self.activities)
        for row in self.choices:
            offered, wait = set(row.offered), set(row.wait_not_invited)
            if not wait <= offered:
                raise ValueError(f"cycle {row.cycle} {row.family}: waitlist outside offered")
            if not offered <= act_ids:
                raise ValueError(f"cycle {row.cycle} {row.family}: unknown activity")
            available = offered - wait
            chosen = {row.selected, row.switch_to} - {None}
            if not chosen <= available:
                raise ValueError(f"cycle {row.cycle} {row.family}: chose unavailable activity")
            if row.declined and row.selected is not None:
                raise ValueError(f"cycle {row.cycle} {row.family}: declined with a selection")
            if row.switch_to is not None and row.selected is None:
                raise ValueError(f"cycle {row.cycle} {row.family}: switch without a selection")
        for row in self.choices:
            if row.status == PENDING and not row.wait_not_invited:
                notes.append(
                    f"第 {row.cycle} 轮 {row.family}：未选择也未拒绝但没有候补记录，按未揭晓处理。"
                )
            if row.declined and row.wait_not_invited:
                notes.append(
                    f"第 {row.cycle} 轮 {row.family}：在候补期间明确不选；"
                    "拒绝被采用，候补项仍不进入选择集。"
                )
        ids = [(r.family, r.cycle) for r in self.choices]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate family/cycle rows")
        if not set(self.evaluation_cycles) <= set(self.cycles):
            raise ValueError("evaluation cycle missing from choices")
        for conf in self.confirmed:
            if conf.activity not in self.activities:
                raise ValueError(f"confirmed receipt for unknown activity {conf.activity}")
            act = self.activities[conf.activity]
            if act.care_pool != conf.care_pool or act.slot != conf.slot:
                raise ValueError(f"confirmed receipt {conf.family} disagrees with activity metadata")
        for change in self.rotation:
            if change.activity not in self.activities:
                raise ValueError(f"rotation for unknown activity {change.activity}")
            if change.kind not in {"迁移", "增设"}:
                raise ValueError(f"unknown rotation type {change.kind!r}")
            if change.slot not in SLOTS:
                raise ValueError(f"unknown slot {change.slot!r}")
        for act in self.activities.values():
            if (act.care_pool, act.slot) not in self.limits:
                raise ValueError(f"missing care limit for {act.care_pool}/{act.slot}")
        # Be explicit about gaps that matter for extrapolation rather than silently
        # smoothing them over.
        sites = sorted({a.site for a in self.activities.values()})
        for site in sites:
            slots = {a.slot for a in self.activities.values() if a.site == site}
            if len(slots) == 1:
                notes.append(
                    f"片区 {site} 在记录中只开放过 {next(iter(slots))} 时段，"
                    "不存在“同片区换时段”的可比记录。"
                )
        return notes

    def input_digest(self) -> str:
        return self.text_digest

    # --------------------------------------------------------------- load helper
    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        acts = {
            a["activity"]: Activity(
                activity=a["activity"],
                site=a["site"],
                subject=a["subject"],
                slot=a["slot"],
                seats=int(a["seats"]),
                care_pool=a["care_pool"],
            )
            for a in payload["activities"]
        }
        choices = [
            ChoiceRow(
                cycle=int(r["cycle"]),
                family=r["family"],
                area=r["area"],
                time_preference=r["time_preference"],
                needs_care=bool(r["needs_care"]),
                offered=tuple(r["offered"]),
                wait_not_invited=tuple(r.get("wait_not_invited", ())),
                selected=r.get("selected"),
                switch_to=r.get("switch_to"),
                declined=bool(r.get("declined", False)),
            )
            for r in payload["choices"]
        ]
        limits = {(l["pool"], l["slot"]): int(l["seats"]) for l in payload["limits"]}
        confirmed = [
            ConfirmedSeat(
                cycle=int(c["cycle"]),
                family=c["family"],
                activity=c["activity"],
                care_pool=c["care_pool"],
                slot=c["slot"],
            )
            for c in payload["confirmed"]
        ]
        rotation = [
            RotationChange(
                proposal=r["proposal"],
                activity=r["activity"],
                target=r["target"],
                slot=r["slot"],
                kind=r["type"],
            )
            for r in payload["rotation"]
        ]
        scenarios = [
            CapacityChange(
                cycle=int(s["cycle"]),
                pool=s["pool"],
                slot=s["slot"],
                new_seat_limit=int(s["new_seat_limit"]),
            )
            for s in payload.get("unshown_scenarios", [])
        ]
        return cls(
            path=path,
            activities=acts,
            choices=choices,
            limits=limits,
            confirmed=confirmed,
            rotation=rotation,
            evaluation_cycles=tuple(int(c) for c in payload["evaluation_cycles"]),
            unshown_scenarios=scenarios,
            text_digest=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        )


def offered_vs_rejected(rows: Sequence[ChoiceRow]) -> dict[str, float]:
    """Show why waitlisted families must not be folded into refusals."""
    decided = [r for r in rows if r.decided]
    pending = [r for r in rows if r.status == PENDING]
    return {
        "rows": len(rows),
        "decided": len(decided),
        "pending": len(pending),
        "decline_rate_decided": sum(r.declined for r in decided) / max(1, len(decided)),
        "decline_rate_all_rows": sum(r.declined for r in rows) / max(1, len(rows)),
        "pending_share": len(pending) / max(1, len(rows)),
    }
