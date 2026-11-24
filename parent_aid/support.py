"""Evidence support for a rotation, judged from observed exposure only.

The central question -- "may we move this content to this place and time?" -- is
answered by asking which *cells* of the (content, place, time) grid have ever
been open to anybody, how often, and for how many complete cycles.  Two cells
that never existed cannot be borrowed from: in these records each site only
ever opens in one time slot, so half of the place-by-time grid is empty and no
amount of modelling can fill it.

Exposure means "was in the family's ``offered`` list and not blocked on the
waitlist".  Selections mean "was actually chosen".  Rows still waiting for an
invitation are never used as evidence of refusal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from .data import Dataset, RotationChange

TIER_ORDER = {"supported": 0, "partial": 1, "unsupported": 2}

MIN_CELL_EXPOSURE = 30
MIN_CELL_CHOICES = 5
MIN_DIM_EXPOSURE = 60
MIN_DIM_CHOICES = 8
MIN_CYCLES_FOR_STABLE = 3
THIN_ACTIVITY_CYCLES = 3


@dataclass(frozen=True)
class CellStats:
    exposure: int
    chosen: int
    cycles: tuple[int, ...]
    care_exposure: int
    care_chosen: int

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)


@dataclass
class ChangeSupport:
    change: RotationChange
    tier: str
    target_cell: CellStats
    content_slot_elsewhere: CellStats
    site_slot_other_content: CellStats
    content_site_other_slot: CellStats
    site_slot_observed: bool
    activity_cycles: tuple[int, ...]
    activity_exposure: int
    borrow: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return self.tier == "supported"

    def as_dict(self) -> dict:
        return {
            "proposal": self.change.proposal,
            "activity": self.change.activity,
            "kind": self.change.kind,
            "target": f"{self.change.target}/{self.change.slot}",
            "tier": self.tier,
            "target_cell": _cell_dict(self.target_cell),
            "content_slot_elsewhere": _cell_dict(self.content_slot_elsewhere),
            "site_slot_other_content": _cell_dict(self.site_slot_other_content),
            "content_site_other_slot": _cell_dict(self.content_site_other_slot),
            "site_slot_observed": self.site_slot_observed,
            "activity_history": {
                "cycles": list(self.activity_cycles),
                "exposure": self.activity_exposure,
            },
            "borrow": self.borrow,
            "gaps": self.gaps,
            "warnings": self.warnings,
        }


def _cell_dict(cell: CellStats) -> dict:
    return {
        "exposure": cell.exposure,
        "chosen": cell.chosen,
        "cycles": list(cell.cycles),
        "n_cycles": cell.n_cycles,
        "care_exposure": cell.care_exposure,
        "care_chosen": cell.care_chosen,
    }


def _aggregate(
    dataset: Dataset, predicate, max_cycle: int
) -> CellStats:
    exposure = chosen = care_exposure = care_chosen = 0
    cycles: set[int] = set()
    for row in dataset.choices:
        if row.cycle > max_cycle:
            continue
        for a in row.available:
            if not predicate(dataset.activities[a]):
                continue
            exposure += 1
            cycles.add(row.cycle)
            care_exposure += 1 if row.needs_care else 0
            if row.selected == a:
                chosen += 1
                care_chosen += 1 if row.needs_care else 0
    return CellStats(exposure, chosen, tuple(sorted(cycles)), care_exposure, care_chosen)


def _evidence_grid(dataset: Dataset, max_cycle: int) -> dict[tuple[str, str, str], CellStats]:
    return {
        cell: _aggregate(
            dataset,
            lambda a, cell=cell: (a.subject, a.site, a.slot) == cell,
            max_cycle,
        )
        for cell in {
            (a.subject, a.site, a.slot) for a in dataset.activities.values()
        }
    }


def judge_change(
    dataset: Dataset,
    change: RotationChange,
    max_cycle: int | None = None,
) -> ChangeSupport:
    """Judge one proposed change without using outcome data from later cycles."""
    if max_cycle is None:
        max_cycle = dataset.latest_cycle
    activity = dataset.activities[change.activity]
    subject = activity.subject
    target_site, target_slot = change.target, change.slot

    grid = _evidence_grid(dataset, max_cycle)

    def cell(predicate) -> CellStats:
        found = [stats for key, stats in grid.items() if predicate(key)]
        exposure = sum(s.exposure for s in found)
        chosen = sum(s.chosen for s in found)
        cycles = sorted({c for s in found for c in s.cycles})
        care_exposure = sum(s.care_exposure for s in found)
        care_chosen = sum(s.care_chosen for s in found)
        return CellStats(exposure, chosen, tuple(cycles), care_exposure, care_chosen)

    target = grid.get((subject, target_site, target_slot), CellStats(0, 0, (), 0, 0))
    content_slot = cell(
        lambda k: k[0] == subject and k[2] == target_slot and k[1] != target_site
    )
    site_slot = cell(lambda k: k[1] == target_site and k[2] == target_slot and k[0] != subject)
    content_site = cell(
        lambda k: k[0] == subject and k[1] == target_site and k[2] != target_slot
    )
    site_slot_observed = (target.exposure + site_slot.exposure) > 0

    activity_cycles: set[int] = set()
    activity_exposure = 0
    for row in dataset.choices:
        if row.cycle > max_cycle:
            continue
        if change.activity in row.available:
            activity_exposure += 1
            activity_cycles.add(row.cycle)

    borrow: list[str] = []
    gaps: list[str] = []
    warnings: list[str] = []

    if target.exposure >= MIN_CELL_EXPOSURE:
        borrow.append(
            f"目标格（{subject}/{target_site}/{target_slot}）本身有 {target.exposure} 次可选、"
            f"{target.chosen} 次被选，覆盖 {target.n_cycles} 个轮次；可借用该格的同内容同地点同时段证据。"
        )
    elif target.exposure > 0:
        borrow.append(
            f"目标格仅有 {target.exposure} 次可选记录（{target.n_cycles} 个轮次），只能作为弱证据。"
        )
    if content_slot.exposure >= MIN_DIM_EXPOSURE:
        borrow.append(
            f"可借用“同内容不同片区、同时段”的证据：{subject} 在 {target_slot} 其他片区可选 "
            f"{content_slot.exposure} 次、被选 {content_slot.chosen} 次。"
        )
    if site_slot.exposure >= MIN_DIM_EXPOSURE:
        borrow.append(
            f"可借用“同片区同时段、不同内容”的证据：{target_site}/{target_slot} 的其他内容可选 "
            f"{site_slot.exposure} 次、被选 {site_slot.chosen} 次。"
        )
    if content_site.exposure >= MIN_DIM_EXPOSURE:
        borrow.append(
            f"可借用“同内容同片区、不同时段”的证据：{subject} 在 {target_site} 其他时段可选 "
            f"{content_site.exposure} 次、被选 {content_site.chosen} 次。"
        )

    if not site_slot_observed:
        gaps.append(
            f"片区 {target_site} 从未在 {target_slot} 时段开放过任何内容，"
            "该“地点×时段”组合没有任何可选记录，无法从现有证据外推。"
        )
    if target.exposure == 0 and content_slot.exposure < MIN_DIM_EXPOSURE and site_slot.exposure < MIN_DIM_EXPOSURE:
        gaps.append(
            f"“{subject}”在 {target_site}/{target_slot} 既无同格记录，也缺少可拼合的两个单维证据。"
        )
    if target.exposure and target.care_chosen < MIN_CELL_CHOICES:
        gaps.append(
            f"目标格中需要照看的家庭仅被选 {target.care_chosen} 次，"
            "不足以支持“需要照看的家庭会接受此处此时”的单独结论。"
        )

    if len(activity_cycles) < THIN_ACTIVITY_CYCLES:
        warnings.append(
            f"活动 {change.activity} 自身只在 {len(activity_cycles)} 个轮次出现（可选 {activity_exposure} 次），"
            "其自身历史偏薄；结论主要靠同内容同格的其他活动支撑。"
        )
    if change.kind == "迁移":
        warnings.append(
            f"迁移会撤走 {change.activity} 在原片区/原时段的名额，"
            "原参与者必须重新选择，不能把新地点的人数直接当作净增量。"
        )
    else:
        warnings.append(
            f"增设在保留原名额的同时新增一处 {change.activity}，"
            "需与同内容原供应一起看是否重复计算同一批家庭。"
        )

    if not site_slot_observed:
        tier = "unsupported"
    elif target.exposure >= MIN_CELL_EXPOSURE and target.chosen >= MIN_CELL_CHOICES:
        tier = "supported" if target.n_cycles >= MIN_CYCLES_FOR_STABLE else "partial"
    elif target.exposure >= MIN_CELL_EXPOSURE:
        tier = "partial"
    elif target.exposure > 0 or (
        content_slot.exposure >= MIN_DIM_EXPOSURE and site_slot.exposure >= MIN_DIM_EXPOSURE
    ):
        tier = "partial"
    else:
        tier = "unsupported"

    return ChangeSupport(
        change=change,
        tier=tier,
        target_cell=target,
        content_slot_elsewhere=content_slot,
        site_slot_other_content=site_slot,
        content_site_other_slot=content_site,
        site_slot_observed=site_slot_observed,
        activity_cycles=tuple(sorted(activity_cycles)),
        activity_exposure=activity_exposure,
        borrow=borrow,
        gaps=gaps,
        warnings=warnings,
    )


def judge_proposal(
    dataset: Dataset, proposal: str, max_cycle: int | None = None
) -> list[ChangeSupport]:
    changes = dataset.proposals().get(proposal, [])
    if not changes:
        raise KeyError(f"unknown proposal {proposal!r}")
    return [judge_change(dataset, change, max_cycle) for change in changes]


def evidence_gaps(dataset: Dataset, max_cycle: int | None = None) -> dict[str, list]:
    """Enumerate where the data cannot support extrapolation at all."""
    if max_cycle is None:
        max_cycle = dataset.latest_cycle
    grid = _evidence_grid(dataset, max_cycle)
    sites = sorted({a.site for a in dataset.activities.values()})
    slots = sorted({a.slot for a in dataset.activities.values()})
    subjects = sorted({a.subject for a in dataset.activities.values()})

    empty_cells = []
    for site in sites:
        for slot in slots:
            exposure = sum(
                stats.exposure
                for (content, cell_site, cell_slot), stats in grid.items()
                if cell_site == site and cell_slot == slot
            )
            if exposure == 0:
                empty_cells.append(f"{site}/{slot}")

    thin_cells = []
    for (subject, site, slot), stats in sorted(grid.items()):
        if stats.exposure < MIN_CELL_EXPOSURE:
            thin_cells.append(
                {
                    "cell": f"{subject}/{site}/{slot}",
                    "exposure": stats.exposure,
                    "chosen": stats.chosen,
                    "cycles": list(stats.cycles),
                }
            )

    sparse_populations = []
    cells: dict[tuple[str, str, bool], int] = defaultdict(int)
    for family in dataset.families.values():
        cells[(family.area, family.time_preference, family.needs_care)] += 1
    for (area, pref, needs_care), n in sorted(cells.items()):
        if n < 5:
            sparse_populations.append(
                {
                    "area": area,
                    "time_preference": pref,
                    "needs_care": needs_care,
                    "families": n,
                }
            )

    return {
        "max_cycle": max_cycle,
        "sites": sites,
        "slots": slots,
        "subjects": subjects,
        "empty_site_slot_cells": empty_cells,
        "thin_content_cells": thin_cells,
        "sparse_populations": sparse_populations,
        "notes": [
            "每个片区在记录中只出现过一个时段，因此“同一片区换时段”完全没有同片区可比记录。",
            "时段偏好（time_preference）与照看需求在家庭间固定，但未表现出可估计的选择差异；"
            "本分析不据此做人群分层外推。",
            "需要照看的家庭与不需要照看的家庭在小格中的样本极少（每格 2–5 户），"
            "无法支持“某类家庭在某格一定接受”的结论。",
        ],
    }
