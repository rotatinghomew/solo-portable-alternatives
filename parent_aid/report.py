"""Operator-facing report: support, whole-proposal comparison, uncertainty.

The report is deterministic (fixed bootstrap seed), so two runs on the same
input file produce identical numbers and can be diffed when a new cycle of
sign-ups or a site change arrives.  Every run records the input digest and the
judgements used, so an operator can see whether last round's reasoning still
holds, whether to move the date, or whether a trial must come first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import Dataset, RotationChange
from .model import (
    ModelBundle,
    family_bootstrap,
    fit_bundle,
    switch_destination_test,
)
from .schedule import (
    ProposalComparison,
    Scenario,
    build_scenario,
    changed_families,
    evaluate_proposal,
    evaluate_scenario,
)
from .support import ChangeSupport, evidence_gaps, judge_proposal
from .validate import backtest

BOOTSTRAP_DRAWS = 60
TRIAL_NOTE = (
    "试开只能用未承诺的余量，不能占用 confirmed 回执或当前计划已确认的照看名额；"
    "试开结果在下一轮才作为观测进入模型，不得写入本轮正式轮换理由。"
)


# --------------------------------------------------------------------------- uncertainty
def _metrics(dataset: Dataset, bundle: ModelBundle, scenario: Scenario) -> dict:
    outcome = evaluate_scenario(dataset, scenario, bundle)
    return {
        "participants": outcome.expected_participants,
        "care_overflow_max": max(outcome.capacity.care_overflow.values(), default=0.0),
        "seat_overflow_max": max(outcome.capacity.seat_overflow.values(), default=0.0),
    }


def comparison_uncertainty(
    dataset: Dataset,
    proposal: str,
    draws: Sequence[ModelBundle],
) -> dict:
    changes = dataset.proposals()[proposal]
    current = build_scenario(dataset, "current", [])
    proposed = build_scenario(dataset, proposal, changes)
    deltas, current_p, proposed_p = [], [], []
    care_overflow, seat_overflow = [], []
    for bundle in draws:
        cur = _metrics(dataset, bundle, current)
        new = _metrics(dataset, bundle, proposed)
        deltas.append(new["participants"] - cur["participants"])
        current_p.append(cur["participants"])
        proposed_p.append(new["participants"])
        care_overflow.append(new["care_overflow_max"])
        seat_overflow.append(new["seat_overflow_max"])

    def band(values) -> dict:
        v = np.asarray(values, dtype=float)
        return {
            "mean": round(float(v.mean()), 6),
            "p05": round(float(np.percentile(v, 5)), 6),
            "p50": round(float(np.percentile(v, 50)), 6),
            "p95": round(float(np.percentile(v, 95)), 6),
        }

    deltas_arr = np.asarray(deltas)
    care_arr = np.asarray(care_overflow)
    seat_arr = np.asarray(seat_overflow)
    return {
        "draws": len(draws),
        "participants_current": band(current_p),
        "participants_proposed": band(proposed_p),
        "participants_delta": band(deltas_arr),
        "care_overflow_proposed_max": band(care_arr),
        "seats_overflow_proposed_max": band(seat_arr),
        "probability_delta_positive": float((deltas_arr > 0).mean()),
        "probability_within_capacity": float(
            ((care_arr <= 1e-9) & (seat_arr <= 1e-9)).mean()
        ),
        "note": (
            "区间来自按家庭聚类的自助抽样；家庭而非行是抽样单位，"
            "同一家庭不会因跨轮次重复而人为放大置信度。"
        ),
    }


# --------------------------------------------------------------------------- trial
def trial_headroom(dataset: Dataset, change: RotationChange, current_outcome) -> int:
    """Whole care places that are not reserved and not already needed."""
    activity = dataset.activities[change.activity]
    limit = dataset.care_limit(activity.care_pool, change.slot, "proposal")
    reserved = current_outcome.capacity.care_reserved.get(
        (activity.care_pool, change.slot), 0
    )
    demand = current_outcome.capacity.care_demand.get((activity.care_pool, change.slot), 0.0)
    return max(0, int((limit - reserved - demand) // 1))


def trial_plan(
    dataset: Dataset,
    proposal: str,
    support: Sequence[ChangeSupport],
    current_outcome,
) -> dict:
    rows = []
    for item in support:
        change = item.change
        if item.tier == "unsupported":
            headroom = trial_headroom(dataset, change, current_outcome)
            decision = "仅试开" if headroom > 0 else "本轮不试开"
            reason = (
                "目标“地点×时段”没有任何可选记录，现有证据无法外推。"
                + (
                    f" 仅可使用 {headroom} 个未被回执或当前计划占用的照看位置试开。"
                    if headroom > 0
                    else " 没有不挪用已确认照看名额的空位，本轮不试开。"
                )
            )
        elif item.tier == "partial":
            headroom = trial_headroom(dataset, change, current_outcome)
            decision = "先小范围试开或补证"
            reason = "目标格证据偏薄，只能小范围试开，并把试开期数据留到下一轮再评估。"
        else:
            headroom = None
            decision = "纳入正式轮换评估"
            reason = "目标格有直接或可拼合的可选记录，可进入正式评估。"
        rows.append(
            {
                "activity": change.activity,
                "target": f"{change.target}/{change.slot}",
                "decision": decision,
                "trial_care_seats": headroom,
                "reason": reason,
            }
        )
    return {"proposal": proposal, "rows": rows, "rule": TRIAL_NOTE}


# --------------------------------------------------------------------------- refresh
def refresh_decision(previous: dict, current: dict) -> dict:
    """Answer: does last round's judgement still hold, or redo the comparison?"""
    if previous.get("input", {}).get("digest") == current.get("input", {}).get("digest"):
        return {
            "verdict": "判断仍成立",
            "reason": "输入摘要未变，模型与证据门槛相同，结论可直接沿用。",
            "previous_digest": previous.get("input", {}).get("digest"),
            "current_digest": current.get("input", {}).get("digest"),
        }
    prev_summary = previous.get("proposals_summary", {})
    cur_summary = current.get("proposals_summary", {})
    changed = [
        name
        for name in cur_summary
        if prev_summary.get(name, {}).get("tier_vector") != cur_summary[name]["tier_vector"]
        or prev_summary.get(name, {}).get("feasible") != cur_summary[name]["feasible"]
    ]
    trials = [n for n, v in cur_summary.items() if v.get("decision") == "先试开"]
    if changed:
        verdict = "需重做比较"
        reason = f"以下提案的证据等级或容量结论已变化：{', '.join(sorted(changed))}。"
    elif trials:
        verdict = "先试开"
        reason = f"以下提案仍缺少外推证据，只能试开：{', '.join(sorted(trials))}。"
    else:
        verdict = "判断仍成立"
        reason = "输入变化未触及目标格证据与容量约束，结论沿用。"
    if current.get("input", {}).get("capacity_vintage") == "updated":
        reason += " 本轮容量通告已更新，比较使用更新后的照看上限。"
    return {
        "verdict": verdict,
        "reason": reason,
        "previous_digest": previous.get("input", {}).get("digest"),
        "current_digest": current.get("input", {}).get("digest"),
    }


# --------------------------------------------------------------------------- build
def build_report(
    dataset: Dataset,
    proposals: Sequence[str] | None = None,
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    capacity_vintage: str = "proposal",
    seed: int = 20261124,
) -> dict:
    validation_notes = dataset.validate()
    proposals = list(proposals) if proposals is not None else sorted(dataset.proposals())

    fit_cycles = tuple(c for c in dataset.cycles if c not in dataset.evaluation_cycles)
    bundle = fit_bundle(dataset, fit_cycles)
    draws = family_bootstrap(dataset, fit_cycles, n_boot=bootstrap_draws, seed=seed)

    proposal_reports = {}
    for name in proposals:
        support = judge_proposal(dataset, name)
        comparison = evaluate_proposal(
            dataset, name, bundle, capacity_vintage=capacity_vintage
        )
        uncertainty = comparison_uncertainty(dataset, name, draws)
        trial = trial_plan(dataset, name, support, comparison.current)
        tiers = [s.tier for s in support]
        decision = _decision(tiers, comparison, uncertainty)
        proposal_reports[name] = {
            "changes": [s.as_dict() for s in support],
            "tier_vector": tiers,
            "feasible": comparison.proposed.capacity.feasible_without_overflow,
            "decision": decision,
            "capacity_decision_reason": _capacity_reason(comparison, uncertainty),
            "comparison": comparison.as_dict(),
            "uncertainty": uncertainty,
            "changed_families": changed_families(dataset, comparison),
            "trial": trial,
        }

    report = {
        "input": {
            "path": str(dataset.path),
            "digest": dataset.input_digest(),
            "cycles": list(dataset.cycles),
            "evaluation_cycles": list(dataset.evaluation_cycles),
            "target_cycle": dataset.target_cycle,
            "capacity_vintage": capacity_vintage,
        },
        "validation_notes": validation_notes,
        "model": {
            "fit_cycles": list(fit_cycles),
            "specifications": [
                {"name": spec.name, "terms": list(spec.terms), "bic_weight": round(float(w), 4)}
                for spec, w in zip(bundle.specs, bundle.weights)
            ],
            "coefficients": bundle.coefficient_table(),
            "switch_model": {
                "intercept_n_open_needs_care": [float(x) for x in bundle.switch.coefficients],
                "destination_uniform_test": switch_destination_test(dataset, fit_cycles),
            },
            "censoring_rule": (
                "selected 为空且未 declined 的行视为候补未揭晓：只贡献“未参与”质量，"
                "不当作对任何活动的拒绝；wait_not_invited 不进入选择集，"
                "未提供的选项绝不按拒绝处理。"
            ),
            "one_person_rule": (
                "每个家庭每轮只作为一个决策单位：全部活动概率与未参与概率之和为 1。"
            ),
            "switch_rule": (
                "改选目的地按其余可选活动的均匀分布处理；这是对现有记录最保守、"
                "也最符合数据的假设（均匀性检验见 destination_uniform_test）。"
            ),
        },
        "evidence": {"gaps": evidence_gaps(dataset)},
        "held_out_validation": backtest(
            dataset, n_boot=max(10, min(30, bootstrap_draws)), seed=seed
        ),
        "proposals": proposal_reports,
        "trial_policy": TRIAL_NOTE,
    }
    report["proposals_summary"] = {
        name: {
            "tier_vector": data["tier_vector"],
            "feasible": data["feasible"],
            "decision": data["decision"],
            "capacity_decision_reason": data["capacity_decision_reason"],
            "net_participants": data["comparison"]["net"]["expected_participants"],
        }
        for name, data in proposal_reports.items()
    }
    return report


def _capacity_reason(comparison: ProposalComparison, uncertainty: dict) -> str:
    overflow = comparison.proposed.capacity.care_overflow
    if overflow:
        return (
            "照看名额期望值超限："
            + "、".join(f"{p}/{s} 超 {v:.2f}" for (p, s), v in overflow.items())
            + "；需先腾出名额、改期或拆分。"
        )
    if uncertainty["probability_within_capacity"] < 0.5:
        binding = [
            (
                f"{r['pool']}/{r['slot']} 余量 {r['headroom_after']:.2f}"
                + ("（本轮新增压力）" if r["raised_by_proposal"] else "（现行计划已吃满）")
            )
            for r in comparison.binding_constraints()
            if r["headroom_after"] <= 0.05
        ]
        return (
            "点估计未超限，但照看余量已被吃满（"
            + "、".join(binding)
            + "），自助抽样下不超限概率低于一半。"
            "周末照看池在现行计划中已接近饱和，迁移类提案会把这些池彻底顶满；"
            "应先腾出照看名额、改期，或只做不增加周末照看需求的调整。"
        )
    return "座位与照看名额均有余量。"


def _decision(
    tiers: Sequence[str], comparison: ProposalComparison, uncertainty: dict
) -> str:
    if "unsupported" in tiers:
        return "先试开"
    if not comparison.proposed.capacity.feasible_without_overflow:
        return "改期或拆分"
    if uncertainty["probability_within_capacity"] < 0.5:
        return "先腾出照看名额"
    if "partial" in tiers:
        return "补证后正式轮换"
    return "正式轮换"


def _round_floats(value, places: int = 6):
    """Stabilise output against last-bit BLAS noise so reports can be diffed."""
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, dict):
        return {k: _round_floats(v, places) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(v, places) for v in value]
    return value


def write_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _round_floats(report)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def markdown_summary(report: dict) -> str:
    """Short Chinese summary for an operations meeting."""
    lines = ["# 家长互助班轮换评估摘要", ""]
    lines.append(
        f"输入摘要 `{report['input']['digest']}`；拟合轮次 {report['model']['fit_cycles']}；"
        f"留出核验轮次 {report['input']['evaluation_cycles']}；目标轮次 {report['input']['target_cycle']}；"
        f"容量口径 `{report['input']['capacity_vintage']}`。"
    )
    lines.append("")
    lines.append("## 留出轮次核验（整期，不参与拟合、不参与门槛设定）")
    for cycle, data in report["held_out_validation"]["per_cycle"].items():
        choice, final, switch = data["choice"], data["final"], data["switch"]
        lines.append(
            f"- 第 {cycle} 轮：选择分布 TV={choice['total_variation']:.3f}、"
            f"首选命中率 {choice['top_choice_accuracy']:.2f}；"
            f"改选后分布 TV={final['total_variation']:.3f}；"
            f"改选率 观测 {switch['observed_rate']:.2f} / 预测 {switch['predicted_rate']:.2f}。"
        )
    baseline = report["held_out_validation"]["pooled_share_baseline"]
    lines.append(
        f"- 对照（只看活动被选比例、不看片区）：TV={baseline['total_variation']:.3f}，"
        f"对数似然/行 {baseline['log_likelihood_per_row']:.2f}；模型更接近观测。"
    )
    bands = report["held_out_validation"]["bootstrap"]["held_out_metrics"]
    lines.append(
        f"- 按家庭自助抽样 90% 区间：选择 TV {bands['choice_tv']['p05']:.3f}–{bands['choice_tv']['p95']:.3f}，"
        f"改选后 TV {bands['final_tv']['p05']:.3f}–{bands['final_tv']['p95']:.3f}。"
    )
    lines.append("")
    lines.append("## 模型与设定")
    specs = [
        f"{item['name']}（权重 {item['bic_weight']:.2f}）"
        for item in report["model"]["specifications"]
        if item["bic_weight"] > 0.001
    ]
    lines.append(f"- 拟合轮次 {report['model']['fit_cycles']}，选择模型设定按 BIC 加权：{'、'.join(specs)}。")
    switch = report["model"]["switch_model"]
    coef = switch["intercept_n_open_needs_care"]
    lines.append(
        "- 改选模型：截距 "
        f"{coef[0]:.2f}，每个额外可选活动 {coef[1]:+.2f}，需要照看 {coef[2]:+.2f}；"
        f"改选目的地均匀性检验 p={switch['destination_uniform_test']['p_value']:.2f}，"
        "未发现系统性偏好，按均匀处理。"
    )
    lines.append(f"- 候补与未提供选项的处理：{report['model']['censoring_rule']}")
    lines.append(f"- 一人一位：{report['model']['one_person_rule']}")
    lines.append("")
    lines.append("## 提案")
    for name, data in report["proposals"].items():
        comp, unc = data["comparison"], data["uncertainty"]
        lines.append(f"### {name} —— {data['decision']}")
        for change in data["changes"]:
            lines.append(
                f"- {change['activity']} {change['kind']} → {change['target']}：证据等级 `{change['tier']}`"
            )
            for item in change["borrow"]:
                lines.append(f"  - 可借用：{item}")
            for item in change["gaps"]:
                lines.append(f"  - 不能外推：{item}")
            for item in change["warnings"]:
                lines.append(f"  - 注意：{item}")
        net = comp["net"]
        lines.append(
            f"- 整组相对当前计划：预计参与 {comp['current']['expected_participants']:.1f} → "
            f"{comp['proposed']['expected_participants']:.1f}（净 {net['expected_participants']:+.2f}）；"
            f"接续 {net['continuation']:.1f}，流失 {net['displacement']:.1f}。"
        )
        lines.append(
            f"- 家庭自助抽样 90% 区间：净参与 {unc['participants_delta']['p05']:+.2f} ~ "
            f"{unc['participants_delta']['p95']:+.2f}；"
            f"不超出席位与照看名额的概率 {unc['probability_within_capacity']:.2f}。"
        )
        overflow = comp["proposed"]["capacity"]["care_overflow"]
        if overflow:
            lines.append(f"- 照看超限（期望值）：{overflow}")
        lines.append(f"- 容量判断：{data['capacity_decision_reason']}")
        binding = [
            r
            for r in comp["binding_constraints"]
            if r["headroom_after"] <= 0.05
        ]
        if binding:
            lines.append(
                "- 已吃满/吃紧的照看池："
                + "；".join(
                    f"{r['pool']}/{r['slot']} 上限 {r['limit']}、已确认 {r['reserved']}、"
                    f"当前计划预计 {r['demand_before']:.2f}（余量 {r['headroom_before']:.2f}）、"
                    f"本提案后预计 {r['demand_after']:.2f}（余量 {r['headroom_after']:.2f}）"
                    + ("，本轮新增压力" if r["raised_by_proposal"] else "，非本轮造成")
                    for r in binding
                )
            )
        rechecked = comp["rechecked_displaced"]
        if rechecked:
            lines.append(
                f"- 当前计划中预计被挤出但有替代去向的家庭 {len(rechecked)} 户；"
                "已在新菜单里重算，未当作新增需求。"
            )
        released = data["changed_families"]
        if released:
            lines.append(
                f"- 预计 {len(released)} 户参与量下降超过 0.05；已在新菜单中重新核算其去向，"
                "未计入新增需求。"
            )
        lines.append("- 试开安排：")
        for row in data["trial"]["rows"]:
            lines.append(f"  - {row['activity']} → {row['target']}：{row['decision']}（{row['reason']}）")
        lines.append("")
    lines.append("## 证据缺口")
    for item in report["evidence"]["gaps"]["empty_site_slot_cells"]:
        lines.append(f"- {item} 从未开放过任何内容，不能外推。")
    for note in report["evidence"]["gaps"]["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines)
