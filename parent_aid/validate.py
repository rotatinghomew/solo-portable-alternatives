"""Back-testing on the held-out full rotation cycles.

``evaluation_cycles`` are complete cycles kept out of every model fit and every
support threshold below, so the numbers there are genuine out-of-sample checks.
The report compares, per held-out cycle:

* the predicted *choice* distribution (first pick) with the observed one,
* the predicted *re-placement* (改选) distribution and switch rate with the
  observed ones,

plus a family-clustered bootstrap band for the headline metrics and a naive
"pooled share" baseline so an operator can see what the model adds.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from .data import ChoiceRow, Dataset, Offering
from .model import ModelBundle, family_bootstrap, fit_bundle, predict_family


def _offering(dataset: Dataset, activity: str) -> Offering:
    a = dataset.activities[activity]
    return Offering(a.activity, a.site, a.slot, a.subject, a.seats, a.care_pool)


def _l1(a: dict, b: dict) -> float:
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def _normalise(counts: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total > 0 else {}


def _choice_and_final_scores(
    dataset: Dataset, bundle: ModelBundle, rows: Sequence[ChoiceRow]
) -> dict:
    ll = brier = 0.0
    correct = 0
    n = 0
    obs_choice: Counter[str] = Counter()
    pred_choice: Counter[str] = Counter()
    obs_final: Counter[str] = Counter()
    pred_final: Counter[str] = Counter()
    obs_switch = pred_switch = 0.0
    for row in rows:
        options = tuple(_offering(dataset, a) for a in row.available)
        if not options:
            continue
        pred = predict_family(bundle, row.area, row.needs_care, options)
        key_index = {o.activity: o.key for o in options}
        n += 1
        chosen = row.selected
        p_chosen = pred.initial.get(key_index.get(chosen), 0.0) if chosen else 0.0
        ll += math.log(p_chosen + 1e-12)
        brier += (pred.p_outside - (1.0 if chosen is None else 0.0)) ** 2
        best = pred.top()
        if chosen and best is not None and best[0] == key_index.get(chosen):
            correct += 1
        obs_choice[chosen or "NONE"] += 1
        for key, mass in pred.initial.items():
            pred_choice[key[0]] += mass
        pred_choice["NONE"] += pred.p_outside
        final = row.final
        obs_final[final or "NONE"] += 1
        for key, mass in pred.final.items():
            pred_final[key[0]] += mass
        pred_final["NONE"] += pred.p_outside
        obs_switch += 1.0 if row.switch_to is not None else 0.0
        pred_switch += pred.p_switch

    total = sum(v for k, v in obs_choice.items() if k != "NONE")
    choice_pred_total = sum(v for k, v in pred_choice.items() if k != "NONE")
    return {
        "n": n,
        "choice": {
            "log_likelihood_per_row": ll / max(1, n),
            "brier_outside": brier / max(1, n),
            "top_choice_accuracy": correct / max(1, n),
            "total_variation": 0.5 * _l1(obs_choice, pred_choice) / max(1, total),
            "observed": _normalise(obs_choice),
            "predicted": _normalise(pred_choice),
        },
        "final": {
            "total_variation": 0.5 * _l1(obs_final, pred_final) / max(1, total),
            "observed": _normalise(obs_final),
            "predicted": _normalise(pred_final),
            "mass_on_activities": choice_pred_total,
        },
        "switch": {
            "observed_rate": obs_switch / max(1, n),
            "predicted_rate": pred_switch / max(1, n),
            "n": n,
        },
    }


def _pooled_share_baseline(dataset: Dataset, train_cycles: Iterable[int]) -> dict:
    """Predict a family with the pooled selection share of each activity.

    This is the closest thing to "average attendance" the records allow.  It
    ignores which site the family lives in, so it overstates cross-site take-up
    and cannot distinguish "nobody wanted it" from "it was never shown".
    """
    train_rows = [r for r in dataset.rows_for(train_cycles) if r.decided]
    share = {}
    for activity in dataset.activities:
        exposure = sum(activity in r.available for r in train_rows)
        chosen = sum(r.selected == activity for r in train_rows)
        share[activity] = chosen / exposure if exposure else 0.0

    eval_rows = [
        r for r in dataset.rows_for(dataset.evaluation_cycles) if r.decided and r.available
    ]
    ll = 0.0
    obs: Counter[str] = Counter()
    pred: Counter[str] = Counter()
    for row in eval_rows:
        options = list(row.available)
        if not options:
            continue
        weights = np.array([share[a] for a in options], dtype=float)
        if weights.sum() <= 0:
            p = np.zeros(len(options))
        else:
            p = weights / (weights.sum() + 0.5)
        p_out = 1.0 - float(p.sum())
        chosen = row.selected
        idx = options.index(chosen) if chosen in options else None
        ll += math.log((p[idx] if idx is not None else p_out) + 1e-12)
        obs[chosen or "NONE"] += 1
        for a, pr in zip(options, p):
            pred[a] += pr
        pred["NONE"] += p_out
    n = len(eval_rows)
    total = sum(obs.values())
    return {
        "log_likelihood_per_row": ll / max(1, n),
        "total_variation": 0.5 * _l1(obs, pred) / max(1, total),
        "note": "只使用训练轮次的活动被选比例，不使用家庭所在片区；用于对照而非执行。",
    }


def backtest(dataset: Dataset, n_boot: int = 60, seed: int = 20261124) -> dict:
    train = tuple(c for c in dataset.cycles if c not in dataset.evaluation_cycles)
    per_cycle = {}
    for cycle in dataset.evaluation_cycles:
        fit_cycles = tuple(c for c in train if c < cycle) or train
        bundle = fit_bundle(dataset, fit_cycles)
        rows = [r for r in dataset.rows_for([cycle]) if r.decided and r.available]
        per_cycle[str(cycle)] = {
            "fit_cycles": list(fit_cycles),
            **_choice_and_final_scores(dataset, bundle, rows),
        }

    draws = family_bootstrap(dataset, train, n_boot=n_boot, seed=seed)
    metrics = []
    for draw in draws:
        rows = [
            r
            for r in dataset.rows_for(dataset.evaluation_cycles)
            if r.decided and r.available
        ]
        score = _choice_and_final_scores(dataset, draw, rows)
        metrics.append(
            {
                "choice_tv": score["choice"]["total_variation"],
                "choice_ll": score["choice"]["log_likelihood_per_row"],
                "final_tv": score["final"]["total_variation"],
            }
        )
    bands = {}
    for key in ("choice_tv", "choice_ll", "final_tv"):
        values = np.array([m[key] for m in metrics])
        bands[key] = {
            "mean": float(values.mean()),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }
    return {
        "train_cycles": list(train),
        "evaluation_cycles": list(dataset.evaluation_cycles),
        "per_cycle": per_cycle,
        "bootstrap": {"draws": len(draws), "held_out_metrics": bands},
        "pooled_share_baseline": _pooled_share_baseline(dataset, train),
    }
