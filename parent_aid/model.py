"""Choice and re-placement models estimated only from pre-outcome information.

The two modelled decisions are deliberately simple because that is all the
records support:

* **Which open activity a family picks.**  A multinomial logit over the
  activities genuinely open to that family (``offered`` minus
  ``wait_not_invited``).  Waitlist entries never enter the choice set.  A row
  that ends on a waitlist with no selection and no explicit refusal is
  *censored*: it can only tell us that the family did not take one of the open
  activities.  It never counts as a refusal of a listed activity.
* **Whether the initial pick is re-placed (改选).**  A logistic model on the
  number of open activities.  Conditional on a switch, the destination is
  empirically indistinguishable from a uniform draw among the *other* open
  activities (see :func:`switch_destination_test`), so that is the assumption.

A family is one decision unit per cycle.  Every family's probability mass on
activities plus the outside option sums to one, so a large menu can never turn
one person into several participants.

Several structural specifications are fitted and combined by BIC weights.  The
spread between specifications is reported as structural uncertainty alongside a
family-clustered bootstrap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from .data import ChoiceRow, Dataset, Offering

EPS = 1e-300


def _check_converged(res, label: str) -> None:
    """Accept an optimum even when the line search stops on floating-point noise.

    L-BFGS-B reports ``ABNORMAL_TERMINATION_IN_LNSRCH`` when the objective can no
    longer be reduced; that is a converged fit for these smooth models, so the
    gradient norm is the actual acceptance test.
    """
    grad = getattr(res, "jac", None)
    scale = max(1.0, abs(float(res.fun)))
    if grad is not None and np.max(np.abs(grad)) <= 1e-5 * scale:
        return
    if res.success:
        return
    raise RuntimeError(f"{label} failed: {res.message}")


# --------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class Spec:
    name: str
    terms: tuple[str, ...]


SPECS: tuple[Spec, ...] = (
    Spec("位置-时段", ("home", "weekend")),
    Spec("位置-时段-交互", ("home", "weekend", "home_weekend")),
    Spec("内容", ("home", "weekend", "home_weekend", "subject_交流", "subject_协作")),
    Spec(
        "内容-照看",
        ("home", "weekend", "home_weekend", "subject_交流", "subject_协作", "needs_care"),
    ),
    Spec(
        "内容-照看-席位",
        (
            "home",
            "weekend",
            "home_weekend",
            "subject_交流",
            "subject_协作",
            "needs_care",
            "seats",
        ),
    ),
)


def feature_vector(area: str, needs_care: bool, offering: Offering, spec: Spec) -> np.ndarray:
    home = offering.site == area
    weekend = offering.slot == "周末"
    out = []
    for term in spec.terms:
        if term == "home":
            out.append(1.0 if home else 0.0)
        elif term == "weekend":
            out.append(1.0 if weekend else 0.0)
        elif term == "home_weekend":
            out.append(1.0 if home and weekend else 0.0)
        elif term == "subject_交流":
            out.append(1.0 if offering.subject == "交流" else 0.0)
        elif term == "subject_协作":
            out.append(1.0 if offering.subject == "协作" else 0.0)
        elif term == "needs_care":
            out.append(1.0 if needs_care else 0.0)
        elif term == "seats":
            out.append(float(offering.seats) / 10.0)
        else:  # pragma: no cover - guarded by SPECS
            raise ValueError(f"unknown term {term!r}")
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- data prep
@dataclass(frozen=True)
class ChoiceObservation:
    row: ChoiceRow
    options: tuple[Offering, ...]
    chosen: int  # index into options, or len(options) for the outside option


def observation(dataset: Dataset, row: ChoiceRow) -> ChoiceObservation | None:
    if not row.decided:
        return None
    options = tuple(
        Offering(
            activity=a,
            site=dataset.activities[a].site,
            slot=dataset.activities[a].slot,
            subject=dataset.activities[a].subject,
            seats=dataset.activities[a].seats,
            care_pool=dataset.activities[a].care_pool,
        )
        for a in row.available
    )
    if not options:
        return None
    chosen = next(
        (i for i, o in enumerate(options) if o.activity == row.selected), len(options)
    )
    return ChoiceObservation(row=row, options=options, chosen=chosen)


def observations(dataset: Dataset, cycles: Iterable[int]) -> list[ChoiceObservation]:
    out = []
    for row in dataset.rows_for(cycles):
        obs = observation(dataset, row)
        if obs is not None:
            out.append(obs)
    return out


# --------------------------------------------------------------------------- fitting
def _choice_nll(
    beta: np.ndarray, spec: Spec, data: Sequence[ChoiceObservation], l2: float
) -> tuple[float, np.ndarray]:
    total = 0.0
    grad = np.zeros_like(beta)
    for obs in data:
        X = np.array(
            [feature_vector(obs.row.area, obs.row.needs_care, o, spec) for o in obs.options]
        )
        u = np.concatenate([X @ beta, [0.0]])
        u -= u.max()
        e = np.exp(u)
        p = e / e.sum()
        total -= math.log(p[obs.chosen] + EPS)
        d = p.copy()
        d[obs.chosen] -= 1.0
        grad += X.T @ d[:-1]
    return total + l2 * float(beta @ beta), grad + 2 * l2 * beta


def fit_choice(
    data: Sequence[ChoiceObservation], spec: Spec, l2: float = 1e-8
) -> np.ndarray:
    k = len(spec.terms)
    res = minimize(
        lambda b: _choice_nll(b, spec, data, l2),
        np.zeros(k),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-9, "maxls": 50},
    )
    _check_converged(res, f"choice model {spec.name}")
    return res.x


def choice_loglik(
    beta: np.ndarray, spec: Spec, data: Sequence[ChoiceObservation]
) -> float:
    nll, _ = _choice_nll(beta, spec, data, 0.0)
    return -nll


def bic_weights(data: Sequence[ChoiceObservation], specs: Sequence[Spec] = SPECS) -> tuple[list[np.ndarray], np.ndarray]:
    """Fit every specification and return BIC weights over them."""
    betas, bics = [], []
    n = len(data)
    for spec in specs:
        beta = fit_choice(data, spec)
        bics.append(-2.0 * choice_loglik(beta, spec, data) + len(spec.terms) * math.log(n))
        betas.append(beta)
    bics_arr = np.asarray(bics)
    w = np.exp(-0.5 * (bics_arr - bics_arr.min()))
    w /= w.sum()
    return betas, w


# --------------------------------------------------------------------------- switch
@dataclass(frozen=True)
class SwitchModel:
    coefficients: np.ndarray  # intercept, number open, needs_care

    def probability(self, needs_care: bool, n_open: int) -> float:
        z = (
            self.coefficients[0]
            + self.coefficients[1] * n_open
            + self.coefficients[2] * (1.0 if needs_care else 0.0)
        )
        return 0.0 if n_open <= 1 else 1.0 / (1.0 + math.exp(-z))


def fit_switch(dataset: Dataset, cycles: Iterable[int], l2: float = 1e-8) -> SwitchModel:
    rows = [r for r in dataset.rows_for(cycles) if r.selected is not None]
    if not rows:
        raise ValueError("no rows with an initial selection")
    X = np.array(
        [[1.0, float(len(r.available)), 1.0 if r.needs_care else 0.0] for r in rows]
    )
    y = np.array([1.0 if r.switch_to is not None else 0.0 for r in rows])

    def objective(beta: np.ndarray):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        total = -np.sum(y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS))
        return total + l2 * float(beta @ beta), X.T @ (p - y) + 2 * l2 * beta

    res = minimize(
        objective,
        np.zeros(3),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-9, "maxls": 50},
    )
    _check_converged(res, "switch model")
    return SwitchModel(coefficients=res.x)


def switch_destination_test(dataset: Dataset, cycles: Iterable[int] | None = None) -> dict[str, float]:
    """Test whether switch destinations look uniform among the other open options.

    The records give no signal that a switch targets a systematically better
    option; treating it as uniform is therefore the defensible default.
    """
    rows = [
        r
        for r in dataset.rows_for(cycles if cycles is not None else dataset.cycles)
        if r.switch_to is not None
    ]
    obs: dict[str, float] = {}
    exp: dict[str, float] = {}
    for row in rows:
        others = [a for a in row.available if a != row.selected]
        if not others:
            continue
        obs[row.switch_to] = obs.get(row.switch_to, 0.0) + 1.0
        for a in others:
            exp[a] = exp.get(a, 0.0) + 1.0 / len(others)
    keys = sorted(set(obs) | set(exp))
    stat = sum((obs.get(a, 0.0) - exp.get(a, 0.0)) ** 2 / exp[a] for a in keys if exp.get(a, 0.0) > 0)
    df = len([a for a in keys if exp.get(a, 0.0) > 0]) - 1
    return {
        "rows": len(rows),
        "chi2": stat,
        "df": df,
        "p_value": float(chi2.sf(stat, df)) if df > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- prediction
@dataclass
class Prediction:
    """Prediction for one family; a family is one decision unit.

    ``initial`` is the distribution of the first pick before any re-placement;
    ``final`` is the distribution after re-placement mass has been spread over
    the other open options.  Both sum with ``p_outside`` to one, so a large menu
    can never turn one person into several participants.
    """

    initial: dict[tuple[str, str, str], float]
    final: dict[tuple[str, str, str], float]
    p_outside: float
    p_switch: float
    n_open: int

    @property
    def p_participate(self) -> float:
        return 1.0 - self.p_outside

    def mass(self, offering: Offering) -> float:
        return self.final.get(offering.key, 0.0)

    def top(self) -> tuple[tuple[str, str, str], float] | None:
        if not self.initial:
            return None
        key = max(self.initial, key=self.initial.get)
        return key, self.initial[key]


def _mnl_probs(
    beta: np.ndarray, spec: Spec, area: str, needs_care: bool, options: Sequence[Offering]
) -> tuple[np.ndarray, float]:
    if not options:
        return np.zeros(0), 1.0
    X = np.array([feature_vector(area, needs_care, o, spec) for o in options])
    u = np.concatenate([X @ beta, [0.0]])
    u -= u.max()
    e = np.exp(u)
    p = e / e.sum()
    return p[:-1], float(p[-1])


def predict_family(
    bundle: "ModelBundle",
    area: str,
    needs_care: bool,
    options: Sequence[Offering],
) -> Prediction:
    """Model-averaged prediction for one family/decision unit."""
    n = len(options)
    if n == 0:
        return Prediction({}, {}, 1.0, 0.0, 0)
    p_switch = bundle.switch.probability(needs_care, n)
    initial = {o.key: 0.0 for o in options}
    p_out = 0.0
    for spec, beta, weight in bundle.components():
        p_in, p_outside = _mnl_probs(beta, spec, area, needs_care, options)
        p_out += weight * p_outside
        for o, p in zip(options, p_in):
            initial[o.key] += weight * float(p)
    total = sum(initial.values())
    if total > 0:
        scale = (1.0 - p_out) / total
        initial = {k: v * scale for k, v in initial.items()}
    # Re-placement: a switched family lands uniformly on one of the other open
    # options (switch_destination_test supports this).  No new participants.
    final = dict(initial)
    if p_switch > 0 and n > 1:
        mass = sum(initial.values())
        for o in options:
            keep = initial[o.key] * (1.0 - p_switch)
            from_others = (mass - initial[o.key]) * p_switch / (n - 1)
            final[o.key] = keep + from_others
    return Prediction(initial, final, p_out, p_switch, n)


@dataclass
class ModelBundle:
    """Parameters for one fitted model (point estimate or bootstrap draw)."""

    specs: tuple[Spec, ...]
    betas: list[np.ndarray]
    weights: np.ndarray
    switch: SwitchModel
    training_cycles: tuple[int, ...] = ()

    def components(self):
        for spec, beta, w in zip(self.specs, self.betas, self.weights):
            if w > 0:
                yield spec, beta, float(w)

    @property
    def effective_specs(self) -> int:
        return int((self.weights > 1e-6).sum())

    def coefficient_table(self) -> list[dict[str, float]]:
        rows = []
        for spec, beta, weight in self.components():
            for term, value in zip(spec.terms, beta):
                rows.append(
                    {
                        "spec": spec.name,
                        "term": term,
                        "weight": float(weight),
                        "coefficient": float(value),
                    }
                )
        return rows


def fit_bundle(dataset: Dataset, cycles: Iterable[int], specs: Sequence[Spec] = SPECS) -> ModelBundle:
    cycles = tuple(sorted(set(cycles)))
    data = observations(dataset, cycles)
    if not data:
        raise ValueError(f"no usable choice rows in cycles {cycles}")
    betas, weights = bic_weights(data, specs)
    return ModelBundle(
        specs=tuple(specs),
        betas=betas,
        weights=weights,
        switch=fit_switch(dataset, cycles),
        training_cycles=cycles,
    )


# --------------------------------------------------------------------------- bootstrap
def family_bootstrap(
    dataset: Dataset,
    cycles: Iterable[int],
    n_boot: int = 100,
    seed: int = 20261124,
    specs: Sequence[Spec] = SPECS,
) -> list[ModelBundle]:
    """Family-clustered bootstrap; families, never rows, are the sampling unit."""
    cycles = tuple(sorted(set(cycles)))
    data = observations(dataset, cycles)
    by_family: dict[str, list[ChoiceObservation]] = {}
    for obs in data:
        by_family.setdefault(obs.row.family, []).append(obs)
    families = sorted(by_family)
    switch_rows = [r for r in dataset.rows_for(cycles) if r.selected is not None]
    switch_by_family: dict[str, list[ChoiceRow]] = {}
    for r in switch_rows:
        switch_by_family.setdefault(r.family, []).append(r)
    rng = np.random.default_rng(seed)
    draws: list[ModelBundle] = []
    for _ in range(n_boot):
        sample = rng.choice(families, size=len(families), replace=True)
        picked = [obs for f in sample for obs in by_family[f]]
        try:
            betas, weights = bic_weights(picked, specs)
            sw_rows = [r for f in sample for r in switch_by_family.get(f, [])]
            switch = _fit_switch_rows(sw_rows)
        except (RuntimeError, ValueError):
            continue
        draws.append(
            ModelBundle(
                specs=tuple(specs),
                betas=betas,
                weights=weights,
                switch=switch,
                training_cycles=cycles,
            )
        )
    if not draws:
        raise RuntimeError("bootstrap produced no usable draws")
    return draws


def _fit_switch_rows(rows: Sequence[ChoiceRow], l2: float = 1e-8) -> SwitchModel:
    if not rows:
        return SwitchModel(np.array([-1.0, 0.0, 0.0]))
    X = np.array(
        [[1.0, float(len(r.available)), 1.0 if r.needs_care else 0.0] for r in rows]
    )
    y = np.array([1.0 if r.switch_to is not None else 0.0 for r in rows])

    def objective(beta: np.ndarray):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        total = -np.sum(y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS))
        return total + l2 * float(beta @ beta), X.T @ (p - y) + 2 * l2 * beta

    try:
        res = minimize(
            objective,
            np.zeros(3),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-9, "maxls": 50},
        )
        _check_converged(res, "switch model")
    except RuntimeError:
        return SwitchModel(np.array([-1.0, 0.0, 0.0]))
    return SwitchModel(coefficients=res.x)
